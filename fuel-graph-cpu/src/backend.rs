//! `GraphBackend` implementation for the CPU (gemm-backed fast path).

use fuel_core_types::{DType, Layout, Shape};
use fuel_graph_executor::{GraphBackend, UnaryOp, BinaryOp};
use fuel_reference_backend::exec::AnyRefTensor;
use fuel_reference_backend::{ops, RefTensor};

use crate::fast_matmul;

/// CPU backend: uses `gemm` for matmul, reference ops for everything else.
pub struct CpuBackend;

impl GraphBackend for CpuBackend {
    type Storage = AnyRefTensor;

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        let n = shape.elem_count();
        Ok(match dtype {
            DType::F32 => AnyRefTensor::F32(RefTensor::from_vec(vec![0.0_f32; n], shape.clone())),
            DType::F64 => AnyRefTensor::F64(RefTensor::from_vec(vec![0.0_f64; n], shape.clone())),
            DType::BF16 => AnyRefTensor::BF16(RefTensor::from_vec(vec![half::bf16::ZERO; n], shape.clone())),
            DType::F16 => AnyRefTensor::F16(RefTensor::from_vec(vec![half::f16::ZERO; n], shape.clone())),
            DType::U32 => AnyRefTensor::U32(RefTensor::from_vec(vec![0_u32; n], shape.clone())),
            _ => fuel_core_types::bail!("CpuBackend: unsupported dtype {dtype:?}"),
        })
    }

    fn upload(&self, buf: &fuel_core_types::HostBuffer, shape: &Shape) -> fuel_core_types::Result<Self::Storage> {
        use fuel_core_types::HostBuffer;
        Ok(match buf {
            HostBuffer::F32(v) => AnyRefTensor::F32(RefTensor::from_vec(v.clone(), shape.clone())),
            HostBuffer::F64(v) => AnyRefTensor::F64(RefTensor::from_vec(v.clone(), shape.clone())),
            HostBuffer::BF16(v) => AnyRefTensor::BF16(RefTensor::from_vec(v.clone(), shape.clone())),
            HostBuffer::F16(v) => AnyRefTensor::F16(RefTensor::from_vec(v.clone(), shape.clone())),
            HostBuffer::U32(v) => AnyRefTensor::U32(RefTensor::from_vec(v.clone(), shape.clone())),
            _ => fuel_core_types::bail!("CpuBackend: unsupported dtype"),
        })
    }

    fn download(&self, storage: &Self::Storage) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        use fuel_core_types::HostBuffer;
        Ok(match storage {
            AnyRefTensor::F32(t) => HostBuffer::F32(t.as_slice().to_vec()),
            AnyRefTensor::F64(t) => HostBuffer::F64(t.as_slice().to_vec()),
            AnyRefTensor::BF16(t) => HostBuffer::BF16(t.as_slice().to_vec()),
            AnyRefTensor::F16(t) => HostBuffer::F16(t.as_slice().to_vec()),
            AnyRefTensor::U32(t) => HostBuffer::U32(t.as_slice().to_vec()),
        })
    }

    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        // CPU storage is Arc-backed. Clone the data but relabel with the
        // layout's shape so downstream ops see the correct rank.
        let target_shape = layout.shape().clone();
        Ok(match storage {
            AnyRefTensor::F32(t) => AnyRefTensor::F32(RefTensor::from_arc(t.as_arc().clone(), target_shape)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64(RefTensor::from_arc(t.as_arc().clone(), target_shape)),
            AnyRefTensor::BF16(t) => AnyRefTensor::BF16(RefTensor::from_arc(t.as_arc().clone(), target_shape)),
            AnyRefTensor::F16(t) => AnyRefTensor::F16(RefTensor::from_arc(t.as_arc().clone(), target_shape)),
            AnyRefTensor::U32(t) => AnyRefTensor::U32(RefTensor::from_arc(t.as_arc().clone(), target_shape)),
        })
    }

    fn copy_strided_src(
        &self,
        src: &Self::Storage,
        dst: &mut Self::Storage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> fuel_core_types::Result<()> {
        // Delegate to reference backend's strided copy
        macro_rules! do_copy {
            ($src_t:ident, $dst_t:ident, $src_ref:expr, $dst_ref:expr) => {{
                let src_data = $src_ref.as_slice();
                let dst_data = $dst_ref.as_slice();
                let shape = src_layout.shape();
                let strides = src_layout.stride();
                let offset = src_layout.start_offset();
                let mut out = dst_data.to_vec();
                let src_shape = shape.dims();
                let n = shape.elem_count();
                // Simple strided iteration
                let mut src_idx = vec![0usize; src_shape.len()];
                for i in 0..n {
                    let mut flat = offset;
                    for d in 0..src_shape.len() {
                        flat += src_idx[d] * strides[d];
                    }
                    out[dst_offset + i] = src_data[flat];
                    // Increment multi-index
                    for d in (0..src_shape.len()).rev() {
                        src_idx[d] += 1;
                        if src_idx[d] < src_shape[d] { break; }
                        src_idx[d] = 0;
                    }
                }
                *$dst_ref = RefTensor::from_vec(out, $dst_ref.shape().clone());
            }};
        }
        match (src, dst) {
            (AnyRefTensor::F32(s), AnyRefTensor::F32(d)) => { do_copy!(s, d, s, d); }
            (AnyRefTensor::F64(s), AnyRefTensor::F64(d)) => { do_copy!(s, d, s, d); }
            (AnyRefTensor::U32(s), AnyRefTensor::U32(d)) => { do_copy!(s, d, s, d); }
            _ => fuel_core_types::bail!("copy_strided: dtype mismatch"),
        }
        Ok(())
    }

    fn storage_dtype(&self, storage: &Self::Storage) -> DType {
        storage.dtype()
    }

    // -- compute --

    fn matmul(
        &self,
        a: &Self::Storage, b: &Self::Storage,
        _bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        // Stride-aware executor may pass lazy-view inputs where the
        // RefTensor's baked-in shape differs from the layout. The CPU
        // matmul (`fast_matmul` / `ops::matmul`) reads the raw storage
        // shape, so we materialize any strided views first.
        let a_mat;
        let b_mat;
        let a = if storage_needs_materialize(a, la) {
            a_mat = materialize_view(a, la);
            &a_mat
        } else { a };
        let b = if storage_needs_materialize(b, lb) {
            b_mat = materialize_view(b, lb);
            &b_mat
        } else { b };

        // GQA support: the LLaMA cached path passes unexpanded K/V
        // [batch, n_kv_heads, ...] against Q [batch, n_heads, ...] and
        // relies on the backend to infer `n_rep = n_heads / n_kv_heads`
        // from the batch-dim mismatch. The Vulkan matmul does this
        // natively; the CPU path has no such shortcut so we expand B
        // here by tiling along the mismatched batch dim.
        let b_expanded;
        let b = if let Some(b_tiled) = expand_b_for_gqa(&storage_shape(a), &storage_shape(b), b) {
            b_expanded = b_tiled;
            &b_expanded
        } else { b };
        Ok(match (a, b) {
            (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) =>
                AnyRefTensor::F32(fast_matmul::matmul_f32(a, b)),
            (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) =>
                AnyRefTensor::F64(fast_matmul::matmul_f64(a, b)),
            (AnyRefTensor::BF16(a), AnyRefTensor::BF16(b)) =>
                AnyRefTensor::BF16(ops::matmul(a, b)),
            (AnyRefTensor::F16(a), AnyRefTensor::F16(b)) =>
                AnyRefTensor::F16(ops::matmul(a, b)),
            // Mixed-precision: activations stay f32 while weights live
            // as bf16 on device. For the CPU reference we upcast B to
            // f32 and run the f32 path — the accuracy of this matches
            // what the Vulkan backend's bf16-unpack-to-f32 kernels
            // produce, so parity tests across backends line up.
            (AnyRefTensor::F32(a), AnyRefTensor::BF16(b)) => {
                let b_data: Vec<f32> = b.as_slice().iter().map(|x| x.to_f32()).collect();
                let b_f32 = fuel_reference_backend::RefTensor::from_vec(b_data, b.shape().clone());
                AnyRefTensor::F32(fast_matmul::matmul_f32(a, &b_f32))
            }
            (a, b) => fuel_core_types::bail!("matmul: dtype mismatch {:?} vs {:?}", a.dtype(), b.dtype()),
        })
    }

    fn unary(&self, op: UnaryOp, a: &Self::Storage, _layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        macro_rules! dispatch {
            ($func:path) => {
                Ok(match a {
                    AnyRefTensor::F32(t) => AnyRefTensor::F32($func(t)),
                    AnyRefTensor::F64(t) => AnyRefTensor::F64($func(t)),
                    AnyRefTensor::BF16(t) => AnyRefTensor::BF16($func(t)),
                    AnyRefTensor::F16(t) => AnyRefTensor::F16($func(t)),
                    _ => fuel_core_types::bail!("unary: unsupported dtype"),
                })
            };
        }
        match op {
            UnaryOp::Neg => dispatch!(ops::neg),
            UnaryOp::Sqr => dispatch!(ops::sqr),
            UnaryOp::Sqrt => dispatch!(ops::sqrt),
            UnaryOp::Exp => dispatch!(ops::exp),
            UnaryOp::Log => dispatch!(ops::log),
            UnaryOp::Sin => dispatch!(ops::sin),
            UnaryOp::Cos => dispatch!(ops::cos),
            UnaryOp::Tanh => dispatch!(ops::tanh),
            UnaryOp::Sigmoid => dispatch!(ops::sigmoid),
            UnaryOp::Silu => dispatch!(ops::silu),
            UnaryOp::Gelu => dispatch!(ops::gelu),
            UnaryOp::Relu => dispatch!(ops::relu),
            UnaryOp::Step => dispatch!(ops::step),
        }
    }

    fn binary(&self, op: BinaryOp, a: &Self::Storage, b: &Self::Storage, la: &Layout, lb: &Layout) -> fuel_core_types::Result<Self::Storage> {
        // Materialize any strided view (lazy permute, lazy broadcast
        // with stride=0) into a contiguous tensor matching the layout's
        // shape. Reference ops require matching shapes.
        let a_mat;
        let b_mat;
        let a = if storage_needs_materialize(a, la) {
            a_mat = materialize_view(a, la);
            &a_mat
        } else { a };
        let b = if storage_needs_materialize(b, lb) {
            b_mat = materialize_view(b, lb);
            &b_mat
        } else { b };

        macro_rules! dispatch {
            ($func:path) => {
                Ok(match (a, b) {
                    (AnyRefTensor::F32(a), AnyRefTensor::F32(b)) => AnyRefTensor::F32($func(a, b)),
                    (AnyRefTensor::F64(a), AnyRefTensor::F64(b)) => AnyRefTensor::F64($func(a, b)),
                    (AnyRefTensor::BF16(a), AnyRefTensor::BF16(b)) => AnyRefTensor::BF16($func(a, b)),
                    (AnyRefTensor::F16(a), AnyRefTensor::F16(b)) => AnyRefTensor::F16($func(a, b)),
                    _ => fuel_core_types::bail!("binary: dtype mismatch"),
                })
            };
        }
        match op {
            BinaryOp::Add => dispatch!(ops::add),
            BinaryOp::Sub => dispatch!(ops::sub),
            BinaryOp::Mul => dispatch!(ops::mul),
            BinaryOp::Div => dispatch!(ops::div),
            BinaryOp::Maximum => dispatch!(ops::maximum),
            BinaryOp::Minimum => dispatch!(ops::minimum),
        }
    }

    fn affine(&self, a: &Self::Storage, _layout: &Layout, mul: f64, add: f64) -> fuel_core_types::Result<Self::Storage> {
        // affine: y = x * mul + add
        Ok(match a {
            AnyRefTensor::F32(t) => {
                let data: Vec<f32> = t.as_slice().iter().map(|&x| (x as f64 * mul + add) as f32).collect();
                AnyRefTensor::F32(RefTensor::from_vec(data, t.shape().clone()))
            }
            AnyRefTensor::F64(t) => {
                let data: Vec<f64> = t.as_slice().iter().map(|&x| x * mul + add).collect();
                AnyRefTensor::F64(RefTensor::from_vec(data, t.shape().clone()))
            }
            _ => fuel_core_types::bail!("affine: unsupported dtype"),
        })
    }

    fn powf(&self, a: &Self::Storage, _layout: &Layout, exp: f64) -> fuel_core_types::Result<Self::Storage> {
        Ok(match a {
            AnyRefTensor::F32(t) => {
                let data: Vec<f32> = t.as_slice().iter().map(|&x| (x as f64).powf(exp) as f32).collect();
                AnyRefTensor::F32(RefTensor::from_vec(data, t.shape().clone()))
            }
            AnyRefTensor::F64(t) => {
                let data: Vec<f64> = t.as_slice().iter().map(|&x| x.powf(exp)).collect();
                AnyRefTensor::F64(RefTensor::from_vec(data, t.shape().clone()))
            }
            _ => fuel_core_types::bail!("powf: unsupported dtype"),
        })
    }

    fn cast(&self, a: &Self::Storage, _layout: &Layout, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        // Same-dtype cast is a clone (common when a graph has a
        // defensive `cast(T)` on an already-T tensor).
        if a.dtype() == dtype {
            return Ok(a.clone());
        }
        // Delegate to the reference backend's cast dispatch
        use fuel_reference_backend::exec::AnyRefTensor as A;
        Ok(match (a, dtype) {
            (A::F32(t), DType::F64) => A::F64(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| x as f64).collect(), t.shape().clone())),
            (A::F64(t), DType::F32) => A::F32(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| x as f32).collect(), t.shape().clone())),
            (A::F32(t), DType::BF16) => A::BF16(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| half::bf16::from_f32(x)).collect(), t.shape().clone())),
            (A::F32(t), DType::F16) => A::F16(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| half::f16::from_f32(x)).collect(), t.shape().clone())),
            (A::BF16(t), DType::F32) => A::F32(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| x.to_f32()).collect(), t.shape().clone())),
            (A::F16(t), DType::F32) => A::F32(RefTensor::from_vec(
                t.as_slice().iter().map(|&x| x.to_f32()).collect(), t.shape().clone())),
            _ => fuel_core_types::bail!("cast: unsupported {:?} → {dtype:?}", a.dtype()),
        })
    }

    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, _layout: &Layout,
        dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage> {
        // For single-dim reductions, delegate to the reference backend
        macro_rules! reduce_dispatch {
            ($func:path, $dim:expr) => {
                Ok(match a {
                    AnyRefTensor::F32(t) => AnyRefTensor::F32($func(t, $dim)),
                    AnyRefTensor::F64(t) => AnyRefTensor::F64($func(t, $dim)),
                    _ => fuel_core_types::bail!("reduce: unsupported dtype"),
                })
            };
        }
        if dims.len() == 1 {
            let d = dims[0];
            match op {
                fuel_core_types::op::ReduceOp::Sum => reduce_dispatch!(ops::sum_dim, d),
                fuel_core_types::op::ReduceOp::Max => reduce_dispatch!(ops::max_dim, d),
                fuel_core_types::op::ReduceOp::Min => reduce_dispatch!(ops::min_dim, d),
                _ => fuel_core_types::bail!("reduce: ArgMin/ArgMax not yet supported"),
            }
        } else {
            // Full reduction (all dims)
            match op {
                fuel_core_types::op::ReduceOp::Sum => Ok(match a {
                    AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::sum_all(t)),
                    AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::sum_all(t)),
                    _ => fuel_core_types::bail!("reduce: unsupported dtype"),
                }),
                fuel_core_types::op::ReduceOp::Max => Ok(match a {
                    AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::max_all(t)),
                    AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::max_all(t)),
                    _ => fuel_core_types::bail!("reduce: unsupported dtype"),
                }),
                fuel_core_types::op::ReduceOp::Min => Ok(match a {
                    AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::min_all(t)),
                    AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::min_all(t)),
                    _ => fuel_core_types::bail!("reduce: unsupported dtype"),
                }),
                _ => fuel_core_types::bail!("reduce: ArgMin/ArgMax not yet supported"),
            }
        }
    }

    fn softmax_last_dim(&self, a: &Self::Storage, _layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        Ok(match a {
            AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::softmax_last_dim(t)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::softmax_last_dim(t)),
            AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::softmax_last_dim(t)),
            AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::softmax_last_dim(t)),
            _ => fuel_core_types::bail!("softmax: unsupported dtype"),
        })
    }

    fn rms_norm_last_dim(&self, a: &Self::Storage, _layout: &Layout, eps: f64)
        -> fuel_core_types::Result<Self::Storage>
    {
        Ok(match a {
            AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::rms_norm_last_dim(t, eps)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::rms_norm_last_dim(t, eps)),
            AnyRefTensor::BF16(t) => AnyRefTensor::BF16(ops::rms_norm_last_dim(t, eps)),
            AnyRefTensor::F16(t) => AnyRefTensor::F16(ops::rms_norm_last_dim(t, eps)),
            _ => fuel_core_types::bail!("rms_norm: unsupported dtype"),
        })
    }

    fn rms_norm_last_dim_backward(
        &self,
        x: &Self::Storage,
        upstream: &Self::Storage,
        _xl: &Layout,
        _ul: &Layout,
        eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        Ok(match (x, upstream) {
            (AnyRefTensor::F32(x), AnyRefTensor::F32(g)) => {
                AnyRefTensor::F32(ops::rms_norm_last_dim_backward(x, g, eps))
            }
            (AnyRefTensor::F64(x), AnyRefTensor::F64(g)) => {
                AnyRefTensor::F64(ops::rms_norm_last_dim_backward(x, g, eps))
            }
            (AnyRefTensor::BF16(x), AnyRefTensor::BF16(g)) => {
                AnyRefTensor::BF16(ops::rms_norm_last_dim_backward(x, g, eps))
            }
            (AnyRefTensor::F16(x), AnyRefTensor::F16(g)) => {
                AnyRefTensor::F16(ops::rms_norm_last_dim_backward(x, g, eps))
            }
            _ => fuel_core_types::bail!("rms_norm_last_dim_backward: dtype mismatch"),
        })
    }

    fn rope(
        &self,
        x: &Self::Storage,
        cos: &Self::Storage,
        sin: &Self::Storage,
        xl: &Layout,
        _cl: &Layout,
        _sl: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        // Stride-aware executor path (`get_gt` on x) may pass a lazy
        // permute view where the underlying RefTensor's baked-in shape
        // differs from the layout's logical shape. Materialize first
        // so `ops::rope` sees a contiguous tensor with the right shape.
        let x_mat;
        let x = if storage_needs_materialize(x, xl) {
            x_mat = materialize_view(x, xl);
            &x_mat
        } else { x };
        Ok(match (x, cos, sin) {
            (AnyRefTensor::F32(x), AnyRefTensor::F32(c), AnyRefTensor::F32(s)) => {
                AnyRefTensor::F32(ops::rope(x, c, s))
            }
            (AnyRefTensor::F64(x), AnyRefTensor::F64(c), AnyRefTensor::F64(s)) => {
                AnyRefTensor::F64(ops::rope(x, c, s))
            }
            (AnyRefTensor::BF16(x), AnyRefTensor::BF16(c), AnyRefTensor::BF16(s)) => {
                AnyRefTensor::BF16(ops::rope(x, c, s))
            }
            (AnyRefTensor::F16(x), AnyRefTensor::F16(c), AnyRefTensor::F16(s)) => {
                AnyRefTensor::F16(ops::rope(x, c, s))
            }
            _ => fuel_core_types::bail!("rope: dtype mismatch"),
        })
    }

    fn add_assign_scaled(
        &self,
        dst: &mut Self::Storage,
        src: &Self::Storage,
        scale: f32,
    ) -> fuel_core_types::Result<()> {
        // Rebuild `dst` by zipping with `src`. `RefTensor` is Arc-
        // backed; there's no real in-place mutation on CPU because
        // our storage is immutable, but we can still avoid the
        // graph-building overhead of a full add-sub-mul pipeline.
        match (&*dst, src) {
            (AnyRefTensor::F32(dt), AnyRefTensor::F32(st)) => {
                let s = scale;
                let new_data: Vec<f32> = dt.as_slice().iter().zip(st.as_slice().iter())
                    .map(|(d, s_v)| d + s_v * s)
                    .collect();
                *dst = AnyRefTensor::F32(RefTensor::from_vec(new_data, dt.shape().clone()));
            }
            (AnyRefTensor::F64(dt), AnyRefTensor::F64(st)) => {
                let s = scale as f64;
                let new_data: Vec<f64> = dt.as_slice().iter().zip(st.as_slice().iter())
                    .map(|(d, s_v)| d + s_v * s)
                    .collect();
                *dst = AnyRefTensor::F64(RefTensor::from_vec(new_data, dt.shape().clone()));
            }
            _ => fuel_core_types::bail!("add_assign_scaled: dtype mismatch or unsupported dtype"),
        }
        Ok(())
    }

    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let ids_u32 = match ids {
            AnyRefTensor::U32(t) => t,
            _ => fuel_core_types::bail!("index_select: ids must be U32"),
        };
        Ok(match src {
            AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::index_select_tensor(t, dim, ids_u32)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::index_select_tensor(t, dim, ids_u32)),
            _ => fuel_core_types::bail!("index_select: unsupported dtype"),
        })
    }

    fn gather(
        &self, src: &Self::Storage, ids: &Self::Storage,
        _src_l: &Layout, _ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        let ids_u32 = match ids {
            AnyRefTensor::U32(t) => t,
            _ => fuel_core_types::bail!("gather: ids must be U32"),
        };
        Ok(match src {
            AnyRefTensor::F32(t) => AnyRefTensor::F32(ops::gather(t, dim, ids_u32)),
            AnyRefTensor::F64(t) => AnyRefTensor::F64(ops::gather(t, dim, ids_u32)),
            _ => fuel_core_types::bail!("gather: unsupported dtype"),
        })
    }
}

/// GQA expansion: if A's batch dims are a multiple of B's (same rank,
/// same trailing-two dims), tile B along the mismatched batch dim(s)
/// to match A. Returns `None` if no expansion is needed.
///
/// Handles the common case in the cached decode path:
///   A = [batch, n_heads, m, k]         (attention scores / Q)
///   B = [batch, n_kv_heads, k, n]      (K^T / V, n_rep = n_heads / n_kv_heads)
/// We tile B's dim[1] from n_kv_heads → n_heads by repeating each
/// kv-head n_rep times.
fn expand_b_for_gqa(
    a_shape: &Shape,
    b_shape: &Shape,
    b: &AnyRefTensor,
) -> Option<AnyRefTensor> {
    let a_dims = a_shape.dims();
    let b_dims = b_shape.dims();
    if a_dims.len() != b_dims.len() || a_dims.len() < 3 {
        return None;
    }
    let rank = a_dims.len();
    // Trailing two dims (matrix dims) must match.
    if a_dims[rank - 2] == 0 || b_dims[rank - 2] == 0 {
        return None;
    }
    // Find a single batch dim where a > b and a % b == 0. Other batch
    // dims must match exactly.
    let mut mismatch_dim = None;
    for i in 0..(rank - 2) {
        if a_dims[i] == b_dims[i] {
            continue;
        }
        if b_dims[i] == 0 || a_dims[i] % b_dims[i] != 0 || mismatch_dim.is_some() {
            return None;
        }
        mismatch_dim = Some(i);
    }
    let dim = mismatch_dim?;
    let n_rep = a_dims[dim] / b_dims[dim];
    if n_rep <= 1 {
        return None;
    }

    // Tile B's `dim` by n_rep. Inner (dim+1..) size and outer (0..dim)
    // size give us the block structure.
    let inner: usize = b_dims[dim + 1..].iter().product::<usize>().max(1);
    let outer: usize = b_dims[..dim].iter().product::<usize>().max(1);
    let per_outer = b_dims[dim] * inner;

    let mut new_dims: Vec<usize> = b_dims.to_vec();
    new_dims[dim] = a_dims[dim];
    let new_shape = Shape::from_dims(&new_dims);

    macro_rules! tile {
        ($data:expr, $variant:ident) => {{
            let src = $data.as_slice();
            let new_len = outer * a_dims[dim] * inner;
            let mut out = Vec::with_capacity(new_len);
            for o in 0..outer {
                for kv in 0..b_dims[dim] {
                    let src_off = o * per_outer + kv * inner;
                    for _ in 0..n_rep {
                        out.extend_from_slice(&src[src_off..src_off + inner]);
                    }
                }
            }
            AnyRefTensor::$variant(RefTensor::from_vec(out, new_shape))
        }};
    }

    Some(match b {
        AnyRefTensor::F32(t) => tile!(t, F32),
        AnyRefTensor::F64(t) => tile!(t, F64),
        AnyRefTensor::BF16(t) => tile!(t, BF16),
        AnyRefTensor::F16(t) => tile!(t, F16),
        AnyRefTensor::U32(t) => tile!(t, U32),
    })
}

/// Returns the shape of the underlying RefTensor.
fn storage_shape(storage: &AnyRefTensor) -> Shape {
    match storage {
        AnyRefTensor::F32(t) => t.shape().clone(),
        AnyRefTensor::F64(t) => t.shape().clone(),
        AnyRefTensor::BF16(t) => t.shape().clone(),
        AnyRefTensor::F16(t) => t.shape().clone(),
        AnyRefTensor::U32(t) => t.shape().clone(),
    }
}

/// True if the storage's baked-in shape doesn't match the layout's
/// shape (lazy view) OR the layout is non-contiguous (stride != row-major).
fn storage_needs_materialize(storage: &AnyRefTensor, layout: &Layout) -> bool {
    storage_shape(storage).dims() != layout.shape().dims() || !layout.is_contiguous()
}

/// Materialize a strided view (lazy permute, lazy broadcast, or any
/// storage whose layout differs from its baked-in shape) into a
/// contiguous `AnyRefTensor` matching `layout.shape()`.
fn materialize_view(storage: &AnyRefTensor, layout: &Layout) -> AnyRefTensor {
    let out_shape = layout.shape().clone();
    let n = out_shape.elem_count();
    let strides = layout.stride();
    let offset = layout.start_offset();
    let dims = out_shape.dims();

    macro_rules! mat {
        ($src:expr, $variant:ident) => {{
            let src = $src.as_slice();
            let mut out = Vec::with_capacity(n);
            let mut idx = vec![0usize; dims.len()];
            for _ in 0..n {
                let mut flat = offset;
                for d in 0..dims.len() {
                    flat += idx[d] * strides[d];
                }
                out.push(src[flat]);
                for d in (0..dims.len()).rev() {
                    idx[d] += 1;
                    if idx[d] < dims[d] { break; }
                    idx[d] = 0;
                }
            }
            AnyRefTensor::$variant(RefTensor::from_vec(out, out_shape))
        }};
    }

    match storage {
        AnyRefTensor::F32(t) => mat!(t, F32),
        AnyRefTensor::F64(t) => mat!(t, F64),
        AnyRefTensor::BF16(t) => mat!(t, BF16),
        AnyRefTensor::F16(t) => mat!(t, F16),
        AnyRefTensor::U32(t) => mat!(t, U32),
    }
}

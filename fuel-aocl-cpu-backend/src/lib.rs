//! # fuel-aocl-cpu-backend
//!
//! AMD AOCL-BLAS-backed CPU executor for the fuel lazy-graph layer.
//!
//! This is the first per-vendor CPU backend (Phase 7b spike). It
//! mirrors `fuel-graph-cpu`'s shape but routes the matmul fast path
//! through `aocl_blas::gemm` instead of the cross-vendor `gemm` crate.
//! All other ops delegate to a contained [`CpuBackend`] so the spike
//! crate stays small.
//!
//! On Zen-class AMD CPUs `aocl_blas::gemm` calls into AOCL-BLAS (BLIS),
//! which exploits per-microarch tuning that the portable `gemm` crate
//! can't match. The Phase 6b judge profiles both at startup and the
//! dispatch table picks per `(op, dtype, size_class)`.
//!
//! # Availability gate
//!
//! [`AoclBackend::try_new`] runs a 2×2 sgemm at construction time; if
//! `libaocl_blas` doesn't load on this machine the constructor returns
//! `Err` and the rest of Fuel never sees an AOCL backend. Backends in
//! Fuel's design own their own availability check — there is no
//! HardwareQuery layer gating them.
//!
//! # Storage type
//!
//! AOCL is a CPU backend, so its `Storage` is `AnyRefTensor` —
//! identical to `fuel-graph-cpu`. That means an `AoclBackend` and a
//! `CpuBackend` can in principle share storage on the same physical
//! CPU. The Phase 6b dispatch table picks among them by `BackendId`,
//! not by `DeviceLocation`.

pub mod binding_table;
mod dll_path;
pub mod probe;

pub use binding_table::register_aocl_cpu_kernels;

use fuel_core_types::{DType, Layout, Result, Shape};
use fuel_graph_cpu::CpuBackend;
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use fuel_reference_backend::exec::AnyRefTensor;
use fuel_reference_backend::RefTensor;

/// Probe `libaocl_blas` with a 2×2 sgemm. Returns `Ok` on a successful
/// call, `Err` if the library can't be loaded (or any deeper failure
/// surfaces). Public so callers that just want a "is AOCL available?"
/// signal can use it without constructing the backend.
///
/// On Windows, this best-effort extends `PATH` with the standard
/// AOCL BLIS install directory if it isn't already there — see
/// [`dll_path`] for the discovery order. The AMD installer doesn't
/// add the BLIS dir to system PATH, so without this, every Windows
/// run would need a manual `set PATH=...` before invocation.
pub fn probe_aocl_loadable() -> Result<()> {
    dll_path::ensure_loadable();
    use aocl_types::Trans;
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 0.0, 0.0, 1.0];
    let mut c = [0.0_f32; 4];
    aocl_blas::gemm(
        Trans::No, Trans::No,
        2, 2, 2,
        1.0_f32,
        &a, &b,
        0.0_f32,
        &mut c,
    ).map_err(|e| fuel_core_types::Error::Msg(
        format!("AOCL probe gemm failed: {e}")
    ))?;
    if c != [1.0, 2.0, 3.0, 4.0] {
        return Err(fuel_core_types::Error::Msg(format!(
            "AOCL probe gemm produced wrong result: {c:?} != [1, 2, 3, 4]"
        )));
    }
    Ok(())
}

/// AOCL-BLAS-backed CPU executor.
///
/// Holds an inner [`CpuBackend`] and delegates every method to it
/// except [`GraphBackend::matmul`], which goes through `aocl_blas::gemm`.
pub struct AoclBackend {
    cpu: CpuBackend,
}

impl AoclBackend {
    /// Construct an `AoclBackend` after probing `libaocl_blas`. Returns
    /// `Err` if the library is missing or the probe gemm produces a
    /// wrong answer (defensive — would indicate a broken install).
    pub fn try_new() -> Result<Self> {
        probe_aocl_loadable()?;
        Ok(Self { cpu: CpuBackend })
    }
}

impl GraphBackend for AoclBackend {
    type Storage = AnyRefTensor;

    // -- memory: delegate -------------------------------------------------

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        self.cpu.alloc_zeros(shape, dtype)
    }
    fn upload(&self, buf: &fuel_core_types::HostBuffer, shape: &Shape) -> Result<Self::Storage> {
        self.cpu.upload(buf, shape)
    }
    fn download(&self, storage: &Self::Storage) -> Result<fuel_core_types::HostBuffer> {
        self.cpu.download(storage)
    }
    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        self.cpu.try_clone(storage, layout)
    }
    fn copy_strided_src(
        &self,
        src: &Self::Storage,
        dst: &mut Self::Storage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        self.cpu.copy_strided_src(src, dst, dst_offset, src_layout)
    }
    fn storage_dtype(&self, storage: &Self::Storage) -> DType {
        self.cpu.storage_dtype(storage)
    }

    // -- compute: matmul native, everything else delegates ----------------

    fn matmul(
        &self,
        a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> Result<Self::Storage> {
        // AOCL has no special handling for non-contiguous inputs,
        // mixed-precision activations × bf16 weights, or the GQA
        // cached-decode tile pattern. Defer all of those to the inner
        // CpuBackend (which already materializes views, expands B for
        // GQA, and upcasts mixed-precision). Only the "happy path" —
        // contiguous f32 × f32 with matching shapes — runs through
        // aocl_blas::gemm.
        let happy_f32 = matches!((a, b), (AnyRefTensor::F32(_), AnyRefTensor::F32(_)))
            && la.is_contiguous()
            && lb.is_contiguous();
        if !happy_f32 {
            return self.cpu.matmul(a, b, bmnk, la, lb);
        }
        let (af, bf) = match (a, b) {
            (AnyRefTensor::F32(af), AnyRefTensor::F32(bf)) => (af, bf),
            _ => unreachable!("happy_f32 guards this match"),
        };
        // Storage shape must match the layout shape for the contiguous
        // happy path.
        if af.shape().dims() != la.shape().dims() || bf.shape().dims() != lb.shape().dims() {
            return self.cpu.matmul(a, b, bmnk, la, lb);
        }
        Ok(AnyRefTensor::F32(matmul_f32_aocl(af, bf)))
    }

    fn conv2d(
        &self,
        input:  &Self::Storage,
        weight: &Self::Storage,
        input_layout:  &Layout,
        weight_layout: &Layout,
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    ) -> Result<Self::Storage> {
        // f32 contiguous fast path: im2col + AOCL gemm. Other dtypes
        // and non-contiguous layouts fall through to CpuBackend
        // (which uses the reference's nested-loop conv2d via the
        // executor's trait-default fallback path).
        let happy_f32 = matches!(input, AnyRefTensor::F32(_))
            && matches!(weight, AnyRefTensor::F32(_))
            && input_layout.is_contiguous()
            && weight_layout.is_contiguous();
        if !happy_f32 {
            return self.cpu.conv2d(
                input, weight, input_layout, weight_layout, stride, padding, groups,
            );
        }
        let i_dims = input_layout.shape().dims();
        let w_dims = weight_layout.shape().dims();
        if i_dims.len() != 4 || w_dims.len() != 4 {
            return self.cpu.conv2d(
                input, weight, input_layout, weight_layout, stride, padding, groups,
            );
        }
        let (xf, wf) = match (input, weight) {
            (AnyRefTensor::F32(x), AnyRefTensor::F32(w)) => (x, w),
            _ => unreachable!("happy_f32 guards this match"),
        };
        let s = fuel_conv::ConvShape {
            batch: i_dims[0], c_in: i_dims[1], h: i_dims[2], w: i_dims[3],
            c_out: w_dims[0], k_h: w_dims[2], k_w: w_dims[3],
            stride, padding, groups,
        };
        if s.validate().is_err() {
            return self.cpu.conv2d(
                input, weight, input_layout, weight_layout, stride, padding, groups,
            );
        }
        let mut out = vec![0.0_f32; s.output_len()];
        let mut patches = vec![0.0_f32; s.im2col_len()];
        fuel_conv::conv2d_via_gemm(
            xf.as_slice(), wf.as_slice(), None,
            &s, &mut out, &mut patches,
            |m, n, k, a, b, c| {
                use aocl_types::Trans;
                aocl_blas::gemm(
                    Trans::No, Trans::No,
                    m, n, k,
                    1.0_f32, a, b,
                    0.0_f32, c,
                ).expect("aocl_blas::gemm in conv2d_via_gemm");
            },
        );
        Ok(AnyRefTensor::F32(RefTensor::from_vec(
            out,
            Shape::from_dims(&[s.batch, s.c_out, s.h_out(), s.w_out()]),
        )))
    }

    fn unary(&self, op: UnaryOp, a: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        self.cpu.unary(op, a, layout)
    }
    fn binary(
        &self, op: BinaryOp, a: &Self::Storage, b: &Self::Storage,
        la: &Layout, lb: &Layout,
    ) -> Result<Self::Storage> {
        self.cpu.binary(op, a, b, la, lb)
    }
    fn affine(&self, a: &Self::Storage, layout: &Layout, mul: f64, add: f64) -> Result<Self::Storage> {
        self.cpu.affine(a, layout, mul, add)
    }
    fn powf(&self, a: &Self::Storage, layout: &Layout, exp: f64) -> Result<Self::Storage> {
        self.cpu.powf(a, layout, exp)
    }
    fn cast(&self, a: &Self::Storage, layout: &Layout, dtype: DType) -> Result<Self::Storage> {
        self.cpu.cast(a, layout, dtype)
    }
    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, layout: &Layout, dims: &[usize],
    ) -> Result<Self::Storage> {
        self.cpu.reduce(op, a, layout, dims)
    }
    fn softmax_last_dim(&self, a: &Self::Storage, layout: &Layout) -> Result<Self::Storage> {
        self.cpu.softmax_last_dim(a, layout)
    }
    fn rms_norm_last_dim(&self, a: &Self::Storage, layout: &Layout, eps: f64) -> Result<Self::Storage> {
        self.cpu.rms_norm_last_dim(a, layout, eps)
    }
    fn rms_norm_last_dim_backward(
        &self, x: &Self::Storage, upstream: &Self::Storage,
        xl: &Layout, ul: &Layout, eps: f64,
    ) -> Result<Self::Storage> {
        self.cpu.rms_norm_last_dim_backward(x, upstream, xl, ul, eps)
    }
    fn rope(
        &self, x: &Self::Storage, cos: &Self::Storage, sin: &Self::Storage,
        xl: &Layout, cl: &Layout, sl: &Layout,
    ) -> Result<Self::Storage> {
        self.cpu.rope(x, cos, sin, xl, cl, sl)
    }
    fn add_assign_scaled(
        &self, dst: &mut Self::Storage, src: &Self::Storage, scale: f32,
    ) -> Result<()> {
        self.cpu.add_assign_scaled(dst, src, scale)
    }
    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        sl: &Layout, il: &Layout, dim: usize,
    ) -> Result<Self::Storage> {
        self.cpu.index_select(src, ids, sl, il, dim)
    }
    fn gather(
        &self, src: &Self::Storage, ids: &Self::Storage,
        sl: &Layout, il: &Layout, dim: usize,
    ) -> Result<Self::Storage> {
        self.cpu.gather(src, ids, sl, il, dim)
    }
}

/// Batched f32 matmul via `aocl_blas::gemm`. Mirrors
/// `fuel_graph_cpu::fast_matmul::matmul_f32` in shape but routes each
/// per-batch slice through AOCL-BLAS.
fn matmul_f32_aocl(a: &RefTensor<f32>, b: &RefTensor<f32>) -> RefTensor<f32> {
    use aocl_types::Trans;
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    let rank = a_dims.len();
    assert!(rank >= 2, "matmul: rank ≥ 2 required");
    assert_eq!(rank, b_dims.len(), "matmul: rank mismatch");
    for i in 0..rank - 2 {
        assert_eq!(a_dims[i], b_dims[i], "matmul: batch dim mismatch axis {i}");
    }
    let m = a_dims[rank - 2];
    let k = a_dims[rank - 1];
    let k2 = b_dims[rank - 2];
    let n = b_dims[rank - 1];
    assert_eq!(k, k2, "matmul: inner dim mismatch");

    let batch_dims = &a_dims[..rank - 2];
    let batch_count: usize = batch_dims.iter().product::<usize>().max(1);
    let mut out_dims: Vec<usize> = batch_dims.to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let mut out = vec![0.0_f32; batch_count * m * n];

    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    for batch in 0..batch_count {
        let a_off = batch * a_stride;
        let b_off = batch * b_stride;
        let c_off = batch * c_stride;
        aocl_blas::gemm(
            Trans::No, Trans::No,
            m, n, k,
            1.0_f32,
            &a_data[a_off..a_off + a_stride],
            &b_data[b_off..b_off + b_stride],
            0.0_f32,
            &mut out[c_off..c_off + c_stride],
        ).expect("aocl_blas::gemm");
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

#[cfg(test)]
mod tests {
    /// Phase 7.5 G2: tests need a real device for slot-populating
    /// constructors. Singleton CpuBackendDevice via OnceLock.
    fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
        static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    use super::*;
    use fuel_graph::Tensor;

    /// AOCL must agree with the reference backend on a small matmul.
    /// Skipped (returns early) when `try_new` errors — this lets the
    /// test pass on machines where AOCL isn't installed.
    #[test]
    fn aocl_matmul_matches_reference_when_available() {
        let backend = match AoclBackend::try_new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("AOCL not available, skipping: {e}");
                return;
            }
        };
        let a = Tensor::from_f32(
            (0..12).map(|i| i as f32 * 0.1 - 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[3, 4]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            (0..20).map(|i| (i as f32 - 10.0) * 0.05).collect::<Vec<_>>(),
            Shape::from_dims(&[4, 5]),
        );
        let c = a.matmul(&b);
        let mut exe = fuel_graph_executor::GraphExecutor::new(backend);
        let out = exe.realize_f32(&c).into_vec();
        let reference = fuel_reference_backend::exec::realize_f32(&c).into_vec();
        assert_eq!(out.len(), reference.len());
        for (i, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
            let denom = o.abs().max(r.abs()).max(f32::MIN_POSITIVE);
            let rel = (o - r).abs() / denom;
            assert!(rel < 1e-4, "at index {i}: aocl={o}, ref={r} (rel {rel})");
        }
    }

    /// AOCL must agree with the reference backend on a depthwise
    /// conv2d (groups = c_in = c_out). Validates that the executor's
    /// post-screen-removal lets `groups > 1` reach the AOCL backend
    /// instead of CPU-falling-back.
    #[test]
    fn aocl_depthwise_conv2d_matches_reference_when_available() {
        let backend = match AoclBackend::try_new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("AOCL not available, skipping: {e}");
                return;
            }
        };
        let (n, c, h, w) = (1usize, 4, 6, 6);
        let (k, pad) = (3, 1);
        let x = Tensor::from_f32(
            (0..(n * c * h * w)).map(|i| ((i as f32) * 1.3e-2).sin()).collect::<Vec<_>>(),
            Shape::from_dims(&[n, c, h, w]),
            cpu_dev(),
        );
        let weight = x.const_f32_like(
            (0..(c * 1 * k * k)).map(|i| ((i as f32) * 1.7e-2).cos()).collect::<Vec<_>>(),
            Shape::from_dims(&[c, 1, k, k]),
        );
        let y = x.conv2d(&weight, None, (1, 1), (pad, pad), c);
        let mut exe = fuel_graph_executor::GraphExecutor::new(backend);
        let out = exe.realize_f32(&y).into_vec();
        let reference = fuel_reference_backend::exec::realize_f32(&y).into_vec();
        assert_eq!(out.len(), reference.len());
        for (i, (&o, &r)) in out.iter().zip(reference.iter()).enumerate() {
            let denom = o.abs().max(r.abs()).max(f32::MIN_POSITIVE);
            let rel = (o - r).abs() / denom;
            assert!(rel < 1e-4, "at index {i}: aocl={o}, ref={r} (rel {rel})");
        }
    }
}

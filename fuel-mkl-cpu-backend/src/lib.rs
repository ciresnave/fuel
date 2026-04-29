//! # fuel-mkl-cpu-backend
//!
//! Intel oneMKL-backed CPU executor for the fuel lazy-graph layer.
//!
//! Mirrors `fuel-aocl-cpu-backend`'s shape but routes the matmul fast
//! path through `onemkl::blas::level3::gemm` instead of AOCL-BLAS or
//! the cross-vendor `gemm` crate. All other ops delegate to a contained
//! [`CpuBackend`] so the spike crate stays small.
//!
//! On Intel CPUs (and many AMD CPUs too — MKL detects vendor at runtime
//! and dispatches accordingly), this should be the matmul winner. The
//! Phase 6b judge profiles MKL alongside any other registered CPU
//! backend (CPU/AOCL/…) and the dispatch table picks per
//! `(op, dtype, size_class)`.
//!
//! # Availability gate
//!
//! [`MklBackend::try_new`] runs a 2×2 sgemm at construction time; if
//! `mkl_rt` doesn't load on this machine the constructor returns `Err`
//! and the rest of Fuel never sees an MKL backend. Backends in Fuel's
//! design own their own availability check — there is no
//! HardwareQuery layer gating them.
//!
//! # Storage type
//!
//! MKL is a CPU backend, so its `Storage` is `AnyRefTensor` —
//! identical to `fuel-graph-cpu`'s and `fuel-aocl-cpu-backend`'s.
//! `AnyStorage::Cpu` wraps all three; switching among them is a
//! vtable swap, not a transfer.

mod dll_path;
pub mod probe;

use fuel_core_types::{DType, Layout, Result, Shape};
use fuel_graph_cpu::CpuBackend;
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use fuel_reference_backend::exec::AnyRefTensor;
use fuel_reference_backend::RefTensor;

/// Probe `mkl_rt` (or whichever oneMKL runtime resolves) with a 2×2
/// sgemm. Returns `Ok` on a successful call producing the right
/// answer, `Err` if the library can't be loaded or the probe gemm
/// returns wrong values. Public so callers that just want a
/// "is oneMKL available?" signal can use it without constructing the
/// backend.
///
/// On Windows, this best-effort extends `PATH` with the standard
/// oneMKL bin directory if it isn't already there — see [`dll_path`]
/// for the discovery order. Without this, a default-launched
/// `cargo run --features onemkl` would crash with
/// `STATUS_DLL_NOT_FOUND` unless the user pre-runs `setvars.bat`.
pub fn probe_mkl_loadable() -> Result<()> {
    dll_path::ensure_loadable();
    use onemkl::enums::{Layout as MklLayout, Transpose};
    use onemkl::matrix::{MatrixMut, MatrixRef};

    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [1.0_f32, 0.0, 0.0, 1.0];
    let mut c = [0.0_f32; 4];

    let a_ref = MatrixRef::new(&a, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixRef::new(a) failed: {e}")))?;
    let b_ref = MatrixRef::new(&b, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixRef::new(b) failed: {e}")))?;
    let mut c_mut = MatrixMut::new(&mut c, 2, 2, MklLayout::RowMajor)
        .map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe MatrixMut::new(c) failed: {e}")))?;

    onemkl::blas::level3::gemm(
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0_f32,
        &a_ref,
        &b_ref,
        0.0_f32,
        &mut c_mut,
    ).map_err(|e| fuel_core_types::Error::Msg(format!("MKL probe gemm failed: {e}")))?;

    if c != [1.0, 2.0, 3.0, 4.0] {
        return Err(fuel_core_types::Error::Msg(format!(
            "MKL probe gemm produced wrong result: {c:?} != [1, 2, 3, 4]"
        )));
    }
    Ok(())
}

/// Intel oneMKL-backed CPU executor.
///
/// Holds an inner [`CpuBackend`] and delegates every method to it
/// except [`GraphBackend::matmul`], which goes through
/// `onemkl::blas::level3::gemm`.
pub struct MklBackend {
    cpu: CpuBackend,
}

impl MklBackend {
    /// Construct an `MklBackend` after probing `mkl_rt`. Returns `Err`
    /// if the library is missing or the probe gemm produces a wrong
    /// answer (defensive — would indicate a broken install).
    pub fn try_new() -> Result<Self> {
        probe_mkl_loadable()?;
        Ok(Self { cpu: CpuBackend })
    }
}

impl GraphBackend for MklBackend {
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
        // Mirror AoclBackend's "happy path or delegate" guard. MKL via
        // the oneMKL crate also doesn't natively handle our lazy-view,
        // mixed-precision, or GQA-tiled paths; defer those to the
        // inner CpuBackend (which materializes views, expands B for
        // GQA, and upcasts mixed-precision).
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
        if af.shape().dims() != la.shape().dims() || bf.shape().dims() != lb.shape().dims() {
            return self.cpu.matmul(a, b, bmnk, la, lb);
        }
        Ok(AnyRefTensor::F32(matmul_f32_mkl(af, bf)))
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
        // f32 contiguous fast path: im2col + oneMKL gemm. Other dtypes
        // and non-contiguous layouts fall through to CpuBackend.
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
                use onemkl::enums::{Layout as MklLayout, Transpose};
                use onemkl::matrix::{MatrixMut, MatrixRef};
                let a_ref = MatrixRef::new(a, m, k, MklLayout::RowMajor)
                    .expect("MatrixRef::new(a) in conv2d_via_gemm");
                let b_ref = MatrixRef::new(b, k, n, MklLayout::RowMajor)
                    .expect("MatrixRef::new(b) in conv2d_via_gemm");
                let mut c_mut = MatrixMut::new(c, m, n, MklLayout::RowMajor)
                    .expect("MatrixMut::new(c) in conv2d_via_gemm");
                onemkl::blas::level3::gemm(
                    Transpose::NoTrans, Transpose::NoTrans,
                    1.0_f32, &a_ref, &b_ref, 0.0_f32, &mut c_mut,
                ).expect("onemkl::gemm in conv2d_via_gemm");
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

/// Batched f32 matmul via `onemkl::blas::level3::gemm`. Mirrors
/// `fuel_aocl_cpu_backend::matmul_f32_aocl` in shape but routes each
/// per-batch slice through oneMKL.
fn matmul_f32_mkl(a: &RefTensor<f32>, b: &RefTensor<f32>) -> RefTensor<f32> {
    use onemkl::enums::{Layout as MklLayout, Transpose};
    use onemkl::matrix::{MatrixMut, MatrixRef};

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
        let a_ref = MatrixRef::new(
            &a_data[a_off..a_off + a_stride], m, k, MklLayout::RowMajor,
        ).expect("MatrixRef::new(a)");
        let b_ref = MatrixRef::new(
            &b_data[b_off..b_off + b_stride], k, n, MklLayout::RowMajor,
        ).expect("MatrixRef::new(b)");
        let mut c_mut = MatrixMut::new(
            &mut out[c_off..c_off + c_stride], m, n, MklLayout::RowMajor,
        ).expect("MatrixMut::new(c)");
        onemkl::blas::level3::gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0_f32,
            &a_ref,
            &b_ref,
            0.0_f32,
            &mut c_mut,
        ).expect("onemkl::gemm");
    }

    RefTensor::from_vec(out, Shape::from_dims(&out_dims))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_graph::Tensor;

    /// MKL must agree with the reference backend on a small matmul.
    /// Skipped (returns early) when `try_new` errors — this lets the
    /// test pass on machines where MKL isn't installed.
    #[test]
    fn mkl_matmul_matches_reference_when_available() {
        let backend = match MklBackend::try_new() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("MKL not available, skipping: {e}");
                return;
            }
        };
        let a = Tensor::from_f32(
            (0..12).map(|i| i as f32 * 0.1 - 0.5).collect::<Vec<_>>(),
            Shape::from_dims(&[3, 4]),
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
            assert!(rel < 1e-4, "at index {i}: mkl={o}, ref={r} (rel {rel})");
        }
    }
}

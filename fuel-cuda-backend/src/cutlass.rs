//! CUTLASS bridge — registers CUTLASS-backed GEMM kernels as alternative
//! implementations at the `(MatMul, *, Cuda)` and `(FusedLinear, *, Cuda)`
//! decision points.
//!
//! Per architecture v1.0, CUTLASS kernels are *siblings* to the cuBLAS
//! path, not replacements. The optimizer + route-picker rank them by
//! `PrecisionGuarantee` + empirical telemetry. cuBLAS provides bit-stable
//! coverage; CUTLASS provides throughput alternatives (TF32, Rrr-layout
//! GEMM, fused Bias/BiasReLU/BiasGELU/BiasSiLU epilogues).
//!
//! Today's binding-table (single impl per `(op, dtypes, backend)` key)
//! still hosts raw `Op::MatMul`. The architecture-target sibling
//! registration shape (`FusedKernelRegistry::register` with append-on-
//! register semantics) is exercised by the `Op::FusedLinear` kernels
//! below. Step 9 will migrate primitive ops to the same surface, at
//! which point the raw-matmul CUTLASS entries gain sibling status
//! alongside cuBLAS automatically.

use baracuda_cutlass::{
    CutlassElement, EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut,
    MatrixRef, PlanPreference, Workspace,
};
use baracuda_types::DeviceRepr;
use fuel_core_types::{Error, Result};
use half::{bf16, f16};

use crate::byte_storage::CudaStorageBytes;

/// Generic CUTLASS Rrr matmul on the byte-storage substrate. Equal-
/// batch path only — caller is responsible for splitting GQA /
/// unequal batches into multiple invocations.
///
/// Layout is `LayoutSku::Rrr`: A row-major `[M, K]`, B row-major
/// `[K, N]`, D row-major `[M, N]` — matches `Op::MatMul`'s natural
/// shape so no transpose pass is needed before launch.
///
/// Returns a freshly allocated `CudaStorageBytes` containing the
/// `batch_count × M × N` output. Stream is sync'd before return so
/// the result is observable through the byte buffer (sync KernelRef
/// per architecture v1.0).
///
/// `T` must be a CUTLASS-supported Rrr element type — alpha.13 ships
/// kernels for `f16` and `bf16`. f32 input is Rcr-only (see B5).
fn cutlass_matmul_rrr<T>(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes>
where
    // `Scalar = f32`: baracuda alpha.26 added an associated `Scalar` type on
    // `Element` (the renamed `CutlassElement`) for the GEMM epilogue's α/β
    // compute precision. f16, bf16, and f32-input kernels all use an f32
    // epilogue; only f64 uses Scalar = f64. Constraining here keeps the
    // generic happy with naked numeric literals.
    T: CutlassElement<Scalar = f32> + DeviceRepr,
{
    let dtype_label = std::any::type_name::<T>();
    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(Error::Msg(format!(
            "cutlass_matmul_rrr<{dtype_label}>: lhs and rhs are on different CUDA \
             devices; cross-device matmul is the caller's responsibility (insert \
             Op::Move first)"
        ))
        .bt());
    }
    let elem = std::mem::size_of::<T>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let out_per_batch = m.saturating_mul(n);
    let need_out_bytes = batch_count
        .saturating_mul(out_per_batch)
        .saturating_mul(elem);
    if need_out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let m_i32: i32 = i32::try_from(m).map_err(|_| {
        Error::Msg(format!("cutlass_matmul_rrr<{dtype_label}>: M={m} exceeds i32 range"))
            .bt()
    })?;
    let n_i32: i32 = i32::try_from(n).map_err(|_| {
        Error::Msg(format!("cutlass_matmul_rrr<{dtype_label}>: N={n} exceeds i32 range"))
            .bt()
    })?;
    let k_i32: i32 = i32::try_from(k).map_err(|_| {
        Error::Msg(format!("cutlass_matmul_rrr<{dtype_label}>: K={k} exceeds i32 range"))
            .bt()
    })?;

    let mut out = device.alloc_zeros::<u8>(need_out_bytes)?;
    let lhs_view = lhs.view_as::<T>()?;
    let rhs_view = rhs.view_as::<T>()?;
    let mut out_view = out.view_as_mut::<T>();
    let stream = device.stream();

    let desc = GemmDescriptor {
        m: m_i32,
        n: n_i32,
        k: k_i32,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = GemmPlan::<T>::select(stream, &desc, PlanPreference::default())
        .map_err(|e| Error::Msg(format!("cutlass plan select ({dtype_label} Rrr): {e}")).bt())?;

    for b in 0..batch_count {
        let a_off = b * lhs_per_batch;
        let b_off = b * rhs_per_batch;
        let d_off = b * out_per_batch;

        let a_slice = lhs_view.slice(a_off..a_off + lhs_per_batch);
        let b_slice = rhs_view.slice(b_off..b_off + rhs_per_batch);
        let d_slice = out_view.slice_mut(d_off..d_off + out_per_batch);

        let args = GemmArgs::<T> {
            a: MatrixRef {
                data: a_slice,
                rows: m_i32,
                cols: k_i32,
                ld: k as i64,
            },
            b: MatrixRef {
                data: b_slice,
                rows: k_i32,
                cols: n_i32,
                ld: n as i64,
            },
            c: None,
            d: MatrixMut {
                data: d_slice,
                rows: m_i32,
                cols: n_i32,
                ld: n as i64,
            },
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        };
        plan.can_implement(&args).map_err(|e| {
            Error::Msg(format!("cutlass can_implement ({dtype_label} Rrr): {e}")).bt()
        })?;
        plan.run(stream, Workspace::None, args).map_err(|e| {
            Error::Msg(format!("cutlass run ({dtype_label} Rrr): {e}")).bt()
        })?;
    }
    drop(out_view);

    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        std::sync::Arc::new(out),
        device,
        need_out_bytes,
    ))
}

/// CUTLASS bf16 matmul. Thin instantiation of [`cutlass_matmul_rrr`]
/// at `T = bf16`. Public entry from byte_kernels::matmul_bf16.
pub fn cutlass_matmul_bf16(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    cutlass_matmul_rrr::<bf16>(lhs, rhs, batch_count, m, n, k)
}

/// CUTLASS f16 matmul. Thin instantiation of [`cutlass_matmul_rrr`]
/// at `T = f16`. Public entry from byte_kernels::matmul_f16.
pub fn cutlass_matmul_f16(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    cutlass_matmul_rrr::<f16>(lhs, rhs, batch_count, m, n, k)
}

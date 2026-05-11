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
    EpilogueKind, GemmArgs, GemmDescriptor, GemmPlan, LayoutSku, MatrixMut, MatrixRef,
    PlanPreference, Workspace,
};
use fuel_core_types::{Error, Result};
use half::bf16;

use crate::byte_storage::CudaStorageBytes;

/// CUTLASS bf16 matmul on the byte-storage substrate. Equal-batch
/// path only — caller is responsible for splitting GQA / unequal
/// batches into multiple invocations.
///
/// Layout is `LayoutSku::Rrr`: A row-major `[M, K]`, B row-major
/// `[K, N]`, D row-major `[M, N]` — matches `Op::MatMul`'s natural
/// shape so no transpose pass is needed before launch.
///
/// Returns a freshly allocated `CudaStorageBytes` containing the
/// `batch_count × M × N` output. Stream is sync'd before return so
/// the result is observable through the byte buffer (sync KernelRef
/// per architecture v1.0).
pub fn cutlass_matmul_bf16(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    batch_count: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(Error::Msg(
            "cutlass_matmul_bf16: lhs and rhs are on different CUDA devices; cross-device \
             matmul is the caller's responsibility (insert Op::Move first)"
                .to_string(),
        )
        .bt());
    }
    let elem = std::mem::size_of::<bf16>();
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
        Error::Msg(format!("cutlass_matmul_bf16: M={m} exceeds i32 range")).bt()
    })?;
    let n_i32: i32 = i32::try_from(n).map_err(|_| {
        Error::Msg(format!("cutlass_matmul_bf16: N={n} exceeds i32 range")).bt()
    })?;
    let k_i32: i32 = i32::try_from(k).map_err(|_| {
        Error::Msg(format!("cutlass_matmul_bf16: K={k} exceeds i32 range")).bt()
    })?;

    let mut out = device.alloc_zeros::<u8>(need_out_bytes)?;
    let lhs_view = lhs.view_as::<bf16>()?;
    let rhs_view = rhs.view_as::<bf16>()?;
    let mut out_view = out.view_as_mut::<bf16>();
    let stream = device.stream();

    let desc = GemmDescriptor {
        m: m_i32,
        n: n_i32,
        k: k_i32,
        layout: LayoutSku::Rrr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = GemmPlan::<bf16>::select(stream, &desc, PlanPreference::default())
        .map_err(|e| Error::Msg(format!("cutlass plan select (bf16 Rrr): {e}")).bt())?;

    for b in 0..batch_count {
        let a_off = b * lhs_per_batch;
        let b_off = b * rhs_per_batch;
        let d_off = b * out_per_batch;

        let a_slice = lhs_view.slice(a_off..a_off + lhs_per_batch);
        let b_slice = rhs_view.slice(b_off..b_off + rhs_per_batch);
        let d_slice = out_view.slice_mut(d_off..d_off + out_per_batch);

        let args = GemmArgs::<bf16> {
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
            Error::Msg(format!("cutlass can_implement (bf16 Rrr): {e}")).bt()
        })?;
        plan.run(stream, Workspace::None, args).map_err(|e| {
            Error::Msg(format!("cutlass run (bf16 Rrr): {e}")).bt()
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

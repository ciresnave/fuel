//! Binding-table integration: register MKL kernels as sibling
//! alternatives on the unified byte-storage CPU dispatch path.
//!
//! Counterpart to the legacy `MklBackend` `GraphBackend` impl in
//! [`crate::MklBackend`]. That impl lives on the `GraphBackend::matmul`
//! trait path which Phase 7.6 Step 9c is migrating away from; this
//! module is where MKL lands in the post-Step-9c world.
//!
//! ## Activation
//!
//! Callers wire MKL into the global binding table after a successful
//! `probe_mkl_loadable()`:
//!
//! ```ignore
//! use fuel_storage::dispatch::extend_global_bindings;
//! use fuel_mkl_cpu_backend::{probe_mkl_loadable, register_mkl_cpu_kernels};
//!
//! if probe_mkl_loadable().is_ok() {
//!     extend_global_bindings(register_mkl_cpu_kernels);
//! }
//! ```
//!
//! Registered as a *sibling* alternative on `(MatMul, [F32, F32, F32],
//! Cpu)`; the binding-table judge picks among MKL and the scalar CPU
//! impl per-(op, dtype, size).

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Error, Layout, Result};
use fuel_storage::{
    dispatch::{cpu_input, cpu_output, read_storage, write_storage},
    kernel::OpParams,
    KernelBindingTable, Storage,
};

/// Register MKL's CPU-side wrappers as sibling alternatives on the
/// unified binding table. Trust the caller has already probed MKL (the
/// `probe_mkl_loadable` call); this function only wires registrations.
///
/// Today: `MatMul, F32`. Conv2D + int GEMM follow in their own commits.
pub fn register_mkl_cpu_kernels(table: &mut KernelBindingTable) {
    let cpu = BackendId::Cpu;
    let f32_dt = DType::F32;
    table.register(
        OpKind::MatMul,
        &[f32_dt, f32_dt, f32_dt],
        cpu,
        matmul_f32_mkl_cpu_wrapper,
    );
}

/// `(MatMul, F32, Cpu)` sibling alternative routed through
/// `onemkl::blas::level3::gemm`. Mirrors the scalar `matmul_f32`
/// wrapper's shape (OpParams extraction + per-axis GQA loop) but
/// dispatches each per-batch `[m, k] @ [k, n]` slice through MKL
/// instead of the textbook triple loop.
///
/// Inputs are guaranteed contiguous by the executor's auto-Contiguize
/// pass — same contract as the scalar wrapper.
fn matmul_f32_mkl_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl wrapper expects 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (lhs_batch_dims, rhs_batch_dims, m, n, k) = match params {
        OpParams::Matmul {
            lhs_batch_dims,
            rhs_batch_dims,
            m,
            n,
            k,
        } => (lhs_batch_dims, rhs_batch_dims, *m, *n, *k),
        other => {
            return Err(Error::Msg(format!(
                "matmul_f32_mkl wrapper expects OpParams::Matmul, got {other:?}",
            ))
            .bt())
        }
    };

    let lhs_guard = read_storage(&inputs[0])?;
    let rhs_guard = read_storage(&inputs[1])?;
    let mut out_guard = write_storage(&outputs[0])?;
    let lhs_cpu = cpu_input(&lhs_guard)?;
    let rhs_cpu = cpu_input(&rhs_guard)?;
    let out_cpu = cpu_output(&mut out_guard)?;

    matmul_f32_mkl_bytes(
        lhs_cpu,
        rhs_cpu,
        out_cpu,
        lhs_batch_dims,
        rhs_batch_dims,
        m,
        n,
        k,
    )
}

/// Batched row-major f32 matmul on byte storage via oneMKL. Per-axis
/// the batch dims either match or follow GQA-style divisibility
/// (`lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0`); each lhs batch slot
/// maps to a rhs slot via `rhs_idx = lhs_idx / n_rep`.
fn matmul_f32_mkl_bytes(
    lhs: &fuel_cpu_backend::CpuStorageBytes,
    rhs: &fuel_cpu_backend::CpuStorageBytes,
    out: &mut fuel_cpu_backend::CpuStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use onemkl::enums::{Layout as MklLayout, Transpose};
    use onemkl::matrix::{MatrixMut, MatrixRef};

    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl: batch ranks must match (lhs={}, rhs={})",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    let batch_rank = lhs_batch_dims.len();
    let mut n_rep: Vec<usize> = Vec::with_capacity(batch_rank);
    for i in 0..batch_rank {
        let la = lhs_batch_dims[i];
        let ra = rhs_batch_dims[i];
        if la == ra {
            n_rep.push(1);
        } else if ra > 0 && la > ra && la % ra == 0 {
            n_rep.push(la / ra);
        } else {
            return Err(Error::Msg(format!(
                "matmul_f32_mkl: batch dim {i} disallowed combination (lhs={la}, rhs={ra}); \
                 must be equal or GQA-divisible (lhs > rhs && lhs % rhs == 0)",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f32>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let out_per_batch = m.saturating_mul(n);
    let lhs_batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_batch_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = lhs_batch_count.saturating_mul(lhs_per_batch).saturating_mul(elem);
    let need_rhs = rhs_batch_count.saturating_mul(rhs_per_batch).saturating_mul(elem);
    let need_out = lhs_batch_count.saturating_mul(out_per_batch).saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f32)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f32)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "matmul_f32_mkl: out bytes={} doesn't match shape {:?} + [{m}, {n}] (f32)",
            out.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    let lhs_view: &[f32] = lhs.as_slice()?;
    let rhs_view: &[f32] = rhs.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;

    let mut lhs_multi = vec![0usize; batch_rank];
    let mut rhs_multi = vec![0usize; batch_rank];
    for b in 0..lhs_batch_count {
        let mut rem = b;
        for d in (0..batch_rank).rev() {
            let s = lhs_batch_dims[d];
            lhs_multi[d] = rem % s;
            rem /= s;
        }
        for d in 0..batch_rank {
            rhs_multi[d] = lhs_multi[d] / n_rep[d];
        }
        let mut rhs_b = 0usize;
        for d in 0..batch_rank {
            rhs_b = rhs_b * rhs_batch_dims[d] + rhs_multi[d];
        }
        let lhs_off = b * lhs_per_batch;
        let rhs_off = rhs_b * rhs_per_batch;
        let out_off = b * out_per_batch;

        let a_slice = &lhs_view[lhs_off..lhs_off + lhs_per_batch];
        let b_slice = &rhs_view[rhs_off..rhs_off + rhs_per_batch];
        let c_slice = &mut out_view[out_off..out_off + out_per_batch];

        let a_ref = MatrixRef::new(a_slice, m, k, MklLayout::RowMajor).map_err(|e| {
            Error::Msg(format!("matmul_f32_mkl: MatrixRef::new(lhs) failed: {e}")).bt()
        })?;
        let b_ref = MatrixRef::new(b_slice, k, n, MklLayout::RowMajor).map_err(|e| {
            Error::Msg(format!("matmul_f32_mkl: MatrixRef::new(rhs) failed: {e}")).bt()
        })?;
        let mut c_mut = MatrixMut::new(c_slice, m, n, MklLayout::RowMajor).map_err(|e| {
            Error::Msg(format!("matmul_f32_mkl: MatrixMut::new(out) failed: {e}")).bt()
        })?;
        onemkl::blas::level3::gemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            1.0_f32,
            &a_ref,
            &b_ref,
            0.0_f32,
            &mut c_mut,
        )
        .map_err(|e| Error::Msg(format!("matmul_f32_mkl: onemkl::gemm failed: {e}")).bt())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::probe::BackendId;
    use fuel_storage::dispatch::register_cpu_kernels;

    /// Registration smoke: after `register_mkl_cpu_kernels`, the
    /// MatMul/F32 binding has at least 2 alternatives (scalar CPU +
    /// MKL). Doesn't need MKL to actually load.
    #[test]
    fn mkl_matmul_registers_as_sibling_alternative() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        let before = table
            .lookup_alternatives(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .len();
        register_mkl_cpu_kernels(&mut table);
        let after = table
            .lookup_alternatives(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .len();
        assert_eq!(
            after,
            before + 1,
            "register_mkl_cpu_kernels must add exactly one alternative to (MatMul, F32, Cpu)",
        );
    }

    /// Parity: the MKL wrapper must produce bit-close output to the
    /// scalar CPU wrapper for a small rank-2 matmul. Skipped when MKL
    /// isn't loadable (probe_mkl_loadable errors).
    #[test]
    fn mkl_matmul_matches_scalar_when_available() {
        use fuel_core_types::dispatch::OpKind;
        use fuel_storage::{kernel::OpParams, BackendStorage, Storage};

        if crate::probe_mkl_loadable().is_err() {
            eprintln!("MKL not available, skipping");
            return;
        }

        // Build the binding table with both CPU + MKL.
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        register_mkl_cpu_kernels(&mut table);

        let alternatives = table.lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        );
        assert!(alternatives.len() >= 2, "need both CPU + MKL alternatives");

        // 3×4 @ 4×5 with simple values.
        let lhs_vals: Vec<f32> = (0..12).map(|i| i as f32 * 0.1 - 0.5).collect();
        let rhs_vals: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.05).collect();

        let lhs = Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(&lhs_vals)),
            DType::F32,
        )));
        let rhs = Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(&rhs_vals)),
            DType::F32,
        )));

        let alloc_out = || {
            Arc::new(RwLock::new(Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_zero_bytes(
                    3 * 5 * std::mem::size_of::<f32>(),
                )),
                DType::F32,
            )))
        };

        let params = OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m: 3,
            n: 5,
            k: 4,
        };

        // Run each alternative; collect outputs.
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            let out = alloc_out();
            let inputs = [lhs.clone(), rhs.clone()];
            let mut outs = [out.clone()];
            (alt.kernel)(&inputs, &mut outs, &[], &params).expect("alt kernel ok");
            let g = out.read().unwrap();
            #[allow(unreachable_patterns)]
            let bytes = match &g.inner {
                BackendStorage::Cpu(c) => c.as_slice::<f32>().unwrap().to_vec(),
                _ => panic!("not cpu"),
            };
            outputs.push(bytes);
        }

        let ref_out = &outputs[0];
        for (i, alt_out) in outputs.iter().enumerate().skip(1) {
            assert_eq!(alt_out.len(), ref_out.len(), "alt {i} length");
            for (j, (&a, &r)) in alt_out.iter().zip(ref_out.iter()).enumerate() {
                let denom = a.abs().max(r.abs()).max(f32::MIN_POSITIVE);
                let rel = (a - r).abs() / denom;
                assert!(rel < 1e-4, "alt {i}, idx {j}: mkl-ish={a}, scalar-ish={r} (rel {rel})");
            }
        }
    }
}

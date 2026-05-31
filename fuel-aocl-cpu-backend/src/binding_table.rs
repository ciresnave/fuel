//! Binding-table integration: register AOCL kernels as sibling
//! alternatives on the unified byte-storage CPU dispatch path.
//!
//! Counterpart to the legacy `AoclBackend` `GraphBackend` impl in
//! [`crate::AoclBackend`]. That impl lives on the `GraphBackend::matmul`
//! trait path which Phase 7.6 Step 9c is migrating away from; this
//! module is where AOCL lands in the post-Step-9c world.
//!
//! ## Activation
//!
//! Callers wire AOCL into the global binding table after a successful
//! `probe_aocl_loadable()`:
//!
//! ```ignore
//! use fuel_dispatch::dispatch::extend_global_bindings;
//! use fuel_aocl_cpu_backend::{probe_aocl_loadable, register_aocl_cpu_kernels};
//!
//! if probe_aocl_loadable().is_ok() {
//!     extend_global_bindings(register_aocl_cpu_kernels);
//! }
//! ```
//!
//! Registered as a *sibling* alternative on `(MatMul, [F32, F32, F32],
//! Cpu)`; the binding-table judge picks among AOCL and the scalar CPU
//! impl per-(op, dtype, size).

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType, Error, Layout, Result};
use fuel_dispatch::{dispatch::{cpu_input, cpu_output, read_storage, write_storage}, fused::PrecisionGuarantee, kernel::OpParams, KernelBindingTable};
use fuel_storage::{Storage};

/// Register AOCL's CPU-side wrappers as sibling alternatives on the
/// unified binding table. Trust the caller has already probed AOCL
/// (the `probe_aocl_loadable` call); this function only wires
/// registrations.
///
/// Today: `MatMul, F32` + `Conv2D, F32` (both no-bias and with-bias
/// shapes).
pub fn register_aocl_cpu_kernels(table: &mut KernelBindingTable) {
    // AOCL-BLAS (BLIS) is run-to-run deterministic on a fixed CPU +
    // thread count by default. The bit_stable_on_same_hardware claim
    // is about run-to-run determinism, not bit-equality vs the scalar
    // reference — those are different and BLIS's blocked accumulation
    // legitimately differs from scalar. Marking as audited-bit-stable
    // keeps these registrations eligible under future Judge policies
    // that filter on PrecisionGuarantee.
    const AOCL_PRECISION: PrecisionGuarantee = PrecisionGuarantee {
        bit_stable_on_same_hardware: true,
        max_ulp: None,
        max_relative: None,
        max_absolute: None,
        notes: "AOCL-BLAS (BLIS): deterministic on fixed CPU + thread \
                count; per-shape ULP bounds land with the step-8 \
                calibration framework.",
    };
    let cpu = BackendId::Cpu;
    let f32_dt = DType::F32;
    table.register_with_precision(
        OpKind::MatMul,
        &[f32_dt, f32_dt, f32_dt],
        cpu,
        matmul_f32_aocl_cpu_wrapper,
        AOCL_PRECISION,
    );
    // Conv2D — same wrapper handles both 3-operand (x, w, out) and
    // 4-operand (x, w, bias, out) keys; the wrapper distinguishes by
    // `inputs.len()`.
    table.register_with_precision(
        OpKind::Conv2D,
        &[f32_dt, f32_dt, f32_dt],
        cpu,
        conv2d_f32_aocl_cpu_wrapper,
        AOCL_PRECISION,
    );
    table.register_with_precision(
        OpKind::Conv2D,
        &[f32_dt, f32_dt, f32_dt, f32_dt],
        cpu,
        conv2d_f32_aocl_cpu_wrapper,
        AOCL_PRECISION,
    );
}

/// `(MatMul, F32, Cpu)` sibling alternative routed through
/// `aocl_blas::gemm`. Mirrors the scalar `matmul_f32` wrapper's shape
/// (OpParams extraction + per-axis GQA loop) but dispatches each
/// per-batch `[m, k] @ [k, n]` slice through AOCL-BLAS.
///
/// Inputs are guaranteed contiguous by the executor's auto-Contiguize
/// pass — same contract as the scalar wrapper.
fn matmul_f32_aocl_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::Msg(format!(
            "matmul_f32_aocl wrapper expects 2 inputs, got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "matmul_f32_aocl wrapper expects 1 output, got {}",
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
                "matmul_f32_aocl wrapper expects OpParams::Matmul, got {other:?}",
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

    matmul_f32_aocl_bytes(
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

/// Batched row-major f32 matmul on byte storage via AOCL-BLAS. Per-axis
/// the batch dims either match or follow GQA-style divisibility
/// (`lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0`); each lhs batch slot
/// maps to a rhs slot via `rhs_idx = lhs_idx / n_rep`.
fn matmul_f32_aocl_bytes(
    lhs: &fuel_cpu_backend::CpuStorageBytes,
    rhs: &fuel_cpu_backend::CpuStorageBytes,
    out: &mut fuel_cpu_backend::CpuStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    use aocl_types::Trans;

    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(Error::Msg(format!(
            "matmul_f32_aocl: batch ranks must match (lhs={}, rhs={})",
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
                "matmul_f32_aocl: batch dim {i} disallowed combination (lhs={la}, rhs={ra}); \
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
            "matmul_f32_aocl: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f32)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(Error::Msg(format!(
            "matmul_f32_aocl: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f32)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    if out.len_bytes() != need_out {
        return Err(Error::Msg(format!(
            "matmul_f32_aocl: out bytes={} doesn't match shape {:?} + [{m}, {n}] (f32)",
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

        aocl_blas::gemm(
            Trans::No,
            Trans::No,
            m,
            n,
            k,
            1.0_f32,
            a_slice,
            b_slice,
            0.0_f32,
            c_slice,
        )
        .map_err(|e| Error::Msg(format!("matmul_f32_aocl: aocl_blas::gemm failed: {e}")).bt())?;
    }
    Ok(())
}

/// `(Conv2D, F32, Cpu)` sibling alternative routed through AOCL-BLAS's
/// sgemm via `fuel_conv::conv2d_via_gemm`. Handles both 2-input
/// (x, weight) and 3-input (x, weight, bias) shapes; the binding-table
/// key carries the operand count.
///
/// `fuel_conv::ConvShape` doesn't carry a dilation field, so any
/// `dilation != (1, 1)` falls back to the scalar `conv2d_f32` kernel.
/// Same for `ConvShape::validate` failures. Inputs are guaranteed
/// contiguous f32 by the executor's auto-Contiguize pass.
fn conv2d_f32_aocl_cpu_wrapper(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    _layouts: &[Layout],
    params: &OpParams,
) -> Result<()> {
    if inputs.len() != 2 && inputs.len() != 3 {
        return Err(Error::Msg(format!(
            "conv2d_f32_aocl wrapper expects 2 or 3 inputs (x, w, [bias]), got {}",
            inputs.len(),
        ))
        .bt());
    }
    if outputs.len() != 1 {
        return Err(Error::Msg(format!(
            "conv2d_f32_aocl wrapper expects 1 output, got {}",
            outputs.len(),
        ))
        .bt());
    }
    let (x_shape, w_shape, out_shape, stride, padding, dilation, groups) = match params {
        OpParams::Conv2D {
            x_shape,
            w_shape,
            out_shape,
            stride,
            padding,
            dilation,
            groups,
        } => (*x_shape, *w_shape, *out_shape, *stride, *padding, *dilation, *groups),
        other => {
            return Err(Error::Msg(format!(
                "conv2d_f32_aocl wrapper expects OpParams::Conv2D, got {other:?}",
            ))
            .bt())
        }
    };

    let x_guard = read_storage(&inputs[0])?;
    let w_guard = read_storage(&inputs[1])?;
    let bias_guard = match inputs.get(2) {
        Some(arc) => Some(read_storage(arc)?),
        None => None,
    };
    let mut out_guard = write_storage(&outputs[0])?;
    let x_cpu = cpu_input(&x_guard)?;
    let w_cpu = cpu_input(&w_guard)?;
    let bias_cpu = match &bias_guard {
        Some(g) => Some(cpu_input(g)?),
        None => None,
    };
    let out_cpu = cpu_output(&mut out_guard)?;

    // Fall back to scalar conv2d_f32 for any shape AOCL's im2col+gemm
    // path doesn't handle: non-(1,1) dilation, or any ConvShape that
    // fails validation. The scalar kernel already handles all of these.
    if dilation != (1, 1) {
        return fuel_cpu_backend::byte_kernels::conv2d_f32(
            x_cpu, w_cpu, bias_cpu, out_cpu,
            x_shape, w_shape, out_shape, stride, padding, dilation, groups,
        );
    }
    let s = fuel_conv::ConvShape {
        batch: x_shape[0],
        c_in: x_shape[1],
        h: x_shape[2],
        w: x_shape[3],
        c_out: w_shape[0],
        k_h: w_shape[2],
        k_w: w_shape[3],
        stride,
        padding,
        groups,
    };
    if s.validate().is_err() {
        return fuel_cpu_backend::byte_kernels::conv2d_f32(
            x_cpu, w_cpu, bias_cpu, out_cpu,
            x_shape, w_shape, out_shape, stride, padding, dilation, groups,
        );
    }

    let x_view: &[f32] = x_cpu.as_slice()?;
    let w_view: &[f32] = w_cpu.as_slice()?;
    let bias_view: Option<&[f32]> = match bias_cpu {
        Some(b) => Some(b.as_slice()?),
        None => None,
    };
    let out_view: &mut [f32] = out_cpu.as_slice_mut()?;
    let mut patches = vec![0.0_f32; s.im2col_len()];

    fuel_conv::conv2d_via_gemm(
        x_view, w_view, bias_view, &s, out_view, &mut patches,
        |m, n, k, a, b, c| {
            use aocl_types::Trans;
            aocl_blas::gemm(
                Trans::No, Trans::No,
                m, n, k,
                1.0_f32, a, b,
                0.0_f32, c,
            )
            .expect("aocl_blas::gemm in conv2d_via_gemm");
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_dispatch::dispatch::register_cpu_kernels;

    /// Registration smoke: after `register_aocl_cpu_kernels`, the
    /// MatMul/F32 binding has one more alternative. Doesn't need AOCL
    /// to actually load.
    #[test]
    fn aocl_matmul_registers_as_sibling_alternative() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        let before = table
            .lookup_alternatives(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .len();
        register_aocl_cpu_kernels(&mut table);
        let after = table
            .lookup_alternatives(OpKind::MatMul, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .len();
        assert_eq!(
            after,
            before + 1,
            "register_aocl_cpu_kernels must add exactly one alternative to (MatMul, F32, Cpu)",
        );
    }

    /// Registration smoke for Conv2D: both no-bias and with-bias
    /// keys gain one alternative each.
    #[test]
    fn aocl_conv2d_registers_as_sibling_alternative() {
        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        let f32_dt = DType::F32;
        let no_bias_before = table
            .lookup_alternatives(OpKind::Conv2D, &[f32_dt, f32_dt, f32_dt], BackendId::Cpu)
            .len();
        let with_bias_before = table
            .lookup_alternatives(OpKind::Conv2D, &[f32_dt, f32_dt, f32_dt, f32_dt], BackendId::Cpu)
            .len();
        register_aocl_cpu_kernels(&mut table);
        let no_bias_after = table
            .lookup_alternatives(OpKind::Conv2D, &[f32_dt, f32_dt, f32_dt], BackendId::Cpu)
            .len();
        let with_bias_after = table
            .lookup_alternatives(OpKind::Conv2D, &[f32_dt, f32_dt, f32_dt, f32_dt], BackendId::Cpu)
            .len();
        assert_eq!(no_bias_after, no_bias_before + 1, "no-bias Conv2D");
        assert_eq!(with_bias_after, with_bias_before + 1, "with-bias Conv2D");
    }

    /// Parity: the AOCL wrapper must produce bit-close output to the
    /// scalar CPU wrapper for a small rank-2 matmul. Skipped when AOCL
    /// isn't loadable (probe_aocl_loadable errors).
    #[test]
    fn aocl_matmul_matches_scalar_when_available() {
        use fuel_storage::{BackendStorage, Storage};

        if crate::probe_aocl_loadable().is_err() {
            eprintln!("AOCL not available, skipping");
            return;
        }

        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        register_aocl_cpu_kernels(&mut table);

        let alternatives = table.lookup_alternatives(
            OpKind::MatMul,
            &[DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        );
        assert!(alternatives.len() >= 2, "need both CPU + AOCL alternatives");

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
                assert!(rel < 1e-4, "alt {i}, idx {j}: aocl-ish={a}, scalar-ish={r} (rel {rel})");
            }
        }
    }

    /// Parity: the AOCL Conv2D wrapper must produce bit-close output
    /// to the scalar CPU wrapper for a small NCHW conv. Exercises the
    /// no-bias path with stride=1, pad=0, dilation=(1,1), groups=1 —
    /// the canonical "happy" shape AOCL's im2col+gemm handles.
    #[test]
    fn aocl_conv2d_matches_scalar_when_available() {
        use fuel_storage::{BackendStorage, Storage};

        if crate::probe_aocl_loadable().is_err() {
            eprintln!("AOCL not available, skipping");
            return;
        }

        let mut table = KernelBindingTable::new();
        register_cpu_kernels(&mut table);
        register_aocl_cpu_kernels(&mut table);

        // Conv shape: N=1, Cin=2, H=4, W=4; Cout=3, kH=3, kW=3.
        let (n, cin, h, w) = (1usize, 2, 4, 4);
        let (cout, kh, kw) = (3usize, 3, 3);
        let (h_out, w_out) = (h - kh + 1, w - kw + 1); // stride=1, pad=0

        let x_vals: Vec<f32> = (0..(n * cin * h * w))
            .map(|i| ((i as f32) * 1.3e-2).sin())
            .collect();
        let w_vals: Vec<f32> = (0..(cout * cin * kh * kw))
            .map(|i| ((i as f32) * 1.7e-2).cos())
            .collect();

        let x_storage = || {
            Arc::new(RwLock::new(Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(&x_vals)),
                DType::F32,
            )))
        };
        let w_storage = || {
            Arc::new(RwLock::new(Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(&w_vals)),
                DType::F32,
            )))
        };
        let alloc_out = || {
            Arc::new(RwLock::new(Storage::new(
                BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_zero_bytes(
                    n * cout * h_out * w_out * std::mem::size_of::<f32>(),
                )),
                DType::F32,
            )))
        };

        let params = OpParams::Conv2D {
            x_shape: [n, cin, h, w],
            w_shape: [cout, cin, kh, kw],
            out_shape: [n, cout, h_out, w_out],
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        };

        let f32_dt = DType::F32;
        let alternatives = table.lookup_alternatives(
            OpKind::Conv2D,
            &[f32_dt, f32_dt, f32_dt],
            BackendId::Cpu,
        );
        assert!(alternatives.len() >= 2, "need both CPU + AOCL conv2d alternatives");

        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(alternatives.len());
        for alt in alternatives {
            let out = alloc_out();
            let inputs = [x_storage(), w_storage()];
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
                assert!(rel < 1e-4, "alt {i}, idx {j}: aocl={a}, scalar={r} (rel {rel})");
            }
        }
    }
}

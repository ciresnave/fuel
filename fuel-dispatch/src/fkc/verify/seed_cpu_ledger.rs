//! Task 4.5b — empirical seeding of the CPU verification ledger.
//!
//! Background: on 2026-07-03 (commit `18502e77`) ~18 fused-op CPU
//! `BackendImpl`s (`SoftmaxLastDim`/`RmsNormLastDim`/`LayerNormLastDim`
//! (+ backward), `FusedLinear`, `QMatMul`, `InplaceAffine`,
//! `FusedSoftmaxCrossEntropy`, `Rope`, `Conv2D`/`ConvTranspose2D`,
//! `CausalConv1d`, `SelectiveScan`, `SsdChunkScan`, `ReduceMaxToBackward`,
//! `PowIBackward` — the FKC-contract-sourced families imported by
//! `register_cpu_linear_quant_fused_from_contract` /
//! `register_cpu_norm_softmax_fused_from_contract` /
//! `register_cpu_conv_rope_fused_from_contract`, `dispatch.rs`) were
//! flipped to `audited: true` (`bit_stable_on_same_hardware: true`) but
//! NEVER empirically verified. A later gate (`ledger::gate_precision`,
//! wired in a follow-up task) downgrades any `bit_stable_on_same_hardware`
//! claim lacking a passing entry in the git-checked-in
//! `docs/kernel-contracts/.fkc-verified-ledger.json` for the kernel's
//! exact `(kernel_revision_hash, backend, dtypes, claim)` tuple.
//!
//! This module RUNS the real, registered CPU kernel for every
//! `(FusedOpId, dtypes)` combination in [`TARGETS`] via [`CpuInvoker`]
//! (Task 4.5), `iters` times per probe, and only records a `"pass"`
//! ledger entry when [`verify_bit_stability`] actually observed
//! byte-identical repeat calls. **Never fabricates a pass**: an op that
//! can't be safely invoked (no probe recipe, an `Err` from the kernel, or
//! even a panic — caught via `catch_unwind` so one bad recipe can't take
//! down the whole harness) is recorded in the attempt log as
//! unverified/failed and contributes NO ledger entry.
//!
//! Hand-written families (`FLASH_ATTN`/`FLASH_ATTN_BACKWARD_*`/
//! `PAGED_ATTN`/`NF4_MATMUL`) are OUT OF SCOPE: their `BackendImpl.revision`
//! is `KernelRevisionHash::UNTRACKED` (not FKC-imported), so the 2026-07-03
//! flip never touched them and the ledger gate has nothing to check there.
//!
//! See `.superpowers/sdd/task-4.5b-report.md` for the full audit trail
//! (which ops verified, which didn't, and why).

use fuel_graph::registry::{FusedOpId, FusedOps, Reduction};
use fuel_ir::probe::BackendId;
use fuel_ir::DType;

use crate::fkc::verify::bit_stability::{
    fill_deterministic, verify_bit_stability, HostTensor, ProbeInputs, VerifyOutcome,
};
use crate::fkc::verify::invoker_cpu::CpuInvoker;
use crate::fkc::verify::ledger::LedgerRecord;
use crate::fused::{default_kernel_registry, BackendImpl};
use crate::kernel::{BindingEntry, MatmulM, OpParams};

/// Repeat-call count per probe for the `bit_stable_on_same_hardware`
/// check — `>= 16` per the task's floor.
const ITERS: usize = 16;

/// Which shape/`OpParams` recipe a given `FusedOpId` needs. One variant
/// per distinct wrapper calling convention in `dispatch.rs` (verified by
/// reading each `cpu_*_wrapper!` macro body — none of them read the
/// `layouts` argument at all; every shape fact a kernel needs travels
/// through `OpParams`, so a probe only needs the RIGHT ELEMENT COUNTS +
/// dtype bytes, not a real `Layout`).
#[derive(Debug, Clone, Copy)]
enum Family {
    /// SoftmaxLastDim forward: 1 input, `OpParams::SoftmaxLastDim`.
    SoftmaxFwd,
    /// RmsNormLastDim / LayerNormLastDim forward: 1 input, `OpParams::NormLastDim`.
    NormFwd,
    /// SoftmaxLastDimBackward: 2 inputs (y, g), `OpParams::SoftmaxLastDim`.
    SoftmaxBwd,
    /// LayerNorm/RmsNormLastDimBackward: 2 inputs (x, g), `OpParams::NormLastDim`.
    NormBwd,
    ReduceMaxToBwd,
    PowiBwd,
    FusedLinear,
    QMatMul,
    InplaceAffine,
    Fsce,
    Rope,
    Conv2D,
    ConvTranspose2D,
    CausalConv1d,
    SelectiveScan,
    SsdChunkScan,
}

/// A synthesized, safe, valid probe for one `(FusedOpId, dtype-tuple)`
/// CPU registration: real inputs the kernel can run on without crashing,
/// the `OpParams` it needs, and the output dtype/shape `CpuInvoker`
/// should allocate.
struct Probe {
    inputs: ProbeInputs,
    params: OpParams,
    out_dtype: DType,
    out_shape: Vec<usize>,
}

/// The Task 4.5b target set: every FKC-contract-sourced fused CPU op
/// flipped to `audited: true` on 2026-07-03. `name` is a diagnostic tag
/// only (the ledger's match key is `(kernel_revision_hash, backend,
/// dtypes, claim)` — see `ledger::VerificationLedger::has_pass`).
const TARGETS: &[(FusedOpId, Family, &str)] = &[
    (FusedOps::SOFTMAX_LAST_DIM, Family::SoftmaxFwd, "softmax_last_dim"),
    (FusedOps::RMS_NORM_LAST_DIM, Family::NormFwd, "rms_norm_last_dim"),
    (FusedOps::LAYER_NORM_LAST_DIM, Family::NormFwd, "layer_norm_last_dim"),
    (FusedOps::SOFTMAX_LAST_DIM_BACKWARD, Family::SoftmaxBwd, "softmax_last_dim_backward"),
    (FusedOps::LAYER_NORM_LAST_DIM_BACKWARD, Family::NormBwd, "layer_norm_last_dim_backward"),
    (FusedOps::RMS_NORM_LAST_DIM_BACKWARD, Family::NormBwd, "rms_norm_last_dim_backward"),
    (FusedOps::REDUCE_MAX_TO_BACKWARD, Family::ReduceMaxToBwd, "reduce_max_to_backward"),
    (FusedOps::POWI_BACKWARD, Family::PowiBwd, "powi_backward"),
    (FusedOps::FUSED_LINEAR, Family::FusedLinear, "fused_linear"),
    (FusedOps::QMATMUL, Family::QMatMul, "qmatmul"),
    (FusedOps::INPLACE_AFFINE, Family::InplaceAffine, "inplace_affine"),
    (FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY, Family::Fsce, "fused_softmax_cross_entropy"),
    (FusedOps::ROPE, Family::Rope, "rope"),
    (FusedOps::CONV2D, Family::Conv2D, "conv2d"),
    (FusedOps::CONV_TRANSPOSE2D, Family::ConvTranspose2D, "conv_transpose2d"),
    (FusedOps::CAUSAL_CONV1D, Family::CausalConv1d, "causal_conv1d"),
    (FusedOps::SELECTIVE_SCAN, Family::SelectiveScan, "selective_scan"),
    (FusedOps::SSD_CHUNK_SCAN, Family::SsdChunkScan, "ssd_chunk_scan"),
];

/// Encode `vals` (logical float values) into `dt`'s byte representation.
/// `None` for any dtype this harness doesn't know how to encode (never
/// guesses — an unencodable dtype means the caller must skip the probe).
fn to_bytes(dt: DType, vals: &[f32]) -> Option<Vec<u8>> {
    Some(match dt {
        DType::F32 => bytemuck::cast_slice(vals).to_vec(),
        DType::F64 => {
            let v: Vec<f64> = vals.iter().map(|&x| x as f64).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::BF16 => {
            let v: Vec<half::bf16> = vals.iter().map(|&x| half::bf16::from_f32(x)).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::F16 => {
            let v: Vec<half::f16> = vals.iter().map(|&x| half::f16::from_f32(x)).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        _ => return None,
    })
}

/// Build a `HostTensor` for `dt` from logical float values. `None` if
/// `dt` isn't an encodable float dtype (see [`to_bytes`]).
fn ht(dt: DType, shape: Vec<usize>, vals: &[f32]) -> Option<HostTensor> {
    Some(HostTensor { dtype: dt, shape, bytes: to_bytes(dt, vals)? })
}

/// Build a real, valid probe for `family` at the exact registered
/// `dtypes` tuple (`imp.dtypes` — the SAME slice the live registry binds,
/// not a guess). `None` ⇒ this harness genuinely could not synthesize a
/// probe (dtype-tuple shape didn't match what the family expects) — the
/// caller records this as unverified and skips; it never becomes a
/// fabricated ledger entry.
fn build_probe(family: Family, dtypes: &[DType], seed: u64) -> Option<Probe> {
    match family {
        Family::SoftmaxFwd | Family::NormFwd => {
            if dtypes.len() != 2 {
                return None;
            }
            let dt = dtypes[0];
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed))?;
            let params = match family {
                Family::SoftmaxFwd => {
                    OpParams::SoftmaxLastDim { outer_count: outer, last_dim: last }
                }
                _ => OpParams::NormLastDim { outer_count: outer, last_dim: last, eps: 1e-5 },
            };
            Some(Probe { inputs: vec![x], params, out_dtype: dt, out_shape: vec![outer * last] })
        }
        Family::SoftmaxBwd | Family::NormBwd => {
            if dtypes.len() != 3 {
                return None;
            }
            let dt = dtypes[0];
            let (outer, last) = (2usize, 4usize);
            let y = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed))?;
            let g = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed ^ 0x9E37_79B9))?;
            let params = match family {
                Family::SoftmaxBwd => {
                    OpParams::SoftmaxLastDim { outer_count: outer, last_dim: last }
                }
                _ => OpParams::NormLastDim { outer_count: outer, last_dim: last, eps: 1e-5 },
            };
            Some(Probe { inputs: vec![y, g], params, out_dtype: dt, out_shape: vec![outer * last] })
        }
        Family::ReduceMaxToBwd => {
            if dtypes.len() != 3 {
                return None;
            }
            let dt = dtypes[0];
            // Degenerate no-op reduction (input_shape == output_shape, both
            // rank-1 length-1): every output position maps to itself, so
            // this is safe regardless of the broadcast-alignment details of
            // `reduce_max_to_backward_impl` — the smallest shape that is
            // unconditionally a valid reduction.
            let x = ht(dt, vec![1], &fill_deterministic(1, seed))?;
            let up = ht(dt, vec![1], &fill_deterministic(1, seed ^ 0xA5A5_A5A5))?;
            Some(Probe {
                inputs: vec![x, up],
                params: OpParams::ReduceMaxToBackward { input_shape: vec![1], output_shape: vec![1] },
                out_dtype: dt,
                out_shape: vec![1],
            })
        }
        Family::PowiBwd => {
            if dtypes.len() != 3 {
                return None;
            }
            let dt = dtypes[0];
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let up = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0xDEAD_BEEF))?;
            Some(Probe {
                inputs: vec![x, up],
                params: OpParams::PowI { exp: 2 },
                out_dtype: dt,
                out_shape: vec![4],
            })
        }
        Family::FusedLinear => {
            if dtypes.len() != 4 {
                return None;
            }
            let dt = dtypes[0];
            let (m, n, k) = (2usize, 2usize, 2usize);
            let lhs = ht(dt, vec![m * k], &fill_deterministic(m * k, seed))?;
            let rhs = ht(dt, vec![k * n], &fill_deterministic(k * n, seed ^ 0x1234))?;
            let bias = ht(dt, vec![n], &fill_deterministic(n, seed ^ 0x5678))?;
            Some(Probe {
                inputs: vec![lhs, rhs, bias],
                params: OpParams::Matmul {
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                    m,
                    n,
                    k,
                    m_compute: MatmulM::All,
                },
                out_dtype: dt,
                out_shape: vec![m * n],
            })
        }
        Family::QMatMul => {
            if dtypes.len() != 3 || dtypes[0] != DType::F32 || dtypes[1] != DType::U32 || dtypes[2] != DType::F32 {
                return None;
            }
            // GGML Q4_0 block: 2-byte f16 scale + 16 packed-nibble bytes = 18
            // bytes/block. d=1.0, every nibble=9 (effective weight 9-8=1) —
            // the exact "unit weight" pattern verified safe+correct by
            // fuel-cpu-backend's own
            // `qmatmul_q4_0_f32_unit_weight_sums_activations` test.
            let block_size = 18usize;
            let mut w_bytes = vec![0u8; 2 * block_size];
            for block in 0..2 {
                let off = block * block_size;
                w_bytes[off..off + 2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                for i in 0..16 {
                    w_bytes[off + 2 + i] = 0x99;
                }
            }
            let act: Vec<f32> = (1..=32).map(|x| x as f32).collect();
            let act_bytes = bytemuck::cast_slice(&act).to_vec();
            let w_len = w_bytes.len() / 4;
            Some(Probe {
                inputs: vec![
                    HostTensor { dtype: DType::F32, shape: vec![32], bytes: act_bytes },
                    HostTensor { dtype: DType::U32, shape: vec![w_len], bytes: w_bytes },
                ],
                params: OpParams::QMatMul {
                    quant_type: fuel_graph::QuantType::Q4_0,
                    batch_count: 1,
                    m: 1,
                    n: 2,
                    k: 32,
                },
                out_dtype: DType::F32,
                out_shape: vec![2],
            })
        }
        Family::InplaceAffine => {
            if dtypes.len() != 2 {
                return None;
            }
            // 0 real inputs — in-place target is adopted by the executor
            // (`cpu_affine_inplace_wrapper!` requires `inputs.is_empty()`).
            Some(Probe {
                inputs: vec![],
                params: OpParams::Affine { mul: 2.0, add: 1.0 },
                out_dtype: dtypes[0],
                out_shape: vec![4],
            })
        }
        Family::Fsce => {
            if dtypes.len() != 3 || dtypes[1] != DType::I64 {
                return None;
            }
            let dt = dtypes[0];
            let (n_rows, vocab) = (2usize, 4usize);
            let logits = ht(dt, vec![n_rows * vocab], &fill_deterministic(n_rows * vocab, seed))?;
            let targets = HostTensor {
                dtype: DType::I64,
                shape: vec![n_rows],
                bytes: bytemuck::cast_slice(&[1i64, 3i64]).to_vec(),
            };
            Some(Probe {
                inputs: vec![logits, targets],
                params: OpParams::FusedSoftmaxCrossEntropy {
                    n_rows,
                    vocab,
                    reduction: Reduction::Mean,
                    ignore_index: -100,
                },
                out_dtype: DType::F32,
                out_shape: vec![1],
            })
        }
        Family::Rope => {
            if dtypes.len() != 4 {
                return None;
            }
            let dt = dtypes[0];
            let (outer, seq, head_dim) = (1usize, 1usize, 2usize);
            let x = ht(dt, vec![outer * seq * head_dim], &fill_deterministic(outer * seq * head_dim, seed))?;
            let cos = ht(dt, vec![seq * head_dim], &fill_deterministic(seq * head_dim, seed ^ 0x1111))?;
            let sin = ht(dt, vec![seq * head_dim], &fill_deterministic(seq * head_dim, seed ^ 0x2222))?;
            Some(Probe {
                inputs: vec![x, cos, sin],
                params: OpParams::Rope { outer_count: outer, seq, head_dim },
                out_dtype: dt,
                out_shape: vec![outer * seq * head_dim],
            })
        }
        Family::Conv2D | Family::ConvTranspose2D => {
            let with_bias = match dtypes.len() {
                3 => false,
                4 => true,
                _ => return None,
            };
            let dt = dtypes[0];
            let is_transpose = matches!(family, Family::ConvTranspose2D);
            let (x_shape, w_shape, out_shape): ([usize; 4], [usize; 4], [usize; 4]) = if is_transpose {
                // H_out = (H_in-1)*stride - 2*pad + dil*(Kh-1) + out_pad + 1
                //       = (2-1)*1 - 0 + 1*(2-1) + 0 + 1 = 3
                ([1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 3, 3])
            } else {
                // H_out = H_in + 2*pad - dil*(Kh-1) - 1)/stride + 1 = 3-2+1 = 2
                ([1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2])
            };
            let (stride, padding, dilation, groups) = ((1usize, 1usize), (0usize, 0usize), (1usize, 1usize), 1usize);
            let x_len: usize = x_shape.iter().product();
            let w_len: usize = w_shape.iter().product();
            let out_len: usize = out_shape.iter().product();
            let cout = out_shape[1];
            let x = ht(dt, vec![x_len], &fill_deterministic(x_len, seed))?;
            let w = ht(dt, vec![w_len], &fill_deterministic(w_len, seed ^ 0x3333))?;
            let mut inputs = vec![x, w];
            if with_bias {
                inputs.push(ht(dt, vec![cout], &fill_deterministic(cout, seed ^ 0x4444))?);
            }
            let params = if is_transpose {
                OpParams::ConvTranspose2D {
                    x_shape,
                    w_shape,
                    out_shape,
                    stride,
                    padding,
                    output_padding: (0, 0),
                    dilation,
                    groups,
                }
            } else {
                OpParams::Conv2D { x_shape, w_shape, out_shape, stride, padding, dilation, groups }
            };
            Some(Probe { inputs, params, out_dtype: dt, out_shape: vec![out_len] })
        }
        Family::CausalConv1d => {
            if dtypes.len() != 4 {
                return None;
            }
            let dt = dtypes[0];
            let (batch, channels, seq_in, seq_out, kernel) = (1usize, 1usize, 4usize, 2usize, 3usize);
            // Hand-verified values (fuel-cpu-backend
            // `causal_conv1d_f32_no_silu_basic`): x pre-padded, out[0]=2.1,
            // out[1]=5.1 — a real, known-sane invocation, not arbitrary bytes.
            let x = ht(dt, vec![batch * channels * seq_in], &[0.0, 0.0, 1.0, 2.0])?;
            let w = ht(dt, vec![channels * kernel], &[0.5, 1.0, 2.0])?;
            let b = ht(dt, vec![channels], &[0.1])?;
            Some(Probe {
                inputs: vec![x, w, b],
                params: OpParams::CausalConv1d { batch, channels, seq_in, seq_out, kernel, use_silu: false },
                out_dtype: dt,
                out_shape: vec![batch * channels * seq_out],
            })
        }
        Family::SelectiveScan => {
            if dtypes.len() != 6 {
                return None;
            }
            let dt = dtypes[0];
            // Hand-verified minimal case (fuel-cpu-backend
            // `selective_scan_f32_single_step_seqlen_1`): batch=seqlen=dim=
            // dstate=1, u=3,delta=1,a=-1,b=2,c=0.5 -> y=3.0.
            let u = ht(dt, vec![1], &[3.0])?;
            let delta = ht(dt, vec![1], &[1.0])?;
            let a = ht(dt, vec![1], &[-1.0])?;
            let b = ht(dt, vec![1], &[2.0])?;
            let c = ht(dt, vec![1], &[0.5])?;
            Some(Probe {
                inputs: vec![u, delta, a, b, c],
                params: OpParams::SelectiveScan { batch: 1, seqlen: 1, dim: 1, dstate: 1, delta_softplus: false },
                out_dtype: dt,
                out_shape: vec![2],
            })
        }
        Family::SsdChunkScan => {
            if dtypes.len() != 6 {
                return None;
            }
            let dt = dtypes[0];
            // Hand-verified minimal case (fuel-cpu-backend
            // `ssd_chunk_scan_f32_minimal`): batch=heads=head_dim=state_dim=
            // seqlen=chunk_size=1, x=3,dt=1,a=-1,b=2,c=0.5 -> y=3.0.
            let x = ht(dt, vec![1], &[3.0])?;
            let dtp = ht(dt, vec![1], &[1.0])?;
            let a = ht(dt, vec![1], &[-1.0])?;
            let b = ht(dt, vec![1], &[2.0])?;
            let c = ht(dt, vec![1], &[0.5])?;
            Some(Probe {
                inputs: vec![x, dtp, a, b, c],
                params: OpParams::SsdChunkScan {
                    batch: 1,
                    seqlen: 1,
                    heads: 1,
                    head_dim: 1,
                    state_dim: 1,
                    chunk_size: 1,
                },
                out_dtype: dt,
                out_shape: vec![2],
            })
        }
    }
}

/// Wrap a fused `BackendImpl` in a `BindingEntry` so it can be driven
/// through `CpuInvoker` (Task 4.5's invoker takes `&BindingEntry`, the
/// primitive-op binding shape — the same pattern `bit_stability.rs`'s
/// `dummy_entry()` test helper uses). `kernel_revision_hash` carries the
/// REAL hash the FKC importer threaded onto `imp.revision` — the SAME
/// hash the (future) gate will look up, so a ledger entry keyed on it is
/// exactly what the gate needs.
fn to_binding_entry(imp: &BackendImpl) -> BindingEntry {
    BindingEntry {
        kernel: imp.kernel,
        caps: imp.caps,
        precision: imp.precision,
        cost: crate::kernel::unknown_cost,
        kernel_source: "",
        is_generic: false,
        kernel_revision_hash: imp.revision.0,
        cost_expr: None,
    }
}

/// One outcome of attempting to verify one `(FusedOpId, dtypes)` CPU
/// registration — kept even for skips/failures so the harness (and the
/// report) can show exactly what did and didn't verify, never silently.
#[derive(Debug)]
pub struct SeedAttempt {
    pub op_name: &'static str,
    pub dtypes: Vec<DType>,
    pub kernel_revision_hash: u64,
    pub outcome: String,
}

/// Empirically verify every Task-4.5b-scoped CPU fused-op registration
/// and return the PASS records to seed into the ledger, plus a full
/// attempt log (including every skip/failure) for the report.
///
/// Never fabricates a pass: a record is only pushed when
/// `verify_bit_stability` actually observed `ITERS` byte-identical
/// repeat calls through the REAL registered `BackendImpl.kernel` fn
/// pointer, driven via the REAL `CpuInvoker` (Task 4.5) — not an
/// assertion that "CPU is deterministic". A kernel invocation that
/// errors OR panics (caught via `catch_unwind` so one bad probe recipe
/// can't take down the whole harness) is recorded as unverified and
/// contributes no ledger entry.
pub fn run_cpu_verification() -> (Vec<LedgerRecord>, Vec<SeedAttempt>) {
    let registry = default_kernel_registry();
    let mut records = Vec::new();
    let mut log = Vec::new();

    for &(id, family, name) in TARGETS {
        for (backend, imp) in registry.impls_for(id) {
            if *backend != BackendId::Cpu {
                continue;
            }
            let dtypes: Vec<DType> = imp.dtypes.to_vec();
            let rev = imp.revision.0;
            let seed = 0x2545_F491_4F6C_DD1D_u64 ^ (id.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let probe = match build_probe(family, imp.dtypes, seed) {
                Some(p) => p,
                None => {
                    log.push(SeedAttempt {
                        op_name: name,
                        dtypes,
                        kernel_revision_hash: rev,
                        outcome: "unverified: no probe recipe for this dtype tuple".to_string(),
                    });
                    continue;
                }
            };
            let entry = to_binding_entry(imp);
            let inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
                .with_params(probe.params.clone());
            let inputs = probe.inputs.clone();
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                verify_bit_stability(&inv, &entry, std::slice::from_ref(&inputs), ITERS)
            }));
            let outcome = match attempt {
                Ok(Ok(VerifyOutcome::Pass)) => {
                    records.push(LedgerRecord {
                        kernel_ref: name.to_string(),
                        backend: "Cpu".to_string(),
                        dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                        kernel_revision_hash: rev,
                        claim: "bit_stable_on_same_hardware".to_string(),
                        result: "pass".to_string(),
                        verified_at: verified_at_string(),
                        protocol_version: 1,
                        evidence: serde_json::json!({
                            "repeat_calls": ITERS,
                            "harness": "task-4.5b/seed_cpu_ledger",
                        }),
                    });
                    "pass".to_string()
                }
                Ok(Ok(VerifyOutcome::Fail { detail })) => format!("fail: {detail}"),
                Ok(Ok(VerifyOutcome::NoReference)) => "unverified: no probes".to_string(),
                Ok(Err(e)) => format!("unverified: invoke error {e:?}"),
                Err(_) => "unverified: kernel invocation panicked".to_string(),
            };
            log.push(SeedAttempt { op_name: name, dtypes, kernel_revision_hash: rev, outcome });
        }
    }
    (records, log)
}

/// `epoch:<unix seconds>` — a fixed, dependency-free timestamp (no
/// `chrono`, per house convention). Informational only (`LedgerRecord`
/// doesn't match on it).
fn verified_at_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Task 4.5b: empirically verify the CPU fused-op family and WRITE
    /// the resulting `"pass"` records to the git-checked-in verification
    /// ledger (`docs/kernel-contracts/.fkc-verified-ledger.json`).
    ///
    /// Run with `--nocapture` to see the full per-op attempt log
    /// (pass/fail/unverified + reason) — every op in [`TARGETS`] is
    /// accounted for, not just the ones that pass.
    #[test]
    fn seed_cpu_verified_ledger() {
        let (records, log) = run_cpu_verification();
        for attempt in &log {
            println!(
                "[task-4.5b] {} {:?} (rev={}): {}",
                attempt.op_name, attempt.dtypes, attempt.kernel_revision_hash, attempt.outcome,
            );
        }
        let passed = records.len();
        let failed_or_unverified = log.iter().filter(|a| a.outcome != "pass").count();
        println!(
            "[task-4.5b] {passed} passed, {failed_or_unverified} unverified/failed, {} total attempts",
            log.len(),
        );
        assert!(
            !records.is_empty(),
            "expected at least one CPU fused op to empirically verify bit-stable; got 0 — see log above",
        );

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/kernel-contracts/.fkc-verified-ledger.json");
        let json = serde_json::to_string_pretty(&records).expect("serialize ledger records");
        let mut f = std::fs::File::create(&path)
            .unwrap_or_else(|e| panic!("failed to open ledger at {path:?} for writing: {e}"));
        f.write_all(json.as_bytes()).expect("write ledger json");
        f.write_all(b"\n").expect("write trailing newline");
        println!("[task-4.5b] wrote {passed} pass record(s) to {}", path.display());
    }
}

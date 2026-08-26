// SPDX-License-Identifier: MIT OR Apache-2.0
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
use fuel_ir::DType;
use fuel_ir::probe::BackendId;

use crate::fkc::verify::bit_stability::{
    HostTensor, ProbeInputs, VerifyOutcome, fill_deterministic, verify_bit_stability,
};
use crate::fkc::verify::invoker_cpu::CpuInvoker;
use crate::fkc::verify::ledger::LedgerRecord;
use crate::fused::{BackendImpl, default_kernel_registry};
use crate::kernel::{BindingEntry, MatmulM, OpParams};

/// Repeat-call count per probe for the `bit_stable_on_same_hardware`
/// check — `>= 16` per the task's floor.
pub(crate) const ITERS: usize = 16;

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
    /// Bytes to PRE-FILL the output buffer with — mirrors
    /// `probe_recipes::Probe::out_seed` and exists for the same reason
    /// (GAP-222): an in-place op reads no inputs, so against the invoker's
    /// default zeroed buffer it is verified on zeros and the resulting pass
    /// measures nothing.
    ///
    /// This is a SECOND `Probe` type, deliberately not merged with the one in
    /// `probe_recipes` while GAP-220 is open — but the defect had to be fixed
    /// in both, and that is worth noticing: two types that must change
    /// together are already one type wearing two names. Evidence for GAP-220,
    /// recorded rather than acted on.
    out_seed: Option<Vec<u8>>,
}

/// The Task 4.5b target set: every FKC-contract-sourced fused CPU op
/// flipped to `audited: true` on 2026-07-03. `name` is a diagnostic tag
/// only (the ledger's match key is `(kernel_revision_hash, backend,
/// dtypes, claim)` — see `ledger::VerificationLedger::has_pass`).
const TARGETS: &[(FusedOpId, Family, &str)] = &[
    (
        FusedOps::SOFTMAX_LAST_DIM,
        Family::SoftmaxFwd,
        "softmax_last_dim",
    ),
    (
        FusedOps::RMS_NORM_LAST_DIM,
        Family::NormFwd,
        "rms_norm_last_dim",
    ),
    (
        FusedOps::LAYER_NORM_LAST_DIM,
        Family::NormFwd,
        "layer_norm_last_dim",
    ),
    (
        FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
        Family::SoftmaxBwd,
        "softmax_last_dim_backward",
    ),
    (
        FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
        Family::NormBwd,
        "layer_norm_last_dim_backward",
    ),
    (
        FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
        Family::NormBwd,
        "rms_norm_last_dim_backward",
    ),
    (
        FusedOps::REDUCE_MAX_TO_BACKWARD,
        Family::ReduceMaxToBwd,
        "reduce_max_to_backward",
    ),
    (FusedOps::POWI_BACKWARD, Family::PowiBwd, "powi_backward"),
    (FusedOps::FUSED_LINEAR, Family::FusedLinear, "fused_linear"),
    (FusedOps::QMATMUL, Family::QMatMul, "qmatmul"),
    (
        FusedOps::INPLACE_AFFINE,
        Family::InplaceAffine,
        "inplace_affine",
    ),
    (
        FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
        Family::Fsce,
        "fused_softmax_cross_entropy",
    ),
    (FusedOps::ROPE, Family::Rope, "rope"),
    (FusedOps::CONV2D, Family::Conv2D, "conv2d"),
    (
        FusedOps::CONV_TRANSPOSE2D,
        Family::ConvTranspose2D,
        "conv_transpose2d",
    ),
    (
        FusedOps::CAUSAL_CONV1D,
        Family::CausalConv1d,
        "causal_conv1d",
    ),
    (
        FusedOps::SELECTIVE_SCAN,
        Family::SelectiveScan,
        "selective_scan",
    ),
    (
        FusedOps::SSD_CHUNK_SCAN,
        Family::SsdChunkScan,
        "ssd_chunk_scan",
    ),
];

/// Encode `vals` (logical float values) into `dt`'s byte representation.
/// `None` for any dtype this harness doesn't know how to encode (never
/// guesses — an unencodable dtype means the caller must skip the probe).
///
/// `pub(crate)` (widened from private) + re-exported via
/// `fkc::verify::to_bytes` so [`crate::jit_ingest_probe`]'s
/// `probe_from_operands` can reuse the exact same encode logic instead of
/// duplicating it (this module is unconditional, not `cuda`-gated, so it's
/// reachable from a `jit`-only build).
///
/// **The float-only domain is LOAD-BEARING — do not widen it, and do not
/// unify it with `probe_recipes::to_bytes`.** They have the same signature
/// and deliberately different domains, because they answer different
/// questions. This one is the JIT-ingest encoder: `probe_from_operands`
/// returning `None` is how an unencodable operand becomes NO PROBE rather
/// than a probe over invented bytes, and
/// `jit_ingest_probe::probe_from_operands_rejects_an_unencodable_integer_operand`
/// asserts exactly that for `I32`. Widening this to cover integers would
/// silently turn that rejection into an acceptance.
///
/// The sibling in `probe_recipes` is the ledger-verification encoder, where
/// the requirement is the opposite: it must cover every dtype any backend's
/// probes fan over, or earned ledger records stop being re-earnable. Commit
/// `23785514` collapsed the two by repointing that module's `ht` here, and
/// orphaned 228 of 530 Vulkan records behind a green build. **Two encoders
/// whose narrowness and whose breadth are each independently asserted is the
/// correct shape; one shared encoder cannot satisfy both tests.**
pub(crate) fn to_bytes(dt: DType, vals: &[f32]) -> Option<Vec<u8>> {
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
    Some(HostTensor {
        dtype: dt,
        shape,
        bytes: to_bytes(dt, vals)?,
    })
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
            let x = ht(
                dt,
                vec![outer * last],
                &fill_deterministic(outer * last, seed),
            )?;
            let params = match family {
                Family::SoftmaxFwd => OpParams::SoftmaxLastDim {
                    outer_count: outer,
                    last_dim: last,
                },
                _ => OpParams::NormLastDim {
                    outer_count: outer,
                    last_dim: last,
                    eps: 1e-5,
                },
            };
            Some(Probe {
                inputs: vec![x],
                params,
                out_dtype: dt,
                out_shape: vec![outer * last],
                out_seed: None,
            })
        }
        Family::SoftmaxBwd | Family::NormBwd => {
            if dtypes.len() != 3 {
                return None;
            }
            let dt = dtypes[0];
            let (outer, last) = (2usize, 4usize);
            let y = ht(
                dt,
                vec![outer * last],
                &fill_deterministic(outer * last, seed),
            )?;
            let g = ht(
                dt,
                vec![outer * last],
                &fill_deterministic(outer * last, seed ^ 0x9E37_79B9),
            )?;
            let params = match family {
                Family::SoftmaxBwd => OpParams::SoftmaxLastDim {
                    outer_count: outer,
                    last_dim: last,
                },
                _ => OpParams::NormLastDim {
                    outer_count: outer,
                    last_dim: last,
                    eps: 1e-5,
                },
            };
            Some(Probe {
                inputs: vec![y, g],
                params,
                out_dtype: dt,
                out_shape: vec![outer * last],
                out_seed: None,
            })
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
                params: OpParams::ReduceMaxToBackward {
                    input_shape: vec![1],
                    output_shape: vec![1],
                },
                out_dtype: dt,
                out_shape: vec![1],
                out_seed: None,
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
                out_seed: None,
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
                out_seed: None,
            })
        }
        Family::QMatMul => {
            if dtypes.len() != 3
                || dtypes[0] != DType::F32
                || dtypes[1] != DType::U32
                || dtypes[2] != DType::F32
            {
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
                    HostTensor {
                        dtype: DType::F32,
                        shape: vec![32],
                        bytes: act_bytes,
                    },
                    HostTensor {
                        dtype: DType::U32,
                        shape: vec![w_len],
                        bytes: w_bytes,
                    },
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
                out_seed: None,
            })
        }
        Family::InplaceAffine => {
            if dtypes.len() != 2 {
                return None;
            }
            // 0 real inputs — in-place target is adopted by the executor
            // (`cpu_affine_inplace_wrapper!` requires `inputs.is_empty()`), so
            // the probe data goes into the OUTPUT buffer. Without the seed the
            // kernel runs on zeros and `affine(0) = 1` everywhere: bit-stable,
            // and evidence of one input value (GAP-222). The four
            // `inplace_affine` rows in the checked-in ledger were earned that
            // way; this is what re-earns them rather than grandfathering them.
            Some(Probe {
                inputs: vec![],
                params: OpParams::Affine { mul: 2.0, add: 1.0 },
                out_dtype: dtypes[0],
                out_shape: vec![4],
                out_seed: Some(to_bytes(dtypes[0], &fill_deterministic(4, seed))?),
            })
        }
        Family::Fsce => {
            if dtypes.len() != 3 || dtypes[1] != DType::I64 {
                return None;
            }
            let dt = dtypes[0];
            let (n_rows, vocab) = (2usize, 4usize);
            let logits = ht(
                dt,
                vec![n_rows * vocab],
                &fill_deterministic(n_rows * vocab, seed),
            )?;
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
                out_seed: None,
            })
        }
        Family::Rope => {
            if dtypes.len() != 4 {
                return None;
            }
            let dt = dtypes[0];
            let (outer, seq, head_dim) = (1usize, 1usize, 2usize);
            let x = ht(
                dt,
                vec![outer * seq * head_dim],
                &fill_deterministic(outer * seq * head_dim, seed),
            )?;
            let cos = ht(
                dt,
                vec![seq * head_dim],
                &fill_deterministic(seq * head_dim, seed ^ 0x1111),
            )?;
            let sin = ht(
                dt,
                vec![seq * head_dim],
                &fill_deterministic(seq * head_dim, seed ^ 0x2222),
            )?;
            Some(Probe {
                inputs: vec![x, cos, sin],
                params: OpParams::Rope {
                    outer_count: outer,
                    seq,
                    head_dim,
                },
                out_dtype: dt,
                out_shape: vec![outer * seq * head_dim],
                out_seed: None,
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
            let (x_shape, w_shape, out_shape): ([usize; 4], [usize; 4], [usize; 4]) =
                if is_transpose {
                    // H_out = (H_in-1)*stride - 2*pad + dil*(Kh-1) + out_pad + 1
                    //       = (2-1)*1 - 0 + 1*(2-1) + 0 + 1 = 3
                    ([1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 3, 3])
                } else {
                    // H_out = H_in + 2*pad - dil*(Kh-1) - 1)/stride + 1 = 3-2+1 = 2
                    ([1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2])
                };
            let (stride, padding, dilation, groups) =
                ((1usize, 1usize), (0usize, 0usize), (1usize, 1usize), 1usize);
            let x_len: usize = x_shape.iter().product();
            let w_len: usize = w_shape.iter().product();
            let out_len: usize = out_shape.iter().product();
            let cout = out_shape[1];
            let x = ht(dt, vec![x_len], &fill_deterministic(x_len, seed))?;
            let w = ht(dt, vec![w_len], &fill_deterministic(w_len, seed ^ 0x3333))?;
            let mut inputs = vec![x, w];
            if with_bias {
                inputs.push(ht(
                    dt,
                    vec![cout],
                    &fill_deterministic(cout, seed ^ 0x4444),
                )?);
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
                OpParams::Conv2D {
                    x_shape,
                    w_shape,
                    out_shape,
                    stride,
                    padding,
                    dilation,
                    groups,
                }
            };
            Some(Probe {
                inputs,
                params,
                out_dtype: dt,
                out_shape: vec![out_len],
                out_seed: None,
            })
        }
        Family::CausalConv1d => {
            if dtypes.len() != 4 {
                return None;
            }
            let dt = dtypes[0];
            let (batch, channels, seq_in, seq_out, kernel) =
                (1usize, 1usize, 4usize, 2usize, 3usize);
            // Hand-verified values (fuel-cpu-backend
            // `causal_conv1d_f32_no_silu_basic`): x pre-padded, out[0]=2.1,
            // out[1]=5.1 — a real, known-sane invocation, not arbitrary bytes.
            let x = ht(dt, vec![batch * channels * seq_in], &[0.0, 0.0, 1.0, 2.0])?;
            let w = ht(dt, vec![channels * kernel], &[0.5, 1.0, 2.0])?;
            let b = ht(dt, vec![channels], &[0.1])?;
            Some(Probe {
                inputs: vec![x, w, b],
                params: OpParams::CausalConv1d {
                    batch,
                    channels,
                    seq_in,
                    seq_out,
                    kernel,
                    use_silu: false,
                },
                out_dtype: dt,
                out_shape: vec![batch * channels * seq_out],
                out_seed: None,
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
                params: OpParams::SelectiveScan {
                    batch: 1,
                    seqlen: 1,
                    dim: 1,
                    dstate: 1,
                    delta_softplus: false,
                },
                out_dtype: dt,
                out_shape: vec![2],
                out_seed: None,
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
                out_seed: None,
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
    /// `String`, not `&'static str`: the primitive pass names its subject with
    /// `format!("{op:?}")` off a live `OpKind`, which has no static name.
    pub op_name: String,
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
            let seed =
                0x2545_F491_4F6C_DD1D_u64 ^ (id.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let probe = match build_probe(family, imp.dtypes, seed) {
                Some(p) => p,
                None => {
                    log.push(SeedAttempt {
                        op_name: name.to_string(),
                        dtypes,
                        kernel_revision_hash: rev,
                        outcome: "unverified: no probe recipe for this dtype tuple".to_string(),
                    });
                    continue;
                }
            };
            let entry = to_binding_entry(imp);
            let mut inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
                .with_params(probe.params.clone());
            if let Some(seed_bytes) = probe.out_seed.clone() {
                inv = inv.with_seeded_output(seed_bytes);
            }
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
                        // `output_seeded` records WHAT THE CLAIM WAS EARNED
                        // AGAINST, which is the whole lesson of GAP-222: a
                        // pass earned on a zeroed in-place target is
                        // byte-identical in every field to one earned on real
                        // data, so the distinction was invisible in the
                        // ledger, in every count, and to `gate_precision`.
                        // Recording it makes that class visible for the first
                        // time. `false` here is not a defect — it is the
                        // correct value for an op that reads its inputs.
                        evidence: serde_json::json!({
                            "repeat_calls": ITERS,
                            "harness": "task-4.5b/seed_cpu_ledger",
                            "output_seeded": probe.out_seed.is_some(),
                        }),
                    });
                    "pass".to_string()
                }
                Ok(Ok(VerifyOutcome::Fail { detail })) => format!("fail: {detail}"),
                Ok(Ok(VerifyOutcome::NoReference)) => "unverified: no probes".to_string(),
                Ok(Err(e)) => format!("unverified: invoke error {e:?}"),
                Err(_) => "unverified: kernel invocation panicked".to_string(),
            };
            log.push(SeedAttempt {
                op_name: name.to_string(),
                dtypes,
                kernel_revision_hash: rev,
                outcome,
            });
        }
    }
    (records, log)
}

/// Empirically verify every CPU **PRIMITIVE** registration and return the PASS
/// records to seed into the ledger, plus a full attempt log.
///
/// This is the half [`run_cpu_verification`] does not cover. That one sweeps
/// the FUSED registry (`default_kernel_registry().impls_for(FusedOpId)`); this
/// one sweeps the primitive `KernelBindingTable` built by the PRODUCTION
/// `register_cpu_kernels`, so what it verifies is what dispatch actually binds
/// — not a parallel list that can drift out of agreement with it.
///
/// **Why this exists: `fill_unset_cpu_precision` asserts a claim nobody
/// measured.** It upgrades every UNAUDITED CPU entry to
/// `PRIMITIVE_DETERMINISTIC_CPU` — i.e. writes `bit_stable_on_same_hardware`
/// onto ~335 registrations wholesale. GAP-077 measured the consequence: **636
/// CPU entries bit-stable, 636 owed to the fill, ZERO earned from a contract.**
/// Retiring that fill requires the records to exist first, which is what this
/// produces. Until they do, retiring it would leave the entries UNAUDITED and
/// fail the step-7b coverage lint — so the seeding lands before, or with, the
/// retirement, never after it.
///
/// **Never fabricates a pass.** A record is pushed only when
/// `verify_bit_stability` actually observed `ITERS` byte-identical results. An
/// op with no probe recipe, an invoke error, or a panic contributes NO record
/// and appears in the log as unverified with its reason. The log is
/// exhaustive: one entry per CPU `BindingEntry` examined, so pass + fail +
/// unverified always reconciles against the table's own count.
pub fn run_cpu_primitive_verification() -> (Vec<LedgerRecord>, Vec<SeedAttempt>) {
    use crate::fkc::verify::probe_recipes::{build_primitive_probe, probe_seed};

    let mut table = crate::kernel::KernelBindingTable::new();
    crate::dispatch::register_cpu_kernels(&mut table);

    let mut records = Vec::new();
    let mut log = Vec::new();

    for (op, dtypes, backend, entry) in table.iter_entries() {
        if backend != BackendId::Cpu {
            continue;
        }
        let dtypes_vec = dtypes.to_vec();
        let rev = entry.kernel_revision_hash;
        let op_name = format!("{op:?}");

        let probe = match build_primitive_probe(op, dtypes, probe_seed(op, dtypes)) {
            Some(p) => p,
            None => {
                log.push(SeedAttempt {
                    op_name,
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "unverified: no probe recipe for this op/dtype tuple".to_string(),
                });
                continue;
            }
        };

        let mut inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        // An in-place op reads no inputs — its target arrives as `outputs[0]`,
        // so the probe data has to go into the output buffer or the kernel is
        // verified against zeros (GAP-222).
        if let Some(seed_bytes) = probe.out_seed.clone() {
            inv = inv.with_seeded_output(seed_bytes);
        }
        let inputs = probe.inputs.clone();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_bit_stability(&inv, entry, std::slice::from_ref(&inputs), ITERS)
        }));
        let outcome = match attempt {
            Ok(Ok(VerifyOutcome::Pass)) => {
                records.push(LedgerRecord {
                    kernel_ref: op_name.clone(),
                    backend: "Cpu".to_string(),
                    dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                    kernel_revision_hash: rev,
                    claim: "bit_stable_on_same_hardware".to_string(),
                    result: "pass".to_string(),
                    verified_at: verified_at_string(),
                    protocol_version: 1,
                    evidence: serde_json::json!({
                        "repeat_calls": ITERS,
                        "harness": "gap-207/seed_cpu_ledger::primitive",
                        // See the note on the fused pass: this records what
                        // the claim was earned AGAINST (GAP-222).
                        "output_seeded": probe.out_seed.is_some(),
                    }),
                });
                "pass".to_string()
            }
            Ok(Ok(VerifyOutcome::Fail { detail })) => format!("fail: {detail}"),
            Ok(Ok(VerifyOutcome::NoReference)) => "unverified: no probes".to_string(),
            Ok(Err(e)) => format!("unverified: invoke error {e:?}"),
            Err(_) => "unverified: kernel invocation panicked".to_string(),
        };
        log.push(SeedAttempt {
            op_name,
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
    }

    (records, log)
}

/// Empirically verify `max_ulp: 0` for every CPU primitive registration that
/// has BOTH a probe recipe and an exact in-process reference.
///
/// **This is the claim the 84 remaining import downgrades are blocked on.**
/// `gate_precision` collapses a whole guarantee if any declared claim is
/// unbacked, so an entry declaring `bit_stable_on_same_hardware` AND
/// `max_ulp` stays fully UNAUDITED until both are earned — which is why
/// earning bit-stability for 572 registrations changed no guarantee for the
/// entries that also declare a bound.
///
/// **The bound is hard-coded to `MaxUlp(0)` and that is deliberate.** Every
/// `max_ulp` line in the contracts owning this population declares `0`, with
/// inline reasons ("exact: f32 is a strict subset of f64"). Reading the
/// declared value back out of the contract is not possible here —
/// `import_bundle_str` gates before returning, so the lowered value is gone —
/// so this verifies the STRICTEST bound. A kernel that passes `MaxUlp(0)`
/// satisfies any larger declared bound, so the record is sound for the
/// population; it would NOT be sound to record `max_ulp` from a looser check.
///
/// **Never fabricates.** An op with no exact reference contributes no record
/// and appears in the log; so does a probe that errors. A `fail` is recorded
/// as a fail — and a disagreement here is a FINDING about the kernel or about
/// the reference, not a harness defect to be tuned away.
pub fn run_cpu_max_ulp_verification() -> (Vec<LedgerRecord>, Vec<SeedAttempt>) {
    run_cpu_max_ulp_verification_inner(None)
}

/// [`run_cpu_max_ulp_verification`], optionally corrupting the reference for
/// ONE op so the attachment of references to ops can be tested.
///
/// **Attachment is the property a record count cannot see.** A reference
/// bound to the wrong op yields exactly the right number of passes — the same
/// blindness that let GAP-228(a)'s generator misattach a clause while its
/// delta held exactly. Poisoning one family and checking the failures are
/// exactly that family's registrations is a control over a DIFFERENT
/// construct, which is the only kind that catches this.
pub(crate) fn run_cpu_max_ulp_verification_inner(
    #[allow(unused_variables)] poison_op: Option<fuel_ir::dispatch::OpKind>,
) -> (Vec<LedgerRecord>, Vec<SeedAttempt>) {
    use crate::fkc::verify::exact_ref::{ExactRefInvoker, has_exact_reference};
    use crate::fkc::verify::probe_recipes::{build_primitive_probe, probe_seed};
    use crate::fkc::verify::ulp::{Bound, verify_precision_bound};

    let mut table = crate::kernel::KernelBindingTable::new();
    crate::dispatch::register_cpu_kernels(&mut table);

    let mut records = Vec::new();
    let mut log = Vec::new();

    for (op, dtypes, backend, entry) in table.iter_entries() {
        if backend != BackendId::Cpu {
            continue;
        }
        let dtypes_vec = dtypes.to_vec();
        let rev = entry.kernel_revision_hash;
        let op_name = format!("{op:?}");

        if !has_exact_reference(op, dtypes) {
            log.push(SeedAttempt {
                op_name,
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "unverified: no exact in-process reference for this op/dtype tuple"
                    .to_string(),
            });
            continue;
        }
        let probe = match build_primitive_probe(op, dtypes, probe_seed(op, dtypes)) {
            Some(p) => p,
            None => {
                log.push(SeedAttempt {
                    op_name,
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "unverified: no probe recipe for this op/dtype tuple".to_string(),
                });
                continue;
            }
        };

        let cand = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let refr = ExactRefInvoker {
            op,
            out_dtype: probe.out_dtype,
            out_shape: probe.out_shape.clone(),
            params: probe.params.clone(),
            #[cfg(test)]
            poison: poison_op == Some(op),
        };
        let inputs = probe.inputs.clone();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_precision_bound(
                &cand,
                &refr,
                entry,
                std::slice::from_ref(&inputs),
                Bound::MaxUlp(0),
            )
        }));
        let outcome = match attempt {
            Ok(Ok(VerifyOutcome::Pass)) => {
                records.push(LedgerRecord {
                    kernel_ref: op_name.clone(),
                    backend: "Cpu".to_string(),
                    dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                    kernel_revision_hash: rev,
                    claim: "max_ulp".to_string(),
                    result: "pass".to_string(),
                    verified_at: verified_at_string(),
                    protocol_version: 1,
                    evidence: serde_json::json!({
                        "bound": "MaxUlp(0)",
                        "reference": "exact in-process (fkc::verify::exact_ref)",
                        "harness": "gap-225/seed_cpu_ledger::max_ulp",
                        // What the claim was earned AGAINST, for the same
                        // reason `output_seeded` exists (GAP-222): a record
                        // earned against a differential and one earned against
                        // a truth reference are otherwise identical in every
                        // field, and only the second is a `max_ulp` bound.
                        "reference_kind": "truth",
                    }),
                });
                "pass".to_string()
            }
            Ok(Ok(VerifyOutcome::Fail { detail })) => format!("fail: {detail}"),
            Ok(Ok(VerifyOutcome::NoReference)) => "unverified: no probes".to_string(),
            Ok(Err(e)) => format!("unverified: invoke error {e:?}"),
            Err(_) => "unverified: kernel invocation panicked".to_string(),
        };
        log.push(SeedAttempt {
            op_name,
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
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

    /// **Every exact reference is attached to the op it claims to be about —
    /// tested per family, because no COUNT can see attachment.**
    ///
    /// GAP-228(a)'s generator misattached an evidence clause and its
    /// justifying delta was exactly right anyway: attaching something to the
    /// wrong subject moves nothing between buckets. **A reference bound to
    /// the wrong `(op, dtypes)` has the same shape** — it would yield exactly
    /// the right number of earned `max_ulp` records while the evidence sat on
    /// a kernel it was never about.
    ///
    /// So this poisons ONE family's reference at a time and asserts the
    /// failures are EXACTLY that family's registrations: **no fewer** (the
    /// reference is reached for every one of them, so none is silently
    /// skipped) and **no more** (it is not also standing in for another op).
    ///
    /// Applied retroactively to the records already landed, which is the
    /// point — the 174 `max_ulp` passes were justified by a count that could
    /// not have detected this.
    #[test]
    fn every_exact_reference_is_attached_to_the_op_it_claims() {
        use fuel_ir::dispatch::OpKind;

        /// How many CPU families production offers an exact reference for.
        /// Pinned so that a family appearing or disappearing is a red gate
        /// rather than a quietly shorter loop.
        const EXACT_REFERENCE_FAMILIES: usize = 9;

        // Baseline: which registrations pass with no poison.
        let (_, clean) = run_cpu_max_ulp_verification_inner(None);
        let passing: Vec<(String, Vec<DType>)> = clean
            .iter()
            .filter(|a| a.outcome == "pass")
            .map(|a| (a.op_name.clone(), a.dtypes.clone()))
            .collect();
        assert!(
            !passing.is_empty(),
            "no registration passes at all — this control would then be vacuous for \
             every family below"
        );

        // THE FAMILY LIST IS DERIVED FROM PRODUCTION, NOT HARDCODED.
        //
        // It used to be a literal list of nine `OpKind`s that happened to equal
        // `has_exact_reference`'s nine arms — and NOTHING ENFORCED THAT
        // EQUALITY IN EITHER DIRECTION. A tenth op added to production would
        // never be poisoned here; deleting one from the list would shrink both
        // sides of `failed.len() == expected.len()` together and still pass.
        // Both losses are silent, and the second is the one a reviewer would
        // wave through, because a shorter list reads as a smaller job.
        //
        // Deriving the list means an op added to `has_exact_reference` comes
        // under this control the moment it exists, and an op removed from it
        // moves the pinned count instead of vanishing.
        let mut fam_table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut fam_table);
        let mut families: Vec<OpKind> = Vec::new();
        for (op, dtypes, backend, _entry) in fam_table.iter_entries() {
            if backend == BackendId::Cpu
                && crate::fkc::verify::exact_ref::has_exact_reference(op, dtypes)
                && !families.contains(&op)
            {
                families.push(op);
            }
        }
        families.sort_by_key(|o| format!("{o:?}"));
        assert_eq!(
            families.len(),
            EXACT_REFERENCE_FAMILIES,
            "production offers an exact reference for {} CPU families, not the pinned              {EXACT_REFERENCE_FAMILIES}: {families:?}. If a family was ADDED, it is now              under this control and the pin moves with it; if one was REMOVED, say why              its records no longer need an attachment check.",
            families.len()
        );

        let mut families_tested = 0usize;
        let mut not_testable: Vec<String> = Vec::new();
        for op in families {
            let name = format!("{op:?}");
            let expected: Vec<&(String, Vec<DType>)> =
                passing.iter().filter(|(n, _)| *n == name).collect();
            if expected.is_empty() {
                // A family with no passing registration cannot be tested for
                // attachment. Saying so beats skipping silently — but a
                // PRINTLN is not saying so, it is a skip with a receipt nobody
                // reads. The count below is what makes it visible.
                println!("[attach] {name}: no passing registration — not testable");
                not_testable.push(name);
                continue;
            }

            let (_, poisoned) = run_cpu_max_ulp_verification_inner(Some(op));
            let failed: Vec<(String, Vec<DType>)> = poisoned
                .iter()
                .filter(|a| a.outcome.starts_with("fail:"))
                .map(|a| (a.op_name.clone(), a.dtypes.clone()))
                .collect();

            let wrong_op: Vec<&String> = failed
                .iter()
                .map(|(n, _)| n)
                .filter(|n| **n != name)
                .collect();
            assert!(
                wrong_op.is_empty(),
                "poisoning {name}'s reference also failed {wrong_op:?} — that reference \
                 is standing in for an op it does not claim to be about, and the record \
                 count cannot see it"
            );
            assert_eq!(
                failed.len(),
                expected.len(),
                "poisoning {name}'s reference failed {} of its {} passing registrations. \
                 FEWER means some registration does not actually reach this reference — \
                 its pass was earned from something else.",
                failed.len(),
                expected.len()
            );
            families_tested += 1;
            println!(
                "[attach] {name}: {} registration(s), all and only",
                failed.len()
            );
        }

        // ⚠️ THE SKIP BRANCH ABOVE IS A SECOND WAY THIS CONTROL SHRINKS, AND
        // IT NEEDS NO SOURCE EDIT AT ALL. A family that stops passing for any
        // reason — a probe that stops building, a kernel revision that moves —
        // is skipped with a println, and every assertion in the loop still
        // holds because the loop simply has less to check. Deleting a line
        // from a list is at least visible in a diff; this one is invisible in
        // the source AND in the result.
        assert_eq!(
            families_tested, EXACT_REFERENCE_FAMILIES,
            "only {families_tested} of {EXACT_REFERENCE_FAMILIES} exact-reference              families were actually poisoned; {not_testable:?} had no passing              registration to poison. Production claims an exact reference for them, so              a skip here means their records rest on an attachment nothing checked."
        );
    }

    /// Classify one sweep outcome into exactly one named bucket.
    ///
    /// **THIS REPLACES A REMAINDER, AND THE REMAINDER WAS THREE DEFECTS
    /// STACKED IN ONE LINE.** Both sweeps used to compute
    /// `let other = log.len() - passes - no_ref - no_probe - failed;` and then
    /// assert `passes + no_ref + no_probe + failed + other == log.len()` under
    /// the message *"the outcome buckets do not partition the log"*.
    ///
    /// 1. **The assertion cannot fail.** It is `x + (L - x) == L`, true for
    ///    every possible classification, including one that classifies nothing
    ///    at all. It reads as a partition check and is an identity.
    /// 2. **The catch-all wore a specific cause's name.** `other` was printed
    ///    as `unverified (invoke error / panic)`, so ANY outcome spelling the
    ///    predicates missed was reported to the reader as an invoke error — a
    ///    false diagnosis manufactured by a bucket defined as "everything
    ///    else".
    /// 3. **One predicate collided.** `starts_with("unverified: no probe")`
    ///    matches BOTH `"unverified: no probe recipe ..."` (a gap in this
    ///    harness — nobody wrote a recipe) AND `"unverified: no probes"`
    ///    (`VerifyOutcome::NoReference` — the verifier had nothing to compare
    ///    against). Two different findings, merged, then reported under the
    ///    name of the first.
    ///
    /// Returning `Option` makes the miss loud: an unrecognised outcome is
    /// `None` and the callers assert there are none. **A new outcome spelling
    /// must become a RED GATE, not a silent increment of a bucket that means
    /// something else.**
    fn outcome_bucket(outcome: &str) -> Option<&'static str> {
        if outcome == "pass" {
            Some("pass")
        } else if outcome.starts_with("fail:") {
            Some("fail")
        } else if outcome.starts_with("unverified: no probe recipe") {
            Some("no_recipe")
        } else if outcome == "unverified: no probes" {
            Some("no_probes")
        } else if outcome.starts_with("unverified: no exact in-process reference") {
            Some("no_reference")
        } else if outcome.starts_with("unverified: invoke error") {
            Some("invoke_error")
        } else if outcome == "unverified: kernel invocation panicked" {
            Some("panicked")
        } else {
            None
        }
    }

    /// **Every outcome this harness can emit lands in its own bucket — the two
    /// that used to collide most of all.**
    ///
    /// The vocabulary is written out rather than derived, because the point is
    /// to pin the SPELLINGS: these strings are produced at seven sites, and a
    /// classifier is only as good as its agreement with them. Add an arm there
    /// without adding it here and the sweeps go red on `unclassified`, naming
    /// the spelling that is new.
    #[test]
    fn the_outcome_vocabulary_is_classified_into_distinct_buckets() {
        let vocabulary: &[(&str, &str)] = &[
            ("pass", "pass"),
            ("fail: byte 3 differs", "fail"),
            (
                "unverified: no probe recipe for this dtype tuple",
                "no_recipe",
            ),
            (
                "unverified: no probe recipe for this op/dtype tuple",
                "no_recipe",
            ),
            // THE COLLISION. This shares the prefix "unverified: no probe"
            // with the two above and means something entirely different: the
            // VERIFIER had no reference, rather than this harness having no
            // recipe.
            ("unverified: no probes", "no_probes"),
            (
                "unverified: no exact in-process reference for this op/dtype tuple",
                "no_reference",
            ),
            ("unverified: invoke error Shape", "invoke_error"),
            ("unverified: kernel invocation panicked", "panicked"),
        ];

        for (spelling, expected) in vocabulary {
            assert_eq!(
                outcome_bucket(spelling),
                Some(*expected),
                "`{spelling}` must classify as `{expected}`"
            );
        }

        // The collision asserted directly, not left implicit in the table:
        // these two must never share a bucket again.
        assert_ne!(
            outcome_bucket("unverified: no probe recipe for this op/dtype tuple"),
            outcome_bucket("unverified: no probes"),
            "a harness gap (no recipe written) and a verifier result (no reference \
             available) are different findings and must not merge — merging them is \
             exactly what a shared `starts_with` prefix did"
        );

        // An unrecognised spelling must be refused rather than absorbed. Built
        // at runtime so the control cannot be mistaken for vocabulary.
        assert_eq!(
            outcome_bucket(&format!("unverified: {} something new", 1 + 1)),
            None,
            "an unknown outcome must return None so the sweeps can name it, instead of \
             falling into a bucket that means something else"
        );
    }

    /// **The conv probes actually exercise their kernels — the output is not
    /// a constant.**
    ///
    /// ⚠️ **Bit-stability has no attachment control of the kind
    /// `every_exact_reference_is_attached_to_the_op_it_claims` provides, and
    /// saying so is more useful than pretending the control transfers.** That
    /// one poisons a REFERENCE; a bit-stability check compares a kernel
    /// against ITSELF across repeats, so there is no second implementation to
    /// misattach and perturbing the probe just yields a different
    /// bit-stable answer.
    ///
    /// Two things do stand in for it. First, a probe built for the wrong op
    /// is REJECTED by the kernel rather than silently passing — that is how
    /// `WriteSliceDoff` was caught twice (wrong operand count, then wrong
    /// offset dtype), and the sweep reports 0 invoke errors. Second, and this
    /// is what this test adds: **a kernel that ignored its inputs entirely
    /// would be perfectly bit-stable.** The all-zeros integer probes were
    /// exactly that, and nothing in a pass count could see it.
    ///
    /// So: invoke each conv probe once and require the output to vary. A
    /// constant output means either the probe is degenerate or the kernel is
    /// not reading it, and both make every conv record evidentially empty.
    #[test]
    fn the_conv_probes_produce_a_non_constant_output() {
        use crate::fkc::verify::bit_stability::KernelInvoker;
        use crate::fkc::verify::probe_recipes::{build_primitive_probe, probe_seed};
        use fuel_ir::dispatch::OpKind;

        let mut table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);

        let mut checked = 0usize;
        let mut constant: Vec<String> = Vec::new();
        for (op, dtypes, backend, entry) in table.iter_entries() {
            if backend != BackendId::Cpu
                || !matches!(
                    op,
                    OpKind::Conv2D | OpKind::ConvTranspose2D | OpKind::CausalConv1d
                )
            {
                continue;
            }
            let Some(probe) = build_primitive_probe(op, dtypes, probe_seed(op, dtypes)) else {
                continue;
            };
            let inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
                .with_params(probe.params.clone());
            let Ok(out) = inv.invoke(entry, &probe.inputs) else {
                continue;
            };
            checked += 1;
            let w = probe.out_dtype.size_in_bytes();
            let distinct: std::collections::HashSet<&[u8]> = out.bytes.chunks(w).collect();
            if distinct.len() < 2 {
                constant.push(format!("{op:?} {dtypes:?} -> {} distinct", distinct.len()));
            }
        }

        assert_eq!(
            checked, 20,
            "expected 20 conv registrations to invoke; got {checked}. Fewer means some \
             probe or kernel did not run, and its record rests on nothing this test saw."
        );
        assert!(
            constant.is_empty(),
            "these conv probes produce a CONSTANT output, so a kernel ignoring its \
             inputs would pass bit-stability identically: {constant:?}"
        );
    }

    /// **The four small non-conv surfaces actually read their inputs.**
    ///
    /// Same obligation as `the_conv_probes_produce_a_non_constant_output` and
    /// the same reason: **a kernel that ignored its inputs entirely would be
    /// perfectly bit-stable**, so a clean sweep says nothing about whether the
    /// probe reached anything. The all-zeros integer probes were exactly that,
    /// and no pass count could see them.
    ///
    /// ⚠️ **TWO MECHANISMS, BECAUSE ONE OF THEM IS INERT ON HALF THE
    /// POPULATION — and picking the wrong one here would be a control that
    /// cannot move, which is the failure this project has already paid for
    /// once (an attachment control whose chosen sibling was unreferenced on
    /// the path under test).**
    ///
    /// - **Seed perturbation** for `PowIElementwiseBackward`, `FusedLinear`
    ///   and `FusedSoftmaxCrossEntropy`: their operands come from
    ///   `fill_deterministic(.., seed)`, so a different seed is a different
    ///   input, and the output must move. This is the stronger check — it
    ///   proves the pipeline is live end to end — and it is the ONLY one
    ///   available to `FusedSoftmaxCrossEntropy`, whose output is a single
    ///   scalar and therefore can never have two distinct elements.
    /// - **Output variation** for `SelectiveScan` and `SsdChunkScan`: their
    ///   probes carry HAND-VERIFIED LITERALS from `fuel-cpu-backend`'s own
    ///   minimal-case tests rather than a fill, so **they do not vary with
    ///   `seed` at all** and a seed-perturbation control would be inert by
    ///   construction. The test asserts the inputs are seed-invariant rather
    ///   than assuming it, so this arm cannot silently become vacuous if the
    ///   probes are later reseeded.
    #[test]
    fn the_small_surface_probes_respond_to_their_inputs() {
        use crate::fkc::verify::bit_stability::KernelInvoker;
        use crate::fkc::verify::probe_recipes::{build_primitive_probe, probe_seed};
        use fuel_ir::dispatch::OpKind;

        let mut table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);

        let mut checked = 0usize;
        let mut inert: Vec<String> = Vec::new();
        for (op, dtypes, backend, entry) in table.iter_entries() {
            let seeded = match op {
                OpKind::PowIElementwiseBackward
                | OpKind::FusedLinear
                | OpKind::FusedSoftmaxCrossEntropy => true,
                OpKind::SelectiveScan | OpKind::SsdChunkScan => false,
                _ => continue,
            };
            if backend != BackendId::Cpu {
                continue;
            }
            let seed = probe_seed(op, dtypes);
            let Some(probe) = build_primitive_probe(op, dtypes, seed) else {
                continue;
            };
            let inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
                .with_params(probe.params.clone());
            let Ok(out) = inv.invoke(entry, &probe.inputs) else {
                continue;
            };
            checked += 1;

            let other = build_primitive_probe(op, dtypes, seed ^ 0xA5A5_A5A5)
                .expect("a recipe that built once must build again");
            let same_inputs = other.inputs.iter().map(|t| &t.bytes).collect::<Vec<_>>()
                == probe.inputs.iter().map(|t| &t.bytes).collect::<Vec<_>>();

            if seeded {
                // The claim and its own precondition, so this arm cannot go
                // vacuous if someone later replaces the fill with literals.
                if same_inputs {
                    inert.push(format!(
                        "{op:?} {dtypes:?}: listed as seed-driven but a different seed \
                         produced IDENTICAL inputs, so the check below proves nothing"
                    ));
                    continue;
                }
                let out2 = inv
                    .invoke(entry, &other.inputs)
                    .expect("second invocation of a probe that already ran");
                if out2.bytes == out.bytes {
                    inert.push(format!(
                        "{op:?} {dtypes:?}: different inputs, IDENTICAL output — the \
                         kernel is not reading them, and it would be bit-stable anyway"
                    ));
                }
            } else {
                if !same_inputs {
                    inert.push(format!(
                        "{op:?} {dtypes:?}: listed as literal-valued but the seed changed \
                         its inputs — use the seed mechanism for it, it is stronger"
                    ));
                    continue;
                }
                let w = probe.out_dtype.size_in_bytes();
                let distinct: std::collections::HashSet<&[u8]> = out.bytes.chunks(w).collect();
                if distinct.len() < 2 {
                    inert.push(format!(
                        "{op:?} {dtypes:?}: constant output ({} distinct), so a kernel \
                         ignoring its inputs would pass identically",
                        distinct.len()
                    ));
                }
            }
        }

        assert_eq!(
            checked, 20,
            "expected 20 registrations across the four small surfaces to invoke; got \
             {checked}. Fewer means a probe or kernel did not run and its record rests \
             on nothing this test saw."
        );
        assert!(
            inert.is_empty(),
            "these probes do not demonstrably reach their kernel: {inert:?}"
        );
    }

    /// **The FlashAttn probe reads its inputs AND reaches the live-prefix
    /// parameterisation it claims to.**
    ///
    /// Two obligations, and the second is the one the architect asked for by
    /// name: **a probe that only reaches one configuration and reports
    /// bit-stability for the whole family is the conformance defect** —
    /// supplying the thing being classified.
    ///
    /// **Mechanism: seed perturbation.** FlashAttn's q/k/v come from
    /// `fill_deterministic(.., seed)`, so a different seed is a different
    /// input and the output must move. The arm asserts that precondition
    /// itself rather than assuming it, so it cannot go vacuous if the probe is
    /// later rebuilt on literals the way the two scans are.
    ///
    /// **Arm reached, ASSERTED rather than documented.** Read at head, the CPU
    /// kernel is ONE loop parameterised by `k_len`
    /// (`flash_attn_native_kernel!`): `causal_offset = k_len.saturating_sub(sq)`
    /// and the score loop runs `0..k_len` over K/V of capacity `sk`. There is
    /// no separate decode arm and no separate prefill arm — the lowering sets
    /// `k_len == sk` for the static path and `k_len < sk` for decode over a
    /// fixed-capacity cache. This test requires the probe to carry
    /// `k_len < sk` and a non-zero `causal_offset`, which is the strictly
    /// stronger configuration: `k_len == sk` reads the whole buffer and
    /// exercises a strict subset. **A comment claiming that would rot; an
    /// assertion cannot.**
    #[test]
    fn the_flash_attn_probe_reaches_the_live_prefix_path_and_reads_its_inputs() {
        use crate::fkc::verify::bit_stability::KernelInvoker;
        use crate::fkc::verify::probe_recipes::{build_primitive_probe, probe_seed};
        use fuel_ir::dispatch::OpKind;

        let mut table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);

        let mut checked = 0usize;
        let mut problems: Vec<String> = Vec::new();
        for (op, dtypes, backend, entry) in table.iter_entries() {
            if backend != BackendId::Cpu || op != OpKind::FlashAttn {
                continue;
            }
            let seed = probe_seed(op, dtypes);
            let Some(probe) = build_primitive_probe(op, dtypes, seed) else {
                continue;
            };

            // The arm-reached claim, as an assertion.
            match &probe.params {
                OpParams::FlashAttn { sq, sk, k_len, .. } => {
                    if k_len >= sk {
                        problems.push(format!(
                            "{dtypes:?}: k_len({k_len}) >= sk({sk}) — the probe reads the \
                             whole K extent, so this record is evidence about the static \
                             path only and the live-prefix range is untouched"
                        ));
                    }
                    if k_len.saturating_sub(*sq) == 0 {
                        problems.push(format!(
                            "{dtypes:?}: causal_offset is 0 (k_len={k_len}, sq={sq}) — the \
                             bottom-right alignment is not exercised"
                        ));
                    }
                }
                other => problems.push(format!("{dtypes:?}: unexpected params {other:?}")),
            }

            let inv = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
                .with_params(probe.params.clone());
            let Ok(out) = inv.invoke(entry, &probe.inputs) else {
                problems.push(format!("{dtypes:?}: probe did not invoke"));
                continue;
            };
            checked += 1;

            let other = build_primitive_probe(op, dtypes, seed ^ 0xA5A5_A5A5)
                .expect("a recipe that built once must build again");
            // Precondition of the seed mechanism, asserted not assumed.
            if other.inputs.iter().map(|t| &t.bytes).collect::<Vec<_>>()
                == probe.inputs.iter().map(|t| &t.bytes).collect::<Vec<_>>()
            {
                problems.push(format!(
                    "{dtypes:?}: a different seed produced IDENTICAL inputs, so the \
                     response check below proves nothing"
                ));
                continue;
            }
            let out2 = inv
                .invoke(entry, &other.inputs)
                .expect("second invocation of a probe that already ran");
            if out2.bytes == out.bytes {
                problems.push(format!(
                    "{dtypes:?}: different inputs, IDENTICAL output — the kernel is not \
                     reading them, and it would be bit-stable anyway"
                ));
            }
        }

        assert_eq!(
            checked, 8,
            "expected 8 FlashAttn registrations to invoke (4 dtypes x {{with, without}} \
             alibi_slopes); got {checked}"
        );
        assert!(problems.is_empty(), "{problems:?}");
    }

    /// **GAP-226 split of the entries that declare nothing: how much can
    /// start now, and how much waits on a vocabulary decision.** Reports;
    /// asserts only invariants.
    ///
    /// Contract-derived CPU entries that declare no machine-checkable claim
    /// are invisible to every census of downgrades — nothing rejects them, so
    /// nothing reports them. The next question is not "fix them" but **which
    /// half is even startable**: an entry whose claim already has a NAME
    /// (`bit_stable_on_same_hardware`, `max_ulp`) and an earned ledger record
    /// is a contract edit; one whose correctness notion has no name yet
    /// (`bit_exact` for an integer output, `agrees_with_<backend>_to_ulp` for
    /// a differential) waits on the vocabulary decision tracked by GAP-227.
    ///
    /// ⚠️ **POPULATION, stated because two of them exist and mixing them is
    /// the defect this program keeps finding.** This counts the
    /// CONTRACT-DERIVED table built from the live CPU contracts, NOT
    /// `register_cpu_kernels`' production table that the sweeps above use.
    /// The two differ and their numbers must not be quoted for each other.
    ///
    /// **What "could declare X" means here, precisely:** the ledger ALREADY
    /// holds a passing record for that entry's `(backend, dtypes,
    /// kernel_revision_hash, claim)`. That is a statement about evidence
    /// that EXISTS, not evidence that could be produced — an entry with no
    /// record might need a probe recipe, an exact reference, or a claim name,
    /// and this measurement cannot separate those three. The residue is
    /// reported as ONE bucket rather than guessed into three.
    #[test]
    fn gap_226_split_of_the_entries_that_declare_nothing() {
        use crate::fkc::verify::VerificationLedger;
        use crate::fused::PrecisionGuarantee;

        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/kernel-contracts/cpu");
        let mut table = crate::kernel::KernelBindingTable::new();
        let mut fused = crate::fused::FusedKernelRegistry::new();
        let mut n_contracts = 0usize;
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
            let path = e.expect("dir entry").path();
            if path.extension().and_then(|x| x.to_str()) != Some("md") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(provider) = crate::fkc::import_bundle_str(&text, &crate::fkc::CpuLinkRegistry)
            else {
                continue;
            };
            if provider.register_into(&mut table, &mut fused).is_ok() {
                n_contracts += 1;
            }
        }
        assert!(
            n_contracts >= 10,
            "only {n_contracts} CPU contracts registered — the split below would be over \
             a subset and would read as a smaller problem than it is"
        );

        let ledger = VerificationLedger::embedded();
        let unaudited_notes = PrecisionGuarantee::UNAUDITED.notes;
        let (mut total, mut nothing, mut both, mut bit_only, mut ulp_only, mut neither) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut neither_by_op: Vec<(String, usize)> = Vec::new();

        for (op, dtypes, backend, entry) in table.iter_entries() {
            total += 1;
            if entry.precision.notes != unaudited_notes {
                continue;
            }
            nothing += 1;
            let rev = entry.kernel_revision_hash;
            let bs = ledger.has_pass(backend, dtypes, rev, "bit_stable_on_same_hardware");
            let ulp = ledger.has_pass(backend, dtypes, rev, "max_ulp");
            match (bs, ulp) {
                (true, true) => both += 1,
                (true, false) => bit_only += 1,
                (false, true) => ulp_only += 1,
                (false, false) => {
                    neither += 1;
                    let name = format!("{op:?}");
                    match neither_by_op.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, c)) => *c += 1,
                        None => neither_by_op.push((name, 1)),
                    }
                }
            }
        }
        neither_by_op.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        println!("[gap-226] population: contract-derived CPU table, {total} entries");
        println!("[gap-226] declaring nothing machine-checkable: {nothing}");
        println!("[gap-226]   ledger already holds BOTH claims : {both}");
        println!("[gap-226]   ledger holds bit_stable only     : {bit_only}");
        println!("[gap-226]   ledger holds max_ulp only        : {ulp_only}");
        println!("[gap-226]   ledger holds NEITHER             : {neither}");
        println!(
            "[gap-226] STARTABLE NOW (name exists, evidence exists, it is a contract edit): {}",
            both + bit_only + ulp_only
        );
        println!("[gap-226] NOT YET (no record — needs a probe, a reference, or a NAME), by op:");
        for (name, c) in &neither_by_op {
            println!("[gap-226]     {c:>3}  {name}");
        }

        // ⚠️ THE BORN-RED THE ARCHITECT ASKED FOR, and it is stronger than it
        // looks because of HOW this table is built.
        //
        // `build the table from contracts` NEVER RUNS `fill_unset_cpu_precision`
        // — the fill is applied inside `register_cpu_kernels`, not by
        // `register_into`. So every entry counted as backed here is backed
        // WITHOUT the fill, from contract + earned ledger record. That is the
        // property the whole increment was for, measured directly rather than
        // by removing the fill and seeing what survives.
        //
        // The numbers are pinned, not floating: after GAP-228(a) flipped 240
        // entries' worth of sections, (b) flipped conv's 20, (c) the four small
        // non-conv surfaces' 20 and (d) FlashAttn's 8, `nothing` must be 32 and
        // the backed count 591. The 32 that remain are FlashAttnBackwardK/Q/V
        // (8 each) and PagedAttn (8) — split from FlashAttn deliberately,
        // because they do NOT mirror it: the backwards take one more operand
        // (`do`) and PagedAttn is a different surface entirely. Letting
        // FlashAttn's tractability size either would be the error this program
        // has now refused three times.
        //
        // THIS COMMENT READ "80 ... 543" UNTIL THE BUCKET SWEEP FOUND IT. I
        // updated the assertion's own message from 80 to 60 and left its twin
        // four lines up: the same right-verdict-wrong-diagnosis defect, one
        // line removed from where I had just fixed it. When a number is
        // corrected, correct every statement of it, not the one being edited.
        // **A shortfall is not a smaller success — it is one
        // declaration with no earned record behind it**, and it would be
        // invisible in a diff.
        // THESE TWO NUMBERS MOVE WITH EVERY INCREMENT, AND THAT IS THE
        // POINT -- but updating them is only honest when the delta was
        // PRE-DECLARED AND MATCHED.
        //
        // 320 -> 80 was GAP-228(a) (240 flipped). 80 -> 60 is (b) (conv's 20).
        // Each time, the number written here is the one called BEFORE the run,
        // never the one observed after. A pin edited to whatever the run
        // produced is decoration; a pin edited to a figure predicted in
        // advance is the record of a prediction that held.
        assert_eq!(
            nothing, 32,
            "expected exactly 32 entries still declaring nothing after GAP-228's flip              (40 before, FlashAttn's 8 flipped). {nothing} means a declaration was not evidenced              by the record it was supposed to rest on, or a kernel revision moved and              silently un-earned one."
        );
        assert_eq!(
            total - nothing,
            591,
            "expected 591 contract-derived entries backed WITHOUT the fill. This table              is built by `register_into`, which does not apply              `fill_unset_cpu_precision` — so this count IS the post-fill-retirement              number for these entries, and a drop means the backing did not actually              move from the fill to contract + record."
        );
        assert_eq!(
            neither, nothing,
            "every entry still declaring nothing must be one with NO ledger record — if              any had a record, it was flippable and GAP-228's generator skipped it"
        );

        assert_eq!(
            both + bit_only + ulp_only + neither,
            nothing,
            "the four buckets do not partition the entries that declare nothing"
        );
        assert!(
            nothing > 0,
            "no contract-derived CPU entry declares nothing any more — that would be \
             the program finishing, not a quiet pass, and THIS test is then the stale \
             thing"
        );
    }

    /// **The `max_ulp: 0` sweep: how many CPU registrations can earn the
    /// claim the remaining 84 import downgrades are blocked on.** Reports;
    /// asserts only invariants.
    ///
    /// A `fail` here would be a FINDING — Fuel's kernel and the exact
    /// in-process reference disagreeing about a correctly-rounded result —
    /// and the count is printed separately from the skips for that reason.
    /// The two residues are not summed: "no exact reference" is a gap in this
    /// module, "fail" is a claim about a kernel.
    #[test]
    fn the_cpu_max_ulp_sweep_accounts_for_every_registration_and_fabricates_nothing() {
        let (records, log) = run_cpu_max_ulp_verification();

        let mut table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);
        let cpu_entries = table
            .iter_entries()
            .filter(|(_, _, backend, _)| *backend == BackendId::Cpu)
            .count();

        assert_eq!(
            log.len(),
            cpu_entries,
            "the sweep logged {} attempts for {cpu_entries} CPU registrations — every              entry must be accounted for, including the ones with no exact reference",
            log.len()
        );
        let passes = log.iter().filter(|a| a.outcome == "pass").count();
        assert_eq!(
            records.len(),
            passes,
            "banked {} max_ulp records against {passes} observed passes",
            records.len()
        );

        let mut buckets: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        let mut unclassified: Vec<String> = Vec::new();
        for a in &log {
            match outcome_bucket(&a.outcome) {
                Some(b) => *buckets.entry(b).or_default() += 1,
                None => unclassified.push(a.outcome.clone()),
            }
        }
        println!("[gap-225] CPU max_ulp:0 sweep over {cpu_entries} registrations:");
        for (name, n) in &buckets {
            println!("[gap-225]     {n:>4}  {name}");
        }
        for a in log.iter().filter(|a| a.outcome.starts_with("fail:")) {
            println!("[gap-225] FAIL {} {:?}: {}", a.op_name, a.dtypes, a.outcome);
        }
        assert!(
            unclassified.is_empty(),
            "these outcomes match no bucket: {unclassified:?}. Under the old remainder \
             they were counted and PRINTED as invoke errors, because the catch-all \
             carried that name."
        );
        assert_eq!(
            buckets.values().sum::<usize>(),
            log.len(),
            "the outcome buckets do not partition the log"
        );
        assert!(
            !records.is_empty(),
            "not one CPU registration earned max_ulp:0 out of {cpu_entries} — that is              a broken harness, not a coverage result: the assertions above would all hold"
        );
    }

    /// The fused and primitive passes both write into ONE ledger keyed on
    /// `(backend, dtypes, kernel_revision_hash, claim)`. `kernel_ref` is NOT
    /// in that key, so if the two populations ever produce the same key for
    /// two DIFFERENT kernels, `upsert` silently drops one and the ledger
    /// records a verdict under a name that did not earn it.
    ///
    /// The seeding test concatenates the two record sets on the assumption
    /// that this cannot happen. That assumption is checked here rather than
    /// asserted in a comment — a same-key collision would be invisible in the
    /// record count, which is the only thing the seeding test reports.
    #[test]
    fn the_fused_and_primitive_passes_do_not_collide_on_the_ledger_key() {
        let (fused, _) = run_cpu_verification();
        let (prim, _) = run_cpu_primitive_verification();
        assert!(
            !fused.is_empty() && !prim.is_empty(),
            "both populations must be non-empty or this check is vacuous              (fused {}, primitive {})",
            fused.len(),
            prim.len()
        );

        let key = |r: &LedgerRecord| {
            (
                r.backend.clone(),
                r.dtypes.clone(),
                r.kernel_revision_hash,
                r.claim.clone(),
            )
        };
        // Positive control on the PREDICATE: zero collisions is the answer we
        // expect, and a `key` that never compares equal would produce the same
        // zero for the wrong reason. Prove it can fire before believing it
        // didn't.
        let a = &fused[0];
        let mut b = a.clone();
        b.kernel_ref = format!("{}-synthetic", a.kernel_ref);
        assert!(
            key(a) == key(&b) && a.kernel_ref != b.kernel_ref,
            "the collision predicate cannot recognise two records that differ              ONLY in kernel_ref — it would report zero collisions no matter              what the two passes produced"
        );

        let mut collisions: Vec<String> = Vec::new();
        for f in &fused {
            for p in &prim {
                if key(f) == key(p) && f.kernel_ref != p.kernel_ref {
                    collisions.push(format!(
                        "{:?} rev={} claimed by both {:?} (fused) and {:?} (primitive)",
                        f.dtypes, f.kernel_revision_hash, f.kernel_ref, p.kernel_ref
                    ));
                }
            }
        }
        assert!(
            collisions.is_empty(),
            "the two CPU passes produce {} colliding ledger key(s); concatenating              their records lets `upsert` drop one verdict and file the other              under the wrong kernel_ref:
  {}",
            collisions.len(),
            collisions.join("
  ")
        );
    }

    /// The primitive sweep must ACCOUNT FOR every CPU registration and must
    /// never bank a pass it did not observe. Runs in the default suite (no
    /// device, no feature) because it writes nothing — it only measures.
    ///
    /// **This is the instrument that turns GAP-207's coverage number from an
    /// estimate into a measurement.** The program's premise is that
    /// `fill_unset_cpu_precision` asserts `bit_stable_on_same_hardware` for
    /// entries nobody verified; how many of those can actually EARN it is the
    /// fact that decides whether the fill can be retired or only narrowed. A
    /// stale estimate in a plan is exactly the shape that goes unchallenged,
    /// so the number is computed here, at head, on every run.
    ///
    /// The assertions are deliberately invariants, not thresholds: a coverage
    /// threshold would either be re-tuned whenever it failed (making it
    /// decoration) or block unrelated work. The split is PRINTED for the
    /// program to read; what is ASSERTED is only what must always hold.
    #[test]
    fn the_cpu_primitive_sweep_accounts_for_every_registration_and_fabricates_nothing() {
        let (records, log) = run_cpu_primitive_verification();

        // How many CPU entries exist, counted independently of the sweep — if
        // these disagree, the sweep skipped something silently, which is the
        // one failure this test exists to make impossible.
        let mut table = crate::kernel::KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);
        let cpu_entries = table
            .iter_entries()
            .filter(|(_, _, backend, _)| *backend == BackendId::Cpu)
            .count();

        assert_eq!(
            log.len(),
            cpu_entries,
            "the sweep logged {} attempts for {cpu_entries} CPU registrations —              every entry must be accounted for, including the ones with no              probe recipe. A silent skip is how a coverage number becomes              larger than the thing it measures.",
            log.len()
        );

        let passes = log.iter().filter(|a| a.outcome == "pass").count();
        assert_eq!(
            records.len(),
            passes,
            "banked {} ledger records against {passes} observed passes. These              must be equal in BOTH directions: more records than passes is a              fabricated claim, fewer is a measurement thrown away.",
            records.len()
        );

        // Non-triviality: an empty or near-empty table would satisfy both
        // assertions above perfectly.
        assert!(
            cpu_entries > 100,
            "only {cpu_entries} CPU registrations found — `register_cpu_kernels`              is the production path and carries several hundred. Both              assertions above pass vacuously against a table this small."
        );
        assert!(
            !records.is_empty(),
            "not one CPU primitive earned `bit_stable_on_same_hardware` out of              {cpu_entries} registrations. That is not a coverage result, it is              a broken harness — the assertions above would both hold."
        );

        let mut buckets: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        let mut unclassified: Vec<String> = Vec::new();
        for a in &log {
            match outcome_bucket(&a.outcome) {
                Some(b) => *buckets.entry(b).or_default() += 1,
                None => unclassified.push(a.outcome.clone()),
            }
        }
        let _no_recipe = buckets.get("no_recipe").copied().unwrap_or(0);
        let _failed = buckets.get("fail").copied().unwrap_or(0);
        println!("[gap-207] CPU primitive bit-stability sweep over {cpu_entries} registrations:");
        for (name, n) in &buckets {
            println!("[gap-207]     {n:>4}  {name}");
        }
        assert!(
            unclassified.is_empty(),
            "these outcomes match no bucket: {unclassified:?}"
        );
        // Name the ops in the no-recipe residue, not just its size. A bare
        // count says how much is uncovered; the names say what to build next,
        // and they are the whole actionable output of this measurement.
        let mut by_op: Vec<(String, usize)> = Vec::new();
        for a in log
            .iter()
            .filter(|a| a.outcome.starts_with("unverified: no probe recipe"))
        {
            match by_op.iter_mut().find(|(n, _)| *n == a.op_name) {
                Some((_, c)) => *c += 1,
                None => by_op.push((a.op_name.clone(), 1)),
            }
        }
        by_op.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        println!(
            "[gap-207] {} distinct ops lack a probe recipe, by registration count:",
            by_op.len()
        );
        for (name, c) in &by_op {
            println!("[gap-207]     {name} x{c}");
        }

        // The in-place arms in `build_primitive_probe` are a HAND-ENUMERATED
        // list of 24 `OpKind` variants. Nothing in the compiler checks that
        // list is complete — a new `<Op>Inplace` would simply fall through to
        // `_ => None` and go missing without a word. This closes that: the
        // residue is checked BY NAME, so an unlisted in-place op shows up here
        // instead of quietly shrinking coverage.
        //
        // Naming is the population, which is a real weakness worth stating:
        // an in-place op that does NOT end in `Inplace` is invisible to this.
        // It is a better guard than none and not a proof.
        let unlisted_inplace: Vec<&str> = log
            .iter()
            .filter(|a| a.outcome.starts_with("unverified: no probe recipe"))
            .map(|a| a.op_name.as_str())
            .filter(|n| n.ends_with("Inplace"))
            .collect();
        assert!(
            unlisted_inplace.is_empty(),
            "these in-place ops have no probe recipe: {unlisted_inplace:?}. The              `*Inplace` arm is a hand-written variant list, so a new one falls              through `_ => None` silently — add it there rather than widening              this assertion."
        );

        // The two residues are NOT the same thing and must never be summed
        // into one "uncovered" number: `no_recipe` is a gap in this harness,
        // `failed` is a claim about a kernel. Only the second is evidence that
        // `fill_unset_cpu_precision` is asserting something untrue.
        assert_eq!(
            buckets.values().sum::<usize>(),
            log.len(),
            "the outcome buckets do not partition the log"
        );
    }

    /// Task 4.5b: empirically verify the CPU fused-op family and WRITE
    /// the resulting `"pass"` records to the git-checked-in verification
    /// ledger (`docs/kernel-contracts/.fkc-verified-ledger.json`).
    ///
    /// Run with `--nocapture` to see the full per-op attempt log
    /// (pass/fail/unverified + reason) — every op in [`TARGETS`] is
    /// accounted for, not just the ones that pass.
    #[test]
    #[ignore = "re-seeding tool: writes docs/kernel-contracts/.fkc-verified-ledger.json; run manually via `cargo test -p fuel-dispatch seed_cpu_verified_ledger -- --ignored --nocapture` to regenerate the ledger. The verified ledger is committed; the default suite must not rewrite it."]
    fn seed_cpu_verified_ledger() {
        // BOTH halves of the CPU surface. The fused registry and the primitive
        // binding table are disjoint populations registered by different
        // production paths; seeding only one leaves the other's claims resting
        // on `fill_unset_cpu_precision`, which is the thing GAP-207 exists to
        // retire. Reported as a SPLIT, never as one total — the two passes
        // have different coverage and different reasons for their misses.
        let (fused_records, fused_log) = run_cpu_verification();
        let (prim_records, prim_log) = run_cpu_primitive_verification();
        // The THIRD claim: `max_ulp: 0` against an exact in-process
        // reference. Separate from bit-stability because it is a different
        // claim about the same kernels, and `gate_precision` needs BOTH
        // before a dual-claim entry's guarantee changes at all.
        let (ulp_records, ulp_log) = run_cpu_max_ulp_verification();

        for (tag, log) in [
            ("fused", &fused_log),
            ("primitive", &prim_log),
            ("max_ulp", &ulp_log),
        ] {
            for attempt in log.iter() {
                println!(
                    "[gap-207/{tag}] {} {:?} (rev={}): {}",
                    attempt.op_name, attempt.dtypes, attempt.kernel_revision_hash, attempt.outcome,
                );
            }
        }
        for (tag, recs, log) in [
            ("fused", &fused_records, &fused_log),
            ("primitive", &prim_records, &prim_log),
            ("max_ulp", &ulp_records, &ulp_log),
        ] {
            println!(
                "[gap-207/{tag}] {} passed, {} unverified/failed, {} total attempts",
                recs.len(),
                log.iter().filter(|a| a.outcome != "pass").count(),
                log.len(),
            );
        }

        assert!(
            !fused_records.is_empty(),
            "expected at least one CPU FUSED op to empirically verify bit-stable; got 0 — see log above",
        );
        assert!(
            !prim_records.is_empty(),
            "expected at least one CPU PRIMITIVE op to empirically verify bit-stable; got 0 — see log above",
        );

        // One flat set for the writer; the split above is the report, this is
        // the payload. `upsert` keys on (backend, dtypes, revision, claim), so
        // an op present in BOTH populations updates one row rather than
        // producing two — which is why concatenating is safe here.
        let mut records = fused_records;
        records.extend(prim_records);
        records.extend(ulp_records);

        // MERGE, never replace: this file also holds Vulkan and CUDA records
        // that need live devices to re-earn and that CI cannot regenerate.
        // The merge lives inside the writer, not here (GAP-210).
        let summary = super::super::ledger::write_merged_ledger(&records);
        println!(
            "[task-4.5b] merged {} fresh CPU record(s) into {} existing -> {} total, written to {}",
            summary.fresh,
            summary.before,
            summary.after,
            summary.path.display(),
        );
    }
}

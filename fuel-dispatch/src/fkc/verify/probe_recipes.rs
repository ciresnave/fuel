// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backend-agnostic probe recipes for primitive kernels.
//!
//! Extracted VERBATIM from `seed_vulkan_ledger` (pure move, no behaviour
//! change). It lived in a `#[cfg(feature = "vulkan")]` module and was named
//! `build_vulkan_probe`, but **nothing in it is Vulkan-specific** — it builds
//! host tensors and `OpParams` from an `(OpKind, dtypes)` pair, and the
//! backend enters only via which `KernelInvoker` runs the result.
//!
//! It is unconditional here because `seed_cpu_ledger` needs the same recipes
//! to earn `bit_stable_on_same_hardware` for CPU primitives, and a SECOND COPY
//! would be a divergence generator: the next dtype migration would fix one and
//! silently orphan the other. That is not hypothetical — GAP-168(c) changed
//! `masked_fill`'s mask `U8 -> Bool`, which orphaned six earned ledger records
//! AND broke the probe that would have re-earned them. One recipe set, all
//! backends, makes "backend-agnostic" true by construction rather than by
//! coincidence.
//!
//! `None` from [`build_primitive_probe`] means NO RECIPE for that op/dtype —
//! the caller logs a skip and writes no record. Never a fabricated entry.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;

use super::bit_stability::{HostTensor, ProbeInputs, fill_deterministic};
use crate::kernel::{MatmulM, OpParams};
use fuel_graph::registry::Reduction;

/// Encode `vals` into `dt`'s byte representation, covering every dtype any
/// backend's probe recipes need — the float set PLUS the integer / fp8 / bool
/// dtypes the byte-level movers fan over. For `bit_stable` / byte-exact
/// verification the values only need to be DETERMINISTIC (the kernel produces
/// identical bytes for identical input bytes on the same hardware), so the
/// integer/fp8 encodings are lossy-but-stable projections of the float probe
/// values. That reasoning is what lets ONE encoder serve all backends.
///
/// **Do not narrow this domain.** It is the single encoder behind every probe
/// in this module, so a dtype dropped here silently stops the corresponding
/// ledger records from being re-earnable — they keep passing the gate off
/// their checked-in copy while nothing can reproduce them. That is not
/// hypothetical: commit `23785514` repointed [`ht`] at the float-only
/// `seed_cpu_ledger::to_bytes`, a **same-signature helper with a strictly
/// smaller domain**, and orphaned 228 of 530 Vulkan records with a green
/// build — the compiler cannot see a refactor that only makes `None` more
/// common. `every_earned_ledger_record_can_still_be_probed` below is the guard
/// that now makes that failure loud.
///
/// Deliberately NOT shared with `seed_cpu_ledger::to_bytes`, which is the
/// narrower JIT-ingest encoder whose narrowness is itself load-bearing and
/// asserted by a test — see the note there.
fn to_bytes(dt: DType, vals: &[f32]) -> Option<Vec<u8>> {
    Some(match dt {
        DType::F32 => bytemuck::cast_slice(vals).to_vec(),
        DType::F64 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| x as f64).collect::<Vec<_>>()).to_vec()
        }
        DType::BF16 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| half::bf16::from_f32(x))
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::F16 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| half::f16::from_f32(x))
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        // ⚠️ SCALED before the conversion, and that is the whole point.
        //
        // `fill_deterministic` returns floats in roughly `[-0.5, 0.5)` and
        // `as` TRUNCATES toward zero, so the previous arms
        // (`x.abs() as u32 % 251`, `x as i16`, ...) mapped EVERY probe value
        // to 0. Every integer probe tensor was all zeros, and every ledger
        // record earned for an integer dtype was earned against a degenerate
        // input: a permutation of zeros equals a copy of zeros, so
        // `Gather`/`IndexSelect` could not be told apart from a plain copy,
        // and a kernel that ignored its input entirely would have passed.
        //
        // Found by SABOTAGE rather than by reading: making the `Gather`
        // reference ignore its indices failed 4 of 9 registrations — the four
        // FLOAT ones. `integer_probe_values_are_not_all_identical` below is
        // the guard that makes it loud.
        //
        // The scale factors spread the sub-unit range across each dtype
        // without reaching its limits, so no arm relies on saturation.
        DType::U8 => vals
            .iter()
            .map(|&x| (((x + 0.5) * 251.0) as u32 % 251) as u8)
            .collect(),
        DType::I8 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| ((x * 240.0) as i32).clamp(-120, 120) as i8)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::I16 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| (x * 60_000.0) as i16)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::U32 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| ((x + 0.5) * 4_000_000_000.0) as u32)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::I32 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| (x * 2_000_000_000.0) as i32)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::I64 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| (f64::from(x) * 9.0e18) as i64)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        // NOTE — there is deliberately NO `DType::Bool` arm, and it must stay
        // that way. This function's contract is "project float probe values
        // into `dt`'s bytes"; for Bool there is no correct projection to
        // write. `x != 0.0`? `x > 0.5`? Every choice invents a convention
        // nothing consumes. Worse, the one real Bool consumer (`MaskedFill`'s
        // mask) needs the byte pattern `[0,1,0,1]` — chosen to exercise BOTH
        // select branches — which is not a function of the float inputs at
        // all. So the mask is built as a direct `HostTensor` literal below,
        // and that is the right shape: a legitimate bypass, not a gap. Adding
        // an arm here to satisfy a coverage gate would be fabricating data to
        // please an instrument.
        // F8E4M3 (OCP **E4M3FN**): one byte per element. Produce a
        // deterministic VALID normal value — the exact value is irrelevant,
        // only that it round-trips stably.
        //
        // ⚠️ The reason previously given here was WRONG, in the way this
        // registry keeps filing: it said the exponent field is "kept out of
        // the 0b1111 inf/nan range". **E4M3FN has no infinities, and exponent
        // 1111 is NOT wholly reserved** — only mantissa 111 is NaN
        // (`S.1111.111`, two encodings), while `1111.000`-`1111.110` are
        // ordinary finite values up to 448. The probe values are fine and
        // staying clear is conservative; the JUSTIFICATION over-reserved, and
        // a true-sounding reason with the wrong scope is exactly what gets
        // repeated as fact.
        DType::F8E4M3 => vals
            .iter()
            .enumerate()
            .map(|(i, _)| 0x30u8 | ((i as u8) & 0x07))
            .collect(),
        _ => return None,
    })
}

pub(crate) fn ht(dt: DType, shape: Vec<usize>, vals: &[f32]) -> Option<HostTensor> {
    Some(HostTensor {
        dtype: dt,
        shape,
        bytes: to_bytes(dt, vals)?,
    })
}

/// Deterministic probe seed for one `(OpKind, dtypes)` registration.
///
/// Any deterministic seed satisfies `bit_stable_on_same_hardware`, which
/// re-invokes ONE probe N times — the seed never has to agree across runs for
/// that claim. It has to agree across BACKENDS for the differential claims
/// (`seed_cuda_ledger` diffs CUDA against a CPU reference), which is why this
/// lives here with [`build_primitive_probe`] rather than once per seeder: a
/// second copy would drift silently, and a differential over two different
/// inputs reports a disagreement that is entirely its own.
///
/// Moved here from `seed_vulkan_ledger` (which was the only definition, and
/// feature-gated). The move is name-resolution-safe by inspection, not by
/// assertion: the body calls NOTHING — only literals, `op as u64` and
/// `dtypes.len()` — so there is no unqualified path that could rebind against
/// the new module's scope. That check is required after `23785514`, where a
/// "pure move" silently rebound `to_bytes` to a same-signature encoder with a
/// smaller domain.
pub(crate) fn probe_seed(op: OpKind, dtypes: &[DType]) -> u64 {
    0x2545_F491_4F6C_DD1D_u64
        ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (dtypes.len() as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// A synthesized, safe, valid probe for one `(OpKind, dtypes)` registration.
pub(crate) struct Probe {
    pub(crate) inputs: ProbeInputs,
    pub(crate) params: OpParams,
    pub(crate) out_dtype: DType,
    pub(crate) out_shape: Vec<usize>,
    /// Bytes to PRE-FILL the output buffer with, for ops whose target is an
    /// output rather than an input (the `*Inplace` family: the executor hands
    /// the in-place target in as `outputs[0]`, so such a probe has NO inputs).
    ///
    /// `None` means the invoker's default — a zeroed buffer. That default is
    /// correct for every op that reads its inputs, and WRONG for an in-place
    /// op, which would then be verified against all zeros: byte-identical
    /// across repeats, one input value, no branch exercised. A claim earned
    /// that way is true and uninformative, and nothing downstream can tell it
    /// from a real one (GAP-222).
    ///
    /// An explicit field rather than an implicit rule like "no inputs means
    /// seed the output": a `Probe` should not have to be decoded to be
    /// understood, and a future zero-input op that is NOT in-place would
    /// silently inherit the wrong treatment.
    pub(crate) out_seed: Option<Vec<u8>>,
}

/// Build a real, valid probe for a primitive `op` at the registered
/// `dtypes`. `None` ⇒ no recipe for that op/dtype yet (logged + skipped, never
/// a fabricated entry).
pub(crate) fn build_primitive_probe(op: OpKind, dtypes: &[DType], seed: u64) -> Option<Probe> {
    let dt = *dtypes.first()?;

    match op {
        // --- Binary elementwise (2 inputs) ---------------------------------
        OpKind::AddElementwise
        | OpKind::SubElementwise
        | OpKind::MulElementwise
        | OpKind::DivElementwise
        | OpKind::MaximumElementwise
        | OpKind::MinimumElementwise
        // Pow / Rem are the same calling convention as the rest of the binary
        // family (`[T, T, T]`, two inputs, `OpParams::None`) and were absent
        // only because nobody listed them. They can produce NaN / inf from the
        // deterministic probe values (`pow` of a negative base, `rem` by zero);
        // that is fine for THIS claim, which is byte-identity across repeated
        // invocations, not a numeric bound — a NaN with a stable payload is
        // stable. If either turns out NOT to be, the sweep records a `fail`
        // and that is a finding about the kernel, not about the probe.
        | OpKind::PowElementwise
        | OpKind::RemElementwise => {
            let a = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let b = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0x9E37_79B9))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::None,
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Unary elementwise (1 input) -----------------------------------
        OpKind::NegElementwise
        | OpKind::ReluElementwise
        | OpKind::SqrElementwise
        | OpKind::SqrtElementwise
        | OpKind::RecipElementwise
        | OpKind::RsqrtElementwise
        | OpKind::AbsElementwise
        | OpKind::TanhElementwise
        | OpKind::ExpElementwise
        | OpKind::LogElementwise
        | OpKind::SinElementwise
        | OpKind::CosElementwise
        | OpKind::SigmoidElementwise
        | OpKind::SiluElementwise
        | OpKind::GeluElementwise
        | OpKind::GeluErfElementwise
        | OpKind::ErfElementwise
        | OpKind::StepElementwise
        | OpKind::SignElementwise
        | OpKind::FloorElementwise
        | OpKind::CeilElementwise
        | OpKind::RoundElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::None,
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Affine / Clamp / PowI (1 input + scalar params) ---------------
        OpKind::Affine => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Affine { mul: 2.0, add: 1.0 },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }
        OpKind::ClampElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Clamp {
                    min: -1.0,
                    max: 1.0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }
        OpKind::PowIElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::PowI { exp: 2 },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Copy / Cast (1 input, dtype may change) -----------------------
        OpKind::Copy => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::None,
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }
        OpKind::Cast => {
            let out_dt = *dtypes.get(1)?;
            // Cast-FROM-Bool is the one source dtype `ht` cannot build, and it
            // is 11 of the registrations here. The right fix is the same one
            // `MaskedFill`'s mask uses: construct the operand DIRECTLY rather
            // than adding a `Bool` arm to `to_bytes`.
            //
            // That distinction is the whole point and is easy to lose: there
            // is no correct f32 -> Bool PROJECTION, so a `to_bytes` arm would
            // have to invent a thresholding convention and every caller would
            // silently inherit it. A Bool tensor's bytes, on the other hand,
            // are perfectly well defined — `[0, 1, 0, 1]` is chosen so a
            // cast exercises both values rather than a constant.
            let x = if dt == DType::Bool {
                HostTensor { dtype: DType::Bool, shape: vec![4], bytes: vec![0u8, 1, 0, 1] }
            } else {
                ht(dt, vec![4], &fill_deterministic(4, seed))?
            };
            Some(Probe {
                inputs: vec![x],
                params: OpParams::None,
                out_dtype: out_dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Flip / Roll / CumSum (1 input, 3-axis flat params) ------------
        OpKind::Flip => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Flip {
                    outer_count: 1,
                    dim_size: 4,
                    inner_count: 1,
                    axis: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }
        OpKind::Roll => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Roll {
                    outer_count: 1,
                    dim_size: 4,
                    inner_count: 1,
                    shift: 1,
                    axis: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }
        OpKind::CumSum => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::CumSum {
                    outer_count: 1,
                    dim_size: 4,
                    inner_count: 1,
                    axis: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Triu / Tril (1 input, [rows, cols]) ---------------------------
        OpKind::Triu | OpKind::Tril => {
            let x = ht(dt, vec![2, 2], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Triangular {
                    batch_count: 1,
                    rows: 2,
                    cols: 2,
                    diagonal: 0,
                },
                out_dtype: dt,
                out_shape: vec![2, 2],
                out_seed: None,
            })
        }

        // --- Concat (2 inputs along axis 0) --------------------------------
        OpKind::Concat => {
            let a = ht(dt, vec![2], &fill_deterministic(2, seed))?;
            let b = ht(dt, vec![2], &fill_deterministic(2, seed ^ 0x5555))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::Concat {
                    outer_count: 1,
                    input_dim_sizes: vec![2, 2],
                    inner_count: 1,
                    axis: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- IndexSelect (src + U32 indices) -------------------------------
        OpKind::IndexSelect => {
            // inner_count MUST be even — the bf16 kernel pair-thread-packs.
            let (outer, source_dim, n_idx, inner) = (1usize, 4usize, 2usize, 2usize);
            let src = ht(
                dt,
                vec![outer * source_dim * inner],
                &fill_deterministic(outer * source_dim * inner, seed),
            )?;
            let indices = HostTensor {
                dtype: DType::U32,
                shape: vec![n_idx],
                bytes: bytemuck::cast_slice(&[0u32, 1u32]).to_vec(),
            };
            Some(Probe {
                inputs: vec![src, indices],
                params: OpParams::IndexSelect {
                    outer_count: outer,
                    source_dim_size: source_dim,
                    n_indices: n_idx,
                    inner_count: inner,
                },
                out_dtype: dt,
                out_shape: vec![outer * n_idx * inner],
                out_seed: None,
            })
        }

        // --- Gather (src + U32 indices of output shape) --------------------
        OpKind::Gather => {
            // source [2,2], gather along dim 1, output [2,2]; indices pick col.
            let src = ht(dt, vec![2, 2], &fill_deterministic(4, seed))?;
            let indices = HostTensor {
                dtype: DType::U32,
                shape: vec![2, 2],
                bytes: bytemuck::cast_slice(&[0u32, 1, 1, 0]).to_vec(),
            };
            Some(Probe {
                inputs: vec![src, indices],
                params: OpParams::Gather {
                    source_shape: vec![2, 2],
                    output_shape: vec![2, 2],
                    dim: 1,
                },
                out_dtype: dt,
                out_shape: vec![2, 2],
                out_seed: None,
            })
        }

        // --- MaskedFill (in + Bool mask) -----------------------------------
        //
        // ⚠️ The mask is `Bool`, NOT `U8`. GAP-168(c) migrated it, and this
        // probe was not migrated with it — the kernel then rejected every
        // attempt with "mask must be Bool, got U8", so all six dtype
        // combinations failed and NO record could be written under the new
        // `[T, Bool, T]` key. The six earned `[T, U8, T]` records were left
        // orphaned in place, and re-running the seeder could never restore
        // them: the harness itself had to move first.
        //
        // That is the shape worth remembering — a contract dtype change
        // invalidates the ledger key silently, AND can break the very harness
        // that would re-earn the record. Fixing the key without fixing the
        // probe just reproduces the hole one run later.
        OpKind::MaskedFill => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let mask = HostTensor {
                dtype: DType::Bool,
                shape: vec![4],
                bytes: vec![0u8, 1, 0, 1],
            };
            // fill_bytes is one element's worth in the output dtype.
            let fill = to_bytes(dt, &[0.0])?;
            Some(Probe {
                inputs: vec![x, mask],
                params: OpParams::MaskedFill { fill_bytes: fill },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Pad (1 input → padded output) ---------------------------------
        OpKind::Pad => {
            let x = ht(dt, vec![3], &fill_deterministic(3, seed))?;
            let fill = to_bytes(dt, &[0.0])?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Pad {
                    in_shape: vec![3],
                    out_shape: vec![8],
                    padding: vec![(2, 3)],
                    mode_tag: 2,
                    fill_bytes: fill,
                },
                out_dtype: dt,
                out_shape: vec![8],
                out_seed: None,
            })
        }

        // --- PadBackward (grad_out → grad_in, scatter-add) -----------------
        OpKind::PadBackward => {
            // mode_tag 0 (constant) — a pure slice of the unpadded region, valid
            // for EVERY dtype. mode_tag 1 (reflect) uses an atomic-CAS scatter-add
            // that is float-only on Vulkan, so it can't seed the U8/U32 fan-outs.
            // n_in a multiple of 4 satisfies every byte-width kernel: b2
            // (f16/bf16) needs it even, b1 (u8/i8) needs it a multiple of 4.
            let go = ht(dt, vec![8], &fill_deterministic(8, seed))?;
            Some(Probe {
                inputs: vec![go],
                params: OpParams::PadBackward {
                    in_shape: vec![4],
                    out_shape: vec![8],
                    padding: vec![(2, 2)],
                    mode_tag: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- WriteSlice (src → dest slab, in-place; dest is the output) ----
        // Last dim is 4 — the b1 (1-byte) kernel packs 4 bytes/u32, so it
        // requires the last-dim range_start and src size to be multiples of 4.
        OpKind::WriteSlice => {
            let src = ht(dt, vec![1, 4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![src],
                params: OpParams::WriteSlice {
                    dest_shape: vec![2, 4],
                    ranges: vec![(0, 1), (0, 4)],
                    deferred_dyn_offset: None,
                },
                out_dtype: dt,
                out_shape: vec![2, 4],
                out_seed: None,
            })
        }

        // --- WriteSliceRotating (src + U32 rank-0 position → dest) ----------
        OpKind::WriteSliceRotating => {
            let src = ht(dt, vec![1, 4], &fill_deterministic(4, seed))?;
            let pos = HostTensor {
                dtype: DType::U32,
                shape: vec![],
                bytes: bytemuck::cast_slice(&[1u32]).to_vec(),
            };
            Some(Probe {
                inputs: vec![src, pos],
                params: OpParams::WriteSliceRotating {
                    dest_shape: vec![2, 4],
                    axis: 0,
                    modulus: 2,
                    ranges: vec![(0, 1), (0, 4)],
                },
                out_dtype: dt,
                out_shape: vec![2, 4],
                out_seed: None,
            })
        }

        // --- ArgMaxDim / ArgMinDim (reduce a dim → U32 indices) ------------
        OpKind::ArgMaxDim | OpKind::ArgMinDim => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(
                dt,
                vec![outer, last],
                &fill_deterministic(outer * last, seed),
            )?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Reduce {
                    dims: vec![1],
                    keepdim: false,
                },
                out_dtype: DType::U32,
                out_shape: vec![outer],
                out_seed: None,
            })
        }

        // --- Rope (x, cos, sin → rotated x) --------------------------------
        OpKind::Rope => {
            let (outer, seq_n, hd) = (1usize, 2usize, 4usize);
            let x = ht(
                dt,
                vec![outer, seq_n, hd],
                &fill_deterministic(outer * seq_n * hd, seed),
            )?;
            let cos = ht(
                dt,
                vec![seq_n, hd],
                &fill_deterministic(seq_n * hd, seed ^ 0xC05),
            )?;
            let sin = ht(
                dt,
                vec![seq_n, hd],
                &fill_deterministic(seq_n * hd, seed ^ 0x51),
            )?;
            Some(Probe {
                inputs: vec![x, cos, sin],
                params: OpParams::Rope {
                    outer_count: outer,
                    seq: seq_n,
                    head_dim: hd,
                },
                out_dtype: dt,
                out_shape: vec![outer, seq_n, hd],
                out_seed: None,
            })
        }

        // --- Comparisons (2 inputs of T -> a Bool mask) --------------------
        // Registered as `[T, T, Bool]`: the output dtype is in the tuple, not
        // inferable from `dtypes.first()`, so it is read from the tuple rather
        // than assumed. `Bool` never reaches `to_bytes` here — the mask is the
        // OUTPUT, which the invoker allocates.
        OpKind::EqualElementwise
        | OpKind::NotEqualElementwise
        | OpKind::LessElementwise
        | OpKind::LessEqualElementwise
        | OpKind::GreaterElementwise
        | OpKind::GreaterEqualElementwise => {
            let out_dtype = *dtypes.get(2)?;
            let a = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let b = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0x9E37_79B9))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::None,
                out_dtype,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // --- Per-axis reduce (1 input -> reduced output) -------------------
        // Same `OpParams::Reduce` shape the ArgMax/ArgMinDim arm already uses,
        // but the output keeps the input dtype instead of becoming U32.
        OpKind::SumReduce | OpKind::MaxReduce | OpKind::MinReduce | OpKind::MeanReduce => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer, last], &fill_deterministic(outer * last, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Reduce { dims: vec![1], keepdim: false },
                out_dtype: dt,
                out_shape: vec![outer],
                out_seed: None,
            })
        }

        // --- Reduce-to-broadcast-target (1 input -> smaller target) --------
        // Distinct from the per-axis reduce above: the target is named by
        // SHAPE, not by axis list, so it carries its own `OpParams` variant.
        // `[2, 4] -> [1, 4]` reduces exactly one axis, which is enough to
        // exercise the accumulation loop without inventing a wide fixture.
        OpKind::ReduceSumTo => {
            let x = ht(dt, vec![2, 4], &fill_deterministic(8, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::ReduceSumTo {
                    input_shape: vec![2, 4],
                    output_shape: vec![1, 4],
                },
                out_dtype: dt,
                out_shape: vec![1, 4],
                out_seed: None,
            })
        }
        OpKind::ReduceMaxTo => {
            let x = ht(dt, vec![2, 4], &fill_deterministic(8, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::ReduceMaxTo {
                    input_shape: vec![2, 4],
                    output_shape: vec![1, 4],
                },
                out_dtype: dt,
                out_shape: vec![1, 4],
                out_seed: None,
            })
        }

        // --- Norm / softmax over the last dim, forward (1 input) -----------
        // Recipe shape read off `seed_cpu_ledger::build_probe`'s FUSED arms
        // for the same families, which are known to work. NOT unified with
        // them — GAP-220 tracks whether the two builders encode the same
        // obligation, and that is a question about their CALLERS, not about
        // the shapes matching. Answering it by merging first would destroy
        // the evidence needed to answer it.
        //
        // The probe is flat (`[outer * last]`) because the kernels take the
        // 2-D geometry through `OpParams`, not through the layout.
        OpKind::SoftmaxLastDim
        | OpKind::LogSoftmaxLastDim
        | OpKind::RmsNormLastDim
        | OpKind::LayerNormLastDim => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer * last], &fill_deterministic(outer * last, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: last_dim_params(op, outer, last)?,
                out_dtype: dt,
                out_shape: vec![outer * last],
                out_seed: None,
            })
        }

        // --- Norm / softmax over the last dim, backward (2 inputs) ---------
        // `(y, g)` for softmax, `(x, g)` for the norms — same arity and same
        // params variant either way, which is why one arm serves both.
        OpKind::SoftmaxLastDimBackward
        | OpKind::LogSoftmaxLastDimBackward
        | OpKind::RmsNormLastDimBackward
        | OpKind::LayerNormLastDimBackward => {
            let (outer, last) = (2usize, 4usize);
            let n = outer * last;
            let y = ht(dt, vec![n], &fill_deterministic(n, seed))?;
            let g = ht(dt, vec![n], &fill_deterministic(n, seed ^ 0x9E37_79B9))?;
            Some(Probe {
                inputs: vec![y, g],
                params: last_dim_params(op, outer, last)?,
                out_dtype: dt,
                out_shape: vec![n],
                out_seed: None,
            })
        }

        // --- ReduceMaxTo backward (x, upstream) ----------------------------
        // Degenerate no-op reduction (`input_shape == output_shape`, both
        // rank-1 length-1): every output position maps to itself, so this is
        // valid regardless of the broadcast-alignment details, which is the
        // same reasoning the fused arm records for this family.
        OpKind::ReduceMaxToBackward => {
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

        // --- The `*Inplace` family (0 inputs; target arrives as outputs[0]) -
        // The executor's `WorkItemKind::InplaceKernel` arm passes the in-place
        // target as `outputs[0]`, and the CPU wrappers REQUIRE `inputs`
        // to be empty — so the probe carries no inputs and seeds the OUTPUT
        // instead. Without that seed the kernel runs on a zeroed buffer:
        // `relu_inplace` reads 0 and writes 0, sixteen repeats agree, and the
        // claim comes back PASS having exercised one input value and no
        // branch. See `Probe::out_seed` and GAP-222.
        //
        // Scalar params ride in `OpParams`, not the dtype list, which is why
        // three of these need their own arm rather than joining the unaries.
        OpKind::AbsInplace
        | OpKind::CeilInplace
        | OpKind::CosInplace
        | OpKind::ErfInplace
        | OpKind::ExpInplace
        | OpKind::FloorInplace
        | OpKind::GeluErfInplace
        | OpKind::GeluInplace
        | OpKind::LogInplace
        | OpKind::NegInplace
        | OpKind::RecipInplace
        | OpKind::ReluInplace
        | OpKind::RoundInplace
        | OpKind::RsqrtInplace
        | OpKind::SigmoidInplace
        | OpKind::SignInplace
        | OpKind::SiluInplace
        | OpKind::SinInplace
        | OpKind::SqrInplace
        | OpKind::SqrtInplace
        | OpKind::TanhInplace => inplace_probe(dt, OpParams::None, seed),
        OpKind::ClampInplace => {
            inplace_probe(dt, OpParams::Clamp { min: -0.5, max: 0.5 }, seed)
        }
        OpKind::InplaceAffine => {
            inplace_probe(dt, OpParams::Affine { mul: 2.0, add: 1.0 }, seed)
        }
        OpKind::PowIInplace => inplace_probe(dt, OpParams::PowI { exp: 3 }, seed),

        // --- GAP-225: the BIT-STABLE-blocked class ------------------------
        // These five families are the entire `["bit_stable_on_same_hardware"]`
        // half of the import downgrades (20 of 104 warnings). They are blocked
        // because nothing can PROBE them, not because any oracle is missing —
        // so they are the part of the precision program that needs no
        // reference implementation at all, and closing them shrinks the
        // `max_ulp` problem to its true size instead of carrying these as ULP
        // work.

        // Where / select: `[Bool, T, T, T]` — cond, then the two branches.
        // The mask is a direct literal for the same reason `MaskedFill`'s is:
        // there is no correct f32 -> Bool projection, and `[0,1,0,1]` exercises
        // both branches rather than one.
        OpKind::Where => {
            let dt = *dtypes.get(1)?;
            let cond = HostTensor {
                dtype: DType::Bool,
                shape: vec![4],
                bytes: vec![0u8, 1, 0, 1],
            };
            let a = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let b = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0x9E37_79B9))?;
            Some(Probe {
                inputs: vec![cond, a, b],
                params: OpParams::None,
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // IndexAdd: `[T, U32, T, T]` — base, indices, src. Indices are a
        // direct U32 literal rather than a projection of the float probe
        // values: an index has to be IN RANGE, which is a correctness
        // precondition, not a numeric one.
        OpKind::IndexAdd => {
            let base = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let idx = HostTensor {
                dtype: DType::U32,
                shape: vec![2],
                bytes: bytemuck::cast_slice(&[0u32, 2]).to_vec(),
            };
            let src = ht(dt, vec![2], &fill_deterministic(2, seed ^ 0x1D1D))?;
            Some(Probe {
                inputs: vec![base, idx, src],
                params: OpParams::IndexAdd {
                    outer_count: 1,
                    base_dim_size: 4,
                    n_indices: 2,
                    inner_count: 1,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // ScatterAdd: same operand shape as IndexAdd, different params —
        // the destination is named by SHAPE plus a dim rather than by flat
        // outer/inner counts.
        OpKind::ScatterAdd => {
            let base = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let idx = HostTensor {
                dtype: DType::U32,
                shape: vec![2],
                bytes: bytemuck::cast_slice(&[0u32, 2]).to_vec(),
            };
            let src = ht(dt, vec![2], &fill_deterministic(2, seed ^ 0x2C2C))?;
            Some(Probe {
                inputs: vec![base, idx, src],
                params: OpParams::ScatterAdd {
                    base_shape: vec![4],
                    src_shape: vec![2],
                    dim: 0,
                },
                out_dtype: dt,
                out_shape: vec![4],
                out_seed: None,
            })
        }

        // MatMul: the blocked registrations are the INTEGER ones
        // (`[I8,I8,I8]`, `[U8,U8,U8]`), which is why this arm exists at all —
        // `ht` covers those dtypes, so the only thing missing was the recipe.
        // 2x2x2 is the smallest shape that exercises an accumulation.
        OpKind::MatMul => {
            let lhs = ht(dt, vec![2, 2], &fill_deterministic(4, seed))?;
            let rhs = ht(dt, vec![2, 2], &fill_deterministic(4, seed ^ 0x3E3E))?;
            Some(Probe {
                inputs: vec![lhs, rhs],
                params: OpParams::Matmul {
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                    m: 2,
                    n: 2,
                    k: 2,
                    m_compute: MatmulM::All,
                },
                out_dtype: dt,
                out_shape: vec![2, 2],
                out_seed: None,
            })
        }

        // WriteSliceDoff: device-offset slice write, `(source, offset)` in
        // and dest out.
        //
        // The offset IS an operand, and the binding key does not say so: the
        // key is `[T, T]` — two entries for what the wrapper reads as two
        // inputs plus an output. **The key's arity is not the operand arity**,
        // which is why the first version of this arm passed one input and the
        // wrapper rejected it by name. Recorded because the key is the only
        // thing the sweep sees, and reading operand count off it is wrong for
        // this family.
        OpKind::WriteSliceDoff => {
            let src = ht(dt, vec![1, 4], &fill_deterministic(4, seed))?;
            // I64, not U32: the wrapper reads 8 bytes and says so
            // (`offset storage has 4 bytes, need >= 8 (I64)`). The binding
            // key names neither the operand nor its dtype, so the kernel's
            // own rejection is the only source for this.
            let offset = HostTensor {
                dtype: DType::I64,
                shape: vec![],
                bytes: bytemuck::cast_slice(&[1i64]).to_vec(),
            };
            Some(Probe {
                inputs: vec![src, offset],
                params: OpParams::WriteSliceDoff {
                    dest_shape: vec![2, 4],
                    axis: 0,
                    ranges: vec![(0, 1), (0, 4)],
                },
                out_dtype: dt,
                out_shape: vec![2, 4],
                out_seed: None,
            })
        }

        // --- GAP-228(b): the conv family ----------------------------------
        //
        // These 20 registrations declare `max_ulp: ~` throughout, so ONLY
        // `bit_stable_on_same_hardware` needs earning and **no exact
        // reference is required at all**. That is a stronger reason for
        // taking conv first than the one the scope was argued on ("its
        // reference is definable") — worth recording, because a correct
        // decision reached through a weaker reason is still a weaker reason.
        //
        // Shapes read off `seed_cpu_ledger::build_probe`'s FUSED arms for the
        // same families, which are known to work. NOT unified with them:
        // GAP-220 tracks whether the two builders encode the same obligation,
        // and that is a question about their CALLERS.
        //
        // Arity comes from the dtype tuple, matching the registration: 3 = no
        // bias, 4 = bias. Read, not assumed — the tuples were measured off the
        // live binding table.
        OpKind::Conv2D | OpKind::ConvTranspose2D => {
            let with_bias = match dtypes.len() {
                3 => false,
                4 => true,
                _ => return None,
            };
            let is_transpose = matches!(op, OpKind::ConvTranspose2D);
            // Transpose: H_out = (H_in-1)*stride - 2*pad + dil*(Kh-1) + out_pad + 1
            //          = (2-1)*1 - 0 + 1*(2-1) + 0 + 1 = 3
            // Forward:   H_out = (H_in + 2*pad - dil*(Kh-1) - 1)/stride + 1 = 2
            let (x_shape, w_shape, out_shape): ([usize; 4], [usize; 4], [usize; 4]) =
                if is_transpose {
                    ([1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 3, 3])
                } else {
                    ([1, 1, 3, 3], [1, 1, 2, 2], [1, 1, 2, 2])
                };
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
                    stride: (1, 1),
                    padding: (0, 0),
                    output_padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                }
            } else {
                OpParams::Conv2D {
                    x_shape,
                    w_shape,
                    out_shape,
                    stride: (1, 1),
                    padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
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

        // Depthwise causal 1-D conv: `(x, weight, bias)`, caller left-pads.
        //
        // The values are the hand-verified ones from
        // `fuel-cpu-backend`'s `causal_conv1d_f32_no_silu_basic` (out[0]=2.1,
        // out[1]=5.1) rather than `fill_deterministic` output — a known-sane
        // invocation instead of arbitrary bytes. For a bit-stability claim
        // any deterministic input works, but a probe whose expected output is
        // known is the one that fails loudly if the calling convention drifts.
        OpKind::CausalConv1d => {
            if dtypes.len() != 4 {
                return None;
            }
            let (batch, channels, seq_in, seq_out, kernel) = (1usize, 1usize, 4usize, 2usize, 3usize);
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

        // --- the four small non-conv surfaces (GAP-228 residue) -----------
        //
        // Like conv, every one of these sections declares `max_ulp: ~`, so only
        // `bit_stable_on_same_hardware` needs earning and no exact reference is
        // required. Shapes mirror `seed_cpu_ledger::build_probe`'s FUSED arms
        // for the same families and are deliberately NOT unified with them
        // (GAP-220 — whether the two builders encode the same obligation is a
        // question about their CALLERS, not one to answer with a refactor).
        //
        // Every arity below was MEASURED off the live binding table, not
        // inferred from the fused arm: PowIElementwiseBackward 3,
        // FusedLinear 4, FusedSoftmaxCrossEntropy 3 (with `I64` targets and an
        // `F32` output regardless of the logits dtype), SelectiveScan 6,
        // SsdChunkScan 6. All four dtypes (F32/F64/BF16/F16) register for each.
        OpKind::PowIElementwiseBackward => {
            if dtypes.len() != 3 {
                return None;
            }
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

        OpKind::FusedLinear => {
            if dtypes.len() != 4 {
                return None;
            }
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

        // Targets are a literal `[1, 3]` rather than a generated fill: they are
        // INDICES into the vocab, and a fill that produced 4 or -7 would be an
        // out-of-range target rather than a harder test. The LOGITS carry the
        // seed, so the probe still varies with it.
        OpKind::FusedSoftmaxCrossEntropy => {
            if dtypes.len() != 3 || dtypes[1] != DType::I64 {
                return None;
            }
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
                // The output is F32 for every logits dtype, which is why this
                // arm does not use `dt` here. Measured, not assumed: all four
                // registrations are `[<logits>, I64, F32]`.
                out_dtype: DType::F32,
                out_shape: vec![1],
                out_seed: None,
            })
        }

        // ⚠️ THE TWO SCANS CARRY HAND-VERIFIED LITERALS, NOT A SEEDED FILL, and
        // that is a deliberate trade with a consequence worth stating: the
        // values come from `fuel-cpu-backend`'s own minimal-case tests, so a
        // drift in the calling convention fails loudly instead of producing
        // plausible garbage — but it also means THESE PROBES DO NOT VARY WITH
        // `seed`, so a seed-perturbation control is inert on them by
        // construction. `the_scan_and_small_surface_probes_respond_to_input`
        // uses the output-variation mechanism for these two and says so.
        OpKind::SelectiveScan => {
            if dtypes.len() != 6 {
                return None;
            }
            // `selective_scan_f32_single_step_seqlen_1`:
            // batch=seqlen=dim=dstate=1, u=3, delta=1, a=-1, b=2, c=0.5 -> y=3.0
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

        OpKind::SsdChunkScan => {
            if dtypes.len() != 6 {
                return None;
            }
            // `ssd_chunk_scan_f32_minimal`: batch=heads=head_dim=state_dim=
            // seqlen=chunk_size=1, x=3, dt=1, a=-1, b=2, c=0.5 -> y=3.0
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

        // --- FlashAttn (GAP-228(d)) ---------------------------------------
        //
        // Authored rather than mirrored: the attention family has NO arm in
        // `seed_cpu_ledger::build_probe` to copy, which is exactly why the
        // architect scoped it separately from the four small surfaces (those
        // were cheap BECAUSE a known-good shape already existed).
        //
        // Registrations MEASURED off the live binding table: 4 dtypes x
        // {4-tuple, 5-tuple} = 8. The tuple counts the OUTPUT, so 4 means
        // `(q, k, v, out)` and 5 means `(q, k, v, alibi_slopes, out)`.
        //
        // ⚠️ **WHICH PARAMETERISATION THIS PROBE REACHES, stated because a
        // probe that only reaches one configuration and reports bit-stability
        // for the whole family is the conformance defect — supplying the thing
        // being classified.**
        //
        // Read at head: the CPU kernel is ONE loop parameterised by `k_len`
        // (`byte_kernels.rs`, `flash_attn_native_kernel!`), with
        // `causal_offset = k_len.saturating_sub(sq)` and the score loop
        // running `0..k_len` over K/V buffers of capacity `sk`. **There is no
        // separate decode arm and no separate prefill arm** — the lowering
        // sets `k_len == sk` for the static path and `k_len < sk` for decode
        // over a fixed-capacity cache, and both traverse the same body.
        //
        // So this probe deliberately picks the STRICTLY STRONGER
        // configuration: `k_len (3) < sk (4)` with `sq = 2`, which gives a
        // NON-ZERO `causal_offset` of 1 and leaves the K/V tail outside the
        // attended range. `k_len == sk` would collapse the offset to `sk - sq`
        // and read the whole buffer, exercising a strict subset. `hq=2, hkv=1`
        // makes it GQA rather than the degenerate `hq == hkv`.
        //
        // NOT exercised, and named so nobody reads more into the record than
        // it carries: `softcap` (None here — the tanh branch), `window_size_*`
        // (None — the sliding-window admissibility branch), and `causal=false`.
        // Each is a separate branch in the same body; this record is evidence
        // about the causal, uncapped, unwindowed configuration only.
        OpKind::FlashAttn => {
            let with_alibi = match dtypes.len() {
                4 => false,
                5 => true,
                _ => return None,
            };
            let (b, hq, hkv, sq, sk, d, k_len) = (1usize, 2usize, 1usize, 2usize, 4usize, 2usize, 3usize);
            let q_len = b * hq * sq * d;
            let kv_len = b * hkv * sk * d;
            let q = ht(dt, vec![q_len], &fill_deterministic(q_len, seed))?;
            let k = ht(dt, vec![kv_len], &fill_deterministic(kv_len, seed ^ 0x11))?;
            let v = ht(dt, vec![kv_len], &fill_deterministic(kv_len, seed ^ 0x22))?;
            let mut inputs = vec![q, k, v];
            if with_alibi {
                inputs.push(ht(dt, vec![hq], &fill_deterministic(hq, seed ^ 0x33))?);
            }
            Some(Probe {
                inputs,
                params: OpParams::FlashAttn {
                    b,
                    hq,
                    hkv,
                    sq,
                    sk,
                    d,
                    k_len,
                    softmax_scale: 0.5,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                },
                out_dtype: dt,
                out_shape: vec![q_len],
                out_seed: None,
            })
        }

        // --- FlashAttn backward, ONE arm parameterised on `which` (GAP-228(e)) ---
        //
        // Three OpKinds, ONE recipe, because the differences are two: one extra
        // operand and a two-way output-shape switch. Read at head rather than
        // assumed — `byte_kernels.rs`'s backward wrapper computes
        // `need_out = match which { Q => q_n, K => kv_n, V => kv_n }`:
        // **three arms, TWO distinct values.** Splitting this into three
        // recipes would triplicate a two-way switch and give three chances to
        // get the same thing subtly different.
        //
        // Relative to `OpKind::FlashAttn` the delta is exactly:
        //   * one more operand, `do` (upstream gradient), shaped like the
        //     forward OUTPUT — inputs are `(q, k, v, do, alibi?)`, measured off
        //     the live binding table as 4 dtypes x {5-tuple, 6-tuple};
        //   * the output is q-shaped for Q and kv-shaped for K and V;
        //   * the params struct is `OpParams::FlashAttn` REUSED OUTRIGHT, on
        //     identical geometry.
        //
        // ⚠️ **`k_len == sk` HERE, AND THAT IS NOT AN OVERSIGHT — IT IS THE
        // OPPOSITE OF FLASHATTN'S CHOICE, FOR A MEASURED REASON.** The forward
        // probe deliberately takes `k_len < sk` to reach the live-prefix
        // parameterisation. **The backward has no such parameterisation to
        // reach**: the wrapper binds `k_len: _` with the comment *"Backward is
        // the static (full-K) training path; the recompute attends the full K
        // extent. k_len ignored."* Setting `k_len < sk` here would produce a
        // probe whose params say one thing and whose kernel does another, and
        // an assertion mirroring FlashAttn's would be TRUE AND MEANINGLESS —
        // a control that cannot move, which is the failure this harness has
        // now caught three times. So: `k_len == sk`, stated.
        //
        // NOT exercised, same as the forward and named for the same reason:
        // `softcap`, `window_size_*`, `causal=false`.
        OpKind::FlashAttnBackwardQ | OpKind::FlashAttnBackwardK | OpKind::FlashAttnBackwardV => {
            let with_alibi = match dtypes.len() {
                5 => false,
                6 => true,
                _ => return None,
            };
            // sq = 3, NOT the forward's 2, AND THE REASON IS A DEFECT THIS
            // FIXTURE HAD UNTIL THE ASSERTION WAS WRITTEN.
            //
            // With the forward's geometry (sq = 2) the two output-shape classes
            // come out NUMERICALLY EQUAL: q_len = 1*2*2*2 = 8 and
            // kv_len = 1*1*4*2 = 8. The `which` switch is the entire reason
            // these three OpKinds are one arm, and a fixture that collapses
            // q-shaped and kv-shaped into the same number CANNOT SEE IT: a bug
            // routing every `which` to the wrong branch would produce a
            // correctly-sized buffer every time, and the sweep passed 24/24
            // against exactly that fixture before this was noticed.
            //
            // sq = 3 gives q_len = 12 against kv_len = 8. The axis under test
            // is the output-shape class, so the fixture must separate it — a
            // green from a collapsed axis is the vacuous-oracle route that no
            // pass count can reveal.
            let (b, hq, hkv, sq, sk, d) = (1usize, 2usize, 1usize, 3usize, 4usize, 2usize);
            let q_len = b * hq * sq * d;
            let kv_len = b * hkv * sk * d;
            // The separation is GATED IN THE TEST, not asserted here. A
            // `debug_assert` in the recipe would shadow the test's diagnostic
            // in debug builds -- and the recipe is also called from the sweep,
            // where a panic surfaces as a confusing kernel-side failure rather
            // than as the finding it is.
            let q = ht(dt, vec![q_len], &fill_deterministic(q_len, seed))?;
            let k = ht(dt, vec![kv_len], &fill_deterministic(kv_len, seed ^ 0x11))?;
            let v = ht(dt, vec![kv_len], &fill_deterministic(kv_len, seed ^ 0x22))?;
            // `do` is shaped like the FORWARD output, not like `out`.
            let d_out = ht(dt, vec![q_len], &fill_deterministic(q_len, seed ^ 0x44))?;
            let mut inputs = vec![q, k, v, d_out];
            if with_alibi {
                inputs.push(ht(dt, vec![hq], &fill_deterministic(hq, seed ^ 0x33))?);
            }
            let out_len = match op {
                OpKind::FlashAttnBackwardQ => q_len,
                _ => kv_len,
            };
            Some(Probe {
                inputs,
                params: OpParams::FlashAttn {
                    b,
                    hq,
                    hkv,
                    sq,
                    sk,
                    d,
                    // Full-K: see the note above. The wrapper ignores this.
                    k_len: sk,
                    softmax_scale: 0.5,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                },
                out_dtype: dt,
                out_shape: vec![out_len],
                out_seed: None,
            })
        }

        // Residue still without a recipe, so the next reader is not guessing:
        // attention (FlashAttn x4 variants, PagedAttn), conv (Conv2D,
        // ConvTranspose2D, CausalConv1d), MatMul / QMatMul / Nf4Matmul,
        // the SSM pair, IndexAdd / ScatterAdd, Where, WriteSliceDoff, and
        // NonZeroIndices — the last of which is data-dependent-shape and so
        // has no fixed `out_shape` to declare at all.
        _ => None,
    }
}

/// A probe for an in-place op: NO inputs, and the target seeded into the
/// output buffer.
///
/// The seed is the load-bearing part. `CpuInvoker` zeroes the output by
/// default, and for an op that reads no inputs that means the kernel is
/// verified against all zeros — byte-identical across repeats, and evidence
/// of nothing (GAP-222). `to_bytes` returning `None` for a dtype propagates
/// as "no probe", never as an unseeded one.
fn inplace_probe(dt: DType, params: OpParams, seed: u64) -> Option<Probe> {
    Some(Probe {
        inputs: vec![],
        params,
        out_dtype: dt,
        out_shape: vec![4],
        out_seed: Some(to_bytes(dt, &fill_deterministic(4, seed))?),
    })
}

/// `OpParams` for the last-dim norm / softmax family, forward or backward.
///
/// Split out because the forward and backward arms need the identical mapping
/// and an inline `match` in each would be a second place to forget a variant.
/// Returns `None` for anything outside the family — a caller that reaches here
/// with another op gets no probe rather than a wrong one.
fn last_dim_params(op: OpKind, outer: usize, last: usize) -> Option<OpParams> {
    Some(match op {
        OpKind::SoftmaxLastDim | OpKind::SoftmaxLastDimBackward => OpParams::SoftmaxLastDim {
            outer_count: outer,
            last_dim: last,
        },
        OpKind::LogSoftmaxLastDim | OpKind::LogSoftmaxLastDimBackward => {
            OpParams::LogSoftmaxLastDim {
                outer_count: outer,
                last_dim: last,
            }
        }
        OpKind::RmsNormLastDim
        | OpKind::LayerNormLastDim
        | OpKind::RmsNormLastDimBackward
        | OpKind::LayerNormLastDimBackward => OpParams::NormLastDim {
            outer_count: outer,
            last_dim: last,
            eps: 1e-5,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::ledger::VerificationLedger;
    use super::*;
    use crate::kernel::KernelBindingTable;

    /// ⚠️ **Every integer probe tensor is ALL ZEROS, so every ledger record
    /// earned for an integer dtype was earned against a degenerate input.**
    ///
    /// `fill_deterministic` returns floats in roughly `[-0.5, 0.5)`, and the
    /// integer arms of `to_bytes` convert with `as`, which TRUNCATES toward
    /// zero. `x.abs() as u32` is 0 for every `|x| < 1`; so is `x as i16`, `x
    /// as i32`, `x as i64`, and `(x as i32).clamp(-120, 120) as i8`.
    ///
    /// This is GAP-222's class: a pass technically earned and evidentially
    /// empty. A permutation of zeros equals a copy of zeros, so
    /// `Gather`/`IndexSelect` cannot be distinguished from a plain copy on an
    /// integer dtype; a bit-stability check on a kernel that returns zeros is
    /// satisfied by a kernel that returns zeros.
    ///
    /// Found by sabotage, not by reading: making the `Gather` reference IGNORE
    /// its indices produced 4 failures, not 9 — the four FLOAT registrations.
    /// The five integer ones passed a reference that had stopped gathering.
    #[test]
    fn integer_probe_values_are_not_all_identical() {
        let vals = fill_deterministic(4, 0x1234_5678);
        assert!(
            vals.iter().all(|v| v.abs() < 1.0),
            "precondition: fill_deterministic returns sub-unit values, which is why a              truncating conversion collapses them: {vals:?}"
        );

        let mut degenerate = Vec::new();
        for dt in [
            DType::U8,
            DType::I8,
            DType::I16,
            DType::U32,
            DType::I32,
            DType::I64,
        ] {
            let Some(bytes) = to_bytes(dt, &vals) else {
                continue;
            };
            let w = bytes.len() / vals.len();
            let distinct: std::collections::HashSet<&[u8]> = bytes.chunks(w).collect();
            if distinct.len() < 2 {
                degenerate.push(format!(
                    "{dt:?} -> {} distinct of {}",
                    distinct.len(),
                    vals.len()
                ));
            }
        }
        assert!(
            degenerate.is_empty(),
            "these integer dtypes encode every probe value to the SAME bytes, so any              record earned with them is evidentially empty — a permutation of              identical elements is indistinguishable from a copy, and a kernel that              ignored its input entirely would pass: {degenerate:?}"
        );
    }

    /// Parse a ledger dtype token back into a `DType`. The ledger stores
    /// `DType`'s `Debug` name (see `ledger::dtypes_match`, which compares the
    /// same way), so `DType::ALL` is the authoritative inverse — and it is a
    /// hand-written const kept complete by a wildcard-free witness match, so a
    /// new dtype cannot silently fall out of this lookup.
    fn dtype_from_token(tok: &str) -> Option<DType> {
        DType::ALL.iter().copied().find(|d| format!("{d:?}") == tok)
    }

    /// Every `pass` already banked in the checked-in ledger must still be
    /// RE-EARNABLE: there must exist at least one registered op for which
    /// [`build_primitive_probe`] can still synthesize a probe from that
    /// record's dtype tuple.
    ///
    /// **The failure this exists to catch is silent by construction.** A
    /// ledger record is consulted by `gate_precision` off its checked-in copy,
    /// so it keeps satisfying the gate forever regardless of whether anything
    /// can still reproduce it. Narrowing the probe path therefore costs
    /// nothing at build time and nothing at test time — it only shows up the
    /// day someone re-runs a seeder and watches records they never touched
    /// fail to come back. Commit `23785514` did exactly this: it repointed
    /// [`ht`] at a same-signature encoder with a strictly smaller domain and
    /// orphaned 228 of 530 Vulkan records with a green build and a green
    /// `--lib` suite.
    ///
    /// **What this gate does and does not establish.** It asserts a NECESSARY
    /// condition (the tuple is still encodable at all), not a sufficient one:
    /// because most recipes key on `dtypes.first()`, a record whose first
    /// dtype is float but whose LATER operands are not is not covered here.
    /// The records whose first dtype is non-float ARE covered, which is what
    /// makes it born-red on the defect above. MEASURED, not predicted, by
    /// repointing `ht` at the narrower encoder exactly as `23785514` did:
    /// **152 of 749 records / 13 of 54 distinct tuples / 117 registered ops**.
    /// Against the 228 records that commit actually orphaned, this gate sees
    /// 152 — the residue is the first-dtype-float cases named above, and the
    /// two numbers must not be conflated. The stronger per-record
    /// form — join the ledger against `iter_entries()` and demand a probe for
    /// that record's actual op — needs the backend feature that registered
    /// the record, so it cannot be this unconditional gate. Stated as a split
    /// rather than implied to be total.
    ///
    /// The three numbers reconcile exactly, and are written down together so
    /// they cannot drift apart or be quoted for each other — each counts a
    /// different construct over the SAME 530 Vulkan records:
    /// **244** carry any non-float dtype *including* `Bool`; **228** carry one
    /// *excluding* `Bool` (the 16-record difference is `masked_fill`, whose
    /// mask is a direct `HostTensor` literal and never goes through `ht`, so
    /// those records are unaffected); **152** have a non-float dtype in FIRST
    /// position. 228 is the orphan count; 152 is what this gate sees; 244 was
    /// a first, loose measurement and is wrong for this purpose.
    ///
    /// Note also that **all 152 are Vulkan** — every CPU and CUDA record has a
    /// float first dtype. So this gate's teeth come entirely from Vulkan-earned
    /// records, which is exactly what the non-float non-triviality assertion in
    /// the body is protecting: lose those records and the gate goes quiet
    /// rather than green.
    ///
    /// Populations are deliberately BOTH external to the code under test: ops
    /// come from the live production binding table, dtype tuples from the
    /// checked-in ledger. Deriving either from `build_primitive_probe`'s own
    /// match arms would let a future narrowing shrink the requirement in
    /// lockstep, and the assertion could never go red.
    #[test]
    fn every_earned_ledger_record_can_still_be_probed() {
        // ---- op population: the LIVE production binding table ------------
        let mut table = KernelBindingTable::new();
        crate::dispatch::register_cpu_kernels(&mut table);
        let mut ops: Vec<OpKind> = Vec::new();
        for (op, _dtypes, _backend, _precision) in table.iter_precision() {
            if !ops.contains(&op) {
                ops.push(op);
            }
        }
        assert!(
            ops.len() >= 20,
            "op population collapsed to {} ops — the assertion below would be \
             vacuous. `register_cpu_kernels` is the production registration \
             path; if it stopped populating the table, fix that, not this bound.",
            ops.len()
        );

        // ---- record population: the CHECKED-IN ledger ---------------------
        let ledger = VerificationLedger::embedded();
        assert!(
            !ledger.is_empty(),
            "embedded ledger is empty — either the checked-in file was \
             truncated (see the CPU seeder's merge discipline) or it failed to \
             parse, and `embedded()` swallows a parse error into an empty \
             ledger by design. Every assertion below would pass vacuously."
        );

        // Probeability is a property of the dtype TUPLE, not of the record, so
        // evaluate once per distinct tuple (54, vs 749 records) and let the
        // failure report be deduplicated for free.
        const SEED: u64 = 0x5EED_0000_0000_0001;
        let mut verdicts: Vec<(Vec<String>, bool)> = Vec::new();
        let mut counts: Vec<(Vec<String>, usize)> = Vec::new();
        let mut non_float_first_tuples = 0usize;

        for rec in ledger.records() {
            match counts.iter_mut().find(|(t, _)| *t == rec.dtypes) {
                Some((_, n)) => *n += 1,
                None => counts.push((rec.dtypes.clone(), 1)),
            }
            if verdicts.iter().any(|(t, _)| *t == rec.dtypes) {
                continue;
            }
            let parsed: Option<Vec<DType>> =
                rec.dtypes.iter().map(|t| dtype_from_token(t)).collect();
            let ok = match &parsed {
                // An unparseable token is a hard failure, not a skip: it means
                // the ledger names a dtype this build does not have.
                None => false,
                Some(dts) => ops
                    .iter()
                    .any(|&op| build_primitive_probe(op, dts, SEED).is_some()),
            };
            if let Some(dts) = &parsed
                && !matches!(
                    dts.first(),
                    Some(DType::F32 | DType::F64 | DType::BF16 | DType::F16)
                )
            {
                non_float_first_tuples += 1;
            }
            verdicts.push((rec.dtypes.clone(), ok));
        }

        // ---- non-triviality: the gate must still be POINTED at the defect --
        // Every recipe encodes floats, so a ledger of float-only tuples would
        // pass this test no matter how far the encoder were narrowed. The
        // records that give it teeth are the ones whose FIRST dtype is not a
        // float — if they ever leave the ledger, this gate has gone vacuous
        // and should say so rather than keep reporting green.
        //
        // The counter ranges over DISTINCT TUPLES, not records — it is
        // incremented past the dedup `continue` above. At time of writing that
        // is 13 tuples covering 152 records; only the `> 0` matters here, but
        // the name says which construct it counts so the two never get quoted
        // for each other.
        assert!(
            non_float_first_tuples > 0,
            "no ledger record has a non-float first dtype, so this gate can no \
             longer detect an encoder narrowing — it would pass against a \
             float-only `to_bytes`. Re-point it or delete it; do not leave it \
             passing."
        );

        let failed: Vec<_> = verdicts.iter().filter(|(_, ok)| !ok).collect();
        if !failed.is_empty() {
            let orphaned: usize = failed
                .iter()
                .map(|(t, _)| counts.iter().find(|(c, _)| c == t).map_or(0, |(_, n)| *n))
                .sum();
            let mut lines = String::new();
            for (t, _) in &failed {
                let n = counts
                    .iter()
                    .find(|(c, _)| *c == **t)
                    .map_or(0, |(_, n)| *n);
                lines.push_str(&format!("\n    {t:?} x{n}"));
            }
            panic!(
                "{} of {} checked-in ledger records ({} of {} distinct dtype \
                 tuples) can no longer be probed by ANY of the {} registered \
                 ops — they are earned passes that nothing can re-earn. This \
                 almost always means `to_bytes` in this module lost a dtype \
                 arm, or `ht` was repointed at a narrower encoder. Do not fix \
                 it by deleting the records.{}",
                orphaned,
                ledger.len(),
                failed.len(),
                verdicts.len(),
                ops.len(),
                lines
            );
        }
    }
}

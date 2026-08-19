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
use crate::kernel::OpParams;

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
        DType::U8 => vals.iter().map(|&x| (x.abs() as u32 % 251) as u8).collect(),
        DType::I8 => bytemuck::cast_slice(
            &vals
                .iter()
                .map(|&x| (x as i32).clamp(-120, 120) as i8)
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        DType::I16 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| x as i16).collect::<Vec<_>>()).to_vec()
        }
        DType::U32 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| x.abs() as u32).collect::<Vec<_>>()).to_vec()
        }
        DType::I32 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| x as i32).collect::<Vec<_>>()).to_vec()
        }
        DType::I64 => {
            bytemuck::cast_slice(&vals.iter().map(|&x| x as i64).collect::<Vec<_>>()).to_vec()
        }
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
        // F8E4M3: one byte per element. Produce a deterministic VALID normal
        // value (exponent field kept out of the 0b1111 inf/nan range) — the
        // exact value is irrelevant, only that it round-trips stably.
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

/// A synthesized, safe, valid probe for one `(OpKind, dtypes)` registration.
pub(crate) struct Probe {
    pub(crate) inputs: ProbeInputs,
    pub(crate) params: OpParams,
    pub(crate) out_dtype: DType,
    pub(crate) out_shape: Vec<usize>,
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
        | OpKind::MinimumElementwise => {
            let a = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            let b = ht(dt, vec![4], &fill_deterministic(4, seed ^ 0x9E37_79B9))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::None,
                out_dtype: dt,
                out_shape: vec![4],
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
            })
        }
        OpKind::PowIElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::PowI { exp: 2 },
                out_dtype: dt,
                out_shape: vec![4],
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
            })
        }
        OpKind::Cast => {
            let out_dt = *dtypes.get(1)?;
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::None,
                out_dtype: out_dt,
                out_shape: vec![4],
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
            })
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::ledger::VerificationLedger;
    use super::*;
    use crate::kernel::KernelBindingTable;

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
            if let Some(dts) = &parsed {
                if !matches!(
                    dts.first(),
                    Some(DType::F32 | DType::F64 | DType::BF16 | DType::F16)
                ) {
                    non_float_first_tuples += 1;
                }
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

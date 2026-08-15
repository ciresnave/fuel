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

use fuel_ir::dispatch::OpKind;
use fuel_ir::DType;

use super::bit_stability::{fill_deterministic, HostTensor, ProbeInputs};
use super::seed_cpu_ledger::to_bytes;
use crate::kernel::OpParams;

pub(crate) fn ht(dt: DType, shape: Vec<usize>, vals: &[f32]) -> Option<HostTensor> {
    Some(HostTensor { dtype: dt, shape, bytes: to_bytes(dt, vals)? })
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
            Some(Probe { inputs: vec![a, b], params: OpParams::None, out_dtype: dt, out_shape: vec![4] })
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
            Some(Probe { inputs: vec![x], params: OpParams::None, out_dtype: dt, out_shape: vec![4] })
        }

        // --- Affine / Clamp / PowI (1 input + scalar params) ---------------
        OpKind::Affine => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::Affine { mul: 2.0, add: 1.0 }, out_dtype: dt, out_shape: vec![4] })
        }
        OpKind::ClampElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::Clamp { min: -1.0, max: 1.0 }, out_dtype: dt, out_shape: vec![4] })
        }
        OpKind::PowIElementwise => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::PowI { exp: 2 }, out_dtype: dt, out_shape: vec![4] })
        }

        // --- Copy / Cast (1 input, dtype may change) -----------------------
        OpKind::Copy => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::None, out_dtype: dt, out_shape: vec![4] })
        }
        OpKind::Cast => {
            let out_dt = *dtypes.get(1)?;
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::None, out_dtype: out_dt, out_shape: vec![4] })
        }

        // --- Flip / Roll / CumSum (1 input, 3-axis flat params) ------------
        OpKind::Flip => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::Flip { outer_count: 1, dim_size: 4, inner_count: 1, axis: 0 }, out_dtype: dt, out_shape: vec![4] })
        }
        OpKind::Roll => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::Roll { outer_count: 1, dim_size: 4, inner_count: 1, shift: 1, axis: 0 }, out_dtype: dt, out_shape: vec![4] })
        }
        OpKind::CumSum => {
            let x = ht(dt, vec![4], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::CumSum { outer_count: 1, dim_size: 4, inner_count: 1, axis: 0 }, out_dtype: dt, out_shape: vec![4] })
        }

        // --- Triu / Tril (1 input, [rows, cols]) ---------------------------
        OpKind::Triu | OpKind::Tril => {
            let x = ht(dt, vec![2, 2], &fill_deterministic(4, seed))?;
            Some(Probe { inputs: vec![x], params: OpParams::Triangular { batch_count: 1, rows: 2, cols: 2, diagonal: 0 }, out_dtype: dt, out_shape: vec![2, 2] })
        }

        // --- Concat (2 inputs along axis 0) --------------------------------
        OpKind::Concat => {
            let a = ht(dt, vec![2], &fill_deterministic(2, seed))?;
            let b = ht(dt, vec![2], &fill_deterministic(2, seed ^ 0x5555))?;
            Some(Probe {
                inputs: vec![a, b],
                params: OpParams::Concat { outer_count: 1, input_dim_sizes: vec![2, 2], inner_count: 1, axis: 0 },
                out_dtype: dt,
                out_shape: vec![4],
            })
        }

        // --- IndexSelect (src + U32 indices) -------------------------------
        OpKind::IndexSelect => {
            // inner_count MUST be even — the bf16 kernel pair-thread-packs.
            let (outer, source_dim, n_idx, inner) = (1usize, 4usize, 2usize, 2usize);
            let src = ht(dt, vec![outer * source_dim * inner], &fill_deterministic(outer * source_dim * inner, seed))?;
            let indices = HostTensor { dtype: DType::U32, shape: vec![n_idx], bytes: bytemuck::cast_slice(&[0u32, 1u32]).to_vec() };
            Some(Probe {
                inputs: vec![src, indices],
                params: OpParams::IndexSelect { outer_count: outer, source_dim_size: source_dim, n_indices: n_idx, inner_count: inner },
                out_dtype: dt,
                out_shape: vec![outer * n_idx * inner],
            })
        }

        // --- Gather (src + U32 indices of output shape) --------------------
        OpKind::Gather => {
            // source [2,2], gather along dim 1, output [2,2]; indices pick col.
            let src = ht(dt, vec![2, 2], &fill_deterministic(4, seed))?;
            let indices = HostTensor { dtype: DType::U32, shape: vec![2, 2], bytes: bytemuck::cast_slice(&[0u32, 1, 1, 0]).to_vec() };
            Some(Probe {
                inputs: vec![src, indices],
                params: OpParams::Gather { source_shape: vec![2, 2], output_shape: vec![2, 2], dim: 1 },
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
            let mask = HostTensor { dtype: DType::Bool, shape: vec![4], bytes: vec![0u8, 1, 0, 1] };
            // fill_bytes is one element's worth in the output dtype.
            let fill = to_bytes(dt, &[0.0])?;
            Some(Probe { inputs: vec![x, mask], params: OpParams::MaskedFill { fill_bytes: fill }, out_dtype: dt, out_shape: vec![4] })
        }

        // --- Pad (1 input → padded output) ---------------------------------
        OpKind::Pad => {
            let x = ht(dt, vec![3], &fill_deterministic(3, seed))?;
            let fill = to_bytes(dt, &[0.0])?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Pad { in_shape: vec![3], out_shape: vec![8], padding: vec![(2, 3)], mode_tag: 2, fill_bytes: fill },
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
                params: OpParams::PadBackward { in_shape: vec![4], out_shape: vec![8], padding: vec![(2, 2)], mode_tag: 0 },
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
                params: OpParams::WriteSlice { dest_shape: vec![2, 4], ranges: vec![(0, 1), (0, 4)], deferred_dyn_offset: None },
                out_dtype: dt,
                out_shape: vec![2, 4],
            })
        }

        // --- WriteSliceRotating (src + U32 rank-0 position → dest) ----------
        OpKind::WriteSliceRotating => {
            let src = ht(dt, vec![1, 4], &fill_deterministic(4, seed))?;
            let pos = HostTensor { dtype: DType::U32, shape: vec![], bytes: bytemuck::cast_slice(&[1u32]).to_vec() };
            Some(Probe {
                inputs: vec![src, pos],
                params: OpParams::WriteSliceRotating { dest_shape: vec![2, 4], axis: 0, modulus: 2, ranges: vec![(0, 1), (0, 4)] },
                out_dtype: dt,
                out_shape: vec![2, 4],
            })
        }

        // --- ArgMaxDim / ArgMinDim (reduce a dim → U32 indices) ------------
        OpKind::ArgMaxDim | OpKind::ArgMinDim => {
            let (outer, last) = (2usize, 4usize);
            let x = ht(dt, vec![outer, last], &fill_deterministic(outer * last, seed))?;
            Some(Probe {
                inputs: vec![x],
                params: OpParams::Reduce { dims: vec![1], keepdim: false },
                out_dtype: DType::U32,
                out_shape: vec![outer],
            })
        }

        // --- Rope (x, cos, sin → rotated x) --------------------------------
        OpKind::Rope => {
            let (outer, seq_n, hd) = (1usize, 2usize, 4usize);
            let x = ht(dt, vec![outer, seq_n, hd], &fill_deterministic(outer * seq_n * hd, seed))?;
            let cos = ht(dt, vec![seq_n, hd], &fill_deterministic(seq_n * hd, seed ^ 0xC05))?;
            let sin = ht(dt, vec![seq_n, hd], &fill_deterministic(seq_n * hd, seed ^ 0x51))?;
            Some(Probe {
                inputs: vec![x, cos, sin],
                params: OpParams::Rope { outer_count: outer, seq: seq_n, head_dim: hd },
                out_dtype: dt,
                out_shape: vec![outer, seq_n, hd],
            })
        }

        _ => None,
    }
}

//! JIT-on-request **adopt glue** — the seam-consumption site (kernel-seam-interop
//! §5.2). Given a [`Synthesizer`] and a Fuel-chosen [`JitRequest`], run the
//! two-step handover and register the result as a Tier-2 runtime fused op:
//!
//! ```text
//! synthesize(req) -> Synthesized{entry_point} | Declined
//!   (Declined -> None; the region stays on primitives)
//! take_kernel(entry_point) -> SynthArtifact{ artifact(PTX), link, contract }
//! load_kernel(&art) -> KernelRef            <- the backend-specific CUDA seam
//! adopt_runtime_fused(entry_point, req.region, kernel, dtypes, backend) -> FusedOpId
//! ```
//!
//! **Backend-agnostic by construction:** the only device-specific step —
//! load the PTX as a module, resolve `link.symbol`, wrap it as a [`KernelRef`] —
//! is the caller-provided `load_kernel` closure (the CUDA backend supplies it at
//! the live call site, via `baracuda_driver::Module::load_ptx`; tests pass a
//! mock). So this orchestration is testable without a device.
//!
//! The recipe's `decompose` is `req.region` (Fuel already holds it), so no
//! contract re-serialization is needed here; the FKC `contract` (cost / precision)
//! is a later refinement over the cost-from-decompose sentinel `adopt` already
//! applies. Gated behind the `jit` feature so the core dispatch layer stays free
//! of the envelope crate.

use baracuda_kernels_types::{ElementKind, OperandDesc};
use fuel_graph::registry::FusedOpId;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Error, Result};
use fuel_kernel_seam::{JitRequest, JitResponse, SynthArtifact, Synthesizer};

use crate::kernel::KernelRef;
use crate::runtime_fused_kernels::adopt_runtime_fused;

/// Baracuda [`ElementKind`] → Fuel [`DType`]. `None` for a kind with no Fuel
/// dtype.
///
/// ⚠️ **It is the inverse of [`dtype_to_element_kind`], NOT of the telemetry
/// provider's `map_element_kind`** — which is what an earlier version of this
/// comment claimed, and the claim was load-bearing in the wrong direction.
///
/// Those two answer to different CONTRACTS, and that — not their domain — is
/// what keeps them apart. This pair is a BIJECTION over the mapped set:
/// `dtype_to_element_kind` is gated by a round-trip identity test, since a
/// non-identity trip would describe operands the caller does not have.
/// `map_element_kind` is **one-way** (structure-key derivation only) and carries
/// no inverse obligation. After GAP-177 (ii) both map the SAME eleven types
/// including `Fp8E4M3`/`Fp8E5M2`, so the domains coincide and only the
/// round-trip obligation distinguishes them.
///
/// Why this matters: it once licensed consolidating the two into "one
/// authority" on the theory that a shared signature over an overlapping domain
/// is a shared contract. It is not — the round-trip invariant lives on this
/// pair and not on `map_element_kind`. **Before consolidating two similar
/// functions, check whether they are subject to the same invariants.** See
/// GAP-177.
///
/// `pub(crate)` (widened from private) so [`crate::jit_ingest_probe`]'s
/// `probe_from_operands` can reuse it instead of duplicating the match —
/// both are `jit`-feature siblings, no visibility escapes the crate.
pub(crate) fn element_kind_to_dtype(ek: ElementKind) -> Option<DType> {
    Some(match ek {
        ElementKind::U8 => DType::U8,
        ElementKind::S8 => DType::I8,
        ElementKind::I32 => DType::I32,
        ElementKind::I64 => DType::I64,
        ElementKind::Bf16 => DType::BF16,
        ElementKind::F16 => DType::F16,
        ElementKind::F32 => DType::F32,
        ElementKind::F64 => DType::F64,
        // Index operands. `U32` is the seam's gather/scatter INDEX ctype
        // (`unsigned int`) — deliberately not a compute dtype upstream (no
        // `Element` impl, no vector path). Fuel still needs it mapped, because
        // any INDEXED region carries one: `IndexSelect`, scatter/gather, and
        // `PagedAttn`'s `block_table` + `context_lens`. Its absence here meant a
        // returned contract naming a U32 operand could not be mapped back into a
        // Fuel `DType` at all — an interop gap that would have bitten on the
        // first indexed region regardless of anything else.
        ElementKind::U32 => DType::U32,
        // FP8 (GAP-177 (ii)). OCP finite E4M3 / E5M2 — the same format Fuel's
        // `DType` names (GAP-169); the telemetry structure-key path asserts the
        // identical pair. See `fp8_maps_to_baracuda_ocp_element_kinds`.
        ElementKind::Fp8E4M3 => DType::F8E4M3,
        ElementKind::Fp8E5M2 => DType::F8E5M2,
        // `Bool` (GAP-193). This sat in the decline list below until
        // `DType::Bool` landed with the GAP-168(c) comparison cut — at which
        // moment the decline's stated reason, "no Fuel `DType` for these seam
        // kinds", became FALSE. baracuda's `Bool` is 1-byte storage with
        // 0/non-zero truthiness and Fuel's is canonically 0/1 in one byte: the
        // same representation on both sides, and the same identity the telemetry
        // `map_element_kind` path already ships.
        //
        // ⚠️ NOTE THE ASYMMETRY THAT LET THIS ROT SILENTLY: adding a Fuel `DType`
        // makes the OUTBOUND match (`dtype_to_element_kind`) a compile error, but
        // this INBOUND match keys on `ElementKind`, so a new Fuel dtype applies no
        // exhaustiveness pressure here at all. The outbound direction broke
        // loudly; this one just became quietly wrong.
        ElementKind::Bool => DType::Bool,
        // No Fuel `DType` for these seam kinds — decline rather than substitute
        // a wrong one. Enumerated (never `_`) so a new baracuda `ElementKind`
        // becomes a COMPILE ERROR here, forcing a map-or-decline decision at the
        // bump instead of a silent decline. `ElementKind` is not
        // `#[non_exhaustive]` at the locked vocab, so this exhaustive match is
        // legal across the crate boundary. GAP-177 (i).
        ElementKind::F32Strict
        | ElementKind::S4
        | ElementKind::U4
        | ElementKind::Bin
        | ElementKind::Complex32
        | ElementKind::Complex64 => return None,
    })
}

/// The outbound direction: a Fuel [`DType`] as the seam's [`ElementKind`].
///
/// Fuel had **no** `DType -> ElementKind` mapping at all, which meant a request
/// could not be *constructed* for a region whose operands Fuel knows the dtypes
/// of — the seam could express them, Fuel just had no way to say them. Inverse
/// of [`element_kind_to_dtype`] over exactly the same set, so the round trip is
/// total on the mapped subset (asserted in tests).
///
/// `None` for dtypes with no seam spelling rather than a lossy substitution — a
/// request that misrepresents its operand types would get an answer about a
/// different kernel than the one asked for.
pub(crate) fn dtype_to_element_kind(dt: DType) -> Option<ElementKind> {
    Some(match dt {
        DType::U8 => ElementKind::U8,
        DType::I8 => ElementKind::S8,
        DType::I32 => ElementKind::I32,
        DType::I64 => ElementKind::I64,
        DType::BF16 => ElementKind::Bf16,
        DType::F16 => ElementKind::F16,
        DType::F32 => ElementKind::F32,
        DType::F64 => ElementKind::F64,
        DType::U32 => ElementKind::U32,
        // FP8 (GAP-177 (ii)). OCP finite E4M3 / E5M2 mapped to baracuda's OCP FP8
        // kinds — the same format on both sides (GAP-169), the identity the
        // telemetry structure-key path already ships and tests. Behaviour change:
        // FP8-operand regions become adoptable on the JIT path where they used to
        // decline. See `fp8_maps_to_baracuda_ocp_element_kinds`.
        DType::F8E4M3 => ElementKind::Fp8E4M3,
        DType::F8E5M2 => ElementKind::Fp8E5M2,
        // `Bool` (GAP-193). The GAP-168(c) comparison cut added `DType::Bool`
        // and did not gate `--features jit`, so this match went non-exhaustive
        // and `main` was RED under that feature from that merge until it was
        // found. The exhaustiveness GAP-177 (i) added is what turned a silent
        // mis-mapping into a build failure — it worked exactly as designed; the
        // gate was simply never pointed at this feature. See the inbound arm for
        // why `Bool` maps rather than declines.
        DType::Bool => ElementKind::Bool,
        // No seam `ElementKind` for these Fuel dtypes — decline rather than a
        // lossy substitution. Enumerated (never `_`) so a new `DType` is a
        // COMPILE ERROR here instead of a silent decline. GAP-177 (i).
        DType::I16 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 | DType::F8E6M2 => {
            return None;
        }
    })
}

/// The per-operand Fuel dtypes from the request operands (the binding-key
/// metadata `adopt` stamps on the runtime op).
fn operand_dtypes(operands: &[OperandDesc]) -> Vec<DType> {
    operands
        .iter()
        .filter_map(|o| element_kind_to_dtype(o.dtype))
        .collect()
}

/// Run the JIT adopt loop for `req.region`. Returns the [`Adopted`] pair
/// (recipe id + **the kernel this call loaded**) on success, `Ok(None)` if the
/// synthesizer declined. `load_kernel`
/// is the backend-specific step (PTX → `KernelRef`); the caller provides it.
///
/// Never a realize-time action — this runs in the optimizer's background
/// (idle-time, G7) adopt path; after it returns, `offer_runtime_fused_arm` will
/// emit the fused arm on the next optimize pass.
/// What an adoption produced: the recipe's runtime [`FusedOpId`] **and the
/// [`KernelRef`] this call loaded**.
///
/// The kernel is returned because it **cannot be recovered from the id**, and
/// that is by design rather than an oversight. A `FusedOpId` names a *recipe*,
/// not an artifact: `register_runtime_fused` deduplicates on the region's
/// base-map hash and ignores the name, so two adoptions of the same recipe
/// share one id (`fuel-graph/src/runtime_fused.rs`, whose own doc says "two
/// calls with the same shape but different `name`s return the same id"), and
/// `docs/architecture/04-optimization.md` ratifies multiple kernels under one
/// decision point.
///
/// So looking the id back up returns *an* alternative, and today that is the
/// **first-registered** one on both paths — `first_runtime_fused` takes
/// `alts.first()`, and `lookup_with_caps`'s own comment records that with no
/// binding setting `requires_broadcast` it is "byte-identical to returning the
/// first-registered alternative". **For the second adopter of a recipe that is
/// somebody else's kernel** (GAP-213); and if that kernel's device has since
/// been dropped it is not merely wrong but undefined (GAP-214).
///
/// Measured: two tests adopting the same `relu(add)` f32 region both received
/// `FusedOpId(32768)`, and the second launched the first's entry point.
#[derive(Clone, Copy, Debug)]
pub struct Adopted {
    /// The recipe's runtime id. Shared with any other adoption of the same recipe.
    pub id: FusedOpId,
    /// The kernel **this** adoption loaded. Dispatch through this; do not
    /// re-derive it from `id`.
    pub kernel: KernelRef,
}

pub fn adopt_from_response(
    synth: &dyn Synthesizer,
    req: &JitRequest,
    backend: BackendId,
    load_kernel: impl FnOnce(&SynthArtifact) -> Result<KernelRef>,
) -> Result<Option<Adopted>> {
    let entry_point = match synth.synthesize(req) {
        JitResponse::Synthesized { entry_point } => entry_point,
        JitResponse::Declined { .. } => return Ok(None),
    };
    let art = synth.take_kernel(&entry_point).ok_or_else(|| {
        Error::Msg(format!(
            "take_kernel({entry_point}): synthesizer retained nothing"
        ))
    })?;
    let kernel = load_kernel(&art)?;
    let dtypes = operand_dtypes(&req.operands);
    // req.region IS the recipe's decompose (fuel_graph::jit::PatternNode re-exports
    // the envelope's PatternNode), so adopt registers it as the runtime op's recipe.
    Ok(
        adopt_runtime_fused(entry_point, req.region.clone(), kernel, dtypes, backend)
            .map(|id| Adopted { id, kernel }),
    )
}

#[cfg(test)]
mod tests {
    /// **The two dtype directions must be exact inverses on the mapped set.**
    ///
    /// `element_kind_to_dtype` existed alone, so Fuel could *read* a returned
    /// contract's operand dtypes but could not *construct* a request naming
    /// them. Adding the outbound direction creates a round trip, and a round
    /// trip that is not the identity is worse than no round trip: a request
    /// would silently describe operands other than the ones the caller has, and
    /// the answer would be about a different kernel.
    ///
    /// Exhaustive over the mapped set — listed explicitly rather than derived,
    /// so ADDING a variant to one direction and not the other fails here instead
    /// of silently narrowing the seam. `U32` is in the list because it is the
    /// index ctype every indexed region carries (`IndexSelect`, scatter/gather,
    /// `PagedAttn`'s `block_table` and `context_lens`) — the case whose absence
    /// blocked a real request.
    #[test]
    fn dtype_and_element_kind_round_trip_exactly() {
        use super::{dtype_to_element_kind, element_kind_to_dtype};
        use fuel_ir::DType;

        const MAPPED: &[DType] = &[
            DType::U8,
            DType::I8,
            DType::I32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::U32,
            // FP8 (GAP-177 (ii)). Same OCP format on both sides — pinned exactly
            // by `fp8_maps_to_baracuda_ocp_element_kinds` below — so they
            // round-trip like the rest.
            DType::F8E4M3,
            DType::F8E5M2,
        ];

        for &dt in MAPPED {
            let ek = dtype_to_element_kind(dt)
                .unwrap_or_else(|| panic!("{dt:?} must have a seam spelling"));
            assert_eq!(
                element_kind_to_dtype(ek),
                Some(dt),
                "{dt:?} -> {ek:?} -> back must be the identity; a non-identity round trip means a request describes operands the caller does not have",
            );
        }

        // CONTROL: the map is PARTIAL by design, not total. Without this, a
        // `dtype_to_element_kind` that returned `Some(F32)` for everything would
        // satisfy every assertion above.
        assert!(
            dtype_to_element_kind(DType::I16).is_none(),
            "I16 has no ElementKind spelling (the seam has S8/U8/I32/I64/U32, no 16-bit int) — it must decline, not substitute — a lossy mapping asks about a different kernel than the caller's",
        );
    }

    /// **FP8 (E4M3/E5M2) maps to baracuda's OCP FP8 element kinds — GAP-177 (ii).**
    ///
    /// Behaviour change: FP8-operand regions become adoptable on the JIT path
    /// where they previously declined. It is safe because it is the SAME format
    /// on both sides, not a newly invented identification:
    /// - Fuel `DType::F8E4M3` is OCP finite E4M3 (bias 7, max ±448, no infinities,
    ///   single NaN) — GAP-169; the sibling `F8E5M2` doc names E4M3 as "the
    ///   OCP-standard FP8 pair's other half".
    /// - baracuda `ElementKind::Fp8E4M3` is documented bias 7 / max-finite 448 /
    ///   no infinities — the same OCP finite E4M3. `Fp8E5M2` is bias 15 / IEEE
    ///   inf-nan — the same OCP E5M2, which Fuel's `F8E5M2` already matches.
    /// - The telemetry structure-key path already ships AND tests this exact pair
    ///   (`telemetry::baracuda_provider::map_element_kind`); this pins the JIT
    ///   path to the same committed identity rather than inventing one.
    #[test]
    fn bool_maps_to_the_seam_bool_and_did_not_stay_declined() {
        use super::{dtype_to_element_kind, element_kind_to_dtype};
        use baracuda_kernels_types::ElementKind;
        use fuel_ir::DType;

        // GAP-193. `DType::Bool` arrived with the GAP-168(c) comparison cut and
        // made this pair's declines wrong in BOTH directions — but only one
        // direction said so. Outbound went non-exhaustive (a hard E0004 that sat
        // on `main` because no gate ran `--features jit`); inbound kept declining
        // under a comment — "no Fuel `DType` for these seam kinds" — that had
        // become false, with nothing to detect it, because a new Fuel dtype
        // applies no exhaustiveness pressure to a match keyed on `ElementKind`.
        //
        // Both are pinned here so a future decline has to argue with a test
        // rather than only with a comment.
        assert_eq!(dtype_to_element_kind(DType::Bool), Some(ElementKind::Bool));
        assert_eq!(element_kind_to_dtype(ElementKind::Bool), Some(DType::Bool));
    }

    #[test]
    fn fp8_maps_to_baracuda_ocp_element_kinds() {
        use super::{dtype_to_element_kind, element_kind_to_dtype};
        use baracuda_kernels_types::ElementKind;
        use fuel_ir::DType;

        // Outbound: Fuel FP8 dtype -> the OCP seam kind (exact, not just
        // round-trip-consistent).
        assert_eq!(
            dtype_to_element_kind(DType::F8E4M3),
            Some(ElementKind::Fp8E4M3)
        );
        assert_eq!(
            dtype_to_element_kind(DType::F8E5M2),
            Some(ElementKind::Fp8E5M2)
        );
        // Inbound: required by the round-trip invariant and by the caller that
        // reads a returned contract's FP8 operands back into Fuel dtypes.
        assert_eq!(
            element_kind_to_dtype(ElementKind::Fp8E4M3),
            Some(DType::F8E4M3)
        );
        assert_eq!(
            element_kind_to_dtype(ElementKind::Fp8E5M2),
            Some(DType::F8E5M2)
        );
    }

    use super::*;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_kernel_seam::{ArtifactKind, JitBudget, LinkEntry};
    use std::sync::{Arc, Mutex, RwLock as StdRwLock};

    fn noop_kernel(
        _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_ir::Layout],
        _params: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// abs(sub(a, b)) as a PatternNode region. Deliberately NOT the
    /// `relu(add(a, b))` shape every other adopted-op test in this crate
    /// uses (`fused_cost`, `runtime_fused_arm`, `runtime_fused_kernels`) —
    /// `register_runtime_fused`'s dedup index is a process-global sidecar
    /// shared by every `#[test]` in this binary (see
    /// `runtime_fused_pathfinder`'s `tanh_mul_region` doc comment for the
    /// full collision rationale): this file's `dtypes` is inputs-only
    /// (`operand_dtypes` over `req.operands`, which — like every other
    /// `CandidateKernel`/`JitRequest.operands` fixture in this crate — lists
    /// only the op's inputs, not its output), so a shared `relu_add()` slot
    /// whose winning row came from a 3-element (input+input+output) `dtypes`
    /// registration elsewhere would leave `adopts_a_synthesized_kernel_end_to_end`
    /// depending on `#[test]` scheduling order to avoid a mismatched-arity row.
    fn abs_sub() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Abs,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Sub,
                attrs: OpAttrs::default(),
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
            }],
        }
    }

    fn artifact(entry_point: &str) -> SynthArtifact {
        SynthArtifact {
            artifact: vec![0xCA, 0xFE],
            kind: ArtifactKind::Ptx,
            link: LinkEntry {
                entry_point: entry_point.into(),
                symbol: "k".into(),
                structure_key: "elementwise:f32".into(),
                revision_hash: 1,
            },
            contract: "## fused_op\ncost: n\n".into(),
        }
    }

    /// Mock synthesizer mirroring Baracuda's two-step handover.
    struct MockSynth {
        decline: bool,
        art: Mutex<Option<SynthArtifact>>,
    }
    impl Synthesizer for MockSynth {
        fn synthesize(&self, _req: &JitRequest) -> JitResponse {
            if self.decline {
                JitResponse::Declined {
                    reason: "mock decline".into(),
                }
            } else {
                JitResponse::Synthesized {
                    entry_point: "mock::abs_sub".into(),
                }
            }
        }
        fn take_kernel(&self, _entry_point: &str) -> Option<SynthArtifact> {
            self.art.lock().unwrap().take()
        }
    }

    fn req() -> JitRequest {
        JitRequest {
            region: abs_sub(),
            operands: vec![
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            ],
            arch: baracuda_kernels_types::ArchSku::Sm89,
            budget: JitBudget {
                max_compile_ms: 250,
            },
        }
    }

    #[test]
    fn adopts_a_synthesized_kernel_end_to_end() {
        let synth = MockSynth {
            decline: false,
            art: Mutex::new(Some(artifact("mock::abs_sub"))),
        };
        // The load_kernel seam: a real backend loads art.artifact as a module +
        // resolves art.link.symbol; here it just yields a no-op KernelRef.
        let adopted = adopt_from_response(&synth, &req(), BackendId::Cpu, |_art| {
            Ok(noop_kernel as KernelRef)
        })
        .expect("no error")
        .expect("synthesized ⇒ adopted");
        let id = adopted.id;

        assert!(id.is_runtime(), "adopted a runtime FusedOpId");
        // The kernel comes back with the id, and is the one THIS call loaded —
        // not whatever the recipe id happens to resolve to. See `Adopted`.
        assert!(
            std::ptr::fn_addr_eq(adopted.kernel, noop_kernel as KernelRef),
            "adoption returned the kernel this call loaded",
        );
        assert!(
            crate::runtime_fused_kernels::fused_kernel_available(id, BackendId::Cpu),
            "the adopted op's kernel is now visible to the capability gate",
        );
    }

    #[test]
    fn declined_synthesis_adopts_nothing() {
        let synth = MockSynth {
            decline: true,
            art: Mutex::new(None),
        };
        let out = adopt_from_response(&synth, &req(), BackendId::Cpu, |_art| {
            panic!("load_kernel must not run on a decline")
        })
        .expect("no error");
        assert!(out.is_none(), "declined ⇒ no adoption, no kernel load");
    }
}

//! The miss signal, read out of ordinary FKC contract matching.
//!
//! A **structural miss** is *definitionally* "at this dispatch key, the tightest
//! admissible contract is the GENERIC one — a structure-specialized cell would
//! have fit, but none is registered" (FKC §4.12). There is **no bolt-on miss
//! detector**: genericity is read off the winning contract's own retained
//! predicates ([`ResolvedLayout`]), and the demand token `wanted` is Baracuda's
//! structure key for the live operands (Fuel never derives it, K1 opacity).
//!
//! # Divergence from the plan (as-built, 2026-07-03)
//!
//! The plan (§4) reads the §4.2 structure-tightness predicates (`inner_div`,
//! `vec_width`) off each admitted contract. The **as-built** `ResolvedLayout`
//! (`fkc/caps_map.rs`) carries the *five-flag* set (`contiguous`, `strided`,
//! `broadcast_stride0`, `start_offset`, `reverse_strides`) — not `inner_div` /
//! `vec_width`. So [`is_generic_contract`] classifies genericity on the
//! five-flag set (permissive-strided ⇒ generic; contiguity-requiring ⇒
//! specialized). The tighter vec/inner-div predicates fold in when the FKC
//! schema grows them; the classifier is written to widen without a wire change.
//!
//! The live *ranker* `Candidate` (`ranker/candidate.rs`) carries only the
//! *projected* single-bool `KernelCaps` + `kernel_source`, not the retained
//! `ResolvedLayout`. Threading the layouts (or an `is_generic` bit) to the
//! dispatch site is the wiring that connects this detector to the live ranker;
//! the miss-first slice ships the detector + its unit proof, and that live
//! wire-in remains (see the module `sink` docs for the boundary).

use super::impl_id::ImplId;
use super::record::{HwStamp, MissRecord, TELEMETRY_SCHEMA_VERSION};
use super::structure_key::{FdxOperandDesc, StructureKeyProvider};
use crate::fkc::{ResolvedLayout, Tri};

/// Is this contract the GENERIC (fully-permissive strided) one — admissible
/// anywhere because it imposes no structure tightness?
///
/// Reads the retained FKC five-flag layout set: a generic contract accepts
/// strided + broadcast on **every** operand and requires contiguity on none.
/// A structure-specialized contract requires contiguity on some operand (or,
/// forward, a tighter vec/inner-div predicate the as-built five-flag set does
/// not yet carry — see module docs). An empty operand set is not generic (no
/// contract to fall back to).
pub fn is_generic_contract(layouts: &[ResolvedLayout]) -> bool {
    !layouts.is_empty()
        && layouts.iter().all(|l| {
            l.strided.is_accepted()
                && l.broadcast_stride0.is_accepted()
                && l.contiguous != Tri::Required
        })
}

/// One admitted contract as the miss detector reads it: its stable identity
/// plus the retained per-operand layout predicate set. Built at the matching
/// site from the FKC-lowered record; the detector reads tightness off
/// `layouts` — no separate detector state.
#[derive(Debug, Clone)]
pub struct AdmittedContract {
    /// The contract's stable [`ImplId`] (the fallback if it is generic).
    pub impl_id: ImplId,
    /// The retained FKC five-flag layout set, one entry per operand.
    pub layouts: Vec<ResolvedLayout>,
}

/// Read a [`MissRecord`] out of ordinary matching.
///
/// Emits a miss **iff** the BEST admissible contract at this dispatch site is
/// GENERIC — a specialized cell would have fit but none is registered. The
/// demand token `wanted` is the provider's structure key for the **live**
/// operands (Fuel calls Baracuda, never derives the token); `fallback` is the
/// generic contract's [`ImplId`]. Each returned record carries `count = 1` (a
/// single observation); the [`super::sink::TelemetrySink`] aggregates by
/// `(wanted, fallback, hw)` into a histogram.
///
/// Returns `None` when:
/// - the best admissible contract is structure-specialized (no miss), OR
/// - the provider yields no key (unlinked — the v1
///   [`super::structure_key::NullStructureKeyProvider`]): without a token there
///   is no demand signal to emit, never a fabricated one.
pub fn detect_miss(
    best: &AdmittedContract,
    op_class: &str,
    operands: &[FdxOperandDesc],
    arch: &str,
    provider: &dyn StructureKeyProvider,
    hw: HwStamp,
) -> Option<MissRecord> {
    // A specialized contract is admissible ⇒ no miss.
    if !is_generic_contract(&best.layouts) {
        return None;
    }
    // The best admissible match is generic. Ask Baracuda for the structure key
    // of the LIVE operands (Fuel calls, never derives). No key (unlinked) ⇒ no
    // demand signal — never a fabricated token.
    let wanted = provider.structure_key(op_class, operands, arch)?;
    Some(MissRecord {
        schema: TELEMETRY_SCHEMA_VERSION,
        wanted,
        fallback: best.impl_id.clone(),
        count: 1,
        hw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::structure_key::{Contiguity, StructureKeyToken};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::{BackendId, DType};

    /// Build one operand's five-flag [`ResolvedLayout`] from `(contiguous,
    /// strided)`; broadcast tracks `strided` (a permissive-strided contract
    /// accepts broadcast too), the rest rejected.
    fn layout(contiguous: Tri, strided: Tri) -> ResolvedLayout {
        ResolvedLayout {
            contiguous,
            strided,
            broadcast_stride0: strided,
            start_offset: Tri::Rejected,
            reverse_strides: Tri::Rejected,
        }
    }

    /// A test-only recording provider: returns a canned token derived from the
    /// operands (so a test can prove the operand structure reached the
    /// provider) and records the last call for inspection.
    struct CannedProvider {
        token: String,
    }
    impl StructureKeyProvider for CannedProvider {
        fn structure_key(
            &self,
            op_class: &str,
            operands: &[FdxOperandDesc],
            _arch: &str,
        ) -> Option<StructureKeyToken> {
            // Encode a couple of live axes into the token so tests can prove
            // the descriptor flowed in verbatim (Fuel returns it as-is).
            let flipped = operands.iter().any(|o| o.flipped);
            Some(StructureKeyToken(format!(
                "{op_class}:{}:flipped={flipped}",
                self.token
            )))
        }
    }

    fn impl_id(kernel_source: &str) -> ImplId {
        ImplId {
            backend: BackendId::Cuda,
            op: OpKind::MatMul,
            dtypes: vec![DType::F16, DType::F16, DType::F16],
            kernel_source: kernel_source.into(),
            kernel_revision_hash: 0xabc,
        }
    }

    fn hw() -> HwStamp {
        HwStamp {
            compute_capability: Some((8, 9)),
            hardware_sku: "NVIDIA GeForce RTX 4070".into(),
            driver_version: "552.44".into(),
        }
    }

    /// A fully-permissive strided contract (accepts strided + broadcast, no
    /// contiguity demand) is generic.
    fn generic_layouts() -> Vec<ResolvedLayout> {
        vec![layout(Tri::NotApplicable, Tri::Accepted); 2]
    }

    /// A contiguity-requiring contract is specialized (not generic).
    fn specialized_layouts() -> Vec<ResolvedLayout> {
        vec![layout(Tri::Required, Tri::Rejected); 2]
    }

    fn operand() -> FdxOperandDesc {
        FdxOperandDesc {
            dtype: DType::F16,
            contiguity: Contiguity::Contiguous,
            broadcast: false,
            flipped: false,
        }
    }

    #[test]
    fn generic_contract_classifier() {
        assert!(is_generic_contract(&generic_layouts()));
        assert!(!is_generic_contract(&specialized_layouts()));
        assert!(!is_generic_contract(&[]), "empty is not generic");
    }

    /// BORN-RED: a generic-only cell with a linked (stub) provider must emit a
    /// miss whose `wanted` is the provider's token and whose `fallback` is the
    /// generic contract's `ImplId`.
    #[test]
    fn generic_only_match_emits_a_miss_record() {
        let best = AdmittedContract {
            impl_id: impl_id("baracuda-generic-strided"),
            layouts: generic_layouts(),
        };
        let provider = CannedProvider { token: "mm".into() };
        let miss = detect_miss(&best, "matmul", &[operand()], "sm_89", &provider, hw())
            .expect("generic-only cell must emit a miss");
        assert_eq!(miss.schema, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(miss.fallback, impl_id("baracuda-generic-strided"));
        assert_eq!(miss.wanted, StructureKeyToken("matmul:mm:flipped=false".into()));
        assert_eq!(miss.count, 1);
        assert_eq!(miss.hw, hw());
    }

    /// A structure-specialized contract is admissible ⇒ no miss.
    #[test]
    fn tight_specialized_match_emits_no_miss() {
        let best = AdmittedContract {
            impl_id: impl_id("baracuda-mm-contig"),
            layouts: specialized_layouts(),
        };
        let provider = CannedProvider { token: "mm".into() };
        assert!(
            detect_miss(&best, "matmul", &[operand()], "sm_89", &provider, hw()).is_none(),
            "an admissible specialized contract is not a miss"
        );
    }

    /// An unlinked provider (no token) forms no demand signal ⇒ no miss, even
    /// on a generic-only cell.
    #[test]
    fn unlinked_provider_yields_no_miss() {
        use crate::telemetry::structure_key::NullStructureKeyProvider;
        let best = AdmittedContract {
            impl_id: impl_id("baracuda-generic-strided"),
            layouts: generic_layouts(),
        };
        assert!(
            detect_miss(&best, "matmul", &[operand()], "sm_89", &NullStructureKeyProvider, hw())
                .is_none(),
            "no structure key ⇒ no miss demand signal"
        );
    }

    /// The flipped demand axis flows end-to-end: a flipped operand reaches the
    /// provider (negative-strides-first-class), so the emitted `wanted` token
    /// reflects the flip. Proves the load-bearing axis is not laundered.
    #[test]
    fn flipped_operand_surfaces_in_wanted_token() {
        let best = AdmittedContract {
            impl_id: impl_id("baracuda-generic-strided"),
            layouts: generic_layouts(),
        };
        let flipped = FdxOperandDesc {
            flipped: true,
            contiguity: Contiguity::Strided,
            ..operand()
        };
        let provider = CannedProvider { token: "mm".into() };
        let miss = detect_miss(&best, "matmul", &[flipped], "sm_89", &provider, hw())
            .expect("generic-only cell must emit a miss");
        assert_eq!(
            miss.wanted,
            StructureKeyToken("matmul:mm:flipped=true".into()),
            "the flip must survive to the demand token"
        );
    }
}

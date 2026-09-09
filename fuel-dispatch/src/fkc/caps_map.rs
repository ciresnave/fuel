// SPDX-License-Identifier: MIT OR Apache-2.0
//! Five-flag layout set → today's [`KernelCaps`] projection (adoption
//! plan §6 / FKC §4.1, §12.2).
//!
//! FKC carries five independent layout facts per operand
//! (`contiguous`, `strided`, `broadcast_stride0`, `start_offset`,
//! `reverse_strides`); today's [`KernelCaps`] (`kernel.rs`) has exactly
//! one bool, `strided_input`. This module parses each tri-state string
//! to a typed [`Tri`] and projects per the **EXACT** §6 rule:
//!
//! ```text
//! KernelCaps.strided_input = (strided == accepted) && (broadcast_stride0 == accepted)
//! ```
//!
//! The other three flags are handled per as-built behavior:
//! - `start_offset` is parsed + retained but **NOT** projected — a
//!   non-zero `byte_offset` operand still routes through auto-Contiguize
//!   today (`kernel.rs` doc-comment). [consumer-ahead].
//! - `reverse_strides` is parsed + retained but **NOT** projected — the
//!   `KernelCaps` flag does not exist yet; a negative-stride operand is
//!   normalized by the planner until the field lands. [consumer-ahead].
//! - `contiguous` is parsed + retained (coherence + forward use).
//!
//! The importer **retains** every parsed flag on a [`ResolvedLayout`] so
//! nothing is lost; it emits only the `strided_input` projection today.
//! The moment `KernelCaps` grows `reverse_strides` / `start_offset_capable`
//! the retained values fill them (the forward-extension hook, §6).

use crate::fkc::error::FkcError;
use crate::fkc::schema::LayoutSpec;
use crate::kernel::KernelCaps;

/// A parsed layout tri-state (FKC §4.1). `contiguous` admits `Required`;
/// the other four flags use only `Accepted` / `Rejected`. `NotApplicable`
/// (`n/a`) is the "no constraint declared" value. An absent flag (YAML
/// key omitted) defaults to [`Tri::Rejected`] for the capability flags
/// (the conservative default matching today's all-false `KernelCaps`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    /// `required` — the kernel demands this property of the operand.
    Required,
    /// `accepted` — the kernel handles this property directly.
    Accepted,
    /// `rejected` — the kernel cannot consume this property (planner
    /// must normalize first).
    Rejected,
    /// `n/a` — no constraint declared.
    NotApplicable,
}

impl Tri {
    /// Parse one tri-state token. `flag`/`operand`/`section` give the
    /// error context. An absent value (`None`) maps to `default`.
    fn parse(
        value: Option<&str>,
        default: Tri,
        section: &str,
        operand: &str,
        flag: &str,
    ) -> Result<Tri, FkcError> {
        match value {
            None => Ok(default),
            Some(v) => match v.trim() {
                "required" => Ok(Tri::Required),
                "accepted" => Ok(Tri::Accepted),
                "rejected" => Ok(Tri::Rejected),
                "n/a" | "na" => Ok(Tri::NotApplicable),
                other => Err(FkcError::BadLayoutFlag {
                    section: section.to_string(),
                    operand: operand.to_string(),
                    flag: flag.to_string(),
                    value: other.to_string(),
                }),
            },
        }
    }

    /// Whether this flag is `accepted` (the projection predicate).
    pub fn is_accepted(self) -> bool {
        matches!(self, Tri::Accepted)
    }

    /// Whether this flag is `required` (the operand MUST carry the property;
    /// for `broadcast_stride0` this marks a baked-broadcast kernel).
    pub fn is_required(self) -> bool {
        matches!(self, Tri::Required)
    }
}

/// The full typed five-flag layout set for one operand — every flag
/// retained, even the ones not yet projected onto `KernelCaps` (§6
/// [consumer-ahead]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLayout {
    /// `required` | `accepted` | `n/a` for the contiguous property.
    pub contiguous: Tri,
    /// `accepted` | `rejected` — kernel walks explicit strides.
    pub strided: Tri,
    /// `accepted` | `rejected` — kernel handles a stride-0 broadcast axis.
    pub broadcast_stride0: Tri,
    /// `accepted` | `rejected` — kernel honors a non-zero start offset.
    /// Retained, not projected (auto-Contiguize handles it today).
    pub start_offset: Tri,
    /// `accepted` | `rejected` — kernel walks negative (reverse) strides.
    /// Retained, not projected (no `KernelCaps` field yet).
    pub reverse_strides: Tri,
}

impl ResolvedLayout {
    /// Project this operand's flags onto today's single-bool
    /// [`KernelCaps`] per the EXACT §6 rule:
    ///
    /// `strided_input = (strided == accepted) && (broadcast_stride0 == accepted)`.
    pub fn project(&self) -> KernelCaps {
        let strided_input = self.strided.is_accepted() && self.broadcast_stride0.is_accepted();
        // `broadcast_stride0: required` marks a baked-broadcast kernel: the
        // realize pick must exclude it from a dense operand (path 1a). See
        // [`KernelCaps::requires_broadcast`].
        let requires_broadcast = self.broadcast_stride0.is_required();
        KernelCaps {
            strided_input,
            requires_broadcast,
        }
    }
}

/// Parse one operand's [`LayoutSpec`] into a typed [`ResolvedLayout`].
/// An absent `layout` block (the `None` case) is the conservative
/// all-`rejected` set (matches today's default all-false `KernelCaps`).
pub fn resolve_layout(
    spec: Option<&LayoutSpec>,
    section: &str,
    operand: &str,
) -> Result<ResolvedLayout, FkcError> {
    let s = spec;
    Ok(ResolvedLayout {
        contiguous: Tri::parse(
            s.and_then(|l| l.contiguous.as_deref()),
            Tri::NotApplicable,
            section,
            operand,
            "contiguous",
        )?,
        strided: Tri::parse(
            s.and_then(|l| l.strided.as_deref()),
            Tri::Rejected,
            section,
            operand,
            "strided",
        )?,
        broadcast_stride0: Tri::parse(
            s.and_then(|l| l.broadcast_stride0.as_deref()),
            Tri::Rejected,
            section,
            operand,
            "broadcast_stride0",
        )?,
        start_offset: Tri::parse(
            s.and_then(|l| l.start_offset.as_deref()),
            Tri::Rejected,
            section,
            operand,
            "start_offset",
        )?,
        reverse_strides: Tri::parse(
            s.and_then(|l| l.reverse_strides.as_deref()),
            Tri::Rejected,
            section,
            operand,
            "reverse_strides",
        )?,
    })
}

/// Project a whole kernel's per-operand layouts onto **one** kernel-level
/// [`KernelCaps`]: a kernel can stride its inputs only if *every* input
/// operand accepts the strided + broadcast properties. (The binding
/// table carries one `KernelCaps` per kernel, not per operand; the
/// conservative AND is the faithful single-bool collapse — if any operand
/// must be contiguous, the kernel as a whole is not strided-capable.)
///
/// An empty operand list yields the all-false default.
pub fn project_kernel_caps(operand_layouts: &[ResolvedLayout]) -> KernelCaps {
    if operand_layouts.is_empty() {
        return KernelCaps::empty();
    }
    let strided_input = operand_layouts.iter().all(|l| l.project().strided_input);
    // If ANY operand must be broadcast (a baked stride-0), the KERNEL is a
    // baked-broadcast kernel — excluded from generic (dense) selection.
    let requires_broadcast = operand_layouts
        .iter()
        .any(|l| l.broadcast_stride0.is_required());
    KernelCaps {
        strided_input,
        requires_broadcast,
    }
}

/// Is this contract the GENERIC (fully-permissive strided) one — admissible
/// anywhere because it imposes no structure tightness?
///
/// Reads the retained FKC five-flag layout set: a generic contract accepts
/// strided + broadcast on **every** operand and requires contiguity on none.
/// A structure-specialized contract requires contiguity on some operand (or,
/// forward, a tighter vec/inner-div predicate the as-built five-flag set does
/// not yet carry). An empty operand set is not generic (no contract to fall
/// back to).
///
/// This is the classifier the Baracuda **miss** telemetry keys on (FKC §4.12):
/// the FKC importer computes it once from a kernel's retained
/// [`ResolvedLayout`] set at registration time and stamps the resulting bit
/// onto its [`crate::kernel::BindingEntry`], so the live dispatch pick site
/// reads a precomputed bool rather than re-deriving genericity from the lossy
/// single-bool [`KernelCaps`]. It lives here (not in the feature-gated
/// `telemetry` module) because genericity is a pure property of the
/// always-compiled `ResolvedLayout`, and the always-compiled FKC importer must
/// reach it without the `telemetry` feature. The `telemetry` module re-exports
/// it for the detector's own use.
pub fn is_generic_contract(layouts: &[ResolvedLayout]) -> bool {
    !layouts.is_empty()
        && layouts.iter().all(|l| {
            l.strided.is_accepted()
                && l.broadcast_stride0.is_accepted()
                && l.contiguous != Tri::Required
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(
        contiguous: Option<&str>,
        strided: Option<&str>,
        broadcast: Option<&str>,
        start_offset: Option<&str>,
        reverse: Option<&str>,
    ) -> LayoutSpec {
        LayoutSpec {
            contiguous: contiguous.map(String::from),
            strided: strided.map(String::from),
            broadcast_stride0: broadcast.map(String::from),
            broadcast_axes: None,
            start_offset: start_offset.map(String::from),
            reverse_strides: reverse.map(String::from),
            awkward_layout_strategy: None,
        }
    }

    fn resolve(s: &LayoutSpec) -> ResolvedLayout {
        resolve_layout(Some(s), "test", "op").expect("layout resolves")
    }

    #[test]
    fn required_broadcast_projects_requires_broadcast_not_strided() {
        // `broadcast_stride0: required` marks a baked-broadcast kernel: the
        // projection sets `requires_broadcast` (→ excluded from dense
        // selection) and does NOT set `strided_input` (Required != Accepted).
        let caps = resolve(&spec(
            Some("accepted"),
            Some("accepted"),
            Some("required"),
            Some("rejected"),
            Some("rejected"),
        ))
        .project();
        assert!(
            caps.requires_broadcast,
            "required broadcast → requires_broadcast"
        );
        assert!(
            !caps.strided_input,
            "Required is not Accepted → not strided-generic"
        );

        // A generic (broadcast_stride0: accepted) operand is dense-safe.
        let generic = resolve(&spec(
            Some("accepted"),
            Some("accepted"),
            Some("accepted"),
            Some("rejected"),
            Some("rejected"),
        ))
        .project();
        assert!(
            !generic.requires_broadcast,
            "accepted broadcast is generic, not baked"
        );
        assert!(generic.strided_input);
    }

    #[test]
    fn project_kernel_caps_requires_broadcast_if_any_operand_does() {
        let baked = resolve(&spec(
            Some("accepted"),
            Some("accepted"),
            Some("required"),
            Some("rejected"),
            Some("rejected"),
        ));
        let generic = resolve(&spec(
            Some("accepted"),
            Some("accepted"),
            Some("accepted"),
            Some("rejected"),
            Some("rejected"),
        ));
        assert!(
            project_kernel_caps(&[generic, baked]).requires_broadcast,
            "any baked operand ⇒ the kernel is baked-broadcast",
        );
        assert!(
            !project_kernel_caps(&[generic, generic]).requires_broadcast,
            "all-generic ⇒ dense-safe",
        );
    }

    #[test]
    /// FKC-CLAUSE: FKC-4.3-0001 FKC-4.3-0002 — the layout tri-states project to
    /// `strided_input`, which is what decides whether the planner inserts
    /// `Op::Contiguize`. `requires_contiguous` (contiguous: required) projects
    /// false -> planner inserts; `contiguize_internally`/`handles_strided`
    /// (strided: accepted) project true -> planner must NOT insert a separate
    /// Contiguize. Validator rule 5 enforces strategy/flag coherence, so the
    /// declared strategy reaches the planner THROUGH these flags.
    fn projection_truth_table() {
        // strided accepted + broadcast accepted ⇒ strided_input true.
        let r = resolve(&spec(
            Some("accepted"),
            Some("accepted"),
            Some("accepted"),
            Some("rejected"),
            Some("rejected"),
        ));
        assert!(r.project().strided_input);

        // strided rejected ⇒ false even if broadcast accepted.
        let r = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("accepted"),
            Some("rejected"),
            Some("rejected"),
        ));
        assert!(!r.project().strided_input);

        // broadcast rejected ⇒ false even if strided accepted.
        let r = resolve(&spec(None, Some("accepted"), Some("rejected"), None, None));
        assert!(!r.project().strided_input);

        // both rejected ⇒ false.
        let r = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("rejected"),
            Some("rejected"),
            Some("rejected"),
        ));
        assert!(!r.project().strided_input);
    }

    #[test]
    fn start_offset_does_not_flip_strided_input() {
        // start_offset accepted but strided rejected ⇒ still false
        // (auto-Contiguize handles offset today; §6).
        let r = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("rejected"),
            Some("accepted"),
            Some("rejected"),
        ));
        assert!(!r.project().strided_input);
        // …but the parsed value is retained.
        assert_eq!(r.start_offset, Tri::Accepted);
    }

    #[test]
    fn reverse_strides_retained_not_projected() {
        let r = resolve(&spec(
            None,
            Some("accepted"),
            Some("accepted"),
            None,
            Some("accepted"),
        ));
        // strided_input still derived only from strided+broadcast.
        assert!(r.project().strided_input);
        // reverse_strides parsed + retained for forward use.
        assert_eq!(r.reverse_strides, Tri::Accepted);
    }

    #[test]
    fn elementwise_binary_contract_projection_is_false() {
        // The real elementwise-binary layout: contiguous required, the
        // rest rejected ⇒ strided_input false.
        let r = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("rejected"),
            Some("rejected"),
            Some("rejected"),
        ));
        assert!(!r.project().strided_input);
        assert_eq!(r.contiguous, Tri::Required);
    }

    #[test]
    fn absent_layout_is_conservative_false() {
        let r = resolve_layout(None, "test", "op").unwrap();
        assert!(!r.project().strided_input);
        assert_eq!(r.strided, Tri::Rejected);
    }

    #[test]
    fn unknown_tristate_value_is_typed_error() {
        let s = spec(Some("maybe"), None, None, None, None);
        let err = resolve_layout(Some(&s), "test", "op").expect_err("bad value errors");
        assert!(matches!(err, FkcError::BadLayoutFlag { .. }), "got {err:?}");
    }

    #[test]
    fn kernel_caps_is_and_of_operands() {
        let strided = resolve(&spec(None, Some("accepted"), Some("accepted"), None, None));
        let contig = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("rejected"),
            None,
            None,
        ));
        // All strided ⇒ true.
        assert!(project_kernel_caps(&[strided, strided]).strided_input);
        // Mixed ⇒ false (conservative collapse).
        assert!(!project_kernel_caps(&[strided, contig]).strided_input);
    }

    #[test]
    fn is_generic_contract_classifies_permissive_vs_specialized() {
        // Fully-permissive strided (accepts strided + broadcast, no contiguity
        // demand) ⇒ generic.
        let generic = resolve(&spec(
            Some("n/a"),
            Some("accepted"),
            Some("accepted"),
            None,
            None,
        ));
        assert!(is_generic_contract(&[generic, generic]));
        // A contiguity-requiring operand ⇒ specialized, even if a sibling is
        // permissive (all operands must be permissive).
        let contig = resolve(&spec(
            Some("required"),
            Some("rejected"),
            Some("rejected"),
            None,
            None,
        ));
        assert!(!is_generic_contract(&[generic, contig]));
        assert!(!is_generic_contract(&[contig, contig]));
        // Empty is not generic (no contract to fall back to).
        assert!(!is_generic_contract(&[]));
    }

    /// ⚠️ **THE FORWARD-EXTENSION HOOK'S CLOCK.**
    ///
    /// This module's own doc says, of the flags it parses but does not project:
    ///
    /// > *"The moment `KernelCaps` grows `reverse_strides` /
    /// > `start_offset_capable` the retained values fill them (the
    /// > forward-extension hook, §6)."*
    ///
    /// **That hook fires only if somebody goes.** `reverse_strides` is parsed,
    /// retained and validated (it has its own `FkcError` arm) across ~1.9k
    /// contract sites and is projected NOWHERE. If a `KernelCaps` field lands
    /// and nobody wires the retained value, every one of those sites keeps
    /// being parsed, retained and ignored — **with nothing red.** A deferral
    /// whose trigger is *"when someone adds a field"* has no detector, because
    /// adding the field is not an event anything watches.
    ///
    /// # Why a destructuring and not an assertion
    ///
    /// The obvious guard — *assert a negative stride implies a non-zero
    /// `start_offset`* — would re-assert **planner behaviour that is already
    /// documented twice** (this module's doc, and `KernelCaps::strided_input`'s
    /// field doc in `kernel.rs`). **The unguarded thing is the HOOK, and you
    /// guard a hook by making the field's ARRIVAL loud — not by restating what
    /// the deferral defers to.**
    ///
    /// `KernelCaps` is deliberately not `#[non_exhaustive]` and its own doc
    /// invites growth: *"Forward-extensible by adding fields (no enum/bitflags
    /// churn)."* **So the struct invites exactly the change that silently
    /// orphans the retained values, and this pattern fails to COMPILE on the
    /// day it happens** — at the moment someone must decide whether the new
    /// field belongs to the hook. `error[E0027]` names the field it is missing.
    ///
    /// # The field's arrival is NOT silent, and that CHANGES what this is for
    ///
    /// Measured: adding `reverse_strides: bool` to `KernelCaps` breaks the
    /// build in SEVEN places before this test is even reached -- `E0063` at
    /// `caps_map.rs` x2 and `kernel.rs` x5. **So "nothing goes red" is false
    /// and this guard is not the only signal.**
    ///
    /// **What it adds is narrower and still worth having: `E0063` demands a
    /// VALUE, not a DECISION.** The path of least resistance at those sites is
    /// `reverse_strides: false` -- which compiles, is green, and silently
    /// orphans every value the importer has been retaining. **The pre-existing
    /// signal exists and points AWAY from the correct action.** This one fires
    /// in the file that owns the deferral, under the sentence saying what the
    /// retained values are for.
    ///
    /// AND RUSTC ITSELF SUGGESTS THE FIX THAT DISARMS THIS GUARD: `E0027`'s
    /// third help is *"or always ignore missing fields here"*, with a `..`
    /// rest pattern. **Taking it silently converts this test into one that can
    /// never fail again.** That is why the exhaustive form is spelled out, and
    /// why this paragraph exists rather than trusting the next reader to
    /// notice.
    ///
    /// **If you are here because this stopped compiling:** decide whether the
    /// new field is one the importer already retains a value for. If it is,
    /// wire it in `layout_caps` and delete the deferral note above. If it is
    /// not, add it to the pattern. **Do not delete this test to make the build
    /// green** — its failure IS the notification.
    ///
    /// Shape borrowed from baracuda, who hit the identical problem on
    /// `OpAttrs` and solved it the same way.
    #[test]
    fn kernel_caps_field_set_is_pinned_to_the_forward_extension_hook() {
        // Exhaustive by construction: a `..` rest pattern here would defeat the
        // entire point, and so would a field-count assertion — that is a
        // hand-written list of the kind whose staleness this file is about.
        let KernelCaps {
            strided_input: _,
            requires_broadcast: _,
        } = KernelCaps::default();
    }
}

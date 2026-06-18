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
        KernelCaps { strided_input }
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
    KernelCaps { strided_input }
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
            start_offset: start_offset.map(String::from),
            reverse_strides: reverse.map(String::from),
            awkward_layout_strategy: None,
        }
    }

    fn resolve(s: &LayoutSpec) -> ResolvedLayout {
        resolve_layout(Some(s), "test", "op").expect("layout resolves")
    }

    #[test]
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
        let r = resolve(&spec(
            None,
            Some("accepted"),
            Some("rejected"),
            None,
            None,
        ));
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
        let contig = resolve(&spec(Some("required"), Some("rejected"), Some("rejected"), None, None));
        // All strided ⇒ true.
        assert!(project_kernel_caps(&[strided, strided]).strided_input);
        // Mixed ⇒ false (conservative collapse).
        assert!(!project_kernel_caps(&[strided, contig]).strided_input);
    }
}

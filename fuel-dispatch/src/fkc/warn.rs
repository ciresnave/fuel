//! Soft, non-fatal FKC import diagnostics.
//!
//! FKC's importer surfaces *failures* as typed [`crate::fkc::FkcError`] values.
//! This module carries the complementary *warnings* — soft-degradation notices
//! (§3.5 "importer warns, does not reject", a precision claim downgraded for
//! lack of a verification-ledger entry, a shape_constraint that resolved to
//! seed shapes) that must NOT fail the import but MUST be visible to a caller.
//! Collected on [`crate::fkc::register::ImportedProvider::warnings`].

/// A single non-fatal FKC import diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportWarning {
    /// The kernel section (`## name`) the warning is about.
    pub section: String,
    /// Human-readable description of the soft-degradation that occurred.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_warning_constructs_and_carries_section_and_message() {
        let w = ImportWarning { section: "add_f32".into(), message: "downgraded bit_stable".into() };
        assert_eq!(w.section, "add_f32");
        assert!(w.message.contains("downgraded"));
        assert_eq!(w.clone(), w); // Clone + PartialEq derive present
    }
}

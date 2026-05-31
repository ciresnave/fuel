//! `AlternativeFilter` trait + `FilterClass` + `FilterContext`.
//!
//! Phase 1.1 of the picker-work arc. The trait shape is the
//! contract every concrete filter in Phase 1.3+ (precision floor,
//! tolerance budget, strided-input preference, bit-stable
//! preference, Judge fastest) implements.
//!
//! Trait object dispatch (`Vec<Box<dyn AlternativeFilter>>`) is the
//! deliberate choice — plan time isn't a hot loop, the heap cost is
//! one allocation per filter per decision point, and runtime
//! composition lets Phase 3's Judge filter slot in without touching
//! the consumer's chain construction.

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::{DType, Layout};

use super::candidate::Candidate;

/// One filter in the optimizer ranker's chain. Given the current
/// candidate list + context, returns the indices of candidates to
/// keep. The chain applies filters in order; each one sees the
/// survivors of the previous.
///
/// # Implementor contract
///
/// - **Pure function of inputs.** Don't read globals or mutate
///   external state — the filter is called at plan time, possibly
///   from a compiler thread, and idempotency matters for plan-cache
///   reuse.
/// - **Sorted, distinct, in-range indices.** The returned `Vec<usize>`
///   must be sorted strictly ascending, contain only indices `<
///   alts.len()`, and have no duplicates. The chain pipeline calls
///   [`crate::ranker::AlternativeSet::retain_indices`] which
///   `debug_assert!`s these invariants.
/// - **Empty result = "I reject everything."** Soft filters that
///   would over-restrict are skipped by the pipeline; hard filters
///   are surfaced as `Error::FilterRejected`. Filters themselves
///   don't need to know which class they're in — just return what
///   they think is admissible.
pub trait AlternativeFilter: Send + Sync {
    /// Apply this filter against the candidate list. Return the
    /// indices to keep (sorted ascending, distinct, in-range).
    fn filter(&self, alts: &[Candidate], ctx: &FilterContext) -> Vec<usize>;

    /// Is this a hard filter (may filter to zero, surfaces as error)
    /// or soft (may only filter if at least `min_remaining` remain)?
    fn classification(&self) -> FilterClass;

    /// Short identifier shown in diagnostics. Convention: short
    /// kebab-case ("precision-floor", "strided-input-pref",
    /// "judge-fastest"). Used in [`fuel_core_types::Error::FilterRejected`].
    fn name(&self) -> &'static str;
}

/// How aggressive a filter is allowed to be when shrinking the
/// candidate set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterClass {
    /// The filter may filter to zero. If it does, the pipeline
    /// surfaces [`fuel_core_types::Error::FilterRejected`] — the
    /// user asked for something the binding-table can't deliver and
    /// we surface it rather than silently substituting a
    /// non-admissible alternative.
    ///
    /// Use for filters that encode a user-requested correctness
    /// floor: precision bound, tolerance budget, explicit backend
    /// pin (if a future filter encodes that). Not for preferences.
    Hard,

    /// The filter must leave at least `min_remaining` candidates.
    /// If applying it would drop below this threshold, the pipeline
    /// skips this filter for this call (logs that it saturated).
    ///
    /// `min_remaining = 1` is the typical preference shape: "narrow
    /// the set when possible, but don't leave nothing." Larger
    /// values reserve diversity for downstream filters that might
    /// further narrow — e.g., a strided-input-preference filter
    /// with `min_remaining = 2` ensures the subsequent Judge filter
    /// still has at least two options to rank empirically.
    Soft { min_remaining: usize },
}

/// Read-only side-channel the filter chain passes to each
/// [`AlternativeFilter`]. The fields are the union of what any
/// shipped or planned filter needs; new fields are additive.
///
/// Borrowed lifetime — the context lives for the duration of one
/// `apply_filter_chain` call, no allocation beyond what the caller
/// already has on hand.
#[derive(Clone, Copy, Debug)]
pub struct FilterContext<'a> {
    /// Op kind at this decision point. Filters that index by op
    /// (a fused-op-specific preference, a Judge lookup) read this.
    pub op: OpKind,
    /// Per-operand dtype list — inputs in order, then outputs.
    /// Matches the binding-table lookup key shape.
    pub dtypes: &'a [DType],
    /// Layouts of the input operands. Caps-preference filters read
    /// this to decide whether to prefer kernels that advertise
    /// `strided_input`.
    pub input_layouts: &'a [Layout],
    /// Optional caller-supplied diagnostic tag. The pipeline echoes
    /// it into `Error::FilterRejected`'s `ctx_summary` when a hard
    /// filter rejects everything.
    pub tag: Option<&'a str>,
}

impl<'a> FilterContext<'a> {
    /// Build a context with no tag — the common case.
    pub fn new(op: OpKind, dtypes: &'a [DType], input_layouts: &'a [Layout]) -> Self {
        Self {
            op,
            dtypes,
            input_layouts,
            tag: None,
        }
    }

    /// Builder-style: attach a diagnostic tag (e.g. the graph
    /// `NodeId`'s display form) for richer error messages.
    pub fn with_tag(mut self, tag: &'a str) -> Self {
        self.tag = Some(tag);
        self
    }

    /// Short diagnostic summary used in `Error::FilterRejected`'s
    /// `ctx_summary` field. Format chosen for grep-ability —
    /// `op=MatMul dtypes=[F32,F32,F32] layouts=2 tag=NodeId(42)`.
    pub fn summary(&self) -> String {
        let layouts = self.input_layouts.len();
        match self.tag {
            Some(t) => format!(
                "op={:?} dtypes={:?} input_layouts={} tag={}",
                self.op, self.dtypes, layouts, t,
            ),
            None => format!(
                "op={:?} dtypes={:?} input_layouts={}",
                self.op, self.dtypes, layouts,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;

    /// Mock filter for tests — keeps exactly the indices it was
    /// configured with, regardless of input.
    pub(crate) struct MockFilter {
        pub keep: Vec<usize>,
        pub class: FilterClass,
        pub name: &'static str,
    }

    impl AlternativeFilter for MockFilter {
        fn filter(&self, _alts: &[Candidate], _ctx: &FilterContext) -> Vec<usize> {
            self.keep.clone()
        }
        fn classification(&self) -> FilterClass {
            self.class
        }
        fn name(&self) -> &'static str {
            self.name
        }
    }

    #[test]
    fn filter_trait_is_dyn_compatible() {
        // The trait must be usable as a trait object for the
        // `Vec<Box<dyn AlternativeFilter>>` chain shape; this is a
        // compile-time check encoded as a runtime construction.
        let f: Box<dyn AlternativeFilter> = Box::new(MockFilter {
            keep: vec![0, 2],
            class: FilterClass::Soft { min_remaining: 1 },
            name: "mock",
        });
        let layouts = [Layout::contiguous(Shape::from(vec![4]))];
        let ctx = FilterContext::new(OpKind::AddElementwise, &[DType::F32], &layouts);
        assert_eq!(f.filter(&[], &ctx), vec![0, 2]);
        assert_eq!(f.name(), "mock");
        assert!(matches!(f.classification(), FilterClass::Soft { min_remaining: 1 }));
    }

    #[test]
    fn filter_context_summary_includes_op_and_dtypes() {
        let layouts = [Layout::contiguous(Shape::from(vec![4]))];
        let ctx = FilterContext::new(
            OpKind::MatMul,
            &[DType::F32, DType::F32, DType::F32],
            &layouts,
        );
        let s = ctx.summary();
        assert!(s.contains("MatMul"));
        assert!(s.contains("F32"));
        assert!(s.contains("input_layouts=1"));
    }

    #[test]
    fn filter_context_with_tag_appears_in_summary() {
        let layouts: [Layout; 0] = [];
        let ctx = FilterContext::new(OpKind::AddElementwise, &[DType::F32], &layouts)
            .with_tag("NodeId(42)");
        assert!(ctx.summary().contains("NodeId(42)"));
    }

    #[test]
    fn filter_class_variants_distinguishable() {
        assert_ne!(
            FilterClass::Hard,
            FilterClass::Soft { min_remaining: 1 },
        );
        assert_ne!(
            FilterClass::Soft { min_remaining: 1 },
            FilterClass::Soft { min_remaining: 2 },
        );
    }
}

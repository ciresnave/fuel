//! Optimizer ranker (Picker 1) substrate — Phase 1.1 of the
//! picker-work arc.
//!
//! Builds the types + filter-chain pipeline the in-flight optimizer
//! ranker will operate against. No SystemTopology integration, no
//! candidate enumeration, no cost composition, no filter
//! implementations — all of that lands in subsequent sub-phases:
//!
//! | Sub-phase | Scope                                                                  |
//! |-----------|------------------------------------------------------------------------|
//! | **1.1**   | This module — `AlternativeSet`, `Candidate`, `AlternativeFilter`, chain |
//! | 1.2       | Candidate enumeration via SystemTopology (cross co-located backends)   |
//! | 1.3       | Precision + tolerance hard filters (`PrecisionGuarantee` consumers)    |
//! | 1.4       | Cost ranking — Layer-1 static composition                              |
//! | 1.5       | `ExecutionPlan` carries `AlternativeSet`s; `compile_plan` integration  |
//!
//! See [`docs/session-prompts/phase-1-1-alternative-set-filter-chain.md`]
//! for the TDP resolutions and the full Phase 1 plan.
//!
//! # The two pickers
//!
//! Per the 2026-05-30 picker-alternatives audit, the architectural
//! endpoint is *two* pickers with very different scopes:
//!
//! - **Picker 1 — the optimizer ranker (this module's eventual
//!   consumer).** Plan-time machinery that enumerates candidates via
//!   SystemTopology, filters by hard correctness constraints, ranks
//!   survivors by composite cost (Layer-1 static + Layer-2 Judge
//!   data), and preserves the top-N per architecture v1.0 §04.
//! - **Picker 2 — the runtime selector.** Dispatch-time component
//!   that reads pre-blessed `AlternativeSet`s and selects among the
//!   top-N based on layer-3 telemetry, checking SystemTopology
//!   generation before each dispatch chunk.
//!
//! Picker 1 lands across Phases 1–3; Picker 2 is Phase 5.

pub mod alternative_set;
pub mod candidate;
pub mod chain;
pub mod chained_selector;
pub mod cost;
pub mod enumerate;
pub mod filter;
pub mod filters;
pub mod judge;
pub mod judge_aware_selector;
pub mod runtime_selector;
pub mod vram_pressure_selector;

pub use alternative_set::{AlternativeSet, DecisionContext, DEFAULT_MAX_N};
pub use candidate::{Candidate, CouplingAdjustment};
pub use chain::apply_filter_chain;
pub use chained_selector::ChainedSelector;
pub use cost::{composite_ns, compute_static_costs, CapabilitiesLookup};
pub use enumerate::{enumerate_candidates, enumerate_candidates_default};
pub use filter::{AlternativeFilter, FilterClass, FilterContext};
pub use filters::{
    default_chain, BitStablePreferenceFilter, PrecisionFloorFilter, PrecisionRequirement,
    StridedInputPreferenceFilter,
};
pub use judge::{HashMapJudge, JudgeOracle};
pub use judge_aware_selector::JudgeAwareSelector;
pub use runtime_selector::{RuntimeSelector, WinnerSelector};
pub use vram_pressure_selector::{
    default_estimate_output_bytes, BackendRuntimeHandle, BackendRuntimeLookup,
    OutputBytesEstimator, VramPressureSelector,
};

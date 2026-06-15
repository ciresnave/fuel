//! # fuel-dispatch
//!
//! Dispatch infrastructure for fuel — extracted from fuel-storage
//! 2026-05-31 per the picker-work phasing
//! ([docs/session-prompts/dispatch-move-to-fuel-core.md](
//! ../../docs/session-prompts/dispatch-move-to-fuel-core.md)).
//!
//! ## What lives here
//!
//! - **`KernelBindingTable` + registration wrappers** — backends
//!   register their per-`(op, dtypes, BackendId)` kernels into a
//!   process-wide table. The picker / optimizer queries this table
//!   to enumerate candidate alternatives at each graph decision point.
//! - **`CompiledNode` + `compile_node`** — the dispatch-time
//!   `(KernelRef, KernelCaps, OpParams)` resolution result that the
//!   executor invokes.
//! - **`ExecutionPlan` + `compile_plan` + `PlanOptions`** — Phase
//!   1.5 reshape of the plan-time picker around per-decision-point
//!   `AlternativeSet`s. Replaced the pre-1.5 `NodeKernelBinding`/
//!   `TolerancePolicy`/`resolve_kernel` triple (which had zero
//!   executor consumers; the verified-empty consumer list let the
//!   rewrite ship without breakage).
//! - **`FusedKernelRegistry`** + `PrecisionGuarantee` +
//!   `KernelRevisionHash` — fused-op dispatch substrate.
//! - **`PipelinedExecutor`** — the production executor that walks a
//!   graph, calls `compile_node` per kernel-bearing node, and runs
//!   the resolved `KernelRef` against the input/output Storage Arcs.
//! - **Cost functions** — Layer-1 static cost estimates per op
//!   family; the optimizer composes these along candidate routes.
//! - **Cast fusion rule** — cast-elision graph rewrite (lives near
//!   dispatch because it inspects binding-table coverage).
//!
//! ## What's NOT here
//!
//! - `BackendStorage` enum + `Storage` wrapper — stays in `fuel-storage`
//!   until retired via Phase 0.2c (move to `fuel-core-types`).
//! - `SystemTopology` + `Judge` + `ProbeReport` — stays in `fuel-core`
//!   today; Phase 1's optimizer ranker will decide whether to relocate.
//! - Backend-specific kernels themselves — those live in their backend
//!   crates (fuel-cpu-backend, fuel-cuda-backend, fuel-vulkan-backend).
//!   This crate hosts the dispatch *wrappers* that bridge erased
//!   `Storage` ↔ typed backend storage.

pub mod baracuda_dispatch;
pub mod cast_fusion;
pub mod compiled;
pub mod cost;
pub mod dispatch;
pub mod fused;
pub mod kernel;
pub mod optimize;
pub mod pipelined;
pub mod plan;
pub mod plan_store;
pub mod ranker;
pub mod residency;
pub mod vulkan_dispatch;

pub use compiled::{compile_node, execute_compiled, CompiledNode};
pub use kernel::{KernelBindingTable, KernelDTypes, KernelRef, OpParams};
pub use pipelined::PipelinedExecutor;
pub use optimize::{optimize_graph, OptimizedGraph};
pub use plan::{compile_plan, ExecutionPlan, PlanOptions};
pub use ranker::{
    apply_filter_chain, apply_inbound_transfer_costs, composite_ns,
    compute_static_costs, default_chain, enumerate_candidates,
    enumerate_candidates_default, AlternativeFilter, AlternativeSet,
    BitStablePreferenceFilter, CapabilitiesLookup, Candidate,
    CouplingAdjustment, FilterClass, FilterContext, HashMapJudge, JudgeOracle,
    PrecisionFloorFilter, PrecisionRequirement, StridedInputPreferenceFilter,
    TransferEstimator, DEFAULT_MAX_N,
};
pub use residency::{
    insert_residency_evictions, EvictReload, LiveRange, ResidencyPlanner,
    ResidencyReport,
};

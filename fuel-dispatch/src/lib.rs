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
//! - **`ExecutionPlan` + `compile_plan` + `resolve_kernel` +
//!   `TolerancePolicy`** — Phase 7.6 step 9b's plan-time picker.
//!   Replaced wholesale by the Phase 1 optimizer ranker; lives here
//!   until that replacement lands.
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
pub mod pipelined;
pub mod plan;
pub mod ranker;
pub mod vulkan_dispatch;

pub use compiled::{compile_node, execute_compiled, CompiledNode};
pub use kernel::{KernelBindingTable, KernelDTypes, KernelRef, OpParams};
pub use pipelined::PipelinedExecutor;
pub use plan::{compile_plan, resolve_kernel, ExecutionPlan, NodeKernelBinding, TolerancePolicy};
pub use ranker::{
    apply_filter_chain, AlternativeFilter, AlternativeSet, Candidate, CouplingAdjustment,
    FilterClass, FilterContext, DEFAULT_MAX_N,
};

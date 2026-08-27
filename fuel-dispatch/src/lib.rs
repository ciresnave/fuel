// SPDX-License-Identifier: MIT OR Apache-2.0
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

// GAP-229: `clippy::identity_op` fires 128x across fuel-core+fuel-dispatch and is a
// defect in 0 of 128 — it measures a house idiom, not debt, so it is allowed at the
// crate root (a MEASURED claim: "identity ops in THIS crate are intentional"; a firing
// in a third crate is a deliberate TRIPWIRE — it reds the gate so someone looks and rules).
// Two intentional classes:
//   * DOC-INDEX: an explicit `0 *`/`1 *` NAMES an index (`idx[i*nb + 0]`, `got[1*head_dim + j]`);
//     the `0 *` partner is separately load-bearing (CLAUDE.md: never delete it), so its `1 *`
//     sibling must survive too — auto-fixing it would strand a bare unexplained `0 *`.
//   * DOC-SHAPE: an explicit unit/batch dim in a shape product (`1 * C * H * W`) mirrors
//     `Shape::from_dims(&[1, C, H, W])`.
// RESIDUAL, named not gated: this ALSO hides FLOAT identity ops, where `x + 0.0` normalizes
// -0.0 -> +0.0 — a real hazard whose population is ZERO today (measured). A float identity op
// landing later is silently admitted; re-measure at the next rust-toolchain.toml pin bump
// (owner-tracked, docs/gaps.md GAP-229) and drop this allow if precision is no longer 0/N.
// GAP-242. `fuel_memory::BackendStorage` gates EVERY non-CPU variant behind a
// feature -- `Cuda` on `cuda`, `Vulkan` on `vulkan`, `Metal` on `metal` plus an
// Apple target -- so with all backend features OFF the enum has exactly ONE
// variant and every `if let BackendStorage::Cpu(..)` in this crate is
// irrefutable BY CONSTRUCTION rather than by mistake.
//
// MEASURED at head 2026-08-27, `cargo clippy -p fuel-dispatch --all-targets`:
//   109 sites, ALL of them `BackendStorage::Cpu`
//        103 pipelined.rs · 2 compiled.rs · 2 kernel.rs · 1 residency.rs
//          1 dispatch.rs   -- every one in `src/`, so this crate attribute
//                             reaches all of them
//     0 sites under `--features vulkan --all-targets`
// The class is ABSENT when any backend feature is on, not merely quieter, and
// the site count is re-derivable from those two commands.
//
// NOT REWRITTEN, deliberately. There is no correctness gain, and the enum's
// arity is `cfg`-controlled: an edit that is correct for the configuration it
// was compiled in can break a different feature set. `--fix` on this lint
// rewrites control flow, which is exactly that hazard.
//
// NOT `#[expect]`, and the reason is specific to this lint rather than
// inherited: an expectation must be fulfilled in EVERY compilation, and this
// class does not fire at all under any backend feature (measured: 0). A crate
// `#[expect(irrefutable_let_patterns)]` would therefore be UNFULFILLED on every
// build that enables `vulkan` or `cuda` -- including CI's `--workspace` job,
// which enables `fuel-dispatch/vulkan`. It would trade 109 warnings in one
// configuration for 1 in all the others.
//
// This allow is honest only because CI now runs a DEFAULT-FEATURES
// `clippy -p fuel-dispatch --all-targets` leg. Without that leg it would
// silence a lint no gate was running; with it, it is a documented exception
// inside a gate that is watching. Those are different artifacts wearing the
// same syntax.
//
// `unreachable_patterns` rides along for the SAME reason and is the stronger
// case of the two. `pipelined.rs:8969` is a `match` on `BackendStorage` with a
// `Cpu(c) => ..` arm and a `_ => panic!("expected CPU output")` catch-all; with
// the enum single-variant the `_` is unreachable. But that arm is REQUIRED the
// moment any backend feature is on -- "fixing" the lint means deleting it,
// which breaks the `vulkan` build outright. Measured: 1 site at default
// features, 0 under `--features vulkan --all-targets`.
#![allow(irrefutable_let_patterns)]
#![allow(unreachable_patterns)]
#![allow(clippy::identity_op)]

pub mod baracuda_dispatch;
pub mod cast_fusion;
pub mod compiled;
pub mod cost;
pub mod decode_flash;
pub mod dispatch;
pub mod driver;
pub mod fkc;
pub mod fused;
pub mod fused_cost;
#[cfg(feature = "jit")]
pub mod jit_adopt;
#[cfg(feature = "jit")]
pub mod jit_carrier;
#[cfg(all(feature = "jit", feature = "cuda"))]
pub mod jit_cuda_load;
#[cfg(feature = "jit")]
mod jit_ingest;
#[cfg(feature = "jit")]
pub use jit_ingest::{
    CandidateKernel, FlagReport, IngestOutcome, IngestionConfig, IngestionService,
    ProviderFeedback, RejectionReport,
};
#[cfg(feature = "jit")]
mod jit_ingest_probe;
pub mod kernel;
/// Reader for the vendored KISS conformance corpus (staged for the corrected
/// `corpus_verdict` seam; see the module doc + its `PROVENANCE.md`).
#[cfg(feature = "jit")]
mod kiss_corpus;
pub mod optimize;
pub mod pipelined;
pub mod plan;
pub mod ranker;
pub mod residency;
pub mod runtime_fused_arm;
pub mod runtime_fused_kernels;
pub mod runtime_fused_pathfinder;
#[cfg(feature = "telemetry")]
pub mod telemetry;
pub mod topology;
pub mod variant_bake;
pub mod vulkan_dispatch;

pub use compiled::{
    CompiledNode, Completion, CompletionHandle, compile_node, dispatched_kernel_ident,
    execute_compiled,
};
pub use driver::{
    FrontierConvergenceOptimizer, OptimizationContext, Optimizer, PassRegistry, Pathfinder,
    PlacementForkPathfinder,
};
pub use kernel::{KernelBindingTable, KernelDTypes, KernelRef, OpParams};
pub use optimize::{OptimizedGraph, optimize_graph};
pub use pipelined::PipelinedExecutor;
pub use plan::{ExecutionPlan, PlanOptions, compile_plan};
pub use ranker::{
    AlternativeFilter, AlternativeSet, BitStablePreferenceFilter, Candidate, CapabilitiesLookup,
    CouplingAdjustment, FilterClass, FilterContext, HashMapJudge, JudgeOracle, KEEP_PER_DEVICE,
    PrecisionFloorFilter, PrecisionRequirement, StridedInputPreferenceFilter, TransferEstimator,
    apply_filter_chain, apply_inbound_transfer_costs, composite_ns, compute_static_costs,
    default_chain, enumerate_candidates,
};
pub use residency::{
    EvictReload, LiveRange, ResidencyPlanner, ResidencyReport, insert_residency_evictions,
};

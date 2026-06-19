//! The canonical stable `ImplId` — the dispatch-telemetry / specialization basis.
//!
//! The basis tuple IS Fuel's kernel identity (FKC §4.11): `(BackendId, op,
//! dtypes, kernel_source, kernel_revision_hash)`. **No new identifier is
//! invented** — every field already exists on the dispatch surface, and every
//! field is serializable data (no function pointer). A telemetry record's impl
//! id and the Judge's measurement key are the same `kernel_source` axis, by
//! construction, so a record captured on one build re-resolves on another.
//!
//! The `classify()` projection onto Baracuda's `{Baracuda|Vendor|FuelNative}`
//! wire form, and the `from_binding`/`from_resolved_primitive` constructors,
//! land in step 2; this module defines the serializable identity itself.

use fuel_core_types::dispatch::OpKind;
use fuel_core_types::{BackendId, DType};

/// The stable, pointer-free implementation id. Basis tuple = FKC kernel
/// identity. Serialized into every `DispatchRecord`/`Candidate`/`MissRecord`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImplId {
    /// The backend the kernel runs on (`Cuda` for a Baracuda kernel).
    pub backend: BackendId,
    /// The Fuel op this kernel implements (a fused-op tag for fused contracts).
    pub op: OpKind,
    /// Operand dtypes, inputs-in-order then outputs (the binding-table key axis).
    pub dtypes: Vec<DType>,
    /// The implementation-source discriminant (`"baracuda"`, `"cublas"`,
    /// `"portable-cpu"`, …) — the same tag the Judge keys its timings on.
    pub kernel_source: String,
    /// Stable per-implementation-version hash; pins the revision so a persisted
    /// plan / telemetry record re-resolves to the exact kernel build. `0` =
    /// untracked (non-FKC kernels until the revision is threaded — step 2).
    pub kernel_revision_hash: u64,
}

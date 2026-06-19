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

/// `kernel_source` tags that denote a third-party vendor library (not Baracuda,
/// not a Fuel-native portable kernel). The discriminant for the `Vendor` arm.
const VENDOR_SOURCES: &[&str] =
    &["cublas", "cudnn", "cutlass", "rocblas", "mkl", "aocl", "onednn"];

/// Baracuda's wire form for an implementation id (FKC §4.11 mapping). The
/// discriminant is `kernel_source` — no reconciliation table. Borrows from the
/// [`ImplId`] so classification is allocation-free.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplClass<'a> {
    /// A Baracuda CUDA kernel (`kernel_source == "baracuda"`). `symbol` is the
    /// kernel entry-point symbol; in v1 it is the `kernel_source`-derived tag
    /// until the FKC `entry_point` is threaded to the dispatch site.
    Baracuda { symbol: &'a str },
    /// A third-party vendor library kernel (cuBLAS / cuDNN / MKL / AOCL / …).
    Vendor { which: &'a str },
    /// A Fuel-native portable kernel (`"portable-cpu"`, `"slang"`, …).
    FuelNative { which: &'a str },
}

impl ImplId {
    /// Project this id onto Baracuda's `{ Baracuda | Vendor | FuelNative }` wire
    /// form. Classification is by `kernel_source` alone (the `"baracuda"` tag
    /// only ever occurs on `BackendId::Cuda`), so no backend-vs-source
    /// reconciliation is needed.
    pub fn classify(&self) -> ImplClass<'_> {
        let src = self.kernel_source.as_str();
        if src == "baracuda" {
            ImplClass::Baracuda { symbol: src }
        } else if VENDOR_SOURCES.contains(&src) {
            ImplClass::Vendor { which: src }
        } else {
            ImplClass::FuelNative { which: src }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::dispatch::OpKind;

    fn id(kernel_source: &str, backend: BackendId) -> ImplId {
        ImplId {
            backend,
            op: OpKind::MatMul,
            dtypes: vec![DType::F16, DType::F16, DType::F16],
            kernel_source: kernel_source.into(),
            kernel_revision_hash: 0xabc,
        }
    }

    #[test]
    fn baracuda_cuda_kernel_classifies_as_baracuda() {
        assert_eq!(
            id("baracuda", BackendId::Cuda).classify(),
            ImplClass::Baracuda { symbol: "baracuda" },
        );
    }

    #[test]
    fn vendor_sources_classify_as_vendor() {
        for v in ["cublas", "cudnn", "mkl", "aocl"] {
            assert_eq!(
                id(v, BackendId::Cuda).classify(),
                ImplClass::Vendor { which: v },
                "{v} must classify as Vendor",
            );
        }
    }

    #[test]
    fn portable_and_unknown_classify_as_fuel_native() {
        assert_eq!(
            id("portable-cpu", BackendId::Cpu).classify(),
            ImplClass::FuelNative { which: "portable-cpu" },
        );
        assert_eq!(
            id("slang", BackendId::Vulkan).classify(),
            ImplClass::FuelNative { which: "slang" },
        );
    }
}

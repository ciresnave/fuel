//! `cuda_storage_status` — the single authority for *"does the CUDA backend have
//! STORAGE for this dtype"*, plus the decline-error helper that single-sources
//! the reason from it (GAP-161).
//!
//! Before this, several construction sites (`storage.rs`, `device.rs`) spelled
//! the reason a dtype declines in a *comment* and then returned the identical
//! reasonless `UnsupportedDtype` for a dtype that is representable-but-unwired
//! (F8E5M2) and one whose encoding is unauthored (F8E6M2). This module makes the
//! reason a typed value on the [`StorageStatus`] WHY axis, decided in one place.

use crate::error::CudaError;
use fuel_ir::{DType, StorageStatus};

/// The ONE place the CUDA backend decides whether a [`DType`] has device
/// storage, and — for a decline — the WHY-axis reason (GAP-161). Storage
/// construction sites derive their accept/decline decision and their decline
/// *reason* from here instead of each restating the distinction.
///
/// Exhaustive on [`DType`] with **no wildcard**: adding a dtype is a compile
/// error here until it is classified, so no dtype is silently unhandled.
///
/// Produces no [`StorageStatus::Impossible`]: no dtype currently in Fuel's
/// `DType` enum is *permanently* unstorable on CUDA — any representable dtype
/// could gain a byte variant. Permanent declines are a fact about the token
/// vocabulary (reserved-by-standard spellings like `f8e4m3fnuz`), which are not
/// `DType` variants and are classified at the recognition/emit surface (the KIND
/// axis), not here.
///
/// ## Known limit (GAP-161) — read before trusting this as a capability trap
///
/// This is the single authority for the *reason*, but is not an airtight trap
/// for the *capability*. The airtight form is a macro that co-generates the
/// [`CudaStorageSlice`](crate::storage::CudaStorageSlice) variants AND the
/// `Present` arms below from one list, so that wiring storage for a dtype is
/// impossible without flipping its classification here. Without it, someone
/// could add a `CudaStorageSlice::F8E5M2` variant and wire it through a
/// construction site that does not gate on this function, leaving the
/// classification stale while the capability works. The `cargo test` expiry
/// canary (`f8e5m2_storage_is_unimplemented_not_present`) and the gated
/// `transfer_to_device` catch the common paths; the macro form is the airtight
/// version and its home is increment 2+.
///
/// ## Coverage boundary (GAP-161) — which expiry triggers this catches
///
/// The consistency canary fires on FUEL-SIDE triggers only: adding a `DType`
/// (the wildcard-free match below stops compiling until the new dtype is
/// classified) and wiring a `CudaStorageSlice` variant (the
/// `f8e5m2_storage_is_unimplemented_not_present` canary reddens). For the
/// STORAGE capability that is the COMPLETE set of triggers — `CudaStorageSlice`
/// is a Fuel-side `enum`, so no external event can create storage for a dtype.
///
/// It does NOT catch an expiry triggered purely by an EXTERNAL dependency
/// gaining a capability — e.g. `baracuda-kernel-vocab` adding an `ElementKind`
/// variant (the live case at `telemetry/baracuda_provider.rs:116`, where a `U32`
/// decline silently expired on an alpha bump). A hand-mirrored Fuel-side
/// expected-set is blind to that trigger BY CONSTRUCTION — the general rule is
/// that a canary must probe on the SAME SIDE as the trigger (same family as a
/// control whose reference point must come from the path under test). So when
/// this WHY-classifier shape is reused for a capability whose expiry trigger is
/// external, its canary must observe the EXTERNAL surface. The best form is
/// build-time and is DEPENDENCY-SPECIFIC, not a general technique: when the
/// external enum is NOT `#[non_exhaustive]` — as `baracuda-kernel-vocab`'s
/// `ElementKind` (alpha.78) is not — a wildcard-free `match` over it becomes a
/// COMPILE ERROR the instant the dependency adds a variant, firing on the bump
/// itself with no API surface required. Where the external enum IS
/// `#[non_exhaustive]` (e.g. `OpKind` in `fuel-core-types`, which `fkc/lower.rs`
/// records forbids a cross-crate exhaustiveness anchor), that instrument is
/// unavailable and a runtime probe is the floor. This is a boundary of the
/// PATTERN, not of storage; `map_element_kind` is a different capability (the
/// emit surface) and is not this function's concern — named only to mark the
/// edge of what a Fuel-mirrored canary can see.
pub(crate) fn cuda_storage_status(dt: DType) -> StorageStatus {
    match dt {
        // Every dtype with a `CudaStorageSlice` variant — the typed variants
        // plus the dummy-byte (`CudaSlice<u8>`) variants for the sub-byte /
        // scale formats. `Present` answers the STORAGE question ("can bytes be
        // held"); it does NOT imply every op has a kernel — e.g. `zeros`/`rand`
        // decline the dummy-byte types, which is an OP decline (increment 2),
        // not a storage decline.
        DType::U8
        | DType::I8
        | DType::U32
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::BF16
        | DType::F16
        | DType::F32
        | DType::F64
        | DType::F8E4M3
        | DType::F6E2M3
        | DType::F6E3M2
        | DType::F4
        | DType::F8E8M0 => StorageStatus::Present,

        // Expiring: representable (OCP-standard, real host type
        // `float8::F8E5M2`) but no `CudaStorageSlice` variant yet. This decline
        // INVERTS the moment a variant is wired — see the expiry canary.
        DType::F8E5M2 => StorageStatus::UnimplementedYet(
            "F8E5M2 has no CudaStorageSlice variant yet (GAP-097): it is OCP-standard and \
             representable, but CUDA storage is not wired — declining rather than mislabelling \
             a real value dtype an unsupported dummy type",
        ),

        // Expiring on a schema event — a DISTINCT trigger from F8E5M2, kept
        // separate so the two do not collapse: the physical encoding is not
        // authored (GAP-045), so storage cannot be defined yet.
        DType::F8E6M2 => StorageStatus::UndefinedEncoding(
            "F8E6M2 has no authored physical encoding (GAP-045), so CUDA storage cannot be \
             defined for it yet",
        ),
    }
}

/// Build the decline error for a dtype with no CUDA storage, single-sourcing the
/// typed reason from [`cuda_storage_status`] (GAP-161).
///
/// If called for a dtype that DOES have storage, that is a stale decline arm
/// surviving past the capability landing — it returns a loud
/// [`CudaError::InternalError`] (NOT a `debug_assert`, which would evaporate in
/// release), so a repoint gone stale surfaces instead of fabricating a bogus
/// decline. Callers convert with `.into()` as usual.
pub(crate) fn storage_unavailable(dt: DType, op: &'static str) -> CudaError {
    match cuda_storage_status(dt) {
        StorageStatus::Present => CudaError::InternalError(
            "storage_unavailable() was called for a dtype that HAS CUDA storage — a GAP-161 \
             decline arm went stale after storage was wired; re-check cuda_storage_status and \
             the calling site",
        ),
        s => CudaError::StorageUnavailable {
            dtype: dt,
            op,
            reason: s.decline_reason().unwrap_or("no CUDA storage"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{cuda_storage_status, storage_unavailable};
    use crate::error::CudaError;
    use fuel_ir::{DType, StorageStatus};

    /// GAP-161 EXPIRY TRAP. When CUDA storage for F8E5M2 lands, whoever wires it
    /// must flip its arm in `cuda_storage_status` to `Present`, and this
    /// assertion INVERTS (reddens), dragging them to the decline sites to
    /// revisit. Do NOT "fix" a future red here by loosening the match — update
    /// it deliberately when you add the `CudaStorageSlice::F8E5M2` variant.
    #[test]
    fn f8e5m2_storage_is_unimplemented_not_present() {
        assert!(matches!(
            cuda_storage_status(DType::F8E5M2),
            StorageStatus::UnimplementedYet(_)
        ));
        assert!(cuda_storage_status(DType::F8E5M2).expires());
        assert!(!cuda_storage_status(DType::F8E5M2).is_present());
    }

    /// F8E6M2 is a DISTINCT expiry (schema event, not variant-wiring) — kept
    /// separate so the two triggers don't collapse.
    #[test]
    fn f8e6m2_storage_is_undefined_encoding() {
        assert!(matches!(
            cuda_storage_status(DType::F8E6M2),
            StorageStatus::UndefinedEncoding(_)
        ));
    }

    /// Consistency canary: the `Present` set must be EXACTLY the dtypes that have
    /// a `CudaStorageSlice` variant. The expected set is hand-written on purpose
    /// — it forces a CONSCIOUS edit in lockstep with the enum, so a
    /// classification that silently drifts from the variants reddens. (Sabotage
    /// check: mark any Present dtype non-Present, or vice versa, and this fails.)
    #[test]
    fn present_set_is_exactly_the_storage_backed_dtypes() {
        for &dt in DType::ALL {
            let present = cuda_storage_status(dt).is_present();
            let expected = matches!(
                dt,
                DType::U8
                    | DType::I8
                    | DType::U32
                    | DType::I16
                    | DType::I32
                    | DType::I64
                    | DType::BF16
                    | DType::F16
                    | DType::F32
                    | DType::F64
                    | DType::F8E4M3
                    | DType::F6E2M3
                    | DType::F6E3M2
                    | DType::F4
                    | DType::F8E8M0
            );
            assert_eq!(
                present, expected,
                "storage-present classification drifted for {dt:?}"
            );
        }
    }

    /// No decline may carry an empty reason (the erasure GAP-161 exists to stop).
    #[test]
    fn every_decline_carries_a_nonempty_reason() {
        for &dt in DType::ALL {
            if let Some(r) = cuda_storage_status(dt).decline_reason() {
                assert!(!r.is_empty(), "empty decline reason for {dt:?}");
            }
        }
    }

    /// The stale-arm tripwire is a hard error, not a silent bogus decline:
    /// calling `storage_unavailable` for a Present dtype yields `InternalError`.
    #[test]
    fn storage_unavailable_on_a_present_dtype_is_a_loud_error() {
        assert!(matches!(
            storage_unavailable(DType::F32, "test"),
            CudaError::InternalError(_)
        ));
    }

    /// And for a genuinely absent dtype it carries the typed reason through.
    #[test]
    fn storage_unavailable_on_an_absent_dtype_carries_the_reason() {
        match storage_unavailable(DType::F8E5M2, "test") {
            CudaError::StorageUnavailable { dtype, op, reason } => {
                assert_eq!(dtype, DType::F8E5M2);
                assert_eq!(op, "test");
                assert!(!reason.is_empty());
            }
            other => panic!("expected StorageUnavailable, got {other:?}"),
        }
    }
}

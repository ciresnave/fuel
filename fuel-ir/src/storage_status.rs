// SPDX-License-Identifier: MIT OR Apache-2.0
//! [`StorageStatus`] — the **WHY axis** of a per-`(backend, dtype)` storage
//! decline.
//!
//! When a backend cannot hold a [`DType`](crate::DType) in device storage, the
//! *reason* it declines is load-bearing information that Fuel has historically
//! erased: several `fuel-cuda-backend` sites spell the distinction in a comment
//! and then return the identical `UnsupportedDtype` for a dtype that is
//! *representable but unwired* and one whose *encoding is not authored yet*
//! (GAP-161). This enum makes that reason a typed value the compiler and tests
//! can read, instead of prose the return type throws away.
//!
//! ## Two axes, kept separate (GAP-161 / GAP-155)
//!
//! A decline is classified on **two orthogonal axes** and they must not be
//! merged into one enum:
//!
//! * **WHY** (this type) — does the decline *expire*? `Present` /
//!   [`Impossible`](StorageStatus::Impossible) /
//!   [`UnimplementedYet`](StorageStatus::UnimplementedYet) /
//!   [`UndefinedEncoding`](StorageStatus::UndefinedEncoding). Only the latter
//!   two expire (when the capability / encoding lands).
//! * **KIND** ([`TokenKind`](crate::TokenKind), *not* this type) — a fact about
//!   vocabulary membership: `supported` / `recognized-but-unsupported` /
//!   `reserved` / `unknown`. It classifies a `&str` TOKEN (including spellings
//!   that are not `DType`s at all), lives at the recognition/emit surface, and
//!   never names a backend. A `StorageStatus` cannot even be *computed* for a
//!   non-`DType` token, so the two cannot merge — a type-level impossibility,
//!   not a discipline. Pair them as `{ kind, why: Option<Self> }`, never a
//!   merged enum.
//!
//! The axes *combine*, they do not collapse. The sharpest live case is
//! **`F8E5M2`**: KIND=`Supported` (it has a legal sk4 token and `dtype_token`
//! emits it) while WHY=[`UnimplementedYet`](StorageStatus::UnimplementedYet)
//! (no `CudaStorageSlice` variant) — both axes apply and give DIFFERENT answers,
//! and a single flat enum would have to pick one and be wrong about the other.
//! By contrast `f8e4m3fnuz` is `reserved` in KIND with no WHY at all (it is not
//! a `DType`), and `F4` on a CUDA GEMM is `recognized-but-unsupported` in KIND
//! and [`Impossible`](StorageStatus::Impossible) in WHY. If you find yourself
//! wanting a `Reserved` variant *here*, that is the collapse — stop.
//!
//! Home rationale: this type names only [`DType`](crate::DType) facts and is a
//! pure data enum, so it lives in the lowest crate every backend already
//! depends on (`fuel-ir`) — zero new dependency edges — rather than in
//! `fuel-backend-contract` (which is also reachable, but the type is a mix of
//! dtype-intrinsic reasons like `UndefinedEncoding` and backend-specific ones
//! like `UnimplementedYet`, so "lowest common crate, no inversion" is the
//! cleaner tiebreak).

/// Why a backend can — or cannot — hold a given [`DType`](crate::DType) in
/// device storage. The **WHY axis** of a decline (see the module docs); it does
/// **not** answer the orthogonal KIND question (reserved / recognized /
/// unknown), and must never grow a variant that does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageStatus {
    /// The backend has storage for this dtype. Not a decline.
    Present,
    /// No storage will *ever* exist for this dtype on this backend — a
    /// **permanent** decline that never expires (e.g. a dtype with no
    /// meaningful device representation, or a reserved-by-standard spelling
    /// that is permanently unimplemented). Correctly a plain wildcard would-be.
    Impossible(&'static str),
    /// The dtype is representable and storage *could* exist, but is not wired
    /// yet. This decline **expires** the moment storage lands — it is the class
    /// GAP-161 exists to trap, because a stale `UnimplementedYet` keeps
    /// declining a dtype that now works while every instrument reports health.
    UnimplementedYet(&'static str),
    /// The dtype's physical encoding is not authored yet, so storage cannot be
    /// *defined* for it. Also **expires**, but on a schema event rather than on
    /// a backend wiring one variant — kept distinct from
    /// [`UnimplementedYet`](Self::UnimplementedYet) because the trigger differs.
    UndefinedEncoding(&'static str),
}

impl StorageStatus {
    /// `true` iff the backend has storage for the dtype.
    pub fn is_present(self) -> bool {
        matches!(self, StorageStatus::Present)
    }

    /// The human-readable reason for a decline, or `None` if
    /// [`Present`](Self::Present).
    pub fn decline_reason(self) -> Option<&'static str> {
        match self {
            StorageStatus::Present => None,
            StorageStatus::Impossible(r)
            | StorageStatus::UnimplementedYet(r)
            | StorageStatus::UndefinedEncoding(r) => Some(r),
        }
    }

    /// `true` iff this decline is **time-dependent** — it will become wrong when
    /// the capability (or encoding) lands. This is the property GAP-161 canaries
    /// key on: an `expires()` decline must be re-decided when its trigger fires,
    /// and nothing in the compiler or a passing test forces that on its own.
    pub fn expires(self) -> bool {
        matches!(
            self,
            StorageStatus::UnimplementedYet(_) | StorageStatus::UndefinedEncoding(_)
        )
    }
}

impl std::fmt::Display for StorageStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageStatus::Present => write!(f, "present"),
            StorageStatus::Impossible(r) => write!(f, "impossible: {r}"),
            StorageStatus::UnimplementedYet(r) => write!(f, "unimplemented yet: {r}"),
            StorageStatus::UndefinedEncoding(r) => write!(f, "undefined encoding: {r}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StorageStatus;

    #[test]
    fn only_unimplemented_and_undefined_expire() {
        assert!(!StorageStatus::Present.expires());
        assert!(!StorageStatus::Impossible("x").expires());
        assert!(StorageStatus::UnimplementedYet("x").expires());
        assert!(StorageStatus::UndefinedEncoding("x").expires());
    }

    #[test]
    fn present_has_no_reason_declines_do() {
        assert_eq!(StorageStatus::Present.decline_reason(), None);
        assert_eq!(StorageStatus::Impossible("r").decline_reason(), Some("r"));
        assert_eq!(
            StorageStatus::UnimplementedYet("r").decline_reason(),
            Some("r")
        );
        assert_eq!(
            StorageStatus::UndefinedEncoding("r").decline_reason(),
            Some("r")
        );
    }
}

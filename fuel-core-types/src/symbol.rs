//! Symbolic runtime values — the [`SymId`] identity, the per-pass [`SymEnv`]
//! binding registry, and the [`DynScalar`] op-param carrier.
//!
//! These are the primitive for runtime-dependent values: chiefly dynamic
//! dimension extents (see [`Extent`](crate::shape::Extent), built on this in
//! step 1b), but also scalar op params like the KV-cache write offset and the
//! RoPE position. A graph carries `SymId`s — stable, serializable,
//! session-independent identities — and each id's concrete value is supplied
//! per forward pass via a [`SymEnv`], a sibling of the realize tensor-data
//! cache. Two places "share" a value by holding the *same* `SymId` and each
//! reading it on demand: single source of truth, no aliasing, no propagation.
//!
//! See `docs/session-prompts/symbolic-extents-and-persistent-decode.md`.
//! Phase D step 1a.

use std::collections::HashMap;

use crate::{Error, Result};

/// The identity of a runtime ("symbolic") value — a key into a [`SymEnv`].
///
/// Stable, serializable, and session-independent: the graph stores `SymId`s
/// (shared + immutable after optimization), and each id's concrete value is
/// supplied per forward pass. **Equal ids denote the same runtime value**,
/// which is how unification works — e.g. a KV cache's K-length and V-length
/// share one `SymId`, so they resolve together without aliasing. Allocate
/// fresh ids with [`SymGen`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(pub u32);

/// Monotonic allocator for fresh [`SymId`]s during graph/shape construction.
#[derive(Debug, Clone, Default)]
pub struct SymGen {
    next: u32,
}

impl SymGen {
    pub fn new() -> Self {
        Self { next: 0 }
    }

    /// Allocate a fresh, never-before-returned `SymId`.
    pub fn fresh(&mut self) -> SymId {
        let id = SymId(self.next);
        self.next += 1;
        id
    }
}

/// A per-forward-pass binding environment: `SymId -> usize`.
///
/// The sibling of the realize tensor-data cache; this carries the scalar/shape
/// inputs. Bindings are **write-once per pass**: each symbol is bound exactly
/// once (up-front for input-determined values like the KV length, or by its
/// producer for data-determined values), then only read. Presence of a `sym`
/// therefore signals "its value is available" (for data-determined symbols,
/// equivalently "its producer has run").
#[derive(Debug, Clone, Default)]
pub struct SymEnv {
    map: HashMap<SymId, usize>,
}

impl SymEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind `sym` to `value`. Write-once: re-binding to the *same* value is
    /// idempotent; re-binding to a *different* value is a contract violation
    /// (a symbol's value is fixed for the pass) and returns an error rather
    /// than silently overwriting.
    pub fn bind(&mut self, sym: SymId, value: usize) -> Result<()> {
        match self.map.get(&sym) {
            Some(&existing) if existing != value => Err(Error::Msg(format!(
                "SymEnv: symbol {sym:?} already bound to {existing}; cannot rebind to {value} \
                 within a forward pass (symbols are write-once)",
            ))
            .bt()),
            _ => {
                self.map.insert(sym, value);
                Ok(())
            }
        }
    }

    /// The concrete value bound to `sym`, or `None` if unbound.
    pub fn get(&self, sym: SymId) -> Option<usize> {
        self.map.get(&sym).copied()
    }

    /// Whether `sym` is bound. For data-determined symbols, presence is
    /// equivalent to "the producing op has run" (a consequence of write-once).
    pub fn is_bound(&self, sym: SymId) -> bool {
        self.map.contains_key(&sym)
    }

    /// Drop all bindings, to reuse the env for the next pass.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// A scalar op-parameter that is either a build-time constant or a runtime
/// symbol resolved through a [`SymEnv`].
///
/// Used for scalar parameters that are *not* dimensions — the KV-cache
/// `WriteSlice` offset, the RoPE position, a fused-attention `k_len`. (Dynamic
/// *dimensions* use [`Extent`](crate::shape::Extent) instead, which also
/// carries capacity bounds.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DynScalar {
    /// A value fixed at graph-construction time.
    Concrete(usize),
    /// A runtime value resolved via the `SymEnv` at realize.
    Sym(SymId),
}

impl DynScalar {
    /// The concrete value: itself if `Concrete`, else `env.get(sym)`. `None`
    /// if a `Sym`'s symbol is unbound.
    pub fn resolve(&self, env: &SymEnv) -> Option<usize> {
        match self {
            DynScalar::Concrete(v) => Some(*v),
            DynScalar::Sym(s) => env.get(*s),
        }
    }

    /// The value if it is a build-time constant, else `None`.
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            DynScalar::Concrete(v) => Some(*v),
            DynScalar::Sym(_) => None,
        }
    }

    /// Whether this is a runtime symbol (vs a build-time constant).
    pub fn is_dynamic(&self) -> bool {
        matches!(self, DynScalar::Sym(_))
    }
}

impl From<usize> for DynScalar {
    fn from(v: usize) -> Self {
        DynScalar::Concrete(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symgen_allocates_distinct_ids() {
        let mut g = SymGen::new();
        let a = g.fresh();
        let b = g.fresh();
        assert_ne!(a, b);
        assert_eq!(a, SymId(0));
        assert_eq!(b, SymId(1));
    }

    #[test]
    fn symenv_bind_get_presence() {
        let mut env = SymEnv::new();
        let s = SymId(7);
        assert!(!env.is_bound(s));
        assert_eq!(env.get(s), None);
        env.bind(s, 53).unwrap();
        assert!(env.is_bound(s));
        assert_eq!(env.get(s), Some(53));
    }

    #[test]
    fn symenv_is_write_once() {
        let mut env = SymEnv::new();
        let s = SymId(1);
        env.bind(s, 53).unwrap();
        // Idempotent re-bind to the same value is fine.
        env.bind(s, 53).unwrap();
        // A conflicting re-bind within a pass is a contract violation.
        assert!(env.bind(s, 54).is_err());
        // The original value is unchanged after the rejected rebind.
        assert_eq!(env.get(s), Some(53));
        env.clear();
        assert!(!env.is_bound(s));
    }

    #[test]
    fn dynscalar_resolves_and_shares_value_by_sym() {
        let mut env = SymEnv::new();
        let s = SymId(3);
        let a = DynScalar::Sym(s);
        let b = DynScalar::Sym(s); // same sym → unified
        let c = DynScalar::Concrete(99);
        assert_eq!(a.resolve(&env), None, "unbound sym resolves to None");
        env.bind(s, 42).unwrap();
        // Two DynScalars sharing a SymId resolve to the same value — the
        // unification property, achieved by indirection, not aliasing.
        assert_eq!(a.resolve(&env), Some(42));
        assert_eq!(b.resolve(&env), Some(42));
        assert_eq!(c.resolve(&env), Some(99));
        assert_eq!(c.as_concrete(), Some(99));
        assert_eq!(a.as_concrete(), None);
        assert!(a.is_dynamic());
        assert!(!c.is_dynamic());
        assert_eq!(DynScalar::from(7usize), DynScalar::Concrete(7));
    }
}

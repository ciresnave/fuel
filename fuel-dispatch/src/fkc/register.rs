//! The REGISTRATION slice: a parsed+lowered provider bundle →
//! live kernel registrations on the two dispatch registries
//! (adoption plan §1.2, §1.3, §2.1, §2.2).
//!
//! This module makes the FKC importer **end-to-end**: a contract bundle
//! file becomes concrete entries in a [`KernelBindingTable`] (primitive
//! `op_kind` contracts) and a [`FusedKernelRegistry`] (`fused_op`
//! contracts). The pipeline is:
//!
//! ```text
//! import_bundle_str(src)  =  parse_file(src) → lower_file(.., link) → ImportedProvider
//! import_bundle(path)     =  read file       → import_bundle_str
//! import_glob(pattern)    =  glob files      → import_bundle_str each → MERGE (front-matter must agree)
//! provider.register_into(table, fused)       =  the actual registry inserts + finalize gate
//! ```
//!
//! ## Cost decision for this slice
//!
//! Two cost paths coexist:
//!
//! 1. **Named cost-fn trampoline (§4.4, Task-F).** A primitive contract whose
//!    `cost.cost_fn` NAMES a registered cost fn has that name resolved through
//!    the provider's [`LinkRegistry::resolve_cost_fn`](crate::fkc::LinkRegistry)
//!    at lower time and the resulting [`CostFn`](crate::kernel::CostFn) pinned
//!    on the binding by [`ImportedProvider::register_into`]. A pinned cost fn
//!    is NOT the [`unknown_cost`] sentinel, so the
//!    [`fill_unset_cost_for_backend`](crate::kernel::KernelBindingTable::fill_unset_cost_for_backend)
//!    pass leaves it — the mechanism that lets a contract carry a real,
//!    shape-aware cost model (e.g. the CUDA `flash_decoding` static
//!    infeasibility gate `cost_flash_decoding_cuda`). An unresolved name is a
//!    typed [`FkcError::UnknownCostFn`], never a silent fallback.
//! 2. **Judge-bootstrapped default (the common case, [consumer-ahead]).** A
//!    contract that names NO cost fn registers with the [`unknown_cost`]
//!    sentinel `CostFn` (primitives) / [`fused_unknown_cost`] (fused ops);
//!    the fill pass then upgrades the primitive sentinel to the op/backend
//!    default, and the Judge refines empirically. The parsed
//!    [`CompiledCostExpr`](crate::fkc::cost_expr::CompiledCostExpr) stays on
//!    the `Resolved*` record. Fused ops have no named-cost-fn path yet (the
//!    fused cost-fn signature differs; no consumer today).

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use fuel_ir::backend::BackendCapabilities;
use fuel_ir::probe::BackendId;
use fuel_ir::Shape;
use fuel_graph::registry::FusedOpParams;

use crate::fkc::error::FkcError;
use crate::fkc::lower::{lower_file, LinkRegistry, Resolved, ResolvedFused, ResolvedPrimitive};
use crate::fkc::parse::parse_file;
use crate::fused::{BackendImpl, CostEstimate, FusedKernelRegistry};
use crate::kernel::{unknown_cost, KernelBindingTable};

// ===========================================================================
// kernel_source interner (§1.3)
// ===========================================================================

/// Intern a `kernel_source` (or any short, bounded-cardinality provider
/// tag) string into a `&'static str`, leaking it on first sighting.
///
/// `BindingEntry.kernel_source` and `BackendImpl.dtypes` use `&'static`
/// data, but a contract's `kernel_source` is read from a file at runtime,
/// so it is not `'static`. Resolution per plan §1.3: a process-wide
/// `OnceLock<Mutex<HashSet<&'static str>>>` interner that `Box::leak`s the
/// owned string on first sighting and returns the cached handle on every
/// repeat.
///
/// **Bounded process-lifetime leak.** Each *distinct* string is leaked
/// exactly once for the life of the process. The set of distinct
/// `kernel_source` tags is a handful (one per provider source: e.g.
/// `"portable-cpu"`, `"aocl"`, `"mkl"`), so the total leaked bytes are
/// trivially bounded and never grow under repeated imports of the same
/// provider. This is the same posture `precision.rs` takes for precision
/// `notes`. (Alternative, out of scope here: widen `kernel_source` to
/// `Cow<'static, str>` in a follow-up.)
pub(crate) fn intern(s: &str) -> &'static str {
    static POOL: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = pool.lock().expect("kernel_source interner poisoned");
    if let Some(existing) = guard.get(s) {
        return existing;
    }
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    guard.insert(leaked);
    leaked
}

/// Intern a per-operand dtype list into a `&'static [DType]` slice for a
/// fused [`BackendImpl`] (whose `dtypes` field is `&'static [DType]`).
/// Backed by the same bounded process-lifetime leak as [`intern`]: each
/// distinct dtype tuple is leaked once and cached. Distinct fused ops use
/// a small, fixed set of dtype tuples, so the leak is bounded.
fn intern_dtypes(dtypes: &[fuel_ir::DType]) -> &'static [fuel_ir::DType] {
    use fuel_ir::DType;
    static POOL: OnceLock<Mutex<HashSet<&'static [DType]>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = pool.lock().expect("fused dtypes interner poisoned");
    if let Some(existing) = guard.get(dtypes) {
        return existing;
    }
    let leaked: &'static [DType] = Box::leak(dtypes.to_vec().into_boxed_slice());
    guard.insert(leaked);
    leaked
}

// ===========================================================================
// Fused-cost sentinel (the fused equivalent of `unknown_cost`)
// ===========================================================================

/// The fused-op cost-fn sentinel — the fused analog of
/// [`unknown_cost`]. There is no public fused sentinel in `fused.rs`
/// (registrations that omit a cost use `CostEstimate::default()` inline),
/// so the importer supplies its own trivial zero-cost fn matching the
/// fused cost-fn signature
/// `fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate`.
///
/// [consumer-ahead]: declared cost priors are retained on
/// `ResolvedFused.cost` but not yet wired into a `CostFn`; the Judge
/// bootstraps all imported costs for now (a follow-up slice adds the cost
/// trampoline, plan §2.3 strategy A).
pub fn fused_unknown_cost(
    _shapes: &[Shape],
    _params: &FusedOpParams,
    _caps: &BackendCapabilities,
) -> CostEstimate {
    CostEstimate::default()
}

// ===========================================================================
// ImportedProvider (§1.2)
// ===========================================================================

/// A parsed + validated provider bundle, holding the resolved per-kernel
/// records ready to register. Construction (`import_*`) already ran the
/// parse + restricted-YAML pre-pass + lowering (string → typed dispatch
/// records); [`Self::register_into`] only does the registry inserts +
/// the duplicate-detection gate.
#[derive(Debug, Clone)]
pub struct ImportedProvider {
    /// `provider.name` (front-matter).
    pub name: String,
    /// `provider.backend` (front-matter), the registry key backend.
    pub backend: BackendId,
    /// `provider.kernel_source` (front-matter), interned to `&'static`.
    pub kernel_source: &'static str,
    /// Lowered `op_kind` contracts → the binding table.
    pub primitives: Vec<ResolvedPrimitive>,
    /// Lowered `fused_op` contracts → the fused registry.
    pub fused: Vec<ResolvedFused>,
    /// Soft import diagnostics (§3.5 warns, precision downgrades, etc.).
    pub warnings: Vec<crate::fkc::ImportWarning>,
}

impl ImportedProvider {
    /// Split a flat `Vec<Resolved>` into the primitive / fused buckets.
    fn from_resolved(
        name: String,
        backend: BackendId,
        kernel_source: &'static str,
        resolved: Vec<Resolved>,
        warnings: Vec<crate::fkc::ImportWarning>,
    ) -> Self {
        let mut primitives = Vec::new();
        let mut fused = Vec::new();
        for r in resolved {
            match r {
                Resolved::Primitive(p) => primitives.push(p),
                Resolved::Fused(f) => fused.push(f),
            }
        }
        ImportedProvider {
            name,
            backend,
            kernel_source,
            primitives,
            fused,
            warnings,
        }
    }

    /// Register every primitive contract into `table` and every fused
    /// contract into `fused`, then run the duplicate-detection gate.
    ///
    /// **Describe-only sections (`registrable: false`, §3.10) are already
    /// excluded** before this point: [`lower_file`] (called by the `import_*`
    /// constructors) filters them out, so `self.primitives` / `self.fused`
    /// contain only registrable kernels. A documentation-only section therefore
    /// never reaches the binding table / fused registry or the duplicate-
    /// `KernelRef` gate — it is honest docs, not a dispatch decision point.
    ///
    /// Per slice (§2.1 / §2.2): each primitive becomes a
    /// [`KernelBindingTable::register_full_with_source`] insert; each
    /// fused op becomes a [`FusedKernelRegistry::register`] of a
    /// [`BackendImpl`]. **Cost is the sentinel for every kernel this
    /// slice** — `unknown_cost` for primitives, [`fused_unknown_cost`]
    /// for fused ops (see the module-level cost note); the parsed cost AST
    /// stays on the `Resolved*` record.
    ///
    /// `register_full_with_source` is append-only and returns `()` today;
    /// the **same** `KernelRef` driven onto one `(op, dtypes, backend)`
    /// key twice is detected by [`KernelBindingTable::finalize`], which
    /// this method calls **after all inserts** and maps to
    /// [`FkcError::DuplicateKernelRef`] (never a panic — the import path is
    /// `Result` end-to-end).
    pub fn register_into(
        &self,
        table: &mut KernelBindingTable,
        fused: &mut FusedKernelRegistry,
    ) -> Result<(), FkcError> {
        // --- Primitives → KernelBindingTable (§2.1) ---
        for p in &self.primitives {
            let kernel_source: &'static str = intern(&p.kernel_source);
            // §4.4 cost-fn trampoline (Task-F): when the contract PINS a cost
            // fn (`cost.cost_fn`, resolved through the LinkRegistry at lower
            // time), stamp THAT `CostFn` — it survives the
            // `fill_unset_cost_for_backend` pass (which only replaces the
            // `unknown_cost` sentinel), so a contract can carry a real,
            // shape-aware cost model (e.g. the CUDA `flash_decoding` static
            // infeasibility gate). When it does NOT pin one, keep the
            // `unknown_cost` sentinel so the fill pass upgrades it to the
            // op/backend default (the Judge-bootstrapped path for every other
            // imported primitive).
            let cost_fn = p.cost_fn.unwrap_or(unknown_cost);
            // FKC gap-closure Task 2.2 (§2.3): compile the contract's
            // declared cost AST (if any, and if no pinned `cost_fn` already
            // wins — see `stamp_primitive_cost_expr`) into the interned
            // `'static` handle threaded onto the binding below.
            let cost_expr = crate::fkc::cost_compile::stamp_primitive_cost_expr(p);
            // Structural-miss fallback bit (FKC §4.12): compute genericity
            // ONCE from the retained five-flag `ResolvedLayout` set and stamp
            // it onto the binding so the live dispatch pick site reads a
            // precomputed bool (never re-derives it from the lossy
            // single-bool `KernelCaps`). Baracuda's miss telemetry keys on it.
            let is_generic = crate::fkc::is_generic_contract(&p.layouts);
            // Thread the contract-computed revision hash onto the binding (FKC
            // §4.11 basis tuple) so the Baracuda dispatch-telemetry `ImplId`
            // pins the exact kernel build. Hand-written kernels stamp
            // `UNTRACKED` (`0`); only this FKC-import path carries a real one.
            table.register_full_with_source_generic(
                p.op,
                &p.dtypes,
                p.backend,
                p.kernel,
                p.caps,
                p.precision,
                cost_fn,
                kernel_source,
                is_generic,
                p.revision.0,
                // FKC gap-closure Task 2.2: the compiled cost AST (or `None`
                // when a pinned `cost_fn` already wins, or the contract
                // declared no cost expression at all).
                cost_expr,
            );
        }

        // --- Fused ops → FusedKernelRegistry (§2.2) ---
        for f in &self.fused {
            // FKC gap-closure Task 2.4 (§2.3): compile the contract's
            // declared cost AST (if any) into the interned `'static` handle
            // threaded onto `cost_expr` below. `fused_layer1_cost` prefers
            // it over both the hand-written `cost` fn (still stamped as the
            // `fused_unknown_cost` sentinel here) and the decompose-derived
            // fallback.
            let dtypes: &'static [fuel_ir::DType] = intern_dtypes(&f.dtypes);
            fused.register(
                f.id,
                f.backend,
                BackendImpl {
                    kernel: f.kernel,
                    dtypes,
                    cost: fused_unknown_cost,
                    precision: f.precision,
                    caps: f.caps,
                    revision: f.revision,
                    cost_expr: crate::fkc::cost_compile::stamp_fused_cost_expr(f),
                },
            );
            // Finding 5.4 (FKC side, Task 3.6): preserve the contract's
            // `return.bundle` per-slot names in the side-table, keyed by the
            // SAME (id, backend, dtypes) tuple just registered above. Empty
            // for a single-output section (the common case) — skip the
            // insert rather than recording a vacuous entry.
            if !f.bundle_slot_names.is_empty() {
                fused.record_bundle_slot_names(f.id, f.backend, dtypes, &f.bundle_slot_names);
            }
        }

        // --- The duplicate-detection gate, surfaced as a typed error ---
        // `register_full_with_source` is append-only; `finalize` is the
        // single pass that detects the same `KernelRef` registered twice
        // at one key. Map its dispatch `Error` message into a typed
        // `FkcError::DuplicateKernelRef` (never-panic on the import path).
        table
            .finalize()
            .map_err(|e| FkcError::DuplicateKernelRef(e.to_string()))?;

        Ok(())
    }
}

// ===========================================================================
// Public import entry points (§1.2)
// ===========================================================================

/// Parse + validate a single bundle markdown file's bytes into an
/// [`ImportedProvider`]. Pure: no I/O of its own (tests pass `&str`).
///
/// `parse_file` → `lower_file(.., link)` → assemble. Every failure is a
/// typed [`FkcError`]; never panics.
pub fn import_bundle_str(
    src: &str,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError> {
    let file = parse_file(src)?;
    // Run the build-time validators (the V-FKC-* battery, §10) AFTER parse so a
    // structurally / coherence-bad contract fails import before any lowering or
    // registration touches the dispatch surface. Validation runs over EVERY
    // section, including describe-only ones (§3.10) — their descriptive checks
    // (dtype, layout, cost, precision) still apply; only their dispatch-resolution
    // checks are skipped. The ONE relaxation: a describe-only section's
    // CONSUMER-AHEAD gate (`GatherNotYetSupported` / `MxNotYetRegistrable`) is a
    // correct "describable-but-not-yet-registrable" outcome, so `validate_file`
    // treats it as non-blocking (the same "deferred" posture the corpus CI lint
    // takes) — a describe-only documentation section must not block a bundle's
    // importable sections.
    crate::fkc::validate::validate_file(&file)?;
    // `lower_file` then EXCLUDES describe-only sections from the registered set
    // (§3.10): they never become a Resolved* record and are never registered.
    let mut warnings: Vec<crate::fkc::ImportWarning> = Vec::new();
    let mut resolved = lower_file(&file, link, &mut warnings)?;
    let provider = &file.front_matter.provider;
    let backend = lower_backend_str(&provider.backend)?;
    let kernel_source = intern(&provider.kernel_source);

    // V-FKC-9 (Task 4.3): gate every lowered record's declared precision
    // against the empirical verification ledger BEFORE it reaches
    // `from_resolved` / the binding table. Any machine-checkable claim
    // (bit-stability, ULP, etc.) with no matching `pass` ledger entry for
    // this exact `(kernel_source, backend, dtypes, kernel_revision_hash)`
    // collapses to `PrecisionGuarantee::UNAUDITED` plus a warning — an
    // asserted-but-never-measured claim must not survive import as if it
    // had been verified. Task 4.5b seeded the embedded ledger with real
    // empirical `pass` entries for the fused CPU ops flipped to
    // `audited: true` on 2026-07-03, so THOSE records are honored here
    // (same revision hash, since `default_kernel_registry()`'s
    // `register_*_fused_from_contract` helpers stamp `BackendImpl.revision`
    // from this SAME `ResolvedFused.revision`); a primitive `op_kind`
    // contract like `add_f32` (never in Task 4.5b's fused-op scope) has no
    // ledger entry and is downgraded.
    let ledger = crate::fkc::verify::embedded();
    for r in &mut resolved {
        match r {
            Resolved::Primitive(p) => {
                let q = crate::fkc::verify::LedgerQuery {
                    kernel_ref: p.kernel_source.as_str(),
                    backend: p.backend,
                    dtypes: p.dtypes.as_slice(),
                    kernel_revision_hash: p.revision.0,
                };
                p.precision = crate::fkc::verify::gate_precision(p.precision, &q, ledger, &mut warnings);
            }
            Resolved::Fused(f) => {
                let q = crate::fkc::verify::LedgerQuery {
                    kernel_ref: f.kernel_source.as_str(),
                    backend: f.backend,
                    dtypes: f.dtypes.as_slice(),
                    kernel_revision_hash: f.revision.0,
                };
                f.precision = crate::fkc::verify::gate_precision(f.precision, &q, ledger, &mut warnings);
            }
        }
    }

    Ok(ImportedProvider::from_resolved(
        provider.name.clone(),
        backend,
        kernel_source,
        resolved,
        warnings,
    ))
}

/// Convenience: read a bundle file at `path`, then [`import_bundle_str`].
pub fn import_bundle(
    path: impl AsRef<Path>,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError> {
    let path = path.as_ref();
    let src = std::fs::read_to_string(path).map_err(|e| FkcError::Io {
        path: path.display().to_string(),
        reason: e.to_string(),
    })?;
    import_bundle_str(&src, link)
}

/// Glob multiple per-kernel / per-bundle FKC files into one provider
/// (§9.2). Each matched file is parsed + lowered independently and then
/// **merged** into a single [`ImportedProvider`]; the front-matter
/// `provider.name` / `provider.backend` / `provider.kernel_source` must
/// **agree** across every file or [`FkcError::ProviderMismatch`].
///
/// Glob order is sorted for determinism (so duplicate-detection /
/// revision-hash order is stable across runs).
///
/// **No new dependency.** `fuel-dispatch` has no `glob` crate dep, and the
/// adoption plan (§1.2) prefers no new dep — so this uses a minimal
/// directory walk + a small filename matcher over the pattern's last
/// component rather than pulling in `glob`. The supported pattern shape is
/// `<dir>/<filename-pattern>` where the filename pattern may contain `*`
/// wildcards (e.g. `docs/kernel-contracts/cpu/*.fkc.md`); `**` is not
/// supported (a single directory level, matching the per-provider layout).
pub fn import_glob(
    pattern: &str,
    link: &dyn LinkRegistry,
) -> Result<ImportedProvider, FkcError> {
    let mut paths = glob_files(pattern)?;
    paths.sort(); // deterministic order

    if paths.is_empty() {
        return Err(FkcError::Io {
            path: pattern.to_string(),
            reason: "glob pattern matched no files".to_string(),
        });
    }

    let mut merged: Option<ImportedProvider> = None;
    for path in &paths {
        let file_label = path.display().to_string();
        let provider = import_bundle(path, link)?;
        match merged.as_mut() {
            None => merged = Some(provider),
            Some(acc) => {
                // Front-matter must agree across files (§9.2).
                if provider.name != acc.name {
                    return Err(FkcError::ProviderMismatch {
                        field: "provider.name".to_string(),
                        expected: acc.name.clone(),
                        found: provider.name.clone(),
                        file: file_label,
                    });
                }
                if provider.backend != acc.backend {
                    return Err(FkcError::ProviderMismatch {
                        field: "provider.backend".to_string(),
                        expected: format!("{:?}", acc.backend),
                        found: format!("{:?}", provider.backend),
                        file: file_label,
                    });
                }
                // kernel_source is interned, so distinct tags compare by
                // pointer-or-content; compare by content for the message.
                if provider.kernel_source != acc.kernel_source {
                    return Err(FkcError::ProviderMismatch {
                        field: "provider.kernel_source".to_string(),
                        expected: acc.kernel_source.to_string(),
                        found: provider.kernel_source.to_string(),
                        file: file_label,
                    });
                }
                // Agreed — fold this file's kernels into the merged provider.
                acc.primitives.extend(provider.primitives);
                acc.fused.extend(provider.fused);
                acc.warnings.extend(provider.warnings);
            }
        }
    }

    Ok(merged.expect("non-empty path list yields a provider"))
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Map a front-matter `backend` string to a [`BackendId`]. Mirrors
/// `lower.rs`'s `lower_backend` (kept local to avoid widening that fn's
/// visibility for one call site). `UnknownBackend` on a bad string.
fn lower_backend_str(s: &str) -> Result<BackendId, FkcError> {
    match s {
        "Cpu" => Ok(BackendId::Cpu),
        "Cuda" => Ok(BackendId::Cuda),
        "Vulkan" => Ok(BackendId::Vulkan),
        "Metal" => Ok(BackendId::Metal),
        other => Err(FkcError::UnknownBackend {
            section: "<front-matter provider>".to_string(),
            backend: other.to_string(),
        }),
    }
}

/// Minimal `<dir>/<filename-pattern>` glob: list the directory and keep
/// files whose name matches the pattern's last component (with `*`
/// wildcards). No new dependency; no `**` support (single directory level).
fn glob_files(pattern: &str) -> Result<Vec<std::path::PathBuf>, FkcError> {
    // Split into directory + filename pattern on the last path separator.
    let p = Path::new(pattern);
    let file_pat = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .ok_or_else(|| FkcError::Io {
            path: pattern.to_string(),
            reason: "glob pattern has no filename component".to_string(),
        })?;
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty());
    let dir = dir.unwrap_or_else(|| Path::new("."));

    let read = std::fs::read_dir(dir).map_err(|e| FkcError::Io {
        path: dir.display().to_string(),
        reason: e.to_string(),
    })?;

    let mut out = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| FkcError::Io {
            path: dir.display().to_string(),
            reason: e.to_string(),
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if matches_glob(&file_pat, &name) {
            out.push(path);
        }
    }
    Ok(out)
}

/// Match a filename against a `*`-wildcard glob pattern (single path
/// component; `*` matches any run of non-separator characters). A
/// pattern with no `*` is an exact match.
fn matches_glob(pattern: &str, name: &str) -> bool {
    // Split the pattern on `*`; every literal segment must appear in
    // order, with the first segment anchored at the start and the last at
    // the end (unless the pattern starts/ends with `*`).
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        // No wildcard — exact match.
        return pattern == name;
    }
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            // First literal segment must anchor at the start.
            if !name[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == parts.len() - 1 {
            // Last literal segment must anchor at the end (and not overlap
            // an already-consumed prefix).
            let rest = &name[pos..];
            return rest.len() >= part.len() && rest.ends_with(part);
        } else {
            // Middle segment: must occur somewhere at/after pos.
            match name[pos..].find(part) {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::caps_map::{ResolvedLayout, Tri};
    use crate::fkc::cost_expr::CompiledCostExpr;
    use crate::fkc::lower::ResolvedFused;
    use crate::fused::{KernelRevisionHash, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, KernelDTypes, KernelRef};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::DType;
    use fuel_graph::registry::FusedOps;
    use smallvec::SmallVec;
    use std::sync::{Arc, RwLock};

    const ELEMENTWISE_BINARY: &str =
        include_str!("../../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md");

    // -- two distinct dummy kernels (distinct fn items ⇒ distinct ptrs) --

    fn dummy_a(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[fuel_ir::Layout],
        _p: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }
    fn dummy_b(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[fuel_ir::Layout],
        _p: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    /// A LinkRegistry stub mapping each DISTINCT `entry_point` string to a
    /// distinct dummy `KernelRef` — so importing a many-section contract
    /// does not collapse every section onto one pointer (which would trip
    /// the finalize gate spuriously). It hands out `dummy_a` / `dummy_b`
    /// in alternation per unique symbol (enough distinct pointers for the
    /// elementwise-binary corpus where every section has a unique key).
    struct DistinctLink {
        seen: Mutex<std::collections::HashMap<String, KernelRef>>,
    }
    impl DistinctLink {
        fn new() -> Self {
            Self {
                seen: Mutex::new(std::collections::HashMap::new()),
            }
        }
        fn resolve(&self, symbol: &str) -> Option<KernelRef> {
            let mut g = self.seen.lock().unwrap();
            if let Some(k) = g.get(symbol) {
                return Some(*k);
            }
            // Alternate between two distinct fn items by current count
            // parity. Each unique symbol gets a stable pointer; the keys
            // in the elementwise-binary corpus are unique per section so
            // two pointers suffice to avoid same-key collisions there.
            let k: KernelRef = if g.len() % 2 == 0 { dummy_a } else { dummy_b };
            g.insert(symbol.to_string(), k);
            Some(k)
        }
    }
    impl LinkRegistry for DistinctLink {
        fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
        fn resolve_fused(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
    }

    /// A LinkRegistry stub that maps EVERY symbol to the SAME pointer —
    /// used for the duplicate-detection test.
    struct SameLink;
    impl LinkRegistry for SameLink {
        fn resolve_primitive(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_a)
        }
        fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_a)
        }
    }

    // =====================================================================
    // END-TO-END: a real contract file → live registration
    // =====================================================================

    #[test]
    fn import_real_elementwise_binary_registers_add_f32_end_to_end() {
        // Parse + lower the authored CPU elementwise-binary bundle, then
        // register it into a FRESH binding table and assert the kernel is
        // present + the table finalizes Ok.
        let link = DistinctLink::new();
        let provider = import_bundle_str(ELEMENTWISE_BINARY, &link)
            .expect("authored elementwise-binary.fkc.md imports");

        assert_eq!(provider.name, "fuel-cpu-backend");
        assert_eq!(provider.backend, BackendId::Cpu);
        assert_eq!(provider.kernel_source, "portable-cpu");
        assert!(
            !provider.primitives.is_empty(),
            "elementwise-binary yields primitives"
        );
        assert!(provider.fused.is_empty(), "no fused ops in this bundle");

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into a fresh table succeeds");

        // The add_f32 contract: (AddElementwise, [F32,F32,F32], Cpu).
        let key_dtypes = [DType::F32, DType::F32, DType::F32];
        let looked_up = table.lookup(OpKind::AddElementwise, &key_dtypes, BackendId::Cpu);
        assert!(
            looked_up.is_ok(),
            "AddElementwise/[F32,F32,F32]/Cpu must be registered: {looked_up:?}"
        );

        // The table is internally consistent.
        assert!(table.finalize().is_ok(), "finalize is Ok after import");
        assert!(table.len() >= provider.primitives.len());
    }

    // =====================================================================
    // VERTICAL SLICE: import through a REAL LinkRegistry → the real kernel
    // =====================================================================

    /// A REAL (non-stub) [`LinkRegistry`]: resolves the authored contract's
    /// `add_f32` entry-point symbol to the ACTUAL production CPU kernel
    /// wrapper (`dispatch::add_elementwise_f32_cpu_wrapper`). Every other
    /// section's symbol gets a distinct dummy so the multi-section bundle
    /// still imports without a spurious finalize collision. This is the first
    /// non-stub LinkRegistry — it proves the FKC import path resolves a
    /// contract symbol to a real, executable kernel, not a placeholder.
    struct EntryPointLink {
        seen: Mutex<std::collections::HashMap<String, KernelRef>>,
    }
    impl EntryPointLink {
        fn new() -> Self {
            Self {
                seen: Mutex::new(std::collections::HashMap::new()),
            }
        }
        fn resolve(&self, symbol: &str) -> Option<KernelRef> {
            if symbol == "fuel_cpu_backend::byte_kernels::add_f32" {
                // THE point of the slice: the real production kernel.
                return Some(crate::dispatch::add_elementwise_f32_cpu_wrapper as KernelRef);
            }
            let mut g = self.seen.lock().unwrap();
            if let Some(k) = g.get(symbol) {
                return Some(*k);
            }
            let k: KernelRef = if g.len() % 2 == 0 { dummy_a } else { dummy_b };
            g.insert(symbol.to_string(), k);
            Some(k)
        }
    }
    impl LinkRegistry for EntryPointLink {
        fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
        fn resolve_fused(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
    }

    #[test]
    fn import_add_f32_through_real_link_registry_binds_the_real_kernel() {
        // The vertical slice: import the authored CPU elementwise-binary
        // contract through a REAL LinkRegistry that resolves add_f32's
        // entry-point symbol to the production kernel wrapper, register it,
        // and prove the registered binding IS that real kernel — by pointer
        // identity AND by executing it on real F32 storage.
        let link = EntryPointLink::new();
        let provider = import_bundle_str(ELEMENTWISE_BINARY, &link)
            .expect("authored elementwise-binary.fkc.md imports through the real link");

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into a fresh table succeeds");

        let key = [DType::F32, DType::F32, DType::F32];
        // The contract registers TWO alternatives at this key: the shared
        // `binary` chassis representative and the concrete `add_f32` — so
        // `lookup` (first alternative) returns the chassis. We assert the real
        // add_f32 wrapper is present *among the alternatives* and invoke it.
        let expected: KernelRef = crate::dispatch::add_elementwise_f32_cpu_wrapper;
        let alts = table.lookup_alternatives(OpKind::AddElementwise, &key, BackendId::Cpu);
        assert!(!alts.is_empty(), "AddElementwise/[F32,F32,F32]/Cpu present after FKC import");

        // (1) The FKC-imported binding IS the real production kernel — not a
        // stub, not a different fn — proven by pointer identity.
        let resolved: KernelRef = alts
            .iter()
            .map(|e| e.kernel)
            .find(|k| *k as usize == expected as usize)
            .expect("FKC import must bind the real add_f32 wrapper among the alternatives");

        // (2) It actually runs end-to-end on real F32 storage: [1,2,3]+[4,5,6].
        let a = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[1.0f32, 2.0, 3.0])));
        let b = Arc::new(RwLock::new(fuel_memory::from_slice_cpu(&[4.0f32, 5.0, 6.0])));
        let out = Arc::new(RwLock::new(
            fuel_memory::alloc_cpu_zeroed(DType::F32, 3).expect("alloc out"),
        ));
        let inputs = [a, b];
        let mut outputs = [out];
        let shape = fuel_ir::Shape::from_dims(&[3]);
        let layouts = [
            fuel_ir::Layout::contiguous(shape.clone()),
            fuel_ir::Layout::contiguous(shape),
        ];
        resolved(&inputs, &mut outputs, &layouts, &crate::kernel::OpParams::None)
            .expect("the FKC-resolved add_f32 kernel executes on real storage");
    }

    // =====================================================================
    // V-FKC-9 IMPORT-TIME GATE: unverified precision claims are downgraded
    // =====================================================================

    #[test]
    fn importing_elementwise_binary_downgrades_add_f32_against_the_ledger() {
        // The authored elementwise-binary contract declares add_f32
        // bit_stable_on_same_hardware + a max_ulp bound — but `add_f32` is a
        // PRIMITIVE `AddElementwise` op_kind contract, not one of the Task
        // 4.5b-seeded FUSED CPU ops, so the verification ledger has no
        // passing entry for it at ITS `kernel_revision_hash`. Task 4.3 wires
        // `gate_precision` into `import_bundle_str` so this import downgrades
        // it to UNAUDITED and records a warning, rather than trusting an
        // asserted-but-unverified claim.
        let link = EntryPointLink::new();
        let provider = import_bundle_str(ELEMENTWISE_BINARY, &link).expect("imports");
        let add = provider
            .primitives
            .iter()
            .find(|p| {
                p.op == OpKind::AddElementwise
                    && p.dtypes.as_slice() == [DType::F32, DType::F32, DType::F32]
            })
            .expect("add_f32 present");
        assert!(
            !add.precision.bit_stable_on_same_hardware,
            "unverified bit_stable claim must be downgraded at import"
        );
        assert!(
            add.precision.max_ulp.is_none(),
            "unverified ulp bound must be dropped"
        );
        assert_eq!(
            add.precision.notes,
            crate::fused::PrecisionGuarantee::UNAUDITED.notes
        );
        assert!(
            provider
                .warnings
                .iter()
                .any(|w| w.message.contains("bit_stable_on_same_hardware")),
            "a downgrade warning was recorded: {:?}",
            provider.warnings
        );
    }

    #[test]
    fn cpu_link_registry_binds_elementwise_binary_to_live_kernels() {
        // The CPU backend is the first FKC conformance reference: import its
        // authored elementwise-binary contract through the PRODUCTION
        // CpuLinkRegistry (not a stub) and verify every (op, dtype) section
        // resolves to the real production wrapper — proving an imported
        // contract describes, and binds to, the live CPU kernels.
        let provider = import_bundle_str(ELEMENTWISE_BINARY, &crate::fkc::CpuLinkRegistry)
            .expect("elementwise-binary imports through the production CpuLinkRegistry");
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into a fresh table succeeds");

        // All 8 ops × 4 dtypes are bound from the contract.
        let ops = [
            OpKind::AddElementwise,
            OpKind::SubElementwise,
            OpKind::MulElementwise,
            OpKind::DivElementwise,
            OpKind::MaximumElementwise,
            OpKind::MinimumElementwise,
            OpKind::PowElementwise,
            OpKind::RemElementwise,
        ];
        let dts = [DType::F32, DType::F64, DType::F16, DType::BF16];
        for op in ops {
            for dt in dts {
                assert!(
                    table.lookup(op, &[dt, dt, dt], BackendId::Cpu).is_ok(),
                    "{op:?}/{dt:?} must be bound from the imported contract",
                );
            }
        }

        // Spot-check two distinct (op, dtype) resolve to the EXACT production
        // wrappers — the link registry bound real symbols, not stubs.
        let add_f32 = table
            .lookup(OpKind::AddElementwise, &[DType::F32; 3], BackendId::Cpu)
            .unwrap();
        assert_eq!(
            add_f32 as usize,
            crate::dispatch::add_elementwise_f32_cpu_wrapper as usize,
        );
        let pow_bf16 = table
            .lookup(OpKind::PowElementwise, &[DType::BF16; 3], BackendId::Cpu)
            .unwrap();
        assert_eq!(
            pow_bf16 as usize,
            crate::dispatch::pow_elementwise_bf16_cpu_wrapper as usize,
        );

        assert!(table.finalize().is_ok(), "finalize is Ok after a real-link import");
    }

    // =====================================================================
    // FUSED SEAM: the production CpuLinkRegistry resolves a REAL authored
    // `fused_op` bundle's entry_point symbols to the live CPU fused wrappers
    // (the fused analogue of the elementwise-binary live-kernel test above).
    // =====================================================================

    /// The authored norm/softmax FUSED bundle — every section declares
    /// `fused_op:` (a `FusedOps::*` id), not `op_kind:`.
    const FUSED_NORM_SOFTMAX: &str =
        include_str!("../../../docs/kernel-contracts/fused/norm-softmax.fkc.md");

    #[test]
    fn cpu_link_registry_binds_norm_softmax_fused_ops_to_live_kernels() {
        // BORN-RED → GREEN (fused dtype-fan, §3.4). Each `fused_op` section
        // enumerates `dtypes: [F32, F64, BF16, F16]` on a BASE `entry_point`, so
        // `lower_fused` now FANS it into 4 per-dtype impls, each resolving
        // `<base>_<dt>` through `CPU_FUSED_NORM_ENTRY_POINTS`. Before the table
        // grew to its 32 per-dtype rows, the fanned symbols (e.g.
        // `…::softmax_last_dim_cpu_f32`) resolved to `None` → `UnknownEntryPoint`
        // and this import errored (RED). With the 32-row table every fanned
        // symbol binds the ACTUAL per-dtype production wrapper (GREEN) — a 1:1
        // replacement for the hand-written `register_default_fused_kernels`
        // per-dtype registrations, not just the F32 representative.
        let provider = import_bundle_str(FUSED_NORM_SOFTMAX, &crate::fkc::CpuLinkRegistry).expect(
            "authored norm-softmax fused bundle imports through the production CpuLinkRegistry",
        );
        // A fused-only bundle: no primitive `op_kind` sections. The 8 sections
        // each fan over 4 dtypes ⇒ 32 fused impls.
        assert!(
            provider.primitives.is_empty(),
            "norm-softmax declares only fused ops",
        );
        assert_eq!(
            provider.fused.len(),
            32,
            "8 fused sections × 4 dtypes = 32 per-dtype impls (fused dtype-fan)",
        );

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into fresh registries succeeds");

        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // Each FORWARD op (key `[dt, dt]`) binds all 4 dtypes to the EXACT
        // per-dtype production wrapper — proven by pointer identity — and carries
        // the contract's REAL revision hash (hand-written regs stamp UNTRACKED).
        let forward = [
            (
                FusedOps::SOFTMAX_LAST_DIM,
                [
                    crate::dispatch::softmax_last_dim_f32_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_f64_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_bf16_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::RMS_NORM_LAST_DIM,
                [
                    crate::dispatch::rms_norm_last_dim_f32_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_f64_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_bf16_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::LAYER_NORM_LAST_DIM,
                [
                    crate::dispatch::layer_norm_last_dim_f32_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_f64_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_bf16_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in forward {
            for (dt, expected) in dts.iter().zip(wrappers) {
                let got = fused
                    .lookup_by_dtypes(id, BackendId::Cpu, &[*dt, *dt])
                    .unwrap_or_else(|| panic!("{id:?} {dt:?} fused impl bound from the contract"));
                assert_eq!(
                    got.kernel as usize, expected,
                    "{id:?} {dt:?} binds its exact per-dtype production wrapper",
                );
                assert_ne!(
                    got.revision,
                    KernelRevisionHash::UNTRACKED,
                    "FKC import stamps the contract's real revision (hand-written stamps UNTRACKED)",
                );
            }
        }

        // Each BACKWARD op (key `[dt, dt, dt]`) binds all 4 dtypes to the EXACT
        // per-dtype production wrapper.
        let backward = [
            (
                FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                [
                    crate::dispatch::softmax_last_dim_backward_f32_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_backward_f64_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_backward_bf16_cpu_wrapper as usize,
                    crate::dispatch::softmax_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
                [
                    crate::dispatch::layer_norm_last_dim_backward_f32_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_backward_f64_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_backward_bf16_cpu_wrapper as usize,
                    crate::dispatch::layer_norm_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
                [
                    crate::dispatch::rms_norm_last_dim_backward_f32_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_backward_f64_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_backward_bf16_cpu_wrapper as usize,
                    crate::dispatch::rms_norm_last_dim_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::REDUCE_MAX_TO_BACKWARD,
                [
                    crate::dispatch::reduce_max_to_backward_f32_cpu_wrapper as usize,
                    crate::dispatch::reduce_max_to_backward_f64_cpu_wrapper as usize,
                    crate::dispatch::reduce_max_to_backward_bf16_cpu_wrapper as usize,
                    crate::dispatch::reduce_max_to_backward_f16_cpu_wrapper as usize,
                ],
            ),
            (
                FusedOps::POWI_BACKWARD,
                [
                    crate::dispatch::powi_backward_f32_cpu_wrapper as usize,
                    crate::dispatch::powi_backward_f64_cpu_wrapper as usize,
                    crate::dispatch::powi_backward_bf16_cpu_wrapper as usize,
                    crate::dispatch::powi_backward_f16_cpu_wrapper as usize,
                ],
            ),
        ];
        for (id, wrappers) in backward {
            for (dt, expected) in dts.iter().zip(wrappers) {
                let got = fused
                    .lookup_by_dtypes(id, BackendId::Cpu, &[*dt, *dt, *dt])
                    .unwrap_or_else(|| panic!("{id:?} {dt:?} fused impl bound from the contract"));
                assert_eq!(
                    got.kernel as usize, expected,
                    "{id:?} {dt:?} binds its exact per-dtype production wrapper",
                );
            }
        }
    }

    // =====================================================================
    // §5 RETURN-CONTRACT CROSS-CHECK: a fused contract's declared shape_rule /
    // dtype_rule must agree with the REAL registered `FusedOpEntry` fn — the
    // born-red test for Task 3.3 (Finding 5.1).
    // =====================================================================

    #[test]
    fn fused_contract_shape_rule_disagreeing_with_registered_fn_is_rejected() {
        // Real SoftmaxLastDim dtype_rule is passthrough(input0) = F32 at the F32 probe.
        // Mutate ONLY the first section's declared dtype_rule to fixed(F16): now it
        // disagrees with the registered fused fn at every probe combo → hard reject.
        let mutated = FUSED_NORM_SOFTMAX.replacen("dtype_rule: passthrough(x)", "dtype_rule: fixed(F16)", 1);
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a return rule that disagrees with the registered fused fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn unmutated_fused_corpus_still_imports_after_return_check() {
        import_bundle_str(FUSED_NORM_SOFTMAX, &crate::fkc::CpuLinkRegistry)
            .expect("real corpus return rules agree with the registered fused fns");
    }

    // =====================================================================
    // Convergence-C C-3: the shape-oracle cross-check ACTIVATED across the
    // remaining expressible fused ops. Each op below is proven to FIRE (not
    // merely import clean) by a MUTATION that makes the DECLARED shape rule
    // disagree with the REAL registered FusedOpEntry fn — a skipped/dormant
    // cross-check would let the mutated import succeed (the false-green risk).
    // The unmutated companions prove the real contracts AGREE.
    //
    // NOTE (adversarial fix, review Finding 2): the `const(N)` mutations below
    // change the output RANK, so they only ever proved RANK discrimination. The
    // GQA probe is now shape-DISTINCT (Hq = 2*Hkv), so a same-rank OPERAND-ROLE
    // drift is caught too — see `flash_attn_backward_k_operand_role_swap_is_rejected`
    // (dK `same_as(k)` mis-declared as `same_as(q)`), which used to import clean.
    // =====================================================================

    /// Finding 1 (arity pre-check): a synthetic 3-input FLASH_ATTN contract
    /// (q/k/v, no alibi) — the shape of the metal `matmul-attn.fkc.md`
    /// `sdpa_vector` sections. Its declared FLASH_ATTN registry fn
    /// `debug_assert!`s 4-or-5 inputs, so a 3-input probe panics in DEBUG (caught
    /// + skipped) but is stripped in RELEASE (fn reads operand 0) — the
    /// debug-vs-release split. The arity pre-check must skip it CONSISTENTLY in
    /// both build modes with a warning (not a silent debug-only skip).
    const SYNTH_FLASH_ATTN_3IN: &str = r#"---
fkc_version: 1
provider:
  name: synth-attn-3input
  backend: Cpu
  kernel_source: "portable-cpu"
---

# synthetic 3-input FLASH_ATTN (Finding 1 arity pre-check)

## flash_attn_3in

Synthetic 3-input flash attention (q, k, v) to exercise the arity pre-check.

```fkc
kernel: flash_attn_3in
fused_op: FLASH_ATTN
blurb: "synthetic 3-input flash attn (q,k,v) to exercise the arity pre-check"
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "synthetic::flash_attn_3in_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: DynScalar }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  in_place: false

cost:
  provenance: judge_measured
  class: attention

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "synthetic"

determinism: same_hardware_bitwise
```
"#;

    /// The three fused-op bundles whose cross-checks C-3 turns on.
    const FUSED_ATTENTION: &str =
        include_str!("../../../docs/kernel-contracts/fused/attention.fkc.md");
    const FUSED_LINEAR_QUANT: &str =
        include_str!("../../../docs/kernel-contracts/fused/linear-quant.fkc.md");
    const FUSED_CONV_ROPE: &str =
        include_str!("../../../docs/kernel-contracts/fused/conv-rope.fkc.md");

    /// Replace `from` → `to` exactly once, asserting `from` occurs EXACTLY once
    /// in `src` — so a mutation test can never silently no-op (a 0-count replace
    /// would leave the bundle unmutated and hide a cross-check that never fired).
    fn mutate_once(src: &str, from: &str, to: &str) -> String {
        assert_eq!(
            src.matches(from).count(),
            1,
            "mutation anchor `{from}` must occur EXACTLY once (got {})",
            src.matches(from).count()
        );
        src.replacen(from, to, 1)
    }

    /// A link registry that resolves EVERY symbol to a no-op kernel. The
    /// graph-side fused `attention.fkc.md` is a spec contract for the registry's
    /// `FusedOpEntry` (`shape_rule`/`dtype_rule`) and is NOT bound on the CPU
    /// `CpuLinkRegistry` (its CPU kernels come from the PRIMITIVE
    /// `cpu/attention.fkc.md`), so its `flash_attn_cpu_*` entry points do not
    /// resolve there. The cross-check runs BEFORE entry resolution and uses the
    /// REAL `default_registry()` fns regardless of link, so a resolve-anything
    /// stub isolates the cross-check under test from the (irrelevant here) CPU
    /// binding step.
    struct StubResolveAll;
    fn stub_kernel(
        _inputs: &[std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>],
        _outputs: &mut [std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_ir::Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }
    impl crate::fkc::lower::LinkRegistry for StubResolveAll {
        fn resolve_primitive(&self, _symbol: &str) -> Option<crate::kernel::KernelRef> {
            Some(stub_kernel)
        }
        fn resolve_fused(&self, _symbol: &str) -> Option<crate::kernel::KernelRef> {
            Some(stub_kernel)
        }
    }

    // ---- Tier 1: same_as / bundle cross-checks (synth-gated) ----

    #[test]
    fn inplace_affine_cross_check_fires() {
        // InplaceAffine now synth-supported; its `same_as(x)` (= input 0) is
        // evaluable. Mutating it to a rank-1 `const(9)` disagrees with the real
        // registry fn (which returns x's shape) at every probe → hard reject.
        let mutated = mutate_once(FUSED_LINEAR_QUANT, "shape_rule: same_as(x)", "shape_rule: const(9)");
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a mutated InplaceAffine shape_rule must be rejected (proves the check FIRES)");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn fused_linear_matmul_cross_check_fires() {
        // fused_linear declares the whole-shape `matmul(a, b)` rule (Tier-3),
        // now evaluable. The registry fn is `matmul_output_shape` (= a_batch ++
        // [M, N]); the UNMUTATED contract agrees (proven below). Mutate the rule
        // to `same_as(bias)` (rank-1 [N]) — disagrees with the rank-2 [M, N]
        // matmul output at every probe → hard reject (proves the check FIRES).
        let mutated = mutate_once(FUSED_LINEAR_QUANT, "shape_rule: matmul(a, b)", "shape_rule: same_as(bias)");
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a mutated fused_linear shape_rule must be rejected (proves the matmul check FIRES)");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn flash_attn_cross_check_fires() {
        // FlashAttn out = same_as(q) (Tier-2 doc fix from from_params(q)); the
        // registry fn returns input 0 (q). Mutate ONLY the value to a rank-1
        // const, preserving the `#` comment (the `q shape` tail is unique to
        // FlashAttn vs PagedAttn's `q (decode)` and dQ's `shape_rule_q`).
        let anchor = "same_as(q)                 # FusedOp.shape_rule = q shape";
        let mutated = mutate_once(FUSED_ATTENTION, anchor, "const(9) # FusedOp.shape_rule = q shape");
        let err = import_bundle_str(&mutated, &StubResolveAll)
            .expect_err("a mutated FlashAttn shape_rule must be rejected (proves the check FIRES)");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn paged_attn_cross_check_fires() {
        // PagedAttn out = same_as(q) (Tier-2 doc fix from from_params(q)); the
        // registry fn returns input 0 (q). Mutate ONLY the value, preserving the
        // `#` comment (`q (decode)` is unique to PagedAttn).
        let anchor = "same_as(q)                 # FusedOp.shape_rule = q (decode)";
        let mutated = mutate_once(FUSED_ATTENTION, anchor, "const(9) # FusedOp.shape_rule = q (decode)");
        let err = import_bundle_str(&mutated, &StubResolveAll)
            .expect_err("a mutated PagedAttn shape_rule must be rejected (proves the check FIRES)");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn flash_attn_backward_q_cross_check_fires() {
        // dQ = same_as(q) (= input 0). FLASH_ATTN_BACKWARD_{Q,K,V} share the
        // FlashAttnBackward variant, now synth-supported. Mutate dQ's rule.
        // (Stub link: attention is a graph-side spec contract, not CPU-bound.)
        let mutated = mutate_once(
            FUSED_ATTENTION,
            "shape_rule: same_as(q)                 # FusedOp.shape_rule_q",
            "shape_rule: const(9)                   # FusedOp.shape_rule_q",
        );
        let err = import_bundle_str(&mutated, &StubResolveAll)
            .expect_err("a mutated FlashAttnBackwardQ shape_rule must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn flash_attn_backward_k_cross_check_fires() {
        // dK = same_as(k) (= input 1); `same_as(k)` (parenthesised) is unique.
        let mutated = mutate_once(FUSED_ATTENTION, "shape_rule: same_as(k)", "shape_rule: const(9)");
        let err = import_bundle_str(&mutated, &StubResolveAll)
            .expect_err("a mutated FlashAttnBackwardK shape_rule must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn flash_attn_backward_v_cross_check_fires() {
        // dV = same_as(v) (= input 2); `same_as(v)` (parenthesised) is unique.
        let mutated = mutate_once(FUSED_ATTENTION, "shape_rule: same_as(v)", "shape_rule: const(9)");
        let err = import_bundle_str(&mutated, &StubResolveAll)
            .expect_err("a mutated FlashAttnBackwardV shape_rule must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn selective_scan_bundle_cross_check_fires() {
        // SelectiveScan now synth-supported ⇒ the bundle arity check runs.
        // Add a phantom 3rd slot: declared 3 ≠ real output_views arity 2.
        let ys = "- { name: y,          dtype_rule: passthrough(u), shape_rule: same_as(u),               layout_guarantee: contiguous }";
        let mutated = mutate_once(
            FUSED_CONV_ROPE,
            ys,
            &format!("{ys}\n    - {{ name: phantom, dtype_rule: passthrough(u), shape_rule: same_as(u), layout_guarantee: contiguous }}"),
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a SelectiveScan bundle with a phantom slot must be rejected");
        assert!(matches!(err, FkcError::BundleArityMismatch { .. }), "expected BundleArityMismatch, got {err:?}");
    }

    #[test]
    fn ssd_chunk_scan_bundle_cross_check_fires() {
        // SsdChunkScan now synth-supported ⇒ the bundle arity check runs.
        let ys = "- { name: y,          dtype_rule: passthrough(x), shape_rule: same_as(x),               layout_guarantee: contiguous }";
        let mutated = mutate_once(
            FUSED_CONV_ROPE,
            ys,
            &format!("{ys}\n    - {{ name: phantom, dtype_rule: passthrough(x), shape_rule: same_as(x), layout_guarantee: contiguous }}"),
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("an SsdChunkScan bundle with a phantom slot must be rejected");
        assert!(matches!(err, FkcError::BundleArityMismatch { .. }), "expected BundleArityMismatch, got {err:?}");
    }

    #[test]
    fn unmutated_c3_bundles_still_import_after_activation() {
        // The real contracts AGREE with the registered fns across every probe.
        // linear-quant + conv-rope are CPU-bound (production import path).
        import_bundle_str(FUSED_LINEAR_QUANT, &crate::fkc::CpuLinkRegistry)
            .expect("linear-quant bundle return rules agree with the registered fused fns");
        import_bundle_str(FUSED_CONV_ROPE, &crate::fkc::CpuLinkRegistry)
            .expect("conv-rope bundle return rules agree with the registered fused fns");
        // attention is a graph-side spec contract (not CPU-bound); the cross-check
        // still validates its declared rules against the registry fns.
        import_bundle_str(FUSED_ATTENTION, &StubResolveAll)
            .expect("attention bundle return rules agree with the registered fused fns");
    }

    // =====================================================================
    // Convergence-C C-3 adversarial fix — FINDING 1 (guard_rule warn +
    // arity pre-check closes the debug-vs-release split).
    // =====================================================================

    #[test]
    fn under_arity_flash_attn_skips_differential_consistently_and_warns() {
        // A 3-input FLASH_ATTN contract (the metal `matmul-attn` `sdpa_vector`
        // shape) feeds a 3-input probe to the registry flash_attn fn whose
        // `debug_assert!(len == 4 || 5)` PANICS in a debug build but is stripped
        // in release. The arity pre-check must skip the differential CONSISTENTLY
        // across build modes and record a warning (no silent, debug-only skip).
        // Import succeeds (never-panic) and the warning fires.
        let provider = import_bundle_str(SYNTH_FLASH_ATTN_3IN, &StubResolveAll)
            .expect("a 3-input FLASH_ATTN contract imports (no panic, no debug-only silent skip)");
        assert!(
            provider
                .warnings
                .iter()
                .any(|w| w.message.contains("requires >= 4") && w.message.contains("skipped")),
            "the under-arity attention differential must be skipped WITH a warning \
             (consistent debug/release): {:?}",
            provider.warnings,
        );
    }

    // =====================================================================
    // Convergence-C C-3 adversarial fix — FINDING 2 (GQA probe now
    // shape-DISTINCT: an operand-role swap in a return rule is caught).
    // =====================================================================

    #[test]
    fn flash_attn_backward_k_operand_role_swap_is_rejected() {
        // With the GQA probe now shape-DISTINCT (Hq = 2*Hkv via the cross-operand
        // `divisible(q.dim[1], k.dim[1])`), a realistic operand-role drift — dK's
        // `shape_rule: same_as(k)` mis-authored as `same_as(q)` — is genuinely
        // caught: the declared q shape [.,2*Hkv,.,.] differs from the registry dK
        // oracle (input 1 = k) [.,Hkv,.,.]. Before the fix q/k seeded identically
        // and this swap imported clean (the const(9) mutation only proved RANK
        // discrimination). Proves operand-role drift is now caught, not just rank.
        let mutated = mutate_once(
            FUSED_ATTENTION,
            "shape_rule: same_as(k)",
            "shape_rule: same_as(q)",
        );
        let err = import_bundle_str(&mutated, &StubResolveAll).expect_err(
            "dK same_as(k) mis-declared as same_as(q) must be rejected (operand-role drift)",
        );
        assert!(
            matches!(err, FkcError::ShapeRuleMismatch { .. }),
            "expected ShapeRuleMismatch, got {err:?}"
        );
    }

    // =====================================================================
    // Convergence-C C-3 adversarial fix — FINDING 3 (bundle slot shapes are
    // now differentially compared to the output_views oracle).
    // =====================================================================

    #[test]
    fn selective_scan_bundle_slot_content_drift_is_rejected() {
        // A SAME-RANK content drift in a bundle slot — SelectiveScan slot 0
        // declared `same_as(b)` instead of `same_as(u)` (both rank 3) — was
        // previously uncaught: the bundle block only checked arity + rank (<= 6),
        // never comparing an evaluated slot shape to the output_views oracle.
        // Over the now role-DISTINCT bundle probe (u [2,2,2] vs bumped b [5,2,2])
        // the drift diverges from the oracle (u-derived [2,2,2]) and is rejected.
        // Stays within §4 (reference = output_views, not decompose).
        let mutated = mutate_once(
            FUSED_CONV_ROPE,
            "shape_rule: same_as(u)",
            "shape_rule: same_as(b)",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry).expect_err(
            "a same-rank bundle slot content drift (same_as(u) -> same_as(b)) must be rejected",
        );
        assert!(
            matches!(err, FkcError::ShapeRuleMismatch { .. }),
            "expected ShapeRuleMismatch, got {err:?}"
        );
    }

    // =====================================================================
    // C-4 T3: the cross-check loops over synthesized param POINTS, so the
    // dtype differentials for the params-DEPENDENT fused ops (Conv2D /
    // ConvTranspose2D / QMatMul / FusedSoftmaxCrossEntropy / CausalConv1d)
    // are now LIVE. Each op is proven to FIRE by a dtype mutation that
    // disagrees with the real registered fn (a dormant differential would
    // let the mutated import succeed — the false-green risk); the unmutated
    // corpus, which agrees, stays importable WITHOUT any guard-caught panic
    // (no debug-only skips hiding under the differential).
    // =====================================================================

    /// EOL-normalize an embedded corpus file. `include_str!` carries the
    /// CHECKOUT's line endings (CRLF on an autocrlf Windows tree, LF
    /// elsewhere), so a multi-line mutation anchor written with `\n` would
    /// fail `mutate_once`'s exactly-once assert on one convention and pass on
    /// the other. Normalizing to LF makes the two-line anchors below
    /// checkout-independent; the FKC parser accepts either convention.
    fn lf(src: &str) -> String {
        src.replace("\r\n", "\n")
    }

    #[test]
    fn fsce_dtype_differential_fires() {
        // FSCE's real registered dtype fn is CONSTANT F32 (the loss
        // accumulator dtype), independent of inputs and params. A contract
        // declaring `fixed(F16)` must be rejected. Before C-4 T3 the FSCE
        // differential never ran at all (`synth_probe_params` returns None
        // for params-dependent variants), so this exact mutation imported
        // clean — the plan's mutation proof that enforcement is now live.
        let mutated = mutate_once(
            FUSED_LINEAR_QUANT,
            "dtype_rule: fixed(F32)          # ALWAYS F32 regardless of input dtype",
            "dtype_rule: fixed(F16)          # ALWAYS F32 regardless of input dtype",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("an FSCE dtype_rule disagreeing with the constant-F32 real fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn qmatmul_dtype_differential_fires() {
        // QMatMul's real dtype fn returns input 0 (`a`, F32 at every probe —
        // the contract declares a: [F32]). Declaring fixed(F16) disagrees.
        let mutated = mutate_once(
            FUSED_LINEAR_QUANT,
            "dtype_rule: fixed(F32)          # output is always F32 (dequant-and-contract); = a (F32)",
            "dtype_rule: fixed(F16)          # output is always F32 (dequant-and-contract); = a (F32)",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a QMatMul dtype_rule disagreeing with the real passthrough fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn conv2d_dtype_differential_fires() {
        // Conv2D's real dtype fn is passthrough(x) (= F32 at the probes; x's
        // first declared dtype). Declaring fixed(F64) disagrees. The bare
        // `dtype_rule: passthrough(x)` line is shared by 4 sections, so the
        // anchor spans into the section-unique shape_rule line (LF-normalized).
        let mutated = mutate_once(
            &lf(FUSED_CONV_ROPE),
            "dtype_rule: passthrough(x)\n      shape_rule: conv2d(params)",
            "dtype_rule: fixed(F64)\n      shape_rule: conv2d(params)",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a Conv2D dtype_rule disagreeing with the real passthrough fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn conv_transpose2d_dtype_differential_fires() {
        // ConvTranspose2D probes carry 3 operands (x, weight, optional bias —
        // the solver includes optional operands), so this differential also
        // pins the registry fn's 2-or-3 arity tolerance: were the fn still
        // `debug_assert_eq!(len, 2)`, a debug build would guard-catch the
        // panic and SKIP (import clean → this test fails), while release
        // checked — the exact debug-vs-release split the pre-check work bans.
        let mutated = mutate_once(
            &lf(FUSED_CONV_ROPE),
            "dtype_rule: passthrough(x)\n      shape_rule: conv_transpose2d(params)",
            "dtype_rule: fixed(F64)\n      shape_rule: conv_transpose2d(params)",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a ConvTranspose2D dtype_rule disagreeing with the real passthrough fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    #[test]
    fn causal_conv1d_dtype_differential_fires() {
        let mutated = mutate_once(
            &lf(FUSED_CONV_ROPE),
            "dtype_rule: passthrough(x)\n      shape_rule: from_params(seq_out)",
            "dtype_rule: fixed(F64)\n      shape_rule: from_params(seq_out)",
        );
        let err = import_bundle_str(&mutated, &crate::fkc::CpuLinkRegistry)
            .expect_err("a CausalConv1d dtype_rule disagreeing with the real passthrough fn must be rejected");
        assert!(matches!(err, FkcError::ShapeRuleMismatch { .. }), "expected ShapeRuleMismatch, got {err:?}");
    }

    /// C-4 T3 (regression + never-panic): the UNMUTATED corpus imports with
    /// its now-live dtype differentials actually RUNNING — and none of them
    /// leans on a guard-caught panic (a `guard_rule` catch would push a
    /// "panicked" warning and silently skip the differential in debug builds
    /// only). The params-dependent ops' `conv2d(params)`/`from_params(...)`
    /// SHAPE rules remain non-evaluable → documented skips, unchanged.
    #[test]
    fn unmutated_corpus_dtype_differentials_run_without_guard_panics() {
        for (src, name) in [(FUSED_LINEAR_QUANT, "linear-quant"), (FUSED_CONV_ROPE, "conv-rope")] {
            let provider = import_bundle_str(src, &crate::fkc::CpuLinkRegistry)
                .unwrap_or_else(|e| panic!("{name} must import clean: {e:?}"));
            let panicked: Vec<_> = provider
                .warnings
                .iter()
                .filter(|w| w.message.contains("panicked"))
                .collect();
            assert!(
                panicked.is_empty(),
                "{name}: live differentials must not rely on guard-caught panics \
                 (debug-only skips): {panicked:?}",
            );
        }
    }

    /// C-4 T3 (Conv2D arity pre-check): a synthetic 1-input CONV2D contract.
    /// The registry conv2d rule fns read `input_shapes[1]` / assert 2-or-3
    /// inputs, so an under-declared contract must be skipped CONSISTENTLY in
    /// debug and release with a warning — mirroring the attention-family
    /// pre-check (real 2-input CONV2D fused sections exist in
    /// `vulkan/conv-attn-rope.fkc.md`, so the under-declaration risk is live).
    const SYNTH_CONV2D_1IN: &str = r#"---
fkc_version: 1
provider:
  name: synth-conv2d-1input
  backend: Cpu
  kernel_source: "portable-cpu"
---

# synthetic 1-input CONV2D (C-4 T3 arity pre-check)

## conv2d_1in

Synthetic 1-input Conv2D (x only, no weight) to exercise the arity pre-check.

```fkc
kernel: conv2d_1in
fused_op: CONV2D
blurb: "synthetic 1-input conv2d (x only) to exercise the arity pre-check"
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "synthetic::conv2d_1in_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
  op_params:
    variant: Conv2D
    fields:
      stride:  { kind: "(usize, usize)" }
      padding: { kind: "(usize, usize)" }
      groups:  { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv2d(params)
      layout_guarantee: contiguous

caps:
  awkward_layout_strategy: requires_contiguous
  in_place: false

cost:
  provenance: judge_measured
  class: conv

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "synthetic"

determinism: same_hardware_bitwise
```
"#;

    #[test]
    fn under_arity_conv2d_skips_differential_consistently_and_warns() {
        // Without the pre-check a 1-input CONV2D contract skips SILENTLY
        // (the shape-coupled synth declines the unreadable combo), leaving no
        // trace of the dead differential. The pre-check surfaces the skip.
        let provider = import_bundle_str(SYNTH_CONV2D_1IN, &StubResolveAll)
            .expect("a 1-input CONV2D contract imports (never-panic; differential skipped)");
        assert!(
            provider
                .warnings
                .iter()
                .any(|w| w.message.contains("requires >= 2") && w.message.contains("skipped")),
            "the under-arity Conv2D differential must be skipped WITH a warning \
             (consistent debug/release): {:?}",
            provider.warnings,
        );
    }

    // =====================================================================
    // DUPLICATE: two sections at the same key+pointer → DuplicateKernelRef
    // =====================================================================

    #[test]
    fn same_pointer_on_one_key_surfaces_duplicate_kernel_ref() {
        // A hand-built bundle: two sections that resolve to the SAME
        // (op, dtypes, backend) key — and the SameLink maps both
        // entry_points to the SAME pointer. register_into's finalize gate
        // must surface DuplicateKernelRef.
        let src = "\
---
fkc_version: 1
provider:
  name: dup-provider
  backend: Cpu
  kernel_source: \"dup-cpu\"
---

# dup bundle

## add_a

A.

```fkc
kernel: add_a
op_kind: AddElementwise
blurb: \"a\"
entry_point: \"x::add_a\"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: \"n\"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```

## add_b

B.

```fkc
kernel: add_b
op_kind: AddElementwise
blurb: \"b\"
entry_point: \"x::add_b\"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: \"n\"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
";
        let provider =
            import_bundle_str(src, &SameLink).expect("the dup bundle imports (lowering is fine)");
        // Both sections share (AddElementwise, [F32,F32,F32], Cpu) AND the
        // same pointer → finalize must reject.
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        let err = provider
            .register_into(&mut table, &mut fused)
            .expect_err("same pointer on one key must error");
        assert!(
            matches!(err, FkcError::DuplicateKernelRef(_)),
            "got {err:?}"
        );
    }

    // =====================================================================
    // COST COMPILER (§2.3, FKC gap-closure Task 2.2): a contract's declared
    // `cost.flops` expression must reach `BindingEntry.cost_expr` as a
    // compiled AST, not be dropped in favor of the `unknown_cost` sentinel.
    // =====================================================================

    #[test]
    fn imported_contract_declared_cost_reaches_binding_cost_fn() {
        let src = "---\nfkc_version: 1\nprovider:\n  name: cost-provider\n  backend: Cpu\n  kernel_source: \"cost-cpu\"\n---\n\n# cost bundle\n\n## add_f32\n\nA.\n\n```fkc\nkernel: add_f32\nop_kind: AddElementwise\nblurb: \"a\"\nentry_point: \"x::add_f32\"\naccept:\n  inputs:\n    - name: lhs\n      dtypes: [F32]\n      layout: { contiguous: required, strided: rejected }\n    - name: rhs\n      dtypes: [F32]\n      layout: { contiguous: required, strided: rejected }\n  op_params: { variant: None }\nreturn:\n  outputs:\n    - name: out\n      dtype_rule: passthrough(lhs)\ncost:\n  provenance: declared\n  class: cheap_elementwise\n  flops: \"n\"\nprecision:\n  bit_stable_on_same_hardware: true\n  audited: true\ndeterminism: same_hardware_bitwise\n```\n";
        let provider = import_bundle_str(src, &SameLink).expect("declared-cost contract imports");
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider.register_into(&mut table, &mut fused).expect("registers");
        let alts = table.lookup_alternatives(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu);
        let entry = alts.first().expect("binding present");
        let expr = entry.cost_expr.expect("declared flops formula reaches the binding as a compiled AST");
        let est = crate::fkc::cost_estimate(expr, OpKind::AddElementwise, &[fuel_ir::Shape::from_dims(&[4])],
            &[DType::F32, DType::F32, DType::F32], &crate::kernel::OpParams::None).expect("declared cost evaluates");
        assert_eq!(est.flops, 4, "flops = n = elem_count([4])");
    }

    // =====================================================================
    // GLOB: agreeing front-matter merges; a mismatch → ProviderMismatch
    // =====================================================================

    fn write_temp(dir: &Path, name: &str, contents: &str) {
        std::fs::write(dir.join(name), contents).expect("write temp fkc file");
    }

    fn bundle_with(provider_backend: &str, kernel: &str, op_kind: &str, entry: &str) -> String {
        // A raw string with real newlines (NO `\`-continuations, which
        // would eat the YAML indentation).
        let template = r#"---
fkc_version: 1
provider:
  name: glob-provider
  backend: __BACKEND__
  kernel_source: "glob-cpu"
---

# glob bundle

## __KERNEL__

blurb.

```fkc
kernel: __KERNEL__
op_kind: __OP_KIND__
blurb: "k"
entry_point: "__ENTRY__"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#;
        template
            .replace("__BACKEND__", provider_backend)
            .replace("__KERNEL__", kernel)
            .replace("__OP_KIND__", op_kind)
            .replace("__ENTRY__", entry)
    }

    #[test]
    fn import_glob_merges_agreeing_files() {
        let dir = std::env::temp_dir().join(format!("fkc_glob_ok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        write_temp(
            &dir,
            "a.fkc.md",
            &bundle_with("Cpu", "add_f32", "AddElementwise", "x::add"),
        );
        write_temp(
            &dir,
            "b.fkc.md",
            &bundle_with("Cpu", "sub_f32", "SubElementwise", "x::sub"),
        );

        let link = DistinctLink::new();
        let pattern = dir.join("*.fkc.md").display().to_string();
        let provider = import_glob(&pattern, &link).expect("two agreeing files merge");

        assert_eq!(provider.name, "glob-provider");
        assert_eq!(provider.backend, BackendId::Cpu);
        assert_eq!(provider.primitives.len(), 2, "both kernels present");

        // And they register end-to-end.
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider.register_into(&mut table, &mut fused).expect("merged provider registers");
        assert!(table
            .lookup(OpKind::AddElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .is_ok());
        assert!(table
            .lookup(OpKind::SubElementwise, &[DType::F32, DType::F32, DType::F32], BackendId::Cpu)
            .is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_glob_mismatched_backend_is_provider_mismatch() {
        let dir = std::env::temp_dir().join(format!("fkc_glob_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Two agreeing CPU files + a third with a different backend.
        write_temp(
            &dir,
            "a.fkc.md",
            &bundle_with("Cpu", "add_f32", "AddElementwise", "x::add"),
        );
        write_temp(
            &dir,
            "b.fkc.md",
            &bundle_with("Cpu", "sub_f32", "SubElementwise", "x::sub"),
        );
        write_temp(
            &dir,
            "c.fkc.md",
            &bundle_with("Vulkan", "mul_f32", "MulElementwise", "x::mul"),
        );

        let link = DistinctLink::new();
        let pattern = dir.join("*.fkc.md").display().to_string();
        let err = import_glob(&pattern, &link).expect_err("mismatched backend must error");
        assert!(
            matches!(
                err,
                FkcError::ProviderMismatch { ref field, .. } if field == "provider.backend"
            ),
            "got {err:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // =====================================================================
    // DESCRIBE-ONLY (§3.10): a registrable: false section is excluded from
    // register_into; a sibling registrable section still registers.
    // =====================================================================

    #[test]
    fn describe_only_section_is_excluded_from_register_into() {
        // A bundle with a describe-only chassis umbrella (`## binary`,
        // op_kind = the DESCRIPTIVE token `binary`, registrable: false) plus
        // one real registrable thunk (`## add_f32`). Only the thunk lowers +
        // registers; the umbrella is documentation.
        let src = "\
---
fkc_version: 1
provider:
  name: describe-provider
  backend: Cpu
  kernel_source: \"describe-cpu\"
---

# describe bundle

## binary

The shared chassis (documentation umbrella).

```fkc
kernel: binary
registrable: false
op_kind: binary
blurb: \"shared binary chassis\"
entry_point: \"x::chassis\"
accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: judge_measured
  class: cheap_elementwise
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```

## add_f32

The real F32 addition thunk.

```fkc
kernel: add_f32
op_kind: AddElementwise
blurb: \"F32 add\"
entry_point: \"x::add_f32\"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: \"n\"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
";
        let link = DistinctLink::new();
        let provider = import_bundle_str(src, &link)
            .expect("describe-only + registrable bundle imports (umbrella is valid docs)");

        // Only the registrable thunk lowered — the describe-only umbrella is
        // excluded from the registered set.
        assert_eq!(
            provider.primitives.len(),
            1,
            "exactly one registrable primitive (the umbrella is documentation-only)"
        );
        assert_eq!(provider.primitives[0].op, OpKind::AddElementwise);

        // It registers end-to-end and the real key is present.
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into succeeds (no describe-only section reaches the table)");
        assert!(table
            .lookup(
                OpKind::AddElementwise,
                &[DType::F32, DType::F32, DType::F32],
                BackendId::Cpu
            )
            .is_ok());
    }

    // =====================================================================
    // DESCRIBE-ONLY GATHER (§3.10 + §14): a `registrable: false` section that
    // trips a consumer-ahead validator (FDX gather → GatherNotYetSupported)
    // must NOT block a bundle's importable sections. The valid sibling still
    // imports + registers; the describe-only gather section is documentation.
    // =====================================================================

    #[test]
    fn describe_only_gather_section_does_not_block_bundle_import() {
        // A bundle with (a) one VALID registrable op_kind section (add_f32) and
        // (b) a `registrable: false` section whose operand carries an
        // `fdx.gather: paged_blocks` block — which the VALIDATE pass flags
        // `GatherNotYetSupported` (§14, consumer-ahead). Import must SKIP the
        // describe-only section's validation (it is documentation, carries no
        // dispatch target) and still register the valid sibling.
        //
        // **Born-red**: before the validate-skip fix, `validate_file` walked
        // EVERY section (incl. describe-only) and `validate_kernel` ran the gather
        // coherence check on the describe-only section → the whole
        // `import_bundle_str` returned `GatherNotYetSupported` (RED). After the
        // fix (`validate_file` skips `registrable: false` sections) → the valid
        // section imports + registers (GREEN).
        let src = r#"---
fkc_version: 1
provider:
  name: gather-describe-provider
  backend: Cpu
  kernel_source: "gather-cpu"
---

# describe-only gather bundle

## add_f32

The real F32 addition thunk (registrable).

```fkc
kernel: add_f32
op_kind: AddElementwise
blurb: "F32 add"
entry_point: "x::add_f32"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```

## paged_gather

Describe-only: a paged KV pool with an FDX gather sidecar (consumer-ahead).

```fkc
kernel: paged_gather
registrable: false
op_kind: PagedAttn
blurb: "describe-only paged gather"
entry_point: "x::paged_gather"
accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: k_cache
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected }
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: PagedAttn }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
cost:
  provenance: declared
  class: attention
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#;
        let link = DistinctLink::new();
        let provider = import_bundle_str(src, &link).expect(
            "a describe-only gather section must NOT block the bundle's importable sections",
        );

        // Only the registrable add_f32 lowered; the describe-only gather section
        // is documentation (excluded from the registered set).
        assert_eq!(
            provider.primitives.len(),
            1,
            "exactly one registrable primitive (the gather section is describe-only)",
        );
        assert_eq!(provider.primitives[0].op, OpKind::AddElementwise);

        // The valid sibling registers end-to-end.
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("the valid sibling registers end-to-end");
        assert!(table
            .lookup(
                OpKind::AddElementwise,
                &[DType::F32, DType::F32, DType::F32],
                BackendId::Cpu
            )
            .is_ok());
    }

    // =====================================================================
    // FUSED: a hand-built ResolvedFused registers into the fused registry
    // =====================================================================

    #[test]
    fn hand_built_fused_registers_into_registry() {
        // The fused corpus declares `fused_op:` names that don't match the
        // registry's CamelCase entry names today, so we cover the fused
        // register path with a hand-built ResolvedFused (faithful to the
        // shape `register_into` consumes).
        let mut dtypes: KernelDTypes = SmallVec::new();
        dtypes.push(DType::F32); // input x
        dtypes.push(DType::F32); // output

        let f = ResolvedFused {
            id: FusedOps::SOFTMAX_LAST_DIM,
            dtypes,
            backend: BackendId::Cpu,
            caps: KernelCaps::empty(),
            layouts: vec![ResolvedLayout {
                contiguous: Tri::Required,
                strided: Tri::Rejected,
                broadcast_stride0: Tri::Rejected,
                start_offset: Tri::Rejected,
                reverse_strides: Tri::Rejected,
            }],
            precision: PrecisionGuarantee::UNAUDITED,
            cost: CompiledCostExpr::Unknown,
            kernel: dummy_a,
            kernel_source: "portable-cpu".to_string(),
            revision: KernelRevisionHash::UNTRACKED,
            variant: None,
            bundle_slot_names: Vec::new(),
        };

        let provider = ImportedProvider {
            name: "fuel-fused-registry".to_string(),
            backend: BackendId::Cpu,
            kernel_source: intern("portable-cpu"),
            primitives: Vec::new(),
            fused: vec![f],
            warnings: Vec::new(),
        };

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("fused-only provider registers");

        // The fused op is present in the registry.
        let got = fused.lookup_by_dtypes(
            FusedOps::SOFTMAX_LAST_DIM,
            BackendId::Cpu,
            &[DType::F32, DType::F32],
        );
        assert!(got.is_some(), "softmax fused impl registered + looked up");
        let impl_ = got.unwrap();
        // The interned dtypes round-trip.
        assert_eq!(impl_.dtypes, &[DType::F32, DType::F32]);
        // The fused-cost sentinel pointer was wired ([consumer-ahead]).
        let sentinel = fused_unknown_cost as *const () as usize;
        assert_eq!(
            impl_.cost as *const () as usize, sentinel,
            "the fused-unknown sentinel CostFn is wired for imported fused ops"
        );
        // The revision rode through unchanged.
        assert_eq!(impl_.revision, KernelRevisionHash::UNTRACKED);
    }

    // =====================================================================
    // MULTI-DTYPE FAN-OUT (§3.4): a section whose operand(s) vary fans out
    // into N per-dtype bindings; `passthrough(role)` resolves the named
    // operand's dtype (NOT blindly the first input — the "where bug").
    // =====================================================================

    /// A [`LinkRegistry`] that resolves EVERY requested symbol to a dummy
    /// pointer (keyed by the symbol string so a symbol maps stably) and
    /// RECORDS every requested symbol — so a test can assert exactly which
    /// `<base>_<suffix>` entry points the importer resolved. Permissive on
    /// purpose: it resolves the base symbol too, so the pre-change importer
    /// (which resolves the base as-is) still imports — letting the born-red
    /// fail on a clean binding-COUNT assertion, not an import error.
    struct FanStubLink {
        seen: Mutex<std::collections::HashMap<String, KernelRef>>,
        requested: Mutex<Vec<String>>,
    }
    impl FanStubLink {
        fn new() -> Self {
            Self {
                seen: Mutex::new(std::collections::HashMap::new()),
                requested: Mutex::new(Vec::new()),
            }
        }
        fn resolve(&self, symbol: &str) -> Option<KernelRef> {
            self.requested.lock().unwrap().push(symbol.to_string());
            let mut g = self.seen.lock().unwrap();
            if let Some(k) = g.get(symbol) {
                return Some(*k);
            }
            let k: KernelRef = if g.len() % 2 == 0 { dummy_a } else { dummy_b };
            g.insert(symbol.to_string(), k);
            Some(k)
        }
    }
    impl LinkRegistry for FanStubLink {
        fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
        fn resolve_fused(&self, symbol: &str) -> Option<KernelRef> {
            self.resolve(symbol)
        }
    }

    /// A synthetic bundle with (a) a UNIFORM multi-dtype section (`relu`: one
    /// input varying over `[F32,F64,BF16,F16]`, base `entry_point`) and (b) a
    /// MIXED section (`where_op`: a fixed `U8` `cond` + two varying operands +
    /// `passthrough(a)`). Authored with BASE entry_points (no dtype suffix) so
    /// the importer must resolve `<base>_<suffix>` per fanned dtype.
    const FANOUT_CONTRACT: &str = r#"---
fkc_version: 1
provider:
  name: fanout-provider
  backend: Cpu
  kernel_source: "fanout-cpu"
---

# fanout bundle

## relu

Uniform multi-dtype unary: one input varying over 4 dtypes.

```fkc
kernel: relu
op_kind: ReluElementwise
blurb: "relu fan-out"
entry_point: "stub::relu"
accept:
  inputs:
    - name: in
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```

## where_op

Mixed: a fixed U8 cond + two varying operands; passthrough(a).

```fkc
kernel: where_op
op_kind: Where
blurb: "where fan-out with passthrough(a)"
entry_point: "stub::where"
accept:
  inputs:
    - name: cond
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected }
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected }
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#;

    #[test]
    fn multi_dtype_section_fans_out_into_per_dtype_bindings() {
        let link = FanStubLink::new();
        let provider =
            import_bundle_str(FANOUT_CONTRACT, &link).expect("fan-out contract imports");

        // (1) FAN-OUT: `relu` (1 varying input) + `where_op` (2 varying inputs)
        // each fan to 4 per-dtype bindings ⇒ 8 primitives. Pre-change the
        // importer keyed on each operand's FIRST dtype and registered ONE
        // binding per section → 2. RED→GREEN.
        assert_eq!(
            provider.primitives.len(),
            8,
            "relu×4 + where×4 = 8 fanned bindings (pre-change: 2 — first-dtype only)"
        );

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("fanned bindings register");

        let dts = [DType::F32, DType::F64, DType::BF16, DType::F16];

        // (2) UNIFORM fan-out: ReluElementwise bound at [dt, dt] for every dt.
        for dt in dts {
            assert!(
                table
                    .lookup(OpKind::ReluElementwise, &[dt, dt], BackendId::Cpu)
                    .is_ok(),
                "relu must be bound at [{dt:?}, {dt:?}]",
            );
        }

        // (3) MIXED fan-out + passthrough-role FIX: Where bound at
        // [U8, dt, dt, dt] — the output mirrors operand `a` (= dt), NOT the
        // first input `cond` (U8).
        for dt in dts {
            assert!(
                table
                    .lookup(OpKind::Where, &[DType::U8, dt, dt, dt], BackendId::Cpu)
                    .is_ok(),
                "where must be bound at [U8, {dt:?}, {dt:?}, {dt:?}] (passthrough(a) = a's dtype)",
            );
            // The OLD buggy key (passthrough mirrored cond=U8) must NOT exist.
            assert!(
                table
                    .lookup(OpKind::Where, &[DType::U8, dt, dt, DType::U8], BackendId::Cpu)
                    .is_err(),
                "the pre-fix buggy key [U8,{dt:?},{dt:?},U8] must NOT be registered",
            );
        }

        // (4) ENTRY-POINT base+suffix: the importer resolved `<base>_<suffix>`
        // for every fanned dtype (canonical `DType::as_str`, not a hand-rolled
        // spelling), NOT the bare base symbol.
        let requested = link.requested.lock().unwrap().clone();
        for suffix in ["f32", "f64", "bf16", "f16"] {
            assert!(
                requested.iter().any(|s| s == &format!("stub::relu_{suffix}")),
                "relu variant must resolve stub::relu_{suffix}; requested={requested:?}",
            );
            assert!(
                requested.iter().any(|s| s == &format!("stub::where_{suffix}")),
                "where variant must resolve stub::where_{suffix}; requested={requested:?}",
            );
        }
    }

    /// Backward-compat guard: a section with NO varying operand (every operand
    /// single-dtype) produces EXACTLY ONE binding and resolves its declared
    /// `entry_point` AS-IS (no `_<suffix>` appended). This is what keeps the
    /// already-migrated per-(op,dtype) binary / affine / cast families
    /// byte-identical under fan-out. GREEN before AND after the change.
    #[test]
    fn single_dtype_section_yields_exactly_one_unchanged_binding() {
        let src = r#"---
fkc_version: 1
provider:
  name: single-provider
  backend: Cpu
  kernel_source: "single-cpu"
---

# single bundle

## add_f32

```fkc
kernel: add_f32
op_kind: AddElementwise
blurb: "single-dtype add"
entry_point: "stub::add_f32"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#;
        let link = FanStubLink::new();
        let provider = import_bundle_str(src, &link).expect("single-dtype imports");
        assert_eq!(
            provider.primitives.len(),
            1,
            "no varying operand ⇒ exactly one binding"
        );
        assert_eq!(
            provider.primitives[0].dtypes.as_slice(),
            &[DType::F32, DType::F32, DType::F32],
        );

        // entry_point resolved AS-IS (no `_f32` suffix appended to a
        // single-variant section — its declared symbol is already specific).
        let requested = link.requested.lock().unwrap().clone();
        assert!(
            requested.iter().any(|s| s == "stub::add_f32"),
            "single-variant resolves the declared symbol as-is; requested={requested:?}",
        );
        assert!(
            !requested.iter().any(|s| s == "stub::add_f32_f32"),
            "must NOT append a suffix to a single-variant section",
        );
    }

    // =====================================================================
    // OPTIONAL-OPERAND FAN-OUT (§3.4): an `optional: true` LAST input fans
    // EACH dtype variant into TWO keys — one OMITTING the optional operand,
    // one INCLUDING it — BOTH bound to the SAME kernel_ref. Composes with
    // the dtype fan-out (multi-dtype AND optional ⇒ 2N bindings).
    // =====================================================================

    /// A synthetic single-dtype contract: one REQUIRED input `x` + one
    /// `optional: true` LAST input `bias`, output `passthrough(x)`. With
    /// optional-operand support the importer registers TWO primitive bindings
    /// — no-bias `[x, out]` = `[F32, F32]` and with-bias `[x, bias, out]` =
    /// `[F32, F32, F32]` — BOTH resolving the SAME (single, as-is) `entry_point`
    /// ⇒ the SAME `KernelRef`.
    ///
    /// **Born-red**: BEFORE the fix the schema `TensorDesc` has no `optional`
    /// field, so serde silently drops `optional: true`; the operand is treated
    /// as REQUIRED and ONLY the with-bias key builds → `primitives.len() == 1`
    /// (RED). After the schema+key-builder fix → 2 (GREEN).
    const OPTIONAL_LAST_OPERAND_CONTRACT: &str = r#"---
fkc_version: 1
provider:
  name: opt-provider
  backend: Cpu
  kernel_source: "opt-cpu"
---

# optional-operand bundle

## opt_add

One required input + one optional LAST input.

```fkc
kernel: opt_add
op_kind: AddElementwise
blurb: "optional last operand"
entry_point: "stub::opt_add"
accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: bias
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
      optional: true
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: "n"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
"#;

    #[test]
    fn optional_last_operand_registers_both_with_and_without_keys() {
        let link = FanStubLink::new();
        let provider = import_bundle_str(OPTIONAL_LAST_OPERAND_CONTRACT, &link)
            .expect("optional-operand contract imports");

        // TWO bindings: the with-optional key AND the without-optional key.
        // Pre-change (no `optional` schema field): serde drops the flag, the
        // operand is required, only the with-operand key builds → 1. RED→GREEN.
        assert_eq!(
            provider.primitives.len(),
            2,
            "an optional LAST input fans into 2 keys (with + without the operand); \
             pre-change (no `optional` schema field) only the with-operand key builds → 1",
        );

        // BOTH keys bind the SAME kernel_ref (same entry_point resolved AS-IS —
        // single-dtype, so no `_<suffix>` is appended).
        let k0 = provider.primitives[0].kernel as usize;
        let k1 = provider.primitives[1].kernel as usize;
        assert_eq!(
            k0, k1,
            "both the with- and without-optional keys resolve the SAME kernel (one entry_point)",
        );

        // The two keys are exactly `[x, out]` and `[x, bias, out]`.
        let mut keys: Vec<Vec<DType>> =
            provider.primitives.iter().map(|p| p.dtypes.to_vec()).collect();
        keys.sort_by_key(|k| k.len());
        assert_eq!(
            keys[0],
            vec![DType::F32, DType::F32],
            "no-optional key = [x, out]",
        );
        assert_eq!(
            keys[1],
            vec![DType::F32, DType::F32, DType::F32],
            "with-optional key = [x, bias, out]",
        );

        // Both register into the binding table + look up.
        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("both keys register");
        assert!(
            table
                .lookup(OpKind::AddElementwise, &[DType::F32, DType::F32], BackendId::Cpu)
                .is_ok(),
            "no-optional key [F32, F32] is bound",
        );
        assert!(
            table
                .lookup(
                    OpKind::AddElementwise,
                    &[DType::F32, DType::F32, DType::F32],
                    BackendId::Cpu,
                )
                .is_ok(),
            "with-optional key [F32, F32, F32] is bound",
        );

        // The declared entry_point resolved AS-IS (single-dtype ⇒ no suffix).
        let requested = link.requested.lock().unwrap().clone();
        assert!(
            requested.iter().any(|s| s == "stub::opt_add"),
            "single-dtype section resolves its declared symbol as-is; requested={requested:?}",
        );

        // BACKWARD-COMPAT: the SAME contract WITHOUT the `optional: true` flag is
        // a plain 2-input section → EXACTLY ONE binding (the fan is driven ONLY
        // by the optional flag; nothing else changed).
        let non_optional =
            OPTIONAL_LAST_OPERAND_CONTRACT.replace("      optional: true\n", "");
        let link2 = FanStubLink::new();
        let provider2 =
            import_bundle_str(&non_optional, &link2).expect("non-optional twin imports");
        assert_eq!(
            provider2.primitives.len(),
            1,
            "no optional operand ⇒ exactly one binding (backward-compat)",
        );
        assert_eq!(
            provider2.primitives[0].dtypes.to_vec(),
            vec![DType::F32, DType::F32, DType::F32],
            "the required-both key [x, bias, out]",
        );
    }

    // =====================================================================
    // COST-FN TRAMPOLINE (§4.4, Task-F): a contract's `cost.cost_fn` NAME
    // resolves through the LinkRegistry to a real CostFn that is stamped on
    // the binding and SURVIVES the fill_unset pass. The LinkRegistry analog
    // of entry_point resolution — a named cost model, not the unknown_cost
    // sentinel. An unknown name is a typed UnknownCostFn (never silent).
    // =====================================================================

    /// A distinctive test `CostFn` (distinct fn item ⇒ distinct pointer) that
    /// the trampoline pins onto a binding. Its 999_999 FLOPs value + its unique
    /// pointer identity make it observable that the CONTRACT's cost fn — not
    /// `unknown_cost` and not the op-family default — is what the binding
    /// carries after `register_into` AND after `fill_unset`.
    fn test_pinned_cost(
        _s: &[fuel_ir::Shape],
        _d: &[DType],
        _p: &crate::kernel::OpParams,
        _c: &fuel_ir::backend::BackendCapabilities,
    ) -> CostEstimate {
        CostEstimate { flops: 999_999, bytes_moved: 0, kernel_overhead_ns: 0 }
    }

    /// A `LinkRegistry` that resolves the `cost.cost_fn` name `test_pinned_cost`
    /// to [`test_pinned_cost`] (and every entry_point to a dummy kernel) — the
    /// test analog of a provider that pins a real cost fn over its own cost-fn
    /// table (e.g. the CUDA `CUDA_COST_FNS` table).
    struct CostPinLink;
    impl LinkRegistry for CostPinLink {
        fn resolve_primitive(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_a)
        }
        fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
            Some(dummy_a)
        }
        fn resolve_cost_fn(&self, name: &str) -> Option<crate::kernel::CostFn> {
            if name == "test_pinned_cost" {
                Some(test_pinned_cost)
            } else {
                None
            }
        }
    }

    /// A one-section bundle naming a cost fn in its `cost:` block. `__COST_FN__`
    /// is replaced by the test with the real (resolvable) or a bogus name.
    const COST_PIN_BUNDLE: &str = "\
---
fkc_version: 1
provider:
  name: cost-pin-provider
  backend: Cpu
  kernel_source: \"cost-pin-cpu\"
---

# cost-pin bundle

## add_pinned

A.

```fkc
kernel: add_pinned
op_kind: AddElementwise
blurb: \"a\"
entry_point: \"x::add_pinned\"
accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected }
  op_params: { variant: None }
return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
cost:
  provenance: declared
  class: cheap_elementwise
  flops: \"n\"
  cost_fn: \"__COST_FN__\"
precision:
  bit_stable_on_same_hardware: true
  audited: true
determinism: same_hardware_bitwise
```
";

    #[test]
    fn contract_pinned_cost_fn_survives_fill_unset() {
        // BORN-RED → GREEN. A contract whose cost block NAMES `test_pinned_cost`
        // must register the binding with THAT CostFn (resolved through the
        // LinkRegistry), and the `fill_unset_cost_for_backend` pass must LEAVE it
        // (that pass only replaces the `unknown_cost` sentinel). RED before the
        // trampoline wiring: `register_into` stamped `unknown_cost`, so fill_unset
        // clobbered the cell with the op-family default `cost_elementwise_binary_cpu`.
        let src = COST_PIN_BUNDLE.replace("__COST_FN__", "test_pinned_cost");
        let provider = import_bundle_str(&src, &CostPinLink)
            .expect("a bundle naming a resolvable cost fn imports");

        // (1) Lowering resolved the NAME to the concrete CostFn on the record.
        assert_eq!(provider.primitives.len(), 1);
        assert!(
            provider.primitives[0].cost_fn.is_some(),
            "lower_primitive must resolve cost.cost_fn through resolve_cost_fn",
        );

        let mut table = KernelBindingTable::new();
        let mut fused = FusedKernelRegistry::new();
        provider
            .register_into(&mut table, &mut fused)
            .expect("register_into succeeds");

        let key = [DType::F32, DType::F32, DType::F32];

        // (2) register_into stamped the PINNED cost fn, NOT the unknown_cost sentinel.
        let cost_before = table.lookup_cost(OpKind::AddElementwise, &key, BackendId::Cpu);
        assert_eq!(
            cost_before as usize, test_pinned_cost as usize,
            "register_into must stamp the contract-pinned CostFn (not unknown_cost)",
        );
        assert_ne!(
            cost_before as usize,
            crate::kernel::unknown_cost as usize,
            "the pinned CostFn must not be the unknown_cost sentinel",
        );

        // (3) THE POINT OF TASK-F: fill_unset must NOT overwrite the pinned cost.
        table.fill_unset_cpu_cost(crate::cost::default_cost_for_op_kind);
        let cost_after = table.lookup_cost(OpKind::AddElementwise, &key, BackendId::Cpu);
        assert_eq!(
            cost_after as usize, test_pinned_cost as usize,
            "fill_unset must leave the contract-pinned CostFn untouched",
        );
        assert_ne!(
            cost_after as usize,
            crate::cost::cost_elementwise_binary_cpu as usize,
            "the pinned cost must NOT have been clobbered by the op-family default",
        );

        // (4) FKC gap-closure Task 2.2: this contract ALSO declares
        // `flops: "n"` alongside the pinned `cost_fn` — the pinned fn must
        // win outright, so the binding's declared-AST slot stays None (the
        // AST does not compete with a contract-pinned CostFn).
        let alts = table.lookup_alternatives(OpKind::AddElementwise, &key, BackendId::Cpu);
        let entry = alts.first().expect("binding present");
        assert!(
            entry.cost_expr.is_none(),
            "a pinned cost_fn wins outright; the declared cost AST must not be stamped alongside it",
        );
    }

    #[test]
    fn unknown_cost_fn_name_is_typed_error() {
        // A cost block naming a cost fn absent from the link registry's cost-fn
        // table is a typed UnknownCostFn at import time — never a silent
        // fallback to unknown_cost.
        let src = COST_PIN_BUNDLE.replace("__COST_FN__", "no_such_cost_fn");
        let err = import_bundle_str(&src, &CostPinLink)
            .expect_err("an unresolved cost_fn name must error");
        assert!(
            matches!(err, FkcError::UnknownCostFn { ref cost_fn, .. } if cost_fn == "no_such_cost_fn"),
            "got {err:?}",
        );
    }

    // =====================================================================
    // WARNINGS SINK (Task 0.2): ImportedProvider carries a warnings vec.
    // =====================================================================

    #[test]
    fn imported_provider_exposes_warnings_field_defaulting_empty() {
        // Direct struct construction (not via import) so this stays green across
        // every later component (component 4's gate will make import-path warnings
        // non-empty for precision-claiming contracts).
        let p = ImportedProvider {
            name: "p".into(),
            backend: fuel_ir::probe::BackendId::Cpu,
            kernel_source: "ks",
            primitives: Vec::new(),
            fused: Vec::new(),
            warnings: Vec::new(),
        };
        assert!(p.warnings.is_empty());
    }
}

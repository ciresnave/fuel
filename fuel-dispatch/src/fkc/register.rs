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
//! ## Cost decision for this slice ([consumer-ahead])
//!
//! [consumer-ahead]: declared cost priors are retained on `Resolved*.cost`
//! but not yet wired into a `CostFn`; the Judge bootstraps all imported
//! costs for now. A follow-up slice adds the cost trampoline (plan §2.3
//! strategy A).
//!
//! Concretely: [`ImportedProvider::register_into`] registers **every**
//! imported primitive with the existing [`unknown_cost`] sentinel `CostFn`
//! and **every** imported fused op with the fused-cost equivalent
//! ([`fused_unknown_cost`]) — regardless of whether the contract's
//! [`CompiledCostExpr`](crate::fkc::cost_expr::CompiledCostExpr) is
//! `Unknown` or a parsed expression. The parsed AST stays on the
//! `Resolved*` record; only the live `CostFn` wiring is deferred. This is
//! faithful to the corpus's `judge_measured` cost convention and gets the
//! importer end-to-end without an fn-pointer trampoline.

use std::collections::HashSet;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use fuel_core_types::backend::BackendCapabilities;
use fuel_core_types::probe::BackendId;
use fuel_core_types::Shape;
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
fn intern_dtypes(dtypes: &[fuel_core_types::DType]) -> &'static [fuel_core_types::DType] {
    use fuel_core_types::DType;
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
}

impl ImportedProvider {
    /// Split a flat `Vec<Resolved>` into the primitive / fused buckets.
    fn from_resolved(
        name: String,
        backend: BackendId,
        kernel_source: &'static str,
        resolved: Vec<Resolved>,
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
            // [consumer-ahead]: declared cost priors are retained on
            // Resolved*.cost but not yet wired into a CostFn; the Judge
            // bootstraps all imported costs for now. A follow-up slice
            // adds the cost trampoline (plan §2.3 strategy A).
            let kernel_source: &'static str = intern(&p.kernel_source);
            table.register_full_with_source(
                p.op,
                &p.dtypes,
                p.backend,
                p.kernel,
                p.caps,
                p.precision,
                unknown_cost,
                kernel_source,
            );
        }

        // --- Fused ops → FusedKernelRegistry (§2.2) ---
        for f in &self.fused {
            // [consumer-ahead]: declared cost priors are retained on
            // Resolved*.cost but not yet wired into a CostFn; the Judge
            // bootstraps all imported costs for now. A follow-up slice
            // adds the cost trampoline (plan §2.3 strategy A).
            let dtypes: &'static [fuel_core_types::DType] = intern_dtypes(&f.dtypes);
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
                },
            );
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
    // still apply; only their dispatch-resolution checks are skipped.
    crate::fkc::validate::validate_file(&file)?;
    // `lower_file` then EXCLUDES describe-only sections from the registered set
    // (§3.10): they never become a Resolved* record and are never registered.
    let resolved = lower_file(&file, link)?;
    let provider = &file.front_matter.provider;
    let backend = lower_backend_str(&provider.backend)?;
    let kernel_source = intern(&provider.kernel_source);
    Ok(ImportedProvider::from_resolved(
        provider.name.clone(),
        backend,
        kernel_source,
        resolved,
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
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::DType;
    use fuel_graph::registry::FusedOps;
    use smallvec::SmallVec;
    use std::sync::{Arc, RwLock};

    const ELEMENTWISE_BINARY: &str =
        include_str!("../../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md");

    // -- two distinct dummy kernels (distinct fn items ⇒ distinct ptrs) --

    fn dummy_a(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[fuel_core_types::Layout],
        _p: &crate::kernel::OpParams,
    ) -> fuel_core_types::Result<()> {
        Ok(())
    }
    fn dummy_b(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[fuel_core_types::Layout],
        _p: &crate::kernel::OpParams,
    ) -> fuel_core_types::Result<()> {
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
        let shape = fuel_core_types::Shape::from_dims(&[3]);
        let layouts = [
            fuel_core_types::Layout::contiguous(shape.clone()),
            fuel_core_types::Layout::contiguous(shape),
        ];
        resolved(&inputs, &mut outputs, &layouts, &crate::kernel::OpParams::None)
            .expect("the FKC-resolved add_f32 kernel executes on real storage");
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
        };

        let provider = ImportedProvider {
            name: "fuel-fused-registry".to_string(),
            backend: BackendId::Cpu,
            kernel_source: intern("portable-cpu"),
            primitives: Vec::new(),
            fused: vec![f],
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
}

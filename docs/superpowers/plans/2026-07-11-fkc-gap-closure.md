# FKC Gap Closure — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four highest-severity FKC design-vs-implementation gaps (shape-constraint solving, cost-expression compilation, return-contract validation, empirical precision verification) so a kernel provider's contract claims are actually cross-checked against ground truth and verified before becoming load-bearing.

**Architecture:** Four components in `fuel-dispatch/src/fkc/`, plus small touches to `fuel-memory`. (1) a new `shape_constraint.rs` parser+solver that turns §3.5's ratified vocabulary into concrete probe shapes; (2) a new `cost_compile.rs` that stores the already-parsed cost AST on the binding entry and evaluates it inline at the two ranking cost sites; (3) a new `return_check.rs` that cross-checks a fused contract's declared shape/dtype rules against the real `fuel_graph` registry function at probe shapes; (4) a new `verify/` submodule with a git-checked-in verification ledger + an import-time downgrade gate + a live-hardware harness. Components 3 and 4 both consume component 1's solver; a shared `warn.rs` (`ImportWarning`) and `ImportedProvider.warnings` field carry soft diagnostics out of the importer.

**Tech Stack:** Rust (edition 2024), `serde`/`serde_yml`/`serde_json`, `thiserror`, the existing `cost_expr.rs` recursive-descent parser (reused), `fuel_graph::registry`, `fuel_memory` storage, `bytemuck`.

**Source blueprints:** This plan was synthesized from five adversarially-verified research blueprints. The full per-component blueprints + verdicts live in the session scratchpad (`bp_*.json` / `verdict__*.json`). The problem statement is `docs/session-prompts/fkc-design-vs-implementation-gap-audit.md` (worktree `capturedrun-executor`); the approved design is `docs/superpowers/specs/2026-07-11-fkc-gap-closure-design.md`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Build discipline:** NEVER run workspace-wide cargo. ALWAYS `-p fuel-dispatch` (or `-p fuel-memory` for component 3 Task 7). ONE cargo invocation at a time; background long builds. Live-GPU tests are `#[ignore]`'d and run one backend feature at a time.
- **Never panic on production paths.** Every new function returns `Result` or an `Option`; soft-degradation paths emit an `ImportWarning`, never `panic!`/`unwrap` on importer input.
- **TDD is mandatory.** Write the failing test, run it, watch it go red (the exact red reason is stated per task), implement the minimum to green, run it green, commit.
- **`ImportWarning` has exactly ONE home:** `fuel-dispatch/src/fkc/warn.rs`, re-exported as `crate::fkc::ImportWarning`. No other module defines its own copy. Structure: `#[derive(Debug, Clone, PartialEq)] pub struct ImportWarning { pub section: String, pub message: String }`.
- **Warnings sink:** `ImportedProvider` carries `pub warnings: Vec<ImportWarning>`. `import_bundle_str` owns a `Vec<ImportWarning>`, threads `&mut` through `lower_file`/`lower_kernel`/`lower_fused`, appends any gate warnings itself, and moves it into `from_resolved`. Producers (solver, cross-check, precision gate) push into this vec.
- **Two intentional deviations from the design spec text** (both proven necessary by reading real code — do not "correct" them back):
  1. The §5 return-contract cross-check runs inside `lower_fused` (lower.rs:1039), NOT inside `register_into` — `ResolvedFused` carries neither the `FusedOpParams` nor the accept/return blocks the check needs; `FkcKernel` (with both) is only in scope in `lower_fused`. Import still fails before registration because `lower_file` runs inside `import_bundle_str` before `register_into`.
  2. The V-FKC-9 precision gate runs as a separate `gate_precision` pass in `import_bundle_str` over the flat `Vec<Resolved>`, NOT as a param added to `lower_precision`. `lower_precision` runs per-section before dtype fan-out and before the revision hash is computed; it has neither the per-variant dtypes nor the revision the gate keys on. `lower_precision` stays the pure declare-mapper.
- **`FkcError` is `#[non_exhaustive]`** (error.rs:20) and no in-crate code exhaustively matches it (only `FkcError::yaml` constructs `Yaml`), so adding variants is non-breaking crate-wide.
- **The cost `CostFn` fn-pointer signature** is `fn(&[Shape], &[DType], &OpParams, &BackendCapabilities) -> CostEstimate` (kernel.rs:809) — a plain fn pointer that receives no op/backend/kernel_source and cannot close over data. The spec's "generic trampoline" is therefore infeasible; the compiled cost AST is stored ON the entry (`Option<&'static CompiledCostExpr>`) and evaluated inline at the two ranking sites.

**Task-group order (dependency-driven):** Group 0 (foundation) → Group 1 (shape solver) → Group 2 (cost compiler; independent, sequenced here for a linear plan) → Group 3 (return check; needs 0+1) → Group 4 (verifier+ledger; needs 0+1).

---

## Task Group 0 — Shared foundation (`ImportWarning`, warnings plumbing, doc fix)

**Files:**
- Create: `fuel-dispatch/src/fkc/warn.rs`
- Modify: `fuel-dispatch/src/fkc/mod.rs` (doc-comment lines 8-10; `mod warn;` after `mod validate;` @54; `pub use warn::ImportWarning;` after `pub use error::FkcError;` @72)
- Modify: `fuel-dispatch/src/fkc/register.rs` (`ImportedProvider` struct @140-152; `from_resolved` @156-177; `import_bundle_str` @289-318; `import_glob` merge @398-399; the test struct-literal @1458-1464)
- Modify: `fuel-dispatch/src/fkc/lower.rs` (`lower_file` @1113; `lower_kernel` @894; `lower_fused` @1039 — add `warnings: &mut Vec<ImportWarning>` param, threaded)

**Interfaces:**
- Produces: `crate::fkc::ImportWarning { section: String, message: String }`; `ImportedProvider.warnings: Vec<ImportWarning>`; `from_resolved(name, backend, kernel_source, resolved, warnings)`; `lower_file(file, link, warnings)`, `lower_kernel(kernel, defaults, link, warnings)`, `lower_fused(kernel, id, defaults, link, warnings)` all gaining a trailing `warnings: &mut Vec<ImportWarning>`.

### Task 0.1 — Create `warn.rs` with the `ImportWarning` type

- [ ] **Step 1: Write the failing test** — in a new `fuel-dispatch/src/fkc/warn.rs`, at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_warning_constructs_and_carries_section_and_message() {
        let w = ImportWarning { section: "add_f32".into(), message: "downgraded bit_stable".into() };
        assert_eq!(w.section, "add_f32");
        assert!(w.message.contains("downgraded"));
        assert_eq!(w.clone(), w); // Clone + PartialEq derive present
    }
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::warn::tests::import_warning_constructs`
  Expected: FAIL — `fuel-dispatch/src/fkc/warn.rs` and `mod warn;` do not exist (compile error).

- [ ] **Step 3: Write the minimal implementation** — the top of `warn.rs`:

```rust
//! Soft, non-fatal FKC import diagnostics.
//!
//! FKC's importer surfaces *failures* as typed [`crate::fkc::FkcError`] values.
//! This module carries the complementary *warnings* — soft-degradation notices
//! (§3.5 "importer warns, does not reject", a precision claim downgraded for
//! lack of a verification-ledger entry, a shape_constraint that resolved to
//! seed shapes) that must NOT fail the import but MUST be visible to a caller.
//! Collected on [`crate::fkc::register::ImportedProvider::warnings`].

/// A single non-fatal FKC import diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportWarning {
    /// The kernel section (`## name`) the warning is about.
    pub section: String,
    /// Human-readable description of the soft-degradation that occurred.
    pub message: String,
}
```

- [ ] **Step 4: Wire the module + re-export** — in `fuel-dispatch/src/fkc/mod.rs`: add `mod warn;` immediately after `mod validate;` (line 54), and `pub use warn::ImportWarning;` immediately after `pub use error::FkcError;` (line 72). Also rewrite the stale doc-comment at mod.rs lines 8-10 (which claims the module "is gated behind the default-off `fkc` cargo feature") to state: `//! This module is unconditional production infrastructure (`pub mod fkc` in lib.rs, no feature gate; the `fkc` feature was removed in bd757464).`

- [ ] **Step 5: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::warn::tests::import_warning_constructs`
  Expected: PASS.

- [ ] **Step 6: Verify the doc-fix landed** — `git grep -c "gated behind the default-off" fuel-dispatch/src/fkc/mod.rs`
  Expected: `0` (no matches).

- [ ] **Step 7: Commit**

```bash
git add fuel-dispatch/src/fkc/warn.rs fuel-dispatch/src/fkc/mod.rs
git commit -m "feat(fkc): add ImportWarning type + fix stale mod gating doc"
```

### Task 0.2 — Add `ImportedProvider.warnings` + thread warnings through `from_resolved`, `import_bundle_str`, `import_glob`

**Interfaces:**
- Consumes: `crate::fkc::ImportWarning` (Task 0.1).
- Produces: `ImportedProvider.warnings: Vec<ImportWarning>`; `from_resolved(..., warnings: Vec<ImportWarning>)`.

- [ ] **Step 1: Write the failing test** — in the existing `mod tests` in `register.rs` (after the existing tests; do NOT create a second `mod tests`):

```rust
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
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::register::tests::imported_provider_exposes_warnings_field`
  Expected: FAIL — `ImportedProvider` has no `warnings` field (E0063 missing field / unknown field).

- [ ] **Step 3: Add the field + thread the param** —
  (a) `ImportedProvider` struct (register.rs:140-152): add `/// Soft import diagnostics (§3.5 warns, precision downgrades, etc.).\n    pub warnings: Vec<crate::fkc::ImportWarning>,` after the `fused` field.
  (b) `from_resolved` (register.rs:156-177): add a trailing param `warnings: Vec<crate::fkc::ImportWarning>,` and set `warnings,` in the returned struct literal (register.rs:170-176).
  (c) `import_bundle_str` (register.rs:308-317): change the body to own a warnings vec and pass it in:

```rust
    let mut warnings: Vec<crate::fkc::ImportWarning> = Vec::new();
    let resolved = lower_file(&file, link, &mut warnings)?;
    let provider = &file.front_matter.provider;
    let backend = lower_backend_str(&provider.backend)?;
    let kernel_source = intern(&provider.kernel_source);
    Ok(ImportedProvider::from_resolved(
        provider.name.clone(),
        backend,
        kernel_source,
        resolved,
        warnings,
    ))
```

  (d) `import_glob` merge arm (register.rs:398-399): add `acc.warnings.extend(provider.warnings);` alongside the existing `acc.primitives.extend(...); acc.fused.extend(...)`.
  (e) The existing test-module struct literal at register.rs:1458-1464 (the fused-only `register_into` test): add `warnings: Vec::new(),` so it still compiles.

- [ ] **Step 4: Thread `&mut Vec<ImportWarning>` through the lower chain** — this is required for (c) to compile:
  - `lower_file` (lower.rs:1113): signature becomes `pub fn lower_file(file: &FkcFile, link: &dyn LinkRegistry, warnings: &mut Vec<crate::fkc::ImportWarning>) -> Result<Vec<Resolved>, FkcError>`. Pass `warnings` into each `lower_kernel` call.
  - `lower_kernel` (lower.rs:894): add trailing `warnings: &mut Vec<crate::fkc::ImportWarning>`. Pass it into the `lower_fused` call in its fused branch; the `lower_primitive` branch ignores it (Phase 1 has no primitive-side producer).
  - `lower_fused` (lower.rs:1039): add trailing `warnings: &mut Vec<crate::fkc::ImportWarning>`. It is unused for now (`let _ = &warnings;` to silence the unused warning, or leave it — component 3 uses it).
  - Update the existing `lower_file` test caller at lower.rs:1293 (`let resolved = lower_file(&file, &StubLink).expect(...)`) to `let resolved = lower_file(&file, &StubLink, &mut Vec::new()).expect(...)`.

- [ ] **Step 5: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::register::tests::imported_provider_exposes_warnings_field`
  Expected: PASS.

- [ ] **Step 6: Confirm no test regressed** — `cargo test -p fuel-dispatch fkc::`
  Expected: all pre-existing fkc tests still PASS (the thread is empty; behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add fuel-dispatch/src/fkc/register.rs fuel-dispatch/src/fkc/lower.rs
git commit -m "feat(fkc): thread ImportWarning sink through the lower chain onto ImportedProvider"
```

---

## Task Group 1 — Shape-constraint parser + solver (`shape_constraint.rs`)

Builds `fuel-dispatch/src/fkc/shape_constraint.rs`, structured like `cost_expr.rs`. The `<expr>` grammar inside `dim[i]=<expr>`/`divisible(...)`/`capacity_ge(...)` REUSES `cost_expr::parse_expr` → `CostNode` (identical grammar, already parses `role.dim[i]`, negatives, calls, arithmetic). The evaluator is written fresh (i64, `Option`-returning, lazy-seeds shared free symbols).

**Corpus reality (verified) that overrides the spec's clean model:** `shape_constraint:` strings are `;`-joined lists mixing ratified vocabulary with FREE TEXT (`"same_as=out; read-modify-written in place (this operand IS the output)"`, `"byte length >= 4 (one u32)"`), use NEGATIVE axis indices (`dim[-1]=k`), open-ended rank ranges (`rank: "2.."`), param-symbol RHS, and genuine dependency CYCLES (`a: same_rank=b` ↔ `b: same_rank=a`). Therefore: the AST is `ShapeConstraint { atoms: Vec<ShapeAtom>, freetext: Vec<String> }`; hard-reject (`UnparseableShapeConstraint`) is narrowed to a keyword-COMMITTED segment with a malformed argument (`rank=banana`, unclosed `divisible(`, `dim[0]=` empty rhs); every other non-vocabulary segment degrades to free text + warning; cycles degrade to seed-fallback + warning (never `Err`).

**Files:**
- Create: `fuel-dispatch/src/fkc/shape_constraint.rs`
- Modify: `fuel-dispatch/src/fkc/error.rs` (add `UnparseableShapeConstraint` after `BundleSlotRankExceeded` @366)
- Modify: `fuel-dispatch/src/fkc/mod.rs` (`mod shape_constraint;` after `mod schema;`; re-exports)
- Modify (optional): `fuel-dispatch/src/fkc/lower.rs` (make `expand_dtype_class` @407 `pub(crate)`)

**Interfaces:**
- Consumes: `cost_expr::{parse_expr, CostNode, BinOp}`; `schema::TensorDesc`; `lower::lower_dtype` (pub(crate)); `crate::fkc::ImportWarning` (Group 0); `fuel_ir::{Shape, DType}`.
- Produces: `pub type ProbeCombo = Vec<(String, Shape, DType)>`; `pub enum RankSpec`; `pub enum AxisIndex { FromStart(usize), FromEnd(usize) }`; `pub enum ShapeAtom`; `pub struct ShapeConstraint { atoms, freetext }`; `pub fn parse_shape_constraint(raw, section, operand) -> Result<ShapeConstraint, FkcError>`; `pub fn parse_rank_spec(&str) -> Option<RankSpec>`; `pub fn solve_probe_shapes(inputs: &[TensorDesc], section: &str, warnings: &mut Vec<ImportWarning>) -> Result<Vec<ProbeCombo>, FkcError>`; `FkcError::UnparseableShapeConstraint`.

### Task 1.1 — AST types + `parse_shape_constraint` (`;`-split, free-text fallback, narrow hard-reject)

- [ ] **Step 1: Write the failing test** — in `shape_constraint.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::cost_expr::CostNode;

    #[test]
    fn parse_covers_vocab_freetext_and_rejects_malformed_vocab() {
        assert_eq!(parse_shape_constraint("same_as=k", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::SameAs("k".into())]);
        assert_eq!(parse_shape_constraint("same_rank=k", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::SameRank("k".into())]);
        assert_eq!(parse_shape_constraint("broadcast_to=x", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::BroadcastTo("x".into())]);
        assert_eq!(parse_shape_constraint("last_dim_eq=x", "s", "v").unwrap().atoms,
                   vec![ShapeAtom::LastDimEq("x".into())]);
        assert_eq!(parse_shape_constraint("rank=4", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Exact(4))]);
        assert_eq!(parse_shape_constraint("rank=2..=4", "s", "x").unwrap().atoms,
                   vec![ShapeAtom::Rank(RankSpec::Range { lo: 2, hi: Some(4) })]);
        // negative axis + bare-symbol RHS (linear-quant.fkc.md:108)
        let a = parse_shape_constraint("dim[-1]=k; same_rank=b", "linear", "a").unwrap();
        assert_eq!(a.atoms.len(), 2);
        match &a.atoms[0] {
            ShapeAtom::DimEq { axis, expr } => {
                assert_eq!(*axis, AxisIndex::FromEnd(1));
                assert_eq!(*expr, CostNode::Sym("k".into()));
            }
            other => panic!("got {other:?}"),
        }
        assert_eq!(a.atoms[1], ShapeAtom::SameRank("b".into()));
        assert!(matches!(parse_shape_constraint("divisible(q.dim[2], k.dim[2])", "f", "k")
            .unwrap().atoms[0], ShapeAtom::Divisible { .. }));
        assert!(matches!(parse_shape_constraint("capacity_ge(dim[0], seqlen)", "f", "kv")
            .unwrap().atoms[0], ShapeAtom::CapacityGe { .. }));
        // free text: valid-vocab head + prose tail (shape-ops.fkc.md:639) — NOT rejected
        let mixed = parse_shape_constraint(
            "same_as=out; read-modify-written in place (this operand IS the output)",
            "shape-ops", "dst").unwrap();
        assert_eq!(mixed.atoms, vec![ShapeAtom::SameAs("out".into())]);
        assert_eq!(mixed.freetext.len(), 1);
        // pure free text (shape-ops.fkc.md:721)
        let ft = parse_shape_constraint("byte length >= 4 (one u32)", "shape-ops", "seed").unwrap();
        assert!(ft.atoms.is_empty());
        assert_eq!(ft.freetext.len(), 1);
        // symbolic index + `==` (shape-ops.fkc.md:98) ⇒ free text, not reject
        let sym_i = parse_shape_constraint("dim[i] == in_shape[i]", "shape-ops", "out").unwrap();
        assert!(sym_i.atoms.is_empty());
        assert_eq!(sym_i.freetext.len(), 1);
        // HARD reject: keyword-committed segment with malformed argument
        assert!(matches!(parse_shape_constraint("rank=banana", "s", "x").unwrap_err(),
                         FkcError::UnparseableShapeConstraint { .. }));
        assert!(parse_shape_constraint("divisible(x.dim[0]", "s", "x").is_err()); // unclosed (
        assert!(parse_shape_constraint("dim[0]=", "s", "x").is_err());           // committed, empty rhs
    }
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::parse_covers_vocab`
  Expected: FAIL — `shape_constraint.rs` / the types / `FkcError::UnparseableShapeConstraint` do not exist.

- [ ] **Step 3: Add the error variant** — in `error.rs`, after `BundleSlotRankExceeded` (line 366), inside the enum:

```rust
    /// A `shape_constraint:` segment committed to §3.5 vocabulary but its
    /// argument is malformed (`rank=banana`, an unclosed `divisible(`, an empty
    /// `dim[0]=`). Non-vocabulary segments degrade to free text (a warning), not
    /// this error — this fires only on a real authoring mistake in the grammar.
    #[error(
        "FKC §3.5: kernel `{section}` operand `{operand}` shape_constraint segment `{raw}` \
         uses vocabulary but is malformed"
    )]
    UnparseableShapeConstraint { section: String, operand: String, raw: String },
```

- [ ] **Step 4: Write the parser** — the top of `shape_constraint.rs`. **Corrections applied vs the blueprint sketch:** the `divisible(`/`capacity_ge(` branches COMMIT on `starts_with` and then require the closing `)` (else `Err`), so the unclosed-paren case is a hard error (fixing the sketch/test inconsistency the verifier caught); `split_two_args` is bracket-depth-aware.

```rust
//! §3.5 shape/rank constraint vocabulary — parser + probe-shape solver.
//!
//! Structured like `cost_expr.rs`. The `<expr>` grammar inside `dim[i]=<expr>`,
//! `divisible(...)`, `capacity_ge(...)` reuses `cost_expr::parse_expr`.
use crate::fkc::cost_expr::{parse_expr as parse_cost_expr, CostNode};
use crate::fkc::error::FkcError;
use crate::fkc::ImportWarning;

pub type ProbeCombo = Vec<(String, fuel_ir::Shape, fuel_ir::DType)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankSpec { Exact(usize), Any, Range { lo: usize, hi: Option<usize> } }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisIndex { FromStart(usize), FromEnd(usize) } // dim[2]=FromStart(2); dim[-1]=FromEnd(1)

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeAtom {
    SameAs(String), SameRank(String), Rank(RankSpec), BroadcastTo(String), LastDimEq(String),
    DimEq { axis: AxisIndex, expr: CostNode },
    Divisible { lhs: CostNode, rhs: CostNode },
    CapacityGe { axis: AxisIndex, sym: String },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShapeConstraint { pub atoms: Vec<ShapeAtom>, pub freetext: Vec<String> }

fn parse_axis(s: &str) -> Option<AxisIndex> {
    let s = s.trim();
    if let Some(n) = s.strip_prefix('-') { n.trim().parse::<usize>().ok().map(AxisIndex::FromEnd) }
    else { s.parse::<usize>().ok().map(AxisIndex::FromStart) }
}

/// `4` | `any` | `2..=4` | `2..` -> RankSpec; None on anything else.
pub fn parse_rank_spec(s: &str) -> Option<RankSpec> {
    let s = s.trim();
    if s == "any" { return Some(RankSpec::Any); }
    if let Ok(n) = s.parse::<usize>() { return Some(RankSpec::Exact(n)); }
    if let Some((lo, hi)) = s.split_once("..=") {
        return Some(RankSpec::Range { lo: lo.trim().parse().ok()?, hi: Some(hi.trim().parse().ok()?) });
    }
    if let Some(lo) = s.strip_suffix("..") {
        return Some(RankSpec::Range { lo: lo.trim().parse().ok()?, hi: None });
    }
    None
}

/// Split `a, b` on the FIRST top-level comma, tracking `(` and `[` depth so
/// `capacity_ge(dim[0], seqlen)` / `divisible(q.dim[2], k.dim[2])` split correctly.
fn split_two_args(inner: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => return Some((&inner[..i], &inner[i + 1..])),
            _ => {}
        }
    }
    None
}

pub fn parse_shape_constraint(raw: &str, section: &str, operand: &str)
    -> Result<ShapeConstraint, FkcError>
{
    let mut out = ShapeConstraint::default();
    for seg_raw in raw.split(';') {
        let seg = seg_raw.trim();
        if seg.is_empty() { continue; }
        let unparse = || FkcError::UnparseableShapeConstraint {
            section: section.into(), operand: operand.into(), raw: seg.to_string() };
        if let Some(r) = seg.strip_prefix("same_as=")    { out.atoms.push(ShapeAtom::SameAs(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("same_rank=")  { out.atoms.push(ShapeAtom::SameRank(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("broadcast_to="){ out.atoms.push(ShapeAtom::BroadcastTo(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("last_dim_eq=") { out.atoms.push(ShapeAtom::LastDimEq(r.trim().into())); continue; }
        if let Some(r) = seg.strip_prefix("rank=") {                 // COMMITTED keyword
            out.atoms.push(ShapeAtom::Rank(parse_rank_spec(r).ok_or_else(unparse)?)); continue;
        }
        if seg.starts_with("divisible(") {                          // COMMITTED: require close paren
            let inner = seg.strip_prefix("divisible(").and_then(|s| s.strip_suffix(')')).ok_or_else(unparse)?;
            let (a, b) = split_two_args(inner).ok_or_else(unparse)?;
            let lhs = parse_cost_expr(a.trim()).map_err(|_| unparse())?;
            let rhs = parse_cost_expr(b.trim()).map_err(|_| unparse())?;
            out.atoms.push(ShapeAtom::Divisible { lhs, rhs }); continue;
        }
        if seg.starts_with("capacity_ge(") {                        // COMMITTED: require close paren
            let inner = seg.strip_prefix("capacity_ge(").and_then(|s| s.strip_suffix(')')).ok_or_else(unparse)?;
            let (a, b) = split_two_args(inner).ok_or_else(unparse)?;
            let axis = a.trim().strip_prefix("dim[").and_then(|s| s.strip_suffix(']'))
                .and_then(parse_axis).ok_or_else(unparse)?;
            out.atoms.push(ShapeAtom::CapacityGe { axis, sym: b.trim().to_string() }); continue;
        }
        if seg.starts_with("dim[") {
            if let Some(close) = seg.find(']') {
                let idx = &seg["dim[".len()..close];
                let after = seg[close + 1..].trim_start();
                match (parse_axis(idx), after.strip_prefix('=')) {
                    // committed `dim[<int>]=<expr>` with a SINGLE '=' (not `==`)
                    (Some(axis), Some(rhs)) if !rhs.starts_with('=') => {
                        let rhs = rhs.trim();
                        if rhs.is_empty() { return Err(unparse()); }
                        out.atoms.push(ShapeAtom::DimEq { axis, expr: parse_cost_expr(rhs).map_err(|_| unparse())? });
                        continue;
                    }
                    // symbolic index (`dim[i]`) or `==` ⇒ pseudocode ⇒ free text
                    _ => { out.freetext.push(seg.to_string()); continue; }
                }
            }
        }
        out.freetext.push(seg.to_string()); // no recognized keyword ⇒ §3.5 notes-style free text
    }
    Ok(out)
}
```

- [ ] **Step 5: Wire the module** — in `mod.rs`, add `mod shape_constraint;` after `mod schema;`, and `pub use shape_constraint::{parse_shape_constraint, parse_rank_spec, solve_probe_shapes, AxisIndex, ProbeCombo, RankSpec, ShapeAtom, ShapeConstraint};` in the pub-use block. (Note: `solve_probe_shapes` lands in Task 1.2 but declaring the re-export now is fine — it will fail to compile until 1.2; if executing strictly one task at a time, add only the parse-related names here and extend the re-export in 1.2.)

- [ ] **Step 6: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::parse_covers_vocab`
  Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_constraint.rs fuel-dispatch/src/fkc/error.rs fuel-dispatch/src/fkc/mod.rs
git commit -m "feat(fkc): §3.5 shape_constraint parser (atoms+freetext, narrow hard-reject)"
```

### Task 1.2 — Rank resolution + unconstrained probe seeding (3 canonical profiles)

**Interfaces:**
- Produces: `solve_probe_shapes(&[TensorDesc], &str, &mut Vec<ImportWarning>) -> Result<Vec<ProbeCombo>, FkcError>` (seed-only in this task); `SeedProfile`/`PROFILES`; `first_probe_dtype`.

- [ ] **Step 1: Write the failing test** — add to `shape_constraint.rs` `mod tests`:

```rust
    fn desc(name: &str, dtypes: &[&str], rank: Option<u64>) -> crate::fkc::schema::TensorDesc {
        crate::fkc::schema::TensorDesc {
            name: Some(name.into()), optional: false,
            dtypes: dtypes.iter().map(|s| s.to_string()).collect(),
            dtype_class: None, layout: None,
            rank: rank.map(|r| serde_yml::Value::Number(r.into())),
            shape_constraint: None, fdx: None, device: None, substrate: None,
        }
    }

    #[test]
    fn seed_unconstrained_operands_over_three_profiles() {
        use fuel_ir::Shape;
        let inputs = vec![desc("lhs", &["F32"], Some(2)), desc("rhs", &["F32"], Some(2))];
        let mut w = Vec::new();
        let combos = solve_probe_shapes(&inputs, "s", &mut w).unwrap();
        assert_eq!(combos.len(), 3);
        assert_eq!(combos[0][0].1, Shape::from_dims(&[2, 2]));  // profile A all-2
        assert_eq!(combos[1][0].1, Shape::from_dims(&[4, 3]));  // profile B all-4, last->3
        assert_eq!(combos[2][0].1, Shape::from_dims(&[8, 8]));  // profile C all-8
        assert!(w.is_empty());
    }

    #[test]
    fn rank_any_defaults_to_4_and_open_range_uses_lo() {
        let any = desc("a", &["F32"], None); // no rank ⇒ Any ⇒ 4
        assert_eq!(solve_probe_shapes(&[any], "s", &mut Vec::new()).unwrap()[0][0].1.rank(), 4);
        let mut open = desc("b", &["F32"], None);
        open.rank = Some(serde_yml::Value::String("2..".into()));
        assert_eq!(solve_probe_shapes(&[open], "s", &mut Vec::new()).unwrap()[0][0].1.rank(), 2);
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::seed_unconstrained`
  Expected: FAIL — `solve_probe_shapes` does not exist.

- [ ] **Step 3: Implement seed-only `solve_probe_shapes` + helpers**:

```rust
use std::collections::HashMap;
use fuel_ir::{DType, Shape};
use crate::fkc::schema::TensorDesc;

#[derive(Clone, Copy)]
struct SeedProfile { base: i64, odd_last: bool }
const PROFILES: [SeedProfile; 3] = [
    SeedProfile { base: 2, odd_last: false }, // A all-2
    SeedProfile { base: 4, odd_last: true },  // B all-4, last axis ->3
    SeedProfile { base: 8, odd_last: false }, // C all-8
];

fn resolve_rank_spec_field(v: Option<&serde_yml::Value>) -> Option<RankSpec> {
    match v {
        Some(serde_yml::Value::Number(n)) => n.as_u64().map(|u| RankSpec::Exact(u as usize)),
        Some(serde_yml::Value::String(s)) => parse_rank_spec(s),
        _ => None,
    }
}
fn rank_for_probe(spec: Option<RankSpec>) -> usize {
    match spec {
        Some(RankSpec::Exact(n)) => n,
        Some(RankSpec::Range { lo, .. }) => lo,
        Some(RankSpec::Any) | None => 4, // `any`/absent default rank 4
    }
}
fn seed_axis(profile: SeedProfile, axis: usize, rank: usize) -> i64 {
    if profile.odd_last && rank > 0 && axis == rank - 1 { 3 } else { profile.base }
}
/// First declared dtype, else first `dtype_class` expansion, else F32.
fn first_probe_dtype(d: &TensorDesc) -> DType {
    if let Some(tok) = d.dtypes.first() {
        if let Ok(dt) = crate::fkc::lower::lower_dtype(tok, "", "") { return dt; }
    }
    match d.dtype_class.as_deref() {
        Some("float") => DType::BF16, Some("int") => DType::I8, Some("uint") => DType::U8,
        _ => DType::F32,
    }
}

pub fn solve_probe_shapes(inputs: &[TensorDesc], section: &str, warnings: &mut Vec<ImportWarning>)
    -> Result<Vec<ProbeCombo>, FkcError>
{
    // Parse each operand's constraint now so a malformed-vocabulary segment is a
    // hard error before we build any probe (Tasks 1.3/1.4 consume `parsed`).
    let mut parsed = Vec::with_capacity(inputs.len());
    for d in inputs {
        let operand = d.name.as_deref().unwrap_or("<unnamed>");
        let sc = match &d.shape_constraint {
            Some(raw) => parse_shape_constraint(raw, section, operand)?,
            None => ShapeConstraint::default(),
        };
        parsed.push(sc);
    }
    let mut combos = Vec::with_capacity(PROFILES.len());
    for profile in PROFILES {
        let mut combo: ProbeCombo = Vec::with_capacity(inputs.len());
        for d in inputs {
            let role = d.name.clone().unwrap_or_default();
            let rank = rank_for_probe(resolve_rank_spec_field(d.rank.as_ref()));
            let dims: Vec<usize> = (0..rank).map(|a| seed_axis(profile, a, rank) as usize).collect();
            combo.push((role, Shape::from_dims(&dims), first_probe_dtype(d)));
        }
        combos.push(combo);
    }
    let _ = (&parsed, warnings); // consumed by Tasks 1.3/1.4/1.5
    Ok(combos)
}
```

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::seed_unconstrained fkc::shape_constraint::tests::rank_any`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_constraint.rs
git commit -m "feat(fkc): shape solver seed profiles + rank resolution"
```

### Task 1.3 — Structural constraint application (same_as/dim[i]=expr/divisible via a shape+symbol evaluator)

**Interfaces:**
- Produces: `eval_dim_expr`, `as_dim_ref`, the per-profile `apply_atom` pass integrated into `solve_probe_shapes`; a per-profile `Solve` state with a lazy shared-symbol env.

- [ ] **Step 1: Write the failing test** — add to `mod tests`:

```rust
    #[test]
    fn solve_same_as_copies_dims_and_divisible_rounds_up() {
        use fuel_ir::Shape;
        let mut k = desc("k", &["F32"], Some(3));
        k.shape_constraint = Some("divisible(dim[0], 8)".into());
        let mut v = desc("v", &["F32"], Some(3));
        v.shape_constraint = Some("same_as=k".into());
        let combos = solve_probe_shapes(&[k, v], "s", &mut Vec::new()).unwrap();
        let a = &combos[0]; // profile A base 2 ⇒ ceil(2/8)*8 = 8
        let ks = &a.iter().find(|(r, _, _)| r == "k").unwrap().1;
        let vs = &a.iter().find(|(r, _, _)| r == "v").unwrap().1;
        assert_eq!(ks, &Shape::from_dims(&[8, 2, 2]));
        assert_eq!(vs, ks);
    }

    #[test]
    fn dim_eq_bare_symbol_is_shared_across_operands() {
        let mut a = desc("a", &["F32"], Some(2));
        a.shape_constraint = Some("dim[-1]=k".into());
        let mut b = desc("b", &["F32"], Some(2));
        b.shape_constraint = Some("dim[-2]=k".into());
        let combos = solve_probe_shapes(&[a, b], "linear", &mut Vec::new()).unwrap();
        let a0 = &combos[0];
        let ak = a0.iter().find(|(r, _, _)| r == "a").unwrap().1.dims()[1];
        let bk = a0.iter().find(|(r, _, _)| r == "b").unwrap().1.dims()[0];
        assert_eq!(ak, bk, "both K axes bind the same seeded symbol `k`");
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::solve_same_as fkc::shape_constraint::tests::dim_eq_bare`
  Expected: FAIL — `solve_probe_shapes` seeds only; atoms are ignored, so the rounded/shared assertions fail.

- [ ] **Step 3: Implement the evaluator + apply pass** (add to `shape_constraint.rs`; integrate a `Solve` state into `solve_probe_shapes`'s per-profile loop, replacing the seed-only body):

```rust
struct Solve { dims: HashMap<String, Vec<i64>>, sym: HashMap<String, i64>, base: i64 }

/// Recognize `dim[i]` (self) or `role.dim[i]` as a dim reference.
fn as_dim_ref(node: &CostNode) -> Option<(Option<String>, AxisIndex)> {
    if let CostNode::Index { base, index } = node {
        let axis = match &**index {
            CostNode::Lit(v) => AxisIndex::FromStart(*v as usize),
            CostNode::Neg(inner) => if let CostNode::Lit(v) = &**inner { AxisIndex::FromEnd(*v as usize) } else { return None },
            _ => return None,
        };
        if let CostNode::Sym(s) = &**base {
            return Some(if let Some(r) = s.strip_suffix(".dim") { (Some(r.to_string()), axis) }
                        else if s == "dim" { (None, axis) } else { return None });
        }
    }
    None
}

fn axis_to_index(axis: AxisIndex, rank: usize) -> Option<usize> {
    match axis { AxisIndex::FromStart(i) => Some(i), AxisIndex::FromEnd(n) => rank.checked_sub(n) }
}

/// Evaluate a CostNode to a concrete i64. None ⇒ genuinely unresolvable.
fn eval_dim_expr(node: &CostNode, s: &mut Solve, ranks: &HashMap<String, usize>, self_role: &str) -> Option<i64> {
    use crate::fkc::cost_expr::BinOp::*;
    match node {
        CostNode::Lit(v) => Some(*v as i64),
        CostNode::Neg(i) => eval_dim_expr(i, s, ranks, self_role).map(|x| -x),
        CostNode::Bin { op, lhs, rhs } => {
            let (l, r) = (eval_dim_expr(lhs, s, ranks, self_role)?, eval_dim_expr(rhs, s, ranks, self_role)?);
            Some(match op { Add => l + r, Sub => l - r, Mul => l * r, Div if r != 0 => l / r, Rem if r != 0 => l % r, _ => return None })
        }
        CostNode::Index { .. } => {
            let (role, axis) = as_dim_ref(node)?;
            let rrole = role.as_deref().unwrap_or(self_role);
            let idx = axis_to_index(axis, *ranks.get(rrole)?)?;
            s.dims.get(rrole)?.get(idx).copied()
        }
        CostNode::Sym(name) => Some(*s.sym.entry(name.clone()).or_insert(s.base)), // lazy-seed shared symbol
        CostNode::Call { .. } => None,
    }
}

fn warn(section: &str, message: String) -> ImportWarning { ImportWarning { section: section.into(), message } }

fn set_axis(s: &mut Solve, role: &str, axis: AxisIndex, rank: usize, v: i64) {
    if let Some(idx) = axis_to_index(axis, rank) {
        if let Some(d) = s.dims.get_mut(role) { if idx < d.len() { d[idx] = v.max(1); } }
    }
}

fn apply_atom(atom: &ShapeAtom, self_role: &str, s: &mut Solve, ranks: &HashMap<String, usize>,
              w: &mut Vec<ImportWarning>, section: &str) {
    let self_rank = *ranks.get(self_role).unwrap_or(&0);
    match atom {
        ShapeAtom::Rank(_) | ShapeAtom::SameRank(_) | ShapeAtom::CapacityGe { .. } => {} // rank-phase / trivial
        ShapeAtom::SameAs(src) | ShapeAtom::BroadcastTo(src) => match s.dims.get(src).cloned() {
            Some(src_dims) => {
                let n = self_rank.min(src_dims.len());
                if let Some(d) = s.dims.get_mut(self_role) { for a in 0..n { d[a] = src_dims[a]; } }
            }
            None => w.push(warn(section, format!("operand `{self_role}` references unknown role `{src}`; using seed shape"))),
        },
        ShapeAtom::LastDimEq(src) => {
            if let (Some(sr), Some(src_rank)) = (s.dims.get(src).and_then(|d| d.last().copied()), ranks.get(src)) {
                let _ = src_rank;
                set_axis(s, self_role, AxisIndex::FromEnd(1), self_rank, sr);
            } else {
                w.push(warn(section, format!("operand `{self_role}` last_dim_eq references unknown role `{src}`; using seed")));
            }
        }
        ShapeAtom::DimEq { axis, expr } => match eval_dim_expr(expr, s, ranks, self_role) {
            Some(v) => set_axis(s, self_role, *axis, self_rank, v),
            None => w.push(warn(section, format!("operand `{self_role}` dim rule unresolved; using seed"))),
        },
        ShapeAtom::Divisible { lhs, rhs } => {
            if let (Some((role, axis)), Some(v)) = (as_dim_ref(lhs), eval_dim_expr(rhs, s, ranks, self_role)) {
                if v > 0 {
                    let target = role.as_deref().unwrap_or(self_role).to_string();
                    let trank = *ranks.get(&target).unwrap_or(&0);
                    if let Some(idx) = axis_to_index(axis, trank) {
                        if let Some(cur) = s.dims.get(&target).and_then(|d| d.get(idx).copied()) {
                            set_axis(s, &target, axis, trank, ((cur + v - 1) / v) * v);
                        }
                    }
                }
            } else if let CostNode::Sym(k) = lhs {
                if let Some(v) = eval_dim_expr(rhs, s, ranks, self_role) {
                    if v > 0 { let e = s.sym.entry(k.clone()).or_insert(s.base); *e = ((*e + v - 1) / v) * v; }
                }
            }
        }
    }
}
```

  Then rewrite `solve_probe_shapes`'s per-profile loop to build a `Solve`, seed all dims, apply every operand's atoms (source order for now — Task 1.4 adds topo ordering), and materialize `Shape::from_dims`. Keep the `ranks: HashMap<role, usize>` computed once. (Full integration: seed `s.dims`/`ranks` for every operand, then `for d in inputs { for atom in &parsed[i].atoms { apply_atom(atom, role, &mut s, &ranks, warnings, section) } }`, then read `s.dims[role]` back into `Shape::from_dims`.)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::solve_same_as fkc::shape_constraint::tests::dim_eq_bare`
  Expected: PASS. Also re-run the whole module: `cargo test -p fuel-dispatch fkc::shape_constraint` — Tasks 1.1/1.2 still PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_constraint.rs
git commit -m "feat(fkc): shape solver structural constraints (same_as/dim=/divisible + shared symbols)"
```

### Task 1.4 — Dependency ordering + cycle detection (warning + seed-fallback, never Err)

- [ ] **Step 1: Write the failing test** — add to `mod tests`:

```rust
    #[test]
    fn dependency_ordering_resolves_and_cycles_fall_back_with_warning() {
        use fuel_ir::Shape;
        // ordering: v depends on k even though v is listed first
        let mut k = desc("k", &["F32"], Some(2));
        k.shape_constraint = Some("divisible(dim[0], 8)".into());
        let mut v = desc("v", &["F32"], Some(2));
        v.shape_constraint = Some("same_as=k".into());
        let combos = solve_probe_shapes(&[v, k], "s", &mut Vec::new()).unwrap();
        assert_eq!(combos[0].iter().find(|(r, _, _)| r == "v").unwrap().1, Shape::from_dims(&[8, 2]));
        // cycle: no panic, no Err, Ok + a `cycle` warning + seed shapes
        let mut ca = desc("a", &["F32"], Some(2));
        ca.shape_constraint = Some("same_as=b".into());
        let mut cb = desc("b", &["F32"], Some(2));
        cb.shape_constraint = Some("same_as=a".into());
        let mut w = Vec::new();
        let combos = solve_probe_shapes(&[ca, cb], "cyc", &mut w).unwrap();
        assert_eq!(combos.len(), 3);
        assert_eq!(combos[0][0].1, Shape::from_dims(&[2, 2]));
        assert!(w.iter().any(|x| x.message.to_lowercase().contains("cycle")), "warns: {w:?}");
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::dependency_ordering`
  Expected: FAIL — atoms apply in source order (v reads k's un-rounded seed) and a mutual cycle produces no `cycle` warning.

- [ ] **Step 3: Implement `dep_sources` + `topo_order` (Kahn, cycle→warn+seed) and apply atoms in dependency order** in `solve_probe_shapes`:

```rust
use std::collections::HashSet;

/// Input roles whose DIMS must resolve before `atoms` can apply.
fn dep_sources(atoms: &[ShapeAtom], input_roles: &HashSet<String>) -> Vec<String> {
    fn collect(n: &CostNode, roles: &HashSet<String>, out: &mut Vec<String>) {
        if let Some((Some(r), _)) = as_dim_ref(n) { if roles.contains(&r) { out.push(r); } }
        match n {
            CostNode::Bin { lhs, rhs, .. } => { collect(lhs, roles, out); collect(rhs, roles, out); }
            CostNode::Neg(i) => collect(i, roles, out),
            CostNode::Index { base, index } => { collect(base, roles, out); collect(index, roles, out); }
            CostNode::Call { args, .. } => for a in args { collect(a, roles, out); },
            _ => {}
        }
    }
    let mut deps = Vec::new();
    for a in atoms {
        match a {
            ShapeAtom::SameAs(r) | ShapeAtom::BroadcastTo(r) | ShapeAtom::LastDimEq(r) if input_roles.contains(r) => deps.push(r.clone()),
            ShapeAtom::DimEq { expr, .. } => collect(expr, input_roles, &mut deps),
            ShapeAtom::Divisible { lhs, rhs } => { collect(lhs, input_roles, &mut deps); collect(rhs, input_roles, &mut deps); }
            _ => {}
        }
    }
    deps
}

/// Kahn topological order over input roles; residual (cyclic) roles get ONE
/// `cycle` warning and are appended in source order so their atoms still run
/// best-effort. Never errors, always terminates.
fn topo_order(order_in: &[String], edges: &HashMap<String, Vec<String>>, section: &str, w: &mut Vec<ImportWarning>) -> Vec<String> {
    let set: HashSet<&String> = order_in.iter().collect();
    let mut indeg: HashMap<&String, usize> = order_in.iter().map(|r| (r, 0usize)).collect();
    for (r, deps) in edges { if set.contains(r) { for d in deps { if set.contains(d) { *indeg.get_mut(r).unwrap() += 1; } } } }
    let mut queue: Vec<&String> = order_in.iter().filter(|r| indeg[r] == 0).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < queue.len() {
        let cur = queue[i]; i += 1; out.push(cur.clone());
        for r in order_in { if let Some(deps) = edges.get(r) {
            if deps.contains(cur) { let e = indeg.get_mut(r).unwrap(); *e = e.saturating_sub(1); if *e == 0 && !out.contains(r) && !queue.contains(&r) { queue.push(r); } }
        } }
    }
    if out.len() < order_in.len() {
        let residual: Vec<&String> = order_in.iter().filter(|r| !out.contains(r)).collect();
        w.push(warn(section, format!("shape_constraint dependency cycle among {residual:?}; using seed shapes")));
        for r in order_in { if !out.contains(r) { out.push(r.clone()); } }
    }
    out
}
```

  Integrate: build `edges: role -> dep_sources(atoms)`, compute `topo_order` once (profile-independent), and apply atoms in that order per profile.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::dependency_ordering` then the whole `fkc::shape_constraint` module.
  Expected: PASS (all shape_constraint tests).

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_constraint.rs
git commit -m "feat(fkc): shape solver dependency ordering + cycle-safe seed fallback"
```

### Task 1.5 — dtype pick + free-text/unknown-role/malformed-rank warnings + `pub(crate)` expand_dtype_class

- [ ] **Step 1: Write the failing test** — add to `mod tests`:

```rust
    #[test]
    fn dtype_pick_and_soft_fallback_warnings() {
        use fuel_ir::{DType, Shape};
        assert_eq!(solve_probe_shapes(&[desc("a", &["BF16", "F16", "F32"], Some(1))], "s", &mut Vec::new()).unwrap()[0][0].2, DType::BF16);
        let mut c = desc("a", &[], Some(1));
        c.dtype_class = Some("float".into());
        assert_eq!(solve_probe_shapes(&[c], "s", &mut Vec::new()).unwrap()[0][0].2, DType::BF16);
        // same_as=out (output role, absent from inputs) ⇒ seed shape + warning naming `out`
        let mut r = desc("residual", &["F32"], Some(2));
        r.shape_constraint = Some("same_as=out".into());
        let mut w = Vec::new();
        let combos = solve_probe_shapes(&[r], "norm-softmax", &mut w).unwrap();
        assert_eq!(combos[0][0].1, Shape::from_dims(&[2, 2]));
        assert!(w.iter().any(|x| x.message.contains("out")));
        // pure free-text constraint ⇒ warning, still Ok
        let mut f = desc("seed", &["U8"], Some(1));
        f.shape_constraint = Some("byte length >= 4 (one u32)".into());
        let mut w2 = Vec::new();
        solve_probe_shapes(&[f], "shape-ops", &mut w2).unwrap();
        assert!(!w2.is_empty());
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch fkc::shape_constraint::tests::dtype_pick`
  Expected: FAIL — free-text warnings are not emitted from `solve_probe_shapes` (the `same_as=out` unknown-role warning IS emitted by Task 1.3's `apply_atom`, but the pure-free-text warning is not).

- [ ] **Step 3: Emit free-text + malformed-rank warnings** — in `solve_probe_shapes`, after parsing each operand's `ShapeConstraint`, for every entry in `sc.freetext` push `warn(section, format!("operand `{operand}` shape_constraint free text: {seg}"))`. When `resolve_rank_spec_field` returns `None` for a PRESENT-but-malformed `rank:` field, push a warning and default to rank 4. (Optional: make `lower::expand_dtype_class` @407 `pub(crate)` and route `first_probe_dtype`'s class branch through its first element for a single source of truth.)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch fkc::shape_constraint`
  Expected: PASS (entire module).

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/shape_constraint.rs fuel-dispatch/src/fkc/lower.rs
git commit -m "feat(fkc): shape solver dtype pick + soft-fallback warnings"
```

---

## Task Group 2 — Cost-expression compiler (`cost_compile.rs`) + V-FKC-9 cost-half

Wires the ALREADY-PARSED contract cost AST (`ResolvedPrimitive.cost: CompiledCostExpr`, lower.rs:92; `ResolvedFused.cost`, lower.rs:133) into the live ranking cost path. **The spec's "generic trampoline" is infeasible** (see Global Constraints): the compiled AST is stored ON the entry as `Option<&'static CompiledCostExpr>` (Copy-preserving `&'static`) and evaluated inline at the two eval sites (`ranker/cost.rs:247` primitive; `fused_cost.rs:71` fused). A contract-pinned `cost.cost_fn:` still wins (stays on `entry.cost`; we set `cost_expr=None` when pinned). This group is independent of Groups 0/1 (it needs neither `ImportWarning` nor the solver), sequenced here for a linear plan.

**Files:**
- Create: `fuel-dispatch/src/fkc/cost_compile.rs`
- Modify: `fuel-dispatch/src/kernel.rs` (`BindingEntry` @838-892 add field; `register_full_with_source_generic` @1100-1131 add param; `register_full_with_source` @1059-1087 pass `None`)
- Modify: `fuel-dispatch/src/fkc/register.rs` (primitive loop @209-244; fused loop @247-265)
- Modify: `fuel-dispatch/src/fused.rs` (`BackendImpl` @48-73 add field; `register_fused_kernel!` macro literal @390; test literals `make_impl` @1608, `weak` @1914)
- Modify: `fuel-dispatch/src/fused_cost.rs` (`fused_layer1_cost` @71-84; `sentinel_impl` literal @345)
- Modify: `fuel-dispatch/src/ranker/cost.rs` (eval site @247)
- Modify: `fuel-dispatch/src/fkc/validate.rs` (Rule 8a placeholder @1050-1071)
- Modify: `fuel-dispatch/src/fkc/mod.rs` (`pub(crate) mod cost_compile;`)

**Interfaces:**
- Produces: `BindingEntry.cost_expr: Option<&'static CompiledCostExpr>`; `BackendImpl.cost_expr: Option<&'static CompiledCostExpr>`; `cost_compile::{intern_cost_expr, stamp_primitive_cost_expr, stamp_fused_cost_expr, fused_cost_estimate, CostClassKind, classify_cost}`.

### Task 2.1 — Add the Copy `cost_expr` field to `BindingEntry` + thread through the low-level registrar (default `None`)

- [ ] **Step 1: Write the failing test** — in `kernel.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn cost_expr_field_defaults_none_for_hand_written_registration() {
    use fuel_ir::probe::BackendId;
    let mut table = KernelBindingTable::new();
    let dts = [DType::F32, DType::F32, DType::F32];
    table.register(OpKind::AddElementwise, &dts, BackendId::Cpu, ok_kernel);
    let alts = table.lookup_alternatives(OpKind::AddElementwise, &dts, BackendId::Cpu);
    assert_eq!(alts.len(), 1);
    assert!(alts[0].cost_expr.is_none(), "hand-written registrations carry no declared cost AST");
}
```

(Use the `ok_kernel`/helper fn already present in the kernel.rs tests module; mirror an existing `register` test.)

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib kernel::tests::cost_expr_field_defaults_none`
  Expected: FAIL — `BindingEntry` has no `cost_expr` field (compile error).

- [ ] **Step 3: Add the field + param** —
  - `BindingEntry` (kernel.rs:891, after `kernel_revision_hash: u64`): add `pub cost_expr: Option<&'static crate::fkc::CompiledCostExpr>,`. (`&'static T` is `Copy`/`Clone`/`Debug` when `T: Debug`; `CompiledCostExpr` derives `Debug, Clone, PartialEq` — the `#[derive(Clone, Copy, Debug)]` on `BindingEntry` still holds.)
  - `register_full_with_source_generic` (kernel.rs:1100): add a trailing param `cost_expr: Option<&'static crate::fkc::CompiledCostExpr>` (after `kernel_revision_hash: u64`) and set `cost_expr,` in the `BindingEntry { .. }` literal (~1114-1122).
  - `register_full_with_source` (kernel.rs:1059, the hand-written path): pass `None` as the new trailing arg alongside its existing `false, KernelRevisionHash::UNTRACKED.0`.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib kernel::tests::cost_expr_field_defaults_none`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/kernel.rs
git commit -m "feat(fkc): add BindingEntry.cost_expr handle (default None)"
```

### Task 2.2 — `cost_compile.rs` interner + stamp fns; declared primitive cost AST reaches the binding

- [ ] **Step 1: Write the failing test** — in `register.rs` `mod tests` (uses `SameLink` which is defined there at register.rs:581):

```rust
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
```

> Note (from Component 4): this contract declares `bit_stable_on_same_hardware: true` + `audited: true`. Group 4's precision gate will later downgrade that to UNAUDITED against the empty ledger — but this test asserts only `cost_expr` and registration, never precision, so it stays green after Group 4.

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::register::tests::imported_contract_declared_cost_reaches_binding`
  Expected: FAIL — `register_into` stamps only `p.cost_fn.unwrap_or(unknown_cost)` (register.rs:221) and DROPS `p.cost`; `entry.cost_expr` is `None` → `.expect(..)` panics.

- [ ] **Step 3: Create `cost_compile.rs` + wire the primitive path**:

```rust
//! Compile a contract's parsed cost AST into the live ranking cost path (§2.3).
//! The parser/evaluator (`cost_expr.rs`) is complete; this only adds wiring.
use std::sync::{Mutex, OnceLock};
use crate::fkc::cost_expr::CompiledCostExpr;
use crate::fkc::lower::{ResolvedPrimitive, ResolvedFused};

/// Bounded, dedup'd process-lifetime leak (mirrors `register::intern`). Unknown → None.
pub fn intern_cost_expr(expr: &CompiledCostExpr) -> Option<&'static CompiledCostExpr> {
    if matches!(expr, CompiledCostExpr::Unknown) { return None; }
    static POOL: OnceLock<Mutex<Vec<&'static CompiledCostExpr>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = pool.lock().expect("cost_expr interner poisoned");
    if let Some(&e) = g.iter().find(|&&x| x == expr) { return Some(e); }
    let leaked: &'static CompiledCostExpr = Box::leak(Box::new(expr.clone()));
    g.push(leaked);
    Some(leaked)
}

/// A contract-pinned cost_fn wins outright (stays on entry.cost); the declared
/// AST does not compete with it, so return None when a fn is pinned.
pub fn stamp_primitive_cost_expr(p: &ResolvedPrimitive) -> Option<&'static CompiledCostExpr> {
    if p.cost_fn.is_some() { return None; }
    intern_cost_expr(&p.cost)
}
pub fn stamp_fused_cost_expr(f: &ResolvedFused) -> Option<&'static CompiledCostExpr> {
    intern_cost_expr(&f.cost)
}
```

  Add `pub(crate) mod cost_compile;` to `mod.rs`. In `register.rs` primitive loop (after `let cost_fn = p.cost_fn.unwrap_or(unknown_cost);` @221): `let cost_expr = crate::fkc::cost_compile::stamp_primitive_cost_expr(p);` and pass `cost_expr` as the trailing arg to `register_full_with_source_generic`.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::register::tests::imported_contract_declared_cost_reaches_binding`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/cost_compile.rs fuel-dispatch/src/fkc/mod.rs fuel-dispatch/src/fkc/register.rs
git commit -m "feat(fkc): stamp declared primitive cost AST onto the binding entry"
```

### Task 2.3 — Evaluate the stamped AST at the primitive ranking cost site

- [ ] **Step 1: Write the failing test** — in `ranker/cost.rs` `mod tests` (mirror the helper/`Candidate`/`AlternativeSet`/`BackendCapabilities` construction from the existing `compute_static_costs_populates_via_binding_lookup` test at ranker/cost.rs:485-523):

```rust
#[test]
fn compute_static_costs_prefers_declared_cost_expr() {
    use fuel_ir::probe::BackendId;
    let expr = crate::fkc::cost_compile::intern_cost_expr(
        &crate::fkc::cost_expr::compile_field(Some("2 * n")).unwrap()).expect("expr interns");
    let mut table = KernelBindingTable::new();
    let dts = [DType::F32, DType::F32, DType::F32];
    let prec = PrecisionGuarantee { bit_stable_on_same_hardware: true, max_ulp: Some(0), max_relative: None, max_absolute: None, notes: "t" };
    // entry.cost is unknown_cost (all-zero) — proves the AST, not the fn, priced the cell.
    table.register_full_with_source_generic(OpKind::AddElementwise, &dts, BackendId::Cpu, noop_a,
        KernelCaps::empty(), prec, unknown_cost, "", false, 0, Some(expr));
    let mut set = AlternativeSet::new();
    set.push(Candidate::new(noop_a as KernelRef, BackendId::Cpu, OpParams::None, ""));
    let caps = test_caps();
    let lookup = |_b: BackendId| -> Option<&BackendCapabilities> { Some(&caps) };
    compute_static_costs(&mut set, OpKind::AddElementwise, &dts, &[Shape::from_dims(&[3])], &table, &lookup, None);
    assert_eq!(set.static_cost(0).flops, 6, "2 * n with n = elem_count([3]) = 3");
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib ranker::cost::tests::compute_static_costs_prefers_declared_cost_expr`
  Expected: FAIL — line 247 calls `(entry.cost)(..)` = `unknown_cost` → flops 0; the stamped AST is ignored, so `assert_eq!(.., 6)` fails.

- [ ] **Step 3: Prefer the compiled AST at the eval site** — replace `ranker/cost.rs:247` `let cost = (entry.cost)(shapes, dtypes, &op_params, caps);` with:

```rust
let cost = match entry.cost_expr {
    Some(expr) => crate::fkc::cost_estimate(expr, op_kind, shapes, dtypes, &op_params)
        .unwrap_or_else(|_| (entry.cost)(shapes, dtypes, &op_params, caps)),
    None => (entry.cost)(shapes, dtypes, &op_params, caps),
};
```

  (An eval error — undefined symbol — degrades to the fn pointer; never panics. `op_kind` is the fn param at ranker/cost.rs:225.)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib ranker::cost::tests::compute_static_costs_prefers_declared_cost_expr`
  Expected: PASS. Re-run `cargo test -p fuel-dispatch --lib ranker::` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/ranker/cost.rs
git commit -m "feat(fkc): price primitives by the declared cost AST at the ranking site"
```

### Task 2.4 — Fused wiring: `BackendImpl.cost_expr` + `fused_layer1_cost` preference + fused symbol binder

- [ ] **Step 1: Write the failing test** — in `fused_cost.rs` `mod tests`:

```rust
#[test]
fn fused_declared_cost_reaches_layer1_not_sentinel() {
    use fuel_ir::backend::BackendCapabilities;
    use fuel_graph::registry::{FusedOps, FusedOpParams};
    let expr = crate::fkc::cost_compile::intern_cost_expr(&crate::fkc::cost_expr::compile_field(Some("n")).unwrap()).unwrap();
    let impl_ = BackendImpl { cost_expr: Some(expr), ..sentinel_impl() };
    let caps = BackendCapabilities::default();
    let est = fused_layer1_cost(&impl_, FusedOps::SOFTMAX_LAST_DIM, &[Shape::from_dims(&[8])], &[DType::F32], &FusedOpParams::SoftmaxLastDim, &caps);
    assert_eq!(est.flops, 8, "declared fused flops = n = 8; not the sentinel/decompose fallback");
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fused_cost::tests::fused_declared_cost_reaches_layer1`
  Expected: FAIL — `BackendImpl` has no `cost_expr` field (compile error).

- [ ] **Step 3: Implement** —
  - `fused.rs`: add `pub cost_expr: Option<&'static crate::fkc::CompiledCostExpr>,` to `BackendImpl` (after `revision` @72). Add `cost_expr: None,` to the `register_fused_kernel!` macro's `BackendImpl { .. }` literal (@390) and to the test literals `make_impl` (@1608) and `weak` (@1914).
  - `fused_cost.rs`: add `cost_expr: None,` to the `sentinel_impl()` literal (@345). In `fused_layer1_cost`, BEFORE the `is_fused_cost_sentinel` branch (@79), prepend:

```rust
    if let Some(expr) = impl_.cost_expr {
        if let Ok(est) = crate::fkc::cost_compile::fused_cost_estimate(expr, input_shapes, input_dtypes, params) {
            return est;
        }
    }
```

  - `cost_compile.rs`: add the fused binder:

```rust
use crate::fused::CostEstimate;
use crate::fkc::cost_expr::{eval, CostEvalError};
use fuel_ir::{Shape, DType};
use fuel_graph::registry::FusedOpParams;

/// Minimal fused-cost symbol binder: `n` (last input elem_count) + `dtype_bytes`.
/// A fused (m,n,k) formula would under-bind and eval-error → the caller falls
/// back to the compose-from-decompose estimate (already a non-zero cost).
pub fn fused_cost_estimate(expr: &CompiledCostExpr, input_shapes: &[Shape], input_dtypes: &[DType], _params: &FusedOpParams)
    -> Result<CostEstimate, CostEvalError> {
    let mut b = std::collections::HashMap::new();
    if let Some(s) = input_shapes.last() { b.insert("n".to_string(), s.elem_count() as f64); }
    if let Some(d) = input_dtypes.last() { b.insert("dtype_bytes".to_string(), d.size_in_bytes() as f64); }
    Ok(CostEstimate { flops: eval(expr, &b)?.max(0.0) as u64, ..Default::default() })
}
```

  - `register.rs` fused loop: add `cost_expr: crate::fkc::cost_compile::stamp_fused_cost_expr(f),` to the `BackendImpl { .. }` literal (@256-263).

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fused_cost::tests::fused_declared_cost_reaches_layer1`, then a scoped `cargo build -p fuel-dispatch` to catch the `register_fused_kernel!` macro ripple across every hand-written fused registration.
  Expected: PASS + clean build.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fused.rs fuel-dispatch/src/fused_cost.rs fuel-dispatch/src/fkc/cost_compile.rs fuel-dispatch/src/fkc/register.rs
git commit -m "feat(fkc): declared fused cost reaches layer-1 pricing (no more permanent sentinel)"
```

### Task 2.5 — V-FKC-9 cost-half: closed `CostClassKind` classifier replaces the `class.is_empty()` escape hatch

- [ ] **Step 1: Write the failing test** — in `validate.rs` `mod tests`. **Correction (from verifier):** `validate.rs::tests` has `StubLink` (validate.rs:1821), NOT `SameLink`; use `StubLink` + `use crate::fkc::register::import_bundle_str;`.

```rust
#[test]
fn placeholder_cost_class_field() {
    use crate::fkc::cost_compile::{classify_cost, CostClassKind};
    assert!(classify_cost("declared", "cheap_elementwise", false, false).is_none());
    assert!(matches!(classify_cost("declared", "free", false, false), Some(CostClassKind::Free)));
    assert!(matches!(classify_cost("declared", "gemm_like", true, false), Some(CostClassKind::DeclaredFormula)));
    assert!(matches!(classify_cost("declared", "gemm_like", false, true), Some(CostClassKind::VendorSpec)));
    assert!(matches!(classify_cost("judge_measured", "gemm_like", false, false), Some(CostClassKind::JudgeMeasured)));
    // integration: a declared+non-free class with NO flops/bytes/cost_fn is rejected on import
    use crate::fkc::register::import_bundle_str;
    let src = "---\nfkc_version: 1\nprovider:\n  name: ph-provider\n  backend: Cpu\n  kernel_source: \"ph-cpu\"\n---\n\n# ph\n\n## add_f32\n\nA.\n\n```fkc\nkernel: add_f32\nop_kind: AddElementwise\nblurb: \"a\"\nentry_point: \"x::add_f32\"\naccept:\n  inputs:\n    - name: lhs\n      dtypes: [F32]\n      layout: { contiguous: required, strided: rejected }\n    - name: rhs\n      dtypes: [F32]\n      layout: { contiguous: required, strided: rejected }\n  op_params: { variant: None }\nreturn:\n  outputs:\n    - name: out\n      dtype_rule: passthrough(lhs)\ncost:\n  provenance: declared\n  class: cheap_elementwise\nprecision:\n  bit_stable_on_same_hardware: true\n  audited: true\ndeterminism: same_hardware_bitwise\n```\n";
    let err = import_bundle_str(src, &StubLink).expect_err("declared + non-free class + no usable cost path must reject");
    assert!(matches!(err, FkcError::PlaceholderCost { .. }), "got {err:?}");
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::validate::tests::placeholder_cost_class_field`
  Expected: FAIL — `classify_cost` does not exist (compile error); and the current gate at validate.rs:1060-1063 uses `class.is_empty()`, so a non-empty class bypasses it.

- [ ] **Step 3: Implement `classify_cost` + rewrite the gate** —
  - `cost_compile.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostClassKind { Free, JudgeMeasured, DeclaredFormula, VendorSpec }

/// A cost block is load-bearing iff it maps to Some(kind). `class: free` is the
/// only no-expression license for a declared block.
pub fn classify_cost(provenance: &str, class: &str, has_any_expr: bool, has_cost_fn: bool) -> Option<CostClassKind> {
    if class == "free" { return Some(CostClassKind::Free); }
    match provenance {
        "judge_measured" => Some(CostClassKind::JudgeMeasured),
        "declared" if has_cost_fn => Some(CostClassKind::VendorSpec),
        "declared" if has_any_expr => Some(CostClassKind::DeclaredFormula),
        _ => None,
    }
}
```

  - `validate.rs`: replace the `if cost.provenance.as_deref() == Some("declared") && !has_any_expr && class != "free" && class.is_empty()` block (1060-1071) with:

```rust
    let has_cost_fn = cost.cost_fn.as_deref().is_some_and(|s| !s.trim().is_empty());
    if cost.provenance.as_deref() == Some("declared")
        && crate::fkc::cost_compile::classify_cost("declared", class, has_any_expr, has_cost_fn).is_none()
    {
        return Err(FkcError::PlaceholderCost {
            section: section.to_string(),
            reason: "provenance: declared with no usable cost path (no flops/bytes_moved expression, \
                     no pinned cost.cost_fn) and class is not `free`".to_string(),
        });
    }
```

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::validate::tests::placeholder_cost_class_field`
  Expected: PASS.

- [ ] **Step 5: Guard against corpus regression** — run the whole-corpus import test (the FKC corpus lint) to confirm no shipped contract relied on the old escape hatch: `cargo test -p fuel-dispatch --lib fkc::validate::tests::ci_lint_corpus_parse_lower_validate` (and any `import_glob`-over-`docs/kernel-contracts` test).
  Expected: PASS. (The verifier's corpus scan found ZERO declared blocks lacking flops AND bytes_moved AND cost_fn with a non-`free` class — cast.fkc.md: 110/110 carry flops+bytes — so nothing should break. If something does, that contract must add a flops hint, pin a cost_fn, or switch to `class: free`/`provenance: judge_measured` — an explicit, reviewed diff, not a silent skip.)

- [ ] **Step 6: Commit**

```bash
git add fuel-dispatch/src/fkc/cost_compile.rs fuel-dispatch/src/fkc/validate.rs
git commit -m "feat(fkc): V-FKC-9 cost-half — closed CostClassKind replaces class.is_empty() escape hatch"
```

---

## Task Group 3 — Return-contract validation (`return_check.rs`) — §5 / V-FKC-7

Cross-checks each `fused_op` contract's declared `return.outputs`/`return.bundle` rules against the REAL `FusedOpEntry::shape_rule`/`dtype_rule`/`output_views` fns from `fuel_graph::registry::default_registry()`, evaluated at Group 1's probe shapes. **Runs in `lower_fused`** (the only site holding both the parsed `FkcKernel` and the resolved `FusedOpId`). **Critical safety fact:** the real `shape_rule` fns for `qmatmul`/`conv2d` PANIC on a mismatched `FusedOpParams` (qmatmul.rs:63, conv2d.rs:86), so the cross-check synthesizes the exact variant from the contract's `op_params.variant` and only invokes the real fn when BOTH the declared rule is evaluable AND synth produced the matching variant.

**Depends on:** Group 0 (ImportWarning, warnings thread through lower_fused), Group 1 (`solve_probe_shapes`, `ProbeCombo`).

**Files:**
- Create: `fuel-dispatch/src/fkc/return_check.rs`
- Modify: `fuel-dispatch/src/fkc/error.rs` (add `ShapeRuleMismatch`, `BundleArityMismatch`)
- Modify: `fuel-dispatch/src/fkc/mod.rs` (`mod return_check;`)
- Modify: `fuel-dispatch/src/fkc/lower.rs` (call cross-check in `lower_fused`; add `ResolvedFused.bundle_slot_names: Vec<String>` @118-144 + set it)
- Modify: `fuel-dispatch/src/fkc/register.rs` (fused loop @247-265: call `record_bundle_slot_names`)
- Modify: `fuel-dispatch/src/fused.rs` (`FusedKernelRegistry` @269-272: side-table field + methods)
- Modify: `fuel-dispatch/src/fkc/validate.rs` (fix the false 963-966 comment)
- Modify: `fuel-memory/src/dlpack_view.rs` (FDX `NAME_TABLE` @432-459)

**Interfaces:**
- Consumes: `solve_probe_shapes`/`ProbeCombo` (Group 1); `ImportWarning` + lower_fused warnings param (Group 0); `default_registry`/`FusedOpEntry` (fuel-graph); `FusedOpParams` (fuel-graph); `lower_dtype`.
- Produces: `eval_dtype_rule`, `eval_shape_rule`, `synth_probe_params`, `cross_check_fused_section`, `bundle_slot_count`, `check_bundle_arity`, `check_slot_rank`, `bundle_slot_names`; `FkcError::{ShapeRuleMismatch, BundleArityMismatch}`; `ResolvedFused.bundle_slot_names`; `FusedKernelRegistry::{record_bundle_slot_names, bundle_slot_names}`; `fuel_memory::dlpack_view::fdx_slot_name`.

### Task 3.1 — Return-rule interpreter (`eval_dtype_rule`/`eval_shape_rule`) + the two error variants

- [ ] **Step 1: Write the failing test** — in `return_check.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Shape};

    #[test]
    fn interpreter_evaluates_supported_vocab_and_skips_the_rest() {
        let combo: Vec<(String, Shape, DType)> = vec![
            ("x".into(), Shape::from_dims(&[2, 3]), DType::F32),
            ("upstream".into(), Shape::from_dims(&[4, 5]), DType::F16),
        ];
        let c: ProbeComboRef = &combo;
        assert_eq!(eval_dtype_rule("fixed(F16)", c, "k").unwrap(), Some(DType::F16));
        assert_eq!(eval_dtype_rule("passthrough(x)", c, "k").unwrap(), Some(DType::F32));
        assert_eq!(eval_dtype_rule("dequant(w)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("same_as(upstream)", c, "k").unwrap(), Some(Shape::from_dims(&[4, 5])));
        assert_eq!(eval_shape_rule("from_params(q)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("matmul(a, b)", c, "k").unwrap(), None);
        assert_eq!(eval_shape_rule("same_as(does_not_exist)", c, "k").unwrap(), None);
    }
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::interpreter_evaluates`
  Expected: FAIL — `return_check.rs` / the fns / `ProbeComboRef` do not exist.

- [ ] **Step 3: Add the error variants + implement the interpreter** —
  - `error.rs` (after `BundleSlotRankExceeded`, near the §5 area):

```rust
    /// A `fused_op` contract's declared §5.1/§5.2 return rule disagrees with the
    /// real registered `FusedOpEntry` fn at a probe shape (V-FKC-7, Finding 5.1).
    /// `expected`/`actual` render either a shape or a dtype.
    #[error(
        "FKC §5 (V-FKC-7): kernel `{section}` output `{role}` declared return rule disagrees with \
         the registered fused fn (declared {expected}, real {actual})"
    )]
    ShapeRuleMismatch { section: String, role: String, expected: String, actual: String },

    /// A `return.bundle` slot count disagrees with the registered
    /// `output_views` arity (V-FKC-7, Finding 5.2).
    #[error("FKC §5.5 (V-FKC-7): kernel `{section}` declares {actual} bundle slots but output_views has {expected}")]
    BundleArityMismatch { section: String, expected: usize, actual: usize },
```

  - `return_check.rs`:

```rust
//! §5 return-contract validation: cross-check a fused contract's declared
//! shape/dtype rules against the real registered FusedOpEntry fns.
use fuel_ir::{DType, Shape};
use crate::fkc::error::FkcError;
use crate::fkc::lower::lower_dtype;

pub type ProbeComboRef<'a> = &'a [(String, Shape, DType)];

fn role<'a>(combo: ProbeComboRef<'a>, name: &str) -> Option<&'a (String, Shape, DType)> {
    combo.iter().find(|(r, _, _)| r == name)
}
fn inner<'a>(rule: &'a str, head: &str) -> Option<&'a str> {
    rule.trim().strip_prefix(head)?.strip_suffix(')').map(str::trim)
}

/// §5.1: `fixed(D)` and `passthrough(role)` are evaluable; every other token is
/// `Ok(None)` = not-evaluable (skip, never a false reject). `fixed(<bad dtype>)`
/// is a hard error (a real authoring bug).
pub fn eval_dtype_rule(rule: &str, combo: ProbeComboRef, section: &str) -> Result<Option<DType>, FkcError> {
    if let Some(tok) = inner(rule, "fixed(") { return Ok(Some(lower_dtype(tok, section, "return")?)); }
    if let Some(r) = inner(rule, "passthrough(") { return Ok(role(combo, r).map(|(_, _, d)| *d)); }
    Ok(None)
}
/// §5.2: only `same_as(role)` is evaluable purely from probe shapes.
pub fn eval_shape_rule(rule: &str, combo: ProbeComboRef, _section: &str) -> Result<Option<Shape>, FkcError> {
    if let Some(r) = inner(rule, "same_as(") { return Ok(role(combo, r).map(|(_, s, _)| s.clone())); }
    Ok(None)
}
```

  Add `mod return_check;` to `mod.rs`.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::interpreter_evaluates`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs fuel-dispatch/src/fkc/error.rs fuel-dispatch/src/fkc/mod.rs
git commit -m "feat(fkc): §5.1/§5.2 return-rule interpreter + ShapeRuleMismatch/BundleArityMismatch"
```

### Task 3.2 — `synth_probe_params`: contract variant → matching `FusedOpParams` (never-panic guard)

- [ ] **Step 1: Write the failing test** — in `return_check.rs` `mod tests`:

```rust
    #[test]
    fn synth_probe_params_builds_the_matching_variant_or_none() {
        use fuel_graph::registry::FusedOpParams;
        assert!(matches!(synth_probe_params(Some("SoftmaxLastDim")).unwrap(), Some(FusedOpParams::SoftmaxLastDim)));
        assert!(matches!(synth_probe_params(Some("RmsNormLastDim")).unwrap(), Some(FusedOpParams::RmsNormLastDim { .. })));
        assert!(synth_probe_params(Some("SsdChunkScan")).unwrap().is_none());
        assert!(synth_probe_params(None).unwrap().is_none());
        // never-panic invariant: QMatMul synth is EITHER QMatMul OR None, never a foreign variant.
        match synth_probe_params(Some("QMatMul")).unwrap() {
            None => {}
            Some(p) => assert!(matches!(p, FusedOpParams::QMatMul { .. })),
        }
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::synth_probe_params`
  Expected: FAIL — `synth_probe_params` is undefined.

- [ ] **Step 3: Implement `synth_probe_params`** (params-dependent ops → `None`, so their real fn is never called):

```rust
use fuel_graph::registry::FusedOpParams;

/// Synthesize the FusedOpParams variant NAMED by the contract's op_params.variant
/// (§3.7). The ONLY correctness requirement is that the returned variant matches
/// the FusedOpId's real fn so shape_rule never hits its wrong-params panic
/// (qmatmul.rs:63, conv2d.rs:86). Params-dependent ops → None (their declared
/// return rules are from_params-style = not-evaluable, so the real fn is skipped).
pub fn synth_probe_params(variant: Option<&str>) -> Result<Option<FusedOpParams>, FkcError> {
    const EPS: f64 = 1e-5;
    Ok(match variant {
        Some("SoftmaxLastDim") => Some(FusedOpParams::SoftmaxLastDim),
        Some("SoftmaxLastDimBackward") => Some(FusedOpParams::SoftmaxLastDimBackward),
        Some("RmsNormLastDim") => Some(FusedOpParams::RmsNormLastDim { eps: EPS }),
        Some("LayerNormLastDim") => Some(FusedOpParams::LayerNormLastDim { eps: EPS }),
        Some("RmsNormLastDimBackward") => Some(FusedOpParams::RmsNormLastDimBackward { eps: EPS }),
        Some("LayerNormLastDimBackward") => Some(FusedOpParams::LayerNormLastDimBackward { eps: EPS }),
        Some("ReduceMaxToBackward") => Some(FusedOpParams::ReduceMaxToBackward),
        Some("PowIBackward") => Some(FusedOpParams::PowIBackward { exp: 2 }),
        Some("Rope") => Some(FusedOpParams::Rope),
        Some("FusedLinear") => Some(FusedOpParams::FusedLinear),
        _ => None,
    })
}
```

  (Verify each variant name/shape against `fuel-graph/src/registry.rs:173-380` when transcribing — the exact variant set may differ slightly; the invariant is "matching-or-None, never foreign".)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::synth_probe_params`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs
git commit -m "feat(fkc): synth_probe_params — matching-or-None variant guard for real shape_rule"
```

### Task 3.3 — The cross-check wired into `lower_fused` (the born-red §5 test)

- [ ] **Step 1: Write the failing test** — in `register.rs` `mod tests` (FUSED_NORM_SOFTMAX const is at register.rs:790; `import_bundle_str` + `CpuLinkRegistry` in scope):

```rust
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
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::register::tests::fused_contract_shape_rule_disagreeing`
  Expected: FAIL — `lower_fused` never inspects `return_`/calls a cross-check, so the mutated bundle imports `Ok`; `expect_err` panics.

- [ ] **Step 3: Implement `cross_check_fused_section` + wire into `lower_fused`**. **Corrections applied:** solver errors are soft-caught (skip + warn), NOT `?`-propagated (protects the currently-green `cpu_link_registry_binds_norm_softmax_fused_ops_to_live_kernels` from a solver-hard-reject regression); the dtype_rule fallback only calls the real fn when synth produced `Some(params)` (mirrors the shape_rule guard).

```rust
use fuel_graph::registry::{FusedOpId, default_registry};
use crate::fkc::schema::FkcKernel;
use crate::fkc::shape_constraint::solve_probe_shapes;
use crate::fkc::ImportWarning;

/// Invariant: the real shape_rule/dtype_rule fns are invoked ONLY when synth
/// produced the matching variant AND the declared rule is evaluable — for every
/// current fused op that coincidence holds (evaluable rules belong to
/// params-independent fns), so qmatmul/conv2d's wrong-params panic is unreachable.
pub fn cross_check_fused_section(kernel: &FkcKernel, id: FusedOpId, warnings: &mut Vec<ImportWarning>) -> Result<(), FkcError> {
    let section = kernel.kernel.as_str();
    let Some(entry) = default_registry().entry(id) else { return Ok(()); };
    let Some(accept) = kernel.accept.as_ref() else { return Ok(()); };
    let Some(ret) = kernel.return_.as_ref() else { return Ok(()); };
    let variant = accept.op_params.as_ref().and_then(|s| s.variant.as_deref());
    // Soft-catch solver errors (e.g. a malformed-vocabulary constraint): skip the
    // cross-check + warn rather than fail the whole import.
    let combos = match solve_probe_shapes(&accept.inputs, section, warnings) {
        Ok(c) => c,
        Err(e) => { warnings.push(ImportWarning { section: section.into(), message: format!("return cross-check skipped: {e}") }); return Ok(()); }
    };
    let params = synth_probe_params(variant)?;
    for combo in &combos {
        let in_shapes: Vec<Shape> = combo.iter().map(|(_, s, _)| s.clone()).collect();
        let in_dtypes: Vec<DType> = combo.iter().map(|(_, _, d)| *d).collect();
        for out in &ret.outputs {
            let role_name = out.name.as_deref().unwrap_or("out");
            if let (Some(rule), Some(p)) = (out.dtype_rule.as_deref(), params.as_ref()) {
                if let Some(declared) = eval_dtype_rule(rule, combo, section)? {
                    let real = (entry.dtype_rule)(&in_dtypes, p);
                    if declared != real {
                        return Err(FkcError::ShapeRuleMismatch { section: section.into(), role: role_name.into(),
                            expected: format!("dtype {declared:?}"), actual: format!("dtype {real:?}") });
                    }
                }
            }
            if let (Some(rule), Some(p)) = (out.shape_rule.as_deref(), params.as_ref()) {
                if let Some(declared) = eval_shape_rule(rule, combo, section)? {
                    let real = (entry.shape_rule)(&in_shapes, p);
                    if declared != real {
                        return Err(FkcError::ShapeRuleMismatch { section: section.into(), role: role_name.into(),
                            expected: format!("shape {declared:?}"), actual: format!("shape {real:?}") });
                    }
                }
            }
        }
    }
    Ok(())
}
```

  In `lower_fused` (lower.rs:1039), after `id` is resolved and with `kernel` + `warnings` in scope: `crate::fkc::return_check::cross_check_fused_section(kernel, id, warnings)?;`. Also add `pub bundle_slot_names: Vec<String>,` to `ResolvedFused` (lower.rs:118-144) and set it to `Vec::new()` in the `out.push(ResolvedFused { .. })` literal for now (Task 3.6 changes this to the real `bundle_slot_names(&kernel.return_)` extraction once that helper exists — keeping each task independently compilable).

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::register::tests::fused_contract_shape_rule_disagreeing fkc::register::tests::unmutated_fused_corpus`
  Expected: PASS. Then `cargo test -p fuel-dispatch --lib fkc::register::tests::cpu_link_registry_binds_norm_softmax` — confirm no regression.

  > If the mutation string `dtype_rule: passthrough(x)` isn't present verbatim in norm-softmax.fkc.md, open the file and pick the actual first-section declared dtype rule + a provably-false replacement dtype; keep the `.replacen(.., .., 1)` shape.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs fuel-dispatch/src/fkc/lower.rs
git commit -m "feat(fkc): §5 cross-check declared return rules vs the real FusedOpEntry fn"
```

### Task 3.4 — Bundle-arity cross-check (Finding 5.2)

- [ ] **Step 1: Write the failing test** — in `return_check.rs` `mod tests`:

```rust
    #[test]
    fn bundle_slot_count_disagreeing_with_output_views_arity_is_rejected() {
        let two_slots: serde_yml::Value = serde_yml::from_str(
            "- { name: y, shape_rule: same_as(u) }\n- { name: last_state, shape_rule: from_params(state) }").unwrap();
        assert_eq!(bundle_slot_count(&two_slots), Some(2));
        let err = check_bundle_arity("selective_scan", 3, &two_slots)
            .expect_err("declared 2 vs 3 real output_views slots must be rejected");
        assert!(matches!(err, FkcError::BundleArityMismatch { expected: 3, actual: 2, .. }), "got {err:?}");
        assert!(check_bundle_arity("selective_scan", 2, &two_slots).is_ok());
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::bundle_slot_count_disagreeing`
  Expected: FAIL — `bundle_slot_count`/`check_bundle_arity` don't exist.

- [ ] **Step 3: Implement + integrate into the cross-check**:

```rust
pub fn bundle_slot_count(bundle: &serde_yml::Value) -> Option<usize> {
    match bundle { serde_yml::Value::Sequence(s) => Some(s.len()), _ => None }
}
pub fn check_bundle_arity(section: &str, output_views_arity: usize, bundle: &serde_yml::Value) -> Result<(), FkcError> {
    if let Some(declared) = bundle_slot_count(bundle) {
        if declared != output_views_arity {
            return Err(FkcError::BundleArityMismatch { section: section.into(), expected: output_views_arity, actual: declared });
        }
    }
    Ok(())
}
```

  Inside `cross_check_fused_section`, when `ret.bundle.is_some()` AND `entry.output_views.is_some()` AND `params.is_some()`: evaluate `output_views` at a probe combo, take its `.len()`, and call `check_bundle_arity`.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::bundle_slot_count_disagreeing`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs
git commit -m "feat(fkc): §5.5 bundle-arity cross-check vs output_views"
```

### Task 3.5 — Relocate the rank≤6 check for shape_rule-derived slots (Finding 5.3)

- [ ] **Step 1: Write the failing test** — in `return_check.rs` `mod tests`:

```rust
    #[test]
    fn shape_rule_derived_bundle_slot_over_rank6_is_rejected() {
        use fuel_ir::Shape;
        let rank7 = Shape::from_dims(&[2, 2, 2, 2, 2, 2, 2]);
        let err = check_slot_rank("s", "big_slot", &rank7).expect_err("rank 7 must be rejected");
        assert!(matches!(err, FkcError::BundleSlotRankExceeded { rank: 7, .. }), "got {err:?}");
        assert!(check_slot_rank("s", "ok_slot", &Shape::from_dims(&[1,1,1,1,1,1])).is_ok());
    }
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::shape_rule_derived_bundle_slot_over_rank6`
  Expected: FAIL — `check_slot_rank` doesn't exist.

- [ ] **Step 3: Implement + call from the cross-check; fix the false comment**:

```rust
pub fn check_slot_rank(section: &str, slot: &str, shape: &Shape) -> Result<(), FkcError> {
    if shape.rank() > 6 {
        return Err(FkcError::BundleSlotRankExceeded { section: section.into(), slot: slot.into(), rank: shape.rank() });
    }
    Ok(())
}
```

  Call it inside `cross_check_fused_section` for every bundle slot whose `shape_rule` evaluated to `Some(shape)`. Update the false `validate.rs:963-966` comment to point at this now-real register-time enforcement (leave the static-`shape:`-literal branch in `check_bundle_ranks`).

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::return_check::tests::shape_rule_derived_bundle_slot_over_rank6`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs fuel-dispatch/src/fkc/validate.rs
git commit -m "feat(fkc): rank<=6 check for shape_rule-derived bundle slots (fixes false comment)"
```

### Task 3.6 — Slot-name side-table on `FusedKernelRegistry` (Finding 5.4, FKC side)

- [ ] **Step 1: Write the failing test** — in `fused.rs` `mod tests`:

```rust
#[test]
fn bundle_slot_names_round_trip_through_the_fused_registry() {
    use fuel_ir::{DType, probe::BackendId};
    use fuel_graph::registry::FusedOps;
    let mut reg = FusedKernelRegistry::new();
    let dtypes = &[DType::F32, DType::F32][..];
    reg.record_bundle_slot_names(FusedOps::SELECTIVE_SCAN, BackendId::Cpu, dtypes,
        &["y".to_string(), "last_state".to_string()]);
    assert_eq!(reg.bundle_slot_names(FusedOps::SELECTIVE_SCAN, BackendId::Cpu, dtypes),
        Some(&["y".to_string(), "last_state".to_string()][..]));
    assert_eq!(reg.bundle_slot_names(FusedOps::SELECTIVE_SCAN, BackendId::Cpu, &[DType::F16][..]), None);
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fused::tests::bundle_slot_names_round_trip`
  Expected: FAIL — no such field/methods.

- [ ] **Step 3: Implement** —
  - `fused.rs`: add `bundle_slot_names: HashMap<(FusedOpId, BackendId, Vec<DType>), Vec<String>>` to `FusedKernelRegistry` (`#[derive(Default)]` still holds), plus:

```rust
impl FusedKernelRegistry {
    pub fn record_bundle_slot_names(&mut self, id: FusedOpId, backend: BackendId, dtypes: &[DType], names: &[String]) {
        self.bundle_slot_names.insert((id, backend, dtypes.to_vec()), names.to_vec());
    }
    pub fn bundle_slot_names(&self, id: FusedOpId, backend: BackendId, dtypes: &[DType]) -> Option<&[String]> {
        self.bundle_slot_names.get(&(id, backend, dtypes.to_vec())).map(|v| v.as_slice())
    }
}
```

  - `return_check.rs`: `pub fn bundle_slot_names(ret: &Option<crate::fkc::schema::ReturnBlock>) -> Vec<String>` extracting each bundle slot's `name` (returns `Vec::new()` when no bundle).
  - `lower.rs`: change the `ResolvedFused.bundle_slot_names` initializer in `lower_fused` (set to `Vec::new()` in Task 3.3) to `crate::fkc::return_check::bundle_slot_names(&kernel.return_)`.
  - `register.rs` fused loop (after `fused.register(...)`): `if !f.bundle_slot_names.is_empty() { fused.record_bundle_slot_names(f.id, f.backend, dtypes, &f.bundle_slot_names); }` (where `dtypes` is the interned `&f.dtypes`).

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fused::tests::bundle_slot_names_round_trip`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/return_check.rs fuel-dispatch/src/fused.rs fuel-dispatch/src/fkc/register.rs
git commit -m "feat(fkc): bundle slot-name side-table on FusedKernelRegistry (Finding 5.4 FKC side)"
```

### Task 3.7 — FDX `NAME_TABLE` (Finding 5.4, FDX side)

- [ ] **Step 1: Write the failing test** — in `fuel-memory/src/dlpack_view.rs` `mod tests`. **Correction:** `OutputView` has no `Default`/`test_default()` (storage.rs:22-45 derives only `Debug, Clone`) — construct all fields explicitly.

```rust
#[test]
fn fdx_output_view_slot_name_is_recoverable_from_the_name_table() {
    use fuel_ir::{DType, Shape, Layout};
    let ov = OutputView {
        byte_offset: 0,
        len_elements: 6,
        dtype: DType::F32,
        shape: Shape::from_dims(&[2, 3]),
        layout: Layout::contiguous(Shape::from_dims(&[2, 3])),
        name: Some("last_state"),
    };
    let fdx = output_view_to_fdx(&ov).expect("rank 2 lowers");
    assert_eq!(fdx.name_hash, fnv1a("last_state"));
    assert_eq!(fdx_slot_name(fdx.name_hash), Some("last_state".to_string()));
    assert_eq!(fdx_slot_name(0), None);
}
```

(Confirm `OutputView`'s exact field set at `fuel-ir/src/storage.rs:22-45` when transcribing; the load-bearing field is `name: Some("last_state")`.)

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-memory --lib dlpack_view::tests::fdx_output_view_slot_name`
  Expected: FAIL — `output_view_to_fdx` discards the name (dlpack_view.rs:456); no `NAME_TABLE`/`fdx_slot_name`.

- [ ] **Step 3: Implement** — in `dlpack_view.rs`:

```rust
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
static FDX_NAME_TABLE: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
fn name_table() -> &'static Mutex<HashMap<u64, String>> { FDX_NAME_TABLE.get_or_init(|| Mutex::new(HashMap::new())) }

/// Recover a bundle slot's source name from its FNV-1a hash (0 = anonymous).
pub fn fdx_slot_name(hash: u64) -> Option<String> {
    if hash == 0 { return None; }
    name_table().lock().ok()?.get(&hash).cloned()
}
```

  In `output_view_to_fdx`, replace the `name_hash: ov.name.map_or(0, fnv1a)` computation with:

```rust
    let name_hash = match ov.name {
        Some(n) => { let h = fnv1a(n); name_table().lock().unwrap().insert(h, n.to_string()); h }
        None => 0,
    };
```

  and use `name_hash` in the `FDXOutputView { .. }` literal. Fix the false 430-431 doc-comment ("reduced to a stable FNV-1a hash side-table entry") to state the side-table now genuinely exists.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-memory --lib dlpack_view::tests::fdx_output_view_slot_name`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-memory/src/dlpack_view.rs
git commit -m "feat(fdx): populate a slot-name side-table in output_view_to_fdx (Finding 5.4 FDX side)"
```

---

## Task Group 4 — Empirical verifier + ledger (`verify/`) — V-FKC-9 precision half

A git-checked-in JSON verification ledger + an import-time gate that downgrades any machine-checkable precision claim to `PrecisionGuarantee::UNAUDITED` unless a matching passing ledger record exists for the contract's `kernel_revision_hash`, plus a live-hardware harness that produces those records. **The gate runs as `gate_precision` in `import_bundle_str`, NOT in `lower_precision`** (see Global Constraints). Keyed on `(kernel_revision_hash, backend, dtypes, claim)`; `kernel_ref` in the record is human-review-only. Ledger embedded via `include_str!` + `OnceLock` so the gate runs in every hardware-free `cargo test`.

**Depends on:** Group 0 (ImportWarning, ImportedProvider.warnings), Group 1 (`solve_probe_shapes` — harness only).

**Files:**
- Create: `docs/kernel-contracts/.fkc-verified-ledger.json` (initial content: `[]`)
- Create: `fuel-dispatch/src/fkc/verify/{mod.rs, ledger.rs, bit_stability.rs, ulp.rs, accept_coverage.rs, invoker_cpu.rs, invoker_cuda.rs, invoker_vulkan.rs}`
- Create: `docs/kernel-contracts/cuda/rope-apply.fkc.md` (acceptance target, Task 4.6)
- Modify: `fuel-dispatch/src/fkc/mod.rs` (`mod verify;`)
- Modify: `fuel-dispatch/src/fkc/register.rs` (gate pass in `import_bundle_str`)
- Modify: `fuel-dispatch/src/kernel.rs` (add `iter_entries`)
- Modify: `fuel-dispatch/Cargo.toml` (serde_json unconditional)

**Interfaces:**
- Consumes: `PrecisionGuarantee` + `UNAUDITED` + `none` (fused.rs); `Resolved`/`ResolvedPrimitive`/`ResolvedFused` (lower.rs); `ImportWarning` + `ImportedProvider.warnings` (Group 0); `solve_probe_shapes` (Group 1); backend storage APIs.
- Produces: `verify::{VerificationLedger, LedgerRecord, LedgerQuery, gate_precision, embedded, KernelInvoker, HostTensor, VerifyError, VerifyOutcome, verify_bit_stability, verify_precision_bound, CpuInvoker}`; `KernelBindingTable::iter_entries`.

### Task 4.1 — Ledger file + `VerificationLedger` type + `embedded()` loader + `has_pass`

- [ ] **Step 1: Preconditions** — create `docs/kernel-contracts/.fkc-verified-ledger.json` containing exactly `[]`. Confirm it is NOT gitignored: `git check-ignore docs/kernel-contracts/.fkc-verified-ledger.json` (Expected: no output = not ignored; if ignored, add a negation rule to `.gitignore` and `git add -f`). In `fuel-dispatch/Cargo.toml`: change line 47 `serde_json = { workspace = true, optional = true }` → `serde_json = { workspace = true }`, and drop `"dep:serde_json"` from the `telemetry` feature list (line 79).

- [ ] **Step 2: Write the failing test** — in `fuel-dispatch/src/fkc/verify/ledger.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{probe::BackendId, DType};

    #[test]
    fn ledger_from_json_roundtrips_and_has_pass_matches_on_revision_and_claim() {
        let json = r#"[{
            "kernel_ref": "rope_apply_f32", "backend": "Cuda", "dtypes": ["F32"],
            "kernel_revision_hash": 1234567890123456789, "claim": "bit_stable_on_same_hardware",
            "result": "pass", "verified_at": "2026-07-11T00:00:00Z", "protocol_version": 1,
            "evidence": {"repeat_calls": 150}
        }]"#;
        let ledger = VerificationLedger::from_json(json).expect("parses");
        assert!(ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456788, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "max_ulp"));
        assert!(!ledger.has_pass(BackendId::Cpu, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        assert!(!ledger.has_pass(BackendId::Cuda, &[DType::F16], 1234567890123456789, "bit_stable_on_same_hardware"));
        let failing = VerificationLedger::from_json(&json.replace("\"pass\"", "\"fail\"")).unwrap();
        assert!(!failing.has_pass(BackendId::Cuda, &[DType::F32], 1234567890123456789, "bit_stable_on_same_hardware"));
        assert_eq!(VerificationLedger::embedded().len(), 0);
    }
}
```

- [ ] **Step 3: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::verify::ledger::tests::ledger_from_json_roundtrips`
  Expected: FAIL — the module/types/file don't exist.

- [ ] **Step 4: Implement** — `verify/mod.rs`: `mod ledger; mod bit_stability; mod ulp; mod accept_coverage; mod invoker_cpu; #[cfg(feature="cuda")] mod invoker_cuda; #[cfg(feature="vulkan")] mod invoker_vulkan;` + `pub use ledger::{VerificationLedger, LedgerRecord, LedgerQuery, gate_precision}; pub use bit_stability::{KernelInvoker, HostTensor, VerifyError, VerifyOutcome, verify_bit_stability}; pub fn embedded() -> &'static VerificationLedger { VerificationLedger::embedded() }`. Add `mod verify;` to `fkc/mod.rs`. `verify/ledger.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use fuel_ir::{probe::BackendId, DType};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LedgerRecord {
    pub kernel_ref: String,
    pub backend: String,                 // "Cpu"|"Cuda"|"Vulkan"|"Metal"
    pub dtypes: Vec<String>,             // DType Debug names, e.g. "F32"
    pub kernel_revision_hash: u64,       // serde_json parses u64 natively
    pub claim: String,                   // bit_stable_on_same_hardware|max_ulp|max_relative|max_absolute|accept_coverage
    pub result: String,                  // pass|fail|no_reference
    pub verified_at: String,
    pub protocol_version: u32,
    #[serde(default)]
    pub evidence: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct VerificationLedger { records: Vec<LedgerRecord> }

const LEDGER_JSON: &str = include_str!("../../../../docs/kernel-contracts/.fkc-verified-ledger.json");

impl VerificationLedger {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { Ok(Self { records: serde_json::from_str(s)? }) }
    pub fn from_records(records: Vec<LedgerRecord>) -> Self { Self { records } }
    pub fn records(&self) -> &[LedgerRecord] { &self.records }
    pub fn push(&mut self, r: LedgerRecord) { self.records.push(r); }
    pub fn len(&self) -> usize { self.records.len() }
    pub fn is_empty(&self) -> bool { self.records.is_empty() }
    pub fn embedded() -> &'static VerificationLedger {
        static L: OnceLock<VerificationLedger> = OnceLock::new();
        L.get_or_init(|| VerificationLedger::from_json(LEDGER_JSON).unwrap_or_default()) // never-panic; empty = conservative
    }
    pub fn has_pass(&self, backend: BackendId, dtypes: &[DType], rev: u64, claim: &str) -> bool {
        self.records.iter().any(|r| r.result == "pass" && r.kernel_revision_hash == rev
            && r.claim == claim && backend_label(backend) == r.backend && dtypes_match(&r.dtypes, dtypes))
    }
}
fn backend_label(b: BackendId) -> &'static str {
    match b { BackendId::Cpu => "Cpu", BackendId::Cuda => "Cuda", BackendId::Vulkan => "Vulkan", BackendId::Metal => "Metal", _ => "Unknown" }
}
fn dtypes_match(rec: &[String], want: &[DType]) -> bool {
    rec.len() == want.len() && rec.iter().zip(want).all(|(s, d)| *s == format!("{d:?}"))
}
```

- [ ] **Step 5: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::verify::ledger::tests::ledger_from_json_roundtrips`
  Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/kernel-contracts/.fkc-verified-ledger.json fuel-dispatch/Cargo.toml fuel-dispatch/src/fkc/mod.rs fuel-dispatch/src/fkc/verify/mod.rs fuel-dispatch/src/fkc/verify/ledger.rs
git commit -m "feat(fkc): verification ledger type + embedded loader + has_pass (V-FKC-9)"
```

### Task 4.2 — `LedgerQuery` + `gate_precision` (the V-FKC-9 precision gate, pure logic)

- [ ] **Step 1: Write the failing test** — in `verify/ledger.rs` (a `mod gate_tests`):

```rust
#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use fuel_ir::{probe::BackendId, DType};

    fn claim() -> PrecisionGuarantee {
        PrecisionGuarantee { bit_stable_on_same_hardware: true, max_ulp: Some(0), max_relative: None, max_absolute: None, notes: "audited exact f32 add" }
    }
    fn q() -> LedgerQuery<'static> {
        LedgerQuery { kernel_ref: "rope_apply_f32", backend: BackendId::Cuda, dtypes: &[DType::F32], kernel_revision_hash: 42 }
    }
    fn pass(c: &str) -> LedgerRecord {
        LedgerRecord { kernel_ref: "rope_apply_f32".into(), backend: "Cuda".into(), dtypes: vec!["F32".into()],
            kernel_revision_hash: 42, claim: c.into(), result: "pass".into(), verified_at: "t".into(), protocol_version: 1, evidence: serde_json::Value::Null }
    }

    #[test]
    fn no_ledger_entry_downgrades_to_unaudited_and_warns() {
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &VerificationLedger::default(), &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(!g.bit_stable_on_same_hardware);
        assert!(g.max_ulp.is_none());
        assert_eq!(w.len(), 1);
        assert!(w[0].message.contains("rope_apply_f32") && w[0].message.contains("bit_stable_on_same_hardware") && w[0].message.contains("max_ulp"));
    }
    #[test]
    fn matching_pass_entries_for_every_claim_are_honored() {
        let ledger = VerificationLedger::from_records(vec![pass("bit_stable_on_same_hardware"), pass("max_ulp")]);
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &ledger, &mut w);
        assert!(g.bit_stable_on_same_hardware && g.max_ulp == Some(0) && w.is_empty());
    }
    #[test]
    fn partial_backing_still_downgrades_the_whole_claim() {
        let ledger = VerificationLedger::from_records(vec![pass("bit_stable_on_same_hardware")]);
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &ledger, &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(w[0].message.contains("max_ulp"));
    }
    #[test]
    fn stale_hash_downgrades_even_with_a_pass_for_the_old_hash() {
        let mut old = pass("bit_stable_on_same_hardware"); old.kernel_revision_hash = 41;
        let mut w = Vec::new();
        let g = gate_precision(claim(), &q(), &VerificationLedger::from_records(vec![old]), &mut w);
        assert_eq!(g.notes, PrecisionGuarantee::UNAUDITED.notes);
    }
    #[test]
    fn no_verifiable_bound_passes_through_untouched() {
        let declared = PrecisionGuarantee::none("audited; no static bound applies");
        let mut w = Vec::new();
        let g = gate_precision(declared, &q(), &VerificationLedger::default(), &mut w);
        assert_eq!(g.notes, declared.notes);
        assert!(w.is_empty());
    }
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::verify::ledger::gate_tests`
  Expected: FAIL — `LedgerQuery`/`gate_precision` don't exist.

- [ ] **Step 3: Implement** — in `verify/ledger.rs`:

```rust
use crate::fused::PrecisionGuarantee;
use crate::fkc::ImportWarning;

pub struct LedgerQuery<'a> {
    pub kernel_ref: &'a str,   // diagnostic-only; NOT part of the match key
    pub backend: BackendId,
    pub dtypes: &'a [DType],
    pub kernel_revision_hash: u64,
}

/// V-FKC-9 precision gate. Any machine-checkable claim in `declared` (bit_stable
/// / max_ulp / max_relative / max_absolute) must have a matching `pass` ledger
/// record for the CURRENT kernel_revision_hash, else the WHOLE guarantee collapses
/// to UNAUDITED + a warning. An audited-none (no bounds) guarantee passes through.
pub fn gate_precision(declared: PrecisionGuarantee, q: &LedgerQuery, ledger: &VerificationLedger, warnings: &mut Vec<ImportWarning>) -> PrecisionGuarantee {
    let mut unbacked: Vec<&'static str> = Vec::new();
    let check = |c: &'static str| ledger.has_pass(q.backend, q.dtypes, q.kernel_revision_hash, c);
    if declared.bit_stable_on_same_hardware && !check("bit_stable_on_same_hardware") { unbacked.push("bit_stable_on_same_hardware"); }
    if declared.max_ulp.is_some()      && !check("max_ulp")      { unbacked.push("max_ulp"); }
    if declared.max_relative.is_some() && !check("max_relative") { unbacked.push("max_relative"); }
    if declared.max_absolute.is_some() && !check("max_absolute") { unbacked.push("max_absolute"); }
    if unbacked.is_empty() { return declared; }
    warnings.push(ImportWarning {
        section: q.kernel_ref.to_string(),
        message: format!("precision claim(s) {unbacked:?} for kernel `{}` ({:?}, dtypes {:?}, rev {}) have no passing \
            verification-ledger entry — downgraded to UNAUDITED (run the fkc_verify harness to earn them)",
            q.kernel_ref, q.backend, q.dtypes, q.kernel_revision_hash),
    });
    PrecisionGuarantee::UNAUDITED
}
```

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::verify::ledger::gate_tests`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/verify/ledger.rs
git commit -m "feat(fkc): gate_precision — downgrade unverified precision claims to UNAUDITED"
```

### Task 4.3 — Wire the gate into `import_bundle_str`

- [ ] **Step 1: Write the failing test** — in `register.rs` `mod tests` (uses `EntryPointLink` @643 + `ELEMENTWISE_BINARY` @519). **Robust across chassis-vs-concrete add_f32** (both declare bit_stable and both downgrade):

```rust
#[test]
fn importing_elementwise_binary_downgrades_add_f32_against_the_empty_embedded_ledger() {
    let link = EntryPointLink::new();
    let provider = import_bundle_str(ELEMENTWISE_BINARY, &link).expect("imports");
    let add = provider.primitives.iter().find(|p|
        p.op == OpKind::AddElementwise && p.dtypes.as_slice() == [DType::F32, DType::F32, DType::F32]).expect("add_f32 present");
    assert!(!add.precision.bit_stable_on_same_hardware, "unverified bit_stable claim must be downgraded at import");
    assert!(add.precision.max_ulp.is_none(), "unverified ulp bound must be dropped");
    assert_eq!(add.precision.notes, crate::fused::PrecisionGuarantee::UNAUDITED.notes);
    assert!(provider.warnings.iter().any(|w| w.message.contains("bit_stable_on_same_hardware")),
        "a downgrade warning was recorded: {:?}", provider.warnings);
}
```

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::register::tests::importing_elementwise_binary_downgrades_add_f32`
  Expected: FAIL — `import_bundle_str` does not run the gate, so add_f32 keeps its declared bit_stable/ulp.

- [ ] **Step 3: Wire the gate** — in `import_bundle_str` (register.rs), between `lower_file` and `from_resolved` (this reuses the `warnings` vec Group 0 already created):

```rust
    use crate::fkc::verify::{self, LedgerQuery, gate_precision};
    use crate::fkc::lower::Resolved;
    let mut resolved = resolved;
    let ledger = verify::embedded();
    for r in &mut resolved {
        match r {
            Resolved::Primitive(p) => {
                let q = LedgerQuery { kernel_ref: p.kernel_source.as_str(), backend: p.backend, dtypes: p.dtypes.as_slice(), kernel_revision_hash: p.revision.0 };
                p.precision = gate_precision(p.precision, &q, ledger, &mut warnings);
            }
            Resolved::Fused(f) => {
                let q = LedgerQuery { kernel_ref: f.kernel_source.as_str(), backend: f.backend, dtypes: f.dtypes.as_slice(), kernel_revision_hash: f.revision.0 };
                f.precision = gate_precision(f.precision, &q, ledger, &mut warnings);
            }
        }
    }
```

  (Change `let resolved = lower_file(...)` to `let mut resolved = lower_file(...)` and keep the `from_resolved(..., resolved, warnings)` call from Group 0 Task 0.2.)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::register::tests::importing_elementwise_binary_downgrades_add_f32`
  Expected: PASS. Then `cargo test -p fuel-dispatch --lib fkc::register::tests` — the existing registration/pointer-identity tests (register.rs:596/676/727) stay green (none assert precision).

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/register.rs
git commit -m "feat(fkc): run the V-FKC-9 precision gate at import time (downgrade + warn)"
```

> **Expected behavior change to note in the PR:** with the empty ledger, the live CPU elementwise-binary contract now imports add_f32 with UNAUDITED precision — the design's stated intent. To avoid regressing `BitStablePreferenceFilter`'s placement preference for decode-critical primitives, seed real ledger entries for them via the harness (Task 4.6) in the same PR.

### Task 4.4 — `KernelInvoker` trait + verify fns + `iter_entries` (fake invoker, no hardware)

- [ ] **Step 1: Write the failing test** — in `verify/bit_stability.rs`:

```rust
#[cfg(test)]
mod fake_tests {
    use super::*;
    use crate::fkc::verify::ulp::{verify_precision_bound, Bound};
    use std::sync::atomic::{AtomicU8, Ordering};
    use fuel_ir::DType;

    struct ConstInvoker(Vec<u8>);
    impl KernelInvoker for ConstInvoker {
        fn invoke(&self, _e: &crate::kernel::BindingEntry, _i: &[HostTensor]) -> Result<HostTensor, VerifyError> {
            Ok(HostTensor { dtype: DType::F32, shape: vec![1], bytes: self.0.clone() })
        }
    }
    struct FlakyInvoker(AtomicU8);
    impl KernelInvoker for FlakyInvoker {
        fn invoke(&self, _e: &crate::kernel::BindingEntry, _i: &[HostTensor]) -> Result<HostTensor, VerifyError> {
            let n = self.0.fetch_add(1, Ordering::Relaxed);
            Ok(HostTensor { dtype: DType::F32, shape: vec![1], bytes: vec![n] })
        }
    }
    fn probe() -> ProbeInputs { vec![HostTensor { dtype: DType::F32, shape: vec![1], bytes: vec![0,0,0,0] }] }
    fn dummy_entry() -> crate::kernel::BindingEntry {
        fn k(_:&[std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>], _:&mut [std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>], _:&[fuel_ir::Layout], _:&crate::kernel::OpParams) -> fuel_ir::Result<()> { Ok(()) }
        crate::kernel::BindingEntry { kernel: k, caps: crate::kernel::KernelCaps::empty(), precision: crate::fused::PrecisionGuarantee::UNAUDITED, cost: crate::kernel::unknown_cost, kernel_source: "", is_generic: false, kernel_revision_hash: 0, cost_expr: None }
    }

    #[test]
    fn verify_bit_stability_passes_for_deterministic_and_fails_for_flaky() {
        let e = dummy_entry();
        assert!(matches!(verify_bit_stability(&ConstInvoker(vec![1,2,3,4]), &e, &[probe()], 16).unwrap(), VerifyOutcome::Pass));
        match verify_bit_stability(&FlakyInvoker(AtomicU8::new(0)), &e, &[probe()], 16).unwrap() {
            VerifyOutcome::Fail { detail } => assert!(detail.contains("diverged")),
            other => panic!("expected Fail, got {other:?}"),
        }
    }
    #[test]
    fn verify_precision_bound_flags_a_candidate_exceeding_max_absolute() {
        let e = dummy_entry();
        let reference = ConstInvoker(1.0f32.to_le_bytes().to_vec());
        let candidate = ConstInvoker(1.5f32.to_le_bytes().to_vec());
        assert!(matches!(verify_precision_bound(&candidate, &reference, &e, &[probe()], Bound::MaxAbsolute(0.25)).unwrap(), VerifyOutcome::Fail { .. }));
        assert!(matches!(verify_precision_bound(&candidate, &reference, &e, &[probe()], Bound::MaxAbsolute(1.0)).unwrap(), VerifyOutcome::Pass));
    }
}
```

(Note the `dummy_entry` includes `cost_expr: None` — the field Group 2 Task 2.1 added.)

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::verify::bit_stability::fake_tests`
  Expected: FAIL — `KernelInvoker`/`HostTensor`/`VerifyError`/`VerifyOutcome`/`verify_bit_stability`/`verify_precision_bound`/`Bound` don't exist.

- [ ] **Step 3: Implement** — `verify/bit_stability.rs`:

```rust
use fuel_ir::DType;
use crate::kernel::BindingEntry;

pub struct HostTensor { pub dtype: DType, pub shape: Vec<usize>, pub bytes: Vec<u8> }
pub type ProbeInputs = Vec<HostTensor>;

#[derive(Debug)] pub enum VerifyError { Invoke(String), NoReference, Backend(String) }
#[derive(Debug)] pub enum VerifyOutcome { Pass, Fail { detail: String }, NoReference }

pub trait KernelInvoker { fn invoke(&self, entry: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError>; }

/// Generalization of the worktree gemm_dense.rs determinism_audit: N repeat calls
/// per probe, byte-identical outputs required.
pub fn verify_bit_stability(inv: &dyn KernelInvoker, e: &BindingEntry, probes: &[ProbeInputs], iters: usize) -> Result<VerifyOutcome, VerifyError> {
    for (pi, probe) in probes.iter().enumerate() {
        let first = inv.invoke(e, probe)?;
        for i in 1..iters {
            if inv.invoke(e, probe)?.bytes != first.bytes {
                return Ok(VerifyOutcome::Fail { detail: format!("probe {pi} diverged at call {i}") });
            }
        }
    }
    Ok(VerifyOutcome::Pass)
}

/// xorshift64* deterministic fill (ported verbatim from the gemm_dense precedent).
pub fn fill_deterministic(len: usize, mut seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
        let r = seed.wrapping_mul(0x2545F4914F6CDD1D);
        v.push(((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
    }
    v
}
```

  `verify/ulp.rs`:

```rust
use crate::kernel::BindingEntry;
use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, ProbeInputs, VerifyError, VerifyOutcome};

pub enum Bound { MaxUlp(u32), MaxRelative(f64), MaxAbsolute(f64) }

/// Diff a candidate against a REFERENCE-tagged invoker on the same probes.
pub fn verify_precision_bound(cand: &dyn KernelInvoker, refr: &dyn KernelInvoker, e: &BindingEntry, probes: &[ProbeInputs], bound: Bound) -> Result<VerifyOutcome, VerifyError> {
    for probe in probes {
        let a = cand.invoke(e, probe)?;
        let b = refr.invoke(e, probe)?;
        let (af, bf): (&[f32], &[f32]) = (bytemuck::cast_slice(&a.bytes), bytemuck::cast_slice(&b.bytes));
        for (x, y) in af.iter().zip(bf) {
            let ok = match bound {
                Bound::MaxAbsolute(m) => (x - y).abs() as f64 <= m,
                Bound::MaxRelative(m) => ((x - y).abs() / y.abs().max(f32::EPSILON)) as f64 <= m,
                Bound::MaxUlp(m) => ((x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs() as u32) <= m,
            };
            if !ok { return Ok(VerifyOutcome::Fail { detail: format!("{x} vs {y} exceeds bound") }); }
        }
    }
    Ok(VerifyOutcome::Pass)
}
```

  `verify/accept_coverage.rs`: a stub `pub fn verify_accept_coverage(...)` returning `VerifyOutcome` (smoke-tests declared combos; reuses Group 3's return-rule interpreter for the output shape/dtype check — can be a minimal placeholder in Phase 1, filled by the harness in Task 4.6). Add `iter_entries` to `kernel.rs`:

```rust
    /// Yield every registered (op, dtypes, backend, &entry) for static ops
    /// (mirrors iter_precision; runtime-fused/JIT entries are filtered out).
    pub fn iter_entries(&self) -> impl Iterator<Item = (OpKind, &[DType], BackendId, &BindingEntry)> {
        self.bindings.iter().filter_map(|((k, d, b), alts)| k.static_op().map(|op| (op, d, b, alts)))
            .flat_map(|(op, d, b, alts)| alts.iter().map(move |e| (op, d.as_slice(), *b, e)))
    }
```

  (Confirm the exact `BindingKey::static_op()`/`KernelDTypes::as_slice()` accessor names against `iter_precision` at kernel.rs:1314-1326 when transcribing.)

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::verify::bit_stability::fake_tests`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/verify/bit_stability.rs fuel-dispatch/src/fkc/verify/ulp.rs fuel-dispatch/src/fkc/verify/accept_coverage.rs fuel-dispatch/src/fkc/verify/mod.rs fuel-dispatch/src/kernel.rs
git commit -m "feat(fkc): KernelInvoker + verify_bit_stability/verify_precision_bound + iter_entries"
```

### Task 4.5 — `CpuInvoker` (testable now) + `#[ignore]`'d CUDA/Vulkan invokers

- [ ] **Step 1: Write the failing test** — in `verify/invoker_cpu.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker};
    use fuel_ir::DType;

    #[test]
    fn cpu_invoker_runs_add_elementwise_f32_end_to_end() {
        // Use the real CPU add wrapper as the KernelRef (mirror the wrapper used
        // by register.rs:717-722's live-kernel test).
        let e = crate::kernel::BindingEntry {
            kernel: crate::dispatch::add_elementwise_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(), precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost, kernel_source: "portable-cpu", is_generic: false, kernel_revision_hash: 0, cost_expr: None,
        };
        let inv = CpuInvoker::new(DType::F32, vec![3]);
        let a = HostTensor { dtype: DType::F32, shape: vec![3], bytes: bytemuck::cast_slice(&[1.0f32,2.0,3.0]).to_vec() };
        let b = HostTensor { dtype: DType::F32, shape: vec![3], bytes: bytemuck::cast_slice(&[4.0f32,5.0,6.0]).to_vec() };
        let out = inv.invoke(&e, &[a, b]).expect("cpu invoke");
        let got: &[f32] = bytemuck::cast_slice(&out.bytes);
        assert_eq!(got, &[5.0, 7.0, 9.0]);
    }
}
```

(Confirm the exact CPU add wrapper path — `crate::dispatch::add_elementwise_f32_cpu_wrapper` or similar — from the existing `cpu_link_registry_binds_elementwise_binary_to_live_kernels` test at register.rs:727.)

- [ ] **Step 2: Run it to verify it fails** — `cargo test -p fuel-dispatch --lib fkc::verify::invoker_cpu::tests::cpu_invoker_runs_add_elementwise_f32`
  Expected: FAIL — `CpuInvoker` doesn't exist.

- [ ] **Step 3: Implement `CpuInvoker`** — `verify/invoker_cpu.rs`:

```rust
use std::sync::{Arc, RwLock};
use fuel_ir::DType;
use crate::kernel::BindingEntry;
use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, VerifyError};

pub struct CpuInvoker { out_dtype: DType, out_shape: Vec<usize>, params: crate::kernel::OpParams }
impl CpuInvoker {
    pub fn new(out_dtype: DType, out_shape: Vec<usize>) -> Self { Self { out_dtype, out_shape, params: crate::kernel::OpParams::None } }
    pub fn with_params(mut self, p: crate::kernel::OpParams) -> Self { self.params = p; self }
}
impl KernelInvoker for CpuInvoker {
    fn invoke(&self, e: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError> {
        let ins: Vec<Arc<RwLock<fuel_memory::Storage>>> = inputs.iter().map(|t| Arc::new(RwLock::new(
            fuel_memory::Storage::new(fuel_memory::BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(&t.bytes)), t.dtype)))).collect();
        let elem = self.out_shape.iter().product::<usize>();
        let out = Arc::new(RwLock::new(fuel_memory::alloc_cpu_zeroed(self.out_dtype, elem).map_err(|e| VerifyError::Backend(e.to_string()))?));
        let mut outs = [out.clone()];
        let layouts: Vec<fuel_ir::Layout> = inputs.iter().map(|t| fuel_ir::Layout::contiguous(fuel_ir::Shape::from_dims(&t.shape)))
            .chain(std::iter::once(fuel_ir::Layout::contiguous(fuel_ir::Shape::from_dims(&self.out_shape)))).collect();
        (e.kernel)(&ins, &mut outs, &layouts, &self.params).map_err(|er| VerifyError::Invoke(format!("{er:?}")))?;
        let g = out.read().unwrap();
        let bytes = fuel_memory::dispatch_storage!(&g.inner, s => s.bytes().to_vec());
        Ok(HostTensor { dtype: self.out_dtype, shape: self.out_shape.clone(), bytes })
    }
}
```

  `verify/invoker_cuda.rs` (`#[cfg(feature="cuda")]`) and `verify/invoker_vulkan.rs` (`#[cfg(feature="vulkan")]`): analogous, uploading `HostTensor.bytes` to `CudaStorage`/`VulkanStorage` (`CudaStorage` = `CudaStorageBytes` alias, fuel-memory:41; use `from_cpu_bytes`/`to_cpu_bytes` as in the gemm_dense precedent) and Vulkan's command-buffer + fence-wait + readback. Their live-hardware tests are `#[ignore]`'d.

- [ ] **Step 4: Run it to verify it passes** — `cargo test -p fuel-dispatch --lib fkc::verify::invoker_cpu::tests::cpu_invoker_runs_add_elementwise_f32`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fuel-dispatch/src/fkc/verify/invoker_cpu.rs fuel-dispatch/src/fkc/verify/invoker_cuda.rs fuel-dispatch/src/fkc/verify/invoker_vulkan.rs fuel-dispatch/src/fkc/verify/mod.rs
git commit -m "feat(fkc): CPU KernelInvoker (live) + CUDA/Vulkan invoker scaffolds"
```

### Task 4.6 — `rope_apply` contract + acceptance harness (writes the ledger; unblocks the paused fuel-core test)

This is the acceptance test the whole program exists to satisfy. It requires a live CUDA device and is `#[ignore]`'d.

- [ ] **Step 1: Author the contract** — create `docs/kernel-contracts/cuda/rope-apply.fkc.md`: front-matter `provider: { name: fuel-cuda-backend, backend: Cuda, kernel_source: baracuda }`; one kernel section fanning dtypes `{F32, F16, BF16, F64}`; `op_kind: Rope`; `entry_point` base `baracuda_kernels_rope_apply` (fanned `_f32`/`_f16`/`_bf16`/`_f64` → `baracuda_kernels_rope_apply_<dt>_run`); `accept.inputs` q/k + cos/sin (contiguous); `return: passthrough(q)` shape/dtype; `precision: { bit_stable_on_same_hardware: true, audited: true, notes: "deterministic rope apply; caller-supplied cos/sin" }`; `cost: { provenance: judge_measured }`. Verify the exact `OpKind`/entry-point symbols against `baracuda-kernels-sys` (`baracuda_attention.cuh:1703`/`1105`/`1017`) and the FFI surface.

- [ ] **Step 2: Write the failing acceptance test** — in `verify/mod.rs`:

```rust
#[test]
#[ignore = "requires a live CUDA device + --features cuda"]
fn fkc_verify_rope_apply_writes_a_pass_ledger_entry() {
    let ledger = run_fkc_verify_harness(&["rope_apply_f32"], true).expect("harness runs");
    assert!(ledger.records().iter().any(|r|
        r.kernel_ref == "rope_apply_f32" && r.backend == "Cuda" && r.claim == "bit_stable_on_same_hardware" && r.result == "pass"));
    let ledger_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/kernel-contracts/.fkc-verified-ledger.json");
    std::fs::write(ledger_path, serde_json::to_string_pretty(ledger.records()).unwrap()).unwrap();
}
```

- [ ] **Step 3: Run it (red)** — `cargo test -p fuel-dispatch --features cuda fkc::verify::fkc_verify_rope_apply -- --ignored`
  Expected: FAIL — `run_fkc_verify_harness` doesn't exist; no ledger entry.
  (Prepend `PATH="/c/Program Files/NVIDIA/CUDNN/v9.23/bin/13.3/x64:$PATH"` if the test binary needs cuDNN to launch; build from a VS Developer shell or set `NVCC_CCBIN` for the CUDA build.)

- [ ] **Step 4: Implement `run_fkc_verify_harness`** — in `verify/mod.rs`: `pub fn run_fkc_verify_harness(kernels: &[&str], force: bool) -> Result<VerificationLedger, VerifyError>`: import the rope-apply contract; for each `BindingEntry` via `table.iter_entries()` whose contract claims something and has no current-hash ledger match (or `force`): solve probe shapes (Group 1 `solve_probe_shapes`, or a rope-specific fixed set with synthesized q/k/cos/sin), run `verify_bit_stability` (CUDA invoker) + `verify_precision_bound` vs the CPU rope REFERENCE alternative, and push a `LedgerRecord { result: pass/fail/no_reference }` keyed by `e.kernel_revision_hash`. `verified_at` from `std::time::SystemTime` epoch (no chrono).

- [ ] **Step 5: Run it (green) + confirm the acceptance signal** — `cargo test -p fuel-dispatch --features cuda fkc::verify::fkc_verify_rope_apply -- --ignored` (PASS, ledger written). Then the FINAL acceptance signal on the paused worktree branch: `cargo test -p fuel-core --features cuda --lib forward_with_kv_context_captured_matches_persistent -- --ignored` passes WITHOUT hand-wiring a rope dispatch wrapper.
  Expected: both PASS. (This is the resume condition in `capturedrun-4b-paused-pending-fkc-verification.md`.)

- [ ] **Step 6: Commit** (the contract + harness + the freshly-written ledger entry)

```bash
git add docs/kernel-contracts/cuda/rope-apply.fkc.md docs/kernel-contracts/.fkc-verified-ledger.json fuel-dispatch/src/fkc/verify/mod.rs
git commit -m "feat(fkc): rope_apply contract + acceptance harness — first verified ledger entry"
```

---

## Post-implementation

- [ ] **Seed ledger entries for decode-critical primitives** — to avoid the intended-but-unwanted placement-preference regression from Task 4.3, run the harness (Task 4.6 pattern) over the CPU/CUDA MatMul/MulElementwise/RmsNormLastDim/Softmax/LogSoftmax/Rope kernels the CapturedRun session hand-audited, writing their verified ledger entries in the same PR. This is a separate, explicitly-scoped follow-on (not blocking the four components).
- [ ] **Docs** — per the project's "docs are part of every material change" convention, update `docs/session-prompts/kernel-contract-adoption-plan.md` §3/§5/§2.3/§12 to reflect the shipped mechanisms (the finalize()-based dup detection, the store-on-entry cost design, the lower_fused cross-check placement, the import_bundle_str gate) and remove the now-resolved §12 `kernel_revision_hash` risk. Add a `docs/architecture/10-decisions-log.md` entry.
- [ ] **Whole-branch review** — `superpowers:requesting-code-review`, then `superpowers:finishing-a-development-branch`.

## Known Phase-1 limitations (call out in the PR, do not silently ship)

- Cost pricing populates `flops` only; `bytes_moved`/overhead stay 0 until contracts carry a `bytes:` expression (a bandwidth-bound op could rank cheaper than reality). Documented v1 limitation.
- `lookup_cost` (kernel.rs:1295) returns a bare `CostFn` and drops `cost_expr` — currently tests-only; add a doc note so a future production consumer knows to price via `compute_static_costs`, not `lookup_cost`.
- The fused symbol binder binds only `n` + `dtype_bytes`; a fused (m,n,k) formula under-binds → falls back to the compose-from-decompose estimate (a correct non-zero cost).
- MKL/AOCL/Metal remain entirely outside FKC's reach; CPU `QMatMul` still has the "contract shipped, never wired" gap. Tracked as separate widening work (audit Part VI item 7).
- The verifier covers `bit_stable_on_same_hardware` + `max_ulp`/`max_relative`/`max_absolute` + a minimal `accept:` smoke test; retroactively re-verifying the full ~400-contract corpus is separate follow-on work.


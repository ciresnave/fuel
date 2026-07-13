# FKC design-vs-implementation gap audit — the complete fingerprint

**Status (2026-07-11): reference document, produced by 5 parallel deep-research passes over the
entire `docs/session-prompts/kernel-contract-adoption-plan.md` design doc, each independently
cross-checking every mechanism it specifies against the actual shipped code in
`fuel-dispatch/src/fkc/`, `fuel-dispatch/src/kernel.rs`, `fuel-ir/src/dlpack/sidecar.rs`,
`fuel-memory/src/dlpack_view.rs`, and `fuel-graph/src/registry.rs`.** This is not a summary — it
is the full fingerprint. Every claim below carries file:line evidence gathered by direct reading
of the live source, not inference. Several findings were independently rediscovered by 2-3 of the
5 research passes working different angles of the doc — those are marked **[cross-validated]** and
should be treated as the highest-confidence findings in this document.

**Purpose**: this document exists so that closing the gap between FKC's design and its
implementation can proceed as real test-driven development — every gap below is written precisely
enough to become a red test (a test that fails against today's code and passes once the gap is
closed), per the user's explicit instruction: *"every crevice where code should change to make FKC
function the way it was intended documented thoroughly so we can generate comprehensive red tests
to ensure that every hole, every leak is plugged."*

**Relationship to other docs**: `docs/session-prompts/capturedrun-4b-paused-pending-fkc-verification.md`
is the resume anchor for the CapturedRun work this audit was triggered by, and contains a *design
sketch* for the fix (the automated per-claim-type verifier). This document is the *problem
statement* that sketch is answering — read this one first if you're deciding what to build; read
the pause doc for how the CapturedRun thread specifically depends on it.

---

## How to use this document

Each finding has the same shape:

- **Design claim** — what `kernel-contract-adoption-plan.md` says should happen (section + paraphrase or quote).
- **Status** — MATCHES / PARTIAL / DRIFTED / MISSING / DEAD-CODE, defined precisely:
  - MATCHES: implemented as specified, no gap.
  - PARTIAL: implemented, but narrower or weaker than specified in a specific, describable way.
  - DRIFTED: implemented differently than specified — not necessarily worse, but the design doc's description of *how* it works is wrong, which matters for anyone reading the doc to understand the system.
  - MISSING: the mechanism does not exist anywhere in the codebase, in any form.
  - DEAD-CODE: an artifact (usually an error variant) exists with the right name but is never actually constructed/reached by any code path — functionally identical to MISSING but worth distinguishing because it creates false confidence when grepping for the name.
- **Evidence** — exact file:line citations.
- **Gap description** — the concrete failure mode this allows, in plain terms.
- **Red-test sketch** — a named test, its fixture, what it currently observes, what it should observe once fixed. Sketches are written to be directly transcribable into real test code.

Findings are grouped by design-doc section (§0 through §12) to make cross-referencing back to the
design doc mechanical, then followed by cross-cutting sections (provider migration status,
downstream blast radius, the shared failure pattern, positive/corrective findings, and a
recommended build order).

---

## Executive summary — the calibrated verdict

The user's working hypothesis was: *"everything we have built on top of FKC is not working
correctly because of it."* The evidence supports a more precise claim, which is arguably worse in
one specific respect and better in most others:

**Narrower than "everything is broken":** the live, unconditional, production blast radius of the
precision-verification gap specifically is `BitStablePreferenceFilter` (a *soft* placement
preference) deprioritizing kernels that lack a verified bit-stability claim — exactly the
CapturedRun symptom already diagnosed. `PrecisionFloorFilter` (the mechanism that *could* hard-fail
or hard-pass on precision) is wired into the default filter chain but has **zero production
callers** setting a non-default requirement — it is dormant, not misbehaving. Cost claims authored
in a contract **never reach the runtime cost model as raw numbers at all** (see §2.3 finding) —
so a wrong or fabricated cost claim in a contract cannot currently cause a bad VRAM/placement
decision, because the mechanism that would consume it doesn't exist. The kernels themselves, the
executor, and the majority of ranker logic are unaffected.

**Worse than a coverage gap, in one specific and important way:** V-FKC-9 — the validator
literally *named after* "a non-reference contract may not ship UNAUDITED precision" — exists, is
invoked on every import, and silently no-ops on exactly the case (`audited: false` + a note) that
essentially all of the 85 kernels this session found used. This is not an acknowledged, flagged gap
(contrast with this same codebase's honest `[consumer-ahead]` annotations elsewhere) — it is a
validator whose name and doc-comment table promise enforcement the code does not perform, which is
a **false sense of security baked into the system's own self-description**.

**The single largest mechanism found completely missing**: §5's "return-contract validation against
graph-side metadata" — the check that a fused op's declared `shape_rule`/`dtype_rule` actually
matches the real registered Rust function in `fuel-graph` — does not exist anywhere, in any form.
**Three independent research passes found this same absence** from three different angles (reading
§4-5 directly, reading §2.2's fused-mapping table, and reading V-FKC-7's validator). One of them
found the codebase's own source comment (`validate.rs:963-966`) making a **false claim** that this
check happens "in the register slice" — it does not. This means every `fused_op` FKC contract's
declared return shape/dtype behavior is, today, unverified prose.

**The second-largest**: §2.3's cost-expression compiler — the mechanism that's supposed to turn a
contract's authored `flops`/`bytes_moved` formulas into a live, evaluated `CostFn` — was never
built. **Three independent research passes converged on this from different angles** (reading §2.3
directly, tracing the blast radius of cost claims, and tracing V-FKC-9's cost half). A contract's
cost data is parsed, validated for *syntax*, and then discarded at registration time unless the
kernel separately pins a hand-written Rust cost function by name — which happens in exactly 1 of
several hundred corpus contracts.

**A structural pattern, not a list of unrelated bugs**: every real gap found — V-FKC-2, V-FKC-7,
both halves of V-FKC-9, the §5 mechanism, the §2.3 compiler, the §8 "shared hash," the §8.5
slot-name side-table, §9's incomplete front-matter check — has the *same shape*: a correctly-named
error variant and/or validator function exists, is documented as covering the design's requirement,
and either (a) is never invoked on the path that matters, (b) is invoked but its trigger condition
is narrower than the documented intent, or (c) doesn't exist at all despite a comment claiming it
does. This is worth internalizing as *the* signature of this gap class, because it means grepping
for an error variant's existence is not sufficient evidence that the check it names actually runs —
you must trace the call site too, every time.

---

## Part I — The shared failure pattern (read this before the catalog)

Given how consistently it recurs, stating it once precisely: **a correctly-named validator or
error variant existing in the code is not evidence it enforces what its name says.** Concretely
observed instantiations of this pattern:

1. **Named-but-uninvoked**: `FkcError::BlurbMismatch` (§1.2/V-FKC-2) is a real enum variant that is
   never constructed anywhere in the crate.
2. **Invoked-but-narrower**: `validate_precision_coherence` ("Rule 9" as coded) is invoked on every
   import, but its actual check is "does `determinism: nondeterministic` agree with
   `bit_stable/audited`" — a hand-authored-field coherence check — not "is this contract's
   precision claim non-placeholder," which is what its name and the design doc's V-FKC-9 promise.
3. **Doesn't-exist-but-documented-as-existing**: `validate.rs:963-966`'s comment says bundle-slot
   rank for `shape_rule:`-derived slots is "covered" by "the register slice's `FusedOpEntry`
   cross-check" — no such cross-check exists anywhere in `register.rs` or elsewhere. This is the
   most serious instance: it is not merely an absence, it is a **false statement in the source**
   that would mislead a future maintainer into believing a safety net exists where there is none.
4. **Escape-hatch-shaped**: `PlaceholderCost`'s trigger condition (`class.is_empty()`) has the
   identical shape to `PlaceholderPrecision`'s (`audited: false` accepted as valid) — a
   non-placeholder-looking value silently satisfies a check meant to catch placeholders.

When building the automated verifier described in the pause doc's design sketch, treat "the
validator exists" as necessary but never sufficient evidence — every claim needs its *call site*
traced to the actual data it inspects, not just its *existence* confirmed.

---

## Part II — The gap catalog, by design-doc section

### §1 — Module structure and public API surface

**Finding 1.1 — File structure drift.** *(Status: DRIFTED, cosmetic)*
Design says 11 files including a single `link.rs`. Actual: 14 files — `link.rs` is split into
`cpu_link.rs`/`cuda_link.rs`/`vulkan_link.rs` (feature-gated, `mod.rs:44-56`), and `register.rs`
(2065 lines, the largest file, holding the entire `ImportedProvider`/`import_*` public API) exists
under a name the design doc never mentions at all — the design doc implies this lives in `mod.rs`.
No red test warranted (not behavior-affecting) — flagging so a reader of the design doc doesn't go
looking for `link.rs` or expect `mod.rs` to hold the registration logic.

**Finding 1.2 — Stale gating doc-comment.** *(Status: DRIFTED, cosmetic but actively misleading)*
`fuel-dispatch/src/fkc/mod.rs:8-10`'s own doc-comment still says the module "is gated behind the
default-off `fkc` cargo feature," directly contradicting `lib.rs:51`'s unconditional `pub mod fkc`
and the design doc's own later "STATUS UPDATE" admitting the feature was removed. A one-line doc
fix, not a red test.

**Finding 1.3 — `ImportedProvider.primitives`/`.fused` are `pub`, not private.** *(Status: DRIFTED, minor)*
`register.rs:141-152`. Design's code sample implies private fields. Widens public API surface
beyond spec; not itself a safety issue.

**Finding 1.4 — `LinkRegistry` has an undocumented third method.** *(Status: DRIFTED, additive)*
`lower.rs:45-65` — `resolve_cost_fn(&self, name: &str) -> Option<CostFn>` (default impl returns
`None`) exists beyond the design's two-method (`resolve_primitive`/`resolve_fused`) trait. This
backs the "named cost-fn pinning" mechanism (see Finding 2.3.1) that the design doc has no
vocabulary for at all.
**Red-test sketch**: `link_registry_default_resolve_cost_fn_returns_none` — construct a minimal
`impl LinkRegistry` overriding only the two documented methods; assert `.resolve_cost_fn("x")`
returns `None`, confirming the compatibility default doesn't silently break for implementors
written against the design doc's two-method trait.

**Finding 1.5 — `FkcError` variant naming drift (cosmetic renames).** *(Status: DRIFTED, name-only)*
Full mapping (design name → actual name), all behavior-preserving unless separately flagged below:

| Design name | Actual name | Notes |
|---|---|---|
| `YamlTabIndent` | `TabIndentation` | renamed |
| `YamlAnchorDisallowed` | `AnchorDisallowed` + `AliasDisallowed` + `MergeKeyDisallowed` | split into 3, finer-grained |
| `RankExceedsSix` | `BundleSlotRankExceeded` | renamed |
| `OpParamsVariantMismatch` | `BadOpParamsVariant` | renamed |
| `DuplicateKernelRef` | `DuplicateKernelRef(String)` | present but shape differs — see Finding 3.1 |
| `ShapeRuleMismatch` | **absent** | see Finding 5.1 — this is not a rename, it's MISSING |

No red test — naming drift only, except where noted as a separate MISSING finding.

**Finding 1.6 — `BlurbMismatch` is dead code.** *(Status: DEAD-CODE)* — see Finding 10.1 below (grouped
with V-FKC-2 since it's the same finding from two research angles).

---

### §2.1/§2.2 — Contract-to-registry field mapping

**Finding 2.1 — `kernel_revision_hash` claim is STALE (the design doc's gap is CLOSED, this is a
positive/corrective finding, not a new gap).** *(Status: DRIFTED — doc is stale in the good direction)*
Design says primitives have no `BindingEntry` slot for the revision hash yet. Actual:
`kernel.rs:883-891` — `BindingEntry.kernel_revision_hash: u64` is a real field, threaded from
`register.rs:229-243`. Also `is_generic: bool` is a wholly new `BindingEntry` field (Baracuda
structural-miss telemetry) with no analog in the design doc at all. **Action**: update
`kernel-contract-adoption-plan.md`'s §12 risk list to remove this item — it's resolved, and leaving
it listed as an open risk could cause someone to redundantly "fix" something already fixed.

**Finding 2.2 — `register_full_with_source(...)` call shape differs (8 args, Result vs 10 args, `()`).**
*(Status: DRIFTED)* — grouped with the fuller §3 finding below (Finding 3.1), same underlying drift.

**Finding 2.3 — §2.2's "bundle validated against `output_views`" claim is MISSING.** *(Status: MISSING)*
— this is the fused-op half of the §5 finding; see Finding 5.2 below, grouped there since it's the
same absent mechanism viewed from the mapping-table side.

---

### §2.3 — The cost-expression compiler **[HIGH SEVERITY, cross-validated by 3 independent research passes]**

**Finding 2.3.1 — The entire interpreter-backed cost side-table mechanism does not exist.**
*(Status: MISSING)*

**Design claim**: a cost expression compiles to a real `fn` via one of two entry points,
`compile_primitive_cost`/`compile_fused_cost`, backed by strategy (A) — a process-wide
`OnceLock<Vec<CompiledCostExpr>>` side-table keyed by `(OpKind|FusedOpId, dtypes, backend)`, with a
generic `fkc_cost_primitive` trampoline `fn` that re-derives its key at call time and evaluates the
stored AST.

**Evidence of absence** (independently confirmed 3 ways):
- `grep -rn "compile_primitive_cost\|compile_fused_cost\|fkc_cost_primitive\|CostTable" fuel-dispatch/src/fkc/` — zero matches for any of these names anywhere.
- `cost_expr.rs` contains only a parser + a capacity-only interpreter (`pub fn eval`, `cost_expr.rs:427`; `pub fn cost_estimate`, `cost_expr.rs:534`) — real, correct, well-tested code for turning an AST into a number **given a concrete shape**, but nothing wires this into a stored `CostFn` fn-pointer at registration time.
- `grep -rn "cost_estimate(" fuel-dispatch/src` finds exactly **one call site**, inside a `#[test]` (`lower.rs:1300-1334`), whose own comment reads: *"This is what FKC import previously dropped in favor of the `unknown_cost` sentinel"* — the test author already knew the interpreter is disconnected from production.
- `register.rs:221`: `let cost_fn = p.cost_fn.unwrap_or(unknown_cost);` — every imported primitive's actual registered `CostFn` is either the bare `unknown_cost` sentinel (later overwritten by a generic per-`OpKind` default, never the contract's own formula), or a hand-written Rust `CostFn` resolved **by explicit name** via the separate, design-doc-unplanned `cost.cost_fn:` field + `LinkRegistry::resolve_cost_fn` (Finding 1.4). Corpus-wide: `grep -rn "cost_fn:" docs/kernel-contracts/**/*.fkc.md` finds this used in **exactly one file** (`docs/kernel-contracts/cuda/flash_decoding.fkc.md`) out of several hundred contract sections.
- Fused ops are worse: `register.rs:253-264` stamps `fused_unknown_cost` **unconditionally**, with a comment admitting "a follow-up slice adds the cost trampoline" — and `KernelBindingTable::fill_unset_cost_for_backend` (`kernel.rs:1247-1266`) explicitly **skips** fused entries when later upgrading primitive sentinels, so fused-op costs never get even the generic per-`OpKind` fallback; they stay `fused_unknown_cost` permanently.

**Gap description**: a contract author can write a precise, carefully-derived `flops:
"2*batch*m*n*k"` cost formula, have it pass `CostExprParse` validation, and have that number
**never once be consulted by the placement/ranking system** — the runtime cost model silently uses
a generic op-family default instead. This makes cost claims in ~99.7% of the corpus decorative.

**Red-test sketch**: `imported_contract_declared_cost_reaches_binding_cost_fn` — import a contract
with a specific, non-default `cost.flops` formula (e.g. the existing test fixture's
`"2 * batch * m * n * k"`), look up the resulting live `BindingEntry.cost` from the registered
`KernelBindingTable`, invoke it with concrete shapes. **Currently observes**: the stored `CostFn` is
`unknown_cost` or the generic per-`OpKind` post-fill default — provably not the contract's declared
formula. **Should observe**: invoking `BindingEntry.cost` reproduces exactly what `cost_estimate()`
computes from the contract's own parsed AST for the same shapes.

**Finding 2.3.2 — §12's flagged cost-side-table keying risk is moot, but for the wrong reason.**
*(Status: N/A — the risk described cannot occur, because the mechanism it warns about doesn't exist)*
§12 worried that two alternatives at one key with different cost expressions need
`kernel_source`-disambiguated side-table keys. Since no side-table exists (Finding 2.3.1), this
specific bug class cannot occur. The *actually-shipped* replacement mechanism (name-based
`resolve_cost_fn` lookup) is inherently disambiguated by construction (distinct Rust symbol names →
distinct functions). No red test needed — this is a "risk correctly avoided, by accident of a
different design" finding, worth recording so nobody spends effort hardening a side-table that
isn't there.

---

### §3 — `register_full_with_source` panic → Result

**Finding 3.1 — The never-panic fix shipped via a different architecture than specified.**
*(Status: DRIFTED — outcome achieved, mechanism materially different)*

**Design claim**: `register_full_with_source` itself becomes `Result`-returning; its thin wrappers
(`register`, `register_with_caps`, `register_with_precision`, `register_with_caps_and_precision`,
`register_full`) all thread `?` and become `Result`-returning too; a new typed
`DuplicateKernelRef{op, dtypes, backend}` struct variant is added, "mirroring `NoBackendForOp`."

**Evidence**:
- `kernel.rs:1059-1087` — `register_full_with_source` still returns `()`. So do all five named
  wrapper functions (verified at `kernel.rs:957, 976, 996, 1013, 1037`).
- Instead, registration became **append-only and unconditionally infallible**
  (`kernel.rs:1123-1130`) — duplicate detection moved to a **separate**, new method
  `pub fn finalize(&self) -> Result<()>` (`kernel.rs:1147-1165`), which the caller must invoke once
  after a batch of registrations.
- `fuel_ir::Error` has **no `DuplicateKernelRef` variant at all** (confirmed via full-file grep of
  `fuel-ir/src/error.rs`) — `finalize()` returns a generic, stringly-typed
  `Error::Msg(format!("KernelBindingTable: duplicate KernelRef registered for (op={op:?}...)..."))`
  (`kernel.rs:1153-1159`), not a structured `{op, dtypes, backend}` variant.
- `FkcError::DuplicateKernelRef` (the importer-facing error) is correspondingly `DuplicateKernelRef(String)`
  (`error.rs:233-234`) — a string wrapper, not the structured type the design's §3.1 code sample implies.
- Hand-written bulk callers (`register_cpu_kernels` etc.) remain `()`-returning, contrary to §3.2's
  claim they "themselves become `-> Result<(), Error>`." Instead, the process-init boundary
  (`global_bindings()`/`extend_global_bindings()`) calls `t.finalize().expect(...)` **once**, at the
  end — functionally matching the design's fail-fast *intent*, via a different call graph.

**Gap description**: the outcome (no panic, importer surfaces `Result`, hand-written tables still
fail fast at init) matches the design's *goals*. But nearly every mechanical claim about *how* is
wrong: `register(...)` itself never returns an error for a duplicate — only a later, separate
`finalize()` call does. Someone reading §3 to find "where does `?` catch a duplicate registration"
would look in the wrong function.

**Red-test sketch**: none needed for behavior (the existing test
`step_9a_duplicate_kernel_ref_detected_by_finalize_not_panic`, `kernel.rs:1654`, correctly covers
the shipped design) — this finding is purely about correcting the design doc's description, not
about a missing safety check. Recommended action: rewrite §3 of the design doc to describe the
actual `finalize()`-based architecture, and add a typed `DuplicateKernelRef{op, dtypes, backend}`
struct variant to `fuel_ir::Error` if the structured-error ergonomics are still wanted (currently a
`String` — functionally fine, but loses machine-inspectability for tooling that wants to react to
*which* op/dtypes/backend collided without string-parsing).

---

### §4.1 — Parse layer (restricted YAML subset)

All three core claims MATCH, with only cosmetic naming drift (Finding 1.5's table covers the
renames: `YamlTabIndent`→`TabIndentation`, `YamlAnchorDisallowed` split 3 ways). The Norway-problem
defense (`family: no` staying the string `"no"`) is implemented via two independent layers
(schema-level `String` typing + a dedicated pre-pass rejecting unquoted Norway tokens,
`parse.rs:241-307`) and is well-tested (`mod.rs:418-486`). No gaps found in this section.

**Finding 4.1.1 — §4.1 born-red tests are wired to run, but the workspace-wide CI job that would
enforce them is documented-red for unrelated reasons.** *(Status: PARTIAL, operational not code)*
`.github/workflows/rust-ci.yml:77,80` runs `cargo test --workspace`, and per `CLAUDE.md`,
`tensor-tools`'s standing `Device::Cpu` break fails even bare `cargo check` at the workspace root —
so while these tests exist and pass when run scoped (`-p fuel-dispatch`), the all-or-nothing
workspace CI invocation that's supposed to gate them never reliably reaches them. Not a red-test
target — an operational/CI-hygiene fix (scope the CI job, or fix `tensor-tools`).

---

### §4.2 — Lower layer

**Finding 4.2.1 — The "exhaustive match forces a compile error" safety property is FALSE for
`OpKind`/`BackendId` (though TRUE for `DType`).** *(Status: DRIFTED — a real, specific safety claim
that does not hold)*

**Design claim** (§2.1): "NOT `FromStr`-by-discriminant — an exhaustive match so a new `OpKind`
forces a compile error to extend the table."

**Evidence**: `lower_op_kind` (`lower.rs:163-298`) has a wildcard `_ => None` arm at line 283. The
code's own comment (`lower.rs:285-290`) admits directly: *"`OpKind` is `#[non_exhaustive]` in
`fuel-ir`, so a wildcard-free exhaustiveness anchor is not possible across the crate boundary... a
new upstream variant simply won't be reachable until a token is added here (an `UnknownOpKind` at
runtime, not a compile error)."* Confirmed: `fuel-ir/src/dispatch.rs:85-86` marks `OpKind`
`#[non_exhaustive]`. Same for `lower_backend` (`lower.rs:430-448`, `BackendId` also
`#[non_exhaustive]`). By contrast `lower_dtype` (`lower.rs:368-403`) genuinely achieves the claimed
property via a second wildcard-free match on the already-resolved `DType` (which is NOT
`#[non_exhaustive]`), backed by a real drift-guard test (`dtype_suffix_is_the_inverse_of_lower_dtype`,
`lower.rs:1692-1706`).

**Gap description**: adding a new `OpKind` variant upstream compiles fine everywhere, silently.
Nothing forces anyone to add the corresponding `lower_op_kind` table entry, and — unlike the
fused-op-id table (which has a real drift-guard, `every_registry_id_is_reachable_through_table`,
`lower.rs:1768`) — there is no runtime drift-guard for `op_kind`/`backend` either. This can go
unnoticed indefinitely; a contract naming the new op_kind just gets `UnknownOpKind` at import time,
forever, with nothing flagging the table is stale.

**Red-test sketch**: `op_kind_table_has_a_drift_guard_vs_fuel_ir` — since true compile-time
exhaustiveness is architecturally impossible under `#[non_exhaustive]`, the achievable fix is a
runtime drift-guard mirroring the fused-op one: enumerate every `OpKind` discriminant `fuel-ir`
exposes (requires `OpKind` to expose something enumerable — if it doesn't, *that absence* is part
of the gap to flag) and assert `lower_op_kind` has table coverage for all of them. Currently: no
such test exists; adding an `OpKind` variant and forgetting the table entry causes zero test
failures anywhere in the suite.

---

### §5 — Return-contract validation against graph-side metadata **[HIGHEST SEVERITY, cross-validated by 3 independent research passes from 3 different entry points]**

**Finding 5.1 — The entire probe-shape sampled-equivalence check does not exist.** *(Status: MISSING)*

**Design claim**: for a `fused_op` contract, the importer looks up the real, registered
`FusedOpEntry` in `fuel_graph::registry::default_registry()` by name and evaluates both the FKC
contract's declared `shape_rule`/`dtype_rule` strings *and* the real registered
`shape_rule`/`dtype_rule` **functions** at one or more probe shapes, requiring agreement —
`ShapeRuleMismatch` on disagreement.

**Evidence, exhaustively confirmed** (this is the single most-corroborated finding in the whole
audit — independently found reading §4-5 directly, reading §2.2's mapping table, and reading
V-FKC-7's validator spec):
- `FkcError::ShapeRuleMismatch` is **not a defined enum variant anywhere**. The only mention in the
  entire crate is a forward-looking promise in a doc-comment (`error.rs:12`): *"...`ShapeRuleMismatch`,
  … land with their respective steps."* That promise was never fulfilled.
- `schema.rs:229` (`OutputDesc::shape_rule: Option<String>`) — the field is parsed and carried as
  opaque text. No `parse_shape_rule` function or equivalent exists anywhere (contrast with
  `dtype_rule`, which genuinely has an interpreting `parse_dtype_rule` for the narrower purpose of
  deriving a binding-table dtype key — see Finding 5.3 below for why even that is narrower than the
  design implies).
- `register_into` (`register.rs:203-277`, the *only* place a fused contract meets a live registry)
  **never imports or calls `fuel_graph::registry::default_registry()`**, never reads
  `FusedOpEntry::shape_rule`/`dtype_rule`, and never evaluates anything at a probe shape. Every
  field on the registered `BackendImpl` comes straight from the contract itself
  (`register.rs:253-264`) — nothing is cross-checked.
- `fuel_graph::registry::default_registry()` **is** used elsewhere in `fuel-dispatch/src/fkc/`, but
  only inside a test, and only to enumerate `FusedOpId`s for the unrelated name-table drift-guard
  (`lower.rs:1768`) — never to fetch or evaluate an actual `shape_rule`/`dtype_rule` function.
- **The codebase's own source contains a false claim about this**: `validate.rs:963-966`'s comment
  reads *"A `shape_rule:` string has no statically-knowable rank without evaluating the rule, so it
  is not rank-checked here (**the register slice's `FusedOpEntry` cross-check covers it**)."* No such
  cross-check exists in `register.rs` or anywhere else. This is not an honest `[consumer-ahead]`
  annotation (which this codebase uses elsewhere, correctly, e.g. for the fused cost-fn gap) — it
  is a statement that something is handled when it is not.
- This is entirely mechanically buildable, not blocked on anything: `fuel-dispatch` already depends
  on `fuel_graph::registry::{FusedOpId, FusedOps, FusedOpParams}` (`lower.rs:21,30`), and the real fn
  signatures exist and are exactly as the design doc describes
  (`fuel-graph/src/registry.rs:118,122`: `shape_rule: fn(&[Shape], &FusedOpParams) -> Shape`,
  `dtype_rule: fn(&[DType], &FusedOpParams) -> DType`).

**Gap description**: a `fused_op` FKC contract's declared return shape/dtype behavior is, in every
literal sense, **unverified prose**. A provider (or a typo, or a copy-paste from a sibling op) can
write `shape_rule: same_as(lhs)` for an op that actually reshapes (a reduction, a transpose, a
fused conv), and `import_bundle_str` returns `Ok` — the contract registers into
`FusedKernelRegistry` exactly as if it were correct. Nothing downstream currently consumes the
`shape_rule` string for a live dispatch decision today (a narrow mitigating fact — see the
dtype-key caveat, Finding 5.3), but the design doc frames §5 as *the* mechanism meant to guarantee
a contract's return-side description matches ground truth before any future consumer trusts it —
and that guarantee is entirely absent.

**Red-test sketch**: `fused_contract_shape_rule_disagreeing_with_registered_fn_is_rejected`
(mirrors the design doc's own stated born-red example almost verbatim — *"a flash-attn contract
whose shape_rule string disagrees with `flash_attn::entry()`'s shape_rule at a probe shape"*).
Fixture: take a real corpus fused contract (e.g. `docs/kernel-contracts/fused/norm-softmax.fkc.md`'s
`RmsNormLastDim`, whose real `FusedOpEntry` computes `same_as(x)`), mutate its `shape_rule:` string
to something provably false at a probe shape (e.g. a fixed constant shape unrelated to input, or a
transpose claim). **Currently observes**: `import_bundle_str` returns `Ok`, `register_into`
succeeds — the wrong claim is silently accepted. **Should observe**: `Err(FkcError::ShapeRuleMismatch{..})`
(needs the variant defined first), raised by evaluating the real `FusedOpEntry::shape_rule` fn at
≥1 probe shape and comparing against the contract's claimed result.

**Finding 5.2 — The `output_views`/bundle-arity cross-check does not exist.** *(Status: MISSING)*

**Design claim**: "For a multi-output fused op the importer cross-checks `output_views(...)[0]`
against `shape_rule`/`dtype_rule`."

**Evidence**: `grep -rn "output_views" fuel-dispatch/src/fkc/` — zero matches anywhere in the
module (the field only exists on the `fuel-graph`-side `FusedOpEntry` struct itself,
`fuel-graph/src/registry.rs:147-148`). The FKC-side `return.bundle` is parsed (`schema.rs:211-215`)
and its *first slot's* dtype_rule is extracted purely to build the binding-table key
(`lower.rs:766-793`) — never compared against `output_views`.

**Red-test sketch**: `bundle_slot_count_mismatch_vs_output_views_arity_is_rejected` — pick a real
multi-output fused op registered with a known `output_views` arity, author a contract with a
`bundle:` of the wrong slot count. **Currently observes**: `Ok` — silently registers a wrong-arity
bundle description. **Should observe**: a typed arity-mismatch error (possibly `ShapeRuleMismatch`,
possibly a dedicated variant, since this is arity not shape-rule-string content per se).

**Finding 5.3 — Bundle slot rank ≤ 6 is only enforced for static shape literals, never for
`shape_rule:`-derived slots (the common case).** *(Status: PARTIAL — narrower than described)*

**Evidence**: `check_bundle_ranks` (`validate.rs:967-997`) only rank-checks a slot when the contract
gives a **static** `shape: [d0, d1, ...]` literal (`validate.rs:984`). A slot described only via a
`shape_rule:` string — the common real-corpus pattern (e.g. `shape_rule: same_as(a)`,
`lower.rs:1514-1515`) — is never rank-checked at all; the validator's own comment defers this to the
nonexistent §5 cross-check (Finding 5.1). In practice, for essentially every real authored bundle
contract, rank ≤ 6 is not actually enforced.

**Gap description**: the doc's own resolved-decisions preamble (line 32) states rank ≤ 6 is a "hard
serialization/wire-format limit" (`[u64;6]`-backed). A `shape_rule`-described slot that would in
fact produce rank > 6 registers without error today.

**Red-test sketch**: `bundle_slot_shape_rule_only_is_never_rank_checked` — a bundle contract with a
`shape_rule:`-only slot whose real evaluated rank (once §5's cross-check exists) would exceed 6.
**Currently observes**: `Ok`. **Should observe** (once Finding 5.1 is fixed and this check is wired
to consume it): `Err(FkcError::BundleSlotRankExceeded{..})`.

**Finding 5.4 — Bundle slot NAMES never survive past the parse layer; the promised side-table does
not exist anywhere [cross-validated independently on both the FKC and FDX sides].** *(Status: MISSING)*

**Design claim** (§5, second half + §8): "Bundle slot names are kept in a side-table for round-trip
— the hash covers the structured contract, the side-table preserves the human-readable slot names
the `[u64;6]`/`name_hash` form would otherwise lose."

**Evidence, both sides of the boundary checked independently**:
- FKC side: no `slot_name`/`SlotName`/`NameTable`/`side_table` artifact exists anywhere in
  `fuel-dispatch/src/*` beyond two local variables used purely for an error message
  (`validate.rs:978,990`). `bundle_primary_dtype_rule` (`lower.rs:766-793`) extracts a slot's `name`
  only as a diagnostic-message fallback (`lower.rs:790`) — never stored on `ResolvedFused`
  (`lower.rs:118-144`, no slot-name field exists there) or on `BackendImpl` (`fused.rs:49+`, same).
- FDX side: `fuel-ir/src/dlpack/sidecar.rs:229-244` (`FDXOutputView`) has only `name_hash: u64` — no
  companion string map anywhere. `output_view_to_fdx` (`fuel-memory/src/dlpack_view.rs:428-459`) is
  strictly one-way; its own doc-comment (line 430-431) *claims* "reduced to a stable FNV-1a hash
  side-table entry," but line 456 shows `name_hash: ov.name.map_or(0, fnv1a)` with **no table
  populated anywhere** — this is the same "comment claims a mechanism that doesn't exist" pattern
  as Finding 5.1, just on the FDX side of the same feature.
- `docs/specs/kernel-contract-format.md` §5.5 describes the intended design but cites no
  implementation.

**Gap description**: after registration, no artifact anywhere carries a bundle's per-slot names —
they are parsed, used transiently for one diagnostic message, and discarded. A tool wanting to
recover a bundle output's human-readable name from a persisted `name_hash` (telemetry, debugging a
captured plan) has no API to do so — the name is permanently lost at construction, on both the FKC
and FDX sides of what the design frames as one shared mechanism.

**Red-test sketch**: `bundle_slot_names_survive_registration_and_are_recoverable` — import a
multi-slot bundle contract, call `register_into`, attempt to recover the original slot names for
the registered `(FusedOpId, backend, dtypes)` key via any public API. **Currently observes**: no
such API exists — names are unreachable after import (test would fail to compile against a
currently-nonexistent method). **Should observe**: a lookup (e.g. `bundle_slot_names(id) ->
&[String]`) returns the original names.

---

### §6 — Five-flag layout set vs `KernelCaps`

The core projection formula (`strided_input = strided==accepted && broadcast_stride0==accepted`)
MATCHES exactly (`caps_map.rs:113-120`), and `start_offset`/`reverse_strides` genuinely are parsed
and retained on the resolved record without being projected, exactly as designed
(`caps_map.rs:346-374`).

**Finding 6.1 — `awkward_layout_strategy` is validated but NOT retained, contradicting the design's
explicit "validated ... and retained" claim.** *(Status: MISSING — retention half only)*

**Evidence**: validation is real (`validate.rs:460-532`, `awkward_strategy_coherence`, reading the
raw `LayoutSpec`). But `ResolvedLayout` (`caps_map.rs:92-106`) has **no** field for it, and
`resolve_layout` (`caps_map.rs:126-169`) never copies it forward. Neither `ResolvedPrimitive` nor
`ResolvedFused` carries it anywhere. Once `validate_file` passes, the parsed string is gone — no
post-validation record contains it.

**Gap description**: a planner follow-up that wants to "insert `Op::Contiguize` + sum its FKC cost"
per the declared strategy (the design's own stated planner consequence) cannot read the strategy off
the resolved record as promised — the value doesn't survive that long. This is a real, narrow
regression specifically for this one flag; `reverse_strides`/`start_offset` genuinely are retained
correctly (contrast case, proving this isn't a systemic retention failure, just this one field).

**Red-test sketch**: `resolved_primitive_retains_awkward_layout_strategy` — import a contract
declaring `awkward_layout_strategy: contiguize_internally` on an operand; assert (once a retention
field exists) `resolved.layouts[i].awkward_layout_strategy == Some("contiguize_internally")`.
Currently: no such field exists to assert against — provable today by asserting neither
`ResolvedLayout` nor `ResolvedPrimitive` expose the string anywhere post-import.

**Finding 6.2 — The "forward-extension hook" claim ("filled the moment `KernelCaps` grows the
fields") is only true for 2 of the 3 predicted fields.** *(Status: PARTIAL)*
`reverse_strides`/`start_offset_capable` genuinely have values sitting ready on `ResolvedLayout`
today (per the MATCHES findings above) — growing `KernelCaps` with these two fields really would be
"free" plumbing, as claimed. `awkward_layout_strategy` is not ready (Finding 6.1) — growing
`KernelCaps` with that field would require a full new plumbing pass (schema → `ResolvedLayout` →
`resolve_layout` → `project`), contrary to what the design implies. No additional red test beyond
6.1 — same underlying gap, just documenting where the "hook" claim over-promises.

**Finding 6.3 — An undocumented third capability flag (`requires_broadcast`/`broadcast_stride0:
required`) shipped after this design-doc section was written and was never folded back into it.**
*(Status: DRIFTED — doc-only staleness, not a code defect)*
`kernel.rs:75-89` — `KernelCaps.requires_broadcast: bool`, driven by a third `Tri` state,
`broadcast_stride0: required` (`caps_map.rs:37-48`, `113-120`), plus a `broadcast_axes` mask
(`schema.rs:300-311`, `validate.rs:419-454`). This is real, correct, well-tested code (landed as
"path 1a," commit `a59b6aa4` per this session's git log) that simply post-dates §6's prose. A reader
relying on §6 alone would not know `KernelCaps` already has a second bool or that `broadcast_stride0`
has 3 meaningful states, not 2. Documentation-sync action, not a red test.

---

### §7 — Quant, sub-byte, scale single-place rule

Most of §7 MATCHES precisely: the GGML variant-name→numeric-code mapping (`validate.rs:80-100`,
correctly rejecting the GGUF-format name `Q4_K_M` in favor of the real `Q4K` code), the scale
single-place rule (`ScaleDoubleDeclared`, `validate.rs:583-609`), and sub-byte logical-shape
handling (never derived from `size_in_bytes()`, `validate.rs:345-352`) are all implemented as
designed, with only a minor caveat that the schema has no field for a "sidecar `scale_buffer`"
concept at all yet, so that half of the rule is unenforceable because the field doesn't exist
(a schema gap, not a validator weakness).

**Finding 7.1 — `MxNotYetRegistrable` fires at parse/import time, not "at the register step" as
documented.** *(Status: DRIFTED)*

**Design claim**: "an MX contract **parse-validates** but returns `MxNotYetRegistrable` **at the
register step** (not at parse) — the contract is legal, the consumer is behind."

**Evidence**: `MxNotYetRegistrable` is raised inside `quant_coherence` (`validate.rs:661-711`),
called from `validate_kernel`→`validate_file`, which runs **inside** `import_bundle_str`
(`register.rs:305`) — before `lower_file` (line 308) and before an `ImportedProvider` is even
constructed (line 312-317). For a `registrable: true` MX kernel, `import_bundle_str` itself fails —
the caller never gets an `ImportedProvider` to call `register_into` on at all. The doc's two-stage
story (succeeds at import, fails later at register) does not exist for registrable sections. (For
`registrable: false`/describe-only sections, the opposite happens: the error is swallowed entirely
at validate time and the section never reaches lowering either — so neither of the two stages the
design describes actually occurs as written.)

**Gap description**: tooling written against the design's description (e.g. "introspect an
`ImportedProvider` for its declared-but-inert MX kernels") has no such introspection point — it
never gets that far.

**Red-test sketch**: `mx_contract_can_import_then_fail_at_register` — the literal claim, pinned as a
test: expect `import_bundle_str` to return `Ok(provider)` for a registrable MX contract, and
`provider.register_into(...)` to be what returns `Err(MxNotYetRegistrable)`. **Currently observes**:
this test fails immediately — `import_bundle_str` itself already returns `Err`, so
`register_into` is never reached.

---

### §8 — `kernel_revision_hash`

Canonicalization (comment/whitespace-insensitive, semantic-change-sensitive), the `"auto"`
derivation from `entry_point + revision_base + canonical block`, and a real frozen-fixture test
pinning the exact FNV-1a algorithm all MATCH precisely (`revhash.rs:52-236`).

**Finding 8.1 — The "shared stable hash function with FDX `name_hash`" claim is FALSE — they are
two independent implementations that happen to agree by convention, not by shared code.**
*(Status: DRIFTED — a real, specific risk, not cosmetic)*

**Evidence**: `fuel-ir/src/dlpack/sidecar.rs:461-469` implements its **own private** `fnv1a(s: &str)
-> u64` with the same offset-basis/prime constants as `fuel-dispatch/src/fkc/revhash.rs:28-41` — two
textually independent, non-`pub`, non-cross-referenced functions (one takes `&str`, the other
`&[u8]`) that converge only because someone matched the constants by hand. No `pub use`, no shared
module, no cross-crate call, and **no test anywhere** cross-checks
`fkc::revhash::fnv1a(s.as_bytes()) == fuel_ir::dlpack::sidecar::fnv1a(s)` for the same input.

**Gap description**: "shared function" is supposed to guarantee a persisted
`(backend, op, dtypes, kernel_revision_hash)` tuple re-resolves consistently against FDX's
`name_hash` (the digest §7 mechanism the design cites as the reason for sharing in the first place).
If either copy's constants are ever tuned independently (an FNV variant swap, a bugfix in one
without the other), the two would silently diverge — exactly the failure mode "sharing" was
supposed to prevent, and nothing would catch it.

**Red-test sketch**: `revision_hash_and_fdx_name_hash_use_the_identical_fnv1a` — a cross-crate test
calling both hash primitives (requires widening visibility on at least one, currently both are
private/`pub(crate)` at best — that visibility gap is itself evidence the "shared" claim is
aspirational) on the same input, asserting equality. Cannot currently be written without a
visibility change.

**Finding 8.2 — the bundle slot-name side-table for `kernel_revision_hash` round-trip is MISSING** —
see Finding 5.4 above (same underlying gap, viewed from §8's framing rather than §5's).

---

### §9 — Single-bundle vs globbed multi-file import

Both import modes correctly produce the same `ImportedProvider` shape (`register.rs:349-405`), and
glob order is genuinely sorted for determinism (`register.rs:353-354`).

**Finding 9.1 — `import_glob`'s front-matter-agreement check only covers 3 of the 5 specified
fields; `revision_base` and `link_registry` mismatches across a globbed provider's files go
completely undetected.** *(Status: PARTIAL — a real, specific provenance-corruption risk)*

**Design claim**: front-matter must agree across a glob's files on `provider.name`/`backend`/
`kernel_source`/`link_registry`/`revision_base`, or `ProviderMismatch`.

**Evidence**: `register.rs:363-401` (the merge loop) checks exactly three fields — `name` (371-378),
`backend` (379-386), `kernel_source` (389-396). `link_registry` and `revision_base` both genuinely
exist in the parsed front-matter schema (`schema.rs:45-58`) but `ImportedProvider` doesn't carry
either forward, so there is nothing to compare during the merge. Confirmed by omission: the existing
tests (`register.rs:1100-1173`) only exercise `name`/`backend` mismatches; constructing a
`revision_base`-mismatched glob today would succeed silently.

**Gap description**: `revision_base` is exactly the value folded into every kernel's
`kernel_revision_hash` (§8) — its entire purpose is giving a whole provider's kernels one consistent
revision identity. If two files in a glob'd provider silently disagree on it (e.g. one file stale
from an earlier commit), each file's kernels hash against a *different* baseline while being merged
into what the design calls one coherent provider — undetected provenance corruption, in exactly the
mechanism designed to prevent it.

**Red-test sketch**: `import_glob_mismatched_revision_base_is_provider_mismatch` — two `.fkc.md`
files, agreeing on name/backend/kernel_source, differing only in `revision_base` (`git:aaaa` vs
`git:bbbb`). **Currently observes**: `Ok(merged_provider)` — silently accepted, each file's kernels
now carry inconsistent revision baselines. **Should observe**:
`Err(FkcError::ProviderMismatch{field: "revision_base", ..})`. Parallel test
`import_glob_mismatched_link_registry_is_provider_mismatch` for the `link_registry` field.

---

### §10 — The ten V-FKC-* validators [cross-validated systematically; original V-FKC-9 finding + 4 more found]

The actual code re-numbers everything internally ("Rule 1" through "Rule 17" — grown well beyond
the original 10, `validate.rs:11-30`'s own table). Findings below map to the design doc's original
V-FKC-N numbering for cross-reference.

**V-FKC-1 (required fields)** — MATCHES (a strict superset; also correctly relaxes requirements for
describe-only sections). No gap.

**V-FKC-2 (blurb equality)** — **MISSING** *(Status: MISSING, `BlurbMismatch` is DEAD-CODE)*
Only the weaker "structured field is non-empty" half is implemented (`validate.rs:243-246`,
`MissingBlurb`). The actual prose-vs-structured *equality* check does not exist — the module's own
comment (`validate.rs:32-40`) admits the parsed form has no access to the raw prose text needed to
compare, and defers to a `fkc fmt --check` lint that also does not exist anywhere
(`grep "fkc fmt\|fmt --check"` → nothing but the one aspirational comment). `FkcError::BlurbMismatch`
is a real enum variant (`error.rs:87`) that is **never constructed anywhere in the crate**.
**Red-test sketch**: `blurb_mismatch_between_prose_and_structured_is_never_caught` — a `.fkc.md`
section whose prose says "Computes ReLU" and whose structured `blurb:` says "Computes softmax."
**Currently observes**: `Ok` — nothing reads the prose at all. **Should observe**:
`Err(FkcError::BlurbMismatch{..})` (requires the parser to retain the prose slice alongside the
fenced block, which it currently discards).

**V-FKC-3 (non-overlapping keys / duplicate KernelRef)** — mechanism itself is correct
(`kernel.rs:1147-1165`, `finalize()`, function-pointer-level comparison, keyed by the 3-tuple
`(op,dtypes,backend)` — actually *stricter* grouping than the design's 4-tuple since `kernel_source`
isn't part of the bucket key at all), and it's exercised correctly by production init call sites and
unit fixtures. **The gap is that the standing corpus-wide CI lint never exercises it at all**
*(Status: PARTIAL — real check exists, but the safety net meant to run it over the real corpus
structurally cannot)*. The lint (`ci_lint_corpus_parse_lower_validate`, `validate.rs:1913`) lowers
each kernel section **in total isolation** — a throwaway one-kernel `FkcFile` clone per section
(`validate.rs:2034-2044`) — and never assembles a shared table or calls `register_into`/`finalize()`.
A duplicate `KernelRef` introduced across two real corpus files would not be caught by this lint;
it would only surface later, if and when a production `register_into(...).expect(...)` call site
happens to be exercised (which for feature-gated backends like CUDA/Vulkan may not run in every
environment).
**Red-test sketch**: `corpus_lint_does_not_catch_cross_file_duplicate_kernel_ref` — two synthetic
corpus files under a `_test/` directory sharing one `entry_point` string (resolving to the same
`KernelRef` under a stub `LinkRegistry`). **Currently observes**: the existing lint passes green
(never calls `register_into`). **Should observe** (once the lint is fixed to build one shared table
per provider and call `register_into`/`finalize()` across the whole corpus): a hard failure.

**V-FKC-4 (layout coherence)** — MATCHES, including a §6-additive `broadcast_axes` check beyond the
original design. No gap.

**V-FKC-5 (op-param namespace)** — **stale hand-copied allowlists, over-strict direction**
*(Status: DRIFTED)*. `is_op_params_variant` (`validate.rs:867-913`) is missing 2 of 44 real
`OpParams` variants (`JitScalars`, `NonZeroIndices`); `is_fused_op_params_variant` (`validate.rs:916-942`)
is missing 1 of 23 real `FusedOpParams` variants (`Runtime` — plausibly deliberate, since runtime-fused
arms may not be meant to be hand-authored contract targets; UNCERTAIN, not confirmed either way).
Both lists are hand-typed string literals with zero `#[derive]`/generated link back to the real
enums, so they will keep drifting as the enums grow. Direction is over-rejection (a legitimate
contract naming `NonZeroIndices` gets wrongly rejected), not the under-strict class of gap elsewhere
in this audit — but the same maintenance-drift risk.
**Red-test sketch**: `op_params_namespace_rejects_real_nonzeroindices_variant` — a contract with
`op_params: { variant: NonZeroIndices }` (a real variant). **Currently observes**:
`Err(BadOpParamsVariant)`. **Should observe**: `Ok`, or — if genuinely intentional — an explicit
doc-comment saying so rather than a silent omission.

**V-FKC-6 (quant coherence)** — MATCHES (see §7 discussion above; the one caveat, sidecar
`scale_buffer` having no schema field, is a schema gap not a validator weakness).

**V-FKC-7 (return coherence + rank)** — **the shape/dtype-rule-vs-graph-metadata half is entirely
MISSING** — this is the same finding as Finding 5.1, confirmed independently a third time from the
validator-numbering angle, including independently re-discovering the same false
`validate.rs:963-966` comment. See Finding 5.1 for the full write-up; nothing to add here beyond the
cross-validation itself.

**V-FKC-8 (cost expr parses)** — MATCHES; a genuine recursive-descent grammar, not a stub
(`cost_expr.rs`). No gap in the *parse-validity* check itself (see Finding 2.3.1 for the separate,
much larger gap in what happens to the parsed result afterward).

**V-FKC-9 (non-placeholder precision/cost)** — **both halves confirmed weaker than spec, in
related but distinct ways** *(Status: DRIFTED, both halves)*:
- *Precision half* (original finding, re-confirmed independently 3 times across this audit):
  `precision.rs:78-82` only raises `PlaceholderPrecision` for a **completely absent** block — an
  explicit, well-formed `audited: false` (`precision.rs:115-116`, `Some(false) =>
  Ok(PrecisionGuarantee::UNAUDITED)`) is accepted as a **valid, successful** lowering, never an
  error. `validate_precision_coherence` ("Rule 9" as coded, `validate.rs:1080-1107`) only fires when
  `determinism: nondeterministic` is separately declared — it never fires on `audited: false` alone.
  There is additionally **no "reference vs non-reference" concept anywhere in the validator** — even
  if the placeholder check were tightened, the code has no way today to know which contracts are
  exempt as reference implementations (the design's "non-reference contract" qualifier is entirely
  unimplemented as a concept).
- *Cost half* (new finding, distinct mechanism from the precision half): `PlaceholderCost`'s trigger
  (`validate.rs:1057-1071`) collapses to "error only when `class` is **completely absent**" — because
  `class != "free" && class.is_empty()` is trivially true only when `class` really is empty, *any*
  non-empty `class:` string (verified: every sampled real corpus contract sets one) silences the
  check entirely. `PlaceholderCost` is thus currently **vacuous over the whole real corpus** — the
  same escape-hatch shape as the precision half, on a different field.
- *A deeper, related fact*: "does this contract's cost pass V-FKC-9" and "does this contract ship a
  real (non-`unknown_cost`) `CostFn`" are two entirely decoupled questions today (see Finding 2.3.1)
  — a contract can honestly pass the cost-expr syntax check and still register `unknown_cost` at
  runtime, because `validate_cost` never inspects whether `p.cost_fn` (the separate name-pinning
  field) is set.
**Red-test sketch 1 (precision escape hatch)**: already covered above and in the pause doc — the
canonical `scatter_add`-style false claim.
**Red-test sketch 2 (cost escape hatch)**: `placeholder_cost_class_field_bypasses_check` — `cost: {
provenance: declared, class: misc, flops: ~, bytes_moved: ~ }` (a non-`free` class label with zero
real coefficients). **Currently observes**: `Ok` (class is non-empty, bypasses). **Should observe**:
`Err(PlaceholderCost)` — the check should validate `class` against a real enum of "load-bearing"
labels rather than "any non-empty string."

**V-FKC-10 (cross-spec token subset)** — **no dedicated test exists at all; the subset property
holds today only by manual, untested duplication** *(Status: PARTIAL)*. `is_fdx_quant_family`/
`is_fdx_granularity` (`validate.rs:63-74`) are hand-copied string-literal match arms, matching
FDX's real numeric code table (`fuel-ir/src/dlpack/codes.rs:76-93`) 1:1 **today**, verified by direct
comparison — but with no shared const, no derive, and **no test anywhere** (confirmed by grep for
"subset"/"V-FKC-10"/"10.12") walking FDX's live table and asserting FKC's is a subset. If FDX adds
or removes a code and FKC's hardcoded list isn't updated in lockstep, nothing notices — in either
direction (over-strict rejection of new valid tokens, or the actual violation V-FKC-10 exists to
prevent: silently accepting a token FDX has since removed).
**Red-test sketch**: `fkc_quant_family_set_is_asserted_subset_of_fdx_codes` — iterate every family/
granularity string FKC accepts, assert each has a live `FDX_QUANT_*`/`FDX_SCALE_GRAN_*` const
counterpart in `fuel_ir::dlpack::codes`; iterate FDX's consts, assert FKC recognizes every current
one. **Currently observes**: no such test exists to run at all.

**§10.1 — The CI lint** — **exists as a materially different artifact than specified, at a
different location, with narrower coverage.** *(Status: DRIFTED)*
`fuel-dispatch/tests/fkc_lint.rs` **does not exist** (confirmed via glob — zero matches; the 27
files under `fuel-dispatch/tests/` are all unrelated `*_live.rs` GPU test files). What exists
instead is an in-crate `#[test]` (`ci_lint_corpus_parse_lower_validate`, `validate.rs:1913`) that
does correctly walk the entire real corpus including `cuda/` (confirmed, not a subset) and does run
under plain `cargo test -p fuel-dispatch` — but it does **not** call the literal public `import_*`
entry points the design specifies; it manually re-implements a narrower parse→validate→lower
sequence, isolating each kernel section (this is the direct cause of the V-FKC-3 gap above — no
shared table is ever built, so `register_into`/`finalize()` is never exercised by the lint at all).
The §8 frozen-fixture test and V-FKC-10 both run under the same `cargo test` invocation as siblings,
not literally *run by* the lint as specified.
**This finding directly explains the "why did nobody notice 85 kernels were unaudited" question**:
since every `audited: false` CUDA contract lowers successfully (precision half of V-FKC-9), the
lint's "0 hard failures" green status is exactly consistent with — and gives zero evidence
against — every CUDA kernel shipping unaudited. The lint isn't malfunctioning; it was never capable
of catching this class of defect, because the check it would need doesn't exist.

---

## Part III — Provider migration status (§11 rollout, ground-truthed against git history)

| Provider | Migration status | Equivalence test exists? | What the equivalence test actually checks |
|---|---|---|---|
| CPU | Most primitive families migrated (elementwise, reduce, norm, rope, indexing, matmul-dense, conv, flash-attn, etc.); **77 `table.register(...)` call sites remain**, including CPU `QMatMul` — **an authored contract exists (`docs/kernel-contracts/quantized/vec-dot-matmul.fkc.md`) and is explicitly commented as staying hand-written, unconsumed** — the identical "contract shipped, never wired" bug class as the `rope_apply_f32` finding that triggered this whole audit, found on a second, independent family. | Yes, multiple | fn-pointer identity + `kernel_source` provenance tag + caps structural match. **Precision is explicitly NOT asserted** — one test's own doc-comment states outright: *"Precision is contract-sourced... so it is NOT asserted uniformly here... the correct 'no audited claim yet' posture."* |
| Vulkan | Claimed "100% contract-sourced" after FlashAttn migration (commit `ec3a94fc`); 15 `table.register(...)` calls remain, not triaged. | Yes, ~15 tests | Same pattern; precision explicitly excluded by doc-comment, same as CPU. |
| CUDA (baracuda) | "CUDA 31/31" families claimed migrated (commit `23c48f36`); 11 `table.register(...)` calls remain, not triaged. Every migrated family's comment says "the SOLE registration path (hand-written regs DELETED)." | Yes, ~10 tests | Same pattern — fn-pointer + tag only, no precision-value assertion found anywhere. |
| Fused (norm-softmax, conv-rope, linear-quant) | Migrated; fused cost is **always** the `fused_unknown_cost` sentinel (Finding 2.3.1) — never anything else. | Not separately audited | — |
| Quantized (GGML), CPU specifically | **NOT migrated** — `qmatmul_f32_cpu_wrapper` registers via plain hand-written `table.register(...)`, explicitly commented as staying outside FKC despite an authored contract existing. Vulkan `QMatMul` **is** migrated. | CPU: none (nothing to test). Vulkan: yes. | — |
| Conv-attn | Largely migrated (CPU flash-attn, Vulkan FlashAttn, CPU/Vulkan conv-with-bias). | Yes, folded into family tests above | Same pattern. |
| MKL / AOCL | **Not migrated at all — entirely outside FKC's reach.** Both backends' binding tables still hand-construct `PrecisionGuarantee` inline in Rust; AOCL's comment admits *"per-shape ULP bounds land with the step-8 calibration framework"* — an unverified claim, permanently exempt from every V-FKC-* validator since these never go through `fkc::import_*` at all. | No | N/A — not FKC. |
| Metal | **Contract files exist** (`docs/kernel-contracts/metal/cast.fkc.md`, `metal/elementwise.fkc.md`) **but nothing in `fuel-metal-backend/src` ever consumes them** — zero references to `register`/`fkc` anywhere in that crate. Pure orphaned documentation. | No | N/A. |

**Bottom line on rollout**: the bulk of the primitive kernel surface (CPU/Vulkan/CUDA elementwise,
reduce, norm, rope, indexing, dense matmul, conv, flash-attn) genuinely is contract-sourced. But
CPU's quantized matmul, all of MKL, all of AOCL, and all of Metal are not — those backends'
precision/cost claims never pass through FKC's validator battery at all, V-FKC-9 or otherwise; they
carry exactly the pre-FKC unaudited-trust problem, permanently, until someone migrates them (§11
step 8 acknowledges this is future work — it is not a broken promise, just genuinely incomplete,
worth tracking precisely rather than assuming "FKC is done" because the design doc's 2026-07-02
status update says steps 1-7 shipped).

---

## Part IV — Downstream blast radius (confirmed consumers, evidence-based)

Traced every real consumer of FKC-populated `PrecisionGuarantee`/`KernelCaps`/`CostEstimate` data:

1. **`BitStablePreferenceFilter`** (`ranker/filters/bit_stable_pref.rs`) — the **one live,
   unconditional, production-active** consumer of `bit_stable_on_same_hardware`. Soft preference
   (`min_remaining: 1`) — deprioritizes non-bit-stable candidates only when a bit-stable alternative
   exists, never hard-rejects. This is the confirmed, sole mechanism behind the CapturedRun blocker
   pattern this whole audit was triggered by.
2. **`PrecisionFloorFilter`** (`ranker/filters/precision_floor.rs`) — a **hard** gate, wired into the
   default chain, and it handles `None` bounds correctly/conservatively (a candidate with
   `max_ulp: None` fails an explicit numeric floor request — confirmed by test). **But it is
   currently dormant in production**: `PrecisionRequirement` defaults to fully unconstrained, and
   the only two call sites setting a non-default requirement anywhere in the repo are inside
   `#[cfg(test)]` modules. No production caller (fuel-core, CapturedRun, anywhere) ever exercises
   this filter's hard-gate behavior today.
3. **`StridedInputPreferenceFilter`** — consumes the lossy single-bool `strided_input`, but only as
   a soft preference, never a hard gate; nothing downstream treats `strided_input=false` as
   "definitely can't handle `start_offset`" (that capability isn't tracked as a separate flag at
   all yet, per §6, so this specific false-assumption risk doesn't currently materialize).
4. **`VramPressureSelector`** — the one place cost claims could be genuinely consequential (OOM
   avoidance via `bytes_moved`-based fit estimation) — but per Finding 2.3.1, contract-declared cost
   numbers never reach a live `CostFn` unless separately name-pinned, so a wrong *contract-declared*
   cost can't cause a bad VRAM decision today. Additionally, this selector could not be found
   instantiated anywhere in `fuel-core/src` — it appears to be ranker-layer infrastructure not yet
   wired into the production executor path at all, so this consequential-looking consumer is
   currently inert in the shipped serving path regardless of the FKC gap.
5. Everything else touching `PrecisionGuarantee`/`KernelCaps`/`CostEstimate` in the ranker is pure
   cost-based tie-breaking **among already-filtered candidates** — "which backend wins," the
   narrowest, already-acknowledged category.

**Verdict** (restated from the executive summary with its full evidentiary basis now given): the
precision-verification gap's live production effect is bounded to placement *preference*
(`BitStablePreferenceFilter`), not placement *correctness* or resource-safety — the mechanisms that
could cause a harder failure (`PrecisionFloorFilter`, live cost numbers reaching VRAM decisions) are
either dormant or don't exist as wired paths yet. The genuinely more serious finding is not
behavioral blast radius — it's the **integrity gap**: V-FKC-9 (and, per this audit, several
siblings — V-FKC-2, V-FKC-7, both cost-adjacent checks) exist, run, and are named after guarantees
they do not provide, which is a different and arguably worse problem than "this filter's effect is
narrower than expected."

---

## Part V — Positive/corrective findings (design doc claims that are stale in the *good* direction)

Worth recording explicitly so nobody re-does already-finished work:

- **`kernel_revision_hash` now has a real `BindingEntry` slot** (`kernel.rs:883-891`) — the design's
  §12 "risk/open item" flagging this as unresolved is itself now stale. Recommend updating
  `kernel-contract-adoption-plan.md`'s §12 to remove this item.
- **`serde_yaml`-unmaintained risk is resolved** — the project uses `serde_yml` (the maintained
  fork), confirmed at `fuel-dispatch/Cargo.toml:45`, exactly the alternative the design doc itself
  flagged as acceptable.
- **The §12 cost-side-table keying-collision risk cannot occur** — not because it was fixed, but
  because the side-table it worried about was never built (Finding 2.3.1/2.3.2); the actually-shipped
  name-pinning mechanism is inherently collision-free by construction.

---

## Part VI — Recommended build/test order

Given the scale (24 distinct findings across 10 sections), a suggested prioritization for whoever
picks this up, roughly by (severity × how directly it blocks the CapturedRun pause) then by
(severity × corpus-wide blast radius):

1. **V-FKC-9 precision half** — the original trigger, and the design sketch in the pause doc already
   scopes a fix (per-claim-type automated verifier, starting with `bit_stable_on_same_hardware`).
2. **§5 / V-FKC-7 return-contract validation** (Findings 5.1-5.4) — the single largest missing
   mechanism, independently found 3 ways, with an actively false comment in the source pointing at
   it. High value: this is the check that would catch a fused op's contract describing the wrong
   shape/dtype behavior outright, a correctness class of bug, not just a placement-preference one.
3. **§2.3 cost-expression compiler** (Finding 2.3.1) — the second-largest missing mechanism,
   independently found 3 ways. Lower urgency than #1/#2 since the blast-radius trace (Part IV) shows
   it's currently inert rather than actively wrong, but "silently discard the contract author's
   careful cost analysis" is a real design promise unmet at scale (~400 corpus contracts).
4. **V-FKC-9 cost half + V-FKC-2 + V-FKC-3's lint gap** — same escape-hatch/dead-code/no-safety-net
   pattern as #1, smaller individual blast radius, cheap to fix once the pattern-recognition from #1
   is established.
5. **§8's "shared hash" and §8.5's slot-name side-table** — lower urgency (no evidence of current
   silent divergence), but a real latent-drift risk worth closing while the FKC internals are being
   revisited anyway.
6. **§9's incomplete `ProviderMismatch` check, §6's `awkward_layout_strategy` retention, V-FKC-5's
   stale allowlists, V-FKC-10's untested subset property** — smaller, mechanical fixes, good
   candidates for a batch pass once the higher-severity items establish the verification
   infrastructure and testing patterns.
7. **Provider migration completeness** (Part III) — MKL/AOCL/Metal being entirely outside FKC, and
   CPU `QMatMul`'s specific "shipped, never wired" gap — track as a separate, explicitly-scoped
   widening effort per the design doc's own §11 step 8, not urgent relative to items 1-6 which are
   about the *correctness* of the mechanism for providers already migrated.

Documentation cleanup (Findings 1.1-1.3, 6.3, 3.1's description-accuracy half) can happen at any
point and should probably accompany whichever code fix touches the same file, per this project's
"docs are part of every material change" convention.

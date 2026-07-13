# FKC gap closure — Phase 1 design

**Status**: approved 2026-07-11, ready for implementation planning.

**Context**: `docs/session-prompts/fkc-design-vs-implementation-gap-audit.md` (in the
`worktree-capturedrun-executor` branch; FKC source itself is byte-identical to `main`) documents 24
findings where FKC's shipped implementation is narrower, drifted, or missing relative to
`docs/session-prompts/kernel-contract-adoption-plan.md`'s design. `docs/session-prompts/
capturedrun-4b-paused-pending-fkc-verification.md` is paused waiting on the two largest of those
findings (§5, §2.3) plus a real V-FKC-9. This spec covers Phase 1: the three highest-priority items
from the audit's Part VI build order, scoped in a 2026-07-11 brainstorming session. All three
top-priority verification claims were independently re-confirmed against current `main` source
before this design was drafted (see conversation; `ShapeRuleMismatch` is genuinely undefined,
`register_into` genuinely never calls `default_registry()`, `precision.rs:116` genuinely accepts
`audited: false` as a successful `UNAUDITED` lowering, no cost-compiler symbols exist anywhere).

**Explicitly out of scope for Phase 1** (tracked as follow-on work per the audit's Part VI items
4-7): V-FKC-2 blurb equality, V-FKC-3's corpus lint gap, V-FKC-5's stale op-param allowlists,
V-FKC-10's untested FDX-subset property, §6's `awkward_layout_strategy` retention, §9's incomplete
`ProviderMismatch` check, and provider-migration completeness (CPU `QMatMul`, MKL, AOCL, Metal).

## Four components

1. **Shape-constraint parser + solver** (new infrastructure, shared by components 3 and 4)
2. **Cost-expression compiler** (closes §2.3 + the V-FKC-9 cost-half escape hatch)
3. **Return-contract validation** (closes §5 / V-FKC-7, Findings 5.1-5.4)
4. **Empirical verifier + ledger** (closes V-FKC-9 precision-half, the audit's original trigger)

Components 2 and 3 have fully settled designs already (the original adoption plan specifies them
precisely; the gap is that they were never built). Component 4 is genuinely new architecture
sketched only informally in the pause doc; component 1 is new supporting infrastructure both 3 and
4 need. This document designs 1, 3, and 4 in full; component 2 is a straightforward "build what
§2.3 already specifies" and is included for completeness of the file-touch list.

---

## Component 1 — Shape-constraint parser + solver

**Location**: `fuel-dispatch/src/fkc/shape_constraint.rs` (new file), structured like
`cost_expr.rs` (recursive-descent parser + a separate solve step).

**Grammar** (§3.5 of `kernel-contract-adoption-plan.md:528-539`, already ratified, never
implemented — `schema.rs:271-273`'s `shape_constraint: Option<String>` carries it as an opaque,
unparsed string today):

```
same_as=<role>              // element-shape equal to another operand/output
same_rank=<role>            // rank equality only
rank=<n>                    // exact rank (also expressible via the separate `rank:` field)
rank=<lo>..=<hi>            // rank range
broadcast_to=<role>         // NumPy/PyTorch broadcast-compatible with <role>
last_dim_eq=<role>          // last axis equal to <role>'s last axis
dim[<i>]=<expr>             // axis i equals an arithmetic expr over role.dim[j] / literals
divisible(<expr>, <expr>)   // first expr's value is evenly divisible by the second's
capacity_ge(dim[<i>], <sym>) // symbolic-axis capacity >= a bound symbol's min (FDX §6.4)
```

Anything not matching this grammar is a **free-text fallback** (§3.5's own "free text in `notes:`
for anything not yet vocabularized — importer warns, does not reject"). Parsing produces
`Result<ShapeConstraint, FkcError>`; a genuinely unparseable `shape_constraint:` string (not empty,
not matching any token) is a NEW `FkcError::UnparseableShapeConstraint { section, operand, raw }` —
this is a hard reject (the author wrote something in the vocabularized field that isn't valid
vocabulary — different from the `notes:` fallback, which was never meant to be parsed).

**AST** (mirrors `cost_expr.rs`'s `CostNode`):

```rust
pub enum ShapeConstraint {
    SameAs(String),
    SameRank(String),
    Rank(RankSpec),                       // reuse the existing rank: parser (exact/any/range)
    BroadcastTo(String),
    LastDimEq(String),
    DimEq { axis: usize, expr: DimExpr },
    Divisible { lhs: DimExpr, rhs: DimExpr },
    CapacityGe { axis: usize, sym: String },
}

pub enum DimExpr {
    Lit(i64),
    Dim { role: String, axis: DimAxis },   // DimAxis::Index(i) | DimAxis::Last
    Bin { op: BinOp, lhs: Box<DimExpr>, rhs: Box<DimExpr> },
}
```

**Solver**: `pub fn solve_probe_shapes(inputs: &[TensorDesc]) -> Result<Vec<ProbeCombo>, FkcError>`
where `ProbeCombo = Vec<(String /* role */, Shape, DType)>`. Algorithm:

1. For each operand, resolve its rank from `rank:` (exact/range/any → default rank 4 for `any`,
   matching the most common corpus shape).
2. Assign **3 canonical seed profiles** to any axis not pinned by a constraint: `[2, 3, 4, 8]`
   picking a different value per profile (profile A = all-2, profile B = all-4 with one axis
   forced to 3 to catch odd/even bugs, profile C = all-8) — 3 `ProbeCombo`s total per contract,
   matching the "several representative shapes, not one" spirit of the hand-written cuBLAS audit's
   5 shapes, without the combinatorial blowup of a full cartesian product.
3. Apply constraints as a simple union-find-style unification pass over operands: `same_as`/
   `same_rank`/`broadcast_to`/`last_dim_eq` copy or broadcast-reconcile dims across the referenced
   roles; `dim[i]=expr`/`divisible` evaluate `DimExpr` against already-assigned dims (a topological
   pass — an operand whose constraint depends on another operand resolves after it; a cycle is a
   solver error, not a panic); `capacity_ge` is satisfied trivially by construction (seed values are
   always ≥ 1, and symbolic-extent capacity binding is a lowering-time concern this solver doesn't
   need to resolve — it just needs *a* capacity ≥ the bound, so it uses the same seed value as the
   axis itself).
4. dtypes: pick the **first** dtype in each operand's `dtypes:` list (or the first `dtype_class`
   expansion) — claim verification doesn't need to iterate every dtype combination for probe-shape
   purposes (component 4's `accept:` coverage check does that separately, reusing this solver once
   per declared combination instead).

**Failure handling**: an unsatisfiable constraint set (e.g. `dim[0]=q.dim[0]` where `q` doesn't
exist as a role) degrades to seed-values-only for the affected operand plus a `ImportWarning`
(component defined below) — never a hard failure. `UnparseableShapeConstraint` (a real syntax
error in the vocabulary) is the one hard-reject case, since it means the author wrote invalid
vocabulary, not merely something the solver can't yet solve.

**New shared type**: `pub struct ImportWarning { pub section: String, pub message: String }` — FKC
has no warning-collection mechanism today despite §3.5's own doc-comment promising "importer warns,
does not reject." `ImportedProvider` gains a `pub warnings: Vec<ImportWarning>` field, populated by
every soft-fallback path across all four components (solver degradation, ledger-miss downgrades,
etc.), so tests can assert on warning content instead of scraping stderr.

---

## Component 2 — Cost-expression compiler (§2.3)

**Location**: `fuel-dispatch/src/fkc/cost_compile.rs` (new file). Reuses `cost_expr.rs`'s existing
`CompiledCostExpr`/`eval`/`cost_estimate`/`bind_cost_symbols` untouched — this component only adds
the missing *wiring*.

- `OnceLock<Vec<(CostKey, CompiledCostExpr)>>` side-table, `CostKey = (OpKind|FusedOpId, &'static
  [DType], BackendId, &'static str /* kernel_source */)` — the `kernel_source` field is required in
  the key (Finding 2.3.2's §12 risk becomes real once a side-table exists: two alternatives at one
  `(op, dtypes, backend)` key with different declared cost formulas must not collide).
- `pub fn compile_primitive_cost(op: OpKind, dtypes: &'static [DType], backend: BackendId,
  kernel_source: &'static str, expr: CompiledCostExpr) -> CostFn` registers the AST into the
  side-table and returns a generic trampoline `fn fkc_cost_primitive(shapes: &[Shape], dtypes:
  &[DType], params: &OpParams) -> CostEstimate` that re-derives its `CostKey` from the call-site
  `OpKind`/dtypes/backend (closed over at registration — the trampoline itself is a single `fn`
  pointer per distinct key, generated via a small macro or a boxed-closure-to-fn-pointer adapter
  matching how `unknown_cost`/hand-written `CostFn`s are already plain `fn` pointers) and evaluates
  via the existing `cost_estimate()`.
- `compile_fused_cost` is the `FusedOpId`-keyed analog for `register.rs:253-264`'s fused path.
- `register.rs:221`: `let cost_fn = p.cost_fn.unwrap_or_else(|| compile_primitive_cost(p.op,
  dtypes, p.backend, kernel_source, p.cost_expr));` — explicit hand-pinned `cost_fn:` (existing
  mechanism, `LinkRegistry::resolve_cost_fn`) still wins when present; the contract's own formula
  becomes the *next* fallback instead of jumping straight to `unknown_cost`.
- `register.rs:253-264`'s unconditional `cost: fused_unknown_cost` gets the same treatment.
- `kernel.rs:1247-1266` (`fill_unset_cost_for_backend`)'s fused-skip (`let BindingKey::Static(op) =
  key else { continue };`) is now correct to keep as-is for *runtime*-fused entries (no `OpKind` key
  to look up a generic default for) but no longer relevant for *compiled-from-contract* fused
  entries, since those now carry a real `CostFn` from the moment they're registered — never
  `fused_unknown_cost` in the first place.
- **V-FKC-9 cost-half fix** (Finding, `validate.rs:1057-1071`): replace `class.is_empty()` with a
  real closed enum of load-bearing labels (`free | judge_measured | declared_formula | vendor_spec`
  — matching the four provenance/class combinations actually observed in the corpus) and
  additionally check that `provenance: declared` implies a *usable* cost path exists — i.e. either
  `has_any_expr` (an expression this component can compile) or an explicitly pinned `cost.cost_fn:`
  — closing the audit's "passes V-FKC-9 syntax check but still registers `unknown_cost`" decoupling.

---

## Component 3 — Return-contract validation (§5 / V-FKC-7)

**Location**: `fuel-dispatch/src/fkc/return_check.rs` (new file), invoked from `register.rs`'s
fused path (`register_into`, currently lines 246-265).

**New errors** (`error.rs`, finally fulfilling the `error.rs:12` doc-comment's forward promise):
```rust
ShapeRuleMismatch { section: String, role: String, expected: String, actual: String },
BundleArityMismatch { section: String, expected: usize, actual: usize },
```

**Return-rule interpreter**: a small parser for the §5.1/§5.2 return-rule vocabulary
(`passthrough(<role>)`, `fixed(<DType>)`, `same_as(<role>)`, `from_params(<field>, ...)`) — same
shape of work as component 1's grammar, in the same file or a shared `rule_expr.rs` submodule.
`OutputDesc::shape_rule`/`dtype_rule` (`schema.rs:224-229`, currently opaque strings) get evaluated
through this interpreter at each of component 1's solved probe shapes.

**The cross-check** (closing Finding 5.1, the single largest MISSING mechanism, and the false
`validate.rs:963-966` comment): for each `fused_op` contract section, `register_into`:

1. Resolves the `FusedOpId` (already done today for the binding-table key).
2. Looks up the real `FusedOpEntry` via `fuel_graph::registry::default_registry()` (already a
   `fuel-dispatch` dependency per `lower.rs:21,30` — no new crate dependency needed).
3. Runs component 1's solver over the contract's `accept.inputs` to get 3 probe combos.
4. At each combo: evaluates the REAL `FusedOpEntry::shape_rule`/`dtype_rule` fn pointers, and
   separately evaluates the contract's OWN declared `shape_rule:`/`dtype_rule:` strings via the
   return-rule interpreter above. Disagreement at ANY probe combo → `Err(ShapeRuleMismatch)`.
5. If `FusedOpEntry::output_views` is `Some` (multi-output), evaluates it at the same probe combos
   and compares its returned arity against `return.bundle`'s slot count → `Err(BundleArityMismatch)`
   on disagreement (closing Finding 5.2); also compares slot 0's shape/dtype against
   `shape_rule`/`dtype_rule` per the `FusedOpEntry` doc-comment's own stated invariant
   (`fuel-graph/src/registry.rs:138-139`).

**Bundle rank fix** (Finding 5.3): `validate.rs:967-997`'s `check_bundle_ranks` currently only
checks static `shape:` literals. Once this component's interpreter exists, a `shape_rule:`-only
slot gets evaluated at a probe combo and its resulting rank checked the same way (`> 6` →
`BundleSlotRankExceeded`) — this check moves from `validate.rs` (parse-time, no registry access) to
this component (register-time, has registry + solver access), matching what the false comment
already claimed happens.

**Slot-name side-table** (Finding 5.4, both FKC and FDX sides): a new
`HashMap<(FusedOpId, BackendId, &'static [DType]), Vec<String>>` populated by `register_into` from
each bundle slot's `name:` field (currently discarded after one diagnostic-message use in
`validate.rs:978,990`), with a lookup fn `bundle_slot_names(id, backend, dtypes) -> Option<&[String]>`
on `FusedKernelRegistry`. On the FDX side, `fuel-ir/src/dlpack/sidecar.rs`'s `FDXOutputView` gains
a companion `NAME_TABLE: OnceLock<HashMap<u64, String>>` populated by `output_view_to_fdx`
(`fuel-memory/src/dlpack_view.rs:428-459`) at the same point it currently computes `name_hash` and
discards the source string — closing the false "reduced to a stable hash side-table entry" comment
on that function.

---

## Component 4 — Empirical verifier + ledger (V-FKC-9 precision half)

**Location**: `fuel-dispatch/src/fkc/verify/` (new submodule: `mod.rs`, `ledger.rs`,
`bit_stability.rs`, `ulp.rs`, `accept_coverage.rs`). Lives in `fuel-dispatch` because it already
depends on all three backend crates (`fuel-cpu-backend` unconditionally, `fuel-cuda-backend`/
`fuel-vulkan-backend` behind the existing `cuda`/`vulkan` features) — no new crate.

### The ledger

`docs/kernel-contracts/.fkc-verified-ledger.json`, a flat JSON array of records, checked into git
like a lockfile (machine-written, human-reviewed on diff):

```json
{
  "kernel_ref": "rope_apply_f32",
  "backend": "Cuda",
  "dtypes": ["F32"],
  "kernel_revision_hash": 1234567890123456789,
  "claim": "bit_stable_on_same_hardware",
  "result": "pass",
  "verified_at": "2026-07-11T00:00:00Z",
  "protocol_version": 1,
  "evidence": { "repeat_calls": 150, "concurrent_load": true, "cross_process_golden": true }
}
```

One record per `(kernel_ref, backend, dtypes, kernel_revision_hash, claim)`. `claim` is one of
`bit_stable_on_same_hardware | max_ulp | max_relative | max_absolute | accept_coverage`.

### The gate (pure logic, no hardware, runs in every `cargo test -p fuel-dispatch`)

`precision.rs::lower_precision` gains a `ledger: &VerificationLedger` parameter. For a contract
claiming `audited: true` and/or a `bit_stable_on_same_hardware`/ULP bound: look up a matching ledger
record keyed by the CURRENT contract's computed `kernel_revision_hash` (already a real field,
Finding 2.1) and claim type.

- **Match found, `result: pass`** → the claim is honored as declared.
- **No match (missing, or hash mismatch — the kernel changed since last verification)** → the claim
  downgrades: `bit_stable_on_same_hardware` forces to `false`, ULP bounds force to `None`,
  `audited` semantics collapse to `PrecisionGuarantee::UNAUDITED` — plus an `ImportWarning` naming
  the kernel and which claim was downgraded. This is a deliberate, agreed behavior change: it will
  downgrade most of the 85 kernels the 2026-07-11 CapturedRun session hand-flipped, since none of
  them have a ledger entry yet — they were never independently verified, only hand-reasoned-about,
  and this gate makes that distinction real instead of nominal. Re-earning `audited: true` for the
  CapturedRun decode-critical path (MatMul, MulElementwise, RmsNormLastDim, Softmax, LogSoftmax,
  Rope, plus the new `rope_apply`) by actually running the harness against them is the natural
  follow-up once this component exists, and doubles as confirmation the harness works end-to-end.
- **`result: fail`** → same as no-match (downgrade + warning), but the warning explicitly says
  verification was attempted and failed, not merely "never attempted."

### The harness (requires live hardware, `#[ignore]`d, one backend feature at a time per the
project's one-live-suite rule)

A per-backend `KernelInvoker` trait, implemented once per backend:

```rust
trait KernelInvoker {
    fn invoke(&self, binding: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError>;
}
```

- CPU: calls the registered fn pointer directly (already a plain Rust `fn`).
- CUDA: allocates device buffers, copies inputs, calls the registered kernel via the existing
  dispatch path, copies back.
- Vulkan: allocates buffers, records a command buffer, submits, waits on a fence, reads back.

Invoked via `cargo test -p fuel-dispatch --features cuda,vulkan -- --ignored fkc_verify`, iterating
every `KernelBindingTable`/`FusedKernelRegistry` entry whose contract claims something and has no
current-hash ledger match (or `--force`, re-verify everything):

1. **`bit_stable_on_same_hardware`**: generalizes `worktree-capturedrun-executor`'s
   `fuel-cuda-backend/src/baracuda/gemm_dense.rs:406-620` `mod determinism_audit` (not yet on
   `main`) into a reusable `fn verify_bit_stability(invoker: &dyn KernelInvoker, binding:
   &BindingEntry, probes: &[ProbeCombo]) -> BitStabilityResult`: for each probe combo, `fill_deterministic`-seeded
   inputs, `ITERS` (150) repeat in-process calls hashed and compared, one run under a concurrent
   noisy-neighbor thread hammering the same device, one cross-process golden-file comparison (write
   on first run, compare on subsequent). Any divergence → `fail` with the diverging call index as
   evidence.
2. **`max_ulp`/`max_relative`/`max_absolute`**: `fn verify_precision_bound(...)` runs the same
   probes through the candidate AND a `PrecisionGuarantee::REFERENCE`-tagged CPU alternative for the
   same `(op, dtypes)` (looked up via the existing binding table); diffs outputs; checks the claimed
   bound holds. No reference alternative exists for this op → a distinct ledger outcome
   `"result": "no_reference"` (never silently `pass` or `fail`) plus a warning that this claim
   cannot currently be independently verified.
3. **`accept:` coverage**: for every declared dtype × layout × optional-operand combination in
   `accept.inputs` (not just the first, unlike component 1's default), solve one probe combo per
   combination, invoke the kernel, confirm it returns `Ok` and the output's shape/dtype matches
   component 3's return-rule interpreter's evaluation — reusing that interpreter rather than
   duplicating it.

### Acceptance test

Author the (currently nonexistent) FKC contract for `rope_apply_f32`/`f16`/`bf16`/`f64`
(`baracuda_kernels_rope_apply_<dt>_run`, referenced in the pause doc as shipped-but-never-wired),
including real `accept:`/`return:` declarations. Run the harness against it; confirm a ledger entry
is written; confirm `cargo test -p fuel-core --features cuda --lib
forward_with_kv_context_captured_matches_persistent -- --ignored` (on the paused
`worktree-capturedrun-executor` branch) then passes without hand-wiring a dispatch wrapper — the
literal acceptance signal the pause doc specifies.

---

## Data flow (put together)

```
.fkc.md file
  → parse (existing)
  → schema (existing)
  → [1: solve probe shapes from accept.inputs]
  → [4 gate: ledger lookup by kernel_revision_hash+claim → pass through, or downgrade+warn]
  → lower (existing)
  → [3: cross-check return.outputs/bundle against the real FusedOpEntry, using probe shapes]
  → [2: compile cost formula into a live CostFn]
  → register_into → KernelBindingTable / FusedKernelRegistry

separately, offline (live hardware, manual invocation):
  [4 harness] → runs kernels for real → writes/updates the ledger → committed to git
    → next import's gate (above) picks up the fresh verification
```

## Error handling

- **Hard `Err` (contract is factually wrong)**: `ShapeRuleMismatch`, `BundleArityMismatch`,
  `UnparseableShapeConstraint`, `BundleSlotRankExceeded` (moved), tightened `PlaceholderCost`.
- **Downgrade + `ImportWarning` (unverified, not wrong)**: missing/stale/failed ledger entries.
- **Never a panic, anywhere** — every new function returns `Result`, per the project's standing
  never-panic doctrine; solver degradation on unsatisfiable *soft* constraints (not a syntax error)
  is a warning + best-effort fallback, not a `Result::Err`, since the contract itself may be
  correct even if this solver can't fully resolve it yet.

## Testing strategy

- **Components 1, 2, 3, and the component-4 gate**: pure logic, no hardware, full TDD red/green
  cycle exactly as the audit's own red-test sketches describe (e.g.
  `fused_contract_shape_rule_disagreeing_with_registered_fn_is_rejected`,
  `imported_contract_declared_cost_reaches_binding_cost_fn`). These run under plain `cargo test -p
  fuel-dispatch`.
- **Component 4's harness**: the shape-solving, ledger read/write, and comparison logic are
  unit-testable with a fake `KernelInvoker` (a closure returning canned bytes) — no hardware needed
  for those tests. Only the three real `KernelInvoker` impls need live GPU/CPU hardware and are
  `#[ignore]`'d, run manually, one backend at a time, per the project's standing live-GPU discipline.

## Non-goals for Phase 1

- Retroactively re-verifying the full ~400-contract corpus (a separate, explicitly-scoped follow-on
  once this mechanism exists and proves out on the decode-critical path + `rope_apply`).
- Vulkan/CPU verifier adapters can land after the CUDA adapter proves the harness shape works, if
  sequencing pressure demands it — but per this session's decision, all three are in scope for the
  initial build, not deferred by default.
- The remaining ~18 audit findings outside components 1-4 (see "explicitly out of scope" above).

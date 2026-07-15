# Recipe-Identity Verification + the Rope Oracle (Increment 1) — design

**Date:** 2026-07-14 · **Status:** design, pre-plan · **Depends on:** Spec B ingestion service ([[spec-b-ingestion-complete]], merged `capturedrun-4b-resume` @ `cbb2e289`).

## Goal

Make `verify_candidate` trustworthy for a kernel Fuel did **not** write, by verifying it against **Fuel's own registered recipe** for the op it claims to implement — not against the candidate's self-supplied decompose — and by checking **recipe identity**: a candidate claiming to be op *X* whose recipe lowers to a different base map than Fuel's registered *X* is *not the same op* and is rejected-with-feedback. Rope is the driver: it closes the real interleaved-vs-rotate-half convention bug ([[rope-convention-mismatch-baracuda-fuel]], [[rope-not-patternnode]]) that motivated this work, and it is the KISS-#16 reference implementation ("normative consumer-side empirical verification… oracle independence"; see `docs/outreach/kiss-conformance-and-divergences.md`).

This is **Increment 1**. The determinism-class + MathPrecision comparator re-spell + the edge-case corpus (KISS goals 4/6/5-corpus) are a separate KISS-aligned spec (C2). The unification of internal + external fused-op recipes onto one declarative form is the **Tier-2 convergence** (§8), sequenced after.

## Background (what the architecture map established)

- **Base-map lowering exists and is reusable.** `RuleRegistry::lowering_only().optimize_to_fixpoint(graph, roots)` (fuel-graph `opt.rs:218/248`) lowers any fragment to its base map — recursive (multi-level fused ops fully dissolve), side-effect-free (decoupled from placement/backend-stamping), and it drives **both** the 24 static registry decomposes and the runtime `PatternNode` decomposes through **one** `LoweringRule` fn-pointer with no branching. Used exactly this way today in `pipelined.rs`/`fuel-core lazy.rs` (test/call sites).
- **`verify_candidate` today** (jit_ingest.rs:369) realizes **the candidate's own** `decompose` as the reference. Two independence holes: a candidate can supply a self-consistent-but-wrong decompose, and the reference shares no *provenance* check against Fuel's notion of the op.
- **`CandidateKernel` carries no claimed op-identity** — only `decompose: Option<PatternNode>` + `kernel_revision_hash`. `register_runtime_fused` unconditionally mints a fresh `FusedOpId` (no dedup by recipe).
- **Structural-comparison building blocks exist**: `PatternNode` derives `PartialEq`; `op_key` (opt.rs:804) is a NodeId-free structural encoding of 71/117 ops incl. `Op::Fused`; `is_commutative` + sort-before-key (opt.rs:1056) already canonicalizes `a+b`≡`b+a`; a **dormant `by_pattern_hash: HashMap<PatternHash, FusedOpId>`** (registry.rs:734) is reserved for exactly this lookup. No cross-graph comparator exists yet.
- **`realize` is topology-agnostic** — `reference_output`'s realize-a-fragment pattern (jit_ingest_probe.rs) generalizes to any pre-built graph fragment; only its `emit_region` *construction* step is elementwise-limited.

## Design

### 4.1 `CandidateKernel.claimed_op: Option<FusedOpId>`

The op-identity the candidate asserts it implements. `Some(id)` → verify against Fuel's registered recipe for `id` (§4.4). `None` → a **novel** op the candidate defines by its own decompose (§4.6, the adaptive-loop path). One new field; all other Spec-B fields unchanged.

### 4.2 `lower_to_base_map(graph, roots) -> Vec<NodeId>`

A thin convenience wrapper over `RuleRegistry::lowering_only().optimize_to_fixpoint(graph, roots)` (the machinery already exists; this just names it). Lives in fuel-graph `opt.rs`. Side-effect-free; never panics (self-returning decompose = clean fixpoint per G2).

### 4.3 Base-map structural comparator (the recipe-identity primitive)

`fn base_map_hash(graph, root) -> u64` — a recursive **content hash**: extend `op_key` to fold in each child's hash (not its `NodeId`) + `Const` bytes, and canonicalize commutative-operand order via the existing `is_commutative` (reuse, don't re-copy). Two independently-built base maps → directly-comparable digests, no cross-arena merge. Populate the dormant `by_pattern_hash` index with it so recipe-lookup-by-identity is O(1) and `register_runtime_fused` can dedup structurally-identical regions.

**Scope of "identity" (honest):** the hash canonicalizes *decomposition-level* differences (fused→primitive, fusion depth) and *commutative* reorderings — a strong, cheap filter. It does **not** canonicalize associativity/distributivity (that needs an e-graph; none exists and we don't build one). The residual — same op, differently-associated primitives — is caught by the **numerical** check (§4.5). Structural compare = cheap pre-filter; numerical-at-tolerance = the gate.

### 4.4 Reference = Fuel's REGISTERED recipe (oracle-independence)

In `verify_candidate`, when `claimed_op = Some(id)`: build a fresh probe graph, push an `Op::Fused(id, params)` node on `Op::Const` probe leaves, **lower it to its base map** via `lower_to_base_map` (§4.2) — which resolves Fuel's registered recipe through the same `LoweringRule` that `default_registry().entry(id).decompose` / `runtime_fused::runtime_region(id)` back (for rope: `registry::rope::decompose` → the 12-node rotate-half primitive fragment) — then **realize the base map** (reusing `reference_output`'s topology-agnostic realize path). The candidate kernel is compared against **this** primitive-realized reference, never against its own decompose or against Fuel's *fused* kernel (which may be the very thing under test). This closes the independence hole.

### 4.5 The recipe-identity gate

For a candidate claiming a **known** op:
- **Structural (cheap pre-filter, opportunistic):** if the candidate *also* submits a `decompose`, `base_map_hash(lower(submitted))` vs `base_map_hash(lower(registered))`. Mismatch → **reject-with-feedback**: "your recipe for {op} differs from Fuel's registered recipe — it is not the same op; rename or adjudicate." Delivered via Spec B's `ProviderFeedback::on_rejected` (the KISS-#12 backchannel, already built). *Note:* today a non-elementwise submitted decompose (rope) can't be expressed as a `PatternNode` (emit limit), so the **structural** check is exercised on elementwise-expressible candidates now; non-elementwise structural identity arrives with the §8 convergence. Rope's gate in Increment 1 is the **numerical** path below.
- **Numerical (the gate):** candidate kernel output vs the §4.4 registered-recipe reference, at the op's declared tolerance (reuse Spec B's `verify_precision_bound` + the shipped total-order `ulp_distance`). This is where the interleaved rope is caught: it differs numerically from the rotate-half reference → rejected.

### 4.6 Novel ops (`claimed_op = None`)

The candidate's submitted decompose *defines* a new op: check it lowers cleanly to trusted primitives (`lower_to_base_map` reaches a fixpoint of only-primitive ops), verify the kernel numerically against that decompose, register a new `FusedOpId` (dedup via `by_pattern_hash` so an identical region doesn't mint a second id). This is the adaptive-fusion-loop path; Increment 1 wires the mechanism but its headline test is the known-op (rope) case.

### 4.7 Rope driver + tests

- Re-expose the reverted interleaved `rope_apply_f32` as a candidate `KernelRef` (test fixture) with rope-shaped probe operands (x, cos, sin via `build_rope_tables`).
- **Rejection (the real bug):** interleaved `rope_apply` candidate, `claimed_op = FusedOps::ROPE` → reference = Fuel's rotate-half recipe realized → numerical mismatch → `Rejected` with a precision claim. GPU, `#[ignore]`.
- **Adoption:** a candidate that genuinely computes rotate-half (or, if no rotate-half CUDA kernel exists, a CPU rope kernel realized as the candidate on a CPU probe) → `Adopted`. GPU/CPU as available.
- **Recipe-identity (structural, elementwise):** a candidate claiming a known elementwise fused op but submitting a *different* decompose → base-map mismatch → `Rejected("recipe-identity")`. No GPU (pure lowering + hash).
- **Comparator unit tests:** `base_map_hash` equal for commutative reorderings + different fusion depths of the same op; distinct for genuinely different ops; `Const` value-equality respected.

## Error handling / never-panic

Every new path is `Result`/`Option`; the verify body stays `catch_unwind`-wrapped (Spec B). `lower_to_base_map` never panics (G2 self-return). A recipe-lookup miss (`claimed_op` names an unregistered id) → an honest `Fail`/`Rejected`, never a crash. No embedded-ledger mutation; fresh in-memory ledger (Spec B discipline).

## Testing

Rope oracle (GPU `#[ignore]`, RTX 4070) + the elementwise recipe-identity + comparator unit tests (no GPU). TDD, born-red. Default build excludes the cuda paths (feature-gated as Spec B).

## Boundaries (explicitly NOT in Increment 1)

- **Growing `emit` to the full op set** (extend `OpAttrs` for Slice/Concat/Pad, real per-op shape inference) — that's §8.
- **Migrating the 24 registry decomposes** from Rust fns to data recipes — §8.
- **The KISS comparator-enum + MathPrecision re-spell + the edge-case corpus** — that's C2 (a separate KISS-aligned spec).
- **The 6 basis-gap ops** (conv2d/conv_transpose_2d/qmatmul/inplace_affine/selective_scan/ssd_chunk_scan) — blocked on IR-primitive work, orthogonal to this.

## The Tier-2 convergence (§8 — sequenced follow-on, "soon but not now")

Increment 1 proves recipe-identity on rope while leaving the two decompose mechanisms in place. The convergence retires the duplication and is the constitution's already-planned **Tier-2 declarative engine** (the JIT-loop prerequisite, 2026-06-20 decisions-log G4):
1. Grow `emit` into a real per-op interpreter — extend `OpAttrs` for the structurally-unrepresentable ops (Slice/Concat/Pad/indexing), replace the `operand[0].shape` shortcut with real per-op shape math (route through the existing `Tensor` builder shape logic, don't duplicate it).
2. Migrate the 10 trivial + 6 moderate registry decomposes from Rust fns to `PatternNode` **data** recipes, each validated **byte-for-byte** against the existing Rust decompose (the tested fn is the oracle for its own migration).
3. Unify internal + external ops on **one** declarative registry populated the same way; wire the 18 stubbed re-fusion matchers through the already-built `match_region`.
4. **North star (user, 2026-07-14):** a `PatternNode` is a graph fragment with bind-holes + match-wildcards — so a *single* expression should serve the recipe, the base-map form, **and** the Baracuda↔Fuel seam wire type. Converge the representations rather than translating between them. (The seam wire type must stay `fuel-graph`-dependency-free, so the shared expression lives in `fuel-kernel-seam-types` and fuel-graph consumes it — not the reverse.)

The 6 basis-gap ops stay opaque until their IR primitives (`Im2Col`/`Col2Im`, GGUF sub-byte unpack, `AffineInplace`, higher-order `Scan`) land — orthogonal to representation.

## Open questions / risks

- **`op_key` coverage:** covers 71/117 ops; recipes that emit currently-`None` ops (indexing, in-place) need `op_key` arms added for their base-map hash. Rope's base map (Reshape/BroadcastTo/Slice/Neg/Concat/Mul/Add) — confirm each is `op_key`-covered or add arms (Slice/Concat may need arms).
- **No rotate-half CUDA rope kernel** may exist for the *adoption* leg — fall back to a CPU rope candidate on a CPU probe, or assert the adoption leg on an elementwise op and keep rope for the rejection leg. Resolve during planning.
- **Const value-equality in the hash** must fold `ConstData` bytes (CSE deliberately excludes Const via Arc-identity) — needed so two independently-built recipes match on equal constants.

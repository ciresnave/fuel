# Recipe-Identity Verification + the Rope Oracle (Increment 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `verify_candidate` verify a candidate kernel against **Fuel's own registered recipe** for the op it claims (lowered to its base map + realized), not the candidate's self-supplied decompose — and reject a candidate whose recipe lowers to a different base map ("not the same op"). Rope drives it: the interleaved `rope_apply` claiming `ROPE` is rejected against Fuel's rotate-half reference.

**Architecture:** Reuse the existing, already-unified base-map lowering (`RuleRegistry::lowering_only().optimize_to_fixpoint`) behind a thin `lower_to_base_map` wrapper; add a recursive content-hash comparator (`base_map_hash`) by extending the existing `op_key` (reusing the existing commutative canonicalization); give `CandidateKernel` a `claimed_op` field; and in `verify_candidate` resolve + lower + realize Fuel's registered recipe as the reference. `PatternNode` stays the recipe type; no `emit` surgery (rope's reference is built by the existing `registry::rope::decompose`).

**Tech Stack:** Rust (edition 2024), `fuel-graph` (`opt.rs`, `registry.rs`, `registry/rope.rs`), `fuel-dispatch` (`jit_ingest.rs`, `jit_ingest_probe.rs`), `fuel-cuda-backend`, baracuda FFI. Reference doc: `docs/superpowers/specs/2026-07-14-recipe-identity-verification-and-rope-oracle-design.md`.

## Global Constraints

- **Build scoping:** always `cargo ... -p <crate>` (`fuel-graph`, `fuel-dispatch`), never workspace-wide. One cargo invocation at a time. Subagents doing GPU builds MUST run cargo in the FOREGROUND (do not use background jobs / do not wait on background-build notifications — a prior session's subagents deadlocked doing that).
- **CUDA builds** need a VS Developer shell; helper `C:\Windows\Temp\cuda_run.bat` (calls vcvars64, prepends CUDA v13.3 `bin` + cuDNN v9.23 to PATH, redirects to `C:\Windows\Temp\cuda_run.log`, appends `CUDA_RUN_EXITCODE`). Invoke FOREGROUND: `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit ...`; read `cuda_run.log` for the result + the `CUDA_RUN_EXITCODE` line (the batch's own exit is NOT cargo's). If missing, recreate per the Spec B plan's recipe.
- **Feature gating:** the fuel-graph pieces (`lower_to_base_map`, `base_map_hash`) are un-gated (fuel-graph has no jit/cuda features). `CandidateKernel.claimed_op` + the identity check are `#[cfg(feature="jit")]`; the registered-recipe reference + rope GPU tests are `#[cfg(feature="cuda")]`.
- **Never panic on production paths.** `lower_to_base_map` never panics (G2 self-return = clean fixpoint). Every new verify path is `Result`/`Option` inside Spec B's `catch_unwind` envelope. No `.unwrap()`/`.expect()` on production paths (test code may).
- **Recipe = `PatternNode`; do NOT swap to a `Graph` fragment** (it is the seam wire type). Do NOT grow `emit` (that is the deferred Tier-2 convergence). The rope reference is built by the existing Rust `registry::rope::decompose`, realized via the topology-agnostic realize path.
- **Ledger discipline:** fresh in-memory `VerificationLedger`; `upsert` never `push`; never mutate the embedded ledger.
- **Reuse, don't duplicate:** `is_commutative` (opt.rs:994), `op_key` (opt.rs:804), the realize-a-fragment pattern (jit_ingest_probe.rs `reference_output`), the ULP total-order `ulp_distance` + `verify_precision_bound` (Spec B). Extend these; do not re-copy.

---

## Prerequisites / starting context (read FIRST)

- **Branch.** Start from a fresh branch off `capturedrun-4b-resume` (currently at ~`f37958a5` — Spec B + the KISS pass + this spec are all there). Spec B's `jit_ingest.rs`/`jit_ingest_probe.rs` are the foundation.
- **Read the design doc** `docs/superpowers/specs/2026-07-14-recipe-identity-verification-and-rope-oracle-design.md` for the "why" and the architecture-map findings before Task 1.
- **Baseline check (before Task 1):** `cargo test -p fuel-graph --lib` green, `cargo test -p fuel-dispatch --lib` green, and `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo build -p fuel-dispatch --features cuda,jit` exits 0. Fix any red baseline first.
- **Key reuse map (verified by a 6-reader architecture study):**
  - Base-map lowering: `RuleRegistry::lowering_only()` (`fuel-graph/src/opt.rs:218`) + `optimize_to_fixpoint(graph, roots)` (`opt.rs:248`). Already unified over static-registry + runtime decomposes; recursive; side-effect-free.
  - Structural key: `op_key(op) -> Option<OpKey>` (`opt.rs:804-987`, covers 71/117 ops incl. `Op::Fused`); `is_commutative` (`opt.rs:994`); the CSE `optimize()` free fn (`opt.rs:998`) is the algorithm shape.
  - Dormant index: `FusedOpRegistry.by_pattern_hash: HashMap<PatternHash, FusedOpId>` (`registry.rs:734`), `PatternHash(u64)` (`registry.rs:844`) — reserved, no hash fn yet.
  - Recipe lookup: `default_registry().entry(id)` (`registry.rs:796`) → `FusedOpEntry.decompose` (`registry.rs:112`); `runtime_fused::runtime_region(id)` (`runtime_fused.rs:85`).
  - Realize a fragment: `reference_output` (`fuel-dispatch/src/jit_ingest_probe.rs:100-173`) + `PipelinedExecutor::realize` (`pipelined.rs:841`, topology-agnostic).
  - Rope: `registry::rope::decompose` (`fuel-graph/src/registry/rope.rs:83`, the 12-node rotate-half fragment); `build_rope_tables` (`fuel-graph/src/lib.rs:9937`); CPU `rope_f32` (`fuel-cpu-backend/src/byte_kernels.rs:1885`). The interleaved `rope_apply` is the reverted registration (`fuel-dispatch/src/baracuda_dispatch.rs:3029`); its `rope_apply_fused_<dt>_into` driver + `RopeTableCache` remain staged.
  - `verify_candidate` (`fuel-dispatch/src/jit_ingest.rs:181-461`); the reference is resolved at `jit_ingest.rs:369`; `CandidateKernel` fields at `jit_ingest.rs:38-48`.

---

## Task 1: `lower_to_base_map` wrapper (fuel-graph)

**Files:**
- Modify: `fuel-graph/src/opt.rs` (add the wrapper near `RuleRegistry::lowering_only`)
- Modify: `fuel-graph/src/lib.rs` (re-export if the crate re-exports opt items)

**Interfaces:**
- Consumes: `RuleRegistry::lowering_only()` (`opt.rs:218`), `RuleRegistry::optimize_to_fixpoint(&self, &SharedGraph, &[NodeId]) -> Vec<NodeId>` (`opt.rs:248`).
- Produces: `pub fn lower_to_base_map(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId>` — lowers every reachable fused op to its primitive base map (fixpoint), returning the (possibly remapped) roots. Never panics.

- [ ] **Step 1: Confirm the exact types.** Read `opt.rs:218-273` — confirm `SharedGraph` is the arg type of `optimize_to_fixpoint` (it is `Arc<RwLock<Graph>>`; check the alias name used in `opt.rs`) and that `lowering_only()` takes no args. Note the `RuleRegistry` import path.

- [ ] **Step 2: Write the failing test** in `fuel-graph/src/opt.rs`'s `#[cfg(test)] mod tests`:

```rust
#[test]
fn lower_to_base_map_dissolves_a_fused_op() {
    use crate::{Graph, Node, Op};
    use crate::registry::{FusedOps, FusedOpParams};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};
    // Build a graph: three Const leaves (x, cos, sin) + one Op::Fused(ROPE).
    let graph = Arc::new(RwLock::new(Graph::new()));
    let sink = {
        let mut g = graph.write().unwrap();
        let shape = Shape::from_dims(&[1, 4, 8]); // (batch, seq, head_dim=8, even)
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: shape.clone(), dtype: DType::F32 });
        let cshape = Shape::from_dims(&[4, 8]);
        let cos = g.push(Node { op: Op::Const, inputs: vec![], shape: cshape.clone(), dtype: DType::F32 });
        let sin = g.push(Node { op: Op::Const, inputs: vec![], shape: cshape, dtype: DType::F32 });
        g.push(Node {
            op: Op::Fused(FusedOps::ROPE, FusedOpParams::Rope),
            inputs: vec![x, cos, sin],
            shape,
            dtype: DType::F32,
        })
    };
    let roots = lower_to_base_map(&graph, &[sink]);
    let g = graph.read().unwrap();
    // No Op::Fused(ROPE) remains reachable; the base map is primitives.
    let has_rope = g.reachable_from(&roots).iter().any(|&n| matches!(g.node(n).op, Op::Fused(FusedOps::ROPE, _)));
    assert!(!has_rope, "ROPE must be lowered to its primitive base map");
}
```

Confirm `FusedOpParams::Rope`'s exact variant name and `Graph::reachable_from` (or the crate's equivalent reachable-node helper) while writing — adjust the assertion to the real reachable-set API. If none exists, iterate `g.nodes` and assert no `Op::Fused(ROPE,_)` remains at all.

- [ ] **Step 3: Run it, watch it fail.** Run: `cargo test -p fuel-graph --lib lower_to_base_map_dissolves -v`. Expected: FAIL (`lower_to_base_map` not defined).

- [ ] **Step 4: Implement the wrapper** in `opt.rs`:

```rust
/// Lower every reachable fused op in `graph` (rooted at `roots`) to its
/// primitive base map — the fixpoint of `decompose` over every node. A thin
/// named wrapper over [`RuleRegistry::lowering_only`] + `optimize_to_fixpoint`
/// (the machinery every fused op — static-registry AND runtime — already flows
/// through). Never panics: a self-returning `decompose` is a clean fixpoint
/// (the never-panic total-decompose contract), not a loop.
pub fn lower_to_base_map(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId> {
    RuleRegistry::lowering_only().optimize_to_fixpoint(graph, roots)
}
```

- [ ] **Step 5: Run it, watch it pass.** Run: `cargo test -p fuel-graph --lib lower_to_base_map -v`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add fuel-graph/src/opt.rs fuel-graph/src/lib.rs
git commit -m "feat(graph): lower_to_base_map — named wrapper over lowering_only fixpoint"
```

---

## Task 2: `base_map_hash` — cross-graph structural content hash (fuel-graph)

**Files:**
- Modify: `fuel-graph/src/opt.rs` (add `base_map_hash` + any missing `op_key` arms)

**Interfaces:**
- Consumes: `op_key(op) -> Option<OpKey>` (`opt.rs:804`), `is_commutative(op) -> bool` (`opt.rs:994`), the `Graph`/`Node` accessors.
- Produces: `pub fn base_map_hash(graph: &Graph, root: NodeId) -> u64` — a NodeId-independent content hash of the subgraph rooted at `root`: recursively hash `(op_key, sorted-if-commutative child hashes)`, folding `Const` bytes in. Two independently-built structurally-equal base maps (up to commutative reordering + fusion depth) hash equal.

- [ ] **Step 1: Read `op_key`** (`opt.rs:804-987`) — note which ops return `None` (in-place, indexing, `WriteSlice`, `Branch`, `Iota`, and — confirm — `Slice`/`Concat`). Note how `Op::Const` is excluded (`opt.rs:804-809,826`) and where const bytes live (`Op::Const`'s storage slot, `lib.rs:218-223`). Note `is_commutative`'s op list (Add/Mul/Maximum/Minimum, `opt.rs:994`).

- [ ] **Step 2: Write the failing tests** in `opt.rs` tests:

```rust
#[test]
fn base_map_hash_commutative_reorder_is_equal() {
    use crate::{Graph, Node, Op};
    use fuel_ir::{DType, Shape};
    let mk = |swap: bool| {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let ins = if swap { vec![b, a] } else { vec![a, b] };
        let sink = g.push(Node { op: Op::Add, inputs: ins, shape: s, dtype: DType::F32 });
        (g, sink)
    };
    let (g0, r0) = mk(false);
    let (g1, r1) = mk(true);
    assert_eq!(base_map_hash(&g0, r0), base_map_hash(&g1, r1), "a+b == b+a");
}

#[test]
fn base_map_hash_distinct_ops_differ() {
    use crate::{Graph, Node, Op};
    use fuel_ir::{DType, Shape};
    let mk = |op: Op| {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let sink = g.push(Node { op, inputs: vec![a, b], shape: s, dtype: DType::F32 });
        (g, sink)
    };
    let (ga, ra) = mk(Op::Add);
    let (gm, rm) = mk(Op::Mul);
    assert_ne!(base_map_hash(&ga, ra), base_map_hash(&gm, rm), "add != mul");
}
```

(If in Step 1 you found `Slice`/`Concat` return `None` from `op_key`, ALSO write a test that a rope-shaped base map containing `Slice`+`Concat` hashes deterministically — build a tiny `Slice`→`Concat` fragment and assert `base_map_hash` of two identical copies is equal; this forces you to add the `op_key` arms.)

- [ ] **Step 3: Run, watch fail.** `cargo test -p fuel-graph --lib base_map_hash -v`. Expected: FAIL (undefined).

- [ ] **Step 4: Implement `base_map_hash`** (+ any needed `op_key` arms). Sketch:

```rust
/// NodeId-independent content hash of the subgraph rooted at `root`. Folds each
/// child's hash (not its NodeId) into a `DefaultHasher`, canonicalizes
/// commutative-operand order (reusing [`is_commutative`]), and folds `Const`
/// bytes in (they are excluded from `op_key`/CSE by design, `opt.rs:804-826`).
/// Two independently-built base maps that are structurally equal up to
/// commutative reordering + fusion depth hash equal — a cheap structural
/// pre-filter for recipe identity (associativity/distributivity are NOT
/// canonicalized; the numeric verify covers that residual).
pub fn base_map_hash(graph: &Graph, root: NodeId) -> u64 {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    fn go(graph: &Graph, id: NodeId, memo: &mut HashMap<NodeId, u64>) -> u64 {
        if let Some(&h) = memo.get(&id) { return h; }
        let n = graph.node(id);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Op identity via op_key when available; fall back to a discriminant + shape/dtype.
        match op_key(&n.op) {
            Some(k) => k.hash(&mut hasher),
            None => {
                std::mem::discriminant(&n.op).hash(&mut hasher);
                // Const bytes: fold the actual constant data so equal constants match.
                if matches!(n.op, Op::Const) {
                    if let Some(bytes) = graph.const_bytes(id) { bytes.hash(&mut hasher); }
                }
                // Ensure shape/dtype participate for ops op_key can't encode.
                n.shape.dims().hash(&mut hasher);
                (n.dtype as u8).hash(&mut hasher);
            }
        }
        let mut child_hashes: Vec<u64> = n.inputs.iter().map(|&c| go(graph, c, memo)).collect();
        if is_commutative(&n.op) { child_hashes.sort_unstable(); }
        child_hashes.hash(&mut hasher);
        let h = hasher.finish();
        memo.insert(id, h);
        h
    }
    go(graph, root, &mut HashMap::new())
}
```

Confirm the real API for reading a `Const`'s bytes (the sketch's `graph.const_bytes(id)` is a placeholder — find how `Op::Const` data is stored, `lib.rs:~218-223` / the `storage_map`, and read it; if a `Const` has no host bytes at hash time — device-only — fold its `NodeId`-stable identity or shape/dtype and note the limitation in a comment). Add `op_key` arms for any base-map ops that returned `None` and that your recipes actually emit (at minimum whatever rope's base map uses that isn't covered — likely `Slice`/`Concat`), so the hash is total over the recipe's op set.

- [ ] **Step 5: Run, watch pass.** `cargo test -p fuel-graph --lib base_map_hash -v`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add fuel-graph/src/opt.rs
git commit -m "feat(graph): base_map_hash — content hash for cross-graph recipe identity"
```

---

## Task 3: `CandidateKernel.claimed_op` (fuel-dispatch)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (`CandidateKernel` struct + every construction site)

**Interfaces:**
- Produces: `CandidateKernel.claimed_op: Option<fuel_graph::registry::FusedOpId>` — the op-identity the candidate asserts it implements. `Some(id)` → verify against Fuel's registered recipe (Task 5); `None` → the Spec B behavior (verify against the candidate's own decompose) is retained.

- [ ] **Step 1: Write the failing test** in `jit_ingest.rs` tests:

```rust
#[test]
fn candidate_kernel_carries_claimed_op() {
    let c = CandidateKernel {
        entry_point: "k".into(),
        kernel: /* any existing test kernel fn-ptr, e.g. */ crate::baracuda_dispatch::binary::add_f32,
        op_params: crate::kernel::OpParams::None,
        decompose: None,
        operands: vec![],
        dtypes: vec![],
        kernel_revision_hash: 0,
        declared: crate::fused::PrecisionGuarantee::REFERENCE,
        backend: fuel_ir::probe::BackendId::Cuda,
        claimed_op: Some(fuel_graph::registry::FusedOps::ROPE),
    };
    assert_eq!(c.claimed_op, Some(fuel_graph::registry::FusedOps::ROPE));
}
```

Confirm `FusedOps::ROPE`'s path (`fuel_graph::registry::FusedOps::ROPE`) + that `FusedOpId`/`FusedOps` derive `PartialEq`/`Copy` (they do — used in `PartialEq` comparisons in the map). This test compiles under `--features jit`.

- [ ] **Step 2: Run, watch fail.** `cargo test -p fuel-dispatch --features jit --lib candidate_kernel_carries_claimed_op -v`. Expected: FAIL (no field `claimed_op`).

- [ ] **Step 3: Add the field** to `CandidateKernel` (jit_ingest.rs:38-48) — `pub claimed_op: Option<fuel_graph::registry::FusedOpId>,` with a doc comment — and add `claimed_op: None` to EVERY existing `CandidateKernel { .. }` construction site (the Spec B tests: `verify_candidate_add_f32_...`, `verify_candidate_refuses_...`, `ingest_rejects_mul_...`, `ingest_adopts_add_...`, the e2e fixtures `e2e_add_candidate`/`e2e_mul_candidate`, plus any doc examples). Grep `CandidateKernel {` to find them all.

- [ ] **Step 4: Run, watch pass** + no regression. `cargo test -p fuel-dispatch --features jit --lib -v` (the whole jit lib suite). Expected: PASS (all Spec B tests still green with `claimed_op: None`).

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(jit): CandidateKernel.claimed_op — the op-identity a candidate asserts"
```

---

## Task 4: Registered-recipe reference builder (fuel-dispatch, cuda)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest_probe.rs` (add `reference_from_registered_recipe`)

**Interfaces:**
- Consumes: `lower_to_base_map` (Task 1), `PipelinedExecutor::realize`, `reference_output`'s realize pattern (`jit_ingest_probe.rs:100-173`), `FusedOpParams` for the claimed op.
- Produces: `#[cfg(feature="cuda")] pub fn reference_from_registered_recipe(claimed_op: FusedOpId, params: &FusedOpParams, probe: &[HostTensor], out_dtype: DType, out_shape: Vec<usize>, device: &CudaDevice) -> Result<HostTensor>` — builds `Op::Fused(claimed_op, params)` on `Op::Const` probe leaves, `lower_to_base_map`s it, realizes the base map, D2H → `HostTensor`.

- [ ] **Step 1: Read `reference_output`** (`jit_ingest_probe.rs:100-173`) — the H2D (`CudaStorageBytes::from_cpu_bytes`), the fresh-Graph + `Op::Const` leaves + `StorageCache`, the `set_target_backend(Cuda)` stamping, `PipelinedExecutor::realize`, the D2H. You will mirror it, replacing the `emit_region` construction (line ~137) with "push `Op::Fused(claimed_op, params)` + `lower_to_base_map`".

- [ ] **Step 2: Write the failing GPU test** (`#[ignore]`): a rope reference for `FusedOps::ROPE` on rope-shaped F32 probes equals a hand-computed rotate-half rope (or the CPU `rope_f32` output).

```rust
#[test]
#[ignore = "requires a live CUDA device"]
fn reference_from_registered_recipe_realizes_rotate_half_rope() {
    let Ok(dev) = CudaDevice::new(0) else { return };
    // Rope-shaped probes: x [1,seq,head_dim], cos/sin [seq,head_dim] via build_rope_tables.
    // Build a small deterministic case; compute the expected rotate-half output on the host
    // with the SAME formula registry::rope::decompose encodes (Slice halves, Neg second,
    // Concat, x*cos + rotated*sin), and assert byte/near-equality.
    // (Construct the probes + expected via a helper; see Task 6 for the rope probe builder.)
    // ... assert bytes_to_f32(out.bytes) ~= expected_rotate_half ...
}
```

Keep this test minimal and deterministic; the exact rope-probe construction is shared with Task 6 — factor a `rope_probe(seq, head_dim, base)` test helper. Expected reference = the rotate-half formula (NOT interleaved).

- [ ] **Step 3: Run (GPU), watch fail.** `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit --lib reference_from_registered_recipe -- --ignored --nocapture` (FOREGROUND); read `cuda_run.log`. Expected: FAIL (fn undefined).

- [ ] **Step 4: Implement `reference_from_registered_recipe`.** Mirror `reference_output` but: push `Op::Fused(claimed_op, params)` as the sink on the `Op::Const` probe leaves; wrap the graph in `Arc<RwLock<>>`; `let roots = lower_to_base_map(&graph, &[fused_id]);` (the fused node dissolves to its primitive base map in place); stamp `set_target_backend(Cuda)` on every reachable non-`Const` node of the lowered base map (reuse `reference_output`'s stamping idiom — walk reachable nodes, skip `Op::Const`); build the `StorageCache` from the probe uploads keyed on the `Op::Const` ids; `PipelinedExecutor::realize(graph, roots[0], cache)`; D2H → `HostTensor`. `Result`; propagate errors; no unwrap on the production path.

- [ ] **Step 5: Run (GPU), watch pass.** Same command as Step 3. Expected: PASS, `CUDA_RUN_EXITCODE: 0`.

- [ ] **Step 6: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest_probe.rs
git commit -m "feat(jit): reference_from_registered_recipe — realize Fuel's registered recipe as the verify reference"
```

---

## Task 5: Wire the registered-recipe reference + recipe-identity gate into `verify_candidate` (fuel-dispatch)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (`verify_candidate_impl` reference-resolution block, ~jit_ingest.rs:368-400)

**Interfaces:**
- Consumes: `reference_from_registered_recipe` (Task 4), `base_map_hash` + `lower_to_base_map` (Tasks 1-2), `runtime_fused::runtime_region` / `default_registry().entry` for structural identity, the existing `check_numeric_bound`/`verify_precision_bound` path.
- Produces: `verify_candidate`, when `claimed_op = Some(id)`, uses the registered-recipe reference (Task 4) as the numeric reference AND — when the candidate also carries a `decompose` — a structural recipe-identity pre-check. A recipe-identity mismatch or a numeric mismatch → `Fail`.

- [ ] **Step 1: Write the failing test** — a structural recipe-identity unit test (no GPU): a candidate claiming a known op but carrying a *different* decompose → the identity check returns a mismatch. Factor the identity check into a small pure fn so it's testable without a device:

```rust
#[test]
fn recipe_identity_rejects_a_mismatched_submitted_decompose() {
    // A candidate claims FusedOps::<some elementwise op> but submits a decompose whose
    // base map differs from that op's registered recipe base map.
    // recipe_identity_matches(claimed_op, &submitted_decompose) == false.
    // Use a claimed op + a deliberately-wrong PatternNode (e.g. claim an Add-region op but
    // submit a Mul-region decompose) and assert the base maps differ.
    assert!(!recipe_identity_matches(/* claimed */, /* wrong submitted PatternNode */));
    assert!(recipe_identity_matches(/* claimed */, /* the correct submitted PatternNode */));
}
```

Design `fn recipe_identity_matches(claimed_op: FusedOpId, submitted: &PatternNode) -> bool` (jit-level, no cuda): resolve the registered recipe (`runtime_region(claimed_op)` or a fresh `Op::Fused(claimed_op)` node), lower BOTH the submitted decompose (emit into a fresh graph via `emit_region` — elementwise-only is fine here, this path only fires for elementwise-expressible submitted recipes) and the registered recipe to base maps, compare `base_map_hash`. Pick a real elementwise registered op for the test (confirm one exists whose recipe is an elementwise `PatternNode`; if none is registered as a runtime op in tests, register a tiny one via `register_runtime_fused` in the test).

- [ ] **Step 2: Run, watch fail.** `cargo test -p fuel-dispatch --features jit --lib recipe_identity -v`. Expected: FAIL (`recipe_identity_matches` undefined).

- [ ] **Step 3: Implement `recipe_identity_matches`** + wire the reference switch in `verify_candidate_impl`:
  1. `recipe_identity_matches` as designed (jit-level).
  2. In `verify_candidate_impl`'s reference block (~jit_ingest.rs:368): if `cand.claimed_op.is_some()`:
     - If `cand.decompose.is_some()`: run `recipe_identity_matches(claimed, &decompose)`; on `false` → `Fail { claim: "recipe_identity", detail: "submitted recipe's base map differs from Fuel's registered recipe for the claimed op — not the same op" }` (record + upsert). (This structural pre-check only fires for elementwise-expressible submitted decomposes; a non-elementwise claim with no submittable PatternNode decompose skips straight to the numeric reference.)
     - Set the numeric reference to `reference_from_registered_recipe(claimed, &params, &probe, out_dtype, out_shape, device)` instead of `reference_output(cand.decompose)`.
  3. If `cand.claimed_op.is_none()`: unchanged Spec B behavior (reference = `reference_output(cand.decompose)`), so all Spec B tests stay green.
  4. Keep everything inside the existing `catch_unwind`; never-panic.

Confirm the `FusedOpParams` you pass to `reference_from_registered_recipe` for a claimed op (e.g. `FusedOpParams::Rope`) — resolve it from the claimed op's family or carry it on the candidate if needed.

- [ ] **Step 4: Run, watch pass** + no Spec-B regression. `cargo test -p fuel-dispatch --features jit --lib -v`. Expected: PASS (identity test green; Spec B tests unaffected — they use `claimed_op: None`).

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(jit): verify against Fuel's registered recipe + recipe-identity gate (claimed_op)"
```

---

## Task 6: Rope oracle — GPU rejection + adoption legs (fuel-dispatch, cuda)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (test fixtures + GPU tests), possibly `fuel-dispatch/src/baracuda_dispatch.rs` (re-expose the interleaved `rope_apply` as a candidate `KernelRef`)

**Interfaces:**
- Consumes: everything above; the interleaved `rope_apply` wrapper (staged, `baracuda_dispatch.rs:3029` area).

- [ ] **Step 1: Re-expose the interleaved rope kernel as a candidate `KernelRef`.** Read `baracuda_dispatch.rs:3029-3039` + the staged `rope_apply_fused_<dt>_into` driver / `RopeTableCache`. Provide a `KernelRef`-shaped wrapper (`(ins, outs, layouts, params)`) that invokes baracuda's interleaved rope — mirror how `binary::add_f32` wraps a baracuda kernel. This is a TEST candidate (the whole point is that it's rejected). Confirm the baracuda symbol is available in the pinned crate (per the memory, `rope_apply` FFI exists; the driver is staged). If wiring a real invocation is heavy, a minimal wrapper that calls the staged driver on the probe is sufficient for the test. Keep it `#[cfg(feature="cuda")]`, test-only if possible.

- [ ] **Step 2: Write the failing GPU rejection test** (`#[ignore]`): the interleaved rope candidate, `claimed_op = FusedOps::ROPE`, rope-shaped F32 probe, `declared = PrecisionGuarantee::REFERENCE` → `verify_candidate` (or `ingest_one`) returns a `Fail`/`Rejected` on a precision claim (interleaved ≠ rotate-half).

```rust
#[test]
#[ignore = "requires a live CUDA device"]
fn rope_oracle_rejects_interleaved_against_rotate_half() {
    let Ok(dev) = CudaDevice::new(0) else { return };
    let cand = interleaved_rope_candidate(); // kernel = interleaved rope_apply wrapper, claimed_op = ROPE
    let (verdict, _records) = verify_candidate(&cand, &dev);
    match verdict {
        VerifyVerdict::Fail { claim, .. } => assert!(claim.contains("max") || claim == "recipe_identity"),
        VerifyVerdict::Pass => panic!("interleaved rope must NOT verify against Fuel's rotate-half recipe"),
    }
}
```

The `interleaved_rope_candidate()` helper builds the `CandidateKernel` with rope-shaped operands (x [1,seq,head_dim] + cos/sin [seq,head_dim]) and `claimed_op: Some(FusedOps::ROPE)`. Reuse the `rope_probe` helper from Task 4.

- [ ] **Step 3: Run (GPU), watch fail; implement; watch pass.** `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit --lib rope_oracle -- --ignored --nocapture` (FOREGROUND); read `cuda_run.log`. First FAIL (fixture/undefined), then implement the fixture, then PASS with `CUDA_RUN_EXITCODE: 0`.

- [ ] **Step 4: Adoption leg (resolve the open question).** Determine whether a *rotate-half* CUDA rope kernel exists as a `KernelRef` (grep `baracuda_dispatch` for a rotate-half rope wrapper). If YES: add `rope_oracle_adopts_rotate_half` (correct candidate → `Adopted`, GPU). If NO: add the adoption leg on the **CPU** rope kernel instead — candidate.kernel = `fuel_cpu_backend::byte_kernels::rope_f32` wrapped as a `KernelRef`, backend = Cpu, realized on a CPU probe via a CPU invoker (mirror the CpuInvoker path in `fkc/verify`), `claimed_op = ROPE` → `Adopted` (rotate-half CPU rope == rotate-half reference). Document which path you took in the test's doc comment. Keep the rejection leg (Step 2) as the headline.

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs fuel-dispatch/src/baracuda_dispatch.rs
git commit -m "feat(jit): rope oracle — interleaved rope_apply rejected vs Fuel's rotate-half recipe"
```

---

## Task 7: `by_pattern_hash` recipe-identity index + `register_runtime_fused` dedup (fuel-graph)

**Files:**
- Modify: `fuel-graph/src/registry.rs` (populate `by_pattern_hash` with `base_map_hash`), `fuel-graph/src/runtime_fused.rs` (dedup on register)

**Interfaces:**
- Consumes: `base_map_hash` (Task 2), the dormant `by_pattern_hash: HashMap<PatternHash, FusedOpId>` (`registry.rs:734`), `register_runtime_fused` (`runtime_fused.rs:60`).
- Produces: `register_runtime_fused` returns the EXISTING id for a structurally-identical (same base-map-hash) region instead of minting a duplicate.

- [ ] **Step 1: Write the failing test** in `runtime_fused.rs` tests:

```rust
#[test]
fn register_runtime_fused_dedups_structurally_identical_regions() {
    let id1 = register_runtime_fused("dedup::a", relu_add_region()).unwrap();
    let id2 = register_runtime_fused("dedup::b", relu_add_region()).unwrap(); // same region, different name
    assert_eq!(id1, id2, "an identical region must resolve to the same FusedOpId, not a duplicate");
}
```

(Reuse the existing `relu_add_region()` test helper.)

- [ ] **Step 2: Run, watch fail.** `cargo test -p fuel-graph --lib register_runtime_fused_dedups -v`. Expected: FAIL (two distinct ids today — `runtime_fused.rs:71-79` always allocates).

- [ ] **Step 3: Implement dedup.** In `register_runtime_fused`: lower the region to its base map (build a throwaway graph, `emit_region` the region on placeholder consts, `lower_to_base_map`), compute `base_map_hash`, look it up in `by_pattern_hash`; on hit return the existing `FusedOpId`; on miss, allocate as today AND insert `(hash -> new_id)`. Keep it never-panic (a region that can't be hashed/lowered → fall back to today's allocate-fresh, with a debug log). Confirm `by_pattern_hash` is reachable/mutable from `register_runtime_fused` (it's on `FusedOpRegistry`; thread access or add a helper).

- [ ] **Step 4: Run, watch pass** + no regression. `cargo test -p fuel-graph --lib runtime_fused -v` + `cargo test -p fuel-graph --lib -v`. Expected: PASS (dedup green; existing runtime_fused tests unaffected — distinct regions still get distinct ids).

- [ ] **Step 5: Commit.**

```bash
git add fuel-graph/src/registry.rs fuel-graph/src/runtime_fused.rs
git commit -m "feat(graph): by_pattern_hash recipe-identity index + register_runtime_fused dedup"
```

---

## Task 8: No-regression gate + module wiring

**Files:**
- Modify: as needed for re-exports (`fuel-graph/src/lib.rs`, `fuel-dispatch/src/lib.rs`)

- [ ] **Step 1: Re-export the new public items** if the crates' `lib.rs` re-export conventions require it (`lower_to_base_map`, `base_map_hash` from fuel-graph opt; `reference_from_registered_recipe` from jit_ingest_probe under `#[cfg(feature="cuda")]`). Follow the existing re-export style.

- [ ] **Step 2: Full default-build regression.** Run: `cargo test -p fuel-graph --lib` and `cargo test -p fuel-dispatch --lib` (default, no jit/cuda). Expected: both green (new fuel-graph fns are un-gated but pure; new fuel-dispatch code is feature-gated out).

- [ ] **Step 3: Full jit + cuda regression (FOREGROUND).** Run: `cargo test -p fuel-dispatch --features jit --lib`, then `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit --lib` (the non-ignored suite) — confirm 0 failures + all Spec B tests still green. Then run the new `#[ignore]` GPU tests once: `... --lib "rope_oracle" -- --ignored --nocapture` and `... reference_from_registered_recipe -- --ignored`. All green, `CUDA_RUN_EXITCODE: 0`.

- [ ] **Step 4: Commit.**

```bash
git add -A
git commit -m "feat(jit): recipe-identity verification + rope oracle — exports + regression gate"
```

---

## Self-review notes (coverage against the spec)

- Spec §4.1 `claimed_op` → Task 3. §4.2 `lower_to_base_map` → Task 1. §4.3 `base_map_hash` + `by_pattern_hash` → Tasks 2 + 7. §4.4 registered-recipe reference → Task 4 + wired in Task 5. §4.5 recipe-identity gate (structural + numeric) → Task 5. §4.6 novel ops → deliberately minimal (claimed_op:None = Spec B behavior; full novel-op registration is convergence work, noted not built). §4.7 rope driver → Task 6. Never-panic + ledger discipline → constraints threaded through Tasks 4-6. Boundaries (no emit growth, no registry migration, no C2 comparator schema) → honored; PatternNode kept; rope reference via registry decompose.
- **Carried open questions the plan resolves in-task:** `op_key` coverage for Slice/Concat (Task 2 Step 4 — add arms as needed); no-rotate-half-CUDA-rope for the adoption leg (Task 6 Step 4 — CPU-rope fallback); `Const`-byte equality in the hash (Task 2 Step 4 — read real const storage, fall back + document if device-only).
- **Confirm-against-real-code notes** (internal signatures I mapped but the implementer must verify at write time): `SharedGraph` alias + `optimize_to_fixpoint` arg types (Task 1); `Graph` reachable-set + const-bytes accessors (Tasks 1-2); `FusedOpParams::Rope` variant name (Tasks 1,4); the `op_key`/`OpKey` exact shape (Task 2); the `reference_output` stamping/StorageCache idiom (Task 4); the interleaved `rope_apply` FFI/driver availability (Task 6).

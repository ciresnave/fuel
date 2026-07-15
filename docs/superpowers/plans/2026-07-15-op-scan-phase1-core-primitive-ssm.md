# Op::Scan Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Fuel's first sub-graph-carrying primitive — `Op::Scan` (a JAX `lax.scan`-shaped bounded recurrence) plus its inert body-hole leaf `Op::ScanPlaceholder` — and re-decompose `selective_scan` + `ssd_chunk_scan` onto it, **closing G3**. After Phase 1, every fused op's `decompose` is total over genuine primitives (no scan caveat): the two SSM ops stop self-returning and instead lower to an `Op::Scan` terminal whose body is the affine SSM step. The payoff is optimizer-basis closure + recipe-identity verification, **not** new runtime capability — the existing fused SSM CPU/CUDA kernels stay the executed path.

**Architecture:** `Op::Scan` follows `scan(body, init_carry, xs, consts) -> (final_carry, stacked_ys)`. The **body lives in the node's own `inputs`** (the `Op::Branch` arms-as-inputs precedent): node layout is `inputs = [ init_carry, xs_0..xs_{n_xs-1}, consts..., body_new_carry, body_y ]` — the **two body-exit NodeIds are the last two inputs**. This makes `base_map_hash` (recurses only through `Node::inputs`), the lowering walker `topo_order_multi` (walks only `inputs`), and reachability all see the body for free. Body holes are `Op::ScanPlaceholder { role, index }`: the body references `ScanPlaceholder{Carry,0}` for the per-step carry and `ScanPlaceholder{Elem,i}` for the per-step slice of `xs[i]`; `consts` are referenced by their real NodeId. `Op::Scan` is an **optimization/verification-time node** — it never reaches the executor un-lowered (the SSM re-decompose emits it but the executor dispatches the *fused* op directly to its kernel; `unroll_scan` materializes real primitives on demand for the verify oracle / kernel-absent fallback). So Phase 1 needs **no CPU/CUDA `Op::Scan` kernel** and the placeholder body is never directly realized.

**Tech Stack:** Rust (edition 2024). Primarily `fuel-graph` (`lib.rs` for the `Op` enum + `Tensor::scan` builder, `opt.rs` for `op_key`, new `scan.rs` for `unroll_scan`, `registry/selective_scan.rs` + `registry/ssd_chunk_scan.rs` for the re-decompose). Gap tests live in `fuel-core` (`src/lazy.rs`). The non-regression gate touches `fuel-core`/`fuel-dispatch`. Constitution diff in `docs/architecture/`. Reference doc: `docs/superpowers/specs/2026-07-15-op-scan-phase1-core-primitive-ssm-design.md`.

## Global Constraints

- **Build scoping:** always `cargo ... -p <crate>` (`fuel-graph`, `fuel-core`, `fuel-dispatch`), **never workspace-wide** (`tensor-tools` has a standing `Device::Cpu` break as a default member, so bare `cargo check` at the root fails). **ONE cargo invocation at a time** (the build-dir lock serializes; parallel invocations thrash). Long builds: background + wait.
- **CUDA is NOT required for Phase 1 correctness.** Every gate in this plan is CPU-only. Any GPU step (only the *optional* Mamba decode bench in Task 8) is flagged optional/local-only — skip it if no live device.
- **TDD, born-red.** For each task: write the failing test FIRST, run it and observe RED (with the exact command + expected failure below), then implement, then observe GREEN. A "born-red" run that fails for the *stated* reason is the evidence the trap is real — do not skip it.
- **Never panic on production paths.** The builder (Task 3), the shape/dtype rule (Task 3), and `unroll_scan` (Task 4) are all `Result`-returning; `early_exit = Some` on any realize/lowering path → a clear `Err`, never a panic. No new `.unwrap()`/`.expect()` on production paths (test code may). The one sanctioned panic is the backward-walk `Op::Scan` arm (Task 1), which mirrors the existing `QMatMul`/`PagedAttn` "not differentiable" precedent in that same infallible `-> GradMap` walk.
- **Validate at graph-build time.** The builder validates body/carry/bound when the node is pushed (the `cumsum`/`triu`/`finalize_branches` discipline).
- **Docs are part of the change.** Task 9 ships the constitution diff (one `03-ir` MAJOR + four MINORs + a decisions-log entry) in the same branch.
- **The two new `Op` variants land under ONE `03-ir` MAJOR bump.** `Op::Scan` (the primitive) + `Op::ScanPlaceholder` (its body hole) together are the "first sub-graph-carrying primitive" claim change. Define the **full** `Op::Scan` shape now (all four fields, incl. `early_exit`) so there is exactly one MAJOR bump and no later re-bump; implement only what the SSM consumer needs.
- **The lax.scan encoding invariants (do not deviate):** body-exits (`body_new_carry`, `body_y`) are the **last two `inputs`** of the `Op::Scan` node; `consts` are referenced by real NodeId; carry is a **single** tensor in v1 (multi-tensor carry is a documented deferral); `emit ∈ {All, Final}`; `Op::Scan` **always** produces the 2-slot bundle (slot 0 = stacked `ys`, slot 1 = final carry); `Op::Scan` gets **no `LoweringRule`** (it is a bare `Op` variant, so nothing matches it → it stays a terminal in the base map — this is correct, it IS the primitive).

## Prerequisites / starting context (read FIRST)

- **Branch.** Start a fresh branch `op-scan-phase1` off `capturedrun-4b-resume`. This plan **depends on** `base_map_hash` (`fuel-graph/src/opt.rs:368`) and `lower_to_base_map` (`opt.rs:358`), both of which landed with Increment 1 on `capturedrun-4b-resume` (@ `afc6ff32`) — they do not exist on `main`.
- **Read the design doc** `docs/superpowers/specs/2026-07-15-op-scan-phase1-core-primitive-ssm-design.md` (components C1–C10, boundaries) before Task 1.
- **Baseline check (before Task 1):** `cargo test -p fuel-graph --lib` green and `cargo test -p fuel-core --lib` green. Fix any red baseline first.
- **Deliberate divergence from the design's C5.** The design spec C5 proposed hand-threading a `Scan` `LoweringRule`. This plan does **NOT** add one (see Global Constraints): `Op::Scan` is a bare variant, no rule matches it, and it correctly stays terminal — `unroll_scan` (Task 4) is a standalone utility, not registered as anyone's `.decompose`. This is the locked design; Task 4/5 test the terminal posture directly.
- **Key file:line map (verify at write-time; drift noted where the design spec was off):**
  - `Op` enum: `fuel-graph/src/lib.rs:216` (`#[derive(Debug, Clone, PartialEq)]` — **no** `Hash`/`Eq`); last variant `Branch { reconverge_at: NodeId }` at `lib.rs:1112`, enum closes at `lib.rs:1113`.
  - `op_short_name`: `lib.rs:1222–1348` (exhaustive, no wildcard → forces a compile edit).
  - `derive_view_output_layout`: `lib.rs:1194` (guarded-total with an `other => Err(..)` catch-all → **no** forced edit; new variants correctly return `Err`).
  - Backward walk: `dispatch_gradient` early-dispatch at `lib.rs:7129` (non-exhaustive, `_ => None`); the legacy exhaustive `match op {` at `lib.rs:7145` (no wildcard → forces a compile edit; `QMatMul`/`PagedAttn` panic precedent at `lib.rs:9386`, inert-drop precedent `Op::Branch` at `lib.rs:9419`).
  - `op_key`: `fuel-graph/src/opt.rs:955` (`fn op_key(op: &Op) -> Option<OpKey>`, **no `&Graph`**, ends `_ => None`); `OpKey { tag: u16, ints: Vec<i64>, bits: Vec<u64>, dims: Vec<usize>, shape: Option<Vec<usize>>, dtype: Option<u32> }`; `is_commutative` (`opt.rs:1124`) = `Add|Mul|Maximum|Minimum` only.
  - `base_map_hash`: `opt.rs:368–486` (recurses only through `Node::inputs`; `Some(op_key)` branch hashes the key, `None` branch hashes discriminant+shape+dtype+const-bytes). **No edit needed** — the new `op_key` arms make it correct via existing recursion.
  - Builder precedents: `Tensor::cumsum` `lib.rs:5033`, `Tensor::triu` `lib.rs:5054` (both `Result`, validate-then-push); `selective_scan_producer` `lib.rs:6052` and `selective_scan_bundled` `lib.rs:6183` (2-slot bundle via `output_views` → `fuel_ir::storage::compose_bundle` → `graph.set_output_views` → `Tensor::view(slot)`, but **panicking** internals — convert every `.expect()`/`assert!` to a typed `Err`).
  - `Node { op, inputs, shape, dtype }` `lib.rs:1374`; `OutputViewSpec { dtype, shape, layout, name: Option<&'static str> }` (`fuel-ir/src/storage.rs`, `::contiguous(dtype, shape)` helper).
  - SSM: `selective_scan::decompose` `fuel-graph/src/registry/selective_scan.rs:210` (self-return `-> NodeId { id }`); its `entry()`/`shape_rule`/`output_views`/`dtype_rule` unchanged; `ssd_chunk_scan::decompose` `registry/ssd_chunk_scan.rs:169` (self-return); stale "decompose panics" doc `ssd_chunk_scan.rs:53`. CPU math: `fuel-cpu-backend/src/byte_kernels.rs:6104` (SelectiveScan recurrence), `:6213` (F64-accumulate loop), `:6328`/`:6478` (SsdChunkScan). `FusedOpParams::SelectiveScan { delta_softplus: bool }` (5 inputs u/delta/a/b/c — `d_skip`/`z`/`delta_bias` are NOT live); `FusedOpParams::SsdChunkScan { chunk_size: usize }` (5 inputs x/dt/a/b/c, chunk_size is a CPU no-op).
  - Gap tests: `fuel-core/src/lazy.rs:2069` (`selective_scan_decompose_is_surfaced_gap_not_a_crash`, the one to flip); `lazy.rs:2124` (`nf4_matmul_decompose_matches_kernel`, the positive-test template). `crate::pipelined_bridge::realize_one_as::<f32>(&graph, root, &dev)`.
  - Reachability (C6, **no edit** — confirmed): `Graph::live_set` `lib.rs:2169` and `effective_roots` `fuel-graph/src/run.rs:208` — both key on `Op::Branch { reconverge_at }`; `Op::Scan` needs **no** extension because its body-exits are ordinary `inputs` and the `Op::Scan` node itself is read by downstream consumers (unlike `Op::Branch`, which is orphaned after finalization).
  - `variant_bake` is at `fuel-dispatch/src/variant_bake.rs` (design spec's `fuel-graph/...` cite is wrong).

---

## Task 1: `Op::Scan` + `Op::ScanPlaceholder` variants + support types + forcing matches (C1)

**Files:**
- Modify: `fuel-graph/src/lib.rs` (add types after the `Op` enum ~`lib.rs:1113`; add variants before the enum's closing `}` at `lib.rs:1113`; `op_short_name` arms ~`lib.rs:1347`; backward-walk arms ~`lib.rs:9431`).

**Interfaces:**
- Produces: `pub enum ScanEmit { All, Final }`, `pub enum ScanRole { Carry, Elem }`, `pub struct ScanPredicate` (a Phase-1 placeholder; never `Some` on a live path but constructible so the guard is testable), and two `Op` variants:
  - `Op::Scan { n_xs: usize, bound: usize, emit: ScanEmit, early_exit: Option<ScanPredicate> }`
  - `Op::ScanPlaceholder { role: ScanRole, index: usize }`

- [ ] **Step 1: Write the failing test** in `fuel-graph/src/lib.rs`'s `#[cfg(test)] mod tests` (grep `mod tests` in `lib.rs` to find it):

```rust
#[test]
fn scan_variants_construct_and_name() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole, ScanPredicate};
    use fuel_ir::{DType, Shape};
    let mut g = Graph::new();
    let s = Shape::from_dims(&[1, 1, 1]);
    let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
    // A ScanPlaceholder leaf (the per-step carry hole).
    let hole = g.push(Node {
        op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 },
        inputs: vec![],
        shape: s.clone(),
        dtype: DType::F32,
    });
    // A trivial body: new_carry = hole (identity); body_y = hole.
    // Op::Scan node: inputs = [init_carry, <no xs>, <no consts>, body_new_carry, body_y].
    let scan = g.push(Node {
        op: Op::Scan { n_xs: 0, bound: 1, emit: ScanEmit::All, early_exit: None },
        inputs: vec![carry, hole, hole],
        shape: s.clone(),
        dtype: DType::F32,
    });
    assert_eq!(g.node(scan).op.short_name(), "Scan");
    assert_eq!(g.node(hole).op.short_name(), "ScanPlaceholder");
    // early_exit is constructible (guarded elsewhere) but never Some on a live path.
    let _guarded = Op::Scan { n_xs: 0, bound: 1, emit: ScanEmit::Final, early_exit: Some(ScanPredicate) };
    // derive_view_output_layout: Scan is NOT a view op -> Err (handled by the catch-all).
    let lay = fuel_ir::Layout::contiguous(s.clone());
    let derived = crate::derive_view_output_layout(
        &Op::Scan { n_xs: 0, bound: 1, emit: ScanEmit::All, early_exit: None },
        &lay,
    );
    assert!(derived.is_err(), "Op::Scan must not be treated as a view op");
    let _ = scan;
}
```

Confirm `fuel_ir::Layout::contiguous(shape)` (used by `selective_scan::output_views`) and `crate::derive_view_output_layout` at write-time. The `derive_view_output_layout` assertion confirms the existing `other => Err(..)` catch-all already handles the new variants (no edit needed there).

- [ ] **Step 2: Run it, watch it fail (compile error).** Run: `cargo test -p fuel-graph --lib scan_variants_construct_and_name -- --exact`. Expected: FAIL — does not compile (`ScanEmit`/`ScanRole`/`ScanPredicate`/`Op::Scan`/`Op::ScanPlaceholder` are undefined; `op_short_name` and the backward `match op` are non-exhaustive).

- [ ] **Step 3: Add the support types** immediately after the `Op` enum's closing `}` (after `lib.rs:1113`):

```rust
/// What a [`Op::Scan`] emits. `All` stacks the per-step `body_y` over the
/// bound (slot 0); `Final` returns only the final carry (slot 1). The op
/// always produces the 2-slot bundle; `emit` selects the builder's default
/// projection. (`Op::Reduce` is `Op::Scan { emit: Final }` conceptually —
/// no separate variant.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanEmit {
    All,
    Final,
}

/// Which body hole a [`Op::ScanPlaceholder`] stands in for. `Carry` is the
/// per-step recurrent carry (always index 0 in v1's single-carry model);
/// `Elem` is the per-step slice of `xs[index]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanRole {
    Carry,
    Elem,
}

/// Phase-1 placeholder for [`Op::Scan`]'s `early_exit` field. The FIELD is
/// defined now so the enum shape is final (one `03-ir` MAJOR bump); the
/// data-dependent early-exit MECHANISM is Phase 2. It is never `Some` on any
/// live Phase-1 path; if a realize/lowering path sees `Some`, that path
/// returns a clear `Err` (a surfaced Phase-2 gap), never a panic.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanPredicate;
```

- [ ] **Step 4: Add the two `Op` variants** just before the enum's closing `}` at `lib.rs:1113` (after `Branch { reconverge_at: NodeId },`):

```rust
    /// Fuel's first sub-graph-carrying primitive: a bounded `lax.scan`-shaped
    /// recurrence `scan(body, init_carry, xs, consts) -> (final_carry, ys)`.
    ///
    /// The body is encoded in this node's own `inputs`, exactly like
    /// [`Op::Branch`]'s arms. Node layout:
    /// `inputs = [ init_carry, xs_0..xs_{n_xs-1}, consts..., body_new_carry, body_y ]`
    /// — the two body-exit NodeIds are the LAST two inputs, so `base_map_hash`,
    /// `topo_order_multi`, and reachability see the body for free. The body
    /// references `Op::ScanPlaceholder{Carry,0}` for the per-step carry and
    /// `Op::ScanPlaceholder{Elem,i}` for the per-step slice of `xs[i]`; consts
    /// are referenced by real NodeId. Single carry tensor in v1.
    ///
    /// An optimization/verification-time node: it never reaches the executor
    /// un-lowered (the SSM decompose emits it, but the executor dispatches the
    /// fused op to its kernel; `unroll_scan` materializes primitives on demand
    /// for the verify oracle / kernel-absent fallback). No `LoweringRule`
    /// matches it — it is a terminal in the base map, which is correct: it IS
    /// the primitive. Always a 2-slot bundle (slot 0 = stacked `ys`, slot 1 =
    /// final carry). `early_exit` is Phase-2 (see [`ScanPredicate`]).
    Scan {
        n_xs: usize,
        bound: usize,
        emit: ScanEmit,
        early_exit: Option<ScanPredicate>,
    },

    /// Inert body-hole leaf for [`Op::Scan`] (no inputs). Substituted by
    /// `unroll_scan` per step. Never directly realized.
    ScanPlaceholder {
        role: ScanRole,
        index: usize,
    },
```

- [ ] **Step 5: Add the `op_short_name` arms** just before the closing `}` of the match at `lib.rs:1347` (after the `Op::Branch{..}` arm):

```rust
        Op::Scan{..}             => "Scan",
        Op::ScanPlaceholder{..}  => "ScanPlaceholder",
```

- [ ] **Step 6: Add the backward-walk arms** in the exhaustive `match op {` (`lib.rs:7145`), alongside the other top-level arms (e.g. after the `Op::Branch { .. }` arm at `lib.rs:9419`). `Op::Scan` is not differentiable in Phase 1 (BPTT is Phase 2) → mirror the `QMatMul`/`PagedAttn` clean-panic precedent; `Op::ScanPlaceholder` is an inert body leaf → drop gradient like `Op::Const`/`Op::Iota`:

```rust
                Op::Scan { .. } => {
                    // Not differentiable in Phase 1. BPTT (differentiate the
                    // unrolled body) lands in Phase 2; mirrors the QMatMul /
                    // PagedAttn "no backward" precedent in this infallible
                    // (-> GradMap) walk. A differentiated graph that reaches an
                    // Op::Scan is a usage error until Phase 2 wires it.
                    panic!(
                        "Tensor::backward: Op::Scan is not differentiable in \
                         Phase 1. BPTT over the unrolled body lands in Phase 2.",
                    );
                }
                Op::ScanPlaceholder { .. } => {
                    // Inert body-hole leaf (no inputs) — never a real
                    // differentiation target. Drop gradient like Op::Const /
                    // Op::Iota; a live backward never reaches it (the body is
                    // materialized by unroll_scan before any realize).
                }
```

- [ ] **Step 7: Compiler sweep for any OTHER exhaustive match.** Run `cargo build -p fuel-graph`. The two known forcing sites are handled above; if the compiler flags any *other* non-exhaustive `match` over `Op` (there are none known in `lib.rs` beyond these two — `derive_view_output_layout`/`op_key`/`infer_storage_class`/`op_to_tag`/`try_simplify` all have catch-alls), add a **conservative** arm mirroring the nearest inert/structural precedent: `Op::Branch`-style drop for gradient/walk sites, `_ => None`/`Transient`/`false` semantics for classifier sites. Do not invent behavior — these are self-revealing and each has an obvious neighbor.

- [ ] **Step 8: Run it, watch it pass.** Run: `cargo test -p fuel-graph --lib scan_variants_construct_and_name -- --exact`. Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add fuel-graph/src/lib.rs
git commit -m "feat(ir): Op::Scan + Op::ScanPlaceholder variants + ScanEmit/ScanRole/ScanPredicate (C1)"
```

---

## Task 2: `op_key` arms — structural body-hash correctness (C2, THE linchpin)

**Files:**
- Modify: `fuel-graph/src/opt.rs` (`op_key` match, before `_ => return None` at ~`opt.rs:1113`; tests in `opt.rs`'s `#[cfg(test)] mod tests`).

**Interfaces:**
- Consumes: `op_key(op: &Op) -> Option<OpKey>` (`opt.rs:955`, NO `&Graph`), `base_map_hash(graph: &Graph, root: NodeId) -> u64` (`opt.rs:368`, unchanged), `is_commutative` (`opt.rs:1124`, `Op::Scan`/`Op::ScanPlaceholder` are correctly NOT commutative → input order preserved).
- Produces: `op_key` now returns `Some` for `Op::Scan` (folding **only its own params** `n_xs`, `bound`, `emit` tag, `early_exit.is_some()` flag) and `Op::ScanPlaceholder` (folding `role` tag + `index`). The body content is hashed automatically by `base_map_hash`'s existing child-recursion because the body-exits are in `inputs`.

- [ ] **Step 1: Read `op_key`** (`opt.rs:955–1116`) — note the `Op::Fused` arm (tag 200, folds `id`/`params.tag`/`params.ints` into `ints`) and the terminal `_ => return None`. Confirm the highest existing `tag` (Fused=200) so 210/211 are free. Confirm `base_map_hash` (`opt.rs:368`): the `Some(k) => k.hash(..)` branch, then `n.inputs.iter().map(|&c| go(..))` recursion, then `if is_commutative(&n.op) { child_hashes.sort_unstable(); }`.

- [ ] **Step 2: Write the failing tests** in `opt.rs`'s test module (grep `fn base_map_hash` / `mod tests` to place them near the existing `base_map_hash` tests):

```rust
#[test]
fn scan_base_map_hash_differs_for_different_bodies() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole};
    use fuel_ir::{DType, Shape};
    // Two scans, IDENTICAL carry shape / n_xs / bound / emit, but STRUCTURALLY
    // DIFFERENT bodies: body A = carry*2 (MulScalar), body B = carry+1
    // (AddScalar). Without a dedicated op_key arm they fall to the None branch
    // (discriminant+shape+dtype only) and collide -> silent CSE / recipe-identity
    // corruption. THIS red run is the evidence the trap is real.
    let mk = |mul: bool| {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[1]);
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let new_carry = g.push(Node {
            op: if mul { Op::MulScalar(2.0) } else { Op::AddScalar(1.0) },
            inputs: vec![hole], shape: s.clone(), dtype: DType::F32,
        });
        // inputs = [init_carry, body_new_carry, body_y]  (n_xs = 0, no consts)
        let scan = g.push(Node {
            op: Op::Scan { n_xs: 0, bound: 3, emit: ScanEmit::All, early_exit: None },
            inputs: vec![carry, new_carry, new_carry],
            shape: s, dtype: DType::F32,
        });
        (g, scan)
    };
    let (ga, ra) = mk(true);
    let (gb, rb) = mk(false);
    assert_ne!(
        base_map_hash(&ga, ra), base_map_hash(&gb, rb),
        "scans with structurally different bodies must hash differently",
    );
}

#[test]
fn scan_base_map_hash_equal_for_same_body_cross_graph() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole};
    use fuel_ir::{DType, Shape};
    // Same body built independently in two graphs -> equal hash (the
    // recipe-identity positive case). base_map_hash is NodeId-independent.
    let mk = || {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[1]);
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let new_carry = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
        let scan = g.push(Node {
            op: Op::Scan { n_xs: 0, bound: 3, emit: ScanEmit::All, early_exit: None },
            inputs: vec![carry, new_carry, new_carry], shape: s, dtype: DType::F32,
        });
        (g, scan)
    };
    let (g0, r0) = mk();
    let (g1, r1) = mk();
    assert_eq!(base_map_hash(&g0, r0), base_map_hash(&g1, r1), "same body -> equal hash");
}

#[test]
fn scan_emit_and_placeholder_role_participate_in_op_key() {
    use crate::{Op, ScanEmit, ScanRole};
    // emit=All vs emit=Final differ.
    let all = op_key(&Op::Scan { n_xs: 0, bound: 3, emit: ScanEmit::All, early_exit: None });
    let fin = op_key(&Op::Scan { n_xs: 0, bound: 3, emit: ScanEmit::Final, early_exit: None });
    assert!(all.is_some() && fin.is_some(), "Op::Scan must produce an op_key, not None");
    assert_ne!(all, fin, "emit=All vs emit=Final must differ in op_key");
    // n_xs / bound participate.
    assert_ne!(op_key(&Op::Scan { n_xs: 1, bound: 3, emit: ScanEmit::All, early_exit: None }), all);
    assert_ne!(op_key(&Op::Scan { n_xs: 0, bound: 4, emit: ScanEmit::All, early_exit: None }), all);
    // ScanPlaceholder Carry/0 vs Elem/0 vs Carry/1 all differ.
    let c0 = op_key(&Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 });
    let e0 = op_key(&Op::ScanPlaceholder { role: ScanRole::Elem, index: 0 });
    let c1 = op_key(&Op::ScanPlaceholder { role: ScanRole::Carry, index: 1 });
    assert!(c0.is_some());
    assert_ne!(c0, e0, "Carry vs Elem must differ");
    assert_ne!(c0, c1, "index must participate");
}
```

- [ ] **Step 3: Run, watch fail.** Run: `cargo test -p fuel-graph --lib scan_base_map_hash_differs_for_different_bodies scan_base_map_hash_equal_for_same_body_cross_graph scan_emit_and_placeholder_role_participate_in_op_key -- --exact`. Expected: `scan_emit_and_placeholder_role_participate_in_op_key` FAILS (`op_key` returns `None`); `scan_base_map_hash_differs_for_different_bodies` may PASS or FAIL depending on whether the differing `MulScalar`/`AddScalar` body nodes already diverge the hash via their own op_keys — but the `op_key`-`None`-for-Scan assertion (`all.is_some()`) is the definitive red. (If `differs` already passes, that is fine — the `op_key` arm is still required by the `is_some()`/emit/placeholder assertions; the `None` branch would collide two scans whose bodies differ *only* below a still-`None` op.)

- [ ] **Step 4: Add the `op_key` arms** immediately before `_ => return None` in `op_key` (`opt.rs:~1113`):

```rust
        // Op::Scan folds ONLY its own params (n_xs, bound, emit tag,
        // early_exit-present flag). The BODY is hashed automatically by
        // base_map_hash's child-recursion, because the body-exit NodeIds are
        // the last two entries of the node's `inputs` (the lax.scan encoding).
        // Two scans with identical params but different bodies therefore get
        // different child-hashes and different base_map_hash. Tag 210.
        Op::Scan { n_xs, bound, emit, early_exit } => {
            let emit_tag: i64 = match emit { ScanEmit::All => 0, ScanEmit::Final => 1 };
            let exit_flag: i64 = if early_exit.is_some() { 1 } else { 0 };
            (210, vec![*n_xs as i64, *bound as i64, emit_tag, exit_flag], vec![], vec![], None, None)
        }
        // Op::ScanPlaceholder folds role tag + index so Carry/0, Elem/0,
        // Carry/1 are all distinct in the body hash. Tag 211.
        Op::ScanPlaceholder { role, index } => {
            let role_tag: i64 = match role { ScanRole::Carry => 0, ScanRole::Elem => 1 };
            (211, vec![role_tag, *index as i64], vec![], vec![], None, None)
        }
```

Add `use crate::{ScanEmit, ScanRole};` to the match's scope if `opt.rs` doesn't already glob-import them (grep the top of `opt.rs` for `use crate::`; the enum names may already be in scope via `crate::Op` — if the arms don't resolve, add the import).

- [ ] **Step 5: Run, watch pass.** Run: `cargo test -p fuel-graph --lib scan_base_map_hash_differs_for_different_bodies scan_base_map_hash_equal_for_same_body_cross_graph scan_emit_and_placeholder_role_participate_in_op_key -- --exact`. Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add fuel-graph/src/opt.rs
git commit -m "feat(graph): op_key arms for Op::Scan/Op::ScanPlaceholder — structural body-hash via inputs (C2)"
```

---

## Task 3: `Tensor::scan` builder + 2-slot bundle + shape/dtype rule (C3)

**Files:**
- Modify: `fuel-graph/src/lib.rs` (add `Tensor::scan` near `cumsum`/`triu` ~`lib.rs:5079`; tests in the `Tensor` test module).

**Interfaces:**
- Consumes: `Node`, `Op::Scan`, `fuel_ir::storage::OutputViewSpec`/`compose_bundle`, `Graph::set_output_views(id, Arc<[OutputView]>)`, `Tensor::view(slot: u32) -> Result<Tensor, fuel_ir::Error>`.
- Produces:
  `pub fn scan(&self, xs: &[Tensor], consts: &[Tensor], body_new_carry: &Tensor, body_y: &Tensor, bound: usize, emit: ScanEmit) -> std::result::Result<Tensor, fuel_ir::Error>` where `&self` is the `init_carry`. Builds ONE `Op::Scan` node with `inputs = [self.id, xs.., consts.., body_new_carry.id, body_y.id]`, `n_xs = xs.len()`, `early_exit: None`, a 2-slot bundle (slot 0 `ys` shape `[bound] ++ body_y.dims()`, slot 1 `carry` = `self.shape()`), and returns `view(0)` for `emit=All` / `view(1)` for `emit=Final`. `Result`, never a panic on a malformed body.

- [ ] **Step 1: Read the precedents** — `Tensor::cumsum` (`lib.rs:5033`, the `Result`/validate/`self.graph.write().unwrap().push` skeleton) and `selective_scan_producer` (`lib.rs:6052`, the `output_views` → `compose_bundle` → `set_output_views` bundle mechanics — but **convert its `.expect()`/`assert!` to typed `Err`**). Confirm `OutputViewSpec::contiguous(dtype, shape)` and `compose_bundle(&specs) -> Result<(usize, Vec<OutputView>)>`.

- [ ] **Step 2: Write the failing tests** in the `Tensor` test module:

```rust
#[test]
fn scan_builder_all_and_final_shapes() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole, Tensor};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};
    let graph = Arc::new(RwLock::new(Graph::new()));
    let cs = Shape::from_dims(&[2]);   // carry shape
    let ys = Shape::from_dims(&[2]);   // per-step y shape
    let (carry, hole, new_carry, body_y) = {
        let mut g = graph.write().unwrap();
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: cs.clone(), dtype: DType::F32 });
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: cs.clone(), dtype: DType::F32 });
        let new_carry = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: cs.clone(), dtype: DType::F32 });
        let body_y = g.push(Node { op: Op::AddScalar(1.0), inputs: vec![new_carry], shape: ys.clone(), dtype: DType::F32 });
        (carry, hole, new_carry, body_y)
    };
    let t = |id| Tensor { graph: graph.clone(), id };  // same-crate access to private fields
    let init = t(carry);
    let (nc, by) = (t(new_carry), t(body_y));
    let bound = 4usize;

    let all = init.scan(&[], &[], &nc, &by, bound, ScanEmit::All).expect("scan All");
    assert_eq!(all.shape().dims(), &[bound, 2], "emit=All -> [bound] ++ body_y");

    let fin = init.scan(&[], &[], &nc, &by, bound, ScanEmit::Final).expect("scan Final");
    assert_eq!(fin.shape().dims(), &[2], "emit=Final -> carry shape");
    let _ = hole;
}

#[test]
fn scan_builder_rejects_zero_bound() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole, Tensor};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};
    let graph = Arc::new(RwLock::new(Graph::new()));
    let cs = Shape::from_dims(&[1]);
    let (carry, new_carry, body_y) = {
        let mut g = graph.write().unwrap();
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: cs.clone(), dtype: DType::F32 });
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: cs.clone(), dtype: DType::F32 });
        let nc = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: cs.clone(), dtype: DType::F32 });
        (carry, nc, nc)
    };
    let t = |id| Tensor { graph: graph.clone(), id };
    let r = t(carry).scan(&[], &[], &t(new_carry), &t(body_y), 0, ScanEmit::All);
    assert!(r.is_err(), "bound == 0 must be a typed Err, not a panic");
}
```

- [ ] **Step 3: Run, watch fail.** Run: `cargo test -p fuel-graph --lib scan_builder_all_and_final_shapes scan_builder_rejects_zero_bound -- --exact`. Expected: FAIL (`Tensor::scan` undefined).

- [ ] **Step 4: Implement `Tensor::scan`** after `triu` (~`lib.rs:5079`):

```rust
    /// Append an [`Op::Scan`] node — a bounded `lax.scan` recurrence.
    /// `self` is the `init_carry`. The body must already be built in the same
    /// graph, referencing `Op::ScanPlaceholder{Carry,0}` for the per-step
    /// carry and `Op::ScanPlaceholder{Elem,i}` for the per-step slice of
    /// `xs[i]`; `consts` are referenced by real NodeId. `body_new_carry` /
    /// `body_y` are the body's two exit nodes. Always a 2-slot bundle
    /// (slot 0 = stacked `ys` `[bound] ++ body_y`, slot 1 = final carry);
    /// `emit` selects which slot this call projects. `early_exit` is Phase 2
    /// (always `None` here).
    ///
    /// **Returns `Result`**: a malformed body/carry/bound surfaces as a typed
    /// error — never a panic.
    pub fn scan(
        &self,
        xs: &[Tensor],
        consts: &[Tensor],
        body_new_carry: &Tensor,
        body_y: &Tensor,
        bound: usize,
        emit: ScanEmit,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let same_graph = |t: &Tensor| Arc::ptr_eq(&self.graph, &t.graph);
        if !same_graph(body_new_carry) || !same_graph(body_y)
            || !xs.iter().all(same_graph) || !consts.iter().all(same_graph)
        {
            return Err(fuel_ir::Error::Msg(
                "scan: init_carry, xs, consts, and body exits must live on one graph".into(),
            ).bt());
        }
        if bound == 0 {
            return Err(fuel_ir::Error::Msg("scan: bound must be >= 1".into()).bt());
        }
        let carry_shape = self.shape();
        if body_new_carry.shape().dims() != carry_shape.dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "scan: body_new_carry shape {:?} must equal init_carry shape {:?}",
                body_new_carry.shape().dims(), carry_shape.dims(),
            )).bt());
        }
        let carry_dtype = self.dtype();
        let y_dtype = body_y.dtype();

        // 2-slot bundle: slot 0 = stacked ys, slot 1 = final carry.
        let mut ys_dims: Vec<usize> = Vec::with_capacity(1 + body_y.shape().dims().len());
        ys_dims.push(bound);
        ys_dims.extend_from_slice(body_y.shape().dims());
        let ys_shape = Shape::from_dims(&ys_dims);
        let specs = vec![
            fuel_ir::storage::OutputViewSpec::contiguous(y_dtype, ys_shape.clone()),
            fuel_ir::storage::OutputViewSpec::contiguous(carry_dtype, carry_shape.clone()),
        ];
        let (_bytes, views) = fuel_ir::storage::compose_bundle(&specs)
            .map_err(|e| fuel_ir::Error::Msg(format!("scan: compose_bundle failed: {e}")).bt())?;

        let mut inputs: Vec<NodeId> = Vec::with_capacity(2 + xs.len() + consts.len() + 2);
        inputs.push(self.id);
        inputs.extend(xs.iter().map(|t| t.id));
        inputs.extend(consts.iter().map(|t| t.id));
        inputs.push(body_new_carry.id);
        inputs.push(body_y.id);

        let id = {
            let mut g = self.graph.write().unwrap();
            // Node.shape/dtype are the PRIMARY (slot-0) shape/dtype per the
            // multi-output authoring contract.
            let id = g.push(Node {
                op: Op::Scan { n_xs: xs.len(), bound, emit, early_exit: None },
                inputs,
                shape: ys_shape,
                dtype: y_dtype,
            });
            g.set_output_views(id, Arc::from(views.into_boxed_slice()))
                .map_err(|e| fuel_ir::Error::Msg(format!("scan: set_output_views failed: {e}")).bt())?;
            id
        };
        let producer = Self { graph: self.graph.clone(), id };
        match emit {
            ScanEmit::All => producer.view(0),
            ScanEmit::Final => producer.view(1),
        }
    }
```

Confirm `Tensor::shape()`/`dtype()` accessors and `NodeId`/`Shape`/`Arc` are already imported in `lib.rs` (they are — used by `cumsum`/`selective_scan_producer`). Confirm `compose_bundle`'s exact `Ok` tuple shape and error type at write-time.

- [ ] **Step 5: Run, watch pass.** Run: `cargo test -p fuel-graph --lib scan_builder_all_and_final_shapes scan_builder_rejects_zero_bound -- --exact`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add fuel-graph/src/lib.rs
git commit -m "feat(graph): Tensor::scan builder — 2-slot bundle, Result/never-panic (C3)"
```

---

## Task 4: `unroll_scan` utility + `Op::Scan`-is-terminal confirmation (C4 + C5)

**Files:**
- Create: `fuel-graph/src/scan.rs`.
- Modify: `fuel-graph/src/lib.rs` (add `pub mod scan;` near the other `pub mod` lines at `lib.rs:46–51`; re-export `pub use scan::unroll_scan;` if the crate re-exports module items — grep for a `pub use` block).

**Interfaces:**
- Consumes: `Graph`, `Node`, `Op::Scan`/`Op::ScanPlaceholder`, `Op::Slice`/`Op::Squeeze`/`Op::Unsqueeze`/`Op::Concat`, `lower_to_base_map` (for the terminal test).
- Produces:
  `pub fn unroll_scan(graph: &mut Graph, scan_id: NodeId, steps: usize) -> std::result::Result<(NodeId, NodeId), fuel_ir::Error>`. Materializes `steps` iterations of the body into real primitive nodes. **Return contract:** `(selected, complementary)` where `emit=All` → `(stacked_ys, final_carry)` and `emit=Final` → `(final_carry, stacked_ys)`. `stacked_ys` is `Concat` (dim 0) of each step's `body_y` unsqueezed at dim 0 → shape `[steps] ++ body_y`. `early_exit = Some` → `Err` (surfaced Phase-2 gap, never a panic). Not registered as anyone's `.decompose` — a standalone oracle/fallback utility.

- [ ] **Step 1: Write the failing tests** in `fuel-graph/src/scan.rs`'s `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use crate::{Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole};
    use crate::scan::unroll_scan;
    use crate::opt::lower_to_base_map;
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};

    // Build a trivial scan: carry [1], body new_carry = carry*2, body_y =
    // new_carry, n_xs = 0, bound = 3, emit = All. Returns (graph_arc, scan_id).
    fn trivial_scan(bound: usize, emit: ScanEmit, early_exit: Option<ScanPredicate>) -> (Arc<RwLock<Graph>>, crate::NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let scan = {
            let mut g = graph.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let nc = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
            g.push(Node {
                op: Op::Scan { n_xs: 0, bound, emit, early_exit },
                inputs: vec![carry, nc, nc],
                shape: Shape::from_dims(&[bound, 1]),
                dtype: DType::F32,
            })
        };
        (graph, scan)
    }

    #[test]
    fn unroll_scan_all_produces_a_concat_of_steps_and_no_scan_nodes() {
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let (ys, _carry) = {
            let mut g = graph.write().unwrap();
            unroll_scan(&mut g, scan, 3).expect("unroll")
        };
        let g = graph.read().unwrap();
        // ys root is a Concat over the 3 steps.
        assert!(matches!(g.node(ys).op, Op::Concat { .. }), "emit=All ys root should be Concat, got {:?}", g.node(ys).op.short_name());
        assert_eq!(g.node(ys).inputs.len(), 3, "one input per step");
        // No Op::Scan / Op::ScanPlaceholder reachable from the unrolled root.
        let reachable = crate::topo_order_multi(&g, &[ys]);
        assert!(!reachable.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
            "unrolled graph must contain no Scan/ScanPlaceholder nodes");
    }

    #[test]
    fn unroll_scan_rejects_early_exit_some() {
        let (graph, scan) = trivial_scan(2, ScanEmit::All, Some(ScanPredicate));
        let mut g = graph.write().unwrap();
        let r = unroll_scan(&mut g, scan, 2);
        assert!(r.is_err(), "early_exit = Some must be a typed Err (Phase-2 gap), never a panic");
    }

    #[test]
    fn op_scan_is_a_terminal_in_the_base_map() {
        // lower_to_base_map must LEAVE Op::Scan in place (no LoweringRule
        // matches a bare Op variant) — not silently expanded, not errored.
        let (graph, scan) = trivial_scan(3, ScanEmit::All, None);
        let roots = lower_to_base_map(&graph, &[scan]);
        let g = graph.read().unwrap();
        let reachable = crate::topo_order_multi(&g, &roots);
        assert!(reachable.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. })),
            "Op::Scan must remain a terminal after lower_to_base_map");
    }
}
```

- [ ] **Step 2: Run, watch fail.** Run: `cargo test -p fuel-graph --lib scan::tests -- --exact` (or the three names individually). Expected: FAIL — `crate::scan` / `unroll_scan` undefined (`op_scan_is_a_terminal_in_the_base_map` may compile-fail too until the module exists).

- [ ] **Step 3: Implement `fuel-graph/src/scan.rs`.** Mirror `flash_attn::decompose`'s node-emission style (short borrow block to read the node, then a loop of `graph.push(Node{..})`). The per-step body is cloned by a topological copy that substitutes `ScanPlaceholder{Carry,0}` → the current carry, `ScanPlaceholder{Elem,i}` → the per-step slice of `xs[i]`, and keeps `consts` (real NodeIds) shared:

```rust
//! `unroll_scan`: materialize a bounded [`crate::Op::Scan`] into real
//! primitive nodes on demand. Used as (a) the FKC/Spec-B numeric oracle and
//! (b) the fallback lowering for a backend without a scan kernel. NOT
//! registered as anyone's `.decompose` — `Op::Scan` is a bare primitive that
//! stays terminal in the base map.

use std::collections::HashMap;

use crate::{Graph, Node, NodeId, Op, ScanEmit, ScanRole};

/// Unroll `steps` iterations of the `Op::Scan` at `scan_id` into primitives.
///
/// Returns `(selected, complementary)`: `emit=All` -> `(stacked_ys,
/// final_carry)`, `emit=Final` -> `(final_carry, stacked_ys)`. `early_exit =
/// Some` -> `Err` (a surfaced Phase-2 gap, never a panic).
pub fn unroll_scan(
    graph: &mut Graph,
    scan_id: NodeId,
    steps: usize,
) -> std::result::Result<(NodeId, NodeId), fuel_ir::Error> {
    // 1. Read the Scan node's params + input layout in a short borrow.
    let (n_xs, bound, emit, has_exit, inputs) = {
        let n = graph.node(scan_id);
        match &n.op {
            Op::Scan { n_xs, bound, emit, early_exit } => {
                (*n_xs, *bound, *emit, early_exit.is_some(), n.inputs.clone())
            }
            other => {
                return Err(fuel_ir::Error::Msg(format!(
                    "unroll_scan: node {} is not an Op::Scan ({})",
                    scan_id.0, other.short_name(),
                )).bt());
            }
        }
    };
    if has_exit {
        return Err(fuel_ir::Error::Msg(
            "unroll_scan: early_exit = Some is a Phase-2 mechanism (not implemented)".into(),
        ).bt());
    }
    if steps == 0 || steps > bound {
        return Err(fuel_ir::Error::Msg(format!(
            "unroll_scan: steps {steps} must be in 1..={bound}",
        )).bt());
    }
    // inputs = [init_carry, xs_0..xs_{n_xs-1}, consts.., body_new_carry, body_y]
    if inputs.len() < 2 + n_xs {
        return Err(fuel_ir::Error::Msg("unroll_scan: malformed Op::Scan inputs".into()).bt());
    }
    let init_carry = inputs[0];
    let xs: Vec<NodeId> = inputs[1..1 + n_xs].to_vec();
    let consts: Vec<NodeId> = inputs[1 + n_xs..inputs.len() - 2].to_vec();
    let body_new_carry = inputs[inputs.len() - 2];
    let body_y = inputs[inputs.len() - 1];
    let consts_set: std::collections::HashSet<NodeId> = consts.iter().copied().collect();

    let mut carry = init_carry;
    let mut ys_steps: Vec<NodeId> = Vec::with_capacity(steps);

    for t in 0..steps {
        // Per-step xs slices: xs[i] sliced at [t, t+1) on scan-axis 0, then
        // squeezed to drop the step axis -> the ScanPlaceholder{Elem,i} shape.
        let mut elem: Vec<NodeId> = Vec::with_capacity(n_xs);
        for &x in &xs {
            let (x_shape, x_dtype) = { let n = graph.node(x); (n.shape.clone(), n.dtype) };
            let sliced_dims: Vec<usize> = std::iter::once(1usize)
                .chain(x_shape.dims().iter().skip(1).copied()).collect();
            let sl = graph.push(Node {
                op: Op::Slice { dim: 0, start: t, len: 1 },
                inputs: vec![x],
                shape: fuel_ir::Shape::from_dims(&sliced_dims),
                dtype: x_dtype,
            });
            let sq_dims: Vec<usize> = x_shape.dims().iter().skip(1).copied().collect();
            let sq = graph.push(Node {
                op: Op::Squeeze { dim: 0 },
                inputs: vec![sl],
                shape: fuel_ir::Shape::from_dims(&sq_dims),
                dtype: x_dtype,
            });
            elem.push(sq);
        }

        // Clone the body subgraph (rooted at {body_new_carry, body_y}),
        // substituting placeholders + keeping consts shared.
        let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
        let next_carry = clone_body_node(graph, body_new_carry, carry, &elem, &consts_set, &mut subst);
        let y_t = clone_body_node(graph, body_y, carry, &elem, &consts_set, &mut subst);
        carry = next_carry;
        ys_steps.push(y_t);
    }

    // stacked_ys = Concat(dim 0) of each y_t unsqueezed at dim 0.
    let mut unsqueezed: Vec<NodeId> = Vec::with_capacity(ys_steps.len());
    for &y in &ys_steps {
        let (y_shape, y_dtype) = { let n = graph.node(y); (n.shape.clone(), n.dtype) };
        let un_dims: Vec<usize> = std::iter::once(1usize).chain(y_shape.dims().iter().copied()).collect();
        let un = graph.push(Node {
            op: Op::Unsqueeze { dim: 0 },
            inputs: vec![y],
            shape: fuel_ir::Shape::from_dims(&un_dims),
            dtype: y_dtype,
        });
        unsqueezed.push(un);
    }
    let (y0_shape, y0_dtype) = { let n = graph.node(ys_steps[0]); (n.shape.clone(), n.dtype) };
    let stacked_dims: Vec<usize> = std::iter::once(ys_steps.len())
        .chain(y0_shape.dims().iter().copied()).collect();
    let stacked_ys = graph.push(Node {
        op: Op::Concat { dim: 0 },
        inputs: unsqueezed,
        shape: fuel_ir::Shape::from_dims(&stacked_dims),
        dtype: y0_dtype,
    });

    Ok(match emit {
        ScanEmit::All => (stacked_ys, carry),
        ScanEmit::Final => (carry, stacked_ys),
    })
}

/// Topological copy of a body node, substituting `ScanPlaceholder{Carry,_}` ->
/// `carry`, `ScanPlaceholder{Elem,i}` -> `elem[i]`, and keeping any node in
/// `consts_set` shared (not copied). Memoized in `subst`.
fn clone_body_node(
    graph: &mut Graph,
    id: NodeId,
    carry: NodeId,
    elem: &[NodeId],
    consts_set: &std::collections::HashSet<NodeId>,
    subst: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&m) = subst.get(&id) { return m; }
    if consts_set.contains(&id) { return id; }
    let (op, in_ids, shape, dtype) = {
        let n = graph.node(id);
        (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
    };
    let mapped = match op {
        Op::ScanPlaceholder { role: ScanRole::Carry, .. } => carry,
        Op::ScanPlaceholder { role: ScanRole::Elem, index } => elem[index],
        _ => {
            let new_inputs: Vec<NodeId> = in_ids.iter()
                .map(|&c| clone_body_node(graph, c, carry, elem, consts_set, subst))
                .collect();
            graph.push(Node { op, inputs: new_inputs, shape, dtype })
        }
    };
    subst.insert(id, mapped);
    mapped
}

#[cfg(test)]
mod tests {
    // (the tests from Step 1)
}
```

Confirm `Op::Slice { dim, start, len }`, `Op::Squeeze { dim }`, `Op::Unsqueeze { dim }`, `Op::Concat { dim }` spellings (all confirmed present in `op_short_name`) and `fuel_ir::Shape::from_dims`. Add `pub mod scan;` to `lib.rs` and (if the crate re-exports) `pub use scan::unroll_scan;`.

- [ ] **Step 4: Run, watch pass.** Run: `cargo test -p fuel-graph --lib scan::tests -- --exact`. Expected: all three PASS.

- [ ] **Step 5: Commit.**

```bash
git add fuel-graph/src/scan.rs fuel-graph/src/lib.rs
git commit -m "feat(graph): unroll_scan utility + Op::Scan terminal-in-base-map (C4/C5)"
```

---

## Task 5: opt-in sites — safe conservative defaults + reachability confirmation (C6)

**Files:**
- Modify (optional doc only): `fuel-graph/src/jit.rs` (extend `op_to_tag`'s exclusion-list doc comment to name `Op::Scan`). No production-code arms — every site below is a confirmed safe default via an existing wildcard/allow-list.
- Tests: `fuel-graph/src/lib.rs` (or `opt.rs`) test module; one `fuel-dispatch` test for `op_to_op_kind`.

**Interfaces (all confirmed safe defaults — no new arms):**
- `op_to_tag(op) -> Option<OpTag>` (`jit.rs:22`, `_ => return None`) → `None` for `Op::Scan` ("not a region node; its decomposition is", the `Op::Fused` precedent).
- `infer_storage_class(op) -> StorageClass` (`lib.rs:1422`, `_ => Transient`) → `Transient`.
- `Op::destructive_input` (`lib.rs:1121`, `_ => None`) → `None`; `Op::is_view_op` (`lib.rs:1170`, allow-list) → `false`; `try_simplify` (`opt.rs:1453`, `_ => None`) → `None`.
- `op_to_op_kind(op) -> Option<OpKind>` (`fuel-dispatch/src/pipelined.rs:3120`, `_ => None`) → `None` (no native Scan kernel in Phase 1).
- Reachability: `Graph::live_set` (`lib.rs:2169`) and `effective_roots` (`run.rs:208`) need **no** extension — body-exits are ordinary `inputs` and the `Op::Scan` node is read by downstream consumers, so the input-closure walk sees the body for free (unlike `Op::Branch`, which is orphaned after finalization).

- [ ] **Step 1: Write the failing/confirming tests.** In `fuel-graph`'s test module:

```rust
#[test]
fn scan_opt_in_sites_are_safe_defaults() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole};
    use fuel_ir::{DType, Shape};
    let scan = Op::Scan { n_xs: 0, bound: 2, emit: ScanEmit::All, early_exit: None };
    let hole = Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 };
    // Classifier defaults.
    assert_eq!(scan.destructive_input(), None);
    assert_eq!(hole.destructive_input(), None);
    assert!(!scan.is_view_op());
    assert!(!hole.is_view_op());
    assert_eq!(crate::infer_storage_class(&scan), crate::StorageClass::Transient);
    assert_eq!(crate::jit::op_to_tag(&scan), None);
    // Reachability: build a graph with an Op::Scan whose body references a hole;
    // the hole (a body node) is reachable from the Scan via inputs, no seeding.
    let mut g = Graph::new();
    let s = Shape::from_dims(&[1]);
    let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
    let h = g.push(Node { op: hole.clone(), inputs: vec![], shape: s.clone(), dtype: DType::F32 });
    let nc = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![h], shape: s.clone(), dtype: DType::F32 });
    let sc = g.push(Node { op: scan.clone(), inputs: vec![carry, nc, nc], shape: Shape::from_dims(&[2,1]), dtype: DType::F32 });
    let reachable = crate::topo_order_multi(&g, &[sc]);
    assert!(reachable.contains(&nc) && reachable.contains(&h),
        "body nodes must be reachable from Op::Scan via inputs (no effective_roots edit needed)");
}
```

Confirm `crate::StorageClass` and `crate::infer_storage_class`/`crate::jit::op_to_tag` visibility at write-time; adjust the import paths to the real ones (grep `pub fn infer_storage_class`, `pub enum StorageClass`, `pub fn op_to_tag`). If any is not reachable from the test module, split that assertion into the module where it lives.

- [ ] **Step 2: Run, watch fail (or pass).** Run: `cargo test -p fuel-graph --lib scan_opt_in_sites_are_safe_defaults -- --exact`. Expected: PASS immediately (these are pre-existing wildcards) — this is a **regression lock**, not a born-red. If any assertion is RED, that site is silently wrong for `Op::Scan` and needs the conservative arm named above; fix it, then green.

- [ ] **Step 3: (Optional) `op_to_op_kind` lock in fuel-dispatch.** Add to `fuel-dispatch`'s test module:

```rust
#[test]
fn scan_has_no_dispatch_op_kind() {
    use fuel_graph::{Op, ScanEmit};
    assert_eq!(
        crate::pipelined::op_to_op_kind(&Op::Scan { n_xs: 0, bound: 2, emit: ScanEmit::All, early_exit: None }),
        None,
    );
}
```

Confirm `op_to_op_kind`'s visibility (`pub(crate)`) — if not reachable from the test path, skip this and rely on the design's confirmed `_ => None`. Run: `cargo test -p fuel-dispatch --lib scan_has_no_dispatch_op_kind -- --exact`. Expected: PASS.

- [ ] **Step 4: (Optional doc) extend `op_to_tag`'s exclusion comment** in `jit.rs` to name `Op::Scan` alongside `Op::Fused`. No behavior change.

- [ ] **Step 5: Commit.**

```bash
git add fuel-graph/src/lib.rs fuel-graph/src/jit.rs fuel-dispatch/src/pipelined.rs
git commit -m "test(graph): Op::Scan opt-in sites are safe defaults + body reachable via inputs (C6)"
```

---

## Task 6: `selective_scan` re-decompose → `Op::Scan` + flip its gap test (C8 + C9, part 1)

**Files:**
- Modify: `fuel-graph/src/registry/selective_scan.rs` (`decompose` body at `:210`; module doc at `:196`).
- Modify: `fuel-core/src/lazy.rs` (flip `selective_scan_decompose_is_surfaced_gap_not_a_crash` at `:2069`).

**Interfaces:**
- `pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId` — **signature unchanged** (it is a `FusedOpEntry.decompose` fn-pointer). Now builds an `Op::Scan { emit: All }` whose body is the affine SSM step `h ← exp(d·a)·h + d·b·u`, carry `h = [batch, dim, dstate]`, `n_xs = 4` (per-step `u`/`delta`/`b`/`c`), `consts = [a]`, `bound = seqlen`, and returns the `Op::Scan` node id (a 2-slot bundle whose slots exactly match `selective_scan::output_views`, so the existing `Op::View(0)`/`(1)` consumers keep working). Wrong params → `return id` (the G2 self-return convention, mirroring `flash_attn::decompose`). `shape_rule`/`dtype_rule`/`output_views`/`backward` (`NotDifferentiable`) stay unchanged.

- [ ] **Step 1: Read the math + precedents.** `byte_kernels.rs:6104` (recurrence: `d = softplus(delta) if flag else delta`; `h[b,i,j] = exp(d·a[i,j])·h[b,i,j] + d·b[b,t,j]·u[b,t,i]`; `y[b,t,i] = sum_j(h·c[b,t,j])`); `byte_kernels.rs:6213` (F64-accumulate loop; stable softplus `raw_d.max(0.0) + (1.0 + (-abs_x).exp()).ln()`). `flash_attn::decompose` (`registry/flash_attn.rs:123`) for the destructure-then-push-loop style. `selective_scan::output_views` (`selective_scan.rs:144`) for the 2-slot specs to reuse. There is no `Op::Log1p` — softplus = `Add(Maximum(x, 0), Log(Add(1, Exp(Neg(Abs(x))))))` over the whole `delta` tensor **before** the scan (softplus has no recurrent dependency).

- [ ] **Step 2: Write the failing test — flip the gap-posture test.** Replace `fuel-core/src/lazy.rs`'s `selective_scan_decompose_is_surfaced_gap_not_a_crash` (`:2069`) with the positive form (mirror the `nf4_matmul_decompose_matches_kernel` template at `:2124`). Keep the `B=1,T=1,dim=1,dstate=1` fixture (`u=2, delta=0.5, a=-1, b=3, c=4`, no softplus → `h=3, y=12`). The numeric leg realizes via `unroll_scan` (the oracle path) since `Op::Scan` has no kernel; the fused-kernel leg (`y.realize_f32()`) stays as the non-regression check:

```rust
#[test]
fn selective_scan_decompose_lowers_to_scan_and_matches() {
    use fuel_graph::registry::FusedOps;
    use fuel_graph::Op;
    let dev = Device::cpu();
    let u = LazyTensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1]), &dev);
    let delta = u.const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]));
    let a = u.const_f32_like(vec![-1.0f32], Shape::from_dims(&[1, 1]));
    let b = u.const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1]));
    let c = u.const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1]));
    let y = u.selective_scan(&delta, &a, &b, &c, /* delta_softplus */ false);

    // (a) NON-REGRESSION: the fused kernel still runs and produces 12.0.
    let got = y.realize_f32();
    assert_eq!(got.len(), 1);
    assert!((got[0] - 12.0).abs() < 1e-4, "fused kernel y: {}", got[0]);

    // (b) VERIFICATION: lowering leaves NO Op::Fused(SELECTIVE_SCAN); an
    // Op::Scan terminal is present; unroll+realize matches h=3,y=12.
    let graph = y.inner.graph().clone();
    let id = y.inner.id();
    let roots = fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
    assert_eq!(roots.len(), 1);
    let scan_id = {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]];
        let mut seen = std::collections::HashSet::new();
        let mut scan_id = None;
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) { continue; }
            let node = g.node(nid);
            assert!(!matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::SELECTIVE_SCAN),
                "SelectiveScan must lower to Op::Scan, not remain fused");
            if matches!(node.op, Op::Scan { .. }) { scan_id = Some(nid); }
            for &inp in &node.inputs { stack.push(inp); }
        }
        scan_id.expect("an Op::Scan terminal must be present after lowering")
    };
    // Unroll the Op::Scan (seqlen = 1) and realize the ys oracle.
    let ys = {
        let mut g = graph.write().unwrap();
        fuel_graph::scan::unroll_scan(&mut g, scan_id, 1).expect("unroll").0
    };
    let oracle = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
        .expect("realize unrolled selective_scan oracle on CPU");
    assert!(oracle.iter().any(|&v| (v - 12.0).abs() < 1e-4),
        "unroll oracle must contain y = 12, got {oracle:?}");
}
```

Confirm `y.inner.graph()`/`y.inner.id()`/`const_f32_like`/`realize_f32`/`realize_one_as` at write-time (all used verbatim by the existing test). Confirm `fuel_graph::scan::unroll_scan` is re-exported (Task 4 Step 3).

- [ ] **Step 3: Run, watch fail.** Run: `cargo test -p fuel-core --lib selective_scan_decompose_lowers_to_scan_and_matches -- --exact`. Expected: FAIL — `decompose` still self-returns, so `lowering_only` leaves `Op::Fused(SELECTIVE_SCAN)` (the `!matches!` assertion fires) and no `Op::Scan` is found.

- [ ] **Step 4: Implement the re-decompose** in `selective_scan.rs`. Replace the self-return `decompose` (`:210`) with an `Op::Scan`-emitting body. Read the 5 input NodeIds off `graph.node(id).inputs` (`u`, `delta`, `a`, `b`, `c`) in a short borrow, then build:
  1. `d` = `delta` if `!delta_softplus`, else the softplus subgraph over the whole `[batch,seqlen,dim]` `delta` tensor (`Add(Maximum(delta,0), Log(Add(1, Exp(Neg(Abs(delta))))))` via `graph.push`).
  2. An `init_carry` zero const of shape `[batch, dim, dstate]` (the `last_state` shape from `output_views`).
  3. The body (referencing `ScanPlaceholder{Carry,0}` = `h` `[batch,dim,dstate]`; `{Elem,0}`=`u_t` `[batch,dim]`, `{Elem,1}`=`d_t` `[batch,dim]`, `{Elem,2}`=`b_t` `[batch,dstate]`, `{Elem,3}`=`c_t` `[batch,dstate]`; `a` `[dim,dstate]` by real NodeId as the single const): `gate = Exp(BroadcastTo(d_t)⊙a)`; `bu = BroadcastTo(d_t·u_t) ⊙ BroadcastTo(b_t)`; `h_new = gate⊙h + bu` (all `[batch,dim,dstate]`); `y_t = ReduceSumTo over dstate of (h_new ⊙ BroadcastTo(c_t))` → `[batch,dim]`. Emit the exact `Op` primitives (`Mul`/`Exp`/`Add`/`BroadcastTo`/`ReduceSumTo`); the broadcast/reshape shapes follow from the `byte_kernels.rs:6104` index math.
  4. Push the `Op::Scan { n_xs: 4, bound: seqlen, emit: All, early_exit: None }` node with `inputs = [init_carry, u_series, d_series, b, c, a, body_new_carry=h_new, body_y=y_t]` and attach `output_views` = the existing `selective_scan::output_views` specs via `compose_bundle` + `graph.set_output_views`, so the `Op::View(0)/(1)` consumers keep projecting `y` / `last_state`. Return the `Op::Scan` node id. On a params mismatch, `return id`.

  > **Numeric-parity note (F64):** the CPU kernel accumulates `h`/`y` in F64 then narrows (`byte_kernels.rs:6213`). For Phase 1 the body computes in the tensor's native dtype; this is **bit-exact at the `T=1`/size-1 gap fixture** (all values represent exactly in F32). Multi-token F64-accumulate parity is validated (and, if it drifts beyond the sabotage-calibrated tolerance, the body carry is cast to F64 internally + narrowed on emit) by Task 8's parity test — the fused kernel remains the production path regardless.

- [ ] **Step 5: Run, watch pass.** Run: `cargo test -p fuel-core --lib selective_scan_decompose_lowers_to_scan_and_matches -- --exact`. Expected: PASS. Then run the whole file's SSM tests: `cargo test -p fuel-graph --lib` (registry compiles) and `cargo test -p fuel-core --lib selective_scan -- --nocapture` (no other SSM test regressed).

- [ ] **Step 6: Update the module doc** at `selective_scan.rs:196` — the "canonical basis gap / decompose returns self" prose is now false; rewrite it to "decomposes to `Op::Scan` (G3 closed); the fused CPU/CUDA kernel stays the executed path."

- [ ] **Step 7: Commit.**

```bash
git add fuel-graph/src/registry/selective_scan.rs fuel-core/src/lazy.rs
git commit -m "feat(ir): selective_scan decomposes to Op::Scan (closes G3, part 1) + flip gap test (C8/C9)"
```

---

## Task 7: `ssd_chunk_scan` re-decompose → `Op::Scan` + new gap test + stale-doc fix (C8 + C9, part 2)

**Files:**
- Modify: `fuel-graph/src/registry/ssd_chunk_scan.rs` (`decompose` at `:169`; stale "decompose panics" doc at `:53`; module doc at `:156`).
- Modify: `fuel-core/src/lazy.rs` (ADD a new `ssd_chunk_scan` positive test — none exists today).
- (Optional, same doc-fix pass) `fuel-cpu-backend/src/byte_kernels.rs:6331–6334` — the stale "single-chunk-only" comment contradicts the code at `:6398–6414`.

**Interfaces:**
- `pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId` — same signature; builds `Op::Scan { emit: All }` for the Mamba-2 SSD recurrence (`byte_kernels.rs:6328`/`:6478`): carry `h_state = [batch, heads, head_dim, state_dim]`, per-head **scalar** gate `exp(dt·a_h)` (simpler than selective_scan's per-`(dim,dstate)` vector gate — `a` is `[heads]`). `n_xs = 4` (`x`/`dt`/`b`/`c`), `consts = [a]`, `bound = seqlen`, **no** softplus. Returns the `Op::Scan` bundle id matching `ssd_chunk_scan::output_views`.

- [ ] **Step 1: Read** `ssd_chunk_scan.rs:156` (self-return + basis-gap doc), `:53` (stale "panics" doc), `byte_kernels.rs:6478` (the per-step loop: `exp_d_a = exp(dt·a_h)` hoisted per head; `h_new = exp_d_a·h + dt·b·x`; `y = sum_j(h·c)`). Note `chunk_size` is a documented CPU no-op (`byte_kernels.rs:6398`).

- [ ] **Step 2: Write the failing test** (author from scratch; mirror Task 6's shape). Use a degenerate `batch=heads=head_dim=state_dim=1, seqlen=chunk_size=1` fixture — pick simple inputs (e.g. `x=2, dt=0.5, a=-1, b=3, c=4` → `exp(0.5·-1)·0 + 0.5·3·2 = 3`, `y = 3·4 = 12`) so the expected `y=12` mirrors Task 6:

```rust
#[test]
fn ssd_chunk_scan_decompose_lowers_to_scan_and_matches() {
    use fuel_graph::registry::FusedOps;
    use fuel_graph::Op;
    let dev = Device::cpu();
    // x [batch, seqlen, heads, head_dim] = [1,1,1,1]; dt [b,s,h]=[1,1,1];
    // a [heads]=[1]; b/c [b,s,h,state]=[1,1,1,1].
    let x  = LazyTensor::from_f32(vec![2.0f32], Shape::from_dims(&[1, 1, 1, 1]), &dev);
    let dt = x.const_f32_like(vec![0.5f32], Shape::from_dims(&[1, 1, 1]));
    let a  = x.const_f32_like(vec![-1.0f32], Shape::from_dims(&[1]));
    let b  = x.const_f32_like(vec![3.0f32], Shape::from_dims(&[1, 1, 1, 1]));
    let c  = x.const_f32_like(vec![4.0f32], Shape::from_dims(&[1, 1, 1, 1]));
    let y = x.ssd_chunk_scan(&dt, &a, &b, &c, /* chunk_size */ 1);

    let got = y.realize_f32();
    assert!((got[0] - 12.0).abs() < 1e-4, "fused ssd kernel y: {}", got[0]);

    let graph = y.inner.graph().clone();
    let id = y.inner.id();
    let roots = fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
    let scan_id = {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]];
        let mut seen = std::collections::HashSet::new();
        let mut scan_id = None;
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) { continue; }
            let node = g.node(nid);
            assert!(!matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::SSD_CHUNK_SCAN),
                "SsdChunkScan must lower to Op::Scan, not remain fused");
            if matches!(node.op, Op::Scan { .. }) { scan_id = Some(nid); }
            for &inp in &node.inputs { stack.push(inp); }
        }
        scan_id.expect("an Op::Scan terminal must be present after lowering")
    };
    let ys = {
        let mut g = graph.write().unwrap();
        fuel_graph::scan::unroll_scan(&mut g, scan_id, 1).expect("unroll").0
    };
    let oracle = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev)
        .expect("realize unrolled ssd_chunk_scan oracle on CPU");
    assert!(oracle.iter().any(|&v| (v - 12.0).abs() < 1e-4), "oracle y=12, got {oracle:?}");
}
```

Confirm `LazyTensor::ssd_chunk_scan`'s exact signature at write-time (grep `fn ssd_chunk_scan` in `fuel-core`/`fuel-graph`); adjust arg order/types to match.

- [ ] **Step 3: Run, watch fail.** Run: `cargo test -p fuel-core --lib ssd_chunk_scan_decompose_lowers_to_scan_and_matches -- --exact`. Expected: FAIL (still self-returns → `Op::Fused(SSD_CHUNK_SCAN)` survives).

- [ ] **Step 4: Implement the re-decompose** in `ssd_chunk_scan.rs` mirroring Task 6 Step 4, but with the **scalar per-head** gate (`a` is `[heads]`, so `exp(dt_t·a_h)` broadcasts a scalar across the whole `head_dim×state_dim` block) and **no** softplus. Reuse `ssd_chunk_scan::output_views` for the 2-slot bundle. Return the `Op::Scan` id; params mismatch → `return id`.

- [ ] **Step 5: Fix the stale docs.** In `ssd_chunk_scan.rs:53` replace "`[decompose]` panics" with "`decompose` lowers to `Op::Scan` (G3 closed)"; rewrite the `:156` basis-gap doc. (Optional, same commit) fix `byte_kernels.rs:6331–6334`'s stale "single-chunk-only" comment to match the correct `:6398–6414` prose.

- [ ] **Step 6: Run, watch pass.** Run: `cargo test -p fuel-core --lib ssd_chunk_scan_decompose_lowers_to_scan_and_matches -- --exact`. Expected: PASS. Then `cargo test -p fuel-graph --lib` (registry compiles).

- [ ] **Step 7: Commit.**

```bash
git add fuel-graph/src/registry/ssd_chunk_scan.rs fuel-core/src/lazy.rs fuel-cpu-backend/src/byte_kernels.rs
git commit -m "feat(ir): ssd_chunk_scan decomposes to Op::Scan (closes G3, part 2) + gap test + stale-doc fix (C8/C9)"
```

---

## Task 8: Non-regression gate — fused SSM kernel stays the executed path (C7)

**Files:**
- Tests: `fuel-core/src/lazy.rs` (a multi-token parity + non-regression test). No production edits expected.

**Interfaces / mechanism (state it in the test's doc comment):** The production executor (`PipelinedExecutor::realize`, `fuel-dispatch/src/pipelined.rs:841`) dispatches `Op::Fused(SELECTIVE_SCAN)` **directly** to its registered kernel via the per-dtype dispatch table (`fuel-dispatch/src/dispatch.rs:3263`); it does **not** run a `RuleRegistry` lowering pass, so `decompose` (→`Op::Scan`) is never called on the execution path. `decompose` is reached only by `lowering_only()`-based verification (the gap tests) and a kernel-absent fallback. Therefore emitting `Op::Scan` from `decompose` cannot regress the kerneled path **by construction** — the fused kernel arm remains costed and reachable. `variant_bake` (`fuel-dispatch/src/variant_bake.rs`) is not involved (the SSM op is a plain `Op::Fused`, not an `Op::Branch` arm, in Phase 1).

- [ ] **Step 1: Write the parity + non-regression test.** A multi-token `selective_scan` (`seqlen = 4`, small `dim`/`dstate`, deterministic inputs): assert (a) `realize_f32()` (the fused kernel) succeeds and matches a hand/reference value — proving the kernel, not an `O(seqlen)` unroll, runs; and (b) the `unroll_scan` oracle over the lowered `Op::Scan` matches the fused kernel within a **sabotage-calibrated** tolerance (this is the F64-accumulate parity gate — if it fails, cast the Task 6 body carry to F64 internally):

```rust
#[test]
fn selective_scan_fused_kernel_is_the_executed_path_and_unroll_matches() {
    use fuel_graph::Op;
    let dev = Device::cpu();
    let seqlen = 4usize;
    // Deterministic [1, seqlen, 1] u/delta, [1,1] a, [1, seqlen, 1] b/c.
    let u     = LazyTensor::from_f32(vec![0.5, 1.0, -0.5, 0.25], Shape::from_dims(&[1, seqlen, 1]), &dev);
    let delta = u.const_f32_like(vec![0.1, 0.2, 0.3, 0.4], Shape::from_dims(&[1, seqlen, 1]));
    let a     = u.const_f32_like(vec![-0.7], Shape::from_dims(&[1, 1]));
    let b     = u.const_f32_like(vec![1.0, 0.5, 0.25, 0.125], Shape::from_dims(&[1, seqlen, 1]));
    let c     = u.const_f32_like(vec![2.0, 1.0, 0.5, 0.25], Shape::from_dims(&[1, seqlen, 1]));
    let y = u.selective_scan(&delta, &a, &b, &c, false);

    // (a) Fused kernel result (the production/executed path).
    let fused = y.realize_f32();
    assert_eq!(fused.len(), seqlen);

    // (b) Unroll oracle over the lowered Op::Scan matches the fused kernel.
    let graph = y.inner.graph().clone();
    let roots = fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[y.inner.id()]);
    let scan_id = {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]]; let mut seen = std::collections::HashSet::new(); let mut s = None;
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) { continue; }
            if matches!(g.node(nid).op, Op::Scan { .. }) { s = Some(nid); }
            for &inp in &g.node(nid).inputs { stack.push(inp); }
        }
        s.expect("Op::Scan present")
    };
    let ys = { let mut g = graph.write().unwrap(); fuel_graph::scan::unroll_scan(&mut g, scan_id, seqlen).expect("unroll").0 };
    let oracle = crate::pipelined_bridge::realize_one_as::<f32>(&graph, ys, &dev).expect("realize oracle");
    // Sabotage-calibrated tolerance: native-dtype unroll vs F64-accumulate kernel.
    // Start at 1e-3; if a genuine sabotage (perturb one input) is NOT caught,
    // tighten; if honest drift exceeds it, cast the Task-6 body carry to F64.
    for (o, f) in oracle.iter().zip(fused.iter()) {
        assert!((o - f).abs() < 1e-3, "unroll oracle {o} vs fused {f} beyond tolerance");
    }
}
```

- [ ] **Step 2: Run, watch it pass (and calibrate).** Run: `cargo test -p fuel-core --lib selective_scan_fused_kernel_is_the_executed_path_and_unroll_matches -- --exact --nocapture`. Expected: PASS. If (b) exceeds `1e-3`, implement the F64-internal body cast in Task 6's `decompose`/the `unroll_scan` path and re-run. **Calibrate the tolerance** per `[[sabotage-test-calibration]]`: temporarily perturb one `u` value and confirm the assertion goes RED (the tolerance catches corruption), then restore.

- [ ] **Step 3: (Optional, local GPU only) Mamba decode bench.** If a live RTX 4070 is available, run the existing Mamba/selective_scan decode bench and confirm the fused CUDA kernel timing is unchanged (no `O(seqlen)` unroll on the hot path). Skip if no device — the CPU test above is the binding gate. Do not run two live-GPU suites at once.

- [ ] **Step 4: Commit.**

```bash
git add fuel-core/src/lazy.rs
git commit -m "test(ir): fused SSM kernel stays the executed path + unroll parity gate (C7)"
```

---

## Task 9: Constitution diff (C10)

**Files:**
- Modify: `docs/architecture/03-ir.md`, `docs/architecture/04-optimization.md`, `docs/architecture/08-pattern-harvest.md`, `docs/architecture/12-multi-output.md`, `docs/architecture/14-lifecycle.md`, `docs/architecture/10-decisions-log.md`; and `ROADMAP.md` (frontier).

This is a docs-only task (no cargo). Per CLAUDE.md "docs are part of every material change," judged against `00-index.md:114` ("MAJOR when a section's core claim changes"). **Read each file's actual version header first** and bump from what is there (the versions below are the expected current values from the architecture-map; correct against reality).

- [ ] **Step 1: `03-ir.md` — MAJOR bump (v0.5 → v0.6).** `Op::Scan`/`Op::ScanPlaceholder` are the first sub-graph-carrying primitive — this shifts the section's character claim ("no generic opaque/Custom node"; a body-region is a new structural kind). Add a subsection describing the `inputs = [init_carry, xs.., consts.., body_new_carry, body_y]` encoding, the single-carry-v1 model, the 2-slot bundle, and `early_exit` as a Phase-2 field.

- [ ] **Step 2: `04-optimization.md` — MINOR bump (v0.8 → v0.9).** The DecompositionMap / cost-from-decompose gains a `Scan` entry: `Op::Scan` is a terminal in the base map (no `LoweringRule`); cost comes from the re-fused/native SSM arm, **not** the `unroll_scan` explosion (the mis-pricing risk the design flags).

- [ ] **Step 3: `08-pattern-harvest.md` — MINOR bump (v0.3 → v0.4).** The G3 basis gap closes: `decompose` is now total over genuine primitives for the whole fused-op set.

- [ ] **Step 4: `12-multi-output.md` — MINOR bump (v1.1 → v1.2).** Fix the stale "selective_scan (and ssd_chunk_scan) decompose panics" prose at `:105` **and** the matching parenthetical in the `:3` Status line — both now decompose to `Op::Scan` (a 2-slot bundle preserved through decompose).

- [ ] **Step 5: `14-lifecycle.md` — MINOR bump (v0.6 → v0.7).** Fix the stale "three current panicking decomposes ... are bugs" prose at `:260` — all three (`nf4_matmul`, `flash_attn`, `selective_scan`/`ssd_chunk_scan`) now carry real recipes.

- [ ] **Step 6: `10-decisions-log.md` — new entry (2026-07-15).** Use the 2026-07-03 entry (`:395–414`) as the structural template. State that G3 is **closed** by `Op::Scan`; forward-reference the 2026-06-20 "higher-order Scan for SSMs" named exception (`:336`) and the 2026-07-03 closing paragraph (`:406`) as now-fulfilled. Note explicitly that **flash_attn's symbolic-`k_len` gap remains separately open** (so readers don't assume all basis gaps closed). Note Phase 2 (early-exit mechanism + BPTT differentiability + Hopfield consumer) is a separate spec.

- [ ] **Step 7: `ROADMAP.md`** — advance the frontier note to "G3 closed; `Op::Scan` Phase 1 shipped (SSM re-decompose); Phase 2 = early-exit + differentiability + Hopfield."

- [ ] **Step 8: Commit.**

```bash
git add docs/architecture/ ROADMAP.md
git commit -m "docs(arch): Op::Scan closes G3 — 03-ir MAJOR + 04/08/12/14 MINOR + decisions-log (C10)"
```

---

## Self-review notes (coverage against the spec)

- **Spec component → task:** C1 → Task 1; C2 → Task 2; C3 → Task 3; C4 → Task 4; **C5 → Task 4** (Op::Scan terminal, `op_scan_is_a_terminal_in_the_base_map`; **no `LoweringRule`** per the locked design — a deliberate divergence from the design-spec's C5 wording); C6 → Task 5; C7 → Task 8; C8 → Tasks 6 + 7; C9 → Tasks 6 + 7 (flip `selective_scan` gap test; new `ssd_chunk_scan` gap test + stale-doc fix); C10 → Task 9.
- **Placeholder / early_exit:** `ScanPredicate` is a constructible unit struct (never `Some` on a live path but usable in the `unroll_scan` guard test, Task 4); `Op::Scan.early_exit` folds `is_some()` into `op_key` (Task 2) so a future Phase-2 predicate cannot silently collide; the guard returns `Err` (Task 4), never a panic.
- **Type-name consistency (verified across all tasks):** `ScanEmit { All, Final }`, `ScanRole { Carry, Elem }`, `ScanPredicate`, `Op::Scan { n_xs, bound, emit, early_exit }`, `Op::ScanPlaceholder { role, index }`, `unroll_scan(graph, scan_id, steps) -> Result<(NodeId, NodeId), fuel_ir::Error>`, `Tensor::scan(&self, xs, consts, body_new_carry, body_y, bound, emit) -> Result<Tensor, fuel_ir::Error>` — spelled identically in every task and code block.
- **The C2 linchpin:** the correctness relies on body-exits being the **last two `inputs`** (Task 1 doc + Task 3 builder + Tasks 6/7 decompose all obey this), which is what lets `base_map_hash`'s existing input-recursion hash the body with only a params-only `op_key` arm (Task 2). The born-red evidence is `op_key` returning `None` for `Op::Scan` before the arm.
- **Never-workspace-wide cargo; one invocation at a time; CPU-only gates** — every command is `-p fuel-graph` / `-p fuel-core` / `-p fuel-dispatch`; the only GPU step (Task 8 Step 3) is explicitly optional/local.
- **Confirm-against-real-code notes (implementer must verify at write-time):** `Graph::layout`/`reachable`-set accessors (Task 1); `compose_bundle` `Ok`-tuple shape + `OutputViewSpec::contiguous` (Task 3); `Op::Slice`/`Squeeze`/`Unsqueeze`/`Concat` field spellings (Task 4); `Tensor { graph, id }` same-crate field access in tests (Tasks 3–4); `const_f32_like`/`realize_f32`/`realize_one_as`/`LazyTensor::ssd_chunk_scan` signatures (Tasks 6–8); the exact SSM broadcast/reshape shape algebra (Tasks 6–7); constitution version headers (Task 9).

## Genuine ambiguities flagged (could not fully resolve from spec + material + locked design)

1. **Backward differentiability (Task 1 Step 6).** The design's C6 says "wire `BackwardKind::Decompose` as `Op::Scan`'s natural default." The material's C6 reader **verified** that `BackwardKind` is *never read anywhere* — the backward walk is a hand-written exhaustive match that panics for unwired ops (`QMatMul`/`PagedAttn`), and even the one live `BackwardKind::Decompose` op (`FUSED_SOFTMAX_CROSS_ENTROPY`) has no branch and would panic. There is no generic decompose-backward mechanism to hook into, and building one is out of Phase-1 scope (BPTT is explicitly Phase 2). **Resolution:** Task 1 wires a clean `NotDifferentiable` **panic** for `Op::Scan` (mirroring the surrounding `QMatMul` precedent in that infallible `-> GradMap` walk) and an inert drop for `Op::ScanPlaceholder`, and adds **no** arm to `dispatch_gradient`. This honors "SSM stays `NotDifferentiable` in Phase 1" and avoids inventing plumbing. Flagging because it diverges from the design's literal "wire `Decompose`" wording.

2. **SSM `decompose` multi-output preservation + full-seqlen F64 parity (Tasks 6–7).** Both `decompose`s must return an `Op::Scan` bundle whose two slots exactly match the SSM's `output_views` so the existing `Op::View(0)/(1)` consumers keep working, AND the general multi-token unroll must match the CPU kernel's F64-accumulate to a sabotage-calibrated tolerance. Phase-1 gap tests are validated at the `T=1`/size-1 fixture (where F32≡F64 and stacking is trivial); the multi-token general case is validated by Task 8's parity test with an F64-cast contingency. The **exact broadcast/reshape shape algebra** for the affine body over general `[batch, seqlen, dim, dstate]` is left as implementer detail (gated by Task 8), because writing it out without a live compiler risks fabricating wrong shapes. This is the highest-complexity, highest-risk part of the plan; flagged so the executor budgets for it and does not treat Tasks 6–7 as mechanical.

3. **C7 non-regression contingency (Task 8).** The plan asserts (with strong grep evidence: `capability_gated_rules`/`default_rules` are not wired into `fuel-dispatch`'s realize path; `PipelinedExecutor::realize` dispatches `Op::Fused` directly via the kernel table) that changing `decompose` cannot regress the executed path. If Task 8's non-regression test unexpectedly goes RED (i.e. some optimize pass *does* lower kerneled SSM ops to `Op::Scan` before realize), the fallback is to add a re-fuse `FusionRule` (`Op::Scan{affine body}` → `Op::Fused(SELECTIVE_SCAN)`) gated behind `has_kernel` — a materially larger sub-task. Flagged as a low-probability-but-high-cost contingency the executor should watch for at Task 8 Step 2.

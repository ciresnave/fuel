# Op::Scan Phase 2 (early-exit + differentiability + Hopfield) Implementation Plan

> **For agentic workers:** execute task-by-task, TDD. For each task: write the failing test → run it and observe RED → implement → run GREEN → commit. Check off each step. Do not skip the RED observation. Do not batch tasks.

Design spec: [`docs/superpowers/specs/2026-07-15-op-scan-phase2-early-exit-differentiability-hopfield-design.md`](../specs/2026-07-15-op-scan-phase2-early-exit-differentiability-hopfield-design.md) — read it first. This plan is grounded against the shipped Phase‑1 code on branch `op-scan-phase1` (2026‑07‑16); every file:line below was read against that tree.

## Goal

Turn the three Phase‑1 `Op::Scan` placeholders into working mechanism and prove them with one real non‑SSM consumer:

1. **early‑exit** — a predicate‑over‑carry carried on `Op::Scan` (extra trailing input when `early_exit = Some`), evaluated at the realize barrier by a host‑driven step loop that stops early (a data‑dependent iteration count under a static capacity `bound`).
2. **differentiability** — a *lower‑`Op::Scan`‑to‑primitives‑then‑differentiate* pre‑walk pass wired into `Tensor::backward`; `BackwardKind` is dead metadata, so the mechanism is **built**, not toggled. Flip `selective_scan`/`ssd_chunk_scan` to differentiable via their existing `Op::Scan` decompose.
3. **Modern Hopfield** — `ξ ← softmax(β·ξ·Xᵀ)·X`, `carry = ξ`, `early_exit = ‖Δξ‖ < ε`, `emit = Final`, executed entirely through the unroll (forward: step driver; backward: unroll pre‑pass). No `Op::Scan` native kernel.

## Architecture

Everything runs over primitives that already have kernels (matmul, softmax‑via‑`Op::Fused`, elementwise, slice, concat, reduce). No `Op::Scan` kernel is added; the slot‑1/`last_state` OOB blocker stays out of scope. Two crates: `fuel-graph` (IR, builders, `unroll_scan`, `backward`, registry) and `fuel-core` (realize bridge, the step driver, the Hopfield module, numeric tests). CPU only.

The body of an `Op::Scan` is encoded as the node's own trailing inputs:
```
early_exit = None :  inputs = [ init_carry, xs_0..xs_{n_xs-1}, consts.., body_new_carry, body_y ]
early_exit = Some :  inputs = [ init_carry, xs_0..xs_{n_xs-1}, consts.., body_new_carry, body_y, pred_exit ]
```
`pred_exit` is a scalar‑`U8` sub‑DAG over the carry; because it is a trailing input, `base_map_hash`/`topo_order_multi`/reachability see it for free. `ScanPredicate` stays a unit marker — its `Some`‑ness signals "peel one extra trailing input".

## Tech Stack

Rust (edition 2024, toolchain 1.96). `fuel-graph` and `fuel-core` are the only crates touched. Backend for realize/tests: `fuel_cpu_backend`. Test framework: built‑in `#[test]`, run per‑crate.

## Global Constraints (binding — copy of the working agreement)

- **`-p <crate>` builds, NEVER workspace‑wide.** `tensor-tools` has a standing `Device::Cpu` break and is a default member, so bare `cargo check`/`cargo test` at the root fails. Always `cargo test -p fuel-graph` / `cargo test -p fuel-core`.
- **ONE cargo invocation at a time.** The build‑dir lock serializes; parallel invocations thrash. Long builds: background + wait.
- **Run cargo FOREGROUND.** A subagent deadlocks waiting on its own backgrounded cargo job (bg notifications reach the main loop, not sub‑subagents). Run cargo in the foreground.
- **TDD, born‑red.** Write the failing test first, run it, *observe* the RED (a compile failure counts only when it is the test's assertion target, not a typo), then implement to green. A behavior change ships with the test that exercises it, and that test must have been observed to run.
- **Never panic on production paths.** `Result` from day one; validation returns typed `Err`, never `.unwrap()`/`.expect()` on a production path. (Tests may `.expect()`.)
- **Validate at graph‑build time.** Every check that *can* run at build time *must* — no `try_*` siblings.
- **Docs are part of the change.** A material change updates the relevant `docs/architecture/` section + `ROADMAP.md` frontier in the same change (Task 9).

## Key file:line anchors (verified 2026‑07‑16)

- `fuel-graph/src/lib.rs`: `Op::Scan` def @ **1134**; `ScanPredicate` @ **1174**; `Tensor::scan` builder @ **5157‑5225**; `Tensor::backward` @ **7237** (topo @ 7242, dispatch_gradient call @ 7275); `Op::Fused` backward arm @ **9132** (final `else` panic @ **9531‑9538**); `Op::Scan` backward panic arm @ **9578‑9588**; `Op::ScanPlaceholder` inert @ **9589**; `Op::View`/`ViewOwned` backward arm @ **9595‑9643**; `GradMap` return @ **9900**; `Tensor::from_existing` @ **2847**; `topo_order` @ **80**, `topo_order_multi` @ **96**; `Graph::rewrite_input` (pub(crate)) @ **2029**; module fns `push_node`/`node_shape`/`node_dtype`/`accumulate_grad`/`build_ones`/`build_filled_const` used throughout backward.
- `fuel-graph/src/scan.rs`: `unroll_scan` @ **16**; param read + `early_exit = Some → Err` @ **28‑46**; layout parse @ **63‑67**; consts slice @ **65**; pre‑mutation placeholder validation @ **70‑96**; per‑step slice/squeeze @ **120‑142**; `clone_body_node` @ **185‑211**; `unroll_scan_rejects_early_exit_some` test @ **259‑264**.
- `fuel-graph/src/grad.rs`: `dispatch_gradient` @ **62‑79** (Add/Mul/Relu/comparisons/Where).
- `fuel-graph/src/registry.rs`: `FusedOpEntry.decompose` fn ptr @ **112**; `FusedOpEntry.backward` @ **114**; `BackwardKind` enum @ **658**; `default_registry()` @ **1027**; lookup `default_registry().entry(id) -> Option<&FusedOpEntry>`.
- `fuel-graph/src/registry/selective_scan.rs`: `backward: BackwardKind::NotDifferentiable` @ **123**; "Why NotDifferentiable" doc @ **92‑100**; `decompose` @ **230** (builds `Op::Scan { n_xs: 4, bound: seqlen, emit: All, early_exit: None }` @ 377‑382, returns permuted `y` @ 398‑413).
- `fuel-graph/src/registry/ssd_chunk_scan.rs`: `backward` @ **110**; doc @ **83‑89**; `decompose` @ **~209**.
- `fuel-graph/src/opt.rs`: consumer‑edge rewrite loop (the mirror for the pre‑pass) @ **334‑352**; `base_map_hash` @ **399**; `lower_to_base_map` @ **364**.
- `fuel-core/src/lib.rs`: module list (add `pub mod hopfield;`) @ ~**71**.
- `fuel-core/src/lazy.rs`: `LazyTensor::from_f32` @ **69**, `const_f32_like` @ **138**, `realize_f32` @ **1474**, `selective_scan` @ **1088**, `ssd_chunk_scan` @ **1068**, `backward` @ **4194**; SSM parity tests @ **2070**, **2131**, **2260**, **2400**.
- `fuel-core/src/pipelined_bridge.rs`: `realize_one_as::<T>` @ **163**, `realize_one_as_with_initial::<T>` @ **279**.

---

## Task 1 — `unroll_scan` accepts `early_exit = Some` (peel + ignore predicate)

Opens the Phase‑2 seam: `unroll_scan` must parse the 3‑trailing layout and produce the full‑`bound` primitive unroll, ignoring the predicate (the BPTT/oracle path). This is the prerequisite for both the step driver (Task 3/4) and the backward pre‑pass on early‑exit scans (Task 8).

**Files:**
- Modify `fuel-graph/src/scan.rs`: delete the `if has_exit { return Err(... Phase-2 mechanism not implemented ...) }` block @ **42‑46**; generalize the trailing‑input parse @ **57‑67**; replace the stale `unroll_scan_rejects_early_exit_some` test @ **259‑264**.

**Interfaces:**
- Consumes/Produces: `pub fn unroll_scan(graph: &mut Graph, scan_id: NodeId, steps: usize) -> std::result::Result<(NodeId, NodeId), fuel_ir::Error>` — signature unchanged; behavior for `early_exit = Some` changes from `Err` to a valid unroll that peels `pred_exit` from the tail and ignores it.

**Steps:**

- [ ] **Write the failing test.** Replace the `unroll_scan_rejects_early_exit_some` test (`scan.rs:259‑264`) with:
```rust
#[test]
fn unroll_scan_early_exit_some_peels_predicate_and_unrolls() {
    // early_exit = Some layout: [carry, consts=[thr], body_new_carry, body_y, pred_exit].
    // unroll must PEEL pred_exit, IGNORE it, and emit a 3-step Concat with no scan nodes.
    let graph = Arc::new(RwLock::new(Graph::new()));
    let scan = {
        let mut g = graph.write().unwrap();
        let s = Shape::from_dims(&[1]);
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let thr   = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let hole  = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let nc    = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
        // predicate sub-DAG over the post-step carry (ignored by unroll).
        let pred  = g.push(Node { op: Op::Ge, inputs: vec![nc, thr], shape: s.clone(), dtype: DType::U8 });
        g.push(Node {
            op: Op::Scan { n_xs: 0, bound: 3, emit: ScanEmit::All, early_exit: Some(ScanPredicate) },
            inputs: vec![carry, thr, nc, nc, pred], // consts=[thr], new_carry=nc, y=nc, pred_exit=pred
            shape: Shape::from_dims(&[3, 1]),
            dtype: DType::F32,
        })
    };
    let (ys, _carry) = {
        let mut g = graph.write().unwrap();
        unroll_scan(&mut g, scan, 3).expect("unroll must peel + ignore the predicate")
    };
    let g = graph.read().unwrap();
    assert!(matches!(g.node(ys).op, Op::Concat { .. }), "emit=All ys root should be Concat");
    assert_eq!(g.node(ys).inputs.len(), 3, "one input per step");
    let reachable = crate::topo_order_multi(&g, &[ys]);
    assert!(!reachable.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
        "unrolled graph must contain no Scan/ScanPlaceholder nodes");
}
```
- [ ] **Run RED:** `cargo test -p fuel-graph unroll_scan_early_exit_some_peels_predicate_and_unrolls`. Expect the `expect("unroll must peel...")` to fire — current `unroll_scan` returns `Err` for `early_exit = Some` (scan.rs:42‑46).
- [ ] **Implement.** In `unroll_scan`:
  - Delete the `if has_exit { return Err(...) }` block (scan.rs:42‑46).
  - Replace the min‑length guard + layout parse (scan.rs:57‑67) with a trailing count that depends on `has_exit`:
```rust
let n_trailing = if has_exit { 3 } else { 2 }; // body_new_carry, body_y, [pred_exit]
if inputs.len() < 1 + n_xs + n_trailing {
    return Err(fuel_ir::Error::Msg(format!(
        "unroll_scan: malformed Op::Scan inputs — need >= {} (init_carry + n_xs={n_xs} + {n_trailing} trailing), got {}",
        1 + n_xs + n_trailing, inputs.len(),
    )).bt());
}
let init_carry = inputs[0];
let xs: Vec<NodeId> = inputs[1..1 + n_xs].to_vec();
let consts: Vec<NodeId> = inputs[1 + n_xs..inputs.len() - n_trailing].to_vec();
let body_new_carry = inputs[inputs.len() - n_trailing];
let body_y = inputs[inputs.len() - n_trailing + 1];
// pred_exit = inputs[inputs.len() - 1] when has_exit — intentionally NOT read; the
// build-time backward unroll differentiates the full static `bound` and ignores the
// runtime early-exit predicate (spec C3 "static-horizon note").
```
  Everything downstream (placeholder validation @ 70‑96, xs‑dim validation @ 98‑115, the per‑step loop) is unchanged.
- [ ] **Run GREEN:** `cargo test -p fuel-graph unroll_scan` (runs the new test + the existing `unroll_scan_*` suite — the `emit=All`, malformed‑short, elem‑out‑of‑range, and nxs‑positive tests must all stay green; the short‑input test still hits the `< 1 + n_xs + n_trailing` guard).
- [ ] **Commit:** `feat(ir): unroll_scan peels + ignores early_exit predicate (Phase 2 C1 seam)`.

**Deliverable:** `unroll_scan` unrolls an `early_exit = Some` scan to full‑`bound` primitives, ignoring the predicate.

---

## Task 2 — `Tensor::scan_until` builder + build‑time validation (C1)

The `Result`‑returning sibling of `Tensor::scan` that carries the predicate as the extra trailing input and validates it at build time.

**Files:**
- Modify `fuel-graph/src/lib.rs`: add `pub fn scan_until(...)` immediately after `Tensor::scan` (after line **5225**).
- Add tests to the existing `#[cfg(test)]` module in `fuel-graph/src/scan.rs` (it already imports `Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole`; add `use crate::Tensor;` and a cpu‑device helper).

**Interfaces:**
- Produces:
```rust
pub fn scan_until(
    &self,                 // init_carry
    xs: &[Tensor],
    consts: &[Tensor],     // MUST include every const referenced by the body OR the predicate
    body_new_carry: &Tensor,
    body_y: &Tensor,
    pred_exit: &Tensor,    // scalar U8 sub-DAG over the carry
    bound: usize,
    emit: ScanEmit,
) -> std::result::Result<Tensor, fuel_ir::Error>
```
- Consumes: `topo_order_multi` (for the predicate‑placeholder check), `fuel_ir::storage::compose_bundle`, `Graph::set_output_views` — all as `Tensor::scan` uses them.

**Steps:**

- [ ] **Write the failing test.** In `scan.rs` tests, add a cpu‑device helper (mirror `grad.rs:216`) and:
```rust
fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_backend_contract::DynBackendDevice> {
    static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_backend_contract::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}

#[test]
fn scan_until_builds_early_exit_node_hashes_distinctly_and_validates() {
    use crate::Tensor;
    // init_carry [1]; body new_carry = carry*2; consts include threshold.
    let init = Tensor::from_f32(vec![1.0f32], Shape::from_dims(&[1]), cpu_dev());
    let graph = init.graph().clone();
    // Build the shared body + predicate at graph level, wrap as Tensor handles.
    let (nc, thr, pred_ok) = {
        let mut g = graph.write().unwrap();
        let s = Shape::from_dims(&[1]);
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let nc   = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
        let thr  = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let pred = g.push(Node { op: Op::Ge, inputs: vec![nc, thr], shape: s.clone(), dtype: DType::U8 });
        (nc, thr, pred)
    };
    let nc_t   = Tensor::from_existing(graph.clone(), nc);
    let thr_t  = Tensor::from_existing(graph.clone(), thr);
    let pred_t = Tensor::from_existing(graph.clone(), pred_ok);

    let out = init.scan_until(&[], &[thr_t.clone()], &nc_t, &nc_t, &pred_t, 5, ScanEmit::Final)
        .expect("well-formed scan_until must build");
    // The producer node behind the emit=Final view is an Op::Scan{early_exit: Some}.
    let scan_id = { let g = graph.read().unwrap(); g.node(out.id()).inputs[0] };
    {
        let g = graph.read().unwrap();
        match &g.node(scan_id).op {
            Op::Scan { early_exit, .. } => assert!(early_exit.is_some(), "early_exit must be Some"),
            other => panic!("expected Op::Scan, got {}", other.short_name()),
        }
        // pred_exit is the LAST input (trailing), so reachability sees it.
        assert_eq!(*g.node(scan_id).inputs.last().unwrap(), pred_ok);
    }

    // base_map_hash distinctness: a scan with the SAME body but a DIFFERENT predicate hashes differently.
    let thr2 = { let mut g = graph.write().unwrap();
        g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[1]), dtype: DType::F32 }) };
    let pred2 = { let mut g = graph.write().unwrap();
        g.push(Node { op: Op::Le, inputs: vec![nc, thr2], shape: Shape::from_dims(&[1]), dtype: DType::U8 }) };
    let pred2_t = Tensor::from_existing(graph.clone(), pred2);
    let out2 = init.scan_until(&[], &[Tensor::from_existing(graph.clone(), thr2)], &nc_t, &nc_t, &pred2_t, 5, ScanEmit::Final)
        .expect("second scan_until builds");
    let scan2 = { let g = graph.read().unwrap(); g.node(out2.id()).inputs[0] };
    let (h1, h2) = { let g = graph.read().unwrap();
        (crate::opt::base_map_hash(&g, scan_id), crate::opt::base_map_hash(&g, scan2)) };
    assert_ne!(h1, h2, "different predicates must hash distinctly (predicate is a trailing input)");

    // Rejection: a NON-scalar predicate is a typed Err (never a panic).
    let big = Tensor::from_f32(vec![0.0f32, 1.0], Shape::from_dims(&[2]), cpu_dev()); // wrong graph AND non-scalar
    assert!(init.scan_until(&[], &[thr_t.clone()], &nc_t, &nc_t, &big, 5, ScanEmit::Final).is_err(),
        "non-same-graph / non-scalar predicate must be a typed Err");
    // Rejection: a non-U8 predicate.
    let f32pred = { let mut g = graph.write().unwrap();
        g.push(Node { op: Op::Sqr, inputs: vec![nc], shape: Shape::from_dims(&[1]), dtype: DType::F32 }) };
    let f32pred_t = Tensor::from_existing(graph.clone(), f32pred);
    assert!(init.scan_until(&[], &[thr_t], &nc_t, &nc_t, &f32pred_t, 5, ScanEmit::Final).is_err(),
        "non-U8 predicate must be a typed Err");
}
```
- [ ] **Run RED:** `cargo test -p fuel-graph scan_until_builds_early_exit_node_hashes_distinctly_and_validates`. Expect a compile error: `scan_until` does not exist. (This compile failure is the test's target — the method is what we are adding.)
- [ ] **Implement `scan_until`** after `Tensor::scan` (lib.rs:5225). Mirror `scan` (5157‑5225) exactly, with these deltas:
  - Same‑graph check also covers `pred_exit`.
  - After the existing `bound == 0` and carry‑shape checks, validate the predicate:
```rust
// Predicate must be a scalar U8 boolean (shape [] or product == 1) — a
// convergence flag, not a per-element mask. Validate at build time.
if pred_exit.dtype() != fuel_ir::DType::U8 {
    return Err(fuel_ir::Error::Msg(format!(
        "scan_until: pred_exit must be U8 (a boolean flag), got {:?}", pred_exit.dtype(),
    )).bt());
}
let pred_dims = pred_exit.shape();
let pred_numel: usize = pred_dims.dims().iter().product();
if pred_numel != 1 {
    return Err(fuel_ir::Error::Msg(format!(
        "scan_until: pred_exit must be a scalar (shape [] or [1]/all-ones), got {:?}",
        pred_dims.dims(),
    )).bt());
}
// Every ScanPlaceholder reachable from pred_exit must be Carry{0} — the
// predicate is over the carry, never a per-step Elem slice.
{
    let g = self.graph.read().unwrap();
    for &id in &crate::topo_order_multi(&g, &[pred_exit.id]) {
        if let Op::ScanPlaceholder { role, index } = &g.node(id).op {
            let bad = matches!(role, ScanRole::Elem) || *index != 0;
            if bad {
                return Err(fuel_ir::Error::Msg(format!(
                    "scan_until: pred_exit references ScanPlaceholder{{{role:?}, {index}}} — \
                     the predicate may only read the carry (Carry, index 0)",
                )).bt());
            }
        }
    }
}
```
  - Build the 2‑slot bundle exactly as `scan` (specs = `[ys_shape, carry_shape]`, `compose_bundle`).
  - Push the node with `early_exit: Some(ScanPredicate)` and the extra trailing input:
```rust
let mut inputs: Vec<NodeId> = Vec::with_capacity(3 + xs.len() + consts.len());
inputs.push(self.id);
inputs.extend(xs.iter().map(|t| t.id));
inputs.extend(consts.iter().map(|t| t.id));
inputs.push(body_new_carry.id);
inputs.push(body_y.id);
inputs.push(pred_exit.id);            // <-- the C1 trailing predicate
// ... g.push(Node { op: Op::Scan { n_xs: xs.len(), bound, emit, early_exit: Some(ScanPredicate) }, inputs, shape: ys_shape, dtype: y_dtype }) ...
```
  - `set_output_views`, then project `view(0)`/`view(1)` by `emit` (identical to `scan`).
  - `pred_exit`, `body_new_carry`, `body_y` reachable, `Tensor` fields via `.id`. Note the doc line: **"`consts` must include every const referenced by the body OR the predicate"** (clone/unroll shares only `consts`; a predicate const outside `consts` would be re‑cloned dataless and fail at realize).
- [ ] **Run GREEN:** `cargo test -p fuel-graph scan_until` and `cargo test -p fuel-graph scan` (no regression on the base builder).
- [ ] **Commit:** `feat(ir): Tensor::scan_until — early-exit scan builder + build-time predicate validation (C1)`.

**Deliverable:** a `Result`‑returning `scan_until` that builds an `Op::Scan { early_exit: Some }` node and rejects malformed predicates at build time.

---

## Task 3 — the per‑step scan builders `parse_scan_layout` + `build_scan_step` (C2, IR half)

Expose the machinery the fuel‑core step driver (Task 4) needs: parse a scan's layout, and build one materialized step (elem slices + body clone + predicate clone with a **shared** substitution map so `pred_exit`'s reference to `body_new_carry` resolves to *this step's* new carry without double‑cloning).

**Files:**
- Modify `fuel-graph/src/scan.rs`: add `pub struct ScanLayout`, `pub struct ScanStep`, `pub fn parse_scan_layout`, `pub fn build_scan_step`. Reuse the existing private `clone_body_node` (scan.rs:185) and mirror the per‑step slice/squeeze block (scan.rs:120‑142).

**Interfaces:**
- Produces:
```rust
pub struct ScanLayout {
    pub n_xs: usize,
    pub bound: usize,
    pub emit: ScanEmit,
    pub init_carry: NodeId,
    pub xs: Vec<NodeId>,
    pub consts: Vec<NodeId>,
    pub body_new_carry: NodeId,
    pub body_y: NodeId,
    pub pred_exit: Option<NodeId>,
}
pub struct ScanStep { pub new_carry: NodeId, pub y: NodeId, pub stop: Option<NodeId> }

pub fn parse_scan_layout(graph: &Graph, scan_id: NodeId)
    -> std::result::Result<ScanLayout, fuel_ir::Error>;
pub fn build_scan_step(graph: &mut Graph, layout: &ScanLayout, t: usize, carry: NodeId)
    -> std::result::Result<ScanStep, fuel_ir::Error>;
```
- Consumes: `clone_body_node`, `Op::Slice`/`Op::Squeeze` (per‑step slicing), `HashMap`.

**Steps:**

- [ ] **Write the failing test** (in `scan.rs` tests):
```rust
#[test]
fn build_scan_step_shares_subst_so_predicate_reads_this_steps_new_carry() {
    // Scan: carry [1]; new_carry = carry*2; pred = Ge(new_carry, thr). n_xs=0, bound=4.
    let graph = Arc::new(RwLock::new(Graph::new()));
    let scan = {
        let mut g = graph.write().unwrap();
        let s = Shape::from_dims(&[1]);
        let carry = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let thr   = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let hole  = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let nc    = g.push(Node { op: Op::MulScalar(2.0), inputs: vec![hole], shape: s.clone(), dtype: DType::F32 });
        let pred  = g.push(Node { op: Op::Ge, inputs: vec![nc, thr], shape: s.clone(), dtype: DType::U8 });
        g.push(Node {
            op: Op::Scan { n_xs: 0, bound: 4, emit: ScanEmit::Final, early_exit: Some(ScanPredicate) },
            inputs: vec![carry, thr, nc, nc, pred],
            shape: Shape::from_dims(&[4, 1]),
            dtype: DType::F32,
        })
    };
    let layout = { let g = graph.read().unwrap(); crate::scan::parse_scan_layout(&g, scan).expect("layout") };
    assert!(layout.pred_exit.is_some());
    let init = layout.init_carry;
    let step = { let mut g = graph.write().unwrap(); crate::scan::build_scan_step(&mut g, &layout, 0, init).expect("step") };
    let stop = step.stop.expect("early-exit scan yields a stop node");
    // The predicate clone must reach step.new_carry (the shared post-step carry),
    // and must reach NO ScanPlaceholder (all substituted away).
    let g = graph.read().unwrap();
    let reach = crate::topo_order_multi(&g, &[stop]);
    assert!(reach.contains(&step.new_carry), "pred must read THIS step's new_carry (shared subst)");
    assert!(!reach.iter().any(|&n| matches!(g.node(n).op, Op::ScanPlaceholder { .. })),
        "no placeholders survive a materialized step");
    // The Ge's first input IS the step's new_carry — proof there was no double-clone.
    let ge_inputs = &g.node(stop).inputs;
    assert_eq!(ge_inputs[0], step.new_carry, "predicate's post-step operand is the shared new_carry");
}
```
- [ ] **Run RED:** `cargo test -p fuel-graph build_scan_step_shares_subst_so_predicate_reads_this_steps_new_carry`. Expect a compile error: `parse_scan_layout`/`build_scan_step` do not exist.
- [ ] **Implement.**
  - `parse_scan_layout`: read `Op::Scan { n_xs, bound, emit, early_exit }` (else typed `Err`); `has_exit = early_exit.is_some()`; `n_trailing = if has_exit {3} else {2}`; same min‑length guard as Task 1; slice `init_carry`/`xs`/`consts`/`body_new_carry`/`body_y`; `pred_exit = has_exit.then(|| inputs[inputs.len()-1])`.
  - `build_scan_step`: mirror `unroll_scan`'s single iteration:
```rust
pub fn build_scan_step(graph: &mut Graph, layout: &ScanLayout, t: usize, carry: NodeId)
    -> std::result::Result<ScanStep, fuel_ir::Error>
{
    // per-step elem slices: xs[i] sliced [t,t+1) on axis 0, squeezed.
    let mut elem: Vec<NodeId> = Vec::with_capacity(layout.n_xs);
    for &x in &layout.xs {
        let (x_shape, x_dtype) = { let n = graph.node(x); (n.shape.clone(), n.dtype) };
        if x_shape.dims().first().map_or(true, |&d0| d0 <= t) {
            return Err(fuel_ir::Error::Msg(format!(
                "build_scan_step: xs slice at t={t} out of range for shape {:?}", x_shape.dims())).bt());
        }
        let sliced: Vec<usize> = std::iter::once(1usize).chain(x_shape.dims().iter().skip(1).copied()).collect();
        let sl = graph.push(Node { op: Op::Slice { dim: 0, start: t, len: 1 },
            inputs: vec![x], shape: fuel_ir::Shape::from_dims(&sliced), dtype: x_dtype });
        let sq_dims: Vec<usize> = x_shape.dims().iter().skip(1).copied().collect();
        let sq = graph.push(Node { op: Op::Squeeze { dim: 0 },
            inputs: vec![sl], shape: fuel_ir::Shape::from_dims(&sq_dims), dtype: x_dtype });
        elem.push(sq);
    }
    let consts_set: std::collections::HashSet<NodeId> = layout.consts.iter().copied().collect();
    let mut subst: HashMap<NodeId, NodeId> = HashMap::new();
    // Clone body_new_carry FIRST so subst records body_new_carry -> new_carry,
    // then clone body_y and pred_exit sharing subst (spec "Predicate referencing
    // body_new_carry" — no double-clone).
    let new_carry = clone_body_node(graph, layout.body_new_carry, carry, &elem, &consts_set, &mut subst);
    let y = clone_body_node(graph, layout.body_y, carry, &elem, &consts_set, &mut subst);
    let stop = layout.pred_exit.map(|p| clone_body_node(graph, p, carry, &elem, &consts_set, &mut subst));
    Ok(ScanStep { new_carry, y, stop })
}
```
  - Note the pre‑mutation placeholder/index validation in `unroll_scan` (70‑96) is NOT re‑run here; `build_scan_step` assumes a scan that `scan_until`/decompose built. Elem out‑of‑range is guarded above; Carry index is single‑carry by construction.
- [ ] **Run GREEN:** `cargo test -p fuel-graph build_scan_step`.
- [ ] **Commit:** `feat(ir): parse_scan_layout + build_scan_step — per-step materializer for the early-exit driver (C2)`.

**Deliverable:** `parse_scan_layout` + `build_scan_step`, with the shared‑subst predicate clone verified structurally.

---

## Task 4 — the realize‑barrier step driver `drive_scan_until_final_f32` (C2, executor half)

The host‑driven step loop: realize each step's carry + predicate, feed the carry forward as a fresh const, stop when the predicate fires. `emit = Final` returns the fixed‑point carry and the runtime step count.

**Integration‑site decision (spec Open question resolved):** the driver is a **standalone `fuel-core` function** invoked explicitly (not auto‑wired into `realize_f32`). It realizes each step via `pipelined_bridge::realize_one_as`, seeding the next step's carry as a fresh `Op::Const` (`Tensor::const_f32_like`). Rationale: it composes with — rather than mutates — the pipelined/plan‑once executor, keeps plan‑once caching for non‑scan graphs pristine, and is fully CPU‑testable. Auto‑routing `realize_f32` to the driver (detect a target that is a `View` of an `early_exit` scan) is a documented follow‑up gated on the plan‑caching interaction (spec Risk) and is **out of scope** here.

**Files:**
- Create `fuel-core/src/hopfield.rs` (the driver lands here; the Hopfield builder joins it in Task 7).
- Modify `fuel-core/src/lib.rs`: add `pub mod hopfield;` (near line 71).

**Interfaces:**
- Produces:
```rust
pub fn drive_scan_until_final_f32(
    graph: &std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>,
    scan_id: fuel_graph::NodeId,
    device: &crate::Device,
) -> Result<(Vec<f32>, usize), fuel_ir::Error>; // (final_carry_bytes, runtime_step_count)
```
- Consumes: `fuel_graph::scan::{parse_scan_layout, build_scan_step}` (Task 3), `crate::pipelined_bridge::realize_one_as::<f32>` and `::<u8>`, `fuel_graph::Tensor::{from_existing, const_f32_like}`.

**Steps:**

- [ ] **Write the failing test** (in `hopfield.rs` `#[cfg(test)]`):
```rust
use crate::{Device, hopfield::drive_scan_until_final_f32};
use fuel_graph::{Graph, Node, Op, ScanEmit, ScanPredicate, ScanRole, Tensor};
use fuel_ir::{DType, Shape};
use std::sync::{Arc, RwLock};

// Build carry[1]=0; new_carry = carry + 1; pred = Ge(new_carry, thr). Deterministic:
// after step t, new_carry = t+1; pred fires at t = thr-1 -> count = thr.
fn counting_scan(bound: usize, thr: f32) -> (Arc<RwLock<Graph>>, fuel_graph::NodeId) {
    let init = Tensor::from_f32(vec![0.0f32], Shape::from_dims(&[1]),
        &*Device::cpu().as_dyn()); // init carry 0
    let graph = init.graph().clone();
    let scan = {
        let mut g = graph.write().unwrap();
        let s = Shape::from_dims(&[1]);
        let one = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let thr_n = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let nc   = g.push(Node { op: Op::Add, inputs: vec![hole, one], shape: s.clone(), dtype: DType::F32 });
        let pred = g.push(Node { op: Op::Ge, inputs: vec![nc, thr_n], shape: s.clone(), dtype: DType::U8 });
        g.push(Node {
            op: Op::Scan { n_xs: 0, bound, emit: ScanEmit::Final, early_exit: Some(ScanPredicate) },
            inputs: vec![init.id(), one, thr_n, nc, nc, pred], // consts=[one, thr], new=nc, y=nc, pred
            shape: Shape::from_dims(&[bound, 1]),
            dtype: DType::F32,
        })
    };
    // Seed the two const values (one=1.0, thr) into storage via const_f32_like reuse:
    // rebuild them as data consts so realize finds their bytes.
    // (Simpler: build one/thr through const_f32_like, see NOTE in implement step.)
    (graph, scan)
}
```
  **NOTE for the test author:** the two body/predicate constants (`one`, `thr`) must carry data at realize. Build them with `Tensor::from_existing(graph.clone(), init.id()).const_f32_like(vec![1.0], [1])` / `...const_f32_like(vec![thr], [1])` and use the returned `.id()` in the scan inputs, instead of raw `Op::Const` leaves. Rewrite `counting_scan` to use `const_f32_like` for `one` and `thr_n`. Then:
```rust
#[test]
fn driver_stops_at_predicate_step_and_returns_that_carry() {
    let dev = Device::cpu();
    let (graph, scan) = counting_scan(/*bound*/ 10, /*thr*/ 3.0);
    let (carry, count) = drive_scan_until_final_f32(&graph, scan, &dev).expect("driver");
    assert_eq!(count, 3, "predicate Ge(new_carry, 3) fires at step index 2 -> count 3");
    assert!((carry[0] - 3.0).abs() < 1e-5, "returned carry is the step-3 value, got {}", carry[0]);
}

#[test]
fn driver_runs_to_bound_when_predicate_never_fires() {
    let dev = Device::cpu();
    let (graph, scan) = counting_scan(/*bound*/ 6, /*thr*/ 999.0);
    let (carry, count) = drive_scan_until_final_f32(&graph, scan, &dev).expect("driver");
    assert_eq!(count, 6, "non-converging predicate runs to bound (no infinite loop)");
    assert!((carry[0] - 6.0).abs() < 1e-5);
}
```
- [ ] **Run RED:** `cargo test -p fuel-core driver_stops_at_predicate_step_and_returns_that_carry`. Expect compile error: module/function absent.
- [ ] **Implement `drive_scan_until_final_f32`:**
```rust
pub fn drive_scan_until_final_f32(
    graph: &Arc<RwLock<Graph>>,
    scan_id: fuel_graph::NodeId,
    device: &crate::Device,
) -> Result<(Vec<f32>, usize), fuel_ir::Error> {
    let layout = { let g = graph.read().unwrap(); fuel_graph::scan::parse_scan_layout(&g, scan_id)? };
    if !matches!(layout.emit, ScanEmit::Final) {
        return Err(fuel_ir::Error::Msg(
            "drive_scan_until_final_f32: only emit=Final is supported (emit=All valid-count \
             buffer is out of scope for Phase 2)".into()).bt());
    }
    let carry_shape = { let g = graph.read().unwrap(); g.node(layout.init_carry).shape.clone() };
    let mut carry_id = layout.init_carry;
    let mut last: Vec<f32> = crate::pipelined_bridge::realize_one_as::<f32>(graph, carry_id, device)
        .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize init_carry: {e}")).bt())?;
    let mut count = 0usize;
    for t in 0..layout.bound {
        let step = { let mut g = graph.write().unwrap();
            fuel_graph::scan::build_scan_step(&mut g, &layout, t, carry_id)? };
        let nc = crate::pipelined_bridge::realize_one_as::<f32>(graph, step.new_carry, device)
            .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize step {t}: {e}")).bt())?;
        let stop = match step.stop {
            Some(stop_id) => {
                let b = crate::pipelined_bridge::realize_one_as::<u8>(graph, stop_id, device)
                    .map_err(|e| fuel_ir::Error::Msg(format!("drive_scan_until: realize predicate {t}: {e}")).bt())?;
                b.first().copied().unwrap_or(0) != 0
            }
            None => false, // no predicate: run to bound
        };
        count = t + 1;
        last = nc.clone();
        // Feed the realized carry forward as a fresh data const so the next step's
        // realize is O(1 step), not O(t) (breaks the recurrent data dependency).
        carry_id = Tensor::from_existing(graph.clone(), layout.init_carry)
            .const_f32_like(nc, carry_shape.clone()).id();
        if stop { break; }
    }
    Ok((last, count))
}
```
  - `realize_one_as::<u8>` reads the U8 predicate byte; `realize_one_as` is generic over `T: Pod` (`u8` qualifies). No dtype guard on that path.
  - Imports at file top: `use std::sync::{Arc, RwLock}; use fuel_graph::{Graph, ScanEmit, Tensor}; use fuel_ir;`.
- [ ] **Run GREEN:** `cargo test -p fuel-core driver_` (both driver tests).
- [ ] **Commit:** `feat(core): drive_scan_until_final_f32 — realize-barrier early-exit step driver (C2)`.

**Deliverable:** a step driver that stops at the predicate's fire step (runtime count == k) and runs to `bound` when it never fires — no infinite loop.

---

## Task 5 — the lower‑then‑differentiate backward pre‑pass + wiring (C3/C4/C5, IR)

Build the generic decompose‑backward hook: before `Tensor::backward` takes its topo order, unroll every reachable `Op::Scan` to `bound` primitives and decompose the two SSM `Op::Fused` ids into their `Op::Scan` recipes, rewiring consumers so the reverse walk sees only differentiable primitives. Convert the `Op::Scan` backward arm to a defensive guard, and flip the SSM `BackwardKind` metadata + docs.

**Files:**
- Modify `fuel-graph/src/lib.rs`: add a module‑level `fn lower_scans_for_backward(...)` (near the backward helpers ~9948); call it at the top of `Tensor::backward` (7237‑7256); reword the `Op::Scan` panic arm (9578‑9588).
- Modify `fuel-graph/src/registry/selective_scan.rs`: `backward: BackwardKind::NotDifferentiable` → `BackwardKind::Decompose` (line 123); update the "Why NotDifferentiable" doc (92‑100).
- Modify `fuel-graph/src/registry/ssd_chunk_scan.rs`: same flip (line 110) + doc (83‑89).

**Interfaces:**
- Produces (private): `fn lower_scans_for_backward(graph: &SharedGraph, root: NodeId) -> NodeId` — decomposes SSM fused nodes + unrolls scans reachable from `root`, rewires consumers, returns the (possibly remapped) root.
- Consumes: `unroll_scan` (Task 1), `default_registry().entry(fid).decompose`, `Graph::rewrite_input`, `topo_order`.

**Steps:**

- [ ] **Write the failing test** (in `lib.rs` tests, or a `scan.rs` test module — must be a `fuel-graph` test since it calls `backward()`). This is a *structural* born‑red (fuel‑graph can't realize): build a hand‑made affine `Op::Scan`, call `backward()`, assert no panic + gradients exist + the scan was lowered.
```rust
#[test]
fn backward_lowers_op_scan_and_no_longer_panics() {
    use crate::{Graph, Node, Op, ScanEmit, ScanRole, Tensor};
    use fuel_ir::{DType, Shape};
    // Affine scan: carry[1]; consts a,b; new_carry = a*carry + b; emit=Final. bound=3.
    let init = Tensor::from_f32(vec![1.0f32], Shape::from_dims(&[1]), cpu_dev());
    let a = Tensor::from_existing(init.graph().clone(), init.id()).const_f32_like(vec![0.5f32], Shape::from_dims(&[1]));
    let b = Tensor::from_existing(init.graph().clone(), init.id()).const_f32_like(vec![0.1f32], Shape::from_dims(&[1]));
    let graph = init.graph().clone();
    let (nc,) = {
        let mut g = graph.write().unwrap();
        let s = Shape::from_dims(&[1]);
        let hole = g.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let ac   = g.push(Node { op: Op::Mul, inputs: vec![a.id(), hole], shape: s.clone(), dtype: DType::F32 });
        let nc   = g.push(Node { op: Op::Add, inputs: vec![ac, b.id()], shape: s.clone(), dtype: DType::F32 });
        (nc,)
    };
    let nc_t = Tensor::from_existing(graph.clone(), nc);
    let out = init.scan(&[], &[a.clone(), b.clone()], &nc_t, &nc_t, 3, ScanEmit::Final).expect("scan");
    // backward() must NOT panic (Phase 1 arm panics here) and must yield a grad for init_carry.
    let grads = out.backward();
    let g_init = grads.get(&init).expect("gradient for init_carry");
    // The gradient's subgraph must contain no Op::Scan/ScanPlaceholder (proof of lowering).
    let g = graph.read().unwrap();
    let reach = crate::topo_order_multi(&g, &[g_init.id()]);
    assert!(!reach.iter().any(|&n| matches!(g.node(n).op, Op::Scan { .. } | Op::ScanPlaceholder { .. })),
        "backward must lower the scan before differentiating");
    assert!(grads.get(&a).is_some() && grads.get(&b).is_some(), "consts a,b get gradients (BPTT)");
}
```
- [ ] **Run RED:** `cargo test -p fuel-graph backward_lowers_op_scan_and_no_longer_panics`. Expect a **panic**: `"Tensor::backward: Op::Scan is not differentiable in Phase 1"` (lib.rs:9584). (A panicking test is a valid RED.)
- [ ] **Implement `lower_scans_for_backward`.** Model the consumer‑edge rewrite on `opt.rs:334‑352`.
```rust
fn lower_scans_for_backward(graph: &SharedGraph, root: NodeId) -> NodeId {
    use std::collections::HashMap;
    let mut root = root;
    // helper: apply a remap to every consumer's inputs + the root, in place.
    fn apply_remap(graph: &SharedGraph, remap: &HashMap<NodeId, NodeId>, root: &mut NodeId) {
        if remap.is_empty() { return; }
        let mut g = graph.write().unwrap();
        let n = g.len();
        for nid in 0..n {
            let ilen = g.node(NodeId(nid)).inputs.len();
            for i in 0..ilen {
                let cur = g.node(NodeId(nid)).inputs[i];
                if let Some(&new) = remap.get(&cur) { g.rewrite_input(NodeId(nid), cur, new); }
            }
        }
        if let Some(&new) = remap.get(root) { *root = new; }
    }

    // --- Pass 1: decompose SSM Op::Fused(SELECTIVE_SCAN|SSD_CHUNK_SCAN) into Op::Scan.
    {
        let ssm = [crate::registry::FusedOps::SELECTIVE_SCAN, crate::registry::FusedOps::SSD_CHUNK_SCAN];
        let targets: Vec<(NodeId, crate::registry::FusedOpId, crate::registry::FusedOpParams)> = {
            let g = graph.read().unwrap();
            crate::topo_order(&g, root).into_iter().filter_map(|id| match &g.node(id).op {
                Op::Fused(fid, params) if ssm.contains(fid) => Some((id, *fid, params.clone())),
                _ => None,
            }).collect()
        };
        let mut remap = HashMap::new();
        for (id, fid, params) in targets {
            if let Some(entry) = crate::registry::default_registry().entry(fid) {
                let y = { let mut g = graph.write().unwrap(); (entry.decompose)(&mut g, id, &params) };
                if y != id { remap.insert(id, y); }
            }
        }
        apply_remap(graph, &remap, &mut root);
    }

    // --- Pass 2: unroll every reachable Op::Scan to `bound` primitives.
    loop {
        let scans: Vec<(NodeId, usize, ScanEmit)> = {
            let g = graph.read().unwrap();
            crate::topo_order(&g, root).into_iter().filter_map(|id| match &g.node(id).op {
                Op::Scan { bound, emit, .. } => Some((id, *bound, *emit)),
                _ => None,
            }).collect()
        };
        if scans.is_empty() { break; }
        let mut remap = HashMap::new();
        for (scan_id, bound, emit) in scans {
            let (a, b) = { let mut g = graph.write().unwrap();
                match unroll_scan(&mut g, scan_id, bound) {
                    Ok(pair) => pair,
                    Err(_) => continue, // malformed scan: leave it; the C4 guard describes it.
                }
            };
            // Normalize to (slot0 = stacked_ys, slot1 = final_carry).
            let (slot0, slot1) = match emit { ScanEmit::All => (a, b), ScanEmit::Final => (b, a) };
            // Remap the scan's View{slot}/ViewOwned{slot} consumers to the unroll outputs.
            let g = graph.read().unwrap();
            for nid in 0..g.len() {
                let node = g.node(NodeId(nid));
                if let Op::View { slot } | Op::ViewOwned { slot } = node.op {
                    if node.inputs.first() == Some(&scan_id) {
                        remap.insert(NodeId(nid), if slot == 0 { slot0 } else { slot1 });
                    }
                }
            }
        }
        apply_remap(graph, &remap, &mut root);
        // A scan whose body itself contained a scan (SSM decompose does not; guard against loops):
        // loop re-scans; unrolled graphs contain no Op::Scan, so this terminates in <= 2 iters.
    }
    root
}
```
  - Wire into `Tensor::backward` (lib.rs:7237): after `let graph_handle = self.graph.clone();`, add `let root = lower_scans_for_backward(&graph_handle, self.id);` and use `root` in place of `self.id` for the topo order (7242), the root shape/dtype read (7248‑7252), and the `upstream.insert(root, ones_id)` seed (7256).
  - Reword the `Op::Scan` backward arm (9578‑9588) — keep it a `panic!` (backward is infallible) but as a defensive internal‑error guard:
```rust
Op::Scan { .. } => {
    // Unreachable on any graph that went through backward(): the C3 pre-pass
    // (lower_scans_for_backward) unrolls every Op::Scan to primitives before the
    // topo order is taken. Reaching here means the pre-pass did not run.
    panic!("Tensor::backward: Op::Scan reached the reverse walk un-lowered — \
            the C3 lower_scans_for_backward pre-pass did not run (internal bug).");
}
```
  - `Op::ScanPlaceholder` arm (9589) stays inert.
- [ ] **Flip SSM `BackwardKind` + docs (C5).** In `selective_scan.rs:123` and `ssd_chunk_scan.rs:110` change `NotDifferentiable` → `Decompose`. Rewrite each "Why `BackwardKind::NotDifferentiable`" doc block to note: differentiability comes from the Phase‑2 `lower_scans_for_backward` pre‑pass (decompose → `Op::Scan` → `unroll_scan` → node‑general autograd); `BackwardKind::Decompose` documents intent (the field is not read by the walk — the pre‑pass replaces these nodes before the walk).
- [ ] **Run GREEN:** `cargo test -p fuel-graph backward_lowers_op_scan_and_no_longer_panics`. Then a no‑regression sweep of the existing autograd suite: `cargo test -p fuel-graph backward` and `cargo test -p fuel-graph grad` (the pre‑pass is a no‑op on scan‑free graphs — one extra `topo_order` scan, no rewiring).
- [ ] **Commit:** `feat(ir): lower_scans_for_backward — BPTT via decompose+unroll pre-pass; SSM BackwardKind=Decompose (C3/C4/C5)`.

**Deliverable:** `backward()` on an `Op::Scan` graph lowers the scan and differentiates the primitives instead of panicking; SSM ops are metadata‑differentiable.

---

## Task 6 — numeric BPTT finite‑difference gradient tests (C3/C4/C5, realize‑backed)

Prove the pre‑pass gradients are correct against finite differences over the *same unrolled graph* (self‑consistent, decoupled from kernel F64 parity). These need realize, so they live in `fuel-core`.

**Files:**
- Add tests to `fuel-core/src/lazy.rs` `#[cfg(test)]` (beside the SSM parity tests @ 2070+). No production code changes — this task validates Task 5.

**Interfaces:** consumes `LazyTensor::{selective_scan, ssd_chunk_scan, backward, realize_f32}`, `fuel_graph::{Tensor, scan::unroll_scan}`, `crate::pipelined_bridge::realize_one_as`.

**Steps:**

- [ ] **Write the failing tests.**

*(a) affine scan BPTT.* Build the affine scan via `fuel_graph::Tensor::scan` at graph level, unroll to `bound` for the forward oracle + FD, and compare to `backward()` grads.
```rust
#[test]
fn affine_scan_bptt_matches_finite_difference() {
    use fuel_graph::{Tensor, ScanEmit, Node, Op, ScanRole};
    use fuel_ir::{DType, Shape};
    let dev = Device::cpu();
    // f(init, a, b): carry_{t+1} = a*carry_t + b, carry_0 = init, bound = 3, loss = carry_3.
    // Closed form: carry_3 = a^3*init + b*(a^2 + a + 1). d/dinit = a^3.
    let build = |init_v: f32, a_v: f32, b_v: f32| -> (std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>, fuel_graph::NodeId, fuel_graph::Tensor, fuel_graph::Tensor, fuel_graph::Tensor) {
        let init = Tensor::from_f32(vec![init_v], Shape::from_dims(&[1]), &*dev.as_dyn());
        let g = init.graph().clone();
        let a = Tensor::from_existing(g.clone(), init.id()).const_f32_like(vec![a_v], Shape::from_dims(&[1]));
        let b = Tensor::from_existing(g.clone(), init.id()).const_f32_like(vec![b_v], Shape::from_dims(&[1]));
        let nc = {
            let mut gw = g.write().unwrap();
            let s = Shape::from_dims(&[1]);
            let hole = gw.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
            let ac = gw.push(Node { op: Op::Mul, inputs: vec![a.id(), hole], shape: s.clone(), dtype: DType::F32 });
            gw.push(Node { op: Op::Add, inputs: vec![ac, b.id()], shape: s.clone(), dtype: DType::F32 })
        };
        let nc_t = Tensor::from_existing(g.clone(), nc);
        let out = init.scan(&[], &[a.clone(), b.clone()], &nc_t, &nc_t, 3, ScanEmit::Final).expect("scan");
        (g, out.id(), init, a, b)
    };
    // Forward value via unroll+realize (self-consistent oracle).
    let fwd = |init_v: f32, a_v: f32, b_v: f32| -> f32 {
        let (g, out_id, _i, _a, _b) = build(init_v, a_v, b_v);
        let scan_id = { let gr = g.read().unwrap(); gr.node(out_id).inputs[0] };
        let carry = { let mut gw = g.write().unwrap();
            fuel_graph::scan::unroll_scan(&mut gw, scan_id, 3).expect("unroll").0 }; // emit=Final -> selected=final_carry
        crate::pipelined_bridge::realize_one_as::<f32>(&g, carry, &dev).expect("realize")[0]
    };
    // Autograd grad w.r.t. init at (1.0, 0.5, 0.1).
    let (g, out_id, init, a, b) = build(1.0, 0.5, 0.1);
    let out = fuel_graph::Tensor::from_existing(g.clone(), out_id);
    let grads = out.backward();
    let realize_grad = |t: &fuel_graph::Tensor| crate::pipelined_bridge::realize_one_as::<f32>(&g, grads.get(t).expect("grad").id(), &dev).expect("realize grad")[0];
    let g_init = realize_grad(&init);
    let g_a = realize_grad(&a);
    let g_b = realize_grad(&b);
    // Central finite differences.
    let h = 1e-3f32;
    let fd = |dinit: f32, da: f32, db: f32| (fwd(1.0+dinit, 0.5+da, 0.1+db) - fwd(1.0-dinit, 0.5-da, 0.1-db)) / (2.0*h);
    assert!((g_init - fd(h,0.0,0.0)).abs() < 2e-2, "d/dinit: autograd {g_init} vs FD {}", fd(h,0.0,0.0));
    assert!((g_a - fd(0.0,h,0.0)).abs() < 2e-2, "d/da: autograd {g_a} vs FD {}", fd(0.0,h,0.0));
    assert!((g_b - fd(0.0,0.0,h)).abs() < 2e-2, "d/db: autograd {g_b} vs FD {}", fd(0.0,0.0,h));
}
```

*(b) selective_scan differentiability.* Reuse the tiny 1×1×1 fixture from `selective_scan_decompose_lowers_to_scan_and_matches` (lazy.rs:2070). Differentiate `y` w.r.t. `u` and finite‑difference the forward (the fused kernel forward is fine as the FD oracle here since we compare against autograd‑over‑decompose; use a loose tol).
```rust
#[test]
fn selective_scan_is_differentiable_backward_matches_fd() {
    let dev = Device::cpu();
    let fwd = |u_v: f32| -> f32 {
        let u = LazyTensor::from_f32(vec![u_v], Shape::from_dims(&[1,1,1]), &dev);
        let delta = u.const_f32_like(vec![0.5f32], Shape::from_dims(&[1,1,1]));
        let a = u.const_f32_like(vec![-1.0f32], Shape::from_dims(&[1,1]));
        let b = u.const_f32_like(vec![3.0f32], Shape::from_dims(&[1,1,1]));
        let c = u.const_f32_like(vec![4.0f32], Shape::from_dims(&[1,1,1]));
        u.selective_scan(&delta, &a, &b, &c, false).realize_f32()[0]
    };
    // Autograd at u=2.0.
    let u = LazyTensor::from_f32(vec![2.0f32], Shape::from_dims(&[1,1,1]), &dev);
    let delta = u.const_f32_like(vec![0.5f32], Shape::from_dims(&[1,1,1]));
    let a = u.const_f32_like(vec![-1.0f32], Shape::from_dims(&[1,1]));
    let b = u.const_f32_like(vec![3.0f32], Shape::from_dims(&[1,1,1]));
    let c = u.const_f32_like(vec![4.0f32], Shape::from_dims(&[1,1,1]));
    let y = u.selective_scan(&delta, &a, &b, &c, false);
    let grads = y.inner.backward(); // LazyTensor::backward delegates; or y.backward()
    let g_u_id = grads.get(u.graph_tensor()).expect("grad u").id();
    let g_u = crate::pipelined_bridge::realize_one_as::<f32>(&u.inner.graph().clone(), g_u_id, &dev).expect("realize")[0];
    let h = 1e-3;
    let fd = (fwd(2.0+h) - fwd(2.0-h)) / (2.0*h);
    assert!((g_u - fd).abs() < 5e-2, "selective_scan d/du: autograd {g_u} vs FD {fd}");
}
```
  **NOTE:** `y = 12` at `u=2` and is linear in `u` here (`y = c·d·b·u = 4·0.5·3·u = 6u`), so both grad and FD ≈ 6.0. Adjust the tol/fixture if the exact softplus‑off recurrence differs; the assertion is autograd≈FD, not a hard‑coded 6.

*(c) ssd_chunk_scan differentiability.* Mirror (b) with the `ssd_chunk_scan` 1×1×1×1 fixture from lazy.rs:2131.

- [ ] **Run RED:** `cargo test -p fuel-core affine_scan_bptt_matches_finite_difference`. Before Task 5 this panics in `backward()`; **since Task 5 is already committed, this test should instead surface any residual numeric/coverage gap** — run it and confirm it passes or shows a real discrepancy. (If Task 5/6 are done together, observe RED by temporarily reverting the Task‑5 wiring; otherwise this task is a validation gate whose RED was Task 5's structural test.)
- [ ] **Implement:** no new production code expected. If a body op lacks a backward arm, the walk panics mid‑test with the offending op name — add that op's arm (spec Risk: "every op in the unroll must have a backward arm"; the SSM/affine op sets — `Mul/Add/Exp/Abs/Neg/Log/Relu/AddScalar/MulScalar/BroadcastTo/Reshape/ReduceSumTo/Permute/Slice/Squeeze/Unsqueeze/MatMul/Concat/Sqr/Sub/Sqrt` and softmax‑via‑`Op::Fused` — are all confirmed covered in the walk).
- [ ] **Run GREEN:** `cargo test -p fuel-core selective_scan_is_differentiable ssd_chunk_scan_is_differentiable affine_scan_bptt`.
- [ ] **Commit:** `test(core): BPTT finite-difference gates — affine scan + selective_scan + ssd_chunk_scan differentiable (C3/C4/C5)`.

**Deliverable:** autograd gradients through `Op::Scan` and the two SSM ops match finite differences.

---

## Task 7 — `hopfield_retrieve` lazy module + convergence test (C6/C7)

Ship the Modern Hopfield associative‑memory retrieval as an `Op::Scan { emit: Final, early_exit: Some }` consumer, and prove it converges early via the Task‑4 driver.

**Files:**
- Modify `fuel-core/src/hopfield.rs` (created in Task 4): add `pub fn hopfield_retrieve(...)`.

**Interfaces:**
- Produces:
```rust
/// Modern (dense) Hopfield retrieval: xi <- softmax(beta * xi * X^T) * X, iterated
/// to a fixed point (carry = xi, early_exit = ||xi_new - xi|| < eps, emit = Final).
/// query: [1, d]; patterns X: [n, d]. Returns the retrieval Tensor (emit=Final view).
pub fn hopfield_retrieve(
    query: &fuel_graph::Tensor,   // [1, d], init_carry
    patterns: &fuel_graph::Tensor,// [n, d], a const
    beta: f32,
    eps: f32,
    max_iters: usize,             // = bound (capacity)
) -> std::result::Result<fuel_graph::Tensor, fuel_ir::Error>;
```
- Consumes: `Tensor::{matmul, transpose, mul_scalar, softmax_last_dim, sub, sqr, reduce_sum_to, sqrt, lt, scan_until}`; the driver `drive_scan_until_final_f32`.

**Steps:**

- [ ] **Write the failing test:**
```rust
#[test]
fn hopfield_retrieves_stored_pattern_and_exits_early() {
    use fuel_graph::Tensor;
    let dev = Device::cpu();
    // Three orthogonal-ish stored patterns [n=3, d=4].
    let x = Tensor::from_f32(
        vec![1.0,0.0,0.0,0.0,  0.0,1.0,0.0,0.0,  0.0,0.0,1.0,0.0],
        Shape::from_dims(&[3,4]), &*dev.as_dyn());
    // Query near pattern 0.
    let q = Tensor::from_existing(x.graph().clone(), x.id())
        .const_f32_like(vec![0.9, 0.2, 0.1, 0.0], Shape::from_dims(&[1,4]));
    let retrieval = crate::hopfield::hopfield_retrieve(&q, &x, /*beta*/ 8.0, /*eps*/ 1e-3, /*max_iters*/ 20)
        .expect("build hopfield retrieval");
    let scan_id = { let g = retrieval.graph().read().unwrap(); g.node(retrieval.id()).inputs[0] };
    let (xi, count) = crate::hopfield::drive_scan_until_final_f32(&retrieval.graph().clone(), scan_id, &dev)
        .expect("drive");
    // Converged to pattern 0 (dominant coordinate 0), and stopped BEFORE the capacity.
    assert!(xi[0] > 0.8 && xi[1] < 0.2 && xi[2] < 0.2, "retrieved xi should snap to pattern 0: {xi:?}");
    assert!(count < 20, "early-exit must stop before bound (converged in {count} < 20 iters)");
    assert!(count >= 1);
}
```
- [ ] **Run RED:** `cargo test -p fuel-core hopfield_retrieves_stored_pattern_and_exits_early`. Expect compile error: `hopfield_retrieve` absent.
- [ ] **Implement `hopfield_retrieve`:**
```rust
pub fn hopfield_retrieve(
    query: &fuel_graph::Tensor,
    patterns: &fuel_graph::Tensor,
    beta: f32, eps: f32, max_iters: usize,
) -> std::result::Result<fuel_graph::Tensor, fuel_ir::Error> {
    use fuel_graph::{Tensor, ScanEmit};
    if !std::sync::Arc::ptr_eq(query.graph(), patterns.graph()) {
        return Err(fuel_ir::Error::Msg("hopfield_retrieve: query and patterns must share a graph".into()).bt());
    }
    let d = { let dims = query.shape(); *dims.dims().last().ok_or_else(|| fuel_ir::Error::Msg("hopfield: query rank 0".into()).bt())? };
    let g = query.graph().clone();
    // carry placeholder xi [1, d].
    let xi = {
        let mut gw = g.write().unwrap();
        gw.push(fuel_graph::Node { op: fuel_graph::Op::ScanPlaceholder { role: fuel_graph::ScanRole::Carry, index: 0 },
            inputs: vec![], shape: fuel_ir::Shape::from_dims(&[1, d]), dtype: fuel_ir::DType::F32 })
    };
    let xi_t = Tensor::from_existing(g.clone(), xi);
    // body: logits = mul_scalar(beta)(xi @ X^T) [1,n]; s = softmax_last(logits); new = s @ X [1,d].
    let xt = patterns.transpose();                       // [d, n]
    let logits = xi_t.matmul(&xt).mul_scalar(beta as f64);
    let s = logits.softmax_last_dim();
    let new_carry = s.matmul(patterns);                  // [1, d]
    // pred: ||new - xi|| < eps  ->  Lt( sqrt(sum((new-xi)^2)), eps ) : U8 [1,1].
    let delta = new_carry.sub(&xi_t);
    let sq = delta.sqr();
    let sumsq = sq.reduce_sum_to(fuel_ir::Shape::from_dims(&[1, 1]));
    let norm = sumsq.sqrt();
    let eps_c = Tensor::from_existing(g.clone(), query.id())
        .const_f32_like(vec![eps], fuel_ir::Shape::from_dims(&[1, 1]));
    let pred = norm.lt(&eps_c);                           // U8 [1,1]
    // scan_until: consts must include EVERY const the body OR predicate reads: X and eps_c.
    query.scan_until(&[], &[patterns.clone(), eps_c], &new_carry, &new_carry, &pred, max_iters, ScanEmit::Final)
}
```
  - Body/predicate ops are the infallible `Tensor` builders; `scan_until` is the `Result` boundary. `consts = [patterns, eps_c]` — the invariant from Task 2's doc.
  - `query` is `init_carry`; `emit = Final` sidesteps the stacked‑`ys` capacity buffer.
- [ ] **Run GREEN:** `cargo test -p fuel-core hopfield_retrieves_stored_pattern_and_exits_early`.
- [ ] **Commit:** `feat(core): hopfield_retrieve — Modern Hopfield associative-memory Op::Scan consumer (C6/C7)`.

**Deliverable:** Hopfield retrieval converges to the stored pattern and early‑exits before the capacity `bound`.

---

## Task 8 — Hopfield gradient finite‑difference test (C7)

Prove BPTT through the unrolled retrieval matches finite differences — the consumer that exercises the Piece‑2 pre‑pass on an early‑exit scan (unrolls to `bound`, ignoring the predicate).

**Files:**
- Add a test to `fuel-core/src/hopfield.rs` `#[cfg(test)]`. No production code unless a gap surfaces.

**Interfaces:** consumes `hopfield_retrieve`, `fuel_graph::Tensor::backward`, `fuel_graph::scan::unroll_scan`, `crate::pipelined_bridge::realize_one_as`.

**Steps:**

- [ ] **Write the failing test.** Use a small `max_iters` (e.g. 3) so the static‑`bound` unroll is cheap. Loss = sum of retrieved `xi`. FD is over the **unrolled** retrieval (self‑consistent, predicate ignored), matching the static‑horizon backward.
```rust
#[test]
fn hopfield_gradient_matches_finite_difference() {
    use fuel_graph::Tensor;
    let dev = Device::cpu();
    // Forward loss L(X) = sum(unroll(retrieve(q, X, beta, eps, 3))), FD over X[0].
    let build = |x_vals: &[f32]| -> (std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>, fuel_graph::Tensor, fuel_graph::Tensor) {
        let x = Tensor::from_f32(x_vals.to_vec(), Shape::from_dims(&[2, 3]), &*dev.as_dyn());
        let q = Tensor::from_existing(x.graph().clone(), x.id()).const_f32_like(vec![0.6, 0.3, 0.1], Shape::from_dims(&[1, 3]));
        let r = crate::hopfield::hopfield_retrieve(&q, &x, 4.0, 1e-6, 3).expect("retrieve");
        (x.graph().clone(), x, r)
    };
    let x0 = vec![1.0f32, 0.0, 0.0,  0.0, 1.0, 0.0];
    // Forward via unroll+realize+sum.
    let fwd = |x_vals: &[f32]| -> f32 {
        let (g, _x, r) = build(x_vals);
        let scan_id = { let gr = g.read().unwrap(); gr.node(r.id()).inputs[0] };
        let bound = { let gr = g.read().unwrap(); match gr.node(scan_id).op { fuel_graph::Op::Scan { bound, .. } => bound, _ => unreachable!() } };
        let carry = { let mut gw = g.write().unwrap(); fuel_graph::scan::unroll_scan(&mut gw, scan_id, bound).expect("unroll").0 };
        crate::pipelined_bridge::realize_one_as::<f32>(&g, carry, &dev).expect("realize").iter().sum()
    };
    // Autograd at x0: loss = sum(retrieval). Build a scalar loss node, backward, grad w.r.t X.
    let (g, x, r) = build(&x0);
    let loss = r.reduce_sum_to(fuel_ir::Shape::from_dims(&[1, 1])); // sum -> scalar
    let grads = loss.backward();
    let g_x_id = grads.get(&x).expect("grad X").id();
    let g_x = crate::pipelined_bridge::realize_one_as::<f32>(&g, g_x_id, &dev).expect("realize gradX");
    // Central FD on X[0].
    let h = 1e-3f32;
    let mut xp = x0.clone(); xp[0] += h;
    let mut xm = x0.clone(); xm[0] -= h;
    let fd0 = (fwd(&xp) - fwd(&xm)) / (2.0*h);
    assert!((g_x[0] - fd0).abs() < 5e-2, "hopfield dL/dX[0]: autograd {} vs FD {fd0}", g_x[0]);
}
```
- [ ] **Run RED / validate.** `cargo test -p fuel-core hopfield_gradient_matches_finite_difference`. If Tasks 1+5 are in place it should pass; if an op in the Hopfield unroll lacks a backward arm it panics with the op name (add the arm). Observe the pass, or the specific gap.
- [ ] **Run GREEN.**
- [ ] **Commit:** `test(core): Hopfield BPTT finite-difference gate (C7)`.

**Deliverable:** `∂(loss)/∂X` through the unrolled Hopfield retrieval matches finite differences.

---

## Task 9 — constitution + roadmap docs (hard gate, same change)

Per CLAUDE.md "docs are part of every material change." Judge MAJOR/MINOR at write‑time against `docs/architecture/00-index.md`.

**Files:**
- `docs/architecture/03-ir.md` (MINOR): the early‑exit realize‑barrier mechanism is now built — a predicate‑over‑carry trailing input evaluated by a host step driver; `bound` = static capacity, runtime `count` carried as a host scalar (the capacity+count claim already exists; this records the built mechanism). Bump its version header.
- `docs/architecture/04-optimization.md` (MINOR): the BPTT‑via‑decompose backward path — `lower_scans_for_backward` unrolls `Op::Scan` and decomposes the SSM fused ops before the reverse walk; BPTT is truncated to the static `bound`.
- `docs/architecture/14-lifecycle.md` (MINOR): `selective_scan`/`ssd_chunk_scan` are now differentiable via the Phase‑2 pre‑pass.
- `docs/architecture/10-decisions-log.md`: a new entry closing the three Phase‑2 obligations named by the Phase‑1 entry (10-decisions-log.md:800); explicitly re‑state that **no `Op::Scan` native kernel was added** (the slot‑1/`last_state` OOB blocker stays open — 10-decisions-log.md:790).
- `ROADMAP.md`: advance the `Op::Scan` frontier line to "Phase 2 shipped: early‑exit + differentiability + Hopfield consumer".

**Steps:**
- [ ] Update each file; bump the version header on any `03/04/14` section touched; add the `10-decisions-log` entry (a MAJOR bump adds a decisions‑log entry — these are MINOR, but the Phase‑2 close is itself a log‑worthy event per the Phase‑1 entry's promise). Cross‑reference the spec.
- [ ] No cargo. Verify internal doc links resolve (the `../../../fuel-graph/...` relative paths from the spec are the pattern).
- [ ] **Commit:** `docs(arch): close Op::Scan Phase-2 obligations — early-exit + BPTT + Hopfield; no Scan kernel (slot-1 blocker stays open)`.

**Deliverable:** constitution + roadmap reflect the shipped Phase‑2 mechanism and the still‑open slot‑1 boundary.

---

## Self‑review

**Spec component → task coverage:**
- **C1 predicate carrier** (peel trailing input; `ScanPredicate` unit marker; hashing distinctness) → Task 1 (`unroll_scan` peel) + Task 2 (`scan_until` builder, validation, `base_map_hash` distinctness).
- **C2 realize‑barrier step driver** → Task 3 (`parse_scan_layout`/`build_scan_step`, shared‑subst predicate clone — resolves the spec's "predicate referencing `body_new_carry`" open question via one shared `subst`) + Task 4 (`drive_scan_until_final_f32`, stop‑at‑k + run‑to‑bound). Integration‑site open question resolved: standalone driver, not `realize_f32` auto‑routing (plan‑caching risk deferred, `emit=All` valid‑count buffer explicitly rejected).
- **C3 lower‑then‑differentiate pre‑walk pass** → Task 5 (`lower_scans_for_backward`: SSM decompose pass + scan‑unroll pass + consumer rewire mirroring `opt.rs:334‑352`; wired into `backward`).
- **C4 wire the backward arms** → Task 5 (`Op::Scan` arm → defensive guard; SSM fused nodes replaced by the pre‑pass before the `Op::Fused` walk arm).
- **C5 flip SSM `BackwardKind`** → Task 5 (`Decompose` + doc rewrite, explicitly intent‑not‑mechanism).
- **C6 `hopfield_retrieve`** → Task 7.
- **C7 tests** → Task 4 (C2 stop), Task 6 (C3/C4/C5 affine + SSM BPTT FD), Task 7 (C6/C7 convergence), Task 8 (C6/C7 gradient FD). The spec's "born‑red op‑coverage per body" risk is handled in Tasks 6/8 (a missing arm panics with the op name).
- **Constitution diff** → Task 9.

**Boundaries honored:** no `Op::Scan` native kernel (Task 4 drives via unroll; Task 5 unrolls for backward) → slot‑1/`last_state` OOB untouched. BPTT truncated to static `bound` (Task 1 explicitly ignores the predicate for the backward unroll; equilibrium/implicit‑diff out of scope). No `PatternNode`/Baracuda‑seam changes, no symbolic `bound`. Mostly `fuel-graph`/`fuel-core`, CPU.

**Type‑name consistency:** `Op::Scan { n_xs, bound, emit, early_exit }`, `ScanPredicate` (unit), `ScanEmit::{All, Final}`, `ScanRole::{Carry, Elem}`, `ScanLayout`/`ScanStep` (new, Task 3), `BackwardKind::{Fused, Decompose, NotDifferentiable}`, `FusedOps::{SELECTIVE_SCAN, SSD_CHUNK_SCAN}`, `GradMap`, `Tensor::{scan, scan_until, from_existing, const_f32_like}`, `drive_scan_until_final_f32`, `hopfield_retrieve`, `lower_scans_for_backward` — all match the grounded code and are used consistently across tasks.

**Sequencing (each deliverable is the next's oracle):** 1 (unroll peel) → 2 (builder, uses peel semantics) → 3 (step builder, uses layout) → 4 (driver, uses step builder) → 5 (backward pre‑pass, uses unroll) → 6 (numeric BPTT, validates 5) → 7 (Hopfield, uses 4 + scan_until) → 8 (Hopfield grad, uses 5 + 7 + unroll‑of‑early‑exit from 1) → 9 (docs).

**Judgment calls made (flag for the executor):**
1. **C2 integration site** — the spec left this open. Chosen: a standalone `fuel-core` driver realizing each step via `realize_one_as`, feeding the carry forward as a fresh `const_f32_like`, invoked explicitly by tests/consumers. This keeps plan‑once caching pristine for non‑scan graphs and is CPU‑testable, at the cost of NOT auto‑converging a plain `retrieval.realize_f32()` (documented follow‑up). If the reviewer wants `realize_f32` auto‑routing, that is a separate increment gated on the plan‑caching interaction.
2. **`emit = All` early‑exit** — rejected by the driver with a typed `Err` (Hopfield is `emit = Final`; the valid‑count capacity buffer is explicitly out of scope per the spec).
3. **Predicate‑const sharing** — `scan_until`/the driver share only the `consts` list; a const referenced by the predicate but omitted from `consts` would be re‑cloned dataless and fail at realize (a typed `Err`, never a panic). Documented as a builder contract rather than fully validated at build time (full transitive‑leaf validation was judged not worth the complexity; the realize error is clean).
4. **`lower_scans_for_backward` operates in place** (spec open question) — chosen in‑place (simpler); the forward fused path is only mutated when `backward()` is called, which a training step wants anyway (BPTT needs the unrolled activations). Inference‑only forward never calls `backward`, so the fused kernel stays.

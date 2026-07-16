# Op::Scan — Phase 2: early-exit mechanism + differentiability + Hopfield consumer — design

**Date:** 2026-07-15 · **Status:** design, pre-plan · **Part of:** the higher-order bounded-scan primitive workstream. **Follows** [Op::Scan Phase 1](2026-07-15-op-scan-phase1-core-primitive-ssm-design.md) (shipped + merged, commits `95dd31c6..d26f16b4` on `capturedrun-4b-resume`; decisions-log [10-decisions-log.md:780](../../architecture/10-decisions-log.md)).

> **Grounding:** file:line citations below were read against the shipped Phase-1 code (branch `op-scan-phase1`) on 2026-07-15. Verify against current code before trusting a citation as load-bearing.

## Goal

Turn the three Phase-1 placeholders into working mechanism, and prove them with one real non-SSM consumer:

1. **Build the `early_exit` mechanism** — a predicate-over-carry, carried on `Op::Scan` and evaluated **at the realize barrier at runtime**, that stops iteration early (a data-dependent iteration count under a static capacity `bound`). The `Op::Scan.early_exit: Option<ScanPredicate>` field exists ([lib.rs:1138](../../../fuel-graph/src/lib.rs), [lib.rs:1174](../../../fuel-graph/src/lib.rs)) but is never `Some` on a live path; a `Some` today is a typed `Err` in `unroll_scan` ([scan.rs:42](../../../fuel-graph/src/scan.rs)). Phase 2 makes `Some` a live, correct path.
2. **Build the decompose-backward plumbing** — a `lower-to-primitives-then-differentiate` mechanism (unroll `Op::Scan` → run the node-general autograd over the primitive unroll → BPTT into the scan's real inputs), wire the `Op::Scan` backward-walk arm to it, and use it to make `selective_scan` + `ssd_chunk_scan` differentiable via their existing `Op::Scan`-decompose path.
3. **Ship a Modern Hopfield associative-memory consumer** — `ξ ← softmax(β·ξ·Xᵀ)·X` iterated to a fixed point, `carry = ξ`, `early_exit = ‖Δξ‖ < ε` — a real non-SSM `Op::Scan` user that exercises **both** early-exit and BPTT, executing entirely through the on-demand unroll (each step = matmul + softmax, primitives that already have kernels). No `Op::Scan` native kernel is added.

Phase 2 adds **no `Op::Scan` native kernel**. Hopfield forward runs via the step-by-step unroll driver; BPTT runs via the unroll-then-autograd path. This keeps the slot-1/`last_state` OOB blocker ([10-decisions-log.md:790](../../architecture/10-decisions-log.md)) out of scope (see Boundaries).

## Background (grounded)

### The shipped `Op::Scan` surface

`Op::Scan { n_xs, bound, emit, early_exit }` ([lib.rs:1134](../../../fuel-graph/src/lib.rs)) is Fuel's first sub-graph-carrying primitive. The body is encoded as the node's **own trailing inputs**, exactly like `Op::Branch`'s arms:

```
inputs = [ init_carry, xs_0..xs_{n_xs-1}, consts..., body_new_carry, body_y ]
```

— the two body-exit NodeIds are always the **last two** inputs, so `base_map_hash`, `topo_order_multi`, and reachability see the body for free ([lib.rs:1118-1125](../../../fuel-graph/src/lib.rs)). Body holes are `Op::ScanPlaceholder { role, index }` leaves ([lib.rs:1143](../../../fuel-graph/src/lib.rs)): `ScanRole::Carry` (index 0, single-carry v1) for the per-step carry, `ScanRole::Elem` for the per-step slice of `xs[i]` ([lib.rs:1164](../../../fuel-graph/src/lib.rs)). `emit ∈ {All, Final}` ([lib.rs:1155](../../../fuel-graph/src/lib.rs)); the node is always a 2-slot bundle (slot 0 = stacked `ys`, slot 1 = final carry). The builder `Tensor::scan(xs, consts, body_new_carry, body_y, bound, emit)` ([lib.rs:5157](../../../fuel-graph/src/lib.rs)) is `Result`-returning, validates same-graph / `bound ≥ 1` / carry-shape-match, composes the 2-slot bundle, and hard-codes `early_exit: None` ([lib.rs:5211](../../../fuel-graph/src/lib.rs)).

`unroll_scan(graph, scan_id, steps)` ([scan.rs:16](../../../fuel-graph/src/scan.rs)) materializes a bounded scan into ordinary primitives on demand — the verify oracle and kernel-absent fallback, **never registered as a `decompose`**. It parses the trailing-two-inputs body layout ([scan.rs:63-67](../../../fuel-graph/src/scan.rs)), validates every reachable placeholder's index in-range **before** any mutation ([scan.rs:70-96](../../../fuel-graph/src/scan.rs)), then for each step slices `xs[i]` at `[t, t+1)` on axis 0, squeezes, clones the body with `ScanPlaceholder{Carry}→carry` / `ScanPlaceholder{Elem,i}→elem[i]` substituted and consts shared ([clone_body_node, scan.rs:185](../../../fuel-graph/src/scan.rs)), and concatenates the stacked `ys`. It returns a **typed `Err`** for `early_exit = Some` ([scan.rs:42-46](../../../fuel-graph/src/scan.rs)) — the Phase-2 seam this spec opens.

`ScanPredicate` ([lib.rs:1174](../../../fuel-graph/src/lib.rs)) is today a constructible unit marker: the field is present so the enum shape is final (one `03-ir` MAJOR bump), the mechanism is deferred.

### The BackwardKind-is-dead finding (verified)

`Tensor::backward() -> GradMap` ([lib.rs:7237](../../../fuel-graph/src/lib.rs)) computes the reverse topo order **once** from the current graph ([lib.rs:7242](../../../fuel-graph/src/lib.rs)), seeds the root with a ones-tensor, then walks each node: it first consults `crate::grad::dispatch_gradient` ([lib.rs:7275](../../../fuel-graph/src/lib.rs)), a per-op `GradientRule` dispatcher ([grad.rs:62](../../../fuel-graph/src/grad.rs), covering `Add`/`Mul`/`Relu`/comparisons/`Where` today), and falls through to a hand-written exhaustive inline `match op` for everything else. That inline match is where `MatMul` ([lib.rs:7338](../../../fuel-graph/src/lib.rs)), `Exp`, `Slice` ([lib.rs:8754](../../../fuel-graph/src/lib.rs)), `Concat` ([lib.rs:8726](../../../fuel-graph/src/lib.rs)), `Squeeze`/`Unsqueeze`/`BroadcastTo`/`ReduceSumTo`/`Permute`/`SumDim`, and the migrated fused ops (softmax et al. via the `Op::Fused` arm, [lib.rs:8598-8604](../../../fuel-graph/src/lib.rs)) are differentiated.

`Op::Scan` is a **clean `NotDifferentiable` panic** in that walk ([lib.rs:9578-9588](../../../fuel-graph/src/lib.rs)); `Op::ScanPlaceholder` is an inert drop ([lib.rs:9589](../../../fuel-graph/src/lib.rs)). **Critically:** `BackwardKind` (the registry field, e.g. [selective_scan.rs:123](../../../fuel-graph/src/registry/selective_scan.rs), [ssd_chunk_scan.rs:110](../../../fuel-graph/src/registry/ssd_chunk_scan.rs)) is **dead metadata** — the decisions-log records it as "verified never read anywhere … there is no generic decompose-backward hook" ([10-decisions-log.md:798](../../architecture/10-decisions-log.md)). So Phase 2 differentiability is **not** "set `BackwardKind::Decompose`" — that would be inert plumbing. It must **build** the decompose-backward mechanism and wire the walk to it.

### The data-dependent-shapes substrate

`03-ir` §"State and runtime extents" ([03-ir.md:108](../../architecture/03-ir.md)) and §"Data-dependent output shapes" ([03-ir.md:101](../../architecture/03-ir.md)) pin the discipline early-exit must obey: a **fixed-capacity buffer allocated once, a per-step write at a runtime host-scalar offset, the true count carried as a resolved host-scalar parameter — never a mutating node shape**, "one plan serves every step." The KV-cache is the canonical instance. The frontier vision states the target directly: "Early-exit (fixed-point convergence for Hopfield/EBM) is a predicate over the carry, evaluated at the realize barrier — not unbounded data-driven search" ([frontier-paradigms-vision.md:81](../../frontier-paradigms-vision.md)); "the body is a fixed sub-graph and the iteration count is bounded (a capacity, exactly like the KV-cache runtime-offset pattern)" ([frontier-paradigms-vision.md:73](../../frontier-paradigms-vision.md)). `bound` is the capacity; `early_exit` supplies the actual runtime stop count.

## Architecture: the three pieces

```
Piece 1 — early_exit          Piece 2 — differentiability        Piece 3 — Hopfield consumer
─────────────────────         ──────────────────────────         ───────────────────────────
predicate sub-DAG carried     lower-to-primitives-then-           hopfield_retrieve(ξ0,X,β,ε,N)
on Op::Scan (trailing input   differentiate pre-walk pass:        = Op::Scan{ body: softmax·
when early_exit=Some)         unroll Op::Scan → node-general       matmul, carry=ξ, emit=Final,
        │                     autograd → BPTT into inputs         early_exit=‖Δξ‖<ε, bound=N }
        ▼                             │                                    │
realize-barrier step driver:  wire Op::Scan + SSM-fused             forward → Piece-1 driver
step body → eval predicate    backward arms to the pass;            backward → Piece-2 pass
on realized carry → stop      flip SSM BackwardKind (intent)       (no Op::Scan kernel involved)
```

All three run over primitives that already have kernels (matmul, softmax, elementwise, slice, concat). Nothing here adds an `Op::Scan` kernel or touches the frozen Baracuda-shared `PatternNode` crate.

## Components

### Piece 1 — early-exit

#### C1 — the predicate carrier (mirror the body encoding)

**Boundary:** structural IR + builder-time validation only; no evaluation here.

The predicate is a **sub-DAG over the carry**, carried on `Op::Scan` by the same trailing-input mechanism the body uses. When `early_exit = Some`, the input layout gains exactly **one** trailing slot:

```
inputs = [ init_carry, xs.., consts.., body_new_carry, body_y, pred_exit ]
```

- `pred_exit` is the exit NodeId of a predicate sub-DAG whose leaves may reference `Op::ScanPlaceholder{Carry, 0}` (the **pre-step** carry) and the shared arena node `body_new_carry` (the **post-step** carry), so a convergence predicate `‖ξ_new − ξ_old‖ < ε` is expressible as ordinary primitives (`Sub → norm → Lt`) with no new op. Its output is a **scalar boolean** (shape `[]` or `[1]`, dtype `U8`).
- Because `pred_exit` is a trailing input, `base_map_hash` / `topo_order_multi` / reachability see the predicate for free — the **same** rationale that makes the body exits trailing inputs ([lib.rs:1118-1125](../../../fuel-graph/src/lib.rs)). `op_key` needs no new arm: the predicate is hashed structurally via `Node::inputs`, so two scans with identical bodies but different predicates already hash distinct.
- `ScanPredicate` stays a **unit marker** — its `Some`-ness signals "peel one extra trailing input" to the layout parser. No NodeId is stored inside the variant (keeps `Op: Clone + PartialEq + Debug` trivial and avoids the `Op::Branch`-style `reconverge_at`-in-variant hashing wart). *(Plan-level: whether `ScanPredicate` gains a field is open, but the invariant — predicate arena-resident, reachable, hashed via inputs — is fixed.)*

`unroll_scan`'s input parse ([scan.rs:63-67](../../../fuel-graph/src/scan.rs)) keys off `early_exit.is_some()` to peel `pred_exit` before computing `consts = inputs[1+n_xs .. len - (2 or 3)]`. The Phase-1 `early_exit = Some → Err` guard ([scan.rs:42](../../../fuel-graph/src/scan.rs)) is **removed** and replaced by (a) full-`bound` unroll ignoring the predicate — the BPTT/oracle path (Piece 2) — and (b) the step driver (C2) for forward realize.

A `Result`-returning builder `Tensor::scan_until(xs, consts, body_new_carry, body_y, pred_exit, bound, emit)` (sibling of `Tensor::scan`, [lib.rs:5157](../../../fuel-graph/src/lib.rs)) validates **at graph-build time**: same-graph for `pred_exit`; `pred_exit` is scalar-shaped and `U8`; every `ScanPlaceholder` reachable from `pred_exit` is `Carry{0}` (no `Elem` — the predicate is over carry, not per-step inputs); `bound ≥ 1`. Any violation → typed `Err`, never a panic.

#### C2 — the realize-barrier step driver

**Boundary:** forward-realize control flow only; no gradient logic; no new kernel.

`Op::Scan` has no native kernel; a bodied scan realizes via the kernel-absent fallback (the unroll). With `early_exit = Some`, the fallback becomes a **host-driven step-by-step realize loop** at the realize barrier:

```
carry ← realize(init_carry)
count ← 0
for t in 0 .. bound:                          # bound = static capacity cap
    elem_t ← realize(slice xs at [t,t+1), squeezed)     # per-step inputs (n_xs may be 0)
    carry_next ← realize(clone(body_new_carry) | Carry←carry, Elem←elem_t)
    y_t        ← realize(clone(body_y)         | Carry←carry, Elem←elem_t)   # emit=All only
    stop ← realize(clone(pred_exit) | Carry←carry, body_new_carry←carry_next)  # → host bool
    push y_t; carry ← carry_next; count ← t+1
    if stop { break }
emit == Final  →  return carry                         # the fixed point (Hopfield case)
emit == All    →  return (ys stacked into the [bound,…] capacity buffer, count)  # valid-count
```

This is exactly the `03-ir` runtime-count discipline ([03-ir.md:101-112](../../architecture/03-ir.md)): `bound` is the static capacity (node shape never mutates), `count` is the runtime valid-count carried as a host scalar, and each step reuses the same cloned sub-plan. `emit = Final` (Hopfield's case) returns just the converged carry, so the capacity-buffer-with-count question does not even arise for the primary consumer. The clone/substitute machinery is the existing `clone_body_node` ([scan.rs:185](../../../fuel-graph/src/scan.rs)), extended so `body_new_carry` resolves to the current step's post-step carry when referenced from `pred_exit`.

Integration point: a realize-barrier front-end (in `fuel-graph`/`fuel-core` realize; **not** a `PipelinedExecutor` kernel entry) that recognizes a to-be-realized `Op::Scan{early_exit: Some}` with no registered kernel and hands it to this driver. Each step is its own realize of a primitive sub-graph, so the driver composes with — rather than replaces — the pipelined executor. *(Exact integration site is an open question, see Risks.)*

### Piece 2 — differentiability

#### C3 — the lower-to-primitives-then-differentiate pre-walk pass (the generic decompose-backward hook)

**Boundary:** builds the mechanism the codebase lacks; does not add per-op backward arms.

`Tensor::backward` is infallible (`-> GradMap`) and computes its topo order once ([lib.rs:7242](../../../fuel-graph/src/lib.rs)). Phase 2 adds a **pre-walk lowering pass** that runs **before** that topo order is taken:

1. Scan the reachable forward set (from `self.id`) for `Op::Scan` nodes and for the two SSM `Op::Fused` ids (`SELECTIVE_SCAN`, `SSD_CHUNK_SCAN`).
2. For each SSM fused node, apply its **existing** `decompose` ([selective_scan.rs:230](../../../fuel-graph/src/registry/selective_scan.rs), [ssd_chunk_scan.rs:209](../../../fuel-graph/src/registry/ssd_chunk_scan.rs)) to get its `Op::Scan` recipe.
3. For each `Op::Scan` node, materialize the **full-`bound` unroll** via `unroll_scan` (`steps = bound`; the predicate is **ignored** for the build-time backward unroll — see the note below), and rewire the scan's output consumers (its `Op::View{0}`/`View{1}` projections) to the unroll's `(stacked_ys, final_carry)` outputs.
4. Take the topo order over the now fully-primitive graph and run the existing reverse walk unchanged.

The unrolled graph contains only ops the existing walk already differentiates (`Slice`/`Squeeze`/`Concat`/`Unsqueeze` from `unroll_scan`; `Mul`/`Add`/`Exp`/`BroadcastTo`/`Reshape`/`ReduceSumTo`/`Permute` from the SSM body; `MatMul` + softmax-via-`Op::Fused` from the Hopfield body). Gradients accumulate back through the unroll into the scan's real inputs (`init_carry`, `xs`, `consts`) — this **is** BPTT. This single pass is the generic "differentiate the decomposition" hook Phase 1 found missing.

**Static-horizon note.** BPTT differentiates the static **`bound`**-unroll. The `early_exit` runtime stop is a **forward-inference** optimization; it is *not* applied to the build-time backward unroll (the convergence count is only known at forward realize, and the gradient graph must be built statically). This is truncated BPTT to the capacity horizon — a clean, defensible separation. *(Differentiate-to-convergence-count vs implicit-function-theorem equilibrium gradients is out of scope; see Open questions.)*

#### C4 — wire the backward-walk arms to the pass

**Boundary:** the two forcing arms; keep the walk infallible.

With C3 running first, the reverse walk never sees an `Op::Scan` (or an SSM `Op::Fused` on a differentiated path) on well-formed input. So:

- The `Op::Scan` arm ([lib.rs:9578](../../../fuel-graph/src/lib.rs)) changes from an unconditional `NotDifferentiable` panic to a **defensive internal-error guard** ("Op::Scan reached the backward walk un-lowered — C3 pre-pass did not run"). It is unreachable on any graph that went through `backward()`.
- The SSM `Op::Fused` backward dispatch is routed to the C3 decompose-then-unroll path rather than the current `NotDifferentiable`. Because the pre-pass replaces these nodes before the walk, the fused arm likewise becomes a guard.
- `Op::ScanPlaceholder` stays an inert drop ([lib.rs:9589](../../../fuel-graph/src/lib.rs)) — a placeholder is never a live differentiation target (the pre-pass substitutes it away).

#### C5 — flip the SSM `BackwardKind` (intent, not mechanism)

**Boundary:** metadata + doc honesty.

Flip `selective_scan`/`ssd_chunk_scan` `BackwardKind::NotDifferentiable` ([selective_scan.rs:123](../../../fuel-graph/src/registry/selective_scan.rs), [ssd_chunk_scan.rs:110](../../../fuel-graph/src/registry/ssd_chunk_scan.rs)) to a differentiable kind **and** update the module docs (both currently say "Why `BackwardKind::NotDifferentiable` for v1"). Because `BackwardKind` is verified-dead metadata ([10-decisions-log.md:798](../../architecture/10-decisions-log.md)), this flip is **intent documentation** — the actual differentiability comes from C3/C4. The spec states this explicitly so the flip is not mistaken for the mechanism (the exact Phase-1 trap).

### Piece 3 — the Hopfield consumer

#### C6 — `hopfield_retrieve` lazy module

**Boundary:** minimal retrieval, not a Hopfield network/layer; no fused `Hopfield{beta}` op.

Modern (dense) Hopfield retrieval ([frontier-paradigms-vision.md:130-132](../../frontier-paradigms-vision.md)): "Modern Hopfield update **is** attention." A lazy builder

```
hopfield_retrieve(query ξ0: [.., d], patterns X: [n, d], beta: f32, eps: f32, max_iters: usize) -> Tensor
```

constructs, over the shared graph:
- **carry** = `ξ` (shape `[.., d]`), `init_carry = ξ0`, `n_xs = 0` (no per-step inputs), `X` a **const** input.
- **body**: `logits = MulScalar(β)(MatMul(ξ, Xᵀ))` → `[.., n]`; `s = softmax_last_dim(logits)`; `body_new_carry = MatMul(s, X)` → `[.., d]`. All primitives with existing CPU/CUDA kernels; softmax is a differentiable `Op::Fused` ([lib.rs:8598-8604](../../../fuel-graph/src/lib.rs)).
- **pred_exit**: `Lt( l2_norm(Sub(body_new_carry, ξ_Carry_placeholder)), ε )` → `U8` scalar.
- `Tensor::scan_until(xs=[], consts=[X], body_new_carry, body_y = body_new_carry, pred_exit, bound = max_iters, emit = Final)`.

Forward realizes through the Piece-1 driver (converges early); backward differentiates through the Piece-2 pass (`∂loss/∂X`, `∂loss/∂ξ0`). No `Op::Scan` kernel, no fused Hopfield kernel. `emit = Final` sidesteps the stacked-`ys` capacity buffer entirely.

#### C7 — tests (see Testing)

## Data flow

**Build:** `hopfield_retrieve(...)` (C6) → one `Op::Scan{early_exit: Some, emit: Final}` node with the predicate as a trailing input (C1). **Forward realize:** the realize barrier routes the kernel-less early-exit scan to the step driver (C2), which realizes body + predicate per step and stops on convergence, returning the fixed-point carry with a runtime step count. **Backward:** `backward()` runs the C3 pre-pass — unroll the scan to `bound` primitives (predicate ignored), rewire the `View` consumers — then the existing reverse walk differentiates the pure-primitive graph (C4), accumulating BPTT gradients into `X`/`ξ0`. **SSM training (validation):** the same C3 path decomposes `Op::Fused(SELECTIVE_SCAN)` → `Op::Scan` → unroll → autograd, so the two SSM ops become differentiable (C5) without a bespoke `*_BACKWARD` fused op.

## Error handling / never-panic

- `Tensor::scan_until` (C1) validates predicate shape/dtype/placeholders/`bound` at **graph-build time** ([03-ir "validate at build time"]); malformed → typed `Err`.
- A malformed predicate reaching `unroll_scan` (wrong dtype, out-of-range `Elem` placeholder) → typed `Err` (extends the existing pre-mutation validation, [scan.rs:70-96](../../../fuel-graph/src/scan.rs)), never a panic.
- The step driver (C2) surfaces a realize failure as the standard realize `Result` error, not a panic; a predicate that never fires runs to `bound` (capacity) and stops — bounded, never an infinite loop.
- The C4 `Op::Scan`/SSM-fused backward arms become internal-error guards (should-never-fire because C3 lowered them); they keep the walk's infallible `-> GradMap` contract by carrying a descriptive message rather than silently dropping a gradient.
- No new `.unwrap()`/`.expect()` on production paths.

## Testing (TDD, born-red)

Pure `fuel-graph` + `fuel-core` logic; the Hopfield forward/gradient tests run on CPU (matmul + softmax kernels exist). Build `-p fuel-graph` / `-p fuel-core` only — never workspace-wide.

- **C1 carrier:** an `Op::Scan{early_exit: Some}` node with a predicate trailing input — `unroll_scan` parses the body correctly (peels `pred_exit`), and `op_key`/`base_map_hash` differ between two scans with identical body but different predicate. Builder rejects a non-scalar / non-`U8` predicate with `Err`.
- **C2 early-exit stop:** a scan whose predicate fires at a known step `k < bound` realizes exactly `k` steps (assert runtime count == k) and returns the step-`k` carry; a non-converging predicate runs to `bound` and stops (no infinite loop).
- **C3/C4 scan BPTT (finite-difference):** a small hand-built affine `Op::Scan` (e.g. `carry ← a·carry + b`, `bound = 3`) — autograd gradients w.r.t. `init_carry`/`consts` match a finite-difference of the **same unrolled graph** (self-consistent, so the check is decoupled from kernel F64-accumulate parity).
- **C4/C5 SSM differentiability:** `selective_scan` on a tiny fixture — `backward()` no longer panics; `∂y/∂u` (and `∂y/∂a`) match finite-difference over the decompose→unroll graph. Same for `ssd_chunk_scan`.
- **C6/C7 Hopfield convergence:** seed `ξ0` near one stored pattern in `X`; retrieval converges to that pattern within `ε`, and the early-exit stopped **before** `bound` (runtime count < capacity).
- **C6/C7 Hopfield gradient:** `∂(scalar loss on retrieved ξ)/∂X` via autograd matches finite-difference over the unrolled retrieval (a small fixed `max_iters`, e.g. 3–4).

## Boundaries

- **No `Op::Scan` native kernel.** Phase 2 executes Hopfield + BPTT via the unroll (forward: step driver; backward: unroll pre-pass). Adding an `Op::Scan` kernel is explicitly **not** in scope — and **must not** be added without first wiring or dropping the slot-1/`last_state` view. The decisions-log records the inversion: with no kernel, an un-composed `view(1)` is a **typed dispatch error**; with a kernel it becomes a **live silent out-of-bounds read** ([10-decisions-log.md:790](../../architecture/10-decisions-log.md)). Because Phase 2 adds no kernel, it does not trigger the blocker — but it also does not resolve it. Slot-1 wiring (a bundle-composer `Op::ScatterIntoSlot` or an explicit slot-1 drop) remains a prerequisite for any future kernel, not this phase.
- **SSM decode-with-init-state is separate.** `SelectiveScanWithInitState` autoregressive resumption (`lazy_mamba.rs`) is not touched; Phase 2 makes the existing forward SSM ops *differentiable*, it does not add a decode-loop variant.
- **BPTT is truncated to the static `bound`.** No implicit-function-theorem / deep-equilibrium gradients; no differentiate-to-runtime-convergence-count.
- **Minimal Hopfield only.** No fused `Hopfield{beta}` op ([frontier-paradigms-vision.md:130](../../frontier-paradigms-vision.md), bucket A), no Krotov polynomial-power energy, no `EnergyMinimizer`, no `nn.energy` leaf — those are separate follow-ons.
- **No `PatternNode`/Baracuda-seam changes**, no symbolic `bound`, no associative/chunked-scan kernel — all remain deferred exactly as in Phase 1.

## Open questions / risks

- **Early-exit executor integration is the newest substrate.** The step driver (C2) is a realize-time control-flow construct with no precedent in the pipelined/plan-once executor. **Open:** its exact integration site (a `fuel-core` realize wrapper vs a `fuel-graph` realize front-end vs a `PipelinedExecutor` pre-dispatch check) — `PipelinedExecutor::realize` was not traced line-by-line for this spec. **Risk:** interaction with plan-once / CapturedRun (each step is a separate realize; does the driver defeat plan caching?) and with data-dependent shapes (the runtime `count` for `emit = All` must ride the existing capacity-buffer + valid-count bundle, [03-ir.md:103](../../architecture/03-ir.md), not a new mechanism). Hopfield uses `emit = Final`, so this risk is deferred for the first consumer but must be resolved before an `emit = All` early-exit consumer.
- **`ScanPredicate` struct shape.** Unit marker (peel-one-trailing-input) is chosen; whether it ever needs a field (e.g. to distinguish predicate flavors) is a plan-level decision. The fixed invariant: predicate arena-resident, reachable, hashed via `inputs`.
- **Predicate referencing `body_new_carry`.** The convergence delta `‖ξ_new − ξ_old‖` needs both pre- and post-step carry. The design has `pred_exit` reference the shared `body_new_carry` arena node (post-step) plus the `Carry` placeholder (pre-step). **Open:** confirm `clone_body_node`'s substitution ([scan.rs:185](../../../fuel-graph/src/scan.rs)) cleanly resolves a `pred_exit` that transitively reaches `body_new_carry` without double-cloning — likely a memoization-key detail, verified at plan time.
- **Every op in the unroll must have a backward arm.** C3 relies on the walk covering every primitive the SSM/Hopfield unroll emits. Softmax-via-`Op::Fused` and `MatMul` are covered, but a gap surfaces as a panic mid-walk. **Risk:** enumerate the unroll's op set against the walk's arms before building (a born-red coverage test per body).
- **`backward()` unrolls the forward graph in place.** C3 rewires the scan's `View` consumers to the unroll, so a subsequent forward realize of the loss also runs the (slower) unroll rather than the fused kernel. Acceptable for a training step (BPTT needs the unrolled activations anyway); inference-only forward keeps the fused kernel. **Open:** whether C3 should operate on a cloned sub-graph to leave the forward fused path pristine, or in place (simpler) — a plan-level choice.
- **Constitution diff (hard gate, same change).** Per CLAUDE.md "docs are part of every material change": `03-ir` (early-exit realize-barrier mechanism now built — MINOR, the capacity+count claim already exists), `04-optimization` (BPTT-via-decompose backward path — MINOR), `14-lifecycle` (SSM ops now differentiable — MINOR), plus a new `10-decisions-log` entry closing the three Phase-2 obligations the Phase-1 entry named ([10-decisions-log.md:800](../../architecture/10-decisions-log.md)) and explicitly re-stating that no `Op::Scan` kernel was added (so the slot-1 blocker stays open). Judged at write-time against [00-index.md](../../architecture/00-index.md) MAJOR/MINOR rules.

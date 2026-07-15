# Op::Scan — Phase 1: core primitive + SSM re-decomposition (closes G3) — design

**Date:** 2026-07-15 · **Status:** design, pre-plan · **Part of:** the higher-order bounded-scan primitive workstream (pulled forward by the recipe-grammar co-design Q5; see [[recipe-grammar-codesign]], [[frontier-paradigms-vision]]). **Phase 2** (early-exit mechanism + differentiability + Hopfield consumer) is a separate follow-on spec.

> **Grounding:** every file:line below comes from the architecture-map workflow (task `wlgq8s329`, 7 readers + synthesis, 2026-07-15). Verify against current code before trusting a citation as load-bearing.

## Goal

Add a new primitive `Op::Scan` to the Fuel IR — Fuel's first `Op` parameterized by a sub-graph `body` — and re-decompose `selective_scan` + `ssd_chunk_scan` onto it, **closing G3**: the last two fused ops whose `decompose` self-returns ([selective_scan.rs:210](fuel-graph/src/registry/selective_scan.rs#L210), [ssd_chunk_scan.rs:169](fuel-graph/src/registry/ssd_chunk_scan.rs#L169)). After Phase 1, `decompose` is **total over the entire fused-op set** — the recipe-principle build-time-closed-basis invariant holds without a scan caveat, and the base map that Fuel's recipe-identity verifier ([[increment-1-recipe-identity-complete]]) resolves to bottoms out at genuine primitives.

Phase 1 defines the **full** `Op::Scan` shape (all fields, so there is exactly one `03-ir` MAJOR bump and no later re-bump) but implements only what the SSM consumer needs. The `early_exit` mechanism, differentiability of the SSM ops, and the Hopfield consumer are **Phase 2**.

## What Phase 1 is NOT (framing correction from the map)

`Op::Scan` does **not** unblock Mamba *execution*. Both `selective_scan` and `ssd_chunk_scan` already have live CPU kernels ([dispatch.rs:3267](fuel-dispatch/src/dispatch.rs#L3267)) and CUDA kernels (`fuel-cuda-backend/src/baracuda/mamba.rs`), backed by baracuda-kernels-sys alpha.77's SSM + scan FFI. Phase 1's payoff is **optimizer-basis closure + recipe-identity verification + a total `decompose`**, not new runtime capability. A key non-regression requirement follows (below): the existing fused kernels must stay the executed path.

## Background: why a new primitive, not a decompose fix

The two SSM ops are documented never-crash surfaced gaps, not bugs ([10-decisions-log.md:406](docs/architecture/10-decisions-log.md#L406), [04-optimization.md:43](docs/architecture/04-optimization.md#L43)). Two in-place decompose recipes are on record and both rejected:

1. **`O(seqlen)` per-step unroll directly as the fused op's decompose** — "total, but node count is unbounded / no finite re-fusing pattern — not a recipe, an explosion" ([selective_scan.rs:53](fuel-graph/src/registry/selective_scan.rs#L53)).
2. **Diagonal-SSM CumSum closed form** `h[t] = exp(a·D[t]) ⊙ cumsum_t(exp(−a·D[s]) ⊙ x[s])` — **overflows** because Mamba's `a = −exp(a_log) < 0` makes `exp(|a|·D[s])` blow up ([10-decisions-log.md:406](docs/architecture/10-decisions-log.md#L406)). `CumSum` also physically can't carry the per-step gate — it is unweighted `+`, no combine parameter ([lib.rs:456](fuel-graph/src/lib.rs#L456)).

`Op::Scan` resolves the dilemma by being **one compact terminal node** that carries the bounded recurrence, unrolling only on demand for verification. With `body`+`bound` fixed at graph-build, the unroll is a *concrete finite bound* (not open-ended), so the "explosion" objection dissolves one level down — and the unroll only has to be *right*, not *fast*, because the optimizer prefers a native/re-fused arm at optimize time. The affine SSM recurrence `h ← A_t·h + B_t` ([byte_kernels.rs:6104](fuel-cpu-backend/src/byte_kernels.rs#L6104)) needs a general `body` (its associative combine is the affine-pair semiring `(a₁·a₂, a₂·b₁+b₂)` — two muls + one FMA, not a single monoid), which a body-carrying scan expresses and a monoid `prefix_scan` cannot.

## Architecture: the op shape

```
Op::Scan {
    body:       ScanBody,            // arena NodeId-region: the per-step recurrence
    carry:      /* real input + output */,
    bound:      usize,               // concrete capacity (max steps)
    emit:       ScanEmit,            // All | Final
    early_exit: Option<ScanPredicate>,  // FIELD present in Phase 1; MECHANISM is Phase 2
}
```

- **`body` — arena NodeId-region** (the `Op::Branch` precedent: [lib.rs:1090](fuel-graph/src/lib.rs#L1090) encodes its arms as ordinary inputs in the *same* arena). `body` designates `body_entry`/`body_exit` NodeIds plus **carry-in / carry-out placeholder leaves** in the same graph arena, so the body participates in the existing reachability/validate/remap machinery rather than a parallel scheme, and **without** touching the frozen Baracuda-shared `PatternNode` crate ([fuel-kernel-seam-types/src/lib.rs:93](fuel-kernel-seam-types/src/lib.rs#L93)). Exact struct encoding is a plan-level detail; the invariant is that the body is arena-resident and hashable (see `op_key` below).
- **`carry` — a real input+output pair from day one.** Carry-in is an implicit zero at prefill and the prior step's `carry` at decode — pre-empting the `SelectiveScanWithInitState` decode-loop blocker ([lazy_mamba.rs:21](fuel-transformers/src/lazy_mamba.rs#L21)) with no second variant. The carry may be a bundle of ≥2 tensors (SSM threads the hidden state `h`; some recurrences thread additional running quantities).
- **`bound` — concrete `usize` capacity + a runtime host-scalar count** (the fixed-capacity-buffer / runtime-offset discipline of [03-ir.md:96](docs/architecture/03-ir.md#L96) "State and runtime extents": allocate once, per-step write at a runtime host-scalar offset, never a mutating node shape, "one plan serves every step"). A **symbolic bound** (SymId/DynScalar interop for variable-length decode) is a **documented deferred gap**, exactly like flash_attn's concrete-vs-`Sym` k_len split ([10-decisions-log.md:404](docs/architecture/10-decisions-log.md#L404)).
- **`emit ∈ {All, Final}`** — reuses the existing Option-C multi-output bundle (slot 0 = stacked per-step `y`, slot 1 = final `carry`) that both SSM ops already expose via `output_views`/`OutputViewSpec` ([selective_scan.rs:100](fuel-graph/src/registry/selective_scan.rs#L100)). Do **not** invent a new multi-output mechanism. `Op::Reduce` is `Op::Scan{emit=Final}` conceptually — **no separate enum variant**; the existing fixed-combine `SumDim`/`ReduceSumTo` ([lib.rs:601](fuel-graph/src/lib.rs#L601)) stay untouched for the common case.
- **`early_exit: Option<ScanPredicate>`** — the field is defined in Phase 1 (so the enum shape is final) but its realize-barrier evaluation is **not implemented** in Phase 1. SSM never sets it. If it is `Some` on any realize/lowering path in Phase 1, the code returns a clear `Err` (a surfaced Phase-2 gap), **never a panic**.

## Components

### C1 — the `Op::Scan` enum variant + the three forcing matches
Add the variant to `pub enum Op` ([lib.rs:216](fuel-graph/src/lib.rs#L216)). Three exhaustive matches will fail to compile until handled (the good forcing functions): `op_short_name` ([lib.rs:1222](fuel-graph/src/lib.rs#L1222)); the legacy inline autograd `match op` in the backward walk ([lib.rs:7145](fuel-graph/src/lib.rs#L7145)); the guarded-total `derive_view_output_layout` ([lib.rs:1194](fuel-graph/src/lib.rs#L1194)).

### C2 — `op_key` with a structural body-hash (THE correctness-critical unit)
`base_map_hash` recurses **only through `Node::inputs`** ([opt.rs:474](fuel-graph/src/opt.rs#L474)); `op_key` ([opt.rs:953](fuel-graph/src/opt.rs#L953)) ends `_ => None`. If `Op::Scan`'s `body` is reachable off-inputs, two scans with different bodies but identical carry/bound shapes **hash equal** → silent CSE corruption **and** silent Spec-B/FKC recipe-identity false-positives. Phase 1 adds a new `op_key` arm that folds a **structural hash of the body region** (op discriminants + attrs + child structure of `body`, commutative-sorted per the existing rule) into the `Op::Scan` key. This is the single most correctness-critical step; it is gated by a dedicated test (two scans, same shapes, different bodies → different `op_key`).

### C3 — `Tensor::scan(...)` builder + shape/dtype rule
A `Result`-returning builder (the `cumsum`/`triu` precedent, [lib.rs:5028](fuel-graph/src/lib.rs#L5028)) that validates the body/carry/bound at graph-build time, wires carry-in as a real input, and computes the output `Shape`/`DType` inline: `emit=All` → the per-step `y` shape stacked over `bound`; `emit=Final` → the `carry` shape. The rule reads the body's `body_exit` shape/dtype; it is `Result`, never a panic on a malformed body.

### C4 — decompose = self (terminal) + the on-demand bounded-unroll utility
`Op::Scan`'s own `decompose` is `self` (a primitive terminal, like every other primitive), so `lower_to_base_map` bottoms out **at** `Op::Scan` rather than exploding into `O(bound)` nodes at every lowering call (which would blow up build time and mis-price the cost model). A **separate** `unroll_scan(scan, steps) -> subgraph` utility materializes the bounded unroll on demand, used as (a) the FKC/Spec-B **numeric oracle** and (b) the **fallback lowering** for a backend with no scan kernel. It must reproduce the SSM CPU kernel's **F64-accumulate-then-narrow** ([byte_kernels.rs](fuel-cpu-backend/src/byte_kernels.rs) — the SelectiveScan kernel accumulates `h` in F64 regardless of storage dtype, then narrows) or the design documents an epsilon drift for the sabotage-calibrated check.

### C5 — the hand-threaded lowering rule
Lowering rules are auto-generated **only** for `FusedOpRegistry`/`runtime_fused` entries ([opt.rs:164](fuel-graph/src/opt.rs#L164)). A primitive `Op::Scan` gets none for free. Phase 1 manually threads a `Scan` lowering/decompose `Rule` into `default_rules`/`capability_gated_rules`/`lowering_only` — otherwise `lower_to_base_map` silently leaves `Scan` un-decomposed everywhere (including the Spec-B recipe-identity path and CapturedRun planning), "works, nothing crashes," and the base-map-closure goal is quietly defeated.

### C6 — the deliberate opt-in sites
Handle each `_ => None`/`false`/`Transient` no-op site deliberately, not by default: `op_to_op_kind` ([pipelined.rs:3120](fuel-dispatch/src/pipelined.rs#L3120)) — **no native Scan kernel in Phase 1**, so no dispatch entry (Scan reaches a backend only via decompose/unroll); `op_to_tag` ([jit.rs:22](fuel-graph/src/jit.rs#L22)) — document as "not a region node; its decomposition is," following the `Fused` precedent; `grad.rs` `GradientRule` ([grad.rs:69](fuel-graph/src/grad.rs#L69)) — wire `BackwardKind::Decompose` as the natural default (the mechanism is nearly free — node-general dispatch differentiates the unroll — and wiring it now avoids re-touching the match in Phase 2), **but BPTT is not validated and the SSM fused ops keep `NotDifferentiable` until Phase 2** (which adds the consumer-backed BPTT test); `infer_storage_class`/`destructive_input`/`is_view_op`/`try_simplify` — safe conservative defaults, likely no edit, each confirmed not-silently-wrong. The `Op::Branch` reachability fixpoint (`effective_roots`, [lib.rs:2187](fuel-graph/src/lib.rs#L2187)) is extended to `Scan` bodies if a body can nest inside another Branch/Scan, so reachability doesn't miss live scan bodies.

### C7 — the re-fuse `pattern` (non-regression)
`variant_bake` defaults to the decomposed arm on any tie/unknown-cost/missing-kernel ([variant_bake.rs](fuel-graph/src/variant_bake.rs)). To keep the existing fused SSM kernels as the executed path, the SSM ops' **native fused-kernel arm must remain present and costed** so it wins over the `Op::Scan`-unroll arm. Phase 1 ensures the fused-kernel arm is reached (a re-fuse `pattern` from `Op::Scan{affine body}` back to the fused op, or the retained native arm — a plan-level choice), and **a bench confirms Mamba does not regress** to the O(seqlen) unroll.

### C8 — the SSM re-decomposition (the G3-closing edit)
Rewrite [selective_scan.rs:210](fuel-graph/src/registry/selective_scan.rs#L210) and [ssd_chunk_scan.rs:169](fuel-graph/src/registry/ssd_chunk_scan.rs#L169) `decompose` to emit `Op::Scan{ body = the affine step `h ← exp(d·a)·h + d·b·u`, carry = h, bound = seqlen, emit = All }` — leaving the ops' shape/dtype/backward contracts unchanged (SSM stays `NotDifferentiable` in Phase 1; the widen makes them differentiable in Phase 2). **Note (map risk):** the CPU `ssd_chunk_scan` does *not* run a real chunked/parallel SSD algorithm — it runs the identical sequential per-token recurrence and asserts `chunk_size` is a no-op ([byte_kernels.rs:6328](fuel-cpu-backend/src/byte_kernels.rs#L6328)); the true chunked matrix-form lives only in retired eager code. So both ops re-decompose to the **same** sequential affine `Op::Scan` — do not assume `ssd_chunk_scan` already demonstrates a working chunked decompose.

### C9 — gap-posture tests → positive
Flip the `selective_scan` gap-posture regression ([lazy.rs:2059](fuel-core/src/lazy.rs#L2059)) from "`Op::Fused` reachable / gap present" to "no `Op::Fused` reachable; the decomposed subgraph matches numerically" (same `h=3, y=12` expectations), matching the NF4/FlashAttn positive-test precedent. **Add the missing `ssd_chunk_scan` regression test** (only `selective_scan` has one today; `ssd_chunk_scan.rs:53` also carries a stale "decompose panics" doc comment to fix).

### C10 — the constitution diff (a hard gate, same change)
Per CLAUDE.md "docs are part of every material change," judged against [00-index.md:114](docs/architecture/00-index.md#L114) ("MAJOR when a section's *core claim* changes"), using the 2026-06-20 entry as the template ([10-decisions-log.md:326](docs/architecture/10-decisions-log.md#L326)):
- **`03-ir` MAJOR** — the first sub-graph-carrying primitive genuinely shifts the section's character claim ("no generic opaque/Custom node"; a body-region is a new structural kind).
- **`04-optimization` MINOR** — the DecompositionMap / cost-from-decompose gains a `Scan` entry (cost from the re-fused arm, not the unroll — the mis-pricing risk).
- **`12-multi-output` MINOR + `14-lifecycle` MINOR** — fix the stale "decompose panics" / "three panicking decomposes" prose ([12-multi-output.md:105](docs/architecture/12-multi-output.md#L105), [14-lifecycle.md:260](docs/architecture/14-lifecycle.md#L260)).
- **`08-pattern-harvest` MINOR** — the basis gap closes.
- A **new decisions-log entry** explicitly closing the 2026-07-03 G3 gap and referencing the 2026-06-20 "higher-order Scan for SSMs" named exception ([10-decisions-log.md:336](docs/architecture/10-decisions-log.md#L336)), noting that flash_attn's symbolic-`k_len` gap remains **separately open** so readers don't assume all basis gaps closed.

## Data flow

Build: `Tensor::scan(body, carry_in, bound, emit)` → a single `Op::Scan` node (C3). Optimize: `lower_to_base_map` reaches `Op::Scan` and stops (terminal, C4); `op_key` hashes it including the body (C2); the SSM fused op offers `{native fused kernel arm, Op::Scan-decompose arm}` and `variant_bake` picks the costed fused arm (C7). Verify (Spec-B / FKC): a candidate claiming an SSM/scan op is checked against `unroll_scan` realized to primitives (C4) — the numeric oracle. Execute: the SSM fused kernel (unchanged); a bodied scan with no fused kernel (future) falls back to the realized unroll.

## Error handling / never-panic

Every new surface is `Result`: the builder (C3), the shape/dtype rule (C3), `unroll_scan` (C4). `early_exit = Some` in Phase 1 → a clear `Err` on the realize/lowering path (surfaced Phase-2 gap), never a panic. No new `.unwrap()`/`.expect()` on production paths. Validation runs at graph-build time (the builder validates body/carry/bound), per "validate at graph-build time."

## Testing (TDD, born-red)

Pure `fuel-graph` + `fuel-dispatch` logic — no GPU required for the core, though the SSM parity + non-regression bench touch live kernels. The gates:
- **C2 op_key:** two `Op::Scan` nodes, identical carry/bound shapes, different bodies → different `op_key` / `base_map_hash` (the CSE/recipe-identity trap).
- **C3 builder:** shape/dtype for `emit=All` vs `emit=Final`; malformed body → `Err` not panic.
- **C4 unroll:** `unroll_scan` of the affine body realizes to the SSM reference numerically (F64-accumulate matched), at the sabotage-calibrated tolerance ([[sabotage-test-calibration]]).
- **C5 lowering:** `lower_to_base_map` on a graph containing `Op::Scan` produces the terminal (not un-decomposed-silently); the Spec-B recipe-identity path resolves it.
- **C8/C9 SSM:** the flipped `lazy.rs` positive tests (both ops, `h=3, y=12`), no `Op::Fused` reachable after decompose.
- **C7 non-regression:** a Mamba decode bench confirms the fused kernel remains the executed arm (no O(seqlen) regression).
- **early_exit guard:** `Op::Scan{early_exit=Some}` → `Err` (never panic) on realize.

## Boundaries — explicitly Phase 2 (separate spec)

- The `early_exit` realize-barrier evaluation + data-dependent iteration.
- Making `selective_scan`/`ssd_chunk_scan` (and `Op::Scan`) **differentiable** end-to-end + BPTT validation.
- The **Modern Hopfield** associative-memory consumer (`ξ ← softmax(β·ξ·Xᵀ)·X`, `early_exit=‖Δξ‖<ε`) + its tests.

## Boundaries — deferred beyond Phase 2 (documented gaps)

- A **native associative/chunked scan kernel** (`op_to_op_kind` + CPU/CUDA + a costed variant arm) — speculative until a non-SSM scan consumer needs fast execution; no Blelloch precedent in Fuel or baracuda.
- A **symbolic `bound`** (Phase-D symbolic-extent bucket specialization) for variable-length decode — the flash_attn concrete-vs-`Sym` precedent.
- A dedicated **fused associative-reverse-scan backward** (baracuda ships `*_backward_run`) — pure perf, no consumer.
- The **grammar schema** for the general bodied scan (Baracuda's open item 3 / the KISS-Ops higher-structural-op name + `op_attrs`) — fed by this design, pinned in the recipe-grammar reply, not built here.

## Open questions / risks (carried into the plan)

- **Body struct encoding.** Arena NodeId-region is chosen; the exact struct (how `body_entry`/`body_exit`/carry placeholders are stored on the variant while keeping `Op: Clone+PartialEq+Debug`) is a plan-level decision. The invariant: hashable by C2, reachable by C6, remappable by the existing arena machinery.
- **Re-fuse vs retained native arm (C7).** Whether the fused kernel stays reachable via a re-fuse `pattern` or by retaining selective_scan's native arm alongside the new decompose is a plan-level choice; the requirement (fused arm wins, costed, no regression) is fixed.
- **F64-accumulate parity (C4).** Confirm the exact accumulator-widening in the CPU SSM kernel and reproduce it in `unroll_scan`, or the exactness check fails; the tolerance must be sabotage-calibrated.
- **CUDA binding surface (map thin-evidence flag).** `baracuda/mamba.rs` was grep-confirmed but not read line-by-line; verify the exact CUDA binding before the plan asserts "already runs on GPU" as load-bearing.
- **MAJOR vs MINOR for `03-ir`.** Recorded as MAJOR (first sub-graph-carrying primitive); the final judgment is made at write-time against [00-index.md:114](docs/architecture/00-index.md#L114) and stated explicitly in the decisions-log entry.

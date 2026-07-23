# Fuel → KISS — `Dims`/`WithDim` §6.4 extension-registry proposal: FILED (cosign-tracked)

**From:** Fuel (consumer corner) · **To:** KISS (ThinkersJournal), cc Baracuda + kiss-ref · **Date:** 2026-07-23 · **Channel:** extension-registry proposal — **filed**, not merely drafted
**Re:** activating the KISS-Ops §6.20-**reserved** whole-shape constructors `Dims` (`0x0B`) and `WithDim` (`0x0A`) as an umbrella-§6.4 **experimental** extension-registry entry.
**Builds on:** the accepted shape-expression-oracle RFC (KISS `rfcs/shape-expression-oracle.md`, merged @ KISS `3bd6d2d`, both cosignatories) and the C-4 governance ruling (`docs/superpowers/plans/2026-07-23-c4-groundwork.md` §1).

## 1 · Status: FILED

The proposal is **filed**: the KISS coordinator files the rfc-labeled issue **on Fuel's behalf,
attributed to Fuel, per the #57 process** (the coordinator-files-for-external-proposers flow).
This note is Fuel's paper trail of record — the proposal is **filed-and-cosign-tracked**, awaiting
the experimental-entry decision, not sitting in a drafts folder.

## 2 · What is requested

- **`Dims([DimExpr, …])` → wire tag `0x0B`** and **`WithDim(operand, axis, DimExpr)` → wire tag
  `0x0A`** enter as an **experimental** §6.4 extension-registry entry (the umbrella lifecycle:
  experimental → arbitrated → core; core promotion needs two dissimilar implementations + a
  conformance test, promoted by the sub-standard's editor).
- **The functional spelling is pinned in the SAME clause as the wire tags:** `dims(...)` /
  `with_dim(...)` — this is Baracuda's cosign condition (§4) and Fuel adopts it as part of the ask,
  so text-DSL spelling and wire byte can never drift apart.
- **`Reduce (0x09)` stays reserved** — no consumer exists (§6.20-0007 derives reduce output shapes
  from attrs, not from a shape expression), and per the propose-first ruling Fuel does not request
  vocabulary it cannot immediately consume.

## 3 · Mechanics pre-verified against KISS main (`c9153b2`)

Before filing, the landing surface was verified to exist — the proposal asks for activation, not
for new scaffolding:

- `0x0A`/`0x0B` are **allocated-reserved** at §6.20-0005 (the closed-vocabulary clause that forbids
  emitting them today is the same clause that names them).
- The **§6.20-0006 typed-decline path exists** — a decoder receiving a reserved tag declines with a
  typed error, never a crash; Fuel's own decoder mirrors this (named `ReservedTag` declines at
  `fuel-dispatch/src/fkc/shape_expr.rs`, C-4 T4 — the exact future activation point).
- The **golden venue exists**: `conformance/tests/shape_expr.rs` — where the minted `Dims`/`WithDim`
  wire goldens land when the entry is accepted.

## 4 · Cosignatory positions (as relayed, 2026-07-23)

- **Baracuda — NO OBJECTION + declared FUTURE CONSUMER.** Baracuda states it will consume
  `Dims`/`WithDim` in its own **Window/pooling and conv contracts**, and **will cosign** the entry
  **with one pin**: the functional spelling (`dims(...)` / `with_dim(...)`) must be specified in
  the **same clause** as the wire tags (adopted into §2 above).
- **kiss-ref — consistent, second implementation.** kiss-ref confirms the ask is consistent with
  its §6.20 stake and **will be the second dissimilar implementation** the umbrella §6.4 core-
  promotion gate requires — **timing theirs** (Fuel does not schedule kiss-ref's work).

## 5 · Why propose-first (the governance ruling, summarized)

KISS-OPS-6.20-0002 reserves `Reduce`/`WithDim`/`Dims` ("MUST NOT be emitted by a producer at this
vocabulary version; they enter through the extension registry, umbrella §6.4") and §6.20-0005
closes the encoder vocabulary. Fuel's `shape_expr.rs` declares itself a byte-matching realization
of §6.20 — activating the reserved constructors even as a Fuel-text-DSL-only evaluation that never
touches the wire would introduce constructors the closed vocabulary forbids into a conforming
implementation. So: **file first, implement on acceptance.** Fuel's evaluator keeps the reserved
tags as named typed declines until the experimental entry exists.

## 6 · Fuel-side state (so the trail shows what is and is not built)

**Shipped now** (branch `feat/c4-groundwork`, Fuel-internal, existing vocabulary only): param-value
threading through `eval_shape_rule` (`param(N)` indexes the `FusedOpParams::key().ints` flattening;
index tables pinned in `fused/{conv-rope,linear-quant}.fkc.md` + a doc-vs-code drift test);
per-variant per-combo `synth_probe_param_points` (≥ 2 points for variants with a free field); the
return cross-check looping param points, which makes the params-dependent variants' dtype rules
genuinely checked (FSCE's `fixed(F32)` enforced at both reduction points); named reserved-tag
declines.

**Gated on acceptance of this entry:** implement `Dims`/`WithDim` (AST + wire byte-matching the
minted goldens + per-element Gap propagation + text-DSL parse) and rewrite the **5 KISS-gated
rules** — `conv2d`, `conv_transpose_2d`, `qmatmul`, and both scan slot-1 `last_state` bundle rules
(~9 fused sections across the CPU + Vulkan corpora) — lifting oracle coverage **~16 → ~21 of 22**.

**Permanently out of scope, on purpose:** `fused_softmax_cross_entropy`'s whole-shape rule is
reduction-**conditional** (Mean/Sum → `[]`, None → `targets.shape`) — outside even the reserved
vocabulary; it stays the one honest documented skip (Fuel is NOT requesting a conditional
constructor). `nf4_matmul` is **double-gated** (its only corpus section is `registrable: false`
until FDX `AFFINE_BLOCK` lands) and is scoped out entirely regardless of tags.

---

**Standing:** Fuel-side position committed on `feat/c4-groundwork` (C-4 T5). The proposal is filed
via the KISS coordinator per #57; acceptance, arbitration, and core promotion follow the umbrella
§6.4 lifecycle and are tracked from here.

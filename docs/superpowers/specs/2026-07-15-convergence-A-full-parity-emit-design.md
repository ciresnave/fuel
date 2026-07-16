# Convergence Increment A — Full-parity `emit` via a shared `primitive_shape` — design

**Date:** 2026-07-15 · **Status:** design, pre-plan · **Part of:** the Tier-2 recipe-representation convergence (design-doc §8 of `2026-07-14-recipe-identity-verification-and-rope-oracle-design.md`; [[increment-1-recipe-identity-complete]]).

> **User note (2026-07-15):** approved as the chosen foundation, not certain to be final. The two alternatives (an emit-only per-op shape-rule table; routing `emit` through the `Tensor` builders) are on record if `primitive_shape` extraction proves wrong.

> **Grammar-pin update (2026-07-15, after Op::Scan Phase 1 shipped + both Baracuda recipe-grammar replies RELAYED):** the shared recipe grammar is now **pinned** (KISS §6.4-0009 node schema `Op{op_name,op_attrs,child_edges} | Bind` + §6.19 positional-blob `op_attrs` with the §6.19.3 per-op schemas Fuel confirmed — see [[recipe-grammar-codesign]], `docs/outreach/baracuda-recipe-grammar-codesign-reply-2.md`). So this increment **conforms to the pinned grammar rather than leading it**: §A.2 is reframed from "Fuel-led additive `OpAttrs` fields" to "make `OpAttrs` the §6.19 canonical positional blob matching the confirmed per-op schemas." §A.5 (the flat-DAG reply) is **DONE** (it collapsed into the co-design, both replies relayed). The basis-gap boundary shrank (Op::Scan shipped — selective_scan/ssd_chunk_scan now decompose to it), and **Op::Scan is a higher-order body-carrying op explicitly scoped OUT of the first-order `primitive_shape`/`emit`** (its output shape depends on the body sub-DAG, not just leaf input shapes — it gets its own shape rule, not the shared first-order fn).

## Goal

Make the runtime `emit` (`PatternNode` → primitive subgraph, `fuel-graph/src/runtime_fused.rs`) handle the **full first-order op set** — everything except the 4 basis-gap ops (conv2d/conv_transpose_2d/qmatmul/inplace_affine) and the higher-order `Op::Scan` (body-carrying; its own shape rule, see the grammar-pin note) — with correct per-op shape/dtype, so the convergence's decompose-migration (Increment C) becomes pure data-movement. Achieve it by extracting **one** `primitive_shape` function that is the single source of truth for primitive op shape+dtype inference, called by BOTH the `Tensor` builders and `emit` (no drift).

Today `emit` is elementwise-only (every re-emitted node's shape/dtype = `operand[0]`'s, runtime_fused.rs), and `tag_to_op` covers 32 of ~72 `OpTag`s. That forced Increment 1 to realize the rope reference via the hand-written `registry::rope::decompose` instead of a `PatternNode` recipe. Closing this gap is the prerequisite for representing non-elementwise recipes as data.

## Background (what the 6-reader map established)

- `emit` (runtime_fused.rs) — `s = graph.node(child_ids[0]).shape; d = ...dtype` for every op. `tag_to_op(OpTag, &OpAttrs) -> Option<Op>` covers binary arith / unary / activations / `AddScalar`/`MulScalar`; returns `None` (→ `validate_representable` rejects at registration) for shape-changing (Transpose/Permute/Reshape/BroadcastTo/Unsqueeze/Squeeze/Slice/Concat/Flip/Roll/Pad/Triu/Tril), reductions (SumDim/MeanDim/ReduceSumTo/ReduceMaxTo/CumSum), dtype-changing (Cast, comparisons), `PowI`/`Clamp`, MatMul, Where/MaskedFill, indexing, Iota.
- `OpAttrs` (fuel-kernel-seam-types/src/lib.rs:71) — a WIRE type. Carries `scalars: Vec<f64>`, `axis: Option<i64>`, `perm: Vec<u8>`, `target_shape: Vec<i64>`, `dims: Vec<u8>` (perm/target_shape/dims/axis added in F1, 2026-07-01). NO fields for Slice `(start,len)`, Cast's target dtype, reduction keepdim, Pad amounts.
- **No reusable per-op primitive shape-inference fn exists.** Shape math is inline in each `Tensor` builder: `try_permute` (lib.rs:4558), `cast` (:5322), `try_broadcast_to` (:5361), `try_reshape` (:5519), `sum_dim` (:5608), `concat` (:6651), `slice` (:6695), etc. — each validates + computes `out_dims` + pushes a `Node`.
- The graph `Op` enum carries op params in the variant (`Op::Slice{dim,start,len}`, `Op::Reshape(Shape)`, `Op::Concat{dim}`, `Op::Cast(DType)`, …). So after `tag_to_op` builds the `Op`, its params are on the variant — `primitive_shape` reads the `Op`, not `OpAttrs`.

## Design

### A.1 `primitive_shape` — the single source of truth

`#[…] pub fn primitive_shape(op: &Op, input_shapes: &[Shape], input_dtypes: &[DType]) -> Result<(Shape, DType)>` in fuel-graph (co-located with the `Op` enum / a new `shape` module). Given a primitive `Op` (params on the variant) + its inputs, returns the output shape + dtype. It is the ONE place that answers "what does this primitive op produce."

**Extraction:** move the `out_dims`/dtype computation out of each `Tensor` builder method into `primitive_shape`; the builder keeps its argument validation, its `Node` push, and its `Arc<RwLock<Graph>>` handling, but obtains the output shape/dtype by *calling* `primitive_shape`. This removes the drift hazard (two places computing "what shape does Slice produce"). A primitive with no builder gets a fresh arm.

**dtype:** most ops are dtype-preserving (= `input_dtypes[0]`); `Cast` → its target dtype; comparisons (Equal/Ne/Lt/Le/Gt/Ge) → `U8`. `primitive_shape` returns both so `emit` (and the migration) get dtype right, not just shape.

### A.2 `OpAttrs` → the §6.19 canonical positional blob (conform to the pinned grammar)

Two parts, both now targeting the **pinned** grammar (not leading it):

1. **Field coverage.** Extend `OpAttrs` (fuel-kernel-seam-types) with the fields the full first-order set needs but can't express today: Slice's `start`/`len` (dim rides `axis`), Cast's target dtype, reduction `keepdim`, Pad's per-dim amounts. Additive optional fields → backward-compatible; the frozen `size_of`/layout tests update. Confirm the minimal set against what `tag_to_op` needs to reconstruct each `Op` variant.
2. **Canonical §6.19 serialization (the pinned-grammar conformance — the §2.A gap fix).** Give `OpAttrs` a canonical, no-elision **positional little-endian blob** serialization matching the **§6.19.3 per-op schemas Fuel confirmed** in `baracuda-recipe-grammar-codesign-reply-2.md`: `reduce{monoid,axes,keepdim}`, `gather{axis,oob,index_operand,index_dtype}`, `scatter{axis,scatter_combine,oob,index_operand,index_dtype}`, etc.; an op with an empty schema serializes as a **zero-length** length-prefixed blob (one canonical byte form). Fuel's internal `OpAttrs` may stay a struct as long as it has this canonical serialization. This is what makes a Fuel recipe byte-comparable with a Baracuda-emitted one. Record the encoding in `kernel-seam-interop.md`. (The `scan`/`scan_placeholder`/`runtime_scalar` tokens Fuel proposed are higher-order / leaf ops handled outside this first-order `OpAttrs` set.)

### A.3 `emit` + `tag_to_op` + `validate_representable` growth

- `tag_to_op(OpTag, &OpAttrs) -> Option<Op>` grows to build every non-basis-gap `Op` from its `OpTag` + the (extended) `OpAttrs` (e.g. `OpTag::Slice` + `axis`/`start`/`len` → `Op::Slice{dim,start,len}`).
- `emit` computes each re-emitted node's `(shape, dtype)` via `primitive_shape(&op, &child_shapes, &child_dtypes)` instead of the `operand[0]` shortcut.
- `validate_representable` (register-time) now accepts the newly-covered `OpTag`s (its accept set = `tag_to_op`'s coverage, unchanged mechanism).

### A.4 Validation — the migration oracle

The existing hand-written `registry::*::decompose` fns are the ground truth. For representative decomposes spanning the new capability, express the region as a `PatternNode` and assert the grown-`emit` re-emission is **byte-for-byte identical** (same op sequence, same shapes/dtypes, same node structure) to the `registry::*::decompose` output:
- **rope** — shape-changing (Reshape/BroadcastTo/Slice/Neg/Concat/Mul/Add).
- **softmax_last_dim** — reduction (ReduceMaxTo/BroadcastTo/Sub/Exp/ReduceSumTo/Div).
- **layer_norm_last_dim** — reduction + broadcast + `AddScalar`/`Sqrt`/`Div`.
- (optionally a `Cast`-bearing region to exercise dtype.)
This is both the acceptance test for A and the de-risking for the C migration (an extraction error surfaces as a decompose mismatch, not a silent wrong shape).

### A.5 The Baracuda flat-DAG reply — ✅ DONE (superseded)

This deliverable is **complete**. The flat-DAG reply collapsed into the recipe-grammar co-design, and **both replies are relayed** (2026-07-15): `baracuda-recipe-grammar-codesign-reply.md` (positions on all 6 questions — the canonicalization rule *is* `base_map_hash`, cap bit in the KISS FEAT range) + `baracuda-recipe-grammar-codesign-reply-2.md` (KISS §6.4-0009 schema adoption, all 4 open items). The flat-DAG *container code* remains a later convergence step; no correspondence deliverable is left in Increment A.

## Error handling / never-panic

`primitive_shape` returns `Result` (a malformed op/shape → `Err`, never a panic). `emit` currently `.expect()`s `tag_to_op` (a non-re-emittable op) — the accept set from `validate_representable` guarantees a registered region only contains re-emittable ops, so the `.expect` stays reachable only for a validation bug; keep it (its consumers wrap `emit` in `catch_unwind` — Increment 1's `recipe_identity_matches` + `register_runtime_fused`). No new `.unwrap()`/`.expect()` on production paths. The `Tensor` builders keep their existing `Result`/validation behavior (they must not regress — the byte-for-byte tests + the full builder test suites are the gate).

## Testing

The byte-for-byte decompose-diff tests (A.4) + the full fuel-graph builder test suite (unchanged behavior after the extraction) + `emit`/`register_runtime_fused` tests (newly-covered ops now register + re-emit). TDD, born-red. No GPU (pure fuel-graph shape/emit logic).

## Boundaries (explicitly NOT in Increment A)

- **The decompose MIGRATION** (Increment C — moving the ~16 migratable `registry::*::decompose` Rust fns to `PatternNode` data). A only makes `emit` *capable*; the migration is C.
- **Unify internal+external registry + wire the 18 stubbed matchers** (Increment D).
- **KISC framing** (Increment E) and the **flat-DAG container code** (the reply is done — §A.5).
- The 4 remaining **basis-gap ops** — `conv2d`/`conv_transpose_2d` (need `Im2Col`/`Col2Im`), `qmatmul` (GGUF unpack), `inplace_affine` (`AffineInplace`) — excluded until their IR primitives land. (The higher-order `Scan` basis gap is CLOSED — Op::Scan shipped 2026-07-15.)
- **`Op::Scan` itself** (the shipped higher-order body-carrying primitive) — explicitly OUT of the first-order `primitive_shape`/`emit` (its shape depends on the body sub-DAG, not just leaf input shapes). It gets its own shape rule; migrating scan-bearing decomposes to a `PatternNode`-with-scan form is later convergence work (the scan grammar Fuel just proposed to Baracuda), not Increment A.

## Open questions / risks

- **Extraction blast radius:** ~20 `Tensor` builders route through `primitive_shape`; a large but mechanical refactor. The builders' own test suites + the byte-for-byte decompose diffs are the safety net. If the extraction proves too invasive, the fallback is the emit-only shape-rule table (accepting the drift risk) — the user-noted alternative.
- **`OpAttrs` minimal field set:** confirm exactly which fields `tag_to_op` needs to reconstruct each `Op` variant (some may already be expressible via `axis`/`dims`/`target_shape`); add only the genuinely-missing ones.
- **Wire-type sensitivity:** `OpAttrs` is sent to Baracuda; the additive increment is Fuel-led but should be recorded in `kernel-seam-interop.md` + flagged for Baracuda to mirror (like F1). Coordinate the field encoding with the eventual flat-DAG grammar so they don't conflict.
- **`Cast`/comparison dtype in `emit`:** the elementwise-only `emit` never changed dtype; `primitive_shape` returning dtype is the fix — confirm every `emit` caller uses the returned dtype (not a carried-over one).

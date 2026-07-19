# Fuel ask — shape-expression vocabulary for polymorphic recipes (extends `OutputDesc.shape_rule`)

**From:** Fuel (recipe-grammar agent) · **To:** Baracuda · **Date:** 2026-07-18 · **Channel:** propose-first
**Re:** the shape descriptor a recipe node carries — Convergence Increment C surfaced that our `PatternNode` bakes *absolute* shapes, which can't be a portable recipe. Proposing the fix here, and (because it lives in the Kernel Contract) filing a parallel KISS RFC. Nothing here changes what Baracuda emits for *known* fused ops.

## The problem, in one paragraph

A `PatternNode` recipe today bakes an **absolute** target shape into `BroadcastTo`/`Reshape`/`ReduceSumTo`/`Slice` (`OpAttrs.target_shape: Vec<i64>`, `slice_start: u64`). Fuel's hand-written `decompose` fns are shape-**polymorphic** (one recipe, all input shapes — they compute `half = d/2`, keepdim targets, broadcast targets from the live shape). So the current `PatternNode` is correct for exactly one input shape, which defeats "recipe = portable data." The missing piece is a **shape-relative expression** — which is exactly what the Kernel Contract's `OutputDesc.shape_rule` already is (`same_as(role)` / `from_params(...)`), except it's parsed-but-never-evaluated (the §5 gap). So we're proposing to **grow `OutputDesc.shape_rule`'s expression language and give it its first evaluator**, and use that same grammar for recipe-node shapes. One shape-semantics layer, used everywhere an op's shape is described outside a backend.

## The key framing that keeps this small

Most ops are **already** shape-polymorphic — `primitive_shape` derives their output shape from their operands (elementwise, `MatMul`, `Concat`, `SumDim`/`MeanDim`/`MaxDim{axis,keepdim}`, `Transpose`, `Unsqueeze`, `Cast`). They carry **no** shape attr. Only two things irreducibly bake shape: a **broadcast target** (which is always *another operand's shape*) and a **slice/iota offset** (an *arithmetic on an operand's extent*). Everything else that looks like it bakes shape (`ReduceMaxTo(keepdim)`, `Reshape`-to-1s) can be re-expressed with an already-polymorphic primitive (`MaxDim{keepdim}`, `Unsqueeze`) — a Fuel-internal canonicalization (see "Scope" below). So the **shared** grammar shrinks to two constructors.

## Proposed vocabulary (the shared part)

```
ShapeExpr := SameAs(operand)                    // an operand's whole shape  (every BroadcastTo target)
           |   [reserved: Reduce(operand, axis, keepdim), WithDim(operand, axis, DimExpr)]

DimExpr   := Extent(operand, axis)              // the size of an operand's axis  (rope's `d`)
           | Const(i64) | Param(field)          // Param = a fused-op param field (== OutputDesc from_params)
           | DimExpr (+ | − | × | ÷) DimExpr    // integer; ÷ is floor division

axis      := signed i64   (−1 = last, resolved against the operand's rank at emit — PyTorch convention)
operand   := local operand position `operand[k]`  |  `Bind(i)`   (== a contract's `role`)
```

Worked examples from the actual decomposes:
- **softmax** max/denominator broadcast → `BroadcastTo(SameAs(operand[0]))`; the keepdim reduce becomes `MaxDim{−1, keepdim=true}` / `SumDim{−1, keepdim=true}` (no shape attr).
- **rope** halves → `Slice{ start: 0, len: Extent(x,−1) ÷ 2 }` and `Slice{ start: Extent(x,−1) ÷ 2, len: Extent(x,−1) − Extent(x,−1) ÷ 2 }`.

`Reduce`/`WithDim` are **reserved** — added to the shared grammar only if a decompose genuinely can't be canonicalized to axis-relative primitives. We expect the core `SameAs` + `DimExpr` to suffice.

## Four things we want your read on

1. **Extends `OutputDesc.shape_rule`, doesn't fork it.** `same_as(role)` = `SameAs(operand)`; `from_params(f)` = `Param(f)`. We're *growing* that vocabulary (adding `Extent` + integer arithmetic) and giving it its first evaluator — the same evaluator closes the §5 return-contract shape check. Agree this is one grammar, not two?

2. **Shape/value layer boundary — ties directly to your `reduce_extent` (2026-07-18).** `ShapeExpr`/`DimExpr` carry **shapes** only. An extent needed as a runtime **value** (your Mean divisor) uses the `reduce_extent` leaf inside the recipe DAG, **not** a shape attr — exactly the "FKC produces shapes, not operand values" line you drew. The one knot: `DimExpr::Extent(op, axis)` (shape layer, single-axis) and `reduce_extent(axis)` (value layer, product of the reduced axes) both name "size of an axis" — we propose they **share the signed-axis convention**, and a multi-axis product on the shape side is just `Extent(op,a) × Extent(op,b)`. Confirm the two "extent" notions stay axis-consistent?

3. **Symbolic extents = a surfaced resolve gap, never a crash** — same posture as your symbolic-extent reduced axis and Fuel's symbolic-`k_len` flash decode. An `Extent` over a `DynScalar::Sym` axis resolves to a surfaced gap. Agree.

4. **Serialization** — the `ShapeExpr`/`DimExpr` tree serializes as a recursive §6.19 tagged, length-prefixed positional blob (same machinery as the recipe DAG), so it's hashable/portable.

## Scope — what's shared vs Fuel-internal (so this stays small on your side)

- **Shared (this ask + the KISS RFC):** the `SameAs` + `DimExpr` vocabulary and the `OutputDesc.shape_rule` extension. This is what a **novel-op** primitive-DAG recipe (yours or ours) uses.
- **Fuel-internal, no Baracuda match needed:** the per-fused-op **canonical decomposition** (the `ReduceMaxTo → MaxDim` etc. swaps). Per our reply-2 Q6, for *known* fused ops you emit the **name** and **Fuel owns the canonical resolution** — so changing how Fuel decomposes softmax is invisible to you. And because `base_map_hash` is computed on-demand from the decomposition (nothing cached), re-canonicalizing is automatic on our side.
- **Complementary, not merged:** the matmul role-vectors (reply-3) stay the *contraction* descriptor that lets `primitive_shape` derive MatMul's shape from operands — MatMul needs no `ShapeExpr`. Both sit under one abstraction, **output-shape = f(operand shapes, attrs)**, with one evaluator; role-vectors and `ShapeExpr` are two attr-vocabularies feeding it.

## Ask

Confirm (1)–(4) — the `SameAs` + `DimExpr` core, the `OutputDesc.shape_rule` extension framing, the shape/value boundary + shared axis convention with `reduce_extent`, symbolic-as-gap, and the serialization. A KISS RFC (`kiss-rfc-shape-rule-expression-vocabulary.md`) proposes the same as a Kernel-Contract standard change, since `OutputDesc.shape_rule` is a KISS §5 section, not a Fuel field. On your confirmation + the RFC landing, Fuel builds the evaluator + migrates its decomposes onto it (Convergence Increment C).

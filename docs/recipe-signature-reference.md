# Fuel recipe-signature reference

Maintainer reference for the **base-op recipe / pattern signature** — the op-DAG a fused kernel decomposes into (its `decompose`) and re-fuses from (its `pattern`). Every factual claim is anchored to a repo-relative `file:line` rendered as inline code. Anchors were verified against the current tree; where an earlier draft anchor was off it is corrected here.

## What a recipe is

A **recipe** (equivalently a **region**, a contract's `pattern:`, or a synthesized op's `decompose`) is a single op-DAG grammar object. It expresses "this fused kernel is *exactly* this subgraph of primitive ops." Fuel lowers every fused op to that subgraph (the **base map**), and the optimizer's whole job is "lower to base map, then find the best cover" — so a recipe is not documentation, it is the operational identity of the fused op.

The grammar object is `PatternNode`, and it plays **one type, three roles** (`fuel-kernel-seam-types/src/lib.rs:9-13`, echoed at `fuel-graph/src/jit.rs:9-11`):

1. a **JIT region** handed Fuel → synthesizer ("build a kernel for this subgraph");
2. a contract's `pattern:` **re-fuse rule** (primitive subgraph → `Op::Fused`);
3. a synthesized op's **`decompose`** (the region re-emitted as primitives).

The co-design with Baracuda unifies a fourth: the KISS-Contract §2.3 Semantics op-DAG (`docs/outreach/baracuda-recipe-grammar-codesign-reply.md:6-8`). All four are "one op-DAG grammar."

## The load-bearing split: STRUCTURE vs INTERFACE

The single most important architectural fact in this document:

> **The recipe DAG is dtype-agnostic and shape-free. Dtypes, shape-rules, cost, precision, and determinism ride the FKC contract that *wraps* the fused op.**

A recipe node carries an op tag, ordered tensor edges, and non-tensor attributes — but **no dtype** (except `Cast`'s target) and **no shape**. Those are *derived* at emit time from the concrete operands by `primitive_shape` (Part I §3), and the *accepted* dtypes / *declared* output shape+dtype rules / cost / precision live on a separate FKC contract joined to the op at runtime by `FusedOpId` (Part I §7). The recipe answers **structure + identity**; the contract answers **interface**. Keep the two apart — most of the in-flight work in Part II is about doing that split cleanly at scale.

The grammar types live in a deliberately dependency-free crate, `fuel-kernel-seam-types` (`fuel-kernel-seam-types/src/lib.rs:3-5`): types only, no `fuel_graph`/`fuel_ir` dependency, so a synthesizer backend (Baracuda) can depend on the grammar without pulling in the Fuel graph. The Fuel-side projection + matcher lives in `fuel-graph/src/jit.rs` and `fuel-graph/src/runtime_fused.rs`; the shape/dtype single-source-of-truth in `fuel-graph/src/shape.rs`; recipe identity in `fuel-graph/src/opt.rs`; the static fused-op catalog in `fuel-graph/src/registry.rs`; the contract wrapper in `fuel-dispatch/src/fkc/`.

**Out of scope for this reference (intentional).** This document covers the recipe *signature* — structure, attributes, canonical identity — and the three in-flight realizations (Part II). It deliberately does not detail: the matcher-walk internals beyond the note in Part I §1; autodiff (`BackwardKind` — how a `decompose` relates to the backward pass); the `extract:` runtime-scalar-slot extraction flow behind `FusedOpParams::Runtime { scalars }`; or `output_views` multi-output recipe handling. Each is named where its struct appears (Part I §5) and left to its own reference.

---

# Part I — The recipe signature (as-built)

## 1. The `PatternNode` node signature

`fuel-kernel-seam-types/src/lib.rs:254-273`:

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum PatternNode {
    Op {
        op: OpTag,
        operands: Vec<PatternNode>,   // one child per tensor input, exact arity, held INLINE by value
        attrs: OpAttrs,
    },
    Bind { index: u8 },               // leaf: bind the producing node as fused input[index]
    SeeThrough { then: Box<PatternNode> },  // matcher-only: skip a transparent wrapper
    Any,                              // matcher-only: wildcard
}
```

### Variant classification

- **`Op` and `Bind` appear in a concrete recipe/region.** A region populates only these two (`lib.rs:249-253`).
- **`SeeThrough` and `Any` are matcher-only** — they "never appear in a concrete region" (`lib.rs:269-272`). This is *enforced*: `validate_representable` rejects them with `RuntimeFusedError::NonConcreteRegion` (`fuel-graph/src/runtime_fused.rs:427-429`), and `emit` treats them as `unreachable!` (`runtime_fused.rs:536-538`).

**The matcher walk.** `SeeThrough` and `Any` exist only for the `pattern:` (re-fuse) direction — the matcher that walks a `PatternTree` root via `crate::jit::match_region` (Part I §5). `SeeThrough` matches its inner node after skipping a transparent wrapper; `Any` matches any single node; operand matching additionally honors `attrs_match` wildcard-on-unset (below), the repeated-`Bind` node-identity guard (`jit.rs:279-288`), and commutative-operand order. The `decompose`/`emit` (build) direction never produces either variant — that is what `validate_representable` enforces. (This reference does not further detail the matcher internals; the two build directions are its focus.)

### Tensor edges and arity

`Op.operands` is **exact tensor arity** — one child per tensor input. Scalar params are attributes, not operands (arity mismatch is rejected at `jit.rs:311-315`; the invariant is documented `lib.rs:256-258`). Note that operands are held **inline, by value** — there is no `Ref(node_index)` / indexed-edge variant. Interior computed nodes therefore cannot be shared in the current representation; only external *inputs* can appear in two places, via a repeated `Bind` (see Part II §A for why this is the thing the flat-DAG migration changes).

### `Bind` — the input leaf

`Bind { index }` is the only concrete-recipe leaf. `bind_indices()` (`lib.rs:278-298`) collects the distinct `Bind` indices post-order, sorted + deduped. A region's binds MUST form a contiguous `[0, n_inputs)` — validated at registration (`runtime_fused.rs:152-156`, error `NonContiguousBinds`) and re-checked after a match (`jit.rs:259-265`). A **repeated `index`** is a node-identity guard on a shared *input* (`lib.rs:264-267`; tested at `lib.rs:322-331`, where `mul(x, x)` → binds `[0]`); the matcher enforces "same index must bind the SAME node" at `jit.rs:279-288`.

### `OpAttrs` — the non-tensor attribute fields (13 concrete + 6 Increment-C-slice-1 additions)

`fuel-kernel-seam-types/src/lib.rs:70-190`, `#[derive(Clone, Debug, Default, PartialEq)]`. Each field rides specific `OpTag`s; an **unset** field is a matcher wildcard (see below). The 13 fields below are the concrete, serialized set; six more were added in Increment C slice 1 (the note after the table).

| Field | Type | Line | Rides which OpTag(s) | Notes |
|---|---|---|---|---|
| `scalars` | `Vec<f64>` | `:74` | `AddScalar`/`MulScalar`/`Clamp`/`PowI`; `MaskedFill` value | snapshot of the slot, **not baked** — re-read live via `extract:` |
| `axis` | `Option<i64>` | `:76` | reductions, `Triu`/`Tril` diagonal, `Slice`/`Concat`/`Flip`/`Roll`/`CumSum`/`IndexSelect`/`Gather`/`IndexAdd`/`ScatterAdd` | |
| `perm` | `Vec<u8>` | `:82` | `Permute`/`Transpose` | **ABSOLUTE** perm (`out.axis[i]=in.axis[perm[i]]`); empty ⇒ wildcard |
| `target_shape` | `Vec<i64>` | `:86` | `BroadcastTo`/`Reshape`/`ReduceSumTo`/`ReduceMaxTo`/`Iota` | LOGICAL output shape; one field, OpTag disambiguates; `Iota` len rides it |
| `dims` | `Vec<u8>` | `:90` | `Squeeze`/`Unsqueeze` | Fuel emits a one-element list |
| `cast_dtype` | `Option<String>` | `:95` | `Cast` target; `MaskedFill` value dtype | stable `DType::as_str()` name (crate can't ref `fuel_ir::DType`) |
| `slice_start` | `Option<u64>` | `:98` | `Slice` | dim rides `axis` |
| `slice_len` | `Option<u64>` | `:101` | `Slice` | |
| `roll_shift` | `Option<i64>` | `:104` | `Roll` | signed; dim rides `axis` |
| `pad_amounts` | `Vec<(u64,u64)>` | `:107` | `Pad` | per-axis `(before, after)` |
| `pad_mode` | `Option<u8>` | `:111` | `Pad` | `0=Constant, 1=Reflect, 2=Replicate` |
| `pad_value` | `Option<f64>` | `:114` | `Pad` | `PadMode::Constant` fill |
| `keepdim` | `Option<bool>` | `:119` | reduce-schema conformance (`SumDim`/`MeanDim`/`CumSum` serialize it) | **NOT consumed by `tag_to_op`** — Fuel encodes keepdim structurally |

The fields `slice_start` … `keepdim` are the Convergence-Increment-A additions.

**Increment C slice 1 added six more fields** (`lib.rs:151-190`), all `Default`-empty so existing regions are unchanged. Four are the shape-**relative** recipe-interior carriers, resolved to the concrete fields above at emit time and **not serialized on the wire** this slice (§A "Shipped in slice 1", D2): `target_shape_rel: Option<ShapeExpr>`, `slice_start_rel`/`slice_len_rel: Option<Dim>`, `axis_last: bool`. Two are the matmul role vectors, **serialized** into the §6.19 blob (§C, T9): `lhs_roles`/`rhs_roles: Vec<u8>`. Neither group is consulted by `attrs_match` this slice.

### `op_to_attrs` — graph-side projection

`fuel-graph/src/jit.rs:139-202` reads typed `Op` payloads (`Op::Permute(Vec<usize>)`, `Op::AddScalar(f64)`, …) into the flat `OpAttrs` surface so the seam-types crate stays graph-free. It is **matcher-driven** (`jit.rs:128-138`): it fills only the fields `attrs_match` needs today, so the axis-bearing ops `tag_to_op` *can* reconstruct (`CumSum`/`IndexSelect`/`Gather`/`IndexAdd`/`ScatterAdd`) are deliberately **not** projected (the `_ => {}` arm at `jit.rs:199`). This is a projection-only gap; the re-emit path gets attrs from the region author directly.

### Wildcard-on-unset matching — `attrs_match`

`fuel-graph/src/jit.rs:213-219`:

```rust
fn attrs_match(pattern: &OpAttrs, node: &OpAttrs) -> bool {
    (pattern.scalars.is_empty()        || pattern.scalars == node.scalars)
        && (pattern.axis.is_none()         || pattern.axis == node.axis)
        && (pattern.perm.is_empty()        || pattern.perm == node.perm)
        && (pattern.target_shape.is_empty()|| pattern.target_shape == node.target_shape)
        && (pattern.dims.is_empty()        || pattern.dims == node.dims)
}
```

An **unset** field on the pattern (`Vec` empty / `Option` `None`) is a wildcard (matches any graph value); a **set** field must equal exactly. This is what keeps every existing attr-agnostic pattern (authored with `OpAttrs::default()`) matching after attrs became comparable, while letting a layout/scalar pattern discriminate. Note that only **5 of the 19 fields** are consulted here — the Convergence-A additions and the Increment-C-slice-1 role vectors serialize into the canonical blob (§6), and the slice-1 rel fields stay off the wire (§A), but none is yet a matcher predicate.

## 2. The `OpTag` base-op vocabulary

`fuel-kernel-seam-types/src/lib.rs:30-59`, `#[non_exhaustive]`, `Clone+Copy+Debug+PartialEq+Eq+Hash`. The functional-op vocabulary, by category:

| Category | Line | Tags |
|---|---|---|
| binary arithmetic / extremum | `:34` | `Add, Sub, Mul, Div, Maximum, Minimum, Pow, Rem` |
| unary math | `:36` | `Neg, Abs, Sqr, Sqrt, Rsqrt, Recip, Exp, Log, Sin, Cos` |
| activations | `:38` | `Tanh, Sigmoid, Silu, Gelu, GeluErf, Relu, Erf, Step` |
| rounding / sign | `:40` | `Floor, Ceil, Round, Sign` |
| scalar-param | `:42` | `AddScalar, MulScalar, PowI, Clamp` |
| comparison → U8 mask | `:44` | `Equal, Ne, Lt, Le, Gt, Ge` |
| select / mask | `:46` | `Where, MaskedFill` |
| reductions | `:48` | `SumAll, MaxAll, MinAll, MeanAll, SumDim, MeanDim, ReduceSumTo, ReduceMaxTo, CumSum` |
| matmul | `:50` | `MatMul` |
| shape / layout | `:52` | `Transpose, Permute, Reshape, BroadcastTo, Unsqueeze, Squeeze, Cast, Slice, Concat, Flip, Roll, Pad, Triu, Tril` |
| indexing / gather-scatter | `:54` | `IndexSelect, Gather, IndexAdd, ScatterAdd` |
| fused-primitive helper | `:56` | `LogSoftmaxLastDim` |
| value source | `:58` | `Iota` |

The activations comment (`lib.rs:37`) pins **`Gelu` = tanh-approx and `GeluErf` = exact erf as DISTINCT tags**. That distinction round-trips through `op_to_tag` (`jit.rs:52-53`), is asserted at `jit.rs:373-376` (`assert_ne!(op_to_tag(&Op::Gelu), op_to_tag(&Op::GeluErf))`), and is re-emittable both ways (`runtime_fused.rs:289-290`).

### Exclusions

The seam-types comment `lib.rs:23-29` lists what is deliberately **outside** the vocabulary: **in-place variants** (a region is the *functional* subgraph; in-place is a Fuel-side scheduling rewrite) and **structural/bookkeeping ops** (`Const`, `Release`, `Alloc`, views). `Op::Fused` itself is also excluded — a fused op is not a region node, its *decomposition* is — but that specific exclusion is documented on the `op_to_tag` side (`jit.rs:22-28`), not in the seam-types comment, which omits it.

### `op_to_tag` (Op → OpTag)

`fuel-graph/src/jit.rs:29-107`. Returns `Option<OpTag>`; the `_ => return None` arm at `jit.rs:105` is the honest miss for in-place, structural/bookkeeping, `Op::Fused`, `Op::Scan`, and `Op::ScanPlaceholder` — never a crash (`jit.rs:22-28`). `Op::Scan`/`Op::ScanPlaceholder` are excluded because a scan isn't a region node — its body is referenced via `inputs`, not a `PatternNode`.

### `tag_to_op` (OpTag + OpAttrs → Op)

`fuel-graph/src/runtime_fused.rs:263-368`. Reconstructs a primitive `Op` from a tag + `OpAttrs`, over the full first-order re-emit vocabulary (`runtime_fused.rs:250-262`). Structural params are decoded from `OpAttrs`. **Honest misses** (return `None`, rejected at registration, `runtime_fused.rs:363-366`): `PowI`/`Clamp` (no i32/two-scalar carrier), `MaskedFill` (no `Scalar::from_f64` reconstructor yet), fused/basis-gap tags, and any tag whose required attrs are unset (e.g. `Iota` with no `target_shape`, tested at `runtime_fused.rs:680`).

## 3. What a node deliberately does NOT carry — and why

A recipe node stores **no shape** and **no dtype**. The only dtype anywhere on a recipe node is `Cast`'s `cast_dtype` (§1). Everything else is *derived* from operands at emit time.

### `primitive_shape` — the single source of truth

`fuel-graph/src/shape.rs:36-40`:

```rust
pub fn primitive_shape(
    op: &Op,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
) -> Result<(Shape, DType), Error>
```

This is the single place that answers "what does this primitive `Op` produce" (`shape.rs:1-5`, `:20-25`). It is called by **both** the `Tensor` builders (`fuel-graph/src/lib.rs`) and the runtime `emit` re-emitter (`runtime_fused.rs:527`). It **reads params off the `Op` variant, not off `OpAttrs`** (`shape.rs:22`), and **never panics** — a malformed op/shape, a leaf/bookkeeping op with no pure inference, or a higher-order/basis-gap op is an honest `Err` (the `_ => return Err(...)` arm at `shape.rs:241-246`).

Contrast the *emitted graph* node, which does carry shape+dtype (`fuel-graph/src/lib.rs:1442-1447`):

```rust
pub struct Node { pub op: Op, pub inputs: Vec<NodeId>, pub shape: Shape, pub dtype: DType }
```

`emit` (`runtime_fused.rs:527-533`) calls `primitive_shape` for every re-emitted node, and only for a *malformed authored region* where it errs falls back to `operand[0]`'s shape/dtype (or a degenerate rank-0 F32 for a zero-operand leaf) — so emit is total and never panics (tested `runtime_fused.rs:740-756`).

Key inference rules (all in `shape.rs`):

- elementwise unary/binary/scalar-param → `(in[0].shape, in[0].dtype)` (`shape.rs:49-55`)
- comparison (`Equal/Ne/Lt/Le/Gt/Ge`) → `(in[0].shape, DType::U8)` (`shape.rs:58-61`)
- `Where` → `(cond=in[0].shape, dtype=a=in[1].dtype)` (`shape.rs:64-67`)
- `Cast(dt)` → `(in[0].shape, dt)` (`shape.rs:70-73`) — **the one dtype exception**
- `Reshape`/`BroadcastTo`/`ReduceSumTo`/`ReduceMaxTo` → target shape carried on the variant (`shape.rs:76-79`)
- `MatMul` → same-rank operands, contracts inner dim (`shape.rs:189-212`; see Part II §C)
- `Iota { len }` → `([len], DType::F32)` (`shape.rs:238`)

`primitive_shape` is intentionally **more permissive than the `Tensor` builders** (`shape.rs:27-35`): it computes derived shape math but does NOT re-run broadcast-compat / squeeze size-1 / reshape elem-count / permute duplicate-axis preconditions — the builders validate those before calling in, and emit re-emits already-validated recipes. It still range-checks its own arithmetic, so a malformed region is `Err`, never an OOB panic.

### Why dtype-agnostic (Q7)

Because `emit` re-derives shape+dtype per node from the concrete input nodes (`runtime_fused.rs:523-534`), **one region serves any input dtype**. The `Cast` region tests confirm the target dtype is taken from the tag, not `operand[0]` (`runtime_fused.rs:722-737`). This is the co-design's Q7 dtype-agnostic-DAG answer: dtypes ride the interface (Part I §7), not the structure.

## 4. Leaf / source-op kinds — shipped vs pinned

### Shipped

In the as-built `PatternNode` grammar the **only** concrete-recipe leaf is `Bind { index }` (`lib.rs:264-267`). `Iota` is **not** a leaf of `PatternNode` — it is a `PatternNode::Op { op: OpTag::Iota, .. }` with `len` riding `target_shape` (`OpTag` at `lib.rs:58`; `op_to_attrs` at `jit.rs:177`; re-emit at `runtime_fused.rs:356`).

At the *graph* `Op` level the leaf/source ops that exist are `Op::Const` (`lib.rs:228`), `Op::Iota { len }` (`lib.rs:238`), and `Op::ScanPlaceholder { role: ScanRole, index }` (`lib.rs:1147-1150`) — but `Op::Const`/`Op::Scan`/`Op::ScanPlaceholder` are *outside* the `OpTag` vocabulary (`op_to_tag` returns `None`, `jit.rs:105`), so they never appear as recipe nodes. `Op::Scan` is a terminal in the base map — it *is* the primitive, no `LoweringRule` matches it (`lib.rs:1131-1137`). The shipped scan-body enums are `ScanEmit { All, Final }` (`lib.rs:1158-1162`) and `ScanRole { Carry, Elem }` (`lib.rs:1167-1171`).

### Pinned (co-designed) — byte arms SHIPPED, graph wiring not

**Four of the five leaf tokens now have a SHIPPED byte arm** (KISS editor ack, 2026-07-23,
"RULING RECORD — four-leaf-arm ack"): `OpTag::{Const, RuntimeScalar, ReducedCount,
ScanPlaceholder}` serialize per §6 of this document, with golden-byte tests. What shipped is
the **wire token + its `op_attrs` body**, nothing more: `jit::op_to_tag` emits none of the four
and `runtime_fused::tag_to_op` declines all four as honest misses (the `_ => return None`
arms), so they never appear in a live Fuel recipe. `OpTag::Const` is the KISS **scalar**
literal leaf and is deliberately NOT wired to `Op::Const` (a constant **tensor** / weight
leaf) — different concepts sharing a name. Making the leaves first-class `PatternNode` nodes
is the flat-DAG-CSE recipe interior (Part II §A), still unbuilt.

The co-designed source-op **leaf** vocabulary — a value a recipe needs as an operand is a source-op leaf inside the recipe DAG (`docs/outreach/kiss-rfc-shape-rule-expression-vocabulary.md:51-53`, the `reduce_extent{axis}` leaf, now renamed `reduced_count`, Part II §B). The broader pinned set is `const{bits}` / `iota{axis}` / `runtime_scalar{slot}` / `scan_placeholder{role,index}` / `reduced_count{axes}`, kept under one abstraction `output-shape = f(operand shapes, attrs)` (`kiss-rfc-shape-rule-expression-vocabulary.md:62`). Today Fuel expresses these implicitly (a `Const` operand, an `Iota`, a `Runtime{scalars}` slot) rather than as first-class `PatternNode` leaf variants — the flat-DAG target (Part II §A) adds them as **op tokens**, not schema variants.

## 5. The two directions: `decompose` vs `pattern`

`fuel-graph/src/registry.rs`.

### `FusedOpEntry`

`registry.rs:92-149`:

```rust
pub struct FusedOpEntry {
    pub id: FusedOpId,
    pub name: &'static str,
    pub family: FusedOpFamily,                                          // Forward|Backward|Quantized|Attention|Norm (:153-160)
    pub pattern: SubgraphPattern,                                       // re-fuse rule
    pub decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,    // :112 — fused → primitive subgraph
    pub backward: BackwardKind,                                         // Fused(id)|Decompose|NotDifferentiable (:657-662)
    pub shape_rule: fn(&[Shape], &FusedOpParams) -> Shape,              // slot-0 shape (:118)
    pub dtype_rule: fn(&[DType], &FusedOpParams) -> DType,              // slot-0 dtype
    pub output_views: Option<fn(&[Shape], &[DType], &FusedOpParams) -> Vec<OutputViewSpec>>,  // multi-output
}
```

**`decompose` is the imperative direction** (`registry.rs:105-112`): it *appends primitive nodes to `graph`* and *returns the `NodeId` of the new root* that replaces the fused node (the 2nd arg). The fused node stays in the arena; the driver rewrites consumer edges to the returned id. This is the fused → primitive-subgraph builder — the recipe itself, hand-written today as `&mut Graph` Rust.

### `SubgraphPattern` — the re-fuse direction

`registry.rs:678-681`:

```rust
pub enum SubgraphPattern {
    Declarative(PatternTree),                              // analyzable; compiled to a matcher
    Callable(fn(&Graph, NodeId) -> Option<PatternMatch>), // closure matcher; maximally flexible
}
```

- **`PatternTree`** (`registry.rs:688-697`): `{ root: PatternNode, params: FusedOpParams }` — the §1 grammar root (the subgraph *sink*) plus the `FusedOpParams` to stamp on the matched fused node. The rule engine walks `root` via `crate::jit::match_region` and emits `Op::Fused(id, params)`.
- **`PatternMatch`** (`registry.rs:715-725`): `{ bindings: Vec<(usize, NodeId)>, params: FusedOpParams }` — index-keyed bindings become the fused node's input list (sorted by index); `params` is the matcher's authority on the emitted op's per-instance params.

### Recursive tree, NOT flat CSE

Both directions today are a **recursive tree**. `emit` (`runtime_fused.rs:486-540`) recurses per operand and unconditionally `graph.push`es a fresh node — it performs **no interior CSE**. A recipe for `(a+b)*(a+b)` must be spelled with two independent, value-equal `Add` subtrees; emit materializes two distinct `Op::Add` NodeIds. This is documented on the migration-oracle test scaffolding (`runtime_fused.rs:863-873`: "`emit` does NOT CSE-dedup shared subterms"). The identity layer (§6) is what makes this invisible downstream — and moving the *representation* to a flat CSE'd table is Part II §A.

### `FusedOpParams` and identity

`FusedOpParams` (`registry.rs:172-380`) is the per-instance payload of `Op::Fused(id, params)` — one variant per fused op (`SoftmaxLastDim`, `RmsNormLastDim { eps }`, `Rope`, `Conv2D {…}`, `FlashAttn { softmax_scale, causal, …, k_len: Option<DynScalar> }`, `QMatMul { quant_type, k, n }`, `SelectiveScan { delta_softplus }`, …). The last variant `Runtime { scalars: Vec<f64> }` (`registry.rs:373-379`) is the runtime-registered (JIT-synthesized) op: identity is the runtime `FusedOpId`; recipe is the region in the `runtime_fused` sidecar; this payload carries only the extracted `extract:` scalar slots in slot order.

`FusedOpParamsKey` (`registry.rs:416-421`): `{ tag: u16, bits: Vec<u64>, ints: Vec<i64> }` — the hashable CSE key. `FusedOpParams::key()` (`registry.rs:428-623`) assigns a distinct `tag` per variant (1..22) and encodes payload as bit patterns (floats via `to_bits`) / ints. Runtime ops share `tag: 0xF000` (`registry.rs:617-621`).

`Op::Fused(FusedOpId, FusedOpParams)` is the single closed-enum arm delegating to the open registry (`lib.rs:1021`). The 24 well-known ids are `FusedOps::*` constants (`registry.rs:881-1018`), assembled into the process-wide `default_registry()` (`registry.rs:1027-1057`, 24 `with_entry` calls). These **24 `FusedOpId` constants** map onto **22 decompose *submodules*** (Part II §A) — a few ids (e.g. backward/attention variants) share one submodule — so counts of "24 ids" here and "22 decomposes" in Part II §A refer to the two different things. `FusedOpId::RUNTIME_FUSED_BASE = 0x8000` (`registry.rs:79`) partitions static (`1..0x8000`) from runtime (`0x8000..`) id space; `is_runtime()` (`registry.rs:83-85`) is the routing bit.

## 6. Canonical serialization + identity

### `OpAttrs::to_canonical_bytes` — the §6.19 positional blob

`fuel-kernel-seam-types/src/lib.rs:179-246`: `pub fn to_canonical_bytes(&self, op: OpTag) -> Vec<u8>`.

**Outer framing** (`lib.rs:243-245`): `out = u32_le(body.len() /* BYTES */) ++ body`. The body is a per-op **positional** little-endian blob (no field names, no elision — the `OpTag` fixes the schema).

**Empty-schema ops** (elementwise, comparison, `Where`, scalar reductions, log-softmax, and any tag added later) hit the `_ => {}` arm → empty body → the single canonical form `[0,0,0,0]` (tested `lib.rs` for `Add`). `MatMul` **no longer falls through** — since T9 it has a named arm (Part II §C); with empty role vectors it still produces `[0,0,0,0]` (`matmul_empty_roles_stay_the_canonical_zero_body`).

**`put_*` byte writers** (`lib.rs:146-153`), all little-endian:

- `put_u32` = 4 LE bytes; `put_u64`/`put_i64`/`put_f64` = 8 LE bytes.
- `put_str` = `u32_le(s.len())` ++ raw UTF-8 (`lib.rs:150`).
- `put_i64_list`/`put_u32_list`/`put_f64_list` = `u32_le(xs.len())` ++ elements — **the list prefix is the ELEMENT COUNT, not a byte length** (`lib.rs:151-153`), in contrast to the outer frame's byte length.

A `put_u8_list` helper (`lib.rs:226`, `u32_le(count) ++ u8s`) was added in Increment C slice 1 T9 for the pinned matmul role-vectors (Part II §C) — the pre-slice-1 blob had no u8-list writer.

Per-op positional arms (`lib.rs:182-242`):

| Tag(s) | Body layout |
|---|---|
| `Reshape`/`BroadcastTo`/`ReduceSumTo`/`ReduceMaxTo`/`Iota` | `put_i64_list(target_shape)` |
| `Permute`/`Transpose` | `put_u32_list(perm as u32)` |
| `Unsqueeze`/`Squeeze` | `put_u32_list(dims as u32)` |
| `Slice` | `u32(axis) ++ u64(start) ++ u64(len)` |
| `Concat`/`Flip`/`Triu`/`Tril`/`IndexSelect`/`Gather`/`IndexAdd`/`ScatterAdd` | `i64(axis)` |
| `Roll` | `i64(axis) ++ i64(shift)` |
| `SumDim`/`MaxDim`/`MeanDim`/`CumSum` | `i64(axis) ++ u8(keepdim)` (`MaxDim` additive, Increment C slice 1 T4 — monoid rides `op_name`) |
| `Cast` | `put_str(cast_dtype)` |
| `Pad` | `u32(count) ++ (u64 before, u64 after)*count ++ u8(mode) ++ f64(value)` |
| `AddScalar`/`MulScalar`/`Clamp`/`PowI` | `put_f64_list(scalars)` |
| `MaskedFill` | `put_f64_list(scalars) ++ put_str(cast_dtype)` |
| `Const` *(leaf)* | `u64(const_bits)` — dtype-agnostic bits; **MBZ narrow-dtype rule** (storage bits LOW-order, upper bits zero); NaN payload verbatim |
| `RuntimeScalar` *(leaf)* | `u32(slot_index)` |
| `ReducedCount` *(leaf)* | `i64(axis)` — the fold row's axis field minus `keepdim` (fold-lockstep, §6.12-0001) |
| `ScanPlaceholder` *(leaf)* | `u8(role: 0=carry, 1=elem) ++ u32(index)` |

The last four rows are the **source-op leaf arms acked by the KISS editor 2026-07-23**
("RULING RECORD — four-leaf-arm ack"; clean, no amendments). They are wire tokens only:
`op_to_tag` emits none of them and `tag_to_op` declines all four as honest misses (see §4),
so they never reach a live Fuel recipe yet. Producers widen a narrow const via
`const_bits_narrow(storage, width_bits)`. All four bodies ride carrier (a) (`u32`-LE outer),
pinned by `leaf_arm_bodies_ride_carrier_a_u32_le`.

**M-3 caveat** (`lib.rs:174-178`): the `unwrap_or(...)` defaults cannot distinguish an *unset* field from a genuine zero (`axis: None` vs `Some(0)`). Harmless today — forward-serialization only, no decoder, and an op reaching a given arm always has the field set. A future decoder must not round-trip `None`.

**Conformance scope** (`lib.rs:129-144`, `:163-172`): byte-comparable with a Baracuda-emitted blob only for the positionally-conformant ops. Two known divergences are reconciled by the pinned node schema `Op{op_name, op_attrs, child_edges}` WITHOUT widening this blob: `reduce{monoid, reduce_axes, keepdim}` (Fuel emits single-axis `{axis,keepdim}`; `monoid` rides `op_name`; multi-axis DEFERRED) and `gather/scatter{axis, oob_policy, …}` (Fuel emits `{axis}`; `oob_policy` a DEFERRED unwired slot).

### `base_map_hash` — recipe IDENTITY

`fuel-graph/src/opt.rs:399` (recursion `go` at `opt.rs:457-483`): `pub fn base_map_hash(graph: &Graph, root: NodeId) -> u64`. A **NodeId-independent content hash** of the subgraph rooted at `root`. Each node is hashed as `(op identity, child hashes)` — folding each child's *hash* (not its `NodeId`, `opt.rs:474-475`), so two independently-built graphs (different arenas, different numbering) that are structurally identical hash equal.

- Op identity comes from `op_key(&n.op)` when available; otherwise the fallback is `(discriminant, shape.dims(), dtype)` (`opt.rs:463-473`).
- **Commutative-operand sorting** (`opt.rs:476-478`): if `is_commutative(&n.op)` (`Add`/`Mul`/`Maximum`/`Minimum`, `opt.rs:1142-1144`) the child hashes are sorted, so `a+b` and `b+a` hash equal.
- **Const bytes are folded** (`opt.rs:425-455`, `:469-471`): `op_key` returns `None` for `Op::Const`, so the fallback folds the const's *real bytes* (floats via `to_bits()`) when readable. Unpopulated/device-only/locked slots are a silent no-op → same-shape/dtype consts collide; the numeric verify pass is the source of truth there.

A crucial consequence for Part II §A: because children fold **by content hash**, the two duplicated `Add` nodes from an un-CSE'd tree each hash to the same value, and their parent's `[h, h]` is identical to what a single shared node would produce. **`base_map_hash` is invariant to whether the interior is shared or duplicated** — the representation's lack of CSE is invisible at the identity layer.

**Honest scope** (`opt.rs:392-398`): this canonicalizes decomposition depth + commutative reordering, but **NOT associativity or distributivity** — `(a+b)+c` vs `a+(b+c)` hash differently; a numeric verify pass covers the residual (the co-design's Q6 answer). Hashes are process-local (`DefaultHasher`) — never persist or cross-process compare.

### `lower_to_base_map`

`fuel-graph/src/opt.rs:364-366`: `pub fn lower_to_base_map(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId>` — a thin wrapper over `RuleRegistry::lowering_only().optimize_to_fixpoint`. It lowers every reachable fused op to its primitive base map (the fixpoint of `decompose`). A self-returning `decompose` is a clean fixpoint (the never-panic total-decompose contract), not a loop. Recipe identity = `base_map_hash` computed *after* `lower_to_base_map`.

Note (relevant to Part II §A): `lowering_only()` (`opt.rs:218-231`) registers only `LoweringRule`s; `optimize_to_fixpoint` runs Lowering → Fusion → Algebraic families (`opt.rs:255-272`). The real node-merging **CSE pass, `optimize` (`opt.rs:1155-1248`), is a standalone function, not a `Rule` in any family** — so `lower_to_base_map` does *not* run it, and the lowered base map still contains the duplicated interior nodes. The identity is content-hash-CSE'd; the representation is not.

### `OpKey` and `op_key`

`opt.rs:940-951`:

```rust
#[derive(Debug, Hash, PartialEq, Eq)]
struct OpKey { tag: u16, ints: Vec<i64>, bits: Vec<u64>, dims: Vec<usize>, shape: Option<Vec<usize>>, dtype: Option<u32> }
```

`op_key` (`opt.rs:953-1134`) is a HashMap-friendly encoding of `Op` (needed because `Op` carries `f64` and const data, neither `Hash+Eq`). Scalar payloads encode as bit patterns; **`Op::Const` is deliberately excluded from CSE** (`opt.rs:954-956`, returns `None`). Representative arms:

| Op | Encoding | Line |
|---|---|---|
| `Slice { dim, start, len }` | tag `81`, `ints=[dim, start, len]` | `:1076-1083` |
| `AddScalar(c)` | tag `90`, `bits=[c.to_bits()]` | `:1085` |
| `MatMul` | tag `30`, all slots empty | `:1042` |
| `Op::Fused(fid, fparams)` | tag `200`, `ints=[fid.0, params.tag, ...params.ints]`, `bits=params.bits` | `:1099-1106` |
| `Op::Scan { n_xs, bound, emit, early_exit }` | tag `210`, `ints=[n_xs, bound, emit_tag, exit_flag]` (body hashed via last two `inputs`) | `:1114-1118` |
| `Op::ScanPlaceholder { role, index }` | tag `211`, `ints=[role_tag, index]` | `:1121-1124` |
| unlisted (indexing, in-place, …) | `_ => return None` → shape/dtype fallback | `:1131` |

### Runtime dedup by hash

`register_runtime_fused` (`runtime_fused.rs:147-198`) emits a runtime region onto placeholder leaves, lowers (`lower_to_base_map`) and hashes it (`region_base_map_hash`, `runtime_fused.rs:117-130`); the hash indexes the **live** `hash_index()` (`runtime_fused.rs:76-79`, `RwLock<HashMap<u64, FusedOpId>>`). A structurally-identical region resolves to the EXISTING `FusedOpId` rather than minting a duplicate (tested `runtime_fused.rs:608-613`); dedup path at `runtime_fused.rs:176-180`. Hashing runs inside `catch_unwind`; any failure is "hash unavailable" and skips dedup, never blocking registration (`runtime_fused.rs:159-168`).

A **dormant** sibling index exists on the static catalog: `FusedOpRegistry::by_pattern_hash` (`registry.rs:750`) is `#[allow(dead_code)]` (`registry.rs:749`), "reserved for step 4's declarative pattern engine" (`registry.rs:763-764`), and its `PatternHash` hashing fn is "filled in alongside `PatternTree`" which does not yet exist (`registry.rs:852-857`). The co-design named making this dormant index live as the migration target (Part II §A).

### Honest identity scope

`base_map_hash` is a **structural pre-filter**, not the verifier. Two recipes that hash equal are structurally identical up to decomposition depth + commutativity; the numeric-at-tolerance verify pass is the actual gate for associativity/distributivity/const-collision residual. Recipe identity in one sentence: *emit region → `lower_to_base_map` → `base_map_hash`, and equality of that hash is `base_map_hash` equality.*

## 7. The interface envelope: the FKC contract

Because a recipe DAG node stores no dtype/shape/cost/precision, those live on the **FKC kernel contract** that *wraps* the fused op — serde mirrors in `fuel-dispatch/src/fkc/schema.rs`. The rule fields are deliberately opaque `Option<String>` expressions parsed by FKC's own mini-parser later, not by YAML (`schema.rs:11-15`, `:217-218`).

### `OutputDesc` — the return contract

`schema.rs:220-236`:

```rust
pub struct OutputDesc {
    #[serde(default)] pub name: Option<String>,          // :223
    #[serde(default)] pub dtype_rule: Option<String>,    // :226  e.g. passthrough(lhs) / fixed(F32)
    #[serde(default)] pub shape_rule: Option<String>,    // :229  e.g. same_as(lhs) / from_params(batch, m, n)
    #[serde(default)] pub layout_guarantee: Option<String>,  // :232  contiguous / preallocated
    #[serde(default)] pub aliasing: Option<String>,          // :235  none
}
```

`OutputDesc.shape_rule` is a **Fuel FKC field, not a KISS §5 field** (Part II §B has the full story). It is checked live against the real registered `shape_rule` fn.

### `TensorDesc` — the `accept:` operand descriptor

`schema.rs:243-283`: `dtypes: Vec<String>` (Fuel `DType` names, `:257-260`), `dtype_class` shorthand (`int|uint|float|any`), `layout: LayoutSpec` (5-flag capability set, `schema.rs:289-321`), `rank`, `shape_constraint`, `fdx: FdxSpec` (quant/sub-byte/symbolic-extent requirements), `optional` (last-input-only optional operand).

### caps / cost / precision / determinism

`FkcKernel` (`schema.rs:144-156`): `CapsBlock` (`schema.rs:391-408`, layout strategy, fast-paths, in_place, alignment), `CostBlock` (`schema.rs:412-447`, `flops`/`bytes_moved` symbolic expressions + a `cost_fn` name pinned through the provider `LinkRegistry`), `PrecisionBlock` (`schema.rs:466-486`, `max_ulp`/`max_relative`/`max_absolute`/`audited` → `PrecisionGuarantee`), and `determinism` (`bitwise`/`same_hardware_bitwise`/`nondeterministic`).

### The DType vocabulary

`fuel-ir/src/dtype.rs:14-47`, 15 variants:

`U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0`

Stable names via `as_str()`/`FromStr` (`dtype.rs:61-105`) — `f8e4m3`, `f6e2m3`, `f6e3m2`, `f4`, `f8e8m0`, etc. This is the string carrier `OpAttrs::cast_dtype` uses (dep-free across the crate boundary). Sub-byte types (`F6E2M3`, `F6E3M2`, `F4`) report `size_in_bytes() == 0` (`dtype.rs:110-128`).

### The orthogonal physical SType / Encoding

`fuel-ir/src/stype.rs`. `DType` is the LOGICAL element type ("what is a value"); `SType` is orthogonal — HOW those logical elements are physically encoded (`stype.rs:1-11`).

- `SType(pub SmallVec<[Encoding; 1]>)` (`stype.rs:82`) — an ordered stack of encoding layers; **empty = plain** (dense `DType`, byte-identical to pre-SType, `stype.rs:76-80`, `:86-88`). `Eq + Hash` so it can feed structure keys / plan caches.
- `Encoding` (`stype.rs:37-64`): `GgmlBlock { ggml_dtype }` (inline scale), `AffineBlock { packed, block_shape, scale: ScaleSpec, zero_point }` (NF4/QLoRA — separate per-block scale operand, model B), and reserved `Mx`.
- `ScaleSpec` (`stype.rs:24-32`) is a *requirement* for a sibling scale operand, not a pointer; the quant scale is a sibling graph operand, not embedded on the node.

So: the recipe DAG answers *structure + identity* (op tags, operands, canonical bytes, base-map hash); the FKC contract answers *dtypes accepted, shape/dtype rules, layout guarantees, cost, precision, determinism*. The two are joined at runtime by `FusedOpId`.

---

# Part II — Realization & in-flight migrations (Convergence Increment C)

**Increment A shipped (2026-07-16).** It grew `emit` to full first-order parity via the shared `primitive_shape`, grew `OpAttrs` + `to_canonical_bytes` (the §6.19 blob), and landed the flat-DAG-schema reply. **Convergence-C shipped the shape-oracle (2026-07-22, merged @ `9156e178`)** — the shape-expression evaluator and its independent §6.20 wire codec, so **§B below is SHIPPED**. **Increment C slice 1 shipped the recipe-interior FOUNDATIONS (2026-07-23, branch `feat/increment-c-slice1`, T1–T9 `fbe96f0d`..`12c102cf`)**:

- the shape-expression vocabulary moved to its permanent dependency-free home `fuel-kernel-seam-types` (T1; `fkc/shape_expr.rs` is now a `pub use` shim — §B's `shape_expr.rs:*` line anchors are byte-identical, only the crate directory changed);
- shape-**relative** `OpAttrs` interior fields + a pure `resolve_rel_attrs` resolver + a children-first resolving `emit` (T2/T3, D2/D3/D4);
- the additive `OpTag::MaxDim` (T4);
- a `decompose_via_recipe` bridge (T5) and **5 of the ~16 migratable registry `decompose` fns migrated to portable `PatternNode` data** — `softmax_last_dim`, `rope`, `rms_norm_last_dim`, `layer_norm_last_dim`, `softmax_last_dim_backward` (T5–T8), each shape- AND rank-polymorphic;
- **the locked matmul role-vector `op_attrs` serialize/resolve, live in both directions (T9), so §C below is SHIPPED**.

What **remains NOT built** in §A is the flat-DAG-CSE indexed node/table WIRE serializer + the maximal-CSE representation (KISS #67-gated, slices 3–4) and the remaining ~11 first-order migrations. So read §A's flat-table WIRE as *target*, its decompose→`PatternNode`-data + shape-relative-attr coupling as *shipped for 5 ops* (§A "Shipped in slice 1"), §B as *as-built*, and §C as *as-built*. The as-built base is Part I.

## A. Flat-DAG-CSE migration

### Shipped in slice 1 (2026-07-23) — the recipe-interior foundations

Increment C slice 1 built the machinery §A needs and migrated the first-order tranche. It did **not** build the flat indexed node/table WIRE serializer or the maximal-CSE representation (below, still target — KISS #67-gated). As-built:

**5 of the ~16 migratable `decompose` fns are now `PatternNode` data.** Each submodule's imperative `&mut Graph` body is replaced by a `OnceLock`-built static `recipe() -> &'static PatternNode`, a per-entry `scalars(&FusedOpParams) -> Option<Vec<f64>>` projection, and a one-line `decompose = decompose_via_recipe(g, id, recipe(), scalars(params))` — the D6 mechanism; the `FusedOpEntry.decompose` fn *signature* (`registry.rs:112`) is untouched:

| Fused op | Recipe (op composition) | Anchors (`recipe`/`decompose`) |
|---|---|---|
| `softmax_last_dim` | `Div(Exp(Sub(x, Bcast(Unsqueeze(MaxDim x)))), Bcast(Unsqueeze(SumDim e)))` — 9 op nodes | `registry/softmax_last_dim.rs:82` / `:130` |
| `rope` | `Add(Mul(x, Bcast cos), Mul(Concat(Neg(Slice₂), Slice₁), Bcast sin))` — 9 recipe nodes, D4 pad → **11-node byte-identical emission** | `registry/rope.rs:102` / `:180` |
| `rms_norm_last_dim` | `Div(x, Bcast(Sqrt(AddScalar[eps](Unsqueeze(MeanDim(Sqr x))))))` — 7 nodes, `eps` open slot | `registry/rms_norm_last_dim.rs:91` / `:143` |
| `layer_norm_last_dim` | `centered = Sub(x, Bcast(Unsqueeze(MeanDim x)))`; `Div(centered, Bcast(Sqrt(AddScalar[eps](Unsqueeze(MeanDim(Sqr centered))))))` — 11 nodes, x-only input | `registry/layer_norm_last_dim.rs:103` / `:163` |
| `softmax_last_dim_backward` | `Mul(s, Sub(g, Bcast(Unsqueeze(SumDim(Mul(g, s))))))` — the D3 swap of the legacy `ReduceSumTo` form; via the `BackwardKind::Fused` autograd path | `registry/softmax_last_dim_backward.rs:113` / `:159` |

`decompose_via_recipe` (`registry.rs:900`) reads the fused node's inputs as binds, projects scalars, and calls the resolving `emit`; **ANY failure** (wrong-params payload, resolution decline, slot mismatch) returns `id` — the fixpoint, surfaced-gap, never-panic posture the imperative bodies carried (G2). The 5 recipes are **shape- AND rank-polymorphic**: the same static data lowers `softmax_last_dim` correctly at both `[2,4]` and `[3,5,7]` — the thing a `PatternNode` that baked absolute shapes could not do (proven in the polymorphic-decompose tests, `runtime_fused.rs`/`registry/*`).

**Shape-relative interior attrs (D2/D3/D4).** Four optional `OpAttrs` fields carry the shape-relative recipe interior, all `Default`-empty (zero behavior change for existing regions): `target_shape_rel: Option<ShapeExpr>` (`SameAs` over the Bind space), `slice_start_rel`/`slice_len_rel: Option<Dim>` (`DimExpr` over bind shapes), `axis_last: bool` (this op's axis-carrier = its per-tag LAST) — `fuel-kernel-seam-types/src/lib.rs:151-163`. They resolve to concrete `OpAttrs` at emit time via `resolve_rel_attrs(attrs, bind_shapes, child_shapes) -> Result<OpAttrs, RelAttrError>` (`runtime_fused.rs:511`), which **reuses `shape_expr::eval_dim`/`resolve_axis`** (no second evaluator) and returns a typed `RelAttrError` (`runtime_fused.rs:427`) — bind-out-of-range, rel+abs both set, `axis_last` on an axis-less tag, symbolic-bind gap — **never a panic**. `emit` reorders children-first, then resolves, then runs the unchanged `tag_to_op` → `primitive_shape` path; `validate_representable` gained a rel-attr probe (rel XOR abs per field + bind-range checks).

**The rel fields are deliberately NOT serialized this slice (D2).** `to_canonical_bytes` serializes only the concrete fields; the pin is executable in `rel_attr_fields_are_absent_from_the_6_19_wire` (`lib.rs:571`) — the §6.19 `broadcast_to`/`slice` arms stay ABSOLUTE and the node-envelope framing that would carry a relative alternative is KISS #67-gated (propose-first).

**The D3 keepdim swap adds `OpTag::MaxDim`** (`lib.rs:55`, additive per `#[non_exhaustive]`) so `ReduceMaxTo(keepdim)+Bcast` becomes the shape-polymorphic `MaxDim+Unsqueeze+Bcast`; it serializes the pinned reduce row (`i64(axis) ++ u8(keepdim)`, byte-identical to `SumDim` — the monoid rides `op_name`; `max_dim_serializes_the_reduce_row`, `lib.rs:615`). **The migrated ops' base maps change** → `base_map_hash` changes (process-local only, no persisted/cross-process blast radius, §6); the `emit_matches_*` parity oracles were held against frozen legacy builders with **bit-exact** numeric parity as the gate.

### Representation vs identity — the boundary the migration turns on

| | REPRESENTATION (the recipe) | IDENTITY (the lowered base map) |
|---|---|---|
| object | `PatternNode` tree | primitive `Op` graph, content-hashed |
| interior sharing | none (Bind leaves only) | CSE-*invariant* hash |
| home | `fuel-kernel-seam-types` | `fuel-graph/src/opt.rs` |

The current representation is a **recursive inline tree** (`lib.rs:254-273`): `operands: Vec<PatternNode>` holds children by value, and the only cross-referenceable node kind is `Bind` (an external *input*, and even then a value-equal duplicate leaf, not a shared reference). `emit` (`runtime_fused.rs:486-540`) performs **no interior CSE** — it recurses per operand and unconditionally `graph.push`es (proof: the migration-oracle comment `runtime_fused.rs:863-873`, and the softmax parity test that builds its shared `Exp` subterm fresh twice at `runtime_fused.rs:927`/`:929` precisely because emit duplicates it). The identity layer hides this: `base_map_hash` folds by content hash, so a duplicated and a shared interior hash identically (Part I §6).

### The KISS §6.4-0009 target and the emitter contract

Pinned across `docs/outreach/baracuda-recipe-grammar-codesign-reply{,-2,-3}.md`: a **flat indexed node table with two closed kinds** —

```
Op{ op_name, op_attrs, child_edges }   |   Bind(input_index)
```

- `PatternNode` restricted to `Op | Bind` **IS** the §6.4-0009 schema (`codesign-reply-2.md:50`); `Any`/`SeeThrough` are matcher-only wildcards with no place in a concrete recipe.
- `child_edges` are **indices into the flat table**, not inline subtrees — this is the representational change. Reductions/scans/matmul are ordinary nodes: a fold node's `child_edges` reference its pre-map inputs; epilogue nodes reference the fold node; `Reduced(i)` is an **edge, not a leaf** (`codesign-reply-2.md:8, :16`). For matmul the fold node *is* the matmul node (`codesign-reply-3.md:12, :57`).
- **Maximal CSE** is first-class: the flat table shares + canonicalizes computed intermediates (reused `(x-mean)`, squared residuals — `codesign-reply.md:13`).
- Source/leaf ops stay within the two kinds by adding **op tokens**, not schema variants: `const→Op{const,{bits},[]}`, `coord→Op{iota,{axis},[]}`, dispatch-bound scalar `→Op{runtime_scalar,{slot_index},[]}` (`codesign-reply-2.md:10-16`; note `count_scalar_slots` at `runtime_fused.rs:394-402`).
- **Scan** serializes flat with no nesting: `child_edges = [init_carry, xs.., consts.., body_new_carry, body_y]`, body holes = `Op{scan_placeholder,{role,index},[]}`, attrs `{n_xs,bound,emit,has_early_exit}` — already matching Fuel's `op_key` encoding (`opt.rs:1108-1124`).

**The emitter contract** (`codesign-reply-2.md:8`, `codesign-reply.md:20`): Baracuda emits a **valid-but-not-necessarily-canonical** DAG; **Fuel canonicalizes on ingest** (lower → maximal-CSE → `base_map_hash`). Verification = resolve-to-base-map (structural pre-filter) + numeric-at-tolerance (the gate). For a *known* named op Baracuda emits only the token and Fuel owns the canonical resolution. Canonical `op_attrs` = the §6.19 positional blob already shipped as `OpAttrs::to_canonical_bytes`. Cap bit `SEAM_CAP_RECIPE_IMPORT = FEAT bit 35` (a co-design doc anchor, not yet in code).

### The decompose → PatternNode-data migration

The registry declares **22** fused-op submodules (`registry.rs:34-55`), each exporting exactly one imperative decompose (`fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId`, `registry.rs:112`). Count reconciliation:

- **22 total** submodule decompose fns.
- **6 excluded from first-order migration**: 4 basis-gap self-returns — `conv2d` (`registry/conv2d.rs:127-129`, `return id`), `conv_transpose_2d` (`registry/conv_transpose_2d.rs:111`), `qmatmul` (`registry/qmatmul.rs:100`), `inplace_affine` (`registry/inplace_affine.rs:67`) (need `Im2Col`/`Col2Im`/GGUF-unpack/`AffineInplace` IR primitives); + 2 higher-order scans — `selective_scan` (`registry/selective_scan.rs:232`), `ssd_chunk_scan` (`registry/ssd_chunk_scan.rs:213`), which decompose onto `Op::Scan` and are outside the first-order `emit`/`primitive_shape` path.
- **~16 migratable** = 22 − 4 − 2 (softmax, layer_norm, rms_norm, rope, fused_linear, the backward helpers, etc.).

These were imperative `&mut Graph` builders. `emit` already produces byte-for-byte-identical output against them — proven by `emit_matches_rope_decompose`, `emit_matches_softmax_last_dim_decompose`, `emit_matches_layer_norm_last_dim_decompose`, each asserting structural equality against the hand-written oracle. The migration replaces each builder with a static `PatternNode` region + `emit`. **As of slice 1, 5 of the ~16 are migrated** (`softmax_last_dim`, `rope`, `rms_norm_last_dim`, `layer_norm_last_dim`, `softmax_last_dim_backward` — see "Shipped in slice 1" above); the remaining ~11 (`fused_linear` and the backward helpers with shape-derived scalar slots / param-conditional structure) stay imperative, which is legal — the `decompose` fn signature is unchanged — until their carriers land (slice 2+).

### The absolute-shape-baking coupling (why C is two changes, not one)

A `PatternNode` recipe bakes **absolute** shapes: `OpAttrs.target_shape: Vec<i64>` (`lib.rs:86`) and `slice_start`/`slice_len: Option<u64>` (`lib.rs:98-101`). You can see it in the emit-parity test — the rope region carries `target_shape: vec![2,4]`, `slice_start: Some(2)`, `slice_len: Some(2)` (`runtime_fused.rs:948-950`), correct for exactly a `[2,4]` input. Contrast the hand-written `registry/rope.rs` decompose (`registry/rope.rs:83-178`), which is shape-**polymorphic** — it reads `x_shape` off the live node (`registry/rope.rs:84-88`), computes `half = d/2` (`registry/rope.rs:95`), and derives broadcast targets from the live dims, then issues **11** `graph.push` calls. (`softmax_last_dim::decompose` at `registry/softmax_last_dim.rs:78-134` reads `x_shape`, computes `keepdim_shape`, and pushes **7** nodes.)

So a `PatternNode` recipe as-is is "correct for exactly ONE input shape, which defeats 'recipe = portable data'" (`docs/outreach/baracuda-shape-expression-grammar-ask.md:10`). **Increment C is two coupled changes**: `decompose → PatternNode-data` AND `absolute-shape attrs → shape-relative attrs` (the `ShapeExpr`/`DimExpr` grammar of §B). You cannot do the first portably without the second.

### Risks / the specific code that changes

1. **`base_map_hash` never sees a `PatternNode`.** It walks the *lowered graph's* `n.inputs` (`opt.rs:457-483`); the flat body is a **representation** change. Because the hash is already duplicate-invariant, a flat body with maximal CSE and the current inline tree **hash identically** — so `hash_index` dedup and every emit-parity test remain valid across the migration. Representationally invasive, semantically safe.
2. **CSE-vs-identity must not be conflated.** `lower_to_base_map` does *not* run `optimize` (the real CSE merge, `opt.rs:1155`); the lowered base map retains duplicates. If the flat schema is expected to be *materially* CSE'd (shared NodeIds, not just an equal hash), that requires either running `optimize` inside ingest, or building the flat table with shared edges *before* lowering. The dormant `by_pattern_hash` + `PatternHash` (`registry.rs:750`, `:852-857`) is where structural CSE would wire into the static catalog. A maintainer must decide which the flat container guarantees.
3. **Shape-polymorphism is the hard constraint on the flat body's `op_attrs`.** `target_shape`/`slice_start` (`lib.rs:86,98`) stay absolute even after flattening; §B's `ShapeExpr`/`DimExpr` must land in the same change, as a new recursive `op_attrs` sub-vocabulary serialized as a nested §6.19 blob. Code that changes: `OpAttrs` (`lib.rs:70-120`); the `to_canonical_bytes` shape-target/slice arms (`lib.rs:184-202`); `tag_to_op`'s shape reconstruction (`runtime_fused.rs:321-379`); the FKC evaluator `eval_shape_rule` (extend for `Extent`/`DimExpr` + a real `from_params`).
4. **`emit` must gain (or defer) node-sharing to consume a flat table.** `emit` (`runtime_fused.rs:486-540`) is tree-shaped; re-emitting a flat table with shared edges needs a `Vec<NodeId>` memo keyed by table index. The emit-parity oracle tests (`runtime_fused.rs:863-1012`) would then expect shared rather than duplicated subterms — unless the flat table is flattened-then-emitted-as-tree (a maintainer choice tied to risk 2).
5. **`Op::Scan` and the 4 basis-gap ops stay outside first-order emit.** They remain surfaced opaque-op gaps until their IR primitives land; not migratable to `PatternNode` data in C.

## B. The shape-expression evaluator

### Shipped state (Convergence-C, merged @ `9156e178`)

`eval_shape_rule` (`return_check.rs:32`) now evaluates the full `SameAs` + `DimExpr` vocabulary **and** the matmul contraction rule, backed by an **independent typed AST + §6.20 wire codec** (`fkc/shape_expr.rs`) and a **parser** (`fkc/shape_expr_parse.rs` — the `parse_shape_rule` analogue the pre-Convergence-C code lacked):

- `same_as(role)` → the role's probe shape.
- a `DimExpr` rule → `parse_dim` → `shape_expr::eval_dim` (`return_check.rs:64-68`): an `Extent`/`Const`/`Param`/`+ − × ÷floor` tree evaluated against the probe operand shapes + params, requiring a non-negative result.
- a `matmul(...)` rule → `parse_matmul_operands` → `shape_expr::matmul_shape` (`return_check.rs:43-58`): role-derives `[..batch, M, N]`, the contraction output that equals **neither** operand.
- an unparseable / unknown rule still returns `Ok(None)` — the "not-evaluable, skip, never a false reject" contract holds.

The evaluator is the shape-side companion to the §6.4-0006 value oracle; `shape_expr::shape_consistent` (`shape_expr.rs:297`) is the §6.4-0011 tie — the Interface `declared` shape is consistent iff it equals the op's `computed` shape rule, and a surfaced `Gap` (symbolic output) is never a hard mismatch. Every malformed input is a **typed decline** (`ShapeExprError`, `shape_expr.rs:89-101`), never a panic.

This **closes the "shape_rule parsed-but-never-evaluated" gap** the pre-Convergence-C code carried: the vocabulary was one production wide (`same_as` only), there was no `parse_shape_rule`, and a non-`same_as` claim was skipped unchecked — the `eval_dtype_rule` twin already had two productions (`fixed`/`passthrough`) while the shape side had one. Convergence-C resolved that asymmetry. (The recipe-interior mirror of the gap — baked absolute `OpAttrs.target_shape`, §A — is **not** closed; that is the still-pending home #2 below.)

### The merged `ShapeExpr` / `DimExpr` vocabulary

Co-design converged on one closed grammar (`docs/outreach/baracuda-shape-expression-grammar-ask.md:19-28`, axis reconciled at `-reply.md:6-13`):

```
ShapeExpr := SameAs(operand)                       // an operand's whole shape (every BroadcastTo target)
           | [reserved: Reduce(operand, axis, keepdim), WithDim(operand, axis, DimExpr)]

DimExpr   := Extent(operand, axis)                 // size of an operand's axis (rope's `d`)
           | Const(i64) | Param(field)             // Param == OutputDesc from_params field
           | DimExpr (+ | − | × | ÷) DimExpr        // integer; ÷ is FLOOR division

axis      := non-negative index | `last`           // last → rank−1 at eval; signed −1 DROPPED
operand   := local operand position operand[k] | Bind(i)   // Bind(i) == a contract's role
```

Backward-compatible: `same_as(role) ≡ SameAs(operand)`, `from_params(f) ≡ Param(f)` (`baracuda-shape-expression-grammar-ask.md:38`).

**As shipped** (`shape_expr.rs:30-49`): `Dim` is the full `DimExpr` enum (`Extent{operand,axis}` / `Const(i64)` / `Param(u8)` / `Add` / `Sub` / `Mul` / `Div`); `ShapeExpr` currently has the single `SameAs{operand}` variant, with `Reduce`/`WithDim`/`Dims` **tag-reserved** (`shape_expr.rs:17-19`, reader-rejected) for the §6.4 extension registry. The **role/index-woven** rules (`reduce`/`gather`/`matmul`) are **not** wire `ShapeExpr` — they are separate op-semantics functions (`reduce_shape` / `gather_shape` / `matmul_shape`, `shape_expr.rs:253-290`), because a gather/matmul output equals *no* operand's shape (which is exactly the false-`same_as(data)` bug the oracle catches).

Design decisions pinned across all three parties:

- **Two irreducible constructors.** Most ops are already shape-polymorphic via `primitive_shape` and carry no shape attr; only a **broadcast target** (→ `SameAs`) and a **slice/iota offset** (→ `DimExpr`) irreducibly bake shape. Everything else canonicalizes to an already-polymorphic primitive (`ReduceMaxTo → MaxDim{keepdim}`, `Reshape`-to-1s → `Unsqueeze`).
- **`Reduce`/`WithDim` reserved** — in the grammar for completeness, promotable via the umbrella §6.4 extension registry.
- **Axis = non-negative | `last`; `−1` dropped** — one encoding across the whole recipe + shape surface; `last` resolves against operand rank at import.
- **Operand: positional is normative, role is a surface alias** — KISS op_dag interior nodes carry no operand-role tuple.
- **`÷` = floor, no remainder error** — producers relying on exact division (even head dim) own that invariant.
- **One grammar, additive, not a competing shape authority** — a recipe-carrying op keeps **omitting** `shape_rule`; the realized recipe / role-vectors remain the sole shape authority. Giving `shape_rule` an evaluator makes the *claim* checkable; it does not promote the claim to an authority.

### §6.20 serialization — the shipped wire codec

The `ShapeExpr`/`DimExpr` tree serializes as a **recursive, tag-prefixed, `u16`-length-prefixed positional blob** (§6.20-0005), byte-matching the KISS goldens. The shipped codec is `shape_expr.rs` `encode`/`encode_binary` (`:53-87`); the `serialization_golden` test (`:309-324`) asserts every anchor byte-for-byte.

**Tag space** (`shape_expr.rs:8-19`, one byte; `0x00` reserved): `SameAs=0x01`, `Extent=0x02`, `Const=0x03`, `Param=0x04`, `Add=0x05`, `Sub=0x06`, `Mul=0x07`, `Div=0x08`; **reserved (reader rejects):** `Reduce=0x09`, `WithDim=0x0A`, `Dims=0x0B`.

**Leaf / node layouts** (`encode`, `:53-87`):

| Node | bytes |
|---|---|
| `SameAs{operand}` | `[0x01, operand:u8]` |
| `Extent{operand, axis}` | `[0x02, operand:u8, axis:u8]` |
| `Const(i64)` | `[0x03, i64_le]` (9 bytes) |
| `Param(field)` | `[0x04, field:u8]` |
| binary `Add`/`Sub`/`Mul`/`Div` | `[tag, u16_le(len childA), childA, u16_le(len childB), childB]` |

**The child-length prefix is `u16`-LE** (`encode_binary`, `:79-87`) — deliberately **distinct from the `op_attrs` outer frame's `u32`-LE byte length** (Part I §6). Both are §6.19/§6.20-canonical; they are two blobs with two widths — `op_attrs` outer = `u32`-LE byte-len, shape-expr children = `u16`-LE (≤ 65535, ample for bounded `DimExpr` subtrees). A maintainer touching either must not conflate them.

**Axis `u8`:** concrete axes `0..MAX_RANK−1` (`MAX_RANK = 8`), with **`0xFF` = the `last` sentinel** (`shape_expr.rs:24-28`), resolved to `rank−1` at eval (`resolve_axis`, `:180-187`). Explicitly **not** byte-identical to §6.19-0020's `0xFFFE` `u16` axis-set **mask** — the shape-side `Extent` axis is a single `u8`; the value-side reduce set is a bitmask. Same axis *semantics*, different field *width*.

**Symbolic extent:** `SYMBOLIC = i64::MIN` (`shape_expr.rs:22`); an `Extent` over it evaluates to `DimValue::Gap` (`eval_dim`, `:213`), which propagates through every binary op (`eval_binary`, `:195-202`) and through a `SameAs` over a symbolic-bearing shape (`eval_shape`, `:238`) — the surfaced-gap-never-crash posture, in code.

**The rope-half golden** — the byte contract of record (`serialization_golden`, `:316-323`): `Div(Extent(operand=0, axis=last), Const(2))` encodes to

```
08  03 00  02 00 FF  09 00  03 02 00 00 00 00 00 00 00
│   │      │          │      └ childB = Const(2) = [0x03, i64_le(2)]  (9 B)
│   │      │          └ u16_le(9)  = len(childB)
│   │      └ childA = Extent(0,last) = [0x02, 0x00, 0xFF]  (3 B)
│   └ u16_le(3) = len(childA)
└ tag Div = 0x08
```

(17 bytes total.) **Emit division:** a producer emits *functional text* (`slice(const(0), div(extent(in0,last), const(2)))`); Fuel parses it (`shape_expr_parse.rs`) and mints this canonical blob on ingest.

**The `reduce_extent → reduced_count` rename.** The value-side reduced-axes leaf was pinned 2026-07-18 as `reduce_extent`, then **renamed 2026-07-19 to `reduced_count`** to converge onto KISS §6.12-0001's canonical token — 1:1 identical, "align not alias," pre-consumer (recorded `kernel-seam-interop.md:517-548`). The canonical §6.12 pair:

- **`extent(axis)`** — the single-axis **shape-side** value leaf that `DimExpr::Extent(op, axis)` spells (`kernel-seam-interop.md:545-548`).
- **`reduced_count(axes)`** — the **value-side** product of extents over the reduced axes; the Mean divisor. `reduce_extent` is the retired name.

`reduced_count`'s canonical body is the fold node's axis field, **byte-identical minus `keepdim`**: `axis: i64` today (single-axis, matching the `SumDim`/`MeanDim` row), growing to a `reduce_axes: i64` list in lockstep with the fold when multi-axis lands — so a canonicalizer checks `reduced_count.axes == fold.axes` by byte-equality (`kernel-seam-interop.md:526-529`). **Increment-C resolver constraint:** Fuel's `reduced_count` axis resolver MUST reuse the fold's axis-resolution codepath verbatim, or a future `last`/mask resolution change could split a pair Baracuda emitted identical (`kernel-seam-interop.md:535-543`). A multi-axis mask (>1 bit) exceeds the single-axis body and honest-misses **both** the fold and the count together, never one.

### The shape / value boundary

`ShapeExpr`/`DimExpr` carry **SHAPES only**. An extent needed as a runtime **value** — the canonical example is the **Mean divisor** — is *not* a shape attr; it is a source-op **leaf inside the recipe DAG** (`reduced_count(axes)`), consumed by a `div` node next to the fold (`baracuda-shape-expression-grammar-ask.md:40`, `kiss-rfc-shape-rule-expression-vocabulary.md:51-53`). The reasoning: the **recipe/Semantics DAG** answers "what does this op compute" — the divisor is a `div` operand, a first-class node, so `Mean == div(reduce[sum,…](pre), reduced_count(axes))` (`kernel-seam-interop.md:525`). The **FKC contract** (`OutputDesc.shape_rule: from_params`) answers "what is this kernel's I/O interface" — output *shapes/dtypes* as functions of params; asking `from_params` to carry a divisor is a category error. Enforcement is structural: recipe-carrying ops **omit `shape_rule`** while keeping `dtype_rule`.

### Symbolic → surfaced gap, never a crash

An `Extent`/`reduced_count` over a **symbolic** axis (`DynScalar::Sym` — a data-dependent / dynamic-length axis) resolves to a **surfaced opaque-op gap, never a crash** — the total-`decompose` / never-panic invariant. Extent resolution is **Fuel-side** (Fuel holds the concrete extents at the seam caller); Baracuda's `StructureKey` carries size *classes*, so an `Extent` often has no literal on Baracuda's side.

This is the same posture as symbolic `k_len` flash decode, whose reference implementation is visible in `registry/flash_attn.rs`. The `decompose` resolves `k_len` into three cases (`registry/flash_attn.rs:101-122`); the symbolic arm is `registry/flash_attn.rs:147`:

```rust
Some(DynScalar::Sym(_)) => return id,   // return self — the never-crash fixpoint signal
```

`Op::Slice` carries a static `usize` len, and no op materializes a `DynScalar` into a length-mask tensor *inside* a `decompose` (which sees only the static graph + params, never the per-realize `SymEnv`). So `decompose` returns self and the symbolic oracle is emitted one layer up by the optimizer's `decode_flash` arm, which holds the `SymEnv`. The `reduced_count` leaf has the identical basis gap and posture; it closes when a `DynScalar`-materialization basis op lands.

### The three homes

The reframe resolved "same vocabulary, two homes" into **three homes** — two now shipped, one pending:

1. **Fuel FKC return-contract — SHIPPED (Convergence-C).** `eval_shape_rule` (`return_check.rs:32`) evaluates `SameAs` + `DimExpr` + the matmul rule (above); `shape_expr::shape_consistent` (`:297`) is the §6.4-0011 tie. This was "the one home with a live evaluator (`same_as` only)"; it is now the full vocabulary.
2. **Fuel recipe interior — FOUNDATIONS SHIPPED (Increment C slice 1), remainder pending.** The interior can now carry shape-*relative* attrs in the same `SameAs`+`DimExpr` grammar: the optional `OpAttrs` rel fields (`target_shape_rel`/`slice_start_rel`/`slice_len_rel`/`axis_last`, `lib.rs:151-163`) resolved at emit by `resolve_rel_attrs` (`runtime_fused.rs:511`, reusing `shape_expr::eval_dim`/`resolve_axis`), and **5 migrated `decompose`s use them** (§A "Shipped in slice 1"). The rel form is resolved to concrete `OpAttrs` at emit and **not yet serialized** (D2 — the §6.19 arms stay absolute, node-envelope framing is KISS #67-gated). Still pending: the remaining ~11 first-order migrations, the flat-DAG-CSE node/table WIRE that would serialize the rel form, and the `reduced_count` leaf. `PatternNode`'s absolute-shape attrs (`OpAttrs.target_shape`, `lib.rs:86`) remain for un-migrated recipes. There is no baked-shape defect in KISS to repair.
3. **KISS shape ORACLE — SHIPPED Fuel-side as the independent reference.** `shape_expr.rs` is Fuel's independent, byte-matching implementation of the KISS §6.20 oracle — the shape-side companion to the §6.4-0006 value oracle. The interior-consistency + Interface-vs-semantics rules `reduce_shape`/`gather_shape`/`matmul_shape` (`:253-290`) + `shape_consistent` (`:297`) catch e.g. a gather advertising `same_as(data)` (its output equals no operand's shape) or a non-keepdim single-axis reduce over rank-3 declaring `rank=3`. KISS contracts are monomorphized per `structure_key`, so this is interior-consistency + Interface-vs-semantics, not making the return contract polymorphic.

One grammar, one serialization, three attachment points. *(The claim that `OutputDesc.shape_rule` was historically mis-framed as a KISS §5 field is corrected: it is a Fuel FKC field — correction banner at `docs/outreach/baracuda-shape-expression-grammar-ask.md:6`. The KISS-repo occurrence counts and the KISS RFC landing at KISS main `@3bd6d2d` are cross-party state asserted in Fuel-side outreach docs; they cannot be verified from the Fuel tree and should be read as external-party status, not Fuel code.)*

### What shipped, and what remains

**Shipped in Convergence-C (@ `9156e178`):** the parser (`shape_expr_parse.rs`, the `parse_shape_rule` analogue), the typed AST + §6.20 wire codec (`shape_expr.rs`), the extended `eval_shape_rule` (`SameAs` / `DimExpr` / matmul, `return_check.rs:32-68`), the §6.4-0011 `shape_consistent` oracle (`:297`), and the `reduce`/`gather`/`matmul` semantic shape rules (`:253-290`). Floor `÷`, `LAST = 0xFF`, `MAX_RANK = 8`, and symbolic → `Gap` are all in `shape_expr.rs` with golden tests.

**Shipped in Increment C slice 1 (2026-07-23):** the recipe-interior FOUNDATIONS (#2) — the `shape_expr` vocabulary moved to its permanent home `fuel-kernel-seam-types` (T1; `fkc/shape_expr.rs` is a `pub use` shim, line anchors byte-identical), the shape-relative `OpAttrs` rel fields + `resolve_rel_attrs` (T2/T3), `OpTag::MaxDim` (T4), the `decompose_via_recipe` bridge (T5), and **5 of the ~16 migratable `decompose` fns migrated to `PatternNode` data** using `SameAs`/`DimExpr` interior attrs. The worked rope halves land exactly as pinned: `Slice{ start: Const(0), len: Extent(x,last) ÷ 2 }` and `Slice{ start: Extent(x,last) ÷ 2, len: Extent(x,last) − Extent(x,last) ÷ 2 }` (`registry/rope.rs`); softmax broadcast → `BroadcastTo(SameAs(operand[0]))`.

**Still pending (Increment C, coupled with §A):** the remaining ~11 first-order `decompose` migrations; serializing the rel form (the §6.19 arms stay absolute this slice, D2 — the flat-DAG node/table WIRE is KISS #67-gated); and the `reduced_count` leaf's own serialization + its fold-axis-resolver reuse (the lockstep constraint above).

Because the shipped `eval_shape_rule` now checks non-`same_as` claims, a `from_params(batch,m,n)` claim on a matmul-shaped op is checkable against the role-derived `matmul_shape` — so Fuel committed to **signal Baracuda before broadening** the checked surface, letting their audit of `same_as(in0)`-emitting cells land first.

## C. Matmul role-vector serialization

### As-built (SHIPPED, Increment C slice 1 T9, `12c102cf`)

MatMul's `op_attrs` now carries the LOCKED u8 role-vectors and serializes/resolves in both directions. Two `OpAttrs` fields hold the roles: `lhs_roles: Vec<u8>` / `rhs_roles: Vec<u8>` (`lib.rs:187,190`), both `Default`-empty. The design settled the "where do ranks come from" question the pin flagged by **populating the role vectors into `OpAttrs`** (not threading rank into the call site): a pure `matmul_roles(lhs_rank, rhs_rank) -> (Vec<u8>, Vec<u8>)` helper (`lib.rs:235`) derives the canonical cell, and a new `put_u8_list` byte writer (`lib.rs:226`, `u32_le(count) ++ u8s`) supplies the u8-per-role framing `put_u32_list` could not.

**Empty-is-implicit** preserves the rank-polymorphic recipe form. Static recipes keep roles empty → the body stays empty → the single canonical `[00,00,00,00]` (the degenerate `body_len = 0` case of the same outer frame; `matmul_empty_roles_stay_the_canonical_zero_body`, `lib.rs:672`, and the untouched `empty_schema_op_serializes_zero_length` golden). Only concrete/ingested nodes get explicit roles that bake rank.

The three seam sites, as-built:

| Site | As-built behavior | Anchor |
|---|---|---|
| `to_canonical_bytes` MatMul | named arm: roles empty → `[00,00,00,00]`; set → `put_u8_list(lhs) ++ put_u8_list(rhs)` under the outer frame | `lib.rs:344`, golden test `:648` |
| `op_key` MatMul | tag `30`, payload slots empty (unchanged — `attrs_match`/CSE do not consult the role fields this slice) | `opt.rs:1042` |
| `tag_to_op` MatMul | resolver **cell**: empty → `Op::MatMul` (implicit-accept); set → validate the canonical cell, else surfaced honest-miss | `runtime_fused.rs:331-337` |

MatMul is representable in the region grammar (`op_to_tag` at `jit.rs:82`); the reverse resolver now honors the role vectors instead of discarding attrs.

### The canonical MatMul cell — `primitive_shape(MatMul)`

`fuel-graph/src/shape.rs:189-212` is the cell the role vectors describe, per-axis:

| axis position | lhs role | rhs role |
|---|---|---|
| `[0 .. rank−2)` (leading) | **Batch** | **Batch** |
| `rank−2` | **FreeM** (`m = l[-2]`) | **ContractedK** (`k2 = r[-2]`) |
| `rank−1` | **ContractedK** (`k = l[-1]`) | **FreeN** (`n = r[-1]`) |

Invariants: same-rank ≥ 2 operands (`shape.rs:193`), exactly one shared `ContractedK` with `k == k2` (`shape.rs:202`), output `Batch.. ++ [m, n]` carried from **lhs's** batch dims (`shape.rs:208`). The `Tensor::matmul` builder (`fuel-graph/src/lib.rs:3912`) enforces the same at build time: rank ≥ 2 (`lib.rs:3934`), `k == k2` (`lib.rs:3983`), shape/dtype delegated to `primitive_shape` (`lib.rs:3991-3996`).

**Nuance for the resolver cell:** the builder auto-broadcasts a rank-2 operand up to the other's batch prefix (`lib.rs:3941-3959`) and permits **GQA-divisible** batch mismatch — `la == ra || (la > ra && ra > 0 && la % ra == 0)` (`lib.rs:3969-3980`, exact line `:3975`) — not strict positional extent equality. The pinned "Batch dims aligned positionally" phrasing is about **role positions**, not extents: role vectors encode *which axis plays Batch*, not its extent, so GQA (differing-but-divisible batch extents) still serializes to identical all-`Batch` leading roles. The resolver cell must check role **positions**, not Batch-axis extent equality.

### The pinned schema (u8 role-vectors)

Source: `docs/outreach/baracuda-recipe-grammar-codesign-reply-3.md` §2 (`:18-47`) / §6 (`:76-91`). **Status: item closed, mutual** (`:91`).

- **Role enum codes, 1 byte each:** `Batch=0, FreeM=1, FreeN=2, ContractedK=3` (`reply-3:37`). Two per-axis role vectors, `lhs_roles` then `rhs_roles`, each of length = operand rank. **Per-role width = u8 is LOCKED** and Baracuda-confirmed (`reply-3:81`).
- **Two framing levels** (`reply-3:82-87`):
  - **INNER (each vector):** `u32_le(element_count) ++ role_bytes` — the count-prefix matches Fuel's `put_*_list` convention (`lib.rs:151-153`), roles narrowed to u8.
  - **OUTER (whole blob):** `out = u32_le(body_len_in_BYTES) ++ body` — exactly the `lib.rs:243-245` framing.

```
op_attrs(matmul) = u32_le(len lhs_roles) ++ lhs_roles ++ u32_le(len rhs_roles) ++ rhs_roles
                   └───────────────────────── body ──────────────────────────────────────┘
full             = u32_le(body_len) ++ body
```

The empty-schema `[00,00,00,00]` MatMul is the degenerate `body_len = 0` case of this same outer frame (empty role vectors) — filling the body lights up the roles. The `put_u8_list` helper this needs was added in slice 1 T9 (`lib.rs:226`).

**Worked rank-2 example (16 bytes)** — `lhs = [FreeM, ContractedK] = [1, 3]`, `rhs = [ContractedK, FreeN] = [3, 2]`:

```
body = u32_le(2) ++ [01,03] ++ u32_le(2) ++ [03,02]                (4+2+4+2 = 12 bytes)
full = u32_le(12) ++ body
     = 0C 00 00 00 | 02 00 00 00 | 01 03 | 02 00 00 00 | 03 02      (16 bytes)
```

**Surface-vs-canonical split** (`reply-3:88`): Baracuda's shipped serializer emits a readable **text surface** — `matmul[m k.k n]`, roles as chars `b/m/n/k`, `.`-separated, lhs-then-rhs. The binary §6.19 op_attrs blob is the canonical/identity layer the text flattens onto; both sides treat the binary form as verified canonical. **As of slice 1 T9, Fuel's binary arm is live and its serializer is FIRST**: Baracuda (#68 anti-fork witness) confirmed the rank-2 golden `0C000000|02000000|0103|02000000|0302` and has **no near-term binary arm**, so this golden is the shared **cross-producer contract of record** (`matmul_role_vectors_serialize_the_locked_rank2_golden`, `lib.rs:648`), not merely a Fuel-internal assertion. *(Baracuda's `AxisRole` enum at `baracuda-kernelgen/src/ir.rs:1333` with the same `{Batch=0,FreeM=1,FreeN=2,ContractedK=3}` discriminants is a sibling-project claim recorded at `reply-3:80`; baracuda is not checked out here and it cannot be verified from the Fuel tree.)*

### Serialize / resolve split

Per `reply-3:45-47`:

- **Serialize (Fuel → recipe) — as-built (T9):** `matmul_roles(lhs_rank, rhs_rank)` (`lib.rs:235`) derives the two vectors `Batch×(r−2), FreeM, ContractedK` / `Batch×(r−2), ContractedK, FreeN` — a **pure function of ranks**, so structurally-equal matmuls produce equal blobs (no CSE hazard, `base_map_hash`-stable). The named `to_canonical_bytes` MatMul arm (`lib.rs:344`) serializes whatever roles the `OpAttrs` carries; empty roles → the canonical `[00,00,00,00]`.
- **Resolve (recipe → base map) — as-built (T9):** the `tag_to_op` cell (`runtime_fused.rs:331-337`) checks incoming role vectors against `matmul_roles(...)` — equal-rank ≥ 2, `lhs_roles`/`rhs_roles` equal to the canonical derivation (which places `ContractedK` at lhs-last & rhs-second-last with Batch leading). **Match → `Op::MatMul`; empty → `Op::MatMul`** (implicit-accept). Any other config (transposed operands, permuted contraction, multi-`ContractedK`, `FreeN`-before-`ContractedK`) → a **surfaced opaque-op gap** (`None`/telemetry), **never a crash**.

### No epilogue attr — fused bias/activation composes as elementwise

Per `reply-3:49-59`: a fused `matmul + bias[N] + relu` is one flat DAG — `relu( add( matmul(in0,in1), Bind(2) ) )`. `Reduced(0)` = the K-sum = the matmul node itself (a child_edge, consistent with the "`Reduced(i)` = child_edge to the fold node" rule). **No `epilogue` field on `matmul`.** This matches Fuel's shipped decompose model: `FusedLinear::decompose` (`registry/fused_linear.rs:82-106`) emits `Op::MatMul` (`:88`), `Op::BroadcastTo` bias (`:94`), then an ordinary `Op::Add` **over** the matmul node (`:100`). The re-fuse direction — `canonical_pattern` (`registry/fused_linear.rs:122`) and the `MatMul → Add(rank-1 bias)` fusion pass `fuse_linear` (`opt.rs:1523`) — recognizes `Add(MatMul(a,b), bias_broadcast)`. The epilogue is structural, not an attribute; the role-vector matmul node slots straight in as the fold node.

### Where it landed & exactly what changed (Increment C slice 1 T9)

The schema was **pinned**; Fuel's code conformed in slice 1 — a bounded named increment, not a blocker (`reply-3:61-72`, `:81`). As-built:

1. **DONE — u8-per-role serialize helper** in `fuel-kernel-seam-types/src/lib.rs:226`, because `put_u32_list` writes 4 bytes/element and roles are pinned u8:
   ```rust
   fn put_u8_list(b: &mut Vec<u8>, xs: &[u8]) { put_u32(b, xs.len() as u32); b.extend_from_slice(xs); }
   ```
2. **DONE — a named MatMul arm in `to_canonical_bytes`** (`lib.rs:344`): roles empty → `[00,00,00,00]`; set → `put_u8_list(lhs_roles); put_u8_list(rhs_roles)`. **The design choice settled the cleaner way** (`reply-3:24, :46`): rather than thread operand rank into a call site that has only `&self, op`, the role vectors are **populated into `OpAttrs`** (`lhs_roles`/`rhs_roles`, `lib.rs:187,190`) at region-construction time and serialized from there; `matmul_roles(lhs_rank, rhs_rank)` (`lib.rs:235`) derives the canonical cell. No new `Op` enum field — roles derive from ranks.
3. **DONE — a resolver cell in `tag_to_op`** (`runtime_fused.rs:331-337`): empty roles → `Op::MatMul` (implicit-accept); set → validate the canonical cell (equal-rank ≥ 2, `lhs_roles == matmul_roles(…).0` and `rhs == …1` at the pinned positions); any transposed/permuted/multi-K/`FreeN`-before-`ContractedK` config returns a surfaced opaque-op gap (`None`/telemetry), never a crash — consistent with `tag_to_op`'s existing honest-miss posture (`runtime_fused.rs:363-366`). Because `matmul_roles` encodes role **positions** not extents, GQA-divisible batch stays all-`Batch` and resolves cleanly.

---

# Appendix

## Byte-layout worked examples

**Empty-schema op (`Add`, `MatMul` today):**

```
to_canonical_bytes = u32_le(0) = [00, 00, 00, 00]
```

**`Slice { axis: 1, start: 2, len: 2 }`** (per-op arm `lib.rs`, `axis:u32 ++ start:u64 ++ len:u64`):

```
body = 01 00 00 00 | 02 00 00 00 00 00 00 00 | 02 00 00 00 00 00 00 00   (4+8+8 = 20 bytes)
full = u32_le(20) ++ body = 14 00 00 00 | <body>                          (24 bytes)
```

**`SumDim { axis: -1, keepdim: true }`** (`axis:i64 ++ keepdim:u8`):

```
body = FF FF FF FF FF FF FF FF | 01                                        (8+1 = 9 bytes)
full = 09 00 00 00 | <body>                                                (13 bytes)
```

**Pinned MatMul rank-2 role blob** (`lhs=[FreeM,ContractedK]=[1,3]`, `rhs=[ContractedK,FreeN]=[3,2]`):

```
body = 02 00 00 00 | 01 03 | 02 00 00 00 | 03 02                           (12 bytes)
full = 0C 00 00 00 | <body>                                                (16 bytes)
```

## Glossary

- **Recipe / region / pattern:** the op-DAG a fused kernel decomposes into / re-fuses from. One `PatternNode` grammar object, three (four with KISS §2.3) roles.
- **`PatternNode`:** the grammar node — `Op | Bind` (concrete) plus `SeeThrough | Any` (matcher-only). `fuel-kernel-seam-types/src/lib.rs:254-273`.
- **`OpTag`:** the frozen functional-op vocabulary (`lib.rs:30-59`). Excludes in-place, structural/bookkeeping, and `Op::Fused`.
- **`OpAttrs`:** the 13 non-tensor attribute fields a `PatternNode::Op` carries (`lib.rs:70-120`). Unset field = matcher wildcard.
- **base map:** the primitive-`Op` subgraph a fused op lowers to (the `decompose` fixpoint). `lower_to_base_map`, `opt.rs:364-366`.
- **`base_map_hash`:** NodeId-independent content hash of the base map; CSE/commutativity-invariant; the recipe-identity pre-filter. `opt.rs:399`.
- **STRUCTURE vs INTERFACE:** the recipe DAG (dtype-agnostic, shape-free) vs the FKC contract (dtypes/shape-rules/cost/precision). Joined by `FusedOpId`.
- **`primitive_shape`:** single source of truth for a primitive op's output shape+dtype, derived from operands. `shape.rs:36-40`.
- **FKC contract:** the kernel contract wrapper (`OutputDesc`/`TensorDesc`/caps/cost/precision). `fuel-dispatch/src/fkc/schema.rs`.
- **`ShapeExpr`/`DimExpr`:** the shape-relative expression grammar (`SameAs` + `Extent`/`Const`/`Param`/arithmetic). Evaluator + §6.20 wire codec SHIPPED (Convergence-C @ `9156e178`); the vocabulary's permanent home is `fuel-kernel-seam-types/src/shape_expr.rs` (moved from `fuel-dispatch/src/fkc/` in Increment C slice 1 T1; `fkc/shape_expr.rs` is a `pub use` shim). The recipe interior now carries it via the `OpAttrs` rel fields (slice 1 §A); the remaining ~11 migrations + wire serialization are pending.
- **`reduced_count` / `extent`:** the value-side (Mean divisor) / shape-side single-axis leaves; canonical KISS §6.12-0001 tokens.
- **Increment A / C:** A (emit full-parity, shipped 2026-07-16). C's shape-oracle (evaluator + §6.20 wire codec) SHIPPED as **Convergence-C @ `9156e178`**. C's recipe-interior migration shipped its **foundations in slice 1** (2026-07-23, branch `feat/increment-c-slice1`): shape-relative `OpAttrs` attrs + `resolve_rel_attrs` + `OpTag::MaxDim` + `decompose_via_recipe` + **5 of ~16 migrated decomposes**, and the **matmul role-vector `op_attrs` serialization is SHIPPED (§C)**. Still NOT built: the flat-DAG-CSE node/table WIRE + the remaining ~11 migrations (slices 2–4).

## Cross-references — co-design records

- Recipe-grammar co-design reply / -2 / -3: `docs/outreach/baracuda-recipe-grammar-codesign-reply.md`, `…-reply-2.md`, `…-reply-3.md` (flat §6.4-0009 schema; matmul role-vectors closed in reply-3 §2/§6).
- Shape-expression ask / reply: `docs/outreach/baracuda-shape-expression-grammar-ask.md`, `…-reply.md` (the `ShapeExpr`/`DimExpr` grammar; the `OutputDesc.shape_rule` Fuel-field correction banner at `ask.md:6`).
- Shape-oracle reframe: `docs/outreach/kiss-shape-oracle-reframe-reply.md`, `docs/outreach/baracuda-shape-oracle-rfc-ask.md` (the three homes; the `0xFF`/`MAX_RANK=8` sentinel at `rfc-ask.md:25`).
- `reduced_count` rename: `docs/outreach/baracuda-reduced-count-rename-reply.md`, recorded `docs/specs/kernel-seam-interop.md:517-548`.
- Shape-rule expression vocabulary RFC: `docs/outreach/kiss-rfc-shape-rule-expression-vocabulary.md`.
- Byte-layout spec: `docs/specs/kernel-seam-interop.md:495-548`.
- Merged shape-oracle RFC: KISS main `@3bd6d2d` (cross-party state, asserted in Fuel-side outreach docs; not verifiable from the Fuel tree).

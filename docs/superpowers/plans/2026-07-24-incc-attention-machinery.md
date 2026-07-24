# Increment C — Thread 1: attention-trio + cross-entropy `decompose` → PatternNode recipe

Migrate the four **needs-extension** `decompose` fns — `fused_softmax_cross_entropy`,
`paged_attn`, `flash_attn` (concrete/None k_len arm), `flash_attn_backward` — onto
declarative `PatternNode` **data** recipes re-emitted through `decompose_via_recipe` +
`tag_to_op`/`emit`. This thread settles the two open machinery pieces the plan-of-record
(`docs/superpowers/plans/2026-07-24-increment-c-decompose-migration.md`, items 5 + 7)
names: **config-branch recipe selection** and **nested-fused references**.

All claims below are re-verified against code at `C:\Projects\fuel` @ main `7172309c`.

## Settled mechanism (1): config-branch = per-call recipe builder (NO new machinery)

The `decompose_via_recipe` bridge is **param-agnostic** — it takes a pre-built
`&PatternNode` + a scalar projection, not params (`registry.rs:900-905`). Param-dependent
STRUCTURE is therefore selected **before** the bridge, in the registry `decompose` fn, by a
`recipe(params, shapes) -> PatternNode` builder that uses ordinary Rust control flow and
constant-folds param/shape-derived scalars & diagonals into baked attrs. This is exactly the
shipped precedent:

- `selective_scan.rs:359-394` builds `recipe(seqlen, delta_softplus)` per call; `delta_softplus`
  selects one of **two** `if`-branched recipe variants; `seqlen` bakes into `scan_bound`.
- `powi_backward` constant-folds the `exp` param into the datum (plan doc lines 238-254).

Option **(a)** is thus already fully supported. Option (b) "conditional PatternNode" is **not
expressible** (`PatternNode` = `Op|Bind|Any|SeeThrough`; `Any`/`SeeThrough` are rejected by
`validate_node`, `runtime_fused.rs:874`). Option (c) "N frozen variants" is a degenerate case
of (a). **All four ops in this thread have ZERO open scalar slots** — softmax_scale, softcap
`cap`, `-inf`, Triu/Tril diagonals, `1/n_rows`, `Cast` targets are all baked per call — so each
`scalars(params)` returns `Some(vec![])` on a matching payload, `None` otherwise (the
`selective_scan` posture).

**No C-4 / `DimExpr::Param` threading is needed.** Every diagonal (`o+1`, `o+r+1`, `o-l-1`,
`o = q_pos_offset = kl-Sq` or `0`) and window bound is a concrete integer at decompose time,
baked into `attrs.axis` (`flash_attn.rs:317-326`; `tag_to_op` Triu/Tril read `attrs.axis`,
`runtime_fused.rs:392`). `resolve_rel_attrs` evaluates with EMPTY params
(`runtime_fused.rs:755,710`), so a `Dim::Param` would decline (`shape_expr.rs:314-318`) — we
never author one.

Supported vocabulary confirmed: `Iota` (`len = target_shape[0]`, `runtime_fused.rs:404`),
`Dims`/`WithDim`/`SameAs` (fully evaluated, `shape_expr.rs:346-379`), `MaskedFill` (A2 carrier,
`runtime_fused.rs:455-462`), `Slice`/`IndexSelect`/`Ge`/`Cast`, and batched 4-D `MatMul` +
`Gather` shape inference (`shape.rs:189-231`). Most interior nodes (elementwise, MatMul,
softmax) need **no** shape attr — `primitive_shape` derives them; only shape-changers
(Reshape/BroadcastTo/Slice/reduce/Iota) carry a baked or rel target. Because the legacy always
Reshapes-before-Broadcast at equal rank, the emit D4 auto-pad never fires — keep the explicit
Reshape nodes so the recipe mirrors the legacy exactly.

## Settled mechanism (2): nested-fused = carry `Op::Fused` as-is (2a), NOT inline (2b)

`recompute_probs` emits a real `Op::Fused(SOFTMAX_LAST_DIM, SoftmaxLastDim)` node
(`flash_attn.rs:329-334`); `paged_attn.rs:224-229` the same; `flash_attn_backward.rs:223-231`
emits `Op::Fused(SOFTMAX_LAST_DIM_BACKWARD, SoftmaxLastDimBackward)`. `tag_to_op` honest-misses
`Fused` (`runtime_fused.rs:467`; there is **no** `OpTag::Fused`, `lib.rs:38-126`).

**Choice: 2a (carry the fused node), not 2b (inline softmax's primitives).**
- `base_map_hash` cannot distinguish them — the base map recursively lowers the nested fused
  past itself, so both yield the identical fully-lowered primitive hash. `base_map_hash` /
  recipe-identity is therefore **invariant** to this choice.
- The **single-step** structural parity test (`assert_structural_eq` vs frozen legacy) REQUIRES
  2a: the frozen legacy contains the fused node; inlining would fail it. 2a also preserves the
  design intent (keep softmax fused so the optimizer can re-cover it).

**2a machinery (built in C-T2):**
1. `OpTag::Fused` — a Fuel-INTERNAL structural token, sibling of `OpTag::Scan`/`View`
   (`lib.rs:105-125`), **not** on the §6.19 wire (hits the empty-body `_` arm of
   `to_canonical_bytes`).
2. An `OpAttrs` fused-op-selector field (a stable name string, mirroring `cast_dtype`'s
   `DType::as_str` precedent).
3. A `tag_to_op` arm reconstructing `Op::Fused(fid, params)` via a small `fid -> params` map
   covering exactly the two param-less nested ops (`SoftmaxLastDim`, `SoftmaxLastDimBackward`,
   `registry.rs:177,207`); any other id is an honest miss (`None`).
4. An `emit` structural-terminal arm (sibling of the Scan/View terminals at
   `runtime_fused.rs:1187-1244`) computing shape/dtype from
   `default_registry().entry(fid).shape_rule/dtype_rule` (`registry.rs:811,1133`), since
   `primitive_shape` honest-misses `Fused` (`shape.rs:241-246`); fall back to operand[0].
5. `scalar_slot_arity(Fused) = 0` (already true by default) so `count_scalar_slots` and the
   validator treat it as slot-free/shareable.

`op_to_tag(Fused)` is **NOT** needed for this thread (the base map lowers past it; parity is a
graph comparison, not a round-trip) — leave it out of scope (a future symmetry only ingestion
would need).

## The rank-0 ReduceSumTo([]) carrier gap (FSCE Sum/Mean tail)

Confirmed gap: FSCE's Sum/Mean tail uses `Op::ReduceSumTo(Shape::from_dims(&[]))`
(`fused_softmax_cross_entropy.rs:298,321`), but `shape_from_attr` treats an empty `target_shape`
as UNSET → `None` (`runtime_fused.rs:489-495`), so a rank-0 target is unrepresentable and
`tag_to_op` declines. Resolve inside C-T1:
- **Option A (recommended, structure-preserving):** add a small rank-0 shape-target
  representation (a boolean/Option marker distinguishing "rank-0 target" from "unset") so
  ReduceSumTo/ReduceMaxTo/Reshape/BroadcastTo can target `[]`. Reusable for every future
  reduce-to-scalar loss; keeps the migration bit-exact vs frozen legacy; no real-backend test.
- **Option B (fallback, zero machinery):** emit `SumAll` for the reduce-to-scalar (`SumAll` is
  representable, `runtime_fused.rs:398`, `shape.rs:183`). This is a RESTRUCTURE, so per RISK-B it
  MUST carry a real-backend numeric-tolerance parity test — FSCE already has real-backend
  references in `fuel-core/src/{lazy,train,lazy_nn_loss}.rs`, so the cost is one test.

## Structure-preservation contract (binds every slice)

Each slice ships a `frozen_legacy_decompose` (verbatim copy of today's imperative body) plus a
`<op>_recipe_decompose_is_polymorphic_and_matches_frozen_legacy` test asserting node-for-node
`assert_structural_eq` (op/shape/dtype/arity/recursive-inputs, the selective_scan pattern,
`selective_scan.rs:583-596`). Because every migration is structure-preserving with baked
constants and **no arithmetic restructure**, each inherits slice-1's toy-interpreter numeric
proof — the toy f64 interpreter is not used for these 4-D graphs. `emit`'s within-call
identity-share (`runtime_fused.rs:1063-1072`) reconstructs the legacy DAG sharing (shared
`shifted`/`neg_inf`/`k_rep`/`probs`/`softcap_tanh` subtrees are slot-free → dedup to one node).
Tests run at F32 (the selective_scan convention).

## Sequence & dependencies

`C-T1` (FSCE) is independent — do it FIRST (cheap proof of mechanism 1 + the rank-0 gap).
`C-T2` (paged_attn) BUILDS the nested-fused carrier (mechanism 2a). `C-T3` (flash_attn) and
`C-T4` (flash_attn_backward) both REUSE the carrier; `C-T3` before `C-T4` because they share a
new `recompute_probs_recipe` builder (added in C-T3 alongside the imperative `recompute_probs`,
which backward still uses until C-T4 retires it). `C-T4` also extends the `fid -> params` map
with `SOFTMAX_LAST_DIM_BACKWARD` (one line).

## Explicitly out of scope

`flash_attn`'s `Some(Sym(k_len))` decode arm is a PERMANENT registry-layer basis gap (no
`DynScalar`-length `Slice` inside a `decompose`, `flash_attn.rs:144-148`). The Sym guard stays
an imperative self-return BEFORE recipe construction; the symbolic oracle remains in
`fuel_dispatch::decode_flash`. Closing it is a future primitive-basis (constitution) decision,
not part of this thread.

## Basis / grammar impact

No new primitive Op. Two Fuel-internal, off-wire seam-types additions in the OpTag::Scan/View
class: `OpTag::Fused` + its OpAttrs selector (C-T2), and (if Option A) a rank-0 shape-target
marker (C-T1). Each gets a doc-comment note (Fuel-internal, not serialized to
`to_canonical_bytes`) + a SHIPPED-section entry in the Increment C plan doc — NOT a
`docs/architecture` bump or a 10-decisions-log primitive entry.

## Attention machinery — SHIPPED (branch `feat/incc-attention-machinery`)

All four thread slices landed **structure-preserving** (each migrated `decompose`'s emitted base
map is node-for-node identical to a `frozen_legacy_*` verbatim copy of the pre-migration
imperative body, asserted by a recursive `assert_structural_eq` over op/shape/dtype/arity/inputs;
tests at F32). No new primitive Op; the config-branch (mechanism 1) and nested-fused (mechanism
2a) machinery is exactly as designed above.

### Which decomposes migrated

- **C-T1 — `fused_softmax_cross_entropy`** (`6d3e5d39`): imperative → per-call `recipe(shapes,
  dtypes, reduction) -> PatternNode` re-emitted through `decompose_via_recipe`. Cheap proof of the
  config-branch mechanism (ordinary Rust `if`/`match` selects the None/Sum/Mean tail; all extents
  baked). Resolved the rank-0 tail via **Option A**: added the Fuel-internal `OpAttrs::rank0_target`
  boolean marker so `ReduceSumTo([])`/`ReduceMaxTo([])`/`BroadcastTo([])` can target a rank-0
  scalar (distinguishing "rank-0 target" from "unset"), reusable for every reduce-to-scalar loss.
- **C-T2 — `paged_attn`** (`24f12138`): imperative → recipe, AND **built the nested-fused carrier
  (mechanism 2a)** — `OpTag::Fused` (a Fuel-internal structural token, sibling of `OpTag::Scan`/
  `View`, hits the empty `_` arm of `to_canonical_bytes` — never on the §6.19 wire), the
  `OpAttrs::fused_op` name selector (mirroring `cast_dtype`'s name-string precedent), a `tag_to_op`
  arm reconstructing `Op::Fused(fid, params)` via a small `fid -> params` map, and an `emit`
  structural-terminal arm computing the nested node's shape/dtype from the named registry entry's
  own `shape_rule`/`dtype_rule` (`primitive_shape` honest-misses `Fused`). The softmax rides as
  `Op::Fused(SOFTMAX_LAST_DIM)`.
- **C-T3 — `flash_attn`** (concrete/None `k_len` arms) (`2dda8d26`): imperative → recipe, reusing
  the C-T2 carrier; added the shared `recompute_probs_recipe` builder (GQA head-repeat, softcap,
  alibi, causal/window `-inf` bands, softmax) that C-T4 also consumes.
- **C-T4 — `flash_attn_backward` (Q and K variants)** (this slice): imperative → recipe, reusing
  the C-T2 carrier for the softmax **backward** (`Op::Fused(SOFTMAX_LAST_DIM_BACKWARD)` rides as an
  `OpTag::Fused` node — the `fid -> params` map already covered it). Variant selection (dQ vs dK)
  is ordinary Rust control flow (mechanism 1); the shared softmax-state recompute rides
  `recompute_probs_recipe`, now returning `k_rep` + `softcap_tanh` too via a new `pub(crate)`
  `AttnRecomputeRecipe` struct (the recipe-side mirror of the imperative `AttnRecompute`) — emit's
  slot-free identity-share dedups the standalone `k_rep`/`softcap_tanh` back to the node inside
  `probs`, reproducing the imperative DAG sharing. The GQA fold is `Reshape([b,hkv,g,s,d]) →
  SumDim(2)` (SumDim removes the reduced axis; shape derived by `primitive_shape`); the softcap
  backprop is the baked `Mul(t,t) → MulScalar(-1) → AddScalar(1)` `1 − tanh²` chain. Verified
  end-to-end on the **CPU real backend** (`fuel-core` `flash_attn_backward_decompose_matches_reference`
  realizes dQ/dK/dV within 1e-4 of a hand-computed SDPA-backward reference).

### Permanent registry-layer gaps (kept as documented never-crash self-returns, NOT crashes)

- **`flash_attn` `Some(Sym(k_len))` decode arm** (C-T3): slicing K/V to a *symbolic* length needs a
  `DynScalar`-length `Slice`/mask primitive the basis lacks; `decompose` self-returns BEFORE any
  recipe construction; the symbolic oracle stays the `fuel_dispatch::decode_flash` optimizer arm.
- **`flash_attn_backward` V variant** (C-T4, NEW): `dV = Pᵀ·dO` does **not** reference V, so a
  recipe over the node's `[q,k,v,do,…]` inputs would leave bind 2 (`v`) unreferenced — a
  `NonContiguousBinds` decline from the ONE shared `validate_recipe` invariant (bind `i` ⇔ input
  `i`, contiguous `[0,n)`; load-bearing for the runtime-fused ingestion identity, deliberately NOT
  weakened for one op). So the V variant keeps the always-correct **imperative primitive form**
  (unchanged from the pre-C-T4 body) — a documented registry-layer expressibility gap, the sibling
  of the `flash_attn` Sym gap. (Q and K reference every input → contiguous binds → they migrate.)
  The parity matrix still covers all three variants: Q/K via the recipe re-emit, V via the retained
  imperative path (trivially identical to frozen).

### Seam-types / grammar additions (all Fuel-internal, off the §6.19 wire)

- `OpTag::Fused` + `OpAttrs::fused_op` name selector (C-T2), with the `fid -> params` map covering
  the two param-less nested softmaxes (`SoftmaxLastDim`, `SoftmaxLastDimBackward`).
- `OpAttrs::rank0_target` boolean marker (C-T1) — rank-0 (`[]`) shape target for reduce/broadcast.
- (fuel-graph-internal, not seam-types) `AttnRecomputeRecipe` `pub(crate)` struct exposing
  `probs`/`k_rep`/`v_rep`/`softcap_tanh` from `recompute_probs_recipe` (C-T4).

### Updated Increment-C migrated count

Branch base (`origin/main @ 7172309c`): **12 / 22** registry `decompose` fns migrated. This thread
adds four (`fused_softmax_cross_entropy`, `paged_attn`, `flash_attn`, `flash_attn_backward`
Q/K) → **16 / 22 migrated** after C-T4. The remaining permanent basis/registry gaps
(`flash_attn` symbolic `k_len`, `flash_attn_backward` V, plus the constitution's basis-gap
self-returns — `selective_scan`/`ssd_chunk_scan` higher-order-`Scan` class) are surfaced
never-crash self-returns, not migration debt.

### Gates (all green, forced-clean; F32; no CUDA/GPU)

- `cargo test -p fuel-graph --lib` — 451 passed / 0 failed (incl. the new
  `flash_attn_backward_*` parity + bind-gap + wrong-params tests; forward C-T3 tests unregressed).
- `cargo test -p fuel-core --lib flash_attn_backward_decompose_matches_reference` — 1 passed
  (CPU real-backend numeric parity for dQ/dK/dV).
- `cargo test -p fuel-dispatch --lib` — 712 passed / 0 failed (structure-preserving; recipe-identity
  `base_map_hash` invariant — unaffected).

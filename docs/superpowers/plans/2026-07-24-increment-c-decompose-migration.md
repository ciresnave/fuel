# Increment C — registry `decompose` → PatternNode data-recipe migration (plan of record)

Migrate hand-written `fn decompose` (fuel-graph/src/registry/*.rs) onto declarative
`PatternNode` **data** recipes, via the `decompose_via_recipe` bridge + shape-relative
`OpAttrs` (`SameAs`/`DimExpr`/`WithDim`/`Dims`) + the `resolve_rel_attrs`/`emit` resolver.
This is the Fuel-side finish of the recipe-grammar convergence.

**State (2026-07-24, main @ 80fb3617):** 22 ops carry a `decompose`; **5 migrated** in slice-1
(softmax_last_dim, softmax_last_dim_backward, rms_norm_last_dim, layer_norm_last_dim, rope).
**17 remain**, classified below from a full read of every decompose fn (not memory).

> **Superseded count (see the "Op::Scan recipe form — SHIPPED" section at the bottom):** slice-2
> (layer_norm_last_dim_backward + fused_linear) and the Op::Scan pair (selective_scan +
> ssd_chunk_scan) have since landed → **9 of 22 migrated, 13 remain** (9 needs-extension + 4
> basis-gap). The classification below is the *original* plan-of-record; the trailing SHIPPED
> section is authoritative for what is done.

## Classification (2 mechanical / 11 needs-extension / 4 basis-gap)

### MECHANICAL — migratable with today's grammar (= SLICE 2)
- **layer_norm_last_dim_backward** — ~20 nodes, elementwise + `MeanDim`/`Unsqueeze`/`BroadcastTo(SameAs 0)`, one `eps` open slot. Structure-preserving; inherits slice-1's parity proof (softmax-bwd idiom + eps-slot idiom). **RISK-A:** `xhat` is shared but carries the eps slot, so emit's slot-free identity-share won't dedup it → recipe emits `xhat` twice (2 eps slots); numerically identical, relies on downstream CSE. Verify CSE collapses or accept redundant compute; may motivate an open-slot subtree-sharing emit extension.
- **fused_linear** — 3 nodes: `Add(MatMul(a,b), BroadcastTo(WithDim{op:0,axis:LAST,dim:Extent{op:1,LAST}})(bias))`. **First live WithDim driver** (evaluator+routing exist, unexercised — this proves the path end-to-end; it is Fuel-INTERNAL shape resolution, NOT §6.19 wire emission, so it is NOT gated on KISS #86). **RISK-C:** first slice-2 op with a live (non-stub) `canonical_pattern` — the lower→fuse round-trip must stay green.

### NEEDS-EXTENSION — sequenced by unlock size (each names the exact missing feature)
1. **`reduced_count` shape-derived-scalar live-emission** (narrowest, highest-value next) → **rms_norm_last_dim_backward** (structure-preserving; reproduces `MulScalar(n=last-dim)` bit-exact). = SLICE 3.
2. **`MaskedFill` re-emit carrier** (`Scalar` in `OpAttrs` + a `tag_to_op` arm) → **reduce_max_to_backward** (its only wall; shapes all SameAs); also unblocks one paged_attn blocker.
3. **`PowI` i32-exponent carrier + param-derived attr** (`exp`, `exp-1`) → **powi_backward**.
4. **WithDim/Dims live-emission (proven via fused_linear) + param threading (C-4)** → the derived-shape halves of **causal_conv1d** (also needs extent-driven K-tap unroll + `use_silu` branch) and **nf4_matmul** (also needs product-collapse shape + dtype-relative `Cast` + dtype branch). Multi-feature, not single-unlock.
5. **Config-branch recipe selection** (per-param recipe / conditional node) → **fused_softmax_cross_entropy** (3 reduction tails + dtype branch + shape-derived `1/n_rows`); prerequisite for the attention ops.
6. **`Op::Scan` PatternNode form + `scan_placeholder` leaf live-emission + `Op::View`/`output_views` bundle** → **selective_scan** + **ssd_chunk_scan** (twin; identical machinery; do as a pair). G3 is already CLOSED in code (both decompose to `Op::Scan` today).
7. **Nested-fused references** (a recipe referencing `Op::Fused(SOFTMAX_LAST_DIM…)`) + all above → the attention trio **flash_attn / flash_attn_backward / paged_attn**. Heaviest and last. **RISK-D:** flash_attn's `Some(Sym(k_len))` decode arm is a PERMANENT registry-layer basis gap (no `DynScalar`-length `Slice`/mask inside a `decompose`); migrate the concrete/None arm only, leave the symbolic self-return — the symbolic oracle stays in `fuel_dispatch::decode_flash`.

### BASIS-GAP — permanent self-returns, need a new PRIMITIVE Op (NOT a shape constructor; NOT unblocked by Dims/WithDim)
- **conv2d** → needs `Op::Im2Col` (sliding-window/Unfold).
- **conv_transpose_2d** → needs `Op::Col2Im`/`Op::Im2Col`.
- **qmatmul** → needs GGML bit-unpack + byte-reinterpret + block-layout primitives.
- **inplace_affine** → trivial value (`MulScalar→AddScalar`) but IS its destructive-aliasing contract; needs `Op::AffineInplace{mul,add}` to migrate without dropping the alias semantics.

## Cross-cutting risk (applies to every slice)
**RISK-B:** the parity harness (`eval_rope`/`eval_norm`/`eval_bwd`) is a toy in-order f64 interpreter asserting bit-exact STRUCTURE. Structure-preserving migrations inherit slice-1's proof. **Any arithmetic RESTRUCTURE** (e.g. rms-bwd via `MeanDim` to dodge `MulScalar(n)`, or reduce-max-bwd via `Where` to dodge `MaskedFill`) changes rounding order and MUST carry a real-backend numerical-tolerance parity test — do not take a restructure shortcut to fake a "clean" migration without the real-backend gate.

## Sequence
Slice 2 (this change): layer_norm_last_dim_backward + fused_linear. Slice 3: `reduced_count`
emission + rms_norm_last_dim_backward. Then the carriers (MaskedFill, PowI) → reduce_max/powi
backward. Then config-branch + Op::Scan-PatternNode (scan pair) + nested-fused (attention trio).
The 4 basis-gaps stay surfaced honest-miss self-returns until their primitive Op lands.

---

## Op::Scan recipe form — SHIPPED (2026-07-24, branch `feat/incc-opscan-recipe`)

Classification item **NEEDS-EXTENSION #6** ("`Op::Scan` PatternNode form + `scan_placeholder`
leaf live-emission + `Op::View`/`output_views` bundle → selective_scan + ssd_chunk_scan") is
**done**, out of numbered order (the scan machinery was self-contained enough to build ahead of
slices 3–5). Both ops migrated; neither DESCOPED. All 13 scan-related tests green forced-clean
(`fuel-graph --lib`, new test names present).

### Re-emit machinery added (B1)

`selective_scan`/`ssd_chunk_scan` already *decompose* to `Op::Scan{...}` (a `ScanPlaceholder`-holed
body, then a 2-slot `[ys, final_carry]` bundle projected by `Op::View`) — G3 was already CLOSED in
code. What was missing: `tag_to_op` honest-missed `Op::Scan`/`Op::ScanPlaceholder`/`Op::View`, so
those decomposes could not round-trip through a `PatternNode` **data** recipe. Added:

- **`fuel-kernel-seam-types/src/lib.rs`** — structural `OpTag::Scan` + `OpTag::View` tokens
  (siblings of the pre-existing `OpTag::ScanPlaceholder`) and **Fuel-internal** `OpAttrs` carriers:
  `scan_n_xs` / `scan_bound` / `scan_emit` / `scan_early_exit` (Scan params), `scan_role` /
  `scan_index` (ScanPlaceholder), `view_slot` (View). These are **NOT on the §6.19 cross-producer
  wire** — a `Scan` node hits the empty-body arm of `to_canonical_bytes`; scan is Fuel's own
  higher-order primitive, not a KISS base op.
- **`fuel-graph/src/runtime_fused.rs` `tag_to_op`** (arms ~`:404`): `T::Scan → Op::Scan{n_xs,
  bound, emit, early_exit}` (params from the `scan_*` carriers; the body sub-graph rides the
  operands per the Phase-1 `lax.scan` input encoding), `T::ScanPlaceholder → Op::ScanPlaceholder
  {role, index}` (from `scan_role`/`scan_index`), `T::View → Op::View{slot}` (from `view_slot`;
  the slot shape/dtype are **not** decoded here). `Op::Scan` stays a **base-map terminal** — no
  native kernel, no `LoweringRule`; the recipe only re-emits it.
- **`tag_to_op` (B2-1) — `ScanPlaceholder` body-shape carrier**: `emit` reads a placeholder's
  declared per-step shape from the same `target_shape` / `target_shape_rel` carrier the other
  recipe ops use (dtype from bind 0, the uniform-dtype scan inputs). Load-bearing: `unroll_scan`
  clones body interior nodes with their *stored* shapes, so a rank-0 placeholder poisons the
  unrolled body (e.g. `du = Mul(d_t, u_t)` comes out rank-0). Shapeless authored placeholders still
  fall back to rank-0/F32, never a panic.
- **`fuel-graph/src/runtime_fused.rs` `emit`** (structural-terminal block ~`:1074`): two
  **multi-output structural terminals** are resolved with the graph in hand, because their shape is
  *not* a `primitive_shape` function of operand shapes (mirroring `Tensor::scan` / `Graph::view`):
  - `Op::Scan` → node primary (slot-0) shape = stacked ys `[bound] ++ body_y`, and the 2-slot
    `output_views` bundle (slot 0 = ys, slot 1 = final carry) is composed via
    `OutputViewSpec::contiguous` and attached **after** the push so downstream `Op::View`s can read
    it.
  - `Op::View` → slot shape/dtype read from the producer's `output_views[slot]`. Both fall back to
    operand[0] on a malformed region — **never a panic**.
- **`base_map_hash` interaction (why slot-1 is load-bearing):** `Op::Scan`/`Op::ScanPlaceholder`
  node shapes are **inert** for `base_map_hash` (their `op_key` tags fold only params + body). But
  `Op::View` (`op_key` None) folds its *node* shape — so slot-1's `last_state` shape lives ONLY in
  the re-attached bundle and would break the hash if `emit` failed to re-compose/read it. The
  `scan_view_slot1_reemits_to_the_same_base_map` round-trip test guards exactly this.

### Which decomposes migrated

- **selective_scan** (`fuel-graph/src/registry/selective_scan.rs`) — **MIGRATED.** Bind space
  `0=u, 1=delta, 2=a, 3=b, 4=c`; every interior shape (carry/elem/reshape-mids) rides a §6.20
  `Dims`/`Extent` target, so ONE recipe form covers all `batch`/`dim`/`dstate`. `bound = seqlen` is
  a shape-dependent structural param with no rel carrier → read from the node and baked into the
  per-call recipe. `delta_softplus` selects one of **two recipe variants** (the minimal
  config-branch): the softplus arm prepends `Relu(d) + Log(1 + Exp(Neg(Abs(d))))`. No open scalar
  slots (the softplus `1.0` and zero-init `0.0` are baked constants). The returned `Permute`
  re-attaches the 2-slot `output_views` bundle unchanged.
- **ssd_chunk_scan** (`fuel-graph/src/registry/ssd_chunk_scan.rs`) — **MIGRATED.** The twin of
  selective_scan with NO softplus and a per-head SCALAR gate. Bind space `0=x, 1=dt, 2=a, 3=b,
  4=c`; interior shapes ride `Dims`/`Extent` (batch = `x.0`, heads = `x.2`, head_dim = `x.3`,
  state_dim = `b.3`). `bound = seqlen` baked per call; **`chunk_size` is a documented CPU no-op**
  and never enters the recipe (both the chunked GPU path and this serial CPU path give the same
  answer; locked by `ssd_chunk_scan_recipe_ignores_chunk_size` via `base_map_hash` equality). No
  open scalar slots (zero-init `0.0` baked). Returned `Permute` re-attaches the 2-slot bundle.

**No DESCOPED arm.** selective_scan's `delta_softplus` (the one config branch either op has) is
fully handled as the two recipe variants above — it did not need to be split off.

### Parity / frozen-legacy pattern (mirrors slice-1/slice-2)

Each op ships a `frozen_legacy_decompose` (the pre-B2 imperative body, verbatim) plus a
`<op>_recipe_decompose_is_polymorphic_and_matches_frozen_legacy` test that runs both through the
graph and asserts the recipe re-emit is **node-for-node identical** across shapes — and, for
selective_scan, across **both** softplus variants. Born-red without B2-1's placeholder-shape
carrier (the placeholders and `du` emit rank-0). Structure-preserving throughout: the recipe
mirrors the imperative `unroll_scan`-shaped node exactly (op types, params, body structure, the
same `OutputViewSpec::contiguous` bundle), so it inherits slice-1's toy-interpreter proof — no
arithmetic was restructured to dodge a carrier. Guards: `<op>_recipe_wrong_params_is_a_fixpoint_
not_a_crash` (bridge decline / wrong-params stay G2 fixpoints), and `ssd_chunk_scan_recipe_
ignores_chunk_size`. fuel-core lazy decompose / unroll-vs-fused-kernel / BPTT parity tests stay
green.

### Updated migrated-count

**9 of 22 registry `decompose` fns migrated; 13 remain.**

- Migrated (9): softmax_last_dim, softmax_last_dim_backward, rms_norm_last_dim, layer_norm_last_dim,
  rope (slice-1); layer_norm_last_dim_backward, fused_linear (slice-2 mechanical); **selective_scan,
  ssd_chunk_scan (this Op::Scan work).**
- Remaining (13): **9 needs-extension** — causal_conv1d, flash_attn, flash_attn_backward,
  fused_softmax_cross_entropy, nf4_matmul, paged_attn, powi_backward, reduce_max_to_backward,
  rms_norm_last_dim_backward; **4 basis-gap** — conv2d, conv_transpose_2d, qmatmul, inplace_affine.

The scan pair is the only NEEDS-EXTENSION item closed so far; items 1–5 and 7 (and the 4
basis-gaps) are untouched by this change.

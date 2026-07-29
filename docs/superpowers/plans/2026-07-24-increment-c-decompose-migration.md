# Increment C — registry `decompose` → PatternNode data-recipe migration (plan of record)

Migrate hand-written `fn decompose` (fuel-graph/src/registry/*.rs) onto declarative
`PatternNode` **data** recipes, via the `decompose_via_recipe` bridge + shape-relative
`OpAttrs` (`SameAs`/`DimExpr`/`WithDim`/`Dims`) + the `resolve_rel_attrs`/`emit` resolver.
This is the Fuel-side finish of the recipe-grammar convergence.

**State (as authored 2026-07-24, main @ 80fb3617):** 22 ops carry a `decompose`; **5 migrated**
in slice-1 (softmax_last_dim, softmax_last_dim_backward, rms_norm_last_dim, layer_norm_last_dim,
rope). **17 remain**, classified below from a full read of every decompose fn (not memory).

**Running count (slices 2–3 shipped):** slice-2 (`layer_norm_last_dim_backward` + `fused_linear`,
on `origin/main`) and slice-3 (the re-emit carriers on branch `feat/incc-reemit-carriers` — see
the **Slice 3 … SHIPPED** section at the end of this doc) bring the running total to **10 of 22
`decompose` migrated / 12 remain** (4 of which are the permanent basis-gap self-returns).

> **Superseded count (see the "Op::Scan recipe form — SHIPPED" section at the bottom):** slice-2
> (layer_norm_last_dim_backward + fused_linear) and the Op::Scan pair (selective_scan +
> ssd_chunk_scan) have since landed → **9 of 22 migrated, 13 remain** (9 needs-extension + 4
> basis-gap). The classification below is the *original* plan-of-record; the trailing SHIPPED
> section is authoritative for what is done.

## Classification (2 mechanical / 11 needs-extension / 4 basis-gap)

> **2026-07-24 correction:** conv2d + conv_transpose_2d are reclassified out of BASIS-GAP into a new
> "migratable-with-existing-primitives (index-gather im2col)" group below → the true basis-gap count
> is now **2** (qmatmul, inplace_affine). Every basis-gap count of **4** in the counts/running-totals
> below (e.g. "4 basis-gap", "12 remain (4 of which are the permanent basis-gap self-returns)")
> predates this correction and should read **2**.

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

### MIGRATABLE-WITH-EXISTING-PRIMITIVES via the index-gather im2col idiom (CORRECTION, 2026-07-24)
> **Reclassified 2026-07-24 (design pass `2026-07-24-incc-conv-im2col`).** conv2d/conv_transpose_2d
> were originally listed below as BASIS-GAP "need a new PRIMITIVE Op." **That was wrong.** im2col is
> an overlapping-window *strided gather* (`Op::IndexSelect` over the `Op::Pad`-padded, flattened
> spatial axis; window→flat-index map from `Op::Iota` + scalar arithmetic + `Op::Cast(U32)`; constant
> node count — NOT the `N·Hout·Wout` `Slice`/`Concat` soup the old note weighed and correctly
> rejected); col2im is its scatter-add adjoint (`Op::IndexAdd`/`Op::ScatterAdd` into a `MulScalar(0.0)`
> zero base, then `Op::Slice` crop); the grouped/batched matmul is `Op::MatMul`'s batched form. All are
> already in the build-time-closed `Op` basis and re-emittable by `tag_to_op`. **No `Op` variant is
> added.** So the basis-gap count drops from **4 to 2** — the "4 basis-gap" counts recorded elsewhere
> in this doc (headers + running totals) are superseded by this correction.
- **conv2d** → total `PatternNode` recipe: `Pad` → flatten → `IndexSelect` (im2col) → batched `MatMul` → `Reshape` (+broadcast `Add` bias). Gated by a real-backend numerical-parity test (`fuel-conv::conv2d_direct`), NOT the toy f64 interpreter (which doesn't implement `IndexSelect`/`Iota`/`MatMul`/`Pad`). See the `2026-07-24-incc-conv-im2col` plan.
- **conv_transpose_2d** → total `PatternNode` recipe: `MatMul` → col2im scatter-add (`IndexAdd`) → `Slice` crop. Reuses conv2d's index-gather helper; gated against the native CPU conv_transpose kernel.

### BASIS-GAP — permanent self-returns, need a new PRIMITIVE Op (NOT a shape constructor; NOT unblocked by Dims/WithDim)
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
  rms_norm_last_dim_backward; **2 basis-gap** — qmatmul, inplace_affine (conv2d + conv_transpose_2d
  reclassified 2026-07-24 to migratable-with-existing-primitives — index-gather im2col; see the
  correction under "Classification" above).

The scan pair is the only NEEDS-EXTENSION item closed so far; items 1–5 and 7 (and the 4
basis-gaps) are untouched by this change.

---

## Slice 3 (re-emit carriers) — SHIPPED

Branch `feat/incc-reemit-carriers` (based on `origin/main` @ `9d7a1380`, which already carries
slices 1+2). The three named first-order backward `decompose` fns — `rms_norm_last_dim_backward`,
`reduce_max_to_backward`, `powi_backward` — **all migrated to portable, shape-/rank-polymorphic
`PatternNode` data recipes; NONE were DESCOPED.** Each was unblocked by a new reusable re-emit
carrier. Every migration is **STRUCTURE-PRESERVING** — the recipe base map is bit-exact-equivalent
to today's imperative body (no arithmetic restructure; at most a metadata-only Reshape→Unsqueeze
keepdim spelling), so each inherits slice-1's toy-interpreter parity proof and ships a
`frozen_legacy_*` verbatim copy of the pre-migration builder plus a
`<op>_recipe_decompose_is_polymorphic_and_matches_frozen_legacy` parity test observed
red-then-green. **Running total after this slice: 10 of 22 `decompose` migrated / 12 remain**
(4 of which are the permanent basis-gap self-returns).

### Reusable carriers added (all in `fuel-graph/src/runtime_fused.rs` unless noted)

1. **`reduced_count` shape-derived scalar live-emission — `OpAttrs.scalar_rel` (`322a3719`).**
   A recipe scalar slot filled at emit time from an input SHAPE, not from the params projection:
   `OpAttrs.scalar_rel: Option<shape_expr::Dim>` (field added in `fuel-kernel-seam-types/src/lib.rs`)
   carries a `DimExpr` over the Bind space (e.g. `Extent{operand:0, axis:LAST}` = n = the
   reduced/last-axis extent — the `MulScalar(n)` divisor a norm backward needs). Rides the SAME
   `eval_dim`/`resolve_axis` machinery as `slice_start_rel`/`slice_len_rel` (no parallel axis
   resolver); `resolve_rel_attrs` folds it into `scalars` (a `Gap` → `SymbolicGap`; any concrete
   value, incl. negative, flows through — a scalar is not an extent/offset). NEVER a params-cursor
   slot (`count_scalar_slots` unchanged, no cursor consume) — mirrors a baked value.
   `rel_abs_conflict_field` makes `scalar_rel` XOR a non-empty `scalars` a typed
   `RelAbsConflict{field:"scalars"}`. NOT on the §6.19 wire (pinned by
   `rel_attr_fields_are_absent_from_the_6_19_wire`).

2. **MaskedFill re-emit carrier (`94c69ec7`).** MaskedFill was the last scalar-carrying op
   honest-missed by `tag_to_op`'s `_ => return None` set. The fill VALUE rides
   `OpAttrs.scalars[0]`; its dtype rides `cast_dtype` when present (else a provisional F32). The
   `tag_to_op` MaskedFill arm reconstructs `Op::MaskedFill{value}` via a never-panic
   `masked_fill_scalar(f64, DType) -> Option<Scalar>` (None on the sub-byte dummy quant formats —
   an honest miss, not a crash). `emit` re-resolves the fill `Scalar` to operand[0]'s emitted
   dtype (the byte executor derives `fill_bytes` at the tensor width, so the Scalar dtype MUST
   equal the tensor dtype — mirrors the BroadcastTo D4 special-case). Baked pattern constant,
   `scalar_slot_arity(MaskedFill) == 0`. Reconstructor `fuel_ir::Scalar::from_f64` added in
   `fuel-ir/src/scalar.rs`.

3. **PowI i32-exponent re-emit carrier (`864a36b7`).** The other `_ => return None`
   scalar-carrying op. The i32 exponent rides `OpAttrs.scalars[0]` as an f64 — EXACT for every
   i32 (|n| < 2^53), reconstructed via `as i32`; the `tag_to_op` PowI arm is
   `Op::PowI(*attrs.scalars.first()? as i32)` (an unset value is an honest miss, never a defaulted
   `PowI(0)`). `op_to_attrs` (`fuel-graph/src/jit.rs`) now projects
   `Op::PowI(n) => scalars = vec![n as f64]`, completing the `op_to_tag ↔ tag_to_op` round-trip —
   the SAME carrier the §6.19 wire already commits to (its `to_canonical_bytes` PowI arm
   serializes `scalars`), so the wire round-trip is now correct for PowI too. Baked constant,
   `scalar_slot_arity(PowI) == 0`. Two tests that used PowI as a carrier-less unrepresentable
   stand-in (`register_rejects_unrepresentable_region`,
   `decompose_via_recipe_declines_an_unknown_token_recipe`) were repointed to **Clamp**, now the
   canonical still-unrepresentable tag (awaiting a two-scalar carrier).

### Migrations

- **rms_norm_last_dim_backward** (A1, migration `72eb481c` over carrier `322a3719`). The
  ~22-node imperative closed-form `grad_x = r_rms·(g − x·s/(n·(mean_sq+eps)))` → data recipe. FIRST
  recipe to carry a shape-derived scalar: `n = MulScalar(scalar_rel = Extent(0, LAST))`,
  emit-resolved from x's shape (not baked, not a params slot). Sibling of slice-2's
  `layer_norm_last_dim_backward`: `MeanDim`/`SumDim(axis_last)` + `Unsqueeze(append)` keepdim
  (D3 swap) + `BroadcastTo(SameAs 0)`; the `eps` `AddScalar` is an OPEN slot filled by the
  projection. Binds: 0=x, 1=upstream. **RISK-A** (as forecast): `denom_kd` carries the eps slot and
  feeds BOTH Rsqrt and MulScalar(n), so emit's slot-free share does not dedup it (recomputed; CSE
  re-collapses) — numerically identical, mirrors the layer-norm-backward posture. Parity/structural
  tests: `frozen_legacy_rms_norm_backward_decompose`,
  `rms_norm_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy` (bit-exact rank 2
  & 3), `rms_norm_backward_recipe_uses_the_unsqueeze_keepdim_swap_and_shape_scalar`. Gate:
  `fuel-graph --lib` 415.

- **reduce_max_to_backward** (A2, migration `98194c48` over carrier `94c69ec7`). The 9-node
  imperative fair-share max subgradient
  (`ReduceMaxTo→BroadcastTo→Equal(U8)→MaskedFill→ReduceSumTo→Div→BroadcastTo→Mul`) → data recipe.
  FIRST recipe to carry a MaskedFill (A2 carrier, baked value 1.0, no `cast_dtype`). Reduce & count
  targets = `SameAs{operand:1}` (upstream's shape); the two broadcasts = `SameAs{operand:0}` (x's
  shape) — all D2, nothing baked. `mask_f` is a slot-free subtree, so emit's identity-share dedups
  its TWO use sites (ReduceSumTo + final Mul) to ONE emitted node = the imperative single-compute
  DAG; no D3 keepdim swap and no D4 pad fire → base map bit-identical (no restructure). Sabotage-
  calibrated (swapping the Div operands turns the parity oracle RED, 1.111 vs 0.9; a fill-value
  perturbation is invisible — the mask scale cancels algebraically). Tests:
  `reduce_max_to_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy`,
  `reduce_max_to_backward_recipe_shares_mask_and_has_no_reshape`,
  `reduce_max_to_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash`. Gates: `fuel-graph` 420,
  `fuel-ir` 40, `fuel-dispatch` 712, `fuel-core --lib` 1385.

- **powi_backward** (A3, migration `fbe746e9` over carrier `864a36b7`). The 3-node imperative
  gradient `grad_x = exp · x^(exp-1) · upstream` (`PowI(exp-1)→MulScalar(exp)→Mul`) → data recipe.
  FIRST recipe to carry a PowI (A3 carrier). Minimal **C-4 posture**, NOT a restructure: unlike the
  shape-polymorphic slice-1/-2 recipes (a `OnceLock` datum + open slots), PowIBackward's structure
  depends on the param VALUE (the exponent is a structural i32, not a shape and not an f64 slot), so
  both param-derived constants are CONSTANT-FOLDED into the datum at build time (`recipe(exp)` built
  per decompose): PowI carries `scalars=[(exp-1) as f64]` (the carrier reconstructs `PowI(exp-1)`),
  MulScalar carries `scalars=[exp as f64]` — both baked ⇒ zero open slots ⇒ the projection is the
  empty vec; a wrong params payload declines before building the recipe (G2 fixpoint). No D3/D4, no
  reduces/broadcasts → base map bit-identical. Sabotage-calibrated (baking the PowI exponent as
  `exp` not `exp-1` turns the oracle RED, 11.64375 vs 7.7625, and the structural test RED,
  PowI(3)≠PowI(2)). Tests:
  `powi_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy` (exponents
  {3,2,5,1,0,-2}), `powi_backward_recipe_is_a_direct_mirror_with_carrier_exponent`,
  `powi_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash`. Gates: `fuel-graph --lib` 425,
  `fuel-dispatch` 712, `fuel-core --lib` 1385 (incl. the real-backend
  `powi_backward_decompose_matches_reference`).

### Remaining after slice 3 (12 `decompose` fns)

8 NEEDS-EXTENSION migrations still queued behind their named unlock — `causal_conv1d`,
`nf4_matmul` (WithDim/Dims live-emission + param threading, C-4); `fused_softmax_cross_entropy`
(config-branch recipe selection); `selective_scan` + `ssd_chunk_scan` (Op::Scan PatternNode form,
the scan pair); `flash_attn` / `flash_attn_backward` / `paged_attn` (nested-fused references;
flash's `Some(Sym(k_len))` decode arm stays a PERMANENT registry-layer basis gap) — plus the 2
BASIS-GAP self-returns (`qmatmul`, `inplace_affine`) that stay surfaced honest-miss until their new
primitive Op lands. (`conv2d` + `conv_transpose_2d` were reclassified 2026-07-24 out of BASIS-GAP —
they migrate via the index-gather im2col recipe with existing primitives; see the "Classification"
correction above and the `2026-07-24-incc-conv-im2col` plan.)

## conv family — SHIPPED (2026-07-24, branch `feat/incc-conv-im2col`)

The two ops reclassified out of BASIS-GAP on 2026-07-24 (`conv2d`, `conv_transpose_2d`)
are now BOTH migrated — via the index-gather im2col recipe using only existing
build-time-closed-basis primitives (NO new `Op`, no `tag_to_op`/`OpAttrs`/`primitive_shape`
change). Design pass + build plan: `2026-07-24-incc-conv-im2col.md` (see its §9 "conv
family — SHIPPED").

- **`conv2d`** (CV1/CV2, `registry/conv2d.rs`) — `Pad`→flatten→`Iota`+arith+`Cast(U32)`
  index→`IndexSelect` (im2col)→batched `MatMul`→`Reshape` (+optional broadcast `Add`
  bias). Any `groups>=1` incl. depthwise (rank-4 batched over `[N, groups]`). Gated vs
  `fuel_conv::conv2d_direct`, `rel<1e-5`.
- **`conv_transpose_2d`** (CV3, `registry/conv_transpose_2d.rs`) — the col2im
  (overlap-add) adjoint: `weightᵀ`-arranged batched `MatMul`→column stack→`IndexAdd`
  (`+=` overlap-add) into a zero base (`MulScalar(0)` of a length-1 `Slice` broadcast, no
  `Const`)→`Reshape`→`Slice` crop. Any `groups>=1` (2-input, no-bias — the builder's
  form). Gated vs the **native CPU `ConvTranspose2D` kernel** (`fuel-conv` has no
  transposed reference; the production `realize_f32` path never fires the decompose, so
  the oracle is independent), `rel<1e-5`, sabotage-calibrated.

Gates (forced-clean): `fuel-graph --lib` 445, `fuel-core --lib` 1385, `fuel-dispatch`
712(+1), `fuel-core --test incc_conv_transpose_im2col_oracle` 3 (+ regressions
`lazy_conv_transpose1d_oracle` 3, `conv_tests` 8) — all green.

### Updated migrated-count (supersedes the earlier layered totals)

Ground truth (directly measured: registry `decompose` fns that re-emit a portable recipe
via `decompose_via_recipe` / the `Op::Scan` recipe form): **14 of 22 migrated.** The 14:
`softmax_last_dim`, `softmax_last_dim_backward`, `layer_norm_last_dim`,
`layer_norm_last_dim_backward`, `rms_norm_last_dim`, `rms_norm_last_dim_backward`, `rope`,
`fused_linear`, `reduce_max_to_backward`, `powi_backward`, `selective_scan`,
`ssd_chunk_scan`, **`conv2d`**, **`conv_transpose_2d`**.

The remaining self-returns: 5 NEEDS-EXTENSION migrations still queued behind their named
unlock (`causal_conv1d`, `nf4_matmul`, `fused_softmax_cross_entropy`, `flash_attn`,
`paged_attn` — plus `flash_attn`'s `Some(Sym(k_len))` decode arm which stays a PERMANENT
registry-layer basis gap) and the **2 permanent BASIS-GAP** self-returns (`qmatmul`
sub-byte bit-unpack, `inplace_affine` destructive-affine aliasing). The conv pair is no
longer in either "remain" bucket — the basis-gap count is now **2**, not 4.

> **SUPERSEDED 2026-07-25 (see "inplace_affine — SHIPPED" at the very bottom):**
> `inplace_affine` migrated — it was mis-labeled a basis gap (its in-place-ness is a
> `destructive_input()` + KISS-Contract facet, not a decompose concern). The permanent
> BASIS-GAP count drops from **2 to 1** — **`qmatmul` is now the SOLE basis gap.**

---

## Attention machinery — SHIPPED (2026-07-24, branch `feat/incc-attention-machinery`)

The config-branch + nested-fused machinery (designed BUILD_AS_DESIGNED, no constitution/basis
change) migrated **4 more ops**, landed to main just ahead of the conv pair:
- **`fused_softmax_cross_entropy`** — config-branch reduction tail (None/Sum/Mean) via ordinary
  Rust control flow in a per-call `recipe(...)` builder (mechanism 1 needs NO new bridge
  machinery — `decompose_via_recipe` is param-agnostic). Added the reusable Fuel-internal,
  off-wire **`OpAttrs::rank0_target`** marker so `ReduceSumTo`/`BroadcastTo` can target `[]`
  (the reduce-to-scalar loss idiom).
- **`paged_attn`, `flash_attn` (concrete/None `k_len` arm), `flash_attn_backward`** — via a
  Fuel-internal **`OpTag::Fused`** structural token (sibling of Scan/View, NOT serialized to
  `to_canonical_bytes`) + an OpAttrs fused-op-selector, so a recipe carries a nested
  `Op::Fused(SOFTMAX_LAST_DIM[_BACKWARD])` node as-is (mechanism 2a — base map lowers past it;
  the optimizer can re-cover it). Config branches (causal/window/softcap/alibi/GQA, Q/K/V
  variants, Triu/Tril diagonals) bake as concrete attrs per-call — **NO C-4 param-threading
  needed**. All structure-preserving (frozen-legacy parity, sabotage-verified); `flash_attn`'s
  `Some(Sym(k_len))` arm stays a documented PERMANENT self-return (registry-layer basis gap,
  symbolic oracle lives in `fuel_dispatch::decode_flash`).

## AUTHORITATIVE TOTAL (both 2026-07-24 parallel streams landed)

**18 of 22 registry `decompose` fns migrated; 4 remain.** = the 14 above + the 4 attention ops
(`fused_softmax_cross_entropy`, `paged_attn`, `flash_attn` concrete arm, `flash_attn_backward`).
Remaining 4: **2 NEEDS-EXTENSION** (`causal_conv1d` extent-driven K-tap unroll + `use_silu`;
`nf4_matmul` product-collapse + dtype-relative Cast) and **2 permanent BASIS-GAP** self-returns
(`qmatmul`, `inplace_affine` — the genuinely-open basis questions).

## inplace_affine — SHIPPED (2026-07-25, branch `feat/incc-inplace-affine`)

`inplace_affine` is **MIGRATED** — it was mis-labeled a permanent basis gap.

## causal_conv1d + nf4_matmul — SHIPPED (2026-07-25)

Both NEEDS-EXTENSION ops migrated structure-preserving via the config-branch mechanism (a per-call
`recipe(params, shapes) -> PatternNode` builder, ordinary Rust control flow, everything baked
concrete — NO C-4 param-threading): **`causal_conv1d`** (`1be34abb`, branch `feat/incc-causal-conv1d`
— the extent-driven `for tap in 0..kernel` unroll is a concrete Rust loop; `use_silu` a Rust `if`;
24-config parity + real-executor `causal_conv1d_decompose_matches_reference`) and **`nf4_matmul`**
(`13da5d09`, branch `feat/incc-nf4-matmul` — product-collapse `M'=∏leading` baked; dtype-relative
`Cast` as the FSCE `needs_cast` idiom; parity + `nf4_matmul_decompose_matches_kernel`).

## FINAL AUTHORITATIVE TOTAL (2026-07-25)

**21 of 22 registry `decompose` fns migrated; 1 remains.** The sole remaining self-return is
**`qmatmul`** — and even that is very likely NOT a compute-basis question but a **storage-decode**
one (GGML `BlockQ*` is a physical `SType`/`Encoding`, decoded at the FDX/storage boundary, not a
new KISS-Ops token; the residual sub-byte unpack/bitcast should get the conv2d-style gate-question
against existing bitwise/reshape/cast primitives first). No new primitive Op was added anywhere in
Increment C; the basis stayed build-time-closed throughout.

**The corrected finding.** The earlier "basis gap" note was over-conservative: it conflated the
*value* layer with the *destructive/aliasing* layer. In-place-ness is a KISS-Contract §4.6/§5.4 +
`Op::destructive_input()` facet — **NOT an op-basis or decompose concern**. So no new primitive
(`Op::AffineInplace` is NOT needed to migrate) and no standard change.
- `Op::destructive_input() -> Some(0)` is a method on the `Op` (a per-fused-id match arm in
  `fuel-graph/src/lib.rs:1213`) that drives `opt::derive_ordering` on the EXECUTION graph, where
  the fused `Op::Fused(INPLACE_AFFINE)` node lives. It is **UNAFFECTED** by what `decompose`
  returns — the migration does not touch it (guarded by
  `inplace_affine_destructive_input_survives_migration`).
- `decompose` feeds only the BASE-MAP COVER (value / `base_map_hash` / verify-oracle /
  correctness-floor fallback), NOT execution. The fused op still executes via its own in-place
  kernel (`affine_inplace_*`, Phase 3, unlanded); the functional decompose is the correctness-floor
  fallback until then.

The functional `MulScalar(mul) → AddScalar(add)` recipe (`x = mul·x + add`) is value-correct,
does not "drop" the destructive contract, and is self-consistent (the functional form doesn't
mutate, so needs no ordering pin) — making `decompose` **total** (the recipe principle) and giving
`InplaceAffine` an executable functional fallback. Two OPEN scalar slots filled from
`FusedOpParams::InplaceAffine { mul, add }` in pattern pre-order (outer `AddScalar` ← `add`, inner
`MulScalar` ← `mul`), so the projection is `vec![add, mul]`.

Tests (all observed red→green where applicable): `fuel-graph`
`inplace_affine_decompose_lowers_to_mul_then_add_scalar` (structural, born-red),
`inplace_affine_destructive_input_survives_migration` (the load-bearing facet guard),
`inplace_affine_wrong_params_is_a_fixpoint_not_a_crash` (G2); `fuel-core`
`inplace_affine_decompose_matches_affine_reference` (real-backend CPU, bit-exact vs `mul·x + add`).
Gates (forced-clean): `fuel-graph --lib` 462, `fuel-dispatch --lib` 712, `fuel-core --lib` 1386 —
all green.

## COMPLETE: 22 of 22 (2026-07-28) — `qmatmul` (Q4_0) migrated

**All 22 registry `decompose` fns are migrated.** The last self-return, `qmatmul`, now carries a
total primitive recipe for **Q4_0** — the sole GGUF format the live loader produces. This closes
the last opaque island in the fused-op set (the optimizer can lower every fused op to base map).

**The corrected finding (supersedes the 2026-07-25 "storage-decode / basis-gap" framing above).**
`qmatmul` is NOT a basis gap and needs NO new primitive. Verified via the propose-first round with
KISS/kiss-ref/Baracuda (`docs/outreach/bitcast-basis-token-design-input-ask.md`): the byte-
reinterpret every quant consumer needs is on a **loaded constant**, so per KISS's §7.3-0002
necessity test it is boundary-hoistable and does NOT justify an `Op::Bitcast` axiom — the floor
stays closed. The recipe uses only existing primitives:
- **U32 → per-block bytes by exact F64 arithmetic.** `Cast(U32→F64)` is exact (`u32 < 2⁵³`); each
  little-endian byte is `⌊u/256ⁱ⌋ mod 256` via `Floor`/`MulScalar`/`Sub`. (F32 can't: `Cast(U32→
  F32)` rounds above `2²⁴`.) Bytes are `< 256` → `Cast(F64→F32)` exact, rest runs in F32.
- **f16 block-scale by arithmetic IEEE-754-half reconstruction** (not a bitcast): field-extract
  sign/exp/mant by `Floor`/`Sub`, build the data-dependent `2^exp` by a 5-bit binary decomposition
  (product of `2^(2^i)` selected per bit — exact multiplies), blend the subnormal branch. Bit-exact
  to `f16::to_f32` for every finite half — the same "bytes as small exact integers + arithmetic"
  technique `nf4_matmul` uses for nibble unpack.
- **block layout = `Reshape` + `Slice`**; nibble unpack + per-block `BroadcastTo` scale + GEMM.

Contained entirely to `fuel-graph/src/registry/qmatmul.rs` — **no** builder/loader/dispatch/kernel/
dtype change (the F64 route avoids the `U32→U8` weight-representation change that would otherwise be
forced). The fused dequant-in-kernel arm stays the cost-preferred cover; the lowering is the
optimizer/basis-map alternative + correctness-floor fallback, never the executed path where the
kernel exists.

Tests (red→green): `fuel-graph` `f16_decode_arithmetic_matches_half_for_all_finite_bit_patterns`
(the load-bearing anchor — bit-exact over all 63488 finite f16 bit patterns),
`qmatmul_q4_0_decompose_fires_and_is_shape_correct`, `qmatmul_nonq4_0_format_is_a_surfaced_gap_
fixpoint` (G2), `qmatmul_wrong_params_is_a_fixpoint_not_a_crash` (G2); `fuel-core`
`incc_qmatmul_q4_0_oracle::qmatmul_q4_0_recipe_matches_exact_dequant` (real-backend CPU-realize vs
exact `BlockQ4_0::to_float`, `rel<1e-5`, sabotage-calibrated).

**Backlog (ROADMAP): `qmatmul` per-format build-out.** The other ten `QuantType`s (`Q4_1`, `Q5_0`,
`Q5_1`, `Q8_0`, `Q8_1`, `Q2K`, `Q3K`, `Q4_K_M`, `Q5K`, `Q6K`) `decompose` as surfaced gaps
(self-return) — each its own recipe over the same technique (F64 byte-extract + shared f16 decode),
differing only in block layout + scale structure (flat / scale+min / scale+high-bit / hierarchical
super-block sub-scales). Sequence behind consumers (wire the format into the loader first). This is
the `flash_attn` concrete-vs-symbolic precedent: migrated with the uncovered configs documented as
gaps.

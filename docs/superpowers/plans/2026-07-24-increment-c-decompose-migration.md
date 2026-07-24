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
flash's `Some(Sym(k_len))` decode arm stays a PERMANENT registry-layer basis gap) — plus the 4
BASIS-GAP self-returns (`conv2d`, `conv_transpose_2d`, `qmatmul`, `inplace_affine`) that stay
surfaced honest-miss until their new primitive Op lands.

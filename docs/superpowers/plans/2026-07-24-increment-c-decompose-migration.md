# Increment C — registry `decompose` → PatternNode data-recipe migration (plan of record)

Migrate hand-written `fn decompose` (fuel-graph/src/registry/*.rs) onto declarative
`PatternNode` **data** recipes, via the `decompose_via_recipe` bridge + shape-relative
`OpAttrs` (`SameAs`/`DimExpr`/`WithDim`/`Dims`) + the `resolve_rel_attrs`/`emit` resolver.
This is the Fuel-side finish of the recipe-grammar convergence.

**State (2026-07-24, main @ 80fb3617):** 22 ops carry a `decompose`; **5 migrated** in slice-1
(softmax_last_dim, softmax_last_dim_backward, rms_norm_last_dim, layer_norm_last_dim, rope).
**17 remain**, classified below from a full read of every decompose fn (not memory).

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

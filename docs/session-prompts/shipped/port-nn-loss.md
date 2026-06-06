# Port: fuel-nn loss functions to lazy

## Eager source

- `fuel-nn/src/loss.rs` (~250 LOC). Functions: `nll`,
  `cross_entropy`, `binary_cross_entropy_with_logit`, `mse`,
  `huber`. All take eager `Tensor` arguments and return eager
  `Tensor`.

## Lazy module name

`fuel-core/src/lazy_nn_loss.rs` (new file).

## Architecture summary

Pure compositions of existing primitives. Each loss is a small
formula expressed in terms of `LazyTensor` ops.

`nll(inp, target)`: `inp` is `[N, C]` log-probabilities, `target` is
`[N]` u32 class labels. Output: scalar = `-mean(inp[i, target[i]])`.
Implement via `index_select` along dim 1 with `target`-as-index,
then `mean`.

`cross_entropy(inp, target)`: applies `log_softmax` then `nll`.
**Use the shipped `FusedSoftmaxCrossEntropy` fused op** — it
already collapses log_softmax + NLL into a single graph op. Just
expose a thin `cross_entropy(inp, target, reduction)` wrapper.

`binary_cross_entropy_with_logit(inp, target)`: scalar mean of
`max(x, 0) - x*y + log(1 + exp(-|x|))`. Straight tensor algebra.

`mse(inp, target)`: `mean((inp - target)^2)`.

`huber(inp, target, delta)`: piecewise loss; `quadratic` for
`|x| < delta`, `linear` outside. Implement via element-wise
where / select.

## Primitives needed

- All shipped: `index_select`, `mean`, `log_softmax`,
  `FusedSoftmaxCrossEntropy`, `sub`, `mul`, `add`, element-wise
  `select` (or `where`) for Huber. Verify `where` exists; if not,
  build via `mask * a + (1-mask) * b`.

## Reusable modules

- Shipped FusedSoftmaxCrossEntropy fused op (FusedOpId 17). Use
  for the CE path.
- LazyTensor binary/unary primitives.

## Open questions

- Reduction enum: match the existing `fuel::loss::Reduction`
  shape (Mean / Sum / None) if it exists, otherwise use a small
  local enum.
- `binary_cross_entropy_with_logit` numerical stability: use the
  log-sum-exp variant from the eager file verbatim (don't
  re-derive).

## Splits

Single session. ~250 LOC mechanical port + 6-8 tests.

## Test strategy

- `cross_entropy_matches_eager_on_tiny`: hand-build a (2, 3)
  logits tensor + 2 class labels; assert the lazy cross_entropy
  output matches a hand-computed expected value within 1e-6.
- `nll_matches_eager_on_tiny`.
- `mse_zero_on_equal_inputs`, `mse_unit_on_unit_diff`.
- `bce_with_logit_matches_sigmoid_then_nll` (golden against the
  textbook formula).
- `huber_quadratic_under_delta`, `huber_linear_over_delta`.

## References

- Eager source: `fuel-nn/src/loss.rs`.
- Shipped FSCE: `lazy::FUSED_SOFTMAX_CROSS_ENTROPY` /
  `project_fused_softmax_cross_entropy_shipped` memory entry.

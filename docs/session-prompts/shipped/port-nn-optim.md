# Port: fuel-nn Optimizer trait + SGD + AdamW to lazy

## Eager source

- `fuel-nn/src/optim.rs` (~600 LOC). `Optimizer` trait,
  `SGD`, `AdamW`, `SGDConfig`, `AdamWConfig`. All operate on
  eager `Var` parameters and update them in-place.

## Lazy module name

`fuel-core/src/lazy_nn_optim.rs` (new file).

## Architecture summary

The eager `Optimizer` trait carries `&mut self`-style in-place
parameter updates over `Var`. The lazy port uses the
`Parameter` + `TrainState`-style on-device parameter storage
that `fuel-core/src/train.rs` already establishes, but
re-exposes the public Optimizer trait surface that nn consumers
expect.

Two concrete optimizers:

**SGD**: `param -= lr * grad`. With momentum: keep per-parameter
velocity buffer. With weight_decay: pre-add `weight_decay * param`
to grad.

**AdamW**: per-parameter (m, v) moment buffers; bias-corrected
moments; decoupled weight decay (subtracted from param BEFORE
the moment update). Uses epsilon for numerical stability in the
denominator.

Surface mirrors the eager trait:

```rust
pub trait LazyOptimizer {
    type Config;
    fn new(params: Vec<LazyVar>, cfg: Self::Config) -> Result<Self>;
    fn step(&mut self, grads: &HashMap<String, LazyTensor>) -> Result<()>;
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
    fn backward_step(&mut self, loss: &LazyTensor) -> Result<()>;
}
```

`LazyVar` is the lazy equivalent of eager `Var` — a wrapper
around a `LazyTensor` that's been promoted to a graph parameter
(autograd-tracked). Either reuse the existing `crate::Var` if
it now lives on the lazy side, or introduce a thin newtype.

## Primitives needed

- LazyTensor backward / GradMap — already shipped via
  `fuel_graph::backward` and the lazy training plumbing in
  `fuel-core/src/train.rs`.
- LazyTensor binary ops (add, sub, mul, mul_scalar, sqrt, div).
- Per-step in-place update vs new-buffer: for v1, allocate a new
  buffer each step (matches eager). The in-place update primitive
  is in port-training-augmentations.md and lands separately.

## Reusable modules

- `fuel-core/src/train.rs`: existing lazy AdamW + SGD impls live
  inside the file as internal training helpers. The optimizer
  step formulas can be lifted into this new module's public
  surface.
- `fuel-core/src/variable.rs` for `Var`-shape conventions.

## Open questions

- Does `LazyVar` already exist? Grep `fuel-core/src` for
  `LazyVar` and `Var`. If not, introduce as a thin wrapper.
- AdamW bias-correction step number: persisted in the optimizer
  or attached to the parameter? Eager has it on the optimizer
  per-parameter. Mirror.

## Splits

Two sub-ports if needed:
1. Sub-port 1: Trait + SGD + tests.
2. Sub-port 2: AdamW + tests.

Otherwise single session.

## Test strategy

- `sgd_zero_lr_does_not_change_params` (regression smoke).
- `sgd_unit_lr_unit_grad_subtracts_unit`.
- `sgd_with_momentum_accumulates_velocity` (two steps, check
  velocity tracking).
- `adamw_first_step_matches_textbook_formula` (hand-computed
  expected param after one step).
- `adamw_weight_decay_subtracts_before_update`.
- `backward_step_runs_loss_backward_then_step` (integration).

## References

- Eager source: `fuel-nn/src/optim.rs`.
- Lazy training reference: `fuel-core/src/train.rs`.
- Original Adam paper, AdamW paper.

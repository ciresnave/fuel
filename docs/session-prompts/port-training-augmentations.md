# Port: Training augmentations (grad accum, clipping, LR sched, mixed-precision)

## Eager source

Distributed across `fuel-nn/src/optim.rs`, `fuel-core/src/train.rs`
(explicit TODOs at top of file), plus any LR-scheduler helpers in
`fuel-nn`. No single eager file — this is the explicit pending-work
list called out at the top of `fuel-core/src/train.rs`:

```text
## Not yet covered
- In-place update primitive: today each step allocates a fresh buffer.
- Gradient accumulation across micro-batches.
- Mixed-precision (bf16 forward / fp32 master weights).
- Gradient clipping.
- LR schedulers.
```

## Lazy module name

`fuel-core/src/lazy_training_augmentations.rs` (new single file
covering scheduler + clipping + accumulation host-side; the
in-place primitive lands as a method on `TrainState` and the
mixed-precision policy as a config on `TrainState`).

## Architecture summary

Five independent additions:

1. **Gradient clipping** — pure tensor algebra on the GradMap:
   `clip_grad_norm(grads, max_norm, norm_type)`. Compute total
   gradient L2/Linf norm; if it exceeds `max_norm`, scale all
   gradients by `max_norm / total_norm`. Optional `clip_grad_value`
   variant for elementwise clamping.

2. **LR schedulers** — pure host-side scalar functions:
   - `CosineSchedule { base_lr, warmup_steps, total_steps }`
   - `LinearWarmupSchedule { base_lr, warmup_steps }`
   - `PolynomialSchedule { base_lr, total_steps, power }`
   - `StepSchedule { base_lr, milestones, gamma }`
   Each implements `LrSchedule::lr_at(step: usize) -> f64`.

3. **Gradient accumulation** — pure host-side counter + GradMap
   accumulator. `GradAccumulator { accum: HashMap<String, LazyTensor>,
   steps: usize, microbatches: usize }`. After N microbatch
   forward+backwards, scale accumulated grads by 1/N and apply via
   optimizer.

4. **Mixed-precision policy** — adds `MixedPrecisionConfig
   { forward_dtype: DType::BF16, master_dtype: DType::F32 }` to
   `TrainState`. Forward runs in bf16; gradients downcast to bf16
   for in-graph efficiency; master weights stay f32. Implementation
   touches `TrainState::step` to cast params before injecting and
   cast grads on return.

5. **In-place parameter update primitive** — `TrainState`
   gets a `step_in_place` variant that uses an in-place add op on
   the parameter storage instead of allocating a fresh buffer. May
   require a new graph op `Op::AddInPlace` or rely on the existing
   in-place ops shipped 2026-05-30.

## Primitives needed

- All shipped except possibly an `add_in_place` primitive used by
  item 5. Check whether the shipped in-place ops cover the
  parameter-update use case.
- LazyTensor `norm`, `clamp`, `mul_scalar` for clipping.

## Reusable modules

- `fuel-core/src/train.rs` — TrainState and Parameter type.
- The shipped in-place op infrastructure
  (`project_inplace_ops_complete` memory entry).

## Splits

Each augmentation is independent. Recommended ship order:

1. Sub-port 1: LR schedulers (pure host-side, smallest).
2. Sub-port 2: Gradient clipping.
3. Sub-port 3: Gradient accumulation.
4. Sub-port 4: Mixed-precision config + TrainState casts.
5. Sub-port 5: In-place parameter update.

Each is its own commit.

## Test strategy

- LR schedules: golden lr values at a few steps for each shape
  (cosine should hit max at warmup, zero at total).
- Gradient clipping: synthetic GradMap with known total norm 10;
  clip to 5; assert all gradients scaled by 0.5.
- Gradient accumulation: two microbatch passes with grads
  [1.0, 2.0] each; assert accumulated grad after divide = 1.5.
- Mixed precision: tiny TrainState step with bf16 forward / f32
  master; assert final param values match an f32-only baseline
  within bf16 tolerance.
- In-place: TrainState.step_in_place produces the same param
  values as TrainState.step.

## References

- Eager source: `fuel-core/src/train.rs` top-of-file TODOs.
- AdamW paper for mixed-precision recipe.

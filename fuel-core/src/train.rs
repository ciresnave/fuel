//! Training utilities: parameters, optimizers, and the training-step
//! driver. All generic over `GraphBackend` — SGD and AdamW work on
//! CPU, CUDA, Vulkan, and any future backend with zero code changes.
//!
//! ## Design
//!
//! Parameters live on-device as `B::Storage` between steps, exactly
//! like the KV cache does for inference. Each training step:
//!
//! 1. User callback builds a fresh forward graph, referring to each
//!    parameter by name.
//! 2. `TrainState` injects each parameter's current device storage
//!    via `executor.pre_populate` (same mechanism the KV cache uses).
//! 3. The step driver calls `loss.backward()` to extend the graph
//!    with backward nodes, fetches `grad` per parameter via the
//!    returned `GradMap`, and appends update ops (`w_new = w - lr·g`
//!    for SGD; AdamW stacks moment-update and bias-correction ops
//!    on top).
//! 4. `realize_split` keeps the new parameter storage on-device;
//!    only the loss scalar is downloaded to host.
//! 5. `TrainState` stores the new storage, ready for the next step.
//!
//! No D2H/H2D per parameter per step. No new backend trait methods
//! — everything is expressed as existing primitives the backend
//! already supports.
//!
//! ## Not yet covered
//!
//! - In-place update primitive: today each step allocates a fresh
//!   buffer for each updated parameter. For TinyLlama-scale that's
//!   fine; for 70B-class training it's wasteful. Addressed later by
//!   a trait-level `add_in_place` method.
//! - Gradient accumulation across micro-batches.
//! - Mixed-precision (bf16 forward / fp32 master weights).
//! - Gradient clipping.
//! - LR schedulers.
//!
//! Each of those is a small addition once the base loop works.

use crate::lazy::LazyTensor;
use fuel_core_types::{DType, HostBuffer, Result, Shape};
use fuel_graph::{NodeId, SharedGraph, Tensor as GraphTensor};
use fuel_graph_executor::{GraphBackend, GraphExecutor};
use std::collections::HashMap;
use std::sync::Arc;

/// A single trainable parameter. Owns its shape and initial values;
/// actual device storage lives in `TrainState` after the first step.
#[derive(Clone)]
pub struct Parameter {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    /// Initial values, uploaded on `TrainState::new`. After that,
    /// parameter values live exclusively on-device in `TrainState`.
    pub initial_data: Arc<[f32]>,
}

impl Parameter {
    pub fn new_f32(name: impl Into<String>, shape: impl Into<Shape>, data: impl Into<Arc<[f32]>>) -> Self {
        let shape = shape.into();
        let data = data.into();
        assert_eq!(
            data.len(),
            shape.elem_count(),
            "Parameter::new_f32: data length must match shape elem_count",
        );
        Self {
            name: name.into(),
            shape,
            dtype: DType::F32,
            initial_data: data,
        }
    }
}

/// Optimizer configuration. Add new variants here, then match them
/// in `TrainState::append_update_ops`.
#[derive(Debug, Clone, Copy)]
pub enum OptimizerConfig {
    /// Plain stochastic gradient descent: `w ← w − lr·g`.
    Sgd { lr: f32 },
    /// AdamW: `m ← β₁·m + (1−β₁)·g`, `v ← β₂·v + (1−β₂)·g²`,
    /// `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`,
    /// `w ← w − lr·(m̂/(√v̂ + ε) + weight_decay·w)`.
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
}

impl OptimizerConfig {
    pub fn sgd(lr: f32) -> Self { Self::Sgd { lr } }

    pub fn adam_w(lr: f32) -> Self {
        Self::AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    /// Return the learning rate stored in this config.
    pub fn lr(&self) -> f32 {
        match *self {
            Self::Sgd { lr } => lr,
            Self::AdamW { lr, .. } => lr,
        }
    }

    /// Return a copy of `self` with `lr` replaced.
    pub fn with_lr(&self, new_lr: f32) -> Self {
        match *self {
            Self::Sgd { .. } => Self::Sgd { lr: new_lr },
            Self::AdamW { beta1, beta2, eps, weight_decay, .. } => Self::AdamW {
                lr: new_lr, beta1, beta2, eps, weight_decay,
            },
        }
    }
}

/// Learning-rate schedule. Any closure-like thing can be one, but
/// the provided impls cover the common cases: constant, linear
/// warmup then cosine decay, and linear warmup then linear decay.
pub trait LrSchedule {
    /// LR to use for the step index `step` (0-indexed).
    fn lr_at(&self, step: u64) -> f32;
}

/// Constant LR — the default. Identical to passing a fixed LR to
/// the optimizer config.
pub struct ConstLr(pub f32);
impl LrSchedule for ConstLr {
    fn lr_at(&self, _step: u64) -> f32 { self.0 }
}

/// Linear warmup for `warmup` steps (LR ramps 0 → `peak`), then
/// cosine decay from `peak` to `final_lr` over the remaining
/// `total - warmup` steps.
pub struct WarmupCosine {
    pub warmup: u64,
    pub total: u64,
    pub peak: f32,
    pub final_lr: f32,
}
impl LrSchedule for WarmupCosine {
    fn lr_at(&self, step: u64) -> f32 {
        if step < self.warmup {
            // Linear ramp 0 → peak across warmup steps.
            if self.warmup == 0 { return self.peak; }
            self.peak * (step as f32 / self.warmup as f32)
        } else if step >= self.total {
            self.final_lr
        } else {
            let progress = (step - self.warmup) as f32
                / (self.total - self.warmup).max(1) as f32;
            // Cosine from peak to final_lr.
            let cos_scale = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.final_lr + (self.peak - self.final_lr) * cos_scale
        }
    }
}

/// Linear warmup then linear decay. Simpler than cosine but often
/// close in practice; and cheaper to eyeball in a log.
pub struct WarmupLinear {
    pub warmup: u64,
    pub total: u64,
    pub peak: f32,
    pub final_lr: f32,
}
impl LrSchedule for WarmupLinear {
    fn lr_at(&self, step: u64) -> f32 {
        if step < self.warmup {
            if self.warmup == 0 { return self.peak; }
            self.peak * (step as f32 / self.warmup as f32)
        } else if step >= self.total {
            self.final_lr
        } else {
            let progress = (step - self.warmup) as f32
                / (self.total - self.warmup).max(1) as f32;
            self.peak + (self.final_lr - self.peak) * progress
        }
    }
}

/// Per-parameter optimizer state that lives on-device between steps.
/// SGD needs nothing; AdamW needs first and second moments.
pub(crate) enum OptState<B: GraphBackend> {
    Sgd,
    AdamW {
        m: B::Storage,
        v: B::Storage,
    },
}

/// Gradient clipping strategy applied per step before the optimizer
/// update runs. Currently a single knob: `GlobalNorm(max_norm)`
/// — the canonical LLM-training clip. More strategies (per-param
/// norm, by-value clipping) can land here later.
#[derive(Debug, Clone, Copy)]
pub enum GradClip {
    /// Clip the concatenated-all-gradients L2 norm to at most `max_norm`.
    /// If `‖grad‖₂ ≤ max_norm`, gradients are untouched; otherwise
    /// every gradient is scaled by `max_norm / ‖grad‖₂`.
    GlobalNorm(f32),
}

/// Training state: all parameters and optimizer state that must
/// persist across steps. Generic over backend.
pub struct TrainState<B: GraphBackend> {
    params: HashMap<String, (B::Storage, Shape)>,
    opt_state: HashMap<String, OptState<B>>,
    param_order: Vec<String>,
    config: OptimizerConfig,
    step_count: u64,
    grad_clip: Option<GradClip>,
}

impl<B: GraphBackend> TrainState<B> {
    /// Upload all parameters to the device and initialize optimizer
    /// state. Call once at the start of training.
    pub fn new(
        parameters: &[Parameter],
        executor: &mut GraphExecutor<B>,
        config: OptimizerConfig,
    ) -> Result<Self> {
        let mut params = HashMap::new();
        let mut opt_state = HashMap::new();
        let mut param_order = Vec::with_capacity(parameters.len());

        for p in parameters {
            let buf = HostBuffer::F32(p.initial_data.to_vec());
            let storage = executor.backend.upload(&buf, &p.shape)?;
            params.insert(p.name.clone(), (storage, p.shape.clone()));
            param_order.push(p.name.clone());

            let st = match config {
                OptimizerConfig::Sgd { .. } => OptState::Sgd,
                OptimizerConfig::AdamW { .. } => {
                    // m and v start at zero, same shape and dtype as the param.
                    let m = executor.backend.alloc_zeros(&p.shape, p.dtype)?;
                    let v = executor.backend.alloc_zeros(&p.shape, p.dtype)?;
                    OptState::AdamW { m, v }
                }
            };
            opt_state.insert(p.name.clone(), st);
        }

        Ok(Self {
            params,
            opt_state,
            param_order,
            config,
            step_count: 0,
            grad_clip: None,
        })
    }

    /// Enable gradient clipping. Applied per step before the
    /// optimizer update. Pass `None` to disable.
    pub fn with_grad_clip(mut self, clip: Option<GradClip>) -> Self {
        self.grad_clip = clip;
        self
    }

    pub fn set_grad_clip(&mut self, clip: Option<GradClip>) {
        self.grad_clip = clip;
    }

    pub fn step_count(&self) -> u64 { self.step_count }

    /// Replace the current learning rate on the optimizer config.
    /// Takes effect on the next `step`. Useful for manual LR control
    /// or when driving via an external scheduler callback.
    pub fn set_lr(&mut self, lr: f32) {
        self.config = self.config.with_lr(lr);
    }

    /// Convenience: pull the next LR from `schedule` using the
    /// current step counter, set it, and invoke `step`. Equivalent
    /// to `state.set_lr(schedule.lr_at(state.step_count())); state.step(...)`.
    pub fn step_with_schedule<S, F>(
        &mut self,
        schedule: &S,
        executor: &mut GraphExecutor<B>,
        build_loss: F,
    ) -> Result<f32>
    where
        S: LrSchedule,
        F: FnOnce(&SharedGraph, &HashMap<String, LazyTensor>) -> LazyTensor,
    {
        self.set_lr(schedule.lr_at(self.step_count));
        self.step(executor, build_loss)
    }

    /// Read-only snapshot of the current optimizer config.
    pub fn optimizer_config(&self) -> OptimizerConfig { self.config }

    /// Download a parameter's current value to host. For checkpoint /
    /// inspection; not a hot-path operation.
    pub fn param_to_host(&self, name: &str, executor: &GraphExecutor<B>) -> Result<Vec<f32>> {
        let (storage, _shape) = self.params.get(name)
            .ok_or_else(|| fuel_core_types::Error::Msg(
                format!("unknown parameter '{name}'")))?;
        let buf = executor.backend.download(storage)?;
        match buf {
            HostBuffer::F32(v) => Ok(v),
            other => Err(fuel_core_types::Error::Msg(
                format!("param_to_host: expected F32, got {:?}", other.dtype()))),
        }
    }

    /// Run one training step.
    ///
    /// `build_loss` receives a graph handle and a map of parameter
    /// LazyTensor leaves (one per parameter, placeholder Const nodes
    /// that the step driver injects real device storage for). It
    /// must return a scalar loss tensor.
    ///
    /// Returns the loss value (a single f32).
    pub fn step<F>(
        &mut self,
        executor: &mut GraphExecutor<B>,
        build_loss: F,
    ) -> Result<f32>
    where
        F: FnOnce(&SharedGraph, &HashMap<String, LazyTensor>) -> LazyTensor,
    {
        // 1. Build parameter placeholder tensors in a fresh graph.
        //    Use a "seed" LazyTensor to get a fresh SharedGraph.
        let seed = LazyTensor::from_f32(vec![0.0f32], Shape::from_dims(&[1]), &crate::Device::cpu());
        let graph = seed.graph_tensor().graph().clone();

        let mut param_tensors: HashMap<String, LazyTensor> = HashMap::new();
        let mut param_node_ids: HashMap<String, NodeId> = HashMap::new();
        for name in &self.param_order {
            let (_storage, shape) = &self.params[name];
            // Zero-filled placeholder Const; pre_populate overrides at realize.
            let zeros: Arc<[f32]> = vec![0.0_f32; shape.elem_count()].into();
            let t = seed.const_f32_like(zeros, shape.clone());
            param_node_ids.insert(name.clone(), t.graph_tensor().id());
            param_tensors.insert(name.clone(), t);
        }

        // 2. User builds the loss graph referring to parameters by name.
        let loss = build_loss(&graph, &param_tensors);

        // 3. Backward to get gradient nodes per parameter.
        let grad_map = loss.graph_tensor().backward();

        // 3a. Collect raw gradients.
        let mut raw_grads: HashMap<String, LazyTensor> = HashMap::new();
        for name in &self.param_order {
            let param = &param_tensors[name];
            let grad = grad_map.get(param.graph_tensor())
                .ok_or_else(|| fuel_core_types::Error::Msg(
                    format!("parameter '{name}' did not appear in loss graph")))?;
            raw_grads.insert(name.clone(), LazyTensor::from_graph_tensor(grad));
        }

        // 3b. If clipping is enabled, compute the global-norm scale
        //     factor as a scalar graph node and use it to scale every
        //     gradient before the update.
        //
        //   scale = clamp(max_norm / norm, 0, 1)
        //
        // Scaling is a no-op when norm ≤ max_norm (scale=1) and
        // active otherwise; no branch in the graph, just one scalar
        // `clamp` — which works identically on every backend.
        let clip_scale: Option<LazyTensor> = match self.grad_clip {
            None => None,
            Some(GradClip::GlobalNorm(max_norm)) => {
                let mut total_sq: Option<LazyTensor> = None;
                for name in &self.param_order {
                    let g = &raw_grads[name];
                    let g_sq_sum = g.sqr().sum_all();
                    total_sq = Some(match total_sq {
                        None => g_sq_sum,
                        Some(acc) => acc.add(&g_sq_sum),
                    });
                }
                let total_sq = total_sq.expect("clip: no gradients");
                let norm = total_sq.sqrt();
                // `max_norm / norm` is a rank-0 scalar. Protect
                // against norm=0 by adding a tiny epsilon.
                let norm_safe = norm.add_scalar(1e-12);
                let ratio = norm_safe.mul_scalar(1.0 / max_norm as f64);
                // ratio = norm/max_norm. We want scale = min(1, 1/ratio).
                // Equivalently: scale = clamp(1/ratio, 0, 1).
                let inv_ratio = ratio.const_f32_like(vec![1.0f32], Shape::from_dims(&[]))
                    .div(&ratio);
                let scale = inv_ratio.clamp(0.0, 1.0);
                Some(scale)
            }
        };

        // 4. Build update ops per parameter.
        //    Returns the new_param LazyTensor plus any new opt-state
        //    LazyTensors (Adam's new m and v).
        let mut new_param_tensors: Vec<LazyTensor> = Vec::with_capacity(self.param_order.len());
        let mut new_opt_tensors: Vec<(String, LazyTensor, LazyTensor)> = Vec::new();

        for name in &self.param_order {
            let param = &param_tensors[name];
            let raw_grad = &raw_grads[name];
            let grad_lt = match &clip_scale {
                None => raw_grad.clone(),
                Some(scale) => {
                    // Broadcast the rank-0 scalar to grad's shape and multiply.
                    let grad_shape = raw_grad.graph_tensor().shape();
                    let scale_bcast = scale.broadcast_to(grad_shape).unwrap();
                    raw_grad.mul(&scale_bcast)
                }
            };

            match self.config {
                OptimizerConfig::Sgd { lr } => {
                    // new = param - lr * grad
                    let scaled = grad_lt.mul_scalar(lr as f64);
                    let new_param = param.sub(&scaled);
                    new_param_tensors.push(new_param);
                }
                OptimizerConfig::AdamW { lr, beta1, beta2, eps, weight_decay } => {
                    // Build placeholders for m and v (injected via pre_populate).
                    let zeros: Arc<[f32]> = vec![0.0_f32; param.graph_tensor().shape().elem_count()].into();
                    let m_placeholder = seed.const_f32_like(zeros.clone(), param.graph_tensor().shape());
                    let v_placeholder = seed.const_f32_like(zeros, param.graph_tensor().shape());
                    // new_m = β1·m + (1-β1)·g
                    let m_decayed = m_placeholder.mul_scalar(beta1 as f64);
                    let g_part = grad_lt.mul_scalar((1.0 - beta1) as f64);
                    let new_m = m_decayed.add(&g_part);
                    // new_v = β2·v + (1-β2)·g²
                    let v_decayed = v_placeholder.mul_scalar(beta2 as f64);
                    let g_sq = grad_lt.sqr();
                    let g_sq_part = g_sq.mul_scalar((1.0 - beta2) as f64);
                    let new_v = v_decayed.add(&g_sq_part);
                    // Bias correction using step+1 (this is the step we're ABOUT to complete).
                    let t = (self.step_count + 1) as f64;
                    let bc1 = 1.0 - (beta1 as f64).powf(t);
                    let bc2 = 1.0 - (beta2 as f64).powf(t);
                    let m_hat = new_m.mul_scalar(1.0 / bc1);
                    let v_hat = new_v.mul_scalar(1.0 / bc2);
                    // update = m_hat / (sqrt(v_hat) + eps)
                    let denom = v_hat.sqrt().add_scalar(eps as f64);
                    let update = m_hat.div(&denom);
                    // Apply weight decay and lr.
                    let wd_term = param.mul_scalar(weight_decay as f64);
                    let step_total = update.add(&wd_term).mul_scalar(lr as f64);
                    let new_param = param.sub(&step_total);
                    new_param_tensors.push(new_param);
                    new_opt_tensors.push((name.clone(), new_m, new_v));

                    // Also record the m/v placeholder NodeIds for pre_populate.
                    param_node_ids.insert(format!("{name}::m"), m_placeholder.graph_tensor().id());
                    param_node_ids.insert(format!("{name}::v"), v_placeholder.graph_tensor().id());
                }
            }
        }

        // 5. Inject current parameter and opt-state storage for placeholder nodes.
        for name in &self.param_order {
            let (storage, shape) = &self.params[name];
            let layout = fuel_core_types::Layout::contiguous(shape);
            let cloned = executor.backend.try_clone(storage, &layout)?;
            executor.pre_populate(param_node_ids[name], cloned, shape.clone());

            if let OptState::AdamW { m, v } = &self.opt_state[name] {
                let m_clone = executor.backend.try_clone(m, &layout)?;
                let v_clone = executor.backend.try_clone(v, &layout)?;
                executor.pre_populate(param_node_ids[&format!("{name}::m")], m_clone, shape.clone());
                executor.pre_populate(param_node_ids[&format!("{name}::v")], v_clone, shape.clone());
            }
        }

        // 6. Build realize root list: [loss, new_params..., (new_m, new_v)...]
        let mut roots: Vec<&LazyTensor> = Vec::new();
        roots.push(&loss);
        for np in &new_param_tensors { roots.push(np); }
        for (_, nm, nv) in &new_opt_tensors {
            roots.push(nm);
            roots.push(nv);
        }

        // 7. Realize: loss → CPU, everything else stays on device.
        let inner: Vec<&GraphTensor> = roots.iter().map(|lt| lt.graph_tensor()).collect();
        let (cpu_results, gpu_results) = executor.realize_split(&inner, 1);
        let loss_vec = cpu_results.into_iter().next().unwrap();
        let loss_scalar = if loss_vec.len() == 1 {
            loss_vec[0]
        } else {
            // loss wasn't rank-0; sum it as a scalar for returning.
            // (Users usually build a scalar loss anyway.)
            loss_vec.iter().sum()
        };

        // 8. Move new storage back into TrainState.
        let mut iter = gpu_results.into_iter();
        for name in &self.param_order {
            let (new_storage, new_shape) = iter.next().unwrap();
            self.params.insert(name.clone(), (new_storage, new_shape));
        }
        for (name, _, _) in &new_opt_tensors {
            let (new_m, _) = iter.next().unwrap();
            let (new_v, _) = iter.next().unwrap();
            self.opt_state.insert(name.clone(), OptState::AdamW { m: new_m, v: new_v });
        }

        self.step_count += 1;
        Ok(loss_scalar)
    }
}

/// Reusable loss functions. All pure LazyTensor graph constructors —
/// every backend runs them via the primitives it already supports.
pub mod loss {
    use crate::lazy::LazyTensor;
    use fuel_core_types::Shape;

    /// Mean-squared-error loss: `mean((pred - target)²)`. Returns a
    /// scalar tensor. Works on any numeric shape — the mean is
    /// over all elements.
    pub fn mse(pred: &LazyTensor, target: &LazyTensor) -> LazyTensor {
        let n = pred.graph_tensor().shape().elem_count();
        let diff = pred.sub(target);
        let sq = diff.sqr();
        sq.sum_all().mul_scalar(1.0 / n as f64)
    }

    /// Cross-entropy loss with integer class targets and raw logits.
    ///
    /// `logits` shape: `[..., C]` where C is the number of classes.
    /// `targets` is currently built into the graph as a one-hot tensor
    /// the caller supplies via `target_one_hot` — integer-target
    /// gather-based cross-entropy requires `index_select` and backward
    /// through it, which is supported but rebuilds the softmax output
    /// in backward (slow). The one-hot variant is the common LLM
    /// training pattern: targets are pre-encoded to [..., C] floats
    /// with 1.0 at the correct class and 0.0 elsewhere.
    ///
    /// Formula: `-mean(sum(target · log_softmax(logits), last))` where
    /// `log_softmax(x)_i = x_i - logsumexp(x)`.
    ///
    /// Stable computation: `logsumexp(x) = max + log(sum(exp(x-max)))`.
    /// This avoids `exp(x)` overflow for large logits.
    pub fn cross_entropy_with_logits(
        logits: &LazyTensor,
        target_one_hot: &LazyTensor,
    ) -> LazyTensor {
        let dims = logits.graph_tensor().shape().dims().to_vec();
        let rank = dims.len();
        assert!(rank >= 1, "cross_entropy_with_logits: logits must have rank >= 1");
        let n_outer: usize = dims[..rank - 1].iter().product::<usize>().max(1);

        // Stable log-softmax along last dim:
        //   max_r = max(logits, last)             shape [..., 1]
        //   shifted = logits - max_r              shape [..., C]
        //   log_sum = log(sum_dim(exp(shifted), last))  shape [..., 1]
        //   log_softmax = shifted - log_sum       shape [..., C]
        let max_r = logits.max_dim(rank - 1);
        let mut keepdim = dims.clone();
        keepdim[rank - 1] = 1;
        let max_kd = max_r.reshape(Shape::from_dims(&keepdim)).unwrap();
        let max_bcast = max_kd.broadcast_to(Shape::from_dims(&dims)).unwrap();
        let shifted = logits.sub(&max_bcast);
        let expd = shifted.exp();
        let sum_exp = expd.sum_dim(rank - 1);
        let log_sum = sum_exp.log();
        let log_sum_kd = log_sum.reshape(Shape::from_dims(&keepdim)).unwrap();
        let log_sum_bcast = log_sum_kd.broadcast_to(Shape::from_dims(&dims)).unwrap();
        let log_softmax = shifted.sub(&log_sum_bcast);

        // -sum(target * log_softmax, last-dim) summed over batch, then mean.
        let per_elem = target_one_hot.mul(&log_softmax);
        // Loss per-sample = -sum over class dim. Then mean over the
        // outer (batch) dims for a scalar loss.
        let neg_per_sample = per_elem.sum_dim(rank - 1).mul_scalar(-1.0);
        let total = neg_per_sample.sum_all();
        total.mul_scalar(1.0 / n_outer as f64)
    }

    /// Fused softmax + negative log-likelihood with integer (class-
    /// index) targets — the standard PyTorch / Liger-Kernel training
    /// loss. The fused CPU forward kernel skips the `[..., V]`
    /// softmax / log-softmax intermediates that
    /// [`cross_entropy_with_logits`] materializes; on Llama-7B this
    /// saves ~12 GiB of transient allocation per forward call.
    ///
    /// Arguments:
    /// - `logits`: `[..., V]` F32.
    /// - `targets`: `[...]` I64 class indices (matches PyTorch's
    ///   `CrossEntropyLoss(target: int64)` convention).
    /// - `reduction`: `Mean` / `Sum` produce a scalar; `None` produces
    ///   per-row losses of `targets.shape`.
    /// - `ignore_index`: rows whose target equals this sentinel
    ///   contribute 0 to the loss and 0 to the Mean denominator. The
    ///   conventional value is `-100`.
    ///
    /// Backward via the registry's `Decompose` lowering — runs the same
    /// primitive log-softmax + gather chain that
    /// [`cross_entropy_with_logits`] uses, so peak memory during
    /// `loss.backward()` matches the primitive composition path (only
    /// the forward saves memory). Datasets that need `ignore_index`-
    /// aware backward should stay on [`cross_entropy_with_logits`]
    /// until the in-place Liger fused backward lands; this one's
    /// lowered backward computes the gradient as if no rows are
    /// masked.
    pub fn fused_softmax_cross_entropy(
        logits: &LazyTensor,
        targets: &LazyTensor,
        reduction: fuel_graph::registry::Reduction,
        ignore_index: i64,
    ) -> LazyTensor {
        logits.fused_softmax_cross_entropy(targets, reduction, ignore_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LazyTensor;
    use fuel_core_types::DType;
    use fuel_graph::registry::Reduction;
    use fuel_graph_cpu::CpuBackend;

    /// Helper: build an I64 LazyTensor on the same graph as `host`,
    /// using the fuel-graph `const_i64_like` builder. LazyTensor has
    /// no native `from_i64` constructor today; the index-only ops
    /// (Gather, etc.) typically wire indices in via `const_u32_like`,
    /// so we go through `from_graph_tensor` here.
    fn lt_const_i64_like(host: &LazyTensor, data: Vec<i64>, shape: Shape) -> LazyTensor {
        LazyTensor::from_graph_tensor(host.graph_tensor().const_i64_like(data, shape))
    }

    /// FusedSoftmaxCrossEntropy matches the primitive
    /// cross_entropy_with_logits chain (one-hot form) for a small
    /// fully-defined batch. Verifies both that the registry+dispatch
    /// path produces the right scalar and that the fused kernel is
    /// numerically consistent with the textbook composition.
    #[test]
    fn fused_softmax_cross_entropy_matches_primitive_composition() {
        let device = crate::Device::cpu();
        // 3 rows, vocab 4. Targets chosen so each row exercises a
        // different argmax position.
        let logits_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            0.5, 0.0, -0.5, 1.0,
            -1.0, -2.0, -3.0, 4.0,
        ];
        let targets_i64: Vec<i64> = vec![3, 0, 3];
        let mut targets_onehot = vec![0.0f32; 3 * 4];
        for (row, &t) in targets_i64.iter().enumerate() {
            targets_onehot[row * 4 + t as usize] = 1.0;
        }

        // Path 1: fused op.
        let logits_fused = LazyTensor::from_f32(
            logits_data.clone(), Shape::from_dims(&[3, 4]), &device,
        );
        let targets_fused = lt_const_i64_like(
            &logits_fused, targets_i64.clone(), Shape::from_dims(&[3]),
        );
        let fused_loss = loss::fused_softmax_cross_entropy(
            &logits_fused, &targets_fused, Reduction::Mean, -100,
        )
        .realize_f32()[0];

        // Path 2: primitive composition. The one-hot targets must
        // live on the same graph as logits — use `const_f32_like`
        // off `logits_prim` so the second leaf joins that graph.
        let logits_prim = LazyTensor::from_f32(
            logits_data, Shape::from_dims(&[3, 4]), &device,
        );
        let targets_prim = logits_prim.const_f32_like(
            targets_onehot, Shape::from_dims(&[3, 4]),
        );
        let prim_loss = loss::cross_entropy_with_logits(&logits_prim, &targets_prim)
            .realize_f32()[0];

        assert!(
            (fused_loss - prim_loss).abs() < 1e-5,
            "fused {fused_loss} vs primitive {prim_loss}",
        );
    }

    /// Reduction::None returns per-row losses with shape == targets.shape.
    #[test]
    fn fused_softmax_cross_entropy_none_returns_per_row() {
        let device = crate::Device::cpu();
        let logits_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let logits = LazyTensor::from_f32(logits_data, Shape::from_dims(&[2, 4]), &device);
        let targets = lt_const_i64_like(&logits, vec![1_i64, 3], Shape::from_dims(&[2]));
        let per_row = loss::fused_softmax_cross_entropy(
            &logits, &targets, Reduction::None, -100,
        )
        .realize_f32();
        assert_eq!(per_row.len(), 2);
        // Hand-computed (see byte_kernels.rs unit test).
        assert!((per_row[0] - 2.44018972).abs() < 1e-5, "row 0: {}", per_row[0]);
        assert!((per_row[1] - 1.38629436).abs() < 1e-5, "row 1: {}", per_row[1]);
    }

    /// ignore_index drops a row from both the loss sum and the mean
    /// denominator. With one of two rows masked, Mean equals the
    /// remaining row's loss exactly.
    #[test]
    fn fused_softmax_cross_entropy_ignore_index_masks_row() {
        let device = crate::Device::cpu();
        let logits_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let logits = LazyTensor::from_f32(logits_data, Shape::from_dims(&[2, 4]), &device);
        let targets = lt_const_i64_like(&logits, vec![1_i64, -100], Shape::from_dims(&[2]));
        let loss_val = loss::fused_softmax_cross_entropy(
            &logits, &targets, Reduction::Mean, -100,
        )
        .realize_f32()[0];
        // Only row 0 contributes; mean of one value is itself.
        assert!(
            (loss_val - 2.44018972).abs() < 1e-5,
            "got {loss_val}, expected ~2.44019",
        );
    }

    /// CausalConv1d end-to-end: build a tiny depthwise-conv graph,
    /// realize through the dispatcher, verify the hand-computed
    /// output. Mirrors the byte-kernel `causal_conv1d_f32_no_silu_basic`
    /// test but goes through Tensor::causal_conv1d + binding-table
    /// dispatch instead of calling the kernel directly.
    #[test]
    fn causal_conv1d_basic_end_to_end() {
        let device = crate::Device::cpu();
        // x = [0, 0, 1, 2] (single batch, single channel, pre-padded)
        let x = LazyTensor::from_f32(
            vec![0.0_f32, 0.0, 1.0, 2.0],
            Shape::from_dims(&[1, 1, 4]),
            &device,
        );
        // weight = [0.5, 1.0, 2.0] for one channel, kernel 3
        let w = x.const_f32_like(
            vec![0.5_f32, 1.0, 2.0],
            Shape::from_dims(&[1, 1, 3]),
        );
        let bias = x.const_f32_like(vec![0.1_f32], Shape::from_dims(&[1]));
        let out = x.causal_conv1d(&w, &bias, false).realize_f32();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 2.1).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 5.1).abs() < 1e-5, "out[1]={}", out[1]);
    }

    /// CausalConv1d with use_silu = true through the dispatcher.
    #[test]
    fn causal_conv1d_with_silu_end_to_end() {
        let device = crate::Device::cpu();
        let x = LazyTensor::from_f32(
            vec![0.0_f32, 0.0, 1.0, 2.0],
            Shape::from_dims(&[1, 1, 4]),
            &device,
        );
        let w = x.const_f32_like(
            vec![0.5_f32, 1.0, 2.0],
            Shape::from_dims(&[1, 1, 3]),
        );
        let bias = x.const_f32_like(vec![0.1_f32], Shape::from_dims(&[1]));
        let out = x.causal_conv1d(&w, &bias, true).realize_f32();
        let expected0 = 2.1_f32 / (1.0 + (-2.1_f32).exp());
        let expected1 = 5.1_f32 / (1.0 + (-5.1_f32).exp());
        assert!((out[0] - expected0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - expected1).abs() < 1e-5, "out[1]={}", out[1]);
    }

    /// Registry: the CausalConv1d entry is wired into the default
    /// registry and reachable by id + name.
    #[test]
    fn causal_conv1d_registry_entry_registered() {
        let r = fuel_graph::registry::default_registry();
        let e = r
            .entry(fuel_graph::registry::FusedOps::CAUSAL_CONV1D)
            .expect("CAUSAL_CONV1D registered");
        assert_eq!(e.name, "CausalConv1d");
        assert_eq!(
            r.id_for_name("CausalConv1d"),
            Some(fuel_graph::registry::FusedOps::CAUSAL_CONV1D),
        );
        // Shape rule produces [batch, channels, seq_in - (kernel - 1)].
        let out_shape = (e.shape_rule)(
            &[
                Shape::from_dims(&[2, 4, 8]),  // x: batch=2, channels=4, seq_in=8
                Shape::from_dims(&[4, 1, 3]),  // weight: channels=4, 1, kernel=3
                Shape::from_dims(&[4]),        // bias: channels=4
            ],
            &fuel_graph::registry::FusedOpParams::CausalConv1d { use_silu: false },
        );
        assert_eq!(out_shape.dims(), &[2, 4, 6]);
    }

    /// SelectiveScan end-to-end: minimal seqlen=1 case via the
    /// dispatcher; verifies the full plumbing from builder → Op::Fused
    /// → op_to_op_kind/op_to_op_params → wrapper → kernel.
    #[test]
    fn selective_scan_basic_end_to_end() {
        let device = crate::Device::cpu();
        // batch=1, seqlen=1, dim=1, dstate=1. Same numbers as the
        // byte-kernel single-step test: expected y = 3.0.
        let u = LazyTensor::from_f32(vec![3.0_f32], Shape::from_dims(&[1, 1, 1]), &device);
        let delta = u.const_f32_like(vec![1.0_f32], Shape::from_dims(&[1, 1, 1]));
        let a = u.const_f32_like(vec![-1.0_f32], Shape::from_dims(&[1, 1]));
        let b = u.const_f32_like(vec![2.0_f32], Shape::from_dims(&[1, 1, 1]));
        let c = u.const_f32_like(vec![0.5_f32], Shape::from_dims(&[1, 1, 1]));
        let y = u.selective_scan(&delta, &a, &b, &c, false).realize_f32();
        assert_eq!(y.len(), 1);
        assert!((y[0] - 3.0).abs() < 1e-5, "got {}", y[0]);
    }

    /// SelectiveScan with delta_softplus through the dispatcher.
    #[test]
    fn selective_scan_with_softplus_end_to_end() {
        let device = crate::Device::cpu();
        let u = LazyTensor::from_f32(vec![1.0_f32], Shape::from_dims(&[1, 1, 1]), &device);
        let delta = u.const_f32_like(vec![0.0_f32], Shape::from_dims(&[1, 1, 1]));
        let a = u.const_f32_like(vec![0.0_f32], Shape::from_dims(&[1, 1]));
        let b = u.const_f32_like(vec![1.0_f32], Shape::from_dims(&[1, 1, 1]));
        let c = u.const_f32_like(vec![1.0_f32], Shape::from_dims(&[1, 1, 1]));
        let y = u.selective_scan(&delta, &a, &b, &c, true).realize_f32();
        let expected = 2.0_f32.ln();
        assert!((y[0] - expected).abs() < 1e-5, "got {} expected {expected}", y[0]);
    }

    /// Registry: the SelectiveScan entry is wired into the default
    /// registry and reachable by id + name.
    #[test]
    fn selective_scan_registry_entry_registered() {
        let r = fuel_graph::registry::default_registry();
        let e = r
            .entry(fuel_graph::registry::FusedOps::SELECTIVE_SCAN)
            .expect("SELECTIVE_SCAN registered");
        assert_eq!(e.name, "SelectiveScan");
        assert_eq!(
            r.id_for_name("SelectiveScan"),
            Some(fuel_graph::registry::FusedOps::SELECTIVE_SCAN),
        );
        // Shape rule: y matches u's shape.
        let out_shape = (e.shape_rule)(
            &[
                Shape::from_dims(&[2, 8, 64]),   // u: [batch, seqlen, dim]
                Shape::from_dims(&[2, 8, 64]),   // delta: same
                Shape::from_dims(&[64, 16]),     // a: [dim, dstate]
                Shape::from_dims(&[2, 8, 16]),   // b: [batch, seqlen, dstate]
                Shape::from_dims(&[2, 8, 16]),   // c: [batch, seqlen, dstate]
            ],
            &fuel_graph::registry::FusedOpParams::SelectiveScan {
                delta_softplus: false,
                return_state: false,
            },
        );
        assert_eq!(out_shape.dims(), &[2, 8, 64]);
    }

    /// SsdChunkScan end-to-end: minimal degenerate case through the
    /// dispatcher; verifies the full plumbing from builder → Op::Fused
    /// → op_to_op_kind/op_to_op_params → wrapper → kernel.
    #[test]
    fn ssd_chunk_scan_basic_end_to_end() {
        let device = crate::Device::cpu();
        // [batch=1, seqlen=1, heads=1, head_dim=1]
        let x = LazyTensor::from_f32(vec![3.0_f32], Shape::from_dims(&[1, 1, 1, 1]), &device);
        let dt = x.const_f32_like(vec![1.0_f32], Shape::from_dims(&[1, 1, 1]));
        let a = x.const_f32_like(vec![-1.0_f32], Shape::from_dims(&[1]));
        let b = x.const_f32_like(vec![2.0_f32], Shape::from_dims(&[1, 1, 1, 1]));
        let c = x.const_f32_like(vec![0.5_f32], Shape::from_dims(&[1, 1, 1, 1]));
        let y = x.ssd_chunk_scan(&dt, &a, &b, &c, 1).realize_f32();
        assert_eq!(y.len(), 1);
        assert!((y[0] - 3.0).abs() < 1e-5, "got {}", y[0]);
    }

    /// SsdChunkScan registry roundtrip + shape rule.
    #[test]
    fn ssd_chunk_scan_registry_entry_registered() {
        let r = fuel_graph::registry::default_registry();
        let e = r
            .entry(fuel_graph::registry::FusedOps::SSD_CHUNK_SCAN)
            .expect("SSD_CHUNK_SCAN registered");
        assert_eq!(e.name, "SsdChunkScan");
        assert_eq!(
            r.id_for_name("SsdChunkScan"),
            Some(fuel_graph::registry::FusedOps::SSD_CHUNK_SCAN),
        );
        // Shape rule: y matches x's shape.
        let out_shape = (e.shape_rule)(
            &[
                Shape::from_dims(&[2, 16, 8, 64]),    // x: [batch, seqlen, heads, head_dim]
                Shape::from_dims(&[2, 16, 8]),        // dt
                Shape::from_dims(&[8]),               // a
                Shape::from_dims(&[2, 16, 8, 128]),   // b: [batch, seqlen, heads, state_dim]
                Shape::from_dims(&[2, 16, 8, 128]),   // c
            ],
            &fuel_graph::registry::FusedOpParams::SsdChunkScan {
                chunk_size: 16,
                return_state: false,
            },
        );
        assert_eq!(out_shape.dims(), &[2, 16, 8, 64]);
    }

    /// Nf4Matmul end-to-end: F32 case through the dispatcher.
    /// Verifies the full plumbing builder → Op::Fused → op_to_op_kind
    /// / op_to_op_params → wrapper → kernel.
    #[test]
    fn nf4_matmul_basic_end_to_end() {
        let device = crate::Device::cpu();
        // m=1, n=2, k=4, block_size=2 — same hand-computed test as
        // the byte-kernel two-outputs-two-blocks check.
        let activations = LazyTensor::from_f32(
            vec![1.0_f32, 2.0, 2.0, 4.0],
            Shape::from_dims(&[1, 4]),
            &device,
        );
        let w_packed_t = activations.graph_tensor().const_u8_like(
            vec![247_u8, 247, 127, 127],
            Shape::from_dims(&[2, 2]),
        );
        let w_packed = LazyTensor::from_graph_tensor(w_packed_t);
        let absmax = activations.const_f32_like(
            vec![1.0_f32, 2.0, 10.0, 20.0],
            Shape::from_dims(&[2, 2]),
        );
        let y = activations.nf4_matmul(&w_packed, &absmax, 2).realize_f32();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 10.0).abs() < 1e-5, "out 0: {}", y[0]);
        assert!((y[1] - 50.0).abs() < 1e-5, "out 1: {}", y[1]);
    }

    /// Nf4Matmul registry roundtrip + shape rule.
    #[test]
    fn nf4_matmul_registry_entry_registered() {
        let r = fuel_graph::registry::default_registry();
        let e = r
            .entry(fuel_graph::registry::FusedOps::NF4_MATMUL)
            .expect("NF4_MATMUL registered");
        assert_eq!(e.name, "Nf4Matmul");
        assert_eq!(
            r.id_for_name("Nf4Matmul"),
            Some(fuel_graph::registry::FusedOps::NF4_MATMUL),
        );
        // Shape rule: output last dim becomes w_packed's n.
        let out_shape = (e.shape_rule)(
            &[
                Shape::from_dims(&[4, 32, 128]),    // activations [..., m, k]
                Shape::from_dims(&[256, 64]),       // w_packed [n, k/2]
                Shape::from_dims(&[256, 2]),        // absmax [n, k/block_size]
            ],
            &fuel_graph::registry::FusedOpParams::Nf4Matmul { block_size: 64 },
        );
        assert_eq!(out_shape.dims(), &[4, 32, 256]);
        // Dtype rule: F16 activations → F16 output.
        let out_dtype = (e.dtype_rule)(
            &[
                fuel_core_types::DType::F16,
                fuel_core_types::DType::U8,
                fuel_core_types::DType::F32,
            ],
            &fuel_graph::registry::FusedOpParams::Nf4Matmul { block_size: 64 },
        );
        assert_eq!(out_dtype, fuel_core_types::DType::F16);
    }

    /// Registry: the FusedSoftmaxCrossEntropy entry is wired into the
    /// default registry and reachable by id + name.
    #[test]
    fn fused_softmax_cross_entropy_registry_entry_registered() {
        let r = fuel_graph::registry::default_registry();
        let e = r
            .entry(fuel_graph::registry::FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY)
            .expect("FUSED_SOFTMAX_CROSS_ENTROPY registered");
        assert_eq!(e.name, "FusedSoftmaxCrossEntropy");
        assert_eq!(
            r.id_for_name("FusedSoftmaxCrossEntropy"),
            Some(fuel_graph::registry::FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY),
        );
        // Output dtype is always F32 regardless of input dtypes.
        let out_dtype = (e.dtype_rule)(
            &[DType::F32, DType::I64],
            &fuel_graph::registry::FusedOpParams::FusedSoftmaxCrossEntropy {
                reduction:    Reduction::Mean,
                ignore_index: -100,
            },
        );
        assert_eq!(out_dtype, DType::F32);
    }

    /// Linear regression: fit y = 2x + 3 given noisy samples.
    /// Trains a single-layer `y_hat = w·x + b` with SGD and checks
    /// w, b converge to ~2, ~3.
    #[test]
    fn sgd_fits_linear_regression() {
        // Training data: y = 2x + 3 at x = 0..10.
        let xs: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();

        let mut exe = GraphExecutor::new(CpuBackend);
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[1]), vec![0.1f32]),
            Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.1f32]),
        ];
        let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.01)).unwrap();

        let n_steps = 2000;
        let x_arc: Arc<[f32]> = xs.clone().into();
        let y_arc: Arc<[f32]> = ys.clone().into();
        for step in 0..n_steps {
            let x_arc_step = x_arc.clone();
            let y_arc_step = y_arc.clone();
            let len = xs.len();
            let loss = state.step(&mut exe, move |_graph, params| {
                let w = &params["w"];
                let b = &params["b"];
                // Build inputs on the SAME graph the parameters live in
                // by using `const_f32_like` off an existing param.
                let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
                let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
                let w_b = w.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let b_b = b.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let y_hat = x.mul(&w_b).add(&b_b);
                let diff = y_hat.sub(&y);
                let sq = diff.sqr();
                sq.sum_all().mul_scalar(1.0 / len as f64)
            }).unwrap();
            if step % 500 == 0 {
                eprintln!("step {step}: loss = {loss}");
            }
        }
        let w_final = state.param_to_host("w", &exe).unwrap()[0];
        let b_final = state.param_to_host("b", &exe).unwrap()[0];
        eprintln!("final: w = {w_final}, b = {b_final}");
        assert!((w_final - 2.0).abs() < 0.05, "w converged to {w_final}, expected ~2.0");
        assert!((b_final - 3.0).abs() < 0.3, "b converged to {b_final}, expected ~3.0");
    }

    /// AdamW fits the same regression. AdamW has per-param state
    /// (first / second moments) that live on-device between steps
    /// via the same `pre_populate` mechanism, so this test proves
    /// the multi-tensor state round-trip works.
    #[test]
    fn adamw_fits_linear_regression() {
        let xs: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();
        let mut exe = GraphExecutor::new(CpuBackend);
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[1]), vec![0.1f32]),
            Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.1f32]),
        ];
        // Weight decay 0 for this tiny problem so it converges cleanly.
        let cfg = OptimizerConfig::AdamW {
            lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0,
        };
        let mut state = TrainState::new(&params, &mut exe, cfg).unwrap();
        let x_arc: Arc<[f32]> = xs.clone().into();
        let y_arc: Arc<[f32]> = ys.clone().into();
        let mut final_loss = 0.0;
        for step in 0..500 {
            let x_arc_step = x_arc.clone();
            let y_arc_step = y_arc.clone();
            let len = xs.len();
            let loss = state.step(&mut exe, move |_graph, params| {
                let w = &params["w"];
                let b = &params["b"];
                let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
                let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
                let w_b = w.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let b_b = b.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let y_hat = x.mul(&w_b).add(&b_b);
                let diff = y_hat.sub(&y);
                diff.sqr().sum_all().mul_scalar(1.0 / len as f64)
            }).unwrap();
            final_loss = loss;
            if step % 100 == 0 {
                eprintln!("adamw step {step}: loss = {loss}");
            }
        }
        let w_final = state.param_to_host("w", &exe).unwrap()[0];
        let b_final = state.param_to_host("b", &exe).unwrap()[0];
        eprintln!("adamw final: w = {w_final}, b = {b_final}, loss = {final_loss}");
        // AdamW converges faster than SGD for this problem — after 500
        // steps we expect both params well within 5% of target.
        assert!((w_final - 2.0).abs() < 0.1);
        assert!((b_final - 3.0).abs() < 0.5);
        assert!(final_loss < 0.1);
    }

    /// Train a 2-class classifier with cross-entropy loss. Fit a
    /// linear layer `logits = x @ W + b` to separate two Gaussians.
    /// Exercises the full CE-with-logits path including backward.
    #[test]
    fn sgd_fits_2class_classifier_cross_entropy() {
        // Synthetic data: 20 samples, 2 features, 2 classes.
        // Class 0: samples near (-1, -1). Class 1: samples near (+1, +1).
        let n = 20;
        let n_feat = 2;
        let n_class = 2;
        let mut xs: Vec<f32> = Vec::with_capacity(n * n_feat);
        let mut ys_onehot: Vec<f32> = vec![0.0; n * n_class];
        for i in 0..n {
            let cls = i % 2;
            // Deterministic "noise" with a small sinusoid.
            let jitter = ((i as f32) * 0.37).sin() * 0.1;
            let sign = if cls == 0 { -1.0 } else { 1.0 };
            xs.push(sign + jitter);
            xs.push(sign - jitter);
            ys_onehot[i * n_class + cls] = 1.0;
        }

        let mut exe = GraphExecutor::new(CpuBackend);
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[n_feat, n_class]),
                vec![0.01f32, -0.01, 0.02, -0.02]),
            Parameter::new_f32("b", Shape::from_dims(&[n_class]),
                vec![0.0f32, 0.0]),
        ];
        let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.1)).unwrap();

        let x_arc: Arc<[f32]> = xs.into();
        let y_arc: Arc<[f32]> = ys_onehot.into();
        let mut final_loss = 0.0;
        for step in 0..500 {
            let x_arc_step = x_arc.clone();
            let y_arc_step = y_arc.clone();
            let loss = state.step(&mut exe, move |_graph, params| {
                let w = &params["w"];
                let b = &params["b"];
                let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[n, n_feat]));
                let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[n, n_class]));
                // logits = x @ W + b_broadcast
                let logits_raw = x.matmul(w);
                let b_b = b.reshape(Shape::from_dims(&[1, n_class])).unwrap()
                    .broadcast_to(Shape::from_dims(&[n, n_class])).unwrap();
                let logits = logits_raw.add(&b_b);
                super::loss::cross_entropy_with_logits(&logits, &y)
            }).unwrap();
            final_loss = loss;
            if step % 100 == 0 {
                eprintln!("ce step {step}: loss = {loss}");
            }
        }
        eprintln!("ce final loss = {final_loss}");
        // For this well-separated 2-class problem, cross-entropy should
        // drop well below ln(2) ≈ 0.693 (random baseline) to near zero.
        assert!(final_loss < 0.1, "CE didn't converge: final loss = {final_loss}");
    }

    /// Sanity-check that the RmsNormLastDim backward path (synthesized
    /// via primitives) produces useful gradients. Train a tiny model
    /// that pipes x through RMSNorm then a linear layer, and check
    /// that loss decreases. If the backward were broken we'd get
    /// NaNs or stagnant loss.
    #[test]
    fn sgd_trains_through_rms_norm() {
        let n = 16;
        let d = 4;
        let xs: Vec<f32> = (0..n * d).map(|i| ((i as f32) * 0.37).sin()).collect();
        // Regress to y = sum(x) — something RMSNorm can't encode directly
        // (because it normalizes away magnitude), so the linear layer
        // has to learn a bias. Verifies gradients propagate through
        // the fused op.
        let ys: Vec<f32> = (0..n)
            .map(|row| {
                let s: f32 = (0..d).map(|c| xs[row * d + c]).sum();
                s
            })
            .collect();

        let mut exe = GraphExecutor::new(CpuBackend);
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[d, 1]), vec![0.1f32; d]),
            Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.0f32]),
        ];
        let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.05)).unwrap();
        let x_arc: Arc<[f32]> = xs.into();
        let y_arc: Arc<[f32]> = ys.into();
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        for step in 0..300 {
            let x_arc_step = x_arc.clone();
            let y_arc_step = y_arc.clone();
            let loss = state.step(&mut exe, move |_graph, params| {
                let w = &params["w"];
                let b = &params["b"];
                let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[n, d]));
                let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[n, 1]));
                let x_norm = x.rms_norm_last_dim(1e-6);
                let logits = x_norm.matmul(w);
                let b_b = b.reshape(Shape::from_dims(&[1, 1])).unwrap()
                    .broadcast_to(Shape::from_dims(&[n, 1])).unwrap();
                let pred = logits.add(&b_b);
                super::loss::mse(&pred, &y)
            }).unwrap();
            if step == 0 { initial_loss = loss; }
            final_loss = loss;
        }
        eprintln!("rms_norm training: initial={initial_loss} final={final_loss}");
        assert!(final_loss < initial_loss * 0.5,
            "loss didn't decrease enough: {initial_loss} -> {final_loss}");
        assert!(final_loss.is_finite(), "got non-finite loss: {final_loss}");
    }

    #[test]
    fn warmup_cosine_schedule_curve() {
        let sch = super::WarmupCosine { warmup: 10, total: 100, peak: 1.0, final_lr: 0.1 };
        assert_eq!(sch.lr_at(0), 0.0);                      // start of warmup
        assert!((sch.lr_at(5) - 0.5).abs() < 1e-6);         // mid warmup
        assert!((sch.lr_at(10) - 1.0).abs() < 1e-6);        // peak
        assert!((sch.lr_at(100) - 0.1).abs() < 1e-6);       // end
        // Monotonic decay after warmup.
        let a = sch.lr_at(20);
        let b = sch.lr_at(40);
        let c = sch.lr_at(60);
        assert!(a > b && b > c, "cosine decay not monotonic: {a} -> {b} -> {c}");
    }

    #[test]
    fn warmup_linear_schedule_curve() {
        let sch = super::WarmupLinear { warmup: 10, total: 30, peak: 1.0, final_lr: 0.2 };
        assert_eq!(sch.lr_at(0), 0.0);
        assert!((sch.lr_at(10) - 1.0).abs() < 1e-6);
        // Halfway through decay: (10+30)/2 = 20 → lr = 0.5*(1.0+0.2) = 0.6
        assert!((sch.lr_at(20) - 0.6).abs() < 1e-6);
        assert_eq!(sch.lr_at(30), 0.2);
    }

    /// With an absurdly high learning rate on a high-magnitude loss
    /// landscape, an unclipped SGD run diverges (produces non-finite
    /// parameters). Clipping the global gradient norm to 1.0 keeps
    /// the step bounded and training remains finite. This also
    /// verifies the clip scale graph is structurally correct — if
    /// the scalar division or broadcasting were broken we'd diverge
    /// or get wrong values.
    #[test]
    fn grad_clip_prevents_divergence() {
        let xs: Vec<f32> = (0..10).map(|i| i as f32 * 100.0).collect();  // huge inputs
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[1]), vec![0.1f32]),
            Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.1f32]),
        ];
        let x_arc: Arc<[f32]> = xs.clone().into();
        let y_arc: Arc<[f32]> = ys.clone().into();
        let len = xs.len();

        // Unclipped: this LR is wild given the input magnitude →
        // expect divergence (non-finite weights).
        {
            let mut exe = GraphExecutor::new(CpuBackend);
            let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.1)).unwrap();
            for _ in 0..10 {
                let x_arc_step = x_arc.clone();
                let y_arc_step = y_arc.clone();
                let _ = state.step(&mut exe, move |_g, p| {
                    let w = &p["w"]; let b = &p["b"];
                    let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
                    let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
                    let w_b = w.broadcast_to(Shape::from_dims(&[len])).unwrap();
                    let b_b = b.broadcast_to(Shape::from_dims(&[len])).unwrap();
                    let y_hat = x.mul(&w_b).add(&b_b);
                    y_hat.sub(&y).sqr().sum_all().mul_scalar(1.0 / len as f64)
                }).unwrap();
            }
            let w = state.param_to_host("w", &exe).unwrap()[0];
            assert!(!w.is_finite(), "expected divergence without clipping, got w={w}");
        }

        // Clipped: global-norm clip at 1.0 keeps every step bounded.
        {
            let mut exe = GraphExecutor::new(CpuBackend);
            let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.1))
                .unwrap()
                .with_grad_clip(Some(GradClip::GlobalNorm(1.0)));
            for _ in 0..200 {
                let x_arc_step = x_arc.clone();
                let y_arc_step = y_arc.clone();
                let _ = state.step(&mut exe, move |_g, p| {
                    let w = &p["w"]; let b = &p["b"];
                    let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
                    let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
                    let w_b = w.broadcast_to(Shape::from_dims(&[len])).unwrap();
                    let b_b = b.broadcast_to(Shape::from_dims(&[len])).unwrap();
                    let y_hat = x.mul(&w_b).add(&b_b);
                    y_hat.sub(&y).sqr().sum_all().mul_scalar(1.0 / len as f64)
                }).unwrap();
            }
            let w = state.param_to_host("w", &exe).unwrap()[0];
            let b = state.param_to_host("b", &exe).unwrap()[0];
            assert!(w.is_finite() && b.is_finite(),
                "clipped training should stay finite, got w={w} b={b}");
        }
    }

    #[test]
    fn sgd_with_warmup_cosine_converges() {
        let xs: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();
        let mut exe = GraphExecutor::new(CpuBackend);
        let params = vec![
            Parameter::new_f32("w", Shape::from_dims(&[1]), vec![0.1f32]),
            Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.1f32]),
        ];
        // Start at 0 LR (pure warmup) and use the schedule to ramp up.
        let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.0)).unwrap();
        let sch = super::WarmupCosine { warmup: 100, total: 2000, peak: 0.01, final_lr: 0.001 };

        let x_arc: Arc<[f32]> = xs.clone().into();
        let y_arc: Arc<[f32]> = ys.clone().into();
        for _ in 0..2000 {
            let x_arc_step = x_arc.clone();
            let y_arc_step = y_arc.clone();
            let len = xs.len();
            state.step_with_schedule(&sch, &mut exe, move |_graph, params| {
                let w = &params["w"];
                let b = &params["b"];
                let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
                let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
                let w_b = w.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let b_b = b.broadcast_to(Shape::from_dims(&[len])).unwrap();
                let y_hat = x.mul(&w_b).add(&b_b);
                let diff = y_hat.sub(&y);
                diff.sqr().sum_all().mul_scalar(1.0 / len as f64)
            }).unwrap();
        }
        let w_final = state.param_to_host("w", &exe).unwrap()[0];
        let b_final = state.param_to_host("b", &exe).unwrap()[0];
        assert!((w_final - 2.0).abs() < 0.05);
        assert!((b_final - 3.0).abs() < 0.3);
    }
}

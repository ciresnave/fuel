//! Lazy-graph port of `fuel-nn::optim`: the [`LazyOptimizer`] trait
//! with [`LazySgd`] and [`LazyAdamW`] implementations.
//!
//! ## Design
//!
//! [`LazyVar`] is the lazy equivalent of eager [`crate::Var`]: a
//! named, mutable F32 host-resident parameter. Each training step:
//!
//! 1. The user builds a forward graph that uses
//!    [`LazyVar::tensor`] to splice each parameter in as a graph
//!    `const`. `tensor` records the issued [`fuel_graph::NodeId`]
//!    on the `LazyVar` so [`LazyOptimizer::backward_step`] can fetch
//!    the gradient back out of the `GradMap`.
//! 2. The user computes a scalar `loss` and calls `backward_step`.
//!    The optimizer runs `loss.backward()`, harvests each
//!    parameter's gradient `LazyTensor`, and calls [`step`].
//! 3. [`step`] builds update ops on the same graph as the
//!    gradients (`param = param - lr*grad`, etc.), realizes the
//!    new parameter values to host f32, and writes them back to
//!    each [`LazyVar`]'s shared host buffer. Per-parameter
//!    optimizer state (SGD velocity / AdamW first+second moments)
//!    rides the same path.
//!
//! v1 is F32 + CPU realize via [`LazyTensor::realize_f32`]. Other
//! dtypes / devices land in follow-ups together with
//! `port-training-augmentations.md`'s in-place update primitive.
//!
//! [`step`]: LazyOptimizer::step

use crate::lazy::{realize_many_f32, LazyTensor};
use crate::Result;
use fuel_core_types::{Error, Shape};
use fuel_graph::NodeId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A lazy-graph trainable parameter. Owns its name, shape, and
/// host-resident f32 values via shared interior mutability so the
/// optimizer can write a new value without consuming the handle.
///
/// Use [`LazyVar::tensor`] inside the forward-graph build step to
/// splice the current parameter value in as a graph const. Each
/// call records the issued [`NodeId`] so [`LazyOptimizer::backward_step`]
/// can find the gradient.
#[derive(Clone, Debug)]
pub struct LazyVar {
    name: String,
    shape: Shape,
    data: Arc<RwLock<Vec<f32>>>,
    last_node: Arc<RwLock<Option<NodeId>>>,
}

impl LazyVar {
    /// Build a new parameter with the given name, shape, and initial
    /// host data. Length must equal `shape.elem_count()`.
    pub fn new(name: impl Into<String>, shape: impl Into<Shape>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error::Msg(format!(
                "LazyVar::new: data len {} != shape elem_count {}",
                data.len(),
                shape.elem_count(),
            ))
            .bt());
        }
        Ok(Self {
            name: name.into(),
            shape,
            data: Arc::new(RwLock::new(data)),
            last_node: Arc::new(RwLock::new(None)),
        })
    }

    /// Convenience: zero-initialized parameter.
    pub fn zeros(name: impl Into<String>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let n = shape.elem_count();
        Self::new(name, shape, vec![0.0_f32; n])
    }

    /// Convenience: ones-initialized parameter.
    pub fn ones(name: impl Into<String>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let n = shape.elem_count();
        Self::new(name, shape, vec![1.0_f32; n])
    }

    /// The parameter's name. Used as the key in the gradient map
    /// passed to [`LazyOptimizer::step`].
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The parameter's shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// A snapshot of the parameter's current host values.
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.read().unwrap().clone()
    }

    /// Splice the parameter into the graph anchored at `anchor` as
    /// a fresh `const`. Records the [`NodeId`] so
    /// [`LazyOptimizer::backward_step`] can recover the gradient.
    pub fn tensor(&self, anchor: &LazyTensor) -> LazyTensor {
        let snapshot = self.data.read().unwrap().clone();
        let lt = anchor.const_f32_like(snapshot, self.shape.clone());
        *self.last_node.write().unwrap() = Some(lt.graph_tensor().id());
        lt
    }

    /// Overwrite the host buffer in place. Used internally by the
    /// optimizer; also useful for manual checkpointing.
    pub fn set(&self, values: Vec<f32>) -> Result<()> {
        if values.len() != self.shape.elem_count() {
            return Err(Error::Msg(format!(
                "LazyVar::set: values len {} != shape elem_count {}",
                values.len(),
                self.shape.elem_count(),
            ))
            .bt());
        }
        *self.data.write().unwrap() = values;
        Ok(())
    }

    fn last_node_id(&self) -> Option<NodeId> {
        *self.last_node.read().unwrap()
    }
}

/// Common lazy-optimizer interface. Mirrors the eager
/// `fuel-nn::Optimizer` trait, but parameters are [`LazyVar`]s and
/// gradients are [`LazyTensor`]s keyed by parameter name.
pub trait LazyOptimizer: Sized {
    type Config;

    fn new(params: Vec<LazyVar>, cfg: Self::Config) -> Result<Self>;

    /// Apply one update step from a precomputed gradient map. Keys
    /// are parameter names; parameters absent from `grads` are left
    /// unchanged (matches the eager trait's "missing-grad ≡ no-op"
    /// semantics).
    fn step(&mut self, grads: &HashMap<String, LazyTensor>) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    /// Compute gradients from `loss` and apply one step. Each
    /// parameter's gradient is looked up via the [`NodeId`]
    /// recorded by [`LazyVar::tensor`] during the forward build.
    /// Parameters that did not contribute to `loss` (no NodeId
    /// recorded, or absent from the [`fuel_graph::GradMap`]) are
    /// skipped.
    fn backward_step(&mut self, loss: &LazyTensor) -> Result<()> {
        let grad_map = loss.backward();
        let mut grads: HashMap<String, LazyTensor> = HashMap::new();
        for var in self.params() {
            let Some(node_id) = var.last_node_id() else {
                continue;
            };
            let handle = fuel_graph::Tensor::from_existing(
                loss.graph_tensor().graph().clone(),
                node_id,
            );
            if let Some(grad) = grad_map.get(&handle) {
                grads.insert(var.name().to_string(), LazyTensor::from_graph_tensor(grad));
            }
        }
        self.step(&grads)
    }

    /// Borrow the parameter set; the trait uses this to drive
    /// `backward_step`. Implementations expose their own
    /// parameter-vector accessor by name.
    fn params(&self) -> &[LazyVar];
}

// ============================================================================
// SGD
// ============================================================================

/// Stochastic gradient descent config.
#[derive(Clone, Copy, Debug)]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: f64,
    /// Momentum coefficient. `0.0` disables the velocity buffer
    /// entirely (plain SGD).
    pub momentum: f64,
    /// L2 weight-decay coefficient. Adds `weight_decay * param` to
    /// the gradient before the (momentum-aware) update.
    pub weight_decay: f64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }
}

impl SgdConfig {
    /// Plain SGD with the given learning rate; `momentum =
    /// weight_decay = 0`.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

/// Stochastic gradient descent over [`LazyVar`] parameters with
/// optional momentum and L2 weight decay. Update rule per param:
///
/// ```text
///   g'   = g + weight_decay * w
///   v    = momentum * v + g'              (when momentum > 0)
///   w   -= lr * v                          (or g' if momentum == 0)
/// ```
#[derive(Debug)]
pub struct LazySgd {
    params: Vec<LazyVar>,
    velocity: HashMap<String, Vec<f32>>,
    cfg: SgdConfig,
}

impl LazySgd {
    /// Borrow the underlying parameter list.
    pub fn parameters(&self) -> &[LazyVar] {
        &self.params
    }

    /// Current velocity snapshot for `name` (only present when
    /// `momentum > 0`).
    pub fn velocity_for(&self, name: &str) -> Option<&[f32]> {
        self.velocity.get(name).map(|v| v.as_slice())
    }
}

impl LazyOptimizer for LazySgd {
    type Config = SgdConfig;

    fn new(params: Vec<LazyVar>, cfg: SgdConfig) -> Result<Self> {
        let velocity = if cfg.momentum != 0.0 {
            params
                .iter()
                .map(|v| (v.name().to_string(), vec![0.0_f32; v.shape().elem_count()]))
                .collect()
        } else {
            HashMap::new()
        };
        Ok(Self {
            params,
            velocity,
            cfg,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.cfg.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.cfg.lr = lr;
    }

    fn params(&self) -> &[LazyVar] {
        &self.params
    }

    fn step(&mut self, grads: &HashMap<String, LazyTensor>) -> Result<()> {
        if self.params.is_empty() {
            return Ok(());
        }
        let momentum = self.cfg.momentum;
        let lr = self.cfg.lr;
        let wd = self.cfg.weight_decay;

        let mut roots: Vec<LazyTensor> = Vec::with_capacity(self.params.len());
        let mut active: Vec<usize> = Vec::with_capacity(self.params.len());

        for (idx, var) in self.params.iter().enumerate() {
            let Some(grad) = grads.get(var.name()) else {
                continue;
            };
            let param_t = var.tensor(grad);

            let g_eff = if wd != 0.0 {
                let decayed = param_t.mul_scalar(wd);
                grad.add(&decayed)?
            } else {
                grad.clone()
            };

            let update = if momentum != 0.0 {
                let v_prev = self.velocity[var.name()].clone();
                let v_prev_t = grad.const_f32_like(v_prev, var.shape().clone());
                v_prev_t.mul_scalar(momentum).add(&g_eff)?
            } else {
                g_eff
            };

            let scaled = update.mul_scalar(lr);
            let new_param = param_t.sub(&scaled)?;
            roots.push(new_param);
            active.push(idx);
        }

        if roots.is_empty() {
            return Ok(());
        }

        let realized = if momentum != 0.0 {
            // Realize new_param and the per-step `update` (which IS the
            // new velocity when momentum > 0) jointly so we get both
            // host snapshots in one graph traversal.
            let mut combined: Vec<&LazyTensor> = Vec::with_capacity(roots.len() * 2);
            for r in &roots {
                combined.push(r);
            }
            // Re-derive velocity tensors by reading back from the graph:
            // we need them on host to seed the next step. The simplest
            // correct path is to realize the gradient-plus-velocity
            // intermediate alongside each new param.
            //
            // We do that by replaying the same expression here as a
            // sibling root. Re-using `roots`' LazyTensor handles is
            // fine because realize_many_f32 walks the shared graph.
            // Concretely: store the `update` tensors directly.
            let mut updates: Vec<LazyTensor> = Vec::with_capacity(active.len());
            for &idx in &active {
                let var = &self.params[idx];
                let grad = &grads[var.name()];
                let g_eff = if wd != 0.0 {
                    let param_t = var.tensor(grad);
                    let decayed = param_t.mul_scalar(wd);
                    grad.add(&decayed)?
                } else {
                    grad.clone()
                };
                let v_prev = self.velocity[var.name()].clone();
                let v_prev_t = grad.const_f32_like(v_prev, var.shape().clone());
                let new_v = v_prev_t.mul_scalar(momentum).add(&g_eff)?;
                updates.push(new_v);
            }
            for u in &updates {
                combined.push(u);
            }
            realize_many_f32(&combined)
        } else {
            let refs: Vec<&LazyTensor> = roots.iter().collect();
            realize_many_f32(&refs)
        };

        let n_params = active.len();
        for (i, &idx) in active.iter().enumerate() {
            let var = &self.params[idx];
            var.set(realized[i].clone())?;
        }
        if momentum != 0.0 {
            for (i, &idx) in active.iter().enumerate() {
                let var = &self.params[idx];
                let v_new = realized[n_params + i].clone();
                self.velocity.insert(var.name().to_string(), v_new);
            }
        }
        Ok(())
    }
}

// ============================================================================
// AdamW
// ============================================================================

/// AdamW config. Defaults match the original
/// [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)
/// recipe: `lr=1e-3`, `(β1, β2) = (0.9, 0.999)`, `ε=1e-8`,
/// `weight_decay=0.01`.
#[derive(Clone, Copy, Debug)]
pub struct AdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl AdamWConfig {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Self::default()
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    pub fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }
    pub fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

/// AdamW optimizer over [`LazyVar`] parameters. Decoupled weight
/// decay: `w` is scaled by `(1 - lr·λ)` BEFORE the moment-based
/// update is subtracted, exactly matching the
/// [Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)
/// algorithm and the eager `fuel-nn::AdamW` impl.
#[derive(Debug)]
pub struct LazyAdamW {
    params: Vec<LazyVar>,
    first_moment: HashMap<String, Vec<f32>>,
    second_moment: HashMap<String, Vec<f32>>,
    cfg: AdamWConfig,
    step_t: usize,
}

impl LazyAdamW {
    pub fn parameters(&self) -> &[LazyVar] {
        &self.params
    }

    /// Number of update steps applied so far. Used by the
    /// bias-correction terms.
    pub fn step_count(&self) -> usize {
        self.step_t
    }

    pub fn first_moment_for(&self, name: &str) -> Option<&[f32]> {
        self.first_moment.get(name).map(|v| v.as_slice())
    }

    pub fn second_moment_for(&self, name: &str) -> Option<&[f32]> {
        self.second_moment.get(name).map(|v| v.as_slice())
    }
}

impl LazyOptimizer for LazyAdamW {
    type Config = AdamWConfig;

    fn new(params: Vec<LazyVar>, cfg: AdamWConfig) -> Result<Self> {
        let mut first_moment = HashMap::new();
        let mut second_moment = HashMap::new();
        for v in &params {
            let n = v.shape().elem_count();
            first_moment.insert(v.name().to_string(), vec![0.0_f32; n]);
            second_moment.insert(v.name().to_string(), vec![0.0_f32; n]);
        }
        Ok(Self {
            params,
            first_moment,
            second_moment,
            cfg,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.cfg.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.cfg.lr = lr;
    }

    fn params(&self) -> &[LazyVar] {
        &self.params
    }

    fn step(&mut self, grads: &HashMap<String, LazyTensor>) -> Result<()> {
        if self.params.is_empty() {
            return Ok(());
        }
        self.step_t += 1;
        let t = self.step_t as i32;

        let lr = self.cfg.lr;
        let beta1 = self.cfg.beta1;
        let beta2 = self.cfg.beta2;
        let eps = self.cfg.eps;
        let lambda = self.cfg.weight_decay;
        let lr_lambda = lr * lambda;

        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);
        let lr_scale_m = lr / bc1;
        let sqrt_scale_v = (1.0 / bc2).sqrt();

        let mut new_params: Vec<LazyTensor> = Vec::with_capacity(self.params.len());
        let mut new_ms: Vec<LazyTensor> = Vec::with_capacity(self.params.len());
        let mut new_vs: Vec<LazyTensor> = Vec::with_capacity(self.params.len());
        let mut active: Vec<usize> = Vec::with_capacity(self.params.len());

        for (idx, var) in self.params.iter().enumerate() {
            let Some(grad) = grads.get(var.name()) else {
                continue;
            };
            let param_t = var.tensor(grad);
            let m_prev = self.first_moment[var.name()].clone();
            let v_prev = self.second_moment[var.name()].clone();
            let m_prev_t = grad.const_f32_like(m_prev, var.shape().clone());
            let v_prev_t = grad.const_f32_like(v_prev, var.shape().clone());

            // m = β1·m + (1-β1)·g
            let m_decayed = m_prev_t.mul_scalar(beta1);
            let g_part = grad.mul_scalar(1.0 - beta1);
            let new_m = m_decayed.add(&g_part)?;

            // v = β2·v + (1-β2)·g²
            let v_decayed = v_prev_t.mul_scalar(beta2);
            let g_sq = grad.sqr();
            let g_sq_part = g_sq.mul_scalar(1.0 - beta2);
            let new_v = v_decayed.add(&g_sq_part)?;

            // denom = sqrt(v) * sqrt(scale_v) + eps  (== sqrt(v/(1-β2^t)) + eps)
            let denom = new_v.sqrt().mul_scalar(sqrt_scale_v).add_scalar(eps);
            // update = (m * (lr / (1 - β1^t))) / denom
            let numer = new_m.mul_scalar(lr_scale_m);
            let update = numer.div(&denom)?;
            // Decoupled weight decay: w ← w·(1 - lr·λ) - update
            let decayed_w = param_t.mul_scalar(1.0 - lr_lambda);
            let new_param = decayed_w.sub(&update)?;

            new_params.push(new_param);
            new_ms.push(new_m);
            new_vs.push(new_v);
            active.push(idx);
        }

        if active.is_empty() {
            return Ok(());
        }

        let mut combined: Vec<&LazyTensor> = Vec::with_capacity(active.len() * 3);
        for r in &new_params {
            combined.push(r);
        }
        for r in &new_ms {
            combined.push(r);
        }
        for r in &new_vs {
            combined.push(r);
        }
        let realized = realize_many_f32(&combined);

        let n = active.len();
        for (i, &idx) in active.iter().enumerate() {
            let var = &self.params[idx];
            var.set(realized[i].clone())?;
            self.first_moment
                .insert(var.name().to_string(), realized[n + i].clone());
            self.second_moment
                .insert(var.name().to_string(), realized[2 * n + i].clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn build_grads_for(vars: &[(&LazyVar, Vec<f32>)]) -> (LazyTensor, HashMap<String, LazyTensor>) {
        let (first_var, first_data) = &vars[0];
        let anchor = LazyTensor::from_f32(
            first_data.clone(),
            first_var.shape().clone(),
            &Device::cpu(),
        );
        let mut map: HashMap<String, LazyTensor> = HashMap::new();
        for (i, (var, data)) in vars.iter().enumerate() {
            let t = if i == 0 {
                anchor.clone()
            } else {
                anchor.const_f32_like(data.clone(), var.shape().clone())
            };
            map.insert(var.name().to_string(), t);
        }
        (anchor, map)
    }

    #[test]
    fn lazy_var_round_trip() {
        let v = LazyVar::new("w", Shape::from_dims(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let anchor = LazyTensor::from_f32(vec![0.0_f32; 3], Shape::from_dims(&[3]), &Device::cpu());
        let t = v.tensor(&anchor);
        let host = t.realize_f32();
        assert_eq!(host, vec![1.0, 2.0, 3.0]);
        v.set(vec![4.0, 5.0, 6.0]).unwrap();
        assert_eq!(v.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn sgd_unit_lr_unit_grad_subtracts_unit() {
        let w = LazyVar::new("w", Shape::from_dims(&[3]), vec![5.0, 5.0, 5.0]).unwrap();
        let cfg = SgdConfig::new(1.0);
        let mut opt = LazySgd::new(vec![w.clone()], cfg).unwrap();
        let (_anchor, grads) = build_grads_for(&[(&w, vec![1.0, 1.0, 1.0])]);
        opt.step(&grads).unwrap();
        // w' = 5 - 1*1 = 4 elementwise
        assert_eq!(w.to_vec(), vec![4.0, 4.0, 4.0]);
    }

    #[test]
    fn sgd_zero_lr_does_not_change_params() {
        let w = LazyVar::new("w", Shape::from_dims(&[2]), vec![7.0, -3.0]).unwrap();
        let cfg = SgdConfig::new(0.0);
        let mut opt = LazySgd::new(vec![w.clone()], cfg).unwrap();
        let (_anchor, grads) = build_grads_for(&[(&w, vec![10.0, 10.0])]);
        opt.step(&grads).unwrap();
        let out = w.to_vec();
        assert!((out[0] - 7.0).abs() < 1e-6, "{:?}", out);
        assert!((out[1] - (-3.0)).abs() < 1e-6, "{:?}", out);
    }

    #[test]
    fn sgd_with_momentum_accumulates_velocity() {
        // momentum = 0.9, lr = 0.1, g = [1, 1]. Plain (no weight decay).
        // step 1: v1 = 0.9*0 + 1 = 1; w1 = 1 - 0.1*1 = 0.9
        // step 2: v2 = 0.9*1 + 1 = 1.9; w2 = 0.9 - 0.1*1.9 = 0.71
        let w = LazyVar::new("w", Shape::from_dims(&[2]), vec![1.0, 1.0]).unwrap();
        let cfg = SgdConfig::new(0.1).with_momentum(0.9);
        let mut opt = LazySgd::new(vec![w.clone()], cfg).unwrap();

        let (_anchor1, grads1) = build_grads_for(&[(&w, vec![1.0, 1.0])]);
        opt.step(&grads1).unwrap();
        let after1 = w.to_vec();
        assert!((after1[0] - 0.9).abs() < 1e-6, "after1: {:?}", after1);

        let v_after1 = opt.velocity_for("w").unwrap().to_vec();
        assert!((v_after1[0] - 1.0).abs() < 1e-6, "v after1: {:?}", v_after1);

        let (_anchor2, grads2) = build_grads_for(&[(&w, vec![1.0, 1.0])]);
        opt.step(&grads2).unwrap();
        let after2 = w.to_vec();
        assert!((after2[0] - 0.71).abs() < 1e-5, "after2: {:?}", after2);

        let v_after2 = opt.velocity_for("w").unwrap().to_vec();
        assert!((v_after2[0] - 1.9).abs() < 1e-5, "v after2: {:?}", v_after2);
    }

    #[test]
    fn adamw_first_step_matches_textbook_formula() {
        // Hand-computed reference for t=1:
        //   β1=0.9, β2=0.999, ε=1e-8, λ=0, lr=0.1
        //   w0 = 2.0, g = 0.5
        //   m1 = 0.9*0 + 0.1*0.5 = 0.05
        //   v1 = 0.999*0 + 0.001*0.25 = 0.00025
        //   m_hat = 0.05 / (1 - 0.9) = 0.5
        //   v_hat = 0.00025 / (1 - 0.999) = 0.25
        //   update = 0.5 / (sqrt(0.25) + 1e-8) = 0.5 / 0.50000001 ≈ 0.999999...
        //   w1 = 2.0 - 0.1 * 0.99999998 ≈ 1.90000000200...
        let w = LazyVar::new("w", Shape::from_dims(&[1]), vec![2.0]).unwrap();
        let cfg = AdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut opt = LazyAdamW::new(vec![w.clone()], cfg).unwrap();
        let (_anchor, grads) = build_grads_for(&[(&w, vec![0.5])]);
        opt.step(&grads).unwrap();
        let after = w.to_vec();

        // Reference computation in f64.
        let m1: f64 = 0.05;
        let v1: f64 = 0.00025;
        let denom = (v1 / 0.001).sqrt() + 1e-8;
        let update = (m1 / 0.1) / denom;
        let expected = 2.0_f64 - 0.1 * update;
        assert!(
            (after[0] as f64 - expected).abs() < 1e-5,
            "got {} expected {}",
            after[0],
            expected,
        );

        // Moment state should match the hand-computed values.
        let m_state = opt.first_moment_for("w").unwrap();
        let v_state = opt.second_moment_for("w").unwrap();
        assert!((m_state[0] as f64 - m1).abs() < 1e-6);
        assert!((v_state[0] as f64 - v1).abs() < 1e-8);
        assert_eq!(opt.step_count(), 1);
    }

    #[test]
    fn adamw_weight_decay_subtracts_before_update() {
        // Decoupled WD: w ← w·(1 - lr·λ) - update.
        // β1=0.9, β2=0.999, ε=1e-8, lr=0.1, λ=0.01, w0=3.0, g=0.0
        // Zero grad ⇒ m1=v1=0 ⇒ update=0.
        // Therefore w1 = 3.0 * (1 - 0.1 * 0.01) = 3.0 * 0.999 = 2.997.
        let w = LazyVar::new("w", Shape::from_dims(&[1]), vec![3.0]).unwrap();
        let cfg = AdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut opt = LazyAdamW::new(vec![w.clone()], cfg).unwrap();
        let (_anchor, grads) = build_grads_for(&[(&w, vec![0.0])]);
        opt.step(&grads).unwrap();
        let after = w.to_vec();
        assert!(
            (after[0] - 2.997).abs() < 1e-5,
            "got {} expected 2.997",
            after[0],
        );
    }

    #[test]
    fn backward_step_runs_loss_backward_then_step() {
        // Build a tiny loss = (w - target)^2 sum_all on a real graph,
        // run backward_step, verify w moved toward target.
        let w = LazyVar::new("w", Shape::from_dims(&[2]), vec![3.0, -1.0]).unwrap();
        let cfg = SgdConfig::new(0.1);
        let mut opt = LazySgd::new(vec![w.clone()], cfg).unwrap();

        let anchor = LazyTensor::from_f32(
            vec![0.0_f32; 2],
            Shape::from_dims(&[2]),
            &Device::cpu(),
        );
        let target = anchor.const_f32_like(vec![1.0_f32, 1.0], Shape::from_dims(&[2]));
        let w_t = w.tensor(&anchor);
        let diff = w_t.sub(&target).unwrap();
        let loss = diff.sqr().sum_all();

        let w0 = w.to_vec();
        opt.backward_step(&loss).unwrap();
        let w1 = w.to_vec();

        // Gradient of (w - target)^2 wrt w is 2*(w - target).
        // For w0 = [3, -1], target = [1, 1], grad = [4, -4].
        // SGD step: w1 = w0 - 0.1 * grad = [3 - 0.4, -1 + 0.4] = [2.6, -0.6].
        assert!((w1[0] - 2.6).abs() < 1e-5, "got {:?}, expected ~[2.6, -0.6]", w1);
        assert!((w1[1] - (-0.6)).abs() < 1e-5, "got {:?}", w1);
        assert!(w1 != w0, "params should have changed");
    }

    #[test]
    fn empty_param_set_is_noop() {
        let cfg = SgdConfig::new(0.1);
        let mut opt = LazySgd::new(vec![], cfg).unwrap();
        let map: HashMap<String, LazyTensor> = HashMap::new();
        opt.step(&map).unwrap();
        assert_eq!(opt.learning_rate(), 0.1);
    }

    #[test]
    fn set_learning_rate_updates_for_next_step() {
        let w = LazyVar::new("w", Shape::from_dims(&[1]), vec![10.0]).unwrap();
        let mut opt = LazySgd::new(vec![w.clone()], SgdConfig::new(1.0)).unwrap();
        opt.set_learning_rate(2.0);
        assert!((opt.learning_rate() - 2.0).abs() < 1e-12);
        let (_a, g) = build_grads_for(&[(&w, vec![1.0])]);
        opt.step(&g).unwrap();
        // w' = 10 - 2*1 = 8
        assert!((w.to_vec()[0] - 8.0).abs() < 1e-6);
    }
}

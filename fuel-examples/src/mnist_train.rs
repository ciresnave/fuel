// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lazy-graph MLP training core for the `mnist-training` example.
//!
//! A minimal 2-layer MLP (`in → hidden → (ReLU) → out`) trained with AdamW on
//! Fuel's lazy autograd — the same [`fuel::train::TrainState`] + `.backward()` +
//! optimizer machinery the `llama-finetune-vulkan` binary demonstrates. The
//! training logic lives here (in the library) rather than in the example binary
//! so it is reachable from an automated test; the binary is a thin loader.
//!
//! ## What the convergence gate proves — and what it does NOT
//!
//! [`tests::mlp_loss_decreases_over_steps`] asserts the training loss decreases
//! over a handful of steps on a fixed-seed synthetic batch. **This proves the
//! training loop is WIRED UP AND MOVING** — parameters are found, gradients flow
//! back through the graph, and the optimizer updates them in a loss-reducing
//! direction.
//!
//! **It does NOT prove the model is correct.** A transposed weight, a wrong axis
//! in the loss, or a wrong data layout can all still produce a decreasing loss
//! curve (the optimizer will happily descend a subtly-wrong objective). So this
//! gate must never be cited as evidence that the MNIST model or its math is
//! right — only that training runs and the gradient path is live. Model
//! correctness is verified by a human running the example against real MNIST and
//! watching the reported test accuracy.
//!
//! It is also, as of this writing, the FIRST automated loss-convergence test in
//! this repo — its stability across the fixed seed is established here; its
//! behaviour across platforms/backends is NOT yet established and must not be
//! assumed.

use std::collections::HashMap;
use std::sync::Arc;

use fuel::lazy::Tensor;
use fuel::train::{OptimizerConfig, Parameter, TrainState, loss};
use fuel::{Device, Result, Shape};

/// A 2-layer MLP: `in_dim → hidden → (ReLU) → out_dim`.
#[derive(Clone, Copy, Debug)]
pub struct MlpConfig {
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
    pub lr: f32,
    /// Seed for deterministic weight init — makes the convergence gate reproducible.
    pub seed: u32,
}

/// A tiny deterministic LCG so weight init (and the test's synthetic data) are
/// reproducible without pulling an rng dependency into the training core.
fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    (*state >> 16) as u16 as f32 / 65_535.0 - 0.5
}

fn vec_of(n: usize, state: &mut u32, scale: f32) -> Arc<[f32]> {
    (0..n)
        .map(|_| lcg(state) * scale)
        .collect::<Vec<_>>()
        .into()
}

/// The four trainable parameters, seeded deterministically. Weights get a small
/// uniform init; biases start at zero.
fn init_params(cfg: &MlpConfig) -> Vec<Parameter> {
    let mut s = cfg.seed.wrapping_add(1);
    let w1_scale = 1.0 / (cfg.in_dim as f32).sqrt();
    let w2_scale = 1.0 / (cfg.hidden as f32).sqrt();
    vec![
        Parameter::new_f32(
            "w1",
            Shape::from_dims(&[cfg.in_dim, cfg.hidden]),
            vec_of(cfg.in_dim * cfg.hidden, &mut s, w1_scale),
        ),
        Parameter::new_f32(
            "b1",
            Shape::from_dims(&[cfg.hidden]),
            vec![0.0f32; cfg.hidden],
        ),
        Parameter::new_f32(
            "w2",
            Shape::from_dims(&[cfg.hidden, cfg.out_dim]),
            vec_of(cfg.hidden * cfg.out_dim, &mut s, w2_scale),
        ),
        Parameter::new_f32(
            "b2",
            Shape::from_dims(&[cfg.out_dim]),
            vec![0.0f32; cfg.out_dim],
        ),
    ]
}

/// The MLP forward: `[n, in] → (ReLU) hidden → logits [n, out]`. Shared by the
/// training step (params from the graph) and the inference eval (params rebuilt
/// from trained host values) so the two can never diverge.
fn mlp_logits(x: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> Result<Tensor> {
    let h = x.matmul(w1)?.broadcast_add(b1)?.relu();
    h.matmul(w2)?.broadcast_add(b2)
}

fn one_hot(labels: &[u32], out_dim: usize) -> Arc<[f32]> {
    let mut oh = vec![0.0f32; labels.len() * out_dim];
    for (i, &l) in labels.iter().enumerate() {
        oh[i * out_dim + (l as usize).min(out_dim - 1)] = 1.0;
    }
    oh.into()
}

/// A persistent MLP trainer — holds the [`TrainState`] across mini-batches.
pub struct MnistTrainer {
    state: TrainState,
    cfg: MlpConfig,
}

impl MnistTrainer {
    pub fn new(cfg: MlpConfig) -> Result<Self> {
        let device = Device::cpu();
        let state = TrainState::new(&init_params(&cfg), &device, OptimizerConfig::adam_w(cfg.lr))?;
        Ok(Self { state, cfg })
    }

    /// One AdamW step over a `[n_samples, in_dim]` batch with `[n_samples]`
    /// labels; returns the batch cross-entropy loss.
    pub fn train_batch(&mut self, images: &[f32], labels: &[u32], n_samples: usize) -> Result<f32> {
        assert_eq!(images.len(), n_samples * self.cfg.in_dim);
        assert_eq!(labels.len(), n_samples);
        let x_data: Arc<[f32]> = images.to_vec().into();
        let t_data = one_hot(labels, self.cfg.out_dim);
        let (in_dim, out_dim) = (self.cfg.in_dim, self.cfg.out_dim);
        self.state
            .step(move |_graph, params: &HashMap<String, Tensor>| {
                let (w1, b1, w2, b2) = (&params["w1"], &params["b1"], &params["w2"], &params["b2"]);
                // Input as a Const on the parameters' graph (the finetune anchor trick).
                let x = w1.const_f32_like(x_data, Shape::from_dims(&[n_samples, in_dim]));
                let logits = mlp_logits(&x, w1, b1, w2, b2)?;
                let target = w1.const_f32_like(t_data, Shape::from_dims(&[n_samples, out_dim]));
                loss::cross_entropy_with_logits(&logits, &target)
            })
    }

    /// Forward-only pass over `[n_samples, in_dim]` images; returns the fraction
    /// of `argmax(logits)` that match `labels`. Reads the trained weights back to
    /// host and runs a fresh inference graph (no gradients, no optimizer).
    pub fn eval_accuracy(&self, images: &[f32], labels: &[u32], n_samples: usize) -> Result<f32> {
        let dev = Device::cpu();
        let cfg = &self.cfg;
        let mk = |name: &str, dims: &[usize]| -> Result<Tensor> {
            let data: Arc<[f32]> = self.state.param_to_host(name)?.into();
            Ok(Tensor::from_f32(data, Shape::from_dims(dims), &dev)?)
        };
        let w1 = mk("w1", &[cfg.in_dim, cfg.hidden])?;
        let b1 = mk("b1", &[cfg.hidden])?;
        let w2 = mk("w2", &[cfg.hidden, cfg.out_dim])?;
        let b2 = mk("b2", &[cfg.out_dim])?;
        let x = Tensor::from_f32(
            images.to_vec(),
            Shape::from_dims(&[n_samples, cfg.in_dim]),
            &dev,
        )?;
        let logits = mlp_logits(&x, &w1, &b1, &w2, &b2)?.realize_f32();

        let mut correct = 0usize;
        for i in 0..n_samples {
            let row = &logits[i * cfg.out_dim..(i + 1) * cfg.out_dim];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(j, _)| j as u32)
                .unwrap_or(0);
            if pred == labels[i] {
                correct += 1;
            }
        }
        Ok(correct as f32 / n_samples as f32)
    }
}

/// Convenience for the convergence gate: train `n_steps` on a single fixed
/// batch, returning the per-step loss (the model memorises the batch, isolating
/// "is training moving?" from the data pipeline).
pub fn train_steps(
    images: &[f32],
    labels: &[u32],
    n_samples: usize,
    cfg: &MlpConfig,
    n_steps: usize,
) -> Result<Vec<f32>> {
    let mut trainer = MnistTrainer::new(*cfg)?;
    (0..n_steps)
        .map(|_| trainer.train_batch(images, labels, n_samples))
        .collect::<Result<Vec<f32>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed-seed synthetic batch. Deterministic, no network, no fuel-datasets.
    fn synthetic_batch(n: usize, in_dim: usize, out_dim: usize, seed: u32) -> (Vec<f32>, Vec<u32>) {
        let mut s = seed;
        let images: Vec<f32> = (0..n * in_dim).map(|_| lcg(&mut s)).collect();
        let labels: Vec<u32> = (0..n)
            .map(|_| ((lcg(&mut s) + 0.5) * out_dim as f32) as u32 % out_dim as u32)
            .collect();
        (images, labels)
    }

    fn tiny_cfg(lr: f32) -> MlpConfig {
        MlpConfig {
            in_dim: 16,
            hidden: 8,
            out_dim: 3,
            lr,
            seed: 7,
        }
    }

    /// THE CONVERGENCE GATE. Loss must be finite throughout and end below where
    /// it started — the model memorising a fixed batch. See the module doc for
    /// exactly what this proves and does NOT.
    #[test]
    fn mlp_loss_decreases_over_steps() {
        let cfg = tiny_cfg(1e-2);
        let (images, labels) = synthetic_batch(8, cfg.in_dim, cfg.out_dim, 42);
        let losses = train_steps(&images, &labels, 8, &cfg, 30).expect("train");
        // Measured trajectory: 1.0918 -> 0.4162 (monotone, ~62% drop). Under the
        // lr=0 sabotage (sibling test) it is flat at 1.0918 and this assertion
        // FAILS — the gate is not vacuously green.
        assert!(
            losses.iter().all(|l| l.is_finite()),
            "all losses finite, got {losses:?}"
        );
        assert!(
            *losses.last().unwrap() < losses[0],
            "training loss must decrease (wired up + gradients flowing): {losses:?}",
        );
    }

    /// Born-red control: with `lr == 0` the optimizer never updates the
    /// parameters, so the loss stays flat and the gate above MUST fail. This
    /// proves the gate discriminates "training is moving" from "training is
    /// inert" — witnessed directly (the decrease gate under lr=0 saw a flat
    /// `[1.0918; 30]` trajectory and failed).
    #[test]
    fn mlp_loss_is_flat_when_learning_rate_is_zero() {
        let cfg = tiny_cfg(0.0);
        let (images, labels) = synthetic_batch(8, cfg.in_dim, cfg.out_dim, 42);
        let losses = train_steps(&images, &labels, 8, &cfg, 30).expect("train");
        assert!(losses.iter().all(|l| l.is_finite()), "losses finite");
        assert!(
            (losses.last().unwrap() - losses[0]).abs() < 1e-6,
            "with lr=0 the loss must stay flat (no learning), got {losses:?}",
        );
    }
}

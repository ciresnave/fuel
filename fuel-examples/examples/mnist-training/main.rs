//! MNIST MLP training example — Fuel's lazy graph + autograd.
//!
//! Trains a 2-layer MLP (`784 → 100 → (ReLU) → 10`) with AdamW on Fuel's lazy
//! autograd (`TrainState` + `.backward()`), printing per-epoch training loss and
//! test-set accuracy. This replaces the retired eager (`fuel_nn` / eager
//! `Tensor`) version deleted in B6.
//!
//! The training core lives in [`fuel_examples::mnist_train`] so it is reachable
//! from an automated convergence gate; this binary is a thin loader over it. Run
//! it and watch the loss fall and accuracy rise:
//!
//! ```sh
//! cargo run --release --example mnist-training --features fuel-datasets
//! ```
//!
//! Model correctness is verified by a human reading the reported test accuracy;
//! the automated gate (`mnist_train::tests`) only proves training is wired up and
//! moving — see that module's doc for the distinction.

use fuel::Result;
use fuel_examples::mnist_train::{MlpConfig, MnistTrainer};

const IMAGE_DIM: usize = 784; // 28 × 28
const LABELS: usize = 10;
const BSIZE: usize = 64;
const EPOCHS: usize = 3;

fn main() -> Result<()> {
    let data = fuel_datasets::vision::mnist::load()?;
    eprintln!(
        "MNIST loaded: {} train / {} test samples, image dims {:?}",
        data.train_samples, data.test_samples, data.image_dims,
    );

    let cfg = MlpConfig { in_dim: IMAGE_DIM, hidden: 100, out_dim: LABELS, lr: 1e-3, seed: 1 };
    let mut trainer = MnistTrainer::new(cfg)?;

    let n_batches = data.train_samples / BSIZE;
    eprintln!("Training MLP (784→100→10) with AdamW lr={}, {EPOCHS} epochs × {n_batches} batches:", cfg.lr);

    for epoch in 0..EPOCHS {
        let mut sum_loss = 0.0f32;
        for b in 0..n_batches {
            let start = b * BSIZE;
            let images = &data.train_images[start * IMAGE_DIM..(start + BSIZE) * IMAGE_DIM];
            let labels = &data.train_labels[start..start + BSIZE];
            sum_loss += trainer.train_batch(images, labels, BSIZE)?;
        }
        let acc = trainer.eval_accuracy(&data.test_images, &data.test_labels, data.test_samples)?;
        eprintln!(
            "  epoch {epoch}: train loss = {:.4}, test accuracy = {:.2}%",
            sum_loss / n_batches as f32,
            acc * 100.0,
        );
    }
    Ok(())
}

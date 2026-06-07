//! Simplified MNIST-style classifier walkthrough on the lazy substrate.
//!
//! Originally authored by Evgeny Igumnov, 2023. This is the
//! lazy-substrate revival: the original eager `fuel-nn` types
//! (`Linear`, `VarBuilder`, `VarMap`, `Optimizer`, `Module`, `loss::nll`,
//! `ops::log_softmax`) were retired during the eager-Tensor retirement
//! program. The walk-through is now expressed entirely against
//! `LazyTensor` + the `fuel_core::lazy_nn*` modules:
//!
//! * `fuel_core::lazy_nn::LazyLinear` / `fuel_core::lazy_nn::linear`
//! * `fuel_core::lazy_nn_varmap::LazyVarMap`
//! * `fuel_core::lazy_nn_varbuilder::LazyVarBuilder`
//! * `fuel_core::lazy_nn_optim::{LazySgd, LazyOptimizer, SgdConfig}`
//! * `fuel_core::lazy_nn_loss::{nll, Reduction}`
//! * `LazyTensor::log_softmax`, `LazyTensor::argmax_dim`
//!
//! The model is a 3-layer MLP trained on a tiny vote-percentage toy
//! dataset to predict the winner of a runoff election. Each forward
//! call builds a fresh graph; `LazySgd::backward_step` runs the graph
//! once per epoch to harvest gradients and write back updated host f32
//! weights into the `LazyVarMap`.

#[rustfmt::skip]
mod tests {

use fuel::{DType, Result, Device, D, Shape};
use fuel::lazy::LazyTensor;
use fuel::lazy_nn::{LazyLinear, LazyModule};
use fuel::lazy_nn_loss::{nll, Reduction};
use fuel::lazy_nn_optim::{LazyOptimizer, LazySgd, SgdConfig};
use fuel::lazy_nn_varbuilder::LazyVarBuilder;
use fuel::lazy_nn_varmap::LazyVarMap;

// ANCHOR: book_training_simplified1
const VOTE_DIM: usize = 2;
const RESULTS: usize = 1;
const EPOCHS: usize = 10;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 2;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_votes: LazyTensor,
    pub train_results: LazyTensor,
    pub test_votes: LazyTensor,
    pub test_results: LazyTensor,
}

struct MultiLevelPerceptron {
    ln1: LazyLinear,
    ln2: LazyLinear,
    ln3: LazyLinear,
}

impl MultiLevelPerceptron {
    fn new(vs: &LazyVarBuilder) -> Result<Self> {
        let ln1 = fuel::lazy_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, &vs.pp("ln1"))?;
        let ln2 = fuel::lazy_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, &vs.pp("ln2"))?;
        let ln3 = fuel::lazy_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, &vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu();
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu();
        self.ln3.forward(&xs)
    }
}

// ANCHOR_END: book_training_simplified1



// ANCHOR: book_training_simplified3
#[tokio::test]
async fn simplified() -> anyhow::Result<()> {

    let dev = Device::cpu();

    let train_votes_vec: Vec<u32> = vec![
        15, 10,
        10, 15,
        5, 12,
        30, 20,
        16, 12,
        13, 25,
        6, 14,
        31, 21,
    ];
    let n_train = train_votes_vec.len() / VOTE_DIM;
    let train_votes_tensor = LazyTensor::from_u32(
        train_votes_vec,
        Shape::from_dims(&[n_train, VOTE_DIM]),
        &dev,
    ).to_dtype(DType::F32)?;

    let train_results_vec: Vec<u32> = vec![
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
    ];
    let train_results_tensor = LazyTensor::from_u32(
        train_results_vec,
        Shape::from_dims(&[n_train]),
        &dev,
    );

    let test_votes_vec: Vec<u32> = vec![
        13, 9,
        8, 14,
        3, 10,
    ];
    let n_test = test_votes_vec.len() / VOTE_DIM;
    let test_votes_tensor = LazyTensor::from_u32(
        test_votes_vec,
        Shape::from_dims(&[n_test, VOTE_DIM]),
        &dev,
    ).to_dtype(DType::F32)?;

    let test_results_vec: Vec<u32> = vec![
        1,
        0,
        0,
    ];
    let test_results_tensor = LazyTensor::from_u32(
        test_results_vec,
        Shape::from_dims(&[n_test]),
        &dev,
    );

    let m = Dataset {
        train_votes: train_votes_tensor,
        train_results: train_results_tensor,
        test_votes: test_votes_tensor,
        test_results: test_results_tensor,
    };

    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            },
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        }

    }

    let real_world_votes: Vec<u32> = vec![
        13, 22,
    ];

    let tensor_test_votes = LazyTensor::from_u32(
        real_world_votes.clone(),
        Shape::from_dims(&[1, VOTE_DIM]),
        &dev,
    ).to_dtype(DType::F32)?;

    let final_result = trained_model.forward(&tensor_test_votes)?;

    let argmax_u32 = final_result.argmax_dim(D::Minus1)?.realize_u32();
    let result = argmax_u32[0] as f32;
    println!("real_life_votes: {:?}", real_world_votes);
    println!("neural_network_prediction_result: {:?}", result);

    Ok(())

}
// ANCHOR_END: book_training_simplified3

// ANCHOR: book_training_simplified2
fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    let _ = dev; // tensors are already on `dev`; lazy substrate has no to_device.
    let train_results = m.train_results.clone();
    let train_votes = m.train_votes.clone();
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, dev.clone());
    let model = MultiLevelPerceptron::new(&vs)?;
    let mut sgd = LazySgd::new(varmap.all_vars(), SgdConfig::new(LEARNING_RATE))?;
    let test_votes = m.test_votes.clone();
    let test_results = m.test_results.clone();
    let test_results_shape = test_results.shape();
    let n_test = test_results_shape.dims()[0];
    let mut final_accuracy: f32 = 0.0;
    for epoch in 1..EPOCHS + 1 {
        let logits = model.forward(&train_votes)?;
        let log_sm = logits.log_softmax(D::Minus1)?;
        let loss = nll(&log_sm, &train_results, Reduction::Mean)?;
        let loss_val = loss.realize_f32()[0];
        sgd.backward_step(&loss)?;

        let test_logits = model.forward(&test_votes)?;
        let predictions = test_logits.argmax_dim(D::Minus1)?;
        let matches = predictions.eq(&test_results)?;
        let sum_ok_vec = matches.to_dtype(DType::F32)?.sum_all().realize_f32();
        let sum_ok = sum_ok_vec[0];
        let test_accuracy = sum_ok / n_test as f32;
        final_accuracy = 100. * test_accuracy;
        println!("Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
                 loss_val,
                 final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}
// ANCHOR_END: book_training_simplified2


}

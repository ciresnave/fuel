# Fuel MNIST Tutorial

## Training Implementation

First, let's create a utility function `make_linear` that accepts a `LazyVarBuilder` and returns an initialized linear layer. The `LazyVarBuilder` writes into a `LazyVarMap`, which is the data structure that stores our trainable parameters.

```rust
use fuel_core::{DType, Device, Result};
use fuel_core::lazy::LazyTensor;
use fuel_core::lazy_nn::{LazyLinear, LazyModule, linear};
use fuel_core::lazy_nn_varbuilder::LazyVarBuilder;
use fuel_core::lazy_nn_varmap::LazyVarMap;

fn make_linear(vs: &LazyVarBuilder, in_dim: usize, out_dim: usize) -> Result<LazyLinear> {
    // The `linear` free factory registers `<prefix>.weight` and
    // `<prefix>.bias` into the builder's underlying VarMap with PyTorch's
    // Kaiming-fan-in uniform init (matches `nn.Linear`'s defaults).
    linear(in_dim, out_dim, vs)
}
```

Next, let's implement a `new` method for our model class to accept a `LazyVarBuilder` and initialize the model. We use `LazyVarBuilder::pp` to "push prefix" so that the parameter names are organized hierarchically: the first layer weights as `first.weight` and `first.bias`, and the second layer weights as `second.weight` and `second.bias`.

```rust
impl Model {
    fn new(vs: &LazyVarBuilder) -> Result<Self> {
        const IMAGE_DIM: usize = 784;
        const HIDDEN_DIM: usize = 100;
        const LABELS: usize = 10;

        let first = make_linear(&vs.pp("first"), IMAGE_DIM, HIDDEN_DIM)?;
        let second = make_linear(&vs.pp("second"), HIDDEN_DIM, LABELS)?;

        Ok(Self { first, second })
    }

    fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}
```

Now, let's add the `fuel-datasets` package to our project to access the MNIST dataset:

```bash
$ cargo add --git https://github.com/huggingface/fuel.git fuel-datasets
```

With the dataset available, we can implement our training loop:

```rust
use fuel_core::{DType, Device, Result, D};
use fuel_core::lazy::LazyTensor;
use fuel_core::lazy_nn::{LazyLinear, LazyModule, linear};
use fuel_core::lazy_nn_loss as loss;
use fuel_core::lazy_nn_optim::{LazyOptimizer, LazySgd};
use fuel_core::lazy_nn_varbuilder::LazyVarBuilder;
use fuel_core::lazy_nn_varmap::LazyVarMap;

fn training_loop(
    m: fuel_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    let dev = fuel::cuda_backend::device_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Initialize a LazyVarMap to store trainable parameters
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, dev.clone());
    let model = Model::new(&vs)?;

    let learning_rate = 0.05;
    let epochs = 10;

    // Initialize a stochastic gradient descent optimizer to update parameters
    let mut sgd = LazySgd::new(varmap.all_vars(), learning_rate)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for epoch in 1..epochs {
        // Perform forward pass on MNIST data
        let logits = model.forward(&train_images)?;
        let log_sm = logits.log_softmax(D::Minus1)?;

        // Compute Negative Log Likelihood loss
        let loss = loss::nll(&log_sm, &train_labels, loss::Reduction::Mean)?;

        // Backward + step in a single call.
        sgd.backward_step(&loss)?;

        // Evaluate model on test set
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            test_accuracy
        );
    }
    Ok(())
}
```

Finally, let's implement our main function:

```rust
pub fn main() -> anyhow::Result<()> {
    let m = fuel_datasets::vision::mnist::load()?;
    return training_loop(m);
}
```

Let's execute the training process:

```bash
$ cargo run --release

> 1 train loss:  2.35449 test acc:  0.12%
> 2 train loss:  2.30760 test acc:  0.15%
> ...
```
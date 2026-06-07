# Fuel MNIST Tutorial

## Modeling

Open `src/main.rs` in your project folder and insert the following code:

```rust
use fuel_core::{DType, Device, Result};
use fuel_core::lazy::LazyTensor;

struct Model {
    first: LazyTensor,
    second: LazyTensor,
}

impl Model {
    fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

fn main() -> Result<()> {
    // Use fuel::cuda_backend::new_device(0)?; to utilize GPU acceleration.
    let device = Device::Cpu;

    let first = LazyTensor::randn((784, 100), 0.0, 1.0, DType::F32, &device)?;
    let second = LazyTensor::randn((100, 10), 0.0, 1.0, DType::F32, &device)?;
    let model = Model { first, second };

    let dummy_image = LazyTensor::randn((1, 784), 0.0, 1.0, DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Execute the program with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

Since random inputs are provided, expect an incoherent output.

## Implementing a `Linear` Layer

To create a more sophisticated layer type, add a `bias` to the weight to construct the standard `Linear` layer.

Replace the entire content of `src/main.rs` with:

```rust
use fuel_core::{DType, Device, Result};
use fuel_core::lazy::LazyTensor;

struct Linear {
    weight: LazyTensor,
    bias: LazyTensor,
}

impl Linear {
    fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use fuel::cuda_backend::new_device(0)?; for GPU acceleration.
    // Use Device::Cpu; for CPU computation.
    let device = fuel::cuda_backend::device_if_available(0)?;

    // Initialize model parameters
    let weight = LazyTensor::randn((784, 100), 0.0, 1.0, DType::F32, &device)?;
    let bias = LazyTensor::randn((100,), 0.0, 1.0, DType::F32, &device)?;
    let first = Linear { weight, bias };
    let weight = LazyTensor::randn((100, 10), 0.0, 1.0, DType::F32, &device)?;
    let bias = LazyTensor::randn((10,), 0.0, 1.0, DType::F32, &device)?;
    let second = Linear { weight, bias };
    let model = Model { first, second };

    let dummy_image = LazyTensor::randn((1, 784), 0.0, 1.0, DType::F32, &device)?;

    // Perform inference
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Execute again with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

## Utilizing `fuel_core::lazy_nn`

Many classical layers (such as [`LazyLinear`](https://github.com/huggingface/fuel/blob/main/fuel-core/src/lazy_nn/linear.rs)) are already implemented in the lazy `nn` substrate that ships inside `fuel-core` itself.

Unlike the retired eager `fuel-nn::Linear` (which stored `[out, in]` and transposed inside `forward` to mirror PyTorch), `LazyLinear` stores its weight directly in `[in, out]` layout — the layout `LazyTensor::matmul` consumes — so there is no per-forward transpose. Parameters are registered into a `LazyVarMap` through a `LazyVarBuilder`, which also threads a default dtype and device through layer factories.

Let's simplify our implementation by leaning on the `lazy_nn::linear` free factory:

```rust
use fuel_core::{DType, Device, Result};
use fuel_core::lazy::LazyTensor;
use fuel_core::lazy_nn::{LazyLinear, LazyModule, linear};
use fuel_core::lazy_nn_varbuilder::LazyVarBuilder;
use fuel_core::lazy_nn_varmap::LazyVarMap;

struct Model {
    first: LazyLinear,
    second: LazyLinear,
}

impl Model {
    fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use fuel::cuda_backend::new_device(0)?; for GPU acceleration.
    let device = Device::Cpu;

    // The VarMap owns the parameters; the VarBuilder threads naming +
    // dtype + device through layer factories.
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, device.clone());

    // No transpose dance — `linear(in, out, vs)` registers the weight in
    // [in, out] layout directly.
    let first = linear(784, 100, &vs.pp("first"))?;
    let second = linear(100, 10, &vs.pp("second"))?;
    let model = Model { first, second };

    let dummy_image = LazyTensor::randn((1, 784), 0.0, 1.0, DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Execute the final version:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```
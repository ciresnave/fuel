# Hello world!

We will now create the hello world of the ML world, building a model capable of solving MNIST dataset.

Open `src/main.rs` and fill in this content:

```rust
# extern crate fuel_core;
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
    // Use fuel::cuda_backend::new_device(0)?; to use the GPU.
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

Everything should now run with:

```bash
cargo run --release
```

## Using a `Linear` layer.

Now that we have this, we might want to complexify things a bit, for instance by adding `bias` and creating
the classical `Linear` layer. We can do as such

```rust
# extern crate fuel_core;
# use fuel_core::{DType, Device, Result};
# use fuel_core::lazy::LazyTensor;
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
```

This will change the model running code into a new function

```rust
# extern crate fuel_core;
# use fuel_core::{DType, Device, Result};
# use fuel_core::lazy::LazyTensor;
# struct Linear {
#     weight: LazyTensor,
#     bias: LazyTensor,
# }
# impl Linear {
#     fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
#         let x = x.matmul(&self.weight)?;
#         x.broadcast_add(&self.bias)
#     }
# }
# 
# struct Model {
#     first: Linear,
#     second: Linear,
# }
# 
# impl Model {
#     fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
#         let x = self.first.forward(image)?;
#         let x = x.relu()?;
#         self.second.forward(&x)
#     }
# }
fn main() -> Result<()> {
    // Use fuel::cuda_backend::new_device(0)?; to use the GPU.
    // Use Device::Cpu; to use the CPU.
    let device = fuel::cuda_backend::device_if_available(0)?;

    // Creating a dummy model
    let weight = LazyTensor::randn((784, 100), 0.0, 1.0, DType::F32, &device)?;
    let bias = LazyTensor::randn((100,), 0.0, 1.0, DType::F32, &device)?;
    let first = Linear { weight, bias };
    let weight = LazyTensor::randn((100, 10), 0.0, 1.0, DType::F32, &device)?;
    let bias = LazyTensor::randn((10,), 0.0, 1.0, DType::F32, &device)?;
    let second = Linear { weight, bias };
    let model = Model { first, second };

    let dummy_image = LazyTensor::randn((1, 784), 0.0, 1.0, DType::F32, &device)?;

    // Inference on the model
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Now it works, it is a great way to create your own layers.
But most of the classical layers are already implemented in `fuel_core::lazy_nn` — the lazy neural-network substrate that sits on top of `LazyTensor`.

## Using `fuel_core::lazy_nn`.

For instance [`LazyLinear`](https://github.com/huggingface/fuel/blob/main/fuel-core/src/lazy_nn/linear.rs) is already there, along with the `lazy_nn::linear(in, out, vs)` free factory that registers the layer's weights into a `LazyVarMap` via a `LazyVarBuilder`. The weight is laid out `[in_features, out_features]`, matching the layout `LazyTensor::matmul` consumes directly — no transpose at forward time.

So instead we can simplify our example by leaning on the lazy `nn` factories:

```rust
# extern crate fuel_core;
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
    // Use fuel::cuda_backend::new_device(0)?; to use the GPU.
    let device = Device::Cpu;

    // The VarMap owns the parameters; the VarBuilder threads naming +
    // dtype + device through layer factories.
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, device.clone());

    let first = linear(784, 100, &vs.pp("first"))?;
    let second = linear(100, 10, &vs.pp("second"))?;
    let model = Model { first, second };

    let dummy_image = LazyTensor::randn((1, 784), 0.0, 1.0, DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Feel free to modify this example to use `Conv2d` to create a classical convnet instead.


Now that we have the running dummy code we can get to more advanced topics:

- [For PyTorch users](../guide/cheatsheet.md)
- [Running existing models](../inference/inference.md)
- [Training models](../training/training.md)



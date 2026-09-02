// SPDX-License-Identifier: MIT OR Apache-2.0
//! Activation modules wrapping the corresponding `Tensor` ops.

use super::Module;
use fuel_core::Result;
use fuel_core::lazy::Tensor;

macro_rules! activation_unit {
    ($name:ident, $method:ident) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;
        impl Module for $name {
            fn forward(&self, xs: &Tensor) -> Result<Tensor> {
                Ok(xs.$method())
            }
        }
    };
}

activation_unit!(Relu, relu);
activation_unit!(Gelu, gelu);
activation_unit!(Silu, silu);
activation_unit!(Sigmoid, sigmoid);
activation_unit!(Tanh, tanh);

/// GELU with the PyTorch `tanh`-approximation parameterization.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// `Tensor::gelu` is the tanh approximation; this is a named
/// alias that documents the intent at use sites that read HF
/// `hidden_act = "gelu_pytorch_tanh"`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GeluPytorchTanh;

impl Module for GeluPytorchTanh {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.gelu())
    }
}

/// LeakyReLU(x) = x if x >= 0 else negative_slope * x.
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu {
    pub negative_slope: f64,
}

impl LeakyRelu {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Module for LeakyRelu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let neg = xs.neg().relu().neg().mul_scalar(self.negative_slope);
        let pos = xs.relu();
        pos.add(&neg)
    }
}

/// ELU(x) = x if x >= 0 else alpha * (exp(x) - 1).
#[derive(Debug, Clone, Copy)]
pub struct Elu {
    pub alpha: f64,
}

impl Elu {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Module for Elu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let pos = xs.relu();
        // neg branch active when x < 0: alpha * (exp(x) - 1). For x >= 0
        // it would evaluate to alpha * (e^x - 1) too, so we mask via the
        // sign indicator (neg(x).relu() / |x|) and clamp by min(x, 0).
        // Simpler exact form: pos + alpha * (exp(min(x, 0)) - 1).
        let zero = xs.mul_scalar(0.0);
        let min_x_zero = {
            let diff = xs.sub(&zero)?;

            diff.neg().relu().neg()
        };
        let exp_min = min_x_zero.exp();
        let neg_branch = exp_min.add_scalar(-1.0).mul_scalar(self.alpha);
        pos.add(&neg_branch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core::Device;
    use fuel_ir::Shape;
    use std::sync::Arc;

    fn scalar_tensor(val: f32) -> Tensor {
        Tensor::from_f32(Arc::from(vec![val]), Shape::from_dims(&[1]), &Device::cpu())
    }

    fn first(t: Tensor) -> f32 {
        t.realize_f32()[0]
    }

    #[test]
    fn relu_clamps_negatives_to_zero() {
        assert_eq!(first(Relu.forward(&scalar_tensor(-2.0)).unwrap()), 0.0);
        assert_eq!(first(Relu.forward(&scalar_tensor(3.5)).unwrap()), 3.5);
    }

    #[test]
    fn gelu_at_zero_is_zero() {
        let got = first(Gelu.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-5, "got {got}");
    }

    #[test]
    fn silu_at_zero_is_zero() {
        let got = first(Silu.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-6, "got {got}");
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let got = first(Sigmoid.forward(&scalar_tensor(0.0)).unwrap());
        assert!((got - 0.5).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let got = first(Tanh.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-6, "got {got}");
    }

    #[test]
    fn leaky_relu_at_minus_one_with_slope_0_1_equals_minus_0_1() {
        let lru = LeakyRelu::new(0.1);
        let got = first(lru.forward(&scalar_tensor(-1.0)).unwrap());
        assert!((got - (-0.1_f32)).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn elu_at_positive_is_identity() {
        let elu = Elu::new(1.0);
        let got = first(elu.forward(&scalar_tensor(2.0)).unwrap());
        assert!((got - 2.0).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn elu_at_large_negative_approaches_minus_alpha() {
        let elu = Elu::new(1.0);
        // x = -10 → 1.0 * (e^-10 - 1) ≈ -0.9999546.
        let got = first(elu.forward(&scalar_tensor(-10.0)).unwrap());
        assert!((got - (-0.9999546_f32)).abs() < 1e-3, "got {got}");
    }
}

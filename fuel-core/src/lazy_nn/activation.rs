//! Activation modules wrapping the corresponding `LazyTensor` ops.

use super::LazyModule;
use crate::Result;
use crate::lazy::LazyTensor;

macro_rules! activation_unit {
    ($name:ident, $method:ident) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;
        impl LazyModule for $name {
            fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
                Ok(xs.$method())
            }
        }
    };
}

activation_unit!(LazyRelu, relu);
activation_unit!(LazyGelu, gelu);
activation_unit!(LazySilu, silu);
activation_unit!(LazySigmoid, sigmoid);
activation_unit!(LazyTanh, tanh);

/// GELU with the PyTorch `tanh`-approximation parameterization.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
/// `LazyTensor::gelu` is the tanh approximation; this is a named
/// alias that documents the intent at use sites that read HF
/// `hidden_act = "gelu_pytorch_tanh"`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LazyGeluPytorchTanh;

impl LazyModule for LazyGeluPytorchTanh {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        Ok(xs.gelu())
    }
}

/// LeakyReLU(x) = x if x >= 0 else negative_slope * x.
#[derive(Debug, Clone, Copy)]
pub struct LazyLeakyRelu {
    pub negative_slope: f64,
}

impl LazyLeakyRelu {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl LazyModule for LazyLeakyRelu {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let neg = xs.neg().relu().neg().mul_scalar(self.negative_slope);
        let pos = xs.relu();
        pos.add(&neg)
    }
}

/// ELU(x) = x if x >= 0 else alpha * (exp(x) - 1).
#[derive(Debug, Clone, Copy)]
pub struct LazyElu {
    pub alpha: f64,
}

impl LazyElu {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl LazyModule for LazyElu {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let pos = xs.relu();
        // neg branch active when x < 0: alpha * (exp(x) - 1). For x >= 0
        // it would evaluate to alpha * (e^x - 1) too, so we mask via the
        // sign indicator (neg(x).relu() / |x|) and clamp by min(x, 0).
        // Simpler exact form: pos + alpha * (exp(min(x, 0)) - 1).
        let zero = xs.mul_scalar(0.0);
        let min_x_zero = {
            let diff = xs.sub(&zero)?;
            let neg_part = diff.neg().relu().neg();
            neg_part
        };
        let exp_min = min_x_zero.exp();
        let neg_branch = exp_min.add_scalar(-1.0).mul_scalar(self.alpha);
        pos.add(&neg_branch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use fuel_ir::Shape;
    use std::sync::Arc;

    fn scalar_tensor(val: f32) -> LazyTensor {
        LazyTensor::from_f32(
            Arc::from(vec![val]),
            Shape::from_dims(&[1]),
            &Device::cpu(),
        )
    }

    fn first(t: LazyTensor) -> f32 {
        t.realize_f32()[0]
    }

    #[test]
    fn relu_clamps_negatives_to_zero() {
        assert_eq!(first(LazyRelu.forward(&scalar_tensor(-2.0)).unwrap()), 0.0);
        assert_eq!(first(LazyRelu.forward(&scalar_tensor(3.5)).unwrap()), 3.5);
    }

    #[test]
    fn gelu_at_zero_is_zero() {
        let got = first(LazyGelu.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-5, "got {got}");
    }

    #[test]
    fn silu_at_zero_is_zero() {
        let got = first(LazySilu.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-6, "got {got}");
    }

    #[test]
    fn sigmoid_at_zero_is_half() {
        let got = first(LazySigmoid.forward(&scalar_tensor(0.0)).unwrap());
        assert!((got - 0.5).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn tanh_at_zero_is_zero() {
        let got = first(LazyTanh.forward(&scalar_tensor(0.0)).unwrap());
        assert!(got.abs() < 1e-6, "got {got}");
    }

    #[test]
    fn leaky_relu_at_minus_one_with_slope_0_1_equals_minus_0_1() {
        let lru = LazyLeakyRelu::new(0.1);
        let got = first(lru.forward(&scalar_tensor(-1.0)).unwrap());
        assert!((got - (-0.1_f32)).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn elu_at_positive_is_identity() {
        let elu = LazyElu::new(1.0);
        let got = first(elu.forward(&scalar_tensor(2.0)).unwrap());
        assert!((got - 2.0).abs() < 1e-5, "got {got}");
    }

    #[test]
    fn elu_at_large_negative_approaches_minus_alpha() {
        let elu = LazyElu::new(1.0);
        // x = -10 → 1.0 * (e^-10 - 1) ≈ -0.9999546.
        let got = first(elu.forward(&scalar_tensor(-10.0)).unwrap());
        assert!((got - (-0.9999546_f32)).abs() < 1e-3, "got {got}");
    }
}

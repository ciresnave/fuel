//! `LazySequential` — a Vec of boxed [`LazyModule`]s applied left-to-right.

use super::LazyModule;
use crate::Result;
use crate::lazy::LazyTensor;

/// A container that chains owned [`LazyModule`]s.
///
/// `forward(xs)` runs `xs` through each contained module in order,
/// passing the previous module's output as the next module's input.
pub struct LazySequential {
    layers: Vec<Box<dyn LazyModule + Send + Sync>>,
}

impl LazySequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push<M: LazyModule + Send + Sync + 'static>(mut self, module: M) -> Self {
        self.layers.push(Box::new(module));
        self
    }

    pub fn add<M: LazyModule + Send + Sync + 'static>(&mut self, module: M) {
        self.layers.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Default for LazySequential {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyModule for LazySequential {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        if self.layers.is_empty() {
            return Ok(xs.clone());
        }
        let mut out = self.layers[0].forward(xs)?;
        for layer in self.layers.iter().skip(1) {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::lazy::WeightStorage;
    use crate::lazy_nn::{LazyLinear, LazyModule};
    use fuel_core_types::Shape;
    use std::sync::Arc;

    fn tiny_xs(b: usize, d: usize, val: f32) -> LazyTensor {
        let data: Vec<f32> = vec![val; b * d];
        LazyTensor::from_f32(
            Arc::from(data),
            Shape::from_dims(&[b, d]),
            &Device::cpu(),
        )
    }

    #[test]
    fn sequential_chains_two_linears_matches_manual_composition() {
        // L1: (4 -> 6), L2: (6 -> 3). Manual composition: L2(L1(x)).
        let w1: Arc<[f32]> = Arc::from((0..24).map(|i| (i as f32) * 0.01 - 0.1).collect::<Vec<_>>());
        let w2: Arc<[f32]> = Arc::from((0..18).map(|i| 0.02 + (i as f32) * 0.005).collect::<Vec<_>>());
        let l1 = LazyLinear::new(WeightStorage::F32(Arc::clone(&w1)), None, 4, 6).unwrap();
        let l2 = LazyLinear::new(WeightStorage::F32(Arc::clone(&w2)), None, 6, 3).unwrap();

        let xs = tiny_xs(1, 4, 0.25);
        let manual = l2.forward(&l1.forward(&xs).unwrap()).unwrap().realize_f32();

        let seq = LazySequential::new()
            .push(LazyLinear::new(WeightStorage::F32(Arc::clone(&w1)), None, 4, 6).unwrap())
            .push(LazyLinear::new(WeightStorage::F32(Arc::clone(&w2)), None, 6, 3).unwrap());
        let chained = seq.forward(&xs).unwrap().realize_f32();
        assert_eq!(manual.len(), chained.len());
        for (i, (a, b)) in manual.iter().zip(chained.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "i={i} manual={a} chained={b}");
        }
    }

    #[test]
    fn empty_sequential_is_identity() {
        let xs = tiny_xs(1, 3, 0.5);
        let seq = LazySequential::new();
        let out = seq.forward(&xs).unwrap().realize_f32();
        let baseline = xs.realize_f32();
        assert_eq!(out, baseline);
    }
}

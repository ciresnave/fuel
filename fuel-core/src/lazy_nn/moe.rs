//! Lazy Mixture-of-Experts router + experts + layer.
//!
//! A standard top-K MoE layer over `LazyTensor`. Tokens are routed
//! through a linear `[hidden, num_experts]` projection; the top-K
//! experts per token are selected and weighted by a softmax over the
//! K picked logits. Each expert is a SwiGLU FFN
//! (`down(silu(gate(x)) * up(x))`). The layer's output is the
//! gating-weighted sum of the expert outputs.
//!
//! Top-K selection runs entirely inside the graph: K iterations of
//! `argmax_dim` along the expert axis, each followed by a
//! `scatter_add` of `-1e30` to mask the previously picked entries.
//! Dispatch enumerates experts (not tokens) — every expert FFN is
//! computed once for every token, then gated by the per-token weight
//! that expert earned through the router. Per-expert weights come
//! from a dense `[N, num_experts]` matrix built by `scatter_add` from
//! the (sparse) `[N, top_k]` router output; the column for expert `e`
//! is the gating coefficient applied to that expert's output. With
//! the typical small `num_experts` (4–16) this is the cheapest
//! dense-graph formulation.

use crate::Result;
use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_nn::{LazyLinear, LazyModule};
use fuel_ir::{DynScalar, Shape};
use std::sync::Arc;

const MASK_NEG: f32 = -1.0e30;

/// Top-K router: projects `[*, hidden]` -> expert logits and picks
/// the top-K experts per token along with their gating weights
/// (softmax over the picked logits only).
#[derive(Debug, Clone)]
pub struct LazyMoeRouter {
    weight: WeightStorage,
    num_experts: usize,
    top_k: usize,
    hidden_size: usize,
    jitter_noise: f64,
}

impl LazyMoeRouter {
    /// Build a router. `weight` is laid out `[hidden_size, num_experts]`
    /// (the convention `WeightStorage::apply_linear` expects). `top_k`
    /// must be in `1..=num_experts`. `jitter_noise` is stored for
    /// future training-time multiplicative noise; v1 does not inject
    /// any noise into the graph (deterministic forward).
    pub fn new(
        weight: WeightStorage,
        num_experts: usize,
        top_k: usize,
        hidden_size: usize,
        jitter_noise: f64,
    ) -> Result<Self> {
        if num_experts == 0 {
            return Err(crate::Error::Msg(
                "LazyMoeRouter::new: num_experts must be > 0".into(),
            ).bt());
        }
        if top_k == 0 || top_k > num_experts {
            return Err(crate::Error::Msg(format!(
                "LazyMoeRouter::new: top_k must be in 1..={num_experts}, got {top_k}",
            )).bt());
        }
        if weight.elem_count() != hidden_size * num_experts {
            return Err(crate::Error::Msg(format!(
                "LazyMoeRouter::new: weight has {} elements but \
                 hidden_size * num_experts = {} * {} = {}",
                weight.elem_count(),
                hidden_size,
                num_experts,
                hidden_size * num_experts,
            )).bt());
        }
        Ok(Self { weight, num_experts, top_k, hidden_size, jitter_noise })
    }

    pub fn num_experts(&self) -> usize { self.num_experts }
    pub fn top_k(&self) -> usize { self.top_k }
    pub fn hidden_size(&self) -> usize { self.hidden_size }
    pub fn jitter_noise(&self) -> f64 { self.jitter_noise }
    pub fn weight(&self) -> &WeightStorage { &self.weight }

    /// Route `xs: [*, hidden_size]` to the top-K experts.
    ///
    /// Returns `(indices, weights)` where both are `[N, top_k]` with
    /// `N = prod(leading_dims)`. `indices` is `U32` (expert ids);
    /// `weights` is `F32` and sums to 1 along the last dim (softmax
    /// over the picked logits).
    pub fn route(&self, xs: &LazyTensor) -> Result<(LazyTensor, LazyTensor)> {
        let dims = xs.shape().dims().to_vec();
        if dims.is_empty() || *dims.last().unwrap() != self.hidden_size {
            return Err(crate::Error::Msg(format!(
                "LazyMoeRouter::route: input last dim must be {}, got shape {:?}",
                self.hidden_size, dims,
            )).bt());
        }
        let n: usize = dims[..dims.len() - 1].iter().product();
        let xs_flat = xs.reshape(Shape::from_dims(&[n, self.hidden_size]))?;
        let logits = self.weight.apply_linear(
            &xs_flat, self.hidden_size, self.num_experts,
        );

        let mut work = logits;
        let mut idx_cols: Vec<LazyTensor> = Vec::with_capacity(self.top_k);
        let mut logit_cols: Vec<LazyTensor> = Vec::with_capacity(self.top_k);
        for _ in 0..self.top_k {
            let idx = work.argmax_dim(1usize)?;
            let idx_col = idx.unsqueeze(1usize)?;
            let picked = work.gather(1usize, &idx_col)?;
            let neg_inf = idx_col.const_f32_like(
                Arc::from(vec![MASK_NEG; n]),
                Shape::from_dims(&[n, 1]),
            );
            work = work.scatter_add(1usize, &idx_col, &neg_inf)?;
            idx_cols.push(idx_col);
            logit_cols.push(picked);
        }
        let mut idx_acc = idx_cols[0].clone();
        let mut logit_acc = logit_cols[0].clone();
        for k in 1..self.top_k {
            idx_acc = idx_acc.concat(&idx_cols[k], 1usize)?;
            logit_acc = logit_acc.concat(&logit_cols[k], 1usize)?;
        }
        let weights = logit_acc.softmax_last_dim()?;
        Ok((idx_acc, weights))
    }
}

/// SwiGLU FFN expert: `down(silu(gate(x)) * up(x))`.
#[derive(Debug, Clone)]
pub struct LazyMoeExpert {
    gate: LazyLinear,
    up: LazyLinear,
    down: LazyLinear,
    hidden_size: usize,
    intermediate_size: usize,
}

impl LazyMoeExpert {
    /// Build an expert from three [`LazyLinear`] projections. `gate`
    /// and `up` must both map `hidden_size -> intermediate_size`;
    /// `down` must map `intermediate_size -> hidden_size`.
    pub fn new(
        gate: LazyLinear,
        up: LazyLinear,
        down: LazyLinear,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if gate.in_features() != hidden_size
            || gate.out_features() != intermediate_size
        {
            return Err(crate::Error::Msg(format!(
                "LazyMoeExpert::new: gate must be ({hidden_size}, {intermediate_size}), \
                 got ({}, {})", gate.in_features(), gate.out_features(),
            )).bt());
        }
        if up.in_features() != hidden_size
            || up.out_features() != intermediate_size
        {
            return Err(crate::Error::Msg(format!(
                "LazyMoeExpert::new: up must be ({hidden_size}, {intermediate_size}), \
                 got ({}, {})", up.in_features(), up.out_features(),
            )).bt());
        }
        if down.in_features() != intermediate_size
            || down.out_features() != hidden_size
        {
            return Err(crate::Error::Msg(format!(
                "LazyMoeExpert::new: down must be ({intermediate_size}, {hidden_size}), \
                 got ({}, {})", down.in_features(), down.out_features(),
            )).bt());
        }
        Ok(Self { gate, up, down, hidden_size, intermediate_size })
    }

    pub fn hidden_size(&self) -> usize { self.hidden_size }
    pub fn intermediate_size(&self) -> usize { self.intermediate_size }
    pub fn gate(&self) -> &LazyLinear { &self.gate }
    pub fn up(&self) -> &LazyLinear { &self.up }
    pub fn down(&self) -> &LazyLinear { &self.down }

    /// Forward `xs: [*, hidden]` through the SwiGLU FFN.
    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let g = self.gate.forward(xs)?.silu();
        let u = self.up.forward(xs)?;
        let h = g.mul(&u)?;
        self.down.forward(&h)
    }

    /// Data-determined-M forward for sparse MoE dispatch: run the SwiGLU
    /// FFN over a `[capacity, hidden]` buffer whose first `count` rows are
    /// the gathered routed tokens, computing **exactly** `count` rows of
    /// each of the three projections via [`LazyTensor::matmul_dyn_m`] (the
    /// FLOP saving). The three matmuls each zero their capacity tail, so —
    /// bias being absent — the whole `[capacity, hidden]` result stays zero
    /// past row `count`, which the layer relies on for a harmless
    /// `index_add` scatter-back of the padding rows.
    ///
    /// F32-only, and the expert must be **bias-free**: a bias would fill
    /// the un-computed tail and corrupt the scatter-back. Both surface as
    /// typed build-time errors.
    pub fn forward_dyn_m(
        &self,
        xs: &LazyTensor,
        count: fuel_ir::DynScalar,
    ) -> Result<LazyTensor> {
        for (name, lin) in [
            ("gate", &self.gate),
            ("up", &self.up),
            ("down", &self.down),
        ] {
            if lin.bias().is_some() {
                return Err(crate::Error::Msg(format!(
                    "LazyMoeExpert::forward_dyn_m: sparse dispatch requires bias-free \
                     experts, but the {name} projection has a bias (it would \
                     contaminate the un-computed capacity tail)",
                )).bt());
            }
        }
        let g = self
            .gate
            .weight()
            .apply_linear_dyn_m(xs, self.hidden_size, self.intermediate_size, count)?
            .silu();
        let u = self
            .up
            .weight()
            .apply_linear_dyn_m(xs, self.hidden_size, self.intermediate_size, count)?;
        let h = g.mul(&u)?;
        self.down
            .weight()
            .apply_linear_dyn_m(&h, self.intermediate_size, self.hidden_size, count)
    }
}

impl LazyModule for LazyMoeExpert {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        LazyMoeExpert::forward(self, xs)
    }
}

/// Mixture-of-Experts layer: router + per-expert SwiGLU FFNs.
#[derive(Debug, Clone)]
pub struct LazyMoeLayer {
    router: LazyMoeRouter,
    experts: Vec<LazyMoeExpert>,
}

impl LazyMoeLayer {
    /// Build a layer. `experts.len()` must equal `router.num_experts()`
    /// and every expert's `hidden_size` must match the router's.
    pub fn new(router: LazyMoeRouter, experts: Vec<LazyMoeExpert>) -> Result<Self> {
        if experts.len() != router.num_experts() {
            return Err(crate::Error::Msg(format!(
                "LazyMoeLayer::new: experts.len() = {} but router.num_experts = {}",
                experts.len(), router.num_experts(),
            )).bt());
        }
        for (i, e) in experts.iter().enumerate() {
            if e.hidden_size() != router.hidden_size() {
                return Err(crate::Error::Msg(format!(
                    "LazyMoeLayer::new: expert {i} hidden_size = {} but router \
                     hidden_size = {}", e.hidden_size(), router.hidden_size(),
                )).bt());
            }
        }
        Ok(Self { router, experts })
    }

    pub fn router(&self) -> &LazyMoeRouter { &self.router }
    pub fn experts(&self) -> &[LazyMoeExpert] { &self.experts }

    /// Forward `xs: [*, hidden]` through the MoE layer; output has
    /// the same shape as the input.
    ///
    /// **Sparse (dropless) dispatch.** Each expert's FFN is computed only
    /// for the tokens the router sent to it, not for all `N` tokens: per
    /// expert `e` we take the gate-weight column (nonzero exactly at the
    /// routed tokens), find those token rows with [`Op::NonZeroIndices`]
    /// (`LazyTensor::nonzero_indices_bundled`) — which also publishes the
    /// runtime count `count_e` into the pass's `SymEnv` — gather those rows
    /// into a `[capacity=N, hidden]` buffer, run the SwiGLU FFN over exactly
    /// `count_e` rows via [`LazyMoeExpert::forward_dyn_m`], scale each row by
    /// its gate weight, and scatter-add back to the token positions with
    /// [`LazyTensor::index_add`]. This is bit-exact to the dense
    /// enumerate-all path ([`Self::forward_dense`]) — the per-token FFN is
    /// row-independent, so gathering doesn't change any dot product — while
    /// cutting the FFN matmul FLOPs from `N·num_experts` token-rows to `N·top_k`.
    ///
    /// The FLOP saving needs the F32 [`LazyTensor::matmul_dyn_m`] path, so
    /// experts must be F32 and bias-free (see [`LazyMoeExpert::forward_dyn_m`]);
    /// otherwise a typed build-time error surfaces. Callers wanting the
    /// unconditional dense formulation can use [`Self::forward_dense`].
    pub fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let dims = xs.shape().dims().to_vec();
        let hidden = self.router.hidden_size();
        if dims.is_empty() || *dims.last().unwrap() != hidden {
            return Err(crate::Error::Msg(format!(
                "LazyMoeLayer::forward: input last dim must be {hidden}, got shape {dims:?}",
            )).bt());
        }
        let n: usize = dims[..dims.len() - 1].iter().product();
        let num_experts = self.router.num_experts();
        let xs_flat = xs.reshape(Shape::from_dims(&[n, hidden]))?;
        let (indices, weights) = self.router.route(xs)?;

        // Dense per-(token, expert) gate weights: column `e` is the gating
        // coefficient token `t` applies to expert `e`, and is exactly 0.0
        // wherever `t` did NOT route to `e` (softmax weights are strictly
        // positive) — so a nonzero in that column marks a routed token.
        let dense_zero = xs_flat.const_f32_like(
            Arc::from(vec![0.0_f32; n * num_experts]),
            Shape::from_dims(&[n, num_experts]),
        );
        let dense_weights = dense_zero.scatter_add(1usize, &indices, &weights)?;

        let mut acc = xs_flat.const_f32_like(
            Arc::from(vec![0.0_f32; n * hidden]),
            Shape::from_dims(&[n, hidden]),
        );
        for (e, expert) in self.experts.iter().enumerate() {
            // Gate-weight column for expert e: [N, 1], nonzero exactly at
            // the tokens routed to e. `nonzero_indices_bundled` flattens it
            // ([N,1] → N flat positions == token rows) into `sel[..count_e]`.
            let gate_col = dense_weights.narrow(1usize, e, 1usize)?; // [N, 1]
            // Fresh, graph-scoped count sym per expert (re-scanning the
            // graph each time yields a strictly higher id than the prior
            // expert's producer — and higher than any earlier stacked
            // layer's, so nothing collides).
            let count_sym = xs_flat.fresh_data_determined_sym();
            let (sel, _count) = gate_col.nonzero_indices_bundled(count_sym)?; // sel [N]
            let count = DynScalar::Sym(count_sym);

            // Gather this expert's routed token hidden states into the
            // capacity-buffer prefix, run its FFN over exactly count_e rows.
            let gathered = xs_flat.index_select(0usize, &sel)?; // [N, hidden]
            let ffn = expert.forward_dyn_m(&gathered, count)?; // [N, hidden]; tail zero
            // Scale each gathered row by its gate weight (same gather order).
            let gate_g = gate_col.index_select(0usize, &sel)?; // [N, 1]
            let scaled = ffn.broadcast_mul(&gate_g)?; // [N, hidden]; tail zero
            // Scatter-add back to token positions. The padding tail
            // (sel==0, scaled==+0.0) adds a harmless +0.0 to row 0.
            acc = acc.index_add(0usize, &sel, &scaled)?;
        }
        acc.reshape(Shape::from_dims(&dims))
    }

    /// The dense (enumerate-all-experts) formulation of [`Self::forward`]:
    /// compute every expert's FFN on **all** `N` tokens, then take the
    /// gating-weighted sum. Kept as the reference the sparse `forward`
    /// matches bit-for-bit, and as a fallback for weight encodings the
    /// sparse F32 [`LazyTensor::matmul_dyn_m`] path does not cover.
    pub fn forward_dense(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        let dims = xs.shape().dims().to_vec();
        let hidden = self.router.hidden_size();
        if dims.is_empty() || *dims.last().unwrap() != hidden {
            return Err(crate::Error::Msg(format!(
                "LazyMoeLayer::forward_dense: input last dim must be {hidden}, got shape {dims:?}",
            )).bt());
        }
        let n: usize = dims[..dims.len() - 1].iter().product();
        let num_experts = self.router.num_experts();
        let xs_flat = xs.reshape(Shape::from_dims(&[n, hidden]))?;
        let (indices, weights) = self.router.route(xs)?;

        let dense_zero = xs_flat.const_f32_like(
            Arc::from(vec![0.0_f32; n * num_experts]),
            Shape::from_dims(&[n, num_experts]),
        );
        let dense_weights = dense_zero.scatter_add(1usize, &indices, &weights)?;

        let mut acc = xs_flat.const_f32_like(
            Arc::from(vec![0.0_f32; n * hidden]),
            Shape::from_dims(&[n, hidden]),
        );
        for (e, expert) in self.experts.iter().enumerate() {
            let exp_out = expert.forward(&xs_flat)?;
            let col = dense_weights.narrow(1usize, e, 1usize)?;
            let weighted = exp_out.broadcast_mul(&col)?;
            acc = acc.add(&weighted)?;
        }
        acc.reshape(Shape::from_dims(&dims))
    }
}

impl LazyModule for LazyMoeLayer {
    fn forward(&self, xs: &LazyTensor) -> Result<LazyTensor> {
        LazyMoeLayer::forward(self, xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn ramp_f32(n: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * scale + offset).collect()
    }

    fn make_linear(
        in_f: usize, out_f: usize, scale: f32, offset: f32,
    ) -> LazyLinear {
        let w: Vec<f32> = ramp_f32(in_f * out_f, scale, offset);
        LazyLinear::new(
            WeightStorage::F32(Arc::from(w)),
            None,
            in_f,
            out_f,
        ).unwrap()
    }

    fn make_expert(hidden: usize, inter: usize, seed: f32) -> LazyMoeExpert {
        let gate = make_linear(hidden, inter, 0.03, seed);
        let up   = make_linear(hidden, inter, 0.04, seed + 0.1);
        let down = make_linear(inter, hidden, 0.02, seed + 0.2);
        LazyMoeExpert::new(gate, up, down, hidden, inter).unwrap()
    }

    #[test]
    fn moe_router_outputs_top_k_indices_and_weights_sum_to_one() {
        let hidden = 4;
        let num_experts = 5;
        let top_k = 3;
        let n = 6;

        let w: Vec<f32> = ramp_f32(hidden * num_experts, 0.05, -0.3);
        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(w)),
            num_experts, top_k, hidden, 0.0,
        ).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.07, -0.4);
        let xs = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );
        let (idx, w_out) = router.route(&xs).unwrap();
        assert_eq!(idx.shape().dims(), &[n, top_k]);
        assert_eq!(w_out.shape().dims(), &[n, top_k]);
        assert_eq!(idx.dtype(), crate::DType::U32);

        let idx_v = idx.realize_u32();
        let w_v = w_out.realize_f32();
        assert_eq!(idx_v.len(), n * top_k);
        assert_eq!(w_v.len(), n * top_k);
        for t in 0..n {
            let row_w = &w_v[t * top_k..(t + 1) * top_k];
            let s: f32 = row_w.iter().copied().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {t}: weights {row_w:?} should sum to 1, got {s}",
            );
            for v in row_w {
                assert!(v.is_finite() && *v >= 0.0,
                    "row {t}: weight {v} not finite or negative");
            }
            let row_idx = &idx_v[t * top_k..(t + 1) * top_k];
            for v in row_idx {
                assert!((*v as usize) < num_experts,
                    "row {t}: idx {v} out of range");
            }
            let mut sorted = row_idx.to_vec();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), top_k,
                "row {t}: idx {row_idx:?} has duplicates");
        }
    }

    #[test]
    fn moe_layer_forward_shape_matches_input() {
        let hidden = 4;
        let inter = 6;
        let num_experts = 4;
        let top_k = 2;
        let n = 5;

        let w: Vec<f32> = ramp_f32(hidden * num_experts, 0.03, -0.1);
        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(w)),
            num_experts, top_k, hidden, 0.0,
        ).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|i| make_expert(hidden, inter, i as f32 * 0.15))
            .collect();
        let layer = LazyMoeLayer::new(router, experts).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.05, -0.2);
        let xs = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );
        let y = layer.forward(&xs).unwrap();
        assert_eq!(y.shape().dims(), &[n, hidden]);
        let got = y.realize_f32();
        assert_eq!(got.len(), n * hidden);
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} not finite");
        }
    }

    /// The Increment-B acceptance test: the sparse (dropless) `forward`
    /// must be **bit-exact** to the dense enumerate-all `forward_dense` on
    /// a small top-k MoE. The per-token FFN is row-independent, so gathering
    /// the routed tokens changes no dot product; the gate-weighted
    /// accumulation happens in the same expert order in both paths (the
    /// sparse path's untouched tokens add exactly +0.0, matching dense's
    /// multiply-by-zero contribution). Any mismatch means the sparse
    /// dispatch dropped, double-counted, or mis-scaled a token.
    #[test]
    fn moe_layer_sparse_forward_bit_exact_to_dense() {
        let hidden = 4;
        let inter = 6;
        let num_experts = 4;
        let top_k = 2;
        let n = 5;

        let w: Vec<f32> = ramp_f32(hidden * num_experts, 0.03, -0.1);
        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(w)),
            num_experts, top_k, hidden, 0.0,
        ).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|i| make_expert(hidden, inter, i as f32 * 0.15))
            .collect();
        let layer = LazyMoeLayer::new(router, experts).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.05, -0.2);
        let xs_sparse = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );
        let xs_dense = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );

        let sparse = layer.forward(&xs_sparse).unwrap().realize_f32();
        let dense = layer.forward_dense(&xs_dense).unwrap().realize_f32();

        assert_eq!(sparse.len(), dense.len());
        for (i, (s, d)) in sparse.iter().zip(dense.iter()).enumerate() {
            assert_eq!(
                s.to_bits(), d.to_bits(),
                "out[{i}]: sparse {s} != dense {d} (bit-exact required)",
            );
        }
    }

    /// A 3-D input `[batch, seq, hidden]` must route correctly (the layer
    /// flattens the leading dims) and stay bit-exact to dense — guards the
    /// gather/scatter row indexing against the `N = batch·seq` flattening.
    #[test]
    fn moe_layer_sparse_3d_input_bit_exact_to_dense() {
        let hidden = 3;
        let inter = 5;
        let num_experts = 5;
        let top_k = 2;
        let (batch, seq) = (2, 3);
        let n = batch * seq;

        let w: Vec<f32> = ramp_f32(hidden * num_experts, 0.04, -0.2);
        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(w)),
            num_experts, top_k, hidden, 0.0,
        ).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|i| make_expert(hidden, inter, i as f32 * 0.11))
            .collect();
        let layer = LazyMoeLayer::new(router, experts).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.06, -0.3);
        let xs_s = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );
        let xs_d = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[batch, seq, hidden]), &Device::cpu(),
        );

        let sparse = layer.forward(&xs_s).unwrap();
        assert_eq!(sparse.shape().dims(), &[batch, seq, hidden]);
        let sparse = sparse.realize_f32();
        let dense = layer.forward_dense(&xs_d).unwrap().realize_f32();
        for (i, (s, d)) in sparse.iter().zip(dense.iter()).enumerate() {
            assert_eq!(
                s.to_bits(), d.to_bits(),
                "out[{i}]: sparse {s} != dense {d} (bit-exact required)",
            );
        }
    }

    /// Bias-carrying experts can't use the sparse tail-zeroing path — the
    /// layer must surface a typed build-time error rather than silently
    /// corrupting the scatter-back.
    #[test]
    fn moe_layer_sparse_forward_rejects_biased_experts() {
        let hidden = 3;
        let inter = 4;
        let num_experts = 2;

        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(vec![0.1_f32; hidden * num_experts])),
            num_experts, 1, hidden, 0.0,
        ).unwrap();
        // A down projection WITH a bias.
        let biased_down = LazyLinear::new(
            WeightStorage::F32(Arc::from(ramp_f32(inter * hidden, 0.02, 0.1))),
            Some(Arc::from(vec![0.5_f32; hidden])),
            inter, hidden,
        ).unwrap();
        let gate = make_linear(hidden, inter, 0.03, 0.0);
        let up = make_linear(hidden, inter, 0.04, 0.1);
        let biased_expert = LazyMoeExpert::new(gate, up, biased_down, hidden, inter).unwrap();
        let plain_expert = make_expert(hidden, inter, 0.2);
        let layer = LazyMoeLayer::new(
            router, vec![biased_expert, plain_expert],
        ).unwrap();

        let xs = LazyTensor::from_f32(
            ramp_f32(2 * hidden, 0.05, -0.1), Shape::from_dims(&[2, hidden]), &Device::cpu(),
        );
        let err = layer.forward(&xs).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("bias-free"),
            "expected a bias-free-requirement error, got: {msg}",
        );
    }

    #[test]
    fn moe_layer_single_expert_top_1_equals_expert_forward_directly() {
        let hidden = 4;
        let inter = 5;
        let n = 3;

        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(vec![0.5_f32; hidden])),
            1, 1, hidden, 0.0,
        ).unwrap();
        let expert = make_expert(hidden, inter, 0.0);
        let layer = LazyMoeLayer::new(router, vec![expert.clone()]).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.06, -0.25);
        let xs1 = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );
        let xs2 = LazyTensor::from_f32(
            x_data, Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );

        let layer_out = layer.forward(&xs1).unwrap().realize_f32();
        let direct_out = expert.forward(&xs2).unwrap().realize_f32();

        assert_eq!(layer_out.len(), direct_out.len());
        for (i, (a, e)) in layer_out.iter().zip(direct_out.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "out[{i}] expected {e}, got {a}",
            );
        }
    }

    #[test]
    fn moe_layer_with_uniform_router_averages_experts() {
        let hidden = 3;
        let inter = 4;
        let num_experts = 3;
        let top_k = num_experts;
        let n = 2;

        let router = LazyMoeRouter::new(
            WeightStorage::F32(Arc::from(vec![0.0_f32; hidden * num_experts])),
            num_experts, top_k, hidden, 0.0,
        ).unwrap();
        let experts: Vec<_> = (0..num_experts)
            .map(|i| make_expert(hidden, inter, i as f32 * 0.3))
            .collect();
        let layer = LazyMoeLayer::new(router, experts.clone()).unwrap();

        let x_data: Vec<f32> = ramp_f32(n * hidden, 0.08, -0.1);
        let xs_layer = LazyTensor::from_f32(
            x_data.clone(), Shape::from_dims(&[n, hidden]), &Device::cpu(),
        );
        let layer_out = layer.forward(&xs_layer).unwrap().realize_f32();

        let mut expected = vec![0.0_f32; n * hidden];
        for e in &experts {
            let xs_e = LazyTensor::from_f32(
                x_data.clone(),
                Shape::from_dims(&[n, hidden]),
                &Device::cpu(),
            );
            let out = e.forward(&xs_e).unwrap().realize_f32();
            for (i, v) in out.iter().enumerate() {
                expected[i] += v;
            }
        }
        let inv_k = 1.0 / (num_experts as f32);
        for v in expected.iter_mut() { *v *= inv_k; }

        assert_eq!(layer_out.len(), expected.len());
        for (i, (a, e)) in layer_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-4,
                "out[{i}] expected {e}, got {a}",
            );
        }
    }
}

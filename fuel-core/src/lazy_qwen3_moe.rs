//! Qwen3-MoE decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Qwen3-MoE = Qwen3 attention (per-head QK-norm
//! + per-layer sliding-window gating + optional Q/K/V/O biases) +
//! per-layer FFN alternation between a dense SwiGLU MLP and a
//! Mixtral-style sparse MoE. `decoder_sparse_step` controls the
//! cadence: layer `i` uses MoE when `(i + 1) % decoder_sparse_step
//! == 0`; other layers run a single SwiGLU.
//!
//! v1 uses **dense routing** for the MoE layers (every expert
//! evaluated, weighted by full router softmax) — same trade-off
//! as Mixtral. No shared expert.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Layer `i` uses MoE iff `(i + 1) % decoder_sparse_step == 0`.
    /// `1` → every layer is MoE; `2` → every other; etc.
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
}

impl Qwen3MoeConfig {
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        self.num_experts > 0 && (layer_idx + 1) % self.decoder_sparse_step == 0
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeExpertWeights {
    pub gate_w: WeightStorage,
    pub up_w: WeightStorage,
    pub down_w: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeLayerWeights {
    pub attn_norm_gain: Arc<[f32]>,
    pub ffn_norm_gain: Arc<[f32]>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_o: WeightStorage,
    /// Per-head QK-norm gains (`[head_dim]` each).
    pub q_norm_gain: Arc<[f32]>,
    pub k_norm_gain: Arc<[f32]>,
    /// FFN variant. `Dense` → single SwiGLU; `Moe` → router + experts.
    pub ffn: Qwen3MoeFfn,
}

#[derive(Debug, Clone)]
pub enum Qwen3MoeFfn {
    Dense {
        gate_w: WeightStorage,
        up_w: WeightStorage,
        down_w: WeightStorage,
    },
    Moe {
        /// `[hidden_size, num_experts]` router.
        router_w: Arc<[f32]>,
        experts: Vec<Qwen3MoeExpertWeights>,
    },
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<Qwen3MoeLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Qwen3MoeModel {
    pub config: Qwen3MoeConfig,
    pub weights: Qwen3MoeWeights,
}

impl Qwen3MoeModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Qwen3-MoE-specific: per-layer sliding-window gate
    /// (`use_sliding_window && layer_idx < max_window_layers`)
    /// and per-token MoE FFN routing are honored.
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. Qwen3-MoE does NOT scale embeddings.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.hidden_size, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.output.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0);

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "Qwen3MoeModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "Qwen3MoeModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "Qwen3MoeConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.head_dim,
        );

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let uses_window = cfg.use_sliding_window && layer_idx < cfg.max_window_layers;
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, uses_window)?;
        }

        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn build_layer_mask(&self, anchor: &LazyTensor, seq: usize, uses_window: bool) -> LazyTensor {
        let cfg = &self.config;
        let window = if uses_window { cfg.sliding_window.unwrap_or(seq + 1) } else { seq + 1 };
        let mut mask_data = vec![0.0_f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i || j + window <= i { mask_data[i * seq + j] = f32::NEG_INFINITY; }
            }
        }
        anchor.const_f32_like(mask_data, Shape::from_dims(&[1, 1, seq, seq]))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Qwen3MoeLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        uses_window: bool,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.attn_norm_gain), cfg.rms_norm_eps)?;

        let q = layer.attn_q.apply_linear(&x_norm, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(&x_norm, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, cfg.head_dim)?;
        let k = k.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;
        let v = v.split_heads(cfg.num_key_value_heads, cfg.head_dim)?;

        // Per-head QK-norm.
        let q = q.rms_norm_affine(std::sync::Arc::clone(&layer.q_norm_gain), cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine(std::sync::Arc::clone(&layer.k_norm_gain), cfg.rms_norm_eps)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        let n_rep = cfg.num_attention_heads / cfg.num_key_value_heads;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (cfg.head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = self.build_layer_mask(x, seq, uses_window);
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        let attn_out = layer.attn_o.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size);

        let h1 = x.add(&attn_out)?;
        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.ffn_norm_gain), cfg.rms_norm_eps)?;

        let ffn_out = self.apply_ffn(&h1_norm, &layer.ffn, batch, seq)?;
        h1.add(&ffn_out)
    }

    fn apply_ffn(
        &self,
        x: &LazyTensor,
        ffn: &Qwen3MoeFfn,
        batch: usize,
        seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        match ffn {
            Qwen3MoeFfn::Dense { gate_w, up_w, down_w } => {
                let inter = cfg.intermediate_size;
                let gate = gate_w.apply_linear(x, h, inter);
                let up = up_w.apply_linear(x, h, inter);
                let swiglu = gate.silu().mul(&up)?;
                Ok(down_w.apply_linear(&swiglu, inter, h))
            }
            Qwen3MoeFfn::Moe { router_w, experts } => {
                let inter = cfg.moe_intermediate_size;
                let router_w_t = x.const_f32_like(
                    router_w.clone(),
                    Shape::from_dims(&[h, cfg.num_experts]),
                );
                let router_logits = x.matmul(&router_w_t)?;
                let router_weights = router_logits.softmax_last_dim()?;

                let mut routed_sum: Option<LazyTensor> = None;
                for (ei, ew) in experts.iter().enumerate() {
                    let gate = ew.gate_w.apply_linear(x, h, inter);
                    let up = ew.up_w.apply_linear(x, h, inter);
                    let swiglu = gate.silu().mul(&up)?;
                    let expert_out = ew.down_w.apply_linear(&swiglu, inter, h);

                    let w_col = router_weights.slice(2_usize, ei, 1)?;
                    let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
                    let gated = expert_out.mul(&w_bc)?;
                    routed_sum = Some(match routed_sum {
                        Some(s) => s.add(&gated)?,
                        None => gated,
                    });
                }
                Ok(routed_sum.expect("Qwen3-MoE: must have at least one expert"))
            }
        }
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl Qwen3MoeWeights {
    /// Load Qwen3-MoE (Qwen/Qwen3-MoE-A*) weights from HF safetensors.
    /// Layer FFN selects Dense vs MoE per `cfg.layer_uses_moe(i)`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Qwen3MoeConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix, load_transposed_matrix_preserve_dtype as ltm};
        let h = cfg.hidden_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let moe_int = cfg.moe_intermediate_size;

        let token_embedding = Arc::from(load_tensor_as_f32(
            st, "model.embed_tokens.weight",
        )?);

        let opt_bias = |name: String| -> Option<Arc<[f32]>> {
            load_tensor_as_f32(st, &name).ok().map(Arc::from)
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.input_layernorm.weight"),
            )?);
            let ffn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_attention_layernorm.weight"),
            )?);
            let attn_q = ltm(st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h)?;
            let attn_q_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.q_proj.bias"))
            } else { None };
            let attn_k = ltm(st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?;
            let attn_k_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.k_proj.bias"))
            } else { None };
            let attn_v = ltm(st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?;
            let attn_v_bias = if cfg.attention_bias {
                opt_bias(format!("{p}.self_attn.v_proj.bias"))
            } else { None };
            let attn_o = ltm(st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim)?;
            let q_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn.q_norm.weight"),
            )?);
            let k_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn.k_norm.weight"),
            )?);

            let ffn = if cfg.layer_uses_moe(i) {
                // HF gate weight: `[num_experts, hidden]`; transpose to
                // `[hidden, num_experts]` for matmul layout.
                let router_w = Arc::from(load_transposed_matrix(
                    st, &format!("{p}.mlp.gate.weight"), cfg.num_experts, h,
                )?);
                let mut experts = Vec::with_capacity(cfg.num_experts);
                for e in 0..cfg.num_experts {
                    let ep = format!("{p}.mlp.experts.{e}");
                    let gate_w_e = ltm(st, &format!("{ep}.gate_proj.weight"), moe_int, h)?;
                    let up_w = ltm(st, &format!("{ep}.up_proj.weight"), moe_int, h)?;
                    let down_w = ltm(st, &format!("{ep}.down_proj.weight"), h, moe_int)?;
                    experts.push(Qwen3MoeExpertWeights {
                        gate_w: gate_w_e, up_w, down_w,
                    });
                }
                Qwen3MoeFfn::Moe { router_w, experts }
            } else {
                let gate_w = ltm(st, &format!("{p}.mlp.gate_proj.weight"), inter, h)?;
                let up_w = ltm(st, &format!("{p}.mlp.up_proj.weight"), inter, h)?;
                let down_w = ltm(st, &format!("{p}.mlp.down_proj.weight"), h, inter)?;
                Qwen3MoeFfn::Dense { gate_w, up_w, down_w }
            };

            layers.push(Qwen3MoeLayerWeights {
                attn_norm_gain, ffn_norm_gain,
                attn_q, attn_q_bias, attn_k, attn_k_bias, attn_v, attn_v_bias, attn_o,
                q_norm_gain, k_norm_gain,
                ffn,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(
            st, "model.norm.weight",
        )?);
        let output = match ltm(st, "lm_head.weight", cfg.vocab_size, h) {
            Ok(w) => w,
            Err(_) => crate::lazy_llama_full::tied_lm_head_from_embeddings(
                &token_embedding, cfg.vocab_size, h,
            ),
        };

        Ok(Self { token_embedding, layers, final_norm_gain, output })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_weights(cfg: &Qwen3MoeConfig) -> Qwen3MoeWeights {
        let mut s: u32 = 13579;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let moe_inter = cfg.moe_intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<Qwen3MoeLayerWeights> = (0..cfg.num_hidden_layers).map(|li| {
            let ffn = if cfg.layer_uses_moe(li) {
                let router_w = vec_of(h * cfg.num_experts, &mut *nb);
                let experts: Vec<Qwen3MoeExpertWeights> = (0..cfg.num_experts).map(|_| {
                    Qwen3MoeExpertWeights {
                        gate_w: WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                        up_w:   WeightStorage::F32(vec_of(h * moe_inter, &mut *nb)),
                        down_w: WeightStorage::F32(vec_of(moe_inter * h, &mut *nb)),
                    }
                }).collect();
                Qwen3MoeFfn::Moe { router_w, experts }
            } else {
                Qwen3MoeFfn::Dense {
                    gate_w: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    up_w:   WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                    down_w: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                }
            };
            Qwen3MoeLayerWeights {
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain:  Arc::from(vec![1.0_f32; h]),
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: if cfg.attention_bias { Some(vec_of(h, &mut *nb)) } else { None },
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: if cfg.attention_bias { Some(vec_of(kv, &mut *nb)) } else { None },
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                ffn,
            }
        }).collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        Qwen3MoeWeights { token_embedding, layers, final_norm_gain, output }
    }

    #[test]
    fn forward_with_alternating_dense_and_moe() {
        // decoder_sparse_step = 2 → layers 1 and 3 (0-indexed) use MoE,
        // layer 0 and 2 use dense.
        let cfg = Qwen3MoeConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 4, num_attention_heads: 2, head_dim: 4,
            attention_bias: false, num_key_value_heads: 2,
            max_position_embeddings: 32,
            sliding_window: None, max_window_layers: 0, use_sliding_window: false,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            decoder_sparse_step: 2, moe_intermediate_size: 8,
            num_experts: 2, num_experts_per_tok: 1,
        };
        // Confirm the FFN-mode mapping is what we expect.
        assert_eq!(cfg.layer_uses_moe(0), false);
        assert_eq!(cfg.layer_uses_moe(1), true);
        assert_eq!(cfg.layer_uses_moe(2), false);
        assert_eq!(cfg.layer_uses_moe(3), true);
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let logits = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 3, cfg.vocab_size]);
        for &v in &logits.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = Qwen3MoeConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 4, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 2,
            max_position_embeddings: 32,
            sliding_window: None, max_window_layers: 0, use_sliding_window: false,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            decoder_sparse_step: 2, moe_intermediate_size: 8,
            num_experts: 2, num_experts_per_tok: 1,
            attention_bias: false,
        };
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            vocab_size: 16, hidden_size: 8, intermediate_size: 16,
            num_hidden_layers: 4, num_attention_heads: 4, num_key_value_heads: 4,
            head_dim: 2,
            max_position_embeddings: 32,
            sliding_window: None, max_window_layers: 0, use_sliding_window: false,
            rope_theta: 10_000.0, rms_norm_eps: 1e-5,
            decoder_sparse_step: 2, moe_intermediate_size: 8,
            num_experts: 2, num_experts_per_tok: 1,
            attention_bias: false,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Qwen3MoE forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = Qwen3MoeModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Qwen3MoE forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}

//! DeepSeek-V2 (Multi-head Latent Attention + MoE) ported to the
//! lazy-graph API.
//!
//! Phase D specialized port. DeepSeek-V2 introduces **Multi-head
//! Latent Attention (MLA)** — a compression-based attention
//! mechanism designed to slash the KV-cache cost during decode
//! while preserving multi-head expressiveness:
//!
//!   - **Q** is split into a NoPE part (`qk_nope_head_dim` per
//!     head) and a RoPE part (`qk_rope_head_dim` per head).
//!     Optionally produced via LoRA (`q_a_proj → norm →
//!     q_b_proj`) when `q_lora_rank` is set; falls back to a
//!     plain projection otherwise.
//!   - **KV** flows through a low-rank latent path:
//!     ```text
//!     compressed_kv, k_pe = kv_a_proj_with_mqa(x).split(
//!                                kv_lora_rank, qk_rope_head_dim)
//!     k_nope, v = kv_b_proj(layernorm(compressed_kv))
//!                     .split(qk_nope_head_dim, v_head_dim)
//!     ```
//!     `k_pe` is **single-head** (MQA-shared) and gets broadcast
//!     across all heads.
//!   - **Attention**: `Q = cat(q_nope, q_pe)`,
//!     `K = cat(k_nope, k_pe_repeated)`. Softmax-scaled with an
//!     mscale-adjusted scale if YaRN scaling is on (v1: plain
//!     RoPE only, YaRN deferred — `softmax_scale = 1 /
//!     sqrt(q_head_dim)`).
//!
//! The MoE block follows the Qwen2-MoE pattern adopted by Phase
//! D batch B: dense routing (full softmax × every expert),
//! plus an always-on **shared-expert** branch (`n_shared_experts
//! > 0`). The `first_k_dense_replace` config skips MoE for the
//! first K layers (they use a plain SwiGLU MLP instead).
//!
//! v1 deferrals:
//!   - **YaRN / Su / Dynamic / Linear RoPE scaling**. v1 uses
//!     plain RoPE with `rope_theta`.
//!   - **Group-limited top-K routing** (`n_group`, `topk_group`,
//!     `TopkMethod::GroupLimitedGreedy`). v1 uses dense softmax
//!     routing (every expert evaluated, weighted).
//!   - **routed_scaling_factor**. Applied as a no-op (factor=1)
//!     by default.
//!
//! # Scope (v1)
//!
//! Forward-only, single sequence (`batch == 1`), no KV cache,
//! F32. Both LoRA-Q (DeepSeek-V2) and plain-Q (DeepSeek-V2-Lite)
//! configurations supported.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::apply_interleaved_partial_rope;
use crate::lazy_latent_cache::LazyLatentCache;
use crate::{DType, Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeek2Activation {
    Silu,
    Gelu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeek2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    /// Layer `i` uses MoE iff `i >= first_k_dense_replace && (i %
    /// moe_layer_freq == 0)` and `n_routed_experts > 0`. Default
    /// is `1` (every layer past the dense replace boundary).
    pub moe_layer_freq: usize,
    pub first_k_dense_replace: usize,
    pub norm_topk_prob: bool,
    pub hidden_activation: DeepSeek2Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub attention_bias: bool,
    /// MLA Q-LoRA rank. `None` → plain Q projection.
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
}

impl DeepSeek2Config {
    pub fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }
    /// True iff this layer uses MoE (else plain dense MLP).
    pub fn layer_uses_moe(&self, layer_idx: usize) -> bool {
        let n_routed = self.n_routed_experts.unwrap_or(0);
        n_routed > 0
            && layer_idx >= self.first_k_dense_replace
            && (layer_idx - self.first_k_dense_replace) % self.moe_layer_freq == 0
    }
}

#[derive(Debug, Clone)]
pub enum DeepSeek2QProj {
    Plain(WeightStorage),
    Lora {
        a: WeightStorage,
        norm_gain: Arc<[f32]>,
        b: WeightStorage,
    },
}

#[derive(Debug, Clone)]
pub struct DeepSeek2MlaWeights {
    pub q_proj: DeepSeek2QProj,
    /// `[hidden, kv_lora_rank + qk_rope_head_dim]`.
    pub kv_a_proj_with_mqa: WeightStorage,
    pub kv_a_layernorm_gain: Arc<[f32]>,
    /// `[kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]`.
    pub kv_b_proj: WeightStorage,
    /// `[num_heads * v_head_dim, hidden]`.
    pub o_proj: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2DenseMlpWeights {
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2ExpertWeights {
    pub gate: WeightStorage,
    pub up: WeightStorage,
    pub down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2MoeWeights {
    /// `[hidden, n_routed_experts]` routing matrix.
    pub router: Arc<[f32]>,
    pub experts: Vec<DeepSeek2ExpertWeights>,
    /// Shared expert (always-on). Intermediate size =
    /// `n_shared_experts * moe_intermediate_size`.
    pub shared_gate: WeightStorage,
    pub shared_up: WeightStorage,
    pub shared_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub enum DeepSeek2FfnWeights {
    Dense(DeepSeek2DenseMlpWeights),
    Moe(DeepSeek2MoeWeights),
}

#[derive(Debug, Clone)]
pub struct DeepSeek2LayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub mla: DeepSeek2MlaWeights,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub ffn: DeepSeek2FfnWeights,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2Weights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<DeepSeek2LayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    /// Optional separate lm_head. None ⇒ tied to token_embedding.
    pub lm_head: Option<WeightStorage>,
}

#[derive(Debug, Clone)]
pub struct DeepSeek2Model {
    pub config: DeepSeek2Config,
    pub weights: DeepSeek2Weights,
}

impl DeepSeek2Weights {
    /// Load DeepSeek-V2 weights from a memory-mapped safetensors file
    /// using the standard HuggingFace top-level naming.
    ///
    /// # Tensor naming convention
    ///
    /// - `model.embed_tokens.weight` → `token_embedding` (row-major
    ///   `[vocab, hidden]`).
    /// - Per layer `i` (`model.layers.{i}`):
    ///   - `input_layernorm.weight` → `input_norm_gain`
    ///   - `post_attention_layernorm.weight` → `post_attn_norm_gain`
    ///   - **MLA attention** (`self_attn.*`):
    ///     - Q-LoRA case (`cfg.q_lora_rank == Some(r)`):
    ///       - `q_a_proj.weight` (`[r, hidden]` HF) →
    ///         `DeepSeek2QProj::Lora.a` (transposed to `[hidden, r]`).
    ///       - `q_a_layernorm.weight` (`[r]`) → `norm_gain`.
    ///       - `q_b_proj.weight` (`[n_heads * q_head_dim, r]` HF) →
    ///         `b` (transposed to `[r, n_heads * q_head_dim]`).
    ///     - Plain Q case: `q_proj.weight`
    ///       (`[n_heads * q_head_dim, hidden]` HF) → `Plain`
    ///       (transposed to `[hidden, n_heads * q_head_dim]`).
    ///     - `kv_a_proj_with_mqa.weight`
    ///       (`[kv_lora_rank + qk_rope_head_dim, hidden]` HF) →
    ///       `kv_a_proj_with_mqa` (transposed).
    ///     - `kv_a_layernorm.weight` (`[kv_lora_rank]`) →
    ///       `kv_a_layernorm_gain`.
    ///     - `kv_b_proj.weight`
    ///       (`[n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]`
    ///       HF) → `kv_b_proj` (transposed).
    ///     - `o_proj.weight` (`[hidden, n_heads * v_head_dim]` HF) →
    ///       `o_proj` (transposed to `[n_heads * v_head_dim, hidden]`).
    ///   - **FFN** depends on `cfg.layer_uses_moe(i)`:
    ///     - Dense (`mlp.*`): `gate_proj.weight`, `up_proj.weight`,
    ///       `down_proj.weight` — same layout as LLaMA.
    ///     - MoE:
    ///       - `mlp.gate.weight` (`[n_routed_experts, hidden]` HF) →
    ///         `router` (transposed to flat row-major
    ///         `[hidden, n_routed_experts]`).
    ///       - `mlp.experts.{ei}.{gate_proj,up_proj,down_proj}.weight`
    ///         per routed expert. Intermediate size is
    ///         `cfg.moe_intermediate_size`.
    ///       - `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`
    ///         with intermediate size
    ///         `n_shared_experts * moe_intermediate_size`.
    /// - `model.norm.weight` → `final_norm_gain`.
    /// - `lm_head.weight` (optional, falls back to tied embeddings) →
    ///   `lm_head`.
    ///
    /// # Deferrals
    ///
    /// `attention_bias=true` biases (`q_a_proj.bias`,
    /// `kv_a_proj_with_mqa.bias`, `o_proj.bias`) are not loaded — the
    /// current `DeepSeek2MlaWeights` struct has no bias fields, matching
    /// the v1 forward path which uses `apply_linear` without bias. Most
    /// public DeepSeek-V2 checkpoints set `attention_bias=false`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &DeepSeek2Config,
    ) -> Result<Self> {
        use crate::lazy::{
            load_tensor_as_f32, load_transposed_matrix,
            load_transposed_matrix_preserve_dtype,
        };

        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;
        let kv_lora = cfg.kv_lora_rank;

        let token_embedding = load_tensor_as_f32(st, "model.embed_tokens.weight")?;
        if token_embedding.len() != cfg.vocab_size * h {
            crate::bail!(
                "model.embed_tokens.weight: {} elts, expected {} ({}×{})",
                token_embedding.len(), cfg.vocab_size * h, cfg.vocab_size, h,
            );
        }

        let mut layers: Vec<DeepSeek2LayerWeights> =
            Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{li}");

            // --- Norms -------------------------------------------------
            let input_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.input_layernorm.weight"),
            )?);
            let post_attn_norm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.post_attention_layernorm.weight"),
            )?);

            // --- MLA attention ----------------------------------------
            let q_proj = match cfg.q_lora_rank {
                Some(lora) => {
                    let a = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.q_a_proj.weight"), lora, h,
                    )?;
                    let norm_gain = Arc::from(load_tensor_as_f32(
                        st, &format!("{p}.self_attn.q_a_layernorm.weight"),
                    )?);
                    let b = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.q_b_proj.weight"),
                        n_heads * q_head_dim, lora,
                    )?;
                    DeepSeek2QProj::Lora { a, norm_gain, b }
                }
                None => {
                    let plain = load_transposed_matrix_preserve_dtype(
                        st, &format!("{p}.self_attn.q_proj.weight"),
                        n_heads * q_head_dim, h,
                    )?;
                    DeepSeek2QProj::Plain(plain)
                }
            };

            let kv_a_proj_with_mqa = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                kv_lora + rope, h,
            )?;
            let kv_a_layernorm_gain = Arc::from(load_tensor_as_f32(
                st, &format!("{p}.self_attn.kv_a_layernorm.weight"),
            )?);
            let kv_b_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.kv_b_proj.weight"),
                n_heads * (nope + v_dim), kv_lora,
            )?;
            let o_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.o_proj.weight"),
                h, n_heads * v_dim,
            )?;

            let mla = DeepSeek2MlaWeights {
                q_proj,
                kv_a_proj_with_mqa,
                kv_a_layernorm_gain,
                kv_b_proj,
                o_proj,
            };

            // --- FFN (dense or MoE) -----------------------------------
            let ffn = if cfg.layer_uses_moe(li) {
                let n_routed = cfg.n_routed_experts.unwrap_or(0);
                let n_shared = cfg.n_shared_experts.unwrap_or(0);
                let inter = cfg.moe_intermediate_size;

                // Router: HF `[n_routed_experts, hidden_size]` →
                // flat `[hidden_size, n_routed_experts]`.
                let router_flat = load_transposed_matrix(
                    st, &format!("{p}.mlp.gate.weight"), n_routed, h,
                )?;
                let router: Arc<[f32]> = Arc::from(router_flat);

                let mut experts: Vec<DeepSeek2ExpertWeights> =
                    Vec::with_capacity(n_routed);
                for ei in 0..n_routed {
                    let ep = format!("{p}.mlp.experts.{ei}");
                    let gate = load_transposed_matrix_preserve_dtype(
                        st, &format!("{ep}.gate_proj.weight"), inter, h,
                    )?;
                    let up = load_transposed_matrix_preserve_dtype(
                        st, &format!("{ep}.up_proj.weight"), inter, h,
                    )?;
                    let down = load_transposed_matrix_preserve_dtype(
                        st, &format!("{ep}.down_proj.weight"), h, inter,
                    )?;
                    experts.push(DeepSeek2ExpertWeights { gate, up, down });
                }

                // Shared experts. When `n_shared_experts == 0`, the
                // forward path early-returns before consuming the
                // shared tensors, so we stash zero-length placeholders.
                let shared_inter = n_shared * inter;
                let (shared_gate, shared_up, shared_down) = if n_shared > 0 {
                    let sp = format!("{p}.mlp.shared_experts");
                    let g = load_transposed_matrix_preserve_dtype(
                        st, &format!("{sp}.gate_proj.weight"), shared_inter, h,
                    )?;
                    let u = load_transposed_matrix_preserve_dtype(
                        st, &format!("{sp}.up_proj.weight"), shared_inter, h,
                    )?;
                    let d = load_transposed_matrix_preserve_dtype(
                        st, &format!("{sp}.down_proj.weight"), h, shared_inter,
                    )?;
                    (g, u, d)
                } else {
                    let empty: Arc<[f32]> = Arc::from(Vec::<f32>::new());
                    (
                        WeightStorage::F32(empty.clone()),
                        WeightStorage::F32(empty.clone()),
                        WeightStorage::F32(empty),
                    )
                };

                DeepSeek2FfnWeights::Moe(DeepSeek2MoeWeights {
                    router,
                    experts,
                    shared_gate,
                    shared_up,
                    shared_down,
                })
            } else {
                let inter = cfg.intermediate_size;
                let gate = load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.mlp.gate_proj.weight"), inter, h,
                )?;
                let up = load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.mlp.up_proj.weight"), inter, h,
                )?;
                let down = load_transposed_matrix_preserve_dtype(
                    st, &format!("{p}.mlp.down_proj.weight"), h, inter,
                )?;
                DeepSeek2FfnWeights::Dense(DeepSeek2DenseMlpWeights {
                    gate, up, down,
                })
            };

            layers.push(DeepSeek2LayerWeights {
                input_norm_gain,
                mla,
                post_attn_norm_gain,
                ffn,
            });
        }

        let final_norm_gain = Arc::from(load_tensor_as_f32(
            st, "model.norm.weight",
        )?);

        // Optional separate lm_head. None ⇒ tied to token_embedding at
        // apply_lm_head time. Honour cfg.tie_word_embeddings first: when
        // the user requested tying, we don't even look for lm_head.
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            load_transposed_matrix_preserve_dtype(
                st, "lm_head.weight", cfg.vocab_size, h,
            ).ok()
        };

        Ok(DeepSeek2Weights {
            token_embedding: Arc::from(token_embedding),
            layers,
            final_norm_gain,
            lm_head,
        })
    }
}

/// Split `kv_b_proj` (row-major `[kv_lora_rank, n_heads * (nope + v_dim)]`
/// — element `[row j in 0..kvr][col c in 0..H*(nope+v)] = arc[j *
/// (H*(nope+v)) + c]`) into the two forms the MLA weight-absorption decode
/// trick needs (see [`DeepSeek2Model::forward_with_latent_cache_absorbed`]
/// for the math):
///
///   - `w_uk_t`: `[H, nope, kvr]` — `w_uk_t[h][i][j] = kv_b[j][h*(nope+v) + i]`
///     (per-head `W_UK[h]`, **transposed** so it can be matmul'd directly
///     against `q_nope[h]`: `q_absorbed[h] = q_nope[h] @ w_uk_t[h]`).
///   - `w_uv`: `[H, kvr, v_dim]` — `w_uv[h][j][i] = kv_b[j][h*(nope+v) + nope + i]`
///     (per-head `W_UV[h]`, as-is).
///
/// F32-only today — any other [`WeightStorage`] variant (BF16, Q4_0,
/// WithLoRA) is a typed error rather than a silent cast. Also validates
/// the input length against `kvr * n_heads * (nope + v_dim)`.
fn absorb_split_kv_b(
    w: &WeightStorage,
    kvr: usize,
    n_heads: usize,
    nope: usize,
    v_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let arc = match w {
        WeightStorage::F32(arc) => arc,
        other => {
            return Err(crate::Error::Msg(format!(
                "absorb_split_kv_b: MLA weight absorption is F32-only today, got {:?}",
                other.dtype(),
            )).bt());
        }
    };
    let out_features = n_heads * (nope + v_dim);
    let expected = kvr * out_features;
    if arc.len() != expected {
        return Err(crate::Error::Msg(format!(
            "absorb_split_kv_b: kv_b_proj has {} elements, expected {expected} \
             (kv_lora_rank={kvr} * n_heads*(nope+v_dim)={out_features})",
            arc.len(),
        )).bt());
    }

    let mut w_uk_t = vec![0.0_f32; n_heads * nope * kvr];
    let mut w_uv = vec![0.0_f32; n_heads * kvr * v_dim];
    for j in 0..kvr {
        let row = &arc[j * out_features..(j + 1) * out_features];
        for h in 0..n_heads {
            let base = h * (nope + v_dim);
            // W_UK[h] transposed: w_uk_t[h][i][j] = kv_b[j][h*(nope+v)+i]
            for i in 0..nope {
                w_uk_t[h * nope * kvr + i * kvr + j] = row[base + i];
            }
            // W_UV[h] as-is: w_uv[h][j][i] = kv_b[j][h*(nope+v)+nope+i]
            for i in 0..v_dim {
                w_uv[h * kvr * v_dim + j * v_dim + i] = row[base + nope + i];
            }
        }
    }
    Ok((w_uk_t, w_uv))
}

impl DeepSeek2Model {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final RmsNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// DeepSeek-V2-specific: MLA attention, per-layer dense /
    /// MoE FFN selection (first `n` dense layers, then MoE).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. DeepSeek-V2 does NOT scale embeddings.
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

    /// MLA cached-decode entry point: run `tokens` against a
    /// [`LazyLatentCache`] holding the growing decode state, returning
    /// the new tokens' logits and the advanced cache.
    ///
    /// This is the MLA decode-time compressed-KV payoff. Per layer the
    /// cache holds two slots: slot 0 is the **post-norm** compressed
    /// latent (trailing `[kv_lora_rank]`) and slot 1 is the **post-RoPE**
    /// single-head rope key `k_pe` (trailing `[qk_rope_head_dim]`).
    ///
    /// - The latent is cached *after* `kv_a_layernorm` because RMS-norm
    ///   is per-token — normalizing the whole prefill at once vs one
    ///   token at a time is mathematically identical — and the
    ///   weight-absorption decode trick (a later increment) attends
    ///   directly against the normed latent, so caching it post-norm
    ///   avoids re-normalizing the prefix on every step.
    /// - `k_pe` is cached *after* RoPE because the rotation depends only
    ///   on the absolute token position, so it is fixed the moment it's
    ///   written — the same reason standard KV caches store the rotated
    ///   K rather than re-rotating the whole prefix every step.
    ///
    /// Mirrors the LlamaModel/PhiModel cached-decode convention: RoPE
    /// tables are rebuilt each step at `start_pos = cached_len`, the
    /// decode causal mask is built once and shared across every layer,
    /// and the cache position is advanced once at the end of the step.
    pub fn forward_with_latent_cache(
        &self,
        tokens: &[u32],
        cache: LazyLatentCache,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        self.forward_with_latent_cache_impl(tokens, cache, false)
    }

    /// Weight-absorption sibling of [`Self::forward_with_latent_cache`] —
    /// the DeepSeek decode trick. Same cached-decode contract (geometry
    /// validation, fresh-graph rebind, per-step RoPE tables, shared decode
    /// mask, cache threading) but attention goes through
    /// [`Self::mla_attention_cached_absorbed`] instead of
    /// [`Self::mla_attention_cached`]: rather than up-projecting the whole
    /// cached latent prefix through `kv_b_proj` every step, `kv_b_proj`'s
    /// per-head `W_UK`/`W_UV` slices are folded into the query/context
    /// math so attention reads the compressed latent directly. See
    /// [`Self::mla_attention_cached_absorbed`]'s doc for the full math and
    /// the bit-exactness caveat (mathematically equivalent, not
    /// bit-identical, to the non-absorbed path).
    pub fn forward_with_latent_cache_absorbed(
        &self,
        tokens: &[u32],
        cache: LazyLatentCache,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        self.forward_with_latent_cache_impl(tokens, cache, true)
    }

    /// Shared body of [`Self::forward_with_latent_cache`] and
    /// [`Self::forward_with_latent_cache_absorbed`] — identical everywhere
    /// except which `mla_attention_cached*` sibling `absorb` routes each
    /// layer's attention through.
    fn forward_with_latent_cache_impl(
        &self,
        tokens: &[u32],
        cache: LazyLatentCache,
        absorb: bool,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        let cfg = &self.config;

        if tokens.is_empty() {
            return Err(crate::Error::Msg(
                "DeepSeek2Model::forward_with_latent_cache: tokens must be non-empty".into(),
            ).bt());
        }
        if cache.n_layers() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_cache: cache n_layers ({}) != model \
                 num_hidden_layers ({})",
                cache.n_layers(), cfg.num_hidden_layers,
            )).bt());
        }
        if cache.n_slots() != 2 {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_cache: MLA latent cache needs exactly 2 \
                 slots (compressed latent + k_pe), got {}",
                cache.n_slots(),
            )).bt());
        }
        if cache.slot_trailing(0).to_vec() != vec![cfg.kv_lora_rank] {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_cache: slot 0 trailing shape {:?} != \
                 [kv_lora_rank={}]",
                cache.slot_trailing(0), cfg.kv_lora_rank,
            )).bt());
        }
        if cache.slot_trailing(1).to_vec() != vec![cfg.qk_rope_head_dim] {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_cache: slot 1 trailing shape {:?} != \
                 [qk_rope_head_dim={}]",
                cache.slot_trailing(1), cfg.qk_rope_head_dim,
            )).bt());
        }

        let cached_len = cache.current_seq_len();
        let seq_new = tokens.len();
        let total_len = cached_len + seq_new;
        if total_len > cache.max_seq_len() {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_cache: cached_len ({cached_len}) + \
                 seq_new ({seq_new}) = {total_len} exceeds cache max_seq_len ({})",
                cache.max_seq_len(),
            )).bt());
        }

        // Re-anchor onto a FRESH graph, rebinding the realized prefix via
        // const_*_like — see Self::rebind_latent_cache_fresh_graph's doc for
        // why this is required (a real `PipelinedExecutor` gap, not a style
        // choice).
        let cache = self.rebind_latent_cache_fresh_graph(cache, cached_len)?;

        // Anchor this step's graph on the cache's existing buffer.
        let anchor = cache.slot_buffer_full(0, 0);
        let h = self.embed_tokens_anchored(&anchor, tokens)?;

        // RoPE tables for THIS step at the absolute start position.
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, cached_len, seq_new, cfg.qk_rope_head_dim,
        );

        // Decode mask, built once and shared across every layer.
        let mask_data = crate::lazy::build_decode_causal_mask(cached_len, seq_new, total_len);
        let mask = h
            .const_f32_like(mask_data, Shape::from_dims(&[seq_new, total_len]))
            .reshape(Shape::from_dims(&[1, 1, seq_new, total_len]))?;

        let mut cache = cache;
        let mut x = h;
        for (idx, layer) in self.weights.layers.iter().enumerate() {
            let (x_next, cache_next) = self.apply_layer_cached(
                &x, layer, idx, &rope_cos, &rope_sin, &mask, cache, cached_len, absorb,
            )?;
            x = x_next;
            cache = cache_next;
        }

        let h_norm = x.rms_norm_affine(
            std::sync::Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps,
        )?;
        let logits = self.apply_lm_head(&h_norm)?;
        let cache = cache.advance_by(seq_new);
        Ok((logits, cache))
    }

    /// MLA increment 4 — the persistent (cross-graph) N-slot decode-cache
    /// entry point. D1-style: a fresh graph per call, but the cache
    /// storage is a device-resident [`crate::inference_context::
    /// LatentKvCache`] that survives across calls — **no host round-trip
    /// of the cached prefix**, unlike [`Self::forward_with_latent_cache`]
    /// / [`Self::forward_with_latent_cache_absorbed`], whose
    /// `rebind_latent_cache_fresh_graph` step realizes the whole prefix to
    /// host `f32` and re-uploads it as fresh Consts every call. Those two
    /// remain the right choice for single-graph / prefill-only use (no
    /// `InferenceContext` bookkeeping); a decode loop should use this
    /// entry point instead. Mirrors [`crate::lazy::LlamaModel::
    /// forward_with_kv_context_impl`]'s D1 shape: per-layer `Const`
    /// placeholders bound to the cache's persistent Storage Arcs via
    /// [`crate::inference_context::InferenceContext::insert`], written in
    /// place via `Op::WriteSlice` at a `SymEnv`-bound dynamic offset, then
    /// unbound after realize.
    ///
    /// Attention always goes through the **absorbed** (weight-absorption)
    /// math — see [`Self::mla_attention_latent_kv`]'s doc for why: under
    /// the full fixed-capacity read this path always performs (no slice to
    /// `cached_len + seq`), the non-absorbed form would re-run
    /// `kv_b_proj`'s up-projection over the *entire* `max_seq_len` capacity
    /// every step; absorption attends directly against the cached latent
    /// and pays no such cost.
    ///
    /// Returns the **last-position** logits, `[vocab_size]` — same shape
    /// choice as [`crate::lazy::LlamaModel::forward_with_kv_context`].
    pub fn forward_with_latent_kv_context(
        &self,
        tokens: &[u32],
        cache: &mut crate::inference_context::LatentKvCache,
        ctx: &mut crate::inference_context::InferenceContext,
    ) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let seq = tokens.len();

        if tokens.is_empty() {
            return Err(crate::Error::Msg(
                "DeepSeek2Model::forward_with_latent_kv_context: tokens must be non-empty".into(),
            ).bt());
        }
        if cache.n_layers() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: cache n_layers ({}) != model \
                 num_hidden_layers ({})",
                cache.n_layers(), cfg.num_hidden_layers,
            )).bt());
        }
        if cache.n_slots() != 2 {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: MLA latent cache needs exactly \
                 2 slots (compressed latent + k_pe), got {}",
                cache.n_slots(),
            )).bt());
        }
        if cache.slot_trailing(0).to_vec() != vec![cfg.kv_lora_rank] {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: slot 0 trailing shape {:?} != \
                 [kv_lora_rank={}]",
                cache.slot_trailing(0), cfg.kv_lora_rank,
            )).bt());
        }
        if cache.slot_trailing(1).to_vec() != vec![cfg.qk_rope_head_dim] {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: slot 1 trailing shape {:?} != \
                 [qk_rope_head_dim={}]",
                cache.slot_trailing(1), cfg.qk_rope_head_dim,
            )).bt());
        }
        if cache.dtype != DType::F32 {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: cache dtype {:?} != F32 \
                 (MLA weight absorption is F32-only today)",
                cache.dtype,
            )).bt());
        }
        let cached_len = cache.cached_len;
        let max_seq_len = cache.max_seq_len;
        if cached_len + seq > max_seq_len {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: cached_len ({cached_len}) + \
                 seq ({seq}) > max_seq_len ({max_seq_len})",
            )).bt());
        }

        // Bootstrap the fresh graph exactly like run_backbone: token
        // embedding lookup → (1, seq, hidden).
        let mut h = LazyTensor::embed_tokens(
            self.weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens,
            &Device::cpu(),
        )?;

        // Phase D: the per-token KV-write offset is a runtime symbol bound
        // through the per-pass SymEnv at realize, not baked into the
        // graph. A fixed id (re-bound each pass) — the same convention
        // `LlamaModel::forward_with_kv_context_impl` uses, even though
        // this entry point rebuilds a fresh graph every call (no held-
        // session reuse yet): keeping the offset symbolic rather than a
        // baked literal keeps this path's graph shape identical to what a
        // future plan-once session would hold.
        let cached_len_sym = fuel_ir::SymId(0);

        // RoPE tables at the absolute start position, shared across layers.
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, cached_len, seq, cfg.qk_rope_head_dim,
        );

        // Decode mask, hoisted once and shared across every layer. Width
        // is max_seq_len (full capacity), NOT cached_len + seq — the
        // persistent path always attends over the full buffer, relying on
        // the mask to hide both future positions and the zero-init/stale
        // tail (same trade LlamaModel's forward_with_kv_context documents).
        let mask_data = crate::lazy::build_decode_causal_mask(cached_len, seq, max_seq_len);
        let mask = h
            .const_f32_like(mask_data, Shape::from_dims(&[seq, max_seq_len]))
            .reshape(Shape::from_dims(&[1, 1, seq, max_seq_len]))?;

        let kvr = cfg.kv_lora_rank;
        let rope = cfg.qk_rope_head_dim;

        // Per layer: bind both cache slot Arcs to fresh Const NodeIds,
        // dispatch through the WriteSlice variant. Track the NodeIds we
        // insert into ctx so we can clean them up after realize (the
        // per-step NodeIds reference a graph that drops at the end of
        // this method; leaving them in ctx.persistent would leak).
        let mut bound_node_ids: Vec<fuel_graph::NodeId> =
            Vec::with_capacity(2 * cfg.num_hidden_layers);
        for (li, layer_weights) in self.weights.layers.iter().enumerate() {
            let latent_arc = cache.slot_storage(li, 0).ok_or_else(|| crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: cache layer {li} has no \
                 latent slot (with_capacity should have populated all layers)",
            )).bt())?;
            let kpe_arc = cache.slot_storage(li, 1).ok_or_else(|| crate::Error::Msg(format!(
                "DeepSeek2Model::forward_with_latent_kv_context: cache layer {li} has no \
                 k_pe slot",
            )).bt())?;

            let latent_c = h.const_placeholder_like(Shape::from_dims(&[max_seq_len, kvr]), DType::F32);
            let kpe_c = h.const_placeholder_like(Shape::from_dims(&[max_seq_len, rope]), DType::F32);
            let latent_id = latent_c.node_id();
            let kpe_id = kpe_c.node_id();
            ctx.insert(latent_id, latent_arc);
            ctx.insert(kpe_id, kpe_arc);
            bound_node_ids.push(latent_id);
            bound_node_ids.push(kpe_id);

            h = self.apply_layer_with_latent_kv_writes(
                &h, layer_weights, li, &rope_cos, &rope_sin, &mask, &latent_c, &kpe_c, cached_len_sym,
            )?;
        }

        let h_norm = h.rms_norm_affine(
            std::sync::Arc::clone(&self.weights.final_norm_gain), cfg.rms_norm_eps,
        )?;
        let logits = self.apply_lm_head(&h_norm)?; // (1, seq, vocab_size)
        let last_pos = seq - 1;
        let logits_root = logits
            .slice(1_usize, last_pos, 1)?
            .reshape(Shape::from_dims(&[cfg.vocab_size]))?;

        // Realize through InferenceContext. The WriteSlice nodes mutate
        // the cache buffers in place at the runtime offset `cached_len`,
        // supplied for this pass via the SymEnv; downstream attention
        // reads the post-write full-capacity buffers.
        let mut sym_env = fuel_ir::SymEnv::new();
        sym_env.bind(cached_len_sym, cached_len).map_err(crate::Error::from)?;
        let logits_vec = ctx.realize_one_as_with_env::<f32>(
            logits_root.graph_handle(),
            logits_root.node_id(),
            &sym_env,
        )?;

        // Clean up per-step bindings from ctx so they don't accumulate
        // across decode steps (each step gets a fresh graph; the previous
        // step's NodeIds are dead).
        for id in bound_node_ids {
            ctx.remove(id);
        }

        // Bump cache state — once per forward call, not per token.
        cache.cached_len += seq;
        for li in 0..cfg.num_hidden_layers {
            cache.bump_version(li, 0);
            cache.bump_version(li, 1);
        }

        Ok(logits_vec)
    }

    /// Cached-decode sibling of [`Self::apply_layer_cached`] for the
    /// persistent [`crate::inference_context::LatentKvCache`] path: same
    /// norms / residuals / dense-or-MoE FFN dispatch, but attention goes
    /// through [`Self::mla_attention_latent_kv`] and there is no functional
    /// cache threading — the cache is written in place via the
    /// `latent_c`/`kpe_c` `Op::WriteSlice` destinations
    /// [`Self::forward_with_latent_kv_context`] bound into `ctx` before
    /// calling this fn.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer_with_latent_kv_writes(
        &self,
        x: &LazyTensor,
        layer: &DeepSeek2LayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        latent_c: &LazyTensor,
        kpe_c: &LazyTensor,
        cached_len_sym: fuel_ir::SymId,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        let x_norm = x.rms_norm_affine(Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
        let attn = self.mla_attention_latent_kv(
            &x_norm, &layer.mla, rope_cos, rope_sin, mask, latent_c, kpe_c, cached_len_sym,
        )?;
        let h1 = x.add(&attn)?;

        let h1_norm = h1.rms_norm_affine(Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
        let expected_moe = cfg.layer_uses_moe(layer_idx);
        let mlp_out = match (&layer.ffn, expected_moe) {
            (DeepSeek2FfnWeights::Dense(w), false) => self.apply_dense_mlp(&h1_norm, w)?,
            (DeepSeek2FfnWeights::Moe(w), true) => self.apply_moe(&h1_norm, w)?,
            _ => return Err(crate::Error::Msg(format!(
                "DeepSeek-V2 layer {layer_idx}: FFN weight kind does not match \
                 config-derived kind (uses_moe={expected_moe}) — config + weights are inconsistent",
            )).bt()),
        };
        h1.add(&mlp_out)
    }

    /// Attention core of [`Self::apply_layer_with_latent_kv_writes`] — the
    /// persistent-cache sibling of [`Self::mla_attention_cached_absorbed`].
    /// `x` is the post-input-norm hidden state for the NEW tokens only,
    /// `(1, seq_new, hidden)`. `latent_c`/`kpe_c` are this layer's
    /// full-capacity `[max_seq_len, kv_lora_rank]` /
    /// `[max_seq_len, qk_rope_head_dim]` `Const` placeholders, bound by the
    /// caller to the [`crate::inference_context::LatentKvCache`]'s
    /// persistent Storage Arcs. `mask` is the shared
    /// `(1, 1, seq_new, max_seq_len)` decode causal mask.
    ///
    /// Always uses the **absorbed** (weight-absorption) math — folding
    /// `kv_b_proj`'s per-head `W_UK`/`W_UV` slices into the query/context
    /// math so attention reads the cached latent directly (see
    /// [`Self::mla_attention_cached_absorbed`]'s doc for the derivation).
    /// This is not merely the faster of two equally-valid choices here:
    /// this path reads the **full fixed-capacity buffer every step** (no
    /// slice to `cached_len + seq_new`, so the graph's shape stays
    /// token-independent — the same trade [`crate::lazy::
    /// apply_layer_with_kv_writes`] documents for LlamaModel). Under that
    /// constraint the non-absorbed form would re-run `kv_b_proj`'s
    /// up-projection over the *entire* `max_seq_len` capacity
    /// (`O(max_seq_len · kv_lora_rank · n_heads · (nope + v_dim))`) on
    /// EVERY decode step, not just the live prefix; the absorbed form's
    /// widest term (`O(seq_new · max_seq_len · n_heads · kv_lora_rank)`)
    /// has no such re-projection cost. The zero-init tail beyond
    /// `cached_len + seq_new` still gets attended (masked to `-inf` before
    /// softmax, contributing exactly zero) — numerically consistent with
    /// LlamaModel's identical full-capacity-read trade.
    #[allow(clippy::too_many_arguments)]
    fn mla_attention_latent_kv(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MlaWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        latent_c: &LazyTensor,
        kpe_c: &LazyTensor,
        cached_len_sym: fuel_ir::SymId,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;
        let kvr = cfg.kv_lora_rank;
        let s = x.shape().dims()[1];
        let max_seq_len = latent_c.shape().dims()[0];

        // ---- Q projection (plain or LoRA), split + RoPE on the pe half ------
        let q = match &w.q_proj {
            DeepSeek2QProj::Plain(plain) => {
                plain.apply_linear(x, cfg.hidden_size, n_heads * q_head_dim)
            }
            DeepSeek2QProj::Lora { a, norm_gain, b } => {
                let lo = a.apply_linear(x, cfg.hidden_size, norm_gain.len());
                let lo_norm = lo.rms_norm_affine(Arc::clone(norm_gain), cfg.rms_norm_eps)?;
                b.apply_linear(&lo_norm, norm_gain.len(), n_heads * q_head_dim)
            }
        };
        let q = q.split_heads(n_heads, q_head_dim)?;
        let q_nope = q.slice(3_usize, 0, nope)?;
        let q_pe = q.slice(3_usize, nope, rope)?;
        let q_pe_rot = apply_interleaved_partial_rope(&q_pe, rope_cos, rope_sin, rope, rope)?;

        // ---- New KV latents for this step's tokens only ---------------------
        let kv_a = w.kv_a_proj_with_mqa.apply_linear(x, cfg.hidden_size, kvr + rope);
        let compressed_kv = kv_a.slice(2_usize, 0, kvr)?;
        let k_pe_single = kv_a.slice(2_usize, kvr, rope)?;

        let compressed_kv_norm = compressed_kv.rms_norm_affine(
            Arc::clone(&w.kv_a_layernorm_gain), cfg.rms_norm_eps,
        )?;

        let k_pe_single_h = k_pe_single.split_heads(1, rope)?;
        let k_pe_rot = apply_interleaved_partial_rope(&k_pe_single_h, rope_cos, rope_sin, rope, rope)?;

        // ---- Write this step's slabs into the persistent capacity buffers ---
        // Destructive on latent_c/kpe_c: the returned tensors are the
        // post-write reference to the SAME underlying Arc — downstream
        // reads MUST go through latent_full/kpe_full, not latent_c/kpe_c.
        let latent_new = compressed_kv_norm.reshape(Shape::from_dims(&[s, kvr]))?; // [s, kvr]
        let kpe_new = k_pe_rot.reshape(Shape::from_dims(&[s, rope]))?; // [s, rope]
        let dyn_off = fuel_ir::DynScalar::Sym(cached_len_sym);
        let latent_full = latent_c.write_slice_dyn(
            &latent_new, vec![(0, s), (0, kvr)], 0, dyn_off,
        )?; // [max_seq, kvr]
        let kpe_full = kpe_c.write_slice_dyn(
            &kpe_new, vec![(0, s), (0, rope)], 0, dyn_off,
        )?; // [max_seq, rope]

        // ---- Absorbed weights: kv_b_proj split into per-head W_UK^T / W_UV --
        let (w_uk_t_data, w_uv_data) = absorb_split_kv_b(&w.kv_b_proj, kvr, n_heads, nope, v_dim)?;
        let w_uk_t = x.const_f32_like(w_uk_t_data, Shape::from_dims(&[1, n_heads, nope, kvr]));
        let w_uv = x.const_f32_like(w_uv_data, Shape::from_dims(&[1, n_heads, kvr, v_dim]));

        // ---- q_absorbed[h] = q_nope[h] @ W_UK[h]^T ---------------------------
        let q_absorbed = q_nope.matmul(&w_uk_t)?; // (1,H,s,nope) @ (1,H,nope,kvr) -> (1,H,s,kvr)

        // ---- Attend over the FULL fixed-capacity buffer (no slice) ----------
        let c4 = latent_full
            .reshape(Shape::from_dims(&[1, 1, max_seq_len, kvr]))?
            .broadcast_to(Shape::from_dims(&[1, n_heads, max_seq_len, kvr]))?;
        let c_t = c4.transpose()?; // (1, H, kvr, max_seq)
        let scores_nope = q_absorbed.matmul(&c_t)?; // (1, H, s, max_seq)

        let kpe4 = kpe_full
            .reshape(Shape::from_dims(&[1, 1, max_seq_len, rope]))?
            .broadcast_to(Shape::from_dims(&[1, n_heads, max_seq_len, rope]))?;
        let kpe_t = kpe4.transpose()?; // (1, H, rope, max_seq)
        let scores_rope = q_pe_rot.matmul(&kpe_t)?; // (1, H, s, max_seq)

        let scale = 1.0_f64 / (q_head_dim as f64).sqrt();
        let scores = scores_nope.add(&scores_rope)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let probs = scores_masked.softmax_last_dim()?;

        // ---- ctx_latent[h] = probs[h] @ c; ctx[h] = ctx_latent[h] @ W_UV[h] -
        let ctx_latent = probs.matmul(&c4)?; // (1, H, s, kvr)
        let ctx = ctx_latent.matmul(&w_uv)?; // (1, H, s, v_dim)

        let merged = ctx.merge_heads()?;
        let out = w.o_proj.apply_linear(&merged, n_heads * v_dim, cfg.hidden_size);
        Ok(out)
    }

    /// Re-anchor `cache` onto a brand-new graph, rebinding its filled
    /// prefix (`[0..cached_len]` of every layer/slot) as fresh `Const`
    /// nodes. A no-op (returns `cache` untouched) when `cached_len == 0`
    /// (nothing to carry over — the incoming cache is already a fresh,
    /// single-generation graph).
    ///
    /// # Why this exists
    ///
    /// Naively threading `LazyLatentCache` directly across
    /// `forward_with_latent_cache` calls (no rebind — every call's ops
    /// land on the SAME ever-growing `Rc<RefCell<Graph>>`) hits a real
    /// `PipelinedExecutor` gap once the model has ≥ 2 layers AND ≥ 2
    /// calls: `fuel-dispatch/src/pipelined.rs`'s realize loop evicts a
    /// `WriteSlice`'s destructive `dest` input from the `StorageCache`
    /// the moment that `WriteSlice` work item completes (see the
    /// `destructive_input` handling around line 904), on the assumption
    /// that a `WriteSlice` destination has exactly one live consumer —
    /// itself. That assumption breaks here: layer `L`'s post-append
    /// buffer from call *N* is read TWICE within the ancestor set of
    /// call *N+1*'s output — once as the (non-destructive) input to
    /// layer `L`'s OWN attention math from call *N* (an ancestor of
    /// layer `L+1`'s call-*N* write, which call *N+1* still depends on
    /// transitively), and again as the *destination* of layer `L`'s
    /// call-*N+1* append. Whichever the scheduler happens to run first
    /// wins; if the destructive append runs first, the eviction removes
    /// the entry before the attention-math view op reads it and realize
    /// fails with `PipelinedExecutor: view-op input NodeId(..) of
    /// NodeId(..) not realized`. This reproduces with 2+ layers and 2+
    /// `forward_with_latent_cache` calls regardless of WHEN `realize` is
    /// called (immediately per step or deferred to the end) — it is a
    /// graph-structural hazard, not a call-ordering one.
    ///
    /// **UPDATE (2026-07-08): the executor gap is FIXED** — the root
    /// cause was `collect_alias_set` (fuel-graph/src/opt.rs) treating
    /// `Op::Reshape`/`Op::Contiguize` as alias-opaque while the executor's
    /// `ContiguizeOf` arm zero-copy ADOPTS the input Arc for contiguous
    /// offset-0 inputs, leaving readers past a reshape hop unpinned
    /// against a destructive write (isolated repros:
    /// `pipelined_write_slice_reshape_adopted_arc_reader_reads_pre_write_bytes`
    /// + the 2-layer MLA-shaped variant in fuel-dispatch). Same-graph
    /// multi-step decode is now correct without this rebind. The rebind
    /// stays because it is independently useful: it caps unbounded graph
    /// growth across decode steps (each call would otherwise replay every
    /// prior step's subgraph). Callers wanting device-resident prefixes
    /// should use [`Self::forward_with_latent_kv_context`] instead.
    ///
    /// The fix implemented here is the strategy [`LazyLatentCache`]'s own
    /// module doc already names as the alternative to per-call graph
    /// reuse: "re-creates the cache on the new step's graph (rebinding
    /// realized latents via `const_*_like`)". Each call now starts from
    /// its OWN fresh graph seeded with the REALIZED (host `f32`) prefix,
    /// so no `WriteSlice` destination is ever shared across calls — every
    /// destructive destination has exactly one consumer again, matching
    /// the executor's assumption. Uses only [`LazyLatentCache`]'s public
    /// API (`new` + `append` + `advance_by`), no changes to that type.
    fn rebind_latent_cache_fresh_graph(
        &self, cache: LazyLatentCache, cached_len: usize,
    ) -> Result<LazyLatentCache> {
        if cached_len == 0 {
            return Ok(cache);
        }
        let cfg = &self.config;
        let fresh_anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let mut fresh = LazyLatentCache::new(
            &fresh_anchor,
            cache.n_layers(),
            cache.max_seq_len(),
            vec![cache.slot_trailing(0).to_vec(), cache.slot_trailing(1).to_vec()],
            DType::F32,
        )?;
        for layer in 0..cache.n_layers() {
            let latent_prefix = cache.slot(layer, 0).realize_f32();
            let kpe_prefix = cache.slot(layer, 1).realize_f32();
            let latent_c = fresh_anchor.const_f32_like(
                latent_prefix, Shape::from_dims(&[cached_len, cfg.kv_lora_rank]),
            );
            let kpe_c = fresh_anchor.const_f32_like(
                kpe_prefix, Shape::from_dims(&[cached_len, cfg.qk_rope_head_dim]),
            );
            fresh = fresh.append(layer, &[&latent_c, &kpe_c])?;
        }
        Ok(fresh.advance_by(cached_len))
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
        let lm_head_w = match &self.weights.lm_head {
            Some(w) => w.clone(),
            None => WeightStorage::F32(self.weights.token_embedding.clone()),
        };
        Ok(lm_head_w.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
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
                "DeepSeek2Model::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "DeepSeek2Model::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        if weights.layers.len() != cfg.num_hidden_layers {
            return Err(crate::Error::Msg(format!(
                "DeepSeek2Weights: layers length ({}) must match num_hidden_layers ({})",
                weights.layers.len(), cfg.num_hidden_layers,
            )).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, cfg.qk_rope_head_dim,
        );

        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, idx, &rope_cos, &rope_sin)?;
        }
        h.rms_norm_affine(std::sync::Arc::clone(&weights.final_norm_gain), cfg.rms_norm_eps)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &DeepSeek2LayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
        let attn = self.mla_attention(&x_norm, &layer.mla, rope_cos, rope_sin)?;
        let h1 = x.add(&attn)?;

        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
        let expected_moe = cfg.layer_uses_moe(layer_idx);
        let mlp_out = match (&layer.ffn, expected_moe) {
            (DeepSeek2FfnWeights::Dense(w), false) => self.apply_dense_mlp(&h1_norm, w)?,
            (DeepSeek2FfnWeights::Moe(w), true) => self.apply_moe(&h1_norm, w)?,
            _ => return Err(crate::Error::Msg(format!(
                "DeepSeek-V2 layer {layer_idx}: FFN weight kind does not match \
                 config-derived kind (uses_moe={expected_moe}) — config + weights are inconsistent",
            )).bt()),
        };
        h1.add(&mlp_out)
    }

    /// Cached-decode sibling of [`Self::apply_layer`]: same norms, same
    /// residuals, same dense/MoE FFN dispatch — only attention goes
    /// through [`Self::mla_attention_cached`] (or, when `absorb` is set,
    /// its weight-absorption sibling
    /// [`Self::mla_attention_cached_absorbed`]) and the cache threads
    /// through functionally.
    fn apply_layer_cached(
        &self,
        x: &LazyTensor,
        layer: &DeepSeek2LayerWeights,
        layer_idx: usize,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        cache: LazyLatentCache,
        cached_len: usize,
        absorb: bool,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        let cfg = &self.config;

        let x_norm = x.rms_norm_affine(std::sync::Arc::clone(&layer.input_norm_gain), cfg.rms_norm_eps)?;
        let (attn, cache) = if absorb {
            self.mla_attention_cached_absorbed(
                &x_norm, &layer.mla, rope_cos, rope_sin, mask, cache, layer_idx, cached_len,
            )?
        } else {
            self.mla_attention_cached(
                &x_norm, &layer.mla, rope_cos, rope_sin, mask, cache, layer_idx, cached_len,
            )?
        };
        let h1 = x.add(&attn)?;

        let h1_norm = h1.rms_norm_affine(std::sync::Arc::clone(&layer.post_attn_norm_gain), cfg.rms_norm_eps)?;
        let expected_moe = cfg.layer_uses_moe(layer_idx);
        let mlp_out = match (&layer.ffn, expected_moe) {
            (DeepSeek2FfnWeights::Dense(w), false) => self.apply_dense_mlp(&h1_norm, w)?,
            (DeepSeek2FfnWeights::Moe(w), true) => self.apply_moe(&h1_norm, w)?,
            _ => return Err(crate::Error::Msg(format!(
                "DeepSeek-V2 layer {layer_idx}: FFN weight kind does not match \
                 config-derived kind (uses_moe={expected_moe}) — config + weights are inconsistent",
            )).bt()),
        };
        let out = h1.add(&mlp_out)?;
        Ok((out, cache))
    }

    /// Cached-decode sibling of [`Self::mla_attention`]. `x` is the
    /// post-input-norm hidden state for the NEW tokens only, `(1,
    /// seq_new, hidden)`. `rope_cos`/`rope_sin` are `[seq_new, rope]`
    /// tables built at the absolute start position `cached_len`. `mask`
    /// is the shared `(1, 1, seq_new, total_len)` decode causal mask.
    ///
    /// Appends this step's post-norm compressed latent and post-RoPE
    /// `k_pe` to the cache, reads back the FULL attended prefix
    /// (`cached_len + seq_new` tokens), up-projects it through
    /// `kv_b_proj`, and attends the new queries against it.
    fn mla_attention_cached(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MlaWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        cache: LazyLatentCache,
        layer_idx: usize,
        cached_len: usize,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        let cfg = &self.config;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;
        let kvr = cfg.kv_lora_rank;
        let s = x.shape().dims()[1];

        // ---- Q projection (plain or LoRA), split + RoPE on the pe half ------
        let q = match &w.q_proj {
            DeepSeek2QProj::Plain(plain) => {
                plain.apply_linear(x, cfg.hidden_size, n_heads * q_head_dim)
            }
            DeepSeek2QProj::Lora { a, norm_gain, b } => {
                let lo = a.apply_linear(x, cfg.hidden_size, norm_gain.len());
                let lo_norm = lo.rms_norm_affine(Arc::clone(norm_gain), cfg.rms_norm_eps)?;
                b.apply_linear(&lo_norm, norm_gain.len(), n_heads * q_head_dim)
            }
        };
        let q = q.split_heads(n_heads, q_head_dim)?;
        let q_nope = q.slice(3_usize, 0, nope)?;
        let q_pe = q.slice(3_usize, nope, rope)?;
        let q_pe_rot = apply_interleaved_partial_rope(&q_pe, rope_cos, rope_sin, rope, rope)?;

        // ---- New KV latents for this step's tokens only ---------------------
        let kv_a = w.kv_a_proj_with_mqa.apply_linear(x, cfg.hidden_size, kvr + rope);
        let compressed_kv = kv_a.slice(2_usize, 0, kvr)?;
        let k_pe_single = kv_a.slice(2_usize, kvr, rope)?;

        let compressed_kv_norm = compressed_kv.rms_norm_affine(
            Arc::clone(&w.kv_a_layernorm_gain), cfg.rms_norm_eps,
        )?;

        let k_pe_single_h = k_pe_single.split_heads(1, rope)?;
        let k_pe_rot = apply_interleaved_partial_rope(&k_pe_single_h, rope_cos, rope_sin, rope, rope)?;

        // ---- Append to cache (squeeze the batch==1 dim) ----------------------
        let latent_new = compressed_kv_norm.reshape(Shape::from_dims(&[s, kvr]))?;
        let kpe_new = k_pe_rot.reshape(Shape::from_dims(&[s, rope]))?;
        let cache = cache.append(layer_idx, &[&latent_new, &kpe_new])?;

        // ---- Read back the FULL attended prefix (cached + new) ---------------
        // Do NOT use cache.slot() here — current_seq_len hasn't advanced yet
        // this step, so it would clip off the tokens just appended.
        let total = cached_len + s;
        let latent_all = cache
            .slot_buffer_full(layer_idx, 0)
            .slice(0_usize, 0, total)?
            .reshape(Shape::from_dims(&[1, total, kvr]))?;
        let kpe_all = cache
            .slot_buffer_full(layer_idx, 1)
            .slice(0_usize, 0, total)?
            .reshape(Shape::from_dims(&[1, 1, total, rope]))?
            .broadcast_to(Shape::from_dims(&[1, n_heads, total, rope]))?;

        // ---- Up-project the whole latent prefix (cached + new) ---------------
        let kv = w.kv_b_proj.apply_linear(&latent_all, kvr, n_heads * (nope + v_dim));
        let kv = kv.split_heads(n_heads, nope + v_dim)?;
        let k_nope = kv.slice(3_usize, 0, nope)?;
        let v = kv.slice(3_usize, nope, v_dim)?;

        // Cat Q and K along the head_dim axis.
        let q_full = q_nope.concat(&q_pe_rot, 3_usize)?; // (1, H, s, qhd)
        let k_full = k_nope.concat(&kpe_all, 3_usize)?; // (1, H, total, qhd)

        // ---- Attention --------------------------------------------------------
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (q_head_dim as f64).sqrt();
        let scores = q_full.matmul(&k_t)?; // (1, H, s, total)
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?; // (1, H, s, v_dim)

        let merged = ctx.merge_heads()?;
        let out = w.o_proj.apply_linear(&merged, n_heads * v_dim, cfg.hidden_size);
        Ok((out, cache))
    }

    /// Weight-absorption sibling of [`Self::mla_attention_cached`] — the
    /// DeepSeek decode trick. [`Self::mla_attention_cached`] re-projects
    /// the WHOLE cached latent prefix through `kv_b_proj` every step
    /// (cost `O(total · kvr · H · (nope+v))`, growing with context). This
    /// fn instead folds `kv_b_proj`'s per-head `W_UK`/`W_UV` slices into
    /// the query/context math and attends DIRECTLY against the cached
    /// (post-norm) latent `c` — no per-step re-projection of the prefix:
    ///
    /// ```text
    /// k_nope[h] = c @ W_UK[h]                      // [total, nope]
    /// scores_nope[h] = q_nope[h] @ k_nope[h]^T
    ///                = q_nope[h] @ (c @ W_UK[h])^T
    ///                = (q_nope[h] @ W_UK[h]^T) @ c^T
    ///                = q_absorbed[h] @ c^T          // q_absorbed: [s, kvr]
    ///
    /// v[h] = c @ W_UV[h]                            // [total, v_dim]
    /// ctx[h] = probs[h] @ v[h]
    ///        = probs[h] @ (c @ W_UV[h])
    ///        = (probs[h] @ c) @ W_UV[h]
    ///        = ctx_latent[h] @ W_UV[h]               // ctx_latent: [s, kvr]
    /// ```
    ///
    /// The RoPE half is unchanged (`scores_rope = q_pe_rot @ kpe_all^T`);
    /// concat-then-matmul on `[q_nope | q_pe]` / `[k_nope | kpe]` (the
    /// non-absorbed path) is exactly the sum of the two partial matmuls,
    /// so `scores = (scores_nope + scores_rope) * scale` reproduces the
    /// same pre-softmax quantity up to summation order.
    ///
    /// Per-step cost becomes `O(s·H·nope·kvr) + O(s·total·H·kvr) +
    /// O(s·H·kvr·v)` — the widest term still scales with `total`, but
    /// nothing re-touches the prefix through a weight matrix.
    ///
    /// **Not bit-identical** to [`Self::mla_attention_cached`]: the
    /// absorption identity reassociates the matmul
    /// (`(q·W_UK)·c` vs `q·(W_UK·c)`), and float addition is not
    /// associative, so summation order differs even though the math is
    /// exact. Mathematically equivalent, tight-epsilon-verified (see the
    /// `forward_with_latent_cache_absorbed_matches_*` tests). Prefill-heavy
    /// callers may prefer the non-absorbed sibling — up-projecting the
    /// prefix once (compute-bound, small `total`) can beat this path's
    /// wider per-step matmuls (memory-bound, large `total`); real systems
    /// switch on sequence length, deferred here as a follow-up.
    fn mla_attention_cached_absorbed(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MlaWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        mask: &LazyTensor,
        cache: LazyLatentCache,
        layer_idx: usize,
        cached_len: usize,
    ) -> Result<(LazyTensor, LazyLatentCache)> {
        let cfg = &self.config;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;
        let kvr = cfg.kv_lora_rank;
        let s = x.shape().dims()[1];

        // ---- Q projection (plain or LoRA), split + RoPE on the pe half ------
        // Identical to `mla_attention_cached` up through obtaining
        // `q_nope`/`q_pe_rot`, the cache append, and the prefix read-back.
        let q = match &w.q_proj {
            DeepSeek2QProj::Plain(plain) => {
                plain.apply_linear(x, cfg.hidden_size, n_heads * q_head_dim)
            }
            DeepSeek2QProj::Lora { a, norm_gain, b } => {
                let lo = a.apply_linear(x, cfg.hidden_size, norm_gain.len());
                let lo_norm = lo.rms_norm_affine(Arc::clone(norm_gain), cfg.rms_norm_eps)?;
                b.apply_linear(&lo_norm, norm_gain.len(), n_heads * q_head_dim)
            }
        };
        let q = q.split_heads(n_heads, q_head_dim)?;
        let q_nope = q.slice(3_usize, 0, nope)?;
        let q_pe = q.slice(3_usize, nope, rope)?;
        let q_pe_rot = apply_interleaved_partial_rope(&q_pe, rope_cos, rope_sin, rope, rope)?;

        // ---- New KV latents for this step's tokens only ---------------------
        let kv_a = w.kv_a_proj_with_mqa.apply_linear(x, cfg.hidden_size, kvr + rope);
        let compressed_kv = kv_a.slice(2_usize, 0, kvr)?;
        let k_pe_single = kv_a.slice(2_usize, kvr, rope)?;

        let compressed_kv_norm = compressed_kv.rms_norm_affine(
            Arc::clone(&w.kv_a_layernorm_gain), cfg.rms_norm_eps,
        )?;

        let k_pe_single_h = k_pe_single.split_heads(1, rope)?;
        let k_pe_rot = apply_interleaved_partial_rope(&k_pe_single_h, rope_cos, rope_sin, rope, rope)?;

        // ---- Append to cache (squeeze the batch==1 dim) ----------------------
        let latent_new = compressed_kv_norm.reshape(Shape::from_dims(&[s, kvr]))?;
        let kpe_new = k_pe_rot.reshape(Shape::from_dims(&[s, rope]))?;
        let cache = cache.append(layer_idx, &[&latent_new, &kpe_new])?;

        // ---- Read back the FULL attended prefix (cached + new) ---------------
        // Do NOT use cache.slot() here — current_seq_len hasn't advanced yet
        // this step, so it would clip off the tokens just appended. Read the
        // latent directly as a rank-4 (1, 1, total, kvr) view — the absorbed
        // path attends against it twice (as K via `c_t`, as V via
        // `c_broadcast`), both needing the `n_heads`-broadcast shape, so
        // there is no use for the rank-3 (1, total, kvr) view
        // `mla_attention_cached` builds for its `kv_b_proj` up-projection.
        let total = cached_len + s;
        let latent_all = cache
            .slot_buffer_full(layer_idx, 0)
            .slice(0_usize, 0, total)?
            .reshape(Shape::from_dims(&[1, 1, total, kvr]))?;
        let kpe_all = cache
            .slot_buffer_full(layer_idx, 1)
            .slice(0_usize, 0, total)?
            .reshape(Shape::from_dims(&[1, 1, total, rope]))?
            .broadcast_to(Shape::from_dims(&[1, n_heads, total, rope]))?;

        // ---- Absorbed weights: kv_b_proj split into per-head W_UK^T / W_UV --
        let (w_uk_t_data, w_uv_data) = absorb_split_kv_b(&w.kv_b_proj, kvr, n_heads, nope, v_dim)?;
        let w_uk_t = x.const_f32_like(w_uk_t_data, Shape::from_dims(&[1, n_heads, nope, kvr]));
        let w_uv = x.const_f32_like(w_uv_data, Shape::from_dims(&[1, n_heads, kvr, v_dim]));

        // ---- q_absorbed[h] = q_nope[h] @ W_UK[h]^T ---------------------------
        let q_absorbed = q_nope.matmul(&w_uk_t)?; // (1,H,s,nope) @ (1,H,nope,kvr) -> (1,H,s,kvr)

        // ---- Attend directly against the cached latent (no kv_b_proj) -------
        let c_broadcast = latent_all.broadcast_to(Shape::from_dims(&[1, n_heads, total, kvr]))?;
        let c_t = c_broadcast.transpose()?; // (1, H, kvr, total)
        let scores_nope = q_absorbed.matmul(&c_t)?; // (1, H, s, total)

        let kpe_t = kpe_all.transpose()?; // (1, H, rope, total)
        let scores_rope = q_pe_rot.matmul(&kpe_t)?; // (1, H, s, total)

        let scale = 1.0_f64 / (q_head_dim as f64).sqrt();
        let scores = scores_nope.add(&scores_rope)?;
        let scores_scaled = scores.mul_scalar(scale);
        let scores_masked = scores_scaled.broadcast_add(mask)?;
        let probs = scores_masked.softmax_last_dim()?;

        // ---- ctx_latent[h] = probs[h] @ c; ctx[h] = ctx_latent[h] @ W_UV[h] -
        let ctx_latent = probs.matmul(&c_broadcast)?; // (1, H, s, kvr)
        let ctx = ctx_latent.matmul(&w_uv)?; // (1, H, s, v_dim)

        let merged = ctx.merge_heads()?;
        let out = w.o_proj.apply_linear(&merged, n_heads * v_dim, cfg.hidden_size);
        Ok((out, cache))
    }

    fn mla_attention(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MlaWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;

        // ---- Q projection (plain or LoRA) -----------------------------------
        let q = match &w.q_proj {
            DeepSeek2QProj::Plain(plain) => {
                plain.apply_linear(x, cfg.hidden_size, n_heads * q_head_dim)
            }
            DeepSeek2QProj::Lora { a, norm_gain, b } => {
                let lo = a.apply_linear(x, cfg.hidden_size, norm_gain.len());
                let lo_norm = lo.rms_norm_affine(Arc::clone(norm_gain), cfg.rms_norm_eps)?;
                b.apply_linear(&lo_norm, norm_gain.len(), n_heads * q_head_dim)
            }
        };
        let _ = (batch, seq);
        let q = q.split_heads(n_heads, q_head_dim)?;
        // Split Q on the last dim into (q_nope, q_pe).
        let q_nope = q.slice(3_usize, 0, nope)?;
        let q_pe = q.slice(3_usize, nope, rope)?;

        // ---- KV compressed projection ---------------------------------------
        let kv_a = w.kv_a_proj_with_mqa.apply_linear(
            x, cfg.hidden_size, cfg.kv_lora_rank + rope,
        );
        let compressed_kv = kv_a.slice(2_usize, 0, cfg.kv_lora_rank)?;
        let k_pe_single = kv_a.slice(2_usize, cfg.kv_lora_rank, rope)?;
        // k_pe shape (b, seq, rope) → (b, 1, seq, rope) for MQA broadcast.
        let k_pe_single_h = k_pe_single.split_heads(1, rope)?;

        let compressed_kv_norm = compressed_kv.rms_norm_affine(std::sync::Arc::clone(&w.kv_a_layernorm_gain), cfg.rms_norm_eps)?;
        let kv = w.kv_b_proj.apply_linear(
            &compressed_kv_norm, cfg.kv_lora_rank, n_heads * (nope + v_dim),
        );
        let kv = kv.split_heads(n_heads, nope + v_dim)?;
        let k_nope = kv.slice(3_usize, 0, nope)?;
        let v = kv.slice(3_usize, nope, v_dim)?;

        // ---- RoPE on q_pe and k_pe (interleaved) ----------------------------
        let q_pe_rot = apply_interleaved_partial_rope(&q_pe, rope_cos, rope_sin, rope, rope)?;
        let k_pe_rot = apply_interleaved_partial_rope(&k_pe_single_h, rope_cos, rope_sin, rope, rope)?;

        // Broadcast k_pe_rot from (b, 1, seq, rope) to (b, n_heads, seq, rope).
        let k_pe_full = k_pe_rot
            .broadcast_to(Shape::from_dims(&[batch, n_heads, seq, rope]))?;

        // Cat Q and K along the head_dim axis.
        let q_full = q_nope.concat(&q_pe_rot, 3_usize)?;
        let k_full = k_nope.concat(&k_pe_full, 3_usize)?;

        // ---- Attention ------------------------------------------------------
        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (q_head_dim as f64).sqrt();
        let scores = q_full.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(x, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let probs = scores_masked.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?; // (b, n_heads, seq, v_dim)

        let merged = ctx.merge_heads()?;
        Ok(w.o_proj.apply_linear(&merged, n_heads * v_dim, cfg.hidden_size))
    }

    fn apply_dense_mlp(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2DenseMlpWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let gate = w.gate.apply_linear(x, h, inter);
        let up = w.up.apply_linear(x, h, inter);
        let activated = match cfg.hidden_activation {
            DeepSeek2Activation::Silu => gate.silu(),
            DeepSeek2Activation::Gelu => gate.gelu_erf(),
        };
        let inner = activated.mul(&up)?;
        Ok(w.down.apply_linear(&inner, inter, h))
    }

    fn apply_moe(
        &self,
        x: &LazyTensor,
        w: &DeepSeek2MoeWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let n_routed = cfg.n_routed_experts.unwrap_or(0);
        let n_shared = cfg.n_shared_experts.unwrap_or(0);
        assert!(n_routed > 0, "MoE block requires n_routed_experts > 0");
        assert_eq!(w.experts.len(), n_routed,
            "MoE weights expert count {} != n_routed_experts {n_routed}",
            w.experts.len());

        // Routed path (dense routing — full softmax × every expert).
        let router_t = x.const_f32_like(
            w.router.clone(),
            Shape::from_dims(&[h, n_routed]),
        );
        let router_logits = x.matmul(&router_t)?;
        let routing_weights = router_logits.softmax_last_dim()?;

        let mut routed_sum: Option<LazyTensor> = None;
        for (ei, ew) in w.experts.iter().enumerate() {
            let gate = ew.gate.apply_linear(x, h, inter);
            let up = ew.up.apply_linear(x, h, inter);
            let activated = match cfg.hidden_activation {
                DeepSeek2Activation::Silu => gate.silu(),
                DeepSeek2Activation::Gelu => gate.gelu_erf(),
            };
            let inner = activated.mul(&up)?;
            let expert_out = ew.down.apply_linear(&inner, inter, h);
            let w_col = routing_weights.slice(2_usize, ei, 1)?;
            let w_bc = w_col.broadcast_to(Shape::from_dims(&[batch, seq, h]))?;
            let gated = expert_out.mul(&w_bc)?;
            routed_sum = Some(match routed_sum {
                Some(s) => s.add(&gated)?,
                None => gated,
            });
        }
        let routed = routed_sum.expect("MoE: at least one expert");

        // Shared-expert path (always on, no gating).
        if n_shared == 0 {
            return Ok(routed);
        }
        let shared_inter = n_shared * inter;
        let s_gate = w.shared_gate.apply_linear(x, h, shared_inter);
        let s_up = w.shared_up.apply_linear(x, h, shared_inter);
        let s_act = match cfg.hidden_activation {
            DeepSeek2Activation::Silu => s_gate.silu(),
            DeepSeek2Activation::Gelu => s_gate.gelu_erf(),
        };
        let s_inner = s_act.mul(&s_up)?;
        let s_out = w.shared_down.apply_linear(&s_inner, shared_inter, h);
        routed.add(&s_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_mla_weights(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2MlaWeights {
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let q_head_dim = cfg.q_head_dim();
        let nope = cfg.qk_nope_head_dim;
        let rope = cfg.qk_rope_head_dim;
        let v_dim = cfg.v_head_dim;

        let q_proj = match cfg.q_lora_rank {
            None => DeepSeek2QProj::Plain(WeightStorage::F32(vec_of(h * n_heads * q_head_dim, &mut **nb))),
            Some(lora) => DeepSeek2QProj::Lora {
                a: WeightStorage::F32(vec_of(h * lora, &mut **nb)),
                norm_gain: Arc::from(vec![1.0_f32; lora]),
                b: WeightStorage::F32(vec_of(lora * n_heads * q_head_dim, &mut **nb)),
            },
        };
        DeepSeek2MlaWeights {
            q_proj,
            kv_a_proj_with_mqa: WeightStorage::F32(vec_of(h * (cfg.kv_lora_rank + rope), &mut **nb)),
            kv_a_layernorm_gain: Arc::from(vec![1.0_f32; cfg.kv_lora_rank]),
            kv_b_proj: WeightStorage::F32(vec_of(cfg.kv_lora_rank * n_heads * (nope + v_dim), &mut **nb)),
            o_proj: WeightStorage::F32(vec_of(n_heads * v_dim * h, &mut **nb)),
        }
    }

    fn tiny_dense_mlp(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2DenseMlpWeights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        DeepSeek2DenseMlpWeights {
            gate: WeightStorage::F32(vec_of(h * i, &mut **nb)),
            up: WeightStorage::F32(vec_of(h * i, &mut **nb)),
            down: WeightStorage::F32(vec_of(i * h, &mut **nb)),
        }
    }

    fn tiny_moe(cfg: &DeepSeek2Config, nb: &mut Box<dyn FnMut() -> f32>) -> DeepSeek2MoeWeights {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size;
        let n_routed = cfg.n_routed_experts.unwrap_or(0);
        let n_shared = cfg.n_shared_experts.unwrap_or(0);
        let shared_inter = n_shared * inter;
        let router = vec_of(h * n_routed, &mut **nb);
        let experts: Vec<DeepSeek2ExpertWeights> = (0..n_routed)
            .map(|_| DeepSeek2ExpertWeights {
                gate: WeightStorage::F32(vec_of(h * inter, &mut **nb)),
                up: WeightStorage::F32(vec_of(h * inter, &mut **nb)),
                down: WeightStorage::F32(vec_of(inter * h, &mut **nb)),
            })
            .collect();
        DeepSeek2MoeWeights {
            router, experts,
            shared_gate: WeightStorage::F32(vec_of(h * shared_inter, &mut **nb)),
            shared_up: WeightStorage::F32(vec_of(h * shared_inter, &mut **nb)),
            shared_down: WeightStorage::F32(vec_of(shared_inter * h, &mut **nb)),
        }
    }

    fn tiny_weights(cfg: &DeepSeek2Config) -> DeepSeek2Weights {
        let mut s: u32 = 99999;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);

        let layers: Vec<DeepSeek2LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let ffn = if cfg.layer_uses_moe(i) {
                    DeepSeek2FfnWeights::Moe(tiny_moe(cfg, &mut nb))
                } else {
                    DeepSeek2FfnWeights::Dense(tiny_dense_mlp(cfg, &mut nb))
                };
                DeepSeek2LayerWeights {
                    input_norm_gain: Arc::from(vec![1.0_f32; h]),
                    mla: tiny_mla_weights(cfg, &mut nb),
                    post_attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                    ffn,
                }
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let lm_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb)))
        };
        DeepSeek2Weights {
            token_embedding, layers,
            final_norm_gain, lm_head,
        }
    }

    fn tiny_config_lora_q() -> DeepSeek2Config {
        DeepSeek2Config {
            vocab_size: 16, hidden_size: 16,
            intermediate_size: 32, moe_intermediate_size: 8,
            num_hidden_layers: 3,
            num_attention_heads: 4,
            n_shared_experts: Some(1),
            n_routed_experts: Some(2),
            num_experts_per_tok: Some(1),
            moe_layer_freq: 1,
            first_k_dense_replace: 1,  // layer 0 is dense; layers 1, 2 are MoE
            norm_topk_prob: false,
            hidden_activation: DeepSeek2Activation::Silu,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            rope_theta: 10_000.0,
            attention_bias: false,
            q_lora_rank: Some(8),
            qk_rope_head_dim: 4,
            kv_lora_rank: 8,
            v_head_dim: 4,
            qk_nope_head_dim: 4,
        }
    }

    fn tiny_config_plain_q() -> DeepSeek2Config {
        DeepSeek2Config { q_lora_rank: None, ..tiny_config_lora_q() }
    }

    #[test]
    fn forward_shape_and_finite_lora_q() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "got non-finite logit {v}");
        }
    }

    #[test]
    fn forward_shape_and_finite_plain_q() {
        let cfg = tiny_config_plain_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// `first_k_dense_replace` actually skips MoE in early layers.
    #[test]
    fn dense_replace_layer_uses_dense_mlp() {
        let cfg = tiny_config_lora_q();
        assert!(!cfg.layer_uses_moe(0));
        assert!(cfg.layer_uses_moe(1));
        assert!(cfg.layer_uses_moe(2));
    }

    /// MLA k_pe is MQA-shared (single head, broadcast). Zero
    /// out the kv_a_proj_with_mqa columns that produce k_pe
    /// (the last `qk_rope_head_dim` columns) and confirm the
    /// output changes.
    #[test]
    fn mla_k_pe_is_wired() {
        let cfg = DeepSeek2Config { num_hidden_layers: 1, ..tiny_config_lora_q() };
        let h = cfg.hidden_size;
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        let rope = cfg.qk_rope_head_dim;
        let kv_a_full_size = cfg.kv_lora_rank + rope;
        // Zero the k_pe slice (the last `rope` columns of kv_a_proj_with_mqa).
        let mut kv_a_v = match &zeroed.layers[0].mla.kv_a_proj_with_mqa {
            WeightStorage::F32(v) => v.to_vec(),
            _ => panic!(),
        };
        for row in 0..h {
            for j in cfg.kv_lora_rank..kv_a_full_size {
                kv_a_v[row * kv_a_full_size + j] = 0.0;
            }
        }
        zeroed.layers[0].mla.kv_a_proj_with_mqa = WeightStorage::F32(Arc::from(kv_a_v));
        let m_base = DeepSeek2Model { config: cfg.clone(), weights: base };
        let m_zero = DeepSeek2Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, 0).unwrap().realize_f32();
        let b = m_zero.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-8,
            "k_pe path must be wired (zeroing kv_a's rope cols alters output), max_diff = {max_diff}");
    }

    /// Shared expert must contribute alongside routed experts.
    #[test]
    fn shared_expert_contributes() {
        let cfg = DeepSeek2Config {
            // One MoE-only layer.
            num_hidden_layers: 1,
            first_k_dense_replace: 0,
            ..tiny_config_lora_q()
        };
        let base = tiny_weights(&cfg);
        let mut zeroed = base.clone();
        if let DeepSeek2FfnWeights::Moe(m) = &mut zeroed.layers[0].ffn {
            let h = cfg.hidden_size;
            let n_shared = cfg.n_shared_experts.unwrap_or(0);
            let shared_inter = n_shared * cfg.moe_intermediate_size;
            m.shared_gate = WeightStorage::F32(Arc::from(vec![0.0_f32; h * shared_inter]));
            m.shared_up = WeightStorage::F32(Arc::from(vec![0.0_f32; h * shared_inter]));
            m.shared_down = WeightStorage::F32(Arc::from(vec![0.0_f32; shared_inter * h]));
        } else {
            panic!("expected MoE FFN");
        }
        let m_base = DeepSeek2Model { config: cfg.clone(), weights: base };
        let m_zero = DeepSeek2Model { config: cfg, weights: zeroed };
        let toks = [1_u32, 2, 3];
        let a = m_base.forward(&toks, 0).unwrap().realize_f32();
        let b = m_zero.forward(&toks, 0).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-8,
            "shared expert path must contribute, max_diff = {max_diff}");
    }

    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "DeepSeek-V2 forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    /// The MLA cached-decode acceptance test: incremental
    /// `forward_with_latent_cache` steps (prefill 2, decode 1, decode 1)
    /// must reproduce the one-shot `forward` over the same 4 tokens,
    /// row for row. Exercised for both plain-Q and LoRA-Q configs.
    #[test]
    fn forward_with_latent_cache_matches_one_shot_forward() {
        for cfg in [tiny_config_plain_q(), tiny_config_lora_q()] {
            let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
            let vocab = cfg.vocab_size;
            let tokens: Vec<u32> = vec![1, 2, 3, 4];

            // One-shot reference over all 4 tokens.
            let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();

            // Cached run: prefill 2 tokens, then two single-token decode
            // steps, each realized BEFORE the next step is built (true
            // decode order). This ordering originally hit a real
            // `PipelinedExecutor` gap — see
            // `DeepSeek2Model::rebind_latent_cache_fresh_graph`'s doc
            // comment for the full root-cause writeup — where a
            // `WriteSlice` cache-append destination shared across calls on
            // one ever-growing graph got evicted by a later step's append
            // before an earlier step's own attention math had read it.
            // `forward_with_latent_cache` now works around it internally
            // (rebinding each call onto its own fresh graph), so plain
            // per-step realize works here without any test-side fallback.
            let anchor = LazyTensor::from_f32(
                vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
            );
            let cache = LazyLatentCache::new(
                &anchor, cfg.num_hidden_layers, 8,
                vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
                DType::F32,
            ).unwrap();

            let (logits_a, cache) = model.forward_with_latent_cache(&[1, 2], cache).unwrap();
            assert_eq!(logits_a.shape().dims(), &[1, 2, vocab]);
            let logits_a = logits_a.realize_f32();

            let (logits_b, cache) = model.forward_with_latent_cache(&[3], cache).unwrap();
            assert_eq!(logits_b.shape().dims(), &[1, 1, vocab]);
            let logits_b = logits_b.realize_f32();

            let (logits_c, cache) = model.forward_with_latent_cache(&[4], cache).unwrap();
            assert_eq!(logits_c.shape().dims(), &[1, 1, vocab]);
            let logits_c = logits_c.realize_f32();

            assert_eq!(cache.current_seq_len(), 4);

            // Parity: rows 0-1 of one-shot == logits_a; row 2 == logits_b;
            // row 3 == logits_c. Try bit-exact first; only fall back to a
            // tight epsilon if small-ulp summation-order drift shows up.
            let mut bit_exact = true;
            let mut max_diff = 0.0_f32;
            let mut check_row = |ref_row: &[f32], got: &[f32]| {
                for (r, g) in ref_row.iter().zip(got.iter()) {
                    if r.to_bits() != g.to_bits() {
                        bit_exact = false;
                    }
                    max_diff = max_diff.max((r - g).abs());
                }
            };
            check_row(&logits_ref[0..2 * vocab], &logits_a);
            check_row(&logits_ref[2 * vocab..3 * vocab], &logits_b);
            check_row(&logits_ref[3 * vocab..4 * vocab], &logits_c);

            if !bit_exact {
                eprintln!(
                    "forward_with_latent_cache_matches_one_shot_forward: not bit-exact \
                     (q_lora_rank={:?}), max abs diff = {max_diff}", cfg.q_lora_rank,
                );
                assert!(max_diff < 1e-5,
                    "forward_with_latent_cache vs one-shot forward diverge beyond tolerance: \
                     max_diff={max_diff}");
            }
        }
    }

    /// Bad cache geometry (slot count, trailing shape, or capacity) must
    /// surface as a typed `Err`, never a panic.
    #[test]
    fn forward_with_latent_cache_rejects_bad_cache_geometry() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );

        // Wrong slot count (1 slot instead of 2).
        let bad_slots = LazyLatentCache::new(
            &anchor, cfg.num_hidden_layers, 8, vec![vec![cfg.kv_lora_rank]], DType::F32,
        ).unwrap();
        assert!(model.forward_with_latent_cache(&[1, 2], bad_slots).is_err());

        // Wrong trailing shape on slot 0.
        let bad_trailing = LazyLatentCache::new(
            &anchor, cfg.num_hidden_layers, 8,
            vec![vec![cfg.kv_lora_rank + 1], vec![cfg.qk_rope_head_dim]], DType::F32,
        ).unwrap();
        assert!(model.forward_with_latent_cache(&[1, 2], bad_trailing).is_err());

        // Exceeding capacity: max_seq_len 2, feed 3 tokens.
        let small_cap = LazyLatentCache::new(
            &anchor, cfg.num_hidden_layers, 2,
            vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]], DType::F32,
        ).unwrap();
        assert!(model.forward_with_latent_cache(&[1, 2, 3], small_cap).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
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
            "DeepSeek-V2 forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }

    /// Argmax of a logits row (greedy-decode robustness check).
    fn argmax_row(row: &[f32]) -> usize {
        row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0
    }

    /// MLA weight-absorption decode trick: `forward_with_latent_cache_absorbed`
    /// must match the non-absorbed `forward_with_latent_cache` up to a tight
    /// epsilon, for both cached paths run over the identical incremental
    /// (prefill 2, decode 1, decode 1) protocol.
    ///
    /// Bit-exact equality is impossible **by construction**: the absorption
    /// identity `(q·W_UK)·c ≡ q·(W_UK·c)` reassociates the matmul, and IEEE
    /// float addition is not associative, so summation order — and
    /// therefore the last few ULPs — differs even though the math is exact.
    /// We instead assert a tight abs-diff tolerance plus per-row argmax
    /// equality (the greedy-decode-robustness bar the rest of this suite
    /// uses).
    ///
    /// **Tolerance calibration (sabotage-measured):** genuine reassociation
    /// drift on this fixture is ~1.5e-8; a deliberately corrupted `W_UK`
    /// (reading the `W_UV` columns) moves the logits by only ~3–5e-5,
    /// because the tiny ±0.025 LCG weights attenuate the attention path's
    /// contribution to the residual stream. A 1e-4 tolerance therefore
    /// MASKS real absorption bugs — 1e-6 sits ~100× above genuine drift
    /// (platform-robust) and ~30× below the measured corruption signal.
    #[test]
    fn forward_with_latent_cache_absorbed_matches_unabsorbed() {
        for cfg in [tiny_config_plain_q(), tiny_config_lora_q()] {
            let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
            let vocab = cfg.vocab_size;

            let anchor = LazyTensor::from_f32(
                vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
            );
            let cache_u = LazyLatentCache::new(
                &anchor, cfg.num_hidden_layers, 8,
                vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
                DType::F32,
            ).unwrap();
            let cache_a = LazyLatentCache::new(
                &anchor, cfg.num_hidden_layers, 8,
                vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
                DType::F32,
            ).unwrap();

            let mut max_diff = 0.0_f32;
            let mut check = |u: &[f32], a: &[f32]| {
                for (uv, av) in u.iter().zip(a.iter()) {
                    max_diff = max_diff.max((uv - av).abs());
                }
                for row in 0..(u.len() / vocab) {
                    let urow = &u[row * vocab..(row + 1) * vocab];
                    let arow = &a[row * vocab..(row + 1) * vocab];
                    assert_eq!(argmax_row(urow), argmax_row(arow),
                        "argmax mismatch absorbed vs unabsorbed (q_lora_rank={:?}, row={row})",
                        cfg.q_lora_rank);
                }
            };

            let (logits_a_u, cache_u) = model.forward_with_latent_cache(&[1, 2], cache_u).unwrap();
            let (logits_a_a, cache_a) = model.forward_with_latent_cache_absorbed(&[1, 2], cache_a).unwrap();
            check(&logits_a_u.realize_f32(), &logits_a_a.realize_f32());

            let (logits_b_u, cache_u) = model.forward_with_latent_cache(&[3], cache_u).unwrap();
            let (logits_b_a, cache_a) = model.forward_with_latent_cache_absorbed(&[3], cache_a).unwrap();
            check(&logits_b_u.realize_f32(), &logits_b_a.realize_f32());

            let (logits_c_u, cache_u) = model.forward_with_latent_cache(&[4], cache_u).unwrap();
            let (logits_c_a, cache_a) = model.forward_with_latent_cache_absorbed(&[4], cache_a).unwrap();
            check(&logits_c_u.realize_f32(), &logits_c_a.realize_f32());

            assert_eq!(cache_u.current_seq_len(), 4);
            assert_eq!(cache_a.current_seq_len(), 4);

            eprintln!(
                "forward_with_latent_cache_absorbed_matches_unabsorbed: q_lora_rank={:?} \
                 max abs diff = {max_diff}", cfg.q_lora_rank,
            );
            assert!(max_diff < 1e-6,
                "absorbed vs unabsorbed cached decode diverge beyond tolerance: max_diff={max_diff}");
        }
    }

    /// Guards against both cached paths sharing a common bug: the absorbed
    /// cached-decode path must ALSO reproduce the one-shot `forward`
    /// reference over the same 4-token protocol.
    #[test]
    fn forward_with_latent_cache_absorbed_matches_one_shot() {
        for cfg in [tiny_config_plain_q(), tiny_config_lora_q()] {
            let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
            let vocab = cfg.vocab_size;
            let tokens: Vec<u32> = vec![1, 2, 3, 4];

            let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();

            let anchor = LazyTensor::from_f32(
                vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
            );
            let cache = LazyLatentCache::new(
                &anchor, cfg.num_hidden_layers, 8,
                vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
                DType::F32,
            ).unwrap();

            let (logits_a, cache) = model.forward_with_latent_cache_absorbed(&[1, 2], cache).unwrap();
            let logits_a = logits_a.realize_f32();
            let (logits_b, cache) = model.forward_with_latent_cache_absorbed(&[3], cache).unwrap();
            let logits_b = logits_b.realize_f32();
            let (logits_c, cache) = model.forward_with_latent_cache_absorbed(&[4], cache).unwrap();
            let logits_c = logits_c.realize_f32();
            assert_eq!(cache.current_seq_len(), 4);

            let mut max_diff = 0.0_f32;
            let mut check_row = |ref_row: &[f32], got: &[f32]| {
                for (r, g) in ref_row.iter().zip(got.iter()) {
                    max_diff = max_diff.max((r - g).abs());
                }
                assert_eq!(argmax_row(ref_row), argmax_row(got),
                    "argmax mismatch absorbed cached decode vs one-shot forward \
                     (q_lora_rank={:?})", cfg.q_lora_rank);
            };
            check_row(&logits_ref[0..2 * vocab], &logits_a);
            check_row(&logits_ref[2 * vocab..3 * vocab], &logits_b);
            check_row(&logits_ref[3 * vocab..4 * vocab], &logits_c);

            eprintln!(
                "forward_with_latent_cache_absorbed_matches_one_shot: q_lora_rank={:?} \
                 max abs diff = {max_diff}", cfg.q_lora_rank,
            );
            assert!(max_diff < 1e-6,
                "forward_with_latent_cache_absorbed vs one-shot forward diverge beyond tolerance: \
                 max_diff={max_diff}");
        }
    }

    /// `absorb_split_kv_b` is F32-only and length-checked — both must
    /// surface as a typed `Err`, never a panic.
    #[test]
    fn absorb_split_kv_b_rejects_non_f32() {
        let cfg = tiny_config_plain_q();
        let n_heads = cfg.num_attention_heads;
        let nope = cfg.qk_nope_head_dim;
        let v_dim = cfg.v_head_dim;
        let kvr = cfg.kv_lora_rank;
        let out_features = n_heads * (nope + v_dim);

        // Non-F32 storage.
        let bf16_w = WeightStorage::BF16(Arc::from(vec![half::bf16::ZERO; kvr * out_features]));
        assert!(absorb_split_kv_b(&bf16_w, kvr, n_heads, nope, v_dim).is_err());

        // F32 but wrong length.
        let bad_len_w = WeightStorage::F32(Arc::from(vec![0.0_f32; kvr * out_features - 1]));
        assert!(absorb_split_kv_b(&bad_len_w, kvr, n_heads, nope, v_dim).is_err());
    }

    // ---- forward_with_latent_kv_context (MLA increment 4 — persistent -----
    // ---- cross-graph N-slot decode cache) ----------------------------------

    /// MLA increment 4 — the persistent (cross-graph) N-slot decode cache.
    /// `forward_with_latent_kv_context` must reproduce a one-shot `forward`
    /// over the full sequence, run incrementally (prefill 3, decode 1),
    /// through the device-resident `LatentKvCache` + `InferenceContext` +
    /// `Op::WriteSlice` path — no host round-trip of the cached prefix
    /// (contrast with `forward_with_latent_cache` /
    /// `forward_with_latent_cache_absorbed`, which rebind the realized
    /// prefix onto a fresh graph every call via
    /// `rebind_latent_cache_fresh_graph`). Exercised for both plain-Q and
    /// LoRA-Q configs.
    #[test]
    fn forward_with_latent_kv_context_decode_matches_one_shot() {
        for cfg in [tiny_config_plain_q(), tiny_config_lora_q()] {
            let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };
            let vocab = cfg.vocab_size;

            let prompt = [1_u32, 2, 3];
            let next_token = 4_u32;
            let full = [prompt[0], prompt[1], prompt[2], next_token];

            // References: one-shot forward, last-position row, over the
            // 3-token prompt and the full 4-token sequence.
            let prefill_logits = model.forward(&prompt, 0).unwrap();
            let prefill_expected = prefill_logits
                .slice(1_usize, prompt.len() - 1, 1).unwrap()
                .reshape(Shape::from_dims(&[vocab])).unwrap()
                .realize_f32();

            let full_logits = model.forward(&full, 0).unwrap();
            let full_expected = full_logits
                .slice(1_usize, full.len() - 1, 1).unwrap()
                .reshape(Shape::from_dims(&[vocab])).unwrap()
                .realize_f32();

            // Cached run through the persistent path.
            let mut cache = crate::inference_context::LatentKvCache::with_capacity(
                cfg.num_hidden_layers,
                /* max_seq_len */ 8,
                vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
                DType::F32,
                &Device::cpu(),
            ).expect("LatentKvCache::with_capacity");
            let mut ctx = crate::inference_context::InferenceContext::new(Device::cpu());

            let prefill_actual = model
                .forward_with_latent_kv_context(&prompt, &mut cache, &mut ctx)
                .expect("prefill");
            assert_eq!(cache.cached_len, prompt.len());

            let decode_actual = model
                .forward_with_latent_kv_context(&[next_token], &mut cache, &mut ctx)
                .expect("decode");
            assert_eq!(cache.cached_len, full.len());

            let mut max_diff = 0.0_f32;
            for (a, b) in prefill_actual.iter().zip(prefill_expected.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            for (a, b) in decode_actual.iter().zip(full_expected.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
            eprintln!(
                "forward_with_latent_kv_context_decode_matches_one_shot: q_lora_rank={:?} \
                 max abs diff = {max_diff}", cfg.q_lora_rank,
            );
            // Tolerance calibration (sabotage-measured, 2026-07-08):
            // genuine drift is ~7.5e-9 / ~1.5e-8 (plain-Q / LoRA-Q) from
            // matmul reassociation vs the one-shot reference (same family
            // as `forward_with_latent_cache_absorbed_matches_*`); a
            // corrupted dyn write offset (cached_len_sym bound to 0, so a
            // decode step overwrites the prefix) moves the logits by
            // ~7.7e-3. 1e-6 sits ~67x above genuine drift and ~7700x
            // below the corruption signal.
            assert!(max_diff < 1e-6,
                "forward_with_latent_kv_context vs one-shot forward diverge beyond tolerance: \
                 max_diff={max_diff}");

            // Version contract: two forward_with_latent_kv_context calls ⇒
            // both slots of every layer bumped exactly twice.
            for li in 0..cfg.num_hidden_layers {
                let layer = cache.layers[li].as_ref().expect("layer populated");
                assert_eq!(layer[0].version, 2, "latent slot version (q_lora_rank={:?})", cfg.q_lora_rank);
                assert_eq!(layer[1].version, 2, "k_pe slot version (q_lora_rank={:?})", cfg.q_lora_rank);
            }
        }
    }

    /// Bad cache geometry (layer count, slot count, trailing shape, or
    /// capacity overflow) must surface as a typed `Err`, never a panic.
    #[test]
    fn forward_with_latent_kv_context_rejects_bad_geometry() {
        let cfg = tiny_config_lora_q();
        let model = DeepSeek2Model { config: cfg.clone(), weights: tiny_weights(&cfg) };

        // Wrong n_layers.
        let mut bad_layers = crate::inference_context::LatentKvCache::with_capacity(
            cfg.num_hidden_layers + 1, 8,
            vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
            DType::F32, &Device::cpu(),
        ).unwrap();
        let mut ctx = crate::inference_context::InferenceContext::new(Device::cpu());
        assert!(model.forward_with_latent_kv_context(&[1, 2], &mut bad_layers, &mut ctx).is_err());

        // Wrong slot count (1 slot instead of 2).
        let mut bad_slots = crate::inference_context::LatentKvCache::with_capacity(
            cfg.num_hidden_layers, 8, vec![vec![cfg.kv_lora_rank]], DType::F32, &Device::cpu(),
        ).unwrap();
        let mut ctx = crate::inference_context::InferenceContext::new(Device::cpu());
        assert!(model.forward_with_latent_kv_context(&[1, 2], &mut bad_slots, &mut ctx).is_err());

        // Wrong trailing shape on slot 0.
        let mut bad_trailing = crate::inference_context::LatentKvCache::with_capacity(
            cfg.num_hidden_layers, 8,
            vec![vec![cfg.kv_lora_rank + 1], vec![cfg.qk_rope_head_dim]],
            DType::F32, &Device::cpu(),
        ).unwrap();
        let mut ctx = crate::inference_context::InferenceContext::new(Device::cpu());
        assert!(model.forward_with_latent_kv_context(&[1, 2], &mut bad_trailing, &mut ctx).is_err());

        // Capacity overflow: max_seq_len 2, feed 3 tokens.
        let mut small_cap = crate::inference_context::LatentKvCache::with_capacity(
            cfg.num_hidden_layers, 2,
            vec![vec![cfg.kv_lora_rank], vec![cfg.qk_rope_head_dim]],
            DType::F32, &Device::cpu(),
        ).unwrap();
        let mut ctx = crate::inference_context::InferenceContext::new(Device::cpu());
        assert!(model.forward_with_latent_kv_context(&[1, 2, 3], &mut small_cap, &mut ctx).is_err());
    }
}

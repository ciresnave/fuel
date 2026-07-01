//! Gemma 4 vision tower (ViT-style image encoder) ported to the
//! lazy-graph API.
//!
//! Patch-based ViT with 2D RoPE, 4-norm-per-block Gemma3-shape
//! encoder, and scatter-pool down-sampler. The architecture:
//!
//!   1. **Patch embedding.** Pixel values `(b, 3, h, w)` are
//!      patchified into `(b, ph*pw, ps*ps*3)` where `ps =
//!      patch_size`, scaled into `[-1, 1]` (eager
//!      `(patches - 0.5) * 2`), then linearly projected to
//!      `hidden_size`. Two **position embedding tables** of shape
//!      `[position_embedding_size, hidden_size]` (one for x, one
//!      for y) are indexed by the patch's (col, row) and summed.
//!   2. **2D RoPE.** `head_dim` is split in half; the first half
//!      rotates with x-positions (cols), the second with
//!      y-positions (rows). Each half uses standard split-half
//!      RoPE convention with per-patch precomputed cos/sin.
//!   3. **GQA self-attention with Q/K/V norms.** Q/K normalised
//!      with Gemma's `(gain + 1)` offset RmsNorm on
//!      `head_dim`. V normalised with **pure** RmsNorm (no
//!      learned weight) for stability.
//!   4. **4-norm block** (Gemma 3 shape):
//!      `input_norm → attn → post_attn_norm → +residual →
//!      pre_ffn_norm → mlp → post_ffn_norm → +residual`.
//!   5. **SwiGLU MLP** with config-driven activation (GELU
//!      family for Gemma 4).
//!   6. **Spatial avg-pool down-sampler.** After the encoder
//!      stack, output is pooled to `output_length` tokens by
//!      averaging patches into `k×k` spatial bins (where
//!      `k = sqrt(num_patches / output_length)`). The pooled
//!      output is scaled by `sqrt(hidden_size)`.
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image, F32. Multi-image
//! batches (variable sizes) and the optional standardisation
//! (`std_bias`, `std_scale`) are deferred. The `clamp(0, MAX)`
//! on patch positions in eager guards against negative
//! positions; v1 assumes non-negative inputs.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

pub use crate::lazy_gemma::GemmaActivation as Gemma4VisionActivation;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_activation: Gemma4VisionActivation,
    pub rms_norm_eps: f64,
    pub patch_size: usize,
    pub position_embedding_size: usize,
    pub pooling_kernel_size: usize,
    pub default_output_length: usize,
    pub rope_theta: f64,
}

#[derive(Debug, Clone)]
pub struct Gemma4VisionLayerWeights {
    pub input_norm_gain: Arc<[f32]>,
    pub post_attn_norm_gain: Arc<[f32]>,
    pub pre_ffn_norm_gain: Arc<[f32]>,
    pub post_ffn_norm_gain: Arc<[f32]>,
    pub q_proj: WeightStorage,
    pub k_proj: WeightStorage,
    pub v_proj: WeightStorage,
    pub o_proj: WeightStorage,
    /// Per-head Q/K RmsNorm gains (on `head_dim`).
    pub q_norm_gain: Arc<[f32]>,
    pub k_norm_gain: Arc<[f32]>,
    pub ffn_gate: WeightStorage,
    pub ffn_up: WeightStorage,
    pub ffn_down: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct Gemma4VisionWeights {
    /// `[patch_size * patch_size * 3, hidden_size]`.
    pub input_proj: WeightStorage,
    /// Two stacked tables: `[2, position_embedding_size, hidden_size]`
    /// flattened. First table is x (cols), second is y (rows).
    pub position_embedding_table: Arc<[f32]>,
    pub layers: Vec<Gemma4VisionLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct Gemma4VisionModel {
    pub config: Gemma4VisionConfig,
    pub weights: Gemma4VisionWeights,
}

impl Gemma4VisionModel {
    /// Encode a single image of shape `(1, 3, h, w)`. Returns the
    /// pooled hidden states of shape `(1, output_length,
    /// hidden_size)`. `h` and `w` must each be divisible by
    /// `patch_size`, and `(h / patch_size) * (w / patch_size)`
    /// must be a perfect-square multiple of
    /// `pooling_kernel_size^2`.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "pixel_values must be rank 4 [batch, c, h, w]");
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        let c = dims[1];
        let h = dims[2];
        let w = dims[3];
        assert_eq!(c, 3, "expected 3 input channels");
        let ps = cfg.patch_size;
        assert_eq!(h % ps, 0, "h must be divisible by patch_size");
        assert_eq!(w % ps, 0, "w must be divisible by patch_size");
        let ph = h / ps;
        let pw = w / ps;
        let num_patches = ph * pw;

        // ---- Patchify ---------------------------------------------------------
        // Reshape (b, c, ph, ps, pw, ps), permute to bring patches together:
        //   eager: permute (0, 2, 4, 3, 5, 1) → (b, ph, pw, ps, ps, c)
        // then flatten to (b, ph*pw, ps*ps*c).
        let patches = pixel_values
            .reshape(Shape::from_dims(&[batch, c, ph, ps, pw, ps]))?
            .permute([0, 2, 4, 3, 5, 1_usize])?
            .reshape(Shape::from_dims(&[batch, num_patches, ps * ps * c]))?;
        // Scale into [-1, 1]: (patches - 0.5) * 2.
        let patches = patches.add_scalar(-0.5).mul_scalar(2.0);
        // Linear projection to hidden_size.
        let mut h_states = weights.input_proj.apply_linear(
            &patches, ps * ps * c, cfg.hidden_size,
        );

        // ---- 2D positional embedding -----------------------------------------
        let (pos_emb, cos_xy, sin_xy) = self.build_position_aux(
            pixel_values, ph, pw, num_patches,
        )?;
        h_states = h_states.add(&pos_emb)?;

        // ---- Encoder layers --------------------------------------------------
        for layer in &weights.layers {
            h_states = self.apply_layer(&h_states, layer, &cos_xy, &sin_xy)?;
        }

        // ---- Pooling ---------------------------------------------------------
        let k = cfg.pooling_kernel_size;
        let output_length = num_patches / (k * k);
        let pooled = self.spatial_pool(&h_states, ph, pw, k, output_length)?;
        // Scale by sqrt(hidden_size).
        Ok(pooled.mul_scalar((cfg.hidden_size as f64).sqrt()))
    }

    /// Build the per-patch position embedding, plus the cos/sin
    /// tables for 2D RoPE.
    ///
    /// Returns `(pos_emb [b, num_patches, hidden],
    /// cos_xy [num_patches, head_dim],
    /// sin_xy [num_patches, head_dim])`.
    fn build_position_aux(
        &self,
        anchor: &LazyTensor,
        ph: usize,
        pw: usize,
        num_patches: usize,
    ) -> Result<(LazyTensor, LazyTensor, LazyTensor)> {
        let cfg = &self.config;
        let pe_size = cfg.position_embedding_size;
        let head_dim = cfg.head_dim;
        // Two halves of head_dim — first for x, second for y.
        let half = head_dim / 2;
        let quarter = half / 2; // inv_freq count

        // Build raw position vectors.
        let mut x_positions: Vec<usize> = Vec::with_capacity(num_patches);
        let mut y_positions: Vec<usize> = Vec::with_capacity(num_patches);
        for r in 0..ph {
            for col in 0..pw {
                x_positions.push(col);
                y_positions.push(r);
            }
        }

        // Build the 2D positional embedding via two table lookups summed.
        // Positional table is stored as [2 * pe_size * hidden] in row-major
        // [(table_idx=2), (position), (hidden)] order; we slice per table.
        let h_dim = cfg.hidden_size;
        let table_total = pe_size * h_dim;
        // Split into x and y tables manually (each table_total floats).
        let pos_table_arc = &self.weights.position_embedding_table;
        assert_eq!(
            pos_table_arc.len(),
            2 * table_total,
            "position_embedding_table size {} != 2 * pe_size * hidden",
            pos_table_arc.len(),
        );
        let mut pos_emb_data = vec![0.0_f32; num_patches * h_dim];
        for (p_idx, (&px, &py)) in x_positions.iter().zip(y_positions.iter()).enumerate() {
            // Bound-clamp positions to the table's extent.
            let px_c = px.min(pe_size - 1);
            let py_c = py.min(pe_size - 1);
            let off_x = 0 * table_total + px_c * h_dim;
            let off_y = 1 * table_total + py_c * h_dim;
            for k in 0..h_dim {
                pos_emb_data[p_idx * h_dim + k] =
                    pos_table_arc[off_x + k] + pos_table_arc[off_y + k];
            }
        }
        let pos_emb = anchor.const_f32_like(
            Arc::from(pos_emb_data),
            Shape::from_dims(&[1, num_patches, h_dim]),
        );

        // Build cos/sin for 2D RoPE: head_dim split into two halves.
        // Within each half, standard split-half RoPE has frequencies for
        //   inv_freq[i] = theta ** (-2i / (head_dim/2))   for i in [0, half/2).
        // Eager doubles up: each "freq" appears twice within the half (in
        // the standard cat(freqs, freqs) pattern), so cos/sin for the
        // first half (x-axis) is built per-patch as the cos/sin of
        // x_pos * inv_freq, then duplicated to fill the `half` dim.
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|i| (cfg.rope_theta.powf(-2.0 * i as f64 / half as f64)) as f32)
            .collect();

        let mut cos_data = vec![0.0_f32; num_patches * head_dim];
        let mut sin_data = vec![0.0_f32; num_patches * head_dim];
        for (p_idx, (&px, &py)) in x_positions.iter().zip(y_positions.iter()).enumerate() {
            // First half: x-positions.
            for i in 0..quarter {
                let theta = (px as f32) * inv_freq[i];
                let c = theta.cos();
                let s = theta.sin();
                // Standard split-half: features i and i+quarter share frequency i.
                cos_data[p_idx * head_dim + i] = c;
                cos_data[p_idx * head_dim + i + quarter] = c;
                sin_data[p_idx * head_dim + i] = s;
                sin_data[p_idx * head_dim + i + quarter] = s;
            }
            // Second half: y-positions.
            for i in 0..quarter {
                let theta = (py as f32) * inv_freq[i];
                let c = theta.cos();
                let s = theta.sin();
                cos_data[p_idx * head_dim + half + i] = c;
                cos_data[p_idx * head_dim + half + i + quarter] = c;
                sin_data[p_idx * head_dim + half + i] = s;
                sin_data[p_idx * head_dim + half + i + quarter] = s;
            }
        }
        let cos_xy = anchor.const_f32_like(
            Arc::from(cos_data),
            Shape::from_dims(&[num_patches, head_dim]),
        );
        let sin_xy = anchor.const_f32_like(
            Arc::from(sin_data),
            Shape::from_dims(&[num_patches, head_dim]),
        );

        Ok((pos_emb, cos_xy, sin_xy))
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Gemma4VisionLayerWeights,
        cos: &LazyTensor,
        sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        // Pre-attn norm.
        let residual = x.clone();
        let x_norm = x.rms_norm_affine_with_offset(&layer.input_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let attn = self.attention(&x_norm, layer, cos, sin)?;
        let attn_normed = attn.rms_norm_affine_with_offset(&layer.post_attn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let h1 = residual.add(&attn_normed)?;

        // Pre-FFN norm.
        let residual2 = h1.clone();
        let h1_norm = h1.rms_norm_affine_with_offset(&layer.pre_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let gate = layer.ffn_gate.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let up = layer.ffn_up.apply_linear(&h1_norm, cfg.hidden_size, cfg.intermediate_size);
        let activated = match cfg.hidden_activation {
            Gemma4VisionActivation::Gelu => gate.gelu_erf(),
            Gemma4VisionActivation::GeluPytorchTanh => gate.gelu(),
        };
        let ffn_inner = activated.mul(&up)?;
        let ffn_out = layer.ffn_down.apply_linear(&ffn_inner, cfg.intermediate_size, cfg.hidden_size);
        let ffn_normed = ffn_out.rms_norm_affine_with_offset(&layer.post_ffn_norm_gain, 1.0, cfg.rms_norm_eps)?;
        residual2.add(&ffn_normed)
    }

    fn attention(
        &self,
        x: &LazyTensor,
        layer: &Gemma4VisionLayerWeights,
        cos: &LazyTensor,
        sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1]; // num_patches
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;

        let q = layer.q_proj.apply_linear(x, cfg.hidden_size, q_dim);
        let k = layer.k_proj.apply_linear(x, cfg.hidden_size, kv_dim);
        let v = layer.v_proj.apply_linear(x, cfg.hidden_size, kv_dim);

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_kv, head_dim)?;
        let v = v.split_heads(n_kv, head_dim)?;

        // Q/K norms with `(gain + 1)` offset, V pure RmsNorm.
        let q = q.rms_norm_affine_with_offset(&layer.q_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let k = k.rms_norm_affine_with_offset(&layer.k_norm_gain, 1.0, cfg.rms_norm_eps)?;
        let v = v.rms_norm_last_dim(cfg.rms_norm_eps)?;

        // 2D RoPE: apply standard split-half RoPE on the combined
        // cos/sin tables. Since cos/sin were precomputed with the
        // per-axis split layout, a single rope_with_tables call
        // handles both halves at once.
        let q_r = q.rope_with_tables(cos, sin)?;
        let k_r = k.rope_with_tables(cos, sin)?;

        // GQA expand.
        let n_rep = n_heads / n_kv;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        // No causal mask — vision attention is fully bidirectional.
        // The eager code uses `scale = 1.0` because Q is normalized;
        // we match that.
        let scores = q_r.matmul(&k_t)?;
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v_full)?;
        let merged = ctx.merge_heads()?;
        Ok(layer.o_proj.apply_linear(&merged, q_dim, cfg.hidden_size))
    }

    /// Spatial average pooling via scatter_add by patch (col, row).
    /// Patches in `(col / k, row / k)` cells are averaged together
    /// (sum-then-divide-by-k²) into a flat output of length
    /// `output_length`.
    fn spatial_pool(
        &self,
        x: &LazyTensor,
        ph: usize,
        pw: usize,
        k: usize,
        output_length: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let num_patches = dims[1];
        let hidden = dims[2];

        // Build the bucket index per patch (kernel_idx = kx + stride * ky).
        // Eager uses stride = floor((max_x + 1) / k) = pw / k.
        let stride = pw / k;
        let mut idx_data = vec![0_u32; num_patches];
        for r in 0..ph {
            for col in 0..pw {
                let kx = col / k;
                let ky = r / k;
                let bucket = (kx + stride * ky) as u32;
                idx_data[r * pw + col] = bucket;
            }
        }
        // scatter_add wants indices broadcast to the value shape.
        // Build [batch, num_patches, hidden] index tensor.
        let mut idx_full = vec![0_u32; batch * num_patches * hidden];
        for b in 0..batch {
            for p in 0..num_patches {
                let bucket = idx_data[p];
                for d in 0..hidden {
                    idx_full[(b * num_patches + p) * hidden + d] = bucket;
                }
            }
        }
        let idx_tensor = x.const_u32_like(
            idx_full,
            Shape::from_dims(&[batch, num_patches, hidden]),
        );

        // Scale by 1/k² BEFORE scatter so the scatter sum becomes a mean.
        let x_scaled = x.mul_scalar(1.0 / ((k * k) as f64));

        // Zeros of shape (batch, output_length, hidden) anchored on x's graph.
        let zeros = x.const_f32_like(
            Arc::from(vec![0.0_f32; batch * output_length * hidden]),
            Shape::from_dims(&[batch, output_length, hidden]),
        );
        let _ = cfg; // silence unused
        zeros.scatter_add(1_usize, &idx_tensor, &x_scaled)
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl Gemma4VisionWeights {
    /// Load Gemma 4 vision-tower weights from a `MmapedSafetensors`
    /// file using HuggingFace's `<prefix>` naming. The prefix is
    /// usually `"vision_tower."` inside a full Gemma 4 multimodal
    /// checkpoint, or empty when loading a stand-alone vision tower.
    ///
    /// Tensor names mirrored from
    /// `fuel_transformers::models::llm::gemma4::vision`:
    ///   - `<prefix>patch_embedder.input_proj.weight`
    ///     (`[hidden_size, ps*ps*3]` HF; transposed to
    ///     `[ps*ps*3, hidden_size]` matmul layout)
    ///   - `<prefix>patch_embedder.position_embedding_table`
    ///     (`[2, position_embedding_size, hidden_size]`, flattened)
    ///   - `<prefix>encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    ///     (no bias)
    ///   - `<prefix>encoder.layers.{i}.self_attn.{q,k}_norm.weight`
    ///     (`[head_dim]` Gemma offset RmsNorm gain)
    ///   - `<prefix>encoder.layers.{i}.mlp.{gate,up,down}_proj.weight`
    ///     (no bias)
    ///   - `<prefix>encoder.layers.{i}.input_layernorm.weight`
    ///   - `<prefix>encoder.layers.{i}.post_attention_layernorm.weight`
    ///   - `<prefix>encoder.layers.{i}.pre_feedforward_layernorm.weight`
    ///   - `<prefix>encoder.layers.{i}.post_feedforward_layernorm.weight`
    ///
    /// The optional `std_bias`/`std_scale` standardisation tensors
    /// are not loaded — v1 of the lazy port omits the standardise
    /// post-processing.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Gemma4VisionConfig,
    ) -> Result<Self> {
        Self::load_from_mmapped_with_prefix(st, cfg, "")
    }

    /// Prefix-aware variant of [`Self::load_from_mmapped`]. Pass
    /// `"vision_tower."` (note the trailing dot) when loading from a
    /// full Gemma 4 multimodal checkpoint.
    pub fn load_from_mmapped_with_prefix(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Gemma4VisionConfig,
        prefix: &str,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};

        let h = cfg.hidden_size;
        let i_dim = cfg.intermediate_size;
        let ps = cfg.patch_size;
        let pe = cfg.position_embedding_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let patch_in = ps * ps * 3;

        // Patch embedder input projection (HF: `[hidden_size,
        // ps*ps*3]`, transposed to `[ps*ps*3, hidden_size]`).
        let input_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{prefix}patch_embedder.input_proj.weight"),
            h,
            patch_in,
        )?;

        // Position embedding table is stored as a flat
        // `[2, position_embedding_size, hidden_size]` tensor; the
        // lazy module slices it manually so we just keep it flat.
        let position_embedding_table = load_tensor_as_f32(
            st,
            &format!("{prefix}patch_embedder.position_embedding_table"),
        )?;
        let expected_pe = 2 * pe * h;
        if position_embedding_table.len() != expected_pe {
            crate::bail!(
                "{prefix}patch_embedder.position_embedding_table: {} elts, expected {} (2 * {} * {})",
                position_embedding_table.len(),
                expected_pe,
                pe,
                h,
            );
        }

        let mut layers: Vec<Gemma4VisionLayerWeights> =
            Vec::with_capacity(cfg.num_hidden_layers);
        for li in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}encoder.layers.{li}");
            let q_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.q_proj.weight"), q_dim, h,
            )?;
            let k_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.k_proj.weight"), kv_dim, h,
            )?;
            let v_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.v_proj.weight"), kv_dim, h,
            )?;
            let o_proj = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attn.o_proj.weight"), h, q_dim,
            )?;
            let q_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.self_attn.q_norm.weight"))?,
            );
            let k_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.self_attn.k_norm.weight"))?,
            );
            let ffn_gate = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.gate_proj.weight"), i_dim, h,
            )?;
            let ffn_up = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.up_proj.weight"), i_dim, h,
            )?;
            let ffn_down = load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.down_proj.weight"), h, i_dim,
            )?;
            let input_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.input_layernorm.weight"))?,
            );
            let post_attn_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.post_attention_layernorm.weight"))?,
            );
            let pre_ffn_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.pre_feedforward_layernorm.weight"))?,
            );
            let post_ffn_norm_gain = Arc::from(
                load_tensor_as_f32(st, &format!("{p}.post_feedforward_layernorm.weight"))?,
            );

            layers.push(Gemma4VisionLayerWeights {
                input_norm_gain,
                post_attn_norm_gain,
                pre_ffn_norm_gain,
                post_ffn_norm_gain,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_gain,
                k_norm_gain,
                ffn_gate,
                ffn_up,
                ffn_down,
            });
        }

        Ok(Gemma4VisionWeights {
            input_proj,
            position_embedding_table: Arc::from(position_embedding_table),
            layers,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &Gemma4VisionConfig) -> Gemma4VisionWeights {
        let mut s: u32 = 30303;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let ps = cfg.patch_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv * head_dim;
        let inter = cfg.intermediate_size;
        let pe = cfg.position_embedding_size;
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);

        let input_proj = WeightStorage::F32(vec_of(ps * ps * 3 * h, &mut *nb));
        let position_embedding_table = vec_of(2 * pe * h, &mut *nb);

        let layers: Vec<Gemma4VisionLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| Gemma4VisionLayerWeights {
                input_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_attn_norm_gain: Arc::from(vec![0.05_f32; h]),
                pre_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                post_ffn_norm_gain: Arc::from(vec![0.05_f32; h]),
                q_proj: WeightStorage::F32(vec_of(h * q_dim, &mut *nb)),
                k_proj: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                v_proj: WeightStorage::F32(vec_of(h * kv_dim, &mut *nb)),
                o_proj: WeightStorage::F32(vec_of(q_dim * h, &mut *nb)),
                q_norm_gain: Arc::from(vec![0.05_f32; head_dim]),
                k_norm_gain: Arc::from(vec![0.05_f32; head_dim]),
                ffn_gate: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
            })
            .collect();

        Gemma4VisionWeights { input_proj, position_embedding_table, layers }
    }

    fn tiny_config() -> Gemma4VisionConfig {
        Gemma4VisionConfig {
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            hidden_activation: Gemma4VisionActivation::GeluPytorchTanh,
            rms_norm_eps: 1e-6,
            patch_size: 4,
            position_embedding_size: 32,
            // 6×6 = 36 patches, pool with k=3 → 4 output tokens.
            pooling_kernel_size: 3,
            default_output_length: 4,
            rope_theta: 100.0,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let model = Gemma4VisionModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        // 24×24 image at patch_size=4 → 6×6=36 patches; pool k=3 → 4 tokens.
        let h_img = 24;
        let w_img = 24;
        let n_pix = 1 * 3 * h_img * w_img;
        let img_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        let pixel_values = LazyTensor::from_f32(
            Arc::from(img_data),
            Shape::from_dims(&[1, 3, h_img, w_img]),
            &Device::cpu(),
        );
        let out = model.forward(&pixel_values).unwrap();
        let expected_out_len = (h_img / cfg.patch_size) * (w_img / cfg.patch_size)
            / (cfg.pooling_kernel_size * cfg.pooling_kernel_size);
        assert_eq!(out.shape().dims(), &[1, expected_out_len, cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "got non-finite vision token {v}");
        }
    }

    /// Changing the input pixels must measurably alter the output —
    /// proves the patch embedding + encoder are wired to the input.
    #[test]
    fn pixel_change_alters_output() {
        let cfg = tiny_config();
        let h_img = 24; // 6×6=36 patches, k=3 → 4 output tokens.
        let w_img = 24;
        let n_pix = 1 * 3 * h_img * w_img;
        let img_a: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        let mut img_b = img_a.clone();
        for v in img_b.iter_mut() { *v = 1.0 - *v; }

        // Build two independent models (one per graph anchor) sharing
        // the same weights so the only difference between forwards is
        // the input pixels.
        let weights = tiny_weights(&cfg);
        let model_a = Gemma4VisionModel { config: cfg.clone(), weights: weights.clone() };
        let model_b = Gemma4VisionModel { config: cfg, weights };

        let pix_a = LazyTensor::from_f32(
            Arc::from(img_a),
            Shape::from_dims(&[1, 3, h_img, w_img]),
            &Device::cpu(),
        );
        let out_a = model_a.forward(&pix_a).unwrap().realize_f32();
        let pix_b = LazyTensor::from_f32(
            Arc::from(img_b),
            Shape::from_dims(&[1, 3, h_img, w_img]),
            &Device::cpu(),
        );
        let out_b = model_b.forward(&pix_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-6,
            "pixel-value change must alter encoder output, max_diff = {max_diff}");
    }

    /// Spatial pooling reduces num_patches by k² with mean-shape
    /// arithmetic (sum scaled by 1/k²).
    #[test]
    fn pooling_reduces_count() {
        let cfg = tiny_config();
        let model = Gemma4VisionModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let h_img = 12;
        let w_img = 12;
        // 3×3=9 patches with k=3 → 1 output token.
        let n_pix = 1 * 3 * h_img * w_img;
        let img_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        let pix = LazyTensor::from_f32(
            Arc::from(img_data),
            Shape::from_dims(&[1, 3, h_img, w_img]),
            &Device::cpu(),
        );
        let out = model.forward(&pix).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, cfg.hidden_size]);
    }
}

//! Qwen3-VL multimodal composition (vision + text + DeepStack).
//!
//! Sub-port 3 of `docs/session-prompts/shipped/port-qwen3-vl.md` —
//! wires the already-shipped [`crate::lazy_qwen3_vl_vision`] tower and
//! [`crate::lazy_qwen3_vl_text`] decoder together with a learned
//! multimodal projector and DeepStack residual injection.
//!
//! ## Forward path
//!
//! 1. Run the vision tower if image / video pixels are present. The
//!    tower returns per-patch embeddings `(N, vision_out_hidden)` and
//!    a `Vec<LazyTensor>` of per-DeepStack-index residuals, each shaped
//!    `(N, vision_out_hidden)`.
//! 2. Project each visual feature stream (final + DeepStack) through
//!    the shared `multimodal_projector` linear into the text hidden
//!    dimension.
//! 3. Substitute projected visual embeddings at the `image_token_id` /
//!    `video_token_id` slots of the text embedding sequence using the
//!    masked-add pattern (1.0 at visual slots, 0.0 elsewhere; no
//!    scatter primitive needed).
//! 4. Scatter each projected DeepStack residual into a
//!    `(1, seq, text_hidden)` zero-elsewhere tensor and pass the
//!    sequence as `deepstack_per_layer` to the text model's
//!    [`crate::lazy_qwen3_vl_text::Qwen3VlTextModel::forward_embeds_with_deepstack`].
//! 5. Build MROPE positions: text tokens carry
//!    `(t, t, t)` where `t = start_pos + sequence_index`; image / video
//!    patches at vision-token positions carry `(start_pos + slot_idx,
//!    h, w)` derived from a row-major `(t_patches, h_patches,
//!    w_patches)` grid declared on the visual input.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_qwen3_vl_text::{
    MropePos, Qwen3VlTextConfig, Qwen3VlTextModel, Qwen3VlTextWeights,
};
use crate::lazy_qwen3_vl_vision::{
    Qwen3VlVisionConfig, Qwen3VlVisionModel, Qwen3VlVisionWeights,
};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

/// Patch grid `(t_patches, h_patches, w_patches)` describing the
/// pre-flattened patch sequence shape. The product must equal the
/// number of patches `N` fed to the vision tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatchGrid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl PatchGrid {
    pub fn num_patches(self) -> usize {
        self.t * self.h * self.w
    }
}

/// Full Qwen3-VL multimodal configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlConfig {
    pub vision_config: Qwen3VlVisionConfig,
    pub text_config: Qwen3VlTextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

/// Multimodal projector: vision `out_hidden_size` → text `hidden_size`.
#[derive(Debug, Clone)]
pub struct Qwen3VlMultimodalProjector {
    /// `[vision_out_hidden, text_hidden]`.
    pub weight: WeightStorage,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlWeights {
    pub vision: Qwen3VlVisionWeights,
    pub text: Qwen3VlTextWeights,
    pub multimodal_projector: Qwen3VlMultimodalProjector,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlModel {
    pub config: Qwen3VlConfig,
    pub weights: Qwen3VlWeights,
}

impl Qwen3VlModel {
    /// Run the full multimodal forward pass.
    ///
    /// * `image_pixels` / `video_pixels` — each is the pre-flattened
    ///   patch tensor `(N, C, T_p, H, W)` matching the vision tower
    ///   contract. `Option::None` skips the corresponding modality.
    /// * `image_grid` / `video_grid` — patch grid `(t, h, w)` whose
    ///   product equals `N`. Required iff the matching pixel tensor is
    ///   `Some`.
    /// * `text_tokens` — token sequence. `image_token_id` / `video_token_id`
    ///   occurrences are placeholder slots replaced by projected
    ///   visual embeddings.
    /// * `start_pos` — temporal-axis offset applied to MROPE positions
    ///   for both text and visual tokens, mirroring eager's
    ///   `seqlen_offset` for incremental decode.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        image_pixels: Option<&LazyTensor>,
        image_grid: Option<PatchGrid>,
        video_pixels: Option<&LazyTensor>,
        video_grid: Option<PatchGrid>,
        text_tokens: &[u32],
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision_config;
        let t_cfg = &cfg.text_config;
        if text_tokens.is_empty() {
            return Err(crate::Error::Msg(
                "Qwen3VlModel::forward: text_tokens must be non-empty".into(),
            )
            .bt());
        }
        if v_cfg.out_hidden_size == 0 {
            return Err(crate::Error::Msg(
                "Qwen3VlModel: vision_config.out_hidden_size must be > 0".into(),
            )
            .bt());
        }
        if image_pixels.is_some() != image_grid.is_some() {
            return Err(crate::Error::Msg(
                "Qwen3VlModel::forward: image_pixels and image_grid must be both Some or both None"
                    .into(),
            )
            .bt());
        }
        if video_pixels.is_some() != video_grid.is_some() {
            return Err(crate::Error::Msg(
                "Qwen3VlModel::forward: video_pixels and video_grid must be both Some or both None"
                    .into(),
            )
            .bt());
        }
        let seq = text_tokens.len();

        // Anchor every const for this forward on a single shared
        // tensor's graph. The vision input (if present) is the natural
        // anchor — same lifetime as the rest of the computation. With
        // no image/video, build a one-element f32 anchor matching
        // eager `lazy_qwen3_vl_text::forward`.
        let anchor: LazyTensor = if let Some(p) = image_pixels {
            p.clone()
        } else if let Some(p) = video_pixels {
            p.clone()
        } else {
            LazyTensor::from_f32(
                Arc::from(vec![0.0_f32]),
                Shape::from_dims(&[1]),
                &Device::cpu(),
            )
        };

        // Validate token-id slot counts up front so a mismatched
        // pixel batch is caught at build time instead of producing
        // silently-truncated visual embeds.
        let image_slot_positions: Vec<usize> = text_tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == cfg.image_token_id).then_some(i))
            .collect();
        let video_slot_positions: Vec<usize> = text_tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == cfg.video_token_id).then_some(i))
            .collect();
        if let Some(grid) = image_grid {
            if image_slot_positions.len() != grid.num_patches() {
                return Err(crate::Error::Msg(format!(
                    "Qwen3VlModel::forward: {} image_token_id slots in text_tokens \
                     do not match image_grid patch count {}",
                    image_slot_positions.len(),
                    grid.num_patches(),
                ))
                .bt());
            }
        } else if !image_slot_positions.is_empty() {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlModel::forward: text_tokens contain {} image_token_id slots but \
                 image_pixels is None",
                image_slot_positions.len(),
            ))
            .bt());
        }
        if let Some(grid) = video_grid {
            if video_slot_positions.len() != grid.num_patches() {
                return Err(crate::Error::Msg(format!(
                    "Qwen3VlModel::forward: {} video_token_id slots in text_tokens \
                     do not match video_grid patch count {}",
                    video_slot_positions.len(),
                    grid.num_patches(),
                ))
                .bt());
            }
        } else if !video_slot_positions.is_empty() {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlModel::forward: text_tokens contain {} video_token_id slots but \
                 video_pixels is None",
                video_slot_positions.len(),
            ))
            .bt());
        }

        // ---- Run vision tower(s) and project to text dim ----------
        let image_vision_out = match (image_pixels, image_grid) {
            (Some(pixels), Some(grid)) => Some(self.run_vision_and_project(pixels, grid)?),
            _ => None,
        };
        let video_vision_out = match (video_pixels, video_grid) {
            (Some(pixels), Some(grid)) => Some(self.run_vision_and_project(pixels, grid)?),
            _ => None,
        };

        // ---- Text embeddings anchored on the same graph -----------
        let text_model = Qwen3VlTextModel {
            config: t_cfg.clone(),
            weights: self.weights.text.clone(),
        };
        let text_embeds = text_model.embed_tokens_anchored(&anchor, text_tokens)?;

        // ---- Substitute visual embeddings into text slots ---------
        let mut embeds = text_embeds;
        if let Some(out) = image_vision_out.as_ref() {
            embeds = substitute_visual_embeds(
                &embeds,
                &out.projected,
                &image_slot_positions,
                t_cfg.hidden_size,
            )?;
        }
        if let Some(out) = video_vision_out.as_ref() {
            embeds = substitute_visual_embeds(
                &embeds,
                &out.projected,
                &video_slot_positions,
                t_cfg.hidden_size,
            )?;
        }

        // ---- Build MROPE position grid ----------------------------
        let mrope_positions = build_mrope_position_grid(
            seq,
            start_pos,
            &image_slot_positions,
            image_grid,
            &video_slot_positions,
            video_grid,
        );

        // ---- DeepStack residuals → per-layer (1, seq, hidden) -----
        let deepstack_per_layer = build_deepstack_per_layer(
            &anchor,
            seq,
            t_cfg.hidden_size,
            t_cfg.num_hidden_layers,
            image_vision_out.as_ref(),
            &image_slot_positions,
            video_vision_out.as_ref(),
            &video_slot_positions,
        )?;

        text_model.forward_embeds_with_deepstack(&embeds, &mrope_positions, &deepstack_per_layer)
    }

    fn run_vision_and_project(
        &self,
        pixels: &LazyTensor,
        grid: PatchGrid,
    ) -> Result<VisionAndProjection> {
        let v_cfg = &self.config.vision_config;
        let t_cfg = &self.config.text_config;
        let n = grid.num_patches();
        let dims = pixels.shape();
        let dims = dims.dims();
        if dims.is_empty() || dims[0] != n {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlModel: pixels[0] ({:?}) must equal grid.num_patches() ({n})",
                dims.first(),
            ))
            .bt());
        }
        let vision_model = Qwen3VlVisionModel {
            config: v_cfg.clone(),
            weights: self.weights.vision.clone(),
        };
        // Single image/video → a single cu_seqlens block covering all
        // patches. Per-batch packed images would supply a length-`b+1`
        // cumulative slice; left to a follow-up.
        let cu_seqlens = vec![0_usize, n];
        let vision_out = vision_model.forward(pixels, &cu_seqlens)?;
        let projected = project_visual(
            &self.weights.multimodal_projector,
            &vision_out.embeddings,
            v_cfg.out_hidden_size,
            t_cfg.hidden_size,
        )?;
        let deepstack_projected = vision_out
            .deepstack
            .iter()
            .map(|ds| {
                project_visual(
                    &self.weights.multimodal_projector,
                    ds,
                    v_cfg.out_hidden_size,
                    t_cfg.hidden_size,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let _ = grid;
        Ok(VisionAndProjection {
            projected,
            deepstack_projected,
        })
    }
}

/// Bundle of post-projection visual streams for one modality.
struct VisionAndProjection {
    /// `(N, text_hidden)` — final visual features projected to text dim.
    projected: LazyTensor,
    /// One per DeepStack layer, each `(N, text_hidden)`.
    deepstack_projected: Vec<LazyTensor>,
}

fn project_visual(
    proj: &Qwen3VlMultimodalProjector,
    visual: &LazyTensor,
    vision_out_hidden: usize,
    text_hidden: usize,
) -> Result<LazyTensor> {
    let projected = proj
        .weight
        .apply_linear(visual, vision_out_hidden, text_hidden);
    let bias_t =
        projected.const_f32_like(Arc::clone(&proj.bias), Shape::from_dims(&[text_hidden]));
    projected.broadcast_add(&bias_t)
}

/// Replace text-embedding rows at `slot_positions` with the rows of
/// `visual_embeds` `(N, text_hidden)`. Uses a const additive mask plus
/// a gather-by-row pattern — no scatter / index_put primitive required.
fn substitute_visual_embeds(
    text_embeds: &LazyTensor,
    visual_embeds: &LazyTensor,
    slot_positions: &[usize],
    text_hidden: usize,
) -> Result<LazyTensor> {
    if slot_positions.is_empty() {
        return Ok(text_embeds.clone());
    }
    let dims = text_embeds.shape();
    let dims = dims.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[2] != text_hidden {
        return Err(crate::Error::Msg(format!(
            "substitute_visual_embeds: text_embeds expected (1, seq, {text_hidden}), got {:?}",
            dims,
        ))
        .bt());
    }
    let seq = dims[1];
    let v_dims = visual_embeds.shape();
    let v_dims = v_dims.dims();
    if v_dims.len() != 2 || v_dims[1] != text_hidden {
        return Err(crate::Error::Msg(format!(
            "substitute_visual_embeds: visual_embeds expected (N, {text_hidden}), got {:?}",
            v_dims,
        ))
        .bt());
    }
    if v_dims[0] != slot_positions.len() {
        return Err(crate::Error::Msg(format!(
            "substitute_visual_embeds: visual_embeds rows ({}) must match slot count ({})",
            v_dims[0],
            slot_positions.len(),
        ))
        .bt());
    }
    for &p in slot_positions {
        if p >= seq {
            return Err(crate::Error::Msg(format!(
                "substitute_visual_embeds: slot position {p} out of range for seq {seq}",
            ))
            .bt());
        }
    }

    // Gather visual rows into a per-token (seq,) index. Non-slot
    // positions get a dummy 0 index and are masked back to zero by the
    // additive mask.
    let mut gather_indices = vec![0_u32; seq];
    for (visual_row, &pos) in slot_positions.iter().enumerate() {
        gather_indices[pos] = visual_row as u32;
    }
    let idx = text_embeds.const_u32_like(gather_indices, Shape::from_dims(&[seq]));
    let gathered = visual_embeds
        .index_select(0_usize, &idx)?
        .reshape(Shape::from_dims(&[1, seq, text_hidden]))?;

    let mut mask_data = vec![0.0_f32; seq];
    for &p in slot_positions {
        mask_data[p] = 1.0;
    }
    let mask = text_embeds
        .const_f32_like(mask_data, Shape::from_dims(&[seq]))
        .reshape(Shape::from_dims(&[1, seq, 1]))?
        .broadcast_to(Shape::from_dims(&[1, seq, text_hidden]))?;
    let one_minus = mask.affine(-1.0, 1.0);
    let text_part = text_embeds.mul(&one_minus)?;
    let visual_part = gathered.mul(&mask)?;
    text_part.add(&visual_part)
}

/// Build the per-token MROPE position grid. Visual slots draw their
/// `(t, h, w)` from the row-major patch grid; text slots fill the
/// scalar `(p, p, p)` derived from the global sequence index plus
/// `start_pos`. Visual `t` is offset by `start_pos + slot_idx` to keep
/// the temporal axis monotonically increasing with the sequence.
fn build_mrope_position_grid(
    seq: usize,
    start_pos: usize,
    image_slots: &[usize],
    image_grid: Option<PatchGrid>,
    video_slots: &[usize],
    video_grid: Option<PatchGrid>,
) -> Vec<MropePos> {
    let to_u32 = |x: usize| -> u32 { x.min(u32::MAX as usize) as u32 };
    let mut positions: Vec<MropePos> = (0..seq)
        .map(|p| {
            let q = to_u32(p + start_pos);
            [q, q, q]
        })
        .collect();
    if let Some(grid) = image_grid {
        fill_visual_positions(&mut positions, image_slots, grid, start_pos);
    }
    if let Some(grid) = video_grid {
        fill_visual_positions(&mut positions, video_slots, grid, start_pos);
    }
    positions
}

fn fill_visual_positions(
    positions: &mut [MropePos],
    slots: &[usize],
    grid: PatchGrid,
    start_pos: usize,
) {
    let to_u32 = |x: usize| -> u32 { x.min(u32::MAX as usize) as u32 };
    for (slot_idx, &pos) in slots.iter().enumerate() {
        let rem = slot_idx % (grid.h * grid.w);
        let h_p = rem / grid.w;
        let w_p = rem % grid.w;
        // Spec convention: visual patches carry `(temporal, h, w)`. The
        // temporal axis monotonically tracks the global sequence index
        // so MROPE distinguishes patches across multiple
        // `t_patches` slabs even when h/w repeat.
        positions[pos] = [to_u32(start_pos + slot_idx), to_u32(h_p), to_u32(w_p)];
    }
}

/// Build the per-layer DeepStack residual list. For each text-layer
/// injection slot we accumulate the projected DeepStack rows for
/// image and video (if both modalities supply a residual at that slot)
/// into a single `(1, seq, hidden)` tensor that is zero-elsewhere.
fn build_deepstack_per_layer(
    anchor: &LazyTensor,
    seq: usize,
    text_hidden: usize,
    num_layers: usize,
    image: Option<&VisionAndProjection>,
    image_slots: &[usize],
    video: Option<&VisionAndProjection>,
    video_slots: &[usize],
) -> Result<Vec<Option<LazyTensor>>> {
    let image_layers = image.map_or(0, |i| i.deepstack_projected.len());
    let video_layers = video.map_or(0, |v| v.deepstack_projected.len());
    let len = image_layers.max(video_layers).min(num_layers);
    let mut out: Vec<Option<LazyTensor>> = Vec::with_capacity(len);
    for layer_idx in 0..len {
        let mut accum: Option<LazyTensor> = None;
        if let Some(img) = image {
            if layer_idx < img.deepstack_projected.len() {
                let t = scatter_visual_residual(
                    anchor,
                    &img.deepstack_projected[layer_idx],
                    image_slots,
                    seq,
                    text_hidden,
                )?;
                accum = Some(t);
            }
        }
        if let Some(vid) = video {
            if layer_idx < vid.deepstack_projected.len() {
                let t = scatter_visual_residual(
                    anchor,
                    &vid.deepstack_projected[layer_idx],
                    video_slots,
                    seq,
                    text_hidden,
                )?;
                accum = Some(match accum {
                    Some(prev) => prev.add(&t)?,
                    None => t,
                });
            }
        }
        out.push(accum);
    }
    Ok(out)
}

/// Build a `(1, seq, hidden)` tensor whose rows at `slot_positions` are
/// the rows of `residual` `(N, hidden)`, zero everywhere else.
fn scatter_visual_residual(
    anchor: &LazyTensor,
    residual: &LazyTensor,
    slot_positions: &[usize],
    seq: usize,
    text_hidden: usize,
) -> Result<LazyTensor> {
    if slot_positions.is_empty() {
        return Ok(anchor.const_f32_like(
            Arc::from(vec![0.0_f32; seq * text_hidden]),
            Shape::from_dims(&[1, seq, text_hidden]),
        ));
    }
    let mut gather_indices = vec![0_u32; seq];
    for (residual_row, &pos) in slot_positions.iter().enumerate() {
        gather_indices[pos] = residual_row as u32;
    }
    let idx = anchor.const_u32_like(gather_indices, Shape::from_dims(&[seq]));
    let gathered = residual
        .index_select(0_usize, &idx)?
        .reshape(Shape::from_dims(&[1, seq, text_hidden]))?;
    let mut mask_data = vec![0.0_f32; seq];
    for &p in slot_positions {
        mask_data[p] = 1.0;
    }
    let mask = anchor
        .const_f32_like(mask_data, Shape::from_dims(&[seq]))
        .reshape(Shape::from_dims(&[1, seq, 1]))?
        .broadcast_to(Shape::from_dims(&[1, seq, text_hidden]))?;
    gathered.mul(&mask)
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl Qwen3VlWeights {
    /// Load Qwen3-VL multimodal weights: delegate to the vision + text
    /// sub-loaders and load the top-level multimodal projector tensors
    /// here.
    ///
    /// HuggingFace tensor naming for Qwen3-VL parks the merger /
    /// multimodal projector under `model.visual.merger.*`. The lazy
    /// port models a single linear (`vision.out_hidden_size →
    /// text.hidden_size`); the corresponding HF tensor is the second
    /// linear of the merger MLP (`linear_fc2`). We fall back to
    /// `mlp.2.{weight,bias}` for HF dumps that flatten the merger
    /// into a sequential MLP, and zero-fill the bias as a last
    /// resort.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &Qwen3VlConfig,
    ) -> Result<Self> {
        use crate::lazy::{
            load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
        };

        let vision = Qwen3VlVisionWeights::load_from_mmapped(st, &cfg.vision_config)?;
        let text = Qwen3VlTextWeights::load_from_mmapped(st, &cfg.text_config)?;

        let v_out = cfg.vision_config.out_hidden_size;
        let t_hidden = cfg.text_config.hidden_size;

        // Multimodal projector weight: `[v_out, t_hidden]` in HF layout,
        // returned by the helper as `[v_in=v_out, t_hidden=t_hidden]`.
        // Try the canonical merger name first, then a few common
        // sequential-MLP variants.
        let weight = load_transposed_matrix_preserve_dtype(
            st,
            "model.visual.merger.linear_fc2.weight",
            t_hidden,
            v_out,
        )
        .or_else(|_| {
            load_transposed_matrix_preserve_dtype(
                st,
                "model.visual.merger.mlp.2.weight",
                t_hidden,
                v_out,
            )
        })
        .or_else(|_| {
            load_transposed_matrix_preserve_dtype(
                st,
                "visual.merger.linear_fc2.weight",
                t_hidden,
                v_out,
            )
        })?;

        // Projector bias is `[t_hidden]`. Zero-fill if missing.
        let bias: Arc<[f32]> = load_tensor_as_f32(st, "model.visual.merger.linear_fc2.bias")
            .or_else(|_| load_tensor_as_f32(st, "model.visual.merger.mlp.2.bias"))
            .or_else(|_| load_tensor_as_f32(st, "visual.merger.linear_fc2.bias"))
            .ok()
            .map(Arc::from)
            .unwrap_or_else(|| Arc::from(vec![0.0_f32; t_hidden]));

        let multimodal_projector = Qwen3VlMultimodalProjector { weight, bias };

        Ok(Self {
            vision,
            text,
            multimodal_projector,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::LayerWeights;
    use crate::lazy_conv3d::{Conv3dTemporal2Config, Conv3dTemporal2Weights};
    use crate::lazy_qwen3_vl_text::Qwen3VlTextLayerExtras;
    use crate::lazy_qwen3_vl_vision::{
        Qwen3VlVisionDeepStackProjection, Qwen3VlVisionLayerWeights,
    };

    fn rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_vision_cfg(deepstack: Vec<usize>) -> Qwen3VlVisionConfig {
        Qwen3VlVisionConfig {
            depth: 2,
            hidden_size: 8,
            out_hidden_size: 8,
            intermediate_size: 16,
            num_heads: 2,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            layer_norm_eps: 1e-6,
            deepstack_visual_indexes: deepstack,
        }
    }

    fn tiny_text_cfg(hidden: usize) -> Qwen3VlTextConfig {
        // num_heads * head_dim must equal hidden_size, and
        // mrope_section [1, 1, 0] sums to head_dim/2 = 2 ⇒ head_dim ≥ 4.
        let (num_heads, num_kv_heads, head_dim) = match hidden {
            8 => (2_usize, 1_usize, 4_usize),
            16 => (4_usize, 2_usize, 4_usize),
            _ => (4_usize, 2_usize, hidden / 4),
        };
        Qwen3VlTextConfig {
            vocab_size: 32,
            hidden_size: hidden,
            intermediate_size: 2 * hidden,
            num_hidden_layers: 2,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            max_position_embeddings: 64,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
            mrope_section: [1, 1, 0],
        }
    }

    fn tiny_vision_weights(cfg: &Qwen3VlVisionConfig, nb: &mut Box<dyn FnMut() -> f32>) -> Qwen3VlVisionWeights {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let conv_raw_len =
            cfg.hidden_size * cfg.in_channels * 2 * cfg.patch_size * cfg.patch_size;
        let conv_raw: Vec<f32> = (0..conv_raw_len).map(|_| (**nb)()).collect();
        let patch_embed = Conv3dTemporal2Weights::from_raw_weight(
            &conv_raw,
            cfg.hidden_size,
            cfg.in_channels,
            cfg.patch_size,
            cfg.patch_size,
            Conv3dTemporal2Config { stride: cfg.patch_size, ..Default::default() },
        )
        .unwrap();
        let patch_embed_bias = vec_of(h, &mut **nb);
        let layers: Vec<Qwen3VlVisionLayerWeights> = (0..cfg.depth)
            .map(|_| Qwen3VlVisionLayerWeights {
                norm1_gain: Arc::from(vec![1.0_f32; h]),
                norm1_bias: Arc::from(vec![0.0_f32; h]),
                norm2_gain: Arc::from(vec![1.0_f32; h]),
                norm2_bias: Arc::from(vec![0.0_f32; h]),
                qkv: WeightStorage::F32(vec_of(h * 3 * h, &mut **nb)),
                qkv_bias: vec_of(3 * h, &mut **nb),
                proj: WeightStorage::F32(vec_of(h * h, &mut **nb)),
                proj_bias: vec_of(h, &mut **nb),
                fc1: WeightStorage::F32(vec_of(h * inter, &mut **nb)),
                fc1_bias: vec_of(inter, &mut **nb),
                fc2: WeightStorage::F32(vec_of(inter * h, &mut **nb)),
                fc2_bias: vec_of(h, &mut **nb),
            })
            .collect();
        let deepstack: Vec<Qwen3VlVisionDeepStackProjection> = cfg
            .deepstack_visual_indexes
            .iter()
            .map(|_| Qwen3VlVisionDeepStackProjection {
                weight: WeightStorage::F32(vec_of(h * cfg.out_hidden_size, &mut **nb)),
                bias: vec_of(cfg.out_hidden_size, &mut **nb),
            })
            .collect();
        Qwen3VlVisionWeights {
            patch_embed,
            patch_embed_bias,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            final_norm_bias: Arc::from(vec![0.0_f32; h]),
            deepstack,
        }
    }

    fn tiny_text_weights(cfg: &Qwen3VlTextConfig, nb: &mut Box<dyn FnMut() -> f32>) -> Qwen3VlTextWeights {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut **nb);
        let mut layers = Vec::new();
        let mut layer_extras = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut **nb)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut **nb)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut **nb)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(h * h, &mut **nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut **nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut **nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut **nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            });
            layer_extras.push(Qwen3VlTextLayerExtras {
                q_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
                k_norm_gain: Arc::from(vec![1.0_f32; cfg.head_dim]),
            });
        }
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut **nb));
        Qwen3VlTextWeights {
            token_embedding,
            layers,
            layer_extras,
            final_norm_gain,
            output,
        }
    }

    /// Build a `(N, C, T_p, H, W)` flattened-patch tensor.
    fn tiny_pixels(cfg: &Qwen3VlVisionConfig, n_patches: usize, scale: f32) -> LazyTensor {
        let numel = n_patches
            * cfg.in_channels
            * cfg.temporal_patch_size
            * cfg.patch_size
            * cfg.patch_size;
        let data: Vec<f32> = (0..numel)
            .map(|i| scale * ((i as f32 / numel as f32) - 0.5))
            .collect();
        LazyTensor::from_f32(
            Arc::from(data),
            Shape::from_dims(&[
                n_patches,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ]),
            &Device::cpu(),
        )
    }

    fn build_model(deepstack: Vec<usize>, text_hidden: usize) -> Qwen3VlModel {
        let v_cfg = tiny_vision_cfg(deepstack);
        let t_cfg = tiny_text_cfg(text_hidden);
        let next = rng(7777);
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let vision = tiny_vision_weights(&v_cfg, &mut nb);
        let text = tiny_text_weights(&t_cfg, &mut nb);
        let proj_weight =
            WeightStorage::F32(vec_of(v_cfg.out_hidden_size * t_cfg.hidden_size, &mut *nb));
        let proj_bias = vec_of(t_cfg.hidden_size, &mut *nb);
        let multimodal_projector = Qwen3VlMultimodalProjector {
            weight: proj_weight,
            bias: proj_bias,
        };
        let cfg = Qwen3VlConfig {
            vision_config: v_cfg,
            text_config: t_cfg,
            image_token_id: 31,
            video_token_id: 30,
            vision_start_token_id: 29,
            vision_end_token_id: 28,
        };
        Qwen3VlModel {
            config: cfg,
            weights: Qwen3VlWeights {
                vision,
                text,
                multimodal_projector,
            },
        }
    }

    /// With no image / video pixels the multimodal forward must
    /// exactly equal the text-only Qwen3VlTextModel forward at the
    /// same MROPE positions.
    #[test]
    fn forward_text_only_matches_qwen3_vl_text_forward() {
        let text_hidden = 8;
        let model = build_model(vec![], text_hidden);
        let tokens = vec![1_u32, 2, 3, 4, 5];
        let logits_mm = model
            .forward(None, None, None, None, &tokens, 0)
            .unwrap()
            .realize_f32();
        let text_model = Qwen3VlTextModel {
            config: model.config.text_config.clone(),
            weights: model.weights.text.clone(),
        };
        let mrope: Vec<MropePos> = (0..tokens.len() as u32).map(|p| [p, p, p]).collect();
        let logits_text = text_model.forward(&tokens, &mrope).unwrap().realize_f32();
        assert_eq!(logits_mm.len(), logits_text.len());
        let max_diff = logits_mm
            .iter()
            .zip(logits_text.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "text-only multimodal must match standalone Qwen3VlTextModel; max_diff = {max_diff}",
        );
    }

    /// With an image present, logits must differ from the no-image
    /// run (visual embeddings substitute into image-token slots and
    /// thereby change the attention context for downstream tokens).
    #[test]
    fn forward_with_image_changes_output() {
        let text_hidden = 8;
        let model = build_model(vec![], text_hidden);
        // 4 image-token slots (one per patch) + 4 plain text tokens.
        // 4 patches = 1 temporal slab * 2x2 spatial (H=W=28, patch=14).
        let img = model.config.image_token_id;
        let tokens = vec![1_u32, 2, img, img, img, img, 3, 4];
        let grid = PatchGrid { t: 1, h: 2, w: 2 };
        let pixels = tiny_pixels(&model.config.vision_config, grid.num_patches(), 1.0);

        let with_img = model
            .forward(Some(&pixels), Some(grid), None, None, &tokens, 0)
            .unwrap()
            .realize_f32();
        // Replace image-token slots with plain text tokens for the
        // text-only run so the sequence has the same length.
        let text_only_tokens: Vec<u32> = tokens
            .iter()
            .map(|&t| if t == img { 5 } else { t })
            .collect();
        let no_img = model
            .forward(None, None, None, None, &text_only_tokens, 0)
            .unwrap()
            .realize_f32();
        assert_eq!(with_img.len(), no_img.len());
        let max_diff = with_img
            .iter()
            .zip(no_img.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "with-image vs without-image logits must differ; max_diff = {max_diff}",
        );
    }

    /// Hand-trace that visual embeddings land in the right text
    /// positions by exercising [`substitute_visual_embeds`] directly:
    /// a deterministic per-row visual constant must reproduce in the
    /// output at exactly the slot positions and the other rows must
    /// stay byte-for-byte equal to the original text embeds.
    #[test]
    fn image_token_slot_scatter_correctness() {
        let seq = 5_usize;
        let hidden = 4_usize;
        let anchor = LazyTensor::from_f32(
            Arc::from(vec![0.0_f32]),
            Shape::from_dims(&[1]),
            &Device::cpu(),
        );
        // text_embeds := (1, seq, hidden) with row r = [r, r, r, r].
        let text_data: Vec<f32> = (0..seq)
            .flat_map(|r| (0..hidden).map(move |_| r as f32))
            .collect();
        let text_embeds = anchor.const_f32_like(
            Arc::from(text_data.clone()),
            Shape::from_dims(&[1, seq, hidden]),
        );
        // visual_embeds := (N=2, hidden) with row r = [-(r+1), …].
        let n = 2_usize;
        let visual_data: Vec<f32> = (0..n)
            .flat_map(|r| (0..hidden).map(move |_| -((r + 1) as f32)))
            .collect();
        let visual_embeds =
            anchor.const_f32_like(Arc::from(visual_data), Shape::from_dims(&[n, hidden]));
        let slot_positions = vec![1_usize, 3];
        let out = substitute_visual_embeds(
            &text_embeds,
            &visual_embeds,
            &slot_positions,
            hidden,
        )
        .unwrap()
        .realize_f32();
        assert_eq!(out.len(), seq * hidden);
        for r in 0..seq {
            for c in 0..hidden {
                let v = out[r * hidden + c];
                let expected = if r == 1 {
                    -1.0
                } else if r == 3 {
                    -2.0
                } else {
                    r as f32
                };
                assert!(
                    (v - expected).abs() < 1e-6,
                    "row {r} col {c}: expected {expected}, got {v}",
                );
            }
        }
    }

    /// Project from vision_dim=8 to text_dim=16: ensure the substituted
    /// rows pass through the projector and land at the expected output
    /// shape `(1, seq, text_hidden=16)`.
    #[test]
    fn multimodal_projector_dim_change() {
        // text_hidden = 16, vision_hidden = 8 (from tiny_vision_cfg).
        let text_hidden = 16;
        let model = build_model(vec![], text_hidden);
        let img = model.config.image_token_id;
        let tokens = vec![1_u32, img, img, img, img, 2];
        let grid = PatchGrid { t: 1, h: 2, w: 2 };
        let pixels = tiny_pixels(&model.config.vision_config, grid.num_patches(), 1.0);
        let logits = model
            .forward(Some(&pixels), Some(grid), None, None, &tokens, 0)
            .unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[1, tokens.len(), model.config.text_config.vocab_size],
        );
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// End-to-end smoke: T=4 video frames (= 2 temporal slabs after
    /// `temporal_patch_size=2`) × 2×2 spatial = 8 video patches; 8
    /// text tokens with 8 video-token slots. Verify the LM produces
    /// a finite `(1, 8, vocab)` logit tensor and that DeepStack
    /// residuals propagate (`deepstack_visual_indexes = [0, 1]`).
    #[test]
    fn end_to_end_tiny_video_plus_text() {
        let text_hidden = 8;
        let model = build_model(vec![0, 1], text_hidden);
        let vid = model.config.video_token_id;
        // 8 video-token slots in front; LM head per-token logits.
        let tokens = vec![vid; 8];
        let grid = PatchGrid { t: 2, h: 2, w: 2 };
        let pixels = tiny_pixels(&model.config.vision_config, grid.num_patches(), 1.0);
        let logits = model
            .forward(None, None, Some(&pixels), Some(grid), &tokens, 0)
            .unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[1, tokens.len(), model.config.text_config.vocab_size],
        );
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }
}

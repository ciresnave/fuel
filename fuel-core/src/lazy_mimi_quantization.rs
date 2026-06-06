//! Mimi residual vector quantizer — lazy port.
//!
//! Mimi (Moshi's audio codec) uses a **split** RVQ where one
//! semantic codebook is followed by a stack of acoustic codebooks.
//! The semantic and acoustic paths quantize the same input
//! *independently* (not residually); the decoder sums their
//! reconstructions.
//!
//! Stack layout:
//!   - `EuclideanCodebook` — fixed-size table of `(codebook_size,
//!     codebook_dim)` centroids; encode finds nearest by the
//!     matmul-distance trick `argmin(c2 − x·E^T)` (where
//!     `c2 = ½·||E||²`); decode is plain `index_select`.
//!   - `VectorQuantization` — optional in/out linear projections
//!     wrap the codebook so the latent dim and the codebook dim
//!     can differ.
//!   - `ResidualVectorQuantization` — stack of N VQ layers, each
//!     coding the residual of the previous step.
//!   - `ResidualVectorQuantizer` — optional 1×1 `conv1d`
//!     projections on the input and output of the RVQ stack
//!     (preserved here as bias-less Conv1d via the lazy
//!     `conv1d` primitive).
//!   - `SplitResidualVectorQuantizer` — pairs a 1-deep `rvq_first`
//!     (semantic) with a `(n_q - 1)`-deep `rvq_rest` (acoustic).
//!
//! **Encode** input is `(batch, dim, time)`; codes shape
//! `(batch, n_q, time)`. **Decode** is the inverse.
//!
//! Pre-computed weight derivation: the eager port derives the
//! codebook embedding from `embedding_sum / max(cluster_usage,
//! ε)` and `c2 = sum(E·E, -1) / 2.0` at load time. The lazy port
//! takes those already-derived `Arc<[f32]>` tables — caller
//! computes them once in host code from the raw checkpoint.
//!
//! v1 scope: F32, batch == 1, forward-only inference.

use crate::lazy::LazyTensor;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EuclideanCodebookWeights {
    /// `(codebook_size, codebook_dim)` — pre-derived from
    /// `embedding_sum / max(cluster_usage, ε)`.
    pub embedding: Arc<[f32]>,
    /// `(codebook_size,)` — pre-derived as `½·||E||²` along the
    /// codebook-dim axis.
    pub c2: Arc<[f32]>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

#[derive(Debug, Clone)]
pub struct VectorQuantizationWeights {
    pub codebook: EuclideanCodebookWeights,
    /// `(dim, codebook_dim)` if `dim != codebook_dim`, else `None`.
    pub project_in_w: Option<Arc<[f32]>>,
    pub project_in_b: Option<Arc<[f32]>>,
    /// `(codebook_dim, dim)` if `dim != codebook_dim`, else `None`.
    pub project_out_w: Option<Arc<[f32]>>,
    pub project_out_b: Option<Arc<[f32]>>,
    pub dim: usize,
}

#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizationWeights {
    pub layers: Vec<VectorQuantizationWeights>,
}

#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizerWeights {
    pub vq: ResidualVectorQuantizationWeights,
    /// 1×1 `conv1d_no_bias` weights `(dim, input_dim, 1)`.
    pub input_proj_w: Option<Arc<[f32]>>,
    /// 1×1 `conv1d_no_bias` weights `(output_dim, dim, 1)`.
    pub output_proj_w: Option<Arc<[f32]>>,
    pub dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone)]
pub struct SplitResidualVectorQuantizerWeights {
    pub rvq_first: ResidualVectorQuantizerWeights,
    pub rvq_rest: ResidualVectorQuantizerWeights,
    pub n_q: usize,
}

// ---- Forward helpers -------------------------------------------------------

/// `EuclideanCodebook::encode`: takes `(M, codebook_dim)` and returns
/// `(M,)` U32 nearest-codebook indices using the matmul-distance trick.
fn codebook_encode(
    x: &LazyTensor, cb: &EuclideanCodebookWeights,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let m = dims[0];
    debug_assert_eq!(dims[1], cb.codebook_dim);
    let embedding = x.const_f32_like(
        Arc::clone(&cb.embedding),
        Shape::from_dims(&[cb.codebook_size, cb.codebook_dim]),
    );
    let e_t = embedding.permute([1, 0_usize])?;
    let dot_prod = x.matmul(&e_t)?;
    let c2 = x.const_f32_like(
        Arc::clone(&cb.c2), Shape::from_dims(&[cb.codebook_size]),
    );
    let c2_b = c2
        .reshape(Shape::from_dims(&[1, cb.codebook_size]))?
        .broadcast_to(Shape::from_dims(&[m, cb.codebook_size]))?;
    let scores = c2_b.sub(&dot_prod)?;
    scores.argmin_dim(1_usize)
}

/// `EuclideanCodebook::decode`: takes flat `(M,)` U32 indices and
/// returns `(M, codebook_dim)` selected embeddings.
fn codebook_decode(
    codes: &LazyTensor, cb: &EuclideanCodebookWeights,
) -> Result<LazyTensor> {
    let embedding = codes.const_f32_like(
        Arc::clone(&cb.embedding),
        Shape::from_dims(&[cb.codebook_size, cb.codebook_dim]),
    );
    embedding.index_select(0_usize, codes)
}

fn apply_linear_opt(
    x: &LazyTensor,
    w: &Option<Arc<[f32]>>,
    b: &Option<Arc<[f32]>>,
    in_features: usize,
    out_features: usize,
) -> Result<LazyTensor> {
    match w {
        None => Ok(x.clone()),
        Some(w_arc) => {
            // Match the eager `linear(in, out)` weight layout: stored
            // (out, in), applied as `x @ w^T`.
            let w_t = x.const_f32_like(
                Arc::clone(w_arc),
                Shape::from_dims(&[out_features, in_features]),
            ).permute([1, 0_usize])?;
            let y = x.matmul(&w_t)?;
            match b {
                None => Ok(y),
                Some(b_arc) => {
                    let bias = x.const_f32_like(
                        Arc::clone(b_arc), Shape::from_dims(&[out_features]),
                    );
                    y.broadcast_add(&bias)
                }
            }
        }
    }
}

fn apply_conv1d_1x1_opt(
    x: &LazyTensor,
    w: &Option<Arc<[f32]>>,
    in_channels: usize,
    out_channels: usize,
) -> Result<LazyTensor> {
    match w {
        None => Ok(x.clone()),
        Some(w_arc) => {
            let weight = x.const_f32_like(
                Arc::clone(w_arc),
                Shape::from_dims(&[out_channels, in_channels, 1]),
            );
            x.conv1d(&weight, None, 1, 0, 1)
        }
    }
}

/// `VectorQuantization::encode` — input `(B, D, T)` → codes `(B, T)`.
fn vq_encode(
    xs: &LazyTensor, w: &VectorQuantizationWeights,
) -> Result<LazyTensor> {
    let dims = xs.shape();
    let dims = dims.dims();
    let b = dims[0]; let d = dims[1]; let t = dims[2];
    debug_assert_eq!(d, w.dim);
    // (B, D, T) → (B, T, D)
    let xs = xs.permute([0, 2, 1_usize])?;
    let projected = apply_linear_opt(
        &xs, &w.project_in_w, &w.project_in_b,
        w.dim, w.codebook.codebook_dim,
    )?;
    // (B, T, codebook_dim) → (B*T, codebook_dim)
    let flat = projected.reshape(Shape::from_dims(&[b * t, w.codebook.codebook_dim]))?;
    let codes_flat = codebook_encode(&flat, &w.codebook)?;
    codes_flat.reshape(Shape::from_dims(&[b, t]))
}

/// `VectorQuantization::decode` — codes `(B, T)` → `(B, D, T)`.
fn vq_decode(
    codes: &LazyTensor, w: &VectorQuantizationWeights,
) -> Result<LazyTensor> {
    let dims = codes.shape();
    let dims = dims.dims();
    let b = dims[0]; let t = dims[1];
    let codes_flat = codes.reshape(Shape::from_dims(&[b * t]))?;
    let quant_flat = codebook_decode(&codes_flat, &w.codebook)?;
    let quant = quant_flat.reshape(Shape::from_dims(&[b, t, w.codebook.codebook_dim]))?;
    let projected = apply_linear_opt(
        &quant, &w.project_out_w, &w.project_out_b,
        w.codebook.codebook_dim, w.dim,
    )?;
    // (B, T, D) → (B, D, T)
    projected.permute([0, 2, 1_usize])
}

/// `ResidualVectorQuantization::encode` — input `(B, D, T)` →
/// codes `(n_q, B, T)`. The encoder runs each VQ on the residual
/// of the previous step.
fn rvq_encode(
    xs: &LazyTensor, w: &ResidualVectorQuantizationWeights,
) -> Result<LazyTensor> {
    let mut residual = xs.clone();
    let mut codes_per_layer = Vec::with_capacity(w.layers.len());
    for layer in &w.layers {
        let indices = vq_encode(&residual, layer)?;
        let quantized = vq_decode(&indices, layer)?;
        residual = residual.sub(&quantized)?;
        codes_per_layer.push(indices);
    }
    // Stack along a fresh leading axis.
    let mut stacked: Option<LazyTensor> = None;
    for c in codes_per_layer {
        let dims = c.shape().dims().to_vec();
        let mut new_dims = vec![1_usize];
        new_dims.extend(dims);
        let c_unsq = c.reshape(Shape::from_dims(&new_dims))?;
        stacked = Some(match stacked {
            None => c_unsq,
            Some(prev) => prev.concat(&c_unsq, 0_usize)?,
        });
    }
    stacked.ok_or_else(|| crate::Error::Msg("rvq_encode: empty layers".into()))
}

/// `ResidualVectorQuantization::decode` — input codes `(n_q, B, T)`
/// → quantized features `(B, D, T)`. Sums per-layer reconstructions.
fn rvq_decode(
    codes: &LazyTensor, w: &ResidualVectorQuantizationWeights,
) -> Result<LazyTensor> {
    assert!(!w.layers.is_empty(), "rvq_decode: empty layers");
    let dims = codes.shape();
    let dims = dims.dims();
    assert_eq!(dims[0], w.layers.len(),
        "rvq_decode: n_q dim mismatch {} vs {}", dims[0], w.layers.len());
    let mut accum: Option<LazyTensor> = None;
    for (i, layer) in w.layers.iter().enumerate() {
        // Slice (1, B, T) and drop the leading dim.
        let slice = codes
            .narrow(0_usize, i, 1)?
            .reshape(Shape::from_dims(&[dims[1], dims[2]]))?;
        let q = vq_decode(&slice, layer)?;
        accum = Some(match accum {
            None => q,
            Some(prev) => prev.add(&q)?,
        });
    }
    Ok(accum.unwrap())
}

/// `ResidualVectorQuantizer::encode` — input `(B, input_dim, T)` →
/// codes `(B, n_q, T)`.
pub fn rvq_quantizer_encode(
    xs: &LazyTensor, w: &ResidualVectorQuantizerWeights,
) -> Result<LazyTensor> {
    let projected = apply_conv1d_1x1_opt(xs, &w.input_proj_w, w.input_dim, w.dim)?;
    let stacked = rvq_encode(&projected, &w.vq)?;
    // (n_q, B, T) → (B, n_q, T)
    stacked.permute([1, 0, 2_usize])
}

/// `ResidualVectorQuantizer::decode` — input codes `(B, n_q, T)` →
/// reconstructed features `(B, output_dim, T)`.
pub fn rvq_quantizer_decode(
    codes: &LazyTensor, w: &ResidualVectorQuantizerWeights,
) -> Result<LazyTensor> {
    // (B, n_q, T) → (n_q, B, T)
    let codes = codes.permute([1, 0, 2_usize])?;
    let quantized = rvq_decode(&codes, &w.vq)?;
    apply_conv1d_1x1_opt(&quantized, &w.output_proj_w, w.dim, w.output_dim)
}

/// `SplitResidualVectorQuantizer::encode` — encodes the same input
/// through both the semantic (1-deep) and acoustic (n_q-1 deep) RVQs
/// **independently**, concatenating along the n_q axis. Returns
/// codes shape `(B, n_q, T)`.
pub fn split_rvq_encode(
    xs: &LazyTensor, w: &SplitResidualVectorQuantizerWeights,
) -> Result<LazyTensor> {
    let semantic = rvq_quantizer_encode(xs, &w.rvq_first)?;
    if w.n_q > 1 {
        let acoustic = rvq_quantizer_encode(xs, &w.rvq_rest)?;
        semantic.concat(&acoustic, 1_usize)
    } else {
        Ok(semantic)
    }
}

/// `SplitResidualVectorQuantizer::decode` — sums semantic and
/// acoustic reconstructions. Returns `(B, output_dim, T)`.
pub fn split_rvq_decode(
    codes: &LazyTensor, w: &SplitResidualVectorQuantizerWeights,
) -> Result<LazyTensor> {
    let dims = codes.shape();
    let dims = dims.dims();
    let b = dims[0]; let total_nq = dims[1]; let t = dims[2];
    assert_eq!(total_nq, w.n_q, "split_rvq_decode: total n_q mismatch");
    let semantic_codes = codes.narrow(1_usize, 0, 1)?;
    let semantic_q = rvq_quantizer_decode(&semantic_codes, &w.rvq_first)?;
    if w.n_q > 1 {
        let acoustic_codes = codes.narrow(1_usize, 1, w.n_q - 1)?;
        let acoustic_q = rvq_quantizer_decode(&acoustic_codes, &w.rvq_rest)?;
        semantic_q.add(&acoustic_q)
    } else {
        let _ = (b, t);
        Ok(semantic_q)
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

/// Load an `EuclideanCodebook`-equivalent set of derived tables from
/// `embed_sum` and `cluster_usage`. Eager:
///
/// ```text
/// embedding = embed_sum / max(cluster_usage, eps).unsqueeze(1)
/// c2        = sum(embedding · embedding, dim=-1) / 2
/// ```
fn load_euclidean_codebook(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    codebook_size: usize, codebook_dim: usize,
) -> Result<EuclideanCodebookWeights> {
    use crate::lazy::load_tensor_as_f32;
    let cluster_usage = load_tensor_as_f32(
        st, &format!("{prefix}.cluster_usage"),
    )?;
    if cluster_usage.len() != codebook_size {
        crate::bail!(
            "{prefix}.cluster_usage: {} elements, expected {codebook_size}",
            cluster_usage.len(),
        );
    }
    let embed_sum = load_tensor_as_f32(
        st, &format!("{prefix}.embed_sum"),
    )?;
    let expected = codebook_size * codebook_dim;
    if embed_sum.len() != expected {
        crate::bail!(
            "{prefix}.embed_sum: {} elements, expected {expected} ({codebook_size}×{codebook_dim})",
            embed_sum.len(),
        );
    }
    // embedding[i, j] = embed_sum[i, j] / max(cluster_usage[i], eps).
    let eps = 1e-5_f32;
    let mut embedding = vec![0.0_f32; expected];
    let mut c2 = vec![0.0_f32; codebook_size];
    for i in 0..codebook_size {
        let denom = cluster_usage[i].max(eps);
        let mut s = 0.0_f64;
        for j in 0..codebook_dim {
            let e = embed_sum[i * codebook_dim + j] / denom;
            embedding[i * codebook_dim + j] = e;
            s += (e as f64) * (e as f64);
        }
        c2[i] = (s / 2.0) as f32;
    }
    Ok(EuclideanCodebookWeights {
        embedding: Arc::from(embedding),
        c2: Arc::from(c2),
        codebook_size,
        codebook_dim,
    })
}

/// Load one `VectorQuantization` layer. In Mimi the residual stack
/// passes `codebook_dim = None` (defaults to `dim`), so the
/// `project_in / project_out` linear layers are skipped at this level.
fn load_vector_quantization(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    dim: usize, codebook_size: usize,
) -> Result<VectorQuantizationWeights> {
    let codebook = load_euclidean_codebook(
        st, &format!("{prefix}.codebook"),
        codebook_size, dim,
    )?;
    Ok(VectorQuantizationWeights {
        codebook,
        project_in_w: None, project_in_b: None,
        project_out_w: None, project_out_b: None,
        dim,
    })
}

/// Load a `ResidualVectorQuantizer` at `{prefix}`. The eager builder
/// uses `force_projection = true` for both semantic and acoustic, so
/// `input_proj` / `output_proj` 1×1 convs are always present even
/// when `dim == input_dim == output_dim`.
fn load_residual_vector_quantizer(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    dim: usize, input_dim: usize, output_dim: usize,
    n_q: usize, codebook_size: usize,
) -> Result<ResidualVectorQuantizerWeights> {
    use crate::lazy::load_tensor_as_f32;
    // 1×1 conv1d_no_bias on input: `(out=dim, in=input_dim, k=1)`.
    let in_w = load_tensor_as_f32(
        st, &format!("{prefix}.input_proj.weight"),
    )?;
    let expected_in = dim * input_dim;
    if in_w.len() != expected_in {
        crate::bail!(
            "{prefix}.input_proj.weight: {} elements, expected {expected_in} ({dim}×{input_dim}×1)",
            in_w.len(),
        );
    }
    let out_w = load_tensor_as_f32(
        st, &format!("{prefix}.output_proj.weight"),
    )?;
    let expected_out = output_dim * dim;
    if out_w.len() != expected_out {
        crate::bail!(
            "{prefix}.output_proj.weight: {} elements, expected {expected_out} ({output_dim}×{dim}×1)",
            out_w.len(),
        );
    }
    // RVQ stack at `{prefix}.layers.{i}`.
    let mut layers = Vec::with_capacity(n_q);
    for i in 0..n_q {
        let vq = load_vector_quantization(
            st, &format!("{prefix}.layers.{i}"),
            dim, codebook_size,
        )?;
        layers.push(vq);
    }
    Ok(ResidualVectorQuantizerWeights {
        vq: ResidualVectorQuantizationWeights { layers },
        input_proj_w: Some(Arc::from(in_w)),
        output_proj_w: Some(Arc::from(out_w)),
        dim, input_dim, output_dim,
    })
}

impl SplitResidualVectorQuantizerWeights {
    /// Load split RVQ weights from a HuggingFace `MmapedSafetensors`
    /// checkpoint at `{prefix}` (e.g. `"quantizer"`). Matches the
    /// eager `SplitResidualVectorQuantizer::new` VarBuilder tree:
    ///
    /// - `{prefix}.semantic_residual_vector_quantizer.input_proj.weight`
    /// - `{prefix}.semantic_residual_vector_quantizer.output_proj.weight`
    /// - `{prefix}.semantic_residual_vector_quantizer.layers.{0}.codebook.{embed_sum, cluster_usage}`
    /// - `{prefix}.acoustic_residual_vector_quantizer.input_proj.weight`
    /// - `{prefix}.acoustic_residual_vector_quantizer.output_proj.weight`
    /// - `{prefix}.acoustic_residual_vector_quantizer.layers.{i}.codebook.{embed_sum, cluster_usage}`
    ///   for `i` in `0..n_q - 1`.
    ///
    /// `dim` is the SRVQ internal dimension (e.g. `cfg.quantizer_dim
    /// = 256` for Mimi v0.1); `input_dim` / `output_dim` are the
    /// outside-facing dims (typically `cfg.seanet.dimension = 512`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        prefix: &str,
        dim: usize, input_dim: usize, output_dim: usize,
        n_q: usize, quantizer_bins: usize,
    ) -> Result<Self> {
        let rvq_first = load_residual_vector_quantizer(
            st, &format!("{prefix}.semantic_residual_vector_quantizer"),
            dim, input_dim, output_dim, 1, quantizer_bins,
        )?;
        let n_rest = n_q.saturating_sub(1).max(1);
        let rvq_rest = load_residual_vector_quantizer(
            st, &format!("{prefix}.acoustic_residual_vector_quantizer"),
            dim, input_dim, output_dim, n_rest, quantizer_bins,
        )?;
        Ok(SplitResidualVectorQuantizerWeights {
            rvq_first,
            rvq_rest,
            n_q,
        })
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }

    fn tiny_codebook(
        codebook_size: usize, codebook_dim: usize, nb: &mut dyn FnMut() -> f32,
    ) -> EuclideanCodebookWeights {
        let emb = vec_of(codebook_size * codebook_dim, nb);
        let mut c2_v = Vec::with_capacity(codebook_size);
        for i in 0..codebook_size {
            let mut s = 0.0_f32;
            for j in 0..codebook_dim {
                let e = emb[i * codebook_dim + j];
                s += e * e;
            }
            c2_v.push(s / 2.0);
        }
        EuclideanCodebookWeights {
            embedding: emb,
            c2: Arc::from(c2_v),
            codebook_size,
            codebook_dim,
        }
    }

    fn tiny_vq(dim: usize, codebook_size: usize, nb: &mut dyn FnMut() -> f32) -> VectorQuantizationWeights {
        // dim == codebook_dim → no projection.
        VectorQuantizationWeights {
            codebook: tiny_codebook(codebook_size, dim, nb),
            project_in_w: None, project_in_b: None,
            project_out_w: None, project_out_b: None,
            dim,
        }
    }

    fn tiny_rvq(
        n_q: usize, dim: usize, codebook_size: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ResidualVectorQuantizationWeights {
        ResidualVectorQuantizationWeights {
            layers: (0..n_q).map(|_| tiny_vq(dim, codebook_size, nb)).collect(),
        }
    }

    fn tiny_quantizer(
        n_q: usize, dim: usize, codebook_size: usize, nb: &mut dyn FnMut() -> f32,
    ) -> ResidualVectorQuantizerWeights {
        ResidualVectorQuantizerWeights {
            vq: tiny_rvq(n_q, dim, codebook_size, nb),
            input_proj_w: None,
            output_proj_w: None,
            dim, input_dim: dim, output_dim: dim,
        }
    }

    #[test]
    fn codebook_encode_picks_nearest() {
        let dim = 4;
        let cs = 3;
        // Hand-built codebook: row 0 = [1,0,0,0]; row 1 = [0,1,0,0]; row 2 = [0,0,1,0].
        let emb_v = vec![
            1.0_f32, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let c2_v: Vec<f32> = (0..cs).map(|i| {
            let row = &emb_v[i * dim..(i + 1) * dim];
            row.iter().map(|v| v * v).sum::<f32>() / 2.0
        }).collect();
        let cb = EuclideanCodebookWeights {
            embedding: Arc::from(emb_v),
            c2: Arc::from(c2_v),
            codebook_size: cs,
            codebook_dim: dim,
        };
        // Query rows close to centroids 1, 2, 0.
        let x_data = vec![
            0.1_f32, 0.9, 0.1, 0.0,
            0.0, 0.0, 1.1, 0.0,
            1.2, 0.1, 0.0, 0.0,
        ];
        let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[3, dim]), &Device::cpu());
        let codes = codebook_encode(&x, &cb).unwrap().realize_u32();
        assert_eq!(codes.as_slice(), &[1, 2, 0]);
    }

    #[test]
    fn codebook_decode_index_select_match() {
        let dim = 3;
        let cs = 4;
        let emb_v: Vec<f32> = (0..(cs * dim)).map(|i| i as f32).collect();
        let cb = EuclideanCodebookWeights {
            embedding: Arc::from(emb_v.clone()),
            c2: Arc::from(vec![0.0_f32; cs]),
            codebook_size: cs,
            codebook_dim: dim,
        };
        let idx = LazyTensor::from_u32(
            vec![2_u32, 0, 3], Shape::from_dims(&[3]), &Device::cpu(),
        );
        let out = codebook_decode(&idx, &cb).unwrap().realize_f32();
        let want = vec![
            6.0_f32, 7.0, 8.0,
            0.0, 1.0, 2.0,
            9.0, 10.0, 11.0,
        ];
        for (a, b) in out.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn vq_encode_decode_shapes() {
        let mut nb = rng_seed(2026);
        let dim = 4; let cs = 6; let b = 1; let t = 5;
        let w = tiny_vq(dim, cs, &mut nb);
        let xs = LazyTensor::from_f32(
            (0..(b * dim * t)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[b, dim, t]), &Device::cpu(),
        );
        let codes = vq_encode(&xs, &w).unwrap();
        assert_eq!(codes.shape().dims(), &[b, t]);
        let codes_data = codes.realize_u32();
        for &c in &codes_data {
            assert!((c as usize) < cs, "code {c} >= codebook_size {cs}");
        }
        let recon = vq_decode(&codes, &w).unwrap();
        assert_eq!(recon.shape().dims(), &[b, dim, t]);
    }

    #[test]
    fn rvq_encode_decode_shapes() {
        let mut nb = rng_seed(2026);
        let dim = 4; let cs = 6; let n_q = 3; let b = 1; let t = 5;
        let w = tiny_rvq(n_q, dim, cs, &mut nb);
        let xs = LazyTensor::from_f32(
            (0..(b * dim * t)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[b, dim, t]), &Device::cpu(),
        );
        let codes = rvq_encode(&xs, &w).unwrap();
        assert_eq!(codes.shape().dims(), &[n_q, b, t]);
        let recon = rvq_decode(&codes, &w).unwrap();
        assert_eq!(recon.shape().dims(), &[b, dim, t]);
    }

    #[test]
    fn split_rvq_encode_decode_shapes() {
        let mut nb = rng_seed(2026);
        let dim = 4; let cs = 6; let n_q = 4; let b = 1; let t = 5;
        let w_split = SplitResidualVectorQuantizerWeights {
            rvq_first: tiny_quantizer(1, dim, cs, &mut nb),
            rvq_rest: tiny_quantizer(n_q - 1, dim, cs, &mut nb),
            n_q,
        };
        let xs = LazyTensor::from_f32(
            (0..(b * dim * t)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[b, dim, t]), &Device::cpu(),
        );
        let codes = split_rvq_encode(&xs, &w_split).unwrap();
        assert_eq!(codes.shape().dims(), &[b, n_q, t]);
        let recon = split_rvq_decode(&codes, &w_split).unwrap();
        assert_eq!(recon.shape().dims(), &[b, dim, t]);
        for &v in &recon.realize_f32() { assert!(v.is_finite()); }
    }
}

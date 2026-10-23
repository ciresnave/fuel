// SPDX-License-Identifier: MIT OR Apache-2.0
//! HuggingFace `*-AWQ` (GEMM-version) weight loading (ROADMAP row 31).
//!
//! Companion to [`fuel_graph::NodeHandle::awq_matmul`] (registered as
//! [`fuel_graph::registry::FusedOps::AWQ_MATMUL`]). The fused op takes the
//! quantized operands in baracuda's W4A16 kernel layout; a HuggingFace
//! checkpoint stores the transpose with AutoAWQ's interleaved nibble order.
//! [`load_awq_layer`] reads the three tensors of one `WQLinear_GEMM` layer and
//! converts them with [`fuel_quantized::awq_repack::repack_hf_awq_for_kernel`].
//!
//! ## On-disk layout (per linear layer, `prefix` = e.g.
//! `model.layers.0.self_attn.q_proj`)
//!
//! | key               | dtype       | shape                  |
//! |-------------------|-------------|------------------------|
//! | `{prefix}.qweight`| I32 (or U32)| `[IC, OC/8]`           |
//! | `{prefix}.qzeros` | I32 (or U32)| `[IC/group_size, OC/8]`|
//! | `{prefix}.scales` | F16         | `[IC/group_size, OC]`  |
//!
//! `group_size` is `quantization_config.group_size` in the checkpoint's
//! `config.json`; it is not in the safetensors file, so the caller passes it.
//! Only 4-bit GEMM-version checkpoints are meaningful here (GEMV-version packs
//! differently and is not read).

use crate::Device;
use crate::lazy::Tensor;
use fuel_ir::{Result, Shape};
use fuel_quantized::awq_repack::repack_hf_awq_for_kernel;
use half::f16;

/// An AWQ-quantized linear weight, materialized as the three `Tensor` inputs
/// [`fuel_graph::NodeHandle::awq_matmul`] expects (kernel layout), plus the
/// geometry the matmul builder needs.
pub struct AwqWeight {
    /// `[OC, IC/8]` U32, sequential nibble order.
    pub weight: Tensor,
    /// `[OC, zeros_w]` U32.
    pub zeros: Tensor,
    /// `[OC, 8 * zeros_w]` F16.
    pub scales: Tensor,
    /// Output features.
    pub oc: usize,
    /// Input features.
    pub ic: usize,
    /// Quantization group size (64 or 128).
    pub group_size: usize,
}

impl AwqWeight {
    /// `activations @ dequant(weight)ᵀ` via the fused `AWQ_MATMUL` op.
    /// `activations` must live on the same graph as the weight tensors (build
    /// it from `self.weight.graph_tensor()`). Output `[..., M, OC]` in the
    /// activations' dtype.
    ///
    /// `split_k_iters` is a CUDA launch-tuning parameter (8 is AutoAWQ's
    /// default); the decompose path ignores it.
    pub fn matmul(&self, activations: &Tensor, split_k_iters: i32) -> Tensor {
        Tensor::from_graph_tensor(activations.graph_tensor().awq_matmul(
            self.weight.graph_tensor(),
            self.zeros.graph_tensor(),
            self.scales.graph_tensor(),
            self.group_size,
            split_k_iters,
        ))
    }
}

/// Build an [`AwqWeight`] from HF-layout buffers (`qweight [IC, OC/8]`,
/// `qzeros [IC/g, OC/8]`, `scales [IC/g, OC]`), repacking to the kernel layout.
pub fn awq_from_hf_buffers(
    qweight: &[u32],
    qzeros: &[u32],
    scales: &[f16],
    ic: usize,
    oc: usize,
    group_size: usize,
    device: &Device,
) -> Result<AwqWeight> {
    let b = repack_hf_awq_for_kernel(qweight, qzeros, scales, ic, oc, group_size)
        .map_err(|e| fuel_ir::Error::Msg(format!("awq_from_hf_buffers: {e}")).bt())?;
    let anchor = Tensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), device)?;
    let g = anchor.graph_tensor();
    let weight = g.const_u32_like(b.weight, Shape::from_dims(&[oc, ic / 8]))?;
    let zeros = g.const_u32_like(b.zeros, Shape::from_dims(&[oc, b.zeros_w]))?;
    let scales = g.const_f16_like(b.scales, Shape::from_dims(&[oc, 8 * b.zeros_w]))?;
    Ok(AwqWeight {
        weight: Tensor::from_graph_tensor(weight),
        zeros: Tensor::from_graph_tensor(zeros),
        scales: Tensor::from_graph_tensor(scales),
        oc,
        ic,
        group_size,
    })
}

fn read_u32_words(
    st: &safetensors::SafeTensors,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<u32>> {
    let view = st.tensor(key).map_err(|e| {
        fuel_ir::Error::Msg(format!(
            "load_awq_layer: tensor {key:?} not found in safetensors: {e}"
        ))
        .bt()
    })?;
    if !matches!(
        view.dtype(),
        safetensors::Dtype::I32 | safetensors::Dtype::U32
    ) {
        return Err(fuel_ir::Error::Msg(format!(
            "load_awq_layer: {key:?} dtype={:?}, expected I32 or U32",
            view.dtype()
        ))
        .bt());
    }
    if view.shape() != [rows, cols] {
        return Err(fuel_ir::Error::Msg(format!(
            "load_awq_layer: {key:?} shape={:?}, expected [{rows}, {cols}]",
            view.shape()
        ))
        .bt());
    }
    // I32 and U32 words are the same bits; the nibbles are read as unsigned.
    Ok(view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

/// Load one `WQLinear_GEMM` layer from a parsed safetensors view. `ic`/`oc` are
/// the layer's logical input/output features (from the model config);
/// `group_size` is `quantization_config.group_size`. Every tensor's dtype and
/// shape is checked against them, so a mismatched geometry errors here rather
/// than reading a plausible-looking wrong layer.
pub fn load_awq_layer(
    st: &safetensors::SafeTensors,
    prefix: &str,
    ic: usize,
    oc: usize,
    group_size: usize,
    device: &Device,
) -> Result<AwqWeight> {
    if group_size == 0 || !ic.is_multiple_of(group_size) || !oc.is_multiple_of(8) {
        return Err(fuel_ir::Error::Msg(format!(
            "load_awq_layer: ic={ic} must be a multiple of group_size={group_size} \
             and oc={oc} a multiple of 8"
        ))
        .bt());
    }
    let groups = ic / group_size;
    let qweight = read_u32_words(st, &format!("{prefix}.qweight"), ic, oc / 8)?;
    let qzeros = read_u32_words(st, &format!("{prefix}.qzeros"), groups, oc / 8)?;
    let scales_key = format!("{prefix}.scales");
    let sview = st.tensor(&scales_key).map_err(|e| {
        fuel_ir::Error::Msg(format!(
            "load_awq_layer: tensor {scales_key:?} not found in safetensors: {e}"
        ))
        .bt()
    })?;
    if sview.dtype() != safetensors::Dtype::F16 {
        return Err(fuel_ir::Error::Msg(format!(
            "load_awq_layer: {scales_key:?} dtype={:?}, expected F16",
            sview.dtype()
        ))
        .bt());
    }
    if sview.shape() != [groups, oc] {
        return Err(fuel_ir::Error::Msg(format!(
            "load_awq_layer: {scales_key:?} shape={:?}, expected [{groups}, {oc}]",
            sview.shape()
        ))
        .bt());
    }
    let scales: Vec<f16> = sview
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
        .collect();
    awq_from_hf_buffers(&qweight, &qzeros, &scales, ic, oc, group_size, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_quantized::awq_repack::HF_AWQ_PACK_ORDER;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;

    const IC: usize = 128;
    const OC: usize = 64;
    const GROUP: usize = 64;

    struct Logical {
        q: Vec<u8>, // [IC, OC]
        z: Vec<u8>, // [IC/G, OC]
        s: Vec<f16>,
    }

    fn logical() -> Logical {
        let groups = IC / GROUP;
        let q = (0..IC * OC).map(|i| ((i * 7 + i / 5) % 16) as u8).collect();
        let z = (0..groups * OC).map(|i| ((i * 3 + 5) % 16) as u8).collect();
        let s = (0..groups * OC)
            .map(|i| f16::from_f32(0.02 + 0.005 * (i % 11) as f32))
            .collect();
        Logical { q, z, s }
    }

    /// AutoAWQ's writer: word `col` of a row holds logical column
    /// `col*8 + order[i]` at nibble `i`. Written independently of the loader.
    fn pack_hf(codes: &[u8], rows: usize, oc: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * oc / 2);
        for r in 0..rows {
            for col in 0..oc / 8 {
                let mut w = 0u32;
                for i in 0..8 {
                    w |= u32::from(codes[r * oc + col * 8 + HF_AWQ_PACK_ORDER[i]]) << (i * 4);
                }
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    fn checkpoint(l: &Logical, qdtype: safetensors::Dtype, sdtype: safetensors::Dtype) -> Vec<u8> {
        let groups = IC / GROUP;
        let qw = pack_hf(&l.q, IC, OC);
        let qz = pack_hf(&l.z, groups, OC);
        let sc: Vec<u8> = l.s.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        let mut t: HashMap<String, TensorView<'_>> = HashMap::new();
        t.insert(
            "l.qweight".into(),
            TensorView::new(qdtype, vec![IC, OC / 8], &qw).unwrap(),
        );
        t.insert(
            "l.qzeros".into(),
            TensorView::new(qdtype, vec![groups, OC / 8], &qz).unwrap(),
        );
        t.insert(
            "l.scales".into(),
            TensorView::new(sdtype, vec![groups, OC], &sc).unwrap(),
        );
        safetensors::serialize(&t, None::<HashMap<String, String>>).unwrap()
    }

    /// `y[m][o] = Σ_i x[m][i] · (q[i][o] − z[i/G][o]) · s[i/G][o]` in f64.
    fn reference(l: &Logical, x: &[f32], m: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; m * OC];
        for r in 0..m {
            for o in 0..OC {
                for i in 0..IC {
                    let g = i / GROUP;
                    let w = (f64::from(l.q[i * OC + o]) - f64::from(l.z[g * OC + o]))
                        * f64::from(l.s[g * OC + o].to_f32());
                    y[r * OC + o] += f64::from(x[r * IC + i]) * w;
                }
            }
        }
        y
    }

    fn run(bytes: &[u8], x: &[f32], m: usize) -> Vec<f32> {
        let st = safetensors::SafeTensors::deserialize(bytes).unwrap();
        let w = load_awq_layer(&st, "l", IC, OC, GROUP, &Device::cpu()).unwrap();
        assert_eq!((w.oc, w.ic, w.group_size), (OC, IC, GROUP));
        let act = Tensor::from_graph_tensor(
            w.weight
                .graph_tensor()
                .const_f32_like(x.to_vec(), Shape::from_dims(&[m, IC]))
                .unwrap(),
        );
        // No CPU kernel for the fused op: realize its decompose (the correctness
        // floor), as the lowering-only rule set rewrites it.
        let y = w.matmul(&act, 8);
        let graph = y.graph_tensor().graph().clone();
        let roots = fuel_graph::opt::RuleRegistry::lowering_only()
            .optimize_to_fixpoint(&graph, &[y.graph_tensor().id()]);
        crate::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &Device::cpu())
            .expect("realize lowered awq_matmul")
    }

    #[test]
    fn loaded_layer_matches_the_dequantise_then_matmul_reference() {
        let l = logical();
        let m = 3;
        let x: Vec<f32> = (0..m * IC)
            .map(|i| ((i * 5) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let want = reference(&l, &x, m);
        for qd in [safetensors::Dtype::I32, safetensors::Dtype::U32] {
            let got = run(&checkpoint(&l, qd, safetensors::Dtype::F16), &x, m);
            assert_eq!(got.len(), m * OC);
            let max_err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(*g) - w).abs())
                .fold(0.0, f64::max);
            let scale = want.iter().fold(0.0f64, |a, w| a.max(w.abs()));
            assert!(
                max_err <= 1e-3 * scale.max(1.0),
                "{qd:?}: max_err={max_err} scale={scale}"
            );
        }
    }

    /// Sabotage sibling: a loader that skipped the HF de-interleave (treated the
    /// checkpoint as sequential) must disagree with the reference, so the test
    /// above cannot pass for the wrong reason.
    #[test]
    fn a_sequentially_read_checkpoint_disagrees_with_the_reference() {
        let l = logical();
        let x: Vec<f32> = (0..IC).map(|i| ((i * 5) % 17) as f32 * 0.1 - 0.8).collect();
        let want = reference(&l, &x, 1);
        // Pack the SAME logical codes sequentially (order = identity) and load
        // them through the HF-order loader.
        let groups = IC / GROUP;
        let seq = |codes: &[u8], rows: usize| -> Vec<u8> {
            let mut out = Vec::new();
            for r in 0..rows {
                for col in 0..OC / 8 {
                    let mut w = 0u32;
                    for i in 0..8 {
                        w |= u32::from(codes[r * OC + col * 8 + i]) << (i * 4);
                    }
                    out.extend_from_slice(&w.to_le_bytes());
                }
            }
            out
        };
        let (qw, qz) = (seq(&l.q, IC), seq(&l.z, groups));
        let sc: Vec<u8> = l.s.iter().flat_map(|h| h.to_bits().to_le_bytes()).collect();
        let mut t: HashMap<String, TensorView<'_>> = HashMap::new();
        t.insert(
            "l.qweight".into(),
            TensorView::new(safetensors::Dtype::I32, vec![IC, OC / 8], &qw).unwrap(),
        );
        t.insert(
            "l.qzeros".into(),
            TensorView::new(safetensors::Dtype::I32, vec![groups, OC / 8], &qz).unwrap(),
        );
        t.insert(
            "l.scales".into(),
            TensorView::new(safetensors::Dtype::F16, vec![groups, OC], &sc).unwrap(),
        );
        let bytes = safetensors::serialize(&t, None::<HashMap<String, String>>).unwrap();
        let got = run(&bytes, &x, 1);
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0.0, f64::max);
        assert!(max_err > 1e-2, "sequential read matched: max_err={max_err}");
    }

    #[test]
    fn wrong_dtype_shape_and_missing_tensor_are_refused() {
        let l = logical();
        let ok = checkpoint(&l, safetensors::Dtype::I32, safetensors::Dtype::F16);
        let st = safetensors::SafeTensors::deserialize(&ok).unwrap();
        // Geometry that disagrees with the file.
        assert!(load_awq_layer(&st, "l", IC, OC * 2, GROUP, &Device::cpu()).is_err());
        assert!(load_awq_layer(&st, "l", IC, OC, 128, &Device::cpu()).is_err());
        // Missing prefix.
        assert!(load_awq_layer(&st, "nope", IC, OC, GROUP, &Device::cpu()).is_err());
        // Bad group size.
        assert!(load_awq_layer(&st, "l", IC, OC, 0, &Device::cpu()).is_err());
        // Wrong scales dtype.
        let bad = checkpoint(&l, safetensors::Dtype::I32, safetensors::Dtype::BF16);
        let st = safetensors::SafeTensors::deserialize(&bad).unwrap();
        let e = load_awq_layer(&st, "l", IC, OC, GROUP, &Device::cpu()).err();
        assert!(e.is_some_and(|e| format!("{e}").contains("expected F16")));
    }
}

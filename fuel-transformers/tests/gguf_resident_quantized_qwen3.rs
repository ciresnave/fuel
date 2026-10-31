// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 124 step 2, Qwen3 family: a Q8_0, Q4_K or Q6_K tensor in a GGUF stays quantized
//! from load to the matmul instead of being expanded to f32. The Llama file
//! (`gguf_resident_quantized.rs`) is the step-1 original; the other eight `lazy_quantized_*`
//! loaders carry the same inserted lines and are checked by compile and clippy only.
//!
//! The file is built by hand (tensor infos only: `from_gguf` takes its configuration from the
//! caller), so no real checkpoint is needed. Two files carry the same weights: one stores the
//! projections in the quantized format, the other stores their dequantized values as F32.
//! Loading the second is the old policy's outcome, so equal logits mean the quantized path
//! computes the same function while holding a fraction of the bytes.

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_graph::QuantType;
use fuel_quantized::{BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType, blocks_as_bytes};
use fuel_tensor::lazy::WeightStorage;
use fuel_transformers::models::lazy_quantized_qwen3::QuantizedQwen3Model;
use fuel_transformers::models::lazy_qwen3::Qwen3Config;

const H: usize = 256;
const I: usize = 256;
const VOCAB: usize = 32;
const LAYERS: usize = 1;

fn cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: VOCAB,
        hidden_size: H,
        intermediate_size: I,
        num_hidden_layers: LAYERS,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 64,
        max_position_embeddings: 64,
        sliding_window: None,
        max_window_layers: 0,
        use_sliding_window: false,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        attention_bias: false,
        tie_word_embeddings: false,
    }
}

struct Lcg(u32);
impl Lcg {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((self.0 >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.2
            })
            .collect()
    }
}

/// One tensor to write: GGML type code, GGUF dims (innermost first), raw bytes.
struct T {
    name: String,
    dims: Vec<u64>,
    code: u32,
    data: Vec<u8>,
}

fn f32_tensor(name: &str, dims: &[u64], v: &[f32]) -> T {
    T {
        name: name.into(),
        dims: dims.to_vec(),
        code: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Quantize `[out, in]` row-major values to `B`; return the block bytes and the values the
/// blocks decode to.
fn quantize<B: GgmlType>(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let n_blocks = w.len() / B::BLCK_SIZE;
    let mut blocks = vec![B::zeros(); n_blocks];
    B::from_float(w, &mut blocks);
    let mut back = vec![0.0_f32; w.len()];
    B::to_float(&blocks, &mut back);
    (blocks_as_bytes(&blocks).to_vec(), back)
}

fn build_gguf(tensors: &[T]) -> Vec<u8> {
    let align = |n: usize| n.div_ceil(32) * 32;
    let mut infos = Vec::new();
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for t in tensors {
        offsets.push(off);
        infos
            .write_u64::<LittleEndian>(t.name.len() as u64)
            .unwrap();
        infos.extend_from_slice(t.name.as_bytes());
        infos
            .write_u32::<LittleEndian>(t.dims.len() as u32)
            .unwrap();
        for d in &t.dims {
            infos.write_u64::<LittleEndian>(*d).unwrap();
        }
        infos.write_u32::<LittleEndian>(t.code).unwrap();
        infos.write_u64::<LittleEndian>(off as u64).unwrap();
        off = align(off + t.data.len());
    }
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x4655_4747).unwrap(); // b"GGUF"
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // no metadata
    buf.extend_from_slice(&infos);
    buf.resize(align(buf.len()), 0);
    let base = buf.len();
    buf.resize(base + off, 0);
    for (t, o) in tensors.iter().zip(offsets) {
        buf[base + o..base + o + t.data.len()].copy_from_slice(&t.data);
    }
    buf
}

/// Every projection weight of the model, `(name, out, in)`.
fn projections() -> Vec<(String, usize, usize)> {
    let mut v = Vec::new();
    for l in 0..LAYERS {
        for (n, o, i) in [
            ("attn_q", H, H),
            ("attn_k", H, H),
            ("attn_v", H, H),
            ("attn_output", H, H),
            ("ffn_gate", I, H),
            ("ffn_up", I, H),
            ("ffn_down", H, I),
        ] {
            v.push((format!("blk.{l}.{n}.weight"), o, i));
        }
    }
    v.push(("output.weight".into(), VOCAB, H));
    v
}

/// The tensors that stay f32 in both files: `(name, dims, values)`.
fn dense_tensors(rng: &mut Lcg) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let mut v = vec![(
        "token_embd.weight".to_string(),
        vec![H as u64, VOCAB as u64],
        rng.vec(H * VOCAB),
    )];
    for l in 0..LAYERS {
        for n in ["attn_norm", "ffn_norm"] {
            v.push((format!("blk.{l}.{n}.weight"), vec![H as u64], vec![1.0; H]));
        }
        for n in ["attn_q_norm", "attn_k_norm"] {
            v.push((format!("blk.{l}.{n}.weight"), vec![64], vec![1.0; 64]));
        }
    }
    v.push(("output_norm.weight".into(), vec![H as u64], vec![1.0; H]));
    v
}

/// Returns `(quantized file, dequantized-f32 file, quantized bytes of the projections)`.
fn files<B: GgmlType>(code: u32, plain_gain: f32) -> (Vec<u8>, Vec<u8>, usize) {
    let mut rng = Lcg(0x5eed);
    let mut quant = Vec::new();
    let mut plain = Vec::new();
    let mut qbytes = 0;
    for (name, dims, values) in dense_tensors(&mut rng) {
        quant.push(f32_tensor(&name, &dims, &values));
        plain.push(f32_tensor(&name, &dims, &values));
    }
    for (name, o, i) in projections() {
        let w = rng.vec(o * i);
        let (bytes, back) = quantize::<B>(&w);
        qbytes += bytes.len();
        quant.push(T {
            name: name.clone(),
            dims: vec![i as u64, o as u64],
            code,
            data: bytes,
        });
        let back: Vec<f32> = back.iter().map(|v| v * plain_gain).collect();
        plain.push(f32_tensor(&name, &[i as u64, o as u64], &back));
    }
    (build_gguf(&quant), build_gguf(&plain), qbytes)
}

fn load(bytes: &[u8], tag: &str) -> QuantizedQwen3Model {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{tag}.gguf"));
    std::fs::write(&path, bytes).unwrap();
    QuantizedQwen3Model::from_gguf(&path, &cfg()).unwrap()
}

fn projection_weights(m: &QuantizedQwen3Model) -> Vec<&WeightStorage> {
    let w = &m.inner().weights;
    let mut v = Vec::new();
    for l in &w.layers {
        v.extend([
            &l.attn_q,
            &l.attn_k,
            &l.attn_v,
            &l.attn_o,
            &l.ffn_gate,
            &l.ffn_up,
            &l.ffn_down,
        ]);
    }
    v.push(&w.output);
    v
}

/// Largest logit difference between the quantized-resident model and the f32 model holding the
/// dequantized values times `plain_gain`, as a fraction of the largest f32 logit. Asserts the
/// residency of the quantized model on the way.
fn run<B: GgmlType>(code: u32, qtype: QuantType, plain_gain: f32) -> f32 {
    let (qfile, pfile, qbytes) = files::<B>(code, plain_gain);
    let q = load(&qfile, "quant");
    let p = load(&pfile, "plain");

    for (i, w) in projection_weights(&q).into_iter().enumerate() {
        assert!(
            matches!(w, WeightStorage::Ggml { qtype: t, .. } if *t == qtype),
            "projection {i} of the {qtype:?} file is not resident as Ggml({qtype:?})"
        );
    }
    let resident: usize = projection_weights(&q)
        .iter()
        .map(|w| w.resident_bytes())
        .sum();
    let f32_bytes: usize = projections().iter().map(|(_, o, i)| o * i * 4).sum();
    assert_eq!(
        resident, qbytes,
        "{qtype:?}: resident bytes must equal the block bytes in the file"
    );
    assert!(
        resident * 3 < f32_bytes,
        "{qtype:?}: {resident} resident bytes is not under a third of the {f32_bytes} f32 bytes"
    );
    // The control: the same weights stored as F32 are resident at 4 bytes per element, so the
    // instrument can read a dequantized model as one.
    let plain_resident: usize = projection_weights(&p)
        .iter()
        .map(|w| w.resident_bytes())
        .sum();
    assert_eq!(plain_resident, f32_bytes);

    let toks = [1_u32, 5, 9];
    let lq = q.forward(&toks, 0).unwrap().realize_f32();
    let lp = p.forward(&toks, 0).unwrap().realize_f32();
    assert_eq!(lq.len(), lp.len());
    let mut worst = 0.0_f32;
    let mut scale = 0.0_f32;
    for (a, b) in lq.iter().zip(&lp) {
        assert!(a.is_finite() && b.is_finite());
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    eprintln!("{qtype:?} gain {plain_gain}: worst |dlogit| {worst:e} vs max |logit| {scale:e}");
    worst / scale.max(1e-3)
}

/// Measured on this box 2026-10-30 at gain 1.0: Q8_0 0.75%, Q4_K 1.1%, Q6_K 1.7% of the largest
/// logit (the CPU kernel quantizes activations to 8 bits, ROADMAP row 115, so it is not exact).
/// The bound sits above that and below what a 10% weight-scale error produces
/// (`a_wrong_weight_scale_is_caught`).
const LOGIT_TOL: f32 = 0.04;

fn check<B: GgmlType>(code: u32, qtype: QuantType) {
    let rel = run::<B>(code, qtype, 1.0);
    assert!(
        rel <= LOGIT_TOL,
        "{qtype:?}: logits differ from the dequantized-f32 model by {rel:e} of the largest logit"
    );
}

/// Sabotage sibling: the reference holds weights 10% too large, which the comparison must see.
fn catches<B: GgmlType>(code: u32, qtype: QuantType) {
    let rel = run::<B>(code, qtype, 1.10);
    assert!(
        rel > LOGIT_TOL,
        "{qtype:?}: a 10% weight-scale error moved the logits by only {rel:e}; LOGIT_TOL is too loose"
    );
}

#[test]
fn q8_0_stays_resident() {
    check::<BlockQ8_0>(8, QuantType::Q8_0);
}

#[test]
fn q4_k_stays_resident() {
    check::<BlockQ4K>(12, QuantType::Q4_K_M);
}

#[test]
fn q6_k_stays_resident() {
    check::<BlockQ6K>(14, QuantType::Q6K);
}

#[test]
fn a_wrong_weight_scale_is_caught() {
    catches::<BlockQ8_0>(8, QuantType::Q8_0);
    catches::<BlockQ4K>(12, QuantType::Q4_K_M);
    catches::<BlockQ6K>(14, QuantType::Q6K);
}

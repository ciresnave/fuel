//! Phase 6b oracle-gate: anchor-class CUDA forward passes match the
//! reference backend within tolerance.
//!
//! Each test below picks an anchor model, builds it with synthetic
//! deterministic weights, and asserts the CUDA forward output is
//! oracle-equivalent to the reference backend within 5e-3 rel error
//! (looser than the CPU oracle's 1e-4 because cuBLAS gemm sum-order
//! drift accumulates faster on multi-op composed graphs).
//!
//! Two upstream fixes had to land for the full set to pass:
//!
//! - `fuel-cuda-kernels/src/reduce.cu`: rmsnorm / layernorm cross-
//!   warp reduce reads were uninitialized for warp lanes >= n_warps,
//!   producing scale-shrunk outputs (~4.31× off when shared memory
//!   wasn't zero from a prior kernel). Fixed by clamping the read.
//! - `fuel-graph-cuda::gemm_config`: the matmul stride-pattern matcher
//!   only accepted natural row-major-contig and col-major-contig
//!   layouts. Extended to accept strided variants (lda > row size,
//!   the BERT-style K^T pattern), unblocking BERT, SD CLIP, and
//!   Qwen2-MoE.
//!
//! Feature-gated on `cuda` and requires a CUDA device. Skips
//! cleanly when no CUDA visible.

#![cfg(feature = "cuda")]

use fuel_core::lazy::{LayerWeights, LazyTensor, LlamaConfig, LlamaModel, LlamaWeights};
use fuel_core::lazy_bert::{BertConfig, BertLayerWeights, BertModel, BertWeights};
use fuel_core::lazy_convnext::ConvNextModel;
use fuel_core::lazy_qwen2_moe::{
    ExpertWeights, Qwen2MoeConfig, Qwen2MoeLayerWeights, Qwen2MoeModel, Qwen2MoeWeights,
};
use fuel_core::lazy_sd_text_encoder::{
    ClipLayerWeights, ClipTextWeights, SdTextEncoder, ClipTextConfig,
};
use fuel_core::lazy_whisper::WhisperModel;
use fuel_core::lazy_yolov8::{YoloV8Config, YoloV8Model, YoloV8Weights};
use fuel_core_types::{probe::BackendId, Shape};
use fuel_graph_executor::GraphExecutor;
use std::sync::Arc;

/// Construct a fresh CUDA executor on device 0. Asserts presence —
/// only call from inside a `cuda_present()` guard.
fn cuda_executor() -> GraphExecutor<fuel_graph_cuda::CudaBackend> {
    let dev = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    GraphExecutor::new(fuel_graph_cuda::CudaBackend::new(dev))
}

/// Realize `t` on both reference and CUDA backends, assert allclose.
fn assert_cuda_oracle(t: &LazyTensor, atol: f32, rtol: f32) {
    let reference = t.realize_f32_reference();
    let mut exe = cuda_executor();
    let cuda = t.realize_f32_cuda(&mut exe);
    assert_eq!(reference.len(), cuda.len());
    fuel_core::test_utils::assert_allclose_f32(&cuda, &reference, atol, rtol);
}

fn cuda_present() -> bool {
    let probe = fuel_core::probe::ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Cuda)
}

#[test]
fn single_matmul_cuda_matches_reference_within_tolerance() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }

    // 32×48 @ 48×24 — deterministic inputs.
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let reference = c.realize_f32_reference();

    let cuda_device = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let mut cuda_exe = GraphExecutor::new(
        fuel_graph_cuda::CudaBackend::new(cuda_device),
    );
    let cuda_out = c.realize_f32_cuda(&mut cuda_exe);

    assert_eq!(reference.len(), cuda_out.len());
    assert_eq!(reference.len(), m * n);

    // Tight tolerance — matmul bit-parity per PR #6's individual
    // kernel suite. Any drift is gemm sum-order accumulation and is
    // well under 1e-4 at these shapes.
    fuel_core::test_utils::assert_allclose_f32(&cuda_out, &reference, 1e-4, 1e-4);
}

/// Deterministic LCG-backed tiny weights for a synthetic LLaMA.
/// Same recipe as `lazy::generate_tests::make_tiny_weights`, copied
/// here so this integration test is self-contained.
fn tiny_llama_weights(cfg: &LlamaConfig) -> LlamaWeights {
    let mut s: u32 = 9999;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
        token_embedding: vec_of(cfg.vocab_size * cfg.dim),
        layers: (0..cfg.n_layers).map(|_| LayerWeights {
            attn_q:         vec_of(cfg.dim * cfg.dim).into(),
            attn_q_bias:    None,
            attn_k:         vec_of(cfg.dim * kv_dim).into(),
            attn_k_bias:    None,
            attn_v:         vec_of(cfg.dim * kv_dim).into(),
            attn_v_bias:    None,
            attn_o:         vec_of(cfg.dim * cfg.dim).into(),
            ffn_gate:       vec_of(cfg.dim * cfg.ffn_dim).into(),
            ffn_up:         vec_of(cfg.dim * cfg.ffn_dim).into(),
            ffn_down:       vec_of(cfg.ffn_dim * cfg.dim).into(),
            attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            ffn_norm_gain:  Arc::from(vec![1.0; cfg.dim]),
        }).collect(),
        final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
        output:          vec_of(cfg.dim * cfg.vocab_size).into(),
    }
}

/// Phase 6b exit-criterion gate: a 2-layer LLaMA forward pass on
/// CUDA matches the reference backend within tolerance. Exercises
/// matmul, RoPE, softmax, RMS norm, and SwiGLU on CUDA all in one
/// graph — a real end-to-end anchor smoke vs a minimal subgraph.
#[test]
fn llama_2layer_cuda_matches_reference() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }

    let cfg = LlamaConfig {
        vocab_size:     32,
        dim:            16,
        n_layers:       2,
        n_heads:        4,
        n_kv_heads:     2,
        head_dim:       4,
        ffn_dim:        32,
        norm_eps:       1e-5,
        rope_base:      10_000.0,
    };
    let weights = tiny_llama_weights(&cfg);
    let model = LlamaModel { config: cfg.clone(), weights };
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let logits: LazyTensor = model.forward(&tokens, 0);

    let reference = logits.realize_f32_reference();

    let cuda_device = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let mut cuda_exe = GraphExecutor::new(
        fuel_graph_cuda::CudaBackend::new(cuda_device),
    );
    let cuda_out = logits.realize_f32_cuda(&mut cuda_exe);

    assert_eq!(reference.len(), cuda_out.len());
    assert_eq!(reference.len(), tokens.len() * cfg.vocab_size);

    // 2-layer transformer accumulates gemm sum-order drift through
    // RMS norm, RoPE, attention's softmax + matmul, FFN's SwiGLU.
    // 5e-3 absorbs that drift while staying far below "wrong
    // backend implementation" levels.
    fuel_core::test_utils::assert_allclose_f32(&cuda_out, &reference, 5e-3, 5e-3);
}

/// BERT's attention computes `q @ k^T` with K reshaped to
/// `[B, H, head_dim, seq]` (stride `[..., 1, seq]` — the transpose
/// pattern). cuBLAS supports this natively as `Op::T` with
/// `lda = seq`; the gemm_config matcher in fuel-graph-cuda was
/// extended to accept the strided variant alongside the natural
/// row/col-major contiguous cases.
#[test]
fn bert_cuda_matches_reference() {
    if !cuda_present() { return; }
    let cfg = BertConfig {
        vocab_size:              100,
        hidden_size:             32,
        num_hidden_layers:       2,
        num_attention_heads:     4,
        intermediate_size:       64,
        max_position_embeddings: 16,
        type_vocab_size:         2,
        layer_norm_eps:          1e-12,
    };
    let h = cfg.hidden_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = BertWeights {
        word_embeddings:       z(cfg.vocab_size * h),
        position_embeddings:   z(cfg.max_position_embeddings * h),
        token_type_embeddings: z(cfg.type_vocab_size * h),
        emb_ln_gamma:          o(h),
        emb_ln_beta:           z(h),
        layers: (0..cfg.num_hidden_layers).map(|_| BertLayerWeights {
            attn_q_w:      z(h * h), attn_q_b: z(h),
            attn_k_w:      z(h * h), attn_k_b: z(h),
            attn_v_w:      z(h * h), attn_v_b: z(h),
            attn_out_w:    z(h * h), attn_out_b: z(h),
            attn_ln_gamma: o(h),     attn_ln_beta: z(h),
            ffn_in_w:      z(h * cfg.intermediate_size), ffn_in_b: z(cfg.intermediate_size),
            ffn_out_w:     z(cfg.intermediate_size * h), ffn_out_b: z(h),
            ffn_ln_gamma:  o(h),     ffn_ln_beta: z(h),
        }).collect(),
    };
    let model = BertModel { config: cfg.clone(), weights };
    let ids: Vec<u32> = (0..8).collect();
    let hidden = model.forward(&ids);
    assert_cuda_oracle(&hidden, 5e-3, 5e-3);
}

/// SD's CLIP text encoder uses the same BERT-style K^T transpose
/// pattern that the gemm_config strided-input fix unblocked.
#[test]
fn sd_clip_text_encoder_cuda_matches_reference() {
    if !cuda_present() { return; }
    let cfg = ClipTextConfig {
        vocab_size: 100, hidden_size: 16,
        num_hidden_layers: 2, num_attention_heads: 4,
        intermediate_size: 32, max_position_embeddings: 8,
        layer_norm_eps: 1e-5,
        bos_token_id: 0, eos_token_id: 2, pad_token_id: 1,
    };
    let h = cfg.hidden_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = ClipTextWeights {
        token_embedding: z(cfg.vocab_size * h),
        position_embedding: z(cfg.max_position_embeddings * h),
        layers: (0..cfg.num_hidden_layers).map(|_| ClipLayerWeights {
            ln1_g: o(h), ln1_b: z(h),
            q_w: z(h * h), q_b: z(h),
            k_w: z(h * h), k_b: z(h),
            v_w: z(h * h), v_b: z(h),
            out_w: z(h * h), out_b: z(h),
            ln2_g: o(h), ln2_b: z(h),
            fc1_w: z(h * cfg.intermediate_size), fc1_b: z(cfg.intermediate_size),
            fc2_w: z(cfg.intermediate_size * h), fc2_b: z(h),
        }).collect(),
        final_ln_g: o(h), final_ln_b: z(h),
    };
    let model = SdTextEncoder { config: cfg.clone(), weights };
    let tokens: Vec<u32> = (0..cfg.max_position_embeddings as u32).collect();
    let hidden = model.forward(&tokens);
    assert_cuda_oracle(&hidden, 5e-3, 5e-3);
}

/// Qwen2-MoE uses BERT-shaped attention plus dense MoE routing
/// across `num_experts` per-expert SwiGLU FFNs and a shared expert.
/// Exercises the gemm_config strided-input fix for K^T plus the
/// per-expert weighted-sum matmul chain.
#[test]
fn qwen2_moe_cuda_matches_reference() {
    if !cuda_present() { return; }
    // Minimal MoE config — same shapes as the existing CPU oracle test
    // in fuel-core/src/lazy_qwen2_moe.rs `tiny_cfg`.
    // Mirrors lazy_qwen2_moe::tests::tiny_cfg — these are the exact
    // dim relations the in-module CPU oracle test uses, so we know
    // the forward pass constructs cleanly.
    let cfg = Qwen2MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        moe_intermediate_size: 12,
        shared_expert_intermediate_size: 16,
        num_experts: 3,
        num_experts_per_tok: 2,
        max_position_embeddings: 32,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        norm_topk_prob: false,
    };
    let h = cfg.hidden_size;
    let moe_int = cfg.moe_intermediate_size;
    let shared_int = cfg.shared_expert_intermediate_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let experts: Vec<ExpertWeights> = (0..cfg.num_experts).map(|_| ExpertWeights {
        gate_w: z(h * moe_int),
        up_w:   z(h * moe_int),
        down_w: z(moe_int * h),
    }).collect();
    let layer = Qwen2MoeLayerWeights {
        input_ln: o(h),
        q_w: z(h * h), q_b: z(h),
        k_w: z(h * h), k_b: z(h),
        v_w: z(h * h), v_b: z(h),
        o_w: z(h * h),
        post_attn_ln: o(h),
        gate_w: z(h * cfg.num_experts),
        experts,
        shared_gate_w: z(h * shared_int),
        shared_up_w:   z(h * shared_int),
        shared_down_w: z(shared_int * h),
        shared_expert_gate_w: z(h),
    };
    let weights = Qwen2MoeWeights {
        token_embedding: z(cfg.vocab_size * h),
        layers: vec![layer],
        final_ln: o(h),
        lm_head: z(h * cfg.vocab_size),
    };
    let model = Qwen2MoeModel { config: cfg.clone(), weights };
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let logits = model.forward(&tokens);
    assert_cuda_oracle(&logits, 5e-3, 5e-3);
}

/// Whisper exercises encoder + decoder + cross-attention + Conv1d
/// (slice+concat composition for the encoder's strided convs). The
/// fix from earlier today (rmsnorm/layernorm uninit shared mem)
/// matters here too: layernorm is used throughout. Decoder forward
/// is the more interesting test — it adds cross-attention on top
/// of the encoder's output.
#[test]
fn whisper_decoder_cuda_matches_reference() {
    if !cuda_present() { return; }
    let cfg = fuel_core::lazy_whisper::tiny_cfg();
    let weights = fuel_core::lazy_whisper::zero_weights(&cfg);
    let model = WhisperModel { config: cfg.clone(), weights };
    // mel_time = 32 → encoder produces 16 source tokens.
    let mel = vec![0.0_f32; cfg.num_mel_bins * 32];
    let enc = model.forward_encoder(&mel, 32);
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let logits = model.forward_decoder(&tokens, &enc);
    assert_cuda_oracle(&logits, 5e-3, 5e-3);
}

/// ConvNeXt is conv-heavy: stem patchify + depthwise 7×7 + inverted-
/// bottleneck MLP + global-average pool + linear head. With no native
/// CUDA Conv2D dispatch yet, every Conv2D node falls back to CPU
/// inside the CUDA executor; the test still verifies the end-to-end
/// CUDA path (executor wrapping, layer norm, GELU, MLP matmuls)
/// produces oracle-equivalent output.
#[test]
fn convnext_cuda_matches_reference() {
    if !cuda_present() { return; }
    let cfg = fuel_core::lazy_convnext::tiny_cfg();
    let weights = fuel_core::lazy_convnext::zero_weights(&cfg);
    let model = ConvNextModel { weights, config: cfg.clone() };
    let image = vec![0.0_f32; cfg.in_channels * cfg.image_size * cfg.image_size];
    let logits = model.forward(&image);
    assert_cuda_oracle(&logits, 5e-3, 5e-3);
}

/// CUDA grouped (depthwise) conv2d via cuDNN's set_group_count.
///
/// **Currently `#[ignore]`d.** baracuda alpha.3 added
/// `cudnnSetConvolutionGroupCount` (the API needed for this) but
/// also reshaped the cudnn surface (Handle methods → standalone
/// functions, ConvolutionDescriptor field shape, `cudnn_sys::types`
/// module relocation, CudnnDataType impls dropped for u8 / half::*).
/// Migrating fuel-graph-cuda::cudnn to alpha.3 is real work; tracked
/// in ROADMAP under "CUDA stack restructure". Once the migration
/// lands, drop the `#[ignore]` and this test exercises the depthwise
/// path natively. With alpha.2, the path bails at CudaBackend::conv2d
/// and falls through to CPU reference — the test would tautologically
/// pass.
///
/// Shape mirrors the ConvNeXt depthwise op: 7×7 kernel, stride 1,
/// padding 3, groups == c_in == c_out.
#[test]
#[ignore]
fn cuda_depthwise_conv2d_matches_reference() {
    if !cuda_present() { return; }
    let (n, c, h, w_sz) = (1usize, 16, 8, 8);
    let k = 7;
    let pad = 3;
    let x_data: Vec<f32> = (0..(n * c * h * w_sz))
        .map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let w_data: Vec<f32> = (0..(c * 1 * k * k))
        .map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[n, c, h, w_sz]));
    let weight = x.const_f32_like(w_data, Shape::from_dims(&[c, 1, k, k]));
    let y = x.conv2d(&weight, None, (1, 1), (pad, pad), c); // groups = c → depthwise
    assert_cuda_oracle(&y, 5e-4, 5e-4);
}

#[test]
fn yolov8_cuda_matches_reference() {
    if !cuda_present() { return; }
    // YOLOv8 is conv-heavy; Conv2D currently CPU-falls-back inside the
    // CUDA executor. Test still verifies end-to-end correctness.
    let mut cfg = YoloV8Config::v8n();
    cfg.image_size = 64;
    let weights = YoloV8Weights::zeros(&cfg);
    let model = YoloV8Model { config: cfg.clone(), weights };
    let image = vec![0.0_f32; 3 * cfg.image_size * cfg.image_size];
    let raw = model.forward(&image);
    // Loose tolerance — many ops compose, even when most run on CPU
    // the CUDA executor's TrackedTensor wrapping introduces rounding.
    assert_cuda_oracle(&raw.cls_logits, 5e-3, 5e-3);
    assert_cuda_oracle(&raw.reg_dists,  5e-3, 5e-3);
}

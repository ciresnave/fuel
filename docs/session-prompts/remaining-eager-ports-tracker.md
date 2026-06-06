# Remaining eager model-code ports — tracker

Master index of eager-only model code still pending lazy translation.
Each entry links to a per-port spec in this directory; when a port
ships, move its spec to `shipped/` and strike its tracker line.

Conventions:
- "Foundational" = something that other ports build on; do these first.
- "Splits" = the port spec breaks the work into sub-prompts when it's
  too large for one focused session.
- LOC counts are the eager `fuel-transformers/.../*.rs` line counts.
- "Defer nothing" — every entry must reach completion; if a piece is
  blocked on missing infrastructure, split that infrastructure out as
  its own foundational port and ship it first.

---

## Foundational primitives & loaders

These unblock multiple downstream model ports. Ship these first.

- [x] [Llama-3 RoPE scaling + full Llama port](shipped/port-llama-full.md)
      — `llm/llama.rs` (673 LOC). Needs Llama-3 per-band frequency
      scaling. Foundation for LLaVA composition and any future
      Llama-3.1+ multimodal. **Shipped** in commit (lazy_llama_full
      with LlamaFullConfig + Llama3RopeConfig + LlamaEosToks +
      Llama3Model + build_llama3_rope_tables; injected via
      LlamaModel::run_backbone_with_rope_tables hook; 9 tests).
- [x] [Llama2.c binary weight loader](shipped/port-llama2-c-weights.md)
      — `llm/llama2_c_weights.rs` (239 LOC). Karpathy binary format
      I/O. Trivial but completes the llama2c story. **Shipped**
      (lazy_llama2c::load_llama2c_bin + load_llama2c_bin_path; v0
      format with optional untied lm_head via signed vocab; freq_cis
      tables discarded, rebuilt host-side; 5 loader tests).
- [x] [Conv3D primitive (decomposition path)](shipped/port-conv3d.md)
      — `multimodal/qwen3_vl/conv3d_temporal_2.rs` (80 LOC).
      No native lazy Conv3D — decompose via slicing + matmul. Blocks
      Qwen3-VL vision. **Shipped** (lazy_conv3d::Conv3dTemporal2Weights
      + Conv3dTemporal2Config; weight pre-split in from_raw_weight,
      apply uses narrow + squeeze + 2× conv2d + add + unsqueeze;
      6 tests including hand-computed kernel-1×1 verification).
- [x] [Causal/streaming Conv1d (Mimi conv)](shipped/port-mimi-conv.md)
      — `audio/mimi/conv.rs` (688 LOC). **Sub-ports 1 + 2 + 3 shipped**
      (lazy_mimi_conv::StreamableConv1d + lazy_mimi_conv_transpose::StreamableConvTranspose1d
      + lazy_mimi_conv_wrappers::ConvDownsample1d + ConvTrUpsample1d;
      all state-as-value streaming + WeightNorm baking; 24 tests).
      Sub-port 4 (TimeGroupNorm) intentionally skipped — the Mimi
      encodec composition that shipped does not use it.
- [x] [STFT + log-mel preprocessing](shipped/port-whisper-audio.md)
      — `audio/whisper/audio.rs` (338 LOC). **Shipped** as
      lazy_whisper_audio (pure host-side; Hann window + direct-DFT STFT
      + log-mel; 4 tests).

## Multimodal vision-language

- [x] [Qwen3-VL (text + vision + composition)](shipped/port-qwen3-vl.md)
      — `multimodal/qwen3_vl/*` (1418 LOC total). **All 3 sub-ports
      shipped** (lazy_qwen3_vl_text + MROPE + forward_embeds_with_deepstack
      hook; lazy_qwen3_vl_vision with Conv3D patch embed + cu_seqlens
      block-diagonal mask + DeepStack residuals; lazy_qwen3_vl composition
      with image-token slot scatter + visual residual injection;
      15 tests including end_to_end_tiny_video_plus_text).
- [x] [PaddleOCR-VL (text + vision + composition)](shipped/port-paddleocr-vl.md)
      — `multimodal/paddleocr_vl/*` (3983 LOC total). **All 3 sub-ports
      shipped** (lazy_paddleocr_vl_text Ernie-style decoder; lazy_paddleocr_vl_vision
      tile-grid ViT; lazy_paddleocr_vl composition with aspect-ratio-driven
      tile partitioning + multimodal_projector + slot scatter;
      15 tests including end_to_end_tiny_image_plus_text).
- [x] [Gemma4 audio (Conformer)](shipped/port-gemma4-audio.md)
      — `llm/gemma4/audio.rs` (874 LOC). **Shipped** as
      lazy_gemma4_audio (SSCP conv front-end + Conformer with
      chunked-attention block-band mask + Shaw-style rel-pos bias +
      depthwise light-conv with GLU; 3 tests).

## Diffusion (Phase F)

- [x] [MMDiT (SD3 + Flux foundation)](shipped/port-mmdit.md)
      — `diffusion/mmdit/*` (1118 LOC across 4 files). **Shipped** as
      lazy_mmdit (DoubleStreamBlock + SingleStreamBlock + AdaLN
      modulation + 2D RoPE patch positions; 3 tests including
      zero-scale/zero-gate modulation regressions).
- [x] [Flux (model + autoencoder + sampling + quantized)](shipped/port-flux.md)
      — `diffusion/flux/*` (1689 LOC across 4 files). **Shipped** as
      lazy_flux (FluxModel DiT with QK-Norm + parallel attention +
      9-param modulation + N-dim per-axis RoPE; FluxVae 16-channel
      8x-downsample; FlowMatchScheduler + generate driver;
      QuantizedFluxModel Q4_0 variant; FluxConfig::dev()/schnell()
      presets; 6 tests including flux_vae_round_trip_tiny and
      quantized_flux_model_q4_0_close_to_source).
- [x] [Wuerstchen (cascaded diffusion)](shipped/port-wuerstchen.md)
      — `diffusion/wuerstchen/*` (1176 LOC across 7 files). **Shipped**
      as lazy_wuerstchen (PaellaVQ decoder + Prior + DiffNext UNet
      with GlobalResponseNorm + end-to-end deterministic generate;
      5 tests including end_to_end_generate_tiny).
- [x] [Z-Image (T2I diffusion-class)](shipped/port-z-image.md)
      — `diffusion/z_image/*` (2829 LOC across 7 files). **Shipped**
      as lazy_z_image (Flow-Matching DiT with 3D RoPE + AdaLN-Zero;
      Qwen3-based text encoder; AutoencoderKL with 16-channel latent;
      FlowMatchEulerDiscrete scheduler; 5 tests including
      generate_end_to_end_tiny).
- [x] [Stable Diffusion samplers + attention](shipped/port-sd-samplers.md)
      — `diffusion/stable_diffusion/{ddim,ddpm,uni_pc,euler_ancestral_discrete,schedulers,attention}.rs`
      (2294 LOC). **Sub-ports 2 + 3 + 4 shipped** (lazy_sd_samplers
      DDIM + DDPM, lazy_sd_samplers_euler EulerAncestralDiscrete,
      lazy_sd_samplers_unipc UniPC order 1/2/3 predictor-corrector;
      20 tests). Sub-port 1 (standalone attention block) skipped —
      lazy_sd_unet already inlines the cross-attention; no consumer
      currently needs a standalone surface.

## Audio (top-level wrappers)

- [x] [Mimi encodec top wrapper](shipped/port-mimi-encodec.md)
      — `audio/mimi/encodec.rs` (272 LOC). **Shipped** as
      lazy_mimi_encodec (top-level Mimi codec composition over the
      shipped SeaNet + transformer + quantizer + resampler sub-modules;
      6 tests including encode/decode round-trip and streaming
      equivalence).
- [x] [MetaVoice main LM](shipped/port-metavoice-main.md)
      — `audio/metavoice.rs` (1072 LOC). **Shipped** as lazy_metavoice
      (decoder LM + speaker conditioning + multi-codebook head;
      4 tests including speaker_conditioning_changes_output).

## Quantized variants

- [x] [Quantized Whisper (GGUF)](shipped/port-quantized-whisper.md)
      — `audio/whisper/quantized_model.rs` (411 LOC). **Shipped** as
      lazy_quantized_whisper (Q4_0 substitution over lazy_whisper
      attention + FFN; conv front-end stays F32; 3 tests).
- [x] [Quantized SmolLM3 (GGUF)](shipped/port-quantized-smollm3.md)
      — `llm/smol/quantized_smollm3.rs` (578 LOC). **Shipped** as
      lazy_quantized_smollm3 (Q4_0 substitution over lazy_smollm3;
      3 tests including q4_0_round_trip_via_dequantize).

---

## Conventions for working through this list

1. Start by reading this tracker file. Pick the next un-shipped port.
2. Read its per-port spec end-to-end before writing any code.
3. If the spec has unfilled "Open questions", do the investigation
   and fill them in before starting the port.
4. If the spec says the work splits into sub-prompts, ship them in
   the listed order. Each sub-prompt commit can stand alone.
5. When a port ships:
   - Move its spec to `docs/session-prompts/shipped/`.
   - Strike its tracker line (replace `[ ]` with `[x]` and add the
     commit hash).
6. Tracker grows when you discover a new foundational primitive
   that needs its own spec — add it to the foundational section.

## Phase G / NN-foundation strands

Eager-only surfaces outside `fuel-transformers/*` that need lazy
ports before the eager `fuel_core::Tensor` type-alias flip
(Phase H) can land.

- [x] [Eager fuel-nn loss functions](shipped/port-nn-loss.md)
      — `fuel-nn/src/loss.rs` (~250 LOC). **Shipped** as
      `fuel-core/src/lazy_nn_loss.rs` (Reduction + nll + cross_entropy
      via shipped FusedSoftmaxCrossEntropy + binary_cross_entropy_with_logit
      + mse + huber; 7 tests).
- [x] [Eager fuel-nn optimizers](shipped/port-nn-optim.md)
      — `fuel-nn/src/optim.rs` (~600 LOC). **Shipped** as
      `fuel-core/src/lazy_nn_optim.rs` (LazyOptimizer trait + LazySgd +
      LazyAdamW + LazyVar wrapper; 9 tests including textbook-formula
      goldens for AdamW first-step + decoupled weight decay).
- [~] [Eager fuel-nn Module wrappers](port-nn-layers.md)
      — `fuel-nn/src/{linear,conv,layer_norm,batch_norm,group_norm,embedding,sequential,rnn,lora,quantizable_linear,activation,encoding,init,kv_cache,rotary_emb,fused_ops,cpu_flash_attention,moe,sampling,func,training_context,var_builder,var_map}.rs`
      (~15k LOC eager). **Sub-port 1 shipped** as
      `fuel-core/src/lazy_nn/{mod,linear,embedding}.rs` (LazyModule
      trait + LazyLinear + LazyEmbedding; 5 tests). Sub-ports 2-7
      remain: Conv, Norm, Sequential+Activation, LoRA+QuantizableLinear,
      MoE, Sampling+Init.
- [~] [Training augmentations](port-training-augmentations.md)
      — **Sub-ports 1+2 shipped** as
      `fuel-core/src/lazy_training_augmentations.rs` (LrSchedule trait +
      Cosine/LinearWarmup/Polynomial/Step + clip_grad_norm + clip_grad_value;
      7 tests). Sub-ports 3-5 remain: gradient accumulation,
      mixed-precision (bf16 forward + fp32 master), in-place parameter
      update primitive.
- [~] [Eager fuel-onnx eval](port-onnx-eval.md)
      — `fuel-onnx/src/eval.rs`. **Sub-port 1 shipped** as
      `fuel-onnx/src/lazy_eval.rs` (core arithmetic + Reshape +
      Transpose + Squeeze/Unsqueeze + Flatten + Gather + Reduce ops +
      Constant + ConstantOfShape + Concat + Split + Cast;
      5 tests). Sub-ports 2-4 remain: Conv+Pad+Pool, Norm+Activation+Softmax,
      optional Quantized ops. (Side effect: pre-existing `Device::Cpu`
      breakage on the eager eval.rs was fixed to `Device::cpu()` so the
      crate compiles on this branch.)

## Binary migrations

108 binaries in `fuel-examples/examples/` use eager `fuel::Tensor`
and need their `main.rs` rewritten against the shipped lazy model
modules. Per the "defer nothing" rule, every one ships before this
tracker is considered complete.

- [ ] [Binary migrations](port-binary-migrations.md) — 108 example
      bins. Mostly mechanical rewrite (load weights → construct lazy
      model → forward → output) — the lazy model modules all exist
      from rounds 1-4. Three special categories:
      - **Training bins** (mnist-training, reinforcement-learning):
        migrate to the shipped LazyOptimizer / LazyVar /
        lazy_nn_loss surface from round 5.
      - **custom-ops** (eager CustomOp1 demo): needs a lazy custom-op
        surface — split out as foundational sub-port if one doesn't
        already exist.
      - **llama_multiprocess** (NCCL multi-GPU): defer ONLY if the
        underlying multi-GPU plumbing isn't shipped yet — and if so,
        split that infra out as its own foundational entry above.

## Out of scope for this tracker

- **Phase H (eager Tensor type-alias flip + bin deletion)** — gated
  on every port + binary migration being shipped first. Phase H is
  the *deletion* commit, not an additional port; nothing left to do
  in Phase H once everything above is checked off.

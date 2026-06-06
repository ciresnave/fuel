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
- [~] [Causal/streaming Conv1d (Mimi conv)](shipped/port-mimi-conv.md)
      — `audio/mimi/conv.rs` (688 LOC). **Sub-ports 1 + 2 shipped**
      (lazy_mimi_conv::StreamableConv1d + lazy_mimi_conv_transpose::StreamableConvTranspose1d,
      both with state-as-value streaming + WeightNorm baking; 18 tests).
      Sub-port 3 (ConvDownsample/ConvTrUpsample wrappers) + sub-port 4
      (TimeGroupNorm if encodec needs it) ship in the Mimi-closure batch.
- [x] [STFT + log-mel preprocessing](shipped/port-whisper-audio.md)
      — `audio/whisper/audio.rs` (338 LOC). **Shipped** as
      lazy_whisper_audio (pure host-side; Hann window + direct-DFT STFT
      + log-mel; 4 tests).

## Multimodal vision-language

- [ ] [Qwen3-VL (text + vision + composition)](port-qwen3-vl.md)
      — `multimodal/qwen3_vl/*` (1418 LOC total). Vision tower uses
      Conv3D + cu_seqlens variable-length attention + DeepStack
      residual injection.
- [ ] [PaddleOCR-VL (text + vision + composition)](port-paddleocr-vl.md)
      — `multimodal/paddleocr_vl/*` (3983 LOC total). Ernie-style text
      LM + OCR-specific ViT with window/patch logic.
- [ ] [Gemma4 audio (Conformer)](port-gemma4-audio.md)
      — `llm/gemma4/audio.rs` (874 LOC). SSCP conv + Conformer blocks
      with chunked attention + relative-position embeddings + light
      Conv1d. Completes Gemma4 multimodal arc.

## Diffusion (Phase F)

- [ ] [MMDiT (SD3 + Flux foundation)](port-mmdit.md)
      — `diffusion/mmdit/*` (1118 LOC across 4 files). Joint
      text/image transformer with modulated layers. Shared substrate
      for Flux.
- [ ] [Flux (model + autoencoder + sampling + quantized)](port-flux.md)
      — `diffusion/flux/*` (1689 LOC across 4 files). DiT with
      double + single stream blocks; flow-matching scheduler;
      GGUF-quantized variant.
- [ ] [Wuerstchen (cascaded diffusion)](port-wuerstchen.md)
      — `diffusion/wuerstchen/*` (1176 LOC across 7 files). PaellaVQ
      VAE + Prior + DiffNext + scheduler.
- [ ] [Z-Image (T2I diffusion-class)](port-z-image.md)
      — `diffusion/z_image/*` (2829 LOC across 7 files). Largest
      single diffusion port. Transformer + VAE + text encoder +
      scheduler + sampling + preprocess.
- [~] [Stable Diffusion samplers + attention](shipped/port-sd-samplers.md)
      — `diffusion/stable_diffusion/{ddim,ddpm,uni_pc,euler_ancestral_discrete,schedulers,attention}.rs`
      (2294 LOC). **Sub-port 2 shipped** as lazy_sd_samplers (SdScheduler
      trait + DDIM + DDPM; 4 tests). Sub-port 3 (Euler-ancestral) +
      sub-port 4 (UniPC) + sub-port 1 (standalone attention block, if
      lazy_sd_unet doesn't already inline it) remain.

## Audio (top-level wrappers)

- [ ] [Mimi encodec top wrapper](port-mimi-encodec.md)
      — `audio/mimi/encodec.rs` (272 LOC). Top-level Mimi codec
      composition. Depends on Mimi conv being shipped.
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

## Out of scope for this tracker

- **Phase G (training)** — separate program; tracker covers
  inference/forward-only ports.
- **Phase H (eager Tensor type-alias flip + bin deletion)** — gated
  on every port + binary migration being shipped first.
- **Binary migrations** (lazy bins for VGG, ViT, DinoV2,
  EfficientNet, etc.) — separate strand; the lazy modules already
  exist, the work is just writing the runner binary. Worth its own
  tracker if/when the load picks up.

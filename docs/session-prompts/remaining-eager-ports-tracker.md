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
- [ ] [Llama2.c binary weight loader](port-llama2-c-weights.md)
      — `llm/llama2_c_weights.rs` (239 LOC). Karpathy binary format
      I/O. Trivial but completes the llama2c story.
- [ ] [Conv3D primitive (decomposition path)](port-conv3d.md)
      — `multimodal/qwen3_vl/conv3d_temporal_2.rs` (80 LOC).
      No native lazy Conv3D — decompose via slicing + matmul. Blocks
      Qwen3-VL vision.
- [ ] [Causal/streaming Conv1d (Mimi conv)](port-mimi-conv.md)
      — `audio/mimi/conv.rs` (688 LOC). Streaming Conv1d primitive +
      ConvDownsample / ConvTrUpsample variants. Blocks Mimi encodec
      top-level, MetaVoice main LM.
- [ ] [STFT + log-mel preprocessing](port-whisper-audio.md)
      — `audio/whisper/audio.rs` (338 LOC). Host-side preprocessing
      (no lazy STFT op). Useful for any audio-input model.

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
- [ ] [Stable Diffusion samplers + attention](port-sd-samplers.md)
      — `diffusion/stable_diffusion/{ddim,ddpm,uni_pc,euler_ancestral_discrete,schedulers,attention}.rs`
      (2294 LOC). Diffusion schedulers (mostly host-side CPU
      control) + cross-attention building blocks. (Existing
      lazy_sd_unet/vae/text_encoder cover the model parts.)

## Audio (top-level wrappers)

- [ ] [Mimi encodec top wrapper](port-mimi-encodec.md)
      — `audio/mimi/encodec.rs` (272 LOC). Top-level Mimi codec
      composition. Depends on Mimi conv being shipped.
- [ ] [MetaVoice main LM](port-metavoice-main.md)
      — `audio/metavoice.rs` (1072 LOC). TTS LM (the speaker
      encoder is already shipped as lazy_metavoice_speaker_encoder).

## Quantized variants

- [ ] [Quantized Whisper (GGUF)](port-quantized-whisper.md)
      — `audio/whisper/quantized_model.rs` (411 LOC). GGUF Q-matmul
      substitution over lazy_whisper.
- [ ] [Quantized SmolLM3 (GGUF)](port-quantized-smollm3.md)
      — `llm/smol/quantized_smollm3.rs` (578 LOC). GGUF SmolLM3.

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

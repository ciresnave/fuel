# Port: fuel-examples binaries from eager to lazy

## Eager source

`fuel-examples/examples/*/main.rs` — 108 runner binaries
demonstrating each shipped model. All use eager
`fuel::Tensor` + `fuel_nn::{VarBuilder, Module}` to load
weights, construct the model, and run forward.

## Migration target

Each `fuel-examples/examples/<name>/main.rs` gets rewritten
in-place to:
1. Parse args via clap (unchanged from eager).
2. Load weights via the lazy module's loader
   (`MmapedSafetensors`-backed `LlamaWeights::load_from_mmapped`,
   `Conv3dTemporal2Weights::from_raw_weight`, etc.) instead of
   eager `VarBuilder::from_mmaped_safetensors`.
3. Construct the lazy model
   (`lazy_llama_full::Llama3Model::new`,
   `lazy_qwen3_vl::Qwen3VlModel`, etc.).
4. Forward with `LazyTensor` token / pixel inputs.
5. Realize outputs with `.realize_f32()` for printing /
   sampling.

Each binary must `cargo build --example <name>` cleanly on
main. Per-binary the rewrite is small (~150-500 LOC of glue);
the lazy modules have already shipped from Rounds 1-4.

## Special categories

- **Training bins** (`mnist-training`, `reinforcement-learning`):
  migrate to the Round 5 lazy training surface — `LazyOptimizer`
  (`LazySgd` / `LazyAdamW`), `LazyVar`, `lazy_nn_loss::*`,
  `lazy_nn::{LazyLinear, LazyConv2d, LazyEmbedding}` from the
  shipping Round 6.
- **`custom-ops`** (eager `CustomOp1` demo): if the lazy graph
  doesn't yet expose a custom-op extension surface, the binary
  port has to ship that surface first (foundational sub-port).
  Most likely landing: a `LazyCustomOp1` trait analogous to
  eager `CustomOp1` that injects an `Op::Fused(USER_DEFINED, ...)`
  node.
- **`llama_multiprocess`** (NCCL multi-GPU): port to the lazy
  multi-GPU path if shipped, otherwise split the foundational
  multi-GPU support out as its own port first.
- **Quantized binaries** (`quantized`, `quantized-gemma`,
  `quantized-glm4`, `quantized-lfm2`, `quantized-phi`,
  `quantized-qwen2-instruct`, `quantized-qwen3`,
  `quantized-qwen3-moe`, `quantized-t5`): use the
  `WeightStorage::Q4_0` surface (and `lazy_quantized_*`
  modules where shipped — smollm3, whisper, flux).
- **ONNX binaries** (`onnx`, `onnx-llm`, `onnx_basics.rs`):
  use `fuel-onnx::LazyOnnxEval` from Round 5. May hit
  unsupported-op errors for ops not yet covered by sub-ports
  1+2+3 — list those in the per-binary commit and ship the
  needed sub-port.

## Lazy module mapping (representative subset)

| Eager binary           | Lazy module to use                   |
|------------------------|--------------------------------------|
| llama                  | `lazy_llama_full::Llama3Model`       |
| llama2-c               | `lazy_llama2c::{Llama2cModel, load_llama2c_bin}` |
| mistral                | `lazy_mistral::MistralModel`         |
| qwen / qwen3           | `lazy_qwen3::Qwen3Model`             |
| paddleocr-vl           | `lazy_paddleocr_vl::PaddleOcrVlModel`|
| flux                   | `lazy_flux::FluxModel`               |
| z_image                | `lazy_z_image::ZImageModel`          |
| stable-diffusion-3     | `lazy_mmdit::MmDitModel`             |
| wuerstchen             | `lazy_wuerstchen::WuerstchenModel`   |
| whisper                | `lazy_whisper::WhisperModel` + `lazy_whisper_audio::pcm_to_mel` |
| metavoice              | `lazy_metavoice::MetaVoiceModel`     |
| mimi                   | `lazy_mimi_encodec::MimiEncodecModel`|
| mnist-training         | `lazy_nn::{LazyLinear, LazyConv2d}` + `lazy_nn_optim::LazyAdamW` + `lazy_nn_loss::cross_entropy` |
| reinforcement-learning | `lazy_nn::*` + `lazy_nn_optim::*`    |
| onnx / onnx-llm        | `fuel_onnx::LazyOnnxEval`            |

## Batching strategy

Migrate by family in parallel batches:

1. **Batch A** — Plain decoder-only LLMs (~25 binaries):
   bert, bert_single_file_binary, bigcode, chatglm, codegeex4-9b,
   debertav2, deepseekv2, distilbert, falcon, gemma, gemma4,
   glm4, granite, granitemoehybrid, helium, jina-bert, llama,
   llama2-c, mistral, modernbert, olmo, persimmon (phi family),
   phi, smollm3, stable-lm, starcoder2, yi.
2. **Batch B** — Other LLMs + audio LMs (~15 binaries):
   gte-qwen, mamba, mamba-minimal, mamba2, marian-mt, mixtral,
   moondream, mpt, nomic-bert, nvembed_v2, qwen, recurrent-gemma,
   replit-code, rwkv, splade, stella-en-v5, t5, voxtral,
   xlm-roberta.
3. **Batch C** — Vision (~22 binaries):
   beit, convmixer, convnext, depth_anything_v2, dinov2,
   dinov2reg4, efficientnet, efficientvit, eva2, fastvit, hiera,
   mobileclip, mobilenetv4, mobileone, repvgg, resnet, segformer,
   segment-anything, siglip, vit, vgg, yolo-v3, yolo-v8.
4. **Batch D** — Vision-language + diffusion + audio (~18 binaries):
   based, blip, chinese_clip, clip, colpali, llava, paddleocr-vl,
   paligemma, pixtral, trocr, flux, stable-diffusion,
   stable-diffusion-3, wuerstchen, z_image, csm, encodec, mimi,
   metavoice, musicgen, parler-tts, snac, whisper,
   whisper-microphone.
5. **Batch E** — Quantized (~9 binaries): quantized,
   quantized-gemma, quantized-glm4, quantized-lfm2, quantized-phi,
   quantized-qwen2-instruct, quantized-qwen3, quantized-qwen3-moe,
   quantized-t5.
6. **Batch F** — Special (~5 binaries): custom-ops,
   mnist-training, reinforcement-learning, onnx, onnx-llm,
   onnx_basics, gguf-tokenizer, llama_multiprocess, silero-vad.

Each batch ships as its own commit.

## Verification per binary

`cargo build --example <name>` must succeed. Each binary commits
once all binaries in its batch compile.

## Splits

The 6 batches above are explicit sub-ports. Each can ship
independently. Within a batch the agent walks the binaries
sequentially and migrates them in alphabetical order.

## References

- Eager bins: `fuel-examples/examples/*/main.rs`.
- Lazy modules: `fuel-core/src/lazy_*.rs` (shipped Rounds 1-5).
- Lazy training: `fuel-core/src/{lazy_nn, lazy_nn_optim,
  lazy_nn_loss, lazy_training_augmentations, train}.rs`.
- Lazy ONNX: `fuel-onnx/src/lazy_eval.rs`.

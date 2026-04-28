# fuel
[![discord server](https://dcbadge.limes.pink/api/server/hugging-face-879548962464493619)](https://discord.gg/hugging-face-879548962464493619)
[![Latest version](https://img.shields.io/crates/v/fuel-core.svg)](https://crates.io/crates/fuel-core)
[![Documentation](https://docs.rs/fuel-core/badge.svg)](https://docs.rs/fuel-core)
[![License](https://img.shields.io/github/license/base-org/node?color=blue)](https://github.com/huggingface/fuel/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/huggingface/fuel/blob/main/LICENSE-APACHE)

Fuel is a minimalist ML framework for Rust with a focus on performance (including GPU support) 
and ease of use. Try our online demos: 
[whisper](https://huggingface.co/spaces/lmz/fuel-whisper),
[LLaMA2](https://huggingface.co/spaces/lmz/fuel-llama2),
[T5](https://huggingface.co/spaces/radames/Fuel-T5-Generation-Wasm),
[yolo](https://huggingface.co/spaces/lmz/fuel-yolo),
[Segment
Anything](https://huggingface.co/spaces/radames/fuel-segment-anything-wasm).

## Get started

Make sure that you have [`fuel-core`](https://github.com/huggingface/fuel/tree/main/fuel-core) correctly installed as described in [**Installation**](https://huggingface.github.io/fuel/guide/installation.html).

Let's see how to run a simple matrix multiplication.
Write the following to your `myapp/src/main.rs` file:
```rust
use fuel_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

`cargo run` should display a tensor of shape `Tensor[[2, 4], f32]`.


Having installed `fuel` with Cuda support, simply define the `device` to be on GPU:

```diff
- let device = Device::Cpu;
+ let device = Device::new_cuda(0)?;
```

For more advanced examples, please have a look at the following section.

## Check out our examples

These online demos run entirely in your browser:
- [yolo](https://huggingface.co/spaces/lmz/fuel-yolo): pose estimation and
  object recognition.
- [whisper](https://huggingface.co/spaces/lmz/fuel-whisper): speech recognition.
- [LLaMA2](https://huggingface.co/spaces/lmz/fuel-llama2): text generation.
- [T5](https://huggingface.co/spaces/radames/Fuel-T5-Generation-Wasm): text generation.
- [Phi-1.5, and Phi-2](https://huggingface.co/spaces/radames/Fuel-Phi-1.5-Wasm): text generation.
- [Segment Anything Model](https://huggingface.co/spaces/radames/fuel-segment-anything-wasm): Image segmentation.
- [BLIP](https://huggingface.co/spaces/radames/Fuel-BLIP-Image-Captioning): image captioning.

We also provide some command line based examples using state of the art models:

- [LLaMA v1, v2, and v3](./fuel-examples/examples/llama/): general LLM, includes
  the SOLAR-10.7B variant.
- [Falcon](./fuel-examples/examples/falcon/): general LLM.
- [Codegeex4](./fuel-examples/examples/codegeex4-9b/): Code completion, code interpreter, web search, function calling, repository-level
- [GLM4](./fuel-examples/examples/glm4/): Open Multilingual Multimodal Chat LMs by THUDM
- [Gemma v1 and v2](./fuel-examples/examples/gemma/): 2b and 7b+/9b general LLMs from Google Deepmind.
- [RecurrentGemma](./fuel-examples/examples/recurrent-gemma/): 2b and 7b
  Griffin based models from Google that mix attention with a RNN like state.
- [Phi-1, Phi-1.5, Phi-2, and Phi-3](./fuel-examples/examples/phi/): 1.3b,
  2.7b, and 3.8b general LLMs with performance on par with 7b models.
- [StableLM-3B-4E1T](./fuel-examples/examples/stable-lm/): a 3b general LLM
  pre-trained on 1T tokens of English and code datasets. Also supports
  StableLM-2, a 1.6b LLM trained on 2T tokens, as well as the code variants.
- [Mamba](./fuel-examples/examples/mamba/): an inference only
  implementation of the Mamba state space model.
- [Mistral7b-v0.1](./fuel-examples/examples/mistral/): a 7b general LLM with
  better performance than all publicly available 13b models as of 2023-09-28.
- [Mixtral8x7b-v0.1](./fuel-examples/examples/mixtral/): a sparse mixture of
  experts 8x7b general LLM with better performance than a Llama 2 70B model with
  much faster inference.
- [StarCoder](./fuel-examples/examples/bigcode/) and
  [StarCoder2](./fuel-examples/examples/starcoder2/): LLM specialized to code generation.
- [Qwen1.5](./fuel-examples/examples/qwen/): Bilingual (English/Chinese) LLMs.
- [RWKV v5 and v6](./fuel-examples/examples/rwkv/): An RNN with transformer level LLM
  performance.
- [Replit-code-v1.5](./fuel-examples/examples/replit-code/): a 3.3b LLM specialized for code completion.
- [Yi-6B / Yi-34B](./fuel-examples/examples/yi/): two bilingual
  (English/Chinese) general LLMs with 6b and 34b parameters.
- [Quantized LLaMA](./fuel-examples/examples/quantized/): quantized version of
  the LLaMA model using the same quantization techniques as
  [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [Quantized Qwen3 MoE](./fuel-examples/examples/quantized-qwen3-moe/): support gguf quantized models of Qwen3 MoE models.

<img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/quantized/assets/aoc.gif" width="600">
  
- [Stable Diffusion](./fuel-examples/examples/stable-diffusion/): text to
  image generative model, support for the 1.5, 2.1, SDXL 1.0 and Turbo versions.

<img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg" width="200">

- [Wuerstchen](./fuel-examples/examples/wuerstchen/): another text to
  image generative model.

<img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/wuerstchen/assets/cat.jpg" width="200">

- [yolo-v3](./fuel-examples/examples/yolo-v3/) and
  [yolo-v8](./fuel-examples/examples/yolo-v8/): object detection and pose
  estimation models.

<img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/yolo-v8/assets/bike.od.jpg" width="200"><img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/yolo-v8/assets/bike.pose.jpg" width="200">
- [segment-anything](./fuel-examples/examples/segment-anything/): image
  segmentation model with prompt.

<img src="https://github.com/huggingface/fuel/raw/main/fuel-examples/examples/segment-anything/assets/sam_merged.jpg" width="200">

- [SegFormer](./fuel-examples/examples/segformer/): transformer based semantic segmentation model.
- [Whisper](./fuel-examples/examples/whisper/): speech recognition model.
- [EnCodec](./fuel-examples/examples/encodec/): high-quality audio compression
  model using residual vector quantization.
- [MetaVoice](./fuel-examples/examples/metavoice/): foundational model for
  text-to-speech.
- [Parler-TTS](./fuel-examples/examples/parler-tts/): large text-to-speech
  model.
- [T5](./fuel-examples/examples/t5), [Bert](./fuel-examples/examples/bert/),
  [JinaBert](./fuel-examples/examples/jina-bert/) : useful for sentence embeddings.
- [DINOv2](./fuel-examples/examples/dinov2/): computer vision model trained
  using self-supervision (can be used for imagenet classification, depth
  evaluation, segmentation).
- [VGG](./fuel-examples/examples/vgg/),
  [RepVGG](./fuel-examples/examples/repvgg): computer vision models.
- [BLIP](./fuel-examples/examples/blip/): image to text model, can be used to
  generate captions for an image.
- [CLIP](./fuel-examples/examples/clip/): multi-model vision and language
  model.
- [TrOCR](./fuel-examples/examples/trocr/): a transformer OCR model, with
  dedicated submodels for hand-writing and printed recognition.
- [Marian-MT](./fuel-examples/examples/marian-mt/): neural machine translation
  model, generates the translated text from the input text.
- [Moondream](./fuel-examples/examples/moondream/): tiny computer-vision model 
  that can answer real-world questions about images.

Run them using commands like:
```
cargo run --example quantized --release
```

In order to use **CUDA** add `--features cuda` to the example command line. If
you have cuDNN installed, use `--features cudnn` for even more speedups.

## Cargo feature flags

Fuel is designed so that a CPU-only build compiles without any GPU toolkit
installed. GPU support is opt-in via Cargo feature flags:

| Feature      | What it enables                                              | Requires                |
| ------------ | ------------------------------------------------------------ | ----------------------- |
| *(none)*     | CPU-only build (portable Rust gemm). No GPU code compiled.   | —                       |
| `cuda`       | NVIDIA GPU backend (cuBLAS, cuDNN).                          | CUDA toolkit ≥ 11       |
| `cudnn`      | Enables `cuda` + cuDNN accelerated conv/norm ops.            | CUDA toolkit + cuDNN    |
| `nccl`       | Multi-GPU communication via NVIDIA NCCL.                     | CUDA + NCCL runtime     |
| `vulkan`     | Cross-vendor GPU backend via Vulkan (precompiled SPIR-V).    | Vulkan ≥ 1.3 loader     |
| `metal`      | Apple Silicon / macOS GPU backend (Metal).                   | macOS 13+               |
| `accelerate` | Apple Accelerate BLAS (CPU, macOS only).                     | macOS                   |
| `mkl`        | Intel MKL BLAS (CPU, Linux/Windows). Faster on Intel CPUs.   | Intel oneMKL runtime    |
| `aocl`       | AMD AOCL-BLAS / BLIS (CPU). Faster on Zen-class AMD CPUs.    | AMD AOCL runtime        |

Multiple CPU backends can coexist: with `--features mkl,aocl` both will register
on startup, the [Phase 6b judge](fuel-core/src/judge.rs) profiles each, and the
dispatch table picks the winner per `(op, dtype, size_class)` empirically. The
"wrong" backend for a given CPU (MKL on AMD, AOCL on Intel) just loses the
profile race — it doesn't break anything, and there's no need to gate via
`#[cfg(target_arch)]` heuristics.

### Runtime requirements (where the shared libraries come from)

Cargo features enable the *Rust glue*. The actual numerical kernels live in
vendor-shipped shared libraries that must be resolvable by the OS dynamic
loader at runtime — not at compile time. `cargo build --features X` succeeds
without the runtime present; the binary fails on first call when it can't find
the DLL / `.so` / `.dylib`.

The required runtime library, default install path, and how the OS loader
finds it for each feature:

**`cuda`** — `nvcuda.dll` / `libcuda.so` from the NVIDIA driver install. The
driver installer adds it to system `PATH` / `ld.so` config automatically.

**`cudnn`** — `cudnn*.dll` / `libcudnn.so` from the CUDA toolkit or a
standalone cuDNN install. Add `<cuda>/bin` to `PATH` (Windows) or
`<cuda>/lib64` to `LD_LIBRARY_PATH` (Linux).

**`vulkan`** — `vulkan-1.dll` / `libvulkan.so.1` from the GPU driver install.
The driver installer adds it to system `PATH` / `ld.so` config automatically.

**`metal`** / **`accelerate`** — built into macOS. Always available.

**`mkl`** — `mkl_rt.2.dll` / `libmkl_rt.so.2`. Default install paths:
`C:\Program Files (x86)\Intel\oneAPI\mkl\<ver>` on Windows;
`/opt/intel/oneapi/mkl/<ver>` on Linux. Run `setvars.bat` / `source setvars.sh`
from the oneAPI install dir, OR add the MKL `redist` / `lib/intel64` directory
to `PATH` / `LD_LIBRARY_PATH`.

**`aocl`** — `AOCL-LibBlis-Win-dll.dll` / `libblis.so`. Default install paths:
`C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64` on Windows;
`/opt/AMD/aocl-linux-*/aocl-blis/lib/LP64` on Linux. **Not added to system
`PATH` by the AOCL installer.** Add the `lib/LP64` directory above to `PATH` /
`LD_LIBRARY_PATH` manually before running.

If a runtime library is missing or off the loader's search path, you'll see
errors like `STATUS_DLL_NOT_FOUND` (Windows error 0xc0000135) or
`error while loading shared libraries` (Linux). The fix is always "make the
directory containing the named library visible to the dynamic loader" —
either through PATH/LD_LIBRARY_PATH, the OS config files (`/etc/ld.so.conf.d/`
on Linux), or by copying the DLL next to your executable on Windows.

Backends self-test on startup. `AoclBackend::try_new()` runs a 2×2 sgemm to
verify the library actually loaded; if the DLL is missing the call returns
`Err`, the backend doesn't register, and the rest of Fuel transparently falls
back to other CPU backends. You won't see a hard crash from a missing optional
runtime — only an `eprintln!` from the probe collector.

On Windows, both `AoclBackend::try_new` and `MklBackend::try_new` discover
the vendor's BLIS / mkl_rt DLL automatically — they look at standard install
paths and the `AOCL_ROOT` / `MKLROOT` env vars and prepend the matching `bin`
directory to the process's `PATH` before the load probe. So `cargo run
--features aocl,onemkl` works out of the box on a normal AOCL / oneAPI install
without any manual `setvars.bat` or path-extension shell prep.

### Activating empirical backend selection

Compiling with `--features aocl,onemkl` *registers* both backends, but by
default `LazyTensor::realize_f32()` keeps using the portable Rust `gemm` —
exactly as it did before the per-vendor backends existed. To switch on
per-op empirical routing, the app calls `populate_dispatch_table()` once:

```rust
use fuel_core::dispatch;

fn main() -> fuel_core::Result<()> {
    // Option 1: blocking on the main thread. First run measures every
    // backend × op × size_class (~10–60s depending on hardware) and
    // persists the profile to disk. Every subsequent run loads from
    // disk in sub-millisecond.
    dispatch::populate_dispatch_table()?;

    // Option 2: background thread. Routing kicks in once the judge
    // returns; the first few realize calls fall through to the
    // portable CPU baseline, which is fine.
    std::thread::spawn(|| {
        let _ = dispatch::populate_dispatch_table();
    });

    // Option 3: skip the call entirely. realize_f32 keeps using the
    // portable CPU path; no behaviour change. The disk-cache lazy-load
    // means a previous process's `populate_dispatch_table()` is still
    // honored — `dispatch::cached()` quietly loads it on first use.

    // ... your model code uses LazyTensor::realize_f32() as normal ...
    Ok(())
}
```

Once a dispatch table is cached, every `LazyTensor::realize_f32()` call
consults it per op. On a Zen-class AMD CPU with both AOCL and oneMKL
enabled, this typically picks AOCL or MKL (whichever wins the empirical
race that run) for matmul-heavy work and stays on the portable backend
for the few percent that's elementwise. No code changes downstream — the
`realize_f32()` call site is identical to the no-routing default.

If a previous profile becomes stale (driver upgrade, BLAS lib swap, OS
kernel update with measurably different behaviour), call
`dispatch::invalidate()`. The next `populate_dispatch_table()` re-runs
the judge and overwrites the persisted profile.

### Where to download the vendor runtimes

- **NVIDIA CUDA driver** — [nvidia.com/Download](https://www.nvidia.com/Download/index.aspx).
  The driver alone is enough; the full CUDA toolkit is not required for
  `--features cuda` at runtime.
- **NVIDIA cuDNN** — [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn).
  Requires an NVIDIA developer account.
- **Vulkan loader** — bundled with GPU drivers (NVIDIA, AMD, Intel, Mesa). The
  Vulkan SDK is also available at
  [vulkan.lunarg.com](https://vulkan.lunarg.com/).
- **Intel oneMKL** —
  [intel.com → oneAPI → oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).
- **AMD AOCL** — [amd.com/en/developer/aocl.html](https://www.amd.com/en/developer/aocl.html).
  Linux and Windows installers available.

To add a GPU backend when running examples:

```sh
# NVIDIA GPU
cargo run --features cuda --example <name> --release

# Apple Silicon
cargo run --features metal --example <name> --release

# CPU only (no GPU toolkit needed)
cargo run --example <name> --release
```

To add GPU support to your own project:

```toml
# Cargo.toml
[dependencies]
fuel-core = { version = "0.10.2", features = ["cuda"] }   # NVIDIA
fuel-core = { version = "0.10.2", features = ["metal"] }  # Apple
fuel-core = { version = "0.10.2" }                        # CPU only
```

There are also some wasm examples for whisper and
[llama2.c](https://github.com/karpathy/llama2.c). You can either build them with
`trunk` or try them online:
[whisper](https://huggingface.co/spaces/lmz/fuel-whisper),
[llama2](https://huggingface.co/spaces/lmz/fuel-llama2),
[T5](https://huggingface.co/spaces/radames/Fuel-T5-Generation-Wasm),
[Phi-1.5, and Phi-2](https://huggingface.co/spaces/radames/Fuel-Phi-1.5-Wasm),
[Segment Anything Model](https://huggingface.co/spaces/radames/fuel-segment-anything-wasm).

For LLaMA2, run the following command to retrieve the weight files and start a
test server:
```bash
cd fuel-wasm-examples/llama2-c
wget https://huggingface.co/spaces/lmz/fuel-llama2/resolve/main/model.bin
wget https://huggingface.co/spaces/lmz/fuel-llama2/resolve/main/tokenizer.json
trunk serve --release --port 8081
```
And then head over to
[http://localhost:8081/](http://localhost:8081/).

<!--- ANCHOR: useful_libraries --->

## Useful External Resources
- [`fuel-tutorial`](https://github.com/ToluClassics/fuel-tutorial): A
  very detailed tutorial showing how to convert a PyTorch model to Fuel.
- [`fuel-lora`](https://github.com/EricLBuehler/fuel-lora): Efficient and
  ergonomic LoRA implementation for Fuel. `fuel-lora` has      
  out-of-the-box LoRA support for many models from Fuel, which can be found
  [here](https://github.com/EricLBuehler/fuel-lora/tree/master/fuel-lora-transformers/examples).
- [`fuel-video`](https://github.com/FerrisMind/fuel-video): Rust library for text-to-video generation (LTX-Video and related models) built on Fuel, focused on fast, Python-free inference.
- [`optimisers`](https://github.com/KGrewal1/optimisers): A collection of optimisers
  including SGD with momentum, AdaGrad, AdaDelta, AdaMax, NAdam, RAdam, and RMSprop.
- [`fuel-vllm`](https://github.com/EricLBuehler/fuel-vllm): Efficient platform for inference and
  serving local LLMs including an OpenAI compatible API server.
- [`fuel-ext`](https://github.com/mokeyish/fuel-ext): An extension library to Fuel that provides PyTorch functions not currently available in Fuel.
- [`fuel-coursera-ml`](https://github.com/vishpat/fuel-coursera-ml): Implementation of ML algorithms from Coursera's [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) course.
- [`kalosm`](https://github.com/floneum/floneum/tree/master/interfaces/kalosm): A multi-modal meta-framework in Rust for interfacing with local pre-trained models with support for controlled generation, custom samplers, in-memory vector databases, audio transcription, and more.
- [`fuel-sampling`](https://github.com/EricLBuehler/fuel-sampling): Sampling techniques for Fuel.
- [`gpt-from-scratch-rs`](https://github.com/jeroenvlek/gpt-from-scratch-rs): A port of Andrej Karpathy's _Let's build GPT_ tutorial on YouTube showcasing the Fuel API on a toy problem.
- [`fuel-einops`](https://github.com/tomsanbear/fuel-einops): A pure rust implementation of the python [einops](https://github.com/arogozhnikov/einops) library.
- [`atoma-infer`](https://github.com/atoma-network/atoma-infer): A Rust library for fast inference at scale, leveraging FlashAttention2 for efficient attention computation, PagedAttention for efficient KV-cache memory management, and multi-GPU support. It is OpenAI api compatible.
- [`llms-from-scratch-rs`](https://github.com/nerdai/llms-from-scratch-rs): A comprehensive Rust translation of the code from Sebastian Raschka's Build an LLM from Scratch book.
- [`vllm.rs`](https://github.com/guoqingbao/vllm.rs): A minimalist vLLM implementation in Rust based on Fuel.

If you have an addition to this list, please submit a pull request.

<!--- ANCHOR_END: useful_libraries --->

<!--- ANCHOR: features --->

## Features

- Simple syntax, looks and feels like PyTorch.
    - Model training.
    - Embed user-defined ops/kernels, such as [flash-attention v2](https://github.com/huggingface/fuel/blob/89ba005962495f2bfbda286e185e9c3c7f5300a3/fuel-flash-attn/src/lib.rs#L152).
- Backends.
    - Optimized CPU backend with optional MKL support for x86 and Accelerate for macs.
    - CUDA backend for efficiently running on GPUs, multiple GPU distribution via NCCL.
    - WASM support, run your models in a browser.
- Included models.
    - Language Models.
        - LLaMA v1, v2, and v3 with variants such as SOLAR-10.7B.
        - Falcon.
        - StarCoder, StarCoder2.
        - Phi 1, 1.5, 2, and 3.
        - Mamba, Minimal Mamba
        - Gemma v1 2b and 7b+, v2 2b and 9b.
        - Mistral 7b v0.1.
        - Mixtral 8x7b v0.1.
        - StableLM-3B-4E1T, StableLM-2-1.6B, Stable-Code-3B.
        - Replit-code-v1.5-3B.
        - Bert.
        - Yi-6B and Yi-34B.
        - Qwen1.5, Qwen1.5 MoE, Qwen3 MoE.
        - RWKV v5 and v6.
    - Quantized LLMs.
        - Llama 7b, 13b, 70b, as well as the chat and code variants.
        - Mistral 7b, and 7b instruct.
        - Mixtral 8x7b.
        - Zephyr 7b a and b (Mistral-7b based).
        - OpenChat 3.5 (Mistral-7b based).
        - Qwen3 MoE (16B-A3B, 32B-A3B)
    - Text to text.
        - T5 and its variants: FlanT5, UL2, MADLAD400 (translation), CoEdit (Grammar correction).
        - Marian MT (Machine Translation).
    - Text to image.
        - Stable Diffusion v1.5, v2.1, XL v1.0.
        - Wurstchen v2.
    - Image to text.
        - BLIP.
        - TrOCR.
    - Audio.
        - Whisper, multi-lingual speech-to-text.
        - EnCodec, audio compression model.
        - MetaVoice-1B, text-to-speech model.
        - Parler-TTS, text-to-speech model.
    - Computer Vision Models.
        - DINOv2, ConvMixer, EfficientNet, ResNet, ViT, VGG, RepVGG, ConvNeXT,
          ConvNeXTv2, MobileOne, EfficientVit (MSRA), MobileNetv4, Hiera, FastViT.
        - yolo-v3, yolo-v8.
        - Segment-Anything Model (SAM).
        - SegFormer.
- File formats: load models from safetensors, npz, ggml, or PyTorch files.
- Serverless (on CPU), small and fast deployments.
- Quantization support using the llama.cpp quantized types.

<!--- ANCHOR_END: features --->

## How to use

<!--- ANCHOR: cheatsheet --->
Cheatsheet:

|            | Using PyTorch                            | Using Fuel                                                     |
|------------|------------------------------------------|------------------------------------------------------------------|
| Creation   | `torch.Tensor([[1, 2], [3, 4]])`         | `Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?`           |
| Creation   | `torch.zeros((2, 2))`                    | `Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?`               |
| Indexing   | `tensor[:, :4]`                          | `tensor.i((.., ..4))?`                                           |
| Operations | `tensor.view((2, 2))`                    | `tensor.reshape((2, 2))?`                                        |
| Operations | `a.matmul(b)`                            | `a.matmul(&b)?`                                                  |
| Arithmetic | `a + b`                                  | `&a + &b`                                                        |
| Device     | `tensor.to(device="cuda")`               | `tensor.to_device(&Device::new_cuda(0)?)?`                            |
| Dtype      | `tensor.to(dtype=torch.float16)`         | `tensor.to_dtype(&DType::F16)?`                                  |
| Saving     | `torch.save({"A": A}, "model.bin")`      | `fuel::safetensors::save(&HashMap::from([("A", A)]), "model.safetensors")?` |
| Loading    | `weights = torch.load("model.bin")`      | `fuel::safetensors::load("model.safetensors", &device)`        |

<!--- ANCHOR_END: cheatsheet --->


## Structure

- [fuel-core](./fuel-core): Core ops, devices, and `Tensor` struct definition
- [fuel-nn](./fuel-nn/): Tools to build real models
- [fuel-examples](./fuel-examples/): Examples of using the library in realistic settings
- [fuel-cuda-kernels](./fuel-cuda-kernels/): CUDA custom kernels
- [fuel-datasets](./fuel-datasets/): Datasets and data loaders.
- [fuel-transformers](./fuel-transformers): transformers-related utilities.
- [fuel-flash-attn](./fuel-flash-attn): Flash attention v2 layer.
- [fuel-onnx](./fuel-onnx/): ONNX model evaluation.

## FAQ

### Why should I use Fuel?

<!--- ANCHOR: goals --->

Fuel's core goal is to *make serverless inference possible*. Full machine learning frameworks like PyTorch
are very large, which makes creating instances on a cluster slow. Fuel allows deployment of lightweight
binaries.

Secondly, Fuel lets you *remove Python* from production workloads. Python overhead can seriously hurt performance,
and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.

Finally, Rust is cool! A lot of the HF ecosystem already has Rust crates, like [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers).

<!--- ANCHOR_END: goals --->

### Other ML frameworks

- [dfdx](https://github.com/coreylowman/dfdx) is a formidable crate, with shapes being included
  in types. This prevents a lot of headaches by getting the compiler to complain about shape mismatches right off the bat.
  However, we found that some features still require nightly, and writing code can be a bit daunting for non rust experts.

  We're leveraging and contributing to other core crates for the runtime so hopefully both crates can benefit from each
  other.

- [burn](https://github.com/burn-rs/burn) is a general crate that can leverage multiple backends so you can choose the best
  engine for your workload.

- [tch-rs](https://github.com/LaurentMazare/tch-rs.git) Bindings to the torch library in Rust. Extremely versatile, but they 
  bring in the entire torch library into the runtime. The main contributor of `tch-rs` is also involved in the development
  of `fuel`.

### Common Errors

#### Missing symbols when compiling with the mkl feature.

If you get some missing symbols when compiling binaries/tests using the mkl
or accelerate features, e.g. for mkl you get:
```
  = note: /usr/bin/ld: (....o): in function `blas::sgemm':
          .../blas-0.22.0/src/lib.rs:1944: undefined reference to `sgemm_' collect2: error: ld returned 1 exit status

  = note: some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified
  = note: use the `-l` flag to specify native libraries to link
  = note: use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo
```
or for accelerate:
```
Undefined symbols for architecture arm64:
            "_dgemm_", referenced from:
                fuel_core::accelerate::dgemm::h1b71a038552bcabe in libfuel_core...
            "_sgemm_", referenced from:
                fuel_core::accelerate::sgemm::h2cf21c592cba3c47 in libfuel_core...
          ld: symbol(s) not found for architecture arm64
```

This is likely due to a missing linker flag that was needed to enable the mkl library. You
can try adding the following for mkl at the top of your binary:
```rust
extern crate intel_mkl_src;
```
or for accelerate:
```rust
extern crate accelerate_src;
```

#### Cannot run the LLaMA examples: access to source requires login credentials

```
Error: request error: https://huggingface.co/meta-llama/Llama-2-7b-hf/resolve/main/tokenizer.json: status code 401
```

This is likely because you're not permissioned for the LLaMA-v2 model. To fix
this, you have to register on the huggingface-hub, accept the [LLaMA-v2 model
conditions](https://huggingface.co/meta-llama/Llama-2-7b-hf), and set up your
authentication token. See issue
[#350](https://github.com/huggingface/fuel/issues/350) for more details.

#### Docker build

When building CUDA kernels inside a Dockerfile, nvidia-smi cannot be used to auto-detect compute capability.

You must explicitly set CUDA_COMPUTE_CAP, for example:

```
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Install git and curl
RUN set -eux; \
  apt-get update; \
  apt-get install -y curl git ca-certificates;

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Clone fuel repo
RUN git clone https://github.com/huggingface/fuel.git

# Set compute capability for the build
ARG CUDA_COMPUTE_CAP=90
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

# Build with explicit compute cap
WORKDIR /app
COPY . .
RUN cargo build --release features cuda
```

#### Compiling with flash-attention fails

```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
```

This is a bug in gcc-11 triggered by the Cuda compiler. To fix this, install a different, supported gcc version - for example gcc-10, and specify the path to the compiler in the NVCC_CCBIN environment variable.
```
env NVCC_CCBIN=/usr/lib/gcc/x86_64-linux-gnu/10 cargo ...
```

#### Linking error on windows when running rustdoc or mdbook tests

```
Couldn't compile the test.
---- .\fuel-book\src\inference\hub.md - Using_the_hub::Using_in_a_real_model_ (line 50) stdout ----
error: linking with `link.exe` failed: exit code: 1181
//very long chain of linking
 = note: LINK : fatal error LNK1181: cannot open input file 'windows.0.48.5.lib'
```

Make sure you link all native libraries that might be located outside a project target, e.g., to run mdbook tests, you should run:

```
mdbook test fuel-book -L .\target\debug\deps\ `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.42.2\lib `
-L native=$env:USERPROFILE\.cargo\registry\src\index.crates.io-6f17d22bba15001f\windows_x86_64_msvc-0.48.5\lib
```

#### Extremely slow model load time with WSL

This may be caused by the models being loaded from `/mnt/c`, more details on
[stackoverflow](https://stackoverflow.com/questions/68972448/why-is-wsl-extremely-slow-when-compared-with-native-windows-npm-yarn-processing).

#### Tracking down errors

You can set `RUST_BACKTRACE=1` to be provided with backtraces when a fuel
error is generated.

#### CudaRC error

If you encounter an error like this one `called `Result::unwrap()` on an `Err` value: LoadLibraryExW { source: Os { code: 126, kind: Uncategorized, message: "The specified module could not be found." } }` on windows. To fix copy and rename these 3 files (make sure they are in path). The paths depend on your cuda version.
`c:\Windows\System32\nvcuda.dll` -> `cuda.dll`
`c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublas64_12.dll` -> `cublas.dll`
`c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\curand64_10.dll` -> `curand.dll`

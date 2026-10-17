# Rust ML libraries: feature-gap comparison with Fuel

Requested by CireSnave 2026-10-17: *"feature gap comparison between Fuel and all other similar ML libraries written in Rust. We may find things we want to add to Fuel."* Docs only. Nothing here is started; the candidate rows are in `ROADMAP.md` item 15.

## Method and limits

- One pass, 2026-10-17. Fuel facts are read from `origin/main` at `67ab8a2d6f88718f9374c1e02a092ee333ac8b88` (0.20.10) with `git show` / `git ls-tree` / `git grep`. Other libraries are read from their README, manifests and file listings at the refs below.
- **"Present in source" is not "works".** No build or test was run for any cell. Fuel cells say what exists in the tree; open `docs/gaps.md` rows are cited where they qualify a cell.
- **UNVERIFIED** means the page read did not say, not that the feature is absent. Web pages were read through a summarising fetch tool, so wording is paraphrase; recheck anything load-bearing at the cited URL. "Not in tree" cells are path-name searches and can miss features in ordinarily-named files.
- No star counts and no benchmark numbers are used as evidence.
- Some "main" refs below were fetched at branch head without recording a commit; those are marked.

| Library | Ref read |
|---|---|
| Fuel | `67ab8a2d6f88718f9374c1e02a092ee333ac8b88` |
| candle | `huggingface/candle@12bb5d1f6350f4ea9827d9086f213feec1f0af6a` (0.11.0) |
| mistral.rs | `EricLBuehler/mistral.rs@28ed6daca455b4ae063896014d1ea698b912c539` (git head 0.9.4; crates.io `mistralrs` is only 0.8.1, so docs describe unreleased code) |
| burn | tag v0.22.0 `b8d75ad07f426537594a5beb9309d62d916bcf05`; README/crate list read at `main` |
| cubecl | `tracel-ai/cubecl@040311ebaf0fb5555628b66e5bc3d9f2c04a366d` (crates.io 0.11.0); README at `main` |
| rlx | `MIT-RLX/rlx@326e3e45bd40b6aa6bb44d6558a9a6118c4c3a2e` (README badge 0.2.18). README cites an arXiv paper that was not checked |
| luminal | `jafioti/luminal@77d3060f01dc9d6f649687cf16a651e3f288aa0c`; README at `main`. crates.io `luminal` 0.2.0 (2024-03) is stale and not confirmed to be this repo |
| tract | tag v0.23.9 `4adabb114e6d28e4b289264446774e161dfdaa0c`; README/CHANGELOG at `main` |
| dfdx | `coreylowman/dfdx@4722a99d303f347d6088d95867d007c75ca6dd78` (2024-01-25; crates.io 0.13.0, 2023-07-27). A fork `chelsea0x3b/dfdx` shows a push on 2024-07-23. Effectively dormant |
| wonnx | `webonnx/wonnx@c62f5d336c152102882a163ebc1eb18e63550de8` (2024-05-18). Dormant |
| ort, tch-rs | contrast only (wrap ONNX Runtime / libtorch, not pure Rust): `pykeio/ort@262d9aa1` (2.0.0-rc.14), `LaurentMazare/tch-rs@4227b89b` (0.26.0) |

Fuel scope reference: `docs/architecture/09-non-goals.md` (v0.4). Relevant declarations: Rust-native, Python interop only as a separate leaf crate that "doesn't shape the architecture below it"; tolerance explicit and hierarchical, never a silent fast-math flag; no mandatory or opt-out telemetry; **"Foundation holds mechanism; orchestration is policy and lives above it"** (batching policy, admission control, session lifecycle, sampling policy, serving loops go to `fuel-inference` or a downstream consumer; training orchestration goes to `fuel-training`).

## Capability matrix

Each cell is a short finding; the source is the repo path (Fuel) or the pinned ref's README/tree/manifest named above. `U` = UNVERIFIED.

### Backends and dtypes

| | Backends | Dtypes |
|---|---|---|
| **Fuel** | CPU (`fuel-cpu-backend`, `fuel-cpu-kernels`), CUDA (`fuel-cuda-backend`, via baracuda), Vulkan (`fuel-vulkan-backend`, Slang kernels), Metal (`fuel-metal-*`, macOS only; GAP-043/120/122/254/277 open), MKL and AOCL as sibling CPU kernels, `fuel-kiss-ref-backend` (correctness oracle). No wgpu/WebGPU. | 18 in `fuel-ir/src/dtype.rs`: U8 I8 U32 I16 I32 I64 BF16 F16 F32 F64 F8E4M3 F8E5M2 F6E2M3 F6E3M2 F4 F8E8M0 F8E6M2 Bool. No complex, no TF32. GAP-097/153 qualify the F6/F4 forms. |
| candle | CPU (+MKL, Accelerate), CUDA (cudarc), Metal. No Vulkan/wgpu path names in tree. | 14: U8 U32 I16 I32 I64 BF16 F16 F32 F64 F8E4M3 F6E2M3 F6E3M2 F4 F8E8M0 (`candle-core/src/dtype.rs`). |
| mistral.rs | CUDA (FlashAttn V2/V3, optional cuTile), Metal, CPU, on a pinned candle git rev. ROCm/Vulkan/WebGPU: U. | BF16 shown; full list U; FP8/NVFP4/MXFP4 appear as quant formats. |
| burn | CubeCL: CUDA, ROCm, Metal, Vulkan, WebGPU; CPU via CubeCL-LLVM and pure-Rust "Flex" (Wasm/no_std). LibTorch removed in 0.22. | F64 F32 Flex32 F16 BF16 I64..I8 U64..U8 Bool, plus `QFloat(QuantScheme)` (`burn-std/src/tensor/dtype.rs`). |
| cubecl | CUDA (PTX/LLVM), HIP, Metal and Vulkan via wgpu, WebGPU/WGSL, CPU (LLVM). Kernel layer under burn, not a model framework. | Widest low-precision set: FP4 E2M1, FP6, FP8 incl. UE8M0, F16 BF16 TF32 F64, I8-I64, U8-U64, Bool, complex C32/C64 (`cubecl-ir/src/type.rs`). |
| rlx | CPU, Metal, MLX, CoreML/ANE, CUDA, ROCm, XDNA NPU, oneAPI, TPU, Hexagon QNN, wgpu, native Vulkan, WebGL, Cortex-M, FPGA export, Cerebras WSE. Self-reported coverage of 187 OpKinds: 161 (oneAPI) to 179 (Metal); CUDA "less field mileage than Metal"; Hexagon is an x86 functional sim. | F32, complex C64, INT8/INT4; FP8/FP6/FP4 scaled matmul "in a linked doc" (not read). |
| luminal | CUDA (`luminal_cuda_lite`), Metal, a reference backend, a `torch.compile` backend; ROCm roadmap. | `half`/`float8` deps present; what the IR supports: U. |
| tract | CPU (`tract-linalg`: x86, ARM, SVE), Metal, CUDA 13; wgpu dir present, status U. | Bool, U8-U64, I8-I64, F16 F32 F64, TDim, quantized QI8/QU8/QI32; no BF16 variant. |
| dfdx | CPU, CUDA (cudarc 0.9), WebGPU feature. | f32 f64, optional f16, AMP type; compile-time shaped tensors. |
| wonnx | wgpu: Vulkan, Metal, DX12, GLES best-effort. | 64-bit ints converted to 32-bit; list U. |
| ort / tch-rs | ONNX Runtime execution providers / libtorch (contrast). | inherit the wrapped runtime. |

### Quantization and import/export

| | Quantization | Import / export |
|---|---|---|
| **Fuel** | GGML set in `fuel-ir/src/quantized.rs`: Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1, Q2K-Q8K, F16/BF16. NF4 and AWQ matmul registry ops (`fuel-graph/src/registry/`). **No IQ formats; an IQ GGUF cannot be opened (GAP-211, open).** `k_quants.rs` has no tests (GAP-227). | Import: safetensors, GGUF (file + mmap), PyTorch pickle state dicts "minimum subset" (`fuel-formats/src/`), ONNX evaluated onto the lazy graph (`fuel-onnx`; GAP-128..131 open), DLPack + FDX sidecar. Export: safetensors/ONNX/GGUF write U; training checkpoints in `fuel-training/src/checkpoint.rs`. |
| candle | Same GGML set, no IQ; imatrix quantize paths; AVX/NEON/SIMD128/CUDA/Metal kernels; no GPTQ/AWQ seen (`candle-core/src/quantized/mod.rs`). | safetensors, npz, ggml, GGUF, PyTorch; `candle-onnx` is evaluation only; ONNX export U. |
| mistral.rs | ISQ + UQFF (own format), GGUF, GPTQ, AWQ, HQQ, FP8, BNB, AFQ, MXFP4, NVFP4 (Blackwell-accelerated); **GGUF IQ types explicitly unsupported**; per-layer topology, imatrix. | Import HF checkpoints, GGUF, UQFF; export UQFF (`mistralrs quantize`). |
| burn | `QFloat(QuantScheme)` in the core enum; two-level scales (release notes); kernel coverage U. | ONNX import by codegen to Rust (limited op set), safetensors and PyTorch via `burn-store`, native mmap'able `burnpack`; GGUF U. |
| cubecl | `QuantValue` (E5M2 E4M3 E2M1 Q8F Q8S) and scale dtypes in the IR; dequantize helpers in `cubecl-std/src/quant`. | n/a (kernel layer). |
| rlx | GGUF DequantMatMul "all llama.cpp schemes (incl. IQ/TQ/MX)", GPU dequant on Metal/CUDA/ROCm/WGPU, QAT (STE, LSQ, EMA), ternary TQ2_0, quantized KV cache q4_0/q5_0/q8_0. | Import ONNX (optional ORT fallback), torch.export (inference), GGUF, safetensors; export GGUF (`gguf-convert`), SystemVerilog/bitstream. |
| luminal | Q8 Llama example; API U. | torch.compile backend; ONNX/GGUF/safetensors U. |
| tract | Block-quant Q4_0, ternary Q2_0_T, ONNX MatMulNBits (incl. 2-bit), int8 GEMM kernels, int8/int4 KV cache (CHANGELOG 0.23.x). | Load ONNX, NNEF, TFLite/TF (legacy); save NNEF, TFLite. |
| dfdx | U | safetensors, npz save. |
| wonnx | quantize ops listed but none marked implemented | ONNX only. |

### Autodiff, distributed, compiler

| | Autodiff | Distributed | Fusion / JIT / compiler |
|---|---|---|---|
| **Fuel** | Reverse mode only; backward graph built from the same lazy IR (`fuel-graph/src/grad.rs`, scaffold, ops migrating; per-op backward registry entries for flash-attn, norms, softmax, powi, reduce-max). **No jvp / vmap / forward mode** (0 hits for `jvp\|vmap\|forward.mode` in `*.rs`). Training in `fuel-training` (SGD, AdamW, schedules, checkpoint). | `fuel-parallel` (experimental): single-process multi-device collectives, tensor parallel, 1F1B/GPipe pipeline, distributed cache. Multi-process / NCCL U. | Decompose then an OptimizationMap with per-decision-point alternatives (`docs/architecture/04-optimization.md`, `fuel-graph/src/opt.rs`); JIT seam (`fuel-kernel-seam*`, `fuel-graph/src/jit.rs`); `Op::Scan`; fused-op registry. GAP-238 (clippy never lints `jit`-gated code), GAP-271 (Max/Min commutativity under CSE). |
| candle | `backprop.rs`, `variable.rs`, `custom_op.rs`; optimisers in `candle-nn`. | "multiple GPU distribution via NCCL" (README); one multiprocess example. | None found by name; hand-written kernels; flash-attn v2/v3 crates. |
| mistral.rs | n/a (inference engine). | NCCL tensor parallel (single node), multi-node NCCL env vars, ring backend. | U; optional cuTile path. |
| burn | Reverse mode, `burn-autodiff` decorator over any backend, gradient checkpointing; jvp/hvp/vmap U. | Remote backend (Beta, Iroh/WebSocket); `all_reduce` appears only in a pre-release note. | `burn-fusion` + `burn-cubecl-fusion` kernel fusion on by default for WGPU/CUDA; `burn-capture` graph capture. |
| cubecl | none | `all_gather`, `reduce_scatter`, `broadcast`, `all_to_all` between devices (latest commit); maturity U. | Per-variant JIT, bundleable compile cache, comptime specialisation, autotune. |
| rlx | grad (reverse), jvp/hvp "thin public API", vmap "MVP, leading-axis". | Data/tensor/expert parallel, iroh transport, WideEP MoE with EPLB, pipeline parallel; CUDA NCCL path called a "scaffold". | HIR to MIR to LIR lowering, fusion, memory planning, compile cache; lazy/eager/AOT. |
| luminal | Autograd modelled ahead of time inside the same graph; `luminal_training` crate; jvp/hvp/vmap U. | Topology search "ahead-of-time"; TP/PP/SP listed as roadmap. | 15 primitive ops; egglog e-graph search over rewrites that can rediscover Flash Attention; symbolic shapes with runtime-specialised kernels. |
| tract | none (inference-only) | U | Graph decluttering at load; fused `KvSdpa`, MoE FFN; CUDA/Metal fused attention. |
| dfdx | Reverse tape, type-checked backward. | U | not stated |

### Serving, edge, bindings, model zoo

| | Serving | Wasm / edge | Bindings | Model zoo |
|---|---|---|---|---|
| **Fuel** | `fuel-inference`: scheduler, speculative decoding (draft/target), prefix cache, chunked prefill, eviction, KV compression, multi-session, tiered storage, MoE routing, tool calls, streaming; `fuel-kv-pool` block pool (not yet consumer-wired per project notes); `PagedAttn` op with a naive CPU kernel (CUDA default is slow per project notes). End-to-end behaviour U. | **No wasm tree**: wasm crates retired 2026-08-14 (ROADMAP "Eager-retirement follow-ups" item 2, GAP-185 closed by deletion); the root `README.md` still advertises inherited Candle wasm demos. | **None in repo**; Python allowed only as a separate leaf crate (09). DLPack C header `fuel-ir/include/fuel_dlpack_ext.h`. | About 140 lazy models in `fuel-transformers/src/models/lazy_*.rs`: LLMs (llama, mistral/mixtral, gemma 1-4, qwen 2/3/3-moe/3-vl, phi, deepseek2, olmo, granite, glm4, ...), SSM (mamba, mamba2, rwkv5-7), encoders, vision, CLIP/SigLIP/LLaVA/pixtral, whisper/voxtral/encodec/musicgen/parler-tts, flux/SD/wuerstchen, quantized variants. |
| candle | `KvCache`, `RotatingKvCache`, `ScatteredKvCache`; "Paged attention: none" in `kv_cache.rs`; spec decoding / continuous batching not in tree by name. | First-class: `candle-wasm-examples`, wasm tests, browser demos. | `candle-pyo3`. | Broad (README list), incl. gemma4, flux, voxtral, pixtral. |
| mistral.rs | PagedAttention default on CUDA, ref-counted prefix caching, continuous batching, MTP/draft-head speculative forms (needs paged attention), FP8 KV cache, OpenAI + Anthropic-style HTTP server, MCP, structured output, Prometheus. | Apple Silicon and CPU binaries; no wasm path in tree. | Python (`mistralrs-pyo3`), Rust, HTTP, CLI, web UI. | "45+ architectures" (README). |
| burn | KV cache / paged attention / batching: none found (U beyond the crate list). | Flex on Wasm/no_std; WGPU in browser; book pages on web-assembly and no-std. | none stated | LLM/vision/etc. via ecosystem; not enumerated here. |
| rlx | Sampler chain, quantized KV cache, optional `mmap-kv`; paged attention / continuous batching / server U. | `rlx-web` in-browser (CPU; WebGPU in bring-up), WebGL2, Cortex-M INT8 "Production", android/ios dirs. | pyrlx (PyO3), `rlx-ffi` C ABI, WASM. | model repo separate (`rlx-models`). |
| luminal | U (LLM chat example only) | U | PyTorch backend | U |
| tract | In-place and GPU KV cache, sliding-window ring cache, quantized KV cache, MoE, n-gram and draft-model speculative decoding in an example; paged attention / multi-request batching U. | wasm32/WASI, embedded ARM, musl. | Rust, Python (PyPI), C dirs. | ONNX/NNEF models; Stable Diffusion examples. |
| wonnx | none | WebGPU in browser (npm) | Rust, Python, JS, CLI | Squeezenet, MNIST, BERT tested. |
| dfdx | none | `no-std` feature; wasm U | none | building blocks, `Transformer`. |

### Libraries noted but not compared in depth

`ratchet` (huggingface, WebGPU browser ML, pushed 2026-09-24), `candle-vllm` (candle-based serving with an OpenAI API), `kornia-rs` (computer vision, not a DL framework), `linfa` (classical ML), `tenflowers` (docs.rs hit, claims unverified). The search for other pure-Rust frameworks was shallow and is not exhaustive.

## Where Fuel is not behind (from the same reads)

- Differential oracle and tolerance machinery: `fuel-kiss-ref-backend`, FKC contract verification (`fuel-dispatch/src/fkc/verify/`), hierarchical tolerance budgets (`docs/architecture/07-tolerance.md`). No other library read here states an equivalent. (Fuel's own open gaps GAP-236/058/095/232/243 qualify how far it is enforced.)
- Four compiled backends with one IR including Vulkan; candle has none for Vulkan, wonnx is wgpu-only.
- Model-zoo breadth at about 140 lazy models, comparable to mistral.rs's stated 45+ architectures and wider than candle's README list (counts are of different things: files vs stated architectures).
- Serving building blocks (prefix cache, speculative decoding, tiered KV, scheduler) exist as library code; the open question is consumer wiring, not existence.

## Gap rows

Decision vocabulary: **add** (belongs in Fuel, candidate row in ROADMAP item 15), **defer** (in scope, not now, record trigger), **out-of-scope** (per `docs/architecture/09`). A decision here is a recommendation for the PM / CireSnave, not a ruling.

| # | Gap | Who has it | Fuel today | Fuel decision |
|---|---|---|---|---|
| G1 | **GGUF IQ-type support** (dequant + open the file) | rlx (all IQ/TQ/MX); candle and mistral.rs both say no | rejected by `GgmlDType::from_u32`, GAP-211 open | **add**: already a registered gap; many public GGUFs use IQ. Start at "open and dequantize on CPU", fused kernels later. |
| G2 | **GPTQ / HQQ / FP8 / MXFP4 / NVFP4 weight formats** | mistral.rs (all); tract (MatMulNBits); candle none | AWQ and NF4 registry ops exist; GPTQ/HQQ/MXFP4/NVFP4 not found | **add**, ordered by what checkpoints consumers actually load; verify the AWQ/NF4 ops run end to end first (U). |
| G3 | **Model export**: GGUF / safetensors / ONNX / NNEF write | rlx (GGUF, HDL), tract (NNEF, TFLite), burn (`burnpack`) | export U; `fuel-onnx` header says "bidirectional" but only read/eval code was found | **add** safetensors + GGUF write (small, symmetric with import); **defer** ONNX/NNEF export. First step is a measurement of what write paths exist. |
| G4 | **Forward-mode autodiff: jvp / hvp / vmap** | rlx (MVP/thin) | none (0 hits) | **defer**: not excluded by 09 (autograd is a graph rewrite over the forward IR); trigger = a consumer needing Jacobian-vector products. Reverse-mode backward is still scaffolding (grad.rs), finish that first. |
| G5 | **Per-backend op-coverage matrix**, generated and published | rlx (README table, 187 OpKinds) | coverage lives in per-backend gap rows and probes; no single generated table | **add**: cheap, generated from the dispatch tables in CI, and makes the Metal/CUDA gaps visible. |
| G6 | **Browser / wasm target** | candle, burn (Flex + WGPU), tract, rlx, wonnx, ratchet | wasm crates retired 2026-08-14; no WebGPU backend | **defer**: needs a wasm32 CPU build of `fuel-cpu-backend` first (not measured), and a WebGPU path would be new backend work. Also fix the stale wasm text in the root `README.md` (docs finding, not a gap). |
| G7 | **Multi-process / multi-node distributed (NCCL)** | mistral.rs, candle (NCCL), burn (Remote, beta), rlx (iroh) | single-process `fuel-parallel`, experimental | **defer**: first confirm whether baracuda exposes NCCL (`baracuda-nccl` is already a workspace pin) and what `fuel-parallel` would need; multi-node is policy-heavy (09 orchestration line). |
| G8 | **Continuous-batching serving engine with HTTP API** | mistral.rs, candle-vllm | `fuel-inference` library pieces; no server | **out-of-scope** for Fuel core (09: serving loops and admission policy live above the foundation); a downstream consumer crate may build it. What is in scope: finishing the wiring of `fuel-kv-pool` and the paged-attention default (existing project notes). |
| G9 | **Python bindings** | candle (`candle-pyo3`), mistral.rs, rlx (pyrlx), tract | none | **out-of-scope** below the leaf: 09 allows only a separate leaf crate (`fuel-py`) that does not shape the architecture. Open question for CireSnave, not a code row. |
| G10 | **Interchange formats NNEF / TFLite / torch.export** | tract (NNEF, TFLite), rlx (torch.export) | none | **defer**: ONNX + GGUF + safetensors cover current consumers; revisit if a consumer needs it. |
| G11 | **Exotic targets**: Cortex-M, FPGA, NPU/TPU/ANE, Cerebras | rlx | none | **out-of-scope** for now: no consumer, and rlx itself labels several as untested or simulated. |
| G12 | **Streaming / pulsified inference** | tract | none | **defer**: no consumer. |
| G13 | **Complex and TF32 dtypes** | cubecl (both), rlx (complex) | not in `fuel-ir` dtype list | **defer**: add complex only with a consumer (FFT, signal); TF32 is a matmul-mode decision under the tolerance regime (07), not a storage dtype. |
| G14 | **Search/e-graph compiler** | luminal | optimizer with per-decision-point alternatives; 09 rejects e-graph saturation on the per-realize hot path and autotuning-search optimisation | **out-of-scope** as declared (09); the cached, off-hot-path form is already ROADMAP Phase 10. |
| G15 | **Runtime kernel authoring in the host language** (`#[cube]`) | cubecl, burn | Slang for Vulkan (`fuel-vulkan-kernels`), baracuda/.cu for CUDA, Metal shaders | **defer**: possible consumer of cubecl is a cross-project decision; CLAUDE.md says Vulkan kernels stay fuel-internal Slang. Not a Fuel gap to close alone. |
| G16 | **Per-request LoRA adapters, structured output, agent runtime** | mistral.rs | not in `fuel-inference` list | **out-of-scope** (09 orchestration/policy), except the mechanism for adapter weight application if a consumer asks. |

## Caveats on this document

- This is a feature inventory, not a quality ranking. A backend that exists in two libraries can differ widely in maturity (rlx labels its own CUDA path as less field-tested than Metal; Fuel's Metal rows are open GAPs).
- Pinned refs matter: mistral.rs docs are ahead of its crates.io release; luminal's repo head is ahead of crates.io; burn's README was read at `main` after v0.22.0.
- Every UNVERIFIED cell is a place a second reader can spend an hour and flip it either way.

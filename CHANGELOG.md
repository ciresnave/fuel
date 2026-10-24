# Changelog
This documents the main changes to the `fuel` workspace.

## v0.25.3 - 2026-10-24

### Added

- ROADMAP row 39: the 22 CUDA f32 elementwise unary kernels and the 8 binary ones carry earned numeric accuracy claims (measured on the RTX 4070 Laptop; ledger 1643 -> 1819 records). The exact ops (`neg`, `abs`, `relu`, `step`, `sign`, `floor`, `ceil`, `round`, `maximum`, `minimum`) claim 0; the rest claim 1e-5 relative and absolute, except `exp_f32` at 1e-5 relative and 1e-4 absolute (its results reach 148; measured abs 1.8e-5). Every claim is inside `DEFAULT_ACCURACY_BOUND`. `unary.fkc.md` and `binary.fkc.md` are split into an F32 section per op (carries the claim) and the F16/BF16/F64 remainder, as `norm.fkc.md` was in 0.24.1
- The claims hold only over the measured inputs: `exp` over [-10, 5), `pow` with base [0.5, 4) and exponent [-3, 3), shapes (1,1) (1,1000) (1,3072) (448,1024) (4,8192) plus edge values. `verify/accuracy_elementwise.rs` holds the specs and f64 references; `the_measurement_sees_a_perturbed_output` is the retained sabotage sibling
- `seed_cuda_ledger` builds `bit_stable` probes for `PowElementwise` and `RemElementwise`, so their F32 entries earn `bit_stable` passes

## v0.25.1 - 2026-10-24

### Changed

- CI: the advisory CodeRipper `portfolio.yml` caller is repinned to `d14a22b6` (installs coderipper 0.9.14; the previous pin installed 0.9.6)

## v0.25.0 - 2026-10-24

### Breaking

- MLMF move, slice D (ROADMAP row 42; CireSnave 2026-10-24: "Move any loading/saving portions to MLMF."): fuel's own GGUF parser is deleted. `fuel_formats::gguf::{Content, read_string, VersionedMagic::read, Value::read}` and `fuel_loaders::quantized::gguf_file::Content::read` no longer exist. The one reader is `Content::open(path)` / the new `Content::from_bytes(bytes, path)` over `mlmf-gguf`; `MmapedContent::from_path` and `from_gguf_bytes` use it. `fuel-formats` keeps the wire types (`Value`, `ValueType`, `TensorInfo`, `VersionedMagic`) and the writer; `fuel-tensor-tools` and the `gguf-tokenizer` example are repointed
- **GGUF v1 files are no longer readable** (32-bit counts, written by llama.cpp before mid-2023; `mlmf-gguf` refuses them on purpose and fuel keeps no second parser): a typed error that names "GGUF v1" and says to re-export the model with a current llama.cpp (PM ruling 2026-10-24)

### Changed

- Equivalence, run before the deletion on the 31-file local corpus (`C:\Models\gguf-corpus`): `Content::open` and the old `Content::read` agreed on the magic, the whole metadata map, the tensor-info table, the unsupported-tensor table and the data offset for 30 files; the 31st is the v1 file. The agreement is frozen as `fuel-loaders/tests/fixtures/gguf_corpus_digests.json` (hashes produced by the old reader) and checked by `tests/gguf_corpus_equivalence.rs`; the four known metadata-decode differences (duplicate keys keep the first, a trailing NUL is kept, a bool byte 2..=255 is accepted, a non-UTF-8 key or tensor name is refused) do not occur on any of the 30. The hand-built `gguf_open_mlmf_adapter.rs` tests now assert absolute values instead of comparing with the deleted reader

### Known

- `--features jit-synth` does not compile at 0.25.0: `baracuda-cuda-seam =0.1.0-alpha.2` requires `fuel-kernel-seam ^0.24`, and the workspace seam crates are now 0.25.0. It returns with `baracuda-cuda-seam` alpha.3 (on `^0.25`) and the pin bump in 0.25.1. This is the cost of the seam crates sharing the workspace version (`docs/design-notes/2026-10-24-seam-crate-versioning.md`)

### Fixed

- CI: the advisory CodeRipper `portfolio` job is repinned to coderipper `30e012b7`
- GAP-205 closed by deletion (the old parser panicked on `general.alignment = 0`, made unbounded allocations from declared lengths, lossily decoded names and kept the last of duplicate keys); GAP-345 (the `open` vs `read` parity differences) closed with it

## v0.24.5 - 2026-10-24

### Changed

- ROADMAP row 41a: `jit-synth` is back on a single seam copy: `baracuda-cuda-seam` `=0.1.0-alpha.2` (requires `fuel-kernel-seam{,-types}` ^0.24.0) replaces `=0.0.1-alpha.161` (^0.23.0). Control: `jit_synth_kernel_live` 6 of 6 in 3 of 3 default (parallel) runs on the RTX 4070. The next breaking bump of the seam crates needs another republish of that crate

## v0.24.4 - 2026-10-24

### Changed

- MLMF move, slice B (ROADMAP row 42): `fuel_loaders::safetensors::{MmapedSafetensors, BufferedSafetensors}` read the header and tensor directory with `mlmf-safetensors` instead of the upstream `safetensors` deserializer. The public API and the `safetensors::TensorView` it returns are unchanged (226 files reference the type). Equivalence, run against the old reader first and again after the swap (`fuel-loaders/tests/safetensors_equivalence.rs`): identical tensor names, dtypes, shapes and bytes for all 15 dtypes plus an empty and a scalar tensor, two-shard last-wins routing, and the real Qwen3-0.6B-AWQ (702 tensors) and Qwen3.5-0.8B (488 tensors) checkpoints; a flipped byte is caught
- Two known differences: a tensor whose dtype `mlmf-safetensors` cannot resolve is unreachable by `get` (the error names it) where the old reader returned a view; and the upstream reader's contiguity validation of the data region is not replicated (each tensor's range is still checked to lie inside the file, so a truncated file is still refused at open)

## v0.24.3 - 2026-10-24

### Added

- MLMF move, slice A (ROADMAP row 42): `fuel_loaders::awq::{awq_group_size, locate_awq_layers}`, AWQ configuration parsing and packed-layer discovery over `mlmf-awq` and `mlmf-safetensors` (one safetensors file; sharded AWQ is not merged yet). The oracle `qwen3_awq_oracle.rs` uses it instead of its own hand parser, which is deleted. Equivalence on the real `Orion-zhen/Qwen3-0.6B-AWQ` checkpoint, checked before the deletion (`fuel-transformers/tests/qwen3_awq_mlmf_slice_a.rs`): 196 layers located = the 196 projections `load_awq_from_mmapped` reads, every `(out, in)` equal, group size 128

### Changed

- `fuel-loaders`: `mlmf-core`, `mlmf-gguf`, `mlmf-source-file` 0.5.10 -> 0.5.11 (the `Content::open` adapter tests still pass 6 of 6)

## v0.24.2 - 2026-10-24

### Changed

- docs only: `docs/design-notes/2026-10-24-mlmf-move-survey.md`, step 1 of moving model loading and saving to MLMF (CireSnave, 2026-10-24: "Agreed on decision 265. Move any loading/saving portions to MLMF."), and ROADMAP row 42. No code, no API change

## v0.24.1 - 2026-10-24

### Added

- ROADMAP row 39: the CUDA f32 `RmsNormLastDim` (`rms_f32`) and `SoftmaxLastDim` (`softmax_f32`) carry earned numeric accuracy claims, `max_relative` = `max_absolute` = 1e-5 (measured worst 2.4e-7 abs / 1.6e-7 rel and 7.9e-8, over the decode and prefill shapes, RTX 4070 Laptop; ledger 1631 -> 1643: +4 accuracy records and +8 `bit_stable` passes at the re-keyed hashes). `norm.fkc.md` and `softmax.fkc.md` are split into an F32 section (carries the claim) and the F16/BF16/F64 section, as `gemm_dense.fkc.md` was in 0.23.2. `fuel-dev-tools seed-cuda-accuracy` / `check-cuda-accuracy` now cover MatMul, RmsNorm and Softmax
- `verify/accuracy_common.rs` now holds the measurement functions, so the Vulkan and CUDA seeders run the same code through a `Runner` (the Vulkan numbers are unchanged: matmul 2.3e-4 / 8.8e-5, rms 2.3e-7, softmax 5.8e-8, rope 1.2e-7)

### Fixed

- The shared norm and softmax cases passed a flat `[rows * cols]` shape; the CUDA kernels take the row length from the last dimension of the layout, so every case with more than one row was normalised over the whole buffer (measured: softmax `rows=16 cols=22` abs error 0.67). They now pass `[rows, cols]`. Nothing was written to the ledger from the flat shape

## v0.24.0 - 2026-10-23

### Breaking

- `fuel-kernel-seam` depends on `unpopped-vocab` 0.22 instead of `baracuda-kernels-types`, so the types in its public API (`OperandDesc`, `ArchSku`, `TargetId`) are `unpopped-vocab` 0.22's: a consumer on an older vocabulary no longer type-checks against it

### Changed

- ROADMAP row 41a (`jit-synth` does not compile at 0.24.0 until `baracuda-cuda-seam` alpha.162 on `fuel-kernel-seam ^0.24` is published and the pin bumped; the figures below are from 0.23.3): `fuel-dispatch`'s `jit-synth` feature uses `baracuda-cuda-seam` (`=0.0.1-alpha.161`) for `BaracudaSynthesizer`, which left `baracuda-cuda-emit` in 0.20; the baracuda family moves to alpha.158 (nvrtc alpha.153; cudnn, curand, runtime stay at alpha.147, their latest), so one `unpopped-vocab` (0.22.0) is in the graph. Verified: `jit_synth_kernel_live` 6/6 on the RTX 4070 in 6 of 6 default (parallel) runs (`main` cannot build `jit-synth` at all at `3002b4e1`: two `unpopped_vocab` versions, so there is no baseline; before the fix below 5 of 5 parallel runs failed 1-3 of the 6 tests), `fuel-dispatch` lib 818 passed under `gpu-run`, `baracuda_awq_live` 2/2
- `fuel-dispatch/tests/jit_synth_kernel_live.rs`: the six live tests share the process-global JIT slot bank and kernel binding table (slot reclamation asserts on the whole bank; adoption binds by signature and saw another test's kernel as `Ambiguous`), so each now takes a process-wide lock on its first line and holds it until its thread ends
- `fuel-kernel-seam` names `unpopped-vocab` directly because `baracuda-kernels-types` only re-exports `ElementKind`, `OperandDesc`, `ArchSku`, `TargetId` from it: the published seam crate carries no baracuda dependency

## v0.23.3 - 2026-10-23

### Added

- ROADMAP row 40 stage 1: `fuel-transformers::models::lazy_qwen3_5` (`Qwen3_5Config`, `Qwen3_5Weights::load_from_mmapped`, `Qwen3_5Model`), the config and weight loader for the dense Qwen3.5 Gated DeltaNet / full-attention hybrid. Every tensor is shape-checked against the config; decoder RMS norms are stored as `1 + w` (zero-centred, measured on the real checkpoint). `Qwen3_5Model::forward` is a typed refusal until the recurrence exists (GAP-368). `tests/qwen3_5_loader_oracle.rs` loads `Qwen/Qwen3.5-0.8B` from `FUEL_QWEN35_DIR` (Hub skip when unset): 320 of 320 language-model tensors, 0 missing, 0 unread

## v0.23.2 - 2026-10-23

### Added

- ROADMAP row 39 (first claim): the CUDA f32 `MatMul` now carries an earned numeric accuracy claim, `max_relative` and `max_absolute` 2e-4 (measured worst 3.8e-5 and 5.0e-5 over 12 shapes, k up to 4096, against an f64 host matmul, RTX 4070 Laptop). `fuel-dev-tools seed-cuda-accuracy` earns it, `check-cuda-accuracy` measures without writing; the helpers shared with the Vulkan seeder moved to `verify/accuracy_common.rs`
- `kernel-contracts/cuda/gemm_dense.fkc.md` is split into `matmul_f32` (carries the claim) and `matmul` (F16, BF16, F64). One section fanned over four dtypes, so a numeric claim there would have needed a pass for each, and F16/BF16 cannot meet the f32 bound. The dtype list is part of the revision hash, so the four MatMul entries are re-keyed: the ledger gains 4 `bit_stable_on_same_hardware` passes at the new hashes plus the 2 accuracy passes (the old-hash records stay, now unused)

### Changed

- CI: the CodeRipper `portfolio` job is repinned to coderipper `a48fc6cc` (sweeps every workspace member)

## v0.23.1 - 2026-10-23

### Added

- CI: an advisory, non-blocking CodeRipper `portfolio` job in `rust-ci.yml` (reusable workflow pinned to a coderipper commit SHA; not a required check)

### Changed

- `fuel-transformers`: `Qwen3Weights::load_layer` split further (`load_norm_gains`) to get under the cyclomatic-complexity limit; no behaviour change

## v0.23.0 - 2026-10-23

### Breaking

- `fuel-tensor`: `WeightStorage` gains a variant, `Awq(Arc<AwqHostWeight>)` (it is not `#[non_exhaustive]`, so an exhaustive `match` on it in downstream code no longer compiles; `fuel-nn`'s `Conv2d::absorb_bn` is updated in this change)

### Added

- `fuel-tensor`: `awq::AwqHostWeight`, a graph-independent AWQ linear in the kernel layout (`from_hf_buffers`, `load`, `matmul`), applied by `WeightStorage::apply_linear`
- `fuel-transformers`: `Qwen3Weights::load_awq_from_mmapped` loads a HuggingFace `*-AWQ` Qwen3 checkpoint; `load_from_mmapped` and it share one body
- ROADMAP row 31c: `tests/qwen3_awq_oracle.rs` runs `Orion-zhen/Qwen3-0.6B-AWQ` end to end and compares 16 greedy tokens with the independent HF/AutoAWQ reference (reads the checkpoint from `FUEL_QWEN3_AWQ_DIR`; a `Hardware::Hub` skip when absent). GAP-367 records the missing CPU `AWQ_MATMUL` binding

### Fixed

- `fuel-cuda-backend`: `awq_gemm_f16` now calls `awq_can_implement_f16` before any allocation or launch (it was already rejected by baracuda's `_run`, which repeats the check; measured on the RTX 4070: `n=96`, `split_k_iters` 0 and -1, `k=256` with `split_k_iters=7`, `group_size=96` and `k=200` all return `Err`, and the honest control runs). The sm_80+ requirement is not a shape check and is not enforced there
- Corrected overclaims about AWQ on CUDA (contract, wrapper comment, `fused.rs` KNOWN_GAPS text, `OpParams::AwqMatmul` doc, the 0.22.7 entry below): only F16 activations are bound, and an F32/BF16 `AWQ_MATMUL` fails with `NoBackendForOp` unless explicitly lowered to its decompose first (GAP-367)
- `fuel-dispatch/tests/baracuda_awq_live.rs`: a refusal test for the shapes above, and a run-to-run determinism check (two launches on identical inputs must match byte for byte) behind the contract's `same_hardware_bitwise` claim

## v0.22.10 - 2026-10-23

### Removed

- `fuel-tensor`: the committed slice of `Orion-zhen/Qwen3-0.6B-AWQ` (added in 0.22.5 with the AWQ loader). That checkpoint is GPL-3.0, not Apache-2.0 as its provenance note said, so no part of it is kept in the repository (git history was rewritten to drop it). The two real-bytes tests now cut the same slice in memory from `$FUEL_QWEN3_AWQ_DIR/model.safetensors` and declare a `Hardware::Hub` skip when it is absent (fatal under `FUEL_REQUIRE_HUB`). It was never in a published crate: fuel-tensor 0.22.4 is the newest on crates.io

## v0.22.9 - 2026-10-23

### Changed

- Docs only: `docs/method-rules.md` gains three architect-approved entries with their `CLAUDE.md` pointers (`an-error-count-is-a-count-of-surfaces-not-causes`, `a-hook-you-route-around-on-a-guess-is-a-hook-you-have-not-read`, `a-name-search-is-structurally-blind-across-a-seam`); GAP-366 narrowed to the six drafts still awaiting the architect; ROADMAP frontier item 41 schedules the adoption of unpopped 0.22.0

## v0.22.8 - 2026-10-23

### Changed

- Docs only: `docs/gaps.md` rows GAP-360..366 and notes on GAP-287/302/303 for eleven stale open issues, and ROADMAP frontier item 40 (Gated DeltaNet)

## v0.22.7 - 2026-10-23

### Added

- `OpKind::AwqMatmul` with a CUDA F16 binding (`baracuda_dispatch::awq::awq_matmul_f16`, W4A16 AWQ GEMM) registered from the FKC contract `kernel-contracts/cuda/awq_matmul.fkc.md`; `AWQ_MATMUL` bridges to it in the runtime fused-kernel table (ROADMAP row 31a). Only F16 activations are bound: F32/BF16 AWQ is reachable only through an explicit lowering to the op's decompose (GAP-367)

## v0.22.6 - 2026-10-23

### Changed

- `fuel-metal-kernels`: `objc2*` dependencies are target-gated to macOS/iOS and the library is empty elsewhere, so `cargo publish` verification builds on non-Apple hosts

## v0.22.5 - 2026-10-23

### Added

- `fuel-tensor`: `awq` module — `load_awq_layer` / `awq_from_hf_buffers` / `AwqWeight::matmul` load a HuggingFace `*-AWQ` (GEMM) layer into the `AWQ_MATMUL` kernel layout (ROADMAP row 31b)

## v0.22.4 - 2026-10-23

### Added

- `fuel-inference`: `PagedSchedulePolicy {AdmissionOrder, Priority}` with `PagedSessionScheduler::set_policy` / `policy`. Under `Priority`, `step` and `step_batched` visit ready sessions (prefill and decode) highest `set_priority` first, ties in admission order; `step_batched` runs the group holding the highest-priority session first. The default stays `AdmissionOrder`, so existing callers are unchanged, and per-session outputs are identical under both policies.

## v0.22.3 - 2026-10-23

### Added

- `fuel-transformers`/`fuel-inference`: `Qwen3Model` and `QuantizedQwen3Model` implement `PagedDecodeModel` (single-step and ragged batched paged decode, validate-first). A correctness slice, not a speed one: there is no fused CUDA `PagedAttn` kernel, so paged attention runs on CPU with D2H copies. Dense configs only (a sliding-window config is a typed `Err`) and the plan-once entry re-plans every token: GAP-359.

## v0.22.2 - 2026-10-23

### Added

- `fuel-inference`: `PagedSessionScheduler::session_new_tokens(id)`, the per-session incremental-token accessor `SessionScheduler` already had, so a streaming serving loop can read each token the tick it is produced instead of only at `reap_finished`.

## v0.22.1 - 2026-10-23

### Added

- `fuel-transformers`/`fuel-inference`: `Llama3Model` and `QuantizedLlama3Model` implement `PagedDecodeModel`, with the llama3-scaled RoPE frequencies threaded through the paged forwards (`LlamaModel::forward_paged_step{,_persistent,_batched}_inv_freq`, `Tensor::rope_tables_const{,_batched}_with_inv_freq`). A correctness slice, not a speed one: there is no fused CUDA `PagedAttn` kernel, so paged attention runs on CPU with D2H copies.

## v0.22.0 - 2026-10-22

### Changed

- **Breaking** `fuel-graph`: `Tensor::awq_matmul` / `AWQ_MATMUL` now take the baracuda kernel layout instead of the HF checkpoint layout: weight `[N, K/8]`, zeros `[N, zeros_w]`, scales `[N, 8*zeros_w]` in F16, group size 64 or 128 (`awq_zeros_w` gives the padded width). HF checkpoints are repacked once on the host by `fuel_quantized::awq_repack::repack_hf_awq_for_kernel`. The op is still decompose-only; the CUDA registration is the next step of ROADMAP row 31.

## v0.21.1 - 2026-10-22

### Fixed

- `fuel-loaders` `derive_config`: an absent `{arch}.rope.freq_base` now defaults to 10000.0, as llama.cpp does (`llama-model.cpp` initialises `rope_freq_base_train = 10000.0f` and reads the key as optional). A present key keeps its value; a present key of the wrong type still declines. Unblocks GGUF files that omit the key (e.g. stories15M).

## v0.21.0 - 2026-10-22

### Changed

- **Breaking** `fuel-dispatch`: the default placement preference is now by accuracy claim, not by bit-stability (board 240, option D). `BitStablePreferenceFilter` is removed and replaced by `AccuracyPreferenceFilter` (`DEFAULT_ACCURACY_BOUND`, `AccuracyBound`, `meets_accuracy_bound`) in `default_chain`: cost ranking now compares, across devices, the kernels whose accuracy claim is within 1e-3 relative and absolute, with a kernel that has an earned `bit_stable` ledger pass counting as reference-grade and a kernel with no claim not winning by default. `PrecisionRequirement { require_bit_stable }` (strict reproducibility) is unchanged. Default-mode graphs that place a claimed non-bit-stable GPU kernel are no longer bit-identical run to run in the last bits.
- `fuel-dispatch`: the Vulkan f32 `MatMul`, `RmsNormLastDim`, `SoftmaxLastDim` and `Rope` contracts declare measured `max_relative` / `max_absolute` claims (`vulkan/matmul.fkc.md`, `vulkan/norm.fkc.md`, `vulkan/rope.fkc.md`; the f32 binding is split out of the fanned norm and rope sections so it can carry its own precision block). `fkc-verified-ledger.json` gains the 8 accuracy records and 58 refreshed Vulkan `bit_stable` passes (records earned at the current kernel revisions; none replaced). New `fuel-dev-tools seed-vulkan-accuracy` and `check-vulkan-accuracy`.
- Qwen3-0.6B decode on Vulkan: all 253 matmuls, and the norms, rope and softmax, now place on Vulkan (decode step 4,271 ms to 36 ms median; ROADMAP item 21).

## v0.20.71 - 2026-10-22

### Changed

- Docs only: ROADMAP row 37 (GGUF `head_count_kv` default, done in 0.20.69) and row 38 (CUDA clippy lint debt).

## v0.20.70 - 2026-10-22

### Changed

- Dependencies: every `baracuda-*` pin goes from `0.0.1-alpha.116` to `0.0.1-alpha.147` (the padded OC-major AWQ layout, baracuda#219; ROADMAP row 31). `unpopped-vocab` 0.20.1 -> 0.20.19 follows transitively.

## v0.20.69 - 2026-10-22

### Fixed

- `fuel-loaders`: `derive_config` reads an absent `{arch}.attention.head_count_kv` as `attention.head_count` (llama.cpp's own convention) instead of declining; a present key keeps its value and a present key of the wrong type still declines (ROADMAP row 37, from Lightbulb's census: stories15M q4_0/q8_0 failed only on the missing key).

## v0.20.68 - 2026-10-22

### Added

- `fuel-quantized`: `awq_repack` (`repack_hf_awq_for_kernel`, `awq_zeros_w`, `HF_AWQ_PACK_ORDER`): host repack of a HuggingFace `*-AWQ` layer into the padded OC-major buffers baracuda's AWQ kernel reads (ROADMAP row 31, loader step). Tested against baracuda's documented read formula; no CUDA needed.

## v0.20.67 - 2026-10-22

### Added

- `fuel-inference`: `PagedSessionScheduler::stats()` (`PagedSchedulerStats`), `session_infos()` (`PagedSessionInfo`) and `set_priority()` (ROADMAP row 33). Mechanism only: the scheduler reports phase counts, KV free blocks and bytes, and lifetime admit/reap/evict/restore/step totals, and stores a per-session priority it never orders by.

### Changed

- Docs: ROADMAP row 31 records baracuda#219 (padded OC-major AWQ layout; the Marlin repack was wrong alpha.64-alpha.145, fixed alpha.146).

## v0.20.66 - 2026-10-22

### Changed

- Docs only: ROADMAP row 36 (un-ignore `t5_gpu_batched_flash_equals_serial_bf16` on a self-hosted GPU runner).

## v0.20.64 - 2026-10-22

### Added

- `fuel-kv-pool`: `CacheStats` and `KvBlockPool::cache_stats()` count block-index lookups, block hits and misses, commits and evictions (ROADMAP row 32). Mechanism only: the pool decides nothing from them.

## v0.20.63 - 2026-10-22

### Changed

- Docs only: ROADMAP rows 32-35 record the gaps Lightbulb found removing Candle (prefix-cache stats, scheduler observability, bf16 CPU weights, and the undecided remainder).

## v0.20.62 - 2026-10-22

### Changed

- Docs only: README records what the 0.20.61 publish covered. All 40 publishable crates went to crates.io at 0.20.61 from `6dded17a`, read back from crates.io (40 present, 0 yanked). Held at 0.20.14: `fuel-examples`, `fuel-lazy-examples`, `fuel-test-support`; the fixture crates are `publish = false`. Holes: the publish ran with `DOCS_RS=1`, so the CUDA kernel (`nvcc`), `mkl` and `aocl` builds were not verified, and `fuel-metal-backend` was published `--no-verify` (`gemm-f16` does not cross-compile on Windows). ROADMAP row 31 (AWQ) records baracuda's scale-layout finding (baracuda#218).

## v0.20.61 - 2026-10-22

### Fixed

- `fuel-dispatch` could not be packaged from a plain source archive (no `.git`): cargo drops hidden files from a non-git file walk, so `kernel-contracts/.fkc-verified-ledger.json`, `include_str!`d by `fkc/verify/ledger.rs`, was missing from the tarball and the verify step failed. The ledger is now `kernel-contracts/fkc-verified-ledger.json`. The 0.20.60 `Packaged crates` job could not see this: it packages from a git checkout, where hidden tracked files are included.

### Added

- `scripts/verify-packaged-crates.py`: a `src/` include of a hidden file (any path component starting with `.`) is now a hard failure, with a selftest control.
- ROADMAP row 31: AWQ/Marlin loader -> op -> kernel gap.

## v0.20.60 - 2026-10-21

### Added

- CI job `Packaged crates` and `scripts/verify-packaged-crates.py`: every publishable crate is packaged (`cargo package --workspace --locked --no-verify`), each `.crate` is scanned for `include_str!`/`include_bytes!`/`include!`/`#[path]` targets outside its own tarball (a `src/` or `build.rs` escape is a hard failure with no allowlist; `tests/`/`benches/`/`examples/` and runtime `CARGO_MANIFEST_DIR/..` escapes live in the shrink-only `scripts/packaged-escapes-allowlist.txt`, 12 entries today, and a stale entry fails), then every `.crate` is extracted into a throwaway workspace patched to itself and `cargo check`ed. Known-bad controls run first (`selftest`). Named holes, printed by the job: the CUDA/forge compile is not verified (`DOCS_RS=1`), and the `WORKSPACE_EXCLUDES` crates are not built. This closes the gap that let `fuel-dispatch` 0.19.1/0.20.0/0.20.14 publish unbuildable (fixed in 0.20.59).

### Changed

- `fuel-dispatch` `fkc::validate` test: the FDX code table (`fuel-ir/src/dlpack/codes.rs`) is read at run time from the workspace instead of `include_str!`d from outside the crate. A tarball cannot run that test; it fails loudly rather than skipping.

## v0.20.59 - 2026-10-21

### Fixed

- `fuel-dispatch` did not build from its published crate: 86 `include_str!` sites in `src/` embedded `docs/kernel-contracts/**`, a path outside the crate that a packaged tarball does not contain. The contract corpus now lives at `fuel-dispatch/kernel-contracts/` (one copy; `docs/kernel-contracts/README.md` is a pointer), and every `include_str!`, the 7 runtime `CARGO_MANIFEST_DIR` reads, the source-scan tests and the live docs follow it. Still outside their crate and test-only (not a consumer build break, but `cargo test` from a tarball fails): `fuel-dispatch` `fkc::validate` reads `fuel-ir/src/dlpack/codes.rs`; `fuel-tensor` and `fuel-cpu-backend` read `fuel-dispatch/fixtures/kiss-corpus`; `fuel` reads sibling manifests. Tracked for follow-up PRs.

## v0.20.58 - 2026-10-21

### Security

- `zip` 8.6.0 -> 9.0.5 (RUSTSEC-2026-0367..0369, fixed in >=9.0.0); `zip::ZipArchive::file_names` now yields `Result`s, handled in `fuel-formats` `read_pth_tensor_info`.
- Lock-only: `rustls` 0.23.45 -> 0.23.46 (RUSTSEC-2026-0346..0349), `rustls-webpki` 0.103.15 -> 0.103.16, `webpki-roots` and `webpki-root-certs` -> 1.0.10 (RUSTSEC-2026-0381..0383); `cargo audit`: no vulnerabilities (3 unmaintained warnings: `paste`, `ttf-parser`, `unsafe-libyaml`).

## v0.20.57 - 2026-10-21

### Changed

- Vulkan: `Pipelines::new` (1,457 lines, cyclomatic complexity ~594) is now driven by one `pipelines!` table in `fuel-vulkan-backend/src/pipelines.rs`: a row is `pipeline_field, layout_field, SHADER, descriptor_layout, kind;` with kind `required`, `gated` or `coop`, built by `Build` methods, so adding a kernel is one row. The 176 `Pipelines` field names, types and drop order are unchanged (no call-site edits); the file shrank from 2,125 to about 550 lines. The `spirv_caps` source scan now counts the `gated`/`coop` rows (49, as before). ROADMAP row 30.

## v0.20.55 - 2026-10-21

### Changed

- Vulkan: a kernel whose shader needs an optional device feature the device does not report (`uniformAndStorageBuffer16BitAccess`, `shaderInt16`, `shaderBufferInt64Atomics`, `vulkanMemoryModel`, `vulkanMemoryModelDeviceScope`) is no longer created. `Pipelines::new` reads each module's `OpCapability` words (new `spirv_caps`, a `Gate` helper outside `Pipelines::new`), so no hand-kept table can drift, and the 49 affected pipeline fields are now `Option`; reaching one on a device that lacks the feature returns a typed `KernelUnavailable { kernel, missing_feature }` (via `Pipelines::pipeline`, listed in `Pipelines::unavailable`). Behaviour on the RTX 4070 and AMD 610M (which report all five) is unchanged. Tests inject each feature absent and assert the declined kernels, with a sabotage sibling that enables the feature and asserts the kernel is offered. Not done: placement does not yet avoid such a kernel before execution (the Vulkan kernel binding table is device-independent); see ROADMAP row 26.

## v0.20.54 - 2026-10-21

### Changed

- Vulkan buffer recycler: the 512 MB total cap is now enforced when a buffer is freed (`VulkanBuffer::drop`, destruction outside the pool lock), not only at the next allocation, so a mass free can no longer park more than the cap. ROADMAP row 28.

## v0.20.53 - 2026-10-21

### Changed

- Vulkan realize: `ContiguizeOf` work items are backend-neutral for chunking (their `Cpu` target is a placeholder, which made every use look like a Vulkan-to-Cpu switch and forced a batch submit); a Vulkan kernel item arms the async path itself, so a pure-Vulkan realize now uses deferred eviction (the evicted buffer is retained on the open batch via `VulkanBackend::retain_in_open_batch`, with no submit) instead of the blocking per-eviction flush, and submits the open batch once it holds 48 dispatches. Single-device Vulkan graphs therefore take the deferred path.
- Fixed a latent write-after-read in the Vulkan buffer recycler that larger batches expose: a buffer freed inside the open batch could be handed out again and written in the same batch while an earlier dispatch still read it. Parked buffers now carry the batch epoch they were freed in (bumped at every submission) and are reused only after a batch boundary.
- Qwen3-0.6B Q4_K_M decode on Vulkan (RTX 4070, per-device preference of the unmerged #426 applied locally), alternating runs in a quiet window: step 43-54 ms -> 30-36 ms. CPU-fixture check unchanged (max logit difference 9.2e-5, 0 of 64 tokens differ); Khronos synchronization validation reports 0 hazards. Not tested: CUDA+Vulkan mixed graphs, more than one Vulkan device.
- ROADMAP row 28: enforce the recycler cap at free time.

## v0.20.52 - 2026-10-21

### Changed

- New example `recycler_pool_probe` and the measured worst-case VRAM of the Vulkan buffer recycler after the depth change: with 64 distinct sizes (2 KiB to 24 MiB) freed at once the pool parks 1,260 MiB until the next allocation, then trims to 511 MiB (the unchanged 512 MB cap). Documentation and example only.

## v0.20.50 - 2026-10-21

### Changed

- Vulkan decode host time: the buffer recycler's per-bucket depth now scales with buffer size (as many as fit in 16 MiB, between 4 and 1,024; it was a flat 4) and trimmed buffers are destroyed outside the pool lock, so a decode step's hundreds of same-sized temporaries are recycled instead of destroyed and recreated; and `upload_host_buffer_to_device` stages directly on Vulkan instead of realizing a transient graph per buffer. New phase timers `decode.token_host_compute` and `decode.token_upload`. Qwen3-0.6B Q4_K_M decode on Vulkan (RTX 4070, per-device preference of the unmerged #426 applied locally), alternating runs: step 59-62 ms -> 43-45 ms. CPU-fixture check unchanged (max logit difference 9.2e-5, 0 of 64 tokens differ). The pool can hold more, smaller buffers within its existing 512 MB cap; VRAM held was not re-measured.

## v0.20.49 - 2026-10-21

### Changed

- ROADMAP rows 26 (kernels unavailable when a declared Vulkan feature is absent) and 27 (test plan for the 11 numerics-changing SPIR-V validation errors). Documentation only.

## v0.20.47 - 2026-10-21

### Fixed

- Vulkan device creation now probes and enables, only when the device reports them, the features the committed SPIR-V already declares: `uniformAndStorageBuffer16BitAccess`, `shaderBufferInt64Atomics`, `shaderInt16`, and on the cooperative-matrix path `vulkanMemoryModel` with `vulkanMemoryModelDeviceScope`. The Khronos validation layer reported 131 errors on a Qwen3 decode; it now reports 11 (one run). CPU-fixture check unchanged (max logit difference 9.2e-5, 0 of 64 tokens differ). A device lacking one is not given it (`VulkanBackend::declared_features_absent`, plus a warning); device creation does not fail. Remaining: shared memory above 32,768 bytes in three flash-attention shaders and f64 `Exp` in `softmax_f64` (numerics, reported). New `scripts/spirv_audit.py` (static per-shader capability audit) and `docs/research/vulkan-spirv-validation.md`.

## v0.20.46 - 2026-10-21

### Changed

- Vulkan batches are now ordered structurally: `Recorder::record_inter_batch_barrier` starts every batch with a compute/transfer memory barrier whose first synchronization scope covers all earlier submissions on the queue, and the pipelined executor no longer waits on the host for in-flight Vulkan batches before recording the next Vulkan op (the old "Invariant E", in both executor loops). The GPU now runs submitted batches while the host records the next. Measured on Qwen3-0.6B Q4_K_M decode on Vulkan (RTX 4070, 5 runs, per-device bit-stable preference of the not-yet-merged PR #426 applied locally): step 122 ms to 71 ms; with the column-parallel matvec of PR #430 stacked, 62 ms (about 16 tokens/s). CPU-fixture check unchanged (max logit difference 9.9e-5, 0 of 64 greedy tokens differ). Vulkan synchronization validation: 0 hazards with the barrier, 20 with it removed (control) on the real decode. New example `batch_order_probe`. Not tested: more than one queue or device, mixed CUDA and Vulkan graphs.
## v0.20.45 - 2026-10-21

### Changed

- `fuel-vulkan-backend`: new column-parallel f32 GEMV kernel `matvec_cols` (`matvec_cols.glsl`) for `m == 1` matmuls with `n % 4 == 0`, replacing the per-column-workgroup `matvec` for those shapes (the old kernel remains for other N and as `set_matvec_cols(false)`). The old kernel read the row-major weight matrix with neighbouring threads N floats apart; the new one has threads own columns, loads `vec4`s and reduces K in a fixed order. Measured: 1.9x to 2.9x on the Qwen3-0.6B layer shapes and 3.6x to 6.6x on the LM head (up to 218 GB/s, about 85% of nominal) at kernel level, within 2.5e-7 of an f64 reference; on the decode step matvec GPU time per step 82.6 ms to 15.2 ms and GPU time per step 86.5 ms to 20.9 ms; step 122 ms to 91 ms median on a local stack with #429 and #426 (CPU-fixture check: max logit difference 9.2e-5, 0 of 64 greedy tokens differ). Result bits differ from the old kernel (different K reduction order). Example `matvec_probe`. ROADMAP item 24.

## v0.20.44 - 2026-10-21

### Changed

- `fuel-dispatch`: `auto_contiguize` on Vulkan storage now materializes the layout with the existing `strided_copy_signed_*` kernels (`vulkan_dispatch::contiguize_vulkan`, 2/4/8-byte elements) instead of downloading to the host, walking it on the CPU and uploading, which also forced a flush and a wait on the open batch each time (123 round trips and about 150 ms of a Qwen3-0.6B decode step). The host round trip remains as the fallback for other element widths, empty tensors, offsets past 32 bits and storages without a backend handle. Measured on Qwen3-0.6B Q4_K_M decode on Vulkan (RTX 4070, 5 runs, with the not-yet-merged per-device bit-stable preference of PR #426 applied locally): step 253 ms to 122 ms (2.1x; 4.0 to 8.2 tokens/s); CPU-fixture check unchanged (max logit difference 9.9e-5, 0 of 64 greedy tokens differ). New example `contiguize_vulkan_check` compares the kernel path with the CPU reference byte for byte (21 of 21 layout/width cases identical).

### Added

- Executor host-phase names per work-item kind (`item.<Kind>`) and inside kernel items (`kernel.gather_inputs`, `kernel.alloc_output`, `kernel.execute_compiled`) for `op_profile::PhaseTimer`. ROADMAP item 23 updated.

## v0.20.43 - 2026-10-21

### Changed

- `fuel-dispatch` executor: work items that touch no device (`Op::Const` adoption, metadata-only views, slot views) no longer take part in the chunk-boundary backend-switch logic. A `Const` is always stamped `Cpu`, so every weight sitting between two Vulkan kernels looked like a Vulkan-to-Cpu-to-Vulkan switch: the executor armed `multi_backend`, submitted the open Vulkan batch, and waited for it before the next Vulkan op, about 400 submit-and-wait round trips per decode step. Applies to both executor loops. Measured on Qwen3-0.6B Q4_K_M decode on Vulkan (RTX 4070, 5 runs, with the not-yet-merged per-device bit-stable preference of PR #426 applied locally): step 343 ms to 251 ms (27% less; 2.9 to 4.0 tokens/s); verified against the CPU fixture (max logit difference 9.9e-5, 0 of 64 greedy tokens differ). Graphs that genuinely mix CUDA and Vulkan were not run.

### Added

- `fuel-dispatch`: `compiled::op_profile::PhaseTimer` and `take_phases` (opt-in named host phases; off by default, one relaxed atomic load each), wired into the pipelined executor, the realize bridge and the persistent-decode rebind. `qwen3_vulkan_profile` prints a host-phase table and gains `QWEN3_VERIFY=1` (the M0/M1 CPU-fixture comparison on Vulkan). ROADMAP item 23 updated.

## v0.20.42 - 2026-10-21

### Added

- `fuel-vulkan-backend`: opt-in GPU timing from Vulkan timestamp queries: `VulkanBackend::set_gpu_timing(bool)` (returns `false` if the compute queue family has no timestamp support), `gpu_time_snapshot()` (`(op, dispatches, ns)`, longest first, plus a `BATCH_TOTAL` row) and `reset_gpu_time()`. Off by default; when on, each batch records one timestamp at its start and one after every dispatch plus one query pool per batch. The per-dispatch figure is a serialised view (overlapping dispatches are charged to whichever retires last). Examples: `gpu_time_probe` (cross-check against the host clock) and a per-op GPU table in `qwen3_vulkan_profile`. Measured with them: of a 306 to 334 ms Qwen3-0.6B Q4_K_M decode step on Vulkan the GPU is busy about 86.5 ms and 95% of that is the M = 1 matmul (the measurement used the per-device bit-stable preference of the not-yet-merged PR #426). ROADMAP items 22 and 23 added. `Recorder::record_batch_dispatch` takes the op name.

## v0.20.41 - 2026-10-21

### Added

- `fuel-dispatch`: `compiled::op_profile::{set_trace, take_trace, LaunchRow}`, an opt-in ordered launch trace (op, backend, source and destination storage, bytes, host time). `fuel-vulkan-backend` example `copy_cost_probe` (bare Vulkan copy cost by size). `qwen3_vulkan_profile` now prints a one-step copy anatomy and the planner's transfer estimates. Measured with them (`docs/research/qwen3-vulkan-decode-profile.md`, "Copy anatomy"): a Qwen3-0.6B decode step on Vulkan reads 2.39 GB from the GPU to the host, mostly the model's own f32 weights (197 large copies, about 55% of the step), because the matmuls are stamped to the CPU; none of the 253 matmuls is Q4_K. ROADMAP item 21 updated. No behaviour change.

## v0.20.40 - 2026-10-21

### Added

- `fuel-dispatch`: `compiled::op_profile`, an opt-in (off by default, one relaxed atomic load per launch) per-`(OpKind, backend)` accounting of HOST wall time of kernel launches. It charges enqueue cost for kernels that only enqueue (Vulkan, CUDA), so it is not a device timeline. `fuel-transformers` gains a `vulkan` feature (forwarded by the `fuel` facade, as `fuel/tests/feature_forwarding.rs` requires) and the example `qwen3_vulkan_profile`. Measured with them: a Qwen3-0.6B Q4_K_M decode step on the Vulkan backend is 4,271 ms, 62% `Copy/Vulkan` and 26% CPU `MatMul`; attention is negligible at that context. ROADMAP items 20 (conditional, not worth starting on this evidence) and 21 (the copies and CPU matmuls) record it; details in `docs/research/qwen3-vulkan-decode-profile.md`.

## v0.20.38 - 2026-10-21

### Changed

- `fuel-vulkan-backend` / `fuel-dispatch`: bf16 and f16 split-KV decode attention (ROADMAP item 17, H12), completing the f32 path of 0.20.32/0.20.34. New shaders `flash_attn_{split,combine}_{bf16,f16}` and `VulkanBackend::flash_attn_{bf16,f16}_split_bytes`; the `FlashAttn` bf16/f16 dispatch uses the same `decode_splits` rule as f32 (`sq == 1`, `b*hq <= 16`, `2048 <= sk <= 4096`, 8 splits). Partials stay f32 and the output is narrowed once at the store. Results in that region are NOT bit-identical to the unsplit half kernels (a few outputs differ by one half-ulp; max abs 4.8e-7 bf16, 6.0e-8 f16 in the route check). `flash_attn_split_probe check|latency` takes a dtype; `flash_attn_route_check` covers all three. Per-call speedup measured (1.3-2.2x at `h_q <= 16`, `Sk >= 2048`, one run, includes readback). Qwen3 on Vulkan does NOT reach this route (its builders emit no `FlashAttn` node and the dispatch rejects capacity-K `k_len != sk`; see the research note), so no model-level speedup is claimed.

## v0.20.37 - 2026-10-21

### Added

- Docs only: ROADMAP item 19 records that CI never lints the Metal crates (`fuel-metal-kernels` 10 and `fuel-metal-backend` 2 clippy errors under `--target aarch64-apple-darwin`, measured locally) and the two steps to fix that. No row started.

## v0.20.36 - 2026-10-21

### Fixed

- `fuel-metal-kernels` tests: `cos_strided_random` compared Metal and CPU `cos` after rounding both to 4 decimals, on unseeded `rand::random` inputs, so a value near a rounding boundary failed it at random (macOS CI, 0.7316 vs 0.7317, on an unrelated Dependabot PR). It now draws from a seeded `StdRng` and compares with `fuel_test_support::assert_all_close` (abs 1e-5). The other unseeded draws in the file (the `reduce_*` shuffles, `const_fill`) are seeded too. Fixed-input comparisons against literal constants are unchanged. macOS behaviour is verified by CI only.

### Added

- `fuel-test-support`: `all_close` / `assert_all_close`, a `|got - want| <= abs + rel*|want|` comparison (NaN-safe, length-checked) with unit tests that run on every platform, including the rejection of an off-by-1e-3 value and the acceptance of the pair that flipped the rounded comparison.

## v0.20.35 - 2026-10-21

### Changed

- `fuel-examples` marian-mt: every pin in `examples/marian-mt/python/requirements.txt` moved to the latest PyPI release (certifi 2026.10.19 as in Dependabot #418, plus 20 more; urllib3 was already current). The old file was not installable as written (uv reported `transformers==5.13.1` incompatible with the other pins); the new one installs and `convert_slow_tokenizer.py` runs on Python 3.12.13 and writes working `tokenizer-marian-base-{fr,en}.json` (encode/decode round trip checked). Supersedes #418.

## v0.20.34 - 2026-10-20

### Changed

- `fuel-dispatch`: the f32 Vulkan `FlashAttn` kernel now takes the split-KV path (ROADMAP item 17, H12) for decode steps with `sq == 1`, `b*hq <= 16` and `2048 <= sk <= 4096` (8 splits; heuristic `decode_splits`, read off the measured table, one device). Results in that region differ from the unsplit kernel in the last bits (deterministic; max abs 5.6e-9 against it in the route check). bf16/f16 attention is unchanged. Adds the live-GPU example `fuel-dispatch/examples/flash_attn_route_check.rs`. Model-level speed is NOT measured. Not bit-identical to the unsplit path (4.6e-9 vs an f64 reference), so exact-output comparisons across versions will see a change; the routed shape is rarely hit by real (bf16/f16) decode today, so no end-to-end claim.

## v0.20.33 - 2026-10-20

### Fixed

- Docs only: corrects the H24 n-gram drafter proxy note (`docs/research/ngram-draft-probe.md`), ROADMAP item 17 and the v0.20.30 entry below. The note said the 1.3-accepted-tokens-per-step hypothesis was "met by a wide margin" by comparing 1.3 to tokens per step, which includes the target's own token. Accepted per step is one lower: prose edits 1.7-6.3, code edits 1.1-2.4 (code edit-10% and edit-30% at draft 3 are 1.24 and 1.08, under 1.3), unrelated control 0.2-0.4. Still a ground-truth proxy upper bound, not a real-model result.

## v0.20.32 - 2026-10-20

### Added

- `fuel-vulkan-backend`: `VulkanBackend::flash_attn_f32_split_bytes`, a split-KV decode attention path (ROADMAP item 17, H12). The key range is cut into `splits` chunks that run as separate workgroups (`flash_attn_split_f32`), then `flash_attn_combine_f32` merges the partials in fixed order. Not wired into any route: nothing selects it yet. Results differ from the unsplit kernel in the last bits (the merge reorders float adds); both match an f64 CPU reference to about 6e-8 on the tested cases. Measured on an RTX 4070 Laptop GPU: per-call latency at `h_q = 8`, `Sk = 4096` fell from about 1.03-1.10 ms to 0.55-0.65 ms (1.7-1.9x, readback included); 1.2-1.25x at 32 and 64 heads; no measured cell got slower. End-to-end decode speed was not measured. See `docs/research/split-kv-decode-measurement.md`.

## v0.20.31 - 2026-10-20

### Added

- `fuel-kv-pool`: a content-addressed block index (ROADMAP item 17, H20). `commit_block` registers a FULL block under a caller-computed 128-bit key (`BlockHash`); `lookup_chain` / `splice_cached_chain` find and share the longest cached run of a request's hash chain at any block boundary; `evictable_cached` lists blocks only the index references, least recently used first; `release_cached` drops the index's reference; `chain_block_hash(parent, tokens, extra)` is a convenience (non-cryptographic) chained hash. New `KvAllocError::{BlockNotFull, UnknownCachedBlock}` variants. Mechanism only: what goes in a key, when to commit and what to release are the caller's. Measured on a synthetic multi-turn trace in `docs/research/block-chain-index-measurement.md`: 86.2% of prompt tokens served with no boundary annotations, against 0% for whole-prefix keys stored at request end and a tie with annotated whole-prefix keys.

## v0.20.30 - 2026-10-20

### Added

- `fuel-inference`: `ngram_draft::NgramDrafter` (longest-recent-n-gram "prompt lookup" drafting, no second model) and `one_hot_logprobs` (the point-mass draft distribution the existing `speculative::verify_draft` needs). Policy-layer, additive, not yet wired into a decode loop. A proxy measurement (ground-truth target, so an upper bound; see the v0.20.33 correction of its hypothesis reading) is in `docs/research/ngram-draft-probe.md` with the `fuel-examples` `ngram_draft_probe` example that produced it (adds `fuel-inference` as a dependency of `fuel-examples`). ROADMAP item 17, H24.

## v0.20.29 - 2026-10-20

### Added

- Docs only: ROADMAP item 18 records that Judge profiles on a machine whose discrete GPU swings between a slow and a fast mode (2-3x, unrelated to measurement order) are not comparable run to run, with the evidence and four candidate fixes (none started).

## v0.20.27 - 2026-10-20

### Fixed

- `fuel-graph`: `NF4_MATMUL`, `AWQ_MATMUL` and `CAUSAL_CONV1D` are now `BackwardKind::Decompose` (they were `NotDifferentiable`, so any `backward()` through them panicked with "no backward arm wired yet"). Autograd lowers them through their total `decompose`; the quantized weights stay constants, so the activation gradient (what QLoRA-style fine-tuning and Mamba training need) is defined. Finite-difference tests in `fuel-tensor` were red on `main` and are green (NF4 and AWQ activation gradient; causal conv1d input, weight and bias gradients, with and without SiLU). PM ruling 2026-10-20; the registry docs that recorded `NotDifferentiable` now say what changed and why. `QMATMUL` and `PAGED_ATTN` are unchanged.

## v0.20.26 - 2026-10-20

### Fixed

- `fuel-graph`: `NodeHandle::backward` now lowers EVERY fused op whose registry entry declares `BackwardKind::Decompose` (previously a hard-coded pair of SSM ids). `FusedSoftmaxCrossEntropy` declared `Decompose` but panicked with "no backward arm wired yet" on any `backward()`, including through `fuel_training::loss::fused_softmax_cross_entropy`; it now differentiates (checked against a finite difference). A pinned-set test makes adding or flipping a `Decompose` op deliberate. ROADMAP item 17, H11.

## v0.20.25 - 2026-10-20

### Fixed

- Docs: withdrew the claim in `docs/research/judge-measurement-notes.md` (0.20.24) that measuring the CPU class before the Vulkan classes biases the Judge. Seven interleaved runs show no order effect; the discrete GPU's readings are instead bimodal from run to run (2-3x), unrelated to order. The notes, ROADMAP item 17 H15 text and the 0.20.24 entry were rewritten.

## v0.20.24 - 2026-10-20

### Added

- Docs/tooling only: `docs/research/judge-measurement-notes.md` records the Judge's own noise floor (84-87% winner agreement between identical runs), that the proposed early-stop does not cut wall time (timed iterations are ~12% of it); `scripts/judge_compare.py` compares two Judge reports. No library code change.

## v0.20.23 - 2026-10-20

### Added

- `fuel-kv-pool`: `KvBudget`, `KvGeometry::fit_to_budget` / `block_bytes`, `non_kv_resident_bytes` and `KvSizingError` — size a pool to the most whole blocks a device budget allows instead of guessing `num_blocks` (ROADMAP item 17, H19). `fuel-tensor`: `kv_block_pool_device::fit_geometry_from_residency` feeds it from a `ResidencyPlanner` report of the decode graph built with a small probe pool. Output bytes only: backend workspace is not counted (H8), so keep `utilization` below 1.

## v0.20.22 - 2026-10-20

### Fixed

- `fuel-tensor` NF4 loader (`load_nf4_layer`): **real bitsandbytes checkpoints now load correctly.** (1) Double-quantized NF4 (`bnb_4bit_use_double_quant=True`; U8 `absmax` plus `nested_absmax`, `nested_quant_map`, `nested_blocksize`, `nested_offset`) is decoded at load by the new `dequantize_nested_absmax` instead of being refused. (2) bitsandbytes packs the even-k code in the UPPER nibble while Fuel's fused NF4 op reads the LOWER one; the loader now swaps nibbles, so single-level NF4 checkpoints, which previously dequantized to garbage (relative RMS error 1.42 against the original weights), now land at 0.09. `nf4_from_bytes` still takes bytes in Fuel's order. `BnbQuantState` gains a `nested` field. FP4 remains refused (it needs its own codebook). Verified against a 16-row real excerpt of `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` checked against the original `Qwen/Qwen3-0.6B` weights; see `fuel-tensor/tests/fixtures/bnb_double_quant/PROVENANCE.md`.

## v0.20.21 - 2026-10-20

### Fixed

- `fuel-graph`: backward through `Op::GeluTanh` and `Op::GeluTanhInplace` no longer panics ("inference-only"); the gradient `0.5(1+t) + 0.5x(1-t²)·c(1+3kx²)` is built from existing primitives and checked against finite differences in `fuel-tensor` (with a retained sabotage sibling). Found by the 2026-10-19 library harvest (ROADMAP item 17).

## v0.20.20 - 2026-10-19

### Changed

- Docs only: ROADMAP items 16 (Strata-derived experiments) and 17 (ideas harvested from 23 ML libraries), none started, with evidence in `docs/research/strata-ideas.md` and `docs/research/ml-library-harvest.md`. No code change.

## v0.20.19 - 2026-10-19

### Changed

- CI: new `cuda-metal-compile-check.yml` (self-hosted Windows runner, push-to-main + workflow_dispatch only, no secrets) that `cargo check --all-targets` the cuda/jit configs hosted runners cannot build. Metal and Accelerate remain unverified there.

## v0.20.18 - 2026-10-19

### Changed

- Dead-code retirement: `allow(dead_code)` removed from fuel-vulkan-backend (`SubmittedBatch` lifetime anchors are now `_`-named), fuel-model-llama (decode helpers gate on `cuda`/test) and the musicgen, yolo-v8 and dp_diff examples (unused `KP_CONNECTIONS` and `print_breakdown` deleted). New `fuel-ir/tests/suppression_ratchet.rs` fails on any new `allow(dead_code)` not in its reasoned list and holds `#[ignore]` to a ceiling of 500.

## v0.20.17 - 2026-10-18

### Changed

- Dead-code retirement (cpu-backend, tensor, ir, graph, transformers): 17 `allow(dead_code)` removed. Deleted unused code: `Device::{rand_uniform,rand_normal,zeros}*` (crate-private, superseded by GAP-023), `FusedOpRegistry::by_pattern_hash`, the cpu-backend `conv2d_direct`/`Conv2dImpl` selector (behaviour unchanged: tiled im2col was already the only reachable default), `f64_to_f32`/`f32_to_f64`. `fuel-cpu-backend/src/mkl.rs` module-wide allow removed. `AttnRecompute` keeps one documented allow (fields read only by tests).

## v0.20.16 - 2026-10-18

### Changed

- fuel-dispatch: all 41 `allow(dead_code)` retired. `jit_ingest`/`jit_ingest_probe` and their `fkc::verify` re-exports now compile only under `cuda` or `jit`+test; removed never-read `DiffOutcome.max_ulp` and `StreamingPick.lookup` (public `realize_with_optimized_picking*` signatures unchanged); `kiss_corpus` is test-only. No behaviour change.

## v0.20.15 - 2026-10-18

### Changed

- GGUF (GAP-211 C): new `GgmlDType::Tq1_0`/`Tq2_0`/`Mxfp4`/`Nvfp4` (ggml codes 34, 35, 39, 40) with CPU dequantization (`fuel_quantized::tq_mx_types`, MIT notice in `THIRD-PARTY-NOTICES.md`). Dequantization compared bit-for-bit with llama.cpp compiled C on random blocks. DECODE-ONLY like the grid-table IQ types: quantizing to them is a typed error, CUDA/Metal matmul decline them. New FDX_GGML_* codes mirror the ggml numbers. Codes 41 and 42 (Q1_0, Q2_0) stay unsupported and are still reported by name.

## v0.20.14 - 2026-10-18

### Changed

- GGUF (GAP-211 B2): new `GgmlDType::Iq2Xxs`/`Iq2Xs`/`Iq2S`/`Iq3Xxs`/`Iq3S`/`Iq1S`/`Iq1M` (ggml codes 16, 17, 22, 18, 21, 19, 29) with CPU dequantization (`fuel_quantized::iq_grid_types`; lookup tables generated from llama.cpp `ggml-common.h` @9d57ce456c by `scripts/gen_iq_grids.py`, MIT notice in `THIRD-PARTY-NOTICES.md`). Dequantization compared bit-for-bit with llama.cpp compiled C on random blocks. DECODE-ONLY: quantizing to them is a typed error (`GgmlDType::check_quantize`); CUDA/Metal matmul decline them. New FDX_GGML_* codes mirror the ggml numbers.

## v0.20.13 - 2026-10-17

### Changed

- GGUF (GAP-211 B1): new `GgmlDType::Iq4Nl` (ggml code 20) and `Iq4Xs` (23) with CPU dequantization (`fuel_quantized::iq4`, verified against llama.cpp `ggml-common.h`/`ggml-quants.c` at 9d57ce456c; MIT notice in the module), so IQ4_NL/IQ4_XS GGUFs load. FDX gains `FDX_GGML_IQ4_NL`=20 and `FDX_GGML_IQ4_XS`=23. The quantizer is a simple nearest-codebook one, not llama.cpp's scale search. CUDA matmul (MMVQ) and Metal matmul decline these types; other IQ types are still reported by name.

## v0.20.12 - 2026-10-17

### Changed

- GGUF (GAP-211, structural half): an unknown ggml type code (every IQ*/TQ* code, F64, the integer codes) no longer aborts `Content::read`/`Content::open`; the tensor is recorded in the new `Content::unsupported_tensors` (name -> raw code) and loading it fails with an error naming the tensor and the code. IQ dequantization is still open.

## v0.20.11 - 2026-10-17

### Changed

- Test infrastructure: the Qwen3 yardstick fixture regenerator and speed harness stop being `#[ignore]`d tests and become `examples/qwen3_yardstick_tool.rs` (new `fuel-transformers` feature `dev-tools`; shared machinery in `fuel_transformers::dev_tools::qwen3_yardstick`, which the yardstick tests now import). #[ignore] retirement step 2b.

## v0.20.10 - 2026-10-17

### Changed

- Lint policy: all 256 per-item `too_many_arguments` attributes removed (242 `#[allow]`, 14 `#[expect]`); `clippy.toml` threshold 10 -> 22, the measured kernel-dispatch maximum (compiler max 20 in CI scope; Metal `call_sdpa_full` 22 hand-counted), with the method and numbers in the file. ROADMAP item 14 row (e). PM ruling 2026-10-17.

## v0.20.8 - 2026-10-17

### Changed

- CI: every `cargo check`/`cargo test`/`cargo clippy` step in `rust-ci.yml` now passes `--locked` (24 invocations; SUP-001), so a build can no longer resolve dependencies differently from the tracked `Cargo.lock`. The two `cargo metadata --no-deps` calls are left as they are (no resolution to lock).

## v0.20.7 - 2026-10-16

### Changed

- Dependencies: tokio 1.53.2 -> 1.53.3 (lock only).

## v0.20.6 - 2026-10-16

### Changed

- Dependencies: adopt the baracuda family at 0.0.1-alpha.116 (latest published; all 18 workspace pins, 21 locked crates; transitive unpopped-vocab 0.14.4 -> 0.20.1). `baracuda-cuda-emit` stays on its own 0.11 line. Verified: `cargo check --all-targets` of fuel-dispatch (cuda, jit) + fuel-cuda-backend, 0 errors; live GPU suite NOT re-run.

## v0.20.5 - 2026-10-16

### Changed

- Docs: ROADMAP item 14 (#[ignore] and #[allow(dead_code)] retirement, self-hosted GPU runner).

## v0.20.4 - 2026-10-16

### Changed

- Test infrastructure: new `Hub` hardware family (`FUEL_REQUIRE_HUB`, Permissive by default) with `fuel_test_support::hub`; the three timm `from_hub_smoke_*` tests and `fuel-datasets` `test_dataset` run when the checkpoint/network exists and otherwise declare a loud `SKIP[hub]`; seven empty `#[ignore]`d `from_hub_smoke_*` tests removed (#[ignore] retirement step 2).

## v0.20.3 - 2026-10-16

### Changed

- fuel-dispatch: the ledger seeders (cpu/cuda/vulkan) and the two GAP-228 contract tools move from #[ignore]d tests to a fuel-dev-tools binary behind the non-default, #[doc(hidden)], semver-exempt dev-tools feature (CI now builds it). Removed the ignored gap_hedges regenerate_allowlist: it erased the 16 hand-written reason columns of the committed allowlist, and the gap-hedge gate already fails on both new and stale keys. #[ignore] retirement, step 3.

## v0.20.2 - 2026-10-16

### Changed

- The six gap315 measurement probes move from ignored tests to examples (cargo run -p <crate> --example gap315_*), and their outcome enums drop six allow(dead_code). #[ignore] retirement, step 4.

## v0.20.1 - 2026-10-16

### Changed

- fuel-dispatch: candidate admission verifies on every admission probe (seeded fill + NaN/inf special values), GAP-236 increment 2; identical NaN/inf now agree under MaxAbsolute/MaxRelative in verify_precision_bound.

## v0.20.0 - 2026-10-16

### Changed

- feat(fuel-dispatch)!: structure_key deriver sk5 -> sk6 (KISS #588): GemCell gains scale_dtype (trailing /q<sdt>), sk5_dtype_token renamed sk6_dtype_token (Breaking)

## v0.19.12 - 2026-10-15

### Changed

- deps: rand 0.10 and rand_distr 0.6; bf16/f16/F8E4M3 random fills sample f32 then convert (half/float8 implement SampleUniform for rand 0.9 only).

## v0.19.11 - 2026-10-15

### Changed

- ci: actions/checkout v7.0.1 and actions/setup-python v7.0.0 (pinned SHAs); repoint 45 stale `fuel-core::<module>` comment paths to `fuel-tensor` (comment-only).

## v0.19.10 - 2026-10-15

### Changed

- docs: ROADMAP 13(c) extensible cost dimensions recorded as deferred (PM ruling).

## v0.19.9 - 2026-10-15

### Changed

- fuel-tensor: strict reproducibility mode (`fuel_tensor::reproducibility`); the realize path now threads a bit-stable floor into the planner when enabled (ROADMAP 13(b)).

## v0.19.8 - 2026-10-15

### Changed

- coderipper triage: `fuel-inference` tests assert discarded found-flags; deliberate `StepReport` discard named.

## v0.19.7 - 2026-10-15

### Changed

- Re-vendored KISS `structure_key_derivation_vectors.json` (36 vectors, blob 697aa543); Fuel's sk5 deriver passes all, incl. `[8,0]`.

## v0.19.6 - 2026-10-15

### Changed

- GAP-326 closed (PR 3/3): decoder rank-3 guards routed through `dims3()`, `--gap326-gate` runs in CI, registry row deleted.

## v0.19.5 - 2026-10-15

### Changed

- GAP-326 PR 2/3: vision rank-4 and channel input guards routed through `input_guard::image_nchw`.

## v0.19.4 - 2026-10-14

### Changed

- Workspace dev-dependencies are path-only; CI check for versioned ones (publish ordering).

## v0.19.3 - 2026-10-14

### Changed

- CI: `cargo metadata --locked` step fails a stale tracked `Cargo.lock`.

## v0.19.2 - 2026-10-14

### Changed

- KISS `ops-int-wrap` (440 integer cells) conformance through the CPU binary op, GAP-356 filed; tracked `Cargo.lock` regenerated (was 0.18.7 + fuel-core).

## v0.19.1 - 2026-10-14

### Changed

- CI/hygiene (CodeRipper triage SUP-008, WSP-001, DOC-009): every GitHub Action pinned to a full commit SHA at the ref the workflows already named (checkout/setup-python v6, trufflehog main), top-level `permissions: contents: read`, `.github/CODEOWNERS`, `GLOSSARY.md`. No behaviour change.

## v0.19.0 - 2026-10-14

### Changed

- BREAKING: `fuel-core` is dissolved and deleted (GAP-348). The `fuel` facade now re-exports the former `fuel-core` surface by explicit name from the crates that own each item (`fuel-tensor`, `fuel-ir`, `fuel-hardware`, `fuel-dispatch`, `fuel-backend-contract`, `fuel-cpu-backend`), and forwards the ten backend features to `fuel-tensor` 1:1. A dependent that named `fuel-core` must name `fuel` (or the owning crate); `fuel::` paths are unchanged. The 13 remaining gated integration tests moved to `fuel-tensor/tests/` earlier (0.18.15); the six unreferenced torch/numpy fixture files under `fuel-core/tests/` are deleted. CI: the vulkan-gated compile step now names `fuel-tensor`. docs: architecture 02-layers status, decisions-log entry, README install snippet.

## v0.18.15 - 2026-10-14

### Changed

- Tests moved (GAP-348 c, batch 2): the 13 feature-gated integration tests under `fuel-core/tests/` (CUDA, Vulkan, aocl, mkl/onemkl) now live in `fuel-tensor/tests/` with `fuel_core::` rewritten to the real crates. Run on an RTX 4070 before the move was rebased: 7 CUDA files 19 passed, vulkan and cuda+vulkan files and the overlap bench passed; the 3 aocl/onemkl files were compile-checked only. GAP-358 filed (fuel-transformers model methods over Codacy complexity limits).

## v0.18.14 - 2026-10-14

### Changed

- Repointed `fuel_core::` to fuel-ir / fuel-tensor / fuel-loaders in fuel-transformers (GAP-348 step 3); manifest no longer depends on fuel-core.

## v0.18.13 - 2026-10-14

### Changed

- Repointed `fuel_core::` to fuel-ir / fuel-tensor in fuel-nn (GAP-348 step 3); registry: GAP-357 filed (phi loader complexity), GAP-305 ownership and KISS-side facts recorded. GAP-355 closed and its row deleted (a struck row cannot carry an owner and the owner-less ceiling only goes down): the quiet-NaN fix for the remaining direct-libm ops landed in fuel#360, and the macOS CI legs (`Check` and `Test Suite`, macOS-latest) were green on fuel#360 and on fuel#361 head 9659d4a0, the instrument the row named.

## v0.18.12 - 2026-10-14

### Changed

- Repointed `fuel_core::` to the real crates in fuel-model-llama, fuel-model-phi, fuel-datasets and fuel-posttrain (GAP-348 step 3, batch A); their manifests now depend on fuel-tensor / fuel-ir instead of fuel-core.

## v0.18.11 - 2026-10-14

### Changed

- ops-transcendental-nan re-vendored at KISS ff9f52f5 (13 vectors, +tcId 13 f64 log of a signalling NaN); x86: 199 vectors, 163 pass, 0 fail, 36 not exercised. No library change.

## v0.18.10 - 2026-10-14

### Changed

- CPU backend quiets a signalling NaN operand for Erf/Floor/Ceil/Round/Sqrt/Pow (and the dyn_impl paths), as KISS-OPS-6.16 requires; GAP-355.

## v0.18.9 - 2026-10-14

### Changed

- sk5 deriver now declines (typed `DeriveDecline`) a collapsed reduction output and an out-of-range reduced axis, as KISS requires; derivation vectors 28 pass + 5 not-expressible (canonicalized 29 + 4). GAP-354.

## v0.18.8 - 2026-10-14

### Changed

- KISS derivation vectors (33, re-vendored at KISS ae16d2a7) run through the sk5 deriver in fuel-dispatch tests; GAP-354 filed. No library change.

## v0.18.7 - 2026-10-14

### Changed

- Tests moved (GAP-348 c, batch 1): the 16 CPU-only, un-gated integration tests under `fuel-core/tests/` now live in `fuel-tensor/tests/`
  with `fuel_core::` rewritten to the real crates; `fuel-core` drops its now-unused `fuel-conv` dev-dependency. 47 passed + 2 ignored in the
  new location, the same counts as before. No library change.

## v0.18.6 - 2026-10-14

### Changed

- Docs only: ROADMAP item 6 and the contraction-tuple note name KISS's `MathFidelity` (not `MathPrecision`); the GAP-349 row
  records what KISS-CLASSIFY-6.1-0001/-0013 say about reserved `f8e6m2`; `docs/outreach/kiss-conformance-and-divergences.md` gets a
  dated staleness banner; registry rows GAP-352 (8-bit float NaN behaviour unverified) and GAP-353 (no `copysign` op) are filed.

## v0.18.5 - 2026-10-14

### Fixed

- CPU `exp`/`log`/`sin`/`cos`/`tanh` (`fuel-cpu-backend` chassis and the `dyn_impl` path) now return a QUIET NaN for a
  signalling-NaN operand. They called `libm` directly, and Apple's `logf` returns `log(sNaN)` as the signalling
  `7F800001` unchanged (found by the new KISS NaN evaluator on the macOS CI leg; x86 libm already quieted). A
  NaN result has its quiet bit set; payload and every non-NaN value are untouched.

### Changed

- Tests, fixtures and docs: the KISS NaN-operand / minmax-NaN / rounding vectors (`ops-narrow-move-nan`, `ops-narrow-select-nan`,
  `ops-transcendental-nan`, `ops-minmax-nan`, `ops-rounding`, KISS `0af5d3b6`) are run through the lazy `Tensor` on CPU by
  `fuel-tensor/tests/kiss_nan_conformance.rs` (198 vectors: 162 pass, 0 fail, 36 not exercised: 8-bit floats, `copysign`).
  The f32/f64 prop-vs-ieee tripwire is replaced by a positive assertion; a `kiss_clause_ledger` row is added for
  KISS-CONFORM-6.8-0010; the `trunc` doc no longer says no KISS vector checks it.

## v0.18.4 - 2026-10-14

### Changed

- Lint-only: `cargo clippy -p fuel-dispatch --features jit --all-targets -- -D warnings` is green again. 17 items in
  `jit_ingest.rs`/`jit_ingest_probe.rs` that are dead under `--features jit` without `cuda` take the module's existing
  `#[cfg_attr(not(feature = "cuda"), allow(dead_code))]` pattern, and the minmax cell map in `kiss_corpus` tests gets a
  type alias for `clippy::type_complexity`. No behaviour change.

## v0.18.3 - 2026-10-14

### Changed

- Vendors the KISS NaN-operand conformance vectors (`ops-narrow-move-nan`, `ops-narrow-select-nan`,
  `ops-transcendental-nan`) and the 122-op `op_manifest.json` at KISS `5b28bf51`; `kiss_corpus` loads them and its
  prop-vs-ieee minmax tripwire is replaced (8 discriminating cells, narrow dtypes only; f32/f64 still blind). Tests and
  fixtures only, no runtime change.

## v0.18.2 - 2026-10-14

### Changed

- Manifest fixes so `cargo publish --workspace` can package: `fuel-kiss-ref-backend` gains license/repository and
  versioned path deps (`fuel-dispatch`'s optional dep on it too); `fuel-training` and `fuel-inference` gain the README.md
  their manifests already named. No code change.

## v0.18.1 - 2026-10-14

### Changed

- `extern crate intel_mkl_src` / `accelerate_src` (the `mkl`/`accelerate` link glue) move from `fuel-core` to
  `fuel-cpu-backend`, beside the `extern "C"` FFI they serve; `fuel-core` drops its two optional `-src` deps and its
  features still forward. No public API change. GAP-348 (b); the GAP-348 row and ROADMAP item 2 now carry the plan.

## v0.18.0 - 2026-10-13

### Changed

- **Breaking:** the `fuel-compression` crate is renamed `fuel-posttrain` (directory, package name, workspace
  entry). The old name is owned on crates.io by an unrelated project, so it could never publish. CireSnave
  ruled "go with fuel-posttrain for now" (2026-10-04). Code is unchanged; the crate was never published and no
  other workspace crate depended on it. Version allocated by the PM (0.18.0, pre-1.0 breaking = second number).

## v0.17.3 - 2026-10-12

### Documentation

- Docs only, no code change: the remaining stale `U8` mask wording left after GAP-168(c) made the
  comparison output, the Where condition and the MaskedFill mask `Bool` is corrected in
  `cuda/indexing.fkc.md`, `dispatch/elementwise-binary.fkc.md` (comparison family and Where) and
  `cpu/shape-ops.fkc.md` (masked_fill). Prose, blurbs and a binding-key note only; each block's
  machine-readable `dtype_rule`/`dtypes` already said `Bool`, and `blurb` is not an input to
  `kernel_revision_hash`.

## v0.17.2 - 2026-10-11

### Documentation

- Docs only, no code change: `docs/gaps.md` GAP-351 records why dependabot #335 (`rand` 0.9 -> 0.10)
  cannot merge. `rand_distr` is left at 0.5.1 (0.6.0 exists), and `half` 2.7.1 / `float8` 0.7.0 implement
  sampling only against `rand` 0.9 / `rand_distr` 0.5, so `bf16`/`f16`/`F8E4M3` are not samplable under 0.10.

## v0.17.1 - 2026-10-11

### Fixed

- Tests only, no kernel change: the three Vulkan `masked_fill` live tests and the CUDA
  indexing registration test keyed a `U8` mask after GAP-168(c) made the MaskedFill mask
  `Bool`, so they looked up a binding that does not exist and read as a missing kernel
  registration. Stale `U8` mask prose in the Vulkan select and CUDA indexing contracts
  corrected.

## v0.17.0 - 2026-10-11

### Breaking

- `fuel-quantized`: `GgmlType` gains a `bytemuck::Pod` supertrait, so a downstream
  `impl GgmlType` for a non-`Pod` type no longer compiles (all 15 in-tree implementors are
  `Pod`). The 12 `Block*` structs derive `Pod` + `Zeroable` and gain `Copy`.

### Added

- `fuel_quantized::{blocks_as_bytes, blocks_from_bytes_cow}`: safe block/byte views.

### Fixed

- `fuel-vulkan-backend`: `upload_params` / `as_bytes` no longer expose uninitialized padding
  bytes of a param struct (`T: NoUninit`; the 90 param structs derive it, compiler-verified
  padding-free); `download_slice` is `T: Pod`.
- `fuel-model-phi`: the k-quant dequant cast no longer assumes a tensor sub-slice of the mmap
  is block-aligned.

### Changed

- Hand-written `unsafe` byte casts in `fuel-quantized`, `fuel-cpu-backend`, `fuel-cuda-backend`,
  `fuel-nn` and `fuel-transformers` are checked `bytemuck` / `blocks_*` calls.

## v0.16.1 - 2026-10-11

### Fixed

- `fuel-tensor` / `fuel-core`: the `cudnn` cfg compiles again. Removed two re-exports of
  `cuda_backend::cudnn` (retired 2026-05-25) and five `?` in `()`-returning `cuda_executor_*`
  tests. No public API change in any default build; no CI job compiles this cfg (#343).

## v0.16.0 - 2026-10-11

### Breaking

- `fuel-dispatch` (optional `telemetry` feature): the independent `structure_key` deriver
  (`telemetry::structure_key_derive`) now derives KISS **sk5**. `FuelOpCategory` and
  `derive_structure_key_token_with_acc_mp` are removed, and `derive_structure_key_token` takes
  `(&DeclaredFacts, &[CellOperand], target, Option<AccMp>)`. The cell class is derived from
  declared facts (M/N/K roles, index slot, reduced set), not chosen by the caller. New:
  `CellClass`, `DeclaredFacts`, `CellOperand`, `OperandRole`, `DequantForm`, `derive_cell_class`,
  `sk5_dtype_token`.

### Changed

- deps: `kiss-ref-core` / `kiss-ops-vocab` / `kiss-classify-vocab` 0.3 -> 0.4.
- `f8e6m2` is declined by the deriver in every keyed dtype position (reserved at sk5, KISS #517);
  `fuel-ir`'s dtype table is unchanged (GAP-349).
- Vendored KISS corpus: `structure_key_vectors.json` + `dtype_manifest.json` at KISS `5b28bf51`
  (per-file pins in `fuel-dispatch/fixtures/kiss-corpus/PROVENANCE.md`).

## v0.15.7 - 2026-10-11

### Changed

- deps: full `cargo update` lock refresh: baracuda-* alpha.86 (alpha.81 to alpha.86 across the
  family), 126 lock entries updated and 11 added transitively (sha2 0.11 / digest 0.11 stack).
  Lock-only; no manifest dependency or source change. `baracuda-cuda-emit` stays on 0.11.x
  (0.14.3 is outside the workspace `^0.11` requirement).

## v0.15.6 - 2026-10-11

### Changed

- deps: tokio 1.53.2 and baracuda-cudnn-sys / baracuda-cudnn / baracuda-curand-sys alpha.84
  (lock-only; supersedes #336 to #339). No manifest dependency or source change.

## v0.15.5 - 2026-10-09

### Fixed

- docs(gaps): close GAP-048 (KISS-Ops §6.15 fmax_ieee/fmin_ieee/rem_trunc) and
  GAP-300 (`trunc` floor-op conformance) — both rows read OPEN/NOT FIXED but
  the fix had already landed 2026-09-06 in #137 (`80def2467`); the rows were
  never flipped. Re-verified at `36302fe5`: `cargo test -p fuel-tensor --lib --
  fmax_fmin_ieee_suppress_nan_where_prop_propagates
  rem_trunc_diverges_from_floored_rem_on_opposite_signs
  trunc_edges_signed_zero_nan_inf_bitwise` — 3 passed, 0 failed. No code
  change; registry-currency only.
- also: rustls 0.23.45 lock bump #330, fsspec pin #329 (lane-merged before the
  PM-gate rule took effect; reviewed after the fact as harmless, no version
  bump needed at the time — riding this entry per HANDOFF).

## v0.15.4 - 2026-10-09

### Changed

- perf(fuel-dispatch,fuel-tensor): cache `order_for`'s dispatch order across
  decode-session realize calls — the third per-token recompute flagged in
  `docs/design/incremental-consumer-index.md` §5 (`extract_runs_multi` +
  `non_chosen_arm_nodes`, ~24% of decode-step time per the v0.15.3
  measurement). `DecodeSession`/`PagedDecodeSession` now hold a
  `OnceLock<Vec<NodeId>>` populated once per session and reused while the
  graph structure, roots, and `OptimizedGraph` generation are unchanged
  (proven via `dispatch_order_is_stable_across_repeated_calls_with_no_mutation`).
  A branched graph with a real runtime selector never populates the cache —
  it routes through the existing `OrderSource::Streaming` path instead
  (proven via `cached_env_never_caches_a_branched_graph_with_a_real_selector`).
  Live verification (Qwen3-0.6B-Q4_K_M, CUDA): steady-state decode 5.01
  tok/s; order cache hit on 59/61 realize calls (96.7%), with the 2 misses
  a legitimate re-derivation at a topology-change boundary; `compiler_work`
  dropped ~41.5x on cache-hit steps vs cache-miss steps.

## v0.15.3 - 2026-10-09

### Changed

- docs(design): revised `docs/design/incremental-consumer-index.md`'s
  scope — measurement after fuel#326 found a THIRD per-token recompute
  (`order_for`'s `extract_runs_multi`/`non_chosen_arm_nodes`, ~24% of
  decode-step time, now the single largest named sub-cost) that the
  consumer/dependents index does NOT fix (different structure needed:
  a run/branch-arm partition, not a reverse-dependency index). Tracked
  as its own, separately-scoped fix track (§5), not folded into the
  consumer-index migration. Also records fuel#331 (differential harness)
  as landed.

## v0.15.2 - 2026-10-09

### Added

- test(fuel-graph): differential safety-analysis harness — step 1 of the
  staged incremental-consumer-index redesign
  (`docs/design/incremental-consumer-index.md` §4). Runs
  `insert_safety_copies`/`derive_ordering` against a corpus of
  representative graph shapes and asserts the result is structurally
  byte-identical whichever side of the comparison it's called from; the
  "new" side is a step-2 placeholder for now (calls the same old pass on
  an independently-built graph instance), so this PR validates the
  comparison machinery and corpus determinism, ready for step 2 to drop
  in the real incremental logic without touching this harness again.

## v0.15.1 - 2026-10-09

### Modified

- perf(fuel-graph): removed two independent per-decode-token O(n²) costs in
  `insert_safety_copies` and `derive_ordering` (`collect_alias_set`) — both
  re-derived their state from scratch, over the full per-token graph, on
  every `realize_inner` call, including the "plan-once" persistent-decode
  fast path. Measured ~2.7–3x steady-state decode throughput on an RTX
  4070 (Qwen3-0.6B, Q4_K_M), going from 1.81–2.03 tok/s to 5.36 tok/s; the
  two costs' combined share of per-token wall time fell from ~64% to ~36%.
  See fuel#326.
- The `alias_groups` replacement for `collect_alias_set` computes the full
  symmetric alias-equivalence partition rather than the old function's
  one-sided (forward-from-query-root-only) walk; wherever the two could
  disagree, the new version is strictly more conservative (more copies/
  pins, never a missed one). Tracked as fuel#327.

## v0.3.1 - Unreleased

### Added

### Modified

## v0.3.0 - 2023-10-01

### Added

- Added the Mistral 7b v0.1 model
  [983](https://github.com/huggingface/candle/pull/983).
- Quantized version of the Mistral model
  [1009](https://github.com/huggingface/candle/pull/1009).
- Add the gelu-erf op and activation function
  [969](https://github.com/huggingface/candle/pull/969).
- Add the mixformer/phi-v1.5 model
  [930](https://github.com/huggingface/candle/pull/930).
- Add the sclice-scatter op
  [927](https://github.com/huggingface/candle/pull/927).
- Add the Wuerstchen diffusion model
  [911](https://github.com/huggingface/candle/pull/911).

### Modified

- Support for simd128 intrinsics in some quantized vecdots
  [982](https://github.com/huggingface/candle/pull/982).
- Optimize the index-select cuda kernel
  [976](https://github.com/huggingface/candle/pull/976).
- Self-contained safetensor wrappers
  [946](https://github.com/huggingface/candle/pull/946).

## v0.2.2 - 2023-09-18

### Added
- Support for `top_p` sampling
  [819](https://github.com/huggingface/candle/pull/819).
- T5 model including decoding
  [864](https://github.com/huggingface/candle/pull/864).
- 1-d upsampling
  [839](https://github.com/huggingface/candle/pull/839).

### Modified
- Bugfix for conv2d
  [820](https://github.com/huggingface/candle/pull/820).
- Support tensor based indexing using `.i`
  [842](https://github.com/huggingface/candle/pull/842).

## v0.2.1 - 2023-09-11

### Added
- Add some RNNs (GRU and LSTM) in `candle-nn`
  [674](https://github.com/huggingface/candle/pull/674),
  [688](https://github.com/huggingface/candle/pull/688).
- gguf v2 support
  [725](https://github.com/huggingface/candle/pull/725).
- Quantized llama example in Python using the pyo3 api
  [716](https://github.com/huggingface/candle/pull/716).
- `candle-nn` layer for conv2d-transposed
  [760](https://github.com/huggingface/candle/pull/760).
- Add the Segment-Anything Model (SAM) as an example
  [773](https://github.com/huggingface/candle/pull/773).
- TinyViT backbone for the segment anything example
  [787](https://github.com/huggingface/candle/pull/787).
- Shape with holes support
  [770](https://github.com/huggingface/candle/pull/770).

### Modified
- Dilations are now supported in conv-transpose2d.
  [671](https://github.com/huggingface/candle/pull/671).
- Interactive mode for the quantized model
  [690](https://github.com/huggingface/candle/pull/690).
- Faster softmax operation
  [747](https://github.com/huggingface/candle/pull/747).
- Faster convolution operations on CPU and CUDA via im2col
  [802](https://github.com/huggingface/candle/pull/802).
- Moving some models to a more central location
  [796](https://github.com/huggingface/candle/pull/796).

## v0.2.0 - 2023-08-30

### Added
- Add the powf op
  [664](https://github.com/huggingface/candle/pull/664).
- Stable Diffusion XL support
  [647](https://github.com/huggingface/candle/pull/647).
- Add the conv-transpose2d op
  [635](https://github.com/huggingface/candle/pull/635).
- Refactor the VarBuilder api
  [627](https://github.com/huggingface/candle/pull/627).
- Add some quantization command
  [625](https://github.com/huggingface/candle/pull/625).
- Support more quantized types, e.g. Q2K, Q4K, Q5K...
  [586](https://github.com/huggingface/candle/pull/586).
- Add pose estimation to the yolo example
  [589](https://github.com/huggingface/candle/pull/589).
- Api to write GGUF files
  [585](https://github.com/huggingface/candle/pull/585).
- Support more quantization types
  [580](https://github.com/huggingface/candle/pull/580).
- Add EfficientNet as an example Computer Vision model
  [572](https://github.com/huggingface/candle/pull/572).
- Add a group parameter to convolutions
  [566](https://github.com/huggingface/candle/pull/566).
- New dtype: int64
  [563](https://github.com/huggingface/candle/pull/563).
- Handling of the GGUF file format.
  [559](https://github.com/huggingface/candle/pull/559).

## v0.1.2 - 2023-08-21

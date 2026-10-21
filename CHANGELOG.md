# Changelog
This documents the main changes to the `fuel` workspace.

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

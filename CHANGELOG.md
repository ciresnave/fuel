# Changelog
This documents the main changes to the `fuel` workspace.

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

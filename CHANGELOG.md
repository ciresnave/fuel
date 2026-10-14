# Changelog
This documents the main changes to the `fuel` workspace.

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

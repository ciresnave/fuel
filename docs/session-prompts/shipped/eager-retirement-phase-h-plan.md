# Phase H — Eager Tensor / fuel-transformers/models retirement

**Status**: Pending Workflow C binary-migration completion.

## Goal

With every fuel-examples binary migrated to the lazy-graph API (Workflow C),
the entire `fuel-transformers/src/models/` tree becomes unreferenced. Phase H
removes that tree (along with three sibling support modules that only serve
models/) and verifies the workspace builds clean.

The eager `fuel_core::Tensor` type alias itself is NOT removed — it is still
used by `fuel_transformers::generation::LogitsProcessor` (which lazy binaries
construct via `realize_f32() → Tensor::new(&[..])` during sampling) and by
some internal fuel-core test paths. A separate session may decouple
`LogitsProcessor` from `Tensor` and remove the alias entirely.

## Audit (what we expect to find unused after C)

After Workflow C, the workspace should have ZERO imports of:
- `fuel_transformers::models::*`
- `fuel_transformers::quantized_nn::*`
- `fuel_transformers::quantized_var_builder::*`
- `fuel_transformers::fused_moe::*`

Verify with:
```sh
grep -rln "use fuel_transformers::models\|use fuel::models\|use fuel_transformers::quantized_nn\|use fuel_transformers::quantized_var_builder\|use fuel_transformers::fused_moe" fuel-examples/ fuel-core/ fuel-inference/ fuel-training/ fuel-nn/
```

Known binaries that Workflow C cannot migrate (encoder-only lazy ports
lacking task heads):
- `xlm-roberta` — needs `XLMRobertaForMaskedLM` + `XLMRobertaForSequenceClassification`
- `debertav2` — needs MLM + classification heads
- `csm` — TBD (per workflow C failure report)

These get **directory-renamed** to skip auto-discovery rather than left
broken.

## Deletion steps

1. Move eager models out of the build:
   ```
   mv fuel-transformers/src/models fuel-transformers/src/_models_retired
   mv fuel-transformers/src/quantized_nn.rs fuel-transformers/src/_quantized_nn_retired.rs
   mv fuel-transformers/src/quantized_var_builder.rs fuel-transformers/src/_quantized_var_builder_retired.rs
   mv fuel-transformers/src/fused_moe.rs fuel-transformers/src/_fused_moe_retired.rs
   ```

2. Edit `fuel-transformers/src/lib.rs` — remove:
   - `pub mod fused_moe;`
   - `pub mod models;`
   - `pub mod quantized_nn;`
   - `pub mod quantized_var_builder;`

   Keep:
   - `pub mod generation;`
   - `pub mod object_detection;`
   - `pub mod pipelines;`
   - `pub mod utils;`

3. Rename un-migratable binaries to skip Cargo auto-discovery:
   ```
   mv fuel-examples/examples/xlm-roberta fuel-examples/examples/_xlm-roberta_retired
   mv fuel-examples/examples/debertav2 fuel-examples/examples/_debertav2_retired
   mv fuel-examples/examples/csm fuel-examples/examples/_csm_retired
   ```
   (Add additional renames as Workflow C surfaces them.)

   Pre-existing skip-list (already gated by `required-features` or
   intentionally not migrated):
   - `custom-ops` (raw CUDA demo)
   - `llama_multiprocess` (requires cuda+nccl+flash-attn)
   - `mnist-training` (requires fuel-datasets)
   - `reinforcement-learning` (requires pyo3)
   - `quantized-lfm2` (no base lazy port yet — LFM2 architecture not ported)

4. Verify:
   ```
   cargo build --workspace --examples
   ```
   Must exit 0.

5. Commit:
   ```
   git add -A
   git commit -m "feat(retire): remove fuel-transformers/src/models + sibling internals (Phase H)"
   ```

## Why we use `_retired` prefix instead of `rm -rf`

- Preserves git history of the eager implementations as a reference for
  cross-checking lazy ports against future bug reports.
- A follow-up session can audit each retired model and either confirm
  decommission (delete) or copy code back if a port turns out to have a
  bug that's easier to fix by diffing against eager.
- `_retired` prefix makes them invisible to Cargo auto-discovery without
  needing manifest edits — the dir simply doesn't match the
  `examples/<name>/main.rs` pattern.

## Follow-up sessions (out of scope here)

1. **LogitsProcessor decoupling** — rewrite `fuel_transformers::generation::LogitsProcessor`
   to take `&[f32]` instead of `&Tensor`. Once done, `fuel_transformers::generation` no
   longer depends on `fuel_core::Tensor`, and the eager `Tensor` type alias can be removed.

2. **Encoder task-head ports** — add `XlmrForMaskedLM`, `XlmrForSequenceClassification`,
   `DebertaV2ForMaskedLM`, `DebertaV2ForSequenceClassification` to their respective
   lazy modules; un-retire the binaries.

3. **CSM port** — diagnose what's missing in lazy_csm and either add it or document
   the architectural reason it can't ship via the lazy port.

4. **fuel-transformers crate simplification** — once models/ is gone, the crate is
   just generation + pipelines + utils + object_detection. Consider folding those
   into `fuel-nn` or a new `fuel-inference-helpers` crate and retiring
   `fuel-transformers` entirely.

5. **Quantized LFM2** — port the LFM2 base architecture to lazy (it's a hybrid SSM +
   attention model), then add `lazy_quantized_lfm2`.

# Migrate Mamba (and friends) from eager `fuel-core::Tensor` to lazy `fuel-graph::Tensor` / `LazyTensor`

## State entering this session

The CPU OpKind coverage plan ([memory `cpu-opkind-coverage-plan`](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_cpu_opkind_coverage_plan.md)) calls for three new FusedOpRegistry entries — `CausalConv1d`, `SelectiveScan`, `SsdChunkScan` — to back Mamba-1 and Mamba-2 inference. The audit that produced that plan assumed Mamba's existing prefill / step code already runs through the lazy `LazyTensor` / `Op::Fused` pipeline. **It doesn't.**

Concretely:
- `fuel-transformers/src/models/llm/mamba.rs` and `mamba2.rs` import `use fuel::{...Tensor...};` where `fuel` aliases to `fuel-core`, so `Tensor` here is the **eager** `fuel_core::Tensor`.
- Eager `Tensor::conv1d` ([fuel-core/src/conv.rs:49](../../fuel-core/src/conv.rs#L49)) dispatches immediately through `fuel_core_types::Storage::conv1d_dyn` and returns via `crate::tensor::from_storage` ([fuel-core/src/tensor.rs:208](../../fuel-core/src/tensor.rs#L208)) — eager-owned storage, no graph link.
- All elementwise / matmul / norm calls on eager `Tensor` in the Mamba code follow the same pattern. Their backprop graph (`BackpropOp` in `fuel-core/src/op.rs`) is a separate eager-tape mechanism, not the new `fuel-graph::Graph` / `Op::Fused` arena.
- Phase 7.5 work item B2 ([memory `phase_7_5_work_item_b2_complete`](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_5_work_item_b2_complete.md)) put eager Tensor factories on a "node-handle mode" path that produces graph-backed storage for **construction**, but the per-op methods (conv1d, matmul, narrow, etc.) still call eager `storage.*` and return `from_storage` — they don't emit `Op::Fused` nodes.

Result: any new FusedOpRegistry entry for `CausalConv1d` / `SelectiveScan` / `SsdChunkScan` has zero real consumers until either (a) Mamba's code is rewritten to use `LazyTensor`, or (b) eager `Tensor`'s per-op methods are lifted to construct `Op::Fused(...)` nodes under the hood instead of immediate-execute.

## Consumer surface

44 LLM files in `fuel-transformers/src/models/llm/` all import `use fuel::{...Tensor...}`. The Mamba pair is the immediate blocker because the CPU OpKind coverage plan targets it, but the architectural question is the same for every model on the eager path.

Lazy-side reference: `fuel-core/src/lazy_*.rs` files (`lazy_bert.rs`, `lazy_convnext.rs`, `lazy_qwen2_moe.rs`, `lazy_sd_text_encoder.rs`, `lazy_sd_unet.rs`, `lazy_sd_vae.rs`, `lazy_whisper.rs`, `lazy_yolov8.rs`) already use `LazyTensor` directly — these are the migrated subset. None of the LLM trio (Llama / Mamba / Mistral / etc.) is in the lazy set.

## Two retirement paths

### Option A: rewrite per-op methods on eager `Tensor` to emit graph nodes under the hood

Inside `fuel-core/src/conv.rs`, `tensor.rs`, etc., change methods like `Tensor::conv1d` to push an `Op::Fused(...)` / appropriate primitive `Op::*` node onto a per-tensor graph link instead of calling `storage.*` immediately. Eager `Tensor` becomes a thin wrapper around `LazyTensor` — keeps the existing API surface, but every op participates in graph optimization + fused dispatch.

- **Pro:** zero consumer code changes. Every model in fuel-transformers gets fused-op dispatch automatically.
- **Pro:** matches the Phase 7.5 "drop eager" direction — eager Tensor becomes a deprecated alias for LazyTensor.
- **Con:** big bang. Every eager Tensor method needs to emit the right graph node. Any method that currently slices/views without graph tracking is a hazard.
- **Con:** subtle behavior changes around realization timing. Eager callers expect `let x = a.matmul(b)?` to have already executed; lazy makes it deferred. May or may not break callers depending on whether they immediately call `.to_vec()`-style methods.

### Option B: port Mamba (only) from eager `Tensor` to `LazyTensor`

Rewrite `mamba.rs` and `mamba2.rs` to use `LazyTensor` directly, mirroring the structure of `lazy_bert.rs` and the other migrated models. Keep eager `Tensor` working unchanged for other models.

- **Pro:** scoped. Only the immediate consumer that needs the fused ops gets migrated; no cross-cutting risk.
- **Pro:** the migrated Mamba code is the test bed for the new fused ops in real inference.
- **Con:** per-op rewrites. The hand-rolled state-ring-buffer loops in Mamba's autoregressive step need careful translation (LazyTensor doesn't have in-place state mutation in the same shape as eager Tensor's storage-level mutation).
- **Con:** 14+ other LLM models stay on the eager path indefinitely.

### Option C: hybrid — Option A as the long-term direction, Option B for Mamba in the meantime

Migrate Mamba now (Option B); document Option A as the eventual destination once the Phase 7.5 "drop eager" work picks up. Mamba's lazy port informs how Option A should treat similar models later.

## Goal of this session

Recommended: **Option B for Mamba**, with Option A captured as a follow-up in the Phase 7.5 plan. Concrete deliverables:

1. **`fuel-core/src/lazy_mamba.rs`** — new file, mirrors `lazy_bert.rs` etc. Implements the Mamba-1 forward pass on `LazyTensor`. Use the existing `Tensor::causal_conv1d` / `selective_scan` / `ssd_chunk_scan` builders (which the CPU OpKind coverage plan adds — see [[causal-conv1d-session-handoff]] and successors). Until those builders land, this file can be marked `#[cfg(feature = "lazy_mamba")]` or use placeholder primitives.

2. **`fuel-core/src/lazy_mamba2.rs`** — Mamba-2 forward pass on LazyTensor, same shape.

3. **Inference parity test** — ported from the existing Mamba inference smoke test (if one exists; if not, build one from the published Mamba-130m weights). Compares token-by-token output between the eager path and the new lazy path. F32 fp-equality tolerance.

4. **Leave eager `mamba.rs` / `mamba2.rs` in place** during the migration. Once the lazy port is parity-verified and consumers (state.rs / inference loops) can swap in, delete the eager versions.

## Dependency / sequencing notes

- This session **does not need** the new fused ops (`CausalConv1d`, `SelectiveScan`, `SsdChunkScan`) to exist first. The lazy port can use primitive `LazyTensor::conv1d` + `bias_add` + `silu` for the prefill path and primitive ops for the recurrence — same shape as eager Mamba does internally. The fused ops simply replace those primitive chains when they land.
- The new fused ops session(s) can also run in parallel without waiting for this migration to complete. Until the migration completes, the fused ops have unit tests but no real Mamba consumer. After this migration completes, the lazy port's Mamba paths immediately benefit (just swap primitive chain → fused-op call inside `lazy_mamba*.rs`).

## Open questions to surface before implementing

1. **Mamba autoregressive state mutation** — eager Mamba uses storage-level mutation for the conv-state ring buffer (`state.conv_states[i] = ...`). LazyTensor doesn't have an equivalent in-place primitive in the same shape today. Either (a) thread state through as a returned `LazyTensor` (functional shape, fits the lazy model), or (b) wait for the in-place ops infrastructure (Phase 4 complete per [memory `inplace_ops_complete`]) to grow a "scatter into existing tensor" primitive. Pick before writing the recurrence step.
2. **Realization boundary** — where in the inference loop should `realize_*` be called? Eager Mamba realizes implicitly per-op; lazy Mamba can realize once per generated token. Performance impact varies by model size.
3. **KvCache compatibility** — `fuel_nn::kv_cache::KvCache` (used by Mamba2) — does it work with LazyTensor, or is it eager-only? Check before porting.

## References

- The original audit + scope decisions: [memory `cpu-opkind-coverage-plan`](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_cpu_opkind_coverage_plan.md).
- The "drop eager" target: [memory `phase_7_5_core_simplification`](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_5_core_simplification.md).
- An existing eager-retirement session prompt for shape reference: [`fuel-flash-attn-cuda-eager-retirement.md`](fuel-flash-attn-cuda-eager-retirement.md).
- The lazy ports already shipped: `fuel-core/src/lazy_bert.rs`, `lazy_convnext.rs`, `lazy_qwen2_moe.rs`, `lazy_sd_text_encoder.rs`, `lazy_sd_unet.rs`, `lazy_sd_vae.rs`, `lazy_whisper.rs`, `lazy_yolov8.rs`.

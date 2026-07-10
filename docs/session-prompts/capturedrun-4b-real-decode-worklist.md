# CapturedRun Increment 4b — real Llama decode capture: de-risked worklist

**Status (2026-07-10):** the CUDA capture mechanism + full non-decode op-set are on `main`
(`5e06c32a`); the input-lifetime bug that faked "cuBLAS/index_select not capture-safe" is FIXED.
This doc is the ground-truth worklist for capturing a REAL TinyLlama F32 persistent-decode step,
from a thorough map of `fuel-core/src/lazy.rs` + `inference_context.rs` + the executor. Six
interdependent pieces — ALL required; none unblocks decode capture alone.

## The decode graph's capture gaps (verified from the map)

The F32 persistent-decode step (`apply_layer_with_kv_writes`, `lazy.rs:6244`) is mostly
already-capture-safe: `IndexSelect` (embed), `RmsNormLastDim`, `MatMul` (q/k/v/o/ffn/logits — GQA
inferred in-kernel), `SoftmaxLastDim`, `SiluElementwise`, `AddElementwise`/`MulElementwise`
(residuals/gate/mask-add) are in the predicate; reshape/permute/transpose/slice are metadata views;
and **`WriteSliceDoff` (KV append) is ALREADY capture-safe** — it compiles to its own
`WorkItemKind::WriteSliceDoff` (not predicate-gated), reads a device-resident rank-0 I64 offset (no
D2H, no alloc), adopts the fixed KV dest in place. Weights + KV are fixed-address in `base_cache`.

**The three genuine gaps:**

1. **`OpKind::Affine`** — the `scores.mul_scalar(1/√head_dim)` attention scale
   (`Op::MulScalar` → `OpKind::Affine`, `pipelined.rs:3079`). Its CUDA wrapper
   (`cuda_affine_baracuda_wrapper!`, `baracuda_dispatch.rs:1837`) is **alloc-and-replace**
   (`let result = ...; *out_cuda = result;`) → allocates inside capture + changes output address.
   FIX: refactor `fuel-cuda-backend/src/baracuda/affine.rs` to a write-into `affine_*_into` (like
   binary/rope), flip the wrapper, add `OpKind::Affine` to `op_kind_is_capture_writeinto`, forced-
   reuse capture test.

2. **RoPE — RESOLVED 2026-07-10: NOT `OpKind::Rope`, and NOT a Kernel-predicate gap.** The fused
   3-input `rope_with_tables` (`Op::Fused(ROPE)`, dtypes `[F32,F32,F32,F32]`) has **NO CUDA kernel**
   — proven by a raw-realize error "no backend supports rope on [F32,F32,F32,F32]". So on CUDA it
   **decomposes at optimize time** (`rope_with_tables_decomposed`: slice/neg/**concat**/broadcast/
   mul/add → ~72 dispatches). The decomposed mul/add/neg are affine/binary (✓ capture-safe), but
   the **`Op::Concat` (`OpKind::Concat`) ALLOCATES its output** — so rope folds into gap #3.
   (`capture_decode` on a RAW graph errors on the fused rope because it doesn't run the optimizer;
   the real decode passes an already-OPTIMIZED graph, where rope is already decomposed — so target
   the optimized graph, and the concat is what needs handling.)

3. **THE REAL CRUX — allocation of intermediate buffers inside the capture scope.** The persistent-
   output executor mode currently reuses only KERNEL outputs. But the decode graph allocates
   intermediates in NON-Kernel arms during capture, which is forbidden:
   - **`Op::Concat`** (decomposed rope) — allocates its output (`OpKind::Concat` not in predicate).
   - **Auto-contiguize** — the attention-merge `reshape` after `permute([0,2,1,3])` (`lazy.rs:6384`)
     has a STRIDED input, so `WorkItemKind::ContiguizeOf` calls `auto_contiguize` → **allocates**
     (`pipelined.rs:5827-5831`). Logits `slice→reshape` is a second candidate.
   THE GENERAL FIX: extend the persistent-output mode (Record/Reuse) to cover EVERY allocating work
   item — `ContiguizeOf`, `Concat`, and any allocating view — not just `Kernel`. In Record, alloc +
   record the buffer keyed by node; in Reuse, reuse it (+ write-into variants: `auto_contiguize_into`,
   concat-into). This is a general, careful executor enhancement — the real remaining hard part of 4b.
   (Alternative per-site: bake contiguous producers / avoid concat in the rope decompose — more
   surgical but less general.)

## The wiring gaps

4. **Target `logits_node`, not `effective_target`.** `prebuild_optimized_capturing` splices an
   `Op::Copy{target:Cpu}` (D2H) on top of `logits_node`; `effective_target` is that Copy, which
   `capture_decode` hard-rejects (`pipelined.rs:1084`). Point capture at the device-resident
   `logits_node` (thread a getter through `DecodeSession`); do the D2H readback yourself from
   `CapturedDecode::output` after `run.replay()`.

5. **Fixed per-token input buffers.** `build_token_rope_mask_arcs` (`lazy.rs:7034`) allocates FRESH
   Arcs per token for `{token_ids, rope_cos, rope_sin, mask, offset}` — new device addresses each
   token, incompatible with a captured graph. Allocate these 4–5 ONCE and H2D-overwrite in place
   (exactly `CapturedDecodeSession.per_token_inputs` + `replay_token`). Weights/KV already fixed.

6. **Fuel-core wiring + verify + bench.** Add a capture path to the decode driver: build the
   session's graph + base_cache (already done by `build_and_realize_first_decode_token`), then
   `CapturedDecodeSession::capture(graph, logits_node, base_cache+fixed_per_token, per_token_ids,
   per_token_sym_env)`; per token, `replay_token` with the new token-id/rope/mask/offset bytes +
   D2H the logits. BIT-EXACT gate: captured logits == uncaptured `realize_token` logits per token
   (byte-exact, greedy tokens match). RE-BENCH: add a third leg to `run_persistent_decode_bench`
   (`lazy.rs:11963`) — captured-replay per-token ms vs D2 (plan-once) vs D1 (rebuild); median of
   ≥8 same-phase runs (current bench is mean/single — improve), discard warmup, log nvidia-smi.
   Build is ~36-min cold; front-load gaps 1–3 (verified in fuel-dispatch synthetic tests) so
   fuel-core iterations are few.

## Suggested increment order (front-load fast fuel-dispatch pieces)

- **4b-α (fuel-dispatch, fast):** Affine write-into + predicate + forced-reuse test.
- **4b-β (fuel-dispatch, fast):** resolve Rope (graph dump); if fused, convert its kernel; test.
- **4b-γ (fuel-dispatch, moderate):** `ContiguizeOf` persistent-output + write-into contiguize;
  synthetic strided-reshape capture test.
- **4b-δ (fuel-core, slow):** `DecodeSession` logits_node getter + fixed per-token buffers +
  `CapturedDecodeSession` wiring; real-model bit-exact.
- **4b-ε (fuel-core, slow):** third bench leg + median protocol + numbers.

Each of α–γ is a self-contained, GPU-verified fuel-dispatch commit; δ–ε are the fuel-core finish.

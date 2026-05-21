# Vulkan kernel catch-up audit

Surveyed 2026-05-21 after the CUDA-side work on baracuda alpha.29
landed (WriteSlice, Contiguize, Triu/Tril, Flip, Roll, CumSum,
Pad/PadBackward, F8E4M3 cast). The Vulkan path is significantly
behind. This doc captures the gap and sequences the work needed to
bring Vulkan to parity.

---

## TL;DR

Vulkan is **not in the pipelined-executor binding-table dispatch
system at all today**. There is no `register_vulkan_kernels` function
in `fuel-storage`; Vulkan flows only through the legacy
`GraphBackend` trait via `fuel-graph-executor::GraphExecutor<VulkanBackend>`.
The substrate is in place (`VulkanStorageBytes` implements
`BackendStorage`; `BackendStorage::Vulkan(_)` exists in the enum) —
the dispatch plumbing isn't.

So the gap isn't just "write some Slang." It's:

1. **Foundation** — wire Vulkan into the pipelined-executor binding-
   table dispatch system the way CUDA is today (V.1).
2. **Wire existing SPIR-V** — ~25 Slang kernels already compile to
   SPIR-V in `fuel-vulkan-kernels`; register them against binding-
   table keys (V.2).
3. **Write new Slang** — close the kernel-coverage gap with CUDA
   (V.3); most net-new authoring is here.
4. **Migrate + retire** — flip Vulkan example bins to the new path;
   drop the legacy `KVCache<B>`, `forward_with_cache_gpu_on`,
   `LayerKVCache` proxy (V.4). This is the same retirement E.3.3.D
   parked.

Estimated full scope: **5–8 sessions** end-to-end, plus per-kernel
Slang authoring time (some kernels are trivial — Triu/Tril are
~30 lines of Slang; some are non-trivial — Cast pairs explode to
the same 64-symbol surface CUDA has).

---

## Current Vulkan kernel inventory

What's in `fuel-vulkan-kernels/spv/` (precompiled SPIR-V from
`fuel-kernels-source/kernels/*.slang`):

### Compute kernels (29 SPIR-V files)

| File | Op family | Dtypes |
|---|---|---|
| `binary.spv` | Add / Sub / Mul / Div | **f32 only** |
| `unary.spv` | Neg / Sqr / Sqrt / Exp / Log / Sin / Cos / Tanh / Sigmoid / Silu / Gelu / Relu / Step (13 ops) | **f32 only** |
| `affine.spv` | `y = a*x + b` | f32 |
| `add_assign_scaled.spv` | `y += alpha * x` (optimizer step) | f32 |
| `matmul.spv` + `matmul_tiled.spv` + `matmul_coop.spv` | MatMul (basic + tiled + coop) | f32 |
| `matmul_tiled_bf16_b.spv` | MatMul with bf16 RHS | f32×bf16 |
| `matmul_q4_0_tiled.spv` + `qmatvec_q4_0.spv` | Quantized matmul | f32 × Q4_0 |
| `matvec.spv` + `matvec_bf16_b.spv` | MatVec | f32, f32×bf16 |
| `reduce.spv` + `reduce_last_dim.spv` | Sum / Max / Min / Mean reduction | f32 |
| `rms_norm_last_dim.spv` | RmsNorm forward | f32 |
| `rms_norm_last_dim_backward.spv` | RmsNorm backward | f32 |
| `layer_norm_last_dim_backward.spv` | LayerNorm backward (**no forward!**) | f32 |
| `rope.spv` | RoPE | f32 |
| `softmax.spv` + `softmax_last_dim_backward.spv` | Softmax + backward | f32 |
| `silu_forward.spv` + `silu_backward.spv` | Silu | f32 |
| `concat_along_dim.spv` | Concat | f32 |
| `conv2d_im2col.spv` | Conv2D via im2col + matmul | f32 |
| `index_select.spv` | IndexSelect | f32, u32 indices |
| `flash_attention.spv` | FlashAttn | f32 |
| `dequant_q4_0.spv`, `dequant_q4_km.spv`, `dequant_q8_0.spv`, `quantize_q8_0.spv` | GGUF dequant + Q8_0 quant | per-format |
| `strided_copy.spv` | Contiguize-equivalent (PERMUTE/BROADCAST/SLICE) | **f32 only, unsigned strides only** |

**Headline numbers**:
- **29 SPIR-V kernels** today
- **f32-only** for almost everything (only matmul has bf16 RHS coverage)
- **No multi-dtype binary/unary/reduce/softmax/RmsNorm/RoPE** — Vulkan can't run an f16 / bf16 / f64 model end-to-end
- **No signed strides in `strided_copy`** — so it can't materialize a `Flip`'s output even if Flip itself shipped

---

## CUDA op coverage Vulkan is missing

Cross-checked against `fuel-storage/src/baracuda_dispatch.rs::register_baracuda_cuda_kernels`
and `dispatch.rs::register_cuda_kernels` (the PTX path). Excludes
view ops (no kernel needed).

### Tier 1 — Critical for E.3.3.D retirement

These block dropping the legacy `KVCache<B>` + `forward_with_cache_gpu_on`
device-resident path:

| OpKind | Vulkan today | CUDA today | Effort |
|---|---|---|---|
| `WriteSlice` | ❌ | ✅ (baracuda b1/b2/b4/b8 + nibble) | Slang: ~80 lines (single-dtype byte-width fan-out via per-dtype variants); 1 session |
| `Contiguize` (signed strides) | partial (unsigned only) | ✅ (baracuda b1/b2/b4/b8) | Slang: extend `strided_copy.slang` for signed strides; <30 lines; 0.5 session |

### Tier 2 — Multi-dtype coverage gap

CUDA covers f32/f64/bf16/f16 for these; Vulkan has **only f32**.
Most need per-dtype Slang variants (Slang has limited generics for
half/bf16) OR a single `f16` -> f32 -> f16 detour pattern:

| OpKind | Vulkan f16/bf16/f64 effort |
|---|---|
| Binary (Add/Sub/Mul/Div/Maximum/Minimum) | ~3-4 variants per dtype = 9 Slang files; ~2 sessions for f16+bf16+f64. Maximum/Minimum also missing on f32. |
| Unary (13 ops) | Same fanout; 2-3 sessions |
| Reduce (Sum/Max/Min/Mean) | 4 dtypes; 1-2 sessions |
| Softmax / RmsNorm / LayerNorm / Rope | LayerNorm forward also missing on f32 even; per-dtype = 1-2 sessions each |
| MatMul (f64) | f64 matmul probably doesn't exist on Vulkan today (no DP4A equivalent); may stay CPU-only or compose via f32-detour. Defer. |

### Tier 3 — Net-new op families

| OpKind | Vulkan today | CUDA today | Slang effort |
|---|---|---|---|
| `Triu` / `Tril` | ❌ | ✅ | ~30 lines each; 0.5 session for both |
| `Flip` | ❌ | ✅ | ~50 lines; ~0.5 session |
| `Roll` | ❌ | ✅ | ~50 lines; ~0.5 session |
| `CumSum` | ❌ | ✅ | Block-scan; ~150 lines; 1 session |
| `Pad` (Constant/Reflect/Replicate) | ❌ | ✅ | 3 mode-dispatch arms; ~120 lines; 1 session |
| `PadBackward` (Constant mode) | ❌ | ✅ | ~50 lines; 0.5 session |
| `Cast` (all pairs) | ❌ | ✅ (64 pairs) | One Slang kernel templated on (src, dst); 1-2 sessions including pairs |
| `MaskedFill` | ❌ | ✅ | ~40 lines; 0.5 session |
| `Gather` | ❌ | ✅ | ~80 lines; 0.5 session |
| `ScatterAdd` | ❌ | ✅ | Atomic-add scatter; ~100 lines; 1 session |
| `IndexAdd` | ❌ | ✅ | ~80 lines; 0.5 session |
| `ArgMaxDim` / `ArgMinDim` | ❌ | ✅ | Per-dim reduce with index tracking; ~100 lines each; 1 session for both |
| `PowI` | ❌ | ✅ | Unary with i32 exponent; ~30 lines; 0.5 session |
| `Clamp` | ❌ | ✅ | ~20 lines; 0.5 session |
| `Maximum` / `Minimum` (binary) | ❌ | ✅ | Adds 2 ops to binary.slang; 0.5 session |
| Compare ops (Equal/Ne/Lt/Le/Gt/Ge) | ❌ | ✅ | Adds 6 ops to binary.slang with u8 output; 0.5 session |
| `Where` | ❌ | ✅ | Ternary select; ~30 lines; 0.5 session |
| Float unary extras (Floor/Ceil/Round/Sign/Erf/GeluErf/Pow/Rsqrt/Rem) | ❌ | ✅ | Adds 9 ops to unary.slang; 0.5 session |
| `Recip` / `Abs` | ❌ | ✅ | Adds 2 ops to unary.slang; trivial |
| `LogSoftmaxLastDim` + backward | ❌ | ✅ | ~80 lines forward + backward; 1 session |
| `Triu` / `Tril` | ❌ | ✅ | Counted above |
| F8E4M3 cast | ❌ | ✅ (6 pairs) | Slang FP8 support depends on Slang version + hardware; probably defer until Vulkan FP8 use case emerges |
| `ConvTranspose2D` | ❌ | ✅ (cuDNN-backed) | Conv-transpose Slang; ~200 lines; 1-2 sessions |
| `PagedAttn` | ❌ | ✅ | Significant work — block-table addressing; 2-3 sessions |
| `FusedLinear` | ❌ | ✅ | Composes from existing matmul + bias-add; mostly registration |
| int8 GEMM | ❌ | ✅ | Vulkan int8 support depends on `VK_KHR_shader_integer_dot_product` extension; defer until target hardware audit |
| GGUF dequant for Q4_K_M-style (full set) | partial (q4_0, q4_km, q8_0) | full set | Defer; CUDA's MMVQ path is the active GGUF runner |

### Tier 4 — Backward / training

CUDA has decomposed-backward (autograd) coverage for most forward
ops. Vulkan training is currently parked. Backward Slang stays
deferred until Vulkan training is on the roadmap. The exceptions
(`rms_norm_last_dim_backward`, `layer_norm_last_dim_backward`,
`softmax_last_dim_backward`, `silu_backward`) are already there
because they were needed before any retreat to inference-only.

---

## Phased migration plan

### V.1 — Foundation (mandatory; blocks everything downstream)

**Goal**: get Vulkan into the pipelined-executor binding-table
dispatch system with ONE op working end-to-end.

**Work units**:

1. Extend `fuel-storage/src/pipelined.rs::execute_work_item`'s output-
   allocation match arm to handle `BackendId::Vulkan`. Mirror the
   CUDA shape: derive the device handle from the first input's
   `BackendStorage::Vulkan(_)` variant, allocate via
   `fuel_graph_vulkan::VulkanStorageBytes::alloc(device, n_bytes)`
   (or equivalent — verify the API).
2. Extend `fuel-storage/src/pipelined.rs::auto_contiguize` with a
   `BackendStorage::Vulkan(_)` arm. For V.1, the slow path (D2H →
   CPU contiguize → H2D) is acceptable; native Vulkan contiguize
   is a V.3 work unit.
3. Create `fuel-storage/src/vulkan_dispatch.rs` parallel to
   `baracuda_dispatch.rs`. Mirror the macro-based dispatch wrapper
   pattern. Empty `register_vulkan_kernels` initially.
4. In `fuel-graph-vulkan` (or a new `fuel-vulkan-backend` crate, TBD
   — see "Architectural questions" below): expose per-kernel Rust
   functions that pipeline SPIR-V → descriptor sets → dispatch.
   Pattern: each kernel takes `&VulkanDevice, &VulkanStorageBytes,
   ...` and returns `Result<VulkanStorageBytes>`. Probably already
   exists in some form; needs audit.
5. Wire ONE op (e.g. Add f32) end-to-end:
   - Slang already exists (`binary.spv` covers Add f32)
   - Add a thin wrapper that loads the SPIR-V, binds descriptors,
     dispatches
   - Register at `(OpKind::AddElementwise, [F32, F32, F32], BackendId::Vulkan)`
   - Add a live-Vulkan test mirroring `baracuda_add_f32` pattern
6. Validate via the existing `cpu_vulkan_diff.rs` test infrastructure
   that the pipelined executor + Vulkan binding-table dispatch
   matches the CPU reference at 1e-5 tolerance.

**Estimated effort**: 1-2 sessions. Most risk is in #4 (the device
plumbing); the rest is mechanical.

### V.2 — Wire existing SPIR-V (mechanical)

**Goal**: register all 29 existing SPIR-V kernels against binding-
table keys.

**Per-kernel work**: ~30 LoC for a dispatch wrapper + 1-3 lines of
registration. Each kernel takes 5-15 minutes once V.1's plumbing
is in place.

**Coverage delta**: after V.2, Vulkan-via-pipelined-executor matches
the legacy `GraphBackend<VulkanBackend>` op surface 1:1. Vulkan
example bins still use legacy because they haven't been migrated
(V.4) but the dispatch path is in place.

**Estimated effort**: 2-3 sessions. The macro chassis from
baracuda_dispatch.rs should be reusable as a template.

### V.3 — Write new Slang

**Goal**: close the kernel-coverage gap with CUDA. Ordered by
priority (Tier 1 first, then Tier 3 in expected-frequency order).

**Sequencing recommendation**:

1. **WriteSlice (Slang)** — unblocks E.3.3.D's full retirement.
   ~80 lines; 1 session including tests.
2. **Contiguize signed-stride extension** — eliminates the D2H/H2D
   fallback that V.1 used as a stopgap. ~30 lines extension to
   `strided_copy.slang` + a per-dtype fanout. 0.5 session.
3. **Triu/Tril + Flip/Roll** — small kernels, often-used in attention
   masks and dataset ops. 1 session for the cluster.
4. **CumSum + Pad/PadBackward** — block-scan + multi-mode kernel
   pair. 1-2 sessions.
5. **Cast (f32/f64/bf16/f16 pairs)** — large coverage hit; one
   templated Slang + 16 pairs. 1-2 sessions.
6. **MaskedFill + Gather + ScatterAdd + IndexAdd** — indexing
   family; 1-2 sessions.
7. **ArgMaxDim/ArgMinDim** — per-dim reduce with index tracking;
   1 session.
8. **Multi-dtype fanout for existing kernels** (binary/unary/reduce/
   softmax/RmsNorm/RoPE for f16/bf16) — Slang's half/bf16 support
   varies; needs investigation. ~3-5 sessions if straightforward,
   more if Slang requires a transpiler workaround.
9. **PowI/Clamp/Maximum/Minimum/Where/compare ops/extra unary** —
   single-kernel additions to existing files. 1 session total.
10. **LogSoftmaxLastDim** — forward + backward; 1 session.
11. **PagedAttn / ConvTranspose2D** — large; defer unless a model
    demands them.

**Architectural decisions to make before V.3**:
- **Slang multi-dtype**: per-dtype variants (one Slang file per dtype)
  vs templated (one Slang file with `T` substituted at compile time).
  Slang's generics support is incomplete; the existing kernels are
  per-dtype-instantiated (verify by reading matmul_tiled_bf16_b.slang
  vs matmul_tiled.slang).
- **Workspace handling**: CUDA's kernels take workspace pointers
  (used by some). Vulkan's compute pipeline model has push constants
  + descriptor sets; transient scratch likely needs explicit binding.
  Audit the existing Vulkan kernel infrastructure for how this is
  handled today.

### V.4 — Migration + retirement

**Goal**: flip Vulkan example bins to the new pipelined-executor
path; retire the legacy `KVCache<B>` + `forward_with_cache_gpu_on`
+ `LayerKVCache` proxy.

**Work units**:

1. Add `LlamaModel::forward_with_kv_context` Vulkan support — should
   "just work" once V.1 ships KvCache + InferenceContext + WriteSlice
   coverage for Vulkan.
2. Migrate `llama-lazy-vulkan` + `phi-lazy-vulkan` example bins to
   `generate_streaming_with_kv_context`. Same shape as the
   `llama-lazy-cuda` migration in commit `e9750e34`.
3. Retire `LayerKVCache` struct (the device-resident proxy).
4. Retire `KVCache<B>` + `KVCacheEntry` enum.
5. Retire `LlamaModel::forward_with_cache_gpu_on*` (3 variants),
   `apply_layer_with_cache` (the host-resident helper),
   `generate_streaming_gpu_on`, `generate_streaming_cuda`.
6. Retire the parallel device-resident path on `PhiModel` (line
   5131 in lazy.rs) and `Gemma2Model` if present.

**Estimated effort**: 2-3 sessions, mostly mechanical deletions
with test updates.

---

## Architectural questions to settle before V.1

These are decisions that shape the V.1 design and shouldn't be made
mid-flight:

1. **`fuel-vulkan-backend` crate vs extending `fuel-graph-vulkan`**?
   The CUDA side has `fuel-cuda-backend` (kernel wrappers) separate
   from `fuel-graph-vulkan`'s equivalent (`fuel-graph-vulkan` mixes
   pipeline cache, recorder, residency, and would gain kernel
   wrappers if we extend it). A new `fuel-vulkan-backend` crate
   parallels CUDA cleanly but adds workspace churn. Recommendation:
   add to `fuel-graph-vulkan` for V.1 to defer the crate-split
   decision; revisit if the dispatch wrappers reach ~10 modules.

2. **Slang per-dtype variants vs templates**? See V.3 architectural
   decisions above.

3. **Vulkan device handle threading through the pipelined executor**?
   The CUDA path derives `CudaDevice` from the first input's
   `BackendStorage::Cuda(_)`. Vulkan would do the same with
   `BackendStorage::Vulkan(_)`. Validate the `VulkanStorageBytes`
   API exposes a `device()` accessor.

4. **Backend choice when Vulkan + CUDA both compiled**? The
   judge/route picker already handles multi-alternative selection
   per `(OpKind, dtypes, BackendId)`. The user's existing telemetry
   pipeline (Phase 6b probe→judge→dispatch) should pick automatically.
   No new architectural work; just confirm V.1's kernel registers
   as `BackendId::Vulkan` (not aliased to Cuda or similar).

---

## What's NOT in this audit

- **Per-kernel performance profiling**. The CUDA-side kernels have
  empirical cost data through the Judge; Vulkan would gain that
  automatically once it's in the binding table. Performance tuning
  (workgroup sizes, tiled patterns, cooperative matmul) is post-V.3
  work.
- **Vulkan subgroups + cooperative matrices** (Slang's `subgroup_*`
  + `__diff` per the shader policy memory). Some existing kernels
  use these (`matmul_coop.glsl`); new Slang authoring should follow
  the same patterns. Not a gating concern; just an authoring style
  note.
- **F8E4M3 / sub-byte FP8 Slang**. Slang's FP8 support depends on
  hardware + extension availability. Defer until a Vulkan FP8 use
  case emerges (none today).
- **Vulkane (the sibling project)**. Vulkane is the Vulkan-side
  analogue of baracuda. For kernel needs, fuel ships Slang sources
  in `fuel-kernels-source` rather than waiting on vulkane. Vulkane
  is relevant for **device / context / driver-API** wrapping
  (similar to `baracuda-driver`'s role), not for kernel-level work.

---

## Recommendation

Start with **V.1** (foundation). Until the binding-table dispatch
is wired for Vulkan, all the Slang in the world has no consumer.
V.1 establishes the path; V.2 fans out mechanically; V.3 is where
the actual kernel-authoring work happens; V.4 finishes the E.3.3.D
retirement.

Once V.1 lands, V.3 work can start in parallel with V.2 since they
touch different code (V.2 = dispatch registrations; V.3 = Slang
sources + new SPIR-V).

The two work units in V.3 that unblock E.3.3.D's full retirement
are **WriteSlice (Slang)** and **Contiguize signed-stride extension**.
If the priority is "finish E.3.3.D entirely as fast as possible,"
sequencing is:
  - V.1 (foundation, any op)
  - V.3.1 (WriteSlice Slang)
  - V.3.2 (Contiguize signed-stride extension)
  - V.4.1 (LlamaModel::forward_with_kv_context Vulkan support)
  - V.4.2 (Vulkan example bin migration)
  - V.4.3 (legacy retirement)

That's a ~4-6 session path from V.1 to E.3.3.D's full cleanup,
deferring the broader Vulkan op-surface catch-up until after.

---

## Reference commits

- `f2d648b6` — E.3.3.D host-resident retirement (CPU + CUDA done;
  Vulkan device-resident path deferred to this audit's V.4)
- `e9750e34` — CUDA example bin migration (template for V.4.2)
- `a22901b8` — WriteSlice CUDA integration (template for V.3.1)
- `c8834bf6` — Contiguize CUDA integration (template for V.3.2)
- `b52d709c` — `forward_with_kv_context` on LlamaModel (V.4.1
  reuses this method; just needs Vulkan kernel coverage to flow
  through it)

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

**Update 2026-05-21**: Three corrections from the initial audit
based on user pushback + verification:

- **Slang generics work** — Slang supports full generic kernels via
  `T:IFloat` / `T:IInteger` interface constraints (see [Slang
  Interfaces and Generics docs](http://shader-slang.org/slang/user-guide/interfaces-generics)).
  The per-dtype Slang files in our codebase are a WGSL-port artifact,
  not a Slang limitation. **Preference: write generic kernels** and
  emit per-dtype SPIR-V via Slang's compile-time specialization;
  fall back to per-dtype source only when a kernel needs SPIR-V
  intrinsics the generic path can't reach (e.g., bf16 hardware ops
  on architectures that need them, FP8 emit paths if/when SPIR-V
  gains them).
- **F8E4M3 cast is NOT deferred** — software pack/unpack + f32 math
  is dtype-agnostic Slang code. Moved to Tier 3 with the other Cast
  pairs. (Hardware FP8 *matmul* would be deferred since it needs
  device support; Cast is pure software.)
- **Workspace handling is not a decision** — see "V.0 — Audited"
  below; the Vulkan side already has a clean per-launch pattern
  (descriptor sets for buffers + ConstantBuffer for params + inline
  `alloc_device` for scratch) that maps cleanly to CUDA's
  `workspace, workspace_bytes` pattern.

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
| F8E4M3 cast | ❌ | ✅ (6 pairs) | Software pack/unpack + f32 math in Slang; same templated dispatch as other Cast pairs. ~0.5 session. (Hardware FP8 *matmul* is separate and depends on Vulkan device extensions.) |
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

### V.0 — Crate rename + workspace audit (prerequisite)

Two short prerequisites surfaced 2026-05-21:

**V.0.1 — Rename `fuel-vulkan-backend` → `fuel-vulkan-backend`** so
the directory structure mirrors `fuel-cuda-backend` exactly.
Different backends should have roughly equivalent crates so a
developer switching from one to another doesn't have to relearn
where things live.

Scope (mechanical find/replace):
- Rename crate dir + `Cargo.toml` package name
- Update workspace `Cargo.toml` member entry
- Update ~25 `use fuel_vulkan_backend::*` references across
  `fuel-core`, `fuel-core-types`, `fuel-cuda-backend`,
  `fuel-graph-router`, `fuel-lazy-examples`, `fuel-storage`, plus
  the crate's own `tests/` and `src/`
- Update `fuel-storage/src/lib.rs:52`
  (`pub use fuel_vulkan_backend::VulkanStorageBytes as VulkanStorage`)

Estimated effort: 1 session, mostly find/replace + workspace
build verification.

**V.0.2 — Workspace audit (done, results below)**:

Compared Vulkan's per-launch state plumbing with CUDA's:

| Concept | CUDA (baracuda) | Vulkan (existing) |
|---|---|---|
| Per-launch params | Typed C function arguments (`stride: u32`, etc.) | `ConstantBuffer<Params>` (uniform buffer) bound to descriptor set, or HLSL `push_constant`; existing fuel kernels use `ConstantBuffer<Params>` |
| Per-launch scratch | Caller passes `workspace: *mut c_void, workspace_bytes: usize` | Inline `alloc_device(bytes, ...)` before dispatch + descriptor-set binding as an additional storage buffer (already in use — see `conv2d` im2col `patches` scratch at `fuel-vulkan-backend/src/lib.rs:2067`) |
| Pipeline-level state | None (each kernel is a function pointer) | Pre-created `PipelineLayout` cached in a pipeline cache (lazy init at first use; structure already exists in `fuel-vulkan-backend/src/pipelines.rs`) |
| Stream / queue | `CUstream` per call | Vulkan command buffer recorded onto the device's queue (existing pattern via the `recorder.rs`) |

**Conclusion**: no new architecture needed. The Vulkan pattern
maps cleanly onto CUDA's workspace concept — kernels that need
scratch allocate it inline; the pipeline / descriptor-set
plumbing handles parameter passing. New V.3 Slang kernels follow
the existing fuel Vulkan pattern (`ConstantBuffer<Params>` +
storage buffer descriptors); the `register_vulkan_kernels`
wrappers in V.1/V.2 follow the existing `conv2d` /
`flash_attention` wrapper pattern in `fuel-vulkan-backend/src/lib.rs`.

### V.1 — Foundation (mandatory; blocks everything downstream)

**Goal**: get Vulkan into the pipelined-executor binding-table
dispatch system with ONE op working end-to-end.

**Work units**:

1. Extend `fuel-storage/src/pipelined.rs::execute_work_item`'s output-
   allocation match arm to handle `BackendId::Vulkan`. Mirror the
   CUDA shape: derive the device handle from the first input's
   `BackendStorage::Vulkan(_)` variant, allocate via
   `fuel_vulkan_backend::VulkanStorageBytes::alloc(device, n_bytes)`
   (or equivalent — verify the API).
2. Extend `fuel-storage/src/pipelined.rs::auto_contiguize` with a
   `BackendStorage::Vulkan(_)` arm. For V.1, the slow path (D2H →
   CPU contiguize → H2D) is acceptable; native Vulkan contiguize
   is a V.3 work unit.
3. Create `fuel-storage/src/vulkan_dispatch.rs` parallel to
   `baracuda_dispatch.rs`. Mirror the macro-based dispatch wrapper
   pattern. Empty `register_vulkan_kernels` initially.
4. In `fuel-vulkan-backend` (or a new `fuel-vulkan-backend` crate, TBD
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

**Architectural notes for V.3** (the audit closed both questions
that were originally framed as decisions):

- **Slang multi-dtype**: prefer generic kernels via `T:IFloat` /
  `T:IInteger` interface constraints — Slang's generics work, and
  the existing per-dtype Slang files are a WGSL-port artifact, not
  a Slang limitation. Each generic source compiles to N SPIR-V
  binaries (one per concrete dtype) via Slang specialization; the
  source code stays single. Fall back to per-dtype source only
  when a kernel needs SPIR-V intrinsics the generic path can't
  reach (rare; only some bf16-hardware-op-specific patterns).
- **Workspace handling**: already audited (see V.0.2 above). No
  decision; just translate CUDA's `workspace, workspace_bytes`
  pattern to Vulkan's "inline `alloc_device` + descriptor-set
  binding" pattern, which fuel's Vulkan kernels already use.

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

## Architectural questions — resolved

All four originally-open questions closed by the 2026-05-21 audit
round:

1. **`fuel-vulkan-backend` crate vs extending `fuel-vulkan-backend`**:
   resolved → rename `fuel-vulkan-backend` to `fuel-vulkan-backend`
   in V.0.1. Different backends should have roughly equivalent
   crate structures so developers can switch contexts without
   relearning where things live.

2. **Slang per-dtype variants vs templates**: resolved → prefer
   generic `T:IFloat` / `T:IInteger` kernels. See V.3 architectural
   notes above.

3. **Vulkan device handle threading through the pipelined executor**:
   verify `VulkanStorageBytes::device()` accessor exists during
   V.1. Mechanical (mirror `CudaStorageBytes::device()`).

4. **Backend choice when Vulkan + CUDA both compiled**: the existing
   judge/route picker (Phase 6b probe→judge→dispatch) handles
   multi-alternative selection automatically. No new architectural
   work; just confirm V.1's kernel registers under
   `BackendId::Vulkan`.

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
- **(No hardware-FP8 deferrals.)** Initial audit parked "Hardware
  FP8 Vulkan matmul" citing per-vendor hardware support; that was
  wrong. The dev machine has an RTX 4070 (Ada Lovelace) with
  native FP8 support; Vulkan exposes both storage and arithmetic
  via `VK_EXT_shader_float8` (Khronos FP8) and FP8 cooperative-
  matrix ops via `VK_NV_cooperative_matrix2`. Both are supported
  on Ada. So hardware FP8 matmul on Vulkan goes in the V.3 list
  alongside other matmul variants — write it generic over T and
  let Slang specialize for the FP8 path on supported hardware,
  with a CPU-or-emulated-f32 fallback for runtime feature-check
  failures.
- **Vulkane (the sibling project)**. Vulkane is the Vulkan-side
  analogue of baracuda. For kernel needs, fuel ships Slang sources
  in `fuel-kernels-source` rather than waiting on vulkane. Vulkane
  is relevant for **device / context / driver-API** wrapping
  (similar to `baracuda-driver`'s role), not for kernel-level work.

---

## Recommendation

Start with **V.0.1** (crate rename `fuel-vulkan-backend` →
`fuel-vulkan-backend`). It's bounded, mechanical, removes a
naming-asymmetry papercut, and produces no ambiguity later when
V.1's new modules need a home that mirrors `fuel-cuda-backend`.

Then **V.1** (foundation). Until the binding-table dispatch is
wired for Vulkan, all the Slang in the world has no consumer.
V.1 establishes the path; V.2 fans out mechanically; V.3 is where
the actual kernel-authoring work happens (write generic
`T:IFloat` kernels per V.0.2's audit conclusion); V.4 finishes
the E.3.3.D retirement.

Once V.1 lands, V.3 work can start in parallel with V.2 since they
touch different code (V.2 = dispatch registrations; V.3 = Slang
sources + new SPIR-V).

The two work units in V.3 that unblock E.3.3.D's full retirement
are **WriteSlice (Slang)** and **Contiguize signed-stride extension**.
If the priority is "finish E.3.3.D entirely as fast as possible,"
sequencing is:

- V.0.1 (crate rename)
- V.1 (foundation, any op)
- V.3.1 (WriteSlice Slang — write generic, get all dtypes for free)
- V.3.2 (Contiguize signed-stride extension)
- V.4.1 (LlamaModel::forward_with_kv_context Vulkan support)
- V.4.2 (Vulkan example bin migration)
- V.4.3 (legacy retirement)

That's a ~5–7 session path from V.0.1 to E.3.3.D's full cleanup,
with the broader Vulkan op-surface catch-up (V.2 + the rest of
V.3) tracked separately.

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

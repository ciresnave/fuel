# Phase 8 Tier 0 audit — fuel-flash-attn / fuel-flash-attn-v3

## TL;DR

Both existing crates are **CUDA-only Dao-AILab kernel ports targeting the
eager `fuel::Tensor` API (not the lazy graph executor)**, with shape-
specialized .cu files (37 + 67 = 104 kernel translation units), CUTLASS
as a build-time dependency, and minimal Rust glue. They cover sm80
(Ampere FA-v2) and sm90 (Hopper FA-v3) respectively, gated behind
`fuel-transformers/flash-attn` Cargo feature, used today by Mimi
(`fuel-transformers/src/models/audio/mimi/transformer.rs`) and MMDiT
(`fuel-transformers/src/models/diffusion/mmdit/blocks.rs`) when
`use_flash_attn=true`. **They do not work with the lazy graph
executor** — calls go through `fuel::Tensor::storage_and_layout()`
which is the eager-mode storage API, not `fuel_graph::Tensor`.

**Recommendation: supersede, don't refactor.** Tier 1 (CPU reference)
and Tier 2 (Slang) should land as fresh code, expressed as a new
`Op::FlashAttn` in the lazy IR with backend dispatch. The existing
crates can stay around as CUDA Tier 3 implementations once Tier 1/2
land; their kernels are usable but their **Rust shape is wrong for
the lazy world** and refactoring the Rust layer in place would be more
work than rewriting. The .cu kernel files themselves have no equivalent
elsewhere — those carry forward as Tier 3 CUDA assets.

---

## Crate inventory

### fuel-flash-attn (FA-v2 sm80)

- **Lines of code**: 990 (lib.rs) + 53 (ffi.rs) + 119 (build.rs) Rust;
  37 .cu kernel files + 17 .h headers; CUTLASS pulled at commit
  `7d49e6c7e2f8896c47f586706e67e1fb215529dc` via cudaforge's
  `with_cutlass`.
- **Target backend**: CUDA only (compute capability sm80 — Ampere). Each
  kernel TU is one `(head_dim, dtype, causal)` specialization:
  - head_dim ∈ {32, 64, 96, 128, 160, 192, 224, 256, 512} (9 sizes)
  - dtype ∈ {fp16, bf16} (2)
  - causal ∈ {true, false} (2)
  - 9·2·2 = 36 forward kernels + `flash_api.cu` glue = 37.
- **Public API surface** (10 entry points): `flash_attn`,
  `flash_attn_windowed`, `flash_attn_alibi`, `flash_attn_alibi_windowed`,
  `flash_attn_alibi_windowed_softcap`, plus `flash_attn_varlen` family
  (5 more). Forward pass only — no backward.
- **Tensor API**: eager `fuel::Tensor`. The crate's
  `cuda_fwd_t<T>(q: &fuel::CudaStorage, ...)` reaches into eager-world
  storage handles and computes raw `CUdeviceptr` values to feed the
  FFI. There is no lazy-graph `Op::FlashAttn` — these can't be plugged
  into a `fuel-graph` realize call without an adapter.
- **Tests**: 3 `#[test]`s in `tests/flash_attn_tests.rs`, all gated on
  `device = fuel::cuda_backend::new_device(0)`. They build small tensors via the eager
  API, run a Rust-side reference attention (`fa_acausal` /
  `fa_acausal_softcap`) and compare to the FFI output. No CI runner
  picks these up unless someone adds CUDA-aware test infrastructure
  (the current `ci_cuda.yaml` runs `cargo test --features cuda`,
  which would compile this crate but doesn't filter to its tests
  specifically).
- **Build cost**: 37 nvcc invocations × CUTLASS compile-heavy templates;
  uses 50% of available threads by default. The build script even
  carries a `FUEL_FLASH_ATTN_BUILD_DIR` env var to cache the artifact
  across rebuilds — itself a tell that compile time is painful.

### fuel-flash-attn-v3 (FA-v3 sm90)

- **Lines of code**: 943 (lib.rs) + 64 (ffi.rs) + 185 (build.rs) Rust;
  67 .cu kernel files; vendored CUTLASS subdirectory (committed to the
  repo, with `exclude = ["cutlass/{docs,test,examples,tools,media}/**"]`
  in Cargo.toml).
- **Target backend**: CUDA sm90 only (Hopper — TMA + WGMMA). Kernel
  matrix is wider than v2:
  - head_dim ∈ {64, 128, 256, 512} (4 sizes)
  - dtype ∈ {fp16, bf16} (e4m3 paths commented out in build.rs)
  - GQA ratios ∈ {1, 2, 4, 8, 16, 32} as separate specializations
  - Backward kernels enumerated in build.rs but commented out — also
    forward-only at the moment.
- **Public API surface**: same shape as v1 (8 entry points), but no
  softcap helper outside the v1-side aliased family.
- **Tensor API**: same eager `fuel::Tensor` story. Same lazy-incompatible
  shape.
- **Tests**: 3 `#[test]`s in `tests/flash_attn_tests.rs`, structurally
  identical to the v1 crate.
- **CUTLASS in-tree**: this version vendors the whole CUTLASS source
  inside `cutlass/`. Cleaner from a reproducibility perspective (no
  network fetch at build) but adds substantial repo size.

---

## Phase 6a state of the world (what callers expect today)

Every Phase 6a anchor model expresses attention as `Tensor::matmul +
SoftmaxLastDim + Tensor::matmul` on the lazy graph:

```
scores = q @ k^T
scores = scores + mask
attn   = softmax(scores)
out    = attn @ v
```

This is the "naive attention" Phase 8 wants to replace. The lazy graph
executor walks these nodes via `eval_node` and dispatches each to a
backend's `matmul` / `softmax_last_dim` / `binary` arms. There is **no
hook** in this dispatch path for flash-attention today. Adding one
needs a new `Op::FlashAttn` (with the parameters: softmax_scale,
causal, window_size, alibi_slopes, softcap) + a `GraphBackend` trait
method + an executor arm.

The two existing crates' eager-mode plumbing can't satisfy that
contract without adapting to the lazy world. The kernels' .cu files
themselves are Op-shape-agnostic — they want pointers + strides + a
config struct. So the kernels can be reused; the Rust glue around them
needs replacement.

---

## Decision per Phase 8 tier

### Tier 0 — Audit (this document) ✓

### Tier 1 — CPU reference (Pure Rust)

**Fresh code, lives in a new crate.** Most likely
`fuel-attention-ref` next to `fuel-reference-backend`, exporting a
`flash_attention_forward<T: Float>(q, k, v, params) -> output` plain-
Rust function. Becomes the oracle for Tier 2/3.

The existing crates contribute the Rust-side fallback at
`tests/flash_attn_tests.rs::fa_acausal` (lines 19-29) — that's the
correctness oracle pattern Tier 1 should follow, just promoted from
test-only to a public function. ~50 lines of Rust the new crate can
copy verbatim and extend with windowing + alibi.

### Tier 2 — Portable GPU (Slang)

**Fresh code.** New `.slang` source in `fuel-kernels-source/kernels/`,
SPIR-V output via the existing `fuel-vulkan-kernels` build pipeline.
Targets Vulkan first (as ROADMAP §1593-1610 spells out), CUDA PTX as
a free-bonus output if Slang's experimental backend cooperates.

The existing FA-v2 .cu files cannot be machine-translated to Slang.
They're CUTLASS-template-heavy and rely on PTX-level primitives Slang
doesn't expose (`cp.async`, `mbarrier`, `wgmma`). A Slang FA-v2
implementation is a separate authoring effort — the algorithm is
public, the ROADMAP correctly sizes this as Tier 2.

### Tier 3 — Hand-tuned CUDA / Hopper+

**Refactor in place is the right call here.** Once Tier 1 and Tier 2
exist, the existing `fuel-flash-attn` (sm80) and `fuel-flash-attn-v3`
(sm90) become Tier 3 CUDA assets. Their .cu files **stay**; the Rust
glue gets replaced to plug into the new `Op::FlashAttn` →
`CudaBackend::flash_attn` dispatch path. The migration looks like:

1. Add `Op::FlashAttn { softmax_scale, causal, window_size_left,
   window_size_right, softcap }` to `fuel-graph::Op`.
2. Add `Tensor::flash_attn` builder.
3. Add `GraphBackend::flash_attn(q, k, v, alibi_opt, params)` trait
   method with default-bail.
4. Add the executor arm calling `self.backend.flash_attn(...)` and
   falling back to naive attention on Err.
5. `fuel-cuda-backend::CudaBackend::flash_attn` becomes the override that
   calls the existing `flash-attn` FFI — but goes through
   `CudaStorage` / `&CudaSlice<T>` (lazy storage), not eager
   `fuel::CudaStorage`. The `cuda_fwd_t<T>` body in the existing crate
   is the recipe to translate.
6. Drop the existing `flash_attn`/`flash_attn_windowed`/etc. eager-mode
   public API. Keep them around for one release as deprecated thin
   shims that build a one-node lazy graph and realize on a CPU-managed
   CUDA executor, if backwards-compat for Mimi/MMDiT matters.

Estimated cost: ~2 days for steps 1-4 (parallels the conv2d / conv-
transpose2d work just done), ~1-2 days for step 5 per backend version
(v1 sm80 + v3 sm90).

### Tier 4 — v4 concepts

Per ROADMAP §1632-1648, this is research-flavoured and follows Tier
1-3. No Tier 0 input on this — the existing crates don't ship v4.

---

## Concrete recommendation

1. **Park the existing crates** (don't delete) until Tier 1 + Tier 2
   land. They still work for Mimi/MMDiT eager-mode flow, and that path
   isn't broken.
2. **Build Tier 1 fresh** in a new `fuel-attention-ref` crate. ~1 day
   for the forward; ~1 more day for backward via recompute as ROADMAP
   §1587-1588 specifies.
3. **Build Tier 2 fresh** as a Slang shader + Vulkan/CUDA dispatch.
   Multi-day effort sized at "FA-v2 portable, no warp specialization,
   no TMA"; do not chase v3 features in this tier.
4. **Migrate existing crates to Tier 3** once Tier 1/2 are green. The
   .cu kernels are valuable; the Rust glue gets thrown out and
   replaced with lazy-graph-shaped dispatch.

---

## Out of scope for Phase 8 Tier 0

- Re-evaluating the cudarc → baracuda migration's effect on
  flash-attn's FFI surface: those crates already use baracuda
  (`baracuda-types`, `baracuda-driver`) per their `Cargo.toml`.
  Nothing new there.
- Sizing the v3 e4m3 / FP8 paths — commented out in build.rs and not
  in scope until dtype plumbing in Fuel-core grows e4m3 support.
- AMD ROCm + Apple Metal Tier 3 plans — those land per-arch
  independently per ROADMAP §1620-1626, no audit input needed.

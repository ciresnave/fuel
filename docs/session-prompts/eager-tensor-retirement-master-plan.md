# Eager `fuel_core::Tensor` retirement — master plan

## Context

Fuel's Phase 7.5 direction is "drop eager." The eager `fuel_core::Tensor` type
plus its sibling autograd-tape machinery (`BackpropOp` in
[fuel-core/src/op.rs](../../fuel-core/src/op.rs)) is being removed in favor of
the lazy `LazyTensor` / `fuel_graph::Tensor` graph-building model. The
bridge-file doc comment in [fuel-core/src/lazy.rs:6-24](../../fuel-core/src/lazy.rs#L6-L24)
made this the design intent from day one: `LazyTensor` is the scaffolding to
make the final merge incremental; once every consumer compiles against the
wrapper, the type alias flips and `fuel_core::Tensor` becomes the lazy variant.

This plan is the multi-session sequencing to get there without losing
capability along the way. The 2026-05-30 audit (four parallel research agents)
confirmed the picture; the user has locked the sequencing decisions.

## Audit findings (compressed)

**Magnitude.** 244 files across 9 crates import eager `fuel_core::Tensor`; 35
files use `LazyTensor` (~12% of consumer surface). Concentrations:
- fuel-transformers — 177 files (LLM 94% eager, diffusion 87%, multimodal 77%,
  vision 81%, audio 76%, encoders 83%)
- fuel-nn — 27/27 files eager (entire NN primitives layer)
- fuel-examples / fuel-datasets / fuel-training / fuel-inference — 35 files
- No mixed-mode files (clean separation, safe for incremental migration)

**Three categories of work, not one.**

1. **API surface gap.** `LazyTensor` exposes ~107 methods; eager `Tensor` has
   ~152. The first cut of the gap is ~45 methods, but reading the underlying
   `fuel_graph::Tensor` shows it ALREADY has many of these (`unsqueeze`,
   `argmin_dim`, `triu`/`tril`, `log_softmax_last_dim`, `masked_fill`,
   `index_add`/`scatter_add`, `try_*` variants, `relu_inplace`/`silu_inplace`
   etc., `backward()`). Most "gaps" are missing LazyTensor wrappers, not
   missing graph ops. **Real graph-layer gaps:** pooling/interpolation family
   (8 ops, blocks all vision), general `conv1d` (only `causal_conv1d` exists),
   `*_like` shape-derived factories, keepdim reductions, `stack`/`repeat`/
   `flatten`/`transpose_last_two`/`dot`/`mv`/`var`/`powf`/`elu`/`meshgrid`/
   `eye`, in-place init family (`const_set`/`zero_set`/`scatter_set`).
2. **State mutation patterns.** KvCache `.append()` is the showstopper — 45
   LLM files call it on the eager `fuel_nn::KvCache`. Rotating KV (Mistral
   sliding window, 8 files) needs a new graph Op because modulo indexing
   isn't compositional on a DAG. Mamba ring buffer (2 files) is solvable with
   existing `Op::WriteSlice` plus an API layer. BatchNorm EMA (30+ files,
   training) needs running-stats exposed as outputs. Other patterns
   (gradient accumulation, optimizer updates) live outside the graph and
   don't migrate.
3. **`fuel-nn` is 100% eager.** But the 8 existing lazy ports (
   [lazy_bert.rs](../../fuel-core/src/lazy_bert.rs),
   [lazy_qwen2_moe.rs](../../fuel-core/src/lazy_qwen2_moe.rs),
   [lazy_whisper.rs](../../fuel-core/src/lazy_whisper.rs), etc., plus inline
   LlamaModel/Gemma2Model/PhiModel in lazy.rs) drop fuel-nn entirely — they
   extract weights as flat `Arc<[f32]>` and inline helper fns (`linear`,
   `layer_norm_affine`). So fuel-nn doesn't need rewriting; it retires when
   all consumers stop calling it.

**Translation cookbook exists.** 8-step pattern, validated across 3
read-deep ports. Per-model effort: encoder (BERT-family) 1–2 hr,
decoder-only 2–4 hr, hybrid 4–8 hr, encoder-decoder 4–6 hr. Recurring
gotchas: graph-anchoring (multiple `from_f32` calls create independent graphs
that can't be composed), missing KV cache means O(N²) decode, Conv1d
hand-decomposition in Whisper.

## Locked decisions (2026-05-30)

1. **Phase A first**, Phase B parallel with A, Phase C if possible alongside.
   D–G wait for A and B.
2. **Phase G (training) is required**, not optional. Fuel will be used for
   training.
3. **LazyKvCache: option (b)** — functional `cache = cache.append(k, v) ->
   new_cache` returning a new `LazyTensor`-backed cache, not the mutable
   wrapper. More consumer churn, cleaner graph semantics, better long-term.
4. **Rotating KV cache: new Op variant.** Eager is going away soon; don't
   leave Mistral-class on the eager fallback.
5. **Diffusion + multimodal in scope** — not "LLM first, defer the rest."
6. **No external-memory GPU sharing** (VK_KHR_external_memory etc.) — keep
   cross-vendor as `HostStaging` per the picker-audit.

## Phase sequence

| Phase | Work | Sessions | Sequencing | Status |
|-------|------|----------|------------|--------|
| **A** | Close API gap on LazyTensor / `fuel_graph::Tensor` | 8–12 | First; blocks all consumer ports | **A.1–A.5 shipped 2026-05-30** + most A.x deferred items |
| **B** | LazyKvCache option (b) | 1–2 | Parallel with A | **Shipped 2026-05-30** (commit `37ea082a`) |
| **C** | Rotating-window KV cache (new Op) | 1 | Parallel with A+B if attention allows | Awake review (new Op variant) |
| **D** | LLM lazy ports (~14 models) | 7–10 | After A+B complete | Pending |
| **E** | Vision/multimodal lazy ports (~25 models) | 8–12 | After A (pooling), parallel with D | Pending |
| **F** | Diffusion lazy ports (~34 files) | 8–12 | After A + any new attention ops | Pending |
| **G** | Training-path migration (BatchNorm EMA, optimizer) | 2–4 | After at least one model trains end-to-end | Pending |
| **H** | Delete eager Tensor + retire fuel-nn | 2–3 | Last; only after every consumer migrated | Pending |

**Total: 35–55 focused sessions.** Multi-month effort, parallelizable per
phase.

### 2026-05-30 overnight session ship summary

Six commits, all additive (zero backend-file touches; chosen to avoid
merge conflicts with the user's parallel in-place-ops dtype-expansion
work in the same fuel-storage / fuel-cpu-backend / fuel-vulkan-backend
trees). 100 new tests across all phases; full fuel-core lib suite at
274/274 green.

- `b1dad029` — Phase A.1–A.5 wrapper / composite push (59 tests)
- `37ea082a` — Phase B LazyKvCache option (b) (8 tests)
- `1453cb7c` — A.x meshgrid (4 tests)
- `befd1a03` — A.x narrow/chunk/get/multi-dim/rand (20 tests)
- `5f5eff87` — A.x arange/linspace/norm (6 tests)
- `b322dcdd` — A.x pad_with_zeros (3 tests)

Phase A sub-phases A.6 (general conv1d), A.7 (pooling/interpolation),
A.8 (signature harmonization), A.9 (in-place init), and Phase C
(rotating KV) all share the same blocker: they need changes to backend
files the user is actively modifying. They're deferred to the user's
awake review.

## Phase A subdivision

Ordered by additive-first → breaking-changes-last, so consumer-side review
load grows monotonically.

### A.1 — Trivial wrapper additions (no new graph ops)

Methods that exist on `fuel_graph::Tensor` but aren't exposed on
`LazyTensor`. Pure delegation; one PR.

- `unsqueeze(dim)`, `try_unsqueeze(dim)`
- `try_permute(axes)`, `try_transpose()`, `try_broadcast_to(shape)`,
  `try_reshape(shape)`
- `triu(diagonal)`, `tril(diagonal)`
- `log_softmax_last_dim()`
- `argmin_dim(dim)`
- `masked_fill(mask, value)`
- `index_add(dim, indices, src)`, `scatter_add(dim, indices, src)`
- `relu_inplace`/`silu_inplace`/`gelu_inplace`/`tanh_inplace`/
  `sigmoid_inplace`/`affine_inplace` (already on graph; Phase 4-5 shipped)
- `const_f64_like`, `const_u32_like`, `const_i64_like`,
  `const_placeholder_like` (already on `fuel_graph::Tensor`)
- `on_device(loc)`, `move_to_device(loc)`, `copy_to_device(loc)`,
  `release()` — device-residency control
- `backward()` — autograd entry point (currently only reachable through the
  `graph_tensor()` escape hatch)

### A.2 — Composite primitives expressible from existing ops

- `stack(tensors, dim)` — `unsqueeze` + `concat`
- `repeat(dims)` — `broadcast_to` after `unsqueeze` + reshape
- `flatten()`, `flatten_to(end_dim)`, `flatten_from(start_dim)`,
  `flatten_all()` — wrappers over `reshape`
- `transpose_last_two()` — `permute` with last two axes swapped
- `unsqueeze` for multi-dim variants if eager has them

### A.3 — Keepdim reductions (new graph ops or wrap)

Eager has `sum_keepdim`/`mean_keepdim`/`max_keepdim`/`min_keepdim`. Lazy has
only the squeezed `*_dim` variants. **Decision needed:** add `_keepdim`
variants as new Op variants, or post-compose `_dim` + `unsqueeze`? The
latter loses gradient-flow context; the former requires more graph plumbing.
**Recommendation:** new Op variants. Document in A.3 PR.

- `sum_keepdim(dim)`, `mean_keepdim(dim)`, `max_keepdim(dim)`,
  `min_keepdim(dim)`
- `var(dim)`, `var_keepdim(dim)` — variance, needed for LayerNorm
  variants and statistics
- `argmin_dim` already exists on graph; just expose

### A.4 — Missing scalar/composite ops

- `dot(other)` — 1D × 1D
- `mv(other)` / `matvec(other)` — matrix × vector
- `broadcast_matmul(other)` — explicit broadcasting matmul
- `powf(exponent)` — scalar float exponent (lazy has `powi` int and `pow`
  tensor)
- `elu(alpha)` — exponential linear unit
- `affine(mul, add)` — `mul * x + add` without in-place semantics
- `scale_and_shift(scale, shift)` — alias / sibling

### A.5 — Factory `*_like` family

- `ones_like()`, `zeros_like()` — shape-derived constructors
- `full(shape, value, dtype, device)` — fill with constant
- `from_iter(iter, shape, device)` — iterator constructor
- `eye(n, dtype, device)` — identity matrix
- `tril2(n, dtype, device)`, `triu2(n, dtype, device)` — triangular masks
- `meshgrid(tensors)` — coordinate grid
- `rand_like()`, `randn_like()` — random initialization with shape

### A.6 — General conv1d

`LazyTensor` only has `causal_conv1d` (Mamba-specific). Need general
`conv1d(weight, bias, params)` matching eager's signature.
[fuel-graph/src/registry/causal_conv1d.rs](../../fuel-graph/src/registry/causal_conv1d.rs)
is the registry pattern reference.

- New `Op` variant: `Op::Conv1D`
- Registry entry mirroring `conv2d.rs` shape
- CPU + CUDA + Vulkan dispatch (CPU first; baracuda has `conv1d` kernels)
- Wrapper: `LazyTensor::conv1d(weight, bias, params)`

### A.7 — Pooling and interpolation family

Highest-complexity Phase A piece. Eight ops, blocks every vision/diffusion
model. Each needs: graph Op variant, registry entry, CPU/CUDA/Vulkan
dispatch.

- `avg_pool2d(kernel_size, stride)`, `avg_pool2d_with_stride(...)`
- `max_pool2d(kernel_size, stride)`, `max_pool2d_with_stride(...)`
- `interpolate1d(target_size)`, `interpolate2d(target_size)`
- `upsample_nearest1d(scale)`, `upsample_nearest2d(scale)`
- `upsample_bilinear2d(scale)`, `upsample_bilinear2d_with_scale(...)`

Sequencing within A.7: pool ops first (used by conv-backbone models),
interpolation second (used by UNet/segmentation/upscaling).

### A.8 — Signature harmonization

Breaking-change PR. Save for last in Phase A so additive work lands cleanly
first. Goal: every eager `Tensor` method that LazyTensor exposes has
**identical signature**, so the eventual type alias flip touches zero
consumer code.

Known divergences:
- `permute(dims: D: Dims)` (eager, trait-bounded) vs `permute(axes: &[usize])`
  (lazy) — pick `&[usize]` (already exists on graph as `try_permute`)
- `matmul` — eager returns `Result<Self>`, lazy returns `Self`. Pick `Result`
  semantics or document the divergence.
- `pow`, `cast`, `to_dtype` — same Result-wrapping question
- `to_dtype` vs `cast` — pick one name and add an alias for the other

### A.9 — In-place init family

`const_set`, `zero_set`, `one_set`, `scatter_set` — used by eager weight
loading patterns. The lazy ports today work around this by loading weights
as `Arc<[f32]>` directly and constructing tensors via `from_f32`. **Decision
needed:** keep the Arc-flat pattern (current cookbook), or build true
in-place init on top of Phase 4-5 in-place infrastructure? The Arc-flat
pattern is cleaner for inference; in-place init becomes relevant for
training scenarios where weights mutate.

**Recommendation:** defer until Phase G (training) and revisit.

## Phase B detail — LazyKvCache (option b, functional)

Per locked decision: `cache.append(k, v)` returns a new `LazyKvCache`, not
mutates in place. Internally appends `Op::WriteSlice` nodes to a graph-held
buffer; the returned cache carries the new logical sequence length.

Sketch:

```rust
pub struct LazyKvCache {
    k_buffer: LazyTensor,   // [n_layers, max_seq, n_kv_heads, head_dim]
    v_buffer: LazyTensor,   // same shape
    current_seq_len: usize,
}

impl LazyKvCache {
    pub fn new(max_seq_len: usize, ...) -> Self;
    pub fn append(self, layer: usize, k: &LazyTensor, v: &LazyTensor) -> Self;
    pub fn k(&self, layer: usize) -> LazyTensor; // slice [:current_seq_len]
    pub fn v(&self, layer: usize) -> LazyTensor;
}
```

Consumer impact: 45 LLM files rewrite their cache-handling code. Pattern is
mechanical (capture the returned cache, pass it forward) but touches every
forward pass.

Existing reference: [fuel-core/src/inference_context.rs](../../fuel-core/src/inference_context.rs)
already has lazy-side cache machinery — review before designing
`LazyKvCache` to avoid duplicating it.

## Phase C detail — Rotating KV cache

Sliding-window attention (Mistral / sliding-window Qwen / Phi-3 variants)
overwrites old K/V entries in a fixed-size ring. Modulo-indexed writes
aren't compositional on a DAG; this needs first-class graph support.

New op variant:
```
Op::WriteSliceRotating {
    target: NodeId,
    src: NodeId,
    position: NodeId,    // dynamic position; mod is implicit
    modulus: usize,      // window size
    stride: usize,
}
```

CPU + CUDA + Vulkan dispatch. Tests for: append-past-modulus correctly
overwrites slot 0; sequence boundary conditions; integration with
LazyKvCache to give a rotating variant.

## Phases D, E, F — model ports

Use the established cookbook from
[fuel-core/src/lazy_bert.rs](../../fuel-core/src/lazy_bert.rs) /
`lazy_qwen2_moe.rs` / `lazy_whisper.rs`. Eight steps:

1. Flatten eager modules into a single `*Weights` struct with `Arc<[T]>` fields
2. Replace VarBuilder with `load_from_mmapped(safetensors, cfg)`
3. Constructor: `pub fn new(cfg, weights) -> Self` (no Result, no VarBuilder)
4. Replace `Module` trait impl with explicit `pub fn forward(...)`
5. Anchor the graph once at forward entry; thread anchor through via
   `const_*_like` for everything else
6. Replace eager ops with lazy equivalents; remove `?` propagation where lazy
   returns plain `Self`
7. Call `realize_*()` once at the end (or in the generation loop)
8. Add zero-weight shape test + oracle parity gate

**Port order within D (LLM, ~14 models):**
- LLaMA family (Llama, CodeLlama variants) — most reused pattern
- Mistral / Mistral-derivatives (uses sliding window — needs Phase C first)
- Qwen family (Qwen, Qwen2, Qwen3 — Qwen2-MoE already done)
- Mamba pair (depends on ring-buffer API atop Op::WriteSlice — addressable
  during A.1 or as its own micro-phase)
- Phi family (Phi-2, Phi-3 — Phi-3 needs sliding window → Phase C)
- Gemma family (Gemma2 already done; verify Gemma)
- Falcon, ChatGLM, others — verify each picks a parent pattern

**Port order within E (vision/multimodal):**
- SAM family (segmentation, light deps)
- DepthAnything (uses interpolation — needs A.7)
- CLIP / ViT-family (used by multimodal)
- YOLOv8 already done; YOLO variants follow
- LLaVA / Pixtral / Qwen-VL (multimodal — need vision encoder + LLM core both
  done)

**Port order within F (diffusion):**
- Stable Diffusion text encoder / UNet / VAE already done; verify
- Flux (DiT-style, needs cross-attention variants)
- MMDiT, Würstchen, Z-Image — survey their attention variants before
  starting

## Phase G detail — training

Inference-only patterns (param updates, BatchNorm `forward_eval`, gradient
accumulation) live outside the graph and don't need migration.

Training-only patterns that DO need work:
- **BatchNorm EMA** — 30+ files; running mean/var mutation via
  `var.set(&new_value)`. Refactor: `forward_train` returns
  `(output, new_running_mean, new_running_var)` and the caller threads
  state. Or: introduce a graph-native EMA op.
- **Optimizer step** — currently mutates `Var` tensors directly. Lazy path:
  graph computes new param value; optimizer applies via `Var::set` (eager
  mutation outside the graph). No graph migration needed if Var stays
  eager; otherwise full refactor.
- **Gradient accumulation across microbatches** — current eager pattern is
  `grad.add_inplace(...)`; lazy needs `Op::WriteSlice`-based accumulator or
  fresh-graph-per-microbatch with explicit `reduce_sum` at end.

Surfacing decision for the Phase G session: does Var become a lazy graph
node, or stay as a host-side eager mutator? Both are viable; the choice
affects how much of fuel-nn's training path can collapse.

## Phase H detail — eager deletion

1. Verify zero remaining `use fuel_core::Tensor` imports in consumer code
2. Mark eager `Tensor` deprecated with a redirect to `LazyTensor`
3. Flip the `pub type Tensor = LazyTensor` alias
4. Delete eager Tensor methods one file at a time, watching CI
5. Retire `BackpropOp` and the eager autograd tape
6. Retire `fuel-nn` once its callers are gone (or migrate fuel-nn to
   `LazyTensor` if any internal users remain)
7. Final cleanup PR: remove `lazy.rs` bridge module's wrapper-only methods
   (delegating to the underlying type now becomes redundant)

## Open questions deferred to per-phase sessions

- **Phase A.3:** keepdim as new Op vs `_dim` + `unsqueeze` post-compose?
- **Phase A.5:** factory devices — do `*_like` factories inherit device from
  the anchor tensor, or take an explicit `&Device`? (Existing `const_*_like`
  inherits; should `ones_like` follow?)
- **Phase A.7:** does pooling get a unified `Op::Pool2D { mode: Avg | Max }`
  or distinct `Op::AvgPool2D` / `Op::MaxPool2D`? (Mode-tagged is more
  extensible; distinct keeps each op's gradient simple.)
- **Phase A.8:** harmonize toward eager's Result-returning signatures or
  toward lazy's plain-Self signatures? (Eager wins on "errors at graph
  build are useful"; lazy wins on ergonomics.)
- **Phase B:** does `LazyKvCache` know about layers (single `append(layer,
  k, v)`) or is it per-layer (caller holds `Vec<LazyKvCache>`)? (Per-layer
  is more orthogonal; combined is more ergonomic.)
- **Phase D Mamba:** ring-buffer state — `Op::WriteSlice` against a
  per-layer state tensor, or a higher-level `Op::MambaState`-style fused
  variant? (Probably WriteSlice first, fuse later.)
- **Phase G:** Var as graph node or host-side eager mutator?
- **Phase H:** does `pub type Tensor = LazyTensor` ship before all bridge
  wrappers are stripped, or as the final step?

## References

- [project_phase_7_5_core_simplification](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_5_core_simplification.md)
  — "drop eager" direction
- [project_phase_7_5_work_item_b2_complete](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase_7_5_work_item_b2_complete.md)
  — eager Tensor factories on node-handle mode (prerequisite)
- [project_inplace_ops_complete](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_inplace_ops_complete.md)
  — Phase 4-5 in-place infrastructure (basis for Phase A.9)
- [fuel-core/src/lazy.rs:6-24](../../fuel-core/src/lazy.rs#L6-L24) —
  bridge-file design rationale
- [mamba-eager-to-lazy-migration.md](./mamba-eager-to-lazy-migration.md) —
  Mamba-specific port prompt (now subsumed by this master plan)

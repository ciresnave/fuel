---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                                  # maps to BackendId::Metal
  kernel_source: "metal-msl"                       # the BindingEntry.kernel_source tag (FuelNative-class, §4.11)
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                    # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — reduce / norm / RoPE kernel contracts

The Metal `reduce.metal` compute family from `fuel-metal-kernels`: the block-reduction kernels
(`fast_<op>`), the last-dim online `softmax`, the affine-capable `rmsnorm` / `layernorm`, and the
three rotary-embedding variants (`rope_i` / `rope` / `rope_thd`). The MSL sources live in
`fuel-metal-kernels/metal_src/reduce.metal`; the Rust dispatch wrappers (param packing, variant
selection, `BufferOffset` plumbing, validation) live in `fuel-metal-kernels/kernels/reduce.rs`; the
live wiring that selects contiguous-vs-strided and computes byte offsets lives in
`fuel-metal-backend/storage.rs` (`reduce_op()`).

**Family-wide facts (each kernel's section overrides where its inventory entry differs):**

- **Two layout regimes, by kernel.** Only the `fast_<op>` block reduction has a true strided
  variant: `fast_<op>_<t>` indexes linearly (`indexer_t<uint,false>`, contiguous-only) while
  `fast_<op>_<t>_strided` uses a templated `strided_indexer` (specialized rank 1-4, fallback loop
  for rank > 4) that walks **arbitrary strides** over the reduced region. The backend `reduce_op()`
  takes the strided path **only** when the reduction sub-shape is non-contiguous, otherwise the
  contiguous path. The `softmax` / `rmsnorm` / `layernorm` / `rope_*` kernels are **contiguous-only**
  — they use the linear `indexer_t<uint,false>` (or fixed dim arithmetic for RoPE) with **no strided
  variant** — so any non-contiguous producer must be contiguized by the planner first (an
  `Op::Contiguize` FKC kernel, §4.3, whose cost is summed per §4.4).
- **Offset-capable via `BufferOffset` everywhere in this file.** Every `call_*` here threads its
  buffers through `BufferOffset { buffer, offset_in_bytes }` (`utils.rs`, set via `set_buffer(pos,
  buf, offset)`); the backend computes `offset_in_bytes = layout.start_offset() *
  dtype.size_in_bytes()`. So every input operand declares `start_offset: accepted` — a non-zero view
  base is consumed natively, **not** routed through a contiguize. (Contrast the Vulkan elementwise
  family, which is offset-incapable.)
- **`reverse_strides: rejected` everywhere.** None of these kernels walks a signed (negative)
  stride. The `fast_<op>` strided indexer decodes coords with **unsigned** strides; the
  contiguous/RoPE kernels do fixed positive-stride arithmetic. A flipped (`Op::Flip`) view feeding
  any of them is normalized to a non-negative copy by an upstream movement kernel before dispatch —
  a `flipped` operand is never handed directly to these kernels (§4.1.1).
- **Output: fresh contiguous, fully overwritten (with the noted exceptions).** Reduction / softmax /
  norm / RoPE all allocate a fresh contiguous output via `device.new_buffer` and write densely
  (`out[tid]` / one element per threadgroup). No input/output aliasing; the kernel reads no prior
  output content (`aliasing: none`).
- **Accumulation precision differs by kernel and IS load-bearing.** The `fast_<op>` value
  reductions (Sum/Mul/Max/Min) and the RoPE rotations compute **in the element type `T`** (no f32
  widening); the softmax normalizer (`d` accumulator), the `rmsnorm` mean-of-squares (`RMS<float>`),
  and the `layernorm` mean/M2 (`LayerNormValue<float>`) accumulate in **f32** then narrow on store
  for f16/bf16. Each section's `precision.notes` states which regime applies. None of these is
  bit-stable across hardware (simd-shuffle tree reductions are scheduler-/lane-order-dependent on the
  GPU); they ARE deterministic on the same hardware (fixed threadgroup geometry, no atomics).
- **Cost is `judge_measured` for every kernel in this file** — the Judge bootstraps and refines the
  empirical coefficients (§4.4). These are GPU dispatches whose launch overhead, occupancy, and
  bandwidth are device-specific and not author-derivable, so `overhead_ns` and any absolute timing
  are left to the Judge. Where the op genuinely admits a structural FLOPs/bandwidth shape it is
  recorded as the measurement's prior: a reduction reads every input element once (`flops ≈ n`,
  bandwidth-bound on `(n + out) · dtype_bytes`); a normalization is `O(n)` over `n` elements with a
  fresh `n`-element output; RoPE is pure elementwise (`flops ≈ n`). `provenance: judge_measured` is a
  first-class, visible marker, not a hidden gap (§4.4 / §10.8a). `n` denotes the product of all
  output elements; `last` (= `el_per_block` / `elements_to_sum`) denotes the last-dim extent;
  `dtype_bytes` the element width.
- **Live-wiring caveat.** Some of these entry points exist in the kernels crate but are **not wired
  into the live `fuel-metal-backend`** (the inventory flags `softmax` as having no wired consumer,
  and the `rope_*` variants as wired only by the retired `_fuel_nn_retired/rotary_emb.rs`). The
  contracts below describe the **kernel as built**; whether the live backend currently routes to it
  is a wiring fact, not a contract fact, and is noted per section.

---

## fast_reduce  (block reduction over the contiguous trailing dims)

Block reduction (Sum/Mul/Max/Min value, ArgMax/ArgMin index) over contiguous trailing dims; reduced dims removed.

The contiguous block-reduction kernel `fast_<op>_<t>` (`reduce.metal:578`, `reduce.rs:6`), dispatched
by the backend `reduce_op()` (`storage.rs:293`) when the reduction sub-shape is contiguous. One
kernel family backs six logical ops selected by the host kernel name: the value-returning
**Sum / Mul / Max / Min** (`OpKind::SumReduce` / `MaxReduce` / `MinReduce`, output =
input dtype — the **Mul** reduction is MSL-level only and has **no backing OpKind today**
[consumer-ahead]) and the index-returning **ArgMax / ArgMin**
(`OpKind::ArgMaxDim` / `ArgMinDim`, `impl_arg_reduce`, output = **U32** index;
`storer<indexed<T>>` writes the `.i` field). The reduced region is indexed **linearly**
(`indexer_t<uint,false>`); one threadgroup folds one group via a simd-shuffle tree reduce and emits
one output element. Output shape = the source with the reduced dims removed (one element per
threadgroup). Min/Max use `fast::min`/`fast::max` for floats; arg ops break ties to the **lowest
index** (the `<` compare includes `i < i` so the first winner survives). The backend errors on an
empty tensor for min/max/arg. Sum/Mul/Min/Max accumulate in the element type `T` (no f32 widening).
Offset-capable on src and dst via `BufferOffset`.

> This section is the representative contract for all six ops (one strided-or-contiguous kernel
> family, op selected by the dispatch name); the per-op dtype keys are driven by the `dtypes` list
> plus the index-output rule below. Sum/Mul/Max/Min key `[in: T, out: T]`; ArgMax/ArgMin key
> `[in: T, out: U32]` (the `dtype_rule` branches on the op).

```fkc
kernel: fast_reduce
op_kind: SumReduce              # representative; the same family backs MaxReduce/MinReduce (value, out=T)
                                # and ArgMaxDim/ArgMinDim (index, out=U32) by dispatch name. The MSL-level
                                # Mul reduction has no backing OpKind today [consumer-ahead].
blurb: "Block reduction (Sum/Mul/Max/Min value, ArgMax/ArgMin index) over contiguous trailing dims; reduced dims removed."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_reduce_contiguous"   # name-selected op/dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, U32, U8, BF16, I64]   # f32/f16/u32/u8 always; +bf16/+i64 guarded (inventory)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Reduce              # OpParams::Reduce (primitive namespace)
    fields:
      src_numel:    { kind: usize, note: "total source element count" }
      num_dims:     { kind: usize }
      dims:         { kind: "Vec<usize>", note: "reduced (trailing) axes; removed from output" }
      el_per_block: { kind: usize, note: "= last-dim / reduced-region extent per group" }
      keepdim:      { kind: bool, constraint: "== false (reduced dims removed; no keepdim path)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)                # Sum/Mul/Max/Min; ArgMax/ArgMin -> fixed(U32) (index output)
      shape_rule: reduce(x, dims, keepdim=false)   # source with reduced dims removed; symbolic non-reduced axes preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous     # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost; strided sub-shape routes to fast_reduce_strided instead
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # Judge bootstraps; the FLOPs/bandwidth hint below is the structural prior it refines
  class: reduction
  flops: "n"                        # one fold op per input element (n = src_numel)
  bytes_moved: "(n + out_elems) * dtype_bytes"   # read every input element, write one element per group
  memory: { device_bytes: "out_elems * dtype_bytes", host_bytes: 0, disk_bytes: 0 }   # fresh per-group output

precision:
  bit_stable_on_same_hardware: true   # fixed threadgroup geometry, no atomics; simd-shuffle tree, but same lane order on same HW
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Sum/Mul/Min/Max accumulate in T (NOT widened to f32 for half); Min/Max via fast::min/max; arg ops tie-break to lowest index; simd-shuffle tree reduce -> NOT bit-stable cross-hardware; empty tensor errors for min/max/arg."

determinism: same_hardware_bitwise
```

---

## fast_reduce_strided  (block reduction over arbitrarily-strided trailing dims)

Block reduction (Sum/Mul/Max/Min/ArgMax/ArgMin) over arbitrarily-strided trailing dims; walks strides directly.

The strided sibling `fast_<op>_<t>_strided` (`reduce.metal:578`, `reduce.rs:6`), dispatched by
`reduce_op()` (`storage.rs:293`) **only** when the reduction sub-shape is non-contiguous. Identical
fold semantics to `fast_reduce` (Sum/Mul/Max/Min value, ArgMax/ArgMin → U32 index; tie-to-lowest;
in-`T` value accumulation; empty errors for min/max/arg) except the reduced region is addressed via a
templated `strided_indexer` (specialized for rank 1-4, fallback loop for rank > 4) instead of the
linear indexer — so the source may carry **arbitrary strides** (transposed / non-contiguous reduced
axes), bounded only by `uint` index range. The extra `strides[]` array rides the op-params. Output is
still a fresh **contiguous** per-group buffer. Offset-capable on src and dst via `BufferOffset`.
**This is the kernel that makes the reduction family `handles_strided`**: a strided reduced region is
walked directly, no upstream contiguize.

```fkc
kernel: fast_reduce_strided
op_kind: SumReduce              # representative; same family as fast_reduce, strided indexer
blurb: "Block reduction (Sum/Mul/Max/Min/ArgMax/ArgMin) over arbitrarily-strided trailing dims; walks strides directly."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_reduce_strided"   # name-selected op/dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, U32, U8, BF16, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # strided_indexer specialized rank 1-4, fallback loop for rank > 4
  op_params:
    variant: Reduce              # OpParams::Reduce; strided variant adds the strides[] array
    fields:
      src_numel:    { kind: usize }
      num_dims:     { kind: usize }
      dims:         { kind: "Vec<usize>", note: "reduced (trailing) axes; removed from output" }
      strides:      { kind: "Vec<usize>", note: "per-axis source strides for the strided indexer" }
      el_per_block: { kind: usize, note: "= reduced-region extent per group" }
      keepdim:      { kind: bool, constraint: "== false (reduced dims removed)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)                # ArgMax/ArgMin -> fixed(U32)
      shape_rule: reduce(x, dims, keepdim=false)   # source with reduced dims removed; symbolic non-reduced axes preserved
      layout_guarantee: contiguous              # fresh dense output even though input is strided
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided        # walks arbitrary strides over the reduced region; no fixup, no contiguize
  fast_paths:
    - { when: "any_input_strided", class: reduction }
    - { when: "all_inputs_contiguous", note: "backend prefers fast_reduce (contiguous) in this case" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"                        # one fold op per input element (n = src_numel)
  bytes_moved: "(n + out_elems) * dtype_bytes"   # strided reads, dense per-group writes
  memory: { device_bytes: "out_elems * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "identical numerics to fast_reduce (in-T value fold, fast::min/max, tie-to-lowest, f32-only nothing here); strided indexer (rank 1-4 specialized, >4 fallback loop); simd-shuffle tree -> NOT bit-stable cross-hardware; empty errors for min/max/arg."

determinism: same_hardware_bitwise
```

---

## softmax  (last-dim online softmax)

Numerically-stable last-dim online softmax (Welford MD normalizer); contiguous-only; f32-internal denominator.

The last-dim softmax kernel `softmax_<t>` (`reduce.metal:848`, `reduce.rs:119`, `call_last_softmax`).
Computes a numerically-stable softmax over the last `el_per_block` elements of each row using an
**online (Welford-style) normalizer** carried in an `MD` struct (running max `m` + denominator `d`),
so the row max-subtract and the exponential-sum are fused into a single streaming pass per row. The
denominator accumulator `d` is **f32** (so the normalization is f32-internal and narrows on store for
f16/bf16). **Contiguous-only**: it uses the linear `indexer_t<uint,false>` with **no strided
variant**; the backend passes raw `(input, input_offset)`, so it is offset-capable but the data must
be dense. Output is written densely, same dtype and shape as the input. Implements
`OpKind::SoftmaxLastDim` (the canonical Fuel softmax op; the migrated fused identity is
`FusedOps::SOFTMAX_LAST_DIM`, `FusedOpId(1)`); the last-dim axis is implicit in the shape, so the op
carries no per-instance params beyond the row geometry.

> **Live-wiring caveat (inventory):** the `softmax` entry point exists in `fuel-metal-kernels` but
> **no wired consumer was found in `fuel-metal-backend`**. The contract describes the kernel as
> built; it is registrable, but the live Metal backend does not currently route the softmax op to it.

```fkc
kernel: softmax
op_kind: SoftmaxLastDim
blurb: "Numerically-stable last-dim online softmax (Welford MD normalizer); contiguous-only; f32-internal denominator."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_last_softmax"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1.."                # rank >= 1; last dim (= el_per_block) is the softmax axis
  op_params:
    variant: SoftmaxLastDim      # OpParams::SoftmaxLastDim — the last-dim axis is implicit (row geometry only)
    fields:
      src_numel:    { kind: usize, note: "total element count" }
      el_per_block: { kind: usize, note: "= last-dim length (softmax window per row)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)        # element-shape preserved; symbolic extents carried through (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # contiguous-only; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): per-row online max-subtract + exp-accumulate + normalize
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }   # fresh n-element output

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "online (Welford MD) max-subtract softmax; denominator d accumulates in f32 (f16/bf16 narrow on store); exp at f32; NOT bit-stable cross-hardware (transcendental + lane order)."

determinism: same_hardware_bitwise
```

---

## rmsnorm  (RMS norm over the last dim, optional affine)

RMS norm x·rsqrt(mean(x^2)+eps), optional inline alpha scale; contiguous-only; f32 mean-of-squares.

The RMSNorm kernel `rmsnorm_<t>` (`reduce.metal:1058`, `reduce.rs:167`, `call_rms_norm`). Per row,
`out = x · rsqrt(mean(x²) + eps)`, with an **optional inline affine scale** `alpha`: when the `alpha`
buffer is non-null it multiplies the normalized output (`out *= alpha`), read **block-local**
(`alpha[i - offset]`), so `alpha` length must equal `el_per_block` (the last dim). When `alpha` is
null (a null pointer) the scale is skipped. There is **no mean-centering** (unlike LayerNorm). The
mean-of-squares reduction/accumulation is in **f32** (`RMS<float>`), so half I/O is f32-internal and
narrows on store. **Contiguous-only** (`indexer_t<uint,false>`, no strided variant); src and alpha
are offset-capable via `BufferOffset`. Implements `OpKind::RmsNormLastDim` (canonical fused identity
`FusedOps::RMS_NORM_LAST_DIM`, `FusedOpId(3)`).

> **As-built divergence from the affine-free fused op.** The migrated `FusedOps::RMS_NORM_LAST_DIM`
> is deliberately affine-free (`FusedOpParams::RmsNormLastDim { eps }` only — the caller applies
> gamma separately). The Metal kernel **folds the affine `alpha` inline** as an optional second
> input, so this contract declares `alpha` as an **optional `accept.inputs` operand** (an honest
> separate buffer the kernel indexes block-local — NOT a sidecar/gather descriptor and NOT a quant
> scale, so no FDX extension and no `scale_operand`). When `alpha` is absent the kernel is the plain
> affine-free RMSNorm.

```fkc
kernel: rmsnorm
op_kind: RmsNormLastDim
blurb: "RMS norm x·rsqrt(mean(x^2)+eps), optional inline alpha scale; contiguous-only; f32 mean-of-squares."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_rms_norm"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1.."                # rank >= 1; last dim (= elements_to_sum) is the norm axis
    - name: alpha                # OPTIONAL inline affine scale; null pointer -> skip scale
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 1                    # length == el_per_block (last dim); read block-local alpha[i-offset]
      optional: true
      shape_constraint: "dim[0]=x.dim[-1]"   # alpha length == last dim
  op_params:
    variant: NormLastDim         # OpParams::NormLastDim { outer_count, last_dim, eps } (kernel.rs:403);
                                 # shared by RMS-norm and LayerNorm, selected by OpKind (here OpKind::RmsNormLastDim)
    fields:
      outer_count: { kind: usize, note: "rows; total element count = outer_count * last_dim" }
      last_dim:    { kind: usize, note: "= last dim (norm window per row; the Metal `elements_to_sum`)" }
      eps:         { kind: f64, note: "added under the rsqrt; coerced to T internally" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)        # symbolic non-norm axes preserved (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): mean(x^2) reduction + reciprocal-rms scale (+ optional alpha mul)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out (alpha re-read per row when present)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "mean(x^2) accumulated in f32 (RMS<float>); no mean-centering; f16/bf16 narrow on store; optional inline alpha scale; simd-shuffle reduce -> NOT bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## layernorm  (LayerNorm over the last dim, optional affine)

LayerNorm (x-mu)·rsqrt(var+eps), optional inline alpha/beta affine; contiguous-only; f32 mean/M2.

The LayerNorm kernel `layernorm_<t>` (`reduce.metal:1241`, `reduce.rs:225`, `call_layer_norm`). Per
row, `out = (x - mu) · rsqrt(var + eps)`, with **optional inline affine** `alpha` (scale) and `beta`
(shift): both are null-guarded — when present they are applied as `out = out · alpha + beta`, read
**block-local** (`alpha[i-offset]` / `beta[i-offset]`), so each must have length `el_per_block` (the
last dim). The mean and second moment (`m2`) accumulate in **f32** (`LayerNormValue<float>`), so half
I/O is f32-internal and narrows on store. **Contiguous-only** (`indexer_t<uint,false>`, no strided
variant); src, alpha and beta are offset-capable via `BufferOffset`. Implements
`OpKind::LayerNormLastDim` (canonical fused identity `FusedOps::LAYER_NORM_LAST_DIM`, `FusedOpId(4)`).

> **As-built divergence from the affine-free fused op** (same shape as `rmsnorm` above): the
> migrated `FusedOps::LAYER_NORM_LAST_DIM` is affine-free; the Metal kernel folds `alpha`/`beta`
> inline as two optional `accept.inputs` operands (honest separate block-local buffers, no FDX
> extension, no quant scale). When both are absent the kernel is the plain affine-free LayerNorm.

```fkc
kernel: layernorm
op_kind: LayerNormLastDim
blurb: "LayerNorm (x-mu)·rsqrt(var+eps), optional inline alpha/beta affine; contiguous-only; f32 mean/M2."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_layer_norm"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1.."                # rank >= 1; last dim (= elements_to_sum) is the norm axis
    - name: alpha                # OPTIONAL inline scale; null-guarded
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 1                    # length == el_per_block (last dim); block-local alpha[i-offset]
      optional: true
      shape_constraint: "dim[0]=x.dim[-1]"
    - name: beta                 # OPTIONAL inline shift; null-guarded
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 1                    # length == el_per_block (last dim); block-local beta[i-offset]
      optional: true
      shape_constraint: "dim[0]=x.dim[-1]"
  op_params:
    variant: NormLastDim         # OpParams::NormLastDim { outer_count, last_dim, eps } (kernel.rs:403);
                                 # shared by RMS-norm and LayerNorm, selected by OpKind (here OpKind::LayerNormLastDim)
    fields:
      outer_count: { kind: usize, note: "rows; total element count = outer_count * last_dim" }
      last_dim:    { kind: usize, note: "= last dim (norm window per row; the Metal `elements_to_sum`)" }
      eps:         { kind: f64, note: "added under the rsqrt; coerced to T internally" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): mean + variance + normalize (+ optional alpha·+beta)
  bytes_moved: "2 * n * dtype_bytes"   # read x, write out (alpha/beta re-read per row when present)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "mean/M2 accumulated in f32 (LayerNormValue<float>); f16/bf16 narrow on store; optional inline alpha·+beta; simd-shuffle reduce -> NOT bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## rope_i  (interleaved rotary position embedding)

Interleaved RoPE: rotate adjacent even/odd pairs with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b.

The interleaved rotary-embedding kernel `rope_i_<t>` (`reduce.metal:1365`, `reduce.rs:288`,
`call_rope_i`). Rotates **adjacent (interleaved) pairs** `(x[2i], x[2i+1])` using caller-supplied
`cos`/`sin` tables. **Contiguous with fixed dim arithmetic** — no general strider; the kernel does
`bh`/`td`/`stride_b` index math directly. `stride_b > 0` enables a **per-batch** cos/sin table
(`stride_b` is the batch stride into the tables); `stride_b == 0` shares one table across batches.
`src`, `cos`, `sin` and `out` are all offset-capable via `BufferOffset`. Arithmetic is in the element
type `T` (pure elementwise rotation — no reduction, no overflow guard). Output same dtype and shape
as the input, fresh contiguous buffer. Implements the RoPE op (`OpKind::Rope`; canonical fused
identity `FusedOps::ROPE`, `FusedOpId(5)`) in its **interleaved** layout.

> **Live-wiring caveat (inventory):** the three `rope_*` entry points are wired by the **retired**
> `_fuel_nn_retired/rotary_emb.rs`, **not** by the live `fuel-metal-backend`. The contract describes
> the kernel as built; live routing is a wiring fact, noted here. The cos/sin tables are honest
> separate contiguous operands (no FDX extension / gather descriptor).

```fkc
kernel: rope_i
op_kind: Rope
blurb: "Interleaved RoPE: rotate adjacent even/odd pairs with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_rope_i"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # fixed bh/td index math; geometry carried in op_params
    - name: cos
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # cos table; per-batch when stride_b > 0
    - name: sin
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=cos
  op_params:
    variant: Rope                # OpParams::Rope (interleaved variant carries bh/td/stride_b)
    fields:
      bh:       { kind: usize, note: "batch*heads count (outer)" }
      td:       { kind: usize, note: "seq*head_dim per (b,h)" }
      stride_b: { kind: usize, note: "per-batch cos/sin stride; 0 = shared table" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: from_params(src)  # output is src's shape; symbolic seq preserved (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # fixed-dim contiguous; planner contiguizes a strided producer first (cost summed, §4.4)
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): pure elementwise interleaved rotate (no reduction)
  bytes_moved: "2 * n * dtype_bytes"   # read src, write out (cos/sin re-read per element/row)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic per-element rotation, no atomics, fixed geometry
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pure elementwise interleaved (adjacent-pair) rotation; arithmetic in T (no f32 widening); cos/sin caller-supplied, per-batch when stride_b>0; not bit-stable cross-hardware (transcendental tables differ)."

determinism: same_hardware_bitwise
```

---

## rope  (default split-half rotary position embedding)

Default split-half RoPE: rotate (x[i], x[i+half]) with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b.

The default rotary-embedding kernel `rope_<t>` (`reduce.metal:1365`, `reduce.rs:288`, `call_rope`).
The **split-half** form: with `half = d/2`, rotates `(x[i], x[i+half])` pairs across the two halves of
the head dimension (the standard rotate-halves RoPE), using caller-supplied `cos`/`sin` tables.
**Contiguous with fixed dim arithmetic** — no general strider; the kernel does `bh`/`td`/`d`/`stride_b`
index math. `stride_b > 0` enables a per-batch cos/sin table. `src`, `cos`, `sin`, `out` are
offset-capable via `BufferOffset`. Arithmetic in `T` (pure elementwise; no reduction). Output same
dtype and shape, fresh contiguous buffer. Implements `OpKind::Rope` (canonical fused identity
`FusedOps::ROPE`, `FusedOpId(5)`) in its **default split-half** layout. The only difference from
`rope_i` is the pairing (split-half vs interleaved) and the extra `d` (head-dim) param.

> **Live-wiring caveat:** wired by the retired `_fuel_nn_retired/rotary_emb.rs`, not the live
> backend (see `rope_i`). cos/sin are honest separate contiguous operands (no FDX extension).

```fkc
kernel: rope
op_kind: Rope
blurb: "Default split-half RoPE: rotate (x[i], x[i+half]) with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_rope"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # fixed bh/td/d index math; head_dim d in op_params
    - name: cos
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # per-batch when stride_b > 0
    - name: sin
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=cos
  op_params:
    variant: Rope                # OpParams::Rope (default variant carries bh/td/d/stride_b)
    fields:
      bh:       { kind: usize, note: "batch*heads count (outer)" }
      td:       { kind: usize, note: "seq*head_dim per (b,h)" }
      d:        { kind: usize, note: "head_dim; half = d/2 is the rotate-halves split" }
      stride_b: { kind: usize, note: "per-batch cos/sin stride; 0 = shared table" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: from_params(src)  # output is src's shape; symbolic seq preserved (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): pure elementwise rotate-halves (no reduction)
  bytes_moved: "2 * n * dtype_bytes"   # read src, write out (cos/sin re-read per element/row)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pure elementwise rotate-halves (split-half pairing); arithmetic in T (no f32 widening); cos/sin caller-supplied, per-batch when stride_b>0; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## rope_thd  ((b,t,h,d)-layout rotary position embedding)

RoPE for (b,t,h,d) layout: rotate-halves with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b.

The `(b, t, h, d)`-layout rotary-embedding kernel `rope_thd_<t>` (`reduce.metal:1365`,
`reduce.rs:288`, `call_rope_thd`). Same rotate-halves rotation as `rope`, but with index arithmetic
for the **`(batch, seq, heads, head_dim)`** tensor layout (heads and seq transposed relative to the
default), so the four explicit dims `b`/`t`/`h`/`d` are carried in op-params. **Contiguous with fixed
dim arithmetic** — no general strider; `stride_b > 0` enables a per-batch cos/sin table. `src`,
`cos`, `sin`, `out` offset-capable via `BufferOffset`. Arithmetic in `T` (pure elementwise; no
reduction). Output same dtype and shape, fresh contiguous buffer. Implements `OpKind::Rope` (canonical
fused identity `FusedOps::ROPE`, `FusedOpId(5)`) in its **(b,t,h,d)** layout.

> **Live-wiring caveat:** wired by the retired `_fuel_nn_retired/rotary_emb.rs`, not the live
> backend (see `rope_i`). cos/sin are honest separate contiguous operands (no FDX extension).

```fkc
kernel: rope_thd
op_kind: Rope
blurb: "RoPE for (b,t,h,d) layout: rotate-halves with cos/sin tables; contiguous fixed-dim; optional per-batch stride_b."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::reduce::call_rope_thd"   # §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                    # (b, t, h, d); dims carried explicitly in op_params
    - name: cos
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any                  # per-batch when stride_b > 0
    - name: sin
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=cos
  op_params:
    variant: Rope                # OpParams::Rope ((b,t,h,d) variant carries b/t/h/d/stride_b)
    fields:
      b:        { kind: usize, note: "batch" }
      t:        { kind: usize, note: "seq length" }
      h:        { kind: usize, note: "heads" }
      d:        { kind: usize, note: "head_dim; half = d/2" }
      stride_b: { kind: usize, note: "per-batch cos/sin stride; 0 = shared table" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: from_params(src)  # output is src's (b,t,h,d) shape; symbolic t preserved (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  flops: "n"                        # ~O(n): pure elementwise rotate-halves over (b,t,h,d)
  bytes_moved: "2 * n * dtype_bytes"   # read src, write out (cos/sin re-read per element/row)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "pure elementwise rotate-halves over (b,t,h,d) layout; arithmetic in T (no f32 widening); cos/sin caller-supplied, per-batch when stride_b>0; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

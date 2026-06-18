---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                    # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"      # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — softmax / norm / cumsum kernel contracts (family: norm)

AOT-compiled Slang/GLSL compute kernels for the last-dim softmax / RMS-norm / LayerNorm family,
their backward passes, and the prefix-sum (cumsum) op. Kernel sources live in
`fuel-kernels-source/kernels/*.slang`; SPIR-V is committed in `fuel-vulkan-kernels/spv/*.spv` and
registered in the `EMBEDDED` table (`fuel-vulkan-kernels/src/lib.rs:39`). The Rust dispatch
wrappers (param packing, layout gating, validation, route picking) live in
`fuel-vulkan-backend/src/lib.rs`; line numbers below cite the as-built inventory
(`docs/kernel-contracts/_inventory/vulkan.md`).

Cross-cutting facts for every kernel in this file (from the inventory + as-built source):

- **Row-wise last-dim reductions** (softmax / rms / layer norm + their backwards): the flat buffer
  is viewed as `n_rows` rows of `n_cols` contiguous elements; **one workgroup per row** computes
  the reduction with **subgroup reductions** (not a serial scan), then rewrites the row. `cumsum`
  is the exception: it is a per-slice **serial walk along one axis** and is **strided-input
  capable** (rank-4 shape + per-input strides), unlike the contiguous-only norms.
- **Layout (norms + backwards): contiguous-only, zero-offset, row-major.** These kernels carry
  only `n_rows`/`n_cols` (no `Layout`, no strides, no offset); any strided / broadcast / non-zero
  offset / reversed input is realized into a dense buffer by an upstream `Op::Contiguize` first
  (`requires_contiguous`, §4.3). `cumsum` walks per-input strides directly (`handles_strided`) and
  needs no Contiguize for a strided source.
- **dtype monomorphization.** `f32` and `f64` evaluate natively (`f64` transcendentals via
  GLSL.std.450 `Exp`/`Sqrt`); `f16` widens reductions to **f32** with native `float16_t` I/O;
  `bf16` is stored as **packed u16-in-u32 lane pairs**, math performed at **f32**, requiring
  `n_cols` (or `n`) **even**. The f32 accumulator for half I/O is the load-bearing precision
  invariant, not an accident.
- **Output: caller-pre-allocated, fully overwritten, contiguous row-major, same dtype & shape as
  the (primary) input; no aliasing.** Every kernel writes its output via the linear dispatch index
  (output contiguity is universal across the Vulkan stack). Backward outputs (`dx`, `grad_x`)
  follow the same rule. None of the forward/backward norm or softmax kernels accumulate atomically
  — they fully overwrite — so there is no zero-init requirement (contrast the scatter/index-add
  family). The one `InterlockedOr` half-word write idiom seen elsewhere in the reduce family
  (`reduce_last_dim` bf16) does **not** appear in softmax/rms/layer norm, which write whole rows.
- **No affine (gamma/beta) parameters** on the norm kernels — they are the bare normalization; an
  affine scale/shift is a separate downstream op. `eps` is an `f32` op-param packed into the Params
  block (`rms_norm` / `layer_norm` carry `n_rows, n_cols, eps, pad`).
- **Determinism: bit-stable on the same hardware, NOT cross-hardware.** Subgroup tree reductions
  over a fixed `n_cols` have a fixed reduction order on a given device (same subgroup width, same
  schedule), so a re-run on the same GPU is bit-identical. Across hardware the subgroup width and
  the GPU's `exp`/`sqrt` approximations differ, so these are not bit-stable cross-hardware — hence
  `bit_stable_on_same_hardware: true` with `determinism: same_hardware_bitwise`, and no
  cross-hardware ULP bound is claimed.
- **Cost provenance is `judge_measured`** for every kernel here — the Judge bootstraps it from
  measurement. Only genuinely derivable bandwidth/FLOP *shape* hints are recorded in the
  cost-expression strings as priors (these are streaming row reductions: **bandwidth-bound** at
  `≈ 2 · n_rows · n_cols · dtype_bytes` for the single-input forwards, `≈ 3 · …` for the two-input
  backwards; arithmetic is `O(n_rows · n_cols)`). The coefficients are owned by measurement, not
  authored.

---

## softmax  (fused last-dim softmax — f32)

Fused last-dim softmax `softmax(x)_i = exp(x_i - row_max) / Σ_j exp(x_j - row_max)`. One workgroup
per row over a contiguous `[n_rows, n_cols]` buffer; the row max and the exp-sum are computed with
**subgroup reductions**, then each element is scaled by `1/sum`
(`softmax.slang:32`; wrapper `softmax_last_dim_f32_bytes`, `lib.rs:1845`). Native f32 arithmetic.
Standard max-subtract stabilization. Output is the same dtype/shape, contiguous, fully overwritten.
Limitation: contiguous-only — any strided/broadcast/offset input is contiguized by the planner
first.

```fkc
kernel: softmax
op_kind: SoftmaxLastDim
blurb: "Fused numerically-stable softmax along the last dim (f32); subgroup row max-subtract, exp, normalize."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      n_rows: { kind: usize, note: "product of all dims before the last = number of rows" }
      n_cols: { kind: usize, note: "reduced last-dim length = row width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # Judge bootstraps; hints below are derivable priors only
  class: normalization
  flops: "n_rows * n_cols * 4"      # HINT: ~max + exp + sum + scale per element (exp dominates)
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"   # HINT: read input once, write out once (bandwidth-bound)
  overhead_ns: ~                    # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed subgroup reduction order per shape on a given GPU
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # audited, no static cross-hardware bound (none(reason)); §4.8
  notes: "native f32; stable softmax (subgroup row max-subtract). Bit-stable same hardware; NOT cross-hardware (subgroup width + GPU exp approx differ)."

determinism: same_hardware_bitwise
```

---

## softmax_f16  (fused last-dim softmax — f16, f32 accumulator)

f16 I/O softmax; reductions and `exp` evaluated in **f32**, native `float16_t` load/store, narrow
to f16 only on the final scale (`softmax.slang:32`; f16 wrapper between `lib.rs:1845`..`:2006`).
Same one-workgroup-per-row subgroup algorithm as the f32 variant. Output f16, same shape,
contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: softmax_f16
op_kind: SoftmaxLastDim
blurb: "Fused numerically-stable softmax along the last dim (f16; f32 accumulator)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 4"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated reductions; fixed subgroup order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, reductions + exp in f32 (narrow on store). Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_bf16  (fused last-dim softmax — bf16, f32 accumulator, packed-u32)

bf16 I/O softmax; bf16 stored as packed u16-in-u32 lane pairs, all arithmetic (row max, exp, sum,
scale) in **f32**, narrow to bf16 only on store (`softmax.slang:32`; bf16 wrapper in the
`lib.rs:1845`..`:2006` range). **Requires `n_cols` even** (lane-pair processing). One workgroup per
row, subgroup reductions. Output bf16, same shape, contiguous, overwritten. Bit-stable on the same
hardware. Contiguous-only.

```fkc
kernel: softmax_bf16
op_kind: SoftmaxLastDim
blurb: "Fused numerically-stable softmax along the last dim (bf16; f32 accumulator; n_cols even)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize, constraint: "n_cols % 2 == 0", note: "packed-u32 bf16 lane pairs" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "n_cols % 2 == 0", note: "bf16 packed-u32 lane-pair fast path (required)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 4"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 math, fixed subgroup order; narrow bf16 on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, all math f32 (narrow RNE upper-16 on store); n_cols even. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_f64  (fused last-dim softmax — f64)

Native f64 softmax; same one-workgroup-per-row subgroup algorithm, all arithmetic in f64, `exp`
via GLSL.std.450 (`softmax.slang:32`; wrapper `lib.rs:2006`). Widest range/precision of the family.
Output f64, same shape, contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: softmax_f64
op_kind: SoftmaxLastDim
blurb: "Fused numerically-stable softmax along the last dim (f64); native f64, GLSL.std.450 Exp."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 4"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64, fixed subgroup order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; stable softmax (subgroup row max-subtract), GLSL.std.450 Exp. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward  (fused softmax last-dim backward — f32)

Fused softmax backward `dx_i = y_i · (g_i - Σ_j y_j·g_j)` per row. Two **contiguous**
`[n_rows, n_cols]` inputs `(y, g)` — `y` the forward softmax output, `g` the upstream gradient; the
per-row dot product `Σ_j y_j·g_j` is computed once via subgroup reduction and reused for every
element (`softmax_last_dim_backward.slang:35`; wrapper `lib.rs:7265`, typed `:7347`). Native f32.
Output `dx`, same dtype/shape, contiguous, overwritten. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward
op_kind: SoftmaxLastDimBackward
blurb: "Fused softmax last-dim backward (f32): dx = y*(g - sum(y*g)) per row; subgroup dot."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_backward_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      n_rows: { kind: usize, note: "rows; n_rows*n_cols == y elem count" }
      n_cols: { kind: usize, note: "reduction width" }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
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
  flops: "3 * n_rows * n_cols"        # HINT: per element ~1 mul (dot) + 1 sub + 1 mul; two-pass
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"   # HINT: read y + g, write dx; bandwidth-bound
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed subgroup dot order per shape; native f32
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 dot + write; fixed subgroup reduction order. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_f16  (fused softmax last-dim backward — f16, f32 dot)

f16 I/O softmax backward; the per-row dot `Σ y·g` accumulated in **f32**, native `float16_t` I/O,
narrow to f16 on store (`softmax_last_dim_backward.slang:35`; typed wrapper `lib.rs:7347`). Two
contiguous `[n_rows, n_cols]` inputs `(y, g)`, one contiguous f16 output `dx`, overwritten.
Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_f16
op_kind: SoftmaxLastDimBackward
blurb: "Fused softmax last-dim backward (f16; f32 dot): dx = y*(g - sum(y*g)) per row."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_backward_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 dot accumulator; narrow on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, dot in f32 (narrow on store). Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_bf16  (fused softmax last-dim backward — bf16, f32 dot, packed-u32)

bf16 I/O softmax backward; packed-u32 pair-thread, dot accumulated in **f32**, narrow to bf16 on
store. **Requires `n_cols` even**; the inventory notes the pair-thread write is race-free (no
atomic accumulation) (`softmax_last_dim_backward.slang:35`; typed wrapper `lib.rs:7347`). Two
contiguous `[n_rows, n_cols]` inputs `(y, g)`, one contiguous bf16 output `dx`, overwritten.
Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_bf16
op_kind: SoftmaxLastDimBackward
blurb: "Fused softmax last-dim backward (bf16; f32 dot; n_cols even): dx = y*(g - sum(y*g)) per row."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_backward_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize, constraint: "n_cols % 2 == 0", note: "packed-u32 bf16 lane pairs" }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "n_cols % 2 == 0", note: "bf16 packed-u32 lane-pair path (required)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 dot; race-free pair-thread write; narrow bf16 on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, dot in f32 (narrow on store); n_cols even; no race. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## softmax_last_dim_backward_f64  (fused softmax last-dim backward — f64)

Native f64 softmax backward; same `dx_i = y_i · (g_i - Σ_j y_j·g_j)` algorithm computed in f64, the
per-row dot accumulated in f64 (`softmax_last_dim_backward.slang:35`; typed wrapper `lib.rs:7347`).
Two contiguous `[n_rows, n_cols]` inputs `(y, g)`, one contiguous f64 output `dx`, overwritten.
Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: softmax_last_dim_backward_f64
op_kind: SoftmaxLastDimBackward
blurb: "Fused softmax last-dim backward (f64): dx = y*(g - sum(y*g)) per row; native f64."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::softmax_last_dim_backward_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(y)
      shape_rule: same_as(y)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: normalization
  flops: "3 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64; fixed subgroup dot order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 dot + write; fixed subgroup reduction order. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim  (fused RMS normalization along the last dim, no affine — f32)

Fused RMS normalization `out_i = x_i / sqrt(mean(x²) + eps)` per row, with
`mean(x²) = (Σ x²) / n_cols`. One workgroup per row over a contiguous `[n_rows, n_cols]` buffer;
the sum-of-squares is a **subgroup reduction**, then the row is scaled (`rms_norm_last_dim.slang:41`;
wrapper `lib.rs:2058`). Native f32. **No affine (gamma) parameter** — bare RMS norm. `eps` is an
f32 op-param. Output f32, same shape, contiguous, overwritten. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: rms_norm_last_dim
op_kind: RmsNormLastDim
blurb: "Fused RMS norm along the last dim, no affine (f32): x / sqrt(mean(x^2) + eps); subgroup reduction."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      n_rows: { kind: usize, note: "number of rows" }
      n_cols: { kind: usize, note: "row width; divisor of the mean" }
      eps:    { kind: f32, note: "packed into Params block" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
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
  flops: "n_rows * n_cols * 3"        # HINT: x^2 + accumulate (reduce pass) + scale (write pass)
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"   # HINT: read input once, write out once
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32; fixed subgroup sum-of-squares order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; no affine; subgroup sum-of-squares + rsqrt. Bit-stable same hardware; NOT cross-hardware (sqrt approx + subgroup width)."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_f16  (fused RMS normalization along the last dim, no affine — f16, f32 accumulator)

f16 I/O RMS norm; sum-of-squares and reciprocal-sqrt computed in **f32**, native `float16_t` I/O,
narrow to f16 on store (`rms_norm_last_dim.slang:41`; f16 wrapper in `lib.rs:2058`..`:2239`). `eps`
f32. No affine. Output f16, same shape, contiguous, overwritten. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: rms_norm_last_dim_f16
op_kind: RmsNormLastDim
blurb: "Fused RMS norm along the last dim, no affine (f16; f32 accumulator)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 3"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated; narrow on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, sum-of-squares + rsqrt in f32 (narrow on store); no affine; eps f32. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_bf16  (fused RMS normalization along the last dim, no affine — bf16, f32 accumulator, packed-u32)

bf16 I/O RMS norm; packed u16-in-u32 lane pairs, sum-of-squares and rsqrt in **f32**, narrow to
bf16 on store (`rms_norm_last_dim.slang:41`; bf16 wrapper in `lib.rs:2058`..`:2239`). **Requires
`n_cols` even.** `eps` f32. No affine. Output bf16, same shape, contiguous, overwritten. Bit-stable
on the same hardware. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_bf16
op_kind: RmsNormLastDim
blurb: "Fused RMS norm along the last dim, no affine (bf16; f32 accumulator; n_cols even)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize, constraint: "n_cols % 2 == 0", note: "packed-u32 bf16 lane pairs" }
      eps:    { kind: f32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "n_cols % 2 == 0", note: "bf16 packed-u32 lane-pair path (required)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 3"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated; narrow bf16 on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, sum-of-squares + rsqrt in f32 (narrow on store); n_cols even; no affine; eps f32. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_f64  (fused RMS normalization along the last dim, no affine — f64)

Native f64 RMS norm; same algorithm, all arithmetic in f64, `sqrt` via GLSL.std.450
(`rms_norm_last_dim.slang:41`; wrapper `lib.rs:2239`). `eps` used natively. No affine. Output f64,
same shape, contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: rms_norm_last_dim_f64
op_kind: RmsNormLastDim
blurb: "Fused RMS norm along the last dim, no affine (f64): x / sqrt(mean(x^2) + eps); native f64."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32, note: "packed into Params block; used natively in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 3"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64; fixed subgroup order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; no affine; subgroup sum-of-squares + GLSL.std.450 sqrt. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## rms_norm_last_dim_backward  (fused RMS normalization backward — f32 only)

Fused RMS-norm backward in closed form: two per-row reductions (`Σ x²` and `Σ g·x`) feed the
analytic `grad_x` (`rms_norm_last_dim_backward.slang:68`). **f32 only** — there is no f16/bf16/f64
backward variant in the inventory. Two **contiguous** `[n_rows, n_cols]` inputs `(x, g_y)` — `x`
the forward *input* (stats recomputed) and `g_y` the upstream gradient; plus an `eps` op-param.
Output `grad_x`, same dtype/shape, contiguous, overwritten. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: rms_norm_last_dim_backward
op_kind: RmsNormLastDimBackward
blurb: "Fused RMS-norm last-dim backward (f32 only): closed-form grad_x from sum(x^2), sum(g*x)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rms_norm_last_dim_backward_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g_y
    - name: g_y
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (RmsNorm/LayerNorm share; OpKind selects); §3.7
    fields:
      n_rows: { kind: usize, note: "number of rows" }
      n_cols: { kind: usize, note: "reduction width" }
      eps:    { kind: f32 }

return:
  outputs:
    - name: grad_x
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
  flops: "5 * n_rows * n_cols"        # HINT: two reductions (sum x^2, sum g*x) + closed-form write
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"   # HINT: read x + g_y, write grad_x; bandwidth-bound
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32; fixed subgroup reduction order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 only; closed-form backward, two subgroup reductions. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim  (fused layer normalization along the last dim, no affine — f32)

Fused LayerNorm `out_i = (x_i - mean(x)) / sqrt(var(x) + eps)` per row, with `mean = (Σ x)/n_cols`
and `var = (Σ (x-mean)²)/n_cols`. One workgroup per row; **two subgroup reductions** (mean, then
variance) feed the per-element normalize (`layer_norm_last_dim.slang:27`; wrapper `lib.rs:5408`,
typed `:5480`). Native f32. **No affine (gamma/beta) parameters** — bare LayerNorm. `eps` f32.
Output f32, same shape, contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim
op_kind: LayerNormLastDim
blurb: "Fused LayerNorm along the last dim, no affine (f32): (x - mean) / sqrt(var + eps); subgroup reductions."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      n_rows: { kind: usize, note: "number of rows" }
      n_cols: { kind: usize, note: "row width; divisor of mean and variance" }
      eps:    { kind: f32, note: "packed into Params block" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
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
  flops: "n_rows * n_cols * 5"        # HINT: mean pass + variance pass + normalize pass
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"   # HINT: read input once, write out once
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32; fixed subgroup mean/variance order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; no affine; two subgroup reductions (mean, variance). Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_f16  (fused layer normalization along the last dim, no affine — f16, f32 accumulator)

f16 I/O LayerNorm; mean, variance, and rsqrt computed in **f32**, native `float16_t` I/O, narrow to
f16 on store (`layer_norm_last_dim.slang:27`; f16 wrapper in `lib.rs:5408`..`:5480`). `eps` f32.
No affine. Output f16, same shape, contiguous, overwritten. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: layer_norm_last_dim_f16
op_kind: LayerNormLastDim
blurb: "Fused LayerNorm along the last dim, no affine (f16; f32 accumulator)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 5"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated; narrow on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, mean/variance/rsqrt in f32 (narrow on store); no affine; eps f32. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_bf16  (fused layer normalization along the last dim, no affine — bf16, f32 accumulator, packed-u32)

bf16 I/O LayerNorm; packed u16-in-u32 lane pairs, mean/variance/rsqrt in **f32**, narrow to bf16 on
store (`layer_norm_last_dim.slang:27`; bf16 wrapper in `lib.rs:5408`..`:5480`). The bf16 path is
packed-u32; per the family rule it requires `n_cols` even. `eps` f32. No affine. Output bf16, same
shape, contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_bf16
op_kind: LayerNormLastDim
blurb: "Fused LayerNorm along the last dim, no affine (bf16; f32 accumulator; n_cols even)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize, constraint: "n_cols % 2 == 0", note: "packed-u32 bf16 lane pairs" }
      eps:    { kind: f32 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "n_cols % 2 == 0", note: "bf16 packed-u32 lane-pair path (required)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 5"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32-accumulated; narrow bf16 on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, mean/variance/rsqrt in f32 (narrow on store); n_cols even; no affine; eps f32. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_f64  (fused layer normalization along the last dim, no affine — f64)

Native f64 LayerNorm; same two-pass mean/variance algorithm in f64, `sqrt` via GLSL.std.450
(`layer_norm_last_dim.slang:27`; f64 wrapper in `lib.rs:5408`..`:5480`). `eps` used natively.
No affine. Output f64, same shape, contiguous, overwritten. Bit-stable on the same hardware.
Contiguous-only.

```fkc
kernel: layer_norm_last_dim_f64
op_kind: LayerNormLastDim
blurb: "Fused LayerNorm along the last dim, no affine (f64): (x - mean) / sqrt(var + eps); native f64."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32, note: "packed into Params block; used natively in f64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: normalization
  flops: "n_rows * n_cols * 5"
  bytes_moved: "2 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64; fixed subgroup mean/variance order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; no affine; two subgroup reductions, GLSL.std.450 sqrt. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward  (fused layer normalization backward — f32)

Fused LayerNorm backward via **four per-row reductions** (`Σ x`, `Σ x²`, `Σ g`, `Σ g·x`) feeding the
analytic `dx` (`layer_norm_last_dim_backward.slang:48`; wrapper `lib.rs:5266`, typed `:5339`).
Native f32. Two **contiguous** `[n_rows, n_cols]` inputs `(x, g)` — `x` the forward *input* (stats
recomputed) and `g` the upstream gradient; plus an `eps` op-param. Output `dx`, same dtype/shape,
contiguous, overwritten. Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward
op_kind: LayerNormLastDimBackward
blurb: "Fused LayerNorm last-dim backward (f32): analytic dx from sum_x, sum_x^2, sum_g, sum_gx."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_backward_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim                 # OpParams::NormLastDim (RmsNorm/LayerNorm share; OpKind selects); §3.7
    fields:
      n_rows: { kind: usize, note: "number of rows" }
      n_cols: { kind: usize, note: "reduction width" }
      eps:    { kind: f32 }

return:
  outputs:
    - name: dx
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
  flops: "7 * n_rows * n_cols"        # HINT: four reductions (sum_x, sum_x^2, sum_g, sum_gx) + analytic write
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"   # HINT: read x + g, write dx; bandwidth-bound
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f32; fixed subgroup reduction order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32; four subgroup reductions, analytic dx. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_f16  (fused layer normalization backward — f16, f32 reductions)

f16 I/O LayerNorm backward; the four per-row reductions accumulated in **f32**, native `float16_t`
I/O, narrow to f16 on store (`layer_norm_last_dim_backward.slang:48`; typed wrapper `lib.rs:5339`).
Two contiguous `[n_rows, n_cols]` inputs `(x, g)`, one contiguous f16 output `dx`, overwritten.
Bit-stable on the same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_f16
op_kind: LayerNormLastDimBackward
blurb: "Fused LayerNorm last-dim backward (f16; f32 reductions): analytic dx."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_backward_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32 }

return:
  outputs:
    - name: dx
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
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "7 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 reductions; narrow on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, four reductions in f32 (narrow on store); analytic dx. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_bf16  (fused layer normalization backward — bf16, f32 reductions, packed-u32)

bf16 I/O LayerNorm backward; packed u16-in-u32 lane pairs, the four per-row reductions accumulated
in **f32**, narrow to bf16 on store (`layer_norm_last_dim_backward.slang:48`; typed wrapper
`lib.rs:5339`). Per the family rule the bf16 path requires `n_cols` even. Two contiguous
`[n_rows, n_cols]` inputs `(x, g)`, one contiguous bf16 output `dx`, overwritten. Bit-stable on the
same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_bf16
op_kind: LayerNormLastDimBackward
blurb: "Fused LayerNorm last-dim backward (bf16; f32 reductions; n_cols even): analytic dx."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_backward_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize, constraint: "n_cols % 2 == 0", note: "packed-u32 bf16 lane pairs" }
      eps:    { kind: f32 }

return:
  outputs:
    - name: dx
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
    - { when: "n_cols % 2 == 0", note: "bf16 packed-u32 lane-pair path (required)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: normalization
  flops: "7 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 reductions; narrow bf16 on store; fixed subgroup order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, four reductions in f32 (narrow on store); n_cols even; analytic dx. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## layer_norm_last_dim_backward_f64  (fused layer normalization backward — f64)

Native f64 LayerNorm backward; same four-reduction analytic algorithm in f64
(`layer_norm_last_dim_backward.slang:48`; typed wrapper `lib.rs:5339`). Two contiguous
`[n_rows, n_cols]` inputs `(x, g)`, one contiguous f64 output `dx`, overwritten. Bit-stable on the
same hardware. Contiguous-only.

```fkc
kernel: layer_norm_last_dim_backward_f64
op_kind: LayerNormLastDimBackward
blurb: "Fused LayerNorm last-dim backward (f64): analytic dx; native f64."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::layer_norm_last_dim_backward_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim
    fields:
      n_rows: { kind: usize }
      n_cols: { kind: usize }
      eps:    { kind: f32, note: "used natively in f64" }

return:
  outputs:
    - name: dx
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
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: normalization
  flops: "7 * n_rows * n_cols"
  bytes_moved: "3 * n_rows * n_cols * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n_rows * n_cols * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # native f64; fixed subgroup reduction order per shape
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64; four subgroup reductions, analytic dx. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## cumsum_f32  (inclusive prefix sum along one axis — f32, strided-input capable)

Inclusive prefix (cumulative) sum along one axis: one thread per slice does a **serial walk** along
the `axis`, accumulating in an **f32** accumulator (`cumsum_f32.slang:27`; wrapper `lib.rs:9213`,
typed `cumsum_typed_bytes` `:9277`). Unlike the norm family, the input is **strided-capable**: the
kernel carries a rank-4 shape + per-input strides (`shape0..3`, `in_s0..3`), so it walks an
arbitrary (non-negative) strided / broadcast view directly with no upstream Contiguize. Output is
contiguous over the input shape, same dtype, overwritten. Bit-stable on the same hardware (the
serial per-slice accumulation has a fixed order).

```fkc
kernel: cumsum_f32
op_kind: Cumsum
blurb: "Inclusive prefix sum along one axis (f32); strided-input capable; serial per-slice walk."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cumsum_f32_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: same_as=out
  op_params:
    variant: Cumsum                  # OpParams::Cumsum (primitive namespace; §3.7)
    fields:
      slice_count: { kind: usize, note: "number of independent slices = product of non-axis dims" }
      axis:        { kind: usize, note: "axis (0..3) along which to scan" }
      dim_size:    { kind: usize, note: "length of the scanned axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks per-input strides directly; no Contiguize for strided input
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction, note: "linear contiguous walk" }
    - { when: "any_input_strided", class: reduction, note: "per-dim stride decode" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"                         # HINT: one add per element along the scanned axis (O(n))
  bytes_moved: "2 * n * dtype_bytes" # HINT: read input once, write out once; bandwidth-bound (n = product of output elems)
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # serial per-slice accumulation, fixed order; native f32
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 serial prefix sum; deterministic fixed-order per-slice. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## cumsum_f64  (inclusive prefix sum along one axis — f64, strided-input capable)

Native f64 inclusive prefix sum; same serial per-slice walk along `axis`, accumulator in f64
(`cumsum_f32.slang:27` family; typed wrapper `cumsum_typed_bytes`, `lib.rs:9277`). Strided-input
capable (rank-4 shape + per-input strides). Output contiguous over the input shape, same dtype,
overwritten. Bit-stable on the same hardware.

```fkc
kernel: cumsum_f64
op_kind: Cumsum
blurb: "Inclusive prefix sum along one axis (f64); strided-input capable; serial per-slice walk."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cumsum_f64_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: same_as=out
  op_params:
    variant: Cumsum
    fields:
      slice_count: { kind: usize }
      axis:        { kind: usize }
      dim_size:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # serial per-slice accumulation; native f64
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f64 serial prefix sum; deterministic fixed-order per-slice. Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## cumsum_f16  (inclusive prefix sum along one axis — f16, f32 accumulator, strided-input capable)

f16 I/O inclusive prefix sum; the serial per-slice scan accumulates in **f32**, native `float16_t`
I/O, narrow to f16 on store (`cumsum_f32.slang:27` family; typed wrapper `cumsum_typed_bytes`,
`lib.rs:9277`). The f32 accumulator is the precision invariant (the inventory notes cumsum is
per-dtype precisely because the accumulator needs a typed add). Strided-input capable. Output
contiguous over the input shape, same dtype, overwritten. Bit-stable on the same hardware.

```fkc
kernel: cumsum_f16
op_kind: Cumsum
blurb: "Inclusive prefix sum along one axis (f16; f32 accumulator); strided-input capable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cumsum_f16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: same_as=out
  op_params:
    variant: Cumsum
    fields:
      slice_count: { kind: usize }
      axis:        { kind: usize }
      dim_size:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 accumulator; narrow on store; serial fixed-order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f16 I/O, serial prefix sum accumulated in f32 (narrow on store). Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

---

## cumsum_bf16  (inclusive prefix sum along one axis — bf16, f32 accumulator, strided-input capable)

bf16 I/O inclusive prefix sum; the serial per-slice scan accumulates in **f32**, narrow to bf16 on
store (`cumsum_f32.slang:27` family; typed wrapper `cumsum_typed_bytes`, `lib.rs:9277`). bf16 is
packed-u16-in-u32; the f32 accumulator is the precision invariant. Strided-input capable. Output
contiguous over the input shape, same dtype, overwritten. Bit-stable on the same hardware.

```fkc
kernel: cumsum_bf16
op_kind: Cumsum
blurb: "Inclusive prefix sum along one axis (bf16; f32 accumulator); strided-input capable."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::cumsum_bf16_bytes"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: same_as=out
  op_params:
    variant: Cumsum
    fields:
      slice_count: { kind: usize }
      axis:        { kind: usize }
      dim_size:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: reduction }
    - { when: "any_input_strided", class: reduction }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: reduction
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  overhead_ns: ~                # judge_measured (non-derivable Vulkan command-buffer submit latency)
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 accumulator; narrow bf16 on store; serial fixed-order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "bf16 packed-u32 I/O, serial prefix sum accumulated in f32 (narrow RNE upper-16 on store). Bit-stable same hardware; NOT cross-hardware."

determinism: same_hardware_bitwise
```

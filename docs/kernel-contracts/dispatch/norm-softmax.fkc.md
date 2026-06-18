---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                  # default backend for the canonical block; CU/VK siblings noted in prose
  kernel_source: "portable-cpu" # the BindingEntry.kernel_source tag for the CPU binding
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4" # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — softmax / norm / rope kernel contracts (family: norm)

The `fuel-dispatch` crate is the registration home for the last-dim **softmax / log-softmax / RMS-norm
/ LayerNorm** family (and their backward passes) plus **Rope** (rotary position embedding). It does
not author kernel math — it ships the dispatch **wrappers** that the `KernelBindingTable` registers as
`KernelRef`s and that forward to the per-backend kernel bodies: `fuel-cpu-backend` byte kernels (CPU),
baracuda CUDA kernels (CU), and `fuel-vulkan-kernels` Slang (VK). Every kernel in this family is a
**row-wise reduction along the last dimension**: the flat buffer is viewed as `outer_count` rows of
`last_dim` contiguous elements, each row reduced (max / sum / mean / variance / dot) and rewritten
independently. Rope is the exception — a per-position rotary rewrite of `x` against broadcast
`cos`/`sin` tables.

**Multi-backend reality (read this before the per-kernel blocks).** A single `(OpKind, [DType…])`
key in this crate is registered for up to three backends, with *materially different* layout
capabilities and precision per backend. The canonical ` ```fkc ` block in each section below describes
the **CPU binding** — the always-built universal fallback (CLAUDE.md "always-built coverage
commitment", §4.8), which has the widest dtype coverage and is `bit_stable_on_same_hardware: true`. The
CU and VK sibling bindings are real, separate `BindingEntry`s at the same key (distinct `KernelRef`,
distinct `backend`, distinct `kernel_source`); they are documented in each section's prose with their
own layout caps, dtypes, precision, and `source` (`file:line`) and would each carry their own ` ```fkc `
block when this crate's contract set is split per-backend. The load-bearing per-backend facts (from
the inventory `docs/kernel-contracts/_inventory/dispatch.md`):

- **CPU** (`register_cpu_kernels`, `dispatch.rs:3880`): `kernel_source: "portable-cpu"`,
  `backend: Cpu`. **Contiguous-only** (`C`): all CPU wrappers take `_layouts` UNUSED and operate on
  raw `CpuStorageBytes`; they rely entirely on the executor's auto-Contiguize pass
  (`awkward_layout_strategy: requires_contiguous`). f32/f64 native; bf16/f16 widen to **f32** for all
  reduction arithmetic and narrow on store. **Bit-stable on the same hardware** (precision
  bulk-upgraded to `PRIMITIVE_DETERMINISTIC_CPU`). Widest dtype coverage of the three.
- **CU / baracuda** (`register_baracuda_cuda_kernels`, `baracuda_dispatch.rs:2353`):
  `kernel_source: "baracuda"`, `backend: Cuda`. The softmax/log-softmax/rms/layer **forwards** and
  **Rope** register `KernelCaps::strided_input` (`S`) — baracuda FFI is stride-driven (passes
  rank-N shape + strides). **NOT offset-capable** (non-zero `start_offset` inputs still
  auto-Contiguize, `compiled.rs:58`). The norm/softmax *backwards* have **no baracuda binding** in
  this crate (CPU/VK only). Half accumulates in f32 in the kernel.
- **VK / Vulkan** (`register_vulkan_kernels`, `vulkan_dispatch.rs`): `kernel_source: "vulkan-slang"`,
  `backend: Vulkan`. Softmax/log-softmax/rms/layer norm + backwards are **contiguous-only** (`C`);
  Rope is **strided** (`S`) on `x` (cos/sin forced contiguous by the wrapper). Vulkan
  reductions/softmax/norm carry `PrecisionGuarantee::none` — they are **not** bit-stable
  (subgroup/atomic accumulation order is scheduler-dependent across hardware, though bit-stable on the
  *same* hardware per the vulkan inventory). LogSoftmax has **no VK binding**.

**Op-param sharing.** SoftmaxLastDim / LogSoftmaxLastDim and their backwards carry
`OpParams::SoftmaxLastDim { outer_count, last_dim }` / `OpParams::LogSoftmaxLastDim { outer_count,
last_dim }`. RMS-norm and LayerNorm (forward + backward) share **one** variant
`OpParams::NormLastDim { outer_count, last_dim, eps }` — the `OpKind` (not the params) selects RMS vs
LayerNorm. Rope carries `OpParams::Rope { outer_count, seq, head_dim }`. The forward norms carry **no
affine (gamma/beta)** parameters — they are bare normalization; an affine scale/shift is a separate
downstream op. `eps` is an `f64` op-param on the CPU path (narrowed to `f32` inside the half/f32
kernels, used natively in f64) and an `f32` op-param on the VK path.

**Cost provenance is `declared`** for every kernel here: each block carries an authored absolute
launch prior (`overhead_ns: 40`), which is a legitimate author prior the Judge later refines (§4.4)
— so the block is `declared`, not `judge_measured` (no authored absolute constant may sit under
`judge_measured`). The `flops` / `bytes_moved` strings are genuinely derivable bandwidth/FLOP *shape*
hints recorded as priors: these are streaming row reductions, **bandwidth-bound** at
`≈ 2 · outer_count · last_dim · dtype_bytes` for the single-input forwards and `≈ 3 · …` for the
two-input backwards; arithmetic is `O(outer_count · last_dim)`. The Judge refines the absolute
coefficients from measurement; the declared values seed it.

---

## SoftmaxLastDim  (numerically-stable softmax along the last dim)

Row-wise softmax `softmax(x)_i = exp(x_i - row_max) / Σ_j exp(x_j - row_max)` with the standard
max-subtract stabilization: per row find `row_max`, write `exp(x - row_max)`, accumulate the sum, then
scale the row by `1/sum`. No affine, no temperature; pure softmax. Output is the same dtype/shape,
contiguous, fully overwritten; no aliasing. half (bf16/f16) widens to f32 for the reduction and narrows
on store (the f32 accumulator is the precision invariant).

The canonical block below is the **CPU** binding (`SoftmaxLastDim`, dtypes f32/f64/bf16/f16,
contiguous-only, bit-stable; `cpu_softmax_last_dim_wrapper!` → `fuel_cpu_backend::byte_kernels::
softmax_last_dim_*`, registered `dispatch.rs:4532-4535`). Sibling bindings at the same key:

- **CU / baracuda** — dtypes f32/f64/bf16/f16, **strided** (`S`; wrapper requires `layouts[0]`),
  `kernel_source: "baracuda"`, `backend: Cuda`; precision bit-stable-same-hardware (baracuda
  deterministic). `awkward_layout_strategy: handles_strided`. Source `baracuda_dispatch.rs:2464`.
- **VK / Vulkan** — dtypes f32 (+ f16/bf16/f64 feature-gated), **contiguous-only** (`C`),
  `kernel_source: "vulkan-slang"`, `backend: Vulkan`; `PrecisionGuarantee::none` (subgroup reduction,
  not bit-stable cross-hardware). Source `vulkan_dispatch.rs:4332`. (Fully contracted in
  `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`.)

```fkc
kernel: softmax_last_dim
op_kind: SoftmaxLastDim
blurb: "Numerically-stable softmax along the last dim; row max-subtract, exp, normalize; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::softmax_last_dim_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "product of all dims before the last = number of rows" }
      last_dim:    { kind: usize, note: "reduced last-dim length = row width" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU contiguous-only; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4); formula hints below are derivable priors
  class: normalization
  flops: "outer_count * last_dim * 4"   # HINT: ~max + exp + sum + scale per element (exp dominates)
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # HINT: read input once, write out once (bandwidth-bound)
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic sequential reduction; native f32/f64, f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 widen to f32, narrow on store. Stable softmax (row max-subtract). Flat accumulator: deterministic, bit-stable same hardware. CU sibling bit-stable; VK sibling PrecisionGuarantee::none."

determinism: same_hardware_bitwise
```

---

## LogSoftmaxLastDim  (numerically-stable log-softmax along the last dim)

Row-wise log-softmax via row-max + log-sum-exp: per row find `row_max` (seeded `NEG_INFINITY`),
accumulate `sum += exp(x - row_max)`, take `log_sum = ln(sum)`, then write `x - row_max - log_sum`. No
affine. Output same dtype/shape, contiguous, overwritten; no aliasing. half widens to f32 for the
reduction and narrows on store.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_log_softmax_wrapper!` → `fuel_cpu_backend::byte_kernels::log_softmax_last_dim_*`, registered
`dispatch.rs:4419-4422`). Sibling binding:

- **CU / baracuda** — dtypes f32/f64/bf16/f16, **strided** (`S`), `kernel_source: "baracuda"`,
  `backend: Cuda`; `awkward_layout_strategy: handles_strided`. Source `baracuda_dispatch.rs:2469`.
- **VK** — **no Vulkan binding** for LogSoftmax in this crate.

```fkc
kernel: log_softmax_last_dim
op_kind: LogSoftmaxLastDim
blurb: "Numerically-stable log-softmax along the last dim; row max + log-sum-exp; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::log_softmax_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: LogSoftmaxLastDim       # OpParams::LogSoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width" }

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
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 4"   # HINT: max + exp/sum + ln + subtract per element
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # HINT: read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: native f32/f64, deterministic sequential reduction; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 widen to f32, narrow on store. Row max + log-sum-exp. Deterministic, bit-stable same hardware. CU sibling strided + bit-stable; no VK binding."

determinism: same_hardware_bitwise
```

---

## SoftmaxLastDimBackward  (fused softmax last-dim backward)

Fused softmax backward `dx_i = y_i · (g_i - Σ_j y_j·g_j)` per row, where `y` is the forward softmax
output and `g` the upstream gradient; the per-row dot `Σ y·g` is computed once and reused for every
element. Two inputs `(y, g)` of identical shape; key `[T, T, T]`. Output `dx`, same dtype/shape,
contiguous, overwritten; no aliasing.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_softmax_last_dim_backward_wrapper!` → `fuel_cpu_backend::byte_kernels::
softmax_last_dim_backward_*`, registered `dispatch.rs:4433-4436`). Sibling binding:

- **VK / Vulkan** — dtypes f32/f16/bf16/f64, **contiguous-only** (`C`), `kernel_source:
  "vulkan-slang"`, `backend: Vulkan`; `PrecisionGuarantee::none`. Source `vulkan_dispatch.rs:4344`.
  (Fully contracted in `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`.)
- **CU** — **no baracuda binding** for the softmax backward in this crate.

```fkc
kernel: softmax_last_dim_backward
op_kind: SoftmaxLastDimBackward
blurb: "Fused softmax last-dim backward: dx = y*(g - sum(y*g)) per row; half via f32 dot."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::softmax_last_dim_backward_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: SoftmaxLastDim          # OpParams::SoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows; outer_count*last_dim == y elem count" }
      last_dim:    { kind: usize, note: "reduction width" }

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
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "3 * outer_count * last_dim"   # HINT: per element ~1 mul (dot) + 1 sub + 1 mul; two-pass
  bytes_moved: "3 * outer_count * last_dim * dtype_bytes"   # HINT: read y + g, write dx; bandwidth-bound
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic dot + write; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64 dot + write; bf16/f16 accumulate dot in f32, narrow on store. Deterministic, bit-stable same hardware. VK sibling PrecisionGuarantee::none; no CU binding."

determinism: same_hardware_bitwise
```

---

## LogSoftmaxLastDimBackward  (fused log-softmax last-dim backward — CPU only)

Fused log-softmax backward `dx_i = g_i - exp(y_i) · Σ_j g_j` per row, where `y` is the forward
log-softmax output and `g` the upstream gradient. Two inputs `(y, g)` of identical shape; key
`[T, T, T]`. Output `dx`, same dtype/shape, contiguous, overwritten; no aliasing.

**CPU-only in this crate** — no baracuda and no Vulkan binding for the log-softmax backward. The
canonical block is the CPU binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_log_softmax_backward_wrapper!` → `fuel_cpu_backend::byte_kernels::log_softmax_last_dim_backward_*`,
registered `dispatch.rs:4425-4428`). half widens to f32 for the row reduction and narrows on store.

```fkc
kernel: log_softmax_last_dim_backward
op_kind: LogSoftmaxLastDimBackward
blurb: "Fused log-softmax last-dim backward (CPU only): dx = g - exp(y)*sum(g) per row; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::log_softmax_backward_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: y
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=y
  op_params:
    variant: LogSoftmaxLastDim       # OpParams::LogSoftmaxLastDim (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "rows; outer_count*last_dim == y elem count" }
      last_dim:    { kind: usize, note: "reduction width" }

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
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "3 * outer_count * last_dim"   # HINT: per element ~exp + mul + sub; one reduction (sum g)
  bytes_moved: "3 * outer_count * last_dim * dtype_bytes"   # HINT: read y + g, write dx; bandwidth-bound
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU only: native f32/f64; bf16/f16 widen to f32, narrow on store. Sum-of-g reduction + exp(y) scale. Deterministic, bit-stable same hardware. No CU/VK binding."

determinism: same_hardware_bitwise
```

---

## RmsNormLastDim  (RMS normalization along the last dim, no affine)

Row-wise RMS normalization `out_i = x_i / sqrt(mean(x²) + eps)` per row, with `mean(x²) = (Σ x²) /
last_dim`. **No affine (gamma) parameter** — bare RMS norm. `eps` arrives as an `f64` op-param
(narrowed to `f32` in the half/f32 CPU kernels, used natively in f64). Output same dtype/shape,
contiguous, overwritten; no aliasing. One reduction pass (sum-of-squares) + one write pass.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_norm_last_dim_wrapper!` → `fuel_cpu_backend::byte_kernels::rms_norm_last_dim_*`, registered
`dispatch.rs:4536-4539`). Sibling bindings at the same key:

- **CU / baracuda** — dtypes f32/f64/f16/bf16, **strided** (`S`; passes rank-N shape + strides to
  FFI), `kernel_source: "baracuda"`, `backend: Cuda`; `awkward_layout_strategy: handles_strided`.
  Source `baracuda_dispatch.rs:2439`.
- **VK / Vulkan** — dtypes f32 (+ f16/bf16/f64 gated), **contiguous-only** (`C`), `eps` an **f32**
  op-param packed into the Params block, `kernel_source: "vulkan-slang"`, `backend: Vulkan`;
  `PrecisionGuarantee::none`. Source `vulkan_dispatch.rs:4333`. (Fully contracted in
  `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`.)

```fkc
kernel: rms_norm_last_dim
op_kind: RmsNormLastDim
blurb: "RMS norm along the last dim, no affine: x / sqrt(mean(x^2) + eps); half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rms_norm_last_dim_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width; divisor of the mean" }
      eps:         { kind: f64, note: "f64 op-param; narrowed to f32 in half/f32 kernels, native in f64 (VK sibling uses f32)" }

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
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 3"   # HINT: x^2 + accumulate (reduce pass) + scale (write pass)
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # HINT: read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: native f32/f64, deterministic; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 sum-of-squares + rsqrt in f32, narrow on store; no affine; eps f64 narrowed to f32. Deterministic, bit-stable same hardware. CU sibling strided + bit-stable; VK sibling PrecisionGuarantee::none, eps f32."

determinism: same_hardware_bitwise
```

---

## LayerNormLastDim  (layer normalization along the last dim, no affine)

Row-wise layer normalization `out_i = (x_i - mean(x)) / sqrt(var(x) + eps)` per row, with `mean = (Σ x)
/ last_dim` and `var = (Σ (x - mean)²) / last_dim`. **No affine (gamma/beta) parameters** — bare
LayerNorm. `eps` arrives as an `f64` op-param (narrowed to `f32` in half/f32 CPU kernels, native in
f64). Two reduction passes per row (mean, then variance) + one write pass. Output same dtype/shape,
contiguous, overwritten; no aliasing.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_norm_last_dim_wrapper!` → `fuel_cpu_backend::byte_kernels::layer_norm_last_dim_*`, registered
`dispatch.rs:4540-4543`). Sibling bindings at the same key:

- **CU / baracuda** — dtypes f32/f64/f16/bf16, **strided** (`S`), `kernel_source: "baracuda"`,
  `backend: Cuda`; `awkward_layout_strategy: handles_strided`. Source `baracuda_dispatch.rs:2444`.
- **VK / Vulkan** — dtypes f32 (+ f16/bf16/f64 gated), **contiguous-only** (`C`), `eps` an **f32**
  op-param, `kernel_source: "vulkan-slang"`, `backend: Vulkan`; `PrecisionGuarantee::none`. Source
  `vulkan_dispatch.rs:4385`. (Fully contracted in `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`.)

```fkc
kernel: layer_norm_last_dim
op_kind: LayerNormLastDim
blurb: "LayerNorm along the last dim, no affine: (x - mean) / sqrt(var + eps); half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::layer_norm_last_dim_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (shared by RmsNorm + LayerNorm; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "row width; divisor of mean and variance" }
      eps:         { kind: f64, note: "f64 op-param; narrowed to f32 in half/f32 kernels, native in f64 (VK sibling uses f32)" }

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
    - { when: "last_dim == 0", note: "early Ok no-op" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "outer_count * last_dim * 5"   # HINT: mean pass + variance pass + normalize pass
  bytes_moved: "2 * outer_count * last_dim * dtype_bytes"   # HINT: read input once, write out once
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: native f32/f64, deterministic two-pass; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 mean/variance/rsqrt in f32, narrow on store; no affine; eps f64 narrowed to f32. Two-pass mean/variance, deterministic, bit-stable same hardware. CU sibling strided + bit-stable; VK sibling PrecisionGuarantee::none, eps f32."

determinism: same_hardware_bitwise
```

---

## RmsNormLastDimBackward  (fused RMS-norm last-dim backward)

Fused RMS-norm backward in closed form: two per-row reductions (`Σ x²` and `Σ g·x`) feed the analytic
`grad_x`, where `x` is the forward *input* (stats recomputed) and `g` (a.k.a. `g_y`) the upstream
gradient; plus the shared `eps` op-param. Two inputs `(x, g)` of identical shape; key `[T, T, T]`.
Output `grad_x`, same dtype/shape, contiguous, overwritten; no aliasing.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_norm_backward_wrapper!` → `fuel_cpu_backend::byte_kernels::rms_norm_last_dim_backward_*`,
registered `dispatch.rs:4441-4444`). half widens to f32 for the two reductions and narrows on store.
Sibling binding:

- **VK / Vulkan** — **f32 only** (no f16/bf16/f64 backward variant), **contiguous-only** (`C`), `eps`
  an **f32** op-param, `kernel_source: "vulkan-slang"`, `backend: Vulkan`; `PrecisionGuarantee::none`.
  Source `vulkan_dispatch.rs` rms-norm backward (see `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`).
- **CU** — **no baracuda binding** for the RMS-norm backward in this crate.

```fkc
kernel: rms_norm_last_dim_backward
op_kind: RmsNormLastDimBackward
blurb: "Fused RMS-norm last-dim backward: closed-form grad_x from sum(x^2), sum(g*x); half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rms_norm_last_dim_backward_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (RmsNorm/LayerNorm share; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "reduction width" }
      eps:         { kind: f64, note: "f64 op-param; narrowed to f32 in half/f32 kernels, native in f64 (VK sibling uses f32)" }

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
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "5 * outer_count * last_dim"   # HINT: two reductions (sum x^2, sum g*x) + closed-form write
  bytes_moved: "3 * outer_count * last_dim * dtype_bytes"   # HINT: read x + g, write grad_x; bandwidth-bound
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: native f32/f64, deterministic; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 reductions in f32, narrow on store; closed-form backward (sum x^2, sum g*x); eps f64 narrowed to f32. Deterministic, bit-stable same hardware. VK sibling f32-only + PrecisionGuarantee::none; no CU binding."

determinism: same_hardware_bitwise
```

---

## LayerNormLastDimBackward  (fused LayerNorm last-dim backward)

Fused LayerNorm backward via four per-row reductions (`Σ x`, `Σ x²`, `Σ g`, `Σ g·x`) feeding the
analytic `dx`, where `x` is the forward *input* (stats recomputed) and `g` the upstream gradient; plus
the shared `eps` op-param. Two inputs `(x, g)` of identical shape; key `[T, T, T]`. Output `dx`, same
dtype/shape, contiguous, overwritten; no aliasing.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`cpu_norm_backward_wrapper!` → `fuel_cpu_backend::byte_kernels::layer_norm_last_dim_backward_*`,
registered `dispatch.rs:4437-4440`). half widens to f32 for the four reductions and narrows on store.
Sibling binding:

- **VK / Vulkan** — dtypes f32/f16/bf16/f64, **contiguous-only** (`C`), `eps` an **f32** op-param,
  `kernel_source: "vulkan-slang"`, `backend: Vulkan`; `PrecisionGuarantee::none`. Source
  `vulkan_dispatch.rs:4371`. (Fully contracted in `docs/kernel-contracts/vulkan/norm-softmax.fkc.md`.)
- **CU** — **no baracuda binding** for the LayerNorm backward in this crate.

```fkc
kernel: layer_norm_last_dim_backward
op_kind: LayerNormLastDimBackward
blurb: "Fused LayerNorm last-dim backward: analytic dx from sum_x, sum_x^2, sum_g, sum_gx; half via f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::layer_norm_last_dim_backward_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=g
    - name: g
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=x
  op_params:
    variant: NormLastDim             # OpParams::NormLastDim (RmsNorm/LayerNorm share; OpKind selects); §3.7
    fields:
      outer_count: { kind: usize, note: "number of rows" }
      last_dim:    { kind: usize, note: "reduction width" }
      eps:         { kind: f64, note: "f64 op-param; narrowed to f32 in half/f32 kernels, native in f64 (VK sibling uses f32)" }

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
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "7 * outer_count * last_dim"   # HINT: four reductions (sum_x, sum_x^2, sum_g, sum_gx) + analytic write
  bytes_moved: "3 * outer_count * last_dim * dtype_bytes"   # HINT: read x + g, write dx; bandwidth-bound
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * last_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: native f32/f64, deterministic four reductions; f32 accumulator for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 reductions in f32, narrow on store; analytic dx (sum_x, sum_x^2, sum_g, sum_gx); eps f64 narrowed to f32. Deterministic, bit-stable same hardware. VK sibling PrecisionGuarantee::none; no CU binding."

determinism: same_hardware_bitwise
```

---

## Rope  (rotary position embedding)

Rotary position embedding applied to `x` against per-position `cos`/`sin` tables: each head-dim pair is
rotated by the precomputed angle, `out = x*cos + rotate_half(x)*sin` over `outer_count` outer rows of
`seq × head_dim`. **Three inputs `(x, cos, sin)`** + one output (the CPU/VK key is `[T, T, T, T]`; the
CU key is the canonical short `[T, T]`). `cos`/`sin` are `[seq, head_dim]` and broadcast across the
outer dims. Output same dtype/shape as `x`, contiguous, overwritten; no aliasing.

The canonical block is the **CPU** binding (dtypes f32/f64/bf16/f16, contiguous-only, bit-stable;
`rope_*_cpu_wrapper` → `fuel_cpu_backend::byte_kernels::rope_*`, registered `dispatch.rs:4555-4557`;
the wrapper takes 3 inputs + 1 output, `dispatch.rs:1296-1332`). Sibling bindings at the same key:

- **CU / baracuda** — dtypes f32/f64/f16/bf16, **strided** (`S`), `kernel_source: "baracuda"`,
  `backend: Cuda`; `awkward_layout_strategy: handles_strided` (canonical short `[T, T]` key). Source
  `baracuda_dispatch.rs:2456`.
- **VK / Vulkan** — dtypes f32/f16/f64/bf16, **strided on `x`** (`S`; cos/sin forced contiguous by the
  wrapper, `rope.slang` carries x strides + a fast-path flag), `kernel_source: "vulkan-slang"`,
  `backend: Vulkan`; `PrecisionGuarantee::none`. Source `vulkan_dispatch.rs:4416`. (Contracted in
  `docs/kernel-contracts/vulkan/conv-attn-rope.fkc.md`.)

Because the CU and VK siblings are stride-capable on `x` while the CPU binding is contiguous-only, this
is exactly the planner's contiguize-vs-strided decision (§4.3/§4.4): a transposed-view `x` feeding the
CPU kernel pays an inserted `Op::Contiguize` (priced from the Contiguize contract), while the CU/VK
siblings consume the strided view directly.

```fkc
kernel: rope
op_kind: Rope
blurb: "Rotary position embedding: rotate x head-dim pairs by per-position cos/sin; out = x*cos + rotate_half(x)*sin."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::rope_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: cos
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                          # [seq, head_dim]; broadcast across outer dims
      shape_constraint: same_as=sin
    - name: sin
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                          # [seq, head_dim]
      shape_constraint: same_as=cos
  op_params:
    variant: Rope                    # OpParams::Rope (primitive namespace; §3.7)
    fields:
      outer_count: { kind: usize, note: "product of dims before (seq, head_dim) = rows of rotation" }
      seq:         { kind: usize, note: "sequence length = cos/sin dim[0]" }
      head_dim:    { kind: usize, note: "rotation width = cos/sin dim[1]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU contiguous-only; CU/VK siblings handle_strided on x
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared              # authored overhead_ns:40 launch prior (declared); Judge refines it later (§4.4)
  class: normalization
  flops: "outer_count * seq * head_dim * 6"   # HINT: per element ~2 mul + 1 add for x*cos + rotate_half(x)*sin
  bytes_moved: "(2 * outer_count * seq * head_dim + 2 * seq * head_dim) * dtype_bytes"   # HINT: read x + cos/sin tables, write out
  overhead_ns: 40
  memory: { device_bytes: 0, host_bytes: "outer_count * seq * head_dim * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # CPU: deterministic elementwise rotation; native f32/f64, f32 for half
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU: native f32/f64; bf16/f16 rotate in f32, narrow on store. Deterministic elementwise rotation, bit-stable same hardware. CU/VK siblings strided on x; VK PrecisionGuarantee::none."

determinism: same_hardware_bitwise
```

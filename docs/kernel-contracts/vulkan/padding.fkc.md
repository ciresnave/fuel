---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                                       # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"                         # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                         # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — padding kernel contracts

Vulkan/Slang padding kernels from `fuel-kernels-source/kernels/pad_*.slang` (AOT-compiled to SPIR-V
in `fuel-vulkan-kernels/spv/`, registered in the `EMBEDDED` table), with Rust dispatch wrappers in
`fuel-vulkan-backend/src/lib.rs`. The family covers the three forward pad modes (Constant / Reflect /
Replicate), the matching backwards (Constant, plus the atomic-accumulate Reflect / Replicate), and
`masked_fill`.

Two distinct kernel shapes live here, faithful to the as-built split:

- **Byte-width-keyed, dtype-agnostic data movers** — `pad_const_b{1,2,4,8}`, `pad_reflect_b{1,2,4,8}`,
  `pad_replicate_b{1,2,4,8}`, `pad_backward_const_b{1,2,4,8}`, `masked_fill_b{1,2,4,8}`. Each variant
  is keyed by element **byte width** (b1=1B u8/i8; b2=2B f16/bf16/i16/u16; b4=4B f32/i32/u32; b8=8B
  f64/i64), reads/writes raw words (u32-addressed), does **no arithmetic**, and is therefore
  **bit-exact for any dtype of that width**. One contract per variant; each lists the dtypes that
  share its element size.
- **Dtype-specific atomic-accumulate backwards** — `pad_backward_reflect_{f32,f64,f16,bf16}` and
  `pad_backward_replicate_{f32,f64,f16,bf16}`. Reflect/Replicate forward map **multiple output
  positions onto one input slot**, so the backward must *sum* gradients into each `grad_in` slot. The
  Vulkan kernels do this with a per-element atomic compare-and-swap (CAS) read-modify-write: f32 via
  uint CAS, f64 via u64 CAS (needs `shaderInt64` + 64-bit atomics + f64), f16/bf16 via sub-word CAS
  (math at f32). `grad_in` **must be zero-filled by the wrapper before dispatch** (the kernel only
  accumulates), and because the CAS accumulation order is scheduler-dependent, FP addition is **not
  associative**, so these are `determinism: nondeterministic` with an audited `none(reason)`
  precision (no silent unaudited nondeterminism).

Every input here is **contiguous, offset 0** — none of these kernels consults a `Layout` or walks a
signed stride. Shape/stride semantics arrive as an explicit `shape_buf` storage buffer
(`[in_shape, out_shape, left_pad]` for the pad kernels; `[src_shape, out_shape]` is not used here),
so `awkward_layout_strategy: requires_contiguous` throughout — the pipelined executor's
auto-Contiguize pass (itself an FKC kernel, §4.3) realizes any strided/broadcast/offset producer into
a dense buffer before these kernels run, and the planner sums that Contiguize's cost. Rank ≤ 8 for
all multi-dim pad/backward kernels (the `shape_buf` is sized for it). Output is **always contiguous**.

Cost on every kernel is `provenance: judge_measured` — the Judge bootstraps it (FKC stays agnostic to
how, §4.4). A `bytes_moved` bandwidth formula hint is given where genuinely derivable (these are pure
byte movers / accumulators, bandwidth-bound over element counts); `overhead_ns` (Vulkan command-buffer
submit/dispatch latency) is left `~` for the Judge to measure, and no FLOPs/latency numbers are
fabricated. Symbol convention: `n_out` = product of output elements; `n_in` = product of input
elements; `dtype_bytes` = the variant's element byte width (1/2/4/8).

---

## pad_const_b1  (constant-fill pad — 1-byte elements)

One-line: constant-fill pad for 1-byte (u8/i8) elements; writes one fill word into pad slots, copies the input region into the interior.

Constant-fill pad (`OpKind::Pad`, Constant mode) for 1-byte elements. One thread per output element:
the thread decomposes its linear output index into rank-N coordinates against `out_shape`
(from `shape_buf`), subtracts the per-axis `left_pad`, and if the resulting coordinate lies inside
`in_shape` for every axis it copies that input element's byte; otherwise it stores the `fill_value`
bit pattern (passed as a u32, low 8 bits used for the 1-byte element). Rank ≤ 8. Numerics: none — a
byte copy / constant store, so it is bit-exact for any 1-byte dtype (the fill value's bits are
supplied pre-encoded by the wrapper). Perf: bandwidth-bound, one linear pass over the output
(`n_out` writes, with the interior also reading `n_in` input bytes). Limitations: src/out
**contiguous, offset 0** only; `out_shape[i]` must equal `in_shape[i] + left_pad[i] + right_pad[i]`
per axis (the dispatch wrapper guarantees this); no symbolic-extent awareness (reads full concrete
shapes from `shape_buf`).

```fkc
kernel: pad_const_b1
op_kind: Pad                        # Constant mode (byte-width-keyed variant, 1B)
blurb: "Constant-fill pad for 1-byte (u8/i8) elements; writes one fill word into pad slots, copies the input region into the interior."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_const_b1"   # wrapper pad_const_bytes lib.rs:7116; kernel pad_const_b4.slang:25 (b1 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8]                       # 1-byte elements (byte-width-keyed, dtype-agnostic)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad                    # OpParams::Pad (primitive namespace)
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      fill_value: { kind: u32, note: "fill element bit pattern (low 8 bits for a 1-byte element)" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad] (rank entries each)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # byte width preserved; dtype = input dtype
      shape_rule: from_params(out_shape)      # per-axis in_shape[i] + left_pad[i] + right_pad[i]
      layout_guarantee: contiguous            # fresh dense row-major; executor pre-allocates
      aliasing: none                          # full overwrite (every output slot written)

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32                 # u32-word addressed (1-byte elements packed in words)

cost:
  provenance: judge_measured                  # Judge bootstraps; bandwidth-bound byte mover
  class: cheap_elementwise
  flops: "0"                                  # pure copy / constant store, no arithmetic
  bytes_moved: "(n_out + n_in) * dtype_bytes" # write every output; read the input interior
  overhead_ns: ~                              # judge_measured (Vulkan dispatch submit)
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true           # byte copy / constant store: bit-exact for any dtype, any hardware
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy (input region) + constant store (pad slots); no arithmetic, bit-exact for every 1-byte dtype."

determinism: bitwise                          # exact byte shuffle/store, hardware-independent
```

## pad_const_b2  (constant-fill pad — 2-byte elements)

One-line: constant-fill pad for 2-byte (f16/bf16/i16/u16) elements; copies the input region, fills pad slots with one constant word.

Constant-fill pad (`OpKind::Pad`, Constant mode) for 2-byte elements. Identical algorithm to
`pad_const_b1` keyed to a 2-byte element: one thread per output element, decompose-against-`out_shape`,
shift by `left_pad`, copy-from-input-if-interior else store the `fill_value` low 16 bits. Rank ≤ 8.
Numerics: none — byte copy / constant store, bit-exact for any 2-byte dtype. Perf: bandwidth-bound,
one linear pass over the output. Limitations: src/out contiguous, offset 0; per-axis
`out_shape[i] == in_shape[i] + left_pad[i] + right_pad[i]`.

```fkc
kernel: pad_const_b2
op_kind: Pad                        # Constant mode (2B)
blurb: "Constant-fill pad for 2-byte (f16/bf16/i16/u16) elements; copies the input region, fills pad slots with one constant word."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_const_b2"   # wrapper pad_const_bytes lib.rs:7116; pad_const_b4.slang:25 (b2 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]               # 2-byte elements share this byte-width variant (u16 carried as I16 slot; no U16 DType)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      fill_value: { kind: u32, note: "fill element bit pattern (low 16 bits for a 2-byte element)" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(n_out + n_in) * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy + constant store; no arithmetic, bit-exact for every 2-byte dtype."

determinism: bitwise
```

## pad_const_b4  (constant-fill pad — 4-byte elements)

One-line: constant-fill pad for 4-byte (f32/i32/u32) elements; copies the input region, fills pad slots with one constant word.

Constant-fill pad (`OpKind::Pad`, Constant mode) for 4-byte elements — the canonical variant whose
Slang source (`pad_const_b4.slang:25`) backs the family. One thread per output element: decompose the
linear output index into rank-N coords against `out_shape` (from `shape_buf`), shift by `left_pad`,
and copy the corresponding input word if in-bounds for all axes else store `fill_value`. Rank ≤ 8.
Numerics: none — a 32-bit word copy / constant store, bit-exact for any 4-byte dtype. Perf:
bandwidth-bound, one linear pass over the output. Limitations: src/out contiguous, offset 0; per-axis
`out_shape[i] == in_shape[i] + left_pad[i] + right_pad[i]`.

```fkc
kernel: pad_const_b4
op_kind: Pad                        # Constant mode (4B canonical)
blurb: "Constant-fill pad for 4-byte (f32/i32/u32) elements; copies the input region, fills pad slots with one constant word."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_const_b4"   # wrapper pad_const_bytes lib.rs:7116; pad_const_b4.slang:25
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]                # 4-byte elements
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      fill_value: { kind: u32, note: "fill element bit pattern (full 32 bits for a 4-byte element)" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(n_out + n_in) * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 32-bit word copy + constant store; no arithmetic, bit-exact for every 4-byte dtype."

determinism: bitwise
```

## pad_const_b8  (constant-fill pad — 8-byte elements)

One-line: constant-fill pad for 8-byte (f64/i64) elements; copies the input region, fills pad slots with one constant value.

Constant-fill pad (`OpKind::Pad`, Constant mode) for 8-byte elements. Same algorithm as the other
`pad_const_b*` variants keyed to an 8-byte element (handled as a 2-word / u32-pair move). One thread
per output element. Rank ≤ 8. Numerics: none — a byte copy / constant store, bit-exact for any 8-byte
dtype. Perf: bandwidth-bound, one linear pass over the output. Limitations: src/out contiguous,
offset 0; per-axis `out_shape[i] == in_shape[i] + left_pad[i] + right_pad[i]`. (The 64-bit
`fill_value` is supplied via the wrapper's parameter packing for the wide element.)

```fkc
kernel: pad_const_b8
op_kind: Pad                        # Constant mode (8B)
blurb: "Constant-fill pad for 8-byte (f64/i64) elements; copies the input region, fills pad slots with one constant value."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_const_b8"   # wrapper pad_const_bytes lib.rs:7116; pad_const_b4.slang:25 (b8 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]                     # 8-byte elements
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      fill_value: { kind: u32, note: "fill element bit pattern (8-byte element packed as a u32 pair by the wrapper)" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32                 # 8-byte elements addressed as u32 word pairs

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(n_out + n_in) * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy + constant store; no arithmetic, bit-exact for every 8-byte dtype."

determinism: bitwise
```

## pad_reflect_b1  (reflect pad — 1-byte elements)

One-line: reflect pad (edge not repeated) for 1-byte elements; maps each output element from its mirrored input position.

Reflect pad (`OpKind::Pad`, Reflect mode, PyTorch "reflect" — no-repeat) for 1-byte elements. One
thread per output element: per axis, for `i = out_coord - left_pad` and input dim `n`, the mirror map
is `i < 0 → -i`; `0 ≤ i < n → i`; `i ≥ n → 2*(n-1) - i` — the edge element is **not** repeated (a true
reflection about the boundary element). The thread reads the mapped input byte and stores it. Rank ≤ 8.
Numerics: none — byte copy via the reflect index map, bit-exact for any 1-byte dtype. Perf:
bandwidth-bound, one linear pass over the output (`n_out` mapped reads + writes).
**PRECONDITION (caller-enforced, load-bearing):** per-axis `left_pad ≤ in_dim-1` AND
`right_pad ≤ in_dim-1`, otherwise the reflection runs off the opposite side and produces out-of-range
indices; the dispatch wrapper validates this, the kernel does not re-check it. Limitations: src/out
contiguous, offset 0.

```fkc
kernel: pad_reflect_b1
op_kind: Pad                        # Reflect mode (1B)
blurb: "Reflect pad (edge not repeated) for 1-byte elements; maps each output element from its mirrored input position."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_reflect_b1"   # wrapper pad_reflect_bytes lib.rs:6835; pad_reflect_b4.slang:42 (b1 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]; per-axis left_pad<=in_dim-1 AND right_pad<=in_dim-1 (reflect validity)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"      # one mapped read + write per output element
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via reflect index map; no arithmetic, bit-exact for every 1-byte dtype. Caller must keep left_pad/right_pad <= in_dim-1 per axis."

determinism: bitwise
```

## pad_reflect_b2  (reflect pad — 2-byte elements)

One-line: reflect pad (edge not repeated) for 2-byte elements; maps each output element from its mirrored input position.

Reflect pad for 2-byte elements — same mirror map as `pad_reflect_b1` keyed to a 2-byte element. One
thread per output element; reads the mapped input word, stores it. Rank ≤ 8. Numerics: none, bit-exact.
Perf: bandwidth-bound, one pass over the output. Same **PRECONDITION**: per-axis `left_pad ≤ in_dim-1`
AND `right_pad ≤ in_dim-1` (wrapper-validated). Contiguous, offset 0.

```fkc
kernel: pad_reflect_b2
op_kind: Pad                        # Reflect mode (2B)
blurb: "Reflect pad (edge not repeated) for 2-byte elements; maps each output element from its mirrored input position."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_reflect_b2"   # wrapper pad_reflect_bytes lib.rs:6835; pad_reflect_b4.slang:42 (b2 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]               # 2-byte byte-width-keyed (u16 carried as I16 slot; no U16 DType)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]; reflect validity left_pad<=in_dim-1 AND right_pad<=in_dim-1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via reflect index map; no arithmetic, bit-exact for every 2-byte dtype. Caller must keep left_pad/right_pad <= in_dim-1 per axis."

determinism: bitwise
```

## pad_reflect_b4  (reflect pad — 4-byte elements)

One-line: reflect pad (edge not repeated) for 4-byte elements; maps each output element from its mirrored input position.

Reflect pad for 4-byte elements — the canonical variant whose Slang source (`pad_reflect_b4.slang:42`)
backs the family. One thread per output element; per-axis mirror map (`i<0→-i`, `0≤i<n→i`,
`i≥n→2*(n-1)-i`), reads the mapped input word, stores it. Rank ≤ 8. Numerics: none, bit-exact. Perf:
bandwidth-bound, one pass over the output. Same **PRECONDITION**: per-axis `left_pad ≤ in_dim-1` AND
`right_pad ≤ in_dim-1` (wrapper-validated). Contiguous, offset 0.

```fkc
kernel: pad_reflect_b4
op_kind: Pad                        # Reflect mode (4B canonical)
blurb: "Reflect pad (edge not repeated) for 4-byte elements; maps each output element from its mirrored input position."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_reflect_b4"   # wrapper pad_reflect_bytes lib.rs:6835; pad_reflect_b4.slang:42
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]; reflect validity left_pad<=in_dim-1 AND right_pad<=in_dim-1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 32-bit word copy via reflect index map; no arithmetic, bit-exact for every 4-byte dtype. Caller must keep left_pad/right_pad <= in_dim-1 per axis."

determinism: bitwise
```

## pad_reflect_b8  (reflect pad — 8-byte elements)

One-line: reflect pad (edge not repeated) for 8-byte elements; maps each output element from its mirrored input position.

Reflect pad for 8-byte elements — same mirror map keyed to an 8-byte element (u32-pair move). One
thread per output element. Rank ≤ 8. Numerics: none, bit-exact. Perf: bandwidth-bound, one pass over
the output. Same **PRECONDITION**: per-axis `left_pad ≤ in_dim-1` AND `right_pad ≤ in_dim-1`
(wrapper-validated). Contiguous, offset 0.

```fkc
kernel: pad_reflect_b8
op_kind: Pad                        # Reflect mode (8B)
blurb: "Reflect pad (edge not repeated) for 8-byte elements; maps each output element from its mirrored input position."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_reflect_b8"   # wrapper pad_reflect_bytes lib.rs:6835; pad_reflect_b4.slang:42 (b8 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]; reflect validity left_pad<=in_dim-1 AND right_pad<=in_dim-1" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via reflect index map; no arithmetic, bit-exact for every 8-byte dtype. Caller must keep left_pad/right_pad <= in_dim-1 per axis."

determinism: bitwise
```

## pad_replicate_b1  (replicate / edge-repeat pad — 1-byte elements)

One-line: replicate (edge-repeat) pad for 1-byte elements; out-of-range coords clamp to [0, in_dim-1].

Replicate / edge-repeat pad (`OpKind::Pad`, Replicate mode) for 1-byte elements. One thread per
output element: per axis, for `i = out_coord - left_pad`, the clamp map is `i < 0 → 0`;
`0 ≤ i < n → i`; `i ≥ n → n-1` — the boundary element is clamped/repeated outward. Reads the mapped
input byte and stores it. Rank ≤ 8. **No precondition on pad sizes** (clamping is well-defined for any
pad width — unlike Reflect). Numerics: none — byte copy via the clamp index map, bit-exact for any
1-byte dtype. Perf: bandwidth-bound, one linear pass over the output. Limitations: src/out contiguous,
offset 0. (Embedded SPIR-V only — `pad_replicate_b*.spv`; contract read from the Rust wrapper +
`EMBEDDED` doc comments.)

```fkc
kernel: pad_replicate_b1
op_kind: Pad                        # Replicate mode (1B)
blurb: "Replicate (edge-repeat) pad for 1-byte elements; out-of-range coords clamp to [0, in_dim-1]."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_replicate_b1"   # wrapper pad_replicate_bytes lib.rs:6710; SPIR-V pad_replicate_b4.spv (b1 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad] (no pad-size precondition for replicate)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via replicate (edge-clamp) index map; no arithmetic, bit-exact for every 1-byte dtype."

determinism: bitwise
```

## pad_replicate_b2  (replicate / edge-repeat pad — 2-byte elements)

One-line: replicate (edge-repeat) pad for 2-byte elements; out-of-range coords clamp to [0, in_dim-1].

Replicate pad for 2-byte elements — same edge-clamp map as `pad_replicate_b1` keyed to a 2-byte
element. One thread per output element; reads the clamped input word, stores it. Rank ≤ 8. No
precondition on pad sizes. Numerics: none, bit-exact. Perf: bandwidth-bound, one pass over the output.
Contiguous, offset 0. (Embedded SPIR-V only.)

```fkc
kernel: pad_replicate_b2
op_kind: Pad                        # Replicate mode (2B)
blurb: "Replicate (edge-repeat) pad for 2-byte elements; out-of-range coords clamp to [0, in_dim-1]."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_replicate_b2"   # wrapper pad_replicate_bytes lib.rs:6710; SPIR-V pad_replicate_b4.spv (b2 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]               # 2-byte byte-width-keyed (u16 carried as I16 slot; no U16 DType)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad] (no pad-size precondition)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via replicate (edge-clamp) index map; no arithmetic, bit-exact for every 2-byte dtype."

determinism: bitwise
```

## pad_replicate_b4  (replicate / edge-repeat pad — 4-byte elements)

One-line: replicate (edge-repeat) pad for 4-byte elements; out-of-range coords clamp to [0, in_dim-1].

Replicate pad for 4-byte elements — the canonical variant. One thread per output element; per-axis
edge-clamp map (`i<0→0`, `0≤i<n→i`, `i≥n→n-1`), reads the clamped input word, stores it. Rank ≤ 8. No
precondition on pad sizes. Numerics: none, bit-exact. Perf: bandwidth-bound, one pass over the output.
Contiguous, offset 0. (Embedded SPIR-V only — `pad_replicate_b4.spv`.)

```fkc
kernel: pad_replicate_b4
op_kind: Pad                        # Replicate mode (4B canonical)
blurb: "Replicate (edge-repeat) pad for 4-byte elements; out-of-range coords clamp to [0, in_dim-1]."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_replicate_b4"   # wrapper pad_replicate_bytes lib.rs:6710; SPIR-V pad_replicate_b4.spv
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad] (no pad-size precondition)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 32-bit word copy via replicate (edge-clamp) index map; no arithmetic, bit-exact for every 4-byte dtype."

determinism: bitwise
```

## pad_replicate_b8  (replicate / edge-repeat pad — 8-byte elements)

One-line: replicate (edge-repeat) pad for 8-byte elements; out-of-range coords clamp to [0, in_dim-1].

Replicate pad for 8-byte elements — same edge-clamp map keyed to an 8-byte element (u32-pair move).
One thread per output element. Rank ≤ 8. No precondition on pad sizes. Numerics: none, bit-exact.
Perf: bandwidth-bound, one pass over the output. Contiguous, offset 0. (Embedded SPIR-V only.)

```fkc
kernel: pad_replicate_b8
op_kind: Pad                        # Replicate mode (8B)
blurb: "Replicate (edge-repeat) pad for 8-byte elements; out-of-range coords clamp to [0, in_dim-1]."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_replicate_b8"   # wrapper pad_replicate_bytes lib.rs:6710; SPIR-V pad_replicate_b4.spv (b8 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad] (no pad-size precondition)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: from_params(out_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via replicate (edge-clamp) index map; no arithmetic, bit-exact for every 8-byte dtype."

determinism: bitwise
```

## pad_backward_const_b1  (Constant-pad backward — 1-byte elements)

One-line: Constant-pad gradient for 1-byte elements; one thread per input element reads grad_out at in_coord+left_pad (no accumulation).

Backward of Constant pad (`OpKind::PadBackward`, Constant mode) for 1-byte elements. Because Constant
pad maps each input element to **exactly one** output position (the pad slots come from the constant,
not the input), the backward is a simple **gather, not an accumulate**: one thread per *input* element
reads `grad_out` at the shifted position `in_coord + left_pad` (computed per-axis from `shape_buf`) and
writes it straight to `grad_in`. No atomics, no reduction. Rank ≤ 8. Numerics: none — byte copy,
bit-exact for any 1-byte dtype. Perf: bandwidth-bound, one linear pass over the input (`n_in` mapped
reads + writes). Output behavior: `grad_in` is **fully written** (every input slot gets exactly one
gradient), so `aliasing: none`. Limitations: grad_out/grad_in contiguous, offset 0.

```fkc
kernel: pad_backward_const_b1
op_kind: PadBackward                # Constant-mode backward (1B, gather)
blurb: "Constant-pad gradient for 1-byte elements; one thread per input element reads grad_out at in_coord+left_pad (no accumulation)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_const_b1"   # wrapper pad_backward_const_bytes lib.rs:6594; pad_backward_const_b4.slang:28 (b1 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in"
  op_params:
    variant: PadBackward            # OpParams::PadBackward (primitive namespace)
    fields:
      n_in:       { kind: u32, constraint: "== product(in_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # byte width preserved
      shape_rule: from_params(in_shape)        # the (smaller) input shape
      layout_guarantee: contiguous
      aliasing: none                           # one gradient per input slot; full write, not RMW

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"                                  # pure gather copy, no arithmetic
  bytes_moved: "2 * n_in * dtype_bytes"       # one mapped read of grad_out + write of grad_in per input element
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true           # gather copy, no accumulation: bit-exact, any hardware
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Constant-pad backward is a 1:1 gather (no accumulation); pure byte copy, bit-exact for every 1-byte dtype."

determinism: bitwise
```

## pad_backward_const_b2  (Constant-pad backward — 2-byte elements)

One-line: Constant-pad gradient for 2-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation).

Backward of Constant pad for 2-byte elements — same 1:1 gather as `pad_backward_const_b1` keyed to a
2-byte element. One thread per input element reads `grad_out[in_coord + left_pad]`, writes `grad_in`.
No atomics. Rank ≤ 8. Numerics: none, bit-exact. Perf: bandwidth-bound, one pass over the input.
`grad_in` fully written; `aliasing: none`. Contiguous, offset 0.

```fkc
kernel: pad_backward_const_b2
op_kind: PadBackward                # Constant-mode backward (2B)
blurb: "Constant-pad gradient for 2-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_const_b2"   # wrapper pad_backward_const_bytes lib.rs:6594; pad_backward_const_b4.slang:28 (b2 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F16, BF16, I16]               # 2-byte byte-width-keyed (u16 carried as I16 slot; no U16 DType)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in"
  op_params:
    variant: PadBackward
    fields:
      n_in:       { kind: u32, constraint: "== product(in_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_in * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Constant-pad backward is a 1:1 gather (no accumulation); pure byte copy, bit-exact for every 2-byte dtype."

determinism: bitwise
```

## pad_backward_const_b4  (Constant-pad backward — 4-byte elements)

One-line: Constant-pad gradient for 4-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation).

Backward of Constant pad for 4-byte elements — the canonical variant whose Slang source
(`pad_backward_const_b4.slang:28`) backs the family. One thread per input element reads
`grad_out[in_coord + left_pad]` (per-axis shift from `shape_buf`), writes `grad_in`. No atomics
(Constant maps 1:1). Rank ≤ 8. Numerics: none, bit-exact. Perf: bandwidth-bound, one pass over the
input. `grad_in` fully written; `aliasing: none`. Contiguous, offset 0.

```fkc
kernel: pad_backward_const_b4
op_kind: PadBackward                # Constant-mode backward (4B canonical)
blurb: "Constant-pad gradient for 4-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_const_b4"   # wrapper pad_backward_const_bytes lib.rs:6594; pad_backward_const_b4.slang:28
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in"
  op_params:
    variant: PadBackward
    fields:
      n_in:       { kind: u32, constraint: "== product(in_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_in * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Constant-pad backward is a 1:1 gather (no accumulation); pure 32-bit word copy, bit-exact for every 4-byte dtype."

determinism: bitwise
```

## pad_backward_const_b8  (Constant-pad backward — 8-byte elements)

One-line: Constant-pad gradient for 8-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation).

Backward of Constant pad for 8-byte elements — same 1:1 gather keyed to an 8-byte element (u32-pair
move). One thread per input element. No atomics. Rank ≤ 8. Numerics: none, bit-exact. Perf:
bandwidth-bound, one pass over the input. `grad_in` fully written; `aliasing: none`. Contiguous,
offset 0.

```fkc
kernel: pad_backward_const_b8
op_kind: PadBackward                # Constant-mode backward (8B)
blurb: "Constant-pad gradient for 8-byte elements; one thread per input element gathers grad_out at in_coord+left_pad (no accumulation)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_const_b8"   # wrapper pad_backward_const_bytes lib.rs:6594; pad_backward_const_b4.slang:28 (b8 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in"
  op_params:
    variant: PadBackward
    fields:
      n_in:       { kind: u32, constraint: "== product(in_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * n_in * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Constant-pad backward is a 1:1 gather (no accumulation); pure byte copy, bit-exact for every 8-byte dtype."

determinism: bitwise
```

## pad_backward_reflect_f32  (Reflect-pad backward — f32, atomic accumulate)

One-line: Reflect-pad gradient on f32; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (uint CAS).

Backward of Reflect pad (`OpKind::PadBackward`, Reflect mode) on `f32`. Reflect forward maps multiple
output positions onto one input slot, so the backward **sums** gradients into each `grad_in` slot:
one thread per *output* element computes the mirrored input position (the same reflect map as the
forward — `i<0→-i`, `0≤i<n→i`, `i≥n→2*(n-1)-i` per axis from `shape_buf`) and atomically adds
`grad_out` into that `grad_in` slot via a **uint compare-and-swap** read-modify-write (bounded
1000-iteration CAS loop; under extreme contention a value may be dropped). `grad_in` **must be
zero-filled by the wrapper before dispatch** — the kernel only accumulates. Rank ≤ 8. Numerics: f32
add. Perf: bandwidth-bound over `n_out` reads + CAS RMWs into `n_in` slots. Output behavior: `grad_in`
is **read-modify-written / atomic-accumulated** (`aliasing: accumulate(grad_in)`). Determinism: the
CAS accumulation **order is scheduler-dependent** and f32 addition is not associative, so the kernel
is **nondeterministic** (audited, no static bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_reflect_f32
op_kind: PadBackward                # Reflect-mode backward, f32 (atomic accumulate)
blurb: "Reflect-pad gradient on f32; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (uint CAS)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_reflect_f32"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_reflect_f32.slang:51
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward            # OpParams::PadBackward (primitive namespace)
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F32 in → F32 out
      shape_rule: from_params(in_shape)        # the (smaller) input shape
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)            # atomic RMW into a wrapper-zeroed grad_in buffer

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false                            # output buffer is a distinct, wrapper-zeroed accumulator
  alignment_bytes: 16
  access_granularity_bits: 32                 # f32 element granular (uint CAS)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"                              # one f32 atomic add per output element
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"   # read grad_out + CAS read-write of grad_in slot
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false          # scheduler-dependent CAS accumulation order; f32 add non-associative
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                               # audited none(reason): atomic FP accumulation order is nondeterministic
  notes: "Atomic uint-CAS accumulate (bounded 1000-iter loop); scheduler-dependent summation order, f32 non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic                 # atomic FP accumulation; run-to-run variation possible
```

## pad_backward_reflect_f64  (Reflect-pad backward — f64, atomic accumulate)

One-line: Reflect-pad gradient on f64; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (u64 CAS).

Backward of Reflect pad on `f64`. Identical structure to `pad_backward_reflect_f32` but the atomic is
a **u64 compare-and-swap** (requires `shaderInt64` + 64-bit atomics + f64 device support). One thread
per output element maps to the mirrored input slot and CAS-accumulates `grad_out` into `grad_in`
(bounded 1000-iter loop). `grad_in` **must be zero-filled by the wrapper before dispatch**. Rank ≤ 8.
Numerics: native f64 add. Perf: bandwidth-bound over `n_out` reads + CAS RMWs. Output: atomic
accumulate (`aliasing: accumulate(grad_in)`). Determinism: scheduler-dependent CAS order, f64 add
non-associative ⇒ **nondeterministic** (audited, no bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_reflect_f64
op_kind: PadBackward                # Reflect-mode backward, f64 (atomic accumulate)
blurb: "Reflect-pad gradient on f64; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (u64 CAS)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_reflect_f64"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_reflect_f32.slang:51 (f64 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F64 in → F64 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64                 # f64 element granular (u64 CAS; needs shaderInt64 + 64-bit atomics + f64)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Atomic u64-CAS accumulate (bounded 1000-iter loop); scheduler-dependent summation order, f64 non-associative, not bit-stable. Needs shaderInt64+64-bit atomics+f64. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_reflect_f16  (Reflect-pad backward — f16, atomic accumulate)

One-line: Reflect-pad gradient on f16; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (sub-word CAS, f32 math).

Backward of Reflect pad on `f16`. Same mirrored-slot accumulation as the f32/f64 variants, but the
atomic is a **sub-word compare-and-swap**: each thread reads the enclosing 32-bit word, widens the
target f16 lane to f32, adds the (f32-widened) `grad_out`, narrows back to f16, and CAS-writes the
updated word (bounded 1000-iter loop; sub-word so a half-word is updated without racing the other
lane). `grad_in` **must be zero-filled by the wrapper before dispatch**. Rank ≤ 8. Numerics: math at
f32, narrow to f16 on store. Perf: bandwidth-bound over `n_out` reads + CAS RMWs. Output: atomic
accumulate (`aliasing: accumulate(grad_in)`). Determinism: scheduler-dependent CAS order + non-assoc
f32 add ⇒ **nondeterministic** (audited, no bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_reflect_f16
op_kind: PadBackward                # Reflect-mode backward, f16 (sub-word atomic accumulate)
blurb: "Reflect-pad gradient on f16; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (sub-word CAS, f32 math)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_reflect_f16"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_reflect_f32.slang:51 (f16 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F16 in → F16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16                 # f16 element granular (sub-word CAS on the enclosing u32)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"                              # one f32-widened add per output element
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Sub-word CAS accumulate, math at f32, narrow to f16 on store; scheduler-dependent order, non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_reflect_bf16  (Reflect-pad backward — bf16, atomic accumulate)

One-line: Reflect-pad gradient on bf16; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (sub-word CAS, f32 math).

Backward of Reflect pad on `bf16`. Identical to `pad_backward_reflect_f16` with the bf16 half type:
sub-word CAS, widen the bf16 lane to f32, add the f32-widened `grad_out`, narrow back to bf16, CAS the
enclosing word (bounded 1000-iter loop). `grad_in` **must be zero-filled by the wrapper before
dispatch**. Rank ≤ 8. Numerics: math at f32, narrow to bf16 on store. Perf: bandwidth-bound. Output:
atomic accumulate. Determinism: **nondeterministic** (scheduler-dependent CAS order; audited, no
bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_reflect_bf16
op_kind: PadBackward                # Reflect-mode backward, bf16 (sub-word atomic accumulate)
blurb: "Reflect-pad gradient on bf16; one thread per output element atomic-accumulates grad_out into the mirrored grad_in slot (sub-word CAS, f32 math)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_reflect_bf16"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_reflect_f32.slang:51 (bf16 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # BF16 in → BF16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16                 # bf16 element granular (sub-word CAS)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Sub-word CAS accumulate, math at f32, narrow to bf16 on store; scheduler-dependent order, non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_replicate_f32  (Replicate-pad backward — f32, atomic accumulate)

One-line: Replicate-pad gradient on f32; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (uint CAS).

Backward of Replicate pad (`OpKind::PadBackward`, Replicate mode) on `f32`. Replicate forward folds
many output positions onto the boundary input slots (every clamped coordinate maps to an edge
element), so the backward **sums** gradients: one thread per output element computes the edge-clamp
input position (the same replicate map — `i<0→0`, `0≤i<n→i`, `i≥n→n-1` per axis from `shape_buf`) and
atomically adds `grad_out` into that `grad_in` slot via a **uint compare-and-swap** RMW (bounded
1000-iter loop). `grad_in` **must be zero-filled by the wrapper before dispatch**. Rank ≤ 8.
Numerics: f32 add. Perf: bandwidth-bound over `n_out` reads + CAS RMWs. Output: atomic accumulate
(`aliasing: accumulate(grad_in)`). Determinism: scheduler-dependent CAS order, f32 non-associative ⇒
**nondeterministic** (audited, no bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_replicate_f32
op_kind: PadBackward                # Replicate-mode backward, f32 (atomic accumulate)
blurb: "Replicate-pad gradient on f32; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (uint CAS)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_replicate_f32"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_replicate_f32.slang (with reflect at :51)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F32 in → F32 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32                 # f32 element granular (uint CAS)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Atomic uint-CAS accumulate (bounded 1000-iter loop); scheduler-dependent order, f32 non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_replicate_f64  (Replicate-pad backward — f64, atomic accumulate)

One-line: Replicate-pad gradient on f64; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (u64 CAS).

Backward of Replicate pad on `f64`. Same edge-clamp folding as the f32 variant but with a **u64
compare-and-swap** (requires `shaderInt64` + 64-bit atomics + f64). One thread per output element maps
to the edge-clamped input slot and CAS-accumulates `grad_out` into `grad_in` (bounded 1000-iter loop).
`grad_in` **must be zero-filled by the wrapper before dispatch**. Rank ≤ 8. Numerics: native f64 add.
Perf: bandwidth-bound. Output: atomic accumulate. Determinism: scheduler-dependent CAS order, f64
non-associative ⇒ **nondeterministic** (audited, no bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_replicate_f64
op_kind: PadBackward                # Replicate-mode backward, f64 (atomic accumulate)
blurb: "Replicate-pad gradient on f64; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (u64 CAS)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_replicate_f64"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_replicate_f64.slang
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F64 in → F64 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64                 # f64 element granular (u64 CAS; needs shaderInt64 + 64-bit atomics + f64)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Atomic u64-CAS accumulate (bounded 1000-iter loop); scheduler-dependent order, f64 non-associative, not bit-stable. Needs shaderInt64+64-bit atomics+f64. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_replicate_f16  (Replicate-pad backward — f16, atomic accumulate)

One-line: Replicate-pad gradient on f16; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (sub-word CAS, f32 math).

Backward of Replicate pad on `f16`. Same edge-clamp folding as the f32/f64 variants, with a **sub-word
compare-and-swap**: read the enclosing 32-bit word, widen the target f16 lane to f32, add the
f32-widened `grad_out`, narrow to f16, CAS the word (bounded 1000-iter loop; sub-word lane update).
`grad_in` **must be zero-filled by the wrapper before dispatch**. Rank ≤ 8. Numerics: math at f32,
narrow to f16 on store. Perf: bandwidth-bound. Output: atomic accumulate. Determinism:
**nondeterministic** (scheduler-dependent order; audited, no bound). Contiguous, offset 0.

```fkc
kernel: pad_backward_replicate_f16
op_kind: PadBackward                # Replicate-mode backward, f16 (sub-word atomic accumulate)
blurb: "Replicate-pad gradient on f16; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (sub-word CAS, f32 math)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_replicate_f16"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_replicate_f16.slang
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F16 in → F16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16                 # f16 element granular (sub-word CAS)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Sub-word CAS accumulate, math at f32, narrow to f16 on store; scheduler-dependent order, non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## pad_backward_replicate_bf16  (Replicate-pad backward — bf16, atomic accumulate)

One-line: Replicate-pad gradient on bf16; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (sub-word CAS, f32 math).

Backward of Replicate pad on `bf16`. Identical to `pad_backward_replicate_f16` with the bf16 half
type: sub-word CAS, widen the bf16 lane to f32, add the f32-widened `grad_out`, narrow to bf16, CAS the
enclosing word (bounded 1000-iter loop). `grad_in` **must be zero-filled by the wrapper before
dispatch**. Rank ≤ 8. Numerics: math at f32, narrow to bf16 on store. Perf: bandwidth-bound. Output:
atomic accumulate. Determinism: **nondeterministic** (scheduler-dependent order; audited, no bound).
Contiguous, offset 0.

```fkc
kernel: pad_backward_replicate_bf16
op_kind: PadBackward                # Replicate-mode backward, bf16 (sub-word atomic accumulate)
blurb: "Replicate-pad gradient on bf16; one thread per output element atomic-accumulates grad_out into the edge-clamped grad_in slot (sub-word CAS, f32 math)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::pad_backward_replicate_bf16"   # wrapper pad_backward_atomic_bytes lib.rs:6462; pad_backward_replicate_bf16.slang
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_rank=grad_in; element_count == n_out"
  op_params:
    variant: PadBackward
    fields:
      n_out:      { kind: u32, constraint: "== product(out_shape)" }
      rank:       { kind: u32, constraint: "<= 8" }
      shape_buf:  { kind: "buffer<u32>", note: "[in_shape, out_shape, left_pad]" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # BF16 in → BF16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: accumulate(grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16                 # bf16 element granular (sub-word CAS)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n_out"
  bytes_moved: "n_out * dtype_bytes + 2 * n_out * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: "n_in * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Sub-word CAS accumulate, math at f32, narrow to bf16 on store; scheduler-dependent order, non-associative, not bit-stable. grad_in must be zero-filled by the wrapper before dispatch."

determinism: nondeterministic
```

## masked_fill_b1  (masked fill — 1-byte elements)

One-line: masked fill for 1-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8).

Masked fill (`OpKind::MaskedFill`) for 1-byte elements: elementwise `out[i] = mask[i] != 0 ? fill :
input[i]`. One thread per element, fully 1:1 (no shape/stride decomposition — pure positional walk).
The **mask is always U8**, packed 4 bytes per u32 word; for the b1 variant the element and the mask
byte are read from packed words and the selected byte is stored. Numerics: none — a select between the
input byte and the `fill_value` bit pattern, bit-exact for any 1-byte dtype. Perf: bandwidth-bound,
one linear pass over the input (`n` reads of input + mask, `n` writes). Limitations: input + mask +
output all **contiguous, offset 0**, same element count; no broadcasting.

```fkc
kernel: masked_fill_b1
op_kind: MaskedFill                 # 1-byte element variant
blurb: "Masked fill for 1-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::masked_fill_b1"   # wrapper masked_fill_bytes lib.rs:6982; masked_fill_b4.slang:16 (b1 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8]                       # 1-byte elements
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out; same_as=mask"
    - name: mask
      dtypes: [U8]                           # mask is ALWAYS U8 (nonzero = fill)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=input"
  op_params:
    variant: MaskedFill             # OpParams::MaskedFill (primitive namespace)
    fields:
      n:          { kind: u32, constraint: "== product(input shape)" }
      fill_value: { kind: u32, note: "fill element bit pattern (low 8 bits for a 1-byte element)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # byte width preserved; dtype = input dtype
      shape_rule: same_as(input)              # elementwise, same shape
      layout_guarantee: contiguous
      aliasing: none                          # fresh output; full overwrite (every slot selected)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32                 # u32-word addressed (packed bytes + mask)

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"                                  # pure select, no arithmetic
  bytes_moved: "(2 * n * dtype_bytes) + n"    # read input + write out (dtype_bytes each) + read mask (1 byte/elem, U8)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true           # byte select: bit-exact for any dtype, any hardware
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Elementwise select between input byte and pre-encoded fill bits; no arithmetic, bit-exact for every 1-byte dtype. Mask always U8."

determinism: bitwise                          # exact data-dependent select, hardware-independent
```

## masked_fill_b2  (masked fill — 2-byte elements)

One-line: masked fill for 2-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8).

Masked fill for 2-byte elements — same elementwise select keyed to a 2-byte element. One thread per
element, 1:1; the mask is **always U8** (packed 4-per-u32), the element/mask are read from packed
words. Numerics: none — select between the input word and `fill_value` low 16 bits, bit-exact for any
2-byte dtype. Perf: bandwidth-bound, one pass. Limitations: input + mask + output contiguous, offset
0, same element count; no broadcasting.

```fkc
kernel: masked_fill_b2
op_kind: MaskedFill                 # 2-byte element variant
blurb: "Masked fill for 2-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::masked_fill_b2"   # wrapper masked_fill_bytes lib.rs:6982; masked_fill_b4.slang:16 (b2 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]               # 2-byte byte-width-keyed (u16 carried as I16 slot; no U16 DType)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out; same_as=mask"
    - name: mask
      dtypes: [U8]                           # mask is ALWAYS U8
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=input"
  op_params:
    variant: MaskedFill
    fields:
      n:          { kind: u32, constraint: "== product(input shape)" }
      fill_value: { kind: u32, note: "fill element bit pattern (low 16 bits for a 2-byte element)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(2 * n * dtype_bytes) + n"    # read input + write out + read mask (1 byte/elem, U8)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Elementwise select between input word and pre-encoded fill bits; no arithmetic, bit-exact for every 2-byte dtype. Mask always U8."

determinism: bitwise
```

## masked_fill_b4  (masked fill — 4-byte elements)

One-line: masked fill for 4-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8).

Masked fill for 4-byte elements — the canonical variant whose Slang source (`masked_fill_b4.slang:16`)
backs the family. One thread per element, 1:1; reads `input[i]` and the mask word indexed `i>>2`
(masks packed 4-per-u32), selects `mask != 0 ? fill_value : input[i]`, stores it. Numerics: none —
32-bit word select, bit-exact for any 4-byte dtype. Perf: bandwidth-bound, one pass. Limitations:
input + mask + output contiguous, offset 0, same element count; no broadcasting.

```fkc
kernel: masked_fill_b4
op_kind: MaskedFill                 # 4-byte element variant (canonical)
blurb: "Masked fill for 4-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::masked_fill_b4"   # wrapper masked_fill_bytes lib.rs:6982; masked_fill_b4.slang:16
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out; same_as=mask"
    - name: mask
      dtypes: [U8]                           # mask is ALWAYS U8 (word indexed i>>2)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=input"
  op_params:
    variant: MaskedFill
    fields:
      n:          { kind: u32, constraint: "== product(input shape)" }
      fill_value: { kind: u32, note: "fill element bit pattern (full 32 bits for a 4-byte element)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(2 * n * dtype_bytes) + n"    # read input + write out + read mask (1 byte/elem, U8)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Elementwise 32-bit word select between input and pre-encoded fill bits; no arithmetic, bit-exact for every 4-byte dtype. Mask always U8 (word indexed i>>2)."

determinism: bitwise
```

## masked_fill_b8  (masked fill — 8-byte elements)

One-line: masked fill for 8-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8).

Masked fill for 8-byte elements — same elementwise select keyed to an 8-byte element (handled as a
u32-pair move). One thread per element, 1:1; the mask is **always U8**. Numerics: none — select
between the input value and `fill_value`, bit-exact for any 8-byte dtype. Perf: bandwidth-bound, one
pass. Limitations: input + mask + output contiguous, offset 0, same element count; no broadcasting.

```fkc
kernel: masked_fill_b8
op_kind: MaskedFill                 # 8-byte element variant
blurb: "Masked fill for 8-byte elements; out[i] = mask[i] != 0 ? fill : input[i] (mask is U8)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::masked_fill_b8"   # wrapper masked_fill_bytes lib.rs:6982; masked_fill_b4.slang:16 (b8 variant)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out; same_as=mask"
    - name: mask
      dtypes: [U8]                           # mask is ALWAYS U8
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=input"
  op_params:
    variant: MaskedFill
    fields:
      n:          { kind: u32, constraint: "== product(input shape)" }
      fill_value: { kind: u32, note: "fill element bit pattern (8-byte element packed as a u32 pair by the wrapper)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32                 # 8-byte element addressed as a u32 word pair

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "(2 * n * dtype_bytes) + n"    # read input + write out + read mask (1 byte/elem, U8)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Elementwise select between input value and pre-encoded fill bits; no arithmetic, bit-exact for every 8-byte dtype. Mask always U8."

determinism: bitwise
```

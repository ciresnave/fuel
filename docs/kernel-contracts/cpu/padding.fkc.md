---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — padding kernel contracts

Portable byte-shaped CPU padding kernels from `fuel-cpu-backend/src/byte_kernels.rs`. All operate
on flat contiguous `CpuStorageBytes` slices (offset 0, row-major); the pipelined executor's
auto-Contiguize pass realizes any strided/broadcast/offset producer into a dense buffer *before*
these kernels run, so the input-layout contract for every kernel here is **contiguous, offset 0**
(`reverse_strides: rejected` throughout — none of these kernels consults a `Layout` or walks a
signed stride; they compute their own non-negative row-major strides via
`compute_row_major_strides`). Shape/stride semantics arrive as explicit `usize` params
(`in_shape` / `out_shape` / `padding` / `mode_tag`), never derived from a layout object.

The three forward modes (Constant / Reflect / Replicate) share **one** `OpKind::Pad` selected by
the `mode_tag` field of `OpParams::Pad`; per-dtype byte width is carried by `dtype_size` +
pre-encoded `fill_bytes`, so the forward kernels are **dtype-agnostic** (one logical kernel per
mode). Backward is per-dtype (`OpKind::PadBackward`) because gradient accumulation needs typed
addition; bf16/f16 (and f32) widen the accumulator scratch to **f64**.

Cost on every kernel is `provenance: judge_measured` — the Judge bootstraps it. A bandwidth
formula hint is given where derivable (these are pure byte movers / accumulators, bandwidth-bound
over element counts); no FLOPs/latency numbers are fabricated.

## pad_const_cpu  (multi-dim Constant pad)

One-line: multi-dim constant-fill pad; tiles the output with one constant element then copies the input region in.

Multi-dimensional Constant (`mode_tag = 0`) pad. Two-pass, dtype-agnostic at the byte level:
**pass 1** copies the single-element `fill_bytes` pattern into every one of the `out_elem` output
slots; **pass 2** walks the input's multi-index via a mixed-radix counter (last axis varies
fastest, no recursion, arbitrary rank) and copies each input element's `dtype_size` bytes into its
shifted output position `out_flat = Σ (idx[k] + padding[k].before) * out_stride[k]`. Padded slots
keep the constant — they never index into the input. Numerics: none — it is a byte copy, so it is
bit-exact for any dtype (the fill value's bits are supplied pre-encoded by the caller). Perf:
bandwidth-bound, two linear passes over the output plus one over the input
(`out_elem` fills + `in_elem` element copies). Validates rank agreement
(`in_shape.len() == out_shape.len() == padding.len()`), `fill_bytes.len() == dtype_size`, and exact
byte lengths (`in == in_elem*dtype_size`, `out == out_elem*dtype_size`), returning `Result` (no
panic). Limitations: contiguous + offset-0 only; no symbolic-extent awareness (reads full concrete
shapes); `out_shape[i]` must equal `in_shape[i] + before + after` per axis (the dispatch wrapper
guarantees this).

```fkc
kernel: pad_const_cpu
op_kind: Pad                        # mode_tag = 0 distinguishes Constant from Reflect/Replicate
blurb: "Multi-dim constant-fill pad; tiles the output with one constant element then copies the input region in."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_const_cpu"   # byte_kernels.rs:497
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]   # dtype-agnostic byte copy (any element width)
      dtype_class: any
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad                    # OpParams::Pad (fuel-dispatch::kernel) — primitive namespace
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "== 0  (Constant)" }
      fill_bytes: { kind: "Vec<u8>", constraint: "len == dtype_size (one element pre-encoded)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # byte width preserved; dtype = input dtype
      shape_rule: from_params(out_shape)     # per-axis in_shape[i] + before + after
      layout_guarantee: contiguous           # fresh dense row-major; executor pre-allocates
      aliasing: none                         # full overwrite (pass 1 fills every slot)

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8                  # byte-granular copy (dtype-agnostic)

cost:
  provenance: judge_measured                 # Judge bootstraps; bandwidth-bound byte mover
  class: cheap_elementwise
  flops: "0"                                  # pure copy, no arithmetic
  bytes_moved: "(2 * out_elem + 2 * in_elem) * dtype_bytes"   # fill writes out_elem; copy reads+writes in_elem
  overhead_ns: ~                              # judge_measured
  memory: { device_bytes: 0, host_bytes: "out_elem * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true          # byte copy: bit-exact for any dtype, any hardware
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy (fill pattern + input region); no arithmetic, bit-exact for every dtype."

determinism: bitwise                         # exact byte shuffle/copy, hardware-independent
```

## pad_reflect_cpu  (multi-dim Reflect pad)

One-line: multi-dim reflect pad (edge not repeated); delegates to the generic pad walker with the reflect index map.

Multi-dimensional Reflect (`mode_tag = 1`) pad. A thin wrapper that delegates to `pad_walk_cpu`
(below) with `reflect_index`. Per-axis mapping for output index `j`, input dim `n`,
`i = j - before`: `i < 0 → in[-i]`; `0 ≤ i < n → in[i]`; `i ≥ n → in[2*(n-1) - i]` — i.e. the edge
element is **not** repeated (a true reflection about the boundary element). Dtype-agnostic byte
copy. Numerics: none (byte copy, bit-exact). Perf: bandwidth-bound, one linear pass over the output
(`out_elem` mapped reads + writes). **Limitation (caller-enforced, load-bearing):** `before ≤ n-1`
and `after ≤ n-1` for every axis — otherwise the reflection runs off the opposite side and
produces out-of-range indices; the dispatch wrapper validates this, the kernel does not re-check it.
Contiguous + offset-0 only.

```fkc
kernel: pad_reflect_cpu
op_kind: Pad                        # mode_tag = 1 distinguishes Reflect
blurb: "Multi-dim reflect pad (edge not repeated); delegates to the generic pad walker with the reflect index map."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_reflect_cpu"   # byte_kernels.rs:580 → pad_walk_cpu:631
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]
      dtype_class: any
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len(); before <= in_shape[i]-1 AND after <= in_shape[i]-1 (reflect validity)" }
      mode_tag:  { kind: u8, constraint: "== 1  (Reflect)" }
      fill_bytes: { kind: "Vec<u8>", constraint: "ignored for Reflect (no constant fill)" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * out_elem * dtype_bytes"   # one mapped read + write per output element
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "out_elem * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via reflect index map; no arithmetic, bit-exact for every dtype. Caller must keep before/after <= n-1 per axis."

determinism: bitwise
```

## pad_replicate_cpu  (multi-dim Replicate / edge-repeat pad)

One-line: multi-dim replicate pad (edge repeated); delegates to the generic pad walker with the replicate index map.

Multi-dimensional Replicate / edge-repeat (`mode_tag = 2`) pad. A thin wrapper that delegates to
`pad_walk_cpu` with `replicate_index`. Per-axis mapping for `i = j - before`: `i < 0 → in[0]`;
`0 ≤ i < n → in[i]`; `i ≥ n → in[n-1]` — the boundary element is clamped/repeated outward.
Dtype-agnostic byte copy. Numerics: none (byte copy, bit-exact). Perf: bandwidth-bound, one linear
pass over the output. No `before ≤ n-1` restriction (clamping is well-defined for any pad width).
Contiguous + offset-0 only.

```fkc
kernel: pad_replicate_cpu
op_kind: Pad                        # mode_tag = 2 distinguishes Replicate
blurb: "Multi-dim replicate pad (edge repeated); delegates to the generic pad walker with the replicate index map."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_replicate_cpu"   # byte_kernels.rs:595 → pad_walk_cpu:631
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]
      dtype_class: any
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "== 2  (Replicate)" }
      fill_bytes: { kind: "Vec<u8>", constraint: "ignored for Replicate (no constant fill)" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * out_elem * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "out_elem * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via replicate (edge-clamp) index map; no arithmetic, bit-exact for every dtype."

determinism: bitwise
```

## pad_walk_cpu  (generic multi-dim pad walker — shared internal core)

One-line: generic dtype-agnostic pad walker; copies each output element from its mapped input position via a per-axis index function.

The shared generic walker that backs both Reflect and Replicate (Constant has its own two-pass
shape because padded slots do not index the input). Walks every one of the `out_elem` output
positions; for each, computes the corresponding input flat offset by applying a per-axis
`map_index(i, n)` function to `i = out_idx[k] - padding[k].before` and summing `m * in_stride[k]`,
then copies `dtype_size` bytes from input to output. Dtype-agnostic. Numerics: none (byte copy).
Perf: bandwidth-bound, one linear pass over the output. Validates rank agreement and exact byte
lengths, returning `Result`.

**Registrability note (faithful to as-built):** `pad_walk_cpu` is an **internal helper**, not a
standalone dispatch entry point. It is generic over the index-map function pointer
`F: Fn(i64, usize) -> usize` (instantiated as `reflect_index` / `replicate_index`), so it is not
itself addressable by a single `(OpKind, dtypes, backend)` key — the registrable kernels are
`pad_reflect_cpu` / `pad_replicate_cpu`, which bind the map. This contract documents its behavior
for completeness (the inventory names it); the importer would register it via its two mode-bound
wrappers, not as a third independent binding. Its `op_kind`/key are therefore the same
`OpKind::Pad` family the wrappers expose, with `mode_tag` selecting the bound map.

```fkc
kernel: pad_walk_cpu
op_kind: Pad                        # internal core for Reflect (mode_tag=1) / Replicate (mode_tag=2)
blurb: "Generic dtype-agnostic pad walker; copies each output element from its mapped input position via a per-axis index function."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_walk_cpu"   # byte_kernels.rs:631 (internal; bound by mode wrappers)
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64]
      dtype_class: any
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=out"
  op_params:
    variant: Pad
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in {1 (Reflect), 2 (Replicate)} — selects the bound map_index" }
      fill_bytes: { kind: "Vec<u8>", constraint: "ignored by the walker (Reflect/Replicate never fill)" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "0"
  bytes_moved: "2 * out_elem * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "out_elem * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy via a per-axis index map (reflect or replicate); no arithmetic, bit-exact. Internal helper bound by the mode wrappers, not an independent dispatch key."

determinism: bitwise
```

## pad_backward_f32  (Pad backward — f32)

One-line: Pad gradient on f32; accumulates grad_out into grad_in over the forward index mapping, f64 scratch accumulator.

Backward of `Pad` on `f32` (`OpKind::PadBackward`). Walks every one of the `out_elem` output
positions and, per `mode_tag`, maps each to an input position (Constant: **skip** positions that
fall in the padded region — they did not come from the input; Reflect: `reflect_index`; Replicate:
`replicate_index`), accumulating `grad_out[out_flat]` into an internal **f64** accumulator slot for
that input element. Many output positions can map to the same input slot (e.g. all corners of a
Replicate-padded image fold onto one input corner), so the f64 accumulator is the load-bearing
precision invariant — it avoids loss when summing many contributions. After the walk, each input
slot is **narrowed** from the f64 accumulator to `f32` and written to `grad_in`. Output behavior:
`grad_in` is **fully written** (every slot is set from its accumulator, which started at 0.0) — it
is *not* read-modified-written against prior `grad_in` contents, so `aliasing: none` (the
accumulation is internal to the kernel's scratch, not into the output buffer). Perf: bandwidth-bound
over `out_elem` reads + a final `in_elem` narrow/store, plus an `in_elem`-sized f64 scratch alloc.
Validates rank agreement and element counts, returning `Result`.

```fkc
kernel: pad_backward_f32
op_kind: PadBackward
blurb: "Pad gradient on f32; accumulates grad_out into grad_in over the forward index mapping, f64 scratch accumulator."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_backward_f32"   # byte_kernels.rs:787 (pad_backward_kernel!:702)
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=grad_in; element_count == out_elem"
  op_params:
    variant: PadBackward            # OpParams::PadBackward (primitive namespace)
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in {0 Constant, 1 Reflect, 2 Replicate}" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)       # F32 in → F32 out
      shape_rule: from_params(in_shape)        # the (smaller) input shape
      layout_guarantee: contiguous
      aliasing: none                           # fully written from internal f64 acc (not RMW of grad_in)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32                  # f32 element granular

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "out_elem"                            # one f64 add per non-skipped output element
  bytes_moved: "(out_elem + in_elem) * 4"      # read grad_out (F32) + write grad_in (F32); + in_elem*8 f64 scratch
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "in_elem * 8", disk_bytes: 0 }   # f64 accumulator scratch

precision:
  bit_stable_on_same_hardware: true            # fixed walk order, deterministic f64 accumulation
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f64 accumulator scratch; deterministic accumulation order; narrow to f32 on store."

determinism: same_hardware_bitwise
```

## pad_backward_f64  (Pad backward — f64)

One-line: Pad gradient on f64; accumulates grad_out into grad_in over the forward index mapping (native f64 accumulator).

Backward of `Pad` on `f64`. Identical structure to `pad_backward_f32` but the accumulator is
native `f64` (the widen/narrow are identity), so it is lossless end-to-end. Walks `out_elem` output
positions, maps each to its input position per `mode_tag` (Constant skips padded slots), accumulates
into an `in_elem`-sized f64 scratch, then writes each input slot. `grad_in` is fully written;
`aliasing: none`. Bandwidth-bound. Returns `Result`.

```fkc
kernel: pad_backward_f64
op_kind: PadBackward
blurb: "Pad gradient on f64; accumulates grad_out into grad_in over the forward index mapping (native f64 accumulator)."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_backward_f64"   # byte_kernels.rs:788
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=grad_in; element_count == out_elem"
  op_params:
    variant: PadBackward
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in {0 Constant, 1 Reflect, 2 Replicate}" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)        # F64 in → F64 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64                  # f64 element granular

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "out_elem"
  bytes_moved: "(out_elem + in_elem) * 8"      # read grad_out (F64) + write grad_in (F64); native f64 scratch
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "in_elem * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true            # native f64; lossless; deterministic walk order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Native f64 accumulator (lossless); deterministic accumulation order."

determinism: same_hardware_bitwise
```

## pad_backward_bf16  (Pad backward — bf16)

One-line: Pad gradient on bf16; accumulates in f64 scratch (widened from bf16), narrows to bf16 on store.

Backward of `Pad` on `bf16`. Same walk as the other backward kernels, but each `grad_out[out_flat]`
is widened `bf16 → f64` before accumulation, the per-input-slot sum is held in an `in_elem`-sized
**f64** scratch, and the final value is narrowed `f64 → f32 → bf16` on store. The f64 accumulator is
the precision invariant — when many output positions fold onto one input slot (Reflect/Replicate
overlaps) a half-precision accumulator would lose contributions, so accumulation is done wide.
`grad_in` is fully written; `aliasing: none`. Bandwidth-bound. Returns `Result`.

```fkc
kernel: pad_backward_bf16
op_kind: PadBackward
blurb: "Pad gradient on bf16; accumulates in f64 scratch (widened from bf16), narrows to bf16 on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_backward_bf16"   # byte_kernels.rs:789
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=grad_in; element_count == out_elem"
  op_params:
    variant: PadBackward
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in {0 Constant, 1 Reflect, 2 Replicate}" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)        # BF16 in → BF16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16                  # bf16 element granular

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "out_elem"                            # one widened f64 add per non-skipped output element
  bytes_moved: "(out_elem + in_elem) * 2"      # read grad_out (BF16) + write grad_in (BF16); + in_elem*8 f64 scratch
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "in_elem * 8", disk_bytes: 0 }   # f64 accumulator scratch

precision:
  bit_stable_on_same_hardware: true            # f64 accumulation; deterministic walk order; narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Accumulator widened to f64 (avoids loss when many outputs fold onto one input slot); narrow f64->f32->bf16 on store."

determinism: same_hardware_bitwise
```

## pad_backward_f16  (Pad backward — f16)

One-line: Pad gradient on f16; accumulates in f64 scratch (widened from f16), narrows to f16 on store.

Backward of `Pad` on `f16`. Identical to `pad_backward_bf16` with the IEEE half type: each
`grad_out` element is widened `f16 → f64`, summed in an `in_elem`-sized f64 scratch, and narrowed
`f64 → f32 → f16` on store. f64 accumulator is the precision invariant for the many-to-one fold.
`grad_in` is fully written; `aliasing: none`. Bandwidth-bound. Returns `Result`.

```fkc
kernel: pad_backward_f16
op_kind: PadBackward
blurb: "Pad gradient on f16; accumulates in f64 scratch (widened from f16), narrows to f16 on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::pad_backward_f16"   # byte_kernels.rs:790
kernel_revision_hash: auto

accept:
  inputs:
    - name: grad_out
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=grad_in; element_count == out_elem"
  op_params:
    variant: PadBackward
    fields:
      in_shape:  { kind: "Vec<usize>" }
      out_shape: { kind: "Vec<usize>", constraint: "out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1" }
      padding:   { kind: "Vec<(usize,usize)>", constraint: "len == in_shape.len()" }
      mode_tag:  { kind: u8, constraint: "in {0 Constant, 1 Reflect, 2 Replicate}" }

return:
  outputs:
    - name: grad_in
      dtype_rule: passthrough(grad_out)        # F16 in → F16 out
      shape_rule: from_params(in_shape)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16                  # f16 element granular

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "out_elem"
  bytes_moved: "(out_elem + in_elem) * 2"      # read grad_out (F16) + write grad_in (F16); + in_elem*8 f64 scratch
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "in_elem * 8", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Accumulator widened to f64 (avoids loss when many outputs fold onto one input slot); narrow f64->f32->f16 on store."

determinism: same_hardware_bitwise
```

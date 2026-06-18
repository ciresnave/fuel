---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                  # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"    # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — data-movement kernel contracts

Vulkan/Slang byte-width-keyed data-movement and concat kernels: the strided→contiguous
materializers (`strided_copy`, `strided_copy_signed_*`), the axis transforms (`flip_*`, `roll_*`),
the triangular masks (`triu_*`, `tril_*`), the in-place slab scatter (`write_slice_*`), and the
2-input concat (`concat_along_dim*`). Family: **shape-ops**. Kernel sources live in
`fuel-kernels-source/kernels/*.slang`, AOT-compiled to SPIR-V in `fuel-vulkan-kernels/spv/*.spv`
and registered in the `EMBEDDED` table (`fuel-vulkan-kernels/src/lib.rs:39`); the Rust dispatch
wrappers (param packing, layout gating, route picking) live in `fuel-vulkan-backend/src/lib.rs`.

Cross-cutting facts for this family (from the Vulkan inventory):

- **Byte-width-keyed, dtype-agnostic.** The `*_b1/_b2/_b4/_b8` kernels are NOT dtype-monomorphized
  — they move raw words keyed only by element size: **b1 = 1 byte** (`U8`, `I8`), **b2 = 2 bytes**
  (`F16`, `BF16`, `I16`), **b4 = 4 bytes** (`F32`, `I32`, `U32`), **b8 = 8 bytes** (`F64`, `I64`).
  Each named kernel gets its own contract section with its byte-width's dtype list; the dispatch
  key (§12.1) carries the actual element dtype, while the *implementation* is the shared byte mover.
  No arithmetic is performed (the triangular kernels write an all-zero bit pattern, valid for every
  Fuel dtype), so every one of these is **bit-exact and deterministic across any hardware**
  (`determinism: bitwise`, `precision.bit_stable_on_same_hardware: true`, no math).
- **Output contiguity is universal.** Every kernel writes its output via the linear dispatch index;
  none emits a strided/offset output. `layout_guarantee: contiguous` (or `same_as(dst)` for the
  in-place `write_slice_*`, whose dst is itself contiguous).
- **Two strided idioms, distinct contracts.** The "rank-4 (shape0..3, strides0..3) Params + `flags`
  contiguity-bit" elementwise idiom (`flip_*`, `roll_*`, per-operand `concat_along_dim*`) is
  **strided + broadcast capable but NOT non-zero-offset capable** — it walks **unsigned** per-input
  strides (`handles_strided`); upstream Contiguize realizes any non-zero offset. The
  `strided_copy*` materializers are the exception that take an explicit `src_offset`
  (`strided_copy` unsigned; `strided_copy_signed_*` **signed** strides + signed offset — the
  negative-stride path, where `reverse_strides` is first-class, §4.1.1).
- **Contiguous-only sub-families.** `triu_*`/`tril_*` (1:1 keep-or-zero) and `write_slice_*`
  (contiguous src into a contiguous dst slab) declare `requires_contiguous`; the planner inserts
  (and prices) an `Op::Contiguize` — itself an FKC kernel, here `strided_copy` (§4.3, §4.4).
- **In-place scatter.** `write_slice_*` partially overwrites — and **aliases** — `dst`
  (`caps.in_place: true`, `aliasing: in_place(dst)`); only the slab bytes are touched.

Cost provenance: every kernel below is marked `provenance: judge_measured` — the Judge bootstraps
the coefficients (FKC stays agnostic to how, §4.4). Where a real bandwidth/FLOPs hint is genuinely
derivable (these are memory-bound byte movers / linear scans, FLOPs ≈ 0), it is recorded in the
expression strings as the honest *shape* of the cost; the Judge refines the constants. Launch
overhead is the Vulkan command-buffer submit + descriptor-bind cost — left as `~` for the Judge to
measure (no fabricated nanoseconds).

---

## strided_copy  (strided/broadcast/offset → dense row-major materialize; the Vulkan Contiguize)

The 4-byte strided→contiguous gather-copy: permute, broadcast, slice, and concat-via-offset over
**unsigned** strides. This is the Vulkan `Op::Contiguize`.

`strided_copy` (`strided_copy.slang:26`) reads `out_size` elements from the source via a
`shape_strides` storage buffer (`shape[0..rank]` then `stride[0..rank]`), starting at an explicit
`src_offset`, and writes them contiguously beginning at `dst_offset`. A stride-0 axis transparently
**replicates** the source element (broadcast without a separate materialize); arbitrary non-negative
strides handle transpose/slice metadata-only views; a non-zero `src_offset` honors a view base. The
buffer is `f32`-typed but **byte-pattern-agnostic for any 4-byte dtype** (it copies the 32-bit word,
not an arithmetic value). It is the kernel the planner inserts (and prices) whenever a downstream
`requires_contiguous` Vulkan kernel is fed a non-contiguous 4-byte operand (§4.3); being an ordinary
FKC kernel makes the contiguize-vs-strided comparison a literal sum of two `CostEstimate`s (§4.4).
Numerics: pure word copy, no arithmetic — bit-exact, hardware-independent. Perf: bandwidth-bound,
one 4-byte word per output element (broadcast replays re-read the same source word). Limitation:
**unsigned strides only** — a negative-stride view goes to `strided_copy_signed_b4` instead.

```fkc
kernel: strided_copy
registrable: false                 # §3.10 describe-only: Contiguize has no real OpKind (it is the planner-inserted materialize/contiguize lowering, §4.3) and OpParams::Contiguize is not a real variant; documented + priced as the materialize kernel, not registered. op_kind/op_params below are forward-looking markers.
op_kind: Contiguize
blurb: "Materialize a dense row-major 4-byte buffer from a strided/broadcast/offset input via shape_strides + src_offset; word copy."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::strided_copy"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]          # 4-byte: byte-pattern-agnostic word copy (f32-typed buffer)
      # The strided handler itself: arbitrary unsigned strides, stride-0 broadcast, non-zero
      # src_offset all accepted. NEGATIVE strides are NOT walked — that is strided_copy_signed_b4.
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any                        # shape/strides arrive in the shape_strides storage buffer
  op_params:
    variant: Contiguize                # shape/strides/offsets via shape_strides buffer + scalar offsets
    fields:
      out_size:   { kind: usize }
      rank:       { kind: usize }
      src_offset: { kind: u32, note: "unsigned element offset of the iteration-first element" }
      dst_offset: { kind: u32, note: "contiguous write base" }
      shape_strides: { kind: "storage<u32>", note: "shape[0..rank] then stride[0..rank]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)         # dtype unchanged; only layout changes
      shape_rule: same_as(input)             # element shape preserved
      layout_guarantee: contiguous           # dense row-major from dst_offset (this kernel's job)
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # it IS the strided handler — walks strides directly, no fixup
  fast_paths:
    - { when: "all_inputs_contiguous", note: "dense source: linear copy, no replication" }
    - { when: "any_input_broadcast", note: "stride-0 axis re-reads the same source word" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; hint: bandwidth-bound, ~2*out_size*4 bytes (read each produced word + write)
  class: strided_elementwise
  flops: "0"                          # pure word copy, no arithmetic
  bytes_moved: "2 * out_size * 4"     # read each produced element + write it (broadcast re-reads inflate reads)
  overhead_ns: ~
  memory: { device_bytes: "out_size * 4", host_bytes: 0, disk_bytes: 0 }   # contiguous output alloc (executor pre-allocates)

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 4-byte word copy, no arithmetic; bit-exact and hardware-independent for every 4-byte dtype."

determinism: bitwise
```

---

## strided_copy_signed_b2  (negative-stride 2-byte view → dense materialize)

The 2-byte strided→contiguous materializer for **negative-stride** views (Flip/Roll/layout-on-Node).

`strided_copy_signed_b2` (shares `strided_copy_signed_b4.slang:26`; wrapper
`strided_copy_signed_bytes` `fuel-vulkan-backend/src/lib.rs:9366`) is the signed-stride sibling of
`strided_copy`: it reads strides from the `shape_strides` buffer **as `i32` via `asint`** and takes
a **signed `i32` `src_offset`**, so the iteration base may be the *last* physical element of an axis
and the walk proceeds toward lower addresses. This is exactly an `Op::Flip`/`Op::Roll` view
materialized into a dense 2-byte buffer. Reads as `uint` (no math); byte-pattern-agnostic for any
2-byte dtype. Because it walks negative strides, `reverse_strides: accepted` (§4.1.1) — and per the
layout-coherence rule a reversed stride is still a strided walk, so `strided: accepted` too.
Numerics: pure 16-bit word copy — bit-exact, hardware-independent. Perf: bandwidth-bound, one 2-byte
word per output element. (No b1 variant exists.)

```fkc
kernel: strided_copy_signed_b2
registrable: false                 # §3.10 describe-only: Contiguize has no real OpKind (planner-inserted materialize lowering, §4.3) and OpParams::Contiguize is not a real variant; documented, not registered. op_kind/op_params below are forward-looking markers.
op_kind: Contiguize
blurb: "Materialize a dense 2-byte buffer from a NEGATIVE-stride view (signed strides + signed src_offset); word copy."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::strided_copy_signed_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]         # 2-byte: byte-pattern-agnostic word copy
      # Signed strides (read i32 via asint) + signed src_offset: the negative-stride path.
      # reverse_strides ⇒ strided (a negative stride is still a strided walk; §10.4 coherence).
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      fdx: { symbolic_extent: rejected }   # concrete extents; describes (will describe) FDX signed strides — [cross-spec dep, §4.1.1]
  op_params:
    variant: Contiguize
    fields:
      out_size:   { kind: usize }
      rank:       { kind: usize }
      src_offset: { kind: i32, note: "SIGNED element offset; base may be the last element of an axis" }
      dst_offset: { kind: u32 }
      shape_strides: { kind: "storage<u32>", note: "shape[0..rank] then stride[0..rank], strides read as i32 via asint" }

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
    - { when: "any_input_reversed", note: "backward axis walk: same per-element cost as forward (no fixup)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*2 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 2"     # read each produced 2-byte element + write it
  overhead_ns: ~
  memory: { device_bytes: "out_size * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 2-byte word copy over signed strides; bit-exact, hardware-independent."

determinism: bitwise
```

---

## strided_copy_signed_b4  (negative-stride 4-byte view → dense materialize)

The 4-byte strided→contiguous materializer for **negative-stride** views (Flip/Roll/layout-on-Node).

`strided_copy_signed_b4` (`strided_copy_signed_b4.slang:26`; wrapper `strided_copy_signed_bytes`
`:9366`) is the canonical signed-stride Contiguize: strides read **as `i32` (`asint`)** and a
**signed `i32` `src_offset`**, so the iteration base may be the last physical element and the walk
proceeds toward lower addresses — a materialized `Op::Flip`/`Op::Roll` view for 4-byte data. Reads as
`uint`; byte-pattern-agnostic for any 4-byte dtype. `reverse_strides: accepted` (§4.1.1). Pure 32-bit
word copy — bit-exact, hardware-independent. Perf: bandwidth-bound, one 4-byte word per output
element. (No b1 variant exists.)

```fkc
kernel: strided_copy_signed_b4
registrable: false                 # §3.10 describe-only: Contiguize has no real OpKind (planner-inserted materialize lowering, §4.3) and OpParams::Contiguize is not a real variant; documented, not registered. op_kind/op_params below are forward-looking markers.
op_kind: Contiguize
blurb: "Materialize a dense 4-byte buffer from a NEGATIVE-stride view (signed strides + signed src_offset); word copy."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::strided_copy_signed_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]          # 4-byte: byte-pattern-agnostic word copy
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      fdx: { symbolic_extent: rejected }   # [cross-spec dep, §4.1.1]
  op_params:
    variant: Contiguize
    fields:
      out_size:   { kind: usize }
      rank:       { kind: usize }
      src_offset: { kind: i32, note: "SIGNED element offset; base may be the last element of an axis" }
      dst_offset: { kind: u32 }
      shape_strides: { kind: "storage<u32>", note: "shape[0..rank] then stride[0..rank], strides read as i32 via asint" }

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
    - { when: "any_input_reversed", note: "backward axis walk: same per-element cost as forward (no fixup)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*4 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 4"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 4-byte word copy over signed strides; bit-exact, hardware-independent."

determinism: bitwise
```

---

## strided_copy_signed_b8  (negative-stride 8-byte view → dense materialize)

The 8-byte strided→contiguous materializer for **negative-stride** views (Flip/Roll/layout-on-Node).

`strided_copy_signed_b8` (shares `strided_copy_signed_b4.slang:26`; wrapper
`strided_copy_signed_bytes` `:9366`) is the 8-byte signed-stride Contiguize: strides read **as `i32`
(`asint`)** and a **signed `i32` `src_offset`**, materializing an `Op::Flip`/`Op::Roll` view for
8-byte data (`F64`/`I64`). Byte-pattern-agnostic 64-bit word copy. `reverse_strides: accepted`
(§4.1.1). Bit-exact, hardware-independent. Perf: bandwidth-bound, one 8-byte word per output element.
(No b1 variant exists.)

```fkc
kernel: strided_copy_signed_b8
registrable: false                 # §3.10 describe-only: Contiguize has no real OpKind (planner-inserted materialize lowering, §4.3) and OpParams::Contiguize is not a real variant; documented, not registered. op_kind/op_params below are forward-looking markers.
op_kind: Contiguize
blurb: "Materialize a dense 8-byte buffer from a NEGATIVE-stride view (signed strides + signed src_offset); word copy."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::strided_copy_signed_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]               # 8-byte: byte-pattern-agnostic word copy
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: accepted }
      rank: any
      fdx: { symbolic_extent: rejected }   # [cross-spec dep, §4.1.1]
  op_params:
    variant: Contiguize
    fields:
      out_size:   { kind: usize }
      rank:       { kind: usize }
      src_offset: { kind: i32, note: "SIGNED element offset; base may be the last element of an axis" }
      dst_offset: { kind: u32 }
      shape_strides: { kind: "storage<u32>", note: "shape[0..rank] then stride[0..rank], strides read as i32 via asint" }

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
    - { when: "any_input_reversed", note: "backward axis walk: same per-element cost as forward (no fixup)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*8 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 8"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 8-byte word copy over signed strides; bit-exact, hardware-independent."

determinism: bitwise
```

---

## flip_b2  (reverse element order along one axis, 2-byte)

Reverse the order of elements along one axis of a 2-byte tensor; dtype-agnostic word reorder.

`flip_b2` (shares `flip_b4.slang:20`; wrapper `flip_bytes` `:9020`) reverses one axis of a tensor
viewed as a flat `outer × dim × inner` over a rank-4 (shape0..3, in_s0..3) Params block. One thread
per output element decomposes its linear out-index into rank-4 coords, reverses the coordinate on
`axis` (`src = dim-1-coord`), applies the **per-input strides**, and writes a 2-byte word
contiguously. It therefore **walks per-input strides** (`handles_strided`) — the source may be a lazy
view — while the output is contiguous over the input shape. Reads as a 16-bit word; no math.
Bit-exact, hardware-independent. Perf: bandwidth-bound, one 2-byte word per output element. (No b1
variant exists.)

```fkc
kernel: flip_b2
op_kind: Flip
blurb: "Reverse element order along one axis of a 2-byte tensor (src=dim-1-coord); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flip_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]         # 2-byte word reorder
      # Strided + broadcast capable (rank-4 shape + per-input strides; stride-0 ⇒ broadcast).
      # NOT non-zero-offset capable (offset realized by an upstream Contiguize). The reversal is
      # MATERIALIZED via coord arithmetic — it does NOT walk negative strides (that is strided_copy_signed).
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"                    # logical rank folded into rank-4 shape0..3
  op_params:
    variant: Flip                      # OpParams::Flip — rank-4 shape + strides + flipped axis
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks per-input strides directly; no contiguize for strided/broadcast input
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*2 bytes moved
  class: strided_elementwise
  flops: "0"                          # pure reorder, no arithmetic
  bytes_moved: "2 * out_size * 2"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 2-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## flip_b4  (reverse element order along one axis, 4-byte)

Reverse the order of elements along one axis of a 4-byte tensor; dtype-agnostic word reorder.

`flip_b4` (`flip_b4.slang:20`; wrapper `flip_bytes` `:9020`) reverses one axis of a flat
`outer × dim × inner` view over a rank-4 (shape0..3, in_s0..3) Params block: one thread per output
element unravels its linear out-index, reverses the `axis` coordinate (`src = dim-1-coord`), applies
per-input strides, and writes a 4-byte word contiguously. Walks per-input strides
(`handles_strided`); output contiguous over the input shape. Reads as a 32-bit word; no math.
Bit-exact, hardware-independent. Perf: bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: flip_b4
op_kind: Flip
blurb: "Reverse element order along one axis of a 4-byte tensor (src=dim-1-coord); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flip_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]          # 4-byte word reorder
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
  op_params:
    variant: Flip
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*4 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 4"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 4-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## flip_b8  (reverse element order along one axis, 8-byte)

Reverse the order of elements along one axis of an 8-byte tensor; dtype-agnostic word reorder.

`flip_b8` (shares `flip_b4.slang:20`; wrapper `flip_bytes` `:9020`) is the 8-byte sibling: reverse
the `axis` coordinate of a flat `outer × dim × inner` view over rank-4 Params, applying per-input
strides, writing an 8-byte word contiguously (`F64`/`I64`). Walks per-input strides
(`handles_strided`); output contiguous. No math; bit-exact, hardware-independent. Perf:
bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: flip_b8
op_kind: Flip
blurb: "Reverse element order along one axis of an 8-byte tensor (src=dim-1-coord); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flip_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]               # 8-byte word reorder
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
  op_params:
    variant: Flip
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*8 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 8"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 8-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## roll_b2  (cyclic shift along one axis, 2-byte)

Cyclically shift elements along one axis of a 2-byte tensor by a pre-normalized offset (always wraps).

`roll_b2` (shares `roll_b4.slang:25`; wrapper `roll_bytes` `:9111`) computes
`src_coord = (out_coord + offset) % shape[axis]` along the rolled axis, where `offset` is
**pre-normalized by the wrapper into `[0, shape[axis])`**, so the kernel itself does an unsigned add
+ modulo with no sign handling. Tensor viewed as a flat rank-4 (shape0..3, in_s0..3) decomposition;
one thread per output element applies **per-input strides** (`handles_strided`), writing a 2-byte
word contiguously. Reads as a 16-bit word; no math. Bit-exact, hardware-independent. Perf:
bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: roll_b2
op_kind: Roll
blurb: "Cyclic shift along one axis of a 2-byte tensor (src=(out+offset) mod dim); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::roll_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
  op_params:
    variant: Roll                      # OpParams::Roll — rank-4 shape + strides + axis + normalized offset
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      offset:   { kind: usize, note: "PRE-NORMALIZED by the wrapper into [0, shape[axis])" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*2 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 2"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 2-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## roll_b4  (cyclic shift along one axis, 4-byte)

Cyclically shift elements along one axis of a 4-byte tensor by a pre-normalized offset (always wraps).

`roll_b4` (`roll_b4.slang:25`; wrapper `roll_bytes` `:9111`) computes
`src_coord = (out_coord + offset) % shape[axis]` along the rolled axis (`offset` pre-normalized into
`[0, shape[axis])` by the wrapper) over a flat rank-4 (shape0..3, in_s0..3) view, one thread per
output element, applying per-input strides (`handles_strided`) and writing a 4-byte word contiguously.
No math; bit-exact, hardware-independent. Perf: bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: roll_b4
op_kind: Roll
blurb: "Cyclic shift along one axis of a 4-byte tensor (src=(out+offset) mod dim); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::roll_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
  op_params:
    variant: Roll
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      offset:   { kind: usize, note: "PRE-NORMALIZED by the wrapper into [0, shape[axis])" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*4 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 4"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 4-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## roll_b8  (cyclic shift along one axis, 8-byte)

Cyclically shift elements along one axis of an 8-byte tensor by a pre-normalized offset (always wraps).

`roll_b8` (shares `roll_b4.slang:25`; wrapper `roll_bytes` `:9111`) is the 8-byte sibling:
`src_coord = (out_coord + offset) % shape[axis]` (offset pre-normalized into `[0, shape[axis])`) over
a flat rank-4 view, applying per-input strides (`handles_strided`), writing an 8-byte word
contiguously (`F64`/`I64`). No math; bit-exact, hardware-independent. Perf: bandwidth-bound. (No b1
variant exists.)

```fkc
kernel: roll_b8
op_kind: Roll
blurb: "Cyclic shift along one axis of an 8-byte tensor (src=(out+offset) mod dim); strided input; dtype-agnostic word reorder."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::roll_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
  op_params:
    variant: Roll
    fields:
      out_size: { kind: usize }
      axis:     { kind: usize, constraint: "0 <= axis <= 3" }
      offset:   { kind: usize, note: "PRE-NORMALIZED by the wrapper into [0, shape[axis])" }
      shape0_3: { kind: "[usize; 4]" }
      in_s0_3:  { kind: "[usize; 4]", note: "per-input strides; 0 ⇒ broadcast axis" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*out_size*8 bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * out_size * 8"
  overhead_ns: ~
  memory: { device_bytes: "out_size * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 8-byte word permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## triu_b2  (upper-triangular mask, 2-byte)

Upper-triangular mask on the last two dims of a 2-byte tensor: keep `input[tid]` where
`j >= i + diagonal`, else write zero.

`triu_b2` (shares `triangular_b4.slang:44`; wrapper context `fuel-vulkan-backend/src/lib.rs`) is the
upper-triangular entry point of the shared triangular kernel: one thread per element, **contiguous
1:1** (`output[tid] = keep ? input[tid] : 0`), where the predicate over the last-two-dims coordinate
is `j >= i + diagonal`. A batch of `batch_count` `rows × cols` matrices. The zero write is an
all-zero **bit pattern**, the correct zero for every Fuel dtype — so the kernel is dtype-agnostic
keyed only on the 2-byte width. Contiguous-only: any non-contiguous input is realized by an upstream
`Op::Contiguize` (`strided_copy`), priced from its contract (§4.3, §4.4). No arithmetic; bit-exact,
hardware-independent. Perf: bandwidth-bound, output fully written once. (No b1 variant exists.)

```fkc
kernel: triu_b2
op_kind: Triu
blurb: "Upper-triangular mask (keep j>=i+diagonal, else zero) on a contiguous 2-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::triu_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]
      # Contiguous-only 1:1: output[tid] = keep ? input[tid] : 0. Non-contig input → upstream Contiguize.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                    # last two dims are rows×cols; leading dims fold into batch_count
  op_params:
    variant: Triangular                # OpParams::Triangular — keep_upper=true for triu
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j >= i + diagonal (triu)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (priced from strided_copy) for non-contig input
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*2) + kept-element reads
  class: strided_elementwise
  flops: "0"                          # comparison + select; no FP arithmetic
  bytes_moved: "2 * batch_count * rows * cols * 2"   # read input + write masked output
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## triu_b4  (upper-triangular mask, 4-byte)

Upper-triangular mask on the last two dims of a 4-byte tensor: keep `input[tid]` where
`j >= i + diagonal`, else write zero.

`triu_b4` (`triangular_b4.slang:44`) is the canonical upper-triangular entry point: contiguous 1:1
`output[tid] = keep ? input[tid] : 0` with predicate `j >= i + diagonal`, over `batch_count`
`rows × cols` matrices. All-zero bit-pattern fill (dtype-agnostic, 4-byte width). Contiguous-only.
No arithmetic; bit-exact, hardware-independent. Perf: bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: triu_b4
op_kind: Triu
blurb: "Upper-triangular mask (keep j>=i+diagonal, else zero) on a contiguous 4-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::triu_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
  op_params:
    variant: Triangular
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j >= i + diagonal (triu)" }

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
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*4)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * 4"
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## triu_b8  (upper-triangular mask, 8-byte)

Upper-triangular mask on the last two dims of an 8-byte tensor: keep `input[tid]` where
`j >= i + diagonal`, else write zero.

`triu_b8` (shares `triangular_b4.slang:44`) is the 8-byte sibling: contiguous 1:1
`output[tid] = keep ? input[tid] : 0` with predicate `j >= i + diagonal`, all-zero bit-pattern fill
(`F64`/`I64`). Contiguous-only. No arithmetic; bit-exact, hardware-independent. Perf:
bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: triu_b8
op_kind: Triu
blurb: "Upper-triangular mask (keep j>=i+diagonal, else zero) on a contiguous 8-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::triu_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
  op_params:
    variant: Triangular
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j >= i + diagonal (triu)" }

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
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*8)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * 8"
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## tril_b2  (lower-triangular mask, 2-byte)

Lower-triangular mask on the last two dims of a 2-byte tensor: keep `input[tid]` where
`j <= i + diagonal`, else write zero.

`tril_b2` (shares `triangular_b4.slang:48`) is the lower-triangular entry point of the shared
triangular kernel (the `tril_b4` companion to `triu_b4` in the same file): contiguous 1:1
`output[tid] = keep ? input[tid] : 0` with predicate `j <= i + diagonal`, over `batch_count`
`rows × cols` matrices. All-zero bit-pattern fill (dtype-agnostic, 2-byte width). Contiguous-only.
No arithmetic; bit-exact, hardware-independent. Perf: bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: tril_b2
op_kind: Tril
blurb: "Lower-triangular mask (keep j<=i+diagonal, else zero) on a contiguous 2-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tril_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16, BF16, I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
  op_params:
    variant: Triangular                # OpParams::Triangular — keep_upper=false for tril
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j <= i + diagonal (tril)" }

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
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*2)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * 2"
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## tril_b4  (lower-triangular mask, 4-byte)

Lower-triangular mask on the last two dims of a 4-byte tensor: keep `input[tid]` where
`j <= i + diagonal`, else write zero.

`tril_b4` (`triangular_b4.slang:48`) is the canonical lower-triangular entry point (shares the file
with `triu_b4` at line 44): contiguous 1:1 `output[tid] = keep ? input[tid] : 0` with predicate
`j <= i + diagonal`, over `batch_count` `rows × cols` matrices. All-zero bit-pattern fill
(dtype-agnostic, 4-byte width). Contiguous-only. No arithmetic; bit-exact, hardware-independent.
Perf: bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: tril_b4
op_kind: Tril
blurb: "Lower-triangular mask (keep j<=i+diagonal, else zero) on a contiguous 4-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tril_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
  op_params:
    variant: Triangular
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j <= i + diagonal (tril)" }

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
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*4)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * 4"
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## tril_b8  (lower-triangular mask, 8-byte)

Lower-triangular mask on the last two dims of an 8-byte tensor: keep `input[tid]` where
`j <= i + diagonal`, else write zero.

`tril_b8` (shares `triangular_b4.slang:48`) is the 8-byte sibling: contiguous 1:1
`output[tid] = keep ? input[tid] : 0` with predicate `j <= i + diagonal`, all-zero bit-pattern fill
(`F64`/`I64`). Contiguous-only. No arithmetic; bit-exact, hardware-independent. Perf:
bandwidth-bound. (No b1 variant exists.)

```fkc
kernel: tril_b8
op_kind: Tril
blurb: "Lower-triangular mask (keep j<=i+diagonal, else zero) on a contiguous 8-byte tensor; dtype-agnostic."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::tril_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
  op_params:
    variant: Triangular
    fields:
      batch_count: { kind: usize }
      rows:        { kind: usize }
      cols:        { kind: usize }
      diagonal:    { kind: i32, note: "keep boundary j <= i + diagonal (tril)" }

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
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*8)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * batch_count * rows * cols * 8"
  overhead_ns: ~
  memory: { device_bytes: "batch_count * rows * cols * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Keep-or-zero select with all-zero bit-pattern fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## write_slice_b1  (in-place rectangular slab write, 1-byte)

In-place rectangular scatter: copy a contiguous 1-byte `src` into a per-axis slab of a larger
contiguous `dst`; only the slab bytes are touched.

`write_slice_b1` (shares `write_slice_b4.slang:29`; wrapper `write_slice_bytes` `:2305`) writes a
contiguous `src` (its own rank-N shape) into a rectangular slab of a larger contiguous `dst` at
per-axis `range_start`, via a `shape_buf` = `src_shape + dst_shape + range_start`. Rank ≤ 8. The
1-byte variant has a **sub-u32 alignment precondition**: `range_start` and `src_shape` on the
innermost axis must be **multiples of 4** so writes land on 32-bit word boundaries; otherwise the
wrapper **bails to the CPU path** (`fuel-vulkan-backend/src/lib.rs:2359`). This is the
`Op::WriteSlice` KV-cache write. **In-place, aliases `dst`** (`caps.in_place: true`,
`aliasing: in_place(dst)`); partial overwrite — bytes outside the slab are preserved. Dtype-agnostic
byte copy; bit-exact, hardware-independent. Perf: bandwidth-bound in the slab volume.

```fkc
kernel: write_slice_b1
op_kind: WriteSlice
blurb: "In-place 1-byte rectangular slab write of src into dst at per-axis range_start (range_start & src_shape mult-of-4, else CPU fallback); aliases dst."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice_b1"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8, I8]                 # 1-byte
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == slab size on axis i; innermost range_start & src_shape multiples of 4 (else CPU fallback)"
    - name: dst
      dtypes: [U8, I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_as=out; read-modify-written in place (the output); larger than src on the sliced axes"
  op_params:
    variant: WriteSlice                # OpParams::WriteSlice — shape_buf = src_shape + dst_shape + range_start
    fields:
      n_src:    { kind: usize, note: "src element count (slab volume)" }
      rank:     { kind: usize, constraint: "1 <= rank <= 8" }
      shape_buf: { kind: "storage<u32>", note: "src_shape[0..rank] + dst_shape[0..rank] + range_start[0..rank]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: same_as(dst)         # dst's contiguous layout preserved; only slab bytes change
      aliasing: in_place(dst)                # output IS dst's buffer; partial (slab-only) overwrite

caps:
  awkward_layout_strategy: requires_contiguous   # both src and dst must be contiguous; non-contig → upstream Contiguize
  fast_paths:
    - { note: "innermost range_start & src_shape mult-of-4: GPU path; otherwise wrapper bails to CPU" }
  in_place: true                      # writes into dst's buffer (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; ~2*n_src bytes (read src + write slab)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * n_src * 1"        # read src + write slab; dst bytes outside the slab untouched
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place; no new allocation

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 1-byte copy into a dst slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## write_slice_b2  (in-place rectangular slab write, 2-byte)

In-place rectangular scatter: copy a contiguous 2-byte `src` into a per-axis slab of a larger
contiguous `dst`; only the slab bytes are touched.

`write_slice_b2` (shares `write_slice_b4.slang:29`; wrapper `write_slice_bytes` `:2305`) writes a
contiguous `src` into a rectangular slab of a larger contiguous `dst` at per-axis `range_start`
(`shape_buf` = `src_shape + dst_shape + range_start`), rank ≤ 8. The 2-byte variant requires the
**innermost `range_start` and `src_shape` to be even** (so writes land on 32-bit word boundaries);
otherwise the wrapper **bails to the CPU path** (`:2359`). `Op::WriteSlice` KV-cache write. In-place,
aliases `dst`; partial overwrite. Dtype-agnostic byte copy; bit-exact, hardware-independent. Perf:
bandwidth-bound in the slab volume.

```fkc
kernel: write_slice_b2
op_kind: WriteSlice
blurb: "In-place 2-byte rectangular slab write of src into dst at per-axis range_start (innermost range_start & src_shape even, else CPU fallback); aliases dst."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice_b2"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F16, BF16, I16]         # 2-byte
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == slab size on axis i; innermost range_start & src_shape even (else CPU fallback)"
    - name: dst
      dtypes: [F16, BF16, I16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_as=out; read-modify-written in place (the output); larger than src on the sliced axes"
  op_params:
    variant: WriteSlice
    fields:
      n_src:    { kind: usize, note: "src element count (slab volume)" }
      rank:     { kind: usize, constraint: "1 <= rank <= 8" }
      shape_buf: { kind: "storage<u32>", note: "src_shape[0..rank] + dst_shape[0..rank] + range_start[0..rank]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: same_as(dst)
      aliasing: in_place(dst)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { note: "innermost range_start & src_shape even: GPU path; otherwise wrapper bails to CPU" }
  in_place: true
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; ~2*n_src*2 bytes
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * n_src * 2"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 2-byte copy into a dst slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## write_slice_b4  (in-place rectangular slab write, 4-byte)

In-place rectangular scatter: copy a contiguous 4-byte `src` into a per-axis slab of a larger
contiguous `dst`; only the slab bytes are touched.

`write_slice_b4` (`write_slice_b4.slang:29`; wrapper `write_slice_bytes` `:2305`) is the canonical
slab-write kernel: a contiguous `src` (rank-N shape) into a rectangular slab of a larger contiguous
`dst` at per-axis `range_start` (`shape_buf` = `src_shape + dst_shape + range_start`), rank ≤ 8. The
4-byte width is naturally 32-bit word-aligned, so no sub-u32 alignment precondition applies. This is
the `Op::WriteSlice` KV-cache write. **In-place, aliases `dst`** (`caps.in_place: true`); partial
overwrite — bytes outside the slab preserved. Dtype-agnostic byte copy; bit-exact,
hardware-independent. Perf: bandwidth-bound in the slab volume.

```fkc
kernel: write_slice_b4
op_kind: WriteSlice
blurb: "In-place 4-byte rectangular slab write of src into dst at per-axis range_start; rank<=8; aliases dst."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice_b4"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, I32, U32]          # 4-byte
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == slab size on axis i"
    - name: dst
      dtypes: [F32, I32, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_as=out; read-modify-written in place (the output); larger than src on the sliced axes"
  op_params:
    variant: WriteSlice
    fields:
      n_src:    { kind: usize, note: "src element count (slab volume)" }
      rank:     { kind: usize, constraint: "1 <= rank <= 8" }
      shape_buf: { kind: "storage<u32>", note: "src_shape[0..rank] + dst_shape[0..rank] + range_start[0..rank]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: same_as(dst)
      aliasing: in_place(dst)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: true
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; ~2*n_src*4 bytes
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * n_src * 4"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 4-byte copy into a dst slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## write_slice_b8  (in-place rectangular slab write, 8-byte)

In-place rectangular scatter: copy a contiguous 8-byte `src` into a per-axis slab of a larger
contiguous `dst`; only the slab bytes are touched.

`write_slice_b8` (shares `write_slice_b4.slang:29`; wrapper `write_slice_bytes` `:2305`) is the
8-byte sibling: a contiguous `src` into a rectangular slab of a larger contiguous `dst` at per-axis
`range_start` (`shape_buf` = `src_shape + dst_shape + range_start`), rank ≤ 8, for `F64`/`I64`. The
`Op::WriteSlice` KV-cache write. In-place, aliases `dst`; partial overwrite. Dtype-agnostic byte
copy; bit-exact, hardware-independent. Perf: bandwidth-bound in the slab volume.

```fkc
kernel: write_slice_b8
op_kind: WriteSlice
blurb: "In-place 8-byte rectangular slab write of src into dst at per-axis range_start; rank<=8; aliases dst."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::write_slice_b8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F64, I64]               # 8-byte
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == slab size on axis i"
    - name: dst
      dtypes: [F64, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "same_as=out; read-modify-written in place (the output); larger than src on the sliced axes"
  op_params:
    variant: WriteSlice
    fields:
      n_src:    { kind: usize, note: "src element count (slab volume)" }
      rank:     { kind: usize, constraint: "1 <= rank <= 8" }
      shape_buf: { kind: "storage<u32>", note: "src_shape[0..rank] + dst_shape[0..rank] + range_start[0..rank]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: same_as(dst)
      aliasing: in_place(dst)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: true
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; ~2*n_src*8 bytes
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * n_src * 8"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure 8-byte copy into a dst slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## concat_along_dim  (2-input concat along an arbitrary dim, f32)

Single-dispatch concatenation of two f32 inputs along an arbitrary dimension into one contiguous
buffer.

`concat_along_dim` (`concat_along_dim.slang:40`; wrapper `concat_along_dim_f32_bytes` `:5159`) joins
two operands `a` and `b` along `concat_dim` in a single dispatch: one thread per output element
decides, from its `concat_dim` coordinate, whether it reads from `a` (coordinate `< a_dim`) or `b`
(`- a_dim`), applies the **per-operand rank-4 strides** (`a_s0..3` / `b_s0..3`), and writes f32
contiguously. Either side may be a **lazy strided/broadcast view** (`handles_strided`; stride-0 ⇒
broadcast). The output's `concat_dim` size is `a_dim + b_dim`. Pure data copy (no arithmetic);
bit-exact, hardware-independent. **Two-input** op (the dispatch key carries `a` and `b` as ordered
f32 operand slots). Perf: bandwidth-bound, output written once.

```fkc
kernel: concat_along_dim
op_kind: Concat
blurb: "2-input concat of f32 along an arbitrary dim (out_d[concat_dim]=a_dim+b_dim); per-operand strided/broadcast; single dispatch."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::concat_along_dim"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32]
      # Per-operand strided + broadcast (rank-4 a_s0..3); a lazy view is fine. NOT offset-capable.
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with b on every axis except concat_dim"
    - name: b
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with a on every axis except concat_dim"
  op_params:
    variant: Concat                    # OpParams::Concat — out dims + concat axis + per-side sizes + strides
    fields:
      out_d0_3:   { kind: "[usize; 4]", note: "out_d[concat_dim] = a_dim + b_dim" }
      concat_dim: { kind: usize, constraint: "0 <= concat_dim <= 3" }
      a_dim:      { kind: usize, note: "a's size along concat_dim" }
      b_dim:      { kind: usize, note: "b's size along concat_dim" }
      total:      { kind: usize, note: "output element count" }
      a_s0_3:     { kind: "[usize; 4]", note: "per-operand a strides; 0 ⇒ broadcast" }
      b_s0_3:     { kind: "[usize; 4]", note: "per-operand b strides; 0 ⇒ broadcast" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)             # a and b share dtype; output is that dtype
      shape_rule: "from_params(concat: out_d with out_d[concat_dim] = a_dim + b_dim)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks per-operand strides directly; no contiguize for strided/broadcast inputs
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*total*4 bytes (read both inputs once + write output)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * total * 4"        # each output element read once from a or b + written
  overhead_ns: ~
  memory: { device_bytes: "total * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure f32 word copy (no arithmetic); bit-exact, hardware-independent."

determinism: bitwise
```

---

## concat_along_dim_f16  (2-input concat along an arbitrary dim, f16)

Single-dispatch concatenation of two f16 (native `float16_t`) inputs along an arbitrary dimension.

`concat_along_dim_f16` (`concat_along_dim.slang:40`; typed wrapper
`concat_along_dim_typed_bytes_with_bind` `:7489`) is the native-f16 instantiation of the concat
shader: one thread per output element selects `a` vs `b` from the `concat_dim` coordinate, applies
per-operand rank-4 strides, and writes f16 contiguously. Either side may be a lazy strided/broadcast
view (`handles_strided`). Pure data copy (no arithmetic), so bit-exact and hardware-independent
despite the half dtype (no f32 round-trip occurs — the f16 value is moved verbatim). Two-input op.
Perf: bandwidth-bound.

```fkc
kernel: concat_along_dim_f16
op_kind: Concat
blurb: "2-input concat of f16 along an arbitrary dim (out_d[concat_dim]=a_dim+b_dim); per-operand strided/broadcast; single dispatch."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::concat_along_dim_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with b on every axis except concat_dim"
    - name: b
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with a on every axis except concat_dim"
  op_params:
    variant: Concat
    fields:
      out_d0_3:   { kind: "[usize; 4]", note: "out_d[concat_dim] = a_dim + b_dim" }
      concat_dim: { kind: usize, constraint: "0 <= concat_dim <= 3" }
      a_dim:      { kind: usize }
      b_dim:      { kind: usize }
      total:      { kind: usize }
      a_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }
      b_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: "from_params(concat: out_d with out_d[concat_dim] = a_dim + b_dim)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*total*2 bytes
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * total * 2"
  overhead_ns: ~
  memory: { device_bytes: "total * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure f16 value copy (no arithmetic, no f32 round-trip); bit-exact, hardware-independent."

determinism: bitwise
```

---

## concat_along_dim_bf16  (2-input concat along an arbitrary dim, bf16)

Single-dispatch concatenation of two bf16 inputs (packed u16-in-u32) along an arbitrary dimension.

`concat_along_dim_bf16` (`concat_along_dim.slang:40`; typed wrapper
`concat_along_dim_typed_bytes_with_bind` `:7489`) is the bf16 instantiation. bf16 is stored
**packed two-per-u32**, so the kernel is **single-thread-per-bf16** to correctly place each half-word
across the `(a, b)` concat boundary, writing half-word stores via `InterlockedOr`; the **wrapper
zero-fills the output first** so the OR-merge is correct. Either side may be a lazy strided/broadcast
view (`handles_strided`). Despite the packing the data is moved verbatim (no f32 round-trip), so the
copy is bit-exact and hardware-independent. Two-input op. Perf: bandwidth-bound (the per-bf16
threading + zero-fill are constant-factor overhead the Judge measures).

```fkc
kernel: concat_along_dim_bf16
op_kind: Concat
blurb: "2-input concat of bf16 (packed-u32, InterlockedOr half-word writes; output pre-zeroed) along an arbitrary dim; per-operand strided/broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::concat_along_dim_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [BF16]                   # packed two-per-u32
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with b on every axis except concat_dim"
    - name: b
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with a on every axis except concat_dim"
  op_params:
    variant: Concat
    fields:
      out_d0_3:   { kind: "[usize; 4]", note: "out_d[concat_dim] = a_dim + b_dim" }
      concat_dim: { kind: usize, constraint: "0 <= concat_dim <= 3" }
      a_dim:      { kind: usize }
      b_dim:      { kind: usize }
      total:      { kind: usize }
      a_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }
      b_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: "from_params(concat: out_d with out_d[concat_dim] = a_dim + b_dim)"
      layout_guarantee: contiguous           # output pre-zeroed by wrapper, then InterlockedOr-merged half-words
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*total*2 bytes + a zero-fill pass; per-bf16 threading is constant-factor
  class: strided_elementwise
  flops: "0"
  bytes_moved: "3 * total * 2"        # zero-fill output + read input + InterlockedOr write (extra pass for packed half-words)
  overhead_ns: ~
  memory: { device_bytes: "total * 2", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure bf16 value copy (no arithmetic, no f32 round-trip); output pre-zeroed; bit-exact, hardware-independent."

determinism: bitwise
```

---

## concat_along_dim_f64  (2-input concat along an arbitrary dim, f64)

Single-dispatch concatenation of two f64 (native double) inputs along an arbitrary dimension.

`concat_along_dim_f64` (`concat_along_dim.slang:40`; typed wrapper
`concat_along_dim_typed_bytes_with_bind` `:7489`) is the native-double instantiation: one thread per
output element selects `a` vs `b` from the `concat_dim` coordinate, applies per-operand rank-4
strides, and writes an 8-byte double contiguously. Either side may be a lazy strided/broadcast view
(`handles_strided`). Pure data copy (no arithmetic); bit-exact, hardware-independent. Two-input op.
Perf: bandwidth-bound.

```fkc
kernel: concat_along_dim_f64
op_kind: Concat
blurb: "2-input concat of f64 along an arbitrary dim (out_d[concat_dim]=a_dim+b_dim); per-operand strided/broadcast; single dispatch."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::concat_along_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with b on every axis except concat_dim"
    - name: b
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=4"
      shape_constraint: "agree with a on every axis except concat_dim"
  op_params:
    variant: Concat
    fields:
      out_d0_3:   { kind: "[usize; 4]", note: "out_d[concat_dim] = a_dim + b_dim" }
      concat_dim: { kind: usize, constraint: "0 <= concat_dim <= 3" }
      a_dim:      { kind: usize }
      b_dim:      { kind: usize }
      total:      { kind: usize }
      a_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }
      b_s0_3:     { kind: "[usize; 4]", note: "0 ⇒ broadcast" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: "from_params(concat: out_d with out_d[concat_dim] = a_dim + b_dim)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~2*total*8 bytes
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * total * 8"
  overhead_ns: ~
  memory: { device_bytes: "total * 8", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure f64 word copy (no arithmetic); bit-exact, hardware-independent."

determinism: bitwise
```

---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — shape-ops kernel contracts

Portable byte-shaped CPU shape/movement, triangular/mask, and prefix-sum kernels from
`fuel-cpu-backend/src/byte_kernels.rs` (the `CpuStorageBytes` production surface the
`fuel_dispatch::dispatch::cpu_wrappers` path extracts and calls). Family: **shape-ops**.

Cross-cutting facts for this family (from the inventory): with the single exception of
`contiguize_cpu` (the strided→contiguous materializer itself), every kernel here operates on flat
`CpuStorageBytes` slices via `bytes()` / `as_slice()` and validates *byte length* against explicit
`usize` shape parameters — none consult a `Layout`/strides/offset internally. The input-layout
contract for all of them (except `contiguize_cpu`) is therefore **contiguous, offset 0, row-major**;
the pipelined executor's auto-Contiguize pass realizes any strided/broadcast/offset input *before*
these kernels run, so each one declares `awkward_layout_strategy: requires_contiguous` and the
planner prices the inserted `Op::Contiguize` from the `contiguize` contract below (§4.3, §4.4).
The movement/mask/triangular kernels are **dtype-agnostic** (the caller passes `dtype_size` and the
kernel copies that many bytes per element); the cumsum kernels are per-dtype (typed add). Output is
caller-pre-allocated and fully overwritten, except the in-place scatter family
(`write_slice_cpu`, `write_slice_rotating_cpu`) which partially overwrite — and alias — a slab of
`dest`. Validation is `Result`-returning byte-length checks; no panics on the production path.

Cost provenance: every kernel below is marked `provenance: judge_measured` — the Judge bootstraps
the coefficients (FKC stays agnostic to how, §4.4). Where a real bandwidth/FLOPs hint is genuinely
derivable from the op (these are memory-bound byte copies / linear prefix scans), it is recorded in
the expression strings as the honest shape of the cost; the Judge refines the constants.

---

## contiguize  (strided/broadcast/offset → dense row-major materialize)

The only strided/broadcast/offset-capable CPU kernel — the strided→contiguous materializer itself.

`contiguize_cpu` (`byte_kernels.rs:1206`) is the one CPU kernel that consumes a `Layout` and walks
arbitrary strides: it iterates `layout.strided_index()` and copies `dtype_size` bytes per produced
element into a freshly-allocated contiguous output of `layout.shape().elem_count() * dtype_size`
bytes. A stride-0 axis transparently **replicates** the source element (broadcast without a separate
materialize), a non-zero `byte_offset`/view base is honored (the iteration starts at the view base),
and the strided walk handles transpose/slice metadata-only views. It is **dtype-agnostic** — only
`dtype_size` matters. This is the kernel the planner inserts (and prices) whenever a downstream
`requires_contiguous` kernel is fed a non-contiguous operand (§4.3); it is itself an ordinary FKC
kernel so the contiguize-vs-strided comparison is a literal sum of two `CostEstimate`s (§4.4). It
allocates and **returns a new `CpuStorageBytes`** (the only allocating kernel here; the rest fill a
caller-buffer). Numerics: pure byte copy, no arithmetic — bit-exact, deterministic across any
hardware. Overflow on `elem_count * dtype_size` or a layout pointing past the input bytes is a typed
`Result` error, never a panic. Perf: bandwidth-bound; one `dtype_size`-byte `copy_from_slice` per
output element (broadcast replays re-read the same source byte range).

```fkc
kernel: contiguize
registrable: false            # §3.10 describe-only: there is NO OpKind::Contiguize in the as-built
                             # enum / lower_op_kind table; Fuel inserts contiguize as an executor
                             # materialize pass, not via a dispatch OpKind. Never invent an OpKind.
op_kind: Contiguize          # descriptive token; no real dispatch OpKind to key against
blurb: "Materialize a dense row-major buffer from a strided/broadcast/offset input; dtype-agnostic byte copy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::contiguize_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3]
      # The ONLY kernel here that walks a Layout: arbitrary strides, stride-0 broadcast,
      # non-zero start offset are all accepted. Negative strides are NOT walked (no flip path).
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
  op_params: { variant: None }      # shape/stride/offset arrive via the Layout argument, not OpParams

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # dtype unchanged; only layout changes
      shape_rule: same_as(input)            # element shape preserved (= layout.shape())
      layout_guarantee: contiguous          # dense row-major, offset 0 (this is the kernel's job)
      aliasing: none                        # freshly allocated CpuStorageBytes (this kernel allocates)

caps:
  awkward_layout_strategy: handles_strided   # it IS the strided handler — walks strides directly, no fixup
  fast_paths:
    - { when: "all_inputs_contiguous", note: "dense source: contiguous span copy, no replication" }
    - { when: "any_input_broadcast", note: "stride-0 axis re-reads the same source element" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; hint: bandwidth-bound, ~2*N*dtype_bytes moved (read src + write dst)
  class: strided_elementwise
  flops: "0"                          # pure byte copy, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"  # read each produced element + write it (broadcast re-reads inflate reads)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }   # allocates the contiguous output

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy, no arithmetic; bit-exact and hardware-independent for every dtype."

determinism: bitwise
```

---

## flip  (reverse element order along one dim)

Reverse the order of elements along one dimension; dtype-agnostic byte reorder.

`flip_cpu` (`byte_kernels.rs:408`) computes `out[outer, j, inner] = in[outer, dim_size-1-j, inner]`
over a contiguous, zero-offset buffer factored into `(outer, dim_size, inner)` by the caller. The
flipped axis is `dim_size`; `inner * dtype_size` bytes per row are copied with one `copy_from_slice`,
reversed in the `dim_size` index. Dtype-agnostic (only `dtype_size` matters). It allocates a scratch
`Vec<u8>`, fills it, then copies into the caller-pre-allocated output (so it is *not* a true
allocating return like `contiguize`; the executor still owns the output buffer). Pure permutation of
bytes — bit-exact, deterministic across any hardware. Byte-length mismatch is a typed `Result` error.
Perf: bandwidth-bound, `outer * dim_size` row copies of `inner * dtype_size` bytes each.

```fkc
kernel: flip
op_kind: Flip
blurb: "Reverse element order along one dim (out[..,j,..]=in[..,dim-1-j,..]); dtype-agnostic byte reorder."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::flip_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # Byte-dtype-agnostic (only dtype_size matters), but the REGISTRATION covers
      # the 6 dtypes the executor actually materializes flips for — the fan builds
      # one `[T, T]` binding per dtype, ALL resolving `flip_cpu_wrapper`. Trimmed
      # from the kernel's full byte-agnostic set to match production truthfully
      # (byte-for-byte the deleted `table.register(Flip, &unary(t), …)` regs).
      dtypes: [F32, F64, BF16, F16, U32, U8]
      # Contiguous-only: the kernel walks flat bytes, axis factored as (outer,dim,inner) usize params.
      # It does NOT walk a Layout's negative strides — it MATERIALIZES the reversal into the output.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # logical rank folded into outer/dim_size/inner by the caller
  op_params:
    variant: Flip                     # OpParams::Flip — carries the flipped dim (lowered to outer/dim_size/inner)
    fields:
      outer:     { kind: usize }
      dim_size:  { kind: usize }
      inner:     { kind: usize }
      dtype_size: { kind: usize, note: "bytes per element; kernel is dtype-agnostic" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (priced from `contiguize`) for non-contig input
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, 2*N*dtype_bytes moved (read each row + write reversed)
  class: strided_elementwise
  flops: "0"                          # pure reorder, no arithmetic
  bytes_moved: "2 * outer * dim_size * inner * dtype_size"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "outer * dim_size * inner * dtype_size", disk_bytes: 0 }   # internal scratch Vec

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## roll  (cyclic shift along one dim)

Cyclically shift elements along one dimension by `shift` positions (always wraps).

`roll_cpu` (`byte_kernels.rs:444`) computes `out[outer, j, inner] = in[outer, (j-shift) mod dim_size,
inner]` with Python-style modulo (`rem_euclid`), so positive `shift` moves elements to higher indices
and negative the opposite; the shift is normalized into `[0, dim_size)` once. Contiguous, zero-offset
buffer factored into `(outer, dim_size, inner)`; `inner * dtype_size` bytes per row copied. `dim_size
== 0` is a no-op. Dtype-agnostic (byte reorder). Allocates a scratch `Vec<u8>`, fills it, copies into
the caller output. Pure permutation — bit-exact, deterministic across any hardware. Byte-length
mismatch is a typed `Result` error. Perf: bandwidth-bound, `outer * dim_size` row copies.

```fkc
kernel: roll
op_kind: Roll
blurb: "Cyclic shift along one dim (out[..,j,..]=in[..,(j-shift) mod dim,..]); always wraps; dtype-agnostic."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::roll_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # Byte-dtype-agnostic; REGISTRATION trimmed to the 6 production dtypes (fan →
      # one `[T, T]` binding per dtype, ALL → `roll_cpu_wrapper`). Matches the
      # deleted `table.register(Roll, &unary(t), …)` regs.
      dtypes: [F32, F64, BF16, F16, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: Roll                     # OpParams::Roll — flattened axis + signed shift
    fields:
      outer:     { kind: usize }
      dim_size:  { kind: usize }
      inner:     { kind: usize }
      shift:     { kind: i64, note: "signed; normalized into [0,dim_size) via Python-style modulo" }
      dtype_size: { kind: usize }

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
    - { when: "dim_size == 0", class: free, note: "empty axis: no-op early return" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, 2*N*dtype_bytes moved
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * outer * dim_size * inner * dtype_size"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "outer * dim_size * inner * dtype_size", disk_bytes: 0 }   # internal scratch Vec

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte permutation; bit-exact, hardware-independent."

determinism: bitwise
```

---

## concat  (join N inputs along one dim)

Concatenate N contiguous inputs along one dimension into a single contiguous output.

`concat_cpu` (`byte_kernels.rs:2711`; `concat_f32` shim at 2777 pins `dtype_size = size_of::<f32>()`)
joins `inputs.len()` operands along one axis whose per-input sizes are `input_dim_sizes[i]`; the
output axis size is their sum. Every operand and the output share the same `(outer_count, inner_count)`
factoring and `dtype_size`; the kernel copies, per input, each `(outer, dim_pos)` row of `inner_count
* dtype_size` bytes into its destination slot at the running `dim_offset`. Dtype-agnostic. Requires
≥1 input; rejects `dtype_size == 0`; validates each input's byte length and the output's against the
declared factoring. Pure byte copy — bit-exact, deterministic. **Variadic input count** (the dispatch
key's operand list is the N joined inputs, all same dtype). Perf: bandwidth-bound, total bytes moved
= the output's byte size (each input byte copied once).

```fkc
kernel: concat
op_kind: Concat
blurb: "Join N contiguous inputs along one dim into one contiguous buffer; dtype-agnostic byte copy."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::concat_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    # Variadic: N>=1 operands, all the same dtype; per-input axis size = input_dim_sizes[i].
    - name: inputs
      # Byte-dtype-agnostic; REGISTRATION trimmed to the 9 production dtypes. The
      # importer treats the variadic list as ONE representative input, so the fan
      # builds the `[T, T]` shorthand key per dtype (the lookup site collapses the
      # actual N+1 dtype list to it), ALL → `concat_cpu_wrapper`. Matches the
      # deleted `table.register(Concat, &unary(dt), …)` loop.
      dtypes: [F32, F64, BF16, F16, U32, U8, I16, I32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_rank=out; all inputs agree on every axis except the concatenated dim"
      variadic: true                  # the operand list is N inputs (>=1); each is a distinct key dtype slot
  op_params:
    variant: Concat                   # OpParams::Concat — flattened (outer,inner) + per-input dim sizes
    fields:
      outer_count:    { kind: usize }
      input_dim_sizes: { kind: "Vec<usize>", constraint: "len == inputs.len(); >=1 entry" }
      inner_count:    { kind: usize }
      dtype_size:     { kind: usize, constraint: "> 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(inputs)        # all inputs share one dtype; output is that dtype
      shape_rule: "from_params(concat: outer × sum(input_dim_sizes) × inner)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, 2*out_bytes moved (read all inputs + write output)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * outer_count * sum(input_dim_sizes) * inner_count * dtype_size"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # output is caller-preallocated; no internal scratch

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy; bit-exact, hardware-independent."

determinism: bitwise
```

---

## write_slice  (in-place rectangular scatter into a dest slab)

In-place rectangular scatter: copy `source` into a per-axis half-open slab of `dest`; only the slab
bytes are touched.

`write_slice_cpu` (`byte_kernels.rs:8300`) writes `source`'s bytes into a rectangular slab of `dest`
defined by `ranges[i] = (start, end)` (half-open) per axis; the slab's size along axis `i` is `end -
start`, and `source`'s element count must equal the slab's. It walks the slab's non-innermost
coordinates in row-major order, copying the innermost (last-axis) run of `slab_inner * dtype_size`
bytes per outer tuple with one `copy_from_slice` at the dest offset shifted by `ranges[i].0` on each
axis. **Partial overwrite that aliases `dest`** (bytes outside the slab are untouched), so the
return-contract aliasing is `in_place(dest)` and the op requires `caps.in_place: true`. Dtype-agnostic.
Validates `dtype_size > 0`, rank ≥ 1, `ranges.len() == rank`, each range `start ≤ end ≤ dest_shape[i]`,
and dest/source byte sizes; empty slab is a no-op. Pure byte copy — bit-exact, deterministic. Perf:
bandwidth-bound in the slab volume (`slab_elems * dtype_size` bytes written; dest is read-modify only
in that no other bytes change — no read of dest content occurs).

**Key modeling (re-author 2026-07-03).** `dest` is the OUTPUT slot, not a key input: the executor's
`Op::WriteSlice` arm adopts `dest`'s Storage Arc as the kernel's output and passes only `source` as a
kernel input, so `build_lookup_dtypes` canonicalizes the binding key to `[T_source, T_out]` = `[T, T]`
(`pipelined.rs`). This section therefore models `source` as the SINGLE input and `dest` as the `out`
output (`aliasing: in_place(dest)`) — exactly the in-place template — so the importer keys `[T, T]`
byte-for-byte the deleted `table.register(WriteSlice, &unary(t), …)` regs. (The earlier deferral
mis-assumed the importer would key `[source, dest, out]`; the aliased `dest` is the output, never a
second key operand.)

```fkc
kernel: write_slice
op_kind: WriteSlice
blurb: "In-place rectangular scatter of source into a per-axis half-open slab of dest; dtype-agnostic; dest is the in-place output slot."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::write_slice_cpu"   # base; §3.4 fans write_slice_cpu_{f32,f64,bf16,f16,u32,u8}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      # Byte-dtype-agnostic (only dtype_size matters); REGISTRATION trimmed to the
      # 6 production dtypes — the fan builds one `[T, T]` binding per dtype, ALL
      # resolving `write_slice_cpu_wrapper`. `dest` is NOT a key input: it is the
      # OUTPUT slot (the executor adopts its Arc in place), so a faithful contract
      # models `source` as the SINGLE input + `dest` as the `out` output, keying
      # `[T_source, T_out]` = `[T, T]` byte-for-byte the deleted hand-written regs.
      dtypes: [F32, F64, BF16, F16, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"                   # source rank == dest rank; slab dim i = ranges[i].end-ranges[i].start
      shape_constraint: "dim[i] == ranges[i].end - ranges[i].start"
  op_params:
    variant: WriteSlice               # OpParams::WriteSlice — dest_shape + per-axis ranges
    fields:
      dest_shape: { kind: "Vec<usize>", constraint: "rank >= 1" }
      ranges:     { kind: "Vec<(usize,usize)>", constraint: "len == rank; 0 <= start <= end <= dest_shape[i]" }
      dtype_size: { kind: usize, constraint: "> 0", note: "read from the dest output Storage's dtype tag at dispatch, not a serialized OpParams field" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)        # out IS dest's buffer; dest's dtype == source's (T in, T out)
      shape_rule: "from_params(write_slice: dest_shape)"   # out adopts dest's shape (== dest_shape)
      layout_guarantee: contiguous           # dest's contiguous layout preserved; only slab bytes change
      aliasing: in_place(dest)               # output IS dest's buffer (executor adopts dest's Arc); partial (slab-only) overwrite

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[0] == 0", class: free, note: "empty slab: no-op early return" }
    - { note: "rank==1: single contiguous span copy" }
  in_place: true                      # writes into dest's buffer (§4.6)
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; 2*slab_bytes moved (read source + write slab)
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * source.n * dtype_size"   # source.n = slab element count; dest bytes outside the slab untouched
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place; no allocation

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy into a dest slab; bit-exact, hardware-independent. Bytes outside the slab are preserved."

determinism: bitwise
```

---

## write_slice_rotating  (in-place ring-buffer scatter)

In-place ring-buffer scatter: write `source` into a slab of `dest` whose `axis` wraps modulo
`modulus`, with the dynamic start read from a `position` operand.

`write_slice_rotating_cpu` (`byte_kernels.rs:8453`) is the sliding-window-KV-cache write. The dynamic
write start on `axis` is `wrapped_start = position % modulus`, where `position` is the first `u32` of
the `position_bytes` operand read host-side (native-endian). With `slab_axis_len = ranges[axis].end -
ranges[axis].start`, it writes rows whose `axis` coord is `(wrapped_start + r) % modulus`; when
`wrapped_start + slab_axis_len > modulus` the write splits across the ring boundary into two
`write_slice_cpu`-style copies (a prefix to `[wrapped_start, modulus)` and a suffix to `[0,
second_len)`), gathering each source half via `extract_strided_chunk`. **v1 constraint (per
inventory): the rotating `axis` is the leading dim (`axis == 0`)** — the documented sliding-window
layout `[seq, n_kv_heads, head_dim]` puts `seq` at axis 0, which makes the source split a flat
prefix/suffix; the param is read generally but axis-0 is the supported/tested case. Dtype-agnostic.
Validates `dtype_size > 0`, rank ≥ 1, `ranges.len() == rank`, `axis < rank`, `0 < modulus ≤
dest_shape[axis]`, `position_bytes.len ≥ 4`, rotating-axis slab ≤ `modulus`, off-axis ranges within
bounds, and dest/source byte sizes. **In-place, aliases `dest`** (`caps.in_place: true`); partial
overwrite. The `position` is a **runtime dynamic scalar** (the write offset is data-determined per
token) read from its operand, not a compile-time param. Pure byte copy — bit-exact, deterministic.
Perf: bandwidth-bound in the slab volume (one or two slab copies).

**Key modeling (re-author 2026-07-03).** Like `write_slice`, `dest` is the OUTPUT slot (the executor
adopts its Arc in place). The `position` operand is a **non-key runtime input**: the executor's
`Op::WriteSliceRotating` arm reads it as a separate kernel input, but `build_lookup_dtypes`
canonicalizes the binding key to `[T_source, T_out]` = `[T, T]` (position is EXCLUDED, `pipelined.rs`).
So this section models `source` as the SINGLE key input and `dest` as the `out` output — position is
documented in the `op_params` note, NOT an `accept.inputs` key slot — so the importer keys `[T, T]`
byte-for-byte the deleted `table.register(WriteSliceRotating, &unary(t), …)` regs. (The earlier
deferral mis-assumed the importer would key `[source, U32, dest, out]`; neither the aliased `dest` nor
the non-key `position` is a key operand.)

```fkc
kernel: write_slice_rotating
op_kind: WriteSliceRotating
blurb: "In-place ring-buffer scatter: write source into a dest slab whose axis wraps mod modulus; dest is the in-place output slot; dynamic start from a non-key position operand."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::write_slice_rotating_cpu"   # base; §3.4 fans write_slice_rotating_cpu_{f32,f64,bf16,f16,u32,u8}; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: source
      # Byte-dtype-agnostic (only dtype_size matters); REGISTRATION trimmed to the
      # 6 production dtypes — the fan builds one `[T, T]` binding per dtype, ALL
      # resolving `write_slice_rotating_cpu_wrapper`. `dest` is the OUTPUT slot
      # (in-place adoption) and `position` is a NON-KEY runtime U32 operand (read
      # by the kernel, excluded from the key by build_lookup_dtypes), so a faithful
      # contract models `source` as the SINGLE key input, keying `[T_source, T_out]`
      # = `[T, T]` byte-for-byte the deleted hand-written regs.
      dtypes: [F32, F64, BF16, F16, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "dim[i] == ranges[i].end - ranges[i].start; dim[axis] <= modulus"
  op_params:
    variant: WriteSliceRotating       # OpParams::WriteSliceRotating — dest_shape + axis + modulus + ranges
    fields:
      dest_shape: { kind: "Vec<usize>", constraint: "rank >= 1" }
      axis:       { kind: usize, constraint: "< rank; v1 supports axis == 0 (leading dim)" }
      modulus:    { kind: usize, constraint: "0 < modulus <= dest_shape[axis]" }
      ranges:     { kind: "Vec<(usize,usize)>", constraint: "len == rank; rotating-axis slab <= modulus; off-axis end <= dest_shape[i]" }
      dtype_size: { kind: usize, constraint: "> 0", note: "read from the dest output Storage's dtype tag at dispatch, not a serialized OpParams field" }
      position:   { kind: DynScalar, note: "the dynamic write offset — read host-side from a SEPARATE runtime rank-1 U32 `position` operand (the executor passes it as a second kernel input; it is NOT part of the binding key, NOT an OpParams field). Data-determined per token." }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(source)        # out IS dest's buffer; dest's dtype == source's (T in, T out)
      shape_rule: "from_params(write_slice_rotating: dest_shape)"   # out adopts dest's shape (== dest_shape)
      layout_guarantee: contiguous           # dest's contiguous layout preserved; only ring-slab bytes change
      aliasing: in_place(dest)               # output IS dest's buffer (executor adopts dest's Arc); partial (ring-slab) overwrite

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[0] == 0", class: free, note: "empty slab: no-op early return" }
    - { note: "wrapped_start + slab_axis_len <= modulus: single (non-split) slab write" }
  in_place: true
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound in slab volume; up to 2 slab copies (ring split) + a small source-gather scratch
  class: strided_elementwise
  flops: "0"
  bytes_moved: "2 * source.n * dtype_size"   # source.n = slab element count; ring split adds a same-volume scratch gather
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "source.n * dtype_size", disk_bytes: 0 }   # extract_strided_chunk scratch on a boundary split

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure byte copy into a ring slab; bit-exact, hardware-independent. Bytes outside the written rows are preserved."

determinism: bitwise
```

---

## triangular  (Triu / Tril mask)

Upper/lower triangular mask: keep `x[..., i, j]` on one side of a `diagonal` offset, zero elsewhere.

`triangular_cpu` (`byte_kernels.rs:905`) zeros the output, then copies the kept positions from the
input. With `keep_upper`, it keeps `j >= i + diagonal` (Triu); otherwise `j <= i + diagonal` (Tril).
A batch of `batch_count` `rows × cols` matrices is processed; `dtype_size` bytes per kept element are
copied with `copy_from_slice`, dropped elements are left as the all-zero fill. **Dtype-agnostic** —
all-zero bytes are the correct zero for every IEEE-754 / integer dtype Fuel supports, so the kernel
zeros the whole output (`fill(0)`) and then overlays kept positions. Validates input/output byte
length against `batch_count * rows * cols * dtype_size`. Pure mask-and-copy — bit-exact,
deterministic. Perf: bandwidth-bound, output fully written once (`batch*rows*cols*dtype_size` bytes).

```fkc
kernel: triangular
registrable: false            # §3.10 describe-only: chassis umbrella backing Triu+Tril (keep_upper
                             # selects); no single OpKind to key against (Triu/Tril are the real
                             # OpKinds, dispatch.rs:278/280). `Triangular` is the OpParams variant
                             # (kernel.rs:512), NOT an OpKind. Not a typo for one variant → describe-only.
op_kind: Triangular          # descriptive umbrella token; the registrable arms are Triu / Tril
blurb: "Triu/Tril mask: keep one side of a diagonal offset (keep_upper flag), zero elsewhere; dtype-agnostic."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::triangular_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [U8, I8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                   # last two dims are rows×cols; leading dims fold into batch_count
  op_params:
    variant: Triangular               # OpParams::Triangular
    fields:
      batch_count: { kind: usize }
      rows:       { kind: usize }
      cols:       { kind: usize }
      diagonal:   { kind: i64, note: "diagonal offset; keep boundary is j vs i+diagonal" }
      keep_upper: { kind: bool, note: "true=Triu (j>=i+diag), false=Tril (j<=i+diag)" }
      dtype_size: { kind: usize }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, output fully written once (~N*dtype_size) + kept-element reads
  class: strided_elementwise
  flops: "0"                          # comparison + copy; no FP arithmetic
  bytes_moved: "2 * batch_count * rows * cols * dtype_size"   # zero-fill output + copy kept inputs
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Mask-and-copy with all-zero fill (valid zero for every Fuel dtype); bit-exact, hardware-independent."

determinism: bitwise
```

---

## masked_fill  (write a fill value where a U8 mask is set)

Where the mask is non-zero, write a fixed `fill` value; elsewhere copy the input.

`masked_fill_cpu` (`byte_kernels.rs:1148`) walks `count = input.len_bytes() / dtype_size` elements:
where `mask[i] != 0` it writes `fill_bytes` (one element's worth of dtype bytes), else it copies the
input element. The **mask is U8, one byte per element** (validated `mask.len == count`, i.e. element
count, not byte count); `fill_bytes.len()` must equal `dtype_size`; input and output byte lengths must
match. **Dtype-agnostic** for the data (the fill value arrives pre-encoded as `dtype_size` bytes).
Pure select-and-copy — bit-exact, deterministic. No broadcasting (mask, input, output all have the
same element count). Perf: bandwidth-bound, output fully written once.

```fkc
kernel: masked_fill
op_kind: MaskedFill
blurb: "Write a fixed fill value where a U8 mask is set, else copy input; dtype-agnostic data, U8 mask."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::masked_fill_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      # Byte-dtype-agnostic data; REGISTRATION trimmed to the 6 production dtypes.
      # `input` is the sole varying operand, so it drives the fan; `mask` stays the
      # fixed U8 slot and `out` is passthrough(input) — key `[T, U8, T]` per dtype,
      # ALL → `masked_fill_cpu_wrapper`. Matches the deleted
      # `table.register(MaskedFill, &masked_dtypes(t), …)` regs.
      dtypes: [F32, F64, BF16, F16, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
    - name: mask
      dtypes: [U8]                    # 1 byte per element; non-zero ⇒ write fill
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "element count == input element count (no broadcasting)"
  op_params:
    variant: MaskedFill               # OpParams::MaskedFill — pre-encoded fill bytes + dtype_size
    fields:
      fill_bytes: { kind: "Vec<u8>", constraint: "len == dtype_size; one element's worth of the data dtype" }
      dtype_size: { kind: usize, constraint: "> 0" }

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
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # hint: bandwidth-bound, ~ (input + mask read + output write) ≈ 2*N*dtype_size + N mask bytes
  class: cheap_elementwise
  flops: "0"                          # branch + copy; no arithmetic
  bytes_moved: "2 * n * dtype_size + n"     # read input + write output (n elems) + read n mask bytes
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Select-and-copy; the fill value is byte-identical to its input encoding; bit-exact, hardware-independent."

determinism: bitwise
```

---

## cumsum_f32  (running prefix sum along one dim, f32)

Running prefix sum along one dimension, computed in native f32.

`cumsum_f32` (`cumsum_kernel!` at `byte_kernels.rs:801`, instantiated 836) computes, for each
`(outer, inner)` lane, `out[..,j,..] = Σ_{t<=j} in[..,t,..]` along `dim_size` with a native **f32**
accumulator seeded at 0. Buffer factored as `(outer, dim_size, inner)`; element count validated
against `outer * dim_size * inner`. This is a typed (not byte-agnostic) kernel — `in`/`out` are
`&[f32]`. Sequential add along the scanned axis ⇒ a fixed, deterministic summation order, so it is
bit-stable on the same hardware (the f32 add order is fixed by the loop). Numerics: standard IEEE-754
f32 accumulation (rounding error accumulates with `dim_size`, the inherent prefix-sum behavior).
Perf: bandwidth-bound, `n` reads + `n` writes, one FLOP (add) per element.

```fkc
kernel: cumsum_f32
op_kind: CumSum
blurb: "Running prefix sum along one dim; native f32 accumulator; bit-stable on same hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cumsum_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # scanned axis folded into outer/dim_size/inner
  op_params:
    variant: CumSum                   # OpParams::CumSum — flattened (outer,dim_size,inner)
    fields:
      outer:    { kind: usize }
      dim_size: { kind: usize }
      inner:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # F32 in, F32 out
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # hint: bandwidth-bound; n = outer*dim_size*inner; ~1 add/elem
  class: reduction
  flops: "outer * dim_size * inner"   # one add per element along the scan
  bytes_moved: "2 * outer * dim_size * inner * 4"   # read input + write output, 4 bytes/f32
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed sequential f32 add order along the scan
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Native f32 accumulator, fixed sequential summation order; IEEE-754 rounding accumulates with dim_size (inherent prefix-sum)."

determinism: same_hardware_bitwise
```

---

## cumsum_f64  (running prefix sum along one dim, f64)

Running prefix sum along one dimension, computed in native f64.

`cumsum_f64` (`cumsum_kernel!` at `byte_kernels.rs:801`, instantiated 837) is the f64 instantiation of
the shared `cumsum_kernel!` macro: `out[..,j,..] = Σ_{t<=j} in[..,t,..]` with a native **f64**
accumulator seeded at 0, over a `(outer, dim_size, inner)` factoring. `in`/`out` are `&[f64]`; element
count validated against `outer * dim_size * inner`. Fixed sequential add order ⇒ bit-stable on the
same hardware. Standard IEEE-754 f64 accumulation. Perf: bandwidth-bound, `n` reads + `n` writes (8
bytes/element), one add per element.

```fkc
kernel: cumsum_f64
op_kind: CumSum
blurb: "Running prefix sum along one dim; native f64 accumulator; bit-stable on same hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cumsum_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: CumSum
    fields:
      outer:    { kind: usize }
      dim_size: { kind: usize }
      inner:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # F64 in, F64 out
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured          # hint: bandwidth-bound; n = outer*dim_size*inner; ~1 add/elem
  class: reduction
  flops: "outer * dim_size * inner"
  bytes_moved: "2 * outer * dim_size * inner * 8"   # 8 bytes/f64
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Native f64 accumulator, fixed sequential summation order; IEEE-754 rounding accumulates with dim_size."

determinism: same_hardware_bitwise
```

---

## cumsum_bf16  (running prefix sum along one dim, bf16 I/O with f32 accumulator)

Running prefix sum along one dim; bf16 input/output, **f32 accumulator** (load-bearing precision
invariant).

`cumsum_bf16` (`byte_kernels.rs:840`) scans `out[..,j,..] = Σ_{t<=j} in[..,t,..]` along `dim_size`
with an **f32 accumulator**: each step widens the bf16 input to f32, adds, and narrows the running
sum back to bf16 on store (`half::bf16::from_f32(acc)`). The f32-accumulator widening is the precision
invariant the chassis encodes — accumulating directly in bf16 over a long axis would lose precision.
I/O dtype is bf16; element count validated against `outer * dim_size * inner`. Fixed sequential add
order ⇒ bit-stable on the same hardware. Perf: bandwidth-bound (2 bytes/element I/O), one
widen-add-narrow per element.

```fkc
kernel: cumsum_bf16
op_kind: CumSum
blurb: "Running prefix sum along one dim; bf16 I/O with f32 accumulator (narrow on store); bit-stable on same hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cumsum_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: CumSum
    fields:
      outer:    { kind: usize }
      dim_size: { kind: usize }
      inner:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # BF16 in, BF16 out
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound; n = outer*dim_size*inner; widen-add-narrow per elem
  class: reduction
  flops: "outer * dim_size * inner"
  bytes_moved: "2 * outer * dim_size * inner * 2"   # 2 bytes/bf16
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed sequential f32 add order; narrow to bf16 on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulator (precision invariant): widen bf16→f32, add, narrow→bf16 on store; fixed summation order."

determinism: same_hardware_bitwise
```

---

## cumsum_f16  (running prefix sum along one dim, f16 I/O with f32 accumulator)

Running prefix sum along one dim; f16 input/output, **f32 accumulator** (load-bearing precision
invariant).

`cumsum_f16` (`byte_kernels.rs:868`) is the f16 sibling of `cumsum_bf16`: it scans `out[..,j,..] =
Σ_{t<=j} in[..,t,..]` along `dim_size` with an **f32 accumulator**, widening each f16 input to f32,
adding, and narrowing on store (`half::f16::from_f32(acc)`). The f32 accumulator is the precision
invariant. I/O dtype is f16; element count validated against `outer * dim_size * inner`. Fixed
sequential add order ⇒ bit-stable on the same hardware. Perf: bandwidth-bound (2 bytes/element I/O),
one widen-add-narrow per element.

```fkc
kernel: cumsum_f16
op_kind: CumSum
blurb: "Running prefix sum along one dim; f16 I/O with f32 accumulator (narrow on store); bit-stable on same hardware."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::cumsum_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
  op_params:
    variant: CumSum
    fields:
      outer:    { kind: usize }
      dim_size: { kind: usize }
      inner:    { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)        # F16 in, F16 out
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured          # hint: bandwidth-bound; n = outer*dim_size*inner; widen-add-narrow per elem
  class: reduction
  flops: "outer * dim_size * inner"
  bytes_moved: "2 * outer * dim_size * inner * 2"   # 2 bytes/f16
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed sequential f32 add order; narrow to f16 on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulator (precision invariant): widen f16→f32, add, narrow→f16 on store; fixed summation order."

determinism: same_hardware_bitwise
```

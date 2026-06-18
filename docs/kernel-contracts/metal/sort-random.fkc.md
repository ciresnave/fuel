---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                                     # maps to BackendId::Metal
  kernel_source: "metal-msl"                         # the BindingEntry.kernel_source tag (FuelNative-class, §4.11)
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS    # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                      # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — sort & random kernel contracts

The Metal per-row **argsort** family (a single-block bitonic argsort plus the MLX block / multi-block
merge-sort pipeline) and the **random-fill** family (uniform / normal). MSL sources live in
`fuel-metal-kernels/metal_src/{sort.metal, mlx_sort.metal, random.metal}`; the Rust dispatch
wrappers (`call_arg_sort`, `call_mlx_arg_sort`, `call_random_uniform`, `call_random_normal`) live in
`fuel-metal-kernels/src/kernels/{sort.rs, random.rs}`; `fuel-metal-backend/src/storage.rs` wires the
random calls (`rand_uniform()` `storage.rs:2087`, `rand_normal()` `storage.rs:2124`) and the argsort
is invoked from `fuel-core/src/sort.rs`.

**Family-wide facts (each section overrides where its inventory entry differs):**

- **[consumer-ahead] no dispatch `OpKind` exists for sort or random yet.** The as-built
  `fuel-core-types::dispatch::OpKind` enum (`dispatch.rs:52`) carries **no** `ArgSort` / `RandUniform`
  / `RandNormal` variant. These contracts therefore declare descriptive `op_kind` names
  (`ArgSort` / `RandUniform` / `RandNormal`) the way the CPU `shape-ops` bundle declares `Triangular`
  / `Contiguize` (also not enum variants) — they parse and document faithfully, but **registration is
  blocked until the matching `OpKind` lands**. This is the standing `MxNotYetRegistrable` discipline
  (§6 / §10.6): an importer reaching one of these before the enum variant exists returns the
  unresolved-op error rather than fabricating a key. The contracts are authored so they register
  unchanged once the variant is added.
- **Argsort returns a U32 index permutation, never a sorted-value copy.** Every kernel in the sort
  family writes a `U32` index buffer (the per-row argsort permutation); the key dtype is the *input*
  dtype and is dropped from the output (`dtype_rule: fixed(U32)`, the same index-output convention as
  `argmax_dim`, reference `reduce.fkc.md`). Padded lanes (the next-pow2 / block tail) sort to the end.
- **Contiguous rows are assumed; src is offset-capable, output is dense.** All sort kernels index
  `src` as `x + row*ncols` (or `stride_sorted_axis = 1`), so a row must be physically contiguous;
  `src` rides a `BufferOffset` (non-zero `start_offset` capable, inventory cross-cutting note). None of
  these kernels walks an arbitrary N-D strider or a negative stride — `strided: rejected`,
  `reverse_strides: rejected`, `awkward_layout_strategy: requires_contiguous` throughout. A
  transposed/strided/flipped key tensor is normalized by an upstream `Op::Contiguize`-class kernel
  whose own contract the planner sums (§4.3 / §4.4).
- **The MLX multi-block path is a pipeline of four monomorphized stages.** `call_mlx_arg_sort`
  (`sort.rs:266`) auto-selects single-block (`carg_block_sort`) vs the multi-block merge pipeline
  (`sort_mbsort` → `partition_mbsort` → `merge_mbsort`) by column count: `tn = 8`, `bn ∈ {128,256,512}`
  chosen by `ncols.div_ceil(tn)` (`≥257 & dtype ≤4 bytes → 512`, `≥129 → 256`, else `128`),
  `n_blocks = ncols.div_ceil(bn*tn)`; `n_blocks > 1` takes the multi-block path. The pipeline ping-pongs
  `dev_vals_{0,1}` / `dev_idxs_{0,1}` scratch + a `block_partitions` buffer and finishes with a
  `copy2d` of the winning index ping-pong buffer into `dst` (row strides `src_s = dst_s = ncols`).
  **`merge_mbsort` is hardwired to `merge_mbsort_float32_uint32`** regardless of key dtype (the merge
  walks the partition/index domain, `sort.rs:115`); `sort_mbsort` / `partition_mbsort` /
  `carg_block_sort` are monomorphized over the MLX key dtype string (`uint8/uint32/int64/float16/
  bfloat16/float32`, `sort.rs:43`).
- **Random kernels take NO input tensor.** They fill a fresh dense output buffer and read-modify-write
  a backend-internal atomic `seed[2]` buffer (thread 0 advances the seed; `MTLResourceUsage::Read |
  Write` on `seed`). The seed is **not** an FKC operand — it is backend-private state, not a graph
  tensor — so the random contracts declare zero `accept.inputs`. Output dtype = requested dtype,
  shape = requested shape, dense, fresh allocation (`new_buffer`, `storage.rs:2100/2137`); no aliasing
  on any graph tensor. RNG math is in f32 then narrowed on store for f16/bf16.
- **Cost is `judge_measured` for every kernel in this file.** These are Metal GPU dispatches whose
  launch overhead, threadgroup occupancy, scratch traffic (the multi-block path allocates four full
  value/index scratch buffers + a partitions buffer + a `copy2d`), and bandwidth are device-specific
  and not author-derivable — bitonic argsort's `O(ncols·log²ncols)` comparator count and the
  merge-sort's `O(ncols·log(nblocks))` merge passes give an asymptotic *shape* but no honest absolute
  coefficient, so only the genuinely structural facts (the per-stage element/byte traffic) are recorded
  as hints and every absolute number is left to the Judge. `provenance: judge_measured` is a
  first-class, visible marker (§4.4), not a hidden gap; the Judge bootstraps and refines these
  coefficients. The few genuinely bandwidth-bound moves (the random elementwise fill, the `copy2d`
  final write) carry an `N·dtype_bytes` byte-traffic hint as a structural fact of the loop.

---

## asort_asc  (single-block bitonic argsort, ascending → U32 index permutation)

Single-block bitonic per-row argsort, ascending; returns U32 index permutation; contiguous rows.

Per-row bitonic argsort returning a **U32 index permutation**, ascending. One threadgroup per row;
`call_arg_sort` (`sort.rs:7`) dispatches `nrows` groups of `ncols_pad` threads, `ncols_pad` = the
next power of two ≥ `ncols`. The bitonic network sorts indices `[0, ncols_pad)` keyed by
`src[row*ncols + idx]`; padded entries (idx ≥ ncols) compare as +infinity so they settle at the end
and the live prefix `[0, ncols)` of `dst` holds the valid permutation. `src` is indexed
`x + row*ncols` → **rows must be
physically contiguous**; `src` is `BufferOffset`-offset-capable. Threadgroup shared memory is
`max(ncols_pad*4, 16)` bytes. Keys f32/f16/u8/u32/i64/bf16; output indices always U32, dense. Not
bit-stable across hardware in general, but the comparison + index-swap network is integer-deterministic
on identical hardware; ties keep the bitonic network's settled order (a stable tie-break is not
guaranteed).

```fkc
kernel: asort_asc
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet (dispatch.rs:52); registers once it lands
blurb: "Single-block bitonic per-row argsort, ascending; returns U32 index permutation; contiguous rows."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::call_arg_sort::asort_asc"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, U8, U32, I64, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1..=8"                  # last dim is ncols (the sort axis); leading dims fold into nrows
      shape_constraint: "last_dim_eq=out"   # output index perm has the same row layout as the key rows
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort (no variant yet; mirrors call_arg_sort args)
    fields:
      nrows:     { kind: usize, note: "product of leading dims" }
      ncols:     { kind: usize, note: "sort-axis length (last dim)" }
      ncols_pad: { kind: usize, note: "next pow2 >= ncols; padded lanes sort to the end" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)        # index permutation, NOT passthrough (key dtype dropped)
      shape_rule: same_as(src)      # one index per key element; same row layout
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[-1] == 1", note: "trivial single-column rows" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32       # u32 index lanes

cost:
  provenance: judge_measured        # hint: bitonic network ~O(nrows * ncols_pad * log^2 ncols_pad) comparator+swap ops; shared mem ncols_pad*4 B
  class: reduction
  flops: ~                          # comparator network; no FP arithmetic, comparisons only — left to the Judge
  bytes_moved: ~                    # shared-mem-resident sort; off-chip traffic ~read nrows*ncols keys + write nrows*ncols u32 — Judge-measured
  overhead_ns: ~
  memory: { device_bytes: "nrows * ncols * 4", host_bytes: 0, disk_bytes: 0 }   # U32 output buffer

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer comparator/index-swap network; deterministic on identical hardware. Padded lanes (idx>=ncols) sort to the end; tie order follows the bitonic settle (no stable-sort guarantee)."

determinism: same_hardware_bitwise
```

---

## asort_desc  (single-block bitonic argsort, descending → U32 index permutation)

Single-block bitonic per-row argsort, descending; returns U32 index permutation; contiguous rows.

Descending counterpart of `asort_asc` — the identical single-block bitonic kernel with the comparator
sense reversed, dispatched through the same `call_arg_sort` (`sort.rs:7`) with the `asort_desc_<t>`
pipeline name. Per-row, one threadgroup of `ncols_pad` (next pow2) threads, `nrows` groups; `src`
indexed `x + row*ncols` (contiguous rows, offset-capable); padded entries settle at the end of the
descending order. Keys f32/f16/u8/u32/i64/bf16; output U32 index permutation, dense.

```fkc
kernel: asort_desc
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet (dispatch.rs:52); descending sibling
blurb: "Single-block bitonic per-row argsort, descending; returns U32 index permutation; contiguous rows."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::call_arg_sort::asort_desc"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, U8, U32, I64, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "last_dim_eq=out"
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort (descending flag distinguishes from asort_asc)
    fields:
      nrows:     { kind: usize }
      ncols:     { kind: usize }
      ncols_pad: { kind: usize, note: "next pow2 >= ncols; padded lanes sort to the end" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[-1] == 1", note: "trivial single-column rows" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: same bitonic network as asort_asc, reversed comparator; ~O(nrows * ncols_pad * log^2 ncols_pad)
  class: reduction
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: "nrows * ncols * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer comparator/index-swap network, descending sense; deterministic on identical hardware. Padded lanes sort to the end; tie order follows the bitonic settle."

determinism: same_hardware_bitwise
```

---

## carg_block_sort  (MLX single-block argsort → U32 index permutation)

MLX single-block per-row argsort; returns U32 index permutation; contiguous rows, written direct to dst.

The single-block leg of the MLX argsort pipeline (`block_sort`, `sort.rs:219`). When
`n_blocks == ncols.div_ceil(bn*tn) == 1` (each row fits one block), `call_mlx_arg_sort` (`sort.rs:266`)
dispatches `carg_block_sort_<dtype>_uint32_bn<bn>_tn<tn>` directly into `dst`: `nrows` threadgroups of
`bn` threads, `tn = 8` elements per thread, `stride_sorted_axis = 1` (contiguous rows), writing the
per-row U32 index permutation. `bn ∈ {128, 256, 512}` is selected by `ncols.div_ceil(tn)`
(`≥257 & key ≤4 bytes → 512`, `≥129 → 256`, else `128`). Keys uint8/uint32/int64/float16/bfloat16/
float32 (the MLX dtype set); output U32 indices, dense, written straight to `dst` (no scratch, no
`copy2d` — that is the multi-block path). `src` offset-capable.

```fkc
kernel: carg_block_sort
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet (dispatch.rs:52); MLX single-block leg
blurb: "MLX single-block per-row argsort; returns U32 index permutation; contiguous rows, written direct to dst."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::call_mlx_arg_sort::carg_block_sort"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8, U32, I64, F16, BF16, F32]   # the MLX dtype set (mlx_dtype_str, sort.rs:43)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1..=8"
      shape_constraint: "last_dim_eq=out"
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort; bn/tn are kernel-internal tile selection from ncols
    fields:
      nrows: { kind: usize }
      ncols: { kind: usize, note: "single-block when ncols.div_ceil(bn*tn) == 1" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[-1] == 1", note: "trivial single-column rows" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: in-threadgroup sort over bn*tn lanes; bn in {128,256,512} by ncols; no scratch, no copy2d
  class: reduction
  flops: ~
  bytes_moved: ~                    # read nrows*ncols keys + write nrows*ncols u32 direct to dst — Judge-measured
  overhead_ns: ~
  memory: { device_bytes: "nrows * ncols * 4", host_bytes: 0, disk_bytes: 0 }   # U32 output; single-block path allocates NO scratch

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "MLX in-threadgroup integer sort; deterministic on identical hardware. Selected only when each row fits one block (n_blocks==1)."

determinism: same_hardware_bitwise
```

---

## sort_mbsort  (MLX multi-block blockwise argsort stage → per-block sorted vals + U32 idxs)

MLX multi-block argsort stage 1: per-block sort each row slice into scratch (sorted vals + U32 idxs).

Stage 1 of the MLX multi-block merge-sort pipeline (`multi_block_sort`, `sort.rs:55`). Dispatched as
`sort_mbsort_<dtype>_uint32_bn<bn>_tn<tn>` over `nblocks × nrows` threadgroups of `bn` threads: each
block independently sorts its `bn*tn`-element slice of the row, emitting **block-local sorted values**
into `dev_vals_0` and **block-local U32 indices** into `dev_idxs_0` (both ping-pong scratch buffers
sized `nrows*ncols`). Reads `src` (offset-capable, `stride_sorted_axis = 1`, `nc_str = ncols`). This
stage is an internal pipeline step: its outputs are **scratch buffers**, not a graph tensor — the
public op is the whole pipeline (`merge_mbsort` produces the final result). It is contracted here for
visibility, with `aliasing: none` against any graph tensor and its scratch footprint surfaced in the
cost memory hint. Keys uint8/uint32/int64/float16/bfloat16/float32; idx U32.

```fkc
kernel: sort_mbsort
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet; internal multi-block stage-1 (not independently dispatched)
blurb: "MLX multi-block argsort stage 1: per-block sort each row slice into scratch (sorted vals + U32 idxs)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::multi_block_sort::sort_mbsort"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [U8, U32, I64, F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: "1..=8"
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort; bn/tn/nblocks are pipeline-internal
    fields:
      nrows:   { kind: usize }
      ncols:   { kind: usize }
      nblocks: { kind: usize, note: "ncols.div_ceil(bn*tn); > 1 selects this multi-block pipeline" }

return:
  outputs:
    - name: block_vals_idxs
      dtype_rule: fixed(U32)        # the consumable result of the pipeline is the U32 index domain; sorted vals are scratch in key dtype
      shape_rule: same_as(src)      # one entry per key element, per block (block-local order)
      layout_guarantee: contiguous  # ping-pong scratch buffers (dev_vals_0 / dev_idxs_0), row-major
      aliasing: none                # scratch only; no graph tensor aliased

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: nblocks*nrows in-block sorts over bn*tn lanes; allocates 2 val + 2 idx scratch buffers + block_partitions
  class: reduction
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  # scratch: dev_vals_{0,1} = 2*nrows*ncols*dtype_bytes ; dev_idxs_{0,1} = 2*nrows*ncols*4 ; block_partitions = nrows*(nblocks+1)*4
  memory: { device_bytes: "nrows * ncols * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Per-block integer sort into scratch; deterministic on identical hardware. Internal stage of the multi-block pipeline (sort_mbsort -> partition_mbsort -> merge_mbsort)."

determinism: same_hardware_bitwise
```

---

## partition_mbsort  (MLX multi-block merge partitioning stage → block partition offsets)

MLX multi-block argsort stage 2: compute per-row merge partition offsets into block_partitions scratch.

Stage 2 of the MLX multi-block pipeline (`multi_block_sort`, `sort.rs:128`). Dispatched as
`partition_mbsort_<dtype>_uint32_bn<bn>_tn<tn>` over `1 × nrows` threadgroups of
`min(nblocks+1, 1024)` threads per merge round: it computes the per-row `block_partitions` offsets
(`nrows*(nblocks+1)` U32 entries) that tell the subsequent `merge_mbsort` where each `merge_tiles`-wide
merge boundary falls. Reads the current ping-pong `dev_vals_in` / `dev_idxs_in`; writes
`block_partitions`. Runs once per merge round in the `while merge_tiles/2 < nblocks` loop, with
`merge_tiles` doubling each round. Internal pipeline stage — its output is the partitions scratch
buffer, not a graph tensor. Monomorphized over the MLX key dtype string; partition entries are U32.

```fkc
kernel: partition_mbsort
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet; internal multi-block stage-2 (merge partitioning)
blurb: "MLX multi-block argsort stage 2: compute per-row merge partition offsets into block_partitions scratch."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::multi_block_sort::partition_mbsort"
kernel_revision_hash: auto

accept:
  inputs:
    - name: block_vals
      dtypes: [U8, U32, I64, F16, BF16, F32]   # the ping-pong sorted-value scratch (dev_vals_in), key dtype
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
    - name: block_idxs
      dtypes: [U32]                # the ping-pong index scratch (dev_idxs_in)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort; merge_tiles/nblocks ride the pipeline loop
    fields:
      nrows:       { kind: usize }
      ncols:       { kind: usize }
      nblocks:     { kind: usize }
      merge_tiles: { kind: usize, note: "doubles each merge round; loop runs while merge_tiles/2 < nblocks" }

return:
  outputs:
    - name: block_partitions
      dtype_rule: fixed(U32)        # partition offsets, U32
      shape_rule: from_params(nrows, nblocks)   # [nrows, nblocks+1] partition table
      layout_guarantee: contiguous  # block_partitions scratch buffer, row-major
      aliasing: none                # scratch only

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: 1*nrows groups of min(nblocks+1,1024) threads per merge round; ~log2(nblocks) rounds
  class: reduction
  flops: ~
  bytes_moved: ~                    # writes nrows*(nblocks+1) u32 partition entries per round — Judge-measured
  overhead_ns: ~
  memory: { device_bytes: "nrows * (nblocks + 1) * 4", host_bytes: 0, disk_bytes: 0 }   # block_partitions buffer

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer partition-offset computation over already-sorted blocks; deterministic on identical hardware. Internal merge-partition stage of the multi-block pipeline."

determinism: same_hardware_bitwise
```

---

## merge_mbsort  (MLX multi-block merge stage → final per-row U32 index permutation)

MLX multi-block argsort stage 3: merge partitioned blocks into the final per-row U32 index permutation (merge tag hardwired float32).

Stage 3 of the MLX multi-block pipeline and the **producer of the final result** (`multi_block_sort`,
`sort.rs:157`). Dispatched as **`merge_mbsort_float32_uint32_bn<bn>_tn<tn>`** — note the **dtype tag
is hardwired to `float32`** regardless of the key dtype, because the merge walks the partition/index
domain rather than re-comparing keys directly (`merge_name`, `sort.rs:115`). Over `nblocks × nrows`
threadgroups of `bn` threads per merge round, it merges the partitioned `dev_vals_in` / `dev_idxs_in`
into `dev_vals_out` / `dev_idxs_out` using the `block_partitions` offsets, doubling `merge_tiles` each
round until the whole row is one sorted run. After the loop, `multi_block_sort` performs a `copy2d`
(`sort.rs:201`, dtype-selected by key) of the winning ping-pong index buffer into `dst` with row
strides `src_s = dst_s = ncols` — that final `copy2d` is a separate movement kernel with its own
contract (bandwidth-bound, `nrows*ncols*4` bytes). The merge's consumable output is the per-row U32
index permutation; keys uint8/uint32/int64/float16/bfloat16/float32 (selected upstream), idx U32.

```fkc
kernel: merge_mbsort
registrable: false               # §3.10 describe-only: NO OpKind::ArgSort / OpParams::ArgSort in fuel-dispatch
op_kind: ArgSort                  # [consumer-ahead] no ArgSort OpKind yet; multi-block pipeline final merge (result producer)
blurb: "MLX multi-block argsort stage 3: merge partitioned blocks into the final per-row U32 index permutation (merge tag hardwired float32)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::sort::multi_block_sort::merge_mbsort"
kernel_revision_hash: auto

accept:
  inputs:
    - name: block_vals
      dtypes: [U8, U32, I64, F16, BF16, F32]   # sorted-value scratch (dev_vals_in); merge tag is float32 but vals carry key dtype
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
    - name: block_idxs
      dtypes: [U32]                # index scratch (dev_idxs_in)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
    - name: block_partitions
      dtypes: [U32]                # partition offsets from partition_mbsort
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                       # [nrows, nblocks+1]
  op_params:
    variant: ArgSort               # [consumer-ahead] OpParams::ArgSort; merge_tiles/nblocks ride the pipeline loop
    fields:
      nrows:       { kind: usize }
      ncols:       { kind: usize }
      nblocks:     { kind: usize }
      merge_tiles: { kind: usize, note: "doubles each merge round until the row is one run" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)        # final index permutation; key dtype dropped
      shape_rule: from_params(nrows, ncols)   # [nrows, ncols] index perm; copied to dst by a downstream copy2d
      layout_guarantee: contiguous  # ping-pong out buffer, then copy2d to dst with row strides ncols
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: nblocks*nrows groups of bn threads per merge round; ~log2(nblocks) rounds; final result via a separate copy2d (nrows*ncols*4 B)
  class: reduction
  flops: ~
  bytes_moved: ~                    # per round reads/writes nrows*ncols (vals key dtype + idx u32) ping-pong; Judge-measured
  overhead_ns: ~
  memory: { device_bytes: "nrows * ncols * 4", host_bytes: 0, disk_bytes: 0 }   # final U32 index permutation (pre-copy2d)

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Integer merge over partitioned sorted blocks; deterministic on identical hardware. Pipeline tag is merge_mbsort_float32_uint32 regardless of key dtype (merges the index/partition domain). Final write to dst is a separate copy2d kernel."

determinism: same_hardware_bitwise
```

---

## rand_uniform  (fill a dense buffer with U(min, max))

Fill a dense buffer with U(min,max) via HybridTaus; no input tensor; advances a backend-internal seed.

Fill a fresh dense output buffer with uniform `U(min, max)` samples via HybridTaus (Tausworthe +
LCG), with a mirror-fill symmetry: each thread produces two elements (`tid` and `size-off-tid`),
dispatched over `length/2 + (length odd)` threads (`random.rs:7`). RNG math runs in **f32** then
narrows to the requested dtype on store. There is **no input tensor** — the kernel reads and advances
a backend-internal atomic `seed[2]` buffer (thread 0 advances the seed; `seed` is RW), and writes the
output. The backend (`storage.rs:2087`) allocates the output fresh (`new_buffer`) and errors when
`min >= max` (`random.rs:18`). dtypes f32/f16/bf16; output dtype = requested dtype, shape = requested
shape, dense, no aliasing on any graph tensor. The seed advance is a side effect on backend-private
state, not an FKC operand, so this contract has zero `accept.inputs`. Not bit-stable across hardware
(f32 RNG narrowing + atomic seed advance ordering); each draw depends on the seed value, so two runs
with the same seed reproduce on the same hardware.

```fkc
kernel: rand_uniform
registrable: false               # §3.10 describe-only: NO OpKind::RandUniform / OpParams::RandUniform in fuel-dispatch
op_kind: RandUniform              # [consumer-ahead] no RandUniform OpKind yet (dispatch.rs:52); registers once it lands
blurb: "Fill a dense buffer with U(min,max) via HybridTaus; no input tensor; advances a backend-internal seed."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::random::call_random_uniform::rand_uniform"
kernel_revision_hash: auto

accept:
  inputs: []                       # NO input tensor; seed is backend-private RW state, not a graph operand
  op_params:
    variant: RandUniform           # [consumer-ahead] OpParams::RandUniform (mirrors call_random_uniform args)
    fields:
      size: { kind: usize, note: "output element count; threads = size/2 + (size odd), 2 elems/thread (mirror fill)" }
      min:  { kind: f32, constraint: "min < max" }
      max:  { kind: f32, constraint: "min < max; backend errors otherwise (random.rs:18)" }

return:
  outputs:
    - name: out
      dtype_rule: cast(out)         # output dtype is the requested output Storage dtype (f32/f16/bf16); not derived from an input
      shape_rule: from_params(size) # the requested fill shape
      layout_guarantee: contiguous
      aliasing: none                # fresh new_buffer; the seed RW is backend-private, not a graph tensor alias

caps:
  awkward_layout_strategy: requires_contiguous   # writes a dense [0,size) buffer
  fast_paths: []
  in_place: false                  # fresh output; (the only RW buffer is the private seed, not an FKC operand)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: bandwidth-bound elementwise fill — touches N=size output elements, 2 per thread; RNG math is cheap fixed ALU
  class: cheap_elementwise
  flops: ~                          # HybridTaus integer mixes per element; not FP-flops-meaningful — Judge-measured
  bytes_moved: "size * dtype_bytes" # write-only fill (plus the tiny RW seed[2]); structural fact of the loop
  overhead_ns: ~
  memory: { device_bytes: "size * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "HybridTaus (Tausworthe+LCG) RNG, math in f32 then narrowed to f16/bf16. Distribution U(min,max). Reproducible per-seed on identical hardware; not bit-stable cross-hardware (f32 narrowing + atomic seed-advance ordering). No formal ULP/error bound on a stochastic generator (audited, no static bound applies)."

determinism: nondeterministic
```

---

## rand_normal  (fill a dense buffer with N(mean, stddev))

Fill a dense buffer with N(mean,stddev) via Box-Muller; no input tensor; advances a backend-internal seed.

Fill a fresh dense output buffer with normal `N(mean, stddev)` samples via Box-Muller (two values per
thread from a pair of uniforms), with the same mirror-fill symmetry and over `length/2 + (length odd)`
threads (`random.rs:41`). RNG math runs in **f32** then narrows on store. As with `rand_uniform`,
there is **no input tensor** — the kernel reads + advances a backend-internal atomic `seed[2]` buffer
(thread 0 advances; `seed` RW) and writes the output, which the backend (`storage.rs:2124`) allocates
fresh (`new_buffer`). dtypes f32/f16/bf16; output dtype = requested dtype, shape = requested shape,
dense, no aliasing on any graph tensor; zero `accept.inputs`. Box-Muller pairs are correlated within a
thread's two outputs by construction (`sqrt(-2 ln u1)·{cos,sin}(2π u2)`); reproducible per-seed on the
same hardware, not bit-stable cross-hardware.

```fkc
kernel: rand_normal
registrable: false                # §3.10 describe-only: NO OpKind::RandNormal / OpParams::RandNormal in fuel-dispatch
op_kind: RandNormal               # [consumer-ahead] no RandNormal OpKind yet (dispatch.rs:52); registers once it lands
blurb: "Fill a dense buffer with N(mean,stddev) via Box-Muller; no input tensor; advances a backend-internal seed."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::kernels::random::call_random_normal::rand_normal"
kernel_revision_hash: auto

accept:
  inputs: []                       # NO input tensor; seed is backend-private RW state, not a graph operand
  op_params:
    variant: RandNormal            # [consumer-ahead] OpParams::RandNormal (mirrors call_random_normal args)
    fields:
      size:   { kind: usize, note: "output element count; threads = size/2 + (size odd), 2 elems/thread (Box-Muller pair)" }
      mean:   { kind: f32 }
      stddev: { kind: f32 }

return:
  outputs:
    - name: out
      dtype_rule: cast(out)         # output dtype is the requested output Storage dtype (f32/f16/bf16)
      shape_rule: from_params(size)
      layout_guarantee: contiguous
      aliasing: none                # fresh new_buffer; seed RW is backend-private, not a graph tensor alias

caps:
  awkward_layout_strategy: requires_contiguous   # writes a dense [0,size) buffer
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured        # hint: bandwidth-bound elementwise fill — N=size output elements, 2 per thread; Box-Muller adds a transcendental (sqrt/log/sin/cos) pair per thread
  class: cheap_elementwise
  flops: ~                          # transcendental Box-Muller pair per thread; not a clean FLOPs count — Judge-measured
  bytes_moved: "size * dtype_bytes" # write-only fill (plus the tiny RW seed[2]); structural fact of the loop
  overhead_ns: ~
  memory: { device_bytes: "size * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Box-Muller (two outputs per thread from two uniforms), math in f32 then narrowed to f16/bf16. Distribution N(mean,stddev). Reproducible per-seed on identical hardware; not bit-stable cross-hardware. No formal ULP/error bound on a stochastic generator (audited, no static bound applies)."

determinism: nondeterministic
```

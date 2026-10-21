---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda                    # maps to BackendId::Cuda
  kernel_source: "baracuda"        # the BindingEntry.kernel_source tag
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-cuda-backend — cast (dtype-conversion) kernel contracts

Dtype-conversion kernels for the CUDA backend (crate `dispatch`, family `cast`), bound to baracuda
(`register_baracuda_cuda_kernels`, `baracuda_dispatch.rs`). Every kernel here implements
`OpKind::Cast` (`fuel-ir/src/dispatch.rs`) for one fixed source→destination dtype pair. The pair is
the dispatch key: `(OpKind::Cast, [SRC, DST], Cuda) + kernel_source` (§3.2, §12.1). The
`OpParams::Cast` variant is a unit marker (`fuel-dispatch/src/kernel.rs`) — the target dtype lives on
the output Storage's `dtype` field, so each section pins the output with a `fixed(DST)` rule (§5.1).

**One dtype-agnostic wrapper, keyed per pair.** Unlike the CPU cast family (per-target
`cpu_cast_wrapper`s) and the Vulkan cast family (three structural wrappers), the CUDA backend
registers a **single** `cast::cast_baracuda_wrapper` that reads both dtypes off the in/out Storage
and dispatches into `fuel_cuda_backend::baracuda::cast::dispatch`, which picks the FFI symbol from
baracuda's 8×8 surface. So every `(SRC, DST)` key resolves to the SAME `KernelRef` — a
synthetic-base **dtype-fan** umbrella (§3.4, the shape / select / pad-copy precedent), not the
per-pair distinct-wrapper form the CPU/Vulkan cast contracts use. This file is authored per
**destination** dtype: each `## cast_to_<dst>` section fans its `src` operand over every accepted
SOURCE dtype and pins `fixed(<DST>)`, so the importer fans `<entry_point>_<src_suffix>` and keys
`[SRC, DST]` for each source — byte-for-byte the deleted hand-written
`table.register(OpKind::Cast, &[src, dst], …)` double-loop + F8E4M3 legs.

**Coverage (exactly production's 70 pairs).** The full 8×8 cross product over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (`cast_to_{f32,f16,bf16,f64,i32,u32,i64,u8}` = 8 + 8 + 8
+ 8 + 8 + 8 + 8 + 8 = 64, incl. the identity `src == dst` diagonal each backend elides at the
optimizer), plus `F8E4M3 → {F32, F16, BF16}` (the extra `f8e4m3` source on `cast_to_{f32,f16,bf16}`,
3 keys) and `{F32, F16, BF16} → F8E4M3` (`cast_to_f8e4m3`, 3 keys). NO `F8E4M3 → F8E4M3` and NO
`F8E4M3 → {F64, I32, U32, I64, U8}` (baracuda's CastSubBytePlan stops at `{F32, F16, BF16}` for FP8).
`U32` collapses to baracuda `i32` at the FFI (bit-identical for non-negative values). F8E5M2 / S4 /
U4 / Bool and the MX formats (F6E2M3 / F6E3M2 / F4 / F8E8M0) are NOT covered — real gaps, out of
scope for this family.

**Universal facts for every cast in this file.** Input is **contiguous-only** — the CUDA cast is
registered with default (all-false) `KernelCaps` (`_inventory/dispatch.md` marks CU Cast as **C**,
distinct from baracuda's strided elementwise kernels), so `strided_input == false` and the planner
inserts an `Op::Contiguize` (itself an FKC kernel, §4.3) for any strided / broadcast / offset
operand. Output is always freshly-allocated **contiguous**, same logical shape as the input, no
aliasing, not in-place. Every kernel is bandwidth-bound elementwise (reads N src elements, writes N
dst elements, so `bytes_moved` is derivable as `n * (src_bytes + dst_bytes)`); `flops` /
`overhead_ns` / the precise frontier number are `judge_measured` (§4.4) — no cost number is
fabricated. Precision is the author-declared seed `audited: false` → `PrecisionGuarantee::UNAUDITED`
(the Judge audits later, §4.8) — byte-for-byte the deleted hand-written `table.register(...)` regs,
which stamped the default `UNAUDITED` (they set no explicit precision). Pointwise, no cross-thread
reduction, so bit-stable on the same hardware.

---

## cast_to_f32  (dtype conversion → F32, contiguous, dtype-fan over all sources)

Cast every accepted source dtype to F32 through the single `cast_baracuda_wrapper`. The `src` operand
fans over `{F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3}` (9 sources — the 8-set diagonal +
widening/narrowing legs, plus the `F8E4M3 → F32` FP8 decode), pinning `fixed(F32)`, so the importer
keys `(Cast, [SRC, F32], Cuda)` for each. Widening (e.g. F16/BF16 → F32) is exact; narrowing
(F64 → F32) is CUDA RNE; int→float follows CUDA convert semantics; `F8E4M3 → F32` is an exact E4M3
decode. Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_f32
op_kind: Cast
blurb: "Dtype cast -> F32 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8,F8E4M3}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); F8E4M3->F32 is exact decode; identity F32->F32 elided by the optimizer."
  op_params: { variant: Cast }     # OpParams::Cast unit marker; target dtype on the output Storage

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # destination dtype pinned (was cast(output); FKC-recognized §5.1)
      shape_rule: same_as(src)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous    # default caps; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured           # Judge bootstraps; bandwidth-bound elementwise hint below (§4.4)
  class: cheap_elementwise
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read n src elems + write n f32 (4 B)
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,F32>, baracuda-kernels-sys/kernels/elementwise/cast.cu + cast_subbyte_fp8.cu for the F8E4M3 leg): one thread per output element (grid-stride loop), pure static_cast/cast_value with no atomics, no shared memory, no cross-thread reduction; F8E4M3->F32 goes through the fixed __nv_cvt_fp8_to_halfraw intrinsic, likewise thread-local. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_f16  (dtype conversion → F16, contiguous, dtype-fan over all sources)

Cast every accepted source dtype to F16. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3}` (9 sources, incl. the `F8E4M3 → F16` FP8 leg),
pinning `fixed(F16)` → keys `(Cast, [SRC, F16], Cuda)`. Narrowing (F32/F64 → F16) is CUDA RNE with
overflow to ±inf; `F8E4M3 → F16` routes exact-decode-then-RNE (E4M3's ±448 finite range is within
F16 normals). Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_f16
op_kind: Cast
blurb: "Dtype cast -> F16 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8,F8E4M3}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); F8E4M3->F16 within F16 normals; identity F16->F16 elided by the optimizer."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F16)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,__half>, cast.cu + cast_subbyte_fp8.cu for the F8E4M3 leg): one thread per output element (grid-stride loop), cast_value<TIn,__half> routes through fixed __float2half/__half2float intrinsics, no atomics/shared-mem/cross-thread reduction; F8E4M3->F16 likewise thread-local via __nv_cvt_fp8_to_halfraw. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_bf16  (dtype conversion → BF16, contiguous, dtype-fan over all sources)

Cast every accepted source dtype to BF16. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3}` (9 sources, incl. the `F8E4M3 → BF16` FP8 leg),
pinning `fixed(BF16)` → keys `(Cast, [SRC, BF16], Cuda)`. Contiguous-only, bandwidth-bound,
pointwise.

```fkc
kernel: cast_to_bf16
op_kind: Cast
blurb: "Dtype cast -> BF16 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8,F8E4M3}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8, F8E4M3]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); F8E4M3->BF16 exact for in-range; identity BF16->BF16 elided by the optimizer."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(BF16)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,__nv_bfloat16>, cast.cu + cast_subbyte_fp8.cu for the F8E4M3 leg): one thread per output element (grid-stride loop), cast_value<TIn,__nv_bfloat16> routes through fixed __float2bfloat16/__bfloat162float intrinsics, no atomics/shared-mem/cross-thread reduction; F8E4M3->BF16 likewise thread-local. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_f64  (dtype conversion → F64, contiguous, dtype-fan over the 8-set)

Cast every 8-set source dtype to F64. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (8 sources — NO F8E4M3 leg; baracuda's FP8 surface stops
at `{F32, F16, BF16}`), pinning `fixed(F64)` → keys `(Cast, [SRC, F64], Cuda)`. Widening (F32/F16/BF16
→ F64) is exact. Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_f64
op_kind: Cast
blurb: "Dtype cast -> F64 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); no F8E4M3->F64 (FP8 surface stops at {F32,F16,BF16}); identity F64->F64 elided."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F64)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,double>, cast.cu 8-set instantiations): one thread per output element (grid-stride loop), pure static_cast/cast_value with no atomics, no shared memory, no cross-thread reduction, no data-dependent control flow. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_i32  (dtype conversion → I32, contiguous, dtype-fan over the 8-set)

Cast every 8-set source dtype to I32. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (8 sources, NO F8E4M3), pinning `fixed(I32)` → keys
`(Cast, [SRC, I32], Cuda)`. Float→int truncates toward zero (CUDA convert); `U32 → I32` reinterprets
the bit pattern (bit-identical for non-negative). Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_i32
op_kind: Cast
blurb: "Dtype cast -> I32 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_i32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); U32<->I32 bit-identical for non-negative; identity I32->I32 elided."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I32)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,int32_t>, cast.cu 8-set instantiations; U32->I32 is fuel-cuda-backend's fixed bit-reinterpret at the FFI boundary, not a device-side branch): one thread per output element (grid-stride loop), pure static_cast/cast_value, no atomics/shared-mem/cross-thread reduction. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_u32  (dtype conversion → U32, contiguous, dtype-fan over the 8-set)

Cast every 8-set source dtype to U32. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (8 sources, NO F8E4M3), pinning `fixed(U32)` → keys
`(Cast, [SRC, U32], Cuda)`. `U32` collapses to baracuda `i32` at the FFI (bit-identical for
non-negative). Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_u32
op_kind: Cast
blurb: "Dtype cast -> U32 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_u32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); U32 collapses to baracuda i32 at the FFI; identity U32->U32 elided."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U32)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited: dst U32 collapses to baracuda's cast_<src>_i32 symbol (fuel_cuda_backend::baracuda::cast::baracuda_dtype_tag), same cast_contig_kernel<TIn,int32_t> as cast_to_i32 — one thread per output element, pure static_cast/cast_value, no atomics/shared-mem/cross-thread reduction. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_i64  (dtype conversion → I64, contiguous, dtype-fan over the 8-set)

Cast every 8-set source dtype to I64. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (8 sources, NO F8E4M3), pinning `fixed(I64)` → keys
`(Cast, [SRC, I64], Cuda)`. Integer widening (I32/U8 → I64) is exact; float→int truncates toward
zero. Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_i64
op_kind: Cast
blurb: "Dtype cast -> I64 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_i64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); integer widening exact; identity I64->I64 elided."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(I64)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,int64_t>, cast.cu 8-set instantiations): one thread per output element (grid-stride loop), pure static_cast/cast_value, no atomics/shared-mem/cross-thread reduction, no data-dependent control flow. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_u8  (dtype conversion → U8, contiguous, dtype-fan over the 8-set)

Cast every 8-set source dtype to U8. The `src` operand fans over
`{F32, F64, F16, BF16, I32, U32, I64, U8}` (8 sources, NO F8E4M3), pinning `fixed(U8)` → keys
`(Cast, [SRC, U8], Cuda)`. Narrowing to U8 wraps / truncates per CUDA convert semantics.
Contiguous-only, bandwidth-bound, pointwise.

```fkc
kernel: cast_to_u8
op_kind: Cast
blurb: "Dtype cast -> U8 (CUDA/baracuda) from {F32,F64,F16,BF16,I32,U32,I64,U8}; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F64, F16, BF16, I32, U32, I64, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); narrowing to U8 per CUDA convert; identity U8->U8 elided."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast.cuh cast_contig_kernel<TIn,uint8_t>, cast.cu 8-set instantiations): one thread per output element (grid-stride loop), pure static_cast/cast_value, no atomics/shared-mem/cross-thread reduction, no data-dependent control flow. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

---

## cast_to_f8e4m3  (dtype conversion → F8E4M3, contiguous, dtype-fan over {F32, F16, BF16})

Cast the FP8-adjacent float sources to F8E4M3 (1-byte E4M3 float; `DType::F8E4M3`). The `src` operand
fans over ONLY `{F32, F16, BF16}` (baracuda's CastSubBytePlan surface for FP8), pinning
`fixed(F8E4M3)` → keys `(Cast, [SRC, F8E4M3], Cuda)`. The narrow is RNE with saturation to the E4M3
finite range ±448 (E4M3 has no inf encoding). F8E4M3 is a full 1-byte `DType`
(`size_in_bytes() == 1`), so no FDX sub-byte/quant descriptor is needed. Contiguous-only,
bandwidth-bound, pointwise.

```fkc
kernel: cast_to_f8e4m3
op_kind: Cast
blurb: "Dtype cast -> F8E4M3 (CUDA/baracuda) from {F32,F16,BF16}; RNE saturate +/-448; one wrapper; contiguous."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::cast_to_f8e4m3"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "one cast_baracuda_wrapper dispatching on (src,dst); FP8 surface stops at {F32,F16,BF16}; no F8E4M3->F8E4M3."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F8E4M3)
      shape_rule: same_as(src)
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
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read n src elems + write n f8 (1 B)
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Source-audited (baracuda_cast_subbyte.cuh t_to_fp8_kernel / f32_to_e4m3, cast_subbyte_fp8.cu): one thread per output element (grid-stride loop), F32_TO_FP8_FN is the fixed __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) intrinsic, no atomics/shared-mem/cross-thread reduction. Deterministic given identical inputs on the same hardware."

determinism: same_hardware_bitwise
```

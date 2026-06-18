---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu
  kernel_source: "portable-cpu"
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cpu-backend — matmul family kernel contracts

Portable byte-shaped CPU GEMM kernels (`fuel-cpu-backend/src/byte_kernels.rs`). Two op families
are covered here: the bare batched matmul (`OpKind::MatMul`) over float/int dtypes, and the fused
matmul + bias-add (`OpKind::FusedLinear`) over float dtypes. Both consume `OpParams::Matmul`
(`lhs_batch_dims`, `rhs_batch_dims`, `m`, `n`, `k`), share the same GQA-divisible batch contract,
and operate on contiguous, zero-offset, row-major buffers — the pipelined executor's
auto-Contiguize pass realizes any strided / broadcast / offset input *before* these kernels run.
Output buffers are caller pre-allocated to the exact byte size; the matmul family fully overwrites
its output (zero-init then accumulate for float, i32-accumulate then store for int), while
FusedLinear seeds the output with the bias before accumulating. The inner loop order is `(i, k, j)`.
Half floats (`bf16`/`f16`) accumulate in **f32** and narrow on store; integer kernels accumulate in
**i32** and **saturate** on store.

All cost blocks are marked `judge_measured`: the Judge bootstraps them. The FLOPs hint
`2 * m * n * k` (× batch) is the genuinely derivable GEMM flop count (one multiply + one add per
inner-product term); FusedLinear adds the `m * n`-element bias seed. No other coefficients are
fabricated — the Judge populates `bytes_moved` / `overhead_ns` / `memory` from measurement.

---

## matmul_f32  (batched matmul, f32)

Batched matrix multiply `out[..b.., i, j] = Σ_k lhs[..b.., i, k] * rhs[..b.., k, j]`, hand-written
f32 kernel (`byte_kernels.rs:3594`). Operands are contiguous, zero-offset, row-major; the kernel
validates lhs / rhs / out byte sizes against `batch×m×k`, `batch×k×n`, `batch×m×n` respectively
(batch = product of `lhs_batch_dims`). Per-axis batch dims must be equal (`n_rep == 1`) or
**GQA-divisible** (`lhs_batch_dims[d] > rhs_batch_dims[d]` and `lhs % rhs == 0`; the rhs batch slot
read is `lhs_slot / n_rep`); the output batch is the lhs batch. The accumulator zero-inits the
output then accumulates over `k` in native f32 with the `(i, k, j)` loop order — so accumulation
order is fixed and the result is bit-stable on the same hardware. Known limitation: contiguous-only,
no broadcasting on the inner matmul dims — any strided / transposed / offset operand must be
contiguized by the planner first.

```fkc
kernel: matmul_f32
op_kind: MatMul
blurb: "Batched f32 matmul; contiguous row-major; GQA-divisible batch; (i,k,j) native accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)        # lhs_batch ++ [m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; FLOPs hint below is the derivable GEMM count
  class: gemm_like
  flops: "2 * batch * m * n * k"        # one multiply + one add per inner-product term, over all batches

precision:
  bit_stable_on_same_hardware: true     # fixed (i,k,j) order, native f32 accumulate
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                        # CPU primitive: importer applies PRIMITIVE_DETERMINISTIC_CPU (§12.4)
  notes: "Native f32 accumulate, fixed (i,k,j) loop order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## matmul_f64  (batched matmul, f64)

Batched matrix multiply in native f64, the `matmul_native_kernel!` mirror of `matmul_f32`
(`byte_kernels.rs:3973`). Identical algorithm, batch contract, layout contract, and `(i, k, j)`
loop order as `matmul_f32` — only the element dtype and accumulator width differ (native f64). Byte
sizes validated against `batch×m×k`, `batch×k×n`, `batch×m×n`. Output zero-init then accumulate;
bit-stable on the same hardware. Contiguous-only; planner contiguizes awkward operands first.

```fkc
kernel: matmul_f64
op_kind: MatMul
blurb: "Batched f64 matmul; contiguous row-major; GQA-divisible batch; (i,k,j) native accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f64 accumulate, fixed (i,k,j) loop order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## matmul_bf16  (batched matmul, bf16 with f32 accumulator)

Batched matrix multiply over bf16 I/O with an **f32 accumulator** (`matmul_half_kernel!`,
`byte_kernels.rs:3716`). Each `lhs`/`rhs` element widens to f32 for the multiply-accumulate; the
inner product accumulates in f32 (the load-bearing half-precision invariant) and the result narrows
back to bf16 on store. Same GQA-divisible batch contract, contiguous-only layout, and `(i, k, j)`
loop order as `matmul_f32`. Byte sizes validated against `batch×m×k`, `batch×k×n`, `batch×m×n` in
bf16. Output zero-init then accumulate; deterministic on the same hardware. Contiguous-only.

```fkc
kernel: matmul_bf16
op_kind: MatMul
blurb: "Batched bf16 matmul with f32 accumulator; narrow on store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: true     # f32 accumulator, fixed (i,k,j) order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 I/O widened to f32 for multiply-accumulate; f32 accumulator; narrow on store; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## matmul_f16  (batched matmul, f16 with f32 accumulator)

Batched matrix multiply over f16 I/O with an **f32 accumulator** (`matmul_half_kernel!`,
`byte_kernels.rs:3716`), the f16 monomorphization of the bf16 kernel above. Elements widen to f32
for the multiply-accumulate, accumulate in f32, and narrow to f16 on store. Same GQA-divisible batch
contract, contiguous-only layout, and `(i, k, j)` loop order. Byte sizes validated against
`batch×m×k`, `batch×k×n`, `batch×m×n` in f16. Output zero-init then accumulate; deterministic on the
same hardware. Contiguous-only.

```fkc
kernel: matmul_f16
op_kind: MatMul
blurb: "Batched f16 matmul with f32 accumulator; narrow on store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 I/O widened to f32 for multiply-accumulate; f32 accumulator; narrow on store; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## matmul_i8  (batched matmul, i8 with i32 accumulator, saturating store)

Batched matrix multiply over i8 I/O with an **i32 accumulator** (`matmul_int_kernel!`,
`byte_kernels.rs:3843`). Each multiply-accumulate is computed in i32; the inner product accumulates
in i32 (no overflow within the kernel) and the result is **saturating-cast** back to i8 on store
(out-of-range values clamp to `i8::MIN` / `i8::MAX`, not wrap). Same GQA-divisible batch contract,
contiguous-only layout, and `(i, k, j)` loop order as the float kernels. Byte sizes validated
against `batch×m×k`, `batch×k×n`, `batch×m×n` in i8 (1 byte/element). Output i32-accumulate then
store. Pure-integer arithmetic with a fixed accumulation order is **bitwise deterministic across any
compatible hardware**. Contiguous-only; planner contiguizes awkward operands first.

```fkc
kernel: matmul_i8
op_kind: MatMul
blurb: "Batched i8 matmul with i32 accumulator; saturating store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_i8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [I8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)         # i8 in, i8 out (saturated)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: true     # exact integer arithmetic; saturating store is exact-defined
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # integer-exact: a real bounded (exact) claim, not the CPU float default
  notes: "i32 accumulate, saturating cast to i8 on store (clamp to i8::MIN/MAX); exact integer arithmetic, bit-identical across compatible hardware."

determinism: bitwise
```

---

## matmul_u8  (batched matmul, u8 with i32 accumulator, saturating store)

Batched matrix multiply over u8 I/O with an **i32 accumulator** (`matmul_int_kernel!`,
`byte_kernels.rs:3843`), the u8 monomorphization of the i8 kernel above. Multiply-accumulate in i32,
**saturating-cast** back to u8 on store (clamp to `0` / `u8::MAX`, not wrap). Same GQA-divisible
batch contract, contiguous-only layout, `(i, k, j)` loop order. Byte sizes validated against
`batch×m×k`, `batch×k×n`, `batch×m×n` in u8 (1 byte/element). Output i32-accumulate then store.
Pure-integer arithmetic with a fixed accumulation order is **bitwise deterministic across any
compatible hardware**. Contiguous-only.

```fkc
kernel: matmul_u8
op_kind: MatMul
blurb: "Batched u8 matmul with i32 accumulator; saturating store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::matmul_u8"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: rhs
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)         # u8 in, u8 out (saturated)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "i32 accumulate, saturating cast to u8 on store (clamp to 0/u8::MAX); exact integer arithmetic, bit-identical across compatible hardware."

determinism: bitwise
```

---

## fused_linear_f32  (matmul + bias-add, f32)

Fused matmul + bias-add `out[..b.., i, j] = bias[j] + Σ_k a[..b.., i, k] * b[..b.., k, j]`, native
f32 (`fused_linear_native_kernel!`, `byte_kernels.rs:4977`). Three inputs: `a` (lhs), `b` (rhs), and
a 1-D `bias` of length `N` broadcast over `batch×M`. Reuses `OpParams::Matmul` for shape; the kernel
seeds the output accumulator with the bias element (instead of zero) then accumulates over `k`. Same
GQA-divisible batch contract, contiguous-only layout, and `(i, k, j)` loop order as `matmul_f32`.
Output is seeded-overwrite (the seed is the bias, then the inner products accumulate on top), so it
is a full overwrite of the output buffer, not a read-modify-write of prior output content. Native
f32 accumulate, deterministic on the same hardware. Contiguous-only; planner contiguizes awkward
operands first.

```fkc
kernel: fused_linear_f32
op_kind: FusedLinear
blurb: "Fused f32 matmul + bias-add; bias[N] broadcast over batch*M; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_linear_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: b
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
    - name: bias
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=n"        # [N], broadcast over batch*M
  op_params:
    variant: Matmul                       # FusedLinear reuses OpParams::Matmul (dispatch.rs:139-143)
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== a.dim[-2]" }
      n: { kind: usize, constraint: "== b.dim[-1] == bias.dim[0]" }
      k: { kind: usize, constraint: "== a.dim[-1] == b.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: matmul(a, b)            # a_batch ++ [m, n]
      layout_guarantee: contiguous
      aliasing: none                      # output seeded with bias then accumulated; full overwrite, not RMW of prior out

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k + batch * m * n"   # GEMM flops + bias seed (one add per output element)

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f32 accumulate seeded with bias; fixed (i,k,j) loop order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## fused_linear_f64  (matmul + bias-add, f64)

Fused matmul + bias-add in native f64, the `fused_linear_native_kernel!` mirror of
`fused_linear_f32` (`byte_kernels.rs:4978`). Same three-input contract (`a`, `b`, `bias[N]`), same
GQA-divisible batch contract, contiguous-only layout, `(i, k, j)` loop order, and bias-seeded
accumulation — only the element dtype and accumulator width differ (native f64). Output is a
bias-seeded full overwrite; deterministic on the same hardware. Contiguous-only.

```fkc
kernel: fused_linear_f64
op_kind: FusedLinear
blurb: "Fused f64 matmul + bias-add; bias[N] broadcast over batch*M; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_linear_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: b
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
    - name: bias
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=n"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== a.dim[-2]" }
      n: { kind: usize, constraint: "== b.dim[-1] == bias.dim[0]" }
      k: { kind: usize, constraint: "== a.dim[-1] == b.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: matmul(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k + batch * m * n"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Native f64 accumulate seeded with bias; fixed (i,k,j) loop order; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## fused_linear_bf16  (matmul + bias-add, bf16 with f32 accumulator)

Fused matmul + bias-add over bf16 I/O with an **f32 accumulator** (`fused_linear_half_kernel!`,
`byte_kernels.rs:5049`). Elements widen to f32 for the multiply-accumulate; the accumulator is
seeded with the (widened) bias element then accumulates the inner products in f32, narrowing back to
bf16 on store. Same three-input contract (`a`, `b`, `bias[N]`), GQA-divisible batch contract,
contiguous-only layout, and `(i, k, j)` loop order. Output is a bias-seeded full overwrite;
deterministic on the same hardware. Contiguous-only.

```fkc
kernel: fused_linear_bf16
op_kind: FusedLinear
blurb: "Fused bf16 matmul + bias-add with f32 accumulator; narrow on store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_linear_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: b
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
    - name: bias
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=n"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== a.dim[-2]" }
      n: { kind: usize, constraint: "== b.dim[-1] == bias.dim[0]" }
      k: { kind: usize, constraint: "== a.dim[-1] == b.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: matmul(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k + batch * m * n"

precision:
  bit_stable_on_same_hardware: true     # f32 accumulator, fixed (i,k,j) order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 I/O widened to f32; f32 accumulator seeded with bias; narrow on store; deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## fused_linear_f16  (matmul + bias-add, f16 with f32 accumulator)

Fused matmul + bias-add over f16 I/O with an **f32 accumulator** (`fused_linear_half_kernel!`,
`byte_kernels.rs:5050`), the f16 monomorphization of the bf16 kernel above. Elements widen to f32;
the accumulator is seeded with the (widened) bias element then accumulates inner products in f32,
narrowing to f16 on store. Same three-input contract (`a`, `b`, `bias[N]`), GQA-divisible batch
contract, contiguous-only layout, and `(i, k, j)` loop order. Output is a bias-seeded full
overwrite; deterministic on the same hardware. Contiguous-only.

```fkc
kernel: fused_linear_f16
op_kind: FusedLinear
blurb: "Fused f16 matmul + bias-add with f32 accumulator; narrow on store; contiguous; GQA-divisible batch."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::fused_linear_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
    - name: b
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
    - name: bias
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      shape_constraint: "dim[0]=n"
  op_params:
    variant: Matmul
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs > rhs && lhs % rhs == 0)" }
      m: { kind: usize, constraint: "== a.dim[-2]" }
      n: { kind: usize, constraint: "== b.dim[-1] == bias.dim[0]" }
      k: { kind: usize, constraint: "== a.dim[-1] == b.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)
      shape_rule: matmul(a, b)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k + batch * m * n"

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f16 I/O widened to f32; f32 accumulator seeded with bias; narrow on store; deterministic on same hardware."

determinism: same_hardware_bitwise
```

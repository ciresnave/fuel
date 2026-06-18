---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                  # the pure-Rust oracle runs host-side on the CPU substrate
  kernel_source: "reference-oracle"
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"  # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — matmul family contracts

Pure-Rust, correctness-first oracle kernels for the **matmul** family: dense batched/2-D
matmul, the rank-2 transpose and last-two-dims transpose, the N-D axis permutation, and the
dequantize-then-matmul quantized path (`eval_qmatmul` plus its two dequant helpers
`dequantize_blocks` / `dequantize_q4_km_block`).

**Crate-wide layout invariant (the load-bearing fact for every contract here).** `RefTensor<T>`
(`fuel-reference-backend/src/lib.rs:68`) is *always* a contiguous, row-major `Vec`/`Arc<[T]>` plus
a `Shape`; it carries **no strides and no offset**. Every kernel below is therefore, by
construction, **contiguous-only, zero-offset** at the data layer — there is no `is_contiguous()`
branch, no `StridedIndex`, no strided/broadcast/offset/reversed input path anywhere. Callers must
materialize any non-contiguous view into a fresh contiguous `RefTensor` *before* calling. Every
input operand consequently declares `{ contiguous: required, strided: rejected, broadcast_stride0:
rejected, start_offset: rejected, reverse_strides: rejected }`, and every kernel declares
`awkward_layout_strategy: requires_contiguous` (the planner inserts an `Op::Contiguize` — itself an
FKC kernel — and sums its cost; §4.3 / §4.4). This crate is an **oracle, not a production path**: it
exists so the Judge can audit precision and bootstrap cost against a known-correct reference.

All `cost` blocks are marked `provenance: judge_measured`: the reference backend ships no
hand-tuned cost numbers, and the Judge bootstraps the cost of each kernel by measuring this oracle.
Where a FLOPs / bytes-moved formula is *genuinely derivable from the op* (dense matmul is
`2·M·N·K`; a metadata-or-copy reorder is bandwidth-bound at `N·dtype_bytes`), a formula **hint** is
recorded alongside the marker — a derivable upper-bound shape for the Judge to refine, never a
fabricated constant. Coefficients the op does not pin (launch overhead, exact bandwidth) carry no
authored number; `judge_measured` is their only provenance.

---

## matmul  (N-D batched matrix multiply)

N-D batched matmul `C[..b.., i, j] = Σ_k A[..b.., i, k] · B[..b.., k, j]`, generic over
`T: num_traits::Float` and monomorphized to `{F32, F64, BF16, F16}`. Both operands are rank ≥ 2 with
**equal rank**, and the leading batch prefix **must match exactly — there is no batch broadcast and
no GQA-style divisibility in the reference** (`assert_eq!` per batch axis, `ops.rs:595-601`; this is
deliberately stricter than the production `OpParams::Matmul`, which permits GQA-divisible batch).
The rank-2 case defers to `matmul_2d` (`ops.rs:612-614`). The inner loop is a naive triple loop
with the **accumulator in the input dtype `T`** — there is **no f32-accumulation widening** (`acc =
T::zero()`, `ops.rs:636`), so a BF16/F16 matmul accumulates in BF16/F16, which is the
precision-relevant fact a consumer must know (the GPU GEMM paths widen; the oracle intentionally
does not, to expose the un-widened reference). Inputs contiguous zero-offset; output is a fresh
contiguous `[..b.., m, n]` buffer, same dtype as the inputs. The executor's `eval_matmul`
(`exec.rs:1264`) adds a mixed-precision arm (F32 activations × BF16 weights → F32 by upcasting B to
F32 exactly, then F32 matmul); that arm has **no standalone kernel** — it reuses `ops::matmul` — so
it is not a separate contract here. Known limitation: contiguous-only, exact-batch-match only.

```fkc
kernel: matmul
op_kind: MatMul
blurb: "N-D batched dense matmul; equal-rank, exact batch match (no broadcast/GQA); T-precision accumulator (no f32 widening); contiguous."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::matmul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
      shape_constraint: "same_rank=rhs"   # equal rank; batch prefix exact (dim[i]=rhs.dim[i] for i<rank-2); last_dim_eq=rhs.dim[-2] (k)
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"
      shape_constraint: "same_rank=lhs"   # k == lhs.dim[-1] == rhs.dim[-2]; batch dims exact-equal to lhs (no broadcast)
  op_params:
    variant: Matmul                       # OpParams::Matmul (primitive namespace; §3.7)
    fields:
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "== lhs_batch_dims (reference rejects GQA-divisible batch)" }
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # output dtype = input dtype (no widening)
      shape_rule: matmul(lhs, rhs)        # [..lhs_batch.., m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts + costs Op::Contiguize for any non-contiguous operand
  fast_paths:
    - { when: "rank == 2", note: "defers to matmul_2d single-batch inner loop" }
  in_place: false
  alignment_bytes: 1            # host Vec; no SIMD-alignment requirement in the reference
  access_granularity_bits: 8

cost:
  provenance: judge_measured                # oracle ships no authored cost; Judge bootstraps it
  class: gemm_like
  flops: "2 * batch * m * n * k"            # FLOPs hint: derivable from the op (mul+add per inner step); Judge refines
  bytes_moved: "(batch*(m*k + k*n + m*n)) * dtype_bytes"   # read A,B + write C; bandwidth hint
  overhead_ns: ~                            # not authored — judge_measured
  memory: { device_bytes: 0, host_bytes: "batch * m * n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true         # deterministic triple loop, fixed summation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "naive triple loop; accumulator in INPUT dtype T (NO f32 widening) — BF16/F16 accumulate in half. Fixed k-order summation; deterministic. This is the oracle the Judge audits other GEMMs against."

determinism: same_hardware_bitwise
```

---

## matmul_2d  (rank-2 matrix multiply — the inner kernel matmul defers to)

Textbook rank-2 matmul `C[i,j] = Σ_k A[i,k] · B[k,j]` (`ops.rs:651`), generic `T: Float`
(`{F32, F64, BF16, F16}`). This is the **inner kernel the N-D `matmul` defers to for `rank == 2`**
(`ops.rs:612-614`); it is also the per-slice primitive a batched caller loops over. It is not a
distinct graph `OpKind` — both reach it through `Op::MatMul` / `OpParams::Matmul` with empty batch
vectors — so it shares the `MatMul` dispatch identity, distinguished only as the rank-2 cell. Same
numerics as `matmul`: **accumulator in the input dtype `T`, no f32 widening**. Inputs `[m,k]·[k,n]`
contiguous zero-offset; output fresh contiguous `[m,n]`, same dtype. Known limitation:
contiguous-only; rank-2 exactly.

```fkc
kernel: matmul_2d
op_kind: MatMul
blurb: "Rank-2 dense matmul [m,k]·[k,n]→[m,n]; T-precision accumulator (no f32 widening); contiguous; the inner kernel N-D matmul defers to."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::matmul_2d"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [m, k]
      shape_constraint: "last_dim_eq=rhs"  # k == lhs.dim[1] == rhs.dim[0]
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [k, n]
  op_params:
    variant: Matmul                        # OpParams::Matmul with empty batch vectors
    fields:
      lhs_batch_dims: { kind: "Vec<usize>", constraint: "empty (rank-2)" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "empty (rank-2)" }
      m: { kind: usize, constraint: "== lhs.dim[0]" }
      n: { kind: usize, constraint: "== rhs.dim[1]" }
      k: { kind: usize, constraint: "== lhs.dim[1] == rhs.dim[0]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)         # [m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * m * n * k"                    # FLOPs hint derivable from the op; Judge refines
  bytes_moved: "(m*k + k*n + m*n) * dtype_bytes"
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "m * n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "naive double loop; accumulator in INPUT dtype T (NO f32 widening). Fixed k-order summation; deterministic."

determinism: same_hardware_bitwise
```

---

## transpose_2d  (rank-2 physical transpose)

Rank-2 transpose `[m,n] → [n,m]` (`ops.rs:2457`), generic `T: Float` (`{F32, F64, BF16, F16}`).
The reference backend has no strides, so this is a **physical reorder into a fresh contiguous
buffer** — it is *not* the metadata-only zero-copy `Op::Transpose` view of the production graph; it
materializes. It is the rank-2 inner kernel that `transpose_last_two` defers to (`ops.rs` rank-2
shortcut) and the helper `eval_qmatmul` calls to transpose the HF `[N,K]` weight to `[K,N]`
(`exec.rs:1521`). Reached in exec via `Op::Transpose` on a rank-2 input. Input contiguous
zero-offset; output fresh contiguous `[n,m]`, same dtype. Known limitation: rank-2 exactly; copies
(no zero-copy view).

```fkc
kernel: transpose_2d
op_kind: Transpose
blurb: "Rank-2 physical transpose [m,n]→[n,m] into a fresh contiguous buffer (reference materializes; not a zero-copy view)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::transpose_2d"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [m, n]
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(x)           # [n, m] — last two axes swapped
      layout_guarantee: contiguous         # physically reordered to fresh row-major (NOT a view)
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: strided_elementwise               # a permuted copy; bandwidth-bound
  flops: "0"                               # pure data movement, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"       # read every element once, write once (n = m·dim count); bandwidth hint
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true        # exact element copy, no arithmetic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact element relocation; bit-identical to input (no numeric op). Reference materializes — distinct from the production zero-copy Op::Transpose view."

determinism: bitwise                        # pure exact shuffle, hardware-independent
```

---

## transpose_last_two  (swap the last two dims, leading dims batched)

Swap the last two axes of a rank ≥ 2 tensor `[..b.., m, n] → [..b.., n, m]` (`ops.rs:2420`),
generic `T: Float` (`{F32, F64, BF16, F16}` — **not** u32; the bound is `T: Float`). Leading
dimensions are batched; the rank-2 case defers to `transpose_2d` (`ops.rs` rank-2 shortcut). This
is the kernel the reference executor's `Op::Transpose` arm dispatches to (`exec.rs:542`,
`unary!(… ops::transpose_last_two)`). As with `transpose_2d`, the reference has no strides, so it
**physically reorders into a fresh contiguous buffer** rather than producing a zero-copy view.
Input contiguous zero-offset; output fresh contiguous, same dtype. Known limitation: contiguous-only;
copies; `T: Float` only (no integer/index dtypes).

```fkc
kernel: transpose_last_two
op_kind: Transpose
blurb: "Swap the last two dims [..b..,m,n]→[..b..,n,m], leading dims batched; physical reorder to fresh contiguous; F32/F64/BF16/F16."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::transpose_last_two"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                        # rank ≥ 2; leading dims batched
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: from_params(x)           # [..b.., n, m] — last two axes swapped, batch preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "rank == 2", note: "defers to transpose_2d" }
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "0"                               # pure data movement
  bytes_moved: "2 * n * dtype_bytes"       # n = product of all dims; read once + write once; bandwidth hint
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact element relocation; bit-identical to input (no numeric op). Reference materializes — not the production zero-copy view."

determinism: bitwise
```

---

## permute  (N-D axis permutation, physically reordered)

N-D axis permutation `out = x.permute(axes)` (`ops.rs:2363`), bound `T: Clone + Default` — so it
accepts **`{F32, F64, BF16, F16}` AND `U32`** (index tensors), the widest dtype set in this family.
`axes` must be a permutation of `0..rank` (every axis used exactly once). The reference physically
**reorders the data to row-major** for the permuted shape into a fresh contiguous buffer — again not
the zero-copy `Op::Permute` view of the production graph. Reached in exec via `Op::Permute(axes)` →
`eval_permute` (`exec.rs:543`, `exec.rs:1761`), which dispatches the F32/F64/BF16/F16/U32 arms.
Input contiguous zero-offset; output fresh contiguous, **dtype unchanged**. Known limitation:
contiguous-only; copies; `axes` must be a valid permutation.

```fkc
kernel: permute
op_kind: Permute
blurb: "N-D axis permutation by `axes` (a permutation of 0..rank); physical reorder to fresh row-major contiguous; F32/F64/BF16/F16/U32."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::ops::permute"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16, U32]   # T: Clone+Default — includes index tensors (U32)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=8"
  op_params:
    variant: Permute                       # Op::Permute(Vec<usize>); axes carried inline
    fields:
      axes: { kind: "Vec<usize>", constraint: "a permutation of 0..rank (each axis exactly once)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # dtype unchanged
      shape_rule: from_params(x)           # x.dims permuted by `axes`
      layout_guarantee: contiguous         # physically reordered to row-major (NOT a view)
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "0"                               # pure data movement, no arithmetic
  bytes_moved: "2 * n * dtype_bytes"       # n = product of dims; read once + write once; bandwidth hint
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true        # exact element copy (works for U32 too)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "exact element relocation; bit-identical to input (no numeric op), U32 included. Reference materializes — not the production zero-copy Op::Permute view."

determinism: bitwise
```

---

## eval_qmatmul  (dequantize-then-matmul: C = A @ dequant(W_Q))

Quantized matmul `C = A @ dequant(W_Q)`, the reference oracle for the `QMATMUL` fused op
(`exec.rs:1494`). **This is a FUSED op**: the param carrier is **`FusedOpParams::QMatMul { quant_type,
k, n }`** (`FusedOpId(14)`, `fuel-graph/src/registry.rs:250`/`890`), **not** an `OpParams` variant —
`OpKind::QMatMul` (`fuel-core-types/src/dispatch.rs:356`) exists as the op-kind tag but is not the
param carrier (same pattern as PagedAttn, §3.9.1). Activations arrive **F32** `[..,M,K]`; the weight
arrives as **opaque quantized bytes** (a `U32`-backed buffer reinterpreted as bytes — the FDX base
carries opaque `uint8` block storage, FDX §6.1). `quant_type ∈ {Q4_0, Q8_0, Q4_K_M}` selects the
dequant path: the kernel dequantizes the weight to a dense F32 `[N,K]` via `dequantize_blocks`
(Q4_0/Q8_0) or `dequantize_q4_km_block` (Q4_K_M), transposes the HF `[N,K]` convention to `[K,N]`
via `transpose_last_two` (`exec.rs:1521`), then runs the dense F32 `matmul`. Other `quant_type`
values `unimplemented!`. The dequant must **bit-match** the GPU `dequant_q4_0` / `q8_0` / `q4_km`
kernels (the cross-backend correctness contract this oracle exists to enforce).

**Quant scale placement (single-place rule, §3.9.3 / §6).** The weight is `family: GGML_BLOCK` with
the scale baked **INLINE** in each quant super-block (the GGML block layout) — there is **no separate
scale graph input**, so `fdx.quant.scale_operand` stays `~` and the scale lives in exactly one place,
the FDX sidecar `scale_placement = INLINE`. The `ggml_dtype` slot takes the `GgmlDType` **variant
name matched by code** (§3.4): `Q4_0` (code 2), `Q8_0` (code 8), and — for the GGUF `Q4_K_M`
file-format name — **`Q4K` (code 12)**, never the string `Q4_K_M` (which would fail §10.6
`QuantIncoherent`). The op-param `FusedOpParams::QMatMul.quant_type` legitimately uses the production
`QuantType::Q4_K_M` enum variant (a GGUF-name op-param discriminant, distinct from the storage
`GgmlDType`); the FDX `ggml_dtype` weight-format slot uses `Q4K`. Output **F32** `[..,M,N]`, fresh
contiguous. Exec dtypes: activations F32 only. Known limitations: F32 activations only; weight
`quant_type` restricted to the three implemented formats.

```fkc
kernel: eval_qmatmul
fused_op: QMATMUL                          # FusedOpId(14); FusedOpParams::QMatMul (fused namespace)
blurb: "Dequantize-then-matmul C = A @ dequant(W_Q); F32 activations × GGML-block weight (Q4_0/Q8_0/Q4K); F32 output; oracle for QMATMUL."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::exec::eval_qmatmul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a                              # activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=8"                        # [.., M, K]
      shape_constraint: "last_dim_eq=k"    # A.dim[-1] == k
    - name: w_q                            # quantized weight: opaque GGML-block bytes (U8 honesty stand-in; accessed 32-bit internally)
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # opaque byte/word buffer; logical [N,K] via quant blocking
      shape_constraint: "divisible(k, 256) for K-quant super-blocks; n*k/block packed bytes"
      fdx:
        requires_ext: true                 # the bytes' meaning needs the FDX quant sidecar
        sub_byte: ~                         # base carries opaque U8 block storage (FDX §3 honesty stand-in)
        quant:
          family: GGML_BLOCK               # FDXQuant.family — static block-quant
          ggml_dtype: Q4K                   # GgmlDType variant by CODE (Q4_0|Q8_0|Q4K); Q4_K_M GGUF name → Q4K (§3.4)
          # block grain rides ggml_dtype (GGML_BLOCK carries ggml_dtype ONLY — no granularity/PerBlock; FDX §6.2, §10.6)
          role: weight
          scale_operand: ~                  # scale is INLINE (baked) in the super-block — NO separate operand (single-place rule)
  op_params:
    variant: QMatMul                       # FusedOpParams::QMatMul (fused namespace; §3.7)
    fields:
      quant_type: { kind: QuantType, constraint: "in {Q4_0, Q8_0, Q4_K_M} (others unimplemented!)" }
      k: { kind: usize, constraint: "== a.dim[-1]; inner contraction dim" }
      n: { kind: usize, constraint: "output last dim; weight logical rows (HF [N,K])" }

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)             # widened result dtype: F32
      shape_rule: from_params(a)           # [.., M, N]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # internally dequant+transpose to dense F32, then dense matmul
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8

cost:
  provenance: judge_measured                # oracle ships no authored cost; Judge bootstraps it
  class: gemm_like
  # FLOPs hint: dense GEMM after dequant = 2·M·N·K; dequant of the N·K weight is the added bandwidth term.
  flops: "2 * m * n * k"
  bytes_moved: "(m*k*4 + n*k*4 + m*n*4) * 1"   # F32 read A + dequantized W + write C (bytes; weight read smaller when packed)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "(n*k + m*n) * 4", disk_bytes: 0 }   # dense-F32 dequant scratch + output

precision:
  bit_stable_on_same_hardware: true         # deterministic dequant + fixed-order F32 matmul
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "F32 dequant then F32 matmul, fixed summation order; deterministic. Dequant MUST bit-match GPU dequant_q4_0/q8_0/q4_km — this oracle defines the bit-exact target. Only Q4_0/Q8_0/Q4_K_M implemented."

determinism: same_hardware_bitwise
```

---

## dequantize_blocks  (GGML Q4_0 / Q8_0 block dequantization → F32)

Dequantize a GGML block-quantized weight buffer to dense **F32**, for the `Q4_0` and `Q8_0` formats
(`exec.rs:1530`). A helper of `eval_qmatmul` (the `QMATMUL` fused path), not a standalone graph op:
it consumes opaque quantized bytes (`U32`-reinterpreted) plus the `quant_type` and produces a fresh
contiguous F32 tensor whose values **bit-match** the GPU `dequant_q4_0` / `dequant_q8_0` kernels.
Q4_0 packs 32 4-bit weights + one F16 scale per 18-byte block; Q8_0 packs 32 8-bit weights + one F16
scale per 34-byte block (block byte sizes per `QuantType::block_size`, `fuel-graph/src/lib.rs:155`).
The scale is **INLINE in each block** (single-place rule, §3.9.3 / §6) — no separate scale operand.
Known limitation: Q4_0 / Q8_0 only (K-quant goes through `dequantize_q4_km_block`).

> **AS-BUILT DISPATCH NOTE — no fabricated `op_kind` (inv10; mirrors `eval_qmatmul` above and the
> sibling `quantized/dequantize.fkc.md` bundle note).** There is **no `OpKind::DequantizeBlocks`**
> and **no `DequantizeBlocks` `OpParams`/`FusedOpParams` variant** in the as-built dispatch surface
> (`fuel-core-types/src/dispatch.rs` / `fuel-dispatch/src/kernel.rs` / `fuel-graph/src/registry.rs`).
> `dequantize_blocks` is an **internal sub-step of the fused `QMATMUL` kernel** with **no independent
> dispatch identity**: it is called inline by `eval_qmatmul` (`exec.rs:1530`) and never reaches the
> binding table or the fused registry on its own. Per the never-invent discipline (§0, inv10) this
> contract is tagged with the **closest coherent real tag — `fused_op: QMATMUL`** (`FusedOpId(14)`,
> `fuel-graph/src/registry.rs:890`) paired with `op_params.variant: QMatMul` in the **`FusedOpParams`
> namespace** (`registry.rs:250`) — resolving the earlier `op_kind: DequantizeBlocks` (primitive)
> ↔ `op_params: QMatMul` (fused) namespace mismatch (§3.7 / §10.7). The op-level distinction the
> helper *would* carry rides the **`Capability::DequantizeQ4_0` / `DequantizeQ8_0`** tokens
> (`fuel-core-types/src/capability.rs:74-75`), recorded in `caps.notes` — these are real Capability
> tokens, not an `OpKind`. **[consumer-ahead]:** a future `OpKind::Dequantize` (or a standalone
> dequant binding key) would let this register as its own kernel; until it lands the fused `QMATMUL`
> identity is the faithful tag and the sub-step path is authoritative.

```fkc
kernel: dequantize_blocks
fused_op: QMATMUL                          # FusedOpId(14); internal sub-step of the fused QMATMUL kernel — NO independent dispatch identity (note above)
blurb: "GGML block dequant (Q4_0/Q8_0) to dense F32; INLINE block scales; bit-matches GPU dequant_q4_0/q8_0."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::exec::dequantize_blocks"
kernel_revision_hash: auto

accept:
  inputs:
    - name: w_q                            # opaque GGML-block bytes (U8 honesty stand-in; accessed 32-bit internally)
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # opaque byte/word buffer
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0                  # GgmlDType variant by CODE: Q4_0 (2) or Q8_0 (8)
          # block grain rides ggml_dtype (GGML_BLOCK carries ggml_dtype ONLY — no granularity/PerBlock; FDX §6.2, §10.6)
          role: weight
          scale_operand: ~                  # scale INLINE (baked) in each block — NO separate operand
  op_params:
    variant: QMatMul                       # FusedOpParams::QMatMul (fused namespace; §3.7) — quant_type ∈ {Q4_0, Q8_0} selects the dequant path
    fields:
      quant_type: { kind: QuantType, constraint: "in {Q4_0, Q8_0}" }
      k: { kind: usize }
      n: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)             # F32
      shape_rule: from_params(w_q)         # dense [N, K] (n*k F32 elements)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8
  notes: "Internal sub-step of the fused QMATMUL kernel (eval_qmatmul, exec.rs:1530) — NO independent dispatch identity; NO OpKind::DequantizeBlocks. Op-level Capability::DequantizeQ4_0 / DequantizeQ8_0 exist (capability.rs:74-75)."

cost:
  provenance: judge_measured
  class: cheap_elementwise                  # one decode + scale-multiply per output element; bandwidth-bound
  flops: "n * k"                            # FLOPs hint: ~1 scale-multiply per dequantized element
  bytes_moved: "(n*k/2 + n*k*4)"            # read packed nibbles (Q4_0 ≈ half-byte/elem) + write F32 (bytes); bandwidth hint
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "n * k * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true         # deterministic fixed-point decode + F16 scale → F32
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "decode 4-bit (Q4_0) / 8-bit (Q8_0) weight × INLINE F16 block scale → F32. MUST bit-match GPU dequant_q4_0/q8_0 — the bit-exact cross-backend target."

determinism: same_hardware_bitwise
```

---

## dequantize_q4_km_block  (GGML Q4_K_M super-block dequantization → F32)

Dequantize one GGML **Q4_K_M** super-block to **F32** (`exec.rs:1583`). The K-quant helper of
`eval_qmatmul`: a 144-byte super-block (`QuantType::Q4_K_M.block_size == 144`,
`fuel-graph/src/lib.rs:163`) expands to **256 F32** weights (`block_elements == 256`,
`lib.rs:179-181`), reproducing llama.cpp's `get_scale_min_k4` 6-bit packed scale/min extraction. The
"Q4_K_M" name is the GGUF mixed-precision file-format name; its storage dtype is `GgmlDType::Q4K`
(code 12), and the op-level distinction rides `Capability::DequantizeQ4KM`
(`fuel-core-types/src/capability.rs:76`), **not** a separate storage dtype (§3.4). The result must
**bit-match** the GPU `dequant_q4_km` kernel. Scales/mins are **INLINE in the super-block**
(single-place rule, §3.9.3 / §6) — no separate scale operand. Known limitation: Q4_K_M format only;
super-block size fixed at 144 bytes → 256 elements.

> **AS-BUILT DISPATCH NOTE — no fabricated `op_kind` (inv10; mirrors `dequantize_blocks` /
> `eval_qmatmul` above and the sibling `quantized/dequantize.fkc.md` bundle note).**
> **`DequantizeQ4KM` exists ONLY as a `Capability` token** (`fuel-core-types/src/capability.rs:76`)
> — there is **no `OpKind::DequantizeQ4KM`** and **no `DequantizeQ4KM` `OpParams`/`FusedOpParams`
> variant** in the as-built dispatch surface. Like `dequantize_blocks`, this is an **internal sub-step
> of the fused `QMATMUL` kernel** with **no independent dispatch identity**: `eval_qmatmul` calls it
> inline (`exec.rs:1583`) and it never reaches the binding table or the fused registry on its own.
> The closest coherent real tag is therefore **`fused_op: QMATMUL`** (`FusedOpId(14)`,
> `fuel-graph/src/registry.rs:890`) paired with `op_params.variant: QMatMul` in the **`FusedOpParams`
> namespace** (`registry.rs:250`) — resolving the earlier `op_kind: DequantizeQ4KM` (a non-existent
> primitive) ↔ `op_params: QMatMul` (fused) namespace mismatch (§3.7 / §10.7). The op-level
> distinction rides the real **`Capability::DequantizeQ4KM`** token (recorded in `caps.notes`), never
> an `OpKind`. **[consumer-ahead]:** a future `OpKind::Dequantize` / standalone dequant key would let
> this register independently; until then the fused `QMATMUL` identity is the faithful tag.

```fkc
kernel: dequantize_q4_km_block
fused_op: QMATMUL                          # FusedOpId(14); internal sub-step of the fused QMATMUL kernel — NO independent dispatch identity (note above); DequantizeQ4KM is a Capability token only
blurb: "GGML Q4_K_M super-block dequant: 144 bytes → 256 F32 via get_scale_min_k4; INLINE 6-bit scale/min; bit-matches GPU dequant_q4_km."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::exec::dequantize_q4_km_block"
kernel_revision_hash: auto

accept:
  inputs:
    - name: w_q                            # opaque Q4_K_M super-block bytes (U8 honesty stand-in; accessed 32-bit internally)
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # 144-byte super-block buffer
      shape_constraint: "byte length == 144 per super-block; k % 256 == 0"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4K                   # GgmlDType variant by CODE: Q4K (12); GGUF Q4_K_M name → Q4K (§3.4)
          # 6-bit packed per-sub-block scale/min is baked INLINE; block grain rides ggml_dtype (GGML_BLOCK carries ggml_dtype ONLY — no granularity/PerBlock; FDX §6.2, §10.6)
          role: weight
          scale_operand: ~                  # scale/min INLINE (baked) in the super-block — NO separate operand
  op_params:
    variant: QMatMul                       # FusedOpParams::QMatMul (fused namespace; §3.7) — quant_type == Q4_K_M selects this K-quant dequant path
    fields:
      quant_type: { kind: QuantType, constraint: "== Q4_K_M" }
      k: { kind: usize }
      n: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)             # F32
      shape_rule: from_params(w_q)         # 256 F32 per super-block; dense [N, K] overall
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 1
  access_granularity_bits: 8
  notes: "Internal sub-step of the fused QMATMUL kernel (eval_qmatmul, exec.rs:1583) — NO independent dispatch identity; NO OpKind::DequantizeQ4KM. DequantizeQ4KM is a Capability token only (capability.rs:76); storage dtype is GgmlDType::Q4K (code 12)."

cost:
  provenance: judge_measured
  class: cheap_elementwise                  # decode + 2 scale ops per element; bandwidth-bound
  flops: "256 * blocks"                     # FLOPs hint: ~ scale·w + min per element (256 elems/super-block)
  bytes_moved: "(144 * blocks + 256 * 4 * blocks)"   # read 144B/super-block + write 256 F32; bandwidth hint
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: "256 * 4 * blocks", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true         # deterministic 6-bit scale/min extraction + F32 expand
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "144-byte Q4_K_M super-block → 256 F32 via llama.cpp get_scale_min_k4 (6-bit packed scale/min). MUST bit-match GPU dequant_q4_km — the bit-exact cross-backend target."

determinism: same_hardware_bitwise
```

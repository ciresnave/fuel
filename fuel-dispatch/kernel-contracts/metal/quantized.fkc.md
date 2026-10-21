---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                     # maps to BackendId::Metal
  kernel_source: "metal-ggml"        # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — quantized matmul kernel contracts

The two GGML quantized matmul entry points reachable through Fuel's `call_quantized_matmul_*`
wrappers. The GGML `quantized.metal` ships a very large kernel library (soft_max, rope, im2col,
flash-attn, cpy, repeat, and `mul_mv`/`mul_mm` for every quant type); only these two matmul
families are wired into `fuel-metal-backend`, so only they are contracted here. Both compute the
**transposed** GGML matmul `dst = W_q @ x` ("everything is in reverse" — the GGML `ne`/`nb`
shape/stride params are laid out trailing-first), dequantizing a packed weight on the fly and
contracting it against dense `f32` activations, with `f32` accumulation and an always-`f32` output.

Cross-cutting facts for this family (from the Metal inventory, "Quantized matmul" and
`kernels/quantized.rs:24-284`):

- **Per-qtype monomorphization, one contract per family.** Each family is a single `call_*`
  wrapper that selects the `.metal` entry point by the `GgmlDType` discriminant
  (`kernels/quantized.rs:123-139` for `mul_mv`, `:231-247` for `mul_mm`). The wrapper also sets a
  per-qtype threadgroup config (`nth0`/`nth1`/`align`, `mul_mv`). The accepted weight qtype set is
  carried on the weight operand's dtype list; the dispatch key distinguishes the qtypes via the
  per-format quant facts (§3.2 / §12.1). The two families differ in their accepted qtype set:
  `mul_mv` accepts all fifteen `GgmlDType` variants; `mul_mm` **rejects `Q8_1` and `Q8K`**
  (`UnsupportedDTypeForOp("Q8_1"|"Q8K", "qmatmul")`, `kernels/quantized.rs:245-246`).
- **Three "qtypes" are dense, not packed.** `F16`, `BF16`, and `F32` are members of the same
  `GgmlDType` dispatch enum and route to `kernel_mul_mv_f16_f32` / `_bf16_f32` / `_f32_f32`
  (resp. `mul_mm`). For these the weight operand is an ordinary dense tensor (`family: none`, no
  block scale); only the `Q*` qtypes are `GGML_BLOCK` packed. The contract enumerates both regimes
  on the one weight operand (the dense dtypes plus the packed opaque `U8` block-stream form — the
  kernel internally reads it as 32-bit `U32` lanes, surfaced in `access_granularity_bits`, but the
  honest operand dtype is the byte stream, FDX §3).
- **Scale single-place rule (§3.9.3).** GGML block scales are **INLINE** in the packed weight
  block (each `#[repr(C)]` GGML block carries its own f16 scale/min bytes), so there is **no**
  separate FKC scale operand — `fdx.quant.scale_operand` stays `~` and the scale rides the FDX
  tensor's `scale_buffer` (placement INLINE). No scale is ever passed as its own graph input here.
- **Layout, per the as-built `set_params!` wiring.**
  - `mul_mv` (`kernels/quantized.rs:24-176`): the **rhs (quantized weight)** is bound as a bare
    `&Buffer` with **no offset** (`rhs`, `:149`) and all weight strides forced to zero
    (`nb00=nb01=nb02=0`, `:43-45`) — i.e. a **contiguous, zero-offset** weight assumption. The
    **lhs (activations `x`)** is bound with a `BufferOffset` (`(lhs, lhs_offset)`, `:149`) — so it
    is non-zero-offset capable. The **dst** is bound with `dst_offset` (`:149`) — offset-capable.
    `r2`/`r3` (`:58-59`) are GGML batch broadcast ratios (`ne12/ne02`, `ne13/ne03`).
  - `mul_mm` (`kernels/quantized.rs:181-284`): the **src0 (weight)** carries real strides
    `nb01`/`nb02`/`nb03` from `src0_stride` (`:203-205`) — so the weight may sit at non-trivial
    row/batch byte strides — and is bound as a bare `&Buffer` (`src0`, `:257`, offset folded into
    the buffer base). The **src1 (X)** is bound with `src1_offset` (`(src1, src1_offset)`, `:258`)
    and carries strides `nb10..nb13`. The **dst** is bound with `dst_offset` (`:259`). Batched via
    `ne12`/`ne13` with `r2`/`r3` broadcast ratios (`:218-219`). `mul_mm` reserves an 8 KB
    threadgroup buffer (`set_threadgroup_memory_length(0, 8192)`, `:280`).
- **Output is always `f32`, pre-allocated, fully overwritten.** Both wrappers write into a
  caller-supplied `dst` buffer at `dst_offset`; no read of prior content (`aliasing: none`,
  `layout_guarantee: preallocated` + `contiguous`). The output dtype is fixed `F32` regardless of
  the weight qtype (the `_f32` entry-point suffix).
- **Cost `provenance: judge_measured`, with derivable formula hints in the fields.** No absolute
  constant is fabricated. A matmul has a genuinely derivable arithmetic intensity, so each cost
  block carries the **derivable formula** directly in `flops` (`2 * batch_count * m * n * k` MACs)
  and `bytes_moved` (activation read + F32 output; the per-qtype packed-weight stream is the prose
  hint), and `provenance: judge_measured` because the Judge bootstraps the empirical coefficients.
  The non-derivable `overhead_ns` (per-device launch cost) is left as `~` (never a fabricated
  number, never the provenance token sitting in a numeric field). Per-qtype dequant cost varies
  widely (2-bit unpack vs 8-bit copy vs K-quant super-block scale reconstruction) and the per-qtype
  Metal threadgroup config (`nth0`/`nth1`/`align`) shifts occupancy — exactly the kind of per-format
  constant best measured, not guessed. FKC stays agnostic to *how* the Judge measures (§4.4); it
  records the derivable formula and marks the provenance as measurement.
- **Precision is author-declared, Judge-audited.** The GGML kernels accumulate the dot product in
  **f32**; the lossy step is the *weight quantization*, fixed at quantize time, not introduced by
  the matmul. These are GPU kernels with threadgroup/SIMD-group tree reductions whose FP add order
  is scheduler-dependent, so `bit_stable_on_same_hardware: false` and `determinism:
  nondeterministic` (with an audited `none(reason)` precision — no cross-quant ULP bound, since the
  error is dominated by the weight quantization, audited at the model level). This differs from the
  CPU family (a deterministic nested loop with a fixed summation order, which *is* bit-stable).

---

## kernel_mul_mv_qtype_f32  (quantized matrix-vector, transposed → F32)

One-line: GGML quantized GEMV dst = W_q @ x over a per-qtype-dispatched packed weight; f32 accumulate, F32 output.

Quantized matrix-vector multiply (the GGML `mul_mv` family, dispatched per `GgmlDType`). Computes
the transposed GGML product `dst = W_q @ x` with `f32` accumulation and an always-`f32` output,
where the activation `x` is dense `f32` and the weight `W_q` is a packed GGML block stream (or a
dense `F16`/`BF16`/`F32` matrix for the three non-packed qtypes). The wrapper
(`call_quantized_matmul_mv_t`, `kernels/quantized.rs:24-176`) takes the shape tuple `(b, m, n, k)`
and lays out the GGML `ne`/`nb` params "in reverse" (`ne00=k`, `ne01=n`, `ne02=b`, `ne10=k`,
`ne11=m`, `ne12=b`, `ne0=n`, `ne1=m`). The weight strides are forced to zero (`nb00=nb01=nb02=0`),
so the weight is read with a **contiguous, zero-offset** assumption; the activation `lhs` and the
`dst` are bound through a `BufferOffset` (non-zero-offset capable). GGML batch broadcast is carried
by `r2 = ne12/ne02` and `r3 = ne13/ne03`. Per qtype the wrapper selects an entry point
(`kernel_mul_mv_q4_0_f32` … `kernel_mul_mv_q8_K_f32`, plus `_f16_f32`/`_bf16_f32`/`_f32_f32`) and a
threadgroup config: `(nth0,nth1,align) = (8,8,8)` for the 32-element legacy quants
(Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1); `(2,32,4)` for Q2_K/Q3_K/Q5_K; `(4,8,4)` for Q4_K; `(2,32,2)`
for Q6_K; `(32,1,8)` for F16/BF16/Q8_K/F32. The dispatch grid is
`(ceil(n/align), m, b)` threadgroups.

This is the **all-fifteen-qtype** family: `mul_mv` accepts every `GgmlDType` variant
(`Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, F16, BF16, F32`); none are
rejected (unlike `mul_mm`, which rejects `Q8_1`/`Q8K`). Numerics: the dot product accumulates in
f32; the only lossy step is the pre-baked weight quantization. Perf: bandwidth-bound on the packed
weight stream — for an `n × k` weight the packed bytes per qtype follow the GGML block sizes
(e.g. Q4_0 ≈ `n*(k/32)*18`, Q8_0 ≈ `n*(k/32)*34`, K-quants `n*(k/256)*{84..210}`). Limitations:
GEMV shape (`m` is the activation batch/row count, the GEMV inner is `k`); the weight is
contiguous/zero-offset only; output always F32.

Dispatch key: `(QMatMul, [F32 act (lhs), <qtype weight (rhs)>, F32 out], Metal, "metal-ggml")` —
the weight's quant facts (`family=GGML_BLOCK, ggml_dtype=<Q*>` for packed qtypes, `family=none`
for F16/BF16/F32) enrich its operand slot so each qtype is a distinct binding. The op-level token
is the per-format `Capability::MatMul*` (`capability.rs`).

FLOPs/bandwidth (derivable formula hints, carried in the cost fields): `flops = 2 * b * m * n * k`
(MACs); weight traffic ≈ the per-qtype packed block bytes (GGML block size × `n × (k / block_elems)`);
activation `b*m*k*4`; output `b*m*n*4`. `provenance: judge_measured` (per-qtype dequant +
threadgroup-config cost measured, not fabricated); `overhead_ns` left `~`.

```fkc
kernel: kernel_mul_mv_qtype_f32
op_kind: QMatMul
blurb: "GGML quantized GEMV dst = W_q @ x over a per-qtype-dispatched packed weight; f32 accumulate, F32 output."
backend: Metal
kernel_source: "metal-ggml"
entry_point: "fuel_metal_kernels::kernels::call_quantized_matmul_mv_t"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations            # GGML src1 / x (the wrapper's `lhs`)
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 3                       # [b, m, k]
      shape_constraint: "dim[-1]=k"
    - name: weight                  # GGML src0 / W_q (the wrapper's `rhs`); ZERO-OFFSET (nb*=0)
      dtypes: [U8, F16, BF16, F32]  # packed Q* blocks are an opaque U8 byte stream (FDX §3 honesty stand-in); OR dense F16/BF16/F32
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                       # [n, k] dense, or [n, k/block_elems] packed blocks
      fdx:
        requires_ext: true          # the U8 base is meaning-bearing: it IS GGML Q* blocks (no-op for dense F16/BF16/F32)
        quant:
          family: GGML_BLOCK        # for the packed Q* qtypes; `none` for the dense F16/BF16/F32 qtypes
          ggml_dtype: Q4_0          # one of Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q8_1,Q2K,Q3K,Q4K,Q5K,Q6K,Q8K (variant name; §3.4)
          # GGML_BLOCK carries ggml_dtype ONLY: scales are baked INLINE in the block struct —
          # NO scale_granularity, NO PerBlock, NO separate scale operand (FDX §6.2 / V5; per the
          # 2026-06-18 GGML regime-separation fix). `PerBlock` is MX-only.
          role: weight
          scale_operand: ~          # INLINE block scale — single-place rule: NOT a separate operand
  op_params:
    variant: QMatMul                # OpParams::QMatMul (primitive namespace; §3.7)
    fields:
      quant_type:   { kind: QuantType, note: "selects the GgmlDType entry point (mv accepts all 15)" }
      batch_count:  { kind: usize, note: "b" }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "== activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)        # output is always F32 (the `_f32` entry-point suffix)
      shape_rule: from_params(batch_count, m, n)   # [b, m, n]
      layout_guarantee: contiguous
      aliasing: none                # dst written at dst_offset, no read of prior content

caps:
  awkward_layout_strategy: requires_contiguous   # weight zero-offset/contiguous; activations contiguized by planner if needed
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32       # weight (U8 byte stream) read internally as 32-bit U32 lanes; per-qtype threadgroup align nth0/nth1

cost:
  provenance: judge_measured        # Judge bootstraps; per-qtype dequant + threadgroup-config cost measured, not fabricated
  class: gemm_like
  # FLOPs/bandwidth are derivable formula hints (the structural prior the Judge refines);
  # overhead_ns is a genuine per-device launch constant left to the Judge ('~', not fabricated).
  flops: "2 * batch_count * m * n * k"     # MACs (2 flops per MAC)
  # weight ~ per-qtype GGML block size * n * (k/block_elems); act b*m*k*4; out b*m*n*4 (output F32).
  bytes_moved: "(batch_count * m * k * 4) + (batch_count * m * n * 4)"   # activation read + F32 output; + per-qtype packed weight stream
  overhead_ns: ~                    # per-device launch cost — Judge-measured, not fabricated
  memory: { device_bytes: "batch_count * m * n * 4", host_bytes: 0, disk_bytes: 0 }   # F32 output alloc

precision:
  bit_stable_on_same_hardware: false   # GPU SIMD-group/threadgroup reduction: scheduler-dependent FP add order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited, no static bound applies (none(reason))
  notes: "f32 accumulate; only lossy step is the pre-baked weight quantization. Not bit-stable cross- or same-hardware (GPU reduction order); per-quant error audited at model level."

determinism: nondeterministic
```

---

## kernel_mul_mm_qtype_f32  (quantized matrix-matrix, transposed → F32)

One-line: GGML quantized GEMM dst = W_q @ X over a per-qtype-dispatched packed weight; steel-style 64x32 tiles, f32 accumulate, F32 output.

Quantized matrix-matrix multiply (the GGML `mul_mm` family, dispatched per `GgmlDType`). Computes
the transposed GGML product `dst = W_q @ X` with `f32` accumulation and an always-`f32` output,
using a steel-style tiled kernel (64×32 output tiles, 8 KB threadgroup scratch). The wrapper
(`call_quantized_matmul_mm_t`, `kernels/quantized.rs:181-284`) reads the GGML `ne`/`nb` params
"in reverse" from the **src0 (weight)** and **src1 (X)** shapes/strides: `ne00 = src0[-1]`,
`ne01 = src0[-2]`, `ne02 = src0[-3]`, `ne03 = src0[-4]`, with weight strides `nb01/nb02/nb03` taken
from `src0_stride[-2..-4]` (`:203-205`) — so unlike `mul_mv`, the weight here may sit at non-trivial
row/batch byte strides (offset folded into the bound `&Buffer` base). `X` carries `nb10..nb13` and
is bound with `src1_offset` (offset-capable); `dst` is bound with `dst_offset`. Batched via `ne12`
(= `src1[-3]`) and `ne13`, with GGML broadcast ratios `r2 = ne12/ne02`, `r3 = ne13/ne03`. The
dispatch grid is `(ceil(ne11/32), ceil(ne01/64), ne12*ne13)` threadgroups of 128 threads, and the
kernel reserves an 8 KB threadgroup buffer (`set_threadgroup_memory_length(0, 8192)`, `:280`).

This family **rejects `Q8_1` and `Q8K`**: `call_quantized_matmul_mm_t` returns
`UnsupportedDTypeForOp("Q8_1"|"Q8K", "qmatmul")` (`kernels/quantized.rs:245-246`). The accepted
qtype set is therefore the thirteen-variant subset
`Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, F16, BF16, F32` (entry points
`kernel_mul_mm_q4_0_f32` … `kernel_mul_mm_f32_f32`). Numerics: the dot product accumulates in f32;
the only lossy step is the pre-baked weight quantization. Perf: tiled GEMM, bandwidth-bound on the
packed weight for small `m` and compute-bound as `m` grows; per-qtype packed weight traffic follows
the GGML block sizes as in `mul_mv`. Limitations: `Q8_1`/`Q8K` not supported on this path (use
`mul_mv` for those); output always F32.

Dispatch key: `(QMatMul, [<qtype weight (src0)>, F32 X (src1), F32 out], Metal, "metal-ggml")` —
the weight's quant facts (`family=GGML_BLOCK, ggml_dtype=<Q*>` for packed qtypes, `family=none`
for F16/BF16/F32) enrich its operand slot. The op-level token is the per-format
`Capability::MatMul*` (`capability.rs`); `Q8_1`/`Q8K` are not registrable on this key (rejected).

FLOPs/bandwidth (derivable formula hints, carried in the cost fields): `flops = 2 * b * m * n * k`
(MACs, with `b = ne12*ne13` batches); weight traffic ≈ the per-qtype packed block bytes (GGML block
size × `n × (k / block_elems)`); `X` ≈ `b*m*k*4`; output `b*m*n*4`. `provenance: judge_measured`
(per-qtype dequant + tiled-GEMM cost measured, not fabricated); `overhead_ns` left `~`.

```fkc
kernel: kernel_mul_mm_qtype_f32
op_kind: QMatMul
blurb: "GGML quantized GEMM dst = W_q @ X over a per-qtype-dispatched packed weight; steel-style 64x32 tiles, f32 accumulate, F32 output."
backend: Metal
kernel_source: "metal-ggml"
entry_point: "fuel_metal_kernels::kernels::call_quantized_matmul_mm_t"
kernel_revision_hash: auto

accept:
  inputs:
    - name: weight                  # GGML src0 / W_q; carries real strides nb01/nb02/nb03
      dtypes: [U8, F16, BF16, F32]  # packed Q* blocks are an opaque U8 byte stream (FDX §3 honesty stand-in); OR dense F16/BF16/F32
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # ne00..ne03 read from src0 trailing-4 shape (batched: [ne03, ne02, n, k]-reversed)
      fdx:
        requires_ext: true          # the U8 base is meaning-bearing: it IS GGML Q* blocks (no-op for dense F16/BF16/F32)
        quant:
          family: GGML_BLOCK        # for the packed Q* qtypes; `none` for the dense F16/BF16/F32 qtypes
          ggml_dtype: Q4_0          # one of Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q2K,Q3K,Q4K,Q5K,Q6K (variant name; §3.4) — Q8_1/Q8K REJECTED
          # GGML_BLOCK carries ggml_dtype ONLY: scales are baked INLINE in the block struct —
          # NO scale_granularity, NO PerBlock, NO separate scale operand (FDX §6.2 / V5; per the
          # 2026-06-18 GGML regime-separation fix). `PerBlock` is MX-only.
          role: weight
          scale_operand: ~          # INLINE block scale — single-place rule: NOT a separate operand
    - name: X                       # GGML src1 / activations; offset-capable, carries nb10..nb13
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # ne1x/nb1x read from src1 trailing-4 shape
      shape_constraint: "dim[-1]=k"
  op_params:
    variant: QMatMul                # OpParams::QMatMul (primitive namespace; §3.7)
    fields:
      quant_type:   { kind: QuantType, note: "selects the GgmlDType entry point; Q8_1/Q8K -> UnsupportedDTypeForOp" }
      batch_count:  { kind: usize, note: "ne12*ne13" }
      m:            { kind: usize, note: "ne1 = dst[-2]" }
      n:            { kind: usize, note: "ne0 = dst[-1]" }
      k:            { kind: usize, constraint: "== X.dim[-1]; == ne00" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)        # output is always F32 (the `_f32` entry-point suffix)
      shape_rule: matmul(weight, X) # [..batch.., m, n]; ne0=n, ne1=m
      layout_guarantee: contiguous
      aliasing: none                # dst written at dst_offset, no read of prior content

caps:
  awkward_layout_strategy: requires_contiguous   # weight carries strides but path assumes packed-row contiguity; X contiguized by planner if needed
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32       # weight (U8 byte stream) read internally as 32-bit U32 lanes; 8 KB threadgroup scratch, 128-thread tiles

cost:
  provenance: judge_measured        # Judge bootstraps; per-qtype dequant + tiled-GEMM cost measured, not fabricated
  class: gemm_like
  # FLOPs/bandwidth are derivable formula hints (the structural prior the Judge refines);
  # overhead_ns is a genuine per-device launch constant left to the Judge ('~', not fabricated).
  flops: "2 * batch_count * m * n * k"     # MACs (2 flops per MAC), batch_count = ne12*ne13
  # weight ~ per-qtype GGML block size * n * (k/block_elems); X b*m*k*4; out b*m*n*4 (output F32).
  bytes_moved: "(batch_count * m * k * 4) + (batch_count * m * n * 4)"   # X read + F32 output; + per-qtype packed weight stream
  overhead_ns: ~                    # per-device launch cost — Judge-measured, not fabricated
  memory: { device_bytes: "batch_count * m * n * 4", host_bytes: 0, disk_bytes: 0 }   # F32 output alloc

precision:
  bit_stable_on_same_hardware: false   # GPU tiled reduction: scheduler-dependent FP add order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                        # audited, no static bound applies (none(reason))
  notes: "f32 accumulate; only lossy step is the pre-baked weight quantization. Not bit-stable cross- or same-hardware (GPU tiled reduction order); per-quant error audited at model level. Q8_1/Q8K rejected on this path."

determinism: nondeterministic
```

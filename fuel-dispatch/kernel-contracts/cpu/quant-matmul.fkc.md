---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — quantized matmul kernel contracts

The portable CPU quantized matmul family: GGML block-quant weight matmuls (`qmatmul_*`,
delegating to `fuel_quantized::matmul::<GgmlType>`) and the bitsandbytes-style NF4 LUT matmul
(`nf4_matmul_*`). Every kernel here dequantizes a packed weight on the fly and contracts it
against dense F32 (GGML) or T (NF4) activations.

Cross-cutting facts for this family (from the CPU inventory, "Quantized matmul" / "NF4 matmul"):

- **Contiguous-only, zero-offset, row-major.** Like every `CpuStorageBytes` byte kernel, these
  consume flat slices via `as_slice()` / `bytes()` and validate byte length against the declared
  shape; none consult a `Layout`/strides/offset. The pipelined executor's auto-Contiguize pass
  realizes any strided/broadcast/offset input into a contiguous buffer *before* dispatch. Hence
  every operand declares `requires_contiguous`; the planner inserts (and costs, from the CPU
  `contiguize` contract, §4.3/§4.4) an `Op::Contiguize` for a non-contiguous producer. None of
  these kernels walk reverse (negative) strides — `reverse_strides: rejected` throughout.
- **Output pre-allocated, fully overwritten.** Output buffers are caller-allocated to the exact
  byte size; the kernel overwrites with no read of prior content (`aliasing: none`,
  `layout_guarantee: preallocated` + `contiguous`).
- **Scale single-place rule (§3.9.3).** GGML block scales are *INLINE* in the packed weight block
  (the GGML `#[repr(C)]` block carries its own f16 scale/min bytes), so there is **no** separate
  FKC scale operand — `fdx.quant.scale_operand` stays `~` and the scale rides the FDX tensor's
  `scale_buffer` (placement INLINE). NF4's per-block `absmax` is instead a **separate graph input**,
  so it is an ordinary `accept.inputs` operand named in `fdx.quant.scale_operand` and is **not**
  also a sidecar `scale_buffer`.
- **Cost is `judge_measured`.** No FLOPs/bandwidth coefficient is fabricated here. A matmul does
  have a genuinely derivable arithmetic intensity, so each section records a FLOPs/bandwidth
  **formula hint** in prose (`2 * batch * m * n * k` MACs + the dequant traffic), but the cost
  block is marked `provenance: judge_measured`: the Judge bootstraps the coefficients empirically
  (per-format dequant cost varies widely — 2-bit unpack vs 8-bit copy vs K-quant super-block
  scale reconstruction — and is exactly the kind of per-format constant best measured, not
  guessed). FKC stays agnostic to *how* the Judge measures (§4.4); it records only that the
  provenance is measurement.
- **Precision is author-declared, Judge-audited.** All GGML kernels accumulate in **f32**
  (`fuel_quantized::matmul`); the lossy step is the *weight quantization*, fixed at quantize time,
  not introduced by the matmul. NF4 likewise dequant+accumulates in f32 (half I/O round-trips
  through f32). Each is `bit_stable_on_same_hardware: true` (deterministic nested loop, fixed
  summation order) but carries no cross-quant ULP bound (the error is dominated by the weight's
  quantization, audited at the model level).

---

## qmatmul_q4_0_f32  (Q4_0 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q4_0 4-bit block weights; f32 accumulate, F32 output.

Computes `out[b, i, j] = Σ_k A[b, i, k] * dequant(W)[j, k]` where `W` is a flat opaque U8 byte
stream of `n * (k / 32)` `BlockQ4_0` blocks (each 32-element block = 2-byte f16 scale + 16 bytes
of packed u4 quants; 18 bytes/block). Activations `A` are dense F32 `[batch, m, k]`; the weight is
`[n, k/32]` blocks; output is dense F32 `[batch, m, n]`. The kernel reinterprets the weight bytes
as `&[BlockQ4_0]` (a `#[repr(C)]` POD cast guarded by `block_slice_from_bytes`) and delegates the
per-batch inner contraction to `fuel_quantized::matmul::<BlockQ4_0>((m, k, n), …)`, which
dequantizes each block on the fly and **accumulates in f32**. `k` MUST be a multiple of the block
size (32); byte lengths of activations / weight / output are validated against
`batch*m*k*4` / `n*(k/32)*18` / `batch*m*n*4` and a mismatch returns `Result::Err` (never panics).
A zero `batch`/`m`/`n` returns `Ok` with no work. Algorithm: triple loop over batch with the
quantized inner GEMV/GEMM per batch slice. Numerics: the only lossy step is the (pre-baked) weight
quantization; the dot product itself is f32-exact for the dequantized operands and deterministic
(fixed summation order). Perf: bandwidth-bound on the packed weight stream — ~`n*(k/32)*18` weight
bytes read per output column-block, ~`batch*m*n*k` f32 MACs. Limitation: contiguous, zero-offset
only; no GQA/batch broadcasting (plain `batch` replication of the weight); activations F32 only.

Dispatch key: `(QMatMul, [F32 act, <Q4_0 weight>, F32 out], Cpu, "portable-cpu")` — the weight's
quant facts (`family=GGML_BLOCK, ggml_dtype=Q4_0`) enrich its operand slot; the op-level token is
`Capability::MatMulQ4_0` (`capability.rs`).

FLOPs/bandwidth hint: `flops ≈ 2 * batch * m * n * k` (MACs); weight bandwidth `≈ n * (k/32) * 18`
bytes + activation `batch*m*k*4` + output `batch*m*n*4`. Marked `judge_measured` (per-format
dequant cost measured, not fabricated).

```fkc
kernel: qmatmul_q4_0_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q4_0 4-bit block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q4_0_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                         # [batch, m, k]
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as BlockQ4_0). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/32] blocks
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: it IS Q4_0 blocks
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0            # GgmlDType variant name (code 2); §3.4 — block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6)
          role: weight
          scale_operand: ~            # INLINE baked block scale — single-place rule: NOT a separate operand
  op_params:
    variant: QMatMul                  # OpParams::QMatMul (primitive namespace; §3.7)
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q4_0" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # output is always F32 (dequant-and-contract)
      shape_rule: from_params(batch_count, m, n)   # [batch, m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # Judge bootstraps; per-format dequant cost measured, not fabricated
  class: gemm_like
  # FLOPs/bandwidth hint (derivable; Judge refines): flops ~ 2*batch*m*n*k MACs;
  # weight bytes ~ n*(k/32)*18; act batch*m*k*4; out batch*m*n*4.
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop, fixed f32 summation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q4_0 weight quantization. Deterministic; per-quant error audited at model level."

determinism: same_hardware_bitwise
```

---

## qmatmul_q8_0_f32  (Q8_0 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q8_0 8-bit block weights; f32 accumulate, F32 output.

Identical structure to `qmatmul_q4_0_f32` but the weight is `BlockQ8_0` (32-element block = 2-byte
f16 scale + 32 bytes of i8 quants; 34 bytes/block). `out[b, i, j] = Σ_k A[b, i, k] *
dequant(W)[j, k]`, f32 accumulate, F32 output, weight `[n, k/32]` blocks, `k % 32 == 0`. Same
`block_slice_from_bytes` POD cast and per-batch delegation to `fuel_quantized::matmul::<BlockQ8_0>`.
Numerically the least-lossy GGML format here (8-bit quants); same deterministic f32 dot product.
Bandwidth hint: ~`n*(k/32)*34` weight bytes — ~2x the Q4_0 weight traffic for the same shape.

Dispatch key adds `Capability::MatMulQ8_0`.

```fkc
kernel: qmatmul_q8_0_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q8_0 8-bit block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q8_0_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/32] blocks
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q8_0            # code 8 — block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6)
          role: weight
          scale_operand: ~            # INLINE baked block scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q8_0" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q8_0 weight quantization (8-bit, least lossy of this family). Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q4_k_m_f32  (Q4_K_M K-quant super-block matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q4_K_M 256-element K-quant super-block weights; f32 accumulate, F32 output.

Same algorithm with the weight as `BlockQ4K` — a **256-element super-block** (2 bytes f16 d +
2 bytes f16 dmin + 12 bytes of 6-bit-packed sub-block scales/mins + 128 bytes of 4-bit-packed
quants; 144 bytes/block). This is the GGUF `Q4_K_M` ("medium" mixed-precision K-quant) format;
its **storage dtype is `GgmlDType::Q4K` (code 12)** — there is no `Q4_K_M` `GgmlDType` variant
(§3.4). The graph-side `OpParams::QMatMul.quant_type` *does* carry a distinct `QuantType::Q4_K_M`
discriminant (144 bytes/block, 256 elements/block), but `fdx.quant.ggml_dtype` MUST be the storage
variant name `Q4K`; the op-level distinction is the `Capability::MatMulQ4KM` token, not a separate
GgmlDType. `k % 256 == 0`. The dequant cost per super-block is materially higher than the simple
32-element formats (it reconstructs per-sub-block scales/mins from 6-bit packing before the dot
product) — a clear case for `judge_measured` over a guessed coefficient.

Dispatch key: weight `family=GGML_BLOCK, ggml_dtype=Q4K`; op-level `Capability::MatMulQ4KM`.

```fkc
kernel: qmatmul_q4_k_m_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q4_K_M 256-element K-quant super-block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q4_k_m_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/256] super-blocks
      shape_constraint: "divisible(k, 256)"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4K             # GgmlDType variant (code 12); GGUF "Q4_K_M" → Q4K (§3.4) — block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6)
          role: weight
          scale_operand: ~            # INLINE baked block scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q4_K_M" }   # OpParams discriminant (144 B/block)
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 256 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over reconstructed K-quant super-block scales/mins; only lossy step is the pre-baked Q4_K_M weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q4_1_f32  (Q4_1 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q4_1 4-bit affine block weights; f32 accumulate, F32 output.

`BlockQ4_1` weight (32-element block = 2-byte f16 scale d + 2-byte f16 min m + 16 bytes packed u4
quants; 20 bytes/block) — Q4_0 plus a per-block minimum (affine block quant). Generated via
`qmatmul_thin_wrapper!`, delegating to `fuel_quantized::matmul::<BlockQ4_1>`. `k % 32 == 0`, f32
accumulate, F32 output, weight `[n, k/32]` blocks. Same structure/numerics as Q4_0; the extra
per-block min is reconstructed during dequant. Weight bandwidth ~`n*(k/32)*20`.

Dispatch key: weight `ggml_dtype=Q4_1` (code 3).

```fkc
kernel: qmatmul_q4_1_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q4_1 4-bit affine block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q4_1_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q4_1, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q4_1" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q4_1 (affine, per-block d+min) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q5_0_f32  (Q5_0 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q5_0 5-bit block weights; f32 accumulate, F32 output.

`BlockQ5_0` weight (32-element block = 2-byte f16 d + 4-byte high-bit field + 16 bytes packed u4
quants → 5 bits/quant; 22 bytes/block). `k % 32 == 0`, f32 accumulate, F32 output, weight
`[n, k/32]` blocks, `qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ5_0>`. Same structure
as Q4_0 with the 5th bit reassembled from the high-bit field during dequant. Weight bandwidth
~`n*(k/32)*22`.

Dispatch key: weight `ggml_dtype=Q5_0` (code 6).

```fkc
kernel: qmatmul_q5_0_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q5_0 5-bit block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q5_0_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_0, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q5_0" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q5_0 (5-bit) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q5_1_f32  (Q5_1 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q5_1 5-bit affine block weights; f32 accumulate, F32 output.

`BlockQ5_1` weight (32-element block = 2-byte f16 d + 2-byte f16 m + 4-byte high-bit field +
16 bytes packed u4; 24 bytes/block) — Q5_0 plus a per-block min (affine). `k % 32 == 0`, f32
accumulate, F32 output, `qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ5_1>`. Weight
bandwidth ~`n*(k/32)*24`.

Dispatch key: weight `ggml_dtype=Q5_1` (code 7).

```fkc
kernel: qmatmul_q5_1_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q5_1 5-bit affine block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q5_1_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5_1, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q5_1" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q5_1 (5-bit affine, per-block d+min) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q8_1_f32  (Q8_1 block-quant weight matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q8_1 8-bit affine block weights; f32 accumulate, F32 output.

`BlockQ8_1` weight (32-element block = 2-byte f16 d + 2-byte f16 s + 32 bytes i8 quants;
36 bytes/block) — Q8_0 plus a per-block sum/min term. `k % 32 == 0`, f32 accumulate, F32 output,
`qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ8_1>`. Heaviest 32-element block weight
traffic (~`n*(k/32)*36`); numerically the least-lossy 8-bit affine format.

Dispatch key: weight `ggml_dtype=Q8_1` (code 9).

```fkc
kernel: qmatmul_q8_1_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q8_1 8-bit affine block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q8_1_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q8_1, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q8_1" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 32 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate; only lossy step is the pre-baked Q8_1 (8-bit affine, per-block d+s) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q2k_f32  (Q2_K super-block matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q2_K 2-bit K-quant super-block weights; f32 accumulate, F32 output.

`BlockQ2K` weight — a **256-element super-block** (scales + 64 bytes of 2-bit-packed quants +
4 bytes f16 d/dmin; 84 bytes/block). `k % 256 == 0`, f32 accumulate, F32 output,
`qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ2K>`. Smallest footprint K-quant
(2-bit), highest quantization error of the family; per-super-block scale reconstruction makes the
dequant cost non-trivial — a `judge_measured` case.

Dispatch key: weight `family=GGML_BLOCK, ggml_dtype=Q2K` (code 10).

```fkc
kernel: qmatmul_q2k_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q2_K 2-bit K-quant super-block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q2k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/256] super-blocks
      shape_constraint: "divisible(k, 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q2K, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q2K" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 256 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over reconstructed Q2_K super-block scales; only lossy step is the pre-baked 2-bit weight quantization (highest error of the family). Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q3k_f32  (Q3_K super-block matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q3_K 3-bit K-quant super-block weights; f32 accumulate, F32 output.

`BlockQ3K` weight — 256-element super-block (hmask + 64 bytes of 2-bit-packed quants + 12-byte
scales + 2 bytes f16 d; 110 bytes/block; the 3rd bit comes from the hmask). `k % 256 == 0`, f32
accumulate, F32 output, `qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ3K>`. Same
super-block structure/numerics as Q2_K with the extra hmask bit reassembled during dequant.

Dispatch key: weight `ggml_dtype=Q3K` (code 11).

```fkc
kernel: qmatmul_q3k_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q3_K 3-bit K-quant super-block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q3k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q3K, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q3K" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 256 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over reconstructed Q3_K super-block scales + hmask bit; only lossy step is the pre-baked 3-bit weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q5k_f32  (Q5_K super-block matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q5_K 5-bit K-quant super-block weights; f32 accumulate, F32 output.

`BlockQ5K` weight — 256-element super-block (d + dmin + 12-byte scales + 32-byte hmask + 128-byte
qs; 176 bytes/block). `k % 256 == 0`, f32 accumulate, F32 output, `qmatmul_thin_wrapper!` →
`fuel_quantized::matmul::<BlockQ5K>`. K-quant analogue of Q4_K with a 5th bit (hmask).

Dispatch key: weight `ggml_dtype=Q5K` (code 13).

```fkc
kernel: qmatmul_q5k_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q5_K 5-bit K-quant super-block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q5k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q5K, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q5K" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 256 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over reconstructed Q5_K super-block scales + hmask bit; only lossy step is the pre-baked 5-bit weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## qmatmul_q6k_f32  (Q6_K super-block matmul → F32)

One-line: Quantized matmul of dense F32 activations against Q6_K 6-bit K-quant super-block weights; f32 accumulate, F32 output.

`BlockQ6K` weight — 256-element super-block (128-byte ql + 64-byte qh + 16-byte i8 scales +
2 bytes f16 d; 210 bytes/block; 6-bit quants from ql low + qh high). `k % 256 == 0`, f32
accumulate, F32 output, `qmatmul_thin_wrapper!` → `fuel_quantized::matmul::<BlockQ6K>`.
Highest-fidelity K-quant of this family (6-bit); heaviest K-quant weight traffic
(~`n*(k/256)*210`).

Dispatch key: weight `ggml_dtype=Q6K` (code 14).

```fkc
kernel: qmatmul_q6k_f32
op_kind: QMatMul
blurb: "Quantized matmul of dense F32 activations against Q6_K 6-bit K-quant super-block weights; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::qmatmul_q6k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U8]                    # opaque packed block byte stream (FDX §3 honesty stand-in; reinterpreted as Block*). Internal access width is in access_granularity_bits.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
      shape_constraint: "divisible(k, 256)"
      fdx:
        requires_ext: true
        quant: { family: GGML_BLOCK, ggml_dtype: Q6K, role: weight, scale_operand: ~ }   # block grain rides ggml_dtype (no granularity for GGML_BLOCK; §10.6); INLINE baked scale
  op_params:
    variant: QMatMul
    fields:
      quant_type:   { kind: QuantType, constraint: "== Q6K" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize }
      k:            { kind: usize, constraint: "k % 256 == 0; == activations.dim[-1]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: from_params(batch_count, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulate over reconstructed Q6_K super-block scales; only lossy step is the pre-baked 6-bit weight quantization (highest-fidelity K-quant). Deterministic."

determinism: same_hardware_bitwise
```

---

## nf4_matmul_f32  (NF4 LUT block matmul, F32 activations → F32)

One-line: Quantized matmul of dense F32 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; f32 accumulate, F32 output.

bitsandbytes-style NF4 matmul. Three inputs: dense F32 activations `[batch, m, k]` (rank ≥ 2;
leading dims flattened into `batch`); `w_packed` U8 `[n, k/2]` (2 NF4 nibbles per byte — even-k =
low nibble, odd-k = high nibble); and `absmax` F32 `[n, k/block_size]`, the per-output-row,
per-block scale (typically 64 in bitsandbytes). Computes
`out[b, i, j] = Σ_k A[b, i, k] * (NF4_LUT[nibble(w_packed[j, k])] * absmax[j, k/block_size])` with
**f32 accumulation**, writing F32 output `[batch, m, n]`. `k` MUST be **even** (the packed nibble
layout) and a multiple of `block_size`; byte lengths of all four buffers are validated against the
shapes (via `nf4_matmul_check_shapes`) and a mismatch / `k` odd / `k % block_size != 0` returns
`Result::Err`. Zero `batch`/`m`/`n`/`k` zero-fills the output and returns `Ok`. Algorithm: dense
4-deep nested loop (batch, m, n, k) with the NF4 LUT lookup + per-block scale applied per element
inside the k loop (no pre-dequant of the whole weight). Numerics: the lossy step is the NF4
quantization of the weight (the 16-entry normalized-float LUT + per-block absmax), fixed at
quantize time; the dot product is f32-exact and deterministic.

**Scale single-place rule (§3.9.3):** NF4's `absmax` is passed as a **separate graph input**, so it
is an ordinary `accept.inputs` operand and the consuming weight operand names it in
`fdx.quant.scale_operand: absmax`. It is therefore **not** also a sidecar `FDXQuant.scale_buffer`
(that would be `ScaleDoubleDeclared`, §10.6).

**[consumer-ahead] / registrability note.** NF4 is a static block-grained affine quant with a
separate F32 per-block scale, which is exactly the FDX **`AFFINE_BLOCK`** family (FDX code 4,
FDX §6.2): low-bit data (the 4-bit LUT codes) plus a **SEPARATE** per-block absmax scale operand,
its block grain carried by **`block_shape`** — **not** the `PerBlock` granularity code (`PerBlock`
is MX-only, FDX §6.2). NF4 is therefore distinct from `AFFINE_FLOAT` (which is dynamic FP8 affine,
`{PerTensor,PerToken,PerChannel}` only), from GGML `GgmlDType`, and from the F8E8M0-scale `MX`
family. Per §6 there is **no as-built block-quant descriptor target type yet** for `AFFINE_BLOCK`
(the as-built `ScaleGranularity` is exactly `{ PerTensor, PerToken, PerChannel }`, and
`AFFINE_BLOCK` does not map onto it — it does not use a granularity code at all). Today the kernel
is reached via the dedicated `OpKind::Nf4Matmul` / `OpParams::Nf4Matmul` path (no `ScaleGranularity`
lookup), so this contract registers on that path; but the *FDX quant-descriptor* `AFFINE_BLOCK` is
the same forward-looking gap as MX — it **parse-validates (§10.6) but returns `MxNotYetRegistrable`
at registration** until a block-quant descriptor target type lands, the same describe-now/
register-later discipline as MX (§6). The dedicated-op path is what makes the kernel dispatchable
in v1; the FDX `AFFINE_BLOCK` descriptor is advertised for when that target type lands.

FLOPs/bandwidth hint: `flops ≈ 2 * batch * m * n * k` MACs; weight traffic ~`n * (k/2)` bytes +
absmax `n * (k/block_size) * 4` + activation `batch*m*k*4` + output `batch*m*n*4`. Marked
`judge_measured` (per-element LUT-lookup + per-block scale cost measured, not fabricated).

```fkc
kernel: nf4_matmul_f32
op_kind: Nf4Matmul
blurb: "Quantized matmul of dense F32 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::nf4_matmul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                         # [batch, m, k]  (leading dims flattened into batch)
      shape_constraint: "dim[-1]=k"
    - name: w_packed
      dtypes: [U8]                    # 2 NF4 nibbles per byte (NF4 LUT)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/2]
      shape_constraint: "divisible(k, 2); divisible(k, block_size)"
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: NF4 nibbles
        quant:
          family: AFFINE_BLOCK        # FDX code 4 (FDX §6.2): low-bit data + SEPARATE per-block absmax; NOT GGML, NOT MX, NOT AFFINE_FLOAT
          ggml_dtype: ~
          block_shape: [block_size]   # per-output-row per-block grain (op param block_size, typically 64); block grain rides block_shape, NOT PerBlock (PerBlock is MX-only — FDX §6.2)
          granularity: ~              # AFFINE_BLOCK does not use a granularity code; block grain is block_shape
          role: weight
          scale_operand: absmax       # ← SEPARATE_BUFFER per-block absmax scale; single-place rule (§3.9.3); named once, never also an FDX scale_buffer
    - name: absmax
      dtypes: [F32]                   # per-block scale; the SEPARATE scale operand
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block_size]
  op_params:
    variant: Nf4Matmul                # OpParams::Nf4Matmul (primitive namespace; §3.7)
    fields:
      batch:       { kind: usize }
      m:           { kind: usize }
      n:           { kind: usize }
      k:           { kind: usize, constraint: "k % 2 == 0; k % block_size == 0; == activations.dim[-1]" }
      block_size:  { kind: usize, note: "per-block scale granularity, typically 64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)   # T = F32 here
      shape_rule: from_params(batch, m, n)
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
  class: gemm_like
  # FLOPs/bandwidth hint: flops ~ 2*batch*m*n*k MACs; w_packed ~ n*(k/2) bytes;
  # absmax n*(k/block_size)*4; act batch*m*k*4; out batch*m*n*4.
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop, fixed f32 summation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 dequant+accumulate; only lossy step is the pre-baked NF4 (16-entry LUT + per-block absmax) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## nf4_matmul_f16  (NF4 LUT block matmul, F16 activations → F16)

One-line: Quantized matmul of F16 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; dequant+accumulate in f32, F16 output.

Identical to `nf4_matmul_f32` but activations and output are **F16** (`half::f16`). Each activation
is widened to f32 (`to_f32`), the dequant+dot accumulates in **f32**, and the result is narrowed
back to f16 on store (`from_f32`). `w_packed` (U8) and `absmax` (F32) are unchanged. Same shape
contract (`k` even, `k % block_size == 0`), same single-place rule (`absmax` is the separate scale
operand), same `judge_measured` cost posture and `[consumer-ahead]` `AFFINE_BLOCK` registrability
note as `nf4_matmul_f32`. The half round-trip is the only numerical difference from the F32 variant.

```fkc
kernel: nf4_matmul_f16
op_kind: Nf4Matmul
blurb: "Quantized matmul of F16 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; dequant+accumulate in f32, F16 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::nf4_matmul_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: w_packed
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/2]
      shape_constraint: "divisible(k, 2); divisible(k, block_size)"
      fdx:
        requires_ext: true
        quant:
          family: AFFINE_BLOCK        # FDX code 4 (FDX §6.2): low-bit data + SEPARATE per-block absmax; NOT AFFINE_FLOAT, NOT MX
          ggml_dtype: ~
          block_shape: [block_size]   # per-output-row per-block grain (op param block_size, typically 64); block grain rides block_shape, NOT PerBlock (PerBlock is MX-only — FDX §6.2)
          granularity: ~              # AFFINE_BLOCK does not use a granularity code; block grain is block_shape
          role: weight
          scale_operand: absmax       # ← SEPARATE_BUFFER per-block absmax; single-place rule (§3.9.3); named once, never also an FDX scale_buffer
    - name: absmax
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block_size]
  op_params:
    variant: Nf4Matmul
    fields:
      batch:       { kind: usize }
      m:           { kind: usize }
      n:           { kind: usize }
      k:           { kind: usize, constraint: "k % 2 == 0; k % block_size == 0; == activations.dim[-1]" }
      block_size:  { kind: usize, note: "per-block scale granularity, typically 64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)   # T = F16
      shape_rule: from_params(batch, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "F16 activations widen to f32; dequant+accumulate in f32; narrow to f16 on store. Only lossy quant step is the pre-baked NF4 weight quantization plus the half round-trip. Deterministic."

determinism: same_hardware_bitwise
```

---

## nf4_matmul_bf16  (NF4 LUT block matmul, BF16 activations → BF16)

One-line: Quantized matmul of BF16 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; dequant+accumulate in f32, BF16 output.

Identical to `nf4_matmul_f16` but activations and output are **BF16** (`half::bf16`): widen to f32,
dequant+accumulate in **f32**, narrow back to bf16 on store. `w_packed` (U8) and `absmax` (F32)
unchanged; same shape contract, single-place rule, `judge_measured` cost, and `[consumer-ahead]`
`AFFINE_BLOCK` registrability note as the other NF4 variants. The bf16 round-trip is the only
numerical difference from the F32 variant.

```fkc
kernel: nf4_matmul_bf16
op_kind: Nf4Matmul
blurb: "Quantized matmul of BF16 activations against NF4 4-bit LUT weights with a separate per-block F32 absmax scale; dequant+accumulate in f32, BF16 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::nf4_matmul_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3
      shape_constraint: "dim[-1]=k"
    - name: w_packed
      dtypes: [U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/2]
      shape_constraint: "divisible(k, 2); divisible(k, block_size)"
      fdx:
        requires_ext: true
        quant:
          family: AFFINE_BLOCK        # FDX code 4 (FDX §6.2): low-bit data + SEPARATE per-block absmax; NOT AFFINE_FLOAT, NOT MX
          ggml_dtype: ~
          block_shape: [block_size]   # per-output-row per-block grain (op param block_size, typically 64); block grain rides block_shape, NOT PerBlock (PerBlock is MX-only — FDX §6.2)
          granularity: ~              # AFFINE_BLOCK does not use a granularity code; block grain is block_shape
          role: weight
          scale_operand: absmax       # ← SEPARATE_BUFFER per-block absmax; single-place rule (§3.9.3); named once, never also an FDX scale_buffer
    - name: absmax
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block_size]
  op_params:
    variant: Nf4Matmul
    fields:
      batch:       { kind: usize }
      m:           { kind: usize }
      n:           { kind: usize }
      k:           { kind: usize, constraint: "k % 2 == 0; k % block_size == 0; == activations.dim[-1]" }
      block_size:  { kind: usize, note: "per-block scale granularity, typically 64" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)   # T = BF16
      shape_rule: from_params(batch, m, n)
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
  class: gemm_like
  flops: ~
  bytes_moved: ~
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: ~, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "BF16 activations widen to f32; dequant+accumulate in f32; narrow to bf16 on store. Only lossy quant step is the pre-baked NF4 weight quantization plus the half round-trip. Deterministic."

determinism: same_hardware_bitwise
```

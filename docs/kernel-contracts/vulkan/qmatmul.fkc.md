---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — QMatMul (block-quant matmul) kernel contract

The production Vulkan **QMatMul** binding: `out[batch,m,n] = A[batch,m,k] @ dequant(W)[n,k]`, F32
activations against a GGML block-quantized weight stream, F32 output. This is the PRIMITIVE
`OpKind::QMatMul` (`fuel-ir/src/dispatch.rs`) that production registers in the shared
`KernelBindingTable` at the single key `(QMatMul, [F32, U32, F32], Vulkan)` — ONE wrapper
`KernelRef` (`vulkan_dispatch::qmatmul::qmatmul_vk`) that route-picks the per-format Vulkan kernel by
the `OpParams::QMatMul.quant_type` discriminant:

- **Q4_0** → the fused Q4_0×F32 kernels `qmatvec_q4_0` (`M == 1` decode gemv, subgroup reduction over
  K) or `matmul_q4_0_tiled` (`M > 1` prefill, TM=8 shared-memory tiling) — `VulkanBackend::matmul_q4_0_bytes`.
- **Q4_K_M** → `dequantize_q4_km` to an f32 scratch buffer, then `matmul_f32_bytes` —
  `VulkanBackend::matmul_q4_km_bytes`.
- **Q8_0** → `dequantize_q8_0` to an f32 scratch buffer, then `matmul_f32_bytes` —
  `VulkanBackend::matmul_q8_0_bytes`.

Other `QuantType`s (Q4_1 / Q5_* / Q2K / Q3K / Q5K / Q6K / Q8_1) are not yet wired — the wrapper
returns a typed `Err` so the route picker falls back to CPU (never a panic).

**As-built binding model — one wrapper `KernelRef` at one key (route-picking is *internal*).** The
finer-grained per-format Slang kernels (`qmatvec_q4_0` / `matmul_q4_0_tiled` for Q4_0; the
dequant-then-`matmul_f32` pipelines for Q4_K_M / Q8_0) are **route-picker alternatives *inside*
`qmatmul_vk`** (`quant_type` + `m` select), NOT distinct bindings in the table — so they are
described in this wrapper's prose, not as separate `##` sections (a per-kernel section per internal
alternative would register duplicate `KernelRef`s at one key, which `register_into`'s `finalize`
rejects). This mirrors the matmul family's per-combo precedent (one registrable section per binding,
several sharing an algorithm). The corpus `vulkan/quantized.fkc.md` models the two Q4_0 kernels as
ASPIRATIONAL `fused_op: QMATMUL` sections (`qmatvec_q4_0`, `matmul_q4_0_tiled`); those are describe-
only (`registrable: false`) — the future *fused* `FusedOps::QMATMUL` decomposition is a SEPARATE
concern and is NOT what production wires here (the PRIMITIVE `OpKind::QMatMul` binding).

**Weight operand dtype — LOGICAL `U32` (DType-logical / SType-physical split).** `accept.dtypes`
carries the dtype the binding key + `BackendImpl.dtypes` actually use — the as-built reg keys
`[F32, U32, F32]`, so the weight slot is **U32** (the Vulkan wrappers `upload_slice(..., DType::U32)`
a freshly-packed blob and read it as a `U32` ByteAddressBuffer). The PHYSICAL byte-honesty — the
weight is an opaque packed GGML block byte stream, not a wide-int tensor — rides the `fdx.quant`
GGML_BLOCK block below (§3.4 / FDX §3), NOT the operand base dtype. This is the maintainer-approved
CPU linear-quant reconciliation (`fused/linear-quant.fkc.md :: qmatmul`): `quant_coherence` (§6) does
not pin the operand base dtype for a GGML_BLOCK weight, so `[U32]` validates cleanly.

**Scale single-place rule (§3.9.3).** Every format here is a GGML block quant whose scales are
**INLINE** in the block stream (the f16 `d` — and the 6-bit packed per-sub-block scales/mins for
Q4_K_M — ride inside each block), so there is **no** separate FKC scale operand: `fdx.quant.scale_operand`
stays `~` and the scale rides the FDX tensor's INLINE `scale_buffer`.

**Layout model — contiguous-only (matches the as-built reg).** The as-built registration is
`register_with_precision` (no strided caps) — `awkward_layout_strategy: requires_contiguous`
(`strided_input == false`): the GGML weight stream has a fixed per-block layout (per-block scale + N
quantized lanes), so arbitrary strides on the weight buffer would break the dequant kernel's block
walk; activations are contiguous in practice. The planner auto-Contiguizes a strided / transposed /
offset operand *first* and sums the `Op::Contiguize` cost (§4.3). Output is always freshly-allocated
**contiguous** row-major `C[batch, M, N]`, no aliasing, not in-place.

**Cost provenance.** The cost block is `judge_measured`: the Judge bootstraps it (§4.4; per-format
dequant cost varies widely — Q4_0 nibble unpack vs Q8_0 i8 copy vs Q4_K_M super-block scale
reconstruction — exactly the per-format constant best measured, not guessed). The derivable GEMM
FLOP hint `2 * batch * m * n * k` is recorded; no other coefficient is fabricated.

**Determinism (corrected — matches the matmul/conv correction).** Every route accumulates in f32
over a scheduler-dependent reduction: the Q4_0 kernels reduce over K with a subgroup reduction
(`qmatvec_q4_0`) or a tiled reduction (`matmul_q4_0_tiled`), and the Q4_K_M / Q8_0 routes dequantize
(deterministically) then contract through the same `matmul_f32` GEMM whose FADD / subgroup order is
scheduler-dependent (`vulkan/matmul.fkc.md`). So this is `determinism: nondeterministic` with
`bit_stable_on_same_hardware: false` and an audited `none(reason)` precision (no silent unaudited
nondeterminism), matching the matmul / flash-attn precedent and §10 rule 9. The retired hand-written
`VULKAN_QMATMUL_PRECISION` const mis-declared `bit_stable_on_same_hardware: true` (the same over-claim
the retired `VULKAN_MATMUL_PRECISION` made); the Judge audits the corrected seed. The only *quant*
loss is the pre-baked weight quantization, a `QuantType` property audited at the model level, not the
kernel's contribution.

---

## qmatmul_vk  (GGML block-quant matmul wrapper; Q4_0 / Q4_K_M / Q8_0 route-pick)

F32 activations × a GGML block-quant weight → F32 output. The production QMatMul binding
(`qmatmul::qmatmul_vk`): dispatches by `OpParams::QMatMul.quant_type` to the fused Q4_0 kernels
(`qmatvec_q4_0` for `M == 1`, `matmul_q4_0_tiled` for `M > 1`) or the dequant-then-`matmul_f32`
pipelines for Q4_K_M / Q8_0, dequantizing weight blocks on the fly and accumulating in f32. The
weight is an opaque GGML block byte stream (`[N, K/block]` blocks) whose scales are INLINE
(single-place rule); `K` must be a multiple of the format's block size. Output `C[batch, M, N]`
row-major contiguous. Contiguous-only binding — a strided / transposed / offset operand is
auto-Contiguized by the planner first. Dispatch key `(QMatMul, [F32, U32, F32], Vulkan)`.

```fkc
kernel: qmatmul_vk
op_kind: QMatMul
blurb: "GGML block-quant matmul wrapper: F32 activations @ dequant(block-quant W); Q4_0/Q4_K_M/Q8_0 route-pick by quant_type; inline block scales; f32 accumulate; F32 out; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::qmatmul_vk"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]                   # dense F32 activations; key slot 0
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=3"                   # [batch, m, k] (batch optional); auto-Contiguized first
      shape_constraint: "dim[-1]=k"
    - name: weight
      dtypes: [U32]                   # LOGICAL dispatch dtype (DType-logical / SType-physical split, docs/specs/storage-encoding.md): accept.dtypes carries the dtype the binding key + BackendImpl.dtypes actually use — the as-built reg keys [F32, U32, F32], so the weight slot is U32 (the wrappers upload_slice(..., DType::U32) and read a U32 ByteAddressBuffer). The PHYSICAL byte-honesty (opaque packed GGML block BYTE stream; FDX §3 / §3.4; read internally at access_granularity_bits) rides the fdx.quant GGML_BLOCK block below, NOT the operand dtype.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block] blocks; opaque block-packed byte buffer
      shape_constraint: "divisible(k, block)"   # block = 32 (Q4_0/Q8_0) or 256 (Q4_K_M super-block)
      fdx:
        requires_ext: true            # the U32 base is opaque quant bytes: it IS GGML block-quant weight (MEANING_REQUIRES_EXT)
        quant:
          family: GGML_BLOCK          # ggml_dtype IS the format: baked INLINE scales, no granularity (FDX §6.2 regime separation)
          ggml_dtype: Q4_0            # representative GgmlDType variant (code 2); quant_type selects the per-format kernel at dispatch (Q4_0 / Q4K / Q8_0); the per-block grain rides ggml_dtype (GGML_BLOCK carries ggml_dtype ONLY, no granularity; §3.4 / §10.6)
          role: weight
          scale_operand: ~            # INLINE baked block scale — single-place rule (§3.9.3): NOT a separate operand
  op_params:
    variant: QMatMul                  # OpParams::QMatMul (primitive namespace; §3.7)
    fields:
      quant_type:   { kind: QuantType, note: "selects the per-format Vulkan kernel: Q4_0 (fused gemv/tiled) / Q4_K_M / Q8_0 (dequant-then-matmul_f32); other formats fall back to CPU" }
      batch_count:  { kind: usize }
      m:            { kind: usize }
      n:            { kind: usize, note: "output cols; == weight.dim[0]" }
      k:            { kind: usize, constraint: "== activations.dim[-1]; k % block(quant_type) == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # output is always F32 (dequant-and-contract)
      shape_rule: from_params(batch_count, m, n)   # [batch, m, n]
      layout_guarantee: contiguous
      aliasing: none                  # fresh output buffer; kernel overwrites, never aliases input

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "m == 1", class: gemm_like, note: "Q4_0 route selects the qmatvec_q4_0 decode gemv" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8          # byte-extract block reads over the (non-u32-aligned) GGML block stride

cost:
  provenance: judge_measured          # Judge bootstraps; per-format dequant cost measured, not fabricated
  class: gemm_like
  # Derivable GEMM FLOP shape (Judge refines the per-format dequant + launch coefficients):
  flops: "2 * batch * m * n * k"      # multiply-accumulate over the K contraction (dequant on the fly)
  bytes_moved: ~                      # per-format weight-stream bandwidth (18B/block Q4_0, 34B/block Q8_0, 144B/super-block Q4_K_M) + act F32 + out F32 — Judge measures
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures
  memory: { device_bytes: "batch * m * n * 4", host_bytes: 0, disk_bytes: 0 }   # F32 output alloc (+ transient f32 dequant scratch for Q4_K_M/Q8_0)

precision:
  bit_stable_on_same_hardware: false  # f32 accumulate over a scheduler-dependent reduction: Q4_0 subgroup/tiled reduction; Q4_K_M/Q8_0 route through the matmul_f32 GEMM (FADD/subgroup order not pinned)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                       # audited none(reason): non-associative f32 reduction, scheduler-dependent order (§4.8) — the corrected posture (retired VULKAN_QMATMUL_PRECISION over-claimed bit-stable)
  notes: "f32 accumulate over dequantized GGML blocks; Q4_0 fused subgroup/tiled reduction, Q4_K_M/Q8_0 dequant-then-matmul_f32; NOT bit-stable (scheduler-dependent reduction order). The only quant loss is the pre-baked weight quantization (a QuantType property, audited at model level), not the kernel's contribution."

determinism: nondeterministic
```

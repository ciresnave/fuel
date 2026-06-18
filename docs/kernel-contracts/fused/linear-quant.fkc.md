---
fkc_version: 1
provider:
  name: fuel-fused
  backend: Cpu                       # FusedOpRegistry graph entries joined to the always-built CPU payload (BackendId::Cpu)
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_dispatch::dispatch::FUSED_ENTRY_POINTS   # §12.6 symbol→KernelRef map (the single join target: register_default_fused_kernels)
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-fused — linear / quantized matmul fused-op contracts (family: matmul)

Fused-op contracts for the `matmul` family of the `fuel-graph` `FusedOpRegistry` (registry plumbing
`fuel-graph/src/registry.rs`), joined at runtime to the always-built CPU payload via
`fuel-dispatch::dispatch::register_default_fused_kernels`. Five kernels are covered here:
`FusedLinear` (id 2, GEMM + bias epilogue), `QMatMul` (id 14, GGUF/llama.cpp block-quant matmul),
`Nf4Matmul` (id 21, bitsandbytes NF4 LUT matmul), `FusedSoftmaxCrossEntropy` (id 17, fused softmax
+ NLL), and `InplaceAffine` (id 16, in-place `x = mul·x + add`).

Cross-cutting facts (from `docs/kernel-contracts/_inventory/fused.md`, "Cross-cutting facts"):

- **These are FUSED ops, not primitives.** Every contract here declares `fused_op:` (a `FusedOpId`
  name) and an `op_params.variant` in the `fuel_graph::registry::FusedOpParams` namespace (§3.7), so
  its cost compiles to the **fused** cost-fn shape `fn(&[Shape], &FusedOpParams,
  &BackendCapabilities)` — there is **no `&[DType]` argument** (§4.4 / §12.3). Return rules compile
  to the graph-side `FusedOp.shape_rule` / `dtype_rule` (`registry.rs:104,108`).
- **Contiguous-only, zero-offset, row-major (LOAD-BEARING).** The graph-side registry encodes no
  layout; the kernel-side CPU wrappers (`fuel-dispatch/src/dispatch.rs`) take `_layouts: &[Layout]`
  and **ignore it** (underscore-prefixed), calling `cpu_input()` which returns the raw byte buffer
  with no stride application. No `register_fused!` call passes `caps`, so caps default to
  `KernelCaps::empty()` — no fused kernel advertises `strided_input`. A non-contiguous / broadcast /
  offset input is realized to contiguous by the executor's auto-Contiguize step
  (`StridedInputPreferenceFilter`) *before* the kernel runs. Hence every operand declares
  `requires_contiguous` and `reverse_strides: rejected`; the planner inserts (and costs, from the
  CPU `contiguize` contract, §4.3/§4.4) an `Op::Contiguize` for a non-contiguous producer.
- **dtype monomorphization.** CPU coverage is registered per-dtype, almost always
  `{F32, F64, BF16, F16}` (QMatMul = F32 only; Nf4Matmul = F32/F16/BF16; FSCE = logits F32 +
  targets I64). Per the inventory's "Listed as a dtype list on one entry," each contract lists the
  accepted dtype set on a single operand descriptor.
- **Output pre-allocated.** Output buffers are caller-allocated; kernels never allocate
  (`layout_guarantee: contiguous` over a preallocated buffer). All produce a fresh buffer
  (`aliasing: none`) **except `InplaceAffine`, which aliases input 0 by contract** (destructive).
- **Cost is `judge_measured`.** No FLOPs/bandwidth coefficient is fabricated. Where a genuinely
  derivable arithmetic-intensity formula exists (matmul `2·M·N·K`; FusedLinear's `M·N` bias seed;
  FSCE/InplaceAffine elementwise bandwidth), it is recorded **directly in the `flops` / `bytes_moved`
  / `host_bytes` fields as a derivable formula** (formula hints are first-class — they carry no
  fabricated absolute constant). The cost block stays `provenance: judge_measured`, and the one
  genuinely non-derivable absolute — the launch `overhead_ns` — is left as `~` (never a fabricated
  number, never the provenance token in a numeric field) for the Judge to measure. The Judge also
  refines the formula coefficients empirically (per-format quant dequant cost, half round-trip cost,
  log-sum-exp cost are the per-shape constants best measured, not guessed). FKC stays agnostic to
  *how* the Judge measures (§4.4); it records only that the provenance is measurement.
- **Precision is author-declared, Judge-audited.** All CPU fused kernels claim
  `bit_stable_on_same_hardware: true` with no static ULP/relative/absolute bound (inventory cross-cut
  + per-family precision constants). BF16/F16 inputs accumulate in **f32** (F64 for F64 input); FSCE
  uses a stable log-sum-exp in **F64** narrowed to F32. These are deterministic (fixed summation
  order), Judge-auditable seeds.
- **Scale single-place rule (§3.9.3).** QMatMul's GGML block scales are *INLINE* in the packed weight
  block, so there is **no** separate FKC scale operand — `fdx.quant.scale_operand` stays `~` and the
  scale rides the FDX tensor's `scale_buffer` (placement INLINE). Nf4Matmul's per-block `absmax` is a
  **separate graph input**, so it is an ordinary `accept.inputs` operand named in
  `fdx.quant.scale_operand` and is **not** also a sidecar `scale_buffer` (`ScaleDoubleDeclared`
  otherwise, §10.6).

---

## fused_linear  (GEMM + bias epilogue, `(a @ b) + bias`)

One-line: Fused batched matmul + bias-add `(a @ b) + bias` over float dtypes; contiguous row-major; rank-matched ranks ≥ 2; half via f32 accumulate.

`FusedOps::FUSED_LINEAR` (id 2; `fuel-graph/src/registry/fused_linear.rs:27`). Forward-family fused
GEMM with a bias epilogue: `out[..b.., i, j] = bias[j] + Σ_k a[..b.., i, k] * b[..b.., k, j]`. Three
inputs: `a` `[..., M, K]`, `b` `[..., K, N]`, and a rank-1 `bias` `[N]` broadcast over `batch×M`. The
`a` / `b` ranks must match and be ≥ 2 (the matmul output is `[..., M, N]`). All three operands share
one dtype (the dtype rule `= a`); the inventory monomorphizes CPU coverage over the four float dtypes
`{F32, F64, BF16, F16}`, listed here as one operand dtype set. Numerics: native f32/f64 accumulate;
half floats (`bf16`/`f16`) widen each operand to f32, accumulate in **f32**, and narrow on store
(`FUSED_LINEAR_CPU_PRECISION`). The kernel seeds the output accumulator with the bias element (rather
than zero) then accumulates over `k` — a full overwrite of the preallocated output, not a
read-modify-write of prior content (`aliasing: none`). Backward is `NotDifferentiable` in the
registry; the real 3-grad decomposition is handled by `Tensor::backward`'s `Op::Fused(FUSED_LINEAR)`
arm. `decompose` lowers to `MatMul → BroadcastTo(bias) → Add`; the live pattern matcher recognizes
`Add(MatMul(a,b), BroadcastTo(rank-1 bias))` (bias len == matmul last dim; inner MatMul
single-consumer). Limitation: contiguous, zero-offset, row-major only — the planner contiguizes any
strided / transposed / offset operand first.

Dispatch key: `(FusedLinear, [a, b, bias, out] dtypes, Cpu, "portable-cpu")` — all four agree on the
one float dtype.

FLOPs/bandwidth hint: `flops ≈ 2 * batch * m * n * k` (GEMM MACs) `+ batch * m * n` (the bias seed,
one add per output element). Marked `judge_measured` (bandwidth/overhead measured, not fabricated).

```fkc
kernel: fused_linear
fused_op: FUSED_LINEAR
blurb: "Fused batched matmul + bias-add (a @ b) + bias over float dtypes; contiguous row-major; ranks matched >= 2; half via f32 accumulate."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::fused_linear_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2.."                     # [..., M, K]; rank >= 2
      shape_constraint: "dim[-1]=k; same_rank=b"
    - name: b
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2.."                     # [..., K, N]; rank >= 2
      shape_constraint: "dim[-2]=k; same_rank=a"
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                         # [N], broadcast over batch*M
      shape_constraint: "dim[0]=n"
  op_params:
    variant: FusedLinear              # FusedOpParams::FusedLinear (fused namespace; §3.7) — no fields (geometry from shapes)

return:
  outputs:
    - name: out
      dtype_rule: passthrough(a)      # = a (all three operands agree)
      shape_rule: matmul(a, b)        # a_batch ++ [m, n]
      layout_guarantee: contiguous
      aliasing: none                  # output seeded with bias then accumulated; full overwrite, not RMW of prior out

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # derivable formula hints below; the Judge bootstraps the launch overhead
  class: gemm_like
  # Derivable FLOP/bandwidth shape (the Judge refines coefficients): GEMM MACs + the bias seed (one add/output).
  flops: "2 * batch * m * n * k + batch * m * n"     # GEMM MACs + bias-seed adds
  bytes_moved: "(batch * m * k + k * n + n + batch * m * n) * dtype_bytes"   # read a, b, bias; write out
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures it
  memory: { device_bytes: 0, host_bytes: "batch * m * n * dtype_bytes", disk_bytes: 0 }   # output alloc

precision:
  bit_stable_on_same_hardware: true   # fixed accumulation order; f32 (f64 for F64) accumulator
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "FUSED_LINEAR_CPU_PRECISION: native f32/f64 accumulate seeded with bias; bf16/f16 widen to f32, accumulate in f32, narrow on store. Deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## qmatmul  (quantized matmul `C = A @ dequant(W_Q)`, GGUF/llama.cpp block stream)

One-line: Quantized matmul of dense F32 activations against a GGUF/llama.cpp block-quant weight stream; dequant on the fly, f32 accumulate, F32 output.

`FusedOps::QMATMUL` (id 14; `fuel-graph/src/registry/qmatmul.rs:40`). Quantized-family fused op
`C = A @ dequant(W_Q)`: dense **F32** activations `A` `[..., M, K]` contracted against a packed GGML
block-quant weight `W_Q`. Two inputs: `a` (F32 activations) and `w_q_bytes` (an opaque packed block
**byte** stream — the FDX **U8** honesty stand-in for an opaque byte buffer, FDX §3 / §3.4 —
reinterpreted internally as `&[BlockQ*]`). The as-built dispatch tuple `QM_F32 = [F32, U32, F32]`
is **F32 only** (activations and output F32) and uses **U32** in the weight slot; the FDX operand
dtype is the honest **U8** byte stand-in, with the 32-bit internal reinterpretation
(`&[BlockQ*]` over a U32-typed buffer) documented here in prose, never in the operand dtype. The
quant format (`QuantType ∈ Q4_0..Q6K`) is carried by `FusedOpParams::QMatMul { quant_type, k, n }`,
and `k` MUST be a multiple of the format's block size. The kernel dequantizes each block on the fly
and **accumulates in f32**; the only lossy step is the (pre-baked) weight quantization, fixed at
quantize time, not introduced by the matmul. Output `[..., M, N]` is F32 (`M` from `a[-2]`, `N` from
params), a fresh preallocated buffer. Backward is `NotDifferentiable` (frozen weights); `decompose`
**panics** (it deliberately avoids the dequant DRAM round-trip — every backend must register a
kernel), so backends without a native kernel use the executor's `cpu_fallback`. Pattern: stub `None`.
Precision `QMATMUL_CPU_PRECISION` (cost `cost_qmatmul_cpu`). Limitation: contiguous, zero-offset,
row-major only; activations F32 only; no GQA/batch broadcasting (plain `batch` replication).

**Scale single-place rule (§3.9.3):** GGML block scales are *INLINE* in the packed block (the GGML
`#[repr(C)]` block carries its own f16 scale/min bytes), so there is **no** separate FKC scale operand
— `fdx.quant.scale_operand` stays `~` and the scale rides the FDX tensor's `scale_buffer` (placement
INLINE). The `ggml_dtype` slot below uses the storage-variant name matched by numeric code; one
QMatMul kernel covers the format family by the `quant_type` discriminant (the `GgmlDType` storage
variant is what the FDX descriptor names — `Q4_0`..`Q6K`, never the GGUF `Q4_K_M` file-format name,
§3.4). The per-format dispatch distinction is the op-level `Capability` token (`MatMulQ4_0` …
`MatMulQ4KM`, `capability.rs`).

Dispatch key: `(QMatMul, [F32 act, <GGML-block weight>, F32 out], Cpu, "portable-cpu")` — the
weight's quant facts (`family=GGML_BLOCK`, `ggml_dtype` per `quant_type`) enrich its operand slot.

FLOPs/bandwidth hint: `flops ≈ 2 * batch * m * n * k` (MACs); weight bandwidth is the packed block
stream `≈ n * (k / block_elems) * block_bytes` (format-dependent: 18 B/block Q4_0, 34 B/block Q8_0,
144 B/super-block Q4K, …) + activation `batch*m*k*4` + output `batch*m*n*4`. Marked `judge_measured`
(per-format dequant cost varies widely — 2-bit unpack vs 8-bit copy vs K-quant super-block scale
reconstruction — exactly the per-format constant best measured, not guessed).

```fkc
kernel: qmatmul
fused_op: QMATMUL
blurb: "Quantized matmul of dense F32 activations against a GGUF block-quant weight stream; dequant on the fly, f32 accumulate, F32 output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::qmatmul_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a
      dtypes: [F32]                   # dense F32 activations; QM_F32 = [F32, U32, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2.."                     # [..., M, K]
      shape_constraint: "dim[-1]=k"
    - name: w_q_bytes
      dtypes: [U8]                    # FDX honesty stand-in for an opaque packed GGML block BYTE stream (kDLUInt bits:8; FDX §3 / §3.4); reinterpreted internally as &[BlockQ*]. As-built dispatch tuple QM_F32 = [F32, U32, F32] uses U32 in the weight slot; the 32-bit internal reinterpretation rides prose, not the operand dtype.
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block_elems] blocks
      shape_constraint: "divisible(k, block_elems)"   # block_elems = 32 (Q4_0..Q8_1) or 256 (K-quants)
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: it IS GGML quant blocks
        quant:
          family: GGML_BLOCK          # ggml_dtype IS the format: baked INLINE scales, no granularity, no PerBlock (FDX §6.2 regime separation)
          ggml_dtype: Q4_0            # GgmlDType STORAGE variant name per quant_type (Q4_0..Q6K, code-matched; §3.4)
          granularity: ~              # GGML carries NO scale_granularity / NO PerBlock — the scale is baked per-format from ggml_dtype (FDX §6.2; PerBlock stays MX-only)
          role: weight
          scale_operand: ~            # INLINE baked block scale — single-place rule: NOT a separate operand
  op_params:
    variant: QMatMul                  # FusedOpParams::QMatMul (fused namespace; §3.7)
    fields:
      quant_type: { kind: QuantType, note: "Q4_0..Q6K; selects block format + block_elems/block_bytes" }
      k:          { kind: usize, constraint: "k % block_elems == 0; == a.dim[-1]" }
      n:          { kind: usize, note: "output cols; == w_q_bytes.dim[0]" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # output is always F32 (dequant-and-contract); = a (F32)
      shape_rule: from_params(a, n)   # [..., M, N]; M from a[-2], N from params
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # per-format dequant cost + launch overhead measured; FLOP/byte shape below is derivable
  class: gemm_like
  # Derivable FLOP/bandwidth shape (the Judge refines coefficients; per-format dequant constant is measured):
  # weight stream = n*(k/block_elems)*block_bytes (block_bytes format-dependent), act F32, out F32.
  flops: "2 * batch * m * n * k"      # GEMM MACs (dequant on the fly)
  bytes_moved: "(batch * m * k * 4) + (n * (k / block_elems) * block_bytes) + (batch * m * n * 4)"   # act F32 + weight blocks + out F32
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures it
  memory: { device_bytes: 0, host_bytes: "batch * m * n * 4", disk_bytes: 0 }   # F32 output alloc

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop, fixed f32 summation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "QMATMUL_CPU_PRECISION: f32 accumulate over dequantized blocks; only lossy step is the pre-baked GGML weight quantization. Deterministic; per-quant error audited at model level."

determinism: same_hardware_bitwise
```

---

## nf4_matmul  (bitsandbytes 4-bit NormalFloat LUT quantized matmul)

One-line: bitsandbytes NF4 4-bit LUT quantized matmul with a separate per-block F32 absmax scale; dequant+accumulate in f32, dtype = activations output.

`FusedOps::NF4_MATMUL` (id 21; `fuel-graph/src/registry/nf4_matmul.rs:68`). Quantized-family fused op
implementing a bitsandbytes-style 4-bit NormalFloat matmul. Three inputs: `activations` `[..., M, K]`
(F32/F16/BF16 in v1); `w_packed` `[N, K/2]` **U8** (two NF4 codes per byte — even-k = low nibble,
odd-k = high nibble; `K` even); and `absmax` `[N, K/block_size]` **F32**, the per-output-row,
per-block scale (typically 64). Computes `out[b, i, j] = Σ_k A[b, i, k] *
(NF4_LUT[nibble(w_packed[j, k])] * absmax[j, k/block_size])` with **f32 accumulation** (F16/BF16
activations up-cast on load, narrow on store), output dtype = activations, `[..., M, N]`
(`N = w_packed[0]`), a fresh preallocated buffer. The 16-entry NF4 LUT is baked into the kernel.
`K` MUST be even and a multiple of `block_size` (`FusedOpParams::Nf4Matmul { block_size }`; require
`K % block_size == 0`). Backward is `NotDifferentiable` (frozen weights); `decompose` **panics**
(avoids the dequant round-trip — backends register a kernel). Pattern: stub `None`. Precision
`NF4_MATMUL_CPU_PRECISION` (f32 inner-product accumulator). Limitation: contiguous, zero-offset,
row-major only.

**Scale single-place rule (§3.9.3):** NF4's `absmax` is a **separate graph input**, so it is an
ordinary `accept.inputs` operand and the consuming weight operand names it in
`fdx.quant.scale_operand: absmax`. It is therefore **not** also a sidecar `FDXQuant.scale_buffer`
(that would be `ScaleDoubleDeclared`, §10.6).

**[consumer-ahead] / registrability note.** NF4 is a static block-grained affine quant: low-bit data
(NF4, the F4 sub-byte code) plus a **separate** per-block F32 absmax scale. In FDX terms this is the
**`AFFINE_BLOCK`** family (FDX §6.2, code 4) — distinct from MX (no F8E8M0 / no `PerBlock`) and from
GGML (the scale is a separate operand, not baked). The block grain rides `block_shape: [64]`, **not** a
`ScaleGranularity` code (`PerBlock` stays MX-only, FDX §6.2); the absmax is named exactly once as a
separate operand (single-place rule). NF4 is neither a GGML `GgmlDType` nor the F8E8M0-scale `MX`
family. Today the kernel is reached via the dedicated `FusedOpParams::Nf4Matmul` path, so this
contract registers on that path; an importer that reaches the `AFFINE_BLOCK` family before the FDX
quant codes land returns the `MxNotYetRegistrable`-class "AFFINE_BLOCK not yet registrable" error.
The dedicated-op path makes the kernel dispatchable in v1; the `AFFINE_BLOCK` family + `block_shape`
is advertised for when the FDX quant descriptor lands.

Dispatch key: `(Nf4Matmul, [act, <NF4 weight>, F32 absmax, out] dtypes, Cpu, "portable-cpu")` — the
weight slot carries `family=AFFINE_BLOCK, block_shape=[64]` (no granularity code); the separate
`absmax` is its own operand slot in the key.

FLOPs/bandwidth hint: `flops ≈ 2 * batch * m * n * k` MACs; weight traffic `≈ n * (k/2)` bytes +
absmax `n * (k/block_size) * 4` + activation `batch*m*k*4` + output `batch*m*n*dtype_bytes`. Marked
`judge_measured` (per-element LUT-lookup + per-block scale + half round-trip cost measured, not
fabricated).

```fkc
kernel: nf4_matmul
fused_op: NF4_MATMUL
blurb: "bitsandbytes NF4 4-bit LUT quantized matmul with a separate per-block F32 absmax scale; dequant+accumulate in f32, dtype = activations output."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::nf4_matmul_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32, F16, BF16]        # v1 activation dtypes; F16/BF16 up-cast on load
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2.."                     # [..., M, K]  (leading dims flattened into batch)
      shape_constraint: "dim[-1]=k"
    - name: w_packed
      dtypes: [U8]                    # 2 NF4 nibbles per byte (16-entry NF4 LUT baked in kernel)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/2]
      shape_constraint: "divisible(k, 2); divisible(k, block_size)"
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: NF4 nibbles
        sub_byte: F4                  # NF4 rides the F4 sub-byte logical_dtype (FDX §6.1); 4-bit, 2 codes/byte
        quant:
          family: AFFINE_BLOCK        # nf4/QLoRA: low-bit data + a SEPARATE block-shaped scale operand (FDX §6.2, code 4); NOT GGML, NOT MX, NOT AFFINE_FLOAT
          ggml_dtype: ~
          granularity: ~              # block grain rides block_shape, NOT a granularity code; PerBlock stays MX-only (FDX §6.2)
          block_shape: [64]           # one absmax scale per 64-element block along the flattened weight (QLoRA default; FDX §13.5a)
          role: weight
          scale_operand: absmax       # ← separate graph input (block-shaped absmax); single-place rule (§3.9.3), NOT also a sidecar scale_buffer
    - name: absmax
      dtypes: [F32]                   # per-block scale; the SEPARATE scale operand
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                         # [n, k/block_size]
  op_params:
    variant: Nf4Matmul                # FusedOpParams::Nf4Matmul (fused namespace; §3.7)
    fields:
      block_size: { kind: usize, note: "per-block scale granularity, typically 64; require k % block_size == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)   # = activations (F32/F16/BF16)
      shape_rule: from_params(activations)   # [..., M, N]; N = w_packed.dim[0]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured          # per-element LUT + per-block scale + half round-trip cost + launch overhead measured
  class: gemm_like
  # Derivable FLOP/bandwidth shape (the Judge refines coefficients; per-element LUT/scale constant is measured).
  flops: "2 * batch * m * n * k"      # GEMM MACs (NF4 dequant via LUT on the fly)
  bytes_moved: "(n * (k / 2)) + (n * (k / block_size) * 4) + (batch * m * k * 4) + (batch * m * n * dtype_bytes)"   # w_packed + absmax + act + out
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures it
  memory: { device_bytes: 0, host_bytes: "batch * m * n * dtype_bytes", disk_bytes: 0 }   # output alloc

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loop, fixed f32 summation order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "NF4_MATMUL_CPU_PRECISION: f32 dequant+accumulate; F16/BF16 activations widen to f32, narrow on store. Only lossy quant step is the pre-baked NF4 (16-entry LUT + per-block absmax) weight quantization. Deterministic."

determinism: same_hardware_bitwise
```

---

## fused_softmax_cross_entropy  (fused softmax + NLL over class-index targets)

One-line: Fused softmax + negative-log-likelihood over class-index targets; logits F32 + targets I64; stable F64 log-sum-exp; output always F32.

`FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY` (id 17;
`fuel-graph/src/registry/fused_softmax_cross_entropy.rs:68`). Forward-family fused op combining a
softmax with a negative-log-likelihood reduction over integer class-index targets. Two inputs:
`logits` `[..., V]` (**F32**) and `targets` `[...]` (**I64** class indices). Params
`FusedOpParams::FusedSoftmaxCrossEntropy { reduction: Reduction(Mean|Sum|None), ignore_index: i64 }`.
Numerics: a **stable log-sum-exp computed in F64** (max-subtract, exp, sum, log), narrowed to F32 on
store (`FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION`). The output **dtype is always F32 regardless of
input dtype** (`dtype_rule: fixed(F32)`); the output **shape** depends on the reduction: `Mean`/`Sum`
→ a scalar `[]`, `None` → `targets.shape` (one loss per row). The forward CPU kernel **does** honor
`ignore_index` (masks those targets). Backward is **`Decompose`** — autograd lowers and runs the
primitive backward (re-introducing `[..., V]` intermediates). `decompose` lowers to log-softmax
(`ReduceMax→Sub→Exp→ReduceSum→Log→Sub`) + `Cast(targets→U32)→Unsqueeze→Gather→Squeeze→MulScalar(−1)`
→ reduce, ending with a Cast to F32 if the work dtype ≠ F32; **`ignore_index` is NOT honored in the
lowered form** (only the forward CPU kernel masks). Pattern: stub `None` (explicit builder opt-in).
Limitation: contiguous, zero-offset, row-major only.

Dispatch key: `(FusedSoftmaxCrossEntropy, [F32 logits, I64 targets, F32 out], Cpu, "portable-cpu")`.

FLOPs/bandwidth hint: the softmax over the vocab axis is `≈ N` log-sum-exp work over the
`prod(targets.shape) * V` logit elements (a few transcendental ops per element); bandwidth ≈ read
`logits` (`prod(targets.shape) * V * 4` B) + `targets` (`prod(targets.shape) * 8` B), write the
reduced output. Marked `judge_measured` (the F64 log-sum-exp transcendental cost is measured, not
fabricated).

```fkc
kernel: fused_softmax_cross_entropy
fused_op: FUSED_SOFTMAX_CROSS_ENTROPY
blurb: "Fused softmax + NLL over class-index targets; logits F32 + targets I64; stable F64 log-sum-exp; output always F32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::fused_softmax_cross_entropy_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: logits
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."                     # [..., V]; last axis = vocab/class dimension
    - name: targets
      dtypes: [I64]                   # class-index targets
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "0.."                     # [...]; == logits.shape minus the last (V) axis
      shape_constraint: "same_as=logits[..-1]"   # one target per row; logits last dim is the class axis
  op_params:
    variant: FusedSoftmaxCrossEntropy   # FusedOpParams::FusedSoftmaxCrossEntropy (fused namespace; §3.7)
    fields:
      reduction:    { kind: Reduction, note: "Mean | Sum | None — selects scalar vs per-row output shape" }
      ignore_index: { kind: i64, note: "targets == ignore_index masked in forward CPU kernel (NOT honored in lowered decompose form)" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)          # ALWAYS F32 regardless of input dtype
      shape_rule: from_params(reduction)   # Mean/Sum -> scalar []; None -> targets.shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # F64 log-sum-exp transcendental cost + launch overhead measured
  class: reduction
  # Derivable shape (the Judge refines coefficients): n_rows = prod(targets.shape); V = logits last dim.
  # A few transcendental ops per logit element over the n_rows*V softmax + the per-row NLL gather.
  flops: "n_rows * v"                 # ~ log-sum-exp work over the n_rows*V logits (transcendental-dominated)
  bytes_moved: "(n_rows * v * 4) + (n_rows * 8)"   # read logits F32 + targets I64; reduced output write is negligible
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures it
  memory: { device_bytes: 0, host_bytes: "n_rows * 4", disk_bytes: 0 }   # output alloc: scalar (Mean/Sum) or per-row F32 (None) — upper bound n_rows*4

precision:
  bit_stable_on_same_hardware: true   # deterministic reduction order; stable log-sum-exp
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "FUSED_SOFTMAX_CROSS_ENTROPY_CPU_PRECISION: stable log-sum-exp accumulated in F64, narrowed to F32. ignore_index masked in forward kernel. Deterministic on same hardware."

determinism: same_hardware_bitwise
```

---

## inplace_affine  (in-place affine `x = mul·x + add`, mutating input 0)

One-line: In-place affine transform x = mul*x + add over float dtypes; output ALIASES input 0 (destructive); contiguous row-major.

`FusedOps::INPLACE_AFFINE` (id 16; `fuel-graph/src/registry/inplace_affine.rs:23`). Forward-family
fused op that applies an affine transform **in place**, mutating its single input: `x = mul·x + add`,
elementwise over every element. One input (the mutated tensor), over the four float dtypes
`{F32, F64, BF16, F16}` (half floats widen to f32 for the multiply-add then narrow on store; native
f32/f64 otherwise — `INPLACE_AFFINE_CPU_PRECISION`). Params `FusedOpParams::InplaceAffine { mul: f64,
add: f64 }`. The output **aliases input 0 by contract** — the registry marks input 0 destructive via
`Op::destructive_input`, and `derive_ordering` pins this op after all non-destructive readers of that
buffer (so the in-place mutation cannot clobber a value another consumer still needs). It is therefore
**in-place / not a fresh buffer** (`caps.in_place: true`, `aliasing: in_place(x)`). Shape and dtype
are passthrough (= input 0). Backward is `NotDifferentiable` (autograd integration is Phase 4);
`decompose` **panics** (there is no non-destructive form to lower to). Pattern: stub `None`.
Limitation: contiguous, zero-offset, row-major only.

Dispatch key: `(InplaceAffine, [x, out] dtypes, Cpu, "portable-cpu")` — out aliases x, same dtype.

FLOPs/bandwidth hint: elementwise, **bandwidth-bound** — `flops ≈ 2 * n` (one mul + one add per
element, `n` = element count) and, because the buffer is read-modified-written in place,
`bytes_moved ≈ 2 * n * dtype_bytes` (read x, write x; no separate output buffer). Marked
`judge_measured` (the elementwise overhead/bandwidth coefficient is measured, not fabricated; the
formula above is the derivable shape).

```fkc
kernel: inplace_affine
fused_op: INPLACE_AFFINE
blurb: "In-place affine transform x = mul*x + add over float dtypes; output ALIASES input 0 (destructive); contiguous row-major."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::inplace_affine_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # elementwise over all elements
  op_params:
    variant: InplaceAffine            # FusedOpParams::InplaceAffine (fused namespace; §3.7)
    fields:
      mul: { kind: f64 }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)
      layout_guarantee: same_as(x)    # writes back into x's buffer in place
      aliasing: in_place(x)           # output IS input 0's buffer (destructive); requires caps.in_place (§5.4)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: true                      # writes output into input 0 (destructive on index 0; §4.6)
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # elementwise bandwidth + launch overhead measured; FLOP/byte shape below is derivable
  class: cheap_elementwise
  # Derivable shape (the Judge refines coefficients): elementwise, bandwidth-bound, in-place (no separate out buffer).
  flops: "2 * n"                      # one mul + one add per element
  bytes_moved: "2 * n * dtype_bytes"  # read x, write x in place
  overhead_ns: ~                      # launch overhead is a non-derivable absolute; Judge measures it
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no separate output alloc

precision:
  bit_stable_on_same_hardware: true   # deterministic elementwise; f32/f64 native, half widens to f32
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "INPLACE_AFFINE_CPU_PRECISION: native f32/f64 mul-add; bf16/f16 widen to f32 then narrow on store. Deterministic on same hardware."

determinism: same_hardware_bitwise
```

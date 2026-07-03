---
fkc_version: 1
provider:
  name: fuel-dispatch
  backend: Cpu                       # family default; per-kernel overrides for CU/VK variants
  kernel_source: "portable-cpu"      # family default; per-kernel overrides per backend
  link_registry: fuel_dispatch::fkc::ENTRY_POINTS   # §12.6 symbol → KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-dispatch — matmul family kernel contracts

Contracts for the `matmul` family of the `fuel-dispatch` crate: dense floating-point GEMM
(`MatMul`), integer GEMM (`MatMul` over I8/U8 keys), Vulkan mixed-precision / tensor-core GEMM
(`MatMul` over mixed-dtype keys), GGUF block-quant matmul (`QMatMul`), bitsandbytes NF4 matmul
(`Nf4Matmul`), and the fused linear `lhs @ rhs + bias` (`FusedLinear`).

All facts below are drawn from `docs/kernel-contracts/_inventory/dispatch.md` and the cited
binding-table registration sites. Every contract here is a **primitive `op_kind`** contract:
its cost compiles to the primitive `CostFn = fn(&[Shape], &[DType], &OpParams,
&BackendCapabilities) -> CostEstimate` (§4.4 / §12.3), and its `dtype_rule` / `shape_rule` are
**checked against the binding key** (no graph-side fn is registered — §5.1 / §12.7). Three of the
six op names (`MatMul`, `MatMulInteger`, `MatMulMixedPrecision`) share the **same `OpKind::MatMul`**
and are distinct only by their dispatch-key dtype slots, exactly the "split a row only when there
are genuinely different kernels at one `(op, dtypes)` key" rule (inventory §"How to read"). They
are authored as separate `## ` sections because their dtypes, numerics, and capabilities differ.

Family-wide as-built facts (inventory §"How to read" / §"Cross-cutting"): every CPU wrapper takes
`_layouts: &[Layout]` **unused** and operates on contiguous, zero-offset byte buffers — so every
matmul kernel here is **contiguous-only** and **not offset-capable**; the executor's auto-Contiguize
pass materializes every input dense before the wrapper runs (`compiled.rs` caps gate). Output
Storage is **always pre-allocated by the executor**; no kernel allocates (`layout_guarantee`
includes `preallocated`, §5.3). No matmul kernel is in-place. Cost is marked **`judge_measured`**
throughout — the Judge bootstraps it (§4.4); the `flops` / `bytes_moved` strings below are honest,
op-derivable formula hints (GEMM = `2*M*N*K`), not fabricated calibrated numbers, and they seed the
Judge as priors.

## matmul  (dense floating-point GEMM, batched / GQA-broadcast)

Dense floating-point batched matrix multiply `out[..batch.., m, n] = lhs[..batch.., m, k] @
rhs[..batch.., k, n]`. Batch dims either match elementwise OR are GQA-divisible
(`lhs_dim % rhs_dim == 0`, the broadcast-the-fewer-heads case). f32/f64 accumulate natively;
bf16/f16 accumulate in f32 and narrow to the input dtype on store. Three backends ship this same
`(MatMul, [T,T,T], backend)` key: CPU (`matmul_*_cpu_wrapper`, `_layouts` unused), baracuda CUDA
(`gemm_dense`, "packed row-major contract", no strided caps), Vulkan (tiled / vec4 f32 kernels that
require contiguous row-major). All three are **contiguous-only**: a transposed or sliced operand is
auto-Contiguized by the planner first, whose cost the planner sums from the `Op::Contiguize`
contract (§4.3 / §4.4). Vulkan ships f32 only here; the mixed-precision tensor-core combos are a
separate kernel section below.

```fkc
kernel: matmul
op_kind: MatMul
blurb: "Dense batched float GEMM out=lhs@rhs; batch match or GQA-divisible; half accumulates in f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::matmul_f32_cpu_wrapper"   # one per (backend,dtype); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"                   # [..batch.., m, k]
      shape_constraint: "last_dim_eq=rhs"     # k = lhs.dim[-1] == rhs.dim[-2]
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"                   # [..batch.., k, n]
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul                   # OpParams::Matmul
    fields:
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "match lhs_batch_dims OR GQA-divisible (lhs_dim % rhs_dim == 0)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # key pins T; checked against key (§5.1)
      shape_rule: matmul(lhs, rhs)        # [..lhs_batch.., m, n]
      layout_guarantee: contiguous        # also preallocated (executor allocs)
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured            # Judge bootstraps cost (§4.4); formulas below are op-derived priors
  class: gemm_like
  flops: "2 * batch * m * n * k"        # GEMM 2*M*N*K, summed over batch
  bytes_moved: "(batch*m*k + batch*k*n + batch*m*n) * dtype_bytes"   # read lhs+rhs, write out
  overhead_ns: ~                        # launch cost is judge_measured (no authored absolute under judge_measured)
  memory: { device_bytes: "batch * m * n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # CPU deterministic FMA loop (PRIMITIVE_DETERMINISTIC_CPU)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU bit-stable per hardware; f32/f64 native, bf16/f16 accumulate in f32 then narrow on store. Vulkan sibling carries VULKAN_MATMUL_PRECISION (deterministic FMA, separate contract)."

determinism: same_hardware_bitwise
```

> **Sibling alternatives at this key.** baracuda CUDA (`kernel_source: "baracuda"`, `entry_point:
> baracuda::gemm_dense_*`, f32/f64/f16/bf16) and Vulkan (`kernel_source: "vulkan-slang"`,
> `entry_point: fuel_vulkan_backend::matmul_f32`, f32 only) register the same `(MatMul, [T,T,T])`
> key as distinct `KernelRef`s — legal sibling `BindingEntry`s the route picker ranks (§12.5).
> The Vulkan f32 sibling's precision is `VULKAN_MATMUL_PRECISION` (`audited: true`, deterministic
> FMA, `bit_stable_on_same_hardware: true`); it is contiguous-only like the CPU kernel.

## matmul_integer  (8-bit integer GEMM, i32 accumulate, saturating cast)

Integer batched matrix multiply with an i32 accumulator and a **saturating cast back to the input
dtype** on store. `OpKind::MatMul` over the integer key `[I8,I8,I8]` (signed) or `[U8,U8,U8]`
(unsigned) — a kernel genuinely distinct from the float matmul at the same `OpKind`, so it is its
own contract. Mirrors baracuda `gemm_{s8,u8}_rrr_sm80_run`. CPU (`matmul_i8/u8_cpu_wrapper`) and
baracuda CUDA both ship it; both are contiguous-only (packed row-major).

```fkc
kernel: matmul_integer
op_kind: MatMul
blurb: "8-bit integer batched GEMM; i32 accumulate; saturating cast back to I8/U8 on store."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::matmul_i8_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [I8, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"
      shape_constraint: "last_dim_eq=rhs"
    - name: rhs
      dtypes: [I8, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"
      shape_constraint: "dim[-2]=k"
  op_params:
    variant: Matmul                   # OpParams::Matmul (m,n,k + batch dims)
    fields:
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # I8→I8 / U8→U8 (i32 accumulator narrowed by saturating cast)
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
  flops: "2 * batch * m * n * k"        # MAC count identical to float GEMM
  bytes_moved: "(batch*m*k + batch*k*n + batch*m*n) * dtype_bytes"   # dtype_bytes == 1 for I8/U8
  overhead_ns: ~                        # launch cost is judge_measured (no authored absolute under judge_measured)
  memory: { device_bytes: "batch * m * n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # exact integer arithmetic; deterministic
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Exact i32 accumulation; SATURATING cast back to I8/U8 on store (clamps to dtype range, not wrap). Bit-stable per hardware."

determinism: bitwise
```

## matmul_mixed_precision  (Vulkan tensor-core / cooperative-matrix GEMM — describe-only chassis)

Vulkan-only mixed-precision tensor-core matmul: a **family of distinct kernels**, one per
input/output dtype combination, each with its own dispatch key under `OpKind::MatMul`. The five
as-built combos (inventory §"MatMul mixed-precision"): `[f32,bf16→f32]`, `[bf16,bf16→f32]`,
`[bf16,bf16→bf16]`, `[f16,f16→f16]`, `[f16,f16→f32]`. All use an **f32 accumulator**; the
`→bf16` / `→f16` variants downcast on store. Cooperative-matrix variants require canonical
row-major tiles with `M%16==0`, `N%16==0`, `K>=16`; the route picker falls back to a cast +
f32-matmul candidate on shapes that miss the tile constraint (a separate route, not this kernel's
concern).

**This section is now `registrable: false` (§3.10 describe-only chassis).** It enumerates the
mixed-precision combo family for documentation, but it is NOT a dispatch target: its `accept` inputs
vary over **different** dtype lists per operand (lhs `[F32,BF16,F16]` vs rhs `[BF16,F16]`), which the
uniform multi-dtype fan-out importer cannot key (`FanoutDtypeMismatch`, §3.4 — a legal-but-not-fannable
multi-axis contract, never a silent pick). The registrable per-combo bindings — each single-dtype per
operand, one `entry_point` symbol per production wrapper — live in
**`docs/kernel-contracts/vulkan/matmul.fkc.md`** (`matmul_f32_bf16_b`, `matmul_bf16_bf16_f32`,
`matmul_bf16_bf16_bf16`, `matmul_f16_f16_f16`, `matmul_f16_f16_f32`), which is the file
`register_vulkan_matmul_from_contract` imports into the binding table. Keeping this describe-only
chassis clears the `FanoutDtypeMismatch` corpus-lint deferral for this section (a describe-only
section is excluded from lowering, §3.10) without discarding the combo-family overview. Precision is
the corrected `VULKAN_MATMUL_TENSORCORE_PRECISION` posture (audited `none(reason)`; the tensor-core
FADD/subgroup order is scheduler-dependent, so `bit_stable_on_same_hardware: false`).

```fkc
kernel: matmul_mixed_precision
registrable: false                # §3.10 describe-only: multi-axis combo-family chassis (lhs/rhs enumerate different dtype lists — FanoutDtypeMismatch, §3.4); the registrable per-combo bindings live in vulkan/matmul.fkc.md
op_kind: MatMul
blurb: "Vulkan tensor-core mixed-precision GEMM chassis; per-combo bindings in vulkan/matmul.fkc.md; f32 accumulate; bf16/f16 in, f32/half out."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::matmul_bf16_bf16_f32"   # one symbol per combo; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, BF16, F16]        # combo-specific; key pins the exact pair (e.g. F32 only for f32,bf16→f32)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"
      shape_constraint: "last_dim_eq=rhs"
      device: Vulkan
      substrate: VulkanBuffer
    - name: rhs
      dtypes: [BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"
      shape_constraint: "dim[-2]=k"
      device: Vulkan
      substrate: VulkanBuffer
  op_params:
    variant: Matmul                   # OpParams::Matmul
    fields:
      m: { kind: usize, note: "coop-matrix variants need m % 16 == 0" }
      n: { kind: usize, note: "coop-matrix variants need n % 16 == 0" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]", note: "coop-matrix variants need k >= 16" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # per combo: F32 for →f32 combos, else passthrough half — key pins the out slot
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # canonical row-major tiles only
  fast_paths:
    - { when: "dim[i] % 16 == 0", note: "m%16==0 && n%16==0 && k>=16: cooperative-matrix tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"        # MAC count; tensor-core throughput is a Judge measurement
  bytes_moved: "(batch*m*k * lhs_bytes + batch*k*n * rhs_bytes + batch*m*n * out_bytes)"
  overhead_ns: ~                        # Vulkan command-buffer submit overhead is judge_measured (placeholder, not an authored constant)
  memory: { device_bytes: "batch * m * n * out_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # VULKAN_MATMUL_TENSORCORE_PRECISION: deterministic FMA tiles
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "VULKAN_MATMUL_TENSORCORE_PRECISION: f32 accumulate; bf16/f16 INPUTS lose mantissa (wider ULP than f32 matmul). →bf16/→f16 variants downcast on store. Deterministic per hardware."

determinism: same_hardware_bitwise
```

> **Per-combo `KernelRef`s (registrable — in `vulkan/matmul.fkc.md`).** Each combo is a distinct
> `entry_point` resolving to a distinct production wrapper `KernelRef` at a distinct
> `(MatMul, [lhs,rhs,out], Vulkan)` key. Those single-dtype-per-operand registrable sections live in
> `docs/kernel-contracts/vulkan/matmul.fkc.md` (the file `register_vulkan_matmul_from_contract`
> imports); the `dtype_rule`/out-slot of each is pinned by its own key. This describe-only chassis
> documents the shared algorithm and the cooperative-matrix tile constraint that gates the fast path.

## qmatmul  (GGUF block-quant matmul, dequantize-on-the-fly)

Quantized GGUF matmul: `out[batch,m,n] = A[batch,m,k] @ dequant(W)[n,k]`, F32 activations against
block-quantized weight bytes, F32 output. CPU-only (inventory). The as-built buffer is U32-typed
(4-byte word access) and the wrapper validates the weight operand's dtype `== U32`, but the FDX base
dtype is the honest **U8** packed-byte stand-in (FDX §3 — FDX never labels an opaque packed quant
block stream as a wide int); the 32-bit access width rides `caps.access_granularity_bits` and the
weight-operand note. The `OpParams::QMatMul.quant_type` selects the per-format typed kernel. The block scales are **baked into the GGML block layout (INLINE)**
— there is no separate scale operand here, so the scale single-place rule (§3.9.3) resolves to the
sidecar-bundled / inline case and `fdx.quant.scale_operand` stays `~`. `quant_type` ∈
{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4_K_M, Q5K, Q6K} as the GGUF file-format names; in
FKC the weight operand names the as-built `GgmlDType` **variant** matched by numeric code (e.g.
`Q4K`, never `Q4_K_M` — §3.4). Contiguous-only.

```fkc
kernel: qmatmul
op_kind: QMatMul
blurb: "GGUF block-quant matmul: F32 activations @ dequant(block-quant W); inline block scales; F32 out."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::qmatmul_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=3"                   # [batch, m, k]
      shape_constraint: "last_dim_eq=weight"   # k = activations.dim[-1] == weight k-extent
    - name: weight
      dtypes: [U8]                    # FDX honesty stand-in: opaque packed GGML block byte stream (FDX §3). The as-built buffer is U32-typed and the wrapper validates dtype==U32 (inventory) — that 32-bit ACCESS width is carried in access_granularity_bits + this note, NOT in the base dtype (FDX never labels a packed quant block stream as a wide int).
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # opaque block-packed byte buffer; logical [n, k] via quant_type
      fdx:
        requires_ext: true            # the U8 base is meaning-bearing: it IS block-quant weight (MEANING_REQUIRES_EXT)
        quant:
          family: GGML_BLOCK          # FDXQuant.family (FDX §6.2)
          ggml_dtype: Q4K             # GgmlDType VARIANT NAME, matched by code; quant_type selects per-format kernel; the per-block grain rides ggml_dtype (no granularity under GGML_BLOCK; FDX §6.2 / FKC §10.6)
          role: weight
          scale_operand: ~            # INLINE baked block scales (single-place rule: no separate scale operand; §3.9.3)
  op_params:
    variant: QMatMul                  # OpParams::QMatMul
    fields:
      quant_type:   { kind: GgmlDType, note: "selects per-format typed kernel: Q4_0..Q6K incl. Q4_K_M(GGUF name) → Q4K" }
      batch_count:  { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== activations.dim[-1]; k % block(quant_type) == 0" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # always F32 (key out slot is F32)
      shape_rule: matmul(activations, weight)   # [batch, m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32           # the as-built buffer is U32-typed (4-byte word access); the base FDX dtype is the honest U8 packed-byte stand-in

cost:
  provenance: judge_measured            # per-format dequant+GEMM throughput is a Judge measurement
  class: gemm_like
  flops: "2 * batch_count * m * n * k"  # GEMM MACs (dequant cost folded into the kernel's bytes_moved)
  bytes_moved: "(batch_count*m*k*4 + n*k*block_bytes(quant_type) + batch_count*m*n*4)"   # read F32 act + packed W + write F32 out
  overhead_ns: ~                        # launch cost is judge_measured (no authored absolute under judge_measured)
  memory: { device_bytes: 0, host_bytes: "batch_count * m * n * 4", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # CPU deterministic; quantization error is in the WEIGHTS, not the kernel
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "F32 accumulate; on-the-fly block dequant per quant_type. Kernel is bit-stable per hardware; the lossy step is the upstream weight quantization, not this matmul. Scale applied ONCE during dequant (inline block scales)."

determinism: same_hardware_bitwise
```

## nf4_matmul  (bitsandbytes NF4 matmul, separate absmax scale operand)

bitsandbytes NF4 4-bit matmul. Three inputs: activations `T`, NF4-packed weight bytes `U8`, and a
**separate `absmax` F32 scale operand** — and one output `T`, where `T ∈ {F32, F16, BF16}`. The
as-built keys are `[F32,U8,F32,F32]`, `[F16,U8,F32,F16]`, `[BF16,U8,F32,BF16]` (inventory). The weight
is **block-grained affine** — low-bit NF4 data plus a separate block-shaped absmax scale — so its
`fdx.quant.family` is the FDX **`AFFINE_BLOCK`** family (FDX §6.2 code 4, the nf4/QLoRA regime),
**not** `AFFINE_FLOAT` (which is dynamic per-tensor/token/channel) and **not** `PerBlock` granularity
(that code stays MX-only; the block grain rides the FDX sidecar `block_shape`, §13.5a). Because the
absmax scale is passed as its **own graph input**, the scale single-place rule (§3.9.3) requires it to
be an FKC `accept.inputs` operand named in the weight operand's `fdx.quant.scale_operand` — and NOT
also an FDX `scale_buffer` (`ScaleDoubleDeclared` otherwise). `k` must be even and `k % block_size ==
0`. CPU-only. Contiguous-only.

```fkc
kernel: nf4_matmul
op_kind: Nf4Matmul
blurb: "bitsandbytes NF4 4-bit matmul; packed-U8 weight + separate absmax F32 scale; out in activation dtype."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::nf4_matmul_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: activations
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=3"                   # [batch, m, k]
      shape_constraint: "last_dim_eq=w_packed"
    - name: w_packed
      dtypes: [U8]                    # NF4 4-bit packed weight bytes
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # opaque packed [n, k] (2 NF4 codes per byte)
      fdx:
        requires_ext: true            # U8 base is meaning-bearing: it IS NF4-packed weight
        sub_byte: F4                  # 4-bit logical element under the opaque U8 (FDX §6.1)
        quant:
          family: AFFINE_BLOCK        # bitsandbytes NF4: block-grained affine, separate absmax block scale (FDX §6.2 code 4)
          granularity: ~              # NOT PerBlock (that code stays MX-only); AFFINE_BLOCK's grain rides the FDX sidecar block_shape (FDX §6.2 / §13.5a)
          role: weight
          scale_operand: absmax       # ← SEPARATE GRAPH INPUT (single-place rule; §3.9.3) — NOT an FDX scale_buffer
    - name: absmax
      dtypes: [F32]                   # per-block scale; SEPARATE operand (named by w_packed.fdx.quant.scale_operand)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any                       # [n * k / block_size] block scales
  op_params:
    variant: Nf4Matmul                # OpParams::Nf4Matmul
    fields:
      batch: { kind: usize }
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "k even AND k % block_size == 0" }
      block_size: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(activations)   # out dtype == activations T (key out slot)
      shape_rule: matmul(activations, w_packed)   # [batch, m, n]
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
  flops: "2 * batch * m * n * k"        # GEMM MACs; NF4 dequant folded into bytes_moved
  bytes_moved: "(batch*m*k * dtype_bytes + n*k/2 + n*k/block_size*4 + batch*m*n * dtype_bytes)"   # act + packed-U8 (2 codes/byte) + absmax + out
  overhead_ns: ~                        # launch cost is judge_measured (no authored absolute under judge_measured)
  memory: { device_bytes: 0, host_bytes: "batch * m * n * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # CPU deterministic; quantization loss is in the WEIGHTS, not the kernel
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Activation-dtype accumulate; NF4 codebook dequant scaled by per-block absmax (applied ONCE). Bit-stable per hardware; lossy step is upstream NF4 quantization, not this matmul."

determinism: same_hardware_bitwise
```

## fused_linear  (lhs @ rhs + bias, single pass)

Fused linear layer `out = lhs @ rhs + bias`, computed in one pass over the GEMM accumulator (bias
added on store). Four operands: `lhs`, `rhs`, `bias`, `out`, all the same dtype `T ∈ {F32, F64,
BF16, F16}` (key `[T,T,T,T]`). CPU-only in the binding table (also present in the fused registry,
but **this is the primitive `op_kind: FusedLinear` binding-table entry**, not the `fused_op`
registration). Matmul-shaped params consumed in the wrapper. Contiguous-only; bf16/f16 accumulate
in f32 and narrow on store.

```fkc
kernel: fused_linear
op_kind: FusedLinear
blurb: "Fused linear out = lhs @ rhs + bias in one pass; same dtype throughout; half accumulates in f32."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::fused_linear_f32_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"                   # [..batch.., m, k]
      shape_constraint: "last_dim_eq=rhs"
    - name: rhs
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "2..=6"                   # [..batch.., k, n]
      shape_constraint: "dim[-2]=k"
    - name: bias
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1..=2"                   # [n] (broadcast across m/batch by the wrapper)
      shape_constraint: "last_dim_eq=out"
  op_params:
    variant: Matmul                   # OpParams::Matmul shape (m,n,k + batch dims); bias added on store
    fields:
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # all operands same T; key out slot is T
      shape_rule: matmul(lhs, rhs)        # [..lhs_batch.., m, n]
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
  flops: "2 * batch * m * n * k + batch * m * n"   # GEMM MACs + the fused bias add
  bytes_moved: "(batch*m*k + batch*k*n + n + batch*m*n) * dtype_bytes"   # read lhs+rhs+bias, write out
  overhead_ns: ~                        # launch cost is judge_measured (no authored absolute under judge_measured)
  memory: { device_bytes: "batch * m * n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # CPU deterministic FMA + add (PRIMITIVE_DETERMINISTIC_CPU)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Bit-stable per hardware; f32/f64 native, bf16/f16 accumulate in f32 then narrow on store. Bias added in the accumulator's wide dtype before the store narrow."

determinism: same_hardware_bitwise
```

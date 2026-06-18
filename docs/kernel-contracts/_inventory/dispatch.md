# fuel-dispatch kernel inventory

Crate: `fuel-dispatch`. This inventory enumerates every distinct kernel/op that
Fuel itself provides through the dispatch layer — the `KernelBindingTable`
registrations, the `KernelRef` wrapper functions, and the `KernelCaps` each
binding expresses.

Sources of truth read:
- `fuel-dispatch/src/kernel.rs` — `KernelRef`, `OpParams`, `KernelCaps`,
  `KernelBindingTable`, `BindingEntry`.
- `fuel-dispatch/src/dispatch.rs` — CPU + (PTX) CUDA wrappers and
  `register_cpu_kernels` / `register_cuda_kernels`.
- `fuel-dispatch/src/baracuda_dispatch.rs` — baracuda CUDA wrappers and
  `register_baracuda_cuda_kernels`.
- `fuel-dispatch/src/vulkan_dispatch.rs` — Vulkan wrappers and
  `register_vulkan_kernels`.
- `fuel-dispatch/src/compiled.rs` — `CompiledNode` / `execute_compiled` (the
  caps → contiguize-gate consumer).

## How to read this inventory

- Each row is **one distinct op (OpKind)** at the dispatch layer.
  Dtype-monomorphized variants (f32/f64/bf16/f16/…) are collapsed into the
  dtype list of a single entry, per the "one entry per kernel" rule. A row is
  split only when there are genuinely different kernels at one `(op, dtypes)`
  key (e.g. Vulkan tensor-core matmul variants by mixed-precision combo).
- **input_layouts** is the *expressed* contract via `KernelCaps`, NOT a guess:
  - `contiguous-only` — registered with default (all-false) caps. The
    executor's auto-Contiguize pass materializes every input contiguous before
    the wrapper runs; the wrapper ignores the `layouts` side-channel
    (`_layouts` in the CPU wrappers). Non-zero `start_offset` inputs *always*
    auto-Contiguize.
  - `strided` — registered `register_with_caps(..., KernelCaps::strided_input())`.
    The wrapper consumes `layouts[..]` and walks input strides (incl. stride-0
    broadcast axes). Per `KernelCaps` doc + `compiled.rs:58`, inputs with
    non-zero `start_offset` STILL go through auto-Contiguize even for
    strided-capable kernels (offset-slicing the device buffer is a separate
    concern). So "strided" here = strided + broadcast capable, NOT offset-capable.
- **Universal layout facts (apply to every CPU wrapper):** all CPU wrappers
  take `_layouts: &[Layout]` UNUSED and operate on raw byte buffers
  (`CpuStorageBytes`); they rely entirely on auto-Contiguize. They are
  therefore `contiguous-only` and not offset-capable. Geometry comes from
  `OpParams`; dtype-agnostic byte kernels read element size from the output
  Storage's `dtype` tag.
- **output_behavior**: output Storage is ALWAYS pre-allocated by the executor;
  no kernel allocates. The wrapper writes into the pre-allocated bytes. Output
  dtype = last entry of the `dtypes` key unless noted.
- **source** points at the binding-table registration site (most load-bearing
  for the contract). Wrapper bodies are cross-referenced in notes.

## Backend coverage summary

- **CPU** (`register_cpu_kernels`, dispatch.rs:3880): the always-built
  universal fallback. ~all ops. Contiguous-only; precision bulk-upgraded to
  `PRIMITIVE_DETERMINISTIC_CPU` (bit-stable per hardware) via
  `fill_unset_cpu_precision`. half-precision (bf16/f16) accumulates in f32.
- **CUDA / baracuda** (`register_baracuda_cuda_kernels`, baracuda_dispatch.rs:2353):
  the single CUDA kernel home post-alpha.67. Elementwise/reduce/norm/softmax/
  rope/indexing/triangular/flip/roll/cumsum/affine/clamp/powi/cast/conv1d/
  reduce-to/int-gemm/dense-gemm. Most register `strided_input` (baracuda FFI
  is stride-driven); MatMul/Cast/IndexSelect/Gather/MaskedFill/ScatterAdd/
  Pad/WriteSlice and the in-place families register contiguous-only.
- **CUDA / PTX** (`register_cuda_kernels`, dispatch.rs:4777): now ONLY
  `Op::Copy` D2H + D2D (every compute kernel migrated to baracuda).
- **Vulkan** (`register_vulkan_kernels`, vulkan_dispatch.rs): elementwise
  (strided), matmul (contiguous-only, incl. mixed-precision tensor-core
  variants), softmax/norm/reduce/argreduce (contiguous-only), indexing/gather/
  masked-fill/scatter/index-add (byte-level), rope/affine/clamp/powi/concat/
  flip/roll/cumsum (strided), triangular/write-slice/cast (byte-level), Copy
  D2H. Vulkan reductions/softmax/norm carry `PrecisionGuarantee::none` (not
  bit-stable: float atomics / non-deterministic accumulation order).

---

## Op inventory

Legend for input_layouts column: `C` = contiguous-only, `S` = strided
(stride/broadcast capable, NOT offset-capable). Backend tags: CPU / CU
(baracuda CUDA) / VK (Vulkan).

### Elementwise binary (Add, Sub, Mul, Div)
- **OpKind**: `AddElementwise`, `SubElementwise`, `MulElementwise`, `DivElementwise`
- **dtypes**: CPU/CU/VK f32, f64, bf16, f16 (key `[T, T, T]`)
- **input_layouts**: CPU **C** (`cpu_binary_wrapper`, `_layouts` unused,
  dispatch.rs:273). CU **S** (`register_with_caps(... strided)`,
  baracuda_dispatch.rs:2372). VK **S** (binary.slang stride-aware,
  vulkan_dispatch.rs:4293).
- **op_params**: `OpParams::None`.
- **output_behavior**: same dtype as inputs; shape = broadcast of inputs
  (broadcast realized by auto-Contiguize on CPU/VK-no-strided; by stride-0 on
  CU/VK strided). Not in-place.
- **precision**: CPU bit-stable; VK pointwise.
- **source**: dispatch.rs:3913 (CPU), baracuda_dispatch.rs:2372 (CU),
  vulkan_dispatch.rs:4293 (VK).

### Elementwise binary — Maximum, Minimum
- **OpKind**: `MaximumElementwise`, `MinimumElementwise`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16; VK f32, f16, bf16, f64
- **input_layouts**: CPU **C**; CU **S**; VK **S**.
- **op_params**: `None`. **output**: same dtype, broadcast shape.
- **source**: dispatch.rs:4264/4481/4503 (CPU), baracuda_dispatch.rs:2376 (CU),
  vulkan_dispatch.rs:4297 (VK).

### Elementwise binary — Pow, Rem
- **OpKind**: `PowElementwise` (tensor^tensor), `RemElementwise` (PyTorch-convention remainder, sign of divisor)
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. (no VK)
- **input_layouts**: CPU **C**; CU **S**.
- **op_params**: `None`. **output**: same dtype, broadcast shape.
- **notes**: CU `Rem` binds baracuda `binary_mod_*` (Python-style), not C99 fmod.
- **source**: dispatch.rs:4340/4350 (CPU), baracuda_dispatch.rs:2381 (CU).

### Elementwise unary — pointwise (Relu, Neg, Sqr, Sqrt, Recip, Abs, Step, Sign)
- **OpKind**: `ReluElementwise`, `NegElementwise`, `SqrElementwise`,
  `SqrtElementwise`, `RecipElementwise`, `AbsElementwise`, `StepElementwise`,
  `SignElementwise`
- **dtypes**: CPU f32, f64, bf16, f16 (Sign also f64/bf16/f16); CU f32, f64, bf16, f16; VK f32, f16, f64 (Step/Relu/Sqr/Sqrt/Neg/Abs/Sign/Recip)
- **input_layouts**: CPU **C** (`cpu_unary_wrapper`, dispatch.rs:310); CU **S**; VK **S**.
- **op_params**: `None`. **output**: same dtype, same shape. Not in-place.
- **precision**: VK float-pointwise; CPU bit-stable.
- **source**: dispatch.rs:3918 (CPU), baracuda_dispatch.rs:2783 (CU),
  vulkan_dispatch.rs:4305 (VK).

### Elementwise unary — transcendental (Tanh, Exp, Log, Sin, Cos, Sigmoid, Silu, Gelu)
- **OpKind**: `TanhElementwise`, `ExpElementwise`, `LogElementwise`,
  `SinElementwise`, `CosElementwise`, `SigmoidElementwise`, `SiluElementwise`,
  `GeluElementwise`
- **dtypes**: CPU/CU f32, f64, bf16, f16; VK f32, f16, f64.
- **input_layouts**: CPU **C**; CU **S**; VK **S**.
- **op_params**: `None`. **output**: same dtype, same shape.
- **precision**: VK `VULKAN_TRANSCENDENTAL_PRECISION` (wider ULP); CPU bit-stable.
- **notes**: `GeluElementwise` is the **tanh** approximation. CU binds baracuda
  `unary_gelu_tanh_*`; the erf-flavored `unary_gelu_*` is registered under
  `GeluErfElementwise` (the two were conflated until the 2026-06-10 sweep).
- **source**: dispatch.rs:3924 (CPU), baracuda_dispatch.rs:2789 (CU),
  vulkan_dispatch.rs:4311 (VK).

### Elementwise unary — rounding/special (Floor, Ceil, Round, Erf, GeluErf, Rsqrt)
- **OpKind**: `FloorElementwise`, `CeilElementwise`, `RoundElementwise`,
  `ErfElementwise`, `GeluErfElementwise`, `RsqrtElementwise`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. (no VK on these)
- **input_layouts**: CPU **C**; CU **S**.
- **op_params**: `None`. **output**: same dtype, same shape.
- **notes**: `Round` = banker's rounding both sides (CPU `round_ties_even`, CU
  `rint`). `Erf` = plain error function (`erff`). `GeluErf` = erf-flavored gelu.
- **source**: dispatch.rs:4310/4330/4335/4345 (CPU), baracuda_dispatch.rs:2813 (CU).

### Clamp (elementwise, scalar bounds)
- **OpKind**: `ClampElementwise`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16; VK f32.
- **input_layouts**: CPU **C**; CU **S** (bounds broadcast via stride-0); VK **S**.
- **op_params**: `OpParams::Clamp { min: f64, max: f64 }`.
- **output**: same dtype, same shape.
- **source**: dispatch.rs:4262/4604 (CPU), baracuda_dispatch.rs:2548 (CU),
  vulkan_dispatch.rs:4664 (VK).

### PowI (elementwise integer power)
- **OpKind**: `PowIElementwise`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16; VK f32.
- **input_layouts**: CPU **C**; CU **S**; VK **S**.
- **op_params**: `OpParams::PowI { exp: i32 }`.
- **output**: same dtype, same shape.
- **source**: dispatch.rs:4263/4607 (CPU), baracuda_dispatch.rs:2528 (CU),
  vulkan_dispatch.rs:4669 (VK).

### PowI backward
- **OpKind**: `PowIElementwiseBackward`
- **dtypes**: CPU/CU f32, f64, bf16, f16 (key `[T, T, T]` = `(x, upstream, grad_x)`)
- **input_layouts**: CPU **C**; CU **S**.
- **op_params**: `OpParams::PowI { exp }`. **output**: `grad_x` same dtype/shape as x.
- **notes**: single-launch alternative to autograd's 3-node decomposition.
- **source**: dispatch.rs:4614 (CPU), baracuda_dispatch.rs:2539 (CU).

### Rem note — see "Elementwise binary — Pow, Rem".

### Comparison family (Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual)
- **OpKind**: `EqualElementwise`, `NotEqualElementwise`, `LessElementwise`,
  `LessEqualElementwise`, `GreaterElementwise`, `GreaterEqualElementwise`
- **dtypes**: CPU f32, f64, bf16, f16 — key `[T, T, U8]`.
- **input_layouts**: CPU **C** (`cpu_binary_wrapper`).
- **op_params**: `None`.
- **output_behavior**: **output dtype = U8** (1 byte/element) regardless of
  input T; `1` where predicate holds else `0`. Shape = broadcast of inputs.
- **source**: dispatch.rs:4271 (CPU).

### Where (ternary select)
- **OpKind**: `Where`
- **dtypes**: CPU f32, f64, bf16, f16 — key `[U8, T, T, T]` = `(cond, a, b, out)`.
- **input_layouts**: CPU **C** (`cpu_where_wrapper`, dispatch.rs:483; validates cond dtype == U8).
- **op_params**: `None`. **output**: same dtype T as a/b, broadcast shape.
- **source**: dispatch.rs:4304 (CPU).

### MatMul (dense floating-point)
- **OpKind**: `MatMul`
- **dtypes**: CPU f32, f64, bf16, f16 (key `[T, T, T]`); CU f32, f64, f16, bf16;
  VK f32 (`[f32,f32,f32]`).
- **input_layouts**: CPU **C** (`matmul_*_cpu_wrapper`, `_layouts` unused,
  dispatch.rs:3625); CU **C** (gemm_dense "packed row-major contract", no strided
  caps — baracuda_dispatch.rs:2892); VK **C** (tiled/vec4 kernels require
  contiguous row-major, vulkan_dispatch.rs:4609).
- **op_params**: `OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k }`.
  Batch dims match OR GQA-divisible (`lhs_dim % rhs_dim == 0`).
- **output_behavior**: out `[..lhs_batch.., m, n]`, dtype = T. Pre-allocated.
- **precision**: CPU bit-stable; VK `VULKAN_MATMUL_PRECISION` (deterministic FMA).
- **source**: dispatch.rs:3965 (CPU), baracuda_dispatch.rs:2892 (CU),
  vulkan_dispatch.rs:4609 (VK).

### MatMul (integer)
- **OpKind**: `MatMul` (integer key)
- **dtypes**: CPU i8 (`[I8,I8,I8]`), u8 (`[U8,U8,U8]`); CU i8, u8.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `OpParams::Matmul`.
- **output_behavior**: i32 accumulator, **saturating cast back to T** on store.
  Mirrors baracuda `gemm_{s8,u8}_rrr_sm80_run`.
- **source**: dispatch.rs:3971 (CPU), baracuda_dispatch.rs:2507 (CU).

### MatMul (mixed-precision / tensor-core) — Vulkan-only, distinct kernels per combo
- **OpKind**: `MatMul`
- **dtypes / distinct kernels** (each a SEPARATE kernel, different key):
  `[f32, bf16, f32]` (matmul_f32_bf16_b), `[bf16, bf16, f32]`
  (matmul_bf16_bf16_f32), `[bf16, bf16, bf16]` (matmul_bf16_bf16_bf16),
  `[f16, f16, f16]` (matmul_f16_f16_f16), `[f16, f16, f32]` (matmul_f16_f16_f32).
- **input_layouts**: **C** (tensor-core cooperative-matrix kernels require
  canonical row-major tiles; coop variants need M%16==0, N%16==0, K>=16 — the
  route picker falls back to cast+f32-matmul on small shapes).
- **op_params**: `OpParams::Matmul`.
- **output_behavior**: f32 accumulator; output dtype per the key's last entry
  (downcast store for the `→bf16`/`→f16` variants).
- **precision**: `VULKAN_MATMUL_TENSORCORE_PRECISION` (wider ULP — bf16/f16 inputs lose mantissa).
- **source**: vulkan_dispatch.rs:4617-4640.

### QMatMul (quantized GGUF matmul)
- **OpKind**: `QMatMul`
- **dtypes**: CPU only — key `[F32, U32, F32]` (activations F32, weight blocks U32, out F32).
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::QMatMul { quant_type, batch_count, m, n, k }`.
  `quant_type` ∈ {Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4_K_M, Q5K, Q6K}.
- **output_behavior**: `A[batch,m,k] @ dequant(W)[n,k] → out[batch,m,n]`, F32.
  Validates weight dtype == U32.
- **source**: dispatch.rs:4561 (registration), dispatch.rs:1608 (wrapper).

### Nf4Matmul (bitsandbytes NF4)
- **OpKind**: `Nf4Matmul`
- **dtypes**: CPU — `[F32,U8,F32,F32]`, `[F16,U8,F32,F16]`, `[BF16,U8,F32,BF16]`
  = `(activations T, w_packed U8, absmax F32, out T)`.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::Nf4Matmul { batch, m, n, k, block_size }`. k even,
  k % block_size == 0.
- **output_behavior**: out `[batch, m, n]`, dtype = activations T.
- **source**: dispatch.rs:4188 (CPU).

### FusedLinear (lhs @ rhs + bias)
- **OpKind**: `FusedLinear`
- **dtypes**: CPU f32, f64, bf16, f16 — key `[T, T, T, T]` = `(lhs, rhs, bias, out)`.
- **input_layouts**: CPU **C**.
- **op_params**: matmul-shaped (consumed in wrapper). **output**: T.
- **notes**: CPU-only in the binding table (also in the fused registry).
- **source**: dispatch.rs:4043 (CPU).

### Reductions — Sum, Max, Min, Mean
- **OpKind**: `SumReduce`, `MaxReduce`, `MinReduce`, `MeanReduce`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16; VK f32 (+ f16/bf16/f64 feature-gated).
- **input_layouts**: CPU **C** (`cpu_reduce_wrapper`); CU **S** (FFI passes
  `current_layout.stride()`); VK **C** (no caps; `PrecisionGuarantee::none`).
- **op_params**: `OpParams::Reduce { dims: Vec<usize>, keepdim: bool }`. Input
  Layout flows through `layouts[0]`.
- **output_behavior**: reduced dtype = T; bf16/f16 **accumulate in f32**.
  keepdim today always false.
- **precision**: CPU bit-stable; VK not bit-stable (atomics / non-det order).
- **source**: dispatch.rs:3956 (CPU), baracuda_dispatch.rs:2416 (CU),
  vulkan_dispatch.rs:4457 (VK).

### ReduceSumTo / ReduceMaxTo (broadcast-reverse reductions)
- **OpKind**: `ReduceSumTo`, `ReduceMaxTo`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16.
- **input_layouts**: CPU **C**; CU **S** (stride-driven on input — transposed-view grads skip Contiguize).
- **op_params**: `OpParams::ReduceSumTo / ReduceMaxTo { input_shape, output_shape }`.
  Left-pads output_shape with 1s; per axis carries through or sums/maxes away.
- **output_behavior**: dtype T, shape = output_shape.
- **source**: dispatch.rs:4032/4037 (CPU), baracuda_dispatch.rs:2900 (CU).

### ReduceMaxToBackward
- **OpKind**: `ReduceMaxToBackward`
- **dtypes**: CPU f32, f64, bf16, f16 — key `[T, T, T]` = `(x, upstream, grad_x)`.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::ReduceMaxToBackward { input_shape, output_shape }`.
- **output_behavior**: recomputes forward max, routes upstream to argmax
  positions (fair-share on ties). grad_x dtype/shape = input_shape.
- **source**: dispatch.rs:4445 (CPU).

### ArgMaxDim / ArgMinDim
- **OpKind**: `ArgMaxDim`, `ArgMinDim`
- **dtypes**: CPU f32, f64, bf16, f16 (input); CU f32, f64, f16, bf16; VK f32/f16/bf16/f64.
  key `[input_dt, U32]`.
- **input_layouts**: CPU **C** (`argmax_dim_u32_cpu_dispatch` internal input-dtype match); CU **S**; VK **C**.
- **op_params**: `OpParams::Reduce { dims, keepdim }` (dim carried in Reduce).
- **output_behavior**: **output dtype = U32 indices** regardless of input T.
- **source**: dispatch.rs:4579 (CPU), baracuda_dispatch.rs:2515 (CU),
  vulkan_dispatch.rs:4507 (VK).

### SoftmaxLastDim
- **OpKind**: `SoftmaxLastDim`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16; VK f32 (+ f16/bf16/f64 gated).
- **input_layouts**: CPU **C**; CU **S** (wrapper requires `layouts[0]`); VK **C** (`PrecisionGuarantee::none`).
- **op_params**: `OpParams::SoftmaxLastDim { outer_count, last_dim }`.
- **output_behavior**: same dtype/shape; walks outer_count rows of last_dim.
  half accumulates in f32 (CPU).
- **source**: dispatch.rs:4532 (CPU), baracuda_dispatch.rs:2464 (CU),
  vulkan_dispatch.rs:4332 (VK).

### LogSoftmaxLastDim
- **OpKind**: `LogSoftmaxLastDim`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16.
- **input_layouts**: CPU **C**; CU **S**.
- **op_params**: `OpParams::LogSoftmaxLastDim { outer_count, last_dim }`.
- **output**: same dtype/shape.
- **source**: dispatch.rs:4419 (CPU), baracuda_dispatch.rs:2469 (CU).

### SoftmaxLastDimBackward
- **OpKind**: `SoftmaxLastDimBackward`
- **dtypes**: CPU f32, f64, bf16, f16 (key `[T,T,T]`); VK f32, f16, bf16, f64.
- **input_layouts**: CPU **C**; VK **C** (`PrecisionGuarantee::none`).
- **op_params**: norm/softmax-shaped (consumed in wrapper). **output**: T.
- **source**: dispatch.rs:4433 (CPU), vulkan_dispatch.rs:4344 (VK).

### LogSoftmaxLastDimBackward
- **OpKind**: `LogSoftmaxLastDimBackward`
- **dtypes**: CPU f32, f64, bf16, f16 (key `[T,T,T]` = `(y, g, out)`).
- **input_layouts**: CPU **C**. **output**: T.
- **source**: dispatch.rs:4425 (CPU).

### RmsNormLastDim
- **OpKind**: `RmsNormLastDim`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16; VK f32 (+ f16/bf16/f64 gated).
- **input_layouts**: CPU **C**; CU **S** (passes rank-N shape + strides to FFI); VK **C** (`none`).
- **op_params**: `OpParams::NormLastDim { outer_count, last_dim, eps }`.
- **output_behavior**: same dtype/shape; no affine flavor. half accum f32 (CPU).
- **source**: dispatch.rs:4536 (CPU), baracuda_dispatch.rs:2439 (CU),
  vulkan_dispatch.rs:4333 (VK).

### LayerNormLastDim
- **OpKind**: `LayerNormLastDim`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16; VK f32 (+ f16/bf16/f64 gated).
- **input_layouts**: CPU **C**; CU **S**; VK **C** (`none`).
- **op_params**: `OpParams::NormLastDim { outer_count, last_dim, eps }`.
- **output**: same dtype/shape; no affine.
- **source**: dispatch.rs:4540 (CPU), baracuda_dispatch.rs:2444 (CU),
  vulkan_dispatch.rs:4385 (VK).

### RmsNormLastDimBackward / LayerNormLastDimBackward
- **OpKind**: `RmsNormLastDimBackward`, `LayerNormLastDimBackward`
- **dtypes**: CPU f32, f64, bf16, f16 (key `[T,T,T]`); VK (LayerNorm bwd) f32, f16, bf16, f64.
- **input_layouts**: CPU **C**; VK **C** (`none`).
- **op_params**: norm-shaped. **output**: T.
- **source**: dispatch.rs:4437/4441 (CPU), vulkan_dispatch.rs:4371 (VK LayerNorm bwd).

### Rope (rotary position embedding)
- **OpKind**: `Rope`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16; VK f32, f16, f64, bf16.
  CPU/VK key `[x, cos, sin, out]` = `[T,T,T,T]`; CU key `[T,T]` (canonical short).
- **input_layouts**: CPU **C** (`rope_*_cpu_wrapper`); CU **S**; VK **S** on `x`
  (cos/sin forced contiguous by the wrapper; rope.slang carries x strides + fast-path flag).
- **op_params**: `OpParams::Rope { outer_count, seq, head_dim }`. cos/sin
  `[seq, head_dim]` broadcast across outer dims.
- **output**: same dtype/shape as x.
- **source**: dispatch.rs:4555 (CPU), baracuda_dispatch.rs:2456 (CU),
  vulkan_dispatch.rs:4416 (VK).

### Affine (y = mul*x + add)
- **OpKind**: `Affine`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16, i32, i64, u8; VK f32, f64, f16, bf16.
- **input_layouts**: CPU **C**; CU **S**; VK **S** for f32/f64/f16, **C** for
  **bf16** (pair-thread packed-u32 kernel, no strided cap — divergence from siblings).
- **op_params**: `OpParams::Affine { mul: f64, add: f64 }`.
- **output**: same dtype/shape.
- **source**: dispatch.rs:4099/4583 (CPU), baracuda_dispatch.rs:2656 (CU),
  vulkan_dispatch.rs:4648 (VK; bf16 contiguous at 4658).

### Cast (dtype conversion)
- **OpKind**: `Cast`
- **dtypes (src→dst pairs)**:
  - CPU: F64→F32, BF16→F32, F16→F32, F32→F64, F32→BF16, F32→F16, plus
    F8E4M3↔{F32,BF16,F16}. Each per-target wrapper matches src dtype internally
    (`cpu_cast_wrapper`, identity arm omitted). Identity casts are elided by the optimizer.
  - CU: full 8×8 cross product over {F32,F64,F16,BF16,I32,U32,I64,U8} (one
    `cast_baracuda_wrapper` dispatching on in/out dtype) + F8E4M3↔{F32,F16,BF16}.
  - VK: `[f32,f16]`,`[f16,f32]`,`[f32,bf16]`,`[bf16,f32]` (cast_f32_half);
    `[f32,f64]`,`[f64,f32]` (gated); F8E4M3↔{F32,F16,BF16} (cast_f8e4m3).
- **input_layouts**: CPU **C** (`_layouts` unused); CU **C**; VK **C**.
- **op_params**: `OpParams::Cast` (target dtype lives on output Storage).
- **output_behavior**: **output dtype = dst** (last key entry), same shape.
  half↔half pivots through f32 on CPU.
- **source**: dispatch.rs:3993 (CPU), baracuda_dispatch.rs:2747 (CU),
  vulkan_dispatch.rs:4680/5014 (VK).

### Conv2D (forward)
- **OpKind**: `Conv2D`
- **dtypes**: CPU f32, f64, bf16, f16; each registered with **both** no-bias
  `[x,w,out]` (`[T,T,T]`) and with-bias `[x,w,bias,out]` (`[T,T,T,T]`) shapes.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::Conv2D { x_shape[4], w_shape[4], out_shape[4],
  stride, padding, dilation, groups }` (asymmetric stride/padding).
- **output_behavior**: out `[N, Cout, Hout, Wout]`, dtype T.
- **source**: dispatch.rs:4013/4019 (CPU).

### ConvTranspose2D (forward)
- **OpKind**: `ConvTranspose2D`
- **dtypes**: CPU f32, f64, bf16, f16; no-bias + with-bias shapes.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::ConvTranspose2D { x_shape, w_shape, out_shape,
  stride, padding, output_padding, dilation, groups }`. weight `[Cin, Cout/groups, Kh, Kw]`.
- **output**: out `[N, Cout, Hout, Wout]`, dtype T.
- **source**: dispatch.rs:4022/4028 (CPU).

### CausalConv1d
- **OpKind**: `CausalConv1d`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. key `[T,T,T,T]`
  = `(x, weight, bias, out)`.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `OpParams::CausalConv1d { batch, channels, seq_in, seq_out,
  kernel, use_silu }`. x pre-padded with `kernel-1` left zeros by caller.
- **output**: `[batch, channels, seq_out]`, dtype T; `use_silu` fuses SiLU on store.
- **source**: dispatch.rs:4146 (CPU), baracuda_dispatch.rs:2911 (CU).

### FlashAttn (forward)
- **OpKind**: `FlashAttn`
- **dtypes**: CPU f32, f64, bf16, f16; both no-alibi `[q,k,v,out]` (`[T,T,T,T]`)
  and with-alibi `[q,k,v,alibi,out]` (`[T,T,T,T,T]`).
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::FlashAttn { b, hq, hkv, sq, sk, d, k_len,
  softmax_scale, causal, window_size_left, window_size_right, softcap }`.
  `sk` = physical K/V extent; `k_len` ≤ sk = logical attended length;
  causal mask bottom-right-aligned at `k_len - sq` (Phase D symbolic extents).
- **output**: `[B, Hq, Sq, D]`, dtype T. GQA: Hkv ≤ Hq divisible.
- **notes**: CPU-only in binding table (no CU/VK FlashAttn binding here).
- **source**: dispatch.rs:4050 (CPU).

### FlashAttn backward (Q, K, V)
- **OpKind**: `FlashAttnBackwardQ`, `FlashAttnBackwardK`, `FlashAttnBackwardV`
- **dtypes**: CPU f32, f64, bf16, f16; no-alibi `[q,k,v,do,out]` (`[T×5]`) and
  with-alibi `[q,k,v,do,alibi,out]` (`[T×6]`).
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::FlashAttn`.
- **output_behavior**: each OpKind emits the requested gradient (dtype T). The
  CPU wrapper computes all three gradients every call and copies the requested
  one — expect ~3× cost vs a single-gradient kernel.
- **source**: dispatch.rs:4079-4084 (CPU).

### PagedAttn
- **OpKind**: `PagedAttn`
- **dtypes**: CPU f32, f64, bf16, f16; no-alibi `[q, kc, vc, U32, U32, out]` and
  with-alibi `[..., alibi, out]`. block_table + context_lens always U32.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::PagedAttn { b, hq, hkv, sq, d, block_size,
  max_blocks_per_seq, num_blocks, softmax_scale, softcap }`.
- **output**: `[B, Hq, Sq, D]`, dtype T. k_cache/v_cache `[num_blocks, block_size, Hkv, D]`.
- **source**: dispatch.rs:4089 (CPU).

### FusedSoftmaxCrossEntropy
- **OpKind**: `FusedSoftmaxCrossEntropy`
- **dtypes**: CPU — `[logits T, I64, F32]` for T ∈ {F32, F64, BF16, F16}.
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::FusedSoftmaxCrossEntropy { n_rows, vocab, reduction, ignore_index }`.
- **output_behavior**: **output dtype always F32**; shape `[]` scalar for
  Mean/Sum, `[n_rows]` for None. Stable log-softmax + NLL + ignore_index mask in one pass.
- **source**: dispatch.rs:4136 (CPU).

### SelectiveScan (Mamba SSM)
- **OpKind**: `SelectiveScan`
- **dtypes**: CPU f32, f64, bf16, f16. key `[T,T,T,T,T,T]` (5 inputs + out).
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::SelectiveScan { batch, seqlen, dim, dstate, delta_softplus }`.
- **output**: `y [batch, seqlen, dim]`, dtype T. Inputs u/delta/a/b/c.
- **source**: dispatch.rs:4165 (CPU).

### SsdChunkScan (Mamba-2 SSD)
- **OpKind**: `SsdChunkScan`
- **dtypes**: CPU f32, f64, bf16, f16. key `[T×6]` (5 inputs + out).
- **input_layouts**: CPU **C**.
- **op_params**: `OpParams::SsdChunkScan { batch, seqlen, heads, head_dim,
  state_dim, chunk_size }`. chunk_size > 0, seqlen % chunk_size == 0.
- **output**: `y` matches `x` shape, dtype T. CPU runs sequential scan regardless of chunk_size.
- **source**: dispatch.rs:4176 (CPU).

### Concat (variadic)
- **OpKind**: `Concat`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8, i16, i32, i64; CU f32, f64, f16, bf16;
  VK f32 (+ f16/bf16/f64 gated). Canonical short key `[T, T]` (uniform-dtype N inputs + output).
- **input_layouts**: CPU **C** (dtype-agnostic byte slabs); CU **S** (stride-aware
  when layouts supplied); VK **S**.
- **op_params**: `OpParams::Concat { outer_count, input_dim_sizes (len N), inner_count, axis }`.
- **output_behavior**: dtype T; shape = inputs concatenated along `axis`. Wrapper
  validates `input_dim_sizes.len() == inputs.len()`.
- **source**: dispatch.rs:4525 (CPU), baracuda_dispatch.rs:2558 (CU),
  vulkan_dispatch.rs:4542 (VK).

### IndexSelect
- **OpKind**: `IndexSelect`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8, i16, i32, i64; CU f32, f64, i32;
  VK f32 (+ f16/bf16/f64 gated). key `[data, U32, data]`.
- **input_layouts**: CPU **C**; CU **C** (default caps); VK **C** (byte-level).
- **op_params**: `OpParams::IndexSelect { outer_count, source_dim_size, n_indices, inner_count }`.
- **output_behavior**: dtype = data; gathers along single dim via rank-1 U32 indices.
- **source**: dispatch.rs:4550 (CPU), baracuda_dispatch.rs:2488 (CU),
  vulkan_dispatch.rs:4435 (VK).

### Gather (N-dim)
- **OpKind**: `Gather`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8, i16, i32, i64; CU f32, f64, i32;
  VK f32 (+ f16/bf16/f64/u8/u32 gated). key `[data, U32, data]`.
- **input_layouts**: CPU **C**; CU **C**; VK **C** (byte-level).
- **op_params**: `OpParams::Gather { source_shape, output_shape, dim }`.
- **output**: dtype = data, shape = output_shape; indices (U32) have output_shape.
- **source**: dispatch.rs:4551 (CPU), baracuda_dispatch.rs:2492 (CU),
  vulkan_dispatch.rs:4584 (VK).

### IndexAdd
- **OpKind**: `IndexAdd`
- **dtypes**: CPU f32, f64, bf16, f16; VK f32, f64, bf16, f16. key `[base, U32, src, out]`.
- **input_layouts**: CPU **C**; VK **C** (`PrecisionGuarantee::none`).
- **op_params**: `OpParams::IndexAdd { outer_count, base_dim_size, n_indices, inner_count }`.
- **output_behavior**: out = base with `src[...,i,...]` accumulated into
  `base[...,indices[i],...]`. dtype = base.
- **source**: dispatch.rs:4564 (CPU), vulkan_dispatch.rs:4473 (VK).

### ScatterAdd (N-dim)
- **OpKind**: `ScatterAdd`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64; VK f32, f64, bf16, f16.
  key `[base, U32, src, out]`.
- **input_layouts**: CPU **C**; CU **C**; VK **C** (`none`).
- **op_params**: `OpParams::ScatterAdd { base_shape, src_shape, dim }`.
- **output**: dtype = base; indices+src share shape; base differs only along dim.
- **source**: dispatch.rs:4568 (CPU), baracuda_dispatch.rs:2500 (CU),
  vulkan_dispatch.rs:4491 (VK).

### MaskedFill
- **OpKind**: `MaskedFill`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8; CU f32, f64, i32; VK f32, f16, bf16, f64, u8, u32.
  key `[T, U8, T]` (x, mask U8, out).
- **input_layouts**: CPU **C** (dtype-agnostic byte kernel); CU **C**; VK **C** (byte-level).
- **op_params**: `OpParams::MaskedFill { fill_bytes }` (pre-encoded one element in output dtype).
- **output**: dtype = x; fills where mask nonzero.
- **source**: dispatch.rs:4453 (CPU), baracuda_dispatch.rs:2496 (CU),
  vulkan_dispatch.rs:4573 (VK).

### Pad (multi-dim, forward)
- **OpKind**: `Pad`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8 (dtype-agnostic byte wrapper);
  CU f32, f64, f16, bf16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `OpParams::Pad { in_shape, out_shape, padding (per-axis
  before/after), mode_tag (0=Constant,1=Reflect,2=Replicate), fill_bytes }`.
- **output_behavior**: dtype T; `out_shape[i] = in_shape[i] + pad.before + pad.after`.
  CPU: Constant wired; Reflect/Replicate error in wrapper today. CU: Constant/Reflect/Replicate.
- **source**: dispatch.rs:4463 (CPU), baracuda_dispatch.rs:2642 (CU).

### PadBackward
- **OpKind**: `PadBackward`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, f16, bf16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C** (Constant only — sum-accumulating backwards CPU-only).
- **op_params**: `OpParams::PadBackward { in_shape, out_shape, padding, mode_tag }`.
- **output**: dtype T (typed accumulation), shape = in_shape.
- **source**: dispatch.rs:4471 (CPU), baracuda_dispatch.rs:2647 (CU).

### Flip
- **OpKind**: `Flip`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8 (one dtype-agnostic byte
  wrapper `flip_cpu_wrapper`); CU f32, f64, f16, bf16; VK f32/f16/bf16/f64/… (per dtype).
- **input_layouts**: CPU **C**; CU **S** (rank-N walk via axis from OpParams); VK **S**.
- **op_params**: `OpParams::Flip { outer_count, dim_size, inner_count, axis }`.
- **output**: dtype T (byte copy), same shape, reversed along dim.
- **source**: dispatch.rs:4359 (CPU), baracuda_dispatch.rs:2618 (CU),
  vulkan_dispatch.rs:4839 (VK).

### Roll
- **OpKind**: `Roll`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8 (byte-agnostic `roll_cpu_wrapper`);
  CU f32, f64, f16, bf16; VK per dtype.
- **input_layouts**: CPU **C**; CU **S**; VK **S**.
- **op_params**: `OpParams::Roll { outer_count, dim_size, inner_count, shift (signed i64), axis }`.
- **output**: dtype T, cyclic shift along dim by `shift` (wraps).
- **source**: dispatch.rs:4366 (CPU), baracuda_dispatch.rs:2624 (CU),
  vulkan_dispatch.rs:4840 (VK).

### CumSum
- **OpKind**: `CumSum`
- **dtypes**: CPU f32, f64, bf16, f16 (per-dtype — typed accumulation, not byte
  copy); CU f32, f64, f16, bf16; VK f32, f64, f16, bf16.
- **input_layouts**: CPU **C**; CU **S**; VK **S**.
- **op_params**: `OpParams::CumSum { outer_count, dim_size, inner_count, axis }`.
- **output**: dtype = input; inclusive prefix sum along dim. (Reverse-cumsum
  expressed upstream as Flip→CumSum→Flip.)
- **source**: dispatch.rs:4374 (CPU), baracuda_dispatch.rs:2632 (CU),
  vulkan_dispatch.rs:4850 (VK).

### Triu / Tril (triangular mask)
- **OpKind**: `Triu`, `Tril`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8 (one byte-agnostic kernel each,
  keyed per dtype); CU f32, f64, f16, bf16, i32, i64; VK per dtype (byte-level).
- **input_layouts**: CPU **C**; CU **S**; VK **C** (byte-level).
- **op_params**: `OpParams::Triangular { batch_count, rows, cols, diagonal (signed) }`.
  Same variant for both; OpKind picks keep-upper vs keep-lower.
- **output**: dtype T, same shape.
- **source**: dispatch.rs:4405/4411 (CPU), baracuda_dispatch.rs:2599 (CU),
  vulkan_dispatch.rs:4837 (VK).

### WriteSlice (in-place rectangular slab assign)
- **OpKind**: `WriteSlice`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8 (one dtype-agnostic byte
  wrapper); CU + VK byte-width-keyed (b1/b2/b4/b8) covering f32, f64, f16,
  bf16, i32, i64, u32, u8, i8. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C**; VK **C** (byte-level). One input (source) + one output (dest).
- **op_params**: `OpParams::WriteSlice { dest_shape, ranges (per-axis start/end) }`.
- **output_behavior**: **IN-PLACE / aliasing** — writes source slab into the
  pre-existing destination buffer (`outputs[0]` IS the dest, mutated in place).
  Backs persistent KV-cache writes (Phase E.3.2). dtype = T.
- **source**: dispatch.rs:4385 (CPU), baracuda_dispatch.rs:2567 (CU),
  vulkan_dispatch.rs:4771 (VK).

### WriteSliceRotating (ring-buffer slab assign)
- **OpKind**: `WriteSliceRotating`
- **dtypes**: same surface as WriteSlice (CPU byte-agnostic; CU/VK byte-width-keyed).
- **input_layouts**: CPU **C**; CU **C**; VK **C**. Two inputs (source, dynamic
  position scalar rank-0 U32) + one output (dest).
- **op_params**: `OpParams::WriteSliceRotating { dest_shape, axis, modulus,
  ranges }`. On the rotating axis `ranges[axis].0` ignored (dynamic), width must
  equal source dim and ≤ modulus.
- **output_behavior**: **IN-PLACE / aliasing** ring write modulo `modulus`;
  position read from the extra input. Backs sliding-window KV caches (Phase C).
- **source**: dispatch.rs:4397 (CPU), baracuda_dispatch.rs:2581 (CU),
  vulkan_dispatch.rs:4785 (VK).

### In-place affine
- **OpKind**: `InplaceAffine`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C** (no strided cap — executor rejects
  strided in-place targets up front).
- **op_params**: `OpParams::Affine { mul, add }`.
- **output_behavior**: **IN-PLACE** — target passed as `outputs[0]`; the wrapper
  rejects non-empty `inputs`. Mutates in place.
- **source**: dispatch.rs:4108 (CPU), baracuda_dispatch.rs:2669 (CU).

### In-place clamp / powi
- **OpKind**: `ClampInplace`, `PowIInplace`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `OpParams::Clamp { min, max }` / `OpParams::PowI { exp }`.
- **output_behavior**: **IN-PLACE**.
- **source**: dispatch.rs:4116/4121 (CPU), baracuda_dispatch.rs:2679/2684 (CU).

### In-place unary activations (Relu, Silu, Gelu, Tanh, Sigmoid)
- **OpKind**: `ReluInplace`, `SiluInplace`, `GeluInplace`, `TanhInplace`, `SigmoidInplace`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `None`. **output**: **IN-PLACE**, dtype T.
- **notes**: CPU half routes through f32-pivot blanket impls (bit-matches non-inplace).
- **source**: dispatch.rs:4213 (CPU), baracuda_dispatch.rs:2693 (CU).

### In-place unary op family (16 ops)
- **OpKind**: `NegInplace`, `AbsInplace`, `SqrInplace`, `SqrtInplace`,
  `RsqrtInplace`, `RecipInplace`, `ExpInplace`, `LogInplace`, `SinInplace`,
  `CosInplace`, `SignInplace`, `FloorInplace`, `CeilInplace`, `RoundInplace`,
  `ErfInplace`, `GeluErfInplace`
- **dtypes**: CPU f32, f64, bf16, f16; CU f32, f64, bf16, f16. key `[T, T]`.
- **input_layouts**: CPU **C**; CU **C**.
- **op_params**: `None`. **output**: **IN-PLACE**, dtype T.
- **source**: dispatch.rs:4239 (CPU loop), baracuda_dispatch.rs:2719 (CU loop).

### Copy (cross-device / same-device byte transfer)
- **OpKind**: `Copy`
- **dtypes**: CPU f32, f64, bf16, f16, u32, u8, i16, i32, i64 (CPU→CPU memcpy);
  CU (source=Cuda) f32, bf16, f16, u32, f64, u8, i16, i32, i64; VK (source=Vulkan)
  every byte-substrate dtype. key `[T, T]`.
- **input_layouts**: **C** (byte-level full-buffer copy). Keyed on the **SOURCE**
  backend; routes on the output's substrate variant.
- **op_params**: `None`.
- **output_behavior**: CUDA wrapper: CPU output → D2H (`to_cpu_bytes`); CUDA
  output → D2D (`slot_copy_to_new`). VK: D2H to CPU. CPU: CPU→CPU memcpy noop.
  Output pre-allocated by executor (`WorkItemKind::Copy`).
- **source**: dispatch.rs:4602 (CPU), dispatch.rs:4790 (CUDA PTX path),
  vulkan_dispatch.rs:5033 (VK).

---

## Cross-cutting contract facts

- **Multi-output ops** (e.g. `SelectiveScan` returning `(y, last_state)`) emit
  ONE `KernelRef` with `outputs.len() == 1`; the single Storage is bundled and
  the kernel writes each logical slot by `byte_offset` from the bundle metadata
  (`kernel.rs:121-149`). (In this crate's current registrations the wired
  multi-output is not exercised — SelectiveScan/SsdChunkScan register a single
  output T.)
- **Cost + precision**: every binding carries a `CostFn` (`unknown_cost`
  sentinel → bulk-filled by `fill_unset_cpu_cost` /
  `fill_unset_cost_for_backend` to `default_cost_for_op_kind`) and a
  `PrecisionGuarantee` (CPU bulk-upgraded to `PRIMITIVE_DETERMINISTIC_CPU`; VK
  reductions/softmax/norm/argreduce explicitly `PrecisionGuarantee::none`).
- **Alternative sets**: a single `(op, dtypes, backend)` key can hold multiple
  `BindingEntry` siblings (e.g. PTX Copy + baracuda kernels coexist on CUDA).
  Duplicate `KernelRef` function pointers at one key panic at registration.
- **Offset capability**: NO kernel in this crate is offset-capable. Even
  strided-capable kernels send non-zero-`start_offset` inputs through
  auto-Contiguize (`compiled.rs` caps gate; `KernelCaps` doc, kernel.rs:66-74).

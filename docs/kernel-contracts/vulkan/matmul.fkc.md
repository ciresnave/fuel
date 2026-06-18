---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — matmul / matvec family kernel contracts

Batched GEMM and GEMV kernels for the Vulkan backend (crate `vulkan`, family `matmul`). Every
kernel here implements `OpKind::MatMul` (`fuel-core-types/src/dispatch.rs:54`) and consumes the
`OpParams::Matmul` variant. The Vulkan `MatmulParams` block carries `M`, `N`, `K`, per-operand
strides `sa_batch/sa_row/sa_col` and `sb_batch/sb_row/sb_col`, the output batch stride `sc_batch`,
and a GQA repeat factor `n_rep` (`fuel-vulkan-backend/src/lib.rs:2685`). The dispatch key is
`(OpKind::MatMul, [A_dtype, B_dtype, C_dtype], Vulkan) + kernel_source` (§3.2, §12.1); the
mixed-precision variants (f32×bf16, bf16×bf16, f16×f16) are distinguished by the per-operand dtype
slots in the key, not by a separate op kind.

**Layout model — stride-capable, offset-incapable (this is the load-bearing difference from the CPU
matmul family).** Unlike the contiguous-only CPU GEMM, the Vulkan matmul/matvec kernels address A
and B through their per-operand strides (`sa_row/sa_col`, `sb_row/sb_col`), so a transposed /
permuted (metadata-only) operand is consumed **directly with no contiguize** — these are
`handles_strided` kernels (§4.3). They are **not** non-zero-offset capable (only a per-batch base
is computed, no element `byte_offset`), so a non-zero-start-offset operand still routes through an
upstream `Op::Contiguize`. GQA is expressed via `n_rep` (`b_off = (batch / n_rep) * sb_batch`), **not**
a stride-0 broadcast axis, so `broadcast_stride0` is rejected. Negative strides are not handled
(`reverse_strides: rejected`). Output is always freshly-allocated **contiguous** row-major
(`C[batch, M, N]`, index `r*N + c`), no aliasing, not in-place — the universal output-contiguity rule
(inventory "Notes / cross-cutting contracts").

**Route picking — sibling alternatives at one key.** The f32×f32×f32 wrapper `matmul_f32_bytes`
(`fuel-vulkan-backend/src/lib.rs:3759`, picker `:3830`) selects `matvec` for `M == 1`, the register-
tiled `matmul` for `1 < M < 32`, and the shared-memory `matmul_tiled` for `M >= 32`. These three are
**distinct `KernelRef`s at the same `(MatMul, [F32,F32,F32], Vulkan)` key** — i.e. sibling
alternatives the route picker ranks (§12.5), each with its own contract below. Likewise the coop /
small families register distinct siblings at their mixed-precision keys.

**Capability-gated coop pipelines.** The six `matmul_coop*` kernels build and dispatch **only when
`VK_KHR_cooperative_matrix` is present** (`has_coop_matrix`; the pipeline objects are `Option`,
inventory "Notes"). They use a cooperative-matrix (tensor-core) M=N=K=16 tile with an f32
accumulator; the `matmul_small_*` scalar-accumulator kernels are the any-shape fallback used when
the coop-tile constraints fail (`M < 16`, `M % 16 != 0`, `N % 16 != 0`, or M==1 gemv). The coop
constraint check is `matmul_coop_ok` (`fuel-vulkan-backend/src/lib.rs:2824`).

**Cost provenance.** Every cost block is marked `judge_measured`: the Judge bootstraps it (§4.4).
The FLOPs hint `2 * batch * m * n * k` is the genuinely derivable GEMM flop count (one multiply +
one add per inner-product term, summed over all batches; for M==1 gemv it reduces to
`2 * batch * n * k`). No other coefficients are fabricated — the Judge populates
`bytes_moved` / `overhead_ns` / `memory` from measurement.

**Determinism (corrected 2026-06-18).** Every matmul/matvec kernel here accumulates in f32 over a
register/shared-memory tile or a subgroup reduction whose FADD / subgroup order is
**scheduler-dependent**, so none is bit-stable even on a re-run on the same device. These are
therefore `determinism: nondeterministic` with `bit_stable_on_same_hardware: false` and an audited
`none(reason)` precision (no silent unaudited nondeterminism) — matching the flash-attn
(conv-attn-rope) and qmatmul (quantized) precedent and §10 rule 9
(`nondeterministic ⇒ bit_stable=false + audited:true`). (They previously mis-declared
`same_hardware_bitwise`, which contradicted their own `bit_stable_on_same_hardware: false`.)

---

## matmul  (batched f32 GEMM, register-tiled 4×4)

WGSL-origin Slang batched GEMM `C = A@B` with a 4×4 register tile and **no shared memory**
(`matmul.slang:26`). The route picker selects this kernel for `1 < M < 32`
(`matmul_f32_bytes`, `fuel-vulkan-backend/src/lib.rs:3759`, picker `:3830`). A and B are addressed
through `sa_batch/sa_row/sa_col` and `sb_batch/sb_row/sb_col`, so any row/col stride is consumed
directly (transpose-friendly) — only a per-batch base is computed, no element offset. GQA is handled
via `n_rep` (`b_off = (batch / n_rep) * sb_batch`); the output batch is the lhs batch. Out-of-range
threads are guarded. f32 multiply-accumulate; the output is `C[batch, M, N]` row-major contiguous.
Known limitation: stride-capable but **not** non-zero-offset capable — a non-zero-start-offset
operand must be contiguized by the planner first; no stride-0 broadcast (GQA is via `n_rep`).

```fkc
kernel: matmul
op_kind: MatMul
blurb: "Batched f32 GEMM, register-tiled 4x4 (no shared mem); stride-capable A/B; GQA via n_rep; picker uses for 1<M<32."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "addressed via sa_batch/sa_row/sa_col; per-batch base only, no element offset; GQA via n_rep not stride-0."
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "addressed via sb_batch/sb_row/sb_col; rhs batch slot read as lhs_slot / n_rep."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs % rhs == 0); packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)        # lhs_batch ++ [m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks sa_*/sb_* strides directly; no contiguize for strided/transposed A/B
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; FLOPs hint below is the derivable GEMM count
  class: gemm_like
  flops: "2 * batch * m * n * k"        # one multiply + one add per inner-product term, over all batches

precision:
  bit_stable_on_same_hardware: false    # f32 register-tile accumulation; subgroup/scheduler order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f32 multiply-accumulate, register-tiled; accumulation order tile-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_tiled  (batched f32 GEMM, shared-memory blocked 64×64, BK=16)

GLSL batched GEMM with a 64×64 shared-memory tile and an inner block depth `BK=16`
(`matmul_tiled.glsl:39`). The route picker selects this kernel for `M >= 32`
(`matmul_f32_bytes`, picker at `fuel-vulkan-backend/src/lib.rs:3830`). It uses the **identical
`MatmulParams` block** as `matmul` — same `sa_*/sb_*` stride model, same `n_rep` GQA, same per-batch
base (no element offset). The shared-memory blocking stages A/B sub-tiles into workgroup-shared
memory to amortize global loads across the 64×64 output tile. f32 multiply-accumulate; output is
`C[batch, M, N]` row-major contiguous. Same layout limitations as `matmul`: stride-capable, not
offset-capable, no stride-0 broadcast.

```fkc
kernel: matmul_tiled
op_kind: MatMul
blurb: "Batched f32 GEMM, shared-memory blocked 64x64 (BK=16); stride-capable A/B; GQA via n_rep; picker uses for M>=32."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_tiled"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "same stride model as matmul (sa_batch/sa_row/sa_col); per-batch base only."
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "sb_batch/sb_row/sb_col; rhs batch slot read as lhs_slot / n_rep."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible (lhs % rhs == 0); packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false    # shared-mem tile accumulation; subgroup/scheduler order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f32 multiply-accumulate, 64x64 shared-mem tile (BK=16); accumulation order tile-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matvec  (f32 GEMV, M==1)

GLSL gemv specialization for `M == 1` (`matvec.glsl:26`), selected by `matmul_f32_bytes` when
`M == 1`. One workgroup per output column computes a subgroup-reduced dot product. A is addressed
via `sa_col`, B via `sb_row/sb_col`, with a per-batch base — permute/transpose-friendly. The
output is `C[batch, N]` contiguous. Same stride-capable, offset-incapable layout model as the GEMM
kernels; GQA via `n_rep`. Because the dot reduction runs across a subgroup, the f32 accumulation
order follows the subgroup schedule and is not pinned, so the result is not bit-stable cross-hardware.

```fkc
kernel: matvec
op_kind: MatMul
blurb: "f32 GEMV (M==1); subgroup-reduced dot, one workgroup/col; stride-aware A/B; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matvec"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=1"        # M == 1
      notes: "addressed via sa_col; per-batch base only, no element offset."
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "addressed via sb_row/sb_col; rhs batch slot read as lhs_slot / n_rep."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== 1" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: matmul(lhs, rhs)        # batch ++ [1, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * n * k"             # GEMV (M==1): one multiply + one add per (n,k) term, over all batches

precision:
  bit_stable_on_same_hardware: false    # subgroup dot reduction; FADD order scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f32 subgroup-reduced dot; reduction order follows subgroup schedule (not pinned); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matvec_bf16_b  (mixed-precision f32 × bf16 GEMV, M==1)

GLSL mixed-precision gemv `C = A @ B` with `M == 1`: f32 A × **bf16 B** → f32 C
(`matvec_bf16_b.glsl:53`; wrapper `matmul_f32_bf16_b_bytes`, `fuel-vulkan-backend/src/lib.rs:2644`).
This is the decode hot path for bf16 weights. B is stored packed 2-bf16-per-u32 and unpacked on load
by **bit shift** (`bits << 16`), which is the **exact** bf16→f32 widening (BF16 is the upper 16 bits
of F32, no mantissa loss). `sb_batch` is counted in bf16 elements; the u32 base is therefore half
the element index. A is f32, addressed via `sa_col`; B via `sb_row/sb_col`. Subgroup-reduced dot in
f32; output `C[N]` contiguous. Stride-aware, offset-incapable, GQA via `n_rep`. The dispatch key is
`(MatMul, [F32, BF16, F32], Vulkan)` — distinct from the all-f32 `matvec` key by the B dtype slot.

```fkc
kernel: matvec_bf16_b
op_kind: MatMul
blurb: "Mixed-precision GEMV (M==1): f32 A x bf16 B -> f32 C; exact bf16 unpack (bits<<16); stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matvec_bf16_b"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=1"        # M == 1
      notes: "f32 A addressed via sa_col; per-batch base only."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B packed 2-per-u32; sb_batch in bf16 elements, u32 base = half; unpacked by bits<<16 (exact)."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== 1" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # f32 C regardless of bf16 B
      shape_rule: matmul(lhs, rhs)        # batch ++ [1, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * n * k"             # GEMV (M==1); bf16 B widened to f32 before MAC

precision:
  bit_stable_on_same_hardware: false    # subgroup dot reduction; bf16->f32 leg is exact
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 B unpacked via bits<<16 (EXACT widening, no mantissa loss); f32 subgroup-reduced dot; reduction order scheduler-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_tiled_bf16_b  (mixed-precision f32 × bf16 GEMM, M>1)

Tiled mixed-precision GEMM for `M > 1`: f32 A × **bf16 B** → f32 C; bf16 B is unpacked to f32 on
load. SPIR-V only (`matmul_tiled_bf16_b.spv`; no Slang source in tree); the contract is read from
the Rust wrapper `matmul_f32_bf16_b_bytes` (`fuel-vulkan-backend/src/lib.rs:2644`) and the
`EMBEDDED` doc comments. It uses the same tiling and `sa_*/sb_*` stride model as `matmul_tiled` —
stride-capable, offset-incapable, GQA via `n_rep`. The bf16→f32 unpack on the B load is the exact
`bits << 16` widening; the multiply-accumulate runs in f32. Output `C[batch, M, N]` row-major
contiguous. Dispatch key `(MatMul, [F32, BF16, F32], Vulkan)` — the GEMM sibling to `matvec_bf16_b`
at the same key (M>1 vs M==1 route choice).

```fkc
kernel: matmul_tiled_bf16_b
op_kind: MatMul
blurb: "Mixed-precision tiled GEMM (M>1): f32 A x bf16 B -> f32 C; exact bf16 unpack on B load; stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_tiled_bf16_b"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "same tiling/stride model as matmul_tiled (sa_batch/sa_row/sa_col); per-batch base only."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B unpacked on load by bits<<16 (exact widening); sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # f32 C regardless of bf16 B
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false    # shared-mem tile accumulation; subgroup/scheduler order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 B unpacked via bits<<16 (EXACT widening); f32 tile multiply-accumulate; not bit-stable cross-hardware. SPIR-V only (no Slang source); contract from wrapper + EMBEDDED docs."

determinism: nondeterministic
```

---

## matmul_coop  (cooperative-matrix f32 × bf16 GEMM → f32; tensor-core)

GLSL cooperative-matrix (tensor-core) GEMM: f32 A × bf16 B → f32 C, with **A and B downcast to f16
on the shared-memory load** and an **f32 accumulator** (coop tile shape M=N=K=16,
`matmul_coop.glsl:54`). Dispatched **only when `VK_KHR_cooperative_matrix` is present** (the pipeline
is `Option`, gated by `has_coop_matrix`); the coop-tile shape constraints are checked by
`matmul_coop_ok` (`fuel-vulkan-backend/src/lib.rs:2824`), and shapes that fail fall back to the
`matmul_small_*` scalar kernels. Stride-aware via `MatmulParams`; GQA via `n_rep`; output `C[batch,
M, N]` row-major contiguous. **Numerics:** both A (f32) and B (bf16) are reduced to **f16** before
the tensor-core multiply — the f32 A operand loses precision to f16 on load — while the accumulation
is f32. This is a real precision reduction relative to the scalar f32 `matmul`, declared here so the
planner's precision pre-filter can see it.

```fkc
kernel: matmul_coop
op_kind: MatMul
blurb: "Cooperative-matrix GEMM: f32 A x bf16 B -> f32 C; A/B downcast to f16 on load, f32 accum; VK_KHR_cooperative_matrix only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_coop"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "stride-aware via Params; coop-tile shape constraints apply (matmul_coop_ok); f32 A downcast to f16 on shared load."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B downcast to f16 on shared load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop-tile constraint (matmul_coop_ok)" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop-tile constraint" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop-tile constraint" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false    # tensor-core coop-matrix accumulation; warp/tile order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "A (f32) AND B (bf16) DOWNCAST to f16 before the tensor-core multiply (f32 A loses precision to f16); f32 accumulator. Real precision reduction vs scalar f32 matmul; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_coop_bf16_bf16  (cooperative-matrix bf16 × bf16 GEMM → f32)

GLSL cooperative-matrix GEMM: bf16 A × bf16 B → **f32** C. Both operands are downcast bf16→f16 on
the shared load; the tensor-core multiply runs with an **f32 accumulator** (coop tile M=N=K=16).
`Option` pipeline — dispatched only when `VK_KHR_cooperative_matrix` is present (wrapper
`matmul_bf16_bf16_f32_bytes`, `fuel-vulkan-backend/src/lib.rs:2790`). Stride-aware via `MatmulParams`;
coop-tile constraints apply (fallback to `matmul_small_bf16_bf16_f32`); GQA via `n_rep`. Output
`C[batch, M, N]` row-major contiguous. **Numerics:** bf16 inputs are reduced to f16 (a narrowing of
the 7-bit bf16 mantissa onto f16's 10-bit mantissa is exact in mantissa but the bf16→f16 path is a
re-encode of the value; per the inventory the inputs "downcast bf16→f16 on load"), f32 accumulate.
Dispatch key `(MatMul, [BF16, BF16, F32], Vulkan)`.

```fkc
kernel: matmul_coop_bf16_bf16
op_kind: MatMul
blurb: "Cooperative-matrix GEMM: bf16 A x bf16 B -> f32 C; both downcast bf16->f16 on load, f32 accum; VK_KHR_cooperative_matrix only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_coop_bf16_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 A downcast to f16 on shared load; stride-aware via Params; coop-tile constraints apply."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B downcast to f16 on shared load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop-tile constraint (matmul_coop_ok)" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop-tile constraint" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop-tile constraint" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 A and B downcast to f16 on load; f32 accumulator; tensor-core accumulation order not pinned; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_coop_f16_f16  (cooperative-matrix f16 × f16 GEMM → f32)

GLSL cooperative-matrix GEMM: f16 A × f16 B → **f32** C, with **native f16 inputs** (no downcast —
f16 is already the tensor-core input type) and an **f32 accumulator** (coop tile M=N=K=16). `Option`
pipeline — dispatched only when `VK_KHR_cooperative_matrix` is present (wrappers
`matmul_f16_f16_f32_bytes`, `fuel-vulkan-backend/src/lib.rs:3118`; `matmul_half_half_f32_coop_bytes`,
`:3154`). Stride-aware via `MatmulParams`; coop-tile constraints apply (fallback to
`matmul_small_f16_f16_f32`); GQA via `n_rep`. Output `C[batch, M, N]` row-major contiguous. The f32
accumulation is the load-bearing precision invariant; f16 inputs are consumed natively. Dispatch key
`(MatMul, [F16, F16, F32], Vulkan)`.

```fkc
kernel: matmul_coop_f16_f16
op_kind: MatMul
blurb: "Cooperative-matrix GEMM: f16 A x f16 B -> f32 C; native f16 inputs, f32 accum; VK_KHR_cooperative_matrix only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_coop_f16_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "native f16 input (no downcast); stride-aware via Params; coop-tile constraints apply."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "native f16 input; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop-tile constraint (matmul_coop_ok)" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop-tile constraint" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop-tile constraint" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "native f16 inputs (no input downcast); f32 accumulator; tensor-core accumulation order not pinned; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_coop_bf16_bf16_bf16  (cooperative-matrix bf16 × bf16 GEMM → bf16, downcast store)

GLSL cooperative-matrix GEMM with a **downcast store** to bf16: bf16 A × bf16 B → **bf16** C — the
half-precision inference chain where the output stays bf16. Inputs downcast bf16→f16 on the shared
load; the f32 accumulator is staged to shared memory and then **packed (narrowed) to bf16 on store**
(packed-u32 output). `Option` pipeline — dispatched only when `VK_KHR_cooperative_matrix` is present
(wrappers `matmul_bf16_bf16_bf16_bytes`, `fuel-vulkan-backend/src/lib.rs:2936`;
`matmul_half_half_half_coop_bytes`, `:2972`). Stride-aware via `MatmulParams`; coop-tile constraints
apply (fallback to `matmul_small_bf16_bf16_bf16`); GQA via `n_rep`. Output `C[batch, M, N]` row-major
contiguous, narrowed on store. Dispatch key `(MatMul, [BF16, BF16, BF16], Vulkan)`.

```fkc
kernel: matmul_coop_bf16_bf16_bf16
op_kind: MatMul
blurb: "Cooperative-matrix GEMM: bf16 A x bf16 B -> bf16 C (downcast store); inputs downcast to f16, f32 accum staged then packed; VK_KHR_cooperative_matrix only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_coop_bf16_bf16_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 A downcast to f16 on shared load; stride-aware via Params; coop-tile constraints apply."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B downcast to f16 on shared load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop-tile constraint (matmul_coop_ok)" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop-tile constraint" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop-tile constraint" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # bf16 in, bf16 out (downcast store)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 inputs downcast to f16 on load; f32 accumulator staged to shared mem then NARROWED to bf16 on store (packed-u32); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_coop_f16_f16_f16  (cooperative-matrix f16 × f16 GEMM → f16, downcast store)

GLSL cooperative-matrix GEMM with a **downcast store** to f16: f16 A × f16 B → **f16** C — the
half-precision inference chain with native f16 inputs and a f16 output. Inputs consumed natively
(f16 is the tensor-core type); the f32 accumulator is staged to shared memory and then **narrowed to
f16 on store**. `Option` pipeline — dispatched only when `VK_KHR_cooperative_matrix` is present
(wrappers `matmul_f16_f16_f16_bytes`, `fuel-vulkan-backend/src/lib.rs:3085`;
`matmul_half_half_half_coop_bytes`, `:2972`). Stride-aware via `MatmulParams`; coop-tile constraints
apply (fallback to `matmul_small_f16_f16_f16`); GQA via `n_rep`. Output `C[batch, M, N]` row-major
contiguous, narrowed on store. Dispatch key `(MatMul, [F16, F16, F16], Vulkan)`.

```fkc
kernel: matmul_coop_f16_f16_f16
op_kind: MatMul
blurb: "Cooperative-matrix GEMM: f16 A x f16 B -> f16 C (downcast store); native f16 inputs, f32 accum staged then narrowed; VK_KHR_cooperative_matrix only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_coop_f16_f16_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "native f16 input (no downcast); stride-aware via Params; coop-tile constraints apply."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "native f16 input; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop-tile constraint (matmul_coop_ok)" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop-tile constraint" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop-tile constraint" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # f16 in, f16 out (downcast store)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "native f16 inputs; f32 accumulator staged to shared mem then NARROWED to f16 on store; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_small_bf16_bf16_f32  (scalar-accumulator GEMM fallback, bf16 × bf16 → f32)

GLSL scalar-accumulator GEMM fallback (one thread per output element) for bf16 A × bf16 B → **f32**
C. This handles **any shape** when the coop-matrix tile constraints fail (`M < 16`, `M % 16 != 0`,
`N % 16 != 0`, or M==1 gemv) — the catch-all behind the `matmul_coop_bf16_bf16` tensor-core path
(`matmul_small_bf16_bf16_f32.glsl:32`, shared inner `matmul_small_half_inner`,
`fuel-vulkan-backend/src/lib.rs:2834`). Stride-aware via `sa_*/sb_*`; 16×16 workgroup, grid
`ceil(N/16) × ceil(M/16) × batch`; bf16 operands unpacked per-load from `uint16_t` typed buffers
into an **f32 accumulator**. GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous. Dispatch
key `(MatMul, [BF16, BF16, F32], Vulkan)` — the always-available scalar sibling to the coop kernel at
that key (selected when coop is absent or the tile shape fails).

```fkc
kernel: matmul_small_bf16_bf16_f32
op_kind: MatMul
blurb: "Scalar-accumulator GEMM fallback: bf16 A x bf16 B -> f32 C; any shape; one thread/output, f32 accum; stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_small_bf16_bf16_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 unpacked per-load from uint16_t buffer; stride-aware (sa_batch/sa_row/sa_col); any shape."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 unpacked per-load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false    # scalar f32 accumulation per output thread; no cross-thread reduction, but FMA contraction not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 unpacked per-load to f32; f32 scalar accumulator (one thread per output elem); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_small_bf16_bf16_bf16  (scalar-accumulator GEMM fallback, bf16 × bf16 → bf16)

GLSL scalar-accumulator GEMM fallback for bf16 A × bf16 B → **bf16** C — the downcast-store sibling
of `matmul_small_bf16_bf16_f32`, the any-shape fallback behind `matmul_coop_bf16_bf16_bf16` when the
coop tile constraints fail (shared inner `matmul_small_half_inner`,
`fuel-vulkan-backend/src/lib.rs:2834`; family file `matmul_small_bf16_bf16_f32.glsl:32`). One thread
per output element; bf16 operands unpacked per-load into an **f32 accumulator**, then the result is
**narrowed to bf16 on store** (packed-u32). Stride-aware via `sa_*/sb_*`; 16×16 workgroup, grid
`ceil(N/16) × ceil(M/16) × batch`; GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous.
Dispatch key `(MatMul, [BF16, BF16, BF16], Vulkan)` — the scalar sibling to
`matmul_coop_bf16_bf16_bf16` at that key.

```fkc
kernel: matmul_small_bf16_bf16_bf16
op_kind: MatMul
blurb: "Scalar-accumulator GEMM fallback: bf16 A x bf16 B -> bf16 C (downcast store); any shape; f32 accum then narrow; stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_small_bf16_bf16_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 unpacked per-load from uint16_t buffer; stride-aware (sa_batch/sa_row/sa_col); any shape."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 unpacked per-load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # bf16 in, bf16 out (downcast store)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "bf16 unpacked per-load to f32; f32 scalar accumulator; result NARROWED to bf16 on store (packed-u32); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_small_f16_f16_f32  (scalar-accumulator GEMM fallback, f16 × f16 → f32)

GLSL scalar-accumulator GEMM fallback for f16 A × f16 B → **f32** C — the any-shape fallback behind
`matmul_coop_f16_f16` when the coop tile constraints fail (`M < 16`, `M % 16 != 0`, `N % 16 != 0`, or
M==1 gemv) (shared inner `matmul_small_half_inner`, `fuel-vulkan-backend/src/lib.rs:2834`; family
file `matmul_small_bf16_bf16_f32.glsl:32`). One thread per output element; f16 operands unpacked
per-load from `uint16_t` typed buffers into an **f32 accumulator**. Stride-aware via `sa_*/sb_*`;
16×16 workgroup, grid `ceil(N/16) × ceil(M/16) × batch`; GQA via `n_rep`; output `C[batch, M, N]`
row-major contiguous. Dispatch key `(MatMul, [F16, F16, F32], Vulkan)` — the always-available scalar
sibling to `matmul_coop_f16_f16` at that key.

```fkc
kernel: matmul_small_f16_f16_f32
op_kind: MatMul
blurb: "Scalar-accumulator GEMM fallback: f16 A x f16 B -> f32 C; any shape; one thread/output, f32 accum; stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_small_f16_f16_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "f16 unpacked per-load from uint16_t buffer; stride-aware (sa_batch/sa_row/sa_col); any shape."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "f16 unpacked per-load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f16 unpacked per-load to f32; f32 scalar accumulator (one thread per output elem); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_small_f16_f16_f16  (scalar-accumulator GEMM fallback, f16 × f16 → f16)

GLSL scalar-accumulator GEMM fallback for f16 A × f16 B → **f16** C — the downcast-store sibling of
`matmul_small_f16_f16_f32`, the any-shape fallback behind `matmul_coop_f16_f16_f16` when the coop
tile constraints fail (shared inner `matmul_small_half_inner`,
`fuel-vulkan-backend/src/lib.rs:2834`; family file `matmul_small_bf16_bf16_f32.glsl:32`). One thread
per output element; f16 operands unpacked per-load into an **f32 accumulator**, then **narrowed to
f16 on store**. Stride-aware via `sa_*/sb_*`; 16×16 workgroup, grid `ceil(N/16) × ceil(M/16) ×
batch`; GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous. Dispatch key
`(MatMul, [F16, F16, F16], Vulkan)` — the scalar sibling to `matmul_coop_f16_f16_f16` at that key.

```fkc
kernel: matmul_small_f16_f16_f16
op_kind: MatMul
blurb: "Scalar-accumulator GEMM fallback: f16 A x f16 B -> f16 C (downcast store); any shape; f32 accum then narrow; stride-aware; GQA via n_rep."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_small_f16_f16_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "f16 unpacked per-load from uint16_t buffer; stride-aware (sa_batch/sa_row/sa_col); any shape."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "f16 unpacked per-load; sb_batch/sb_row/sb_col."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]" }
      n: { kind: usize, constraint: "== rhs.dim[-1]" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # f16 in, f16 out (downcast store)
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
    - { when: "any_input_strided", class: gemm_like, note: "stride walk, no fixup" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f16 unpacked per-load to f32; f32 scalar accumulator; result NARROWED to f16 on store; not bit-stable cross-hardware."

determinism: nondeterministic
```

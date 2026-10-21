---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — matmul family kernel contracts

Batched GEMM / GEMV kernels for the Vulkan backend (crate `vulkan`, family `matmul`). Every kernel
here implements `OpKind::MatMul` (`fuel-ir/src/dispatch.rs`) and consumes the `OpParams::Matmul`
variant (`m`, `n`, `k`, per-operand batch dims; the Vulkan `MatmulParams` block carries the
per-operand strides `sa_*/sb_*`, the output batch stride `sc_batch`, and a GQA repeat factor
`n_rep`). The dispatch key is `(OpKind::MatMul, [A_dtype, B_dtype, C_dtype], Vulkan) + kernel_source`
(§3.2, §12.1); the mixed-precision variants (f32×bf16, bf16×bf16, f16×f16) are distinguished by the
per-operand dtype slots in the key, **not** by a separate op kind.

**As-built binding model — one wrapper `KernelRef` per dtype-combo key (route-picking is *internal*).**
Production registers exactly **six** `KernelRef`s here, one per `(MatMul, [lhs,rhs,out], Vulkan)`
key, and each section below is that ONE registrable binding. The finer-grained Slang kernels the
inventory documents (`matvec` / register-tiled `matmul` / shared-memory `matmul_tiled` for f32;
`matvec_bf16_b` / `matmul_tiled_bf16_b` / `matmul_coop*` for the mixed combos; the `matmul_small_*`
scalar-accumulator fallbacks) are **route-picker alternatives *inside* each wrapper**
(`VulkanBackend::matmul_*_bytes` selects by `M` and by cooperative-matrix availability), not distinct
bindings in the table — so they are described in each wrapper's prose, not as separate `##` sections
(a per-kernel section per internal alternative would register duplicate `KernelRef`s at one key,
which `register_into`'s `finalize` rejects). This mirrors the cast family's per-pair precedent: one
registrable section per binding, several sharing an algorithm.

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** Although the
underlying Slang kernels address A/B through their per-operand strides, the production registrations
are `register_with_precision` (no strided caps) — i.e. `awkward_layout_strategy: requires_contiguous`
(`strided_input == false`): the coop / vec4 / cooperative-matrix loads require canonical row-major
tiles, so the planner auto-Contiguizes a transposed / sliced / non-zero-offset operand *first* and
sums the `Op::Contiguize` cost (§4.3). GQA is expressed via `n_rep` (`b_off = (batch / n_rep) *
sb_batch`), **not** a stride-0 broadcast axis. Output is always freshly-allocated **contiguous**
row-major (`C[batch, M, N]`), no aliasing, not in-place (the universal output-contiguity rule).

**Route picking — the f32 wrapper.** `matmul_f32` (`VulkanBackend::matmul_f32_bytes`, picker at
`fuel-vulkan-backend/src/lib.rs:3830`) selects `matvec` for `M == 1`, the register-tiled `matmul`
for `1 < M < 32`, and the shared-memory `matmul_tiled` for `M >= 32`. All three accumulate in f32
over a register / shared-memory tile or a subgroup reduction; the wrapper is one binding.

**Capability-gated coop pipelines.** The mixed-precision wrappers dispatch cooperative-matrix
(tensor-core) kernels **only when `VK_KHR_cooperative_matrix` is present** (`has_coop_matrix`; the
pipeline objects are `Option`). The coop tile is M=N=K=16 with an f32 accumulator; when the coop
constraints fail (`M < 16`, `M % 16 != 0`, `N % 16 != 0`, or the M==1 gemv), the wrapper falls back
to the `matmul_small_*` scalar-accumulator kernel (any shape). Both paths are the SAME binding.

**Cost provenance.** Every cost block is `judge_measured`: the Judge bootstraps it (§4.4). The FLOPs
hint `2 * batch * m * n * k` is the genuinely derivable GEMM flop count (one multiply + one add per
inner-product term, summed over batches). No other coefficients are fabricated.

**Determinism (corrected 2026-06-18).** Every matmul / matvec kernel accumulates in f32 over a
register / shared-memory tile or a subgroup reduction whose FADD / subgroup order is
**scheduler-dependent**, so none is bit-stable even on a re-run on the same device. These are
therefore `determinism: nondeterministic` with `bit_stable_on_same_hardware: false` and an audited
`none(reason)` precision (no silent unaudited nondeterminism) — matching the flash-attn and qmatmul
precedent and §10 rule 9. (An earlier revision mis-declared `same_hardware_bitwise`, which
contradicted its own `bit_stable_on_same_hardware: false`; that is the retired hand-written
`VULKAN_MATMUL_PRECISION` / `VULKAN_MATMUL_TENSORCORE_PRECISION` posture. The Judge audits the
corrected seed.)

---

## matmul_f32  (f32 GEMM/GEMV wrapper; matvec / reg-tile / tiled route-pick)

f32 A × f32 B → f32 C. The production f32 matmul binding (`matmul::matmul_f32` →
`VulkanBackend::matmul_f32_bytes`): route-picks `matvec` (`M == 1`, subgroup-reduced dot, one
workgroup/col), the register-tiled 4×4 `matmul` (`1 < M < 32`, no shared memory), or the
shared-memory blocked 64×64 `matmul_tiled` (`M >= 32`, BK=16). f32 multiply-accumulate throughout;
GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous. Contiguous-only at the binding
boundary (the deleted `register_with_precision` reg) — a strided / transposed / offset operand is
auto-Contiguized by the planner first.

```fkc
kernel: matmul_f32
op_kind: MatMul
blurb: "f32 GEMM/GEMV wrapper C=A@B; internal matvec(M==1)/reg-tile(M<32)/tiled(M>=32) route-pick; GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "f32 A; auto-Contiguized before the wrapper (coop/vec4 loads need row-major); GQA via n_rep not stride-0."
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "f32 B; rhs batch slot read as lhs_slot / n_rep."
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
      dtype_rule: passthrough(lhs)        # f32 in, f32 out; key pins [F32,F32,F32]
      shape_rule: matmul(lhs, rhs)        # lhs_batch ++ [m, n]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: gemm_like
  flops: "2 * batch * m * n * k"

precision:
  bit_stable_on_same_hardware: false    # f32 tile / subgroup accumulation; scheduler/subgroup order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "f32 multiply-accumulate (matvec/reg-tile/tiled); accumulation order tile/subgroup-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_f32_bf16_b  (mixed-precision f32 × bf16 → f32 wrapper)

f32 A × **bf16 B** → f32 C — the decode hot path for bf16 weights (`matmul::matmul_f32_bf16_b` →
`VulkanBackend::matmul_f32_bf16_b_bytes`). Internally route-picks `matvec_bf16_b` (`M == 1`, subgroup
dot), `matmul_tiled_bf16_b` (`M > 1` tiled), or the cooperative-matrix `matmul_coop` (tensor-core,
when `VK_KHR_cooperative_matrix` is present). bf16 B is unpacked to f32 by the **exact** `bits << 16`
widening (BF16 is the upper 16 bits of F32, no mantissa loss); the multiply-accumulate runs in f32.
The `matmul_coop` path additionally downcasts the f32 A to f16 on the shared load (a real precision
reduction, audited below). GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous.
Contiguous-only binding. Dispatch key `(MatMul, [F32, BF16, F32], Vulkan)` — distinct from the
all-f32 key by the B dtype slot.

```fkc
kernel: matmul_f32_bf16_b
op_kind: MatMul
blurb: "Mixed-precision f32 A x bf16 B -> f32 C wrapper; matvec_bf16_b/matmul_tiled_bf16_b/matmul_coop route-pick; exact bf16 unpack (bits<<16); GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_f32_bf16_b"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "f32 A; auto-Contiguized first; coop path downcasts f32 A to f16 on the shared load."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B unpacked by bits<<16 (EXACT widening); coop path downcasts to f16 on load; sb_* stride model."
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
      dtype_rule: fixed(F32)              # f32 C regardless of bf16 B; key [F32,BF16,F32]
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path when VK_KHR_cooperative_matrix present" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

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
  notes: "bf16 B unpacked via bits<<16 (EXACT); f32 accumulate. Coop path downcasts f32 A AND bf16 B to f16 before the tensor-core multiply (wider ULP than scalar f32); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_bf16_bf16_f32  (cooperative-matrix bf16 × bf16 → f32 wrapper)

bf16 A × bf16 B → **f32** C (`matmul::matmul_bf16_bf16_f32` →
`VulkanBackend::matmul_bf16_bf16_f32_bytes`). Both operands downcast bf16→f16 on the shared load; the
cooperative-matrix (tensor-core) multiply runs with an **f32 accumulator** (coop tile M=N=K=16), when
`VK_KHR_cooperative_matrix` is present. When the coop tile constraints fail, the wrapper falls back
to the `matmul_small_bf16_bf16_f32` scalar-accumulator kernel (one thread per output element, any
shape, f32 accumulator). GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous.
Contiguous-only binding. Dispatch key `(MatMul, [BF16, BF16, F32], Vulkan)`.

```fkc
kernel: matmul_bf16_bf16_f32
op_kind: MatMul
blurb: "Cooperative-matrix bf16 A x bf16 B -> f32 C wrapper; f32 accum; matmul_small_bf16_bf16_f32 scalar fallback; GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_bf16_bf16_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 A downcast to f16 on shared load (coop) or unpacked to f32 (scalar fallback); auto-Contiguized first."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B; sb_* stride model; coop-tile constraints gate the tensor-core path."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop path needs m % 16 == 0" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop path needs n % 16 == 0" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop path needs k >= 16" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # f32 C; key [BF16,BF16,F32]
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path; scalar fallback otherwise" }
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
  notes: "bf16 A and B downcast to f16 on load (coop) / unpacked to f32 (scalar); f32 accumulator; tensor-core / scalar accumulation order not pinned; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_bf16_bf16_bf16  (cooperative-matrix bf16 × bf16 → bf16 wrapper, downcast store)

bf16 A × bf16 B → **bf16** C (`matmul::matmul_bf16_bf16_bf16` →
`VulkanBackend::matmul_bf16_bf16_bf16_bytes`) — the end-to-end bf16 inference chain where the output
stays bf16. Inputs downcast bf16→f16 on the shared load; the f32 accumulator is staged to shared
memory and **narrowed to bf16 on store** (packed-u32). Coop tensor-core path when
`VK_KHR_cooperative_matrix` is present; `matmul_small_bf16_bf16_bf16` scalar fallback (f32 accumulate
then narrow) otherwise. GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous, narrowed on
store. Contiguous-only binding. Dispatch key `(MatMul, [BF16, BF16, BF16], Vulkan)`.

```fkc
kernel: matmul_bf16_bf16_bf16
op_kind: MatMul
blurb: "Cooperative-matrix bf16 A x bf16 B -> bf16 C wrapper (downcast store); f32 accum staged then narrowed; matmul_small_bf16_bf16_bf16 scalar fallback; GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_bf16_bf16_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "bf16 A downcast to f16 on shared load (coop) / unpacked to f32 (scalar); auto-Contiguized first."
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "bf16 B; sb_* stride model; coop-tile constraints gate the tensor-core path."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop path needs m % 16 == 0" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop path needs n % 16 == 0" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop path needs k >= 16" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # bf16 in, bf16 out (downcast store); key [BF16,BF16,BF16]
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path; scalar fallback otherwise" }
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
  notes: "bf16 inputs downcast to f16 on load; f32 accumulator staged then NARROWED to bf16 on store (packed-u32); not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_f16_f16_f16  (cooperative-matrix f16 × f16 → f16 wrapper, downcast store)

f16 A × f16 B → **f16** C (`matmul::matmul_f16_f16_f16` →
`VulkanBackend::matmul_f16_f16_f16_bytes`) — the f16 inference chain with native f16 inputs (no input
downcast — f16 is the tensor-core input type) and an f16 output. The f32 accumulator is staged to
shared memory and **narrowed to f16 on store**. Coop tensor-core path when
`VK_KHR_cooperative_matrix` is present; `matmul_small_f16_f16_f16` scalar fallback (f32 accumulate
then narrow) otherwise. GQA via `n_rep`; output `C[batch, M, N]` row-major contiguous, narrowed on
store. Contiguous-only binding. Dispatch key `(MatMul, [F16, F16, F16], Vulkan)`.

```fkc
kernel: matmul_f16_f16_f16
op_kind: MatMul
blurb: "Cooperative-matrix f16 A x f16 B -> f16 C wrapper (downcast store); native f16 inputs, f32 accum staged then narrowed; matmul_small_f16_f16_f16 scalar fallback; GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_f16_f16_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "native f16 A (no downcast); auto-Contiguized first."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "native f16 B; sb_* stride model; coop-tile constraints gate the tensor-core path."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop path needs m % 16 == 0" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop path needs n % 16 == 0" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop path needs k >= 16" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)        # f16 in, f16 out (downcast store); key [F16,F16,F16]
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path; scalar fallback otherwise" }
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
  notes: "native f16 inputs; f32 accumulator staged then NARROWED to f16 on store; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## matmul_f16_f16_f32  (cooperative-matrix f16 × f16 → f32 wrapper)

f16 A × f16 B → **f32** C (`matmul::matmul_f16_f16_f32` →
`VulkanBackend::matmul_f16_f16_f32_bytes`) — native f16 inputs, f32 output. The f32 accumulation is
the load-bearing precision invariant; f16 inputs are consumed natively. Coop tensor-core path when
`VK_KHR_cooperative_matrix` is present; `matmul_small_f16_f16_f32` scalar fallback (one thread per
output element, f32 accumulator) otherwise. GQA via `n_rep`; output `C[batch, M, N]` row-major
contiguous. Contiguous-only binding. Dispatch key `(MatMul, [F16, F16, F32], Vulkan)`.

```fkc
kernel: matmul_f16_f16_f32
op_kind: MatMul
blurb: "Cooperative-matrix f16 A x f16 B -> f32 C wrapper; native f16 inputs, f32 accum; matmul_small_f16_f16_f32 scalar fallback; GQA via n_rep; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_f16_f16_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-1]=k"
      notes: "native f16 A (no downcast); auto-Contiguized first."
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: "dim[-2]=k"
      notes: "native f16 B; sb_* stride model; coop-tile constraints gate the tensor-core path."
  op_params:
    variant: Matmul
    fields:
      m: { kind: usize, constraint: "== lhs.dim[-2]; coop path needs m % 16 == 0" }
      n: { kind: usize, constraint: "== rhs.dim[-1]; coop path needs n % 16 == 0" }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]; coop path needs k >= 16" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", constraint: "per-axis == lhs_batch_dims OR GQA-divisible; packed into n_rep" }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)              # f32 C; key [F16,F16,F32]
      shape_rule: matmul(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: gemm_like, note: "coop tensor-core path; scalar fallback otherwise" }
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
  notes: "native f16 inputs (no input downcast); f32 accumulator; tensor-core / scalar accumulation order not pinned; not bit-stable cross-hardware."

determinism: nondeterministic
```

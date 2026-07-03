---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — 2D convolution family kernel contracts

Direct 2D convolution for the Vulkan backend (crate `vulkan`, family `conv`). Every kernel here
implements the PRIMITIVE `OpKind::Conv2D` (`fuel-ir/src/dispatch.rs`) and consumes the
`OpParams::Conv2D` variant (`x_shape`, `w_shape`, `out_shape`, `stride`, `padding`, `dilation`,
`groups`). The dispatch key is `(OpKind::Conv2D, [x_dtype, weight_dtype, out_dtype], Vulkan) +
kernel_source` (§3.2, §12.1); the three variants (f32, bf16, f16) are distinguished by the
per-operand dtype slots in the key, **not** by a separate op kind.

**As-built binding model — one wrapper `KernelRef` per dtype-combo key (im2col + GEMM is *internal*).**
Production registers exactly **three** `KernelRef`s here, one per `(Conv2D, [x, weight, out],
Vulkan)` key, and each section below is that ONE registrable binding. The internal two-stage pipeline
the inventory documents (`conv2d_im2col` NCHW→patches rearrangement, then a GEMM over the reshaped
weight — the f32 path via `matmul_tiled`, the bf16/f16 paths via the cooperative-matrix
`matmul_coop_*`) runs **inside each wrapper** (`VulkanBackend::conv2d_*_bytes`), not as distinct
bindings in the table. The aspirational `fused_op: CONV2D` **im2col-STAGE** sections in
`vulkan/conv-attn-rope.fkc.md` (`conv2d_im2col_f32` / `conv2d_im2col_bf16`, whose `return.patches`
is the intermediate matrix, NOT the conv output) describe that future *fused* decomposition; they are
a SEPARATE concern and do NOT register the primitive `OpKind::Conv2D` binding this file migrates
(mirroring the matmul family's split between the aspirational `dispatch/matmul.fkc.md ::
matmul_mixed_precision` chassis and the production per-combo `vulkan/matmul.fkc.md`).

**Accept surface — x + weight only, NO bias (matches the as-built reg).** The Vulkan conv wrappers
accept 2 inputs (`x [N, Cin, H, W]`, `weight [Cout, Cin/groups, Kh, Kw]`) and one output; a 3-input
(bias-fused) call BAILS at the wrapper (`vulkan_dispatch::conv2d::conv2d_f32: bias-fused conv2d not
supported on Vulkan yet` — the route picker is expected to choose a CPU/CUDA fused-conv alternative
when a bias is present). So — UNLIKE the CPU conv contract's `optional: true` bias that fans a
`[T,T,T,T]` with-bias key — these sections declare NO bias operand and each keys ONLY the 3-slot
`[x, weight, out]`, byte-for-byte the deleted hand-written
`table.register_with_precision(OpKind::Conv2D, &[T, T, T], …)` reg. **Dilation is fixed at `(1, 1)`**
(the wrapper bails on any other dilation); grouped / depthwise convolution IS supported via `groups`.

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** The internal
`conv2d_im2col` reads `x` as canonical row-major NCHW with computed strides, so the production
registrations are `register_with_precision` (no strided caps) — i.e.
`awkward_layout_strategy: requires_contiguous` (`strided_input == false`): the planner
auto-Contiguizes a transposed / sliced / non-zero-offset `x` or `weight` *first* and sums the
`Op::Contiguize` cost (§4.3). Output is always freshly-allocated **contiguous** row-major
`out[N, Cout, H_out, W_out]`, no aliasing, not in-place (the universal output-contiguity rule).

**Cost provenance.** Every cost block is `judge_measured`: the Judge bootstraps it (§4.4). The FLOPs
hint `2 · N·Cout·H_out·W_out · (Cin/groups)·Kh·Kw` is the genuinely derivable dense conv flop count
(one multiply + one add per MAC, summed over the output tensor and the per-group receptive field —
the dense upper bound; the padding skip only reduces it at the borders). No other coefficients are
fabricated; the imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by the
`fill_unset_cost_for_backend` pass at registration.

**Determinism (conservative seed).** Each conv wrapper's internal GEMM stage accumulates in f32 over
a shared-memory / cooperative-matrix tile whose FADD / subgroup order is **scheduler-dependent**, so
none is bit-stable even on a re-run on the same device. These are therefore
`determinism: nondeterministic` with `bit_stable_on_same_hardware: false` and an audited `none(reason)`
precision (no silent unaudited nondeterminism) — matching the matmul / flash-attn precedent and §10
rule 9. (This is the CONSERVATIVE correction of the retired hand-written `VULKAN_MATMUL_PRECISION` /
`VULKAN_MATMUL_TENSORCORE_PRECISION` consts these regs used to carry, which mis-declared
`bit_stable_on_same_hardware: true` — the same over-claim the matmul migration corrected. The Judge
audits the corrected seed.)

---

## conv2d_f32  (2D convolution, f32; im2col + f32 GEMM)

f32 `x` × f32 `weight` → f32 `out` (`conv2d::conv2d_f32` → `VulkanBackend::conv2d_f32_bytes`). The
NCHW `conv2d_im2col` rearranges `x` into a `[batch·groups, cin_per_g·Kh·Kw, H_out·W_out]` patches
matrix (OOB taps zero-filled), then a f32 GEMM against the reshaped weight produces the output; f32
multiply-accumulate throughout. Grouped / depthwise via `groups`; asymmetric stride / padding
supported; **dilation fixed `(1, 1)`** (the wrapper bails otherwise → route picker falls back to
CPU/CUDA). Contiguous-only at the binding boundary (the deleted `register_with_precision` reg) — a
strided / transposed / offset operand is auto-Contiguized by the planner first. Dispatch key
`(Conv2D, [F32, F32, F32], Vulkan)`.

```fkc
kernel: conv2d_f32
op_kind: Conv2D
blurb: "Direct 2D convolution (f32) via internal im2col + f32 GEMM; x[N,Cin,H,W], weight[Cout,Cin/groups,Kh,Kw], grouped/depthwise; dilation==(1,1); no bias; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::conv2d_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H_in, W_in] NCHW
    - name: weight
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
  op_params:
    variant: Conv2D                        # OpParams::Conv2D (primitive namespace; §3.7)
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; kernel trusts this geometry" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)", constraint: "== (1, 1) (wrapper bails otherwise → CPU/CUDA fallback)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # f32 in, f32 out; key pins [F32, F32, F32]
      shape_rule: conv2d(params)           # [N, Cout, H_out, W_out] from OpParams::Conv2D geometry (§5.2)
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated contiguous buffer

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "groups == 1", class: conv }        # dense (non-grouped) convolution
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"

precision:
  bit_stable_on_same_hardware: false    # internal f32 GEMM tile / subgroup accumulation; scheduler/subgroup order not pinned
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "im2col (bit-exact rearrange + zero fill) then f32 GEMM multiply-accumulate; GEMM accumulation order tile/subgroup-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## conv2d_bf16  (2D convolution, bf16; im2col_bf16 + cooperative-matrix bf16 GEMM)

bf16 `x` × bf16 `weight` → bf16 `out` (`conv2d::conv2d_bf16` → `VulkanBackend::conv2d_bf16_bytes`).
The bf16 `conv2d_im2col_bf16` (byte-level packed-u16 rearrange, OOB zero-fill) feeds the
cooperative-matrix bf16 GEMM (`matmul_coop_bf16_bf16_bf16`, f32 accumulator, narrowed to bf16 on
store). COOP-ONLY shape constraints: `Cout % 16 == 0` AND `(H_out · W_out) % 16 == 0` — the wrapper
bails on smaller shapes (route picker falls through to f32 conv2d via `Cast`). Grouped / depthwise via
`groups`; **dilation fixed `(1, 1)`**. Wider ULP than the f32 conv (the coop tensor-core inputs are
f16-narrowed); tensor-core precision reflected below. Contiguous-only binding. Dispatch key
`(Conv2D, [BF16, BF16, BF16], Vulkan)`.

```fkc
kernel: conv2d_bf16
op_kind: Conv2D
blurb: "Direct 2D convolution (bf16) via im2col_bf16 + cooperative-matrix bf16 GEMM (f32 accum, narrow on store); coop constraints Cout%16==0 && (Hout*Wout)%16==0; grouped/depthwise; dilation==(1,1); no bias; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::conv2d_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H_in, W_in] NCHW, packed u16-in-u32
    - name: weight
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
  op_params:
    variant: Conv2D
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; coop needs Cout%16==0 && (Hout*Wout)%16==0" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)", constraint: "== (1, 1) (wrapper bails otherwise → CPU/CUDA fallback)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # bf16 in, bf16 out (narrow on store); key pins [BF16, BF16, BF16]
      shape_rule: conv2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "im2col_bf16 (byte-exact rearrange + zero fill) then cooperative-matrix GEMM: bf16 inputs downcast to f16 on the tensor-core load, f32 accumulator narrowed to bf16 on store (wider ULP than f32 conv); accumulation order tile/subgroup-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## conv2d_f16  (2D convolution, f16; im2col + cooperative-matrix f16 GEMM)

f16 `x` × f16 `weight` → f16 `out` (`conv2d::conv2d_f16` → `VulkanBackend::conv2d_f16_bytes`) —
sibling of `conv2d_bf16`: it reuses the 2-byte dtype-opaque im2col shuffle then the cooperative-matrix
f16 GEMM (`matmul_coop_f16_f16_f16`, f32 accumulator, narrowed to f16 on store). Same COOP-ONLY shape
constraints (`Cout % 16 == 0` AND `(H_out · W_out) % 16 == 0`; bails otherwise → f32 conv2d via
`Cast`), grouped / depthwise via `groups`, **dilation fixed `(1, 1)`**. Differs from bf16 only in the
IEEE half-precision storage format. Contiguous-only binding. Dispatch key
`(Conv2D, [F16, F16, F16], Vulkan)`.

```fkc
kernel: conv2d_f16
op_kind: Conv2D
blurb: "Direct 2D convolution (f16) via im2col + cooperative-matrix f16 GEMM (f32 accum, narrow on store); coop constraints Cout%16==0 && (Hout*Wout)%16==0; grouped/depthwise; dilation==(1,1); no bias; contiguous-only binding."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::conv2d_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [N, Cin, H_in, W_in] NCHW
    - name: weight
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [Cout, Cin/groups, Kh, Kw]
      shape_constraint: "divisible(x.dim[1], weight.dim[1]); weight.dim[1] == x.dim[1] / groups"
  op_params:
    variant: Conv2D
    fields:
      x_shape:   { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:   { kind: "[usize; 4]", note: "[Cout, Cin/groups, Kh, Kw]" }
      out_shape: { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; coop needs Cout%16==0 && (Hout*Wout)%16==0" }
      stride:    { kind: "(usize, usize)" }
      padding:   { kind: "(usize, usize)" }
      dilation:  { kind: "(usize, usize)", constraint: "== (1, 1) (wrapper bails otherwise → CPU/CUDA fallback)" }
      groups:    { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)           # f16 in, f16 out (narrow on store); key pins [F16, F16, F16]
      shape_rule: conv2d(params)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", class: conv }
    - { when: "depthwise", note: "groups == Cin, Cout == Cin special case" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: conv
  flops: "2 * out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * (x_shape[1] / groups) * w_shape[2] * w_shape[3]"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent FADD/subgroup order, non-associative f32 (§4.8)
  notes: "im2col (2-byte dtype-opaque rearrange + zero fill) then cooperative-matrix GEMM: native f16 inputs, f32 accumulator narrowed to f16 on store; accumulation order tile/subgroup-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

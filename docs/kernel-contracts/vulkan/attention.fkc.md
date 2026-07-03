---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"  # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — FlashAttention (naive single-pass) family kernel contracts

The Vulkan backend's **FlashAttention** primitives (crate `vulkan`, family `attention`): the naive
single-pass multi-head SDPA forward (`OpKind::FlashAttn`) plus the three recompute-based backward
selectors (`OpKind::FlashAttnBackward{Q,K,V}`, `fuel-ir/src/dispatch.rs`). These are the PRIMITIVE
`op_kind:` bindings production actually wires onto the `KernelBindingTable`; they are a SEPARATE
concern from the aspirational `fused_op: FLASH_ATTN` / `FLASH_ATTN_BACKWARD_*` decompositions (and
the tiled FA-2 `flash_attention` kernel) in `vulkan/conv-attn-rope.fkc.md`, which describe a future
FUSED-registry seam — NOT how these primitive bindings register. Both files name the SAME production
`entry_point`s (`fuel_vulkan_backend::fkc::flash_attn_*`); the difference is the registration
namespace (`op_kind: FlashAttn` here, `fused_op: FLASH_ATTN` there), the same split the matmul family
draws between the aspirational `dispatch/matmul.fkc.md :: matmul_mixed_precision` chassis and the
production per-combo `vulkan/matmul.fkc.md`.

**As-built binding model — per-(op, dtype) sections (the conv2d precedent).** Production registers
exactly **12** `KernelRef`s in this family: the forward at THREE per-dtype wrappers
(`flash_attn::flash_attn_{f32,bf16,f16}`) and each of the three backward selectors at ONE f32-only
wrapper (`flash_attn::flash_attn_backward_{q,k,v}_f32`) — the bf16/f16 backward kernels are a future
session. Because the forward wrappers are DISTINCT per dtype (not one dtype-agnostic umbrella), each
dtype is authored as its OWN single-dtype-per-operand section (the conv2d `conv2d_{f32,bf16,f16}`
precedent — and the CPU attention contract's per-dtype `flash_attn_*` sections), resolving its
`entry_point` AS-IS through [`crate::fkc::VulkanLinkRegistry`] (no dtype-fan suffix). The forward
attends the K/V axis with a per-`(b, h, q)` workgroup that materializes one `[Sk]` score row in
shared memory (`Sk ≤ 4096`, `D ≤ 256`); it is NOT the tiled FA-2 `flash_attention`.

**Accept surface — `alibi_slopes` as an `optional: true` LAST input (dual-key fan).** Forward inputs
are `q [B, Hq, Sq, D]`, `k`/`v [B, Hkv, Sk, D]` (GQA, `Hq % Hkv == 0`) plus an OPTIONAL 4th input
`alibi_slopes [Hq]`; backward adds the upstream gradient `do [B, Hq, Sq, D]` (so `q, k, v, do` +
optional `alibi_slopes`). The importer's optional-last fan (§3.4) registers EACH section as TWO keys
— one OMITTING the optional operand and one INCLUDING it — byte-for-byte the deleted hand-written
regs' dual `[q,k,v,out]` / `[q,k,v,alibi,out]` (forward) and `[q,k,v,do,out]` /
`[q,k,v,do,alibi,out]` (backward) shapes. Output dtype is `passthrough(q)`, so the key's trailing
output slot pins q's dtype (all-F32 for backward; per-dtype for forward).

**Shared `OpParams::FlashAttn` carrier.** The forward AND all three backward selectors carry the
single `OpParams::FlashAttn` variant (`fuel-dispatch/src/kernel.rs`) — there is no dedicated backward
variant, so every section names `op_params.variant: FlashAttn` (the CPU attention precedent). The
geometry (`b, hq, hkv, sq, sk, d`) rides on the operand SHAPES / `KernelRef`, not the variant fields.
The Vulkan v1 wrappers **bail** on `window_size_left` / `window_size_right` / `softcap` (route picker
falls back to CPU/CUDA), and the forward requires `k_len == sk` (STATIC full-extent only — a runtime
live-prefix `k_len < sk`, capacity-K decode, bails to CPU/CUDA), so — UNLIKE the CPU FlashAttn
contract's `fdx.symbolic_extent` on k/v — these sections declare NO symbolic-extent FDX block (the
static-only Vulkan binding does not read a live `k_len` from the `SymEnv`).

**Layout model — contiguous-only at the binding boundary (matches the as-built reg).** The deleted
hand-written regs were plain `register_with_precision` (no strided caps), i.e.
`awkward_layout_strategy: requires_contiguous` (`strided_input == false`): the kernels read canonical
row-major q/k/v/do, so the planner auto-Contiguizes a transposed / sliced / non-zero-offset operand
FIRST and sums the `Op::Contiguize` cost (§4.3). Output is always freshly-allocated **contiguous**,
no aliasing, not in-place.

**Cost provenance.** Every cost block is `judge_measured`: the Judge bootstraps it (§4.4). The
`flops` hint is the genuine SDPA FLOP count (QK^T + PV, two MACs/score); no other coefficient is
fabricated. The imported `unknown_cost` sentinel is upgraded to the shared OpKind cost fn by the
`fill_unset_cost_for_backend` pass at registration — the SAME cost the deleted hand-written regs got
from that pass.

**Determinism (conservative correction of the retired consts).** The naive single-pass kernel does
its softmax over the `[Sk]` score row with a per-`(b, h, q)` workgroup shared-memory reduction whose
FADD order is **scheduler-dependent** (the backward recompute likewise, over the query group), so
none is bit-stable even on a re-run on the same device. These are therefore
`determinism: nondeterministic` with `bit_stable_on_same_hardware: false` and an audited
`none(reason)` precision — the CONSERVATIVE correction of the retired hand-written
`VULKAN_FLOAT_POINTWISE_PRECISION` / `VULKAN_HALF_POINTWISE_PRECISION` consts these regs used to
carry, which mis-declared `bit_stable_on_same_hardware: true` + `max_ulp: 1`. Those consts describe
per-thread pointwise arithmetic (Add/Mul/…); applied to a shared-memory softmax reduction they
OVER-CLAIM bit-stability (the aspirational `conv-attn-rope.fkc.md` FlashAttn sections already carry
this honest `nondeterministic` posture). No silent unaudited nondeterminism (§10 rule 9); the Judge
audits the corrected seed.

---

## flash_attn_f32  (naive single-pass multi-head SDPA forward, f32)

f32 `q [B, Hq, Sq, D]` × f32 `k`/`v [B, Hkv, Sk, D]` → f32 `out [B, Hq, Sq, D]`
(`flash_attn::flash_attn_f32` → `VulkanBackend::flash_attn_f32_bytes`). Per-`(b, h, q)` workgroup
materializes one `[Sk]` score row in shared memory, applies the causal mask + optional per-head ALiBi
slope, max-subtract softmax, and accumulates `Σ p·v` — native f32 throughout. GQA (`Hq % Hkv == 0`);
`Sk ≤ 4096`, `D ≤ 256`; `window`/`softcap` bail; STATIC `k_len == sk` only. Contiguous-only binding.
Dispatch keys `(FlashAttn, [F32, F32, F32, F32], Vulkan)` (no alibi) and
`(FlashAttn, [F32, F32, F32, F32, F32], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_f32
op_kind: FlashAttn
blurb: "Naive single-pass MHSA forward (f32): per-(b,h,q) shared-mem score row; GQA, causal, softmax_scale, optional alibi; Sk<=4096, D<=256; static k_len==sk; window/softcap bail; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]  (Sk = CAPACITY; Vulkan v1 reads full extent)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: alibi_slopes                   # optional 4th input; presence implicit in inputs.len()==4
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) is carried by the operand SHAPES / KernelRef, not this variant.
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      k_len:             { kind: usize, note: "Vulkan v1 requires k_len == sk (static full extent); runtime capacity-K bails → CPU/CUDA" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)           # f32 in, f32 out; key pins the trailing [F32] output slot
      shape_rule: from_params(q)           # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated contiguous buffer

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize + sums its cost (§4.3)
  fast_paths:
    - { when: "k_len == sk", note: "static path (the only Vulkan v1 path)" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  # QK^T (2·D MACs/score) + PV (2·D MACs/score) over B·Hq·Sq·k_len; v1 evaluates at CAPACITY (sk).
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false    # per-(b,h,q) shared-mem softmax row reduction: scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent intra-row reduction, non-associative f32 (§4.8)
  notes: "native f32 SDPA; single-pass shared-mem softmax (max-subtract) over the [Sk] score row; intra-row reduction order tile/scheduler-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## flash_attn_bf16  (naive single-pass multi-head SDPA forward, bf16 I/O, f32 compute)

bf16 sibling of `flash_attn_f32` (`flash_attn::flash_attn_bf16` → `VulkanBackend::flash_attn_bf16_bytes`):
identical algorithm, masking, GQA, `Sk ≤ 4096` / `D ≤ 256` limits, and `window`/`softcap`/static-`k_len`
bail set, but Q/K/V/O are bf16 (packed u16-in-u32) with all math in f32 — scores, softmax, and the
`Σ p·v` combination accumulate in f32 and narrow to bf16 (RNE) only on store. Contiguous-only binding.
Dispatch keys `(FlashAttn, [BF16, BF16, BF16, BF16], Vulkan)` (no alibi) and
`(FlashAttn, [BF16, BF16, BF16, BF16, BF16], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_bf16
op_kind: FlashAttn
blurb: "Naive single-pass MHSA forward (bf16 I/O, f32 compute): per-(b,h,q) shared-mem score row; GQA, causal, softmax_scale, optional alibi; Sk<=4096, D<=256; static k_len==sk; window/softcap bail; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D], packed u16-in-u32
    - name: k
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: alibi_slopes
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      k_len:             { kind: usize, note: "Vulkan v1 requires k_len == sk (static full extent); runtime capacity-K bails → CPU/CUDA" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)           # bf16 in, bf16 out (narrow on store); key pins the trailing [BF16] output slot
      shape_rule: from_params(q)           # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path (the only Vulkan v1 path)" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent intra-row reduction, non-associative f32 accum (§4.8)
  notes: "bf16 I/O widened to f32 compute (scores/softmax/PV in f32, narrow to bf16 RNE on store); single-pass shared-mem softmax; intra-row reduction order scheduler-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## flash_attn_f16  (naive single-pass multi-head SDPA forward, f16 I/O, f32 compute)

f16 sibling of `flash_attn_bf16` (`flash_attn::flash_attn_f16` → `VulkanBackend::flash_attn_f16_bytes`):
byte-for-byte the same code path with native f16 (IEEE half) substituted for bf16 — the dot products,
softmax, and `Σ p·v` run in f32 and narrow to f16 (RNE) only on store. Same GQA / masking /
`Sk ≤ 4096` / `D ≤ 256` / `window`/`softcap`/static-`k_len` bail set. Contiguous-only binding.
Dispatch keys `(FlashAttn, [F16, F16, F16, F16], Vulkan)` (no alibi) and
`(FlashAttn, [F16, F16, F16, F16, F16], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_f16
op_kind: FlashAttn
blurb: "Naive single-pass MHSA forward (f16 I/O, f32 compute): per-(b,h,q) shared-mem score row; GQA, causal, softmax_scale, optional alibi; Sk<=4096, D<=256; static k_len==sk; window/softcap bail; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: alibi_slopes
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      k_len:             { kind: usize, note: "Vulkan v1 requires k_len == sk (static full extent); runtime capacity-K bails → CPU/CUDA" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)           # f16 in, f16 out (narrow on store); key pins the trailing [F16] output slot
      shape_rule: from_params(q)           # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path (the only Vulkan v1 path)" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent intra-row reduction, non-associative f32 accum (§4.8)
  notes: "f16 (IEEE half) I/O widened to f32 compute (scores/softmax/PV in f32, narrow to f16 RNE on store); single-pass shared-mem softmax; intra-row reduction order scheduler-dependent; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## flash_attn_backward_q_f32  (FlashAttention backward dQ, f32)

Recompute-based FlashAttention backward producing the **dQ** gradient, native f32
(`flash_attn::flash_attn_backward_q_f32` → `VulkanBackend::flash_attn_backward_q_f32_bytes`). One
workgroup per `(b, h, q)` recomputes the per-query softmax / dP / dS in shared memory (no stored
attention matrix) from `q, k, v`, the upstream gradient `do [B, Hq, Sq, D]`, and optional
`alibi_slopes [Hq]`; writes `dQ` (= q's shape). GQA (`Hq % Hkv == 0`); bails on `window` / `softcap` /
`Sk > 4096` / `D > 256`. Reuses the `OpParams::FlashAttn` carrier. Contiguous-only binding. Dispatch
keys `(FlashAttnBackwardQ, [F32, F32, F32, F32, F32], Vulkan)` (no alibi) and
`(FlashAttnBackwardQ, [F32, F32, F32, F32, F32, F32], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_backward_q_f32
op_kind: FlashAttnBackwardQ
blurb: "FlashAttention backward dQ (f32) from (q, k, v, do, [alibi]); per-(b,h,q) shared-mem recompute of softmax/dP/dS; writes dQ (= q shape); GQA, causal, softmax_scale, alibi; bails on window/softcap/Sk>4096/D>256; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_q_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad dO, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes                   # optional 5th input; presence implicit in inputs.len()==5
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardQ), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }

return:
  outputs:
    - name: dq                           # dQ; this selector (FlashAttnBackwardQ) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ => q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in the recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  # backward recomputes scores + dP + dS over the full Sk extent (~2x the forward QK^T/PV work).
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false    # shared-mem recompute reductions: scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                         # audited none(reason): scheduler-dependent intra-row reduction, non-associative f32 (§4.8)
  notes: "native f32 recompute backward; per-(b,h,q) shared-mem softmax/dP/dS recompute; intra-row reduction order scheduler-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## flash_attn_backward_k_f32  (FlashAttention backward dK, f32)

Recompute-based FlashAttention backward producing the **dK** gradient, native f32
(`flash_attn::flash_attn_backward_k_f32` → `VulkanBackend::flash_attn_backward_k_f32_bytes`). One
workgroup per `(b, h_kv, k_j)` loops over the GQA group's query heads and all query positions,
recomputing the per-query softmax / dS to accumulate the key gradient at row `k_j`; writes `dK`
(= k's shape). Same admissibility as the dQ kernel (GQA, causal, `softmax_scale`, optional ALiBi;
bails on `window` / `softcap` / `Sk > 4096` / `D > 256`). Reuses the `OpParams::FlashAttn` carrier.
Contiguous-only binding. Dispatch keys
`(FlashAttnBackwardK, [F32, F32, F32, F32, F32], Vulkan)` (no alibi) and
`(FlashAttnBackwardK, [F32, F32, F32, F32, F32, F32], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_backward_k_f32
op_kind: FlashAttnBackwardK
blurb: "FlashAttention backward dK (f32) from (q, k, v, do, [alibi]); per-(b,h_kv,k) loop over query heads/positions recomputing dS; writes dK (= k shape); GQA, causal, softmax_scale, alibi; bails on window/softcap/Sk>4096/D>256; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad dO, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }

return:
  outputs:
    - name: dk                           # dK; this selector (FlashAttnBackwardK) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(k)            # dK => k shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in the recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hkv * sk * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 recompute backward; per-(b,h_kv,k) shared-mem dS recompute over the query group; intra-row reduction order scheduler-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

---

## flash_attn_backward_v_f32  (FlashAttention backward dV, f32)

Recompute-based FlashAttention backward producing the **dV** gradient, native f32
(`flash_attn::flash_attn_backward_v_f32` → `VulkanBackend::flash_attn_backward_v_f32_bytes`). Sibling
of the dK kernel — a per-`(b, h_kv, k_j)` workgroup recomputes the per-query softmax weights to
accumulate the value gradient at row `k_j`; writes `dV` (= v's shape). Same admissibility (GQA,
causal, `softmax_scale`, optional ALiBi; bails on `window` / `softcap` / `Sk > 4096` / `D > 256`).
Reuses the `OpParams::FlashAttn` carrier. Contiguous-only binding. Dispatch keys
`(FlashAttnBackwardV, [F32, F32, F32, F32, F32], Vulkan)` (no alibi) and
`(FlashAttnBackwardV, [F32, F32, F32, F32, F32, F32], Vulkan)` (with alibi).

```fkc
kernel: flash_attn_backward_v_f32
op_kind: FlashAttnBackwardV
blurb: "FlashAttention backward dV (f32) from (q, k, v, do, [alibi]); per-(b,h_kv,k) recompute of softmax weights; writes dV (= v shape); GQA, causal, softmax_scale, alibi; bails on window/softcap/Sk>4096/D>256; contiguous-only."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_v_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad dO, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (Vulkan v1 bails → CPU/CUDA fallback)" }

return:
  outputs:
    - name: dv                           # dV; this selector (FlashAttnBackwardV) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(v)            # dV => v shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in the recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2 * hq * sq * d + 2 * hkv * sk * d) * dtype_bytes"
  memory: { device_bytes: "b * hkv * sk * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "native f32 recompute backward; per-(b,h_kv,k) shared-mem softmax-weight recompute over the query group; intra-row reduction order scheduler-dependent, not pinned cross-run; not bit-stable cross-hardware."

determinism: nondeterministic
```

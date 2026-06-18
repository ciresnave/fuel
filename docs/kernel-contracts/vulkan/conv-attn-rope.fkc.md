---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                                        # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"                          # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                          # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — conv / attention / RoPE contracts

This bundle covers the `conv-attn` family of the Vulkan stack: the `conv2d_im2col` patch
rearrangement that feeds the Conv2D matmul, the two flash-attention forward families
(`flash_attn_*` naive single-pass and `flash_attention` tiled FA2), the three flash-attention
backward kernels, and the four RoPE dtype variants.

Sources: kernel inventory `docs/kernel-contracts/_inventory/vulkan.md` (Conv / Attention / RoPE
sections); Slang/GLSL sources under `fuel-kernels-source/kernels/`; Rust wrappers in
`fuel-vulkan-backend/src/lib.rs`. Every kernel here is a **fused op** (`FusedOpParams`
namespace, §3.7): Conv2D = `CONV2D` (FusedOpId 6), FlashAttn = `FLASH_ATTN` (12),
the three backward kernels = `FLASH_ATTN_BACKWARD_Q/K/V` (22/23/24), RoPE = `ROPE` (5).

All costs are marked **`provenance: judge_measured`** — the Judge bootstraps them (§4.4). A
FLOPs / bandwidth hint is supplied only where genuinely derivable from the op's arithmetic
(im2col = bandwidth-bound gather; attention = `2·B·Hq·Sq·k_len·D·2`; RoPE = elementwise
bandwidth). No fabricated launch-overhead / scalar coefficients are author-declared; the
expression strings are derivation hints the Judge refines, and the provenance marker says so.

---

## conv2d_im2col  (NCHW → im2col patches matrix that feeds the Conv2D matmul)

Im2col patch rearrangement (f32). One thread per output-patches element; lowers a NCHW
cross-correlation into a dense `[batch·groups, cin_per_g·k_h·k_w, h_out·w_out]` patches matrix
that a downstream GEMM multiplies by the (reshaped) weight. Supports grouped convolution and
asymmetric stride/padding; out-of-bounds taps (where `stride·out + k - pad` leaves the input
window) are zero-filled rather than clamped, matching the explicit-zero-pad convention. Input
`x` must be contiguous NCHW `[batch, c_in, h, w]`; the kernel reads it with computed NCHW
strides (no FKC stride flags — packed only) and writes the patches buffer linearly, so the
output is always contiguous. This is the **im2col stage** of the `Conv2D` fused op
(`FusedOpId::CONV2D`); the geometry comes from `FusedOpParams::Conv2D` (stride / padding /
groups) plus the operand shapes. Limitations: contiguous-only input (any strided/offset
producer is contiguized upstream); spatial dilation is fixed at 1 (the fused param carries no
dilation); no internal matmul — the GEMM is a separate kernel/cost.

```fkc
kernel: conv2d_im2col
fused_op: CONV2D
blurb: "Im2col patch rearrangement (f32): contiguous NCHW x → dense patches matrix feeding the Conv2D GEMM; groups + asymmetric stride/pad; OOB zero-fill."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::conv2d_im2col_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [batch, c_in, h, w] NCHW
    - name: weight
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [c_out, cin_per_g, k_h, k_w]; consumed by the downstream GEMM, present for the Conv2D key
  op_params:
    variant: Conv2D               # FusedOpParams::Conv2D (fused namespace; §3.7)
    fields:
      stride:  { kind: "(usize, usize)" }
      padding: { kind: "(usize, usize)" }
      groups:  { kind: usize, constraint: "c_in % groups == 0" }

return:
  outputs:
    - name: patches
      dtype_rule: passthrough(x)                    # f32 in, f32 out
      shape_rule: from_params(conv2d_im2col)        # [batch*groups, cin_per_g*k_h*k_w, h_out*w_out]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous       # ← planner inserts Op::Contiguize (itself an FKC kernel) for a non-NCHW-contiguous x
  fast_paths:
    - { when: "groups == 1", note: "dense single-group gather; no per-group base offset" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured                          # Judge bootstraps; no fabricated coefficients
  class: conv
  # bandwidth-bound gather: one read + one write per patches element.
  bytes_moved: "2 * batch * groups * cin_per_g * k_h * k_w * h_out * w_out * dtype_bytes"
  flops: "0"                                          # pure rearrangement; the multiply is the downstream GEMM
  memory: { device_bytes: "batch * groups * cin_per_g * k_h * k_w * h_out * w_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true                   # pure data movement + zero fill; no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Bit-exact rearrangement (f32 copy + explicit zero fill for OOB taps); no FP math, no rounding."

determinism: same_hardware_bitwise
```

---

## conv2d_im2col_bf16  (NCHW → im2col patches matrix, bf16, pairs with coop bf16 matmul)

bf16 im2col patch rearrangement. Same NCHW→patches algorithm as `conv2d_im2col`, but bf16 is
stored as packed u16-in-u32 and this variant is laid out to pair with the cooperative-matrix
bf16 GEMM (`matmul_coop_*_bf16`) that consumes the patches. Input `x` is contiguous NCHW bf16;
out-of-bounds taps zero-fill; output is the contiguous patches matrix in bf16. Because the
rearrangement is a byte-level move (no arithmetic), bf16 values round-trip exactly — no
narrowing occurs in this stage (any precision loss lives in the downstream coop matmul, not
here). Limitations: contiguous-only input; spatial dilation fixed at 1; sub-byte alignment
constraints follow the packed-u16 convention of the bf16 movement family.

```fkc
kernel: conv2d_im2col_bf16
fused_op: CONV2D
blurb: "Im2col patch rearrangement (bf16): contiguous NCHW x → packed-bf16 patches matrix feeding the coop bf16 GEMM; groups + asymmetric stride/pad; OOB zero-fill; byte-exact."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::conv2d_im2col_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [batch, c_in, h, w] NCHW, packed u16-in-u32
    - name: weight
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [c_out, cin_per_g, k_h, k_w]; consumed by the downstream coop GEMM
  op_params:
    variant: Conv2D               # FusedOpParams::Conv2D (fused namespace; §3.7)
    fields:
      stride:  { kind: "(usize, usize)" }
      padding: { kind: "(usize, usize)" }
      groups:  { kind: usize, constraint: "c_in % groups == 0" }

return:
  outputs:
    - name: patches
      dtype_rule: passthrough(x)                    # bf16 in, bf16 out (no narrowing in im2col)
      shape_rule: from_params(conv2d_im2col)        # [batch*groups, cin_per_g*k_h*k_w, h_out*w_out]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "groups == 1", note: "dense single-group gather" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16                          # packed u16 access granularity

cost:
  provenance: judge_measured
  class: conv
  # bandwidth-bound gather; dtype_bytes == 2 for bf16.
  bytes_moved: "2 * batch * groups * cin_per_g * k_h * k_w * h_out * w_out * dtype_bytes"
  flops: "0"
  memory: { device_bytes: "batch * groups * cin_per_g * k_h * k_w * h_out * w_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true                   # byte-level move; bf16 bit pattern preserved exactly
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Byte-exact bf16 rearrangement + zero fill; no narrowing or arithmetic in the im2col stage (the coop matmul carries the f32-accum precision)."

determinism: same_hardware_bitwise
```

---

## flash_attn_f32  (naive single-pass multi-head attention forward, f32)

Naive single-pass multi-head scaled-dot-product attention forward — **not** a tiled online-
softmax kernel: each `(b, h, q_i)` workgroup materializes one `[Sk]` score row in shared memory,
takes the row max, exponentiates, sums, and produces the weighted V combination. f32 throughout
(inputs, accumulation, output). Supports GQA (`kv_h = hi / (Hq/Hkv)`), causal masking,
`softmax_scale`, and an optional per-head ALiBi slope vector. The K/V `Sk` axis is the physical
**capacity**; the kernel attends the live prefix `k_len ≤ Sk` resolved from the SymEnv
(`FusedOpParams::FlashAttn.k_len`), with the static path (`k_len == Sk`) byte-identical to a
plain `0..Sk` loop. Q `[B,Hq,Sq,D]`, K/V `[B,Hkv,Sk,D]`, O `[B,Hq,Sq,D]`, all contiguous
NCHW-like; optional `alibi_slopes [Hq]` (a dummy buffer is bound when absent). Limits: `Sk ≤
4096`, `D ≤ 256`; sliding-window and softcap requests bail to another backend (this kernel does
not implement them). Fully-masked rows (no valid K under the causal mask) emit zeros.

```fkc
kernel: flash_attn_f32
fused_op: FLASH_ATTN
blurb: "Naive single-pass MHSA forward (f32): per-(b,h,q) shared-mem score row over a fixed-capacity KV cache; attends live k_len <= Sk; GQA, causal, alibi. Sk<=4096, D<=256; window/softcap bail."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required   # attends live k_len from SymEnv; stride keyed to Sk
        extent_kind: range          # single bounded SymId k_len <= Sk
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: required   # k_len == v_len ⇒ SAME SymId (FDX unification)
        extent_kind: range
    - name: alibi_slopes          # optional; presence implicit in inputs.len() == 4
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails to another backend)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }
      k_len:             { kind: DynScalar, note: "live attended length <= Sk; rides SymEnv (None ⇒ k_len == Sk)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  # QK^T + PV; symbolic over k_len, evaluated at CAPACITY (sk) in v1 (§4.4). live-k_len re-eval is [consumer-ahead].
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # shared-mem row reductions: scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulation; single-pass shared-mem softmax. Not bit-stable cross-hardware; intra-row reduction order is scheduler-dependent (Judge-audited bound)."

determinism: nondeterministic
```

---

## flash_attn_bf16  (naive single-pass multi-head attention forward, bf16 I/O, f32 accum)

bf16 variant of the naive single-pass attention forward. Identical algorithm and admissibility
to `flash_attn_f32` (GQA, causal, ALiBi, live `k_len ≤ Sk` over a capacity KV cache; `Sk ≤
4096`, `D ≤ 256`; window/softcap bail), but Q/K/V/O are bf16 (packed u16-in-u32) with all math
done in f32 — scores, softmax, and the V combination accumulate in f32, and the result narrows
to bf16 (RNE upper-16, canonical qNaN) only on store. Same single-pass shared-memory score row,
same fully-masked-row zeroing.

```fkc
kernel: flash_attn_bf16
fused_op: FLASH_ATTN
blurb: "Naive single-pass MHSA forward (bf16 I/O, f32 accum): live k_len <= Sk over a capacity KV cache; GQA, causal, alibi. Sk<=4096, D<=256; window/softcap bail."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required
        extent_kind: range
    - name: v
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: required   # k_len == v_len ⇒ SAME SymId
        extent_kind: range
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }
      k_len:             { kind: DynScalar, note: "live attended length <= Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # bf16 out
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16           # packed u16 access granularity

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulation, bf16 I/O narrowing on store (RNE upper-16, canonical qNaN). Not bit-stable cross-hardware; scheduler-dependent reduction order (Judge-audited)."

determinism: nondeterministic
```

---

## flash_attn_f16  (naive single-pass multi-head attention forward, native f16 I/O, f32 accum)

f16 variant of the naive single-pass attention forward. Same algorithm and admissibility as
`flash_attn_f32`/`flash_attn_bf16` (GQA, causal, ALiBi, live `k_len ≤ Sk`; `Sk ≤ 4096`, `D ≤
256`; window/softcap bail), but Q/K/V/O use native `float16_t` storage with f32 accumulation —
the dot products, softmax, and V combination run in f32 and narrow to f16 (RNE) only on store.
Single-pass shared-memory score row; fully-masked rows zeroed.

```fkc
kernel: flash_attn_f16
fused_op: FLASH_ATTN
blurb: "Naive single-pass MHSA forward (native f16 I/O, f32 accum): live k_len <= Sk over a capacity KV cache; GQA, causal, alibi. Sk<=4096, D<=256; window/softcap bail."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]  (Sk = CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required
        extent_kind: range
    - name: v
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: required   # k_len == v_len ⇒ SAME SymId
        extent_kind: range
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }
      k_len:             { kind: DynScalar, note: "live attended length <= Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # f16 out
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * k_len * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 accumulation, native f16 I/O narrowing on store (RNE). Not bit-stable cross-hardware; scheduler-dependent reduction order (Judge-audited)."

determinism: nondeterministic
```

---

## flash_attention  (tiled FlashAttention-2 forward with online softmax, f32)

Tiled scaled-dot-product attention forward (FlashAttention-2) with **online softmax** —
distinct from the `flash_attn_*` naive single-pass family. Block-tiled `BR = BC = 16` over the
query and key/value axes; the running max / running sum are carried per query tile so no full
`[Sk]` score row is materialized. f32. Supports GQA (`groups`), causal masking, **sliding window
`(left, right)`**, ALiBi, and **softcap** — the full feature surface that the naive family bails
on. `head_dim ≤ 128` (`D_MAX`). Q/K/V/O contiguous `[B, H, S, D]`; optional `alibi[Hq]`. Grid is
`(B, Hq, ceil(Sq/16))`. The K/V `Sk` axis is the attended length directly in this kernel's
as-built param surface (no separate live-vs-capacity `k_len` field — the FA2 param block carries
`sk`, `window_left/right`, `has_*` flags, `softcap`). When placed against a `FusedOpParams
::FlashAttn` node it occupies the same `FLASH_ATTN` key, offering the window/softcap features as
a sibling alternative to the naive kernels (the route picker ranks them). Limitation: `D ≤ 128`
(tighter than the naive family's `D ≤ 256`).

```fkc
kernel: flash_attention
fused_op: FLASH_ATTN
blurb: "Tiled FlashAttention-2 forward (f32) with online softmax (BR=BC=16); GQA, causal, sliding window, alibi, softcap. head_dim<=128."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attention"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]; D <= 128
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: tolerated  # works whether Sk is concrete or symbolic; reads the FULL attended Sk capacity (no separate live k_len field in the FA2 param block) — liveness ignored ⇒ scalar/tolerated (§10 rule 15)
        extent_kind: scalar         # reads full capacity Sk (tolerated semantics); NOT range (range ⇒ symbolic_extent: required)
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: tolerated  # reads the FULL attended Sk capacity (liveness ignored) — tolerated semantics
        extent_kind: scalar         # full-capacity read; NOT range (§10 rule 15)
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", note: "supported (FA2 sliding window left)" }
      window_size_right: { kind: "Option<usize>", note: "supported (FA2 sliding window right)" }
      softcap:           { kind: "Option<f32>",   note: "supported (tanh softcap)" }
      k_len:             { kind: DynScalar, note: "attended Sk; tolerated symbolic, evaluated at capacity" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch; all key tiles attended" }
    - { when: "dim[3] % 16 == 0", note: "head_dim aligned to the 16-wide tile" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  # QK^T + PV over the attended Sk; symbolic evaluated at capacity (§4.4).
  flops: "2 * b * hq * sq * sk * d * 2"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # online-softmax rescale + tile reductions: scheduler-dependent order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Online softmax, f32 accumulate; running-max rescale order is tile/scheduler-dependent, so not bit-stable cross-hardware (Judge-audited)."

determinism: nondeterministic
```

---

## flash_attn_backward_q_f32  (FlashAttention backward dQ, f32)

FlashAttention backward producing **dQ**, f32. One workgroup per `(b, h_q, q_i)`; it recomputes
the softmax, dP, and dS for the query row in shared memory (no stored attention matrix) and
accumulates the query gradient. Supports GQA (`kv_h = hi/(Hq/Hkv)`), causal masking,
`softmax_scale`, and optional ALiBi; bails on sliding window, softcap, `Sk > 4096`, or `D > 256`.
Inputs `q, k, v, dO` contiguous `[B, H, S, D]`, optional `alibi[Hq]` (the as-built ABI is 6
storage + 1 uniform, `layout_6s1u`). dQ has the same shape as Q; rows that were fully masked in
the forward pass produce zero gradient. Reuses `FusedOpParams::FlashAttnBackward` (shared by
Q/K/V; the `FusedOpId` distinguishes which gradient).

```fkc
kernel: flash_attn_backward_q_f32
fused_op: FLASH_ATTN_BACKWARD_Q
blurb: "FlashAttention backward dQ (f32): per-(b,h,q) recompute of softmax/dP/dS in shared mem; GQA, causal, scale, alibi. Bails on window/softcap/Sk>4096/D>256."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_q_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
    - name: do                       # upstream gradient dO, shape == q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward    # FusedOpParams::FlashAttnBackward (shared Q/K/V; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }

return:
  outputs:
    - name: dq
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ shape == Q
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  # backward recomputes scores + dP + dS: ~2x the forward QK^T/PV work.
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d) * dtype_bytes"
  memory: { device_bytes: "b * hq * sq * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # shared-mem recompute reductions: scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 recompute + accumulation; not bit-stable cross-hardware (intra-row reduction order is scheduler-dependent). Judge-audited."

determinism: nondeterministic
```

---

## flash_attn_backward_k_f32  (FlashAttention backward dK, f32)

FlashAttention backward producing **dK**, f32. One workgroup per `(b, h_kv, k_j)` looping over
the GQA group's query heads and all query positions `q_i`, recomputing the per-query softmax/dS
to accumulate the key gradient at row `k_j`. Same admissibility as the dQ kernel: GQA, causal,
`softmax_scale`, optional ALiBi; bails on window, softcap, `Sk > 4096`, `D > 256`. Inputs
`q, k, v, dO` contiguous `[B, H, S, D]`, optional `alibi[Hq]` (6 storage + 1 uniform). dK has the
same shape as K. Reuses `FusedOpParams::FlashAttnBackward`.

```fkc
kernel: flash_attn_backward_k_f32
fused_op: FLASH_ATTN_BACKWARD_K
blurb: "FlashAttention backward dK (f32): per-(b,h_kv,k) loop over query heads/positions recomputing dS; GQA, causal, scale, alibi. Bails on window/softcap/Sk>4096/D>256."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_k_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }

return:
  outputs:
    - name: dk
      dtype_rule: passthrough(k)
      shape_rule: same_as(k)            # dK shape == K
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d) * dtype_bytes"
  memory: { device_bytes: "b * hkv * sk * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 recompute + accumulation over the query loop; not bit-stable cross-hardware (scheduler-dependent reduction order). Judge-audited."

determinism: nondeterministic
```

---

## flash_attn_backward_v_f32  (FlashAttention backward dV, f32)

FlashAttention backward producing **dV**, f32. One workgroup per `(b, h_kv, k_j)` looping over
the GQA group's query heads and positions, recomputing the softmax weights `P` to accumulate the
value gradient `dV[k_j] += Σ P · dO`. Same admissibility as dQ/dK: GQA, causal, `softmax_scale`,
optional ALiBi; bails on window, softcap, `Sk > 4096`, `D > 256`. Inputs `q, k, v, dO` contiguous
`[B, H, S, D]`, optional `alibi[Hq]` (6 storage + 1 uniform). dV has the same shape as V. Reuses
`FusedOpParams::FlashAttnBackward`.

```fkc
kernel: flash_attn_backward_v_f32
fused_op: FLASH_ATTN_BACKWARD_V
blurb: "FlashAttention backward dV (f32): per-(b,h_kv,k) loop recomputing softmax weights to accumulate dV; GQA, causal, scale, alibi. Bails on window/softcap/Sk>4096/D>256."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::flash_attn_backward_v_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      rank: 1                       # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", constraint: "must be None (window bails)" }
      window_size_right: { kind: "Option<usize>", constraint: "must be None (window bails)" }
      softcap:           { kind: "Option<f32>",   constraint: "must be None (softcap bails)" }

return:
  outputs:
    - name: dv
      dtype_rule: passthrough(v)
      shape_rule: same_as(v)            # dV shape == V
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in recompute" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: attention
  flops: "2 * b * hq * sq * sk * d * 4"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d) * dtype_bytes"
  memory: { device_bytes: "b * hkv * sk * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 recompute + accumulation over the query loop; not bit-stable cross-hardware (scheduler-dependent reduction order). Judge-audited."

determinism: nondeterministic
```

---

## rope  (fused rotary position embedding, f32)

Fused rotary position embedding (RoPE) in the **rotate-half** convention. One thread per
`(o, s, i)`: it reads the `cos`/`sin` table entries for position `s` and rotation index `i`, then
writes the rotated pair into output positions `i` and `i + head_dim/2`. f32 rotation math. Unlike
the contiguous-only attention family, `rope` is **strided-capable on `x`**: a contiguous fast path
(`x_contiguous` flag set) indexes linearly, otherwise the kernel applies per-dim strides
(`x_s0`, `x_s1`, `x_s_seq`, `x_s_hd`), so a lazy `[0, 2, 1, 3]` permute of `x` is consumed without
a contiguize. The `cos`/`sin` tables are **always contiguous** `[seq, head_dim]`. Output is always
contiguous in `x`'s logical shape. This is the `ROPE` fused op (`FusedOpId(5)`,
`FusedOpParams::Rope`, parameterless — seq/head_dim recovered from input shapes). No window /
position-offset param surface here; the caller bakes positions into the `cos`/`sin` tables.

```fkc
kernel: rope
fused_op: ROPE
blurb: "Fused rotary position embedding (f32, rotate-half): strided-capable x (lazy permute, contiguous fast path) + contiguous cos/sin tables; contiguous output."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [outer, heads, seq, head_dim]; lazy [0,2,1,3] permute tolerated
    - name: cos
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2                       # [seq, head_dim]
    - name: sin
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2                       # [seq, head_dim]
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope                 # FusedOpParams::Rope (parameterless; seq/head_dim from shapes)
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)            # x's logical shape
      layout_guarantee: contiguous      # written linearly even when x is strided
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks per-dim strides on x; no contiguize for a permuted x
  fast_paths:
    - { when: "all_inputs_contiguous", note: "x_contiguous flag set; linear index, no stride decode" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  # one rotate per element pair; bandwidth-bound elementwise.
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # per-element rotate, no cross-element reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 rotation math (cos/sin from caller-supplied tables); deterministic per-element, no reduction. Bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

---

## rope_f16  (fused rotary position embedding, native f16, f32 rotation math)

f16 RoPE. Same rotate-half algorithm and strided-`x` capability as `rope`, but `x`/output are
native `float16_t` while the rotation arithmetic runs in f32 (cos/sin tables stay f32-precision
inputs) and narrows to f16 (RNE) on store. Contiguous fast path via `x_contiguous`, per-dim
stride decode otherwise; cos/sin contiguous `[seq, head_dim]`; output contiguous.

```fkc
kernel: rope_f16
fused_op: ROPE
blurb: "Fused rotary position embedding (native f16, f32 rotation math): strided-capable x + contiguous cos/sin; narrows on store; contiguous output."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [outer, heads, seq, head_dim]
    - name: cos
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2                       # [seq, head_dim]
    - name: sin
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # f16 out
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", note: "x_contiguous flag set; linear index" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 rotation math, native f16 I/O narrowing on store (RNE); deterministic per-element, no reduction. Bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

---

## rope_bf16  (fused rotary position embedding, packed bf16 pair-thread, f32 rotation math)

bf16 RoPE. Same rotate-half algorithm as `rope`, but bf16 is stored packed (u16-in-u32) and this
variant is **pair-threaded**: each thread handles 2 u32 words = 4 bf16 positions, so it requires
`head_dim % 4 == 0`. Rotation math in f32; bf16 narrows on store (RNE upper-16, canonical qNaN).
`x` is strided-capable (contiguous fast path + per-dim stride decode), cos/sin contiguous f32
tables, output contiguous.

```fkc
kernel: rope_bf16
fused_op: ROPE
blurb: "Fused rotary position embedding (packed bf16 pair-thread, f32 rotation math): requires head_dim % 4 == 0; strided-capable x + contiguous cos/sin; narrows on store."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [outer, heads, seq, head_dim]; head_dim % 4 == 0 (pair-thread, 4 bf16/thread)
      shape_constraint: "divisible(x.dim[3], 4)"
    - name: cos
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2                       # [seq, head_dim]
    - name: sin
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # bf16 out
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", note: "x_contiguous flag set; linear index" }
    - { when: "dim[3] % 4 == 0", note: "required: pair-thread 4-bf16/thread; not a fast-path, a precondition" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 rotation math, packed-bf16 I/O narrowing on store (RNE upper-16, canonical qNaN); deterministic per-element. Bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

---

## rope_f64  (fused rotary position embedding, native f64)

f64 RoPE. Same rotate-half algorithm and strided-`x` capability as `rope`, but `x`/output are
native `double` and the rotation arithmetic runs in f64 (no narrowing). Contiguous fast path via
`x_contiguous`, per-dim stride decode otherwise; cos/sin contiguous f32 tables (widened to f64
for the rotation); output contiguous.

```fkc
kernel: rope_f64
fused_op: ROPE
blurb: "Fused rotary position embedding (native f64): strided-capable x + contiguous cos/sin; full-precision rotation, no narrowing; contiguous output."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::rope_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [outer, heads, seq, head_dim]
    - name: cos
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2                       # [seq, head_dim]
    - name: sin
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }   # contiguous-only aux under the kernel-wide handles_strided default — per-operand override (§4.3.1 / §10.5)
      rank: 2
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope
    fields: {}

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)        # f64 out
      shape_rule: same_as(x)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", note: "x_contiguous flag set; linear index" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64           # native f64 element

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"    # dtype_bytes == 8 for f64
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Native f64 rotation, no narrowing (cos/sin widened from f32 tables); deterministic per-element, no reduction. Bit-stable on the same hardware."

determinism: same_hardware_bitwise
```

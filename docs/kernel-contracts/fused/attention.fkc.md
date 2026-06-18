---
fkc_version: 1
provider:
  name: fuel-graph-fused-cpu
  backend: Cpu                                   # maps to BackendId::Cpu
  kernel_source: "portable-cpu"                  # the BindingEntry.kernel_source tag
  link_registry: fuel_dispatch::dispatch::FUSED_ENTRY_POINTS   # symbol→KernelRef map (§12.6)
  revision_base: "git:f41137b4"                  # provider build id, folded into kernel_revision_hash
---

# fuel-graph fused attention — kernel contracts (family `attention`)

The Fuel `FusedOpRegistry` attention family: FlashAttn, PagedAttn, and the three FlashAttn
backward gradients (Q/K/V). Each is a **fused op** (`fused_op:`/`FusedOpParams`, §3.7), so its
cost compiles to the **fused** cost-fn shape `fn(&[Shape], &FusedOpParams, &BackendCapabilities)`
(no `&[DType]` arg) and its return rules compile to the graph-side `FusedOp.shape_rule` /
`dtype_rule` (§12.7).

These contracts describe the **always-built CPU reference** kernels reached via
`GraphExecutor::cpu_fallback`. As-built facts (from the inventory + sources): the graph-side
registry does **not** encode layout, and the CPU wrappers take `_layouts: &[Layout]` and ignore
it, calling `cpu_input()` which returns the raw byte buffer with no stride application — so every
fused CPU attention kernel is **contiguous-only, offset-0, row-major**, and a non-contiguous
input is auto-contiguized by the executor before the kernel runs. None of these entries register
`caps`, so caps default to `KernelCaps::empty()`. CPU precision is `ATTN_CPU_PRECISION` /
`ATTN_BACKWARD_CPU_PRECISION` (the naive reference is bit-stable on the same hardware, F32
accumulation for BF16/F16 inputs; the tiled-softmax GPU forms differ and declare their own
precision when they register). `decompose` **panics** for all five (no primitive form — a
decomposition would re-materialize the `[B, Hq, Sq, Sk]` score matrix the fused form exists to
avoid). Backward is `NotDifferentiable` in the registry for all five (forward grads route through
the three FlashAttnBackward ids; the backward ids are not higher-order differentiable).

> **GPU-aware operand vocabulary.** FlashAttn's K/V `Sk` axis is the **physical capacity** of a
> fixed-capacity KV cache and the kernel attends only the live prefix `k_len ≤ Sk` (a dynamic
> scalar on the `SymEnv`). The K/V operands therefore declare `fdx.symbolic_extent: required` and
> `fdx.extent_kind: range` (a single bounded `SymId`; §4.5) so a GPU sibling at the same key can
> consume the live extent — the as-built CPU reference reads the full extent when `k_len` is
> `None`, the static path. PagedAttn declares the FDX **gather** descriptor (paged blocks) and
> carries the block-table / context-lens as separate operands per the single-place rule (§3.9.1).

---

## FlashAttn  (multi-head scaled-dot-product attention, KV-cache aware)

Fused multi-head scaled-dot-product (FlashAttention) attention over a fixed-capacity KV cache.

`q [B, Hq, Sq, D]`, `k`/`v [B, Hkv, Sk, D]` with grouped-query attention (`Hkv ≤ Hq`,
GQA-divisible `Hq % Hkv == 0`); optional 4th input `alibi_slopes [Hq]`. The K/V `Sk` axis is the
**physical capacity** of the KV allocation (strides and byte-length checks key off it); the kernel
attends only the first `k_len ≤ Sk` rows — the live prefix from a fixed-capacity KV cache — and
bottom-right-aligns the causal mask at offset `k_len − Sq` (the standard FA2 mask placement).
`k_len` is a dynamic scalar (`Option<DynScalar>`) resolved per token via the `SymEnv`: `k_len ==
None` ⇒ the full K extent (the static path, byte-identical to a plain `0..Sk` loop and
`k_len == Sk`); `Some(_)` ⇒ a runtime live-prefix over a capacity KV cache. The CPU reference
computes naive attention internally (`matmul → scale → mask → softmax → matmul`) and is
bit-stable on the same hardware with F32 accumulation (F64 for F64 input); a GPU tiled/online
softmax form is **not** bit-identical and registers its own precision. `softmax_scale`, optional
`softcap`, and sliding-window `(left, right)` are honored. Output is `q`'s shape and dtype.
Decompose **panics** (a primitive lowering would re-materialize the `[B, Hq, Sq, Sk]` score
matrix); backends without a flash kernel route through `cpu_fallback`. Known limitation:
contiguous-only — any strided/broadcast/offset operand is auto-contiguized by the executor first.

```fkc
kernel: FlashAttn
fused_op: FLASH_ATTN
blurb: "Fused MHSA over a fixed-capacity KV cache; attends live prefix k_len <= Sk; GQA; causal/window/softcap."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]  (Sk = physical CAPACITY)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required            # attends live k_len from SymEnv; stride keyed to Sk
        extent_kind: range                   # single bounded SymId: k_len <= Sk (§4.5)
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
      fdx:
        symbolic_extent: required            # k_len ≡ v_len ⇒ SAME SymId (FDX unification)
        extent_kind: range
    - name: alibi_slopes                     # optional 5th input; presence is inputs.len() == 5
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                       # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: DynScalar, note: "None ⇒ full K extent (k_len == Sk static path); Some ⇒ live prefix <= Sk, rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)             # FusedOp.dtype_rule = input 0
      shape_rule: from_params(q)             # FusedOp.shape_rule = q shape [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous           # fresh row-major buffer; executor pre-allocates
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU wrapper ignores layouts; executor auto-Contiguizes (cost summed from the Contiguize contract, §4.3/§4.4)
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to 0..Sk loop" }
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false

cost:
  provenance: judge_measured                 # the Judge bootstraps this cost (§4.4). Hint below records the derivable FLOP form only.
  class: attention
  # Fused cost-fn shape (no &[DType] arg). Symbolic over the live k_len; v1 evaluates at CAPACITY (Sk).
  # Derivable hint: QK^T is 2*B*Hq*Sq*k_len*D, PV is the same ⇒ ~4*B*Hq*Sq*k_len*D MACs.
  flops_hint: "4 * b * hq * sq * k_len * d"  # author hint; the Judge measures the real cost
  memory:
    device_bytes: 0                          # CPU backend; output lives in host bytes
    host_bytes: "b * hq * sq * d * dtype_bytes"   # output alloc (executor pre-allocates)
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: true          # ATTN_CPU_PRECISION: naive reference, deterministic; F32 accum (F64 for F64 input)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU naive-attention reference: bit-stable same-hardware; F32 accumulate for BF16/F16, F64 for F64. GPU tiled/online-softmax forms differ and declare their own precision."

determinism: same_hardware_bitwise
```

---

## PagedAttn  (paged-cache attention, decode-only)

Fused paged-cache attention (decode-only) over a vLLM-style blocked KV cache.

`q [B, Hq, Sq, D]`; the K/V caches are physical block pools `[num_blocks, block_size, Hkv, D]`
re-interpreted per sequence via a `block_table [B, max_blocks_per_seq] U32` and per-sequence live
lengths `context_lens [B] U32`; optional 6th input `alibi_slopes [Hq]`. In FDX terms a paged
cache is a single tensor — an honest contiguous block-pool base plus an `FDXIndexedResidency`
gather sidecar (`kind = paged_blocks`) — but the as-built `PagedAttn` ABI takes the pool, block
table, and context lengths as **separate graph inputs** (`q, k_cache, v_cache, block_table,
context_lens, [alibi]`), so each is its own `accept.inputs` operand (the single-place rule,
§3.9.1: the operand's `fdx.gather.block_table` / `context_lens` fields name those same input
roles, not a duplicate of the data). The per-sequence live length is data-determined and symbolic
(`symbolic_extent: required`). Output is `q`'s shape and dtype. Decompose **panics**; backward is
`NotDifferentiable` (decode-only). CPU reference is bit-stable same-hardware with F32 accumulation.
Known limitation: contiguous-only operands (auto-contiguized by the executor).

> **Capability gate.** A backend consumes the paged pool directly only if it advertises
> `Capability::DlpackExtGather`; otherwise the planner inserts an explicit materialize (dense
> un-paged copy) priced from the materialize kernel's own FKC contract. **[consumer-ahead]:** the
> FDX gather descriptor / `Capability::DlpackExtGather` are the 2026-06-17 FDX addition (no code
> yet); an importer that reaches the `gather`-bearing pool operand before the FDX gather codes
> land returns `GatherNotYetSupported` (the `MxNotYetRegistrable` discipline, §3.9.1).

```fkc
kernel: PagedAttn
fused_op: PAGED_ATTN
blurb: "Decode-only paged-cache attention over a blocked KV pool indexed by a per-sequence block table + context lengths."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::paged_attn_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]
    - name: k_cache
      dtypes: [F32, F64, BF16, F16]          # the TRUE per-token pool element type
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # physical pool [num_blocks, block_size, Hkv, D]
      fdx:
        requires_ext: true                   # MEANING_REQUIRES_EXT mandatory for a paged pool (FDX gather V19)
        symbolic_extent: required            # per-seq live length is symbolic (context_lens)
        extent_kind: range                   # single bounded SymId per sequence (live <= capacity)
        gather:
          kind: paged_blocks                 # FDX FDX_GATHER_PAGED_BLOCKS
          block_table: block_table           # role of the SEPARATE block-table accept.input (below)
          context_lens: context_lens         # role of the SEPARATE context-lens accept.input (below)
    - name: v_cache
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [num_blocks, block_size, Hkv, D]
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        extent_kind: range
        gather:
          kind: paged_blocks
          block_table: block_table
          context_lens: context_lens
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                                # [B, max_blocks_per_seq]
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [B]
      fdx:
        symbolic_extent: required            # per-seq live lengths (data-determined sym)
        extent_kind: range
    - name: alibi_slopes                     # optional 6th input; presence is inputs.len() == 6
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Hq]
      optional: true
  op_params:
    variant: PagedAttn                       # FusedOpParams::PagedAttn (fused namespace; §3.7)
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, note: "KV-cache block size; physical pool dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)             # FusedOp.dtype_rule = input 0
      shape_rule: from_params(q)             # FusedOp.shape_rule = q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU wrapper ignores layouts; executor auto-Contiguizes
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false

cost:
  provenance: judge_measured                 # the Judge bootstraps this cost (§4.4). Hint records the derivable FLOP form only.
  class: attention
  # Per-sequence live length is data-determined (context_lens); the FLOP count scales with the
  # summed live lengths, which the Judge measures. Author hint uses the per-sequence live len ctx_len.
  flops_hint: "4 * b * hq * sq * ctx_len * d"
  memory:
    device_bytes: 0
    host_bytes: "b * hq * sq * d * dtype_bytes"   # output alloc
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: true          # ATTN_CPU_PRECISION (CPU reference)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU paged-attention reference: bit-stable same-hardware; F32 accumulate for BF16/F16, F64 for F64. GPU forms declare their own precision."

determinism: same_hardware_bitwise
```

---

## FlashAttnBackwardQ  (FlashAttention backward — dQ)

Fused FlashAttention backward producing the query gradient dQ.

Inputs `(q, k, v, do, [alibi])` — 4 or 5 — where `do` is the upstream gradient of the forward
output (same shape and dtype as `q`). dtypes F32/F64/BF16/F16, all inputs share dtype. Output dQ
has `q`'s shape (input 0) and `q`'s dtype. The shared `FusedOpParams::FlashAttnBackward` carries
the same shape parameters as the forward `FlashAttn` (`softmax_scale`, `causal`,
`window_size_(left|right)`, `softcap`) so the recompute pass reproduces identical scores; the
`FusedOpId` distinguishes Q vs K vs V. v1 design: each backward variant **recomputes the softmax
state independently** (3× recompute on the CPU reference; a single-launch multi-output variant
that shares the recompute is a follow-up). Decompose **panics** (a primitive form would
re-materialize the `[Sq, Sk]` score matrix — every backend must register a kernel); pattern is
`None` (autograd emits this op directly through a FlashAttn forward); backward is
`NotDifferentiable` (no higher-order grads). CPU reference is bit-stable same-hardware with F32
accumulation. Known limitation: contiguous-only operands (auto-contiguized).

```fkc
kernel: FlashAttnBackwardQ
fused_op: FLASH_ATTN_BACKWARD_Q
blurb: "FlashAttention backward dQ from (q, k, v, do, [alibi]); recomputes softmax state; output = q shape/dtype."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do                               # upstream gradient of the forward output
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]  (= forward out shape = q shape)
      shape_constraint: "same_as=q"
    - name: alibi_slopes                     # optional 5th input; presence is inputs.len() == 5
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward               # FusedOpParams::FlashAttnBackward (shared by Q/K/V; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq
      dtype_rule: passthrough(q)             # FusedOp.dtype_rule = input 0 (all inputs share dtype)
      shape_rule: same_as(q)                 # FusedOp.shape_rule_q = input 0 (q) shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU wrapper ignores layouts; executor auto-Contiguizes
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false

cost:
  provenance: judge_measured                 # the Judge bootstraps this cost (§4.4)
  class: attention
  # Backward attention ≈ a small constant multiple of forward FLOPs; v1 CPU recomputes softmax
  # independently per gradient. Author hint records only the recompute FLOP form.
  flops_hint: "5 * b * hq * sq * sk * d"
  memory:
    device_bytes: 0
    host_bytes: "b * hq * sq * d * dtype_bytes"   # dQ alloc (= q shape)
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: true          # ATTN_BACKWARD_CPU_PRECISION (CPU reference)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU FlashAttn-backward reference: bit-stable same-hardware; F32 accumulate for BF16/F16, F64 for F64. v1 recomputes softmax independently. GPU forms declare their own precision."

determinism: same_hardware_bitwise
```

---

## FlashAttnBackwardK  (FlashAttention backward — dK)

Fused FlashAttention backward producing the key gradient dK.

Inputs `(q, k, v, do, [alibi])` — 4 or 5 — `do` is the upstream gradient (same shape/dtype as the
forward output = `q`). dtypes F32/F64/BF16/F16, all inputs share dtype. Output dK has `k`'s shape
(input 1) and the shared dtype. Shares `FusedOpParams::FlashAttnBackward` with the Q and V
variants (`softmax_scale`, `causal`, `window_size_(left|right)`, `softcap`); the `FusedOpId`
selects dK. v1 recomputes the softmax state independently. Decompose **panics**; pattern `None`;
backward `NotDifferentiable`. CPU reference bit-stable same-hardware, F32 accumulation.
Contiguous-only (auto-contiguized).

```fkc
kernel: FlashAttnBackwardK
fused_op: FLASH_ATTN_BACKWARD_K
blurb: "FlashAttention backward dK from (q, k, v, do, [alibi]); recomputes softmax state; output = k shape/dtype."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]  (= forward out shape = q shape)
      shape_constraint: "same_as=q"
    - name: alibi_slopes                     # optional 5th input; presence is inputs.len() == 5
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward               # FusedOpParams::FlashAttnBackward (shared by Q/K/V; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk
      dtype_rule: passthrough(q)             # FusedOp.dtype_rule = input 0 (all inputs share dtype)
      shape_rule: same_as(k)                 # FusedOp.shape_rule_k = input 1 (k) shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU wrapper ignores layouts; executor auto-Contiguizes
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false

cost:
  provenance: judge_measured                 # the Judge bootstraps this cost (§4.4)
  class: attention
  flops_hint: "5 * b * hq * sq * sk * d"     # author hint; v1 recomputes softmax independently per gradient
  memory:
    device_bytes: 0
    host_bytes: "b * hkv * sk * d * dtype_bytes"   # dK alloc (= k shape)
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: true          # ATTN_BACKWARD_CPU_PRECISION (CPU reference)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU FlashAttn-backward reference: bit-stable same-hardware; F32 accumulate for BF16/F16, F64 for F64. v1 recomputes softmax independently. GPU forms declare their own precision."

determinism: same_hardware_bitwise
```

---

## FlashAttnBackwardV  (FlashAttention backward — dV)

Fused FlashAttention backward producing the value gradient dV.

Inputs `(q, k, v, do, [alibi])` — 4 or 5 — `do` is the upstream gradient (same shape/dtype as the
forward output = `q`). dtypes F32/F64/BF16/F16, all inputs share dtype. Output dV has `v`'s shape
(input 2) and the shared dtype. Shares `FusedOpParams::FlashAttnBackward` with the Q and K
variants (`softmax_scale`, `causal`, `window_size_(left|right)`, `softcap`); the `FusedOpId`
selects dV. v1 recomputes the softmax state independently. Decompose **panics**; pattern `None`;
backward `NotDifferentiable`. CPU reference bit-stable same-hardware, F32 accumulation.
Contiguous-only (auto-contiguized).

```fkc
kernel: FlashAttnBackwardV
fused_op: FLASH_ATTN_BACKWARD_V
blurb: "FlashAttention backward dV from (q, k, v, do, [alibi]); recomputes softmax state; output = v shape/dtype."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_cpu"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                                # [B, Hq, Sq, D]  (= forward out shape = q shape)
      shape_constraint: "same_as=q"
    - name: alibi_slopes                     # optional 5th input; presence is inputs.len() == 5
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                                # [Hq]
      optional: true
  op_params:
    variant: FlashAttnBackward               # FusedOpParams::FlashAttnBackward (shared by Q/K/V; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv
      dtype_rule: passthrough(q)             # FusedOp.dtype_rule = input 0 (all inputs share dtype)
      shape_rule: same_as(v)                 # FusedOp.shape_rule_v = input 2 (v) shape
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # CPU wrapper ignores layouts; executor auto-Contiguizes
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false

cost:
  provenance: judge_measured                 # the Judge bootstraps this cost (§4.4)
  class: attention
  flops_hint: "5 * b * hq * sq * sk * d"     # author hint; v1 recomputes softmax independently per gradient
  memory:
    device_bytes: 0
    host_bytes: "b * hkv * sk * d * dtype_bytes"   # dV alloc (= v shape)
    disk_bytes: 0

precision:
  bit_stable_on_same_hardware: true          # ATTN_BACKWARD_CPU_PRECISION (CPU reference)
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "CPU FlashAttn-backward reference: bit-stable same-hardware; F32 accumulate for BF16/F16, F64 for F64. v1 recomputes softmax independently. GPU forms declare their own precision."

determinism: same_hardware_bitwise
```

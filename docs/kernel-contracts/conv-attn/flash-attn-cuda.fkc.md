---
fkc_version: 1
provider:
  name: fuel-flash-attn-cuda
  backend: Cuda                  # maps to BackendId::Cuda
  kernel_source: "flash-attn-cuda"   # the BindingEntry.kernel_source tag (Dao-AILab FA2 sm80 static lib via fuel-flash-attn-cuda-sys::run_mha)
  link_registry: fuel_flash_attn_cuda::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"  # provider build id, folded into kernel_revision_hash
---

# fuel-flash-attn-cuda — kernel contracts (attention family)

CUDA-only FlashAttention-v2 scaled-dot-product attention for the `conv-attn` crate
(`fuel-flash-attn-cuda`). Two kernels: a fixed-(padded)-length rank-4 form (`flash-attn`) and a
variable-length packed rank-3 form (`flash-attn-varlen`). Both are `fuel::CustomOp3` ops applied
via `q.apply_op3(k, v, op)`, monomorphized over `q.dtype()` (`cuda_fwd_t::<f16|bf16>`), and both
dispatch into the **same** `fuel_flash_attn_cuda_sys::run_mha` FFI entry point (Ampere sm80
static lib). They reject CPU storage and any dtype other than f16/bf16.

> **As-built note on the dispatch key.** This crate's surface is a `CustomOp3`, not a graph-fused
> registry entry of its own. The closest as-built registry analog — and the form this contract
> targets so the planner can cost/admit/dispatch it — is the graph-side fused op
> `FusedOps::FLASH_ATTN` (`FusedOpId(12)`, `fuel-graph/src/registry.rs:885`) carried by
> `FusedOpParams::FlashAttn` (`registry.rs:231`) / `OpKind::FlashAttn`
> (`fuel-core-types/src/dispatch.rs:148`). Both kernels below therefore declare
> `fused_op: FLASH_ATTN` with `op_params.variant: FlashAttn` (the FusedOpParams namespace, §3.7).
> **`flash-attn-varlen` has no distinct as-built fused id or params variant** — varlen is a
> `fuel-flash-attn-cuda`-only CustomOp3 whose `seqlens_q`/`seqlens_k`/`max_seqlen_*` are op
> struct fields, not a registered `FusedOpParams` variant. It is contracted here as a sibling
> alternative at the same `FLASH_ATTN` key, distinguished by `kernel_source` + `entry_point`
> (its varlen-ness is a `notes:`-documented kernel fact and the rank-3 + cu_seqlens accept
> contract); registering it through `FusedOpParams::FlashAttn` is a [consumer-ahead] mapping until
> a varlen FusedOpParams variant (with `cu_seqlens` operands) lands. Its declared `op_params`
> fields below therefore match the **real** `FusedOpParams::FlashAttn` variant
> (`registry.rs:231-238`: `softmax_scale`, `causal`, `window_size_left`, `window_size_right`,
> `softcap`, `k_len`); the varlen-specific `max_seqlen_q` / `max_seqlen_k` and the `cu_seqlens`
> pointers are **not** as-built fused fields and are documented in prose as [consumer-ahead] kernel
> facts (the rank-3 + cumulative-seqlens accept contract), never invented onto the variant.
>
> **As-built note on the awkward-layout strategy (per-operand, §10.5).** Both kernels genuinely
> walk arbitrary **outer** strides on q/k/v (batch/row/head strides forwarded straight from the
> `Layout`) while **requiring the inner/last axis (head_size) to be contiguous**
> (`stride[rank-1] == 1`, else bail). They do **not** contiguize internally and do **not** require
> fully-dense q/k/v — so the honest q/k/v strategy is `handles_strided` (the load-bearing fact),
> with the inner-contiguous requirement documented as the `inner_contiguous(q)` hard-requirement
> fast-path in `notes:`. The auxiliary operands are genuinely contiguous-only: `flash-attn`'s
> `alibi_slopes` is read as a flat slice, and `flash-attn-varlen`'s `seqlens_q`/`seqlens_k` bail
> unless `contiguous_offsets()` succeeds. **FKC §10.5 now resolves awkward-layout coherence PER
> OPERAND** (§4.3.1): each operand declares its own `layout.awkward_layout_strategy`, so q/k/v
> carry `handles_strided` (with `strided: accepted`) while the aux operands carry
> `requires_contiguous` (with `contiguous: required`). The planner contiguizes only a
> non-contiguous aux operand, never q/k/v. The aux operands' `strided: rejected` below is the
> faithful fact; the per-operand strategy makes the whole contract pass §10.5 with no falsified
> flag. The earlier "spec-vs-as-built gap" note is resolved by the 2026-06-18 per-operand
> awkward-layout addition.

## flash-attn  (FlashAttention-v2 SDPA, fixed-length rank-4, CUDA f16/bf16)

CUDA FlashAttention-v2 scaled-dot-product attention `softmax(Q @ K^T * softmax_scale) @ V` over a
fixed (padded) batch layout. q/k/v are rank-4 `(B, S, H, D)` (the as-built `dims4()` order — batch,
seqlen, num_heads, head_size — NOT `[B, H, S, D]`); k and v share `(B, Sk, Hkv, D)`. Supports
MHA / MQA / GQA (`H % Hkv == 0`), causal & sliding-window masking, ALiBi slopes (F32), and
Gemma-style softcap. Ampere sm80 only. Dispatched `CustomOp3::fwd` → `cuda_fwd_t::<T>` →
`ffi::run_mha` (`unpadded_lse = 0`).

Long description. The kernel walks q/k/v from arbitrary outer strides — batch / row(seq) / head
strides are read straight from the `Layout` and forwarded as `q_batch_stride` / `q_row_stride =
stride[rank-3]` / `q_head_stride = stride[rank-2]` (lib.rs:176-188) — but **requires the last dim
(head_size) to be contiguous** (`stride[rank-1] == 1`, else bail, lib.rs:60-68). It is therefore
strided-capable on the outer 3 axes and contiguous-only on the inner axis ("strided outer, packed
inner"). Each input is sliced from its `start_offset()` (lib.rs:41-43), so it is **non-zero-offset
capable**; ALiBi slopes are likewise sliced from their own offset (lib.rs:113). Shape constraints
enforced at build time via `fuel::bail!` (no panic on bad shape/dtype): rank exactly 4
(lib.rs:55), `head_size <= 512` (lib.rs:79), `head_size % 8 == 0` (lib.rs:82),
`H % Hkv == 0` (lib.rs:86), k/v shape match q on (B, Sk, Hkv, D) (lib.rs:70-78). Internal rounding:
head_size → mult of 8 then 32 (`d_rounded`), seqlen_q/k → mult of 128 (lib.rs:136-139). Masking
semantics: causal = (`window_right == 0 && window_left < 0`); a one-sided window is expanded to
seqlen_k on the open side (lib.rs:149-159); window values > seqlen_k are treated as None/-1
(lib.rs:123-134). Numerics are FlashAttention-v2 online-softmax with **fp32 accumulation** inside
the kernel and the LSE in fp32; softcap applies a tanh-based logit cap before softmax. The
softmax_lse F32 scratch (`b * 128 * num_heads * seqlen_q`) is allocated and written by the kernel
but **not returned** — `fwd` returns only `(dst, out_shape)` (single output). Output is a freshly
allocated **contiguous** buffer equal in shape to q. CPU storage bails ("no cpu support for
flash-attn", lib.rs:227). Known limitations: f16/bf16 only; sm80 only; last dim must be
contiguous; one latent `.unwrap()` on the ALiBi RwLock read guard (lib.rs:101). Not bit-stable
across hardware (warp-reduction order).

```fkc
kernel: flash-attn
fused_op: FLASH_ATTN
blurb: "CUDA FlashAttention-v2 SDPA, fixed-length rank-4 (B,S,H,D) f16/bf16; GQA; causal/window/softcap/ALiBi."
backend: Cuda
kernel_source: "flash-attn-cuda"
entry_point: "fuel_flash_attn_cuda::flash_attn_fwd"   # CustomOp3 FlashAttn → ffi::run_mha; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 4                       # [B, S, H, D] (batch, seqlen_q, num_heads, head_size)
      shape_constraint: "rank=4"
      # last dim (head_size) MUST be contiguous; outer 3 axes may be strided. See notes.
    - name: k
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 4                       # [B, Sk, Hkv, D]
      shape_constraint: "divisible(q.dim[2], k.dim[2])"   # GQA: H % Hkv == 0 (heads at index 2 in [B,S,H,D])
    - name: v
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 4                       # [B, Sk, Hkv, D]; must equal k's shape
      shape_constraint: "same_as=k"
    - name: alibi_slopes          # optional 4th input; presence implicit in inputs.len()==4
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }
      rank: 1                       # [H] (num_heads_q); read as a flat slice (contiguous-only)
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; registry.rs:231)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool, note: "causal = (window_right == 0 && window_left < 0)" }
      window_size_left:  { kind: "Option<usize>", note: "values > seqlen_k treated as None; one-sided window expanded to seqlen_k" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>", note: "0.0 disables; tanh logit cap before softmax" }
      k_len:             { kind: "Option<DynScalar>", note: "None ⇒ attend full Sk (this kernel's behavior); Some rides SymEnv (live prefix) — graph-form only, not the CustomOp3 surface" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # dev.alloc::<T>(elem_count), T = q's f16/bf16 (lib.rs:142)
      shape_rule: from_params(q)        # out_shape = q's shape [B, S, H, D] (lib.rs:35)
      layout_guarantee: contiguous      # Layout::contiguous(&out_shape) (lib.rs:36); freshly allocated
      aliasing: none                    # fresh buffer; no overlap with inputs
      # side scratch softmax_lse [b*128*H*Sq] F32 is written but NOT returned (single output).

caps:
  awkward_layout_strategy: handles_strided   # kernel-wide DEFAULT (q/k/v walk outer strides); alibi_slopes overrides to requires_contiguous per-operand (§4.3.1)
  fast_paths:
    - { when: "inner_contiguous(q)", note: "required: last dim stride==1 for q/k/v (else bail)" }
    - { when: "causal == false", note: "no causal mask branch" }
    - { when: "softcap == None", note: "no tanh softcap branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16    # f16/bf16 element access

cost:
  provenance: judge_measured     # Judge bootstraps/refines empirically (§4.4); no fabricated coefficients
  class: attention
  # FLOPs hint (derivable from the op): QK^T (2*B*H*Sq*Sk*D) + softmax·V (2*B*H*Sq*Sk*D).
  # Halved when causal (lower-triangular). Fused cost-fn shape (no &[DType] arg); §4.4.
  flops: "4 * b * h * sq * sk * d"
  bytes_moved: "(b*sq*h*d + 2*b*sk*hkv*d + b*sq*h*d) * dtype_bytes"   # read q,k,v + write out
  overhead_ns: ~                 # judge_measured — CUDA launch overhead populated by calibration
  memory: { device_bytes: "b * sq * h * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }   # output alloc; softmax_lse scratch additionally allocated device-side

precision:
  bit_stable_on_same_hardware: false   # online-softmax + warp reductions: scheduler-dependent FADD order
  max_ulp: ~
  max_relative: ~
  max_absolute: 0.001            # F32-reference: < 1e-5 acausal, < 1e-3 softcap (flash_attn_tests.rs:113,151); pessimistic bound 1e-3
  audited: true
  notes: "FlashAttention-v2 f16/bf16 inputs, fp32 internal accumulation + fp32 LSE; tanh softcap before softmax. Validated vs fp32 reference: < 1e-5 acausal, < 1e-3 softcap. Not bit-stable cross-hardware (warp reduction order)."

determinism: nondeterministic    # warp-reduction order; bit_stable=false + audited=true per §4.9
```

## flash-attn-varlen  (FlashAttention-v2 SDPA, variable-length packed rank-3, CUDA f16/bf16)

CUDA FlashAttention-v2 with **variable-length (ragged) batching** — packed sequences indexed by
cumulative `seqlens_q` / `seqlens_k` (each length `batch_size + 1`, U32, cumulative offsets), no
padded batch axis. Same attention math, masking, ALiBi, and softcap as `flash-attn`, over rank-3
q/k/v `(total_tokens, num_heads, head_size)`. Ampere sm80 only. Dispatched `CustomOp3::fwd` →
`cuda_fwd_t::<T>` → `ffi::run_mha` with `cu_seqlens_*` pointers and `unpadded_lse = 1`.

Long description. q/k/v rank must be exactly 3 (`q_rank != 3` bail, lib.rs:511); the **last dim
(head_size) must be contiguous** (`stride[rank-1] == 1`, lib.rs:516-524) while the outer axes are
strided (row = stride[rank-3], head = stride[rank-2] forwarded; batch strides forced to 0 because
varlen, lib.rs:645-657) — same "strided outer, packed inner" capability as `flash-attn`. q/k/v are
non-zero-offset capable (sliced `start_offset()..len`, lib.rs:497-499); ALiBi likewise
(lib.rs:580). The cumulative-sequence-length inputs `seqlens_q` / `seqlens_k` are **U32** and
**must be contiguous** — read via `contiguous_offsets()`, bail if `None` (lib.rs:478-481,489-492);
they are passed to the FFI as `*const i32`. `batch_size = nseqlens_q - 1` (lib.rs:555);
constraints: `nseqlens_q >= 2` (lib.rs:547), `nseqlens_q == nseqlens_k` (lib.rs:551),
`head_size <= 512` & `% 8 == 0`, `H % Hkv == 0` (lib.rs:535-544). `max_seqlen_q` / `max_seqlen_k`
drive the internal rounding (mult-of-128) and window clamping (lib.rs:605-606,622-625); window /
causal semantics are identical to `flash-attn` but clamped against `max_seqlen_k`
(lib.rs:590-626). Numerics are FlashAttention-v2 online-softmax, fp32 accumulation + fp32 LSE. The
softmax_lse F32 scratch (`num_heads * total_q`) is written but **not returned**
(`unpadded_lse = 1`, lib.rs:610) — single output. Output is a freshly allocated **contiguous**
buffer equal in shape to q `(total_q, num_heads, head_size)`. CPU storage bails (lib.rs:696). Known
limitations: f16/bf16 only; sm80 only; last dim contiguous; seqlens contiguous + U32; latent
`.unwrap()` on the seqlens/ALiBi RwLock read guards (lib.rs:473,484,568). Not bit-stable across
hardware.

```fkc
kernel: flash-attn-varlen
fused_op: FLASH_ATTN
blurb: "CUDA FlashAttention-v2 SDPA, variable-length packed rank-3 (total_tokens,H,D) f16/bf16 via cu_seqlens; GQA; causal/window/softcap/ALiBi."
backend: Cuda
kernel_source: "flash-attn-cuda-varlen"   # distinct from flash-attn → sibling alternative at the same FLASH_ATTN key (§12.5)
entry_point: "fuel_flash_attn_cuda::flash_attn_varlen_fwd"   # CustomOp3 FlashAttnVarLen → ffi::run_mha (unpadded_lse=1); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 3                       # [total_q, H, D]
      shape_constraint: "rank=3"
      # last dim (head_size) MUST be contiguous; row/head axes may be strided; batch stride forced 0 (varlen).
    - name: k
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 3                       # [total_k, Hkv, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: H % Hkv == 0 (heads at index 1 in [total,H,D])
    - name: v
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 3                       # [total_k, Hkv, D]
      shape_constraint: "same_rank=k"
    - name: seqlens_q             # cumulative query seqlens; SEPARATE graph input
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }
      rank: 1                       # [batch_size + 1]; read via contiguous_offsets() (bail if None); passed to FFI as *const i32
      shape_constraint: "dim[0]=batch_size+1"
    - name: seqlens_k             # cumulative key seqlens; SEPARATE graph input
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }
      rank: 1                       # [batch_size + 1]; nseqlens_q == nseqlens_k; contiguous_offsets() (bail if None)
      shape_constraint: "same_as=seqlens_q"
    - name: alibi_slopes          # optional
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: requires_contiguous }
      rank: 1                       # [H]; read as a flat slice (contiguous-only)
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (registry.rs:231-238) — fields match the REAL variant.
    fields:
      # These six are the as-built FusedOpParams::FlashAttn fields (registry.rs:232-237), identical
      # to the flash-attn kernel above. The varlen op's max_seqlen_q/max_seqlen_k and the
      # cu_seqlens pointers are CustomOp3 struct fields with NO as-built fused counterpart — they
      # are documented in prose as [consumer-ahead] kernel facts, NOT declared onto the variant.
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>", note: "clamped against max_seqlen_k (a varlen kernel fact; the clamp ceiling itself is a [consumer-ahead] CustomOp3 field, not a fused param)" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>", note: "0.0 disables; tanh logit cap before softmax" }
      k_len:             { kind: "Option<DynScalar>", note: "as-built fused field (registry.rs:237); None ⇒ attend full Sk. Rides SymEnv (live prefix) — graph-form only, not the CustomOp3 varlen surface" }
      # [consumer-ahead] varlen-only kernel facts (NO as-built FusedOpParams field): max_seqlen_q /
      # max_seqlen_k drive the mult-of-128 rounding + window clamp ceiling (lib.rs:605-606,622-625),
      # and seqlens_q / seqlens_k are accept.inputs operands above (cumulative offsets passed as the
      # cu_seqlens FFI pointers), not scalar fields. A varlen FusedOpParams variant carrying these
      # has not landed; this contract maps onto FusedOpParams::FlashAttn until it does (header note).

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # dev.alloc::<T>(elem_count), T = q's f16/bf16 (lib.rs:609)
      shape_rule: from_params(q)        # out_shape = q's shape [total_q, H, D] (lib.rs:469)
      layout_guarantee: contiguous      # Layout::contiguous (lib.rs:470); freshly allocated
      aliasing: none                    # fresh buffer; no overlap with inputs
      # side scratch softmax_lse [H * total_q] F32 written but NOT returned (unpadded_lse=1).

caps:
  awkward_layout_strategy: handles_strided   # kernel-wide DEFAULT (q/k/v walk outer strides); seqlens_q/seqlens_k/alibi_slopes override to requires_contiguous per-operand (§4.3.1)
  fast_paths:
    - { when: "inner_contiguous(q)", note: "required: last dim stride==1 for q/k/v (else bail)" }
    - { when: "causal == false", note: "no causal mask branch" }
    - { when: "softcap == None", note: "no tanh softcap branch" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured     # Judge bootstraps/refines empirically (§4.4); no fabricated coefficients
  class: attention
  # FLOPs hint (derivable): QK^T + softmax·V over the ragged batch. Per-sequence work sums to
  # 2*(Σ_b sq_b * sk_b)*H*D for each of the two matmuls; capacity-evaluated upper bound uses
  # total_q tokens against max_seqlen_k. Fused cost-fn shape (no &[DType] arg); §4.4.
  flops: "4 * total_q * h * max_seqlen_k * d"
  bytes_moved: "(total_q*h*d + 2*total_k*hkv*d + total_q*h*d) * dtype_bytes"   # read q,k,v + write out
  overhead_ns: ~                 # judge_measured — CUDA launch overhead populated by calibration
  memory: { device_bytes: "total_q * h * d * dtype_bytes", host_bytes: 0, disk_bytes: 0 }   # output alloc; softmax_lse scratch additionally device-side

precision:
  bit_stable_on_same_hardware: false   # online-softmax + warp reductions: scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: 0.001            # same FA2 numerics as flash-attn; F32-reference < 1e-5 acausal / < 1e-3 softcap
  audited: true
  notes: "FlashAttention-v2 f16/bf16 inputs, fp32 internal accumulation + fp32 LSE; identical numerics to flash-attn over packed ragged sequences. Not bit-stable cross-hardware (warp reduction order)."

determinism: nondeterministic    # warp-reduction order; bit_stable=false + audited=true per §4.9
```

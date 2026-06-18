---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                       # maps to BackendId::Metal
  kernel_source: "metal-msl"           # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"         # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — matmul + attention contracts

The matmul / attention family from `fuel-metal-kernels` (`mlx_gemm.metal`,
`scaled_dot_product_attention.metal`) and the `call_*` dispatch wrappers in
`kernels/{mlx_gemm,sdpa}.rs`, wired by `fuel-metal-backend/src/storage.rs`. Five kernels:
the MLX/steel fused GEMM, and the four SDPA variants (tiled prefill, single-pass vector decode,
and the two split-K vector-decode passes). All are Apple-Metal-only (`device: Metal`,
`substrate: MetalBuffer`). Costs are `judge_measured` — the Judge bootstraps them; the
author-derivable FLOPs/bandwidth hints below are commented in each block, not asserted as a
declared prior.

> **Attention op-kind note.** All four SDPA kernels implement Fuel's scaled-dot-product
> attention, which the fused-op vocabulary models as `FLASH_ATTN` (`FusedOpId(12)`,
> `fuel-graph/src/registry.rs:885`) carried by `FusedOpParams::FlashAttn { softmax_scale,
> causal, window_size_left, window_size_right, softcap, k_len }`
> (`fuel-dispatch/src/kernel.rs:299`). They are therefore **fused-op** contracts and compile to
> the fused cost-fn shape `fn(&[Shape], &FusedOpParams, &BackendCapabilities)` (no `&[DType]`
> arg — §4.4). `mlx_gemm` is a **primitive** `op_kind: MatMul` (`OpKind::MatMul`,
> `fuel-core-types/src/dispatch.rs:54`) carried by `OpParams::Matmul`
> (`fuel-dispatch/src/kernel.rs:202`). As-built, only `steel_attention` and `sdpa_vector*` are
> the two `call_sdpa_*` entry points and `sdpa_vector_has_mask` is a compile-time function
> constant; the 2-pass partials (`intermediate`/`sums`/`maxs`) are kernel-internal scratch, not
> graph outputs.

## mlx_gemm  (batched GEMM `D = A @ B`, MLX/steel tiles)

Batched matrix multiply `D = A @ B` over the steel templated GEMM
(`gemm_<trans>_<itype>_<otype>_<bm>_<bn>_<bk>_<wm>_<wn>`). The four transpose variants
(`nn`/`tn`/`nt`/`tt`) are selected from the last-two-dim strides of A and B: each operand must be
row-major **or** column-major in its trailing 2 dims (lda/ldb derived, transpose detected from
strides); any other minor-stride pattern raises `MatMulNonContiguous`, so the kernel walks
exactly the contiguous-or-transposed layout class rather than arbitrary strides. Byte offsets
make it offset-capable on both inputs. Batch dims flow through `GemmParams.batch_strides`; a B
operand broadcast along the batch axis (stride 0) triggers a batch-collapse into M. The tile
config (`bm/bn/bk/wm/wn`) and `align_m/n/k` / `has_batch` / gather+axpby-off are function
constants chosen by dtype/size/device. The output is a fresh `b*m*n` row-major buffer
(`ldd = n`); there is no accumulate (`use_out_source = false`). itype must equal otype; the
backend `matmul()` admits only f32/f16/bf16. Accumulation is internal to the steel tile
(typically f32 for f16/bf16 inputs). Not bit-stable across hardware (tile/warp reduction order).

```fkc
kernel: mlx_gemm
op_kind: MatMul
blurb: "Batched GEMM D = A @ B (MLX/steel tiles); A/B row- or col-major in last 2 dims; f32/f16/bf16; in==out dtype."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_backend::metal_src::mlx_gemm"   # resolves to the trans/dtype/tile-monomorphized gemm
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16]
      # row-major OR column-major in the last 2 dims; transpose detected from strides.
      # NOT a general strider: any other minor-stride pattern is MatMulNonContiguous.
      # contiguous: required under the kernel-wide requires_contiguous strategy (§10.5); the
      # strided: accepted slot covers only the col-major/transposed contiguous-class view.
      layout: { contiguous: required, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 2..=6                       # [..batch.., m, k]; batch via batch_strides
      shape_constraint: "last_dim_eq=rhs"   # k = lhs.dim[-1] == rhs.dim[-2]
    - name: rhs
      dtypes: [F32, F16, BF16]
      # batch-dim broadcast (stride 0) IS tolerated and collapses the batch into M.
      # contiguous: required under the kernel-wide requires_contiguous strategy (§10.5).
      layout: { contiguous: required, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 2..=6                       # [..batch.., k, n]
      shape_constraint: "same_dtype=lhs"
  op_params:
    variant: Matmul                     # OpParams::Matmul (primitive namespace; §3.7)
    fields:
      m: { kind: usize }
      n: { kind: usize }
      k: { kind: usize, constraint: "== lhs.dim[-1] == rhs.dim[-2]" }
      lhs_batch_dims: { kind: "Vec<usize>" }
      rhs_batch_dims: { kind: "Vec<usize>", note: "B batch broadcast (stride 0) collapses batch into M" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)      # itype == otype; out dtype = input dtype
      shape_rule: matmul(lhs, rhs)      # [..batch.., m, n], row-major (ldd = n)
      layout_guarantee: contiguous      # fresh b*m*n buffer, row-major
      aliasing: none                    # no accumulate (use_out_source = false)

caps:
  # accepts the contiguous|transposed layout class directly (no fixup), but rejects general
  # strides → not handles_strided in the §10.5 "every input strided:accepted for arbitrary
  # strides" sense. The planner contiguizes only an operand outside the contig|transposed class.
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", note: "nn path; lda=k, ldb=n" }
    - { when: "any_input_strided", note: "tn/nt/tt transpose-of-contiguous path; no copy" }
    - { when: "any_input_broadcast", note: "B batch-broadcast → batch-collapse into M" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; coefficients below are derivation HINTS only
  class: gemm_like
  # FLOPs derivable from the op: dense GEMM = 2*M*N*K (batched: * prod(batch)). bandwidth ~ read
  # A (m*k) + B (k*n) + write D (m*n), * batch * dtype_bytes. Left to the Judge to populate.
  # flops:       "2 * b * m * n * k"
  # bytes_moved: "b * (m*k + k*n + m*n) * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false    # steel tile / warp reduction order is scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "steel tiles; f16/bf16 accumulate internally in f32. Not bit-stable cross-hardware (tile/warp reduction order)."

determinism: nondeterministic
```

## steel_attention  (tiled full/prefill scaled-dot-product attention)

Tiled "steel" scaled-dot-product attention for the full / prefill case
(`steel_attention_<itype>_bq_bk_bd_wm_wn_mask<mtype>`). Inputs `q [b, h, ql, d]`, `k`/`v
[b, h, kl, d]`, with GQA via `gqa_factor = q_heads / kv_heads`. An optional additive mask
(f16/bf16/f32, or none) and optional causal masking are pipeline **function constants**
(`align_Q`/`align_K`, `has_mask`, `do_causal`), so a given pipeline either takes the mask path
or does not. head_dim `bd ∈ {32,64,72,80,96,128,256,512}` (bd=512 is f16/bf16 only). q/k/v are
passed as contiguous 3-stride arrays with `(b,h,seq,d)` shape; byte offsets make them
offset-capable. `softcapping` is fixed at 1.0 (disabled) in the wired wrapper. Output matches the
itype dtype and q's shape, contiguous. f32-accumulated streaming softmax; not bit-stable across
hardware (tile/warp reduction order). The `AttnParams` struct carries `(b,h,d,ql,kl,gqa,scale,
softcapping,tiling,…)`. Source: `metal_src/scaled_dot_product_attention.metal`;
`kernels/sdpa.rs:21-247` (`call_sdpa_full`).

```fkc
kernel: steel_attention
fused_op: FLASH_ATTN
blurb: "Tiled prefill SDPA q[b,h,ql,d] over k/v[b,h,kl,d]; GQA; optional additive mask + causal (function constants); f32-accum."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_backend::kernels::sdpa::call_sdpa_full"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, ql, d]
    - name: k
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, kl, d]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: q_heads % kv_heads == 0
      fdx: { symbolic_extent: tolerated, extent_kind: scalar }   # prefill reads the full kl capacity (liveness ignored)
    - name: v
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: tolerated, extent_kind: scalar }   # kl ≡ vl, read at full capacity (FDX unification)
    - name: mask                        # optional additive mask; presence is a pipeline function constant (has_mask)
      dtypes: [F16, BF16, F32]
      # mask is contiguous-or-strided per inventory; per-operand override of the kernel-wide
      # requires_contiguous so its strided: accepted is coherent (§4.3.1/§10.5).
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected, awkward_layout_strategy: handles_strided }
      rank: 2..=4
      optional: true
  op_params:
    variant: FlashAttn                  # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32, note: "AttnParams.scale" }
      causal:            { kind: bool, note: "do_causal function constant" }
      window_size_left:  { kind: "Option<usize>", note: "not exercised by this Metal kernel" }
      window_size_right: { kind: "Option<usize>", note: "not exercised by this Metal kernel" }
      softcap:           { kind: "Option<f32>", note: "AttnParams.softcapping fixed 1.0 (disabled) in the wired wrapper" }
      k_len:             { kind: DynScalar, note: "prefill: live kl; tolerated, read at capacity; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # out dtype = itype
      shape_rule: from_params(q)        # = q shape [b, h, ql, d]; symbolic ql preserved
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no mask branch (do_causal=0)" }
    - { when: "dim[3] == 512", note: "bd=512 tile; f16/bf16 only" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured            # Judge bootstraps; HINT below only
  class: attention
  # FLOPs derivable: QK^T + PV ≈ 2 * b * h * ql * kl * d * 2 (causal halves it). Left to the Judge.
  # flops: "2 * b * h * ql * kl * d * 2"

precision:
  bit_stable_on_same_hardware: false    # streaming softmax + tile/warp reductions: scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32-accumulated online softmax; softcapping disabled (1.0). Not bit-stable cross-hardware (tile/warp reduction order)."

determinism: nondeterministic
```

## sdpa_vector  (single-pass vector / decode attention)

Single-pass vector SDPA for the decode case (q seq-len 1), one threadgroup per `(b*h)`
(`sdpa_vector_<itype>_<bk>`). head_dim `bk ∈ {32,64,96,128,256,512}`. q/k/v are contiguous, but
k/v are addressed through `k_stride[1]` / `v_stride[1]` (the head stride), so the KV cache may be
laid out with a per-head stride; all three are offset-capable. The `sdpa_vector_has_mask` function
constant is `false` here — this kernel has no mask path. GQA via `gqa_factor`. `alpha` is the
softmax scale, pre-divided by `softcapping` when capping is enabled. Output matches itype and q's
shape, contiguous. f32-accumulated; not bit-stable across hardware. op params: `gqa_factor, n
(= kv_len), kstride, vstride, alpha (scale), softcapping`. Source: `kernels/sdpa.rs:253-360`
(`call_sdpa_vector`).

```fkc
kernel: sdpa_vector
fused_op: FLASH_ATTN
blurb: "Single-pass vector/decode SDPA (q seq-len 1), one threadgroup per (b*h); GQA via head-stride KV; no mask; f32-accum."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_backend::kernels::sdpa::call_sdpa_vector"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, 1, d]  (decode: q seq-len 1)
    - name: k
      # k addressed via k_stride[1] (head stride): a per-head KV layout is honoured directly.
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, n, d]  (n = kv_len = capacity)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: q_heads % kv_heads == 0
      fdx: { symbolic_extent: required, extent_kind: range }   # decode over a KV cache; n is the live kv_len
    - name: v
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }   # n ≡ kv_len for k and v ⇒ SAME SymId
  op_params:
    variant: FlashAttn                  # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32, note: "alpha (pre-divided by softcapping when capping enabled)" }
      causal:            { kind: bool, note: "no mask path (sdpa_vector_has_mask=false)" }
      window_size_left:  { kind: "Option<usize>", note: "not exercised" }
      window_size_right: { kind: "Option<usize>", note: "not exercised" }
      softcap:           { kind: "Option<f32>", note: "softcapping; folds into alpha when enabled" }
      k_len:             { kind: DynScalar, note: "n = live kv_len; rides SymEnv; strides keyed to capacity" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # = q shape [b, h, 1, d]
      layout_guarantee: contiguous
      aliasing: none

caps:
  # q/k/v must be contiguous in the dense sense; the head-stride is the only stride freedom and is
  # consumed directly via k_stride[1]/v_stride[1], not via a general strider.
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[2] == 1", note: "decode fast path: q seq-len 1, one threadgroup per (b*h)" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured            # Judge bootstraps; HINT below only
  class: attention
  # decode FLOPs ≈ 2 * b * h * n * d * 2 (ql=1). Bandwidth-bound on reading the n-row KV. Judge-populated.
  # flops: "2 * b * h * n * d * 2"

precision:
  bit_stable_on_same_hardware: false    # single-pass online softmax over the KV; reduction order scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32-accumulated; alpha pre-divided by softcapping when capping enabled. Not bit-stable cross-hardware."

determinism: nondeterministic
```

## sdpa_vector_2pass_1  (split-K vector decode — partial pass)

Pass 1 of the split-K (two-pass) vector decode attention (`sdpa_vector_2pass_1_*`). The KV range
is split across `SDPA_2PASS_BLOCKS = 32` blocks; each block computes a partial attention output
plus its running `sums` and `maxs`, written to three **scratch** buffers
(`intermediate`/`sums`/`maxs`) that pass 2 then reduces. q/k/v are contiguous with the same
per-head KV stride freedom as `sdpa_vector` (`k_stride[1]`/`v_stride[1]`), offset-capable.
head_dim `bk ∈ {32,64,96,128,256,512}`. GQA via `gqa_factor`. op params (pass 1):
`gqa_factor, n, kstride, vstride, alpha, softcapping` plus the `intermediate`/`sums`/`maxs`
scratch. This pass produces **no graph output** — only kernel-internal partials consumed by
`sdpa_vector_2pass_2`; the bundle's single FDX output is emitted by pass 2. Source:
`kernels/sdpa.rs:362-547` (`call_sdpa_vector_2pass`).

```fkc
kernel: sdpa_vector_2pass_1
fused_op: FLASH_ATTN
blurb: "Split-K vector-decode SDPA pass 1: per-block partial output + running sums/maxs over 32 blocks into scratch."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_backend::kernels::sdpa::call_sdpa_vector_2pass_1"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, 1, d]
    - name: k
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, n, d]  (n = kv_len)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA
      fdx: { symbolic_extent: required, extent_kind: range }   # n is the live kv_len; split across 32 blocks
    - name: v
      dtypes: [F16, BF16, F32]
      layout: { contiguous: required, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }   # n ≡ kv_len ⇒ SAME SymId
  op_params:
    variant: FlashAttn                  # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale: { kind: f32, note: "alpha" }
      causal:        { kind: bool, note: "no mask path" }
      softcap:       { kind: "Option<f32>", note: "softcapping; alpha pre-divided when enabled" }
      k_len:         { kind: DynScalar, note: "n = live kv_len; rides SymEnv; split across SDPA_2PASS_BLOCKS=32" }

return:
  # Pass 1 writes only kernel-internal scratch (intermediate/sums/maxs); it produces no graph
  # tensor. The fused op's single FDX output is emitted by sdpa_vector_2pass_2. Per the as-built
  # 12-multi-output ABI a kernel still declares one output slot; here it is the partial-state
  # scratch bundle the second pass reduces, marked layout_guarantee: preallocated (executor owns
  # the scratch, the kernel only fills it).
  outputs:
    - name: partials
      dtype_rule: fixed(F32)            # intermediate/sums/maxs accumulators are f32
      shape_rule: from_params(scratch)  # [b, h, SDPA_2PASS_BLOCKS, ...]; capacity-sized scratch
      layout_guarantee: preallocated    # executor-allocated scratch; not a contiguous graph output
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "dim[2] == 1", note: "decode: q seq-len 1" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured            # Judge bootstraps; HINT below only
  class: attention
  # pass-1 FLOPs ≈ 2 * b * h * n * d (the QK^T+partial-PV over the full n, split 32-way). Judge-populated.
  # flops: "2 * b * h * n * d"

precision:
  bit_stable_on_same_hardware: false    # split-K partials reduced in pass 2; reduction order scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 partial accumulators (intermediate/sums/maxs); final numerics depend on pass-2 reduction. Not bit-stable cross-hardware."

determinism: nondeterministic
```

## sdpa_vector_2pass_2  (split-K vector decode — reduce pass)

Pass 2 of the split-K vector decode attention (`sdpa_vector_2pass_2_*`). It reduces the 32
per-block partials produced by `sdpa_vector_2pass_1` — combining each block's `intermediate`
output with its `sums`/`maxs` via the online-softmax rescale — into the final attention output.
head_dim `bk ∈ {32,64,96,128,256,512}`. Its inputs are the three scratch buffers
(`intermediate`/`sums`/`maxs`), not q/k/v; op params: `intermediate, sums, maxs` (the same
`gqa_factor, n, …` carried through for addressing). Output matches the itype dtype and q's shape,
contiguous — this is the graph-visible attention result. f32 reduction; not bit-stable across
hardware. Source: `kernels/sdpa.rs:362-547` (`call_sdpa_vector_2pass`).

```fkc
kernel: sdpa_vector_2pass_2
fused_op: FLASH_ATTN
blurb: "Split-K vector-decode SDPA pass 2: reduce 32 per-block partials (intermediate/sums/maxs) into the final output."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_backend::kernels::sdpa::call_sdpa_vector_2pass_2"
kernel_revision_hash: auto

accept:
  # Inputs are the pass-1 scratch partials, not q/k/v. They are dense executor-owned buffers.
  inputs:
    - name: intermediate
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                           # [b, h, SDPA_2PASS_BLOCKS, d] per-block partial outputs
    - name: sums
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 3                           # [b, h, SDPA_2PASS_BLOCKS] running softmax denominators
      shape_constraint: "same_rank=maxs"
    - name: maxs
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 3                           # [b, h, SDPA_2PASS_BLOCKS] running block maxima
      shape_constraint: "same_as=sums"
  op_params:
    variant: FlashAttn                  # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale: { kind: f32, note: "carried for output dtype/geometry; reduction is over partials" }
      causal:        { kind: bool, note: "no mask path" }
      softcap:       { kind: "Option<f32>", note: "softcapping" }
      k_len:         { kind: DynScalar, note: "n = live kv_len; determines how many of the 32 blocks held data" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)        # final attention output dtype = q's itype
      shape_rule: from_params(q)        # = q shape [b, h, 1, d]
      layout_guarantee: contiguous      # fresh dense output (the graph-visible attention result)
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", note: "dense reduce over the 32 partials" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured            # Judge bootstraps; HINT below only
  class: attention
  # pass-2 reduce FLOPs ≈ b * h * SDPA_2PASS_BLOCKS * d (combine 32 partials per (b*h)). Judge-populated.
  # flops: "b * h * 32 * d"

precision:
  bit_stable_on_same_hardware: false    # online-softmax rescale of split-K partials; reduction order scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "f32 reduction of pass-1 partials via online-softmax rescale; narrows to itype on store. Not bit-stable cross-hardware."

determinism: nondeterministic
```

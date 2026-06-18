---
fkc_version: 1
provider:
  name: fuel-reference-backend
  backend: Cpu                       # the pure-Rust oracle runs on host (BackendId::Cpu)
  kernel_source: "reference-oracle"  # the BindingEntry.kernel_source tag
  link_registry: fuel_reference_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-reference-backend — attention kernel contracts

The `fuel-reference-backend` crate is the correctness-first, pure-Rust oracle (`src/attention.rs`,
generic over `T: num_traits::Float`, monomorphized to `{f32, f64, bf16, f16}` by the executor).
Per the crate-wide layout invariant (`src/lib.rs:68`), `RefTensor<T>` is *always* a contiguous,
row-major `Vec`/`Arc<[T]>` plus a `Shape` — **no strides, no offset**. Every kernel here is
therefore **contiguous-only, zero-offset** at the data layer, and every output is a fresh
contiguous `RefTensor::from_vec`. Callers must materialize any non-contiguous view before calling.

All four attention kernels share the `[B, H, S, D]` (batch-first, heads-second) shape convention,
GQA via `Hq` a multiple of `Hkv` (each KV head broadcast over its Q group), stable softmax, and the
`AttentionParams` family (`attention.rs:36`): `softmax_scale: f32`, `causal: bool`,
`window_size_left/right: Option<usize>`, `softcap: Option<f32>`, plus optional ALiBi slopes `[Hq]`.
Mask admissibility is `position_admissible` (`attention.rs:76`). Execution wiring varies per kernel
and is stated in each section's long description (the FLASH_ATTN executor arm routes to
`attention_naive` as its oracle; `attention_flash`, `attention_paged_naive`, and
`attention_flash_backward` have the as-built wiring noted below).

## attention_naive  (multi-head scaled-dot-product attention; materialized `[B,H,Sq,Sk]` matrix)

The math-definition oracle for MHSDPA. Materializes the full per-`(b,h,qi)` score row, applies
mask + softcap + ALiBi, runs a stable (subtract-row-max) softmax, then accumulates the weighted
sum over `V`.

Builds the textbook attention: for each `(batch, head, query)` it computes raw scores
`S[qi,kj] = scale · (Q_row · K_row)` over all `Sk` key positions, applying — in order — softcap
(`tanh(s/c)·c`), then ALiBi bias (`slope_h · (kj − qi)`); positions failing `position_admissible`
(causal `kj > qi`, or sliding-window `kj + left < qi` / `kj > qi + right`) are left at `-inf`. A
stable softmax subtracts the row max (`-inf` entries map to 0), and the output row is the
probability-weighted sum of `V` rows. A **fully-masked row** (no finite score) yields an
**all-zero output row** — the same fully-masked handling FlashAttention uses. **Numerics:** all
arithmetic is in the generic `T` (no f32-widened accumulator); for `bf16`/`f16` the dot-products
and softmax run at half precision, so this kernel is precision-sensitive at half. **Perf:** O(N²)
memory for the per-row `scores` vector, O(N²·D) compute; the materialized form is the simple
correctness reference, not a memory-efficient path. **Wiring:** this is the executor's `FLASH_ATTN`
oracle arm (`attention_flash` is *not* the exec arm) — exec dtypes are **f32/f64 only**.
**Limitations:** contiguous zero-offset rank-4 only; batch prefix must match exactly across q/k/v
(no batch broadcast); `Hkv` must divide `Hq`; ALiBi slopes, when present, must be exactly `[Hq]`.

```fkc
kernel: attention_naive
fused_op: FLASH_ATTN              # FusedOpId(12); the registry FLASH_ATTN exec arm uses this as oracle
blurb: "Materialized multi-head SDPA oracle; stable softmax; GQA/causal/window/softcap/ALiBi; fully-masked row -> zero."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::attention::attention_naive"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"  # Hkv, Sk, D match k exactly; B matches q
    - name: alibi_slopes            # optional 4th input; presence implicit in inputs.len()==4
      dtypes: [F32, F64, BF16, F16] # same T as q/k/v (RefTensor<T>)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      shape_constraint: "dim[0]=q.dim[1]"   # exactly [Hq]
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "None ⇒ full Sk (oracle uses full extent); rides SymEnv when Some" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # RefTensor is contiguous-only; planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch" }
  in_place: false
  alignment_bytes: 64               # host buffer
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; flops/bytes hints below are derivable but coefficients are calibrated
  class: attention
  # QK^T (2·B·Hq·Sq·Sk·D) + PV (2·B·Hq·Sq·Sk·D) over the materialized matrix; oracle attends full Sk.
  flops: "4 * b * hq * sq * sk * d"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: ~                    # launch cost not authored — judge_measured (no authored constant under judge_measured)
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic nested loops; no atomics, fixed reduction order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "math in T (no f32-widened accumulator); precision-sensitive at bf16/f16. Stable softmax (subtract row max). Fully-masked row -> all-zero output row."

determinism: same_hardware_bitwise
```

## attention_flash  (FlashAttention-v2 forward; tiled online softmax, never materializes the matrix)

FlashAttention-v2 forward in pure Rust. Tiles `K`/`V` into `BC=16`-column blocks, processes them
with online softmax over `BR=16`-row Q tiles, and never materializes the full `[B,H,Sq,Sk]`
attention matrix.

For each `(batch, head)` it sweeps Q rows in `BR=16` tiles and, per tile, sweeps K/V columns in
`BC=16` blocks. Per Q-row online-softmax state is the running max `m`, running denominator `l`, and
the partial output accumulator `O`; on each K-block it computes block scores
`S = scale · QKᵀ` (+ softcap + ALiBi, same as naive), rescales `m`/`l`/`O` by `exp(m_old − m_new)`,
accumulates `P_ij = exp(S − m_new)` against `V`, and finalizes by dividing each row's accumulator
by `l`. The `BR`/`BC` constants are for cache friendliness only — correctness is independent of the
tile choice. **Numerics:** bit-for-bit equal to `attention_naive` **up to f32-associativity drift**
in the online partial sums (the rescale/accumulate reorders the additions); math runs in `T`. A
fully-masked Q row (`l == 0`) outputs zero. **Perf:** O(N·D) memory (only `m`/`l`/`O` per row, no
attention matrix), O(N²·D) compute — the Tier-1 deliverable Tiers 2/3 must match within the same
drift envelope. **Wiring:** this is **not** the executor `FLASH_ATTN` arm (exec routes FLASH_ATTN
to `attention_naive`); `attention_flash` is the algorithmic reference. **Limitations:** contiguous
zero-offset rank-4 only; GQA-divisible; ALiBi slopes exactly `[Hq]` when present.

```fkc
kernel: attention_flash
fused_op: FLASH_ATTN              # FusedOpId(12); same fused op as the materialized oracle — a sibling implementation at the key
blurb: "FlashAttention-v2 forward; tiled online softmax (BR=BC=16); never materializes the matrix; equals naive up to f32-assoc drift."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::attention::attention_flash"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
    - name: alibi_slopes            # optional 4th input; presence implicit in inputs.len()==4
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      shape_constraint: "dim[0]=q.dim[1]"
      optional: true
  op_params:
    variant: FlashAttn            # FusedOpParams::FlashAttn (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "None ⇒ full Sk; rides SymEnv when Some (live-prefix decode)" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # RefTensor contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch over K tiles" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; tiling makes the precise coefficients measurement-derived
  class: attention
  # QK^T + PV both 2·B·Hq·Sq·Sk·D over the live extent; tiled but same FLOP order as naive.
  flops: "4 * b * hq * sq * sk * d"
  bytes_moved: "b * (hq*sq*d + 2*hkv*sk*d + hq*sq*d) * dtype_bytes"
  overhead_ns: ~                    # launch cost not authored — judge_measured (no authored constant under judge_measured)
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic tile order; no atomics — re-run on same hardware is bit-identical
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "equals attention_naive up to f32-associativity drift in the online partial sums (rescale/accumulate reorders additions); math in T; fully-masked row -> zero. Deterministic per-hardware (fixed BR/BC sweep), but NOT bit-identical to naive."

determinism: same_hardware_bitwise
```

## attention_paged_naive  (paged-cache attention; per-sequence block table + variable context lengths)

Paged-cache attention reference. Reads K/V from a blocked KV pool via a per-sequence block table,
supporting variable per-sequence context lengths within one batch — the vLLM-style paged layout.

For each `(batch, head, query)` it walks logical key positions `0..context_lens[b]`, translating
each to a physical pool location via `block_table[b, k_pos / block_size]` and `k_pos % block_size`
(the pool is `[num_blocks, block_size, Hkv, D]`). The **causal mask is implicit** and tied to the
global decode position: query slot `q_pos` maps to absolute position
`context_lens[b] − Sq + q_pos`, and key positions strictly after that are masked. Scores get the
same `scale`/softcap/ALiBi treatment (ALiBi delta is `k_pos − abs_pos`), then a stable softmax and
weighted `V` accumulation; a fully-masked row outputs zero. **FDX framing (§3.9.1):** the KV pool is
an indexed-residency (gather) operand — `fdx.gather.kind: paged_blocks` over a contiguous `uint8`
block pool re-interpreted by the block table — and the per-sequence live length is the symbolic
extent (`context_lens`). Because this kernel takes the block table and context-lens as **separate
graph inputs** (the as-built operand order `[q, k_cache, v_cache, block_table, context_lens,
alibi?]`, `fuel-dispatch/src/kernel.rs:314-331`), they are ordinary `accept.inputs` operands and the
pool operand's `fdx.gather.block_table`/`context_lens` carry their **role names**, not a duplicate
table (single-place rule). **Numerics:** math in `T`; stable softmax. **Wiring:** registry
`PAGED_ATTN`; exec dtypes **f32/f64 only**. **Limitations:** contiguous zero-offset; `k_cache`/
`v_cache` rank-4 with matching `block_size`/`Hkv`/`D`; `Hkv` divides `Hq`; physical block index
bounds-checked against `num_blocks`.

```fkc
kernel: attention_paged_naive
fused_op: PAGED_ATTN             # FusedOpId(13); param carrier is FusedOpParams::PagedAttn (§3.9.1), NOT an OpParams variant
blurb: "Paged-cache attention; per-seq block table + variable context_lens; implicit causal mask at context_lens[b]-Sq+q_pos."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::attention::attention_paged_naive"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k_cache
      dtypes: [F32, F64, BF16, F16] # TRUE per-token pool element type (FDX FDXDTypeExt)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # physical pool [num_blocks, block_size, Hkv, D]
      fdx:
        requires_ext: true          # paged pool is mandatorily meaning-bearing (FDX gather V19)
        symbolic_extent: required    # per-seq live length is symbolic (context_lens)
        extent_kind: range           # single SymId per-seq length; affine is forward-looking (§3.9.2)
        gather:
          kind: paged_blocks         # FDX FDX_GATHER_PAGED_BLOCKS
          block_table: block_table   # role of the SEPARATE block-table accept.input (below)
          context_lens: context_lens # role of the SEPARATE context-lens accept.input (below)
    - name: v_cache
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
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
      rank: 2                        # [B, max_blocks_per_seq]
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                        # [B]
      fdx: { symbolic_extent: required }   # per-seq live lengths (data-determined)
    - name: alibi_slopes            # optional 6th input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                        # [Hq]
      shape_constraint: "dim[0]=q.dim[1]"
      optional: true
  op_params:
    variant: PagedAttn            # FusedOpParams::PagedAttn (fused namespace; §3.7, §3.9.1)
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, constraint: "== k_cache.dim[1] == v_cache.dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)        # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # RefTensor contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "softcap == false", note: "no softcap branch on scores" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; data-dependent context_lens makes work shape-vs-runtime, calibrated
  class: attention
  # Cost is over the LIVE per-seq context length; v1 evaluates the symbolic extent at CAPACITY.
  # Capacity bound = max_blocks_per_seq * block_size keys per sequence. Hint at capacity:
  flops: "4 * b * hq * sq * (block_table.dim[1] * block_size) * d"
  bytes_moved: "b * hq * sq * d * dtype_bytes + 2 * b * (block_table.dim[1] * block_size) * hkv * d * dtype_bytes"
  overhead_ns: ~                    # launch cost not authored — judge_measured (no authored constant under judge_measured)
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic walk over a fixed gather order; no atomics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "math in T; stable softmax; implicit causal mask at abs_pos = context_lens[b]-Sq+q_pos (saturating); fully-masked row -> zero; phys block index bounds-checked."

determinism: same_hardware_bitwise
```

## attention_flash_backward  (attention backward via recompute; 3-output bundle dQ/dK/dV)

Attention backward via recompute. Given `Q, K, V, dO` (and optional ALiBi), re-runs the forward
softmax per row to recover `P`, then walks the gradient closed form to produce `(dQ, dK, dV)` — a
**three-output bundle**. The recompute approach trades an extra forward pass for O(N·D) memory (no
saved attention matrix), matching what real FA backward kernels do.

For each `(batch, head, query)` it recomputes `S` and the stable-softmax `P[qi,:]` (including the
pre-softcap score for the softcap derivative), then accumulates: `dV[...,j,:] += P[i,j]·dO[i,:]`;
`dP[i,j] = dO[i,:]·V[j,:]`; the softmax backward `dS[i,j] = (dP[i,j] − Σ_j' P[i,j']·dP[i,j'])·P[i,j]`
followed — when softcap is set — by the softcap-derivative factor `1 − tanh²(s_pre/c)`;
`dQ[i,:] += scale·Σ_j dS[i,j]·K[j,:]`; and `dK[j,:] += scale·Σ_i dS[i,j]·Q[i,:]`. **GQA:** `dK`/`dV`
are shaped `[B, Hkv, Sk, D]` and **summed over the Q-group heads** (each KV head accumulates from all
Q heads in its group); `dQ` is `[B, Hq, Sq, D]`. A fully-masked row contributes nothing.
**Numerics:** math in `T`; recompute reproduces the forward softmax so gradients are consistent with
the forward oracle. **As-built dispatch split:** the live dispatch surface splits this into three
single-output FusedOpId-marked nodes — `FLASH_ATTN_BACKWARD_Q` (FusedOpId 22), `..._K` (23), `..._V`
(24) — all sharing the one `FusedOpParams::FlashAttnBackward` variant (the FusedOpId distinguishes
which gradient). This reference function emits all three at once; the contract below describes it as
a **multi-output bundle** (§5.5) faithful to the kernel, and notes the split for the dispatch mapping.
**Wiring:** **not** wired in the executor (multi-output) — functional-oracle only. **Limitations:**
contiguous zero-offset rank-4; GQA-divisible; ALiBi slopes exactly `[Hq]` when present.

```fkc
kernel: attention_flash_backward
fused_op: FLASH_ATTN_BACKWARD_Q  # FusedOpId(22); as-built the dQ/dK/dV gradients are FusedOpId 22/23/24 sharing FusedOpParams::FlashAttnBackward. This oracle emits all three (bundle); the split is a dispatch-mapping note.
blurb: "Attention backward via recompute; returns (dQ, dK, dV); dK/dV summed over GQA groups; not exec-wired (oracle only)."
backend: Cpu
kernel_source: "reference-oracle"
entry_point: "fuel_reference_backend::attention::attention_flash_backward"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hq, Sq, D]
    - name: k
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
    - name: do_grad               # upstream gradient dO; shape == forward out [B, Hq, Sq, D]
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=q"
    - name: alibi_slopes            # optional 5th input
      dtypes: [F32, F64, BF16, F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                       # [Hq]
      shape_constraint: "dim[0]=q.dim[1]"
      optional: true
  op_params:
    variant: FlashAttnBackward    # FusedOpParams::FlashAttnBackward (fused namespace; §3.7)
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  bundle:                          # 3-output bundle (§5.5); as-built split into 3 single-output FusedOpId nodes
    - { name: dq, dtype_rule: passthrough(q), shape_rule: same_as(q), layout_guarantee: contiguous }
    - { name: dk, dtype_rule: passthrough(k), shape_rule: same_as(k), layout_guarantee: contiguous }   # [B, Hkv, Sk, D]; summed over GQA groups
    - { name: dv, dtype_rule: passthrough(v), shape_rule: same_as(v), layout_guarantee: contiguous }   # [B, Hkv, Sk, D]; summed over GQA groups

caps:
  awkward_layout_strategy: requires_contiguous   # RefTensor contiguous-only; planner inserts Op::Contiguize + sums its cost
  fast_paths:
    - { when: "causal == false", note: "no causal-mask branch in recompute" }
    - { when: "softcap == false", note: "skips the 1 - tanh^2 softcap-derivative factor" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 8

cost:
  provenance: judge_measured        # Judge bootstraps; recompute + 3 grad walks make coefficients measurement-derived
  class: attention
  # Recompute forward (4·B·Hq·Sq·Sk·D) + dV/dP/dQ/dK gradient walks (each ~2·B·Hq·Sq·Sk·D).
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "b * (2*hq*sq*d + 2*hkv*sk*d) * dtype_bytes + b * (hq*sq*d + 2*hkv*sk*d) * dtype_bytes"
  overhead_ns: ~                    # launch cost not authored — judge_measured (no authored constant under judge_measured)
  memory: { device_bytes: 0, host_bytes: "b * (hq*sq*d + 2*hkv*sk*d) * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic recompute + nested accumulation; no atomics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "recompute matches forward softmax; math in T; softcap derivative 1 - tanh^2(s_pre/c) applied when softcap set; dK/dV summed over GQA groups; fully-masked row contributes nothing."

determinism: same_hardware_bitwise
```

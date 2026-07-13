---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: "fuel_dispatch::fkc::cuda_link::CudaLinkRegistry (production; CUDA_ROPE_APPLY_FUSED_ENTRY_POINTS chained into resolve_fused)"
  revision_base: "git:a7a4d223"
---

# fuel-cuda-backend — FUSED `FusedOps::ROPE` via baracuda's `rope_apply_<dt>_run`

This contract registers baracuda's caller-supplied-cos/sin `rope_apply_<dt>_run` FFI family
(`baracuda-kernels-sys` 0.0.1-alpha.77; see `docs/kernel-contracts/cuda/rope-apply.fkc.md` for the
raw FFI ABI notes) as a CUDA candidate for the **FUSED** `FusedOps::ROPE` dispatch key
(`fuel-graph/src/registry/rope.rs`), the key the real Llama decode graph's fused Rope node
(`Tensor::rope_with_tables`) actually builds. This is a **separate registration** from
`docs/kernel-contracts/cuda/rope-apply.fkc.md` (Task 4.6), which targets the **primitive**
`op_kind: Rope` key and resolves through a verification-harness-local `LinkRegistry`
(`fuel_dispatch::fkc::verify::harness::RopeApplyLinkRegistry`) — that contract is not production-wired
and does not serve `FusedOps::ROPE`. It is also separate from `docs/kernel-contracts/cuda/rope.fkc.md`
(baracuda's OTHER, device-computed-trig `rope_<dt>_run` family, no caller-supplied cos/sin at all),
which is the primitive `op_kind: Rope`'s SOLE production CUDA candidate today.

**Before this contract, `FusedOps::ROPE` had ZERO registered CUDA candidates** — every CUDA fused-op
family (`register_baracuda_cuda_kernels`, `fuel-dispatch/src/baracuda_dispatch.rs`) registers only
*primitive* `op_kind` contracts into a `KernelBindingTable`; `register_default_fused_kernels`
(`fuel-dispatch/src/dispatch.rs`) only imports **CPU** fused bundles
(`docs/kernel-contracts/fused/{linear-quant,norm-softmax,conv-rope}.fkc.md`). This section is
the FIRST fused-op contract the production `CudaLinkRegistry::resolve_fused` actually resolves —
that method was a permanent `None` stub before this change (no CUDA fused contract existed to
populate it).

**The half-width vs full-width cos/sin ABI gap — and its resolution.**

baracuda's `rope_apply_<dt>_run` wants **HALF-WIDTH** cos/sin tables `[seq, head_dim/2]` (one trig
value per rotation *pair*; see `rope-apply.fkc.md`'s ABI note). Fuel's real graph-level fused Rope
builder, `Tensor::rope_with_tables` (`fuel-graph/src/lib.rs:6423-6452`), **hard-asserts FULL-WIDTH**
`cos.shape() == sin.shape() == [seq, head_dim]` at graph-build time — the same convention the CPU
fused rope section (`docs/kernel-contracts/fused/conv-rope.fkc.md`'s `## rope` section) and the
primitive CUDA `rope.fkc.md` both declare. A contract that declared half-width accept-shapes (mirroring
`rope-apply.fkc.md` verbatim) would therefore NEVER match the real graph's operands — an unreachable
candidate. This contract instead declares the REAL full-width accept-shape (matching `x`, matching
`conv-rope.fkc.md`'s CPU section) and the **wrapper drivers narrow the tables before invoking the
kernel**.

**Correctness of the narrowing is derived, not assumed.** `Tensor::rope_with_tables_decomposed`
(`fuel-graph/src/lib.rs:6486-6509`) computes, for pair index `j` in `[0, half)`:

```text
out[j]      = x[j]      * cos[j]      - x[j+half] * sin[j]
out[j+half] = x[j+half] * cos[j+half] + x[j]      * sin[j+half]
```

For this to be the standard shared-angle RoPE rotation, `cos[j] == cos[j+half]` and
`sin[j] == sin[j+half]` for every `j` — i.e. Fuel's full-width convention is *by construction* the
half-width table duplicated across both halves. Extracting the first `head_dim/2` columns of the
full-width table is therefore byte-for-byte baracuda's half-width table, not an approximation.

**Implementation.** `fuel_cuda_backend::baracuda::attention::rope_apply_fused_<dt>_into` narrows
`cos`/`sin` via a single `cuMemcpy2DAsync` device-to-device copy of each row's first `head_dim/2` F32
elements into a freshly allocated, tightly-packed `[seq, head_dim/2]` buffer (mirrors the existing
production pattern `fuel_cuda_backend::baracuda::mamba::strip_prepad_d2d` uses for the causal_conv1d
pre-pad bridge — async-only, no host sync), then forwards to `rope_apply_<dt>_into`.

**KNOWN GAP, flagged not hidden**: the narrow-copy allocates two fresh device buffers per call
(mirrors `strip_prepad_d2d`'s existing alloc-per-call posture, not a capture-mode
"never-allocate" pattern like `WorkspaceCache`/`device.flash_workspace()`). This is correctness-safe
for ordinary (non-captured) `realize`, but does **not** yet satisfy CapturedRun's
zero-alloc-during-capture invariant — the very use case this registration exists to unblock. A
grow-only scratch-cache integration for the narrowed cos/sin buffers is a follow-up, not implemented
here; see the report for what the controller must verify/finish on GPU.

**cos/sin dtype**: always **F32**, independent of `x`'s dtype (baracuda ABI fact) — unlike the CPU
fused rope section and the primitive CUDA `rope.fkc.md`, which both keep cos/sin at the SAME dtype as
`x`. `x` fans over `{F32, F16, BF16, F64}`; `cos`/`sin` are pinned single-dtype `[F32]` (FKC §3.4:
a single-valued dtype list is a pinned constant, not a fan axis) — mirrors `rope-apply.fkc.md`'s
existing accept block exactly. The fanned key set is `[F32,F32,F32,F32]`, `[F16,F32,F32,F16]`,
`[BF16,F32,F32,BF16]`, `[F64,F32,F32,F64]`; the real Llama decode graph under test today needs only
the first.

---

## rope_apply_fused  (FUSED Rope — {F32, F16, BF16, F64} on x; cos/sin fixed F32, full-width; contiguous only)

Apply RoPE rotation to `x [outer_count, seq, head_dim]` using caller-supplied, FULL-WIDTH (Fuel
convention) `cos`/`sin` tables of shape `[seq, head_dim]`, always F32. Backs `FusedOps::ROPE` (the
fused dispatch key `Tensor::rope_with_tables` emits) as a CUDA candidate. The wrapper narrows
cos/sin to baracuda's half-width `[seq, head_dim/2]` convention before launch (see the ABI-gap note
above). Contiguous input only (mirrors `rope-apply.fkc.md`'s posture — no strided path). Output:
fresh, contiguous, no aliasing.

```fkc
kernel: rope_apply_fused
fused_op: ROPE
blurb: "Fused RoPE (rotate_half) via baracuda's caller-supplied-cos/sin rope_apply_<dt>_run (CUDA/baracuda) {F32,F16,BF16,F64} on x; cos/sin fixed F32 full-width [seq,head_dim], narrowed to half-width before launch; contiguous only."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::rope_apply_fused"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16, F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 3                              # [outer_count, seq, head_dim]
      shape_constraint: "divisible(x.dim[2], 2)"   # head_dim even (baracuda ABI requirement)
    - name: cos
      dtypes: [F32]                        # ALWAYS F32 regardless of x's dtype (baracuda ABI)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim] — FULL-WIDTH (Fuel's rope_with_tables convention), narrowed by the wrapper before launch
      shape_constraint: "last_dim_eq=x"    # head_dim matches x; seq matches x.dim[1]
    - name: sin
      dtypes: [F32]                        # ALWAYS F32 regardless of x's dtype (baracuda ABI)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [seq, head_dim]
      shape_constraint: "same_as=cos"
  op_params:
    variant: Rope                          # FusedOpParams::Rope (fused namespace; no fields; §3.7) — outer_count/seq/head_dim
    fields: {}                             # recovered from x's shape generically by pipelined.rs's Op::Fused(_, FusedOpParams::Rope) arm

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: same_as(x)               # [outer_count, seq, head_dim]
      layout_guarantee: contiguous
      aliasing: none                       # fresh buffer; baracuda's own doc comment: aliasing y with x/cos/sin is UNSAFE

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: normalization }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: normalization
  # Two rotation planes -> two FMA pairs per element (matches the CPU rope section's derivable
  # prior); bandwidth adds the narrow-copy's extra read+write of the half-width tables on top of
  # the base kernel's own table reads.
  flops: "4 * outer_count * seq * head_dim"
  bytes_moved: "(2 * outer_count * seq * head_dim) * dtype_bytes + (4 * seq * head_dim / 2) * 4"
  memory: { device_bytes: "outer_count * seq * head_dim * dtype_bytes + 2 * seq * (head_dim / 2) * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "deterministic rope apply (rope_apply_<dt>_run) fed a derived half-width cos/sin narrowed from Fuel's full-width tables via a deterministic cuMemcpy2DAsync D2D copy (no reduction, no cross-thread state). Author-declared audited: true, but the import-time V-FKC-9 gate (fkc::verify::gate_precision) downgrades this to UNAUDITED for any (backend, dtypes, kernel_revision_hash) tuple lacking a passing .fkc-verified-ledger.json entry — no such entry exists yet for this NEW fused registration (unlike the primitive rope-apply.fkc.md, whose ledger entry the Task 4.6 harness earned separately). Not yet cross-checked end-to-end on GPU: the full-to-half narrowing's mathematical correctness is derived in the module note above from rope_with_tables_decomposed, but the cuMemcpy2DAsync wiring itself is unverified by any compiler or test run in this environment (never built with --features cuda here) — the controller's GPU pass is the first real check."

determinism: same_hardware_bitwise
```

---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend — FlashDecoding (decode-flash) kernel contract

The FIRST capacity-K CUDA `OpKind::FlashAttn` binding: the baracuda `flash_decoding_{f16,bf16}`
decode kernel (alpha.72; stride-decoupled, GQA-native), adapted to the `KernelRef` signature in
[`crate::baracuda_dispatch::flash_decoding`]. This is the NO-alibi **decode** arm (`seq_q == 1`): a
single query attends the whole live `[0, k_len)` prefix of a fixed-capacity KV cache. Three inputs
`q, k, v` (no `alibi_slopes`), keyed `[q, k, v, out] = [T; 4]` over `{F16, BF16}` — byte-for-byte the
two hand-written `register_full(FlashAttn, [f16;4] / [bf16;4], …)` regs it replaces. `sk` is the
**physical** K/V capacity (strides + byte-length key off it); `k_len ≤ sk` is the live attended
prefix resolved per token from the `SymEnv` in `pipelined.rs`. The wrapper forwards the per-tensor
input `Layout`s so it derives capacity strides directly (no Contiguize) — hence `handles_strided`
(caps project `strided_input = true`, AND-ed across q/k/v: each accepts `strided` + `broadcast_stride0`).

**Cost is CONTRACT-PINNED (§4.4 cost-fn trampoline, Task-F).** Unlike every other migrated CUDA
family (which registers the `unknown_cost` sentinel and lets `fill_unset_cost_for_backend` upgrade it
to the shared per-`OpKind` cost fn), this section NAMES `cost_flash_decoding_cuda` in its `cost.cost_fn`.
The importer resolves that name through [`crate::fkc::CudaLinkRegistry`]'s `CUDA_COST_FNS` table and
stamps THAT `CostFn` on the binding, which SURVIVES the fill pass (the fill pass only replaces the
`unknown_cost` sentinel). `cost_flash_decoding_cuda` doubles as the decode kernel's **static ranker
gate**: it returns an INFEASIBLE cost (`composite_ns → u64::MAX`) for any shape the kernel cannot do —
`seq_q != 1`, or `head_dim` outside `[1, 128]` — so the ranker keeps the decomposed base map / CPU
oracle for those and never PLACES the decode kernel on an unsupported shape (Fuel dispatch is
fail-fast: a registered kernel that returns `Err` fails `realize`). Losing this gate — which is
exactly what a plain `unknown_cost` import + `fill_unset` would do (upgrading it to the shape-blind
shared `FlashAttn` cost) — is what Task-F prevents.

`window` / `softcap` / `alibi` are NOT implemented by the decode kernel and are statically excluded:
the dtype key gates the dtype axis, the cost gate deprioritizes the shape axis, and the wrapper
hard-errors defensively on `window`/`softcap` (fail-fast). The `causal` flag is accepted-and-ignored
(at `seq_q == 1` the single query attends the whole prefix either way).

---

## flash_decoding  (FlashAttn decode arm over a fixed-capacity KV cache — {F16, BF16}, capacity-K)

Decode-flash SDPA (`seq_q == 1`). `q [B, Hq, 1, D]`, `k`/`v [B, Hkv, Sk, D]` with GQA grouping
(`Hq % Hkv == 0`). Attends the live prefix `k_len ≤ Sk` (capacity-K; `k_len` rides the `SymEnv`).
Fans `q/k/v` over `{F16, BF16}` (base `entry_point` → `flash_decoding_<dtype>` resolved through
[`crate::fkc::CudaLinkRegistry`]); output `passthrough(q)`. Consumes per-tensor capacity strides
directly (no Contiguize). Cost is the CONTRACT-PINNED `cost_flash_decoding_cuda` static gate.

```fkc
kernel: flash_decoding
op_kind: FlashAttn
blurb: "FlashDecoding decode arm (CUDA/baracuda) {F16, BF16}; seq_q==1 over a fixed-capacity KV cache; attends live prefix k_len <= Sk; GQA; no alibi/window/softcap; capacity-strided."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::flash_decoding"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, 1, D]  (seq_q == 1)
    - name: k
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]  (Sk = CAPACITY; strides key off it)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required          # reads live k_len from SymEnv; strides keyed to capacity Sk
        extent_kind: range                 # single bounded SymId: k_len <= Sk
    - name: v
      dtypes: [F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"        # k_len ≡ v_len ⇒ SAME SymId
      fdx: { symbolic_extent: required, extent_kind: range }
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) is carried by the operand SHAPES / KernelRef, not this variant.
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool, note: "accepted-and-ignored at seq_q==1 (single query attends the whole prefix)" }
      window_size_left:  { kind: "Option<usize>", note: "unimplemented; wrapper hard-errors (fail-fast) and the ranker excludes it" }
      window_size_right: { kind: "Option<usize>", note: "unimplemented; wrapper hard-errors (fail-fast) and the ranker excludes it" }
      softcap:           { kind: "Option<f32>", note: "unimplemented; wrapper hard-errors (fail-fast) and the ranker excludes it" }
      k_len:             { kind: "Option<DynScalar>", note: "live attended length <= sk; None ⇒ k_len==Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)           # [B, Hq, 1, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer

caps:
  awkward_layout_strategy: handles_strided # capacity strides consumed directly (no Contiguize) ⇒ strided_input=true
  fast_paths:
    - { when: "k_len == sk", note: "static path; full-capacity prefix" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: declared                     # author prior + CONTRACT-PINNED cost fn below
  class: attention
  # §4.4 cost-fn trampoline (Task-F): PIN the real, shape-aware CostFn. It doubles as the static
  # ranker gate — INFEASIBLE cost for seq_q!=1 / head_dim outside [1,128] — resolved through
  # CudaLinkRegistry's CUDA_COST_FNS table and stamped on the binding, surviving fill_unset.
  cost_fn: cost_flash_decoding_cuda
  # The symbolic hints (decode: QK^T + softmax·V over the live prefix) are docs/telemetry; the pinned
  # cost fn above is authoritative.
  flops: "4 * b * hq * k_len * d"
  bytes_moved: "(2 * b * hq * d + 2 * b * hq * k_len * d) * dtype_bytes"
  overhead_ns: 200
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true        # author-declared; `audited: false` ⇒ lowers to PrecisionGuarantee::UNAUDITED, byte-for-byte the hand-written flash_precision default
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                           # UNAUDITED author seed (the hand-written reg stamped PrecisionGuarantee::UNAUDITED)
  notes: "baracuda flash_decoding decode kernel; f16/bf16 I/O with f32 compute; capacity-K over the live prefix; UNAUDITED author seed (not yet Judge-audited)."

determinism: same_hardware_bitwise
```

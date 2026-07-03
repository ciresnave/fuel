---
fkc_version: 1
provider:
  name: fuel-cpu-backend
  backend: Cpu                       # maps to BackendId::Cpu
  kernel_source: "portable-cpu"      # the BindingEntry.kernel_source tag
  link_registry: fuel_cpu_backend::byte_kernels::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-cpu-backend — attention kernel contracts

Attention kernels for the portable `CpuStorageBytes` surface: forward FlashAttn (naive math-def
SDPA over a fixed-capacity KV cache), FlashAttn backward (recompute-based dQ/dK/dV), and PagedAttn
(naive attention over a paged/blocked KV cache). Each is monomorphized over the four float dtypes
`{F32, F64, BF16, F16}` — every dtype is a distinct registered kernel (distinct `entry_point` →
`KernelRef`), sharing one accept/return shape but differing in element width and the
accumulation/narrowing rule (half floats `bf16`/`f16` widen each operand to **f32**, do the dot
product / softmax / output accumulation in f32, then narrow on store; `f32`/`f64` are native).

These contracts model the **`KernelBindingTable` dispatch path**: each is a primitive `op_kind:`
section — `OpKind::FlashAttn`, the three `OpKind::FlashAttnBackward{Q,K,V}` selectors, and
`OpKind::PagedAttn` (`fuel-ir/src/dispatch.rs`) — bound at a per-operand dtype key with the
softmax/mask/geometry knobs riding in the matching `OpParams` variant (the primitive namespace,
§3.7). The forward FlashAttn AND the three backward selectors share the single **`OpParams::FlashAttn`**
param carrier (`fuel-dispatch/src/kernel.rs:299`, `{b, hq, hkv, sq, sk, d, softmax_scale, causal,
window_size_left, window_size_right, softcap, k_len}` — there is no dedicated backward variant, so
each backward section names `op_params.variant: FlashAttn` too); PagedAttn uses **`OpParams::PagedAttn`**
(`kernel.rs:320`, `{b, hq, sq, d, softmax_scale, block_size, softcap, num_blocks}`). A primitive
contract compiles its cost to the **primitive** cost-fn shape `fn(&[Shape], &OpParams,
&BackendCapabilities)` — the shape the CPU cost dispatcher (`default_cost_for_op_kind`, arms
`FlashAttn`/`PagedAttn`) fills via `fill_unset_cpu_cost` (§4.4 / §12.3).

These families ALSO have a **separate** `FusedKernelRegistry` seam — `FLASH_ATTN = FusedOpId(12)`,
`PAGED_ATTN = FusedOpId(13)`, and `FLASH_ATTN_BACKWARD_{Q,K,V}` (`fuel-graph/src/registry.rs`),
dispatched when the graph carries an `Op::Fused(FLASH_ATTN*)` node and hand-registered in
`register_default_fused_kernels`. That seam is **not** described here; it stays hand-written and can
be FKC-modeled later when the fused import seam goes live (the importer registers primitive
`op_kind:` sections onto the binding table, not the fused registry). The per-section prose below may
still name the `FLASH_ATTN*` fused id — that refers to this separate seam, not to how the section
registers.

A load-bearing distinction: the **shape geometry** (`b, hq, hkv, sq, sk, d`, block sizes) is carried
by the operand shapes / `KernelRef` payload, NOT by the `OpParams` variant beyond the geometry
fields the carrier already holds. The cost/shape symbols below name those geometry dims by role for
the importer to bind from the shapes.

**PagedAttn is DESCRIBE-ONLY here (`registrable: false`, §3.10).** Its paged KV pool carries an
`fdx.gather: paged_blocks` sidecar (§3.9.1) whose FDX gather codes are [consumer-ahead] — the
importer's VALIDATE pass returns `GatherNotYetSupported` for such an operand — so the four paged
sections are documentation, not a registration target; the production `PagedAttn` binding stays
hand-written until the FDX gather codes land. (The importer's validate pass SKIPS describe-only
sections, so these four do not block the bundle's importable FlashAttn sections.)

These kernels are the production `CpuStorageBytes` path the dispatch wrapper
(`fuel_dispatch::dispatch::cpu_wrappers`) extracts and calls; they consume flat contiguous,
zero-offset, row-major slices plus explicit `usize` geometry, never a `Layout`/strides/offset (the
executor's auto-Contiguize pass realizes any strided/broadcast/offset input first). Validation is
byte-length checks returning `Result`, never a panic on the production path.

---

## flash_attn_f32  (multi-head SDPA over a fixed-capacity KV cache, f32 native)

Naive (math-definition, **not tiled**) scaled-dot-product attention. `q [B, Hq, Sq, D]`,
`k`/`v [B, Hkv, Sk, D]` with GQA grouping (`Hq % Hkv == 0`, `groups = Hq/Hkv`, `kv_h = hi/groups`);
optional 4th input `alibi_slopes [Hq]`. The K/V `Sk` axis is the **physical capacity** (strides and
byte-length checks key off it: `k.len_bytes == B·Hkv·Sk·D·4`); the kernel attends only the first
`k_len ≤ Sk` rows (the live prefix of a fixed-capacity KV cache, `byte_kernels.rs:6085-6089` rejects
`k_len > sk`) and bottom-right-aligns the causal mask at `causal_offset = k_len − Sq` (so query row
`qi` sits at absolute position `aq = qi + causal_offset`, `:6090`/`:6104`). `k_len` is a dynamic
scalar resolved per token via the `SymEnv` (`FusedOpParams::FlashAttn.k_len: Option<DynScalar>`,
`registry.rs:237`); `None` ⇒ `k_len == Sk` (the static path, byte-identical to a plain `0..Sk`
loop). Per (batch, head, query) it builds admissible scores over `0..k_len` (`flash_attn_admissible`
applies causal + sliding-window `(left,right)` masks, `:6011-6017`), subtracts the row max, exps,
normalizes (`inv_sum = 1/Σ`), and accumulates `Σ p·v` into the output row. Optional `softcap`
(`s = tanh(s/c)·c`) and ALiBi bias (`s += slope·(kj − aq)`) are applied to raw scores before the
softmax. Native f32 arithmetic throughout (`flash_attn_native_kernel!`, `byte_kernels.rs:6021`;
instantiated `:6166`). Output is zeroed up front so masked / fully-inadmissible rows stay zero
(`:6093`); fully-overwritten otherwise. Numerics: max-subtract softmax (stable). Known limitations:
contiguous zero-offset only (planner must Contiguize strided/broadcast/offset inputs first); not a
streaming/tiled flash kernel (allocates per-row `scores`/`admissible` Vecs of length `k_len`); no
in-place; `k_len ≤ sk` enforced at runtime.

```fkc
kernel: flash_attn_f32
op_kind: FlashAttn
blurb: "Naive multi-head SDPA over a fixed-capacity KV cache, f32 native; attends live prefix k_len <= Sk; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::flash_attn_f32"
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
      rank: 4                              # [B, Hkv, Sk, D]  (Sk = CAPACITY; strides key off it)
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
      fdx:
        symbolic_extent: required          # reads live k_len from SymEnv; strides keyed to capacity Sk
        extent_kind: range                 # single bounded SymId: k_len <= Sk (as-built flash path)
    - name: v
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"        # k_len ≡ v_len ⇒ SAME SymId
      fdx: { symbolic_extent: required, extent_kind: range }
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
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "live attended length <= sk; None ⇒ k_len==Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)           # [B, Hq, Sq, D]; symbolic Sq preserved
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, zeroed then accumulated

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to plain 0..Sk loop" }
    - { when: "causal == false", note: "no causal mask branch (window may still mask)" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Symbolic over the live k_len; v1 evaluates at CAPACITY (sk). QK^T (2·D MACs/score) + PV (2·D
  # MACs/score) => 4·B·Hq·Sq·k_len·D FLOPs; live-k_len re-eval is [consumer-ahead] (§4.4).
  flops: "4 * b * hq * sq * k_len * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic naive loop; native f32; row-max-subtract softmax
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive-class: family default applies (§4.8/§12.4)
  notes: "native f32 throughout; max-subtract stable softmax; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

## flash_attn_f64  (multi-head SDPA over a fixed-capacity KV cache, f64 native)

Identical algorithm, masking, GQA grouping, `k_len ≤ Sk` semantics, and bottom-right causal
alignment as `flash_attn_f32`, evaluated in native f64 throughout (`flash_attn_native_kernel!`
instantiated `byte_kernels.rs:6167`; 8-byte element, byte-length checks against `·8`). f64 gives the
widest precision of the family — no widen/narrow round-trip. Same zero-then-accumulate output,
same per-row `scores`/`admissible` Vec allocation, same `softcap`/ALiBi raw-score adjustments.
Limitations match `flash_attn_f32`: contiguous zero-offset only, not tiled, no in-place,
`k_len ≤ sk` enforced at runtime.

```fkc
kernel: flash_attn_f64
op_kind: FlashAttn
blurb: "Naive multi-head SDPA over a fixed-capacity KV cache, f64 native; attends live prefix k_len <= Sk; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::flash_attn_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: alibi_slopes
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "live attended length <= sk; None ⇒ k_len==Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to plain 0..Sk loop" }
    - { when: "causal == false", note: "no causal mask branch" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * k_len * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 throughout; widest precision of the family (no widen/narrow round-trip); deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## flash_attn_bf16  (multi-head SDPA over a fixed-capacity KV cache, bf16 I/O with f32 compute)

The `flash_attn_half_kernel!`-instantiated bf16 kernel (`byte_kernels.rs:6171`, instantiated
`:6311`). Same algorithm, masking, GQA grouping, `k_len ≤ Sk` semantics, and bottom-right causal
alignment as `flash_attn_f32`, but **bf16 in/out with f32 compute**: each `q`/`k`/`v` element is
widened via `.to_f32()`, the dot product, `softmax_scale`, `softcap`, ALiBi, max-subtract softmax,
and the `Σ p·v` output accumulation all run in f32 (an explicit per-row f32 `row_acc` buffer,
`:6286-6295`), then `<bf16>::from_f32(...)` narrows on store (`:6301`). This is the family's
load-bearing precision invariant: compute is f32, only I/O is bf16. 2-byte element; byte-length
checks against `·2`. Limitations match the family: contiguous zero-offset only, not tiled, no
in-place, `k_len ≤ sk` enforced at runtime.

```fkc
kernel: flash_attn_bf16
op_kind: FlashAttn
blurb: "Naive multi-head SDPA over a fixed-capacity KV cache, bf16 I/O with f32 compute; attends live prefix k_len <= Sk; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::flash_attn_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: alibi_slopes
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "live attended length <= sk; None ⇒ k_len==Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to plain 0..Sk loop" }
    - { when: "causal == false", note: "no causal mask branch" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * k_len * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O; max-subtract stable softmax; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## flash_attn_f16  (multi-head SDPA over a fixed-capacity KV cache, f16 I/O with f32 compute)

The `flash_attn_half_kernel!`-instantiated f16 kernel (`byte_kernels.rs:6312`). Byte-for-byte the
same code path as `flash_attn_bf16` with `half::f16` substituted for `half::bf16`: f32-compute
round-trip (widen on load, `<f16>::from_f32(...)` narrow on store), same masking / GQA /
`k_len ≤ Sk` / bottom-right causal semantics, same per-row f32 `row_acc`. Differs from bf16 only in
the IEEE half-precision storage format (10-bit mantissa vs bf16's 7-bit, narrower exponent range).
2-byte element. Limitations match the family: contiguous zero-offset only, not tiled, no in-place,
`k_len ≤ sk` enforced at runtime.

```fkc
kernel: flash_attn_f16
op_kind: FlashAttn
blurb: "Naive multi-head SDPA over a fixed-capacity KV cache, f16 I/O with f32 compute; attends live prefix k_len <= Sk; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::flash_attn_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k.dim[1])"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: v
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k"
      fdx: { symbolic_extent: required, extent_kind: range }
    - name: alibi_slopes
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: FlashAttn
    fields:
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }
      k_len:             { kind: "Option<DynScalar>", note: "live attended length <= sk; None ⇒ k_len==Sk; rides SymEnv" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "k_len == sk", note: "static path; byte-identical to plain 0..Sk loop" }
    - { when: "causal == false", note: "no causal mask branch" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * k_len * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half); max-subtract stable softmax; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_q_f32  (FlashAttn backward dQ (recompute), native f32 throughout)

Recompute-based FlashAttn backward producing the **dQ** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f32_inner (byte_kernels.rs:6468)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f32`
(`byte_kernels.rs:6625, instantiated :6703`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = Q` (dQ). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_Q`** (`FusedOps::FLASH_ATTN_BACKWARD_Q`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dQ` has q's shape `[B, Hq, Sq, D]` and q's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f32 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_q_f32
op_kind: FlashAttnBackwardQ
blurb: "FlashAttn backward dQ from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dQ (= q shape); native f32 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_f32_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
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
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq                           # dQ; this id (FLASH_ATTN_BACKWARD_Q) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ => q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32 recompute backward; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_k_f32  (FlashAttn backward dK (recompute), native f32 throughout)

Recompute-based FlashAttn backward producing the **dK** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f32_inner (byte_kernels.rs:6468)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f32`
(`byte_kernels.rs:6625, instantiated :6703`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = K` (dK). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_K`** (`FusedOps::FLASH_ATTN_BACKWARD_K`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dK` has k's shape `[B, Hkv, Sk, D]` and k's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f32 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_k_f32
op_kind: FlashAttnBackwardK
blurb: "FlashAttn backward dK from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dK (= k shape); native f32 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_f32_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardK), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk                           # dK; this id (FLASH_ATTN_BACKWARD_K) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(k)            # dK => k shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32 recompute backward; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_v_f32  (FlashAttn backward dV (recompute), native f32 throughout)

Recompute-based FlashAttn backward producing the **dV** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f32_inner (byte_kernels.rs:6468)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f32`
(`byte_kernels.rs:6625, instantiated :6703`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = V` (dV). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_V`** (`FusedOps::FLASH_ATTN_BACKWARD_V`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dV` has v's shape `[B, Hkv, Sk, D]` and v's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f32 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_v_f32
op_kind: FlashAttnBackwardV
blurb: "FlashAttn backward dV from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dV (= v shape); native f32 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_f32_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardV), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv                           # dV; this id (FLASH_ATTN_BACKWARD_V) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(v)            # dV => v shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f32 recompute backward; deterministic; not bit-stable cross-hardware (FMA contraction may differ)."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_q_f64  (FlashAttn backward dQ (recompute), native f64 throughout)

Recompute-based FlashAttn backward producing the **dQ** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f64_inner` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f64`
(`flash_attn_backward_native_wrapper! instantiated byte_kernels.rs:6704`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = Q` (dQ). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_Q`** (`FusedOps::FLASH_ATTN_BACKWARD_Q`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dQ` has q's shape `[B, Hq, Sq, D]` and q's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f64 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_q_f64
op_kind: FlashAttnBackwardQ
blurb: "FlashAttn backward dQ from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dQ (= q shape); native f64 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_f64_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do_grad
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F64]
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
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq                           # dQ; this id (FLASH_ATTN_BACKWARD_Q) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ => q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 recompute backward; widest precision of the family (no widen/narrow round-trip); deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_k_f64  (FlashAttn backward dK (recompute), native f64 throughout)

Recompute-based FlashAttn backward producing the **dK** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f64_inner` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f64`
(`flash_attn_backward_native_wrapper! instantiated byte_kernels.rs:6704`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = K` (dK). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_K`** (`FusedOps::FLASH_ATTN_BACKWARD_K`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dK` has k's shape `[B, Hkv, Sk, D]` and k's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f64 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_k_f64
op_kind: FlashAttnBackwardK
blurb: "FlashAttn backward dK from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dK (= k shape); native f64 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_f64_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do_grad
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardK), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk                           # dK; this id (FLASH_ATTN_BACKWARD_K) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(k)            # dK => k shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 recompute backward; widest precision of the family (no widen/narrow round-trip); deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_v_f64  (FlashAttn backward dV (recompute), native f64 throughout)

Recompute-based FlashAttn backward producing the **dV** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f64_inner` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f64`
(`flash_attn_backward_native_wrapper! instantiated byte_kernels.rs:6704`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = V` (dV). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_V`** (`FusedOps::FLASH_ATTN_BACKWARD_V`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dV` has v's shape `[B, Hkv, Sk, D]` and v's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Native f64 throughout.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_v_f64
op_kind: FlashAttnBackwardV
blurb: "FlashAttn backward dV from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dV (= v shape); native f64 throughout; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_f64_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "divisible(q.dim[1], k.dim[1])"   # GQA: Hq % Hkv == 0
    - name: v
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hkv, Sk, D]
      shape_constraint: "same_as=k"
    - name: do_grad
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardV), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv                           # dV; this id (FLASH_ATTN_BACKWARD_V) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(v)            # dV => v shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 recompute backward; widest precision of the family (no widen/narrow round-trip); deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_q_bf16  (FlashAttn backward dQ (recompute), bf16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dQ** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_bf16_inner (byte_kernels.rs:6612)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_bf16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6707, instantiated :6785`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = Q` (dQ). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_Q`** (`FusedOps::FLASH_ATTN_BACKWARD_Q`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dQ` has q's shape `[B, Hq, Sq, D]` and q's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Bf16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_q_bf16
op_kind: FlashAttnBackwardQ
blurb: "FlashAttn backward dQ from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dQ (= q shape); bf16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_bf16_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
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
    - name: do_grad
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [BF16]
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
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq                           # dQ; this id (FLASH_ATTN_BACKWARD_Q) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ => q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O; deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_k_bf16  (FlashAttn backward dK (recompute), bf16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dK** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_bf16_inner (byte_kernels.rs:6612)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_bf16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6707, instantiated :6785`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = K` (dK). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_K`** (`FusedOps::FLASH_ATTN_BACKWARD_K`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dK` has k's shape `[B, Hkv, Sk, D]` and k's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Bf16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_k_bf16
op_kind: FlashAttnBackwardK
blurb: "FlashAttn backward dK from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dK (= k shape); bf16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_bf16_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
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
    - name: do_grad
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardK), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk                           # dK; this id (FLASH_ATTN_BACKWARD_K) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(k)            # dK => k shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O; deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_v_bf16  (FlashAttn backward dV (recompute), bf16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dV** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_bf16_inner (byte_kernels.rs:6612)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_bf16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6707, instantiated :6785`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = V` (dV). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_V`** (`FusedOps::FLASH_ATTN_BACKWARD_V`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dV` has v's shape `[B, Hkv, Sk, D]` and v's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. Bf16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_v_bf16
op_kind: FlashAttnBackwardV
blurb: "FlashAttn backward dV from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dV (= v shape); bf16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_bf16_cpu_wrapper"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
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
    - name: do_grad
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardV), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv                           # dV; this id (FLASH_ATTN_BACKWARD_V) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(v)            # dV => v shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 I/O; deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_q_f16  (FlashAttn backward dQ (recompute), f16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dQ** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f16_inner (byte_kernels.rs:6613)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6786`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = Q` (dQ). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_Q`** (`FusedOps::FLASH_ATTN_BACKWARD_Q`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dQ` has q's shape `[B, Hq, Sq, D]` and q's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. F16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_q_f16
op_kind: FlashAttnBackwardQ
blurb: "FlashAttn backward dQ from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dQ (= q shape); f16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_q_f16_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F16]
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
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dq                           # dQ; this id (FLASH_ATTN_BACKWARD_Q) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(q)            # dQ => q shape [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half); deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_k_f16  (FlashAttn backward dK (recompute), f16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dK** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f16_inner (byte_kernels.rs:6613)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6786`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = K` (dK). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_K`** (`FusedOps::FLASH_ATTN_BACKWARD_K`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dK` has k's shape `[B, Hkv, Sk, D]` and k's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. F16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_k_f16
op_kind: FlashAttnBackwardK
blurb: "FlashAttn backward dK from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dK (= k shape); f16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_k_f16_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardK), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dk                           # dK; this id (FLASH_ATTN_BACKWARD_K) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(k)            # dK => k shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half); deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## flash_attn_backward_v_f16  (FlashAttn backward dV (recompute), f16 I/O with f32 compute)

Recompute-based FlashAttn backward producing the **dV** gradient (matches the reference backend's
`attention_flash_backward` math, byte-storage flavor). The inner core `flash_attn_backward_f16_inner (byte_kernels.rs:6613)` computes **all three**
gradients (dQ, dK, dV) jointly into pre-allocated buffers from the forward inputs `q, k, v`, the upstream
gradient `do_grad`, and optional `alibi_slopes`; the byte wrapper `flash_attn_backward_f16`
(`flash_attn_backward_half_wrapper! byte_kernels.rs:6786`) selects which gradient to write into its **single** output buffer via the
`which: FaBackwardWhich {Q,K,V}` selector — this section pins `which = V` (dV). On the CPU path
the Q/K/V variants each recompute the same backward state (3x recompute total; GPU kernels can fuse it).
The op is dispatched as the fused id **`FLASH_ATTN_BACKWARD_V`** (`FusedOps::FLASH_ATTN_BACKWARD_V`,
`fuel-graph/src/registry.rs`), all three Q/K/V ids sharing the single param carrier
`FusedOpParams::FlashAttnBackward` (`registry.rs:295`), which carries the same
`{softmax_scale, causal, window_size_left, window_size_right, softcap}` shape params as the forward
`FlashAttn` so the recompute produces identical scores — note there is **no `k_len`** in backward (the
full `sk` extent is used). Output `dV` has v's shape `[B, Hkv, Sk, D]` and v's dtype; the wrapper
validates `out.len_bytes` against the selected gradient's element count. F16 I/O with f32 compute.
Limitations: contiguous zero-offset only; 3x recompute cost on CPU; no in-place.

```fkc
kernel: flash_attn_backward_v_f16
op_kind: FlashAttnBackwardV
blurb: "FlashAttn backward dV from (q, k, v, do_grad, [alibi_slopes]); recomputes softmax state; writes dV (= v shape); f16 I/O with f32 compute; GQA; causal/window/softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_dispatch::dispatch::flash_attn_backward_v_f16_cpu_wrapper"
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
    - name: do_grad
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # upstream grad, [B, Hq, Sq, D] (same as q)
      shape_constraint: "same_as=q"
    - name: alibi_slopes
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: FlashAttn                     # OpParams::FlashAttn — shared by the Q/K/V selectors (primitive namespace; §3.7)
    fields:
      # geometry (b,hq,hkv,sq,sk,d) carried by operand SHAPES / KernelRef; the Q/K/V distinction is the
      # OpKind (FlashAttnBackwardV), NOT a variant field. No k_len in backward (full sk extent).
      softmax_scale:     { kind: f32 }
      causal:            { kind: bool }
      window_size_left:  { kind: "Option<usize>" }
      window_size_right: { kind: "Option<usize>" }
      softcap:           { kind: "Option<f32>" }

return:
  outputs:
    - name: dv                           # dV; this id (FLASH_ATTN_BACKWARD_V) writes exactly this gradient
      dtype_rule: passthrough(q)
      shape_rule: same_as(v)            # dV => v shape [B, Hkv, Sk, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, full overwrite (copy_from_slice)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "causal == false", note: "no causal mask branch in the recompute" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # Recompute backward over the full Sk extent; this CPU path computes all three gradients per call
  # (3x recompute across the Q/K/V ids). ~3-4x the forward FLOPs over [B,Hq,Sq,Sk,D].
  flops: "12 * b * hq * sq * sk * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hkv * sk * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hkv * sk * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic recompute
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 I/O (IEEE half); deterministic recompute; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## paged_attn_f32  (attention over a paged/blocked KV cache, f32 native)

Naive attention over a vLLM-style **paged (blocked) KV cache**. `q [B, Hq, Sq, D]`; the cache is a
block pool `k_cache`/`v_cache [num_blocks, block_size, Hkv, D]` indexed per sequence by a
`block_table [B, max_blocks_per_seq]` (U32, logical→physical block id) and a `context_lens [B]`
(U32, true context length per sequence); optional `alibi_slopes [Hq]`
(`paged_attn_native_kernel!`, `byte_kernels.rs:6805`, instantiated `:6975`). GQA grouping is
`groups = Hq/Hkv`, `kv_h = hi/groups`. Causal masking is **implicit**: query `qi` is at absolute
position `q_pos_abs = ctx_len + qi − Sq` and admits keys `kj ≤ q_pos_abs` (`:6891`/`:6898`). For
each admissible `kj` the kernel maps it to `logical_block = kj / block_size`,
`block_off = kj % block_size`, looks up `physical_block = block_table[bi, logical_block]` (range-
checked against `num_blocks`, `:6901-6909`), and reads the K/V row at
`physical_block·(block_size·Hkv·D) + block_off·(Hkv·D) + kv_h·D` (`:6911-6914`). Per (batch, head,
query) it does the usual max-subtract softmax over `0..ctx_len` and accumulates `Σ p·v`. Runtime
checks: `block_size != 0`, `Hq % Hkv == 0`, `ctx_len ≤ max_blocks_per_seq·block_size`
(`:6877-6883`). Optional `softcap` (`tanh(s/c)·c`) and ALiBi (`s += slope·(kj − q_pos_abs)`) on raw
scores. Output zeroed up front so empty / fully-masked sequences (`ctx_len == 0`) stay zero
(`:6873-6876`). Dispatched as **`FusedOpParams::PagedAttn`** (`FusedOpId(13)`, `registry.rs:241`,
carrying `{softmax_scale, block_size, softcap}`); the remaining geometry
(`b, hq, hkv, sq, d, max_blocks_per_seq, num_blocks`) is carried by the operand shapes /
`KernelRef::PagedAttn` (operand order `[q, k_cache, v_cache, block_table, context_lens, alibi?]`,
`fuel-dispatch/src/kernel.rs:314-331`). Native f32 throughout. Limitations: contiguous zero-offset
only; naive per-row Vec allocation over `ctx_len`; no in-place; `Sq ≤ ctx_len` assumed by
`q_pos_abs = ctx_len + qi − Sq` (a longer query window would underflow the absolute position).

Per the FDX gather single-place rule (§3.9.1), the paged KV pool is described as an
`FDX_GATHER_PAGED_BLOCKS` operand and the `block_table` / `context_lens` are **separate
`accept.inputs`** (the as-built ABI passes them as their own graph inputs); the pool operand's
`fdx.gather.{block_table,context_lens}` name those input roles rather than duplicating the data.
Direct admission of the paged operand is gated on `Capability::DlpackExtGather` and is
**[consumer-ahead]** — the FDX gather codes are the 2026-06-17 addition with no code yet, so an
importer reaching the `gather` block before they land returns `GatherNotYetSupported` rather than
fabricating a descriptor.

```fkc
kernel: paged_attn_f32
registrable: false                     # DESCRIBE-ONLY: fdx.gather paged_blocks is [consumer-ahead] (§3.9.1); the production PagedAttn binding stays hand-written
op_kind: PagedAttn
blurb: "Naive attention over a paged/blocked KV cache, f32 native; per-seq block_table + context_lens; implicit causal; GQA; softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::paged_attn_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [B, Hq, Sq, D]
    - name: k_cache
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # physical pool [num_blocks, block_size, Hkv, D]
      shape_constraint: "divisible(q.dim[1], k_cache.dim[2])"   # GQA: Hq % Hkv == 0 (Hkv = pool dim[2])
      fdx:
        requires_ext: true                 # paged pool meaning needs the FDX gather sidecar (§3.9.1)
        symbolic_extent: required          # per-seq live length is context_lens (data-determined)
        gather:
          kind: paged_blocks               # FDX FDX_GATHER_PAGED_BLOCKS
          block_table: block_table         # role of the SEPARATE block-table accept.input (below)
          context_lens: context_lens       # role of the SEPARATE context-lens accept.input (below)
    - name: v_cache
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4                              # [num_blocks, block_size, Hkv, D]; shares block layout with k_cache
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                              # [B, max_blocks_per_seq]; logical→physical block id
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [B]; true context length per sequence
      fdx: { symbolic_extent: required }   # per-seq live lengths (data-determined sym)
    - name: alibi_slopes                   # optional 6th input
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                              # [Hq]
      optional: true
  op_params:
    variant: PagedAttn                     # OpParams::PagedAttn (primitive namespace; §3.7)  [describe-only: rule-7 namespace check skipped for registrable:false]
    fields:
      # geometry (b,hq,hkv,sq,d,max_blocks_per_seq,num_blocks) carried by operand SHAPES / KernelRef.
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, constraint: "block_size != 0; == k_cache.dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)           # [B, Hq, Sq, D]
      layout_guarantee: contiguous
      aliasing: none                       # fresh preallocated buffer, zeroed then accumulated

caps:
  awkward_layout_strategy: requires_contiguous   # ← planner inserts Op::Contiguize (itself an FKC kernel) + sums its cost
  fast_paths:
    - { when: "any_input_strided", class: attention, note: "no strided fast path; pool is dense+gathered" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  # ctx = per-seq context length (from context_lens; symbolic, evaluated at capacity
  # max_blocks_per_seq*block_size in v1). QK^T + PV = 4·D MACs/score over B·Hq·Sq·ctx scores.
  flops: "4 * b * hq * sq * ctx * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hq * sq * ctx * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic naive loop; native f32; max-subtract stable softmax
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                         # CPU primitive-class: family default applies (§4.8/§12.4)
  notes: "native f32 throughout; implicit causal; max-subtract stable softmax; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## paged_attn_f64  (attention over a paged/blocked KV cache, f64 native)

Identical algorithm, paged-cache indexing, implicit-causal masking, GQA grouping, and
`softcap`/ALiBi semantics as `paged_attn_f32`, evaluated in native f64 throughout
(`paged_attn_native_kernel!` instantiated `byte_kernels.rs:6976`; 8-byte K/V element, U32
block_table/context_lens unchanged). f64 gives the widest precision — no widen/narrow round-trip.
Same zero-then-accumulate output, same `ctx_len ≤ max_blocks_per_seq·block_size` and
`physical_block < num_blocks` runtime checks. Limitations match `paged_attn_f32`: contiguous
zero-offset only, naive per-row allocation, no in-place, `Sq ≤ ctx_len`.

```fkc
kernel: paged_attn_f64
registrable: false                     # DESCRIBE-ONLY: fdx.gather paged_blocks is [consumer-ahead] (§3.9.1); the production PagedAttn binding stays hand-written
op_kind: PagedAttn
blurb: "Naive attention over a paged/blocked KV cache, f64 native; per-seq block_table + context_lens; implicit causal; GQA; softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::paged_attn_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k_cache
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k_cache.dim[2])"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: v_cache
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      fdx: { symbolic_extent: required }
    - name: alibi_slopes
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: PagedAttn
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, constraint: "block_size != 0; == k_cache.dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "any_input_strided", class: attention, note: "no strided fast path; pool is dense+gathered" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 64

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * ctx * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hq * sq * ctx * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f64 throughout; widest precision of the family (no widen/narrow round-trip); implicit causal; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## paged_attn_bf16  (attention over a paged/blocked KV cache, bf16 I/O with f32 compute)

The `paged_attn_half_kernel!`-instantiated bf16 kernel (`byte_kernels.rs:6979`, instantiated
`:7140`). Same algorithm, paged-cache indexing, implicit-causal masking, GQA grouping, and
`softcap`/ALiBi semantics as `paged_attn_f32`, but **bf16 K/V/Q I/O with f32 compute**: each element
is widened via `.to_f32()`, the dot product / softmax / `Σ p·v` accumulation run in f32, then
`<bf16>::from_f32(...)` narrows on store. This is the family's precision invariant: compute is f32,
only the K/V/Q/out I/O is bf16 (the `block_table`/`context_lens` stay U32). 2-byte K/V element.
Limitations match the family: contiguous zero-offset only, naive per-row allocation, no in-place,
`Sq ≤ ctx_len`.

```fkc
kernel: paged_attn_bf16
registrable: false                     # DESCRIBE-ONLY: fdx.gather paged_blocks is [consumer-ahead] (§3.9.1); the production PagedAttn binding stays hand-written
op_kind: PagedAttn
blurb: "Naive attention over a paged/blocked KV cache, bf16 I/O with f32 compute; per-seq block_table + context_lens; implicit causal; GQA; softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::paged_attn_bf16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k_cache
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k_cache.dim[2])"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: v_cache
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      fdx: { symbolic_extent: required }
    - name: alibi_slopes
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: PagedAttn
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, constraint: "block_size != 0; == k_cache.dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "any_input_strided", class: attention, note: "no strided fast path; pool is dense+gathered" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * ctx * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hq * sq * ctx * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, bf16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); bf16 K/V/Q/out I/O (block_table/context_lens U32); implicit causal; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## paged_attn_f16  (attention over a paged/blocked KV cache, f16 I/O with f32 compute)

The `paged_attn_half_kernel!`-instantiated f16 kernel (`byte_kernels.rs:7141`). Byte-for-byte the
same code path as `paged_attn_bf16` with `half::f16` substituted for `half::bf16`: f32-compute
round-trip (widen on load, `<f16>::from_f32(...)` narrow on store), same paged-cache indexing,
implicit-causal masking, GQA grouping, and `softcap`/ALiBi semantics. Differs from bf16 only in the
IEEE half-precision storage format (10-bit mantissa vs bf16's 7-bit). 2-byte K/V element.
Limitations match the family: contiguous zero-offset only, naive per-row allocation, no in-place,
`Sq ≤ ctx_len`.

```fkc
kernel: paged_attn_f16
registrable: false                     # DESCRIBE-ONLY: fdx.gather paged_blocks is [consumer-ahead] (§3.9.1); the production PagedAttn binding stays hand-written
op_kind: PagedAttn
blurb: "Naive attention over a paged/blocked KV cache, f16 I/O with f32 compute; per-seq block_table + context_lens; implicit causal; GQA; softcap/alibi."
backend: Cpu
kernel_source: "portable-cpu"
entry_point: "fuel_cpu_backend::byte_kernels::paged_attn_f16"
kernel_revision_hash: auto

accept:
  inputs:
    - name: q
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
    - name: k_cache
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "divisible(q.dim[1], k_cache.dim[2])"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: v_cache
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 4
      shape_constraint: "same_as=k_cache"
      fdx:
        requires_ext: true
        symbolic_extent: required
        gather: { kind: paged_blocks, block_table: block_table, context_lens: context_lens }
    - name: block_table
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2
    - name: context_lens
      dtypes: [U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      fdx: { symbolic_extent: required }
    - name: alibi_slopes
      dtypes: [F16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1
      optional: true
  op_params:
    variant: PagedAttn
    fields:
      softmax_scale: { kind: f32 }
      block_size:    { kind: usize, constraint: "block_size != 0; == k_cache.dim[1]" }
      softcap:       { kind: "Option<f32>" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(q)
      shape_rule: from_params(q)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "any_input_strided", class: attention, note: "no strided fast path; pool is dense+gathered" }
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 16

cost:
  provenance: declared                   # author prior (overhead_ns launch cost); Judge refines the formula hints below (§4.4)
  class: attention
  flops: "4 * b * hq * sq * ctx * d"
  bytes_moved: "(2 * b * hq * sq * d + 2 * b * hq * sq * ctx * d) * dtype_bytes"
  overhead_ns: 4000
  memory: { device_bytes: 0, host_bytes: "b * hq * sq * d * dtype_bytes", disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true      # deterministic loop; f32 compute, f16 narrow on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "compute in f32 (widen on load, narrow on store); f16 K/V/Q/out I/O (IEEE half; block_table/context_lens U32); implicit causal; deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

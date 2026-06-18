---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                                  # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"                    # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"                    # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — elementwise (unary / binary / affine / clamp / powi / add_assign_scaled) kernel contracts

The Vulkan elementwise compute family: the 16-op unary selector, the 6-op binary selector, the
affine transform `y = x·mul + add`, the bounded `clamp`, the integer power `powi`, and the in-place
scaled accumulate `add_assign_scaled`. Slang/GLSL sources live in
`fuel-kernels-source/kernels/*.slang`, are AOT-compiled to the SPIR-V committed in
`fuel-vulkan-kernels/spv/*.spv` and registered in the `EMBEDDED` table
(`fuel-vulkan-kernels/src/lib.rs:39`); the Rust dispatch wrappers (param packing, layout gating,
validation) live in `fuel-vulkan-backend/src/lib.rs`.

**Family-wide facts (each kernel's section overrides where its inventory entry differs):**

- **Two distinct layout regimes — this family is NOT uniformly contiguous-only.** The f32 / f16 /
  f64 variants of `unary` / `binary` / `affine`, plus `clamp` and `powi`, carry a rank-4
  `(shape0..3, strides0..3)` Params block + a `flags` contiguity bit: when the contiguous flag is
  set they index linearly (fast path), otherwise they decompose the linear out-index into rank-4
  coords and apply **per-input strides** (`stride == 0 ⇒ broadcast`). These are therefore
  **strided + broadcast capable** (`awkward_layout_strategy: handles_strided`) but **NOT**
  non-zero-offset capable — any non-zero `byte_offset` is realized by an upstream `Op::Contiguize`
  before dispatch (inventory cross-cutting note; the kernel reads no element offset). The `*_bf16`
  variants of `unary` / `affine` (and `add_assign_scaled`) are instead **contiguous-only**,
  element-aligned movers (`requires_contiguous`) — the planner contiguizes a strided/broadcast/
  offset producer first and sums that `Op::Contiguize` contract's cost (§4.3 / §4.4). `binary_bf16`
  is the exception among the half variants: its strided path masks single lanes out of the packed
  u32, so it remains strided + broadcast capable (`handles_strided`) like the wide binary kernels.
- **`reverse_strides: rejected` everywhere in this file.** None of these kernels walks a signed
  (negative) stride. The strided variants decode rank-4 coords with **unsigned** strides; a flipped
  view feeding any of them is normalized to a non-negative copy by an upstream movement kernel
  (`strided_copy_signed_*`, a separate contract) before dispatch. A `flipped` operand is therefore
  never handed directly to these kernels.
- **Output contiguity is universal.** Every kernel writes its output via the linear dispatch index;
  none emits a strided or offset output. Output dtype = input/operand dtype, output shape = the
  (broadcasted, for binary) input shape, output layout = contiguous row-major, output is a fresh
  pre-allocated buffer with no aliasing — **except** `add_assign_scaled`, which is in-place on `dst`
  (`caps.in_place: true`, `aliasing: in_place(dst)`).
- **All math is f32; half narrows on store.** f32 computes natively; f64 uses native `double`
  (GLSL.std.450); bf16 is stored as packed-u16-in-u32 and computed at f32 (bf16↔f32 is the exact
  bit extension `bits << 16` on load, RNE upper-16 + canonical qNaN on store); f16 uses native
  `float16_t` with f32 intermediate. **`Gelu` in the unary selector is the tanh approximation, NOT
  erf** (inventory). The half-packing constraints are load-bearing limitations, declared per
  section (`unary_bf16`: `n` even; `binary_bf16`/`affine_bf16` even `out_size`; `affine_bf16`
  contiguous-only pair-thread).
- **Cost is `judge_measured` for every kernel in this file** — the Judge bootstraps and refines the
  empirical coefficients (§4.4); these are GPU dispatches whose launch overhead, occupancy, and
  bandwidth are device-specific and not author-derivable. Where the op genuinely admits a structural
  formula hint, it is recorded: every kernel here is **bandwidth-bound elementwise** — it touches
  `n` (= `out_size`) output elements with a fixed read/write set (`flops ≈ n`,
  `bytes_moved ≈ k · n · dtype_bytes` for `k` buffers touched). These are structural facts of the
  loop, not fabricated timings; `overhead_ns` (Vulkan command-buffer submit) and any absolute timing
  are left null for the Judge. `provenance: judge_measured` is a first-class, visible marker, not a
  hidden gap (§4.4 / §10.8a). `powi`'s `flops` is left null because the per-element op count scales
  with the exponent (special-cased `e=0/1/2/3`, else `pow`).
- **Precision** is author-declared as a Judge-audited seed (§4.8). The wide (f32/f64) elementwise
  ops are bit-stable on the same hardware (deterministic per-element dispatch, no FP reduction
  reordering); the half variants are bit-stable on the same hardware via the deterministic f32
  round-trip. `add_assign_scaled` is a plain per-element `dst += src·scale` (no atomics), so it too
  is bit-stable on the same hardware. Cross-hardware bit-stability is not claimed (transcendentals
  in the unary selector and `pow` differ across drivers).

---

## unary  (Neg / Sqr / Sqrt / Exp / Log / Sin / Cos / Tanh / Sigmoid / Silu / Gelu(tanh) / Relu / Step / Abs / Sign / Recip — f32)

Element-wise unary `out[i] = op(in[i])` over an f32 buffer, with a 16-op uniform `op_id` selector.

The Slang kernel (`unary.slang:63`) carries a rank-4 `(shape0..3, in_s0..3)` Params block plus a
`flags` bit0 contiguity flag: contiguous inputs index linearly (fast path); otherwise the linear
out-index is decomposed into rank-4 coords and read through the per-input strides (`stride == 0 ⇒
broadcast). It is therefore **strided + broadcast capable, offset-incapable** (a non-zero
`byte_offset` is handled by an upstream `Op::Contiguize`). All 16 ops compute at f32. One
dtype-monomorphized kernel backs the whole 16-op selector keyed by `op_id`; each concrete op pins a
distinct Fuel unary `OpKind`, so the section below is the representative contract (one `entry_point`,
`op_id`-selected). `Gelu` is the **tanh approximation**, not erf. Output: f32, input shape,
contiguous, fresh buffer, no aliasing.

```fkc
kernel: unary
op_kind: ReluElementwise      # representative; the same f32 kernel backs the 16-op op_id selector
                              # (Neg/Sqr/Sqrt/Exp/Log/Sin/Cos/Tanh/Sigmoid/Silu/Gelu(tanh)/Relu/
                              #  Step/Abs/Sign/Recip — each pins its own unary OpKind, same shape)
blurb: "Elementwise unary out[i]=op(in[i]) (f32); 16-op op_id selector; strided+broadcast; Gelu=tanh approx."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::unary_f32"   # wrapper lib.rs:9527; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4                  # rank-4 Params block; rank carried in op_params
      shape_constraint: same_as=out
  op_params:
    variant: Unary               # out_size, op_id, rank, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize, note: "= n, the output element count" }
      op_id:    { kind: u32, note: "0..15 selects Neg..Recip" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }   # flags bit0 set ⇒ linear index
    - { when: "any_input_strided", class: strided_elementwise }     # rank-4 coord decompose path
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # GPU dispatch; Judge bootstraps overhead/occupancy (§4.4)
  class: cheap_elementwise       # strided shapes hit strided_elementwise (fast_paths)
  flops: "n"                     # one op per element (op-dependent magnitude; Judge refines)
  bytes_moved: "2 * n * dtype_bytes"   # read in, write out — bandwidth-bound elementwise
  overhead_ns: ~                 # Vulkan command-buffer submit; judge_measured
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic per-element dispatch; no FP reduction reorder
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                 # author-declared seed; Judge audits transcendentals
  notes: "f32 math throughout. Gelu = tanh approximation (NOT erf). Transcendentals (Exp/Log/Sin/Cos/Tanh/Sigmoid) not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## unary_f16  (16-op unary selector, native f16)

Element-wise unary for `F16`, native `float16_t` storage with f32 intermediate math. Same 16-op
`op_id` selector, same rank-4 Params + `flags` bit0 contiguity model as `unary`, so it is likewise
**strided + broadcast capable, offset-incapable**. Each element widens to f32, applies the op,
narrows back to f16 on store. Output: f16, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: unary_f16
op_kind: ReluElementwise      # representative; 16-op op_id selector (same surface as unary)
blurb: "Elementwise unary (f16, native float16_t; f32 math, narrow on store); 16-op selector; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::unary_f16"   # wrapper lib.rs:8664; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Unary               # out_size, op_id, rank, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize }
      op_id:    { kind: u32, note: "0..15 selects Neg..Recip" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native float16_t storage, f32 intermediate, narrow on store. Gelu = tanh approximation. Transcendentals not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## unary_f64  (16-op unary selector, native f64)

Element-wise unary for `F64`, native `double` arithmetic (GLSL.std.450). Same 16-op `op_id`
selector and rank-4 Params + `flags` bit0 contiguity model as `unary`; **strided + broadcast
capable, offset-incapable**. Output: f64, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: unary_f64
op_kind: ReluElementwise      # representative; 16-op op_id selector (same surface as unary)
blurb: "Elementwise unary (f64, native double); 16-op selector; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::unary_f64"   # wrapper lib.rs:8680; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Unary               # out_size, op_id, rank, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize }
      op_id:    { kind: u32, note: "0..15 selects Neg..Recip" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 8 (f64)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native double (GLSL.std.450). Gelu = tanh approximation. Transcendentals not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

## unary_bf16  (16-op unary selector, bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise unary for `BF16`. Unlike the wide unary variants, this is the **contiguous-only**
pair-thread kernel (`unary_bf16.slang:71`): bf16 is stored as packed-u16-in-u32 and each thread
processes one u32 (two bf16 lanes), so it dispatches `n_pairs = n/2` threads and **`n` must be
even**. It carries no rank/shape/stride Params — element-aligned 1:1 only — so any strided /
broadcast / offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). Math is at f32 (bf16↔f32 exact `bits << 16` load,
RNE upper-16 + canonical qNaN store). The op list is the same 16-op surface as `unary`. Output:
bf16, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: unary_bf16
op_kind: ReluElementwise      # representative; 16-op op_id selector (same surface as unary)
blurb: "Elementwise unary (bf16 packed-u32 pair-thread; f32 math, narrow on store); CONTIGUOUS-ONLY; n must be even."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::unary_bf16"   # wrapper lib.rs:8777; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Unary               # bf16 path: n_pairs, op_id only (NO rank/shape/stride)
    fields:
      n_pairs: { kind: usize, note: "= n/2; one thread per packed u32 (two bf16 lanes)" }
      op_id:   { kind: u32, note: "0..15 selects Neg..Recip" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "dim[i] % 2 == 0", note: "n must be even; one u32 per thread = 2 bf16 lanes" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (bf16); read in, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Gelu = tanh approximation. Transcendentals not bit-stable cross-hardware. Requires even n."

determinism: same_hardware_bitwise
```

## binary  (Add / Sub / Mul / Div / Max / Min — f32)

Element-wise binary `out[i] = op(lhs[i], rhs[i])` over f32 buffers, with a 6-op `op_id` selector
(Add, Sub, Mul, Div, Max, Min).

The Slang kernel (`binary.slang:44`) carries per-operand rank-4 strides (`a_s0..3` / `b_s0..3`) and
a `flags` field where bit0 = `a` contiguous and bit1 = `b` contiguous (both-contiguous fast path).
It is therefore **per-operand strided + broadcast capable, offset-incapable** (`stride == 0 ⇒
broadcast); the output shape is the broadcasted shape. One dtype-monomorphized kernel backs the
6-op selector keyed by `op_id`; each concrete op pins a distinct binary `OpKind`. Output: f32,
broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: binary
op_kind: AddElementwise       # representative; the f32 kernel backs the 6-op op_id selector (Add/Sub/Mul/Div/Max/Min)
blurb: "Elementwise binary out[i]=op(lhs[i],rhs[i]) (f32); 6-op selector; per-operand strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::binary_f32"   # wrapper lib.rs:1722 (shared binary_typed_bytes :1628); §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params:
    variant: Binary              # out_size, op_id, rank, flags, shape0..3, a_s0..3, b_s0..3
    fields:
      out_size: { kind: usize, note: "= n, the broadcasted output element count" }
      op_id:    { kind: u32, note: "0..5 selects Add/Sub/Mul/Div/Max/Min" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = a contiguous, bit1 = b contiguous" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # per-operand rank-4 unsigned strides; broadcast via stride 0; NO offset
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }   # flags bit0 & bit1 ⇒ both linear
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                     # one op per output element
  bytes_moved: "3 * n * dtype_bytes"   # read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 math. Div IEEE inf/NaN; Max/Min NaN-as-missing. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## binary_f16  (Add / Sub / Mul / Div / Max / Min — native f16)

Element-wise binary for `F16`, native storage with f32 intermediate math. Same 6-op `op_id`
selector and per-operand rank-4 stride + `flags` (bit0=a_contig, bit1=b_contig) model as `binary`;
**per-operand strided + broadcast capable, offset-incapable**. Output: f16, broadcasted shape,
contiguous, fresh buffer, no aliasing.

```fkc
kernel: binary_f16
op_kind: AddElementwise       # representative; 6-op op_id selector (same surface as binary)
blurb: "Elementwise binary (f16, native; f32 math, narrow on store); 6-op selector; per-operand strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::binary_f16"   # wrapper lib.rs:1591; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params:
    variant: Binary              # out_size, op_id, rank, flags, shape0..3, a_s0..3, b_s0..3
    fields:
      out_size: { kind: usize }
      op_id:    { kind: u32, note: "0..5 selects Add/Sub/Mul/Div/Max/Min" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = a contiguous, bit1 = b contiguous" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # dtype_bytes = 2 (f16); read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f16 storage, f32 intermediate, narrow on store. Div IEEE inf/NaN; Max/Min NaN-as-missing."

determinism: same_hardware_bitwise
```

## binary_f64  (Add / Sub / Mul / Div / Max / Min — native f64)

Element-wise binary for `F64`, native `double` arithmetic. Same 6-op `op_id` selector and
per-operand rank-4 stride + `flags` model as `binary`; **per-operand strided + broadcast capable,
offset-incapable**. Output: f64, broadcasted shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: binary_f64
op_kind: AddElementwise       # representative; 6-op op_id selector (same surface as binary)
blurb: "Elementwise binary (f64, native double); 6-op selector; per-operand strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::binary_f64"   # wrapper lib.rs:1609; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params:
    variant: Binary              # out_size, op_id, rank, flags, shape0..3, a_s0..3, b_s0..3
    fields:
      out_size: { kind: usize }
      op_id:    { kind: u32, note: "0..5 selects Add/Sub/Mul/Div/Max/Min" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = a contiguous, bit1 = b contiguous" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # dtype_bytes = 8 (f64); read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native double. Div IEEE inf/NaN; Max/Min NaN-as-missing."

determinism: same_hardware_bitwise
```

## binary_bf16  (Add / Sub / Mul / Div / Max / Min — bf16 packed-u32)

Element-wise binary for `BF16`, stored as packed-u16-in-u32, math at f32. Unlike the half *unary*
and *affine* variants, `binary_bf16` (`binary_bf16.slang:70`) **remains strided + broadcast
capable**: its strided path reads single lanes by masking the packed u32, while the contiguous path
reads u32 pairs. It uses the same per-operand rank-4 stride + `flags` model as `binary`. **`out_size`
must be even** (the wrapper pads an odd count). bf16↔f32 is the exact `bits << 16` load, RNE
upper-16 + canonical qNaN store. Output: bf16, broadcasted shape, contiguous, fresh buffer, no
aliasing.

```fkc
kernel: binary_bf16
op_kind: AddElementwise       # representative; 6-op op_id selector (same surface as binary)
blurb: "Elementwise binary (bf16 packed-u32; f32 math, narrow on store); 6-op selector; strided+broadcast (lane-masked); out_size even."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::binary_bf16"   # wrapper lib.rs:8836; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
    - name: rhs
      dtypes: [BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: broadcast_to=out
  op_params:
    variant: Binary              # out_size, op_id, rank, flags, shape0..3, a_s0..3, b_s0..3
    fields:
      out_size: { kind: usize, note: "must be even; wrapper pads an odd count" }
      op_id:    { kind: u32, note: "0..5 selects Add/Sub/Mul/Div/Max/Min" }
      rank:     { kind: u32 }
      flags:    { kind: u32, note: "bit0 = a contiguous, bit1 = b contiguous" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)
      shape_rule: broadcast(lhs, rhs)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # strided path masks single lanes from packed u32
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }   # contiguous path reads u32 pairs
    - { when: "any_input_strided", class: strided_elementwise }     # single-lane masked reads
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"   # dtype_bytes = 2 (bf16); read lhs + rhs, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 packed-u32; f32 math; load bits<<16 exact, store RNE upper-16 + canonical qNaN. Div IEEE inf/NaN; Max/Min NaN-as-missing. Requires even out_size."

determinism: same_hardware_bitwise
```

## affine  (y = x·mul + add — f32)

Element-wise affine `out[i] = mul · in[i] + add` over an f32 buffer; backs `AddScalar` (`mul=1`),
`MulScalar` (`add=0`), and the general `Affine`. The Slang kernel (`affine.slang:22`) carries the
affine Params shape + a `flags` bit0 contiguity flag, so it is **strided + broadcast capable,
offset-incapable** (the same rank-4 coord-decompose path as `unary`). The scalar params `(mul, add)`
arrive on `OpParams::Affine { mul: f64, add: f64 }` and are consumed at f32. Output: f32, input
shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: affine
op_kind: Affine               # OpParams::Affine; covers AddScalar (mul=1) / MulScalar (add=0)
blurb: "Elementwise affine y = mul*x + add (f32); covers AddScalar/MulScalar; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine_f32"   # wrapper build_affine_f32_dispatch lib.rs:3648 / :3711; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Affine              # OpParams::Affine { mul: f64, add: f64 }; + out_size, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize, note: "= n, the output element count" }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      mul:      { kind: f64, note: "consumed at f32 for this dtype" }
      add:      { kind: f64, note: "consumed at f32 for this dtype" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"                # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 mul-then-add. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar. Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## affine_f16  (y = x·mul + add — native f16, f32 math)

Element-wise affine for `F16`, native storage with f32 math (`mul`/`add` taken at f32). Same affine
Params + `flags` bit0 contiguity model as `affine`; **strided + broadcast capable,
offset-incapable**. Each element widens to f32, computes `mul·x + add`, narrows to f16 on store.
Output: f16, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: affine_f16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f16, native; f32 math, narrow on store); strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine_f16"   # wrapper lib.rs:3501; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Affine              # OpParams::Affine { mul: f64, add: f64 }; consumed at f32
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      mul:      { kind: f64, note: "narrowed to f32 for the half compute" }
      add:      { kind: f64, note: "narrowed to f32 for the half compute" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 16

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (f16)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native f16 storage; widen to f32, mul-then-add in f32, narrow to f16 on store. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## affine_f64  (y = x·mul + add — native f64)

Element-wise affine for `F64`, native `double` arithmetic (`mul`/`add` taken directly as f64). Same
affine Params + `flags` bit0 contiguity model as `affine`; **strided + broadcast capable,
offset-incapable**. Output: f64, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: affine_f64
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (f64, native double); strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine_f64"   # wrapper lib.rs:3421; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Affine              # OpParams::Affine { mul: f64, add: f64 }
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      mul:      { kind: f64 }
      add:      { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 64

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 8 (f64)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "native double mul-then-add. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## affine_bf16  (y = x·mul + add — bf16 packed-u32 pair-thread, CONTIGUOUS-ONLY)

Element-wise affine for `BF16`. Unlike the wide affine variants, this is the **contiguous-only**
pair-thread kernel (packed-u32, one thread per u32 = two bf16 lanes). It carries no rank/stride
Params, so any strided / broadcast / offset input is realized by an upstream `Op::Contiguize` first
(`awkward_layout_strategy: requires_contiguous`). `mul`/`add` arrive f64 (`OpParams::Affine`),
narrow to f32; each element widens to f32, computes `mul·x + add`, narrows to bf16 on store. Output:
bf16, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: affine_bf16
op_kind: Affine
blurb: "Elementwise affine y = mul*x + add (bf16 packed-u32 pair-thread; f32 math, narrow on store); CONTIGUOUS-ONLY."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::affine_bf16"   # wrapper lib.rs:3583; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=out
  op_params:
    variant: Affine              # OpParams::Affine { mul: f64, add: f64 }; consumed at f32; + out_size
    fields:
      out_size: { kind: usize, note: "pair-thread; one u32 per thread = 2 bf16 lanes" }
      mul:      { kind: f64, note: "narrowed to f32 for the half compute" }
      add:      { kind: f64, note: "narrowed to f32 for the half compute" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (an FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32     # operates on packed u32 words

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"
  bytes_moved: "2 * n * dtype_bytes"   # dtype_bytes = 2 (bf16)
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "bf16 packed-u32; widen to f32, mul-then-add in f32 (params f64→f32), narrow to bf16 on store. mul=1 ⇒ AddScalar; add=0 ⇒ MulScalar."

determinism: same_hardware_bitwise
```

## clamp  (y = clamp(x, lo, hi) — f32)

Element-wise bounded clamp `out[i] = clamp(in[i], lo, hi)` over an f32 buffer (`clamp.slang:22`).
**f32 only.** Carries the affine Params shape + `flags` bit0 contiguity flag, so it is **strided +
broadcast capable, offset-incapable**. The scalar bounds `(lo, hi)` arrive on
`OpParams::Clamp { min: f64, max: f64 }`, consumed at f32. Output: f32, input shape, contiguous,
fresh buffer, no aliasing.

```fkc
kernel: clamp
op_kind: ClampElementwise
blurb: "Elementwise clamp(x, lo, hi) (f32 only); strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::clamp_f32"   # source clamp.slang:22; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: Clamp               # OpParams::Clamp { min: f64, max: f64 }; + out_size, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      lo:       { kind: f64, note: "min bound; consumed at f32" }
      hi:       { kind: f64, note: "max bound; consumed at f32" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                     # two compares (min/max) per element; ~1 op/element
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 clamp; exact (no rounding). Deterministic per-element dispatch."

determinism: same_hardware_bitwise
```

## powi  (y = x^exp — f32)

Element-wise integer power `out[i] = in[i]^exp` over an f32 buffer (`powi.slang:27`). **f32 only.**
The exponent special-cases `e == 0 / 1 / 2 / 3` (direct multiplies) and otherwise calls `pow`;
`pow(0, -k) → +inf` matches the CPU reference. Carries the affine Params shape + `flags` bit0
contiguity flag, so it is **strided + broadcast capable, offset-incapable**. The exponent arrives on
`OpParams::PowI { exp: i32 }`. Output: f32, input shape, contiguous, fresh buffer, no aliasing.

```fkc
kernel: powi
op_kind: PowIElementwise
blurb: "Elementwise integer power y = x^exp (f32 only); special-cased e=0/1/2/3 else pow; strided+broadcast."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::powi_f32"   # source powi.slang:27; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: input
      dtypes: [F32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 2..=4
      shape_constraint: same_as=out
  op_params:
    variant: PowI                # OpParams::PowI { exp: i32 }; + out_size, flags, shape0..3, in_s0..3
    fields:
      out_size: { kind: usize }
      flags:    { kind: u32, note: "bit0 = input contiguous fast path" }
      exp:      { kind: i32, note: "e=0/1/2/3 special-cased; else pow" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(input)
      shape_rule: same_as(input)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis broadcast; no materialize" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured     # per-element op count scales with exp (special-case vs pow) — Judge measures
  class: cheap_elementwise
  flops: ~                       # e=0/1/2/3: a few muls; else a transcendental pow; not a fixed constant
  bytes_moved: "2 * n * dtype_bytes"   # read input, write out
  overhead_ns: ~
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32. e=0/1/2/3 by direct multiply; else GLSL pow (not bit-stable cross-hardware). pow(0,-k) -> +inf matches CPU."

determinism: same_hardware_bitwise
```

## add_assign_scaled  (in-place dst[i] += src[i]·scale — f32)

In-place scaled accumulate `dst[i] += src[i] · scale` over two equal-length f32 buffers
(`add_assign_scaled.slang:15`). **f32 only.** Binding 0 is the read-write `dst`, binding 1 is `src`;
the kernel mutates `dst` in place — the **output IS the `dst` input buffer**
(`caps.in_place: true`, `aliasing: in_place(dst)`). It is element-aligned 1:1, **contiguous-only**
(no rank/shape/stride Params), so any strided / broadcast / offset operand is contiguized by an
upstream `Op::Contiguize` first. The `scale` arrives as an `f` param. Each element does one
fused-shape `dst + src·scale` at f32; no atomics (one thread per element, distinct outputs), so it
is bit-stable on the same hardware. No new output allocation — the buffer is `dst`.

```fkc
kernel: add_assign_scaled
op_kind: AddAssignScaled
blurb: "In-place scaled accumulate dst[i] += src[i]*scale (f32 only); contiguous; dst is RW (output aliases dst)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::add_assign_scaled"   # source add_assign_scaled.slang:15; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst                  # binding 0: READ-WRITE accumulator; the in-place output buffer
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=src
    - name: src                  # binding 1: read-only addend source
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      shape_constraint: same_as=dst
  op_params:
    variant: AddAssignScaled     # n, scale (f)
    fields:
      n:     { kind: usize, note: "element count; dst and src element-aligned 1:1" }
      scale: { kind: f32 }

return:
  outputs:
    - name: dst
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: contiguous
      aliasing: in_place(dst)    # output IS dst's buffer (caps.in_place: true, §4.6)

caps:
  awkward_layout_strategy: requires_contiguous   # element-aligned 1:1; planner contiguizes a non-contiguous producer
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                 # binding 0 (dst) is RW; output aliases dst (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "2 * n"                 # one multiply + one add per element (dst + src*scale)
  bytes_moved: "3 * n * dtype_bytes"   # read dst + src, write dst (in-place RMW)
  overhead_ns: ~
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true   # one thread per element, distinct outputs; no atomics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32 dst + src*scale; plain per-element accumulate (no atomics); IEEE inf/NaN. Bit-stable on same hardware."

determinism: same_hardware_bitwise
```

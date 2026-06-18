---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                  # maps to BackendId::Metal
  kernel_source: "metal-msl"      # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — elementwise kernel contracts

Elementwise kernels for the Metal backend (crate `metal`, family `elementwise`), from
`metal_src/{unary,binary,affine,fill}.metal` and the `kernels/{unary,binary,affine,fill}.rs`
dispatch wrappers, wired by `fuel-metal-backend` `storage.rs`. This bundle covers the unary map,
binary arithmetic/comparison, affine (`mul·x + add`), `powf` (`pow(x, mul)`), `elu`, the
`const_set` / `copy2d` scatter-copies, and the dense `fill` — i.e. the as-built Metal entries
`unary_kernel(_strided)`, `const_set(_strided)`, `copy2d`, `binary_kernel(_strided)`,
`affine_kernel(_strided)`, `powf_kernel(_strided)`, `elu_kernel(_strided)`, `fill`
(inventory `docs/kernel-contracts/_inventory/metal.md`).

**Cross-cutting Metal facts (apply to every kernel below unless a section says otherwise).**

- **Contiguous/strided pairing.** Every contiguous host-name kernel is paired with a `_strided`
  variant emitted by the same `init_*` macro. The Rust wrappers (`call_*_contiguous` vs
  `call_*_strided`) and `storage.rs` pick the variant on `layout.is_contiguous()`. The contiguous
  variant is dense, length-only addressing (`requires_contiguous`); the strided variant walks the
  generic `get_strided_index` mixed-radix de-linearizer, so it accepts **arbitrary strides
  including broadcast (stride 0) and transposed/overlapping layouts** (`handles_strided`), bounded
  only by the `uint` index range (inventory cross-cutting facts).
- **Offset capability.** Element/byte offsets ride `BufferOffset { buffer, offset_in_bytes }`
  (`utils.rs`), set via `set_buffer(pos, buf, offset)`; the backend computes
  `offset_in_bytes = layout.start_offset() * dtype.size_in_bytes()`. Every `call_*` that takes a
  `BufferOffset` is therefore **non-zero-offset capable** for that operand (`start_offset:
  accepted`).
- **Reverse strides are NOT accepted.** `get_strided_index` is a **non-negative** mixed-radix
  indexer; there is no signed-stride walk anywhere in this family. A reversed view (`Op::Flip`)
  must be normalized to a non-negative copy before any kernel here, so every operand declares
  `reverse_strides: rejected` (§4.1.1).
- **Tiling.** Contiguous kernels tile with `work_per_thread<T>() = ceil(8/sizeof T)`
  (`get_tile_size`: f32→2, f16/bf16→4, u8→8); strided kernels are one-thread-per-element.
- **Output.** Out-of-place kernels allocate a **fresh contiguous** `device.new_buffer`, same dtype
  and logical shape as the input, no aliasing. The exceptions are the scatter/in-place kernels
  (`const_set`, `copy2d`, `fill`) which write **through a caller-supplied dst** (offset/alias
  capable) — flagged per section.
- **Cost provenance.** Every cost block marks `cost.provenance: judge_measured` — the Judge
  bootstraps/refines the measured cost for this GPU surface (§4.4). Where genuinely derivable from
  the op, the block carries the **bandwidth hint** an elementwise kernel admits (`bytes_moved`,
  reads + writes; `flops` = the per-element op count). The launch (command-buffer submit) overhead
  and any tiling-dependent throughput are `judge_measured` — **no cost number is fabricated**.
- **Precision.** Per-op numerics are noted per section (several ops compute internally at f32 and
  narrow on store; `elu` and `binary` compute in `T`). Bounds are author-declared seeds the Judge
  audits (§4.8).

> **Five kernels in this family have NO backing Fuel `OpKind` today (faithfulness flag).** The
> as-built Metal entries `powf_kernel(_strided)`, `elu_kernel(_strided)`, `const_set(_strided)`,
> `copy2d`, and `fill` are **backend primitives with no `OpKind` / `OpParams` in
> `fuel-core-types/src/dispatch.rs` or `fuel-dispatch/src/kernel.rs`** (verified: no `Powf`, `Elu`,
> `ConstSet`, `Copy2d`, or `Fill` variant exists; `OpKind::Copy` is the *cross-device* copy `[T,T]`,
> and `Op::ZeroFill` is a fixed zero-fill, neither of which is the scalar `powf`/`elu`/`fill`/
> `const_set`/`copy2d` semantics). FKC's dispatch key requires a real `OpKind` (§3.3, §10 rule 2),
> and the agreement forbids inventing one. Those five sections are therefore authored as **honest
> documentation with `op_kind: ~`** and an explicit `# UNREGISTRABLE` marker — they parse as
> documentation but are **not registrable** until a Fuel `OpKind` lands (the same "no hidden gap"
> discipline the spec applies to MX / `MxNotYetRegistrable`, §6). Their as-built dtypes, layouts,
> output behavior, and source lines are recorded faithfully so the contract is ready the moment the
> op-kind exists. The six kernels that **do** map to a real `OpKind` (`unary_kernel(_strided)`,
> `binary_kernel(_strided)`, `affine_kernel(_strided)`) are fully registrable.

---

## unary_kernel  (contiguous elementwise unary, dense, tiled)

One-line: Elementwise unary out[i]=op(in[i]) contiguous (dense, tiled); f32/f16/bf16 (+u8/u32/i64 for copy).

Contiguous elementwise unary map `out[i] = op(in[i])` over a dense, zero-stride, row-major buffer
(`metal_src/unary.metal:37-66,168-278`; `kernels/unary.rs:16-68`). One struct per op covers cos,
sin, exp, sqr, sqrt, neg, log, gelu (tanh-approx), gelu_erf, abs, ceil, floor, relu, round, erf,
recip, silu, sign, sigmoid, tanh, and copy (`uid`); the op is selected by the `OpKind` / kernel
name. The kernel addresses linearly by element index (length only) and uses the contiguous
`work_per_thread<T>() = ceil(8/sizeof T)` tiling. Math ops are float-family only
(`init_unary_float`: f32/f16/bf16) — the backend `unary()` wires f32/f16/bf16 for all math ops
(plus `sign` for i64); the `copy` op additionally covers u8/u32/i64. Integer math ops are **not**
generated. Several ops compute internally at f32 (gelu uses `precise::tanh`; `erf` is the A&S
7.1.26 polynomial — **not** exact erf; `abs` casts to float), narrowing on store. Input is
contiguous-only and offset-capable (`BufferOffset`); output is a fresh contiguous buffer, same
dtype and shape. Bandwidth-bound: one element read, one written. This contract registers under one
representative unary `OpKind` per (op, dtype); the binding-table monomorphization expands one key
per op×dtype. Known limitation: contiguous-only — strided/broadcast/offset operands route to
`unary_kernel_strided` or are contiguized first.

```fkc
kernel: unary_kernel
op_kind: ExpElementwise        # representative; the shared chassis backs the whole float-unary set
                               # (Cos/Sin/Exp/Sqr/Sqrt/Neg/Log/Gelu/GeluErf/Abs/Ceil/Floor/Relu/
                               #  Round/Erf/Recip/Silu/Sign/Sigmoid/Tanh + copy) — one OpKind per op
blurb: "Elementwise unary out[i]=op(in[i]) contiguous (dense, tiled); f32/f16/bf16 (+u8/u32/i64 for copy)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "unary_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "Float-family math ops (init_unary_float) = f32/f16/bf16; the `copy` op additionally accepts U8/U32/I64 and `sign` additionally I64. Dense length-only addressing; tiled work_per_thread = ceil(8/sizeof T). Offset-capable (BufferOffset)."
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)        # same dtype as input
      shape_rule: same_as(in)            # same shape; symbolic extents carry through
      layout_guarantee: contiguous       # fresh device.new_buffer, dense
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bytes_moved below is a derivable bandwidth hint
  class: cheap_elementwise
  flops: "n"                            # one op per element (n = output element count)
  bytes_moved: "2 * n * dtype_bytes"    # read in, write out — bandwidth-bound
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # deterministic per-element map
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                        # author-declared seed; Judge audits (§4.8)
  notes: "Per-op numerics vary: gelu uses precise::tanh; erf is the A&S 7.1.26 polynomial (NOT exact erf — error-bearing); abs casts to float. Several ops compute internally at f32 and narrow on store; bf16/f16 narrow on store. Deterministic per element."

determinism: same_hardware_bitwise
```

---

## unary_kernel_strided  (strided elementwise unary, arbitrary/broadcast strides → dense)

One-line: Elementwise unary from arbitrary/broadcast strides via get_strided_index, gather-to-dense.

Strided elementwise unary `out[tid] = op(in[get_strided_index(tid)])`
(`metal_src/unary.metal:37-66,168-278`; `kernels/unary.rs:16-68`). The kernel de-linearizes each
output index through the generic `get_strided_index` mixed-radix indexer, so the input may carry
**arbitrary strides — including broadcast (stride 0) and transposed/overlapping layouts** — bounded
only by the `uint` range. One thread per element (no tiling); it reads from the strided input and
**writes the output densely** (gather-from-strided-input → contiguous output). The backend passes a
`BufferOffset` **dst** as well, so this variant is offset-capable on *both* input and output (this
is the path `copy_strided_src` uses). The op set and per-op numerics are identical to the
contiguous variant; only the input (and optional dst) addressing differs. Negative/reverse strides
are **not** accepted (`get_strided_index` is non-negative; §4.1.1). Bandwidth-bound: reads N
(scattered) src, writes N dst.

```fkc
kernel: unary_kernel_strided
op_kind: ExpElementwise        # representative; same float-unary set + copy as unary_kernel
blurb: "Elementwise unary from arbitrary/broadcast strides via get_strided_index, gather-to-dense."
backend: Metal
kernel_source: "metal-msl"
entry_point: "unary_kernel_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "get_strided_index mixed-radix de-linearizer ⇒ arbitrary/broadcast/transposed strides, one thread per element. Offset-capable on input AND output (BufferOffset dst). `copy` op adds U8/U32/I64. Non-negative strides only."
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous       # writes out[tid] densely
      aliasing: none                     # fresh dense output (also usable with a BufferOffset dst by copy_strided_src)

caps:
  awkward_layout_strategy: handles_strided        # walks strides directly; no contiguize inserted
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; strided gather has worse locality than the contiguous variant
  class: strided_elementwise
  flops: "n"                            # one op per element
  bytes_moved: "2 * n * dtype_bytes"    # read (scattered) in, write dense out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Numerics identical to unary_kernel (gelu precise::tanh; erf A&S 7.1.26 polynomial, not exact; abs via float). Addressing-only difference. Deterministic per element."

determinism: same_hardware_bitwise
```

---

## const_set  (contiguous scalar fill of a dst region)

One-line: Fill a contiguous dst region [0,length) with a scalar value (writes through caller dst).

> **UNREGISTRABLE — no Fuel OpKind.** `const_set` fills an existing buffer region with a scalar
> (`out[i] = value`, `metal_src/unary.metal:68-95`; `kernels/unary.rs:70-120`; backend `const_set`
> `storage.rs:449-541`). There is **no `OpKind::ConstSet` / `OpParams::ConstSet`** in Fuel today
> (`OpKind::Copy` is cross-device copy; `Op::ZeroFill` is a fixed zero-fill, not a scalar fill;
> `OpKind::MaskedFill` is mask-gated, not unconditional). So this section is **documentation only**
> (`op_kind: ~`) until a Fuel scalar-fill op-kind lands. As-built facts: dtypes f32/f16/bf16/u8/u32/
> i64; the scalar arrives **by value**; the contiguous variant fills `[0, length)`; it writes
> **through the caller's dst buffer** — output **aliases caller storage** (mutates it), dtype = dst
> dtype. Offset-capable dst. Bandwidth-bound: write-only (no input read), N elements.

```fkc
kernel: const_set
registrable: false             # §3.10 describe-only: NO OpKind/OpParams backs a scalar buffer-fill today (see note above)
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind backs a scalar buffer-fill today (see note above)
blurb: "Fill a contiguous dst region [0,length) with a scalar value (writes through caller dst)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "const_set"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst                # the in-place fill target (read-modify-write region; scalar is by-value, not an operand)
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      notes: "Contiguous fill of [0,length); scalar value passed by value (NOT a tensor operand). Offset-capable dst (BufferOffset)."
  op_params: { variant: ~ }    # no OpParams variant exists for this op

return:
  outputs:
    - name: dst
      dtype_rule: passthrough(dst)       # dtype = dst dtype
      shape_rule: same_as(dst)
      layout_guarantee: contiguous
      aliasing: in_place(dst)            # mutates the caller's storage in place

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                # writes through the caller's dst buffer (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; write-only bandwidth hint below
  class: cheap_elementwise
  flops: "0"                            # no arithmetic, pure store
  bytes_moved: "n * dtype_bytes"        # write-only (no input read)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true     # exact store of the scalar
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact store of the scalar value into every element; bit-exact. UNREGISTRABLE pending a Fuel scalar-fill OpKind."

determinism: same_hardware_bitwise
```

---

## const_set_strided  (strided scalar scatter into a dst)

One-line: Scatter a scalar into a strided dst via get_strided_index (writes through caller dst).

> **UNREGISTRABLE — no Fuel OpKind** (see `const_set` above). The strided variant writes
> `out[get_strided_index(tid)] = value`, scattering the scalar into a **strided dst**
> (`metal_src/unary.metal:68-95`; `kernels/unary.rs:70-120`). Arbitrary/broadcast strides via the
> generic indexer; offset-capable dst. Output **aliases caller storage**, dtype = dst dtype.
> Negative strides not accepted. Documentation only (`op_kind: ~`).

```fkc
kernel: const_set_strided
registrable: false             # §3.10 describe-only: NO OpKind/OpParams backs a scalar buffer-fill today
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind backs a scalar buffer-fill today
blurb: "Scatter a scalar into a strided dst via get_strided_index (writes through caller dst)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "const_set_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      notes: "Scatters out[get_strided_index(tid)] = value into a strided dst; scalar by value. Offset-capable dst. Non-negative strides only."
  op_params: { variant: ~ }

return:
  outputs:
    - name: dst
      dtype_rule: passthrough(dst)
      shape_rule: same_as(dst)
      layout_guarantee: same_as(dst)     # writes into the dst's own (strided) layout
      aliasing: in_place(dst)

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: true
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "0"
  bytes_moved: "n * dtype_bytes"        # write-only scatter
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact scalar store; bit-exact. UNREGISTRABLE pending a Fuel scalar-fill OpKind."

determinism: same_hardware_bitwise
```

---

## copy2d  (2-D strided block copy between row-strided src and dst)

One-line: 2-D strided block copy out[x*dst_s+y]=in[x*src_s+y] over a d1×d2 grid (one stride pair).

> **UNREGISTRABLE — no Fuel OpKind.** `copy2d` (`copy2d_<t>`) copies a `d1×d2` block with one
> stride pair: `out[x*dst_s + y] = in[x*src_s + y]` (`metal_src/unary.metal:97-111`;
> `kernels/unary.rs:122-173`; backend `storage.rs:1763-1821`). There is **no `OpKind::Copy2d` /
> `CopyStrided`** in Fuel (`OpKind::Copy` is the cross-device `[T,T]` copy with different
> semantics). It is a same-device backend primitive used by `copy2d()` (which falls back to a blit
> when `src_s == d2 == dst_s`) and by mlx multi-block sort. As-built facts: dtypes
> f32/f16/bf16/u8/u32/i64; takes explicit `(d1, d2, src_s, dst_s)` + byte offsets on **both** src
> and dst; supports a row-strided 2-D source and dest; it is **not** a general N-D strider (exactly
> one stride pair). Writes into the caller's dst at `dst_o_in_bytes` (**offset/alias capable**),
> dtype = src dtype. Bandwidth-bound: read N + write N (`N = d1*d2`). Documentation only
> (`op_kind: ~`).

```fkc
kernel: copy2d
registrable: false             # §3.10 describe-only: NO OpKind/OpParams backs a same-device 2-D strided block copy today
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind backs a same-device 2-D strided block copy today
blurb: "2-D strided block copy out[x*dst_s+y]=in[x*src_s+y] over a d1×d2 grid (one stride pair)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "copy2d"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 2
      notes: "Explicit (d1,d2,src_s,dst_s) + byte offset; exactly ONE stride pair (row-strided 2-D), NOT a general N-D strider. Offset-capable. Non-negative strides."
  op_params: { variant: ~ }    # params (d1,d2,src_s,dst_s,offsets) carried by the call wrapper; no OpParams variant

return:
  outputs:
    - name: dst                # the caller-supplied dst (written at dst_o_in_bytes)
      dtype_rule: passthrough(src)       # dtype = src dtype
      shape_rule: same_as(src)           # d1×d2 block
      layout_guarantee: same_as(dst)     # writes into dst's row-strided layout at its byte offset
      aliasing: in_place(dst)            # writes through caller dst (offset/alias capable)

caps:
  awkward_layout_strategy: handles_strided        # one row-stride pair walked directly (blit fast path when src_s==d2==dst_s)
  fast_paths:
    - { when: "all_inputs_contiguous", note: "src_s==d2==dst_s ⇒ falls back to a blit" }
    - { when: "any_input_strided", class: strided_elementwise }
  in_place: true                # writes through the caller's dst at a byte offset
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bandwidth hint below (N = d1*d2)
  class: strided_elementwise
  flops: "0"                            # pure copy
  bytes_moved: "2 * d1 * d2 * dtype_bytes"   # read N + write N, N = d1*d2
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # writes a caller dst (no fresh alloc here)

precision:
  bit_stable_on_same_hardware: true     # byte copy / exact element copy
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact element copy (no arithmetic); bit-exact. UNREGISTRABLE pending a Fuel same-device strided-copy OpKind."

determinism: same_hardware_bitwise
```

---

## binary_kernel  (contiguous elementwise binary arithmetic/comparison)

One-line: Elementwise binary out[i]=op(l[i],r[i]) contiguous; arithmetic→T, comparison→U8; f32/f16/bf16/u8/u32/i64.

Contiguous elementwise binary `out[i] = op(l[i], r[i])` over two dense, zero-stride, row-major
buffers (`metal_src/binary.metal:61-196`; `kernels/binary.rs`; backend `binary()`
`storage.rs:1889-1957`, `cmp()` `storage.rs:437-447`). Two op groups share the kernel: arithmetic
`badd/bsub/bmul/bdiv/bminimum/bmaximum` (return `T`) and comparisons `eq/ne/le/lt/ge/gt`
(`init_boolean_binary`, return **bool → u8**). Both groups cover f32/f16/bf16/u8/u32/i64. Plain
operators; min/max use the `MIN`/`MAX` macro (NaN-naive — picks by `>`/`<`). Both inputs are
contiguous-only and offset-capable; output is a fresh contiguous buffer driven by the lhs
shape/dims. Arithmetic output dtype = input dtype; comparison output dtype = **U8** (the backend
sets the out dtype to U8 for the boolean ops). One representative `OpKind` per op; the binding-table
key carries the output dtype slot so the comparison key is `[T, T, U8]`. Bandwidth-bound: two reads
+ one write. Known limitation: contiguous-only — strided operands route to the `_strided` family
(below) or are contiguized first.

```fkc
kernel: binary_kernel
op_kind: AddElementwise        # representative; the shared chassis backs arithmetic
                               # (Add/Sub/Mul/Div/Maximum/Minimum → T) and comparisons
                               # (Equal/NotEqual/Less/LessEqual/Greater/GreaterEqual → U8) — one OpKind per op
blurb: "Elementwise binary out[i]=op(l[i],r[i]) contiguous; arithmetic→T, comparison→U8; f32/f16/bf16/u8/u32/i64."
backend: Metal
kernel_source: "metal-msl"
entry_point: "binary_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=rhs"
      notes: "Dense, length-only; offset-capable (BufferOffset). Output shape driven by lhs dims."
    - name: rhs
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=lhs"
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)       # arithmetic ops; comparison ops are fixed(U8) — keyed [T,T,U8] (§5.1)
      shape_rule: same_as(lhs)
      layout_guarantee: contiguous       # fresh device.new_buffer
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bytes_moved is a derivable bandwidth hint
  class: cheap_elementwise
  flops: "n"                            # one scalar op per element
  bytes_moved: "3 * n * dtype_bytes"    # read lhs + rhs, write out — bandwidth-bound
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # deterministic per-element op
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Plain operators; min/max via MIN/MAX macro (NaN-naive: picks by >/<). Arithmetic in the dtype's native op; comparisons return U8 (1/0). Deterministic per element."

determinism: same_hardware_bitwise
```

---

## binary_kernel_strided  (strided/broadcast binary; _strided / _lstrided / _rstrided variants)

One-line: Elementwise binary over arbitrary/broadcast strides via get_strided_index (both/left/right strided).

Strided elementwise binary (`metal_src/binary.metal:61-196`; `kernels/binary.rs`; backend
`storage.rs:1889-1957`). The strided family has **four monomorphizations per (op, dtype)** the
backend selects between: `_strided` (both operands strided), `_lstrided` (left strided, right
contiguous), and `_rstrided` (left contiguous, right strided). All use the generic
`get_strided_index`, so each strided operand accepts **arbitrary strides including broadcast
(stride 0) and transposed/overlapping layouts**. Both inputs are offset-capable. The output is a
fresh contiguous buffer (lhs-shape-driven); arithmetic → `T`, comparison → **U8**. Op set and
numerics are identical to the contiguous variant. Negative/reverse strides are **not** accepted
(`get_strided_index` is non-negative; §4.1.1). This one contract represents the `_strided` /
`_lstrided` / `_rstrided` indexer-variant set (a single capability advertisement; the backend picks
the concrete monomorphization from which operands are strided). Bandwidth-bound: two reads + one
write.

```fkc
kernel: binary_kernel_strided
op_kind: AddElementwise        # representative; same arithmetic (→T) + comparison (→U8) set as binary_kernel
blurb: "Elementwise binary over arbitrary/broadcast strides via get_strided_index (both/left/right strided)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "binary_kernel_strided"   # represents _strided / _lstrided / _rstrided per (op,dtype)
kernel_revision_hash: auto

accept:
  inputs:
    - name: lhs
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=rhs"
      notes: "get_strided_index ⇒ arbitrary/broadcast/transposed strides. The backend picks _lstrided when only lhs is strided. Offset-capable. Non-negative strides only."
    - name: rhs
      dtypes: [F32, F16, BF16, U8, U32, I64]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=lhs"
      notes: "Backend picks _rstrided when only rhs is strided, _strided when both are."
  op_params: { variant: None }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(lhs)       # arithmetic; comparison ops fixed(U8), keyed [T,T,U8]
      shape_rule: same_as(lhs)           # output driven by lhs shape/dims
      layout_guarantee: contiguous       # fresh dense output
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided        # walks strides directly; no contiguize inserted
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; strided gather worse locality than contiguous variant
  class: strided_elementwise
  flops: "n"
  bytes_moved: "3 * n * dtype_bytes"    # read lhs + rhs, write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Numerics identical to binary_kernel (plain operators; min/max NaN-naive via MIN/MAX macro; comparisons → U8). Addressing-only difference. Deterministic per element."

determinism: same_hardware_bitwise
```

---

## affine_kernel  (contiguous fused multiply-add, y = x*mul + add)

One-line: Elementwise affine out=fma(x,mul,add) contiguous; u8/u32/i64/f32/f16/bf16; f32-internal accumulate.

Contiguous fused multiply-add `out = fma(float(in), mul, add)` over a dense, zero-stride,
row-major buffer (`metal_src/affine.metal:36-68`; `kernels/affine.rs:6-75`; backend `affine()`
`storage.rs:122-183`). The `mul`/`add` scalars ride `OpParams::Affine { mul: f64, add: f64 }`
(§3.7); this one op covers `Op::AddScalar(c)` (`mul=1, add=c`) and `Op::MulScalar(c)` (`mul=c,
add=0`). Dtypes are the full Metal six — u8/u32/i64/f32/f16/bf16 (the backend wires all six;
integers via both contiguous and strided). **Accumulation is in f32 then cast back**, so f16/bf16
affine is f32-internal. Input is contiguous-only and offset-capable; output is a fresh contiguous
buffer, same dtype and shape. Bandwidth-bound: one read + one write, one mul + one add per element.
Known limitation: contiguous-only — strided operands route to `affine_kernel_strided` or are
contiguized first.

```fkc
kernel: affine_kernel
op_kind: Affine                # OpParams::Affine { mul: f64, add: f64 }; covers AddScalar/MulScalar
blurb: "Elementwise affine out=fma(x,mul,add) contiguous; u8/u32/i64/f32/f16/bf16; f32-internal accumulate."
backend: Metal
kernel_source: "metal-msl"
entry_point: "affine_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [U8, U32, I64, F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "Dense, length-only, tiled. Offset-capable (BufferOffset). mul/add by value via OpParams::Affine."
  op_params:
    variant: Affine            # OpParams::Affine { mul: f64, add: f64 } (fuel-dispatch/src/kernel.rs:356)
    fields:
      mul: { kind: f64, note: "consumed at f32 internally (fma accumulate in f32)" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)        # same dtype as input
      shape_rule: same_as(in)
      layout_guarantee: contiguous       # fresh device.new_buffer
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bytes_moved derivable bandwidth hint
  class: cheap_elementwise
  flops: "2 * n"                        # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"    # read in, write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # deterministic per-element fma
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Accumulates in f32 then casts back ⇒ f16/bf16 affine is f32-internal (single rounding on store). IEEE inf/NaN. mul==1 ⇒ AddScalar; add==0 ⇒ MulScalar. Deterministic per element."

determinism: same_hardware_bitwise
```

---

## affine_kernel_strided  (strided fused multiply-add → dense)

One-line: Elementwise affine over arbitrary/broadcast strides via get_strided_index, gather-to-dense.

Strided affine `out[tid] = fma(float(in[get_strided_index(tid)]), mul, add)`
(`metal_src/affine.metal:36-68`; `kernels/affine.rs:6-75`). De-linearizes each output index via the
generic indexer, so the input may carry **arbitrary strides including broadcast (stride 0) and
transposed/overlapping layouts**; one thread per element; reads strided, **writes dense**.
Offset-capable input. Same `OpParams::Affine` scalars and the same f32-internal accumulation as the
contiguous variant. Negative/reverse strides are **not** accepted (§4.1.1). Bandwidth-bound: one
(scattered) read + one dense write.

```fkc
kernel: affine_kernel_strided
op_kind: Affine
blurb: "Elementwise affine over arbitrary/broadcast strides via get_strided_index, gather-to-dense."
backend: Metal
kernel_source: "metal-msl"
entry_point: "affine_kernel_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [U8, U32, I64, F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "get_strided_index ⇒ arbitrary/broadcast/transposed strides, one thread per element; reads strided, writes out[tid] dense. Offset-capable. Non-negative strides only."
  op_params:
    variant: Affine
    fields:
      mul: { kind: f64, note: "f32-internal accumulate" }
      add: { kind: f64 }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous       # writes out[tid] densely
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "2 * n"                        # one multiply + one add per element
  bytes_moved: "2 * n * dtype_bytes"    # read (scattered) in, write dense out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32-internal accumulate then cast (f16/bf16 single rounding on store). Numerics identical to affine_kernel; addressing-only difference. Deterministic per element."

determinism: same_hardware_bitwise
```

---

## powf_kernel  (contiguous scalar float power, y = pow(x, mul))

One-line: Elementwise scalar float power out=pow(float(x),mul) contiguous; f32/f16/bf16; f32-internal.

> **UNREGISTRABLE — no Fuel OpKind.** `powf_kernel` computes `out = pow(float(in), mul)` with a
> scalar f64 exponent (`metal_src/affine.metal:70-99`; `kernels/affine.rs:77-135`; backend
> `powf()` `storage.rs:185-238`). There is **no `OpKind::Powf` / `OpParams::Powf`** in Fuel
> (`PowElementwise` is the *tensor-tensor* power `pow(a[i], b[i])`, not a scalar exponent;
> `PowIElementwise` is an *integer* exponent). So this section is **documentation only**
> (`op_kind: ~`) until a scalar-float-power op-kind lands. As-built facts: dtypes f32/f16/bf16
> only; the exponent `mul` arrives by value; computes `pow` in **f32 then casts back**
> (f16/bf16 = f32-internal); contiguous-only + offset-capable input; fresh contiguous output, same
> dtype/shape. Bandwidth-bound (one read + one write; one transcendental `pow` per element).

```fkc
kernel: powf_kernel
registrable: false             # §3.10 describe-only: NO OpKind/OpParams for a scalar float power (PowElementwise is tensor-tensor; PowIElementwise is integer)
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind for a scalar float power (PowElementwise is tensor-tensor; PowIElementwise is integer)
blurb: "Elementwise scalar float power out=pow(float(x),mul) contiguous; f32/f16/bf16; f32-internal."
backend: Metal
kernel_source: "metal-msl"
entry_point: "powf_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "Dense, length-only, tiled. Offset-capable (BufferOffset). Exponent `mul` (f64) passed by value, NOT a tensor operand."
  op_params: { variant: ~ }    # no OpParams variant for a scalar-float-power op exists

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bandwidth hint below
  class: cheap_elementwise
  flops: "n"                            # one transcendental pow per element
  bytes_moved: "2 * n * dtype_bytes"    # read in, write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "pow computed in f32 then cast back ⇒ f16/bf16 is f32-internal. Transcendental error of the MSL pow. IEEE NaN (e.g. pow(-2,0.5)=NaN). UNREGISTRABLE pending a Fuel scalar-float-power OpKind."

determinism: same_hardware_bitwise
```

---

## powf_kernel_strided  (strided scalar float power → dense)

One-line: Elementwise scalar float power over arbitrary/broadcast strides via get_strided_index, gather-to-dense.

> **UNREGISTRABLE — no Fuel OpKind** (see `powf_kernel` above). Strided variant
> `out[tid] = pow(float(in[get_strided_index(tid)]), mul)` (`metal_src/affine.metal:70-99`;
> `kernels/affine.rs:77-135`). Arbitrary/broadcast strides via the generic indexer; offset-capable
> input; reads strided, writes dense; f32-internal. Negative strides not accepted. Documentation
> only (`op_kind: ~`).

```fkc
kernel: powf_kernel_strided
registrable: false             # §3.10 describe-only: NO OpKind/OpParams for a scalar float power
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind for a scalar float power
blurb: "Elementwise scalar float power over arbitrary/broadcast strides via get_strided_index, gather-to-dense."
backend: Metal
kernel_source: "metal-msl"
entry_point: "powf_kernel_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "get_strided_index ⇒ arbitrary/broadcast/transposed strides; reads strided, writes dense. Offset-capable. Exponent mul by value. Non-negative strides only."
  op_params: { variant: ~ }

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
    - { when: "any_input_broadcast", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "n"                            # one transcendental pow per element
  bytes_moved: "2 * n * dtype_bytes"    # read (scattered) in, write dense out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "f32-internal pow then cast; numerics identical to powf_kernel; addressing-only difference. UNREGISTRABLE pending a Fuel scalar-float-power OpKind."

determinism: same_hardware_bitwise
```

---

## elu_kernel  (contiguous ELU, y = x>0 ? x : mul*(exp(x)-1))

One-line: Elementwise ELU out=(x>0?x:mul*(exp(x)-1)) contiguous; f32/f16/bf16; computed in T.

> **UNREGISTRABLE — no Fuel OpKind.** `elu_kernel` computes ELU `out = x > 0 ? x : mul*(exp(x)-1)`
> with a scalar `mul` (alpha) (`metal_src/affine.metal:101-132`; `kernels/affine.rs:137-195`;
> backend `elu()` `storage.rs:240-291`). There is **no `OpKind::Elu` / `OpParams::Elu`** in Fuel.
> So this section is **documentation only** (`op_kind: ~`) until an ELU op-kind lands. As-built
> facts: dtypes f32/f16/bf16; `mul`/alpha by value; the `exp` is computed **in `T`** (NOT promoted
> to f32 in the strided/contiguous body — distinct from affine/powf which accumulate in f32);
> contiguous-only + offset-capable input; fresh contiguous output, same dtype/shape.
> Bandwidth-bound (one read + one write; one `exp` per negative element).

```fkc
kernel: elu_kernel
registrable: false             # §3.10 describe-only: NO OpKind/OpParams for ELU
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind / OpParams for ELU
blurb: "Elementwise ELU out=(x>0?x:mul*(exp(x)-1)) contiguous; f32/f16/bf16; computed in T."
backend: Metal
kernel_source: "metal-msl"
entry_point: "elu_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "Dense, length-only, tiled. Offset-capable (BufferOffset). alpha `mul` (f64) by value, NOT a tensor operand."
  op_params: { variant: ~ }    # no OpParams variant for ELU exists

return:
  outputs:
    - name: out
      dtype_rule: passthrough(in)
      shape_rule: same_as(in)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; bandwidth hint below
  class: cheap_elementwise
  flops: "n"                            # one exp per negative element + a compare/select per element
  bytes_moved: "2 * n * dtype_bytes"    # read in, write out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "exp computed in T (NOT promoted to f32) — so f16/bf16 ELU is lower-precision than the f32-internal affine/powf. Positive branch is the identity. UNREGISTRABLE pending a Fuel ELU OpKind."

determinism: same_hardware_bitwise
```

---

## elu_kernel_strided  (strided ELU → dense)

One-line: Elementwise ELU over arbitrary/broadcast strides via get_strided_index, gather-to-dense.

> **UNREGISTRABLE — no Fuel OpKind** (see `elu_kernel` above). Strided variant
> `out[tid] = elu(in[get_strided_index(tid)], mul)` (`metal_src/affine.metal:101-132`;
> `kernels/affine.rs:137-195`). Arbitrary/broadcast strides via the generic indexer; offset-capable
> input; reads strided, writes dense; `exp` in `T`. Negative strides not accepted. Documentation
> only (`op_kind: ~`).

```fkc
kernel: elu_kernel_strided
registrable: false             # §3.10 describe-only: NO OpKind/OpParams for ELU
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind / OpParams for ELU
blurb: "Elementwise ELU over arbitrary/broadcast strides via get_strided_index, gather-to-dense."
backend: Metal
kernel_source: "metal-msl"
entry_point: "elu_kernel_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: in
      dtypes: [F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "get_strided_index ⇒ arbitrary/broadcast/transposed strides; reads strided, writes dense. Offset-capable. alpha mul by value. Non-negative strides only."
  op_params: { variant: ~ }

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
    - { when: "any_input_broadcast", class: strided_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: strided_elementwise
  flops: "n"
  bytes_moved: "2 * n * dtype_bytes"    # read (scattered) in, write dense out
  memory: { device_bytes: "n * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "exp in T (not f32-promoted); numerics identical to elu_kernel; addressing-only difference. UNREGISTRABLE pending a Fuel ELU OpKind."

determinism: same_hardware_bitwise
```

---

## fill  (dense scalar fill of a buffer, no offset, no strided variant)

One-line: Dense scalar fill out[tid]=value over [0,numel); no offset, no strided variant; in-place on caller buffer.

> **UNREGISTRABLE — no Fuel OpKind.** `fill` (`fill_<t>`) writes `out[tid] = value` for
> `tid < numel` (`metal_src/fill.metal`; `kernels/fill.rs`, `call_const_fill`). There is **no
> `OpKind::Fill`** in Fuel; `Op::ZeroFill` is a *fixed zero* fill (not a scalar fill), and
> `OpKind::Copy` / `MaskedFill` are different semantics. So this section is **documentation only**
> (`op_kind: ~`) until a scalar-fill op-kind lands. As-built facts: dtypes u8/u32/i64/f16/f32/bf16;
> **dense output only — no offset, no strided variant** (writes `[0, numel)` of the buffer);
> **distinct from `const_set`** (which has both offset and a strided variant). Writes into the
> supplied buffer, dtype = `T`, **in-place / aliasing on the caller buffer**. Bandwidth-bound:
> write-only, N elements.

```fkc
kernel: fill
registrable: false             # §3.10 describe-only: NO OpKind/OpParams for a scalar buffer-fill (Op::ZeroFill is fixed-zero only)
op_kind: ~                     # UNREGISTRABLE: no Fuel OpKind for a scalar buffer-fill (Op::ZeroFill is fixed-zero only)
blurb: "Dense scalar fill out[tid]=value over [0,numel); no offset, no strided variant; in-place on caller buffer."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fill"
kernel_revision_hash: auto

accept:
  inputs:
    - name: dst                # the in-place fill target; scalar value is by-value, not an operand
      dtypes: [U8, U32, I64, F16, F32, BF16]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: any
      notes: "Dense [0,numel) only — NO offset and NO strided variant (distinct from const_set). Scalar value passed by value."
  op_params: { variant: ~ }    # no OpParams variant exists

return:
  outputs:
    - name: dst
      dtype_rule: passthrough(dst)       # dtype = T
      shape_rule: same_as(dst)
      layout_guarantee: contiguous       # dense [0,numel)
      aliasing: in_place(dst)            # writes into the supplied buffer

caps:
  awkward_layout_strategy: requires_contiguous   # dense-only; no offset, no strided path
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: true                # mutates the caller's buffer (§4.6)
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured            # Judge bootstraps; write-only bandwidth hint below
  class: cheap_elementwise
  flops: "0"                            # no arithmetic, pure store
  bytes_moved: "n * dtype_bytes"        # write-only (no input read)
  memory: { device_bytes: 0, host_bytes: 0, disk_bytes: 0 }   # in-place: no new alloc

precision:
  bit_stable_on_same_hardware: true     # exact store of the scalar
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "Exact store of the scalar value into every element of [0,numel); bit-exact. Dense-only, no offset/strided (unlike const_set). UNREGISTRABLE pending a Fuel scalar-fill OpKind."

determinism: same_hardware_bitwise
```

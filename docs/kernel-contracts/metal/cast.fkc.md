---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                  # maps to BackendId::Metal
  kernel_source: "metal-msl"      # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"    # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — cast (dtype-conversion) kernel contracts

Dtype-conversion kernels for the Metal backend (crate `metal`, family `cast`), from
`metal_src/cast.metal:36-104` and the `kernels/cast.rs` dispatch wrappers, wired by
`fuel-metal-backend` `to_dtype()` (`storage.rs:543-663`). Both kernels implement `OpKind::Cast`
(`fuel-core-types/src/dispatch.rs:117`). The `OpParams::Cast` variant is a unit marker
(`fuel-dispatch/src/kernel.rs:352`) — the **target** dtype is not a param; it lives on the output
Storage's `dtype` field, so the output dtype rule is `cast(output)` (§5.1). The conversion is a
straight `out = U(IR(in))` cast chain where `IR` (the intermediate) defaults to `T` and is **unused
by the dispatch** (always `= T`), so there is no special int-truncation intermediate — it is a plain
`static_cast` with no rounding control.

Two kernels back the whole family (the contiguous/strided split is the cross-cutting Metal pattern —
the same `init_*` macro emits a paired `_strided` variant, and the Rust wrappers + `storage.rs`
choose by `layout.is_contiguous()`):

- **`cast_kernel`** — the contiguous variant. Dense, length-only addressing with
  `work_per_thread<T>()` tiling.
- **`cast_kernel_strided`** — the strided variant. Uses the generic `get_strided_index` mixed-radix
  de-linearizer, so it accepts **arbitrary strides including broadcast (stride 0) and
  transposed/overlapping layouts**, one thread per element; it reads from the strided input and
  writes a **dense** contiguous output (gather-to-dense).

**Dtypes — Metal is the {f32, f16, bf16, i64, u32, u8} family only.** `init_cast_all` monomorphizes
**all ordered (src → dst) pairs** over `{f32, f16, bf16, i64, u32, u8}` (6 src × the cross
destinations). There is **no F64 and no F8E4M3** on the Metal cast path (unlike the CPU/Vulkan cast
families). Each (src, dst) pair is a distinct dispatch key via the ordered per-operand dtype slots
(§12.1): `(OpKind::Cast, [SRC, DST], Metal) + kernel_source`. The two contracts below carry the full
accepted src/dst dtype sets; the binding-table monomorphization expands one key per ordered pair.

> **Contiguous-vs-strided wiring asymmetry (faithful to the inventory).** Backend `to_dtype` wires
> the **full strided set**, but the **contiguous set MINUS same-dtype and MINUS the
> bf16-as-source contiguous entries**, which route through the strided variant instead (see the
> `to_dtype` name tables, `storage.rs:543-663`). Net effect: every cross cast is reachable, but a
> bf16-source cast always dispatches through `cast_kernel_strided` even when the bf16 input is
> contiguous, and an identity (same-dtype) cast is not a `cast_kernel` entry at all. This is a
> backend routing fact, not a kernel-capability fact: the `cast_kernel` MSL kernel is dtype-generic;
> the asymmetry is in which monomorphizations the wrapper registers.

**Universal facts for both kernels.** Output is always a **freshly-allocated contiguous** buffer
(`device.new_buffer`), target dtype `U`, **same logical shape** as the input (element count
preserved; only the byte width changes), no aliasing, not in-place. Both are offset-capable on the
input via `BufferOffset { buffer, offset_in_bytes }` (`utils.rs`; the backend sets
`offset_in_bytes = layout.start_offset() * dtype.size_in_bytes()`). Every cast is a bandwidth-bound
elementwise op: it reads N source elements and writes N destination elements, so `bytes_moved` is
genuinely derivable (`n*(src_bytes + dst_bytes)`) and `flops` is `0` (a pure copy/convert). The
precise frontier number, launch overhead, and any tiling-dependent throughput are `judge_measured` —
the Judge bootstraps cost (§4.4); no cost number is fabricated here.

---

## cast_kernel  (contiguous dtype conversion, dense, tiled)

One-line: Cast dtype contiguous (dense, length-only), tiled `static_cast` over {f32,f16,bf16,i64,u32,u8}.

Contiguous dtype conversion `out[i] = U(in[i])` over a dense, zero-stride, row-major input
(`metal_src/cast.metal:36-104`, `kernels/cast.rs`). The kernel addresses linearly by element index
(length only — no strides consulted) and uses the contiguous `work_per_thread<T>() = ceil(8/sizeof T)`
tiling (`get_tile_size`: f32→2, f16/bf16→4, u8→8; `i64`/`u32` per their widths). The conversion is a
plain `static_cast` chain with the intermediate `IR` fixed to `T` (no truncation intermediate, no
rounding control), so narrowing directions (e.g. `f32→f16`, `i64→u32`) follow the hardware/MSL
`static_cast` semantics for that pair and widening directions (e.g. `f16→f32`, `u8→u32`) are exact.
Input is contiguous-only and offset-capable (`BufferOffset`); the output is a fresh contiguous `U`
buffer of the same logical shape. Bandwidth-bound: reads N src elements, writes N dst elements, no
arithmetic beyond the per-element cast. The dispatch key is `(Cast, [SRC, DST], Metal)`; one
contract here covers every contiguous ordered pair the wrapper registers (note: **bf16-source pairs
route through `cast_kernel_strided`** and same-dtype identity is not a `cast_kernel` entry — see the
file header routing note). Known limitation: contiguous-only — any strided/broadcast/non-zero-offset
operand must be contiguized by the planner first (or routed to `cast_kernel_strided`).

```fkc
kernel: cast_kernel
op_kind: Cast
blurb: "Cast dtype contiguous (dense, length-only), tiled static_cast over {f32,f16,bf16,i64,u32,u8}."
backend: Metal
kernel_source: "metal-msl"
entry_point: "cast_kernel"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, I64, U32, U8]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "dense, length-only addressing; tiled work_per_thread = ceil(8/sizeof T). Offset-capable (BufferOffset). bf16-source pairs are wired through cast_kernel_strided, not here; same-dtype identity is not registered."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: cast(output)          # target dtype U = output Storage dtype, key-pinned (§5.1)
      shape_rule: same_as(src)          # element count preserved; only byte width changes
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous   # planner inserts Op::Contiguize (its own FKC kernel) + sums its cost
  fast_paths:
    - { when: "all_inputs_contiguous", class: cheap_elementwise }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured            # Judge bootstraps; bytes_moved below is a bandwidth hint
  class: cheap_elementwise
  flops: "0"                            # pure copy/convert; no arithmetic
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read N src + write N dst; elementwise = bandwidth-bound
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # deterministic per-element static_cast
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                        # author-declared seed; Judge audits (§4.8)
  notes: "Plain static_cast chain (IR fixed to T, no truncation intermediate, no rounding control). Widening pairs exact; narrowing pairs per MSL static_cast for that (src,dst). Deterministic per element."

determinism: same_hardware_bitwise
```

---

## cast_kernel_strided  (strided dtype conversion, arbitrary/broadcast strides → dense)

One-line: Cast dtype from arbitrary/broadcast strides via get_strided_index, gather-to-dense, static_cast.

Strided dtype conversion `out[tid] = U(in[get_strided_index(tid)])` (`metal_src/cast.metal:36-104`,
`kernels/cast.rs`). The kernel de-linearizes each output index through the generic
`get_strided_index` mixed-radix indexer (shared with unary/binary/affine/ternary/indexing), so the
input may carry **arbitrary strides — including broadcast (stride 0) and transposed/overlapping
layouts** — bounded only by the `uint` index range. It runs one thread per element (no tiling) and
**writes the output densely** (gather-from-strided-input → contiguous output), which is exactly the
path `copy_strided_src` and `to_dtype`'s strided branch use. Input is offset-capable
(`BufferOffset`); the output is a fresh contiguous `U` buffer of the same logical shape. The
conversion itself is the same plain `static_cast` chain as the contiguous variant (`IR = T`, no
rounding control) — only the input addressing differs. This is the variant the backend selects when
`layout.is_contiguous()` is false, **and** the variant every bf16-source pair routes through (file
header routing note). Negative/reverse strides are **not** accepted: `get_strided_index` is a
non-negative mixed-radix de-linearizer, so a reversed view (`Op::Flip`) must be normalized to a
non-negative copy before this kernel (§4.1.1). Bandwidth-bound: reads N src elements (scattered),
writes N dst elements (dense). Dispatch key `(Cast, [SRC, DST], Metal)`; covers every ordered pair
the wrapper registers for the strided set (the full set).

```fkc
kernel: cast_kernel_strided
op_kind: Cast
blurb: "Cast dtype from arbitrary/broadcast strides via get_strided_index, gather-to-dense, static_cast."
backend: Metal
kernel_source: "metal-msl"
entry_point: "cast_kernel_strided"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, I64, U32, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: any
      shape_constraint: "same_as=out"
      notes: "get_strided_index mixed-radix de-linearizer: arbitrary/broadcast(stride-0)/transposed/overlapping strides, one thread per element, offset-capable (BufferOffset). Non-negative strides only (no reverse/Flip walk). Output written dense."
  op_params: { variant: Cast }

return:
  outputs:
    - name: out
      dtype_rule: cast(output)          # target dtype U = output Storage dtype, key-pinned (§5.1)
      shape_rule: same_as(src)          # element count preserved; only byte width changes
      layout_guarantee: contiguous      # gather-from-strided-input -> dense contiguous output
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # walks strides directly; no contiguize fixup needed
  fast_paths:
    - { when: "any_input_strided", class: strided_elementwise }
    - { when: "any_input_broadcast", note: "stride-0 axis read repeatedly; still one thread per output element" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8

cost:
  provenance: judge_measured            # Judge bootstraps; bytes_moved below is a bandwidth hint
  class: strided_elementwise
  flops: "0"                            # pure copy/convert; no arithmetic
  bytes_moved: "n * (src_bytes + dst_bytes)"   # read N src (scattered) + write N dst (dense); bandwidth-bound
  memory: { device_bytes: "n * dst_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true     # deterministic per-element static_cast; addressing does not change numerics
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                        # author-declared seed; Judge audits (§4.8)
  notes: "Same plain static_cast chain as cast_kernel (IR fixed to T, no rounding control); only input addressing differs (get_strided_index). Widening pairs exact; narrowing per MSL static_cast. Deterministic per element."

determinism: same_hardware_bitwise
```

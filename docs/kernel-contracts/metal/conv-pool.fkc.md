---
fkc_version: 1
provider:
  name: fuel-metal-kernels
  backend: Metal                     # maps to BackendId::Metal
  kernel_source: "metal-msl"         # the BindingEntry.kernel_source tag
  link_registry: fuel_metal_backend::fkc::ENTRY_POINTS   # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"      # provider build id, folded into kernel_revision_hash
---

# fuel-metal-kernels — convolution, pooling & upsample kernel contracts

Metal (MSL) contracts for the conv / pooling / upsample family of `fuel-metal-kernels`
(`metal_src/conv.metal` + `kernels/convolution.rs` dispatch wrappers, wired by
`fuel-metal-backend/src/storage.rs`), family `conv-attn` in `_inventory/metal.md`. Nine kernels:
the two im2col gathers (`im2col`, `im2col1d`), the col2im scatter (`col2im1d`), the two upsamplers
(`upsample_nearest2d`, `upsample_bilinear2d`), the two poolers (`max_pool2d`, `avg_pool2d`), and the
two naive transposed convolutions (`conv_transpose1d`, `conv_transpose2d`). Each is monomorphized
over the dtype list its `init_*` macro emits.

**Family-wide facts (each section overrides where its inventory entry differs):**

- **These kernels read a STRIDED source — this family is NOT contiguous-only.** Every kernel except
  `col2im1d` takes `src_dims[]` + `src_strides[]` (and the transposes additionally take
  `k_dims[]`/`k_strides[]`) and indexes the source through them, so the source may be
  non-contiguous, transposed, or broadcast (a stride-0 axis): `strided: accepted`,
  `broadcast_stride0: accepted`, `awkward_layout_strategy: handles_strided`. The planner passes a
  strided/transposed producer straight through — **no** inserted `Op::Contiguize`, no contiguize
  cost. (`col2im1d` is the lone exception: it computes its internal strides from the shape and does
  **not** read a `src_strides[]` array, so it is `requires_contiguous`.)
- **Offset capability is per-kernel and split.** Element/byte offsets ride a `BufferOffset {
  buffer, offset_in_bytes }` (`utils.rs`), set by the backend as `start_offset() *
  dtype.size_in_bytes()`. `im2col`, `im2col1d`, `upsample_nearest2d`, `upsample_bilinear2d`,
  `conv_transpose1d`, `conv_transpose2d` are **offset-capable** (`start_offset: accepted`).
  `max_pool2d` and `avg_pool2d` are **NOT** offset-capable: the backend passes the raw `&self.buffer`
  (`storage.rs:1295` / `:1252`), so a non-zero-offset producer is contiguized by the planner first
  (pricing the inserted `Op::Contiguize` from its own FKC contract, §4.3 / §4.4). `col2im1d` carries
  a `BufferOffset` on src but is fed zero-offset in practice.
- **`reverse_strides: rejected` everywhere in this file.** The mixed-radix de-linearizer indexes
  with **unsigned** `uint` strides (bounded by `uint` index range), so none of these kernels walks a
  signed (negative) stride. A flipped/reversed view is normalized to a non-negative copy by an
  upstream movement kernel before dispatch (the as-built behavior; §4.1.1).
- **Output is always a fresh contiguous buffer, `aliasing: none`** (`device.new_buffer`); the kernel
  writes `out[tid]` densely. No in-place op in this family.
- **Numerics by op:** the im2col / col2im / upsample-nearest / max-pool kernels move data (gather /
  scatter / select) — max-pool is exact (a selection, not an accumulation); `col2im1d` accumulates
  in the operand dtype `T`. `upsample_bilinear2d` interpolates in **f32** then casts; `avg_pool2d`
  uses an **f32** accumulator for float dtypes (integer accumulator for int types); the transposed
  convolutions accumulate in **f32** for float dtypes. Per-element store narrows once.
- **Dispatch carriers — most of this family has no `fuel-dispatch` op yet.** Only `conv_transpose2d`
  has a real `OpKind::ConvTranspose2D` + `OpParams::ConvTranspose2D` carrier
  (`dispatch.rs:127` / `kernel.rs:244`) and is **registrable as-is**. The other eight are building
  blocks or graph-only ops with **no `OpKind` and (mostly) no `OpParams` variant** in
  `fuel-dispatch`; their contracts parse-validate but, like an MX contract (§6), are **NOT
  registrable until a dispatch carrier lands** — an importer returns the unknown-`OpKind` /
  `BadOpParamsVariant` error (the `MxNotYetRegistrable`-class discipline). Each such kernel carries a
  **Status** note and a `[consumer-ahead]` marker on `op_kind` / `op_params.variant`. The two
  im2col gathers and `col2im1d` are the internal building blocks of the backend `conv2d()` /
  `conv1d()` / `conv_transpose1d()` paths (im2col → matmul → copy), so they would be folded into the
  conv op-kind's lowering rather than dispatched directly.

This file is rendered by mdBook and parsed by the FKC importer; the prose is documentation and the
` ```fkc ` block is authoritative (§3.1). Costs are marked `provenance: judge_measured` — the Judge
bootstraps them. FLOPs/bandwidth are given as derivable formula hints where genuinely derivable from
the op geometry (§4.4) and are the author's structural prior; the non-derivable `overhead_ns`
(per-device launch cost) is left as `~`, never a fabricated absolute constant.

---

## im2col  (2-D im2col gather; building block of conv2d, NO own dispatch carrier)

2-D im2col: gather `(b,h_out,w_out,c_in,h_k,w_k)` patches from a strided NCHW source, zero-padded out of range; f32/f16/bf16/u8/u32; strided + offset-capable source; data-movement only.

Lowers a 2-D convolution to a matmul by gathering each output position's receptive field into a
contiguous patch matrix. Destination `(b, h_out, w_out, c_in, h_k, w_k)` is gathered from source
`x (b, c_in, h_in, w_in)` walked through `src_strides[0..4]`; the source coordinate is
`in_h = h_out·stride + h_k·dilation − padding`, `in_w = w_out·stride + w_k·dilation − padding`, and
positions outside `[0, h_in) × [0, w_in)` write a literal zero (`metal_src/conv.metal:7-69`).
Because the source is read through explicit `src_strides[]`, the input may be non-contiguous,
transposed, or broadcast; it is offset-capable via `BufferOffset`. The output is a fresh contiguous
buffer in im2col layout (the backend then matmuls it against the reshaped weight and copies into the
final conv output, `storage.rs:1099-1181`). No arithmetic — pure indexed copy + zero-fill — so the
result is exact and bit-stable.

**Status — building block, NO dispatch carrier.** Per the inventory (`metal.md:261`) `im2col` is an
**internal building block** of the backend `conv2d()` path, not a directly dispatched op: there is
**no `OpKind::Im2Col` and no `OpParams::Im2Col` in `fuel-dispatch`** (the only conv carriers are
`Conv2D` / `ConvTranspose2D`, `dispatch.rs:122` / `:127`). The carrier op for this gather is
`OpKind::Conv2D` / `OpParams::Conv2D` (im2col is the *lowering strategy* the planner picks for
Conv2D, not its own op-kind). This contract therefore parse-validates but is **NOT registrable as a
standalone op** (§10.7: no real `Im2Col` `OpKind`/`OpParams`); the `op_kind` field names the
intended `Im2Col` marker as a forward-looking lowering tag. An importer returns the unknown-`OpKind`
/ `BadOpParamsVariant` error until im2col-based conv lowering is modeled as a registrable sub-op.

```fkc
kernel: im2col
op_kind: Im2Col                # [consumer-ahead] NO such OpKind/OpParams in fuel-dispatch; building block of
                               # OpKind::Conv2D's im2col lowering, not a standalone dispatched op (see Status)
blurb: "2-D im2col gather: (b,h_out,w_out,c_in,h_k,w_k) patches from a strided NCHW source, zero-padded; data-movement only."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::im2col_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [b, c_in, h_in, w_in] NCHW; read via src_strides[0..4]
  op_params:
    variant: Im2Col               # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      dst_numel: { kind: usize }
      h_out:     { kind: usize }
      w_out:     { kind: usize }
      h_k:       { kind: usize }
      w_k:       { kind: usize }
      stride:    { kind: usize }
      padding:   { kind: usize }
      dilation:  { kind: usize }
      src_dims:    { kind: "[usize; 4]" }
      src_strides: { kind: "[usize; 4]", note: "source walked through these strides; broadcast = stride 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)      # output dtype = source dtype
      shape_rule: "from_params(b=src.dim[0], h_out, w_out, c_in=src.dim[1], h_k, w_k)"   # im2col layout
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # reads src_strides[] directly; no contiguize needed
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the bytes hint below is the structural prior
  class: strided_elementwise
  # pure gather + zero-fill: one read + one write per dst element (no FLOPs)
  flops: "0"
  bytes_moved: "2 * dst_numel * dtype_bytes"
  overhead_ns: ~                      # Metal command-buffer submit launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "dst_numel * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # indexed copy + zero-fill; no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure gather with zero-padding (literal 0 out of range); no arithmetic, exact and bitwise-stable. dtype preserved."

determinism: bitwise
```

---

## im2col1d  (1-D im2col gather; building block of conv1d, NO own dispatch carrier)

1-D im2col: gather `(b,l_out,c_in,l_k)` patches from a strided source, zero-padded; f32/f16/bf16/u8/u32; strided + offset-capable source; data-movement only.

The 1-D analogue of `im2col`: lowers conv1d to a matmul by gathering each output position's
receptive field. Destination `(b, l_out, c_in, l_k)` is gathered from source `x (b, c_in, l_in)`
walked through `src_strides[0..3]`; source coordinate `in_l = l_out·stride + l_k·dilation − padding`,
out-of-range positions write a literal zero (`metal_src/conv.metal:116-159`). Strided source
(non-contiguous / transposed / broadcast capable), offset-capable via `BufferOffset`; fresh
contiguous im2col output that the backend then matmuls and copies into the conv1d output
(`storage.rs:905-980`). No arithmetic — exact and bit-stable.

**Status — building block, NO dispatch carrier.** Per the inventory (`metal.md:271`) `im2col1d` is
the **internal building block** of the backend `conv1d()` path. There is **no `OpKind::Im2Col1d`**
in `fuel-dispatch` (no `OpKind::Conv1D` either — the only forward-conv OpKind is `Conv2D`); the
related `OpParams::Conv1D(ParamsConv1D)` variant *does* exist (`kernel.rs:211`) but there is no
matching `OpKind::Conv1D` and no `Im2Col1d` op-kind to register against. This contract
parse-validates but is **NOT registrable** (§10.7); the `op_kind` field names the intended `Im2Col1d`
lowering marker. An importer returns the unknown-`OpKind` error until a conv1d dispatch carrier lands.

```fkc
kernel: im2col1d
op_kind: Im2Col1d              # [consumer-ahead] NO such OpKind in fuel-dispatch; building block of conv1d's
                               # im2col lowering (OpParams::Conv1D exists, kernel.rs:211, but no OpKind::Conv1D) (see Status)
blurb: "1-D im2col gather: (b,l_out,c_in,l_k) patches from a strided source, zero-padded; data-movement only."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::im2col1d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 3                       # [b, c_in, l_in]; read via src_strides[0..3]
  op_params:
    variant: Conv1D               # [consumer-ahead] OpParams::Conv1D(ParamsConv1D) exists (kernel.rs:211) but
                                  # no OpKind::Conv1D / Im2Col1d carrier; not registrable (see Status)
    fields:
      dst_numel: { kind: usize }
      l_out:     { kind: usize }
      l_k:       { kind: usize }
      stride:    { kind: usize }
      padding:   { kind: usize }
      dilation:  { kind: usize }
      src_dims:    { kind: "[usize; 3]" }
      src_strides: { kind: "[usize; 3]", note: "source walked through these strides; broadcast = stride 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(b=src.dim[0], l_out, c_in=src.dim[1], l_k)"   # im2col layout
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the bytes hint below is the structural prior
  class: strided_elementwise
  flops: "0"                          # pure gather + zero-fill
  bytes_moved: "2 * dst_numel * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "dst_numel * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure gather with zero-padding; no arithmetic, exact and bitwise-stable. dtype preserved."

determinism: bitwise
```

---

## col2im1d  (1-D col2im scatter-accumulate; building block of conv_transpose1d, NO own dispatch carrier)

1-D col2im: scatter-accumulate columns back to an image; dst (b,c_out,l_out) from src (b,l_in,c_out,l_k); f32/f16/bf16/u8/u32; contiguous-assumed source; T-precision accumulate.

The inverse of `im2col1d`: scatters the column matrix back into image space and accumulates
overlapping contributions. Output `(b, c_out, l_out)` is zero-initialized then for each
`(b, l_in, c_out, l_k)` source element accumulated (`+=`) into `l_out = l_in·stride + l_k`
(`metal_src/conv.metal:71-113`); the backend uses it for the col2im branch of `conv_transpose1d()`
(`storage.rs:982-1097`). Unlike the rest of this family, `col2im1d` computes its **internal strides
from the shape** and does **not** read a `src_strides[]` array, so the source is **contiguous-assumed**
(`requires_contiguous`); it carries a `BufferOffset` on src but is fed zero-offset in practice.
Accumulation is in the operand dtype `T` (no f32 widening), with a fixed scatter order, so the
result is deterministic and same-hardware bit-stable.

**Status — building block, NO dispatch carrier.** Per the inventory (`metal.md:279`) `col2im1d` is
the **internal building block** of the backend `conv_transpose1d()` col2im branch. There is **no
`OpKind::Col2Im1d` and no `OpKind::ConvTranspose1D`** in `fuel-dispatch` (the related
`OpParams::ConvTranspose1D(ParamsConvTranspose1D)` exists, `kernel.rs:233`, but with no matching
OpKind). This contract parse-validates but is **NOT registrable** (§10.7); the `op_kind` field names
the intended `Col2Im1d` lowering marker. An importer returns the unknown-`OpKind` error until a
conv_transpose1d dispatch carrier lands.

```fkc
kernel: col2im1d
op_kind: Col2Im1d             # [consumer-ahead] NO such OpKind in fuel-dispatch; building block of
                              # conv_transpose1d's col2im branch (OpParams::ConvTranspose1D exists, kernel.rs:233) (see Status)
blurb: "1-D col2im scatter-accumulate: dst (b,c_out,l_out) from src (b,l_in,c_out,l_k), zero-init then +=; T-precision accumulate."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::col2im1d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [b, l_in, c_out, l_k]; internal strides from shape (NO src_strides[] array)
  op_params:
    variant: ConvTranspose1D      # [consumer-ahead] OpParams::ConvTranspose1D exists (kernel.rs:233) but no
                                  # OpKind::ConvTranspose1D / Col2Im1d carrier; not registrable (see Status)
    fields:
      dst_el: { kind: usize }
      l_out:  { kind: usize }
      l_in:   { kind: usize }
      c_out:  { kind: usize }
      k_size: { kind: usize }
      stride: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(b=src.dim[0], c_out, l_out)"
      layout_guarantee: contiguous
      aliasing: none                # fresh zero-init buffer scatter-accumulated; no input aliased

caps:
  awkward_layout_strategy: requires_contiguous   # ← contiguous-assumed src; planner inserts Op::Contiguize + sums its cost
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes hints below are the structural prior
  class: strided_elementwise
  # one add per source column element scattered into the image: src_numel = b * l_in * c_out * k_size
  flops: "src.dim[0] * src.dim[1] * src.dim[2] * src.dim[3]"
  bytes_moved: "(src.dim[0]*src.dim[1]*src.dim[2]*src.dim[3] + dst_el) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "dst_el * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # fixed scatter-accumulation order into a zero-init buffer, T-precision
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "T-precision accumulator (NO f32 widening for half); deterministic fixed scatter order into a fresh zero-init buffer; overlapping contributions accumulated."

determinism: same_hardware_bitwise
```

---

## upsample_nearest2d  (nearest-neighbour 2-D upsample, NO dispatch carrier)

Nearest-neighbour 2-D upsample over NCHW; f32/f16/bf16/u8/u32; strided + offset-capable source; data-movement only (exact).

Nearest-neighbour spatial upsample on a rank-4 NCHW input (treated `b, c, w, h` internally). For
each output `(w_out, h_out)` position the source coordinate is `floor(out · scale)` where
`w_scale`/`h_scale` are the in/out ratios; the picked source value is copied through
(`metal_src/conv.metal:161-200`). Source is read through `src_strides[0..4]` (strided / transposed /
broadcast capable), offset-capable via `BufferOffset`; fresh contiguous output
(`storage.rs:1342-1380`). No arithmetic on the values — a pure indexed copy — so the result is exact
and bit-stable.

**Status — NO dispatch carrier.** Per the inventory (`metal.md:288`) this kernel is reachable
through the backend `upsample_nearest2d()` but has **no `OpKind::UpsampleNearest2D` and no
`OpParams::UpsampleNearest2D` in `fuel-dispatch`**. A graph-side `Op::UpsampleNearest2D` exists in
`fuel-core` (`op.rs:103`, fields `target_h: usize`, `target_w: usize`) but is not yet lowered to a
dispatch op-kind. Like an MX contract (§6) this parse-validates but is **NOT registrable as-is**
(§10.7: no real `OpKind`/`OpParams`); the `op_kind`/`variant` fields name the intended carrier as a
forward marker. An importer returns the unknown-`OpKind` / `BadOpParamsVariant` error until a
dispatch carrier lands.

```fkc
kernel: upsample_nearest2d
op_kind: UpsampleNearest2D    # [consumer-ahead] NO such OpKind/OpParams in fuel-dispatch (graph Op::UpsampleNearest2D
                              # exists, op.rs:103, but no dispatch carrier) (see Status)
blurb: "Nearest-neighbour 2-D upsample over NCHW; strided + offset-capable source; pure indexed copy (exact)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::upsample_nearest2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [N, C, H, W]; read via src_strides[0..4]
  op_params:
    variant: UpsampleNearest2D    # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      w_out:   { kind: usize }
      h_out:   { kind: usize }
      w_scale: { kind: f32, note: "in/out ratio along W" }
      h_scale: { kind: f32, note: "in/out ratio along H" }
      src_dims:    { kind: "[usize; 4]" }
      src_strides: { kind: "[usize; 4]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(N=src.dim[0], C=src.dim[1], h_out, w_out)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the bytes hint below is the structural prior
  class: strided_elementwise
  # one read + one write per output element; out_numel = N * C * h_out * w_out
  flops: "0"
  bytes_moved: "2 * src.dim[0] * src.dim[1] * h_out * w_out * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "src.dim[0] * src.dim[1] * h_out * w_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # nearest = pure value copy, no arithmetic
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Nearest-neighbour copies the source value through unchanged; no arithmetic, exact and bitwise-stable. dtype preserved."

determinism: bitwise
```

---

## upsample_bilinear2d  (bilinear 2-D upsample, NO dispatch carrier)

Bilinear 2-D upsample over NCHW (PyTorch align_corners + scale semantics); f32/f16/bf16/u8/u32; strided + offset-capable source; f32 interpolation then cast.

Bilinear spatial upsample matching PyTorch `align_corners`/scale semantics. For each output
`(h_out, w_out)` the four surrounding source samples are weighted and summed in **f32**, then cast
back to the operand dtype on store (`metal_src/conv.metal:202-284`). Source read through
`src_strides[0..4]` (strided / transposed / broadcast capable), offset-capable via `BufferOffset`;
fresh contiguous NCHW output (`storage.rs:1382-1437`). `align_corners` and the optional per-axis
scale factors (`has_scale_h`/`scale_h_factor`, `has_scale_w`/`scale_w_factor`) are explicit params.
Because interpolation accumulates in f32 with a fixed four-tap order, the result is deterministic and
same-hardware bit-stable, but half-precision output narrows on store (drifts versus an all-f32 path).

**Status — NO dispatch carrier.** Per the inventory (`metal.md:296`) this kernel is reachable through
the backend `upsample_bilinear2d()` but has **no `OpKind::UpsampleBilinear2D` and no
`OpParams::UpsampleBilinear2D` in `fuel-dispatch`**. A graph-side `Op::UpsampleBilinear2D` exists
(`op.rs:108`, fields `target_h`, `target_w`, `align_corners: bool`) but is not yet lowered to a
dispatch op-kind. This parse-validates but is **NOT registrable as-is** (§10.7); the
`op_kind`/`variant` fields name the intended carrier. An importer returns the unknown-`OpKind` /
`BadOpParamsVariant` error until a dispatch carrier lands.

```fkc
kernel: upsample_bilinear2d
op_kind: UpsampleBilinear2D   # [consumer-ahead] NO such OpKind/OpParams in fuel-dispatch (graph Op::UpsampleBilinear2D
                              # exists, op.rs:108, but no dispatch carrier) (see Status)
blurb: "Bilinear 2-D upsample over NCHW (align_corners + scale); strided + offset-capable source; f32 interpolation then cast."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::upsample_bilinear2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [N, C, H, W]; read via src_strides[0..4]
  op_params:
    variant: UpsampleBilinear2D   # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      w_out:          { kind: usize }
      h_out:          { kind: usize }
      align_corners:  { kind: bool }
      has_scale_h:    { kind: bool }
      scale_h_factor: { kind: f32 }
      has_scale_w:    { kind: bool }
      scale_w_factor: { kind: f32 }
      src_dims:    { kind: "[usize; 4]" }
      src_strides: { kind: "[usize; 4]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(N=src.dim[0], C=src.dim[1], h_out, w_out)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided
  fast_paths:
    - { when: "align_corners == false", note: "half-pixel sample-coordinate path" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes hints below are the structural prior
  class: strided_elementwise
  # four-tap weighted sum per output element (~8 flops: 4 muls + 3 adds + coord math); out_numel = N*C*h_out*w_out
  flops: "8 * src.dim[0] * src.dim[1] * h_out * w_out"
  bytes_moved: "(4 * src.dim[0] * src.dim[1] * h_out * w_out + src.dim[0] * src.dim[1] * h_out * w_out) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "src.dim[0] * src.dim[1] * h_out * w_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 four-tap interpolation, fixed order; half narrows on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Interpolation accumulated in f32 (fixed four-tap order) then cast to the output dtype; half (f16/bf16) narrows on store. Deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## max_pool2d  (2-D max pooling, NO dispatch carrier)

2-D max pooling over NCHW; f32/f16/bf16/u32/u8; strided source but ZERO-OFFSET only; exact (selection, no accumulation).

2-D max pooling on a rank-4 NCHW input. For each `(w_k × h_k)` window stepped by
`(w_stride, h_stride)` the maximum source value is emitted (`metal_src/conv.metal:431-497`). The
source is read through `src_strides[0..4]` (strided / transposed / broadcast capable), **but the
backend passes the raw `&self.buffer`** (`storage.rs:1295`), so the kernel is **zero-offset only** —
a non-zero-offset producer is contiguized by the planner first. Channels pass through unchanged
(pool is per-channel); fresh contiguous output. A window max is a pure selection (no rounding, no
accumulation), so the result is exact and bit-stable; NaN follows the `fast::max`/`>` window scan.

**Status — NO dispatch carrier.** Per the inventory (`metal.md:305`) this kernel is reachable
through the backend `max_pool2d()` but has **no `OpKind::MaxPool2D` and no `OpParams::MaxPool2D` in
`fuel-dispatch`**. A graph-side `Op::MaxPool2D` exists in `fuel-core` (`op.rs:93`, fields
`kernel_size: (usize, usize)`, `stride: (usize, usize)`) but is not yet lowered to a dispatch
op-kind. This parse-validates but is **NOT registrable as-is** (§10.7); the `op_kind`/`variant`
fields name the intended carrier (tuple `kernel_size`/`stride`, matching the metal kernel's
`(w_k, h_k)` / `(w_stride, h_stride)`). An importer returns the unknown-`OpKind` /
`BadOpParamsVariant` error until a dispatch carrier lands.

```fkc
kernel: max_pool2d
op_kind: MaxPool2D            # [consumer-ahead] NO such OpKind/OpParams in fuel-dispatch (graph Op::MaxPool2D
                              # exists, op.rs:93, but no dispatch carrier) (see Status)
blurb: "2-D max pooling over NCHW; strided source, zero-offset only; per-channel; exact (selection, no accumulation)."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::max_pool2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U32, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, C, H, W]; read via src_strides[0..4]; ZERO-OFFSET only (raw buffer)
  op_params:
    variant: MaxPool2D            # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      w_k:      { kind: usize }
      h_k:      { kind: usize }
      w_stride: { kind: usize }
      h_stride: { kind: usize }
      src_dims:    { kind: "[usize; 4]" }
      src_strides: { kind: "[usize; 4]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(N=src.dim[0], C=src.dim[1], h_out=(src.dim[2]-h_k)/h_stride+1, w_out=(src.dim[3]-w_k)/w_stride+1)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # reads src_strides[]; offset NOT supported (zero-offset only)
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes hints below are the structural prior
  class: reduction
  # one max-compare per window element: N * C * h_out * w_out * (w_k * h_k); out_numel = N*C*h_out*w_out
  flops: "src.dim[0] * src.dim[1] * ((src.dim[2]-h_k)/h_stride+1) * ((src.dim[3]-w_k)/w_stride+1) * w_k * h_k"
  bytes_moved: "(src.dim[0]*src.dim[1]*src.dim[2]*src.dim[3] + src.dim[0]*src.dim[1]*((src.dim[2]-h_k)/h_stride+1)*((src.dim[3]-w_k)/w_stride+1)) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "src.dim[0] * src.dim[1] * ((src.dim[2]-h_k)/h_stride+1) * ((src.dim[3]-w_k)/w_stride+1) * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # max is exact (selection, not accumulation): bit-identical everywhere
  max_ulp: 0
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Pure max-reduction over the window: no rounding, no accumulation; result is bitwise-stable. NaN follows the fast::max / '>' window-scan semantics."

determinism: same_hardware_bitwise
```

---

## avg_pool2d  (2-D average pooling, NO dispatch carrier)

2-D average pooling over NCHW (sum / (w_k·h_k)); f32/f16/bf16/u32/u8; strided source but ZERO-OFFSET only; f32 accumulator for floats.

2-D average pooling on a rank-4 NCHW input. For each `(w_k × h_k)` window stepped by
`(w_stride, h_stride)` the source values are summed and divided by `w_k·h_k`
(`metal_src/conv.metal:370-429`; shares the `call_pool2d` wrapper with `max_pool2d`). Float dtypes
accumulate in an **f32** accumulator (`A = float`) then cast on store; integer dtypes accumulate in
the integer type. Source read through `src_strides[0..4]` (strided / transposed / broadcast capable),
**but zero-offset only** (backend passes the raw `&self.buffer`, `storage.rs:1252`) — a non-zero
producer is contiguized first. Per-channel; fresh contiguous output. The f32 accumulation order is
fixed, so the result is deterministic and same-hardware bit-stable; half output narrows on store.

**Status — NO dispatch carrier.** Per the inventory (`metal.md:314`) this kernel is reachable through
the backend `avg_pool2d()` but has **no `OpKind::AvgPool2D` and no `OpParams::AvgPool2D` in
`fuel-dispatch`**. A graph-side `Op::AvgPool2D` exists in `fuel-core` (`op.rs:87`, fields
`kernel_size: (usize, usize)`, `stride: (usize, usize)`) but is not yet lowered to a dispatch
op-kind. This parse-validates but is **NOT registrable as-is** (§10.7); the `op_kind`/`variant`
fields name the intended carrier. An importer returns the unknown-`OpKind` / `BadOpParamsVariant`
error until a dispatch carrier lands.

```fkc
kernel: avg_pool2d
op_kind: AvgPool2D            # [consumer-ahead] NO such OpKind/OpParams in fuel-dispatch (graph Op::AvgPool2D
                              # exists, op.rs:87, but no dispatch carrier) (see Status)
blurb: "2-D average pooling over NCHW (sum / w_k*h_k); strided source, zero-offset only; per-channel; f32 accumulator for floats."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::avg_pool2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U32, U8]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: rejected, reverse_strides: rejected }
      rank: 4                       # [N, C, H, W]; read via src_strides[0..4]; ZERO-OFFSET only (raw buffer)
  op_params:
    variant: AvgPool2D            # [consumer-ahead] NO such OpParams variant; see Status
    fields:
      w_k:      { kind: usize }
      h_k:      { kind: usize }
      w_stride: { kind: usize }
      h_stride: { kind: usize }
      src_dims:    { kind: "[usize; 4]" }
      src_strides: { kind: "[usize; 4]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(N=src.dim[0], C=src.dim[1], h_out=(src.dim[2]-h_k)/h_stride+1, w_out=(src.dim[3]-w_k)/w_stride+1)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # reads src_strides[]; offset NOT supported (zero-offset only)
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes hints below are the structural prior
  class: reduction
  # one add per window element + one divide per output element; out_numel = N*C*h_out*w_out
  flops: "src.dim[0] * src.dim[1] * ((src.dim[2]-h_k)/h_stride+1) * ((src.dim[3]-w_k)/w_stride+1) * (w_k * h_k + 1)"
  bytes_moved: "(src.dim[0]*src.dim[1]*src.dim[2]*src.dim[3] + src.dim[0]*src.dim[1]*((src.dim[2]-h_k)/h_stride+1)*((src.dim[3]-w_k)/w_stride+1)) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "src.dim[0] * src.dim[1] * ((src.dim[2]-h_k)/h_stride+1) * ((src.dim[3]-w_k)/w_stride+1) * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 accumulator (floats), fixed window-sum order; half narrows on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Float dtypes accumulate the window sum in f32 then divide and cast on store; integer dtypes accumulate in the integer type. Fixed summation order, deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## conv_transpose1d  (naive 1-D transposed convolution, NO dispatch carrier)

Naive 1-D transposed convolution (per-output accumulate over kernel × c_in); f32/f16/bf16/u8/u32; strided + offset-capable src AND kernel; f32 accumulator for floats.

Naive (non-col2im) 1-D transposed convolution: each output element accumulates over the kernel
window and input channels (`metal_src/conv.metal:501-567`). Both the source `src` and the kernel
weight `k` are read through their own stride arrays (`src_strides`, `k_strides`) and are each
offset-capable via `BufferOffset`, so neither needs contiguizing. Float dtypes accumulate in an
**f32** accumulator (`A = float`) then cast on store; integer dtypes accumulate in the integer type.
Fresh contiguous output `(b, c_out, l_out)`; the backend uses this naive branch of
`conv_transpose1d()` when it does not take the col2im path (`storage.rs:982-1097`). Fixed
accumulation order ⇒ deterministic and same-hardware bit-stable; half narrows on store.

**Status — NO dispatch carrier.** Per the inventory (`metal.md:322`) this kernel is reachable through
the backend `conv_transpose1d()` naive branch but has **no `OpKind::ConvTranspose1D` in
`fuel-dispatch`** (the only conv OpKinds are `Conv2D` / `ConvTranspose2D`). The related
`OpParams::ConvTranspose1D(ParamsConvTranspose1D)` *does* exist (`kernel.rs:233`) but with no
matching `OpKind`, so there is no key to register against. This parse-validates but is **NOT
registrable as-is** (§10.7); the `op_kind` field names the intended `ConvTranspose1D` carrier. An
importer returns the unknown-`OpKind` error until a conv_transpose1d dispatch op-kind lands.

```fkc
kernel: conv_transpose1d
op_kind: ConvTranspose1D      # [consumer-ahead] NO such OpKind in fuel-dispatch (OpParams::ConvTranspose1D exists,
                              # kernel.rs:233, but no OpKind::ConvTranspose1D carrier) (see Status)
blurb: "Naive 1-D transposed convolution; strided + offset-capable src and kernel; f32 accumulator for floats."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::conv_transpose1d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 3                       # [b, c_in, l_in]; read via src_strides
    - name: kernel
      dtypes: [F32, F16, BF16, U8, U32]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 3                       # [c_in, c_out, k_size]; read via k_strides
      shape_constraint: "same_as=src.dim[1]"   # kernel.dim[0] (c_in) == src.dim[1] (c_in)
  op_params:
    variant: ConvTranspose1D      # [consumer-ahead] OpParams::ConvTranspose1D(ParamsConvTranspose1D) exists
                                  # (kernel.rs:233) but no OpKind::ConvTranspose1D carrier; not registrable (see Status)
    fields:
      l_out:       { kind: usize }
      stride:      { kind: usize }
      padding:     { kind: usize }
      out_padding: { kind: usize }
      dilation:    { kind: usize }
      src_dims:    { kind: "[usize; 3]" }
      src_strides: { kind: "[usize; 3]" }
      k_dims:      { kind: "[usize; 3]" }
      k_strides:   { kind: "[usize; 3]" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(src)
      shape_rule: "from_params(b=src.dim[0], c_out=kernel.dim[1], l_out)"
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # reads src_strides[] and k_strides[]; offset-capable on both
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps; the flops/bytes hints below are the structural prior
  class: conv
  # MACs = b * c_out * c_in * k_size * l_out; 2 flops per MAC (mul + accumulate)
  flops: "2 * src.dim[0] * kernel.dim[1] * src.dim[1] * kernel.dim[2] * l_out"
  bytes_moved: "(src.dim[0]*src.dim[1]*src.dim[2] + kernel.dim[0]*kernel.dim[1]*kernel.dim[2] + src.dim[0]*kernel.dim[1]*l_out) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "src.dim[0] * kernel.dim[1] * l_out * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 accumulator (floats), fixed accumulation order; half narrows on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Float dtypes accumulate in f32 (A=float) then cast on store; integer dtypes accumulate in the integer type. Fixed accumulation order, deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

---

## conv_transpose2d  (naive 2-D transposed convolution; registry CONV_TRANSPOSE2D, REGISTRABLE)

Naive 2-D transposed convolution; f32/f16/bf16; strided + offset-capable src AND kernel; f32 accumulator (no int variants).

Naive 2-D transposed convolution ("deconv"): each output element accumulates over the kernel window
and input channels (`metal_src/conv.metal:569-647`). Both the input `src` and the kernel weight `k`
are read through their own stride arrays (`input_stride`, `k_stride`) and are each offset-capable via
`BufferOffset`, so neither needs contiguizing. Accumulation is in an **f32** accumulator (`A = float`)
then cast on store. **No integer variants are emitted** — f32/f16/bf16 only. Fresh contiguous output
`(N, Cout, Hout, Wout)`; the backend wires it through `conv_transpose2d()` (`storage.rs:1183-1250`).
Fixed accumulation order ⇒ deterministic and same-hardware bit-stable; half narrows on store.

**Registrable.** This is the one kernel of the family with a real dispatch carrier:
`OpKind::ConvTranspose2D` (`dispatch.rs:127`) + `OpParams::ConvTranspose2D` (`kernel.rs:244`). Note
the **transposed channel order** of the weight versus forward conv (`[Cin, Cout/groups, Kh, Kw]`,
matching the `OpParams::ConvTranspose2D` ABI; the inventory lists the metal kernel's op-params with
the conv-transpose geometry). The metal kernel itself carries no `groups` param in its launch
signature (the dispatch geometry supplies `groups`); this contract declares the
`OpParams::ConvTranspose2D` field set so the key and op-param schema match the carrier.

```fkc
kernel: conv_transpose2d
op_kind: ConvTranspose2D       # OpParams::ConvTranspose2D carrier (kernel.rs:244); OpKind::ConvTranspose2D (dispatch.rs:127)
blurb: "Naive 2-D transposed convolution; f32/f16/bf16; strided + offset-capable src and kernel; f32 accumulator."
backend: Metal
kernel_source: "metal-msl"
entry_point: "fuel_metal_kernels::convolution::conv_transpose2d_f32"   # one per dtype; §12.6
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [N, Cin, H_in, W_in] NCHW; read via input_stride[0..4]
    - name: weight
      dtypes: [F32, F16, BF16]
      layout: { contiguous: accepted, strided: accepted, broadcast_stride0: accepted, start_offset: accepted, reverse_strides: rejected }
      rank: 4                       # [Cin, Cout/groups, Kh, Kw] — transposed channel order vs Conv2D; read via k_stride[0..4]
      shape_constraint: "same_as=x.dim[1]"   # weight.dim[0] (Cin) == x.dim[1] (Cin)
  op_params:
    variant: ConvTranspose2D      # OpParams::ConvTranspose2D (primitive namespace; §3.7)
    fields:
      x_shape:        { kind: "[usize; 4]", note: "[N, Cin, H_in, W_in]" }
      w_shape:        { kind: "[usize; 4]", note: "[Cin, Cout/groups, Kh, Kw]" }
      out_shape:      { kind: "[usize; 4]", note: "[N, Cout, H_out, W_out]; kernel trusts this geometry" }
      stride:         { kind: "(usize, usize)" }
      padding:        { kind: "(usize, usize)" }
      output_padding: { kind: "(usize, usize)", note: "disambiguates H_out/W_out; transpose-only param" }
      dilation:       { kind: "(usize, usize)" }
      groups:         { kind: usize, constraint: "groups != 0; Cin % groups == 0; Cout % groups == 0" }

return:
  outputs:
    - name: out
      dtype_rule: passthrough(x)
      shape_rule: conv_transpose2d(params)   # [N, Cout, H_out, W_out] from OpParams::ConvTranspose2D geometry (§5.2)
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: handles_strided   # reads input_stride[] and k_stride[]; offset-capable on both
  fast_paths:
    - { when: "groups == 1", note: "full deconv; no per-group channel slicing" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 32

cost:
  provenance: judge_measured          # Judge bootstraps/refines (§4.4); the formula hints below are the derivable prior
  class: conv
  # scatter MACs = N * Cin * H_in * W_in * (Cout/groups) * Kh * Kw; 2 flops per MAC (mul + accumulate)
  flops: "2 * x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3] * (out_shape[1] / groups) * w_shape[2] * w_shape[3]"
  bytes_moved: "(x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3] + w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3] + out_shape[0]*out_shape[1]*out_shape[2]*out_shape[3]) * dtype_bytes"
  overhead_ns: ~                      # Metal launch cost — Judge-measured (not a fabricated constant)
  memory: { device_bytes: "out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * dtype_bytes", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # f32 accumulator, fixed accumulation order; half narrows on store
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true
  notes: "Accumulation in an f32 accumulator (A=float) then cast on store; f32/f16/bf16 only (no int variants). Fixed accumulation order, deterministic; not bit-stable cross-hardware."

determinism: same_hardware_bitwise
```

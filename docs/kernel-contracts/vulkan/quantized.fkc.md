---
fkc_version: 1
provider:
  name: fuel-vulkan-kernels
  backend: Vulkan                 # maps to BackendId::Vulkan
  kernel_source: "vulkan-slang"   # the BindingEntry.kernel_source tag
  link_registry: fuel_vulkan_backend::fkc::ENTRY_POINTS  # §12.6 symbol→KernelRef map
  revision_base: "git:f41137b4"   # provider build id, folded into kernel_revision_hash
---

# fuel-vulkan-kernels — quantized (GGML) kernel contracts

The `quant-matmul` family of the Vulkan stack: GGML block dequantizers (Q4_0 / Q8_0 /
Q4_K_M → f32), the two fused Q4_0×F32 matmul implementations (gemv decode path + tiled
prefill path), and the F32→Q8_0 quantizer used for KV-cache compression. Kernel sources
live in `fuel-kernels-source/kernels/*.slang`, AOT-compiled to SPIR-V in
`fuel-vulkan-kernels/spv/*.spv`, registered in `EMBEDDED`
(`fuel-vulkan-kernels/src/lib.rs:39`); the Rust dispatch wrappers live in
`fuel-vulkan-backend/src/lib.rs`.

All scales obey the **single-place rule** (§3.9.3): every kernel here consumes GGML
block-quant data whose scales are **INLINE** in the block stream (the f16 `d` — and the
6-bit packed scales/mins for Q4_K_M — ride inside each block). The scale therefore lives
in the FDX tensor's sidecar (`FDXQuant.scale_placement = INLINE`), **not** as a separate
graph input, so the quant operand's `fdx.quant.scale_operand` stays `~` and no FKC scale
operand is declared.

> **As-built dispatch-tag note (faithful to the inventory + sources).** The three
> dequantizers and the quantizer have **no dedicated `OpKind` variant** in the as-built
> `fuel-core-types::dispatch::OpKind` enum and **no `FusedOpId`** — they are dispatched
> directly by the Vulkan backend wrappers and tagged by the `fuel-core-types::capability`
> `Capability` tokens `DequantizeQ4_0` / `DequantizeQ8_0` / `DequantizeQ4KM` /
> `QuantizeQ8_0`. FKC §4.3 names `Op::Dequantize` as an ordinary FKC kernel, but that op
> arm does not yet exist in `op.rs` / `OpKind`. Each such section therefore records its
> dispatch tag in `op_kind:` as the `Capability` token and flags the missing `OpKind` in
> its long description (no fabricated op-kind). The two Q4_0 matmuls, by contrast, ARE the
> as-built fused op `QMATMUL` (`FusedOpId(14)`, `FusedOpParams::QMatMul`) and are authored
> as `fused_op:` contracts (fused cost-fn shape, no `&[DType]` argument; §4.4).

> **Quant input layout is a contiguous byte stream — `start_offset: rejected`.** Every
> kernel here uploads / reads the quant blocks as a `DType::U32`-typed ByteAddressBuffer
> from element 0 (the wrappers `upload_slice(..., DType::U32)` a freshly-packed blob, and
> the matmul activations are contiguous `[K]` / `[M,K]`). None walks strides, broadcasts,
> reverses strides, or takes a non-zero base offset, so each operand declares
> `{contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset:
> rejected, reverse_strides: rejected}` and `awkward_layout_strategy: requires_contiguous`.

## dequant_q4_0  (GGML Q4_0 block dequant → f32)

One-line: dequantize a GGML Q4_0 block stream (18-byte / 32-elem blocks) to a dense f32 buffer.

GGML Q4_0 stores each 32-element block as a 2-byte f16 scale `d` followed by 16 bytes of
packed 4-bit quants (one nibble per element). Dequant is `x = (nibble - 8) * d` — the
nibbles are unsigned 0..15, biased by 8 to center on zero, then scaled by the per-block
f16 scale `d` widened to f32. The kernel walks the raw byte stream as a `DType::U32`
ByteAddressBuffer: one thread per `(k, k+16)` nibble pair, with unaligned 1-byte reads
synthesized by `load_u8` word-extract (the 18-byte block stride is not u32-aligned). The
scale is **INLINE** in the block (single-place rule, §3.9.3) — there is no separate scale
operand. Output is a fresh contiguous f32 buffer of `n_blocks * 32` elements; the wrapper
allocates it (`alloc_device(..., DType::F32)`, `fuel-vulkan-backend/src/lib.rs:9630`), so
the kernel only fills bytes and the output never aliases the input. Numerics: the only
lossy step is the f16→f32 scale widen and the integer nibble bias — both exact in f32;
dequant itself is bit-stable on the same hardware (deterministic linear walk, no atomics).
Perf: bandwidth-bound — reads `n_blocks*18` bytes, writes `n_blocks*32*4` bytes; the f32
write dominates. Limitation: input must be a well-formed Q4_0 blob whose length is exactly
`n_blocks*18` (the wrapper validates this); contiguous-only, no strided/offset input.

As-built: dispatched by `Capability::DequantizeQ4_0` (`fuel-core-types/src/capability.rs:75`);
there is **no** `OpKind::DequantizeQ4_0` (the `op_kind:` slot below records the `Capability`
token as the dispatch tag — see the bundle header note). The weight operand is GGML_BLOCK
family `ggml_dtype: Q4_0` (`GgmlDType` code 2; §3.4).

```fkc
kernel: dequant_q4_0
registrable: false               # §3.10 describe-only: DequantizeQ4_0 is a Capability token, not a real OpKind; documented, not registered. op_kind below is the forward-looking dispatch-tag marker.
op_kind: DequantizeQ4_0          # as-built Capability tag (no OpKind variant; see header note)
blurb: "Dequantize a GGML Q4_0 block stream (18B/32-elem; (nibble-8)*d) to dense f32."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::dequant_q4_0"
kernel_revision_hash: auto

accept:
  inputs:
    - name: w_q
      dtypes: [U8]                 # opaque packed Q4_0 block byte stream (FDX §3 honesty stand-in); read internally as a U32 ByteAddressBuffer (access_granularity_bits)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat byte blob; n_blocks * 18 bytes
      fdx:
        requires_ext: true         # the U32 base is opaque quant bytes; meaning needs FDX quant sidecar
        quant:
          family: GGML_BLOCK       # FDXQuant.family
          ggml_dtype: Q4_0         # GgmlDType code 2; §3.4 — block grain rides ggml_dtype (INLINE f16 d per 32-elem block); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: weight
          scale_operand: ~         # scale is INLINE baked in the block (single-place rule, §3.9.3)
  op_params:
    variant: None                  # the wrapper packs (n_blocks, out_elements); no OpParams variant

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)     # Q4_0 → F32
      shape_rule: from_params(n_blocks)   # [n_blocks * 32]
      layout_guarantee: contiguous
      aliasing: none               # fresh alloc_device buffer; kernel fills, never aliases input

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []                   # single uniform path; no declared fast cell
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8       # 1-byte word-extract reads (unaligned 18B block stride)

cost:
  provenance: judge_measured       # Judge bootstraps; do not fabricate numbers
  class: cheap_elementwise         # bandwidth-bound block dequant
  flops: "n"                       # ~1 mul + 1 sub per output element (n = n_blocks * 32)
  bytes_moved: "n * 0.5625 + n * 4"   # read 18B/32-elem (0.5625 B/elem) + write 4B/elem f32
  overhead_ns: ~                   # judge_measured
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true   # deterministic linear walk; no atomics, no reduction
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                    # exactness of (nibble-8)*widen(d) not yet Judge-audited
  notes: "x = (nibble-8) * widen_f16_to_f32(d); only loss is the f16 scale widen (exact in f32) + integer bias."

determinism: same_hardware_bitwise
```

## dequant_q8_0  (GGML Q8_0 block dequant → f32)

One-line: dequantize a GGML Q8_0 block stream (34-byte / 32-elem blocks) to a dense f32 buffer.

GGML Q8_0 stores each 32-element block as a 2-byte f16 scale `d` followed by 32 signed
8-bit quants. Dequant is `x = qs * d` — each i8 quant widened and multiplied by the
per-block f16 scale `d` (widened to f32). The kernel walks the raw byte stream as a
`DType::U32` ByteAddressBuffer with **one thread per output element**; the 34-byte block
stride is not u32-aligned, so reads are byte-extracted from the word stream. The scale is
**INLINE** in the block (single-place rule). Output is a fresh contiguous f32 buffer
allocated by the wrapper (`dequantize_q8_0` `:9656`; a `_from_storage` variant at `:10025`
reads an already-resident device blob). Numerics: only the f16→f32 scale widen is lossy
(exact in f32); the i8 quant is exact; deterministic, no atomics. Perf: bandwidth-bound —
reads `n_blocks*34` bytes (~1.0625 B/elem), writes `n*4` bytes f32. Limitation:
contiguous-only; input length must be exactly `n_blocks*34`.

As-built: dispatched by `Capability::DequantizeQ8_0` (`capability.rs:74`); no
`OpKind::DequantizeQ8_0`. Weight operand is GGML_BLOCK `ggml_dtype: Q8_0` (code 8).

```fkc
kernel: dequant_q8_0
registrable: false               # §3.10 describe-only: DequantizeQ8_0 is a Capability token, not a real OpKind; documented, not registered. op_kind below is the forward-looking dispatch-tag marker.
op_kind: DequantizeQ8_0          # as-built Capability tag (no OpKind variant; see header note)
blurb: "Dequantize a GGML Q8_0 block stream (34B/32-elem; qs*d) to dense f32."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::dequant_q8_0"
kernel_revision_hash: auto

accept:
  inputs:
    - name: w_q
      dtypes: [U8]                 # opaque packed Q8_0 block byte stream (FDX §3 honesty stand-in); read internally as a U32 ByteAddressBuffer (access_granularity_bits)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat byte blob; n_blocks * 34 bytes
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q8_0         # GgmlDType code 8; §3.4 — block grain rides ggml_dtype (INLINE f16 d per 32-elem block); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: weight
          scale_operand: ~         # INLINE baked scale (single-place rule)
  op_params:
    variant: None                  # wrapper packs (n_blocks, out_elements)

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)     # Q8_0 → F32
      shape_rule: from_params(n_blocks)   # [n_blocks * 32]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8       # byte-extract reads over the 34B (non-u32-aligned) block stride

cost:
  provenance: judge_measured
  class: cheap_elementwise
  flops: "n"                       # 1 mul per output element (n = n_blocks * 32)
  bytes_moved: "n * 1.0625 + n * 4"   # read 34B/32-elem (1.0625 B/elem) + write 4B/elem f32
  overhead_ns: ~
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x = qs * widen_f16_to_f32(d); i8 quant exact, only the f16 scale widen is lossy (exact in f32)."

determinism: same_hardware_bitwise
```

## dequant_q4_km  (GGML Q4_K_M super-block dequant → f32)

One-line: dequantize a GGML Q4_K_M super-block stream (144-byte / 256-elem; 6-bit packed
scales+mins) to a dense f32 buffer.

GGML Q4_K_M is a K-quant: each 144-byte super-block carries 256 elements split into eight
32-element sub-blocks, with per-super-block f16 `d` and `dmin`, and **6-bit packed
per-sub-block scales and mins** decoded by llama.cpp's `get_scale_min_k4`. The dequant is
`x = d * scale_j * q - dmin * min_j` per sub-block `j`. The kernel runs one workgroup per
super-block, 32 threads, 8 elements per thread, with the 6-bit scale/min unpacking done
per sub-block. All scales/mins are **INLINE** in the super-block (single-place rule). Output
is a fresh contiguous f32 buffer of `n_blocks * 256` elements
(`dequantize_q4_km` `:10160`). Numerics: the f16 widens of `d`/`dmin` and the 6-bit scale
decode are the loss sources; deterministic, no atomics. Perf: bandwidth-bound — reads
`n_blocks*144` bytes (0.5625 B/elem), writes `n*4` bytes f32; the 6-bit unpack adds a small
compute term over plain Q4_0. Limitation: contiguous-only; input length exactly
`n_blocks*144`; this is the GGUF "Q4_K_M" mixed K-quant — its storage dtype is
`GgmlDType::Q4K` (code 12), the "medium" distinction being a kernel/dispatch fact, not a
separate storage dtype (§3.4).

As-built: dispatched by `Capability::DequantizeQ4KM` (`capability.rs:76`); no
`OpKind::DequantizeQ4KM`. Weight operand is GGML_BLOCK `ggml_dtype: Q4K` (code 12), the
GGUF `Q4_K_M` weight — `ggml_dtype` MUST be written `Q4K`, never `Q4_K_M` (§3.4 /
`QuantIncoherent`).

```fkc
kernel: dequant_q4_km
registrable: false               # §3.10 describe-only: DequantizeQ4KM is a Capability token, not a real OpKind; documented, not registered. op_kind below is the forward-looking dispatch-tag marker.
op_kind: DequantizeQ4KM          # as-built Capability tag (no OpKind variant; see header note)
blurb: "Dequantize a GGML Q4_K_M super-block stream (144B/256-elem; 6-bit packed scales+mins) to dense f32."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::dequant_q4_km"
kernel_revision_hash: auto

accept:
  inputs:
    - name: w_q
      dtypes: [U8]                 # opaque packed Q4_K_M super-block byte stream (FDX §3 honesty stand-in); read internally as a U32 ByteAddressBuffer (access_granularity_bits)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat byte blob; n_blocks * 144 bytes
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4K          # GgmlDType code 12 (GGUF Q4_K_M weight); §3.4 — never "Q4_K_M"; block grain rides ggml_dtype (INLINE f16 d/dmin + 6-bit packed per-sub-block scales/mins); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: weight
          scale_operand: ~         # all scales/mins INLINE baked (single-place rule)
  op_params:
    variant: None                  # wrapper packs (n_blocks, out_elements)

return:
  outputs:
    - name: out
      dtype_rule: dequant(w_q)     # Q4_K_M → F32
      shape_rule: from_params(n_blocks)   # [n_blocks * 256]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8       # byte/bit-extract reads (6-bit packed scales+mins)

cost:
  provenance: judge_measured
  class: cheap_elementwise         # bandwidth-bound with a small 6-bit-unpack compute term
  flops: "2 * n"                   # ~1 mul + 1 mul-sub per output elem + per-sub-block scale decode (n = n_blocks * 256)
  bytes_moved: "n * 0.5625 + n * 4"   # read 144B/256-elem (0.5625 B/elem) + write 4B/elem f32
  overhead_ns: ~
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: true
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "x = d*scale_j*q - dmin*min_j (llama.cpp get_scale_min_k4); f16 d/dmin widen + 6-bit scale decode are the loss sources; deterministic."

determinism: same_hardware_bitwise
```

## qmatvec_q4_0  (fused Q4_0×F32 gemv, M==1)

One-line: fused Q4_0-weight × f32-activation gemv (decode hot path); `out[n] = Σ_k A[k]·dequant(W)[n,k]`.

The decode hot path for Q4_0-quantized linear layers (M==1). The f32 activation vector
`A [K]` is multiplied against a Q4_0 weight matrix `W [N, K/32]` stored as a raw block
stream, dequantizing each weight block on the fly (`(nibble-8)*d`) and accumulating in
f32. One workgroup per output column `n`, 128 threads, subgroup reduction over the `K`
contraction; `K` must be a multiple of 32 (block size). The weight scale is **INLINE** in
the Q4_0 block stream (single-place rule, §3.9.3) — there is no separate scale operand.
Output is a fresh contiguous f32 vector `[N]`. Numerics: f32 accumulation with the same
f16-scale-widen dequant as `dequant_q4_0`; the subgroup-reduction order is
scheduler-dependent, so it is **not bit-stable** across runs of a reduction with
non-associative f32 addition. Perf: gemm_like / gemv — dominated by streaming the weight
blocks once (`N*K*0.5625` bytes) and the `2*N*K` MAC FLOPs over the contraction.
Limitation: M==1 only (the picker routes M>1 to `matmul_q4_0_tiled`); `K % 32 == 0`;
contiguous activation and contiguous weight blob.

As-built: this is one implementation of the **fused op** `QMATMUL` (`FusedOpId(14)`,
`fuel-graph/src/registry.rs:890`), carried by `FusedOpParams::QMatMul { quant_type, k, n }`
(`registry.rs:250`) — so it is a `fused_op:` contract (fused cost-fn shape, no `&[DType]`
argument; §4.4). Routed by `matmul_q4_0_bytes` `:4001` when `m == 1`; wrapper
`qmatvec_q4_0` `:10064`, `_slice` `:10105`.

```fkc
kernel: qmatvec_q4_0
fused_op: QMATMUL                # FusedOpId(14); FusedOpParams::QMatMul
blurb: "Fused Q4_0-weight x f32 gemv (M==1, decode hot path); INLINE Q4_0 scales; K % 32 == 0."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::qmatvec_q4_0"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a                      # f32 activation vector
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # [K]
    - name: w_q                    # Q4_0 weight block stream [N, K/32]
      dtypes: [U8]                 # opaque packed Q4_0 block byte stream (FDX §3 honesty stand-in); read internally as a U32 ByteAddressBuffer (access_granularity_bits)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat byte blob; N * (K/32) * 18 bytes
      shape_constraint: "divisible(k, 32)"   # K must be a multiple of the 32-elem block
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0         # GgmlDType code 2; §3.4 — block grain rides ggml_dtype (INLINE f16 d per 32-elem block); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: weight
          scale_operand: ~         # INLINE baked scale (single-place rule, §3.9.3)
  op_params:
    variant: QMatMul              # FusedOpParams::QMatMul (fused namespace; §3.7)
    fields:
      quant_type: { kind: QuantType, constraint: "== Q4_0" }
      k: { kind: usize, constraint: "k % 32 == 0 && == a.dim[-1]" }
      n: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)       # C = A @ dequant(W); output f32
      shape_rule: from_params(n)   # [N]
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "m == 1", note: "this kernel IS the M==1 route; picker selects it for gemv" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8       # byte-extract Q4_0 nibble reads over the 18B block stride

cost:
  provenance: judge_measured       # Judge bootstraps; do not fabricate numbers
  class: gemm_like
  # fused cost-fn shape (no &[DType] arg). M==1 gemv: 2 MACs per (n,k).
  flops: "2 * n * k"               # multiply-accumulate over the K contraction, N columns
  bytes_moved: "n * k * 0.5625 + k * 4 + n * 4"   # stream Q4_0 weight (0.5625 B/elem) + read A + write C
  overhead_ns: ~
  memory: { device_bytes: "n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # subgroup reduction order is scheduler-dependent
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                     # audited: no static bound (non-associative f32 reduction order)
  notes: "f32 accumulate; (nibble-8)*widen_f16(d) dequant; NOT bit-stable (subgroup reduction order varies)."

determinism: nondeterministic
```

## matmul_q4_0_tiled  (fused Q4_0×F32 tiled matmul, M>1)

One-line: fused Q4_0-weight × f32-activation tiled matmul (prefill path, M>1); TM=8 m-rows/tile.

The prefill path for Q4_0-quantized linear layers (M>1). The f32 activation matrix
`A [M, K]` is multiplied against a Q4_0 weight matrix `W [N, K/32]` (raw block stream),
dequantizing weight blocks on the fly and accumulating in f32. Shared-memory tiling with
TM=8 m-rows per tile and a 128-thread workgroup; one workgroup per `(m_tile, n_col)`. The
weight scale is **INLINE** (single-place rule). Output is a fresh contiguous f32 matrix
`[M, N]` row-major. Numerics: f32 accumulation; the tiled reduction order is
scheduler/tile-order-dependent, so **not bit-stable**. Perf: gemm_like — `2*M*N*K` MAC
FLOPs; the Q4_0 weight is streamed once per n-column-block (`N*K*0.5625` bytes) and reused
across the TM=8 m-rows in a tile, amortizing weight bandwidth. Limitation: M>1 (the picker
routes M==1 to `qmatvec_q4_0`); `K % 32 == 0`; contiguous activation and weight.

As-built: the second implementation of fused op `QMATMUL` (`FusedOpId(14)`),
`FusedOpParams::QMatMul { quant_type, k, n }` — a `fused_op:` contract (§4.4). Routed by
`matmul_q4_0_bytes` `:4001` for `m > 1`; wrapper `matmul_q4_0_tiled` `:10199`. It is a
sibling alternative to `qmatvec_q4_0` at the `QMATMUL` key, distinguished by its M>1
fast-path predicate (the picker, not the contract, makes the M==1-vs-M>1 choice; both
contracts are admissible and the planner ranks them on cost — §12.5).

```fkc
kernel: matmul_q4_0_tiled
fused_op: QMATMUL               # FusedOpId(14); FusedOpParams::QMatMul
blurb: "Fused Q4_0-weight x f32 tiled matmul (M>1 prefill; TM=8); INLINE Q4_0 scales; K % 32 == 0."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::matmul_q4_0_tiled"
kernel_revision_hash: auto

accept:
  inputs:
    - name: a                      # f32 activation matrix
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 2                      # [M, K]
    - name: w_q                    # Q4_0 weight block stream [N, K/32]
      dtypes: [U8]                 # opaque packed Q4_0 block byte stream (FDX §3 honesty stand-in); read internally as a U32 ByteAddressBuffer (access_granularity_bits)
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat byte blob; N * (K/32) * 18 bytes
      shape_constraint: "divisible(k, 32)"
      fdx:
        requires_ext: true
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q4_0         # GgmlDType code 2; §3.4 — block grain rides ggml_dtype (INLINE f16 d per 32-elem block); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: weight
          scale_operand: ~         # INLINE baked scale (single-place rule, §3.9.3)
  op_params:
    variant: QMatMul              # FusedOpParams::QMatMul (fused namespace; §3.7)
    fields:
      quant_type: { kind: QuantType, constraint: "== Q4_0" }
      k: { kind: usize, constraint: "k % 32 == 0 && == a.dim[-1]" }
      n: { kind: usize }

return:
  outputs:
    - name: out
      dtype_rule: fixed(F32)       # C = A @ dequant(W); output f32
      shape_rule: matmul(a, w_q)   # [M, N] row-major
      layout_guarantee: contiguous
      aliasing: none

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths:
    - { when: "m > 1", note: "this kernel IS the M>1 prefill route; picker selects it for matmul" }
  in_place: false
  alignment_bytes: 16
  access_granularity_bits: 8       # byte-extract Q4_0 nibble reads over the 18B block stride

cost:
  provenance: judge_measured       # Judge bootstraps; do not fabricate numbers
  class: gemm_like
  # fused cost-fn shape (no &[DType] arg).
  flops: "2 * m * n * k"           # standard GEMM MAC count
  bytes_moved: "n * k * 0.5625 + m * k * 4 + m * n * 4"   # stream Q4_0 weight + read A + write C
  overhead_ns: ~
  memory: { device_bytes: "m * n * 4", host_bytes: 0, disk_bytes: 0 }

precision:
  bit_stable_on_same_hardware: false   # tiled/scheduler-dependent f32 reduction order
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: true                     # audited: no static bound (non-associative tiled f32 reduction)
  notes: "f32 accumulate; (nibble-8)*widen_f16(d) dequant; NOT bit-stable (tiled reduction order varies)."

determinism: nondeterministic
```

## quantize_q8_0  (F32 → GGML Q8_0 quantize)

One-line: quantize a dense f32 buffer to a GGML Q8_0 block stream (34-byte / 32-elem; `d = max|x|/127`).

KV-cache compression: pack a contiguous f32 buffer into GGML Q8_0 blocks. Each 32-element
block computes its scale `d = max|x| / 127` over the block, then stores the f16-narrowed
`d` (2 bytes) followed by 32 signed-8-bit quants `qs = round(x / d)`. One thread per block
walks its 32 elements serially; `n_elements` must be a multiple of 32. Because Q8_0's
34-byte block stride is not u32-aligned, byte writes into the packed U32 output buffer are
done via an `InterlockedXor` read-modify-write so a thread writing a sub-word does not race
a neighbor sharing the same boundary u32 word — this RMW is **boundary-word safety within
the freshly-allocated output**, not accumulation against a prior tensor. Output is a fresh
contiguous `DType::U32`-typed byte stream (`alloc_device(..., DType::U32)`,
`fuel-vulkan-backend/src/lib.rs:10265`) holding the Q8_0 blob; it does **not** alias the
input (`aliasing: none`). Numerics: lossy quantization — f32→i8 with a per-block f16-narrowed
scale; the `round` and f16 narrow are the loss sources. The per-block max and round are
deterministic (no cross-thread reduction), so it is bit-stable on the same hardware despite
the InterlockedXor (each output byte is written by exactly one block-thread). Perf:
bandwidth-bound — reads `n*4` f32 bytes, writes `n_blocks*34` bytes (~1.0625 B/elem).
Limitation: `n_elements % 32 == 0`; src must be F32 (wrapper bails otherwise, `:10248`);
contiguous-only.

As-built: dispatched by `Capability::QuantizeQ8_0` (`capability.rs:73`); there is **no**
`OpKind::QuantizeQ8_0` (the `op_kind:` slot records the `Capability` token — see header
note). The output is the opaque packed Q8_0 block byte stream — declared as the FDX honesty
stand-in `U8` (FDX §3), though the as-built wrapper allocates the buffer `DType::U32`-typed
(`alloc_device(..., DType::U32)`) and writes it via a U32 ByteAddressBuffer (the internal access
width rides `access_granularity_bits`, not the operand dtype). It is `family: GGML_BLOCK`
(`ggml_dtype: Q8_0`), not an AFFINE_FLOAT-style quant: the per-block f16 scale `d = max|x|/127`
is **INLINE baked** into each block, so the block grain rides `ggml_dtype` (no `granularity`,
no separate scale operand). Computing `d` dynamically per block at quantize time is the kernel's
behavior; the resulting tensor is still a GGML_BLOCK baked-scale block stream (single-place rule)
— the output operand declares the GGML_BLOCK `Q8_0` form it produces.

```fkc
kernel: quantize_q8_0
registrable: false               # §3.10 describe-only: QuantizeQ8_0 is a Capability token, not a real OpKind; documented, not registered. op_kind below is the forward-looking dispatch-tag marker.
op_kind: QuantizeQ8_0            # as-built Capability tag (no OpKind variant; see header note)
blurb: "Quantize dense f32 to a GGML Q8_0 block stream (34B/32-elem; d = max|x|/127)."
backend: Vulkan
kernel_source: "vulkan-slang"
entry_point: "fuel_vulkan_backend::fkc::quantize_q8_0"
kernel_revision_hash: auto

accept:
  inputs:
    - name: src
      dtypes: [F32]                # dense f32 source
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: 1                      # flat [n_elements]
      shape_constraint: "divisible(n, 32)"   # n_elements multiple of the 32-elem block
  op_params:
    variant: None                  # wrapper packs (n_elements, n_blocks)

return:
  outputs:
    - name: out
      dtype_rule: fixed(U8)        # opaque packed Q8_0 block byte stream (FDX §3 honesty stand-in); written internally via a U32 ByteAddressBuffer (access_granularity_bits)
      shape_rule: from_params(n_blocks)   # n_blocks * 34 bytes
      layout_guarantee: contiguous
      aliasing: none               # fresh alloc_device buffer; InterlockedXor is boundary-word safety, not accumulate
      fdx:
        requires_ext: true         # the U32 output is opaque quant bytes; meaning needs the FDX quant sidecar
        quant:
          family: GGML_BLOCK
          ggml_dtype: Q8_0         # produces GgmlDType code 8 blocks; §3.4 — block grain rides ggml_dtype (INLINE baked per-block f16 d = max|x|/127); GGML_BLOCK carries ggml_dtype ONLY, no granularity (§10.6)
          role: activation         # KV-cache activation compression
          scale_operand: ~         # scale INLINE baked in the produced block (single-place rule, §3.9.3)

caps:
  awkward_layout_strategy: requires_contiguous
  fast_paths: []
  in_place: false                  # output is a fresh buffer; binding RMW is intra-output boundary safety
  alignment_bytes: 16
  access_granularity_bits: 8       # InterlockedXor byte writes into the 34B (non-u32-aligned) block stride

cost:
  provenance: judge_measured       # Judge bootstraps; do not fabricate numbers
  class: cheap_elementwise         # bandwidth-bound; per-block serial max+round
  flops: "2 * n"                   # per-block max scan + per-element round/scale (n = n_elements)
  bytes_moved: "n * 4 + n * 1.0625"   # read 4B/elem f32 + write 34B/32-elem (1.0625 B/elem)
  overhead_ns: ~
  memory: { device_bytes: "n * 1.0625", host_bytes: 0, disk_bytes: 0 }   # Q8_0 output ~1.0625 B/elem

precision:
  bit_stable_on_same_hardware: true   # per-block deterministic; each output byte written by one block-thread
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false                    # quantization error (f32→i8 with f16-narrowed per-block scale) not yet Judge-audited
  notes: "d = max|x|/127 (f16-narrowed); qs = round(x/d). Lossy: f32->i8 + f16 scale narrow. Deterministic (no cross-thread reduction)."

determinism: same_hardware_bitwise
```

# Session prompt — Quantize-as-graph-op (`Op::Quantize`)

## What this session is for

xn idea #4 from the 2026-05-29 architectural review: make
dynamic-quantization quantize/dequantize a **first-class graph
op** instead of an implicit step inside specific Linear layers.

End state: model code emits
`xs.quantize(DType::F8E4M3, ScaleGranularity::PerToken)?`
into the graph; the executor dispatches it like any other op;
quantized tensors flow through subsequent ops via the storage-
variant model the same way GGUF Q4_0 tensors already do.

## Why now

Two pieces are already in place from 2026-05-29 sessions:

1. **`ScaleGranularity` + `ScalePair`** in `fuel-core-types/src/
   quant_scale.rs`. The dispatch-time parameter for which
   per-tensor / per-token / per-channel layout the kernel should
   produce.
2. **Named wrapper types** for static quants (`AwqWeight`,
   `MarlinWeight`, `NF4Weight`) in
   `fuel-cuda-backend/src/baracuda/quant_w4a16.rs`. Show the
   pattern for holding scale buffers alongside data buffers as
   typed handles.

The missing piece is the *graph integration* — letting users emit
quantize/dequantize as Op nodes that the optimizer and executor
schedule like any other op.

## Design

### `Op::Quantize` variant

Add to `fuel-graph/src/lib.rs::Op`:

```rust
/// Dynamic quantization of an input tensor to a narrower dtype
/// with caller-chosen scale granularity. Output is a Fp8Tensor /
/// Int8Tensor / etc named handle that downstream ops can consume
/// via their storage-variant dispatch.
///
/// Static-quant formats (GGUF, AWQ, Marlin, NF4) load
/// pre-quantized weights via dedicated loader paths and do NOT
/// emit this op. Their scale layouts are baked into the binary
/// format; there's no per-call granularity choice.
Quantize {
    to: DType,
    granularity: ScaleGranularity,
},

/// Inverse op — dequantize a quantized tensor back to a standard
/// fp dtype. Used at op boundaries where the next consumer doesn't
/// have a quant-aware dispatch path.
Dequantize {
    to: DType,
},
```

### Storage representation

For F8E4M3 dynamic quant, the output of `Op::Quantize` is a tuple
of `(packed_bytes, scales_f32)` plus metadata (the
`ScaleGranularity`). Two options for storage representation:

**Option A — extend the storage-variant enum** with new arms
that carry scales:
```rust
enum CudaStorageSlice {
    ...
    F8E4M3PerTensor { data: CudaSlice<u8>, scale: CudaSlice<f32> },
    F8E4M3PerToken  { data: CudaSlice<u8>, scales: CudaSlice<f32> },
    F8E4M3PerChannel{ data: CudaSlice<u8>, scales: CudaSlice<f32> },
    ...
}
```
Three new variants per quant dtype × granularity combination.
Permeates dispatch tables. Heavy but consistent with how Fuel
handles GGUF today.

**Option B — single F8E4M3 variant + sidecar scale tensor**
where the scale lives in a separate `Tensor` (or
`Arc<Storage>`) that the consumer reads alongside the data
tensor. Mirrors the xn `Fp8Tensor` named-wrapper approach
already prototyped for `AwqWeight` et al.

Recommendation: **Option B**. Less enum churn; aligns with the
xn-pattern named-wrapper types already in `baracuda/
quant_w4a16.rs`. The graph machinery treats the quantize op as
producing TWO output tensors (data + scales); consumers that
understand the quant format read both. Generic ops that don't
understand the format error at dispatch time (no fallback path
should silently dequantize).

### Backend execution

For F8E4M3 PerToken on CUDA, the backend method is essentially
xn's `quantize_fp8_per_token` (which I already showed in the
earlier audit):

1. Allocate scales buffer (`f32[n_tokens]`).
2. Allocate output buffer (`u8[numel]`).
3. Launch baracuda's
   `dynamic_per_token_scaled_fp8_quant_<dt>_run` (or
   equivalent — verify what's exposed in current baracuda alpha).
4. Wrap both in `Arc<CudaStorageBytes>` and return alongside
   metadata.

baracuda has the kernel families already exposed in
`baracuda/quant_w4a16.rs` and others (FP8 specifically lives in
the `fp8.cu` kernels per the FP8 audit memo from xn). Verify the
current alpha's symbol coverage before designing the wrapper
signature.

### Dispatch surface

Consumers of quantized tensors (e.g. `Op::MatMul` with an
`Fp8Tensor` operand) check the operand's storage variant and the
sidecar scale shape, then call the right `fp8_matmul_*` kernel.
The dispatch lookup is keyed on `(op, lhs_dtype, lhs_granularity,
rhs_dtype, rhs_granularity)` — the `ScalePair` type from
`fuel-core-types` is the natural key.

## What this session unlocks

- FP8 inference path end-to-end via lazy graph (today FP8 is
  exposed at the baracuda level but doesn't have a graph-side
  integration).
- Symmetric path for Int8 dynamic quant when that family lands.
- A worked example of "quant format as graph op" that future
  quant integrations can mirror.

## What NOT to do

- Don't add `Op::Quantize` for GGUF / AWQ / Marlin / NF4. Those
  formats are loaded pre-quantized; quantization is the
  loader's job, not a runtime graph op.
- Don't conflate quantize (dtype narrowing + scales) with cast
  (dtype change without scales). `Op::Cast` already exists for
  the latter.
- Don't expose a "dequantize on read" fallback path. If a consumer
  doesn't understand the quant format, it should error at
  dispatch time, not silently materialize a wide-dtype copy.

## Test scope

- Quantize → dequantize round-trip per granularity per dtype.
  Max error within FP8's documented bound (~3% relative for
  PerTensor on activations; <1% for PerToken).
- `Op::MatMul` consuming an F8E4M3 PerToken activation +
  F8E4M3 PerChannel weight produces the same numerical result
  (within tolerance) as the BF16 reference.
- Graph rewrite: emitting `quantize → matmul → dequantize`
  produces correct output; optimizer can fuse the dequantize away
  when the next consumer is also quant-aware.

## Scope realism

1-2 sessions. The Op IR additions + dispatch wiring + backend
kernel integration are mechanical given the primitives already
land in baracuda. The interesting design choice is Option A vs B
(captured above). Recommend deciding that explicitly at the start
of the implementation session.

Link:
[`fuel-core-types/src/quant_scale.rs`](../../fuel-core-types/src/quant_scale.rs),
[`fuel-cuda-backend/src/baracuda/quant_w4a16.rs`](../../fuel-cuda-backend/src/baracuda/quant_w4a16.rs).

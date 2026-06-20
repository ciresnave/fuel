# Session prompt — Quantize-as-graph-op (`Op::Quantize`)

> **Reconciled 2026-06-20 (adaptive-fusion decision, [`../architecture/10-decisions-log.md`](../architecture/10-decisions-log.md) G1/G2 + constitution build-time-validation rule):** two corrections to the design below. (1) **Recipe principle (G1/G2):** `Op::Quantize` / `Op::Dequantize` must ship a **total, never-panic `decompose`** plus a `pattern` (see §Recipe below) — they cannot be opaque islands the way the nf4-class fused ops currently are (`nf4_matmul.rs:120` panicking `decompose` is the anti-pattern to avoid, not copy). (2) **Validate at build time:** the consumer-format-mismatch check below was specified to error *at dispatch time*; per the constitution's build-time-validation rule it is **build-time-checkable** (operand dtype/granularity and consumer op are known when the graph is built) and **must move to graph-build time**. Both corrections are folded into the sections below.

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

### Recipe (mandatory both directions — G1/G2)

Per the recipe principle ([`../architecture/10-decisions-log.md`](../architecture/10-decisions-log.md) G1/G2;
[04-optimization §`decompose` is total](../architecture/04-optimization.md)), both `Op::Quantize` and
`Op::Dequantize` ship a recipe in **two inverse directions, both required**:

- **`decompose` (total + never-panic + primitive→self).** Each lowers onto the existing primitive basis:
  `Quantize` = compute the scale per `ScaleGranularity` (a reduce-max / divide over the chosen axis) →
  scale → round → clamp → cast-to-narrow; `Dequantize` = cast-to-wide → multiply by the broadcast scale.
  These are exact primitive sequences (the *math* definition); the native FP8 kernel is the faster
  numerically-close *implementation* governed by the FKC `precision` tolerance, not a substitute for the
  recipe. The `decompose` **never `panic!`s** — the base map is the fixpoint of `decompose`, so a panicking
  or absent `decompose` here would break the optimizer (optimization = lower-to-base-map + find-best-cover),
  not just a downstream feature. **Do not copy the nf4-class panicking `decompose`** (`nf4_matmul.rs:120`);
  that is a known bug to fix, not the pattern.
- **`pattern`.** Recognize the primitive scale→round→clamp→cast (resp. cast→mul) subgraph and re-fuse it back
  to `Op::Quantize` / `Op::Dequantize`, so an imported or hand-written quant chain collapses to the fused op
  and the missing-fusion telemetry can see across it. A `decompose`-only entry would be an opaque island.

The recipe is what lets the optimizer reason about the quant chain (e.g. the "fuse the dequantize away when
the next consumer is also quant-aware" rewrite in §Test scope) instead of treating these ops as black boxes.

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
understand the format **error at graph-build time** (no fallback
path should silently dequantize). Per the constitution's
build-time-validation rule, this consumer-format-mismatch check is
**not** deferred to dispatch — the operand's quant dtype +
granularity and the consuming op are both known when the edge is
added to the graph, so the check that "this consumer has a
quant-aware dispatch path for this `(dtype, granularity)`" runs as
a `Result`-returning build-time validation, surfacing the mismatch
before any kernel is scheduled.

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
`fuel-core-types` is the natural key. The *existence* of a
quant-aware path for that key is validated at **graph-build time**
(per the build-time-validation rule above); dispatch only *selects*
among paths already known to exist, never discovers a missing one.

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
  **graph-build time** (the operand dtype/granularity and the
  consumer op are known then — validate at build time, per the
  constitution), not silently materialize a wide-dtype copy. Don't
  defer this build-time-checkable mismatch to dispatch.

## Test scope

- Quantize → dequantize round-trip per granularity per dtype.
  Max error within FP8's documented bound (~3% relative for
  PerTensor on activations; <1% for PerToken).
- `Op::MatMul` consuming an F8E4M3 PerToken activation +
  F8E4M3 PerChannel weight produces the same numerical result
  (within tolerance) as the BF16 reference.
- Graph rewrite: emitting `quantize → matmul → dequantize`
  produces correct output; optimizer can fuse the dequantize away
  when the next consumer is also quant-aware (this rewrite reads the
  `pattern` half of the recipe — see §Recipe).
- Recipe round-trip: `Op::Quantize`'s `decompose` (scale → round →
  clamp → cast) lowers to primitives and matches the native kernel
  within the FKC tolerance; the `pattern` re-fuses that primitive
  subgraph back to `Op::Quantize`. Assert neither `decompose`
  panics (G2: total + never-panic).
- Build-time validation: wiring an `Fp8Tensor` into a consumer with
  no quant-aware path for its `(dtype, granularity)` returns a
  `Result` error at graph-build time, not at dispatch.

## Scope realism

1-2 sessions. The Op IR additions + dispatch wiring + backend
kernel integration are mechanical given the primitives already
land in baracuda. The interesting design choice is Option A vs B
(captured above). Recommend deciding that explicitly at the start
of the implementation session.

Link:
[`fuel-core-types/src/quant_scale.rs`](../../fuel-core-types/src/quant_scale.rs),
[`fuel-cuda-backend/src/baracuda/quant_w4a16.rs`](../../fuel-cuda-backend/src/baracuda/quant_w4a16.rs).

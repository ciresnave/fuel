# Phase C — Rotating KV cache (resumption notes)

Phase C of the eager-Tensor retirement program. Paused 2026-06-01 because
the parallel multi-output (Option C) session was still in flight in the
same files (`fuel-graph/src/lib.rs`, `fuel-dispatch/src/pipelined.rs`,
`fuel-dispatch/src/dispatch.rs`, `fuel-dispatch/src/kernel.rs`,
`fuel-cpu-backend/src/byte_kernels.rs`); the workspace doesn't build at
HEAD until that session lands. Pick this up after the multi-output
session is committed.

## What's locked from the master plan

- New `Op::WriteSliceRotating` variant (NOT eager-fallback for
  sliding-window models). Locked by user.
- CPU + CUDA + Vulkan dispatch. Locked by user 2026-06-01.
- Reference + CPU dispatch + LazyKvCache integration test for the
  validation slice. Locked by user 2026-06-01.

## Design — Op variant

```rust
/// Like Op::WriteSlice but the `axis` axis wraps modulo `modulus`.
/// inputs[0] = destination buffer (rotating ring)
/// inputs[1] = source slab
/// inputs[2] = dynamic write position, rank-0 U32
///
/// Position is wrapped (`position % modulus`) inside the kernel.
/// ranges[axis].0 is ignored (dynamic start); ranges[axis].1 - .0
/// is the write length on axis (must equal source.dims()[axis],
/// must not exceed modulus).
WriteSliceRotating {
    axis: usize,
    modulus: usize,
    ranges: Vec<(usize, usize)>,
},
```

Destructive on inputs[0] (same as Op::WriteSlice). Non-differentiable;
backward panics (KV cache writes are forward-only). `op_short_name`
arm: `"WriteSliceRotating"`. `destructive_input` arm: `Some(0)`.

## Design — builder

`Tensor::write_slice_rotating(source, position, axis, modulus, ranges)`
in `fuel-graph/src/lib.rs`, returning `Result<Tensor>`. Validation:

- ranges.len() == dest.rank() == source.rank()
- axis < rank
- modulus >= 1 and <= dest_dims[axis]
- For axis: slab > 0, slab <= modulus, source.dims()[axis] == slab,
  ranges[axis].1 <= modulus
- For other axes: end <= dest_dims[i], source.dims()[i] == end - start
- dtype parity (source == self)
- position dtype == U32, position shape == [] (rank 0)

`LazyTensor::write_slice_rotating(...)` wrapper in
`fuel-core/src/lazy.rs` matching the `LazyTensor::write_slice` shape.

## Design — dispatch

- `OpKind::WriteSliceRotating` in `fuel-core-types/src/dispatch.rs`,
  display string `"write_slice_rotating"`.
- `OpParams::WriteSliceRotating { dest_shape, axis, modulus, ranges }`
  in `fuel-dispatch/src/kernel.rs`.
- `WorkItemKind::WriteSliceRotating { dest, source, position }` in
  `fuel-dispatch/src/pipelined.rs`.
- Compile-side: extend `op_to_op_kind`, `build_lookup_dtypes` (canonicalize
  to `[T_src, T_out]` like WriteSlice), `op_to_op_params`, and add a
  dispatch arm in `compile_one` after `Op::WriteSlice`. Also add
  `Op::WriteSliceRotating { .. }` to the in-place-op-exclusion `matches!`
  at the top of `compile_one`.
- Execute-side: `WorkItemKind::WriteSliceRotating` arm in
  `execute_work_item` modeled on the existing `WorkItemKind::WriteSlice`
  arm. Kernel inputs are `[source, position]`; output is `dest` (in-place).
  Layout side-channel is `[source_layout, position_layout, dest_layout]`.
- Cost: `cost_shape_op_cpu` (same family as WriteSlice).
- Backward stub: panic with "non-differentiable" message in
  `fuel-graph/src/lib.rs` backward arm.

## Design — CPU kernel

`fuel_cpu_backend::byte_kernels::write_slice_rotating_cpu(source, position_bytes, dest, dest_shape, axis, modulus, ranges, dtype_size)`:

1. Read `position` from the first u32 of `position_bytes.bytes()`.
   Compute `wrapped_start = position % modulus`.
2. Compute `slab_axis = ranges[axis].1 - ranges[axis].0`.
3. `first_len = min(slab_axis, modulus - wrapped_start)`.
4. `second_len = slab_axis - first_len`.
5. v1 constraint: `axis == 0` (leading axis). This makes the source-side
   byte split trivial (prefix/suffix); strided multi-axis splits are a
   follow-up. Mistral / Phi-3 sliding-window caches store K/V as
   `[seq, n_kv_heads, head_dim]` per-layer with seq as axis 0.
6. Issue two `write_slice_cpu` calls: one for the first chunk at
   `(wrapped_start, wrapped_start + first_len)`, one for the wrap-around
   chunk at `(0, second_len)` if `second_len > 0`. The source bytes are
   split prefix/suffix on byte length matching `first_len * inner_per_row_bytes`.

## Wrapper registration

`fuel-dispatch/src/dispatch.rs`:

```rust
fn write_slice_rotating_cpu_wrapper(inputs, outputs, _layouts, params) -> Result<()> {
    // inputs = [source, position], outputs = [dest]
    // ... reads params via OpParams::WriteSliceRotating, calls byte kernel.
}

table.register(WriteSliceRotating, &unary(f32_dt), cpu, write_slice_rotating_cpu_wrapper);
// f64/bf16/f16/u32/u8 same.
```

Per-binding caps: default (no `strided_input`). Same surface as WriteSlice.

## CUDA + Vulkan (deferred until CPU lands)

- **CUDA**: probably hand-write the kernel (baracuda doesn't have rotating
  scatter today). Two-chunk write same as CPU; launch geometry is one
  thread per byte in the contiguous span. Bypass through baracuda would
  require new kernel ask — defer.
- **Vulkan**: Slang kernel mirroring the CPU two-chunk pattern. Position
  read from a U32 storage buffer; modulo+wrap in the shader.

## LazyKvCache::append_rotating

`fuel-core/src/lazy_kv_cache.rs`: new variant `LazyKvCache::rotating(...)`
constructor + `append_rotating(layer, k, v, position) -> Result<Self>`
that emits `Op::WriteSliceRotating` on the per-layer K/V buffers. Stores
`window_size` (== modulus) on the cache; position is a graph node the
caller supplies (typically the monotonic logical step counter, materialized
as a U32 const). v1 functional API per existing LazyKvCache pattern.

## Tests

- Reference test: build a tiny ring + write past modulus; expected
  output matches a hand-written rotating buffer.
- CPU oracle: `Op::WriteSliceRotating` matches `write_slice_rotating_cpu`
  byte-for-byte.
- Sequence boundary: position == 0, position == modulus-1, position ==
  modulus, position == 2*modulus, position == 2*modulus+slab_len-1.
- LazyKvCache integration: a Mistral-style 4-step decode loop where
  steps 1-3 fit in the window, step 4 overwrites slot 0; the resulting
  cache slab is exactly `[step1, step2, step3, step4]` rotated.
- CUDA + Vulkan oracle gates: same shapes through the live-GPU dispatch.

## Surface-area files

- `fuel-graph/src/lib.rs` — Op variant + builder + backward panic +
  op_short_name + destructive_input
- `fuel-core-types/src/dispatch.rs` — OpKind + display string
- `fuel-dispatch/src/kernel.rs` — OpParams variant
- `fuel-dispatch/src/cost.rs` — cost arm
- `fuel-dispatch/src/pipelined.rs` — WorkItemKind + compile arm +
  execute arm + op_to_op_kind + build_lookup_dtypes + in-place
  exclusion + op_to_op_params
- `fuel-dispatch/src/dispatch.rs` — CPU wrapper + binding-table
  registrations
- `fuel-cpu-backend/src/byte_kernels.rs` — `write_slice_rotating_cpu`
- `fuel-reference-backend/src/exec.rs` — unreachable arm
- `fuel-graph-cpu/src/lib.rs` — unreachable arm
- `fuel-core/src/lazy.rs` — `LazyTensor::write_slice_rotating` wrapper
- `fuel-core/src/lazy_kv_cache.rs` — `LazyKvCache::rotating` constructor +
  `append_rotating`
- New: CUDA + Vulkan dispatch + Slang kernel (deferred until CPU lands)
- New: integration test (`fuel-core/tests/phase_c_rotating_kv.rs`)

## Open questions

1. Should position be read on host (executor extracts u32) or stay
   device-side (kernel reads from buffer)? v1 plan: device-side (kernel
   reads from inputs[1]'s first u32). Simplifies cross-backend dispatch;
   one D2H roundtrip per token is the cost.
2. v1 restriction: rotating axis must be axis 0. Mistral/Phi-3 caches
   store seq as the leading dim per layer, so this is fine. Multi-axis
   strided splits are follow-up.
3. Vulkan kernel: separate Slang or generalize the existing
   `write_slice_b*` kernels with a `modulus` param. Recommend separate
   kernel to keep the existing WriteSlice path zero-overhead.

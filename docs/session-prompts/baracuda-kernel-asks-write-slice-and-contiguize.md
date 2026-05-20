# Baracuda kernel asks — WriteSlice + Contiguize

Two CUDA kernels that fuel needs from the next baracuda release. Both
have CPU equivalents already shipped in fuel; both currently have no
native CUDA path. The first is blocking the persistent KV-cache
integration (Phase 7.6 step 9c E.3.3+); the second is a long-standing
slow-path TODO in fuel-storage's pipelined executor.

If both can land in one baracuda release, fuel's CUDA inference path
closes the remaining "must round-trip to host" gaps and unlocks the
final stretch of the legacy-executor retirement.

---

## 1. WriteSlice — rectangular slab assignment

### Motivation

Persistent KV-cache writes during autoregressive decoding. The
destination is a pre-allocated `[max_seq_len, n_kv_heads, head_dim]`
buffer that lives across forward passes; on each new token the model
writes its `[1, n_kv_heads, head_dim]` K (and V) slab into the row at
position `cached_len`, then advances `cached_len`. This pattern
generalizes to "copy a small dense tensor into a rectangular
sub-region of a larger dense tensor."

Today fuel handles this via a CPU `write_slice_cpu` byte-kernel
(shipped in commit `89611528`). The CUDA dispatch arm returns a
typed `"no binding for OpKind::WriteSlice on Cuda"` error pending a
native kernel — we deliberately avoided a D2H→CPU→H2D fallback
because that contradicts the pipelined executor's fail-fast dispatch
commitment and would hide perf cliffs.

### Op shape

```text
write_slice(
    dest:        Tensor<T> of shape [D_0, D_1, …, D_{R-1}],     // mutated in place
    source:      Tensor<T> of shape [S_0, S_1, …, S_{R-1}],
    ranges:      [(start_0, end_0), …, (start_{R-1}, end_{R-1})],
) -> dest                                                       // same Storage, post-write
```

Per-axis contract:

- `S_i == end_i - start_i` (source's shape along axis `i` equals the
  slab width along that axis).
- `0 <= start_i <= end_i <= D_i`.
- `dest` and `source` share dtype.
- `dest` is contiguous + zero-offset (fuel rejects strided destinations
  at compile time; only the source may need a contiguize beforehand).
- `source` is contiguous + zero-offset at the kernel surface (fuel
  auto-contiguizes upstream if not).

After the op runs:

- Bytes inside the slab `dest[start_0..end_0, …, start_{R-1}..end_{R-1}]`
  equal `source`.
- Bytes outside the slab are untouched.
- No accumulation (this is the key distinction from baracuda's existing
  `ScatterAddPlan` — WriteSlice ASSIGNS, doesn't add).

### Dtype coverage (priority order)

1. **`f16`, `bf16`, `f32`** — KV cache writes during inference; the
   first batch of dtypes that need to be live for E.3.3.
2. **`f64`** — completeness; numerical tests + dtype-agnostic
   compositions.
3. **`u32`, `u8`** — for future index-table / mask scatter use cases
   (not blocking E.3.3).
4. **`i8` / `i32`** — optional; would round out integer coverage.

The CPU equivalent is **dtype-agnostic at the byte level** (one
`copy_from_slice` per innermost row, no arithmetic). A CUDA equivalent
can do the same — there's no FP math, so the kernel could be a single
templated `T memcpy` parameterized only by `sizeof(T)`.

### Rank coverage

- Rank 1 through ~6 covers every case fuel will hit.
- KV cache append uses rank 3 (`[max_len, H, D]` ← `[1, H, D]`).
- A general bank could go to rank 8 to match other baracuda
  primitives like `IndexSelectPlan` / `ScatterAddPlan`.

### Suggested kernel surface

Mirror `ScatterAddPlan<T, N>`'s shape but with assign semantics. One
candidate signature:

```rust
pub struct WriteSlicePlan<T, const N: usize> { /* … */ }
pub fn baracuda_kernels_write_slice_T_run(
    dest_ptr:          CUdeviceptr,        // out (in-place mutation)
    source_ptr:        CUdeviceptr,        // in (contiguous, slab-shaped)
    dest_shape:        *const u32,         // length N
    range_starts:      *const u32,         // length N — per-axis start
    range_ends:        *const u32,         // length N — per-axis end (exclusive)
    stream:            CUstream,
) -> CUresult
```

The kernel walks every coordinate in the slab (product of `end_i -
start_i`); for each coordinate it computes the linear offset in `dest`
(using `dest_shape` to derive strides + `range_starts` to offset each
axis) and the linear offset in `source` (slab strides). A single
write per thread. No reductions, no atomics, no shared memory.

A natural optimization: when all axes except the outermost are
"full-width" (`ranges[i] == (0, D_i)` for `i > 0`), the entire source
is one contiguous chunk inside dest's memory and the kernel
degenerates to a single `cuMemcpyDtoDAsync` of `source_bytes` at
offset `start_0 * stride_0 * sizeof(T)`. This is the KV-cache append
case — the most performance-critical shape.

### Reference: fuel's CPU implementation

`fuel-cpu-backend/src/byte_kernels.rs::write_slice_cpu` (~120 LoC).
Dtype-agnostic; takes `dest_shape: &[usize]`, `ranges: &[(usize,
usize)]`, `dtype_size: usize`. Tests at the same file cover the KV
append shape, interior 2-D slabs, 1-D slabs, dtype-agnostic byte
correctness for f64, and rejection paths.

Determinism: bit-stable (no atomic accumulation; each output byte is
written exactly once by exactly one thread).

### Backward

Non-differentiable per fuel's autograd model. fuel's IR layer
(`Op::WriteSlice` in fuel-graph) panics in `backward()` and points
callers to `Gather + IndexAdd` as the differentiable analogue.
Baracuda need not ship a backward kernel.

---

## 2. Contiguize — strided→contiguous copy

### Motivation

After a metadata-only view op (`Transpose`, `Permute`, `BroadcastTo`,
`Slice` with non-trivial strides), the resulting tensor's bytes are
laid out in a non-contiguous fashion. Today's CUDA kernels assume
contiguous inputs, so fuel inserts an auto-contiguize step at every
kernel call site whose input is non-contiguous.

The CPU implementation is a one-axis-at-a-time multi-index walk that
copies elements one-by-one (or innermost-row at a time when the
innermost axis is contiguous-stride).

The CUDA implementation today is a **slow D2H→CPU contiguize→H2D**
fallback (see `fuel-storage/src/pipelined.rs::auto_contiguize`,
line ~2117). Two device round-trips per non-contiguous input.

A native CUDA contiguize would eliminate the cliff. Every existing
view-op path (BroadcastTo for binary broadcasting, Transpose for the
K/V transpose patterns in attention, Slice when reading KV cache
prefixes) currently materializes via this slow path when run on CUDA
inputs.

### Op shape

```text
contiguize(
    source:        Tensor<T> with arbitrary strides + offset,
    source_layout: Layout { shape: [D_0, …, D_{R-1}], strides: [S_0, …, S_{R-1}], offset },
) -> Tensor<T> of shape [D_0, …, D_{R-1}], contiguous, zero offset
```

Per-axis contract:

- Output shape equals source's logical shape.
- Output strides are the canonical row-major strides for that shape.
- Output offset is 0.
- For each multi-index `(i_0, …, i_{R-1})`, `output[i_0, …, i_{R-1}] ==
  source[i_0 * S_0 + … + i_{R-1} * S_{R-1} + offset]` (treating
  source as a flat byte buffer indexed by linear-element offset).
- Strides may be **negative** (fuel's `Flip` op produces negative
  strides). The byte offset is still `start_offset + Σ i_k * S_k`
  with signed arithmetic.
- Strides may be **zero** (broadcast axes). A zero-stride axis means
  the same source bytes are read for every output coordinate along
  that axis — this is where the "expand" / "broadcast_to" view's
  duplication materializes.

### Dtype coverage

Dtype-agnostic at the byte level (it's a pure memcpy pattern). A
single kernel templated only on `sizeof(T)` covers every dtype fuel
might hold:

1. **`f16`, `bf16`, `f32`, `f64`** — first batch.
2. **`u8`, `u32`, `i8`, `i32`, `i64`** — completeness.
3. Custom-width dtypes (`F4`, `F6E2M3`, `F6E3M2`, `F8E4M3`,
   `F8E8M0`) — only if their byte size is a multiple of a supported
   width. Otherwise fuel will compose contiguize at a wider dtype.

### Suggested kernel surface

```rust
pub fn baracuda_kernels_contiguize_T_run(
    dest_ptr:          CUdeviceptr,        // out (newly-allocated, contiguous)
    source_ptr:        CUdeviceptr,        // in (offset already applied by caller)
    shape:             *const u32,         // length R
    source_strides:    *const i64,         // length R — signed (Flip support)
    rank:              u32,
    stream:            CUstream,
) -> CUresult
```

The kernel walks every output element (product of `shape`). For each
output linear index, decompose into multi-index, dot with
`source_strides` to derive the source linear index, copy the
`sizeof(T)`-byte element. One read + one write per thread, no
synchronization needed.

A natural optimization: detect "innermost dim has stride 1" at host
time and replace the inner loop with a `cuMemcpyDtoDAsync` of
`shape[-1] * sizeof(T)` bytes per outer coordinate. Halves
instruction count for the common case.

Even simpler optimization: detect "source is already contiguous +
zero offset" at host time and just `cuMemcpyDtoDAsync` the whole
buffer. fuel already shortcircuits this before calling the
contiguize wrapper, but defensive coverage doesn't hurt.

### Rank coverage

Up to 8 dims to match `IndexSelectPlan` / `ScatterAddPlan`.

### Reference: fuel's CPU implementation

`fuel-cpu-backend/src/byte_kernels.rs::contiguize_cpu`. Dtype-agnostic
byte-level walk; takes `Layout` + `dtype_size: usize`. Handles
negative strides + zero strides correctly. Well-tested through every
view op's E2E test sweep.

### Backward

Non-differentiable in the sense that there's no gradient to
materialize — the strided→contiguous transition is invisible to
autograd (it's metadata-only). Baracuda need not ship a backward
kernel.

---

## Other CUDA gaps (lower priority, not blocking E.3.3)

Surfaced for the team's awareness; only ship if scheduling allows.

- **Flip / Roll / CumSum / Triu / Tril** — currently CPU-only. Low
  inference impact (mostly training / dataset prep). Skip unless
  cheap.
- **Pad / PadBackward** — currently CPU-only. Used by some image
  models in fuel-transformers; not on the critical path for LLM
  inference.
- **Cast for the F8 / F4 / F6 dtypes** — fuel has byte-level
  representations for sub-byte custom dtypes but no CUDA-resident
  cast paths. Only relevant when serving Q4_K_M-style models with
  fused dequant; the existing GGUF MMVQ kernels handle the common
  paths.

None of these are blocking — they're listed in case the baracuda team
is already in the area for WriteSlice / Contiguize and wants to bundle.

---

## fuel-side integration plan once kernels land

For each kernel:

1. Add a thin `fuel-cuda-backend/src/baracuda/<kernel>.rs` integration
   following the existing pattern (e.g. `concat.rs`, `gemm_int.rs`).
2. Register dispatch wrappers in
   `fuel-storage/src/baracuda_dispatch.rs` against the
   `(OpKind::WriteSlice, [T_src, T_out], BackendId::Cuda)` and
   `(OpKind::Cast, [T, T], BackendId::Cuda)` keys (Contiguize routes
   through Cast-to-self today; we may add a dedicated `OpKind`).
3. Drop the D2H/CPU/H2D fallback in
   `fuel-storage/src/pipelined.rs::auto_contiguize` and replace with
   a direct `BackendStorage::Cuda(_)` arm that calls the new kernel.
4. Lift the CUDA-WriteSlice guard in `compile_one` (currently surfaces
   `"no binding for OpKind::WriteSlice on Cuda"`).

Estimated fuel-side integration time once baracuda ships: 2-4 hours
per kernel, all isolated to the integration files above. The
`cuda_dispatch_live` sweep + 2 new live-CUDA tests per kernel exercise
the integration.

---

## Reference commits (fuel-side)

- `77ff8fbf` — `Op::WriteSlice` IR variant
- `838393de` — OpKind/OpParams + executor dispatch
- `89611528` — CPU WriteSlice kernel + 8 tests (2 E2E)
- `a405e7c0` — `InferenceContext` + `KvCache` (the consumer)
- `8b6b...` (this commit) — CUDA WriteSlice + Contiguize spec doc.

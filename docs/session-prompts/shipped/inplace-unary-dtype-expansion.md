# Session prompt — In-place unary dtype expansion (bf16 / f16 / f64)

## What this session is for

Extend the 5 in-place unary activations (Relu / Silu / Gelu / Tanh /
Sigmoid) shipped 2026-05-30 from f32-only to also cover bf16, f16,
and f64 on CPU. CUDA bf16/f16/f64 coverage is contingent on baracuda
exposing the relevant per-dtype symbols (verify during the session).

Per `feedback_no_consumer_not_a_reason`: ship the capability before
consumers materialize. The phase 1-5 work deliberately scoped down
to f32 unary for the initial integration; this session closes the
gap for the obvious dtype expansion.

INPLACE_AFFINE (the fused op) already covers f32+f64+bf16+f16 on
CPU but only f32+f64 on CUDA — that's a separate baracuda dependency
(needs `affine_inplace_{bf16,f16}_run` symbols which don't exist
today in alpha.60). Out of scope here; track separately.

## Scope

### CPU — straightforward, ~30 lines

The existing `unary_inplace_thunk!` macro in
`fuel-cpu-backend/src/byte_kernels.rs` already generalizes via the
chassis. Adding bf16/f16/f64 is one line per (variant × dtype) = 15
new kernels:

```rust
// f64 — direct, the chassis has f64 impls for Relu/Silu/Gelu/Tanh/Sigmoid
unary_inplace_thunk!(relu_inplace_f64,    f64, Relu,     f64);
unary_inplace_thunk!(silu_inplace_f64,    f64, Silu,     f64);
unary_inplace_thunk!(gelu_inplace_f64,    f64, GeluTanh, f64);
unary_inplace_thunk!(tanh_inplace_f64,    f64, Tanh,     f64);
unary_inplace_thunk!(sigmoid_inplace_f64, f64, Sigmoid,  f64);

// bf16/f16 — chassis's blanket impls route through f32 pivot.
// Verify the chassis has those blanket impls; if not, add them
// (one line per dtype per op via the via-f32 round-trip pattern
// the affine_inplace kernels use).
unary_inplace_thunk!(relu_inplace_bf16,    half::bf16, Relu,     bf16);
// ... × 5 ops × 2 half-precision dtypes
```

Then in `fuel-storage/src/dispatch.rs`, add the wrapper +
registration for each:

```rust
cpu_unary_inplace_wrapper!(relu_inplace_f64_cpu_wrapper, relu_inplace_f64);
// ... × 15

table.register(ReluInplace, &unary(f64_dt),  cpu, relu_inplace_f64_cpu_wrapper);
table.register(ReluInplace, &unary(bf16_dt), cpu, relu_inplace_bf16_cpu_wrapper);
table.register(ReluInplace, &unary(f16_dt),  cpu, relu_inplace_f16_cpu_wrapper);
// ... × 5 ops
```

### CUDA — verify baracuda surface, then add

The existing `unary_inplace_kernel!` macro in
`fuel-cuda-backend/src/baracuda/elementwise.rs` reuses baracuda's
forward-only `unary_*_run` symbols with same-pointer dispatch
(safe for elementwise unary). For each (op × dtype) pair, the
session needs to verify:

```bash
grep "fn baracuda_kernels_unary_relu_bf16_run" ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/baracuda-kernels-sys-0.0.1-alpha.60/src/lib.rs
# ... × each (op × dtype)
```

If the symbol exists, one line:

```rust
unary_inplace_kernel!(unary_inplace_relu_bf16, unary_relu_bf16, 2, "unary_inplace_relu_bf16");
```

Plus a `cuda_unary_inplace_baracuda_wrapper!` invocation and a
`table.register(ReluInplace, &u(bf16), cuda, unary::relu_inplace_bf16);`
entry in `fuel-storage/src/baracuda_dispatch.rs`.

If the symbol does NOT exist, document the gap in the commit
message and ship CPU-only for that (op, dtype) pair. Don't open a
baracuda ask in this session — track separately if multiple gaps
surface.

### Tests

- 5 new unit tests in fuel-graph or fuel-cpu-backend covering one
  representative dtype per non-f32 path (e.g.,
  `relu_inplace_bf16_round_trip`, `silu_inplace_f64_matches_silu_functional`).
- 1 live CUDA test per dtype that has baracuda coverage, mirroring
  `fuel-storage/tests/baracuda_unary_inplace_live.rs` (just bump
  dtype + parametrize).
- Regression sweep: fuel-storage + fuel-cpu-backend + fuel-graph
  lib tests; live CUDA `--ignored` for any added dtypes.

## What's NOT in scope

- **bf16/f16 INPLACE_AFFINE on CUDA**. baracuda alpha.60 ships
  `affine_inplace_{f32,f64}_run` only. Adding bf16/f16 requires
  either a baracuda symbol ask or a Fuel-side Cast→Affine→Cast
  composition (defeats in-place semantics). Defer until a consumer
  needs it or baracuda ships the variants.
- **View-mediated cycle resolution in `insert_safety_copies`**. Phase
  5 follow-up; not unblocked by this work.
- **PipelinedExecutor ordering integration**. Independent session
  (`pipelined-executor-ordering-integration.md`); the
  dtype-expansion work doesn't depend on it.
- **Bookkeeping in the `inplace_ops_complete` memory entry** — the
  session should update the dtype coverage table at the end.

## Scope estimate

- CPU kernel additions: ~30 min (mostly mechanical via the chassis)
- CPU wrapper + registration boilerplate: ~30 min
- CUDA verification + additions (per-dtype): ~30 min total
- Tests: ~45 min
- Total: 1 focused session, 1 commit (or 2 if CPU and CUDA split
  cleanly).

## Dependencies + references

- 2026-05-30 commits: `2a985c27` (the original f32 Phase 3e
  shipment — pattern to mirror)
- Memory: `project_inplace_ops_complete.md` (coverage table to
  update)
- Architectural framing: `feedback_no_consumer_not_a_reason.md`
  (why this work matters even without a consumer in tree)

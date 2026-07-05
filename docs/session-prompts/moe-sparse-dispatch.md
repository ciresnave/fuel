# MoE sparse dispatch — session prompt (continue here)

**Status (2026-07-04):** the data-determined dynamic-shapes SUBSTRATE + the dropless-MoE
COMPUTE ATOM are shipped and tested on this branch (`feat/data-dependent-shapes`, worktree
`C:/Projects/fuel-ddshapes`). This doc is the handoff for the two remaining MoE increments.
CPU-verifiable throughout — no GPU needed. TDD each (born-red first).

Branch base is `839c3ae5`; `main` has since moved to `42fb5a58` — rebase before the eventual
merge, not required to continue. Build discipline: always `-p <crate>`, never workspace-wide.

## What's already built (the atoms you compose)

- **`Op::NonZeroIndices { count_sym }`** (`677c23af`) — data-determined producer. Bundle output
  (slot 0 = `indices [capacity] U32`, slot 1 = `count [1] U32`); `Tensor::nonzero_indices_bundled`
  / `LazyTensor::nonzero_indices_bundled`. CPU kernel `nonzero_indices_{f32,u32}`.
- **Producer→SymEnv bind (`3dafb8b0`) + consumer resolve-at-execute (`e50ff565`)** — the
  executor publishes a producer's runtime count into a loop-local `produced_syms: SymEnv`
  mid-realize; consumers resolve it at execute (single-threaded topo order guarantees producer
  precedes consumer). Vehicle proven: a data-determined `WriteSlice` offset. Key helpers to
  mirror: `bind_data_determined_count`, `resolve_deferred_write_slice` (fuel-dispatch/src/
  pipelined.rs), `Graph::binds_data_determined_sym` (fuel-graph/src/lib.rs).
- **`Tensor::matmul_dyn_m(rhs, row_count: DynScalar)`** (`f17e0074` + `e19b6973`) — F32-only.
  A matmul that computes exactly `row_count` rows of an `m`-row CAPACITY buffer (the FLOP
  saving). Row count rides a GRAPH SIDE-TABLE (`Graph::node_matmul_row_count`), NOT an
  Op::MatMul field. CPU kernel `matmul_f32_capacity`. Execute-resolve: `resolve_deferred_matmul`.
  Test: `fuel-core/tests/nonzero_indices.rs::nonzero_count_drives_dynamic_m_matmul`.

## STATUS UPDATE (2026-07-04): Increments A + B SHIPPED (commit `e89fa6b8`)

Both remaining MoE increments are done and CPU-verified. The next frontier is
**MLA / KV-cache compression** (see "After MoE" below).

- **Increment A — gather-by-count:** needs NO new op. Plain `index_select(values,
  0, indices)` over the capacity-sized `NonZeroIndices` index buffer gathers the
  routed rows into a `[capacity, hidden]` prefix; `count_sym` threads separately
  to `matmul_dyn_m`. Test `nonzero_indices_gather_by_count_selects_routed_rows`.
- **Increment B — sparse layer rewrite:** `LazyMoeLayer::forward` is now sparse
  (per-expert `NonZeroIndices(gate_col) → index_select gather → forward_dyn_m
  FFN (matmul_dyn_m ×3) → broadcast_mul gate → index_add scatter-back`). The old
  dense path is preserved as `forward_dense` and is the **bit-exact reference**;
  `moe_layer_sparse_forward_bit_exact_to_dense` (+ a 3-D variant) assert it
  bit-for-bit (sabotage-verified). Supporting atoms: `WeightStorage::
  apply_linear_dyn_m`, `LazyMoeExpert::forward_dyn_m` (F32 + bias-free, else typed
  error), `Graph::next_data_determined_sym` / `LazyTensor::
  fresh_data_determined_sym` (graph-scoped fresh count-sym so per-expert
  producers and stacked layers never collide). FLOP note: expert FFN matmuls drop
  from `N·num_experts` to `N·top_k` token-rows.
- **Known follow-ups (not blocking MLA):** experts are F32 + bias-free only
  (matmul_dyn_m is F32-only; a down-bias would contaminate the zeroed tail); the
  elementwise silu/mul still run over the full capacity buffer (only the matmuls
  are count-bounded); GPU wiring of the data-determined path is Part 2.

---

## Increment A — gather-by-count op (DONE — see status above)

Gather routed tokens into a capacity buffer, producing the lhs `matmul_dyn_m` consumes.
- Given a value/token tensor `[N, hidden]` and a data-determined index list (the `indices`
  slot from `NonZeroIndices`, valid prefix = `count`), gather the selected rows into a
  `[capacity, hidden]` buffer (first `count` rows valid; tail unspecified/zeroed).
- The output's leading dim is the same data-determined `count` (`count_sym`) — feed that same
  sym as `matmul_dyn_m`'s `row_count`. (No shape-level Extent needed if you thread `count_sym`
  directly to the matmul builder, mirroring how the tests wire it.)
- Model it on the existing `Op::Gather`/`IndexSelect` end-to-end wiring (Op variant → shape/
  dtype → CPU kernel → op_to_op_params → OpKind → binding). Gather already exists — this may be
  expressible as `index_select` over the `indices[..count]` prefix + a capacity pad, OR a small
  new op. Prefer reusing `index_select` if the count-bounded prefix can be expressed.
- TDD: mask → NonZeroIndices → gather rows at `indices[..count]` → assert the gathered capacity
  buffer's first `count` rows equal the selected input rows.

## Increment B — the sparse layer rewrite

`fuel-core/src/lazy_nn/moe.rs` — today `LazyMoeLayer::forward` computes ALL N expert FFNs on
ALL tokens (dense; ~32× over-compute for 256-expert/top-8). Rewrite to per-expert sparse:
- For each expert `e`: `mask = (router_indices == e)` → `NonZeroIndices(mask)` → `(sel, count_e)`
  (which (token,slot) pairs routed to `e`).
- Gather those tokens' hidden states (Increment A) into a `[capacity, hidden]` buffer.
- Run expert `e`'s SwiGLU FFN using `matmul_dyn_m(..., count_e)` for the three projections
  (compute only `count_e` rows).
- Scale rows by the gate weight, scatter-add back into the `[N, hidden]` output at the token
  positions.
- **Bit-exact to the current dense path** is the acceptance test (add a born-red test asserting
  sparse == dense on a small MoE), plus a note/measurement of the FLOP reduction.
- Keep `WeightStorage`/`LazyLinear` F32 (matmul_dyn_m is F32-only today).

## After MoE

MLA / KV-cache compression (frontier-architecture-gaps.md §2): generalize `LazyKvCache` /
`InferenceContext::KvCache` (both hardwire a symmetric K/V pair) to hold an arbitrary latent
tensor → MLA decode caches only the compressed latent + `k_pe` → weight-absorption. Then Part 2
(BF16 CUDA decode / the 21× fix — GPU-gated).

See also the machine-local memory `data-dependent-shapes.md` for the same map with more detail.

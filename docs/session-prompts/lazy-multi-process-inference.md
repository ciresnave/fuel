# Session prompt — Lazy multi-process inference driver (revive `llama_multiprocess`)

> Reconciled 2026-06-15 against the 2026-06-14 redirection + current git: still an active reservation (ROADMAP designates this prompt as the authoritative recreation source for `llama_multiprocess/{main.rs,model.rs}`); retired-ExecutionPlan language and a moved phase-H link updated.

## What this session is for

Design and ship the lazy substrate that lets
`fuel-examples/examples/_llama_multiprocess_retired` come back to life as a
working `cargo run --example llama_multiprocess --features cuda,nccl,flash-attn`
binary. The original was a tensor-parallel LLaMA driver — N child processes,
one CUDA device each, weights sharded across ranks, an NCCL AllReduce after
every row-parallel linear. It was quarantined in Phase H
([eager-retirement-phase-h-plan.md](shipped/eager-retirement-phase-h-plan.md), commit
`cfcb35cf`) because every load-bearing piece (the eager `LlamaConfig`, the
eager `CustomOp1` AllReduce, the eager `Linear`, the `ShardedSafeTensors`
var-builder) lived in code that retired with `fuel-transformers/models/`.

The `.rs` files in `_llama_multiprocess_retired/` are **deleted** — the
directory exists only as a quarantine marker. The reference for what it
*used* to do is the git history at commit `4ed0c05c`
(`fuel-examples/examples/llama_multiprocess/{main.rs,model.rs}`):

- `main.rs` — `Args::rank` self-fork; rank 0 generates the `ncclUniqueId`
  and broadcasts via a temp file; every rank calls
  `baracuda_nccl::Communicator::init_rank(num_shards, id, rank)`.
- `model.rs` — wraps `fuel_nn::Linear` in `TensorParallelColumnLinear`
  (no comm) and `TensorParallelRowLinear` (`apply_op1_no_bwd(AllReduce { comm })`).
  The `AllReduce` is a custom `CustomOp1` that downcasts to `CudaBackendStorage`
  and calls `baracuda_nccl::all_reduce` with `RedOp::Sum`. Weights load via
  `ShardedSafeTensors::var_builder(&filenames, dtype, &device)` with a per-tensor
  `Shard { dim, rank, world_size }` hint.

None of those crates or types exist in the lazy world. Everything below is
greenfield against `fuel-core::lazy_llama_full::Llama3Model`, `LazyTensor`,
`Op::Fused`, the binding-table dispatch path, and the existing
`fuel-parallel::tensor_parallel` scaffolding
([fuel-parallel/src/tensor_parallel.rs:200-248](../../fuel-parallel/src/tensor_parallel.rs#L200-L248)).

## The one architectural question to answer first

Before any code is written: **does the lazy substrate expose tensor-parallel
forward as a first-class concept (collective ops are `Op` variants and the
graph carries comm dependencies), or does multi-process inference orchestrate
above the lazy substrate (each rank realizes its local subgraph; AllReduce is
a host-side step between realizations)?**

Read [fuel-parallel/src/tensor_parallel.rs](../../fuel-parallel/src/tensor_parallel.rs)
in full and
[fuel-parallel/src/comm.rs](../../fuel-parallel/src/comm.rs#L55-L74) (the
`Communicator` trait with `all_reduce`, `all_gather`, `reduce_scatter`,
`broadcast`, `barrier`). The eager scaffolding sketches the answer already:
`ColumnParallel` is just a wrapped `Linear` with no comm; `RowParallel` calls
`self.comm.all_reduce(&local, ReduceOp::Sum)` *outside* the matmul. So eager
already chose "orchestrate above" — the comm is a host-side call between
already-realized tensors.

### Option I — Comm as graph op (first-class)

> 2026-06-15 note: when Option I is actually picked up, re-evaluate it against
> the 2026-06-14 branch-point / RUN-dispatch model — comm ops fall inside RUNs
> (op-sequences between decision points), so the "executor's ordering pass sees
> the comm dependency natively" pro should be re-read in those terms rather than
> against the old per-node-alternatives / ExecutionPlan framing.

Add `Op::AllReduce { op: ReduceOp }`, `Op::AllGather { dim }`,
`Op::ReduceScatter { op, dim }`. Backend dispatch routes through a per-Comm
binding-table entry that calls `baracuda_nccl::*`. The forward graph for one
TP layer looks like `... → matmul → all_reduce → ...`, and the executor's
ordering pass sees the comm dependency natively.

- **Pro:** parity with how every other op is expressed. The optimizer can
  reorder around comm (e.g. fuse multiple ranks' `all_reduce` into a single
  `reduce_scatter + all_gather`).
- **Pro:** zero host-side glue per layer — the user just writes
  `q.lazy_row_parallel(...)`.
- **Pro:** future tensor-parallel training (backward) has somewhere natural
  to put the gradient comm op.
- **Con:** the comm op needs a `Communicator` handle. Stuffing a comm handle
  into `OpParams` is a *fundamentally* different kind of param than the
  current scalar-only `FusedOpParams` — the param has device-bound runtime
  state. Closest existing analog: nothing.
- **Con:** every backend that doesn't have NCCL (CPU, Vulkan, Metal, AOCL,
  MKL) needs an `Op::AllReduce` dispatch arm — even if it's just
  "single-rank identity passthrough."
- **Con:** topology questions leak into the graph. Two TP groups, one PP
  group → which comm does the `Op::AllReduce` reference? Today's `Op` enum
  has no answer.

### Option II — Comm as host-side step (orchestrate above)

`LazyColumnParallel::forward(x) -> LazyTensor` is just `linear.apply_linear(x)`
(no graph change). `LazyRowParallel::forward(x) -> LazyTensor` is:

```rust
let local = self.linear.apply_linear(x);     // graph op
let local = local.realize()?;                 // realize this slice's result
let merged = self.comm.all_reduce(&local, ReduceOp::Sum)?; // host-side comm
LazyTensor::from_realized(merged, &x.device())   // re-wrap as a leaf for next layer
```

The graph splits into per-comm-boundary realization slices. The
`Communicator` trait already exists in `fuel-parallel::comm`; what's missing
is a `from_realized` / `from_storage` constructor on `LazyTensor` that takes
the comm result and turns it back into a graph-leaf for the next layer's
matmul.

- **Pro:** zero new ops. Zero new dispatch arms. Existing executor unaware.
- **Pro:** matches the eager scaffolding exactly — `RowParallel` already does
  this with `Linear::forward` + `comm.all_reduce`.
- **Pro:** the comm handle lives in `LazyRowParallel`, not in the IR — no
  topology leak.
- **Con:** forces a realization barrier per row-parallel layer. For LLaMA
  that's `num_layers * 2` per token (o_proj + down_proj). The lazy executor
  can't see across the barrier, so it can't fuse projections with the
  AllReduce, can't reorder, can't pipeline comm with compute. NCCL's whole
  performance story relies on overlapping comm with the next layer's
  compute; this design forecloses that.
- **Con:** "realize → re-wrap as leaf" loses the graph context for
  autograd. For inference-only this is fine; if TP training ever lands,
  this design dead-ends.

### Recommended answer: **Option II for v1, Option I as a follow-up**

Reasoning:

- Eager already chose Option II. The migration is mechanical: rebuild
  `LazyColumnParallel` and `LazyRowParallel` against `LazyTensor` instead of
  eager `Tensor`. No new IR. No backend churn.
- Comm/compute overlap (the main argument for I) is a v2 optimization. v1
  goal is "binary runs at all on N GPUs with correct output." Once that
  exists and there are real numbers, I becomes evaluable on its merits.
- The `Op::AllReduce` design has unresolved questions (how to thread a
  `Communicator` through `OpParams`; what no-NCCL backends do; how to model
  groups). Settling those without a working consumer is the wrong order.

If the session author disagrees, the decision is reversible — Option I can
land later without breaking Option II consumers (the host-side path becomes
a fallback when the IR variant isn't beneficial). What matters is that this
session picks one and ships it. **Do not stack uncertainty by leaving the
choice open.**

## Subtasks for this work (each may be its own session)

### Subtask A — `LazyColumnParallel` + `LazyRowParallel` (Option II shape)

Mirror `fuel-parallel/src/tensor_parallel.rs:ColumnParallel` and `RowParallel`
in a new file `fuel-parallel/src/lazy_tensor_parallel.rs`.

**Surface:**

```rust
pub struct LazyColumnParallel {
    /// Local weight shard, shape `[in_features, out_features / world_size]`.
    weight: WeightStorage,
    in_features: usize,
    out_local: usize,
    bias: Option<LazyBias>,
}

impl LazyColumnParallel {
    pub fn new(weight: WeightStorage, in_features: usize, out_local: usize) -> Self;
    pub fn forward(&self, x: &LazyTensor) -> LazyTensor {
        // Pure local: weight.apply_linear(x, in_features, out_local).
        // No comm, no realize.
    }
}

pub struct LazyRowParallel<C: Communicator> {
    weight: WeightStorage,
    in_local: usize,
    out_features: usize,
    bias: Option<LazyBias>,
    comm: Arc<C>,
}

impl<C: Communicator> LazyRowParallel<C> {
    pub fn new(weight: WeightStorage, in_local: usize, out_features: usize, comm: Arc<C>) -> Self;
    pub fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let local = self.weight.apply_linear(x, self.in_local, self.out_features);
        let local_realized = local.realize()?;  // Storage handle
        let merged = self.comm.all_reduce_lazy(&local_realized, ReduceOp::Sum)?;
        Ok(merged)
    }
}
```

The `Communicator` trait today takes the eager `fuel::Tensor`. **It needs an
equivalent that takes a realized storage handle (or a LazyTensor with a
realization barrier built in)** — see Subtask B.

The `WeightStorage::apply_linear` builder already handles F32 / BF16 / Q4_0
on the lazy side
([fuel-core/src/lazy.rs:4526-4570](../../fuel-core/src/lazy.rs#L4526-L4570)),
so the shard's dtype is whatever the caller stored.

### Subtask B — `Communicator` for lazy tensors

Extend the `Communicator` trait — or add a sibling `LazyCommunicator` trait —
that takes and returns `LazyTensor` instead of eager `fuel::Tensor`. Implementation
options:

1. **Realize-inside-comm:** `LazyCommunicator::all_reduce(&self, t: &LazyTensor, op: ReduceOp) -> Result<LazyTensor>`
   calls `t.realize()?` internally, dispatches `baracuda_nccl::all_reduce` on
   the result, and rewraps via `LazyTensor::from_storage(...)`. Cleanest API.
2. **Realize-outside:** `LazyCommunicator::all_reduce(&self, s: &Storage, op: ReduceOp) -> Result<Storage>`
   — caller realizes, calls comm, rewraps. Matches eager more closely but
   pushes the realize/rewrap boilerplate into every `LazyRowParallel`.

Recommend option 1 — the abstraction is the comm op itself, not "comm op
plus boilerplate."

The NCCL implementation lives in `fuel-parallel` (or in a new
`fuel-parallel-nccl` if cyclic deps with baracuda-nccl are an issue) behind a
`nccl` feature, mirroring how `baracuda-nccl` is gated in
`fuel-examples/Cargo.toml:25` and `feature nccl = [..., "dep:baracuda-nccl"]`
at line 89.

The dispatch path is direct — no binding-table entry, no `Op::AllReduce`,
just `Storage → cuda_slice → baracuda_nccl::all_reduce → Storage`. The
`LazyCommunicator` impl is the only place that knows about NCCL.

**`all_gather(dim)` and `reduce_scatter(op, dim)`** follow the same shape.
`all_gather` is needed for column-parallel output gather when the consumer
isn't itself a row-parallel layer; `reduce_scatter` is needed if Subtask C's
"sequence-parallel" follow-up lands.

### Subtask C — Multi-process driver

`fuel-parallel/src/multi_process.rs` — process-group bootstrap. Three pieces:

1. **Self-fork:** rank == None → spawn N child processes with
   `--rank {i}` (or use the env-var convention `RANK=`, `WORLD_SIZE=`,
   `MASTER_ADDR=`, `MASTER_PORT=` that distributed PyTorch uses; pick one
   and document). The eager driver
   ([4ed0c05c:fuel-examples/examples/llama_multiprocess/main.rs:129-141](`git show 4ed0c05c:fuel-examples/examples/llama_multiprocess/main.rs`))
   self-forks via `std::process::Command::new(env::args().next())` —
   acceptable for v1.

2. **`ncclUniqueId` distribution:** rank 0 generates `baracuda_nccl::UniqueId::new()`,
   writes raw bytes to a comm file (temp + rename for atomicity), every
   other rank polls for the file. Equivalent to the eager path at
   `main.rs:144-156`. File-based IPC is fine for v1; network bootstrap is a
   follow-up.

3. **`Communicator::init_rank(world_size, id, rank)` → `Arc<Comm>`** wrapped
   in a `NcclLazyCommunicator { comm: Arc<baracuda_nccl::Communicator> }`
   that impls `LazyCommunicator`.

Output: `pub fn bootstrap_nccl_comm(args: &MultiProcessArgs) -> Result<(usize, usize, Arc<NcclLazyCommunicator>)>`
returning `(rank, world_size, comm)`.

### Subtask D — Sharded weight loading

Each rank reads only its shard out of safetensors. The eager driver used
`ShardedSafeTensors::var_builder` + `vb.get_with_hints((), "weight", Shard { dim, rank, world_size })`
which sliced inside the var-builder
([_fuel_nn_retired/src/var_builder.rs:835-846](../../_fuel_nn_retired/src/var_builder.rs#L835-L846)).
The lazy path has no var-builder — weights load via `load_tensor_as_f32` /
`load_transposed_matrix_preserve_dtype`
([fuel-core/src/lazy.rs:5655-5700](../../fuel-core/src/lazy.rs#L5655-L5700))
directly from a `MmapedSafetensors`.

Add a sharded sibling: `load_transposed_matrix_sharded(st, name, in_total, out_total, dim, rank, world_size) -> Result<WeightStorage>`.
Behavior:

- `dim == 0` (row-parallel, shard along `in_features`): output has shape
  `[in_total / world_size, out_total]`, byte slice
  `[rank * shard_in_bytes .. (rank + 1) * shard_in_bytes]` after transpose.
- `dim == 1` (column-parallel, shard along `out_features`): output has shape
  `[in_total, out_total / world_size]`, column slice copied with byte gather
  (non-contiguous in the source).

Mirror the slicing logic from `_fuel_nn_retired/src/var_builder.rs:913-961`.
Test on F32 first, then BF16; Q4_0 sharding is a follow-up (block-aligned
shards require `in_features % 32 == 0` on the shard dim, which is true for
LLaMA-3-8B's hidden_size=4096 but worth a guard).

### Subtask E — `Llama3MultiProcess` (TP-aware lazy LLaMA wrapper)

Wrap `crate::lazy_llama_full::Llama3Model` with a TP variant. For LLaMA-3:

- `attn_q`, `attn_k`, `attn_v`, `mlp.gate_proj`, `mlp.up_proj` → column-parallel
- `attn_o`, `mlp.down_proj` → row-parallel (followed by AllReduce)
- Embeddings + LM head → replicated (every rank holds full copy)
- RmsNorm gains → replicated (broadcasted scalars)

The simplest shape is *not* to fork `Llama3Model` — instead, write a new
`TpLlama3Model` in the binary itself
(`fuel-examples/examples/llama_multiprocess/model.rs`) that mirrors
`Llama3Model::forward` but routes each linear through `LazyColumnParallel` /
`LazyRowParallel`. Pulls weights via Subtask D's sharded loader.

If the per-binary approach starts feeling like copypasta, promote to
`fuel-core/src/lazy_llama_multiprocess.rs` in a follow-up session.

### Subtask F — Binary revival

`fuel-examples/examples/llama_multiprocess/main.rs` + `model.rs` (rename
`_llama_multiprocess_retired` → `llama_multiprocess`, files were deleted in
Phase H so this is creating them from scratch). Structure mirrors the eager
original at commit `4ed0c05c`:

- `Args` struct (CLI parsing) — copy from eager, swap `which: Which` to
  `v3-8b` default since v2 / v3-70b need the same lazy port plus we already
  have Llama3.
- Self-fork dispatch — copy from eager.
- NCCL `UniqueId` file IPC — copy from eager, swap to Subtask C's helper.
- `Device::new_cuda(rank)` — works as-is on the lazy side.
- Model load — use Subtask D's sharded `MmapedSafetensors` loader, build
  `TpLlama3Model` via Subtask E.
- Inference loop — `model.forward(&tokens, index_pos)?.realize()?` per
  token; tokenize / sample via existing `fuel_transformers::generation::LogitsProcessor`
  which already handles `LazyTensor` (Phase H confirmed `generation` stays).

Re-add to `fuel-examples/Cargo.toml`:

```toml
[[example]]
name = "llama_multiprocess"
required-features = ["cuda", "nccl", "flash-attn"]
```

Verify with `cargo build --example llama_multiprocess --features cuda,nccl,flash-attn`.
Runtime verification requires a Linux box with CUDA runtime + NCCL + ≥2
CUDA devices (or a single-rank smoke pass on the Windows dev machine, which
exercises the codepath but degenerates AllReduce to identity).

## Estimated scope

| Phase | Subtasks | Sessions |
|-------|----------|----------|
| Substrate | A (LazyColumnParallel + LazyRowParallel) + B (LazyCommunicator trait + NCCL impl) | 2 |
| Driver    | C (process-group bootstrap) + D (sharded safetensors loader) | 1–2 |
| Model     | E (TpLlama3Model wrapper) | 1 |
| Binary    | F (revive binary, add to Cargo.toml, validate build) | 0.5 |
| **Total** |  | **4.5–5.5** |

Higher end if Subtask B turns out to need an `Op::AllReduce` (Option I)
after all — that pulls in 1 extra session for backend dispatch arms across
fuel-cpu-backend / fuel-cuda-backend / fuel-vulkan-kernels / etc.

## The optional, larger question — should this even land now?

Phase H quarantined this binary alongside `csm`, `debertav2`, `llava`,
`mamba-minimal`, `metavoice`, `paddleocr-vl`, `quantized-lfm2`. Every other
quarantined binary is being unblocked by the normal eager-retirement program
(lazy ports of missing model architectures, lazy ports of config / image
processor helpers). `llama_multiprocess` is qualitatively different — its
blocker is **multi-GPU infrastructure**, not a missing model. Every other
lazy port shipped fine without tensor parallelism.

**Open question:** should multi-process inference be deferred behind a
feature flag until a Fuel consumer actually asks for multi-GPU? Memory
entry [project_unified_durable_tensor_store](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_unified_durable_tensor_store.md)
already calls multi-GPU the "north star, parked until consumers need it";
[project_phase6d_d2d_design](../../.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_phase6d_d2d_design.md)
parked the cross-device-copy design with the same rationale.

Arguments for deferring:

- The eager binary was a demo, not a production path. No Fuel-internal code
  paths depend on TP working end-to-end. The eager `Communicator` trait +
  `ColumnParallel` / `RowParallel` scaffolding in `fuel-parallel` was shipped
  ahead of demand and has no consumers today.
- The architectural question (Option I vs Option II) is better answered
  *with* a real consumer in hand. Picking now is informed guessing.
- Phase 7.5 still has open structural work (picker integration, Op::Copy
  D2H bridge retirement, the load-time planner / in-place graph optimization —
  the separate ExecutionPlan artifact was retired by the 2026-06-14 "plan IS
  the graph" decision) — that's where session capacity is most productive.

Arguments for landing now:

- Phase H's quarantine list is a TODO; leaving entries unrevived
  indefinitely is a smell.
- The substrate is small (4.5–5.5 sessions) compared to the model-port
  backlog.
- Doing the work surfaces the real shape of the comm-as-graph-op question
  much faster than continuing to design in the abstract.

**Recommended posture:** defer this session unless a consumer surfaces (a
serving-side use case, a Fuel-internal benchmark that needs multi-GPU, a
downstream lazy-port that wants to use `LazyRowParallel` as a primitive).
When it does, the substrate work is small enough to ship in a single
multi-session push. Until then, leave `_llama_multiprocess_retired` in
quarantine and treat this prompt as a reservation slot.

## References

- Original implementation (read-only reference): git commit `4ed0c05c`,
  `fuel-examples/examples/llama_multiprocess/{main.rs,model.rs}`.
- Phase H retirement context: [eager-retirement-phase-h-plan.md](shipped/eager-retirement-phase-h-plan.md);
  [eager-tensor-retirement-master-plan.md](eager-tensor-retirement-master-plan.md).
- Existing eager TP scaffolding: [fuel-parallel/src/tensor_parallel.rs](../../fuel-parallel/src/tensor_parallel.rs);
  [fuel-parallel/src/comm.rs](../../fuel-parallel/src/comm.rs).
- Lazy LLaMA port to wrap: [fuel-core/src/lazy_llama_full.rs](../../fuel-core/src/lazy_llama_full.rs);
  [fuel-core/src/lazy.rs:LlamaModel](../../fuel-core/src/lazy.rs).
- Lazy weight-loading helpers to extend with shard-awareness:
  [fuel-core/src/lazy.rs:5655](../../fuel-core/src/lazy.rs#L5655) (`load_from_mmapped` for `LlamaWeights`).
- Reference for sharded var-builder slicing logic: [_fuel_nn_retired/src/var_builder.rs:835-961](../../_fuel_nn_retired/src/var_builder.rs#L835).
- baracuda-nccl Cargo wiring: [fuel-examples/Cargo.toml](../../fuel-examples/Cargo.toml) (`baracuda-nccl` dep at L25, `nccl` feature at L89).
- For the "first-class comm op" path if it gets picked instead: parallel
  to the fused-op registration pattern at
  [fuel-graph/src/registry/flash_attn.rs](../../fuel-graph/src/registry/flash_attn.rs).

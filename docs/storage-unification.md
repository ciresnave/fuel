# Storage Unification + Backend Contract

**Status**: design v1, 2026-05-03. Iterating before code lands.

> **2026-05-09 update — anchored to architecture v1.0**: this design doc was drafted before the architecture set in [`docs/architecture/`](architecture/00-index.md) was established. Most of its commitments survive unchanged in the v1.0 architecture and should be read alongside the relevant architecture sections:
>
> - **Storage as `(bytes, dtype, device)` substrate**: aligned with [03-ir](architecture/03-ir.md) and [05-backend-contract](architecture/05-backend-contract.md).
> - **Backend capability advertisement** (`(op, dtype)` support matrix, required alignment, transfer paths): aligned with [05-backend-contract §What every backend provides](architecture/05-backend-contract.md#what-every-backend-provides). Architecture v1.0 adds two more advertised dimensions: per-kernel `PrecisionGuarantee` (replaces the OracleGrade-style flag) and slot capacity (the per-device parallelism advertisement).
> - **CPU as universal fallback**: aligned with the architecture v1.0 "always-built backend coverage commitment" — the always-built backend (fuel-cpu-backend by convention) provides at least one `bit_stable_on_same_hardware: true` kernel for every primitive op as the architectural coverage guarantee.
> - **Phase A/B/C/D phasing**: still applicable; resume against the v1.0-aligned target.
>
> Where this doc and the architecture set conflict, the architecture set wins. This doc carries the implementation-side detail (Phase A through D migration steps); the architecture set carries the durable architectural commitments.

## TL;DR

Today's storage architecture is asymmetric: GPU backends already store data as bytes addressable through opaque handles (`CUdeviceptr`, `VkBuffer`), while CPU storage is over-typed via the 14-variant `HostBuffer` enum. Realize bridges, executor caches, and dispatch logic each separately re-derive what dtype a chunk of memory holds.

**The unification**: Storage is `(bytes, dtype, device)` everywhere — a uniform shape across all backends. Bytes are an addressable region in some address space; backend variation lives behind one closed enum (CPU, CUDA, Vulkan, Metal); dtype is a single tag. Backends advertise their capabilities as an `(op, dtype)` matrix plus required-alignment plus available transfer paths; the DAG layer consumes those advertisements to route ops and insert transfers as needed.

**The contract floor is minimal**: a backend must support `alloc(bytes)` and `copy_from(other_storage)`. Everything else (ops, dtypes, transfer paths) is capability-flagged. CPU is the universal fallback because fuel-cpu-backend supports every op for every standard dtype.

**Pauses Phase 7.5 B3** mid-flight (after Commit 1 — storage seam → `Result` — already on HEAD). B3 resumes against the unified foundation; remaining B3 commits become smaller and more correct.

---

## Goals

- **Single Storage shape across backends.** CPU/Vulkan/CUDA/Metal storage all expose the same surface; differences are isolated to the backend impl.
- **Dtype is metadata, not type identity.** No type parameter on `Storage`/`Tensor` — the dtype tag plus `bytemuck::cast_slice` (or device-equivalent) handles type access at consumer sites.
- **Capabilities are advertised facts.** Backends declare what they support `(op × dtype, alignment, granularity, transfer paths)`. Routing is a separate concern that consumes those facts.
- **Minimal required floor.** A backend only needs to be able to hold bytes and transfer to/from other backends' storage. CPU as universal fallback covers everything else.
- **Dispatch happens once.** At DAG construction (or earlier), not per-execution. The executor walks an already-specialized graph.
- **No production panics.** `Result`-returning throughout. Per existing project rule.

## Non-goals (this work item)

- **DAG router redesign.** Router stays roughly as-is; we extend its inputs (capability advertisements) but don't rewrite the routing algorithm.
- **Eliminating typed kernels.** Per-op kernels remain monomorphic over `T`; the change is at storage and dispatch, not kernel implementations.
- **Phase 8 tiers / serving / multi-GPU.** Those are downstream; this work item is foundation only.

---

## Current state

### `Storage` (fuel-core-types/src/storage.rs)

```rust
pub struct Storage(pub Box<dyn DynBackendStorage>);
```

A newtype over a trait object. The trait has dozens of methods covering the eager-dispatch op surface. Dtype is queried via `storage.dtype()` at runtime.

### `HostBuffer` (fuel-core-types/src/cpu_storage.rs)

```rust
pub enum HostBuffer {
    U8(Vec<u8>), U32(Vec<u32>), I16(Vec<i16>), I32(Vec<i32>), I64(Vec<i64>),
    BF16(Vec<bf16>), F16(Vec<f16>), F32(Vec<f32>), F64(Vec<f64>),
    F8E4M3(Vec<F8E4M3>),
    F6E2M3(Vec<u8>), F6E3M2(Vec<u8>), F4(Vec<u8>), F8E8M0(Vec<u8>),
}
```

14 variants. CPU dispatch matches on the variant to route to the right typed kernel.

### `CpuStorage` (fuel-cpu-backend/src/dyn_impl.rs)

Wraps `HostBuffer`. Implements `DynBackendStorage` and a separate `HostStorage` trait. The `HostStorage` split is a vestige — CPU is just a backend; treating "host" as conceptually distinct is unnecessary.

### `CudaStorage`, `VulkanStorage`, `MetalStorage`

Each holds a backend-specific buffer handle (`CUdeviceptr`, `VkBuffer`, `MTLBuffer`) plus byte length. Dtype is a metadata field. **Already aligned with the unified model.**

### `AnyTensor` (fuel-graph-cpu/src/lib.rs, internal)

```rust
enum AnyTensor {
    F32(RefTensor<f32>), F64(RefTensor<f64>),
    BF16(RefTensor<bf16>), F16(RefTensor<f16>),
    U32(RefTensor<u32>),
}
```

Executor's intermediate cache. Erased over dtype; matches happen at op-eval sites — these matches should move to DAG-construction time.

### Asymmetry summary

- GPU storage: `(handle, len_bytes, dtype)` — uniform, dtype is metadata.
- CPU storage: `HostBuffer` 14-variant enum, dtype implicit in variant identity.
- Cache types: `AnyTensor` (CPU) and `AnyRefTensor` (reference) — yet more enums per crate.
- HostStorage trait separate from BackendStorage trait — same concept twice.

The same data model surfaces multiple ways. Each consumer re-derives "what dtype is this?" via match at execution time when it should be known once at construction time.

---

## Target architecture

### Single `Storage` type

```rust
pub struct Storage {
    inner: BackendStorage,   // closed enum, one variant per backend
    dtype: DType,            // single tag
    // shape lives in Layout (unchanged)
}
```

`BackendStorage` is a closed enum over backend variants:

```rust
pub enum BackendStorage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanStorage),
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
}
```

Each variant is a concrete type that holds bytes + device handle. Adding a new backend means a new enum variant + matching arms — compile-checked, exhaustive.

**Trade-off accepted**: closed set vs open ecosystem. Backends in Fuel are workspace-defined; the enum gives faster dispatch (jump table), better compiler help, no vtable indirection. Trait-object plug-in extensibility is a theoretical use case we deliberately defer.

### CPU storage — bytes-uniform

```rust
pub struct CpuStorage {
    bytes: Arc<[u8]>,
    device: Arc<CpuDevice>,
}
```

`HostBuffer` enum **retires**. CPU stores bytes uniformly. Per-op kernels at entry do `bytemuck::cast_slice::<T>(&bytes)` to get a typed slice; at exit they go from typed result back to bytes. Alignment guaranteed by the allocation path (always allocated to alignment ≥ largest supported dtype size; see open question on exact value).

### GPU storage — minimal change

`CudaStorage`, `VulkanStorage`, `MetalStorage` are already shaped as `(handle, len_bytes, dtype)`. Implement the unified surface; their internals are unchanged.

### One `BackendStorageOps` trait — no HostStorage split

```rust
pub trait BackendStorageOps {
    fn len_bytes(&self) -> usize;
    fn device(&self) -> &dyn BackendDevice;

    /// Allocate fresh storage on this backend. Bytes uninitialized.
    fn alloc(&self, len_bytes: usize) -> Result<Self> where Self: Sized;

    /// Receive bytes from another backend's storage. Implementation
    /// uses the cheapest available transfer path (P2P, device copy,
    /// shared memory, or host staging fallback).
    fn copy_from(&mut self, src: &BackendStorage) -> Result<()>;
}
```

`BackendStorage` (the enum) implements convenience methods that match-dispatch to the variant's impl:

```rust
impl BackendStorage {
    pub fn len_bytes(&self) -> usize {
        match self {
            BackendStorage::Cpu(s) => s.len_bytes(),
            #[cfg(feature = "cuda")] BackendStorage::Cuda(s) => s.len_bytes(),
            // ...
        }
    }
    // similarly for device(), alloc(), copy_from()
}
```

CPU is just a backend variant. `BackendStorage::Cpu` is the host. There's no separate HostStorage concept.

### Capability advertisement

A backend advertises its capabilities once at registration:

```rust
pub struct BackendCapabilities {
    pub backend_id: BackendId,
    pub device: Arc<dyn BackendDevice>,

    /// (op, dtype) → supported. Static after backend init.
    pub op_dtype_support: HashSet<(OpKind, DType)>,

    /// Required alignment in bytes for storage on this backend.
    pub required_alignment: usize,

    /// Smallest addressable unit in bits. Most are 8 (byte-addressable).
    pub access_granularity_bits: u32,

    /// Transfer paths this backend can use as the source / destination.
    pub transfer_paths: Vec<(DeviceId, TransferPath)>,
}

pub trait BackendCapabilityProvider {
    fn capabilities(&self) -> BackendCapabilities;
}
```

`Router` collects `BackendCapabilities` from each registered backend at startup. The collected matrix is what the DAG layer queries during routing.

### Required floor

The minimum every backend must support:

- `BackendStorageOps::alloc(len_bytes)` — can hold bytes.
- `BackendStorageOps::copy_from(src)` — can receive bytes from any other backend's storage.

That's it. No required ops, no required dtypes. **CPU backend is the universal fallback**: fuel-cpu-backend supports every (op, dtype) combination, so as long as CPU is registered, every op has *some* backend that can run it.

If no backend in the registered set supports a particular (op, dtype), the DAG construction fails with a clear typed error pointing at the specific (op, dtype, available backends) gap. No silent panics.

### Transfer paths

```rust
pub enum TransferPath {
    /// No transfer needed (same device instance).
    SameDevice,
    /// Direct peer-to-peer (CUDA P2P, NVLink, Infinity Fabric, GPUDirect).
    Peer { peer_device: DeviceId },
    /// Zero-copy via shared memory mapping (UMA, ResizableBAR, dma-buf).
    SharedMemory { kind: SharedMemoryKind },
    /// Bulk transfer engine (cudaMemcpy, vkCmdCopyBuffer + staging).
    DeviceCopy,
    /// Through CPU as intermediary. Universal fallback.
    HostStaging,
}

impl TransferPath {
    /// Estimated bytes/sec for this path. Used by Router to pick
    /// among available paths between two devices.
    fn bandwidth(&self) -> Option<f64> { ... }

    /// Setup latency in microseconds.
    fn latency_us(&self) -> f64 { ... }
}
```

Each backend advertises its outbound paths in `BackendCapabilities::transfer_paths`. Router builds a transfer matrix at registration:

```rust
pub struct TransferMatrix {
    paths: HashMap<(DeviceId, DeviceId), TransferPath>,
}
```

When the DAG inserts an `Op::Move` or `Op::Copy`, Router looks up `paths[(src, dst)]` to pick the actual transfer mechanism. If no direct path exists, Router composes via `HostStaging` (the universal fallback).

The DAG sees uniform `Op::Move` / `Op::Copy` ops. Backends see only their own bytes (sender or receiver). Router does the routing bookkeeping in the middle.

### Inter-device transfer is not a new op

`Op::Move` (destructive — releases source) and `Op::Copy` (non-destructive) **already exist** and already cover transfer between any two devices. CPU is just a device. There is no separate `Op::ToHost` / `Op::ToDevice`; transfer to CPU is `Op::Move/Copy` targeting CPU; transfer between GPUs is `Op::Move/Copy` targeting the other GPU. Force-value paths (`to_vec*`, `Display`, `save_safetensors`) trigger an `Op::Copy` to CPU before reading bytes.

This is already how the lazy stack expresses transfers (per memory: PR #1 + #4 from earlier work). The unified storage model fits that existing graph-level abstraction without adding new ops.

### Dispatch at DAG construction, not per-op-eval

When a graph node is built (in the op-method or via `fuel_graph::Tensor::relu()` etc.), the dispatch decision happens **once**:

1. The op-method knows its `OpKind` (e.g., `OpKind::Unary(Relu)`).
2. The op-method knows its inputs' dtype (from input nodes).
3. Result dtype is computable from op semantics (relu preserves input dtype; cast changes it; matmul follows accumulation rules).
4. Router (or dispatch layer) is consulted: which backend's kernel handles `(OpKind::Unary(Relu), dtype)` cheapest given input residency?
5. The selected kernel function reference is **stored on the graph node**.

When the executor walks the graph, each node already has its kernel reference. Execution is just `node.kernel_fn(inputs)` — no match-on-dtype, no dispatch table lookup at execution time.

```rust
// Conceptual node shape post-unification:
pub struct Node {
    op: Op,
    inputs: Vec<NodeId>,
    shape: Shape,
    dtype: DType,
    kernel: KernelRef,       // <-- pre-resolved at construction
    target_backend: BackendId,
    // ...
}
```

This collapses runtime work for tight loops. A training step that runs the same graph 10k times pays the dispatch cost once, at graph build time, not 10k times at each iteration.

### Pipelined compilation

Compilation and execution can overlap, cutting time-to-first-token for inference. The model:

- Compilation walks the graph in topological order, resolving `(op, dtype, target_backend) → kernel ref` for each node.
- Compiled nodes get pushed onto a `crossbeam::channel` (or equivalent) shared between compiler and executor threads.
- The executor consumes from the channel; if it needs a node's inputs and they're not yet computed, it blocks until they are.
- Compiler runs ahead of executor; both are concurrent threads.

For a deep network (LLaMA forward pass: ~100 layers × ~10 ops/layer ≈ 1000 nodes), compiling the first few nodes takes microseconds, so the executor can start producing output while the compiler is still resolving later nodes. This mirrors what TorchInductor does in PyTorch.

Sync alternative: a single-threaded "compile then execute" model. We support both — the threaded model is opt-in (e.g. via a `Router::with_pipelined_compilation(true)` setting). Default to the threaded model for inference; sync mode for tests + debugging.

Key requirement: dispatch resolution for node N must be independent of node N+1. Compiler emits a `CompiledNode { kernel, inputs, output_storage_handle }` per node into the channel; executor uses the handle to resolve the actual storage at run time.

### Realize bridge — no dtype handling

```rust
// fuel-graph-cpu (similar entries in vulkan, cuda):
pub fn realize_into_storage(link: &fuel_graph::Tensor) -> Result<Storage> {
    // Executor walk; cache is HashMap<NodeId, Storage>
    // Each node's pre-resolved kernel runs against its inputs from cache
    // Returns the link.id() node's Storage
}

// fuel-core/src/lazy_realize.rs:
pub fn realize_into_storage(link: &fuel_graph::Tensor) -> Result<Arc<RwLock<Storage>>> {
    if let Some(arc) = link.storage_for() { return Ok(arc); }
    let storage = router().realize_into_storage(link)?;
    let arc = Arc::new(RwLock::new(storage));
    link.graph().write().unwrap().set_storage(link.id(), arc.clone());
    Ok(arc)
}
```

No `DtypeRealizer` trait. No `AnyTensor` enum. The bridge is dtype-unaware. Per-op kernels are dtype-specialized and pre-bound to nodes; the executor just calls them.

---

## Backend contract details

### Capability matrix shape

```rust
pub struct BackendCapabilities {
    pub backend_id: BackendId,
    pub device: Arc<dyn BackendDevice>,
    pub op_dtype_support: HashSet<(OpKind, DType)>,
    pub required_alignment: usize,
    pub access_granularity_bits: u32,
    pub transfer_paths: Vec<(DeviceId, TransferPath)>,
}
```

`HashSet<(OpKind, DType)>` is bounded — at most ~50 ops × ~14 dtypes = ~700 entries, populated by what each backend actually implements. Static after backend init.

### Granularity / alignment handling

If a backend declares `access_granularity_bits = 64` (only 64-bit aligned reads/writes), Router's responsibility:

- Refuse to route ops there if the source data layout doesn't meet 64-bit alignment, OR
- Insert a packing/repacking op before the route, OR
- Route elsewhere.

Backends advertise the constraint; Router/DAG decides what to do with it. Backends never need to "handle" misaligned input — they just refuse.

Same for `required_alignment`. Backends report; Router routes accordingly.

### Op coverage examples (illustrative, not exhaustive)

CPU backend (fuel-cpu-backend) — universal fallback:

- All ops × all dtypes (slow paths exist for every combination)

CUDA backend (fuel-cuda-backend):

- matmul × {f16, bf16, f32, f64}
- elementwise (add, mul, …) × {f16, bf16, f32, f64, u32, i64}
- conv2d × {f32, f16, bf16}
- q-matmul × {q4_0, q4_k_m, q8_0}
- flash-attn × {f16, bf16}

Vulkan backend (fuel-vulkan-backend):

- matmul × {f32, f16}
- elementwise × {f32, f16, u32}
- q-matmul × {q4_0, q4_k_m}
- conv2d × {f32}

The union covers every op every backend implements; the intersection is the floor of "any backend can handle this." The matrix is queried as `caps.op_dtype_support.contains(&(op, dtype))`.

---

## Mutation: copy-on-write + graph-walk invalidation

Two distinct correctness problems, both solved:

### Problem 1: shared Arc bytes

When `t = v.transpose()` shares the same `Arc<[u8]>` as `v`, mutating `v` would corrupt `t`'s view. Solution: copy-on-write via `Arc::make_mut` inside `Storage::storage_mut()`. If the Arc is uniquely held, mutation is in-place (no copy). If shared, the bytes clone before mutation; the original holders see the unchanged data.

```rust
impl Storage {
    pub fn storage_mut(&mut self) -> Result<&mut [u8]> {
        // CoW: only clones if shared. Cheap atomic load otherwise.
        Ok(Arc::make_mut(&mut self.inner.bytes_mut()))
    }
}
```

Centralized location is deliberate: every mutating call site gets the safety guarantee for free; no opt-out, no chance to forget.

### Problem 2: stale cached results

When `R = f(v)` was computed and cached in `R`'s slot, mutating `v` does *not* invalidate `R`'s cache. `R`'s slot still holds bytes computed from old `v`; readers see stale results. CoW does not solve this — `R`'s bytes are independent of `v`'s bytes; the cache staleness is at the *graph dependency* level, not the storage level.

Solution: explicit graph-walk invalidation at known mutation entry points (`Variable::set`, `Tensor::const_set`, `scatter_set`, `scatter_add_set`, `slice_set`). When these entry points fire, they walk forward from `self.id()` in the graph and `remove_storage(downstream_id)` for every transitive dependent. Dependents re-realize next time they're read.

```rust
impl Variable {
    pub fn set(&self, src: &Tensor) -> Result<()> {
        // 1. CoW-safe write.
        let bytes = self.storage_mut()?;
        bytes.copy_from_slice(&src.host_bytes()?);

        // 2. Invalidate dependents.
        if let Some(link) = self.graph_link() {
            let mut graph = link.graph().write().unwrap();
            for dependent_id in transitive_dependents(&graph, link.id()) {
                graph.remove_storage(dependent_id);
            }
        }
        Ok(())
    }
}
```

The two mechanisms are complementary, not alternatives. CoW handles direct alias safety at the bytes layer. Graph-walk handles cache staleness at the dependency layer. Both are required for correctness.

### Why not "just CoW" or "just graph-walk"?

- Just CoW: `R = f(v)` cached → `v` mutated → cache shows stale `R`. Wrong answer next read.
- Just graph-walk: `t = v.transpose()` shares Arc → `v` mutated → `t`'s bytes silently change underneath any code holding it. Worse — silent data corruption rather than just stale cache.

Both prevent different failure modes. Both are cheap. Both go in.

---

## Variable / autograd interaction

`Tensor_::op` (`BackpropOp`) is a separate structure from `fuel_graph::Graph`. The autograd-graph-rewrite work item (Phase 7.5 work item C) eventually unifies them, but in this Storage unification work, BackpropOp stays as a parallel mechanism on `Tensor_`. No change to autograd semantics; only the storage substrate changes.

`Variable::set` integrates with mutation handling above — CoW at the storage layer, graph-walk invalidation at the value layer. Variables remain the canonical mutable parameter type; the autograd graph still flows through `BackpropOp::new1` / `new2` etc.

Once work item C lands (later — not in scope here), `BackpropOp` retires in favor of graph-rewrite passes that compute backward as additional Op nodes. The Storage unification doesn't block C; C builds on the unified Storage cleanly.

---

## KernelRef shape

```rust
pub type KernelRef = fn(
    inputs: &[&Storage],
    outputs: &mut [Storage],
    params: &OpParams,
) -> Result<()>;
```

- **Multi-output ops** (topk → values + indices, var_mean → variance + mean) get a multi-element `outputs` slice. Single-output ops get a 1-element slice. No special-casing.
- **Output Storage pre-allocated** by the executor before calling the kernel. Kernel writes into pre-allocated bytes; doesn't allocate.
- **OpParams** is a typed enum with one variant per op family that needs extra data:

```rust
pub enum OpParams {
    None,                    // most elementwise ops
    Reduce { dims: Vec<usize>, keepdim: bool },
    Conv2D { kernel_size: (usize, usize), stride: (usize, usize),
             padding: (usize, usize), dilation: (usize, usize), groups: usize },
    Slice { start: usize, end: usize, step: usize },
    Matmul { transpose_lhs: bool, transpose_rhs: bool },
    // ... per-op variants as needed
}
```

Each kernel knows which `OpParams` variant it expects. Mismatches are programming bugs (caught at graph-construction time by the dispatcher).

**Custom user ops** (`CustomOp1`/`CustomOp2`/`CustomOp3`) wrap as kernels with a `OpParams::Custom { boxed: Box<dyn CustomOpData> }` variant. Slight ceremony, but custom ops are rare.

**In-place ops** (the optimizer's job in work item D) reuse the input Storage as the output. The framework provides this as an executor-level optimization; kernels are written assuming distinct buffers.

---

## Error types

```rust
pub enum Error {
    // ... existing variants ...
    NoBackendForOp {
        op: OpKind,
        dtype: DType,
        available_backends: Vec<BackendId>,
        supported_combinations: Vec<(BackendId, OpKind, DType)>,  // diagnostic
    },
    UnsupportedTransfer {
        from: DeviceId,
        to: DeviceId,
        available_paths: Vec<TransferPath>,
    },
    AlignmentMismatch {
        backend: BackendId,
        required: usize,
        actual: usize,
    },
}
```

`NoBackendForOp` fires at DAG construction (not at execution) — when the dispatcher can't find a backend supporting `(op, dtype)` given input residency. The error includes diagnostic data so users see actionable text:

```text
no backend supports matmul on i64
  available backends: cpu, cuda
  cpu supports matmul on: f32, f64, f16, bf16, u32
  cuda supports matmul on: f32, f64, f16, bf16
```

`UnsupportedTransfer` and `AlignmentMismatch` fire similarly — at planning time, not execution.

---

## Test plan

Pyramid:

### Unit tests

Each backend's `BackendStorage` impl in isolation:

- alloc / copy_from / device identity / len_bytes
- specifically: Arc::make_mut behavior on shared/unique CpuStorage
- alignment guarantees on CPU allocation

Per-backend capability advertisement:

- supported_dtypes / supported_ops match what's actually implemented
- transfer_paths discovered correctly

### Integration tests

Multi-backend graph realization through Router:

- f32/f64/bf16/f16 round-trip on CPU
- Cross-backend Op::Move/Op::Copy with various TransferPath kinds (same-device, peer, host-staging)
- Capability-driven dispatch: build a graph with an op only one backend supports; verify it routes there
- NoBackendForOp error path: build a graph with an op no backend supports; verify the error includes diagnostic data

### End-to-end tests

Real model inference, before and after, comparing wallclock + tokens/sec:

- Small reference model: GPT-2 124M (already used in fuel-examples)
- Validates correctness (output tokens identical to pre-unification baseline)
- Validates performance (no >5% regression on CPU, GPU per-op throughput unchanged)

### Regression / benchmark

- Existing dispatch-table benchmark (per memory: Phase 7b shipped) is the right vehicle for per-op throughput
- New benchmark: time-to-first-token with pipelined compilation enabled vs disabled
- New benchmark: graph-build time pre vs post-unification (dispatch resolution moves earlier)

Tests land in roughly this order during migration: unit tests with each backend impl in Phase A, integration tests in Phase B, end-to-end in Phase C, benchmarks in Phase D.

---

## Feature flag handling

`BackendStorage` enum variants are conditionally compiled:

```rust
pub enum BackendStorage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]    Cuda(CudaStorage),
    #[cfg(feature = "vulkan")]  Vulkan(VulkanStorage),
    #[cfg(feature = "metal")]   Metal(MetalStorage),
}
```

Match arms throughout the codebase need matching `#[cfg]` attributes:

```rust
match backend_storage {
    BackendStorage::Cpu(s) => s.len_bytes(),
    #[cfg(feature = "cuda")]   BackendStorage::Cuda(s) => s.len_bytes(),
    #[cfg(feature = "vulkan")] BackendStorage::Vulkan(s) => s.len_bytes(),
    #[cfg(feature = "metal")]  BackendStorage::Metal(s) => s.len_bytes(),
}
```

Mechanical but visually noisy. Helper macros can collapse the boilerplate at common dispatch sites:

```rust
macro_rules! dispatch_storage {
    ($s:expr, $name:ident => $body:expr) => {
        match $s {
            BackendStorage::Cpu($name) => $body,
            #[cfg(feature = "cuda")]   BackendStorage::Cuda($name) => $body,
            #[cfg(feature = "vulkan")] BackendStorage::Vulkan($name) => $body,
            #[cfg(feature = "metal")]  BackendStorage::Metal($name) => $body,
        }
    };
}
```

Used as `dispatch_storage!(self, s => s.len_bytes())`. Macro hides the cfg arms; readers see the dispatch shape.

---

## Public API inventory

### Added (post-unification)

- `BackendStorage` (enum) — primary backend-erased storage type
- `BackendStorageOps` (trait) — required surface for backend storage impls
- `BackendCapabilities` (struct) — capability advertisement
- `BackendCapabilityProvider` (trait) — emits capabilities at registration
- `KernelRef` (type alias) — function-pointer kernel signature
- `OpParams` (enum) — typed extras per op family
- `TransferPath` (enum) — inter-device transfer mechanisms
- `TransferMatrix` (struct, in fuel-graph-router) — `(src, dst) → path` lookup
- `Error::NoBackendForOp`, `Error::UnsupportedTransfer`, `Error::AlignmentMismatch` — typed dispatcher errors
- `dispatch_storage!` macro — feature-flag-aware match helper

### Removed (post-unification)

- `HostBuffer` enum — collapses into `CpuStorage { bytes: Arc<[u8]>, dtype }`
- `HostStorage` trait — CPU is just a backend; surface merges into `BackendStorageOps`
- Old `CpuStorage` (HostBuffer-based) — replaced by bytes-based version
- `AnyTensor` (fuel-graph-cpu) — cache becomes `HashMap<NodeId, Storage>`
- `AnyRefTensor` (fuel-reference-backend) — same reason
- `DynBackendStorage` trait — methods migrate to `BackendStorageOps` or `BackendCapabilities`; some prune entirely (per-op methods retire when ops migrate to graph-builder shape in B3)
- Per-dtype `realize_f32`/`realize_f64`/`realize_bf16`/`realize_f16` free functions (in fuel-graph-cpu) — replaced by single `realize_into_storage(link) -> Storage`

### Preserved (no change to consumers)

- `Tensor` and `Tensor_` (fuel-core) — public API stays identical; internal storage swap is invisible
- `fuel_graph::Tensor` — same shape; backing changes
- `fuel_graph::Graph` — same; just stores different `Storage` shape per slot
- All factory functions (`zeros`, `ones`, `from_slice`, etc.) — same signatures
- All op methods (matmul, add, ...) — same signatures (B3 changes their bodies later)
- Force-value paths (`to_vec*`, `Display`, `save_safetensors`) — same signatures
- `Layout` (Shape + strides + start_offset) — separate from Storage; unchanged

---

## Migration plan

Sequenced commits, tree green throughout. Estimated 2-3 weeks of focused work. Pattern: each phase is a milestone; commit often within phases; write tests as soon as a component is testable.

### Branch state

Today (2026-05-03):

- `main` is far behind (~50 commits) — last commit predates the candle→fuel rename.
- `refactor/step-11-quantized-kernels` (active) has all in-flight work: Phase 6/7/8 complete + G/G2/B1/B2/B3-step-1 from Phase 7.5.
- `refactor/type-unification-and-crate-extraction` is a strict subset of step-11 (no divergent commits).

Recommended sequencing:

1. Merge `refactor/step-11-quantized-kernels` into `main` (catches main up; squash-merging the two reverts in step-11 history is acceptable for clean history).
2. Delete `refactor/type-unification-and-crate-extraction` (redundant).
3. Cut new `feature/storage-unification` from updated `main` for this work.
4. After foundation lands and merges back to main, B3 resumes on a fresh branch from main.

### Phase A: substrate (1 week)

- **A1**: Define new `BackendStorage` enum, new `Storage` shape, new `BackendStorageOps` trait. Live in a `storage_v2` module alongside existing types so the tree stays green.
- **A2**: Implement variants for CPU (new `CpuStorage` with `Arc<[u8]>` instead of `HostBuffer`).
- **A3**: Implement for Cuda, Vulkan, Metal — mostly mechanical, they're already byte-shaped.
- **A4**: Add `BackendCapabilities` + `BackendCapabilityProvider` trait. Per-backend impls advertise `op_dtype_support`, alignment, transfer paths.
- **A5**: Router collects capabilities at registration; builds `TransferMatrix` and `op_dtype_dispatch` lookup tables.

### Phase B: dispatch resolution at DAG construction (3-4 days)

- Graph `Node` gains a `kernel: KernelRef` field (or equivalent — see open question on shape).
- Op-method calls (or alternatively the executor's first pass) populate the kernel ref by querying Router for the cheapest backend supporting `(op, dtype, input_residency)`.
- Executor's per-eval match-on-dtype path goes away; eval just calls `node.kernel(inputs)`.

### Phase C: per-op kernel migration (1 week)

- For each op family in fuel-cpu-backend: rewrite to bytemuck-cast at entry/exit instead of `HostBuffer` variant match.
- One commit per op family. Old and new paths coexist while migration is in progress.
- GPU backends mostly need adapter shims since their internals are already bytes-based.

### Phase D: cleanup + retire (3 days)

- Once nothing uses old types: retire `HostBuffer` enum, old `CpuStorage`, `AnyTensor`, `AnyRefTensor`, `HostStorage` trait split.
- Update downstream crates (`fuel-nn`, `fuel-transformers`, …) — most don't change because public `Tensor` API is preserved.
- Final pass: docs, README, GUIDE.md/PATTERNS.md updates.

### Resume B3

Once foundation lands:

- B3 step 2 collapses to a single small commit: "wire `lazy_realize::realize_into_storage` into `realized_storage()` seam." (Was previously DtypeRealizer trait + wire-up.)
- B3 step 2.5 (mutation invalidation) — easier with `Arc<[u8]>` and `Arc::make_mut` (copy-on-write).
- B3 step 3 (lazy realize on read) — same.
- B3 step 4 (unary as graph-builder) — same, but built against a clean foundation.

---

## Resolved decisions

After three rounds of iteration with the user, all major design questions are resolved:

### Architecture

- **Closed enum** for `BackendStorage` (not trait object) — workspace-defined backends, faster dispatch.
- **Single `BackendStorageOps` trait** — no `HostStorage` split; CPU is just a backend.
- **`(op, dtype)` matrix** for capability advertisement — `HashSet<(OpKind, DType)>`, static after backend init.
- **Required floor**: `alloc(bytes)` + `copy_from(other_storage)`. Nothing else required.
- **CPU as universal fallback** — fuel-cpu-backend supports every op × every standard dtype.
- **No new transfer ops** — `Op::Move` (destructive) and `Op::Copy` (non-destructive) cover all inter-device transfer including to/from CPU.
- **Dispatch at DAG construction**, not per-op-eval — kernel ref resolved once, cached on the Node, executed N times without re-dispatch.
- **Pipelined compilation** — compiler and executor as concurrent threads, default-on for inference.

### Storage model

- **CPU storage**: `CpuStorage { bytes: Arc<[u8]>, device: Arc<CpuDevice> }`. `HostBuffer` enum retires.
- **GPU storage**: minimal change — already byte-shaped via handles.
- **Alignment**: 64-byte CPU baseline (AVX-512 friendly; trades 56 bytes max waste per tensor for SIMD speed).
- **Layout stays separate** from Storage — Shape + strides + start_offset is "how to interpret the bytes," orthogonal to "which device's bytes."

### Mutation

- **CoW + graph-walk both required**, not alternatives.
  - CoW (`Arc::make_mut`) prevents corruption when storage Arcs are shared (e.g., views over the same bytes).
  - Graph-walk invalidation prevents stale cached results when a mutation invalidates downstream computations.
- **CoW lives in `Storage::storage_mut()`** — centralized; no opt-out.
- **Graph-walk lives in mutation entry points** — `Variable::set`, `Tensor::const_set`, `scatter_set`, `scatter_add_set`, `slice_set`. Each walks forward and `remove_storage(dependent_id)` for transitive dependents.

### Dispatch + execution

- **`KernelRef = fn(&[&Storage], &mut [Storage], &OpParams) -> Result<()>`** — uniform signature; multi-element `outputs` slice handles multi-output ops.
- **`OpParams` is a typed enum** — one variant per op family that needs extras.
- **Output `Storage` pre-allocated** by the executor before kernel call.
- **In-place ops** are an executor optimization (work item D) — kernels written assuming distinct buffers.

### Capabilities

- **No versioning** — capabilities are static at backend instantiation; no runtime mutation.
- **Quantized dtypes (q4_0, q4_k_m, q8_0) extend `DType` enum** — uniform matrix key.
- **`OpKind` is a compile-checked enum** — adding an op requires updating every supporting backend (compile error catches misses).
- **FFI raw handles via per-variant typed accessors** — `BackendStorage::Cuda(s).handle()` etc. Not a uniform `raw_handle()` method.

### Backend registration

- **Explicit `Router::register(backend)` from app code**, made process-wide via `OnceLock`. Lazy `inventory`-style auto-registration is a possible later refinement but not in scope.
- **Multiple op-providers per device** (e.g., fuel-cpu-backend + fuel-aocl-cpu-backend + fuel-mkl-cpu-backend on CPU): each is a separate `BackendCapabilityProvider`; they share the same `BackendStorage::Cpu` variant for storage but advertise different `(op, dtype)` coverage. Router picks per-(op, dtype) winner. Same pattern extends to GPU when multiple backends address the same device (experimental).

### Variable / autograd

- **`BackpropOp` stays unchanged** — autograd graph is a parallel mechanism on `Tensor_`. Storage unification doesn't touch it.
- Work item C (autograd-as-graph-rewrite) eventually retires `BackpropOp` in favor of graph-rewrite passes; this work doesn't gate or block C.

### Branches

- **Migration starts from `main` after merging current `refactor/step-11-quantized-kernels`** — see "Branch state" in the migration plan section.

### Test pyramid

- Unit per backend (alloc, copy_from, capability advertisement)
- Integration multi-backend (Op::Move/Copy with TransferPath variants, dispatch errors, capability-driven routing)
- End-to-end on small reference model (GPT-2 124M before/after)
- Benchmarks for graph-build time, time-to-first-token, per-op throughput

### Typed errors

- `Error::NoBackendForOp { op, dtype, available_backends, supported_combinations }`
- `Error::UnsupportedTransfer { from, to, available_paths }`
- `Error::AlignmentMismatch { backend, required, actual }`

All fire at DAG construction (planning time), not at execution. Production paths panic-free per project rule.

---

## What's NOT in this design

- **DAG router internals.** Router stays as-is; we extend its inputs (capability advertisements) but don't change how it routes.
- **Performance tuning.** Once correctness is in, tuning (SIMD widening, allocator tuning, transfer-path optimization) is its own work.
- **Open-ended backend ecosystem.** Plugin-style third-party backends are explicitly deferred.
- **Storage on disk / memory mapping.** Out of scope here.

---

## Iteration log

- **2026-05-03 v0**: initial strawman, drafted from thread context.
- **2026-05-03 v1**: revisions after user pass — closed enum over trait object, single backend trait (no HostStorage), `(op, dtype)` capability matrix, removed Op::ToHost/ToDevice (use existing Op::Move/Op::Copy), dispatch at DAG construction not at op-eval, required floor reduced to alloc + copy_from. Open questions reset to round 2.
- **2026-05-03 v2**: round-2 questions all resolved. Added: pipelined compilation subsection; mutation handling (CoW + graph-walk both); Variable/autograd interaction note; KernelRef multi-output signature; OpParams typed enum; typed errors (NoBackendForOp, UnsupportedTransfer, AlignmentMismatch); test pyramid; feature-flag handling; public API inventory (added/removed/preserved); branch state recommendation. All design questions consolidated into "Resolved decisions" section. Doc is ready for implementation.

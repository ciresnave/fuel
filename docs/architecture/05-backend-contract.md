# Backend contract

**Status**: v0.4 (draft, 2026-06-07). v0.4 changes: Reference backend retired from the dispatch system. No `BackendId::Reference` enum variant, no `ReferenceFactory`, no `LazyTensor::realize_f32_reference()`, no `fuel_reference_backend::probe` module. Judge correctness comparison now uses pairwise consensus across whatever backends are present at each profiling cell — no privileged oracle. The `fuel-reference-backend` crate remains as a test-oracle utility (`exec::realize_f32` + `attention` + `ops`) for callers that explicitly want textbook scalar math, but it is no longer architecturally privileged. v0.3 changes (preserved): concrete trait surface, Tier 1/2/3 mandatory/optional framing, per-backend compliance snapshot, inference-vs-training capability split, `Option<u64>` honest returns for runtime state. v0.2 changes (preserved): `PrecisionGuarantee` structure replacing binary OracleGrade; Reference's architectural privilege replaced by the "always-built coverage commitment" on fuel-cpu-backend; opt-in kernel-stat telemetry upload.

What backends provide to the Foundation layer, what they don't decide, and how the boundary is enforced. Anchored in the architectural principle from [01-identity](01-identity.md): **backends advertise; they don't decide.** Every strategic choice (placement, fusion, kernel selection, slot assignment, tolerance trade-off) lives at the DAG level. Backends provide the substrate the optimizer reasons about.

---

## What every backend provides

Every backend (CPU, CUDA, Vulkan, Metal, AOCL, MKL, future ones) exposes a uniform surface to the Foundation layer. The contract is composed of static capability advertisement plus dynamic telemetry.

### Static capability advertisement (registered at startup)

Reported once when the backend instance is registered; immutable thereafter without restart:

- **Identity**: a `BackendId` (Cpu, Cuda, Vulkan, Metal, AoclCpu, MklCpu, ...) plus a `DeviceLocation` distinguishing physical instances within the backend (CUDA GPU 0 vs GPU 1).
- **Op-dtype support set**: `(OpKind, DType)` pairs the backend has kernels for. The Router reads this to know what's dispatchable here. Routing checks `contains(&(op, dtype))` before dispatching.
- **Per-kernel cost annotation**: a function `cost(shapes, params, capabilities) -> CostEstimate { flops, bytes_moved, kernel_overhead_ns }` per registered kernel. Used by the optimizer's static-cost layer (see [04-optimization](04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism)). Pessimistic upper bounds are the convention; conservative on uncertainty.
- **Per-kernel `PrecisionGuarantee`**: each kernel declares its precision properties as a structure with multiple optional bounds (see [Per-kernel precision guarantees](#per-kernel-precision-guarantees) below). Replaces the binary "oracle vs production" framing. The optimizer reads these to decide whether a kernel is admissible under a given tolerance budget; calibration tooling reads them to pick comparators for tolerance discovery.
- **Per-kernel `KernelCaps`**: capability flags like `strided_input` (the kernel handles non-contiguous layouts directly) used by the optimizer's layout-fixup pass.
- **Required allocation alignment + access granularity**: alignment in bytes (Router pads/repacks if a source storage doesn't meet the destination's alignment); smallest addressable unit in bits (Router routes around granularity limits or refuses to route there).
- **Outbound transfer paths**: list of `(destination DeviceLocation, TransferPath)` tuples advertising how this backend can move bytes outward (`SameDevice`, `Peer`, `SharedMemory`, `DeviceCopy`, `HostStaging`). Router builds a transfer matrix from these.
- **Slot capacity per device**: the maximum concurrent execution contexts the device supports — CUDA stream count, Vulkan queue count, CPU thread-pool size. The optimizer reads this to size parallelism-budget cost contributions; the runtime route picker reads it for dispatch decisions (see [06-runtime](06-runtime.md)).
- **Kernel-revision hash per registered kernel**: a stable hash identifying the kernel's implementation version. Used by the persistence layer to detect when a cached optimization plan was built against a kernel that has since been updated. See [11-persistence §Invalidation](11-persistence.md).

### Dynamic telemetry (reported continuously)

Reported on a heartbeat (or queried on demand) while fuel is running. Used by the runtime route picker to adapt per-realize:

- **Currently-available slot count per device**: how many slots are currently idle vs in-flight. Drives the runtime's bounded dispatch lookahead.
- **Memory pressure**: bytes available for new allocations vs total device memory. The route picker prefers memory-conserving alternatives when pressure is high.
- **Queue depth**: pending work in the backend's own queue. Predicts how long a new dispatch will wait before executing.
- **Currently-resident weights**: which model weights/parameters are already loaded on the device (informs decisions about which routes avoid re-load costs). Optional; backends that don't track this report empty.
- **Local Judge profile data accumulation**: the empirical-cost layer of the cost model (see [04-optimization §Cost model](04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism)) accumulates per-(op, dtype, size_class, backend, device) latency measurements as ops run. This data is local; it informs the runtime route picker. Users may opt in to upload locally-aggregated summary statistics (median, P95, P99, sample count per cell) to the project's telemetry server, where aggregated data feeds the cache-generation tool's empirical priors and serves as a starting baseline for new users on similar hardware (see [08-pattern-harvest](08-pattern-harvest.md#shared-infrastructure-with-tolerance-recipes)).

The architectural commitment: **all the information the optimizer or route picker needs to reason about a backend lives on the contract surface, not inside the backend.** The optimizer never has to introspect a backend's internals; the route picker never has to guess at availability. Backends report; fuel reasons.

## What backends do NOT decide

The line is sharp:

- **Placement**: which (backend, device) runs which op. Decided by the optimizer (static cost) and the runtime route picker (telemetry). Backends accept what they're handed.
- **Fusion**: which subgraphs collapse to a fused op. Decided by the OptimizationMap; backends register fused-op kernels but don't choose when those kernels apply.
- **Kernel-variant selection** (across alternatives): if the optimizer kept multiple kernel-variant alternatives for an op at a decision point (cuBLAS vs custom matmul vs tensor-cores), the route picker chooses which one runs based on telemetry. Backends don't pick.
- **Slot assignment** (which stream/queue/thread runs a given dispatched op): the runtime allocates work to the backend's advertised slots based on current availability. Backends accept work on the named slot; they don't redirect.
- **Tolerance budget allocation**: backends advertise per-kernel error characteristics; the optimizer decides which kernels are admissible under the route's cumulative tolerance budget.

The single architectural commitment that ties this together: **strategic decisions happen on the DAG; backends execute.**

## What backends *do* control: intra-kernel concurrency

Within a single kernel call dispatched to a single slot, the backend retains full control over internal concurrency. Examples:

- **cuBLAS** internally launches multiple sub-kernels for one user-visible matmul; CUDA decides their scheduling.
- **Vulkan driver** may opportunistically parallelize command-buffer execution within a queue.
- **vendor BLAS thread pools** internal to one OpenBLAS/AOCL/MKL call.
- **rayon work-stealing** within one fuel-cpu-backend kernel's parallel loop.

Fuel sees one dispatch to one slot; whatever happens inside that dispatch is the backend's tactical concern. This is the principled exception to "backends don't decide" — it acknowledges that backends know their device's micro-architecture better than fuel does, and intra-kernel scheduling is irrelevant to the strategic decisions fuel reasons about.

The boundary: **fuel controls dispatch granularity (which slot, which device, which kernel variant); backends control intra-kernel internals (how the chosen kernel uses the slot's resources).**

## Slot semantics across backends

A "slot" is a uniform abstraction at the contract surface. What it represents underneath differs per backend:

| Backend | Slot maps to | Strictness |
| ------- | ------------ | ---------- |
| CUDA | Stream | Strict — concurrent kernels actually run on the device when streams have capacity |
| Vulkan | Queue | Strict — separate queues execute independently |
| CPU (rayon kernels) | Bounded sub-pool of the global rayon pool | Strict — `pool.install` reserves N threads |
| CPU (BLAS-linked) | Thread-count assignment via `set_num_threads` | Advisory — vendor BLAS may oversubscribe under specific conditions |
| Metal | Command queue | Strict |
| AOCL/MKL CPU | Per-call thread-count via library API | Advisory |

The contract advertises slot count uniformly. The strictness column is implementation detail backend-side. Architecturally: a slot represents a bounded execution context; the strictness of the bound depends on the backend's underlying primitive. The runtime tolerates mild advisory-bound overruns (some scheduling jitter) as long as backends honor the bound on a best-effort basis.

## Per-backend kernel registration

Each kernel a backend ships is registered against the binding-table catalog. Registration includes:

- The op + per-operand dtypes the kernel implements.
- The `KernelRef` function pointer (the actual call surface).
- The kernel's `KernelCaps` (strided-input, future capability flags).
- The kernel's static cost-estimate function (for the optimizer's cost-model layer 1).
- The kernel's error characteristics (strict / approximate-with-bound / calibrated).
- The kernel's revision hash (for cache-invalidation detection by the persistence layer).

Registration is at backend init; the catalog is frozen for the process lifetime thereafter. Adding a kernel to a backend means re-compiling fuel (registration happens in the backend crate's setup code).

## Backend implementation guidelines

Three commitments backend implementers must honor:

1. **Honor the kernel signature.** `KernelRef` is `fn(inputs: &[Arc<RwLock<Storage>>], outputs: &mut [Arc<RwLock<Storage>>], layouts: &[Layout], params: &OpParams) -> Result<()>`. Outputs are pre-allocated by the executor; kernels write into them, never allocate. Kernels return `Result`, never panic.

2. **Honor the slot assignment.** When the executor dispatches `(kernel, inputs, outputs, slot_id)`, the backend executes on the named slot to the strictness its primitive supports. CPU backends with vendor BLAS use `set_num_threads(slot.threads)` per call; CUDA backends use the named stream; etc.

3. **Honor the cost annotation.** The static cost the backend declares should be a *pessimistic upper bound* — better to overestimate cost (the optimizer demotes the kernel; loss is missed opportunity) than underestimate (the optimizer over-uses it; loss is misallocation that empirical Judge data eventually corrects). When in doubt, round up.

## What this rules out

- **No backend-internal placement decisions.** A backend that internally redirects work between its own devices ("CUDA backend with 4 GPUs, automatically load-balances") violates the contract. The Router decides which (backend, device) handles each op; backends accept their placement.
- **No backend-side fusion.** A backend that internally fuses adjacent kernel calls violates the contract — the fused-form should be a registered fused-op kernel that the optimizer chose. Otherwise the optimizer's cost model is wrong (it priced two ops; backend ran one).
- **No silent kernel-variant substitution.** A backend that internally swaps one kernel for a "better" variant violates the contract. Variant selection happens at the optimizer's decision points.
- **No backend-internal cache.** A backend that caches results between calls violates the contract — caching is the executor's concern (and currently not in scope; the architecture is purely re-execute-from-scratch).

These rejections are what makes the optimizer's reasoning sound. If backends could silently change behavior, the optimizer's cost model would systematically lie and the runtime route picker would chase phantoms.

## Per-kernel precision guarantees

Each kernel registered with the binding-table catalog declares its precision properties as a `PrecisionGuarantee` structure. This replaces the older "fuel-reference-backend as distinguished oracle" framing — instead of one backend being the oracle, kernels across any backend can declare guarantees that qualify them as comparators for correctness testing and tolerance calibration.

```rust
pub struct PrecisionGuarantee {
    /// Same inputs → same bits on the same hardware. The strictest commitment;
    /// only achievable for plain scalar code with controlled rounding mode and
    /// no thread-order non-determinism (no rayon, no vendor BLAS, no SIMD with
    /// implementation-defined reduction order).
    pub bit_stable_on_same_hardware: bool,

    /// Maximum error in ULP relative to the correctly-rounded result, if the
    /// kernel author has characterized this. `None` means uncharacterized;
    /// conservative consumers treat it as "no commitment, assume worst case."
    pub max_ulp: Option<u32>,

    /// Maximum relative error per output element, if known.
    pub max_relative: Option<f64>,

    /// Maximum absolute error per output element, if known.
    pub max_absolute: Option<f64>,

    /// Free-text notes about edge cases (denormals, NaN propagation,
    /// hardware-specific quirks). For documentation, not for filtering.
    pub notes: &'static str,
}
```

The `Option`s let kernel authors declare what they can characterize without forcing fabrications where they can't. Kernels for primitives with well-known IEEE behavior (add, mul, the IEEE-required correctly-rounded operations) can declare tight bounds easily. Kernels for fused operations with internal reductions need actual numerical analysis to declare bounds rigorously; without it, conservative defaults apply.

How consumers use the structure:

- **The optimizer's tolerance-budget pass** (see [04-optimization](04-optimization.md#tolerance-budgets-gate-which-rules-fire)) filters per-decision-point alternatives by their `PrecisionGuarantee`: a route admissible under the user's budget is one whose accumulated error along every path stays within the budget. Kernels declaring tight bounds participate in tighter routes.
- **The optimizer's precision-filter pass** runs before cost ranking: alternatives that exceed the user's per-call precision requirement are pruned regardless of how cheap they are.
- **Calibration tooling** (see [07-tolerance §Tolerance discovery and calibration](07-tolerance.md#tolerance-discovery-and-calibration)) picks comparators by querying for kernels with `bit_stable_on_same_hardware: true` and tight `max_ulp`. Multiple kernels can serve as oracles; the framework picks the most-precise available.
- **Empirical Judge refinement** (per the trajectory "annotations now → empirical Judge later"): the Judge can measure actual error per-cell against an oracle-grade comparator and refine the registered `max_ulp` / `max_relative` / `max_absolute` values from data over time.

## The always-built coverage commitment

The architecture makes one structural commitment that replaces the historical "reference backend always available" guarantee:

**The always-built backend (fuel-cpu-backend by current convention) commits to providing at least one kernel with `bit_stable_on_same_hardware: true` for every op in the closed primitive set.** Ops without such a coverage kernel cannot be tolerance-calibrated, cannot be correctness-checked, cannot serve as anchors for cross-backend equivalence tests.

This makes the coverage guarantee a contract clause, not a separate crate. It's testable: a CI lint asserts the always-built backend has a `bit_stable` kernel for every primitive `Op` variant. New primitives added to the IR (rare) trigger a coverage-failure until the backend ships the corresponding kernel.

The `fuel-reference-backend` crate, as of v0.4 (2026-06-07), no longer participates in the dispatch system. `BackendId::Reference` is removed, `ReferenceFactory` is removed, `LazyTensor::realize_f32_reference()` is removed, and the crate's `probe` module is deleted. What remains is `fuel_reference_backend::exec::realize_f32` + `attention` + `ops` — a test-oracle utility available to callers that explicitly want textbook scalar math (e.g. backend test suites validating their own kernels). The crate is no longer architecturally privileged; the bit-stable coverage commitment is fulfilled by fuel-cpu-backend's portable kernels, and the Judge's correctness comparison uses pairwise consensus across the backends present at each profiling cell (no privileged oracle). See §Pairwise consensus correctness below.

## Pairwise consensus correctness

Pre-v0.4 the Judge compared every backend's output against a privileged Reference kernel and recorded `max_rel_error` as "drift vs Reference." v0.4 retires the privileged oracle: for each `(op, dtype, size_class)` profiling cell, the Judge runs every backend present in the probe, clusters their outputs by mutual `rel_err < CONSENSUS_EPSILON` (default 1e-3), and records each backend's `max_rel_error` as "drift vs the consensus cluster's other members."

Semantics:

- **N=0 backends at a cell**: no entries emitted.
- **N=1**: trivial consensus `[0]`; the lone backend's `max_rel_error` is `0.0` by convention (no peers means no comparison reference). Callers interpret this as "no cross-backend signal available," not "perfect."
- **N=2 agreeing within epsilon**: consensus is both; each reports its rel_err against the other (typically small, reflecting f32 accumulation-order drift between honest implementations).
- **N=2 disagreeing**: consensus is `[0]` (first wins ties by index); the other gets the disagreement as its rel_err. The discrepancy is surfaced but neither answer is independently validated — callers should treat this as "human review needed" rather than "the first one is right."
- **N≥3 with one outlier**: consensus is the cluster that mutually agrees (typically the majority); outlier reports its rel_err vs the cluster.

This is more honest than the Reference-vs-everyone model. Reference's bit-stability was always per-hardware (textbook scalar loops on different CPUs produce different bits), so "matches Reference" was never a global truth. Consensus instead asks "does this backend match the other honest implementations available here?" — which is the actual question correctness telemetry should answer.

The pre-retirement Reference comparator can still be invoked by tests that explicitly want it via `fuel_reference_backend::exec::realize_f32(&graph_tensor)` — the crate remains as a test-oracle utility. It just isn't the dispatch system's special oracle.

Future work: capture consensus outputs into a distributable fixture file (proposed in [`docs/session-prompts/reference-backend-retirement.md`](../session-prompts/reference-backend-retirement.md)) so subsequent Judge runs on systems with fewer backends can validate against pre-agreed outputs instead of needing ≥2 backends locally for inline consensus. v0.4 ships the inline-consensus core; fixture capture is the optimization on top.

## Cross-backend precision comparisons

A kernel marked `bit_stable_on_same_hardware: true` is bit-stable *on the hardware it runs on*. Two such kernels, one for CPU and one for CUDA, will not be byte-equivalent across backends — IEEE rounding semantics differ between hardware, scalar code paths differ between CPU FPUs and GPU shader cores, and so on. This was true with the historical reference-backend model too.

The architecture commits to:

- Bit-stability *within* a backend on the same hardware (deterministic, reproducible).
- Bounded-error correspondence *across* backends (kernels at the same precision tier produce results within the union of their declared bounds when compared cross-hardware).

Cross-backend correctness tests have to use bounded-error comparison, not bit-equivalence. The exact tolerance for cross-backend comparison is the sum of the two kernels' declared bounds, or — when one or both is unbounded — a coarse epsilon (e.g., 1e-5 relative for F32) with an explicit warning that the comparison is not rigorous.

---

## See also

- [01-identity](01-identity.md) — the "backends advertise; they don't decide" principle this section operationalizes.
- [03-ir](03-ir.md) — the `Op`, `KernelRef`, `OpParams` types backends register against.
- [04-optimization](04-optimization.md) — how the optimizer consumes per-kernel cost, error chars, slot capacity.
- [06-runtime](06-runtime.md) — how the runtime consumes telemetry to dispatch.
- [07-tolerance](07-tolerance.md) — per-kernel error characteristics and how they gate admissibility.
- [11-persistence](11-persistence.md) — kernel-revision hashes used for cache invalidation.
- ROADMAP §"Phase 7" — backend-modularity and pluggable-dispatch decisions.

---

## Trait surface

The prose above describes what backends provide. This section pins
that to the concrete Rust trait shapes every backend implements.
The traits compose: every backend implements Tier 1; backends with
the corresponding underlying primitive implement Tiers 2 and 3.

Naming and layout convention:

- All Tier 1 trait methods live in `fuel-core-types::backend` so
  every crate can name them without depending on backend
  implementations.
- Trait impls live in the per-backend crate (`fuel-cpu-backend`,
  `fuel-cuda-backend`, `fuel-vulkan-backend`, `fuel-aocl-cpu-backend`,
  `fuel-mkl-cpu-backend`, `fuel-reference-backend`).
- The picker / optimizer / planner consume only the trait — never
  inherent methods on a specific backend type.

### Tier 1 — Mandatory (every backend must implement)

Tier 1 is the picker's load-bearing surface. A backend that cannot
satisfy Tier 1 cannot participate in dispatch.

#### `BackendIdentity`

```rust
pub trait BackendIdentity {
    /// Stable enum membership — `BackendId::Cpu`, `BackendId::Cuda`,
    /// `BackendId::Vulkan`, etc.
    fn backend_id(&self) -> BackendId;

    /// Specific device within the backend's family. Stateless
    /// backends (CPU, Reference, AOCL, MKL) return
    /// `DeviceLocation::Cpu`; multi-device backends return their
    /// concrete location.
    fn device_location(&self) -> DeviceLocation;

    /// Identity comparison — same backend AND same device. Used by
    /// SystemTopology's `shares_storage` and by the executor's
    /// dispatch-chunk boundary detection.
    fn same_device(&self, other: &dyn BackendIdentity) -> bool;
}
```

`DynBackendDevice` is the existing trait; v0.3 names the contract
explicitly so a future split (e.g. separating identity from the
allocator surface) doesn't break consumers.

#### `BackendCapabilityProvider`

```rust
pub trait BackendCapabilityProvider {
    /// Snapshot of the backend's capabilities. Capabilities are
    /// static at backend instantiation — no runtime mutation, no
    /// versioning. Adding a new dtype or op to a backend requires
    /// recompiling Fuel.
    fn capabilities(&self) -> BackendCapabilities;
}
```

Already present at [fuel-core-types/src/backend.rs](../../fuel-core-types/src/backend.rs).
v0.3 confirms it stays in Tier 1 unchanged.

#### `BackendRuntime` (NEW in v0.3)

```rust
pub trait BackendRuntime {
    /// Bytes currently available for new allocations on this
    /// backend's device. `None` when the backend genuinely cannot
    /// measure (no OS query, no vendor API exposes it). Selectors
    /// treat `None` as "no pressure signal — fall back to static
    /// cost."
    ///
    /// Cheap to call (cached for ~100ms internally if the
    /// underlying query is non-trivial). Selectors poll at
    /// sub-realize granularity, not in tight loops.
    fn available_bytes(&self) -> Option<u64>;

    /// Total memory on this backend's device. Static after first
    /// call; cached unconditionally. `None` for backends with
    /// unbounded notional capacity (e.g. Reference — synthetic
    /// "infinite memory").
    fn total_bytes(&self) -> Option<u64>;

    /// Predictive fit-check: would an allocation of `size` bytes
    /// likely succeed given current state? Returns `Tight` when
    /// projected usage crosses a configurable pressure threshold
    /// (typically 0.85 of `total_bytes`).
    ///
    /// Default implementation derives from `available_bytes` +
    /// `total_bytes`. Backends with native predictive APIs (Vulkan
    /// `VK_EXT_memory_budget`) override for accuracy.
    fn would_fit(&self, size: u64) -> FitStatus {
        match (self.available_bytes(), self.total_bytes()) {
            (Some(avail), Some(total)) => {
                if size > avail { FitStatus::WontFit }
                else if (avail - size) as f64 / total as f64 > 0.85 { FitStatus::Tight }
                else { FitStatus::Comfortable }
            }
            _ => FitStatus::Unknown,
        }
    }
}

pub enum FitStatus {
    /// Allocation projected to fit comfortably.
    Comfortable,
    /// Allocation projected to fit but leaves the device tight.
    /// Selector should prefer a less-loaded backend if available.
    Tight,
    /// Allocation projected NOT to fit. Selector should pick a
    /// different backend or surface a planner-level error.
    WontFit,
    /// Backend cannot answer — `available_bytes` returned `None`.
    /// Selector falls back to static cost.
    Unknown,
}
```

Every backend implements `BackendRuntime`. The `Option<u64>` returns
let backends honestly report what they can measure without forcing
fabrication. The default `would_fit` implementation makes the
pressure-aware selector backend-agnostic: it queries the trait, the
trait reports honestly, the selector decides.

### Tier 2 — Conditional (backend-dependent primitives)

Tier 2 is for backends whose underlying primitive exposes the
signal. Selectors check at runtime whether the backend implements
the trait (via downcast through `BackendId` dispatch) and adapt.

#### `BackendStreams`

For backends with a deferred-execution / queue model.

```rust
pub trait BackendStreams: BackendRuntime {
    /// Number of slots (streams / queues / thread-pool tasks)
    /// currently busy with submitted-but-not-yet-finished work.
    /// `None` when the backend dispatches synchronously and has no
    /// queue concept.
    fn pending_work_count(&self) -> Option<u32>;

    /// Maximum concurrent in-flight work this backend supports
    /// (advertised slot capacity from `BackendCapabilities`). Used
    /// by the runtime route picker for dispatch lookahead sizing.
    fn slot_capacity(&self) -> u32;

    /// Block until all submitted work on this backend's slots has
    /// completed. Used at realize boundaries and by training-loop
    /// barriers (gradient accumulation, optimizer step).
    fn flush(&self) -> Result<()>;
}
```

Implemented by: CUDA (streams), Vulkan (queues / command buffers).
Not implemented by: CPU (synchronous dispatch), Reference (synchronous).
Future backends with stream-like primitives (Metal command queues, ROCm streams)
implement when added.

#### `BackendPressureSignals`

For backends with push-based pressure notification.

```rust
pub trait BackendPressureSignals: BackendRuntime {
    /// Register a callback that fires when memory pressure crosses
    /// `threshold` (as fraction of `total_bytes`, e.g. 0.85). The
    /// `hysteresis` prevents rapid re-fire as usage oscillates;
    /// `Relieved` fires when usage drops `hysteresis` below threshold.
    ///
    /// Callbacks run on whatever thread crossed the threshold. The
    /// backend MUST release internal locks before firing so the
    /// callback may safely re-enter the backend (e.g. to trigger
    /// eviction).
    fn register_pressure_callback(
        &self,
        threshold: f64,
        hysteresis: f64,
        callback: Box<dyn Fn(PressureKind) + Send + Sync>,
    ) -> Result<PressureCallbackId>;

    fn unregister_pressure_callback(&self, id: PressureCallbackId) -> bool;
}

pub enum PressureKind {
    /// Usage crossed above `threshold`.
    Crossed,
    /// Usage dropped below `threshold - hysteresis`.
    Relieved,
}
```

Implemented by: Vulkan (`VK_EXT_memory_budget` callbacks via
vulkane). Not implemented by: most others. Selectors that want push
notifications check `BackendId` against the registry of implementors.

### Tier 3 — Optional (introspection / diagnostics)

Tier 3 is purely for diagnostics, debugging, and user-facing
inspection. No selector / planner / optimizer takes a dependency on
Tier 3.

#### `BackendDiagnostics`

```rust
pub trait BackendDiagnostics {
    /// Human-readable vendor + device identifier.
    /// E.g. "NVIDIA GeForce RTX 4070" / "AMD Ryzen 9 7950X" /
    /// "Apple M3 Pro".
    fn device_name(&self) -> String;

    /// Driver / runtime version string.
    /// E.g. "CUDA 12.4" / "Vulkan 1.3.290" / "AOCL 4.2".
    fn driver_version(&self) -> Option<String>;

    /// Vendor-reported peak compute throughput in FLOPS, by dtype.
    /// Best-effort — many backends report nothing.
    fn peak_throughput(&self, dtype: DType) -> Option<u64>;

    /// Vendor-reported peak memory bandwidth in bytes/sec.
    fn peak_memory_bandwidth(&self) -> Option<u64>;
}
```

For now: implemented opportunistically; absent until a consumer
needs it (e.g. a `fuel-info` subcommand that prints "what backends
are visible and what can they do").

## Mandatory vs optional — summary

```text
Tier 1 (mandatory)        Tier 2 (conditional)    Tier 3 (optional)
─────────────────         ───────────────────     ─────────────────
BackendIdentity           BackendStreams          BackendDiagnostics
BackendCapabilityProvider BackendPressureSignals
BackendRuntime
```

The picker, optimizer ranker, route picker, executor, and planner
only consume Tier 1 directly. Tier 2 is consumed via feature-detect
(`if let Some(streams) = backend.as_streams()`); Tier 3 is consumed
by tools and diagnostics, never by the dispatch hot path.

## Current compliance

Snapshot of where each backend stands relative to the v0.3 ideal as
of 2026-06-07. This table is descriptive (what exists today), not
prescriptive (what must exist); the gaps are the migration backlog.

| Backend | Identity | Caps | Runtime | Streams | Pressure | Diag |
| ------- | -------- | ---- | ------- | ------- | -------- | ---- |
| `fuel-cpu-backend` | ✅ | ✅ | ❌ planned | n/a | ❌ planned | ❌ |
| `fuel-aocl-cpu-backend` | ✅ | ✅ | ❌ planned | n/a | ❌ planned | ❌ |
| `fuel-mkl-cpu-backend` | ✅ | ✅ | ❌ planned | n/a | ❌ planned | ❌ |
| `fuel-cuda-backend` (baracuda) | ✅ | ✅ | ✅ (baracuda alpha.66 `cuMemGetInfo`) | partial | ❌ | ❌ |
| `fuel-vulkan-backend` (vulkane) | ✅ | ✅ | partial inherent | ✅ inherent | ✅ inherent | ❌ |

`fuel-reference-backend` v0.4 retirement note: the Reference backend
was removed from the dispatch system in v0.4 (2026-06-07). It no
longer has a `BackendId` variant, doesn't participate in the picker
or Judge, and isn't part of the contract surface. The crate remains
as a test-oracle utility — see §The always-built coverage commitment.

Legend:

- ✅ — implements the trait fully today.
- ⏳ — actively in progress (e.g. a request out to a sibling crate).
- ❌ planned — known gap, migration scheduled in upcoming work.
- partial — provides the signal via inherent methods but not via
  the trait surface; migrate to trait impl as part of v0.3 rollout.
- n/a — Tier 2 trait doesn't apply (e.g. CPU has no stream concept).

Migration order recommended by this contract:

1. **`BackendRuntime` for CPU + Vulkan + Reference** — ship the
   trait, move Vulkan's inherent methods behind it, add CPU OS-query
   impl, add Reference synthetic-∞ impl. Single commit. *(In flight
   immediately following this doc.)*
2. **`BackendRuntime` for CUDA** — ✅ done (baracuda alpha.66). The
   `cuMemGetInfo` wrappers (`baracuda_driver::mem_get_info` +
   `Device::vram_info`) landed, so `CudaDevice::available_bytes` /
   `total_bytes` report real device memory; a failed driver query
   maps to `None` and selectors fall back to static cost as before.
   The query pushes the device's context for the duration of the call
   so it is correct when polled off the dispatch thread.
3. **`BackendStreams` for CUDA + Vulkan** — formalize stream / queue
   counts as trait methods. Trim out the inherent methods that
   selectors currently touch directly.
4. **`BackendPressureSignals` for Vulkan** — promote vulkane's
   pressure-callback API to the trait. CUDA impl waits on baracuda
   exposing notifications (lower priority than `MemGetInfo`).
5. **Tier 3 diagnostics** — opportunistic, no scheduled migration;
   add when a consumer materializes.

## Capability requirements — inference vs training

Different workloads load-bear on different parts of the contract.
A backend complete for inference may be insufficient for training,
and vice versa.

### Required for inference

A backend is **inference-complete** when it implements:

- All Tier 1 (`BackendIdentity`, `BackendCapabilityProvider`,
  `BackendRuntime`).
- Forward kernels for the inference op set: matmul / fused-matmul,
  the activation set (silu / gelu / relu / softmax / etc.), the
  normalization set (rms_norm / layer_norm), attention (FlashAttention
  or PagedAttention), elementwise unary + binary, reductions, casts,
  conv2d if vision is in scope.
- `Op::Copy` kernels into and out of CPU (the realize-root D2H path
  depends on this).
- Quantized matmul (Q4_0 / Q4_K_M / Q8_0 at minimum) if the backend
  serves quantized inference workloads.

`BackendStreams` is helpful for inference (pipeline depth, multi-
request batching) but not strictly required — synchronous dispatch
suffices for single-request inference.

### Additional requirements for training

A backend becomes **training-complete** when it adds, on top of
inference-complete:

- Backward kernels for every forward kernel it ships, OR a fused-op
  backward (e.g. `Op::Fused(POWI_BACKWARD)`, `Op::Fused(FLCE_BACKWARD)`).
  The autograd machinery in fuel-graph emits backward ops; backends
  that lack backward kernels fall back to CPU at every gradient
  edge, which is usually too slow to be useful.
- In-place op variants (`Op::AddInplace`, `Op::SubInplace`, etc.)
  for optimizer-step efficiency. Optional in v1 — without them,
  every parameter update allocates a fresh storage; with them,
  Adam-style updates mutate in place. The architecture's
  graph-tracked version safety (see
  [project_graph_tracked_version_safety.md](../../C:/Users/cires/.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_graph_tracked_version_safety.md))
  makes this safe.
- `BackendStreams::flush()` — gradient accumulation across multiple
  micro-batches requires barrier semantics. Without `flush()` the
  training loop cannot reliably sequence backward, optimizer step,
  and zero-grad in order.
- Deterministic-seed RNG for backends running dropout / weight-init.
  Currently scoped via [project_baracuda_alpha_31_integration.md](../../C:/Users/cires/.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_baracuda_alpha_31_integration.md)
  pattern (`set_seed` + `get_current_seed` on the device handle).

`BackendPressureSignals` is helpful for training (large activation
checkpoints generate variable memory pressure) but Tier 2 — backends
without it fall back to `BackendRuntime::would_fit` polling.

### Pluggability target

A new backend (e.g. ROCm, Metal compute, TPU) is "Fuel-pluggable"
when it implements Tier 1. That suffices for it to be part of the
optimizer's plan-time enumeration and the executor's dispatch. Tier
2 / 3 / training-completeness are progressive enhancements; they
unlock specific features but don't block participation.

## What the v0.3 trait surface enables

For the picker arc (Phases 1–5):

- **Picker 1 (optimizer ranker)** — already complete; queries
  `BackendCapabilityProvider::capabilities()` and the binding-table
  for op-coverage filtering. v0.3 changes nothing here.
- **Picker 2 (`RuntimeSelector`)** — `VramPressureSelector`,
  `MemoryPressureSelector`, future `LoadAwareSelector` all become
  backend-agnostic: they query Tier 1 / Tier 2 traits, never
  branch on `BackendId`.
- **Coupled-cost composition (Phase 2.3)** — derives transfer costs
  from `BackendCapabilities::transfer_paths` (already in Tier 1)
  and bandwidth estimates from peak-throughput diagnostics if
  available, falling back to probed values. Buildable today.

For the training arc:

- **Gradient checkpointing decisions** — read `available_bytes` to
  decide whether to materialize an activation now or recompute on
  backward. Today's checkpoint heuristics are static; reading the
  trait lets them adapt to live memory state.
- **Mixed-precision casting** — `BackendCapabilities::op_dtype_support`
  already determines admissibility; v0.3 doesn't change this.

For the multi-device / multi-host arc (out of scope today, but
worth naming):

- **Backend hotplug** — `SystemTopology::generation` already detects
  topology changes; per-backend registration / deregistration goes
  through the same path. Tier 1 backends just need to keep their
  `BackendRuntime` impl returning sane values across their lifecycle.
- **Cross-host transfer paths** — when a future backend exposes
  network-attached devices, `BackendCapabilities::transfer_paths`
  gains entries for the remote `DeviceLocation`s and the optimizer
  plans across them without further contract changes.

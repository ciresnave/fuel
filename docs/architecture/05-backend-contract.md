# Backend contract

**Status**: v0.2 (draft, 2026-05-09). v0.2 changes: (1) per-kernel error characteristics formalized as a `PrecisionGuarantee` structure with multiple optional bounds (replaces the binary OracleGrade concept); (2) the "Reference backend special status" section is replaced with "Per-kernel precision guarantees and the always-built coverage commitment"; (3) backend telemetry includes opt-in upload of locally-aggregated kernel-stat summaries to the project's telemetry server.

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
|---------|--------------|------------|
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

The fuel-reference-backend crate, where it exists today, becomes "the backend whose entire kernel set has `bit_stable_on_same_hardware: true`." It may continue to exist for clarity (some users prefer a single crate where every kernel is oracle-grade); architecturally its role is no longer special. Its kernels could equivalently live as `bit_stable`-tagged kernels inside fuel-cpu-backend; the choice is implementation convenience, not architectural commitment.

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

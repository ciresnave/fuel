# Row 137 step 6.3b: the JIT probe seam (design note, no code)

**Status:** design only. Measured at `origin/main@659ce448` (0.44.0, 2026-11-07) by reading `fuel-dispatch/src/jit_ingest.rs`, `jit_ingest_probe.rs`, `fkc/verify/{mod,bit_stability,invoker_cuda}.rs`, `device_registry.rs` and `fuel-cuda-dispatch/Cargo.toml`. Nothing was built or run. Owner Fuel 1. Parent: ROADMAP row 137 step 6 (`2026-11-03-row137-step6-plan.md`, plan note 4.3).

**DECIDED 2026-11-07 (PM ruling, not CireSnave's): `ProbeTensor`, the minimal public type. `KernelInvoker` and the verify API are NOT published for this (GAP-236 stays its own item). `ProbeTensor`/`ProbeDevice` carry no CUDA names, so FireLine can adopt the trait as a provider seam later.**

## 1. The problem 6.3a left

6.3a moved the JIT *loader* (`jit_cuda_load`) to `fuel-cuda-dispatch`. The JIT *ingestion* pipeline stayed in `fuel-dispatch` and still names CUDA directly, because it runs a candidate on a device and needs a reference to compare against. `fuel-dispatch` cannot depend on `fuel-cuda-dispatch` (the latter depends on the former), so moving those functions as they are would be a cycle: `jit_ingest.rs` calls them.

CUDA-named items in `fuel-dispatch` that this step must remove (all behind `feature = "cuda"`):

| Where | What it needs from the device |
|---|---|
| `jit_ingest_probe::reference_output` (candidate's own `decompose`) | upload probes, stamp the backend on emitted nodes, realize on a one-device registry, download |
| `jit_ingest_probe::reference_from_registered_recipe` (Fuel's registered recipe, lowered to primitives) | the same four things |
| `jit_ingest::verify_candidate_impl` (around line 1881) | `CudaInvoker::new(device, out_dtype, out_shape).with_params(..).invoke(&entry, &probe)`: run the candidate kernel |
| `jit_ingest::{verify_candidate, ingest_one, JitIngestionPipeline::start}` | take `&CudaDevice` / `CudaDevice` by value |
| `fkc/verify/invoker_cuda.rs`, `seed_cuda_ledger.rs`, `seed_cuda_accuracy.rs`, `harness.rs` | `CudaInvoker` and live-device tests |
| `DeviceRegistry::with_cuda(fuel_cuda_backend::CudaDevice)` | the registry arm both reference functions use (row 137 S3 step 3 owns this arm) |

Everything else in those two files (probe generation, verdict logic, ledger, kiss-ref advisory diff, `advisory_ulp_band`) is device-free.

## 2. What the device must do (the narrowest interface)

Only three operations touch a device, and none needs a CUDA type in its signature:

1. **Invoke a candidate kernel** on host-supplied probe tensors and return host bytes.
2. **Realize a graph** whose leaves are host-supplied tensors and return the sink as host bytes (this serves both references: they differ only in how the graph is built).
3. **Say which backend it is**, so the neutral code stamps `set_target_backend` with the right `BackendId`.

Proposed trait, defined in `fuel-dispatch` (neutral, no backend feature):

```rust
pub struct ProbeTensor { pub dtype: DType, pub shape: Vec<usize>, pub bytes: Vec<u8> }

pub trait ProbeDevice: Send + Sync {
    fn backend(&self) -> BackendId;
    fn invoke_candidate(&self, entry: &BindingEntry, params: &OpParams,
                        inputs: &[ProbeTensor], out_dtype: DType, out_shape: &[usize])
                        -> Result<ProbeTensor>;
    fn realize(&self, graph: Arc<RwLock<Graph>>, sink: NodeId,
               inputs: &[(NodeId, ProbeTensor)]) -> Result<ProbeTensor>;
}
```

`reference_output` and `reference_from_registered_recipe` become device-free: they build the graph, stamp `device.backend()`, call `realize`. `verify_candidate`, `ingest_one` and `start` take `&dyn ProbeDevice` / `Arc<dyn ProbeDevice>`.

`fuel-cuda-dispatch` (feature `jit`) gets `CudaProbeDevice { device: CudaDevice }` implementing it: `invoke_candidate` is today's `CudaInvoker` body; `realize` is today's steps (a) upload, (e) bind to the storage cache, (f) `PipelinedExecutor::realize_with_devices(.., DeviceRegistry::new().with_cuda(device))`, (g) download. `CudaInvoker`, `seed_cuda_ledger.rs`, `seed_cuda_accuracy.rs` and the CUDA arm of `harness.rs` move with it. The CUDA-gated live tests in `jit_ingest.rs` (about 14, each `CudaDevice::new(0)` under `gpu-run`) move to `fuel-cuda-dispatch/tests`, where the 6.3a live test already is.

## 3. Findings that change the plan

- **`HostTensor` and `KernelInvoker` are private.** `fkc::verify` is a private module tree (GAP-236, "unpublished verify API"; its expiry lives in GAP-236 and Unpopped's handback guard). The trait therefore cannot reuse them without publishing the verify API. The proposal above defines a small new public `ProbeTensor` and converts at the edge (`HostTensor` has the same three fields). Reusing `KernelInvoker` is the smaller diff but changes GAP-236's subject; **decision for the PM**.
- **Vulkan and CPU are already shaped like this.** `invoker_cpu.rs` and `invoker_vulkan.rs` exist beside `invoker_cuda.rs`. A `ProbeDevice` makes the JIT pipeline backend-agnostic in principle; the first and only implementer is CUDA, because the JIT synthesizer today is baracuda (CUDA). Do not add a Vulkan implementer in this step.
- **`DeviceRegistry::with_cuda` stays a CUDA arm in `fuel-dispatch` until S3 step 3.** `CudaProbeDevice::realize` can only call it from above `fuel-dispatch` if `DeviceRegistry` and `realize_with_devices` stay public, as they are now. That is a dependency on S3 step 3 not breaking them, not a reason to wait.
- **FireLine overlap.** The Synthesizer side of the seam is FireLine's (`fireline-synth`); this trait is on the *verification* side, which is Fuel-internal (it is how Fuel checks what a provider returned), so it is not a FireLine interface and does not wait for wave 1. If the provider-crate plan later moves the CUDA backend out of core, `ProbeDevice` is the shape a provider's verification hook would take; that is a note for that plan, not a commitment.

## 4. Order and gates

1. PR A: add `ProbeTensor` / `ProbeDevice`; make the two references and `verify_candidate`/`ingest_one`/`start` generic over it; keep `CudaProbeDevice` in `fuel-dispatch` behind `cuda` for one step so the diff is only the signature change. Behaviour-preserving; red-first sabotage sibling: a stub `ProbeDevice` that returns wrong bytes must produce a `Fail`, and one that errors must produce the same `resolve_reference_failure` outcome as today. These tests need no GPU.
2. PR B: move `CudaProbeDevice`, `CudaInvoker`, `seed_cuda_*`, the harness CUDA arm and the CUDA live tests to `fuel-cuda-dispatch`; drop `fuel-dispatch`'s `cuda`-gated `jit_ingest*` imports. Breaking if `fuel-dispatch`'s public `verify_candidate`/`ingest_one`/`start` change signature (second number; the PM allocates).
3. Verification for B: compile through a `cuda-build.ps1` slot (about 56 to 93 minutes cold, detached), then the moved live tests under `gpu-run`. **Nothing in CI compiles the CUDA arm**, so until that run the move is unverified; the PR says so.

Measure on the way out (row 137 rule): count of `cfg(feature = "cuda")` lines and `fuel_cuda_backend` references in `jit_ingest.rs`, `jit_ingest_probe.rs`, `fkc/verify/` before and after.

## 5. Not verified

The exact count of CUDA-gated live tests (about 14 by `cfg` lines, not by `#[test]`); whether `DeviceRegistry` is re-exported publicly (read from the definition only); whether anything outside the workspace calls `verify_candidate`/`ingest_one`; the effect on `fuel-dispatch`'s public API listing (no semver-check run).

# Spec B — Candidate-kernel ingestion service (verify → adopt/reject)

**Date:** 2026-07-13
**Status:** design (approved in brainstorm; pending implementation plan)
**Program:** JIT auto-trigger (the "Fuel auto-synthesizes / auto-adopts kernels" end-state)
**This is Spec B of a decomposition.** Spec A (JIT region discovery + synthesis request) and
Spec 2 (background scheduling of A + proactive re-optimize + in-use-safe trimming) are separate,
sequenced specs. Build order: **B → A → 2.**

---

## 1. Motivation

Fuel already has the *consumption* half of a JIT-on-request loop: `adopt_runtime_fused` registers a
synthesized fused op's recipe + kernel; `runtime_fused_pathfinder` + `offer_runtime_fused_arm` make
the optimizer offer it as an `Op::Branch` arm; the executor picks it. What is missing is the
*trigger* — nothing discovers a region, requests a kernel, and adopts the result. (Confirmed:
`JitRequest` is constructed only in a unit test; the whole path is behind the non-default `jit`
feature.)

A brainstorm on wiring that trigger surfaced a cleaner factoring than a single "discover → request →
adopt" loop. Two activities with **different triggers and different lifetimes** were conflated:

- **Group A — discovery + request** is *graph*-triggered and speculative: "here is a spot that could
  be improved, go build/ask for a kernel." Optional.
- **Group B — ingest + verify + adopt** is *kernel*-triggered and obligatory: "a kernel arrived — is
  it correct, put it in rotation, and if not, tell whoever sent it why." It should fire on **any**
  arriving kernel, not only as a response inside a request loop.

Group B already exists in spirit: this project's `fuel-dispatch/src/fkc/verify/seed_cuda_ledger.rs`
*is* Group B for ahead-of-time kernels — it fires on every registered kernel, verifies its contract's
claims (bit-stability; `max_ulp` vs a CPU reference), and records the verdict in the FKC ledger. Spec
B **generalizes that** so it also ingests a *received fused* kernel (verify against its decompose),
and makes any producer — a JIT synthesis response, an unsolicited provider push, an AOT contract — a
source that feeds the same service.

Why B first: it is the reusable core, it is useful standalone (any kernel source), and building it on
a proven foundation de-risks A. It also owns the background write (adopt), so the one real concurrency
prerequisite lands here where it belongs.

---

## 2. Goal & scope

**Goal.** A source-agnostic ingestion service: given a `CandidateKernel`, verify its contract's
declared claims against a reference, ledger the verdict, **adopt on pass** / **reject-with-report-to-
the-provider on fail** — governed so verification never fans out or swamps live inference.

**In scope (Spec B).**

- The `CandidateKernel` seam and the `IngestOutcome`.
- Verification of a received kernel against its decompose (fused) or a cross-backend/host reference
  (primitive), reusing the FKC verify machinery + writing a ledger entry.
- Adoption on pass (`adopt_runtime_fused`); structured rejection + provider feedback on fail.
- Resource governance: a bounded ingestion queue, a single idle-aware verify worker (concurrency
  configurable, default 1), backpressure via feedback.
- The lock-nesting prerequisite (make background adopt safe against the optimizer's pathfinder read).
- CUDA backend.

**Explicitly NOT in Spec B.**

- Region discovery, `JitRequest` construction, calling the `Synthesizer` (Spec A).
- Proactive re-optimize of plans that could use a newly adopted kernel; in-use-safe generational
  trimming (Spec 2).
- Vulkan synthesis/verify (later; the seam is backend-parameterized so it does not preclude it).

---

## 3. Architecture

```text
   producers (any source)                Group B (this spec)
   ---------------------                  --------------------
   Spec A: JIT discovery ─┐
   unsolicited push ──────┼─►  enqueue(CandidateKernel)  ─►  [ bounded queue ]
   AOT contract (existing)┘                                        │
                                                                   ▼
                                          idle-aware single worker (concurrency = 1 default)
                                                                   │  pulls one when GPU load is low
                                                                   ▼
                                   verify(candidate)  ── reference = realized decompose (fused)
                                          │                        or cross-backend/host ref (primitive)
                                          │            candidate  = raw KernelRef via CudaInvoker + op_params
                                          │            checks     = repeat-call bit-stability
                                          │                       + declared precision claims
                                          │            → ledger upsert (pass|fail)
                                 pass ─────┴───── fail
                                   │               │
                          adopt_runtime_fused   RejectionReport → feedback.on_rejected(report)
                                   │               │
                          IngestOutcome::Adopted   IngestOutcome::Rejected
```

The synchronous **verify-one-candidate core** is the heart; the queue + idle-aware worker is a thin
resource-governance wrapper around it.

---

## 4. Components

Each is small, one-purpose, and independently testable.

### 4.1 `CandidateKernel` (the source-agnostic seam)

```rust
CandidateKernel {
    entry_point: String,                 // stable name (from LinkEntry / contract)
    kernel: KernelRef,                   // already-loaded callable (producer loads PTX before enqueue)
    contract: ParsedFkcContract,         // accept / op_params / precision / determinism (+ decompose)
    decompose: Option<PatternNode>,      // Some for a fused op (the region = the recipe); None for a primitive
    operands: Vec<OperandDesc>,          // shapes/dtypes in bind order (drives probe synthesis)
    dtypes: Vec<DType>,                  // the binding-key dtypes to adopt under
    backend: BackendId,
    feedback: Option<Arc<dyn ProviderFeedback>>,  // where a rejection (or "queue full") is reported
}
```

- **Interface:** value type handed to `enqueue`.
- **Depends on:** `fuel_graph::jit::PatternNode`, the FKC contract parser, `OperandDesc`.
- **Note:** the producer loads the artifact into a `KernelRef` before enqueueing (the CUDA PTX load
  is `jit_cuda_load::load_synth_kernel`). B is load-agnostic — it receives a live `KernelRef`.

### 4.2 `ProviderFeedback` (the notify-on-refuse channel)

```rust
trait ProviderFeedback: Send + Sync {
    fn on_rejected(&self, report: &RejectionReport);
    fn on_adopted(&self, entry_point: &str, id: FusedOpId) {}   // default no-op
}
```

- The JIT synthesizer gets a **default-no-op** `on_rejected` added to the `Synthesizer` trait (in
  `fuel-kernel-seam`, Fuel-owned) — non-breaking; baracuda's existing impl still compiles and can
  override later. **Cross-project coordination point** (propose-first per project norms): the
  eventual baracuda override that consumes the rejection is a baracuda ask, not built here.
- An unsolicited pusher supplies its own `ProviderFeedback`.

### 4.3 `RejectionReport`

```rust
RejectionReport {
    entry_point: String,
    failed_claim: &'static str,   // "bit_stable_on_same_hardware" | "max_ulp" | "vs_decompose" | "invoke_error" | "queue_full"
    detail: String,               // measured vs bound, the diverging element, the invoke error, etc.
    ledger_record: Option<LedgerRecord>,  // the fail record written (None for queue_full / pre-probe errors)
}
```

- Actionable for the provider: which claim failed, by how much, on which probe.

### 4.4 Verify-one-candidate core (`#[cfg(feature = "cuda")]`)

`fn verify_candidate(cand: &CandidateKernel, device: &CudaDevice) -> VerifyVerdict`

1. **Probe synthesis** from `cand.operands` (generalizes `seed_cuda_ledger::build_cuda_probe` — keyed
   on the region/op shapes rather than a single OpKind).
2. **Reference output.**
   - Fused (`decompose = Some`): realize the decompose region on the probe inputs → the ground-truth
     output. Re-emitting a `PatternNode` region as a primitive subgraph is the existing "synthesized
     op's decompose (the region re-emitted)" operation (`fuel-graph/src/jit.rs`); bind the region's
     inputs to probe consts and realize on `device`. (For a JIT candidate, Spec A already holds the
     original real subgraph and MAY pass it directly to skip re-emit; for an unsolicited push, the
     `PatternNode` from the contract is the only form, so re-emit is the general path.)
   - Primitive (`decompose = None`): the cross-backend/host reference per the declared claim — exactly
     what the seeding harness does for `max_ulp` (CUDA candidate vs CPU reference).
3. **Candidate output.** Invoke `cand.kernel` via `CudaInvoker` + the contract's `op_params` (the same
   raw-kernel invocation the seeding harness already uses). **No tentative adoption needed.**
4. **Checks.** Repeat-call bit-stability (≥16 iters, byte-identical) **always**; plus each precision
   claim the contract declares (`bit_stable` / `max_ulp` / `max_relative` / `max_absolute`) vs the
   reference. Verdict + evidence.
5. **Ledger.** `upsert` a `pass`/`fail` record keyed on `(backend, dtypes, kernel_revision_hash,
   claim)` — the exact FKC gate key, so a passing verify un-gates the kernel for placement.

- **Depends on:** `CudaInvoker`, `verify_bit_stability`, `verify_precision_bound`, `CpuInvoker` (for
  primitive references), the realize path (for decompose references), the ledger.
- **Never-panic:** any probe/invoke failure → a `fail` verdict with the error, never a crash
  (`catch_unwind`, as the seeding harness does).

### 4.5 Ingestion queue + idle-aware worker

- **`enqueue(candidate) -> Result<(), Backpressure>`** — non-blocking; pushes onto a **bounded**
  channel. Full → return `Backpressure` and fire `feedback.on_rejected("queue full, retry later")`
  (no silent drop).
- **Worker thread** (started once, owns the `CudaDevice` handle for verification):
  - Concurrency is **1 by default** (config: a small semaphore) — never more than N GPU verifications
    at once.
  - **Idle-aware:** before pulling the next candidate, consult the per-device in-flight-op counter
    (Step E's load signal, `fuel-dispatch`'s `inflight_count`); if load ≥ threshold, wait/back off.
    Best-effort — verification may lag under sustained load, which is fine (**adoption is an
    optimization, never required for correctness**).
  - Per candidate: `verify_candidate` → pass ⇒ `adopt_runtime_fused` + `feedback.on_adopted`; fail ⇒
    `feedback.on_rejected(report)`.
- **Interface:** `IngestionService { enqueue(candidate), shutdown() }`; verdicts arrive via
  `ProviderFeedback` (async), not a return value.

### 4.6 Lock-nesting prerequisite

The worker's `adopt_runtime_fused` takes the global-bindings **write** lock on a background thread.
Today `runtime_fused_pathfinder` calls `fused_kernel_available` (a **read** on the same lock) while
`optimize_graph` already holds a read guard — a nested read that a background write queued between the
pathfinder's two reads could deadlock (writer-preferring `RwLock`). Fix, per
`runtime_fused_kernels.rs:105-113`: thread the already-held `&KernelBindingTable` into the pathfinder
via `OptimizationContext` (the pathfinder already receives one) and have its availability gate read
from that instead of re-acquiring `global_bindings()`. Self-contained; verified by the pathfinder's
existing arm-offering tests plus a new test that the gate reads the threaded table.

---

## 5. Error posture

Never-panic throughout. Decline-to-load, un-probeable operands, invoke error, verify divergence,
queue-full — all become a structured verdict/report and (where a provider is attached) a feedback
call. No path aborts a realize or the worker. A kernel that fails verification simply is not adopted;
the region/op stays on its existing (primitive or CPU) kernel — always correct, just unaccelerated.

---

## 6. Feature gating

- The `CandidateKernel` / `ProviderFeedback` / `RejectionReport` types and the queue are feature-light
  (usable by a future Vulkan path).
- The verify core + CUDA invocation + adopt live behind `jit` (+ `cuda` for the device probe). The
  `Synthesizer::on_rejected` addition is in `fuel-kernel-seam` behind the same envelope gating as the
  rest of that crate.

---

## 7. Testing

**Unit (no GPU):**

- Ingest a mock `CandidateKernel` whose mock kernel matches a known decompose → `Adopted` + a `pass`
  ledger record + `on_adopted` fired.
- A mock kernel that diverges from the decompose → `Rejected` + `on_rejected` called with a report
  naming the failed claim; **not** adopted.
- A mock kernel that panics on invoke → `Rejected` (invoke_error), no crash.
- Queue full → `enqueue` returns `Backpressure` + `on_rejected("queue full")`.
- Idle-gate: with a stubbed load signal above threshold, the worker defers; below, it proceeds.
- Lock-prereq: the pathfinder offers arms correctly reading the threaded binding table.

**GPU (`jit` + `cuda`, `#[ignore]`):**

- **The validating test.** Feed B **baracuda's interleaved `rope_apply` as a candidate for the
  rotate-half rope region** (decompose = the rope rotate-half `PatternNode`). B must **reject** it
  with a `vs_decompose` precision report. This is exactly the correctness bug this program hit on
  2026-07-13 (interleaved ≠ rotate-half), now caught **automatically** by the ingestion service — the
  strongest possible proof the verify-before-adopt gate does its job.
- A *correct* fused candidate for a simple region (e.g. an add→mul chain) → verified byte-exact vs the
  realized decompose → `Adopted`; a graph containing the region then realizes byte-identical, now via
  the one adopted fused op (checked by the pathfinder offering + the executor picking the arm).

---

## 8. Open items / boundaries to Spec A and Spec 2

- **Spec A** (next): region discovery (maximal in-grammar primitive regions, no fused kernel,
  bounded, single-consumer interior), `region_to_request` (subgraph → `PatternNode` + operands),
  calling the `Synthesizer`, loading the artifact, and emitting a `CandidateKernel` into B. Also where
  the "does baracuda's synthesizer accept the ~7-op rope region" question is answered.
- **Spec 2** (later): running A on a background/idle trigger; the plan↔pattern registry so a newly
  adopted kernel proactively re-optimizes the plans whose op-pattern it matches; in-use-safe
  generational trimming (an old arm keeps serving live work until it drains, then trims). B's queue is
  *not* that scheduler — B governs only its own verify concurrency.
- **Cross-project:** the `Synthesizer::on_rejected` default-no-op is added Fuel-side now; a baracuda
  override that actually consumes the rejection (to stop re-offering a rejected kernel / improve it)
  is a propose-first baracuda ask, out of scope here.

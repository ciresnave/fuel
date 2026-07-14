# Candidate-Kernel Ingestion Service (Spec B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-agnostic service that ingests a received kernel, verifies it against its decompose (or a reference), and adopts it on pass / rejects-with-provider-feedback on fail — governed by a bounded queue + a single idle-aware verify worker so it never swamps live inference.

**Architecture:** A synchronous `verify_candidate` core (reuses the FKC verify path from `seed_cuda_ledger.rs`) wrapped by a bounded ingestion queue + one background worker (concurrency 1, defers to GPU-idle via the in-flight-op load signal). On pass the worker calls `adopt_runtime_fused`; on fail it builds a `RejectionReport` and calls the producer's `ProviderFeedback`. Includes the lock-nesting prerequisite that makes the background adopt safe against the optimizer pathfinder's read.

**Tech Stack:** Rust (edition 2024), `fuel-dispatch`, `fuel-kernel-seam`, `fuel-graph`, `fuel-cuda-backend` (baracuda FFI). Reference doc: `docs/superpowers/specs/2026-07-13-jit-candidate-kernel-ingestion-spec-b-design.md`.

## Global Constraints

- **Build scoping:** always `cargo ... -p <crate>`, never workspace-wide. One cargo invocation at a time. CUDA builds need a VS Developer shell (vcvars64 so nvcc finds cl.exe); helper: `C:\Windows\Temp\cuda_run.bat` (calls vcvars64, prepends CUDA v13.3 `bin` + cuDNN v9.23 to PATH, runs the passed cargo command, appends `CUDA_RUN_EXITCODE`). Invoke: `cmd //c 'C:\Windows\Temp\cuda_run.bat' <cargo args...>`.
- **Feature gating:** verify/invoke/adopt code is `#[cfg(feature = "cuda")]`; the seam types + queue are feature-light. The `jit` feature (`fuel-dispatch/Cargo.toml`) pulls `dep:fuel-kernel-seam` + `dep:baracuda-kernels-types`.
- **Never panic on production paths.** Any probe/invoke/load failure becomes a `fail` verdict / `RejectionReport`, never a crash (`std::panic::catch_unwind(AssertUnwindSafe(...))`, as `seed_cuda_ledger.rs` does).
- **TDD:** write the failing test, watch it fail, minimal code to green, commit. GPU tests are `#[ignore]` (run manually on the RTX 4070).
- **Ledger discipline:** always `VerificationLedger::upsert`, never `push` (the embedded ledger recompiles → a naive re-run appends duplicates).
- **Adoption is an optimization, never required for correctness** — a rejected/failed candidate leaves the region on its existing kernel.

---

## Prerequisites / starting context (read FIRST)

- **Branch.** This plan builds on branch `capturedrun-4b-resume` (commits `a127c190`..`525f93f4`), **NOT yet merged to `main`**. The verify-invoke template this plan reuses — `fuel-dispatch/src/fkc/verify/seed_cuda_ledger.rs` — and the 219-record verified ledger were added on this branch. **Start from this branch** (or after it merges to main). On `main` alone the foundation does not exist. Do the work on a fresh branch off `capturedrun-4b-resume`.
- **Auto-loaded context.** A fresh session loads `CLAUDE.md` (build/GPU discipline; environment: Windows 11, RTX 4070, CUDA 13.3, cuDNN v9.23) and the per-machine memory index (`~/.claude/.../MEMORY.md` — the rope-convention finding, CapturedRun, and seeding memories). Read the **design doc** `docs/superpowers/specs/2026-07-13-jit-candidate-kernel-ingestion-spec-b-design.md` for the "why" before Task 1.
- **CUDA build helper.** `C:\Windows\Temp\cuda_run.bat` (Global Constraints) is scratch and may not survive a reboot/temp-clear. If missing, recreate it verbatim:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64;%PATH%"
cd /d C:\Projects\fuel
%* > C:\Windows\Temp\cuda_run.log 2>&1
echo CUDA_RUN_EXITCODE: %ERRORLEVEL% >> C:\Windows\Temp\cuda_run.log
```

  Invoke as `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit ...`; read `C:\Windows\Temp\cuda_run.log` for output + the `CUDA_RUN_EXITCODE` line (the background-run's own exit is the batch's, not cargo's).
- **Baseline check (before Task 1).** Confirm the starting point is green: `cargo test -p fuel-dispatch --lib` (default), and `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo build -p fuel-dispatch --features cuda,jit` exits 0. If `--features jit` doesn't build on its own today, fix that baseline first — it's a prerequisite, not part of any task.
- **No synthesizer dependency.** Spec B's tests use EXISTING kernels as candidates (the CUDA `add_f32` wrapper; baracuda's `rope_apply_f32`), NOT baracuda's JIT synthesizer (that's Spec A). So no baracuda alpha bump is needed for this plan.

---

## File structure

- **Create** `fuel-dispatch/src/jit_ingest.rs` — the ingestion module: `CandidateKernel`, `ProviderFeedback`, `RejectionReport`, `IngestOutcome`, `verify_candidate`, `ingest_one` (sync verify+adopt/reject), and `IngestionService` (queue + worker). Module declared `#[cfg(feature = "jit")]` in `lib.rs`.
- **Create** `fuel-dispatch/src/jit_ingest_probe.rs` — probe synthesis from `OperandDesc` + the decompose-reference realize helper (the two pure-ish pieces, split out to keep `jit_ingest.rs` focused). `#[cfg(feature = "cuda")]`.
- **Modify** `fuel-kernel-seam/src/lib.rs` — add the default-no-op `Synthesizer::on_rejected`.
- **Modify** `fuel-dispatch/src/lib.rs` — declare `mod jit_ingest;` / `jit_ingest_probe` under the right cfgs.
- **Modify** `fuel-dispatch/src/driver.rs` (or wherever `OptimizationContext` is defined) + `fuel-dispatch/src/runtime_fused_pathfinder.rs` + `fuel-dispatch/src/runtime_fused_kernels.rs` — the lock-nesting prerequisite (thread the binding table into the pathfinder's availability gate).

Reference (read before starting): `fuel-dispatch/src/fkc/verify/seed_cuda_ledger.rs` (the verify-invoke template this reuses), `fuel-dispatch/src/jit_adopt.rs` (`adopt_from_response`), `fuel-dispatch/src/runtime_fused_kernels.rs` (`adopt_runtime_fused`, `fused_kernel_available` + its lock note at :97-113), `fuel-graph/src/runtime_fused.rs` (`decompose_region`, `emit`), `fuel-dispatch/src/dispatch.rs` (`inflight_count`).

---

## Task 1: Lock-nesting prerequisite (thread binding table into the pathfinder gate)

Do this FIRST — it is the foundation the background worker's adopt depends on, and it is independent of the ingestion types.

**Files:**
- Modify: `fuel-dispatch/src/runtime_fused_kernels.rs` (add a table-taking variant of the availability check)
- Modify: `fuel-dispatch/src/runtime_fused_pathfinder.rs` (call the new variant)
- Modify: `fuel-dispatch/src/driver.rs` (thread `&KernelBindingTable` into `OptimizationContext` if not already reachable there)

**Interfaces:**
- Consumes: `OptimizationContext<'_>` (already passed to `Pathfinder::propose`), `KernelBindingTable`.
- Produces: `pub fn fused_kernel_available_in(table: &KernelBindingTable, id: FusedOpId, backend: BackendId) -> bool` — the non-relocking variant.

- [ ] **Step 1: Read the current call site + lock note.** Read `runtime_fused_kernels.rs:97-134` (the `fused_kernel_available` doc + body) and `runtime_fused_pathfinder.rs` (find where it calls `fused_kernel_available`). Confirm `OptimizationContext` already carries — or can carry — the `&KernelBindingTable` `optimize_graph` holds (`optimize.rs:197` passes `bindings_table: &KernelBindingTable`).

- [ ] **Step 2: Write the failing test** (in `runtime_fused_kernels.rs` tests): a table with a runtime-fused row → `fused_kernel_available_in(&table, id, backend)` returns true without touching `global_bindings()`; an empty table → false.

```rust
#[test]
fn fused_kernel_available_in_reads_the_passed_table_only() {
    let mut table = KernelBindingTable::new();
    // A runtime-fused id bound in THIS table (not the global one).
    let id = fuel_graph::runtime_fused::register_runtime_fused("t", relu_add_pattern()).unwrap();
    register_runtime_kernel_into(&mut table, id, &[DType::F32], BackendId::Cuda, noop_kernel);
    assert!(fused_kernel_available_in(&table, id, BackendId::Cuda));
    assert!(!fused_kernel_available_in(&KernelBindingTable::new(), id, BackendId::Cuda));
}
```

- [ ] **Step 3: Run it, watch it fail.** Run: `cargo test -p fuel-dispatch --lib fused_kernel_available_in -v`. Expected: FAIL (function/`register_runtime_kernel_into` not defined).

- [ ] **Step 4: Implement `fused_kernel_available_in`** — the body of `fused_kernel_available` but reading the passed `table` instead of `global_bindings()`:

```rust
/// Table-passing variant of [`fused_kernel_available`] — reads the CALLER'S
/// binding table (the one `optimize_graph` already holds) instead of
/// re-acquiring `global_bindings()`. This eliminates the nested read that a
/// background adopt's write could deadlock (see this fn's sibling's doc).
pub fn fused_kernel_available_in(table: &KernelBindingTable, id: FusedOpId, backend: BackendId) -> bool {
    default_kernel_registry().lookup(id, backend).is_some()
        || table.has_runtime_fused(id, backend)
        || static_binding_table_bridge_in(table, id, backend)
}
```

Add `static_binding_table_bridge_in(table, ...)` mirroring `static_binding_table_bridge` but scanning `table.iter_keys()` instead of `global_bindings().iter_keys()`. Add a test-only `register_runtime_kernel_into(table, id, dtypes, backend, kernel)` that writes a `BindingKey::RuntimeFused` row into a specific table (factor from `register_runtime_kernel`).

- [ ] **Step 5: Point the pathfinder at the new variant.** In `runtime_fused_pathfinder.rs`, replace its `fused_kernel_available(id, backend)` call with `fused_kernel_available_in(ctx.bindings(), id, backend)` (add a `bindings()` accessor on `OptimizationContext` returning the threaded `&KernelBindingTable` if one isn't already exposed).

- [ ] **Step 6: Run the pathfinder's existing tests + the new test.** Run: `cargo test -p fuel-dispatch --lib runtime_fused -v`. Expected: PASS (arm-offering behavior unchanged; the new test green).

- [ ] **Step 7: Commit.**

```bash
git add fuel-dispatch/src/runtime_fused_kernels.rs fuel-dispatch/src/runtime_fused_pathfinder.rs fuel-dispatch/src/driver.rs
git commit -m "fix(jit): pathfinder reads the threaded binding table (background-adopt-safe)"
```

---

## Task 2: Seam types — `ProviderFeedback`, `RejectionReport`, `Synthesizer::on_rejected`

**Files:**
- Create: `fuel-dispatch/src/jit_ingest.rs` (the type definitions portion)
- Modify: `fuel-dispatch/src/lib.rs` (`#[cfg(feature = "jit")] mod jit_ingest;`)
- Modify: `fuel-kernel-seam/src/lib.rs` (default-no-op `on_rejected` on `Synthesizer`)

**Interfaces:**
- Produces:
  - `pub struct RejectionReport { pub entry_point: String, pub failed_claim: &'static str, pub detail: String, pub ledger_record: Option<crate::fkc::verify::LedgerRecord> }`
  - `pub trait ProviderFeedback: Send + Sync { fn on_rejected(&self, report: &RejectionReport); fn on_adopted(&self, _entry_point: &str, _id: fuel_graph::registry::FusedOpId) {} }`
  - `pub enum IngestOutcome { Adopted(fuel_graph::registry::FusedOpId), Rejected(RejectionReport) }`

- [ ] **Step 1: Write the failing test** (in `jit_ingest.rs`): a `RejectionReport` round-trips its fields; a stub `ProviderFeedback` records the report it receives.

```rust
#[test]
fn provider_feedback_receives_the_report() {
    use std::sync::Mutex;
    struct Rec(Mutex<Vec<String>>);
    impl ProviderFeedback for Rec {
        fn on_rejected(&self, r: &RejectionReport) { self.0.lock().unwrap().push(r.failed_claim.into()); }
    }
    let rec = Rec(Mutex::new(vec![]));
    rec.on_rejected(&RejectionReport { entry_point: "k".into(), failed_claim: "max_ulp", detail: "d".into(), ledger_record: None });
    assert_eq!(rec.0.lock().unwrap().as_slice(), &["max_ulp".to_string()]);
}
```

- [ ] **Step 2: Run it, watch it fail.** Run: `cargo test -p fuel-dispatch --features jit --lib provider_feedback_receives_the_report -v`. Expected: FAIL (types undefined).

- [ ] **Step 3: Define the types** in `jit_ingest.rs` (the `Interfaces` block above). Declare the module in `lib.rs`: `#[cfg(feature = "jit")] mod jit_ingest;`.

- [ ] **Step 4: Add `Synthesizer::on_rejected`** in `fuel-kernel-seam/src/lib.rs` as a default-no-op so baracuda's existing impl still compiles:

```rust
    /// Fuel refused a kernel this synthesizer produced (verify-against-decompose
    /// failed, over budget on re-offer, etc.). Default no-op; a synthesizer MAY
    /// override to stop re-offering / to log. `report` is a JSON-ish detail
    /// string (Fuel's RejectionReport rendered) so this crate stays dependency-light.
    fn on_rejected(&self, _entry_point: &str, _report: &str) {}
```

(Note: the trait stays free of `fuel-dispatch` types — Fuel renders `RejectionReport` to a string at the call site.)

- [ ] **Step 5: Run the test + verify fuel-kernel-seam still builds.** Run: `cargo test -p fuel-dispatch --features jit --lib provider_feedback_receives_the_report -v` then `cargo build -p fuel-kernel-seam`. Expected: PASS + clean build.

- [ ] **Step 6: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs fuel-dispatch/src/lib.rs fuel-kernel-seam/src/lib.rs
git commit -m "feat(jit): ingestion seam types (CandidateKernel feedback/report) + Synthesizer::on_rejected default"
```

---

## Task 3: `CandidateKernel` + probe synthesis from `OperandDesc`

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (add `CandidateKernel`)
- Create: `fuel-dispatch/src/jit_ingest_probe.rs` (`probe_from_operands`)

**Interfaces:**
- Produces:
  - `pub struct CandidateKernel { pub entry_point: String, pub kernel: crate::kernel::KernelRef, pub op_params: crate::kernel::OpParams, pub decompose: Option<fuel_graph::jit::PatternNode>, pub operands: Vec<baracuda_kernels_types::OperandDesc>, pub dtypes: Vec<fuel_ir::DType>, pub kernel_revision_hash: u64, pub declared: crate::fused::PrecisionGuarantee, pub backend: fuel_ir::probe::BackendId }`
  - `pub fn probe_from_operands(operands: &[OperandDesc], seed: u64) -> Option<Vec<crate::fkc::verify::HostTensor>>` — deterministic float-fill probe inputs sized from each operand's `rank`/`shape`; `None` for a non-encodable dtype (mirrors `seed_cuda_ledger::build_cuda_probe`'s `to_bytes`).

- [ ] **Step 1: Write the failing test** (`jit_ingest_probe.rs`): two rank-1 F32 operands of extent 4 → two `HostTensor`s of shape `[4]`, F32, 16 bytes each, deterministic.

```rust
#[test]
fn probe_from_operands_builds_sized_float_inputs() {
    let od = OperandDesc::new(1, [4,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0], ElementKind::F32, 16);
    let p = probe_from_operands(&[od, od], 0x1234).expect("probe");
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].shape, vec![4]);
    assert_eq!(p[0].dtype, DType::F32);
    assert_eq!(p[0].bytes.len(), 16);
    assert_eq!(probe_from_operands(&[od, od], 0x1234).unwrap()[0].bytes, p[0].bytes); // deterministic
}
```

(Confirm `OperandDesc::new`'s exact arg order from `baracuda-kernels-types/src/structure_key.rs:~315` while writing this.)

- [ ] **Step 2: Run it, watch it fail.** Run: `cargo test -p fuel-dispatch --features jit --lib probe_from_operands -v`. Expected: FAIL.

- [ ] **Step 3: Implement `probe_from_operands`.** For each operand: extent product from `shape[..rank]`; `element_kind_to_dtype(operand.dtype)` (the helper already exists in `seed_cuda_ledger`/`jit_adopt` — reuse it); encode `fill_deterministic(n, seed ^ i)` via the dtype (reuse `seed_cuda_ledger`'s `to_bytes`); return `HostTensor { dtype, shape: shape[..rank], bytes }`. `None` for an unencodable dtype.

- [ ] **Step 4: Run it, watch it pass.** Run: `cargo test -p fuel-dispatch --features jit --lib probe_from_operands -v`. Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs fuel-dispatch/src/jit_ingest_probe.rs
git commit -m "feat(jit): CandidateKernel + probe synthesis from OperandDesc"
```

---

## Task 4: Decompose-reference realize (`reference_output`)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest_probe.rs`

**Interfaces:**
- Produces: `#[cfg(feature="cuda")] pub fn reference_output(decompose: &PatternNode, probe: &[HostTensor], out_dtype: DType, out_shape: Vec<usize>, device: &CudaDevice) -> Result<HostTensor>` — realizes the decompose region on probe consts and returns the output bytes.

- [ ] **Step 1: Read `fuel-graph/src/runtime_fused.rs:230-260`** (`decompose_region` + `emit`) to learn how a `PatternNode` region is re-emitted into a `Graph` given input `NodeId`s. Note `emit(graph, &region, &inputs, &mut cursor) -> NodeId`.

- [ ] **Step 2: Write the failing GPU test** (`#[ignore]`): a 2-input `Add` region `PatternNode` + two F32 `[4]` probes → `reference_output` returns the elementwise sum bytes.

```rust
#[test]
#[ignore = "requires a live CUDA device"]
fn reference_output_realizes_the_decompose() {
    let Ok(dev) = CudaDevice::new(0) else { return };
    let region = PatternNode::Op { op: OpTag::Add, attrs: OpAttrs::default(),
        operands: vec![PatternNode::Bind{index:0}, PatternNode::Bind{index:1}] };
    let a = ht_f32(&[1.0,2.0,3.0,4.0]); let b = ht_f32(&[10.0,20.0,30.0,40.0]);
    let out = reference_output(&region, &[a,b], DType::F32, vec![4], &dev).unwrap();
    assert_eq!(bytes_to_f32(&out.bytes), vec![11.0,22.0,33.0,44.0]);
}
```

- [ ] **Step 3: Run it (GPU), watch it fail.** Run: `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit --lib reference_output_realizes_the_decompose -- --ignored --nocapture`. Expected: FAIL (fn undefined).

- [ ] **Step 4: Implement `reference_output`.** Build a fresh `Graph`; push one `Op::Const`-backed input node per probe (upload probe bytes as a CUDA `Storage` bound to the node, mirroring `invoker_cuda.rs`'s H2D); `emit(&mut graph, decompose, &input_ids, &mut cursor)` to get the region's sink node; realize the sink on `device` via the standard realize path; D2H the output to a `HostTensor`. (Cross-check the const-binding shape against how `pipelined.rs`'s decode capture binds per-token consts.)

- [ ] **Step 5: Run it (GPU), watch it pass.**

- [ ] **Step 6: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest_probe.rs
git commit -m "feat(jit): realize a decompose region on probes (verification reference)"
```

---

## Task 5: `verify_candidate` core (candidate vs reference + bit-stability + ledger)

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs`

**Interfaces:**
- Produces: `#[cfg(feature="cuda")] pub fn verify_candidate(cand: &CandidateKernel, device: &CudaDevice) -> VerifyVerdict` where `pub enum VerifyVerdict { Pass, Fail { claim: &'static str, detail: String }, }` plus the earned `LedgerRecord`s.

- [ ] **Step 1: Write the failing GPU test** (`#[ignore]`): a candidate whose kernel is the CUDA `add_f32` wrapper with a 2-input `Add` decompose → `Pass` (candidate == reference, bit-stable). (Model the candidate construction on `invoker_cuda.rs`'s `cuda_invoker_runs_add_elementwise_f32_end_to_end`.)

- [ ] **Step 2: Run it (GPU), watch it fail.**

- [ ] **Step 3: Implement `verify_candidate`** by adapting `seed_cuda_ledger::run_cuda_verification`'s per-entry body:
  1. `probe = probe_from_operands(&cand.operands, seed)`; `catch_unwind`-wrap everything.
  2. Candidate output: build a `BindingEntry { kernel: cand.kernel, precision: cand.declared, kernel_revision_hash: cand.kernel_revision_hash, .. }` and invoke via `CudaInvoker::new(device.clone(), out_dtype, out_shape).with_params(cand.op_params.clone()).invoke(&entry, &probe)` (the exact pattern in `seed_cuda_ledger`).
  3. Bit-stability: `verify_bit_stability(&inv, &entry, std::slice::from_ref(&probe), 16)` — must be `Pass`.
  4. Reference: if `cand.decompose.is_some()`, `reference_output(...)`; else fall back to the CPU-reference path (`register_cpu_kernels` + `CpuInvoker`) exactly as `seed_cuda_ledger`'s `max_ulp` loop does.
  5. Precision: for each declared claim in `cand.declared` (`bit_stable_on_same_hardware`, `max_ulp`, `max_relative`, `max_absolute`), compare candidate vs reference with the matching `Bound` (`max_ulp_ok` from `seed_cuda_ledger`, or `verify_precision_bound`). First failure → `Fail { claim, detail }`.
  6. Upsert a `pass`/`fail` `LedgerRecord` per checked claim keyed on `(Cuda, cand.dtypes, cand.kernel_revision_hash, claim)`.

- [ ] **Step 4: Run it (GPU), watch it pass.**

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(jit): verify_candidate — candidate vs decompose/reference + bit-stability + ledger"
```

---

## Task 6: `ingest_one` — sync verify → adopt / reject-with-feedback

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs`

**Interfaces:**
- Produces: `#[cfg(feature="cuda")] pub fn ingest_one(cand: &CandidateKernel, device: &CudaDevice) -> IngestOutcome`. On `Pass` → `adopt_runtime_fused(cand.entry_point.clone(), cand.decompose.clone().expect("fused candidate has a decompose"), cand.kernel, cand.dtypes.clone(), cand.backend)` → `Adopted(id)`. On `Fail{claim,detail}` → `Rejected(RejectionReport{ ... })`.

- [ ] **Step 1: Write the failing GPU test** (the validating one): a candidate whose kernel is baracuda's **interleaved** `rope_apply_f32` but whose `decompose` is the **rotate-half** rope `PatternNode` (build it via `fuel-graph`'s rope `decompose`) on a rope-shaped probe → `ingest_one` returns `Rejected` with `failed_claim` a precision claim.

```rust
#[test]
#[ignore = "requires a live CUDA device"]
fn ingest_rejects_interleaved_rope_for_the_rotate_half_region() {
    let Ok(dev) = CudaDevice::new(0) else { return };
    let cand = interleaved_rope_candidate_for_rotate_half_region(); // helper: kernel=rope_apply_f32, decompose=rotate-half pattern
    match ingest_one(&cand, &dev) {
        IngestOutcome::Rejected(r) => assert!(r.failed_claim.contains("max") || r.failed_claim == "vs_decompose"),
        IngestOutcome::Adopted(_) => panic!("interleaved kernel must NOT be adopted for a rotate-half region"),
    }
}
```

- [ ] **Step 2: Run it (GPU), watch it fail** (fn undefined). Then implement `ingest_one` (verify_candidate → match verdict → adopt or build report + `catch_unwind` around adopt).

- [ ] **Step 3: Run it (GPU), watch it pass** — the interleaved kernel is rejected. Add a second `#[ignore]` test: a *correct* candidate (CUDA `add_f32` for the `Add` region) → `Adopted(id)` and `fused_kernel_available_in(global table, id, Cuda)` after.

- [ ] **Step 4: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(jit): ingest_one — verify then adopt-or-reject (rejects interleaved rope for rotate-half)"
```

---

## Task 7: `IngestionService` — bounded queue + idle-aware concurrency-1 worker

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs`

**Interfaces:**
- Produces:
  - `pub struct IngestionConfig { pub queue_bound: usize /*default 32*/, pub max_concurrent: usize /*default 1*/, pub idle_load_threshold: u32 /*default 1*/ }`
  - `pub struct IngestionService { .. }` with `pub fn start(device: CudaDevice, cfg: IngestionConfig) -> Self`, `pub fn enqueue(&self, cand: CandidateKernel, feedback: Option<Arc<dyn ProviderFeedback>>) -> Result<(), Backpressure>`, `pub fn shutdown(self)`.
  - `pub struct Backpressure;`

- [ ] **Step 1: Write a no-GPU test** driving the queue + backpressure with an **injectable verify closure** (so it runs without a device): a bounded queue of 1, enqueue two items with a blocked worker → the second returns `Backpressure` + fires `on_rejected("queue full")`. Refactor `IngestionService::start` to take the verify step as `Fn(&CandidateKernel) -> IngestOutcome` so tests inject a mock and production passes `|c| ingest_one(c, &device)`.

```rust
#[test]
fn enqueue_backpressures_and_notifies_when_full() { /* mock verify blocks; assert Backpressure + on_rejected("queue full") */ }
```

- [ ] **Step 2: Run it, watch it fail.** Run: `cargo test -p fuel-dispatch --features jit --lib enqueue_backpressures -v`.

- [ ] **Step 3: Implement `IngestionService`.** A bounded `std::sync::mpsc::sync_channel(queue_bound)`; `enqueue` uses `try_send` → `Err(Full)` maps to `Backpressure` + `feedback.on_rejected(queue_full_report())`. One worker thread: loop `recv()`; before each verify, **idle-gate** — while `inflight_count(DeviceLocation::cuda(0)) >= cfg.idle_load_threshold`, `thread::sleep(small)` (best-effort); run the injected verify; on `Adopted` → `feedback.on_adopted`, on `Rejected` → `feedback.on_rejected`. `max_concurrent` via a semaphore (default 1 → strictly serial). `shutdown` drops the sender + joins.

- [ ] **Step 4: Run it, watch it pass.** Add a second no-GPU test: mock verify returning `Adopted` → `on_adopted` fired; a mock verify that panics → worker survives + logs (no crash).

- [ ] **Step 5: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs
git commit -m "feat(jit): IngestionService — bounded queue + idle-aware concurrency-1 verify worker"
```

---

## Task 8: End-to-end GPU wiring test + module exports

**Files:**
- Modify: `fuel-dispatch/src/jit_ingest.rs` (re-exports), `fuel-dispatch/src/lib.rs` (pub use)

**Interfaces:**
- Consumes: everything above.
- Produces: `pub use jit_ingest::{CandidateKernel, IngestionService, IngestionConfig, ProviderFeedback, RejectionReport, IngestOutcome};` under `#[cfg(feature="jit")]`.

- [ ] **Step 1: Write the failing GPU end-to-end test** (`#[ignore]`): start an `IngestionService`, enqueue the *correct* `add_f32` candidate with a recording `ProviderFeedback`, wait for the callback → assert `on_adopted` fired and the op is `fused_kernel_available`. Then enqueue the interleaved-rope candidate → assert `on_rejected` fired with a precision claim.

- [ ] **Step 2: Run it (GPU), watch it fail; implement the re-exports; run, watch it pass.** Run: `cmd //c 'C:\Windows\Temp\cuda_run.bat' cargo test -p fuel-dispatch --features cuda,jit --lib jit_ingest -- --ignored --nocapture`.

- [ ] **Step 3: Verify no default-build regression.** Run: `cargo test -p fuel-dispatch --lib` (default, no `jit`/`cuda`). Expected: the existing suite still passes (the new module is feature-gated out).

- [ ] **Step 4: Commit.**

```bash
git add fuel-dispatch/src/jit_ingest.rs fuel-dispatch/src/lib.rs
git commit -m "feat(jit): end-to-end ingestion service exports + GPU wiring test"
```

---

## Self-review notes (coverage against the spec)

- Spec §4.1 CandidateKernel → Task 3. §4.2 ProviderFeedback + Synthesizer::on_rejected → Task 2. §4.3 RejectionReport → Task 2. §4.4 verify core (probe/reference/candidate/checks/ledger) → Tasks 3–5. §4.5 queue + idle-aware worker + backpressure → Task 7. §4.6 lock-nesting prerequisite → Task 1. §5 never-panic → `catch_unwind` in Tasks 5–7. §6 feature gating → cfgs throughout + Task 8 Step 3. §7 tests incl. the interleaved-rope rejection → Task 6/8. §8 boundaries (no discovery, no background scheduling) → not implemented here by design.
- The one carried assumption to confirm during Task 4: re-emitting a `PatternNode` on fresh probe consts via `emit(...)` binds inputs the way the realize path expects (validated by Task 4's GPU test before it's depended on in Task 5).

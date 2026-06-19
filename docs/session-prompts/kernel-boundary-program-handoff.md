# Kernel-Boundary Program — MASTER fresh-instance handoff

**Status:** Living index + sequencer (2026-06-18). Branch **`feat/kernel-contracts-dlpack`**
(unmerged; `main` untouched). This is the single entry point for a new Claude instance picking up
the kernel-boundary program with no prior context beyond `CLAUDE.md` + memory.

This document is an **INDEX and SEQUENCER**, not a restatement. It tells you *what exists*, *what is
locked*, *what to build next*, and *which detailed doc owns each piece*. Read the detailed docs it
points at before executing — they carry the step-by-step, the file:line anchors, and the born-red
tests. Do not duplicate their content here; when in doubt, the detailed doc wins on detail and the
specs win on meaning.

---

## 0. What this program is

Fuel passes tensors to kernels, and kernels advertise themselves to Fuel's planner. This program
builds the **three-spec boundary** that makes both halves explicit, honest, and optimizer-visible:

1. **FDX** — [`docs/specs/dlpack-extension.md`](../specs/dlpack-extension.md). The *tensor* axis:
   a standard versioned DLPack base (`DLTensor`) plus an optional nullable `*const FDXSidecar` for
   the facts standard DLPack can't carry (sub-byte/quant/symbolic/gather/bundle). **Honesty
   invariant:** the base `DLTensor` is never a lie — a sidecar-blind reader gets opaque bytes, never
   wrong numbers.
2. **FKC** — [`docs/specs/kernel-contract-format.md`](../specs/kernel-contract-format.md). The
   *advertisement* axis: a markdown contract that auto-registers a kernel (dispatch key, accept/return
   contracts, caps, cost, precision, determinism) onto Fuel's dispatch surface — zero hand-written glue.
3. **Storage encoding (`SType`/`Encoding`)** — [`docs/specs/storage-encoding.md`](../specs/storage-encoding.md).
   The *self-describing-storage* axis (NEW, 2026-06-18): the encoding **scheme** travels on the tensor
   (`Storage.stype`), so an op knows its bytes are e.g. NF4 block-affine without an op-param. FDX is
   `SType`'s **projection** at the kernel boundary.

The whole program lives on **`feat/kernel-contracts-dlpack`**. Cross-project providers (Baracuda
kernels, Vulkane FFI) are coordinated by **outreach proposals**, never unilateral sibling edits.

### Source-of-truth hierarchy (higher wins on conflict)

1. **`docs/architecture/`** — the constitution. Authoritative over everything.
2. **The three specs** (`storage-encoding.md`, `dlpack-extension.md`, `kernel-contract-format.md`) —
   own their respective type/boundary meaning. `storage-encoding.md` owns the internal `SType`/`Encoding`
   type; FDX owns the boundary code shape; when they disagree on the boundary mapping, reconcile and flag.
2b. **`ROADMAP.md`** — phase sequencing / current frontier.
3. **The plan docs** (this doc's §4 queue) — per-workstream execution steps. **Verify against git
   before trusting** — many describe shipped work.
4. **This handoff** — the index that threads them together.

> Rule when code and a sketch/plan disagree: **the code is ground truth.** Read the file, cite
> file:line, note the divergence. Every plan in this program already did one such reconciliation pass;
> redo it for any fact you depend on (the tree moves).

---

## 1. STATE — DONE vs STILL TO BUILD

### DONE (committed + pushed on the branch)

- **FDX spec** + **FKC spec** — DRAFT, complete (`docs/specs/dlpack-extension.md`,
  `docs/specs/kernel-contract-format.md`).
- **390 internal kernels inventoried**; ~69 `.fkc.md` contract-bundle files across 9 crate dirs;
  corpus lint-clean + CI-gated (0 hard failures).
- **Never-panic prerequisite (partial):** `KernelBindingTable::register*` is append-only +
  `finalize() -> Result` dup gate (`fuel-dispatch/src/kernel.rs`, `dispatch.rs`).
- **`fuel-core-types::dlpack` module** (behind `dlpack` feature): abi/codes/sidecar/validate/convert
  + C header cross-check (`fuel_dlpack_ext.h`), **106 tests**.
- **`fuel-dispatch` FKC importer** (behind `fkc` feature): parse/lower/register/validate + CI corpus
  lint, **104 tests**; `registrable: false` describe-only affordance (FKC §3.10).
- **DlpackView comm-layer slice 1** (`fuel-memory/src/dlpack_view.rs`): borrowed
  `(Storage, Layout[, SymEnv]) -> DLTensor + sidecar`; sidecars built for sub-byte dtype-ext,
  symbolic extents, bundle. **DEFERRED inside it:** quant + gather sidecars (quant was blocked on
  op-context — **NOW UNBLOCKED by the SType decision**), GPU ptr extraction.
- **`fuel-metal-kernels` `DType -> MetalDType` rename** — grep-clean; cannot compile-verify on win32
  (objc2 is Apple-only — pre-existing platform limit, not a rename break).
- **The three outreach docs are DRAFTED (not sent):**
  [`baracuda-reply.md`](../outreach/baracuda-reply.md),
  [`vulkane-dlpack-fkc-ask.md`](../outreach/vulkane-dlpack-fkc-ask.md), and the prior
  [`baracuda-dlpack-fkc-ask.md`](../outreach/baracuda-dlpack-fkc-ask.md).
- **The four planning docs are AUTHORED** (the §4 queue items): self-describing-storage,
  baracuda-telemetry, plus the pre-existing comm-layer / contract-adoption / kernel-conversion plans.

### STILL TO BUILD (each owned by a plan in §4)

- **Self-describing Storage** — steps 1–3 SHIPPED 2026-06-19 (`SType`/`Encoding`/`ScaleSpec` in
  `fuel-core-types/src/stype.rs`; `Storage.stype` on both structs, default-empty; `SType::to_fdx()`
  projection + `view()` fills the quant sidecar from `storage.stype`). Commits `4bbe566c` /
  `f241c87c` / `fb385c4b`. **Remaining:** step 4 `view_with_quant` (bind the AFFINE scale-buffer
  index — the graph-op already declares the absmax sibling in `nf4_matmul.rs`; the open wrinkle is
  the V6 block-geometry projection off the logical shape) + steps 5–7 (loader notes, round-up). See
  `self-describing-storage-plan.md` Progress header.
- **Remaining comm-layer** — quant + gather sidecars (quant now unblocked), external `__dlpack__`
  boundary, capability negotiation via `KernelCaps`/`Capability` tokens, `KernelRef` ABI migration.
- **FKC adoption** — provider `LinkRegistry`s + wire the importer into dispatch init + per-kernel
  conversion of the 390 kernels.
- **Cost trampoline** — declared cost priors -> `CostFn` (the FKC `cost_expr` compiler). **Deferred**
  behind the importer-into-init wiring.
- **Baracuda telemetry emission** — `DispatchRecord`/`MissRecord` JSONL writer over the (existing)
  Judge oracle + miss signal + opt-in flag + `ImplId`/`StructureKey` join. **Retention is DONE; only
  emission remains** (see §3).
- **Send the outreach** — Baracuda reply, Baracuda ask, Vulkane ask (after a final read; they are drafts).

---

## 2. LOCKED DECISIONS — do NOT re-litigate

These were settled by verified investigation. A fresh instance must build *on* them, not reopen them.

1. **Self-describing storage = `DType` + `SType`/`Encoding`.** `DType` stays the **logical** element
   type (an NF4 weight is logically `F16`/`F32`, not "4-bit"). `SType` (a **named newtype** over
   `SmallVec<[Encoding; 1]>`, default empty = plain) is the **physical** encoding stack. `Encoding` is
   data-free, `Eq + Hash`, holds only static descriptors (geometry/scheme/dtype codes + scale
   *requirements*). Full spec: [`storage-encoding.md`](../specs/storage-encoding.md).

2. **Scale DATA model = (B) sibling operand + FDX sidecar composite at the boundary** — decided
   **AGAINST** model A (composite-by-reference: embedding the scale buffer inside the weight's
   Storage/Encoding). The weight's `Encoding` declares only the *requirement* (`AffineBlock` +
   `ScaleSpec`); the consuming op binds the actual per-block absmax as a **separate graph operand**;
   FDX re-unites `{scheme, scale-buffer index}` into one `AFFINE_BLOCK` descriptor
   (`scale_placement = SEPARATE_BUFFER`). Verified rationale (storage-encoding.md §4): multi-output
   machinery is **one-buffer-only** (`fuel-graph/src/lib.rs:962-976`,
   `fuel-dispatch/src/pipelined.rs:273-295`, `docs/architecture/12-multi-output.md`); FDX *already*
   specifies `AFFINE_BLOCK` scales as `SEPARATE_BUFFER` (`dlpack-extension.md` §6.2); B is cheaper,
   honest, and matches GPTQ/HF/bnb convention.

3. **Negative strides are first-class** (reversed 2026-06-17). FDX describes signed `int64` strides;
   OOB is a signed *touched-range* check (V13), not a ban. The planner normalizes a flip to a copy
   **only** for an incapable consumer — never universally, never between capable internal kernels.
   This keeps the `flipped`/`reverse_strides` demand axis **visible** (load-bearing for the telemetry
   miss histogram, §4).

4. **GGML stays INLINE** (forced, not a choice). GGUF on-disk is interleaved struct-packed
   (Q4_0 = `{f16 d; u8 qs[16]}` = 18 bytes/block; `fuel-core-types/src/quantized.rs:87-113`); the
   format + k-quants + ~40 quantized kernels + zero-copy mmap all assume it. `Encoding::GgmlBlock` =
   inline, no sibling scale. **Do NOT generalize interleaving to NF4** (would force a repack on load
   from bnb's separate-tensor format, killing zero-copy, for no win). **Efficiency rule:** layout
   follows source — GGML interleaved; NF4/GPTQ/bnb separate. No universal winner.

5. **Judge retains per-impl timings** (see §3 — corrects stale memory). **`candidates[]` is feasible.**

6. **Cost lives in FKC, not FDX.** FDX is pure description, carries no cost and no decision (FDX
   P3/G7). The advertisement/cost axis is FKC; the decision is the planner's, from advertised
   capability. Do not add cost to FDX.

7. **Honesty invariant.** The base `DLTensor` is always honestly interpretable as standard DLPack on
   its own. Ignoring the sidecar can lose *meaning* (opaque `uint8` bytes), never produce *wrong
   numbers*. Enforced mechanically by the validators; the producer policy
   refuses-or-materializes (`IS_COPIED`) for a meaning-requires-ext tensor to a sidecar-blind consumer.

8. **Ground-truth divergences from the original sketches (already reconciled — don't re-discover):**
   there are **two** `Storage` structs — `fuel-memory/src/lib.rs:89-101` (`{ inner, dtype, bundle }`)
   and `fuel-core-types/src/storage.rs:216-224` (`{ inner, bundle }`, **no `dtype` field** —
   delegates to `inner.dtype_dyn()`); both gain `stype`. There is **no `PerBlock`** in
   `ScaleGranularity` (`quant_scale.rs:38`) — `AffineBlock` block grain rides `block_shape`. There is
   **no `NF4` dtype** — NF4 reuses `DType::F4` (`dtype.rs:44` → FDX code 13). `Encoding::Mx` +
   `AffineInt`/`AffineFloat`/`Compressed` are reserved, not v1.

---

## 3. The JUDGE resolution (corrects stale memory)

> **The Judge RETAINS per-impl timings — any memory saying "winner-only" or "latencies are f32
> squares" is STALE; follow the code (verified 2026-06-18). Distinguish two things: the *retention
> mechanism* is shipped and stable (not mid-rebuild), but the Judge's *coverage* is narrow and being
> actively, extensively expanded (more dtypes, every op lacking a declared cost, flash-vs-decomposed
> arm comparison). Treat coverage as in-flight; treat retention as done.**

The Judge **already retains per-`(op, dtype, size_class, backend, kernel_source)` timings, including
losing alternatives, as `u64` nanoseconds.** Two artifacts, both verified:

- **`ProfileReport` / `ProfileEntry`** (persistent JSON) — `fuel-core-types/src/dispatch.rs:655-692`.
  One entry **per measured alternative including losers**; `latency_ns: u64` (NOT f32 squares);
  `kernel_source: String` distinguishes siblings. `PROFILE_REPORT_VERSION == 2`.
- **`HashMapJudge` / `JudgeOracle`** (in-memory) — `fuel-dispatch/src/ranker/judge.rs:53-75`, keyed
  `(OpKind, DType, SizeClass, BackendId, String) -> u64`. Adapter `ProfileJudgeOracle::from_report`
  at `fuel-core/src/judge/oracle.rs:65-92` indexes **every** entry (losers included); the
  loser-retention rationale is in the module docs (`oracle.rs:10-23`). Sibling non-collision is a
  shipped passing test, `sibling_kernel_sources_do_not_collide` (`oracle.rs:170-195`).

**Consequence:** Baracuda Open-Q-1 ("per-(shape,impl) timings or winner-only?") is **answered YES**;
`candidates[]` is feasible by reading the oracle — no retention rebuild.

**Coverage caveat (transient, not a retention gap):** the Judge's *populated coverage* is currently a
bounded profiling matrix — **F32 only** (measurement loop hardcodes `DType::F32` at
`fuel-core/src/judge/mod.rs:476, 512`, guarded at `:730`), an offline-profiled **square-matmul size
ladder** (no GEMV / decode-shaped cells), a fixed primitive set, no online exploration. Non-F32 /
GEMV / quantized cells **miss** (`None`) — the correct "no measurement" signal, never a fabricated
number. This is being **actively, extensively expanded** (per CireSnave, the Judge "will not be
F32-only for long"). **Build any consumer (e.g. the Baracuda telemetry feed) coverage-agnostic** —
read whatever the oracle holds so it densifies automatically as the matrix grows, with no format
change. Document the limit; don't fabricate a number.

This is fully written up in [`docs/outreach/baracuda-reply.md`](../outreach/baracuda-reply.md) §2.

---

## 4. The DEPENDENCY-ORDERED work queue

Two tracks run in parallel: the **build spine** (A→F, mostly serial) and a **parallel coordination
track** (P, no code dependency). Within any step, obey the discipline in §5 (one cargo invocation,
`-p <crate>`, born-red).

### Build spine (serial)

**A. Self-describing Storage (`SType`/`Encoding`)** — the keystone; unblocks the deferred quant sidecar.
- Plan: [`self-describing-storage-plan.md`](self-describing-storage-plan.md) (7 steps, TDD-first).
  Spec: [`storage-encoding.md`](../specs/storage-encoding.md).
- **Unblocked by:** nothing — the `dlpack` module + `DlpackView` slice 1 already exist (§1 DONE).
- **Unblocks:** the quant sidecar in `view()` (B), and the consuming-op scale-sibling wiring.
- Shape: step 1 new `fuel-core-types/src/stype.rs` (types); step 2 `stype: SType` field on **both**
  `Storage` structs (default empty = byte-identical); step 3 `SType::to_fdx()` fills the deferred
  quant sidecar in `dlpack_view.rs`; step 4 consuming-op binds the scale operand (model B at the
  graph layer); steps 5-6 GGML-inline + loader notes; step 7 test round-up.

**B. Quant + gather sidecars (the comm-layer deferrals)** — finish what `DlpackView` slice 1 left.
- Plan: [`dlpack-comm-layer-plan.md`](dlpack-comm-layer-plan.md) §2.3 (sidecar build), §4 (validators).
- **Quant sidecar** is **NOW UNBLOCKED by A** (the scheme travels on `storage.stype`; `view()` reads
  it instead of needing op-context). It is literally step 3 of plan A.
- **Gather sidecar** is **NOT unblocked by A** (it needs paged-pool/block-table op-context, not an
  encoding scheme). It follows the same model-B pattern but is sequenced with attention kernels (see
  the gather draft `docs/specs/_drafts/fdx-addition-gather.md` and the conversion plan Phase 5).
- **Unblocks:** the external boundary (C) and the per-kernel conversion of quant/attention kernels (E).

**C. External `__dlpack__` boundary + capability negotiation** — comm-layer §3, §5.
- Plan: [`dlpack-comm-layer-plan.md`](dlpack-comm-layer-plan.md) §3 (boundary b: managed/deleter,
  deleter-identity gating), §5 (extend `Capability` tokens + grow `KernelCaps.reverse_strides`).
- **Unblocked by:** B (sidecars must exist to export). **Unblocks:** the planner
  normalize-only-for-incapable-consumer decision and the `KernelRef` ABI migration.

**D. Provider `LinkRegistry`s + wire the FKC importer into dispatch init** — adoption plan §11.6-11.9.
- Plan: [`kernel-contract-adoption-plan.md`](kernel-contract-adoption-plan.md) (the importer exists,
  §1 DONE; this is the *adoption* — wiring real providers and flipping the default).
- **Unblocked by:** the importer (DONE) + the never-panic register path (DONE). **Unblocks:** the
  per-kernel conversion (E) — each provider needs a `LinkRegistry` mapping `entry_point` → `KernelRef`.

**E. Per-kernel conversion of the 390 kernels** — the bulk.
- Plan: [`internal-kernel-dlpack-conversion-plan.md`](internal-kernel-dlpack-conversion-plan.md)
  (5 categories A-E; phase order CPU-contiguous → other backends → strided → quant → attention →
  bundle → reference). Each kernel: author contract, set `entry_point`, import green, delete the
  hand-written `register*`, born-red equivalence + numeric-parity test.
- **Unblocked by:** B (quant/gather sidecars), C (caps), D (LinkRegistries). **Unblocks:** F (cost
  trampoline has real contracts to compile).

**F. Cost trampoline (declared priors → `CostFn`)** — DEFERRED.
- Plan: [`kernel-contract-adoption-plan.md`](kernel-contract-adoption-plan.md) §2.3 (the two-target
  cost-expr compiler, strategy A: interpreter-backed `fn` + global side-table keyed by
  `(op|id, dtypes, backend, kernel_source)`). Capacity-only eval for v1.
- **Unblocked by:** E (contracts authored). Lowest priority — sequence last.

### Parallel coordination track (no code dependency on the spine)

**P. Baracuda telemetry emission + send the three outreach docs.**
- **Telemetry emission plan:** [`baracuda-telemetry-plan.md`](baracuda-telemetry-plan.md) (8 steps in
  a new `fuel-dispatch/src/telemetry/` behind a `telemetry` feature; the JSONL sink in
  `fuel-core/src/telemetry.rs`). Builds `DispatchRecord`/`MissRecord`, `ImplId` (+ thread `revision`
  onto `BindingEntry`), the `StructureKey` provider seam (Fuel **calls** Baracuda's `structure_key`,
  never reimplements), the miss signal (best-admissible-match-is-generic), `candidates[]` from the
  oracle, the opt-in flag, the batch JSONL sink. **Retention is DONE (§3); this is emission only.**
- **Send the Baracuda reply** — [`baracuda-reply.md`](../outreach/baracuda-reply.md) (resolves the
  Judge-retention deferral, answers Open-Q-1 = YES). After sending, update
  [`baracuda-dlpack-fkc-ask.md`](../outreach/baracuda-dlpack-fkc-ask.md) §6 per its maintainer note.
- **Send the Vulkane ask** — [`vulkane-dlpack-fkc-ask.md`](../outreach/vulkane-dlpack-fkc-ask.md)
  (FDX-only: carry the nullable sidecar, preserve signed strides/offset/256-byte alignment, plural
  buffer table; FKC is N/A unless Vulkane exposes compute). (Its earlier "storage-encoding.md does
  not yet exist" divergence note was corrected 2026-06-18 — the spec now exists.)

These three sends and the telemetry build are **independent of the spine** (telemetry reads the
existing oracle; the outreach is cross-project coordination). Do them whenever; they are not blocked.

---

## 5. Build / test DISCIPLINE (verbatim guardrails — non-negotiable)

- **NEVER run workspace-wide `cargo check`/`cargo test`.** `tensor-tools` has a standing `Device::Cpu`
  break and is a default-member, so even bare `cargo check` at the root fails. **Always `-p <crate>`.**
- **ONE cargo invocation at a time** (the build-dir lock serializes; parallel invocations thrash).
  Long builds: background + wait.
- **ONE live-GPU test suite at a time** (two concurrent live suites OOM the dev GPU — RTX 4070, 12 GB).
  Run `#[ignore]`'d live-GPU tests locally after kernel/executor work.
- **TDD, born-red.** Write the failing test FIRST, run it, **observe it go red**, then make it green.
  Record the red→green transition in the commit body. Shipping tests that never ran is banned.
- **Docs in the SAME change as behavior.** When a change alters a core claim/commitment/interface,
  update the relevant `docs/architecture/` section (bump version + `10-decisions-log.md` on a MAJOR)
  and the `ROADMAP.md` frontier in the same change. Treat doc-vs-code drift as a defect.
- **Never panic on production paths.** `Result` from day one; no `try_*` siblings — just the
  `Result`-returning version.
- **WIP on `feat/kernel-contracts-dlpack`, not `main`.** With CI red, "main builds" is a convention —
  keep it true.
- **Sibling path deps must exist** for the workspace to parse: `../aocl`, `../vulkane/vulkane`
  (beside `fuel/`, outside the repo); baracuda from crates.io. **To check whether baracuda has a
  kernel, grep `baracuda-kernels-sys` (the FFI surface), NOT the plan facade.** Never enable the
  `aocl` cargo feature in tests.
- **Ask before modifying sibling projects** (baracuda, aocl, vulkane, lightbulb, mlmf). Missing Vulkan
  kernels are fuel-internal Slang (`fuel-vulkan-kernels`), never a baracuda ask.
- Commit messages end with the line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## 6. First 3 moves for a fresh instance

1. **Orient against ground truth (15 min, no edits).** Confirm you are on
   `feat/kernel-contracts-dlpack`. Read the three specs' §0 headers
   ([`storage-encoding.md`](../specs/storage-encoding.md),
   [`dlpack-extension.md`](../specs/dlpack-extension.md),
   [`kernel-contract-format.md`](../specs/kernel-contract-format.md)). Spot-check that §1's "DONE"
   claims still hold: the `dlpack` + `fkc` features compile and their test counts pass
   (`cargo test -p fuel-core-types --features dlpack`, `cargo test -p fuel-dispatch --features fkc`
   — **one at a time**). Confirm §3's Judge facts at `fuel-core-types/src/dispatch.rs:655-692` and
   `fuel-dispatch/src/ranker/judge.rs:53-75` (already verified for this handoff; re-verify if you
   touch them).

2. **Start the keystone: workstream A, step 1.** Open
   [`self-describing-storage-plan.md`](self-describing-storage-plan.md). Write the 5 born-red tests
   for `SType`/`Encoding`/`ScaleSpec` in a new `fuel-core-types/src/stype.rs`, run
   `cargo test -p fuel-core-types stype`, **watch them fail**, then implement the types per the plan's
   step 1. This unblocks the quant sidecar (B) and everything downstream.

3. **Kick the parallel track while the spine compiles.** The telemetry emission and the outreach
   sends have no dependency on A. Either: (a) start
   [`baracuda-telemetry-plan.md`](baracuda-telemetry-plan.md) step 1 (the `DispatchRecord`/`MissRecord`
   JSONL schema in a new `telemetry` feature), or (b) do a final read of the three outreach drafts and
   send them — applying the maintainer note in
   [`baracuda-reply.md`](../outreach/baracuda-reply.md) to
   [`baracuda-dlpack-fkc-ask.md`](../outreach/baracuda-dlpack-fkc-ask.md) §6 (move the
   Judge-retention deferral to RESOLVED).

After these three, return to the §4 spine and proceed A → B → C → D → E → F, keeping P moving in
parallel. The program is done when all 390 kernels are contract-described, the importer drives
dispatch init, the comm-layer boundaries are wired, and the telemetry feed emits over the (already
retained) Judge timings.

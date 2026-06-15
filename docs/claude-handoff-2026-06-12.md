# Claude handoff — 2026-06-12 (machine wipe evacuation)

Written by the outgoing Claude instance for the next one. The dev
machine was wiped and reinstalled; the per-machine Claude memory
directory (`~/.claude/projects/.../memory/`) did NOT survive, so
this document carries everything you need to resume. Read it fully
before doing anything.

## What Fuel is, and who you're working with

Fuel is the user's Rust ML framework (~40 workspace crates): lazy
graph IR + autograd, a unified `PipelinedExecutor`, a cost-based
picker (per-node kernel alternatives ranked by static cost + Judge
empirical data + runtime selectors), multi-backend (CPU portable +
AOCL/MKL as `kernel_source` siblings, CUDA via the baracuda sibling
project, Vulkan via fuel-internal Slang kernels, Metal stub).

The user (CireSnave) is the author of Fuel AND the sibling projects.
Collaboration norms they have explicitly established — treat these
as standing instructions:

- **Engage critically.** They welcome architectural pushback, not
  deferral. Investigate "why does X work that way" fully.
- **Never panic in production paths** — Result-returning from day
  one. (One standing violation is queued: see Backlog.)
- **Validate at graph-build time** — every check that CAN run at
  build MUST. No `try_*` siblings; just the Result version.
- **Lazy-only / no-deferrals-until-eager-retired**: the eager
  retirement backlog is ONE program; ship missing primitives rather
  than punting blocked ports; new features land lazy-only.
- **"No consumer" is not a reason** to skip building a capability.
- **Ask before modifying sibling projects** (baracuda, vulkane,
  lightbulb, mlmf, aocl) — propose cross-project edits first.
  Baracuda is CUDA-only; missing Vulkan kernels are fuel-internal
  Slang (`fuel-vulkan-kernels`), never a baracuda ask.
- **Match external convention for well-known ops** (PyTorch/CUDA
  semantics) over internal consistency; design param surfaces
  (Reduction modes etc.) up front.
- **Test-gated ships are mandatory** — multiple "born-red" tests
  came from batch commits verified with cargo-check only.

## Hard-won environment + discipline facts

- Windows 11, RTX 4070 (CUDA 13 + Vulkan both work live). Run
  `#[ignore]`'d live-GPU tests locally after kernel/executor work.
- **NEVER `cargo check/test` the whole workspace** — tensor-tools
  has a pre-existing `Device::Cpu` break. Always `-p <crate>`.
- **ONE cargo invocation at a time** — the build-dir lock serializes
  and parallel invocations thrash. Long builds: background + wait.
- **Live-GPU suites are exclusive** — two concurrent live suites
  OOM the 4070. One live run at a time, period.
- **`../aocl` must exist** — workspace `Cargo.toml` has PATH deps
  (`aocl-blas`, `aocl-types` at `../aocl/crates/...`). Clone the
  aocl project beside fuel or the workspace won't parse. Never
  enable the `aocl` cargo feature in tests (AMD's installer doesn't
  put lib/LP64 on PATH; DLL load fails).
- Baracuda deps come from crates.io (pinned `0.0.1-alpha.67`); the
  local `../baracuda` checkout is for reference only. **When
  checking whether baracuda has a kernel, grep
  `baracuda-kernels-sys` (the FFI surface), NOT the plan facade** —
  two real capabilities (reduce_to since alpha.46, unary_step) were
  missed by facade-level audits.
- Vulkan f64 transcendentals are ~f32-accurate on NVIDIA Windows
  (1e-7 abs tolerance); OpenCL.std SPIR-V is unusable (passes
  spirv-val, rejected by vkCreateShaderModule); Windows timer
  resolution stretches sub-ms sleeps to ~15ms (this shaped the
  TopologyChanged retry design — see c85dd06b/e5ac60c5 history).
- The Judge's pairwise-consensus epsilon (1e-3) can MASK kernel
  flavor divergences (~1e-4) — gelu tanh-vs-erf slipped through it;
  only value-level live tests caught it. Kernel-vs-OpKind contract
  review at registration time is the defense.
- Multi-agent note: long-running subagents that arm background
  cargo tasks and idle can die without returning structured output.
  Their COMMITS survive; recover from `git log` + workflow journals
  and re-run only the unfinished tail. Guard aggregation code
  against null lane results.

## State at evacuation (main @ bc5ab384, all pushed)

Suite baselines (all green at last full verification):
fuel-core --lib 1320/0, fuel-dispatch --lib 339/0 (cuda 340),
fuel-graph --lib 232/0, cuda_dispatch_live 55/55,
vulkan_dispatch_live 196/196, residency_eviction_live 1/1,
cpu_vulkan_diff vulkan_trains 2/2, judge:: cuda 30/0,
train:: 24/24. NOTE: the last two evacuation commits (303ae8ca WIP
+ bc5ab384) came AFTER that verification — see "In flight" below.

### Completed programs (high level — git log is the detail)

1. **Picker arc** (Phases 1-5 + remediation): per-node
   AlternativeSets, JudgeOracle Layer-2, ChainedSelector
   (VramPressure→JudgeAware→Winner) default-ON, kernel_source
   end-to-end, pairwise-consensus correctness + distributable
   fixtures (capture tool + triple-agreement loader gates).
2. **Baracuda alpha.67**: gemm_dense MatMul (4 dtypes incl. f64),
   reduce_to (4 dtypes), `register_cuda_kernels` = Op::Copy only;
   `byte_kernels.rs` deleted. Ask/reply docs:
   `docs/baracuda-ask-fp-gemm-reduce-to-2026-06-10.md`.
3. **Executor unification Sessions 1-6** (re-audit doc:
   `docs/session-prompts/executor-unification-reaudit-2026-06-11.md`):
   typed realize entries, WorkItemKind::Move, Judge re-point
   (BridgeRealizer), the legacy Router branch deleted (S3 —
   PipelinedExecutor is THE executor on every realize entry),
   model-family ports (S4, bit-identical generation), train.rs
   (S5, bit-exact SGD parity), residency eviction on pipelined +
   **fuel-graph-router crate DELETED** + BandwidthMatrix onto
   TransferCalibration (S6). fuel-core/src legacy census: doc
   comments only. fuel-graph-executor survives ONLY as a fuel-core
   dev-dep for 2 FA2 oracle tests.
4. **Load-time planner Stages 1-4a** (program doc:
   `docs/session-prompts/load-time-incremental-planner.md`;
   architecture: 04-optimization v0.4 §Load-time incremental
   planning + 06-runtime v1.1): transfer calibration probe,
   residency-priced placement (always-enumerate), carry-forward
   placement DP with fused jumps + exit pricing, PlanStore +
   Planner::warm + coverage-wait realize (measured 1.8×/token on
   synthetic decode loop). The DECISION doc behind all of this:
   planning starts at model load; realize() = wait-for-coverage +
   dispatch.
5. **Robustness fixes the verifiers forced**: generation-aware
   TopologyChanged settle budget (c85dd06b); GPU Copy wrappers
   route D2H-or-same-device on output substrate (d9c958e8);
   insert_safety_copies is dependency-based, not topo-position
   (21b103bd — the old code was also UNDER-protective in 2 cases);
   Vulkan D2H HOST_CACHED staging ~140 MB/s → ~3 GB/s (79a0fe6a);
   Q4_0 bake gates derived per-model (qwen2 + 7-sibling sweep —
   qwen3_moe was UNDER-gated).

### In flight at evacuation — READ CAREFULLY

A two-lane workflow (Session 7 ∥ Stage 4b) was running when the
machine died. **Neither lane completed.**

- **Session 7 produced ZERO commits.** Its work (FA2 oracle-test
  port → FA2 eager wrapper retirement → GraphBackend impls+trait
  deletion → fuel-graph-executor + fuel-graph-cpu::realize_any
  retirement → cpu_vulkan_diff/conv2d_oracle migration) must be
  re-run from scratch. The lane brief is reproduced in the Backlog
  below.
- **Stage 4b left commit 303ae8ca**: `wip(planner)` — +417 lines in
  fuel-dispatch/src/plan_store.rs (background-warm latch +
  revision-submission surface). **UNVERIFIED, may not compile.**
  First action on resume: `cargo check -p fuel-dispatch`. Treat it
  as raw material; revert freely if a clean re-run of 4b is easier.
- bc5ab384 carries ROADMAP.md additions + fuel-graph's float8
  workspace dep (no source references it yet; harmless).

## Resume order (the user-approved program)

0. **Re-verify the baseline** after environment setup:
   `cargo test -p fuel-core --lib`, `-p fuel-dispatch --lib`,
   `-p fuel-graph --lib`, then the live suites one at a time.
   Expect the baselines above EXCEPT possibly fuel-dispatch if the
   WIP commit doesn't compile — fix or revert 303ae8ca first.
1. **Stage 4b** (redo/finish): background warm driver +
   ahead-of-frontier plan adoption. Spec: planner prompt Stage 4 +
   06-runtime §Background re-optimization. Key invariants: a
   realize arriving mid-warm blocks on a per-graph latch (no double
   planning); revisions adopt at dispatch-chunk boundaries ONLY
   when the executed prefix matches node-for-node (reject
   otherwise — safety property, test it); no-revision fast path is
   an atomic read.
2. **Session 7** (redo): the deletion session. Gate: port the two
   FA2 oracle tests (fuel-core/tests/flash_attn_cuda.rs +
   flash_attn_oracle.rs) onto the bridge with comparisons
   unweakened; then FA2 eager wrapper retirement; then GraphBackend
   impls + trait + fuel-graph-executor crate + workspace membership
   + fuel-graph-cpu::realize_any; migrate cpu_vulkan_diff +
   conv2d_oracle legacy-realize tests (the vulkan_trains pair is
   ALREADY ported — don't touch). fuel-reference-backend STAYS
   (architecture v0.4: test oracle). Audit before deleting: if the
   pipelined side imports anything from fuel-graph-executor
   (derive_ordering? execution_plan?), MOVE it to its consumer
   first.
3. **Session 8** (the eager tail): full surgical plan is committed
   at `docs/session-prompts/eager-tail-session-8-surgical-plan.md`
   — ~5,790 LOC across 14 ordered commits, keep-boundary, risks,
   CI gates. Follow it.
4. **Tensor::from_* .expect() → Result sweep** — the standing
   never-panic violation at fuel-graph/src/lib.rs:2259 (it turned a
   Vulkan-training regression into a panic once already). Pairs
   with the deferred realize-signature Result sweep (354+ call
   sites; coordinated breaking change).
5. **Planner Stages 5-6**: cross-graph plan-fragment memoization by
   structural hash (layer stamping; also the persisted-cache key +
   CUDA-Graph capture unit), then plan-driven weight prefetch
   (the plan IS the prefetch schedule; mmap page-in + H2D streams
   ahead of the execution frontier). Specs in the planner prompt.
6. **Stage 4 DP debts** (documented in 0a786821): fan-out prices
   first consumer only; diamond cross-edges unpriced;
   repeat-realize stamp residency nuance.

### Smaller queued items

- Vulkan-flavored cost dispatcher (kernel_overhead_ns ~5000 vs
  CPU's ~50; Judge corrects empirically — low urgency).
- Per-kernel Vulkan KernelCaps strided_input audit.
- compile_node legacy fallback retirement in fuel-dispatch.
- Real capture run on the 4070 + commit the first distributable
  fixture set (fuel-capture-fixtures binary exists; the capture
  matrix lacks reduction ops — when adding them, fix the
  reduction-formula divergence first: capture's unary input is
  2.1e-3 sin, Judge's reduction arm is 1.7e-3 sin).
- `_retired` tree drop (post-S8 final audit).
- In-place optimizer updates via InplaceKernel (parity-kept
  fresh-buffer for now; swap when large-scale training matters).
- Eager FA2 wrapper retirement rides Session 7 (above).

## Key documents (all in-repo)

- `docs/architecture/` — 13-section doc set. 04-optimization v0.4 +
  06-runtime v1.1 carry the load-time-planning decision;
  05-backend-contract v0.4 carries Reference retirement +
  kernel_source extensions + BackendRuntime compliance.
- `docs/session-prompts/load-time-incremental-planner.md` — the
  planner program (7 stages; 1-4a done).
- `docs/session-prompts/executor-unification-reaudit-2026-06-11.md`
  — the unification program (Sessions 1-6 done; 7-8 remain).
- `docs/session-prompts/eager-tail-session-8-surgical-plan.md` —
  Session 8 blueprint.
- `docs/session-prompts/eager-tensor-retirement-master-plan.md` +
  `shipped/eager-retirement-phase-h-plan.md` — the umbrella.
- `docs/baracuda-ask-*.md` — sibling-project coordination history.

## Sibling projects (separate repos, coordinate with the user)

- `../baracuda` — CUDA kernel home (crates.io: 0.0.1-alpha.67).
- `../aocl` — REQUIRED path dep (see environment facts).
- vulkane, lightbulb, mlmf — coordinated separately; ask first.

## Rebuild your memory

Recreate memory entries from this document's "norms" and
"environment facts" sections first (they encode the user's standing
feedback), then a program-progress entry pointing at this file and
the session-prompt docs. The user values: outcome-first reporting,
honest deferral notes, adversarial verification of shipped work
(it caught 8+ critical bugs across this program — keep the
ship→verify→fix cadence), and bit-exact parity gates on ports.

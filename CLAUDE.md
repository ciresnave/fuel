# Fuel — agent working agreement

Fuel is a **DAG-first, lazy-only ML framework for Rust** (forked from HuggingFace Candle, now diverged). Every model is a lazy DAG; the optimizer that reads the DAG is where the intelligence lives. Backends advertise capabilities/costs/telemetry but never make strategic decisions.

**Source-of-truth hierarchy** (when they conflict, higher wins):
1. `docs/architecture/` — the **constitution** (13 sections + decisions log). Authoritative over everything below.
2. `ROADMAP.md` — the **path** (phases, sequencing, current frontier).
3. `docs/session-prompts/` — per-program plans (note: many describe *shipped* work; verify against git before trusting as a queue).
4. `docs/claude-handoff-2026-06-12.md` — post-wipe resume anchor + program state.

Per-machine memory lives at `~/.claude/projects/c--Projects-fuel/memory/` (wiped 2026-06; rebuilt). Read `MEMORY.md` there.

---

## Build & environment discipline (hard rules — learned the expensive way)

- **NEVER run `cargo check`/`cargo test` workspace-wide.** `tensor-tools` has a standing `Device::Cpu` break and is a default-member, so even bare `cargo check` at the root fails. **Always `-p <crate>`.**
- **ONE cargo invocation at a time.** The build-dir lock serializes; parallel invocations thrash. Long builds: background + wait.
- **One live-GPU test suite at a time.** Two concurrent live suites OOM the dev GPU (RTX 4070, 12 GB). Run `#[ignore]`'d live-GPU tests locally after kernel/executor work.
- **Multiple agent sessions share ONE working tree + `.git/index` here — never mutate git state in the shared checkout concurrently.** Several Claude sessions run in `C:\Projects\fuel` at once (the claude-peers channel lists them); they share one working directory *and* one git index, so concurrent `git add`/`commit`/`checkout`/`reset`/`rebase`/`stash` clobber each other silently — a `git add` in one session gets swept into another session's `commit` (observed 2026-07-20). Rules: **(1)** treat the shared checkout's `main` as **read-only** — never develop, `git add`, or commit on it while other sessions may be active; **(2)** do all commit-producing work — **code AND docs** — in an isolated **`git worktree`**: `git worktree add ../fuel-<task> -b <branch> origin/main`, edit/commit *there*, then `git push origin HEAD:main` (or open a PR). Sibling path-deps `../aocl`/`../vulkane` still resolve from a `C:\Projects`-sibling worktree, so the workspace parses. **(3)** Branch from **pushed** `origin/main`, never a stale local base; **re-fetch right before pushing** — a peer may have advanced `main` under you between fetch and push. **(4)** If two sessions genuinely must share the tree, coordinate over the peers channel so only one performs git ops at a time. This supersedes the looser "WIP goes on a branch" note below for the multi-session case.
- **Sibling path deps must exist for the workspace to parse** (they live *beside* `fuel/`, outside the repo):
  - `../aocl` (github.com/ciresnave/aocl) — `aocl-blas`, `aocl-types` path deps. Never enable the `aocl` cargo feature in tests (AMD's DLLs aren't on PATH).
  - `../vulkane/vulkane` (github.com/ciresnave/vulkane) — Vulkan FFI, used by `fuel-vulkan-backend` (a default-member).
  - `baracuda` (CUDA kernels) comes from **crates.io** pinned `0.0.1-alpha.72`; a local `../baracuda` checkout is reference-only. **To check whether baracuda has a kernel, grep `baracuda-kernels-sys` (the FFI surface), NOT the plan facade.** (2026-07-03: `--features cuda` builds from a plain shell fail with `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` — NOT a CUDA-13.3/CUTLASS issue. Workaround until the next baracuda alpha (which self-resolves via vswhere): set `NVCC_CCBIN=<path-to-cl.exe>` or build from a VS Developer shell.) **Runtime PATH (2026-07-04):** the `fuel-core` cuda test exe directly imports `cudnn64_9.dll` (+ `cublas64_13.dll`); cuDNN's CUDA-13.3-matched build must be on PATH to *launch* it or the process dies `0xc0000135 STATUS_DLL_NOT_FOUND`. Prepend `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64` (installed there, not on the default PATH). `fuel-dispatch` cuda tests don't hit this (no cuDNN link).
- Environment (as of 2026-06-13): Windows 11, RTX 4070, CUDA 13 + Vulkan SDK installed, Rust 1.96 / edition 2024. `fuel-dispatch` checks clean (warnings only).

## Engineering process (these are expectations, not suggestions)

- **Test-driven development is the default.** Write the failing test first, watch it go red, then make it green. "Born-red" tests are the *goal*, not an accident to apologize for. The historical failure mode — batch commits verified with `cargo check` only, shipping tests that never ran — is banned. A change that touches behavior ships with the test that exercises it, and that test must have been observed to run.
- **Docs are part of every material change.** When a change alters a core claim, a commitment, or an interface, update the relevant `docs/architecture/` section (bump its version + add a `10-decisions-log.md` entry on a MAJOR bump) and the `ROADMAP.md` frontier in the *same* change. Periodically re-check that docs still match code; treat doc-vs-code drift as a defect.
- **Validate at graph-build time.** Every check that *can* run at build time *must*. No `try_*` siblings — just the `Result`-returning version.
- **Never panic on production paths.** `Result` from day one. (Standing violation to fix: `Tensor::from_*` `.expect()` at `fuel-graph/src/lib.rs:~2256`.) The three ex-"panicking" `decompose`s are **resolved** (2026-07-03; a prior G2 pass had already converted the panics to self-returns — the real work was recipes, not crashes): **`nf4_matmul`** now carries a total primitive recipe (nibble-unpack + indicator-sum NF4 codebook + per-block scale → matmul); **`flash_attn`** decomposes its concrete-`k_len` decode (static `Slice` + bottom-right-aligned SDPA) — only symbolic (`Sym`) `k_len` stays a documented registry-layer gap (no `DynScalar`-length `Slice`; the symbolic oracle is the `decode_flash` optimizer arm, which holds the `SymEnv`); **`selective_scan`** is the constitution's canonical basis gap (G3 — needs a higher-order `Scan` `Op`; the CumSum closed-form overflows for `a<0`), kept as a never-crash surfaced gap. Parity + gap-posture tests in `fuel-core/src/lazy.rs`.
- **The recipe principle / total `decompose` is a build-time invariant (G1/G2/G3).** Every fused op ships with BOTH a `decompose` (fused → primitive subgraph) and a `pattern` (re-fuse); `decompose` is total + never-panic + primitive→self (base map = its fixpoint); the primitive `Op` basis is build-time-closed. A non-basis op that won't decompose is a surfaced opaque-op gap (telemetry), never a crash — and it breaks the optimizer itself (optimization = lower-to-base-map + find-best-cover). See [docs/architecture/10-decisions-log.md](docs/architecture/10-decisions-log.md) (2026-06-20 "Adaptive runtime fusion").
- **WIP/unverified work goes on a branch, not `main`.** With CI currently red, "main builds" is only a convention — keep it true.
- **Ship → verify → fix.** Adversarial verification of shipped work has repeatedly caught bugs tests missed (gelu erf-vs-tanh under the Judge epsilon, under-protective safety copies, Vulkan D2H staging). Keep the cadence.

## Collaboration norms (CireSnave)

- **Engage critically.** Architectural pushback is welcome; investigate "why does X work this way" fully before accepting it. Deliver assessments, not deferral.
- **Lazy-only / no deferrals until eager is fully retired.** The eager-retirement backlog is one program; ship the missing primitive rather than punting a blocked port. New features land lazy-only.
- **"No consumer" is not a reason to skip building a capability** — but it *is* a reason to sequence it behind things with consumers (see ROADMAP priority).
- **Ask before modifying sibling projects** (baracuda, aocl, vulkane, lightbulb, mlmf). Propose cross-project edits first; missing Vulkan kernels are fuel-internal Slang (`fuel-vulkan-kernels`), never a baracuda ask.
- **Match external convention for well-known ops** (PyTorch/CUDA semantics) over internal consistency; design param surfaces up front.

## Reporting

Outcome first. Report test results faithfully (with the actual output); say plainly when something is skipped, unverified, or failing. Don't claim "done" without having run the gate.

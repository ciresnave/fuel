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
- **Sibling path deps must exist for the workspace to parse** (they live *beside* `fuel/`, outside the repo):
  - `../aocl` (github.com/ciresnave/aocl) — `aocl-blas`, `aocl-types` path deps. Never enable the `aocl` cargo feature in tests (AMD's DLLs aren't on PATH).
  - `../vulkane/vulkane` (github.com/ciresnave/vulkane) — Vulkan FFI, used by `fuel-vulkan-backend` (a default-member).
  - `baracuda` (CUDA kernels) comes from **crates.io** pinned `0.0.1-alpha.67`; a local `../baracuda` checkout is reference-only. **To check whether baracuda has a kernel, grep `baracuda-kernels-sys` (the FFI surface), NOT the plan facade.**
- Environment (as of 2026-06-13): Windows 11, RTX 4070, CUDA 13 + Vulkan SDK installed, Rust 1.96 / edition 2024. `fuel-dispatch` checks clean (warnings only).

## Engineering process (these are expectations, not suggestions)

- **Test-driven development is the default.** Write the failing test first, watch it go red, then make it green. "Born-red" tests are the *goal*, not an accident to apologize for. The historical failure mode — batch commits verified with `cargo check` only, shipping tests that never ran — is banned. A change that touches behavior ships with the test that exercises it, and that test must have been observed to run.
- **Docs are part of every material change.** When a change alters a core claim, a commitment, or an interface, update the relevant `docs/architecture/` section (bump its version + add a `10-decisions-log.md` entry on a MAJOR bump) and the `ROADMAP.md` frontier in the *same* change. Periodically re-check that docs still match code; treat doc-vs-code drift as a defect.
- **Validate at graph-build time.** Every check that *can* run at build time *must*. No `try_*` siblings — just the `Result`-returning version.
- **Never panic on production paths.** `Result` from day one. (Standing violation to fix: `Tensor::from_*` `.expect()` at `fuel-graph/src/lib.rs:~2256`.)
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

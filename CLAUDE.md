# Fuel — agent working agreement

Fuel is a **DAG-first, lazy-only ML framework for Rust** (forked from HuggingFace Candle, now diverged). Every model is a lazy DAG; the optimizer that reads the DAG is where the intelligence lives. Backends advertise capabilities/costs/telemetry but never make strategic decisions.

**Source-of-truth hierarchy** (when they conflict, higher wins):
1. `docs/architecture/` — the **constitution** (13 sections + decisions log). Authoritative over everything below.
2. `ROADMAP.md` — the **path** (phases, sequencing, current frontier).
3. `docs/session-prompts/` — per-program plans (note: many describe *shipped* work; verify against git before trusting as a queue).
4. `docs/claude-handoff-2026-06-12.md` — post-wipe resume anchor + program state.

`docs/gaps.md` is the **source of truth for open defects + ownership** — read it before claiming or closing a gap, and update the row in the same change that closes one. (Registry only; the reasoning for a material fix still belongs in `docs/architecture/`.)

Per-machine memory lives at `~/.claude/projects/c--Projects-fuel/memory/` (wiped 2026-06; rebuilt). Read `MEMORY.md` there.

**How to read this file.** Each bullet is the CURRENT rule plus the mechanism that keeps it from being rationalised away. Its evidence lives in the `docs/method-rules.md` section it links to. ⚠️ **Before acting on a rule, read its section: the bullet is a pointer, not a summary to reason from.** This file was trimmed on 2026-09-16 (141,563 → under 35,000 bytes, base c4936cfc), and every removed line survives verbatim in a linked section.

---

## Establishing facts on a shared box

- **Read facts at `origin/main`, never from a working tree** — several sessions share `C:\Projects\fuel`, your Bash cwd defaults to it, and a stale tree returns a CLEAN answer, which is the direction nobody questions. `git fetch` per question, or compare `git rev-parse origin/main` with `git ls-remote origin main`. A null result needs its positive control, its ref, and a reconciliation with any finding it contradicts. → [`establish-facts-at-origin-main`](docs/method-rules.md#establish-facts-at-origin-main)
- **`git show origin/main:./.github/…` — prefix a leading-dot path with `./`.** MSYS rewrites `ref:path` when BOTH sides look path-like and the fatal goes to stderr, so a wrapped `grep -c` prints a clean `0`. ⚠️ Never "fix" it with `main:`/`HEAD:` — those return a plausible, complete, STALE file. Safe slash-free forms: `FETCH_HEAD` after a fetch, or a bare sha. → [`git-show-mangles-a-leading-dot-path`](docs/method-rules.md#git-show-mangles-a-leading-dot-path)
- **The shared checkout's `main` is READ-ONLY; do all commit-producing work in a `git worktree`, branched from a CAPTURED sha** (`BASE=$(git rev-parse origin/main)`) — linked worktrees share refs, so any peer's fetch moves `origin/main` under you. `gh pr merge --delete-branch` is a local git mutation: never from the shared tree. ⚠️ **`core.hooksPath` is the absolute shared hook dir: NEVER edit `.git/hooks/*`** — edit `scripts/install-hooks.sh` / `scripts/check-gaps-table.py`. → [`the-shared-tree-and-the-shared-hook`](docs/method-rules.md#the-shared-tree-and-the-shared-hook)
- **A local branch rots like the shared tree** — `git checkout main` in a worktree lands you in the past. → [`a-local-branch-goes-stale-too`](docs/method-rules.md#a-local-branch-goes-stale-too)
- **A sha is not a stable name in a rebase workflow** — anchor cross-lane checks on content or subject. → [`sha-is-not-a-stable-name`](docs/method-rules.md#sha-is-not-a-stable-name)
- **In a REBASE `--theirs` is YOUR commit.** → [`git-rebase-inverts-ours-and-theirs`](docs/method-rules.md#git-rebase-inverts-ours-and-theirs)
- **A stale TOOL is a wrong ACTION** — executing an old `gpu-run.ps1` (from the shared tree or your own pre-fix worktree) carries machine-wide state and leaves no record of which version ran. → [`a-stale-tool-is-a-wrong-action`](docs/method-rules.md#a-stale-tool-is-a-wrong-action)
- **`from_cwd`, a peer summary, a branch name say where a process STARTED, never where it ACTED** — ask whether the changed set intersects what was measured, and send the probe without the accusation. → [`where-a-process-starts-is-not-where-it-acts`](docs/method-rules.md#where-a-process-starts-is-not-where-it-acts)
- **Reach for the instrument that answers the question, not the nearest one** — a session-start CLAUDE.md snapshot is indistinguishable from head; use `git show <ref>:<path>` for your own project's files too. → [`the-instrument-nearest-to-hand`](docs/method-rules.md#the-instrument-nearest-to-hand)
- **Sibling checkouts (`../aocl`, `../vulkane`, `../baracuda`) are REFERENCE-ONLY** — `vulkane`, `aocl-*` and `baracuda-*` resolve from crates.io; read `Cargo.lock`, never the manifest or the sibling tree, for what Fuel builds. When a peer corrects one entry in a list you wrote, re-measure the whole list. **Ask before modifying sibling projects.** → [`sibling-checkouts-are-reference-only`](docs/method-rules.md#sibling-checkouts-are-reference-only)
- **This box:** Windows 11, **RTX 4070 Laptop GPU, 8 GB** (a 12 GB budget OOMs silently) plus an AMD iGPU via Vulkan; `new_device()` uses `PreferDiscrete` and returns the NVIDIA card, so a tri-vendor test must pick the Vulkan adapter explicitly. → [`this-box`](docs/method-rules.md#this-box)

## Build discipline

- **Prefer `-p <crate>`.** A root `cargo check` works but costs ~31 min on a shared box; `cargo test` workspace-wide still pulls live-GPU and network tests. → [`prefer-p-crate-and-the-root-check`](docs/method-rules.md#prefer-p-crate-and-the-root-check)
- **ONE cargo invocation at a time** (the build-dir lock serialises; parallel invocations thrash), **and cap it at `-j 4`** — at the default ~22 jobs rustc races and blames random crates. Discriminator without an ICE dump: `exit != 0` with ZERO `level:error` in `--message-format json`. Never delete `rustc-ice-*.txt` before it is explained. → [`cap-build-parallelism-at-j4`](docs/method-rules.md#cap-build-parallelism-at-j4)
- **Run bare `cargo`/`cargo fmt`/`cargo clippy`, never `+toolchain`** — `rust-toolchain.toml` is the single source and an explicit `+toolchain` overrides it. Confirm the file is on disk first (older checkouts default to nightly). → [`run-bare-cargo-the-toolchain-is-pinned`](docs/method-rules.md#run-bare-cargo-the-toolchain-is-pinned)
- **A remembered control replaced by a structural one COMPETES with it.** → [`a-defence-can-outlive-its-defect`](docs/method-rules.md#a-defence-can-outlive-its-defect)
- **The gate must match the change kind:** an **enum variant** on an `EXHAUSTIVE-BY-DESIGN` enum needs a workspace-scope check plus each feature-gated consumer — `-p <defining crate>` is the one gate guaranteed not to see it. → [`an-enum-variant-needs-a-workspace-gate`](docs/method-rules.md#an-enum-variant-needs-a-workspace-gate)
- A **struct-field add** needs `--all-targets` on every CONSTRUCTING crate, unless all fields are private. → [`a-struct-field-add-breaks-constructing-crates`](docs/method-rules.md#a-struct-field-add-breaks-constructing-crates)
- **Adding a binding** is gated by `cargo clippy -p <crate> --tests -- -D warnings`, not by the test. → [`adding-a-binding-is-gated-by-the-lint`](docs/method-rules.md#adding-a-binding-is-gated-by-the-lint)
- **Backend crates have no `cuda`/`vulkan` feature — they ARE the backend.** Gate them with a plain `cargo check -p fuel-<x>-backend --all-targets`; the wrong flag is `exit 101` with zero artifacts. → [`backend-crates-have-no-backend-feature`](docs/method-rules.md#backend-crates-have-no-backend-feature)
- **Gate several crates in ONE invocation with `pkg/feature` syntax** (`cargo check -p fuel-core -p fuel-cuda-backend --features fuel-core/cuda --all-targets`) — split invocations resolve features differently and share no cache. → [`gate-several-crates-in-one-invocation`](docs/method-rules.md#gate-several-crates-in-one-invocation)
- **`cargo check --workspace` needs a NAMED exclusion set, and every exclusion is named where the green is read** — CI's `WORKSPACE_EXCLUDES`: `fuel-metal-backend`/`fuel-metal-kernels` (wrong platform; macOS keeps them) · `fuel-mkl-cpu-backend`/`fuel-aocl-cpu-backend` (missing SDK) · `fuel-cuda-backend` (forge COST — so no CI job compiles Fuel's CUDA code). Read the HALT line, never the error count. A local `--workspace` green is not evidence about CI. Prefer installing a tool over excluding a crate. → [`the-workspace-check-and-its-exclusions`](docs/method-rules.md#the-workspace-check-and-its-exclusions)
- **`--all-targets` ≠ `--all-features`, and `cargo test --workspace` unifies features through dev-dependencies** — a `cfg` gate is evidence of neither coverage nor skipping; use `cargo tree -e features --workspace`. A test that cannot run must be `ignored`, never `ok`. Boundary statements take `// GAP(GAP-NNN)`, not the hedge allowlist. `mkl`/`aocl` BUILD here (CI cannot see them); don't enable `aocl` in test runs (AMD DLLs aren't on PATH). → [`all-targets-is-not-all-features-and-the-converse`](docs/method-rules.md#all-targets-is-not-all-features-and-the-converse)
- **`--lib` builds neither `tests/` nor an in-file `#[cfg(test)] mod tests`, and `--all-targets` skips doctests** — state the target kinds with any green. → [`lib-does-not-build-tests`](docs/method-rules.md#lib-does-not-build-tests)
- **`cargo check` builds metadata; `cargo test` compiles and links** — a fast check predicts nothing about a test build (31 min of codegen once hid behind a 50 s check). Commit WIP before ending a turn. → [`check-is-not-a-test-build`](docs/method-rules.md#check-is-not-a-test-build)
- **A cargo fingerprint is keyed on features+flags** — the file existing does not make YOUR invocation warm. → [`the-fingerprint-filename-is-not-the-fingerprint`](docs/method-rules.md#the-fingerprint-filename-is-not-the-fingerprint) · Force a rebuild by touching a SOURCE file; a guessed `.fingerprint` glob can match nothing. → [`clearing-a-fingerprint-by-guessed-path`](docs/method-rules.md#clearing-a-fingerprint-by-guessed-path)
- **`cargo fmt` does not touch doc-comment content, and the pre-commit hook runs no clippy** — run `cargo clippy -p <crate> --tests` after editing doc comments. → [`cargo-fmt-does-not-touch-doc-comments`](docs/method-rules.md#cargo-fmt-does-not-touch-doc-comments)
- **`fuel-onnx` needs `protoc`** — check `command -v protoc` first; if a stale parent environment hides it, prepend its WinGet dir or set `PROTOC`. It is not a default-member, so gate it explicitly. → [`fuel-onnx-needs-protoc`](docs/method-rules.md#fuel-onnx-needs-protoc)

## GPU, CUDA and long builds

- **Every GPU-touching run goes through `pwsh scripts/gpu-run.ps1 -Project <name> -- <cmd>`** — live-GPU tests, `--features cuda|vulkan` runs, capture/replay, sanitizers, benches. It is the machine-wide mutex that prevents a repeat of the 2026-07-31 bugcheck; keep live-GPU tests `#[ignore]`. It serialises processes, not in-process fan-out. → [`all-gpu-runs-go-through-gpu-run`](docs/method-rules.md#all-gpu-runs-go-through-gpu-run)
- **CUDA builds:** read `Cargo.lock` for baracuda's version (the manifest's `alpha.78` is a caret minimum); build through `vcvarsall amd64` (`C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat`; harvest its env in a throwaway `cmd /c "vcvarsall amd64 && set"` — never on a Git-Bash command line, which mangles `/c` and the quotes — and never hand-set `NVCC_CCBIN`); **the forge runs on any `--features cuda` OR `--workspace` build** (it ignores the `cuda` feature), so **take a build slot through `scripts/cuda-build.ps1`, which sets `BARACUDA_FORGE_THREADS` to its protocol value and overrides any value you set** (outside a slot, set it explicitly); prepend `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64` to PATH to launch cuda test exes (else `0xc0000135`). Grep baracuda-kernels-sys, not the plan facade, for kernels. → [`the-baracuda-dependency-and-the-cuda-build-recipe`](docs/method-rules.md#the-baracuda-dependency-and-the-cuda-build-recipe)
- Why nothing runs after vcvarsall in a `.bat` → [`a-297-byte-log-has-three-causes`](docs/method-rules.md#a-297-byte-log-has-three-causes)
- **Never accept an exit code as evidence a CUDA build ran.** `exit≠0 + artifact` = real failure · `exit≠0 + none` = harness never ran · `exit=0 + artifact` = success · `exit=0 + none` = warm cache OR nothing attempted (only `--message-format json` separates them). A warm cache hides warnings too.
- **ONE `--features cuda` build on this box at a time** — `ptxas fatal : Memory allocation failure` appears with 22 GB free, and `gpu-run` does not serialise builds. Attribute peer processes by target path or parentage, never by count. Grep `ptxas fatal`/`nvcc fatal` before blaming anything. → [`one-cuda-build-at-a-time-and-the-misleading-red`](docs/method-rules.md#one-cuda-build-at-a-time-and-the-misleading-red)
- **Background tasks die at ~10 min; the forge takes 56–93 min — launch DETACHED** (`Win32_Process Create` or `Start-Process`), then **poll on your own turn**. Three checks: pid alive
- A completion signal carrying CARGO's exit code
- The target crate's compile line. → [`background-tasks-are-killed-at-ten-minutes`](docs/method-rules.md#background-tasks-are-killed-at-ten-minutes)
- **Derive completion from an artifact the BUILD writes** (cargo's json output, the binary's mtime) — a marker appended by a killable wrapper dies with it. A `killed`/`completed` notification names the TASK and reports the WRAPPER's exit, never cargo's; write a marker on ANY backgrounded build you intend to believe, to a separate file from the log. → [`a-long-cuda-build-runs-detached-with-a-marker`](docs/method-rules.md#a-long-cuda-build-runs-detached-with-a-marker) · [`the-marker-can-be-eaten-by-its-own-log`](docs/method-rules.md#the-marker-can-be-eaten-by-its-own-log)
- **A log is evidence only once you know which process wrote it and when** — a killed wrapper's cargo child keeps the handle, the delete fails silently, and old output sits above new. Check mtime and the pid in the log. → [`a-log-is-evidence-only-with-its-writer`](docs/method-rules.md#a-log-is-evidence-only-with-its-writer)

## Gates, artifacts and CI

- **For a scoped check the artifact is the TARGET crate's compile line** → [`target-crate-compile-line`](docs/method-rules.md#target-crate-compile-line)
- **A crate compile line proves nothing about a `cfg`'d MODULE** — require something only that module can emit, and read the `cfg` at head. → [`a-crate-compile-line-is-not-a-module-compile-line`](docs/method-rules.md#a-crate-compile-line-is-not-a-module-compile-line)
- **A compile line licenses only the phases that ran** — no E0004 means nothing after an E0432; sabotage-check exhaustiveness, clearing the fingerprint on both transitions. → [`a-compile-line-licenses-only-the-phases-that-ran`](docs/method-rules.md#a-compile-line-licenses-only-the-phases-that-ran)
- **A structural guarantee is only as strong as the compile coverage of the code carrying it** — name the configuration that compiles it and confirm a gate builds it. → [`a-structural-guarantee-needs-compile-coverage`](docs/method-rules.md#a-structural-guarantee-needs-compile-coverage)
- **One feature is not two** — gate the intersections a site actually needs. → [`one-feature-is-not-two`](docs/method-rules.md#one-feature-is-not-two)
- **Validating a gate means reading its MESSAGE**; a check that only prints is not a gate. → [`validating-a-gate-means-reading-it`](docs/method-rules.md#validating-a-gate-means-reading-it)
- **`&&` only helps if the exit code carries the verdict** (`clippy` exits 0 on warnings without `-D`). → [`a-report-is-not-a-gate`](docs/method-rules.md#a-report-is-not-a-gate)
- **A workflow that never starts looks like one that fails** — the discriminator is `steps: 0`. Set `fail-fast: false` AND `--no-fail-fast`; say whether a failure list is a list or a prefix. **No CI job may need a GPU; CUDA is tested locally only.** → [`a-workflow-that-never-starts`](docs/method-rules.md#a-workflow-that-never-starts) · [`wip-on-a-branch-and-no-gpu-ci`](docs/method-rules.md#wip-on-a-branch-and-no-gpu-ci)
- **Ask of inherited config whether anything ever COMPILED it.** → [`inherited-config-provenance`](docs/method-rules.md#inherited-config-provenance)
- **Every conformance/byte-match report states its exact command and feature flags.** → [`a-conformance-report-states-its-command`](docs/method-rules.md#a-conformance-report-states-its-command)
- **A clean rebase says nothing about a gate whose subject is DATA** — re-run it before pushing when the rebase touched its inputs. → [`a-clean-rebase-says-nothing-about-a-data-gate`](docs/method-rules.md#a-clean-rebase-says-nothing-about-a-data-gate)
- **Measure a gate before building it** (flag count, false positives). → [`measure-the-gate-before-building-it`](docs/method-rules.md#measure-the-gate-before-building-it)
- **A gate that fires on your own tooling: move the tool, don't teach the gate an exception.** → [`a-correct-total-can-hide-a-wrong-distribution`](docs/method-rules.md#a-correct-total-can-hide-a-wrong-distribution)
- **An exclusion, suppression or known-failing entry carries a detector that reddens when its cause dissolves.** → [`an-allowlist-entry-that-reddens-when-its-reason-dissolves`](docs/method-rules.md#an-allowlist-entry-that-reddens-when-its-reason-dissolves)
- **An expiry fires on a checkpoint that will occur, and lives in an owned registry row, not in prose at the site.** → [`an-expiry-must-fire-on-a-checkpoint`](docs/method-rules.md#an-expiry-must-fire-on-a-checkpoint)
- **A new file `git status` does not list is a finding** (a bare `data/` ignore swallows fixtures). → [`a-new-file-that-git-does-not-list-is-a-finding`](docs/method-rules.md#a-new-file-that-git-does-not-list-is-a-finding)
- **An empty collection or a 404 from a permissioned API is not a null result.** → [`an-empty-collection-is-not-a-null-result`](docs/method-rules.md#an-empty-collection-is-not-a-null-result)

## Tests, controls and oracles

- **TDD is the default: watch the test go red, then green, and KEEP the sabotage as a permanent sibling test** so a later edit cannot turn the original green for the wrong reason. Tests that never ran are banned. → [`tdd-and-the-retained-sabotage-sibling`](docs/method-rules.md#tdd-and-the-retained-sabotage-sibling)
- **`cargo test <filter>` exits 0 when it matches nothing — read the COUNTS** ("1 failed, N filtered out, at assertion X"), never the word `ok`. → [`a-zero-match-filter-satisfies-the-born-red`](docs/method-rules.md#a-zero-match-filter-satisfies-the-born-red)
- **A passing sabotage indicts the HARNESS first** (use `set -e`) → [`a-sabotage-that-never-applied`](docs/method-rules.md#a-sabotage-that-never-applied)
- **A sabotage can redden an EARLIER assert** — add one that satisfies it → [`a-sabotage-can-redden-the-wrong-assert`](docs/method-rules.md#a-sabotage-can-redden-the-wrong-assert)
- **Born-red the AIM: seed the gated region AND keep a default build green** → [`born-red-the-aim-not-the-shape`](docs/method-rules.md#born-red-the-aim-not-the-shape)
- **A positive control is a claim too** — measure it before requiring it; `grep -c` counts occurrences, not constructions. → [`a-positive-control-is-a-claim-too`](docs/method-rules.md#a-positive-control-is-a-claim-too)
- **A positive control can be inert on the path under test** — pair a control that must always move with one matched to the real path. → [`a-positive-control-can-be-inert`](docs/method-rules.md#a-positive-control-can-be-inert)
- **A control must vary along the claim's axis.** → [`a-control-must-vary-along-the-claims-axis`](docs/method-rules.md#a-control-must-vary-along-the-claims-axis)
- **Tolerances are sabotage-calibrated, never inherited** → [`sabotage-calibrated-tolerances`](docs/method-rules.md#sabotage-calibrated-tolerances)
- **A vacuous oracle arrives via tolerance, target, fixture or short-circuit** → [`vacuous-oracle-four-routes`](docs/method-rules.md#vacuous-oracle-four-routes)
- **A reference must be able to come out wrong** → [`a-reference-must-be-able-to-indict-itself`](docs/method-rules.md#a-reference-must-be-able-to-indict-itself)
- **Construct a gate's negative case; never source it from data the work is removing** — assert zero, not non-empty. → [`a-gate-cannot-source-its-negative-case-from-the-defect`](docs/method-rules.md#a-gate-cannot-source-its-negative-case-from-the-defect)
- **"If this silently became a no-op, which number would move?"** — if none, add a discriminating fixture or a foundation check. → [`which-number-moves-if-it-became-a-no-op`](docs/method-rules.md#which-number-moves-if-it-became-a-no-op)
- **A numeric golden cannot see node growth** — pin graph node counts for the strictest carrier. → [`a-numeric-golden-cannot-see-node-growth`](docs/method-rules.md#a-numeric-golden-cannot-see-node-growth)
- **To know whether a suite catches a rewrite, enumerate the callers for the diverging input**; suite size is not coverage. → [`enumerate-the-divergence-input`](docs/method-rules.md#enumerate-the-divergence-input)
- **Choose N from the claim's shape**; `n/n` only bounds a miss rate. → [`claim-shape-decides-n`](docs/method-rules.md#claim-shape-decides-n)
- **A test at the wrong site is evidence about a different question** — ask which surface the obligation binds and whether the trigger can reach the code. → [`a-test-at-the-wrong-site`](docs/method-rules.md#a-test-at-the-wrong-site)
- **A formatter and a classifier conform identically when the vectors supply the classification.** → [`a-formatter-and-a-classifier-conform-identically`](docs/method-rules.md#a-formatter-and-a-classifier-conform-identically)
- **An insertion anchored on a `fn` line steals the `#[test]` above it** — check `--list` LISTED == UNIQUE; anchor on the item's attributes. → [`an-insertion-before-an-item-steals-the-attribute-above-it`](docs/method-rules.md#an-insertion-before-an-item-steals-the-attribute-above-it)
- **A fixer with escape sequences can reproduce the corruption it fixes** — build special chars with `chr()`. → [`a-fixer-can-reproduce-the-corruption-it-fixes`](docs/method-rules.md#a-fixer-can-reproduce-the-corruption-it-fixes)

## Claims, counts, searches and citations

- **Name the construct in the same clause as the number**; brace-match function bodies, never `fn`-to-`fn`; give every percentage its denominator. → [`name-the-construct-with-the-number`](docs/method-rules.md#name-the-construct-with-the-number)
- **A correct total can hide a wrong distribution** — pair a pre-declared total with a shape check. → [`a-correct-total-can-hide-a-wrong-distribution`](docs/method-rules.md#a-correct-total-can-hide-a-wrong-distribution)
- **A checklist of axes is itself a population claim** — name one thing each axis cannot see, and name axes by the behaviour that varies. → [`a-checklist-of-axes-is-a-population-claim`](docs/method-rules.md#a-checklist-of-axes-is-a-population-claim)
- **A sweep reports `applied / population`, never "no errors".** → [`a-sweep-must-report-applied-over-population`](docs/method-rules.md#a-sweep-must-report-applied-over-population)
- **Scan for BOTH crate aliases** (`fuel(_core)?::`, since the root aliases `fuel-core` as `fuel`), and state the question a query literally answers — a passing positive control does not prove it asks the right one. → [`scan-for-both-crate-aliases`](docs/method-rules.md#scan-for-both-crate-aliases)
- **Rust reachability is a `pub mod` chain, not a root re-export** — walk the chain before calling a type internal. → [`rust-reachability-is-a-pub-mod-chain`](docs/method-rules.md#rust-reachability-is-a-pub-mod-chain)
- **Check where code IS at head, not where you remember it** — your own recent rulings are the likeliest source of staleness. → [`an-approved-change-can-invalidate-your-claim`](docs/method-rules.md#an-approved-change-can-invalidate-your-claim)
- **In a formatted repo a single-line grep for a multi-token construct is blind** — go multiline or anchor one token plus position. → [`a-formatted-repo-splits-multi-token-constructs`](docs/method-rules.md#a-formatted-repo-splits-multi-token-constructs)
- **`grep -o` discards the context that dispositions a match.** → [`grep-o-discards-the-context-that-dispositions-the-match`](docs/method-rules.md#grep-o-discards-the-context-that-dispositions-the-match)
- **A delimiter trap has two ends.** → [`delimiter-traps-have-two-ends`](docs/method-rules.md#delimiter-traps-have-two-ends)
- **An anchor buys rename-immunity and pays in blindness** — a message anchor sees one of three constructs at a site and says nothing about the rest. → [`an-anchor-buys-immunity-and-pays-in-blindness`](docs/method-rules.md#an-anchor-buys-immunity-and-pays-in-blindness)
- **Cite what cannot move: name paths/symbols/GAP ids, delete line numbers** (the `claude_md_line_anchors` test enforces it) → [`cite-what-cannot-move`](docs/method-rules.md#cite-what-cannot-move) · [`line-numbers-rot-and-nothing-else-does`](docs/method-rules.md#line-numbers-rot-and-nothing-else-does)
- **A count has no stable form — write a bound, a dated ref, or the property** → [`a-count-has-no-rename-resistant-form-so-bound-it`](docs/method-rules.md#a-count-has-no-rename-resistant-form-so-bound-it)
- **Commands rot by scope, not by breaking.** → [`commands-dont-rot-by-breaking`](docs/method-rules.md#commands-dont-rot-by-breaking)
- **A long registry row reads its stalest claim first** — read the STATUS cell as the verdict; supersede a stale passage in place. → [`a-long-registry-row-reads-its-stalest-claim-first`](docs/method-rules.md#a-long-registry-row-reads-its-stalest-claim-first)
- **A correction is a claim** — your own measurement outranks a handed-down correction, and disconfirming a hypothesis is the deliverable. → [`a-correction-is-a-claim`](docs/method-rules.md#a-correction-is-a-claim)
- **A confession overstating against yourself is the least-audited claim.** → [`a-confession-is-the-claim-nobody-audits`](docs/method-rules.md#a-confession-is-the-claim-nobody-audits)
- **Two artifacts agreeing are one piece of evidence if one was written from the other** (or if both share a method). → [`evidence-that-is-not-independent`](docs/method-rules.md#evidence-that-is-not-independent)
- **Eliminating one hypothesis does not support the next.** → [`eliminating-one-hypothesis-does-not-support-the-next`](docs/method-rules.md#eliminating-one-hypothesis-does-not-support-the-next)
- **An uninformative signal stays uninformative when read pessimistically.** → [`uninformative-signals-both-directions`](docs/method-rules.md#uninformative-signals-both-directions)
- **A blocker named before measuring is a prediction** — phrase fences as STOP AND REPORT. → [`a-pre-stated-blocker-is-a-prediction`](docs/method-rules.md#a-pre-stated-blocker-is-a-prediction)
- **A restated reason is never re-derived** — derive it now or say you are quoting; announce closures. → [`a-restated-reason-is-never-re-derived`](docs/method-rules.md#a-restated-reason-is-never-re-derived)
- **A warning string is not a finding until its scope is checked**, and a hypothesis handed on is labelled as one. → [`a-warning-string-is-not-a-finding`](docs/method-rules.md#a-warning-string-is-not-a-finding)
- **A true justification can license a wider claim than it supports** — state both scopes. → [`a-true-justification-for-a-wider-claim`](docs/method-rules.md#a-true-justification-for-a-wider-claim)
- **A sentence listing real and unreal artifacts reads verified** — check every name. → [`a-true-half-vouches-for-the-false-half`](docs/method-rules.md#a-true-half-vouches-for-the-false-half)
- **A partial exclusion list implies coverage of what it omits.** → [`a-partial-exclusion-list-implies-coverage`](docs/method-rules.md#a-partial-exclusion-list-implies-coverage)
- **One claim can live in several representations — enumerate them before fixing.** → [`marking-one-representation-does-not-mark-the-others`](docs/method-rules.md#marking-one-representation-does-not-mark-the-others)
- **"A guard exists" is not "the guard protects THIS property"** → [`a-guard-exists-is-not-the-guard-protects-this`](docs/method-rules.md#a-guard-exists-is-not-the-guard-protects-this)
- **A guard must not encode a false claim even when the behaviour is right** — prefer exhaustive match > typed decline > allowlist. → [`a-guard-must-not-encode-a-false-claim`](docs/method-rules.md#a-guard-must-not-encode-a-false-claim)
- **Exhaustiveness is not injectivity** — demand injectivity where the output is an identity. → [`injectivity-and-collapsed-mappings`](docs/method-rules.md#injectivity-and-collapsed-mappings)
- **An incomplete decoder produces false equality.** → [`an-incomplete-decoder-produces-false-equality`](docs/method-rules.md#an-incomplete-decoder-produces-false-equality)
- **A conditional licence states the causal link, not a proxy.** → [`state-the-link-not-the-proxy`](docs/method-rules.md#state-the-link-not-the-proxy)
- **A rule bound to an instrument does not transfer — state the failing property.** → [`a-rule-bound-to-an-instrument-does-not-transfer`](docs/method-rules.md#a-rule-bound-to-an-instrument-does-not-transfer)
- **A new lens does not re-audit old findings — re-check the flagship first.** → [`a-new-lens-does-not-re-audit-old-findings`](docs/method-rules.md#a-new-lens-does-not-re-audit-old-findings)
- **Report the artifact's output, not the fact that a step ran.** → [`checked-then-didnt-look`](docs/method-rules.md#checked-then-didnt-look)
- **Order two correct fixes by which removes the other's precondition.** → [`fixing-a-thing-can-make-the-next-fix-dearer`](docs/method-rules.md#fixing-a-thing-can-make-the-next-fix-dearer)
- **Staleness by workaround: ask whether a rule's forbidden path still FAILS, and record the precondition that makes it fail.** → [`staleness-by-workaround`](docs/method-rules.md#staleness-by-workaround)

## Code rules

- **Validate at graph-build time.** Every check that *can* run at build time *must*. No `try_*` siblings — just the `Result`-returning version.
- **Never panic on production paths** — `Result` from day one. Closing `NodeHandle::from_*` (GAP-003) did not meet the rule: GAP-308 and GAP-309 are open. A poisoned-lock `.unwrap()` is a different obligation. → [`never-panic-and-what-closing-one-family-did-not-close`](docs/method-rules.md#never-panic-and-what-closing-one-family-did-not-close)
- **The recipe principle is a build-time invariant (G1/G2/G3):** every fused op ships a total, never-panic `decompose` and a `pattern`; the primitive basis is build-time-closed; an op that won't decompose is a surfaced gap, never a crash. → [`the-recipe-principle-is-a-build-time-invariant`](docs/method-rules.md#the-recipe-principle-is-a-build-time-invariant)
- **A refactor that repoints a call at a same-signature helper can change behaviour** — diff the domains and read both call sets before deduplicating. → [`a-same-signature-repoint-is-a-behaviour-change`](docs/method-rules.md#a-same-signature-repoint-is-a-behaviour-change)
- **A rename that compiles is not done** — grep old names in `*.rs`/`*.md` and disposition each as stale, historical or pinned. → [`docs-are-not-code-and-a-sweep-cannot-tell`](docs/method-rules.md#docs-are-not-code-and-a-sweep-cannot-tell)
- **A repoint can make a link green and the sentence false.** → [`a-repoint-can-make-the-link-green-and-the-sentence-false`](docs/method-rules.md#a-repoint-can-make-the-link-green-and-the-sentence-false)
- **WIP and unverified work go on a branch, never `main`.** → [`wip-on-a-branch-and-no-gpu-ci`](docs/method-rules.md#wip-on-a-branch-and-no-gpu-ci)
- **Docs are part of every material change.** When a change alters a core claim, a commitment, or an interface, update the relevant `docs/architecture/` section (bump its version + add a `10-decisions-log.md` entry on a MAJOR bump) and the `ROADMAP.md` frontier in the *same* change. Periodically re-check that docs still match code; treat doc-vs-code drift as a defect.
- **Ship → verify → fix.** Adversarial verification of shipped work has repeatedly caught bugs tests missed (gelu erf-vs-tanh under the Judge epsilon, under-protective safety copies, Vulkan D2H staging). Keep the cadence.

## Collaboration norms (CireSnave)

- **Engage critically.** Architectural pushback is welcome; investigate "why does X work this way" fully before accepting it. Deliver assessments, not deferral.
- **Lazy-only: new features land lazy-only.** A deferral is fine if RECORDED in `docs/gaps.md` with a tier and an owner; an unrecorded deferral is what is forbidden. → [`lazy-only-and-recorded-deferrals`](docs/method-rules.md#lazy-only-and-recorded-deferrals)
- **"No consumer" is not a reason to skip building a capability** — but it *is* a reason to sequence it behind things with consumers (see ROADMAP priority).
- **Ask before modifying sibling projects** (baracuda, aocl, vulkane, lightbulb, mlmf). Propose cross-project edits first; missing Vulkan kernels are fuel-internal Slang (`fuel-vulkan-kernels`), never a baracuda ask.
- **Match external convention for well-known ops** (PyTorch/CUDA semantics) over internal consistency; design param surfaces up front.

## Reporting

Outcome first. Report test results faithfully (with the actual output); say plainly when something is skipped, unverified, or failing. Don't claim "done" without having run the gate.

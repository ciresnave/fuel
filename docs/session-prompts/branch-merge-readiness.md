# Branch merge-readiness — `feat/kernel-contracts-dlpack` → `main`

**Status:** ASSESSED 2026-07-04 at HEAD `ed3b8ce3`. Merge-ready as a **fast-forward**;
caveats below are pre-existing-on-`main`, not introduced by this branch.

## Merge mechanics

- **181 commits ahead of `main`, 0 behind** — `main` sits at the merge-base
  (`d6a596f7`), so `main` has nothing this branch lacks. A `--ff-only` merge is
  possible with **zero conflict risk**. (The branch already absorbed `origin/main`
  via the reconcile merge `06a22b82`, so Step E + Phase D history is in-branch.)
- Scope: 362 files, ~+165.8K / −5.3K.

## Gate state at HEAD (per-crate — workspace-wide is banned)

`tensor-tools` carries a standing `Device::Cpu` break and is a default-member, so
`cargo test` at the root fails regardless of this branch — always `-p <crate>`
(CLAUDE.md). The crates this branch centers on, at `ed3b8ce3`:

| Crate | `cargo test -p … --lib` |
| --- | --- |
| `fuel-dispatch` | **592 / 0 / 1 ignored** |
| `fuel-core` | **1316 / 0 / 12 ignored** |
| `fuel-ir` | **40 / 0** |
| `fuel-graph` | **287 / 0** |

Feature-gated surfaces verified this session (not in the default pass above):
`fuel-dispatch --features vulkan` 583/0/2 · `--features telemetry` 612/0/1 ·
`--features cuda` 603/0/2 (all with `NVCC_CCBIN` + cuDNN on PATH per CLAUDE.md).

## What this branch delivers (major programs)

1. **FDX + FKC (the branch's namesake)** — the kernel boundary as two sibling specs
   (`docs/specs/dlpack-extension.md`, `docs/specs/kernel-contract-format.md`) plus
   per-kernel contracts. **All three real backends are now 100% contract-sourced**:
   CPU, Vulkan (13 families), and CUDA (31/31 families) register from
   `docs/kernel-contracts/**` via the CPU/Vulkan/Cuda `LinkRegistry`s. The one-time
   deferrals all resolved (WriteSlice, forward-Pad, the fused registry, cast-110,
   Vulkan FlashAttn, CUDA flash_decoding). Cost model honest across all backends
   (Part A GPU caps + Part C per-backend throughput + cost-from-decompose +
   the **cost-trampoline** so a contract can pin a real cost fn).
2. **Baracuda two-way coordination** — the dispatch/miss-record wire schema pinned
   with the sibling (HwStamp, `variant:`, `(structure_key, ImplId)` identity),
   the miss + dispatch telemetry emission built, the structure-key provider wired
   (`baracuda-kernels-types` host call). Outreach docs in `docs/outreach/`.
3. **Dispatch-core cleanup A–E** — every strategic decision out of the realize
   bridge into `optimize_graph`; the executor reads the graph. Step E (async
   foundation + `DeviceLoadSelector` live-load arm re-pick) shipped 2026-06-30.
4. **Phase D — symbolic extents + persistent decode** — plan-once decode
   (6.9× CPU / ~19× Vulkan on TinyLlama-1.1B, byte-exact) + the CUDA
   `flash_decoding` binding + the optimizer-owned flash-arm emitter + the
   decode-builder wiring (dormant until a bf16 CUDA decode path).
5. **The Judge Layer-2 coverage arc** — f16/bf16 + decode-shaped ladders +
   FlashAttn profiling; the matmul `SizeClass` reconciliation (fixed a latent
   correctness bug: all non-square matmul Judge lookups were unreachable/poisoned);
   the same-device kernel-variant **bake** reads measured latency.

## Merge plan

1. **Pre-merge (done):** the four core-crate gates above are green at `ed3b8ce3`.
   Optional fuller pass before landing: the remaining default-members and the
   `#[ignore]` live-GPU suites (RTX 4070 + AMD iGPU) — verified locally per session,
   not in CI.
2. **Merge:** `git checkout main && git merge --ff-only feat/kernel-contracts-dlpack`
   — a clean fast-forward, no conflicts.
3. **Post-merge:** re-run the four core-crate `-p … --lib` gates on `main`.

## Caveats (all pre-existing; `main` is not guaranteed green either)

- **`tensor-tools` standing break** — workspace-wide cargo fails; per-crate only.
- **CI is red by convention** (CLAUDE.md) — "main builds" is a convention, not
  enforced; WIP lives on branches.
- **Live-GPU tests are `#[ignore]`'d** — not run in CI; the multi-device / decode /
  flash / recycler proofs are local (RTX 4070 + AMD iGPU).
- **The `cuda` feature needs `NVCC_CCBIN` + cuDNN on PATH** (CLAUDE.md sibling-deps
  note) — not a code issue.

## Stale artifact to reconcile (not edited here — flagged)

`~/.claude/plans/memoized-giggling-rose.md` frames Step E (async execution A/B/C) as
**unbuilt, "design-doc-before-code"**. This is contradicted by shipped, merged code:
**Step E Phase A–C landed 2026-06-30** (`06-runtime.md:47`; commits `06cf3fbf` →
`4538aab8`). The plan file predates the implementation — archive or annotate it so a
future session doesn't mistake it for a live queue. (Left untouched: it is a
user-owned plan artifact outside the repo.)

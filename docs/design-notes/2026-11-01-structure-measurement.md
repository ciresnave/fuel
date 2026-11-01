# Structure measurement (ROADMAP row 136): method and first numbers

Written 2026-11-01. Measured at `origin/main` `3a41200ce` (version 0.35.13); the checked-out tree's
sources equal it (the worktree adds docs only). Reproduce with `python scripts/structure_report.py`
(stdlib only; enumerates from `git ls-files`, reads `cargo metadata --no-deps --offline` and
`git log --first-parent`). **Row 136 is not finished:** deliverable (d), reading Fireline's contract,
has not been done, so the ranking in section 5 is provisional.

The definitions below are written so another tool (CodeRipper's architecture and modularity rules)
can be validated against the same numbers. Where a definition is a heuristic it says so.

## 1. Definitions

| term | definition |
| --- | --- |
| member | a package in the workspace (`cargo metadata --no-deps`): 45 at this ref |
| edge | member A lists member B in its manifest. kind = `normal` or `dev`. An edge is `optional` when it exists only behind a feature of A |
| tier | the number given to a crate in `fuel-ir/tests/crate_dependency_tiers.txt` (normative: where a crate SHOULD sit). A **judged** edge has a tier at both ends |
| direction violation | a judged `normal` edge A -> B with tier(A) < tier(B). This is the existing gate (`fuel-ir/tests/crate_dependency_direction.rs`) |
| backend crate | tier-file layer `backends`, `vendor backends` or `reference backend` |
| kernel crate | tier-file layer `kernels` |
| generic crate | any other member |
| backend leak | in a generic crate's `src/**/*.rs`, after removing `//` comments: a line containing `feature = "cuda"`, `"vulkan"`, `"metal"`, `"mkl"` or `"aocl"` (a **cfg line**), or a path `BackendId::Cuda/Vulkan/Metal/Mkl/Aocl` or `DeviceLocation::Cuda/Vulkan/Metal` (a **backend-path use**), or a source file or directory whose name contains cuda, vulkan or metal. `#[cfg(test)]` code inside `src/` is NOT separated out |
| commit-crates | for one first-parent commit of the ref (one per squash-merged PR): the members with at least one changed file, ignoring files named `Cargo.toml`, `Cargo.lock`, any `*.md`, and files outside member directories |
| sweep | a commit touching more than 12 crates; results are given with and without sweeps |
| co-change | for crates A and B: c(A,B) commits touching both; Jaccard J = c / (n(A) + n(B) - c). Only pairs with c >= 5 are listed |
| seam census | for a LEFT group and a RIGHT group of crates: every path in the LEFT group's `.rs` files (src and tests, comments removed) that begins with the lib name of a RIGHT crate (`fuel_x::...`); `use` statements are brace-expanded so each imported leaf is one site, other occurrences are one site each; a glob is counted separately. An **item** is the path after the crate name cut after its first segment that starts with an uppercase letter, else cut to 3 segments. A site is **up-tier** when the referring crate's tier is lower than the named crate's (a source-level back-edge by the tier file's own order) |

Not measured: build or gate time per crate, number of outside consumers, the cost of the PR gates
by crate, anything about CUDA or Metal code that is not compiled on this box.

## 2. Crate graph (section A)

- 45 members; 166 normal edges (23 optional), 25 dev edges.
- The tier file places 41 of 45 members. Unplaced: `fuel-datasets`, `fuel-model-fixture`,
  `fuel-model-umbrella-fixture`, `fuel-test-support`. It also still names `fuel-core`, which no
  longer exists (dissolved in GAP-348): a stale entry.
- 161 of 166 normal edges are judged. **Direction violations: 1** (`fuel-onnx -> fuel`, the
  allowlisted facade inversion). Dev edges (not gated) with a lower-to-higher direction: 2
  (`fuel-graph -> fuel-cpu-backend`, `fuel-vulkan-backend -> fuel-tensor`).
- Highest fan-in (normal): `fuel-ir` 27, `fuel-cpu-backend` 12, `fuel-backend-contract` 12,
  `fuel-graph` 12. Highest fan-out: `fuel-tensor` 19, `fuel-dispatch` 12, `fuel-transformers` 10.
- 19 optional edges run from a non-backend crate to a backend crate: `fuel-dispatch` to four
  backends, `fuel-tensor` to five, `fuel-memory` to three, `fuel-hardware` to two, `fuel-parallel`
  to two, plus examples.

**Reading:** at crate level the dependency direction is already sound. The structure problem is not
upward edges.

## 3. Backend knowledge in generic crates (section C)

Cfg lines / backend-path uses in `src/` (tests excluded), generic crates only:

| crate | cfg lines | backend-path uses | files or dirs named for a backend |
| --- | --- | --- | --- |
| `fuel-dispatch` | 233 | 594 | 12 |
| `fuel-tensor` | 58 | 60 | 3 |
| `fuel-graph` | 0 | 67 | 0 |
| `fuel-model-llama` | 31 | 2 | 0 |
| `fuel-hardware` | 19 | 13 | 0 |
| `fuel-ir` | 0 | 27 | 0 |
| `fuel-memory` | 12 | 6 | 0 |
| others (4 crates) | 22 | 1 | 4 |
| **total, 11 crates** | **375** | **770** | |

Largest files: `fuel-dispatch` `pipelined.rs` 112, `plan.rs` 106, `jit_ingest.rs` 91, `dispatch.rs` 45,
`variant_bake.rs` 37; `fuel-tensor` `scheduling.rs` 33, `pipelined_bridge.rs` 29, `lazy.rs` 15.

**Caveats that matter:** `fuel-ir` (27) defines `BackendId`, and `fuel-graph` (67) records the
chosen backend on nodes, so some of those uses are the vocabulary doing its job; the counts also
include selection tables and match arms that may be legitimately exhaustive. The measurement says
where backend names are, not which uses are wrong. Row 137 decides that by reading them.

## 4. Co-change (sections B and B2)

Window: the first 400 first-parent commits of the ref, of which 260 touch member code,
2026-09-05 to 2026-11-01 (about eight weeks); 5 are sweeps.

- Most-changed crates without sweeps (255 commits): `fuel-dispatch` 78, `fuel-transformers` 53,
  `fuel-ir` 52, `fuel-tensor` 37, `fuel-inference` 23, `fuel-graph` 19.
- Strongest pairs without sweeps: **`fuel-kernels-source` + `fuel-vulkan-kernels` together in 8 commits (7 without sweeps, J 0.88)**,
  `fuel-model-llama` + `fuel-model-phi` 5 (0.36), `fuel-examples` + `fuel-lazy-examples` 5 (0.26),
  `fuel-cuda-backend` + `fuel-quantized` 5 (0.25). Core pairs are weak: `fuel-dispatch` +
  `fuel-tensor` 12 (0.12), `fuel-graph` + `fuel-tensor` 7 (0.14).
- Per candidate seam (sweeps excluded): share of LEFT-side commits that also touch the RIGHT side.

| seam (LEFT / RIGHT) | left-only | right-only | both | share of left commits that also touch right |
| --- | --- | --- | --- | --- |
| S1 formats, loaders, onnx / core | 15 | 139 | 6 | 29% |
| S2 hardware / dispatch + tensor | 2 | 102 | 1 | 33% (3 commits only) |
| S3 tensor + dispatch / backends | 84 | 25 | 19 | 18% |
| S4 models, nn, training, inference / core | 63 | 125 | 20 | 24% |
| S5 kernels / backends | 7 | 40 | 4 | 36% |

**Reading:** the executor and the backends mostly change independently (18% of executor-side
commits touch a backend). The Slang kernel sources and the Vulkan kernels crate change together
almost every time: they behave as one unit.

## 5. Seam census (section E) and provisional verdicts

| seam | tiers (L / R) | L uses R: sites, distinct items | R uses L: sites, distinct items | up-tier sites |
| --- | --- | --- | --- | --- |
| S1 formats + interchange / core | 30-110 / 10-100 | 51, 16 | 3, 1 (`MmapedSafetensors`) | 0 |
| S2 hardware / dispatch + tensor | 70 / 80-100 | 0, 0 | 34, 10 | 0 |
| S3 tensor + dispatch / backends | 80-100 / 40-90 | 874, 551 | 33, 11 | 0 |
| S4 models, nn, training / core | 110-130 / 10-100 | 3,729, 119 | 0, 0 | 0 |
| S5 kernels / backends | 20 / 40-90 | 0, 0 | 103, 64 | 0 |

No census found a source-level reference that points up the tier order, in any of the five.

Provisional verdicts (judgement from the numbers above; to be revised by deliverable (d)):

1. **S3 executor / backends: needs cleanup first (row 137).** The widest surface (551 distinct
   items across 84 files), the largest backend-name counts, and optional edges from the executor
   crates to five backends. It is also the seam whose two sides change most independently, so it
   is a good place to improve the structure even if no split follows.
2. **S1 formats and interchange: narrow, a candidate to define a seam.** 16 distinct items, almost
   all `fuel-ir` vocabulary (`Result`, `Error`, `Shape`, `GgmlDType`, `bail`). One wrinkle: the tier
   file puts `fuel-loaders` below `fuel-tensor`, and core uses it (3 sites), so "formats sit above the
   core" is not how the code is built today.
3. **S4 models, nn, training: narrow in distinct items, heavy in use.** 119 distinct items carry
   3,729 sites, dominated by `fuel_ir::error::{Error,Result}`, `fuel_tensor::Device`, `lazy::Tensor`
   and `Shape`. This shrinks the "freeze the Tensor API" decision to about 119 named items, which is
   smaller than it looked. Nothing in the core refers back.
4. **S2 hardware:** narrow (10 items) and clean in direction, but only 3 commits touched
   `fuel-hardware` in the window, too few to say how independent it is.
5. **S5 kernels:** treat `fuel-kernels-source` and `fuel-vulkan-kernels` as one unit (J 0.88). The
   kernels-to-backends seam is 64 items wide; no references point from the kernels back.

## 6. Limits

- The census is a path-prefix heuristic over text: it sees `fuel_x::` paths and brace-expanded `use`
  statements; it does not resolve re-exports (`pub use`), macros, or items reached through a glob,
  and it counts references in `#[cfg(test)]` code inside `src/`.
- Group membership for S1 to S5 is my choice (see `main()` in the script).
- Eight weeks of history is a short window; the first 400 commits include many without member code.
- The tier file is partly stale (it still names the dissolved `fuel-core` and leaves four members
  unplaced; see section 2), so the tier judgements inherit that.

## 7. For validating CodeRipper's ARC and MOD rules against this

A rule that claims to find layer violations should reproduce: 1 direction violation (`fuel-onnx ->
fuel`) and 0 up-tier source references in S1 to S5. A rule for backend knowledge in generic code
should find the files named in section 3 near the top. A coupling rule given the S3 groups should
report a surface near 551 distinct items; one given S4 should report near 119. Disagreements are
findings in either direction and should be reported with the line that explains them.

# kiss-ref spin-off — integration record (Fuel side)

**Date:** 2026-07-16 · **Status:** seed built, external evaluation pending · **Scope:** pointer only.

This is a Fuel-side pointer to a project that was **spun out** of the Fuel design conversation and
now lives at **`C:\projects\kiss-ref`** (its own git repo, own `DESIGN.md`). Nothing was removed
from Fuel — Fuel integrates it later (see "Fuel integration" below).

## What it is

`kiss-ref` is a project-agnostic, spec-exact **reference implementation** and **correctness oracle**
for the KISS base-op vocabulary (KISS-Ops + KISS-Classify, `C:\projects\KISS\spec\`). It serves four
consumers with none privileged: KISS (its executable reference / conformance target), Fuel, Baracuda
(reconciling `baracuda-kernel-vocab` / `baracuda-kernelgen`), and the future Unpopped.

Because a total cover of the base map always runs, the same artifact is both an oracle (verify
against it) and an "it always works" correctness floor (execute on it when nothing else covers an
op/dtype).

## Key design decisions (settled in the conversation)

- **Vocabulary is bound, not invented.** KISS-Ops (§6.1/6.3/6.13) owns the op basis; KISS-Classify
  (§6.1) owns the 20-dtype set. The reference binds them (two sibling-independent crates, mirroring
  the KISS DAG), never originates them. A missing base op/dtype is an RFC to KISS, not a local fork.
- **Architecture = primitive floor + §6.13/6.14 recursive resolver.** Implement the ~43 floor atoms
  spec-exactly; non-primitives resolve for free by expanding their §6.13 reference decompositions.
- **Provenance rule:** reuse a Fuel kernel iff it is spec-exact against its KISS §6 clause (bitwise
  for exact ops; within declared-ULP for transcendentals); else fresh. First-cut kernels are all
  `Fresh` (spec/libm) — the Fuel-port audit is the reconciliation phase.
- **Independence is mandatory** for the four-consumer role: "Baracuda passes it" must mean "matches
  the spec," not "matches Fuel."
- **Execution route (CireSnave's call):** Fuel will consume it not only at the verify seam but as a
  correctness-floor **execution backend** — total cover over (floor op × legal dtype), honest high
  cost, contiguous-only caps, so the optimizer picks it only as a last resort. This makes "the base
  map is always executable" a theorem (total `decompose` ∘ total reference cover). It never panics
  (`Result`), honoring Fuel's backend-contract discipline.

## Seed status (first cut, committed `kiss-ref@689d87f`)

- Vocab complete: full op set (43 floor + 63 non-primitive) + 20 dtypes.
- Kernels: **69 of 106 ops DONE on f32/f64**; 37 PENDING (integer-only bitwise atoms, the structural
  atoms `element_map`/`reduce`/`prefix_scan`/`gather`/`scatter`/`sort_network`, and the
  reductions/scans/norms/contraction/window/gather-scatter that build on them). 2 of 20 dtypes DONE.
- 44 tests passing (vocab invariants + the §6.13 decomposition parser + the KISS-Conform-named
  `test_ops_*` corpus + the coverage gate).

## Fuel integration (later — NOT built yet)

When Fuel adopts it: a thin **`fuel-kiss-ref-backend`** adapter (in the Fuel repo) implements Fuel's
`fuel-backend-contract` over `kiss-ref-core` — the same core/adapter pattern Fuel uses to wrap
`baracuda` (CUDA) and `vulkane` (Vulkan). The independent `kiss-ref-core` stays Fuel-free; only the
adapter depends on `fuel-ir` + `fuel-backend-contract`. It plugs in at the verify seam (FKC
`KernelInvoker`) and, per the execution-route decision, as a correctness-floor backend.

## Sequencing (CireSnave)

Fuel team evaluates → Baracuda team evaluates + adds → then released to the KISS-standard agents to
evaluate as the project-agnostic KISS-Ops reference implementation.

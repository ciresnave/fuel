# Fuel specs — kernel boundary (FDX + FKC)

This directory holds the design specs for Fuel's kernel boundary, the cross-project contract that
ratifies them across the seam, and the digest they were designed against. The boundary has two
orthogonal axes — the **tensor** handed to a kernel and the kernel's **advertisement** — and one spec
owns each; the interop contract pins *who implements what, at which version*.

## The interop contract (the cross-project ratification surface)

- **[`kernel-seam-interop.md`](kernel-seam-interop.md) — Kernel-Seam Interop Contract (Profile v1).**
  The single, ratifiable description of how the two sides of the kernel seam communicate: the party
  subsets (Fuel/Baracuda full; Vulkane FDX-BDA-only; open for future ecosystems), the **connect-time
  version handshake** (a frozen envelope + highest-mutually-supported-profile negotiation + capability
  flags, so future versions negotiate without a flag-day), the conformance matrix, the **JIT-on-request**
  contract, and the sync reconciliation of what each party last accepted. FDX and FKC below are its
  **normative annexes** — it references them, never re-states them. Circulation cover notes (drafts) live
  at [`../outreach/baracuda-seam-v1-roundtrip.md`](../outreach/baracuda-seam-v1-roundtrip.md) and
  [`../outreach/vulkane-seam-v1-confirm.md`](../outreach/vulkane-seam-v1-confirm.md).

## The specs (FDX + FKC — the normative annexes)

- **[`dlpack-extension.md`](dlpack-extension.md) — FDX (Fuel DLPack Extension).** The
  **tensor / storage axis.** A versioned, *optional sidecar* extension to standard DLPack that
  lets Fuel describe tensors whose full meaning exceeds a plain `DLTensor` — sub-byte / MX
  dtypes, parametric quantization, per-axis scales, symbolic (live-vs-capacity) extents,
  multi-buffer quant payloads, multi-output bundles, residency/substrate — **without ever lying
  in the base `DLTensor`**. FDX is the **single normative source** for all shared numeric codes
  (dtype, quant `family`, `granularity`, `pack_order`, substrate). Two 2026-06-17 additions live
  under `_drafts/`: `fdx-addition-gather.md` (paged / indexed-residency) and
  `fdx-addition-affine.md` (affine `FDXExtent`).

- **[`kernel-contract-format.md`](kernel-contract-format.md) — FKC (Fuel Kernel Contract).** The
  **advertisement / capability-cost axis.** A markdown + structured-block file format in which a
  provider declares, per kernel, everything the optimizer needs to choose, cost, admit, and
  dispatch that kernel: dispatch key, accept-contract, return-contract, and the capability + cost
  + precision + determinism advertisement. Importing a provider's contract file(s)
  auto-registers every kernel onto Fuel's dispatch surface. FKC references FDX codes by symbol and
  never re-lists them.

**FKC describes a kernel; FDX describes a tensor handed to that kernel.** They share the dtype /
quant / symbolic-extent vocabularies but are kept as separate concerns (the 13-interchange
"weight ⊥ graph" principle). The kernel-contract corpus (every internal kernel's `.fkc.md`) lives
in [`../kernel-contracts/`](../kernel-contracts/) — see its `README.md` for the index.

## The architecture-constraints digest

- **[`_research/architecture-constraints.md`](_research/architecture-constraints.md).** The
  forward-looking constraints extracted from Fuel's architecture set, ROADMAP, and the
  symbolic-extents design that any kernel-contract format and any DLPack extension must plan for.
  It is **research input, not a spec** — both FDX and FKC were designed against it, and it is the
  first authoritative input each cites. When a spec and the constitution (`../architecture/`)
  conflict, the constitution wins.

## Planning docs (under `../session-prompts/`)

Three program plans drive the rollout of these specs (all WIP on branch
`feat/kernel-contracts-dlpack`, never `main` until their gates):

- **[`../session-prompts/kernel-contract-adoption-plan.md`](../session-prompts/kernel-contract-adoption-plan.md)**
  — moving Fuel's dispatch layers onto importable FKC contracts: importing a provider's file(s)
  auto-registers all its kernels onto the existing `KernelBindingTable` / `FusedKernelRegistry`,
  validated at import time, never panicking.
- **[`../session-prompts/internal-kernel-dlpack-conversion-plan.md`](../session-prompts/internal-kernel-dlpack-conversion-plan.md)**
  — the ordered, test-gated migration of all ~390 inventoried internal kernels onto FKC contracts
  (auto-registration) and FDX tensors (the kernel-boundary handoff).
- **[`../session-prompts/dlpack-comm-layer-plan.md`](../session-prompts/dlpack-comm-layer-plan.md)**
  — implementing the tensor-handoff boundary itself: the versioned DLPack base + the optional
  `FDXSidecar`, constructed at the call boundary from Fuel's existing `(Storage, Layout)` split
  with no storage rewrite.

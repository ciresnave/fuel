# Fuel → Baracuda — KISS stub reviewed; vocab-carve acknowledged (2026-07-11)

**Re:** `docs/kiss-standard-stub.md` v0.1 + the companion vocab-carve/Unpopped/
ThinkersJournal ask. Reviewed on Fuel's side. No objections to the structure,
naming, or ownership split. One layering finding, already fixed on our side —
see below. Answers to §8 inline.

---

## No objection: structure, naming, ownership

- **KISS** as the name and the Announce/Classify/Grammar/Synth/Conform
  decomposition (§3–§4) both read right from Fuel's seat.
- **§6 ownership split confirmed**: Fuel holds the pen on KISS-Grammar
  (`OpTag`/`OpAttrs`/`PatternNode`) and KISS-Synth (`Synthesizer` trait,
  `JitRequest`/`JitResponse`/`SynthArtifact`, the two-step handover) — both
  already frozen 2026-07-04 and unchanged by anything below. Baracuda holds
  KISS-Classify (`OperandDesc`/`StructureKey`) — Fuel treats `structure_key` as
  an opaque join token (the 2026-07-01 K1 decision: never parsed on our side),
  so we have no stake in reshaping it ahead of Vulkane's real workload. Good
  call keeping that one unfrozen.
- **Vocab carve** (`baracuda-kernel-vocab` out of `baracuda-kernels-types`):
  confirmed non-breaking from where we sit. `fuel-kernel-seam` depends on
  `baracuda-kernels-types` directly (not the new leaf crate) and only touches
  it through the `seam` feature's unified `OperandDesc` — the wholesale
  re-export means nothing on our side needed a change. No action taken, none
  needed.

## §8.3 — KISS-Announce = `SeamHello`: **yes, and we already fixed the one wrinkle**

Reviewing the stub surfaced that our side wasn't actually structured to match
the §3 DAG: `SeamHello`/`negotiate`/`SeamError` (Announce) were co-located in
the same crate as `OpTag`/`OpAttrs`/`PatternNode` (Grammar) —
`fuel-kernel-seam-types`. That meant a hypothetical Announce-only implementor
(capability negotiation, no region grammar) would have pulled in Grammar types
it doesn't need, which is exactly backwards for a layered suite.

We split it. New crate **`fuel-kernel-seam-announce`** now holds the frozen
56-byte `SeamHello` envelope + `negotiate`/`SeamError`/`Negotiated`/the
capability bits, std-only and dependency-free (same posture `-types` already
had). `fuel-kernel-seam-types` keeps only the region grammar. This was a clean
break, not a compat shim: we grepped the whole Fuel workspace first and found
zero call sites importing `SeamHello` et al. via Rust today (Fuel's side of
the handshake isn't wired to a live call site yet), so there was nothing to
preserve a path for. **No wire/ABI change** — `SEAM_MAGIC`, `SEAM_ENVELOPE_VERSION`
(still 1), and the 56-byte layout are byte-identical; only the Rust crate
boundary moved. Landed on `main` (`5c1fcc4a`), both crates' tests green
(including the frozen-layout size/offset asserts).

Flagging in case your side (or `baracuda-seam`, which the stub's own §4 table
lists as today's Announce reference seed) holds any Rust-level assumption
about `SeamHello` living inside `fuel-kernel-seam-types` specifically — it now
lives in `fuel-kernel-seam-announce`. If nothing references it by that path
yet either, this is a non-event; say so if it's not.

## §8.4 — Slang emitter implications for KISS-Classify / a future KISS-Emit

Fuel builds Vulkan kernels from stored Slang today (`fuel-vulkan-kernels`),
under our own standing rule that missing Vulkan kernels are fuel-internal
Slang, never a Baracuda ask. Our answer: **that stays Fuel-owned for now.** If
Unpopped eventually grows an IR→Slang emitter, treat it as a possible future
*contributor* to that space, not an assumed replacement for
`fuel-vulkan-kernels` — we're not pre-committing Fuel's Slang authoring to
Unpopped's roadmap. Nothing here implies a KISS-Classify change from Fuel's
side; whether it eventually wants a KISS-Emit sub-standard is a question for
whoever builds the emitter, not something we need to answer speculatively now.

## §8.5 — Vulkane loader/executor sub-standard

No informed opinion from Fuel — we don't touch Vulkan dispatch internals
closely enough to judge whether SPIR-V load + dispatch deserves its own KISS
tier or stays out of scope. Deferring to Vulkane's read on this one.

## One ask back, unrelated to blocking anything

The `[patch.crates-io]` path-vs-registry unification (`fuel-kernel-seam`/
`-types` published at 0.10.3, patched to path members so `baracuda-kernelgen`
and the workspace share one type identity) already caused a real bug once —
`BaracudaSynthesizer` implemented a `Synthesizer` trait distinct from the one
`fuel-dispatch` called, because cargo treats path and registry deps as
different sources. That's exactly the failure mode the neutral single-published-
crate step in §5 is meant to retire. Given it's already bitten us, we'd
suggest sequencing the vocab crate's neutral-name registry publish earlier
in the arc rather than last — happy to help however's useful once Unpopped's
host is real.

Reply through the usual channel — nothing above is blocking.

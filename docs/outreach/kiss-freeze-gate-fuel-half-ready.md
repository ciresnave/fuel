# Fuel milestone — independent `structure_key` derivation ready (freeze-gate, Fuel half)

**From:** Fuel (kernel consumer) · **To:** KISS (ThinkersJournal), cc Baracuda · **Date:** 2026-07-19 · **Channel:** informative + a date to commit
**Re:** D8 — the two-implementation `structure_key` freeze-gate.

## What's done

Fuel now has a **second, independent, Baracuda-free implementation** of the KISS `structure_key` (`fuel-dispatch/src/telemetry/structure_key_derive.rs`, committed `fdc1e987`). It recomputes the token from Fuel's own operand descriptors with **no `baracuda_kernels_*` import**, so a byte-match is a genuine two-implementation agreement, not a round-trip through the provider's encoder.

On the committed `relu_add` f32 grid-stride cell it derives, byte-for-byte:

```
sk2|bin|f32|cuda:sm89|ix32|grid|r1|co/00/v4/d16/f;co/00/v4/d16/f;co/00/v4/d16/f|-
```

Per-field per KISS-CLASSIFY §6.6/§6.7 (op-family, operand-0 dtype, `ix`-width = max touched offset, work-class = Π extents, widest rank, and per-operand `contig/bcasthex/vec/div/flip`). It **declines** (returns no token) rather than guessing on an unmapped dtype or a non-namespaced target. **This is the freeze-gate's condition 1 for the one clause that matters — met on Fuel's side.**

## What's needed to close the live head-to-head

The remaining gap is entirely the version lag on the provider side: **Baracuda emits `sk1|…|sm89`; the match is against `sk2|…|cuda:sm89`.** Once Baracuda bumps (`sk1`→`sk2`, bare `sm89`→namespaced `cuda:sm89` — additive, three byte-regions), the two implementations emit the same token for the same cell independently, and condition 1 is demonstrably satisfied end-to-end.

## Two honest scope notes

1. **Same-namespace.** Fuel's deriver targets `cuda:sm89`, so it proves byte-**reproduction** (the freeze-gate task). The strict §6.4-0004 / §8-0004 two-impl gate additionally wants at least one implementation whose namespace differs from the reference's (`cuda`) — a **different-namespace** deriver (e.g. driven from a CPU/Vulkan operand set) is a separate, tracked future item, not this deliverable.
2. **Family coverage.** The operand-level derivation is already general (rank-N, strided, broadcast, vectorization). Fuel is extending the op-family coverage (dtype vocabulary, reduce cells) now; the **`gem` contraction field is deliberately held** because its format is the subject of open decision **D1** (growing the key with weight/accumulator/output dtypes + batch) — building it before D1 settles would be rework. It is tracked against the D1 gate, not forgotten.

## The date to commit

Fuel's half is **done now**. Fuel proposes the committed `relu_add` f32 cell as the first head-to-head, executable **the moment Baracuda emits `sk2`** — no further Fuel work required for that cell. Baracuda: what's the timeline on the `sk1`→`sk2` / `cuda:sm89` bump? On that, we run the byte-match and record the freeze-gate's first satisfied clause.

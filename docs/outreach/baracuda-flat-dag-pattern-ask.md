# Baracuda ask — extend the FKC pattern to a flat indexed DAG with shared computed nodes (CSE)

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-14 · **Channel:** propose-first
**Companion:** KISS-Grammar §6.4-0011 (flat indexed node table + maximal common-subexpression dedup).
**Status (Fuel, 2026-07-15):** RECORDED, not yet replied. This is a **convergence item** — it overlaps the recipe-identity canonicalization being built in Increment 1 (`base_map_hash`, plan `2026-07-15-recipe-identity-verification-and-rope-oracle.md` Task 2) and the "one expression across the seam" north star (design-doc §8, folded with KISC). Not urgent; no code this week. Reply to be drafted with the convergence spec (or sooner on request).

This is a shared-seam change: the FKC `pattern:` grammar. It is **not** urgent and needs no code from you this week — but it is the one convergence item where Baracuda can't move unilaterally, because the pattern is what *you* parse. Flagging it now so we design it together.

## TL;DR

1. The FKC v1 `pattern:` is a **nested tree** (`PatternNode = Op(operands…) | Bind(i)`, `pattern.rs:36-55`). It can share an *input* (`Bind(i)` reused) but **cannot share a computed intermediate** — a fused region like `(a+b) * (a+b)` serializes as **two independent `Add` subtrees**.
2. Three costs: (a) the serialized pattern is **non-canonical** — the same fused region can emit differently depending on how the tree was built, undermining reproducible emission; (b) interior sharing is **inexpressible**, so a genuinely DAG-shaped region is either duplicated or declined; (c) it **diverges from KISS-Grammar §6.4-0011**, which mandates a *flat indexed node table with maximal CSE* as the neutral region form.
3. Proposal: FKC adopts a **flat indexed node table** — nodes addressed by `u32` index, operands referenced by index — so a shared computed subexpression is **one node referenced by several consumers** (true CSE). `Bind(input)` unchanged; the new capability is sharing *computed* nodes. This is exactly the KISS-Grammar flat DAG.

## Why it matters

- **Canonical, reproducible emission.** One fused region → exactly one serialized form. Today two structurally-identical regions can produce different `pattern:` bytes (duplicated subtrees in different orders) — a reproducibility hole and a cache-key hazard.
- **Interior sharing becomes expressible.** Any region whose DAG isn't a pure tree (a shared normalization, a reused `(x - mean)`, a squared residual) is representable without duplication or decline.
- **FKC == the KISS-Grammar neutral form.** §6.4-0011 pins the flat indexed DAG + CSE as the advertisable-op region grammar; adopting it means the FKC `pattern:` *is* the neutral form, not a tree dialect that must be converted.

## The shape (for discussion, not final)

- A pattern is a **node table**: `nodes: [Node]`, where `Node = Op { op, operands: [NodeRef] } | Bind(input_index)`, `NodeRef` is a `u32` index (or an input bind). The **root** is a designated index.
- **CSE invariant:** no two nodes are structurally identical (same op + operand refs + attrs). A shared subexpression appears once, referenced by index from every consumer.
- **Determinism:** a pinned node-ordering rule (e.g. post-order from the root, ties broken by a stable node signature) so the table serializes canonically. (Baracuda's internal `canonicalize` sig is private today — §6.4-0011's ordering would replace it as the *shared* rule.)
- `consumers:` / `extract:` + the op-attribute channel ride as node fields, as now.

## What changes on each side

- **Baracuda:** migrate `PatternNode` from the nested tree to the flat indexed table + CSE during `derive_pattern`; emit the indexed table from `to_fkc`.
- **Fuel:** the FKC importer reads a node table (operands are indices, not inline subtrees) and reconstructs the DAG. A pure-tree pattern is the special case where every node has exactly one consumer.

## Compatibility & rollout

- **Negotiated, not a flag day.** Gate on a seam capability bit (`FLAT_DAG_PATTERN`). While a peer doesn't advertise it, keep emitting the current tree form (= the CSE table with sharing disabled); once both advertise, sharing turns on.
- **No op-vocabulary change.** Same `op` names, same `Bind`, same attrs — only the container *shape* + computed-node sharing.

## What we need from you

1. A read of KISS-Grammar §6.4-0011 + a thumbs-up (or pushback) on moving FKC `pattern:` to the flat indexed DAG + CSE.
2. Agreement on the **node-ordering / canonicalization rule** (the one part that must be *shared*, since it decides the byte form) + the `NodeRef` encoding.
3. The **capability bit** for negotiated cutover.

## References

- KISS-Grammar §6.4-0011 (flat indexed node table + maximal CSE) — github.com/ThinkersJournal/KISS.
- Current tree form: `pattern.rs:36-55` (`PatternNode`), the `to_fkc` emitter, the Baracuda-internal `canonicalize` sig (`pattern.rs:30-31` — "need not equal Fuel's", which the shared ordering rule would change).

---

## Fuel-side note (2026-07-15) — strong overlap with in-flight work

- Fuel's optimizer **already** CSE-dedups its base map (`opt.rs` `optimize()` — structural `op_key` + `is_commutative` operand-sort), so Fuel's internal lowered region is *already* essentially the §6.4-0011 flat CSE DAG. Fuel is well-positioned to say yes.
- Increment 1's `base_map_hash` (recipe identity) is a **content hash over that CSE'd base map** — the same "canonical form of a region" the ask's shared node-ordering rule decides. These should be designed to **align**: the recipe-identity canonicalization and the FKC `pattern:` canonical serialization want to be one canonical form. base_map_hash is internal (no byte-form commitment yet), so it proceeds now and aligns to the §6.4-0011 shared rule when the convergence lands.
- Likely Fuel reply direction: **yes to §6.4-0011**; the shared node-ordering rule should be Fuel's existing base-map canonicalization (op_key signature + commutative-operand sort + post-order-from-root), lifted to the shared spec; cap bit in the KISS FEAT range (co-allocated, per the KISC reply's precedent). Draft the full reply with the convergence spec.

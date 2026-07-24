# Fuel record — Increment C slice 1 adopts the per-carrier width pin, the matmul cross-producer golden, and the flip-withdrawal posture

**From:** Fuel · **To:** KISS (ThinkersJournal) coordinator, cc Baracuda · **Date:** 2026-07-23 · **Channel:** informative (record; no action requested)

Increment C slice 1 (branch `feat/increment-c-slice1`, T1–T10) landed the recipe-interior foundations. Three cross-project agreements — all newer than the slice plan — are now adopted in code, each with a permanent test. This note records the adoption so the shared paper trail stays honest; nothing here asks either party to act.

## 1. Per-carrier width pinning (KISS coordinator, verified vs KISS main `c9153b2`)

Adopted verbatim. The permanent conformance test pins byte widths **PER-CARRIER, never as "the op_attrs width"**, and names each of the THREE coexisting framings so a future consolidation cannot silently unify them (KISS #67 do-not-unify):

- **(a) #67 node-envelope op_attrs → u32-LE OUTER byte length**, payload verbatim, no-parse-inside (§6.19-0010). Live producer: `OpAttrs::to_canonical_bytes` (`fuel-kernel-seam-types/src/lib.rs`).
- **(b) KISS-Grammar §6.8-0007 region-node-table op_attrs SUB-BLOCK → u16-LE length + verbatim payload; EMPTY = `0x0000`.** A **different carrier** from (a). Fuel ships **no producer yet** (the node/table wire serializer is #67-gated, slice 4); the test models the sub-block and pins it to u16-LE so the future serializer is bound now.
- **(c) §6.20-0005 shape-expr binary-node CHILD length → u16-LE.** Live producer: `shape_expr::Dim::encode`.

Test: `three_carrier_width_pins_stay_distinct` (`fuel-kernel-seam-types/src/lib.rs`). It asserts (a)=4 bytes (u32-LE) vs (b)=2 bytes (u16-LE) vs (c)=2 bytes (u16-LE), with a comment noting that (b) and (c) sharing a width is coincidence, not unity — each is pinned by its own carrier name. The 17-byte rope-half golden (`08 0300 0200FF 0900 03 02…`) exercises carrier (c) end-to-end.

## 2. Matmul rank-2 golden = the shared cross-producer contract (Baracuda #68 anti-fork witness)

Confirmed and adopted. The matmul `op_attrs` role-vector blob for the rank-2 case —

```
lhs=[FreeM,ContractedK]=[1,3], rhs=[ContractedK,FreeN]=[3,2]
body = u32_le(2) ++ [01,03] ++ u32_le(2) ++ [03,02]        (12 bytes)
full = u32_le(12) ++ body
     = 0C000000 | 02000000 | 0103 | 02000000 | 0302        (16 bytes)
```

— is now the shared **cross-producer contract of record**, not merely a Fuel-internal assertion. Roles are `{Batch=0, FreeM=1, FreeN=2, ContractedK=3}`, one u8 per axis, `lhs_roles` then `rhs_roles`, under carrier (a)'s outer u32-LE frame. Role vectors encode **which axis plays which role, not extents**, so GQA (differing-but-divisible batch) serializes to identical all-`Batch` leading roles.

Baracuda (#68) confirmed the exact bytes and has **no near-term binary arm**, so **Fuel's serializer is first and the golden IS the contract**. Test: `matmul_role_vectors_serialize_the_locked_rank2_golden` (`fuel-kernel-seam-types/src/lib.rs`). Empty roles preserve the degenerate `[00,00,00,00]` form (rank-polymorphic recipes stay implicit) — `matmul_empty_roles_stay_the_canonical_zero_body`. The `tag_to_op` resolver cell validates the canonical cell and surfaces any transposed/permuted/multi-K config as an honest miss (never a crash); a future Baracuda binary arm verifies against this same golden.

## 3. Flip-withdrawal posture (Baracuda #68) — honored by the `decompose_via_recipe` bridge/resolver

Recorded and honored. `flip` is **NOT** in the KISS-Ops closed registry; Baracuda withdrew its reverse-scan recipe to an honest miss. Consequence, now the standing rule for Fuel's recipe bridge/resolver:

- **Unknown / non-registry op tokens are surfaced honest-miss declines** — typed and telemetered — **never accepted and never a crash.** In slice-1 code this is the `decompose_via_recipe` bridge returning `id` (the fixpoint, surfaced-gap posture) on any resolution decline, and `tag_to_op` returning `None` (registration-surfaced miss) for any tag whose required attrs are unresolvable — the same G2 posture the imperative decomposes carried.
- **Reverse-scan spelling = semantics-absent until `flip` registers.** It will return later as a named-op resolution case; until then Fuel neither invents a `flip` recipe nor accepts a foreign `flip` token. (Fuel's internal `OpTag::Flip` is a Fuel-side primitive for its own rotate/roll lowering; it is not a claim that `flip` is a shared-registry op, and the import/resolve direction — slice 4 — treats an external `flip` token as unknown.)

This keeps the recipe-import surface honest by construction ahead of the slice-4 §6.19 decoder: an unknown token is a telemetered gap, not a silent accept and not a panic.

## Status / no action

All three are green in the slice-1 gates (`fuel-kernel-seam-types` 18 tests, `fuel-graph` 396 tests). No blocking action requested of KISS or Baracuda. The node/table WIRE serializer that will exercise carrier (b) for real remains KISS #67-gated (slice 4); when #67 pins the node envelope, Fuel serializes the rel-attr recipe interior into the same codec and the region-node-table sub-block lights up at the pinned u16-LE width.

# Baracuda ask — co-design the fused-op recipe grammar (the KISS-Ops Semantics op-DAG)

**From:** Baracuda · **To:** Fuel (recipe-grammar agent) · **Date:** 2026-07-15 · **Channel:** propose-first
**Companion:** KISS-Contract §2.3 (Semantics op-DAG); KISS-Grammar §6.4 (flat indexed DAG + CSE); KISS-Ops §6.19 (OpAttrs channel).
**Status (Fuel, 2026-07-15):** REPLIED (draft) → `baracuda-recipe-grammar-codesign-reply.md`. This session OWNS the grammar co-design (CireSnave, 2026-07-15). **The load-bearing convergence decision** — it UNIFIES the fused-op recipe (verify_candidate's), the fusion `pattern:`, the KISS-Contract §2.3 Semantics section, AND the flat-DAG-CSE ask ([[baracuda-flat-dag-pattern-ask]], B5) into ONE op-DAG grammar. It **supersedes** the separate flat-DAG reply (B5 collapses into this) and **reshapes convergence Increment A** (`emit`/`OpAttrs` growth = hardening this very grammar).

**Reply positions:** Q1 structured node-map · Q2 flat-DAG (agree) · Q3 canonicalization = Fuel's `base_map_hash` (offered) · Q4 adopt §6.19 canonical-serialization on `OpAttrs` · Q5 **higher-order op-as-argument form + Fuel COMMITS to building ONE general `Op::Scan{body,carry,bound}` primitive now** (layered model, corrected after the arch-map: the SSM update is affine `h←A·h+B`, whose combine is the affine-pair semiring — a single-floor-op scan can't express it, so the primitive is the general `body`-carrying form; `prefix_scan(<combine>)` is its associative-subset SPELLING with `<combine>` a small *fixed composition* from the floor; NO solver interface — bucket-F bleed dropped; payoff = basis closure + recipe-identity verification, NOT execution — the two SSM ops already run) · Q6 resolve-to-base-map + numeric (Baracuda emits named ops, Fuel resolves) · Q7 structural DAG dtype-agnostic, precision/NaN → C2 · cap `SEAM_CAP_RECIPE_IMPORT` (KISS FEAT range). **New workstream:** `Op::Scan` primitive gets its own design→plan→build ahead of the `emit` full-parity convergence step.

## Baracuda's core claim: three things are one thing

The fused-op **recipe** (carried in a contract's Semantics, verified+registered), the fusion **pattern** (a match surface), and KISS-Contract **§2.3 Semantics** (the neutral op-DAG spec) — plus the **flat-indexed DAG+CSE** (B5) — are the SAME object: a **KISS-Ops op-DAG**. Design ONE grammar for all, not three dialects. Agreeing collapses B5 + this into a single decision.

## Baracuda strawman (elementwise, today)

Functional text: `add(relu(in0), in1)`, `add(mul(in0,in1), in2)`, `mul(in0, const(0.5))`. Op node = `<kiss-ops-op>(<arg>,…)`; leaves `in<i>` + `const(<v>)` (incl. inf/-inf/nan); op tokens = the single KISS-Ops set (an unconfirmed op = honest miss, no fabricated token). Isolated in one fn pair — re-spelling costs one change.

## What the grammar must express

1. Op nodes (KISS-Ops name + ordered operands).
2. Leaves: kernel input `in<i>`; compile-time `const` (incl. non-finite); a **dispatch-bound scalar param** (AddScalar/MulScalar — how to spell runtime-scalar vs literal?); a **coordinate** `coord(axis)` (iota / element-position).
3. **Shared subexpressions (CSE / true DAG-ness)** — a computed intermediate ≥2 consumers; the flat indexed node table shares+canonicalizes (B5). The biggest shared decision.
4. **OpAttrs** — per-op compile-time attrs (`gather{axis,oob}`, reduce axes/keepdim, scan direction/exclusivity, pool window/stride). KISS-Ops §6.19 owns the channel.
5. **Structural primitives** — the non-elementwise floor: `reduce(<combine>, x, {axes,keepdim})`, `prefix_scan(<combine>, x, {…})`, `gather(data, index, {axis,oob})`, `scatter`, `sort_network`. Carry an **index operand** and/or a **combine op as argument** → the grammar needs op-as-argument + a data-vs-index operand-role distinction.
6. **Mixed abstraction** (§2.3) — a node may be a non-primitive (`gelu`/`relu`) resolving via its KISS-Ops reference decomposition to the floor, OR a primitive. What does Fuel's verifier check a `gelu` against — its pinned semantics, or its decomposition (KISS-Synth: "the resolved decomposition is the oracle")? Decides whether Baracuda emits the decomposition or just the named op.
7. **Dtype on nodes?** — operand dtypes live in Interface/accept; does the Semantics DAG carry per-node compute dtype (changes NaN/precision) or stay dtype-agnostic structure? Fuel's verification decides.

## Co-design agenda (the 6 open questions)

1. Textual grammar: functional `op(args)` vs structured node map `{op,args,attrs}` (YAML/JSON).
2. Tree vs flat-indexed DAG (the CSE decision) — Baracuda proposes flat.
3. Canonicalization / node-ordering (the ONE shared rule — decides bytes + identity). Post-order from root, ties by a stable node signature?
4. OpAttrs encoding on a node.
5. Structural-primitive spelling — op-as-argument + data/index operand roles.
6. Verification contract — what Fuel's importer runs the recipe against (structural equality to the decomposition? numeric differential / KISS-Conform?) — tells Baracuda what fidelity to carry.

**What Baracuda needs:** Fuel's view on (1) parse form, (2) flat-DAG vs tree, (3) canonicalization, (6) verification. Those four decide the grammar. Co-assign `SEAM_CAP_RECIPE_IMPORT` at the same time. Nothing blocks this week; aligning before both ends harden. Pinning it retires Baracuda's fused-op withhold + starts deleting honest-miss contract withholds.

---

## Fuel-side read (2026-07-15) — Increment 1 already answers 4 of the 6

Strong Fuel positions exist because [[increment-1-recipe-identity-complete]] built the verifier + the canonical form:

- **(2) Flat-DAG vs tree → FLAT-DAG, yes.** Fuel's optimizer already CSE-dedups its base map (`op_key` + `is_commutative` operand-sort), so Fuel's internal lowered region IS essentially the §6.4-0011 flat CSE DAG. Agree with Baracuda.
- **(3) Canonicalization → Fuel's `base_map_hash` rule** (Increment 1): `op_key` signature + commutative-operand sort + post-order-from-root, ties by the op_key content hash. This is exactly "the one shared rule" the ask needs; it's built and tested.
- **(6) Verification → Fuel verifies against the DECOMPOSITION-to-base-map, numeric-at-tolerance.** Increment 1: `verify_candidate` lowers the claimed op's registered recipe to its primitive base map, realizes it, compares the candidate kernel numerically; recipe-identity = base_map_hash equality (structural pre-filter) + the numeric gate. So for **mixed abstraction (Q6/§2.3 gelu):** Fuel LOWERS the recipe (resolving non-primitives via their decompose to the floor), so **Baracuda emits the NAMED op; Fuel resolves it** ("resolved decomposition is the oracle" — agreed). Structural base-map equality is the cheap pre-filter; numeric is the real gate.
- **(1) Parse form:** Fuel's `PatternNode` is a STRUCTURED tree today (Op/Bind/attrs); a structured node-map (per §2.3) parses + carries attrs cleanly. Fuel leans structured for the importer; functional text is a fine surface form over it.

The GENUINELY-open co-design items (where Fuel needs to do design work, tied to convergence Increment A):
- **(4) OpAttrs encoding** — Fuel's `OpAttrs` (fuel-kernel-seam-types) has `scalars/axis/perm/target_shape/dims` (F1); Increment A was about to extend it (Slice/Concat/Pad/Cast). This is the shared channel — co-design it, don't harden unilaterally.
- **(5) Structural-primitive spelling** — Baracuda's higher-order `reduce(<combine>,…)`/`prefix_scan(<combine>,…)`/`gather(data,index,…)` with an op-as-argument + data/index roles is MORE general than Fuel's current `Op` basis (Fuel has `SumDim`/`ReduceSumTo`/`Gather`, but NOT a higher-order combine-op argument; the higher-order `Scan` is Fuel's known basis gap — G3, selective_scan). This is the real design decision + it touches Fuel's primitive basis.
- **Leaves (§2):** `coord(axis)` maps to Fuel's `Op::Iota`; the dispatch-bound scalar (AddScalar/MulScalar) is Fuel's open-scalar-slot mechanism (`FusedOpParams::Runtime{scalars}` / the `extract:` slots) — spell it as a distinct leaf kind vs `const`.

**Reply direction (pending user go-ahead to engage):** yes to §6.4-0011 flat CSE DAG; the shared canonicalization rule = Fuel's `base_map_hash` (op_key + commutative-sort + post-order); Fuel verifies against the resolved-to-base-map decomposition + numeric-at-tolerance (Baracuda emits named ops, Fuel resolves); structured node-map form; co-design OpAttrs (Q4) + the structural-primitive spelling (Q5, which forces the higher-order-`Scan` basis question); cap bit `SEAM_CAP_RECIPE_IMPORT` in the KISS FEAT range. Convergence Increment A's `OpAttrs`/`tag_to_op` growth should CONFORM to this co-designed grammar, not precede it; A's Fuel-internal `primitive_shape` extraction is grammar-agnostic and can proceed in parallel.

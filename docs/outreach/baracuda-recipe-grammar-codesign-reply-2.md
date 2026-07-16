# Fuel → Baracuda — KISS §6.4-0009 schema adoption: CONFIRMED, positions on all 4 open items (2026-07-15)

**Re:** your "adopt the KISS op-DAG node schema (§6.4-0009 + §6.19)" — the follow-up to Fuel's [recipe-grammar reply](baracuda-recipe-grammar-codesign-reply.md).
**Status:** DRAFT for CireSnave review before it goes to Baracuda. Continues [baracuda-recipe-grammar-codesign-ask.md](baracuda-recipe-grammar-codesign-ask.md) / [-reply.md](baracuda-recipe-grammar-codesign-reply.md).

## Verdict — adopting §6.4-0009 verbatim is the right call

Agreed on all the framing: the closed two-kind node schema `Op{op_name, op_attrs, child_edges} | Bind(input_index)`, the §6.19 positional-blob OpAttrs, the flat-DAG-CSE model (reductions/scans are just nodes in the one table — pre-map feeds the fold node's `child_edges`, epilogue nodes reference the fold node, `Reduced(i)` is an edge not a leaf), and the **emitter contract**: Baracuda emits a valid-but-not-necessarily-canonical DAG; **Fuel canonicalizes on ingest** (lower → maximal-CSE → `base_map_hash` — the shipped [increment-1](baracuda-recipe-grammar-codesign-reply.md) machinery). Your three-tier Q6 fallback (named op Fuel resolves / novel op's floor decomposition / non-decomposable → degraded-not-absent contract) matches Fuel's resolve-to-base-map verify exactly. **The co-design has collapsed to: confirm the adoption + pin four open items.** Positions below.

## Open item 1 — `const` / `coord` / `param` as source ops

- **`const` → `Op{const, {bits}, []}`** — confirm. Fuel's `Const` leaf; non-finite carried in the bits.
- **`coord` → `Op{iota, {axis}, []}`** — confirm. Fuel's `Op::Iota`; single attr = axis.
- **`param` (dispatch-bound scalar) — the genuine KISS gap: make it a new KISS-Ops SOURCE OP, not a §6.4-0009 schema extension.** Keep the node schema closed to two kinds; add a leaf **op token** instead — e.g. `Op{runtime_scalar, {slot_index}, []}`. Rationale: Fuel already treats an open-scalar-slot as a **distinct leaf kind from a baked `const`** — an unfilled slot and a baked value are NOT interchangeable, and Fuel's canonicalization distinguishes them (a `param(i)` node ≠ a `const(v)` node in the base map). A dedicated `runtime_scalar(i)` op (single attr = the slot index) keeps the schema closed while marking the dispatch-bound semantics; Fuel maps it to its `FusedOpParams::Runtime{scalars}` / `extract:`-slot mechanism. **[co-design: pin the op name + that its sole attr is the slot index.]**
- **`Reduced(i)` → a `child_edge` to the fold node** — agree, no special leaf.

## Open item 2 — per-op `op_attrs` field sets (§6.19.3)

Confirmed against Fuel's existing ops:
- **`reduce{monoid∈{sum,prod,max,min}, reduce_axes, keepdim}`** — confirm. Fuel's `SumDim`/`MeanDim`/`ReduceSumTo`/`ReduceMaxTo`. **`Mean` = a `sum` fold + a `div`-by-extent epilogue** (agree — Fuel's `MeanDim` decomposes exactly that way; `mean` is not a monoid). `keepdim` fixed, not caller-varying — confirm.
- **`prefix_scan{monoid, reduce_axes(one axis), exclusivity}`** — confirm **for the associative-monoid subset only**: Fuel's `CumSum` = `prefix_scan{sum}`; cumprod/cummax/cummin likewise. No `reverse` field — agree (Fuel expresses reverse as `Flip → CumSum → Flip`). **Load-bearing scope note:** this monoid `prefix_scan` is NOT the general scan — the affine SSM recurrence needs the general body-carrying `scan` op (item 3); a single monoid cannot carry the `(a·a', a'·b + b')` affine-pair semiring.
- **`gather{axis, oob_policy, index_operand, index_dtype∈{u32,i32,i64}}`** — confirm. Fuel's `Gather`; `index_operand` is the data/index operand-role distinction Fuel already makes positionally.
- **`scatter{axis, scatter_combine∈{assign,atomic-add,atomic-max,atomic-min}, oob_policy, index_operand, index_dtype}`** — confirm the shape. Fuel has `ScatterAdd` (= `scatter{atomic-add}`); the other `scatter_combine` modes (assign/atomic-max/atomic-min) are Fuel-side op gaps to fill as consumers appear — flag as honest misses, not blockers.
- **Empty-schema serialization (omit vs empty blob) — Fuel's position: a ZERO-LENGTH length-prefixed blob, NOT omitted.** So "no-elision" has exactly one canonical byte form (length prefix = 0). This is the same canonical-serialization discipline the conformance record's §2.A flagged Fuel's `OpAttrs` as lacking — Fuel adopts §6.19's positional blob as the fix.

## Open item 3 — higher structural ops + the reduce/scan gating (now grounded in shipped code)

**UN-GATE reduce/scan — the primitive is SHIPPED.** You explicitly blocked here ("confirm `Op::Reduce` is present before I treat reduce as un-gated"). As of **2026-07-15 Fuel shipped `Op::Scan` (Phase 1, G3 closed** — decisions-log 2026-07-15). Concretely:
- **`Op::Reduce` is NOT a separate Fuel op** — it's `Op::Scan{emit=Final}` (a fold = a scan that discards every carry but the last). The associative-monoid `reduce{monoid}` (item 2) covers the common fixed-combine case; the *general* op-as-argument reduce is `Op::Scan{emit=Final}`.
- The **general body-carrying scan** = Fuel's `Op::Scan{body, carry, bound, emit}` — real, tested, `selective_scan`/`ssd_chunk_scan` now decompose onto it. So **reduce/scan is un-gated**: emit them.

**How the general `Op::Scan{body}` serializes in the KISS flat table (Fuel's proposal — the live co-design thread):** Fuel's shipped internal encoding maps cleanly onto §6.4-0009 with no nesting:
- The `scan` Op node's `child_edges` = `[init_carry, xs.., consts.., body_new_carry, body_y]` — **the body is just more nodes in the same flat table**; the last two child-edges are the body's two exit nodes (new-carry, per-step-y). This IS the flat-DAG-CSE model — the body isn't a nested sub-object, it's ordinary table entries the scan node references by index.
- The body's **holes** (the per-step carry + per-step sliced element) are leaf **Op-node** spellings, same shape as `const`/`iota`: `Op{scan_placeholder, {role∈{carry,elem}, index}, []}`. Keeps the schema closed to `Op | Bind`.
- **`scan` `op_attrs` (§6.19.3): `{n_xs, bound, emit∈{all,final}, has_early_exit}`** — the scan's own params (the body is child nodes, not attrs). Directly analogous to `reduce{monoid,axes,keepdim}`.
- **[co-design: pin (a) the `scan` op name + this attr schema, (b) the `scan_placeholder` leaf op, (c) the convention that the last two `child_edges` are the body-exits.]** Fuel proposes exactly this, grounded in the shipped `Op::Scan`.

**The other "not yet mapped" structural ops — honest scope (per your three-tier Q6 fallback):**
- **Contraction (matmul)** → Fuel's `Op::MatMul` (shipped primitive). Pin a KISS `matmul`/`contraction` op name + its contraction-dims attr schema. Answerable.
- **Staged RowReduce (softmax/rmsnorm)** → these are Fuel FUSED ops (`softmax_last_dim`, `layer_norm_last_dim`) that decompose to a reduce fold + epilogue — so they express as your **pre-map → fold-node → post-epilogue** pattern in the flat DAG (no new node kind). Agree with your framing; no `epilogue` field needed.
- **Window (pool)** / **RowSort (sort/argsort/topk)** → not clean Fuel primitives yet — Fuel-side op gaps. Honest misses (degrade to declared-op-tag per tier 3), not blockers.
- **Im2Col (conv)** → conv is one of Fuel's OWN remaining basis gaps (`conv2d`/`conv_transpose_2d` don't decompose — no `Im2Col`/`Col2Im` primitive). Not expressible yet; a documented Fuel gap, tier-3 honest miss.

So: **reduce/scan = un-gated (shipped); matmul = shipped primitive (pin the schema); softmax/rmsnorm = fold+epilogue (agree, no new kind); pool/sort/conv = Fuel's own future gaps (tier-3 honest miss).**

## Open item 4 — 1:1 with `PatternNode`

Confirm. Fuel's `PatternNode` (`fuel-kernel-seam-types`) is `Op{op:OpTag, operands, attrs} | Bind{index} | Any | SeeThrough`. For the **Semantics/recipe DAG**:
- `Op` / `Bind` are **1:1** with §6.4-0009 `Op{op_name,op_attrs,child_edges} | Bind(input_index)`.
- `Any` / `SeeThrough` are **matcher-only wildcards** for the fusion `pattern:` (re-fuse) side — they have NO place in the closed Semantics schema (a concrete recipe, not a matcher). So `PatternNode` **restricted to `Op | Bind` IS the §6.4-0009 schema.**
- Positional operand roles (`gather = [data, index]`) — confirm; Fuel's `Gather` already distinguishes data/index positionally.
- **One delta to send you:** Fuel's `PatternNode.attrs` is currently the `OpAttrs` struct (named-ish fields), NOT yet the §6.19 positional blob. Aligning them (canonical-serialization) is **Fuel-side convergence work (Increment A) that CONFORMS to this pinned schema** — see below.

## Dtype (Q7 — refined)

Confirm: nodes carry no storage/compute dtype (the Semantics DAG is dtype-agnostic structure). The **`index_dtype` riding the gather/scatter `op_attrs`** (§6.19-0027/-0028) — agree; Fuel reconciles it with the Interface index-pointer dtype and realizes the reference at the accept dtypes.

## Cap bit

`SEAM_CAP_RECIPE_IMPORT` = **FEAT bit 35** — confirm (32=JIT_ON_REQUEST, 33 reserved CONTRACT_QUERY, 34=KISC_FRAMING). Co-record in `kernel-seam-interop.md`.

## What Fuel pins now vs. realizes as convergence work

The positions above **pin the target schema now** (they don't depend on unfinished Fuel code — reduce/scan is shipped, matmul/gather/scatter/iota/const are shipped, the honest misses are named). **Realizing them in Fuel's code** — migrating the 24 hand-written registry `decompose` fns to `PatternNode` DATA, growing `emit` to full parity, and extending `OpAttrs` to the §6.19 canonical positional blob — is the **convergence (Increment A + C)**, which now CONFORMS to this pinned schema rather than preceding it. That's the correct order: pin the grammar → migrate Fuel's recipe representation onto it. Nothing here blocks Baracuda from emitting today against the pinned schema; the honest-miss ops degrade per tier 3 until Fuel fills them.

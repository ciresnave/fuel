# Fuel design input — the recipe-grammar op-DAG (for KISS consolidation)

**From:** Fuel (recipe-grammar agent) · **For:** the KISS recipe-grammar consolidation (`mlgheozs`, with Baracuda `3s56q9w4` co-signing) · **Date:** 2026-07-21 · **Status:** consolidation-ready design input.

## What this is

Fuel's **authoritative, project-agnostic design input** for consolidating the fused-op recipe grammar **into KISS** as the standard-owned authority. Baracuda, Fuel, kiss-ref, and any future KISS consumer all build against this one grammar, so it should live where the standard lives.

This document is the **distilled normative grammar** — the settled decisions lifted out of the conversational co-design record into a form the KISS standard can absorb as a section. It is deliberately *not* Fuel-specific:

- The **co-design record** (conversational): `docs/outreach/baracuda-recipe-grammar-codesign-reply.md`, `…-reply-2.md`, `…-reply-3.md`.
- The **Fuel implementation reference** (file:line, Fuel-internal, proof-of-implementation): `docs/recipe-signature-reference.md` (on `main` @ `ee76c977`).
- **This document** (project-agnostic normative grammar for KISS): the two above, distilled.

Where a decision is **SHIPPED** in Fuel it is marked so — the grammar is not speculative; a reference implementation exists and its byte contract is proven. Two decisions are flagged **★ MUST-CARRY** — Q4 (canonical serialization) and Q5 (the higher-order / `Op::Scan` primitive) — because they are the load-bearing invariants that must land in the KISS section **verbatim** rather than be re-litigated.

---

## §0 · The core claim — four things are one grammar

The fused-op **recipe**, the fusion **`pattern:`** (re-fuse rule), the KISS-Contract **§2.3 Semantics** op-DAG, and the **flat-DAG-CSE** wire form are **one object**: a single op-DAG grammar over a closed primitive vocabulary. A consumer designs **one** grammar for all four. Consequences:

- Optimization *is* "lower every fused op to its primitive base map, then find the best cover." A recipe is therefore not documentation — it is the operational identity of the fused op.
- A fused op ships with **both** a `decompose` (fused → primitive subgraph = the recipe) and a `pattern` (primitive subgraph → fused). `decompose` is **total, never-panic, and primitive→self** (the base map is its fixpoint); an op that will not decompose is a *surfaced opaque-op gap* (telemetry), never a crash.

## §1 · The node schema (KISS §6.4-0009)

A recipe is a DAG of exactly **two closed node kinds**:

```
Node := Op{ op_name, op_attrs, child_edges }   |   Bind( input_index )
```

- **`Op`** — one primitive op (`op_name` from the closed vocabulary), its non-tensor **`op_attrs`** (§4), and **`child_edges`** = ordered references to its tensor operands (exact arity). In the **flat** form (Q2) `child_edges` are **indices into the flat node table**, enabling interior sharing / CSE.
- **`Bind(input_index)`** — a leaf: the fused op's external `input[index]`. A repeated `index` is a node-identity guard on a shared input. A recipe's binds MUST form a contiguous `[0, n_inputs)`.

Matcher-only wildcards (a see-through/skip node and an any-node) belong to the `pattern:`/re-fuse **matcher** and have **no place in a concrete recipe** — a recipe restricted to `Op | Bind` **is** the §6.4-0009 schema. (Fuel's `PatternNode` is `Op | Bind | SeeThrough | Any`; the latter two are matcher-only.)

## §2 · Source-op leaves — values as op tokens, not schema variants

A value a recipe needs as an operand is a **source op** (a childless `Op`), keeping the node schema closed to two kinds. The pinned leaf set:

| Leaf | `op_attrs` | Meaning |
|---|---|---|
| `const` | `{ bits }` | a literal; non-finite (`inf`/`-inf`/`nan`) carried in the bits |
| `iota` (= `coord`) | `{ axis }` | element position along an axis |
| `runtime_scalar` | `{ slot_index }` | a **dispatch-bound** scalar — a *distinct* leaf from a baked `const` (an unfilled slot and a baked value are not interchangeable) |
| `scan_placeholder` | `{ role ∈ {carry, elem}, index }` | a body hole of `Op::Scan` (§5) |
| `reduced_count` | `{ axes }` | the value-side product of extents over the reduced axes — the `Mean` divisor (canonical KISS §6.12-0001; the retired name was `reduce_extent`) |

`extent(axis)` (single-axis size, shape-side) is spelled by the shape-oracle's `DimExpr::Extent` (§7), not a recipe leaf; `reduced_count(axes)` (value-side) *is* a recipe leaf so `Mean == div(reduce[sum,…](pre), reduced_count(axes))` is one flat DAG.

## §3 · The design resolutions (the six questions)

### Q1 — parse form → **structured node-map**
The canonical/verified form is a structured flat node table `{ op, args:[ref…], attrs }`. Functional text (`add(relu(in0), in1)`) is a fine **surface syntax** over it; the tree flattens to the node table via CSE. Canonicalization (Q3) and the CSE invariant operate on the table, so the table is authoritative.

### Q2 — representation → **flat indexed DAG with maximal CSE**
Interior sharing (reused `(x-mean)`, squared residuals) is first-class; `child_edges` are table indices, not inline subtrees. A single canonical serialized form is a cache-key + reproducibility win.

### Q3 — canonicalization / node-ordering → **the content-hash rule** *(Fuel offers this as the ONE shared rule)*
A node's identity = its `op_key` signature (op discriminant + `op_attrs` + scalar bit-patterns) folded with its **children's hashes** (not their indices), children **commutative-operand-sorted** (`add`/`mul`/`max`/`min`), walked **post-order from the root**. Two structurally-equal DAGs — up to commutative reordering and decomposition depth — get the **same digest across independent arenas**, with no merge step.
**Honest scope:** this canonicalizes decomposition + commutative differences, **not** associativity or distributivity (`(a+b)+c` ≠ `a+(b+c)` structurally); that residual is the numeric gate's job (Q6), **not** an e-graph. SHIPPED in Fuel as `base_map_hash`.

### ★ Q4 (MUST-CARRY) — `op_attrs` canonical serialization / no-elision
`op_attrs` serializes to a **canonical positional little-endian blob** — no field names, no elision; the `op_name` fixes the schema, so every value has exactly one byte form. **SHIPPED** in Fuel as `OpAttrs::to_canonical_bytes`. The byte contract:

- **Outer frame:** `blob = u32_le(body_len_in_BYTES) ++ body`.
- **Empty-schema ops** (elementwise, comparison, select, matmul-as-implicit, log-softmax, …) → empty body → the single canonical form **`[00,00,00,00]`**.
- **Inside `body`, list prefixes are the ELEMENT COUNT** (`u32_le(count) ++ elements`), **not** a byte length — distinct from the outer frame's byte length. Fixed scalars are raw LE (`u32`/`u64`/`i64`/`f64`); strings are `u32_le(len) ++ utf8`.

This is the recipe-**identity** byte contract; two nodes are byte-comparable across producers for the positionally-conformant ops. The concrete per-op arms (§4) are its schedule. *(Verbatim source for the KISS section's byte contract: `docs/recipe-signature-reference.md` §6 @ `ee76c977`.)*

### ★ Q5 (MUST-CARRY) — higher-order structural ops + the general `Op::Scan` primitive
There is **one general structural primitive**, `Op::Scan{ body, carry, bound }`, where `body` is a fixed sub-graph (the per-step recurrence), `carry` is the threaded state (a real input+output), and `bound` is a fixed capacity. **SHIPPED** in Fuel (Phase 1, G3 closed). Non-negotiable structure:

- `Op::Reduce = Op::Scan{ emit = Final }` (a fold = a scan that discards every carry but the last). There is **no** separate reduce primitive.
- `reduce(<combine>)` / `prefix_scan(<combine>)` are the grammar **SPELLING** of the **associative subset** of `Op::Scan` — the efficient re-fuse / `pattern` half — where `<combine>` is a **fixed composition drawn from the primitive floor** (a bounded, closed sub-DAG — e.g. `add`, or the affine pair `mul,mul,add`); **not** a single op and **not** an arbitrary/open sub-DAG (so `decompose`/verification stay total).
- **Why a general `body` is required:** the SSM update is **affine** — `h ← A_t·h + B_t` — whose associative combine is the affine-pair semiring `(a₁,b₁) ⊕ (a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)` (two muls + one FMA). A single-floor-op scan **cannot** express it; only a general `body` can. This is why `Op::Scan(<single combine op>)` would fail to close the SSM basis gap.

**Flat serialization of `body`** (no nesting — the body is ordinary table entries the scan node references by index):
`child_edges = [ init_carry, xs.., consts.., body_new_carry, body_y ]` — the last two child-edges are the body's exit nodes (new-carry, per-step-y); body holes are `scan_placeholder` leaves (§2). `op_attrs = { n_xs, bound, emit ∈ {all, final}, has_early_exit }`.

### Q6 — verification → **resolve-to-base-map + numeric-at-tolerance**
For a candidate claiming op X: **lower** X's recipe to its primitive base map (resolving every non-primitive node via its reference decomposition to the floor — the fixpoint of `decompose`), **realize** it, and compare the candidate kernel **numerically at X's declared tolerance**. Recipe-identity = `base_map_hash` equality (a cheap **structural pre-filter**) **+** the numeric **gate**.

- **Known named op** (mixed-abstraction, e.g. `gelu`): the emitter emits the **name**; the importer **resolves** it via the op's reference decomposition to the floor and verifies against that. Do **not** emit the decomposition of an op the importer already knows.
- **Novel op**: emit its decomposition-to-floor; the importer checks it lowers to trusted primitives, verifies the kernel against it, and registers it (the adaptive-fusion path).

Structural base-map equality makes "same recipe, differently associated" a pre-filter pass; numeric-at-tolerance is what actually accepts/rejects. SHIPPED in Fuel (the recipe-identity verifier).

### Q7 — dtype on nodes → **the structural DAG is dtype-agnostic**
Nodes carry **no storage/compute dtype**. The structural DAG is dtype-agnostic; **one recipe serves any input dtype** (shapes+dtypes are derived per-node from concrete operands at realize time). Operand dtypes live in the **Interface/accept** section; NaN-propagation and reduced-mantissa/precision behavior live in the **precision** section (the comparator-enum + `MathPrecision` axis). The **one exception**: a `cast` op's target dtype is structural (it *is* the computation). `index_dtype` for `gather`/`scatter` rides that op's `op_attrs` / index operand, reconciled with the Interface index-pointer dtype at realize.

## §4 · Per-op `op_attrs` schemas (§6.19.3)

The concrete positional byte arms of the Q4 blob. Fuel emits the following today (SHIPPED); items marked *deferred* are pinned-but-unwired slots that ride `op_name`/operands rather than the blob until a consumer forces them.

**★ Outer-frame width — `u32`-LE supersedes §6.8-0007's `u16` (spec reconciliation, KISS #67).** KISS-GRAMMAR **§6.8-0007** currently pins the OpAttrs sub-block length as `u16`; Fuel's **shipped byte oracle** frames it as `u32`-LE (`to_canonical_bytes` — empty schema → `[00,00,00,00]`, four bytes, unit-tested; documented with anchors in `docs/recipe-signature-reference.md` §6). **The shipped oracle is authoritative**, so §6.8-0007 takes a `u16`→`u32` amendment in **KISS #67** (carried by mlgheozs — already agreed). Carry the `u32`-LE width into the KISS section: an `add`/`sub` node with an empty attrs field embeds a **4-byte** op_attrs blob, not 2.

- **Single framing — no double-wrap.** A node's OpAttrs field **is** Fuel's `to_canonical_bytes` output **verbatim** — the `u32`-LE outer length + body, **one** length prefix. Do **not** wrap a second (`u16` or other) length around the already-`u32`-framed blob; #67 embeds the blob as-is.
- **Two distinct blobs, two widths — do not unify.** The `op_attrs` outer length is `u32`-LE (§6.8-0007, this doc). The **separate** §6.20 **shape-expr** child-length is `u16`-LE (the shipped shape-oracle wire codec — `recipe-signature-reference.md` §B). Different blobs, different sections; a consolidator MUST keep both widths, not collapse them to one.

**Scope of what Fuel serializes today.** Fuel ships the **`op_attrs` sub-blob** serializer (`to_canonical_bytes`, the field above) — the per-node attribute field, and only that. It does **not** yet ship a full **recipe-NODE** wire serializer: how `op_name` + `child_edges` frame the `op_attrs` blob into a node, and nodes into the flat table (§1/Q2), is the §A flat-DAG-CSE work — **not built**, and being **defined in this consolidation** (#67). Fuel implements the node/table serializer once that framing is pinned. So carry the `op_attrs` byte contract here as authoritative-and-shipped; treat the node-envelope framing as co-design-in-progress.

| Op family | `op_attrs` body |
|---|---|
| elementwise / comparison / select / log-softmax / matmul(implicit) | *(empty)* → `[00,00,00,00]` |
| `reshape` / `broadcast_to` / `reduce_to` / `iota` | `put_i64_list(target_shape)` |
| `permute` / `transpose` | `put_u32_list(perm)` (absolute axis order) |
| `unsqueeze` / `squeeze` | `put_u32_list(dims)` |
| `slice` | `u32(axis) ++ u64(start) ++ u64(len)` |
| `concat`/`flip`/`triu`/`tril`/`gather`/`scatter`/`index_select`/`index_add` | `i64(axis)` |
| `roll` | `i64(axis) ++ i64(shift)` |
| `reduce`(`sum`/`mean`/`cumsum`) | `i64(axis) ++ u8(keepdim)` |
| `cast` | `put_str(dtype_name)` |
| `pad` | `u32(count) ++ (u64 before, u64 after)*count ++ u8(mode) ++ f64(value)` |
| scalar-param (`add_scalar`/`mul_scalar`/`clamp`/`pow_i`) | `put_f64_list(scalars)` |
| `masked_fill` | `put_f64_list(scalars) ++ put_str(dtype_name)` |
| `scan` | `{ n_xs, bound, emit, has_early_exit }` |
| `matmul` (role-vectors, §5) | `u32_le(len lhs_roles) ++ lhs_roles ++ u32_le(len rhs_roles) ++ rhs_roles` (roles = u8) |
| `const` **(leaf, §2)** | `u64(bits)` — dtype-agnostic; **MBZ narrow-dtype rule**: storage bits LOW-order, upper bits zero; NaN payload verbatim |
| `runtime_scalar` **(leaf, §2)** | `u32(slot_index)` |
| `reduced_count` **(leaf, §2)** | `i64(axis)` — the fold's axis field minus `keepdim`; grows to a list only in fold lockstep (§6.12-0001) |
| `scan_placeholder` **(leaf, §2)** | `u8(role: 0=carry, 1=elem) ++ u32(index)` |

The last four rows are the **four leaf arms acked by the KISS editor 2026-07-23** ("RULING RECORD — four-leaf-arm ack" — [KISS #67 comment 5061571967](https://github.com/ThinkersJournal/KISS/issues/67#issuecomment-5061571967), acking Fuel's proposal [comment 5060303085](https://github.com/ThinkersJournal/KISS/issues/67#issuecomment-5060303085) of the same day; clean, no amendments) and **SHIPPED** in `to_canonical_bytes` with golden-byte tests. They ride **carrier (a)** — the #67 node-envelope `op_attrs` blob, `u32`-LE outer, verbatim (§6.19-0010) — never carrier (b) (§6.8-0007 region-table, `u16`-LE) or carrier (c) (§6.20-0005 shape-expr child, `u16`-LE). **Honest scope:** these are wire tokens only — Fuel's graph produces none of them yet (`op_to_tag` emits none; `tag_to_op` declines all four as honest misses), because a first-class recipe leaf needs the §A flat-DAG-CSE interior.

**Reconciled without widening the blob** (the closed node schema carries these): `reduce{monoid, reduce_axes, keepdim}` — `monoid` rides `op_name` (`sum`/`prod`/`max`/`min`), Fuel emits single-axis `{axis, keepdim}` (multi-axis `reduce_axes` deferred); `gather`/`scatter{axis, oob_policy, index_operand, index_dtype}` — Fuel emits `{axis}`, `oob_policy` a deferred slot, `scatter_combine` rides `op_name` (`scatter_add` = `atomic-add`; assign/atomic-max/atomic-min are consumer-gated op gaps), the index operand/dtype ride `child_edges`/that operand node. `mean` is **not a monoid** — it is `sum` fold + a `div`-by-`reduced_count` epilogue.

## §5 · The matmul contraction-attr = role vectors *(closed, mutual — reply-3)*

`matmul`'s `op_attrs` is **two per-axis role vectors** over the roles `{ Batch=0, FreeM=1, FreeN=2, ContractedK=3 }` (one **u8** per axis), `lhs_roles` then `rhs_roles`, each length = operand rank:

```
op_attrs(matmul) = u32_le(len lhs_roles) ++ lhs_roles ++ u32_le(len rhs_roles) ++ rhs_roles
```

- **Per-role width = u8** (LOCKED, mutual). The two vectors are wrapped by the Q4 outer `u32_le(body_len)` frame.
- **Canonical cell:** same-rank ≥ 2 operands, exactly one shared `ContractedK` (lhs-last dim == rhs-second-last), one `FreeM` (lhs-second-last), one `FreeN` (rhs-last), N ≥ 0 leading `Batch` dims. Role vectors encode **which axis plays which role**, not extents, so GQA (differing-but-divisible batch extents) serializes to identical all-`Batch` leading roles.
- **Serialize:** derive the vectors from operand ranks — a pure function of the node (identity-hash-stable).
- **Resolve:** check incoming vectors against the canonical cell → match → `matmul`; any other config (transposed/permuted/multi-`ContractedK`/`FreeN`-before-`K`) → a surfaced opaque-op gap, never a crash.
- **No `epilogue` attr:** a fused `matmul + bias + act` is one flat DAG — `relu(add(matmul(in0,in1), Bind(2)))`; the matmul node **is** the fold node; the epilogue is ordinary elementwise nodes over it.

**Worked rank-2 example** (`lhs=[FreeM,ContractedK]=[1,3]`, `rhs=[ContractedK,FreeN]=[3,2]`):

```
body = u32_le(2) ++ [01,03] ++ u32_le(2) ++ [03,02]                (12 bytes)
full = u32_le(12) ++ body
     = 0C 00 00 00 | 02 00 00 00 | 01 03 | 02 00 00 00 | 03 02      (16 bytes)
```

Byte codes agree 1:1 with Baracuda's `AxisRole` enum; Baracuda's shipped **text surface** (`matmul[m k.k n]`, roles as chars `b/m/n/k`) is a readable form over this binary canonical layer.

## §6 · The emitter + canonicalization contract

- **The emitter emits a valid-but-not-necessarily-canonical DAG; the importer canonicalizes on ingest** (lower → maximal-CSE → `base_map_hash`). Producers need not canonicalize; that authority sits with the importer.
- **Empty `op_attrs` = a zero-length length-prefixed blob** (`[00,00,00,00]`), not omitted — "no-elision" has exactly one canonical byte form.
- **Known ops emit only the token; the importer owns canonical resolution** (Q6). Re-canonicalizing (e.g. a `ReduceMaxTo → MaxDim` swap) is invisible to the emitter and free on the importer side (the hash is an on-demand pure function, nothing cached).

## §7 · Relationship to the shape-oracle (already in KISS)

The **shape-expression vocabulary** (`ShapeExpr := SameAs(operand)`; `DimExpr := Extent(operand,axis) | Const | Param | DimExpr(+ − × ÷floor)DimExpr`; axis = non-negative | `last`) is **already consolidated into KISS** as the shape-oracle (KISS-Ops §6.20 + KISS-Contract §6.4-0011, merged on KISS main @ `3bd6d2d`, Baracuda cosignatory). This design input covers the **structural** recipe grammar; the shape-oracle covers **output-shape derivation**. They meet under one abstraction — **output-shape = f(operand shapes, attrs)** — with two attr-vocabularies feeding one evaluator: the **matmul role-vectors** (§5, the contraction descriptor) and **`ShapeExpr`/`DimExpr`** (everything else). The shape/value boundary is normative: `ShapeExpr`/`DimExpr` carry **shapes only**; an extent needed as a **value** (the `Mean` divisor) is the `reduced_count` **leaf** inside the recipe DAG (§2), never a shape attr.

## §8 · Capability bit

`SEAM_CAP_RECIPE_IMPORT` = **FEAT bit 35** (32 = JIT_ON_REQUEST, 33 reserved CONTRACT_QUERY, 34 = KISC_FRAMING, 35 = RECIPE_IMPORT). Negotiated cutover, no flag day; pinning this grammar retires the fused-op / honest-miss contract withholds.

## §9 · Implementation status (proof-of-implementation)

| Element | Status in Fuel |
|---|---|
| `op_attrs` **sub-blob** serializer (Q4) | **SHIPPED** (`to_canonical_bytes`, `u32`-LE outer — see §4) |
| Node schema `Op \| Bind` (design) + full recipe-**node** / flat-table serializer | schema pinned; the node/table wire serializer (`op_name` + `child_edges` envelope around the `op_attrs` blob) **NOT built** — §A, being defined in KISS #67 |
| Canonicalization / `base_map_hash` rule (Q3) | **SHIPPED** |
| Resolve-to-base-map + numeric verify (Q6) | **SHIPPED** (recipe-identity verifier) |
| `Op::Scan{body,carry,bound}`, `Op::Reduce = Scan{emit=Final}` (Q5) | **SHIPPED** (Phase 1, G3 closed) |
| Source-op leaves (`const`/`iota`/`runtime_scalar`/`scan_placeholder`/`reduced_count`) | **byte arms SHIPPED** for the four acked leaves (`const`/`runtime_scalar`/`reduced_count`/`scan_placeholder`, 2026-07-23 ack — see §4); `iota` already rides `OpTag::Iota`. **Wire tokens only** — the graph-side wiring (first-class `PatternNode` leaves) needs the §A flat-DAG interior, still Increment C |
| matmul role-vectors (§5) | schema **closed/mutual**; the u8-role serialize+resolve lands in Increment C (emits empty `[00,00,00,00]` today) |
| flat-DAG-CSE representation + `decompose`→data migration (§A) | **NOT built** — the Convergence-C recipe-interior home, still pending |
| shape-oracle `ShapeExpr`/`DimExpr` evaluator + §6.20 wire codec (§7) | **SHIPPED** (Convergence-C @ `9156e178`, `fkc/shape_expr.rs`); KISS RFC merged @ `3bd6d2d` |

**Cross-references:** the conversational co-design record (`docs/outreach/baracuda-recipe-grammar-codesign-reply{,-2,-3}.md`), the Fuel implementation reference with file:line anchors (`docs/recipe-signature-reference.md` @ `ee76c977` — the verbatim byte-contract source (§6) + the shipped shape-oracle wire codec (§B)), and the merged shape-oracle RFC (KISS main @ `3bd6d2d`).

---

*Fuel co-owns this grammar and offers it for consolidation into KISS as the project-agnostic authority. The two ★ MUST-CARRY pins (Q4 canonical serialization, Q5 the general `Op::Scan` primitive) are the load-bearing invariants; carry them into the KISS section verbatim. Fuel's standing cross-project POC (`jvwnb5ut`) is Fuel's voice for the consolidation; route grammar questions there.*

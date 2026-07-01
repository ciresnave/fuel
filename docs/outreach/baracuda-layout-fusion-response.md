# To the Baracuda team — layout/shape wire facts for cross-boundary layout fusion (Fuel reply)

**To:** Baracuda · **From:** Fuel · **Re:** `fuel-ask-layout-shape-facts-2026-06-30.md` (item 01, layout/shape IR nodes)
**Status: OWNER-DECIDED 2026-07-01 — ready to send.** Fuel-side answers to K1/K2/F1/F2a/F2b/S1/F3. The two
owner-confirm items are RULED by the Fuel owner: **K2 — APPROVED** (proceed with `1 → 2`; keep the identity
view segment byte-identical so only the `sk1`/`sk2` prefix differs — a pure version check); **S1 — (a) on
layout-fused regions, (b) everywhere else** (Fuel commits to the net-new peel-the-permute seam projection).
**Nature:** design/contract response, grounded in Fuel's current seam code (evidence file:line inline).

Thanks for proposing this before locking — the propose-first cut is exactly right, and it lets us converge
the layout model instead of forking it. Below, every answer is grounded in Fuel's *current* seam code (not
assumption), with the evidence file:line inline so you can re-verify. Where Fuel does not yet have the
capability, we say so plainly.

---

## K1 — Fuel treats the `StructureKey` token as OPAQUE (does not parse internals)

**Answer: OPAQUE. Fuel never splits, parses, or inspects the token internals — it hashes/compares it whole
as a join/dispatch/telemetry key, and it never derives the key itself.**

Grounding:

- Fuel's only `StructureKey` type is `StructureKeyToken(pub String)`
  (`fuel-dispatch/src/telemetry/structure_key.rs:13`). Its own doc comment states the contract verbatim:
  *"Baracuda owns the structure-key encoding and ships the callable `structure_key(op_class, operands, arch)
  -> StructureKey`. Fuel **calls** it … and **never derives the key itself**. Here the token is treated as
  opaque bytes for the join"* (`structure_key.rs:1-13`). It derives `Hash`, `Eq`, `PartialEq` on the wrapped
  `String` and nothing else — there is no `from_token`, no field split, no `perm`/`bcast` accessor anywhere
  Fuel-side.
- This matches the pinned division recorded in the round-trip cover note: *"`StructureKey` computed by you
  and *called* by us via the minimal `FdxOperandDesc` projection … we never re-derive it"*
  (`docs/outreach/baracuda-seam-v1-roundtrip.md:51`).

**What this means for your layout fields:** the new per-operand `perm` + `view_kind` in the token are
**invisible to Fuel by construction** — Fuel round-trips the token as an opaque string, so a longer/richer v2
token flows through the telemetry join and the FKC-`accept`/link-registry lookup unchanged. Fuel does not need
to learn to parse them. The layout facts matter to Fuel **only** through the *typed* seam surface (`OpAttrs` /
`OperandDesc`), never through the token. That is why F1 (typed `OpAttrs` fields), not K1/K2, is the real
live-seam blocker.

**Not an OWNER-CONFIRM decision** — this is a statement of current fact, fully evidenced.

---

## K2 — `STRUCTURE_KEY_VERSION 1 → 2` + the identity-view back-compat rule: **APPROVED (owner-decided 2026-07-01)**

**Decision: YES — proceed with the `1 → 2` bump and the back-compat rule.** Per the caveat below, the Fuel
owner rules the cleanest form: keep the identity-view segment **byte-identical**, so a v1↔v2 comparison is a
**pure version-prefix check** (`sk1`→`sk2` only) — not a trailing-field-appended difference. Please assert that
form in your round-trip/injectivity tests.

Grounding for *why it's safe on Fuel's side*:

- Because Fuel treats the token as opaque (K1), the bump has **zero parse impact** on Fuel: a `sk2|…` token
  is just a different string than a `sk1|…` token. Fuel's telemetry/dispatch joins are equality/hash over the
  whole string (`structure_key.rs:12-13`), so two schema versions never silently collide and never mis-join.
- Your back-compat rule ("an all-identity-view cell encodes byte-identical to a v1 cell **modulo the version
  field**; a v1 token stays distinguishable by its version — no silent parse as a defaulted-identity v2") is
  the correct discipline and it is the one your own `structure_key.rs:47-48` doc already promises: *"Bumped
  when a predicate axis is added or altered; old-version tokens stay distinguishable by this field."* We
  endorse it.
- One **caveat we want on the record** (not an objection): the "byte-identical modulo version" claim is only
  true if the identity `view_kind`/`perm` fields are appended in a way that renders to **nothing** in the
  per-operand `<contig>/<bcasthex>/<vec>/<div>/<flip>` gloss for the identity case, OR the version prefix is
  the sole differentiator. If the identity view adds a literal `/…` field to every operand's token segment,
  then a v2-identity token is *not* byte-identical to a v1 token even modulo the leading `sk1`→`sk2` — it also
  gains a trailing field. Fuel does not care (opaque), but your own round-trip/injectivity tests
  (`01-layout-shape-ir-nodes.md` §7, "a version-1 token is still distinguishable") should assert which of the
  two you mean. Recommend: keep the identity segment byte-identical and let **only** the `sk1`/`sk2` prefix
  differ, so a v1↔v2 comparison is a pure version check — cleanest for both sides.

**Tradeoff:** none material on Fuel's side (opaque token). The whole cost is Baracuda-internal (codec +
round-trip tests). The only shared risk is the caveat above, which is a test-assertion clarity issue, not an
ABI risk.

**OWNER DECISION (2026-07-01): APPROVED.** The Fuel owner gives the formal go for the `1 → 2` ratified-annex
bump. Baracuda may release its held key-field commit. (Chosen form: identity segment byte-identical, only the
version prefix differs.)

---

## F1 — `OpAttrs` shape-fact fields Fuel wants (the live-seam blocker)

**Recommended field shape: extend `OpAttrs` additively with three typed, layout-purpose fields. This is
Fuel's call per the ask; Baracuda matches its emit encoding to it.**

Current state (the gap, exactly as you diagnosed):

- `OpTag` already lists the layout tags: `Transpose, Permute, Reshape, BroadcastTo, Unsqueeze, Squeeze`
  (`fuel-kernel-seam-types/src/lib.rs:52`).
- `OpAttrs` carries only `scalars: Vec<f64>` + `axis: Option<i64>` (`fuel-kernel-seam-types/src/lib.rs:70-77`)
  — no perm vector, no target shape, no broadcast target, no dim list. So a layout region node cannot express
  its transform across the seam today.
- The matcher **originally ignored `attrs` entirely**: `match_node`'s `Op` arm destructured
  `PatternNode::Op { op, operands, .. }` and dropped `attrs` — so even the existing `scalars`/`axis` were
  carried, not compared. **Landed 2026-07-01 (F1):** `match_node` now compares `OpAttrs` with a
  **wildcard-on-unset** rule — an empty/unset pattern field (empty `Vec` / `axis: None`) is a wildcard; a
  *set* pattern field must equal the graph node's projected value (via `fuel_graph::jit::op_to_attrs`, which
  reads the transform off the typed `Op` payload). This keeps every existing attr-agnostic pattern (all
  authored `OpAttrs::default()`) matching unchanged, and lets a layout/scalar pattern that *sets*
  `perm`/`target_shape`/`dims`/`scalars`/`axis` discriminate. So the three new fields are *carried across the
  seam* AND *matched on* as of this change (`fuel-graph/src/jit.rs`; `fkc-fusion-patterns.md` §3a rule 2b).

Recommended concrete shape (additive to `OpAttrs`, all `Default`-empty so every existing region is unchanged):

```rust
pub struct OpAttrs {
    pub scalars: Vec<f64>,        // unchanged
    pub axis: Option<i64>,        // unchanged
    /// Permute/Transpose: the new axis order, ABSOLUTE, a permutation of `0..rank`
    /// with `out.axis[i] = in.axis[perm[i]]` (matches Fuel `Op::Permute` semantics
    /// exactly — see F2a). Empty ⇒ not a permuting node.
    pub perm: Vec<u8>,
    /// BroadcastTo/Reshape: the target LOGICAL shape (the op's output shape).
    /// Empty ⇒ not a shape-target node. (See F2b for BroadcastTo vs. bcast mask.)
    pub target_shape: Vec<i64>,
    /// Squeeze/Unsqueeze: the affected dim list (0-based, in output-rank terms).
    pub dims: Vec<u8>,
}
```

Rationale, grounded in how Fuel actually represents these ops (so your emit encoding matches Fuel's ingest by
construction):

- **Permute/Transpose → `perm: Vec<u8>` (absolute).** Fuel's `Op::Permute(Vec<usize>)` is *defined* as the
  absolute new axis order — `out.shape[i] = in.shape[axes[i]]`, "the axes vector must be a permutation of
  `0..rank`" (`fuel-graph/src/lib.rs:540-544`). `Op::Transpose` is the rank-2 / last-two-axes special case
  (`lib.rs:538-539`; the rank-3 test swaps the last two dims, `lib.rs:10056`). So a `Vec<u8>` absolute perm is
  the exact mirror of what the graph already carries, and `Transpose` maps to the absolute perm that swaps the
  last two axes. We chose `Vec<u8>` over your suggested `Vec<u8>` verbatim (agreement) — `MAX_RANK=8` fits a
  `u8` per axis with room to spare.
- **BroadcastTo → `target_shape: Vec<i64>` (the target logical shape).** Fuel's `Op::BroadcastTo(Shape)`
  carries the full target shape (`lib.rs:555`, `lib.rs:4966`), and the backward/lowering path is
  shape-driven (`ReduceSumTo(Shape)` is its symmetric inverse, `lib.rs:591-597`). Delivering the target shape
  (rather than only an axis mask) keeps the region node self-describing and lets the matcher reconstruct the
  broadcast axis set from `(input_shape, target_shape)` — which is also how the operand-side broadcast mask is
  derived (stride-0 axes; see F2b). We recommend target shape as the *authoritative* field; an axis mask is a
  lossy projection of it.
- **Reshape → `target_shape: Vec<i64>` (target logical shape), reusing the same field.** Fuel's
  `Op::Reshape(Shape)` carries the target shape (`lib.rs:561`). We deliberately reuse `target_shape` for both
  `Reshape` and `BroadcastTo` — the `OpTag` already disambiguates which op it is, so one field serves both and
  keeps `OpAttrs` minimal. (Your §5.3 note "do NOT add the full producer shape to the *key*" still holds — the
  *key* stays extent-free; this shape lives on the typed `OpAttrs` region node, not in the `StructureKey`
  token. Different surfaces, no contradiction.)
- **Squeeze/Unsqueeze → `dims: Vec<u8>`.** Fuel's `Op::Unsqueeze { dim }` / `Op::Squeeze { dim }` are
  single-dim today (`lib.rs:584`, `lib.rs:590`), but a `Vec<u8>` future-proofs the multi-dim case and matches
  PyTorch `squeeze(dim=[…])`. For the single-dim ops Fuel emits a one-element list.

Re-read / `extract:` semantics (mirroring how scalar params work): like `scalars`, these fields are the
region's **snapshot** of the transform, re-read live from the matched graph node at match time via the
`extract:` path — they are the slot identity, not a baked constant. That mirrors the existing `OpAttrs` doc
(`fuel-kernel-seam-types/src/lib.rs:66-69`: *"the value is not baked — it identifies the slot … the matcher
re-reads the live value from the matched graph node at match time"*). Note that for a *pure permutation of a
contiguous producer* the perm is a compile-time structural fact, so unlike a scalar it does not vary at
runtime — but keeping the re-read discipline uniform costs nothing and avoids a special case.

**Not an OWNER-CONFIRM decision** — this is Fuel's own type surface, and the ask explicitly delegates the
field shape to Fuel ("we will match our emit-side encoding to whatever you land"). Owner review still welcome,
but no cross-contract lock is at stake. Note: landing these fields is an additive change to the frozen
seam-types crate — additive-only, `Default`-empty, no existing field moved — consistent with the additive
versioning discipline (`dlpack-extension.md` P8).

---

## F2a — `perm` is ABSOLUTE (a permutation of `0..rank`), not relative-to-input-rank

**Recommended answer: ABSOLUTE. Express `perm=[1,0]` (a permutation of `0..rank`), never relative.**

Grounding:

- Fuel's `Op::Permute` is absolute: "the new axis order: `out.shape[i] = in.shape[axes[i]]` … The axes vector
  must be a permutation of `0..rank`" (`fuel-graph/src/lib.rs:540-544`). Fuel's autodiff inverts it as an
  absolute permutation (`Op::Permute(inv)` where `inv[axes[i]] = i`, `lib.rs:6797-6815`) — there is no
  relative/rank-delta representation anywhere in the graph.
- Fuel's `Layout::permute` implements the same absolute convention: `perm_stride[i] = stride[idx]` for
  `idx = idxs[i]` (`fuel-core-types/src/layout.rs:205-228`). So both the graph `Op` and the runtime layout
  speak the same absolute perm.

On canonicalization ("both sides canonicalize before matching," §3a.2a): the §3a.2a canonical order is about
**commutative-operand ordering** (`Add`/`Mul`/`Maximum`/`Minimum` operands sorted by a stable structural key —
`fkc-fusion-patterns.md:248-257`; implemented as `is_commutative` + try-both-orderings in
`fuel-graph/src/jit.rs:111-113,183-196`). A permutation vector is **not** an operand-ordering question — it is
an op *attribute*. There is therefore no operand-reordering canonicalization that could desync a perm: an
absolute perm on the pattern node must equal the absolute perm on the graph node, byte-for-byte, for the match
to fire. Because both repos will read/write the *same* absolute convention (`out.axis[i] = in.axis[perm[i]]`),
they canonicalize identically by construction — there is nothing to normalize.

**One flag for convergence (verify on your side):** confirm your `View::Permute { perm }` uses the **same
direction** as Fuel's `Op::Permute` — i.e. iteration/output axis `d` reads producer/input axis `perm[d]`
(`out[d] = in[perm[d]]`). Your design doc §5.1 says *"iteration axis `d` indexes producer axis `perm[d]`"*
(`01-layout-shape-ir-nodes.md:271-272`), which matches Fuel exactly. Your own §8 adversarial checklist calls
out the `perm` vs `perm⁻¹` inversion bug as the top failure mode (`:480-484`) — good; the shared, explicit
`out[d]=in[perm[d]]` statement is the anti-inversion anchor for both repos. `Transpose` (rank-2) is `[1,0]`.

**Not an OWNER-CONFIRM decision** — dictated by Fuel's existing absolute `Op::Permute` semantics; there is no
degree of freedom to hand the owner. (It is a *convergence* item: both sides must state the same direction,
which they already do.)

---

## F2b — `BroadcastTo` target vs. the operand's existing broadcast mask: they are DIFFERENT surfaces; both may be present; the OPERAND STRIDE-0 MASK is authoritative for keying

**Recommended answer: `BroadcastTo` (the region-node attribute) and the `<bcasthex>` broadcast mask (the
per-operand `StructureKey` fact) are on two different surfaces and describe two different things. Both can be
present. For *keying/classification* the operand stride-0 mask wins (it is derived, honest, and already in the
token); the `BroadcastTo` region attribute is the *recognition* fact (what the fused op re-emits). They must
be consistent, not compete.**

Grounding — the two surfaces:

1. **Operand-side broadcast (the `<bcasthex>` mask).** In the `StructureKey`, per-operand `bcast: AxisMask` is
   *derived from the strides*: "extent-> 1 axes with stride 0" — `derive_operand_key` sets `bcast.set(d)` iff
   `shape[d] > 1 && strides[d] == 0` (`baracuda-kernels-types/src/structure_key.rs:463-469`), and that drives
   `Contiguity::Broadcast` (`:492-495`). This is a *property of the live operand's layout* — a stride-0 axis —
   and it is exactly the fact FDX carries via a stride-0 axis on `DLTensor.strides` (§4.1:
   *"broadcast — via a stride-0 axis on `DLTensor.strides`"*, `dlpack-extension.md:670`).
2. **Region-node `BroadcastTo` (the `OpAttrs.target_shape` you asked for in F1).** This is the *op* in the
   fused subgraph — "expand this producer to this target shape" — a *recognition* fact for the pattern
   matcher, mirroring Fuel's `Op::BroadcastTo(Shape)` (`fuel-graph/src/lib.rs:555`).

Why both can be present and there is no conflict:

- A `BroadcastTo` **node** in a region, when applied, *produces* an operand whose layout has stride-0 axes —
  so downstream the operand mask and the node attribute describe the *same* broadcast, one as a transform and
  one as its stride footprint. They are consistent by construction (the node is *why* the mask is what it is).
- **Which wins if they ever disagree:** the operand stride-0 mask is the authority for *keying and codegen*,
  because it is derived from the real strides Fuel delivers (`derive_operand_key`, `structure_key.rs:463-469`)
  and the emitter's broadcast-axis drop / fully-broadcast hoist keys off it (`is_fully_broadcast`, per your
  `01-layout-shape-ir-nodes.md:174-176`). The `BroadcastTo` region attribute is the authority for
  *recognition* (matching the subgraph and re-emitting the `decompose`). A well-formed region has both agree;
  a disagreement is a producer bug, and the operand mask (being the ground truth of the bytes) should be
  treated as correct while the region is declined/flagged, never silently reconciled.

**Recommendation to keep them from ever competing:** treat `BroadcastTo` as the **sole *recognition*** source
(it names the transform in the `pattern:`) and the stride-0 mask as the **sole *keying/codegen*** source (it
describes the resulting operand). Do not encode the broadcast twice into the *same* surface — i.e. don't also
add a redundant broadcast-axis field to `OpAttrs` when `target_shape` + the input shape already imply the axis
set, and don't let the `view_kind=Broadcast` in the v2 token contradict the derived `<bcasthex>` (they must be
the same broadcast). Your §5.2 already does exactly this (*"`Broadcast` is the named IR form of what the mask
already does,"* `01-layout-shape-ir-nodes.md:317-318`) — this answer just pins it normatively across the seam.

**Not an OWNER-CONFIRM decision** — it is a consistency rule between two existing surfaces, grounded in the
derived-from-strides mask. Flag: this is the one answer where Fuel is *recommending a discipline* rather than
reporting a single as-built fact, because Fuel does not yet key on broadcast in its own matcher (it ignores
`attrs`, `jit.rs:169`) — so the "operand mask is authoritative" rule is a forward commitment Fuel will honor
when it wires attr-matching, not something Fuel's matcher enforces today.

---

## S1 — Stride convention at the operand boundary: ~~(a) on layout-fused regions, (b) everywhere else~~ **SUPERSEDED 2026-07-01 → convention (c)**

> **SUPERSEDED 2026-07-01.** Baracuda's follow-up steelman found convention **(c)** — keep Fuel's existing
> `Layout::permute` (b) and route the transpose via `OpAttrs.perm` — functionally equivalent to (a) for the
> elementwise scope with **no** peel-the-permute projection, no `StructureKey` perm field, no version bump.
> Fuel adopted (c); **peel-the-permute is stood down** (never built). See
> `baracuda-stride-convention-c-confirm.md`. The (a) analysis below is retained for the record.

This is the load-bearing ABI decision. Fuel's *current* behavior is (b); Baracuda prefers (a) for fusion.
**The Fuel owner rules the hybrid: adopt (a) on layout-fused regions, keep (b) for every existing op.** Fuel
commits to build the net-new peel-the-permute seam projection this requires (see below).

**What Fuel does TODAY — grounded, unambiguous: option (b).** For a permuted operand, Fuel's `Layout::permute`
**pre-permutes the stride array into iteration/output-axis order**: `perm_stride[i] = stride[idx]` where
`idx = idxs[i]` (`fuel-core-types/src/layout.rs:205-228`); `transpose` likewise swaps the stride entries
(`layout.rs:193-201`). So the strides Fuel would project into an `FdxOperandDesc`/`OperandDesc` for a permuted
operand are **already in iteration-axis order** — the transpose is baked into the stride *values*, exactly the
"caller pre-permutes; the kernel is generic" model you describe as (b) in the ask (§4). This is corroborated
by the FDX spec: strides are "keyed to capacity" in the operand's own axis order (`dlpack-extension.md:717`),
FDX describes a transpose purely as the resulting strides (§4.1: contiguity/broadcast/flip are read *from the
strides* — `:665-674`), and your own design doc confirms a Fuel transpose "shows up only as
`Contiguity::Strided`" with no named perm (`01-layout-shape-ir-nodes.md:156-160`). Baracuda's current strided
emit consumes exactly these iteration-order strides (`offset = Σ_d c[d]·s{k}[d]`, `01-…:76-77`).

**What Baracuda prefers — (a):** producer-axis-order strides, with the kernel applying the perm
(`offset = Σ_d c[d]·s{k}[perm[d]]`), so a *contiguous producer* can be read transposed with no materialized
copy — the read-through-a-view fusion win.

**Fuel's recommended answer: adopt (a) for the layout-fusion seam path, as a NEW capability, without changing
the default (b) meaning of `OperandDesc.strides`.** Concretely:

- Keep `OperandDesc.strides` meaning **unchanged and (b)** for every existing op — that is the ratified,
  as-built convention and what the whole `structure_key` derivation already reads
  (`derive_operand_key`/`classify_contiguity` all assume the operand's own axis order,
  `structure_key.rs:460-520`). Do **not** silently redefine it, or every existing cell's keying breaks.
- For a **layout-fused region** specifically, deliver (a): Fuel hands the **producer's own
  (producer-axis-order) strides** in `OperandDesc.strides` **plus** the absolute `perm` (via the F1
  `OpAttrs.perm` on the region node and the v2 token's per-operand `perm: PermCode`), and the kernel applies
  `s{k}[perm[d]]`. The `perm` being present in the region/key is precisely the signal that the strides are
  producer-order-with-perm rather than pre-permuted-iteration-order. This is only reachable when the producer
  is contiguous (a permuted *view of a strided* producer is a genuine gather — out of scope, decline, per your
  §5.1 Reshape note and §8).

Why this is the right split (the tradeoff):

- **(a) is what unlocks the fusion** — it is the entire point of the item; delivering (b) pre-permuted strides
  makes the transpose invisible to the kernel and there is nothing to fuse *through*. Baracuda states (a) is
  its preference and the AOT path it controls end-to-end already validates with (a) (`fuel-ask…:85-87`).
- **But (a) is a change Fuel must make** — Fuel does not naturally produce producer-order strides for a
  permuted operand today; `Layout::permute` pre-permutes (`layout.rs:219-221`). To deliver (a), the Fuel-side
  seam projection must, for a layout-fused region, **peel the permute** — project the *producer's* contiguous
  strides + the perm, rather than the permuted view's strides. That is net-new Fuel plumbing on the
  base-emission seam (which the cover note flags as still-being-built, `baracuda-seam-v1-roundtrip.md:105`),
  not a behavior that exists now.
- **Scoping it to the layout-fused path** (gated by `perm`/`view_kind` present) means existing dense/strided
  ops keep the (b) convention with zero disruption and zero re-keying, while the new fusion path gets (a). No
  flag-day, no reinterpretation of already-shipped operand descriptions.

**Uncertainty / what Fuel could NOT verify:** the seam-path projection that would build `OperandDesc` from
`FdxOperandDesc` for a *layout-fused* region is **not implemented yet** on either the FDX side or the dispatch
side — FDX's §4.1 feed is marked "[consumer-ahead: deferred Baracuda telemetry feed]"
(`dlpack-extension.md:684-688`) and the JIT base-emission seam is still under construction
(`baracuda-seam-v1-roundtrip.md:105`). So (a) is a **forward ABI commitment**, not a description of running
code; Fuel is choosing the target convention now so both sides build to it. If the owner instead prefers to
keep (b) universally (kernel stays generic, no read-through-view fusion, the transpose is always a real
copy Fuel inserts), that is coherent too — it just forgoes the headline win. Fuel recommends (a) for the
fused path.

**OWNER DECISION (2026-07-01): (a) on the layout-fused path, (b) everywhere else.** The Fuel owner rules the
split: `OperandDesc.strides` keeps its (b) iteration-order meaning for all existing ops (no re-keying), and a
layout-fused region carries producer-axis-order strides + the absolute `perm` (kernel applies `s{k}[perm[d]]`).
Fuel commits to the peel-the-permute seam projection this requires (net-new; tracked as Fuel-side work). Both
sides build the layout-fused emit to convention (a).

---

## F3 — Convergence confirm: YES, the `View` vocabulary + F1 field shape is the agreed realization of the item-3 "layout nodes with shape facts" line. No fork.

**Answer: CONFIRMED — converge, do not fork.**

Grounding:

- Fuel's cited reply does list the item-3 workstream verbatim: *"layout nodes (`Reshape`/`BroadcastTo`/
  `Transpose`) with shape facts"* as part of "the dedicated norm/linear workstream"
  (`fuel-reply-fkc-patterns-2026-06-19.md:175-178`). Your ask quotes the same line (`fuel-ask…:90-93`).
- The vocabularies line up one-to-one:
  - Baracuda `View::Permute { perm }` ↔ Fuel `OpTag::Permute` + `Op::Permute(Vec<usize>)` (absolute perm,
    F2a) ↔ F1 `OpAttrs.perm`.
  - Baracuda `View::Broadcast { bcast }` ↔ Fuel `OpTag::BroadcastTo` + `Op::BroadcastTo(Shape)` ↔ F1
    `OpAttrs.target_shape` (with the F2b operand-mask relationship pinned).
  - Baracuda `View::Reshape { producer_rank }` ↔ Fuel `OpTag::Reshape` + `Op::Reshape(Shape)` ↔ F1
    `OpAttrs.target_shape`. (Note the one small representation difference below.)
  - Baracuda `View::Identity` ↔ Fuel: no region node at all (an identity view is a bare `Bind`, not a layout
    `Op` — consistent with your §5.4 "a non-`Identity` view … is no longer a bare `Bind(i)`",
    `01-layout-shape-ir-nodes.md:368-372`).
  - `Transpose` (rank-2) is the special-case absolute perm on both sides (Fuel `Op::Transpose`,
    `lib.rs:538-539`; your `View::Permute` rank-2 case).
  - `Squeeze`/`Unsqueeze` are in Fuel's `OpTag` (`lib.rs:52`) and get `OpAttrs.dims` (F1) — your ask lists
    them for completeness; they're covered.

**Two convergence notes (small, worth pinning so we don't drift):**

1. **Reshape representation differs slightly and that's fine.** Baracuda's `View::Reshape { producer_rank }`
   carries only the producer rank (because for a contiguous producer a reshape is a linear-index pass-through
   — no address math, `01-…:298-303`), whereas Fuel's `Op::Reshape(Shape)` and F1's `OpAttrs.target_shape`
   carry the full target shape. These are compatible: the target shape *contains* the target rank, and the
   producer rank is available from the bound input's shape. Recommend Baracuda derives `producer_rank` from
   the operand and Fuel supplies `target_shape`; neither needs the other's exact field. If you'd rather the
   region node carry `producer_rank` explicitly too, say so and we'll add it — but `target_shape` +
   input-shape already determines it, so we left it out to keep `OpAttrs` minimal.
2. **The `StructureKey` `view_kind`/`perm` (v2) vs. the `OpAttrs` fields are different surfaces and must not be
   conflated.** The token fields are opaque-to-Fuel keying facts (K1); the `OpAttrs` fields are the typed
   recognition/re-emit facts. They describe the same transforms but Fuel only *acts on* the `OpAttrs` ones.
   Your design keeps them mirrored (`01-…:349-351`), which is exactly right — just don't assume Fuel reads the
   token's `perm` (it doesn't; it reads `OpAttrs.perm`).

**Fork risk assessment: LOW.** The models are the same shape (per-operand layout view; absolute perm;
broadcast-as-target-shape; reshape-as-rank-change; identity ⇒ no node). The only genuine open decision that
could cause divergence is **S1** (stride axis order) — if Fuel shipped (b)-universally and Baracuda emitted
assuming (a), a transpose-fused pattern would match structurally but the kernel would read the wrong strides.
That is why S1 is the OWNER-CONFIRM item and why we recommend pinning (a)-on-the-fused-path now, before either
side hardens its emit. Everything else converges cleanly.

**Not an OWNER-CONFIRM decision** — it's a confirmation, contingent only on the S1 ruling landing consistently.

---

## Summary of what Fuel is asking / committing

- **K1:** opaque — no action, no risk.
- **K2:** **APPROVED (owner, 2026-07-01)** — proceed with `→ 2` + the back-compat rule. Chosen form: identity
  segment byte-identical, only the `sk1`/`sk2` version prefix differs (a pure version check). Baracuda may
  release its held key-field commit.
- **F1:** Fuel will land additive `OpAttrs` fields `perm: Vec<u8>` (absolute), `target_shape: Vec<i64>`
  (BroadcastTo + Reshape), `dims: Vec<u8>` (Squeeze/Unsqueeze). Baracuda matches its emit encoding. (Also
  requires Fuel to extend `match_node` to actually compare `attrs` — currently ignored, `jit.rs:169`.)
- **F2a:** absolute perm; no owner call needed (dictated by `Op::Permute`).
- **F2b:** two surfaces, both may be present; operand stride-0 mask authoritative for keying, `BroadcastTo`
  authoritative for recognition; a forward discipline commitment.
- **S1:** **DECIDED (owner, 2026-07-01): (a) on the layout-fused path, (b) everywhere else.** Fuel today
  delivers (b) (`Layout::permute`, `layout.rs:219-221`); the fused path adopts (a) (producer-axis-order strides
  + absolute `perm`, kernel applies `s[perm[d]]`), with zero re-keying of existing ops. Fuel commits to the
  net-new peel-the-permute seam projection (Fuel-side work, not yet built).
- **F3:** confirmed convergence; low fork risk; S1 is now ruled, so the divergence hazard is closed.

— Fuel (owner-decided 2026-07-01; ready to send)

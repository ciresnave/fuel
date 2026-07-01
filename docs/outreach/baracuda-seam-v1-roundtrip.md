# To the Baracuda team — Kernel-Seam Interop Contract (Profile v1), for ratification

**Status: DRAFT — not sent.** Cover note for circulating `docs/specs/kernel-seam-interop.md` to Baracuda.
The owner sends; this is the Fuel-side draft.

---

Thanks again for the 2026-06-19 round — everything you locked there stands, and it's the backbone of what
follows. Since that exchange we've consolidated the seam into a single ratifiable document and added one
new layer. We'd like your feedback toward **ratifying Profile v1** so all three projects (Fuel, Baracuda,
Vulkane) implement the *same* version — now, not eventually.

The full contract is **`docs/specs/kernel-seam-interop.md`**. **This is now a self-contained bundle** — your
review correctly refused to ratify a profile pinning a fusion-patterns version (rev 4) you hadn't seen, so
the circulation manifest (contract §8) now travels with *every annex version Profile v1 pins*: FDX v1, FKC
v1, **fusion-patterns rev 4**, and a focused **rev-2 → rev-4 delta** (`baracuda-fusion-patterns-rev4-delta.md`)
so you can confirm your findings without diffing.

## Round 2 — your conditions, resolved

- **Blocking condition (rev-4 fusion-patterns: A1, A2, E1) — all resolved.** Verified against the current
  spec: **A1** (`self.axis == input(0).rank - 1` arithmetic) → `self.axis == -1` (axes normalize
  negative-from-end; §5/§8.2); **A2** (`operand(0)` on a `bind` leaf) → `dim[0] == input(1).dim[-1]` (§8.1);
  **E1 — commutative-operand canonicalization is now NORMATIVE** (§3a.2a: Fuel canonicalizes commutative
  operands by the same stable key `structure_key` uses, so your `derive_pattern` emits **one** ordering and
  Fuel matches it either-way — no 2ᵏ blow-up). All three were fixed in **rev 3** and carried unchanged into
  rev 4; rev 4 changed *nothing* in the grammar your generator targets (it only reconciled prose to the
  adaptive-fusion decision). The delta doc walks each item to its spec anchor — so "re-verify your generator
  against rev 4" should be fast and pass.
- **Pin-before-freeze (the `SeamHello` C realization) — pinned.** You're right that a frozen-forever
  envelope with a variable-length member is the worst thing to get wrong. `SeamHello.profiles` is now a
  **fixed-max array** (`SEAM_MAX_PROFILES = 16`) with a `profiles_len` count — a fixed-size POD with frozen
  offsets — and the calling convention is an **out-param**: `int baracuda_seam_hello(SeamHello* out)`
  (no by-value return across the ABI). Full C struct + the `offset_of!`/size-assert discipline are in
  contract §3.1.
- **§5 JIT — your e-graph alignment is on the record (contract §5.1).** Agreed and stated normatively: "no
  backend-side opportunity-finding" bounds *region selection*, **not** optimization *within* the region.
  Your e-graph synthesizer is "the synthesizer within Fuel's chosen region" — fully compatible; it's
  pointed only inward at a Fuel-chosen subgraph, never scanning Fuel's graph to pick regions.
- **§7.1 correction applied.** We'd understated you. The contract now records that `structure_key` **and**
  `baracuda-kernelgen` (IR, three schedules, `f32/f16/bf16/f64`, `derive_pattern` emitting
  `AddScalar`/`MulScalar` with `extract:`) are **built and GPU-validated on PR #2**, and that what remains
  for a conforming publish is the full FKC emitter (`accept`/`return`/`cost`/`precision`; `pattern:` already
  emitted), the `link_registry`, `baracuda_seam_hello()`, and packaging the `structure_key` callable.

## What's unchanged (your 2026-06-19 acceptances, now pinned into Profile v1)

The honesty invariant; FDX + FKC core; negative-stride capability-gating; gather + affine extents; the
**`ImplId` 5-field basis tuple** (separable wire fields, ready to freeze); **`StructureKey`** computed by
you and *called* by us via the minimal `FdxOperandDesc` projection (strides, dtype, alignment, quant,
symbolic extent) — we never re-derive it; Judge per-cell retention incl. losers (`candidates[]` feasible);
miss = "best admissible match is a generic contract"; the `I4/U4/B1` dtype codes; `F32Strict` as a precision
*mode*; FKC as a *generated* projection of your `KernelSku`/OP-matrix. All of that is Profile v1's §2 bundle
and §4 conformance matrix.

## What's new (three things)

1. **A connect-time version handshake (contract §3).** This is the one structural addition we want in from
   the start, exactly so we *don't* have to do a lockstep flag-day every time the seam evolves. Each side
   advertises a tiny, frozen-forever envelope (`SeamHello`: magic + supported profile integers + a
   capability bitset); the connection selects the **highest mutually-supported profile** and **hard-fails on
   mismatch** (never proceeds on an assumed version). On your side it's a small C-ABI entry point —
   `baracuda_seam_hello() -> SeamHello` — plus a `seam_profiles: [1]` line in each FKC bundle's front-matter.
   At v1 this is trivially "both say `[1]` → 1"; the point is the mechanism exists so Profile v2 negotiates
   gracefully. Your existing FDX `BackendProbe` tokens (`DlpackExtMx/Ggml/Affine/Symbolic/Gather`) become the
   low bits of the capability set — no new concept, just lifted into the handshake.

2. **A JIT-on-request layer (contract §5) — and it's capability-gated, so you lose nothing by not building
   it yet.** This is the part you haven't seen. Fuel detects a fusion it lacks a kernel for, and — *on Fuel's
   schedule, in idle time, resource-aware* — sends you a **partial base map + a budget**; you synthesize the
   best kernel for **that Fuel-chosen region** and return `(kernel + a full FKC contract + the declarative
   recipe)`; Fuel **cost-gates adoption** (it competes as a multi-sibling alternative and is used only if it
   wins). The division is deliberate and keeps our constitution intact: **Fuel chooses the region (that *is*
   the fusion decision) and decides whether to adopt; you synthesize within it.** No backend-side
   opportunity-finding — this is explicitly *not* backend-internal fusion. Because it's gated by a
   `SeamCapJitOnRequest` capability bit, a Profile-v1 Baracuda that ships FDX+FKC but not (yet) the JIT
   endpoint is **fully conformant**; the bit lights up when both sides implement it. The request feed is the
   missing-fusion telemetry we owe you (closed-world `FusionMissRecord{NoBackendKernel}` first; open-world
   co-occurrence deferred) — none of it is built yet, so this is a design review, not a "ship it" ask.

3. **Tier-2 declarative registration (the mechanism behind the JIT layer).** Your generated FKC contracts
   already carry the recipe; what's new is that Fuel will register a *new fused-op identity* at runtime from
   a declarative `pattern:` + `decompose` (append-only, stable never-reused ids). This is Fuel-internal
   plumbing — it doesn't change what you author — but it's why the JIT loop can adopt a brand-new fusion you
   synthesize. (Implementing our declarative-pattern engine, currently a stub, is the prerequisite; that's
   on us.)

## What we're asking (the ball is back with you)

Both of your conditions are met — the rev-4 fusion-patterns + delta are in this bundle (A1/A2/E1 resolved),
and the `SeamHello` C realization is pinned (§3.1). So:

- **Re-verify `derive_pattern` against rev 4** (fast — it's the rev-3 grammar) and confirm A1/A2/E1 read as
  resolved in the delta.
- **Confirm the pinned `SeamHello` C ABI** (§3.1: fixed-max `profiles[16]` + `int baracuda_seam_hello(SeamHello* out)`)
  works for your realization.
- **Confirm the Profile v1 bundle** (§2) and which capability bits you'd advertise at first (we expect FDX+FKC
  now, `SeamCapJitOnRequest` later).

Then we **ratify Profile v1**, stamp it, and you publish a conforming Baracuda version (the full FKC emitter,
`link_registry`, `baracuda_seam_hello()`, the packaged `structure_key`) — Fuel bumps the crates.io pin, same
as the vulkane 0.8.2 BDA bump. Vulkane has already confirmed conformance, so once you re-verify we're clear
to ratify all three together.

No rush on the JIT layer's *implementation* — we're still building the Fuel-side base-emission seam it rides
on. What we want *now* is agreement on the **shape** so we all build to the same Profile v1.

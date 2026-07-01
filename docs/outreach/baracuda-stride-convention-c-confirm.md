# To the Baracuda team — Fuel confirms convention (c); peel-the-permute stood down (Fuel reply)

**To:** Baracuda · **From:** Fuel · **Re:** your `stride convention: adopt convention (c)` (2026-07-01),
revising S1 of `baracuda-layout-fusion-response.md`.
**Status: OWNER-CONFIRMED 2026-07-01.** Fuel adopts (c). This supersedes the S1 = (a) ruling in the prior
reply.

## Confirmed

**(c) accepted — and we agree it is strictly leaner for Fuel.** Your steelman is correct, and the disproof
is in our own code, as you noted:

- **Strides — no change on Fuel's side.** Fuel keeps its existing `Layout::permute` (b): a transposed operand
  is delivered pre-permuted into iteration-axis order (`perm_stride[i] = stride[idxs[i]]`,
  `fuel-core-types/src/layout.rs:205‑228`). Under (c) that classifies honestly as `Strided` and your existing
  generic strided cell reads it correctly. **We stand down the net-new peel-the-permute projection** — it was
  never built (it was a queued commitment, not code), so there is nothing to revert.
- **Your decisive argument holds.** Under (a) we'd hand the producer's natural strides (contiguous `[2,3,4]`
  → `[12,4,1]`), whose inner-stride-1 mis-keys as `Contig` on your side → a contiguous/vectorized cell reading
  untransposed. The perm-key + perm-kernel + version bump existed only to repair that mis-key. (c) avoids
  creating it. We verified the memory-access identity for `perm=[2,0,1]`: both conventions compute
  `offset = c0·1 + c1·12 + c2·4` — byte-identical.

## Reconciliation with the prior reply

| Ref | Prior | Now under (c) |
| --- | --- | --- |
| K1 (opaque) | opaque | ✅ unchanged — reinforces (c) (token perm would be dead weight; our matcher acts on `OpAttrs.perm`) |
| K2 (version bump) | approved | **withdrawn (yours) — no Fuel action.** `STRUCTURE_KEY_VERSION` is your surface; Fuel never changed it, so nothing to revert. Stays v1. |
| F1 (`OpAttrs.perm/target_shape/dims` + `match_node` attr-compare) | landing | ✅ **LANDED** (`90a7d331`, feat/kernel-contracts-dlpack) — and it IS (c)'s recognition carrier. See below. |
| F2a (absolute perm) | absolute | ✅ unchanged |
| F2b (two surfaces) | mask keys / BroadcastTo recognizes | ✅ unchanged |
| **S1** | (a) + peel-the-permute | **revised to (c): keep (b); peel-the-permute stood down.** |
| F3 (converged) | confirmed | ✅ unchanged |

## What Fuel has already done / will do under (c)

- **DONE — F1 (`90a7d331`):** `OpAttrs` gained `perm: Vec<u8>` (absolute, F2a) / `target_shape: Vec<i64>` /
  `dims: Vec<u8>`; `match_node` now compares `OpAttrs` with a **wildcard-on-unset** rule (empty pattern field
  = wildcard; a set field must equal the graph node's projected value, via `op_to_attrs`). This is exactly the
  recognition carrier (c) needs — a `Permute`-bearing subgraph can carry its absolute perm in `OpAttrs.perm`
  and the matcher can route on it. No regression (all existing empty-attr patterns stay wildcards;
  fuel-graph --lib 275/0).
- **Remaining, and correctly consumer-ahead:** wiring the actual `Permute→elementwise` **routing** to your
  generic strided cell (whether via a see-through `Permute` per §4.3 or a perm-value compare — both work with
  (b), your call to mirror). F1 is the prerequisite; the routing lands with the live-seam adoption, which is
  still consumer-ahead on both sides. No urgency, no new stride/kernel/key work.
- **NOT building:** peel-the-permute; any `OperandDesc.strides` reinterpretation; any token/version change.

## Scope caveat — agreed

We keep the `View`/`perm` representation (F1's `OpAttrs.perm`) as the recognition carrier and do **not** wire
it into any elementwise stride/emit/key change. If a future perm-aware *specialized* schedule (a tiled
coalesced-transpose kernel, or a perm interacting with a reduction axis — items 03/10) makes a compile-time
perm earn its keep, we re-open the stride convention for *that* schedule specifically. For the item-01
elementwise scope, (c) is the convention.

## Net

Fuel does **less** under (c) and has nothing to undo: keep `Layout::permute` (b), keep F1 (landed), route
`Permute` subgraphs to your generic strided cell when the seam goes live. Converged, not forked.

— Fuel (owner-confirmed 2026-07-01)

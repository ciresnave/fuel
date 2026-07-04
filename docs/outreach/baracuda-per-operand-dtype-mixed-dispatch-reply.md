# Baracuda reply — per-operand dtype for mixed-dtype op keying/dispatch (2026-07-04)

**Re:** Baracuda kernelgen's "per-operand dtype for mixed-dtype op keying/dispatch"
ask (IR-expansion ramp, increments 0a…#4 GATHER, commit `41c3010`).
**Status:** all four questions answered; **Model A**, no `STRUCTURE_KEY_VERSION`
bump on Fuel's side — but with a premise correction that makes the answer stronger
than the ask assumed. Grounded in the current `feat/kernel-contracts-dlpack`
source (FKC importer + binding table + graph op contracts + CPU byte kernels).

---

## TL;DR

1. **Model A — and there is no coarse token to be subordinate to.** Fuel's dispatch
   key **is** the full per-operand dtype tuple `(OpKind, [in dtypes…, out dtypes…],
   BackendId)`. The `accept.inputs[i].dtype` list is *literally* what the importer
   assembles into that key, and the runtime lookup is built from the node's actual
   per-operand dtypes. Per-input dtype is fully load-bearing. **No wire change, no
   version bump.** Fill the accept block honestly and Fuel keys on it. Wrong-bind is
   structurally impossible — an `i32`-index and an `i64`-index gather derive
   *different* Fuel keys.
2. **Emit `u32`-index kernels.** Fuel is **U32-index everywhere** — the graph layer
   hard-requires U32 indices at graph-build time (backend-agnostic), and both
   `cpu_link` and `cuda_link` key the index operand as a fixed U32 slot → `[T, U32,
   T]`. An `i32`/`i64`-index gather is **unreachable** from Fuel today (no graph node
   can carry it). Emit `u32`; `{u32,i32,i64}` is harmless as a forward-compatible
   superset but only `u32` binds.
3. **Yes — advertise an OOB policy field.** Fuel's gather/index_select is
   **in-bounds-only and returns a typed error on OOB** — it does *not* skip,
   zero-fill, or clamp. That is a genuine semantic mismatch with your generated
   gather (skips) and embedding (zero-fills), so the policy should be explicit in the
   contract. Fuel advertises `error`. Fuel adds the schema slot + import validation
   **when you wire the gather contracts** (sequenced behind the consumer, not built
   speculatively).
4. **Model B / token-layout constraints: N/A** — we chose Model A; Fuel has no
   structure_key token for you to lay out. If Baracuda's *own* internal key wants a
   per-operand dtype field, that is a Baracuda-internal choice Fuel never consumes.

---

## Q1 — Model A or B? → **Model A**, and the premise needs a correction

**The premise to correct.** The ask frames Fuel as having a coarse `structure_key`
token that is the "sole admissibility predicate," with the `accept` block possibly a
"human-readable gloss subordinate to" it. **Fuel has no such token.** Fuel's binding
table is keyed directly on a per-operand dtype tuple:

- **The key type is the per-operand dtype list.** `KernelDTypes` is
  `SmallVec<[DType; 8]>` — "per-operand dtypes (inputs in order, then outputs)"
  (`fuel-dispatch/src/kernel.rs:52`, `:687`). The binding map is
  `HashMap<(OpKind, KernelDTypes, BackendId), …>` (`kernel.rs:800`). There is no
  layout/size token in the key at all — dtype admissibility *is* the tuple.
- **Registration builds the tuple straight from the accept block.** The importer's
  `assemble_dtype_variants` (`fuel-dispatch/src/fkc/lower.rs:561`) walks
  `accept.inputs`, resolves each operand's dtype list, and emits the key as
  `[input dtypes in order] ++ [output dtypes]`. A **fixed** (single-enumerated) input
  contributes exactly that dtype (this is how `where`'s `cond`=U8 and `masked_fill`'s
  `mask`=U8 land as fixed slots); a **varying** input drives the §3.4 fan. So the
  per-input `dtype:` is not a gloss — it is the key.
- **Lookup builds the tuple from the node's actual operands.** `lookup_with_caps`
  keys `(op, SmallVec::from_slice(dtypes), backend)` (`kernel.rs:1242`) against that
  same map. A miss is a typed `NoBackendForOp`, never a coerce-and-bind.

**Consequence for gather.** Because the index operand occupies its own slot in the
tuple, an `i32`-index call derives key `[T, I32, T]` and an `i64`-index call derives
`[T, I64, T]` — *distinct keys*. A kernel registered at `[T, U32, T]` (Fuel's current
gather slot) simply **does not match** an `i32`/`i64`-index call. The wrong-bind your
honest-miss rule forbids is **structurally impossible** on Fuel's side; the honest
miss is the default. You do not need the incidental `vec_width` leakage to
discriminate — the dtype slot itself does it.

**Decision: Model A.** Fill `accept.inputs[i].dtype` with the *actual* per-operand
dtype (data `T`, index dtype). Fuel keys on it as-is. **No `STRUCTURE_KEY_VERSION`
bump, no coordinated schema evolution.** This is a Baracuda-only change (emit the
contract honestly) exactly as your "Not blocking" note anticipated for Model A.

> Note on your token: if Baracuda's *own* `StructureKey` still wants a per-operand
> dtype field for Baracuda-internal keying, that's fine and orthogonal — Fuel never
> reads your token. Fuel binds off the FKC `accept`/`return` blocks. So Model B on
> *your* side (for your own reasons) and Model A on *the seam* are not in tension.

## Q2 — Index dtype reconciliation → **emit `u32`; Fuel is U32-index everywhere**

Fuel's `[T, U32, T]` slot is not a CPU-link accident — it is enforced at the **graph
contract** level, before any backend is chosen:

- `index_select`: *"index tensor must be U32"* (`fuel-graph/src/lib.rs:6220`)
- `gather`: *"index tensor must be U32"* (`lib.rs:6262`)
- `index_add` / `scatter_add`: *"index must be U32"* (`lib.rs:6466`, `:6509`)

Because this is a graph-build check, **a non-U32-index gather/index_select node
cannot be constructed in Fuel at all.** Both backends agree at the dispatch layer:
`cpu_link.rs` keys the index as a fixed U32 slot → `[T, U32, T]`
(`fuel-dispatch/src/fkc/cpu_link.rs:662`), and `cuda_link.rs` likewise — *"the `U32`
index / `U8` mask slot is a FIXED single-dtype operand"* → `[T, U32, T]`
(`cuda_link.rs:634`). (Heads-up on a naming collision: your `gather_f32_i32` names the
`data_index` pair, but Fuel's CUDA `gather_i32` symbol names the **data** dtype with an
*implicit* U32 index — same coordinate, different spelling.)

**Decision: emit `u32`-index gather/index_select/embedding kernels.** They bind Fuel's
existing slot with zero Fuel change. `i32`/`i64`-index kernels would be **dead from
Fuel's side** — no graph node can carry a non-U32 index, so Fuel could never key
against them. If it's cheap for you to also emit `i32`/`i64` for other consumers,
`{u32,i32,i64}` is a harmless superset, but do not count on Fuel exercising anything
but `u32` today.

**Honest caveat we owe you.** Fuel's U32-index choice **diverges from PyTorch/torch
convention** (torch gather/index uses `i64`), and our own working agreement says
"match external convention for well-known ops." So U32-index is a known internal wart,
not a considered endorsement of U32 over i64. *Widening* Fuel's graph contract to
accept i64 indices is a **separate, larger Fuel decision** — it ripples through every
index op (the four asserts above), the CPU/CUDA fixed-U32 slots, and the
`assert!`→`Result` never-panic cleanup those call sites also need — and it is **not
gated by this dispatch-model ask.** If/when Fuel widens, your `i32`/`i64` kernels
become reachable and the superset pays off; until then, `u32` is the contract.

## Q3 — OOB contract → **yes, advertise an OOB policy; Fuel is `error` (in-bounds-only)**

Fuel's CPU gather/index_select **validate every index and return a typed error on
OOB** — they do not skip, zero-fill, or clamp:

- `index_select_cpu`: on `i >= source_dim_size` returns `Error::Msg("… out of bounds
  for source dim …")` (`fuel-cpu-backend/src/byte_kernels.rs:1833`); doc: *"Out-of-
  bounds indices return a typed error rather than reading garbage"* (`:1786`).
- `gather_cpu`: on `src_dim_idx >= source_shape[dim]` returns `Error::Msg("gather_cpu:
  index … out of bounds …")` (`byte_kernels.rs:2213`).

So Fuel's contract is: **caller guarantees in-bounds; OOB is a validated hard error,
never silent.** That is a real mismatch with your generated gather (skips the cell)
and embedding (zero-fills). Binding a skip/zero-fill kernel to a call site that
assumes error/in-bounds semantics would be a silent behavior change — exactly the kind
of thing the contract should make explicit rather than leave to entry-point-name lore.

**Decision: add an OOB-policy field to the gather contract.** Proposed value set:
`{ in_bounds_only | error | skip | zero_fill | clamp }` (fold `in_bounds_only`/`error`
if you prefer a 4-value set — the distinction is "UB if OOB" vs "defined error if
OOB"; Fuel is the latter). **Fuel advertises `error`.** Your generated dense gather
advertises `skip`; embedding advertises `zero_fill`. With the policy on the contract,
Fuel's matcher can refuse to bind a `skip`/`zero_fill` kernel where `error`/in-bounds
is required (or accept it knowingly), instead of the seam papering over the
difference.

**Fuel-side sequencing.** This is an additive FKC schema field (`#[serde(default)]`,
no `deny_unknown_fields`, so forward-compatible per FKC G7/§11). Consistent with our
"no consumer is a reason to *sequence behind*, not skip" norm, **Fuel wires the schema
slot + import validation when you wire the gather contracts** — we won't land an
unvalidated field speculatively ahead of the emitter. The moment #4 emits a gather
contract with an `oob_policy`, Fuel reads and enforces it.

## Q4 — Model B token-layout constraints → **N/A**

We chose Model A, so there is no Fuel-side `structure_key` token whose byte layout
needs a new per-operand dtype field. Nothing to constrain, no back-compat surface to
protect on the token. (If you evolve your *own* token for your own keying, Fuel is
indifferent — it binds off `accept`/`return`.)

---

## What changes, and where

| Side | Change | When |
|---|---|---|
| **Baracuda** | Fill `accept.inputs[i].dtype` with the real per-operand dtype (data `T`, index `u32`). Emit **`u32`**-index gather/index_select/embedding. Add an `oob_policy` field (Fuel wants `error` advertised for its consumers; you advertise `skip`/`zero_fill` for yours). | Model A ⇒ Baracuda-only; unblocks keyed dispatch for the whole gather family. |
| **Fuel** | **Nothing for Q1/Q2** — the `[T, U32, T]` slot already keys per-operand dtype honestly. **Q3:** add the additive `oob_policy` schema slot + import validation. | Sequenced to land *with* your first gather contract, not before. |

**Not blocking, confirmed from our side too.** Gather runs AOT for you today and Fuel
has no gather consumer waiting on the seam, so there is no urgency — but the model is
now pinned: **Model A, `u32` index, explicit OOB policy.** Wire it whenever #4 is
ready; it's a Baracuda-only change plus Fuel's sequenced OOB-field follow-up.

---

### Source anchors (for your audit)

- Binding key = per-operand dtype tuple: `fuel-dispatch/src/kernel.rs:52,687,800,1242`
- Importer builds the key from `accept`: `fuel-dispatch/src/fkc/lower.rs:561`
  (`assemble_dtype_variants`), schema at `fuel-dispatch/src/fkc/schema.rs:244`
  (`TensorDesc.dtypes`)
- Gather slot `[T, U32, T]`: `fuel-dispatch/src/fkc/cpu_link.rs:662`,
  `fuel-dispatch/src/fkc/cuda_link.rs:634`
- U32-index graph contract: `fuel-graph/src/lib.rs:6220,6262,6466,6509`
- OOB = typed error: `fuel-cpu-backend/src/byte_kernels.rs:1833` (index_select),
  `:2213` (gather)

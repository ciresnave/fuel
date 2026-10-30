# Hybrid decode state (ROADMAP row 117, GAP-098 class): design, before any code

Facts read at `origin/main@750cd2fc3` (`fuel-inference/src/decode_state_spec.rs`, `fuel-inference/src/multi_session.rs`,
`fuel-transformers/src/models/lazy_qwen3_5.rs`, `docs/gaps.md` GAP-098 and GAP-166). Evidence tags: **READ** (source
opened), **ARITHMETIC** (computed from a stated assumption), **NOT MEASURED**.

## 1. A correction to the audit note first

`docs/design-notes/2026-10-30-qwen35-27b-audit.md` section 3 says `LayerStateSpec::Slots(Vec<StateSlot>)` "can describe a
GDN layer's state". **That is wrong, and it is the dangerous kind of wrong.** **READ** `StateSlot::trailing` is the
*per-token* trailing shape, "excluding the sequence axis": a slot is a buffer that **grows by one entry per token**
(`LatentCache` appends). A Gated DeltaNet layer's recurrent state `S` (`[heads, dk, dv]`, f32) and its conv window have
**no sequence axis**: they are overwritten every step and never grow. Describing `S` as a `Slots` entry type-checks and
would allocate a cache that grows by `heads*dk*dv` elements per token: the right size at token 1, the wrong state
forever after. This is the same failure `decode_state_spec.rs` was written to prevent (a spec that is syntactically
satisfiable and allocates the wrong state), one level down. The audit note is corrected in the same PR.

## 2. What state a hybrid layer keeps

**READ** `Qwen3_5Model::forward` is a typed refusal today (GAP-368); no forward exists, so row 117 cannot be exercised
end to end until row 115. What can be settled now is the *vocabulary*:

| Layer kind | State | Grows with tokens? |
|---|---|---|
| full attention (16 of 64 in the 27B, per the audit note) | K and V, `[n_kv_heads, head_dim]` per token | yes |
| Gated DeltaNet (48 of 64) | `S`: `[heads, dk, dv]` f32; conv window: `[conv_dim, K-1]` | **no, fixed size** |

**ARITHMETIC** (audit note section 3 assumptions, `dk = dv = 128`, `K = 4`): about 3.1 MB of `S` plus 123 KB of conv state
per GDN layer, about 151 MB across 48 layers per sequence, independent of context length. LFM2's ShortConv blocks (GAP-098)
are the same kind with a smaller window.

## 3. Proposed vocabulary

Add one `LayerStateSpec` variant, named for the property that varies (fixed vs growing), not for the first model that
needs it:

```text
LayerStateSpec::Fixed(Vec<FixedSlot>)     // FixedSlot { shape: Vec<usize>, dtype: DType }
```

- `dtype` is carried because the recurrence state is f32 even when weights and activations are bf16 or quantized; a spec
  that cannot say so invites a bf16 `S`, which is a numerics change nobody chose.
- `slots()` stays the *growing-slot* view: it returns the per-token slots and **returns none for `Fixed`**; a new
  `fixed_slots()` returns the fixed ones. `kv_dims()` stays `None` for `Fixed` (it never fabricates a pair).
- `collapse_uniform` keeps failing loudly for any model with a `Fixed` layer. It is not widened: a hybrid model does not
  collapse to one `(n_kv_heads, head_dim)`, and a helper that picked "the attention layers" and ignored the rest would be
  the hiding place the module's own comment warns about.
- A new **`HybridGeometry`** (name provisional) is what the scheduler consumes for a hybrid: the *attention* layers'
  uniform `(n_kv_heads, head_dim)` plus the ordered list of per-layer kinds, so `KvCache` is allocated for attention
  layers only and a separate fixed-state container for the rest. It is built by a fallible constructor that refuses a
  model whose attention layers are not uniform, exactly as `collapse_uniform` does today.

**Variant gate.** `LayerStateSpec` consumers need a workspace-scope check, not `-p fuel-inference` alone (CLAUDE.md, enum
variant on an exhaustive enum). Step 1 below enumerates the `match` sites before the variant lands.

## 4. Where the state lives at runtime

**READ** plan-once decode (Phase D) keeps the graph fixed and re-binds inputs per token; the KV write is a
`WriteSlice` at a symbolic offset into a fixed-capacity buffer. A fixed-size recurrent state needs the opposite shape of
write: a **full overwrite** of a same-sized buffer, output of step *t* bound as input of step *t+1*. Two ways:

1. **Functional**: the graph takes `S_in` and returns `S_out`; the decode session swaps the two bindings after each step.
   Fits the lazy-DAG premise (no mutation in the graph), costs one extra buffer of `S` per layer (about 151 MB for the
   27B, ARITHMETIC) and a rebind per step. **Recommended**: it needs no new op and no ordering analysis.
2. **In place**: a `WriteSlice`-style overwrite of `S`. Saves the extra buffer but puts a destructive write on a buffer
   that the same step also reads, which `derive_ordering` / `insert_safety_copies` must prove safe (the parallel-branch
   audit found destructive single-owner writes are exactly where sharing breaks). Not first.

Whether the rebind keeps plan-once decode's measured win is **NOT MEASURED**; the rebind count grows from 2 (K, V per
attention layer) to 2 plus 2 per GDN layer, about 100 more per token at 48 layers.

## 5. Scheduler consequences

- **Forking and prefix sharing** (`KvCache::cloned_persistent`, rung-1 prefix sharing): a recurrent state is a summary of
  the *whole* prefix, so a session can share a prefix only at the exact token where a state snapshot was taken. Rung-1
  sharing of a KV block range does not carry over; for a hybrid, prefix reuse needs a state snapshot per shared point
  (151 MB each for the 27B, ARITHMETIC). Stated as a limit, not solved here.
- **Truncate / rollback** (speculative decode, `truncate_to`): impossible on a fixed-size state without a snapshot.
  A hybrid model declines rollback with a typed error until snapshots exist.
- **Paged decode** (`PagedDecodeModel`) for a hybrid is Lightbulb's ask; the attention layers page as today, the fixed
  state is per session and not paged. Sequence-count capacity is bounded by `151 MB * sessions` in addition to KV.

## 6. Steps (each red-first; none starts before row 115 has a forward)

1. Enumerate every `match`/construct site of `LayerStateSpec` and `DecodeModel::layer_state_specs` (workspace, all
   features) and record the count with its predicate. No code.
2. Red test: a `Fixed`-layer spec through `collapse_uniform` must fail loudly; a `Fixed` spec must not appear in
   `slots()`. Then the variant.
3. `HybridGeometry` + fallible constructor, with the sabotage sibling (non-uniform attention layers refuse).
4. Fixed-state container and the functional rebind in `DecodeSession`; the oracle is a hybrid toy (2 GDN + 1 attention
   layer, tiny dims) whose per-token decode logits must equal the whole-sequence forward's logits at each position.
5. Measure plan-once decode with the rebinds on the 0.8B file against the composed forward without a persisted plan.

## 7. Not settled

- The 27B's attention KV head count (still unread) and whether its 16 attention layers are uniform.
- Whether `StateSlot` should gain an `is_growing` flag instead of a new variant (smaller diff, but it moves the meaning
  of `Slots` for existing implementors; the module's own text argues for a new variant).
- LFM2 (GAP-098) would adopt `Fixed` for its conv window; whether to fold that in here or leave it is the PM's call.

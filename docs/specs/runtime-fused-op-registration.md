# Runtime fused-op registration — adopting a synthesized kernel (design)

**Status:** design, 2026-06-20. Branch `feat/kernel-contracts-dlpack`.
**Consumes:** [fkc-fusion-patterns.md](fkc-fusion-patterns.md) §3 (the `PatternNode` grammar),
§3a (matching semantics), §5 (JIT-on-request). **Builds on:** the now-live declarative
fusion engine (`PatternKind::Declarative` → `crate::jit::match_region`,
[fuel-graph/src/opt.rs](../../fuel-graph/src/opt.rs), commit `1ed5713c`).

This is the **Fuel-side half** of JIT-on-request §5: once Baracuda (the synthesizer)
returns a kernel + contract for a Fuel-chosen region, how does Fuel *adopt* it — give it a
runtime identity, fuse matching subgraphs to it, dispatch to its kernel, and (the honesty
invariant) decompose it back to primitives when the kernel is absent. It is equally the path
for **any** import-time-registered fused op (a backend that ships a fused kernel + a `pattern:`),
not just JIT — JIT is the dynamic case of the same mechanism.

---

## 1. The problem: three build-time-closed things

A synthesized/imported fused op needs an **identity**, **params**, and **metadata**, and all
three of Fuel's representations are closed at build time:

1. **`FusedOpId(pub u16)`** — build-time ids are dense `1..N`, assigned in
   [registry.rs](../../fuel-graph/src/registry.rs) (`SOFTMAX_LAST_DIM = 1`, … `CONV2D = 6`, …).
2. **`FusedOpParams`** — a closed enum; a synthesized op's per-instance params are not one of
   its variants.
3. **`FusedOpRegistry`** — "Built at process startup, frozen thereafter (architecture v1.0:
   no runtime extensibility)" ([registry.rs](../../fuel-graph/src/registry.rs):695). Holds the
   `pattern`, `decompose`, `shape_rule`, `dtype_rule` for each static fused op.

The design lifts all three **without disturbing the static path** — static ids stay dense and
stable, the static registry stays frozen, and `FusedOpParams`' existing variants are untouched.
Runtime ops live in a parallel, append-only **sidecar** that the static lookups fall through to.

## 2. Runtime identity: a reserved `FusedOpId` range

`FusedOpId` is a `u16`. Reserve the top of the range for runtime ids:

```rust
pub const RUNTIME_FUSED_BASE: u16 = 0x8000; // 32768
impl FusedOpId {
    pub fn is_runtime(self) -> bool { self.0 >= RUNTIME_FUSED_BASE }
}
```

- Static ids: `1 ..= 0x7FFF` (dense, stable, build-time — far more headroom than the ~dozen
  real fused ops will ever need).
- Runtime ids: `0x8000 ..= 0xFFFF`, allocated by an `AtomicU16` at registration.
- The single `is_runtime` predicate is the **routing bit**: every `FusedOpId` consumer
  (dispatch, decompose, `op_short_name`, telemetry) checks it and routes to the runtime
  sidecar instead of indexing the static `Vec`. No collision is possible by construction.

`is_runtime` is the *only* new branch the static path grows; the static `Vec`-indexed lookups
are unchanged for `!is_runtime` ids.

## 3. Runtime params: one new `FusedOpParams::Runtime` variant

A synthesized epilogue's only per-instance state is its **extracted scalar args** — the
`extract:` slots of the emitted `pattern:` (FKC §5.3: `AddScalar.value`, `Clamp.min/.max`, a
`Reduction.axis`). One variant carries them:

```rust
FusedOpParams::Runtime { scalars: SmallVec<[Scalar; 4]> },
```

- A parameterless synthesized op (`relu(add(a,b))`) → `Runtime { scalars: [] }`.
- A scalar-param epilogue (`clamp(mul_scalar(x, k), lo, hi)`) → `Runtime { scalars: [k, lo, hi] }`,
  in `extract:`-slot order. The op id (in `Op::Fused(id, _)`) selects *which* runtime op; the
  scalars are *its* bound values.

**Ripple (bounded, verified 2026-06-20):** every `decompose` already has a `_ => …` catch-all,
so only the two *exhaustive* matches need a `Runtime` arm —
`FusedOpParamsKey` ([registry.rs](../../fuel-graph/src/registry.rs):~448, the CSE/dedup key:
`tag` + `op_id` + a hash of `scalars`) and the param projection in
[pipelined.rs](../../fuel-dispatch/src/pipelined.rs). Two arms, not a sweep.

## 4. Runtime metadata: the append-only sidecar registry

A process-global, append-only registry, parallel to the frozen static one:

```rust
pub struct RuntimeFusedOpEntry {
    pub id:      FusedOpId,        // >= RUNTIME_FUSED_BASE
    pub name:    String,           // synthesized, e.g. "jit::relu_add::sm89::<hash>"
    pub region:  crate::jit::PatternNode, // THE recipe — the §3 region, OpTag-keyed
    pub kernel:  Option<KernelRef>,// the synthesized binding; None ⇒ decompose-only
    // shape/dtype come from the region's sink at decompose time; no rules needed.
}

static RUNTIME_FUSED_OPS: RwLock<Vec<RuntimeFusedOpEntry>>; // index = id - RUNTIME_FUSED_BASE
static NEXT_RUNTIME_ID: AtomicU16; // starts at RUNTIME_FUSED_BASE
```

This is the **Tier-2** extensibility named in the kernel-seam program: the static registry is
Tier-0 (frozen build-time), the link-registry kernel bindings are already Tier-1-extensible
(`extend_global_bindings`), and this is the Tier-2 *metadata* sidecar. `RwLock` (not `OnceLock`)
because it grows across the run; reads (dispatch/decompose, the hot direction) take the read
lock, registration (rare) takes the write lock. Lookups try static first (`!is_runtime`), then
the sidecar.

## 5. The recipe principle holds: `decompose` = the region, re-emitted

The load-bearing simplification: a runtime fused op **is** its region, so its `decompose` is
not a hand-written function — it is the region re-emitted as primitives, the exact inverse of
the `match_region` fold that created the fused node:

```
decompose(graph, fused_id, Runtime { scalars }):
    entry  = RUNTIME_FUSED_OPS[fused_id]
    inputs = graph.node(fused_id).inputs          // [in0, in1, …]
    re-emit entry.region bottom-up:
        Bind { index: i }      → inputs[i]         // external leaves
        Op { op, operands, .. } → graph.push(primitive op, re-emitted operands,
                                              scalars stamped into AddScalar/Clamp/… slots)
    return the new root (the re-emitted sink)
```

Because `OpTag` is the **functional-primitive** vocabulary only (the reconcile §3 deltas: no
in-place, no structural ops), every node the region re-emits is in the build-time-closed
primitive basis. So this `decompose` is **total + never-panic + primitive→self** by
construction — it satisfies G1/G2/G3 (the recipe principle, `10-decisions-log.md` 2026-06-20)
for free, with no per-op authoring. A synthesized op can never be an opaque-op gap: its recipe
*is* the region Fuel handed the synthesizer.

## 6. The honesty invariant: kernel-absent ⇒ primitives, never a crash

`kernel: Option<KernelRef>` is the safety hinge:

- **Kernel present** — dispatch resolves `Op::Fused(runtime_id, _)` via the sidecar to the
  synthesized `KernelRef` and runs it.
- **Kernel absent** (JIT declined, link failed conformance, a cold-start graph before
  synthesis completes, or a serialized graph reloaded where the kernel is gone) — the op
  **decomposes to its region** (§5) and runs on primitives. Identical numerics, slower. The
  fused node is a *performance* claim layered on a primitive *correctness* recipe that always
  exists. This is the FDX/FKC honesty discipline applied to identity: a fused op never asserts
  a capability it can't deliver, because the primitive floor is always reachable.

## 7. End-to-end flow

```
Fuel optimizer picks a region R (a PatternNode, §3 grammar) in a base-map graph
        │
        ├─ SeamHello negotiated SeamCapJitOnRequest? ──no──> leave R as primitives
        │                                              yes
        ▼
JitRequest{ region: R, operands: [OperandDesc…], arch } ──direct-Rust──▶ Baracuda synthesize
        ▼
JitResponse{ kernel, contract{ pattern:, extract:, cost: } }  (or Declined ⇒ stay primitive)
        ▼
Fuel adopts:
  1. id   = NEXT_RUNTIME_ID.fetch_add(1)
  2. RUNTIME_FUSED_OPS.push(entry{ id, region: R, kernel: Some(ref) })
  3. link-registry.extend(contract.entry_point → kernel)          // Tier-1, already exists
  4. register a FusionRule from SubgraphPattern::Declarative(PatternTree{
         root: contract.pattern, params: Runtime{ scalars: extract-slots } })  // §3, NOW LIVE
        ▼
Next optimize(): match_region folds every matching R-subgraph → Op::Fused(id, Runtime{…})
        ▼
realize: dispatch → sidecar → synthesized kernel   (absent ⇒ decompose → primitives, §6)
```

Step 4 is the piece that **just landed** (`1ed5713c`): the declarative engine fuses a
registered `PatternNode`. Steps 1–3 + the dispatch/decompose routing (§3–§6) are the remaining
implementation.

## 8. Cost-gated adoption (Fuel stays the strategist)

Adoption is not unconditional — JIT-on-request §5 keeps Fuel the cost authority. The
`contract.cost:` AST rides the cost trampoline (the `cost_expr` eval core, already built):
Fuel evaluates the synthesized op's cost against the region's primitive-path cost and **only
registers the FusionRule if the fused estimate wins**. A synthesized kernel that doesn't beat
primitives is dropped on the floor — the sidecar never grows for a loss. This is why the region
is *Fuel's* choice (it picks where to spend a synthesis) and adoption is *Fuel's* gate (it
refuses a kernel that doesn't pay). Baracuda synthesizes; Fuel decides.

## 9. Implementation increments (sequenced)

1. **`FusedOpId::is_runtime` + `RUNTIME_FUSED_BASE`** — the routing predicate (no behavior yet).
2. **`FusedOpParams::Runtime { scalars }`** + the two exhaustive-match arms (§3). Lands with the
   sidecar (§4) so the variant has a constructor + consumer — not before (no orphan capability).
3. **`RuntimeFusedOpEntry` + `RUNTIME_FUSED_OPS` sidecar + register/lookup** (§4), `is_runtime`
   routing in `op_short_name`/telemetry.
4. **Region-re-emit `decompose`** (§5) — the generic `PatternNode → primitive subgraph`
   walker; test: register a region, fuse, decompose, assert numeric identity vs the un-fused
   graph (the totality/never-panic gate).
5. **Dispatch routing** (§6) — `Op::Fused(runtime_id, _)` → sidecar kernel, kernel-absent ⇒
   force decompose. Test the absent-kernel fallback explicitly.
6. **Adoption entry point** (§7 steps 1–4) + **cost gate** (§8) — `adopt_synthesized(region,
   contract, kernel) -> Option<FusedOpId>`, gated on the cost-trampoline comparison.
7. **`JitRequest`/`JitResponse` wire types** + the live `synthesize` call (FKC §5 transport),
   then advertise `SeamCapJitOnRequest`.

Increments 1–5 are pure Fuel-internal and independently testable against a hand-built region
(no Baracuda dependency). 6–7 are the live seam. None of 1–7 blocks Baracuda's reconcile — the
two frozen types they wait on (`OperandDesc`, `PatternNode`) are already cut (`2d31443d`).

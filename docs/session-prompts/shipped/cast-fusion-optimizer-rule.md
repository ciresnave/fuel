# Session prompt — Cast-fusion optimizer rule

## What this session is for

Add an optimizer rule that fuses `Cast(src_dtype → dst_dtype) → Consumer(dst_dtype)` into a single `Consumer(src_dtype)` call whenever the consumer has a kernel registered for the cast-source dtype. The cast node becomes orphan and is dropped from the live subgraph.

The architectural payoff: today, every `(op, dtype, backend)` cell without a native kernel forces the user (or an auto-Contiguize-style executor pass) to insert a `Cast` to a dtype the consumer *does* support, then do the work, then potentially cast back. That's at least one extra kernel launch per cast, plus an intermediate materialization that consumes memory bandwidth. The fusion rule eliminates the extra launch + materialization whenever the consumer can handle the source dtype natively.

Strictly more general than "add native kernels for every (op × dtype × backend)" cell:

- Where native kernels exist for both `src` and `dst`, the rule picks `src` and drops the cast — saves a launch.
- Where only `dst` has a native kernel, the rule doesn't fire — exactly today's behavior, no regression.
- Where only `src` has a native kernel (rare but possible after dtype-coverage expansion), the rule fires and the cast is dropped.

Doesn't replace native-kernel expansion (native kernels are still faster for some shapes, particularly tiny ops where launch overhead dominates). Pairs with the CPU trait-chassis refactor (see `docs/session-prompts/cpu-kernel-trait-chassis-refactor.md`) — together they cover both the high-leverage native cells and the long tail.

This session is parallel-safe with most other work, including the chassis refactor (different files, different layer). It touches `fuel-graph/src/opt.rs` (where existing rules live) and adds queries against the binding table at rule-construction time — coordinate with any other session adding new rules.

## Read first (in this order)

1. **`docs/architecture/04-optimization.md`** §"Cost model" + §"Per-decision-point alternatives" — the architecture-v1.0 commitment to alternatives-not-replacement. Cast-fusion is *one* alternative at a decision point; the original "cast-then-consume" path is *another*. The optimizer ranks them.
2. **`docs/architecture/04-optimization.md`** §"Cross-cutting transformations the optimizer is responsible for" — dtype changes (Cast insertions) are explicitly listed there. Cast-fusion is the inverse direction: dtype-change *removal*.
3. **`fuel-graph/src/opt.rs`** §`pub trait Rule` (line ~100) — the Rule trait. The new cast-fusion rule implements this same trait.
4. **`fuel-graph/src/opt.rs`** §`pub enum RuleFamily` (line ~78) — two existing families: `Lowering` and `Fusion`. The cast-fusion rule fits `Fusion` semantically (it shrinks the graph) but is arguably its own family `Algebraic`. **Engage critically**: pick the cleaner home and document why. If `Algebraic` makes sense, add the variant; if `Fusion` is fine, use it.
5. **`fuel-graph/src/opt.rs`** §existing `LoweringRule` / `FusionRule` (lines ~330–430) — concrete `Rule` implementations. The cast-fusion rule has a similar shape: a matcher + a rewriter.
6. **`fuel-storage/src/dispatch.rs`** §`KernelBindingTable` — the source of truth for "which (op, dtypes, backend) cells have kernels registered." The cast-fusion rule must consult this to know whether to fire on a given match.
7. **Memory entry `project_pr3_rule_registry_shipped.md`** — historical context on the rule registry's introduction (PR 3, commit `37663bb7`).
8. **Memory entry `project_phase_7_6_step_3_shipped.md`** — current state of registry-driven lowering + fusion rules for fused ops. The cast-fusion rule is the first algebraic rule landing through this same registry.

## What this session must NOT do

- **Don't fire on Cast where the source has more than one consumer.** A `Cast` whose output feeds two consumers can't be eliminated for either consumer without recomputing the cast for the other. Either: (a) only fuse when the cast has exactly one consumer, OR (b) fuse for one consumer and keep the cast alive for the others (DAG-correct but more complex). Pick (a) for the MVP — it covers the common case (insertion-from-promotion) and the complex DAG case can land in a follow-up if it shows up in profiles.
- **Don't fire on Cast that crosses Device/Backend boundaries.** The cast-fusion rule changes the consumer's input dtype, not its placement. If a Cast is doing a device-side dtype change while a Copy is doing the device transfer, those compose differently and the rule shouldn't conflate them. Match only on intra-device casts.
- **Don't fire on Cast where the consumer's kernel for the source dtype has incompatible `PrecisionGuarantee`.** If the user's tolerance budget admits the dst-dtype kernel but not the src-dtype kernel (e.g., dst is f32-bit-stable, src is bf16-approximate), the rule must respect that. Per architecture v1.0 §"Precision-filter pass," precision filtering runs before cost ranking — so the rule registers the fusion as an *alternative* at a decision point, and the precision-filter prunes it if it would violate guarantees.
- **Don't introduce `Op` enum variants.** The rule operates on existing `Op::Cast` + consumer ops; no new IR is needed.
- **Don't query the global binding table from inside the rule's `matches`.** Pass a capability predicate at rule construction (closure or trait object); the rule's `matches` calls the predicate. This keeps `fuel-graph` decoupled from `fuel-storage` — `fuel-graph::opt` doesn't know binding tables exist, just "this predicate says yes."
- **Don't push to remote.**

## Branch and starting state

- **Current branch**: whichever the user is on. Verify with `git status` + `git log --oneline -5`.
- **Coordination**: parallel sessions are likely. The chassis refactor session works in `fuel-cpu-backend`; this works in `fuel-graph/src/opt.rs`. No file conflict. Op-adding sessions touch `fuel-graph/src/lib.rs` (Op enum) — no conflict with this session's rule additions.

## Concrete work — sized

### Step 1 — Decide the rule's home

Two options:

**Option A**: extend `RuleFamily` with `Algebraic`. New family runs as a third phase after Fusion: lowering → fusion → algebraic. Cast-fusion is the first algebraic rule; future algebraic rewrites (e.g., constant folding, common-subexpression hoisting, identity elimination) join the same family.

**Option B**: classify cast-fusion as `RuleFamily::Fusion` because the architectural comment for Fusion says "always shrinks the graph" and cast-fusion does shrink (eliminates one node per fire). Document the broader meaning of "fusion" to encompass algebraic rewrites that contract two nodes into one.

**Recommendation**: Option A. The Phase 7.6 design talks about distinct rule families (lowering, fusion, algebraic, tolerance-gated) per architecture v1.0; this is the first algebraic rule and adding the family now sets up the long-term shape. But A is more code (new enum variant, new pass loop, ordering decision). Engage critically and pick.

### Step 2 — Implement the rule

```rust
// fuel-graph/src/opt/cast_fusion.rs (new file) or inline in opt.rs.

/// Predicate: "does the consumer at `consumer_id` have a kernel for
/// the cast-source dtype?" Supplied at rule construction so the rule
/// stays decoupled from any specific binding-table implementation.
pub type CapabilityPredicate = Arc<dyn Fn(/* consumer: */ &Op, /* src_dtype: */ DType) -> bool + Send + Sync>;

pub struct CastFusionRule {
    capabilities: CapabilityPredicate,
}

impl CastFusionRule {
    pub fn new(capabilities: CapabilityPredicate) -> Self {
        Self { capabilities }
    }
}

impl Rule for CastFusionRule {
    fn name(&self) -> &'static str { "cast_fusion" }
    fn family(&self) -> RuleFamily { RuleFamily::Algebraic /* or Fusion */ }

    fn matches(&self, graph: &Graph, id: NodeId) -> bool {
        // 1. Node at `id` is the consumer (not the cast itself).
        // 2. At least one input to `id` is a Cast(src→dst) where:
        //    a) the cast has exactly one consumer (= this `id`);
        //    b) the consumer's Op + src_dtype satisfies the capability predicate;
        //    c) the cast doesn't cross device/backend boundaries.
        // 3. The consumer's current input dtype on that edge is the cast's dst_dtype.
        // ... walk inputs, return true if any input meets the conditions.
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        // 1. Find the qualifying input edge (the one whose source is a Cast meeting the criteria).
        // 2. Get the cast's input (`pre_cast_id`).
        // 3. Append a new consumer node identical to the original at `id`, but with:
        //    - the qualifying input replaced by `pre_cast_id`;
        //    - the node's dtype field updated accordingly (it stays at consumer's *output* dtype,
        //      which doesn't change — the consumer still produces what it produced; only its
        //      input dtype changes);
        //    - shape unchanged.
        // 4. remap.insert(id, new_id) so consumer edges get rewired.
        // The original cast node + original consumer node are orphaned; they remain in the arena
        // but are unreachable from roots after consumer rewiring (same garbage pattern as fusion).
    }
}
```

**Key correctness points** to encode in tests:

- Single-consumer constraint: `cast → op` where cast also feeds another node → rule doesn't fire.
- Capability gate: rule only fires when predicate says the consumer supports the source dtype.
- Output dtype preserved: consumer's *output* dtype is unchanged by the rewrite; only its *input* dtype shifts.
- Cast chains: `x → Cast(F32→BF16) → Cast(BF16→F32) → Op` should collapse to `x → Op` (engage: does the rule fire once and the cleanup pass handles the second cast, or do you handle chains explicitly? Pick one.)
- Multi-input ops: `Op::Add(Cast(x), y)` — only the left input matches; the rule fires on that input only, leaves `y` alone.

### Step 3 — Build the capability predicate

`fuel-storage/src/dispatch.rs` (or wherever the binding-table lives) exposes a function that takes an `Op`, an input-dtype, and a backend, and returns whether a kernel is registered. The Router (or whichever crate composes the rule registry) constructs the `CastFusionRule` with a closure that calls this function for whatever backend is current.

```rust
// fuel-storage/src/dispatch.rs (extend the existing KernelBindingTable API)
impl KernelBindingTable {
    pub fn has_kernel_with_input_dtypes(&self, op: OpKind, input_dtypes: &[DType], backend: BackendId) -> bool {
        // Lookup using the existing dispatch-key shape: (OpKind, SmallVec<[DType; 8]>, BackendId).
    }
}

// fuel-graph-router/src/lib.rs or wherever Router is constructed:
let table = /* binding table */;
let cap_predicate: CapabilityPredicate = Arc::new(move |op, src_dtype| {
    // Build the input-dtype vector for this consumer: replace the cast-source position
    // with src_dtype, keep others as the consumer's current input dtypes.
    table.has_kernel_with_input_dtypes(op.kind(), &/* derived input dtypes */, backend)
});
let registry = RuleRegistry::default_rules().with_rule(Box::new(CastFusionRule::new(cap_predicate)));
```

The predicate's signature may need to take more context than the sketch (e.g., the consumer's other input dtypes, the cast's *exact* source dtype) — refine as you implement.

### Step 4 — Tests

New file `fuel-graph/src/opt/cast_fusion_tests.rs` or inline tests in `opt.rs`:

- `cast_fusion_drops_cast_when_consumer_supports_source_dtype` — happy path.
- `cast_fusion_no_fire_when_consumer_lacks_source_dtype` — capability predicate returns false.
- `cast_fusion_no_fire_when_cast_has_multiple_consumers` — single-consumer constraint.
- `cast_fusion_handles_chained_casts` — pick a semantic and assert it.
- `cast_fusion_preserves_output_dtype` — consumer's output dtype unchanged.
- `cast_fusion_no_fire_across_device_boundary` — Cast that's effectively a device transfer doesn't fuse.

End-to-end: an integration test in `fuel-graph-cpu` or `fuel-core` that realizes a small graph with `Cast → Sum → ...`, observes the cast was eliminated, and the result matches the un-fused realize bit-for-bit.

### Step 5 — Wire into `RuleRegistry::default_rules`

Once the rule is correct, add it to the default registry so production realize paths pick it up. Behind a Cargo feature or a constructor parameter — engage critically about whether the rule should be on-by-default immediately or opt-in for a release. Architecture v1.0's commitment is that optimizations are always-on after they pass correctness review; the question is whether this rule is mature enough to commit to that.

## Test commands

After each PR:

```bash
cargo test -p fuel-graph --lib
cargo test -p fuel-storage --lib
cargo test -p fuel-graph-cpu --lib
cargo test -p fuel-core --lib
```

End-of-session sweep:

```bash
cargo check --workspace
cargo test --workspace --lib
```

If existing tests regress, audit whether they relied on the cast being explicit in the optimized graph (e.g., a test that asserts "after optimize, the graph contains a Cast node" would now fail correctly). Fix the test by asserting the *behavior* (output matches expected) rather than the *structure* (cast present).

## Operating principles

- **Engage critically.** The rule's home (Algebraic vs Fusion family), the predicate's signature, the chain-of-casts policy, and the on-by-default decision are real design choices, not mechanical. Surface options, recommend one, document why.
- **No production panics.** The rule's matcher / rewriter return cleanly when the conditions aren't met; no unwraps on graph indexing without bounds-checked alternatives.
- **DAG-correctness first.** Any rewrite that orphans a node whose output is still consumed elsewhere is a bug. The single-consumer constraint exists precisely to prevent this; if you relax it in a follow-up, the relaxation needs careful consumer-edge accounting.
- **Cost-model integration is follow-up.** The MVP rule fires unconditionally when the capability predicate says yes. Architecture v1.0 §"Per-decision-point alternatives" wants the rule to register an *alternative* (cast-eliminated path vs cast-present path), with the cost model picking — but that requires per-decision-point machinery the optimizer doesn't have yet. For this session: rule fires deterministically; the per-decision-point version is a Phase 7.x follow-up.
- **Memory updates per PR.** A short entry (`project_cast_fusion_rule_shipped.md`) capturing the rule's scope, what it fires on, what it doesn't, and any landmines.
- **Don't push to remote unless asked.**

## End-of-session deliverable

At minimum:

- `CastFusionRule` exists in `fuel-graph/src/opt.rs` (or a sibling module).
- Capability predicate plumbed through `KernelBindingTable` API + constructed at Router/registry-init time.
- Tests cover the happy path + the no-fire cases + chained casts + multi-input ops.
- Rule registered in `RuleRegistry::default_rules` (or behind a feature, your call).
- One end-to-end integration test confirms a realistic `Cast → Op` graph realizes correctly with the cast eliminated.

Stretch:

- Decision-point integration (rule registers as alternative, cost model picks). Requires the per-decision-point machinery; defer if it doesn't exist yet.
- Multi-consumer Cast handling (DAG-correct fan-out): rewrite for one consumer, keep cast alive for others.

## Coordination notes

- **Pairs with the CPU kernel trait-chassis refactor** (`docs/session-prompts/cpu-kernel-trait-chassis-refactor.md`). Independent; either order is fine. The chassis makes adding native kernels cheap *when we want to*; the cast-fusion rule means we don't *have to* add a native kernel everywhere because the optimizer takes care of the long tail.
- **Conflict surface**: only with other sessions adding rules in `fuel-graph/src/opt.rs`. Mechanical merge.
- **Future**: once the rule lands, per-(op, dtype, backend) coverage decisions can be made on cost-benefit, not "we need this to avoid implicit casts." The combined effect lets the architecture's "maximum native, zero-cost cast everywhere else" property emerge without combinatorial maintenance.

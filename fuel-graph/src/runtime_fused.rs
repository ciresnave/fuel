//! Runtime fused-op registration — the Tier-2 sidecar
//! (`docs/specs/runtime-fused-op-registration.md`).
//!
//! A runtime-registered (JIT-synthesized or import-time) fused op **is** its
//! region: its identity is a runtime [`FusedOpId`], its recipe is the §3
//! [`PatternNode`] region kept here, and its `decompose` is that region
//! re-emitted as primitives — so the recipe principle (total / never-panic /
//! primitive→self) holds for free, since [`OpTag`] is the functional-primitive
//! vocabulary only. No kernel field: the kernel binding lives in fuel-dispatch's
//! `FusedKernelRegistry` (Tier-1 extensible); this sidecar holds only the
//! graph-side recipe + the optimizer rules built from it.
//!
//! v1 scope: **same-shape elementwise** regions (the synthesizer's increment-1
//! epilogues). Interior shape inference for broadcast/reduction regions is a
//! follow-up — a re-emitted node takes its first operand's shape/dtype, exact
//! for type-preserving same-shape ops and rejected-at-registration otherwise.

use std::collections::HashMap;
use std::sync::RwLock;

use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

use crate::registry::{FusedOpId, FusedOpParams};
use crate::{Graph, Node, NodeId, Op};

/// A runtime-registered fused op's metadata (the graph-side recipe).
#[derive(Clone, Debug)]
pub struct RuntimeFusedOpEntry {
    /// The allocated runtime id (`>= FusedOpId::RUNTIME_FUSED_BASE`).
    pub id: FusedOpId,
    /// A human/telemetry name (e.g. `"jit::relu_add::sm89::<hash>"`).
    pub name: String,
    /// The §3 region (the subgraph sink) — the op's primitive recipe.
    pub region: PatternNode,
}

/// A registration failure — never a panic (build-time validation).
#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeFusedError {
    /// The region's bind indices don't form a contiguous `[0, n)` (the op's
    /// external inputs).
    NonContiguousBinds(Vec<u8>),
    /// The region carries an op with no primitive re-emission (outside the v1
    /// re-emit vocabulary) — it could not decompose, so we refuse to register
    /// it (the totality guard).
    UnRepresentable(OpTag),
    /// The region contains a matcher-only node (`Any`/`SeeThrough`) — a
    /// concrete region must be `Op`/`Bind` only.
    NonConcreteRegion,
    /// The runtime id space (`u16` above `RUNTIME_FUSED_BASE`) is exhausted.
    IdSpaceExhausted,
}

static RUNTIME_FUSED_OPS: RwLock<Vec<RuntimeFusedOpEntry>> = RwLock::new(Vec::new());

/// The recipe-identity index for runtime-registered ops: base-map content
/// hash ([`crate::opt::base_map_hash`]) → the [`FusedOpId`] that first
/// registered a region hashing to it. A **sibling** to `RUNTIME_FUSED_OPS`,
/// not a reuse of [`crate::registry::FusedOpRegistry::by_pattern_hash`] —
/// that field lives on the STATIC catalog (`FusedOpRegistry`, an
/// `OnceLock`-frozen struct built at process startup for build-time-known
/// ids `1..RUNTIME_FUSED_BASE`); runtime ops never populate a
/// `FusedOpRegistry` instance at all, they live in this module's own
/// `RUNTIME_FUSED_OPS` global with the disjoint `RUNTIME_FUSED_BASE..` id
/// space, so `by_pattern_hash` is unreachable from here. This index is the
/// natural home for runtime-region dedup: same lifetime/global-ness as
/// `RUNTIME_FUSED_OPS`, cleared in the same breath by
/// `clear_runtime_fused_for_tests`.
///
/// `HashMap::new()` isn't `const`, so this can't be a plain
/// `static … : RwLock<HashMap<..>> = RwLock::new(HashMap::new())` the way
/// `RUNTIME_FUSED_OPS` is a plain `RwLock::new(Vec::new())` — `Vec::new()`
/// is `const`, `HashMap::new()` is not. `OnceLock` lazy-inits it instead
/// (same pattern as `registry.rs`'s `static REGISTRY: OnceLock<..>` and the
/// per-function `OnceLock` CPU-device singletons in `opt.rs`/`grad.rs`).
fn hash_index() -> &'static RwLock<HashMap<u64, FusedOpId>> {
    static IDX: std::sync::OnceLock<RwLock<HashMap<u64, FusedOpId>>> = std::sync::OnceLock::new();
    IDX.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Push `arity` uniform placeholder leaves (`Op::Const`, F32 `[1]`, no
/// storage) onto `g` and return their ids. Uniform + storage-free is
/// load-bearing: two independently-built graphs' leaves must hash
/// IDENTICALLY under [`crate::opt::base_map_hash`] (which folds a const's
/// shape/dtype and silently no-ops on an unpopulated storage slot) for the
/// dedup comparison to be meaningful. Mirrors
/// `fuel_dispatch::jit_ingest::push_placeholder_leaves` — that crate
/// depends on this one (not the other way around), so the few-line helper
/// is duplicated here rather than shared.
fn push_placeholder_leaves(graph: &mut Graph, arity: usize) -> Vec<NodeId> {
    (0..arity)
        .map(|_| {
            graph.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: fuel_ir::Shape::from_dims(&[1]),
                dtype: fuel_ir::DType::F32,
            })
        })
        .collect()
}

/// `region`'s structural-identity hash: emit it onto placeholder leaves
/// (via [`emit_region`]), lower to the primitive base map
/// ([`crate::opt::lower_to_base_map`]), hash the result
/// ([`crate::opt::base_map_hash`]). `None` on any structural failure (a
/// poisoned lock or an empty lowering result) — the caller
/// (`register_runtime_fused`) treats `None` as "hash unavailable" and skips
/// dedup for this registration, never blocking it.
///
/// Every caller in this module runs this AFTER `validate_representable`
/// already passed for the same region, so `emit_region`'s own panic risks
/// (an unrepresentable `OpTag`, a `Bind` index out of range) are already
/// ruled out here — `register_runtime_fused` still wraps the call in
/// `catch_unwind` as the never-panic contract's last-resort guard for
/// anything this doesn't anticipate.
fn region_base_map_hash(region: &PatternNode) -> Option<u64> {
    let n_binds = region.bind_indices().len();
    let scalars = vec![0.0; count_scalar_slots(region)];
    let graph: crate::SharedGraph = std::sync::Arc::new(RwLock::new(Graph::new()));
    let sink = {
        let mut g = graph.write().ok()?;
        let inputs = push_placeholder_leaves(&mut g, n_binds);
        emit_region(&mut g, region, &inputs, &scalars)
    };
    let roots = crate::opt::lower_to_base_map(&graph, &[sink]);
    let root = *roots.first()?;
    let g = graph.read().ok()?;
    Some(crate::opt::base_map_hash(&g, root))
}

/// Register a runtime fused op for `region`, returning its runtime
/// [`FusedOpId`]. Validates **before** allocating that the region's bind
/// indices form the op's input list and that every op re-emits to
/// primitives (totality) — a non-decomposable region is rejected, never
/// registered.
///
/// **Dedup (recipe identity):** a region that is structurally identical
/// (same [`crate::opt::base_map_hash`] over its primitive lowering) to an
/// already-registered region resolves to the EXISTING [`FusedOpId`] instead
/// of minting a duplicate — two calls with the same shape but different
/// `name`s return the same id, and only the first call's `name`/region is
/// kept in `RUNTIME_FUSED_OPS`. Never-panic: hashing runs inside
/// `catch_unwind`; any failure (a poisoned lock, an unanticipated panic) is
/// treated as "hash unavailable" and simply skips the dedup check —
/// registration always proceeds to today's allocate-fresh path either way.
pub fn register_runtime_fused(
    name: impl Into<String>,
    region: PatternNode,
) -> Result<FusedOpId, RuntimeFusedError> {
    let name = name.into();
    let binds = region.bind_indices();
    let n = binds.len() as u8;
    if binds != (0..n).collect::<Vec<_>>() {
        return Err(RuntimeFusedError::NonContiguousBinds(binds));
    }
    validate_representable(&region)?;

    let hash = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        region_base_map_hash(&region)
    }))
    .unwrap_or(None);
    if hash.is_none() {
        eprintln!(
            "register_runtime_fused: base-map hash unavailable for {name:?}; \
             registering without dedup (allocate-fresh fallback)"
        );
    }

    // Hold the hash index's write lock across the whole check-then-insert
    // sequence below (not read-then-separately-write) so two concurrent
    // registrations of the same NEW region can't both miss the lookup and
    // each mint their own id: the second caller blocks on this lock and,
    // once it acquires it, observes the first caller's insert.
    let mut idx = hash_index().write().unwrap();
    if let Some(h) = hash {
        if let Some(&existing) = idx.get(&h) {
            return Ok(existing);
        }
    }

    // The Vec length under the write lock is the allocator: id = BASE + index,
    // so the index is always `id - BASE` with no allocate/push race.
    let mut w = RUNTIME_FUSED_OPS.write().unwrap();
    let raw = FusedOpId::RUNTIME_FUSED_BASE as usize + w.len();
    if raw > u16::MAX as usize {
        return Err(RuntimeFusedError::IdSpaceExhausted);
    }
    let id = FusedOpId(raw as u16);
    w.push(RuntimeFusedOpEntry { id, name, region });
    drop(w);

    if let Some(h) = hash {
        idx.insert(h, id);
    }

    Ok(id)
}

/// The region (recipe) for a runtime fused op, or `None` if `id` is not a
/// registered runtime op.
pub fn runtime_region(id: FusedOpId) -> Option<PatternNode> {
    if !id.is_runtime() {
        return None;
    }
    let idx = (id.0 - FusedOpId::RUNTIME_FUSED_BASE) as usize;
    RUNTIME_FUSED_OPS.read().unwrap().get(idx).map(|e| e.region.clone())
}

/// A runtime op's name (telemetry / `op_short_name` routing).
pub fn runtime_name(id: FusedOpId) -> Option<String> {
    if !id.is_runtime() {
        return None;
    }
    let idx = (id.0 - FusedOpId::RUNTIME_FUSED_BASE) as usize;
    RUNTIME_FUSED_OPS.read().unwrap().get(idx).map(|e| e.name.clone())
}

/// All registered runtime ops — the optimizer iterates this to build a fusion
/// rule + a lowering rule per runtime op (`RuleRegistry::default_rules` /
/// `lowering_only`).
pub fn runtime_entries() -> Vec<RuntimeFusedOpEntry> {
    RUNTIME_FUSED_OPS.read().unwrap().clone()
}

/// **TEST-ONLY.** Clear the metadata sidecar AND the recipe-identity
/// `hash_index` in the same breath. Because the Vec length *is* the id
/// allocator (`id = BASE + index`), clearing restarts allocation — any
/// sidecar keyed by prior runtime ids MUST be cleared alongside it or a
/// reused id resolves stale data. This was already true for
/// `fuel_dispatch::runtime_fused_kernels::clear_runtime_fused_for_tests`'s
/// kernel sidecar (call that one, not this, from dispatch-level tests) and
/// is now ALSO true for `hash_index`: leaving a stale `hash → old_id`
/// entry after a clear would let a later registration's dedup lookup
/// return an id that no longer names the region it was hashed from (the
/// slot at that index now holds whatever the NEXT registration after the
/// clear pushed there). Adopting tests share one process, so callers must
/// also serialize with any other adopting test (dd-shapes coordination,
/// 2026-07-08: the hook alone races). `#[doc(hidden)] pub` rather than
/// `#[cfg(test)]` because adopting tests live in downstream crates, which
/// compile this crate without `cfg(test)`.
#[doc(hidden)]
pub fn clear_runtime_fused_for_tests() {
    RUNTIME_FUSED_OPS.write().unwrap().clear();
    hash_index().write().unwrap().clear();
}

// ---- the region → primitive re-emit (the runtime op's `decompose`) ---------

/// Project a region [`OpTag`] (+ its [`OpAttrs`]) back to a primitive [`Op`].
/// The inverse of `jit::op_to_tag`, over the **v1 re-emit vocabulary**:
/// type-preserving elementwise + scalar-param ops. Returns `None` for ops that
/// need structural params or change dtype (comparisons, `Where`, reductions,
/// `MatMul`, shape/index ops) — those are rejected at registration so this is
/// total for every registered region.
fn tag_to_op(tag: OpTag, attrs: &OpAttrs) -> Option<Op> {
    use OpTag as T;
    Some(match tag {
        T::Add => Op::Add,
        T::Sub => Op::Sub,
        T::Mul => Op::Mul,
        T::Div => Op::Div,
        T::Maximum => Op::Maximum,
        T::Minimum => Op::Minimum,
        T::Pow => Op::Pow,
        T::Rem => Op::Rem,
        T::Neg => Op::Neg,
        T::Abs => Op::Abs,
        T::Sqr => Op::Sqr,
        T::Sqrt => Op::Sqrt,
        T::Rsqrt => Op::Rsqrt,
        T::Recip => Op::Recip,
        T::Exp => Op::Exp,
        T::Log => Op::Log,
        T::Sin => Op::Sin,
        T::Cos => Op::Cos,
        T::Tanh => Op::Tanh,
        T::Sigmoid => Op::Sigmoid,
        T::Silu => Op::Silu,
        T::Gelu => Op::Gelu,
        T::GeluErf => Op::GeluErf,
        T::Relu => Op::Relu,
        T::Erf => Op::Erf,
        T::Step => Op::Step,
        T::Floor => Op::Floor,
        T::Ceil => Op::Ceil,
        T::Round => Op::Round,
        T::Sign => Op::Sign,
        // Scalar-param ops: the value rides `attrs.scalars` (the slot snapshot;
        // live-value substitution via the `extract:` path is a follow-up).
        T::AddScalar => Op::AddScalar(*attrs.scalars.first()?),
        T::MulScalar => Op::MulScalar(*attrs.scalars.first()?),
        _ => return None,
    })
}

/// How many scalar values `tag` consumes from `attrs.scalars` when re-emitted.
/// The slot machinery (extraction, validation dummy-fill, decompose fill) is
/// keyed on this; extend alongside `tag_to_op` when a new scalar-param op joins
/// the v1 vocabulary.
fn scalar_slot_arity(tag: OpTag) -> usize {
    matches!(tag, OpTag::AddScalar | OpTag::MulScalar) as usize
}

/// Count the region's open scalar **slots** in pattern pre-order — scalar-param
/// ops whose `attrs.scalars` is empty (a baked value is a pattern constant, not
/// a slot). This is the length of the `scalars` vec `match_region_extract`
/// returns for a match, and of the `FusedOpParams::Runtime { scalars }` the
/// fused node must carry for [`decompose_region`] to fill the re-emit.
pub fn count_scalar_slots(node: &PatternNode) -> usize {
    match node {
        PatternNode::Op { op, operands, attrs } => {
            let own = if attrs.scalars.is_empty() { scalar_slot_arity(*op) } else { 0 };
            own + operands.iter().map(count_scalar_slots).sum::<usize>()
        }
        _ => 0,
    }
}

fn validate_representable(node: &PatternNode) -> Result<(), RuntimeFusedError> {
    match node {
        PatternNode::Op { op, operands, attrs } => {
            // A scalar-param op with empty scalars is a SLOT template —
            // validate re-emittability with a dummy fill (the live value is
            // substituted from the fused node's `Runtime { scalars }` at
            // decompose time).
            let representable = if attrs.scalars.is_empty() && scalar_slot_arity(*op) > 0 {
                let mut filled = attrs.clone();
                filled.scalars = vec![0.0; scalar_slot_arity(*op)];
                tag_to_op(*op, &filled).is_some()
            } else {
                tag_to_op(*op, attrs).is_some()
            };
            if !representable {
                return Err(RuntimeFusedError::UnRepresentable(*op));
            }
            for o in operands {
                validate_representable(o)?;
            }
            Ok(())
        }
        PatternNode::Bind { .. } => Ok(()),
        PatternNode::Any | PatternNode::SeeThrough { .. } => {
            Err(RuntimeFusedError::NonConcreteRegion)
        }
    }
}

/// Decompose a runtime `Op::Fused(id, Runtime { .. })` node by re-emitting its
/// region as primitives, returning the new root (the re-emitted sink). If `id`
/// is not a registered runtime op the node is returned unchanged (a fixpoint —
/// no recipe, G2). The matched node's inputs are the region's bound external
/// inputs in bind-index order.
pub fn decompose_region(graph: &mut Graph, node_id: NodeId) -> NodeId {
    let (fid, node_scalars) = match &graph.node(node_id).op {
        Op::Fused(id, FusedOpParams::Runtime { scalars }) => (*id, scalars.clone()),
        Op::Fused(id, _) => (*id, Vec::new()),
        _ => return node_id,
    };
    let region = match runtime_region(fid) {
        Some(r) => r,
        None => return node_id,
    };
    // The node's live scalars must fill the region's slots exactly (pattern
    // pre-order, the same canon `match_region_extract` produced them in). A
    // mismatch is a malformed fused node — surfaced as a no-op fixpoint (the
    // lowering driver records no progress), never a crash (G2).
    if node_scalars.len() != count_scalar_slots(&region) {
        return node_id;
    }
    let inputs = graph.node(node_id).inputs.clone();
    let mut cursor = node_scalars.as_slice();
    emit(graph, &region, &inputs, &mut cursor)
}

/// Re-emit a validated region on the given external input nodes (public entry
/// for callers holding a raw [`PatternNode`] + input [`NodeId`]s — e.g. the
/// reference realization during candidate-kernel verification, which has a raw
/// region and freshly-pushed `Op::Const` input nodes rather than a Fused node
/// already in the graph). `scalars` fill the region's open scalar slots in
/// pre-order (the canonical order `match_region_extract` recorded them in);
/// pass `&[]` for a parameterless region. Thin wrapper over the private
/// [`emit`]; the same re-emittability caveat applies (a non-re-emittable
/// `OpTag` panics inside `emit` — validated decomposes never carry one).
/// Second panic risk: `emit`'s scalar-cursor fill (`scalars.split_at(arity)`)
/// panics if `scalars` is shorter than the region's total open-slot count.
/// [`decompose_region`] (the node-driven caller) guards this with its own
/// length check before ever calling `emit`; `emit_region` deliberately does
/// NOT — it's a thin wrapper, so validating the length is the caller's job.
/// Callers must pass a `scalars` slice at least as long as the region's
/// open-slot count.
pub fn emit_region(
    graph: &mut Graph,
    region: &PatternNode,
    inputs: &[NodeId],
    scalars: &[f64],
) -> NodeId {
    let mut cursor = scalars;
    emit(graph, region, inputs, &mut cursor)
}

fn emit(
    graph: &mut Graph,
    node: &PatternNode,
    inputs: &[NodeId],
    scalars: &mut &[f64],
) -> NodeId {
    match node {
        PatternNode::Bind { index } => inputs[*index as usize],
        PatternNode::Op { op, operands, attrs } => {
            // Fill an open scalar slot from the cursor in PRE-order (before
            // descending into operands) — the same canonical order
            // `match_region_extract` recorded the live values in.
            let arity = scalar_slot_arity(*op);
            let filled;
            let attrs = if attrs.scalars.is_empty() && arity > 0 {
                let (take, rest) = scalars.split_at(arity);
                *scalars = rest;
                filled = OpAttrs { scalars: take.to_vec(), ..attrs.clone() };
                &filled
            } else {
                attrs
            };
            let prim = tag_to_op(*op, attrs).expect("region validated re-emittable at registration");
            let child_ids: Vec<NodeId> =
                operands.iter().map(|o| emit(graph, o, inputs, scalars)).collect();
            // v1 same-shape elementwise: a node's shape/dtype = its first
            // operand's (these ops are type-preserving + shape-preserving).
            let s = graph.node(child_ids[0]).shape.clone();
            let d = graph.node(child_ids[0]).dtype;
            graph.push(Node { op: prim, inputs: child_ids, shape: s, dtype: d })
        }
        PatternNode::Any | PatternNode::SeeThrough { .. } => {
            unreachable!("region validated concrete (Op/Bind) at registration")
        }
    }
}

/// A [`crate::opt::LoweringRule`]-shaped `decompose` for runtime ops: re-emit
/// the region. The scalar `extract:` substitution rides on the NODE (its
/// `FusedOpParams::Runtime { scalars }` fills the region's open slots inside
/// [`decompose_region`]), so the rule-shaped `params` argument stays unused.
pub fn runtime_lowering_decompose(
    graph: &mut Graph,
    node_id: NodeId,
    _params: &FusedOpParams,
) -> NodeId {
    decompose_region(graph, node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{DType, Shape};

    fn relu_add_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    /// Structurally DISTINCT from `relu_add_region()` (`Mul` inner op, not
    /// `Add`) — used only by
    /// `register_allocates_a_runtime_id_and_keeps_the_region`, whose
    /// assertion on `runtime_name` needs a region no OTHER test in this
    /// module also registers. Since Task 7's dedup (`register_runtime_fused`
    /// above) resolves any two structurally-identical regions registered
    /// anywhere in the process to the SAME id — and `RUNTIME_FUSED_OPS` /
    /// `hash_index` are process-global statics shared by every `#[test]` in
    /// this binary, which `cargo test` runs concurrently by default — a
    /// `runtime_name` assertion tied to one specific registration call would
    /// be racy against any other test using `relu_add_region()` (both
    /// `decompose_region_re_emits_relu_add` and
    /// `register_runtime_fused_dedups_structurally_identical_regions` do):
    /// whichever call reaches the shared hash slot FIRST wins the name, and
    /// thread scheduling decides which that is. Those other two tests never
    /// assert on `runtime_name`, so they're unaffected by dedup either way;
    /// this one needs its own hash to stay deterministic.
    fn relu_mul_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Mul,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    #[test]
    fn register_allocates_a_runtime_id_and_keeps_the_region() {
        let id = register_runtime_fused("test::relu_mul", relu_mul_region()).unwrap();
        assert!(id.is_runtime(), "allocated id is in the runtime range");
        assert_eq!(runtime_region(id), Some(relu_mul_region()));
        assert_eq!(runtime_name(id).as_deref(), Some("test::relu_mul"));
    }

    #[test]
    fn register_runtime_fused_dedups_structurally_identical_regions() {
        let id1 = register_runtime_fused("dedup::a", relu_add_region()).unwrap();
        let id2 = register_runtime_fused("dedup::b", relu_add_region()).unwrap(); // same region, different name
        assert_eq!(id1, id2, "an identical region must resolve to the same FusedOpId, not a duplicate");
    }

    #[test]
    fn register_rejects_non_contiguous_binds() {
        // bind indices {0, 2} — missing 1.
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 2 }],
        };
        assert_eq!(
            register_runtime_fused("bad", region),
            Err(RuntimeFusedError::NonContiguousBinds(vec![0, 2]))
        );
    }

    #[test]
    fn register_rejects_unrepresentable_region() {
        // MatMul has no v1 primitive re-emission.
        let region = PatternNode::Op {
            op: OpTag::MatMul,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        assert_eq!(
            register_runtime_fused("bad", region),
            Err(RuntimeFusedError::UnRepresentable(OpTag::MatMul))
        );
    }

    #[test]
    fn decompose_region_re_emits_relu_add() {
        let id = register_runtime_fused("test::relu_add::decompose", relu_add_region()).unwrap();
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let fused = g.push(Node {
            op: Op::Fused(id, FusedOpParams::Runtime { scalars: vec![] }),
            inputs: vec![a, b],
            shape: s.clone(),
            dtype: DType::F32,
        });

        let root = decompose_region(&mut g, fused);

        // The re-emitted sink is Relu over Add(a, b) — the region, on primitives.
        assert!(matches!(g.node(root).op, Op::Relu));
        let add_id = g.node(root).inputs[0];
        assert!(matches!(g.node(add_id).op, Op::Add));
        assert_eq!(g.node(add_id).inputs, vec![a, b]);
        // Shapes propagated from the leaves (same-shape elementwise).
        assert_eq!(g.node(root).shape, s);
        assert_eq!(g.node(add_id).shape, s);
    }

    // ---- scalar slots (the `extract:` substitution) ---------------------

    /// tanh(mul_scalar(a)) with the scalar left OPEN (a slot template).
    fn tanh_mul_scalar_slot_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Tanh,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::MulScalar,
                attrs: OpAttrs::default(), // empty scalars = an open slot
                operands: vec![PatternNode::Bind { index: 0 }],
            }],
        }
    }

    #[test]
    fn slot_template_registers_and_counts() {
        // Born-red before slot support: validation rejected an AddScalar/
        // MulScalar pattern node with no baked value.
        let id = register_runtime_fused(
            "test::tanh_mul_scalar::slot",
            tanh_mul_scalar_slot_region(),
        )
        .expect("a slot template is registrable");
        let region = runtime_region(id).unwrap();
        assert_eq!(count_scalar_slots(&region), 1, "one open slot");
    }

    #[test]
    fn decompose_fills_slots_from_the_node_scalars() {
        let id = register_runtime_fused(
            "test::tanh_mul_scalar::fill",
            tanh_mul_scalar_slot_region(),
        )
        .unwrap();
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let fused = g.push(Node {
            op: Op::Fused(id, FusedOpParams::Runtime { scalars: vec![2.5] }),
            inputs: vec![a],
            shape: s.clone(),
            dtype: DType::F32,
        });

        let root = decompose_region(&mut g, fused);

        // tanh(mul_scalar(a, 2.5)) — the LIVE value filled the slot.
        assert!(matches!(g.node(root).op, Op::Tanh));
        let ms = g.node(root).inputs[0];
        assert!(
            matches!(g.node(ms).op, Op::MulScalar(v) if v == 2.5),
            "slot filled with the node's live scalar, got {:?}",
            g.node(ms).op,
        );
        assert_eq!(g.node(ms).inputs, vec![a]);
    }

    #[test]
    fn decompose_slot_count_mismatch_is_a_fixpoint_not_a_crash() {
        let id = register_runtime_fused(
            "test::tanh_mul_scalar::mismatch",
            tanh_mul_scalar_slot_region(),
        )
        .unwrap();
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        // One slot, but the node carries NO scalars — malformed; must be a
        // no-op fixpoint (G2), never a panic.
        let fused = g.push(Node {
            op: Op::Fused(id, FusedOpParams::Runtime { scalars: vec![] }),
            inputs: vec![a],
            shape: s.clone(),
            dtype: DType::F32,
        });
        assert_eq!(decompose_region(&mut g, fused), fused, "mismatch ⇒ fixpoint");
    }
}

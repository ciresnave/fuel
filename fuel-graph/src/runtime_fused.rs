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

use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode, matmul_roles};

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
    /// A shape-relative attr (D2) that can never resolve at ANY shape — a
    /// STRUCTURAL authoring error caught at registration: a rel field and its
    /// concrete sibling both set, a bind reference outside the region's bind
    /// space, `axis_last` on an axis-less tag, or a `Param` reference (no
    /// param threading until C-4). Value-dependent declines (a `Negative` or
    /// symbolic-extent result at some particular shape) do NOT reject
    /// registration — they surface at emit time as a decompose fixpoint (G2).
    InvalidRelAttrs { tag: OpTag, error: RelAttrError },
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
/// poisoned lock, a rel-attr resolution decline at the placeholder shapes,
/// or an empty lowering result) — the caller
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
        // Fallible entry: a rel-attr region that declines at the rank-1 `[1]`
        // placeholder shapes yields `None` — "hash unavailable", dedup skipped
        // for this registration (allocate-fresh), never a panic.
        try_emit_region(&mut g, region, &inputs, &scalars).ok()?
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
    validate_recipe(&region)?;

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
/// The inverse of `jit::op_to_tag`, over the **full first-order re-emit
/// vocabulary** (Convergence Increment A): every non-basis-gap, non-`Scan`,
/// non-`Fused` op — elementwise, comparison, `Where`, `Cast`, shape/layout
/// (Transpose/Permute/Reshape/BroadcastTo/(Un)squeeze/Slice/Concat/Flip/Roll/
/// Pad/Triu/Tril), reductions (SumDim/MaxDim/MeanDim/ReduceSumTo/ReduceMaxTo/
/// CumSum/SumAll/MaxAll/MinAll/MeanAll), `MatMul`, `Iota`, and indexing (IndexSelect/
/// Gather/IndexAdd/ScatterAdd). Structural params are decoded from the
/// (extended) [`OpAttrs`]. Returns `None` (an honest miss, rejected at
/// registration) for ops with no first-order re-emission: `PowI`/`Clamp`
/// (no i32/two-scalar carrier), `MaskedFill` (no `Scalar::from_f64`
/// reconstructor yet), fused/basis-gap tags, and any tag whose required attrs
/// are unset (e.g. `Iota` with no `target_shape`).
fn tag_to_op(tag: OpTag, attrs: &OpAttrs) -> Option<Op> {
    use OpTag as T;
    use fuel_ir::{DType, Shape};
    use std::str::FromStr;
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

        // --- Convergence Increment A: the full first-order set ---
        // Comparison (dtype→U8 handled by primitive_shape, not here).
        T::Equal => Op::Equal,
        T::Ne => Op::Ne,
        T::Lt => Op::Lt,
        T::Le => Op::Le,
        T::Gt => Op::Gt,
        T::Ge => Op::Ge,
        // Ternary select.
        T::Where => Op::Where,
        // Dtype-changing: target dtype rides `cast_dtype` (the stable name).
        T::Cast => Op::Cast(DType::from_str(attrs.cast_dtype.as_deref()?).ok()?),
        // MatMul: the LOCKED role-vector contraction cell (§5/D5). Empty roles
        // = the rank-polymorphic recipe form → implicit-accept (unchanged from
        // today; recipes keep matmul implicit). Explicit roles must match the
        // canonical cell EXACTLY — same-rank ≥ 2, leading Batch, lhs=[..,FreeM,
        // ContractedK], rhs=[..,ContractedK,FreeN] — checked by role POSITION,
        // not extent (so GQA-divisible batch stays all-Batch). Any other config
        // (transposed / multi-ContractedK / FreeN-before-K / rank mismatch) is a
        // SURFACED honest miss (`None`, rejected at registration), never a crash.
        T::MatMul => {
            if attrs.lhs_roles.is_empty() && attrs.rhs_roles.is_empty() {
                Op::MatMul
            } else {
                let (canon_lhs, canon_rhs) = matmul_roles(attrs.lhs_roles.len(), attrs.rhs_roles.len());
                if attrs.lhs_roles.len() == attrs.rhs_roles.len()
                    && attrs.lhs_roles.len() >= 2
                    && attrs.lhs_roles == canon_lhs
                    && attrs.rhs_roles == canon_rhs
                {
                    Op::MatMul
                } else {
                    return None;
                }
            }
        }
        T::LogSoftmaxLastDim => Op::LogSoftmaxLastDim,
        // Shape / layout.
        T::Transpose => Op::Transpose,
        T::Permute => Op::Permute(attrs.perm.iter().map(|&x| x as usize).collect()),
        T::Reshape => Op::Reshape(shape_from_attr(attrs)?),
        T::BroadcastTo => Op::BroadcastTo(shape_from_attr(attrs)?),
        T::ReduceSumTo => Op::ReduceSumTo(shape_from_attr(attrs)?),
        T::ReduceMaxTo => Op::ReduceMaxTo(shape_from_attr(attrs)?),
        T::Unsqueeze => Op::Unsqueeze { dim: *attrs.dims.first()? as usize },
        T::Squeeze => Op::Squeeze { dim: *attrs.dims.first()? as usize },
        T::Slice => Op::Slice {
            dim: attrs.axis? as usize,
            start: attrs.slice_start? as usize,
            len: attrs.slice_len? as usize,
        },
        T::Concat => Op::Concat { dim: attrs.axis? as usize },
        T::Flip => Op::Flip { dim: attrs.axis? as usize },
        T::Roll => Op::Roll { dim: attrs.axis? as usize, shift: attrs.roll_shift? },
        T::Pad => Op::Pad {
            padding: attrs.pad_amounts.iter().map(|&(b, e)| (b as usize, e as usize)).collect(),
            mode: match attrs.pad_mode? {
                0 => crate::PadMode::Constant,
                1 => crate::PadMode::Reflect,
                2 => crate::PadMode::Replicate,
                _ => return None,
            },
            value: attrs.pad_value.unwrap_or(0.0),
        },
        T::Triu => Op::Triu { diagonal: attrs.axis? },
        T::Tril => Op::Tril { diagonal: attrs.axis? },
        // Reductions (dim rides `axis`; keepdim reductions ride `target_shape`).
        T::SumDim => Op::SumDim(attrs.axis? as usize),
        T::MaxDim => Op::MaxDim(attrs.axis? as usize),
        T::MeanDim => Op::MeanDim(attrs.axis? as usize),
        T::SumAll => Op::SumAll,
        T::MaxAll => Op::MaxAll,
        T::MinAll => Op::MinAll,
        T::MeanAll => Op::MeanAll,
        T::CumSum => Op::CumSum { dim: attrs.axis? as usize },
        // Value source leaf (len rides `target_shape` as a 1-element shape).
        T::Iota => Op::Iota { len: *attrs.target_shape.first()? as usize },
        // Indexing (dim rides `axis`).
        T::IndexSelect => Op::IndexSelect { dim: attrs.axis? as usize },
        T::Gather => Op::Gather { dim: attrs.axis? as usize },
        T::IndexAdd => Op::IndexAdd { dim: attrs.axis? as usize },
        T::ScatterAdd => Op::ScatterAdd { dim: attrs.axis? as usize },

        // Honest misses (rejected at registration): PowI/Clamp (no carrier),
        // MaskedFill (no Scalar::from_f64 reconstructor yet), and any tag whose
        // required attrs are unset or that is added to OpTag later.
        _ => return None,
    })
}

/// Decode a target [`fuel_ir::Shape`] from `attrs.target_shape` (the shared
/// LOGICAL-shape carrier for Reshape/BroadcastTo/ReduceSumTo/ReduceMaxTo).
/// `None` for an unset (empty) target — an honest miss, not a rank-0 shape.
fn shape_from_attr(attrs: &OpAttrs) -> Option<fuel_ir::Shape> {
    if attrs.target_shape.is_empty() {
        return None;
    }
    let dims: Vec<usize> = attrs.target_shape.iter().map(|&d| d as usize).collect();
    Some(fuel_ir::Shape::from_dims(&dims))
}

/// How many scalar values `tag` consumes from `attrs.scalars` when re-emitted.
/// The slot machinery (extraction, validation dummy-fill, decompose fill) is
/// keyed on this; extend alongside `tag_to_op` when a new scalar-param op joins
/// the v1 vocabulary.
fn scalar_slot_arity(tag: OpTag) -> usize {
    matches!(tag, OpTag::AddScalar | OpTag::MulScalar) as usize
}

// ---- shape-relative attr resolution (Increment C slice 1, T2/D2) -----------

/// A shape-relative attr resolution failure — a typed decline, never a panic.
/// The emit-integration caller (T3) surfaces any of these as a decompose
/// fixpoint (`return id`, G2); the registration-validation caller rejects the
/// region. The `field` names the CONCRETE sibling the rel field resolves into
/// (`"target_shape"`, `"slice_start"`, `"slice_len"`, `"axis"`, `"dims"`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelAttrError {
    /// A rel field and its concrete sibling are BOTH set — ambiguous authoring
    /// (rel XOR abs per field), refused rather than given a silent precedence.
    RelAbsConflict { field: &'static str },
    /// The underlying shape-expression evaluation declined (bind out of range,
    /// axis out of range, divide-by-zero, `Param` with no param threading, …).
    Expr(fuel_kernel_seam_types::shape_expr::ShapeExprError),
    /// The expression evaluated over a SYMBOLIC bind extent → a surfaced gap
    /// (§6.20-0004): a rel attr cannot resolve concrete at emit time.
    SymbolicGap { field: &'static str },
    /// The expression produced a negative value where a non-negative
    /// extent/offset is required.
    Negative { field: &'static str, value: i64 },
    /// `axis_last` on a tag with no axis carrier (e.g. `Add`) — meaningless,
    /// refused (build-time validation, never silently ignored).
    AxisLastUnsupported { tag: OpTag },
    /// `axis_last` with no child operand — no rank to resolve LAST against.
    NoChildOperand,
    /// The region's Bind-space broadcast **frame** is assembled by per-axis max
    /// across ≥2 binds (`a[N,1] ⊗ b[1,M] → [N,M]`), so NO single operand carries
    /// it — and `SameAs { operand }`, the §6.20 EXPRESSION kind's only
    /// whole-shape constructor, therefore cannot express it. Accepting the
    /// `SameAs` would SILENTLY resolve a `BroadcastTo` target to a PARTIAL frame
    /// (`[N,1]` or `[1,M]`) and emit the wrong graph, so it is refused instead
    /// (Baracuda's §6.20 finding). A **Dims-class gap**: `missing_ctor` is the
    /// reserved wire tag of the constructor that WOULD express it
    /// ([`fuel_kernel_seam_types::shape_expr::TAG_DIMS`] = `0x0B`, a §6.20-0002
    /// extension-registry entrant — proposal filed KISS #80). `frame` is the
    /// computed per-axis-max frame, for telemetry.
    FrameNotExpressible { field: &'static str, frame: Vec<i64>, missing_ctor: u8 },
}

/// Whether `attrs` carries any shape-RELATIVE field (D2) — the emit fast-path
/// guard: rel-free attrs skip resolution entirely (zero behavior change for
/// existing concrete regions).
fn has_rel_attrs(attrs: &OpAttrs) -> bool {
    attrs.target_shape_rel.is_some()
        || attrs.slice_start_rel.is_some()
        || attrs.slice_len_rel.is_some()
        || attrs.axis_last
}

/// The ONE rel-XOR-abs mutual-exclusion oracle (shared by
/// [`resolve_rel_attrs`] and the registration rel-probe — no second copy to
/// drift). Returns the first conflicted field name in canonical field order
/// (`target_shape`, `slice_start`, `slice_len`, then the `axis_last` carrier —
/// `dims` for Squeeze/Unsqueeze, `axis` otherwise), or `None`. Note the
/// `axis_last` arm reports a carrier conflict even for a tag the resolver
/// would refuse as [`RelAttrError::AxisLastUnsupported`] — both are typed
/// authoring declines, and both-set is checked first.
fn rel_abs_conflict_field(tag: OpTag, attrs: &OpAttrs) -> Option<&'static str> {
    if attrs.target_shape_rel.is_some() && !attrs.target_shape.is_empty() {
        return Some("target_shape");
    }
    if attrs.slice_start_rel.is_some() && attrs.slice_start.is_some() {
        return Some("slice_start");
    }
    if attrs.slice_len_rel.is_some() && attrs.slice_len.is_some() {
        return Some("slice_len");
    }
    if attrs.axis_last {
        match tag {
            OpTag::Unsqueeze | OpTag::Squeeze => {
                if !attrs.dims.is_empty() {
                    return Some("dims");
                }
            }
            _ => {
                if attrs.axis.is_some() {
                    return Some("axis");
                }
            }
        }
    }
    None
}

/// The region's Bind-space broadcast **frame**: the NumPy right-aligned
/// per-axis max over EVERY bind shape (`[N,1] ⊗ [1,M] → [N,M]`) — the shape an
/// elementwise consumer of all the binds produces. `None` when the binds carry
/// no joint elementwise frame at all: no binds, a SYMBOLIC extent (the frame is
/// itself a gap — the `SameAs` arm already declines those as
/// [`RelAttrError::SymbolicGap`]), or mutually broadcast-INcompatible binds (a
/// matmul/gather region, where per-axis max is meaningless). Pure and total —
/// unlike the graph-builder `compute_broadcast_shape`, incompatibility is
/// `None`, never a panic.
fn bind_broadcast_frame(bind_shapes: &[Vec<i64>]) -> Option<Vec<i64>> {
    use fuel_kernel_seam_types::shape_expr::SYMBOLIC;
    let rank = bind_shapes.iter().map(Vec::len).max()?;
    let mut frame = vec![1i64; rank];
    for s in bind_shapes {
        if s.iter().any(|&e| e == SYMBOLIC || e < 0) {
            return None;
        }
        let pad = rank - s.len(); // right-aligned: pad the shorter with leading 1s
        for (i, &e) in s.iter().enumerate() {
            let f = &mut frame[pad + i];
            if *f == e || e == 1 {
                continue;
            }
            if *f == 1 {
                *f = e;
                continue;
            }
            return None; // incompatible at this axis ⇒ no joint frame
        }
    }
    Some(frame)
}

/// The SameAs **degradation guard** (I1, Baracuda's §6.20 finding). Even an
/// ELEMENTWISE output shape is not always expressible as `SameAs(operand)`:
/// when the region's broadcast frame is assembled by per-axis max across TWO
/// binds (`a[N,1] ⊗ b[1,M] → [N,M]`) no single operand carries the full frame,
/// so every `SameAs` spelling resolves to a PARTIAL frame. `BroadcastTo` is the
/// recipe's frame carrier (its target IS the elementwise output shape), so a
/// `SameAs` target is accepted only when SOME bind does carry the whole frame;
/// otherwise the frame is surfaced as a typed Dims-class gap rather than
/// silently resolved to one operand's partial shape.
///
/// Deliberately narrow — it fires ONLY when a joint frame exists and NO bind
/// equals it. Sub-frame broadcasts (bind1 `[T,D]` inside a `[B,T,D]` region)
/// and frame-less regions (matmul binds) are untouched, and all 5 slice-1
/// migrated recipes are safe by construction (their `SameAs` target is bind 0,
/// which carries the frame).
fn same_as_frame_guard(
    tag: OpTag,
    se: &fuel_kernel_seam_types::shape_expr::ShapeExpr,
    bind_shapes: &[Vec<i64>],
) -> Result<(), RelAttrError> {
    use fuel_kernel_seam_types::shape_expr::{ShapeExpr, TAG_DIMS};
    // Exhaustive on purpose: the Dims/WithDim extension entrants (KISS #80)
    // express the max-frame DIRECTLY (a whole-shape ctor), so they must NOT be
    // routed through this partial-`SameAs`-frame guard — they return early.
    match se {
        ShapeExpr::SameAs { .. } => {}
        ShapeExpr::WithDim { .. } | ShapeExpr::Dims(_) => return Ok(()),
    }
    if tag != OpTag::BroadcastTo {
        return Ok(());
    }
    let Some(frame) = bind_broadcast_frame(bind_shapes) else {
        return Ok(()); // no joint elementwise frame at play
    };
    if bind_shapes.iter().any(|b| *b == frame) {
        return Ok(()); // some operand carries the whole frame ⇒ expressible
    }
    Err(RelAttrError::FrameNotExpressible {
        field: "target_shape",
        frame,
        missing_ctor: TAG_DIMS,
    })
}

/// Resolve `attrs`' shape-RELATIVE fields (`target_shape_rel`,
/// `slice_start_rel`/`slice_len_rel`, `axis_last` — D2) into their concrete
/// siblings, returning a fully-concrete [`OpAttrs`] ready for the unchanged
/// `tag_to_op` → `primitive_shape` path. Pure: no graph access.
///
/// * `bind_shapes` — the region's **Bind-space** shapes, `bind_shapes[i]` =
///   `Bind { index: i }`'s shape. This is what `ShapeExpr::SameAs { operand }`
///   and `Dim::Extent { operand, .. }` index (the recipe-interior reference
///   convention, same as the merged KISS shape-oracle RFC's contract roles).
/// * `child_shapes` — THIS op's direct operand shapes (the already-emitted
///   children), which `axis_last` resolves its rank against — a region
///   interior node's shape generally matches NO bind.
///
/// Evaluation reuses `shape_expr::eval_dim`/`eval_shape`/`resolve_axis` — the
/// single §6.20 evaluator, no second one. `Dim::Param` declines with a typed
/// [`ShapeExprError::ParamOutOfRange`] until param threading lands (C-4);
/// symbolic bind extents decline as [`RelAttrError::SymbolicGap`]. Rel fields
/// are CLEARED in the output (rel+abs both set in the RESULT would trip the
/// mutual-exclusion check on a second resolve).
pub fn resolve_rel_attrs(
    tag: OpTag,
    attrs: &OpAttrs,
    bind_shapes: &[Vec<i64>],
    child_shapes: &[Vec<i64>],
) -> Result<OpAttrs, RelAttrError> {
    use fuel_kernel_seam_types::shape_expr::{
        self, Dim, DimValue, LAST, ShapeValue, resolve_axis,
    };
    // Mutual exclusion FIRST, for every field, before any evaluation — so a
    // value-dependent decline in an earlier field can't mask a rel+abs
    // authoring conflict in a later one (the registration probe relies on
    // this completeness).
    if let Some(field) = rel_abs_conflict_field(tag, attrs) {
        return Err(RelAttrError::RelAbsConflict { field });
    }
    let mut out = attrs.clone();

    // target_shape_rel → target_shape (SameAs over the Bind space).
    if let Some(se) = &attrs.target_shape_rel {
        match shape_expr::eval_shape(se, bind_shapes, &[]).map_err(RelAttrError::Expr)? {
            ShapeValue::Concrete(s) => {
                if let Some(&bad) = s.iter().find(|&&e| e < 0) {
                    return Err(RelAttrError::Negative { field: "target_shape", value: bad });
                }
                // I1: refuse a `SameAs` target whose region frame no operand
                // carries — a silent PARTIAL frame otherwise (see
                // [`same_as_frame_guard`]). Runs AFTER evaluation so the
                // structural declines (`OperandOutOfRange`, `SymbolicGap`)
                // keep their existing precedence.
                same_as_frame_guard(tag, se, bind_shapes)?;
                out.target_shape = s;
            }
            ShapeValue::Gap => {
                return Err(RelAttrError::SymbolicGap { field: "target_shape" });
            }
        }
        out.target_shape_rel = None;
    }

    // slice_{start,len}_rel → slice_{start,len} (DimExpr over the Bind space).
    let eval_dim_field = |d: &Dim, field: &'static str| -> Result<u64, RelAttrError> {
        match shape_expr::eval_dim(d, bind_shapes, &[]).map_err(RelAttrError::Expr)? {
            DimValue::Concrete(v) if v < 0 => Err(RelAttrError::Negative { field, value: v }),
            DimValue::Concrete(v) => Ok(v as u64),
            DimValue::Gap => Err(RelAttrError::SymbolicGap { field }),
        }
    };
    if let Some(d) = &attrs.slice_start_rel {
        out.slice_start = Some(eval_dim_field(d, "slice_start")?);
        out.slice_start_rel = None;
    }
    if let Some(d) = &attrs.slice_len_rel {
        out.slice_len = Some(eval_dim_field(d, "slice_len")?);
        out.slice_len_rel = None;
    }

    // axis_last → the per-tag axis carrier, resolved against operand[0]'s rank.
    if attrs.axis_last {
        let rank = child_shapes.first().ok_or(RelAttrError::NoChildOperand)?.len();
        use OpTag as T;
        match tag {
            // `axis`-carrier tags: this op's LAST = rank − 1 via the shared
            // §6.20 resolver (typed AxisOutOfRange on a rank-0 operand).
            T::SumDim | T::MaxDim | T::MeanDim | T::CumSum | T::Concat | T::Flip | T::Slice
            | T::Roll | T::IndexSelect | T::Gather | T::IndexAdd | T::ScatterAdd => {
                let a = resolve_axis(LAST, rank).map_err(RelAttrError::Expr)?;
                out.axis = Some(a as i64);
            }
            // `dims`-carrier: Unsqueeze APPENDS — dim == rank (`primitive_shape`
            // permits `dim == rank`; keepdim-restore spelling, D3).
            T::Unsqueeze => {
                out.dims = vec![rank as u8];
            }
            // `dims`-carrier: Squeeze drops the trailing axis = rank − 1.
            T::Squeeze => {
                let a = resolve_axis(LAST, rank).map_err(RelAttrError::Expr)?;
                out.dims = vec![a as u8];
            }
            other => return Err(RelAttrError::AxisLastUnsupported { tag: other }),
        }
        out.axis_last = false;
    }

    Ok(out)
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

/// The ONE recipe-validation oracle, shared by [`register_runtime_fused`]
/// (runtime Tier-2 registration) and the static-registry
/// [`crate::registry::decompose_via_recipe`] bridge (T5): bind indices form a
/// contiguous `[0, n)` AND every op re-emits to primitives (totality — incl.
/// the rel-attr probe). A recipe carrying a semantics-absent op token (no
/// primitive re-emission — the flip-withdrawal posture: unknown/non-registry
/// tokens are surfaced honest-miss declines, never accepted, never a crash)
/// is a typed [`RuntimeFusedError::UnRepresentable`] decline here.
pub(crate) fn validate_recipe(region: &PatternNode) -> Result<(), RuntimeFusedError> {
    let binds = region.bind_indices();
    let n = binds.len() as u8;
    if binds != (0..n).collect::<Vec<_>>() {
        return Err(RuntimeFusedError::NonContiguousBinds(binds));
    }
    validate_representable(region)
}

fn validate_representable(region: &PatternNode) -> Result<(), RuntimeFusedError> {
    let n_binds = region.bind_indices().len();
    validate_node(region, n_binds)
}

fn validate_node(node: &PatternNode, n_binds: usize) -> Result<(), RuntimeFusedError> {
    match node {
        PatternNode::Op { op, operands, attrs } => {
            // A rel-attr op is a SHAPE-POLYMORPHIC template — probe-resolve it
            // (T3, mirror of the scalar slot dummy-fill below) so the
            // `tag_to_op` representability check can run on concrete attrs.
            // Structural authoring errors reject the region with a typed
            // decline; value-dependent declines at the probe shape register
            // fine and surface at emit time as a decompose fixpoint.
            let probed;
            let attrs = if has_rel_attrs(attrs) {
                probed = rel_probe(*op, attrs, n_binds)
                    .map_err(|error| RuntimeFusedError::InvalidRelAttrs { tag: *op, error })?;
                &probed
            } else {
                attrs
            };
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
                validate_node(o, n_binds)?;
            }
            Ok(())
        }
        PatternNode::Bind { .. } => Ok(()),
        PatternNode::Any | PatternNode::SeeThrough { .. } => {
            Err(RuntimeFusedError::NonConcreteRegion)
        }
    }
}

/// The registration-time rel-attr probe: resolve `attrs` against a fixed
/// `[2, 4]` probe shape (every bind + the child) through the ONE resolver.
/// * `Ok(resolved)` — fully-concrete attrs for the `tag_to_op` probe.
/// * `Err` — a STRUCTURAL authoring error that can never resolve at ANY
///   shape: rel+abs both set, a bind/`Param` reference out of range,
///   `axis_last` on an axis-less tag.
/// * A VALUE-dependent decline at the probe shape (`Negative`,
///   `AxisOutOfRange` against the probe rank, `DivideByZero` through a
///   derived extent, …) is NOT an authoring error — the attrs get a dummy
///   concrete fill instead (the emit-time resolver is the real gate; its
///   decline there is a G2 fixpoint).
fn rel_probe(tag: OpTag, attrs: &OpAttrs, n_binds: usize) -> Result<OpAttrs, RelAttrError> {
    use fuel_kernel_seam_types::shape_expr::ShapeExprError as E;
    let probe: Vec<i64> = vec![2, 4];
    let bind_shapes = vec![probe.clone(); n_binds];
    match resolve_rel_attrs(tag, attrs, &bind_shapes, std::slice::from_ref(&probe)) {
        Ok(resolved) => Ok(resolved),
        Err(
            e @ (RelAttrError::RelAbsConflict { .. }
            | RelAttrError::AxisLastUnsupported { .. }
            | RelAttrError::Expr(E::OperandOutOfRange { .. } | E::ParamOutOfRange { .. })),
        ) => Err(e),
        Err(_) => Ok(dummy_fill_rel(tag, attrs)),
    }
}

/// Clear `attrs`' rel fields and dummy-fill their concrete siblings (only
/// where the sibling is unset — a rel+abs conflict never reaches here, the
/// probe rejects it first) so `tag_to_op` representability can be checked.
fn dummy_fill_rel(tag: OpTag, attrs: &OpAttrs) -> OpAttrs {
    let mut out = attrs.clone();
    if out.target_shape_rel.take().is_some() && out.target_shape.is_empty() {
        out.target_shape = vec![1];
    }
    if out.slice_start_rel.take().is_some() && out.slice_start.is_none() {
        out.slice_start = Some(0);
    }
    if out.slice_len_rel.take().is_some() && out.slice_len.is_none() {
        out.slice_len = Some(1);
    }
    if out.axis_last {
        out.axis_last = false;
        match tag {
            OpTag::Unsqueeze | OpTag::Squeeze => {
                if out.dims.is_empty() {
                    out.dims = vec![0];
                }
            }
            _ => {
                if out.axis.is_none() {
                    out.axis = Some(0);
                }
            }
        }
    }
    out
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
    let bind_shapes = bind_operand_shapes(graph, &inputs);
    let mut cursor = node_scalars.as_slice();
    // A shape-relative attr that fails to resolve at THESE input shapes (a
    // symbolic extent, a negative result, …) is a typed decline surfaced as a
    // no-op fixpoint (G2) — same posture as the slot-count mismatch above,
    // never a panic. Any child nodes emitted before the decline stay in the
    // push-only graph as unreferenced dead nodes (inert).
    emit(graph, &region, &inputs, &bind_shapes, &mut cursor, &mut Vec::new()).unwrap_or(node_id)
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
/// open-slot count. Third (T3): a shape-RELATIVE attr (D2) that fails to
/// resolve at these input shapes panics through the wrapper's `expect` —
/// rel-attr callers use [`try_emit_region`], which surfaces it as a typed
/// [`RelAttrError`] instead.
pub fn emit_region(
    graph: &mut Graph,
    region: &PatternNode,
    inputs: &[NodeId],
    scalars: &[f64],
) -> NodeId {
    try_emit_region(graph, region, inputs, scalars).expect(
        "rel-attr resolution failed — emit_region callers pass concrete-attr or \
         shape-compatible pre-validated regions; fallible callers use try_emit_region",
    )
}

/// The FALLIBLE re-emit entry (Increment C slice 1, T3): like [`emit_region`]
/// but surfacing a shape-relative attr resolution failure (D2) as a typed
/// [`RelAttrError`] instead of a panic. This is the resolving entry the
/// registry `decompose_via_recipe` bridge calls (any failure ⇒ `return id`,
/// the G2 fixpoint). Concrete-attr regions can never hit the `Err` arm — for
/// them this is exactly the legacy `emit_region`. The `emit_region` panic
/// caveats (non-re-emittable `OpTag`, short `scalars` slice) apply unchanged.
pub fn try_emit_region(
    graph: &mut Graph,
    region: &PatternNode,
    inputs: &[NodeId],
    scalars: &[f64],
) -> Result<NodeId, RelAttrError> {
    let bind_shapes = bind_operand_shapes(graph, inputs);
    let mut cursor = scalars;
    emit(graph, region, inputs, &bind_shapes, &mut cursor, &mut Vec::new())
}

/// A graph [`fuel_ir::Shape`] as a §6.20 evaluator operand: per-axis extents
/// with a bounded-symbolic (`Extent::Range`) axis mapped to the
/// [`shape_expr::SYMBOLIC`] sentinel — so a rel attr over a symbolic extent
/// declines as [`RelAttrError::SymbolicGap`] (surfaced gap, §6.20-0004)
/// instead of silently resolving against the capacity bound.
fn shape_expr_operand(shape: &fuel_ir::Shape) -> Vec<i64> {
    use fuel_kernel_seam_types::shape_expr::SYMBOLIC;
    (0..shape.rank())
        .map(|a| if shape.extent(a).is_dynamic() { SYMBOLIC } else { shape.dims()[a] as i64 })
        .collect()
}

/// The region's **Bind-space** shapes (`bind_shapes[i]` = `inputs[i]`'s shape)
/// in §6.20 operand form — what `ShapeExpr::SameAs`/`Dim::Extent` index.
fn bind_operand_shapes(graph: &Graph, inputs: &[NodeId]) -> Vec<Vec<i64>> {
    inputs.iter().map(|&id| shape_expr_operand(&graph.node(id).shape)).collect()
}

/// The recursive re-emit core. `memo` is the per-emit-call identity-share
/// table (T5): a REPEATED slot-free subtree — the tree spelling of a DAG
/// recipe's shared interior (e.g. softmax's `e = Exp(..)`, consumed by both
/// the denominator reduce and the final Div) — emits ONCE, so the emitted
/// graph is the DAG, not a duplicated-compute tree. Lookup is by structural
/// equality (`PatternNode: PartialEq`; regions are tiny, a linear scan is
/// fine) and is sound because, within one call, `inputs`/`bind_shapes` are
/// fixed and emission is deterministic — equal subtrees emit equal nodes.
/// Subtrees with OPEN scalar slots are NEVER shared: each occurrence takes
/// its own value(s) from the pre-order cursor. (The flat-DAG node table with
/// real CSE is slice 3; this is only within-call identity-share.)
fn emit<'r>(
    graph: &mut Graph,
    node: &'r PatternNode,
    inputs: &[NodeId],
    bind_shapes: &[Vec<i64>],
    scalars: &mut &[f64],
    memo: &mut Vec<(&'r PatternNode, NodeId)>,
) -> Result<NodeId, RelAttrError> {
    match node {
        PatternNode::Bind { index } => Ok(inputs[*index as usize]),
        PatternNode::Op { op, operands, attrs } => {
            // Identity-share: a slot-free subtree already emitted in THIS call
            // re-uses its node (see the fn doc). Checked before the cursor
            // fill — a slot-free subtree never moves the cursor, so a hit
            // cannot misalign later slots.
            let sharable = count_scalar_slots(node) == 0;
            if sharable {
                if let Some(&(_, id)) = memo.iter().find(|(p, _)| *p == node) {
                    return Ok(id);
                }
            }
            // Fill an open scalar slot from the cursor in PRE-order (before
            // descending into operands) — the same canonical order
            // `match_region_extract` recorded the live values in. (T3 note:
            // children are now EMITTED before the attrs are USED, but the
            // cursor fill stays right here, before the descent — the cursor
            // order is authoring order, not emission order.)
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
            // Children FIRST (T3 reorder): their emitted shapes feed the
            // rel-attr resolver (`axis_last`'s rank, D4's pad decision).
            let mut child_ids = Vec::with_capacity(operands.len());
            for o in operands {
                child_ids.push(emit(graph, o, inputs, bind_shapes, scalars, memo)?);
            }
            let mut child_shapes: Vec<fuel_ir::Shape> =
                child_ids.iter().map(|&c| graph.node(c).shape.clone()).collect();
            let child_dtypes: Vec<fuel_ir::DType> =
                child_ids.iter().map(|&c| graph.node(c).dtype).collect();
            // Shape-RELATIVE attrs (D2) resolve to fully-concrete siblings
            // against the region's Bind space + this op's operand shapes; the
            // unchanged tag_to_op → primitive_shape path then runs on the
            // result. A failure is a typed decline the caller surfaces
            // (`decompose_region` ⇒ fixpoint, `try_emit_region` ⇒ `Err`) —
            // never a panic. Nodes already pushed for the children stay in the
            // graph as unreferenced (dead) nodes: `Graph` is push-only and
            // base-map extraction walks from roots, so they are inert.
            let resolved;
            let attrs = if has_rel_attrs(attrs) {
                let child_ops: Vec<Vec<i64>> =
                    child_shapes.iter().map(shape_expr_operand).collect();
                resolved = resolve_rel_attrs(*op, attrs, bind_shapes, &child_ops)?;
                &resolved
            } else {
                attrs
            };
            let prim = tag_to_op(*op, attrs).expect("region validated re-emittable at registration");
            // D4: a `BroadcastTo` whose target rank EXCEEDS its operand's rank
            // first materializes the legacy `Reshape` pad (1-padded left,
            // right-aligned — byte-identical to `registry::rope`'s hand-built
            // broadcast prep, since `check_broadcast_compatible` is
            // right-aligned). Recipes stay free of rank-dependent nodes while
            // the emitted graph matches the legacy imperative builders.
            // Applied uniformly (rel-resolved AND absolute targets); an
            // equal-rank broadcast is unchanged (no pad).
            if let Op::BroadcastTo(target) = &prim {
                if let Some(cs) = child_shapes.first() {
                    if target.rank() > cs.rank() {
                        let mut padded: Vec<usize> = vec![1; target.rank() - cs.rank()];
                        padded.extend_from_slice(cs.dims());
                        let pad_shape = fuel_ir::Shape::from_dims(&padded);
                        let pad = graph.push(Node {
                            op: Op::Reshape(pad_shape.clone()),
                            inputs: vec![child_ids[0]],
                            shape: pad_shape.clone(),
                            dtype: child_dtypes[0],
                        });
                        child_ids[0] = pad;
                        child_shapes[0] = pad_shape;
                    }
                }
            }
            // Convergence Increment A: full-parity (shape, dtype) via the single
            // source of truth (`primitive_shape`) — correct for shape-changing,
            // reducing, and dtype-changing ops, not just same-shape elementwise.
            // The Err arm is only reachable for a MALFORMED authored region (a
            // registration-validated region's ops all infer). Real never-panic
            // guarantee: emit always returns a node (or a typed rel decline),
            // never panics here. Fall back to operand[0]'s shape/dtype; and
            // because `validate_representable` checks `tag_to_op(op).is_some()`
            // but NOT arity — and `emit_region` is a public raw-region entry
            // (candidate verification) that does not re-validate — a
            // zero-operand op has no operand shape to borrow, so `.first()`
            // (never `[0]`) guards the index and a degenerate rank-0 F32 node
            // is emitted for that malformed leaf.
            let (s, d) = crate::shape::primitive_shape(&prim, &child_shapes, &child_dtypes)
                .unwrap_or_else(|_| {
                    (
                        child_shapes.first().cloned().unwrap_or_else(|| fuel_ir::Shape::from_dims(&[])),
                        child_dtypes.first().copied().unwrap_or(fuel_ir::DType::F32),
                    )
                });
            let out = graph.push(Node { op: prim, inputs: child_ids, shape: s, dtype: d });
            if sharable {
                memo.push((node, out));
            }
            Ok(out)
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
        // Convergence A made MatMul/shape/reduction ops representable; PowI
        // stays an honest miss (no i32-exponent carrier in OpAttrs), so it is
        // the current canonical still-unrepresentable tag.
        let region = PatternNode::Op {
            op: OpTag::PowI,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        assert_eq!(
            register_runtime_fused("bad", region),
            Err(RuntimeFusedError::UnRepresentable(OpTag::PowI))
        );
    }

    #[test]
    fn tag_to_op_reconstructs_shape_changing_ops() {
        use fuel_ir::Shape;
        // Slice{dim:1,start:2,len:3}
        let attrs = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::Slice, &attrs), Some(Op::Slice { dim: 1, start: 2, len: 3 })));
        // Concat{dim:0}
        let attrs = OpAttrs { axis: Some(0), ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::Concat, &attrs), Some(Op::Concat { dim: 0 })));
        // Reshape([6])
        let attrs = OpAttrs { target_shape: vec![6], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::Reshape, &attrs), Some(Op::Reshape(Shape::from_dims(&[6]))));
        // BroadcastTo([2,3])
        let attrs = OpAttrs { target_shape: vec![2, 3], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::BroadcastTo, &attrs), Some(Op::BroadcastTo(Shape::from_dims(&[2, 3]))));
        // ReduceMaxTo([2,1])
        let attrs = OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::ReduceMaxTo, &attrs), Some(Op::ReduceMaxTo(Shape::from_dims(&[2, 1]))));
    }

    #[test]
    fn tag_to_op_reconstructs_reductions_dtype_and_matmul() {
        use fuel_ir::DType;
        assert!(matches!(super::tag_to_op(OpTag::MeanDim, &OpAttrs { axis: Some(1), ..OpAttrs::default() }), Some(Op::MeanDim(1))));
        assert!(matches!(super::tag_to_op(OpTag::MatMul, &OpAttrs::default()), Some(Op::MatMul)));
        // Cast target dtype via name.
        let attrs = OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::Cast, &attrs), Some(Op::Cast(DType::F16)));
        // Comparison.
        assert!(matches!(super::tag_to_op(OpTag::Lt, &OpAttrs::default()), Some(Op::Lt)));
    }

    #[test]
    fn tag_to_op_matmul_resolves_canonical_roles() {
        // T9 (D5): explicit CANONICAL role vectors resolve to Op::MatMul. The
        // resolver checks role POSITIONS against the locked cell, not extents.
        let attrs = OpAttrs { lhs_roles: vec![1, 3], rhs_roles: vec![3, 2], ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::MatMul, &attrs), Some(Op::MatMul)));
        // Rank-4 canonical (leading Batch dims) also resolves — GQA-divisible
        // batch extents stay all-Batch (positions, not extents).
        let attrs4 = OpAttrs { lhs_roles: vec![0, 0, 1, 3], rhs_roles: vec![0, 0, 3, 2], ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::MatMul, &attrs4), Some(Op::MatMul)));
    }

    #[test]
    fn tag_to_op_matmul_empty_roles_implicit_accept() {
        // Empty roles = the rank-polymorphic recipe form → implicit-accept
        // (unchanged from today; recipes keep matmul implicit).
        assert!(matches!(super::tag_to_op(OpTag::MatMul, &OpAttrs::default()), Some(Op::MatMul)));
    }

    #[test]
    fn tag_to_op_matmul_rejects_noncanonical_roles() {
        // Non-canonical role configs are a SURFACED honest miss (typed decline at
        // registration), never a crash.
        // (1) transposed lhs = [ContractedK, FreeM] = [3,1] instead of [1,3].
        let transposed = OpAttrs { lhs_roles: vec![3, 1], rhs_roles: vec![3, 2], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &transposed), None);
        // (2) multi-ContractedK on lhs = [3,3].
        let multi_k = OpAttrs { lhs_roles: vec![3, 3], rhs_roles: vec![3, 2], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &multi_k), None);
        // (3) FreeN-before-K on rhs = [FreeN, ContractedK] = [2,3] instead of [3,2].
        let freen_before_k = OpAttrs { lhs_roles: vec![1, 3], rhs_roles: vec![2, 3], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &freen_before_k), None);
    }

    #[test]
    fn tag_to_op_reconstructs_max_dim() {
        // T4 (Increment C slice 1): OpTag::MaxDim → Op::MaxDim(axis), the
        // axis riding `attrs.axis` exactly like SumDim/MeanDim.
        assert!(matches!(
            super::tag_to_op(OpTag::MaxDim, &OpAttrs { axis: Some(1), ..OpAttrs::default() }),
            Some(Op::MaxDim(1))
        ));
        // An unset axis is an honest miss (typed decline at registration),
        // never a defaulted axis.
        assert_eq!(super::tag_to_op(OpTag::MaxDim, &OpAttrs::default()), None);
        // Not a scalar-param op: zero scalar slots.
        assert_eq!(super::scalar_slot_arity(OpTag::MaxDim), 0);
    }

    #[test]
    fn max_dim_axis_last_resolves_to_rank_minus_one() {
        // D3 consumer: migrated recipes spell keepdim as MaxDim(axis_last) +
        // Unsqueeze(append), so MaxDim must be an `axis`-carrier tag for the
        // rel-attr resolver (rank − 1 via the shared §6.20 LAST resolver).
        let attrs = OpAttrs { axis_last: true, ..OpAttrs::default() };
        let resolved = super::resolve_rel_attrs(
            OpTag::MaxDim,
            &attrs,
            &[vec![2, 3, 4]],
            &[vec![2, 3, 4]],
        )
        .expect("axis_last must resolve on MaxDim, not AxisLastUnsupported");
        assert_eq!(resolved.axis, Some(2), "rank-3 operand → LAST = axis 2");
        assert!(!resolved.axis_last, "rel carrier must be cleared post-resolve");
    }

    #[test]
    fn tag_to_op_still_rejects_basis_gap_and_scan() {
        // qmatmul/conv flow through Op::Fused (no OpTag); Scan is higher-order.
        assert_eq!(super::tag_to_op(OpTag::Iota, &OpAttrs::default()), None, "Iota needs a len (target_shape) — empty attrs is a miss");
    }

    #[test]
    fn validate_representable_now_accepts_a_slice_region() {
        // Region: Concat{0}(Neg(Slice{...}(bind0)), bind0) — the rope rotate-half shape.
        let region = PatternNode::Op {
            op: OpTag::Concat,
            attrs: OpAttrs { axis: Some(0), ..OpAttrs::default() },
            operands: vec![
                PatternNode::Op {
                    op: OpTag::Neg,
                    attrs: OpAttrs::default(),
                    operands: vec![PatternNode::Op {
                        op: OpTag::Slice,
                        attrs: OpAttrs { axis: Some(0), slice_start: Some(0), slice_len: Some(1), ..OpAttrs::default() },
                        operands: vec![PatternNode::Bind { index: 0 }],
                    }],
                },
                PatternNode::Bind { index: 0 },
            ],
        };
        assert!(super::validate_representable(&region).is_ok(), "slice/concat region must now validate");
    }

    #[test]
    fn emit_gets_shape_right_for_a_reduction_region() {
        use fuel_ir::{DType, Shape};
        // Region: ReduceSumTo([2,1])(bind0). Input [2,5] → output [2,1].
        let region = PatternNode::Op {
            op: OpTag::ReduceSumTo,
            attrs: OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2, 5]), dtype: DType::F32 });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::ReduceSumTo(_)));
        assert_eq!(g.node(root).shape, Shape::from_dims(&[2, 1]), "emit must use the reduced shape, not operand[0]");
        assert_eq!(g.node(root).dtype, DType::F32);
    }

    #[test]
    fn emit_gets_dtype_right_for_a_cast_region() {
        use fuel_ir::{DType, Shape};
        // Region: Cast(F16)(bind0). Input F32 → output F16, same shape.
        let region = PatternNode::Op {
            op: OpTag::Cast,
            attrs: OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[3, 3]), dtype: DType::F32 });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::Cast(DType::F16)));
        assert_eq!(g.node(root).dtype, DType::F16, "emit must take Cast's target dtype, not operand[0]'s");
        assert_eq!(g.node(root).shape, Shape::from_dims(&[3, 3]));
    }

    #[test]
    fn emit_zero_operand_representable_region_is_panic_free() {
        // M-1 never-panic hardening: a MALFORMED region — a binary op given ZERO
        // operands. `validate_representable` accepts it (it checks
        // `tag_to_op(op).is_some()`, NOT arity), and `emit_region` is a public
        // raw-region entry (candidate-kernel verification) that does not
        // re-validate. `primitive_shape(Add, [], [])` errs, so the fallback runs
        // with an EMPTY child_shapes — it must NOT index-panic. emit stays total:
        // it returns a node (with a degenerate rank-0 shape), never a panic.
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![],
        };
        let mut g = Graph::new();
        let root = emit_region(&mut g, &region, &[], &[]);
        assert!(matches!(g.node(root).op, Op::Add), "emit returns a node, not a panic");
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

    // ---- Task 5: byte-for-byte emit == registry::*::decompose parity --------
    //
    // The A.4 acceptance gate: express each hand-written decompose as a
    // PatternNode region, re-emit it via the grown `emit`, and assert the
    // result is structurally identical (op + shape + dtype at every node) to
    // the decompose-oracle output — the migration oracle. Since T5, `emit`
    // identity-shares repeated slot-free subtrees within one call, so a
    // shared oracle node compares against an equally-shared emitted node;
    // `assert_structural_eq` is recursive + order-sensitive (no commutative
    // canonicalization — stricter than `base_map_hash`), catching an
    // operand-swap the hash would mask.

    fn op_node(op: OpTag, attrs: OpAttrs, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op { op, attrs, operands }
    }
    fn bind(i: u8) -> PatternNode {
        PatternNode::Bind { index: i }
    }

    /// Recursively assert two subgraphs are identical: same Op, shape, dtype,
    /// arity, and recursively-equal inputs. Shared leaves (same NodeId) match
    /// by identity. This is the "byte-for-byte" node-structure check.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return; // shared leaf (bound external input)
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.shape, nb.shape, "shape mismatch at {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(na.inputs.len(), nb.inputs.len(), "arity mismatch at {:?}", na.op);
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::softmax_last_dim::decompose` (the legacy 7-node
    /// `ReduceMaxTo`/`ReduceSumTo` keepdim spelling), copied VERBATIM from
    /// that module @ `af4b7dd4` before T5 replaced the live body with the
    /// data recipe. Two consumers: the T5 numeric-parity oracle
    /// (`recipe_bridge` below) and `emit_matches_softmax_last_dim_decompose`
    /// (whose oracle was repointed here — the live decompose no longer emits
    /// this spelling).
    fn frozen_legacy_softmax_decompose(
        graph: &mut Graph,
        id: NodeId,
        _params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.shape.clone(), n.dtype)
        };
        let dims = x_shape.dims().to_vec();
        let rank = dims.len();
        let last = rank - 1;

        let mut keepdim_dims = dims.clone();
        keepdim_dims[last] = 1;
        let keepdim_shape = Shape::from_dims(&keepdim_dims);

        let m_id = graph.push(Node {
            op:     Op::ReduceMaxTo(keepdim_shape.clone()),
            inputs: vec![x_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let mb_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![m_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let s_id = graph.push(Node {
            op:     Op::Sub,
            inputs: vec![x_id, mb_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let e_id = graph.push(Node {
            op:     Op::Exp,
            inputs: vec![s_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let d_id = graph.push(Node {
            op:     Op::ReduceSumTo(keepdim_shape.clone()),
            inputs: vec![e_id],
            shape:  keepdim_shape,
            dtype,
        });
        let db_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![d_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let out_id = graph.push(Node {
            op:     Op::Div,
            inputs: vec![e_id, db_id],
            shape:  x_shape,
            dtype,
        });
        out_id
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::rope::decompose` (the legacy 11-node spelling with two
    /// leading-1-padded `Reshape` prep nodes), copied VERBATIM from that
    /// module @ `af4b7dd4` before T6 replaced the live body with the data
    /// recipe. Two consumers: `emit_matches_rope_decompose` (whose oracle was
    /// repointed here — the live decompose now emits the recipe, which at
    /// EQUAL rank elides the no-op prep `Reshape`, D4) and the T6
    /// numeric/structural parity oracle (`rope_recipe` below).
    fn frozen_legacy_rope_decompose(
        graph: &mut Graph,
        id: NodeId,
        _params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, cos_id, sin_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.inputs[2], n.shape.clone(), n.dtype)
        };
        let dims = x_shape.dims().to_vec();
        let rank = dims.len();
        let seq = dims[rank - 2];
        let d = dims[rank - 1];
        let half = d / 2;
        let last = rank - 1;

        let mut broadcast_shape_dims: Vec<usize> = vec![1usize; rank];
        broadcast_shape_dims[rank - 2] = seq;
        broadcast_shape_dims[rank - 1] = d;
        let broadcast_shape = Shape::from_dims(&broadcast_shape_dims);

        let cos_reshaped_id = graph.push(Node {
            op:     Op::Reshape(broadcast_shape.clone()),
            inputs: vec![cos_id],
            shape:  broadcast_shape.clone(),
            dtype,
        });
        let sin_reshaped_id = graph.push(Node {
            op:     Op::Reshape(broadcast_shape.clone()),
            inputs: vec![sin_id],
            shape:  broadcast_shape,
            dtype,
        });
        let cos_bcast_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![cos_reshaped_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let sin_bcast_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![sin_reshaped_id],
            shape:  x_shape.clone(),
            dtype,
        });

        let mut half_dims = dims.clone();
        half_dims[last] = half;
        let half_shape = Shape::from_dims(&half_dims);

        let first_half_id = graph.push(Node {
            op:     Op::Slice { dim: last, start: 0, len: half },
            inputs: vec![x_id],
            shape:  half_shape.clone(),
            dtype,
        });
        let second_half_id = graph.push(Node {
            op:     Op::Slice { dim: last, start: half, len: half },
            inputs: vec![x_id],
            shape:  half_shape.clone(),
            dtype,
        });
        let neg_second_id = graph.push(Node {
            op:     Op::Neg,
            inputs: vec![second_half_id],
            shape:  half_shape,
            dtype,
        });
        let rotated_half_id = graph.push(Node {
            op:     Op::Concat { dim: last },
            inputs: vec![neg_second_id, first_half_id],
            shape:  x_shape.clone(),
            dtype,
        });

        let left_id = graph.push(Node {
            op:     Op::Mul,
            inputs: vec![x_id, cos_bcast_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let right_id = graph.push(Node {
            op:     Op::Mul,
            inputs: vec![rotated_half_id, sin_bcast_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let out_id = graph.push(Node {
            op:     Op::Add,
            inputs: vec![left_id, right_id],
            shape:  x_shape,
            dtype,
        });
        out_id
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::rms_norm_last_dim::decompose` (the legacy 7-node
    /// `MeanDim → Reshape(keepdim) → AddScalar(eps)` spelling), copied VERBATIM
    /// from that module before T7 replaced the live body with the data recipe.
    /// Consumer: the T7 numeric-parity oracle (`norm_recipe` below). The live
    /// decompose now emits the D3 shrink-via-swap spelling (`Unsqueeze` append
    /// in place of `Reshape(keepdim)`), so the parity test evaluates BOTH
    /// through the shared reference interpreter and asserts bit-exact
    /// equivalence (the swap is metadata-only).
    fn frozen_legacy_rms_norm_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.shape.clone(), n.dtype)
        };
        let eps = match params {
            FusedOpParams::RmsNormLastDim { eps } => *eps,
            _ => return id,
        };
        let dims = x_shape.dims().to_vec();
        let rank = dims.len();
        let last = rank - 1;

        let mut keepdim_dims = dims.clone();
        keepdim_dims[last] = 1;
        let keepdim_shape = Shape::from_dims(&keepdim_dims);
        let mut reduced_dims = dims.clone();
        reduced_dims.remove(last);
        let reduced_shape = Shape::from_dims(&reduced_dims);

        let sq_id = graph.push(Node {
            op:     Op::Sqr,
            inputs: vec![x_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let mean_id = graph.push(Node {
            op:     Op::MeanDim(last),
            inputs: vec![sq_id],
            shape:  reduced_shape,
            dtype,
        });
        let mean_kd_id = graph.push(Node {
            op:     Op::Reshape(keepdim_shape.clone()),
            inputs: vec![mean_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let denom_sq_id = graph.push(Node {
            op:     Op::AddScalar(eps),
            inputs: vec![mean_kd_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let denom_id = graph.push(Node {
            op:     Op::Sqrt,
            inputs: vec![denom_sq_id],
            shape:  keepdim_shape,
            dtype,
        });
        let denom_bcast_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![denom_id],
            shape:  x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op:     Op::Div,
            inputs: vec![x_id, denom_bcast_id],
            shape:  x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::layer_norm_last_dim::decompose` (the legacy 11-node spelling
    /// with two `Reshape(keepdim)` restores and the `centered` subterm shared
    /// between `Sqr` and the final `Div`), copied VERBATIM from that module
    /// before T7 replaced the live body with the data recipe. Two consumers:
    /// `emit_matches_layer_norm_last_dim_decompose` (whose oracle was repointed
    /// here — the live decompose now emits the `Unsqueeze` D3 spelling) and the
    /// T7 numeric-parity oracle (`norm_recipe` below).
    fn frozen_legacy_layer_norm_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.shape.clone(), n.dtype)
        };
        let eps = match params {
            FusedOpParams::LayerNormLastDim { eps } => *eps,
            _ => return id,
        };
        let dims = x_shape.dims().to_vec();
        let rank = dims.len();
        let last = rank - 1;

        let mut keepdim_dims = dims.clone();
        keepdim_dims[last] = 1;
        let keepdim_shape = Shape::from_dims(&keepdim_dims);
        let mut reduced_dims = dims.clone();
        reduced_dims.remove(last);
        let reduced_shape = Shape::from_dims(&reduced_dims);

        let mean_id = graph.push(Node {
            op:     Op::MeanDim(last),
            inputs: vec![x_id],
            shape:  reduced_shape.clone(),
            dtype,
        });
        let mean_kd_id = graph.push(Node {
            op:     Op::Reshape(keepdim_shape.clone()),
            inputs: vec![mean_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let mean_bcast_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![mean_kd_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let centered_id = graph.push(Node {
            op:     Op::Sub,
            inputs: vec![x_id, mean_bcast_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let centered_sq_id = graph.push(Node {
            op:     Op::Sqr,
            inputs: vec![centered_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let var_id = graph.push(Node {
            op:     Op::MeanDim(last),
            inputs: vec![centered_sq_id],
            shape:  reduced_shape,
            dtype,
        });
        let var_kd_id = graph.push(Node {
            op:     Op::Reshape(keepdim_shape.clone()),
            inputs: vec![var_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let var_eps_id = graph.push(Node {
            op:     Op::AddScalar(eps),
            inputs: vec![var_kd_id],
            shape:  keepdim_shape.clone(),
            dtype,
        });
        let denom_id = graph.push(Node {
            op:     Op::Sqrt,
            inputs: vec![var_eps_id],
            shape:  keepdim_shape,
            dtype,
        });
        let denom_bcast_id = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![denom_id],
            shape:  x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op:     Op::Div,
            inputs: vec![centered_id, denom_bcast_id],
            shape:  x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::softmax_last_dim_backward::decompose` (the legacy 5-node
    /// `Mul`/`ReduceSumTo(keepdim)`/`BroadcastTo`/`Sub`/`Mul` spelling), copied
    /// VERBATIM from that module @ `aa2eee3c` before T8 replaced the live body
    /// with the data recipe. Sole consumer: the T8 numeric-parity oracle
    /// (`softmax_backward_recipe` below). Reads `inputs[0] = s` (the forward
    /// softmax output) and `inputs[1] = g` (the upstream gradient) off the node
    /// — the same convention the autograd `BackwardKind::Fused` edge emits
    /// (`lib.rs` softmax-backward arm: `vec![id, up_id]`).
    fn frozen_legacy_softmax_backward_decompose(
        graph: &mut Graph,
        id: NodeId,
        _params: &FusedOpParams,
    ) -> NodeId {
        let (s_id, g_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
        };
        // keepdim shape: last dim → 1.
        let mut kd = x_shape.dims().to_vec();
        let last = kd.len() - 1;
        kd[last] = 1;
        let keepdim = Shape::from_dims(&kd);

        let gs = graph.push(Node {
            op:     Op::Mul,
            inputs: vec![g_id, s_id],
            shape:  x_shape.clone(),
            dtype,
        });
        let summed = graph.push(Node {
            op:     Op::ReduceSumTo(keepdim.clone()),
            inputs: vec![gs],
            shape:  keepdim,
            dtype,
        });
        let summed_b = graph.push(Node {
            op:     Op::BroadcastTo(x_shape.clone()),
            inputs: vec![summed],
            shape:  x_shape.clone(),
            dtype,
        });
        let sub = graph.push(Node {
            op:     Op::Sub,
            inputs: vec![g_id, summed_b],
            shape:  x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op:     Op::Mul,
            inputs: vec![s_id, sub],
            shape:  x_shape,
            dtype,
        })
    }

    #[test]
    fn emit_matches_softmax_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        // Oracle: the FROZEN legacy builder (reads inputs[0] + shape + dtype
        // off the node). T5 repointed this from the live registry decompose —
        // which now emits the 9-node recipe spelling — so this test keeps
        // pinning the Increment-A guarantee it always pinned: the grown
        // `emit` reconstructs the LEGACY imperative structure from the legacy
        // region datum.
        let fused = g.push(Node { op: Op::Const, inputs: vec![x], shape: sh.clone(), dtype: DType::F32 });
        let oracle = frozen_legacy_softmax_decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);

        // keepdim shape [2,1]; full shape [2,4].
        let kd = OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() };
        let full = OpAttrs { target_shape: vec![2, 4], ..OpAttrs::default() };
        // e = Exp(Sub(x, BroadcastTo(ReduceMaxTo(x)))) — mirrors decompose order
        // `Sub{[x, mb]}` exactly; built fresh each call so numerator and the
        // denominator's ReduceSumTo input are identical subtrees.
        let softmax_e = |kd: &OpAttrs, full: &OpAttrs| {
            op_node(OpTag::Exp, OpAttrs::default(), vec![
                op_node(OpTag::Sub, OpAttrs::default(), vec![
                    bind(0),
                    op_node(OpTag::BroadcastTo, full.clone(), vec![
                        op_node(OpTag::ReduceMaxTo, kd.clone(), vec![bind(0)]),
                    ]),
                ]),
            ])
        };
        // out = Div(e, BroadcastTo(ReduceSumTo(e))) — mirrors `Div{[e, db]}`.
        let region = op_node(OpTag::Div, OpAttrs::default(), vec![
            softmax_e(&kd, &full),
            op_node(OpTag::BroadcastTo, full.clone(), vec![
                op_node(OpTag::ReduceSumTo, kd.clone(), vec![softmax_e(&kd, &full)]),
            ]),
        ]);
        let emitted = emit_region(&mut g, &region, &[x], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }

    #[test]
    fn emit_matches_rope_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]); // seq=2, d=4, half=2
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let cos = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let sin = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let fused = g.push(Node { op: Op::Const, inputs: vec![x, cos, sin], shape: sh.clone(), dtype: DType::F32 });
        // Oracle: the FROZEN legacy builder. T6 repointed this from the live
        // registry decompose — which now emits the data recipe (byte-identical
        // to legacy where a rank-raise occurs, but at EQUAL rank the recipe
        // elides the legacy's no-op prep `Reshape`, D4). This test keeps
        // pinning the Increment-A guarantee it always pinned: the grown `emit`
        // reconstructs the LEGACY imperative structure from a legacy-spelled
        // region datum.
        let oracle = frozen_legacy_rope_decompose(&mut g, fused, &FusedOpParams::Rope);

        // decompose's broadcast_shape for rank-2 [2,4] is [seq,d] = [2,4]; half slices along last dim.
        let full = OpAttrs { target_shape: vec![2, 4], ..OpAttrs::default() };
        let sl_first = OpAttrs { axis: Some(1), slice_start: Some(0), slice_len: Some(2), ..OpAttrs::default() };
        let sl_second = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(2), ..OpAttrs::default() };
        let cat = OpAttrs { axis: Some(1), ..OpAttrs::default() };
        let bcast_reshape = |full: &OpAttrs, i: u8| {
            op_node(OpTag::BroadcastTo, full.clone(), vec![
                op_node(OpTag::Reshape, full.clone(), vec![bind(i)]),
            ])
        };
        // left = Mul(x, cos_bcast); right = Mul(rotated_half, sin_bcast); out = Add(left, right).
        // rotated_half = Concat{dim:1}(Neg(second_half), first_half).
        let rotated = op_node(OpTag::Concat, cat, vec![
            op_node(OpTag::Neg, OpAttrs::default(), vec![op_node(OpTag::Slice, sl_second, vec![bind(0)])]),
            op_node(OpTag::Slice, sl_first, vec![bind(0)]),
        ]);
        let left = op_node(OpTag::Mul, OpAttrs::default(), vec![bind(0), bcast_reshape(&full, 1)]);
        let right = op_node(OpTag::Mul, OpAttrs::default(), vec![rotated, bcast_reshape(&full, 2)]);
        let region = op_node(OpTag::Add, OpAttrs::default(), vec![left, right]);

        let emitted = emit_region(&mut g, &region, &[x, cos, sin], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }

    #[test]
    fn emit_matches_layer_norm_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]); // last=1, reduced [2], keepdim [2,1]
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let fused = g.push(Node { op: Op::Const, inputs: vec![x], shape: sh.clone(), dtype: DType::F32 });
        // Oracle: the FROZEN legacy builder (the `Reshape(keepdim)` spelling).
        // T7 repointed this from the live registry decompose — which now emits
        // the D3 `Unsqueeze` swap — so this test keeps pinning the Increment-A
        // guarantee it always pinned: the grown `emit` reconstructs the LEGACY
        // imperative structure from a legacy-spelled (Reshape) region datum.
        let oracle = frozen_legacy_layer_norm_decompose(
            &mut g, fused, &FusedOpParams::LayerNormLastDim { eps: 1e-5 },
        );

        let kd = OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() };
        let full = OpAttrs { target_shape: vec![2, 4], ..OpAttrs::default() };
        let md = OpAttrs { axis: Some(1), ..OpAttrs::default() };
        let eps_attrs = OpAttrs { scalars: vec![1e-5], ..OpAttrs::default() }; // BAKED constant, not a slot
        // centered = Sub(x, BroadcastTo(Reshape(MeanDim(x)))) — shared subterm.
        let centered = op_node(OpTag::Sub, OpAttrs::default(), vec![
            bind(0),
            op_node(OpTag::BroadcastTo, full.clone(), vec![
                op_node(OpTag::Reshape, kd.clone(), vec![
                    op_node(OpTag::MeanDim, md.clone(), vec![bind(0)]),
                ]),
            ]),
        ]);
        // denom_bcast = BroadcastTo(Sqrt(AddScalar(eps)(Reshape(MeanDim(Sqr(centered)))))).
        let denom_bcast = op_node(OpTag::BroadcastTo, full.clone(), vec![
            op_node(OpTag::Sqrt, OpAttrs::default(), vec![
                op_node(OpTag::AddScalar, eps_attrs, vec![
                    op_node(OpTag::Reshape, kd.clone(), vec![
                        op_node(OpTag::MeanDim, md.clone(), vec![
                            op_node(OpTag::Sqr, OpAttrs::default(), vec![centered.clone()]),
                        ]),
                    ]),
                ]),
            ]),
        ]);
        // out = Div(centered, denom_bcast).
        let region = op_node(OpTag::Div, OpAttrs::default(), vec![centered, denom_bcast]);

        let emitted = emit_region(&mut g, &region, &[x], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }

    // ---- T2 (Increment C slice 1): shape-relative attr resolution ----------
    //
    // `resolve_rel_attrs` is the PURE resolver behind recipe polymorphism: it
    // turns the shape-relative interior fields (`target_shape_rel` /
    // `slice_{start,len}_rel` / `axis_last`, D2) into the concrete sibling
    // fields against the given bind/child shapes, reusing `shape_expr`'s
    // evaluator (no second evaluator). Every failure is a typed
    // [`RelAttrError`], never a panic.

    mod resolve_rel {
        use super::super::{RelAttrError, resolve_rel_attrs};
        use fuel_kernel_seam_types::shape_expr::{
            Dim, LAST, SYMBOLIC, ShapeExpr, ShapeExprError, TAG_DIMS,
        };
        use fuel_kernel_seam_types::{OpAttrs, OpTag};

        fn half_of_bind0_last() -> Dim {
            Dim::Div(
                Box::new(Dim::Extent { operand: 0, axis: LAST }),
                Box::new(Dim::Const(2)),
            )
        }

        #[test]
        fn same_as_bind0_tracks_the_bind_shape() {
            // The polymorphism seed: ONE recipe datum, two shapes, two targets.
            let attrs = OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                ..OpAttrs::default()
            };
            let r = resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &[vec![2, 3]], &[vec![2, 1]])
                .expect("resolves");
            assert_eq!(r.target_shape, vec![2, 3]);
            assert!(r.target_shape_rel.is_none(), "resolved attrs are fully concrete");
            let r = resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &[vec![4, 5]], &[vec![4, 1]])
                .expect("resolves");
            assert_eq!(r.target_shape, vec![4, 5]);
        }

        #[test]
        fn slice_bounds_track_the_bind_extent() {
            // start = len = Extent(bind0, LAST) / 2 — the rope-half worked example.
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start_rel: Some(half_of_bind0_last()),
                slice_len_rel: Some(half_of_bind0_last()),
                ..OpAttrs::default()
            };
            let r = resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]])
                .expect("resolves at d=4");
            assert_eq!(r.slice_start, Some(2));
            assert_eq!(r.slice_len, Some(2));
            assert!(r.slice_start_rel.is_none() && r.slice_len_rel.is_none());
            let r = resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 8]], &[vec![2, 8]])
                .expect("resolves at d=8");
            assert_eq!(r.slice_start, Some(4));
            assert_eq!(r.slice_len, Some(4));
        }

        #[test]
        fn axis_last_resolves_per_tag() {
            let attrs = OpAttrs { axis_last: true, ..OpAttrs::default() };
            // Reduce family (axis carrier): LAST = rank − 1.
            let r = resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 4]]).expect("rank 2");
            assert_eq!(r.axis, Some(1));
            assert!(!r.axis_last, "resolved attrs are fully concrete");
            let r = resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 3, 4]]).expect("rank 3");
            assert_eq!(r.axis, Some(2));
            // Concat rides the same axis carrier.
            let r = resolve_rel_attrs(OpTag::Concat, &attrs, &[], &[vec![2, 3, 4]]).expect("concat");
            assert_eq!(r.axis, Some(2));
            // Unsqueeze (dims carrier): APPEND — dim == rank (`primitive_shape`
            // permits `dim == rank`).
            let r = resolve_rel_attrs(OpTag::Unsqueeze, &attrs, &[], &[vec![2, 4]]).expect("unsqueeze");
            assert_eq!(r.dims, vec![2]);
            assert!(!r.axis_last);
            // Squeeze (dims carrier): LAST = rank − 1.
            let r = resolve_rel_attrs(OpTag::Squeeze, &attrs, &[], &[vec![2, 4, 1]]).expect("squeeze");
            assert_eq!(r.dims, vec![2]);
        }

        #[test]
        fn bind_out_of_range_is_a_typed_decline() {
            let attrs = OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 3 }),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &[vec![2, 3]], &[vec![2, 3]]),
                Err(RelAttrError::Expr(ShapeExprError::OperandOutOfRange { operand: 3, operands: 1 })),
            );
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start_rel: Some(Dim::Extent { operand: 7, axis: LAST }),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::Expr(ShapeExprError::OperandOutOfRange { operand: 7, operands: 1 })),
            );
        }

        #[test]
        fn rel_and_abs_both_set_is_a_typed_conflict() {
            // target_shape XOR target_shape_rel.
            let attrs = OpAttrs {
                target_shape: vec![2, 3],
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &[vec![2, 3]], &[vec![2, 3]]),
                Err(RelAttrError::RelAbsConflict { field: "target_shape" }),
            );
            // slice_start XOR slice_start_rel.
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start: Some(0),
                slice_start_rel: Some(half_of_bind0_last()),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "slice_start" }),
            );
            // slice_len XOR slice_len_rel.
            let attrs = OpAttrs {
                axis: Some(1),
                slice_len: Some(2),
                slice_len_rel: Some(half_of_bind0_last()),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "slice_len" }),
            );
            // axis XOR axis_last.
            let attrs = OpAttrs { axis: Some(0), axis_last: true, ..OpAttrs::default() };
            assert_eq!(
                resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "axis" }),
            );
            // dims XOR axis_last (Unsqueeze's carrier is `dims`).
            let attrs = OpAttrs { dims: vec![0], axis_last: true, ..OpAttrs::default() };
            assert_eq!(
                resolve_rel_attrs(OpTag::Unsqueeze, &attrs, &[], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "dims" }),
            );
        }

        #[test]
        fn negative_result_is_a_typed_decline() {
            // 0 − 2 = −2: a negative slice offset is malformed, not a wrap.
            let neg = Dim::Sub(Box::new(Dim::Const(0)), Box::new(Dim::Const(2)));
            let attrs = OpAttrs { axis: Some(1), slice_start_rel: Some(neg), ..OpAttrs::default() };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::Negative { field: "slice_start", value: -2 }),
            );
        }

        #[test]
        fn symbolic_extent_is_a_surfaced_gap_decline() {
            // A symbolic bind extent → the expression evaluates to Gap → typed
            // decline (the emit caller surfaces it as a fixpoint, G2).
            let attrs = OpAttrs {
                axis: Some(1),
                slice_len_rel: Some(half_of_bind0_last()),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, SYMBOLIC]], &[vec![2, 4]]),
                Err(RelAttrError::SymbolicGap { field: "slice_len" }),
            );
            let attrs = OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &[vec![2, SYMBOLIC]], &[vec![2, 4]]),
                Err(RelAttrError::SymbolicGap { field: "target_shape" }),
            );
        }

        #[test]
        fn two_operand_max_frame_declines_instead_of_a_partial_shape() {
            // I1 (Baracuda §6.20 finding): an ELEMENTWISE output frame is not
            // always expressible as `SameAs(operand)`. When the frame is
            // assembled by per-axis max across TWO binds — `a[2,1] ⊗ b[1,3] →
            // [2,3]` — NO single operand carries it, so BOTH spellings resolve
            // to a PARTIAL frame ([2,1] / [1,3]) and would silently emit the
            // wrong `BroadcastTo` target. Must be a typed decline.
            // The decline is typed and names the Dims-class constructor that
            // WOULD express it (reserved tag 0x0B, KISS #80) — a surfaced gap,
            // never a panic and never a wrong shape.
            let binds = vec![vec![2, 1], vec![1, 3]]; // frame = [2,3], carried by neither
            for operand in [0u8, 1] {
                let attrs = OpAttrs {
                    target_shape_rel: Some(ShapeExpr::SameAs { operand }),
                    ..OpAttrs::default()
                };
                let child = binds[operand as usize].clone();
                assert_eq!(
                    resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &binds, &[child]),
                    Err(RelAttrError::FrameNotExpressible {
                        field: "target_shape",
                        frame: vec![2, 3],
                        missing_ctor: TAG_DIMS,
                    }),
                    "SameAs {{ operand: {operand} }} must not resolve to a PARTIAL frame",
                );
            }
        }

        #[test]
        fn frame_guard_does_not_fire_when_an_operand_carries_the_frame() {
            // The guard is deliberately narrow. It must NOT degrade the cases
            // the 5 migrated recipes actually use.
            let attrs = OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                ..OpAttrs::default()
            };
            // (a) bind0 IS the frame (softmax/rope/rms-norm/layer-norm shape).
            let binds = vec![vec![2, 3, 4], vec![4]];
            let r = resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &binds, &[vec![4]])
                .expect("bind0 carries the frame");
            assert_eq!(r.target_shape, vec![2, 3, 4]);
            // (b) a SUB-frame target is legitimate: the frame is bind0's, and
            // naming bind1's smaller shape is an ordinary interior broadcast.
            let sub = OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 1 }),
                ..OpAttrs::default()
            };
            let r = resolve_rel_attrs(OpTag::BroadcastTo, &sub, &binds, &[vec![4]])
                .expect("sub-frame broadcast stays expressible");
            assert_eq!(r.target_shape, vec![4]);
            // (c) binds with NO joint elementwise frame (a matmul region) —
            // per-axis max is meaningless there, so the guard stays out.
            let mm = vec![vec![8, 4096], vec![4096, 1024]];
            let r = resolve_rel_attrs(OpTag::BroadcastTo, &attrs, &mm, &[vec![8, 1]])
                .expect("no joint frame ⇒ no guard");
            assert_eq!(r.target_shape, vec![8, 4096]);
            // (d) a NON-frame-carrier tag is untouched: only `BroadcastTo`'s
            // target IS the elementwise output frame.
            let two = vec![vec![2, 1], vec![1, 3]];
            let r = resolve_rel_attrs(OpTag::Reshape, &attrs, &two, &[vec![2, 1]])
                .expect("Reshape's target is not a frame claim");
            assert_eq!(r.target_shape, vec![2, 1]);
        }

        #[test]
        fn axis_last_on_an_axisless_tag_or_without_a_child_declines() {
            let attrs = OpAttrs { axis_last: true, ..OpAttrs::default() };
            // Add has no axis carrier — axis_last is meaningless, a typed decline
            // (build-time validation posture: never silently ignore).
            assert_eq!(
                resolve_rel_attrs(OpTag::Add, &attrs, &[], &[vec![2, 4], vec![2, 4]]),
                Err(RelAttrError::AxisLastUnsupported { tag: OpTag::Add }),
            );
            // No child operand → no rank to resolve LAST against.
            assert_eq!(
                resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[]),
                Err(RelAttrError::NoChildOperand),
            );
            // Rank-0 child: LAST has no axis → the shared resolve_axis decline.
            assert_eq!(
                resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![]]),
                Err(RelAttrError::Expr(ShapeExprError::AxisOutOfRange { axis: LAST, rank: 0 })),
            );
        }

        #[test]
        fn rel_free_attrs_pass_through_unchanged() {
            // The no-rel fast path: absolute attrs resolve to themselves.
            let attrs = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
            let r = resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]).expect("no-op");
            assert_eq!(r, attrs);
        }
    }

    // ---- T3 (Increment C slice 1): resolving emit + D4 pad + rel validation ----
    //
    // The emit integration behind recipe polymorphism: children are emitted
    // FIRST (their shapes feed the rel-attr resolver), `resolve_rel_attrs`
    // produces fully-concrete attrs, then the unchanged tag_to_op →
    // primitive_shape path runs. A resolved `BroadcastTo` whose target rank
    // exceeds its operand's materializes the legacy `Reshape` pad (D4).
    // `validate_representable` accepts rel-attr regions via a probe-resolve
    // (mirror of the scalar slot dummy-fill) and rejects structural authoring
    // errors (rel+abs conflict, bind out of range) with a typed decline.

    mod emit_rel {
        use super::super::*;
        use super::{assert_structural_eq, bind, op_node};
        use fuel_ir::{DType, Shape};
        use fuel_kernel_seam_types::shape_expr::{Dim, ShapeExpr};

        fn cst(g: &mut Graph, dims: &[usize]) -> NodeId {
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(dims),
                dtype: DType::F32,
            })
        }

        fn bcast_same_as_0() -> OpAttrs {
            OpAttrs { target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }), ..OpAttrs::default() }
        }

        #[test]
        fn rel_region_emits_polymorphically_across_shapes() {
            // The headline polymorphism: ONE region datum —
            // Add(bind0, BroadcastTo{SameAs{0}}(bind1)) — emitted at two
            // different shapes produces the correct target BOTH times
            // (impossible with absolute attrs: a baked target matches exactly
            // one shape).
            let region = op_node(OpTag::Add, OpAttrs::default(), vec![
                bind(0),
                op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
            ]);
            let mut g = Graph::new();
            let x1 = cst(&mut g, &[2, 3]);
            let t1 = cst(&mut g, &[1, 3]);
            let r1 = emit_region(&mut g, &region, &[x1, t1], &[]);
            assert!(matches!(g.node(r1).op, Op::Add));
            assert_eq!(g.node(r1).shape, Shape::from_dims(&[2, 3]));
            let b1 = g.node(r1).inputs[1];
            assert_eq!(g.node(b1).op, Op::BroadcastTo(Shape::from_dims(&[2, 3])));
            assert_eq!(g.node(b1).shape, Shape::from_dims(&[2, 3]));

            // The SAME region datum at different shapes → a different target.
            let x2 = cst(&mut g, &[4, 5]);
            let t2 = cst(&mut g, &[1, 5]);
            let r2 = emit_region(&mut g, &region, &[x2, t2], &[]);
            assert_eq!(g.node(r2).shape, Shape::from_dims(&[4, 5]));
            let b2 = g.node(r2).inputs[1];
            assert_eq!(g.node(b2).op, Op::BroadcastTo(Shape::from_dims(&[4, 5])));
        }

        #[test]
        fn two_operand_max_frame_region_declines_through_emit() {
            // I1 end-to-end: the ONE region spelling that WANTS the per-axis-max
            // frame — Mul(BroadcastTo(a[2,1]), BroadcastTo(b[1,3])) → [2,3],
            // which Fuel's primitive `Mul` requires explicitly (`primitive_shape`
            // takes in[0]'s shape, it does not broadcast). `SameAs` cannot name
            // [2,3], so the resolving emit surfaces the typed Dims-class gap
            // instead of emitting a partial-frame BroadcastTo.
            let region = op_node(OpTag::Mul, OpAttrs::default(), vec![
                op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(0)]),
                op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
            ]);
            let mut g = Graph::new();
            let a = cst(&mut g, &[2, 1]);
            let b = cst(&mut g, &[1, 3]);
            assert_eq!(
                try_emit_region(&mut g, &region, &[a, b], &[]),
                Err(RelAttrError::FrameNotExpressible {
                    field: "target_shape",
                    frame: vec![2, 3],
                    missing_ctor: fuel_kernel_seam_types::shape_expr::TAG_DIMS,
                }),
            );
        }

        #[test]
        fn broadcast_rank_raise_materializes_the_legacy_reshape_pad() {
            // D4: a rank-1 bind1 broadcast to rank-3 bind0's shape — the
            // resolver must first push the legacy `Reshape` (1-padded left,
            // right-aligned; `registry::rope`'s hand-built broadcast prep).
            let region = op_node(OpTag::Mul, OpAttrs::default(), vec![
                bind(0),
                op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
            ]);
            let mut g = Graph::new();
            let x = cst(&mut g, &[2, 3, 4]);
            let t = cst(&mut g, &[4]);
            // Hand-built legacy reference:
            // Reshape([1,1,4])(t) → BroadcastTo([2,3,4]) → Mul(x, ·).
            let pad_shape = Shape::from_dims(&[1, 1, 4]);
            let full = Shape::from_dims(&[2, 3, 4]);
            let pad = g.push(Node {
                op: Op::Reshape(pad_shape.clone()),
                inputs: vec![t],
                shape: pad_shape,
                dtype: DType::F32,
            });
            let bc = g.push(Node {
                op: Op::BroadcastTo(full.clone()),
                inputs: vec![pad],
                shape: full.clone(),
                dtype: DType::F32,
            });
            let reference = g.push(Node {
                op: Op::Mul,
                inputs: vec![x, bc],
                shape: full,
                dtype: DType::F32,
            });

            let emitted = emit_region(&mut g, &region, &[x, t], &[]);
            assert_structural_eq(&g, reference, emitted);
        }

        #[test]
        fn concrete_broadcast_rank_raise_also_pads() {
            // D4 applies uniformly: an ABSOLUTE rank-raising BroadcastTo also
            // materializes the pad (deterministic emission, matches the graph
            // builders' right-aligned rank-raising semantics). Equal-rank
            // broadcasts stay pad-free (pinned by the softmax parity oracle).
            let region = op_node(
                OpTag::BroadcastTo,
                OpAttrs { target_shape: vec![2, 3, 4], ..OpAttrs::default() },
                vec![bind(0)],
            );
            let mut g = Graph::new();
            let t = cst(&mut g, &[3, 4]);
            let emitted = emit_region(&mut g, &region, &[t], &[]);
            assert_eq!(g.node(emitted).op, Op::BroadcastTo(Shape::from_dims(&[2, 3, 4])));
            let pad = g.node(emitted).inputs[0];
            assert_eq!(
                g.node(pad).op,
                Op::Reshape(Shape::from_dims(&[1, 3, 4])),
                "rank-raise inserts the legacy 1-padded-left Reshape",
            );
            assert_eq!(g.node(pad).inputs, vec![t]);
        }

        #[test]
        fn scalar_cursor_fill_stays_pre_order_after_the_reorder() {
            // Risk-2 guard: children are now EMITTED first, but the scalar
            // cursor fill stays PRE-order (parent before descent) — the
            // canonical authoring order `match_region_extract` records.
            let region = op_node(OpTag::AddScalar, OpAttrs::default(), vec![
                op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)]),
            ]);
            let mut g = Graph::new();
            let x = cst(&mut g, &[4]);
            let root = emit_region(&mut g, &region, &[x], &[10.0, 20.0]);
            assert!(
                matches!(g.node(root).op, Op::AddScalar(v) if v == 10.0),
                "parent takes scalars[0] (pre-order), got {:?}",
                g.node(root).op,
            );
            let child = g.node(root).inputs[0];
            assert!(
                matches!(g.node(child).op, Op::MulScalar(v) if v == 20.0),
                "child takes scalars[1], got {:?}",
                g.node(child).op,
            );
        }

        #[test]
        fn validate_accepts_a_rel_attr_region() {
            // Born-red: today tag_to_op(BroadcastTo, {empty target_shape}) →
            // None → UnRepresentable. The rel-probe must accept the template.
            let region = op_node(OpTag::Add, OpAttrs::default(), vec![
                bind(0),
                op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
            ]);
            register_runtime_fused("t3::rel_bcast", region)
                .expect("a rel-attr region is registrable");
        }

        #[test]
        fn validate_rejects_rel_abs_conflict_and_bind_out_of_range() {
            // rel+abs both set → a typed authoring reject, never a silent
            // precedence.
            let conflicted = op_node(
                OpTag::BroadcastTo,
                OpAttrs {
                    target_shape: vec![2, 3],
                    target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            assert_eq!(
                register_runtime_fused("t3::conflict", conflicted),
                Err(RuntimeFusedError::InvalidRelAttrs {
                    tag: OpTag::BroadcastTo,
                    error: RelAttrError::RelAbsConflict { field: "target_shape" },
                }),
            );
            // A bind reference outside the region's bind space can never
            // resolve at ANY shape → a typed authoring reject.
            let oob = op_node(
                OpTag::BroadcastTo,
                OpAttrs {
                    target_shape_rel: Some(ShapeExpr::SameAs { operand: 7 }),
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            assert_eq!(
                register_runtime_fused("t3::oob", oob),
                Err(RuntimeFusedError::InvalidRelAttrs {
                    tag: OpTag::BroadcastTo,
                    error: RelAttrError::Expr(
                        fuel_kernel_seam_types::shape_expr::ShapeExprError::OperandOutOfRange {
                            operand: 7,
                            operands: 1,
                        },
                    ),
                }),
            );
        }

        #[test]
        fn decompose_rel_resolution_failure_is_a_fixpoint_not_a_crash() {
            // slice_start_rel = 0 − 2 → Negative at emit time. Registration
            // TOLERATES it (a value-dependent decline at the probe shape is
            // not an authoring error); the decompose-path resolution failure
            // surfaces as a no-op fixpoint (G2), never a panic.
            let neg = Dim::Sub(Box::new(Dim::Const(0)), Box::new(Dim::Const(2)));
            let region = op_node(
                OpTag::Slice,
                OpAttrs {
                    axis: Some(1),
                    slice_start_rel: Some(neg),
                    slice_len: Some(1),
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            let id = register_runtime_fused("t3::neg_slice", region)
                .expect("a value-dependent decline still registers");
            let mut g = Graph::new();
            let x = cst(&mut g, &[2, 4]);
            let fused = g.push(Node {
                op: Op::Fused(id, FusedOpParams::Runtime { scalars: vec![] }),
                inputs: vec![x],
                shape: Shape::from_dims(&[2, 4]),
                dtype: DType::F32,
            });
            assert_eq!(decompose_region(&mut g, fused), fused, "resolution decline ⇒ fixpoint");
        }

        #[test]
        fn try_emit_region_surfaces_typed_resolution_errors() {
            // Negative → typed error from the fallible entry, never a panic.
            let neg = Dim::Sub(Box::new(Dim::Const(0)), Box::new(Dim::Const(2)));
            let region = op_node(
                OpTag::Slice,
                OpAttrs {
                    axis: Some(1),
                    slice_start_rel: Some(neg),
                    slice_len: Some(1),
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            let mut g = Graph::new();
            let x = cst(&mut g, &[2, 4]);
            assert_eq!(
                try_emit_region(&mut g, &region, &[x], &[]),
                Err(RelAttrError::Negative { field: "slice_start", value: -2 }),
            );
            // A graph-side SYMBOLIC bind extent maps to the §6.20 SYMBOLIC
            // sentinel → SymbolicGap (the surfaced-gap posture, §6.20-0004).
            let region = op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(0)]);
            let dyn_shape =
                Shape::from_dims(&[2, 8]).with_dynamic_axis(1, 0, fuel_ir::SymId(0));
            let d = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: dyn_shape,
                dtype: DType::F32,
            });
            assert_eq!(
                try_emit_region(&mut g, &region, &[d], &[]),
                Err(RelAttrError::SymbolicGap { field: "target_shape" }),
            );
        }
    }

    // ---- T5 (Increment C slice 1): identity-share of repeated subtrees ----
    //
    // A `PatternNode` recipe is a TREE; a DAG recipe (softmax's shared
    // `e = Exp(..)` interior, consumed by both the denominator reduce and the
    // final Div) is spelled by REPEATING the subtree. `emit` must emit a
    // repeated slot-free subtree ONCE per emit call (identity-share), so the
    // emitted graph is the DAG, not a duplicated-compute tree. Subtrees with
    // OPEN scalar slots are never shared — each occurrence takes its own
    // cursor value. (The flat-DAG node table with real CSE is slice 3.)

    #[test]
    fn emit_shares_repeated_slot_free_subtrees() {
        let region = op_node(OpTag::Add, OpAttrs::default(), vec![
            op_node(OpTag::Exp, OpAttrs::default(), vec![bind(0)]),
            op_node(OpTag::Exp, OpAttrs::default(), vec![bind(0)]),
        ]);
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[]);
        let add = g.node(root);
        assert!(matches!(add.op, Op::Add));
        assert_eq!(
            add.inputs[0], add.inputs[1],
            "structurally-equal slot-free subtrees must share ONE emitted node",
        );
    }

    #[test]
    fn emit_does_not_share_subtrees_with_open_scalar_slots() {
        // Two open MulScalar slots take DIFFERENT cursor values (pre-order
        // fill) — sharing them would silently drop the second live value.
        let region = op_node(OpTag::Add, OpAttrs::default(), vec![
            op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)]),
            op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)]),
        ]);
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[2.0, 3.0]);
        let a = g.node(root).inputs[0];
        let b = g.node(root).inputs[1];
        assert_ne!(a, b, "open-slot subtrees are never shared");
        assert!(matches!(g.node(a).op, Op::MulScalar(v) if v == 2.0));
        assert!(matches!(g.node(b).op, Op::MulScalar(v) if v == 3.0));
    }

    #[test]
    fn emit_matches_cast_over_add_reference() {
        // Exercises the dtype path through assert_structural_eq: a hand-built
        // two-node reference `Cast(F16)(Add(a, b))` vs the emitted region.
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        // Reference graph.
        let add = g.push(Node { op: Op::Add, inputs: vec![a, b], shape: sh.clone(), dtype: DType::F32 });
        let reference = g.push(Node { op: Op::Cast(DType::F16), inputs: vec![add], shape: sh.clone(), dtype: DType::F16 });

        let region = op_node(OpTag::Cast, OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() }, vec![
            op_node(OpTag::Add, OpAttrs::default(), vec![bind(0), bind(1)]),
        ]);
        let emitted = emit_region(&mut g, &region, &[a, b], &[]);
        assert_structural_eq(&g, reference, emitted);
    }

    // ---- T5 (Increment C slice 1): decompose_via_recipe bridge + the
    // softmax_last_dim pilot migration --------------------------------------
    //
    // The registry bridge (`crate::registry::decompose_via_recipe`, design
    // D6) makes a static entry's `decompose` a re-emit of portable
    // `PatternNode` DATA: node inputs are the binds, a per-entry projection
    // supplies the open-slot scalars, the resolving emit does the rest. ANY
    // failure — wrong params payload, a semantics-absent op token (the
    // flip-withdrawal posture: unknown/non-registry tokens are surfaced
    // honest-miss declines, NEVER accepted, NEVER a crash), a bind/arity or
    // slot-count mismatch, a rel-resolution decline at these shapes — returns
    // `id` (fixpoint, G2), never panics.

    mod recipe_bridge {
        use super::super::*;
        use super::{bind, frozen_legacy_softmax_decompose, op_node};
        use crate::registry::{FusedOps, decompose_via_recipe};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the primitive vocabulary the
        /// two softmax spellings use (Const leaves, last-axis reduces, keepdim
        /// restores, last-dim broadcast, elementwise). BOTH parity sides run
        /// through it, with in-order accumulation per row — so the bit-exact
        /// assert isolates recipe STRUCTURE; float noise can't differ between
        /// two evaluations of the same interpreter. (Not code evaluation: a
        /// closed match over our own `Op` enum — no dynamic execution.)
        fn eval(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            let node = g.node(id);
            match &node.op {
                Op::Const => leaves.get(&id).expect("leaf data provided").clone(),
                Op::Exp => eval(g, node.inputs[0], leaves).iter().map(|v| v.exp()).collect(),
                Op::Sub => {
                    let a = eval(g, node.inputs[0], leaves);
                    let b = eval(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x - y).collect()
                }
                Op::Div => {
                    let a = eval(g, node.inputs[0], leaves);
                    let b = eval(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x / y).collect()
                }
                // Last-axis reduces — one arm per spelling pair, identical
                // in-order fold.
                Op::MaxDim(_) | Op::ReduceMaxTo(_) => {
                    let input = eval(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                        .collect()
                }
                Op::SumDim(_) | Op::ReduceSumTo(_) => {
                    let input = eval(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input.chunks(last).map(|row| row.iter().sum()).collect()
                }
                // Metadata-only keepdim restores.
                Op::Unsqueeze { .. } | Op::Reshape(_) => eval(g, node.inputs[0], leaves),
                // Broadcast a keepdim/reduced tensor back along the last axis.
                Op::BroadcastTo(target) => {
                    let input = eval(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    let last = *target.dims().last().unwrap();
                    assert_eq!(
                        input.len() * last,
                        out_n,
                        "broadcast is a last-dim repeat in these graphs",
                    );
                    input
                        .iter()
                        .flat_map(|&v| std::iter::repeat(v).take(last))
                        .collect()
                }
                other => panic!("eval: unhandled op {other:?}"),
            }
        }

        fn softmax_fused_node(g: &mut Graph, dims: &[usize]) -> (NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
            let fused = g.push(Node {
                op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
                inputs: vec![x],
                shape: sh,
                dtype: DType::F32,
            });
            (x, fused)
        }

        /// T5 red (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3
        /// (the polymorphism the baked-shape legacy body never had), and its
        /// numerics match the FROZEN legacy builder bit-exactly under the
        /// shared reference interpreter.
        #[test]
        fn softmax_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (x, fused) = softmax_fused_node(&mut g, &dims);
                let sh = Shape::from_dims(&dims);
                let new_root = crate::registry::softmax_last_dim::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::SoftmaxLastDim,
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(g.node(new_root).shape, sh, "softmax is shape-preserving");
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root =
                    frozen_legacy_softmax_decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);

                let n: usize = dims.iter().product();
                let data: Vec<f64> =
                    (0..n).map(|i| ((i as f64) * 0.37).sin() * 3.0 - 0.5).collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, data);
                let got = eval(&g, new_root, &leaves);
                let want = eval(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "softmax[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// T5 red (b): the structural golden — the ratified D3 shrink-via-swap
        /// spelling, 9 op nodes with the `e = Exp(..)` interior SHARED (node
        /// identity) between the denominator reduce and the final Div.
        #[test]
        fn softmax_recipe_emits_the_nine_node_shared_spelling() {
            let mut g = Graph::new();
            let (x, fused) = softmax_fused_node(&mut g, &[2, 4]);
            let sh = Shape::from_dims(&[2, 4]);
            let root = crate::registry::softmax_last_dim::decompose(
                &mut g,
                fused,
                &FusedOpParams::SoftmaxLastDim,
            );

            // out = Div(e, db)
            assert!(matches!(g.node(root).op, Op::Div));
            let e = g.node(root).inputs[0];
            let db = g.node(root).inputs[1];
            assert!(matches!(g.node(e).op, Op::Exp));
            assert_eq!(g.node(db).op, Op::BroadcastTo(sh.clone()));
            // db = BroadcastTo(Unsqueeze(SumDim(e))) — the SAME e node.
            let u2 = g.node(db).inputs[0];
            assert!(matches!(g.node(u2).op, Op::Unsqueeze { dim: 1 }));
            assert_eq!(g.node(u2).shape, Shape::from_dims(&[2, 1]));
            let d = g.node(u2).inputs[0];
            assert!(matches!(g.node(d).op, Op::SumDim(1)));
            assert_eq!(g.node(d).shape, Shape::from_dims(&[2]));
            assert_eq!(
                g.node(d).inputs[0], e,
                "the denominator reduces the SHARED Exp node — identity-share, not a duplicate",
            );
            // e = Exp(Sub(x, mb)); mb = BroadcastTo(Unsqueeze(MaxDim(x))).
            let s = g.node(e).inputs[0];
            assert!(matches!(g.node(s).op, Op::Sub));
            assert_eq!(g.node(s).inputs[0], x);
            let mb = g.node(s).inputs[1];
            assert_eq!(g.node(mb).op, Op::BroadcastTo(sh.clone()));
            let u1 = g.node(mb).inputs[0];
            assert!(matches!(g.node(u1).op, Op::Unsqueeze { dim: 1 }));
            let m = g.node(u1).inputs[0];
            assert!(matches!(g.node(m).op, Op::MaxDim(1)));
            assert_eq!(g.node(m).inputs[0], x);
            // 9 op nodes + the x leaf = 10 reachable (NO duplicated interior).
            assert_eq!(
                crate::topo_order_multi(&g, &[root]).len(),
                10,
                "MaxDim/Unsqueeze/Bcast/Sub/Exp/SumDim/Unsqueeze/Bcast/Div + leaf",
            );
        }

        /// T5 red (c): totality — a wrong params payload is a typed decline
        /// surfaced as a fixpoint (G2), never a panic, and declines BEFORE any
        /// emission (no partial nodes).
        #[test]
        fn softmax_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, fused) = softmax_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = crate::registry::softmax_last_dim::decompose(
                &mut g,
                fused,
                &FusedOpParams::Rope,
            );
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }

        /// INJECTED item (2), the flip-withdrawal posture (Baracuda #68 /
        /// KISS-Ops closed registry): an op token with no registry semantics —
        /// the in-memory analog of the withdrawn reverse-scan "flip" spelling —
        /// must surface as a typed honest-miss decline: the node stays fused
        /// (fixpoint), NEVER accepted, NEVER a crash. The fabricated recipe
        /// stands in for a foreign token by carrying `OpTag::PowI` — a tag
        /// with NO primitive re-emission today (`tag_to_op` → `None`), exactly
        /// the semantics-absent posture an unregistered op name resolves to.
        /// If/when the token registers (flip returning to KISS-Ops; PowI
        /// gaining its carrier in slice 2), it becomes a NAMED-op resolution
        /// case — semantics arrive via registration, never via silent
        /// acceptance here.
        #[test]
        fn decompose_via_recipe_declines_an_unknown_token_recipe() {
            let fabricated = op_node(
                OpTag::PowI,
                OpAttrs { scalars: vec![3.0], ..OpAttrs::default() },
                vec![bind(0)],
            );
            let mut g = Graph::new();
            let (_x, fused) = softmax_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = decompose_via_recipe(&mut g, fused, &fabricated, Some(Vec::new()));
            assert_eq!(out, fused, "semantics-absent token ⇒ honest-miss fixpoint");
            assert_eq!(g.len(), before, "declined BEFORE any emission — no partial nodes");
        }

        /// The bridge's bind/input arity guard: a recipe over 2 binds cannot
        /// decompose a 1-input node — fixpoint, not a crash (and not a
        /// misbound emission).
        #[test]
        fn decompose_via_recipe_bind_arity_mismatch_is_a_fixpoint() {
            let recipe = op_node(OpTag::Add, OpAttrs::default(), vec![bind(0), bind(1)]);
            let mut g = Graph::new();
            let (_x, fused) = softmax_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = decompose_via_recipe(&mut g, fused, &recipe, Some(Vec::new()));
            assert_eq!(out, fused, "bind/input arity mismatch ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    // rope migration (Increment C slice 1, T6) ------------------------------
    //
    // Rope's 11-node imperative body becomes a 9-node portable `PatternNode`
    // DATA recipe: cos/sin broadcasts carry `SameAs { operand: 0 }`, the two
    // half-slices carry `DimExpr` start/len over the Bind space (the
    // reference-doc worked example: `start=Const(0), len=Div(E,2)` /
    // `start=Div(E,2), len=Sub(E, Div(E,2))`), the last-axis Concat carries
    // `axis_last`, and the two leading-1-padded prep `Reshape`s are NOT in the
    // datum — the emit resolver MATERIALIZES them (D4) only where the
    // broadcast target out-ranks its operand. Consequence: at a rank-RAISING
    // broadcast (the real attention consumer: cos/sin `[seq,d]`, x `[..,seq,d]`
    // rank ≥ 3) emission is BYTE-IDENTICAL to legacy (11 nodes, both pads); at
    // EQUAL rank (x itself rank 2 = `[seq,d]`) the recipe emits the 9-node
    // form, eliding legacy's no-op `Reshape([seq,d]→[seq,d])` — numerically
    // identical, structurally leaner. D4 is shared with softmax/norms and MUST
    // NOT add reshapes at equal rank (that would break the softmax parity
    // oracle), so the equal-rank elision is intrinsic, not a defect.
    mod rope_recipe {
        use super::super::*;
        use super::frozen_legacy_rope_decompose;
        use crate::registry::{FusedOps, rope};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the rope primitive vocabulary
        /// (Const leaves, metadata-only rank-pad Reshape, leading-dim
        /// BroadcastTo, last-axis Slice/Concat, Neg, elementwise Mul/Add).
        /// BOTH parity sides run through it in identical in-order arithmetic —
        /// so a bit-exact assert isolates recipe STRUCTURE. (Not code
        /// evaluation: a closed match over our own `Op` enum.)
        fn eval_rope(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            let node = g.node(id);
            match &node.op {
                Op::Const => leaves.get(&id).expect("leaf data provided").clone(),
                Op::Neg => eval_rope(g, node.inputs[0], leaves).iter().map(|v| -v).collect(),
                Op::Mul => {
                    let a = eval_rope(g, node.inputs[0], leaves);
                    let b = eval_rope(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::Add => {
                    let a = eval_rope(g, node.inputs[0], leaves);
                    let b = eval_rope(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x + y).collect()
                }
                // Metadata-only leading-1 rank-pad: same row-major order.
                Op::Reshape(_) => eval_rope(g, node.inputs[0], leaves),
                // Broadcast a `[1,..,1,seq,d]` inner block over leading dims:
                // the block (all of `input`, since the leading dims are 1)
                // tiles to fill the target — row-major `input[i % block]`.
                Op::BroadcastTo(target) => {
                    let input = eval_rope(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    (0..out_n).map(|i| input[i % input.len()]).collect()
                }
                Op::Slice { dim, start, len } => {
                    let input = eval_rope(g, node.inputs[0], leaves);
                    let in_dims = g.node(node.inputs[0]).shape.dims().to_vec();
                    let last = in_dims.len() - 1;
                    assert_eq!(*dim, last, "rope slices along the last axis");
                    let row = in_dims[last];
                    input
                        .chunks(row)
                        .flat_map(|r| r[*start..*start + *len].to_vec())
                        .collect()
                }
                Op::Concat { dim } => {
                    let a = eval_rope(g, node.inputs[0], leaves);
                    let b = eval_rope(g, node.inputs[1], leaves);
                    let a_last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    let b_last = *g.node(node.inputs[1]).shape.dims().last().unwrap();
                    let last = g.node(node.inputs[0]).shape.dims().len() - 1;
                    assert_eq!(*dim, last, "rope concats along the last axis");
                    let mut out = Vec::with_capacity(a.len() + b.len());
                    let mut ai = a.chunks(a_last);
                    let mut bi = b.chunks(b_last);
                    loop {
                        match (ai.next(), bi.next()) {
                            (Some(ra), Some(rb)) => {
                                out.extend_from_slice(ra);
                                out.extend_from_slice(rb);
                            }
                            (None, None) => break,
                            _ => panic!("concat row-count mismatch"),
                        }
                    }
                    out
                }
                other => panic!("eval_rope: unhandled op {other:?}"),
            }
        }

        /// Build a fused Rope node over `x [..,seq,d]`, `cos [seq,d]`,
        /// `sin [seq,d]`. Returns `(x, cos, sin, fused)`.
        fn rope_fused_node(
            g: &mut Graph,
            x_dims: &[usize],
        ) -> (NodeId, NodeId, NodeId, NodeId) {
            let rank = x_dims.len();
            let table_dims = [x_dims[rank - 2], x_dims[rank - 1]];
            let x_sh = Shape::from_dims(x_dims);
            let t_sh = Shape::from_dims(&table_dims);
            let x = g.push(Node { op: Op::Const, inputs: vec![], shape: x_sh.clone(), dtype: DType::F32 });
            let cos = g.push(Node { op: Op::Const, inputs: vec![], shape: t_sh.clone(), dtype: DType::F32 });
            let sin = g.push(Node { op: Op::Const, inputs: vec![], shape: t_sh, dtype: DType::F32 });
            let fused = g.push(Node {
                op: Op::Fused(FusedOps::ROPE, FusedOpParams::Rope),
                inputs: vec![x, cos, sin],
                shape: x_sh,
                dtype: DType::F32,
            });
            (x, cos, sin, fused)
        }

        /// T6 red (a): ONE recipe datum decomposes at BOTH rank 2 and rank 4
        /// — the shape/rank polymorphism the baked-shape legacy body never had
        /// — and its numerics match the FROZEN legacy builder bit-exactly
        /// under the shared reference interpreter.
        #[test]
        fn rope_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for x_dims in [vec![2usize, 4], vec![1, 2, 3, 8]] {
                let mut g = Graph::new();
                let (x, cos, sin, fused) = rope_fused_node(&mut g, &x_dims);
                let x_sh = Shape::from_dims(&x_dims);
                let rank = x_dims.len();
                let seq = x_dims[rank - 2];
                let d = x_dims[rank - 1];

                let new_root = rope::decompose(&mut g, fused, &FusedOpParams::Rope);
                assert_ne!(new_root, fused, "recipe decompose must fire at {x_dims:?}");
                assert_eq!(g.node(new_root).shape, x_sh, "rope is shape-preserving");
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root =
                    frozen_legacy_rope_decompose(&mut g, fused, &FusedOpParams::Rope);

                // Distinct, deterministic leaf data for x / cos / sin.
                let x_n: usize = x_dims.iter().product();
                let t_n = seq * d;
                let x_data: Vec<f64> = (0..x_n).map(|i| ((i as f64) * 0.31).sin() * 2.0 - 0.4).collect();
                let cos_data: Vec<f64> = (0..t_n).map(|i| ((i as f64) * 0.17 + 0.5).cos()).collect();
                let sin_data: Vec<f64> = (0..t_n).map(|i| ((i as f64) * 0.23 - 0.2).sin()).collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, x_data);
                leaves.insert(cos, cos_data);
                leaves.insert(sin, sin_data);

                let got = eval_rope(&g, new_root, &leaves);
                let want = eval_rope(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len(), "element count at {x_dims:?}");
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "rope[{i}] at {x_dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// T6 red (b): at a rank-RAISING broadcast (rank 4 — the real attention
        /// consumer's shape), the recipe emission is BYTE-IDENTICAL to legacy:
        /// D4 materializes both leading-1-padded prep `Reshape`s, so the whole
        /// 11-node DAG matches node-for-node (op, shape, dtype, wiring). This
        /// is the byte-identity guarantee that retires all backend risk for the
        /// live rope-decompose path.
        #[test]
        fn rope_recipe_is_byte_identical_to_legacy_at_rank_raise() {
            let mut g = Graph::new();
            let (_x, _cos, _sin, fused) = rope_fused_node(&mut g, &[1, 2, 3, 8]);
            let recipe_root = rope::decompose(&mut g, fused, &FusedOpParams::Rope);
            let legacy_root = frozen_legacy_rope_decompose(&mut g, fused, &FusedOpParams::Rope);
            super::assert_structural_eq(&g, recipe_root, legacy_root);
            // Both leading-1 prep Reshapes are present (byte-identical, 11 ops).
            let reachable = crate::topo_order_multi(&g, &[recipe_root]);
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert_eq!(reshapes, 2, "rank-raise materializes both legacy prep Reshapes (D4)");
        }

        /// T6 red (c): at EQUAL rank (x itself `[seq,d]`) the recipe emits the
        /// 9-node form — the resolver adds NO pad `Reshape` (D4 pads only on a
        /// rank-raise), eliding legacy's no-op `Reshape([seq,d]→[seq,d])`.
        /// Numerically identical (covered by the parity test), structurally
        /// leaner: 9 op nodes + 3 leaves, zero `Reshape`.
        #[test]
        fn rope_recipe_elides_the_noop_prep_reshape_at_equal_rank() {
            let mut g = Graph::new();
            let (_x, _cos, _sin, fused) = rope_fused_node(&mut g, &[2, 4]);
            let root = rope::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_ne!(root, fused, "recipe decompose fires");
            let reachable = crate::topo_order_multi(&g, &[root]);
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert_eq!(reshapes, 0, "equal-rank broadcast needs no prep Reshape (D4)");
            let op_nodes = reachable
                .iter()
                .filter(|&&n| !matches!(g.node(n).op, Op::Const))
                .count();
            assert_eq!(op_nodes, 9, "the 9-node rope recipe (2×Bcast/2×Slice/Neg/Concat/2×Mul/Add)");
        }

        /// T6 red (d): totality (G2) — a wrong params payload is a typed
        /// decline surfaced as a fixpoint, never a panic, and declines BEFORE
        /// any emission (no partial nodes). The legacy imperative body ignored
        /// `params` entirely and always decomposed; the recipe bridge gates on
        /// the projection, so a non-`Rope` payload now correctly no-ops.
        #[test]
        fn rope_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, _cos, _sin, fused) = rope_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = rope::decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    // rms_norm + layer_norm migration (Increment C slice 1, T7) --------------
    //
    // Both norms' imperative bodies become portable `PatternNode` DATA recipes.
    // Two forces at play beyond softmax/rope:
    //
    // * `eps` is an OPEN scalar slot. The recipe's `AddScalar` carries EMPTY
    //   `scalars`, so it is a slot template; the per-entry projection
    //   (`RmsNormLastDim { eps } → vec![eps]` / `LayerNormLastDim { eps } →
    //   vec![eps]`) supplies the live value, and the resolving emit fills the
    //   slot in pre-order. The eps-wiring tests below decompose the SAME op at
    //   two eps values and assert the realized outputs DIFFER accordingly — the
    //   proof that eps rides the projection→slot path, not a baked constant.
    //
    // * The keepdim restore is the RATIFIED D3 shrink-via-swap: `Reshape(keepdim)`
    //   → `Unsqueeze(axis_last = append)` (a node-TYPE change, metadata-only, so
    //   numerically bit-exact). `MeanDim(axis_last)` stays a rank-reducing mean.
    //   The parity tests evaluate the new recipe emission and the FROZEN legacy
    //   builder through one reference interpreter (which treats `Reshape` and
    //   `Unsqueeze` identically) and assert bit-exact equivalence at two ranks.
    //
    // Neither norm ever trips D4 (the keepdim restore rebuilds rank BEFORE the
    // broadcast, so every `BroadcastTo` operand already matches its target's
    // rank — no leading-1 pad `Reshape` is materialized).
    mod norm_recipe {
        use super::super::*;
        use super::{frozen_legacy_layer_norm_decompose, frozen_legacy_rms_norm_decompose};
        use crate::registry::{FusedOps, layer_norm_last_dim, rms_norm_last_dim};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the norm primitive vocabulary
        /// (Const leaves, `Sqr`, last-axis `MeanDim`, metadata-only keepdim
        /// restores `Unsqueeze`/`Reshape`, `AddScalar`, `Sqrt`, last-dim
        /// `BroadcastTo`, elementwise `Sub`/`Div`). BOTH parity sides run
        /// through it with identical in-order arithmetic, so a bit-exact assert
        /// isolates recipe STRUCTURE (the `Unsqueeze`-vs-`Reshape` swap can't
        /// perturb it). Not code evaluation: a closed match over our own `Op`.
        fn eval_norm(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            let node = g.node(id);
            match &node.op {
                Op::Const => leaves.get(&id).expect("leaf data provided").clone(),
                Op::Sqr => eval_norm(g, node.inputs[0], leaves).iter().map(|v| v * v).collect(),
                Op::Sqrt => eval_norm(g, node.inputs[0], leaves).iter().map(|v| v.sqrt()).collect(),
                Op::AddScalar(e) => {
                    eval_norm(g, node.inputs[0], leaves).iter().map(|v| v + e).collect()
                }
                Op::Sub => {
                    let a = eval_norm(g, node.inputs[0], leaves);
                    let b = eval_norm(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x - y).collect()
                }
                Op::Div => {
                    let a = eval_norm(g, node.inputs[0], leaves);
                    let b = eval_norm(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x / y).collect()
                }
                // Last-axis mean — rank-reducing; identical fold both spellings.
                Op::MeanDim(_) => {
                    let input = eval_norm(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().sum::<f64>() / last as f64)
                        .collect()
                }
                // Metadata-only keepdim restores (the D3 swap and its legacy
                // twin evaluate identically here).
                Op::Unsqueeze { .. } | Op::Reshape(_) => eval_norm(g, node.inputs[0], leaves),
                // Broadcast a keepdim `[.., 1]` tensor back along the last axis.
                Op::BroadcastTo(target) => {
                    let input = eval_norm(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    let last = *target.dims().last().unwrap();
                    assert_eq!(
                        input.len() * last,
                        out_n,
                        "broadcast is a last-dim repeat in these graphs",
                    );
                    input
                        .iter()
                        .flat_map(|&v| std::iter::repeat(v).take(last))
                        .collect()
                }
                other => panic!("eval_norm: unhandled op {other:?}"),
            }
        }

        /// Build a fused RmsNormLastDim node over `x [dims]`, carrying `eps`.
        /// Returns `(x, fused)`.
        fn rms_norm_fused_node(g: &mut Graph, dims: &[usize], eps: f64) -> (NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::RMS_NORM_LAST_DIM,
                    FusedOpParams::RmsNormLastDim { eps },
                ),
                inputs: vec![x],
                shape: sh,
                dtype: DType::F32,
            });
            (x, fused)
        }

        /// Build a fused LayerNormLastDim node over `x [dims]`, carrying `eps`.
        /// Returns `(x, fused)`.
        fn layer_norm_fused_node(g: &mut Graph, dims: &[usize], eps: f64) -> (NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::LAYER_NORM_LAST_DIM,
                    FusedOpParams::LayerNormLastDim { eps },
                ),
                inputs: vec![x],
                shape: sh,
                dtype: DType::F32,
            });
            (x, fused)
        }

        /// T7 red (a, rms): ONE recipe datum decomposes at BOTH rank 2 and rank
        /// 3 (the polymorphism the baked-shape legacy body never had), and its
        /// numerics match the FROZEN legacy builder bit-exactly.
        #[test]
        fn rms_norm_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (x, fused) = rms_norm_fused_node(&mut g, &dims, 1e-5);
                let sh = Shape::from_dims(&dims);
                let new_root = rms_norm_last_dim::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDim { eps: 1e-5 },
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(g.node(new_root).shape, sh, "rms_norm is shape-preserving");
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_rms_norm_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDim { eps: 1e-5 },
                );

                let n: usize = dims.iter().product();
                let data: Vec<f64> =
                    (0..n).map(|i| ((i as f64) * 0.37).sin() * 3.0 - 0.5).collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, data);
                let got = eval_norm(&g, new_root, &leaves);
                let want = eval_norm(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "rms_norm[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// T7 red (a, layer): same polymorphism + bit-exact parity for the
        /// 11-node layer-norm recipe (with the `centered` subterm identity-
        /// shared between `Sqr` and the final `Div`).
        #[test]
        fn layer_norm_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (x, fused) = layer_norm_fused_node(&mut g, &dims, 1e-5);
                let sh = Shape::from_dims(&dims);
                let new_root = layer_norm_last_dim::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::LayerNormLastDim { eps: 1e-5 },
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(g.node(new_root).shape, sh, "layer_norm is shape-preserving");
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_layer_norm_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::LayerNormLastDim { eps: 1e-5 },
                );

                let n: usize = dims.iter().product();
                let data: Vec<f64> =
                    (0..n).map(|i| ((i as f64) * 0.29).cos() * 2.0 + 0.3).collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, data);
                let got = eval_norm(&g, new_root, &leaves);
                let want = eval_norm(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "layer_norm[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// T7 red (structural, rms): the keepdim restore is the D3 shrink-via-
        /// swap — `Unsqueeze` append, NOT a baked `Reshape(keepdim)`. This is
        /// the crisp discriminator against the pre-migration imperative body.
        #[test]
        fn rms_norm_recipe_uses_the_unsqueeze_keepdim_swap() {
            let mut g = Graph::new();
            let (_x, fused) = rms_norm_fused_node(&mut g, &[2, 4], 1e-5);
            let root = rms_norm_last_dim::decompose(
                &mut g,
                fused,
                &FusedOpParams::RmsNormLastDim { eps: 1e-5 },
            );
            assert_ne!(root, fused, "recipe decompose fires");
            let reachable = crate::topo_order_multi(&g, &[root]);
            let unsqueezes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Unsqueeze { .. }))
                .count();
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert_eq!(unsqueezes, 1, "keepdim restored via Unsqueeze append (D3 swap)");
            assert_eq!(reshapes, 0, "no baked keepdim Reshape after the D3 swap");
        }

        /// T7 red (structural, layer): both keepdim restores are `Unsqueeze`
        /// appends; zero `Reshape` (the equal-rank broadcasts add no D4 pad).
        #[test]
        fn layer_norm_recipe_uses_the_unsqueeze_keepdim_swap() {
            let mut g = Graph::new();
            let (_x, fused) = layer_norm_fused_node(&mut g, &[2, 4], 1e-5);
            let root = layer_norm_last_dim::decompose(
                &mut g,
                fused,
                &FusedOpParams::LayerNormLastDim { eps: 1e-5 },
            );
            assert_ne!(root, fused, "recipe decompose fires");
            let reachable = crate::topo_order_multi(&g, &[root]);
            let unsqueezes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Unsqueeze { .. }))
                .count();
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert_eq!(unsqueezes, 2, "both keepdim restores via Unsqueeze append (D3 swap)");
            assert_eq!(reshapes, 0, "no baked keepdim Reshape / no D4 pad after the swap");
            // The `centered` Sub is SHARED (Sqr input == final Div numerator):
            // 11 op nodes + 1 leaf = 12 reachable, not the 12-op unshared tree.
            let op_nodes = reachable
                .iter()
                .filter(|&&n| !matches!(g.node(n).op, Op::Const))
                .count();
            assert_eq!(op_nodes, 11, "11 op nodes with `centered` identity-shared");
        }

        /// T7 red (eps-wiring, rms): the eps rides the projection→open-slot
        /// path. Decomposing the SAME op at two eps values yields DIFFERENT
        /// realized outputs — impossible if eps were dropped or baked to a
        /// single constant. Small `x` (so `mean(x²) ≈ eps`) makes the eps
        /// choice materially move every element.
        #[test]
        fn rms_norm_recipe_eps_flows_through_the_open_slot() {
            let dims = [2usize, 4];
            let n: usize = dims.iter().product();
            let data: Vec<f64> =
                (0..n).map(|i| 0.001 * (((i as f64) * 0.37).sin() + 1.2)).collect();

            let realize = |eps: f64| -> Vec<f64> {
                let mut g = Graph::new();
                let (x, fused) = rms_norm_fused_node(&mut g, &dims, eps);
                let root = rms_norm_last_dim::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDim { eps },
                );
                let mut leaves = HashMap::new();
                leaves.insert(x, data.clone());
                eval_norm(&g, root, &leaves)
            };
            let a = realize(1e-5);
            let b = realize(1e-6);
            assert_eq!(a.len(), b.len());
            assert_ne!(
                a, b,
                "different eps must change the output — proves projection→slot, not a baked constant",
            );
        }

        /// T7 red (eps-wiring, layer): same proof for layer-norm's open slot.
        #[test]
        fn layer_norm_recipe_eps_flows_through_the_open_slot() {
            let dims = [2usize, 4];
            let n: usize = dims.iter().product();
            // Near-constant rows so the variance ≈ eps and the eps choice moves
            // the output materially.
            let data: Vec<f64> =
                (0..n).map(|i| 1.0 + 0.001 * ((i as f64) * 0.37).sin()).collect();

            let realize = |eps: f64| -> Vec<f64> {
                let mut g = Graph::new();
                let (x, fused) = layer_norm_fused_node(&mut g, &dims, eps);
                let root = layer_norm_last_dim::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::LayerNormLastDim { eps },
                );
                let mut leaves = HashMap::new();
                leaves.insert(x, data.clone());
                eval_norm(&g, root, &leaves)
            };
            let a = realize(1e-5);
            let b = realize(1e-6);
            assert_eq!(a.len(), b.len());
            assert_ne!(
                a, b,
                "different eps must change the output — proves projection→slot, not a baked constant",
            );
        }

        /// T7 red (totality, rms): a wrong params payload is a typed decline
        /// surfaced as a fixpoint (G2), never a panic, declining BEFORE any
        /// emission (no partial nodes).
        #[test]
        fn rms_norm_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, fused) = rms_norm_fused_node(&mut g, &[2, 4], 1e-5);
            let before = g.len();
            let out = rms_norm_last_dim::decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }

        /// T7 red (totality, layer): same fixpoint posture for layer-norm.
        #[test]
        fn layer_norm_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, fused) = layer_norm_fused_node(&mut g, &[2, 4], 1e-5);
            let before = g.len();
            let out = layer_norm_last_dim::decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    // softmax_last_dim_backward migration (Increment C slice 1, T8) ----------
    //
    // The 5-node imperative backward body `s · (g − sum(g·s, last, keepdim))`
    // becomes a portable `PatternNode` DATA recipe. Bind space: `0 = s` (the
    // forward softmax output), `1 = g` (the upstream gradient) — the order the
    // autograd `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)` edge emits. The
    // keepdim restore is the ratified D3 shrink-via-swap
    // (`ReduceSumTo(keepdim)` → `SumDim(axis_last)` + `Unsqueeze(axis_last =
    // append)`, node-TYPE change, numerically bit-exact) and the broadcast
    // targets `SameAs { operand: 0 }` over the Bind space (D2). D4 never fires
    // (the `Unsqueeze` rebuilds rank BEFORE the broadcast, so the broadcast
    // operand already matches its target's rank — no leading-1 pad `Reshape`).
    // This activates the registry's backward-helper edge END-TO-END on a data
    // recipe for the first time.
    mod softmax_backward_recipe {
        use super::super::*;
        use super::frozen_legacy_softmax_backward_decompose;
        use crate::registry::{FusedOps, softmax_last_dim_backward};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the softmax-backward primitive
        /// vocabulary (leaf-lookup FIRST, then `Mul`/`Sub`, last-axis
        /// `SumDim`/`ReduceSumTo`, metadata-only keepdim restore
        /// `Unsqueeze`/`Reshape`, last-dim `BroadcastTo`). Leaf-first lets ANY
        /// node stand in as a bound input — a `Const`, or the autograd path's
        /// forward-softmax (`Op::Fused`) and upstream nodes. BOTH parity sides
        /// run through it with identical in-order arithmetic, so a bit-exact
        /// assert isolates recipe STRUCTURE (the `SumDim`+`Unsqueeze`-vs-
        /// `ReduceSumTo` swap can't perturb it). Not code evaluation: a closed
        /// match over our own `Op`.
        fn eval_bwd(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::Mul => {
                    let a = eval_bwd(g, node.inputs[0], leaves);
                    let b = eval_bwd(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::Sub => {
                    let a = eval_bwd(g, node.inputs[0], leaves);
                    let b = eval_bwd(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x - y).collect()
                }
                // Last-axis sum — one arm per spelling pair, identical fold.
                Op::SumDim(_) | Op::ReduceSumTo(_) => {
                    let input = eval_bwd(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input.chunks(last).map(|row| row.iter().sum()).collect()
                }
                // Metadata-only keepdim restores (the D3 swap and its legacy
                // twin evaluate identically here).
                Op::Unsqueeze { .. } | Op::Reshape(_) => eval_bwd(g, node.inputs[0], leaves),
                // Broadcast a keepdim/reduced tensor back along the last axis.
                Op::BroadcastTo(target) => {
                    let input = eval_bwd(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    let last = *target.dims().last().unwrap();
                    assert_eq!(
                        input.len() * last,
                        out_n,
                        "broadcast is a last-dim repeat in these graphs",
                    );
                    input
                        .iter()
                        .flat_map(|&v| std::iter::repeat(v).take(last))
                        .collect()
                }
                other => panic!("eval_bwd: unhandled op {other:?}"),
            }
        }

        /// Build a fused SoftmaxLastDimBackward node over `s [dims]` (input 0,
        /// the forward output) and `g [dims]` (input 1, the upstream gradient).
        /// Returns `(s, g, fused)`.
        fn softmax_backward_fused_node(
            g: &mut Graph,
            dims: &[usize],
        ) -> (NodeId, NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let s = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
            let up = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                    FusedOpParams::SoftmaxLastDimBackward,
                ),
                inputs: vec![s, up],
                shape: sh,
                dtype: DType::F32,
            });
            (s, up, fused)
        }

        /// T8 red (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3
        /// (the polymorphism the baked-shape legacy body never had), and its
        /// numerics match the FROZEN legacy builder bit-exactly under the
        /// shared reference interpreter.
        #[test]
        fn softmax_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (s, up, fused) = softmax_backward_fused_node(&mut g, &dims);
                let sh = Shape::from_dims(&dims);
                let new_root = softmax_last_dim_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::SoftmaxLastDimBackward,
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(g.node(new_root).shape, sh, "softmax backward is shape-preserving");
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_softmax_backward_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::SoftmaxLastDimBackward,
                );

                let n: usize = dims.iter().product();
                let s_data: Vec<f64> =
                    (0..n).map(|i| ((i as f64) * 0.29).sin() * 0.5 + 0.5).collect();
                let g_data: Vec<f64> =
                    (0..n).map(|i| ((i as f64) * 0.53).cos() * 2.0 - 0.3).collect();
                let mut leaves = HashMap::new();
                leaves.insert(s, s_data);
                leaves.insert(up, g_data);
                let got = eval_bwd(&g, new_root, &leaves);
                let want = eval_bwd(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "softmax_backward[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// T8 red (structural): the keepdim restore is the D3 shrink-via-swap —
        /// `SumDim(last)` + `Unsqueeze` append, NOT the baked
        /// `ReduceSumTo(keepdim)`. The crisp discriminator against the
        /// pre-migration imperative body; the backward root is the outer `Mul`.
        #[test]
        fn softmax_backward_recipe_uses_the_sumdim_unsqueeze_swap() {
            let mut g = Graph::new();
            let (_s, _up, fused) = softmax_backward_fused_node(&mut g, &[2, 4]);
            let root = softmax_last_dim_backward::decompose(
                &mut g,
                fused,
                &FusedOpParams::SoftmaxLastDimBackward,
            );
            assert_ne!(root, fused, "recipe decompose fires");
            assert!(matches!(g.node(root).op, Op::Mul), "backward root is the outer Mul");
            let reachable = crate::topo_order_multi(&g, &[root]);
            let sumdims = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::SumDim(_)))
                .count();
            let unsqueezes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Unsqueeze { .. }))
                .count();
            let reduce_sum_tos = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::ReduceSumTo(_)))
                .count();
            assert_eq!(sumdims, 1, "the reduce is SumDim(last) — the D3 swap");
            assert_eq!(unsqueezes, 1, "keepdim restored via Unsqueeze append");
            assert_eq!(reduce_sum_tos, 0, "no baked keepdim ReduceSumTo after the swap");
        }

        /// T8 red (totality): a wrong params payload is a typed decline
        /// surfaced as a fixpoint (G2), never a panic, declining BEFORE any
        /// emission. (The pre-migration imperative body IGNORED params and
        /// always decomposed; the recipe bridge's `scalars(params)` projection
        /// is what makes a wrong payload a fixpoint.)
        #[test]
        fn softmax_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_s, _up, fused) = softmax_backward_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = softmax_last_dim_backward::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }

        /// T8 red (autograd path): the `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD)`
        /// edge exercised END-TO-END. Build a softmax forward, backprop; the
        /// input gradient node is `Op::Fused(SOFTMAX_LAST_DIM_BACKWARD)` over
        /// `[y, upstream]`; decomposing it fires the MIGRATED recipe (the D3
        /// SumDim spelling) and matches the frozen legacy numerically — the
        /// "realize" leg, via the reference interpreter feeding synthetic leaf
        /// data on the two bound inputs (`y = s`, `upstream = g`).
        #[test]
        fn softmax_backward_reaches_the_recipe_through_autograd() {
            let dev: std::sync::Arc<dyn fuel_backend_contract::DynBackendDevice> =
                std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice);
            let x = crate::Tensor::from_f32(
                vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
                Shape::from_dims(&[2, 3]),
                &dev,
            );
            let y = x.softmax_last_dim();
            let y_id = y.id();
            let grads = y.backward();
            let g_x = grads.get(&x).expect("softmax has an input gradient");
            let handle = g_x.graph();
            let bwd_id = g_x.id();

            // The input-gradient node IS the registry backward fused op over
            // `[y, upstream]` (x feeds only the softmax, so no accumulation Add).
            let up_id = {
                let gr = handle.read().unwrap();
                let node = gr.node(bwd_id).clone();
                match node.op {
                    Op::Fused(fid, params) => {
                        assert_eq!(
                            fid,
                            FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                            "autograd emits the registry backward fused op",
                        );
                        assert!(matches!(params, FusedOpParams::SoftmaxLastDimBackward));
                    }
                    other => panic!("expected the backward fused op, got {other:?}"),
                }
                assert_eq!(node.inputs[0], y_id, "backward input 0 = the forward softmax output");
                node.inputs[1]
            };

            // Decompose the SAME autograd backward node both ways (push-only
            // graph — the fused node survives), then compare numerically.
            let (new_root, legacy_root, sh) = {
                let mut gr = handle.write().unwrap();
                let sh = gr.node(bwd_id).shape.clone();
                let new_root = softmax_last_dim_backward::decompose(
                    &mut gr,
                    bwd_id,
                    &FusedOpParams::SoftmaxLastDimBackward,
                );
                let legacy_root = frozen_legacy_softmax_backward_decompose(
                    &mut gr,
                    bwd_id,
                    &FusedOpParams::SoftmaxLastDimBackward,
                );
                (new_root, legacy_root, sh)
            };

            let gr = handle.read().unwrap();
            assert_ne!(new_root, bwd_id, "the autograd backward node decomposes via the recipe");
            assert_eq!(gr.node(new_root).shape, sh, "shape-preserving");
            let reachable = crate::topo_order_multi(&gr, &[new_root]);
            assert!(
                reachable.iter().any(|&n| matches!(gr.node(n).op, Op::SumDim(_))),
                "the autograd path reaches the D3 SumDim spelling",
            );

            // Numeric parity (leaf-first interpreter over `[y = s, up = g]`).
            let n: usize = sh.dims().iter().product();
            let s_data: Vec<f64> =
                (0..n).map(|i| ((i as f64) * 0.31).sin() * 0.5 + 0.5).collect();
            let g_data: Vec<f64> =
                (0..n).map(|i| ((i as f64) * 0.47).cos() - 0.1).collect();
            let mut leaves = HashMap::new();
            leaves.insert(y_id, s_data);
            leaves.insert(up_id, g_data);
            let got = eval_bwd(&gr, new_root, &leaves);
            let want = eval_bwd(&gr, legacy_root, &leaves);
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "autograd softmax_backward[{i}]: recipe={a} vs legacy={b}",
                );
            }
        }
    }
}

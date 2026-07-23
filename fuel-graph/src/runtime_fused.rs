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
/// The inverse of `jit::op_to_tag`, over the **full first-order re-emit
/// vocabulary** (Convergence Increment A): every non-basis-gap, non-`Scan`,
/// non-`Fused` op — elementwise, comparison, `Where`, `Cast`, shape/layout
/// (Transpose/Permute/Reshape/BroadcastTo/(Un)squeeze/Slice/Concat/Flip/Roll/
/// Pad/Triu/Tril), reductions (SumDim/MeanDim/ReduceSumTo/ReduceMaxTo/CumSum/
/// SumAll/MaxAll/MinAll/MeanAll), `MatMul`, `Iota`, and indexing (IndexSelect/
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
        // MatMul + last-dim log-softmax (no structural params).
        T::MatMul => Op::MatMul,
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
    let mut out = attrs.clone();

    // target_shape_rel → target_shape (SameAs over the Bind space).
    if let Some(se) = &attrs.target_shape_rel {
        if !attrs.target_shape.is_empty() {
            return Err(RelAttrError::RelAbsConflict { field: "target_shape" });
        }
        match shape_expr::eval_shape(se, bind_shapes, &[]).map_err(RelAttrError::Expr)? {
            ShapeValue::Concrete(s) => {
                if let Some(&bad) = s.iter().find(|&&e| e < 0) {
                    return Err(RelAttrError::Negative { field: "target_shape", value: bad });
                }
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
        if attrs.slice_start.is_some() {
            return Err(RelAttrError::RelAbsConflict { field: "slice_start" });
        }
        out.slice_start = Some(eval_dim_field(d, "slice_start")?);
        out.slice_start_rel = None;
    }
    if let Some(d) = &attrs.slice_len_rel {
        if attrs.slice_len.is_some() {
            return Err(RelAttrError::RelAbsConflict { field: "slice_len" });
        }
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
            T::SumDim | T::MeanDim | T::CumSum | T::Concat | T::Flip | T::Slice | T::Roll
            | T::IndexSelect | T::Gather | T::IndexAdd | T::ScatterAdd => {
                if attrs.axis.is_some() {
                    return Err(RelAttrError::RelAbsConflict { field: "axis" });
                }
                let a = resolve_axis(LAST, rank).map_err(RelAttrError::Expr)?;
                out.axis = Some(a as i64);
            }
            // `dims`-carrier: Unsqueeze APPENDS — dim == rank (`primitive_shape`
            // permits `dim == rank`; keepdim-restore spelling, D3).
            T::Unsqueeze => {
                if !attrs.dims.is_empty() {
                    return Err(RelAttrError::RelAbsConflict { field: "dims" });
                }
                out.dims = vec![rank as u8];
            }
            // `dims`-carrier: Squeeze drops the trailing axis = rank − 1.
            T::Squeeze => {
                if !attrs.dims.is_empty() {
                    return Err(RelAttrError::RelAbsConflict { field: "dims" });
                }
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
            // Convergence Increment A: full-parity (shape, dtype) via the single
            // source of truth (`primitive_shape`) — correct for shape-changing,
            // reducing, and dtype-changing ops, not just same-shape elementwise.
            // The Err arm is only reachable for a MALFORMED authored region (a
            // registration-validated region's ops all infer). Real never-panic
            // guarantee: emit always returns a node, never panics. Fall back to
            // operand[0]'s shape/dtype; and because `validate_representable`
            // checks `tag_to_op(op).is_some()` but NOT arity — and `emit_region`
            // is a public raw-region entry (candidate verification) that does not
            // re-validate — a zero-operand op has no operand shape to borrow, so
            // `.first()` (never `[0]`) guards the index and a degenerate rank-0
            // F32 node is emitted for that malformed leaf.
            let child_shapes: Vec<fuel_ir::Shape> =
                child_ids.iter().map(|&c| graph.node(c).shape.clone()).collect();
            let child_dtypes: Vec<fuel_ir::DType> =
                child_ids.iter().map(|&c| graph.node(c).dtype).collect();
            let (s, d) = crate::shape::primitive_shape(&prim, &child_shapes, &child_dtypes)
                .unwrap_or_else(|_| {
                    (
                        child_shapes.first().cloned().unwrap_or_else(|| fuel_ir::Shape::from_dims(&[])),
                        child_dtypes.first().copied().unwrap_or(fuel_ir::DType::F32),
                    )
                });
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
    // the registry `decompose` output — the migration oracle. `emit` does NOT
    // CSE-dedup shared subterms, so a shared oracle node compares structurally
    // against two identical emitted subtrees; `assert_structural_eq` is
    // recursive + order-sensitive (no commutative canonicalization — stricter
    // than `base_map_hash`), catching an operand-swap the hash would mask.

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

    #[test]
    fn emit_matches_softmax_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        // Oracle: registry decompose reads inputs[0] + shape + dtype off the node.
        let fused = g.push(Node { op: Op::Const, inputs: vec![x], shape: sh.clone(), dtype: DType::F32 });
        let oracle = crate::registry::softmax_last_dim::decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);

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
        let oracle = crate::registry::rope::decompose(&mut g, fused, &FusedOpParams::Rope);

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
        let oracle = crate::registry::layer_norm_last_dim::decompose(
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
        use fuel_kernel_seam_types::shape_expr::{Dim, LAST, ShapeExpr, ShapeExprError, SYMBOLIC};
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
}

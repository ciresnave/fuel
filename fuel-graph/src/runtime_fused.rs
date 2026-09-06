// SPDX-License-Identifier: MIT OR Apache-2.0
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
    if let Some(h) = hash
        && let Some(&existing) = idx.get(&h)
    {
        return Ok(existing);
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
    RUNTIME_FUSED_OPS
        .read()
        .unwrap()
        .get(idx)
        .map(|e| e.region.clone())
}

/// A runtime op's name (telemetry / `op_short_name` routing).
pub fn runtime_name(id: FusedOpId) -> Option<String> {
    if !id.is_runtime() {
        return None;
    }
    let idx = (id.0 - FusedOpId::RUNTIME_FUSED_BASE) as usize;
    RUNTIME_FUSED_OPS
        .read()
        .unwrap()
        .get(idx)
        .map(|e| e.name.clone())
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
/// vocabulary** (Convergence Increment A): every non-basis-gap, non-`Fused`
/// op — elementwise, comparison, `Where`, `Cast`, shape/layout
/// (Transpose/Permute/Reshape/BroadcastTo/(Un)squeeze/Slice/Concat/Flip/Roll/
/// Pad/Triu/Tril), reductions (SumDim/MaxDim/MeanDim/ReduceSumTo/ReduceMaxTo/
/// CumSum/SumAll/MaxAll/MinAll/MeanAll), `MatMul`, `Iota`, and indexing (IndexSelect/
/// Gather/IndexAdd/ScatterAdd) — plus the **`Op::Scan` multi-output structural
/// terminal** (Increment C, B1): `Scan` (params ride the `scan_*` carriers, body
/// sub-graph rides the operands), its `ScanPlaceholder` body holes, and the
/// `View` slot projection (`view_slot`) that reads the scan's `output_views`
/// bundle at emit time — and the scalar-carrying `MaskedFill` (Increment C
/// carriers, A2 — fill value on `attrs.scalars[0]`, dtype on `cast_dtype` when
/// present else a provisional F32 that `emit` re-resolves to operand[0]'s dtype)
/// and `PowI` (A3 — the i32 exponent on `scalars[0]`). Structural params are
/// decoded from the (extended) [`OpAttrs`]. Returns `None` (an honest miss,
/// rejected at registration) for ops with no first-order re-emission: `Clamp`
/// (no two-scalar carrier yet), fused/basis-gap tags, and any tag whose required
/// attrs are unset (e.g. `Iota` with no `target_shape`, `Scan` with no
/// `scan_bound`, `View` with no `view_slot`, `PowI` with no exponent, or
/// `MaskedFill` with no fill value / a dtype `Scalar` cannot represent).
fn tag_to_op(tag: OpTag, attrs: &OpAttrs) -> Option<Op> {
    use OpTag as T;
    use fuel_ir::DType;
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
        // PowI (Increment C carriers, A3): the i32 exponent rides
        // `attrs.scalars[0]` as an f64 — an EXACT round-trip for every i32
        // (`|n| < 2^53`), reconstructed via `as i32`. This is the same carrier
        // the §6.19 wire already commits to (the `to_canonical_bytes` PowI arm
        // serializes `scalars`) and mirrors `MaskedFill`'s baked-scalar posture:
        // the exponent is a BAKED pattern constant, NOT an open cursor slot
        // (`scalar_slot_arity(PowI) == 0`), so the recipe author supplies it and
        // the emitter never draws it from the params projection. An unset value
        // is an honest miss (`None`), never a defaulted `PowI(0)`.
        T::PowI => Op::PowI(*attrs.scalars.first()? as i32),

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
                let (canon_lhs, canon_rhs) =
                    matmul_roles(attrs.lhs_roles.len(), attrs.rhs_roles.len());
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
        T::Unsqueeze => Op::Unsqueeze {
            dim: *attrs.dims.first()? as usize,
        },
        T::Squeeze => Op::Squeeze {
            dim: *attrs.dims.first()? as usize,
        },
        T::Slice => Op::Slice {
            dim: attrs.axis? as usize,
            start: attrs.slice_start? as usize,
            len: attrs.slice_len? as usize,
        },
        T::Concat => Op::Concat {
            dim: attrs.axis? as usize,
        },
        T::Flip => Op::Flip {
            dim: attrs.axis? as usize,
        },
        T::Roll => Op::Roll {
            dim: attrs.axis? as usize,
            shift: attrs.roll_shift?,
        },
        T::Pad => Op::Pad {
            padding: attrs
                .pad_amounts
                .iter()
                .map(|&(b, e)| (b as usize, e as usize))
                .collect(),
            mode: match attrs.pad_mode? {
                0 => crate::PadMode::Constant,
                1 => crate::PadMode::Reflect,
                2 => crate::PadMode::Replicate,
                _ => return None,
            },
            value: attrs.pad_value.unwrap_or(0.0),
        },
        T::Triu => Op::Triu {
            diagonal: attrs.axis?,
        },
        T::Tril => Op::Tril {
            diagonal: attrs.axis?,
        },
        // Reductions (dim rides `axis`; keepdim reductions ride `target_shape`).
        T::SumDim => Op::SumDim(attrs.axis? as usize),
        T::MaxDim => Op::MaxDim(attrs.axis? as usize),
        T::MeanDim => Op::MeanDim(attrs.axis? as usize),
        T::SumAll => Op::SumAll,
        T::MaxAll => Op::MaxAll,
        T::MinAll => Op::MinAll,
        T::MeanAll => Op::MeanAll,
        T::CumSum => Op::CumSum {
            dim: attrs.axis? as usize,
        },
        // Value source leaf (len rides `target_shape` as a 1-element shape).
        T::Iota => Op::Iota {
            len: *attrs.target_shape.first()? as usize,
        },
        // Indexing (dim rides `axis`).
        T::IndexSelect => Op::IndexSelect {
            dim: attrs.axis? as usize,
        },
        T::Gather => Op::Gather {
            dim: attrs.axis? as usize,
        },
        T::IndexAdd => Op::IndexAdd {
            dim: attrs.axis? as usize,
        },
        T::ScatterAdd => Op::ScatterAdd {
            dim: attrs.axis? as usize,
        },

        // --- Op::Scan structural re-emit (Increment C, B1) ---
        // A scan's body sub-graph rides the node's operands (the trailing
        // inputs); this arm reconstructs only the `Op::Scan` params. The body
        // holes re-emit through the `ScanPlaceholder` arm below. `Op::Scan`
        // stays a base-map terminal (no native kernel, no `LoweringRule`) — the
        // recipe just re-emits it so a decompose can round-trip through data.
        T::Scan => Op::Scan {
            n_xs: attrs.scan_n_xs? as usize,
            bound: attrs.scan_bound? as usize,
            emit: match attrs.scan_emit? {
                0 => crate::ScanEmit::All,
                1 => crate::ScanEmit::Final,
                _ => return None,
            },
            // Phase-1 `ScanPredicate` is a marker (the predicate DAG rides the
            // trailing `pred_exit` operand); `Some(true)` re-emits the marker.
            early_exit: if attrs.scan_early_exit.unwrap_or(false) {
                Some(crate::ScanPredicate)
            } else {
                None
            },
        },
        // A body hole of `Op::Scan` (a childless leaf): `role` + `index` ride
        // the dedicated `scan_role`/`scan_index` carriers.
        T::ScanPlaceholder => Op::ScanPlaceholder {
            role: match attrs.scan_role? {
                fuel_kernel_seam_types::SCAN_ROLE_CARRY => crate::ScanRole::Carry,
                fuel_kernel_seam_types::SCAN_ROLE_ELEM => crate::ScanRole::Elem,
                _ => return None,
            },
            index: attrs.scan_index? as usize,
        },
        // Multi-output slot projection. The slot's shape/dtype are NOT decoded
        // here (they come from the producer's `output_views` bundle at emit
        // time, mirroring `Graph::view`); this arm reconstructs only the op.
        T::View => Op::View {
            slot: attrs.view_slot?,
        },

        // Nested fused op carried AS-IS (Increment C, C-T2, mechanism 2a). The
        // `fused_op` selector names the registry entry (mirroring `cast_dtype`'s
        // name-string precedent); resolve it to a `FusedOpId` and reconstruct
        // the param-less `Op::Fused(fid, params)` via the small `fid -> params`
        // map. An unset selector, an unknown name, or a param-carrying id (not in
        // the map — its per-instance params can't be recovered from a name alone)
        // is a surfaced honest miss (`None`), never a crash. The nested node's
        // shape/dtype are computed at emit time from the entry's shape/dtype
        // rules (`primitive_shape` honest-misses `Fused`); this arm reconstructs
        // only the op. Fuel-INTERNAL — a `Fused` node never reaches the §6.19 wire.
        T::Fused => {
            let fid =
                crate::registry::default_registry().id_for_name(attrs.fused_op.as_deref()?)?;
            Op::Fused(fid, fused_params_for(fid)?)
        }

        // MaskedFill (Increment C carriers, A2): the fill VALUE rides
        // `attrs.scalars[0]` (the `op_to_attrs` projection); its dtype rides
        // `cast_dtype` when present (a concrete round-trip), else a provisional
        // F32 that `emit` re-resolves to operand[0]'s dtype (the
        // dtype-polymorphic recipe path — the byte executor derives `fill_bytes`
        // at the filled tensor's width, so the Scalar dtype must match). An
        // unset value or a dtype `Scalar` cannot represent is an honest miss.
        T::MaskedFill => {
            let v = *attrs.scalars.first()?;
            let dtype = match attrs.cast_dtype.as_deref() {
                Some(name) => DType::from_str(name).ok()?,
                None => DType::F32,
            };
            Op::MaskedFill {
                value: masked_fill_scalar(v, dtype)?,
            }
        }

        // Honest misses (rejected at registration): Clamp (no two-scalar
        // carrier yet), and any tag whose required attrs are unset or that is
        // added to OpTag later.
        _ => return None,
    })
}

/// The `fid -> params` map for the nested fused ops a recipe carries as-is
/// (Increment C, C-T2, mechanism 2a): the two **param-less** attention-trio
/// nested softmaxes — `SoftmaxLastDim` (forward, used by `flash_attn` /
/// `paged_attn`) and `SoftmaxLastDimBackward` (used by `flash_attn_backward`).
/// A param-carrying fused op is NOT recoverable from the name-only `fused_op`
/// selector — its per-instance params (softmax_scale, block_size, …) have no
/// carrier here — so any other id is a surfaced honest miss (`None`), never a
/// crash. Extend this map only when a new *param-less* nested fused op needs to
/// ride a recipe.
fn fused_params_for(fid: FusedOpId) -> Option<FusedOpParams> {
    use crate::registry::FusedOps;
    Some(match fid {
        FusedOps::SOFTMAX_LAST_DIM => FusedOpParams::SoftmaxLastDim,
        FusedOps::SOFTMAX_LAST_DIM_BACKWARD => FusedOpParams::SoftmaxLastDimBackward,
        _ => return None,
    })
}

/// Reconstruct a `MaskedFill` fill [`fuel_ir::Scalar`] from its `f64` value +
/// the FILLED tensor's dtype (the A2 carrier). `None` for a dtype a `Scalar`
/// cannot represent (the sub-byte dummy quant formats) — an honest miss, never
/// a panic (so [`fuel_ir::Scalar::from_f64`]'s dummy-dtype panic is unreachable
/// through this guard). The value rides `attrs.scalars[0]`; the dtype is the
/// filled tensor's (operand[0]) dtype, resolved at emit time.
fn masked_fill_scalar(value: f64, dtype: fuel_ir::DType) -> Option<fuel_ir::Scalar> {
    // Honest miss for any dtype without a scalar rep; the Result collapses
    // the old hand-written dummy-dtype guard.
    fuel_ir::Scalar::from_f64(value, dtype).ok()
}

/// Decode a target [`fuel_ir::Shape`] from `attrs.target_shape` (the shared
/// LOGICAL-shape carrier for Reshape/BroadcastTo/ReduceSumTo/ReduceMaxTo).
/// `None` for an unset (empty) target — an honest miss — UNLESS the
/// [`OpAttrs::rank0_target`] marker is set, which denotes the intentional
/// rank-0 (`[]`, scalar) shape (C-T1: a reduce-to-scalar loss tail such as
/// FSCE's `ReduceSumTo([])`). The marker is what distinguishes an authored
/// rank-0 target from the empty-`target_shape` wildcard/unset state.
fn shape_from_attr(attrs: &OpAttrs) -> Option<fuel_ir::Shape> {
    if attrs.target_shape.is_empty() {
        if attrs.rank0_target {
            return Some(fuel_ir::Shape::from_dims(&[]));
        }
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
    FrameNotExpressible {
        field: &'static str,
        frame: Vec<i64>,
        missing_ctor: u8,
    },
}

/// Whether `attrs` carries any shape-RELATIVE field (D2) — the emit fast-path
/// guard: rel-free attrs skip resolution entirely (zero behavior change for
/// existing concrete regions).
fn has_rel_attrs(attrs: &OpAttrs) -> bool {
    attrs.target_shape_rel.is_some()
        || attrs.slice_start_rel.is_some()
        || attrs.slice_len_rel.is_some()
        || attrs.axis_last
        || attrs.scalar_rel.is_some()
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
    if attrs.scalar_rel.is_some() && !attrs.scalars.is_empty() {
        return Some("scalars");
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
    if bind_shapes.contains(&frame) {
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
    use fuel_kernel_seam_types::shape_expr::{self, Dim, DimValue, LAST, ShapeValue, resolve_axis};
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
                    return Err(RelAttrError::Negative {
                        field: "target_shape",
                        value: bad,
                    });
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
                return Err(RelAttrError::SymbolicGap {
                    field: "target_shape",
                });
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

    // scalar_rel → scalars (DimExpr over the Bind space; the reduced_count
    // concept — `Dim::Extent { operand, axis: LAST }` = n = extent of that
    // operand's last axis, the norm-backward `MulScalar(n)` divisor). Rides the
    // SAME §6.20 `eval_dim`/`resolve_axis` evaluator as the slice_* rel fields,
    // resolved against the Bind space (not this op's children — a scalar rides no
    // operand's rank). Unlike an extent/offset, a scalar may be negative, so only
    // a symbolic Gap declines (§6.20-0004); every concrete value flows through.
    if let Some(d) = &attrs.scalar_rel {
        match shape_expr::eval_dim(d, bind_shapes, &[]).map_err(RelAttrError::Expr)? {
            DimValue::Concrete(v) => out.scalars = vec![v as f64],
            DimValue::Gap => return Err(RelAttrError::SymbolicGap { field: "scalars" }),
        }
        out.scalar_rel = None;
    }

    // axis_last → the per-tag axis carrier, resolved against operand[0]'s rank.
    if attrs.axis_last {
        let rank = child_shapes
            .first()
            .ok_or(RelAttrError::NoChildOperand)?
            .len();
        use OpTag as T;
        match tag {
            // `axis`-carrier tags: this op's LAST = rank − 1 via the shared
            // §6.20 resolver (typed AxisOutOfRange on a rank-0 operand).
            T::SumDim
            | T::MaxDim
            | T::MeanDim
            | T::CumSum
            | T::Concat
            | T::Flip
            | T::Slice
            | T::Roll
            | T::IndexSelect
            | T::Gather
            | T::IndexAdd
            | T::ScatterAdd => {
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
        PatternNode::Op {
            op,
            operands,
            attrs,
        } => {
            // A `scalar_rel` node is filled from an input SHAPE at emit time, NOT
            // from the params cursor — so it is never an open slot (mirrors a
            // baked-value node). Only an empty-`scalars`, no-`scalar_rel`
            // scalar-param op is a cursor slot.
            let own = if attrs.scalars.is_empty() && attrs.scalar_rel.is_none() {
                scalar_slot_arity(*op)
            } else {
                0
            };
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
        PatternNode::Op {
            op,
            operands,
            attrs,
        } => {
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
    if out.scalar_rel.take().is_some() && out.scalars.is_empty() {
        out.scalars = vec![1.0];
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
    emit(
        graph,
        &region,
        &inputs,
        &bind_shapes,
        &mut cursor,
        &mut Vec::new(),
    )
    .unwrap_or(node_id)
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
    emit(
        graph,
        region,
        inputs,
        &bind_shapes,
        &mut cursor,
        &mut Vec::new(),
    )
}

/// A graph [`fuel_ir::Shape`] as a §6.20 evaluator operand: per-axis extents
/// with a bounded-symbolic (`Extent::Range`) axis mapped to the
/// [`shape_expr::SYMBOLIC`] sentinel — so a rel attr over a symbolic extent
/// declines as [`RelAttrError::SymbolicGap`] (surfaced gap, §6.20-0004)
/// instead of silently resolving against the capacity bound.
fn shape_expr_operand(shape: &fuel_ir::Shape) -> Vec<i64> {
    use fuel_kernel_seam_types::shape_expr::SYMBOLIC;
    (0..shape.rank())
        .map(|a| {
            if shape.extent(a).is_dynamic() {
                SYMBOLIC
            } else {
                shape.dims()[a] as i64
            }
        })
        .collect()
}

/// The region's **Bind-space** shapes (`bind_shapes[i]` = `inputs[i]`'s shape)
/// in §6.20 operand form — what `ShapeExpr::SameAs`/`Dim::Extent` index.
fn bind_operand_shapes(graph: &Graph, inputs: &[NodeId]) -> Vec<Vec<i64>> {
    inputs
        .iter()
        .map(|&id| shape_expr_operand(&graph.node(id).shape))
        .collect()
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
        PatternNode::Op {
            op,
            operands,
            attrs,
        } => {
            // Identity-share: a slot-free subtree already emitted in THIS call
            // re-uses its node (see the fn doc). Checked before the cursor
            // fill — a slot-free subtree never moves the cursor, so a hit
            // cannot misalign later slots.
            let sharable = count_scalar_slots(node) == 0;
            if sharable && let Some(&(_, id)) = memo.iter().find(|(p, _)| *p == node) {
                return Ok(id);
            }
            // Fill an open scalar slot from the cursor in PRE-order (before
            // descending into operands) — the same canonical order
            // `match_region_extract` recorded the live values in. (T3 note:
            // children are now EMITTED before the attrs are USED, but the
            // cursor fill stays right here, before the descent — the cursor
            // order is authoring order, not emission order.)
            // A `scalar_rel` node is filled from an input shape by the rel-attr
            // resolver below (NOT the params cursor) — matching
            // `count_scalar_slots`, which does not count it as a slot, so it must
            // not consume a cursor value here either.
            let arity = scalar_slot_arity(*op);
            let filled;
            let attrs = if attrs.scalars.is_empty() && arity > 0 && attrs.scalar_rel.is_none() {
                let (take, rest) = scalars.split_at(arity);
                *scalars = rest;
                filled = OpAttrs {
                    scalars: take.to_vec(),
                    ..attrs.clone()
                };
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
            let mut child_shapes: Vec<fuel_ir::Shape> = child_ids
                .iter()
                .map(|&c| graph.node(c).shape.clone())
                .collect();
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
            let mut prim =
                tag_to_op(*op, attrs).expect("region validated re-emittable at registration");
            // A `MaskedFill` fill Scalar must carry the FILLED tensor's
            // (operand[0]) dtype — the byte executor derives `fill_bytes` at
            // that width. A recipe authors the value dtype-polymorphically (no
            // `cast_dtype`), so re-resolve the Scalar at operand[0]'s emitted
            // dtype here, mirroring the imperative `Scalar::one(dtype)`. Concrete
            // round-trips already carry the matching dtype (identity); a dtype a
            // Scalar cannot represent leaves the provisional value untouched (an
            // inert dead node, never a panic — the executor rejects it later).
            if let Op::MaskedFill { value } = &prim
                && let Some(&dt) = child_dtypes.first()
                && value.dtype() != dt
                && let Some(fixed) = masked_fill_scalar(value.to_f64(), dt)
            {
                prim = Op::MaskedFill { value: fixed };
            }
            // D4: a `BroadcastTo` whose target rank EXCEEDS its operand's rank
            // first materializes the legacy `Reshape` pad (1-padded left,
            // right-aligned — byte-identical to `registry::rope`'s hand-built
            // broadcast prep, since `check_broadcast_compatible` is
            // right-aligned). Recipes stay free of rank-dependent nodes while
            // the emitted graph matches the legacy imperative builders.
            // Applied uniformly (rel-resolved AND absolute targets); an
            // equal-rank broadcast is unchanged (no pad).
            if let Op::BroadcastTo(target) = &prim
                && let Some(cs) = child_shapes.first()
                && target.rank() > cs.rank()
            {
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
            // Shape/dtype for the emitted node. Most ops use the single source
            // of truth (`primitive_shape`) — a pure function of operand shapes,
            // correct for shape-changing/reducing/dtype-changing ops. Two
            // structural multi-output terminals are NOT such a function and are
            // resolved here with the graph in hand, mirroring `NodeHandle::scan` /
            // `Graph::view` (Increment C, B1):
            //   * `Op::Scan` — the node's PRIMARY (slot-0) shape is the stacked
            //     ys `[bound] ++ body_y`, and its 2-slot `output_views` bundle
            //     (slot 0 = stacked ys, slot 1 = final carry) is attached AFTER
            //     the push so downstream `Op::View`s can read it.
            //   * `Op::View` — the slot's shape/dtype come from the producer's
            //     `output_views[slot]` (set when the producing `Scan` emitted).
            // Both fall back to operand[0]'s shape/dtype on a malformed authored
            // region (never a panic) — the same posture as the primitive_shape
            // fallback (`.first()`, never `[0]`, guards a zero-operand leaf).
            fn fallback_sd(
                cs: &[fuel_ir::Shape],
                cd: &[fuel_ir::DType],
            ) -> (fuel_ir::Shape, fuel_ir::DType) {
                (
                    cs.first()
                        .cloned()
                        .unwrap_or_else(|| fuel_ir::Shape::from_dims(&[])),
                    cd.first().copied().unwrap_or(fuel_ir::DType::F32),
                )
            }
            let mut scan_bundle: Option<Vec<fuel_ir::storage::OutputViewSpec>> = None;
            let (s, d) = match &prim {
                Op::Scan {
                    n_xs,
                    bound,
                    early_exit,
                    ..
                } => {
                    // inputs = [init_carry, xs(n_xs), consts.., body_new_carry,
                    // body_y, [pred_exit]] — the Phase-1 lax.scan encoding.
                    let n_trailing = if early_exit.is_some() { 3 } else { 2 };
                    let len = child_ids.len();
                    if len >= 1 + *n_xs + n_trailing {
                        let carry_shape = child_shapes[0].clone();
                        let carry_dtype = child_dtypes[0];
                        let body_y_i = len - n_trailing + 1;
                        let y_shape = &child_shapes[body_y_i];
                        let y_dtype = child_dtypes[body_y_i];
                        let mut ys_dims = Vec::with_capacity(1 + y_shape.rank());
                        ys_dims.push(*bound);
                        ys_dims.extend_from_slice(y_shape.dims());
                        let ys_shape = fuel_ir::Shape::from_dims(&ys_dims);
                        scan_bundle = Some(vec![
                            fuel_ir::storage::OutputViewSpec::contiguous(y_dtype, ys_shape.clone()),
                            fuel_ir::storage::OutputViewSpec::contiguous(carry_dtype, carry_shape),
                        ]);
                        (ys_shape, y_dtype)
                    } else {
                        fallback_sd(&child_shapes, &child_dtypes)
                    }
                }
                Op::View { slot } => child_ids
                    .first()
                    .and_then(|&producer| {
                        graph
                            .output_views(producer)
                            .and_then(|v| v.get(*slot as usize))
                            .map(|spec| (spec.shape.clone(), spec.dtype))
                    })
                    .unwrap_or_else(|| fallback_sd(&child_shapes, &child_dtypes)),
                // A nested fused node carried as-is (Increment C, C-T2, mechanism
                // 2a). `primitive_shape` honest-misses `Fused` — a fused op is not
                // a first-order shape-inferable primitive — so its output frame
                // comes from the named registry entry's OWN `shape_rule` /
                // `dtype_rule` (e.g. softmax passes both through). Falls back to
                // operand[0] on an unregistered id (never a panic), the same
                // posture as the primitive_shape fallback.
                Op::Fused(fid, params) => match crate::registry::default_registry().entry(*fid) {
                    Some(e) => (
                        (e.shape_rule)(&child_shapes, params),
                        (e.dtype_rule)(&child_dtypes, params),
                    ),
                    None => fallback_sd(&child_shapes, &child_dtypes),
                },
                // A childless scan-body hole (`Op::ScanPlaceholder`). Its shape
                // is DECLARED on the recipe node (`target_shape` /
                // `target_shape_rel`, resolved above) rather than inferred from
                // (absent) operands, so the re-emitted body carries the same
                // per-step shapes the imperative `selective_scan`/`ssd_chunk_scan`
                // decompose stamps on its placeholders. This is load-bearing:
                // `unroll_scan` CLONES body interior nodes with their STORED
                // shapes, so a rank-0 fallback here would poison the unrolled
                // graph (e.g. `du = Mul(d_t, u_t)` would come out rank-0). Dtype =
                // the scan's working dtype = bind 0's (uniform-dtype scan inputs).
                // Both fall back to the operand/F32 default on a shapeless
                // authored placeholder (B1's round-trip recipes), never a panic.
                Op::ScanPlaceholder { .. } => (
                    shape_from_attr(attrs)
                        .unwrap_or_else(|| fallback_sd(&child_shapes, &child_dtypes).0),
                    inputs
                        .first()
                        .map(|&b| graph.node(b).dtype)
                        .unwrap_or_else(|| fallback_sd(&child_shapes, &child_dtypes).1),
                ),
                _ => crate::shape::primitive_shape(&prim, &child_shapes, &child_dtypes)
                    .unwrap_or_else(|_| fallback_sd(&child_shapes, &child_dtypes)),
            };
            let out = graph.push(Node {
                op: prim,
                inputs: child_ids,
                shape: s,
                dtype: d,
            });
            // Attach the scan's 2-slot bundle (mirrors `NodeHandle::scan`). A
            // malformed spec (compose/validate failure) leaves the producer
            // bundle-less — a later `View` then falls back — never a panic.
            if let Some(specs) = scan_bundle
                && let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&specs)
            {
                let _ = graph.set_output_views(out, std::sync::Arc::from(views.into_boxed_slice()));
            }
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
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
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
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
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
        assert_eq!(
            id1, id2,
            "an identical region must resolve to the same FusedOpId, not a duplicate"
        );
    }

    #[test]
    fn register_rejects_non_contiguous_binds() {
        // bind indices {0, 2} — missing 1.
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![
                PatternNode::Bind { index: 0 },
                PatternNode::Bind { index: 2 },
            ],
        };
        assert_eq!(
            register_runtime_fused("bad", region),
            Err(RuntimeFusedError::NonContiguousBinds(vec![0, 2]))
        );
    }

    #[test]
    fn register_rejects_unrepresentable_region() {
        // Convergence A made MatMul/shape/reduction ops representable, and
        // Increment C carriers made MaskedFill (A2) + PowI (A3) representable;
        // `Clamp` stays an honest miss (no two-scalar carrier yet), so it is the
        // current canonical still-unrepresentable tag.
        let region = PatternNode::Op {
            op: OpTag::Clamp,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        assert_eq!(
            register_runtime_fused("bad", region),
            Err(RuntimeFusedError::UnRepresentable(OpTag::Clamp))
        );
    }

    #[test]
    fn tag_to_op_reconstructs_shape_changing_ops() {
        use fuel_ir::Shape;
        // Slice{dim:1,start:2,len:3}
        let attrs = OpAttrs {
            axis: Some(1),
            slice_start: Some(2),
            slice_len: Some(3),
            ..OpAttrs::default()
        };
        assert!(matches!(
            super::tag_to_op(OpTag::Slice, &attrs),
            Some(Op::Slice {
                dim: 1,
                start: 2,
                len: 3
            })
        ));
        // Concat{dim:0}
        let attrs = OpAttrs {
            axis: Some(0),
            ..OpAttrs::default()
        };
        assert!(matches!(
            super::tag_to_op(OpTag::Concat, &attrs),
            Some(Op::Concat { dim: 0 })
        ));
        // Reshape([6])
        let attrs = OpAttrs {
            target_shape: vec![6],
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Reshape, &attrs),
            Some(Op::Reshape(Shape::from_dims(&[6])))
        );
        // BroadcastTo([2,3])
        let attrs = OpAttrs {
            target_shape: vec![2, 3],
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::BroadcastTo, &attrs),
            Some(Op::BroadcastTo(Shape::from_dims(&[2, 3])))
        );
        // ReduceMaxTo([2,1])
        let attrs = OpAttrs {
            target_shape: vec![2, 1],
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::ReduceMaxTo, &attrs),
            Some(Op::ReduceMaxTo(Shape::from_dims(&[2, 1])))
        );
    }

    #[test]
    fn shape_target_ops_represent_rank0_via_the_marker() {
        // C-T1: an INTENTIONAL rank-0 (`[]`) reduce/reshape/broadcast target.
        // A rank-0 shape has empty `target_shape` — the same empty state as an
        // unset/wildcard target — so the `rank0_target` marker disambiguates.
        use fuel_ir::Shape;
        let empty = Shape::from_dims(&[]);
        // Marker SET → the concrete rank-0 shape (RED before the shape_from_attr
        // rank0 arm: an empty `target_shape` honest-missed to `None`).
        let rank0 = OpAttrs {
            rank0_target: true,
            ..OpAttrs::default()
        };
        assert_eq!(super::shape_from_attr(&rank0), Some(empty.clone()));
        assert_eq!(
            super::tag_to_op(OpTag::ReduceSumTo, &rank0),
            Some(Op::ReduceSumTo(empty.clone()))
        );
        assert_eq!(
            super::tag_to_op(OpTag::ReduceMaxTo, &rank0),
            Some(Op::ReduceMaxTo(empty.clone()))
        );
        assert_eq!(
            super::tag_to_op(OpTag::Reshape, &rank0),
            Some(Op::Reshape(empty.clone()))
        );
        assert_eq!(
            super::tag_to_op(OpTag::BroadcastTo, &rank0),
            Some(Op::BroadcastTo(empty))
        );
        // Marker UNSET + empty target_shape stays an honest miss (wildcard).
        assert_eq!(super::shape_from_attr(&OpAttrs::default()), None);
        assert_eq!(
            super::tag_to_op(OpTag::ReduceSumTo, &OpAttrs::default()),
            None
        );
    }

    #[test]
    fn tag_to_op_reconstructs_reductions_dtype_and_matmul() {
        use fuel_ir::DType;
        assert!(matches!(
            super::tag_to_op(
                OpTag::MeanDim,
                &OpAttrs {
                    axis: Some(1),
                    ..OpAttrs::default()
                }
            ),
            Some(Op::MeanDim(1))
        ));
        assert!(matches!(
            super::tag_to_op(OpTag::MatMul, &OpAttrs::default()),
            Some(Op::MatMul)
        ));
        // Cast target dtype via name.
        let attrs = OpAttrs {
            cast_dtype: Some("f16".into()),
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Cast, &attrs),
            Some(Op::Cast(DType::F16))
        );
        // Comparison.
        assert!(matches!(
            super::tag_to_op(OpTag::Lt, &OpAttrs::default()),
            Some(Op::Lt)
        ));
    }

    #[test]
    fn tag_to_op_reconstructs_masked_fill() {
        // A2 (Increment C carriers): the MaskedFill re-emit carrier. Value on
        // `scalars[0]`, dtype on `cast_dtype` (the `op_to_attrs` projection) —
        // RED before A2 (MaskedFill was the `_ => return None` honest miss).
        use fuel_ir::Scalar;
        let attrs = OpAttrs {
            scalars: vec![-1.0],
            cast_dtype: Some("f16".into()),
            ..OpAttrs::default()
        };
        match super::tag_to_op(OpTag::MaskedFill, &attrs) {
            Some(Op::MaskedFill { value }) => {
                assert_eq!(value.dtype(), DType::F16, "dtype rides cast_dtype");
                assert_eq!(
                    value,
                    Scalar::from_f64(-1.0, DType::F16).unwrap(),
                    "value rides scalars[0]"
                );
            }
            other => panic!("expected MaskedFill, got {other:?}"),
        }
        // No fill value ⇒ honest miss (None), never a defaulted 0.
        assert_eq!(
            super::tag_to_op(OpTag::MaskedFill, &OpAttrs::default()),
            None
        );
        // A dtype `Scalar` cannot represent (sub-byte dummy) ⇒ honest miss.
        let dummy = OpAttrs {
            scalars: vec![1.0],
            cast_dtype: Some("f4".into()),
            ..OpAttrs::default()
        };
        assert_eq!(super::tag_to_op(OpTag::MaskedFill, &dummy), None);
        // dtype-polymorphic authoring (no cast_dtype) reconstructs a provisional
        // F32 that `emit` later re-resolves to operand[0]'s dtype.
        let poly = OpAttrs {
            scalars: vec![1.0],
            ..OpAttrs::default()
        };
        assert!(matches!(
            super::tag_to_op(OpTag::MaskedFill, &poly),
            Some(Op::MaskedFill { .. })
        ));
        // The fill value is a BAKED pattern constant, not a cursor slot.
        assert_eq!(super::scalar_slot_arity(OpTag::MaskedFill), 0);
    }

    #[test]
    fn tag_to_op_reconstructs_powi() {
        // A3 (Increment C carriers): the PowI i32-exponent re-emit carrier. The
        // exponent rides `scalars[0]` as an f64 (an EXACT round-trip for every
        // i32 — |n| < 2^53), reconstructed via `as i32`. RED before A3 (PowI was
        // in the `_ => return None` honest-miss set).
        assert!(matches!(
            super::tag_to_op(
                OpTag::PowI,
                &OpAttrs {
                    scalars: vec![3.0],
                    ..OpAttrs::default()
                }
            ),
            Some(Op::PowI(3))
        ));
        // Negative exponent (e.g. PowI(-1) = reciprocal — the exp==0 backward arm).
        assert!(matches!(
            super::tag_to_op(
                OpTag::PowI,
                &OpAttrs {
                    scalars: vec![-1.0],
                    ..OpAttrs::default()
                }
            ),
            Some(Op::PowI(-1))
        ));
        // No exponent ⇒ honest miss (None), never a defaulted PowI(0).
        assert_eq!(super::tag_to_op(OpTag::PowI, &OpAttrs::default()), None);
        // The exponent is a BAKED pattern constant (like MaskedFill's fill), not
        // a cursor slot — so it never draws from the params projection.
        assert_eq!(super::scalar_slot_arity(OpTag::PowI), 0);
    }

    #[test]
    fn masked_fill_region_registers_and_emits_with_operand_dtype() {
        // Before A2 this region was `UnRepresentable(MaskedFill)`; now it
        // registers, and `emit` resolves the fill Scalar to operand[0]'s dtype
        // (dtype-polymorphic — the imperative `Scalar::one(dtype)` behaviour).
        // No `clear_runtime_fused_for_tests()` here: this test asserts on the
        // EMITTED op (dtype/value), never on the registry state or the minted
        // id, so the clear served nothing — and calling it unserialized in this
        // parallel #[test] binary raced every other registry-touching test by
        // wiping ids mid-run (GAP-219; the fn's own doc requires callers to
        // serialize). Registering a unique region without a clean slate is fine.
        let region = PatternNode::Op {
            op: OpTag::MaskedFill,
            attrs: OpAttrs {
                scalars: vec![1.0],
                ..OpAttrs::default()
            },
            operands: vec![
                PatternNode::Bind { index: 0 },
                PatternNode::Bind { index: 1 },
            ],
        };
        register_runtime_fused("mf::poly", region.clone())
            .expect("MaskedFill region registers (was UnRepresentable before A2)");
        // Emit over an F16 tensor + U8 mask ⇒ MaskedFill with an F16 fill Scalar.
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F16,
        });
        let mask = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::U8,
        });
        let root = emit_region(&mut g, &region, &[x, mask], &[]);
        match &g.node(root).op {
            Op::MaskedFill { value } => {
                assert_eq!(
                    value.dtype(),
                    DType::F16,
                    "fill Scalar re-resolved to operand[0] dtype"
                );
                assert_eq!(
                    value.to_f64(),
                    1.0,
                    "fill VALUE preserved through the re-resolution"
                );
            }
            other => panic!("expected MaskedFill, got {other:?}"),
        }
        assert_eq!(g.node(root).dtype, DType::F16);
    }

    #[test]
    fn powi_region_registers_and_emits() {
        // Before A3 a bare PowI region was `UnRepresentable(PowI)`; now it
        // registers, and `emit` reconstructs the i32 exponent from `scalars[0]`.
        // No `clear_runtime_fused_for_tests()` here — see the masked_fill test
        // above: this test asserts on the emitted op, not registry state, and
        // an unserialized clear in a parallel binary raced other tests (GAP-219).
        let region = PatternNode::Op {
            op: OpTag::PowI,
            attrs: OpAttrs {
                scalars: vec![3.0],
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        register_runtime_fused("powi::cube", region.clone())
            .expect("PowI region registers (was UnRepresentable before A3)");
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(
            matches!(g.node(root).op, Op::PowI(3)),
            "emit reconstructs Op::PowI(3) from scalars[0]",
        );
        assert_eq!(
            g.node(root).shape,
            Shape::from_dims(&[2, 3]),
            "PowI is shape-preserving"
        );
        assert_eq!(g.node(root).dtype, DType::F32);
    }

    #[test]
    fn tag_to_op_matmul_resolves_canonical_roles() {
        // T9 (D5): explicit CANONICAL role vectors resolve to Op::MatMul. The
        // resolver checks role POSITIONS against the locked cell, not extents.
        let attrs = OpAttrs {
            lhs_roles: vec![1, 3],
            rhs_roles: vec![3, 2],
            ..OpAttrs::default()
        };
        assert!(matches!(
            super::tag_to_op(OpTag::MatMul, &attrs),
            Some(Op::MatMul)
        ));
        // Rank-4 canonical (leading Batch dims) also resolves — GQA-divisible
        // batch extents stay all-Batch (positions, not extents).
        let attrs4 = OpAttrs {
            lhs_roles: vec![0, 0, 1, 3],
            rhs_roles: vec![0, 0, 3, 2],
            ..OpAttrs::default()
        };
        assert!(matches!(
            super::tag_to_op(OpTag::MatMul, &attrs4),
            Some(Op::MatMul)
        ));
    }

    #[test]
    fn tag_to_op_matmul_empty_roles_implicit_accept() {
        // Empty roles = the rank-polymorphic recipe form → implicit-accept
        // (unchanged from today; recipes keep matmul implicit).
        assert!(matches!(
            super::tag_to_op(OpTag::MatMul, &OpAttrs::default()),
            Some(Op::MatMul)
        ));
    }

    #[test]
    fn tag_to_op_matmul_rejects_noncanonical_roles() {
        // Non-canonical role configs are a SURFACED honest miss (typed decline at
        // registration), never a crash.
        // (1) transposed lhs = [ContractedK, FreeM] = [3,1] instead of [1,3].
        let transposed = OpAttrs {
            lhs_roles: vec![3, 1],
            rhs_roles: vec![3, 2],
            ..OpAttrs::default()
        };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &transposed), None);
        // (2) multi-ContractedK on lhs = [3,3].
        let multi_k = OpAttrs {
            lhs_roles: vec![3, 3],
            rhs_roles: vec![3, 2],
            ..OpAttrs::default()
        };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &multi_k), None);
        // (3) FreeN-before-K on rhs = [FreeN, ContractedK] = [2,3] instead of [3,2].
        let freen_before_k = OpAttrs {
            lhs_roles: vec![1, 3],
            rhs_roles: vec![2, 3],
            ..OpAttrs::default()
        };
        assert_eq!(super::tag_to_op(OpTag::MatMul, &freen_before_k), None);
    }

    #[test]
    fn tag_to_op_reconstructs_max_dim() {
        // T4 (Increment C slice 1): OpTag::MaxDim → Op::MaxDim(axis), the
        // axis riding `attrs.axis` exactly like SumDim/MeanDim.
        assert!(matches!(
            super::tag_to_op(
                OpTag::MaxDim,
                &OpAttrs {
                    axis: Some(1),
                    ..OpAttrs::default()
                }
            ),
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
        let attrs = OpAttrs {
            axis_last: true,
            ..OpAttrs::default()
        };
        let resolved =
            super::resolve_rel_attrs(OpTag::MaxDim, &attrs, &[vec![2, 3, 4]], &[vec![2, 3, 4]])
                .expect("axis_last must resolve on MaxDim, not AxisLastUnsupported");
        assert_eq!(resolved.axis, Some(2), "rank-3 operand → LAST = axis 2");
        assert!(
            !resolved.axis_last,
            "rel carrier must be cleared post-resolve"
        );
    }

    #[test]
    fn tag_to_op_still_rejects_basis_gap_and_scan() {
        // qmatmul/conv flow through Op::Fused (no OpTag); Scan is higher-order.
        assert_eq!(
            super::tag_to_op(OpTag::Iota, &OpAttrs::default()),
            None,
            "Iota needs a len (target_shape) — empty attrs is a miss"
        );
    }

    // ---- C-T2 (Increment C): the nested-fused re-emit carrier (mechanism 2a) --

    /// `OpTag::Fused` reconstructs the param-less nested fused ops from the
    /// `fused_op` selector (name → fid → params), and honest-misses anything
    /// else. Born-red before the C-T2 `tag_to_op` arm: `Fused` falls to the
    /// `_ => return None` catch-all, so every `Some(..)` assertion below fails.
    #[test]
    fn tag_to_op_reconstructs_nested_fused() {
        use crate::registry::{FusedOpParams, FusedOps};
        let sm = OpAttrs {
            fused_op: Some("SoftmaxLastDim".into()),
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Fused, &sm),
            Some(Op::Fused(
                FusedOps::SOFTMAX_LAST_DIM,
                FusedOpParams::SoftmaxLastDim
            )),
            "SoftmaxLastDim selector reconstructs the param-less nested fused op",
        );
        let smb = OpAttrs {
            fused_op: Some("SoftmaxLastDimBackward".into()),
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Fused, &smb),
            Some(Op::Fused(
                FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                FusedOpParams::SoftmaxLastDimBackward,
            )),
            "SoftmaxLastDimBackward selector reconstructs its param-less nested fused op",
        );
        // Honest misses (never a crash): unset selector, unknown name, and a
        // param-carrying fused op (not round-trippable through the name-only
        // selector — its params can't be reconstructed from the map).
        assert_eq!(
            super::tag_to_op(OpTag::Fused, &OpAttrs::default()),
            None,
            "an unset fused_op selector is an honest miss",
        );
        let bogus = OpAttrs {
            fused_op: Some("NotARealFusedOp".into()),
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Fused, &bogus),
            None,
            "an unknown name is an honest miss"
        );
        let paged = OpAttrs {
            fused_op: Some("PagedAttn".into()),
            ..OpAttrs::default()
        };
        assert_eq!(
            super::tag_to_op(OpTag::Fused, &paged),
            None,
            "a registered but param-carrying fused op is an honest miss (not in the fid->params map)",
        );
    }

    /// A hand-built recipe carrying a nested `Op::Fused(SOFTMAX_LAST_DIM)` node
    /// validates as re-emittable and round-trips through `emit`: the emitted node
    /// is the real nested fused op, with shape/dtype from the registry entry's
    /// `shape_rule`/`dtype_rule` (softmax passes both through) since
    /// `primitive_shape` honest-misses `Fused`. Born-red before the C-T2 arms:
    /// `validate_recipe` rejects the region `UnRepresentable`.
    #[test]
    fn emit_reconstructs_nested_fused_shape_and_dtype() {
        use crate::registry::{FusedOpParams, FusedOps};
        use fuel_ir::{DType, Shape};
        let region = PatternNode::Op {
            op: OpTag::Fused,
            attrs: OpAttrs {
                fused_op: Some("SoftmaxLastDim".into()),
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        assert!(
            super::validate_recipe(&region).is_ok(),
            "the nested-fused recipe validates as re-emittable (C-T2 tag_to_op arm)",
        );
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3, 4]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert_eq!(
            g.node(root).op,
            Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            "the nested fused node is reconstructed as-is (2a)",
        );
        assert_eq!(
            g.node(root).shape,
            Shape::from_dims(&[2, 3, 4]),
            "shape from softmax shape_rule (passthrough), NOT primitive_shape (which misses Fused)",
        );
        assert_eq!(
            g.node(root).dtype,
            DType::F32,
            "dtype from softmax dtype_rule (passthrough)"
        );
        assert_eq!(
            g.node(root).inputs,
            vec![x],
            "the fused node's single operand is the bound input"
        );
    }

    #[test]
    fn validate_representable_now_accepts_a_slice_region() {
        // Region: Concat{0}(Neg(Slice{...}(bind0)), bind0) — the rope rotate-half shape.
        let region = PatternNode::Op {
            op: OpTag::Concat,
            attrs: OpAttrs {
                axis: Some(0),
                ..OpAttrs::default()
            },
            operands: vec![
                PatternNode::Op {
                    op: OpTag::Neg,
                    attrs: OpAttrs::default(),
                    operands: vec![PatternNode::Op {
                        op: OpTag::Slice,
                        attrs: OpAttrs {
                            axis: Some(0),
                            slice_start: Some(0),
                            slice_len: Some(1),
                            ..OpAttrs::default()
                        },
                        operands: vec![PatternNode::Bind { index: 0 }],
                    }],
                },
                PatternNode::Bind { index: 0 },
            ],
        };
        assert!(
            super::validate_representable(&region).is_ok(),
            "slice/concat region must now validate"
        );
    }

    #[test]
    fn emit_gets_shape_right_for_a_reduction_region() {
        use fuel_ir::{DType, Shape};
        // Region: ReduceSumTo([2,1])(bind0). Input [2,5] → output [2,1].
        let region = PatternNode::Op {
            op: OpTag::ReduceSumTo,
            attrs: OpAttrs {
                target_shape: vec![2, 1],
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 5]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::ReduceSumTo(_)));
        assert_eq!(
            g.node(root).shape,
            Shape::from_dims(&[2, 1]),
            "emit must use the reduced shape, not operand[0]"
        );
        assert_eq!(g.node(root).dtype, DType::F32);
    }

    #[test]
    fn emit_gets_dtype_right_for_a_cast_region() {
        use fuel_ir::{DType, Shape};
        // Region: Cast(F16)(bind0). Input F32 → output F16, same shape.
        let region = PatternNode::Op {
            op: OpTag::Cast,
            attrs: OpAttrs {
                cast_dtype: Some("f16".into()),
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 3]),
            dtype: DType::F32,
        });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::Cast(DType::F16)));
        assert_eq!(
            g.node(root).dtype,
            DType::F16,
            "emit must take Cast's target dtype, not operand[0]'s"
        );
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
        assert!(
            matches!(g.node(root).op, Op::Add),
            "emit returns a node, not a panic"
        );
    }

    #[test]
    fn decompose_region_re_emits_relu_add() {
        let id = register_runtime_fused("test::relu_add::decompose", relu_add_region()).unwrap();
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
        let b = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
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
        let id =
            register_runtime_fused("test::tanh_mul_scalar::slot", tanh_mul_scalar_slot_region())
                .expect("a slot template is registrable");
        let region = runtime_region(id).unwrap();
        assert_eq!(count_scalar_slots(&region), 1, "one open slot");
    }

    #[test]
    fn decompose_fills_slots_from_the_node_scalars() {
        let id =
            register_runtime_fused("test::tanh_mul_scalar::fill", tanh_mul_scalar_slot_region())
                .unwrap();
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
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
        let a = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: DType::F32,
        });
        // One slot, but the node carries NO scalars — malformed; must be a
        // no-op fixpoint (G2), never a panic.
        let fused = g.push(Node {
            op: Op::Fused(id, FusedOpParams::Runtime { scalars: vec![] }),
            inputs: vec![a],
            shape: s.clone(),
            dtype: DType::F32,
        });
        assert_eq!(
            decompose_region(&mut g, fused),
            fused,
            "mismatch ⇒ fixpoint"
        );
    }

    // ---- shape-derived scalar slot (A1: the reduced_count live-emission) -----

    /// `mul_scalar(a)` whose scalar is a SHAPE-DERIVED value: `scalar_rel =
    /// Extent(operand 0, LAST)` — the reduced_count of bind 0's last axis, filled
    /// from the input shape at emit time (NOT from the params cursor). This is
    /// the narrow carrier a norm-backward's `MulScalar(n = dims[last])` needs.
    fn mul_scalar_reduced_count_region() -> PatternNode {
        use fuel_kernel_seam_types::shape_expr::{Dim, LAST};
        PatternNode::Op {
            op: OpTag::MulScalar,
            attrs: OpAttrs {
                scalar_rel: Some(Dim::Extent {
                    operand: 0,
                    axis: LAST,
                }),
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        }
    }

    /// A1 born-red: a `scalar_rel` node is filled from an input SHAPE, so it is
    /// NOT a params-cursor slot (`count_scalar_slots == 0`) and its emitted value
    /// is the reduced_count of the resolving input's last axis. RED before the
    /// wiring: `count_scalar_slots` counts the empty-`scalars` MulScalar as one
    /// open slot, and `emit` ignores `scalar_rel` (fills from the cursor).
    #[test]
    fn scalar_rel_is_shape_derived_not_a_cursor_slot() {
        let region = mul_scalar_reduced_count_region();
        assert_eq!(
            count_scalar_slots(&region),
            0,
            "a scalar_rel node is shape-derived, never a params-cursor slot",
        );
        // Emit over x[3,5]: n = last extent = 5, with NO cursor scalars.
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 5]),
            dtype: DType::F32,
        });
        let root = try_emit_region(&mut g, &region, &[x], &[])
            .expect("scalar_rel resolves against the input shape");
        assert!(
            matches!(g.node(root).op, Op::MulScalar(v) if v == 5.0),
            "scalar_rel = Extent(0, LAST) over x[3,5] ⇒ MulScalar(5.0), got {:?}",
            g.node(root).op,
        );
        assert_eq!(g.node(root).inputs, vec![x]);
        // Rank-polymorphic: the SAME datum resolves to 7 over x[..,7].
        let mut g3 = Graph::new();
        let x3 = g3.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 4, 7]),
            dtype: DType::F32,
        });
        let root3 = try_emit_region(&mut g3, &region, &[x3], &[]).unwrap();
        assert!(
            matches!(g3.node(root3).op, Op::MulScalar(v) if v == 7.0),
            "same datum, x[2,4,7] ⇒ MulScalar(7.0), got {:?}",
            g3.node(root3).op,
        );
    }

    /// A1: a `scalar_rel` set together with a non-empty `scalars` is a typed
    /// rel-XOR-abs authoring conflict at emit (never a silent precedence).
    #[test]
    fn scalar_rel_and_baked_scalar_together_is_a_typed_conflict() {
        use fuel_kernel_seam_types::shape_expr::{Dim, LAST};
        let region = PatternNode::Op {
            op: OpTag::MulScalar,
            attrs: OpAttrs {
                scalars: vec![2.0],
                scalar_rel: Some(Dim::Extent {
                    operand: 0,
                    axis: LAST,
                }),
                ..OpAttrs::default()
            },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 5]),
            dtype: DType::F32,
        });
        assert_eq!(
            try_emit_region(&mut g, &region, &[x], &[]),
            Err(RelAttrError::RelAbsConflict { field: "scalars" }),
            "scalar_rel XOR scalars — both set is a typed decline",
        );
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
        PatternNode::Op {
            op,
            attrs,
            operands,
        }
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
        assert_eq!(
            na.shape, nb.shape,
            "shape mismatch at {:?} vs {:?}",
            na.op, nb.op
        );
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(
            na.inputs.len(),
            nb.inputs.len(),
            "arity mismatch at {:?}",
            na.op
        );
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
            op: Op::ReduceMaxTo(keepdim_shape.clone()),
            inputs: vec![x_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let mb_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![m_id],
            shape: x_shape.clone(),
            dtype,
        });
        let s_id = graph.push(Node {
            op: Op::Sub,
            inputs: vec![x_id, mb_id],
            shape: x_shape.clone(),
            dtype,
        });
        let e_id = graph.push(Node {
            op: Op::Exp,
            inputs: vec![s_id],
            shape: x_shape.clone(),
            dtype,
        });
        let d_id = graph.push(Node {
            op: Op::ReduceSumTo(keepdim_shape.clone()),
            inputs: vec![e_id],
            shape: keepdim_shape,
            dtype,
        });
        let db_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![d_id],
            shape: x_shape.clone(),
            dtype,
        });

        graph.push(Node {
            op: Op::Div,
            inputs: vec![e_id, db_id],
            shape: x_shape,
            dtype,
        })
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
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.shape.clone(),
                n.dtype,
            )
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
            op: Op::Reshape(broadcast_shape.clone()),
            inputs: vec![cos_id],
            shape: broadcast_shape.clone(),
            dtype,
        });
        let sin_reshaped_id = graph.push(Node {
            op: Op::Reshape(broadcast_shape.clone()),
            inputs: vec![sin_id],
            shape: broadcast_shape,
            dtype,
        });
        let cos_bcast_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![cos_reshaped_id],
            shape: x_shape.clone(),
            dtype,
        });
        let sin_bcast_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![sin_reshaped_id],
            shape: x_shape.clone(),
            dtype,
        });

        let mut half_dims = dims.clone();
        half_dims[last] = half;
        let half_shape = Shape::from_dims(&half_dims);

        let first_half_id = graph.push(Node {
            op: Op::Slice {
                dim: last,
                start: 0,
                len: half,
            },
            inputs: vec![x_id],
            shape: half_shape.clone(),
            dtype,
        });
        let second_half_id = graph.push(Node {
            op: Op::Slice {
                dim: last,
                start: half,
                len: half,
            },
            inputs: vec![x_id],
            shape: half_shape.clone(),
            dtype,
        });
        let neg_second_id = graph.push(Node {
            op: Op::Neg,
            inputs: vec![second_half_id],
            shape: half_shape,
            dtype,
        });
        let rotated_half_id = graph.push(Node {
            op: Op::Concat { dim: last },
            inputs: vec![neg_second_id, first_half_id],
            shape: x_shape.clone(),
            dtype,
        });

        let left_id = graph.push(Node {
            op: Op::Mul,
            inputs: vec![x_id, cos_bcast_id],
            shape: x_shape.clone(),
            dtype,
        });
        let right_id = graph.push(Node {
            op: Op::Mul,
            inputs: vec![rotated_half_id, sin_bcast_id],
            shape: x_shape.clone(),
            dtype,
        });

        graph.push(Node {
            op: Op::Add,
            inputs: vec![left_id, right_id],
            shape: x_shape,
            dtype,
        })
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
            op: Op::Sqr,
            inputs: vec![x_id],
            shape: x_shape.clone(),
            dtype,
        });
        let mean_id = graph.push(Node {
            op: Op::MeanDim(last),
            inputs: vec![sq_id],
            shape: reduced_shape,
            dtype,
        });
        let mean_kd_id = graph.push(Node {
            op: Op::Reshape(keepdim_shape.clone()),
            inputs: vec![mean_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let denom_sq_id = graph.push(Node {
            op: Op::AddScalar(eps),
            inputs: vec![mean_kd_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let denom_id = graph.push(Node {
            op: Op::Sqrt,
            inputs: vec![denom_sq_id],
            shape: keepdim_shape,
            dtype,
        });
        let denom_bcast_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![denom_id],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Div,
            inputs: vec![x_id, denom_bcast_id],
            shape: x_shape,
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
            op: Op::MeanDim(last),
            inputs: vec![x_id],
            shape: reduced_shape.clone(),
            dtype,
        });
        let mean_kd_id = graph.push(Node {
            op: Op::Reshape(keepdim_shape.clone()),
            inputs: vec![mean_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let mean_bcast_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![mean_kd_id],
            shape: x_shape.clone(),
            dtype,
        });
        let centered_id = graph.push(Node {
            op: Op::Sub,
            inputs: vec![x_id, mean_bcast_id],
            shape: x_shape.clone(),
            dtype,
        });
        let centered_sq_id = graph.push(Node {
            op: Op::Sqr,
            inputs: vec![centered_id],
            shape: x_shape.clone(),
            dtype,
        });
        let var_id = graph.push(Node {
            op: Op::MeanDim(last),
            inputs: vec![centered_sq_id],
            shape: reduced_shape,
            dtype,
        });
        let var_kd_id = graph.push(Node {
            op: Op::Reshape(keepdim_shape.clone()),
            inputs: vec![var_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let var_eps_id = graph.push(Node {
            op: Op::AddScalar(eps),
            inputs: vec![var_kd_id],
            shape: keepdim_shape.clone(),
            dtype,
        });
        let denom_id = graph.push(Node {
            op: Op::Sqrt,
            inputs: vec![var_eps_id],
            shape: keepdim_shape,
            dtype,
        });
        let denom_bcast_id = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![denom_id],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Div,
            inputs: vec![centered_id, denom_bcast_id],
            shape: x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::layer_norm_last_dim_backward::decompose` (the legacy ~20-node
    /// `MeanDim`/`Reshape(keepdim)`/`BroadcastTo` recompute of
    /// `grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))`), copied VERBATIM
    /// from that module @ `b967bdb1` before slice-2 replaced the live body with
    /// the data recipe. Sole consumer: the slice-2 numeric-parity oracle
    /// (`layer_norm_backward_recipe` below). Reads `inputs[0] = x`, `inputs[1] =
    /// g` (the upstream gradient) off the node — the order the autograd
    /// `BackwardKind::Fused(LAYER_NORM_LAST_DIM_BACKWARD)` edge emits.
    fn frozen_legacy_layer_norm_backward_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, g_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
        };
        let eps = match params {
            FusedOpParams::LayerNormLastDimBackward { eps } => *eps,
            // G2: total + never-panic — impossible params; return self.
            _ => return id,
        };
        let dims = x_shape.dims().to_vec();
        let last = dims.len() - 1;
        let mut kd = dims.clone();
        kd[last] = 1;
        let keepdim = Shape::from_dims(&kd);
        let mut rd = dims.clone();
        rd.remove(last);
        let reduced = Shape::from_dims(&rd);

        // reduce-mean over the last dim, keepdim, broadcast back to x_shape.
        let mean_b = |graph: &mut Graph, src: NodeId| -> NodeId {
            let m = graph.push(Node {
                op: Op::MeanDim(last),
                inputs: vec![src],
                shape: reduced.clone(),
                dtype,
            });
            let m_kd = graph.push(Node {
                op: Op::Reshape(keepdim.clone()),
                inputs: vec![m],
                shape: keepdim.clone(),
                dtype,
            });
            graph.push(Node {
                op: Op::BroadcastTo(x_shape.clone()),
                inputs: vec![m_kd],
                shape: x_shape.clone(),
                dtype,
            })
        };

        // xhat = (x − mean(x)) · istd ; istd = rsqrt(var + eps).
        let mean_x = mean_b(graph, x_id);
        let xc = graph.push(Node {
            op: Op::Sub,
            inputs: vec![x_id, mean_x],
            shape: x_shape.clone(),
            dtype,
        });
        let xc_sq = graph.push(Node {
            op: Op::Sqr,
            inputs: vec![xc],
            shape: x_shape.clone(),
            dtype,
        });
        let var = mean_b(graph, xc_sq);
        let var_eps = graph.push(Node {
            op: Op::AddScalar(eps),
            inputs: vec![var],
            shape: x_shape.clone(),
            dtype,
        });
        let istd = graph.push(Node {
            op: Op::Rsqrt,
            inputs: vec![var_eps],
            shape: x_shape.clone(),
            dtype,
        });
        let xhat = graph.push(Node {
            op: Op::Mul,
            inputs: vec![xc, istd],
            shape: x_shape.clone(),
            dtype,
        });

        // grad_x = istd · (g − mean(g) − xhat·mean(g·xhat)).
        let mean_g = mean_b(graph, g_id);
        let g_xhat = graph.push(Node {
            op: Op::Mul,
            inputs: vec![g_id, xhat],
            shape: x_shape.clone(),
            dtype,
        });
        let mean_gxh = mean_b(graph, g_xhat);
        let t1 = graph.push(Node {
            op: Op::Sub,
            inputs: vec![g_id, mean_g],
            shape: x_shape.clone(),
            dtype,
        });
        let t2 = graph.push(Node {
            op: Op::Mul,
            inputs: vec![xhat, mean_gxh],
            shape: x_shape.clone(),
            dtype,
        });
        let inner = graph.push(Node {
            op: Op::Sub,
            inputs: vec![t1, t2],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Mul,
            inputs: vec![istd, inner],
            shape: x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::rms_norm_last_dim_backward::decompose` (the legacy ~22-node
    /// `MeanDim`/`SumDim`/`Reshape(keepdim)`/`AddScalar`/`MulScalar(n)`/`Rsqrt`/
    /// `BroadcastTo`/`Mul`/`Sub`/`Div` spelling, with the `n = dims[last]`
    /// reduced-count baked as an absolute `MulScalar`), copied VERBATIM from that
    /// module @ `9d7a1380` before A1 replaced the live body with the shape-
    /// polymorphic data recipe. Sole consumer: the A1 numeric-parity oracle
    /// (`rms_norm_backward_recipe` below). Reads `inputs = [x, upstream]` + the
    /// output shape off the node — the convention the autograd
    /// `BackwardKind::Fused` edge emits.
    fn frozen_legacy_rms_norm_backward_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, up_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
        };
        let eps = match params {
            FusedOpParams::RmsNormLastDimBackward { eps } => *eps,
            // G2: total + never-panic — impossible params; return self.
            _ => return id,
        };
        let dims = x_shape.dims().to_vec();
        let last = dims.len() - 1;
        let n = dims[last] as f64;
        let mut kd = dims.clone();
        kd[last] = 1;
        let keepdim = Shape::from_dims(&kd);
        let mut rd = dims.clone();
        rd.remove(last);
        let reduced = Shape::from_dims(&rd);

        // denom = mean(x²) + eps  (keepdim).
        let sq = graph.push(Node {
            op: Op::Sqr,
            inputs: vec![x_id],
            shape: x_shape.clone(),
            dtype,
        });
        let mean = graph.push(Node {
            op: Op::MeanDim(last),
            inputs: vec![sq],
            shape: reduced.clone(),
            dtype,
        });
        let mean_kd = graph.push(Node {
            op: Op::Reshape(keepdim.clone()),
            inputs: vec![mean],
            shape: keepdim.clone(),
            dtype,
        });
        let denom_kd = graph.push(Node {
            op: Op::AddScalar(eps),
            inputs: vec![mean_kd],
            shape: keepdim.clone(),
            dtype,
        });
        // r_rms = rsqrt(denom), broadcast.
        let rrms_kd = graph.push(Node {
            op: Op::Rsqrt,
            inputs: vec![denom_kd],
            shape: keepdim.clone(),
            dtype,
        });
        let rrms_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![rrms_kd],
            shape: x_shape.clone(),
            dtype,
        });
        // s = sum(g·x, last)  (keepdim).
        let gx = graph.push(Node {
            op: Op::Mul,
            inputs: vec![up_id, x_id],
            shape: x_shape.clone(),
            dtype,
        });
        let s = graph.push(Node {
            op: Op::SumDim(last),
            inputs: vec![gx],
            shape: reduced,
            dtype,
        });
        let s_kd = graph.push(Node {
            op: Op::Reshape(keepdim.clone()),
            inputs: vec![s],
            shape: keepdim.clone(),
            dtype,
        });
        let s_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![s_kd],
            shape: x_shape.clone(),
            dtype,
        });
        // term = x·s / (n·denom).
        let ndenom_kd = graph.push(Node {
            op: Op::MulScalar(n),
            inputs: vec![denom_kd],
            shape: keepdim.clone(),
            dtype,
        });
        let ndenom_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![ndenom_kd],
            shape: x_shape.clone(),
            dtype,
        });
        let xs = graph.push(Node {
            op: Op::Mul,
            inputs: vec![x_id, s_b],
            shape: x_shape.clone(),
            dtype,
        });
        let term = graph.push(Node {
            op: Op::Div,
            inputs: vec![xs, ndenom_b],
            shape: x_shape.clone(),
            dtype,
        });
        // grad_x = r_rms · (g − term).
        let inner = graph.push(Node {
            op: Op::Sub,
            inputs: vec![up_id, term],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Mul,
            inputs: vec![rrms_b, inner],
            shape: x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::reduce_max_to_backward::decompose` (the legacy 9-node
    /// `ReduceMaxTo`→`BroadcastTo`→`Equal(U8)`→`MaskedFill`→`ReduceSumTo`→`Div`→
    /// `BroadcastTo`→`Mul` spelling, sharing the single `mask_f`), copied VERBATIM
    /// from that module @ `94c69ec7` before A2 replaced the live body with the
    /// data recipe. Sole consumer: the A2 numeric-parity oracle
    /// (`reduce_max_to_backward_recipe` below). Reads `inputs[0] = x` and
    /// `inputs[1] = up` (the upstream gradient) off the node — the order autograd
    /// emits. The `MaskedFill` fill is `Scalar::one(dtype)` at the node's dtype.
    fn frozen_legacy_reduce_max_to_backward_decompose(
        graph: &mut Graph,
        id: NodeId,
        _params: &FusedOpParams,
    ) -> NodeId {
        use fuel_ir::Scalar;
        let (x_id, up_id, x_shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
        };
        let target = graph.node(up_id).shape.clone();

        // y = per-window max, broadcast back to x's shape.
        let y = graph.push(Node {
            op: Op::ReduceMaxTo(target.clone()),
            inputs: vec![x_id],
            shape: target.clone(),
            dtype,
        });
        let y_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![y],
            shape: x_shape.clone(),
            dtype,
        });
        // U8 mask = (x == max), then a float mask = MaskedFill(1.0 into zeros).
        let mask_u8 = graph.push(Node {
            op: Op::Equal,
            inputs: vec![x_id, y_b],
            shape: x_shape.clone(),
            dtype: DType::U8,
        });
        let zeros = graph.push(Node {
            op: Op::MulScalar(0.0),
            inputs: vec![x_id],
            shape: x_shape.clone(),
            dtype,
        });
        // A reduce-max backward over a non-real (packed/scale) dtype is not a
        // thing this frozen oracle handles; self-return the node unchanged
        // (decompose-fixpoint / surfaced-gap convention) rather than panic.
        let fill = match Scalar::one(dtype) {
            Ok(s) => s,
            Err(_) => return id,
        };
        let mask_f = graph.push(Node {
            op: Op::MaskedFill { value: fill },
            inputs: vec![zeros, mask_u8],
            shape: x_shape.clone(),
            dtype,
        });
        // ties = count per window; share = upstream / ties (fair share for ties).
        let ties = graph.push(Node {
            op: Op::ReduceSumTo(target.clone()),
            inputs: vec![mask_f],
            shape: target.clone(),
            dtype,
        });
        let share = graph.push(Node {
            op: Op::Div,
            inputs: vec![up_id, ties],
            shape: target,
            dtype,
        });
        let share_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![share],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Mul,
            inputs: vec![mask_f, share_b],
            shape: x_shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::powi_backward::decompose` (the legacy 3-node
    /// `PowI(exp-1)`→`MulScalar(exp)`→`Mul` spelling), copied VERBATIM from that
    /// module before A3 replaced the live body with the param-derived data
    /// recipe. Sole consumer: the A3 numeric-parity oracle (`powi_backward_recipe`
    /// below). Reads `inputs[0] = x` and `inputs[1] = up` (the upstream gradient)
    /// off the node — the order autograd emits — plus the output shape/dtype;
    /// `exp` rides the params.
    fn frozen_legacy_powi_backward_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let (x_id, up_id, shape, dtype) = {
            let n = graph.node(id);
            (n.inputs[0], n.inputs[1], n.shape.clone(), n.dtype)
        };
        let exp = match params {
            FusedOpParams::PowIBackward { exp } => *exp,
            // G2: total + never-panic — impossible params; return self.
            _ => return id,
        };
        let pow = graph.push(Node {
            op: Op::PowI(exp - 1),
            inputs: vec![x_id],
            shape: shape.clone(),
            dtype,
        });
        let scaled = graph.push(Node {
            op: Op::MulScalar(exp as f64),
            inputs: vec![pow],
            shape: shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Mul,
            inputs: vec![scaled, up_id],
            shape,
            dtype,
        })
    }

    /// FROZEN copy of the pre-migration imperative
    /// `registry::fused_linear::decompose` (the legacy 3-node `MatMul` +
    /// `BroadcastTo(rank-1 bias)` + `Add` spelling, broadcasting the rank-1
    /// bias DIRECTLY with no leading-1 pad), copied VERBATIM from that module @
    /// `b967bdb1` before slice-2 replaced the live body with the WithDim data
    /// recipe. Sole consumer: the slice-2 numeric-parity oracle
    /// (`fused_linear_recipe` below). Reads `inputs = [a, b, bias]` + the
    /// matmul output shape off the node.
    fn frozen_legacy_fused_linear_decompose(
        graph: &mut Graph,
        id: NodeId,
        _params: &FusedOpParams,
    ) -> NodeId {
        let (a_id, b_id, bias_id, out_shape, dtype) = {
            let n = graph.node(id);
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.shape.clone(),
                n.dtype,
            )
        };
        let mm_id = graph.push(Node {
            op: Op::MatMul,
            inputs: vec![a_id, b_id],
            shape: out_shape.clone(),
            dtype,
        });
        let bias_bcst_id = graph.push(Node {
            op: Op::BroadcastTo(out_shape.clone()),
            inputs: vec![bias_id],
            shape: out_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Add,
            inputs: vec![mm_id, bias_bcst_id],
            shape: out_shape,
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
            op: Op::Mul,
            inputs: vec![g_id, s_id],
            shape: x_shape.clone(),
            dtype,
        });
        let summed = graph.push(Node {
            op: Op::ReduceSumTo(keepdim.clone()),
            inputs: vec![gs],
            shape: keepdim,
            dtype,
        });
        let summed_b = graph.push(Node {
            op: Op::BroadcastTo(x_shape.clone()),
            inputs: vec![summed],
            shape: x_shape.clone(),
            dtype,
        });
        let sub = graph.push(Node {
            op: Op::Sub,
            inputs: vec![g_id, summed_b],
            shape: x_shape.clone(),
            dtype,
        });
        graph.push(Node {
            op: Op::Mul,
            inputs: vec![s_id, sub],
            shape: x_shape,
            dtype,
        })
    }

    #[test]
    fn emit_matches_softmax_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        // Oracle: the FROZEN legacy builder (reads inputs[0] + shape + dtype
        // off the node). T5 repointed this from the live registry decompose —
        // which now emits the 9-node recipe spelling — so this test keeps
        // pinning the Increment-A guarantee it always pinned: the grown
        // `emit` reconstructs the LEGACY imperative structure from the legacy
        // region datum.
        let fused = g.push(Node {
            op: Op::Const,
            inputs: vec![x],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let oracle = frozen_legacy_softmax_decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);

        // keepdim shape [2,1]; full shape [2,4].
        let kd = OpAttrs {
            target_shape: vec![2, 1],
            ..OpAttrs::default()
        };
        let full = OpAttrs {
            target_shape: vec![2, 4],
            ..OpAttrs::default()
        };
        // e = Exp(Sub(x, BroadcastTo(ReduceMaxTo(x)))) — mirrors decompose order
        // `Sub{[x, mb]}` exactly; built fresh each call so numerator and the
        // denominator's ReduceSumTo input are identical subtrees.
        let softmax_e = |kd: &OpAttrs, full: &OpAttrs| {
            op_node(
                OpTag::Exp,
                OpAttrs::default(),
                vec![op_node(
                    OpTag::Sub,
                    OpAttrs::default(),
                    vec![
                        bind(0),
                        op_node(
                            OpTag::BroadcastTo,
                            full.clone(),
                            vec![op_node(OpTag::ReduceMaxTo, kd.clone(), vec![bind(0)])],
                        ),
                    ],
                )],
            )
        };
        // out = Div(e, BroadcastTo(ReduceSumTo(e))) — mirrors `Div{[e, db]}`.
        let region = op_node(
            OpTag::Div,
            OpAttrs::default(),
            vec![
                softmax_e(&kd, &full),
                op_node(
                    OpTag::BroadcastTo,
                    full.clone(),
                    vec![op_node(
                        OpTag::ReduceSumTo,
                        kd.clone(),
                        vec![softmax_e(&kd, &full)],
                    )],
                ),
            ],
        );
        let emitted = emit_region(&mut g, &region, &[x], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }

    #[test]
    fn emit_matches_rope_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]); // seq=2, d=4, half=2
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let cos = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let sin = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let fused = g.push(Node {
            op: Op::Const,
            inputs: vec![x, cos, sin],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        // Oracle: the FROZEN legacy builder. T6 repointed this from the live
        // registry decompose — which now emits the data recipe (byte-identical
        // to legacy where a rank-raise occurs, but at EQUAL rank the recipe
        // elides the legacy's no-op prep `Reshape`, D4). This test keeps
        // pinning the Increment-A guarantee it always pinned: the grown `emit`
        // reconstructs the LEGACY imperative structure from a legacy-spelled
        // region datum.
        let oracle = frozen_legacy_rope_decompose(&mut g, fused, &FusedOpParams::Rope);

        // decompose's broadcast_shape for rank-2 [2,4] is [seq,d] = [2,4]; half slices along last dim.
        let full = OpAttrs {
            target_shape: vec![2, 4],
            ..OpAttrs::default()
        };
        let sl_first = OpAttrs {
            axis: Some(1),
            slice_start: Some(0),
            slice_len: Some(2),
            ..OpAttrs::default()
        };
        let sl_second = OpAttrs {
            axis: Some(1),
            slice_start: Some(2),
            slice_len: Some(2),
            ..OpAttrs::default()
        };
        let cat = OpAttrs {
            axis: Some(1),
            ..OpAttrs::default()
        };
        let bcast_reshape = |full: &OpAttrs, i: u8| {
            op_node(
                OpTag::BroadcastTo,
                full.clone(),
                vec![op_node(OpTag::Reshape, full.clone(), vec![bind(i)])],
            )
        };
        // left = Mul(x, cos_bcast); right = Mul(rotated_half, sin_bcast); out = Add(left, right).
        // rotated_half = Concat{dim:1}(Neg(second_half), first_half).
        let rotated = op_node(
            OpTag::Concat,
            cat,
            vec![
                op_node(
                    OpTag::Neg,
                    OpAttrs::default(),
                    vec![op_node(OpTag::Slice, sl_second, vec![bind(0)])],
                ),
                op_node(OpTag::Slice, sl_first, vec![bind(0)]),
            ],
        );
        let left = op_node(
            OpTag::Mul,
            OpAttrs::default(),
            vec![bind(0), bcast_reshape(&full, 1)],
        );
        let right = op_node(
            OpTag::Mul,
            OpAttrs::default(),
            vec![rotated, bcast_reshape(&full, 2)],
        );
        let region = op_node(OpTag::Add, OpAttrs::default(), vec![left, right]);

        let emitted = emit_region(&mut g, &region, &[x, cos, sin], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }

    #[test]
    fn emit_matches_layer_norm_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]); // last=1, reduced [2], keepdim [2,1]
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let fused = g.push(Node {
            op: Op::Const,
            inputs: vec![x],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        // Oracle: the FROZEN legacy builder (the `Reshape(keepdim)` spelling).
        // T7 repointed this from the live registry decompose — which now emits
        // the D3 `Unsqueeze` swap — so this test keeps pinning the Increment-A
        // guarantee it always pinned: the grown `emit` reconstructs the LEGACY
        // imperative structure from a legacy-spelled (Reshape) region datum.
        let oracle = frozen_legacy_layer_norm_decompose(
            &mut g,
            fused,
            &FusedOpParams::LayerNormLastDim { eps: 1e-5 },
        );

        let kd = OpAttrs {
            target_shape: vec![2, 1],
            ..OpAttrs::default()
        };
        let full = OpAttrs {
            target_shape: vec![2, 4],
            ..OpAttrs::default()
        };
        let md = OpAttrs {
            axis: Some(1),
            ..OpAttrs::default()
        };
        let eps_attrs = OpAttrs {
            scalars: vec![1e-5],
            ..OpAttrs::default()
        }; // BAKED constant, not a slot
        // centered = Sub(x, BroadcastTo(Reshape(MeanDim(x)))) — shared subterm.
        let centered = op_node(
            OpTag::Sub,
            OpAttrs::default(),
            vec![
                bind(0),
                op_node(
                    OpTag::BroadcastTo,
                    full.clone(),
                    vec![op_node(
                        OpTag::Reshape,
                        kd.clone(),
                        vec![op_node(OpTag::MeanDim, md.clone(), vec![bind(0)])],
                    )],
                ),
            ],
        );
        // denom_bcast = BroadcastTo(Sqrt(AddScalar(eps)(Reshape(MeanDim(Sqr(centered)))))).
        let denom_bcast = op_node(
            OpTag::BroadcastTo,
            full.clone(),
            vec![op_node(
                OpTag::Sqrt,
                OpAttrs::default(),
                vec![op_node(
                    OpTag::AddScalar,
                    eps_attrs,
                    vec![op_node(
                        OpTag::Reshape,
                        kd.clone(),
                        vec![op_node(
                            OpTag::MeanDim,
                            md.clone(),
                            vec![op_node(
                                OpTag::Sqr,
                                OpAttrs::default(),
                                vec![centered.clone()],
                            )],
                        )],
                    )],
                )],
            )],
        );
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
                Box::new(Dim::Extent {
                    operand: 0,
                    axis: LAST,
                }),
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
            assert!(
                r.target_shape_rel.is_none(),
                "resolved attrs are fully concrete"
            );
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
            let attrs = OpAttrs {
                axis_last: true,
                ..OpAttrs::default()
            };
            // Reduce family (axis carrier): LAST = rank − 1.
            let r = resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 4]]).expect("rank 2");
            assert_eq!(r.axis, Some(1));
            assert!(!r.axis_last, "resolved attrs are fully concrete");
            let r =
                resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 3, 4]]).expect("rank 3");
            assert_eq!(r.axis, Some(2));
            // Concat rides the same axis carrier.
            let r =
                resolve_rel_attrs(OpTag::Concat, &attrs, &[], &[vec![2, 3, 4]]).expect("concat");
            assert_eq!(r.axis, Some(2));
            // Unsqueeze (dims carrier): APPEND — dim == rank (`primitive_shape`
            // permits `dim == rank`).
            let r =
                resolve_rel_attrs(OpTag::Unsqueeze, &attrs, &[], &[vec![2, 4]]).expect("unsqueeze");
            assert_eq!(r.dims, vec![2]);
            assert!(!r.axis_last);
            // Squeeze (dims carrier): LAST = rank − 1.
            let r =
                resolve_rel_attrs(OpTag::Squeeze, &attrs, &[], &[vec![2, 4, 1]]).expect("squeeze");
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
                Err(RelAttrError::Expr(ShapeExprError::OperandOutOfRange {
                    operand: 3,
                    operands: 1
                })),
            );
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start_rel: Some(Dim::Extent {
                    operand: 7,
                    axis: LAST,
                }),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::Expr(ShapeExprError::OperandOutOfRange {
                    operand: 7,
                    operands: 1
                })),
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
                Err(RelAttrError::RelAbsConflict {
                    field: "target_shape"
                }),
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
                Err(RelAttrError::RelAbsConflict {
                    field: "slice_start"
                }),
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
            let attrs = OpAttrs {
                axis: Some(0),
                axis_last: true,
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::SumDim, &attrs, &[], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "axis" }),
            );
            // dims XOR axis_last (Unsqueeze's carrier is `dims`).
            let attrs = OpAttrs {
                dims: vec![0],
                axis_last: true,
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Unsqueeze, &attrs, &[], &[vec![2, 4]]),
                Err(RelAttrError::RelAbsConflict { field: "dims" }),
            );
        }

        #[test]
        fn negative_result_is_a_typed_decline() {
            // 0 − 2 = −2: a negative slice offset is malformed, not a wrap.
            let neg = Dim::Sub(Box::new(Dim::Const(0)), Box::new(Dim::Const(2)));
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start_rel: Some(neg),
                ..OpAttrs::default()
            };
            assert_eq!(
                resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]]),
                Err(RelAttrError::Negative {
                    field: "slice_start",
                    value: -2
                }),
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
                resolve_rel_attrs(
                    OpTag::BroadcastTo,
                    &attrs,
                    &[vec![2, SYMBOLIC]],
                    &[vec![2, 4]]
                ),
                Err(RelAttrError::SymbolicGap {
                    field: "target_shape"
                }),
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
            let attrs = OpAttrs {
                axis_last: true,
                ..OpAttrs::default()
            };
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
                Err(RelAttrError::Expr(ShapeExprError::AxisOutOfRange {
                    axis: LAST,
                    rank: 0
                })),
            );
        }

        #[test]
        fn rel_free_attrs_pass_through_unchanged() {
            // The no-rel fast path: absolute attrs resolve to themselves.
            let attrs = OpAttrs {
                axis: Some(1),
                slice_start: Some(2),
                slice_len: Some(3),
                ..OpAttrs::default()
            };
            let r = resolve_rel_attrs(OpTag::Slice, &attrs, &[vec![2, 4]], &[vec![2, 4]])
                .expect("no-op");
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
            OpAttrs {
                target_shape_rel: Some(ShapeExpr::SameAs { operand: 0 }),
                ..OpAttrs::default()
            }
        }

        #[test]
        fn rel_region_emits_polymorphically_across_shapes() {
            // The headline polymorphism: ONE region datum —
            // Add(bind0, BroadcastTo{SameAs{0}}(bind1)) — emitted at two
            // different shapes produces the correct target BOTH times
            // (impossible with absolute attrs: a baked target matches exactly
            // one shape).
            let region = op_node(
                OpTag::Add,
                OpAttrs::default(),
                vec![
                    bind(0),
                    op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
                ],
            );
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
            let region = op_node(
                OpTag::Mul,
                OpAttrs::default(),
                vec![
                    op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(0)]),
                    op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
                ],
            );
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
            let region = op_node(
                OpTag::Mul,
                OpAttrs::default(),
                vec![
                    bind(0),
                    op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
                ],
            );
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
                OpAttrs {
                    target_shape: vec![2, 3, 4],
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            let mut g = Graph::new();
            let t = cst(&mut g, &[3, 4]);
            let emitted = emit_region(&mut g, &region, &[t], &[]);
            assert_eq!(
                g.node(emitted).op,
                Op::BroadcastTo(Shape::from_dims(&[2, 3, 4]))
            );
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
            let region = op_node(
                OpTag::AddScalar,
                OpAttrs::default(),
                vec![op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)])],
            );
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
            let region = op_node(
                OpTag::Add,
                OpAttrs::default(),
                vec![
                    bind(0),
                    op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(1)]),
                ],
            );
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
                    error: RelAttrError::RelAbsConflict {
                        field: "target_shape"
                    },
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
            assert_eq!(
                decompose_region(&mut g, fused),
                fused,
                "resolution decline ⇒ fixpoint"
            );
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
                Err(RelAttrError::Negative {
                    field: "slice_start",
                    value: -2
                }),
            );
            // A graph-side SYMBOLIC bind extent maps to the §6.20 SYMBOLIC
            // sentinel → SymbolicGap (the surfaced-gap posture, §6.20-0004).
            let region = op_node(OpTag::BroadcastTo, bcast_same_as_0(), vec![bind(0)]);
            let dyn_shape = Shape::from_dims(&[2, 8]).with_dynamic_axis(1, 0, fuel_ir::SymId(0));
            let d = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: dyn_shape,
                dtype: DType::F32,
            });
            assert_eq!(
                try_emit_region(&mut g, &region, &[d], &[]),
                Err(RelAttrError::SymbolicGap {
                    field: "target_shape"
                }),
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
        let region = op_node(
            OpTag::Add,
            OpAttrs::default(),
            vec![
                op_node(OpTag::Exp, OpAttrs::default(), vec![bind(0)]),
                op_node(OpTag::Exp, OpAttrs::default(), vec![bind(0)]),
            ],
        );
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
        let region = op_node(
            OpTag::Add,
            OpAttrs::default(),
            vec![
                op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)]),
                op_node(OpTag::MulScalar, OpAttrs::default(), vec![bind(0)]),
            ],
        );
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
        let a = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let b = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        // Reference graph.
        let add = g.push(Node {
            op: Op::Add,
            inputs: vec![a, b],
            shape: sh.clone(),
            dtype: DType::F32,
        });
        let reference = g.push(Node {
            op: Op::Cast(DType::F16),
            inputs: vec![add],
            shape: sh.clone(),
            dtype: DType::F16,
        });

        let region = op_node(
            OpTag::Cast,
            OpAttrs {
                cast_dtype: Some("f16".into()),
                ..OpAttrs::default()
            },
            vec![op_node(
                OpTag::Add,
                OpAttrs::default(),
                vec![bind(0), bind(1)],
            )],
        );
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
                Op::Exp => eval(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v.exp())
                    .collect(),
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
                        .flat_map(|&v| std::iter::repeat_n(v, last))
                        .collect()
                }
                other => panic!("eval: unhandled op {other:?}"),
            }
        }

        fn softmax_fused_node(g: &mut Graph, dims: &[usize]) -> (NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
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
                let data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.37).sin() * 3.0 - 0.5)
                    .collect();
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
                g.node(d).inputs[0],
                e,
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
            let out =
                crate::registry::softmax_last_dim::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }

        /// INJECTED item (2), the flip-withdrawal posture (Baracuda #68 /
        /// KISS-Ops closed registry): an op token with no registry semantics —
        /// the in-memory analog of the withdrawn reverse-scan "flip" spelling —
        /// must surface as a typed honest-miss decline: the node stays fused
        /// (fixpoint), NEVER accepted, NEVER a crash. The fabricated recipe
        /// stands in for a foreign token by carrying `OpTag::Clamp` — a tag
        /// with NO primitive re-emission today (`tag_to_op` → `None`), exactly
        /// the semantics-absent posture an unregistered op name resolves to.
        /// (This stand-in was `OpTag::PowI` until Increment C carriers A3 gave
        /// PowI its i32-exponent carrier; `Clamp` — still awaiting a two-scalar
        /// carrier — is the current canonical carrier-less token.) If/when the
        /// token registers, it becomes a NAMED-op resolution case — semantics
        /// arrive via registration, never via silent acceptance here.
        #[test]
        fn decompose_via_recipe_declines_an_unknown_token_recipe() {
            let fabricated = op_node(
                OpTag::Clamp,
                OpAttrs {
                    scalars: vec![0.0, 1.0],
                    ..OpAttrs::default()
                },
                vec![bind(0)],
            );
            let mut g = Graph::new();
            let (_x, fused) = softmax_fused_node(&mut g, &[2, 4]);
            let before = g.len();
            let out = decompose_via_recipe(&mut g, fused, &fabricated, Some(Vec::new()));
            assert_eq!(out, fused, "semantics-absent token ⇒ honest-miss fixpoint");
            assert_eq!(
                g.len(),
                before,
                "declined BEFORE any emission — no partial nodes"
            );
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
                Op::Neg => eval_rope(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| -v)
                    .collect(),
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
        fn rope_fused_node(g: &mut Graph, x_dims: &[usize]) -> (NodeId, NodeId, NodeId, NodeId) {
            let rank = x_dims.len();
            let table_dims = [x_dims[rank - 2], x_dims[rank - 1]];
            let x_sh = Shape::from_dims(x_dims);
            let t_sh = Shape::from_dims(&table_dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: x_sh.clone(),
                dtype: DType::F32,
            });
            let cos = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: t_sh.clone(),
                dtype: DType::F32,
            });
            let sin = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: t_sh,
                dtype: DType::F32,
            });
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

                let legacy_root = frozen_legacy_rope_decompose(&mut g, fused, &FusedOpParams::Rope);

                // Distinct, deterministic leaf data for x / cos / sin.
                let x_n: usize = x_dims.iter().product();
                let t_n = seq * d;
                let x_data: Vec<f64> = (0..x_n)
                    .map(|i| ((i as f64) * 0.31).sin() * 2.0 - 0.4)
                    .collect();
                let cos_data: Vec<f64> =
                    (0..t_n).map(|i| ((i as f64) * 0.17 + 0.5).cos()).collect();
                let sin_data: Vec<f64> =
                    (0..t_n).map(|i| ((i as f64) * 0.23 - 0.2).sin()).collect();
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
            assert_eq!(
                reshapes, 2,
                "rank-raise materializes both legacy prep Reshapes (D4)"
            );
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
            assert_eq!(
                reshapes, 0,
                "equal-rank broadcast needs no prep Reshape (D4)"
            );
            let op_nodes = reachable
                .iter()
                .filter(|&&n| !matches!(g.node(n).op, Op::Const))
                .count();
            assert_eq!(
                op_nodes, 9,
                "the 9-node rope recipe (2×Bcast/2×Slice/Neg/Concat/2×Mul/Add)"
            );
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
                Op::Sqr => eval_norm(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * v)
                    .collect(),
                Op::Sqrt => eval_norm(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v.sqrt())
                    .collect(),
                Op::AddScalar(e) => eval_norm(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v + e)
                    .collect(),
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
                        .flat_map(|&v| std::iter::repeat_n(v, last))
                        .collect()
                }
                other => panic!("eval_norm: unhandled op {other:?}"),
            }
        }

        /// Build a fused RmsNormLastDim node over `x [dims]`, carrying `eps`.
        /// Returns `(x, fused)`.
        fn rms_norm_fused_node(g: &mut Graph, dims: &[usize], eps: f64) -> (NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
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
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
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
                let data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.37).sin() * 3.0 - 0.5)
                    .collect();
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
                let data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.29).cos() * 2.0 + 0.3)
                    .collect();
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
            assert_eq!(
                unsqueezes, 1,
                "keepdim restored via Unsqueeze append (D3 swap)"
            );
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
            assert_eq!(
                unsqueezes, 2,
                "both keepdim restores via Unsqueeze append (D3 swap)"
            );
            assert_eq!(
                reshapes, 0,
                "no baked keepdim Reshape / no D4 pad after the swap"
            );
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
            let data: Vec<f64> = (0..n)
                .map(|i| 0.001 * (((i as f64) * 0.37).sin() + 1.2))
                .collect();

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
            let data: Vec<f64> = (0..n)
                .map(|i| 1.0 + 0.001 * ((i as f64) * 0.37).sin())
                .collect();

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
                        .flat_map(|&v| std::iter::repeat_n(v, last))
                        .collect()
                }
                other => panic!("eval_bwd: unhandled op {other:?}"),
            }
        }

        /// Build a fused SoftmaxLastDimBackward node over `s [dims]` (input 0,
        /// the forward output) and `g [dims]` (input 1, the upstream gradient).
        /// Returns `(s, g, fused)`.
        fn softmax_backward_fused_node(g: &mut Graph, dims: &[usize]) -> (NodeId, NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let s = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let up = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
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
                assert_eq!(
                    g.node(new_root).shape,
                    sh,
                    "softmax backward is shape-preserving"
                );
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_softmax_backward_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::SoftmaxLastDimBackward,
                );

                let n: usize = dims.iter().product();
                let s_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.29).sin() * 0.5 + 0.5)
                    .collect();
                let g_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.53).cos() * 2.0 - 0.3)
                    .collect();
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
            assert!(
                matches!(g.node(root).op, Op::Mul),
                "backward root is the outer Mul"
            );
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
            assert_eq!(
                reduce_sum_tos, 0,
                "no baked keepdim ReduceSumTo after the swap"
            );
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
            let x = crate::NodeHandle::from_f32(
                vec![0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6],
                Shape::from_dims(&[2, 3]),
                &dev,
            )
            .unwrap();
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
                assert_eq!(
                    node.inputs[0], y_id,
                    "backward input 0 = the forward softmax output"
                );
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
            assert_ne!(
                new_root, bwd_id,
                "the autograd backward node decomposes via the recipe"
            );
            assert_eq!(gr.node(new_root).shape, sh, "shape-preserving");
            let reachable = crate::topo_order_multi(&gr, &[new_root]);
            assert!(
                reachable
                    .iter()
                    .any(|&n| matches!(gr.node(n).op, Op::SumDim(_))),
                "the autograd path reaches the D3 SumDim spelling",
            );

            // Numeric parity (leaf-first interpreter over `[y = s, up = g]`).
            let n: usize = sh.dims().iter().product();
            let s_data: Vec<f64> = (0..n)
                .map(|i| ((i as f64) * 0.31).sin() * 0.5 + 0.5)
                .collect();
            let g_data: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.47).cos() - 0.1).collect();
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

    // layer_norm_last_dim_backward migration (Increment C slice 2, S2-1) ------
    //
    // The ~20-node imperative backward body
    // `grad_x = istd · (g − mean(g) − xhat·mean(g·xhat))` (mean over the last
    // dim, `xhat = (x − mean(x))·istd`, `istd = rsqrt(var + eps)`) becomes a
    // portable `PatternNode` DATA recipe. Bind space: `0 = x`, `1 = g` (the
    // upstream gradient) — the order the autograd
    // `BackwardKind::Fused(LAYER_NORM_LAST_DIM_BACKWARD)` edge emits. The four
    // keepdim restores are the ratified D3 shrink-via-swap (`Reshape(keepdim)`
    // → `MeanDim(axis_last)` shrink + `Unsqueeze(axis_last = append)`,
    // node-TYPE change, numerically bit-exact); each broadcast targets
    // `SameAs { operand: 0 }` (x's full shape) over the Bind space (D2). One
    // `eps` open scalar slot. D4 never fires at equal rank (the `Unsqueeze`
    // rebuilds rank before every broadcast). RISK-A: `xhat`/`istd` carry the
    // eps slot, so emit's slot-free identity-share does NOT dedup them — see
    // `..._open_slot_xhat_is_recomputed_...` below.
    mod layer_norm_backward_recipe {
        use super::super::*;
        use super::frozen_legacy_layer_norm_backward_decompose;
        use crate::registry::{FusedOps, layer_norm_last_dim_backward};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the layer-norm-backward
        /// primitive vocabulary (leaf-lookup FIRST, then elementwise
        /// `Mul`/`Sub`, unary `Sqr`/`Rsqrt`, `AddScalar`, last-axis `MeanDim`,
        /// metadata-only keepdim restore `Unsqueeze`/`Reshape`, last-dim
        /// `BroadcastTo`). BOTH parity sides run through it with identical
        /// in-order arithmetic, so a bit-exact assert isolates recipe STRUCTURE
        /// — neither the `Reshape`→`Unsqueeze` D3 swap NOR the open-slot
        /// `xhat`/`istd` recompute can perturb it (recompute is deterministic).
        fn eval_lnb(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::Mul => {
                    let a = eval_lnb(g, node.inputs[0], leaves);
                    let b = eval_lnb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::Sub => {
                    let a = eval_lnb(g, node.inputs[0], leaves);
                    let b = eval_lnb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x - y).collect()
                }
                Op::Sqr => eval_lnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * v)
                    .collect(),
                Op::Rsqrt => eval_lnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| 1.0 / v.sqrt())
                    .collect(),
                Op::AddScalar(e) => eval_lnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v + e)
                    .collect(),
                // Last-axis mean — rank-reducing; identical fold both spellings.
                Op::MeanDim(_) => {
                    let input = eval_lnb(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().sum::<f64>() / last as f64)
                        .collect()
                }
                // Metadata-only keepdim restores (the D3 swap and its legacy
                // twin evaluate identically here).
                Op::Unsqueeze { .. } | Op::Reshape(_) => eval_lnb(g, node.inputs[0], leaves),
                // Broadcast a keepdim `[.., 1]` tensor back along the last axis.
                Op::BroadcastTo(target) => {
                    let input = eval_lnb(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    let last = *target.dims().last().unwrap();
                    assert_eq!(
                        input.len() * last,
                        out_n,
                        "broadcast is a last-dim repeat in these graphs",
                    );
                    input
                        .iter()
                        .flat_map(|&v| std::iter::repeat_n(v, last))
                        .collect()
                }
                other => panic!("eval_lnb: unhandled op {other:?}"),
            }
        }

        /// Build a fused LayerNormLastDimBackward node over `x [dims]` (input 0)
        /// and `g [dims]` (input 1, the upstream gradient), carrying `eps`.
        /// Returns `(x, g, fused)`.
        fn ln_backward_fused_node(
            g: &mut Graph,
            dims: &[usize],
            eps: f64,
        ) -> (NodeId, NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let up = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
                    FusedOpParams::LayerNormLastDimBackward { eps },
                ),
                inputs: vec![x, up],
                shape: sh,
                dtype: DType::F32,
            });
            (x, up, fused)
        }

        /// S2-1 (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3 (the
        /// polymorphism the baked-shape legacy body never had), and its
        /// numerics match the FROZEN legacy builder bit-exactly under the
        /// shared reference interpreter.
        #[test]
        fn layer_norm_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (x, up, fused) = ln_backward_fused_node(&mut g, &dims, 1e-5);
                let sh = Shape::from_dims(&dims);
                let new_root = layer_norm_last_dim_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::LayerNormLastDimBackward { eps: 1e-5 },
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(
                    g.node(new_root).shape,
                    sh,
                    "layer_norm backward is shape-preserving"
                );
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_layer_norm_backward_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::LayerNormLastDimBackward { eps: 1e-5 },
                );

                let n: usize = dims.iter().product();
                let x_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.29).sin() * 2.0 + 0.3)
                    .collect();
                let g_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.53).cos() * 1.5 - 0.2)
                    .collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, x_data);
                leaves.insert(up, g_data);
                let got = eval_lnb(&g, new_root, &leaves);
                let want = eval_lnb(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "layer_norm_backward[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// S2-1 (structural, born-red): every keepdim restore is the D3
        /// shrink-via-swap `Unsqueeze` append, NOT the baked `Reshape(keepdim)`
        /// — the crisp discriminator against the pre-migration imperative body
        /// (which emits `Reshape`, so this test is RED until migration). Zero
        /// `Reshape` also confirms no D4 pad fires (equal rank throughout). The
        /// backward root is the outer `Mul(istd, inner)`.
        #[test]
        fn layer_norm_backward_recipe_uses_the_unsqueeze_keepdim_swap() {
            let mut g = Graph::new();
            let (_x, _up, fused) = ln_backward_fused_node(&mut g, &[2, 4], 1e-5);
            let root = layer_norm_last_dim_backward::decompose(
                &mut g,
                fused,
                &FusedOpParams::LayerNormLastDimBackward { eps: 1e-5 },
            );
            assert_ne!(root, fused, "recipe decompose fires");
            assert!(
                matches!(g.node(root).op, Op::Mul),
                "backward root is the outer Mul(istd, inner)"
            );
            let reachable = crate::topo_order_multi(&g, &[root]);
            let unsqueezes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Unsqueeze { .. }))
                .count();
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert!(
                unsqueezes >= 1,
                "keepdim restored via Unsqueeze append (D3 swap)"
            );
            assert_eq!(
                reshapes, 0,
                "no baked keepdim Reshape / no D4 pad after the D3 swap"
            );
        }

        /// S2-1 (RISK-A): `xhat = Mul(xc, istd)` and `istd = Rsqrt(AddScalar[
        /// eps](...))` carry the eps OPEN slot, so emit's slot-free
        /// identity-share does NOT dedup them — the recipe emits them once per
        /// occurrence, yielding a STRICTLY HEAVIER base map than the legacy DAG
        /// (which shares them). This is the ACCEPTED redundancy documented on
        /// `recipe()` (numerically identical — deterministic recompute — and no
        /// open-slot-sharing emit extension is built in this slice). The
        /// downstream full-optimizer CSE re-collapses the duplicates, which this
        /// pins as the mitigation.
        #[test]
        fn layer_norm_backward_recipe_open_slot_xhat_is_recomputed_then_cse_recollapses() {
            let count_ops = |g: &Graph, root: NodeId| -> usize {
                crate::topo_order_multi(g, &[root])
                    .iter()
                    .filter(|&&n| !matches!(g.node(n).op, Op::Const))
                    .count()
            };

            // Recipe emission (base map — no fused ops remain to lower).
            let mut gr = Graph::new();
            let (_x, _up, fused) = ln_backward_fused_node(&mut gr, &[2, 4], 1e-5);
            let recipe_root = layer_norm_last_dim_backward::decompose(
                &mut gr,
                fused,
                &FusedOpParams::LayerNormLastDimBackward { eps: 1e-5 },
            );
            let recipe_n = count_ops(&gr, recipe_root);

            // Legacy DAG (shares xhat/istd via let-bindings).
            let mut gl = Graph::new();
            let (_x2, _up2, fused2) = ln_backward_fused_node(&mut gl, &[2, 4], 1e-5);
            let legacy_root = frozen_legacy_layer_norm_backward_decompose(
                &mut gl,
                fused2,
                &FusedOpParams::LayerNormLastDimBackward { eps: 1e-5 },
            );
            let legacy_n = count_ops(&gl, legacy_root);

            assert!(
                recipe_n > legacy_n,
                "RISK-A: emit's slot-free share does NOT dedup the eps-slot-carrying \
                 xhat/istd — recipe base map {recipe_n} > legacy {legacy_n}",
            );

            // Downstream full-optimizer CSE re-collapses the redundant recompute.
            let shared: crate::SharedGraph = std::sync::Arc::new(std::sync::RwLock::new(gr));
            let cse_roots = crate::opt::optimize(&shared, &[recipe_root]);
            let gr = shared.read().unwrap();
            let cse_n = count_ops(&gr, cse_roots[0]);
            assert!(
                cse_n < recipe_n,
                "downstream CSE re-collapses the redundant recompute: {cse_n} < {recipe_n}",
            );
            assert_eq!(
                cse_n, legacy_n,
                "CSE collapses the recipe base map back to the legacy DAG node count",
            );
        }

        /// S2-1 (totality): a wrong params payload is a typed decline surfaced
        /// as a fixpoint (G2), never a panic, declining BEFORE any emission.
        #[test]
        fn layer_norm_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, _up, fused) = ln_backward_fused_node(&mut g, &[2, 4], 1e-5);
            let before = g.len();
            let out = layer_norm_last_dim_backward::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    // rms_norm_last_dim_backward migration (Increment C carriers, A1) ---------
    //
    // The ~22-node imperative closed-form
    // `grad_x = r_rms · (g − x·s / (n·(mean_sq + eps)))` becomes a portable
    // `PatternNode` DATA recipe — the FIRST recipe to drive a SHAPE-DERIVED
    // scalar through live emit: `n = dims[last]` (the reduced_count) is a
    // `MulScalar(scalar_rel = Extent(0, LAST))`, resolved from x's shape at emit
    // (not baked, not a params slot). Sibling of the layer-norm-backward recipe:
    // `MeanDim(axis_last)` + `Unsqueeze(append)` keepdim (D3 swap) + broadcast to
    // `SameAs 0`; the `eps` `AddScalar` is an OPEN slot filled by the projection.
    // Binds: `0 = x`, `1 = upstream`. RISK-A: `denom_kd` (which carries the eps
    // slot) is consumed by BOTH the `Rsqrt` and the `MulScalar(n)`, so emit's
    // slot-free identity-share does NOT dedup it — the recipe emits it twice
    // (numerically identical deterministic recompute; the legacy DAG shared it),
    // and downstream CSE re-collapses. The parity oracle runs both sides through
    // a toy f64 interpreter (Reshape/Unsqueeze = metadata passthrough) so neither
    // the D3 swap nor the recompute can perturb the bit-exact assert.
    mod rms_norm_backward_recipe {
        use super::super::*;
        use super::frozen_legacy_rms_norm_backward_decompose;
        use crate::registry::{FusedOps, rms_norm_last_dim_backward};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the rms-norm-backward primitive
        /// vocabulary (leaf-lookup FIRST, then elementwise `Mul`/`Sub`/`Div`,
        /// unary `Sqr`/`Rsqrt`, `AddScalar`/`MulScalar`, last-axis
        /// `MeanDim`/`SumDim`, metadata-only keepdim restore
        /// `Unsqueeze`/`Reshape`, last-dim `BroadcastTo`). BOTH parity sides run
        /// through it with identical in-order arithmetic, so a bit-exact assert
        /// isolates recipe STRUCTURE — neither the `Reshape`→`Unsqueeze` D3 swap
        /// NOR the open-slot `denom_kd` recompute can perturb it.
        fn eval_rnb(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::Mul => {
                    let a = eval_rnb(g, node.inputs[0], leaves);
                    let b = eval_rnb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::Sub => {
                    let a = eval_rnb(g, node.inputs[0], leaves);
                    let b = eval_rnb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x - y).collect()
                }
                Op::Div => {
                    let a = eval_rnb(g, node.inputs[0], leaves);
                    let b = eval_rnb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x / y).collect()
                }
                Op::Sqr => eval_rnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * v)
                    .collect(),
                Op::Rsqrt => eval_rnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| 1.0 / v.sqrt())
                    .collect(),
                Op::AddScalar(e) => eval_rnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v + e)
                    .collect(),
                Op::MulScalar(m) => eval_rnb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * m)
                    .collect(),
                // Last-axis reductions — rank-reducing; identical fold both
                // spellings.
                Op::MeanDim(_) => {
                    let input = eval_rnb(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().sum::<f64>() / last as f64)
                        .collect()
                }
                Op::SumDim(_) => {
                    let input = eval_rnb(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().sum::<f64>())
                        .collect()
                }
                // Metadata-only keepdim restores (the D3 swap and its legacy twin
                // evaluate identically here).
                Op::Unsqueeze { .. } | Op::Reshape(_) => eval_rnb(g, node.inputs[0], leaves),
                // Broadcast a keepdim `[.., 1]` tensor back along the last axis.
                Op::BroadcastTo(target) => {
                    let input = eval_rnb(g, node.inputs[0], leaves);
                    let out_n: usize = target.dims().iter().product();
                    let last = *target.dims().last().unwrap();
                    assert_eq!(
                        input.len() * last,
                        out_n,
                        "broadcast is a last-dim repeat in these graphs",
                    );
                    input
                        .iter()
                        .flat_map(|&v| std::iter::repeat_n(v, last))
                        .collect()
                }
                other => panic!("eval_rnb: unhandled op {other:?}"),
            }
        }

        /// Build a fused RmsNormLastDimBackward node over `x [dims]` (input 0)
        /// and `g [dims]` (input 1, the upstream gradient), carrying `eps`.
        /// Returns `(x, g, fused)`.
        fn rms_backward_fused_node(
            g: &mut Graph,
            dims: &[usize],
            eps: f64,
        ) -> (NodeId, NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let up = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
                    FusedOpParams::RmsNormLastDimBackward { eps },
                ),
                inputs: vec![x, up],
                shape: sh,
                dtype: DType::F32,
            });
            (x, up, fused)
        }

        /// A1 (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3 (the
        /// polymorphism the baked-shape legacy body never had — including the
        /// shape-derived `MulScalar(n)`), and its numerics match the FROZEN
        /// legacy builder bit-exactly under the shared reference interpreter.
        #[test]
        fn rms_norm_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (x, up, fused) = rms_backward_fused_node(&mut g, &dims, 1e-5);
                let sh = Shape::from_dims(&dims);
                let new_root = rms_norm_last_dim_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDimBackward { eps: 1e-5 },
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {dims:?}");
                assert_eq!(
                    g.node(new_root).shape,
                    sh,
                    "rms_norm backward is shape-preserving"
                );
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_rms_norm_backward_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDimBackward { eps: 1e-5 },
                );

                let n: usize = dims.iter().product();
                let x_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.41).sin() * 2.0 + 0.7)
                    .collect();
                let g_data: Vec<f64> = (0..n)
                    .map(|i| ((i as f64) * 0.61).cos() * 1.3 - 0.15)
                    .collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, x_data);
                leaves.insert(up, g_data);
                let got = eval_rnb(&g, new_root, &leaves);
                let want = eval_rnb(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "rms_norm_backward[{i}] at {dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// A1 (structural, born-red): every keepdim restore is the D3
        /// shrink-via-swap `Unsqueeze` append, NOT the baked `Reshape(keepdim)`
        /// (the crisp discriminator against the pre-migration imperative body,
        /// which emits `Reshape` — so this is RED until migration), and the
        /// reduced-count divisor is a live `MulScalar(n = dims[last])` resolved
        /// from x's shape by the `scalar_rel` carrier (evidence the shape-derived
        /// scalar reached emit). The backward root is the outer `Mul(rrms, inner)`.
        #[test]
        fn rms_norm_backward_recipe_uses_the_unsqueeze_keepdim_swap_and_shape_scalar() {
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                let mut g = Graph::new();
                let (_x, _up, fused) = rms_backward_fused_node(&mut g, &dims, 1e-5);
                let root = rms_norm_last_dim_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::RmsNormLastDimBackward { eps: 1e-5 },
                );
                assert_ne!(root, fused, "recipe decompose fires at {dims:?}");
                assert!(
                    matches!(g.node(root).op, Op::Mul),
                    "backward root is the outer Mul(rrms, inner)"
                );
                let reachable = crate::topo_order_multi(&g, &[root]);
                let unsqueezes = reachable
                    .iter()
                    .filter(|&&n| matches!(g.node(n).op, Op::Unsqueeze { .. }))
                    .count();
                let reshapes = reachable
                    .iter()
                    .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                    .count();
                assert!(
                    unsqueezes >= 1,
                    "keepdim restored via Unsqueeze append (D3 swap) at {dims:?}"
                );
                assert_eq!(
                    reshapes, 0,
                    "no baked keepdim Reshape / no D4 pad after the D3 swap at {dims:?}"
                );
                // The reduced-count divisor resolved to n = dims[last] from x's
                // shape (the scalar_rel carrier), at whatever rank.
                let n = *dims.last().unwrap() as f64;
                let mul_scalars: Vec<f64> = reachable
                    .iter()
                    .filter_map(|&nid| match g.node(nid).op {
                        Op::MulScalar(v) => Some(v),
                        _ => None,
                    })
                    .collect();
                assert!(
                    mul_scalars.contains(&n),
                    "a MulScalar(n={n}) resolved from x's last-axis extent at {dims:?}, got {mul_scalars:?}",
                );
            }
        }
    }

    // fused_linear migration (Increment C slice 2, S2-2) ----------------------
    //
    // The 3-node imperative `Add(MatMul(a, b), BroadcastTo(rank-1 bias))`
    // becomes a portable `PatternNode` DATA recipe, and the FIRST recipe to
    // drive `WithDim` through live emit: the bias broadcast targets
    // `WithDim { operand: 0, axis: LAST, dim: Extent { operand: 1, axis: LAST } }`
    // — a's shape with its LAST axis (K) replaced by b's LAST extent (N) = the
    // matmul output `[..batch, M, N]`. This is Fuel-INTERNAL shape resolution
    // (not §6.19 wire emission), so it is NOT gated on KISS #86. Binds:
    // `0 = a`, `1 = b`, `2 = bias`. The rank-1 bias rank-raises to the matmul
    // output, so emit materializes a leading-1 pad `Reshape` (D4) the imperative
    // body never had — the numeric-parity oracle below runs both sides through a
    // toy interpreter (Reshape = metadata passthrough) to isolate it. The
    // `canonical_pattern` see-through of that pad (RISK-C) + the lower→fuse
    // round-trip live in `registry/fused_linear.rs`'s own tests.
    mod fused_linear_recipe {
        use super::super::*;
        use super::frozen_legacy_fused_linear_decompose;
        use crate::registry::{FusedOps, fused_linear};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Right-aligned NumPy broadcast of `input` (shape `in_shape`) to
        /// `target` — a size-1 or padded leading dim contributes stride 0.
        /// General on purpose: fused_linear's bias broadcast tiles along the
        /// LEADING `M`/batch dims (`[1,N] → [M,N]`), unlike the norm recipes'
        /// last-dim keepdim repeat.
        fn broadcast(input: &[f64], in_shape: &[usize], target: &[usize]) -> Vec<f64> {
            let rank = target.len();
            let pad = rank - in_shape.len();
            let mut real_strides = vec![0isize; in_shape.len()];
            let mut s = 1isize;
            for i in (0..in_shape.len()).rev() {
                real_strides[i] = s;
                s *= in_shape[i] as isize;
            }
            let mut in_strides = vec![0isize; rank];
            for (i, stride) in in_strides.iter_mut().enumerate() {
                if i >= pad {
                    let id = i - pad;
                    *stride = if in_shape[id] == 1 {
                        0
                    } else {
                        real_strides[id]
                    };
                }
            }
            let out_n: usize = target.iter().product();
            let mut out = Vec::with_capacity(out_n);
            let mut idx = vec![0usize; rank];
            for _ in 0..out_n {
                let fi: isize = (0..rank).map(|i| idx[i] as isize * in_strides[i]).sum();
                out.push(input[fi as usize]);
                for i in (0..rank).rev() {
                    idx[i] += 1;
                    if idx[i] < target[i] {
                        break;
                    }
                    idx[i] = 0;
                }
            }
            out
        }

        /// Deterministic batched f64 matmul `a[..batch, M, K] · b[..batch, K, N]`
        /// (same-rank ≥ 2, aligned batch). BOTH parity sides share the SAME
        /// `MatMul(a, b)` binds, so identical accumulation order ⇒ identical
        /// bits.
        fn matmul(a: &[f64], ash: &[usize], b: &[f64], bsh: &[usize]) -> Vec<f64> {
            let r = ash.len();
            let (m, k, n) = (ash[r - 2], ash[r - 1], bsh[r - 1]);
            let batch: usize = ash[..r - 2].iter().product();
            let (a_bs, b_bs, o_bs) = (m * k, k * n, m * n);
            let mut out = vec![0.0f64; batch * o_bs];
            for bi in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f64;
                        for kk in 0..k {
                            acc += a[bi * a_bs + i * k + kk] * b[bi * b_bs + kk * n + j];
                        }
                        out[bi * o_bs + i * n + j] = acc;
                    }
                }
            }
            out
        }

        /// Tiny f64 reference interpreter over the fused-linear primitive
        /// vocabulary (leaf-lookup FIRST, then `MatMul`, metadata-only
        /// `Reshape` pad, right-aligned `BroadcastTo`, elementwise `Add`). BOTH
        /// parity sides run through it with identical arithmetic, so a bit-exact
        /// assert isolates recipe STRUCTURE — the D4 pad `Reshape` (recipe only)
        /// is a passthrough here.
        fn eval_fl(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::MatMul => {
                    let a = eval_fl(g, node.inputs[0], leaves);
                    let b = eval_fl(g, node.inputs[1], leaves);
                    let ash: Vec<usize> = g.node(node.inputs[0]).shape.dims().to_vec();
                    let bsh: Vec<usize> = g.node(node.inputs[1]).shape.dims().to_vec();
                    matmul(&a, &ash, &b, &bsh)
                }
                // Metadata-only leading-1 pad (recipe's D4) — values unchanged.
                Op::Reshape(_) => eval_fl(g, node.inputs[0], leaves),
                Op::BroadcastTo(target) => {
                    let input = eval_fl(g, node.inputs[0], leaves);
                    let in_shape: Vec<usize> = g.node(node.inputs[0]).shape.dims().to_vec();
                    broadcast(&input, &in_shape, target.dims())
                }
                Op::Add => {
                    let a = eval_fl(g, node.inputs[0], leaves);
                    let b = eval_fl(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x + y).collect()
                }
                other => panic!("eval_fl: unhandled op {other:?}"),
            }
        }

        /// Build a fused FusedLinear node over `a [a_dims]`, `b [b_dims]`, and a
        /// rank-1 `bias [N]` (N = b's last dim). Returns `(a, b, bias, fused)`.
        fn fused_linear_fused_node(
            g: &mut Graph,
            a_dims: &[usize],
            b_dims: &[usize],
        ) -> (NodeId, NodeId, NodeId, NodeId) {
            let ar = a_dims.len();
            let n = b_dims[b_dims.len() - 1];
            let mut out_dims = a_dims[..ar - 2].to_vec();
            out_dims.push(a_dims[ar - 2]);
            out_dims.push(n);
            let a = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(a_dims),
                dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(b_dims),
                dtype: DType::F32,
            });
            let bias = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[n]),
                dtype: DType::F32,
            });
            let fused = g.push(Node {
                op: Op::Fused(FusedOps::FUSED_LINEAR, FusedOpParams::FusedLinear),
                inputs: vec![a, b, bias],
                shape: Shape::from_dims(&out_dims),
                dtype: DType::F32,
            });
            (a, b, bias, fused)
        }

        /// S2-2 (a): ONE recipe datum decomposes at BOTH rank-2 and rank-3
        /// (batched) activations — the WithDim-derived broadcast target tracks
        /// the matmul output shape — and its numerics match the FROZEN legacy
        /// builder bit-exactly under the shared reference interpreter.
        #[test]
        fn fused_linear_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for (a_dims, b_dims) in [
                (vec![2usize, 3], vec![3usize, 4]),
                (vec![2, 2, 3], vec![2, 3, 4]),
            ] {
                let mut g = Graph::new();
                let (a, b, bias, fused) = fused_linear_fused_node(&mut g, &a_dims, &b_dims);
                let out_sh = g.node(fused).shape.clone();
                let new_root = fused_linear::decompose(&mut g, fused, &FusedOpParams::FusedLinear);
                assert_ne!(new_root, fused, "recipe decompose fires at a={a_dims:?}");
                assert_eq!(
                    g.node(new_root).shape,
                    out_sh,
                    "output = matmul output shape"
                );
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_fused_linear_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::FusedLinear,
                );

                let an: usize = a_dims.iter().product();
                let bn: usize = b_dims.iter().product();
                let bias_n = b_dims[b_dims.len() - 1];
                let a_data: Vec<f64> = (0..an)
                    .map(|i| ((i as f64) * 0.31).sin() * 1.5 - 0.2)
                    .collect();
                let b_data: Vec<f64> = (0..bn)
                    .map(|i| ((i as f64) * 0.47).cos() * 0.8 + 0.1)
                    .collect();
                let bias_data: Vec<f64> = (0..bias_n)
                    .map(|i| ((i as f64) * 0.7).sin() * 2.0)
                    .collect();
                let mut leaves = HashMap::new();
                leaves.insert(a, a_data);
                leaves.insert(b, b_data);
                leaves.insert(bias, bias_data);
                let got = eval_fl(&g, new_root, &leaves);
                let want = eval_fl(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (x, y)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "fused_linear[{i}] a={a_dims:?}: recipe={x} vs legacy={y}",
                    );
                }
            }
        }

        /// S2-2 (structural, born-red): the WithDim-derived broadcast target
        /// rank-raises the rank-1 bias to the matmul output `[M,N]`, so emit
        /// materializes exactly one leading-1 pad `Reshape` (the D4 path, driven
        /// LIVE by WithDim). The imperative body broadcast the rank-1 bias
        /// directly (zero Reshape) — so this is RED until migration. The bias
        /// broadcast target equals the matmul output shape.
        #[test]
        fn fused_linear_recipe_drives_withdim_broadcast_with_a_d4_pad() {
            let mut g = Graph::new();
            let (_a, _b, _bias, fused) = fused_linear_fused_node(&mut g, &[2, 3], &[3, 4]);
            let root = fused_linear::decompose(&mut g, fused, &FusedOpParams::FusedLinear);
            assert_ne!(root, fused, "recipe decompose fires");
            assert!(
                matches!(g.node(root).op, Op::Add),
                "root is Add(mm, bias_bcst)"
            );
            let reachable = crate::topo_order_multi(&g, &[root]);
            let reshapes = reachable
                .iter()
                .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                .count();
            assert_eq!(
                reshapes, 1,
                "WithDim rank-raise materializes exactly one D4 pad Reshape on the bias",
            );
            let bcst = reachable
                .iter()
                .find(|&&n| matches!(g.node(n).op, Op::BroadcastTo(_)))
                .expect("bias broadcast present");
            assert_eq!(
                g.node(*bcst).shape.dims(),
                &[2, 4],
                "bias broadcast target = the matmul output shape (WithDim resolved)",
            );
        }

        /// S2-2 (totality): a wrong params payload is a typed decline surfaced
        /// as a fixpoint (G2), never a panic, declining BEFORE any emission.
        #[test]
        fn fused_linear_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_a, _b, _bias, fused) = fused_linear_fused_node(&mut g, &[2, 3], &[3, 4]);
            let before = g.len();
            let out = fused_linear::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    /// Increment C, B1 — the `Op::Scan` structural re-emit machinery. These
    /// tests prove a hand-built `Op::Scan` recipe (an `OpTag::Scan` node whose
    /// body sub-graph — `ScanPlaceholder` holes + body ops — rides its trailing
    /// operands, per the Phase-1 lax.scan encoding) round-trips through
    /// `tag_to_op`/`emit` to a graph whose base map is bit-identical to the
    /// same scan built imperatively (the way `selective_scan::decompose` emits
    /// it). `Op::Scan` stays a base-map terminal — the recipe just RE-EMITS it.
    ///
    /// Structure-preserving: the recipe mirrors the imperative
    /// `unroll_scan`-shaped scan node exactly (same op types, same params, same
    /// body structure, same const-sharing via a shared `Bind`), so the
    /// `base_map_hash` equality is a structural-identity proof. Neither
    /// `Op::Scan` nor `Op::ScanPlaceholder` folds its own node shape into
    /// `base_map_hash` (op_key tags 210/211 fold only their params + the body
    /// via child recursion), so emit's fallback interior shapes are inert for
    /// the identity — the base map matches by construction.
    mod scan_recipe_roundtrip {
        use super::super::*;
        use super::{bind, op_node};
        use crate::opt::base_map_hash;
        use crate::{ScanEmit, ScanPredicate, ScanRole};
        use fuel_ir::storage::{OutputViewSpec, compose_bundle};
        use fuel_ir::{DType, Shape};
        use fuel_kernel_seam_types::{SCAN_ROLE_CARRY, SCAN_ROLE_ELEM};
        use std::sync::Arc;

        fn placeholder(role: u8, index: u32) -> PatternNode {
            PatternNode::Op {
                op: OpTag::ScanPlaceholder,
                attrs: OpAttrs {
                    scan_role: Some(role),
                    scan_index: Some(index),
                    ..OpAttrs::default()
                },
                operands: vec![],
            }
        }

        /// The recipe: `scan(n_xs=1, bound=3, emit=All)` over
        /// `[init(B0), xs0(B1), const(B2), body_new_carry, body_y]` where
        /// `body_new_carry = Add(carry_hole, elem_hole)` and
        /// `body_y = Mul(body_new_carry, const)`. The `Add(carry,elem)` sub-tree
        /// appears in BOTH body exits — emit's slot-free identity-share collapses
        /// them to one node, matching the imperative `sum` referenced twice.
        fn scan_recipe() -> PatternNode {
            let sum = || {
                op_node(
                    OpTag::Add,
                    OpAttrs::default(),
                    vec![
                        placeholder(SCAN_ROLE_CARRY, 0),
                        placeholder(SCAN_ROLE_ELEM, 0),
                    ],
                )
            };
            PatternNode::Op {
                op: OpTag::Scan,
                attrs: OpAttrs {
                    scan_n_xs: Some(1),
                    scan_bound: Some(3),
                    scan_emit: Some(0), // All
                    scan_early_exit: None,
                    ..OpAttrs::default()
                },
                operands: vec![
                    bind(0),                                                       // init_carry
                    bind(1),                                                       // xs0
                    bind(2),                                                       // const
                    sum(),                                                         // body_new_carry
                    op_node(OpTag::Mul, OpAttrs::default(), vec![sum(), bind(2)]), // body_y
                ],
            }
        }

        /// Build the reference scan imperatively (the shape/layout
        /// `selective_scan::decompose` + `unroll_scan` assume) and return the
        /// scan NodeId. Leaves are unpopulated `Op::Const`s — same shape/dtype
        /// as the emitted recipe's, so they hash identically.
        fn reference_scan(g: &mut Graph) -> NodeId {
            let cs = Shape::from_dims(&[1]);
            let xs = Shape::from_dims(&[3, 1]);
            let init = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let xs0 = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: xs,
                dtype: DType::F32,
            });
            let konst = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let carry = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let elem = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 0,
                },
                inputs: vec![],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let sum = g.push(Node {
                op: Op::Add,
                inputs: vec![carry, elem],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::Mul,
                inputs: vec![sum, konst],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            g.push(Node {
                op: Op::Scan {
                    n_xs: 1,
                    bound: 3,
                    emit: ScanEmit::All,
                    early_exit: None,
                },
                inputs: vec![init, xs0, konst, sum, y],
                shape: Shape::from_dims(&[3, 1]),
                dtype: DType::F32,
            })
        }

        /// `tag_to_op` reconstructs the `Op::Scan` params + `ScanPlaceholder`
        /// role/index verbatim from the carriers.
        #[test]
        fn tag_to_op_reconstructs_scan_and_placeholder() {
            let scan_attrs = OpAttrs {
                scan_n_xs: Some(1),
                scan_bound: Some(3),
                scan_emit: Some(0),
                scan_early_exit: None,
                ..OpAttrs::default()
            };
            assert!(matches!(
                tag_to_op(OpTag::Scan, &scan_attrs),
                Some(Op::Scan {
                    n_xs: 1,
                    bound: 3,
                    emit: ScanEmit::All,
                    early_exit: None
                }),
            ));
            let fin = OpAttrs {
                scan_emit: Some(1),
                scan_early_exit: Some(true),
                ..scan_attrs.clone()
            };
            assert!(matches!(
                tag_to_op(OpTag::Scan, &fin),
                Some(Op::Scan {
                    emit: ScanEmit::Final,
                    early_exit: Some(ScanPredicate),
                    ..
                }),
            ));
            // A scan node with no bound is an honest miss (unset required attr).
            assert!(tag_to_op(OpTag::Scan, &OpAttrs::default()).is_none());
            let carry = OpAttrs {
                scan_role: Some(SCAN_ROLE_CARRY),
                scan_index: Some(0),
                ..OpAttrs::default()
            };
            assert!(matches!(
                tag_to_op(OpTag::ScanPlaceholder, &carry),
                Some(Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0
                }),
            ));
            let elem = OpAttrs {
                scan_role: Some(SCAN_ROLE_ELEM),
                scan_index: Some(2),
                ..OpAttrs::default()
            };
            assert!(matches!(
                tag_to_op(OpTag::ScanPlaceholder, &elem),
                Some(Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 2
                }),
            ));
            assert!(tag_to_op(OpTag::ScanPlaceholder, &OpAttrs::default()).is_none());
        }

        /// The recipe is a valid, representable region: contiguous binds `[0,1,2]`
        /// and every op (incl. `Scan`/`ScanPlaceholder`) re-emits to a primitive.
        #[test]
        fn scan_recipe_validates_as_representable() {
            assert_eq!(scan_recipe().bind_indices(), vec![0, 1, 2]);
            assert!(
                validate_recipe(&scan_recipe()).is_ok(),
                "scan recipe is a total, representable region"
            );
        }

        /// THE round-trip: emit the recipe onto fresh input leaves and assert its
        /// base map is bit-identical to the imperatively-built scan.
        #[test]
        fn scan_recipe_reemits_to_the_same_base_map_as_the_imperative_scan() {
            // Reference.
            let mut gr = Graph::new();
            let ref_scan = reference_scan(&mut gr);
            let want = base_map_hash(&gr, ref_scan);

            // Emitted from the recipe onto identical leaves.
            let mut ge = Graph::new();
            let cs = Shape::from_dims(&[1]);
            let init = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: cs.clone(),
                dtype: DType::F32,
            });
            let xs0 = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 1]),
                dtype: DType::F32,
            });
            let konst = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: cs,
                dtype: DType::F32,
            });
            let root = emit_region(&mut ge, &scan_recipe(), &[init, xs0, konst], &[]);
            assert!(
                matches!(
                    ge.node(root).op,
                    Op::Scan {
                        n_xs: 1,
                        bound: 3,
                        emit: ScanEmit::All,
                        early_exit: None
                    }
                ),
                "recipe re-emits an Op::Scan terminal"
            );
            let got = base_map_hash(&ge, root);

            assert_eq!(
                got, want,
                "scan recipe re-emit base map == imperative scan base map"
            );
        }

        // ---- The View / output_views bundle half (B1) ----------------------
        //
        // A scan recipe whose root is an `Op::View{slot}` must re-emit the scan,
        // attach the 2-slot `output_views` bundle (slot 0 = stacked ys, slot 1 =
        // final carry — the SelectiveScan bundle contract), and project the
        // slot — round-tripping to the same base map AND the correct slot shape
        // as the imperative scan+view. Real shapes throughout (init/const via
        // Binds carry real shapes; `body_y = Mul(const, sum)` takes the const's
        // shape) so the emitted bundle is byte-faithful.

        /// `view(slot)(scan(n_xs=1, bound=3, emit=All))` over binds `[init[2],
        /// xs0[3,2], const[2]]`. `body_y = Mul(const, sum)` is const-first so its
        /// re-emitted shape is `[2]` (the const's), making slot 0 = `[3,2]` and
        /// slot 1 = `[2]`.
        fn view_scan_recipe(slot: u32) -> PatternNode {
            let sum = || {
                op_node(
                    OpTag::Add,
                    OpAttrs::default(),
                    vec![
                        placeholder(SCAN_ROLE_CARRY, 0),
                        placeholder(SCAN_ROLE_ELEM, 0),
                    ],
                )
            };
            let scan = PatternNode::Op {
                op: OpTag::Scan,
                attrs: OpAttrs {
                    scan_n_xs: Some(1),
                    scan_bound: Some(3),
                    scan_emit: Some(0),
                    scan_early_exit: None,
                    ..OpAttrs::default()
                },
                operands: vec![
                    bind(0),                                                       // init_carry [2]
                    bind(1),                                                       // xs0 [3,2]
                    bind(2),                                                       // const [2]
                    sum(),                                                         // body_new_carry
                    op_node(OpTag::Mul, OpAttrs::default(), vec![bind(2), sum()]), // body_y = Mul(const, sum)
                ],
            };
            op_node(
                OpTag::View,
                OpAttrs {
                    view_slot: Some(slot),
                    ..OpAttrs::default()
                },
                vec![scan],
            )
        }

        /// The imperative reference: the scan + its composed 2-slot bundle + the
        /// slot projection (mirrors `NodeHandle::scan` + `Graph::view`).
        fn reference_view(g: &mut Graph, slot: u32) -> NodeId {
            let two = Shape::from_dims(&[2]);
            let init = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let xs0 = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            let konst = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let carry = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Carry,
                    index: 0,
                },
                inputs: vec![],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let elem = g.push(Node {
                op: Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 0,
                },
                inputs: vec![],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let sum = g.push(Node {
                op: Op::Add,
                inputs: vec![carry, elem],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let body_y = g.push(Node {
                op: Op::Mul,
                inputs: vec![konst, sum],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let ys = Shape::from_dims(&[3, 2]);
            let scan = g.push(Node {
                op: Op::Scan {
                    n_xs: 1,
                    bound: 3,
                    emit: ScanEmit::All,
                    early_exit: None,
                },
                inputs: vec![init, xs0, konst, sum, body_y],
                shape: ys.clone(),
                dtype: DType::F32,
            });
            let specs = vec![
                OutputViewSpec::contiguous(DType::F32, ys),
                OutputViewSpec::contiguous(DType::F32, two),
            ];
            let (_bytes, views) = compose_bundle(&specs).expect("compose_bundle");
            g.set_output_views(scan, Arc::from(views.into_boxed_slice()))
                .expect("set_output_views");
            let (sh, dt) = {
                let v = g.output_views(scan).expect("bundle set");
                let s = &v[slot as usize];
                (s.shape.clone(), s.dtype)
            };
            g.push(Node {
                op: Op::View { slot },
                inputs: vec![scan],
                shape: sh,
                dtype: dt,
            })
        }

        fn roundtrip_view(slot: u32, want_shape: &[usize]) {
            // Reference.
            let mut gr = Graph::new();
            let ref_view = reference_view(&mut gr, slot);
            let want = base_map_hash(&gr, ref_view);

            // Emitted from the recipe onto identical leaves.
            let mut ge = Graph::new();
            let two = Shape::from_dims(&[2]);
            let init = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: two.clone(),
                dtype: DType::F32,
            });
            let xs0 = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 2]),
                dtype: DType::F32,
            });
            let konst = ge.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: two,
                dtype: DType::F32,
            });
            let root = emit_region(&mut ge, &view_scan_recipe(slot), &[init, xs0, konst], &[]);

            // The View projects the requested slot with the bundle-derived shape.
            assert!(
                matches!(ge.node(root).op, Op::View { slot: s } if s == slot),
                "recipe root re-emits Op::View{{slot={slot}}}"
            );
            assert_eq!(
                ge.node(root).shape.dims(),
                want_shape,
                "view slot {slot} shape comes from the re-attached output_views bundle"
            );
            // The producing scan carries the faithful 2-slot bundle.
            let producer = ge.node(root).inputs[0];
            assert!(
                matches!(ge.node(producer).op, Op::Scan { .. }),
                "producer is the scan"
            );
            assert_eq!(
                ge.output_views(producer).map(|v| v.len()),
                Some(2),
                "emit re-attaches the 2-slot bundle so slot>=1 views resolve"
            );

            let got = base_map_hash(&ge, root);
            assert_eq!(
                got, want,
                "view+scan recipe re-emit base map == imperative scan+view base map"
            );
        }

        /// Slot 0 (stacked ys): its shape equals the scan node's primary shape,
        /// so even the child-fallback would find it — but it must come through
        /// the bundle for the layout/offset to be right.
        #[test]
        fn scan_view_slot0_reemits_to_the_same_base_map() {
            roundtrip_view(0, &[3, 2]);
        }

        /// Slot 1 (final carry / last_state) is the load-bearing case: its shape
        /// (`[2]`) lives ONLY in the bundle and differs from the scan's primary
        /// (`[3,2]`). `op_key(View) = None` folds the node shape into the hash,
        /// so a wrong slot-1 shape would break the round-trip — this proves the
        /// bundle is re-attached and read.
        #[test]
        fn scan_view_slot1_reemits_to_the_same_base_map() {
            roundtrip_view(1, &[2]);
        }

        /// A view recipe validates as a representable region (contiguous binds,
        /// every op — incl. `View`/`Scan`/`ScanPlaceholder` — re-emittable).
        #[test]
        fn view_scan_recipe_validates_as_representable() {
            assert_eq!(view_scan_recipe(1).bind_indices(), vec![0, 1, 2]);
            assert!(validate_recipe(&view_scan_recipe(1)).is_ok());
            // tag_to_op reconstructs the View op from `view_slot`.
            assert!(matches!(
                tag_to_op(
                    OpTag::View,
                    &OpAttrs {
                        view_slot: Some(1),
                        ..OpAttrs::default()
                    }
                ),
                Some(Op::View { slot: 1 }),
            ));
            assert!(tag_to_op(OpTag::View, &OpAttrs::default()).is_none());
        }

        /// B2 emit carrier (born-red without it): a `ScanPlaceholder` that
        /// declares its per-step shape via `target_shape_rel` emits with THAT
        /// shape, not the rank-0 `primitive_shape` fallback. This is
        /// load-bearing for the SSM scan recipes — `unroll_scan` clones body
        /// interior nodes with their STORED shapes, so a rank-0 placeholder
        /// would poison the unrolled body (e.g. `du = Mul(d_t, u_t)` rank-0).
        /// Dtype comes from bind 0 (the uniform-dtype scan inputs).
        #[test]
        fn scan_placeholder_recipe_emits_its_declared_shape() {
            use fuel_kernel_seam_types::shape_expr::{Dim, ShapeExpr};
            let dims2 = || {
                ShapeExpr::Dims(vec![
                    Dim::Extent {
                        operand: 0,
                        axis: 0,
                    },
                    Dim::Extent {
                        operand: 0,
                        axis: 1,
                    },
                ])
            };
            let ph = |index: u32| PatternNode::Op {
                op: OpTag::ScanPlaceholder,
                attrs: OpAttrs {
                    scan_role: Some(SCAN_ROLE_ELEM),
                    scan_index: Some(index),
                    target_shape_rel: Some(dims2()),
                    ..OpAttrs::default()
                },
                operands: vec![],
            };
            // Mul takes operand[0]'s shape, so its shape mirrors the placeholder.
            // A `Bind { 0 }` operand supplies the frame the placeholder's Dims
            // reference (operand 0) — so the recipe carries a valid bind.
            let recipe = op_node(OpTag::Mul, OpAttrs::default(), vec![ph(0), bind(0)]);
            assert!(validate_recipe(&recipe).is_ok());
            let mut g = Graph::new();
            let bind0 = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 3]),
                dtype: DType::F32,
            });
            let root = emit_region(&mut g, &recipe, &[bind0], &[]);
            let ph_id = g.node(root).inputs[0];
            assert!(matches!(
                g.node(ph_id).op,
                Op::ScanPlaceholder {
                    role: ScanRole::Elem,
                    index: 0
                }
            ));
            assert_eq!(
                g.node(ph_id).shape.dims(),
                &[2, 3],
                "placeholder emits its declared target_shape_rel shape, not rank-0",
            );
            assert_eq!(
                g.node(ph_id).dtype,
                DType::F32,
                "placeholder dtype = bind 0's"
            );
            assert_eq!(
                g.node(root).shape.dims(),
                &[2, 3],
                "Mul(placeholder,..) inherits the declared shape"
            );
        }
    }

    // reduce_max_to_backward migration (Increment C carriers, A2) -------------
    //
    // The 9-node imperative fair-share max subgradient (ReduceMaxTo→BroadcastTo→
    // Equal(U8)→MaskedFill→ReduceSumTo→Div→BroadcastTo→Mul, sharing the single
    // `mask_f`) becomes a portable `PatternNode` DATA recipe, and the FIRST recipe
    // to carry a `MaskedFill` — driven LIVE by the A2 fill-Scalar re-emit carrier
    // (the fill dtype re-resolves to x's dtype, matching `Scalar::one(dtype)`).
    // A DIRECT structural mirror (no D3 keepdim swap, no D4 pad), so the parity
    // oracle runs both sides through a toy f64 interpreter and asserts BIT-EXACT
    // structure — the discriminating power is confirmed by the sabotage-calibrated
    // negative in `reduce_max_to_backward_recipe_shares_mask_and_has_no_reshape`
    // (identity-share ⇒ ONE MaskedFill node across its two uses). Binds:
    // `0 = x`, `1 = up`. The reduce/count targets are `SameAs 1` (upstream's
    // shape); the broadcasts are `SameAs 0` (x's shape).
    mod reduce_max_to_backward_recipe {
        use super::super::*;
        use super::frozen_legacy_reduce_max_to_backward_decompose;
        use crate::registry::{FusedOps, reduce_max_to_backward};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Right-aligned NumPy broadcast of `input` (shape `in_shape`) to
        /// `target` — a size-1 or padded leading dim contributes stride 0. General
        /// on purpose (both `bcast_x` sites tile a `[..,1]` keepdim back to x).
        fn broadcast(input: &[f64], in_shape: &[usize], target: &[usize]) -> Vec<f64> {
            let rank = target.len();
            let pad = rank - in_shape.len();
            let mut real_strides = vec![0isize; in_shape.len()];
            let mut s = 1isize;
            for i in (0..in_shape.len()).rev() {
                real_strides[i] = s;
                s *= in_shape[i] as isize;
            }
            let mut in_strides = vec![0isize; rank];
            for (i, stride) in in_strides.iter_mut().enumerate() {
                if i >= pad {
                    let id = i - pad;
                    *stride = if in_shape[id] == 1 {
                        0
                    } else {
                        real_strides[id]
                    };
                }
            }
            let out_n: usize = target.iter().product();
            let mut out = Vec::with_capacity(out_n);
            let mut idx = vec![0usize; rank];
            for _ in 0..out_n {
                let fi: isize = (0..rank).map(|i| idx[i] as isize * in_strides[i]).sum();
                out.push(input[fi as usize]);
                for i in (0..rank).rev() {
                    idx[i] += 1;
                    if idx[i] < target[i] {
                        break;
                    }
                    idx[i] = 0;
                }
            }
            out
        }

        /// Tiny f64 reference interpreter over the reduce-max-backward primitive
        /// vocabulary (leaf-lookup FIRST, then elementwise `Mul`/`Div`,
        /// `MulScalar`, `Equal` (1.0/0.0), the `MaskedFill` fill from the op's
        /// `Scalar`, last-axis `ReduceMaxTo`/`ReduceSumTo`, right-aligned
        /// `BroadcastTo`). BOTH parity sides run through it with identical in-order
        /// arithmetic, so a bit-exact assert isolates recipe STRUCTURE. Test dims
        /// reduce the LAST axis (`up = [.., 1]`), so the reduces are a last-axis
        /// chunk fold.
        fn eval_rmb(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::Mul => {
                    let a = eval_rmb(g, node.inputs[0], leaves);
                    let b = eval_rmb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::Div => {
                    let a = eval_rmb(g, node.inputs[0], leaves);
                    let b = eval_rmb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x / y).collect()
                }
                Op::MulScalar(m) => eval_rmb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * m)
                    .collect(),
                Op::Equal => {
                    let a = eval_rmb(g, node.inputs[0], leaves);
                    let b = eval_rmb(g, node.inputs[1], leaves);
                    a.iter()
                        .zip(&b)
                        .map(|(x, y)| if x == y { 1.0 } else { 0.0 })
                        .collect()
                }
                Op::MaskedFill { value } => {
                    let input = eval_rmb(g, node.inputs[0], leaves);
                    let mask = eval_rmb(g, node.inputs[1], leaves);
                    let fill = value.to_f64();
                    input
                        .iter()
                        .zip(&mask)
                        .map(|(x, m)| if *m != 0.0 { fill } else { *x })
                        .collect()
                }
                // Last-axis reduce-to-shape (up = [.., 1]); identical fold both
                // parity sides.
                Op::ReduceMaxTo(_) => {
                    let input = eval_rmb(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                        .collect()
                }
                Op::ReduceSumTo(_) => {
                    let input = eval_rmb(g, node.inputs[0], leaves);
                    let last = *g.node(node.inputs[0]).shape.dims().last().unwrap();
                    input
                        .chunks(last)
                        .map(|row| row.iter().sum::<f64>())
                        .collect()
                }
                Op::BroadcastTo(target) => {
                    let input = eval_rmb(g, node.inputs[0], leaves);
                    let in_shape: Vec<usize> = g.node(node.inputs[0]).shape.dims().to_vec();
                    broadcast(&input, &in_shape, target.dims())
                }
                other => panic!("eval_rmb: unhandled op {other:?}"),
            }
        }

        /// Build a fused ReduceMaxToBackward node over `x [x_dims]` (input 0) and
        /// `up [up_dims]` (input 1, the upstream gradient = the forward reduce
        /// target). Returns `(x, up, fused)`.
        fn rmb_fused_node(
            g: &mut Graph,
            x_dims: &[usize],
            up_dims: &[usize],
        ) -> (NodeId, NodeId, NodeId) {
            let xsh = Shape::from_dims(x_dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: xsh.clone(),
                dtype: DType::F32,
            });
            let up = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(up_dims),
                dtype: DType::F32,
            });
            let fused = g.push(Node {
                op: Op::Fused(
                    FusedOps::REDUCE_MAX_TO_BACKWARD,
                    FusedOpParams::ReduceMaxToBackward,
                ),
                inputs: vec![x, up],
                shape: xsh,
                dtype: DType::F32,
            });
            (x, up, fused)
        }

        /// A2 (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3 (the
        /// shape polymorphism the baked-shape legacy body never had), and its
        /// numerics match the FROZEN legacy builder bit-exactly under the shared
        /// reference interpreter — the MaskedFill carrier driving live emission.
        #[test]
        fn reduce_max_to_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            for (x_dims, up_dims) in [
                (vec![2usize, 4], vec![2usize, 1]),
                (vec![3, 5, 7], vec![3, 5, 1]),
            ] {
                let mut g = Graph::new();
                let (x, up, fused) = rmb_fused_node(&mut g, &x_dims, &up_dims);
                let xsh = Shape::from_dims(&x_dims);
                let new_root = reduce_max_to_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::ReduceMaxToBackward,
                );
                assert_ne!(new_root, fused, "recipe decompose must fire at {x_dims:?}");
                assert_eq!(
                    g.node(new_root).shape,
                    xsh,
                    "reduce_max backward is x-shaped"
                );
                assert_eq!(g.node(new_root).dtype, DType::F32);

                let legacy_root = frozen_legacy_reduce_max_to_backward_decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::ReduceMaxToBackward,
                );

                let xn: usize = x_dims.iter().product();
                let upn: usize = up_dims.iter().product();
                let x_data: Vec<f64> = (0..xn)
                    .map(|i| ((i as f64) * 0.37).sin() * 2.0 + 0.3)
                    .collect();
                let up_data: Vec<f64> = (0..upn)
                    .map(|i| ((i as f64) * 0.53).cos() * 1.1 - 0.2)
                    .collect();
                let mut leaves = HashMap::new();
                leaves.insert(x, x_data);
                leaves.insert(up, up_data);
                let got = eval_rmb(&g, new_root, &leaves);
                let want = eval_rmb(&g, legacy_root, &leaves);
                assert_eq!(got.len(), want.len());
                for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "reduce_max_to_backward[{i}] at {x_dims:?}: recipe={a} vs legacy={b}",
                    );
                }
            }
        }

        /// A2 (structural): the single `mask_f` is IDENTITY-SHARED across its two
        /// use sites (`ReduceSumTo` and the final `Mul`) — exactly ONE MaskedFill
        /// node reachable, the same single-compute DAG the imperative body had (a
        /// duplicated emit would break this). A direct structural mirror: no D3/D4
        /// `Reshape`, and the MaskedFill fill Scalar re-resolved to x's (F32)
        /// dtype (evidence the A2 carrier reached emit).
        #[test]
        fn reduce_max_to_backward_recipe_shares_mask_and_has_no_reshape() {
            for (x_dims, up_dims) in [
                (vec![2usize, 4], vec![2usize, 1]),
                (vec![3, 5, 7], vec![3, 5, 1]),
            ] {
                let mut g = Graph::new();
                let (_x, _up, fused) = rmb_fused_node(&mut g, &x_dims, &up_dims);
                let root = reduce_max_to_backward::decompose(
                    &mut g,
                    fused,
                    &FusedOpParams::ReduceMaxToBackward,
                );
                assert_ne!(root, fused, "recipe decompose fires at {x_dims:?}");
                assert!(
                    matches!(g.node(root).op, Op::Mul),
                    "root is Mul(mask_f, share_b)"
                );
                let reachable = crate::topo_order_multi(&g, &[root]);
                let masked = reachable
                    .iter()
                    .filter(|&&n| matches!(g.node(n).op, Op::MaskedFill { .. }))
                    .count();
                assert_eq!(
                    masked, 1,
                    "mask_f identity-shared across its two uses at {x_dims:?} (ONE MaskedFill)",
                );
                let reshapes = reachable
                    .iter()
                    .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                    .count();
                assert_eq!(
                    reshapes, 0,
                    "direct structural mirror — no D3/D4 Reshape at {x_dims:?}"
                );
                let fill_dtype = reachable
                    .iter()
                    .find_map(|&n| match &g.node(n).op {
                        Op::MaskedFill { value } => Some(value.dtype()),
                        _ => None,
                    })
                    .expect("a MaskedFill is present");
                assert_eq!(
                    fill_dtype,
                    DType::F32,
                    "the A2 carrier re-resolved the fill Scalar to x's dtype at {x_dims:?}",
                );
            }
        }

        /// A2 (totality): a wrong params payload is a typed decline surfaced as a
        /// fixpoint (G2), never a panic, declining BEFORE any emission.
        #[test]
        fn reduce_max_to_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, _up, fused) = rmb_fused_node(&mut g, &[2, 4], &[2, 1]);
            let before = g.len();
            let out = reduce_max_to_backward::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }

    // powi_backward migration (Increment C carriers, A3) ----------------------
    //
    // The 3-node imperative closed-form gradient `grad_x = exp · x^(exp-1) ·
    // upstream` (PowI(exp-1)→MulScalar(exp)→Mul) becomes a portable `PatternNode`
    // DATA recipe, and the FIRST recipe to carry a `PowI` — driven LIVE by the A3
    // i32-exponent re-emit carrier (the exponent rides scalars[0]) — and the first
    // whose STRUCTURE is param-derived (both `exp` and `exp-1` are constant-folded
    // into the datum at build time, the minimal "param-derived const in the
    // recipe" C-4 posture, NOT a restructure to dodge the missing thread). A
    // DIRECT structural mirror (no D3 keepdim swap, no D4 pad), so the parity
    // oracle runs both sides through a toy f64 interpreter and asserts BIT-EXACT
    // structure across ranks AND exponents. Binds: `0 = x`, `1 = up`.
    mod powi_backward_recipe {
        use super::super::*;
        use super::frozen_legacy_powi_backward_decompose;
        use crate::registry::{FusedOps, powi_backward};
        use fuel_ir::{DType, Shape};
        use std::collections::HashMap;

        /// Tiny f64 reference interpreter over the powi-backward primitive
        /// vocabulary (leaf-lookup FIRST, then elementwise `Mul`, `MulScalar`,
        /// and integer-power `PowI` via `f64::powi`). BOTH parity sides run
        /// through it with identical in-order arithmetic, so a bit-exact assert
        /// isolates recipe STRUCTURE — the exponent transported by the A3 carrier
        /// and the baked `MulScalar`. `x` and `up` are the same shape (`PowI` is
        /// elementwise), so every op is a flat elementwise map.
        fn eval_pb(g: &Graph, id: NodeId, leaves: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
            if let Some(v) = leaves.get(&id) {
                return v.clone();
            }
            let node = g.node(id);
            match &node.op {
                Op::Mul => {
                    let a = eval_pb(g, node.inputs[0], leaves);
                    let b = eval_pb(g, node.inputs[1], leaves);
                    a.iter().zip(&b).map(|(x, y)| x * y).collect()
                }
                Op::MulScalar(m) => eval_pb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v * m)
                    .collect(),
                Op::PowI(n) => eval_pb(g, node.inputs[0], leaves)
                    .iter()
                    .map(|v| v.powi(*n))
                    .collect(),
                other => panic!("eval_pb: unhandled op {other:?}"),
            }
        }

        /// Build a fused PowIBackward node over `x [dims]` (input 0) and `up
        /// [dims]` (input 1, the upstream gradient — same shape as x), carrying
        /// `exp`. Returns `(x, up, fused)`.
        fn pb_fused_node(g: &mut Graph, dims: &[usize], exp: i32) -> (NodeId, NodeId, NodeId) {
            let sh = Shape::from_dims(dims);
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let up = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: sh.clone(),
                dtype: DType::F32,
            });
            let fused = g.push(Node {
                op: Op::Fused(FusedOps::POWI_BACKWARD, FusedOpParams::PowIBackward { exp }),
                inputs: vec![x, up],
                shape: sh,
                dtype: DType::F32,
            });
            (x, up, fused)
        }

        /// A3 (a): ONE recipe datum decomposes at BOTH rank 2 and rank 3 and
        /// across several exponents, and its numerics match the FROZEN legacy
        /// builder bit-exactly under the shared reference interpreter — the A3
        /// PowI carrier driving live emission of `Op::PowI(exp-1)`.
        #[test]
        fn powi_backward_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
            // x strictly positive so exp==0's PowI(-1) reciprocal is finite/clean;
            // the exponents span +, the exp==1 identity, exp==0 (MulScalar(0)=0),
            // and a negative exponent (PowI(-3)).
            for dims in [vec![2usize, 4], vec![3, 5, 7]] {
                for exp in [3i32, 2, 5, 1, 0, -2] {
                    let mut g = Graph::new();
                    let (x, up, fused) = pb_fused_node(&mut g, &dims, exp);
                    let sh = Shape::from_dims(&dims);
                    let new_root = powi_backward::decompose(
                        &mut g,
                        fused,
                        &FusedOpParams::PowIBackward { exp },
                    );
                    assert_ne!(
                        new_root, fused,
                        "recipe decompose must fire at {dims:?} exp={exp}"
                    );
                    assert_eq!(
                        g.node(new_root).shape,
                        sh,
                        "powi backward is shape-preserving"
                    );
                    assert_eq!(g.node(new_root).dtype, DType::F32);

                    let legacy_root = frozen_legacy_powi_backward_decompose(
                        &mut g,
                        fused,
                        &FusedOpParams::PowIBackward { exp },
                    );

                    let n: usize = dims.iter().product();
                    let x_data: Vec<f64> = (0..n)
                        .map(|i| ((i as f64) * 0.41).sin() * 0.5 + 1.5)
                        .collect();
                    let up_data: Vec<f64> = (0..n)
                        .map(|i| ((i as f64) * 0.61).cos() * 1.3 - 0.15)
                        .collect();
                    let mut leaves = HashMap::new();
                    leaves.insert(x, x_data);
                    leaves.insert(up, up_data);
                    let got = eval_pb(&g, new_root, &leaves);
                    let want = eval_pb(&g, legacy_root, &leaves);
                    assert_eq!(got.len(), want.len());
                    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "powi_backward[{i}] at {dims:?} exp={exp}: recipe={a} vs legacy={b}",
                        );
                    }
                }
            }
        }

        /// A3 (structural): a DIRECT structural mirror — the root is `Mul(scaled,
        /// up)`, the A3 carrier resolves the exponent to `Op::PowI(exp-1)`, the
        /// scale is a baked `Op::MulScalar(exp as f64)`, and NO D3/D4 `Reshape` is
        /// materialized (evidence the i32 carrier + baked scale reached emit at the
        /// right values).
        #[test]
        fn powi_backward_recipe_is_a_direct_mirror_with_carrier_exponent() {
            for exp in [3i32, 0, -2] {
                let mut g = Graph::new();
                let (_x, _up, fused) = pb_fused_node(&mut g, &[2, 4], exp);
                let root =
                    powi_backward::decompose(&mut g, fused, &FusedOpParams::PowIBackward { exp });
                assert_ne!(root, fused, "recipe decompose fires at exp={exp}");
                assert!(
                    matches!(g.node(root).op, Op::Mul),
                    "root is Mul(scaled, up)"
                );
                let reachable = crate::topo_order_multi(&g, &[root]);
                // The A3 carrier reconstructed exactly one Op::PowI(exp-1).
                let powis: Vec<i32> = reachable
                    .iter()
                    .filter_map(|&n| match g.node(n).op {
                        Op::PowI(k) => Some(k),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    powis,
                    vec![exp - 1],
                    "the A3 carrier resolved Op::PowI(exp-1) at exp={exp}"
                );
                // The scale is a single baked MulScalar(exp as f64).
                let mul_scalars: Vec<f64> = reachable
                    .iter()
                    .filter_map(|&n| match g.node(n).op {
                        Op::MulScalar(v) => Some(v),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    mul_scalars,
                    vec![exp as f64],
                    "baked MulScalar(exp) at exp={exp}"
                );
                // Direct mirror — no keepdim/pad Reshape.
                let reshapes = reachable
                    .iter()
                    .filter(|&&n| matches!(g.node(n).op, Op::Reshape(_)))
                    .count();
                assert_eq!(
                    reshapes, 0,
                    "direct structural mirror — no D3/D4 Reshape at exp={exp}"
                );
            }
        }

        /// A3 (totality): a wrong params payload is a typed decline surfaced as a
        /// fixpoint (G2), never a panic, declining BEFORE any emission.
        #[test]
        fn powi_backward_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
            let mut g = Graph::new();
            let (_x, _up, fused) = pb_fused_node(&mut g, &[2, 4], 3);
            let before = g.len();
            let out = powi_backward::decompose(&mut g, fused, &FusedOpParams::Rope);
            assert_eq!(out, fused, "wrong params ⇒ typed decline ⇒ fixpoint");
            assert_eq!(g.len(), before, "declined before any emission");
        }
    }
}

//! FusedOpRegistry — metadata side. Phase 7.6 step 1 (skeleton).
//!
//! Architecture v1.0 splits op identity into two layers:
//! - the closed [`Op`] enum carries primitive variants exhaustively;
//! - one arm — `Op::Fused(FusedOpId, FusedOpParams)` — delegates to an
//!   open registry of fused-op entries populated at process startup.
//!
//! This module holds the *graph-side metadata* half of the registry:
//! identity ([`FusedOpId`], [`FusedOps`]), pattern + decomposition + backward
//! identity, and shape/dtype rules. The *kernel side* (BackendImpl with
//! its function-pointer KernelRef, cost estimate, PrecisionGuarantee, caps,
//! revision hash) lives in `fuel-storage::fused` because it carries
//! `KernelRef`, which fuel-graph cannot import without inverting the
//! existing fuel-storage → fuel-graph dependency direction. The two halves
//! are joined at runtime by [`FusedOpId`].
//!
//! See [docs/fused-op-registry.md](../../docs/fused-op-registry.md) for the
//! full design and [docs/architecture/03-ir.md](../../docs/architecture/03-ir.md)
//! for the architectural commitment.
//!
//! ## Status (step 1)
//!
//! Types only. No callers; no behavior change. Subsequent steps:
//! - Step 2: extend `Op` with `Op::Fused(FusedOpId, FusedOpParams)` arm.
//! - Step 3: migrate SoftmaxLastDim end-to-end through the registry.
//! - Step 4: migrate the remaining 12 fused ops.
//! - Step 5: drop the per-fused-op `Op` variants once nothing emits them.

use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};
use std::collections::HashMap;

pub mod conv2d;
pub mod conv_transpose_2d;
pub mod flash_attn;
pub mod fused_linear;
pub mod inplace_affine;
pub mod layer_norm_last_dim;
pub mod layer_norm_last_dim_backward;
pub mod paged_attn;
pub mod powi_backward;
pub mod qmatmul;
pub mod reduce_max_to_backward;
pub mod rms_norm_last_dim;
pub mod rms_norm_last_dim_backward;
pub mod rope;
pub mod softmax_last_dim;
pub mod softmax_last_dim_backward;

/// Stable identifier for a registered fused op. Indexes into
/// [`FusedOpRegistry::entries`]. Newtype over `u16` (~65K capacity is
/// plenty; today's catalog is 13-14 entries).
///
/// Constants for the well-known ids are exposed via [`FusedOps`]
/// associated constants for ergonomic pattern matching in rule code.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FusedOpId(pub u16);

/// Reserved sentinel: an unallocated id. Never appears in a populated
/// registry; useful as a placeholder during step-2-and-step-3 migration
/// where a `BackwardKind::Fused(id)` may be written before the backward's
/// own id is assigned.
impl FusedOpId {
    pub const UNASSIGNED: FusedOpId = FusedOpId(0);
}

/// Per-fused-op metadata. Identity, pattern, decomposition, backward,
/// shape/dtype rules. The kernel-side metadata (per-backend KernelRef +
/// cost + PrecisionGuarantee + caps) lives in `fuel-storage::fused` and is
/// joined to this entry by [`FusedOpEntry::id`].
pub struct FusedOpEntry {
    /// Stable id for this fused op.
    pub id: FusedOpId,
    /// Stable human-readable name. Shows up in op_short_name, error
    /// messages, telemetry.
    pub name: &'static str,
    /// Categorical bucket. Used by telemetry and for cost-model defaults.
    pub family: FusedOpFamily,
    /// Canonical primitive subgraph this fused op represents. Used by
    /// fusion rules to recognize the pattern in a base map.
    pub pattern: SubgraphPattern,
    /// Decompose a fused-op node into its primitive subgraph. Used by
    /// lowering rules (and by autograd when [`Self::backward`] is
    /// `BackwardKind::Decompose`).
    ///
    /// Contract: the function appends primitive nodes to `graph` and
    /// returns the [`NodeId`] of the new root that replaces the fused
    /// node identified by the second argument. The fused node itself
    /// remains in the arena; the driver rewrites consumer edges to point
    /// at the returned id.
    pub decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,
    /// Backward identity for this fused op.
    pub backward: BackwardKind,
    /// Output shape rule, computed from input shapes and params.
    pub shape_rule: fn(&[Shape], &FusedOpParams) -> Shape,
    /// Output dtype rule, computed from input dtypes and params.
    pub dtype_rule: fn(&[DType], &FusedOpParams) -> DType,
}

/// Categorical bucket for a fused op. Drives telemetry grouping and
/// some cost-model defaults; not load-bearing for dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FusedOpFamily {
    Forward,
    Backward,
    Quantized,
    Attention,
    Norm,
}

/// Per-instance parameters carried by `Op::Fused(id, params)` nodes.
/// Step 1 ships only the SoftmaxLastDim variant — the proof-of-concept
/// migration target. Step 4 extends this enum with the remaining 12
/// fused-op variants, one per migrated op.
///
/// The variant tag is implicitly aligned with the registry entry's
/// [`FusedOpId`] via the convention that registering an id places its
/// matching variant here. CSE and `op_key` derive a hashable key from
/// the tuple `(FusedOpId, FusedOpParams)` so identical fused-op nodes
/// with identical params dedupe.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOpParams {
    /// SoftmaxLastDim has no per-instance parameters; the last-dim axis
    /// is implicit in the input shape.
    SoftmaxLastDim,
    /// FusedLinear ((a @ b) + bias). No per-instance parameters: the
    /// matmul shape and bias rank are implicit in the three inputs.
    FusedLinear,
    /// RmsNormLastDim — carries the epsilon term used to stabilize the
    /// division by RMS magnitude. The last-dim axis itself is implicit
    /// in the input shape.
    RmsNormLastDim { eps: f64 },
    /// LayerNormLastDim — carries the epsilon term used to stabilize
    /// the division by sqrt(variance + eps). Like RmsNormLastDim, the
    /// last-dim axis is implicit in the input shape.
    LayerNormLastDim { eps: f64 },
    /// Rope — rotary position embedding with caller-supplied cos/sin
    /// tables. No per-instance parameters: `seq` and `head_dim` are
    /// recovered from the inputs at execution time (see
    /// `fuel-storage::pipelined::op_to_op_params`).
    Rope,
    /// Conv2D — 2-D cross-correlation with stride / padding / groups.
    /// Spatial dilation is always 1 today; the field is omitted from
    /// the param payload until backward's dilation-as-stride trick
    /// forces the primitive variant to grow one (per the comments
    /// in `Tensor::backward`'s `Op::Conv2D` arm).
    Conv2D {
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    },
    /// SoftmaxLastDimBackward — `s * (g - sum(g * s, last, keepdim))`.
    /// Two inputs (forward_y, upstream); parameterless. Higher-order
    /// gradients panic per `Tensor::backward`'s MVP behavior.
    SoftmaxLastDimBackward,
    /// LayerNormLastDimBackward — recomputes mean/variance from the
    /// original x. Two inputs (x, upstream) + eps.
    LayerNormLastDimBackward { eps: f64 },
    /// RmsNormLastDimBackward — `r_rms · (g − x·s / (n·(mean_sq + eps)))`.
    /// Two inputs (x, upstream) + eps.
    RmsNormLastDimBackward { eps: f64 },
    /// ReduceMaxToBackward — routes upstream to argmax positions
    /// (fair-share on ties). Two inputs (x, upstream); parameterless.
    ReduceMaxToBackward,
    /// PowIBackward — `grad_x = exp · x^(exp-1) · upstream`. Two
    /// inputs `(x, upstream)`. Carries the same `exp` as the forward
    /// `Op::PowI(exp)` so the backward kernel can evaluate the
    /// derivative coefficient directly.
    PowIBackward { exp: i32 },
    /// ConvTranspose2D — fractionally-strided 2-D convolution.
    /// Two inputs `[x, weight]` (no bias). Carries the full
    /// stride/padding/output_padding/dilation/groups bundle.
    ConvTranspose2D {
        stride:         (usize, usize),
        padding:        (usize, usize),
        output_padding: (usize, usize),
        dilation:       (usize, usize),
        groups:         usize,
    },
    /// FlashAttn — multi-head scaled-dot-product attention with
    /// FlashAttention-shaped kernel hooks. 4-5 inputs (q, k, v,
    /// optional alibi).
    FlashAttn {
        softmax_scale:     f32,
        causal:            bool,
        window_size_left:  Option<usize>,
        window_size_right: Option<usize>,
        softcap:           Option<f32>,
    },
    /// PagedAttn — paged-cache attention. 5-6 inputs (q, k_cache,
    /// v_cache, block_table, context_lens, optional alibi).
    PagedAttn {
        softmax_scale: f32,
        block_size:    usize,
        softcap:       Option<f32>,
    },
    /// QMatMul — quantized matrix multiply `C = A @ dequant(W_Q)`.
    /// Two inputs `[a, w_q_bytes]`. The U32-typed w_q_bytes carries
    /// a Q-type block stream; `quant_type` selects the dequant
    /// implementation; `k` / `n` are the logical weight dims.
    QMatMul {
        quant_type: crate::QuantType,
        k:          usize,
        n:          usize,
    },
    /// InplaceAffine — `x = mul·x + add`, mutating input 0. Single
    /// input. Destructive on index 0 (marked via
    /// `Op::destructive_input`). Phase 1 of the in-place ops
    /// infrastructure.
    InplaceAffine { mul: f64, add: f64 },
}

/// Hashable key for [`FusedOpParams`]. Used by `op_key`/CSE so that two
/// `Op::Fused(id, params)` nodes with identical params dedupe.
///
/// The encoding is variant-tag + payload as a `Vec<u64>` (bit patterns
/// for floats, repr for ints/usize). Mirrors the existing
/// [`crate::opt`]-style `OpKey` encoding so future extensions slot in
/// without rebuilding the CSE map.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedOpParamsKey {
    pub tag: u16,
    pub bits: Vec<u64>,
    pub ints: Vec<i64>,
}

impl FusedOpParams {
    /// Hashable encoding. Step 4 adds variants per-fused-op as each
    /// migrates; the tag uniquely identifies the variant and the
    /// bits/ints slots carry any payload (none for parameterless ops
    /// like SoftmaxLastDim and FusedLinear).
    pub fn key(&self) -> FusedOpParamsKey {
        match self {
            FusedOpParams::SoftmaxLastDim => FusedOpParamsKey {
                tag: 1,
                bits: Vec::new(),
                ints: Vec::new(),
            },
            FusedOpParams::FusedLinear => FusedOpParamsKey {
                tag: 2,
                bits: Vec::new(),
                ints: Vec::new(),
            },
            FusedOpParams::RmsNormLastDim { eps } => FusedOpParamsKey {
                tag: 3,
                // Encode eps as its raw bit pattern so CSE on two
                // RmsNormLastDim nodes with the same eps dedupes
                // (and two with different eps don't).
                bits: vec![eps.to_bits()],
                ints: Vec::new(),
            },
            FusedOpParams::LayerNormLastDim { eps } => FusedOpParamsKey {
                tag: 4,
                bits: vec![eps.to_bits()],
                ints: Vec::new(),
            },
            FusedOpParams::Rope => FusedOpParamsKey {
                tag: 5,
                bits: Vec::new(),
                ints: Vec::new(),
            },
            FusedOpParams::Conv2D { stride, padding, groups } => FusedOpParamsKey {
                tag: 6,
                bits: Vec::new(),
                // Stride.{0,1}, padding.{0,1}, groups. Five i64 slots
                // — enough that two Conv2D nodes with different
                // strides or paddings hash to distinct CSE keys.
                ints: vec![
                    stride.0 as i64,
                    stride.1 as i64,
                    padding.0 as i64,
                    padding.1 as i64,
                    *groups as i64,
                ],
            },
            FusedOpParams::SoftmaxLastDimBackward => FusedOpParamsKey {
                tag: 7,
                bits: Vec::new(),
                ints: Vec::new(),
            },
            FusedOpParams::LayerNormLastDimBackward { eps } => FusedOpParamsKey {
                tag: 8,
                bits: vec![eps.to_bits()],
                ints: Vec::new(),
            },
            FusedOpParams::RmsNormLastDimBackward { eps } => FusedOpParamsKey {
                tag: 9,
                bits: vec![eps.to_bits()],
                ints: Vec::new(),
            },
            FusedOpParams::ReduceMaxToBackward => FusedOpParamsKey {
                tag: 10,
                bits: Vec::new(),
                ints: Vec::new(),
            },
            FusedOpParams::ConvTranspose2D {
                stride, padding, output_padding, dilation, groups,
            } => FusedOpParamsKey {
                tag: 11,
                bits: Vec::new(),
                // 9 i64 slots: stride.{0,1}, padding.{0,1},
                // output_padding.{0,1}, dilation.{0,1}, groups.
                ints: vec![
                    stride.0 as i64,
                    stride.1 as i64,
                    padding.0 as i64,
                    padding.1 as i64,
                    output_padding.0 as i64,
                    output_padding.1 as i64,
                    dilation.0 as i64,
                    dilation.1 as i64,
                    *groups as i64,
                ],
            },
            FusedOpParams::FlashAttn {
                softmax_scale, causal, window_size_left,
                window_size_right, softcap,
            } => FusedOpParamsKey {
                tag: 12,
                // softmax_scale f32 → u64 bit pattern; softcap
                // f32 → u64 bit pattern (NaN sentinel for None so
                // two FlashAttn nodes with no softcap still hash
                // equal).
                bits: vec![
                    softmax_scale.to_bits() as u64,
                    softcap.map(|s| s.to_bits() as u64).unwrap_or(u64::MAX),
                ],
                // causal as i64; window sizes carry MIN as the
                // "None" sentinel so a real Some(0) still distinguishes.
                ints: vec![
                    if *causal { 1 } else { 0 },
                    window_size_left.map(|w| w as i64).unwrap_or(i64::MIN),
                    window_size_right.map(|w| w as i64).unwrap_or(i64::MIN),
                ],
            },
            FusedOpParams::PagedAttn {
                softmax_scale, block_size, softcap,
            } => FusedOpParamsKey {
                tag: 13,
                bits: vec![
                    softmax_scale.to_bits() as u64,
                    softcap.map(|s| s.to_bits() as u64).unwrap_or(u64::MAX),
                ],
                ints: vec![*block_size as i64],
            },
            FusedOpParams::QMatMul { quant_type, k, n } => FusedOpParamsKey {
                tag: 14,
                bits: Vec::new(),
                // QuantType encoded as stable per-variant i64; k, n
                // as usize.
                ints: vec![quant_type_key(*quant_type), *k as i64, *n as i64],
            },
            FusedOpParams::PowIBackward { exp } => FusedOpParamsKey {
                tag: 15,
                bits: Vec::new(),
                ints: vec![*exp as i64],
            },
            FusedOpParams::InplaceAffine { mul, add } => FusedOpParamsKey {
                tag: 16,
                bits: vec![mul.to_bits(), add.to_bits()],
                ints: Vec::new(),
            },
        }
    }
}

/// Stable per-variant key for [`crate::QuantType`]. Used by
/// [`FusedOpParams::key`] so CSE on two `Op::Fused(QMATMUL, _)`
/// nodes with the same `(quant_type, k, n)` collapses correctly.
fn quant_type_key(q: crate::QuantType) -> i64 {
    use crate::QuantType::*;
    match q {
        Q4_0   => 1,
        Q4_1   => 2,
        Q5_0   => 3,
        Q5_1   => 4,
        Q8_0   => 5,
        Q8_1   => 6,
        Q2K    => 7,
        Q3K    => 8,
        Q4_K_M => 9,
        Q5K    => 10,
        Q6K    => 11,
    }
}

/// What kind of backward this fused op has.
///
/// - `Fused(id)` — emit a fused-backward op with the given id. Used by
///   ops whose backward is awkward to express as a primitive
///   decomposition (e.g. SoftmaxLastDimBackward's fused subtract-and-
///   project formula).
/// - `Decompose` — autograd derives the backward from the primitive
///   decomposition (lower the fused node, then run autograd over the
///   primitives).
/// - `NotDifferentiable` — backward returns Err / panics cleanly.
///   Matches today's QMatMul / ArgMaxDim treatment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackwardKind {
    Fused(FusedOpId),
    Decompose,
    NotDifferentiable,
}

/// Canonical primitive-subgraph pattern that a fused op represents.
/// Used by fusion rules.
///
/// Two reasonable shapes; the registry supports both:
/// - `Callable` — a closure-style matcher, mirrors PR 3's
///   `canonical_softmax_pattern`. Maximally flexible (consumer-count
///   guards, cross-checks); less analyzable.
/// - `Declarative` — a recursive `PatternTree` that the rule engine
///   compiles to a matcher. More analyzable; auto-generation from the
///   registry entry is straightforward.
///
/// Step 3 ports SoftmaxLastDim's existing matcher to `Callable`. The
/// declarative form (see Q1 in `docs/fused-op-registry.md`) is wired up
/// in step 4 once a second op exercises the abstraction.
pub enum SubgraphPattern {
    Declarative(PatternTree),
    Callable(fn(&Graph, NodeId) -> Option<PatternMatch>),
}

/// Placeholder for the declarative pattern tree (Q1 of the design doc).
/// Step 1 ships the type so the `SubgraphPattern::Declarative` arm
/// compiles; the actual recursive shape (`Op + Vec<Pattern>` with
/// variables) is filled in alongside the second op's migration in
/// step 4.
#[derive(Debug, Clone, Default)]
pub struct PatternTree {
    /// Reserved for future expansion. The empty placeholder type keeps
    /// this enum variant valid in step 1 without locking in a shape
    /// before a second op forces the design.
    _reserved: (),
}

/// Result of a successful pattern match. Carries the bindings — the
/// concrete `NodeId`s that the pattern's variables matched against in
/// the host graph — plus the [`FusedOpParams`] payload the fusion rule
/// should stamp onto the emitted `Op::Fused(id, params)` node.
///
/// `bindings` is index-keyed: bindings sorted by index become the
/// fused-op node's input list (`inputs[0]` ← index 0, `inputs[1]` ←
/// index 1, …). SoftmaxLastDim's match has one binding `(0, x_id)`
/// and so emits a single-input node; FusedLinear (3 inputs:
/// `[a, b, bias]`) emits three bindings indexed 0–2.
///
/// `params` is the matcher's authority on the resulting fused-op's
/// per-instance parameters. The matcher knows what variant of
/// [`FusedOpParams`] it's recognizing; carrying that decision in the
/// match result keeps [`crate::opt::FusionRule::rewrite`] generic
/// across all registered fused ops.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// Variable-id → resolved NodeId. The fusion rule sorts by index
    /// and uses the resolved ids in order as the emitted node's inputs.
    pub bindings: Vec<(usize, NodeId)>,
    /// Per-instance parameters the matcher stamps onto the fused-op
    /// node it produces. Parameterless ops (SoftmaxLastDim, FusedLinear)
    /// stamp their unit variant; parameterized ops (RmsNormLastDim,
    /// FlashAttn) recover their payload from the matched subgraph.
    pub params: FusedOpParams,
}

/// The metadata-side registry. Built at process startup, frozen
/// thereafter (architecture v1.0: no runtime extensibility). Lookups
/// are by [`FusedOpId`] (O(1)), by name, or by pattern hash.
#[derive(Default)]
pub struct FusedOpRegistry {
    entries: Vec<FusedOpEntry>,
    by_name: HashMap<&'static str, FusedOpId>,
    /// Reserved for the declarative-pattern path (step 4). The fusion
    /// driver hashes a base-map subgraph to a `PatternHash` and looks
    /// it up here to decide which fused-op to emit. Unused in step 1.
    #[allow(dead_code)]
    by_pattern_hash: HashMap<PatternHash, FusedOpId>,
}

impl FusedOpRegistry {
    /// Empty registry. Step 3 builds one with `with_entry(...)` calls.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an entry. Returns self for builder-style chaining.
    /// Idempotent on (id, name); a duplicate id is a programming bug
    /// (asserts in debug builds, last-writer-wins in release).
    ///
    /// Pattern-hash indexing is reserved for step 4's declarative
    /// pattern engine; the `Declarative` arm wires into it then.
    pub fn with_entry(mut self, entry: FusedOpEntry) -> Self {
        debug_assert!(
            !self.by_name.contains_key(entry.name),
            "FusedOpRegistry: duplicate entry name {:?}",
            entry.name
        );
        debug_assert!(
            entry.id.0 as usize == self.entries.len() + 1
                || self.entry(entry.id).is_none(),
            "FusedOpRegistry: id {:?} already populated",
            entry.id
        );
        self.by_name.insert(entry.name, entry.id);
        // Grow `entries` so direct id-indexing works. Slot 0 stays
        // reserved (FusedOpId::UNASSIGNED).
        let slot = entry.id.0 as usize;
        if self.entries.len() <= slot {
            self.entries.reserve(slot + 1 - self.entries.len());
            // Step 1 doesn't need to fill placeholder slots — production
            // paths register every id starting from 1 contiguously, so
            // the Vec stays dense. If a gap arises, the lookup helper
            // returns `None` for missing ids.
        }
        // Insert at the matching index when contiguous; otherwise
        // append. Step 3's first registration is FusedOpId(1), so the
        // simple append-with-slot-0-empty path is what we'll exercise.
        if self.entries.is_empty() {
            // Reserve slot 0 with a unit placeholder. Direct indexing is
            // the hot path; we accept one wasted slot.
            self.entries.push(placeholder_entry());
        }
        if slot < self.entries.len() {
            self.entries[slot] = entry;
        } else {
            // Pad if necessary, then push.
            while self.entries.len() < slot {
                self.entries.push(placeholder_entry());
            }
            self.entries.push(entry);
        }
        self
    }

    /// Look up an entry by id. Returns `None` for `FusedOpId::UNASSIGNED`
    /// or any unregistered id.
    pub fn entry(&self, id: FusedOpId) -> Option<&FusedOpEntry> {
        let slot = id.0 as usize;
        if slot == 0 || slot >= self.entries.len() {
            return None;
        }
        let e = &self.entries[slot];
        if e.id == FusedOpId::UNASSIGNED {
            None
        } else {
            Some(e)
        }
    }

    /// Look up an entry by name. Returns the [`FusedOpId`] when present.
    pub fn id_for_name(&self, name: &str) -> Option<FusedOpId> {
        self.by_name.get(name).copied()
    }

    /// Iterate over every registered (non-placeholder) entry. Used by
    /// [`crate::opt::RuleRegistry::default_rules`] to auto-generate
    /// one [`crate::opt::LoweringRule`] + [`crate::opt::FusionRule`]
    /// pair per fused op without naming each entry by hand.
    pub fn entries_iter(&self) -> impl Iterator<Item = &FusedOpEntry> {
        self.entries
            .iter()
            .filter(|e| e.id != FusedOpId::UNASSIGNED)
    }

    /// Number of registered (non-placeholder) entries.
    pub fn len(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.id != FusedOpId::UNASSIGNED)
            .count()
    }

    /// Whether the registry has any registered entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hash of a primitive-subgraph pattern. Reserved for step 4's
/// declarative pattern engine. Step 1 ships the newtype so the
/// `by_pattern_hash` index field type-checks; the hashing function
/// itself is filled in alongside `PatternTree`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PatternHash(pub u64);

/// Internal placeholder used to keep id-indexed `entries` dense. Never
/// returned by [`FusedOpRegistry::entry`].
fn placeholder_entry() -> FusedOpEntry {
    FusedOpEntry {
        id: FusedOpId::UNASSIGNED,
        name: "<unassigned>",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(|_g, _id| None),
        decompose: |_g, id, _p| id,
        backward: BackwardKind::NotDifferentiable,
        shape_rule: |_shapes, _params| Shape::from_dims(&[]),
        dtype_rule: |_dtypes, _params| DType::F32,
    }
}

/// Associated constants for the well-known fused-op ids. Rule code
/// matches `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, _)` — almost as
/// ergonomic as today's `Op::SoftmaxLastDim`. Steps 3-4 fill these in
/// as each fused op migrates to the registry.
pub struct FusedOps;

impl FusedOps {
    /// SoftmaxLastDim — proof-of-concept migration target (step 3).
    pub const SOFTMAX_LAST_DIM: FusedOpId = FusedOpId(1);
    /// FusedLinear — first multi-input fused op migrated to the
    /// registry (step 4). Three inputs `[a, b, bias]`, output
    /// `(a @ b) + bias`. The CUTLASS bias-epilogue integration in the
    /// baracuda-cutlass alpha.13 plan registers here.
    pub const FUSED_LINEAR: FusedOpId = FusedOpId(2);
    /// RmsNormLastDim — `x / sqrt(mean(x²) + eps)` along the last dim.
    /// Migrated in step 4 (continued). Single input + eps param.
    pub const RMS_NORM_LAST_DIM: FusedOpId = FusedOpId(3);
    /// LayerNormLastDim — `(x - mean) / sqrt(var + eps)` along the
    /// last dim (no affine). Single input + eps param.
    pub const LAYER_NORM_LAST_DIM: FusedOpId = FusedOpId(4);
    /// Rope — rotary position embedding with caller-supplied cos/sin
    /// tables. Three inputs `[x, cos, sin]`, parameterless (the seq/
    /// head_dim shape parameters live in the input shapes).
    pub const ROPE: FusedOpId = FusedOpId(5);
    /// Conv2D — 2-D cross-correlation. Two or three inputs `[x,
    /// weight]` or `[x, weight, bias]`; carries stride / padding /
    /// groups in [`FusedOpParams::Conv2D`].
    pub const CONV2D: FusedOpId = FusedOpId(6);
    /// SoftmaxLastDimBackward — fused backward helper for the
    /// SoftmaxLastDim forward. Two inputs `[y, upstream]`,
    /// parameterless. Wired through `BackwardKind::Fused` from
    /// `SOFTMAX_LAST_DIM`'s entry.
    pub const SOFTMAX_LAST_DIM_BACKWARD: FusedOpId = FusedOpId(7);
    /// LayerNormLastDimBackward — fused backward helper for the
    /// LayerNormLastDim forward. Two inputs `[x, upstream]` + eps.
    pub const LAYER_NORM_LAST_DIM_BACKWARD: FusedOpId = FusedOpId(8);
    /// RmsNormLastDimBackward — fused backward helper for the
    /// RmsNormLastDim forward. Two inputs `[x, upstream]` + eps.
    pub const RMS_NORM_LAST_DIM_BACKWARD: FusedOpId = FusedOpId(9);
    /// ReduceMaxToBackward — fused backward helper for the
    /// `Op::ReduceMaxTo` primitive (routes upstream to argmax
    /// positions, fair-share on ties). Two inputs `[x, upstream]`,
    /// parameterless. Note this is the backward of a *primitive*
    /// (`Op::ReduceMaxTo`), not of a fused forward — there's no
    /// `BackwardKind::Fused` edge wired to this helper, autograd
    /// reaches it directly from `Op::ReduceMaxTo`'s arm.
    pub const REDUCE_MAX_TO_BACKWARD: FusedOpId = FusedOpId(10);
    /// ConvTranspose2D — fractionally-strided 2-D convolution.
    /// Two inputs `[x, weight]` (no bias); carries
    /// stride/padding/output_padding/dilation/groups in
    /// [`FusedOpParams::ConvTranspose2D`].
    pub const CONV_TRANSPOSE2D: FusedOpId = FusedOpId(11);
    /// FlashAttn — multi-head attention with FlashAttention-shaped
    /// kernel hooks. 4-5 inputs (q, k, v, optional alibi).
    pub const FLASH_ATTN: FusedOpId = FusedOpId(12);
    /// PagedAttn — paged-cache attention. 5-6 inputs.
    pub const PAGED_ATTN: FusedOpId = FusedOpId(13);
    /// QMatMul — quantized matmul `C = A @ dequant(W_Q)`. Two
    /// inputs `[a, w_q_bytes]`.
    pub const QMATMUL: FusedOpId = FusedOpId(14);
    /// PowIBackward — fused backward helper for `Op::PowI(n)`.
    /// `grad_x = n · x^(n-1) · upstream`. Two inputs `[x, upstream]`,
    /// carries `exp: i32` in [`FusedOpParams::PowIBackward`]. Replaces
    /// the autograd primitive decomposition (PowI(n-1) → MulScalar →
    /// Mul, 3 nodes) with a single launch of baracuda alpha.31's
    /// `unary_powi_backward_*` kernel. Note this is the backward of a
    /// primitive (`Op::PowI`), not of a fused forward — autograd
    /// reaches it directly from `Op::PowI`'s arm rather than via a
    /// `BackwardKind::Fused` edge.
    pub const POWI_BACKWARD: FusedOpId = FusedOpId(15);

    /// In-place affine: `x = mul·x + add`. Single input (the mutated
    /// tensor), single launch into baracuda's `affine_inplace_*`
    /// symbol on CUDA. Carries `mul: f64, add: f64` in
    /// [`FusedOpParams::InplaceAffine`]. Marked destructive on input
    /// 0 via `Op::destructive_input`, so `opt::derive_ordering` pins
    /// it to run after every non-destructive reader. Phase 1 of the
    /// in-place ops infrastructure
    /// (`docs/session-prompts/in-place-ops-infrastructure.md`);
    /// dispatch + autograd integration deferred to Phases 3 + 4.
    pub const INPLACE_AFFINE: FusedOpId = FusedOpId(16);
}

/// Process-wide default registry: the union of every fused op's
/// metadata-side entry. Built once on first access via
/// [`std::sync::OnceLock`]; immutable thereafter (architecture v1.0:
/// no runtime extensibility).
///
/// Step 3 populates only SoftmaxLastDim; step 4 fills in the other
/// 12 fused ops as they migrate.
pub fn default_registry() -> &'static FusedOpRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<FusedOpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        FusedOpRegistry::new()
            .with_entry(softmax_last_dim::entry())
            .with_entry(fused_linear::entry())
            .with_entry(rms_norm_last_dim::entry())
            .with_entry(layer_norm_last_dim::entry())
            .with_entry(rope::entry())
            .with_entry(conv2d::entry())
            .with_entry(softmax_last_dim_backward::entry())
            .with_entry(layer_norm_last_dim_backward::entry())
            .with_entry(rms_norm_last_dim_backward::entry())
            .with_entry(reduce_max_to_backward::entry())
            .with_entry(conv_transpose_2d::entry())
            .with_entry(flash_attn::entry())
            .with_entry(paged_attn::entry())
            .with_entry(qmatmul::entry())
            .with_entry(powi_backward::entry())
            .with_entry(inplace_affine::entry())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: id newtype is comparable + hashable.
    #[test]
    fn fused_op_id_basic() {
        let a = FusedOpId(1);
        let b = FusedOpId(1);
        let c = FusedOpId(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Smoke: empty registry has no entries.
    #[test]
    fn registry_empty() {
        let r = FusedOpRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.entry(FusedOpId(1)).is_none());
        assert!(r.entry(FusedOpId::UNASSIGNED).is_none());
        assert!(r.id_for_name("anything").is_none());
    }

    /// Smoke: register one entry, look it up by id and by name.
    #[test]
    fn registry_register_one() {
        fn dummy_decompose(_g: &mut Graph, id: NodeId, _p: &FusedOpParams) -> NodeId {
            id
        }
        fn dummy_pattern(_g: &Graph, _id: NodeId) -> Option<PatternMatch> {
            None
        }
        fn dummy_shape(_s: &[Shape], _p: &FusedOpParams) -> Shape {
            Shape::from_dims(&[1])
        }
        fn dummy_dtype(_d: &[DType], _p: &FusedOpParams) -> DType {
            DType::F32
        }

        let r = FusedOpRegistry::new().with_entry(FusedOpEntry {
            id: FusedOps::SOFTMAX_LAST_DIM,
            name: "SoftmaxLastDim",
            family: FusedOpFamily::Forward,
            pattern: SubgraphPattern::Callable(dummy_pattern),
            decompose: dummy_decompose,
            backward: BackwardKind::NotDifferentiable,
            shape_rule: dummy_shape,
            dtype_rule: dummy_dtype,
        });

        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
        let entry = r.entry(FusedOps::SOFTMAX_LAST_DIM).expect("registered");
        assert_eq!(entry.name, "SoftmaxLastDim");
        assert_eq!(entry.family, FusedOpFamily::Forward);
        assert_eq!(
            r.id_for_name("SoftmaxLastDim"),
            Some(FusedOps::SOFTMAX_LAST_DIM)
        );
    }

    /// FusedOpParams::key produces a stable encoding.
    #[test]
    fn fused_op_params_key_softmax() {
        let k1 = FusedOpParams::SoftmaxLastDim.key();
        let k2 = FusedOpParams::SoftmaxLastDim.key();
        assert_eq!(k1, k2);
        assert_eq!(k1.tag, 1);
        assert!(k1.bits.is_empty());
        assert!(k1.ints.is_empty());
    }
}

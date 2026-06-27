//! # fuel-graph
//!
//! Lazy computation graph for Phase 6 of the fuel ML framework.
//!
//! ## Role
//!
//! Phase 6 transitions fuel from eager execution to a lazy computation graph
//! with an autonomous router. This crate holds the minimal vocabulary that
//! makes that transition possible:
//!
//! - [`Op`] — a closed enum describing every operation that can appear in a
//!   graph node. New ops are added here and consumed by every backend.
//! - [`Graph`] — an arena of [`Node`]s. Nodes are immutable once added and
//!   referenced by index. The arena grows; it never shrinks.
//! - [`Tensor`] — a cheaply-cloneable handle to a `(graph, node_id)` pair.
//!   Users build computations by calling methods on `Tensor`; each method
//!   appends a node to the underlying graph and returns a new handle.
//!
//! Execution is deliberately not implemented here. A separate executor crate
//! (today [`fuel_reference_backend::exec`](../fuel_reference_backend/exec/))
//! walks a `Graph` and produces concrete outputs. Future backends (CPU-fast,
//! CUDA, Metal, etc.) will plug in via the same pattern.
//!
//! ## Scope of the MVP
//!
//! This is the first landing of the graph types. It is deliberately minimal:
//!
//! - Four float dtypes: `f32`, `f64`, `bf16`, `f16`. Integer dtypes will be
//!   added alongside the reference backend's integer op coverage, not
//!   speculatively here.
//! - A small starter `Op` catalog: `Const`, `Add`, `Mul`, `MatMul`,
//!   `Transpose`, `Relu`, `Sqr`, `Exp`. More ops are added as validation
//!   tests for them land.
//! - Unfused backward autograd: [`Tensor::backward`] walks the forward
//!   graph in reverse, applies per-op gradient rules, and emits new graph
//!   nodes for the gradient of every leaf. Fused backward and symbolic
//!   graph rewriting are deferred to Phase 6d.
//! - No fusion, no planner. Those belong to later sub-phases of Phase 6.
//! - Thread-safe. The graph is wrapped in `Arc<RwLock<_>>` so that
//!   `fuel_graph::Tensor` (and any handle that embeds it, including
//!   `fuel_core::Tensor` post Phase 7.5 work item G) is `Send + Sync`.
//!   Borrow access goes through `read().unwrap()` / `write().unwrap()`.

pub mod grad;
pub mod opt;
pub mod registry;
pub mod run;

#[doc(inline)]
pub use run::{
    branch_density, branch_density_multi, branches_in_topo_order, extract_runs,
    extract_runs_multi, lower_picked_route, lower_run, lower_runs_arm0, passes_fewness_gate,
    PickedRoute, Run, FEWNESS_THRESHOLD,
};

use crate::registry::{FusedOpId, FusedOpParams};
use fuel_ir::{DeviceLocation, DType, DynScalar, Layout, Scalar, Shape, probe::BackendId};
use fuel_backend_contract::Storage;
use half::{bf16, f16};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Compute the topological order of every node reachable from `root`.
///
/// The returned order places inputs before the nodes that depend on them,
/// so iterating the list and caching each result as you go guarantees every
/// dependency is available when its dependents are computed. Uses an
/// explicit stack iterative post-order DFS — no recursion, so arbitrarily
/// deep graphs are safe.
///
/// This is the utility an executor walks to realize a tensor, and the
/// utility [`Tensor::backward`] walks in reverse to construct the backward
/// graph.
pub fn topo_order(graph: &Graph, root: NodeId) -> Vec<NodeId> {
    topo_order_multi(graph, &[root])
}

/// Compute the topological order of every node reachable from ANY of the
/// given roots. Each node appears in the output exactly once; the order
/// still satisfies the dependency-before-dependent property, so walking
/// it and caching results as you go computes every input before any node
/// that depends on it.
///
/// Used by the multi-output `realize_many` entry points in executors:
/// KV-cached inference wants the per-step logits AND the updated K/V
/// tensors for every layer as outputs of a single forward pass. Asking
/// the executor for them via `topo_order_multi` walks the graph exactly
/// once for the combined dependency set, then the executor reads each
/// requested root out of the cache.
pub fn topo_order_multi(graph: &Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let mut order = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    // Each stack entry is (node, processed). `processed = false` means
    // "push children then revisit me"; `processed = true` means "all my
    // children are already in `order`, add me now."
    let mut stack: Vec<(NodeId, bool)> = roots.iter().map(|&r| (r, false)).collect();
    while let Some((id, processed)) = stack.pop() {
        if processed {
            order.push(id);
            continue;
        }
        if visited.contains(&id) {
            continue;
        }
        visited.insert(id);
        stack.push((id, true));
        for &child in &graph.node(id).inputs {
            if !visited.contains(&child) {
                stack.push((child, false));
            }
        }
    }
    order
}

/// A node ID in the arena. Stable for the lifetime of the [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

/// Quantization block format for [`Op::QMatMul`]. Matches the
/// GGML/GGUF block layouts; each variant implies a fixed bytes-per-block
/// and elements-per-block. Only variants for which a backend has a
/// fused dequant-in-kernel matmul are currently exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// 32-element block = 2 bytes f16 scale + 16 bytes packed u4 quants.
    Q4_0,
    /// 32-element block = 2 bytes f16 d + 2 bytes f16 m + 16 bytes packed u4 quants.
    Q4_1,
    /// 32-element block = 2 bytes f16 d + 4 bytes high-bit + 16 bytes packed u4 quants.
    Q5_0,
    /// 32-element block = 2 bytes f16 d + 2 bytes f16 m + 4 bytes high-bit + 16 bytes packed u4.
    Q5_1,
    /// 32-element block = 2 bytes f16 scale + 32 bytes i8 quants.
    Q8_0,
    /// 32-element block = 2 bytes f16 d + 2 bytes f16 s + 32 bytes i8 quants.
    Q8_1,
    /// 256-element super-block = scales + 64-byte 2-bit packed quants + 4 bytes f16 d/dmin.
    Q2K,
    /// 256-element super-block = hmask + 64-byte 2-bit packed quants + 12-byte scales + 2 bytes f16 d.
    Q3K,
    /// 256-element super-block = 2 bytes f16 d + 2 bytes f16 dmin +
    /// 12 bytes of 6-bit-packed sub-block scales/mins + 128 bytes
    /// of 4-bit-packed quants. GGML k-quant "medium" format.
    Q4_K_M,
    /// 256-element super-block = d + dmin + 12-byte scales + 32-byte hmask + 128-byte qs.
    Q5K,
    /// 256-element super-block = 128-byte ql + 64-byte qh + 16-byte i8 scales + 2 bytes f16 d.
    Q6K,
}

impl QuantType {
    /// Bytes per quantization block. Mirrors `std::mem::size_of::<BlockQ*>()`.
    pub fn bytes_per_block(self) -> usize {
        match self {
            QuantType::Q4_0 => 18,
            QuantType::Q4_1 => 20,
            QuantType::Q5_0 => 22,
            QuantType::Q5_1 => 24,
            QuantType::Q8_0 => 34,
            QuantType::Q8_1 => 36,
            QuantType::Q2K => 84,
            QuantType::Q3K => 110,
            QuantType::Q4_K_M => 144,
            QuantType::Q5K => 176,
            QuantType::Q6K => 210,
        }
    }
    /// Elements per quantization block.
    pub fn elements_per_block(self) -> usize {
        match self {
            QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q8_0
            | QuantType::Q8_1 => 32,
            QuantType::Q2K
            | QuantType::Q3K
            | QuantType::Q4_K_M
            | QuantType::Q5K
            | QuantType::Q6K => 256,
        }
    }
}

/// Fill mode for [`Op::Pad`]. Only [`PadMode::Constant`] is
/// implemented in the v1 cut; the other variants are accepted at
/// the IR level so the enum is forward-stable, but the executor
/// returns a clean error until they ship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadMode {
    /// Fill padded slots with the constant `value` parameter on
    /// the [`Op::Pad`] variant.
    Constant,
    /// Reflect input around the padded edges (without repeating the
    /// edge value). Not yet implemented.
    Reflect,
    /// Repeat the edge value into the padded region. Not yet
    /// implemented.
    Replicate,
}

/// The closed enum of operations a graph node can represent.
///
/// This is the API contract between the graph layer and every backend.
/// Adding a new op means (1) adding a variant here, (2) teaching every
/// backend how to execute it, and (3) teaching the reference backend how
/// to compute its textbook answer.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // --- leaves ---
    /// A concrete constant tensor — a leaf with no input nodes. The
    /// realized bytes live in the graph's `storage_map` slot for this
    /// node, populated at construction time. The executor's slot-first
    /// dispatch returns the slot's `Arc<RwLock<Storage>>` directly.
    /// (Phase 7.5 work item G2 retired the host-side `ConstData`
    /// payload — the slot is the sole source of bytes.)
    Const,

    // --- element-wise binary ---
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction (`a - b`).
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division (`a / b`).
    Div,

    // --- element-wise unary ---
    /// Element-wise negation (`-x`).
    Neg,
    /// Element-wise square (`x * x`).
    Sqr,
    /// Element-wise square root.
    Sqrt,
    /// Element-wise exponential (`e^x`).
    Exp,
    /// Element-wise natural logarithm.
    Log,
    /// Element-wise sine.
    Sin,
    /// Element-wise cosine.
    Cos,
    /// Element-wise hyperbolic tangent.
    Tanh,
    /// Element-wise logistic sigmoid.
    Sigmoid,
    /// SiLU activation (x · sigmoid(x)), also called Swish.
    Silu,
    /// GELU activation (tanh approximation).
    Gelu,
    /// Rectified linear unit (`max(0, x)`), element-wise.
    Relu,
    /// Heaviside step function: `1` where `x > 0`, `0` otherwise. Serves
    /// as the derivative of `Relu` and as a building block for other
    /// comparison-based gradients.
    Step,
    /// Element-wise reciprocal (`1 / x`). On `x = 0` the IEEE-754 result
    /// is `±inf`; the kernel matches the obvious `1.0 / x` expression
    /// rather than treating the input as an error.
    Recip,
    /// Element-wise absolute value (`|x|`).
    Abs,

    // --- in-place elementwise unary (Phase 1 of the in-place ops
    // infrastructure landed 2026-05-30; see
    // `docs/session-prompts/in-place-ops-infrastructure.md`) ---
    //
    // Each variant mutates input 0 in place; output aliases input 0.
    // Pinned to run after every non-destructive reader of input 0 by
    // `opt::derive_ordering` via the `destructive_input() -> Some(0)`
    // contract. Backward integration deferred to Phase 4 (the
    // mutation-safety pass auto-clones tape-tracked inputs).
    /// In-place Relu: `x = max(0, x)`. Same semantics as `Op::Relu`,
    /// mutates input 0. Not yet wired through autograd.
    ReluInplace,
    /// In-place Silu: `x = x * sigmoid(x)`. Same semantics as
    /// `Op::Silu`, mutates input 0. Not yet wired through autograd.
    SiluInplace,
    /// In-place Gelu (tanh approximation): mutates input 0 with the
    /// same formula as `Op::Gelu`. Not yet wired through autograd.
    GeluInplace,
    /// In-place Tanh: mutates input 0. Same semantics as `Op::Tanh`.
    /// Not yet wired through autograd.
    TanhInplace,
    /// In-place Sigmoid: mutates input 0. Same semantics as
    /// `Op::Sigmoid`. Not yet wired through autograd.
    SigmoidInplace,

    // --- in-place elementwise unary, dtype expansion + op family
    // expansion shipped 2026-05-30. Same `destructive_input() ->
    // Some(0)` contract as the 5 above; backward emitters mirror
    // the non-inplace cousin (chain-rule formulas relying on
    // Phase 4a view-aware ordering for pre-mutation reads of x).
    /// In-place Neg: `x = -x`. Same semantics as `Op::Neg`.
    NegInplace,
    /// In-place Abs: `x = |x|`. Same semantics as `Op::Abs`.
    AbsInplace,
    /// In-place Sqr: `x = x²`. Same semantics as `Op::Sqr`.
    SqrInplace,
    /// In-place Sqrt: `x = √x`. Same semantics as `Op::Sqrt`.
    SqrtInplace,
    /// In-place Rsqrt: `x = 1/√x`. Same semantics as `Op::Rsqrt`.
    RsqrtInplace,
    /// In-place Recip: `x = 1/x`. Same semantics as `Op::Recip`.
    RecipInplace,
    /// In-place Exp: `x = exp(x)`. Same semantics as `Op::Exp`.
    ExpInplace,
    /// In-place Log: `x = ln(x)`. Same semantics as `Op::Log`.
    LogInplace,
    /// In-place Sin: `x = sin(x)`. Same semantics as `Op::Sin`.
    SinInplace,
    /// In-place Cos: `x = cos(x)`. Same semantics as `Op::Cos`.
    CosInplace,
    /// In-place Sign: `x = sign(x)`. Same semantics as `Op::Sign`.
    /// Backward drops gradient (mirrors `Op::Sign`).
    SignInplace,
    /// In-place Floor: `x = ⌊x⌋`. Same semantics as `Op::Floor`.
    /// Backward drops gradient (mirrors `Op::Floor`).
    FloorInplace,
    /// In-place Ceil: `x = ⌈x⌉`. Same semantics as `Op::Ceil`.
    /// Backward drops gradient (mirrors `Op::Ceil`).
    CeilInplace,
    /// In-place Round: `x = round(x)`. Same semantics as `Op::Round`.
    /// Backward drops gradient (mirrors `Op::Round`).
    RoundInplace,
    /// In-place Erf: `x = erf(x)`. Same semantics as `Op::Erf`.
    ErfInplace,
    /// In-place exact-GeLU: `x = 0.5 · x · (1 + erf(x/√2))`. Same
    /// semantics as `Op::GeluErf`.
    GeluErfInplace,
    /// In-place clamp: `x = clamp(x, min, max)`. Same semantics as
    /// `Op::Clamp`. Backward gates the upstream by `(x ≥ min) ∧
    /// (x ≤ max)` (mirrors Op::Clamp).
    ClampInplace { min: f64, max: f64 },
    /// In-place integer-power: `x = x.powi(exp)`. Same semantics as
    /// `Op::PowI`. Backward emits `exp · upstream · x.powi(exp-1)`
    /// (mirrors Op::PowI).
    PowIInplace(i32),

    // --- element-wise comparison (output is U8 mask) ---
    /// Element-wise equality (`a == b`) producing a `U8` mask: `1`
    /// where the inputs are equal, `0` otherwise. Both operands must
    /// share dtype and shape. NaN follows IEEE-754 (`NaN == NaN` is
    /// false). Output dtype is always `U8`, regardless of input dtype
    /// — the binding-table key is `[T, T, U8]`.
    ///
    /// Non-differentiable: backward returns `None` for both inputs
    /// (registered via [`crate::grad::NoGradientBinaryRule`]).
    Equal,
    /// Element-wise inequality (`a != b`) producing a `U8` mask: `1`
    /// where unequal, `0` otherwise. Same shape/dtype contract as
    /// [`Op::Equal`]. NaN follows IEEE-754 (`NaN != NaN` is true →
    /// `1`). Non-differentiable.
    Ne,
    /// Element-wise strictly-less (`a < b`) producing a `U8` mask: `1`
    /// where lhs is strictly less than rhs, `0` otherwise. Same
    /// shape/dtype contract as [`Op::Equal`]. NaN-on-either-side is
    /// always `0` (IEEE-754 unordered comparison). Non-differentiable.
    Lt,
    /// Element-wise less-or-equal (`a <= b`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Op::Equal`]. NaN-on-either-side
    /// is always `0`. Non-differentiable.
    Le,
    /// Element-wise strictly-greater (`a > b`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Op::Equal`]. NaN-on-either-side
    /// is always `0`. Non-differentiable.
    Gt,
    /// Element-wise greater-or-equal (`a >= b`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Op::Equal`]. NaN-on-either-side
    /// is always `0`. Non-differentiable. Final variant of the
    /// comparison family (`Equal`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`).
    Ge,

    // --- rounding family (non-differentiable) ---
    /// Element-wise floor (`⌊x⌋`). Same dtype as input. Backward is
    /// the zero distribution almost everywhere; gradient is dropped
    /// silently (treated like `Op::Step`'s no-grad backward).
    Floor,
    /// Element-wise ceiling (`⌈x⌉`). Same dtype as input. Backward
    /// drops gradient (mirrors [`Op::Floor`]).
    Ceil,
    /// Element-wise round-to-nearest with **banker's rounding**
    /// (round-half-to-even, IEEE 754 roundeven). 0.5 → 0, 1.5 → 2,
    /// 2.5 → 2 (NOT 3), 3.5 → 4. Matches NumPy/PyTorch defaults and
    /// differs from C99 `round()` (half-away-from-zero) at exact ties.
    /// Backward drops gradient (mirrors [`Op::Floor`]).
    Round,
    /// Element-wise sign (`-1` / `0` / `1`). `sign(0) = 0` by
    /// subgradient convention (same convention `Op::Abs`'s backward
    /// uses). Same dtype as input. Backward drops gradient (zero
    /// almost everywhere).
    Sign,
    /// Element-wise Gauss error function (`erf(x)`). Same dtype as
    /// input. Differentiable: `d/dx erf(x) = (2/√π) * exp(-x²)`.
    /// Backward emitted via existing primitives (Sqr → Neg → Exp →
    /// MulScalar(2/√π) → Mul(upstream, .)).
    Erf,
    /// Element-wise GELU activation, **exact erf formulation**:
    /// `0.5 * x * (1 + erf(x/√2))`. Distinct from [`Op::Gelu`]
    /// (tanh approximation, faster but slightly less accurate).
    /// Differentiable; backward decomposes into the standard-normal
    /// CDF + `x * φ(x)` (PDF) chain via primitives.
    GeluErf,
    /// Element-wise binary power: `out = pow(a, b)` with real `b`.
    /// Both inputs share dtype `T` and shape. Distinct from
    /// [`Op::PowI`] (scalar `i32` exponent). Backward:
    ///   d/da pow(a,b) = b * pow(a, b-1)
    ///   d/db pow(a,b) = pow(a,b) * ln(a)
    /// NaN follows IEEE-754 (`pow(-2, 0.5) = NaN`); ln of negative `a`
    /// in the d/db term yields NaN at those positions.
    Pow,
    /// Element-wise reciprocal square root: `out = 1 / sqrt(x)`.
    /// Same dtype as input. Distinct from `Sqrt` followed by `Recip`
    /// (one op instead of two — saves a kernel launch and matches the
    /// RMSNorm shape `x * rsqrt(mean(x²)+eps)`). Backward:
    ///   d/dx (x^(-1/2)) = -0.5 * x^(-3/2) = -0.5 * y³
    /// where `y = 1/sqrt(x)` is the forward output (so the chain
    /// reuses the forward node and never touches `x` directly —
    /// safe near x=0).
    Rsqrt,
    /// Element-wise remainder, **PyTorch convention**:
    /// `out = a - floor(a / b) * b`. The result has the sign of the
    /// divisor (matches `torch.remainder`). Distinct from C99 fmod
    /// (sign of dividend) and `f32::rem_euclid` (always non-negative).
    /// Backward:
    ///   d/da = 1
    ///   d/db = -floor(a/b)
    Rem,
    /// Reverse the order of elements along `dim`. Materializing op
    /// (not a view) — `Layout`'s strides are `usize` so the
    /// negative-stride view path isn't representable. Backward is
    /// `Flip { dim }` itself (involutive).
    Flip { dim: usize },
    /// Cyclic shift along `dim` by `shift` positions (positive
    /// shifts move elements to higher indices, wrapping around).
    /// Materializing op. Backward is `Roll { dim, -shift }`.
    Roll { dim: usize, shift: i64 },
    /// Running cumulative sum along `dim`. Same shape as input.
    /// Backward: reverse cumsum, expressed as `Flip → CumSum → Flip`
    /// on the same dim using existing primitives.
    CumSum { dim: usize },
    /// Upper triangular: keep `x[..., i, j]` when `j >= i + diagonal`,
    /// zero otherwise. Operates on the last two dims; leading dims
    /// are batched. `diagonal` selects which diagonal is the cutoff
    /// (0 = main diagonal, positive = above, negative = below).
    /// Same shape as input. Backward is `Triu { diagonal }` itself
    /// (linear mask; gradient passes through kept positions).
    Triu { diagonal: i64 },
    /// Lower triangular: keep `x[..., i, j]` when `j <= i + diagonal`,
    /// zero otherwise. Operates on the last two dims; leading dims
    /// are batched. `diagonal` selects which diagonal is the cutoff
    /// (0 = main diagonal, positive = above, negative = below). The
    /// canonical causal-attention mask is `tril(diagonal = 0)`.
    /// Same shape as input. Backward is `Tril { diagonal }` itself.
    Tril { diagonal: i64 },
    /// Numerically-stable log-softmax along the last dimension:
    /// `y_i = x_i - max_j(x_j) - log(sum_j exp(x_j - max_j(x_j)))`.
    /// Same shape as input. Used in NLL / cross-entropy loss where
    /// the explicit `log(softmax(x))` decomposition risks
    /// over/underflow. Backward: see [`Op::LogSoftmaxLastDimBackward`].
    LogSoftmaxLastDim,
    /// MaskedFill: `out = where(mask != 0, value, x)`. Inputs:
    /// `(x, mask)` where `mask` is U8 (any nonzero counts as "fill").
    /// `value` is a scalar of the same dtype as `x`, stored on the
    /// op. Output shape == x shape (mask must broadcast to x but the
    /// simple-shape case requires identical shapes today). Backward
    /// passes the upstream gradient through `x` at positions where
    /// `mask == 0`, and contributes zero to `mask` (U8 anyway).
    MaskedFill { value: Scalar },
    /// Backward helper for [`Op::Pad`]. Single input is the upstream
    /// gradient (shape == padded forward output shape); output is the
    /// gradient with respect to the original input (shape ==
    /// `in_shape`). Mode determines accumulation behavior:
    /// - Constant: gradient at padded slots is dropped; the unpadded
    ///   region is the input gradient.
    /// - Reflect / Replicate: gradient accumulates from output
    ///   positions that map back to each input position via the
    ///   forward index function.
    ///
    /// Carries `in_shape` so the kernel can size the output tensor
    /// (Op output shape isn't enough — Op::Pad's output shape
    /// derivation needs reversing).
    PadBackward {
        in_shape: Shape,
        padding: Vec<(usize, usize)>,
        mode: PadMode,
    },

    /// Multi-dim Pad: per-axis `(before, after)` extends each
    /// dimension. Output shape: `out[i] = in[i] + padding[i].0 + padding[i].1`.
    /// `mode` selects the fill behavior; `value` is the fill
    /// constant for [`PadMode::Constant`] (ignored otherwise — see
    /// the variant docs for those modes).
    ///
    /// Multi-dim shape is the form PyTorch's `F.pad` exposes — a
    /// single call covers e.g. "pad an image by 1 px on every side"
    /// (`padding = [(0,0), (0,0), (1,1), (1,1)]` for `[N,C,H,W]`).
    /// Single-dim is just the special case `padding[i] = (0,0)`
    /// everywhere except one axis.
    ///
    /// Only Constant mode is fully implemented in v1. Reflect /
    /// Replicate are accepted at the IR level but the executor
    /// returns a clean "not yet implemented" error.
    ///
    /// Backward (Constant): slice along each padded axis to drop
    /// the padded regions, restoring the input shape.
    Pad { padding: Vec<(usize, usize)>, mode: PadMode, value: f64 },

    // --- ternary select ---
    /// Ternary select: `out[i] = if cond[i] != 0 { a[i] } else { b[i] }`.
    /// Inputs `(cond, a, b)`: `cond` is `U8` (typically the output of
    /// a comparison op), `a` and `b` share dtype `T` and shape with
    /// `cond`. Output dtype = `T`, shape = `cond.shape()`.
    ///
    /// Differentiable through `a` and `b` (gradient flows only to
    /// the picked positions); `cond` is non-differentiable. Backward
    /// rule expressed via `Op::Cast` + scalar arithmetic on existing
    /// primitives, registered in [`crate::grad::WhereRule`].
    Where,

    // --- linear algebra and shape ---
    /// Rank-2 matrix multiply.
    MatMul,
    /// Rank-2 transpose.
    Transpose,
    /// N-dimensional permutation. The parameter is the new axis order:
    /// `out.shape[i] = in.shape[axes[i]]`. For a rank-3 input `[a, b, c]`
    /// with `axes = [2, 0, 1]`, output shape is `[c, a, b]`. The axes
    /// vector must be a permutation of `0..rank`.
    Permute(Vec<usize>),

    // --- dtype, shape, and broadcasting ---
    /// Convert a tensor to a different dtype. The target dtype is carried
    /// on the op itself; the source dtype is whatever the input node's
    /// dtype is.
    Cast(DType),
    /// Broadcast a tensor to a larger shape. The input shape must be
    /// broadcast-compatible with the target (NumPy rules). Used directly
    /// by user code and as the backward rule for `SumAll`, `MeanAll`,
    /// and axis reductions.
    BroadcastTo(Shape),
    /// Reshape a tensor to a new shape with the same element count. Data
    /// is unchanged; only the shape metadata is replaced. Serves as the
    /// general building block for arbitrary shape changes;
    /// see [`Op::Unsqueeze`] for the metadata-only-view variant that
    /// inserts a size-1 dim without ever materializing.
    Reshape(Shape),
    /// Materialize a contiguous copy of the input. Output shape and
    /// dtype match the input; only the layout changes (strides become
    /// row-major, start_offset becomes 0). Zero-cost when the input
    /// is already contiguous + zero-offset — the executor adopts the
    /// input's Storage Arc unchanged in that case.
    ///
    /// First-class IR concern so the optimizer (Phase 2.2) can insert
    /// layout-fixups before kernels that don't advertise
    /// [`crate::KernelCaps::strided_input`] without overloading
    /// [`Op::Reshape`]'s "change shape" semantics. The executor
    /// compiles this to the same `WorkItemKind::ContiguizeOf` arm
    /// `Op::Reshape` uses; the only difference is that
    /// `Op::Contiguize`'s output shape equals its input shape, so
    /// the executor's element-count sanity check is trivial.
    Contiguize,
    /// Insert a size-1 dimension at position `dim` (range `0..=rank`,
    /// where `dim == rank` appends to the end). Strictly more efficient
    /// than [`Op::Reshape`] for the size-1-insertion case: this is a
    /// metadata-only view op that shares bytes with its input via the
    /// Layout side-table, so it preserves any upstream
    /// strided/transposed/broadcast layout instead of triggering an
    /// auto-Contiguize.
    Unsqueeze { dim: usize },
    /// Drop a size-1 dimension at position `dim` (range `0..rank`).
    /// Inverse of [`Op::Unsqueeze`]; metadata-only view that shares
    /// bytes with its input via the Layout side-table. Panics at build
    /// time if the dim's size isn't 1 (mirrors Unsqueeze's bounds
    /// check — graph-builder validation, not runtime).
    Squeeze { dim: usize },
    /// Sum-reduce a tensor to a smaller shape by summing along any dims
    /// where the source was broadcast against the target. This is the
    /// backward rule for `BroadcastTo` and is symmetric with it: both ops
    /// are each other's gradient. For a source shape `[2, 3, 4]` being
    /// reduced to `[3, 4]`, the sum is along dim 0; for `[2, 3, 4]` to
    /// `[2, 1, 4]`, the sum is along dim 1 while keeping the dim.
    ReduceSumTo(Shape),
    /// Max-reduce a tensor to a smaller shape, the maximum-symmetric
    /// counterpart of [`Op::ReduceSumTo`]. The target shape must be
    /// broadcast-compatible into the source: along any axis where the
    /// padded target is 1 (or absent), the input is reduced via max;
    /// other axes carry through. Used by lowering rules (notably
    /// SoftmaxLastDim's max-subtract step) and as the keepdim form of
    /// per-axis max reductions.
    ///
    /// Backward through max-selection routes the upstream gradient
    /// only to the argmax position, which requires holding onto the
    /// original input and recomputing the argmax. PR 3.5 ships this
    /// op without a backward rule (panics cleanly on `.backward()`,
    /// matching `Op::ArgMaxDim`'s precedent); add the backward when a
    /// real consumer needs it.
    ReduceMaxTo(Shape),

    // --- reductions to a scalar ---
    /// Sum of every element, producing a rank-0 tensor.
    SumAll,
    /// Maximum of every element, producing a rank-0 tensor.
    MaxAll,
    /// Minimum of every element, producing a rank-0 tensor.
    MinAll,
    /// Arithmetic mean of every element, producing a rank-0 tensor.
    MeanAll,

    // --- reductions along one dimension ---
    /// Sum along the given dim; the reduced dim is removed from the output.
    SumDim(usize),
    /// Max along the given dim; the reduced dim is removed from the output.
    MaxDim(usize),
    /// Min along the given dim; the reduced dim is removed from the output.
    MinDim(usize),
    /// Mean along the given dim; the reduced dim is removed from the output.
    MeanDim(usize),

    // --- compositions ---
    //
    // Phase 7.6 step 5 (2026-05-11): the per-fused-op primitive variants
    // (SoftmaxLastDim, LayerNormLastDim { eps }, RmsNormLastDim { eps },
    // Rope, Conv2D { … }, FusedLinear) plus the four backward-helper
    // variants (SoftmaxLastDimBackward, LayerNormLastDimBackward,
    // RmsNormLastDimBackward, ReduceMaxToBackward) have been dropped.
    // The closed primitive set no longer carries fused-op concepts;
    // every fused op flows through `Op::Fused(FusedOpId, FusedOpParams)`.
    // See `docs/fused-op-registry.md` for the migration record.

    // Phase 7.6 step 5 (continued, 2026-05-11): `QMatMul {
    // quant_type, k, n }` dropped — flows through
    // `Op::Fused(QMATMUL, FusedOpParams::QMatMul { … })`.

    // --- backward helpers ---
    //
    // Phase 7.6 step 5 (2026-05-11): the four migrated backward-helper
    // variants (SoftmaxLastDimBackward, LayerNormLastDimBackward,
    // RmsNormLastDimBackward, ReduceMaxToBackward) have been dropped.
    // They flow through `Op::Fused(FusedOpId, FusedOpParams)` per the
    // registry split. LogSoftmaxLastDimBackward is the only remaining
    // backward-helper primitive variant — its forward primitive
    // (`Op::LogSoftmaxLastDim`, not in scope for step 5) hasn't migrated
    // to the registry yet, so the backward stays primitive too.
    //
    /// LogSoftmax-last-dim backward. Inputs: `(forward_log_softmax_output,
    /// upstream)`. Output: the gradient of the input to log-softmax.
    /// Formula: `g - exp(y) * sum(g, last_dim, keepdim=true)` where
    /// `y` is the forward log-softmax output and `g` is the upstream
    /// gradient. Independent of `Op::SoftmaxLastDimBackward` because
    /// the closed-form rule uses `exp(y) = softmax(x)` directly, not
    /// the upstream-times-softmax composition.
    LogSoftmaxLastDimBackward,

    // --- integer-producing reductions ---
    //
    // These reduce a float tensor along one dim and emit a U32 tensor of
    // indices. Non-differentiable — they exist for classification
    // prediction and similar discrete workloads, not as part of any
    // training graph. Backward panics cleanly if a user builds these
    // inside a graph they then differentiate.
    /// Index of the maximum along `dim`. Output: U32 with the reduced
    /// dim removed.
    ArgMaxDim(usize),
    /// Index of the minimum along `dim`. Output: U32 with the reduced
    /// dim removed.
    ArgMinDim(usize),

    // --- shape manipulation ---
    /// Concatenate two tensors along `dim`. Both must have the same rank,
    /// same dtype, and equal sizes in every dim except `dim`.
    Concat { dim: usize },
    /// Slice (narrow) a tensor along `dim`: take elements `[start, start+len)`.
    /// Output shape: same as input but with `dim` shrunk to `len`.
    Slice { dim: usize, start: usize, len: usize },

    // --- scalar-by-tensor ops ---
    //
    // These exist to avoid building a full-shape const just to add or
    // multiply by a constant. Stored as f64 on the op and converted to
    // the target dtype at realize time.
    /// Add a scalar to every element.
    AddScalar(f64),
    /// Multiply every element by a scalar.
    MulScalar(f64),
    /// Raise every element to an integer power via repeated multiplication.
    /// Works without an extra `log`/`exp` pair so it's numerically clean
    /// for small integer exponents like squaring or cubing.
    PowI(i32),
    /// Clamp every element to the inclusive range `[min, max]`.
    Clamp { min: f64, max: f64 },

    // --- element-wise max/min between two tensors ---
    /// Element-wise maximum of two tensors with matching shape.
    Maximum,
    /// Element-wise minimum of two tensors with matching shape.
    Minimum,

    // --- indexing ops ---
    //
    // These take a "data" tensor (floats) as their first input and an
    // "index" tensor (integer) as their second input. The executor
    // reads both operands from the cache, dispatches on the data
    // tensor's dtype for the output, and uses the index tensor as-is.
    /// Index-select along a single dimension with a 1-D index vector.
    /// Output shape: same as input, but with `dim` replaced by
    /// `indices.elem_count()`.
    IndexSelect { dim: usize },
    /// N-dimensional gather along `dim`. Output shape matches the index
    /// tensor's shape. `out[..., k, ...] = data[..., indices[..., k, ...], ...]`
    /// with `dim` being the position where the index value substitutes
    /// for the output coordinate.
    Gather { dim: usize },
    /// Index-add: the backward of `IndexSelect`. Takes a base tensor,
    /// a 1-D `U32` index vector, and a src tensor, and adds src to base
    /// at the indexed positions along `dim`. `out[..., indices[i], ...] +=
    /// src[..., i, ...]`. Non-indexed positions of base are copied
    /// through unchanged. Works functionally (returns a new tensor).
    IndexAdd { dim: usize },
    /// Scatter-add: the backward of `Gather`. Takes a base tensor, an
    /// N-D `U32` index tensor matching the src shape, and a src tensor,
    /// and accumulates src values into base at positions given by
    /// substituting `indices` at position `dim`. `out[p with dim ←
    /// indices[p]] += src[p]`.
    ScatterAdd { dim: usize },

    // --- 2-D convolution + fused linear ---
    //
    // Phase 7.6 step 5 (2026-05-11): `Conv2D { stride, padding, groups }`
    // and `FusedLinear` have been dropped — both flow through
    // `Op::Fused(FusedOpId, FusedOpParams)` per the registry split.
    // See `fuel-graph/src/registry/{conv2d,fused_linear}.rs` for the
    // current entry shapes.

    // Phase 7.6 step 5 (continued, 2026-05-11): `FlashAttn`,
    // `PagedAttn`, and `ConvTranspose2D` dropped — all three flow
    // through `Op::Fused(<id>, FusedOpParams::<variant>)` per the
    // registry split. See `fuel-graph/src/registry/{flash_attn,
    // paged_attn, conv_transpose_2d}.rs`.

    // --- cross-device transfer ---
    /// Copy the input tensor to a specific device. Source stays
    /// resident (non-destructive). Source device is implicit from
    /// `inputs[0]`'s current residency. When source and target
    /// devices match, the backend is free to turn this into a cheap
    /// clone; when they differ, the backend must actually transfer
    /// the data (today: through a host buffer — Phase 3 has no P2P
    /// fast path yet).
    ///
    /// Phase 3 of the unified scheduler work. The scheduler (Phase 4)
    /// inserts these nodes automatically; users can also construct
    /// them explicitly via `Tensor::copy_to_device`.
    Copy { target: DeviceLocation },

    /// Release the input tensor's device-resident storage. Produces a
    /// zero-element marker output (NodeId placeholder for graph
    /// bookkeeping; no meaningful data). After this op runs, the
    /// input's device storage is gone; any subsequent reader of the
    /// input's NodeId must get its data from a different path (a
    /// sibling Copy output, or re-execution of the producer).
    ///
    /// Destructive on `inputs[0]`. Scheduler must pin `Op::Release`
    /// to run after every other reader of its input via
    /// [`opt::derive_ordering`].
    ///
    /// **Multi-output bundles** (Option C, Session 3): when
    /// `inputs[0]` is a multi-output producer (its Storage carries a
    /// [`fuel_ir::storage::OutputView`] bundle), the bundle
    /// is the single eviction unit — the Release evicts the whole
    /// bundle, not a single slot. [`opt::collect_alias_set`] treats
    /// every `Op::View` of the producer as part of the producer's
    /// alias set, so `derive_ordering` pins Release after every
    /// reader of every View; the bundle drops only when the LAST
    /// View consumer finishes. Op::ViewOwned consumers may run
    /// freely after Release: their forward memcpy ran before the
    /// Release fires (data-dep edge), and the resulting standalone
    /// Storage is independent of the producer's bundle. Per-slot
    /// eviction (releasing one slot's bytes while another stays
    /// live) is intentionally a follow-up — v1 keeps the bundle as
    /// a single unit.
    Release,

    /// Transfer the input tensor to `target` device, destroying the
    /// source in the process. Semantically equivalent to
    /// `Op::Copy { target }` followed by `Op::Release` on the source,
    /// with the explicit guarantee that backends MAY fast-path the
    /// fused form as a single transfer (e.g., a zero-copy handoff if
    /// source and target are the same memory type).
    ///
    /// Output is a fresh tensor on `target` with the input's shape
    /// and dtype. Destructive on `inputs[0]`; the scheduler pins this
    /// op to run after every non-destructive reader of the source via
    /// [`opt::derive_ordering`].
    Move { target: DeviceLocation },

    /// In-place scatter write: copies `inputs[1]`'s bytes into a
    /// rectangular slab of `inputs[0]`'s storage, defined per-dim by
    /// `ranges[i] = (start, end)`. After the op runs, `inputs[0]`'s
    /// bytes inside the slab are replaced; bytes outside the slab are
    /// untouched.
    ///
    /// Shape contract: `inputs[0].rank() == inputs[1].rank() ==
    /// ranges.len()`. For each axis `i`,
    /// `inputs[1].dims()[i] == ranges[i].1 - ranges[i].0`, and
    /// `ranges[i].1 <= inputs[0].dims()[i]`. Dtypes must match.
    ///
    /// Output is a marker that adopts `inputs[0]`'s Storage Arc and
    /// Layout — same shape, same bytes, post-write. Consumers that
    /// want a sub-extent of the destination compose an explicit
    /// `Op::Slice` after the write (WriteSlice does not encode
    /// post-write extents).
    ///
    /// Destructive on `inputs[0]` (the destination is consumed; only
    /// the write op's output NodeId may read its bytes afterward).
    /// Non-destructive on `inputs[1]`. The scheduler pins this op to
    /// run after every other reader of the destination via
    /// [`opt::derive_ordering`].
    ///
    /// Phase E.3.2: introduced to back persistent KV-cache writes
    /// (`InferenceContext` + `KvCache`). Non-differentiable; backward
    /// panics — KV-cache writes are forward-only.
    ///
    /// `dyn_offset` (Phase D symbolic extents): `None` ⇒ fully static
    /// (`ranges` alone defines the slab — today's behavior). `Some((axis,
    /// off))` ⇒ the destination start on `axis` is `off.resolve(env)`
    /// against the per-pass [`fuel_ir::SymEnv`] at realize,
    /// overriding `ranges[axis].0`; the slab width on that axis stays
    /// `ranges[axis].1 - ranges[axis].0`. This is the input-determined
    /// dynamic-offset path (a [`DynScalar`] over the one `SymEnv`),
    /// distinct from [`Op::WriteSliceRotating`]'s data-determined
    /// position (a tensor input). It backs the persistent decode
    /// KV-cache write, whose append offset (`cached_len`) is a per-token
    /// runtime value over a fixed-capacity buffer, so the graph
    /// structure stays identical across tokens and the plan is reused.
    /// Build a node with this set via [`Tensor::write_slice_dyn`]; the
    /// resolved `offset + width` is bounds-checked against the
    /// destination capacity at realize (a typed error, never a panic).
    WriteSlice {
        ranges: Vec<(usize, usize)>,
        dyn_offset: Option<(usize, DynScalar)>,
    },

    /// Like [`Op::WriteSlice`] but the `axis` axis wraps modulo
    /// `modulus`. The dynamic write position comes through `inputs[2]`
    /// (rank-0 U32); at realize time the kernel computes
    /// `start = position % modulus` and writes `inputs[1]` into
    /// `inputs[0]` at the wrapped offset, splitting across the
    /// boundary if `start + write_len > modulus`.
    ///
    /// Inputs:
    /// - `inputs[0]`: destination tensor (the rotating buffer). Shape
    ///   on `axis` is the full storage extent — typically equal to
    ///   `modulus`.
    /// - `inputs[1]`: source slice. Shape on non-rotating axes is the
    ///   slab in `ranges`; shape on `axis` is the write length.
    /// - `inputs[2]`: dynamic write position, rank-0 `U32`. Wrapped
    ///   modulo `modulus` inside the kernel — callers pass the
    ///   monotonic logical position (token index, etc.).
    ///
    /// `ranges` describes the destination slab on every axis. For the
    /// rotating axis `axis`, `ranges[axis].0` is ignored (start is
    /// dynamic) and `ranges[axis].1 - ranges[axis].0` must equal
    /// `inputs[1].dims()[axis]` (the write length). For other axes
    /// the same shape contract as `Op::WriteSlice` applies.
    ///
    /// Output is a marker that adopts `inputs[0]`'s Storage Arc and
    /// Layout, post-write. Destructive on `inputs[0]`; scheduled like
    /// `Op::WriteSlice`.
    ///
    /// Phase C of the eager-Tensor retirement program. Backs
    /// sliding-window KV caches (Mistral / Phi-3 sliding-window /
    /// sliding-window Qwen). Non-differentiable; backward panics.
    WriteSliceRotating {
        axis: usize,
        modulus: usize,
        ranges: Vec<(usize, usize)>,
    },

    /// Allocate a fresh, zero-initialized Storage of the node's shape +
    /// dtype on `target`. Zero inputs — `Op::Alloc` is a *source* op
    /// like [`Op::Const`], but its bytes are computed (zeros) rather
    /// than seeded from the host.
    ///
    /// Bridge-retirement Phase 3a (post-9c). Replaces the per-device-
    /// location match in `fuel-core::inference_context::alloc_zeroed_on`
    /// with a graph-level node the optimizer can see + the executor's
    /// `WorkItemKind::Alloc` arm dispatches per-backend.
    ///
    /// **Device-handle requirement**: the executor's Alloc arm derives
    /// the per-backend handle (e.g. `&CudaDevice`, `&Arc<VulkanBackend>`)
    /// by searching the input cache for any Storage already on
    /// `target`'s backend. Callers must seed the cache with at least
    /// one such storage before realizing an Op::Alloc on a non-CPU
    /// target. `fuel-core::pipelined_bridge::device_seed_storage` is
    /// the canonical seeder.
    Alloc { target: DeviceLocation },

    /// Fill the input tensor's bytes with zero, in place. Aliases
    /// `inputs[0]`'s Storage Arc as the output (same Storage; the
    /// bytes are mutated). Destructive on `inputs[0]` — readers of
    /// the input's NodeId after this op see post-fill bytes; new
    /// readers should use this op's NodeId.
    ///
    /// Bridge-retirement Phase 3a follow-up. Paired with
    /// [`Op::Alloc`] to give the architecturally clean "alloc uninit
    /// + explicit zero" pipeline. Backends with native device-side
    /// fills (CUDA `cuMemsetD8Async`, Vulkan `vkCmdFillBuffer`)
    /// dispatch through their kernels; CPU does an
    /// `Arc::make_mut`-style in-place memset.
    ///
    /// Future extension: an `Op::Fill { value: u8 }` (or `Scalar`)
    /// for non-zero patterns. ZeroFill is the only variant today
    /// because zero is the only init pattern any current caller
    /// needs.
    ZeroFill,

    /// Phase 7.6 single-arm delegate to the open
    /// [`crate::registry::FusedOpRegistry`]. The id selects which
    /// fused op (SoftmaxLastDim, RmsNormLastDim, FlashAttn, ...) and
    /// the params carry per-instance data. The registry's metadata
    /// side (decompose, backward identity, shape/dtype rules) lives in
    /// `fuel-graph::registry`; the kernel side (per-backend `KernelRef`
    /// + cost + precision) lives in `fuel-storage::fused`.
    ///
    /// Adding a new fused op is one registry entry plus one kernel
    /// function per backend that supports it — no new `Op` variant.
    /// Step 3 of Phase 7.6 migrates SoftmaxLastDim through this arm
    /// as the proof of concept.
    Fused(FusedOpId, FusedOpParams),

    /// Project one logical output out of a multi-output producer node.
    /// The producer's realized [`Storage`] carries a
    /// [`fuel_ir::storage::OutputView`] side-table; this op
    /// reads slot `slot` of that side-table and exposes its
    /// dtype/shape/layout as the View node's output.
    ///
    /// Single input (the multi-output producer). Output shares the
    /// producer's `Arc<RwLock<Storage>>` — zero bytes copied. The
    /// bundle stays alive as long as any View consumer (or the
    /// original producer handle) holds a clone.
    ///
    /// At graph-build time, the View's shape + dtype + layout come
    /// from the producer's slot specs recorded via
    /// [`Graph::set_output_views`]; building a `View { slot }` over a
    /// producer that hasn't declared slot specs is a typed error
    /// surfaced by the View-emitting builder helper. The bundle's
    /// per-slot dtype is independent of the producer node's own
    /// `Node::dtype` (which is the bundle's primary/slot-0 dtype).
    ///
    /// Backward (see `fuel-graph::grad`): the per-slot upstream
    /// gradient is scattered into a zero-filled bundle of the
    /// producer's overall byte shape; the producer's own backward
    /// rule then sees a "bundle gradient" as its upstream. Until a
    /// real multi-output op author exists (Session 2), the backward
    /// rule emits an [`Op::Const`] zero of the producer's primary
    /// shape — the scatter-into-slot primitive lands alongside the
    /// first differentiable multi-output op.
    View { slot: u32 },

    /// Like [`Op::View`], but copies the slot's bytes into a fresh
    /// standalone Storage. Costs one allocation + one slot-sized
    /// memcpy; the producer's bundle Arc can drop as soon as every
    /// outstanding `Op::View` over the same producer also drops.
    ///
    /// The planner promotes a `View` to `ViewOwned` when a slot's
    /// liveness substantially outlasts the rest of the bundle (the
    /// classic case: Mamba's `last_state` slot retained across an
    /// autoregressive step while `y` is consumed immediately). v1
    /// emits the same dispatch shape as `Op::Copy` — backend-side it's
    /// `Copy` with byte offset + length pulled from the slot spec.
    ///
    /// Same shape/dtype/layout inference path as `View`; same backward
    /// shape too (the forward already paid the copy).
    ViewOwned { slot: u32 },

    /// Multi-output autograd primitive (Option C, item 4). Takes two
    /// inputs:
    ///   - `inputs[0]`: a bundled tensor (typically a zero-bundle)
    ///   - `inputs[1]`: the slot's gradient
    ///
    /// Produces a new bundled tensor identical to `inputs[0]` except
    /// that slot `slot`'s byte range is overlaid by `inputs[1]`'s
    /// bytes. The output is alias-free with respect to `inputs[0]`
    /// (a fresh Storage with the new bytes) so multiple
    /// `ScatterIntoSlot` operations on the same `bundle_zero` chain
    /// sequentially without aliasing pitfalls.
    ///
    /// Emitted by `Op::View` / `Op::ViewOwned` backward rules: each
    /// View consumer scatters its upstream gradient into a fresh
    /// zero-bundle, then the producer's backward receives the
    /// composed bundle gradient as its upstream.
    ///
    /// **Status**: IR-level primitive only in this session. No CPU
    /// kernel registration yet — the production multi-output ops
    /// (SelectiveScan, SsdChunkScan) are `BackwardKind::NotDifferentiable`,
    /// so the autograd never reaches a `ScatterIntoSlot` realization.
    /// When the first differentiable multi-output op materializes (the
    /// Mamba training session), it lights up the kernel side along
    /// with its own backward.
    ScatterIntoSlot { slot: u32 },

    /// In-place multi-path (phi/merge) node — the arena representation
    /// of the optimized form's divergent-then-reconvergent routes.
    ///
    /// A `Branch` node's divergent "arms" are encoded in the node's
    /// existing `inputs: Vec<NodeId>`: each input is the *exit* of one
    /// candidate route, and `reconverge_at` names the explicit node at
    /// which those routes merge back into a single value. Because the
    /// multi-path structure is an arena fact rather than an overlay, it
    /// persists, compacts, graph-walks, and path-filters exactly like
    /// any other node; a graph with **zero** `Op::Branch` nodes is
    /// exactly today's single-route graph (back-compatible by
    /// construction).
    ///
    /// **Inert until PR-A1.** This variant is the PR-A0 scaffold: the
    /// closed `Op` enum carries it and every exhaustive match handles
    /// it, but nothing constructs it yet. The `open_branch` /
    /// `add_arm` / `finalize_branches` builders (with build-time
    /// validation: reconverge must descend from the diverge point, arms
    /// internally disjoint, cast-to-uniform at reconverge) land in
    /// PR-A1. Until then, accessors return a constant ("branch") and
    /// fallible paths (shape inference, autograd, lowering) return a
    /// descriptive `Err` rather than panicking.
    Branch { reconverge_at: NodeId },
}

impl Op {
    /// Index into `inputs` that this op destroys on execution. `None`
    /// means the op is non-destructive — every input remains readable
    /// after the op completes. Destructive ops need the scheduler to
    /// pin them to run after all other readers of the destroyed input,
    /// via ordering edges derived by [`opt::derive_ordering`].
    pub fn destructive_input(&self) -> Option<usize> {
        match self {
            Op::Release | Op::Move { .. } | Op::WriteSlice { .. } | Op::WriteSliceRotating { .. } | Op::ZeroFill => Some(0),
            // In-place unary ops mutate input 0.
            Op::ReluInplace
            | Op::SiluInplace
            | Op::GeluInplace
            | Op::TanhInplace
            | Op::SigmoidInplace
            | Op::NegInplace
            | Op::AbsInplace
            | Op::SqrInplace
            | Op::SqrtInplace
            | Op::RsqrtInplace
            | Op::RecipInplace
            | Op::ExpInplace
            | Op::LogInplace
            | Op::SinInplace
            | Op::CosInplace
            | Op::SignInplace
            | Op::FloorInplace
            | Op::CeilInplace
            | Op::RoundInplace
            | Op::ErfInplace
            | Op::GeluErfInplace => Some(0),
            Op::ClampInplace { .. } | Op::PowIInplace(_) => Some(0),
            Op::Fused(id, _) if *id == crate::registry::FusedOps::INPLACE_AFFINE => Some(0),
            _ => None,
        }
    }

    /// Short human-readable name for this op — used by executor error
    /// messages to identify the offending graph node without spilling
    /// all of `Debug`'s field contents. Keeps panic messages one-liner-
    /// friendly while still telling you which kind of op blew up.
    pub fn short_name(&self) -> &'static str {
        op_short_name(self)
    }

    /// Whether this op is a metadata-only view: its output shares bytes
    /// with its single input, reinterpreting them through new strides
    /// and/or offset. Executors realize these without allocating new
    /// storage — they wrap the input's Storage in a fresh Layout.
    ///
    /// Note that `Op::Reshape` is *not* a view op. Reshape produces
    /// contiguous output: zero-copy when its input is contiguous, but
    /// it materializes via auto-Contiguize when the input is strided.
    /// The other shape-altering ops in this set are always view-shaped
    /// regardless of the input layout.
    pub fn is_view_op(&self) -> bool {
        matches!(
            self,
            Op::Transpose
                | Op::Permute(_)
                | Op::BroadcastTo(_)
                | Op::Slice { .. }
                | Op::Unsqueeze { .. }
                | Op::Squeeze { .. }
                | Op::Flip { .. }
        )
    }
}

/// Compute the output [`Layout`] of a metadata-only view op from its
/// input layout + the op variant. Returns `Err` if the op variant
/// isn't a view op (caller's contract — typically guarded by
/// [`Op::is_view_op`]).
///
/// Used by [`Graph::push`] to auto-populate the layout side-table for
/// view-op nodes at construction time, and by lowering rules in
/// [`opt`] that emit view-op nodes and need to populate the side-table
/// for them. The compiler reads the resulting layouts via
/// [`Graph::layout`] without re-deriving.
pub fn derive_view_output_layout(
    op: &Op,
    input_layout: &Layout,
) -> Result<Layout, fuel_ir::Error> {
    match op {
        Op::Transpose => {
            let rank = input_layout.shape().rank();
            if rank < 2 {
                return Err(fuel_ir::Error::Msg(format!(
                    "Op::Transpose requires rank >= 2, input rank is {rank}",
                ))
                .bt());
            }
            input_layout.transpose(rank - 2, rank - 1)
        }
        Op::Permute(axes) => input_layout.permute(axes),
        Op::BroadcastTo(target_shape) => input_layout.broadcast_as(target_shape.clone()),
        Op::Slice { dim, start, len } => input_layout.narrow(*dim, *start, *len),
        Op::Unsqueeze { dim } => input_layout.unsqueeze(*dim),
        Op::Squeeze { dim } => input_layout.squeeze(*dim),
        Op::Flip { dim } => input_layout.flip(*dim),
        other => Err(fuel_ir::Error::Msg(format!(
            "derive_view_output_layout called with non-view op {other:?}",
        ))
        .bt()),
    }
}

fn op_short_name(op: &Op) -> &'static str {
    match op {
        Op::Const             => "Const",
        Op::Add                  => "Add",
        Op::Sub                  => "Sub",
        Op::Mul                  => "Mul",
        Op::Div                  => "Div",
        Op::Neg                  => "Neg",
        Op::Sqr                  => "Sqr",
        Op::Sqrt                 => "Sqrt",
        Op::Exp                  => "Exp",
        Op::Log                  => "Log",
        Op::Sin                  => "Sin",
        Op::Cos                  => "Cos",
        Op::Tanh                 => "Tanh",
        Op::Sigmoid              => "Sigmoid",
        Op::Silu                 => "Silu",
        Op::Gelu                 => "Gelu",
        Op::Relu                 => "Relu",
        Op::Step                 => "Step",
        Op::Recip                => "Recip",
        Op::Abs                  => "Abs",
        Op::ReluInplace          => "ReluInplace",
        Op::SiluInplace          => "SiluInplace",
        Op::GeluInplace          => "GeluInplace",
        Op::TanhInplace          => "TanhInplace",
        Op::SigmoidInplace       => "SigmoidInplace",
        Op::NegInplace           => "NegInplace",
        Op::AbsInplace           => "AbsInplace",
        Op::SqrInplace           => "SqrInplace",
        Op::SqrtInplace          => "SqrtInplace",
        Op::RsqrtInplace         => "RsqrtInplace",
        Op::RecipInplace         => "RecipInplace",
        Op::ExpInplace           => "ExpInplace",
        Op::LogInplace           => "LogInplace",
        Op::SinInplace           => "SinInplace",
        Op::CosInplace           => "CosInplace",
        Op::SignInplace          => "SignInplace",
        Op::FloorInplace         => "FloorInplace",
        Op::CeilInplace          => "CeilInplace",
        Op::RoundInplace         => "RoundInplace",
        Op::ErfInplace           => "ErfInplace",
        Op::GeluErfInplace       => "GeluErfInplace",
        Op::ClampInplace{..}     => "ClampInplace",
        Op::PowIInplace(_)       => "PowIInplace",
        Op::Equal                => "Equal",
        Op::Ne                   => "Ne",
        Op::Lt                   => "Lt",
        Op::Le                   => "Le",
        Op::Gt                   => "Gt",
        Op::Ge                   => "Ge",
        Op::Where                => "Where",
        Op::Floor                => "Floor",
        Op::Ceil                 => "Ceil",
        Op::Round                => "Round",
        Op::Sign                 => "Sign",
        Op::Erf                  => "Erf",
        Op::GeluErf              => "GeluErf",
        Op::Pow                  => "Pow",
        Op::Rsqrt                => "Rsqrt",
        Op::Rem                  => "Rem",
        Op::Flip{..}             => "Flip",
        Op::Roll{..}             => "Roll",
        Op::CumSum{..}           => "CumSum",
        Op::Triu{..}             => "Triu",
        Op::Tril{..}             => "Tril",
        Op::LogSoftmaxLastDim    => "LogSoftmaxLastDim",
        Op::MaskedFill{..}       => "MaskedFill",
        Op::Pad{..}              => "Pad",
        Op::PadBackward{..}      => "PadBackward",
        Op::MatMul               => "MatMul",
        Op::Transpose            => "Transpose",
        Op::Permute(_)           => "Permute",
        Op::Cast(_)              => "Cast",
        Op::BroadcastTo(_)       => "BroadcastTo",
        Op::Reshape(_)           => "Reshape",
        Op::Contiguize           => "Contiguize",
        Op::Unsqueeze{..}        => "Unsqueeze",
        Op::Squeeze{..}          => "Squeeze",
        Op::ReduceSumTo(_)       => "ReduceSumTo",
        Op::ReduceMaxTo(_)       => "ReduceMaxTo",
        Op::SumAll               => "SumAll",
        Op::MaxAll               => "MaxAll",
        Op::MinAll               => "MinAll",
        Op::MeanAll              => "MeanAll",
        Op::SumDim(_)            => "SumDim",
        Op::MaxDim(_)            => "MaxDim",
        Op::MinDim(_)            => "MinDim",
        Op::MeanDim(_)           => "MeanDim",
        Op::LogSoftmaxLastDimBackward
                                 => "LogSoftmaxLastDimBackward",
        Op::ArgMaxDim(_)         => "ArgMaxDim",
        Op::ArgMinDim(_)         => "ArgMinDim",
        Op::Concat{..}           => "Concat",
        Op::Slice{..}            => "Slice",
        Op::AddScalar(_)         => "AddScalar",
        Op::MulScalar(_)         => "MulScalar",
        Op::PowI(_)              => "PowI",
        Op::Clamp{..}            => "Clamp",
        Op::Maximum              => "Maximum",
        Op::Minimum              => "Minimum",
        Op::IndexSelect{..}      => "IndexSelect",
        Op::Gather{..}           => "Gather",
        Op::IndexAdd{..}         => "IndexAdd",
        Op::ScatterAdd{..}       => "ScatterAdd",
        Op::Copy{..}             => "Copy",
        Op::Release              => "Release",
        Op::Move{..}             => "Move",
        Op::WriteSlice{..}       => "WriteSlice",
        Op::WriteSliceRotating{..} => "WriteSliceRotating",
        Op::Alloc{..}            => "Alloc",
        Op::ZeroFill             => "ZeroFill",
        // Phase 7.6: registry-extended fused ops. Step 3 wires per-id
        // names through a static lookup; until then, all fused ops
        // share one short name. Distinguishing in error messages is
        // future work — `id` is in the Debug repr.
        Op::Fused(_, _)          => "Fused",
        Op::View{..}             => "View",
        Op::ViewOwned{..}        => "ViewOwned",
        Op::ScatterIntoSlot{..}  => "ScatterIntoSlot",
        // PR-A0 inert scaffold (the multi-path phi/merge node).
        Op::Branch{..}           => "Branch",
    }
}

/// G2 helper: element count of a `HostBuffer`. Used by Tensor's
/// constructors to validate that the supplied data matches the
/// declared shape's `elem_count` before allocating Storage.
fn host_buffer_elem_count(buf: &fuel_ir::HostBuffer) -> usize {
    use fuel_ir::HostBuffer;
    match buf {
        HostBuffer::F32(v) => v.len(),
        HostBuffer::F64(v) => v.len(),
        HostBuffer::BF16(v) => v.len(),
        HostBuffer::F16(v) => v.len(),
        HostBuffer::U8(v) => v.len(),
        HostBuffer::U32(v) => v.len(),
        HostBuffer::I64(v) => v.len(),
        other => panic!(
            "Tensor::from_*: unsupported host-buffer dtype {:?}",
            other.dtype(),
        ),
    }
}

/// A single node in the graph: an operation and the IDs of its inputs.
/// The node also caches its output shape and dtype so downstream builders
/// can validate without walking back through the graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub op:     Op,
    pub inputs: Vec<NodeId>,
    pub shape:  Shape,
    pub dtype:  DType,
}

/// How a node's storage is shared and how long it lives — its **storage
/// class** ([03-ir §"Storage classes and sessions"]). A node is a
/// structural fact and does not need storage to exist; when storage *is*
/// attached, its class governs sharing and lifetime so one optimized graph
/// can serve many concurrent sessions (weights shared, session state
/// `SessionId`-keyed, activations per-realize scratch).
///
/// The class is **inferred from the op** ([`infer_storage_class`]) with an
/// **explicit override** recorded in the graph's `storage_class` side-table
/// — e.g. KV-cache placeholder `Op::Const`s, which are session state, not
/// shared weights.
///
/// Phase D substrate (PR-D1): the classification is recorded but not yet
/// consumed by realize. Session keying (D2) and the persistent decode graph
/// (D3–D5) build on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageClass {
    /// Weights / constants: one storage for the whole model, identical
    /// across all sessions; keyed by `NodeId`. The op-inferred default for
    /// `Op::Const`.
    Shared,
    /// Per-session durable buffers (KV-caches + explicit cache-write
    /// targets): one storage per session, keyed by `(NodeId, SessionId)`.
    /// The op-inferred default for the in-place cache-write ops
    /// (`Op::WriteSlice` / `Op::WriteSliceRotating`); KV-cache placeholder
    /// `Op::Const`s take it via an explicit override.
    SessionState,
    /// Activations / scratch: ephemeral, allocated per realize and freed
    /// after use; never persisted. The op-inferred default for everything
    /// else. A transient value may cross devices mid-realize (a D2D
    /// `Op::Copy`) but never crosses to disk.
    Transient,
}

/// The op-inferred storage class for a node, **before** any explicit
/// override (see [`StorageClass`] and [`Graph::storage_class`]):
/// - `Op::Const` → [`StorageClass::Shared`] (weights);
/// - `Op::WriteSlice` / `Op::WriteSliceRotating` → [`StorageClass::SessionState`]
///   (they write into a per-session durable buffer);
/// - everything else → [`StorageClass::Transient`].
pub fn infer_storage_class(op: &Op) -> StorageClass {
    match op {
        Op::Const => StorageClass::Shared,
        Op::WriteSlice { .. } | Op::WriteSliceRotating { .. } => StorageClass::SessionState,
        _ => StorageClass::Transient,
    }
}

/// The graph arena. Stores every node added during a computation-building
/// session. Nodes are append-only and indexed by [`NodeId`].
///
/// Placement metadata (`DeviceLocation` per node) lives in a side-table
/// so existing `Node { ... }` construction sites don't need to be
/// modified. This is the Phase-1 shape: placements are opt-in, inert
/// hints that the executor may validate but does not yet act on.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    /// Sparse map of per-node placement hints. Entries are only present
    /// when explicitly set via [`Graph::set_placement`]. Absent entries
    /// mean "inherit from the executor's default device."
    placements: HashMap<NodeId, DeviceLocation>,
    /// NodeIds the executor must schedule even when their outputs are
    /// unreachable from the user's roots. Populated by graph-mutating
    /// rules that emit side-effecting ops (e.g. `Op::Release` for
    /// residency eviction). See [`Graph::add_side_effect_root`].
    side_effect_roots: Vec<NodeId>,
    /// Phase 7.5 work item G: graph-owned realized-storage map.
    /// Slots are populated for `Op::Const` leaves at construction
    /// time and for non-leaf nodes at realize time. Values are
    /// `Arc<RwLock<Storage>>` so consumers (Tensor handles, executor
    /// caches, residency machinery) share Storage cheaply via Arc
    /// clones. Lifetime is tied to the Graph; when the graph drops,
    /// any slots not held by external Arc clones are freed.
    storage_map: HashMap<NodeId, Arc<RwLock<Storage>>>,
    /// Phase 7.5 storage-unification B2: per-node dispatch
    /// resolution result. Set by the dispatch resolver (Phase B3+)
    /// when DAG construction picks which backend will execute the
    /// node's op given its inputs' dtype + residency. Sparse like
    /// `placements`: absent entries mean "not yet resolved" and the
    /// executor falls back to per-op-eval dispatch (today's
    /// behavior). After full migration, every Node has an entry.
    ///
    /// Note: this is a *side-table*, not a `Node` field, so that
    /// existing `Node { ... }` constructors don't need to change
    /// during the migration. Mirrors the `placements` pattern.
    target_backends: HashMap<NodeId, BackendId>,
    /// Phase 7.5 storage-unification — Layout-on-Node side-table.
    /// Sparse: every Node has a logical [`Layout`], but for the
    /// common case (contiguous result of a kernel) we don't store
    /// it explicitly — [`Graph::layout`] returns
    /// `Layout::contiguous(node.shape)` as the fallback. Entries
    /// are written here exclusively by metadata-only view ops
    /// (`Op::Transpose`, `Op::Permute`, `Op::BroadcastTo`,
    /// 0-copy `Op::Reshape`) where the output is a strided view
    /// over an upstream node's Storage.
    ///
    /// Coherence rule: if `layouts[id]` is present, its
    /// `shape()` must equal `nodes[id].shape` — Layout adds
    /// strides + offset; the visible shape is identical.
    layouts: HashMap<NodeId, Layout>,
    /// Per-node multi-output side-table. Populated only for nodes
    /// whose op produces a bundled [`Storage`] with N>1 logical
    /// outputs (Session 1 of the multi-output-nodes design — see
    /// [`OutputView`](fuel_ir::storage::OutputView)). `Op::View`
    /// and `Op::ViewOwned` builders read this side-table to derive
    /// their own output shape/dtype/layout from the producer's
    /// declared slot specs.
    ///
    /// Coherence rule: when present, slot 0's `dtype` must equal the
    /// producer node's `Node::dtype` (which the multi-output op's
    /// allocator also uses as the inner backend storage's dtype).
    /// The producer's `Node::shape` reflects slot 0's logical shape
    /// for back-compat with single-output infrastructure; non-primary
    /// slots may have unrelated shapes and dtypes.
    node_output_views: HashMap<NodeId, Arc<[fuel_ir::storage::OutputView]>>,
    /// Per-node **storage-class override** ([`StorageClass`]). Sparse:
    /// entries are present only where the class differs from the op-inferred
    /// default ([`infer_storage_class`]) — chiefly KV-cache placeholder
    /// `Op::Const`s, which are session state, not shared weights.
    /// [`Graph::storage_class`] returns the override if present, else the
    /// inferred default, so every node has a well-defined class.
    /// Phase D substrate (PR-D1); not yet consumed by realize.
    storage_class: HashMap<NodeId, StorageClass>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            placements: HashMap::new(),
            side_effect_roots: Vec::new(),
            storage_map: HashMap::new(),
            target_backends: HashMap::new(),
            layouts: HashMap::new(),
            node_output_views: HashMap::new(),
            storage_class: HashMap::new(),
        }
    }

    /// Look up the resolved target backend for a node, if any has
    /// been set. Returns `None` if dispatch resolution hasn't
    /// happened yet for this node — common during the migration
    /// because old op-builder paths don't call the resolver.
    ///
    /// Phase 7.5 B2.
    pub fn target_backend(&self, id: NodeId) -> Option<BackendId> {
        self.target_backends.get(&id).copied()
    }

    /// Record the resolved target backend for `id`. Set once at DAG
    /// construction time (or first-realize time, depending on the
    /// dispatch policy chosen in Phase B3). Idempotent: calling
    /// twice with the same value is fine; calling twice with
    /// different values overwrites.
    ///
    /// Phase 7.5 B2.
    pub fn set_target_backend(&mut self, id: NodeId, backend: BackendId) {
        assert!(
            id.0 < self.nodes.len(),
            "set_target_backend: id out of bounds",
        );
        self.target_backends.insert(id, backend);
    }

    /// Number of nodes with a resolved target backend. Mostly for
    /// tests and diagnostics.
    pub fn target_backend_count(&self) -> usize {
        self.target_backends.len()
    }

    /// The storage class of `id`: the explicit override if one was set via
    /// [`Graph::set_storage_class`], else the op-inferred default
    /// ([`infer_storage_class`]). Every node has a well-defined class.
    ///
    /// Phase D substrate (PR-D1).
    pub fn storage_class(&self, id: NodeId) -> StorageClass {
        if let Some(c) = self.storage_class.get(&id).copied() {
            return c;
        }
        infer_storage_class(&self.nodes[id.0].op)
    }

    /// Record an explicit storage-class override for `id`, replacing the
    /// op-inferred default. Used for nodes whose op does not determine the
    /// class — chiefly KV-cache placeholder `Op::Const`s, which are session
    /// state rather than shared weights.
    ///
    /// Phase D substrate (PR-D1).
    pub fn set_storage_class(&mut self, id: NodeId, class: StorageClass) {
        assert!(
            id.0 < self.nodes.len(),
            "set_storage_class: id out of bounds",
        );
        self.storage_class.insert(id, class);
    }

    /// Whether `id` carries an explicit storage-class override (as opposed
    /// to relying on the op-inferred default).
    pub fn has_storage_class_override(&self, id: NodeId) -> bool {
        self.storage_class.contains_key(&id)
    }

    /// Number of explicit storage-class overrides. Mostly for tests and
    /// diagnostics.
    pub fn storage_class_override_count(&self) -> usize {
        self.storage_class.len()
    }

    /// The [`Layout`] associated with `id`. If the side-table has
    /// no explicit entry, returns `Layout::contiguous(shape)` —
    /// every Node logically has a Layout; the side-table is just
    /// the sparse storage for the strided / offset cases.
    ///
    /// Phase 7.5 storage-unification: metadata-only view ops
    /// (`Op::Transpose`, `Op::Permute`, `Op::BroadcastTo`,
    /// 0-copy `Op::Reshape`) populate this side-table during
    /// graph construction. Kernel ops produce contiguous results
    /// and rely on the fallback.
    pub fn layout(&self, id: NodeId) -> Layout {
        if let Some(l) = self.layouts.get(&id) {
            return l.clone();
        }
        Layout::contiguous(self.node(id).shape.clone())
    }

    /// Record an explicit [`Layout`] for `id`. The Layout's
    /// `shape()` must equal `nodes[id].shape` — otherwise the
    /// graph becomes incoherent (downstream ops were validated
    /// against `node.shape`). Panics in `debug_assertions` mode
    /// on mismatch; in release the mismatch is silently kept
    /// because the alternative would require this method to
    /// return `Result`, which would break call ergonomics in
    /// graph-construction code.
    ///
    /// Set this exactly when an op produces a strided / offset
    /// view over upstream Storage. For the common case
    /// (contiguous kernel output), don't call this at all —
    /// [`Graph::layout`] returns `Layout::contiguous(shape)`
    /// as the implicit fallback.
    pub fn set_layout(&mut self, id: NodeId, layout: Layout) {
        assert!(id.0 < self.nodes.len(), "set_layout: id out of bounds");
        debug_assert_eq!(
            layout.shape(),
            &self.nodes[id.0].shape,
            "set_layout: Layout shape disagrees with Node shape \
             (Layout sets strides/offset, not the visible shape)",
        );
        self.layouts.insert(id, layout);
    }

    /// Whether `id` has an explicit Layout entry in the side-table
    /// (i.e. it's a strided / offset view; not the default
    /// contiguous case).
    pub fn has_explicit_layout(&self, id: NodeId) -> bool {
        self.layouts.contains_key(&id)
    }

    /// Number of nodes with an explicit (non-contiguous) Layout.
    /// Mostly for tests and diagnostics.
    pub fn explicit_layout_count(&self) -> usize {
        self.layouts.len()
    }

    // -------------------------------------------------------------------
    // Multi-output side-table (Op::View / Op::ViewOwned)
    // -------------------------------------------------------------------

    /// Borrow the per-slot output-view specs for `id`, if it's been
    /// declared as a multi-output producer. Returns `None` for
    /// single-output nodes (the common case).
    ///
    /// `Op::View` and `Op::ViewOwned` builders read this to derive
    /// the slot's shape/dtype/layout at graph-build time.
    pub fn output_views(
        &self,
        id: NodeId,
    ) -> Option<&[fuel_ir::storage::OutputView]> {
        self.node_output_views.get(&id).map(|a| a.as_ref())
    }

    /// Clone the `Arc` handle to `id`'s per-slot output-view specs.
    /// Used by the realization path to attach the same bundle metadata
    /// to the realized [`Storage`] without copying the slice.
    pub fn output_views_arc(
        &self,
        id: NodeId,
    ) -> Option<Arc<[fuel_ir::storage::OutputView]>> {
        self.node_output_views.get(&id).cloned()
    }

    /// Declare `id` as a multi-output producer with the given per-slot
    /// specs. Validates at graph-build time:
    /// 1. `id` is in bounds.
    /// 2. `views` is non-empty (a "multi-output" node with zero slots
    ///    is a contract bug — use single-output if there's nothing to
    ///    project).
    /// 3. Slot 0's dtype equals `nodes[id].dtype` (the bundle's
    ///    primary dtype convention).
    /// 4. Slot 0's shape equals `nodes[id].shape` (slot 0 is the
    ///    "primary" slot whose shape the single-output infrastructure
    ///    already sees).
    /// 5. Each slot's `layout.shape()` equals the slot's `shape`
    ///    (Layout adds strides/offset; the visible shape is identical).
    ///
    /// Idempotent on (id, views). Replacing an existing declaration
    /// asserts the new specs are byte-identical to the old in debug
    /// builds; in release the last write wins (matches the layouts
    /// side-table pattern).
    pub fn set_output_views(
        &mut self,
        id:    NodeId,
        views: Arc<[fuel_ir::storage::OutputView]>,
    ) -> Result<(), fuel_ir::Error> {
        if id.0 >= self.nodes.len() {
            return Err(fuel_ir::Error::Msg(format!(
                "set_output_views: NodeId({}) out of bounds (len={})",
                id.0, self.nodes.len(),
            )).bt());
        }
        if views.is_empty() {
            return Err(fuel_ir::Error::Msg(
                "set_output_views: slot list must be non-empty"
                    .into(),
            ).bt());
        }
        let node = &self.nodes[id.0];
        let slot0 = &views[0];
        if slot0.dtype != node.dtype {
            return Err(fuel_ir::Error::Msg(format!(
                "set_output_views: slot 0 dtype {:?} disagrees with \
                 Node::dtype {:?} on Node#{} (slot 0 is the primary; \
                 dtypes must match)",
                slot0.dtype, node.dtype, id.0,
            )).bt());
        }
        if slot0.shape != node.shape {
            return Err(fuel_ir::Error::Msg(format!(
                "set_output_views: slot 0 shape {:?} disagrees with \
                 Node::shape {:?} on Node#{} (slot 0 is the primary)",
                slot0.shape, node.shape, id.0,
            )).bt());
        }
        for (i, v) in views.iter().enumerate() {
            if v.layout.shape() != &v.shape {
                return Err(fuel_ir::Error::Msg(format!(
                    "set_output_views: slot {i} layout.shape() = {:?} \
                     disagrees with declared slot shape {:?}",
                    v.layout.shape(), v.shape,
                )).bt());
            }
        }
        self.node_output_views.insert(id, views);
        Ok(())
    }

    /// Whether `id` has been declared as a multi-output producer.
    pub fn is_multi_output(&self, id: NodeId) -> bool {
        self.node_output_views.contains_key(&id)
    }

    /// Number of nodes declared as multi-output producers. Mostly for
    /// tests and diagnostics.
    pub fn multi_output_count(&self) -> usize {
        self.node_output_views.len()
    }

    /// Look up the realized storage for a node, if any has been
    /// registered. Returns an Arc clone — the slot stays in the map
    /// after this call.
    pub fn storage_for(&self, id: NodeId) -> Option<Arc<RwLock<Storage>>> {
        self.storage_map.get(&id).cloned()
    }

    /// Register a realized-storage slot for `id`. Replaces any
    /// existing entry (callers responsible for ensuring this is the
    /// intended semantics — e.g. re-realization after eviction).
    pub fn set_storage(&mut self, id: NodeId, storage: Arc<RwLock<Storage>>) {
        assert!(id.0 < self.nodes.len(), "set_storage: id out of bounds");
        self.storage_map.insert(id, storage);
    }

    /// Convenience: register an owned `Storage` directly. Wraps in
    /// `Arc<RwLock<...>>`.
    pub fn set_storage_owned(&mut self, id: NodeId, storage: Storage) {
        self.set_storage(id, Arc::new(RwLock::new(storage)));
    }

    /// Remove a slot. Used by eviction / `Op::Release` paths.
    /// Returns the Arc if present so callers that want to keep the
    /// bytes alive can hold it; otherwise drops on the floor and any
    /// outstanding Arc clones still see the bytes.
    pub fn remove_storage(&mut self, id: NodeId) -> Option<Arc<RwLock<Storage>>> {
        self.storage_map.remove(&id)
    }

    /// Whether a slot is currently registered for `id`.
    pub fn has_storage(&self, id: NodeId) -> bool {
        self.storage_map.contains_key(&id)
    }

    /// Number of registered storage slots.
    pub fn storage_len(&self) -> usize {
        self.storage_map.len()
    }

    /// Register a NodeId as a side-effect root — a node whose output
    /// isn't reachable from any user-requested root but whose
    /// execution the executor must still schedule. Used for
    /// destructive ops (e.g. `Op::Release`) emitted by graph-mutating
    /// rules: the rule needs Release to run (freeing device memory),
    /// but Release's zero-element marker has no consumer.
    ///
    /// The executor's realize paths walk the user's roots AND these
    /// side-effect roots via a single combined `execution_plan` call.
    pub fn add_side_effect_root(&mut self, id: NodeId) {
        assert!(id.0 < self.nodes.len(), "add_side_effect_root: id out of bounds");
        if !self.side_effect_roots.contains(&id) {
            self.side_effect_roots.push(id);
        }
    }

    /// Snapshot the registered side-effect roots. Executor concatenates
    /// these with the user's requested roots before computing the plan.
    pub fn side_effect_roots(&self) -> &[NodeId] {
        &self.side_effect_roots
    }

    /// The number of nodes currently in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has any nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Borrow a node by ID. Panics if the ID is out of bounds — it should
    /// not be possible for a valid `Tensor` handle to produce an invalid
    /// ID, so any such panic indicates a bug in graph construction.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Produce a one-line human-readable description of a node and its
    /// immediate inputs — the identifier string that executor panics
    /// prepend so realize-time failures can be localized to the exact
    /// graph position that blew up. Format is stable enough for grep
    /// but explicitly intended for humans reading panic output, not for
    /// parsing.
    pub fn describe_node(&self, id: NodeId) -> String {
        let n = self.node(id);
        let op_short = n.op.short_name();
        if n.inputs.is_empty() {
            return format!(
                "Node#{} ({op_short}, out shape={:?} dtype={:?})",
                id.0, n.shape, n.dtype,
            );
        }
        let inputs: Vec<String> = n.inputs.iter().map(|&inp| {
            let ni = self.node(inp);
            let ip = ni.op.short_name();
            format!("Node#{}[{ip}, {:?}, {:?}]", inp.0, ni.shape, ni.dtype)
        }).collect();
        format!(
            "Node#{} ({op_short}, out shape={:?} dtype={:?}, inputs=[{}])",
            id.0, n.shape, n.dtype, inputs.join(", "),
        )
    }

    /// Append a node and return its fresh ID. Used by the `Tensor`
    /// builders, by `opt` passes that canonicalize or rewrite the
    /// graph by appending fresh nodes, and by external consumers
    /// (fuel-storage's pipelined executor tests, custom-op authors)
    /// that need direct graph construction.
    ///
    /// For metadata-only view ops (`Op::Transpose`, `Op::Permute`,
    /// `Op::BroadcastTo`, `Op::Slice`) this also populates the
    /// [`layouts`](Self::layout) side-table by deriving the output
    /// layout from the input's layout via
    /// [`derive_view_output_layout`]. After this call,
    /// `graph.layout(new_id)` returns the strided/offset view layout
    /// without any further work — the side-table is the single source
    /// of truth for layout, with [`Layout::contiguous`] as the
    /// fallback for kernel-output nodes.
    ///
    /// Misuse (malformed Node, dangling input ids, etc.) is caught
    /// at execute time as a typed error, not a panic — per project's
    /// no-panic-in-production rule.
    pub fn push(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        let is_view = node.op.is_view_op();
        let view_input = if is_view { node.inputs.first().copied() } else { None };
        let op_for_derive = if is_view { Some(node.op.clone()) } else { None };
        self.nodes.push(node);
        if let (Some(input_id), Some(op)) = (view_input, op_for_derive) {
            if input_id.0 < self.nodes.len() {
                let input_layout = self.layout(input_id);
                if let Ok(out_layout) = derive_view_output_layout(&op, &input_layout) {
                    self.set_layout(id, out_layout);
                }
            }
        }
        id
    }

    /// Rewrite every occurrence of `old_input` in `node`'s input list to
    /// `new_input`. In-place mutation — the *only* exception to the
    /// otherwise append-only invariant.
    ///
    /// Callers are responsible for ensuring `new_input` is semantically
    /// equivalent to `old_input` at this edge: same shape, same dtype,
    /// and produces data the consumer can read without error on the
    /// relevant device. Used by transform passes (residency eviction,
    /// fusion) that need to redirect a specific consumer's edge.
    pub(crate) fn rewrite_input(
        &mut self,
        node: NodeId,
        old_input: NodeId,
        new_input: NodeId,
    ) {
        assert!(node.0 < self.nodes.len(), "rewrite_input: node out of bounds");
        let inputs = &mut self.nodes[node.0].inputs;
        for inp in inputs.iter_mut() {
            if *inp == old_input {
                *inp = new_input;
            }
        }
    }

    /// Tag a node with a target device. The executor will validate
    /// (post-Phase-1) that the node's op can be evaluated on that
    /// device. In Phase 1 the tag is informational only.
    pub fn set_placement(&mut self, id: NodeId, loc: DeviceLocation) {
        assert!(id.0 < self.nodes.len(), "set_placement: id out of bounds");
        self.placements.insert(id, loc);
    }

    /// Read a node's placement hint, or `None` if the node inherits from
    /// the executor default.
    pub fn placement(&self, id: NodeId) -> Option<DeviceLocation> {
        self.placements.get(&id).copied()
    }

    /// Open a [`BranchBuilder`] rooted at a shared `diverge` point — the
    /// first half of the `open_branch` / `add_arm` / `finalize_branches`
    /// trio that constructs an [`Op::Branch`] (phi/merge) node with
    /// build-time validation (Phase A PR-A1 of the "plan IS the graph"
    /// rebuild).
    ///
    /// `diverge` names the single node from which every candidate route
    /// (arm) departs; each arm's *exit* (added via
    /// [`BranchBuilder::add_arm`]) becomes one `inputs[i]` of the emitted
    /// `Op::Branch`. The builder borrows nothing — it is a pure
    /// accumulator of `NodeId`s — so the graph stays free to keep building
    /// arms between `open_branch` and `finalize_branches`. All validation
    /// (descendant `reconverge_at`, internally-disjoint arms,
    /// cast-to-uniform shape/dtype, arm-0 runnability) happens in
    /// [`BranchBuilder::finalize_branches`], which returns a typed
    /// [`Error::InvalidBranch`] rather than panicking.
    pub fn open_branch(&self, diverge: NodeId) -> BranchBuilder {
        BranchBuilder { diverge, arms: Vec::new() }
    }

    /// Forward-reachable set (the node and all its transitive *consumers*)
    /// from `from`, computed against a precomputed consumer adjacency.
    /// Used by branch validation to test descendant-hood and arm
    /// containment without re-scanning the whole arena per query.
    fn forward_reachable(
        consumers: &HashMap<NodeId, Vec<NodeId>>,
        from: NodeId,
    ) -> HashSet<NodeId> {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(cs) = consumers.get(&n) {
                for &c in cs {
                    if !seen.contains(&c) {
                        stack.push(c);
                    }
                }
            }
        }
        seen
    }

    /// Backward-reachable set (the node and all its transitive *inputs*)
    /// from `from`. Used to bound an arm to the nodes that actually lie
    /// on a path `diverge → exit`.
    fn backward_reachable(&self, from: NodeId) -> HashSet<NodeId> {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            for &inp in &self.nodes[n.0].inputs {
                if !seen.contains(&inp) {
                    stack.push(inp);
                }
            }
        }
        seen
    }

    // -------------------------------------------------------------------
    // Compaction of the append-only arena (PR-B4)
    // -------------------------------------------------------------------

    /// The set of nodes that constitute the **live structure** of the
    /// graph relative to `roots`: everything reachable, following node
    /// `inputs`, from the union of
    ///
    /// - the caller's `roots`,
    /// - the `side_effect_roots` (nodes the executor must schedule even
    ///   when no output reads them — e.g. `Op::Release` eviction), and
    /// - the full multi-path structure: for every reachable `Op::Branch`,
    ///   its `reconverge_at` and **all its arms** (the arm exits, which
    ///   are the Branch node's `inputs`, and transitively their interiors).
    ///   Phase C's route picker needs every arm, so the alternative arms
    ///   must survive even though nothing downstream reads them directly
    ///   (the `reconverge_at` node reads only arm 0).
    ///
    /// A finalized `Op::Branch` is itself typically orphaned — nothing
    /// downstream reads it (PR-A1's runnability invariant has the merge
    /// read arm 0 directly, not the Branch). So a plain forward walk from
    /// `roots` would miss it. We therefore mirror [`run::extract_runs_multi`]'s
    /// `effective_roots` discipline: scan the arena and seed any Branch
    /// whose `reconverge_at` (or any arm exit) is already reachable, to a
    /// fixpoint, then take the input-closure of all seeds. The result is
    /// the exact set compaction must keep.
    fn live_set(&self, roots: &[NodeId]) -> HashSet<NodeId> {
        // Seeds: roots + side-effect roots. We grow this set with the
        // Branch nodes that participate in the reachable computation, then
        // take its full input-closure.
        let mut seeds: Vec<NodeId> = Vec::new();
        let mut seed_set: HashSet<NodeId> = HashSet::new();
        let push_seed = |id: NodeId, seeds: &mut Vec<NodeId>, seed_set: &mut HashSet<NodeId>| {
            if id.0 < self.nodes.len() && seed_set.insert(id) {
                seeds.push(id);
            }
        };
        for &r in roots {
            push_seed(r, &mut seeds, &mut seed_set);
        }
        for &r in &self.side_effect_roots {
            push_seed(r, &mut seeds, &mut seed_set);
        }

        // Grow the seed set with participating Branch nodes to a fixpoint.
        // A Branch "participates" when its merge target or any arm exit is
        // already reachable from the current seeds (following inputs). Each
        // newly-seeded Branch may pull in further arms whose interiors then
        // make another Branch participate, hence the fixpoint.
        loop {
            let reachable = self.input_closure(&seeds);
            let mut added = false;
            for idx in 0..self.nodes.len() {
                let id = NodeId(idx);
                if seed_set.contains(&id) {
                    continue;
                }
                let Op::Branch { reconverge_at } = self.nodes[idx].op else {
                    continue;
                };
                let participates = reachable.contains(&reconverge_at)
                    || self.nodes[idx].inputs.iter().any(|a| reachable.contains(a));
                if participates {
                    seed_set.insert(id);
                    seeds.push(id);
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        // The live set is the full input-closure of the (roots + side-effect
        // + participating-Branch) seeds, plus each reachable Branch's
        // explicit `reconverge_at` (which is downstream of the arms, so it
        // is not pulled in by following inputs from the Branch). The merge
        // node is normally reachable from `roots` already, but seeding it
        // explicitly makes the live set self-consistent — a Branch never
        // outlives its merge target.
        let mut live = self.input_closure(&seeds);
        let recon_seeds: Vec<NodeId> = live
            .iter()
            .filter_map(|&id| match self.nodes[id.0].op {
                Op::Branch { reconverge_at } => Some(reconverge_at),
                _ => None,
            })
            .collect();
        if !recon_seeds.is_empty() {
            for n in self.input_closure(&recon_seeds) {
                live.insert(n);
            }
        }
        live
    }

    /// Every node reachable from any of `from`, following node `inputs`
    /// transitively (the backward / input cone). Mirrors
    /// [`topo_order_multi`] but returns just the set.
    fn input_closure(&self, from: &[NodeId]) -> HashSet<NodeId> {
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = from.to_vec();
        while let Some(n) = stack.pop() {
            if n.0 >= self.nodes.len() || !seen.insert(n) {
                continue;
            }
            for &inp in &self.nodes[n.0].inputs {
                if !seen.contains(&inp) {
                    stack.push(inp);
                }
            }
        }
        seen
    }

    /// **Required compaction of the append-only arena** (PR-B4).
    ///
    /// Drops every node *not* in the live structure relative to `roots`
    /// (the exploration debris that pathfinders leave) and rebuilds the
    /// arena with contiguous, renumbered [`NodeId`]s, remapping **every**
    /// `NodeId` reference in the graph. Returns the [`NodeRemap`] from old
    /// to new ids so callers can fix up their own `Tensor` handles / roots.
    ///
    /// The live set follows node `inputs` from: `roots`, the
    /// `side_effect_roots`, and the full multi-path structure — for every
    /// reachable [`Op::Branch`], its `reconverge_at` **and all its arms**
    /// (so the route picker's alternatives survive). See [`Graph::live_set`].
    ///
    /// What is remapped (the correctness crux — a missed remap is silent
    /// corruption). Every NodeId-bearing structure in the graph:
    /// - each surviving [`Node`]'s `inputs`;
    /// - the `NodeId` carried *inside* an op variant — [`Op::Branch`]'s
    ///   `reconverge_at` (the only `Op` variant carrying a `NodeId`);
    /// - every NodeId-keyed side-table on [`Graph`]: `placements`,
    ///   `target_backends`, `layouts`, `storage_map`, `node_output_views`;
    /// - the `side_effect_roots` vector.
    ///
    /// # Not in the per-realize hot path
    ///
    /// Compaction renumbers `NodeId`s, which would invalidate a realize
    /// already built around the current ids (the dispatch order and the
    /// `StorageCache` key are `NodeId`). It is therefore a **standalone
    /// pass**, deliberately **not** registered as an `Optimizer` in the
    /// per-realize driver / `optimize_graph` loop. It runs at load-time
    /// **between optimization rounds** (Phase D) and is **required before
    /// finalize-to-disk** (Phase E) so the persisted `.fuel` holds the lean
    /// result, not exploration debris. Neither caller exists yet.
    ///
    /// Orphans never change results, so this is a *size* pass, not a
    /// *correctness* one — but the remap it performs must be exhaustive or
    /// it would itself introduce corruption, which is the property the
    /// born-red tests + [`Graph::verify_no_dangling`] guard.
    ///
    /// This is the implementation behind the free function [`compact`];
    /// call that.
    fn compact_in_place(&mut self, roots: &[NodeId]) -> NodeRemap {
        let live = self.live_set(roots);
        let n = self.nodes.len();

        // Assign new contiguous ids in ascending old-id order, so the
        // relative order of surviving nodes (and hence any topological
        // property over them) is preserved.
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; n];
        let mut next = 0usize;
        for idx in 0..n {
            if live.contains(&NodeId(idx)) {
                old_to_new[idx] = Some(NodeId(next));
                next += 1;
            }
        }
        let remap = NodeRemap { old_to_new };

        // Rebuild the node arena, remapping each surviving node's inputs and
        // any op-carried NodeId. Surviving nodes' inputs are themselves
        // always live (the live set is input-closed), so every `map` here
        // resolves.
        let mut new_nodes: Vec<Node> = Vec::with_capacity(next);
        for idx in 0..n {
            if remap.old_to_new[idx].is_none() {
                continue;
            }
            let mut node = self.nodes[idx].clone();
            for inp in node.inputs.iter_mut() {
                *inp = remap.expect(*inp);
            }
            if let Op::Branch { reconverge_at } = &mut node.op {
                *reconverge_at = remap.expect(*reconverge_at);
            }
            new_nodes.push(node);
        }
        self.nodes = new_nodes;

        // Remap every NodeId-keyed side-table. Drop entries for dropped
        // nodes; rekey survivors to their new id.
        self.placements = remap.rekey(std::mem::take(&mut self.placements));
        self.target_backends = remap.rekey(std::mem::take(&mut self.target_backends));
        self.layouts = remap.rekey(std::mem::take(&mut self.layouts));
        self.storage_map = remap.rekey(std::mem::take(&mut self.storage_map));
        self.node_output_views = remap.rekey(std::mem::take(&mut self.node_output_views));
        self.storage_class = remap.rekey(std::mem::take(&mut self.storage_class));

        // Remap the side-effect-roots vector (every entry is live — they
        // were reachability seeds — so each maps).
        self.side_effect_roots = self
            .side_effect_roots
            .iter()
            .map(|&id| remap.expect(id))
            .collect();

        debug_assert!(
            self.verify_no_dangling().is_ok(),
            "compact left a dangling NodeId reference: {:?}",
            self.verify_no_dangling().err(),
        );
        remap
    }

    /// Verify that **no** `NodeId` reference in the graph points outside
    /// the arena — the post-compaction safety net for a missed remap.
    /// Checks every node's `inputs`, the op-carried `Op::Branch`
    /// `reconverge_at`, every NodeId-keyed side-table, and the
    /// `side_effect_roots` vector. Returns the first offending reference as
    /// a typed error (never panics); used by [`Graph::compact`]'s
    /// `debug_assert` and by tests.
    pub fn verify_no_dangling(&self) -> std::result::Result<(), fuel_ir::Error> {
        let n = self.nodes.len();
        let bad = |what: String| {
            Err(fuel_ir::Error::Msg(format!(
                "dangling NodeId reference: {what} (arena has {n} nodes)",
            )))
        };
        for (idx, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                if inp.0 >= n {
                    return bad(format!("Node#{idx} input Node#{}", inp.0));
                }
            }
            if let Op::Branch { reconverge_at } = node.op {
                if reconverge_at.0 >= n {
                    return bad(format!("Node#{idx} Branch.reconverge_at Node#{}", reconverge_at.0));
                }
            }
        }
        for id in self.placements.keys() {
            if id.0 >= n {
                return bad(format!("placements key Node#{}", id.0));
            }
        }
        for id in self.target_backends.keys() {
            if id.0 >= n {
                return bad(format!("target_backends key Node#{}", id.0));
            }
        }
        for id in self.layouts.keys() {
            if id.0 >= n {
                return bad(format!("layouts key Node#{}", id.0));
            }
        }
        for id in self.storage_map.keys() {
            if id.0 >= n {
                return bad(format!("storage_map key Node#{}", id.0));
            }
        }
        for id in self.node_output_views.keys() {
            if id.0 >= n {
                return bad(format!("node_output_views key Node#{}", id.0));
            }
        }
        for id in self.storage_class.keys() {
            if id.0 >= n {
                return bad(format!("storage_class key Node#{}", id.0));
            }
        }
        for &id in &self.side_effect_roots {
            if id.0 >= n {
                return bad(format!("side_effect_root Node#{}", id.0));
            }
        }
        Ok(())
    }
}

/// **Required compaction of the append-only arena** (PR-B4).
///
/// Drops every node *not* in the live structure relative to `roots` (the
/// exploration debris that pathfinders leave) and rebuilds the arena with
/// contiguous, renumbered [`NodeId`]s, remapping **every** `NodeId`
/// reference in the graph. Returns the [`NodeRemap`] from old → new ids so
/// callers can fix up their own roots / `Tensor` handles.
///
/// The live set follows node `inputs` from `roots`, the
/// `side_effect_roots`, and the full multi-path structure — for every
/// reachable [`Op::Branch`], its `reconverge_at` **and all its arms** (so
/// Phase C's route picker keeps its alternatives). Everything else is
/// dropped.
///
/// **Not in the per-realize hot path.** Compaction renumbers `NodeId`s,
/// which would invalidate a realize built around the current ids (the
/// dispatch order + the `StorageCache` key are `NodeId`). It is a
/// standalone pass — deliberately **not** registered as an `Optimizer` in
/// the per-realize driver / `optimize_graph` loop — run at load-time
/// **between optimization rounds** (Phase D) and **required before
/// finalize-to-disk** (Phase E). Neither caller exists yet.
///
/// See [`Graph::verify_no_dangling`] for the post-condition the
/// `debug_assert` checks: no `NodeId` reference (node inputs, the
/// op-carried `Op::Branch.reconverge_at`, any side-table, or
/// `side_effect_roots`) points outside the new arena.
pub fn compact(graph: &mut Graph, roots: &[NodeId]) -> NodeRemap {
    graph.compact_in_place(roots)
}

/// The old → new `NodeId` mapping produced by [`compact`].
///
/// Indexed by the *old* `NodeId`: `old_to_new[old.0]` is `Some(new)` if the
/// node survived compaction, or `None` if it was dropped as unreachable
/// debris. Callers use [`NodeRemap::get`] to translate roots / `Tensor`
/// ids they still hold; passes that know a node must have survived use
/// [`NodeRemap::expect`].
#[derive(Debug, Clone)]
pub struct NodeRemap {
    old_to_new: Vec<Option<NodeId>>,
}

impl NodeRemap {
    /// The new id for `old`, or `None` if `old` was dropped (or is out of
    /// range of the pre-compaction arena).
    pub fn get(&self, old: NodeId) -> Option<NodeId> {
        self.old_to_new.get(old.0).copied().flatten()
    }

    /// Whether `old` survived compaction.
    pub fn survived(&self, old: NodeId) -> bool {
        self.get(old).is_some()
    }

    /// The new id for `old`, asserting it survived. Used internally when
    /// remapping a reference that is known-live (a surviving node's input,
    /// a reachable Branch's `reconverge_at`, a side-effect root) — a `None`
    /// here would mean the live-set computation was unsound.
    fn expect(&self, old: NodeId) -> NodeId {
        self.get(old).unwrap_or_else(|| {
            panic!(
                "NodeRemap::expect: Node#{} was expected to survive compaction \
                 but was dropped — the live set is unsound (a referenced node \
                 was not kept)",
                old.0,
            )
        })
    }

    /// Rekey a `NodeId`-keyed side-table: drop entries whose key was
    /// dropped, rewrite surviving keys to their new id. Values are moved,
    /// not cloned.
    fn rekey<V>(&self, table: HashMap<NodeId, V>) -> HashMap<NodeId, V> {
        let mut out = HashMap::with_capacity(table.len());
        for (old, v) in table {
            if let Some(new) = self.get(old) {
                out.insert(new, v);
            }
        }
        out
    }
}

/// Accumulates the arms of an in-construction [`Op::Branch`] (phi/merge)
/// node before [`finalize_branches`](Self::finalize_branches) validates
/// them and emits the node. Created by [`Graph::open_branch`].
///
/// The multi-path structure is an **arena fact, not an overlay**: a
/// `Branch` node's divergent arms are its `inputs` (each input is one
/// route's *exit*), and the op carries an explicit `reconverge_at`. The
/// builder is a pure `NodeId` accumulator — it holds no graph reference —
/// so the graph remains free to keep appending nodes (further arms) while
/// the builder is open. Nothing mutates the arena until
/// `finalize_branches`, which is the single validation gate.
#[derive(Debug, Clone)]
pub struct BranchBuilder {
    /// The shared node every arm departs from.
    diverge: NodeId,
    /// Each arm's *exit* node, in priority order. `arms[0]` is the
    /// runnability fallback (arm 0): the route a finalized-but-unpicked
    /// graph realizes on.
    arms: Vec<NodeId>,
}

impl BranchBuilder {
    /// Append one arm by its *exit* node — the node whose value the merge
    /// reads for this route. The first arm added is **arm 0**, the
    /// runnability fallback: `reconverge_at` must read arm 0's exit so a
    /// finalized-but-not-yet-route-picked graph still realizes (validated
    /// in [`finalize_branches`](Self::finalize_branches)).
    ///
    /// Returns `&mut Self` for chaining. Validation is deferred to
    /// finalize — `add_arm` never inspects the graph and never fails.
    pub fn add_arm(&mut self, exit: NodeId) -> &mut Self {
        self.arms.push(exit);
        self
    }

    /// The shared diverge point this branch departs from.
    pub fn diverge(&self) -> NodeId {
        self.diverge
    }

    /// The arm exits accumulated so far, in add order (arm 0 first).
    pub fn arms(&self) -> &[NodeId] {
        &self.arms
    }

    /// Validate and emit the [`Op::Branch`] node, returning its fresh
    /// `NodeId`. This is the single build-time gate; it **never panics**,
    /// surfacing every rejection as [`Error::InvalidBranch`].
    ///
    /// Returns:
    /// - `Ok(Some(branch_id))` — a multi-arm branch was emitted.
    /// - `Ok(None)` — a **single-arm branch collapsed** to that arm (no
    ///   real decision point), leaving the arena untouched.
    /// - `Err(Error::InvalidBranch { .. })` — a validation rule failed.
    ///
    /// Validation rules (all checked before any mutation):
    /// 1. At least one arm; `diverge`, `reconverge_at`, and every arm
    ///    exit are in-bounds.
    /// 2. **Descendant `reconverge_at`** — the merge must be a forward
    ///    descendant of (and distinct from) `diverge`.
    /// 3. **Internally-disjoint arms** — each arm's interior nodes (the
    ///    nodes on paths `diverge → exit`, excluding the shared `diverge`)
    ///    must neither be shared with another arm nor read by any node
    ///    outside the branch (`reconverge_at` reading an arm exit is the
    ///    one allowed external consumer).
    /// 4. **Cast-to-uniform** — all arm exits agree on shape & dtype.
    ///    PR-A1 *validates* uniformity and errors on mismatch (it does not
    ///    insert `Cast` nodes); dtype-lowered alternatives must therefore
    ///    cast back to the merge dtype *inside* their arm before the exit.
    /// 5. **Arm-0 runnability** — `reconverge_at` must read arm 0's exit,
    ///    so a finalized-but-unpicked graph still realizes on arm 0.
    pub fn finalize_branches(
        self,
        graph: &mut Graph,
        reconverge_at: NodeId,
    ) -> std::result::Result<Option<NodeId>, fuel_ir::Error> {
        let n = graph.nodes.len();
        let invalid = |reason: String| fuel_ir::Error::InvalidBranch { reason };

        // (1a) At least one arm.
        if self.arms.is_empty() {
            return Err(invalid(
                "branch has no arms; at least one route is required".to_string(),
            ));
        }
        // (1b) Bounds-check every referenced node up front so later
        // arena indexing cannot panic.
        let in_bounds = |id: NodeId| id.0 < n;
        if !in_bounds(self.diverge) {
            return Err(invalid(format!(
                "diverge Node#{} is out of bounds (graph has {n} nodes)",
                self.diverge.0,
            )));
        }
        if !in_bounds(reconverge_at) {
            return Err(invalid(format!(
                "reconverge_at Node#{} is out of bounds (graph has {n} nodes)",
                reconverge_at.0,
            )));
        }
        for &arm in &self.arms {
            if !in_bounds(arm) {
                return Err(invalid(format!(
                    "arm exit Node#{} is out of bounds (graph has {n} nodes)",
                    arm.0,
                )));
            }
        }

        // (d) Single-arm branches collapse back to that arm — no real
        // decision point, so no Op::Branch is emitted and the arena is
        // left untouched.
        if self.arms.len() == 1 {
            return Ok(None);
        }

        // Build a consumer (reverse) adjacency once for the reachability
        // queries below.
        let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            let consumer = NodeId(idx);
            for &inp in &node.inputs {
                consumers.entry(inp).or_default().push(consumer);
            }
        }

        // (2) reconverge_at must be a forward descendant of diverge and
        // distinct from it.
        if reconverge_at == self.diverge {
            return Err(invalid(format!(
                "reconverge_at Node#{} must be a strict descendant of the \
                 diverge point, not the diverge point itself",
                reconverge_at.0,
            )));
        }
        let fwd_from_diverge = Graph::forward_reachable(&consumers, self.diverge);
        if !fwd_from_diverge.contains(&reconverge_at) {
            return Err(invalid(format!(
                "reconverge_at Node#{} is not a descendant of diverge Node#{}",
                reconverge_at.0, self.diverge.0,
            )));
        }

        // (4) Cast-to-uniform: every arm exit must agree on shape & dtype
        // with arm 0 (PR-A1 validates uniformity rather than inserting
        // Cast nodes).
        let arm0 = self.arms[0];
        let (want_shape, want_dtype) = {
            let a0 = &graph.nodes[arm0.0];
            (a0.shape.clone(), a0.dtype)
        };
        for &arm in &self.arms[1..] {
            let node = &graph.nodes[arm.0];
            if node.shape != want_shape || node.dtype != want_dtype {
                return Err(invalid(format!(
                    "arm exit Node#{} ({:?} {:?}) disagrees with arm 0 Node#{} \
                     ({:?} {:?}); arms must cast to a uniform shape & dtype at \
                     reconverge",
                    arm.0, node.shape, node.dtype, arm0.0, want_shape, want_dtype,
                )));
            }
        }

        // (5) Arm-0 runnability: reconverge_at must read arm 0's exit, so
        // an unpicked graph realizes on arm 0.
        if !graph.nodes[reconverge_at.0].inputs.contains(&arm0) {
            return Err(invalid(format!(
                "reconverge_at Node#{} does not read arm-0 exit Node#{}; the \
                 arm-0 runnability invariant requires the merge to read arm 0 \
                 so a not-yet-route-picked graph still realizes",
                reconverge_at.0, arm0.0,
            )));
        }

        // (3) Internal disjointness. Each arm's node set is the nodes on
        // paths diverge → exit: backward-reachable from the exit AND
        // forward-reachable from diverge. The arm *interior* excludes the
        // shared diverge point.
        let mut arm_interiors: Vec<HashSet<NodeId>> = Vec::with_capacity(self.arms.len());
        for &exit in &self.arms {
            let back = graph.backward_reachable(exit);
            let mut interior: HashSet<NodeId> = back
                .intersection(&fwd_from_diverge)
                .copied()
                .collect();
            interior.remove(&self.diverge);
            // The exit itself is always part of the arm even if (for a
            // degenerate arm) it equals the diverge point — but a
            // diverge-equals-exit arm has an empty interior, which would
            // make the arm indistinguishable from "no route", so reject.
            if interior.is_empty() {
                return Err(invalid(format!(
                    "arm exit Node#{} has no interior strictly after diverge \
                     Node#{}; an arm must add at least one node between the \
                     diverge point and its exit",
                    exit.0, self.diverge.0,
                )));
            }
            arm_interiors.push(interior);
        }

        // (3a) Arms must not share interior nodes.
        for i in 0..arm_interiors.len() {
            for j in (i + 1)..arm_interiors.len() {
                if let Some(shared) = arm_interiors[i].intersection(&arm_interiors[j]).next() {
                    return Err(invalid(format!(
                        "arms {i} and {j} share interior Node#{}; arms must be \
                         internally disjoint",
                        shared.0,
                    )));
                }
            }
        }

        // (3b) No arm-interior node may be read from outside the branch.
        // The branch's universe is the union of all arm interiors plus
        // the merge node; reconverge_at reading an arm exit is the one
        // legitimate "external" consumer.
        let mut branch_universe: HashSet<NodeId> = HashSet::new();
        for interior in &arm_interiors {
            branch_universe.extend(interior.iter().copied());
        }
        branch_universe.insert(reconverge_at);
        for (i, interior) in arm_interiors.iter().enumerate() {
            for &m in interior {
                if let Some(cs) = consumers.get(&m) {
                    for &c in cs {
                        if !branch_universe.contains(&c) {
                            return Err(invalid(format!(
                                "arm {i} interior Node#{} is read from outside \
                                 the branch by Node#{}; arm interiors must not be \
                                 reachable from outside the branch",
                                m.0, c.0,
                            )));
                        }
                    }
                }
            }
        }

        // All validation passed — emit the Branch node. Its inputs ARE the
        // arm exits, in priority order (arm 0 first); reconverge_at is
        // carried on the op. Shape/dtype mirror arm 0 (the uniform merge
        // type validated above).
        let branch = graph.push(Node {
            op: Op::Branch { reconverge_at },
            inputs: self.arms.clone(),
            shape: want_shape,
            dtype: want_dtype,
        });
        Ok(Some(branch))
    }
}

/// A shared, mutable graph. Builders clone this cheaply and hand it to
/// every derived tensor so that all tensors in one computation point to
/// the same arena.
pub type SharedGraph = Arc<RwLock<Graph>>;

/// A handle to a node in a shared graph. Cheap to clone. Carries the
/// graph reference so builder methods have everything they need to append
/// a new node.
#[derive(Debug, Clone)]
pub struct Tensor {
    graph: SharedGraph,
    id:    NodeId,
}

impl Tensor {
    /// Wrap an existing `(graph, node_id)` pair as a Tensor handle.
    /// Used by graph-rewriting passes (e.g., `opt::insert_copies`)
    /// that produce new root `NodeId`s and need to hand them back to
    /// call sites as Tensors.
    pub fn from_existing(graph: SharedGraph, id: NodeId) -> Self {
        Self { graph, id }
    }

    /// The node ID this handle points to.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// The shared graph this handle belongs to. Exposed so executors and
    /// tests can walk the underlying arena.
    pub fn graph(&self) -> &SharedGraph {
        &self.graph
    }

    /// The shape of this tensor, read from the underlying node.
    pub fn shape(&self) -> Shape {
        self.graph.read().unwrap().node(self.id).shape.clone()
    }

    /// The dtype of this tensor, read from the underlying node.
    pub fn dtype(&self) -> DType {
        self.graph.read().unwrap().node(self.id).dtype
    }

    /// Phase 7.5 work item G: look up the realized-storage slot for
    /// this tensor's node, if any. Returns an Arc clone so the slot
    /// stays in the map after this call and the caller's Arc keeps
    /// the bytes alive even if a later eviction removes the map
    /// entry. `None` means "not realized yet" — the caller should
    /// realize the graph first.
    pub fn storage_for(&self) -> Option<Arc<RwLock<Storage>>> {
        self.graph.read().unwrap().storage_for(self.id)
    }

    /// The placement hint for this tensor's node, if one was set. `None`
    /// means "inherit from the executor's default device."
    pub fn placement(&self) -> Option<DeviceLocation> {
        self.graph.read().unwrap().placement(self.id)
    }

    /// Tag this tensor's node with a target device. The executor will
    /// validate (post-Phase-1) that the node's op can be evaluated on
    /// that device. Returns the same `Tensor` handle so this composes
    /// with builder-style graph construction:
    ///
    /// ```ignore
    /// let y = a.matmul(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    /// ```
    pub fn on_device(self, loc: DeviceLocation) -> Self {
        self.graph.write().unwrap().set_placement(self.id, loc);
        self
    }

    /// Release this tensor's device-resident storage. Appends an
    /// `Op::Release` node whose output is a zero-element marker (shape
    /// `[0]`, same dtype as self). Destructive on `self` — after this
    /// op runs, `self`'s storage is gone.
    ///
    /// The ordering-analysis pass (`opt::derive_ordering`, arriving in
    /// a follow-up PR) automatically pins the Release to run after
    /// every non-destructive reader of `self`. Callers should NOT
    /// invoke this directly in a graph that has other readers of
    /// `self` until that pass is in place — today's executor has no
    /// mechanism to stop them running in the wrong order.
    pub fn release(&self) -> Self {
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Release,
            inputs: vec![self.id],
            shape:  Shape::from_dims(&[0]),
            dtype,
        });
        Tensor { graph: Arc::clone(&self.graph), id }
    }

    /// Move this tensor's data to `target` device — destroying the
    /// source in the process. Appends an `Op::Move` node. The output
    /// has the same shape and dtype as the input and lives on
    /// `target`; the source's device storage is freed once the op
    /// runs.
    ///
    /// Semantically equivalent to `copy_to_device(target)` followed
    /// by dropping the source. Use this when you KNOW the source
    /// won't be needed after the transfer — it lets backends skip
    /// the intermediate alloc in fast-path cases and lets the
    /// scheduler free the source's device memory immediately.
    ///
    /// If the source may still be read by other ops, use
    /// [`Self::copy_to_device`] instead — Copy leaves the source
    /// resident.
    ///
    /// The ordering-analysis pass (`opt::derive_ordering`)
    /// automatically pins this op to run after every non-destructive
    /// reader of the source. The caller does NOT need to worry about
    /// topo ordering — just emit the Move wherever the data-flow
    /// says it should go.
    pub fn move_to_device(&self, target: DeviceLocation) -> Self {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Move { target },
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Tensor { graph: Arc::clone(&self.graph), id }
    }

    /// Copy this tensor's data to `target` device. Source stays
    /// resident. Appends an `Op::Copy` node. The output has the same
    /// shape and dtype as the input; only residency changes.
    ///
    /// Same-device copies are still a legal thing to ask for — the
    /// backend can optimize them into a cheap clone (or the scheduler
    /// can elide the Copy node entirely). Cross-device copies resolve
    /// through the pipelined executor's Copy arm (kernel lookup at
    /// `(OpKind::Copy, [dt, dt], source backend)`) — today a host
    /// round-trip; P2P comes later.
    pub fn copy_to_device(&self, target: DeviceLocation) -> Self {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Copy { target },
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Tensor { graph: Arc::clone(&self.graph), id }
    }

    /// Project one logical output out of a multi-output producer node.
    ///
    /// `self` must have been declared as a multi-output producer via
    /// [`Graph::set_output_views`] (Session 2's authoring contract will
    /// drive this for real fused ops; Session 1 tests drive it
    /// directly). The View's output shape, dtype, and layout come from
    /// `output_views(self.id)[slot]`. Zero bytes are moved at
    /// realization time — the View shares the producer's bundled
    /// `Arc<RwLock<Storage>>` and exposes the slot's typed window.
    ///
    /// **Returns `Result`**: the producer must have declared its slot
    /// specs; `slot` must be in range. Errors at graph-build time
    /// (per the `validate-at-graph-build-time` rule); the executor
    /// never sees a malformed View node.
    pub fn view(
        &self,
        slot: u32,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let mut graph_w = self.graph.write().unwrap();
        let (slot_shape, slot_dtype, slot_byte_offset, slot_layout) = {
            let views = graph_w.output_views(self.id).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "Tensor::view: Node#{} is not a multi-output \
                     producer (no output_views registered)",
                    self.id.0,
                )).bt()
            })?;
            let idx = slot as usize;
            let v = views.get(idx).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "Tensor::view: slot {idx} out of range \
                     (producer has {} slots)",
                    views.len(),
                )).bt()
            })?;
            (v.shape.clone(), v.dtype, v.byte_offset, v.layout.clone())
        };
        let id = graph_w.push(Node {
            op:     Op::View { slot },
            inputs: vec![self.id],
            shape:  slot_shape.clone(),
            dtype:  slot_dtype,
        });
        // Compose the slot's intrinsic layout (which carries
        // slot-relative strides + start_offset) with the slot's
        // byte_offset inside the bundle. The result is the layout a
        // downstream kernel sees when treating the producer's bundle
        // bytes as `slot_dtype` elements: start_offset is in slot-
        // dtype-element units; bundle byte_offset is divided by the
        // dtype's size_in_bytes to land at the slot's first byte. The
        // bundled allocator (`compose_bundle`) aligns each slot's
        // byte_offset to its dtype size — so the division is exact.
        let dtype_bytes = slot_dtype.size_in_bytes().max(1);
        let extra_offset = slot_byte_offset / dtype_bytes;
        let effective_layout = if extra_offset == 0 {
            slot_layout
        } else {
            Layout::new(
                slot_layout.shape().clone(),
                slot_layout.stride().to_vec().into_iter().collect(),
                slot_layout.start_offset() + extra_offset,
            )
        };
        // The side-table is set whenever the effective layout differs
        // from `Layout::contiguous(slot_shape)` — i.e. whenever the
        // slot has a strided layout OR a non-zero composed offset
        // (slot 1+ in any bundle).
        if effective_layout != Layout::contiguous(slot_shape) {
            graph_w.set_layout(id, effective_layout);
        }
        drop(graph_w);
        Ok(Tensor { graph: Arc::clone(&self.graph), id })
    }

    /// Owned variant of [`Self::view`] — the slot's bytes are copied
    /// into a fresh contiguous Storage at realization time, so the
    /// producer's bundle Arc can drop independently. Output layout
    /// is always contiguous.
    ///
    /// Same validation as `view`; same error surface.
    pub fn view_owned(
        &self,
        slot: u32,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let mut graph_w = self.graph.write().unwrap();
        let (slot_shape, slot_dtype) = {
            let views = graph_w.output_views(self.id).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "Tensor::view_owned: Node#{} is not a multi-output \
                     producer (no output_views registered)",
                    self.id.0,
                )).bt()
            })?;
            let idx = slot as usize;
            let v = views.get(idx).ok_or_else(|| {
                fuel_ir::Error::Msg(format!(
                    "Tensor::view_owned: slot {idx} out of range \
                     (producer has {} slots)",
                    views.len(),
                )).bt()
            })?;
            (v.shape.clone(), v.dtype)
        };
        let id = graph_w.push(Node {
            op:     Op::ViewOwned { slot },
            inputs: vec![self.id],
            shape:  slot_shape,
            dtype:  slot_dtype,
        });
        drop(graph_w);
        Ok(Tensor { graph: Arc::clone(&self.graph), id })
    }

    /// Append an `Op::WriteSlice` node — copies `source`'s bytes into
    /// `self` at the rectangular slab defined by `ranges`. Per-axis,
    /// `ranges[i] = (start, end)` is the half-open destination range;
    /// `source.dims()[i]` must equal `end - start`, and `end` must
    /// not exceed `self.dims()[i]`. Dtypes must match.
    ///
    /// Destructive on `self`: after the write, `self`'s NodeId is no
    /// longer readable (the scheduler pins this op to run after every
    /// other reader of `self` via [`opt::derive_ordering`]). The
    /// returned tensor adopts `self`'s Storage and Layout; downstream
    /// consumers read post-write bytes through this NodeId.
    ///
    /// Phase E.3.2: introduced to back persistent KV-cache writes
    /// (`InferenceContext` + `KvCache`). Non-differentiable.
    ///
    /// **Returns `Result`**: rank/shape/range mismatches surface as a
    /// typed error.
    pub fn write_slice(
        &self,
        source: &Tensor,
        ranges: Vec<(usize, usize)>,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let dest_shape = self.shape();
        let dest_dims = dest_shape.dims();
        let src_shape = source.shape();
        let src_dims = src_shape.dims();
        let rank = dest_dims.len();
        if ranges.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice: ranges.len() ({}) must equal destination rank ({rank})",
                ranges.len(),
            )).bt());
        }
        if src_dims.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice: source rank ({}) must equal destination rank ({rank})",
                src_dims.len(),
            )).bt());
        }
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end < start {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice: ranges[{i}] = ({start}, {end}) has end < start"
                )).bt());
            }
            if end > dest_dims[i] {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice: ranges[{i}].end ({end}) > destination dim {i} ({})",
                    dest_dims[i],
                )).bt());
            }
            let slab = end - start;
            if src_dims[i] != slab {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice: source dim {i} ({}) must equal slab width ({slab}) \
                     = ranges[{i}].end - ranges[{i}].start",
                    src_dims[i],
                )).bt());
            }
        }
        if self.dtype() != source.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice: dtype mismatch — destination {:?} vs source {:?}",
                self.dtype(), source.dtype(),
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::WriteSlice { ranges, dyn_offset: None },
            inputs: vec![self.id, source.id],
            shape:  dest_shape,
            dtype,
        });
        Ok(Tensor { graph: Arc::clone(&self.graph), id })
    }

    /// Append an `Op::WriteSlice` whose start on `dyn_axis` is a runtime
    /// value resolved through the per-pass [`fuel_ir::SymEnv`] at
    /// realize, rather than the build-time `ranges[dyn_axis].0`.
    ///
    /// This is the **input-determined** dynamic-offset path (a
    /// [`DynScalar`] over the one `SymEnv`), distinct from
    /// [`Self::write_slice_rotating`]'s **data-determined** position (a
    /// tensor input read mid-pass). It backs the persistent decode
    /// KV-cache write: the append offset is `cached_len`, a per-token
    /// runtime value over a fixed-capacity buffer, so the graph
    /// structure (and every shape) stays identical across tokens and the
    /// plan is reused (Phase D symbolic extents).
    ///
    /// On `dyn_axis`, `ranges[dyn_axis].0` is ignored (the start is
    /// dynamic) and `ranges[dyn_axis].1 - ranges[dyn_axis].0` is the slab
    /// width, which must equal `source.dims()[dyn_axis]` and not exceed
    /// the destination capacity on that axis. On every other axis the
    /// same shape contract as [`Self::write_slice`] applies. The runtime
    /// `offset + width` is bounds-checked against the destination
    /// capacity at realize (a typed error, never a panic).
    ///
    /// **Returns `Result`**: rank/dtype/axis-bound/width mismatches
    /// surface as a typed error at build time.
    pub fn write_slice_dyn(
        &self,
        source: &Tensor,
        ranges: Vec<(usize, usize)>,
        dyn_axis: usize,
        offset: DynScalar,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let dest_shape = self.shape();
        let dest_dims = dest_shape.dims();
        let src_shape = source.shape();
        let src_dims = src_shape.dims();
        let rank = dest_dims.len();
        if ranges.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_dyn: ranges.len() ({}) must equal destination rank ({rank})",
                ranges.len(),
            )).bt());
        }
        if src_dims.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_dyn: source rank ({}) must equal destination rank ({rank})",
                src_dims.len(),
            )).bt());
        }
        if dyn_axis >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_dyn: dyn_axis ({dyn_axis}) out of bounds for rank {rank}",
            )).bt());
        }
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end < start {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice_dyn: ranges[{i}] = ({start}, {end}) has end < start"
                )).bt());
            }
            let slab = end - start;
            if src_dims[i] != slab {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice_dyn: source dim {i} ({}) must equal slab width ({slab}) \
                     = ranges[{i}].end - ranges[{i}].start",
                    src_dims[i],
                )).bt());
            }
            if i == dyn_axis {
                // The start on `dyn_axis` is dynamic, so `ranges[i].0`/`.1`
                // are not bounded by `dest_dims[i]` here — only the slab
                // WIDTH must fit the capacity. The runtime `offset + width`
                // is bounds-checked against `dest_dims[i]` at realize.
                if slab > dest_dims[i] {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_dyn: dynamic-axis slab width ({slab}) > destination capacity \
                         dim {i} ({})",
                        dest_dims[i],
                    )).bt());
                }
            } else if end > dest_dims[i] {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice_dyn: ranges[{i}].end ({end}) > destination dim {i} ({})",
                    dest_dims[i],
                )).bt());
            }
        }
        if self.dtype() != source.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_dyn: dtype mismatch — destination {:?} vs source {:?}",
                self.dtype(), source.dtype(),
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::WriteSlice { ranges, dyn_offset: Some((dyn_axis, offset)) },
            inputs: vec![self.id, source.id],
            shape:  dest_shape,
            dtype,
        });
        Ok(Tensor { graph: Arc::clone(&self.graph), id })
    }

    /// Append an [`Op::WriteSliceRotating`] node — copies `source`'s
    /// bytes into `self` at the rectangular slab defined by `ranges`,
    /// with the `axis` axis wrapping modulo `modulus`. The dynamic
    /// write position comes through `position` (a rank-0 `U32`
    /// tensor); the kernel computes `start = position % modulus` and
    /// splits the write across the ring boundary if needed.
    ///
    /// `ranges[axis].0` is ignored (the rotating-axis start is
    /// dynamic). `ranges[axis].1 - ranges[axis].0` is the write
    /// length on the rotating axis and must equal `source.dims()[axis]`.
    /// On every other axis the same shape contract as `write_slice`
    /// applies.
    ///
    /// Destructive on `self`; scheduled like `write_slice`. Backward
    /// panics — sliding-window KV-cache writes are forward-only.
    ///
    /// **Returns `Result`**: rank/dtype/axis-bound/modulus/range
    /// mismatches surface as a typed error at build time.
    pub fn write_slice_rotating(
        &self,
        source: &Tensor,
        position: &Tensor,
        axis: usize,
        modulus: usize,
        ranges: Vec<(usize, usize)>,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let dest_shape = self.shape();
        let dest_dims = dest_shape.dims();
        let src_shape = source.shape();
        let src_dims = src_shape.dims();
        let rank = dest_dims.len();
        if ranges.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: ranges.len() ({}) must equal destination rank ({rank})",
                ranges.len(),
            )).bt());
        }
        if src_dims.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: source rank ({}) must equal destination rank ({rank})",
                src_dims.len(),
            )).bt());
        }
        if axis >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: axis {axis} out of bounds for rank {rank}",
            )).bt());
        }
        if modulus == 0 {
            return Err(fuel_ir::Error::Msg(
                "write_slice_rotating: modulus must be >= 1".into(),
            ).bt());
        }
        if modulus > dest_dims[axis] {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: modulus ({modulus}) must not exceed destination dim {axis} ({})",
                dest_dims[axis],
            )).bt());
        }
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end < start {
                return Err(fuel_ir::Error::Msg(format!(
                    "write_slice_rotating: ranges[{i}] = ({start}, {end}) has end < start"
                )).bt());
            }
            let slab = end - start;
            if i == axis {
                if slab == 0 {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: ranges[axis={axis}] slab is 0",
                    )).bt());
                }
                if slab > modulus {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: rotating-axis write length ({slab}) > modulus ({modulus})",
                    )).bt());
                }
                if src_dims[i] != slab {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: source dim {i} ({}) must equal slab width ({slab})",
                        src_dims[i],
                    )).bt());
                }
                // ranges[axis].1 itself isn't bounded by dest_dims[axis] —
                // the kernel wraps the start, so the logical end is meaningless.
                // We still require .1 <= modulus to make the slab description
                // self-consistent inside the window.
                if end > modulus {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: ranges[axis={axis}].end ({end}) > modulus ({modulus})",
                    )).bt());
                }
            } else {
                if end > dest_dims[i] {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: ranges[{i}].end ({end}) > destination dim {i} ({})",
                        dest_dims[i],
                    )).bt());
                }
                if src_dims[i] != slab {
                    return Err(fuel_ir::Error::Msg(format!(
                        "write_slice_rotating: source dim {i} ({}) must equal slab width ({slab})",
                        src_dims[i],
                    )).bt());
                }
            }
        }
        if self.dtype() != source.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: dtype mismatch — destination {:?} vs source {:?}",
                self.dtype(), source.dtype(),
            )).bt());
        }
        if position.dtype() != DType::U32 {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: position must be U32, got {:?}",
                position.dtype(),
            )).bt());
        }
        if !position.shape().dims().is_empty() {
            return Err(fuel_ir::Error::Msg(format!(
                "write_slice_rotating: position must be rank-0 scalar, got {:?}",
                position.shape().dims(),
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::WriteSliceRotating { axis, modulus, ranges },
            inputs: vec![self.id, source.id, position.id],
            shape:  dest_shape,
            dtype,
        });
        Ok(Tensor { graph: Arc::clone(&self.graph), id })
    }

    /// Build a `Const` tensor from an `f32` slice and shape on a fresh
    /// graph. The new graph is returned via the tensor's `graph` handle.
    ///
    /// `data` takes `impl Into<Arc<[f32]>>` so both `Vec<f32>` (one-time
    /// conversion at the call site) and `Arc<[f32]>` (free clone that
    /// shares buffers across forward passes) work.
    ///
    /// Phase 7.5 work item G2: `device` is the device on which the
    /// realized Storage is allocated. The graph's storage_map slot is
    /// populated with that Storage and `Op::Const` is emitted —
    /// no host-side `ConstData` payload rides on the node.
    pub fn from_f32(
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f32]> = data.into();
        let buf = fuel_ir::HostBuffer::F32(v.to_vec());
        Self::from_host_buffer(buf, DType::F32, shape, device)
    }

    /// Build a `Const` tensor from an `f64` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_f64(
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f64]> = data.into();
        let buf = fuel_ir::HostBuffer::F64(v.to_vec());
        Self::from_host_buffer(buf, DType::F64, shape, device)
    }

    /// Build a `Const` tensor from a `bf16` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_bf16(
        data: impl Into<Arc<[bf16]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[bf16]> = data.into();
        let buf = fuel_ir::HostBuffer::BF16(v.to_vec());
        Self::from_host_buffer(buf, DType::BF16, shape, device)
    }

    /// Build a `Const` tensor from an `f16` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_f16(
        data: impl Into<Arc<[f16]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f16]> = data.into();
        let buf = fuel_ir::HostBuffer::F16(v.to_vec());
        Self::from_host_buffer(buf, DType::F16, shape, device)
    }

    /// Build a `Const` tensor from a `u32` slice and shape on a fresh
    /// graph. Primarily used to construct index tensors for gather /
    /// scatter / index_select. `u32` is the index type all real fuel
    /// backends use (Candle CPU, CUDA, Metal), so keeping the reference
    /// on the same type means oracle-equivalence tests do not need any
    /// index-type translation. `device` selects where the realized
    /// Storage is allocated.
    pub fn from_u32(
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[u32]> = data.into();
        let buf = fuel_ir::HostBuffer::U32(v.to_vec());
        Self::from_host_buffer(buf, DType::U32, shape, device)
    }

    /// G2 internal funnel: allocate Storage on `device` from `buf`,
    /// register the slot, emit `Op::Const`. Per-dtype `from_*` methods
    /// delegate here.
    fn from_host_buffer(
        buf: fuel_ir::HostBuffer,
        dtype: DType,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_backend_contract::DynBackendDevice>,
    ) -> Self {
        let shape = shape.into();
        let n = host_buffer_elem_count(&buf);
        assert_eq!(
            n,
            shape.elem_count(),
            "Tensor::from_*: data length {n} does not match shape element count {}",
            shape.elem_count(),
        );
        let backend_storage = device
            .storage_from_host_buffer_owned_dyn(buf)
            .expect("Tensor::from_*: device.storage_from_host_buffer_owned_dyn failed");
        let storage_arc = Arc::new(RwLock::new(Storage::from_dyn(backend_storage)));
        let graph = Arc::new(RwLock::new(Graph::new()));
        let id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape,
                dtype,
            });
            g.set_storage(id, storage_arc);
            id
        };
        Self { graph, id }
    }

    /// Phase 7.5 work item G2: build a `Const` leaf on a fresh graph
    /// whose realized bytes already live in `storage`. The graph's
    /// storage_map slot for the new node is populated with `storage`
    /// directly. The executor's slot-first dispatch returns the slot's
    /// Arc on realize.
    pub fn from_storage(
        storage: Arc<RwLock<Storage>>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Self {
        let shape = shape.into();
        // Sanity-check the slot's dtype matches the declared dtype.
        // The slot's bytes don't carry a logical shape — that's the
        // node's job — so we don't validate elem_count here.
        debug_assert_eq!(
            storage.read().unwrap().dtype(),
            dtype,
            "Tensor::from_storage: declared dtype {:?} does not match storage dtype {:?}",
            dtype,
            storage.read().unwrap().dtype(),
        );
        let graph = Arc::new(RwLock::new(Graph::new()));
        let id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape,
                dtype,
            });
            g.set_storage(id, storage);
            id
        };
        Self { graph, id }
    }

    /// Build a second `Const` tensor that lives on the same graph as
    /// `self`. Use this to add more inputs to an existing computation.
    ///
    /// Phase 7.5 G2: the realized Storage is allocated on the device
    /// derived from `self`'s graph (any existing slot's device — the
    /// graph always has at least one slot-bearing leaf by the time
    /// const_*_like is called). For cross-device const construction,
    /// build a fresh graph with [`Tensor::from_f32`] and link via
    /// [`Op::Move`] / [`Op::Copy`].
    pub fn const_f32_like(
        &self,
        data: impl Into<Arc<[f32]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[f32]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::F32(v.to_vec()), DType::F32, shape,
        )
    }

    /// Build a second `Const f64` tensor on the same graph as `self`.
    pub fn const_f64_like(
        &self,
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[f64]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::F64(v.to_vec()), DType::F64, shape,
        )
    }

    /// Build a second `Const bf16` tensor on the same graph as `self`.
    pub fn const_bf16_like(
        &self,
        data: impl Into<Arc<[bf16]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[bf16]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::BF16(v.to_vec()), DType::BF16, shape,
        )
    }

    /// Build a second `Const f16` tensor on the same graph as `self`.
    pub fn const_f16_like(
        &self,
        data: impl Into<Arc<[f16]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[f16]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::F16(v.to_vec()), DType::F16, shape,
        )
    }

    /// Build a second `Const u32` (index) tensor on the same graph as `self`.
    pub fn const_u32_like(
        &self,
        data: impl Into<Arc<[u32]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[u32]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::U32(v.to_vec()), DType::U32, shape,
        )
    }

    /// Build a sibling U8 `Const` on the same graph. Used by
    /// byte-stream inputs (e.g. NF4 quantized weights packed two
    /// codes per byte).
    pub fn const_u8_like(
        &self,
        data: impl Into<Arc<[u8]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[u8]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::U8(v.to_vec()), DType::U8, shape,
        )
    }

    /// Build a sibling I64 `Const` on the same graph. Used by
    /// integer-target ops (e.g. cross-entropy with class indices in
    /// PyTorch convention).
    pub fn const_i64_like(
        &self,
        data: impl Into<Arc<[i64]>>,
        shape: impl Into<Shape>,
    ) -> Self {
        let v: Arc<[i64]> = data.into();
        self.const_like_host_buffer(
            fuel_ir::HostBuffer::I64(v.to_vec()), DType::I64, shape,
        )
    }

    /// G2 internal funnel for the per-graph const_*_like family. The
    /// device is derived from `self`'s graph slot (any existing one)
    /// so callers don't have to thread a device through hot paths
    /// like RoPE table construction or LoRA application — the const
    /// goes on the same device as the graph it joins.
    fn const_like_host_buffer(
        &self,
        buf: fuel_ir::HostBuffer,
        dtype: DType,
        shape: impl Into<Shape>,
    ) -> Self {
        let shape = shape.into();
        let n = host_buffer_elem_count(&buf);
        assert_eq!(
            n,
            shape.elem_count(),
            "const_*_like: data length {n} does not match shape element count {}",
            shape.elem_count(),
        );
        let device = pick_device_from_graph(&self.graph);
        let backend_storage = device
            .storage_from_host_buffer_owned_dyn(buf)
            .expect("Tensor::const_like: device.storage_from_host_buffer_owned_dyn failed");
        let storage_arc = Arc::new(RwLock::new(Storage::from_dyn(backend_storage)));
        let id = {
            let mut g = self.graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape,
                dtype,
            });
            g.set_storage(id, storage_arc);
            id
        };
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Push a `Const` node on the same graph as `self` **without**
    /// populating the graph's storage_map. The caller is responsible
    /// for binding the storage Arc into the realize call's initial
    /// StorageCache, typically via
    /// [`InferenceContext::insert`](../fuel_core/inference_context/struct.InferenceContext.html#method.insert).
    ///
    /// Used by the Phase 7.6 step 9c E.3.3 forward path to bind
    /// pre-allocated KV-cache storage Arcs (held as the new
    /// `Arc<RwLock<fuel_memory::Storage>>` type, not the legacy
    /// `fuel_backend_contract::Storage` that `const_like_from_storage` takes)
    /// into a per-step graph without re-uploading or type-converting.
    ///
    /// **Caller contract**: the same NodeId must appear in the
    /// `initial` StorageCache passed to the realize call. If not, the
    /// executor surfaces a clean error pointing at the missing slot
    /// — there's no implicit fallback to `graph.storage_map`.
    pub fn const_placeholder_like(
        &self,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Self {
        let shape = shape.into();
        let id = {
            let mut g = self.graph.write().unwrap();
            g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape,
                dtype,
            })
        };
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Phase 7.5 work item G2: build a second `Const` leaf on the same
    /// graph as `self` whose realized bytes already live in `storage`.
    /// Companion to [`Tensor::from_storage`] for the multi-input case.
    pub fn const_like_from_storage(
        &self,
        storage: Arc<RwLock<Storage>>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Self {
        let shape = shape.into();
        debug_assert_eq!(
            storage.read().unwrap().dtype(),
            dtype,
            "Tensor::const_like_from_storage: declared dtype {:?} does not match storage dtype {:?}",
            dtype,
            storage.read().unwrap().dtype(),
        );
        let id = {
            let mut g = self.graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape,
                dtype,
            });
            g.set_storage(id, storage);
            id
        };
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `Add` node `self + other` to the shared graph and return
    /// a handle to the result. Requires matching shapes and matching
    /// graphs.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary_op("add", Op::Add, other, self.shape())
    }

    /// Append a `Mul` node `self * other`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.binary_op("mul", Op::Mul, other, self.shape())
    }

    /// Append a `MatMul` node. Both operands must have rank ≥ 2. The
    /// last two dims are the matrix dims; leading dims are batch dims.
    ///
    /// - If both operands have the same rank, their batch dims must
    ///   match exactly.
    /// - If one operand is rank-2 and the other is higher rank, the
    ///   rank-2 operand is auto-broadcast to match the higher-rank
    ///   operand's batch prefix. This is the common "linear layer
    ///   across a batch" pattern (`[batch, seq, d_model] @ [d_model, d_out]`).
    /// - Any other rank combination panics; users must reshape first.
    ///
    /// Shape examples:
    /// - `[m, k] @ [k, n]` → `[m, n]`
    /// - `[batch, m, k] @ [batch, k, n]` → `[batch, m, n]`
    /// - `[batch, seq, k] @ [k, n]` → `[batch, seq, n]` (rhs auto-broadcast)
    /// - `[k, n]` @ `[batch, n, m]` → `[batch, k, m]` (lhs auto-broadcast)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &other.graph),
            "matmul: tensors must live on the same graph",
        );
        // Mixed-precision matmul: activations stay in their native
        // dtype while weights can live as a lower-precision type on
        // device. Currently supported: (F32 × BF16) → F32 for bf16-
        // quantized LLM serving. Other heterogeneous combinations
        // are rejected.
        let dtypes_ok = self.dtype() == other.dtype()
            || (self.dtype() == DType::F32 && other.dtype() == DType::BF16);
        assert!(
            dtypes_ok,
            "matmul: unsupported dtype combination: lhs={:?}, rhs={:?}",
            self.dtype(),
            other.dtype(),
        );
        let l = self.shape();
        let r = other.shape();
        let ld = l.dims();
        let rd = r.dims();
        assert!(
            ld.len() >= 2 && rd.len() >= 2,
            "matmul: both operands must be rank ≥ 2, got {ld:?} and {rd:?}",
        );

        // Auto-broadcast the rank-2 operand against the other's batch
        // prefix. If ranks already match, this is a no-op.
        let (lhs, rhs) = if ld.len() == rd.len() {
            (self.clone(), other.clone())
        } else if ld.len() == 2 && rd.len() > 2 {
            let mut bc_shape: Vec<usize> = rd[..rd.len() - 2].to_vec();
            bc_shape.push(ld[0]);
            bc_shape.push(ld[1]);
            (self.broadcast_to(Shape::from_dims(&bc_shape)), other.clone())
        } else if rd.len() == 2 && ld.len() > 2 {
            let mut bc_shape: Vec<usize> = ld[..ld.len() - 2].to_vec();
            bc_shape.push(rd[0]);
            bc_shape.push(rd[1]);
            (self.clone(), other.broadcast_to(Shape::from_dims(&bc_shape)))
        } else {
            panic!(
                "matmul: unsupported rank combination {} vs {} — only same-rank or (rank-2 × higher-rank) is supported; reshape first",
                ld.len(),
                rd.len(),
            );
        };

        // Now both operands have the same rank. Validate batch prefix
        // and build the output node.
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let l = lhs_shape.dims();
        let r = rhs_shape.dims();
        let rank = l.len();
        let batch_rank = rank - 2;
        for i in 0..batch_rank {
            // GQA-style matmul: A may have more batch heads than B
            // (e.g. Q[1,32,1,64] @ K^T[1,4,64,S]). The n_rep factor
            // is inferred by the backend at dispatch time; at the
            // graph level we just validate divisibility.
            let (la, ra) = (l[i], r[i]);
            let ok = la == ra || (la > ra && ra > 0 && la % ra == 0);
            assert!(
                ok,
                "matmul: batch dim mismatch at axis {i}: {la} vs {ra} (not equal and not a GQA-divisible pair)",
            );
        }
        let m = l[rank - 2];
        let k = l[rank - 1];
        let k2 = r[rank - 2];
        let n = r[rank - 1];
        assert_eq!(
            k, k2,
            "matmul: inner dim mismatch (lhs k={k}, rhs k={k2})",
        );
        let mut out_dims: Vec<usize> = l[..batch_rank].to_vec();
        out_dims.push(m);
        out_dims.push(n);
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::MatMul,
            inputs: vec![lhs.id, rhs.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::QMatMul`] node that multiplies `self` (F32
    /// activations, shape `[..., M, K]`) with a Q-type quantized
    /// weight matrix of logical shape `[N, K]` stored as a raw byte
    /// stream (passed in as a U32 tensor; length = n_bytes / 4).
    ///
    /// Output shape: `[..., M, N]` F32. The backend dequantizes the
    /// weight blocks on the fly inside the matmul kernel, so the
    /// quantized blocks stay resident at their compressed size.
    pub fn qmatmul(
        &self,
        weight_bytes: &Tensor,
        quant_type: QuantType,
        k: usize,
        n: usize,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &weight_bytes.graph),
            "qmatmul: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            DType::F32,
            "qmatmul: activations must be F32, got {:?}",
            self.dtype(),
        );
        assert_eq!(
            weight_bytes.dtype(),
            DType::U32,
            "qmatmul: weight_bytes must be U32 (raw block bytes reinterpreted), got {:?}",
            weight_bytes.dtype(),
        );
        let a_dims = self.shape();
        let a_dims = a_dims.dims();
        assert!(
            a_dims.len() >= 2,
            "qmatmul: activations must be rank ≥ 2, got {a_dims:?}",
        );
        assert_eq!(
            a_dims[a_dims.len() - 1], k,
            "qmatmul: last dim of activations ({}) must equal k ({k})",
            a_dims[a_dims.len() - 1],
        );
        assert_eq!(
            k % quant_type.elements_per_block(),
            0,
            "qmatmul: k={k} must be a multiple of {quant_type:?}'s block size ({})",
            quant_type.elements_per_block(),
        );
        // Validate the weight byte count matches [N, K/block_size] blocks.
        let expected_bytes = n * (k / quant_type.elements_per_block()) * quant_type.bytes_per_block();
        let expected_u32_elems = expected_bytes / 4;
        assert_eq!(
            weight_bytes.shape().elem_count(), expected_u32_elems,
            "qmatmul: weight_bytes has {} u32 elements, expected {expected_u32_elems} for N={n}, K={k}, {quant_type:?}",
            weight_bytes.shape().elem_count(),
        );
        let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
        out_dims.push(n);
        // Phase 7.6 step 4 (final): emits Op::Fused(QMATMUL, _) per
        // the registry split.
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::QMATMUL,
                crate::registry::FusedOpParams::QMatMul { quant_type, k, n },
            ),
            inputs: vec![self.id, weight_bytes.id],
            shape:  Shape::from_dims(&out_dims),
            dtype:  DType::F32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::Conv2D`] node. `self` must be `[N, Cin, H, W]`
    /// (rank 4); `weight` must be `[Cout, Cin/groups, Kh, Kw]` (rank
    /// 4) and live on the same graph. `bias` is optional — when
    /// present it must be rank 1 with length `Cout`. Returns a rank-4
    /// tensor of shape `[N, Cout, Hout, Wout]`.
    ///
    /// Panics if the input ranks don't match, the channel counts are
    /// inconsistent with `groups`, or the output spatial dims would
    /// be non-positive.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &weight.graph),
            "conv2d: x and weight must live on the same graph",
        );
        if let Some(b) = bias {
            assert!(
                Arc::ptr_eq(&self.graph, &b.graph),
                "conv2d: bias must live on the same graph",
            );
        }
        assert!(groups >= 1, "conv2d: groups must be ≥ 1, got {groups}");
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let w_dims = weight.shape();
        let w_dims = w_dims.dims();
        assert_eq!(
            x_dims.len(), 4,
            "conv2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 4,
            "conv2d: weight must be rank 4 [Cout, Cin/groups, Kh, Kw], got {w_dims:?}",
        );
        let (n, cin, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let (cout, cin_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        assert_eq!(
            cin, cin_per_g * groups,
            "conv2d: x has {cin} in-channels but weight expects {} ({}·{groups})",
            cin_per_g * groups, cin_per_g,
        );
        assert_eq!(
            cout % groups, 0,
            "conv2d: Cout={cout} must be divisible by groups={groups}",
        );
        if let Some(b) = bias {
            let b_dims = b.shape();
            let b_dims = b_dims.dims();
            assert_eq!(
                b_dims, &[cout],
                "conv2d: bias shape {b_dims:?} must match [Cout={cout}]",
            );
        }
        // Hout = (H + 2·pad.0 − Kh) / stride.0 + 1
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        assert!(stride_h >= 1 && stride_w >= 1, "conv2d: stride must be ≥ 1");
        let h_padded = h_in + 2 * pad_h;
        let w_padded = w_in + 2 * pad_w;
        assert!(
            h_padded >= kh && w_padded >= kw,
            "conv2d: padded input ({h_padded}×{w_padded}) smaller than kernel ({kh}×{kw})",
        );
        let h_out = (h_padded - kh) / stride_h + 1;
        let w_out = (w_padded - kw) / stride_w + 1;
        let dtype = self.dtype();
        let mut inputs = vec![self.id, weight.id];
        if let Some(b) = bias {
            inputs.push(b.id);
        }
        // Phase 7.6 step 4 (continued): emit
        // `Op::Fused(FusedOps::CONV2D, FusedOpParams::Conv2D { … })`
        // through the registry-extended arm. The legacy
        // `Op::Conv2D { … }` variant remains in the enum during
        // migration (Conv2D's backward still constructs inner
        // `Op::Fused(CONV2D, _)` nodes; the legacy variant is reached
        // only by direct tests until step 5 drops it).
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::CONV2D,
                crate::registry::FusedOpParams::Conv2D { stride, padding, groups },
            ),
            inputs,
            shape: Shape::from_dims(&[n, cout, h_out, w_out]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::PagedAttn`] node. `self` is the Q tensor of shape
    /// `[B, Hq, Sq, D]`; `k_cache` and `v_cache` are the paged caches
    /// shaped `[num_blocks, block_size, Hkv, D]`; `block_table` is
    /// `[B, max_num_blocks_per_seq]` (u32) mapping logical → physical
    /// blocks; `context_lens` is `[B]` (u32) of per-sequence lengths;
    /// `alibi_slopes` is the optional `[Hq]` per-head bias.
    ///
    /// Returns a tensor with `q`'s shape `[B, Hq, Sq, D]`.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_attn(
        &self,
        k_cache:      &Tensor,
        v_cache:      &Tensor,
        block_table:  &Tensor,
        context_lens: &Tensor,
        alibi_slopes: Option<&Tensor>,
        softmax_scale: f32,
        block_size:    usize,
        softcap:       Option<f32>,
    ) -> Tensor {
        let g = &self.graph;
        assert!(Arc::ptr_eq(g, &k_cache.graph), "paged_attn: q + k_cache must share graph");
        assert!(Arc::ptr_eq(g, &v_cache.graph), "paged_attn: q + v_cache must share graph");
        assert!(Arc::ptr_eq(g, &block_table.graph), "paged_attn: q + block_table must share graph");
        assert!(Arc::ptr_eq(g, &context_lens.graph), "paged_attn: q + context_lens must share graph");
        if let Some(a) = alibi_slopes { assert!(Arc::ptr_eq(g, &a.graph), "paged_attn: alibi_slopes must share graph"); }
        assert!(block_size >= 1, "paged_attn: block_size must be ≥ 1");

        let q_dims = self.shape();
        let q_dims = q_dims.dims();
        let kc_dims = k_cache.shape();
        let kc_dims = kc_dims.dims();
        let vc_dims = v_cache.shape();
        let vc_dims = vc_dims.dims();
        let bt_dims = block_table.shape();
        let bt_dims = bt_dims.dims();
        let cl_dims = context_lens.shape();
        let cl_dims = cl_dims.dims();
        assert_eq!(q_dims.len(), 4, "paged_attn: q must be rank 4 [B, Hq, Sq, D], got {q_dims:?}");
        assert_eq!(kc_dims.len(), 4, "paged_attn: k_cache must be rank 4 [num_blocks, block_size, Hkv, D], got {kc_dims:?}");
        assert_eq!(vc_dims.len(), 4, "paged_attn: v_cache must be rank 4 [num_blocks, block_size, Hkv, D], got {vc_dims:?}");
        assert_eq!(bt_dims.len(), 2, "paged_attn: block_table must be rank 2 [B, max_blocks], got {bt_dims:?}");
        assert_eq!(cl_dims.len(), 1, "paged_attn: context_lens must be rank 1 [B], got {cl_dims:?}");
        let (b, hq, _sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        assert_eq!(kc_dims[1], block_size, "paged_attn: k_cache block dim {} != block_size {block_size}", kc_dims[1]);
        assert_eq!(vc_dims[1], block_size, "paged_attn: v_cache block dim {} != block_size {block_size}", vc_dims[1]);
        let hkv = kc_dims[2];
        assert_eq!(vc_dims[2], hkv, "paged_attn: Hkv mismatch k_cache vs v_cache");
        assert_eq!(kc_dims[3], d, "paged_attn: D mismatch q vs k_cache");
        assert_eq!(vc_dims[3], d, "paged_attn: D mismatch q vs v_cache");
        assert_eq!(hq % hkv, 0, "paged_attn: Hq={hq} must be a multiple of Hkv={hkv}");
        assert_eq!(bt_dims[0], b, "paged_attn: block_table batch dim {} != B={b}", bt_dims[0]);
        assert_eq!(cl_dims[0], b, "paged_attn: context_lens len {} != B={b}", cl_dims[0]);
        assert_eq!(block_table.dtype(), crate::DType::U32, "paged_attn: block_table must be U32");
        assert_eq!(context_lens.dtype(), crate::DType::U32, "paged_attn: context_lens must be U32");
        if let Some(a) = alibi_slopes {
            let ad = a.shape();
            let ad = ad.dims();
            assert_eq!(ad, &[hq], "paged_attn: alibi_slopes must be [Hq={hq}], got {ad:?}");
        }

        let dtype = self.dtype();
        let mut inputs = vec![self.id, k_cache.id, v_cache.id, block_table.id, context_lens.id];
        if let Some(a) = alibi_slopes { inputs.push(a.id); }
        // Phase 7.6 step 4 (final): emits Op::Fused(PAGED_ATTN, _).
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::PAGED_ATTN,
                crate::registry::FusedOpParams::PagedAttn { softmax_scale, block_size, softcap },
            ),
            inputs,
            shape: Shape::from_dims(q_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::ConvTranspose2D`] node. `self` must be
    /// `[N, Cin, H, W]`; `weight` must be `[Cin, Cout/groups, Kh, Kw]`
    /// (note transposed channel order vs `conv2d`). Returns a rank-4
    /// tensor `[N, Cout, Hout, Wout]`.
    ///
    /// Panics if ranks don't match, channel counts are inconsistent
    /// with `groups`, or output spatial dims would be non-positive.
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &weight.graph),
            "conv_transpose2d: x and weight must live on the same graph",
        );
        assert!(groups >= 1, "conv_transpose2d: groups must be ≥ 1, got {groups}");
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let w_dims = weight.shape();
        let w_dims = w_dims.dims();
        assert_eq!(
            x_dims.len(), 4,
            "conv_transpose2d: x must be rank 4 [N, Cin, H, W], got {x_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 4,
            "conv_transpose2d: weight must be rank 4 [Cin, Cout/groups, Kh, Kw], got {w_dims:?}",
        );
        let (n, cin, h_in, w_in) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
        let (cin_w, cout_per_g, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
        assert_eq!(
            cin, cin_w,
            "conv_transpose2d: x has {cin} in-channels but weight has {cin_w}",
        );
        assert_eq!(
            cin % groups, 0,
            "conv_transpose2d: Cin={cin} must be divisible by groups={groups}",
        );
        let cout = cout_per_g * groups;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let (out_pad_h, out_pad_w) = output_padding;
        let (dil_h, dil_w) = dilation;
        assert!(stride_h >= 1 && stride_w >= 1, "conv_transpose2d: stride must be ≥ 1");
        assert!(dil_h >= 1 && dil_w >= 1, "conv_transpose2d: dilation must be ≥ 1");
        // Hout = (Hin − 1)·stride − 2·pad + dil·(K − 1) + out_pad + 1
        let h_out = (h_in.saturating_sub(1)) * stride_h
            + dil_h * (kh - 1) + out_pad_h + 1;
        let w_out = (w_in.saturating_sub(1)) * stride_w
            + dil_w * (kw - 1) + out_pad_w + 1;
        assert!(
            h_out > 2 * pad_h && w_out > 2 * pad_w,
            "conv_transpose2d: padding ({pad_h}×{pad_w}) is larger than the produced output dims",
        );
        let h_out = h_out - 2 * pad_h;
        let w_out = w_out - 2 * pad_w;
        let dtype = self.dtype();
        // Phase 7.6 step 4 (final): emits Op::Fused(CONV_TRANSPOSE2D, _).
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::CONV_TRANSPOSE2D,
                crate::registry::FusedOpParams::ConvTranspose2D {
                    stride, padding, output_padding, dilation, groups,
                },
            ),
            inputs: vec![self.id, weight.id],
            shape: Shape::from_dims(&[n, cout, h_out, w_out]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a transposed 1D convolution by lifting `self` and
    /// `weight` into the rank-4 path and dispatching through
    /// [`Self::conv_transpose2d`]. `self` is `[N, Cin, Lin]`;
    /// `weight` is `[Cin, Cout/groups, K]` (transposed channel
    /// order, matches PyTorch). Returns `[N, Cout, Lout]` where
    /// `Lout = (Lin − 1)·stride − 2·pad + dil·(K − 1) + out_pad + 1`.
    ///
    /// Used by audio codec models (DAC / EnCodec / SNAC / Mimi /
    /// Parler-TTS / MetaVoice / CSM) where the decoder upsamples
    /// quantized latents back to waveform with strided transposed
    /// convs.
    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Tensor {
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let w_dims = weight.shape();
        let w_dims = w_dims.dims();
        assert_eq!(
            x_dims.len(), 3,
            "conv_transpose1d: x must be rank 3 [N, Cin, Lin], got {x_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 3,
            "conv_transpose1d: weight must be rank 3 [Cin, Cout/groups, K], got {w_dims:?}",
        );
        let (n, cin, l_in) = (x_dims[0], x_dims[1], x_dims[2]);
        let (cin_w, cout_per_g, k) = (w_dims[0], w_dims[1], w_dims[2]);
        let x4 = self.reshape(Shape::from_dims(&[n, cin, 1, l_in]));
        let w4 = weight.reshape(Shape::from_dims(&[cin_w, cout_per_g, 1, k]));
        let y4 = x4.conv_transpose2d(
            &w4,
            (1, stride),
            (0, padding),
            (0, output_padding),
            (1, dilation),
            groups,
        );
        let y_dims = y4.shape();
        let y_dims = y_dims.dims();
        let (n_o, cout, _h_one, l_out) = (y_dims[0], y_dims[1], y_dims[2], y_dims[3]);
        y4.reshape(Shape::from_dims(&[n_o, cout, l_out]))
    }

    /// Append a [`Op::FlashAttn`] node. `self` is `q` of shape
    /// `[B, Hq, Sq, D]`; `k` and `v` are `[B, Hkv, Sk, D]` with
    /// `Hq` a multiple of `Hkv` (GQA). `alibi_slopes` (optional) is
    /// `[Hq]`. Returns a tensor with `q`'s shape.
    pub fn flash_attn(
        &self,
        k: &Tensor,
        v: &Tensor,
        alibi_slopes: Option<&Tensor>,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
    ) -> Tensor {
        assert!(Arc::ptr_eq(&self.graph, &k.graph), "flash_attn: q + k must live on the same graph");
        assert!(Arc::ptr_eq(&self.graph, &v.graph), "flash_attn: q + v must live on the same graph");
        if let Some(a) = alibi_slopes {
            assert!(Arc::ptr_eq(&self.graph, &a.graph), "flash_attn: alibi_slopes must live on the same graph");
        }
        let q_dims = self.shape();
        let q_dims = q_dims.dims();
        let k_dims = k.shape();
        let k_dims = k_dims.dims();
        let v_dims = v.shape();
        let v_dims = v_dims.dims();
        assert_eq!(q_dims.len(), 4, "flash_attn: q must be rank 4 [B, Hq, Sq, D], got {q_dims:?}");
        assert_eq!(k_dims.len(), 4, "flash_attn: k must be rank 4 [B, Hkv, Sk, D], got {k_dims:?}");
        assert_eq!(v_dims.len(), 4, "flash_attn: v must be rank 4 [B, Hkv, Sk, D], got {v_dims:?}");
        let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (bk, hkv, sk, dk) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        let (bv, hkv_v, sk_v, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
        assert_eq!(b, bk, "flash_attn: B mismatch q vs k ({b} vs {bk})");
        assert_eq!(b, bv, "flash_attn: B mismatch q vs v");
        assert_eq!(hkv, hkv_v, "flash_attn: Hkv mismatch k vs v");
        assert_eq!(sk, sk_v, "flash_attn: Sk mismatch k vs v");
        assert_eq!(d, dk, "flash_attn: head_dim mismatch q vs k");
        assert_eq!(d, dv, "flash_attn: head_dim mismatch q vs v");
        assert_eq!(hq % hkv, 0, "flash_attn: Hq={hq} must be a multiple of Hkv={hkv}");
        if let Some(a) = alibi_slopes {
            let ad = a.shape();
            let ad = ad.dims();
            assert_eq!(ad, &[hq], "flash_attn: alibi_slopes must be [Hq={hq}], got {ad:?}");
        }
        let dtype = self.dtype();
        let mut inputs = vec![self.id, k.id, v.id];
        if let Some(a) = alibi_slopes {
            inputs.push(a.id);
        }
        // Phase 7.6 step 4 (final): emits Op::Fused(FLASH_ATTN, _).
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::FLASH_ATTN,
                crate::registry::FusedOpParams::FlashAttn {
                    softmax_scale, causal, window_size_left, window_size_right, softcap,
                    k_len: None,
                },
            ),
            inputs,
            shape: Shape::from_dims(&[b, hq, sq, d]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a [`Op::FlashAttn`] over a fixed-**capacity** K/V whose
    /// attended length is a **runtime** value (`k_len`) resolved through
    /// the per-pass [`fuel_ir::SymEnv`] at realize, decoupled
    /// from K's allocated shape.
    ///
    /// `self` is `q` of shape `[B, Hq, Sq, D]`; `k`/`v` are the capacity
    /// buffers `[B, Hkv, max_seq, D]` (GQA: `Hq % Hkv == 0`). Only the
    /// first `k_len.resolve(env)` rows along the K/V length axis are
    /// attended; the causal mask (when `causal`) is bottom-right-aligned
    /// at offset `k_len - Sq` — the standard FlashAttention-2 convention,
    /// equal to `cached_len` in autoregressive decode. Returns a tensor
    /// with `q`'s shape.
    ///
    /// This is the flash counterpart of [`Self::write_slice_dyn`]: the
    /// decode KV-cache write appends at the runtime offset and flash
    /// attends the runtime prefix, so the graph structure stays
    /// identical across tokens (Phase D symbolic extents).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_dyn(
        &self,
        k: &Tensor,
        v: &Tensor,
        alibi_slopes: Option<&Tensor>,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
        k_len: fuel_ir::DynScalar,
    ) -> Tensor {
        assert!(Arc::ptr_eq(&self.graph, &k.graph), "flash_attn_dyn: q + k must live on the same graph");
        assert!(Arc::ptr_eq(&self.graph, &v.graph), "flash_attn_dyn: q + v must live on the same graph");
        if let Some(a) = alibi_slopes {
            assert!(Arc::ptr_eq(&self.graph, &a.graph), "flash_attn_dyn: alibi_slopes must live on the same graph");
        }
        let q_dims = self.shape();
        let q_dims = q_dims.dims();
        let k_dims = k.shape();
        let k_dims = k_dims.dims();
        let v_dims = v.shape();
        let v_dims = v_dims.dims();
        assert_eq!(q_dims.len(), 4, "flash_attn_dyn: q must be rank 4 [B, Hq, Sq, D], got {q_dims:?}");
        assert_eq!(k_dims.len(), 4, "flash_attn_dyn: k must be rank 4 [B, Hkv, max_seq, D], got {k_dims:?}");
        assert_eq!(v_dims.len(), 4, "flash_attn_dyn: v must be rank 4 [B, Hkv, max_seq, D], got {v_dims:?}");
        let (b, hq, sq, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
        let (bk, hkv, sk, dk) = (k_dims[0], k_dims[1], k_dims[2], k_dims[3]);
        let (bv, hkv_v, sk_v, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
        assert_eq!(b, bk, "flash_attn_dyn: B mismatch q vs k ({b} vs {bk})");
        assert_eq!(b, bv, "flash_attn_dyn: B mismatch q vs v");
        assert_eq!(hkv, hkv_v, "flash_attn_dyn: Hkv mismatch k vs v");
        assert_eq!(sk, sk_v, "flash_attn_dyn: capacity (max_seq) mismatch k vs v ({sk} vs {sk_v})");
        assert_eq!(d, dk, "flash_attn_dyn: head_dim mismatch q vs k");
        assert_eq!(d, dv, "flash_attn_dyn: head_dim mismatch q vs v");
        assert_eq!(hq % hkv, 0, "flash_attn_dyn: Hq={hq} must be a multiple of Hkv={hkv}");
        // A build-time constant k_len must fit the capacity and cover Sq.
        if let fuel_ir::DynScalar::Concrete(kl) = k_len {
            assert!(kl <= sk, "flash_attn_dyn: k_len ({kl}) exceeds K capacity ({sk})");
            assert!(kl >= sq, "flash_attn_dyn: k_len ({kl}) must be >= Sq ({sq}) for a valid causal prefix");
        }
        if let Some(a) = alibi_slopes {
            let ad = a.shape();
            let ad = ad.dims();
            assert_eq!(ad, &[hq], "flash_attn_dyn: alibi_slopes must be [Hq={hq}], got {ad:?}");
        }
        let dtype = self.dtype();
        let mut inputs = vec![self.id, k.id, v.id];
        if let Some(a) = alibi_slopes {
            inputs.push(a.id);
        }
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::FLASH_ATTN,
                crate::registry::FusedOpParams::FlashAttn {
                    softmax_scale, causal, window_size_left, window_size_right, softcap,
                    k_len: Some(k_len),
                },
            ),
            inputs,
            shape: Shape::from_dims(&[b, hq, sq, d]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Relu` node `max(0, self)`.
    pub fn relu(&self) -> Tensor {
        self.unary_op(Op::Relu)
    }

    /// Append a `Sqr` node `self * self`.
    pub fn sqr(&self) -> Tensor {
        self.unary_op(Op::Sqr)
    }

    /// Append an `Exp` node `e^self`.
    pub fn exp(&self) -> Tensor {
        self.unary_op(Op::Exp)
    }

    /// Append a `Permute` node rearranging the axes according to `axes`.
    /// `out.shape[i] = self.shape[axes[i]]`. The axes vector must be a
    /// permutation of `0..rank`. For a rank-3 input with `axes = [2, 0, 1]`,
    /// the shape transform is `[a, b, c] → [c, a, b]`.
    pub fn permute(&self, axes: &[usize]) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        assert_eq!(
            axes.len(),
            rank,
            "permute: axes length {} must equal tensor rank {}",
            axes.len(),
            rank,
        );
        // Validate that axes is a permutation of 0..rank.
        let mut seen = vec![false; rank];
        for &ax in axes {
            assert!(ax < rank, "permute: axis {ax} out of bounds for rank {rank}");
            assert!(!seen[ax], "permute: duplicate axis {ax} in axes");
            seen[ax] = true;
        }
        let out_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Permute(axes.to_vec()),
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::permute`]. Surfaces bad
    /// axes (wrong length, out-of-bounds entry, or duplicate) as a
    /// typed error rather than panicking.
    pub fn try_permute(&self, axes: &[usize]) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if axes.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "permute: axes length {} must equal tensor rank {}",
                axes.len(), rank,
            )).bt());
        }
        let mut seen = vec![false; rank];
        for &ax in axes {
            if ax >= rank {
                return Err(fuel_ir::Error::Msg(format!(
                    "permute: axis {ax} out of bounds for rank {rank}",
                )).bt());
            }
            if seen[ax] {
                return Err(fuel_ir::Error::Msg(format!(
                    "permute: duplicate axis {ax} in axes",
                )).bt());
            }
            seen[ax] = true;
        }
        let out_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Permute(axes.to_vec()),
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Ok(Self { graph: self.graph.clone(), id })
    }

    /// Append a `Transpose` node that swaps the last two dimensions of a
    /// tensor of rank ≥ 2. Leading dims are unchanged. For a rank-2
    /// tensor this is the ordinary matrix transpose; for higher ranks,
    /// every batch slice is transposed independently.
    pub fn transpose(&self) -> Tensor {
        let in_dims = self.shape();
        let d = in_dims.dims();
        assert!(
            d.len() >= 2,
            "transpose: input must be rank ≥ 2, got shape {d:?}",
        );
        let rank = d.len();
        let mut out: Vec<usize> = d.to_vec();
        out.swap(rank - 2, rank - 1);
        let out_shape = Shape::from_dims(&out);
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Transpose,
            inputs: vec![self.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::transpose`]. Surfaces
    /// rank < 2 as a typed error rather than panicking.
    pub fn try_transpose(&self) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_dims = self.shape();
        let d = in_dims.dims();
        if d.len() < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "transpose: input must be rank ≥ 2, got shape {d:?}",
            )).bt());
        }
        let rank = d.len();
        let mut out: Vec<usize> = d.to_vec();
        out.swap(rank - 2, rank - 1);
        let out_shape = Shape::from_dims(&out);
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Transpose,
            inputs: vec![self.id],
            shape: out_shape,
            dtype,
        });
        Ok(Self { graph: self.graph.clone(), id })
    }

    // --- additional element-wise binary ops ---

    /// Append a `Sub` node `self - other`. Requires matching shapes and dtypes.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary_op("sub", Op::Sub, other, self.shape())
    }

    /// Append a `Div` node `self / other`. Requires matching shapes and dtypes.
    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary_op("div", Op::Div, other, self.shape())
    }

    // --- additional element-wise unary ops ---

    /// Append a `Neg` node `-self`.
    pub fn neg(&self) -> Tensor {
        self.unary_op(Op::Neg)
    }

    /// Append a `Sqrt` node `sqrt(self)`.
    pub fn sqrt(&self) -> Tensor {
        self.unary_op(Op::Sqrt)
    }

    /// Append a `Log` node `ln(self)`.
    pub fn log(&self) -> Tensor {
        self.unary_op(Op::Log)
    }

    /// Append a `Sin` node.
    pub fn sin(&self) -> Tensor {
        self.unary_op(Op::Sin)
    }

    /// Append a `Cos` node.
    pub fn cos(&self) -> Tensor {
        self.unary_op(Op::Cos)
    }

    /// Append a `Tanh` node.
    pub fn tanh(&self) -> Tensor {
        self.unary_op(Op::Tanh)
    }

    /// Append a `Sigmoid` node.
    pub fn sigmoid(&self) -> Tensor {
        self.unary_op(Op::Sigmoid)
    }

    /// Append a `Silu` node (SiLU/Swish activation: `x · sigmoid(x)`).
    pub fn silu(&self) -> Tensor {
        self.unary_op(Op::Silu)
    }

    /// Append a `Gelu` node (tanh-approximation GELU).
    pub fn gelu(&self) -> Tensor {
        self.unary_op(Op::Gelu)
    }

    /// Append a `Step` node (Heaviside step: `1` where `x > 0`, else `0`).
    pub fn step(&self) -> Tensor {
        self.unary_op(Op::Step)
    }

    /// Append a `Recip` node (`1 / self`).
    pub fn recip(&self) -> Tensor {
        self.unary_op(Op::Recip)
    }

    /// Append an `Abs` node (`|self|`).
    pub fn abs(&self) -> Tensor {
        self.unary_op(Op::Abs)
    }

    // ---- In-place unary builders (Phase 2 of the in-place ops
    // infrastructure) ----
    //
    // Each emits an `Op::*Inplace` node whose input is `self`. The
    // returned `Tensor` shares the underlying graph and is the new
    // root; subsequent ops should use the returned handle to make the
    // mutation-after-read ordering observable to `derive_ordering`.
    //
    // Phase 4 (the mutation-safety pass) is what makes these safe to
    // call on tape-tracked tensors. Until Phase 4 lands, calling these
    // on a tensor that's been saved for backward will panic at
    // `Tensor::backward` time (clear error, no silent gradient
    // corruption).

    /// Append a `ReluInplace` node — mutates `self`'s storage with
    /// `max(0, self)`. See `Tensor::relu` for the functional variant.
    pub fn relu_inplace(&self) -> Tensor {
        self.unary_op(Op::ReluInplace)
    }

    /// Append a `SiluInplace` node — mutates `self`'s storage with
    /// `self * sigmoid(self)`. See `Tensor::silu` for the functional
    /// variant.
    pub fn silu_inplace(&self) -> Tensor {
        self.unary_op(Op::SiluInplace)
    }

    /// Append a `GeluInplace` node — mutates `self`'s storage with
    /// the tanh-approximation GELU. See `Tensor::gelu` for the
    /// functional variant.
    pub fn gelu_inplace(&self) -> Tensor {
        self.unary_op(Op::GeluInplace)
    }

    /// Append a `TanhInplace` node — mutates `self`'s storage with
    /// `tanh(self)`. See `Tensor::tanh` for the functional variant.
    pub fn tanh_inplace(&self) -> Tensor {
        self.unary_op(Op::TanhInplace)
    }

    /// Append a `SigmoidInplace` node — mutates `self`'s storage
    /// with `sigmoid(self)`. See `Tensor::sigmoid` for the functional
    /// variant.
    pub fn sigmoid_inplace(&self) -> Tensor {
        self.unary_op(Op::SigmoidInplace)
    }

    /// Append a `NegInplace` node — mutates `self`'s storage with
    /// `-self`. See `Tensor::neg` for the functional variant.
    pub fn neg_inplace(&self) -> Tensor {
        self.unary_op(Op::NegInplace)
    }

    /// Append an `AbsInplace` node — mutates `self`'s storage with
    /// `|self|`. See `Tensor::abs` for the functional variant.
    pub fn abs_inplace(&self) -> Tensor {
        self.unary_op(Op::AbsInplace)
    }

    /// Append a `SqrInplace` node — mutates `self`'s storage with
    /// `self²`. See `Tensor::sqr` for the functional variant.
    pub fn sqr_inplace(&self) -> Tensor {
        self.unary_op(Op::SqrInplace)
    }

    /// Append a `SqrtInplace` node — mutates `self`'s storage with
    /// `√self`. See `Tensor::sqrt` for the functional variant.
    pub fn sqrt_inplace(&self) -> Tensor {
        self.unary_op(Op::SqrtInplace)
    }

    /// Append a `RsqrtInplace` node — mutates `self`'s storage with
    /// `1/√self`. See `Tensor::rsqrt` for the functional variant.
    pub fn rsqrt_inplace(&self) -> Tensor {
        self.unary_op(Op::RsqrtInplace)
    }

    /// Append a `RecipInplace` node — mutates `self`'s storage with
    /// `1/self`. See `Tensor::recip` for the functional variant.
    pub fn recip_inplace(&self) -> Tensor {
        self.unary_op(Op::RecipInplace)
    }

    /// Append an `ExpInplace` node — mutates `self`'s storage with
    /// `exp(self)`. See `Tensor::exp` for the functional variant.
    pub fn exp_inplace(&self) -> Tensor {
        self.unary_op(Op::ExpInplace)
    }

    /// Append a `LogInplace` node — mutates `self`'s storage with
    /// `ln(self)`. See `Tensor::log` for the functional variant.
    pub fn log_inplace(&self) -> Tensor {
        self.unary_op(Op::LogInplace)
    }

    /// Append a `SinInplace` node — mutates `self`'s storage with
    /// `sin(self)`. See `Tensor::sin` for the functional variant.
    pub fn sin_inplace(&self) -> Tensor {
        self.unary_op(Op::SinInplace)
    }

    /// Append a `CosInplace` node — mutates `self`'s storage with
    /// `cos(self)`. See `Tensor::cos` for the functional variant.
    pub fn cos_inplace(&self) -> Tensor {
        self.unary_op(Op::CosInplace)
    }

    /// Append a `SignInplace` node — mutates `self`'s storage with
    /// `sign(self)`. See `Tensor::sign` for the functional variant.
    pub fn sign_inplace(&self) -> Tensor {
        self.unary_op(Op::SignInplace)
    }

    /// Append a `FloorInplace` node — mutates `self`'s storage with
    /// `⌊self⌋`. See `Tensor::floor` for the functional variant.
    pub fn floor_inplace(&self) -> Tensor {
        self.unary_op(Op::FloorInplace)
    }

    /// Append a `CeilInplace` node — mutates `self`'s storage with
    /// `⌈self⌉`. See `Tensor::ceil` for the functional variant.
    pub fn ceil_inplace(&self) -> Tensor {
        self.unary_op(Op::CeilInplace)
    }

    /// Append a `RoundInplace` node — mutates `self`'s storage with
    /// `round(self)`. See `Tensor::round` for the functional variant.
    pub fn round_inplace(&self) -> Tensor {
        self.unary_op(Op::RoundInplace)
    }

    /// Append an `ErfInplace` node — mutates `self`'s storage with
    /// `erf(self)`. See `Tensor::erf` for the functional variant.
    pub fn erf_inplace(&self) -> Tensor {
        self.unary_op(Op::ErfInplace)
    }

    /// Append a `GeluErfInplace` node — mutates `self`'s storage
    /// with the exact-GeLU formula. See `Tensor::gelu_erf` for the
    /// functional variant.
    pub fn gelu_erf_inplace(&self) -> Tensor {
        self.unary_op(Op::GeluErfInplace)
    }

    /// Append a `ClampInplace { min, max }` node — mutates `self`'s
    /// storage with `clamp(self, min, max)`. See `Tensor::clamp` for
    /// the functional variant.
    pub fn clamp_inplace(&self, min: f64, max: f64) -> Tensor {
        self.unary_op(Op::ClampInplace { min, max })
    }

    /// Append a `PowIInplace(exp)` node — mutates `self`'s storage
    /// with `self.powi(exp)`. See `Tensor::powi` for the functional
    /// variant.
    pub fn powi_inplace(&self, exp: i32) -> Tensor {
        self.unary_op(Op::PowIInplace(exp))
    }

    /// Append an `Op::Fused(INPLACE_AFFINE, ...)` node — mutates
    /// `self`'s storage with `self = mul · self + add`. Single-input
    /// fused op (no functional `Op::Affine` exists on
    /// `fuel-graph::Op`; the non-inplace equivalent is `self
    /// .mul_scalar(mul).add_scalar(add)`). Wired to baracuda's
    /// `affine_inplace_*` symbol on CUDA when Phase 3 lands.
    pub fn affine_inplace(&self, mul: f64, add: f64) -> Tensor {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::INPLACE_AFFINE,
                crate::registry::FusedOpParams::InplaceAffine { mul, add },
            ),
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Floor` node (`⌊self⌋`). Output dtype = input dtype.
    /// Backward is the zero distribution almost everywhere; gradient
    /// is dropped silently.
    pub fn floor(&self) -> Tensor {
        self.unary_op(Op::Floor)
    }

    /// Append a `Ceil` node (`⌈self⌉`). Output dtype = input dtype.
    /// Backward drops gradient (non-differentiable almost everywhere).
    pub fn ceil(&self) -> Tensor {
        self.unary_op(Op::Ceil)
    }

    /// Append a `Round` node — banker's rounding (round-half-to-even,
    /// IEEE 754 roundeven). 0.5 → 0, 1.5 → 2, 2.5 → 2, 3.5 → 4.
    /// Output dtype = input dtype. Backward drops gradient.
    pub fn round(&self) -> Tensor {
        self.unary_op(Op::Round)
    }

    /// Append a `Sign` node (`-1` / `0` / `1`). `sign(0) = 0` by
    /// subgradient convention. Output dtype = input dtype. Backward
    /// drops gradient.
    pub fn sign(&self) -> Tensor {
        self.unary_op(Op::Sign)
    }

    /// Append an `Erf` node (Gauss error function). Output dtype =
    /// input dtype. Differentiable.
    pub fn erf(&self) -> Tensor {
        self.unary_op(Op::Erf)
    }

    /// Append a `GeluErf` node — exact-erf formulation of GELU
    /// (`0.5 * x * (1 + erf(x/√2))`). Distinct from [`Self::gelu`]
    /// (tanh approximation). Output dtype = input dtype.
    /// Differentiable.
    pub fn gelu_erf(&self) -> Tensor {
        self.unary_op(Op::GeluErf)
    }

    /// Append a `Pow` node `pow(self, other)` element-wise (real
    /// exponent). Both operands must share dtype and shape. Distinct
    /// from [`Self::powi`] (scalar `i32` exponent). NaN follows
    /// IEEE-754 (`pow(-2, 0.5) = NaN`).
    ///
    /// **Returns `Result`**: dtype/shape mismatch surfaces as a
    /// typed error, not a panic.
    pub fn pow(&self, other: &Tensor) -> std::result::Result<Tensor, fuel_ir::Error> {
        let out_shape = self.shape();
        self.try_binary_op("pow", Op::Pow, other, out_shape)
    }

    /// Append an `Rsqrt` node (`1 / sqrt(self)`). Same dtype as
    /// input. Single op rather than `sqrt(x).recip()` — one kernel
    /// launch and matches RMSNorm's `x * rsqrt(...)` shape.
    /// Differentiable.
    pub fn rsqrt(&self) -> Tensor {
        self.unary_op(Op::Rsqrt)
    }

    /// Append a `Rem` node `self % other` element-wise (PyTorch
    /// convention: `a - floor(a/b) * b`, sign of result matches
    /// divisor). Both operands must share dtype and shape.
    /// Differentiable.
    ///
    /// **Returns `Result`**: dtype/shape mismatch surfaces as a
    /// typed error, not a panic.
    pub fn rem(&self, other: &Tensor) -> std::result::Result<Tensor, fuel_ir::Error> {
        let out_shape = self.shape();
        self.try_binary_op("rem", Op::Rem, other, out_shape)
    }

    /// Append a `Flip` node — reverses element order along `dim`.
    /// Output shape == input shape. Materializing op (real byte
    /// shuffle; not a metadata-only view). Differentiable
    /// (involutive: backward is another Flip on the same dim).
    ///
    /// **Returns `Result`**: bad `dim` surfaces as a typed error.
    pub fn flip(&self, dim: usize) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "flip: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Flip { dim },
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `Roll` node — cyclic shift along `dim` by `shift`
    /// positions. Positive `shift` moves elements to higher indices
    /// (wrapping); negative the opposite. Output shape == input
    /// shape. Differentiable (backward is `Roll { dim, -shift }`).
    ///
    /// **Returns `Result`**: bad `dim` surfaces as a typed error.
    pub fn roll(&self, dim: usize, shift: i64) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "roll: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Roll { dim, shift },
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `CumSum` node — running cumulative sum along `dim`.
    /// Output shape == input shape. Differentiable; backward is
    /// reverse-cumsum (`Flip → CumSum → Flip`).
    ///
    /// **Returns `Result`**: bad `dim` surfaces as a typed error.
    pub fn cumsum(&self, dim: usize) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "cumsum: dim {dim} out of bounds for rank {rank}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::CumSum { dim },
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `Triu` node — upper-triangular mask along the last two
    /// dims. `diagonal = 0` keeps the main diagonal and above; positive
    /// values shift the cutoff up (keeping less); negative shift down
    /// (keeping more, including subdiagonals).
    ///
    /// **Returns `Result`**: rank < 2 surfaces as a typed error.
    pub fn triu(&self, diagonal: i64) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "triu: input must have rank >= 2, got {rank}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Triu { diagonal },
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `Tril` node — lower-triangular mask along the last two
    /// dims. `diagonal = 0` keeps the main diagonal and below.
    /// `tril(diagonal = 0)` is the canonical causal-attention mask.
    ///
    /// **Returns `Result`**: rank < 2 surfaces as a typed error.
    pub fn tril(&self, diagonal: i64) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if rank < 2 {
            return Err(fuel_ir::Error::Msg(format!(
                "tril: input must have rank >= 2, got {rank}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Tril { diagonal },
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `LogSoftmaxLastDim` node. Output shape == input shape.
    /// Numerically stable (max-subtracting formula) — preferred over
    /// `log(softmax(x))` for NLL / cross-entropy loss.
    ///
    /// **Returns `Result`**: rank < 1 surfaces as a typed error.
    pub fn log_softmax_last_dim(&self) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        if in_shape.dims().is_empty() {
            return Err(fuel_ir::Error::Msg(
                "log_softmax_last_dim: input must have rank >= 1".to_string(),
            ).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::LogSoftmaxLastDim,
            inputs: vec![self.id],
            shape:  in_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `MaskedFill` node. `mask` must be U8 and have the same
    /// shape as `self`. Every position where `mask != 0` gets `value`;
    /// every position where `mask == 0` passes `self` through. `value`
    /// is a scalar of `self`'s dtype.
    ///
    /// **Returns `Result`**: shape or dtype mismatches surface as typed
    /// errors.
    pub fn masked_fill(
        &self,
        mask: &Tensor,
        value: Scalar,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        if self.shape().dims() != mask.shape().dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "masked_fill: x.shape={:?} != mask.shape={:?}",
                self.shape().dims(), mask.shape().dims(),
            )).bt());
        }
        if mask.dtype() != DType::U8 {
            return Err(fuel_ir::Error::Msg(format!(
                "masked_fill: mask dtype must be U8, got {:?}",
                mask.dtype(),
            )).bt());
        }
        if value.dtype() != self.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "masked_fill: value dtype {:?} != x dtype {:?}",
                value.dtype(), self.dtype(),
            )).bt());
        }
        let dtype = self.dtype();
        let shape = self.shape();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::MaskedFill { value },
            inputs: vec![self.id, mask.id],
            shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append a `Pad` node — multi-dim padding. `padding[i] = (before, after)`
    /// for axis `i`; `padding.len()` must equal `self.rank()`.
    /// Output shape: `out[i] = in[i] + padding[i].0 + padding[i].1`.
    /// `value` is the fill constant for [`PadMode::Constant`]
    /// (ignored for other modes).
    ///
    /// Only Constant mode is wired through the executor in the v1
    /// cut; the other modes produce a clean error at realize time.
    /// Differentiable for Constant (backward slices the gradient
    /// along each padded axis to restore the input shape).
    ///
    /// **Returns `Result`**: rank mismatch surfaces as a typed error.
    pub fn pad(
        &self,
        padding: Vec<(usize, usize)>,
        mode: PadMode,
        value: f64,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if padding.len() != rank {
            return Err(fuel_ir::Error::Msg(format!(
                "pad: padding.len() ({}) must equal tensor rank ({rank})",
                padding.len(),
            )).bt());
        }
        let out_dims: Vec<usize> = in_dims.iter().zip(padding.iter())
            .map(|(&d, &(b, a))| d + b + a)
            .collect();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Pad { padding, mode, value },
            inputs: vec![self.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append an `Equal` node (`self == other`) producing a `U8` mask.
    /// Both operands must share dtype and shape; output dtype is `U8`
    /// (`1` where equal, `0` otherwise). NaN follows IEEE-754
    /// (`NaN == NaN` is false). Non-differentiable.
    pub fn eq(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("eq", Op::Equal, other)
    }

    /// Append a `Ne` node (`self != other`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Self::eq`]. NaN follows
    /// IEEE-754 (`NaN != NaN` is true → `1`). Non-differentiable.
    pub fn ne(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("ne", Op::Ne, other)
    }

    /// Append an `Lt` node (`self < other`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Self::eq`]. NaN-on-either-side
    /// is always `0` (IEEE-754 unordered). Non-differentiable.
    pub fn lt(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("lt", Op::Lt, other)
    }

    /// Append an `Le` node (`self <= other`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Self::eq`]. NaN-on-either-side
    /// is always `0`. Non-differentiable.
    pub fn le(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("le", Op::Le, other)
    }

    /// Append a `Gt` node (`self > other`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Self::eq`]. NaN-on-either-side
    /// is always `0`. Non-differentiable.
    pub fn gt(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("gt", Op::Gt, other)
    }

    /// Append a `Ge` node (`self >= other`) producing a `U8` mask.
    /// Same shape/dtype contract as [`Self::eq`]. NaN-on-either-side
    /// is always `0`. Non-differentiable.
    pub fn ge(&self, other: &Tensor) -> Tensor {
        self.binary_compare_op("ge", Op::Ge, other)
    }

    /// Ternary select: `result[i] = if self[i] != 0 { a[i] } else { b[i] }`.
    /// Receiver is the `cond` mask (must be `DType::U8`); `a` and `b`
    /// must share dtype with each other and shape with `self`. Output
    /// dtype matches `a`/`b`. Returning a new tensor where each slot is
    /// picked from `a` (cond=1) or `b` (cond=0).
    ///
    /// Named `where_cond` because `where` is a Rust reserved keyword.
    /// Common spelling in Candle/PyTorch families.
    ///
    /// Differentiable through `a` and `b` only; gradient through the
    /// cond mask is `None` (registered in
    /// [`crate::grad::WhereRule`]).
    pub fn where_cond(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &a.graph) && Arc::ptr_eq(&self.graph, &b.graph),
            "where_cond: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            DType::U8,
            "where_cond: cond must be U8, got {:?}",
            self.dtype(),
        );
        assert_eq!(
            a.dtype(), b.dtype(),
            "where_cond: a/b dtype mismatch: a={:?}, b={:?}",
            a.dtype(), b.dtype(),
        );
        assert_eq!(
            self.shape().dims(), a.shape().dims(),
            "where_cond: cond/a shape mismatch: cond={:?}, a={:?}",
            self.shape().dims(), a.shape().dims(),
        );
        assert_eq!(
            self.shape().dims(), b.shape().dims(),
            "where_cond: cond/b shape mismatch: cond={:?}, b={:?}",
            self.shape().dims(), b.shape().dims(),
        );
        let shape = self.shape();
        let dtype = a.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Where,
            inputs: vec![self.id, a.id, b.id],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- dtype and broadcasting ---

    /// Append a `Cast` node converting this tensor's element type to
    /// `target`. Shape is preserved.
    pub fn cast(&self, target: DType) -> Tensor {
        let shape = self.shape();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Cast(target),
            inputs: vec![self.id],
            shape,
            dtype: target,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `BroadcastTo` node that expands `self` to the given target
    /// shape using NumPy broadcasting rules (right-align, pad with 1s,
    /// expand size-1 dims). The new tensor has the target shape and the
    /// same dtype as `self`.
    pub fn broadcast_to(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        let src_dims = self.shape();
        // Validate broadcast compatibility up front so users hear about
        // bad target shapes at build time, not realize time.
        check_broadcast_compatible(src_dims.dims(), target.dims());
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::BroadcastTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::broadcast_to`]. Surfaces
    /// shape incompatibility as a typed error rather than panicking.
    pub fn try_broadcast_to(&self, target: impl Into<Shape>) -> std::result::Result<Tensor, fuel_ir::Error> {
        let target = target.into();
        let src_dims = self.shape();
        try_check_broadcast_compatible(src_dims.dims(), target.dims())?;
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::BroadcastTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Ok(Self { graph: self.graph.clone(), id })
    }

    /// Append an `Unsqueeze` node that inserts a size-1 dimension at
    /// position `dim`. `dim` must be in `0..=rank` (where `dim == rank`
    /// appends to the end). Pure metadata — the output shares bytes
    /// with `self`, with a new size-1 axis layered into the Layout
    /// side-table. Strictly more efficient than `reshape` for the
    /// size-1-insertion case because non-contiguous (e.g. transposed
    /// or broadcast) inputs flow through without auto-Contiguize.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        assert!(
            dim <= rank,
            "unsqueeze: dim {dim} out of bounds for rank {rank} (must be <= rank)",
        );
        let mut out_dims: Vec<usize> = in_dims.to_vec();
        out_dims.insert(dim, 1);
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Unsqueeze { dim },
            inputs: vec![self.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::unsqueeze`]. Surfaces
    /// `dim > rank` as a typed error rather than panicking.
    pub fn try_unsqueeze(&self, dim: usize) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if dim > rank {
            return Err(fuel_ir::Error::Msg(format!(
                "unsqueeze: dim {dim} out of bounds for rank {rank} (must be <= rank)",
            )).bt());
        }
        let mut out_dims: Vec<usize> = in_dims.to_vec();
        out_dims.insert(dim, 1);
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Unsqueeze { dim },
            inputs: vec![self.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Ok(Self { graph: self.graph.clone(), id })
    }

    /// Append a `Squeeze` node that drops the size-1 dimension at
    /// position `dim` (range `0..rank`). Inverse of [`Self::unsqueeze`].
    /// Metadata-only view: the output shares bytes with `self`, with
    /// the named dim pruned from the Layout side-table.
    ///
    /// **Returns `Result`** rather than panicking — production paths
    /// can recover from a bad `dim` instead of crashing. Bad `dim`
    /// (out of bounds OR `shape[dim] != 1`) surfaces as a typed error.
    pub fn squeeze(&self, dim: usize) -> std::result::Result<Tensor, fuel_ir::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if dim >= rank {
            return Err(fuel_ir::Error::Msg(format!(
                "squeeze: dim {dim} out of bounds for rank {rank} (must be < rank)",
            )).bt());
        }
        if in_dims[dim] != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "squeeze: dim {dim} has size {}, expected 1",
                in_dims[dim],
            )).bt());
        }
        let out_dims: Vec<usize> = in_dims.iter().enumerate()
            .filter_map(|(i, &d)| if i == dim { None } else { Some(d) })
            .collect();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Squeeze { dim },
            inputs: vec![self.id],
            shape:  Shape::from_dims(&out_dims),
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    /// Append an `Op::Contiguize` node that materializes a
    /// contiguous copy of `self`. Output shape and dtype match the
    /// input; only the layout changes to contiguous + zero-offset.
    /// Zero-copy when the input is already contiguous + zero-offset
    /// (the executor adopts the input Storage Arc unchanged).
    ///
    /// First-class IR node so the optimizer (Phase 2.2) can insert
    /// layout-fixups before kernels that don't advertise
    /// [`crate::KernelCaps::strided_input`] without overloading
    /// [`Self::reshape`]'s "change shape" semantics.
    pub fn contiguize(&self) -> Tensor {
        let shape = self.shape().clone();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Contiguize,
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Reshape` node producing `self`'s data under a new shape.
    /// The new shape must have the same total element count as the
    /// current shape.
    pub fn reshape(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        assert_eq!(
            self.shape().elem_count(),
            target.elem_count(),
            "reshape: element count mismatch: from {} to {}",
            self.shape().elem_count(),
            target.elem_count(),
        );
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Reshape(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::reshape`]. Surfaces
    /// element-count mismatch as a typed error rather than panicking.
    pub fn try_reshape(&self, target: impl Into<Shape>) -> std::result::Result<Tensor, fuel_ir::Error> {
        let target = target.into();
        let from = self.shape().elem_count();
        let to = target.elem_count();
        if from != to {
            return Err(fuel_ir::Error::Msg(format!(
                "reshape: element count mismatch: from {from} to {to}",
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Reshape(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Ok(Self { graph: self.graph.clone(), id })
    }

    /// Append a `ReduceSumTo` node that sum-reduces `self` to a smaller
    /// shape. The target must be reachable from `self.shape()` via
    /// reduction of dims (i.e. `self.shape()` could be produced from
    /// `target` by broadcasting).
    pub fn reduce_sum_to(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        // Symmetric check: self must be broadcast-compatible FROM the
        // smaller target. That is, the target must be broadcast-ready to
        // self.shape().
        check_broadcast_compatible(target.dims(), self.shape().dims());
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::ReduceSumTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `ReduceMaxTo` node that max-reduces `self` to a smaller
    /// shape — the max-symmetric counterpart of [`Self::reduce_sum_to`].
    /// The target must be reachable from `self.shape()` via reduction
    /// of dims (i.e. `self.shape()` could be produced from `target` by
    /// broadcasting).
    pub fn reduce_max_to(&self, target: impl Into<Shape>) -> Tensor {
        let target = target.into();
        check_broadcast_compatible(target.dims(), self.shape().dims());
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::ReduceMaxTo(target.clone()),
            inputs: vec![self.id],
            shape: target,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- reductions to a scalar ---

    /// Append a `SumAll` node reducing every element to a scalar (rank-0).
    pub fn sum_all(&self) -> Tensor {
        self.scalar_reduction(Op::SumAll)
    }

    /// Append a `MaxAll` node producing the max element as a scalar.
    pub fn max_all(&self) -> Tensor {
        self.scalar_reduction(Op::MaxAll)
    }

    /// Append a `MinAll` node producing the min element as a scalar.
    pub fn min_all(&self) -> Tensor {
        self.scalar_reduction(Op::MinAll)
    }

    /// Append a `MeanAll` node producing the arithmetic mean as a scalar.
    pub fn mean_all(&self) -> Tensor {
        self.scalar_reduction(Op::MeanAll)
    }

    // --- reductions along one dimension ---

    /// Append a `SumDim(dim)` node. Reduces along `dim`; output rank is
    /// `input rank - 1` (the reduced dim is removed).
    pub fn sum_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("sum_dim", Op::SumDim(dim), dim)
    }

    /// Append a `MaxDim(dim)` node.
    pub fn max_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("max_dim", Op::MaxDim(dim), dim)
    }

    /// Append a `MinDim(dim)` node.
    pub fn min_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("min_dim", Op::MinDim(dim), dim)
    }

    /// Append a `MeanDim(dim)` node.
    pub fn mean_dim(&self, dim: usize) -> Tensor {
        self.axis_reduction("mean_dim", Op::MeanDim(dim), dim)
    }

    /// Append an `ArgMaxDim(dim)` node. Produces a U32 tensor with the
    /// reduced dim removed, whose values are the index of the maximum
    /// along that dim. Non-differentiable — trying to run `backward()`
    /// through an ArgMax node will panic.
    pub fn argmax_dim(&self, dim: usize) -> Tensor {
        self.index_reduction("argmax_dim", Op::ArgMaxDim(dim), dim)
    }

    /// Append an `ArgMinDim(dim)` node. Same semantics as `argmax_dim`
    /// but for the minimum.
    pub fn argmin_dim(&self, dim: usize) -> Tensor {
        self.index_reduction("argmin_dim", Op::ArgMinDim(dim), dim)
    }

    /// Internal helper for index-producing reductions. Validates the
    /// dim, builds the reduced shape, and stamps the output with
    /// `DType::U32` regardless of the input dtype.
    fn index_reduction(&self, name: &'static str, op: Op, dim: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(
            dim < in_dims.len(),
            "{name}: dim {dim} out of bounds for shape {in_dims:?}",
        );
        let out_dims: Vec<usize> = in_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &d)| d)
            .collect();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype: DType::U32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- compositions ---

    /// Append a `SoftmaxLastDim` node. Shape is preserved.
    ///
    /// Phase 7.6 step 3: emits `Op::Fused(FusedOps::SOFTMAX_LAST_DIM,
    /// FusedOpParams::SoftmaxLastDim)` — the registry-extended arm.
    /// The legacy `Op::SoftmaxLastDim` variant remains in the enum
    /// during the migration; step 5 drops it once nothing emits it.
    pub fn softmax_last_dim(&self) -> Tensor {
        assert!(
            !self.shape().dims().is_empty(),
            "softmax_last_dim: input must be rank >= 1",
        );
        self.unary_op(Op::Fused(
            crate::registry::FusedOps::SOFTMAX_LAST_DIM,
            crate::registry::FusedOpParams::SoftmaxLastDim,
        ))
    }

    /// Append a `CausalConv1d` node — depthwise 1-D causal convolution
    /// + bias + optional fused SiLU. Three inputs:
    /// - `self` (x): `[batch, channels, seq + kernel - 1]` F32
    ///   (caller pre-pads with `kernel - 1` zeros on the left for the
    ///   causal mask — matches Mamba-2's prefill convention).
    /// - `weight`: `[channels, 1, kernel]` F32 (depthwise — one filter
    ///   per channel; `groups == channels`).
    /// - `bias`: `[channels]` F32 (required — matches baracuda's
    ///   `causal_conv1d_*_run` signature; pass zeros if you don't want
    ///   a bias).
    ///
    /// Output: `[batch, channels, seq]` F32 where `seq = x.dims[2] -
    /// (kernel - 1)`.
    ///
    /// Emits `Op::Fused(FusedOps::CAUSAL_CONV1D,
    /// FusedOpParams::CausalConv1d { use_silu })`. See
    /// `fuel-graph/src/registry/causal_conv1d.rs` for the registry
    /// entry. No primitive decomposition exists (fuel-graph has no
    /// `Op::Conv1D`); backends without a native kernel fall through
    /// to the executor's cpu_fallback path.
    pub fn causal_conv1d(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        use_silu: bool,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &weight.graph)
                && Arc::ptr_eq(&self.graph, &bias.graph),
            "causal_conv1d: tensors must live on the same graph",
        );
        let dtype = self.dtype();
        assert!(
            matches!(dtype, DType::F32 | DType::F64 | DType::BF16 | DType::F16),
            "causal_conv1d: x must be F32/F64/BF16/F16, got {dtype:?}",
        );
        assert_eq!(
            weight.dtype(), dtype,
            "causal_conv1d: weight dtype {:?} must match x dtype {dtype:?}",
            weight.dtype(),
        );
        assert_eq!(
            bias.dtype(), dtype,
            "causal_conv1d: bias dtype {:?} must match x dtype {dtype:?}",
            bias.dtype(),
        );
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let w_dims = weight.shape();
        let w_dims = w_dims.dims();
        let b_dims = bias.shape();
        let b_dims = b_dims.dims();
        assert_eq!(
            x_dims.len(), 3,
            "causal_conv1d: x must be rank 3 [batch, channels, seq+pad], got {x_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 3,
            "causal_conv1d: weight must be rank 3 [channels, 1, kernel], got {w_dims:?}",
        );
        assert_eq!(
            b_dims.len(), 1,
            "causal_conv1d: bias must be rank 1 [channels], got {b_dims:?}",
        );
        let batch = x_dims[0];
        let channels = x_dims[1];
        let x_seq = x_dims[2];
        let kernel = w_dims[2];
        assert_eq!(
            w_dims[0], channels,
            "causal_conv1d: weight's first dim {} must equal channels {channels}", w_dims[0],
        );
        assert_eq!(
            w_dims[1], 1,
            "causal_conv1d: weight's middle dim must be 1 (depthwise), got {}", w_dims[1],
        );
        assert_eq!(
            b_dims[0], channels,
            "causal_conv1d: bias length {} must equal channels {channels}", b_dims[0],
        );
        assert!(
            x_seq >= kernel - 1,
            "causal_conv1d: x time dim {x_seq} must be ≥ kernel - 1 = {} \
             (caller must pre-pad with {} zeros)", kernel - 1, kernel - 1,
        );
        let out_seq = x_seq - (kernel - 1);
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::CAUSAL_CONV1D,
                crate::registry::FusedOpParams::CausalConv1d { use_silu },
            ),
            inputs: vec![self.id, weight.id, bias.id],
            shape:  Shape::from_dims(&[batch, channels, out_seq]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `Nf4Matmul` node — bitsandbytes-style 4-bit
    /// NormalFloat quantized matrix multiply. Three inputs:
    /// - `self` (activations): `[..., M, K]`, dtype ∈ {F32, F16, BF16}
    /// - `w_packed`: `[N, K/2]` U8 (two 4-bit codes per byte)
    /// - `absmax`:   `[N, K/block_size]` F32 (per-output-row, per-block)
    ///
    /// Output: `[..., M, N]` matching activations' dtype.
    ///
    /// `block_size` is the NF4 quantization block size (typically 64
    /// in bitsandbytes). `K` must be even AND a multiple of
    /// `block_size`.
    ///
    /// Emits `Op::Fused(FusedOps::NF4_MATMUL, FusedOpParams::Nf4Matmul
    /// { block_size })`. No primitive decomposition (the dequant +
    /// matmul roundtrip is exactly what NF4's fused dequant-in-kernel
    /// design avoids); backends without a native kernel fall through
    /// to the executor's cpu_fallback path.
    pub fn nf4_matmul(
        &self,
        w_packed: &Tensor,
        absmax: &Tensor,
        block_size: usize,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &w_packed.graph)
                && Arc::ptr_eq(&self.graph, &absmax.graph),
            "nf4_matmul: tensors must live on the same graph",
        );
        let act_dtype = self.dtype();
        assert!(
            matches!(act_dtype, DType::F32 | DType::F16 | DType::BF16),
            "nf4_matmul v1: activations must be F32/F16/BF16, got {:?}", act_dtype,
        );
        assert_eq!(
            w_packed.dtype(),
            DType::U8,
            "nf4_matmul: w_packed must be U8 (two 4-bit codes per byte), got {:?}",
            w_packed.dtype(),
        );
        assert_eq!(
            absmax.dtype(),
            DType::F32,
            "nf4_matmul: absmax must be F32, got {:?}", absmax.dtype(),
        );
        let a_dims = self.shape();
        let a_dims = a_dims.dims();
        let w_dims = w_packed.shape();
        let w_dims = w_dims.dims();
        let abs_dims = absmax.shape();
        let abs_dims = abs_dims.dims();
        assert!(
            a_dims.len() >= 2,
            "nf4_matmul: activations must be rank ≥ 2, got {a_dims:?}",
        );
        assert_eq!(
            w_dims.len(), 2,
            "nf4_matmul: w_packed must be rank 2 [n, k/2], got {w_dims:?}",
        );
        assert_eq!(
            abs_dims.len(), 2,
            "nf4_matmul: absmax must be rank 2 [n, k/block_size], got {abs_dims:?}",
        );
        let m = a_dims[a_dims.len() - 2];
        let k = a_dims[a_dims.len() - 1];
        let n = w_dims[0];
        assert!(
            k % 2 == 0,
            "nf4_matmul: k={k} must be even (w_packed holds 2 nibbles per byte along k)",
        );
        assert!(
            block_size > 0 && k % block_size == 0,
            "nf4_matmul: k={k} must be a positive multiple of block_size={block_size}",
        );
        assert_eq!(
            w_dims[1], k / 2,
            "nf4_matmul: w_packed second dim {} must equal k/2 = {}", w_dims[1], k / 2,
        );
        assert_eq!(
            abs_dims, &[n, k / block_size][..],
            "nf4_matmul: absmax {abs_dims:?} must be [n={n}, k/block_size={}]",
            k / block_size,
        );
        let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
        out_dims.push(n);
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::NF4_MATMUL,
                crate::registry::FusedOpParams::Nf4Matmul { block_size },
            ),
            inputs: vec![self.id, w_packed.id, absmax.id],
            shape:  Shape::from_dims(&out_dims),
            dtype:  act_dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `SsdChunkScan` node — Mamba-2's State-Space
    /// Duality chunked scan (forward). Five inputs:
    /// - `self` (x): `[batch, seqlen, heads, head_dim]` F32
    /// - `dt`:       `[batch, seqlen, heads]` F32
    /// - `a`:        `[heads]` F32 (per-head scalar log A)
    /// - `b`:        `[batch, seqlen, heads, state_dim]` F32
    /// - `c`:        `[batch, seqlen, heads, state_dim]` F32
    ///
    /// Output: `y: [batch, seqlen, heads, head_dim]` F32 (same as x).
    ///
    /// `chunk_size` is the SSD block size (typically 256 in Mamba-2).
    /// It controls GPU parallelism granularity but the mathematical
    /// result is identical to a sequential scan; the CPU kernel runs
    /// sequential regardless. Validation: `chunk_size > 0` and
    /// `seqlen % chunk_size == 0`.
    ///
    /// Emits `Op::Fused(FusedOps::SSD_CHUNK_SCAN,
    /// FusedOpParams::SsdChunkScan { chunk_size })`. No primitive
    /// decomposition; backends without a native kernel fall through
    /// to the executor's cpu_fallback path.
    /// Internal: build the bundled SsdChunkScan producer node and
    /// return its NodeId. Callers project per-slot via
    /// [`Tensor::view`] / [`Tensor::view_owned`].
    fn ssd_chunk_scan_producer(
        &self,
        dt: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        chunk_size: usize,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &dt.graph)
                && Arc::ptr_eq(&self.graph, &a.graph)
                && Arc::ptr_eq(&self.graph, &b.graph)
                && Arc::ptr_eq(&self.graph, &c.graph),
            "ssd_chunk_scan: all tensors must live on the same graph",
        );
        let dtype = self.dtype();
        assert!(
            matches!(dtype, DType::F32 | DType::F64 | DType::BF16 | DType::F16),
            "ssd_chunk_scan: x must be F32/F64/BF16/F16, got {dtype:?}",
        );
        for (name, t) in [("dt", dt), ("a", a), ("b", b), ("c", c)] {
            assert_eq!(
                t.dtype(), dtype,
                "ssd_chunk_scan: {name} dtype {:?} must match x dtype {dtype:?}",
                t.dtype(),
            );
        }
        let x_dims = self.shape();
        let x_dims = x_dims.dims();
        let dt_dims = dt.shape();
        let dt_dims = dt_dims.dims();
        let a_dims = a.shape();
        let a_dims = a_dims.dims();
        let b_dims = b.shape();
        let b_dims = b_dims.dims();
        let c_dims = c.shape();
        let c_dims = c_dims.dims();
        assert_eq!(
            x_dims.len(), 4,
            "ssd_chunk_scan: x must be rank 4 [batch, seqlen, heads, head_dim], got {x_dims:?}",
        );
        let batch = x_dims[0];
        let seqlen = x_dims[1];
        let heads = x_dims[2];
        let head_dim = x_dims[3];
        assert_eq!(
            dt_dims, &[batch, seqlen, heads][..],
            "ssd_chunk_scan: dt {dt_dims:?} must be [batch={batch}, seqlen={seqlen}, heads={heads}]",
        );
        assert_eq!(
            a_dims, &[heads][..],
            "ssd_chunk_scan: a {a_dims:?} must be [heads={heads}]",
        );
        assert_eq!(
            b_dims.len(), 4,
            "ssd_chunk_scan: b must be rank 4 [batch, seqlen, heads, state_dim], got {b_dims:?}",
        );
        let state_dim = b_dims[3];
        assert_eq!(
            b_dims, &[batch, seqlen, heads, state_dim][..],
            "ssd_chunk_scan: b {b_dims:?} must be [batch={batch}, seqlen={seqlen}, heads={heads}, state_dim={state_dim}]",
        );
        assert_eq!(
            c_dims, &[batch, seqlen, heads, state_dim][..],
            "ssd_chunk_scan: c {c_dims:?} must match b shape",
        );
        let params = crate::registry::FusedOpParams::SsdChunkScan { chunk_size };
        let input_shapes = [
            Shape::from_dims(x_dims),
            Shape::from_dims(dt_dims),
            Shape::from_dims(a_dims),
            Shape::from_dims(b_dims),
            Shape::from_dims(c_dims),
        ];
        let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
        let entry = crate::registry::default_registry()
            .entry(crate::registry::FusedOps::SSD_CHUNK_SCAN)
            .expect("SSD_CHUNK_SCAN registered");
        let specs = (entry.output_views.expect("multi-output"))(
            &input_shapes, &input_dtypes, &params,
        );
        let (_total_bytes, views) = fuel_ir::storage::compose_bundle(&specs)
            .expect("SsdChunkScan compose_bundle");
        let mut g = self.graph.write().unwrap();
        let id = g.push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::SSD_CHUNK_SCAN,
                params,
            ),
            inputs: vec![self.id, dt.id, a.id, b.id, c.id],
            shape:  Shape::from_dims(&[batch, seqlen, heads, head_dim]),
            dtype,
        });
        g.set_output_views(id, Arc::from(views.into_boxed_slice()))
            .expect("SsdChunkScan set_output_views");
        drop(g);
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Single-output SsdChunkScan: returns the `y` slot of the
    /// bundled producer (View(0)). See
    /// [`Self::ssd_chunk_scan_bundled`] for the multi-output variant.
    pub fn ssd_chunk_scan(
        &self,
        dt: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        chunk_size: usize,
    ) -> Tensor {
        let producer = self.ssd_chunk_scan_producer(dt, a, b, c, chunk_size);
        producer.view(0).expect("SsdChunkScan view(0) must succeed (just registered slots)")
    }

    /// Append a `SelectiveScan` node — Mamba-1's selective
    /// state-space scan (forward). Five inputs:
    /// - `self` (u): `[batch, seqlen, dim]` F32
    /// - `delta`:    `[batch, seqlen, dim]` F32
    /// - `a`:        `[dim, dstate]` F32
    /// - `b`:        `[batch, seqlen, dstate]` F32
    /// - `c`:        `[batch, seqlen, dstate]` F32
    ///
    /// Output: `y: [batch, seqlen, dim]` F32.
    ///
    /// `delta_softplus` toggles the softplus(delta) activation before
    /// use (matches baracuda's `selective_scan_*_run` flag). The v1
    /// op surface skips the optional `d_skip`, `z`, `delta_bias`
    /// inputs and the `last_state` second output — see
    /// `fuel-graph/src/registry/selective_scan.rs` for the rationale.
    ///
    /// Emits `Op::Fused(FusedOps::SELECTIVE_SCAN,
    /// FusedOpParams::SelectiveScan { delta_softplus })`. No
    /// primitive decomposition exists (the scan is a sequential
    /// recurrence); backends without a native kernel fall through to
    /// the executor's cpu_fallback path.
    /// Internal: build the bundled SelectiveScan producer node and
    /// return its NodeId (wrapped in a `Tensor`). Callers project
    /// per-slot via [`Tensor::view`] / [`Tensor::view_owned`].
    fn selective_scan_producer(
        &self,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        delta_softplus: bool,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &delta.graph)
                && Arc::ptr_eq(&self.graph, &a.graph)
                && Arc::ptr_eq(&self.graph, &b.graph)
                && Arc::ptr_eq(&self.graph, &c.graph),
            "selective_scan: all tensors must live on the same graph",
        );
        let dtype = self.dtype();
        assert!(
            matches!(dtype, DType::F32 | DType::F64 | DType::BF16 | DType::F16),
            "selective_scan: u must be F32/F64/BF16/F16, got {dtype:?}",
        );
        for (name, t) in [("delta", delta), ("a", a), ("b", b), ("c", c)] {
            assert_eq!(
                t.dtype(), dtype,
                "selective_scan: {name} dtype {:?} must match u dtype {dtype:?}",
                t.dtype(),
            );
        }
        let u_dims = self.shape();
        let u_dims = u_dims.dims();
        let delta_dims = delta.shape();
        let delta_dims = delta_dims.dims();
        let a_dims = a.shape();
        let a_dims = a_dims.dims();
        let b_dims = b.shape();
        let b_dims = b_dims.dims();
        let c_dims = c.shape();
        let c_dims = c_dims.dims();
        assert_eq!(
            u_dims.len(), 3,
            "selective_scan: u must be rank 3 [batch, seqlen, dim], got {u_dims:?}",
        );
        assert_eq!(
            delta_dims, u_dims,
            "selective_scan: delta {delta_dims:?} must match u {u_dims:?}",
        );
        assert_eq!(
            a_dims.len(), 2,
            "selective_scan: a must be rank 2 [dim, dstate], got {a_dims:?}",
        );
        let batch = u_dims[0];
        let seqlen = u_dims[1];
        let dim = u_dims[2];
        let dstate = a_dims[1];
        assert_eq!(
            a_dims[0], dim,
            "selective_scan: a's first dim {} must equal dim {dim}", a_dims[0],
        );
        assert_eq!(
            b_dims, &[batch, seqlen, dstate][..],
            "selective_scan: b {b_dims:?} must be [batch={batch}, seqlen={seqlen}, dstate={dstate}]",
        );
        assert_eq!(
            c_dims, &[batch, seqlen, dstate][..],
            "selective_scan: c {c_dims:?} must be [batch={batch}, seqlen={seqlen}, dstate={dstate}]",
        );
        let params = crate::registry::FusedOpParams::SelectiveScan {
            delta_softplus,
        };
        let input_shapes = [
            Shape::from_dims(u_dims),
            Shape::from_dims(delta_dims),
            Shape::from_dims(a_dims),
            Shape::from_dims(b_dims),
            Shape::from_dims(c_dims),
        ];
        let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
        let entry = crate::registry::default_registry()
            .entry(crate::registry::FusedOps::SELECTIVE_SCAN)
            .expect("SELECTIVE_SCAN registered");
        let specs = (entry.output_views.expect("multi-output"))(
            &input_shapes, &input_dtypes, &params,
        );
        let (_total_bytes, views) = fuel_ir::storage::compose_bundle(&specs)
            .expect("SelectiveScan compose_bundle");
        let mut g = self.graph.write().unwrap();
        let id = g.push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::SELECTIVE_SCAN,
                params,
            ),
            inputs: vec![self.id, delta.id, a.id, b.id, c.id],
            shape:  Shape::from_dims(&[batch, seqlen, dim]),
            dtype,
        });
        g.set_output_views(id, Arc::from(views.into_boxed_slice()))
            .expect("SelectiveScan set_output_views");
        drop(g);
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Single-output SelectiveScan: returns the `y` slot of the
    /// bundled producer, with the View(0) projection emitted so
    /// callers can downstream-realize / copy / etc. transparently.
    /// See [`Self::selective_scan_bundled`] for the multi-output
    /// variant that also exposes `last_state` for autoregressive
    /// resumption.
    pub fn selective_scan(
        &self,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        delta_softplus: bool,
    ) -> Tensor {
        let producer = self.selective_scan_producer(delta, a, b, c, delta_softplus);
        producer.view(0).expect("SelectiveScan view(0) must succeed (just registered slots)")
    }

    /// Multi-output variant of [`Self::selective_scan`]. Returns
    /// `(y, last_state)` — both Op::View tensors projected from the
    /// shared bundled producer. `y` matches the existing
    /// single-output `selective_scan` shape; `last_state` is the
    /// final hidden state `[batch, dim, dstate]` for autoregressive
    /// resumption.
    pub fn selective_scan_bundled(
        &self,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        delta_softplus: bool,
    ) -> std::result::Result<(Tensor, Tensor), fuel_ir::Error> {
        let producer = self.selective_scan_producer(delta, a, b, c, delta_softplus);
        let y = producer.view(0)?;
        let last_state = producer.view(1)?;
        Ok((y, last_state))
    }

    /// Multi-output variant of [`Self::ssd_chunk_scan`]. Returns
    /// `(y, last_state)`.
    pub fn ssd_chunk_scan_bundled(
        &self,
        dt: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        chunk_size: usize,
    ) -> std::result::Result<(Tensor, Tensor), fuel_ir::Error> {
        let producer = self.ssd_chunk_scan_producer(dt, a, b, c, chunk_size);
        let y = producer.view(0)?;
        let last_state = producer.view(1)?;
        Ok((y, last_state))
    }

    /// Append a `FusedSoftmaxCrossEntropy` node. Two inputs:
    /// - `self` (logits): `[..., V]` F32 / F64 / BF16 / F16
    /// - `targets`: `[...]` I64 (class indices; matches PyTorch /
    ///   baracuda convention)
    ///
    /// Output dtype is always F32 (the FSCE declared dtype — losses
    /// accumulate in F64 and narrow to F32, matching PyTorch and the
    /// baracuda kernel) regardless of the logits dtype. Output shape
    /// depends on `reduction`:
    /// - `Reduction::Mean` / `Reduction::Sum` → scalar `[]`
    /// - `Reduction::None` → same as `targets.shape`
    ///
    /// `ignore_index` rows are dropped from the loss accumulator and
    /// the Mean denominator. The conventional sentinel is `-100`.
    ///
    /// Emits `Op::Fused(FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
    /// FusedOpParams::FusedSoftmaxCrossEntropy { reduction, ignore_index })`.
    /// See `fuel-graph/src/registry/fused_softmax_cross_entropy.rs` for
    /// the registry entry and decompose chain.
    pub fn fused_softmax_cross_entropy(
        &self,
        targets: &Tensor,
        reduction: crate::registry::Reduction,
        ignore_index: i64,
    ) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &targets.graph),
            "fused_softmax_cross_entropy: tensors must live on the same graph",
        );
        assert!(
            matches!(
                self.dtype(),
                DType::F32 | DType::F64 | DType::BF16 | DType::F16,
            ),
            "fused_softmax_cross_entropy: logits must be F32/F64/BF16/F16, got {:?}",
            self.dtype(),
        );
        assert_eq!(
            targets.dtype(),
            DType::I64,
            "fused_softmax_cross_entropy: targets must be I64, got {:?}",
            targets.dtype(),
        );
        let logits_dims = self.shape();
        let logits_dims = logits_dims.dims();
        assert!(
            !logits_dims.is_empty(),
            "fused_softmax_cross_entropy: logits must have rank ≥ 1",
        );
        let target_dims = targets.shape();
        let target_dims = target_dims.dims();
        assert_eq!(
            target_dims, &logits_dims[..logits_dims.len() - 1],
            "fused_softmax_cross_entropy: targets shape {target_dims:?} must equal \
             logits shape {logits_dims:?} minus the last dim",
        );
        let out_shape = match reduction {
            crate::registry::Reduction::Mean | crate::registry::Reduction::Sum => {
                Shape::from_dims(&[])
            }
            crate::registry::Reduction::None => Shape::from_dims(target_dims),
        };
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY,
                crate::registry::FusedOpParams::FusedSoftmaxCrossEntropy {
                    reduction,
                    ignore_index,
                },
            ),
            inputs: vec![self.id, targets.id],
            shape:  out_shape,
            dtype:  DType::F32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `LayerNormLastDim` node with the given epsilon. Shape is
    /// preserved.
    ///
    /// Phase 7.6 step 4 (continued): emits
    /// `Op::Fused(FusedOps::LAYER_NORM_LAST_DIM, FusedOpParams::LayerNormLastDim { eps })`
    /// through the registry-extended arm. The legacy
    /// `Op::LayerNormLastDim { eps }` variant remains in the enum
    /// during migration; step 5 drops it once nothing emits it.
    pub fn layer_norm_last_dim(&self, eps: f64) -> Tensor {
        let dims = self.shape();
        let d = dims.dims();
        assert!(
            !d.is_empty() && *d.last().unwrap() > 0,
            "layer_norm_last_dim: input must have a non-zero last dim, got {d:?}",
        );
        self.unary_op(Op::Fused(
            crate::registry::FusedOps::LAYER_NORM_LAST_DIM,
            crate::registry::FusedOpParams::LayerNormLastDim { eps },
        ))
    }

    /// Apply rotary position embeddings (RoPE) to `self`. The input is
    /// expected to have shape `[..., seq, d]` where `seq` is the
    /// sequence length and `d` is the per-head feature dim (which must
    /// be even). `base` is the frequency base — 10000.0 for LLaMA.
    /// `start_pos` lets callers apply RoPE at a non-zero offset, which
    /// is what KV-cached inference does when the query for position
    /// `n + k` is processed against a cache that already holds
    /// positions `0..n+k`.
    ///
    /// Uses the half-split layout (the one LLaMA and HuggingFace use):
    /// pair feature `i` with feature `i + d/2`. Under that layout the
    /// rotation reduces to the identity `x' = x·cos + rotate_half(x)·sin`
    /// where `rotate_half(x)` concatenates `[-x[d/2:], x[:d/2]]` along
    /// the last dim.
    ///
    /// Implemented as a composition of existing graph primitives —
    /// `Slice`, `Neg`, `Concat`, `Reshape`, `BroadcastTo`, `Mul`, `Add`
    /// — plus two precomputed `Const` tensors for the cos/sin frequency
    /// tables. Every primitive already has a working backward rule, so
    /// gradients flow through RoPE without any new backward machinery.
    ///
    /// Only `f32` is supported today. Extending to other float dtypes
    /// is mechanical: build the frequency tables in the target dtype.
    pub fn rope(&self, base: f64, start_pos: usize) -> Tensor {
        let (seq, d) = {
            let dims = self.shape();
            let v = dims.dims();
            let rank = v.len();
            assert!(rank >= 2, "rope: input must have rank ≥ 2");
            (v[rank - 2], v[rank - 1])
        };
        let (cos, sin) = build_rope_tables(base, start_pos, seq, d);
        // Phase 7.5 G2: const_*_like derives the device from self's
        // graph internally — RoPE tables go on the same device as self.
        let cos_t = self.const_f32_like(cos, Shape::from_dims(&[seq, d]));
        let sin_t = self.const_f32_like(sin, Shape::from_dims(&[seq, d]));
        self.rope_with_tables(&cos_t, &sin_t)
    }

    /// Apply RoPE using caller-supplied `cos` and `sin` tables. Each
    /// table has shape `[seq, head_dim]` and matches the layout
    /// [`build_rope_tables`] produces. This is the hot-path entry
    /// point: in a transformer forward pass every attention layer
    /// applies RoPE to Q and K with the *same* `(start_pos, seq,
    /// head_dim)`, so the caller can build the tables once and share
    /// the const nodes across all layers rather than re-duplicating
    /// them inside each `.rope()` call. The classic [`rope`] entry
    /// point funnels through this after building the tables itself.
    pub fn rope_with_tables(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        assert_eq!(
            self.dtype(),
            DType::F32,
            "rope: only f32 is supported today (cast explicitly for other dtypes)",
        );
        let in_shape = self.shape();
        let dims_vec: Vec<usize> = in_shape.dims().to_vec();
        let rank = dims_vec.len();
        assert!(rank >= 2, "rope: input must have rank ≥ 2, got {dims_vec:?}");
        let seq = dims_vec[rank - 2];
        let d = dims_vec[rank - 1];
        assert!(
            d.is_multiple_of(2),
            "rope: feature dim {d} must be even",
        );
        let cos_shape = cos.shape();
        let sin_shape = sin.shape();
        assert_eq!(
            cos_shape.dims(),
            &[seq, d],
            "rope_with_tables: cos shape {:?} does not match [seq, d] = [{seq}, {d}]",
            cos_shape.dims(),
        );
        assert_eq!(
            sin_shape.dims(),
            &[seq, d],
            "rope_with_tables: sin shape {:?} does not match [seq, d] = [{seq}, {d}]",
            sin_shape.dims(),
        );
        let _ = rank;
        // Emit a single fused Rope node. The decomposed version
        // (slice+neg+concat+broadcast+mul+add) produces ~72 dispatches
        // on GPU backends because concat-along-last-dim has a per-row
        // host loop. Fused path dispatches once.
        //
        // Phase 7.6 step 4 (continued): emits
        // `Op::Fused(FusedOps::ROPE, FusedOpParams::Rope)` through the
        // registry-extended arm. The legacy `Op::Rope` variant remains
        // in the enum during migration; step 5 drops it once nothing
        // emits it.
        let out_shape = in_shape.clone();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::ROPE,
                crate::registry::FusedOpParams::Rope,
            ),
            inputs: vec![self.id, cos.id, sin.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// The pre-fused decomposition of `rope_with_tables`. Retained so
    /// backends without a native `Op::Rope` kernel can synthesize from
    /// primitives, and so correctness tests can cross-check the fused
    /// path against the primitive path.
    #[doc(hidden)]
    pub fn rope_with_tables_decomposed(&self, cos: &Tensor, sin: &Tensor) -> Tensor {
        let in_shape = self.shape();
        let dims_vec: Vec<usize> = in_shape.dims().to_vec();
        let rank = dims_vec.len();
        let seq = dims_vec[rank - 2];
        let d = dims_vec[rank - 1];
        let half = d / 2;

        let mut broadcast_shape: Vec<usize> = vec![1_usize; rank];
        broadcast_shape[rank - 2] = seq;
        broadcast_shape[rank - 1] = d;
        let cos_reshaped = cos.reshape(Shape::from_dims(&broadcast_shape));
        let sin_reshaped = sin.reshape(Shape::from_dims(&broadcast_shape));
        let cos_bcast = cos_reshaped.broadcast_to(in_shape.clone());
        let sin_bcast = sin_reshaped.broadcast_to(in_shape);

        let first_half = self.slice(rank - 1, 0, half);
        let second_half = self.slice(rank - 1, half, half);
        let neg_second = second_half.neg();
        let rotated_half = neg_second.concat(&first_half, rank - 1);

        let left = self.mul(&cos_bcast);
        let right = rotated_half.mul(&sin_bcast);
        left.add(&right)
    }

    /// Root-Mean-Square Normalization along the last dim, without
    /// affine parameters. `y = x / sqrt(mean(x², last) + eps)`.
    ///
    /// This is the normalization layer used by LLaMA-family models.
    /// Unlike LayerNorm, RmsNorm does not subtract the mean — it only
    /// divides by the root-mean-square, which is both cheaper and
    /// empirically just as effective in transformer blocks.
    ///
    /// Emits a fused RmsNormLastDim node. Backends that have a native
    /// implementation dispatch it as one kernel; ones that don't get a
    /// CPU fallback via the reference implementation.
    ///
    /// Phase 7.6 step 4 (continued): emits
    /// `Op::Fused(FusedOps::RMS_NORM_LAST_DIM, FusedOpParams::RmsNormLastDim { eps })`
    /// through the registry-extended arm. The legacy
    /// `Op::RmsNormLastDim { eps }` variant remains in the enum during
    /// migration; step 5 drops it once nothing emits it.
    ///
    /// Use [`rms_norm_last_dim_decomposed`](Self::rms_norm_last_dim_decomposed)
    /// instead if you need to differentiate through this op —
    /// backward through the fused op is not yet implemented.
    pub fn rms_norm_last_dim(&self, eps: f64) -> Tensor {
        let x_dims_vec: Vec<usize> = self.shape().dims().to_vec();
        assert!(
            !x_dims_vec.is_empty() && *x_dims_vec.last().unwrap() > 0,
            "rms_norm_last_dim: input must have non-zero last dim, got {x_dims_vec:?}",
        );
        self.unary_op(Op::Fused(
            crate::registry::FusedOps::RMS_NORM_LAST_DIM,
            crate::registry::FusedOpParams::RmsNormLastDim { eps },
        ))
    }

    /// Decomposed version — emits the (sqr → mean_dim → reshape →
    /// add_scalar → sqrt → broadcast_to → div) primitive chain. Has
    /// working backward rules through its primitive subgraph. Use
    /// this when you need to differentiate through RMSNorm; use
    /// [`rms_norm_last_dim`](Self::rms_norm_last_dim) for inference.
    pub fn rms_norm_last_dim_decomposed(&self, eps: f64) -> Tensor {
        let x_shape = self.shape();
        let x_dims_vec: Vec<usize> = x_shape.dims().to_vec();
        assert!(
            !x_dims_vec.is_empty() && *x_dims_vec.last().unwrap() > 0,
            "rms_norm_last_dim_decomposed: input must have non-zero last dim, got {x_dims_vec:?}",
        );
        let last = x_dims_vec.len() - 1;
        let mean_sq = self.sqr().mean_dim(last);
        let mut keepdim_dims = x_dims_vec.clone();
        keepdim_dims[last] = 1;
        let mean_sq_kd = mean_sq.reshape(Shape::from_dims(&keepdim_dims));
        let denom = mean_sq_kd.add_scalar(eps).sqrt();
        let denom_bcast = denom.broadcast_to(x_shape);
        self.div(&denom_bcast)
    }

    // --- indexing ---

    /// Append an `IndexSelect` node that picks slices from `self` along
    /// `dim` using a 1-D `u32` index tensor. The output has the same
    /// shape as `self` except dimension `dim` is replaced by
    /// `indices.shape()[0]`.
    pub fn index_select(&self, dim: usize, indices: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &indices.graph),
            "index_select: data and index tensors must live on the same graph",
        );
        assert_eq!(
            indices.dtype(),
            DType::U32,
            "index_select: index tensor must be U32, got {:?}",
            indices.dtype(),
        );
        let idx_dims = indices.shape();
        assert_eq!(
            idx_dims.dims().len(),
            1,
            "index_select: index tensor must be rank 1, got {:?}",
            idx_dims.dims(),
        );
        let data_dims = self.shape();
        assert!(
            dim < data_dims.dims().len(),
            "index_select: dim {dim} out of bounds for data shape {:?}",
            data_dims.dims(),
        );
        let mut out_dims: Vec<usize> = data_dims.dims().to_vec();
        out_dims[dim] = idx_dims.dims()[0];
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::IndexSelect { dim },
            inputs: vec![self.id, indices.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Gather` node that performs an N-dimensional gather along
    /// `dim`. The `indices` tensor must be `U32` with the same rank as
    /// `self`. Output shape equals `indices.shape()`.
    pub fn gather(&self, dim: usize, indices: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &indices.graph),
            "gather: data and index tensors must live on the same graph",
        );
        assert_eq!(
            indices.dtype(),
            DType::U32,
            "gather: index tensor must be U32, got {:?}",
            indices.dtype(),
        );
        let data_rank = self.shape().dims().len();
        let idx_rank = indices.shape().dims().len();
        assert_eq!(
            data_rank, idx_rank,
            "gather: data and index must have the same rank, got {data_rank} vs {idx_rank}",
        );
        assert!(
            dim < data_rank,
            "gather: dim {dim} out of bounds for rank {data_rank}",
        );
        let dtype = self.dtype();
        let out_shape = indices.shape();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Gather { dim },
            inputs: vec![self.id, indices.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Concat` node joining `self` and `other` along `dim`.
    pub fn concat(&self, other: &Tensor, dim: usize) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &other.graph),
            "concat: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "concat: dtype mismatch {:?} vs {:?}",
            self.dtype(),
            other.dtype(),
        );
        let a = self.shape();
        let b = other.shape();
        let ad = a.dims();
        let bd = b.dims();
        assert_eq!(ad.len(), bd.len(), "concat: rank mismatch");
        assert!(dim < ad.len(), "concat: dim out of bounds");
        for i in 0..ad.len() {
            if i != dim {
                assert_eq!(
                    ad[i], bd[i],
                    "concat: non-dim shapes must match (dim {i}: {} vs {})",
                    ad[i], bd[i],
                );
            }
        }
        let mut out_dims: Vec<usize> = ad.to_vec();
        out_dims[dim] = ad[dim] + bd[dim];
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Concat { dim },
            inputs: vec![self.id, other.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `Slice` (narrow) node taking elements `[start, start+len)`
    /// along `dim`. The output has the same rank; only `dim` shrinks.
    pub fn slice(&self, dim: usize, start: usize, len: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(dim < in_dims.len(), "slice: dim out of bounds");
        assert!(
            start + len <= in_dims[dim],
            "slice: [start={start}, len={len}) exceeds dim size {}",
            in_dims[dim],
        );
        let mut out_dims: Vec<usize> = in_dims.to_vec();
        out_dims[dim] = len;
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::Slice { dim, start, len },
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append an `AddScalar` node: `y[i] = x[i] + c`. Convenient for
    /// avoiding a full-shape const node when you just want to bias a
    /// tensor by a constant.
    pub fn add_scalar(&self, c: f64) -> Tensor {
        self.unary_op(Op::AddScalar(c))
    }

    /// Append a `MulScalar` node: `y[i] = x[i] * c`.
    pub fn mul_scalar(&self, c: f64) -> Tensor {
        self.unary_op(Op::MulScalar(c))
    }

    /// Append a `PowI` node raising each element to an integer power.
    pub fn powi(&self, n: i32) -> Tensor {
        self.unary_op(Op::PowI(n))
    }

    /// Append a `Clamp` node restricting each element to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Tensor {
        assert!(
            min <= max,
            "clamp: min ({min}) must be <= max ({max})",
        );
        self.unary_op(Op::Clamp { min, max })
    }

    /// Append a `Maximum` node `max(self, other)` element-wise. Matching
    /// shapes required.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        self.binary_op("maximum", Op::Maximum, other, self.shape())
    }

    /// Append a `Minimum` node `min(self, other)` element-wise.
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        self.binary_op("minimum", Op::Minimum, other, self.shape())
    }

    /// Element-wise addition with automatic broadcasting. Unlike `add`,
    /// which requires matching shapes, `broadcast_add` computes the
    /// broadcast shape of the two operands, inserts explicit
    /// `BroadcastTo` nodes as needed, and then emits a regular `Add`.
    /// Useful for bias addition (`[batch, hidden] + [hidden]`) without
    /// the caller writing the broadcast out explicitly.
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_add", other);
        a.add(&b)
    }

    /// Element-wise subtraction with automatic broadcasting.
    pub fn broadcast_sub(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_sub", other);
        a.sub(&b)
    }

    /// Element-wise multiplication with automatic broadcasting.
    pub fn broadcast_mul(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_mul", other);
        a.mul(&b)
    }

    /// Element-wise division with automatic broadcasting.
    pub fn broadcast_div(&self, other: &Tensor) -> Tensor {
        let (a, b) = self.auto_broadcast_pair("broadcast_div", other);
        a.div(&b)
    }

    /// Build both operands broadcast to their common shape. Used by the
    /// `broadcast_*` wrappers above. Inserts explicit `BroadcastTo` nodes
    /// on whichever side needs them — either or both may pass through
    /// unchanged if already at the target shape.
    fn auto_broadcast_pair(&self, op: &'static str, other: &Tensor) -> (Tensor, Tensor) {
        assert!(
            Arc::ptr_eq(&self.graph, &other.graph),
            "{op}: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "{op}: dtype mismatch {:?} vs {:?}",
            self.dtype(),
            other.dtype(),
        );
        let a_shape = self.shape();
        let b_shape = other.shape();
        let target_dims = compute_broadcast_shape(a_shape.dims(), b_shape.dims());
        let target = Shape::from_dims(&target_dims);
        let a = if a_shape.dims() == target.dims() {
            self.clone()
        } else {
            self.broadcast_to(target.clone())
        };
        let b = if b_shape.dims() == target.dims() {
            other.clone()
        } else {
            other.broadcast_to(target)
        };
        (a, b)
    }

    /// Append an `IndexAdd` node — the functional inverse of
    /// `IndexSelect`. Returns `base` with `src` added at positions given
    /// by `indices` along `dim`.
    pub fn index_add(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &indices.graph) && Arc::ptr_eq(&self.graph, &src.graph),
            "index_add: all tensors must live on the same graph",
        );
        assert_eq!(indices.dtype(), DType::U32, "index_add: index must be U32");
        assert_eq!(self.dtype(), src.dtype(), "index_add: base and src dtypes must match");
        let base_dims = self.shape();
        let src_dims = src.shape();
        assert_eq!(
            base_dims.dims().len(),
            src_dims.dims().len(),
            "index_add: base and src must have the same rank",
        );
        assert_eq!(
            indices.shape().dims().len(),
            1,
            "index_add: index must be rank 1",
        );
        assert_eq!(
            src_dims.dims()[dim],
            indices.shape().dims()[0],
            "index_add: src dim {dim} ({}) must match index length ({})",
            src_dims.dims()[dim],
            indices.shape().dims()[0],
        );
        let dtype = self.dtype();
        let out_shape = base_dims;
        let id = self.graph.write().unwrap().push(Node {
            op: Op::IndexAdd { dim },
            inputs: vec![self.id, indices.id, src.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `ScatterAdd` node — the functional inverse of `Gather`.
    /// Returns `base` with values from `src` accumulated at positions
    /// given by `indices` (with `indices[p]` substituted at `dim`).
    pub fn scatter_add(&self, dim: usize, indices: &Tensor, src: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &indices.graph) && Arc::ptr_eq(&self.graph, &src.graph),
            "scatter_add: all tensors must live on the same graph",
        );
        assert_eq!(indices.dtype(), DType::U32, "scatter_add: index must be U32");
        assert_eq!(self.dtype(), src.dtype(), "scatter_add: base and src dtypes must match");
        assert_eq!(
            indices.shape().dims(),
            src.shape().dims(),
            "scatter_add: index and src must have the same shape",
        );
        let dtype = self.dtype();
        let out_shape = self.shape();
        let id = self.graph.write().unwrap().push(Node {
            op: Op::ScatterAdd { dim },
            inputs: vec![self.id, indices.id, src.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // --- internal helpers for the new builders ---

    fn scalar_reduction(&self, op: Op) -> Tensor {
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&[]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    fn axis_reduction(&self, name: &'static str, op: Op, dim: usize) -> Tensor {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        assert!(
            dim < in_dims.len(),
            "{name}: dim {dim} out of bounds for shape {in_dims:?}",
        );
        let out_dims: Vec<usize> = in_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &d)| d)
            .collect();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id],
            shape: Shape::from_dims(&out_dims),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    // Internal helpers. Both shape-validate and append.

    fn binary_op(&self, name: &'static str, op: Op, other: &Tensor, out_shape: Shape) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &other.graph),
            "{name}: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "{name}: dtype mismatch: lhs={:?}, rhs={:?}",
            self.dtype(),
            other.dtype(),
        );
        assert_eq!(
            self.shape().dims(),
            other.shape().dims(),
            "{name}: shape mismatch: lhs={:?}, rhs={:?}",
            self.shape().dims(),
            other.shape().dims(),
        );
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id, other.id],
            shape: out_shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Result-returning sibling of [`Self::binary_op`]. Validates the
    /// graph/dtype/shape preconditions and returns `Err` rather than
    /// panicking — the production-correct shape new builders should
    /// use. Existing panicking sites are kept for back-compat; new
    /// ops should call this directly.
    fn try_binary_op(
        &self,
        name: &'static str,
        op: Op,
        other: &Tensor,
        out_shape: Shape,
    ) -> std::result::Result<Tensor, fuel_ir::Error> {
        if !Arc::ptr_eq(&self.graph, &other.graph) {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: tensors must live on the same graph",
            )).bt());
        }
        if self.dtype() != other.dtype() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: dtype mismatch: lhs={:?}, rhs={:?}",
                self.dtype(),
                other.dtype(),
            )).bt());
        }
        if self.shape().dims() != other.shape().dims() {
            return Err(fuel_ir::Error::Msg(format!(
                "{name}: shape mismatch: lhs={:?}, rhs={:?}",
                self.shape().dims(),
                other.shape().dims(),
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id, other.id],
            shape: out_shape,
            dtype,
        });
        Ok(Self {
            graph: self.graph.clone(),
            id,
        })
    }

    fn unary_op(&self, op: Op) -> Tensor {
        let shape = self.shape();
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id],
            shape,
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Binary builder for ops whose inputs share dtype/shape but whose
    /// output is a `U8` mask (`Op::Equal` and the rest of the
    /// comparison family). Differs from [`Self::binary_op`] only in the
    /// node's output dtype: always `DType::U8` regardless of input
    /// dtype.
    fn binary_compare_op(&self, name: &'static str, op: Op, other: &Tensor) -> Tensor {
        assert!(
            Arc::ptr_eq(&self.graph, &other.graph),
            "{name}: tensors must live on the same graph",
        );
        assert_eq!(
            self.dtype(),
            other.dtype(),
            "{name}: dtype mismatch: lhs={:?}, rhs={:?}",
            self.dtype(),
            other.dtype(),
        );
        assert_eq!(
            self.shape().dims(),
            other.shape().dims(),
            "{name}: shape mismatch: lhs={:?}, rhs={:?}",
            self.shape().dims(),
            other.shape().dims(),
        );
        let shape = self.shape();
        let id = self.graph.write().unwrap().push(Node {
            op,
            inputs: vec![self.id, other.id],
            shape,
            dtype: DType::U8,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Build the backward graph from `self`, treating `self` as the output
    /// whose gradient drives the rest of the computation.
    ///
    /// Walks the forward graph in reverse topological order, applies the
    /// per-op gradient rule to each node, and emits new graph nodes for
    /// the gradient with respect to every node reachable from `self`. The
    /// returned [`GradMap`] lets you look up the gradient tensor for any
    /// forward tensor from the same graph.
    ///
    /// The initial upstream gradient is a ones tensor of the same shape
    /// and dtype as `self` — the conventional `dL/dL = 1` seed for
    /// backprop. For a scalar loss this is `[1.0]`; for a non-scalar
    /// output it is "each element of the output contributes with weight 1
    /// to the total derivative."
    ///
    /// ## Supported ops in the MVP
    ///
    /// `Const`, `Add`, `Mul`, `MatMul`, `Transpose`, `Sqr`, `Exp` all have
    /// gradient rules implemented. `Relu` panics — its gradient needs a
    /// `Step`/`Where`/`Sign` op that is not yet in the catalog and will
    /// land in a follow-up once an indicator-valued op is defined.
    ///
    /// ## Ownership and the forward graph
    ///
    /// Backward extends the same graph that held the forward pass. The
    /// gradient nodes live alongside the forward nodes in one arena. After
    /// `backward()`, realizing a gradient tensor re-executes whatever part
    /// of the forward pass the gradient depends on — the executor walks
    /// the combined graph as one.
    pub fn backward(&self) -> GradMap {
        let graph_handle = self.graph.clone();

        // Compute topological order of reachable nodes, then reverse it
        // so we walk from the root (output) toward the leaves (inputs).
        let mut order = topo_order(&graph_handle.read().unwrap(), self.id);
        order.reverse();

        // Initial upstream gradient for the root: ones tensor of matching
        // shape and dtype. For a scalar loss this is [1.0]; for vector
        // outputs it seeds each element with weight 1.
        let (root_shape, root_dtype) = {
            let g = graph_handle.read().unwrap();
            let n = g.node(self.id);
            (n.shape.clone(), n.dtype)
        };
        let ones_id = build_ones(&graph_handle, root_shape, root_dtype);

        let mut upstream: HashMap<NodeId, NodeId> = HashMap::new();
        upstream.insert(self.id, ones_id);

        for id in order {
            let up_id = match upstream.get(&id).copied() {
                Some(v) => v,
                None => continue, // unreachable from the root, no gradient to propagate
            };
            // Snapshot the op and its input IDs so we can drop the read
            // borrow before taking a mutable borrow to append new nodes.
            let (op, inputs) = {
                let g = graph_handle.read().unwrap();
                let node = g.node(id);
                (node.op.clone(), node.inputs.clone())
            };

            // Symbolic-autograd dispatch (Phase 6d Track 2). If a
            // `GradientRule` is registered for this op, use it. Ops
            // that haven't migrated yet fall through to the legacy
            // inline `match` below.
            if let Some(grads) = crate::grad::dispatch_gradient(
                &graph_handle, &op, &inputs, id, up_id,
            ) {
                debug_assert_eq!(
                    grads.len(), inputs.len(),
                    "GradientRule for {:?} returned {} gradients but op has {} inputs",
                    op.short_name(), grads.len(), inputs.len(),
                );
                for (input_id, grad) in inputs.iter().zip(grads.iter()) {
                    if let Some(g) = grad {
                        accumulate_grad(&mut upstream, *input_id, *g, &graph_handle);
                    }
                }
                continue;
            }

            match op {
                Op::Const => {
                    // Leaf. The upstream value has already been stored and
                    // is what `GradMap::get` will return for this input.
                }
                Op::Add => {
                    // d(a + b)/da = 1, d(a + b)/db = 1.
                    // Upstream flows unchanged into both inputs.
                    accumulate_grad(&mut upstream, inputs[0], up_id, &graph_handle);
                    accumulate_grad(&mut upstream, inputs[1], up_id, &graph_handle);
                }
                Op::Mul => {
                    // d(a * b)/da = b, d(a * b)/db = a.
                    // Upstream * b goes to a; upstream * a goes to b.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, b],
                        a_shape,
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, a],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::MatMul => {
                    // Forward: Y = A @ B,  A: [m, k], B: [k, n], Y: [m, n]
                    // Backward: dA = dY @ B^T,  dB = A^T @ dY.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    // B^T: shape [n, k]
                    let b_t_shape = transposed_shape(&b_shape);
                    let b_t = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![b],
                        b_t_shape,
                        dtype,
                    );
                    // dA = upstream @ B^T, shape [m, k] = a_shape
                    let grad_a = push_node(
                        &graph_handle,
                        Op::MatMul,
                        vec![up_id, b_t],
                        a_shape.clone(),
                        dtype,
                    );
                    // A^T: shape [k, m]
                    let a_t_shape = transposed_shape(&a_shape);
                    let a_t = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![a],
                        a_t_shape,
                        dtype,
                    );
                    // dB = A^T @ upstream, shape [k, n] = b_shape
                    let grad_b = push_node(
                        &graph_handle,
                        Op::MatMul,
                        vec![a_t, up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Transpose => {
                    // d(x^T)/dx is a transpose on the way back.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Transpose,
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Permute(ref axes) => {
                    // Backward: apply the inverse permutation to upstream.
                    // If forward permutes with `axes` (out[i] = in[axes[i]]),
                    // the inverse `inv` satisfies `inv[axes[i]] = i`.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let mut inv = vec![0_usize; axes.len()];
                    for (i, &ax) in axes.iter().enumerate() {
                        inv[ax] = i;
                    }
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Permute(inv),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sqr => {
                    // d(x^2)/dx = 2x. Expressed as `x + x` to avoid
                    // needing a scalar constant broadcast — both sums
                    // have the same shape as x, the existing Add op
                    // handles them natively.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_x = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![x, x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, two_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Exp => {
                    // d(e^x)/dx = e^x, which is the forward output — i.e.
                    // this node `id`. Reuse it directly instead of
                    // recomputing.
                    let x = inputs[0];
                    let out_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, id],
                        out_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sub => {
                    // d(a - b)/da = 1, d(a - b)/db = -1.
                    let a = inputs[0];
                    let b = inputs[1];
                    accumulate_grad(&mut upstream, a, up_id, &graph_handle);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, b);
                    let neg_up = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, b, neg_up, &graph_handle);
                }
                Op::Div => {
                    // y = a / b
                    // dy/da = 1/b     →  grad_a = upstream / b
                    // dy/db = -a/b²   →  grad_b = -(upstream * a / (b*b))
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    // grad_a
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, b],
                        a_shape.clone(),
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    // grad_b: -(upstream * a / (b*b))
                    let b_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![b],
                        b_shape.clone(),
                        dtype,
                    );
                    let up_times_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, a],
                        a_shape,
                        dtype,
                    );
                    let quotient = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_times_a, b_sq],
                        b_shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![quotient],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Neg => {
                    // d(-x)/dx = -1. grad_x = -upstream.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sqrt => {
                    // y = sqrt(x), dy/dx = 1/(2*sqrt(x)) = 1/(2y).
                    // grad_x = upstream / (y + y). Reuse the forward node.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_y = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![id, id],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, two_y],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Log => {
                    // d(ln(x))/dx = 1/x. grad_x = upstream / x.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![up_id, x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sin => {
                    // d(sin(x))/dx = cos(x). grad_x = upstream * cos(x).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let cos_x = push_node(
                        &graph_handle,
                        Op::Cos,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, cos_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Cos => {
                    // d(cos(x))/dx = -sin(x). grad_x = -(upstream * sin(x)).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let sin_x = push_node(
                        &graph_handle,
                        Op::Sin,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let up_sin = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, sin_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_sin],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Tanh => {
                    // y = tanh(x), dy/dx = 1 - y². grad_x = upstream * (1 - y*y).
                    // Build: ones_of_y_shape - y*y, then multiply by upstream.
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let y_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![id],
                        y_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_sq = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, y_sq],
                        y_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, one_minus_sq],
                        y_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Sigmoid => {
                    // y = sigmoid(x), dy/dx = y * (1 - y).
                    // grad_x = upstream * y * (1 - y).
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_y = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, id],
                        y_shape.clone(),
                        dtype,
                    );
                    let y_times_1my = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![id, one_minus_y],
                        y_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, y_times_1my],
                        y_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Silu => {
                    // y = x * sigmoid(x)
                    // dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    //       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    //
                    // Simpler: let s = sigmoid(x). dy/dx = s + y * (1 - s),
                    // where y = x * s is the forward output (this node).
                    // grad_x = upstream * (s + y * (1 - s)).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let s = push_node(
                        &graph_handle,
                        Op::Sigmoid,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let one_minus_s = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, s],
                        x_shape.clone(),
                        dtype,
                    );
                    let y_times_1ms = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![id, one_minus_s],
                        x_shape.clone(),
                        dtype,
                    );
                    let inner = push_node(
                        &graph_handle,
                        Op::Add,
                        vec![s, y_times_1ms],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, inner],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Gelu => {
                    // GELU's tanh-approximation backward is non-trivial.
                    // For the MVP, compute it numerically stable via
                    // central differences — no, that's wrong for
                    // backward. The proper rule differentiates
                    // `0.5 x (1 + tanh(c * (x + 0.044715 x³)))` by chain
                    // rule. That requires several new intermediate
                    // nodes. Implemented below:
                    //
                    // Let u = c * (x + 0.044715 x³), t = tanh(u).
                    // y = 0.5 x (1 + t)
                    // du/dx = c * (1 + 3 * 0.044715 * x²)
                    //       = c + 3 c * 0.044715 * x²
                    // dt/dx = (1 - t²) * du/dx
                    // dy/dx = 0.5 * (1 + t) + 0.5 * x * dt/dx
                    //
                    // This is enough ops that it deserves a dedicated
                    // backward node. For the MVP, panic — users who need
                    // GELU backward should use Silu, which we just
                    // implemented properly.
                    panic!(
                        "backward: Gelu gradient is not yet supported. \
                         Use Silu for differentiable training; Gelu is \
                         currently inference-only."
                    );
                }
                Op::Relu => {
                    // d(relu(x))/dx = step(x). grad_x = upstream * step(x).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let step_x = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, step_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Step => {
                    // d(step(x))/dx is 0 almost everywhere (and a Dirac
                    // delta at 0 which has no finite representation). The
                    // standard move is to treat the derivative as 0 and
                    // stop propagation — Step's backward is silently a
                    // no-op. Callers that need a smooth surrogate should
                    // use Sigmoid or Tanh instead.
                }
                Op::Recip => {
                    // y = 1/x, dy/dx = -1/x² = -y².
                    // grad_x = -upstream * y * y. Reuse forward output (id)
                    // so we don't recompute the reciprocal.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let y_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![id],
                        x_shape.clone(),
                        dtype,
                    );
                    let up_y_sq = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, y_sq],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![up_y_sq],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Abs => {
                    // y = |x|, dy/dx = sign(x), with sign(0)=0 by
                    // subgradient convention. Simplified to a direct
                    // `Op::Sign` use after Op::Sign landed in PR B2;
                    // the previous form synthesized `step(x) - step(-x)`
                    // (5 backward nodes) but Sign expresses the same
                    // function in 1 node, so the chain shrinks to
                    // `Mul(upstream, Sign(x))` — 2 nodes total.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let sign_x = push_node(
                        &graph_handle,
                        Op::Sign,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, sign_x],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Where => {
                    // Handled by `WhereRule` via `dispatch_gradient`;
                    // this arm only exists for exhaustiveness.
                }
                Op::Floor | Op::Ceil | Op::Round | Op::Sign => {
                    // d(floor(x))/dx, d(ceil(x))/dx, d(round(x))/dx,
                    // d(sign(x))/dx are all 0 almost everywhere (with
                    // a Dirac train at integer / half-integer / zero
                    // that has no finite representation). Treat the
                    // derivative as 0 and stop propagation — mirrors
                    // `Op::Step`'s backward.
                }
                Op::Erf => {
                    // y = erf(x). dy/dx = (2/√π) * exp(-x²).
                    // grad_x = upstream * (2/√π) * exp(-x²).
                    // Decomposes to: Sqr(x) → Neg → Exp → MulScalar(2/√π)
                    //                → Mul(upstream, .).
                    // 2/√π = 1.1283791670955125738961589031...
                    const TWO_OVER_SQRT_PI: f64 =
                        1.128_379_167_095_512_6_f64;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_sq = push_node(
                        &graph_handle,
                        Op::Sqr,
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let neg_x_sq = push_node(
                        &graph_handle,
                        Op::Neg,
                        vec![x_sq],
                        x_shape.clone(),
                        dtype,
                    );
                    let exp_neg_x_sq = push_node(
                        &graph_handle,
                        Op::Exp,
                        vec![neg_x_sq],
                        x_shape.clone(),
                        dtype,
                    );
                    let scaled = push_node(
                        &graph_handle,
                        Op::MulScalar(TWO_OVER_SQRT_PI),
                        vec![exp_neg_x_sq],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, scaled],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Flip { dim } => {
                    // y = flip(x, dim). Backward is another Flip on
                    // the same dim — Flip is its own inverse
                    // (involutive). One backward node, exact gradient.
                    let dim = dim;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle, Op::Flip { dim },
                        vec![up_id], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Pad { padding, mode, value: _ } => {
                    // Backward delegates to Op::PadBackward — a single
                    // node that handles all three modes uniformly via
                    // its kernel. Constant: slice the unpadded region.
                    // Reflect/Replicate: accumulate gradient at the
                    // mirrored / replicated positions.
                    let padding = padding.clone();
                    let mode = mode;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::PadBackward { in_shape: x_shape.clone(), padding, mode },
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::PadBackward { .. } => {
                    // Backward helper — typically a leaf in autograd
                    // (the upstream gradient itself never gets a
                    // higher-order backward in v1). No-op arm for
                    // exhaustiveness; if higher-order autograd needs
                    // PadBackward's backward, that's its own scope.
                }
                Op::CumSum { dim } => {
                    // y[..., i, ...] = sum_{k=0..=i} x[..., k, ...]
                    // dL/dx[..., i, ...] = sum_{k=i..n} dL/dy[..., k, ...]
                    // i.e. reverse cumsum. Express via Flip → CumSum → Flip.
                    let dim = dim;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let flip1 = push_node(
                        &graph_handle, Op::Flip { dim },
                        vec![up_id], x_shape.clone(), dtype,
                    );
                    let cs = push_node(
                        &graph_handle, Op::CumSum { dim },
                        vec![flip1], x_shape.clone(), dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle, Op::Flip { dim },
                        vec![cs], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Roll { dim, shift } => {
                    // y = roll(x, dim, shift). Backward is the opposite
                    // shift along the same dim. One backward node.
                    let dim = dim;
                    let shift = shift;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle, Op::Roll { dim, shift: -shift },
                        vec![up_id], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Triu { diagonal } => {
                    // y = triu(x, diagonal). Mask is a binary indicator
                    // — gradient passes through kept positions and is
                    // zero on the masked half. Same op on the upstream.
                    let diagonal = diagonal;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle, Op::Triu { diagonal },
                        vec![up_id], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Tril { diagonal } => {
                    // Mirror of Op::Triu — same mask gradient.
                    let diagonal = diagonal;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle, Op::Tril { diagonal },
                        vec![up_id], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::LogSoftmaxLastDim => {
                    // grad_x = upstream - exp(y) * sum(upstream, last_dim, keepdim).
                    // Folded into Op::LogSoftmaxLastDimBackward — takes
                    // (forward_output, upstream), avoids re-evaluating
                    // the softmax outside the kernel.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle, Op::LogSoftmaxLastDimBackward,
                        vec![id, up_id], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::LogSoftmaxLastDimBackward => {
                    // Backward helper — no higher-order autograd in v1.
                }
                Op::MaskedFill { value: _ } => {
                    // y = mask ? value : x. dy/dx = (mask == 0).
                    // grad_x = MaskedFill(upstream, mask, value=0_of_dtype).
                    // The mask is integer-only; no gradient flows to it.
                    let x = inputs[0];
                    let mask = inputs[1];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let zero = fuel_ir::Scalar::zero(dtype);
                    let grad_x = push_node(
                        &graph_handle, Op::MaskedFill { value: zero },
                        vec![up_id, mask], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                    // No gradient to mask: it's discrete.
                }
                Op::Rem => {
                    // y = a - floor(a/b) * b (PyTorch convention).
                    // d/da = 1 (the floor term is treated as a constant
                    //          w.r.t. a almost everywhere; the actual
                    //          derivative has a Dirac component at
                    //          integer a/b ratios that we drop, matching
                    //          PyTorch's autograd convention).
                    // d/db = -floor(a/b)
                    // grad_a = upstream
                    // grad_b = upstream * (-floor(a/b))
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    accumulate_grad(&mut upstream, a, up_id, &graph_handle);
                    let div_ab = push_node(
                        &graph_handle, Op::Div,
                        vec![a, b], a_shape.clone(), dtype,
                    );
                    let floor_div = push_node(
                        &graph_handle, Op::Floor,
                        vec![div_ab], a_shape.clone(), dtype,
                    );
                    let neg_floor = push_node(
                        &graph_handle, Op::Neg,
                        vec![floor_div], a_shape.clone(), dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle, Op::Mul,
                        vec![up_id, neg_floor], a_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Rsqrt => {
                    // y = x^(-1/2). dy/dx = -0.5 * x^(-3/2).
                    // Identity: y² = 1/x, so x^(-3/2) = y/x = y * y² = y³.
                    // grad_x = -0.5 * upstream * y³.
                    // Reuses the forward output id for y; never touches
                    // x directly (avoids the divide-by-zero singularity
                    // that the obvious `-0.5 * upstream * y / x` form
                    // would have).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let y_sq = push_node(
                        &graph_handle, Op::Sqr,
                        vec![id], x_shape.clone(), dtype,
                    );
                    let y_cu = push_node(
                        &graph_handle, Op::Mul,
                        vec![id, y_sq], x_shape.clone(), dtype,
                    );
                    let scaled = push_node(
                        &graph_handle, Op::MulScalar(-0.5),
                        vec![y_cu], x_shape.clone(), dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle, Op::Mul,
                        vec![up_id, scaled], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Pow => {
                    // y = pow(a, b).
                    // d/da = b * pow(a, b-1)
                    // d/db = y * ln(a)
                    // grad_a = upstream * b * pow(a, b-1)
                    // grad_b = upstream * y * ln(a)
                    // The forward output `id` IS y; we reuse it.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    // d/da branch: pow(a, b - 1)
                    let b_minus_one = push_node(
                        &graph_handle, Op::AddScalar(-1.0),
                        vec![b], a_shape.clone(), dtype,
                    );
                    let pow_a_bm1 = push_node(
                        &graph_handle, Op::Pow,
                        vec![a, b_minus_one], a_shape.clone(), dtype,
                    );
                    let b_times_pow = push_node(
                        &graph_handle, Op::Mul,
                        vec![b, pow_a_bm1], a_shape.clone(), dtype,
                    );
                    let grad_a = push_node(
                        &graph_handle, Op::Mul,
                        vec![up_id, b_times_pow], a_shape.clone(), dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    // d/db branch: y * ln(a)
                    let log_a = push_node(
                        &graph_handle, Op::Log,
                        vec![a], a_shape.clone(), dtype,
                    );
                    let y_log_a = push_node(
                        &graph_handle, Op::Mul,
                        vec![id, log_a], a_shape.clone(), dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle, Op::Mul,
                        vec![up_id, y_log_a], a_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::GeluErf => {
                    // y = 0.5 * x * (1 + erf(x/√2)).
                    // dy/dx = Φ(x) + x · φ(x)
                    //       = 0.5 * (1 + erf(x/√2))
                    //       + x * (1/√(2π)) * exp(-x²/2)
                    // where Φ is the standard-normal CDF (= y/x for x≠0;
                    // we recompute it explicitly so the backward stays
                    // safe at x=0) and φ is the PDF.
                    //
                    // Chain (12 nodes):
                    //   x_over_sqrt2 = MulScalar(1/√2)(x)
                    //   erf_arg      = Erf(x_over_sqrt2)
                    //   plus_one     = AddScalar(1.0)(erf_arg)
                    //   cdf_term     = MulScalar(0.5)(plus_one)
                    //   x_sq         = Sqr(x)
                    //   half_x_sq    = MulScalar(0.5)(x_sq)
                    //   neg_half_x_sq= Neg(half_x_sq)
                    //   exp_term     = Exp(neg_half_x_sq)
                    //   phi          = MulScalar(1/√(2π))(exp_term)
                    //   pdf_term     = Mul(x, phi)
                    //   d_gelu       = Add(cdf_term, pdf_term)
                    //   grad_x       = Mul(upstream, d_gelu)
                    const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
                    // 1/√(2π) = 0.39894228040143267793994605993...
                    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7_f64;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    // CDF half: 0.5 * (1 + erf(x/√2)).
                    let x_over_sqrt2 = push_node(
                        &graph_handle, Op::MulScalar(INV_SQRT_2),
                        vec![x], x_shape.clone(), dtype,
                    );
                    let erf_arg = push_node(
                        &graph_handle, Op::Erf,
                        vec![x_over_sqrt2], x_shape.clone(), dtype,
                    );
                    let plus_one = push_node(
                        &graph_handle, Op::AddScalar(1.0),
                        vec![erf_arg], x_shape.clone(), dtype,
                    );
                    let cdf_term = push_node(
                        &graph_handle, Op::MulScalar(0.5),
                        vec![plus_one], x_shape.clone(), dtype,
                    );
                    // PDF half: x * (1/√(2π)) * exp(-x²/2).
                    let x_sq = push_node(
                        &graph_handle, Op::Sqr,
                        vec![x], x_shape.clone(), dtype,
                    );
                    let half_x_sq = push_node(
                        &graph_handle, Op::MulScalar(0.5),
                        vec![x_sq], x_shape.clone(), dtype,
                    );
                    let neg_half_x_sq = push_node(
                        &graph_handle, Op::Neg,
                        vec![half_x_sq], x_shape.clone(), dtype,
                    );
                    let exp_term = push_node(
                        &graph_handle, Op::Exp,
                        vec![neg_half_x_sq], x_shape.clone(), dtype,
                    );
                    let phi = push_node(
                        &graph_handle, Op::MulScalar(INV_SQRT_2PI),
                        vec![exp_term], x_shape.clone(), dtype,
                    );
                    let pdf_term = push_node(
                        &graph_handle, Op::Mul,
                        vec![x, phi], x_shape.clone(), dtype,
                    );
                    // Sum the two halves and apply upstream.
                    let d_gelu = push_node(
                        &graph_handle, Op::Add,
                        vec![cdf_term, pdf_term], x_shape.clone(), dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle, Op::Mul,
                        vec![up_id, d_gelu], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Equal | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                    // Comparison family: handled by `NoGradientBinaryRule`
                    // via `dispatch_gradient`. The `if let Some(grads) =
                    // dispatch_gradient(...) { continue; }` block above
                    // intercepts before we get here. This arm exists only
                    // for exhaustiveness; it should be unreachable in
                    // practice. Keeping it as a no-op rather than
                    // `unreachable!()` defends against a future hand-edit
                    // that removes the dispatcher arm by mistake — the
                    // graph still terminates traversal cleanly.
                }
                Op::Cast(_) => {
                    // Forward: y = cast(x, target_dtype).
                    // Backward: dL/dx = cast(upstream, source_dtype).
                    let x = inputs[0];
                    let x_dtype = node_dtype(&graph_handle, x);
                    let x_shape = node_shape(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Cast(x_dtype),
                        vec![up_id],
                        x_shape,
                        x_dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SumAll => {
                    // Forward: y = sum(x), scalar.
                    // Backward: dL/dx = broadcast_to(upstream, x.shape).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MeanAll => {
                    // Forward: y = mean(x) = sum(x) / n, scalar.
                    // Backward: dL/dx = broadcast_to(upstream / n, x.shape).
                    // We build `ones / n` as a constant of shape x.shape
                    // and multiply by broadcast_to(upstream, x.shape).
                    // Simpler: broadcast upstream, then divide by an n-const.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let n_elem = x_shape.elem_count();
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape.clone(),
                        dtype,
                    );
                    let n_const =
                        build_filled_const(&graph_handle, x_shape.clone(), dtype, n_elem as f64);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![broadcast_up, n_const],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::BroadcastTo(_) => {
                    // Forward: y = broadcast_to(x, target).
                    // Backward: dL/dx = reduce_sum_to(upstream, x.shape) —
                    // sum away every dim that was expanded or newly added.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::ReduceSumTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Reshape(_) => {
                    // Forward: y = reshape(x, target). Data unchanged.
                    // Backward: reshape upstream back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Reshape(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Contiguize => {
                    // Forward: y = contiguize(x). Identity at the
                    // value level; only the layout differs. Backward:
                    // pass upstream through unchanged. The
                    // downstream `accumulate_grad` will sum this
                    // into x's gradient regardless of x's own layout.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::Unsqueeze { dim: _ } => {
                    // Forward: y = unsqueeze(x, dim) — inserts a size-1
                    // axis without touching bytes. Backward: drop that
                    // axis. Bytes are unchanged either direction; a
                    // Reshape to x.shape is the simplest expression of
                    // "same bytes, different rank" that the executor
                    // handles natively.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Reshape(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Squeeze { dim } => {
                    // Forward: y = squeeze(x, dim) — drops a size-1 axis,
                    // metadata-only. Backward: re-insert the axis at the
                    // same position via Unsqueeze. Both directions are
                    // metadata-only views (no bytes touched).
                    let dim = dim;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Unsqueeze { dim },
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::ReduceSumTo(_) => {
                    // Forward: y = reduce_sum_to(x, target).
                    // Backward: broadcast upstream back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MaxAll | Op::MinAll => {
                    // For y = max(x) (or min), the gradient is an
                    // indicator at the argmax positions, distributed
                    // across ties. Expressed without a dedicated argmax
                    // op using the Step trick:
                    //
                    //   broadcasted_y = broadcast_to(y, x.shape)
                    //   diff = broadcasted_y - x   (for Max)
                    //        | x - broadcasted_y   (for Min)
                    //   indicator = ones - step(diff)
                    //   grad_x    = broadcast_to(upstream, x.shape) * indicator
                    //
                    // At argmax positions `diff == 0`, `step(0) == 0`, so
                    // the indicator is 1. Elsewhere `diff > 0` so step is
                    // 1 and the indicator is 0. Ties share the gradient.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let is_max = matches!(op, Op::MaxAll);
                    let broadcasted_y = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![id],
                        x_shape.clone(),
                        dtype,
                    );
                    let diff = if is_max {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![broadcasted_y, x],
                            x_shape.clone(),
                            dtype,
                        )
                    } else {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![x, broadcasted_y],
                            x_shape.clone(),
                            dtype,
                        )
                    };
                    let step_diff = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let indicator = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, step_diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_id],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![broadcast_up, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SumDim(dim) => {
                    // Insert a size-1 at `dim` via reshape, then broadcast
                    // back to x.shape.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let reshaped = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![reshaped],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MeanDim(dim) => {
                    // SumDim-style backward, then divide by the reduced dim size.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let reduced_size = x_shape.dims()[dim];
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let reshaped = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let broadcast = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![reshaped],
                        x_shape.clone(),
                        dtype,
                    );
                    let n_const = build_filled_const(
                        &graph_handle,
                        x_shape.clone(),
                        dtype,
                        reduced_size as f64,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Div,
                        vec![broadcast, n_const],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::MaxDim(dim) | Op::MinDim(dim) => {
                    // Same indicator trick as MaxAll/MinAll, but apply the
                    // "reshape to keepdim then broadcast" pattern to both
                    // the forward output and upstream before multiplying.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let is_max = matches!(op, Op::MaxDim(_));
                    let keepdim_shape = reshape_to_keepdim(x_shape.dims(), dim);
                    let y_keepdim = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![id],
                        keepdim_shape.clone(),
                        dtype,
                    );
                    let broadcasted_y = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![y_keepdim],
                        x_shape.clone(),
                        dtype,
                    );
                    let diff = if is_max {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![broadcasted_y, x],
                            x_shape.clone(),
                            dtype,
                        )
                    } else {
                        push_node(
                            &graph_handle,
                            Op::Sub,
                            vec![x, broadcasted_y],
                            x_shape.clone(),
                            dtype,
                        )
                    };
                    let step_diff = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let indicator = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, step_diff],
                        x_shape.clone(),
                        dtype,
                    );
                    let up_keepdim = push_node(
                        &graph_handle,
                        Op::Reshape(keepdim_shape.clone()),
                        vec![up_id],
                        keepdim_shape,
                        dtype,
                    );
                    let broadcast_up = push_node(
                        &graph_handle,
                        Op::BroadcastTo(x_shape.clone()),
                        vec![up_keepdim],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![broadcast_up, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                // Phase 7.6 step 5 (2026-05-11): the legacy
                // `Op::SoftmaxLastDim` / `Op::LayerNormLastDim` /
                // `Op::RmsNormLastDim` / `Op::Rope` /
                // `Op::Conv2D` / `Op::FusedLinear` arms have been
                // dropped together with the variants themselves; the
                // `Op::Fused(fid, _)` arm below dispatches per id and
                // is the single source of truth for migrated forward
                // ops' backward.
                Op::ArgMaxDim(_) | Op::ArgMinDim(_) => {
                    // Argmax/argmin produce integer indices, which are
                    // non-differentiable. Panic rather than silently
                    // zero-propagating — if the user has one of these in
                    // their differentiated graph, they've almost
                    // certainly made a mistake.
                    panic!(
                        "backward: ArgMaxDim/ArgMinDim produce integer \
                         indices and cannot be differentiated through. \
                         Use MaxDim/MinDim if you want a differentiable \
                         selection."
                    );
                }
                Op::ReduceMaxTo(_) => {
                    // Forward: y = reduce_max_to(x, target).
                    // Backward: route upstream to position(s) where
                    // x equals its per-window max via the fused
                    // backward helper Op::Fused(REDUCE_MAX_TO_BACKWARD,
                    // _) (takes (x, upstream) and emits grad_x of
                    // x.shape). Tied maxes share the gradient equally
                    // (fair-share subgradient). Note this is the
                    // backward of a *primitive* (Op::ReduceMaxTo),
                    // not a fused forward — no BackwardKind::Fused
                    // edge in the registry drives this; the explicit
                    // emission is the source of truth.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Fused(
                            crate::registry::FusedOps::REDUCE_MAX_TO_BACKWARD,
                            crate::registry::FusedOpParams::ReduceMaxToBackward,
                        ),
                        vec![x, up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                // Phase 7.6 step 5 (final, 2026-05-11): legacy
                // `Op::QMatMul` arm dropped with the variant; QMatMul
                // backward panics via the Op::Fused arm below.
                // Phase 7.6 step 5 (2026-05-11): the legacy
                // `Op::SoftmaxLastDimBackward | Op::LayerNormLastDimBackward |
                // Op::RmsNormLastDimBackward | Op::ReduceMaxToBackward`
                // higher-order panic arm has been dropped together
                // with the variants themselves. The Op::Fused arm
                // below panics for those four ids.
                Op::IndexSelect { dim } => {
                    // Forward: out = index_select(data, dim, indices).
                    // Backward: grad_data = index_add(zeros_like(data), dim, indices, upstream).
                    let data = inputs[0];
                    let idx = inputs[1];
                    let data_shape = node_shape(&graph_handle, data);
                    let dtype = node_dtype(&graph_handle, data);
                    let zeros = build_filled_const(
                        &graph_handle,
                        data_shape.clone(),
                        dtype,
                        0.0,
                    );
                    let grad_data = push_node(
                        &graph_handle,
                        Op::IndexAdd { dim },
                        vec![zeros, idx, up_id],
                        data_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, data, grad_data, &graph_handle);
                    // Index tensor is itself non-differentiable (integer dtype).
                }
                Op::Gather { dim } => {
                    // Forward: out = gather(data, dim, indices).
                    // Backward: grad_data = scatter_add(zeros_like(data), dim, indices, upstream).
                    let data = inputs[0];
                    let idx = inputs[1];
                    let data_shape = node_shape(&graph_handle, data);
                    let dtype = node_dtype(&graph_handle, data);
                    let zeros = build_filled_const(
                        &graph_handle,
                        data_shape.clone(),
                        dtype,
                        0.0,
                    );
                    let grad_data = push_node(
                        &graph_handle,
                        Op::ScatterAdd { dim },
                        vec![zeros, idx, up_id],
                        data_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, data, grad_data, &graph_handle);
                }
                Op::IndexAdd { dim } => {
                    // Forward: out = index_add(base, dim, indices, src).
                    // Backward:
                    //   grad_base = upstream (pass-through — base flows
                    //               through the non-indexed positions and
                    //               the accumulated positions don't zero
                    //               it out, they add to it)
                    //   grad_src  = index_select(upstream, dim, indices)
                    //               (the pieces of upstream that came from
                    //                src in the forward).
                    let base = inputs[0];
                    let idx = inputs[1];
                    let src = inputs[2];
                    let src_shape = node_shape(&graph_handle, src);
                    let dtype = node_dtype(&graph_handle, src);
                    // grad_base: upstream passes through.
                    accumulate_grad(&mut upstream, base, up_id, &graph_handle);
                    // grad_src: select from upstream at the indices.
                    let grad_src = push_node(
                        &graph_handle,
                        Op::IndexSelect { dim },
                        vec![up_id, idx],
                        src_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, src, grad_src, &graph_handle);
                }
                Op::Concat { dim } => {
                    // Forward: out = concat(a, b, dim).
                    // Backward: split upstream along `dim` at a.size(dim),
                    // routing the left slice to a and the right slice to b.
                    let a = inputs[0];
                    let b = inputs[1];
                    let a_shape = node_shape(&graph_handle, a);
                    let b_shape = node_shape(&graph_handle, b);
                    let dtype = node_dtype(&graph_handle, a);
                    let a_len = a_shape.dims()[dim];
                    let b_len = b_shape.dims()[dim];
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Slice { dim, start: 0, len: a_len },
                        vec![up_id],
                        a_shape,
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Slice { dim, start: a_len, len: b_len },
                        vec![up_id],
                        b_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Slice { dim, start, len: _ } => {
                    // Forward: out = slice(x, dim, start, len).
                    // Backward: build a zeros tensor of x.shape, then
                    // scatter the upstream into the slice range. We use
                    // a concat of three pieces: left zeros, upstream,
                    // right zeros (along `dim`).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_dims = x_shape.dims();
                    let dim_size = x_dims[dim];
                    let up_len = x_dims[dim]
                        .min(node_shape(&graph_handle, up_id).dims()[dim]);
                    let left_len = start;
                    let right_len = dim_size - start - up_len;
                    // Build left zero pad if needed.
                    let mut pieces: Vec<NodeId> = Vec::new();
                    if left_len > 0 {
                        let mut left_dims: Vec<usize> = x_dims.to_vec();
                        left_dims[dim] = left_len;
                        let left_shape = Shape::from_dims(&left_dims);
                        pieces.push(build_filled_const(
                            &graph_handle,
                            left_shape,
                            dtype,
                            0.0,
                        ));
                    }
                    pieces.push(up_id);
                    if right_len > 0 {
                        let mut right_dims: Vec<usize> = x_dims.to_vec();
                        right_dims[dim] = right_len;
                        let right_shape = Shape::from_dims(&right_dims);
                        pieces.push(build_filled_const(
                            &graph_handle,
                            right_shape,
                            dtype,
                            0.0,
                        ));
                    }
                    // Fold concat-left-to-right.
                    let mut current = pieces[0];
                    let mut current_dims: Vec<usize> = {
                        let n = graph_handle.read().unwrap();
                        n.node(current).shape.dims().to_vec()
                    };
                    for &next in &pieces[1..] {
                        let next_dims: Vec<usize> = {
                            let n = graph_handle.read().unwrap();
                            n.node(next).shape.dims().to_vec()
                        };
                        let mut combined = current_dims.clone();
                        combined[dim] = current_dims[dim] + next_dims[dim];
                        let combined_shape = Shape::from_dims(&combined);
                        current = push_node(
                            &graph_handle,
                            Op::Concat { dim },
                            vec![current, next],
                            combined_shape,
                            dtype,
                        );
                        current_dims = combined;
                    }
                    accumulate_grad(&mut upstream, x, current, &graph_handle);
                }
                Op::AddScalar(_) => {
                    // y = x + c, dy/dx = 1. Upstream passes through.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::MulScalar(c) => {
                    // y = x * c, dy/dx = c. Multiply upstream by c via
                    // another MulScalar node.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::MulScalar(c),
                        vec![up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::PowI(n) => {
                    // y = x^n, dy/dx = n * x^(n-1).
                    // grad_x = upstream * n * x^(n-1). Emitted as a
                    // single fused PowIBackward node; the executor
                    // routes it to baracuda's `unary_powi_backward_*`
                    // kernel (alpha.31) on CUDA, or falls back to the
                    // primitive decomposition (PowI(n-1) → MulScalar →
                    // Mul) on backends that haven't registered the
                    // fused kernel.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Fused(
                            crate::registry::FusedOps::POWI_BACKWARD,
                            crate::registry::FusedOpParams::PowIBackward { exp: n },
                        ),
                        vec![x, up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Clamp { min, max } => {
                    // Subgradient: 1 where min <= x <= max, 0 elsewhere.
                    // Expressed via (step(x - min) - step(x - max))
                    // which is 1 in the interior (x > min and x <= max...
                    // actually step(0) = 0, so the right end is a bit
                    // tricky). For a reference implementation the
                    // simplest approach is a dedicated "clamp backward"
                    // indicator. We build it out of the existing ops:
                    //
                    //   lower_ok = step(x - min)   // 1 where x > min
                    //   upper_ok = step(max - x)   // 1 where x < max
                    //   indicator = lower_ok * upper_ok
                    //   grad_x = upstream * indicator
                    //
                    // This underestimates the gradient on the exact
                    // boundary by a measure-zero amount — standard.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let min_shifted = push_node(
                        &graph_handle,
                        Op::AddScalar(-min),
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let lower_ok = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![min_shifted],
                        x_shape.clone(),
                        dtype,
                    );
                    let max_minus_x = push_node(
                        &graph_handle,
                        Op::MulScalar(-1.0),
                        vec![x],
                        x_shape.clone(),
                        dtype,
                    );
                    let max_minus_x = push_node(
                        &graph_handle,
                        Op::AddScalar(max),
                        vec![max_minus_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let upper_ok = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![max_minus_x],
                        x_shape.clone(),
                        dtype,
                    );
                    let indicator = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![lower_ok, upper_ok],
                        x_shape.clone(),
                        dtype,
                    );
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, indicator],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::Maximum => {
                    // y_i = max(a_i, b_i). Gradient flows to whichever
                    // input was the max at that position:
                    //   mask_a = step(a - b)   (1 where a > b, else 0)
                    //   mask_b = 1 - mask_a    (approximately — at ties
                    //                          gradient goes to b due to
                    //                          step(0) = 0)
                    //   grad_a = upstream * mask_a
                    //   grad_b = upstream * mask_b
                    let a = inputs[0];
                    let b = inputs[1];
                    let shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    let diff = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![a, b],
                        shape.clone(),
                        dtype,
                    );
                    let mask_a = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, shape.clone(), dtype);
                    let mask_b = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_b],
                        shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::Minimum => {
                    // Symmetric to Maximum. mask_a = step(b - a).
                    let a = inputs[0];
                    let b = inputs[1];
                    let shape = node_shape(&graph_handle, a);
                    let dtype = node_dtype(&graph_handle, a);
                    let diff = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![b, a],
                        shape.clone(),
                        dtype,
                    );
                    let mask_a = push_node(
                        &graph_handle,
                        Op::Step,
                        vec![diff],
                        shape.clone(),
                        dtype,
                    );
                    let ones = build_ones(&graph_handle, shape.clone(), dtype);
                    let mask_b = push_node(
                        &graph_handle,
                        Op::Sub,
                        vec![ones, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_a = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_a],
                        shape.clone(),
                        dtype,
                    );
                    let grad_b = push_node(
                        &graph_handle,
                        Op::Mul,
                        vec![up_id, mask_b],
                        shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                    accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                }
                Op::ScatterAdd { dim } => {
                    // Forward: out = scatter_add(base, dim, indices, src).
                    // Backward:
                    //   grad_base = upstream (pass-through)
                    //   grad_src  = gather(upstream, dim, indices)
                    let base = inputs[0];
                    let idx = inputs[1];
                    let src = inputs[2];
                    let src_shape = node_shape(&graph_handle, src);
                    let dtype = node_dtype(&graph_handle, src);
                    accumulate_grad(&mut upstream, base, up_id, &graph_handle);
                    let grad_src = push_node(
                        &graph_handle,
                        Op::Gather { dim },
                        vec![up_id, idx],
                        src_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, src, grad_src, &graph_handle);
                }
                Op::Copy { .. } => {
                    // Forward: out = copy(x, target). Backward: the
                    // gradient flows back to `x`'s original device, so
                    // the reverse would be another Copy. The source
                    // device is x's residency — we don't have the typed
                    // "where was x?" here without tracking it, so we
                    // fall back to the graph's placement hint on x (if
                    // any) or trust the scheduler to re-insert the
                    // correct copy during Phase 4. For Phase 3 we
                    // assume inference-only use and pass upstream through.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::Release => {
                    // Release has no data output and no gradient to
                    // propagate. Input is destroyed; any downstream
                    // node that needs gradient through the original
                    // tensor must get it from a sibling Copy's backward
                    // path, not this one.
                }
                Op::Move { .. } => {
                    // Move is Copy + destroy-source. For backward,
                    // treat it the same as Copy — gradient flows back
                    // through the move. The Phase 3-style assumption
                    // still applies: we pass upstream through and rely
                    // on the scheduler to re-insert the correct reverse
                    // transfer in a training pass. For inference-only
                    // use (the current case), this is a no-op.
                    let x = inputs[0];
                    accumulate_grad(&mut upstream, x, up_id, &graph_handle);
                }
                Op::WriteSlice { .. } => {
                    // KV-cache writes are forward-only — the slab being
                    // written *into* gets mutated in place, and the
                    // source slab is read non-destructively. There is
                    // no meaningful gradient through a mutating write
                    // in fuel's tape-based autograd model. If a use
                    // case needs differentiable scatter, the gradient
                    // path is a `Gather` (read the slab back) plus an
                    // accumulating `IndexAdd` on the destination —
                    // express that explicitly in forward.
                    panic!(
                        "Tensor::backward: Op::WriteSlice is non-differentiable. \
                         Use Gather + IndexAdd if you need a differentiable scatter."
                    );
                }
                Op::WriteSliceRotating { .. } => {
                    // Same forward-only contract as WriteSlice — the
                    // ring buffer's bytes are mutated in place at a
                    // dynamic offset. Sliding-window KV caches don't
                    // need a gradient path; a differentiable scatter
                    // is expressible as Gather + IndexAdd in forward.
                    panic!(
                        "Tensor::backward: Op::WriteSliceRotating is non-differentiable. \
                         Use Gather + IndexAdd if you need a differentiable scatter."
                    );
                }
                // Phase 7.6 step 5 (2026-05-11): the legacy
                // `Op::Conv2D { stride, padding, groups }` backward
                // arm has been dropped together with the variant;
                // `Op::Fused(CONV2D, _)` below carries the full
                // backward logic (including the inner dW Conv2D
                // emission, which itself goes through Op::Fused).
                // Phase 7.6 step 4 (final): the builders for
                // ConvTranspose2D, FlashAttn, and PagedAttn now emit
                // `Op::Fused(<id>, _)`; these legacy backward arms
                // still fire for direct legacy-variant constructions
                // Phase 7.6 step 5 (final, 2026-05-11): the legacy
                // `Op::ConvTranspose2D` / `Op::FlashAttn` /
                // `Op::PagedAttn` / `Op::FusedLinear` backward arms
                // have all been dropped together with the variants;
                // `Op::Fused(<id>, _)` below panics for each id.
                Op::Fused(fid, params) => {
                    // Per-id backward dispatch. Each migrated fused op
                    // gets a branch here that emits the appropriate
                    // gradient subgraph.
                    if fid == crate::registry::FusedOps::SOFTMAX_LAST_DIM {
                        // grad_x = softmax_last_dim_backward(y, upstream).
                        // Phase 7.6 step 4 (backward-helper batch):
                        // emits Op::Fused(SOFTMAX_LAST_DIM_BACKWARD, _)
                        // — the architecturally-correct registry form,
                        // matching the BackwardKind::Fused edge on
                        // SOFTMAX_LAST_DIM's entry.
                        let x = inputs[0];
                        let y_shape = node_shape(&graph_handle, id);
                        let dtype = node_dtype(&graph_handle, id);
                        let grad_x = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::SOFTMAX_LAST_DIM_BACKWARD,
                                crate::registry::FusedOpParams::SoftmaxLastDimBackward,
                            ),
                            vec![id, up_id],
                            y_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                    } else if fid == crate::registry::FusedOps::LAYER_NORM_LAST_DIM {
                        // grad_x = layer_norm_last_dim_backward(x, upstream, eps).
                        // Phase 7.6 step 4 (backward-helper batch): emits
                        // registry form Op::Fused(LAYER_NORM_LAST_DIM_BACKWARD, _).
                        let eps = match params {
                            crate::registry::FusedOpParams::LayerNormLastDim { eps } => eps,
                            _ => panic!(
                                "Tensor::backward: Op::Fused(LAYER_NORM_LAST_DIM, _) \
                                 expected FusedOpParams::LayerNormLastDim, got {params:?}",
                            ),
                        };
                        let x = inputs[0];
                        let x_shape = node_shape(&graph_handle, x);
                        let dtype = node_dtype(&graph_handle, x);
                        let grad_x = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::LAYER_NORM_LAST_DIM_BACKWARD,
                                crate::registry::FusedOpParams::LayerNormLastDimBackward { eps },
                            ),
                            vec![x, up_id],
                            x_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                    } else if fid == crate::registry::FusedOps::RMS_NORM_LAST_DIM {
                        // grad_x = rms_norm_last_dim_backward(x, upstream, eps).
                        // Phase 7.6 step 4 (backward-helper batch): emits
                        // registry form Op::Fused(RMS_NORM_LAST_DIM_BACKWARD, _).
                        let eps = match params {
                            crate::registry::FusedOpParams::RmsNormLastDim { eps } => eps,
                            _ => panic!(
                                "Tensor::backward: Op::Fused(RMS_NORM_LAST_DIM, _) \
                                 expected FusedOpParams::RmsNormLastDim, got {params:?}",
                            ),
                        };
                        let x = inputs[0];
                        let x_shape = node_shape(&graph_handle, x);
                        let dtype = node_dtype(&graph_handle, x);
                        let grad_x = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::RMS_NORM_LAST_DIM_BACKWARD,
                                crate::registry::FusedOpParams::RmsNormLastDimBackward { eps },
                            ),
                            vec![x, up_id],
                            x_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                    } else if fid == crate::registry::FusedOps::ROPE {
                        // grad_x = rope(upstream, cos, -sin) — same
                        // formula as the legacy Op::Rope arm above.
                        // Emit the registry form for the backward
                        // Rope node so the gradient subgraph routes
                        // through the same Op::Fused dispatch as the
                        // forward.
                        let x = inputs[0];
                        let cos = inputs[1];
                        let sin = inputs[2];
                        let x_shape = node_shape(&graph_handle, x);
                        let sin_shape = node_shape(&graph_handle, sin);
                        let dtype = node_dtype(&graph_handle, x);
                        let neg_sin = push_node(
                            &graph_handle, Op::Neg, vec![sin], sin_shape, dtype);
                        let grad_x = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::ROPE,
                                crate::registry::FusedOpParams::Rope,
                            ),
                            vec![up_id, cos, neg_sin],
                            x_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                    } else if fid == crate::registry::FusedOps::FUSED_LINEAR {
                        // (a @ b) + bias backward — same three-grad
                        // decomposition as the legacy Op::FusedLinear
                        // arm above. We reuse MatMul backward (no
                        // intermediate matmul output is cached, so the
                        // grad computes from upstream + a, b directly).
                        let a = inputs[0];
                        let b = inputs[1];
                        let bias = inputs[2];
                        let a_shape = node_shape(&graph_handle, a);
                        let b_shape = node_shape(&graph_handle, b);
                        let bias_shape = node_shape(&graph_handle, bias);
                        let dtype = node_dtype(&graph_handle, a);
                        let b_t_shape = transposed_shape(&b_shape);
                        let b_t = push_node(&graph_handle, Op::Transpose, vec![b], b_t_shape, dtype);
                        let grad_a = push_node(&graph_handle, Op::MatMul, vec![up_id, b_t], a_shape.clone(), dtype);
                        let a_t_shape = transposed_shape(&a_shape);
                        let a_t = push_node(&graph_handle, Op::Transpose, vec![a], a_t_shape, dtype);
                        let grad_b = push_node(&graph_handle, Op::MatMul, vec![a_t, up_id], b_shape, dtype);
                        let grad_bias = push_node(
                            &graph_handle,
                            Op::ReduceSumTo(bias_shape.clone()),
                            vec![up_id],
                            bias_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, a, grad_a, &graph_handle);
                        accumulate_grad(&mut upstream, b, grad_b, &graph_handle);
                        accumulate_grad(&mut upstream, bias, grad_bias, &graph_handle);
                    } else if fid == crate::registry::FusedOps::CONV2D {
                        // Same backward formulation as the legacy
                        // Op::Conv2D arm above. The inner Conv2D node
                        // emitted for grad_w also goes through the
                        // registry (Op::Fused(CONV2D, _)) so the
                        // gradient subgraph routes consistently. The
                        // ConvTranspose2D node stays as a primitive
                        // variant until that op migrates in its own
                        // step-4 commit.
                        let (stride, padding, groups) = match params {
                            crate::registry::FusedOpParams::Conv2D { stride, padding, groups } => {
                                (stride, padding, groups)
                            }
                            _ => panic!(
                                "Tensor::backward: Op::Fused(CONV2D, _) \
                                 expected FusedOpParams::Conv2D, got {params:?}",
                            ),
                        };
                        let x      = inputs[0];
                        let weight = inputs[1];
                        let x_shape = node_shape(&graph_handle, x);
                        let w_shape = node_shape(&graph_handle, weight);
                        let dtype = node_dtype(&graph_handle, x);

                        // dX via transposed conv (see legacy arm for derivation).
                        let x_dims = x_shape.dims();
                        let w_dims = w_shape.dims();
                        let dy_dims = node_shape(&graph_handle, id).dims().to_vec();
                        let (sh, sw) = stride;
                        let (ph, pw) = padding;
                        let (kh, kw) = (w_dims[2], w_dims[3]);
                        let (h_in, w_in) = (x_dims[2], x_dims[3]);
                        let (dy_h, dy_w) = (dy_dims[2], dy_dims[3]);
                        let base_h = (dy_h - 1) * sh + (kh - 1) + 1;
                        let base_w = (dy_w - 1) * sw + (kw - 1) + 1;
                        let want_h = h_in + 2 * ph;
                        let want_w = w_in + 2 * pw;
                        let out_pad_h = want_h.saturating_sub(base_h);
                        let out_pad_w = want_w.saturating_sub(base_w);
                        if groups != 1 {
                            panic!(
                                "Tensor::backward: Op::Fused(CONV2D) groups>1 \
                                 backward not yet implemented (got groups={groups}). \
                                 The stride/padding/groups=1 case is wired; grouped \
                                 backward needs a per-group weight reshape.",
                            );
                        }
                        let perm_01 = vec![1usize, 0, 2, 3];
                        // Phase 7.6 step 4 (final): emit registry form
                        // for the inner dX node so the gradient subgraph
                        // routes consistently through Op::Fused dispatch.
                        let grad_x = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::CONV_TRANSPOSE2D,
                                crate::registry::FusedOpParams::ConvTranspose2D {
                                    stride,
                                    padding,
                                    output_padding: (out_pad_h, out_pad_w),
                                    dilation: (1, 1),
                                    groups,
                                },
                            ),
                            vec![up_id, weight],
                            x_shape.clone(),
                            dtype,
                        );
                        accumulate_grad(&mut upstream, x, grad_x, &graph_handle);

                        // dW via correlation expressed as a conv2d.
                        if sh != 1 || sw != 1 {
                            panic!(
                                "Tensor::backward: Op::Fused(CONV2D) stride>1 \
                                 backward not yet implemented (got stride={stride:?}). \
                                 Conv2D needs a `dilation` field to express this \
                                 without composing extra ops.",
                            );
                        }
                        let x_swapped_shape = {
                            let d = x_dims;
                            Shape::from_dims(&[d[1], d[0], d[2], d[3]])
                        };
                        let x_swapped = push_node(
                            &graph_handle,
                            Op::Permute(perm_01.clone()),
                            vec![x],
                            x_swapped_shape,
                            dtype,
                        );
                        let dy_swapped_shape = Shape::from_dims(&[
                            dy_dims[1], dy_dims[0], dy_dims[2], dy_dims[3],
                        ]);
                        let dy_swapped = push_node(
                            &graph_handle,
                            Op::Permute(perm_01.clone()),
                            vec![up_id],
                            dy_swapped_shape,
                            dtype,
                        );
                        let conv_out_shape = Shape::from_dims(&[
                            w_dims[1], w_dims[0], w_dims[2], w_dims[3],
                        ]);
                        let grad_w_pre_transpose = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::CONV2D,
                                crate::registry::FusedOpParams::Conv2D {
                                    stride: (1, 1),
                                    padding,
                                    groups: 1,
                                },
                            ),
                            vec![x_swapped, dy_swapped],
                            conv_out_shape,
                            dtype,
                        );
                        let grad_w = push_node(
                            &graph_handle,
                            Op::Permute(perm_01),
                            vec![grad_w_pre_transpose],
                            w_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, weight, grad_w, &graph_handle);

                        // Bias (if present): sum dY over N, H, W via
                        // ReduceSumTo([1, Cout, 1, 1]) then reshape to
                        // [Cout]. Same path as the legacy arm.
                        if inputs.len() >= 3 {
                            let bias = inputs[2];
                            let bias_shape = node_shape(&graph_handle, bias);
                            let cout_4d = Shape::from_dims(&[1, dy_dims[1], 1, 1]);
                            let reduced = push_node(
                                &graph_handle,
                                Op::ReduceSumTo(cout_4d.clone()),
                                vec![up_id],
                                cout_4d,
                                dtype,
                            );
                            let grad_b = push_node(
                                &graph_handle,
                                Op::Reshape(bias_shape.clone()),
                                vec![reduced],
                                bias_shape,
                                dtype,
                            );
                            accumulate_grad(&mut upstream, bias, grad_b, &graph_handle);
                        }
                    } else if fid == crate::registry::FusedOps::SOFTMAX_LAST_DIM_BACKWARD
                        || fid == crate::registry::FusedOps::LAYER_NORM_LAST_DIM_BACKWARD
                        || fid == crate::registry::FusedOps::RMS_NORM_LAST_DIM_BACKWARD
                        || fid == crate::registry::FusedOps::REDUCE_MAX_TO_BACKWARD
                    {
                        // Higher-order gradients through backward
                        // helpers panic.
                        panic!(
                            "backward: higher-order gradients through \
                             softmax/layer_norm/rms_norm/reduce_max_to backward \
                             helpers are not yet supported in the MVP."
                        );
                    } else if fid == crate::registry::FusedOps::CONV_TRANSPOSE2D {
                        // Higher-order grad of the transposed conv
                        // isn't needed for Conv2D's backward (which
                        // only consumes the forward output). Adding
                        // it requires the same dilation-as-stride
                        // trick as Conv2D's dW. Punt until a real
                        // consumer asks for it.
                        panic!(
                            "Tensor::backward: ConvTranspose2D does \
                             not yet have its own gradient rule \
                             (only used in the forward path of \
                             Conv2D's backward).",
                        );
                    } else if fid == crate::registry::FusedOps::FLASH_ATTN {
                        // Emit three backward nodes: dQ via
                        // FLASH_ATTN_BACKWARD_Q, dK via _K, dV via _V.
                        // Each takes (q, k, v, dO, [alibi]) and
                        // recomputes the softmax state independently
                        // in CPU's math-definition path. GPU kernels
                        // (when they ship) can fuse the recompute.
                        let q = inputs[0];
                        let k = inputs[1];
                        let v = inputs[2];
                        let alibi = inputs.get(3).copied();
                        let q_shape = node_shape(&graph_handle, q);
                        let k_shape = node_shape(&graph_handle, k);
                        let v_shape = node_shape(&graph_handle, v);
                        let dtype = node_dtype(&graph_handle, q);
                        let (scale, causal, wl, wr, sc) = match params {
                            crate::registry::FusedOpParams::FlashAttn {
                                softmax_scale, causal,
                                window_size_left, window_size_right, softcap,
                                // Backward over the static (full-K) flash
                                // only; the runtime-k_len capacity form is
                                // forward-only (decode). k_len is ignored
                                // here.
                                k_len: _,
                            } => (
                                softmax_scale, causal,
                                window_size_left, window_size_right, softcap,
                            ),
                            other => panic!(
                                "Tensor::backward: FlashAttn node carries \
                                 unexpected params {other:?}",
                            ),
                        };
                        let bw_params = crate::registry::FusedOpParams::FlashAttnBackward {
                            softmax_scale: scale,
                            causal,
                            window_size_left: wl,
                            window_size_right: wr,
                            softcap: sc,
                        };
                        let mut bw_inputs = vec![q, k, v, up_id];
                        if let Some(a) = alibi { bw_inputs.push(a); }
                        let grad_q = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::FLASH_ATTN_BACKWARD_Q,
                                bw_params.clone(),
                            ),
                            bw_inputs.clone(),
                            q_shape,
                            dtype,
                        );
                        let grad_k = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::FLASH_ATTN_BACKWARD_K,
                                bw_params.clone(),
                            ),
                            bw_inputs.clone(),
                            k_shape,
                            dtype,
                        );
                        let grad_v = push_node(
                            &graph_handle,
                            Op::Fused(
                                crate::registry::FusedOps::FLASH_ATTN_BACKWARD_V,
                                bw_params,
                            ),
                            bw_inputs,
                            v_shape,
                            dtype,
                        );
                        accumulate_grad(&mut upstream, q, grad_q, &graph_handle);
                        accumulate_grad(&mut upstream, k, grad_k, &graph_handle);
                        accumulate_grad(&mut upstream, v, grad_v, &graph_handle);
                    } else if fid == crate::registry::FusedOps::PAGED_ATTN {
                        // Paged attention is decode-side only by
                        // construction (variable-length KV cache, no
                        // training pass writes through it). No
                        // gradient rule.
                        panic!(
                            "Tensor::backward: PagedAttn is \
                             decode-only; no gradient rule exists.",
                        );
                    } else if fid == crate::registry::FusedOps::QMATMUL {
                        // Quantized matmul: weights are frozen
                        // bytes; activation gradient isn't
                        // implemented today. Dequantize-then-matmul
                        // is the workaround if differentiable
                        // quantized inference is needed.
                        panic!(
                            "backward: QMatMul is not differentiable \
                             (quantized weights are frozen). Use a \
                             dequantize + standard matmul if you \
                             need gradients through this input."
                        );
                    } else {
                        panic!(
                            "Tensor::backward: Op::Fused id {fid:?} has no \
                             backward arm wired yet. This is a programming \
                             bug — extend the match when the op migrates \
                             to the registry.",
                        );
                    }
                }
                Op::Alloc { .. } => {
                    // Op::Alloc is a source op (zero inputs) that
                    // produces a fresh zero-init buffer. No upstream
                    // gradient to propagate — the alloc's "input" is
                    // the abstract notion of fresh memory, which has
                    // no derivative. Mirrors Op::Const's treatment.
                    // (Phase 3a of bridge-retirement; see
                    // `WorkItemKind::Alloc` in fuel-storage.)
                }
                Op::ZeroFill => {
                    // Op::ZeroFill writes deterministic constants
                    // (zeros) to its input's bytes. The output is
                    // a constant tensor; gradient w.r.t. it is
                    // meaningless. Like Op::Const / Op::Alloc, no
                    // gradient propagates back.
                }
                Op::ScatterIntoSlot { .. } => {
                    // Item 4 IR-only scaffold — the backward graph
                    // emits this op as the producer's gradient
                    // assembler, but `ScatterIntoSlot` itself has no
                    // backward (it's a structural op composing
                    // already-differentiated slot gradients). No
                    // gradient propagates back; matches the
                    // Op::Const / Op::Alloc / Op::ZeroFill treatment.
                }
                Op::Branch { .. } => {
                    // PR-A0 inert scaffold. `Op::Branch` is the
                    // multi-path phi/merge node; it is never constructed
                    // in A0, so this arm is unreachable today. When the
                    // PR-A1 builders land, gradient flow through a Branch
                    // routes via its arms (each an `inputs[i]` route exit)
                    // and reconverges at `reconverge_at` — not through the
                    // merge node itself. `backward` is infallible
                    // (`-> GradMap`), so the no-panic inert handling is to
                    // drop the gradient at the structural node, matching
                    // the Op::Const / Op::ScatterIntoSlot treatment above.
                    // The real per-arm routing lands with the builders.
                }
                Op::View { slot } | Op::ViewOwned { slot } => {
                    // Multi-output projection — item 4 backward
                    // composition (Option C, 2026-06-01).
                    //
                    // Forward (Op::View): shares the producer's
                    // bundled Storage Arc and exposes one slot's
                    // typed window. Forward (Op::ViewOwned): memcpys
                    // the slot's bytes into a fresh Storage.
                    //
                    // Backward: scatter the upstream gradient into
                    // the slot's byte range of a fresh zero-bundle of
                    // the producer's primary shape, then accumulate
                    // that partial-bundle gradient as the producer's
                    // input gradient. Multiple View consumers of the
                    // same producer accumulate via the standard
                    // sum-of-partials path:
                    //   bundle_grad =
                    //     ScatterIntoSlot{0}(0, g_y)  +
                    //     ScatterIntoSlot{1}(0, g_state) + ...
                    //
                    // **Status (item 4)**: IR-level wiring only —
                    // emits the scatter chain in the backward graph,
                    // but no Op::ScatterIntoSlot kernel is registered.
                    // The production multi-output producers
                    // (SelectiveScan, SsdChunkScan) are
                    // `BackwardKind::NotDifferentiable`, so autograd
                    // panics at the producer before this scatter is
                    // realized. When a differentiable multi-output op
                    // materializes (Mamba training, FSCE loss+grad
                    // fused), it lights up the ScatterIntoSlot CPU
                    // kernel + tests alongside its own backward.
                    let producer = inputs[0];
                    let producer_shape = node_shape(&graph_handle, producer);
                    let producer_dtype = node_dtype(&graph_handle, producer);
                    let zero = build_filled_const(
                        &graph_handle,
                        producer_shape.clone(),
                        producer_dtype,
                        0.0,
                    );
                    let scattered = push_node(
                        &graph_handle,
                        Op::ScatterIntoSlot { slot },
                        vec![zero, up_id],
                        producer_shape,
                        producer_dtype,
                    );
                    accumulate_grad(&mut upstream, producer, scattered, &graph_handle);
                }
                Op::ReluInplace => {
                    // Same backward as Op::Relu — d(relu(x))/dx = step(x).
                    // The view-aware `derive_ordering` (Phase 4a) pins
                    // `Op::Step(x)` to run BEFORE Op::ReluInplace, so the
                    // step kernel sees x's original bytes.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let step_x = push_node(&graph_handle, Op::Step, vec![x], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, step_x], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SiluInplace => {
                    // Same backward as Op::Silu — grad_x = upstream * (s + y*(1-s))
                    // where s = sigmoid(x) and y = silu(x) = this in-place node's
                    // output. The post-mutation bytes at `id` ARE silu(x) (the
                    // in-place op runs after the Sigmoid(x) read because of the
                    // alias-aware ordering pass). `Op::Sigmoid(x)` reads x's
                    // pre-mutation bytes; `Op::Mul(id, ...)` reads post-mutation
                    // bytes — both correct.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let s = push_node(&graph_handle, Op::Sigmoid, vec![x], x_shape.clone(), dtype);
                    let ones = build_ones(&graph_handle, x_shape.clone(), dtype);
                    let one_minus_s = push_node(&graph_handle, Op::Sub, vec![ones, s], x_shape.clone(), dtype);
                    let y_times_1ms = push_node(&graph_handle, Op::Mul, vec![id, one_minus_s], x_shape.clone(), dtype);
                    let inner = push_node(&graph_handle, Op::Add, vec![s, y_times_1ms], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, inner], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::TanhInplace => {
                    // Same backward as Op::Tanh — grad_x = upstream * (1 - y²)
                    // where y = tanh(x) = this in-place node's output. `Op::Sqr(id)`
                    // reads post-mutation bytes (= tanh(x) by definition of the
                    // in-place op); no need to reference x at all.
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let y_sq = push_node(&graph_handle, Op::Sqr, vec![id], y_shape.clone(), dtype);
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_sq = push_node(&graph_handle, Op::Sub, vec![ones, y_sq], y_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, one_minus_sq], y_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SigmoidInplace => {
                    // Same backward as Op::Sigmoid — grad_x = upstream * y * (1 - y)
                    // where y = sigmoid(x) = this in-place node's output.
                    let x = inputs[0];
                    let y_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let ones = build_ones(&graph_handle, y_shape.clone(), dtype);
                    let one_minus_y = push_node(&graph_handle, Op::Sub, vec![ones, id], y_shape.clone(), dtype);
                    let y_times_1my = push_node(&graph_handle, Op::Mul, vec![id, one_minus_y], y_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, y_times_1my], y_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::GeluInplace => {
                    // Mirrors Op::Gelu's panic — the tanh-approximation
                    // gradient is non-trivial and not yet implemented
                    // for the non-inplace variant either. Users who
                    // need a differentiable GELU should use SiluInplace
                    // (which has a proper backward).
                    panic!(
                        "backward: GeluInplace gradient is not yet supported. \
                         Use SiluInplace for differentiable training; GeluInplace \
                         is currently inference-only (mirrors Op::Gelu's status)."
                    );
                }
                Op::NegInplace => {
                    // Same backward as Op::Neg — grad_x = -upstream. No
                    // dependency on x or y; Phase 4a ordering not needed.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(&graph_handle, Op::Neg, vec![up_id], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::AbsInplace => {
                    // Same backward as Op::Abs — grad_x = sign(x) * upstream.
                    // `Op::Sign(x)` reads x's pre-mutation bytes (Phase 4a
                    // pins the in-place op after the Sign read).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let sign_x = push_node(&graph_handle, Op::Sign, vec![x], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, sign_x], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SqrInplace => {
                    // Same backward as Op::Sqr — grad_x = 2x * upstream,
                    // expressed as `x + x` to avoid a scalar broadcast.
                    // `x + x` reads pre-mutation bytes (Phase 4a).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_x = push_node(&graph_handle, Op::Add, vec![x, x], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, two_x], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SqrtInplace => {
                    // Same backward as Op::Sqrt — y = √x, grad_x = upstream/(2y).
                    // Reads only the forward output `id`, which is post-mutation =
                    // √x by definition; no pre-mutation x access needed.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let two_y = push_node(&graph_handle, Op::Add, vec![id, id], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Div, vec![up_id, two_y], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::RsqrtInplace => {
                    // Same backward as Op::Rsqrt — y = x^(-1/2),
                    // grad_x = -0.5 * upstream * y³. Reads forward output (id)
                    // only — post-mutation = y by definition.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let y_sq = push_node(&graph_handle, Op::Sqr, vec![id], x_shape.clone(), dtype);
                    let y_cu = push_node(&graph_handle, Op::Mul, vec![id, y_sq], x_shape.clone(), dtype);
                    let scaled = push_node(&graph_handle, Op::MulScalar(-0.5), vec![y_cu], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, scaled], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::RecipInplace => {
                    // Same backward as Op::Recip — y = 1/x, grad_x = -upstream * y².
                    // Reads forward output (id) only — post-mutation = 1/x.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let y_sq = push_node(&graph_handle, Op::Sqr, vec![id], x_shape.clone(), dtype);
                    let up_y_sq = push_node(&graph_handle, Op::Mul, vec![up_id, y_sq], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Neg, vec![up_y_sq], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::ExpInplace => {
                    // Same backward as Op::Exp — d(exp(x))/dx = exp(x) = y = id.
                    // grad_x = upstream * y. Reads forward output (id) only.
                    let x = inputs[0];
                    let out_shape = node_shape(&graph_handle, id);
                    let dtype = node_dtype(&graph_handle, id);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, id], out_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::LogInplace => {
                    // Same backward as Op::Log — grad_x = upstream / x.
                    // `Op::Div(up, x)` reads pre-mutation bytes of x (Phase 4a).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(&graph_handle, Op::Div, vec![up_id, x], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SinInplace => {
                    // Same backward as Op::Sin — grad_x = upstream * cos(x).
                    // `Op::Cos(x)` reads pre-mutation x via Phase 4a ordering.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let cos_x = push_node(&graph_handle, Op::Cos, vec![x], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, cos_x], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::CosInplace => {
                    // Same backward as Op::Cos — grad_x = -(upstream * sin(x)).
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let sin_x = push_node(&graph_handle, Op::Sin, vec![x], x_shape.clone(), dtype);
                    let up_sin = push_node(&graph_handle, Op::Mul, vec![up_id, sin_x], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Neg, vec![up_sin], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::SignInplace
                | Op::FloorInplace
                | Op::CeilInplace
                | Op::RoundInplace => {
                    // Same backward as Op::{Sign, Floor, Ceil, Round} —
                    // derivative is 0 almost everywhere; drop gradient
                    // (no accumulation, mirroring Op::Step's pattern).
                }
                Op::ErfInplace => {
                    // Same backward as Op::Erf — grad_x = upstream * (2/√π) * exp(-x²).
                    // `Op::Sqr(x)` reads pre-mutation x via Phase 4a ordering.
                    const TWO_OVER_SQRT_PI: f64 = 1.128_379_167_095_512_6_f64;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_sq = push_node(&graph_handle, Op::Sqr, vec![x], x_shape.clone(), dtype);
                    let neg_x_sq = push_node(&graph_handle, Op::Neg, vec![x_sq], x_shape.clone(), dtype);
                    let exp_neg = push_node(&graph_handle, Op::Exp, vec![neg_x_sq], x_shape.clone(), dtype);
                    let scaled = push_node(&graph_handle, Op::MulScalar(TWO_OVER_SQRT_PI), vec![exp_neg], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, scaled], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::ClampInplace { min, max } => {
                    // Same backward as Op::Clamp — indicator is 1 where
                    // min ≤ x ≤ max, 0 elsewhere. Reads pre-mutation x via
                    // Phase 4a ordering.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let min_shifted = push_node(&graph_handle, Op::AddScalar(-min), vec![x], x_shape.clone(), dtype);
                    let lower_ok = push_node(&graph_handle, Op::Step, vec![min_shifted], x_shape.clone(), dtype);
                    let neg_x = push_node(&graph_handle, Op::MulScalar(-1.0), vec![x], x_shape.clone(), dtype);
                    let max_minus_x = push_node(&graph_handle, Op::AddScalar(max), vec![neg_x], x_shape.clone(), dtype);
                    let upper_ok = push_node(&graph_handle, Op::Step, vec![max_minus_x], x_shape.clone(), dtype);
                    let indicator = push_node(&graph_handle, Op::Mul, vec![lower_ok, upper_ok], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, indicator], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::PowIInplace(n) => {
                    // Same backward as Op::PowI — single fused
                    // POWI_BACKWARD node `(x, upstream) → exp · x^(exp-1) ·
                    // upstream`. Reads pre-mutation x via Phase 4a ordering.
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let grad_x = push_node(
                        &graph_handle,
                        Op::Fused(
                            crate::registry::FusedOps::POWI_BACKWARD,
                            crate::registry::FusedOpParams::PowIBackward { exp: n },
                        ),
                        vec![x, up_id],
                        x_shape,
                        dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
                Op::GeluErfInplace => {
                    // Same backward as Op::GeluErf — exact-GELU derivative
                    // is Φ(x) + x·φ(x). Both halves read pre-mutation x via
                    // Phase 4a ordering. Identical 12-node chain to Op::GeluErf.
                    const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
                    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7_f64;
                    let x = inputs[0];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let x_over_sqrt2 = push_node(&graph_handle, Op::MulScalar(INV_SQRT_2), vec![x], x_shape.clone(), dtype);
                    let erf_arg = push_node(&graph_handle, Op::Erf, vec![x_over_sqrt2], x_shape.clone(), dtype);
                    let plus_one = push_node(&graph_handle, Op::AddScalar(1.0), vec![erf_arg], x_shape.clone(), dtype);
                    let cdf_term = push_node(&graph_handle, Op::MulScalar(0.5), vec![plus_one], x_shape.clone(), dtype);
                    let x_sq = push_node(&graph_handle, Op::Sqr, vec![x], x_shape.clone(), dtype);
                    let half_x_sq = push_node(&graph_handle, Op::MulScalar(0.5), vec![x_sq], x_shape.clone(), dtype);
                    let neg_half = push_node(&graph_handle, Op::Neg, vec![half_x_sq], x_shape.clone(), dtype);
                    let exp_term = push_node(&graph_handle, Op::Exp, vec![neg_half], x_shape.clone(), dtype);
                    let phi = push_node(&graph_handle, Op::MulScalar(INV_SQRT_2PI), vec![exp_term], x_shape.clone(), dtype);
                    let pdf_term = push_node(&graph_handle, Op::Mul, vec![x, phi], x_shape.clone(), dtype);
                    let d_gelu = push_node(&graph_handle, Op::Add, vec![cdf_term, pdf_term], x_shape.clone(), dtype);
                    let grad_x = push_node(&graph_handle, Op::Mul, vec![up_id, d_gelu], x_shape, dtype);
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
            }
        }

        GradMap {
            graph:           graph_handle,
            forward_to_grad: upstream,
        }
    }
}

/// Map from forward-graph node IDs to the IDs of their accumulated
/// backward-graph gradient nodes, returned by [`Tensor::backward`].
///
/// All gradient nodes live on the same graph as the forward nodes. Getting
/// a gradient by calling [`GradMap::get`] returns a new [`Tensor`] handle
/// pointing at the gradient node so callers can realize it, plug it into
/// further computation, or back-propagate through it again.
pub struct GradMap {
    graph:           SharedGraph,
    forward_to_grad: HashMap<NodeId, NodeId>,
}

impl GradMap {
    /// Get the gradient tensor for a forward tensor from the same graph.
    ///
    /// Returns `None` if the forward tensor was not reachable from the
    /// root passed to `backward` (i.e. not part of the computation being
    /// differentiated).
    pub fn get(&self, forward: &Tensor) -> Option<Tensor> {
        assert!(
            Arc::ptr_eq(&self.graph, &forward.graph),
            "GradMap::get: tensor is from a different graph",
        );
        let &grad_id = self.forward_to_grad.get(&forward.id)?;
        Some(Tensor {
            graph: self.graph.clone(),
            id:    grad_id,
        })
    }

    /// Number of forward nodes that have a recorded gradient.
    pub fn len(&self) -> usize {
        self.forward_to_grad.len()
    }

    /// Whether this map has no gradients (e.g. a backward over a lone const).
    pub fn is_empty(&self) -> bool {
        self.forward_to_grad.is_empty()
    }
}

// -------- internal helpers shared by the backward pass --------------------

/// Read a node's shape without holding the borrow past the call site.
fn node_shape(graph: &SharedGraph, id: NodeId) -> Shape {
    graph.read().unwrap().node(id).shape.clone()
}

/// Read a node's dtype without holding the borrow past the call site.
fn node_dtype(graph: &SharedGraph, id: NodeId) -> DType {
    graph.read().unwrap().node(id).dtype
}

/// Shape with its last two dims swapped. Used by MatMul's backward rule
/// to construct transpose-node output shapes for any rank ≥ 2.
fn transposed_shape(shape: &Shape) -> Shape {
    let d = shape.dims();
    assert!(
        d.len() >= 2,
        "transposed_shape: expected rank ≥ 2, got {d:?}",
    );
    let rank = d.len();
    let mut out: Vec<usize> = d.to_vec();
    out.swap(rank - 2, rank - 1);
    Shape::from_dims(&out)
}

/// Given a full tensor shape and a dim that was reduced away, produce the
/// keepdim shape — i.e. the full shape with size 1 at `dim`. Used by
/// axis-reduction backward rules to reshape the reduced upstream back to
/// a shape that broadcasts against the original input.
fn reshape_to_keepdim(full_dims: &[usize], dim: usize) -> Shape {
    assert!(
        dim < full_dims.len(),
        "reshape_to_keepdim: dim {dim} out of bounds for {full_dims:?}",
    );
    let mut out: Vec<usize> = full_dims.to_vec();
    out[dim] = 1;
    Shape::from_dims(&out)
}

/// Validate that `src` can be broadcast to `dst` using NumPy rules:
/// right-align, pad the shorter shape with 1s on the left, and for each
/// aligned dimension the src size must equal the dst size or be 1.
fn check_broadcast_compatible(src: &[usize], dst: &[usize]) {
    assert!(
        src.len() <= dst.len(),
        "broadcast_to: source rank {} exceeds target rank {}",
        src.len(),
        dst.len(),
    );
    let pad = dst.len() - src.len();
    for (i, &s) in src.iter().enumerate() {
        let d = dst[pad + i];
        assert!(
            s == d || s == 1,
            "broadcast_to: dim {i} of source ({s}) is incompatible with dim {} of target ({d})",
            pad + i,
        );
    }
}

/// Result-returning sibling of [`check_broadcast_compatible`].
fn try_check_broadcast_compatible(
    src: &[usize], dst: &[usize],
) -> std::result::Result<(), fuel_ir::Error> {
    if src.len() > dst.len() {
        return Err(fuel_ir::Error::Msg(format!(
            "broadcast_to: source rank {} exceeds target rank {}",
            src.len(), dst.len(),
        )).bt());
    }
    let pad = dst.len() - src.len();
    for (i, &s) in src.iter().enumerate() {
        let d = dst[pad + i];
        if s != d && s != 1 {
            return Err(fuel_ir::Error::Msg(format!(
                "broadcast_to: dim {i} of source ({s}) is incompatible with dim {} of target ({d})",
                pad + i,
            )).bt());
        }
    }
    Ok(())
}

/// Compute the NumPy broadcast shape of two input shapes. Right-align,
/// pad the shorter with leading 1s, and take the element-wise max at
/// each position. Panics if the shapes are incompatible (neither can
/// expand to match the other at some dim).
fn compute_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    let a_pad = n - a.len();
    let b_pad = n - b.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ai = if i < a_pad { 1 } else { a[i - a_pad] };
        let bi = if i < b_pad { 1 } else { b[i - b_pad] };
        let oi = if ai == bi {
            ai
        } else if ai == 1 {
            bi
        } else if bi == 1 {
            ai
        } else {
            panic!(
                "compute_broadcast_shape: incompatible shapes {a:?} vs {b:?} at axis {i}: {ai} vs {bi}",
            );
        };
        out.push(oi);
    }
    out
}

/// Push a node to the graph and return its ID. Used by backward to append
/// gradient-computing nodes without going through the public builders,
/// which would re-validate shapes the backward pass has already derived.
fn push_node(
    graph: &SharedGraph,
    op: Op,
    inputs: Vec<NodeId>,
    shape: Shape,
    dtype: DType,
) -> NodeId {
    graph.write().unwrap().push(Node {
        op,
        inputs,
        shape,
        dtype,
    })
}

/// Build a ones tensor of the given shape and dtype as a `Const` node.
/// Used to seed the initial upstream gradient at the root of the backward
/// pass (`dL/dL = 1`).
fn build_ones(graph: &SharedGraph, shape: Shape, dtype: DType) -> NodeId {
    build_filled_const(graph, shape, dtype, 1.0)
}

/// Build a constant tensor of the given shape and dtype filled with
/// `value` (expressed as `f64` and converted to the target dtype). Used
/// by backward rules that need a scalar-valued broadcast, for example
/// `MeanAll`'s `1/n` factor expressed as "divide by an n-filled const."
///
/// Build RoPE cos/sin tables for a given `(base, start_pos, seq,
/// head_dim)`. Each returned `Vec<f32>` has shape `[seq, head_dim]`
/// (row-major) in the half-split layout [`Tensor::rope_with_tables`]
/// expects: positions `[:, :half]` and `[:, half:]` hold the same
/// value for the same `(p, i)`, so a single elementwise multiply with
/// `self` does the right thing alongside `rotate_half(self) * sin`.
///
/// Exposed as a free function so callers can share the tables across
/// many RoPE applications (e.g. every attention layer in a transformer
/// forward pass).
pub fn build_rope_tables(
    base: f64,
    start_pos: usize,
    seq: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        head_dim.is_multiple_of(2),
        "build_rope_tables: head_dim {head_dim} must be even",
    );
    let half = head_dim / 2;
    let mut cos_data = vec![0.0_f32; seq * head_dim];
    let mut sin_data = vec![0.0_f32; seq * head_dim];
    for p in 0..seq {
        let pos = (start_pos + p) as f64;
        for i in 0..half {
            let freq = base.powf(-2.0 * (i as f64) / (head_dim as f64));
            let theta = pos * freq;
            let c = theta.cos() as f32;
            let s = theta.sin() as f32;
            cos_data[p * head_dim + i] = c;
            cos_data[p * head_dim + i + half] = c;
            sin_data[p * head_dim + i] = s;
            sin_data[p * head_dim + i + half] = s;
        }
    }
    (cos_data, sin_data)
}

/// Only float dtypes are supported here — integer/index tensors never
/// appear as gradients, so a `U32` call indicates a bug in a backward
/// rule rather than a missing feature.
///
/// Phase 7.5 G2: gradient consts are slot-populating Op::Const
/// nodes. Device is picked from any existing slot (the forward pass
/// always populated at least one slot-bearing leaf by the time
/// backward runs). For a CPU-rooted graph the device's
/// `storage_from_host_buffer_owned_dyn` is a zero-copy wrap; for
/// GPU-rooted, an H2D upload (matching today's eval_const path on
/// first realize).
fn build_filled_const(graph: &SharedGraph, shape: Shape, dtype: DType, value: f64) -> NodeId {
    let n = shape.elem_count();
    let buf = match dtype {
        DType::F32 => fuel_ir::HostBuffer::F32(vec![value as f32; n]),
        DType::F64 => fuel_ir::HostBuffer::F64(vec![value; n]),
        DType::BF16 => fuel_ir::HostBuffer::BF16(vec![bf16::from_f64(value); n]),
        DType::F16 => fuel_ir::HostBuffer::F16(vec![f16::from_f64(value); n]),
        other => panic!(
            "backward: build_filled_const: unsupported dtype {other:?} \
             (gradients are always floats — this would indicate a bug in \
             a backward rule that tried to differentiate through an \
             integer tensor)",
        ),
    };
    let device = pick_device_from_graph(graph);
    let backend_storage = device
        .storage_from_host_buffer_owned_dyn(buf)
        .expect("build_filled_const: storage_from_host_buffer_owned_dyn failed");
    let storage_arc = Arc::new(RwLock::new(Storage::from_dyn(backend_storage)));
    let id = push_node(graph, Op::Const, vec![], shape, dtype);
    graph.write().unwrap().set_storage(id, storage_arc);
    id
}

/// G2 helper: walk the graph for any populated slot and return its
/// device. Used by gradient builders that don't have an explicit
/// device parameter — the graph always has at least one slot-bearing
/// Const leaf by the time backward() runs (the forward pass's inputs
/// are slot-rooted Const leaves), so this can be relied on.
fn pick_device_from_graph(graph: &SharedGraph) -> Arc<dyn fuel_backend_contract::DynBackendDevice> {
    let g = graph.read().unwrap();
    for i in 0..g.len() {
        if let Some(slot_arc) = g.storage_for(NodeId(i)) {
            return slot_arc.read().unwrap().device();
        }
    }
    panic!(
        "build_filled_const: graph has no populated storage slots; backward \
         requires at least one slot-bearing Const leaf to pick a gradient \
         device from. Was the forward pass built with G2-migrated \
         constructors (from_f32/etc. with explicit &Device)?"
    )
}

/// Accumulate a new gradient contribution into the upstream map.
///
/// If `forward_id` already has an upstream gradient, emit an `Add` node to
/// sum the two contributions. If not, record the new one as-is. This is
/// the multi-variable chain rule in its operational form: when a forward
/// node has multiple downstream consumers, the gradients from each must
/// be summed.
fn accumulate_grad(
    upstream: &mut HashMap<NodeId, NodeId>,
    forward_id: NodeId,
    new_grad_id: NodeId,
    graph: &SharedGraph,
) {
    match upstream.get(&forward_id).copied() {
        None => {
            upstream.insert(forward_id, new_grad_id);
        }
        Some(existing_id) => {
            let shape = node_shape(graph, forward_id);
            let dtype = node_dtype(graph, forward_id);
            let combined = push_node(
                graph,
                Op::Add,
                vec![existing_id, new_grad_id],
                shape,
                dtype,
            );
            upstream.insert(forward_id, combined);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 7.5 G2: tests need a real device to allocate slot
    /// storage through the new constructor API. `cpu_dev()` returns a
    /// stable singleton CpuBackendDevice handle so every call site can
    /// just pass `cpu_dev()` without per-test boilerplate.
    fn cpu_dev() -> &'static Arc<dyn fuel_backend_contract::DynBackendDevice> {
        static D: std::sync::OnceLock<Arc<dyn fuel_backend_contract::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    #[test]
    fn destructive_input_release_is_some_zero() {
        assert_eq!(Op::Release.destructive_input(), Some(0));
    }

    /// Phase 7.5 storage-unification B2: target_backend side-table
    /// is sparse — empty by default, set explicitly per node.
    #[test]
    fn target_backend_side_table_starts_empty() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1]),
            dtype: DType::F32,
        });
        assert_eq!(g.target_backend(id), None);
        assert_eq!(g.target_backend_count(), 0);
    }

    /// set_target_backend then read returns the same value.
    #[test]
    fn target_backend_set_and_read() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1]),
            dtype: DType::F32,
        });
        g.set_target_backend(id, BackendId::Cpu);
        assert_eq!(g.target_backend(id), Some(BackendId::Cpu));
        assert_eq!(g.target_backend_count(), 1);
    }

    /// set_target_backend overwrites prior value.
    #[test]
    fn target_backend_overwrite() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1]),
            dtype: DType::F32,
        });
        g.set_target_backend(id, BackendId::Cpu);
        g.set_target_backend(id, BackendId::Cuda);
        assert_eq!(g.target_backend(id), Some(BackendId::Cuda));
        assert_eq!(g.target_backend_count(), 1);
    }

    /// Layouts side-table: default fallback returns
    /// `Layout::contiguous(node.shape)` for any node without an
    /// explicit entry.
    #[test]
    fn layout_default_is_contiguous_over_node_shape() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 3]),
            dtype: DType::F32,
        });
        assert!(!g.has_explicit_layout(id));
        assert_eq!(g.explicit_layout_count(), 0);
        let l = g.layout(id);
        assert_eq!(l.shape().dims(), &[2, 3]);
        assert!(l.is_contiguous());
        assert_eq!(l.start_offset(), 0);
    }

    /// set_layout records a strided Layout (e.g. for a transpose
    /// view). Subsequent reads return the explicit entry.
    #[test]
    fn layout_explicit_strided_is_remembered() {
        use fuel_ir::DimVec;
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Transpose,
            inputs: vec![],
            shape: Shape::from_dims(&[3, 2]),
            dtype: DType::F32,
        });
        // Strided view: shape stays [3, 2] (the post-transpose view),
        // strides are [1, 3] (row-of-output is column-of-source).
        let l = Layout::new(
            Shape::from_dims(&[3, 2]),
            fuel_ir::StrideVec::from_slice(&[1_isize, 3]),
            0,
        );
        g.set_layout(id, l);
        assert!(g.has_explicit_layout(id));
        assert_eq!(g.explicit_layout_count(), 1);
        assert_eq!(g.layout(id).stride(), &[1, 3]);
        assert!(!g.layout(id).is_contiguous());
    }

    /// set_layout twice with different layouts overwrites.
    #[test]
    fn layout_set_overwrites() {
        use fuel_ir::StrideVec;
        let mut g = Graph::new();
        let id = g.push(Node {
            op: Op::Transpose,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        g.set_layout(id, Layout::contiguous(Shape::from_dims(&[2])));
        g.set_layout(
            id,
            Layout::new(Shape::from_dims(&[2]), StrideVec::from_slice(&[3_isize]), 1),
        );
        assert_eq!(g.layout(id).start_offset(), 1);
        assert_eq!(g.explicit_layout_count(), 1);
    }

    /// Phase 7.5 work item G — empty-map invariants. End-to-end
    /// tests of the storage map (with real `Storage` values) live in
    /// fuel-core's `tensor::node_handle_tests`, since constructing a
    /// real `Storage` requires a `DynBackendStorage` implementation
    /// from a backend crate, and fuel-graph deliberately depends only
    /// on fuel-core-types.
    #[test]
    fn graph_storage_map_starts_empty() {
        let g = Graph::new();
        assert_eq!(g.storage_len(), 0);
        assert!(!g.has_storage(NodeId(0)));
        assert!(g.storage_for(NodeId(0)).is_none());
    }

    #[test]
    fn conv2d_builder_emits_conv2d_node_with_right_shape() {
        // k=3 s=1 p=1 keeps H and W.
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 2 * 4 * 4], Shape::from_dims(&[1, 2, 4, 4]), cpu_dev());
        let w = x.const_f32_like(vec![0.0_f32; 3 * 2 * 3 * 3], Shape::from_dims(&[3, 2, 3, 3]));
        let b = x.const_f32_like(vec![0.0_f32; 3], Shape::from_dims(&[3]));
        let y = x.conv2d(&w, Some(&b), (1, 1), (1, 1), 1);
        assert_eq!(y.shape().dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn conv2d_builder_stride_and_no_padding() {
        // k=3 s=2 p=0 on H=W=8 gives (8-3)/2+1 = 3.
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 2 * 8 * 8], Shape::from_dims(&[1, 2, 8, 8]), cpu_dev());
        let w = x.const_f32_like(vec![0.0_f32; 4 * 2 * 3 * 3], Shape::from_dims(&[4, 2, 3, 3]));
        let y = x.conv2d(&w, None, (2, 2), (0, 0), 1);
        assert_eq!(y.shape().dims(), &[1, 4, 3, 3]);
    }

    #[test]
    fn conv2d_builder_depthwise_groups() {
        // groups=Cin=Cout=4 is the depthwise case. Weight per channel is [Cin/groups=1, kH, kW].
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 4 * 4 * 4], Shape::from_dims(&[1, 4, 4, 4]), cpu_dev());
        let w = x.const_f32_like(vec![0.0_f32; 4 * 1 * 3 * 3], Shape::from_dims(&[4, 1, 3, 3]));
        let y = x.conv2d(&w, None, (1, 1), (1, 1), 4);
        assert_eq!(y.shape().dims(), &[1, 4, 4, 4]);
    }

    #[test]
    fn conv_transpose2d_builder_emits_node_with_right_shape() {
        // Hin=4, Kh=3, s=2, pad=1, out_pad=1 → Hout = (4-1)*2 + (3-1) + 1 + 1 - 2 = 8.
        let x = Tensor::from_f32(vec![0.0_f32; 1 * 2 * 4 * 4], Shape::from_dims(&[1, 2, 4, 4]), cpu_dev());
        let w = x.const_f32_like(vec![0.0_f32; 2 * 3 * 3 * 3], Shape::from_dims(&[2, 3, 3, 3]));
        let y = x.conv_transpose2d(&w, (2, 2), (1, 1), (1, 1), (1, 1), 1);
        assert_eq!(y.shape().dims(), &[1, 3, 8, 8]);
    }

    #[test]
    fn conv_transpose1d_builder_shape_stride_2_pad_1() {
        // Lin=4, K=3, s=2, pad=1, out_pad=1, dil=1
        // Lout = (4-1)*2 + (3-1) + 1 + 1 - 2 = 8.
        let x = Tensor::from_f32(
            vec![0.0_f32; 1 * 2 * 4], Shape::from_dims(&[1, 2, 4]), cpu_dev(),
        );
        let w = x.const_f32_like(
            vec![0.0_f32; 2 * 3 * 3], Shape::from_dims(&[2, 3, 3]),
        );
        let y = x.conv_transpose1d(&w, 2, 1, 1, 1, 1);
        assert_eq!(y.shape().dims(), &[1, 3, 8]);
    }

    #[test]
    fn conv_transpose1d_builder_shape_stride_4_no_pad() {
        // Lin=2, K=4, s=4, pad=0, out_pad=0, dil=1
        // Lout = (2-1)*4 + (4-1) + 0 + 1 - 0 = 8.
        let x = Tensor::from_f32(
            vec![0.0_f32; 1 * 1 * 2], Shape::from_dims(&[1, 1, 2]), cpu_dev(),
        );
        let w = x.const_f32_like(
            vec![0.0_f32; 1 * 1 * 4], Shape::from_dims(&[1, 1, 4]),
        );
        let y = x.conv_transpose1d(&w, 4, 0, 0, 1, 1);
        assert_eq!(y.shape().dims(), &[1, 1, 8]);
    }

    #[test]
    fn conv_transpose1d_builder_with_groups() {
        // groups=2: input Cin=4 splits into 2 groups of 2; weight
        // Cin=4 first dim matches input; Cout/group=3 → total Cout=6.
        // Lin=3, K=3, s=1, pad=0, out_pad=0, dil=1 → Lout = (3-1)*1 + 2 + 0 + 1 = 5.
        let x = Tensor::from_f32(
            vec![0.0_f32; 1 * 4 * 3], Shape::from_dims(&[1, 4, 3]), cpu_dev(),
        );
        let w = x.const_f32_like(
            vec![0.0_f32; 4 * 3 * 3], Shape::from_dims(&[4, 3, 3]),
        );
        let y = x.conv_transpose1d(&w, 1, 0, 0, 1, 2);
        assert_eq!(y.shape().dims(), &[1, 6, 5]);
    }

    #[test]
    fn conv2d_backward_grads_have_input_shapes() {
        // Forward Y = conv2d(X, W) with stride=1, pad=1, groups=1 keeps H,W.
        // Backward should produce dX with X's shape and dW with W's shape.
        let x = Tensor::from_f32(
            (0..(1*2*4*4)).map(|i| (i as f32) * 0.05 - 0.5).collect::<Vec<f32>>(),
            Shape::from_dims(&[1, 2, 4, 4]),
            cpu_dev(),
        );
        let w = x.const_f32_like(
            (0..(3*2*3*3)).map(|i| (i as f32) * 0.07 - 0.4).collect::<Vec<f32>>(),
            Shape::from_dims(&[3, 2, 3, 3]));
        let y = x.conv2d(&w, None, (1, 1), (1, 1), 1);
        let scalar_out = y.sum_all();
        let grads = scalar_out.backward();
        let dx = grads.get(&x).expect("conv2d backward produced no dX");
        let dw = grads.get(&w).expect("conv2d backward produced no dW");
        assert_eq!(dx.shape().dims(), x.shape().dims());
        assert_eq!(dw.shape().dims(), w.shape().dims());
    }

    #[test]
    fn destructive_input_move_is_some_zero() {
        assert_eq!(Op::Move { target: DeviceLocation::Cpu }.destructive_input(), Some(0));
    }

    #[test]
    fn destructive_input_non_destructive_ops_are_none() {
        assert_eq!(Op::Copy { target: DeviceLocation::Cpu }.destructive_input(), None);
        assert_eq!(Op::Add.destructive_input(), None);
        assert_eq!(Op::Mul.destructive_input(), None);
        assert_eq!(Op::MatMul.destructive_input(), None);
        assert_eq!(Op::Relu.destructive_input(), None);
    }

    #[test]
    fn move_to_device_emits_op_move_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let moved = a.move_to_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let g = moved.graph().read().unwrap();
        match &g.node(moved.id()).op {
            Op::Move { target } => {
                assert_eq!(*target, DeviceLocation::Vulkan { gpu_id: 0 });
            }
            other => panic!("expected Op::Move, got {other:?}"),
        }
        assert_eq!(g.node(moved.id()).inputs, vec![a.id()]);
        // Output has the input's shape (residency changes, data doesn't).
        assert_eq!(g.node(moved.id()).shape.elem_count(), 2);
    }

    #[test]
    fn copy_to_device_emits_op_copy_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.copy_to_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let g = b.graph().read().unwrap();
        match &g.node(b.id()).op {
            Op::Copy { target } => {
                assert_eq!(*target, DeviceLocation::Vulkan { gpu_id: 0 });
            }
            other => panic!("expected Op::Copy, got {other:?}"),
        }
        // Source stays resident: the original `a` node still in the graph
        // and is the input of the Copy node.
        assert_eq!(g.node(b.id()).inputs, vec![a.id()]);
    }

    #[test]
    fn release_builder_emits_op_release_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let released = a.release();
        let g = released.graph().read().unwrap();
        assert!(matches!(g.node(released.id()).op, Op::Release));
        assert_eq!(g.node(released.id()).inputs, vec![a.id()]);
        // Output is a zero-element marker.
        assert_eq!(g.node(released.id()).shape.elem_count(), 0);
    }

    #[test]
    fn placement_is_none_by_default() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        assert_eq!(a.placement(), None);
    }

    #[test]
    fn on_device_sets_placement_hint() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        // Only tag the Add node; the const leaves remain unplaced.
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        assert_eq!(c.placement(), Some(DeviceLocation::Vulkan { gpu_id: 0 }));
        assert_eq!(a.placement(), None);
        assert_eq!(b.placement(), None);
    }

    #[test]
    fn placement_survives_graph_re_reads() {
        let a = Tensor::from_f32(vec![1.0], Shape::from_dims(&[1]), cpu_dev());
        let tagged = a.clone().on_device(DeviceLocation::Cpu);
        // Re-read from a fresh borrow — round-trips through the side-table.
        assert_eq!(tagged.graph().read().unwrap().placement(tagged.id()), Some(DeviceLocation::Cpu));
    }

    #[test]
    fn from_f32_creates_single_const_node() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        assert_eq!(a.graph().read().unwrap().len(), 1);
        assert_eq!(a.shape().dims(), &[3]);
        assert_eq!(a.dtype(), DType::F32);
        let node = a.graph().read().unwrap().node(a.id()).clone();
        // Phase 7.5 G2: factory now emits Op::Const + slot.
        assert!(matches!(node.op, Op::Const));
        assert!(node.inputs.is_empty());
        // The slot is populated with F32 storage.
        let slot = a.graph().read().unwrap().storage_for(a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::F32);
    }

    #[test]
    fn add_appends_a_node_and_tracks_inputs() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b);
        assert_eq!(c.graph().read().unwrap().len(), 3); // const, const, add
        let node = c.graph().read().unwrap().node(c.id()).clone();
        assert!(matches!(node.op, Op::Add));
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.inputs[0], a.id());
        assert_eq!(node.inputs[1], b.id());
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn chained_ops_all_share_one_graph() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]));
        let c = a.add(&b).mul(&a).sqr().relu();
        assert_eq!(c.graph().read().unwrap().len(), 6); // 2 consts + add + mul + sqr + relu
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn matmul_validates_shapes_and_produces_correct_output_shape() {
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn add_panics_on_shape_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "inner dim mismatch")]
    fn matmul_panics_on_inner_dim_mismatch() {
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0; 8], Shape::from_dims(&[4, 2]));
        let _ = a.matmul(&b);
    }

    #[test]
    #[should_panic(expected = "must live on the same graph")]
    fn cross_graph_op_is_rejected() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = Tensor::from_f32(vec![4.0, 5.0, 6.0], Shape::from_dims(&[3]), cpu_dev());
        let _ = a.add(&b);
    }

    // ----- multi-dtype graph builders -----

    #[test]
    fn from_f64_tags_node_with_f64_dtype() {
        let a = Tensor::from_f64(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        assert_eq!(a.dtype(), DType::F64);
        let node = a.graph().read().unwrap().node(a.id()).clone();
        // Phase 7.5 G2: slot-rooted Const, dtype validated via slot.
        assert!(matches!(node.op, Op::Const));
        let slot = a.graph().read().unwrap().storage_for(a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::F64);
    }

    #[test]
    fn from_bf16_tags_node_with_bf16_dtype() {
        let a = Tensor::from_bf16(
            vec![bf16::from_f32(1.0), bf16::from_f32(2.0)],
            Shape::from_dims(&[2]),
            cpu_dev(),
        );
        assert_eq!(a.dtype(), DType::BF16);
        let node = a.graph().read().unwrap().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const));
        let slot = a.graph().read().unwrap().storage_for(a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::BF16);
    }

    #[test]
    fn from_f16_tags_node_with_f16_dtype() {
        let a = Tensor::from_f16(
            vec![f16::from_f32(1.0), f16::from_f32(2.0)],
            Shape::from_dims(&[2]),
            cpu_dev(),
        );
        assert_eq!(a.dtype(), DType::F16);
        let node = a.graph().read().unwrap().node(a.id()).clone();
        assert!(matches!(node.op, Op::Const));
        let slot = a.graph().read().unwrap().storage_for(a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::F16);
    }

    #[test]
    #[should_panic(expected = "dtype mismatch")]
    fn add_panics_on_mixed_dtype() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f64_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.add(&b);
    }

    // ----- transpose -----

    #[test]
    fn transpose_swaps_shape_dims() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::from_dims(&[2, 3]), cpu_dev());
        let t = a.transpose();
        assert_eq!(t.shape().dims(), &[3, 2]);
        let node = t.graph().read().unwrap().node(t.id()).clone();
        assert!(matches!(node.op, Op::Transpose));
        assert_eq!(node.inputs, vec![a.id()]);
    }

    #[test]
    #[should_panic(expected = "rank ≥ 2")]
    fn transpose_rejects_rank_1() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let _ = a.transpose();
    }

    #[test]
    fn transpose_on_rank_3_swaps_last_two_dims() {
        // [2, 3, 4] → [2, 4, 3]
        let a = Tensor::from_f32(vec![0.0_f32; 24], Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let t = a.transpose();
        assert_eq!(t.shape().dims(), &[2, 4, 3]);
    }

    // ----- additional builder validation tests -----

    #[test]
    fn matmul_rank_3_batched_shape() {
        // [2, 3, 4] @ [2, 4, 5] → [2, 3, 5]
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 40], Shape::from_dims(&[2, 4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    #[should_panic(expected = "batch dim mismatch")]
    fn matmul_rank_3_rejects_batch_dim_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 60], Shape::from_dims(&[3, 4, 5]));
        let _ = a.matmul(&b);
    }

    #[test]
    fn matmul_auto_broadcasts_rank_2_rhs_against_batched_lhs() {
        // [batch=2, seq=3, k=4] @ [k=4, n=5] → [2, 3, 5]. This is the
        // canonical "linear layer across a batch" pattern and should
        // Just Work without an explicit broadcast_to on the RHS.
        let a = Tensor::from_f32(vec![0.0; 24], Shape::from_dims(&[2, 3, 4]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 20], Shape::from_dims(&[4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn matmul_auto_broadcasts_rank_2_lhs_against_batched_rhs() {
        // [m=3, k=4] @ [batch=2, k=4, n=5] → [2, 3, 5].
        let a = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 40], Shape::from_dims(&[2, 4, 5]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().dims(), &[2, 3, 5]);
    }

    #[test]
    fn concat_output_shape_sums_along_dim() {
        // [2, 3] concat [2, 4] along dim 1 → [2, 7]
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 8], Shape::from_dims(&[2, 4]));
        let c = a.concat(&b, 1);
        assert_eq!(c.shape().dims(), &[2, 7]);
    }

    #[test]
    #[should_panic(expected = "non-dim shapes")]
    fn concat_rejects_nondim_shape_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 12], Shape::from_dims(&[3, 4]));
        let _ = a.concat(&b, 1);
    }

    #[test]
    #[should_panic(expected = "rank mismatch")]
    fn concat_rejects_rank_mismatch() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 6], Shape::from_dims(&[6]));
        let _ = a.concat(&b, 0);
    }

    #[test]
    fn slice_shrinks_only_the_slice_dim() {
        // [3, 4] slice dim 1, start 1, len 2 → [3, 2]
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let s = x.slice(1, 1, 2);
        assert_eq!(s.shape().dims(), &[3, 2]);
    }

    #[test]
    #[should_panic(expected = "exceeds dim size")]
    fn slice_rejects_out_of_bounds_range() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let _ = x.slice(1, 1, 3); // start=1, len=3 → would need dim>=4
    }

    #[test]
    fn broadcast_add_shape_promotes_to_common_shape() {
        // [4, 1] + [1, 3] → [4, 3]
        let a = Tensor::from_f32(vec![0.0; 4], Shape::from_dims(&[4, 1]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 3], Shape::from_dims(&[1, 3]));
        let c = a.broadcast_add(&b);
        assert_eq!(c.shape().dims(), &[4, 3]);
    }

    #[test]
    fn broadcast_sub_pads_shorter_shape_with_leading_ones() {
        // [3] - [2, 3] → [2, 3]
        let a = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 6], Shape::from_dims(&[2, 3]));
        let c = a.broadcast_sub(&b);
        assert_eq!(c.shape().dims(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn broadcast_add_rejects_incompatible_shapes() {
        let a = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![0.0; 8], Shape::from_dims(&[2, 4]));
        let _ = a.broadcast_add(&b);
    }

    #[test]
    fn argmax_dim_is_u32_and_removes_reduced_dim() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let am = x.argmax_dim(1);
        assert_eq!(am.dtype(), DType::U32);
        assert_eq!(am.shape().dims(), &[2]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn argmax_dim_rejects_bad_dim() {
        let x = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let _ = x.argmax_dim(5);
    }

    #[test]
    fn index_add_shape_validation() {
        let base = Tensor::from_f32(vec![0.0; 10], Shape::from_dims(&[10]), cpu_dev());
        let idx = base.const_u32_like(vec![1, 3, 5], Shape::from_dims(&[3]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        let out = base.index_add(0, &idx, &src);
        assert_eq!(out.shape().dims(), &[10]);
    }

    #[test]
    #[should_panic(expected = "dtypes must match")]
    fn index_add_rejects_dtype_mismatch() {
        let base = Tensor::from_f32(vec![0.0; 5], Shape::from_dims(&[5]), cpu_dev());
        let idx = base.const_u32_like(vec![0, 2], Shape::from_dims(&[2]));
        let src = base.const_f64_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = base.index_add(0, &idx, &src);
    }

    #[test]
    fn scatter_add_validates_index_matches_src() {
        let base = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let idx = base.const_u32_like(vec![0, 2, 1, 0], Shape::from_dims(&[2, 2]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        let out = base.scatter_add(1, &idx, &src);
        assert_eq!(out.shape().dims(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "same shape")]
    fn scatter_add_rejects_index_src_shape_mismatch() {
        let base = Tensor::from_f32(vec![0.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let idx = base.const_u32_like(vec![0, 1], Shape::from_dims(&[2]));
        let src = base.const_f32_like(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]));
        let _ = base.scatter_add(1, &idx, &src);
    }

    #[test]
    fn reduce_sum_to_validates_compatibility() {
        // [3, 4] can reduce to [4] (sum along dim 0) or [3, 1] (sum along dim 1).
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let r1 = x.reduce_sum_to(Shape::from_dims(&[4]));
        assert_eq!(r1.shape().dims(), &[4]);
        let r2 = x.reduce_sum_to(Shape::from_dims(&[3, 1]));
        assert_eq!(r2.shape().dims(), &[3, 1]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn reduce_sum_to_rejects_non_broadcast_target() {
        // [3, 4] cannot reduce to [3, 2] — target must be broadcast-into-source.
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let _ = x.reduce_sum_to(Shape::from_dims(&[3, 2]));
    }

    #[test]
    fn unsqueeze_inserts_size_one_dim() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                 Shape::from_dims(&[2, 3]), cpu_dev());
        // Insert at position 0: [2, 3] -> [1, 2, 3]
        let y0 = x.unsqueeze(0);
        assert_eq!(y0.shape().dims(), &[1, 2, 3]);
        // Insert at position 1: [2, 3] -> [2, 1, 3]
        let y1 = x.unsqueeze(1);
        assert_eq!(y1.shape().dims(), &[2, 1, 3]);
        // Insert at the end (rank): [2, 3] -> [2, 3, 1]
        let y2 = x.unsqueeze(2);
        assert_eq!(y2.shape().dims(), &[2, 3, 1]);
        // Op variant correctness.
        let g = x.graph().read().unwrap();
        assert!(matches!(g.node(y0.id()).op, Op::Unsqueeze { dim: 0 }));
        assert!(matches!(g.node(y1.id()).op, Op::Unsqueeze { dim: 1 }));
        assert!(matches!(g.node(y2.id()).op, Op::Unsqueeze { dim: 2 }));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn unsqueeze_rejects_dim_above_rank() {
        let x = Tensor::from_f32(vec![1.0; 4], Shape::from_dims(&[4]), cpu_dev());
        // dim=2 > rank=1 → panic.
        let _ = x.unsqueeze(2);
    }

    #[test]
    fn unsqueeze_layout_side_table_populated() {
        // After Graph::push auto-derives the Layout side-table for view
        // ops, an unsqueeze node should have an explicit layout entry
        // with a stride-0 axis at the inserted position.
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0],
                                 Shape::from_dims(&[2, 2]), cpu_dev());
        let y = x.unsqueeze(1);
        let g = x.graph().read().unwrap();
        assert!(g.has_explicit_layout(y.id()),
            "Graph::push should auto-populate layout for view ops");
        let l = g.layout(y.id());
        assert_eq!(l.shape().dims(), &[2, 1, 2]);
        // Stride at the inserted axis should be 0 (per the convention
        // chosen in Layout::unsqueeze).
        assert_eq!(l.stride()[1], 0);
    }

    #[test]
    fn reduce_max_to_validates_compatibility() {
        // [3, 4] can reduce to [4] (max along dim 0) or [3, 1] (max along dim 1).
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let r1 = x.reduce_max_to(Shape::from_dims(&[4]));
        assert_eq!(r1.shape().dims(), &[4]);
        assert!(matches!(r1.graph().read().unwrap().node(r1.id()).op, Op::ReduceMaxTo(_)));
        let r2 = x.reduce_max_to(Shape::from_dims(&[3, 1]));
        assert_eq!(r2.shape().dims(), &[3, 1]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn reduce_max_to_rejects_non_broadcast_target() {
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let _ = x.reduce_max_to(Shape::from_dims(&[3, 2]));
    }

    #[test]
    fn reshape_preserves_element_count() {
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let r = x.reshape(Shape::from_dims(&[2, 6]));
        assert_eq!(r.shape().dims(), &[2, 6]);
    }

    #[test]
    #[should_panic(expected = "element count mismatch")]
    fn reshape_rejects_different_element_count() {
        let x = Tensor::from_f32(vec![0.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let _ = x.reshape(Shape::from_dims(&[3, 3]));
    }

    #[test]
    #[should_panic(expected = "same graph")]
    fn concat_across_graphs_panics() {
        let a = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]), cpu_dev());
        let b = Tensor::from_f32(vec![0.0; 3], Shape::from_dims(&[3]), cpu_dev());
        let _ = a.concat(&b, 0);
    }

    #[test]
    fn scalar_ops_preserve_shape_and_dtype() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.add_scalar(5.0).mul_scalar(2.0).powi(2).clamp(0.0, 100.0);
        assert_eq!(y.shape().dims(), &[3]);
        assert_eq!(y.dtype(), DType::F32);
    }

    #[test]
    fn maximum_requires_matching_shapes() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![4.0, 1.0, 5.0], Shape::from_dims(&[3]));
        let m = a.maximum(&b);
        assert_eq!(m.shape().dims(), &[3]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn maximum_rejects_shape_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0, 2.0], Shape::from_dims(&[2]));
        let _ = a.maximum(&b);
    }

    // ----- topo_order -----

    #[test]
    fn topo_order_places_inputs_before_dependents() {
        // Build: c = (a + b) * a
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let sum = a.add(&b);
        let c = sum.mul(&a);
        let order = topo_order(&c.graph().read().unwrap(), c.id());
        // The order should contain exactly 4 nodes (a, b, sum, c) and
        // place each input strictly before its dependents.
        assert_eq!(order.len(), 4);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a.id()) < pos(sum.id()));
        assert!(pos(b.id()) < pos(sum.id()));
        assert!(pos(sum.id()) < pos(c.id()));
        assert!(pos(a.id()) < pos(c.id()));
    }

    #[test]
    fn topo_order_visits_each_node_once_when_shared() {
        // (a + a) — `a` appears twice in the Add's inputs but the topo
        // pass should still visit it exactly once.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let double = a.add(&a);
        let order = topo_order(&double.graph().read().unwrap(), double.id());
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], a.id());
        assert_eq!(order[1], double.id());
    }

    #[test]
    fn topo_order_multi_unions_multiple_roots() {
        // Build a DAG where two "output" nodes share a common input.
        //
        //      a ─┬─> add1 (= a + b)  ← root_1
        //         └─> add2 (= a + c)  ← root_2
        //      b ───> add1
        //      c ───> add2
        //
        // topo_order_multi(&[add1, add2]) must contain all 5 nodes
        // (a, b, c, add1, add2) with a before add1/add2 and b before
        // add1 and c before add2.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.const_f32_like(vec![5.0, 6.0], Shape::from_dims(&[2]));
        let add1 = a.add(&b);
        let add2 = a.add(&c);
        let order = topo_order_multi(
            &add1.graph().read().unwrap(),
            &[add1.id(), add2.id()],
        );
        assert_eq!(order.len(), 5);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(a.id()) < pos(add1.id()));
        assert!(pos(a.id()) < pos(add2.id()));
        assert!(pos(b.id()) < pos(add1.id()));
        assert!(pos(c.id()) < pos(add2.id()));
    }

    // ----- backward -----

    #[test]
    fn backward_of_lone_const_seeds_ones() {
        // backward(a) gives a = 1s (the root's upstream is a ones tensor,
        // which is the gradient stored for the root itself).
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let grads = a.backward();
        let g_a = grads.get(&a).expect("root gets a seed gradient");
        // The seed is a Const ones node of matching shape. Phase 7.5
        // G2: gradient consts are slot-rooted Op::Const — no
        // host-side ConstData on the node.
        let node = g_a.graph().read().unwrap().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Const));
        let slot = g_a.graph().read().unwrap().storage_for(g_a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::F32);
        assert_eq!(g_a.shape().dims(), &[3]);
    }

    #[test]
    fn backward_of_add_passes_upstream_through() {
        // c = a + b  ⇒  dc/da = 1, dc/db = 1.
        // Upstream seed is a ones tensor. So grad_a and grad_b are both
        // the same ones node (no new math emitted for Add's backward).
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let grads = c.backward();
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        assert_eq!(g_a.id(), g_b.id(), "Add backward routes upstream unchanged");
        assert_eq!(g_a.shape().dims(), &[2]);
    }

    #[test]
    fn backward_of_mul_emits_two_mul_nodes() {
        // c = a * b  ⇒  dc/da = b, dc/db = a (upstream is 1s).
        // The backward pass should emit two new Mul nodes (upstream * b,
        // upstream * a).
        let a = Tensor::from_f32(vec![2.0, 3.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![5.0, 7.0], Shape::from_dims(&[2]));
        let c = a.mul(&b);
        let nodes_before = c.graph().read().unwrap().len();
        let grads = c.backward();
        let nodes_after = grads.graph.read().unwrap().len();
        // Backward adds: one ones const + two Mul nodes = 3 new nodes.
        assert_eq!(nodes_after - nodes_before, 3);
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        // Both gradient nodes should be Mul nodes.
        let node_a = g_a.graph().read().unwrap().node(g_a.id()).clone();
        let node_b = g_b.graph().read().unwrap().node(g_b.id()).clone();
        assert!(matches!(node_a.op, Op::Mul));
        assert!(matches!(node_b.op, Op::Mul));
    }

    #[test]
    fn backward_accumulates_when_node_used_twice() {
        // c = a * a  ⇒  dc/da = 2a via two separate contributions.
        // After backward, the gradient for a should be an Add node
        // combining the two Mul contributions (one from each input slot
        // of the forward Mul).
        let a = Tensor::from_f32(vec![3.0, 5.0], Shape::from_dims(&[2]), cpu_dev());
        let c = a.mul(&a);
        let grads = c.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().read().unwrap().node(g_a.id()).clone();
        // The final gradient for `a` is an Add combining the two
        // upstream-times-other-input contributions.
        assert!(
            matches!(node.op, Op::Add),
            "expected accumulated gradient to be an Add node, got {:?}",
            node.op,
        );
    }

    #[test]
    fn backward_of_matmul_emits_transpose_and_matmul_nodes() {
        // Forward: Y = A @ B,  A:[2,3], B:[3,4], Y:[2,4].
        // Backward: dA = dY @ B^T (shape [2,3]),  dB = A^T @ dY (shape [3,4]).
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0; 12], Shape::from_dims(&[3, 4]));
        let y = a.matmul(&b);
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let g_b = grads.get(&b).unwrap();
        // Gradient shapes must match the forward shapes.
        assert_eq!(g_a.shape().dims(), &[2, 3]);
        assert_eq!(g_b.shape().dims(), &[3, 4]);
        // Both should be MatMul nodes (the outermost op of each gradient).
        let node_a = g_a.graph().read().unwrap().node(g_a.id()).clone();
        let node_b = g_b.graph().read().unwrap().node(g_b.id()).clone();
        assert!(matches!(node_a.op, Op::MatMul));
        assert!(matches!(node_b.op, Op::MatMul));
    }

    // ----- new builder validation -----

    #[test]
    fn cast_tags_node_with_target_dtype() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.cast(DType::F64);
        assert_eq!(b.dtype(), DType::F64);
        assert_eq!(b.shape().dims(), &[2]);
        let node = b.graph().read().unwrap().node(b.id()).clone();
        assert!(matches!(node.op, Op::Cast(DType::F64)));
    }

    #[test]
    fn broadcast_to_accepts_right_aligned_expansion() {
        // [3] broadcasts to [2, 3]: pad with leading 1, expand.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.broadcast_to(Shape::from_dims(&[2, 3]));
        assert_eq!(b.shape().dims(), &[2, 3]);
    }

    #[test]
    fn broadcast_to_accepts_size_one_expansion() {
        // [3, 1] broadcasts to [3, 4]: size-1 dim expands.
        let a = Tensor::from_f32(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3, 1]), cpu_dev());
        let b = a.broadcast_to(Shape::from_dims(&[3, 4]));
        assert_eq!(b.shape().dims(), &[3, 4]);
    }

    #[test]
    #[should_panic(expected = "incompatible")]
    fn broadcast_to_rejects_incompatible_dim() {
        // [3] cannot broadcast to [2, 4] — the source dim 3 must match 4.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let _ = a.broadcast_to(Shape::from_dims(&[2, 4]));
    }

    #[test]
    fn sum_all_produces_rank_zero_output() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let s = a.sum_all();
        assert_eq!(s.shape().dims(), &[] as &[usize]);
    }

    #[test]
    fn sum_dim_removes_reduced_dim_from_shape() {
        let a = Tensor::from_f32(vec![1.0; 24], Shape::from_dims(&[2, 3, 4]), cpu_dev());
        // Reducing dim 1 should give shape [2, 4].
        let s = a.sum_dim(1);
        assert_eq!(s.shape().dims(), &[2, 4]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn sum_dim_rejects_bad_dim() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let _ = a.sum_dim(5);
    }

    #[test]
    fn softmax_and_layer_norm_preserve_shape() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), cpu_dev());
        assert_eq!(a.softmax_last_dim().shape().dims(), &[2, 2]);
        assert_eq!(a.layer_norm_last_dim(1e-5).shape().dims(), &[2, 2]);
    }

    #[test]
    fn neg_sub_div_sqrt_log_sin_cos_tanh_sigmoid_all_build() {
        // Smoke test: every new builder produces a node with the expected
        // shape and dtype. Numerical correctness is exercised in exec.rs.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        for tensor in [
            a.neg(),
            a.sub(&b),
            a.div(&b),
            a.sqrt(),
            a.log(),
            a.sin(),
            a.cos(),
            a.tanh(),
            a.sigmoid(),
            a.step(),
        ] {
            assert_eq!(tensor.shape().dims(), &[3]);
            assert_eq!(tensor.dtype(), DType::F32);
        }
    }

    // ----- index tensors and indexing ops -----

    #[test]
    fn from_u32_tags_node_with_u32_dtype() {
        let a = Tensor::from_u32(vec![1, 2, 3], Shape::from_dims(&[3]), cpu_dev());
        assert_eq!(a.dtype(), DType::U32);
        let node = a.graph().read().unwrap().node(a.id()).clone();
        // Phase 7.5 G2: slot-rooted Const, dtype validated via slot.
        assert!(matches!(node.op, Op::Const));
        let slot = a.graph().read().unwrap().storage_for(a.id()).unwrap();
        assert_eq!(slot.read().unwrap().dtype(), DType::U32);
    }

    #[test]
    fn index_select_produces_shape_with_dim_replaced() {
        let data = Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        let idx = data.const_u32_like(vec![0, 2, 1, 0, 2], Shape::from_dims(&[5]));
        let out = data.index_select(0, &idx);
        assert_eq!(out.shape().dims(), &[5, 4]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    #[should_panic(expected = "must be U32")]
    fn index_select_rejects_float_index() {
        let data = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let bad = data.const_f32_like(vec![0.0, 1.0], Shape::from_dims(&[2]));
        let _ = data.index_select(0, &bad);
    }

    #[test]
    #[should_panic(expected = "must be rank 1")]
    fn index_select_rejects_multi_dim_index() {
        let data = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let idx = data.const_u32_like(vec![0, 1, 0, 1], Shape::from_dims(&[2, 2]));
        let _ = data.index_select(0, &idx);
    }

    #[test]
    fn gather_output_shape_matches_index_shape() {
        let data = Tensor::from_f32(vec![1.0; 12], Shape::from_dims(&[3, 4]), cpu_dev());
        // Index shape [2, 5] — same rank as data (rank 2).
        let idx = data.const_u32_like(vec![0; 10], Shape::from_dims(&[2, 5]));
        let out = data.gather(1, &idx);
        assert_eq!(out.shape().dims(), &[2, 5]);
    }

    #[test]
    #[should_panic(expected = "same rank")]
    fn gather_rejects_rank_mismatch() {
        let data = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        // Rank-1 index for rank-2 data → error.
        let idx = data.const_u32_like(vec![0, 1, 0], Shape::from_dims(&[3]));
        let _ = data.gather(1, &idx);
    }

    #[test]
    fn backward_of_relu_emits_step_node() {
        // Before: this used to panic. After adding Step + Relu backward,
        // it should successfully emit a backward graph rooted in a Mul
        // whose second input is a Step node.
        let a = Tensor::from_f32(vec![-1.0, 2.0, -3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = a.relu();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().read().unwrap().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Mul));
        // Find a Step node somewhere in the node's inputs.
        let any_step = node.inputs.iter().any(|&id| {
            matches!(
                g_a.graph().read().unwrap().node(id).op,
                Op::Step,
            )
        });
        assert!(any_step, "Relu backward must reference a Step node");
    }

    #[test]
    fn recip_and_abs_builders_produce_unary_nodes() {
        // Smoke test: Tensor::recip()/abs() build single-input nodes
        // with the expected op variant and shape passthrough.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let r = a.recip();
        let b = a.abs();
        assert_eq!(r.shape().dims(), &[3]);
        assert_eq!(b.shape().dims(), &[3]);
        let r_node = r.graph().read().unwrap().node(r.id()).clone();
        let b_node = b.graph().read().unwrap().node(b.id()).clone();
        assert!(matches!(r_node.op, Op::Recip));
        assert!(matches!(b_node.op, Op::Abs));
        assert_eq!(r_node.inputs, vec![a.id()]);
        assert_eq!(b_node.inputs, vec![a.id()]);
    }

    #[test]
    fn backward_of_recip_is_neg_of_upstream_times_y_squared() {
        // y = 1/x ⇒ dy/dx = -y². The backward graph should be a Neg
        // wrapping a Mul whose inputs include a Sqr node — and the Sqr
        // should reference the forward Recip output, not a fresh recompute.
        let a = Tensor::from_f32(vec![2.0, 4.0], Shape::from_dims(&[2]), cpu_dev());
        let y = a.recip();
        let y_id = y.id();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let g = g_a.graph().read().unwrap();
        let neg_node = g.node(g_a.id()).clone();
        assert!(matches!(neg_node.op, Op::Neg), "outermost backward op must be Neg");
        let mul_id = neg_node.inputs[0];
        let mul_node = g.node(mul_id).clone();
        assert!(matches!(mul_node.op, Op::Mul));
        // One of the Mul's inputs is a Sqr node fed by the forward Recip.
        let sqr_id = mul_node.inputs.iter().copied().find(|&id| {
            matches!(g.node(id).op, Op::Sqr)
        }).expect("backward chain must contain a Sqr node");
        let sqr_node = g.node(sqr_id).clone();
        assert_eq!(sqr_node.inputs, vec![y_id],
            "Sqr's input must be the forward Recip output, not a recompute");
    }

    #[test]
    fn backward_of_abs_emits_sign_mul() {
        // y = |x|, dy/dx = sign(x). After PR B2 landed Op::Sign, the
        // backward chain shrinks from 5 nodes (step+neg+step+sub+mul)
        // to 2 (sign+mul): grad_x = upstream * sign(x), where Sign is
        // a single primitive that returns -1/0/1 directly.
        let a = Tensor::from_f32(vec![-2.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let y = a.abs();
        let grads = y.backward();
        let g_a = grads.get(&a).unwrap();
        let g = g_a.graph().read().unwrap();
        let mul_node = g.node(g_a.id()).clone();
        assert!(matches!(mul_node.op, Op::Mul));
        // Exactly one input of the Mul is a Sign node (the other is
        // the upstream gradient — a Const-rooted ones-tensor here).
        let sign_count = mul_node.inputs.iter().filter(|&&id| {
            matches!(g.node(id).op, Op::Sign)
        }).count();
        assert_eq!(sign_count, 1,
            "Abs backward must reference exactly 1 Sign node, got {sign_count}");
        // The Sign node should feed off the original input `a`.
        let sign_id = mul_node.inputs.iter().copied().find(|&id| {
            matches!(g.node(id).op, Op::Sign)
        }).unwrap();
        assert_eq!(g.node(sign_id).inputs, vec![a.id()],
            "Sign's input must be the forward input `a`");
    }

    #[test]
    fn eq_builder_produces_u8_output_with_input_shape() {
        // Tensor::eq builds a binary node whose dtype is U8 regardless
        // of input dtype, with shape == lhs.shape() (and == rhs.shape()).
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![1.0, 5.0, 3.0], Shape::from_dims(&[3]));
        let m = a.eq(&b);
        assert_eq!(m.shape().dims(), &[3]);
        assert_eq!(m.dtype(), DType::U8, "eq output must be U8");
        let m_node = m.graph().read().unwrap().node(m.id()).clone();
        assert!(matches!(m_node.op, Op::Equal));
        assert_eq!(m_node.inputs, vec![a.id(), b.id()]);
    }

    #[test]
    fn try_unsqueeze_returns_err_on_dim_above_rank() {
        // try_unsqueeze surfaces bad dim as Err; unsqueeze panics
        // on the same input. Both share the validation message.
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let err = a.try_unsqueeze(5).expect_err("dim above rank must error");
        assert!(format!("{err:?}").contains("out of bounds"),
            "error must mention bounds, got: {err:?}");
        // Happy path still works.
        let ok = a.try_unsqueeze(0).expect("dim=0 valid for any rank");
        assert_eq!(ok.shape().dims(), &[1, 3]);
    }

    #[test]
    fn try_reshape_returns_err_on_count_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let err = a.try_reshape(Shape::from_dims(&[3])).expect_err("count mismatch must error");
        assert!(format!("{err:?}").contains("element count mismatch"),
            "error must mention count mismatch, got: {err:?}");
        let ok = a.try_reshape(Shape::from_dims(&[2, 2])).expect("matching count is ok");
        assert_eq!(ok.shape().dims(), &[2, 2]);
    }

    #[test]
    fn try_broadcast_to_returns_err_on_incompatible_shape() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let err = a.try_broadcast_to(Shape::from_dims(&[2, 4]))
            .expect_err("source dim 3 cannot broadcast to dim 4");
        assert!(format!("{err:?}").contains("incompatible"),
            "error must mention incompatibility, got: {err:?}");
        let ok = a.try_broadcast_to(Shape::from_dims(&[2, 3])).expect("dim 3 broadcasts to dim 3");
        assert_eq!(ok.shape().dims(), &[2, 3]);
    }

    #[test]
    fn try_transpose_returns_err_on_rank_below_2() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let err = a.try_transpose().expect_err("rank-1 input cannot transpose");
        assert!(format!("{err:?}").contains("rank ≥ 2"),
            "error must mention rank requirement, got: {err:?}");
        let m = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let mt = m.try_transpose().expect("rank-2 transposes ok");
        assert_eq!(mt.shape().dims(), &[3, 2]);
    }

    #[test]
    fn try_permute_returns_err_on_bad_axes() {
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        // Wrong axes length:
        let err = a.try_permute(&[0]).expect_err("axes length mismatch");
        assert!(format!("{err:?}").contains("must equal tensor rank"),
            "error must mention length, got: {err:?}");
        // Out-of-bounds axis:
        let err = a.try_permute(&[0, 5]).expect_err("axis 5 out of rank 2");
        assert!(format!("{err:?}").contains("out of bounds"));
        // Duplicate axis:
        let err = a.try_permute(&[0, 0]).expect_err("duplicate axis");
        assert!(format!("{err:?}").contains("duplicate"));
        // Happy path:
        let ok = a.try_permute(&[1, 0]).expect("valid permutation");
        assert_eq!(ok.shape().dims(), &[3, 2]);
    }

    #[test]
    fn squeeze_drops_size_one_dim_metadata_only() {
        // Build [2, 1, 3] → squeeze(1) → [2, 3]. Op::Squeeze, view-op,
        // shape pruned, dtype preserved, single input slot.
        let a = Tensor::from_f32(
            vec![1.0; 6],
            Shape::from_dims(&[2, 1, 3]),
            cpu_dev(),
        );
        let s = a.squeeze(1).expect("squeeze on size-1 dim");
        assert_eq!(s.shape().dims(), &[2, 3]);
        assert_eq!(s.dtype(), DType::F32);
        let node = s.graph().read().unwrap().node(s.id()).clone();
        assert!(matches!(node.op, Op::Squeeze { dim: 1 }));
        assert_eq!(node.inputs, vec![a.id()]);
        // Squeeze joins the view-op set so the Layout side-table is
        // populated automatically.
        assert!(node.op.is_view_op(),
            "Op::Squeeze must register as a view op");
    }

    #[test]
    fn squeeze_rejects_non_size_one_dim() {
        // Result-returning: bad dim surfaces as Err, not panic.
        let a = Tensor::from_f32(vec![1.0; 6], Shape::from_dims(&[2, 3]), cpu_dev());
        let err = a.squeeze(1).expect_err("squeezing a non-size-1 dim must error");
        assert!(format!("{err:?}").contains("expected 1"),
            "error message should mention the size-1 expectation, got: {err:?}");
    }

    #[test]
    fn squeeze_rejects_dim_above_rank() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let err = a.squeeze(5).expect_err("dim above rank must error");
        assert!(format!("{err:?}").contains("out of bounds"),
            "error message should mention bounds, got: {err:?}");
    }

    #[test]
    fn backward_through_squeeze_emits_unsqueeze() {
        // y = squeeze(x, 1). Backward: re-insert dim 1 via Unsqueeze.
        // Gradient shape must equal x.shape exactly.
        let a = Tensor::from_f32(
            vec![1.0; 6],
            Shape::from_dims(&[2, 1, 3]),
            cpu_dev(),
        );
        let y = a.squeeze(1).expect("squeeze on size-1 dim");
        let grads = y.backward();
        let g_a = grads.get(&a).expect("gradient for a");
        assert_eq!(g_a.shape().dims(), &[2, 1, 3],
            "Squeeze backward gradient must restore the original shape");
        // The chain is: ones-seed (post-squeeze, shape [2,3])
        //               → Unsqueeze(dim=1) → shape [2, 1, 3].
        let g_node = g_a.graph().read().unwrap().node(g_a.id()).clone();
        assert!(matches!(g_node.op, Op::Unsqueeze { dim: 1 }),
            "Squeeze backward must be Unsqueeze at the same dim");
    }

    #[test]
    fn floor_builder_produces_unary_node_same_dtype() {
        // Tensor::floor builds a unary node preserving shape + dtype.
        let a = Tensor::from_f32(vec![1.5, 2.5, 3.5], Shape::from_dims(&[3]), cpu_dev());
        let f = a.floor();
        assert_eq!(f.shape().dims(), &[3]);
        assert_eq!(f.dtype(), DType::F32);
        let node = f.graph().read().unwrap().node(f.id()).clone();
        assert!(matches!(node.op, Op::Floor));
        assert_eq!(node.inputs, vec![a.id()]);
    }

    #[test]
    fn backward_through_floor_drops_gradient_silently() {
        // Op::Floor has zero gradient almost everywhere; the inline
        // backward arm is a no-op. backward() must succeed and the
        // input gets no gradient (the input's upstream is never
        // populated through the Floor node).
        let a = Tensor::from_f32(vec![1.5, 2.5, 3.5], Shape::from_dims(&[3]), cpu_dev());
        let y = a.floor();
        let grads = y.backward();
        // a's gradient is dropped — no entry in the GradMap.
        assert!(grads.get(&a).is_none(),
            "Floor must not propagate gradient to its input");
    }

    #[test]
    fn where_cond_builder_produces_ternary_with_a_dtype() {
        // self is U8 cond; a/b are F32. Output dtype = F32 (= a's dtype),
        // shape = self.shape() (= a.shape() = b.shape()). Op::Where
        // node carries 3 inputs in order (cond, a, b).
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let b = a.const_f32_like(vec![10.0, 20.0, 30.0], Shape::from_dims(&[3]));
        let eq_a_b = a.eq(&b);  // U8 mask
        let picked = eq_a_b.where_cond(&a, &b);
        assert_eq!(picked.shape().dims(), &[3]);
        assert_eq!(picked.dtype(), DType::F32, "Where output dtype = a/b dtype");
        let node = picked.graph().read().unwrap().node(picked.id()).clone();
        assert!(matches!(node.op, Op::Where));
        assert_eq!(node.inputs.len(), 3);
        assert_eq!(node.inputs, vec![eq_a_b.id(), a.id(), b.id()],
            "Where input order: (cond, a, b)");
    }

    #[test]
    fn backward_reuses_exp_forward_output() {
        // Exp's backward rule uses the forward output directly. The
        // gradient for x should be a Mul whose inputs include the
        // forward Exp node (not a new Exp node).
        let a = Tensor::from_f32(vec![0.0, 1.0], Shape::from_dims(&[2]), cpu_dev());
        let e = a.exp();
        let exp_forward_id = e.id();
        let grads = e.backward();
        let g_a = grads.get(&a).unwrap();
        let node = g_a.graph().read().unwrap().node(g_a.id()).clone();
        assert!(matches!(node.op, Op::Mul));
        // One of the Mul's inputs should be the original forward Exp node.
        assert!(
            node.inputs.contains(&exp_forward_id),
            "Exp backward should reference the forward output ({exp_forward_id:?}), \
             got inputs {:?}",
            node.inputs,
        );
    }

    // ---- Op::WriteSlice (Phase E.3.2) ---------------------------------------

    #[test]
    fn write_slice_destructive_input_is_zero() {
        let op = Op::WriteSlice { ranges: vec![(0, 1), (0, 32), (0, 128)], dyn_offset: None };
        assert_eq!(op.destructive_input(), Some(0));
    }

    #[test]
    fn write_slice_short_name() {
        let op = Op::WriteSlice { ranges: vec![(0, 1)], dyn_offset: None };
        assert_eq!(op.short_name(), "WriteSlice");
    }

    #[test]
    fn write_slice_emits_op_writeslice_node() {
        // dest shape [4, 3]; source shape [1, 3]; write at row 2.
        let dest = Tensor::from_f32(
            vec![0.0_f32; 12], Shape::from_dims(&[4, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[1, 3]));
        let out = dest.write_slice(&src, vec![(2, 3), (0, 3)])
            .expect("write_slice should accept matching shapes");
        let g = out.graph().read().unwrap();
        match &g.node(out.id()).op {
            Op::WriteSlice { ranges, dyn_offset } => {
                assert_eq!(ranges, &vec![(2, 3), (0, 3)]);
                assert_eq!(dyn_offset, &None);
            }
            other => panic!("expected Op::WriteSlice, got {other:?}"),
        }
        assert_eq!(g.node(out.id()).inputs, vec![dest.id(), src.id()]);
        // Output shape == destination shape; bytes are post-write same buffer.
        assert_eq!(g.node(out.id()).shape.dims(), &[4, 3]);
    }

    #[test]
    fn write_slice_dyn_records_dynamic_offset() {
        use fuel_ir::{DynScalar, SymId};
        // dest capacity [8, 3]; source [1, 3]; dynamic start on axis 0.
        let dest = Tensor::from_f32(
            vec![0.0_f32; 24], Shape::from_dims(&[8, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[1, 3]));
        let sym = SymId(0);
        // ranges[0].0 is ignored (start is dynamic); width = 1 - 0 = 1
        // matches source dim 0. Axis 1 is static, full width.
        let out = dest
            .write_slice_dyn(&src, vec![(0, 1), (0, 3)], 0, DynScalar::Sym(sym))
            .expect("write_slice_dyn should accept matching widths");
        let g = out.graph().read().unwrap();
        match &g.node(out.id()).op {
            Op::WriteSlice { ranges, dyn_offset } => {
                assert_eq!(ranges, &vec![(0, 1), (0, 3)]);
                assert_eq!(dyn_offset, &Some((0, DynScalar::Sym(sym))));
            }
            other => panic!("expected Op::WriteSlice, got {other:?}"),
        }
        assert_eq!(g.node(out.id()).inputs, vec![dest.id(), src.id()]);
        assert_eq!(g.node(out.id()).shape.dims(), &[8, 3]);
    }

    #[test]
    fn write_slice_dyn_rejects_slab_wider_than_capacity() {
        use fuel_ir::{DynScalar, SymId};
        // dest capacity on axis 0 is 2, but the dynamic-axis slab is
        // width 4 — can never fit regardless of the runtime offset.
        let dest = Tensor::from_f32(
            vec![0.0_f32; 6], Shape::from_dims(&[2, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![0.0_f32; 12], Shape::from_dims(&[4, 3]));
        let err = dest.write_slice_dyn(&src, vec![(0, 4), (0, 3)], 0, DynScalar::Sym(SymId(0)));
        assert!(err.is_err(), "dynamic-axis slab wider than capacity must error at build");
    }

    #[test]
    fn write_slice_dyn_rejects_axis_out_of_bounds() {
        use fuel_ir::{DynScalar, SymId};
        let dest = Tensor::from_f32(
            vec![0.0_f32; 6], Shape::from_dims(&[2, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[1, 3]));
        let err = dest.write_slice_dyn(&src, vec![(0, 1), (0, 3)], 5, DynScalar::Sym(SymId(0)));
        assert!(err.is_err(), "dyn_axis past rank must error");
    }

    #[test]
    fn flash_attn_dyn_records_runtime_k_len() {
        use fuel_ir::{DynScalar, SymId};
        // q [1, 2, 3, 4]; K/V capacity [1, 1, 8, 4] (GQA Hq=2, Hkv=1).
        let q = Tensor::from_f32(
            vec![0.0_f32; 1 * 2 * 3 * 4], Shape::from_dims(&[1, 2, 3, 4]), cpu_dev(),
        );
        let k = q.const_f32_like(vec![0.0_f32; 1 * 1 * 8 * 4], Shape::from_dims(&[1, 1, 8, 4]));
        let v = q.const_f32_like(vec![0.0_f32; 1 * 1 * 8 * 4], Shape::from_dims(&[1, 1, 8, 4]));
        let sym = SymId(0);
        let out = q.flash_attn_dyn(
            &k, &v, None, /*scale*/ 0.5, /*causal*/ true,
            None, None, None, DynScalar::Sym(sym),
        );
        // Output adopts q's shape, not the capacity.
        assert_eq!(out.shape().dims(), &[1, 2, 3, 4]);
        let g = out.graph().read().unwrap();
        match &g.node(out.id()).op {
            Op::Fused(fid, crate::registry::FusedOpParams::FlashAttn { causal, k_len, .. })
                if *fid == crate::registry::FusedOps::FLASH_ATTN =>
            {
                assert!(*causal);
                assert_eq!(k_len, &Some(DynScalar::Sym(sym)));
            }
            other => panic!("expected Op::Fused(FLASH_ATTN, FlashAttn), got {other:?}"),
        }
        assert_eq!(g.node(out.id()).inputs, vec![q.id(), k.id(), v.id()]);
    }

    #[test]
    #[should_panic(expected = "k_len")]
    fn flash_attn_dyn_concrete_k_len_exceeding_capacity_panics() {
        use fuel_ir::DynScalar;
        let q = Tensor::from_f32(
            vec![0.0_f32; 1 * 1 * 2 * 4], Shape::from_dims(&[1, 1, 2, 4]), cpu_dev(),
        );
        let k = q.const_f32_like(vec![0.0_f32; 1 * 1 * 4 * 4], Shape::from_dims(&[1, 1, 4, 4]));
        let v = q.const_f32_like(vec![0.0_f32; 1 * 1 * 4 * 4], Shape::from_dims(&[1, 1, 4, 4]));
        // Concrete k_len=5 > capacity 4 → build-time panic.
        let _ = q.flash_attn_dyn(&k, &v, None, 1.0, true, None, None, None, DynScalar::Concrete(5));
    }

    #[test]
    fn write_slice_rejects_rank_mismatch() {
        let dest = Tensor::from_f32(
            vec![0.0_f32; 12], Shape::from_dims(&[4, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[3]));
        // ranges has rank 2 (matches dest) but source has rank 1.
        let err = dest.write_slice(&src, vec![(0, 1), (0, 3)]);
        assert!(err.is_err(), "rank mismatch must error");
    }

    #[test]
    fn write_slice_rejects_slab_width_mismatch() {
        let dest = Tensor::from_f32(
            vec![0.0_f32; 12], Shape::from_dims(&[4, 3]), cpu_dev(),
        );
        // Source has 2 elements along axis 0, but slab is width 1.
        let src = dest.const_f32_like(
            vec![1.0_f32; 6], Shape::from_dims(&[2, 3]),
        );
        let err = dest.write_slice(&src, vec![(2, 3), (0, 3)]);
        assert!(err.is_err(), "slab-width mismatch must error");
    }

    #[test]
    fn write_slice_rejects_range_past_dest_extent() {
        let dest = Tensor::from_f32(
            vec![0.0_f32; 12], Shape::from_dims(&[4, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[1, 3]));
        // axis 0: dest extent is 4; range [4, 5) is out of bounds.
        let err = dest.write_slice(&src, vec![(4, 5), (0, 3)]);
        assert!(err.is_err(), "range past dest extent must error");
    }

    #[test]
    fn write_slice_derives_ordering_after_other_readers() {
        // Graph:
        //   dest = ...; src = ...
        //   ro = dest.relu()                          (non-destructive reader of dest)
        //   w  = dest.write_slice(&src, [(0,1),(0,3)])  (destructive on dest)
        // Expected ordering: w must run after ro.
        let dest = Tensor::from_f32(
            vec![0.0_f32; 12], Shape::from_dims(&[4, 3]), cpu_dev(),
        );
        let src = dest.const_f32_like(vec![1.0, 2.0, 3.0], Shape::from_dims(&[1, 3]));
        let ro = dest.relu();
        let w = dest.write_slice(&src, vec![(0, 1), (0, 3)]).unwrap();
        let ord = crate::opt::derive_ordering(
            &dest.graph().read().unwrap(),
            &[ro.id(), w.id()],
        );
        let deps = ord.deps_of(w.id());
        assert_eq!(deps.len(), 1, "write_slice should have one ordering dep (the relu)");
        assert_eq!(deps[0], ro.id(), "write_slice must run after the non-destructive reader");
    }

    // ---- In-place ops Phase 1 -----------------------------------------------
    //
    // Smoke tests for the 5 new in-place unary variants
    // (`Op::ReluInplace`, `Op::SiluInplace`, `Op::GeluInplace`,
    // `Op::TanhInplace`, `Op::SigmoidInplace`) and `FusedOps::INPLACE_AFFINE`.
    // Phase 1 only ships the structural plumbing (Op IR + destructive_input +
    // short_name + scheduler integration); dispatch + autograd land in Phases
    // 3 + 4. See `docs/session-prompts/in-place-ops-infrastructure.md`.

    #[test]
    fn inplace_unary_destructive_input_is_zero() {
        for op in [
            Op::ReluInplace,
            Op::SiluInplace,
            Op::GeluInplace,
            Op::TanhInplace,
            Op::SigmoidInplace,
            Op::NegInplace,
            Op::AbsInplace,
            Op::SqrInplace,
            Op::SqrtInplace,
            Op::RsqrtInplace,
            Op::RecipInplace,
            Op::ExpInplace,
            Op::LogInplace,
            Op::SinInplace,
            Op::CosInplace,
            Op::SignInplace,
            Op::FloorInplace,
            Op::CeilInplace,
            Op::RoundInplace,
            Op::ErfInplace,
            Op::GeluErfInplace,
        ] {
            assert_eq!(
                op.destructive_input(), Some(0),
                "in-place unary {op:?} must declare input 0 destructive",
            );
        }
    }

    #[test]
    fn inplace_unary_short_names_round_trip() {
        assert_eq!(Op::ReluInplace.short_name(), "ReluInplace");
        assert_eq!(Op::SiluInplace.short_name(), "SiluInplace");
        assert_eq!(Op::GeluInplace.short_name(), "GeluInplace");
        assert_eq!(Op::TanhInplace.short_name(), "TanhInplace");
        assert_eq!(Op::SigmoidInplace.short_name(), "SigmoidInplace");
        assert_eq!(Op::NegInplace.short_name(), "NegInplace");
        assert_eq!(Op::AbsInplace.short_name(), "AbsInplace");
        assert_eq!(Op::SqrInplace.short_name(), "SqrInplace");
        assert_eq!(Op::SqrtInplace.short_name(), "SqrtInplace");
        assert_eq!(Op::RsqrtInplace.short_name(), "RsqrtInplace");
        assert_eq!(Op::RecipInplace.short_name(), "RecipInplace");
        assert_eq!(Op::ExpInplace.short_name(), "ExpInplace");
        assert_eq!(Op::LogInplace.short_name(), "LogInplace");
        assert_eq!(Op::SinInplace.short_name(), "SinInplace");
        assert_eq!(Op::CosInplace.short_name(), "CosInplace");
        assert_eq!(Op::SignInplace.short_name(), "SignInplace");
        assert_eq!(Op::FloorInplace.short_name(), "FloorInplace");
        assert_eq!(Op::CeilInplace.short_name(), "CeilInplace");
        assert_eq!(Op::RoundInplace.short_name(), "RoundInplace");
        assert_eq!(Op::ErfInplace.short_name(), "ErfInplace");
        assert_eq!(Op::GeluErfInplace.short_name(), "GeluErfInplace");
        assert_eq!(Op::ClampInplace { min: 0.0, max: 1.0 }.short_name(), "ClampInplace");
        assert_eq!(Op::PowIInplace(3).short_name(), "PowIInplace");
    }

    #[test]
    fn inplace_affine_fused_op_destructive_input_is_zero() {
        let op = Op::Fused(
            crate::registry::FusedOps::INPLACE_AFFINE,
            crate::registry::FusedOpParams::InplaceAffine { mul: 2.0, add: 1.0 },
        );
        assert_eq!(op.destructive_input(), Some(0));
    }

    #[test]
    fn inplace_affine_params_key_dedupe_on_same_mul_add() {
        // Two `InplaceAffine` params with identical (mul, add) hash to the
        // same key; differing values produce distinct keys. Lets CSE / the
        // existing op_key infrastructure collapse identical in-place affines
        // (when callers wire them through the equivalent graph machinery).
        let a = crate::registry::FusedOpParams::InplaceAffine { mul: 2.0, add: 1.0 };
        let b = crate::registry::FusedOpParams::InplaceAffine { mul: 2.0, add: 1.0 };
        let c = crate::registry::FusedOpParams::InplaceAffine { mul: 2.0, add: 0.5 };
        assert_eq!(a.key(), b.key(), "identical mul/add must dedupe");
        assert_ne!(a.key(), c.key(), "different add must produce distinct key");
    }

    // ---- Phase 4 — autograd through in-place ops ----
    //
    // The view-aware `derive_ordering` (Phase 4a) ensures backward
    // grad nodes that read forward inputs run BEFORE any in-place
    // mutation of those inputs. The backward arms in
    // `Tensor::backward` (Phase 4b) emit the same gradient graph as
    // the non-inplace cousins. These tests prove that calling
    // `.backward()` through an in-place node:
    //   (1) does NOT panic (was previously guarded by the Phase 1
    //       defensive panic);
    //   (2) emits the same gradient-node structure as the non-inplace
    //       cousin (matching node counts + op types as a structural
    //       proxy for numerical equivalence — the actual numerics are
    //       exercised by the live oracle tests in fuel-storage).

    #[test]
    fn backward_through_relu_inplace_does_not_panic() {
        // y = x.relu_inplace(); loss = y.sum_all(); loss.backward()
        // Phase 1's defensive panic is removed; backward emits
        // Op::Step(x) + Op::Mul(upstream, step) — same as Op::Relu.
        let x = Tensor::from_f32(vec![1.0, -2.0, 3.0], Shape::from_dims(&[3]), cpu_dev());
        let y = x.relu_inplace();
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = grads.get(&x).expect("gradient for x should be present");
        // Same structural form as Op::Relu's backward: gradient node
        // is an Op::Mul (upstream × step(x)).
        let g_node = g_x.graph().read().unwrap().node(g_x.id()).clone();
        assert!(matches!(g_node.op, Op::Mul), "relu backward grad is Mul; got {:?}", g_node.op);
    }

    #[test]
    fn backward_through_sigmoid_inplace_does_not_panic() {
        let x = Tensor::from_f32(vec![0.5, -0.5], Shape::from_dims(&[2]), cpu_dev());
        let y = x.sigmoid_inplace();
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = grads.get(&x);
        assert!(g_x.is_some());
    }

    #[test]
    fn backward_through_tanh_inplace_does_not_panic() {
        let x = Tensor::from_f32(vec![0.1, 0.2, 0.3], Shape::from_dims(&[3]), cpu_dev());
        let y = x.tanh_inplace();
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = grads.get(&x);
        assert!(g_x.is_some());
    }

    #[test]
    fn backward_through_silu_inplace_does_not_panic() {
        let x = Tensor::from_f32(vec![0.1, 0.2], Shape::from_dims(&[2]), cpu_dev());
        let y = x.silu_inplace();
        let loss = y.sum_all();
        let grads = loss.backward();
        let g_x = grads.get(&x);
        assert!(g_x.is_some());
    }

    #[test]
    fn backward_through_gelu_inplace_panics_matching_op_gelu() {
        // GeluInplace mirrors Op::Gelu's "currently inference-only"
        // panic — both rely on the same not-yet-implemented gradient.
        // SiluInplace is the recommended differentiable alternative.
        let x = Tensor::from_f32(vec![0.1], Shape::from_dims(&[1]), cpu_dev());
        let y = x.gelu_inplace();
        let loss = y.sum_all();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loss.backward()
        }));
        assert!(result.is_err(), "GeluInplace backward should panic (mirrors Op::Gelu)");
    }

    #[test]
    fn backward_through_relu_inplace_emits_same_graph_shape_as_functional() {
        // Structural parity: relu_inplace backward emits the same node
        // count + same op variants as relu backward, since the
        // gradient formula is identical.
        let x_inp = Tensor::from_f32(vec![1.0, -1.0], Shape::from_dims(&[2]), cpu_dev());
        let y_inp = x_inp.relu_inplace();
        let loss_inp = y_inp.sum_all();
        let nodes_before_inp = loss_inp.graph().read().unwrap().len();
        let _ = loss_inp.backward();
        let added_inp = loss_inp.graph().read().unwrap().len() - nodes_before_inp;

        let x_fn = Tensor::from_f32(vec![1.0, -1.0], Shape::from_dims(&[2]), cpu_dev());
        let y_fn = x_fn.relu();
        let loss_fn = y_fn.sum_all();
        let nodes_before_fn = loss_fn.graph().read().unwrap().len();
        let _ = loss_fn.backward();
        let added_fn = loss_fn.graph().read().unwrap().len() - nodes_before_fn;

        assert_eq!(
            added_inp, added_fn,
            "ReluInplace backward should add the same number of nodes as Relu backward ({added_fn}), got {added_inp}",
        );
    }

    #[test]
    fn relu_inplace_builder_emits_op_reluinplace() {
        let x = Tensor::from_f32(
            vec![1.0_f32, -1.0, 2.0, -2.0], Shape::from_dims(&[4]), cpu_dev(),
        );
        let y = x.relu_inplace();
        let g = y.graph().read().unwrap();
        let node = g.node(y.id());
        assert!(matches!(node.op, Op::ReluInplace));
        assert_eq!(node.inputs, vec![x.id()]);
        assert_eq!(node.shape.dims(), &[4]);
        assert_eq!(node.dtype, DType::F32);
    }

    #[test]
    fn affine_inplace_builder_emits_fused_inplace_affine() {
        let x = Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev(),
        );
        let y = x.affine_inplace(2.0, 0.5);
        let g = y.graph().read().unwrap();
        let node = g.node(y.id());
        match &node.op {
            Op::Fused(id, crate::registry::FusedOpParams::InplaceAffine { mul, add }) => {
                assert_eq!(*id, crate::registry::FusedOps::INPLACE_AFFINE);
                assert_eq!(*mul, 2.0);
                assert_eq!(*add, 0.5);
            }
            other => panic!("expected Op::Fused(INPLACE_AFFINE, InplaceAffine), got {other:?}"),
        }
        assert_eq!(node.inputs, vec![x.id()]);
        assert_eq!(node.shape.dims(), &[4]);
    }

    #[test]
    fn all_inplace_unary_builders_round_trip_through_op_variants() {
        // One smoke per variant — each builder emits exactly the
        // matching `Op::*Inplace` variant, single input, shape +
        // dtype unchanged.
        let x = Tensor::from_f32(
            vec![0.5_f32; 8], Shape::from_dims(&[8]), cpu_dev(),
        );
        fn check(y: Tensor, x_id: NodeId, expect: fn(&Op) -> bool) {
            let g = y.graph().read().unwrap();
            let node = g.node(y.id());
            assert!(expect(&node.op), "wrong Op variant: {:?}", node.op);
            assert_eq!(node.inputs, vec![x_id]);
            assert_eq!(node.shape.dims(), &[8]);
            assert_eq!(node.dtype, DType::F32);
        }
        check(x.relu_inplace(),    x.id(), |o| matches!(o, Op::ReluInplace));
        check(x.silu_inplace(),    x.id(), |o| matches!(o, Op::SiluInplace));
        check(x.gelu_inplace(),    x.id(), |o| matches!(o, Op::GeluInplace));
        check(x.tanh_inplace(),    x.id(), |o| matches!(o, Op::TanhInplace));
        check(x.sigmoid_inplace(), x.id(), |o| matches!(o, Op::SigmoidInplace));
        check(x.neg_inplace(),     x.id(), |o| matches!(o, Op::NegInplace));
        check(x.abs_inplace(),     x.id(), |o| matches!(o, Op::AbsInplace));
        check(x.sqr_inplace(),     x.id(), |o| matches!(o, Op::SqrInplace));
        check(x.sqrt_inplace(),    x.id(), |o| matches!(o, Op::SqrtInplace));
        check(x.rsqrt_inplace(),   x.id(), |o| matches!(o, Op::RsqrtInplace));
        check(x.recip_inplace(),   x.id(), |o| matches!(o, Op::RecipInplace));
        check(x.exp_inplace(),     x.id(), |o| matches!(o, Op::ExpInplace));
        check(x.log_inplace(),     x.id(), |o| matches!(o, Op::LogInplace));
        check(x.sin_inplace(),     x.id(), |o| matches!(o, Op::SinInplace));
        check(x.cos_inplace(),     x.id(), |o| matches!(o, Op::CosInplace));
        check(x.sign_inplace(),    x.id(), |o| matches!(o, Op::SignInplace));
        check(x.floor_inplace(),   x.id(), |o| matches!(o, Op::FloorInplace));
        check(x.ceil_inplace(),    x.id(), |o| matches!(o, Op::CeilInplace));
        check(x.round_inplace(),   x.id(), |o| matches!(o, Op::RoundInplace));
        check(x.erf_inplace(),     x.id(), |o| matches!(o, Op::ErfInplace));
        check(x.gelu_erf_inplace(),x.id(), |o| matches!(o, Op::GeluErfInplace));
        check(x.clamp_inplace(-1.0, 1.0), x.id(), |o| matches!(o, Op::ClampInplace { .. }));
        check(x.powi_inplace(3),         x.id(), |o| matches!(o, Op::PowIInplace(3)));
    }

    /// Backward smoke for the expanded in-place op family — each variant
    /// (except zero-grad ones) must produce a gradient node for x without
    /// panicking. Zero-grad variants (Sign/Floor/Ceil/Round) drop the
    /// gradient entirely (no entry in the GradMap), mirroring their
    /// non-inplace cousins.
    #[test]
    fn backward_through_expanded_inplace_unary_emits_grad() {
        fn check_emits_grad(make: impl FnOnce(&Tensor) -> Tensor) {
            let x = Tensor::from_f32(vec![0.5_f32, 1.5, 2.5], Shape::from_dims(&[3]), cpu_dev());
            let y = make(&x);
            let loss = y.sum_all();
            let grads = loss.backward();
            assert!(grads.get(&x).is_some(), "backward must emit gradient for x");
        }
        check_emits_grad(|x| x.neg_inplace());
        check_emits_grad(|x| x.abs_inplace());
        check_emits_grad(|x| x.sqr_inplace());
        check_emits_grad(|x| x.sqrt_inplace());
        check_emits_grad(|x| x.rsqrt_inplace());
        check_emits_grad(|x| x.recip_inplace());
        check_emits_grad(|x| x.exp_inplace());
        check_emits_grad(|x| x.log_inplace());
        check_emits_grad(|x| x.sin_inplace());
        check_emits_grad(|x| x.cos_inplace());
        check_emits_grad(|x| x.erf_inplace());
        check_emits_grad(|x| x.gelu_erf_inplace());
        check_emits_grad(|x| x.clamp_inplace(-1.0, 2.0));
        check_emits_grad(|x| x.powi_inplace(3));
    }

    #[test]
    fn backward_through_zero_grad_inplace_drops_gradient() {
        fn check_drops(make: impl FnOnce(&Tensor) -> Tensor) {
            let x = Tensor::from_f32(vec![1.5_f32, 2.5, -3.5], Shape::from_dims(&[3]), cpu_dev());
            let y = make(&x);
            let loss = y.sum_all();
            let grads = loss.backward();
            assert!(grads.get(&x).is_none(), "zero-grad in-place op must not emit gradient");
        }
        check_drops(|x| x.sign_inplace());
        check_drops(|x| x.floor_inplace());
        check_drops(|x| x.ceil_inplace());
        check_drops(|x| x.round_inplace());
    }

    #[test]
    fn inplace_unary_derives_ordering_after_non_destructive_readers() {
        // Graph:
        //   x   = const f32 [4]
        //   y_a = x.relu()                  (non-destructive reader of x)
        //   y_b = manually-pushed Op::ReluInplace on x  (destructive)
        // Expected: y_b must run after y_a.
        let x = Tensor::from_f32(
            vec![1.0_f32, -1.0, 2.0, -2.0], Shape::from_dims(&[4]), cpu_dev(),
        );
        let y_a = x.relu();
        let y_b_id = x.graph().write().unwrap().push(Node {
            op: Op::ReluInplace,
            inputs: vec![x.id()],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        let ord = crate::opt::derive_ordering(
            &x.graph().read().unwrap(),
            &[y_a.id(), y_b_id],
        );
        let deps = ord.deps_of(y_b_id);
        assert_eq!(deps.len(), 1, "in-place unary should have one ordering dep");
        assert_eq!(deps[0], y_a.id(), "in-place unary must run after the non-destructive reader");
    }

    // ------------------------------------------------------------------
    // Multi-output nodes (Option C, Session 1)
    // ------------------------------------------------------------------

    use fuel_ir::storage::OutputView;
    use fuel_backend_contract::storage::Storage as CoreStorage;

    /// Build a real CPU `Storage` of `n` F32 zeros so `Storage`-level
    /// tests can attach + read bundle metadata against a live backend
    /// allocator. The dtype is fixed at F32 because every existing
    /// `OutputView` test below uses F32 for slot 0; bundle dtype is
    /// validated at `with_bundle` time against this primary dtype.
    fn cpu_f32_storage(n: usize) -> CoreStorage {
        let dev = cpu_dev();
        let buf = fuel_ir::HostBuffer::F32(vec![0.0_f32; n]);
        let inner = dev
            .storage_from_host_buffer_owned_dyn(buf)
            .expect("test fixture: F32 storage_from_host_buffer_owned_dyn failed");
        CoreStorage::from_dyn(inner)
    }

    /// Build a 2-slot bundle: slot 0 `[2, 3]` F32 `y`-shape, slot 1
    /// `[2, 4]` F32 `last_state`-shape. Returns the views and the
    /// total element count (used to size the backing CPU buffer so
    /// slot 1's byte range stays inside the bundle's bytes).
    fn two_slot_views() -> (Vec<OutputView>, usize) {
        let s0 = Shape::from_dims(&[2, 3]);
        let s1 = Shape::from_dims(&[2, 4]);
        let v0 = OutputView {
            byte_offset:  0,
            len_elements: s0.elem_count(),
            dtype:        DType::F32,
            shape:        s0.clone(),
            layout:       Layout::contiguous(s0),
            name:         Some("y"),
        };
        let v1 = OutputView {
            byte_offset:  v0.len_bytes(),
            len_elements: s1.elem_count(),
            dtype:        DType::F32,
            shape:        s1.clone(),
            layout:       Layout::contiguous(s1),
            name:         Some("last_state"),
        };
        let total = v0.len_elements + v1.len_elements;
        (vec![v0, v1], total)
    }

    /// `Storage::is_bundled` is false by default; `slot_count` is 1;
    /// `bundle` / `bundle_arc` / `slot_view` / `slot_dtype` all return
    /// `None` for the non-multi-output case.
    #[test]
    fn storage_single_output_defaults() {
        let s = cpu_f32_storage(8);
        assert!(!s.is_bundled());
        assert_eq!(s.slot_count(), 1);
        assert!(s.bundle().is_none());
        assert!(s.bundle_arc().is_none());
        assert!(s.slot_view(0).is_none());
        assert_eq!(s.slot_dtype(0), None);
        assert_eq!(s.primary_dtype(), DType::F32);
        assert_eq!(s.dtype(), DType::F32);
    }

    /// `Storage::with_bundle` attaches the side-table and per-slot
    /// lookups report the declared shape/dtype.
    #[test]
    fn storage_with_bundle_attaches_views() {
        let (views, total) = two_slot_views();
        let s = cpu_f32_storage(total)
            .with_bundle(views.clone().into())
            .expect("with_bundle should accept matching slot-0 dtype");
        assert!(s.is_bundled());
        assert_eq!(s.slot_count(), 2);
        let slot0 = s.slot_view(0).unwrap();
        assert_eq!(slot0.shape, Shape::from_dims(&[2, 3]));
        assert_eq!(slot0.dtype, DType::F32);
        assert_eq!(slot0.byte_offset, 0);
        assert_eq!(slot0.name, Some("y"));
        let slot1 = s.slot_view(1).unwrap();
        assert_eq!(slot1.shape, Shape::from_dims(&[2, 4]));
        assert_eq!(slot1.byte_offset, 24); // 6 F32s = 24 bytes
        assert_eq!(slot1.name, Some("last_state"));
        // Out-of-range slot reads `None`, doesn't panic.
        assert!(s.slot_view(2).is_none());
        assert_eq!(s.slot_dtype(2), None);
    }

    /// `Storage::with_bundle` rejects an empty bundle slice (a
    /// zero-slot bundle is a contract bug — use a single-output
    /// Storage if there's nothing to project).
    #[test]
    fn storage_with_bundle_rejects_empty() {
        let s = cpu_f32_storage(1);
        let empty: Arc<[OutputView]> = Arc::from(Vec::<OutputView>::new().into_boxed_slice());
        let err = s.with_bundle(empty).err()
            .expect("empty bundle slice must error");
        let msg = format!("{err}");
        assert!(msg.contains("non-empty"), "error message: {msg}");
    }

    /// `Storage::with_bundle` rejects slot 0 dtype that disagrees with
    /// the inner backend dtype — the bundle invariant "slot 0 dtype ==
    /// primary_dtype()" must hold.
    #[test]
    fn storage_with_bundle_rejects_dtype_mismatch() {
        let s = cpu_f32_storage(8);
        let bad = OutputView {
            byte_offset:  0,
            len_elements: 4,
            dtype:        DType::F64,
            shape:        Shape::from_dims(&[2, 2]),
            layout:       Layout::contiguous(Shape::from_dims(&[2, 2])),
            name:         None,
        };
        let err = s.with_bundle(Arc::from(vec![bad].into_boxed_slice())).err()
            .expect("slot 0 dtype mismatch must error");
        let msg = format!("{err}");
        assert!(msg.contains("dtype"), "error message: {msg}");
    }

    /// `Graph::output_views` is `None` for ordinary single-output
    /// nodes; `multi_output_count` is 0.
    #[test]
    fn graph_output_views_defaults() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        assert!(g.output_views(id).is_none());
        assert!(!g.is_multi_output(id));
        assert_eq!(g.multi_output_count(), 0);
    }

    /// `Graph::set_output_views` round-trips: read returns what was
    /// written; `is_multi_output` flips.
    #[test]
    fn graph_set_output_views_roundtrip() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        let (views, _) = two_slot_views();
        g.set_output_views(id, views.clone().into())
            .expect("valid 2-slot specs");
        assert!(g.is_multi_output(id));
        assert_eq!(g.multi_output_count(), 1);
        let read = g.output_views(id).expect("entry present");
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].shape, Shape::from_dims(&[2, 4]));
    }

    /// `Graph::set_output_views` rejects slot 0 dtype/shape that
    /// disagree with the producer node's primary shape/dtype.
    #[test]
    fn graph_set_output_views_validates_primary() {
        let mut g = Graph::new();
        let id = g.push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        // dtype mismatch on slot 0
        let bad_dtype = OutputView {
            byte_offset:  0,
            len_elements: 6,
            dtype:        DType::F64,
            shape:        Shape::from_dims(&[2, 3]),
            layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
            name:         None,
        };
        let err = g.set_output_views(id, vec![bad_dtype].into()).err()
            .expect("primary dtype mismatch must error");
        let msg = format!("{err}");
        assert!(msg.contains("dtype"), "error message: {msg}");
        // shape mismatch on slot 0
        let bad_shape = OutputView {
            byte_offset:  0,
            len_elements: 4,
            dtype:        DType::F32,
            shape:        Shape::from_dims(&[2, 2]),
            layout:       Layout::contiguous(Shape::from_dims(&[2, 2])),
            name:         None,
        };
        let err = g.set_output_views(id, vec![bad_shape].into()).err()
            .expect("primary shape mismatch must error");
        let msg = format!("{err}");
        assert!(msg.contains("shape"), "error message: {msg}");
    }

    /// `Tensor::view(slot)` against a producer that hasn't been
    /// declared multi-output returns Err — fails fast at build time.
    #[test]
    fn tensor_view_on_non_multi_output_errors() {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let id = graph.write().unwrap().push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        let producer = Tensor::from_existing(Arc::clone(&graph), id);
        let err = producer.view(0).err()
            .expect("view on non-multi-output producer must error");
        let msg = format!("{err}");
        assert!(msg.contains("multi-output"), "error message: {msg}");
    }

    /// `Tensor::view(slot)` with slot OOB returns Err.
    #[test]
    fn tensor_view_oob_slot_errors() {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let producer_id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  Shape::from_dims(&[2, 3]),
                dtype:  DType::F32,
            });
            let (views, _) = two_slot_views();
            g.set_output_views(id, views.into())
                .expect("valid 2-slot specs");
            id
        };
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let err = producer.view(2).err()
            .expect("slot 2 of a 2-slot producer must error");
        let msg = format!("{err}");
        assert!(msg.contains("out of range"), "error message: {msg}");
    }

    /// `Tensor::view(slot)` produces a tensor whose shape/dtype match
    /// the slot spec; producer's primary shape stays as slot 0's. The
    /// View node holds the producer in `inputs[0]`.
    #[test]
    fn tensor_view_shape_dtype_from_slot() {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let producer_id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  Shape::from_dims(&[2, 3]),
                dtype:  DType::F32,
            });
            let (views, _) = two_slot_views();
            g.set_output_views(id, views.into())
                .expect("valid 2-slot specs");
            id
        };
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v0 = producer.view(0).expect("slot 0 view");
        assert_eq!(v0.shape(), Shape::from_dims(&[2, 3]));
        assert_eq!(v0.dtype(), DType::F32);
        let v1 = producer.view(1).expect("slot 1 view");
        assert_eq!(v1.shape(), Shape::from_dims(&[2, 4]));
        assert_eq!(v1.dtype(), DType::F32);
        // View node carries the producer in inputs[0] and the slot
        // index in the Op variant — verify the Graph state directly.
        let g = graph.read().unwrap();
        let v1_node = g.node(v1.id());
        assert_eq!(v1_node.inputs, vec![producer_id]);
        assert!(matches!(v1_node.op, Op::View { slot: 1 }));
        // Slot 0 starts at byte_offset 0 + contiguous → layout
        // matches Layout::contiguous(slot0.shape) → no side-table
        // entry (the contiguous fallback covers it).
        assert!(!g.has_explicit_layout(v0.id()));
        // Slot 1 starts at byte_offset 24 (= 6 F32 elements past
        // slot 0). Session 4 bakes that into the layout's
        // start_offset so a downstream kernel reading the producer's
        // bytes as F32 elements lands on slot 1's first byte. Side-
        // table entry is mandatory.
        assert!(g.has_explicit_layout(v1.id()));
        let l1 = g.layout(v1.id());
        assert_eq!(l1.start_offset(), 6, "slot 1 byte_offset 24 / 4 = 6 F32 elements");
        assert_eq!(l1.shape(), &Shape::from_dims(&[2, 4]));
    }

    /// `Tensor::view(slot)` over a slot whose layout is non-contiguous
    /// (strided) populates the Graph's layout side-table with the
    /// slot's layout — so downstream consumers see the strided view.
    #[test]
    fn tensor_view_propagates_non_contiguous_slot_layout() {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let strided_shape = Shape::from_dims(&[2, 3]);
        let strided_layout = Layout::contiguous(strided_shape.clone())
            .transpose(0, 1)
            .expect("transpose [2,3] -> [3,2]");
        // Slot 0 = primary (contiguous, matches producer's Node shape).
        let s0 = OutputView {
            byte_offset:  0,
            len_elements: strided_shape.elem_count(),
            dtype:        DType::F32,
            shape:        strided_shape.clone(),
            layout:       Layout::contiguous(strided_shape.clone()),
            name:         Some("primary"),
        };
        // Slot 1 = strided projection. Same byte range as slot 0; the
        // shape after transpose is [3, 2]; layout's start offset stays
        // within the slot.
        let s1 = OutputView {
            byte_offset:  0,
            len_elements: strided_shape.elem_count(),
            dtype:        DType::F32,
            shape:        Shape::from_dims(&[3, 2]),
            layout:       strided_layout.clone(),
            name:         Some("strided_view"),
        };
        let producer_id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  strided_shape.clone(),
                dtype:  DType::F32,
            });
            g.set_output_views(id, vec![s0, s1].into())
                .expect("valid 2-slot strided spec");
            id
        };
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v1 = producer.view(1).expect("strided slot view");
        let g = graph.read().unwrap();
        assert!(
            g.has_explicit_layout(v1.id()),
            "strided slot must populate Graph::layouts side-table",
        );
        let read_layout = g.layout(v1.id());
        assert_eq!(read_layout.shape(), strided_layout.shape());
        assert_eq!(read_layout.stride(), strided_layout.stride());
    }

    /// `Tensor::view_owned(slot)` produces the same shape/dtype as
    /// `Tensor::view(slot)` but its output layout is ALWAYS contiguous
    /// — the forward memcpy produces a fresh standalone buffer, so
    /// the slot's (possibly strided) layout is not propagated.
    #[test]
    fn tensor_view_owned_layout_is_contiguous() {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let strided_shape = Shape::from_dims(&[2, 3]);
        let strided_layout = Layout::contiguous(strided_shape.clone())
            .transpose(0, 1)
            .expect("transpose [2,3] -> [3,2]");
        let s0 = OutputView {
            byte_offset:  0,
            len_elements: strided_shape.elem_count(),
            dtype:        DType::F32,
            shape:        strided_shape.clone(),
            layout:       Layout::contiguous(strided_shape.clone()),
            name:         None,
        };
        let s1 = OutputView {
            byte_offset:  0,
            len_elements: strided_shape.elem_count(),
            dtype:        DType::F32,
            shape:        Shape::from_dims(&[3, 2]),
            layout:       strided_layout,
            name:         None,
        };
        let producer_id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  strided_shape,
                dtype:  DType::F32,
            });
            g.set_output_views(id, vec![s0, s1].into())
                .expect("valid 2-slot specs");
            id
        };
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v1 = producer.view_owned(1).expect("ViewOwned slot 1");
        assert_eq!(v1.shape(), Shape::from_dims(&[3, 2]));
        assert_eq!(v1.dtype(), DType::F32);
        let g = graph.read().unwrap();
        // ViewOwned is a fresh allocation — no explicit layout entry,
        // contiguous layout falls through.
        assert!(!g.has_explicit_layout(v1.id()));
        let owned_node = g.node(v1.id());
        assert!(matches!(owned_node.op, Op::ViewOwned { slot: 1 }));
    }

    // ------------------------------------------------------------------
    // Session 2: bundled allocator + FusedOpEntry::output_views +
    // View-vs-ViewOwned planner pass. The tests use the bundled
    // storage allocator and a synthetic 2-output FusedOp entry — no
    // real fused-op author migrates in Session 2; the
    // selective-scan-ssd-chunk-multi-output-followup session lights
    // up the first real consumer.
    // ------------------------------------------------------------------

    use fuel_backend_contract::storage::allocate_bundled_storage;
    use fuel_ir::storage::OutputViewSpec;

    /// `allocate_bundled_storage` allocates one Storage on the device,
    /// attaches the bundle, and slot lookups report the spec-derived
    /// byte_offsets.
    #[test]
    fn allocate_bundled_storage_two_slot_roundtrip() {
        let dev = cpu_dev();
        let specs = vec![
            OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[2, 3])),
            OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[2, 4])),
        ];
        let s = allocate_bundled_storage(dev.as_ref(), &specs)
            .expect("two-slot F32 alloc succeeds");
        assert!(s.is_bundled());
        assert_eq!(s.slot_count(), 2);
        assert_eq!(s.primary_dtype(), DType::F32);
        let v0 = s.slot_view(0).unwrap();
        let v1 = s.slot_view(1).unwrap();
        assert_eq!(v0.byte_offset, 0);
        assert_eq!(v0.shape, Shape::from_dims(&[2, 3]));
        assert_eq!(v1.byte_offset, 24);
        assert_eq!(v1.shape, Shape::from_dims(&[2, 4]));
    }

    /// `allocate_bundled_storage` handles mixed F32 + F64 slots — slot
    /// 0 is primary (F32), slot 1's bytes are aligned to its F64
    /// boundary, and the underlying allocation has enough flat F32
    /// elements to cover the total bundle bytes.
    #[test]
    fn allocate_bundled_storage_mixed_dtype() {
        let dev = cpu_dev();
        let specs = vec![
            OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[6])),
            OutputViewSpec::contiguous(DType::F64, Shape::from_dims(&[3])),
        ];
        let s = allocate_bundled_storage(dev.as_ref(), &specs)
            .expect("mixed-dtype alloc succeeds");
        assert_eq!(s.primary_dtype(), DType::F32);
        let v0 = s.slot_view(0).unwrap();
        let v1 = s.slot_view(1).unwrap();
        assert_eq!(v0.dtype, DType::F32);
        assert_eq!(v1.dtype, DType::F64);
        assert_eq!(v0.byte_offset, 0);
        // F32[6] = 24 bytes; 24 is already multiple of 8, so v1
        // starts at 24 without further padding.
        assert_eq!(v1.byte_offset, 24);
    }

    /// `allocate_bundled_storage` propagates compose_bundle errors —
    /// an empty spec list errors out before touching the device.
    #[test]
    fn allocate_bundled_storage_rejects_empty() {
        let dev = cpu_dev();
        let err = allocate_bundled_storage(dev.as_ref(), &[]).err()
            .expect("empty specs error");
        assert!(format!("{err}").contains("non-empty"));
    }

    /// The two SelectiveScan / SsdChunkScan migrations from item 3
    /// (2026-06-01) are the first `default_registry` entries to
    /// declare `output_views`. Any new multi-output op author should
    /// extend the expected set here.
    #[test]
    fn default_registry_multi_output_entries() {
        let reg = crate::registry::default_registry();
        let mut multi_out_names: Vec<&'static str> = reg
            .entries_iter()
            .filter(|e| e.output_views.is_some())
            .map(|e| e.name)
            .collect();
        multi_out_names.sort();
        assert_eq!(
            multi_out_names,
            vec!["SelectiveScan", "SsdChunkScan"],
            "the multi-output registry set should match the item-3 \
             migrations (SelectiveScan + SsdChunkScan); add new \
             multi-output ops to this list as they migrate",
        );
    }

    /// Authoring-contract invariant: for every multi-output entry in
    /// the process-wide registry, `shape_rule(inputs)` MUST equal
    /// `output_views(inputs)[0].shape` and `dtype_rule(inputs)` MUST
    /// equal `output_views(inputs)[0].dtype`. Slot 0 is the primary
    /// and the bundled-allocation contract assumes the producer's
    /// `Node.shape`/`Node.dtype` reflect it.
    #[test]
    fn multi_output_entries_slot_0_matches_primary_rules() {
        use crate::registry::FusedOpParams;
        let reg = crate::registry::default_registry();

        // Each multi-output entry needs a representative input batch
        // (shapes + dtypes + params) that satisfies its
        // shape_rule/dtype_rule preconditions. The check below runs
        // once per entry using these fixtures.
        struct Fixture {
            shapes: Vec<Shape>,
            dtypes: Vec<DType>,
            params: FusedOpParams,
        }
        fn selective_scan_fixture() -> Fixture {
            Fixture {
                shapes: vec![
                    Shape::from_dims(&[2, 4, 8]),  // u: [batch, seqlen, dim]
                    Shape::from_dims(&[2, 4, 8]),  // delta: same
                    Shape::from_dims(&[8, 16]),    // a: [dim, dstate]
                    Shape::from_dims(&[2, 4, 16]), // b: [batch, seqlen, dstate]
                    Shape::from_dims(&[2, 4, 16]), // c: same
                ],
                dtypes: vec![DType::F32; 5],
                params: FusedOpParams::SelectiveScan { delta_softplus: false },
            }
        }
        fn ssd_chunk_scan_fixture() -> Fixture {
            Fixture {
                shapes: vec![
                    Shape::from_dims(&[2, 4, 3, 8]),  // x: [batch, seqlen, heads, head_dim]
                    Shape::from_dims(&[2, 4, 3]),     // dt: [batch, seqlen, heads]
                    Shape::from_dims(&[3]),           // a: [heads]
                    Shape::from_dims(&[2, 4, 3, 16]), // b: [batch, seqlen, heads, state_dim]
                    Shape::from_dims(&[2, 4, 3, 16]), // c: same
                ],
                dtypes: vec![DType::F32; 5],
                params: FusedOpParams::SsdChunkScan { chunk_size: 4 },
            }
        }

        for entry in reg.entries_iter().filter(|e| e.output_views.is_some()) {
            let fx = match entry.name {
                "SelectiveScan" => selective_scan_fixture(),
                "SsdChunkScan"  => ssd_chunk_scan_fixture(),
                other => panic!(
                    "multi_output_entries_slot_0_matches_primary_rules: \
                     no fixture for new multi-output entry {other:?}. \
                     Add one to this test as you register a new op.",
                ),
            };
            let primary_shape = (entry.shape_rule)(&fx.shapes, &fx.params);
            let primary_dtype = (entry.dtype_rule)(&fx.dtypes, &fx.params);
            let specs = entry.output_views.unwrap()(
                &fx.shapes, &fx.dtypes, &fx.params,
            );
            assert!(
                !specs.is_empty(),
                "{}: output_views returned empty Vec (multi-output ops \
                 must have ≥ 1 slot)",
                entry.name,
            );
            assert_eq!(
                specs[0].shape, primary_shape,
                "{}: output_views[0].shape {:?} must equal shape_rule \
                 {:?} (slot 0 is the primary; Node::shape reflects it)",
                entry.name, specs[0].shape, primary_shape,
            );
            assert_eq!(
                specs[0].dtype, primary_dtype,
                "{}: output_views[0].dtype {:?} must equal dtype_rule \
                 {:?} (slot 0 is the primary; Node::dtype reflects it)",
                entry.name, specs[0].dtype, primary_dtype,
            );
            // Layout-shape coherence: each slot's layout shape must
            // equal its declared shape (the same invariant
            // set_output_views enforces at graph-build time).
            for (i, spec) in specs.iter().enumerate() {
                assert_eq!(
                    spec.layout.shape(), &spec.shape,
                    "{}: slot {i} layout.shape() {:?} disagrees with \
                     declared spec.shape {:?}",
                    entry.name, spec.layout.shape(), spec.shape,
                );
            }
        }
    }

    /// Build a synthetic multi-output `FusedOpEntry` to exercise the
    /// authoring contract end-to-end without polluting the production
    /// registry. The output_views fn returns a 2-slot spec list whose
    /// slot 0 matches shape_rule / dtype_rule (the primary).
    fn synthetic_two_output_entry() -> crate::registry::FusedOpEntry {
        fn shape(_: &[Shape], _: &crate::registry::FusedOpParams) -> Shape {
            Shape::from_dims(&[2, 3])
        }
        fn dtype(_: &[DType], _: &crate::registry::FusedOpParams) -> DType {
            DType::F32
        }
        fn ovs(
            _: &[Shape],
            _: &[DType],
            _: &crate::registry::FusedOpParams,
        ) -> Vec<OutputViewSpec> {
            vec![
                OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[2, 3])),
                OutputViewSpec::contiguous(DType::F32, Shape::from_dims(&[2, 4])),
            ]
        }
        crate::registry::FusedOpEntry {
            id:           crate::registry::FusedOps::SOFTMAX_LAST_DIM, // any id; not registered
            name:         "<synthetic-2-output>",
            family:       crate::registry::FusedOpFamily::Forward,
            pattern:      crate::registry::SubgraphPattern::Callable(|_g, _id| None),
            decompose:    |_g, id, _p| id,
            backward:     crate::registry::BackwardKind::NotDifferentiable,
            shape_rule:   shape,
            dtype_rule:   dtype,
            output_views: Some(ovs),
        }
    }

    /// The synthetic entry's `output_views` matches the
    /// `shape_rule`/`dtype_rule` contract on slot 0.
    #[test]
    fn synthetic_entry_output_views_matches_primary_contract() {
        let entry = synthetic_two_output_entry();
        let ovs_fn = entry.output_views.expect("synthetic entry is multi-output");
        let specs = ovs_fn(&[], &[], &crate::registry::FusedOpParams::SoftmaxLastDim);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].dtype, (entry.dtype_rule)(&[], &crate::registry::FusedOpParams::SoftmaxLastDim));
        assert_eq!(specs[0].shape, (entry.shape_rule)(&[], &crate::registry::FusedOpParams::SoftmaxLastDim));
    }

    /// Build a producer Const node and declare it multi-output via
    /// `Graph::set_output_views`. Returns the producer's NodeId and a
    /// SharedGraph handle.
    fn multi_output_producer(slots: Vec<OutputView>) -> (SharedGraph, NodeId) {
        let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
        let id = {
            let mut g = graph.write().unwrap();
            let id = g.push(Node {
                op:     Op::Const,
                inputs: vec![],
                shape:  slots[0].shape.clone(),
                dtype:  slots[0].dtype,
            });
            g.set_output_views(id, slots.into()).expect("valid synthetic specs");
            id
        };
        (graph, id)
    }

    /// Two slots, both consumed at the same topo depth → planner does
    /// nothing (symmetric lifetimes, bundle drops naturally).
    #[test]
    fn promote_views_symmetric_lifetimes_is_noop() {
        let slot_specs = vec![
            OutputView {
                byte_offset:  0,
                len_elements: 6,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 3]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
                name:         Some("y"),
            },
            OutputView {
                byte_offset:  24,
                len_elements: 8,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 4]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 4])),
                name:         Some("state"),
            },
        ];
        let (graph, producer_id) = multi_output_producer(slot_specs);
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        // Each slot has exactly one View, consumed at the same depth
        // (a no-op Relu downstream of each).
        let v0 = producer.view(0).expect("slot 0 view");
        let v1 = producer.view(1).expect("slot 1 view");
        let (v0_id, v0_shape, v0_dtype) = (v0.id(), v0.shape(), v0.dtype());
        let (v1_id, v1_shape, v1_dtype) = (v1.id(), v1.shape(), v1.dtype());
        let roots = {
            let mut g = graph.write().unwrap();
            let r0 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v0_id],
                shape:  v0_shape,
                dtype:  v0_dtype,
            });
            let r1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v1_id],
                shape:  v1_shape,
                dtype:  v1_dtype,
            });
            vec![r0, r1]
        };
        let promotions = {
            let mut g = graph.write().unwrap();
            crate::opt::promote_views_for_liveness(&mut g, &roots)
        };
        assert_eq!(promotions, 0, "symmetric lifetimes shouldn't promote anything");
    }

    /// Asymmetric lifetimes: slot 0 consumed early, slot 1 consumed
    /// late → planner promotes slot 1's View to ViewOwned.
    #[test]
    fn promote_views_asymmetric_promotes_long_lived_slot() {
        let slot_specs = vec![
            OutputView {
                byte_offset:  0,
                len_elements: 6,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 3]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
                name:         Some("y"),
            },
            OutputView {
                byte_offset:  24,
                len_elements: 6,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 3]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
                name:         Some("state"),
            },
        ];
        let (graph, producer_id) = multi_output_producer(slot_specs);
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v_y = producer.view(0).expect("slot 0 view");
        let v_state = producer.view(1).expect("slot 1 view");
        // Slot 0 ("y") consumed at depth 1 (single Relu).
        // Slot 1 ("state") consumed at a much deeper chain — Relu →
        // Relu → Relu. Topo positions push state's last-use later
        // than y's, triggering promotion of state's View.
        let (v_y_id, v_y_shape, v_y_dtype) = (v_y.id(), v_y.shape(), v_y.dtype());
        let (v_state_id, v_state_shape, v_state_dtype) = (v_state.id(), v_state.shape(), v_state.dtype());
        let roots = {
            let mut g = graph.write().unwrap();
            let y_relu = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v_y_id],
                shape:  v_y_shape,
                dtype:  v_y_dtype,
            });
            let s1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v_state_id],
                shape:  v_state_shape.clone(),
                dtype:  v_state_dtype,
            });
            let s2 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![s1],
                shape:  v_state_shape.clone(),
                dtype:  v_state_dtype,
            });
            let s3 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![s2],
                shape:  v_state_shape,
                dtype:  v_state_dtype,
            });
            vec![y_relu, s3]
        };
        let promotions = {
            let mut g = graph.write().unwrap();
            crate::opt::promote_views_for_liveness(&mut g, &roots)
        };
        assert_eq!(promotions, 1, "exactly the long-lived slot 1 View should promote");
        // Slot 1's consumer (the first Relu in the long chain) now
        // reads a fresh Op::ViewOwned node, not the old Op::View.
        let g = graph.read().unwrap();
        // Find the new ViewOwned node — it was pushed AFTER the
        // existing nodes, so it's the last node in the arena. Its op
        // must be Op::ViewOwned { slot: 1 } and its input must be the
        // producer.
        let last_id = NodeId(g.len() - 1);
        let last_node = g.node(last_id);
        assert!(matches!(last_node.op, Op::ViewOwned { slot: 1 }));
        assert_eq!(last_node.inputs, vec![producer_id]);
        // The slot 0 View is untouched — still Op::View.
        let v_y_node = g.node(v_y_id);
        assert!(matches!(v_y_node.op, Op::View { slot: 0 }));
        // The old slot 1 View stays in the arena (orphaned). No
        // consumer points at it anymore.
        let v_state_node = g.node(v_state_id);
        assert!(matches!(v_state_node.op, Op::View { slot: 1 }));
    }

    /// Running the pass twice on the same graph promotes nothing on
    /// the second call — `Op::ViewOwned` is a fixpoint (skipped by
    /// the planner because `is_owned` is true).
    #[test]
    fn promote_views_idempotent() {
        let slot_specs = vec![
            OutputView {
                byte_offset:  0,
                len_elements: 4,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 2]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 2])),
                name:         None,
            },
            OutputView {
                byte_offset:  16,
                len_elements: 4,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 2]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 2])),
                name:         None,
            },
        ];
        let (graph, producer_id) = multi_output_producer(slot_specs);
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v_y = producer.view(0).expect("slot 0 view");
        let v_state = producer.view(1).expect("slot 1 view");
        let (v_y_id, v_y_shape, v_y_dtype) = (v_y.id(), v_y.shape(), v_y.dtype());
        let (v_state_id, v_state_shape, v_state_dtype) = (v_state.id(), v_state.shape(), v_state.dtype());
        let roots = {
            let mut g = graph.write().unwrap();
            let y_r = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v_y_id],
                shape:  v_y_shape,
                dtype:  v_y_dtype,
            });
            let s1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v_state_id],
                shape:  v_state_shape.clone(),
                dtype:  v_state_dtype,
            });
            let s2 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![s1],
                shape:  v_state_shape,
                dtype:  v_state_dtype,
            });
            vec![y_r, s2]
        };
        let first = {
            let mut g = graph.write().unwrap();
            crate::opt::promote_views_for_liveness(&mut g, &roots)
        };
        let second = {
            let mut g = graph.write().unwrap();
            crate::opt::promote_views_for_liveness(&mut g, &roots)
        };
        assert_eq!(first, 1, "first call promotes the long-lived slot");
        assert_eq!(second, 0, "second call is idempotent — no further promotions");
    }

    /// Graph with no multi-output producers: pass returns 0 and
    /// touches nothing.
    #[test]
    fn promote_views_noop_on_graph_without_multi_output() {
        let mut g = Graph::new();
        let a = g.push(Node {
            op:     Op::Const,
            inputs: vec![],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        let r = g.push(Node {
            op:     Op::Relu,
            inputs: vec![a],
            shape:  Shape::from_dims(&[2, 3]),
            dtype:  DType::F32,
        });
        let n_before = g.len();
        let promotions = crate::opt::promote_views_for_liveness(&mut g, &[r]);
        assert_eq!(promotions, 0);
        assert_eq!(g.len(), n_before, "no node should have been added");
    }

    // ------------------------------------------------------------------
    // Session 3: scheduler / destructive-cleanup integration —
    // bundle-aware liveness. A destructive op on a multi-output
    // producer (or any of its Views) must run after every reader of
    // every other View, because all Views share the bundle's storage
    // Arc. Op::ViewOwned is excluded — it forks a fresh standalone
    // Storage at execute time.
    // ------------------------------------------------------------------

    /// Helper: build two contiguous slot specs for a producer of
    /// `Node::shape = [2, 3] F32`. Slot 0 is the primary (matches the
    /// producer's Node::shape/dtype, required by set_output_views);
    /// slot 1 carries a sibling shape that doesn't collide with the
    /// primary.
    fn two_slot_views_2x3_and_2x4() -> Vec<OutputView> {
        vec![
            OutputView {
                byte_offset:  0,
                len_elements: 6,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 3]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
                name:         Some("y"),
            },
            OutputView {
                byte_offset:  24,
                len_elements: 8,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 4]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 4])),
                name:         Some("state"),
            },
        ]
    }

    /// Op::Release targeting a multi-output producer pins after every
    /// consumer of any View of that producer. Without bundle-aware
    /// aliasing the Views' downstream Relu nodes would be missed and
    /// could race the Release.
    #[test]
    fn derive_ordering_release_of_bundled_producer_pins_after_view_consumers() {
        let (graph, producer_id) = multi_output_producer(two_slot_views_2x3_and_2x4());
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v0 = producer.view(0).expect("slot 0 view");
        let v1 = producer.view(1).expect("slot 1 view");
        let (v0_id, v0_shape, v0_dtype) = (v0.id(), v0.shape(), v0.dtype());
        let (v1_id, v1_shape, v1_dtype) = (v1.id(), v1.shape(), v1.dtype());
        let (r0, r1, release_id) = {
            let mut g = graph.write().unwrap();
            let r0 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v0_id],
                shape:  v0_shape,
                dtype:  v0_dtype,
            });
            let r1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v1_id],
                shape:  v1_shape,
                dtype:  v1_dtype,
            });
            let release = g.push(Node {
                op:     Op::Release,
                inputs: vec![producer_id],
                shape:  Shape::from_dims(&[0]),
                dtype:  DType::F32,
            });
            (r0, r1, release)
        };
        let ord = crate::opt::derive_ordering(
            &graph.read().unwrap(),
            &[r0, r1, release_id],
        );
        let deps = ord.deps_of(release_id);
        let dep_set: std::collections::HashSet<NodeId> = deps.iter().copied().collect();
        assert!(
            dep_set.contains(&r0),
            "Release must run after the slot-0 View's consumer (r0)",
        );
        assert!(
            dep_set.contains(&r1),
            "Release must run after the slot-1 View's consumer (r1) — \
             both Views share the bundle Arc",
        );
    }

    /// Destructive op targeting one of a bundle's Op::View nodes
    /// pins after every sibling View's consumers too — every View
    /// of the producer shares the bundle's Arc, so writing through
    /// one of them clobbers the bytes the others read.
    #[test]
    fn derive_ordering_destructive_on_view_pins_sibling_view_consumers() {
        let (graph, producer_id) = multi_output_producer(two_slot_views_2x3_and_2x4());
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v0 = producer.view(0).expect("slot 0 view");
        let v1 = producer.view(1).expect("slot 1 view");
        let (v0_id, _v0_shape, _v0_dtype) = (v0.id(), v0.shape(), v0.dtype());
        let (v1_id, v1_shape, v1_dtype) = (v1.id(), v1.shape(), v1.dtype());
        // Destructive op (Release) targets v0 directly — its
        // alias set must reach v1's consumer via the producer-sibling
        // hop in collect_alias_set.
        let (r1, release_id) = {
            let mut g = graph.write().unwrap();
            let r1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v1_id],
                shape:  v1_shape,
                dtype:  v1_dtype,
            });
            let release = g.push(Node {
                op:     Op::Release,
                inputs: vec![v0_id],
                shape:  Shape::from_dims(&[0]),
                dtype:  DType::F32,
            });
            (r1, release)
        };
        let ord = crate::opt::derive_ordering(
            &graph.read().unwrap(),
            &[r1, release_id],
        );
        let deps = ord.deps_of(release_id);
        let dep_set: std::collections::HashSet<NodeId> = deps.iter().copied().collect();
        assert!(
            dep_set.contains(&r1),
            "Release targeting slot-0 View must pin after sibling slot-1 \
             View's consumer (r1) — both share the bundle's storage Arc",
        );
    }

    /// Op::ViewOwned does NOT extend the alias set: its forward
    /// memcpy produces an independent Storage. derive_ordering must
    /// still pin Release after the ViewOwned itself (via the standard
    /// data-dependency: ViewOwned consumes the producer), but
    /// downstream consumers of the ViewOwned are NOT in the alias
    /// set's reader sweep — they read independent bytes and may run
    /// freely after the Release.
    #[test]
    fn derive_ordering_view_owned_is_not_alias_extending() {
        let (graph, producer_id) = multi_output_producer(two_slot_views_2x3_and_2x4());
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        // Slot 0 → View (shared Arc); slot 1 → ViewOwned (independent).
        let v_view = producer.view(0).expect("slot 0 view");
        let v_owned = producer.view_owned(1).expect("slot 1 viewowned");
        let (v_view_id, _, _) = (v_view.id(), v_view.shape(), v_view.dtype());
        let (v_owned_id, v_owned_shape, v_owned_dtype)
            = (v_owned.id(), v_owned.shape(), v_owned.dtype());
        let (consumer_of_owned, release_id) = {
            let mut g = graph.write().unwrap();
            let consumer = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v_owned_id],
                shape:  v_owned_shape,
                dtype:  v_owned_dtype,
            });
            let release = g.push(Node {
                op:     Op::Release,
                inputs: vec![producer_id],
                shape:  Shape::from_dims(&[0]),
                dtype:  DType::F32,
            });
            (consumer, release)
        };
        let ord = crate::opt::derive_ordering(
            &graph.read().unwrap(),
            &[consumer_of_owned, release_id],
        );
        let deps = ord.deps_of(release_id);
        let dep_set: std::collections::HashSet<NodeId> = deps.iter().copied().collect();
        // Release MUST wait for ViewOwned to run (ViewOwned reads
        // the producer's bytes during its memcpy).
        assert!(
            dep_set.contains(&v_owned_id),
            "Release must pin after ViewOwned — ViewOwned reads the \
             producer's bundle during its memcpy phase",
        );
        // Release must NOT depend on the ViewOwned's downstream
        // consumer — that consumer reads ViewOwned's independent
        // bytes, not the producer's bundle.
        assert!(
            !dep_set.contains(&consumer_of_owned),
            "Release must NOT pin after consumers of ViewOwned — \
             ViewOwned forks a fresh Storage; its consumers don't \
             read the producer's bundle",
        );
        // Op::View's consumers stay reached via the alias chain; here
        // there are none (v_view has no consumer), so v_view itself
        // is the only alias-set entry beyond the producer.
        let _ = v_view_id;
    }

    /// Long-tail bundle-aware schedule: a producer with three Views,
    /// each consumed by a chain of different depth, and a destructive
    /// Release on the producer. The Release must end up after every
    /// terminal consumer of every chain — bundle-as-single-eviction-
    /// unit semantics fall out of derive_ordering's alias-set walk
    /// reaching each chain's leaf via the producer-then-View hop.
    #[test]
    fn derive_ordering_bundle_release_pins_after_all_chains() {
        // Build a 3-slot bundle. Slot 0 primary [2,3] F32 satisfies
        // the producer's Node::shape/dtype constraint; slots 1 + 2
        // add sibling shapes.
        let slot_specs = vec![
            OutputView {
                byte_offset:  0,
                len_elements: 6,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 3]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 3])),
                name:         Some("a"),
            },
            OutputView {
                byte_offset:  24,
                len_elements: 4,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 2]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 2])),
                name:         Some("b"),
            },
            OutputView {
                byte_offset:  40,
                len_elements: 8,
                dtype:        DType::F32,
                shape:        Shape::from_dims(&[2, 4]),
                layout:       Layout::contiguous(Shape::from_dims(&[2, 4])),
                name:         Some("c"),
            },
        ];
        let (graph, producer_id) = multi_output_producer(slot_specs);
        let producer = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let v0 = producer.view(0).unwrap();
        let v1 = producer.view(1).unwrap();
        let v2 = producer.view(2).unwrap();
        let (v0_id, v0_shape, v0_dtype) = (v0.id(), v0.shape(), v0.dtype());
        let (v1_id, v1_shape, v1_dtype) = (v1.id(), v1.shape(), v1.dtype());
        let (v2_id, v2_shape, v2_dtype) = (v2.id(), v2.shape(), v2.dtype());
        let (leaf_a, leaf_b, leaf_c, release_id) = {
            let mut g = graph.write().unwrap();
            let leaf_a = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v0_id],
                shape:  v0_shape,
                dtype:  v0_dtype,
            });
            let b1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v1_id],
                shape:  v1_shape.clone(),
                dtype:  v1_dtype,
            });
            let leaf_b = g.push(Node {
                op:     Op::Relu,
                inputs: vec![b1],
                shape:  v1_shape,
                dtype:  v1_dtype,
            });
            let c1 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![v2_id],
                shape:  v2_shape.clone(),
                dtype:  v2_dtype,
            });
            let c2 = g.push(Node {
                op:     Op::Relu,
                inputs: vec![c1],
                shape:  v2_shape.clone(),
                dtype:  v2_dtype,
            });
            let leaf_c = g.push(Node {
                op:     Op::Relu,
                inputs: vec![c2],
                shape:  v2_shape,
                dtype:  v2_dtype,
            });
            let release = g.push(Node {
                op:     Op::Release,
                inputs: vec![producer_id],
                shape:  Shape::from_dims(&[0]),
                dtype:  DType::F32,
            });
            (leaf_a, leaf_b, leaf_c, release)
        };
        let ord = crate::opt::derive_ordering(
            &graph.read().unwrap(),
            &[leaf_a, leaf_b, leaf_c, release_id],
        );
        let deps = ord.deps_of(release_id);
        let dep_set: std::collections::HashSet<NodeId> = deps.iter().copied().collect();
        // The direct readers of any View are the FIRST relu in each
        // chain — those are the nodes that read the bundle's bytes.
        // Every consumer further down reads its predecessor's
        // standalone output Storage, so they aren't in the alias-set
        // reader sweep. Verify the Release pins on the direct
        // readers — sufficient for bundle-single-eviction safety
        // because the chains are themselves data-dep ordered after
        // their direct readers.
        assert!(
            dep_set.contains(&leaf_a),
            "single-Relu chain a: leaf is the direct reader",
        );
        // For the b/c chains, the direct readers are the first Relus
        // (b1 / c1), not the leaves. The Release must include those.
        // Find them by walking inputs from leaves.
        let g = graph.read().unwrap();
        let b1 = g.node(leaf_b).inputs[0];
        let (c2, c1) = {
            let c2 = g.node(leaf_c).inputs[0];
            let c1 = g.node(c2).inputs[0];
            (c2, c1)
        };
        let _ = c2;
        assert!(
            dep_set.contains(&b1),
            "two-Relu chain b: first Relu is the direct bundle reader",
        );
        assert!(
            dep_set.contains(&c1),
            "three-Relu chain c: first Relu is the direct bundle reader",
        );
    }

    // ------------------------------------------------------------------
    // Item 4 — Op::View / Op::ViewOwned autograd emits Op::ScatterIntoSlot
    // ------------------------------------------------------------------

    /// The View's backward arm pushes an `Op::ScatterIntoSlot { slot }`
    /// node into the gradient graph as the producer's input gradient.
    /// Synthetic test: build a 2-slot producer + Op::View(slot 1) +
    /// a Relu downstream + invoke Tensor::backward + walk the produced
    /// gradient graph and assert a ScatterIntoSlot was emitted.
    #[test]
    fn op_view_backward_emits_scatter_into_slot() {
        let producer_t = Tensor::from_f32(
            vec![0.0_f32; 6],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let graph: SharedGraph = Arc::clone(producer_t.graph());
        let producer_id = producer_t.id();
        let view_id = {
            let mut g = graph.write().unwrap();
            let specs = two_slot_views_2x3_and_2x4();
            g.set_output_views(producer_id, specs.into()).expect("output_views");
            drop(g);
            let v1 = producer_t.view(1).expect("slot 1 view");
            v1.id()
        };
        // Build a tiny tape-tracked loss = sum(relu(v1)).
        let v1_tensor = Tensor::from_existing(Arc::clone(&graph), view_id);
        let relu_t = v1_tensor.relu();
        let loss_t = relu_t.sum_all();

        // Take backward — exercises the View's backward arm.
        let grads = loss_t.backward();

        // The producer's input gradient should be reachable via a
        // chain that includes an Op::ScatterIntoSlot { slot: 1 } node.
        let producer_tensor = Tensor::from_existing(Arc::clone(&graph), producer_id);
        let grad_for_producer = grads
            .get(&producer_tensor)
            .expect("producer has an accumulated gradient");
        let grad_id = grad_for_producer.id();
        let g = graph.read().unwrap();
        let scatter_present = matches!(g.node(grad_id).op, Op::ScatterIntoSlot { slot: 1 })
            || {
                // Walk one hop back through `Op::Add` accumulations
                // if any.
                let n = g.node(grad_id);
                n.inputs.iter().any(|inp| matches!(
                    g.node(*inp).op, Op::ScatterIntoSlot { slot: 1 }
                ))
            };
        assert!(
            scatter_present,
            "View(slot=1) backward must emit an Op::ScatterIntoSlot {{ slot: 1 }} \
             into the gradient graph; got grad node op = {:?}",
            g.node(grad_id).op,
        );
    }

    // ----- Op::Branch inert scaffold (Phase A PR-A0 of the
    // "plan IS the graph" rebuild) -----

    /// Back-compat by construction: a graph with ZERO `Op::Branch`
    /// nodes builds with exactly the arena it had before the variant
    /// existed. The locked design guarantees "a graph with zero
    /// Op::Branch nodes is exactly today's single-route graph" — this
    /// test pins that: the same builder calls produce the same node
    /// count, the same topo order, and not a single `Op::Branch` node.
    #[test]
    fn zero_branch_graph_builds_unchanged() {
        // Build: c = (a + b) * a — the same DAG the topo-order tests
        // use, exercised here purely to prove the arena is untouched by
        // the new variant.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let sum = a.add(&b);
        let c = sum.mul(&a);

        let g = c.graph();
        let guard = g.read().unwrap();

        // Exactly four nodes (a, b, sum, c) — no implicit Branch
        // anywhere.
        assert_eq!(guard.len(), 4, "zero-Branch graph must have the same node count as before");

        // Topo order is unchanged and contains zero Branch nodes.
        let order = topo_order(&guard, c.id());
        assert_eq!(order.len(), 4);
        let branch_count = order
            .iter()
            .filter(|&&id| matches!(guard.node(id).op, Op::Branch { .. }))
            .count();
        assert_eq!(branch_count, 0, "no builder constructs Op::Branch in PR-A0");

        // The ops are exactly the ones the builders emitted (no
        // structural rewrite slipped in a merge node).
        assert!(matches!(guard.node(a.id()).op, Op::Const));
        assert!(matches!(guard.node(b.id()).op, Op::Const));
        assert!(matches!(guard.node(sum.id()).op, Op::Add));
        assert!(matches!(guard.node(c.id()).op, Op::Mul));
    }

    /// The variant exists and its infallible accessor behaves: a
    /// hand-constructed `Op::Branch` reports `short_name() == "Branch"`,
    /// and the multi-path encoding (arms = `inputs`, explicit
    /// `reconverge_at`) round-trips through the arena. This is the only
    /// place in PR-A0 that constructs the variant — the production
    /// builders land in PR-A1.
    #[test]
    fn branch_variant_exists_and_short_name_works() {
        // short_name / the Op::short_name() facade both return the
        // stable "Branch" constant.
        let reconverge = NodeId(7);
        let branch_op = Op::Branch { reconverge_at: reconverge };
        assert_eq!(branch_op.short_name(), "Branch");
        assert_eq!(op_short_name(&branch_op), "Branch");

        // Structural nodes carry no destructive input (matches the
        // other structural ops: Const / View / ScatterIntoSlot).
        assert_eq!(branch_op.destructive_input(), None);

        // Encode a 2-arm merge directly on the arena: two route exits
        // as `inputs`, the reconvergence node named explicitly. Prove
        // the encoding survives a push + read-back unchanged.
        let mut g = Graph::new();
        let arm0 = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        let arm1 = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        let merged = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        let branch = g.push(Node {
            op: Op::Branch { reconverge_at: merged },
            inputs: vec![arm0, arm1],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });

        let node = g.node(branch);
        assert_eq!(node.op.short_name(), "Branch");
        // Arms are the node's inputs (each input = one route's exit).
        assert_eq!(node.inputs, vec![arm0, arm1]);
        // The explicit reconvergence node is carried on the op.
        match node.op {
            Op::Branch { reconverge_at } => assert_eq!(reconverge_at, merged),
            ref other => panic!("expected Op::Branch, got {other:?}"),
        }
    }

    // ----- Op::Branch builders + build-time validation
    // (Phase A PR-A1 of the "plan IS the graph" rebuild) -----

    /// Hand-build a 2-arm diamond directly on the arena and return the
    /// pieces the branch builders will stitch together:
    /// `diverge` → {`arm0`, `arm1`} → `reconverge`. The reconverge node
    /// already reads arm0 as input 0, which is exactly the runnability
    /// invariant the builders must preserve (a finalized-but-unpicked
    /// branch still realizes on arm 0). Every node is `[2] f32` so the
    /// cast-to-uniform check has agreeing shape/dtype across arms.
    fn diamond_2arm() -> (Graph, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let diverge = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        // arm0 = relu(diverge), arm1 = silu(diverge): two interior nodes,
        // each reachable only through its own arm from the diverge point.
        let arm0 = g.push(Node {
            op: Op::Relu,
            inputs: vec![diverge],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        let arm1 = g.push(Node {
            op: Op::Silu,
            inputs: vec![diverge],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        // reconverge reads arm0 (runnability: arm 0 is always input 0).
        let reconverge = g.push(Node {
            op: Op::Relu,
            inputs: vec![arm0],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });
        (g, diverge, arm0, arm1, reconverge)
    }

    /// (a) A 2-arm diamond round-trips through
    /// `open_branch`/`add_arm`/`finalize_branches`: it produces exactly
    /// one valid `Op::Branch` node whose `inputs` are the arm exits in
    /// order and whose `reconverge_at` is the named merge node.
    #[test]
    fn two_arm_diamond_round_trips_to_branch_node() {
        let (mut g, diverge, arm0, arm1, reconverge) = diamond_2arm();
        let before = g.len();

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch must finalize")
            .expect("2 arms survive — not collapsed to a single arm");

        // Exactly one node was appended (the Branch); no orphan debris.
        assert_eq!(g.len(), before + 1);

        let node = g.node(branch);
        assert!(matches!(node.op, Op::Branch { .. }));
        // Arms are the node's inputs, in add order: arm0 first.
        assert_eq!(node.inputs, vec![arm0, arm1]);
        match node.op {
            Op::Branch { reconverge_at } => assert_eq!(reconverge_at, reconverge),
            ref other => panic!("expected Op::Branch, got {other:?}"),
        }
        // The merge node still carries arm0's shape/dtype (cast-to-uniform
        // validated, not rewritten).
        assert_eq!(node.shape, g.node(arm0).shape);
        assert_eq!(node.dtype, g.node(arm0).dtype);
    }

    /// (b) A `reconverge_at` that is NOT a descendant of the diverge
    /// point returns `Error::InvalidBranch` — never a panic.
    #[test]
    fn reconverge_not_descendant_of_diverge_is_typed_err() {
        let (mut g, diverge, arm0, arm1, _reconverge) = diamond_2arm();
        // A sibling node that does NOT descend from `diverge`.
        let unrelated = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let err = b
            .finalize_branches(&mut g, unrelated)
            .expect_err("reconverge that doesn't descend from diverge must be rejected");
        assert!(
            matches!(err, fuel_ir::Error::InvalidBranch { .. }),
            "expected InvalidBranch, got {err:?}",
        );
    }

    /// (c) A non-disjoint arm — an arm interior node reachable from
    /// outside that arm (shared with the rest of the graph) — returns
    /// `Error::InvalidBranch`.
    #[test]
    fn non_disjoint_arm_is_typed_err() {
        let (mut g, diverge, arm0, arm1, reconverge) = diamond_2arm();
        // Add a reader OUTSIDE the branch that consumes arm0's interior.
        // arm0 is now reachable from a node that isn't part of the merge,
        // so the arm is not internally disjoint.
        let _external_reader = g.push(Node {
            op: Op::Relu,
            inputs: vec![arm0],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let err = b
            .finalize_branches(&mut g, reconverge)
            .expect_err("an arm interior read from outside the branch must be rejected");
        assert!(
            matches!(err, fuel_ir::Error::InvalidBranch { .. }),
            "expected InvalidBranch, got {err:?}",
        );
    }

    /// (c') Arms that disagree on shape/dtype at their exits violate
    /// cast-to-uniform and return `Error::InvalidBranch` (validate-and-Err
    /// — no implicit Cast insertion in PR-A1).
    #[test]
    fn arms_disagreeing_on_dtype_is_typed_err() {
        let (mut g, diverge, arm0, _arm1, reconverge) = diamond_2arm();
        // A second arm with a DIFFERENT dtype than arm0.
        let arm1_f16 = g.push(Node {
            op: Op::Silu,
            inputs: vec![diverge],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F16,
        });

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1_f16);
        let err = b
            .finalize_branches(&mut g, reconverge)
            .expect_err("arms with mismatched dtype must be rejected (cast-to-uniform)");
        assert!(
            matches!(err, fuel_ir::Error::InvalidBranch { .. }),
            "expected InvalidBranch, got {err:?}",
        );
    }

    /// (d) `finalize_branches` drops a single-arm branch: with only one
    /// surviving arm there is no real decision point, so no `Op::Branch`
    /// node is emitted (`Ok(None)`), the arena is left untouched, and the
    /// graph collapses back to that arm.
    #[test]
    fn single_arm_branch_is_dropped() {
        let (mut g, diverge, arm0, _arm1, reconverge) = diamond_2arm();
        let before = g.len();

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        let result = b
            .finalize_branches(&mut g, reconverge)
            .expect("a single-arm branch is valid — it just collapses");
        assert!(result.is_none(), "single-arm branch must NOT emit an Op::Branch");
        // No node appended: the lone arm is the graph, unchanged.
        assert_eq!(g.len(), before, "dropping a single-arm branch leaves the arena untouched");
        let branch_count = (0..g.len())
            .filter(|&i| matches!(g.node(NodeId(i)).op, Op::Branch { .. }))
            .count();
        assert_eq!(branch_count, 0);
    }

    /// (e) The runnability invariant: the emitted `Op::Branch`'s inputs
    /// always include arm-0's exit as input 0, so a finalized-but-not-yet-
    /// route-picked graph still realizes on arm 0 (a graph with a
    /// finalized branch must still be executable). The named reconverge
    /// node reads arm-0's exit, so realizing on arm 0 alone produces a
    /// defined value at the merge.
    #[test]
    fn finalized_branch_preserves_arm0_runnability() {
        let (mut g, diverge, arm0, arm1, reconverge) = diamond_2arm();

        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed branch finalizes")
            .expect("2 arms survive");

        // arm 0 is input 0 of the Branch — the always-realizable route.
        assert_eq!(g.node(branch).inputs.first().copied(), Some(arm0));
        // And the named reconverge node reads arm0, so realizing on
        // arm 0 alone produces a defined value at the merge.
        assert!(
            g.node(reconverge).inputs.contains(&arm0),
            "reconverge must read arm-0's exit so the unpicked graph runs on arm 0",
        );
    }

    // ===================================================================
    // PR-B4: required compaction of the append-only arena.
    // ===================================================================

    use crate::run::{extract_runs, lower_runs_arm0};

    /// Build the same 2-arm diamond the run.rs tests use, finalized into a
    /// real `Op::Branch`. Returns `(g, diverge, arm0, arm1, reconverge,
    /// branch, post)`. `post` reads `reconverge`, so the post-merge region
    /// is a non-empty run and `post` is a stable root.
    fn diamond_with_branch_b4() -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mk = |g: &mut Graph, op: Op, inputs: Vec<NodeId>| {
            g.push(Node { op, inputs, shape: Shape::from_dims(&[2]), dtype: DType::F32 })
        };
        let mut g = Graph::new();
        let pre = mk(&mut g, Op::Const, vec![]);
        let diverge = mk(&mut g, Op::Relu, vec![pre]);
        let arm0 = mk(&mut g, Op::Silu, vec![diverge]);
        let arm1 = mk(&mut g, Op::Gelu, vec![diverge]);
        let reconverge = mk(&mut g, Op::Relu, vec![arm0]);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch finalizes")
            .expect("2 arms survive");
        let post = mk(&mut g, Op::Tanh, vec![reconverge]);
        (g, diverge, arm0, arm1, reconverge, branch, post)
    }

    /// PR-D1 storage-class substrate: the op-inferred class is correct
    /// (`Op::Const` → Shared, `Op::WriteSlice` → SessionState, other →
    /// Transient), an explicit override beats the inferred default (a
    /// KV-cache placeholder `Op::Const` marked SessionState), and `compact`
    /// carries the override to the node's new id while inferred defaults
    /// still hold. Born-red: fails if classification, override precedence,
    /// or the compaction remap is wrong.
    #[test]
    fn storage_class_inference_override_and_compaction() {
        let mut g = Graph::new();
        // A shared weight Const, a session-state KV placeholder Const, a
        // transient activation, and an in-place cache-write target.
        let weight = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let kv = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let act = g.push(Node { op: Op::Relu, inputs: vec![weight], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let write = g.push(Node {
            op: Op::WriteSlice { ranges: vec![(0, 2)], dyn_offset: None },
            inputs: vec![kv, act],
            shape: Shape::from_dims(&[2]),
            dtype: DType::F32,
        });

        // Op-inferred defaults.
        assert_eq!(g.storage_class(weight), StorageClass::Shared, "Const → Shared");
        assert_eq!(g.storage_class(act), StorageClass::Transient, "activation → Transient");
        assert_eq!(g.storage_class(write), StorageClass::SessionState, "WriteSlice → SessionState");
        // No overrides set yet.
        assert_eq!(g.storage_class(kv), StorageClass::Shared, "KV placeholder Const infers Shared before override");
        assert!(!g.has_storage_class_override(kv));
        assert_eq!(g.storage_class_override_count(), 0);

        // Explicit override: the KV placeholder is session state, not a
        // shared weight.
        g.set_storage_class(kv, StorageClass::SessionState);
        assert_eq!(g.storage_class(kv), StorageClass::SessionState, "override beats inferred default");
        assert!(g.has_storage_class_override(kv));
        assert_eq!(g.storage_class_override_count(), 1);

        // Compaction carries the override to the new id; inferred defaults
        // (no side-table entry) still resolve from the op.
        let remap = compact(&mut g, &[write]);
        let n_kv = remap.get(kv).expect("kv survives (input of write)");
        let n_weight = remap.get(weight).expect("weight survives (transitively under act)");
        let n_write = remap.get(write).expect("write is the root");
        assert_eq!(g.storage_class(n_kv), StorageClass::SessionState, "override remapped by compact");
        assert!(g.has_storage_class_override(n_kv));
        assert_eq!(g.storage_class_override_count(), 1, "exactly the one override survives");
        assert_eq!(g.storage_class(n_weight), StorageClass::Shared, "inferred Shared still holds post-compact");
        assert_eq!(g.storage_class(n_write), StorageClass::SessionState, "inferred SessionState still holds post-compact");
        assert!(g.verify_no_dangling().is_ok());
    }

    /// (a) A deliberately-orphaned node (pushed, referenced by nothing) is
    /// dropped by `compact`; the node count shrinks by exactly the orphan
    /// count, and every still-reachable node survives.
    #[test]
    fn compact_drops_orphan_debris() {
        let mut g = Graph::new();
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Relu, inputs: vec![a], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let c = g.push(Node { op: Op::Silu, inputs: vec![b], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        // Two pieces of exploration debris: reachable from nothing in the
        // live structure (no root, no side-effect, no branch references them).
        let _orphan0 = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let _orphan1 = g.push(Node { op: Op::Tanh, inputs: vec![_orphan0], shape: Shape::from_dims(&[2]), dtype: DType::F32 });

        let before = g.len();
        let remap = compact(&mut g, &[c]);

        assert_eq!(
            g.len(),
            before - 2,
            "compaction drops exactly the two orphan-debris nodes",
        );
        // The three reachable nodes all survived (have new ids).
        assert!(remap.get(a).is_some());
        assert!(remap.get(b).is_some());
        assert!(remap.get(c).is_some());
        // The orphans are gone.
        assert!(remap.get(_orphan0).is_none());
        assert!(remap.get(_orphan1).is_none());
        // The surviving chain still resolves: c' reads b' reads a'.
        let (na, nb, nc) = (remap.get(a).unwrap(), remap.get(b).unwrap(), remap.get(c).unwrap());
        assert_eq!(g.node(nc).inputs, vec![nb]);
        assert_eq!(g.node(nb).inputs, vec![na]);
        assert_eq!(g.node(na).inputs, Vec::<NodeId>::new());
        assert_no_dangling(&g);
    }

    /// (b) Branches preserved: a graph with a finalized `Op::Branch` keeps
    /// every arm + `reconverge_at` after compaction (remapped), and the
    /// arm-0 lowering yields the same logical route.
    #[test]
    fn compact_preserves_branch_arms_and_reconverge() {
        let (mut g, _diverge, arm0, arm1, reconverge, branch, post) = diamond_with_branch_b4();

        // Inject orphan debris so compaction has something to drop and the
        // surviving-branch claim is non-trivial.
        let _debris = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });

        // Baseline arm-0 route over old ids, mapped to the op sequence.
        let before_ops: Vec<&'static str> = lower_runs_arm0(&g, &[post])
            .iter()
            .map(|&id| g.node(id).op.short_name())
            .collect();

        let remap = compact(&mut g, &[post]);

        // Every branch participant survived.
        let narm0 = remap.get(arm0).expect("arm0 survives");
        let narm1 = remap.get(arm1).expect("arm1 (the non-fallback arm) survives");
        let nrecon = remap.get(reconverge).expect("reconverge survives");
        let nbranch = remap.get(branch).expect("the Branch node itself survives");
        let npost = remap.get(post).expect("post survives");

        // The Branch's inputs are the remapped arm exits (arm0 first), and
        // its reconverge_at is the remapped reconverge — it still resolves.
        let bn = g.node(nbranch);
        assert!(matches!(bn.op, Op::Branch { .. }));
        assert_eq!(bn.inputs, vec![narm0, narm1], "arm exits remapped, arm0 first");
        match bn.op {
            Op::Branch { reconverge_at } => assert_eq!(reconverge_at, nrecon),
            ref other => panic!("expected Op::Branch, got {other:?}"),
        }
        // reconverge still reads arm0 (runnability preserved through remap).
        assert!(g.node(nrecon).inputs.contains(&narm0));
        // The arm-0 route is logically identical (same op sequence).
        let after_ops: Vec<&'static str> = lower_runs_arm0(&g, &[npost])
            .iter()
            .map(|&id| g.node(id).op.short_name())
            .collect();
        assert_eq!(before_ops, after_ops, "arm-0 route is logically unchanged");
        assert_no_dangling(&g);
    }

    /// (c) Base map preserved: a branchless, fully-reachable graph compacts
    /// to itself — no node dropped; ids may renumber but the structure is
    /// identical.
    #[test]
    fn compact_branchless_reachable_is_identity_in_structure() {
        let mut g = Graph::new();
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let b = g.push(Node { op: Op::Relu, inputs: vec![a], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let c = g.push(Node { op: Op::Silu, inputs: vec![b], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        let d = g.push(Node { op: Op::Tanh, inputs: vec![c], shape: Shape::from_dims(&[2]), dtype: DType::F32 });

        let before_ops: Vec<&'static str> =
            extract_runs(&g, d).iter().flat_map(|r| r.members.iter().map(|&id| g.node(id).op.short_name()).collect::<Vec<_>>()).collect();
        let before_len = g.len();

        let remap = compact(&mut g, &[d]);

        assert_eq!(g.len(), before_len, "a fully-reachable graph drops nothing");
        for n in [a, b, c, d] {
            assert!(remap.get(n).is_some(), "every node survives");
        }
        // Structure identical: same op sequence over the compacted graph.
        let nd = remap.get(d).unwrap();
        let after_ops: Vec<&'static str> =
            extract_runs(&g, nd).iter().flat_map(|r| r.members.iter().map(|&id| g.node(id).op.short_name()).collect::<Vec<_>>()).collect();
        assert_eq!(before_ops, after_ops);
        assert_no_dangling(&g);
    }

    /// (d) No dangling + round-trip: after `compact`, NO NodeId reference
    /// (node inputs, op-carried `reconverge_at`, or any side-table) points
    /// outside the new arena, AND `extract_runs` / arm-0 lowering on the
    /// compacted graph yields the identical logical result, including for
    /// graphs carrying every side-table (placements, target_backends,
    /// layouts, storage_map, node_output_views, side_effect_roots).
    #[test]
    fn compact_no_dangling_and_round_trip_with_side_tables() {
        let (mut g, diverge, arm0, _arm1, reconverge, _branch, post) = diamond_with_branch_b4();

        // Populate side-tables on surviving nodes so the remap must carry
        // them. (target_backend on a couple nodes; placement on one.)
        g.set_target_backend(arm0, BackendId::Cpu);
        g.set_target_backend(reconverge, BackendId::Cpu);
        g.set_placement(diverge, DeviceLocation::Cpu);
        // A side-effect root that is itself orphan-from-roots: a Release-like
        // op the executor must keep even though no output reads it.
        let se = g.push(Node { op: Op::Const, inputs: vec![post], shape: Shape::from_dims(&[2]), dtype: DType::F32 });
        g.add_side_effect_root(se);

        // Orphan debris (referenced by nothing live) to force a real drop.
        let _debris = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2]), dtype: DType::F32 });

        let before_run_ops: Vec<&'static str> = lower_runs_arm0(&g, &[post])
            .iter()
            .map(|&id| g.node(id).op.short_name())
            .collect();

        let remap = compact(&mut g, &[post]);

        // No dangling references anywhere.
        assert_no_dangling(&g);

        // The side-effect root survived (a side-effect-only node is a
        // reachability seed) and its side-table entry was remapped.
        let nse = remap.get(se).expect("side-effect root survives compaction");
        assert!(
            g.side_effect_roots().contains(&nse),
            "the remapped side-effect root is registered post-compaction",
        );
        // The surviving target_backend side-table is keyed by the new ids.
        assert_eq!(g.target_backend(remap.get(arm0).unwrap()), Some(BackendId::Cpu));
        assert_eq!(g.target_backend(remap.get(reconverge).unwrap()), Some(BackendId::Cpu));
        assert_eq!(g.placement(remap.get(diverge).unwrap()), Some(DeviceLocation::Cpu));

        // Round-trip: the arm-0 route is logically identical.
        let npost = remap.get(post).unwrap();
        let after_run_ops: Vec<&'static str> = lower_runs_arm0(&g, &[npost])
            .iter()
            .map(|&id| g.node(id).op.short_name())
            .collect();
        assert_eq!(
            before_run_ops, after_run_ops,
            "compaction preserves the realized arm-0 route exactly",
        );
    }

    /// Helper: assert NO NodeId reference in the graph points outside the
    /// arena. Covers node `inputs`, the op-carried `Op::Branch.reconverge_at`,
    /// every NodeId-keyed side-table, and the `side_effect_roots` vec — the
    /// exhaustive no-dangling safety net for a missed remap.
    fn assert_no_dangling(g: &Graph) {
        let n = g.len();
        let ok = |id: NodeId| id.0 < n;
        for i in 0..n {
            let node = g.node(NodeId(i));
            for &inp in &node.inputs {
                assert!(ok(inp), "Node#{i} input {:?} dangles (len={n})", inp);
            }
            if let Op::Branch { reconverge_at } = node.op {
                assert!(ok(reconverge_at), "Node#{i} Branch.reconverge_at {:?} dangles (len={n})", reconverge_at);
            }
        }
        for &r in g.side_effect_roots() {
            assert!(ok(r), "side_effect_root {:?} dangles (len={n})", r);
        }
        // Use the graph's own verification entry point as the load-bearing
        // assert (it covers every side-table by reading the private fields).
        assert!(
            g.verify_no_dangling().is_ok(),
            "Graph::verify_no_dangling found a dangling reference: {:?}",
            g.verify_no_dangling().err(),
        );
    }
}

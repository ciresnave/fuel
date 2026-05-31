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

use crate::registry::{FusedOpId, FusedOpParams};
use fuel_core_types::{DeviceLocation, DType, Layout, Scalar, Shape, Storage, probe::BackendId};
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
    WriteSlice { ranges: Vec<(usize, usize)> },

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
}

impl Op {
    /// Index into `inputs` that this op destroys on execution. `None`
    /// means the op is non-destructive — every input remains readable
    /// after the op completes. Destructive ops need the scheduler to
    /// pin them to run after all other readers of the destroyed input,
    /// via ordering edges derived by [`opt::derive_ordering`].
    pub fn destructive_input(&self) -> Option<usize> {
        match self {
            Op::Release | Op::Move { .. } | Op::WriteSlice { .. } | Op::ZeroFill => Some(0),
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
) -> Result<Layout, fuel_core_types::Error> {
    match op {
        Op::Transpose => {
            let rank = input_layout.shape().rank();
            if rank < 2 {
                return Err(fuel_core_types::Error::Msg(format!(
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
        other => Err(fuel_core_types::Error::Msg(format!(
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
        Op::Alloc{..}            => "Alloc",
        Op::ZeroFill             => "ZeroFill",
        // Phase 7.6: registry-extended fused ops. Step 3 wires per-id
        // names through a static lookup; until then, all fused ops
        // share one short name. Distinguishing in error messages is
        // future work — `id` is in the Debug repr.
        Op::Fused(_, _)          => "Fused",
    }
}

/// G2 helper: element count of a `HostBuffer`. Used by Tensor's
/// constructors to validate that the supplied data matches the
/// declared shape's `elem_count` before allocating Storage.
fn host_buffer_elem_count(buf: &fuel_core_types::HostBuffer) -> usize {
    use fuel_core_types::HostBuffer;
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
    /// can elide the Copy node entirely). Cross-device copies go
    /// through the active backend's `GraphBackend::copy_to` — today a
    /// host round-trip in the
    /// [`Router`](../fuel_graph_router/struct.Router.html); P2P comes
    /// later.
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
    ) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let dest_shape = self.shape();
        let dest_dims = dest_shape.dims();
        let src_shape = source.shape();
        let src_dims = src_shape.dims();
        let rank = dest_dims.len();
        if ranges.len() != rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "write_slice: ranges.len() ({}) must equal destination rank ({rank})",
                ranges.len(),
            )).bt());
        }
        if src_dims.len() != rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "write_slice: source rank ({}) must equal destination rank ({rank})",
                src_dims.len(),
            )).bt());
        }
        for (i, &(start, end)) in ranges.iter().enumerate() {
            if end < start {
                return Err(fuel_core_types::Error::Msg(format!(
                    "write_slice: ranges[{i}] = ({start}, {end}) has end < start"
                )).bt());
            }
            if end > dest_dims[i] {
                return Err(fuel_core_types::Error::Msg(format!(
                    "write_slice: ranges[{i}].end ({end}) > destination dim {i} ({})",
                    dest_dims[i],
                )).bt());
            }
            let slab = end - start;
            if src_dims[i] != slab {
                return Err(fuel_core_types::Error::Msg(format!(
                    "write_slice: source dim {i} ({}) must equal slab width ({slab}) \
                     = ranges[{i}].end - ranges[{i}].start",
                    src_dims[i],
                )).bt());
            }
        }
        if self.dtype() != source.dtype() {
            return Err(fuel_core_types::Error::Msg(format!(
                "write_slice: dtype mismatch — destination {:?} vs source {:?}",
                self.dtype(), source.dtype(),
            )).bt());
        }
        let dtype = self.dtype();
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::WriteSlice { ranges },
            inputs: vec![self.id, source.id],
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
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f32]> = data.into();
        let buf = fuel_core_types::HostBuffer::F32(v.to_vec());
        Self::from_host_buffer(buf, DType::F32, shape, device)
    }

    /// Build a `Const` tensor from an `f64` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_f64(
        data: impl Into<Arc<[f64]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f64]> = data.into();
        let buf = fuel_core_types::HostBuffer::F64(v.to_vec());
        Self::from_host_buffer(buf, DType::F64, shape, device)
    }

    /// Build a `Const` tensor from a `bf16` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_bf16(
        data: impl Into<Arc<[bf16]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[bf16]> = data.into();
        let buf = fuel_core_types::HostBuffer::BF16(v.to_vec());
        Self::from_host_buffer(buf, DType::BF16, shape, device)
    }

    /// Build a `Const` tensor from an `f16` slice and shape on a fresh graph.
    /// `device` selects where the realized Storage is allocated.
    pub fn from_f16(
        data: impl Into<Arc<[f16]>>,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[f16]> = data.into();
        let buf = fuel_core_types::HostBuffer::F16(v.to_vec());
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
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
    ) -> Self {
        let v: Arc<[u32]> = data.into();
        let buf = fuel_core_types::HostBuffer::U32(v.to_vec());
        Self::from_host_buffer(buf, DType::U32, shape, device)
    }

    /// G2 internal funnel: allocate Storage on `device` from `buf`,
    /// register the slot, emit `Op::Const`. Per-dtype `from_*` methods
    /// delegate here.
    fn from_host_buffer(
        buf: fuel_core_types::HostBuffer,
        dtype: DType,
        shape: impl Into<Shape>,
        device: &Arc<dyn fuel_core_types::DynBackendDevice>,
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
            fuel_core_types::HostBuffer::F32(v.to_vec()), DType::F32, shape,
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
            fuel_core_types::HostBuffer::F64(v.to_vec()), DType::F64, shape,
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
            fuel_core_types::HostBuffer::BF16(v.to_vec()), DType::BF16, shape,
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
            fuel_core_types::HostBuffer::F16(v.to_vec()), DType::F16, shape,
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
            fuel_core_types::HostBuffer::U32(v.to_vec()), DType::U32, shape,
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
            fuel_core_types::HostBuffer::U8(v.to_vec()), DType::U8, shape,
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
            fuel_core_types::HostBuffer::I64(v.to_vec()), DType::I64, shape,
        )
    }

    /// G2 internal funnel for the per-graph const_*_like family. The
    /// device is derived from `self`'s graph slot (any existing one)
    /// so callers don't have to thread a device through hot paths
    /// like RoPE table construction or LoRA application — the const
    /// goes on the same device as the graph it joins.
    fn const_like_host_buffer(
        &self,
        buf: fuel_core_types::HostBuffer,
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
    /// `Arc<RwLock<fuel_storage::Storage>>` type, not the legacy
    /// `fuel_core_types::Storage` that `const_like_from_storage` takes)
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
    pub fn try_permute(&self, axes: &[usize]) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if axes.len() != rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "permute: axes length {} must equal tensor rank {}",
                axes.len(), rank,
            )).bt());
        }
        let mut seen = vec![false; rank];
        for &ax in axes {
            if ax >= rank {
                return Err(fuel_core_types::Error::Msg(format!(
                    "permute: axis {ax} out of bounds for rank {rank}",
                )).bt());
            }
            if seen[ax] {
                return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn try_transpose(&self) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_dims = self.shape();
        let d = in_dims.dims();
        if d.len() < 2 {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn pow(&self, other: &Tensor) -> std::result::Result<Tensor, fuel_core_types::Error> {
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
    pub fn rem(&self, other: &Tensor) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let out_shape = self.shape();
        self.try_binary_op("rem", Op::Rem, other, out_shape)
    }

    /// Append a `Flip` node — reverses element order along `dim`.
    /// Output shape == input shape. Materializing op (real byte
    /// shuffle; not a metadata-only view). Differentiable
    /// (involutive: backward is another Flip on the same dim).
    ///
    /// **Returns `Result`**: bad `dim` surfaces as a typed error.
    pub fn flip(&self, dim: usize) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn roll(&self, dim: usize, shift: i64) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn cumsum(&self, dim: usize) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn triu(&self, diagonal: i64) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if rank < 2 {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn tril(&self, diagonal: i64) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let rank = in_shape.dims().len();
        if rank < 2 {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn log_softmax_last_dim(&self) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        if in_shape.dims().is_empty() {
            return Err(fuel_core_types::Error::Msg(
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
    ) -> std::result::Result<Tensor, fuel_core_types::Error> {
        if self.shape().dims() != mask.shape().dims() {
            return Err(fuel_core_types::Error::Msg(format!(
                "masked_fill: x.shape={:?} != mask.shape={:?}",
                self.shape().dims(), mask.shape().dims(),
            )).bt());
        }
        if mask.dtype() != DType::U8 {
            return Err(fuel_core_types::Error::Msg(format!(
                "masked_fill: mask dtype must be U8, got {:?}",
                mask.dtype(),
            )).bt());
        }
        if value.dtype() != self.dtype() {
            return Err(fuel_core_types::Error::Msg(format!(
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
    ) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if padding.len() != rank {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn try_broadcast_to(&self, target: impl Into<Shape>) -> std::result::Result<Tensor, fuel_core_types::Error> {
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
    pub fn try_unsqueeze(&self, dim: usize) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if dim > rank {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn squeeze(&self, dim: usize) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let in_shape = self.shape();
        let in_dims = in_shape.dims();
        let rank = in_dims.len();
        if dim >= rank {
            return Err(fuel_core_types::Error::Msg(format!(
                "squeeze: dim {dim} out of bounds for rank {rank} (must be < rank)",
            )).bt());
        }
        if in_dims[dim] != 1 {
            return Err(fuel_core_types::Error::Msg(format!(
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
    pub fn try_reshape(&self, target: impl Into<Shape>) -> std::result::Result<Tensor, fuel_core_types::Error> {
        let target = target.into();
        let from = self.shape().elem_count();
        let to = target.elem_count();
        if from != to {
            return Err(fuel_core_types::Error::Msg(format!(
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
    /// `chunk_size` is the SSD block size. v1 requires
    /// `chunk_size == seqlen` (single-chunk degenerate case);
    /// multi-chunk inter-block decay propagation is a follow-up.
    /// The kernel returns an error if the constraint is violated.
    ///
    /// Emits `Op::Fused(FusedOps::SSD_CHUNK_SCAN,
    /// FusedOpParams::SsdChunkScan { chunk_size })`. No primitive
    /// decomposition; backends without a native kernel fall through
    /// to the executor's cpu_fallback path.
    pub fn ssd_chunk_scan(
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
        for (name, t) in [("x", self), ("dt", dt), ("a", a), ("b", b), ("c", c)] {
            assert_eq!(
                t.dtype(),
                DType::F32,
                "ssd_chunk_scan v1: {name} must be F32, got {:?}", t.dtype(),
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
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::SSD_CHUNK_SCAN,
                crate::registry::FusedOpParams::SsdChunkScan { chunk_size },
            ),
            inputs: vec![self.id, dt.id, a.id, b.id, c.id],
            shape:  Shape::from_dims(&[batch, seqlen, heads, head_dim]),
            dtype:  DType::F32,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
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
    pub fn selective_scan(
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
        let id = self.graph.write().unwrap().push(Node {
            op:     Op::Fused(
                crate::registry::FusedOps::SELECTIVE_SCAN,
                crate::registry::FusedOpParams::SelectiveScan { delta_softplus },
            ),
            inputs: vec![self.id, delta.id, a.id, b.id, c.id],
            shape:  Shape::from_dims(&[batch, seqlen, dim]),
            dtype,
        });
        Self {
            graph: self.graph.clone(),
            id,
        }
    }

    /// Append a `FusedSoftmaxCrossEntropy` node. Two inputs:
    /// - `self` (logits): `[..., V]` F32
    /// - `targets`: `[...]` I64 (class indices; matches PyTorch /
    ///   baracuda convention)
    ///
    /// Output dtype is always F32; output shape depends on `reduction`:
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
        assert_eq!(
            self.dtype(),
            DType::F32,
            "fused_softmax_cross_entropy v1: logits must be F32, got {:?}",
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
    ) -> std::result::Result<Tensor, fuel_core_types::Error> {
        if !Arc::ptr_eq(&self.graph, &other.graph) {
            return Err(fuel_core_types::Error::Msg(format!(
                "{name}: tensors must live on the same graph",
            )).bt());
        }
        if self.dtype() != other.dtype() {
            return Err(fuel_core_types::Error::Msg(format!(
                "{name}: dtype mismatch: lhs={:?}, rhs={:?}",
                self.dtype(),
                other.dtype(),
            )).bt());
        }
        if self.shape().dims() != other.shape().dims() {
            return Err(fuel_core_types::Error::Msg(format!(
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
                    let zero = fuel_core_types::Scalar::zero(dtype);
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
                        // Backward via recompute is implemented in
                        // fuel_reference_backend::attention::attention_flash_backward.
                        // Wiring it as a graph rewrite needs three
                        // new gradient nodes (dQ, dK, dV) plus the
                        // recompute pass — sized as a follow-up.
                        // Today's lazy gradient is undefined for
                        // FlashAttn; users who need
                        // training-on-attention should compose
                        // attention from matmul + softmax (which has
                        // working gradients) until the rule lands.
                        panic!(
                            "Tensor::backward: FlashAttn does not \
                             yet have a gradient rule. Compose \
                             attention from matmul + softmax for \
                             differentiable use.",
                        );
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
) -> std::result::Result<(), fuel_core_types::Error> {
    if src.len() > dst.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "broadcast_to: source rank {} exceeds target rank {}",
            src.len(), dst.len(),
        )).bt());
    }
    let pad = dst.len() - src.len();
    for (i, &s) in src.iter().enumerate() {
        let d = dst[pad + i];
        if s != d && s != 1 {
            return Err(fuel_core_types::Error::Msg(format!(
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
        DType::F32 => fuel_core_types::HostBuffer::F32(vec![value as f32; n]),
        DType::F64 => fuel_core_types::HostBuffer::F64(vec![value; n]),
        DType::BF16 => fuel_core_types::HostBuffer::BF16(vec![bf16::from_f64(value); n]),
        DType::F16 => fuel_core_types::HostBuffer::F16(vec![f16::from_f64(value); n]),
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
fn pick_device_from_graph(graph: &SharedGraph) -> Arc<dyn fuel_core_types::DynBackendDevice> {
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
    fn cpu_dev() -> &'static Arc<dyn fuel_core_types::DynBackendDevice> {
        static D: std::sync::OnceLock<Arc<dyn fuel_core_types::DynBackendDevice>>
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
        use fuel_core_types::DimVec;
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
            fuel_core_types::StrideVec::from_slice(&[1_isize, 3]),
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
        use fuel_core_types::StrideVec;
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
        let op = Op::WriteSlice { ranges: vec![(0, 1), (0, 32), (0, 128)] };
        assert_eq!(op.destructive_input(), Some(0));
    }

    #[test]
    fn write_slice_short_name() {
        let op = Op::WriteSlice { ranges: vec![(0, 1)] };
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
            Op::WriteSlice { ranges } => {
                assert_eq!(ranges, &vec![(2, 3), (0, 3)]);
            }
            other => panic!("expected Op::WriteSlice, got {other:?}"),
        }
        assert_eq!(g.node(out.id()).inputs, vec![dest.id(), src.id()]);
        // Output shape == destination shape; bytes are post-write same buffer.
        assert_eq!(g.node(out.id()).shape.dims(), &[4, 3]);
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
}

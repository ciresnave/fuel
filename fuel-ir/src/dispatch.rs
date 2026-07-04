//! Cross-crate dispatch table types — Phase 6b's empirical
//! `(op, dtype, size_class) → (backend, device)` lookup.
//!
//! These types live here, not in `fuel-core`, because both
//! `fuel-core` (which produces tables via the [`Judge`] in its
//! `judge` module) and `fuel-graph-router` (which consumes them at
//! op-dispatch time to pick between competing backends) need the
//! same shapes. The producer-side `Judge` and the
//! `populate_dispatch_table` / `cached` cache APIs live in
//! `fuel-core` because they call into `LazyTensor`'s realize path.
//! Only the data + the pure lookup helpers are here.
//!
//! [`Judge`]: https://docs.rs/fuel-core/latest/fuel_core/judge/struct.Judge.html

use crate::DType;
use crate::Shape;
use crate::probe::BackendId;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Schema version for persisted profile reports. Bump when the
/// entry layout changes in a way that can't be covered by
/// `#[serde(default)]`.
///
/// **v2 (2026-06-08)**: per-alternative measurement landed —
/// [`ProfileEntry`] gains `kernel_source` so the AOCL/MKL/portable-cpu
/// kernel siblings at one `(op, dtype, size_class, BackendId::Cpu)`
/// cell each get their own measured entry. v1 reports lacked the
/// field and were ambiguous about which CPU-substrate kernel they
/// timed; old reports load as `Ok(None)` (cache miss → re-Judge).
///
/// **v3 (2026-07-04)**: dtype-axis coverage landed — the Judge's
/// measurement matrix now iterates `{F32, F16, BF16}` for every
/// profiled op/size (Judge Layer-2 coverage arc, dtype slice). The
/// [`ProfileEntry`] *layout* is unchanged (`dtype` was always a
/// field), but the *coverage* is: a v3 report carries f16/bf16 cells
/// a v2 report never had. The bump is deliberate rather than
/// serde-forced: without it, `populate_dispatch_table`'s idempotence
/// guard would keep an upgraded install's f32-only v2 profile forever
/// (the guard no-ops when a cached report exists), silently denying
/// the ranker the new per-dtype latencies. Bumping makes v2 reports
/// load as `Ok(None)` (cache miss → clean re-Judge of the full dtype
/// matrix) — the same migration shape v1→v2 used. A v2 report is not
/// *wrong*, just partial; the empty-oracle version gate in
/// `ProfileJudgeOracle::from_report` and `ProfileReport::load` both
/// reject it so no partial data leaks into the cost composer.
///
/// **v4 (2026-07-04)**: matmul [`SizeClass`] repr change — the
/// aspect-blind `log2(total_elements)` key that keyed a matmul on
/// `m·n` (output) is replaced, *for matmul only*, by an aspect key
/// packing `(log2(m·n), log2(m), log2(k))` (see [`SizeClass`]). This
/// fixes a latent producer/consumer key disagreement: the Judge keyed
/// matmul cells on `m·n` while the `fuel-dispatch` ranker keyed its
/// realize-time lookup on `m·k` (LHS input), so every *non-square*
/// matmul lookup missed the profiled cell. Producer and consumer now
/// derive the key from the operand shapes through the SAME
/// [`SizeClass::matmul`] / [`SizeClass::for_op`] helper. The
/// [`SizeClass`] field also widens `u8 → u32` to hold the packed key.
/// v3 reports (scalar matmul keys) load as `Ok(None)` — cache miss →
/// clean re-Judge — the established migration (v1→v2, v2→v3). The
/// `ProfileReport::load` / `ProfileJudgeOracle::from_report` version
/// gates reject the older schema so no scalar matmul key leaks into
/// the aspect-keyed lookup path.
pub const PROFILE_REPORT_VERSION: u32 = 4;

/// Op kinds the Judge profiles. Adding a variant + a Judge match
/// arm extends the profile matrix; existing reports parse forward
/// thanks to `#[non_exhaustive]`.
///
/// Phase 7.5 storage unification — Phase C grows this enum one op
/// family per migration step. Each variant is the dispatch key that
/// pairs with [`crate::DType`] and
/// [`crate::probe::BackendId`] to select a kernel. The naming
/// convention is `<Op><Family>` (e.g. `ReluElementwise`,
/// `MulElementwise`); families currently in use:
///
/// - elementwise binary: `Add/Sub/Mul/Div` + `Elementwise`
/// - elementwise unary: `Relu/Neg/Sqr/Sqrt/Recip/Abs/Tanh/...`
///   + `Elementwise`
/// - dense linear algebra: `MatMul`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum OpKind {
    /// Dense matrix multiply, `[M, K] @ [K, N] → [M, N]`.
    MatMul,
    /// Elementwise addition of two equally-shaped tensors.
    AddElementwise,
    /// Elementwise subtraction (`a - b`).
    SubElementwise,
    /// Elementwise multiplication.
    MulElementwise,
    /// Elementwise division (`a / b`).
    DivElementwise,
    /// Elementwise rectified linear unit (`max(0, x)`).
    ReluElementwise,
    /// Elementwise negation (`-x`).
    NegElementwise,
    /// Elementwise square (`x * x`).
    SqrElementwise,
    /// Elementwise square root.
    SqrtElementwise,
    /// Elementwise reciprocal (`1 / x`).
    RecipElementwise,
    /// Elementwise absolute value.
    AbsElementwise,
    /// Elementwise hyperbolic tangent.
    TanhElementwise,
    /// Elementwise exponential (`e^x`).
    ExpElementwise,
    /// Elementwise natural logarithm (`ln(x)`).
    LogElementwise,
    /// Elementwise sine.
    SinElementwise,
    /// Elementwise cosine.
    CosElementwise,
    /// Elementwise logistic sigmoid (`1 / (1 + e^-x)`).
    SigmoidElementwise,
    /// SiLU / Swish activation (`x * sigmoid(x)`).
    SiluElementwise,
    /// GELU activation, tanh approximation
    /// (`0.5*x*(1 + tanh(√(2/π) * (x + 0.044715*x³)))`).
    GeluElementwise,
    /// Heaviside step function (`1` where `x > 0`, `0` otherwise) —
    /// the derivative of [`OpKind::ReluElementwise`].
    StepElementwise,

    /// Sum-reduce one or more dimensions of the input. The reduced
    /// dims and the input shape live in
    /// [`OpParams::Reduce`](super::dispatch::OpKind); the output is
    /// the input with those dims dropped (or rank-0 when every dim
    /// is reduced).
    SumReduce,
    /// Max-reduce — same shape contract as [`SumReduce`].
    MaxReduce,
    /// Min-reduce — same shape contract as [`SumReduce`].
    MinReduce,
    /// Arithmetic-mean reduce — same shape contract as
    /// [`SumReduce`]; divides the sum by the product of reduced
    /// dim sizes.
    MeanReduce,

    /// Convert input bytes from one dtype to another. The target
    /// dtype lives on the output Storage's `dtype` field (and on
    /// the Node's dtype); the source dtype is read from the input
    /// Storage at runtime by the wrapper, which dispatches to the
    /// right typed conversion kernel internally. Binding-table
    /// lookup is keyed on the *target* dtype (= the Node's dtype).
    Cast,
    /// 2D convolution forward pass. Shape + geometry flow through
    /// [`OpParams::Conv2D`](super::dispatch::OpKind). Two-input
    /// case: `(x, weight)`; three-input case adds `bias`. Output
    /// shape `[N, Cout, Hout, Wout]`.
    Conv2D,
    /// 2D transposed convolution. Shape + geometry flow through
    /// `OpParams::ConvTranspose2D`. Inputs and bias mirror Conv2D,
    /// but weight has transposed channel order (`[Cin, Cout/groups,
    /// Kh, Kw]`) and the variant carries `output_padding`.
    ConvTranspose2D,
    /// Sum-reduce a tensor to a smaller broadcast-compatible shape.
    /// The output rank may be lower than the input's; in that case
    /// the output shape is implicitly left-padded with 1s for axis
    /// alignment. Carried by `OpParams::ReduceSumTo` with both the
    /// input and target shapes.
    ReduceSumTo,
    /// Max-reduce a tensor to a smaller broadcast-compatible shape —
    /// the max-symmetric counterpart of `ReduceSumTo`. Same axis
    /// alignment rules; per-axis reduction is `max` instead of `+`.
    /// Carried by `OpParams::ReduceMaxTo` with both shapes.
    ReduceMaxTo,
    /// Fused matmul + bias-add. Inputs `[a, b, bias]`, output
    /// `[..., M, N] = a @ b + bias[None..., :]`. Reuses
    /// `OpParams::Matmul` for shape (kernel inits accumulator with
    /// the bias element instead of zero).
    FusedLinear,
    /// Multi-head scaled-dot-product attention (the math definition,
    /// not a tiled FlashAttention-2 kernel — naive on CPU). Inputs
    /// `[q, k, v, optional alibi_slopes]`. Geometry, softmax_scale,
    /// causal, window, softcap all flow through `OpParams::FlashAttn`.
    FlashAttn,
    /// Backward of [`FlashAttn`]: produces dQ from `(q, k, v, do, [alibi])`.
    /// Reuses [`OpParams::FlashAttn`] (same shape/geometry/causal flags
    /// as the forward — the recompute pass needs every forward parameter).
    /// Output shape == q shape. The dK and dV gradients are emitted as
    /// separate [`FlashAttnBackwardK`] / [`FlashAttnBackwardV`] nodes
    /// against the same inputs; CPU backends recompute the softmax
    /// state independently per call. A 3-output fused variant would
    /// share the recompute but needs multi-output infrastructure that
    /// doesn't exist yet.
    FlashAttnBackwardQ,
    /// Backward of [`FlashAttn`]: produces dK. See [`FlashAttnBackwardQ`].
    /// Output shape == k shape.
    FlashAttnBackwardK,
    /// Backward of [`FlashAttn`]: produces dV. See [`FlashAttnBackwardQ`].
    /// Output shape == v shape.
    FlashAttnBackwardV,
    /// Paged-cache scaled-dot-product attention. Inputs `[q, k_cache,
    /// v_cache, block_table, context_lens, optional alibi_slopes]`.
    /// Geometry + scale + softcap flow through `OpParams::PagedAttn`.
    PagedAttn,
    /// Affine transformation `y = mul * x + add` with scalar
    /// coefficients. `Op::AddScalar(c)` maps as `mul=1, add=c`;
    /// `Op::MulScalar(c)` maps as `mul=c, add=0`.
    Affine,
    /// Element-wise clamp: `y = clamp(x, min, max)`.
    ClampElementwise,
    /// Element-wise integer power: `y = x.powi(exp)`.
    PowIElementwise,
    /// Backward of [`PowIElementwise`]: `(x, upstream) → grad_x = exp ·
    /// x^(exp-1) · upstream`. Two inputs; carries the same `exp: i32`
    /// in `OpParams::PowI` as the forward. Single-launch alternative
    /// to the autograd primitive decomposition (PowI(exp-1) →
    /// MulScalar → Mul, 3 nodes). Per-dtype.
    PowIElementwiseBackward,
    /// Element-wise tensor maximum: `y[i] = max(lhs[i], rhs[i])`.
    MaximumElementwise,
    /// Element-wise tensor minimum: `y[i] = min(lhs[i], rhs[i])`.
    MinimumElementwise,
    /// Element-wise equality `a == b`. Both inputs share dtype `T`;
    /// output is a `U8` mask (`1` where equal, `0` otherwise). The
    /// binding-table dtype list is keyed `[T, T, U8]`, mirroring the
    /// ArgMax convention of carrying the output dtype explicitly when
    /// it differs from the inputs.
    EqualElementwise,
    /// Element-wise inequality `a != b`. Same shape contract as
    /// [`EqualElementwise`]; output `U8` mask (`1` where unequal,
    /// `0` otherwise). NaN follows IEEE-754: `NaN != NaN` is true,
    /// so `ne` returns `1` on NaN-vs-NaN positions.
    NotEqualElementwise,
    /// Element-wise strictly-less `a < b`. Same shape contract as
    /// [`EqualElementwise`]; output `U8` mask. NaN-on-either-side is
    /// always false (IEEE-754 unordered comparison).
    LessElementwise,
    /// Element-wise less-or-equal `a <= b`. Same shape contract.
    /// NaN comparisons are unordered → `0`.
    LessEqualElementwise,
    /// Element-wise strictly-greater `a > b`. Same shape contract.
    /// NaN comparisons are unordered → `0`.
    GreaterElementwise,
    /// Element-wise greater-or-equal `a >= b`. Same shape contract.
    /// NaN comparisons are unordered → `0`.
    GreaterEqualElementwise,
    /// Ternary select: `out[i] = if cond[i] != 0 { a[i] } else { b[i] }`.
    /// Inputs `(cond, a, b)` — `cond` is `U8`, `a` and `b` share dtype
    /// `T` and shape with `cond`. Output is `T` shape `cond.shape()`.
    /// Binding-table dtype list `[U8, T, T, T]`.
    Where,
    /// Element-wise floor (`⌊x⌋`). Same dtype as input. Backward is
    /// the zero distribution almost everywhere; gradient through
    /// rounding ops is dropped silently.
    FloorElementwise,
    /// Element-wise ceiling (`⌈x⌉`). Same dtype as input. Backward
    /// drops gradient (mirrors Floor).
    CeilElementwise,
    /// Element-wise round-to-nearest with **banker's rounding**
    /// (round-half-to-even, IEEE 754 roundeven). Backward drops
    /// gradient (mirrors Floor/Ceil).
    RoundElementwise,
    /// Element-wise sign (`-1` if negative, `0` if zero, `1` if
    /// positive). Same dtype as input. Backward drops gradient
    /// (zero almost everywhere; the subgradient at 0 is taken as 0).
    SignElementwise,
    /// Element-wise Gauss error function (`erf(x)`). Same dtype as
    /// input. Backward: `d/dx erf(x) = (2/√π) * exp(-x²)` —
    /// expressed via `Sqr` + `Neg` + `Exp` + `MulScalar`.
    ErfElementwise,
    /// Element-wise GELU activation, **exact erf formulation**:
    /// `0.5 * x * (1 + erf(x/√2))`. Distinct from
    /// [`GeluElementwise`] (tanh approximation). Same dtype as input.
    /// Backward decomposes into the standard-normal CDF + `x * φ(x)`
    /// (PDF) chain via existing primitives.
    GeluErfElementwise,
    /// Element-wise binary power: `out[i] = pow(a[i], b[i])`. Both
    /// inputs share dtype `T` and shape; output is `T` with the same
    /// shape. Distinct from [`PowIElementwise`] (scalar `i32`
    /// exponent). NaN follows IEEE-754 (e.g. `pow(-2, 0.5) = NaN`).
    PowElementwise,
    /// Element-wise reciprocal square root: `out[i] = 1 / sqrt(x[i])`.
    /// Single op (vs Sqrt + Recip — combining loses precision and
    /// doubles kernel launches). Critical for RMSNorm (`x / sqrt(mean(x²)+eps)`
    /// → `x * rsqrt(mean(x²)+eps)`).
    RsqrtElementwise,
    /// Element-wise remainder, **PyTorch convention**:
    /// `out[i] = a[i] - floor(a[i] / b[i]) * b[i]`. The sign of the
    /// result matches the sign of the divisor (differs from C99 `fmod`
    /// which has the sign of the dividend; `f32::%` is also fmod-style).
    /// Same dtype + shape on both inputs.
    RemElementwise,
    /// Reverse the order of elements along one dim. Dtype-agnostic
    /// at the byte level. Materializing op (Layout strides are
    /// unsigned; the negative-stride view path requires a wider
    /// stride representation that's a separate scope).
    Flip,
    /// Cyclic shift along one dim by a signed `shift`. Wraps
    /// (e.g. `roll([0,1,2], shift=1) = [2,0,1]`). Materializing op,
    /// dtype-agnostic.
    Roll,
    /// Running cumulative sum along one dim:
    /// `out[..., i, ...] = sum(in[..., 0..=i, ...])`. Same shape as
    /// input. Per-dtype kernel (sum needs typed addition).
    CumSum,
    /// Pad along one dim with `before`/`after` extra slots. Modes:
    /// Constant (fill with a value), Reflect (mirror edges),
    /// Replicate (repeat edges). Only Constant is implemented in
    /// the v1 cut; Reflect/Replicate fall through to a clean error.
    Pad,
    /// Upper-triangular mask along the last two dims. Materializing,
    /// dtype-agnostic at the byte level (output is x or zero per
    /// position; just selects bytes from src or the zero-init buffer).
    Triu,
    /// Lower-triangular mask along the last two dims (mirror of [`Triu`]).
    Tril,
    /// Numerically-stable log-softmax along the last dim. Per-dtype
    /// (uses log/exp). Output shape == input shape.
    LogSoftmaxLastDim,
    /// Backward of [`LogSoftmaxLastDim`]: takes `(forward_output, upstream)`
    /// and produces the input gradient. Per-dtype.
    LogSoftmaxLastDimBackward,
    /// MaskedFill: fill positions where mask is nonzero with a scalar.
    /// Inputs `(x, mask)` where mask is U8. Per-dtype on x; output
    /// shape == x shape.
    MaskedFill,
    /// Backward helper for `Pad`. Per-dtype since accumulation is
    /// typed addition. Routes all 3 modes uniformly — the kernel
    /// switches on `mode_tag`.
    PadBackward,
    /// Concatenate N inputs along one dim. Inputs must agree on
    /// every dim except the concat dim; output's concat-dim size
    /// is the sum of inputs' concat-dim sizes.
    Concat,
    /// Softmax along the last dim, numerically stable
    /// (subtract per-row max, exp, divide by sum).
    SoftmaxLastDim,
    /// Backward of [`SoftmaxLastDim`]: `(y, g) → y · (g - sum(y · g, last))`.
    /// Per-dtype; output shape == y shape.
    SoftmaxLastDimBackward,
    /// RMS normalization along the last dim, no affine params:
    /// `y = x / sqrt(mean(x², last) + eps)`.
    RmsNormLastDim,
    /// Backward of [`RmsNormLastDim`]: `(x, g_y) → grad_x` per the
    /// closed-form formula in
    /// `fuel-reference-backend::ops::rms_norm_last_dim_backward`.
    /// Per-dtype + eps; output shape == x shape.
    RmsNormLastDimBackward,
    /// Layer normalization along the last dim, no affine params:
    /// `y = (x - mean(x)) / sqrt(var(x) + eps)`.
    LayerNormLastDim,
    /// Backward of [`LayerNormLastDim`]: `(x, g) → grad_x` per the
    /// canonical formula. Per-dtype + eps; output shape == x shape.
    LayerNormLastDimBackward,
    /// Backward of [`ReduceMaxTo`]: `(x, upstream) → grad_x` of x's
    /// shape. Routes upstream to argmax positions; ties split
    /// equally. Per-dtype; carries shape pair via
    /// [`OpParams::ReduceMaxToBackward`].
    ReduceMaxToBackward,
    /// Pick slices from a source tensor along `dim` using a rank-1
    /// U32 index tensor. Output's `dim` size = number of indices.
    IndexSelect,
    /// N-dimensional gather along `dim`. Source and indices have
    /// the same rank; output shape equals indices.shape(). For
    /// each output position, source is read at the same multi-index
    /// except `dim`'s coord is read from the indices tensor.
    Gather,
    /// Fused rotary position embedding. Inputs `(x, cos, sin)`;
    /// `cos`/`sin` broadcast across leading dims. Used by every
    /// modern transformer's attention layer.
    Rope,
    /// Add `src` values into a copy of `base` at positions given
    /// by a rank-1 U32 `indices` tensor along `dim`. Inputs:
    /// `(base, indices, src)`. Output shape == base shape.
    IndexAdd,
    /// N-dimensional scatter-add: the functional inverse of
    /// `Gather`. Inputs `(base, indices, src)`; indices and src
    /// share the same shape; for each position `p`, the source
    /// value is added into base at the multi-index where `dim`'s
    /// coord is `indices[p]`.
    ScatterAdd,
    /// Argmax along one dim — produces a U32 index tensor with
    /// `dim` removed from the output shape.
    ArgMaxDim,
    /// Argmin along one dim — same shape contract as
    /// [`ArgMaxDim`].
    ArgMinDim,
    /// Quantized matmul: `C = A @ dequant(W_Q)`. Activations are
    /// f32 (or eventually bf16); weights are a U32-typed byte
    /// stream of quantized blocks. The quant format is carried in
    /// [`OpParams::QMatMul`](super::dispatch::OpKind).
    QMatMul,
    /// In-place rectangular scatter write: copies source's bytes
    /// into a slab of destination defined per-axis by `ranges`.
    /// Backs Phase E.3.2's persistent KV-cache writes
    /// (`InferenceContext` + `KvCache`). See
    /// [`Op::WriteSlice`](fuel_graph::Op::WriteSlice) for the IR
    /// contract and [`OpParams::WriteSlice`](super::dispatch::OpKind)
    /// for the kernel-side params.
    WriteSlice,
    /// In-place ring-buffer scatter write: copies source's bytes into
    /// destination at a slab that wraps modulo `modulus` on the
    /// rotating axis. The write position comes from a dynamic input
    /// (rank-0 U32). Backs sliding-window KV caches (Mistral /
    /// Phi-3 sliding-window / sliding-window Qwen). See
    /// [`Op::WriteSliceRotating`](fuel_graph::Op::WriteSliceRotating)
    /// for the IR contract.
    WriteSliceRotating,
    /// Cross-device copy: produce a fresh tensor on the target
    /// device, copying bytes from the input's residency. Backs
    /// [`Op::Copy`](fuel_graph::Op::Copy) for the bridge-retirement
    /// trajectory's Phase 2 (D2H through the binding table).
    ///
    /// Binding-table key shape `[T, T]` (input dtype, output dtype —
    /// always equal). The `BackendId` axis encodes the **source**
    /// backend (where the kernel runs — the source owns its own
    /// download path); the executor allocates the output Storage on
    /// the `Op::Copy { target }` location field via a dedicated
    /// `WorkItemKind::Copy` arm.
    Copy,
    // ---- In-place ops (Phase 3 of the in-place ops infrastructure;
    // ---- see docs/session-prompts/in-place-ops-infrastructure.md) ----
    //
    // Each in-place OpKind mirrors a non-inplace functional cousin.
    // The dispatch difference is structural rather than computational:
    // the executor adopts the target's Storage Arc as the output slot
    // instead of allocating fresh bytes. The kernel reads + writes the
    // same buffer through a single write lock. Binding-table key
    // shape `[T]` (the in-place op's single input dtype = output
    // dtype). `KernelCaps::strided_input` semantics apply identically
    // to the non-inplace cousin (same-pointer dispatch handles strided
    // inputs as long as the stride pattern doesn't produce overlapping
    // writes; the unary kernels write index `i` after reading index `i`
    // so no aliasing issues).
    /// In-place [`OpKind::ReluElementwise`] — `x = max(0, x)`.
    ReluInplace,
    /// In-place [`OpKind::SiluElementwise`] — `x = x · sigmoid(x)`.
    SiluInplace,
    /// In-place [`OpKind::GeluElementwise`] (tanh approximation) —
    /// `x = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))`.
    GeluInplace,
    /// In-place [`OpKind::TanhElementwise`] — `x = tanh(x)`.
    TanhInplace,
    /// In-place [`OpKind::SigmoidElementwise`] — `x = 1 / (1 + exp(-x))`.
    SigmoidInplace,
    /// In-place [`OpKind::NegElementwise`] — `x = -x`.
    NegInplace,
    /// In-place [`OpKind::AbsElementwise`] — `x = |x|`.
    AbsInplace,
    /// In-place [`OpKind::SqrElementwise`] — `x = x²`.
    SqrInplace,
    /// In-place [`OpKind::SqrtElementwise`] — `x = √x`.
    SqrtInplace,
    /// In-place [`OpKind::RsqrtElementwise`] — `x = 1/√x`.
    RsqrtInplace,
    /// In-place [`OpKind::RecipElementwise`] — `x = 1/x`.
    RecipInplace,
    /// In-place [`OpKind::ExpElementwise`] — `x = exp(x)`.
    ExpInplace,
    /// In-place [`OpKind::LogElementwise`] — `x = ln(x)`.
    LogInplace,
    /// In-place [`OpKind::SinElementwise`] — `x = sin(x)`.
    SinInplace,
    /// In-place [`OpKind::CosElementwise`] — `x = cos(x)`.
    CosInplace,
    /// In-place [`OpKind::SignElementwise`] — `x = sign(x)`.
    SignInplace,
    /// In-place [`OpKind::FloorElementwise`] — `x = ⌊x⌋`.
    FloorInplace,
    /// In-place [`OpKind::CeilElementwise`] — `x = ⌈x⌉`.
    CeilInplace,
    /// In-place [`OpKind::RoundElementwise`] — `x = round(x)`.
    RoundInplace,
    /// In-place [`OpKind::ErfElementwise`] — `x = erf(x)`.
    ErfInplace,
    /// In-place [`OpKind::GeluErfElementwise`] — exact-GeLU
    /// `x = 0.5 · x · (1 + erf(x/√2))`.
    GeluErfInplace,
    /// In-place [`OpKind::ClampElementwise`] — `x = clamp(x, min, max)`.
    /// Scalar `(min, max)` flow through [`OpParams::Clamp`].
    ClampInplace,
    /// In-place [`OpKind::PowIElementwise`] — `x = x.powi(exp)`.
    /// Scalar `exp` flows through [`OpParams::PowI`].
    PowIInplace,
    /// In-place [`OpKind::Affine`] — `x = mul · x + add`. The
    /// `(mul, add)` coefficients flow through
    /// [`OpParams::Affine`]; the kernel reads + writes the same
    /// buffer. Single-input, single-output. Backs
    /// [`FusedOps::INPLACE_AFFINE`](fuel_graph::registry::FusedOps::INPLACE_AFFINE).
    InplaceAffine,
    /// Fused softmax + negative log-likelihood with integer class
    /// targets — the standard PyTorch / Liger-Kernel training loss.
    /// Inputs `[logits, targets]` where `logits: [n_rows, vocab]`
    /// (F32, flattened from rank-N) and `targets: [n_rows]` (I64).
    /// Output F32 — scalar for Mean/Sum reductions, `[n_rows]` for
    /// None. Geometry + reduction + `ignore_index` flow through
    /// `OpParams::FusedSoftmaxCrossEntropy`. Backs
    /// [`FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY`](fuel_graph::registry::FusedOps::FUSED_SOFTMAX_CROSS_ENTROPY).
    FusedSoftmaxCrossEntropy,
    /// Depthwise 1-D causal convolution + bias + optional fused SiLU.
    /// Inputs `[x, weight, bias]` where
    /// `x: [batch, channels, seq + kernel - 1]` (left-pre-padded by
    /// caller), `weight: [channels, 1, kernel]`, `bias: [channels]`.
    /// Output `[batch, channels, seq]`. Geometry + `use_silu` flow
    /// through `OpParams::CausalConv1d`. Backs
    /// [`FusedOps::CAUSAL_CONV1D`](fuel_graph::registry::FusedOps::CAUSAL_CONV1D)
    /// — the Mamba-1 / Mamba-2 prefill convolution fusion.
    CausalConv1d,
    /// Mamba-1's selective state-space scan (forward). Five inputs
    /// `[u, delta, a, b, c]` — see
    /// [`FusedOps::SELECTIVE_SCAN`](fuel_graph::registry::FusedOps::SELECTIVE_SCAN)
    /// for the full shape contract. Output `y: [batch, seqlen, dim]`.
    /// Geometry + `delta_softplus` flow through
    /// `OpParams::SelectiveScan`.
    SelectiveScan,
    /// Mamba-2's State-Space Duality chunked scan (forward). Five
    /// inputs `[x, dt, a, b, c]` — see
    /// [`FusedOps::SSD_CHUNK_SCAN`](fuel_graph::registry::FusedOps::SSD_CHUNK_SCAN)
    /// for the full shape contract. Output
    /// `y: [batch, seqlen, heads, head_dim]`. Geometry + `chunk_size`
    /// flow through `OpParams::SsdChunkScan`.
    SsdChunkScan,
    /// bitsandbytes-style 4-bit NormalFloat quantized matmul. Three
    /// inputs `[activations, w_packed, absmax]` — see
    /// [`FusedOps::NF4_MATMUL`](fuel_graph::registry::FusedOps::NF4_MATMUL)
    /// for the full shape contract. Output `[..., M, N]` matches the
    /// activations' dtype. Geometry + `block_size` flow through
    /// `OpParams::Nf4Matmul`.
    Nf4Matmul,
}

impl OpKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OpKind::MatMul           => "matmul",
            OpKind::AddElementwise   => "add",
            OpKind::SubElementwise   => "sub",
            OpKind::MulElementwise   => "mul",
            OpKind::DivElementwise   => "div",
            OpKind::ReluElementwise  => "relu",
            OpKind::NegElementwise   => "neg",
            OpKind::SqrElementwise   => "sqr",
            OpKind::SqrtElementwise  => "sqrt",
            OpKind::RecipElementwise => "recip",
            OpKind::AbsElementwise   => "abs",
            OpKind::TanhElementwise  => "tanh",
            OpKind::ExpElementwise    => "exp",
            OpKind::LogElementwise    => "log",
            OpKind::SinElementwise    => "sin",
            OpKind::CosElementwise    => "cos",
            OpKind::SigmoidElementwise => "sigmoid",
            OpKind::SiluElementwise   => "silu",
            OpKind::GeluElementwise   => "gelu",
            OpKind::StepElementwise   => "step",
            OpKind::SumReduce         => "sum_reduce",
            OpKind::MaxReduce         => "max_reduce",
            OpKind::MinReduce         => "min_reduce",
            OpKind::MeanReduce        => "mean_reduce",
            OpKind::Cast              => "cast",
            OpKind::Conv2D            => "conv2d",
            OpKind::ConvTranspose2D   => "conv_transpose2d",
            OpKind::ReduceSumTo       => "reduce_sum_to",
            OpKind::ReduceMaxTo       => "reduce_max_to",
            OpKind::FusedLinear       => "fused_linear",
            OpKind::FlashAttn         => "flash_attn",
            OpKind::FlashAttnBackwardQ => "flash_attn_backward_q",
            OpKind::FlashAttnBackwardK => "flash_attn_backward_k",
            OpKind::FlashAttnBackwardV => "flash_attn_backward_v",
            OpKind::PagedAttn         => "paged_attn",
            OpKind::Affine            => "affine",
            OpKind::ClampElementwise  => "clamp",
            OpKind::PowIElementwise   => "powi",
            OpKind::PowIElementwiseBackward => "powi_backward",
            OpKind::MaximumElementwise => "maximum",
            OpKind::MinimumElementwise => "minimum",
            OpKind::EqualElementwise   => "eq",
            OpKind::NotEqualElementwise => "ne",
            OpKind::LessElementwise    => "lt",
            OpKind::LessEqualElementwise => "le",
            OpKind::GreaterElementwise => "gt",
            OpKind::GreaterEqualElementwise => "ge",
            OpKind::Where              => "where",
            OpKind::FloorElementwise   => "floor",
            OpKind::CeilElementwise    => "ceil",
            OpKind::RoundElementwise   => "round",
            OpKind::SignElementwise    => "sign",
            OpKind::ErfElementwise     => "erf",
            OpKind::GeluErfElementwise => "gelu_erf",
            OpKind::PowElementwise     => "pow",
            OpKind::RsqrtElementwise   => "rsqrt",
            OpKind::RemElementwise     => "rem",
            OpKind::Flip               => "flip",
            OpKind::Roll               => "roll",
            OpKind::CumSum             => "cumsum",
            OpKind::Pad                => "pad",
            OpKind::PadBackward        => "pad_backward",
            OpKind::Triu               => "triu",
            OpKind::Tril               => "tril",
            OpKind::LogSoftmaxLastDim  => "log_softmax_last_dim",
            OpKind::LogSoftmaxLastDimBackward
                                       => "log_softmax_last_dim_backward",
            OpKind::MaskedFill         => "masked_fill",
            OpKind::Concat            => "concat",
            OpKind::SoftmaxLastDim    => "softmax_last_dim",
            OpKind::SoftmaxLastDimBackward => "softmax_last_dim_backward",
            OpKind::RmsNormLastDim    => "rms_norm_last_dim",
            OpKind::RmsNormLastDimBackward => "rms_norm_last_dim_backward",
            OpKind::LayerNormLastDim  => "layer_norm_last_dim",
            OpKind::LayerNormLastDimBackward => "layer_norm_last_dim_backward",
            OpKind::ReduceMaxToBackward => "reduce_max_to_backward",
            OpKind::IndexSelect       => "index_select",
            OpKind::Gather            => "gather",
            OpKind::Rope              => "rope",
            OpKind::IndexAdd          => "index_add",
            OpKind::ScatterAdd        => "scatter_add",
            OpKind::ArgMaxDim         => "argmax_dim",
            OpKind::ArgMinDim         => "argmin_dim",
            OpKind::QMatMul           => "qmatmul",
            OpKind::WriteSlice        => "write_slice",
            OpKind::WriteSliceRotating => "write_slice_rotating",
            OpKind::Copy              => "copy",
            OpKind::ReluInplace       => "relu_inplace",
            OpKind::SiluInplace       => "silu_inplace",
            OpKind::GeluInplace       => "gelu_inplace",
            OpKind::TanhInplace       => "tanh_inplace",
            OpKind::SigmoidInplace    => "sigmoid_inplace",
            OpKind::NegInplace        => "neg_inplace",
            OpKind::AbsInplace        => "abs_inplace",
            OpKind::SqrInplace        => "sqr_inplace",
            OpKind::SqrtInplace       => "sqrt_inplace",
            OpKind::RsqrtInplace      => "rsqrt_inplace",
            OpKind::RecipInplace      => "recip_inplace",
            OpKind::ExpInplace        => "exp_inplace",
            OpKind::LogInplace        => "log_inplace",
            OpKind::SinInplace        => "sin_inplace",
            OpKind::CosInplace        => "cos_inplace",
            OpKind::SignInplace       => "sign_inplace",
            OpKind::FloorInplace      => "floor_inplace",
            OpKind::CeilInplace       => "ceil_inplace",
            OpKind::RoundInplace      => "round_inplace",
            OpKind::ErfInplace        => "erf_inplace",
            OpKind::GeluErfInplace    => "gelu_erf_inplace",
            OpKind::ClampInplace      => "clamp_inplace",
            OpKind::PowIInplace       => "powi_inplace",
            OpKind::InplaceAffine     => "inplace_affine",
            OpKind::FusedSoftmaxCrossEntropy => "fused_softmax_cross_entropy",
            OpKind::CausalConv1d        => "causal_conv1d",
            OpKind::SelectiveScan       => "selective_scan",
            OpKind::SsdChunkScan        => "ssd_chunk_scan",
            OpKind::Nf4Matmul           => "nf4_matmul",
        }
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dispatch size bucket. One `u32` field carries two representations,
/// disambiguated by the [`OpKind`] that always sits beside it in the
/// dispatch key (a matmul key is only ever compared to matmul keys):
///
/// - **Total-elements key** (every non-matmul op — elementwise,
///   reduce, cast, …): the value is `log2(total_elements)`. A 256×256
///   elementwise tensor (65,536 elements) → `16`; a 1024×1024
///   (1,048,576) → `20`. Built by [`SizeClass::from_elem_count`].
///   Two shapes that round to the same bucket share a profile entry.
///
/// - **MatMul aspect key**: matmul cost is aspect-dependent — a
///   `[1,K]×[K,N]` GEMV is bandwidth-bound, a square `[S,S]×[S,S]`
///   GEMM of the *same output size* is compute-bound — so
///   `log2(total_elements)` alone can't tell them apart. A matmul
///   packs all three dims: `(log2(m·n) << 16) | (log2(m) << 8) |
///   log2(k)`, read from `lhs=[…,m,k]` / `rhs=[…,k,n]`. `log2(m·n)`
///   occupies the high byte so [`DispatchTable::pick_nearest`]'s
///   magnitude ordering still tracks output size, while the low bytes
///   distinguish aspect (a GEMV and a same-output-size square never
///   collide, because their `log2(m)` bytes differ). Built by
///   [`SizeClass::matmul`].
///
/// **Producer/consumer single-sourcing.** The `fuel-core` Judge
/// (producer) and every consumer (the `fuel-dispatch` ranker at
/// realize time) derive the key through the SAME helpers —
/// [`SizeClass::matmul`] from native `(m,n,k)` on the producer side,
/// [`SizeClass::for_op`] from the operand shapes on the consumer side.
/// A given matmul shape therefore maps to one identical key on both
/// sides, so a cell the Judge profiles is found by the ranker at
/// dispatch time (the bug this repr fixes: the old scalar key had the
/// producer on `m·n` and the consumer on `m·k`, so non-square matmul
/// lookups always missed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SizeClass(pub u32);

impl SizeClass {
    /// Bucket a raw element count as `log2(n)` — the total-elements key
    /// used by every non-matmul op. `n == 0` clamps to `1` (→ `0`).
    pub fn from_elem_count(n: usize) -> Self {
        let n = n.max(1);
        let log2 = 63 - (n as u64).leading_zeros();
        SizeClass(log2)
    }

    /// Aspect-carrying dispatch key for a matmul `[m,k] × [k,n]`.
    ///
    /// THE single source of matmul-key derivation: both the Judge
    /// (producer, from its native `(m,n,k)`) and the ranker (consumer,
    /// via [`SizeClass::for_op`] from operand shapes) call this, so both
    /// sides agree for every shape. Packs `(log2(m·n) << 16) |
    /// (log2(m) << 8) | log2(k)`; each `log2` fits a byte (≤ 63 for any
    /// `usize`). A `[1,K]×[K,N]` GEMV (`m = 1`) never collides with a
    /// square GEMM of equal output size — the `log2(m)` byte differs.
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        let mn = Self::from_elem_count(m.saturating_mul(n)).0 & 0xff;
        let ml = Self::from_elem_count(m).0 & 0xff;
        let kl = Self::from_elem_count(k).0 & 0xff;
        SizeClass((mn << 16) | (ml << 8) | kl)
    }

    /// Derive the dispatch key for `op` from its input operand shapes —
    /// the single entry point every consumer uses. A [`OpKind::MatMul`]
    /// reads `(m,n,k)` from `lhs=[…,m,k]` / `rhs=[…,k,n]` (trailing two
    /// dims; leading batch dims don't affect the key) and keys via
    /// [`SizeClass::matmul`]; every other op keys on the first operand's
    /// total element count via [`SizeClass::from_elem_count`]. Empty
    /// shapes (nullary op) → `SizeClass(0)`.
    pub fn for_op(op: OpKind, input_shapes: &[Shape]) -> Self {
        if op == OpKind::MatMul {
            if let (Some(lhs), Some(rhs)) = (input_shapes.first(), input_shapes.get(1)) {
                let (ld, rd) = (lhs.dims(), rhs.dims());
                if ld.len() >= 2 && rd.len() >= 2 {
                    let m = ld[ld.len() - 2];
                    let k = ld[ld.len() - 1];
                    let n = rd[rd.len() - 1];
                    return Self::matmul(m, n, k);
                }
            }
        }
        match input_shapes.first() {
            Some(s) => Self::from_elem_count(s.elem_count()),
            None => SizeClass(0),
        }
    }
}

/// Single (op_kind, dtype, size_class) × (backend, device_index,
/// kernel_source) datum produced by one measurement run.
///
/// **Per-alternative measurement (2026-06-08)**: `kernel_source`
/// distinguishes kernels that register at the same
/// `(op, dtypes, backend)` decision point in the binding table —
/// AOCL vs MKL vs portable-cpu, cuBLAS vs CUTLASS, etc. Each
/// sibling alternative produces its own [`ProfileEntry`]; the
/// dispatch table picks the best entry per cell across both
/// backends and kernel_sources. Default empty string preserves the
/// pre-v2 shape for kernels that don't need to distinguish.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProfileEntry {
    pub op:            OpKind,
    pub dtype:         DType,
    pub size_class:    SizeClass,
    pub backend:       BackendId,
    pub device_index:  u32,
    /// Median wall-clock time per invocation over `iterations`.
    pub latency_ns:    u64,
    /// Number of timed iterations that produced `latency_ns`.
    pub iterations:    u32,
    /// Max relative element-wise error vs the reference backend's
    /// output on the same input.
    pub max_rel_error: f32,
    /// Diagnostic tag identifying which kernel sibling produced this
    /// measurement when multiple alternatives register at the same
    /// `(op, dtypes, backend)` binding-table key. Mirrors
    /// `BindingEntry::kernel_source` (in `fuel-dispatch`). `""` is
    /// the pre-v2 default for kernels that don't need to distinguish;
    /// conventional values include `"portable-cpu"`, `"aocl"`,
    /// `"mkl"`, `"cublas"`, `"cutlass"`.
    ///
    /// Owned `String` (not `&'static str`) so the field round-trips
    /// through serde — deserialized reports can't carry borrowed
    /// data with a `'static` lifetime. The producer-side `Judge`
    /// converts from `BindingEntry::kernel_source: &'static str`
    /// via `.to_string()` at measurement time.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kernel_source: String,
}

/// A persistable table of every profile measurement the Judge
/// produced in one run.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProfileReport {
    pub version: u32,
    pub entries: Vec<ProfileEntry>,
}

#[cfg(feature = "serde")]
impl ProfileReport {
    /// Atomic write to `path` as JSON (sibling `.tmp` + rename).
    pub fn save(&self, path: &std::path::Path) -> crate::Result<()> {
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| crate::Error::Msg(format!("judge: JSON encode failed: {e}")))?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &json)
            .map_err(|e| crate::Error::Msg(format!("judge: write {tmp:?} failed: {e}")))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| crate::Error::Msg(format!("judge: rename {tmp:?} → {path:?} failed: {e}")))?;
        Ok(())
    }

    /// Load a previously-persisted report. Returns `Ok(None)` on a
    /// missing file or schema-version mismatch (both are "cache miss,
    /// re-run the Judge" signals).
    pub fn load(path: &std::path::Path) -> crate::Result<Option<Self>> {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(crate::Error::Msg(format!("judge: read {path:?} failed: {e}"))),
        };
        let report: Self = serde_json::from_slice(&bytes)
            .map_err(|e| crate::Error::Msg(format!("judge: parse {path:?} failed: {e}")))?;
        if report.version != PROFILE_REPORT_VERSION {
            return Ok(None);
        }
        Ok(Some(report))
    }
}

/// A selection criterion — what "best" means for a lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Criterion {
    /// Lowest median latency.
    Fastest,
    /// Lowest max relative error vs the reference backend.
    MostAccurate,
    /// Weighted blend — lower is better.
    Balanced,
}

impl Criterion {
    pub fn as_str(self) -> &'static str {
        match self {
            Criterion::Fastest      => "fastest",
            Criterion::MostAccurate => "accurate",
            Criterion::Balanced     => "balanced",
        }
    }
}

impl std::fmt::Display for Criterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default weight applied to `max_rel_error` in the Balanced
/// criterion's cost function.
pub const DEFAULT_ACCURACY_PENALTY: f64 = 100.0;

/// Lookup key into [`DispatchTable`]. Combines the per-op axes
/// (what + on what data + how big) with the user's criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DispatchKey {
    op:         OpKind,
    dtype:      DType,
    size_class: SizeClass,
    criterion:  Criterion,
}

/// Where the dispatch table decided an op should run, and which
/// kernel sibling won at that decision point.
///
/// **Per-alternative measurement (2026-06-08)**: `kernel_source`
/// surfaces the winning [`ProfileEntry::kernel_source`] so
/// consumers (Router, planner, telemetry) can name the actual
/// kernel — critical when multiple alternatives register at
/// `(BackendId::Cpu)` (e.g. AOCL vs MKL vs portable-cpu) and the
/// (backend, device) pair alone is ambiguous. `""` for legacy
/// reports / single-impl cells.
///
/// `Pick` stays `Copy` (and `&'static str` is `Copy`); the picker
/// preserves the string identity by reading it directly from the
/// winning `ProfileEntry` — its `String` value is matched against
/// the well-known interned set via [`kernel_source_intern`] so the
/// `&'static str` is the same pointer the binding-table entry
/// carries.
///
/// `Pick` is only ever constructed at runtime by [`DispatchTable::rebuild_from`]
/// from in-memory [`ProfileEntry`] data — never deserialized from JSON
/// or other owned input. We therefore intentionally do NOT derive
/// `Deserialize` (the `&'static str` field can't be deserialized from
/// owned input). `Serialize` is also dropped for symmetry; the public
/// serialization surface is [`ProfileReport`] / [`ProfileEntry`], and
/// `Pick` is regenerated by calling [`DispatchTable::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pick {
    pub backend:      BackendId,
    pub device_index: u32,
    /// Diagnostic tag from the winning [`ProfileEntry::kernel_source`].
    /// `""` when no sibling distinguishes the cell.
    pub kernel_source: &'static str,
}

/// Intern a kernel-source string into a `&'static str` matching the
/// well-known conventional tags. Unknown tags fall back to `""`
/// (diagnostic-only field — losing an unknown tag is preferable to
/// leaking a heap allocation through `Box::leak`). The set mirrors
/// the documented convention on
/// [`BindingEntry::kernel_source`](../../fuel_dispatch/struct.BindingEntry.html#structfield.kernel_source).
///
/// Unknown tags trigger a `debug_assert!` panic in dev builds and
/// a `eprintln!` warning in release builds — silently dropping an
/// unfamiliar tag was previously masking router gaps (the router
/// can't resolve `(backend, device, "")` to the right sibling kernel
/// when multiple alternatives register at the same backend slot).
/// Add new tags here as new vendor backends land.
pub fn kernel_source_intern(s: &str) -> &'static str {
    match s {
        ""             => "",
        "portable-cpu" => "portable-cpu",
        "aocl"         => "aocl",
        "mkl"          => "mkl",
        "cublas"       => "cublas",
        "cutlass"      => "cutlass",
        "slang"        => "slang",
        tag            => {
            debug_assert!(
                false,
                "kernel_source_intern: unknown tag {tag:?}; map silently drops it. \
                 Add it to the known-set in fuel-core-types/src/dispatch.rs."
            );
            eprintln!(
                "warning: kernel_source_intern: unknown tag {tag:?} dropped to \"\"; \
                 router will not be able to disambiguate same-backend siblings."
            );
            ""
        },
    }
}

/// Options for building a [`DispatchTable`].
#[derive(Debug, Clone, Copy)]
pub struct DispatchOptions {
    pub accuracy_penalty:  f64,
}

impl Default for DispatchOptions {
    fn default() -> Self {
        Self { accuracy_penalty: DEFAULT_ACCURACY_PENALTY }
    }
}

impl DispatchOptions {
    pub fn with_balanced_penalty(mut self, k: f64) -> Self {
        self.accuracy_penalty = k;
        self
    }
}

/// O(1) runtime dispatch table, constructed once from a
/// [`ProfileReport`] and then queried at realize time.
#[derive(Debug, Clone)]
pub struct DispatchTable {
    entries: HashMap<DispatchKey, Pick>,
    /// All size classes present for each `(op, dtype)` — sorted
    /// ascending so `pick_nearest` can do a linear scan for the
    /// closest profiled bucket.
    size_index: HashMap<(OpKind, DType), Vec<SizeClass>>,
    accuracy_penalty: f64,
}

impl DispatchTable {
    /// Build a dispatch table from a profile report with default options.
    pub fn build(report: &ProfileReport) -> Self {
        Self::build_with(report, DispatchOptions::default())
    }

    pub fn build_with(report: &ProfileReport, opts: DispatchOptions) -> Self {
        let mut tbl = Self {
            entries:          HashMap::new(),
            size_index:       HashMap::new(),
            accuracy_penalty: opts.accuracy_penalty,
        };
        tbl.rebuild_from(report);
        tbl
    }

    fn rebuild_from(&mut self, report: &ProfileReport) {
        self.entries.clear();
        self.size_index.clear();
        let mut groups: HashMap<(OpKind, DType, SizeClass), Vec<&ProfileEntry>> = HashMap::new();
        for e in &report.entries {
            groups.entry((e.op, e.dtype, e.size_class)).or_default().push(e);
        }
        for ((op, dtype, size_class), group) in &groups {
            for &criterion in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
                if let Some(winner) = self.pick_winner(group, criterion) {
                    let key = DispatchKey { op: *op, dtype: *dtype, size_class: *size_class, criterion };
                    self.entries.insert(key, Pick {
                        backend:       winner.backend,
                        device_index:  winner.device_index,
                        kernel_source: kernel_source_intern(&winner.kernel_source),
                    });
                }
            }
            self.size_index.entry((*op, *dtype)).or_default().push(*size_class);
        }
        for classes in self.size_index.values_mut() {
            classes.sort_by_key(|s| s.0);
            classes.dedup();
        }
    }

    fn pick_winner<'a>(&self, group: &[&'a ProfileEntry], crit: Criterion) -> Option<&'a ProfileEntry> {
        match crit {
            Criterion::Fastest => group.iter().copied()
                .min_by_key(|e| e.latency_ns),
            Criterion::MostAccurate => group.iter().copied()
                .min_by(|a, b| {
                    a.max_rel_error.total_cmp(&b.max_rel_error)
                        .then(a.latency_ns.cmp(&b.latency_ns))
                }),
            Criterion::Balanced => group.iter().copied()
                .min_by(|a, b| {
                    let sa = a.latency_ns as f64 * (1.0 + self.accuracy_penalty * a.max_rel_error as f64);
                    let sb = b.latency_ns as f64 * (1.0 + self.accuracy_penalty * b.max_rel_error as f64);
                    sa.total_cmp(&sb)
                }),
        }
    }

    /// Exact lookup — returns `None` if the requested size class
    /// wasn't profiled.
    pub fn pick(&self, op: OpKind, dtype: DType, size_class: SizeClass, criterion: Criterion) -> Option<Pick> {
        self.entries.get(&DispatchKey { op, dtype, size_class, criterion }).copied()
    }

    /// Nearest-neighbour lookup. Ties go to the larger size class.
    pub fn pick_nearest(&self, op: OpKind, dtype: DType, size_class: SizeClass, criterion: Criterion) -> Option<Pick> {
        if let Some(p) = self.pick(op, dtype, size_class, criterion) {
            return Some(p);
        }
        let classes = self.size_index.get(&(op, dtype))?;
        if classes.is_empty() {
            return None;
        }
        let target = size_class.0 as i32;
        let nearest = classes.iter()
            .min_by_key(|c| {
                let diff = (c.0 as i32 - target).abs();
                (diff, -(c.0 as i32))
            })?;
        self.pick(op, dtype, *nearest, criterion)
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Every distinct `(op, dtype, size_class)` for which the table
    /// has at least one criterion entry. Sorted, stable across calls.
    pub fn keys(&self) -> Vec<(OpKind, DType, SizeClass)> {
        let mut seen: std::collections::HashSet<(OpKind, DType, SizeClass)> = Default::default();
        for k in self.entries.keys() {
            seen.insert((k.op, k.dtype, k.size_class));
        }
        let mut out: Vec<_> = seen.into_iter().collect();
        out.sort_by(|a, b| {
            a.0.as_str().cmp(b.0.as_str())
                .then(format!("{:?}", a.1).cmp(&format!("{:?}", b.1)))
                .then(a.2.0.cmp(&b.2.0))
        });
        out
    }
}

#[cfg(test)]
mod size_class_tests {
    use super::*;

    /// The shared matmul key derivation gives producer == consumer for a
    /// NON-SQUARE matmul (SizeClass v4, slice 2.5).
    ///
    /// **Born-red**: before v4 `SizeClass` was `log2(total_elements)` and
    /// matmul keyed on a single scalar — the Judge (producer) on `m·n`,
    /// the ranker (consumer) on `m·k` (LHS elem count). For a non-square
    /// matmul those disagree, so the ranker never found the profiled
    /// cell. The `matmul` / `for_op` helpers now derive one identical
    /// key on both sides.
    #[test]
    fn matmul_key_producer_equals_consumer_non_square() {
        // FFN-width decode GEMV `[1,2048]×[2048,5632]`: m=1, k=2048, n=5632.
        let (m, n, k) = (1usize, 5632usize, 2048usize);

        // Producer derives from its native (m, n, k).
        let producer = SizeClass::matmul(m, n, k);
        // Consumer derives from the operand shapes lhs=[m,k]/rhs=[k,n]
        // through the shared `for_op` entry point.
        let consumer = SizeClass::for_op(
            OpKind::MatMul,
            &[Shape::from_dims(&[m, k]), Shape::from_dims(&[k, n])],
        );
        assert_eq!(
            producer, consumer,
            "producer (native m,n,k) and consumer (operand shapes) must \
             derive one identical key",
        );

        // Old-bug witness: the pre-v4 scalar keys the two sides used
        // (m·n vs m·k) disagree for this non-square shape.
        assert_ne!(
            SizeClass::from_elem_count(m * n), // old producer key
            SizeClass::from_elem_count(m * k), // old consumer key
            "the non-square shape is exactly where the old scalar keys \
             diverged",
        );
    }

    /// Batched matmul keys the same as its per-head 2D shape — `for_op`
    /// reads only the trailing two dims, so a Judge that profiles the 2D
    /// per-head GEMV and a consumer costing the batched op agree.
    #[test]
    fn matmul_key_batched_ignores_leading_dims() {
        let per_head = SizeClass::for_op(
            OpKind::MatMul,
            &[Shape::from_dims(&[1, 64]), Shape::from_dims(&[64, 128])],
        );
        let batched = SizeClass::for_op(
            OpKind::MatMul,
            &[Shape::from_dims(&[32, 1, 64]), Shape::from_dims(&[32, 64, 128])],
        );
        assert_eq!(per_head, batched);
        assert_eq!(per_head, SizeClass::matmul(1, 128, 64));
    }

    /// A bandwidth-bound GEMV and a compute-bound square GEMM of the SAME
    /// output size key to DISTINCT buckets — the aspect distinction.
    /// Under the old scalar `log2(m·n)` key they collided.
    #[test]
    fn gemv_and_same_output_square_are_distinct() {
        // GEMV `[1,2048]×[2048,4096]` → output m·n = 4096.
        let gemv = SizeClass::matmul(1, 4096, 2048);
        // Square `[64,64]×[64,64]` → output m·n = 4096 — SAME total.
        let square = SizeClass::matmul(64, 64, 64);

        // Old scalar key: identical (both log2(4096) = 12) → collision.
        assert_eq!(
            SizeClass::from_elem_count(1 * 4096),
            SizeClass::from_elem_count(64 * 64),
            "sanity: equal output size → the OLD scalar keys collided",
        );
        // v4 aspect key: distinct (the log2(m) byte differs, 0 vs 6).
        assert_ne!(
            gemv, square,
            "a bandwidth-bound GEMV must not share a dispatch cell with a \
             compute-bound square of equal output size",
        );
    }

    /// Non-matmul ops are untouched: `for_op` keys on the first
    /// operand's total element count, identical to `from_elem_count`.
    #[test]
    fn non_matmul_keeps_total_elements_key() {
        let sc = SizeClass::for_op(OpKind::AddElementwise, &[Shape::from_dims(&[1024])]);
        assert_eq!(sc, SizeClass::from_elem_count(1024));
        assert_eq!(sc.0, 10); // log2(1024)
    }
}

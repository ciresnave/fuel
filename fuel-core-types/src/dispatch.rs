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
use crate::probe::BackendId;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Schema version for persisted profile reports. Bump when the
/// entry layout changes in a way that can't be covered by
/// `#[serde(default)]`.
pub const PROFILE_REPORT_VERSION: u32 = 1;

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
    /// Concatenate N inputs along one dim. Inputs must agree on
    /// every dim except the concat dim; output's concat-dim size
    /// is the sum of inputs' concat-dim sizes.
    Concat,
    /// Softmax along the last dim, numerically stable
    /// (subtract per-row max, exp, divide by sum).
    SoftmaxLastDim,
    /// RMS normalization along the last dim, no affine params:
    /// `y = x / sqrt(mean(x², last) + eps)`.
    RmsNormLastDim,
    /// Layer normalization along the last dim, no affine params:
    /// `y = (x - mean(x)) / sqrt(var(x) + eps)`.
    LayerNormLastDim,
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
            OpKind::PagedAttn         => "paged_attn",
            OpKind::Affine            => "affine",
            OpKind::ClampElementwise  => "clamp",
            OpKind::PowIElementwise   => "powi",
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
            OpKind::Concat            => "concat",
            OpKind::SoftmaxLastDim    => "softmax_last_dim",
            OpKind::RmsNormLastDim    => "rms_norm_last_dim",
            OpKind::LayerNormLastDim  => "layer_norm_last_dim",
            OpKind::IndexSelect       => "index_select",
            OpKind::Gather            => "gather",
            OpKind::Rope              => "rope",
            OpKind::IndexAdd          => "index_add",
            OpKind::ScatterAdd        => "scatter_add",
            OpKind::ArgMaxDim         => "argmax_dim",
            OpKind::ArgMinDim         => "argmin_dim",
            OpKind::QMatMul           => "qmatmul",
        }
    }
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Log2-bucketed total element count. A 256×256 matmul input has
/// 65,536 elements → `size_class = 16`; a 1024×1024 has 1,048,576 →
/// `size_class = 20`. Two shapes that round to the same size class
/// share a profile entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SizeClass(pub u8);

impl SizeClass {
    /// Bucket a raw element count. Saturates at `u8::MAX`.
    pub fn from_elem_count(n: usize) -> Self {
        let n = n.max(1);
        let log2 = 63 - (n as u64).leading_zeros() as u8;
        SizeClass(log2)
    }
}

/// Single (op_kind, dtype, size_class) × (backend, device_index)
/// datum produced by one measurement run.
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

/// Where the dispatch table decided an op should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Pick {
    pub backend:      BackendId,
    pub device_index: u32,
}

/// Options for building a [`DispatchTable`].
#[derive(Debug, Clone, Copy)]
pub struct DispatchOptions {
    pub include_reference: bool,
    pub accuracy_penalty:  f64,
}

impl Default for DispatchOptions {
    fn default() -> Self {
        Self { include_reference: false, accuracy_penalty: DEFAULT_ACCURACY_PENALTY }
    }
}

impl DispatchOptions {
    pub fn with_reference_backend(mut self, include: bool) -> Self {
        self.include_reference = include;
        self
    }
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
    include_reference: bool,
}

impl DispatchTable {
    /// Build a dispatch table from a profile report with default options.
    pub fn build(report: &ProfileReport) -> Self {
        Self::build_with(report, DispatchOptions::default())
    }

    pub fn build_with(report: &ProfileReport, opts: DispatchOptions) -> Self {
        let mut tbl = Self {
            entries:           HashMap::new(),
            size_index:        HashMap::new(),
            accuracy_penalty:  opts.accuracy_penalty,
            include_reference: opts.include_reference,
        };
        tbl.rebuild_from(report);
        tbl
    }

    fn rebuild_from(&mut self, report: &ProfileReport) {
        self.entries.clear();
        self.size_index.clear();
        let mut groups: HashMap<(OpKind, DType, SizeClass), Vec<&ProfileEntry>> = HashMap::new();
        for e in &report.entries {
            if !self.include_reference && e.backend == BackendId::Reference {
                continue;
            }
            groups.entry((e.op, e.dtype, e.size_class)).or_default().push(e);
        }
        for ((op, dtype, size_class), group) in &groups {
            for &criterion in &[Criterion::Fastest, Criterion::MostAccurate, Criterion::Balanced] {
                if let Some(winner) = self.pick_winner(group, criterion) {
                    let key = DispatchKey { op: *op, dtype: *dtype, size_class: *size_class, criterion };
                    self.entries.insert(key, Pick {
                        backend:      winner.backend,
                        device_index: winner.device_index,
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

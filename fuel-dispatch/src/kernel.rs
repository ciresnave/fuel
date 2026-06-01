//! Kernel reference + op parameters + binding table. Phase 7.5 B1+B5.
//!
//! [`KernelRef`] is the uniform function-pointer type that every
//! dispatch wrapper matches. Backend-specific typed kernels live in
//! their backend crates (e.g. `fuel_cpu_backend::byte_kernels`); the
//! *wrapper* functions here in fuel-storage bridge the dispatch-
//! erased `Storage` to those typed kernels by matching on
//! `BackendStorage::Cpu(...)` etc.
//!
//! [`OpParams`] is the typed extras bag — one variant per op family
//! that needs auxiliary data beyond inputs and outputs. Most
//! elementwise ops use `OpParams::None`; reductions carry their
//! reduce dims; conv2d carries kernel/stride/padding; etc.
//!
//! [`KernelBindingTable`] (B5) maps
//! `(OpKind, SmallVec<[DType; N]>, BackendId) -> KernelRef`. The dtype
//! list carries per-operand types — inputs in order, then outputs —
//! so mixed-precision ops (e.g. `Cast: src→dst`) and same-dtype ops
//! (e.g. `Add: [T, T, T]`) share the same key shape. Backends register
//! their dispatch wrappers via the `dispatch::register_*` functions;
//! op-builder methods consult the table at DAG construction time using
//! `Graph::target_backend(id)` as the BackendId key.
//!
//! ## Architecture (cycle-avoidance)
//!
//! - Backend crates (`fuel-cpu-backend`, …): typed kernels on their
//!   concrete storage types (`CpuStorageBytes`, …). No fuel-storage
//!   dependency.
//! - fuel-storage (this crate): dispatch wrappers that match
//!   `BackendStorage::Cpu(...)`, extract the typed storage, and
//!   call the backend's typed kernel. KernelBindingTable lives here.
//! - Backend crates depend on fuel-storage? No — only the wrappers
//!   do. fuel-storage already depends on backend crates for variant
//!   types; this round-trip closes naturally.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use fuel_core_types::conv::{ParamsConv1D, ParamsConvTranspose1D};
use fuel_core_types::dispatch::OpKind;
use fuel_core_types::probe::BackendId;
use fuel_core_types::{DType, Error, Layout, Result};
use fuel_graph::QuantType;
use smallvec::SmallVec;

/// Inline capacity for the per-operand dtype list in the binding-table
/// key. 8 covers every op currently in flight without spilling to heap:
/// PagedAttn (q + 2 caches + block_table + context_lens + alibi + out)
/// at 7 entries is the worst case in inference; mixed-precision matmul
/// is ≤ 4. Bumping later is one constant change.
pub type KernelDTypes = SmallVec<[DType; 8]>;

/// Per-binding capability flags. Today carries one flag — `strided_input`
/// — that signals "this kernel walks input strides explicitly and so
/// can consume non-contiguous input layouts (including stride-0
/// broadcast axes) without auto-Contiguize materializing them first."
/// The executor's contiguize gate consults this to skip the materialize
/// step for capable kernels, so broadcast/transpose/slice layouts can
/// reach the kernel as metadata-only views.
///
/// Default is the conservative all-false; binding sites opt in via
/// [`KernelBindingTable::register_with_caps`]. Forward-extensible by
/// adding fields (no enum/bitflags churn).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelCaps {
    /// Kernel handles non-contiguous input layouts directly (consumes
    /// the strides side-channel from `KernelRef::layouts`). When true,
    /// the executor passes non-contiguous inputs through unchanged
    /// instead of running auto-Contiguize first. Inputs with non-zero
    /// `start_offset` still go through auto-Contiguize today (slicing
    /// the device buffer to honor offset is a separate concern).
    pub strided_input: bool,
}

impl KernelCaps {
    /// All flags off. Equivalent to `Default::default()`; provided as a
    /// const for use in const-context registration tables.
    pub const fn empty() -> Self {
        Self { strided_input: false }
    }

    /// Just `strided_input` on. Ergonomic for binary/unary registrations
    /// that opt in to the wrapper-side broadcast path.
    pub const fn strided_input() -> Self {
        Self { strided_input: true }
    }
}

use fuel_storage::Storage;

/// Uniform function-pointer signature for per-backend op kernels.
///
/// Inputs are passed as a slice of `&Storage` to handle multi-input
/// ops (binary, ternary, custom). Outputs are passed as a slice of
/// `&mut Storage` to handle multi-output ops (topk, var_mean,
/// custom). Most ops use single-element slices.
///
/// `layouts` is a side-channel that parallels `inputs.append(outputs)`
/// — i.e. `layouts[0..inputs.len()]` are the input layouts in order,
/// followed by `layouts[inputs.len()..]` for the output layouts. This
/// is the load-bearing primitive for stride-aware kernels (broadcast,
/// transpose, gather-from-strided). Kernels that only support
/// contiguous-equal-shape inputs (today's default) can ignore
/// `layouts` because the executor's auto-Contiguize pass guarantees
/// every input is contiguous before this is called; the layout slice
/// then carries `Layout::contiguous(shape)` entries that are useful
/// for shape inference but redundant with `Storage.len_bytes / dtype`.
///
/// `OpParams` carries op-family-specific extras (reduce dims,
/// conv2d geometry, etc.). Most kernels match a specific variant;
/// mismatches are programming bugs that the dispatch resolver
/// must prevent. Pure layout/shape duplication that used to live in
/// `OpParams` (e.g. the old `Reduce::input_layout`) now flows through
/// `layouts` instead.
///
/// **Output Storage is pre-allocated** by the executor before the
/// kernel is called. Kernels write into the pre-allocated bytes;
/// they never allocate.
///
/// ## Multi-output kernels (Option C, Session 5)
///
/// Multi-output ops (e.g. SelectiveScan returning `(y, last_state)`)
/// emit ONE `KernelRef`. The contract:
///
/// - `outputs.len() == 1`. The single `Arc<RwLock<Storage>>` is the
///   producer's *bundled* Storage — its `bundle()` returns
///   `Some(&[OutputView; N])` describing each logical slot's
///   `dtype`/`shape`/`layout`/`byte_offset` inside the bundle's
///   underlying byte buffer.
/// - The kernel writes each logical output into its slot's byte
///   range by acquiring a single write lock on the output and
///   striding by the slot's `byte_offset`. The bundle metadata is
///   the authoritative per-slot spec — `outputs[0].read().unwrap()
///   .bundle().expect("bundled storage").get(slot_idx)` is the
///   canonical access path.
/// - Per-slot dtype tags do NOT travel through the kernel's
///   `KernelDTypes` key (which describes inputs + the bundle's
///   primary dtype only); the bundle metadata IS the per-slot
///   dispatch info.
/// - The bundle is pre-allocated by the executor via
///   `allocate_bundled_storage(device, &output_views_spec)` (see
///   `fuel_core_types::storage::allocate_bundled_storage`). Kernels
///   never allocate; they only fill bytes.
///
/// Consumers of multi-output producers are NOT multi-output kernels —
/// they're `Op::View` (zero-copy slot projection) or `Op::ViewOwned`
/// (independent slot buffer), which the executor handles directly
/// without invoking a kernel.
///
/// **Production-correct**: kernels return `Result`, never panic.
pub type KernelRef = fn(
    inputs: &[Arc<RwLock<Storage>>],
    outputs: &mut [Arc<RwLock<Storage>>],
    layouts: &[Layout],
    params: &OpParams,
) -> Result<()>;

/// Typed extras bag passed to kernels via [`KernelRef`]. One variant
/// per op family that needs auxiliary parameters; `None` for
/// parameter-less ops (most elementwise unary and binary).
#[derive(Debug, Clone)]
pub enum OpParams {
    /// Op needs no auxiliary parameters. Used by elementwise
    /// unary (relu, neg, sqr, …), elementwise binary (add, mul,
    /// sub, div, …), shape-only ops (reshape, transpose), etc.
    None,

    /// Reduction (sum, max, mean, …) along specific dims. The input
    /// tensor's [`Layout`] flows through the new `layouts` side-channel
    /// (`layouts[0]`) on `KernelRef`, so this variant carries only the
    /// op-specific extras: which dims to reduce and the keepdim flag.
    ///
    /// `dims` is the sorted list of dims to reduce; `keepdim`
    /// controls whether reduced dims are retained as size-1 in
    /// the output (today fuel-graph never asks for keepdim, but
    /// the field is reserved for the future).
    Reduce {
        dims: Vec<usize>,
        keepdim: bool,
    },

    /// Matrix multiplication. Carries the dimensions explicitly
    /// because [`Storage`](fuel_storage::Storage) only holds bytes + dtype;
    /// the kernel needs the batch shape and `(m, n, k)` to walk
    /// inputs and outputs.
    ///
    /// Shape contract: lhs `[..lhs_batch.., m, k]` @
    /// rhs `[..rhs_batch.., k, n]` → out `[..lhs_batch.., m, n]`.
    /// Per-axis the batch dims must either match or follow GQA-style
    /// divisibility (`lhs_dim > rhs_dim && lhs_dim % rhs_dim == 0`)
    /// — the kernel maps each lhs batch slot to the corresponding
    /// rhs slot via `rhs_axis_idx = lhs_axis_idx / n_rep_axis`.
    ///
    /// Equal-batch case: `lhs_batch_dims == rhs_batch_dims`. Rank-2
    /// case: both batch vectors are empty. Both work uniformly.
    ///
    /// Inputs are guaranteed contiguous by the executor's
    /// auto-Contiguize pass; transpose flags don't appear here
    /// because `Op::Transpose` is its own metadata-only op in
    /// fuel-graph.
    Matmul {
        lhs_batch_dims: Vec<usize>,
        rhs_batch_dims: Vec<usize>,
        m: usize,
        n: usize,
        k: usize,
    },

    /// 1D convolution geometry (forward path).
    Conv1D(ParamsConv1D),

    /// 2D convolution geometry (forward path). Carries the tuple-shaped
    /// stride/padding that fuel-graph's `Op::Conv2D` uses (asymmetric
    /// supported), plus the input/weight/output shapes the kernel
    /// needs to walk the multi-index. Storage holds only bytes +
    /// dtype, so spatial shapes flow through OpParams.
    ///
    /// Inputs: `x` shape `[N, Cin, Hin, Win]`, `weight` shape
    /// `[Cout, Cin/groups, Kh, Kw]`, optional `bias` shape `[Cout]`.
    /// Output: `[N, Cout, Hout, Wout]`.
    Conv2D {
        x_shape: [usize; 4],
        w_shape: [usize; 4],
        out_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    /// 1D transposed-convolution geometry.
    ConvTranspose1D(ParamsConvTranspose1D),

    /// 2D transposed-convolution geometry. Mirrors `OpParams::Conv2D`
    /// in shape (inline fields, asymmetric stride/padding/dilation,
    /// `groups`), with the additional `output_padding` parameter that
    /// transposed conv needs to disambiguate output spatial size.
    ///
    /// Inputs: `x` shape `[N, Cin, Hin, Win]`, `weight` shape
    /// `[Cin, Cout/groups, Kh, Kw]` (note transposed channel order
    /// vs Conv2D), optional `bias` shape `[Cout]`.
    /// Output: `[N, Cout, Hout, Wout]`.
    ConvTranspose2D {
        x_shape: [usize; 4],
        w_shape: [usize; 4],
        out_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    /// Sum-reduce a tensor to a smaller broadcast-compatible target
    /// shape. The kernel left-pads `output_shape` with 1s to match
    /// `input_shape`'s rank, then for each axis: if the padded output
    /// dim equals the input dim that axis carries through; if it's 1
    /// the axis is summed away.
    ReduceSumTo {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Max-reduce a tensor to a smaller broadcast-compatible target
    /// shape — the max-symmetric counterpart of `ReduceSumTo`. Same
    /// axis-alignment rules; per-axis reduction is `max` instead of
    /// `+`.
    ReduceMaxTo {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Backward of `ReduceMaxTo`. Inputs: `(x, upstream)` where
    /// `x.shape == input_shape` and `upstream.shape == output_shape`.
    /// Output: `grad_x` of `input_shape`. The kernel recomputes the
    /// forward max via `input_shape → output_shape` axis alignment,
    /// builds a tie-count mask, and routes upstream back to argmax
    /// positions (fair-share on ties).
    ReduceMaxToBackward {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },

    /// Multi-head scaled-dot-product attention shape + math params.
    /// `q` is `[B, Hq, Sq, D]`, `k` and `v` are `[B, Hkv, Sk, D]`
    /// (Hkv ≤ Hq, GQA-divisible). Optional 4th input `alibi_slopes`
    /// has shape `[Hq]` (presence is implicit in `inputs.len() == 4`).
    FlashAttn {
        b: usize,
        hq: usize,
        hkv: usize,
        sq: usize,
        sk: usize,
        d: usize,
        softmax_scale: f32,
        causal: bool,
        window_size_left: Option<usize>,
        window_size_right: Option<usize>,
        softcap: Option<f32>,
    },

    /// Paged-cache attention. `q` is `[B, Hq, Sq, D]`; `k_cache` and
    /// `v_cache` are `[num_blocks, block_size, Hkv, D]`. The 4th input
    /// is `block_table: [B, max_blocks_per_seq]` (U32 — physical block
    /// index per logical position). The 5th is `context_lens: [B]`
    /// (U32 — true context length per sequence). Optional 6th is
    /// `alibi_slopes: [Hq]`.
    PagedAttn {
        b: usize,
        hq: usize,
        hkv: usize,
        sq: usize,
        d: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
        num_blocks: usize,
        softmax_scale: f32,
        softcap: Option<f32>,
    },

    /// Slice along a single dim with explicit start/end/step. The
    /// dim is implicit in the input Layout's relabeling for
    /// multi-dim slice; this variant covers the simple case.
    Slice {
        dim: usize,
        start: usize,
        end: usize,
        step: usize,
    },

        // (Earlier `OpParams::Pad { padding: Vec<(usize, usize)>, fill_bytes: Vec<u8> }`
    // was a speculative multi-dim shape with no consumers. The single-
    // dim shape that Op::Pad actually emits lives at the bottom of this
    // enum; if multi-dim padding lands later, it can extend either
    // shape additively.)

    /// Cast input dtype → target dtype. The target lives on the
    /// output Storage's `dtype` field; this variant signals the
    /// op family without requiring a re-read.
    Cast,

    /// Affine transformation `y = mul * x + add`. Used by
    /// `Tensor::affine` / `scale_and_shift`.
    Affine {
        mul: f64,
        add: f64,
    },

    /// Element-wise clamp: `y = clamp(x, min, max)`.
    Clamp {
        min: f64,
        max: f64,
    },

    /// Element-wise integer power: `y = x.powi(exp)`.
    PowI {
        exp: i32,
    },

    /// Concatenate N inputs along one dim. The kernel needs the
    /// outer/inner element counts (product of dims before/after
    /// the concat dim) plus each input's size along the concat
    /// dim — that's all that distinguishes a concat from a
    /// sequence of slab-copies. Order matches `Node::inputs`.
    ///
    /// `axis` is the original concat dim in the output's shape;
    /// kernels that want to walk strided rank-N inputs need it
    /// to build the per-axis stride mask (the outer/inner factoring
    /// alone loses the position info).
    Concat {
        /// Product of output dims before the concat dim.
        outer_count: usize,
        /// Per-input size along the concat dim (length = N inputs).
        input_dim_sizes: Vec<usize>,
        /// Product of output dims after the concat dim.
        inner_count: usize,
        /// Original concat axis index in the output's rank-N shape.
        axis: usize,
    },

    /// Softmax along the last dim. The kernel walks
    /// `outer_count` rows of `last_dim` elements each.
    SoftmaxLastDim {
        outer_count: usize,
        last_dim: usize,
    },

    /// Last-dim norm parameters shared by RMS-norm and LayerNorm
    /// (no affine flavor for both today). The OpKind selects which
    /// kernel reads this variant.
    NormLastDim {
        outer_count: usize,
        last_dim: usize,
        eps: f64,
    },

    /// Pick slices along a single dim using a rank-1 U32 index
    /// tensor. The kernel needs the outer/inner counts (dims
    /// before/after the selected axis), the source's selected-dim
    /// size for index bounds checking, and the index count.
    IndexSelect {
        outer_count: usize,
        source_dim_size: usize,
        n_indices: usize,
        inner_count: usize,
    },

    /// N-dimensional gather along `dim`. Source and output shapes
    /// agree on every dim except `dim`; the indices tensor (U32)
    /// has output_shape and supplies the source coord for `dim`
    /// at every output position.
    Gather {
        source_shape: Vec<usize>,
        output_shape: Vec<usize>,
        dim: usize,
    },

    /// Rotary position embedding parameters. The kernel walks
    /// `outer_count` × `seq` × `head_dim` elements; `cos`/`sin`
    /// have `[seq, head_dim]` shape and broadcast across the
    /// outer dims.
    Rope {
        outer_count: usize,
        seq: usize,
        head_dim: usize,
    },

    /// Quantized matmul shape: `A [batch, m, k] @ dequant(W) [n, k] →
    /// out [batch, m, n]`. The weight tensor is a flat U32-typed
    /// byte stream representing `n * k / elements_per_block` blocks
    /// of `quant_type`. Activations are F32 today; output is F32.
    QMatMul {
        quant_type: QuantType,
        batch_count: usize,
        m: usize,
        n: usize,
        k: usize,
    },

    /// Index-add along a single dim with rank-1 U32 indices.
    /// Output is `base` with `src[..., i, ...]` accumulated into
    /// `base[..., indices[i], ...]` for every i ∈ 0..n_indices.
    IndexAdd {
        outer_count: usize,
        base_dim_size: usize,
        n_indices: usize,
        inner_count: usize,
    },

    /// N-dimensional scatter-add. Indices and src share the same
    /// shape; base may differ only along `dim`. The kernel walks
    /// every src/indices position, reads `indices[p]` for the
    /// destination's `dim` coord, and accumulates `src[p]` into
    /// `base` at that destination.
    ScatterAdd {
        base_shape: Vec<usize>,
        src_shape: Vec<usize>,
        dim: usize,
    },

    /// Flip the order of elements along one dim. The flat-3-axis
    /// view (outer × dim × inner) lets the kernel walk a tight loop
    /// without re-deriving the axis split per call. `axis` is the
    /// original dim index in the input's rank-N shape — needed by
    /// stride-aware kernels to build a per-axis flip mask.
    Flip {
        outer_count: usize,
        dim_size: usize,
        inner_count: usize,
        axis: usize,
    },

    /// Cyclic shift along one dim by `shift` positions (positive
    /// shifts move elements to higher indices, wrapping around).
    /// Same flat-3-axis view as `Flip`. `shift` is signed: negative
    /// shifts move elements to lower indices.
    Roll {
        outer_count: usize,
        dim_size: usize,
        inner_count: usize,
        shift: i64,
        axis: usize,
    },

    /// Running cumulative sum along one dim. Same flat-3-axis view
    /// as `Flip`/`Roll`. Output is always the same dtype as input;
    /// kernel needs typed addition so it's per-dtype (unlike Flip/Roll).
    CumSum {
        outer_count: usize,
        dim_size: usize,
        inner_count: usize,
        axis: usize,
    },

    /// Triangular mask parameters (used by both Triu and Tril — the
    /// op-kind picks the direction). `batch_count` is the product of
    /// leading dims; `rows`/`cols` are the last two dims. `diagonal`
    /// is signed (0 = main diagonal, positive shifts up, negative
    /// shifts down).
    Triangular {
        batch_count: usize,
        rows: usize,
        cols: usize,
        diagonal: i64,
    },

    /// LogSoftmax along the last dim. Walks `outer_count` rows of
    /// `last_dim` elements each. Per-dtype kernel (uses log/exp).
    LogSoftmaxLastDim {
        outer_count: usize,
        last_dim: usize,
    },

    /// MaskedFill: per-element fill where mask is nonzero. The kernel
    /// reads the element count from the layout; `fill_bytes` is
    /// pre-encoded in the output's dtype (one element's worth).
    MaskedFill {
        fill_bytes: Vec<u8>,
    },

    /// Backward helper for Pad. Carries the input shape so the kernel
    /// can size its scatter-add buffer; the output shape is implicit
    /// from the input shape + padding.
    PadBackward {
        in_shape: Vec<usize>,
        out_shape: Vec<usize>,
        padding: Vec<(usize, usize)>,
        mode_tag: u8,
    },

    /// Multi-dim Pad: per-axis (before, after) plus a `mode_tag`
    /// (0=Constant, 1=Reflect, 2=Replicate) and pre-encoded
    /// `fill_bytes` for Constant fill. Dtype-agnostic at the byte
    /// level: the kernel just copies bytes per the input/output shapes,
    /// which is why fill is bytes (already encoded in the output's
    /// dtype) rather than `f64` (which would force the kernel to
    /// know its dtype).
    ///
    /// `in_shape.len() == out_shape.len() == padding.len()`, and
    /// `out_shape[i] == in_shape[i] + padding[i].0 + padding[i].1`.
    /// `fill_bytes.len() == dtype_size_in_bytes` (one element's worth).
    Pad {
        in_shape: Vec<usize>,
        out_shape: Vec<usize>,
        padding: Vec<(usize, usize)>,
        mode_tag: u8,
        fill_bytes: Vec<u8>,
    },

    /// In-place scatter write parameters. The destination shape +
    /// source shape together determine the kernel walk; `ranges`
    /// gives the per-axis (start, end) slab inside the destination.
    /// `dest_shape[i] >= ranges[i].1`, and source shape is implicitly
    /// `ranges[i].1 - ranges[i].0` along axis `i`.
    ///
    /// Phase E.3.2: backs `Op::WriteSlice` for persistent KV-cache
    /// writes.
    WriteSlice {
        dest_shape: Vec<usize>,
        ranges: Vec<(usize, usize)>,
    },

    /// FusedSoftmaxCrossEntropy execution parameters. Inputs:
    /// `logits [n_rows, vocab]` (F32, flattened from the original
    /// `[..., V]` shape) and `targets [n_rows]` (I64). The kernel
    /// walks row-by-row, computing stable log-softmax + NLL +
    /// `ignore_index` masking + the requested reduction in a single
    /// pass, allocating only an `[n_rows]` per-row accumulator (and
    /// a scalar for Mean/Sum). Output is F32 scalar `[]` for
    /// Mean/Sum, F32 `[n_rows]` for None.
    ///
    /// `n_rows` and `vocab` are derived at translate time from the
    /// logits layout; the kernel uses them to iterate without
    /// re-parsing shapes.
    FusedSoftmaxCrossEntropy {
        n_rows:       usize,
        vocab:        usize,
        reduction:    fuel_graph::registry::Reduction,
        ignore_index: i64,
    },

    /// CausalConv1d execution parameters. Inputs:
    /// `x [batch, channels, seq_in]` (pre-padded by caller with
    /// `kernel - 1` left zeros), `weight [channels, 1, kernel]`,
    /// `bias [channels]`. Output `[batch, channels, seq_out]` where
    /// `seq_out = seq_in - (kernel - 1)`. All tensors share dtype.
    ///
    /// Carries both `seq_in` and `seq_out` so the kernel walks the
    /// time axis without re-deriving from shapes. `use_silu` toggles
    /// the fused SiLU activation on the output store (matches
    /// baracuda's `causal_conv1d_*_run` signature flag).
    CausalConv1d {
        batch:    usize,
        channels: usize,
        seq_in:   usize,
        seq_out:  usize,
        kernel:   usize,
        use_silu: bool,
    },

    /// SelectiveScan execution parameters. 5 required inputs:
    /// `u [batch, seqlen, dim]`, `delta [batch, seqlen, dim]`,
    /// `a [dim, dstate]`, `b [batch, seqlen, dstate]`,
    /// `c [batch, seqlen, dstate]`. Output `y [batch, seqlen, dim]`.
    /// All tensors share dtype.
    ///
    /// `delta_softplus` toggles applying softplus(delta) before use
    /// (matches baracuda's `selective_scan_*_run` flag).
    SelectiveScan {
        batch:          usize,
        seqlen:         usize,
        dim:            usize,
        dstate:         usize,
        delta_softplus: bool,
    },

    /// SsdChunkScan execution parameters. 5 inputs:
    /// `x [batch, seqlen, heads, head_dim]`,
    /// `dt [batch, seqlen, heads]`, `a [heads]`,
    /// `b [batch, seqlen, heads, state_dim]`,
    /// `c [batch, seqlen, heads, state_dim]`. Output `y` matches `x`'s
    /// shape. All tensors share dtype.
    ///
    /// `chunk_size` is the SSD block size (GPU-parallelism knob).
    /// On CPU the kernel runs a sequential scan regardless;
    /// validation requires `chunk_size > 0` and
    /// `seqlen % chunk_size == 0`.
    SsdChunkScan {
        batch:      usize,
        seqlen:     usize,
        heads:      usize,
        head_dim:   usize,
        state_dim:  usize,
        chunk_size: usize,
    },

    /// Nf4Matmul execution parameters. 3 inputs:
    /// `activations [batch, m, k]` (rank ≥ 2; leading dims flattened
    /// into `batch`), `w_packed [n, k/2]` U8, `absmax [n, k/block_size]`
    /// F32. Output `[batch, m, n]` matching activations' dtype.
    ///
    /// `block_size` is the per-output-row, per-block scale granularity
    /// (typically 64 in bitsandbytes). `k` must be even (w_packed
    /// layout requirement) and a multiple of `block_size`.
    Nf4Matmul {
        batch:      usize,
        m:          usize,
        n:          usize,
        k:          usize,
        block_size: usize,
    },
}

// =============================================================================
// Phase 7.5 B5 — kernel binding table
// =============================================================================

/// Maps `(OpKind, KernelDTypes, BackendId)` triples to dispatch wrapper
/// functions, where `KernelDTypes` lists per-operand dtypes (inputs in
/// order, then outputs). Built once at backend registration time
/// (typically a process-wide `OnceLock` though that's not enforced
/// here for testability), consulted at execute time via lookup.
///
/// Same-dtype ops register `[T, T, ..., T]` for the right operand
/// count; mixed-precision ops (Cast, future F32×BF16→F32 matmul)
/// register the exact combo. Variadic ops (Concat) register a
/// canonical short shape `[T, T]` (one input dtype + output dtype);
/// the lookup site for those ops collapses its dtypes vector to the
/// same shorthand.
///
/// Backends register their wrappers via the `dispatch::register_*`
/// functions in this crate (e.g.
/// [`crate::dispatch::register_cpu_kernels`]).
/// Cost-fn signature for primitive-op registrations stored in
/// [`KernelBindingTable`]. Mirrors the fused-op cost-fn shape but
/// takes [`OpParams`] (the binding-table param payload) instead of
/// `FusedOpParams`.
///
/// Implementations return a [`crate::fused::CostEstimate`] computed
/// statically from shapes + dtypes + op-specific params + backend
/// capabilities. The architecture's Layer-1 (FLOP-count + bandwidth)
/// cost model lives here; Layer-2 empirical refinement composes on
/// top via the telemetry framework, not by changing this signature.
pub type CostFn = fn(
    &[fuel_core_types::Shape],
    &[DType],
    &OpParams,
    &fuel_core_types::backend::BackendCapabilities,
) -> crate::fused::CostEstimate;

/// Sentinel "no cost claim" function — returns
/// [`crate::fused::CostEstimate::default`] (all-zero). The fill pass
/// (`fill_unset_cpu_cost`) recognizes this exact function pointer
/// to decide which entries get the OpKind-family default.
pub fn unknown_cost(
    _shapes: &[fuel_core_types::Shape],
    _dtypes: &[DType],
    _params: &OpParams,
    _caps: &fuel_core_types::backend::BackendCapabilities,
) -> crate::fused::CostEstimate {
    crate::fused::CostEstimate::default()
}

/// One concrete kernel registered against a `(OpKind, dtypes, backend)`
/// decision-point key. Phase 7.6 step 9a: a single key now carries
/// `SmallVec<[BindingEntry; 2]>` — multiple alternatives compete at
/// the same decision point (e.g. cuBLAS bf16 matmul + CUTLASS bf16
/// matmul at `(MatMul, [BF16, BF16, BF16], Cuda)`). The route picker
/// (step 9b) selects among them at plan time; today's single-impl
/// callers (`lookup`, `lookup_with_caps`, `lookup_precision`,
/// `lookup_cost`) return the first entry, preserving the pre-9a
/// behavior for sites that haven't migrated.
#[derive(Clone, Copy, Debug)]
pub struct BindingEntry {
    pub kernel: KernelRef,
    pub caps: KernelCaps,
    pub precision: crate::fused::PrecisionGuarantee,
    pub cost: CostFn,
}

#[derive(Default)]
pub struct KernelBindingTable {
    bindings: HashMap<(OpKind, KernelDTypes, BackendId), SmallVec<[BindingEntry; 2]>>,
}

impl KernelBindingTable {
    pub fn new() -> Self {
        Self { bindings: HashMap::new() }
    }

    /// Register a dispatch wrapper for `(op, dtypes, backend)` with the
    /// default (all-false) capabilities. `dtypes` is the per-operand
    /// dtype list — inputs in order, then outputs.
    ///
    /// Phase 7.6 step 9a: multiple distinct kernels may register
    /// against the same `(op, dtypes, backend)` key — they become
    /// sibling alternatives at one decision point. Registering the
    /// **same** `KernelRef` function pointer twice **panics** at
    /// registration time as a programmer-error guard (registration
    /// runs at module init via `Lazy`/`OnceLock`, so a panic there
    /// fails fast at startup rather than at runtime).
    ///
    /// PrecisionGuarantee defaults to [`PrecisionGuarantee::UNAUDITED`].
    /// Step-7b convention: the always-built backend
    /// (fuel-cpu-backend) runs [`Self::fill_unset_cpu_precision`] at
    /// the end of its bulk registration pass to upgrade every UNAUDITED
    /// CPU entry to `PRIMITIVE_DETERMINISTIC_CPU` (bit-stable per
    /// hardware). Kernels with weaker guarantees should call
    /// [`Self::register_with_precision`] explicitly *before* the fill
    /// pass to opt out of the default.
    pub fn register(
        &mut self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: KernelRef,
    ) {
        self.register_full(
            op, dtypes, backend, kernel,
            KernelCaps::empty(),
            crate::fused::PrecisionGuarantee::UNAUDITED,
            unknown_cost,
        );
    }

    /// Register a dispatch wrapper with explicit capability flags.
    /// Used by binding sites that opt into kernel-side broadcast or
    /// other non-default behavior — the executor consults the caps to
    /// decide whether to auto-Contiguize inputs before kernel call.
    pub fn register_with_caps(
        &mut self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: KernelRef,
        caps: KernelCaps,
    ) {
        self.register_full(
            op, dtypes, backend, kernel, caps,
            crate::fused::PrecisionGuarantee::UNAUDITED,
            unknown_cost,
        );
    }

    /// Register with an explicit [`crate::fused::PrecisionGuarantee`].
    /// Use this when a kernel has a precision claim that differs from
    /// the bulk-fill default (e.g., a multi-threaded reduction with
    /// non-deterministic accumulation order — bit_stable_on_same_hardware
    /// must be false).
    pub fn register_with_precision(
        &mut self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: KernelRef,
        precision: crate::fused::PrecisionGuarantee,
    ) {
        self.register_full(
            op, dtypes, backend, kernel, KernelCaps::empty(),
            precision, unknown_cost,
        );
    }

    /// Backwards-compatible alias for full-form registration without
    /// an explicit cost function. New code should prefer
    /// [`Self::register_full`].
    pub fn register_with_caps_and_precision(
        &mut self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: KernelRef,
        caps: KernelCaps,
        precision: crate::fused::PrecisionGuarantee,
    ) {
        self.register_full(op, dtypes, backend, kernel, caps, precision, unknown_cost);
    }

    /// Phase 7.6 step 8: full-form register with caps + precision +
    /// cost. Cost defaults to [`unknown_cost`] in the other
    /// signatures; sites with non-default cost claims must use this
    /// form (and not be overwritten by the bulk
    /// [`Self::fill_unset_cpu_cost`] pass, which only touches
    /// entries whose cost is still `unknown_cost`).
    ///
    /// Phase 7.6 step 9a: appends to the alternative set for `(op,
    /// dtypes, backend)`. Distinct `KernelRef` function pointers
    /// compose as siblings. Registering the same `KernelRef` twice
    /// panics — see [`Self::register`]'s doc-comment for the
    /// rationale.
    pub fn register_full(
        &mut self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
        kernel: KernelRef,
        caps: KernelCaps,
        precision: crate::fused::PrecisionGuarantee,
        cost: CostFn,
    ) {
        let key = (op, SmallVec::from_slice(dtypes), backend);
        let entry = BindingEntry { kernel, caps, precision, cost };
        let alts = self.bindings.entry(key).or_default();
        let new_ptr = kernel as *const () as usize;
        if alts.iter().any(|e| (e.kernel as *const () as usize) == new_ptr) {
            panic!(
                "KernelBindingTable: duplicate KernelRef registered for \
                 (op={op:?}, dtypes={dtypes:?}, backend={backend:?}). \
                 Same function pointer registered twice is programmer \
                 error. Distinct alternatives at one decision point \
                 must be distinct `fn` items.",
            );
        }
        alts.push(entry);
    }

    /// Phase 7.6 step 7b: upgrade every UNAUDITED-precision CPU
    /// registration to the supplied `default`. Convention is to call
    /// this at the *end* of bulk registration, so any
    /// `register_with_precision(...)` calls that explicitly claimed a
    /// non-default value are preserved. The architecture-target
    /// shape would be precision-per-call-site, but for the ~335 CPU
    /// primitive registrations that all share the deterministic
    /// F32-accumulator property, a fill pass keeps the call sites
    /// concise without sacrificing the architectural commitment
    /// (every entry ends up with an explicit, non-UNAUDITED precision
    /// claim before lookup is ever exercised).
    ///
    /// Only entries with `backend == BackendId::Cpu` and the current
    /// `PrecisionGuarantee::UNAUDITED` sentinel are touched. Non-CPU
    /// backends register their own precision claims; non-UNAUDITED
    /// entries are preserved.
    ///
    /// Step 9a: applies to **every alternative** registered under a
    /// CPU key, not just the first — so a key with N siblings all
    /// starting at UNAUDITED ends with N entries upgraded to `default`.
    pub fn fill_unset_cpu_precision(&mut self, default: crate::fused::PrecisionGuarantee) {
        for ((_, _, backend), alts) in self.bindings.iter_mut() {
            if *backend != BackendId::Cpu {
                continue;
            }
            for entry in alts.iter_mut() {
                let p = &mut entry.precision;
                if !p.bit_stable_on_same_hardware
                    && p.max_ulp.is_none()
                    && p.max_relative.is_none()
                    && p.max_absolute.is_none()
                {
                    // Structural detection of UNAUDITED: all four
                    // value fields at sentinel defaults. A real
                    // weaker claim would set at least one of them.
                    *p = default;
                }
            }
        }
    }

    /// Phase 7.6 step 8: upgrade every still-`unknown_cost` CPU
    /// registration to the cost function returned by `dispatcher` for
    /// its OpKind. The dispatcher's contract: given an `OpKind`,
    /// return the appropriate cost-family function — typically
    /// [`crate::cost::default_cost_for_op_kind`] in production code,
    /// but tests may pass their own dispatcher to exercise specific
    /// cost shapes.
    ///
    /// Convention is to call this at the *end* of bulk registration,
    /// so any `register_full(...)` calls that explicitly claimed a
    /// non-default cost are preserved. The function-pointer equality
    /// check identifies UNKNOWN entries — any caller that wants a
    /// "weaker" claim with non-zero values must use a distinct
    /// function pointer (which won't compare equal to `unknown_cost`
    /// and thus won't be overwritten).
    ///
    /// Step 9a: applies to **every alternative** with a CPU backend —
    /// each sibling entry at a key starting at `unknown_cost` is
    /// upgraded independently.
    pub fn fill_unset_cpu_cost(&mut self, dispatcher: fn(OpKind) -> CostFn) {
        self.fill_unset_cost_for_backend(BackendId::Cpu, dispatcher);
    }

    /// Per-backend variant of [`Self::fill_unset_cpu_cost`]. Walks
    /// every alternative whose backend matches `backend` and replaces
    /// `unknown_cost` entries with the dispatcher's choice for the
    /// op. Used by [`crate::vulkan_dispatch::register_vulkan_kernels`]
    /// to bulk-fill cost functions after per-kernel precision
    /// registration; future MKL/AOCL/Metal backends register through
    /// this same helper.
    ///
    /// Backend-specific cost adjustments (e.g. higher
    /// `kernel_overhead_ns` for Vulkan command-buffer submission vs
    /// CPU's nested-loop call overhead) are a Layer-2 refinement —
    /// today's `default_cost_for_op_kind` returns the same CPU-flavored
    /// functions regardless of backend, which over-estimates Vulkan
    /// throughput on small tensors. Either pass a Vulkan-flavored
    /// dispatcher to this method, or rely on the empirical calibration
    /// framework's per-(op, dtype, backend) refinement.
    pub fn fill_unset_cost_for_backend(
        &mut self,
        backend: BackendId,
        dispatcher: fn(OpKind) -> CostFn,
    ) {
        let sentinel = unknown_cost as usize;
        for ((op, _, this_backend), alts) in self.bindings.iter_mut() {
            if *this_backend != backend {
                continue;
            }
            for entry in alts.iter_mut() {
                if (entry.cost as usize) == sentinel {
                    entry.cost = dispatcher(*op);
                }
            }
        }
    }

    /// Look up the [`crate::fused::PrecisionGuarantee`] for a
    /// registered `(op, dtypes, backend)` triple. Returns the **first
    /// alternative**'s precision; multi-impl callers wanting all
    /// alternatives use [`Self::lookup_alternatives`]. Returns
    /// `PrecisionGuarantee::UNAUDITED` if the binding is missing —
    /// callers (notably the step-7b lint) treat that as "no claim"
    /// rather than panicking.
    pub fn lookup_precision(
        &self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
    ) -> crate::fused::PrecisionGuarantee {
        let key = (op, SmallVec::from_slice(dtypes), backend);
        self.bindings
            .get(&key)
            .and_then(|alts| alts.first())
            .map(|e| e.precision)
            .unwrap_or(crate::fused::PrecisionGuarantee::UNAUDITED)
    }

    /// Look up the [`CostFn`] for a registered `(op, dtypes, backend)`
    /// triple. Returns the **first alternative**'s cost; multi-impl
    /// callers wanting all alternatives use
    /// [`Self::lookup_alternatives`]. Returns [`unknown_cost`] for
    /// missing bindings — the step-8 lint treats that as "no cost
    /// claim."
    pub fn lookup_cost(
        &self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
    ) -> CostFn {
        let key = (op, SmallVec::from_slice(dtypes), backend);
        self.bindings
            .get(&key)
            .and_then(|alts| alts.first())
            .map(|e| e.cost)
            .unwrap_or(unknown_cost)
    }

    /// Iterate `(op, dtypes, backend, precision)` over every
    /// registered alternative — one tuple per `BindingEntry`, not per
    /// key. Used by the step-7b coverage lint, which groups by
    /// OpKind and checks the bit-stable commitment per group; the
    /// grouping handles N-alternatives-per-key naturally.
    pub fn iter_precision(
        &self,
    ) -> impl Iterator<Item = (OpKind, &[DType], BackendId, crate::fused::PrecisionGuarantee)>
    {
        self.bindings.iter().flat_map(|((op, dtypes, backend), alts)| {
            alts.iter()
                .map(move |e| (*op, dtypes.as_slice(), *backend, e.precision))
        })
    }

    /// Iterate `(op, dtypes, backend, cost_fn)` over every registered
    /// alternative — one tuple per `BindingEntry`. Used by the step-8
    /// coverage lint to check the "every primitive op has a
    /// non-default cost function" commitment per OpKind.
    pub fn iter_cost(
        &self,
    ) -> impl Iterator<Item = (OpKind, &[DType], BackendId, CostFn)>
    {
        self.bindings.iter().flat_map(|((op, dtypes, backend), alts)| {
            alts.iter()
                .map(move |e| (*op, dtypes.as_slice(), *backend, e.cost))
        })
    }

    /// Iterate every registered `(op, dtypes, backend)` decision-point
    /// key — one item per key, regardless of how many alternatives
    /// share it. Consumed by `SystemTopology::build_at` to derive
    /// which backends have kernels registered + which (op, dtype)
    /// pairs each backend covers.
    pub fn iter_keys(
        &self,
    ) -> impl Iterator<Item = (OpKind, &[DType], BackendId)> {
        self.bindings
            .keys()
            .map(|(op, dtypes, backend)| (*op, dtypes.as_slice(), *backend))
    }

    /// Look up the wrapper for `(op, dtypes, backend)`. Returns the
    /// **first alternative**'s [`KernelRef`]; for capability-aware
    /// lookup use [`Self::lookup_with_caps`]; for the full alternative
    /// set use [`Self::lookup_alternatives`].
    pub fn lookup(
        &self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
    ) -> Result<KernelRef> {
        self.lookup_with_caps(op, dtypes, backend).map(|(k, _)| k)
    }

    /// Capability-aware lookup. Returns the **first alternative**'s
    /// wrapper paired with its registered [`KernelCaps`]. Surfaces
    /// [`Error::NoBackendForOp`] on missing binding
    /// (production-correct: no panic).
    pub fn lookup_with_caps(
        &self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
    ) -> Result<(KernelRef, KernelCaps)> {
        let key = (op, SmallVec::from_slice(dtypes), backend);
        self.bindings
            .get(&key)
            .and_then(|alts| alts.first())
            .map(|e| (e.kernel, e.caps))
            .ok_or_else(|| {
                let available_backends: Vec<BackendId> = self
                    .bindings
                    .keys()
                    .map(|(_, _, b)| *b)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let supported_combinations: Vec<(BackendId, OpKind, Vec<DType>)> = self
                    .bindings
                    .keys()
                    .map(|(o, d, b)| (*b, *o, d.to_vec()))
                    .collect();
                Error::NoBackendForOp {
                    op,
                    dtypes: dtypes.to_vec(),
                    available_backends,
                    supported_combinations,
                }
                .bt()
            })
    }

    /// Phase 7.6 step 9a: return the full set of registered
    /// alternatives at the `(op, dtypes, backend)` decision point.
    /// Order is registration order — append-on-register is the step-9a
    /// contract. The empty slice means "no kernel registered at this
    /// decision point."
    ///
    /// The route picker (step 9b) consumes this to rank alternatives
    /// by cost + precision at plan time. Single-impl callers continue
    /// to use [`Self::lookup_with_caps`] (which returns the first
    /// alternative — the route picker's job is the longer-term home
    /// for selection logic).
    pub fn lookup_alternatives(
        &self,
        op: OpKind,
        dtypes: &[DType],
        backend: BackendId,
    ) -> &[BindingEntry] {
        let key = (op, SmallVec::from_slice(dtypes), backend);
        self.bindings
            .get(&key)
            .map(|alts| alts.as_slice())
            .unwrap_or(&[])
    }

    /// Total number of registered alternatives across all keys (not
    /// just unique keys). Step 9a: a key with N siblings counts N.
    pub fn len(&self) -> usize {
        self.bindings.values().map(|alts| alts.len()).sum()
    }

    /// Empty binding table.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Total number of unique `(op, dtypes, backend)` keys. Step 9a:
    /// distinct from [`Self::len`], which counts alternatives.
    pub fn key_count(&self) -> usize {
        self.bindings.len()
    }
}

impl std::fmt::Debug for KernelBindingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelBindingTable")
            .field("keys", &self.bindings.len())
            .field("total_alternatives", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::DType;

    /// Smoke: KernelRef can be constructed and stored.
    #[test]
    fn kernel_ref_stores_function_pointer() {
        fn dummy_kernel(
            _inputs: &[Arc<RwLock<Storage>>],
            _outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            Ok(())
        }
        let k: KernelRef = dummy_kernel;
        // Function pointer is Copy + Clone.
        let _k2 = k;
        let _k3: KernelRef = k;
    }

    /// Smoke: OpParams variants construct cleanly and Debug-format.
    #[test]
    fn op_params_variants_construct() {
        let _ = OpParams::None;
        let _ = OpParams::Reduce {
            dims: vec![0, 1],
            keepdim: false,
        };
        let _ = OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m: 4,
            n: 8,
            k: 16,
        };
        let _ = OpParams::Slice { dim: 0, start: 0, end: 10, step: 1 };
        let _ = OpParams::Cast;
        let _ = OpParams::Affine { mul: 2.0, add: 1.0 };
        // Debug format compiles.
        let _ = format!("{:?}", OpParams::None);
    }

    /// Smoke: a hand-constructed kernel that allocates the output
    /// Storage outside this crate would be:
    ///   1. allocate output via fuel_storage::alloc_cpu_zeroed
    ///   2. wrap inputs as &[Arc<RwLock<Storage>>]
    ///   3. call the kernel
    /// Phase B5 lands the first such migration; B1 just type-checks
    /// the surface.
    #[test]
    fn kernel_ref_can_be_called() {
        fn copy_kernel(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            // Simplest real kernel: copy bytes from inputs[0] to outputs[0].
            let in_arc = &inputs[0];
            let out_arc = &outputs[0];
            let in_guard = in_arc.read().unwrap();
            let mut out_guard = out_arc.write().unwrap();
            let in_bytes = match &in_guard.inner {
                fuel_storage::BackendStorage::Cpu(s) => s.bytes(),
                #[allow(unreachable_patterns)]
                _ => return Ok(()),
            };
            match &mut out_guard.inner {
                fuel_storage::BackendStorage::Cpu(s) => {
                    s.bytes_mut().copy_from_slice(in_bytes);
                }
                #[allow(unreachable_patterns)]
                _ => {}
            }
            Ok(())
        }

        let input = fuel_storage::from_slice_cpu(&[1.0_f32, 2.0, 3.0, 4.0]);
        let output = fuel_storage::alloc_cpu_zeroed(DType::F32, 4).unwrap();
        let inputs = vec![Arc::new(RwLock::new(input))];
        let mut outputs = vec![Arc::new(RwLock::new(output))];

        let k: KernelRef = copy_kernel;
        k(&inputs, &mut outputs, &[], &OpParams::None).unwrap();

        // Output bytes match input.
        let out_guard = outputs[0].read().unwrap();
        if let fuel_storage::BackendStorage::Cpu(s) = &out_guard.inner {
            let typed: &[f32] = s.as_slice().unwrap();
            assert_eq!(typed, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    fn ok_kernel(
        _inputs: &[Arc<RwLock<Storage>>],
        _outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn ok_kernel_alt(
        _inputs: &[Arc<RwLock<Storage>>],
        _outputs: &mut [Arc<RwLock<Storage>>],
        _layouts: &[Layout],
        _params: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// Phase 7.6 step 9a: two distinct `KernelRef`s register against
    /// the same `(op, dtypes, backend)` key as sibling alternatives.
    /// `lookup_with_caps` returns the first; `lookup_alternatives`
    /// returns both in registration order.
    #[test]
    fn step_9a_two_alternatives_append_then_first_wins_on_legacy_lookup() {
        use fuel_core_types::probe::BackendId;
        let mut table = KernelBindingTable::new();
        let dts = [DType::BF16, DType::BF16, DType::BF16];
        table.register(OpKind::MatMul, &dts, BackendId::Cuda, ok_kernel);
        table.register(OpKind::MatMul, &dts, BackendId::Cuda, ok_kernel_alt);

        // Legacy single-impl lookup returns the first-registered.
        let (k1, _caps) = table
            .lookup_with_caps(OpKind::MatMul, &dts, BackendId::Cuda)
            .unwrap();
        assert_eq!(k1 as *const () as usize, ok_kernel as *const () as usize);

        // Multi-impl lookup returns both in registration order.
        let alts = table.lookup_alternatives(OpKind::MatMul, &dts, BackendId::Cuda);
        assert_eq!(alts.len(), 2);
        assert_eq!(alts[0].kernel as *const () as usize, ok_kernel as *const () as usize);
        assert_eq!(alts[1].kernel as *const () as usize, ok_kernel_alt as *const () as usize);

        // Step-9a accounting helpers stay consistent.
        assert_eq!(table.key_count(), 1, "one decision point");
        assert_eq!(table.len(), 2, "two alternatives total");
    }

    /// Phase 7.6 step 9a: registering the same `KernelRef` function
    /// pointer twice against one key panics — "Strict — panic on
    /// exact duplicate" per the user's choice (programmer-error
    /// guard, registration runs at module init).
    #[test]
    #[should_panic(expected = "duplicate KernelRef")]
    fn step_9a_duplicate_kernel_ref_panics() {
        use fuel_core_types::probe::BackendId;
        let mut table = KernelBindingTable::new();
        let dts = [DType::F32, DType::F32, DType::F32];
        table.register(OpKind::MatMul, &dts, BackendId::Cpu, ok_kernel);
        table.register(OpKind::MatMul, &dts, BackendId::Cpu, ok_kernel);
    }

    /// Phase 7.6 step 9a: `fill_unset_cpu_precision` upgrades every
    /// alternative under a CPU key, not just the first.
    #[test]
    fn step_9a_fill_passes_touch_every_alternative() {
        use crate::fused::PrecisionGuarantee;
        use fuel_core_types::probe::BackendId;
        let mut table = KernelBindingTable::new();
        let dts = [DType::F32, DType::F32, DType::F32];
        table.register(OpKind::AddElementwise, &dts, BackendId::Cpu, ok_kernel);
        table.register(OpKind::AddElementwise, &dts, BackendId::Cpu, ok_kernel_alt);

        // Both start at UNAUDITED.
        for e in table.lookup_alternatives(OpKind::AddElementwise, &dts, BackendId::Cpu) {
            assert!(!e.precision.bit_stable_on_same_hardware);
        }

        let bit_stable = PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: Some(0),
            max_relative: None,
            max_absolute: None,
            notes: "test default",
        };
        table.fill_unset_cpu_precision(bit_stable);

        // Both got upgraded.
        let alts = table.lookup_alternatives(OpKind::AddElementwise, &dts, BackendId::Cpu);
        assert_eq!(alts.len(), 2);
        for e in alts {
            assert!(e.precision.bit_stable_on_same_hardware);
            assert_eq!(e.precision.max_ulp, Some(0));
        }
    }
}

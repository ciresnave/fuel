//! Nf4Matmul — bitsandbytes-style 4-bit NormalFloat quantized matrix
//! multiply. Fifth FusedOpRegistry entry from the re-framed CPU
//! OpKind coverage plan; the only one whose mechanical shape diverges
//! from the FSCE / Mamba trio (new dtype-level quant format + new
//! 3-input fused-matmul shape).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   total `decompose` to the primitive dequantize→matmul recipe — see the
//!   "recipe" note below — and a stubbed pattern).
//!
//! Inputs: `[activations, w_packed, absmax]`.
//!   - `activations`: `[..., M, K]` — caller's dtype (F32/F16/BF16
//!     in v1).
//!   - `w_packed`:    `[N, K/2]` U8 — two NF4 codes per byte; `K`
//!     must be even. Lower nibble at column `k_byte` holds the code
//!     for `k = 2·k_byte`; upper nibble holds `k = 2·k_byte + 1`.
//!     This matches the bitsandbytes convention for the standard
//!     K-fastest packing.
//!   - `absmax`:      `[N, K/block_size]` F32 — per-output-row,
//!     per-block scale. `K` must be a multiple of `block_size`
//!     (typically 64 in bitsandbytes).
//!
//! Output: `[..., M, N]` matching the activations' dtype.
//!
//! ## NF4 NormalFloat lookup table
//!
//! The 16 NormalFloat values [-1, -0.696, …, +1] (the inverse-CDF
//! quantiles of the standard normal that minimize the expected
//! quantization error for N(0, 1)-distributed weights) are **baked
//! into the kernel** — not a runtime input. Modifying them would
//! mean a different quantization format entirely.
//!
//! ## Why a new fused op (not extending QMATMUL)
//!
//! [`super::qmatmul`] takes a single `w_q_bytes` input that holds a
//! self-contained block stream (per GGUF / llama.cpp's `BlockQ*`
//! convention: each block embeds its own scale). NF4 splits weight
//! and scales into **two separate tensors** (the packed codes and
//! the absmax scales), which doesn't fit QMATMUL's single-input
//! shape. Adding NF4 as a `QuantType` variant would require
//! special-casing the input count throughout the dispatch path —
//! more disruptive than just adding a sibling fused op.
//!
//! ## Architectural note — primitive decomposition (the recipe)
//!
//! Unlike [`super::qmatmul`] (whose GGUF block stream embeds its scales
//! inline), NF4's `(w_packed, absmax)` split *is* expressible in the
//! primitive basis, so [`decompose`] emits the total recipe
//! `dequantize(w_packed, absmax) → matmul` (per G2 2026-06-20 — every
//! fused op carries a total, never-panic `decompose`; a self-return
//! would strand an opaque island that breaks the optimizer). The dequant
//! is built from primitives with **no data-carrying `Const` and no
//! device handle** (a `decompose` fn has neither):
//!   1. **nibble unpack** — `Cast(U8→F32)` then `lower = wf − 16·⌊wf/16⌋`,
//!      `upper = ⌊wf/16⌋` (exact for `wf ∈ 0..256`, `1/16 = 2⁻⁴`);
//!   2. **interleave** the two `[N, K/2]` half-planes to codes `[N, K]`
//!      via `Unsqueeze → Concat → Reshape` (lower at even `k`, upper at
//!      odd `k` — the K-fastest bnb packing);
//!   3. **codebook lookup** as an indicator sum `Σᵢ LUTᵢ·relu(1−|c−i|)`
//!      — pure elementwise `AddScalar/Abs/Neg/Relu/MulScalar/Add`, exact
//!      because codes are exact small integers (only `i == c` contributes);
//!   4. **per-block scale** — broadcast `absmax[N, K/bs]` across the block
//!      to `[N, K]` and multiply;
//!   5. cast to the activation dtype, transpose to `[K, N]`, `MatMul`.
//!
//! This is the *math* the kernel computes; the fused kernel stays the
//! faster path (it avoids the dequant DRAM round-trip). Whether to keep
//! the fused form or use this lowering is the optimizer's cost-guided
//! call — `decompose` only supplies the recipe. `cpu_fallback` handles
//! backends without a native kernel.
//!
//! ## Why `BackwardKind::NotDifferentiable`
//!
//! NF4 is an inference format. The weight is frozen (the U8 byte
//! stream isn't a smooth function of any continuous parameter), and
//! the activation gradient via "dequantize then standard matmul" is
//! the wrong recipe (any caller wanting that should use F32 weights
//! to begin with). Mirrors QMATMUL's same decision.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// The 16 NF4 NormalFloat values (bitsandbytes standard quantization
/// curve). Kept byte-identical to `fuel_cpu_backend::byte_kernels::NF4_LUT`
/// — the fused CPU kernel bakes the same table. Duplicated here (rather
/// than depending on the backend crate, which would invert the dependency
/// direction) because these values *define the format*: changing them is a
/// different quantization scheme, so drift would be a correctness bug the
/// decompose-vs-kernel parity test (`nf4_matmul_decompose_matches_kernel`)
/// catches.
const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Metadata-side registry entry for Nf4Matmul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::NF4_MATMUL,
        name:       "Nf4Matmul",
        family:     FusedOpFamily::Quantized,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule: `[..., M, N]` where M is activations' second-
/// to-last dim and N is the weight's first dim (per
/// `w_packed: [N, K/2]`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    let a_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert!(
        a_dims.len() >= 2,
        "Nf4Matmul: activations must be rank ≥ 2, got {a_dims:?}"
    );
    debug_assert_eq!(
        w_dims.len(), 2,
        "Nf4Matmul: w_packed must be rank 2 [N, K/2], got {w_dims:?}"
    );
    let n = w_dims[0];
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n);
    Shape::from_dims(&out_dims)
}

/// Dtype rule: output dtype matches input 0 (activations). The
/// U8 w_packed and F32 absmax don't influence the output dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    input_dtypes[0]
}

/// Total primitive decomposition of Nf4Matmul: `dequantize(w_packed,
/// absmax) → matmul`, built entirely from the primitive basis (see the
/// module-level "recipe" note for the step-by-step). Per G2 (2026-06-20)
/// this is total and never panics — the only self-return is the
/// belt-and-suspenders wrong-params guard (structurally impossible for a
/// `NF4_MATMUL` node), which is the driver's fixpoint signal, not a crash.
///
/// The recipe is the *math* the kernel computes; the fused kernel remains
/// the faster path (it fuses the dequant into the GEMM, avoiding the
/// materialized-`[N, K]`-weight DRAM round-trip). The optimizer chooses
/// between them by cost.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let block_size = match params {
        FusedOpParams::Nf4Matmul { block_size } => *block_size,
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };

    let (a_id, w_id, abs_id, a_shape, w_shape, dtype) = {
        let n = graph.node(id);
        let a_shape = graph.node(n.inputs[0]).shape.clone();
        let w_shape = graph.node(n.inputs[1]).shape.clone();
        (n.inputs[0], n.inputs[1], n.inputs[2], a_shape, w_shape, n.dtype)
    };
    let f32 = DType::F32;

    let w_dims = w_shape.dims();
    let n_out = w_dims[0];
    let k_half = w_dims[1];
    let k = k_half * 2;
    let a_dims = a_shape.dims();
    // Collapse every leading activation dim into a single M' so the GEMM is a
    // plain 2-D `[M', K] @ [K, N]`; reshape back to `[..., N]` at the end.
    let m_prime: usize = a_dims[..a_dims.len() - 1].iter().product();

    let half_shape = Shape::from_dims(&[n_out, k_half]);
    let code_shape = Shape::from_dims(&[n_out, k]);

    // --- 1. nibble unpack: w_packed U8 → F32, split each byte into two codes.
    let wf = graph.push(Node {
        op: Op::Cast(f32), inputs: vec![w_id], shape: half_shape.clone(), dtype: f32,
    });
    let wf_div16 = graph.push(Node {
        op: Op::MulScalar(1.0 / 16.0), inputs: vec![wf], shape: half_shape.clone(), dtype: f32,
    });
    let upper = graph.push(Node {
        op: Op::Floor, inputs: vec![wf_div16], shape: half_shape.clone(), dtype: f32,
    });
    let up16 = graph.push(Node {
        op: Op::MulScalar(16.0), inputs: vec![upper], shape: half_shape.clone(), dtype: f32,
    });
    let lower = graph.push(Node {
        op: Op::Sub, inputs: vec![wf, up16], shape: half_shape.clone(), dtype: f32,
    });

    // --- 2. interleave lower (even k) + upper (odd k) → codes [N, K].
    let three_shape = Shape::from_dims(&[n_out, k_half, 1]);
    let lower3 = graph.push(Node {
        op: Op::Unsqueeze { dim: 2 }, inputs: vec![lower], shape: three_shape.clone(), dtype: f32,
    });
    let upper3 = graph.push(Node {
        op: Op::Unsqueeze { dim: 2 }, inputs: vec![upper], shape: three_shape, dtype: f32,
    });
    let stacked = graph.push(Node {
        op: Op::Concat { dim: 2 }, inputs: vec![lower3, upper3],
        shape: Shape::from_dims(&[n_out, k_half, 2]), dtype: f32,
    });
    let codes = graph.push(Node {
        op: Op::Reshape(code_shape.clone()), inputs: vec![stacked],
        shape: code_shape.clone(), dtype: f32,
    });

    // --- 3. codebook lookup as an indicator sum: Σᵢ LUTᵢ · relu(1 − |c − i|).
    // Codes are exact small integers, so exactly one indicator is 1 per
    // element and the sum equals `LUT[code]` with no rounding. Entries with
    // `LUT == 0` contribute nothing and are skipped.
    let mut nf4val: Option<NodeId> = None;
    for (i, &v) in NF4_LUT.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let diff = graph.push(Node {
            op: Op::AddScalar(-(i as f64)), inputs: vec![codes], shape: code_shape.clone(), dtype: f32,
        });
        let ad = graph.push(Node {
            op: Op::Abs, inputs: vec![diff], shape: code_shape.clone(), dtype: f32,
        });
        let neg = graph.push(Node {
            op: Op::Neg, inputs: vec![ad], shape: code_shape.clone(), dtype: f32,
        });
        let one_minus = graph.push(Node {
            op: Op::AddScalar(1.0), inputs: vec![neg], shape: code_shape.clone(), dtype: f32,
        });
        let ind = graph.push(Node {
            op: Op::Relu, inputs: vec![one_minus], shape: code_shape.clone(), dtype: f32,
        });
        let term = graph.push(Node {
            op: Op::MulScalar(v as f64), inputs: vec![ind], shape: code_shape.clone(), dtype: f32,
        });
        nf4val = Some(match nf4val {
            None => term,
            Some(prev) => graph.push(Node {
                op: Op::Add, inputs: vec![prev, term], shape: code_shape.clone(), dtype: f32,
            }),
        });
    }
    // NF4_LUT always has nonzero entries; the fixpoint guard keeps this total.
    let nf4val = nf4val.unwrap_or(codes);

    // --- 4. per-block absmax scale: broadcast [N, K/bs] across the block → [N, K].
    let n_blocks = k / block_size;
    let abs3 = graph.push(Node {
        op: Op::Unsqueeze { dim: 2 }, inputs: vec![abs_id],
        shape: Shape::from_dims(&[n_out, n_blocks, 1]), dtype: f32,
    });
    let abs_b = graph.push(Node {
        op: Op::BroadcastTo(Shape::from_dims(&[n_out, n_blocks, block_size])), inputs: vec![abs3],
        shape: Shape::from_dims(&[n_out, n_blocks, block_size]), dtype: f32,
    });
    let scale_full = graph.push(Node {
        op: Op::Reshape(code_shape.clone()), inputs: vec![abs_b], shape: code_shape.clone(), dtype: f32,
    });
    let dequant = graph.push(Node {
        op: Op::Mul, inputs: vec![nf4val, scale_full], shape: code_shape, dtype: f32,
    });

    // --- 5. cast to activation dtype, transpose to [K, N], batched matmul.
    let dequant_typed = if dtype == f32 {
        dequant
    } else {
        graph.push(Node {
            op: Op::Cast(dtype), inputs: vec![dequant], shape: Shape::from_dims(&[n_out, k]), dtype,
        })
    };
    let dequant_t = graph.push(Node {
        op: Op::Transpose, inputs: vec![dequant_typed], shape: Shape::from_dims(&[k, n_out]), dtype,
    });
    let a2 = graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&[m_prime, k])), inputs: vec![a_id],
        shape: Shape::from_dims(&[m_prime, k]), dtype,
    });
    let out2 = graph.push(Node {
        op: Op::MatMul, inputs: vec![a2, dequant_t],
        shape: Shape::from_dims(&[m_prime, n_out]), dtype,
    });
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n_out);
    graph.push(Node {
        op: Op::Reshape(Shape::from_dims(&out_dims)), inputs: vec![out2],
        shape: Shape::from_dims(&out_dims), dtype,
    })
}

/// Matcher stub — Nf4Matmul nodes originate from the explicit
/// `Tensor::nf4_matmul` builder. There's no primitive subgraph to
/// recognize (the NF4 unpacking + lookup-table dequant doesn't
/// exist as fuel-graph primitives).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

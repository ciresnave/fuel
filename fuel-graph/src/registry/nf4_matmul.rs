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
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

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
        destructive_input: None,
        id: FusedOps::NF4_MATMUL,
        name: "Nf4Matmul",
        family: FusedOpFamily::Quantized,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
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
        input_shapes.len(),
        3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    let a_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert!(
        a_dims.len() >= 2,
        "Nf4Matmul: activations must be rank ≥ 2, got {a_dims:?}"
    );
    debug_assert_eq!(
        w_dims.len(),
        2,
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
        input_dtypes.len(),
        3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    input_dtypes[0]
}

/// Nf4Matmul's `dequantize(w_packed, absmax) → matmul` primitive recipe as
/// portable [`PatternNode`] DATA (Increment C) — the structure-preserving
/// migration of the pre-migration imperative `decompose` body onto the re-emit
/// machinery. It is a `recipe(a_shape, w_shape, dtype, block_size) ->
/// PatternNode` builder that uses ORDINARY Rust control flow to select
/// structure + bake concrete shapes/scalars/dtypes per call — no new bridge
/// machinery (`decompose_via_recipe` is param-agnostic; the caller builds the
/// per-call recipe then hands it to the bridge). The config surface:
///   * the **15-entry NF4 codebook** — a fixed Rust loop emitting one baked
///     indicator term per nonzero `NF4_LUT` value (a compile-time-constant
///     unroll, not a data-dependent branch);
///   * the **product-collapse** — every leading activation dim folds into a
///     single `M'` computed CONCRETELY here and BAKED into the pre-GEMM
///     `Reshape([M', K])` target (and the post-GEMM `Reshape([.., N])` restore),
///     so no `DimExpr::Param` threading is needed;
///   * the **`block_size`** — the per-block absmax `BroadcastTo` target dims are
///     concrete per call;
///   * the **output-dtype tail `Cast(dtype)`** — emitted ONLY when
///     `dtype != F32` (the F32 dequant needs no re-cast). This is the
///     config-branch / DTYPE-branch idiom (mirrors
///     `fused_softmax_cross_entropy`'s `needs_cast` tail), with the target
///     dtype baked from the fused node's declared output dtype.
///
/// Bind space: `0 = activations [.., M, K]`, `1 = w_packed [N, K/2] U8`,
/// `2 = absmax [N, K/block_size] F32` — the fused node's input order. Every
/// shape is a concrete integer at decompose time, so the shape-target ops
/// (`Reshape`/`BroadcastTo`) carry BAKED absolute `target_shape` attrs and every
/// scalar (`1/16`, `16`, the per-entry `-i` / `LUT[i]`, `1.0`) is a BAKED
/// pattern constant — NO open scalar slots, NO rel carriers. The interior
/// elementwise nodes carry no shape attr (`primitive_shape` derives them). The
/// whole nibble-unpack + codebook chain runs in F32 (the `Cast(F32)` on
/// `w_packed` seeds it; absmax is already F32); the single dtype-polymorphic tail
/// re-casts to the activation dtype before the transpose + GEMM. Nodes reused by
/// the imperative DAG — `wf` (feeds the /16 chain AND the low-nibble `Sub`),
/// `upper` (feeds `*16` AND the odd-`k` unsqueeze), `codes` (feeds all 15
/// indicator terms) — are written as shared subtrees; emit's within-call
/// identity-share collapses each to one node, reconstructing the imperative
/// DAG's sharing.
///
/// The lowered form (per the pre-migration imperative body):
///
/// ```text
///   wf          = Cast(F32)(bind1)                       # w_packed U8 → F32
///   upper       = Floor(MulScalar(1/16)(wf))             # high nibble
///   lower       = Sub(wf, MulScalar(16)(upper))          # low  nibble
///   codes       = Reshape([N, K])(Concat{2}(            # interleave even/odd k
///                     Unsqueeze{2}(lower), Unsqueeze{2}(upper)))
///   nf4val      = Σᵢ MulScalar(LUTᵢ)(Relu(AddScalar(1)(  # indicator-sum codebook
///                     Neg(Abs(AddScalar(-i)(codes))))))   #   (nonzero LUTᵢ only)
///   scale_full  = Reshape([N, K])(BroadcastTo(           # per-block absmax
///                     [N, K/bs, bs])(Unsqueeze{2}(bind2)))
///   dequant     = Mul(nf4val, scale_full)                # F32 weight [N, K]
///   dequant_t   = Transpose([dtype-cast if dtype != F32](dequant))   # → [K, N]
///   a2          = Reshape([M', K])(bind0)                # product-collapse
///   out         = Reshape([.., N])(MatMul(a2, dequant_t))
/// ```
fn recipe(a_shape: &Shape, w_shape: &Shape, dtype: DType, block_size: usize) -> PatternNode {
    use OpTag as T;
    let w_dims = w_shape.dims();
    let n_out = w_dims[0];
    let k_half = w_dims[1];
    let k = k_half * 2;
    let a_dims = a_shape.dims();
    // Collapse every leading activation dim into a single M' so the GEMM is a
    // plain 2-D `[M', K] @ [K, N]` — computed CONCRETELY and baked into the
    // Reshape target (no DimExpr::Param threading), reshaped back to `[.., N]`.
    let m_prime: usize = a_dims[..a_dims.len() - 1].iter().product();
    let n_blocks = k / block_size;

    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs,
        operands,
    };
    let bind = |i: u8| PatternNode::Bind { index: i };
    let shape_attr = |dims: &[usize]| OpAttrs {
        target_shape: dims.iter().map(|&d| d as i64).collect(),
        ..OpAttrs::default()
    };
    let cast_attr = |dt: DType| OpAttrs {
        cast_dtype: Some(dt.as_str().to_string()),
        ..OpAttrs::default()
    };
    let scalar_attr = |v: f64| OpAttrs {
        scalars: vec![v],
        ..OpAttrs::default()
    };
    let unsqueeze2 = || OpAttrs {
        dims: vec![2],
        ..OpAttrs::default()
    };
    let concat2 = || OpAttrs {
        axis: Some(2),
        ..OpAttrs::default()
    };

    // --- 1. nibble unpack: w_packed U8 → F32, split each byte into two codes.
    // `wf` is shared (the /16 chain AND the low-nibble Sub) — identity-share dedups.
    let wf = op(T::Cast, cast_attr(DType::F32), vec![bind(1)]);
    let wf_div16 = op(T::MulScalar, scalar_attr(1.0 / 16.0), vec![wf.clone()]);
    let upper = op(T::Floor, OpAttrs::default(), vec![wf_div16]);
    let up16 = op(T::MulScalar, scalar_attr(16.0), vec![upper.clone()]);
    let lower = op(T::Sub, OpAttrs::default(), vec![wf, up16]);

    // --- 2. interleave lower (even k) + upper (odd k) → codes [N, K].
    let lower3 = op(T::Unsqueeze, unsqueeze2(), vec![lower]);
    let upper3 = op(T::Unsqueeze, unsqueeze2(), vec![upper]);
    let stacked = op(T::Concat, concat2(), vec![lower3, upper3]);
    let codes = op(T::Reshape, shape_attr(&[n_out, k]), vec![stacked]);

    // --- 3. codebook lookup as an indicator sum: Σᵢ LUTᵢ · relu(1 − |c − i|).
    // Codes are exact small integers, so exactly one indicator is 1 per
    // element and the sum equals `LUT[code]` with no rounding. Entries with
    // `LUT == 0` contribute nothing and are skipped. `codes` is shared across
    // all 15 terms — identity-share dedups it to one node.
    let mut nf4val: Option<PatternNode> = None;
    for (i, &v) in NF4_LUT.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        let diff = op(T::AddScalar, scalar_attr(-(i as f64)), vec![codes.clone()]);
        let ad = op(T::Abs, OpAttrs::default(), vec![diff]);
        let neg = op(T::Neg, OpAttrs::default(), vec![ad]);
        let one_minus = op(T::AddScalar, scalar_attr(1.0), vec![neg]);
        let ind = op(T::Relu, OpAttrs::default(), vec![one_minus]);
        let term = op(T::MulScalar, scalar_attr(v as f64), vec![ind]);
        nf4val = Some(match nf4val {
            None => term,
            Some(prev) => op(T::Add, OpAttrs::default(), vec![prev, term]),
        });
    }
    // NF4_LUT always has nonzero entries; the `codes` fallback keeps this total.
    let nf4val = nf4val.unwrap_or(codes);

    // --- 4. per-block absmax scale: broadcast [N, K/bs] across the block → [N, K].
    // abs3 is rank 3 and the target is rank 3, so no D4 rank-pad fires.
    let abs3 = op(T::Unsqueeze, unsqueeze2(), vec![bind(2)]);
    let abs_b = op(
        T::BroadcastTo,
        shape_attr(&[n_out, n_blocks, block_size]),
        vec![abs3],
    );
    let scale_full = op(T::Reshape, shape_attr(&[n_out, k]), vec![abs_b]);
    let dequant = op(T::Mul, OpAttrs::default(), vec![nf4val, scale_full]);

    // --- 5. cast to activation dtype (DTYPE config branch), transpose, matmul.
    let dequant_typed = if dtype == DType::F32 {
        dequant
    } else {
        op(T::Cast, cast_attr(dtype), vec![dequant])
    };
    let dequant_t = op(T::Transpose, OpAttrs::default(), vec![dequant_typed]);
    let a2 = op(T::Reshape, shape_attr(&[m_prime, k]), vec![bind(0)]);
    let out2 = op(T::MatMul, OpAttrs::default(), vec![a2, dequant_t]);
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n_out);
    op(T::Reshape, shape_attr(&out_dims), vec![out2])
}

/// Lower a fused Nf4Matmul node to its `dequantize(w_packed, absmax) → matmul`
/// primitive subgraph and return the new root id. Since Increment C a re-emit of
/// [`recipe`]'s portable data through the [`decompose_via_recipe`] bridge
/// (structure-preserving: the emitted base map is node-for-node identical to the
/// pre-migration imperative body — see the parity test in `tests`). The per-call
/// recipe bakes the concrete shapes (incl. the product-collapsed `M'`) and
/// selects the dtype-cast tail via ordinary Rust control flow (the config-branch
/// mechanism). No open scalar slots — every constant is baked — so the bridge
/// gets `Some(Vec::new())`.
///
/// Per G2 this is total + never-panic: a wrong-params payload or a malformed node
/// (wrong input arity, non-rank-2 `w_packed`, rank-<2 activations, a
/// `block_size` that doesn't divide `K`) returns `id` (the driver's fixpoint
/// signal) BEFORE any recipe build, and any bridge decline (validation,
/// bind-arity, emit) returns `id` too. The recipe is the *math* the kernel
/// computes; the fused kernel remains the faster path (it fuses the dequant into
/// the GEMM, avoiding the materialized-`[N, K]`-weight DRAM round-trip). The
/// optimizer chooses between them by cost.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let block_size = match params {
        FusedOpParams::Nf4Matmul { block_size } => *block_size,
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };
    let (a_shape, w_shape, dtype) = {
        let n = graph.node(id);
        // Malformed node → fixpoint self-return (never panic).
        if n.inputs.len() != 3 {
            return id;
        }
        let a_shape = graph.node(n.inputs[0]).shape.clone();
        let w_shape = graph.node(n.inputs[1]).shape.clone();
        (a_shape, w_shape, n.dtype)
    };
    // Structural guards (never panic): w_packed is `[N, K/2]` (rank 2),
    // activations are rank ≥ 2, and `block_size` must evenly tile `K`.
    let w_dims = w_shape.dims();
    if w_dims.len() != 2 || a_shape.rank() < 2 {
        return id;
    }
    let k = w_dims[1] * 2;
    if block_size == 0 || k % block_size != 0 {
        return id;
    }

    let recipe_node = recipe(&a_shape, &w_shape, dtype, block_size);
    // No open scalar slots (every constant is a baked pattern constant).
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — Nf4Matmul nodes originate from the explicit
/// `Tensor::nf4_matmul` builder. There's no primitive subgraph to
/// recognize (the NF4 unpacking + lookup-table dequant doesn't
/// exist as fuel-graph primitives).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    /// FROZEN copy of the pre-Increment-C imperative `nf4_matmul::decompose`
    /// body, verbatim (the explicit-`Shape` `graph.push` spelling), before the
    /// live body was replaced by the [`recipe`] data + [`decompose_via_recipe`]
    /// bridge. The structure-preservation oracle: the migrated recipe re-emit
    /// must produce a graph structurally identical to this.
    fn frozen_legacy_nf4_matmul_decompose(
        graph: &mut Graph,
        id: NodeId,
        params: &FusedOpParams,
    ) -> NodeId {
        let block_size = match params {
            FusedOpParams::Nf4Matmul { block_size } => *block_size,
            _ => return id,
        };

        let (a_id, w_id, abs_id, a_shape, w_shape, dtype) = {
            let n = graph.node(id);
            let a_shape = graph.node(n.inputs[0]).shape.clone();
            let w_shape = graph.node(n.inputs[1]).shape.clone();
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                a_shape,
                w_shape,
                n.dtype,
            )
        };
        let f32 = DType::F32;

        let w_dims = w_shape.dims();
        let n_out = w_dims[0];
        let k_half = w_dims[1];
        let k = k_half * 2;
        let a_dims = a_shape.dims();
        let m_prime: usize = a_dims[..a_dims.len() - 1].iter().product();

        let half_shape = Shape::from_dims(&[n_out, k_half]);
        let code_shape = Shape::from_dims(&[n_out, k]);

        let wf = graph.push(Node {
            op: Op::Cast(f32),
            inputs: vec![w_id],
            shape: half_shape.clone(),
            dtype: f32,
        });
        let wf_div16 = graph.push(Node {
            op: Op::MulScalar(1.0 / 16.0),
            inputs: vec![wf],
            shape: half_shape.clone(),
            dtype: f32,
        });
        let upper = graph.push(Node {
            op: Op::Floor,
            inputs: vec![wf_div16],
            shape: half_shape.clone(),
            dtype: f32,
        });
        let up16 = graph.push(Node {
            op: Op::MulScalar(16.0),
            inputs: vec![upper],
            shape: half_shape.clone(),
            dtype: f32,
        });
        let lower = graph.push(Node {
            op: Op::Sub,
            inputs: vec![wf, up16],
            shape: half_shape.clone(),
            dtype: f32,
        });

        let three_shape = Shape::from_dims(&[n_out, k_half, 1]);
        let lower3 = graph.push(Node {
            op: Op::Unsqueeze { dim: 2 },
            inputs: vec![lower],
            shape: three_shape.clone(),
            dtype: f32,
        });
        let upper3 = graph.push(Node {
            op: Op::Unsqueeze { dim: 2 },
            inputs: vec![upper],
            shape: three_shape,
            dtype: f32,
        });
        let stacked = graph.push(Node {
            op: Op::Concat { dim: 2 },
            inputs: vec![lower3, upper3],
            shape: Shape::from_dims(&[n_out, k_half, 2]),
            dtype: f32,
        });
        let codes = graph.push(Node {
            op: Op::Reshape(code_shape.clone()),
            inputs: vec![stacked],
            shape: code_shape.clone(),
            dtype: f32,
        });

        let mut nf4val: Option<NodeId> = None;
        for (i, &v) in NF4_LUT.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            let diff = graph.push(Node {
                op: Op::AddScalar(-(i as f64)),
                inputs: vec![codes],
                shape: code_shape.clone(),
                dtype: f32,
            });
            let ad = graph.push(Node {
                op: Op::Abs,
                inputs: vec![diff],
                shape: code_shape.clone(),
                dtype: f32,
            });
            let neg = graph.push(Node {
                op: Op::Neg,
                inputs: vec![ad],
                shape: code_shape.clone(),
                dtype: f32,
            });
            let one_minus = graph.push(Node {
                op: Op::AddScalar(1.0),
                inputs: vec![neg],
                shape: code_shape.clone(),
                dtype: f32,
            });
            let ind = graph.push(Node {
                op: Op::Relu,
                inputs: vec![one_minus],
                shape: code_shape.clone(),
                dtype: f32,
            });
            let term = graph.push(Node {
                op: Op::MulScalar(v as f64),
                inputs: vec![ind],
                shape: code_shape.clone(),
                dtype: f32,
            });
            nf4val = Some(match nf4val {
                None => term,
                Some(prev) => graph.push(Node {
                    op: Op::Add,
                    inputs: vec![prev, term],
                    shape: code_shape.clone(),
                    dtype: f32,
                }),
            });
        }
        let nf4val = nf4val.unwrap_or(codes);

        let n_blocks = k / block_size;
        let abs3 = graph.push(Node {
            op: Op::Unsqueeze { dim: 2 },
            inputs: vec![abs_id],
            shape: Shape::from_dims(&[n_out, n_blocks, 1]),
            dtype: f32,
        });
        let abs_b = graph.push(Node {
            op: Op::BroadcastTo(Shape::from_dims(&[n_out, n_blocks, block_size])),
            inputs: vec![abs3],
            shape: Shape::from_dims(&[n_out, n_blocks, block_size]),
            dtype: f32,
        });
        let scale_full = graph.push(Node {
            op: Op::Reshape(code_shape.clone()),
            inputs: vec![abs_b],
            shape: code_shape.clone(),
            dtype: f32,
        });
        let dequant = graph.push(Node {
            op: Op::Mul,
            inputs: vec![nf4val, scale_full],
            shape: code_shape,
            dtype: f32,
        });

        let dequant_typed = if dtype == f32 {
            dequant
        } else {
            graph.push(Node {
                op: Op::Cast(dtype),
                inputs: vec![dequant],
                shape: Shape::from_dims(&[n_out, k]),
                dtype,
            })
        };
        let dequant_t = graph.push(Node {
            op: Op::Transpose,
            inputs: vec![dequant_typed],
            shape: Shape::from_dims(&[k, n_out]),
            dtype,
        });
        let a2 = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[m_prime, k])),
            inputs: vec![a_id],
            shape: Shape::from_dims(&[m_prime, k]),
            dtype,
        });
        let out2 = graph.push(Node {
            op: Op::MatMul,
            inputs: vec![a2, dequant_t],
            shape: Shape::from_dims(&[m_prime, n_out]),
            dtype,
        });
        let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
        out_dims.push(n_out);
        graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&out_dims)),
            inputs: vec![out2],
            shape: Shape::from_dims(&out_dims),
            dtype,
        })
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). A shared leaf (same NodeId — a
    /// bound external input) matches by identity. Shape-sensitive at EVERY node,
    /// so it catches the product-collapse `Reshape` targets and the per-block
    /// broadcast dims.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(
            na.shape, nb.shape,
            "shape mismatch at {:?}: {:?} vs {:?}",
            na.op, na.shape, nb.shape
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

    /// Build a fused Nf4Matmul node over `activations [..leading, K]` (dtype
    /// `work`), `w_packed [N, K/2]` U8, and `absmax [N, K/block_size]` F32.
    /// Returns the fused NodeId.
    fn fused_node(
        g: &mut Graph,
        leading: &[usize],
        k: usize,
        n: usize,
        block_size: usize,
        work: DType,
    ) -> NodeId {
        let mut a_dims = leading.to_vec();
        a_dims.push(k);
        let act = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&a_dims),
            dtype: work,
        });
        let w = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[n, k / 2]),
            dtype: DType::U8,
        });
        let abs = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[n, k / block_size]),
            dtype: DType::F32,
        });
        let mut out_dims = leading.to_vec();
        out_dims.push(n);
        g.push(Node {
            op: Op::Fused(
                FusedOps::NF4_MATMUL,
                FusedOpParams::Nf4Matmul { block_size },
            ),
            inputs: vec![act, w, abs],
            shape: Shape::from_dims(&out_dims),
            dtype: work,
        })
    }

    /// Increment C: the recipe decompose fires for every `block_size` ×
    /// output-dtype × leading-dim config, and its emitted base map is
    /// node-for-node identical to the FROZEN pre-migration imperative body. The
    /// representative matrix proves the three migration challenges are
    /// structure-preserving: the two `block_size`s exercise the baked per-block
    /// `BroadcastTo` dims; the F32-vs-F16 axis exercises the dtype config branch
    /// (`Cast` tail present iff `dtype != F32`); the `[1]` vs `[2, 3]` leading
    /// dims exercise the product-collapsed `M'` baked into the pre/post-GEMM
    /// `Reshape` targets. Born-red with the recipe absent (a fixpoint-returning
    /// `decompose`): `new_root == fused` trips the `assert_ne`.
    #[test]
    fn nf4_matmul_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
        for block_size in [2usize, 4] {
            for work in [DType::F32, DType::F16] {
                for leading in [vec![1usize], vec![2usize, 3]] {
                    let k = 8;
                    let n = 5;
                    let params = FusedOpParams::Nf4Matmul { block_size };
                    let mut g = Graph::new();
                    let fused = fused_node(&mut g, &leading, k, n, block_size, work);
                    let out_sh = g.node(fused).shape.clone();

                    let new_root = decompose(&mut g, fused, &params);
                    assert_ne!(
                        new_root, fused,
                        "recipe decompose fires (block_size={block_size}, work={work:?}, leading={leading:?})"
                    );
                    assert_eq!(
                        g.node(new_root).shape,
                        out_sh,
                        "output shape matches shape_rule (block_size={block_size}, leading={leading:?})"
                    );
                    assert_eq!(
                        g.node(new_root).dtype,
                        work,
                        "output dtype matches activations (work={work:?})"
                    );

                    let legacy_root = frozen_legacy_nf4_matmul_decompose(&mut g, fused, &params);
                    assert_structural_eq(&g, new_root, legacy_root);
                }
            }
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn nf4_matmul_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node(&mut g, &[2, 3], 8, 5, 2, DType::F32);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}

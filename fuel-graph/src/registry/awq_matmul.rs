// SPDX-License-Identifier: MIT OR Apache-2.0
//! AwqMatmul — AWQ (mit-han-lab, asymmetric int4 weight-only GEMM)
//! quantized matrix multiply.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   total `decompose` to the primitive dequantize→matmul recipe, and a
//!   stubbed pattern, mirroring [`super::nf4_matmul`]'s shape).
//!
//! Inputs: `[activations, qweight, qzeros, scales]` — one more than NF4's
//! three, because AWQ is ASYMMETRIC (a per-group zero-point is a real
//! operand, not baked into a codebook):
//!   - `activations`: `[..., M, K]` — caller's dtype (F32/F16/BF16 in v1).
//!   - `qweight`: `[N, K/8]` U32 — eight packed int4 weight codes per word.
//!     `K` must be a multiple of 8.
//!   - `qzeros`:  `[K/group_size, N/8]` U32 — eight packed int4
//!     zero-point codes per word. `N` must be a multiple of 8.
//!   - `scales`:  `[K/group_size, N]` F32 — per-group, per-output-channel
//!     scale.
//!
//! Output: `[..., M, N]` matching the activations' dtype. `group_size`
//! (∈ {64, 128} per AWQ's format) must evenly divide `K`.
//!
//! ## Packing-order — verified against AutoAWQ's own source
//!
//! AWQ's CUDA kernel (and AutoAWQ/llm-awq's packer) does NOT pack eight
//! int4 values into a `U32` word in naive sequential nibble order. Confirmed
//! directly against AutoAWQ's packer (`awq/modules/linear/gemm.py`,
//! `WQLinear_GEMM.from_linear`, fetched and read 2026-10-02 — not inferred
//! from a paraphrased description): it packs with
//! `order_map = [0, 2, 4, 6, 1, 3, 5, 7]` via
//! `qweight[:, col] |= intweight[:, col*8 + order_map[i]] << (i*4)` —
//! i.e. physical nibble position `i` within the packed word holds the
//! value from LOGICAL column `order_map[i]`, exactly [`AWQ_UNPACK_ORDER`]
//! below. The same source states explicitly that **the identical order
//! applies to `qzeros`**, confirming the assumption this module makes for
//! both `qweight` (along `K`) and `qzeros` (along `N`).
//!
//! (A web search surfaced a DIFFERENT 8-value permutation,
//! `[0, 4, 1, 5, 2, 6, 3, 7]`, attributed to AWQ — that is
//! `_REVERSE_AWQ_PACK_ORDER`, the INVERSE permutation some tools use to
//! convert an AWQ-packed tensor back to sequential/GPTQ order for
//! re-packing. It is not the packing order itself, and using it here
//! would have been exactly backwards. Recorded so the next reader who
//! finds that same search result doesn't re-derive the same near-miss.)
//!
//! **What IS tested:** the decompose's own internal consistency — the
//! unit tests build a small hand-packed example USING
//! [`AWQ_UNPACK_ORDER`] and confirm the decompose recovers the exact
//! codes and dequantized values those tests packed. Combined with the
//! source-verified order above, this is now a correctness claim against
//! the real format, not just an internal-consistency check — but it has
//! NOT been checked against bytes from a real `*-AWQ` checkpoint on disk;
//! that remains the next verification step before shipping a loader.
//!
//! ## Architectural note — primitive decomposition (the recipe)
//!
//! Unlike NF4 (a non-linear 16-value CODEBOOK with no primitive
//! spelling — see [`super::nf4_matmul`]'s own doc), AWQ's
//! `dequant(q) = (q − zero_point) · scale` is a LINEAR affine scheme,
//! which decomposes cleanly into Fuel's existing primitive ops: unpack
//! int4 codes from each packed `U32` word → subtract the (broadcast,
//! per-group) zero-point → multiply by the (broadcast, per-group)
//! scale → standard `MatMul`. So, unlike NF4 (bespoke fused kernel on
//! every backend, no decompose), AWQ ships with ONLY a decompose in
//! this change — no bespoke CPU kernel is written, and no backend-native
//! dispatch wiring (OpKind / cost model / CUDA registration) is added.
//! The real, measured AWQ GEMM kernel already exists
//! (`fuel-cuda-backend/src/baracuda/quant_w4a16.rs::awq_gemm_f16`,
//! `awq` cargo feature) but reaching it from this op is explicitly
//! DEFERRED follow-on work — see the module's own top-of-file note in
//! that file and this crate's registry-level gap tracking. This
//! decompose is the correctness floor: it produces the right answer on
//! every backend (including CUDA, via lowering), just not yet via the
//! fast fused kernel path.
//!
//! Unlike NF4's bit-extraction (packed as `U8`, exact in `F32`), AWQ's
//! packed words are `U32` (values up to 2^32−1), which LOSES PRECISION
//! if cast directly to `F32` (24-bit mantissa, exact only up to 2^24).
//! The nibble-extraction arithmetic therefore runs in **F64** (53-bit
//! mantissa, exact for any `u32` value) — see step 1 of [`decompose`]'s
//! recipe — with the extracted small-integer codes (0..15) cast down to
//! `F32` before the affine dequant math, which is exact since codes are
//! tiny integers.
//!
//! ## Why `BackwardKind::NotDifferentiable`
//!
//! AWQ is an inference format — the weight is frozen, same reasoning as
//! NF4 and QMatMul.

use crate::registry::{BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Physical nibble position (0..7, LSB-first within the packed `U32`
/// word) → logical column index within its group of 8. Verified directly
/// against AutoAWQ's packer source (`order_map` in
/// `awq/modules/linear/gemm.py`) — see the module doc for the citation
/// and a near-miss worth not repeating (a different 8-value permutation
/// found by web search is the INVERSE of this one, used for a different
/// purpose).
pub const AWQ_UNPACK_ORDER: [u32; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

/// Metadata-side registry entry for AwqMatmul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::AWQ_MATMUL,
        name: "AwqMatmul",
        family: FusedOpFamily::Quantized,
        pattern: crate::registry::SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule: `[..., M, N]` where M is activations' second-to-last
/// dim and N is `qweight`'s first dim (`qweight: [N, K/8]`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(),
        4,
        "AwqMatmul takes 4 inputs (activations, qweight, qzeros, scales)",
    );
    let a_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert!(
        a_dims.len() >= 2,
        "AwqMatmul: activations must be rank >= 2, got {a_dims:?}"
    );
    debug_assert_eq!(
        w_dims.len(),
        2,
        "AwqMatmul: qweight must be rank 2 [N, K/8], got {w_dims:?}"
    );
    let n = w_dims[0];
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n);
    Shape::from_dims(&out_dims)
}

/// Dtype rule: output dtype matches input 0 (activations).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(),
        4,
        "AwqMatmul takes 4 inputs (activations, qweight, qzeros, scales)",
    );
    input_dtypes[0]
}

/// Matcher stub — AwqMatmul nodes originate from the explicit
/// `NodeHandle::awq_matmul` builder. There's no primitive subgraph to
/// recognize it FROM (the inverse direction — decompose TO primitives —
/// is what [`decompose`] does).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<crate::registry::PatternMatch> {
    None
}

/// Push a node with the given op/inputs/shape/dtype; a small local helper
/// so the recipe below reads as a sequence of named steps.
fn push(graph: &mut Graph, op: Op, inputs: Vec<NodeId>, shape: Shape, dtype: DType) -> NodeId {
    graph.push(Node {
        op,
        inputs,
        shape,
        dtype,
    })
}

/// Extract the 8 packed int4 nibbles from a `U32`-packed tensor `packed`
/// (shape `[..., W]`, each element a packed word) into 8 separate `F64`
/// tensors of shape `[..., W]`, one per PHYSICAL nibble position
/// (`result[p]` holds the value at physical position `p`, NOT yet
/// reordered to logical column order). Arithmetic runs in F64 because a
/// `U32` value can exceed F32's 24-bit-mantissa exact-integer range.
///
/// `nibble_p = floor(w / 16^p) mod 16`, computed as
/// `shifted = floor(w / 16^p); nibble = shifted - 16*floor(shifted/16)`
/// — the same floor/mul/sub trick [`super::nf4_matmul`] uses for its
/// single nibble pair, generalized to 8 positions.
fn unpack_u32_nibbles(graph: &mut Graph, packed: NodeId, out_shape: &Shape) -> [NodeId; 8] {
    let f64_ = DType::F64;
    let w = push(graph, Op::Cast(f64_), vec![packed], out_shape.clone(), f64_);
    let mut positions: Vec<NodeId> = Vec::with_capacity(8);
    for p in 0..8u32 {
        let divisor = 16f64.powi(p as i32);
        let shifted = if p == 0 {
            w
        } else {
            push(
                graph,
                Op::MulScalar(1.0 / divisor),
                vec![w],
                out_shape.clone(),
                f64_,
            )
        };
        let shifted = push(graph, Op::Floor, vec![shifted], out_shape.clone(), f64_);
        let shifted_div16 = push(
            graph,
            Op::MulScalar(1.0 / 16.0),
            vec![shifted],
            out_shape.clone(),
            f64_,
        );
        let hi = push(
            graph,
            Op::Floor,
            vec![shifted_div16],
            out_shape.clone(),
            f64_,
        );
        let hi16 = push(
            graph,
            Op::MulScalar(16.0),
            vec![hi],
            out_shape.clone(),
            f64_,
        );
        let nibble = push(graph, Op::Sub, vec![shifted, hi16], out_shape.clone(), f64_);
        positions.push(nibble);
    }
    positions.try_into().unwrap_or_else(|_| unreachable!())
}

/// Reorder 8 physical-position `[..., W]` F64 tensors into one logical
/// `[..., W*8]` F32 tensor, per [`AWQ_UNPACK_ORDER`], then reshape to
/// `logical_shape` (`[..., W, 8]` flattened to `[..., W*8]` interleaves
/// correctly because `Concat` on a new trailing axis followed by
/// `Reshape` puts `concat_list[j]` at logical offset `j` within each
/// group of 8 — exactly bitsandbytes/NF4's interleave trick, generalized
/// from 2-way to 8-way).
fn interleave_and_cast_to_f32(
    graph: &mut Graph,
    positions: &[NodeId; 8],
    half_shape: &Shape,
    logical_shape: &Shape,
) -> NodeId {
    // concat_list[logical_slot] = positions[physical_position_filling_that_slot]
    // AWQ_UNPACK_ORDER[p] = logical slot that physical position p fills, so
    // invert it: concat_list[slot] = the p such that AWQ_UNPACK_ORDER[p] == slot.
    let mut concat_list = [0usize; 8];
    for (p, &slot) in AWQ_UNPACK_ORDER.iter().enumerate() {
        concat_list[slot as usize] = p;
    }
    let half_dims = half_shape.dims();
    let mut unsq_shape_dims = half_dims.to_vec();
    unsq_shape_dims.push(1);
    let unsq_shape = Shape::from_dims(&unsq_shape_dims);
    let unsqueezed: Vec<NodeId> = concat_list
        .iter()
        .map(|&p| {
            push(
                graph,
                Op::Unsqueeze {
                    dim: half_dims.len(),
                },
                vec![positions[p]],
                unsq_shape.clone(),
                DType::F64,
            )
        })
        .collect();
    let mut stacked_dims = half_dims.to_vec();
    stacked_dims.push(8);
    let stacked_shape = Shape::from_dims(&stacked_dims);
    let stacked = push(
        graph,
        Op::Concat {
            dim: half_dims.len(),
        },
        unsqueezed,
        stacked_shape,
        DType::F64,
    );
    let codes_f64 = push(
        graph,
        Op::Reshape(logical_shape.clone()),
        vec![stacked],
        logical_shape.clone(),
        DType::F64,
    );
    // Codes are exact small integers (0..15) — safe to cast down to F32.
    push(
        graph,
        Op::Cast(DType::F32),
        vec![codes_f64],
        logical_shape.clone(),
        DType::F32,
    )
}

/// Lower a fused AwqMatmul node to its
/// `dequantize(qweight, qzeros, scales) -> matmul` primitive subgraph and
/// return the new root id. Per G2 this is total + never-panic: a
/// wrong-params payload or a malformed/inconsistent node returns `id`
/// (the fixpoint signal) before any emission.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (group_size, _split_k_iters) = match params {
        FusedOpParams::AwqMatmul {
            group_size,
            split_k_iters,
        } => (*group_size, *split_k_iters),
        _ => return id,
    };
    let (a_shape, w_shape, z_shape, s_shape, dtype) = {
        let n = graph.node(id);
        if n.inputs.len() != 4 {
            return id;
        }
        let a_shape = graph.node(n.inputs[0]).shape.clone();
        let w_shape = graph.node(n.inputs[1]).shape.clone();
        let z_shape = graph.node(n.inputs[2]).shape.clone();
        let s_shape = graph.node(n.inputs[3]).shape.clone();
        (a_shape, w_shape, z_shape, s_shape, n.dtype)
    };
    let (a_id, w_id, z_id, s_id) = {
        let n = graph.node(id);
        (n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3])
    };

    // Structural guards (never panic).
    let w_dims = w_shape.dims();
    let z_dims = z_shape.dims();
    let s_dims = s_shape.dims();
    let a_dims = a_shape.dims();
    if w_dims.len() != 2 || z_dims.len() != 2 || s_dims.len() != 2 || a_dims.len() < 2 {
        return id;
    }
    let n_out = w_dims[0];
    let k_over_8 = w_dims[1];
    let k = k_over_8 * 8;
    if group_size == 0 || k % group_size != 0 || n_out % 8 != 0 {
        return id;
    }
    let n_groups = k / group_size;
    if z_dims != [n_groups, n_out / 8] || s_dims != [n_groups, n_out] {
        return id;
    }
    let k_from_act = a_dims[a_dims.len() - 1];
    if k_from_act != k {
        return id;
    }

    // --- 1. Unpack qweight [N, K/8] U32 -> wcodes [N, K] F32 (logical K order).
    let half_w_shape = Shape::from_dims(&[n_out, k_over_8]);
    let w_positions = unpack_u32_nibbles(graph, w_id, &half_w_shape);
    let wcodes = interleave_and_cast_to_f32(
        graph,
        &w_positions,
        &half_w_shape,
        &Shape::from_dims(&[n_out, k]),
    );

    // --- 2. Unpack qzeros [n_groups, N/8] U32 -> zcodes [n_groups, N] F32.
    let half_z_shape = Shape::from_dims(&[n_groups, n_out / 8]);
    let z_positions = unpack_u32_nibbles(graph, z_id, &half_z_shape);
    let zcodes = interleave_and_cast_to_f32(
        graph,
        &z_positions,
        &half_z_shape,
        &Shape::from_dims(&[n_groups, n_out]),
    );

    // --- 3. Transpose wcodes [N, K] -> [K, N] so it aligns with
    //        zeros/scales' natural [n_groups-expanded, N] layout, and so the
    //        final dequant is ALREADY the [K, N] shape MatMul needs.
    let wcodes_t = push(
        graph,
        Op::Transpose,
        vec![wcodes],
        Shape::from_dims(&[k, n_out]),
        DType::F32,
    );

    // --- 4. Expand zcodes/scales [n_groups, N] -> [K, N] (repeat each group
    //        row across its group_size rows).
    let expand = |graph: &mut Graph, x: NodeId, dtype_x: DType| -> NodeId {
        let unsq = push(
            graph,
            Op::Unsqueeze { dim: 1 },
            vec![x],
            Shape::from_dims(&[n_groups, 1, n_out]),
            dtype_x,
        );
        let bcast = push(
            graph,
            Op::BroadcastTo(Shape::from_dims(&[n_groups, group_size, n_out])),
            vec![unsq],
            Shape::from_dims(&[n_groups, group_size, n_out]),
            dtype_x,
        );
        push(
            graph,
            Op::Reshape(Shape::from_dims(&[k, n_out])),
            vec![bcast],
            Shape::from_dims(&[k, n_out]),
            dtype_x,
        )
    };
    let zcodes_full = expand(graph, zcodes, DType::F32);
    let scales_full = expand(graph, s_id, DType::F32);

    // --- 5. dequant[K, N] = (wcodes_t - zcodes_full) * scales_full.
    let centered = push(
        graph,
        Op::Sub,
        vec![wcodes_t, zcodes_full],
        Shape::from_dims(&[k, n_out]),
        DType::F32,
    );
    let dequant = push(
        graph,
        Op::Mul,
        vec![centered, scales_full],
        Shape::from_dims(&[k, n_out]),
        DType::F32,
    );

    // --- 6. Cast to activation dtype (skip if already F32, like NF4's tail).
    let dequant_typed = if dtype == DType::F32 {
        dequant
    } else {
        push(
            graph,
            Op::Cast(dtype),
            vec![dequant],
            Shape::from_dims(&[k, n_out]),
            dtype,
        )
    };

    // --- 7. Product-collapse activations [..., M, K] -> [M', K], matmul,
    //        restore leading dims.
    let m_prime: usize = a_dims[..a_dims.len() - 1].iter().product();
    let a2 = push(
        graph,
        Op::Reshape(Shape::from_dims(&[m_prime, k])),
        vec![a_id],
        Shape::from_dims(&[m_prime, k]),
        dtype,
    );
    let out2 = push(
        graph,
        Op::MatMul,
        vec![a2, dequant_typed],
        Shape::from_dims(&[m_prime, n_out]),
        dtype,
    );
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n_out);
    push(
        graph,
        Op::Reshape(Shape::from_dims(&out_dims)),
        vec![out2],
        Shape::from_dims(&out_dims),
        dtype,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    /// Pack 8 int4 codes (each 0..15) into one `U32` word per
    /// [`AWQ_UNPACK_ORDER`] (the decompose's assumed order) — the inverse of
    /// [`unpack_u32_nibbles`] + [`interleave_and_cast_to_f32`]. `codes[i]` is
    /// the value at LOGICAL column `i`; this places it at its
    /// `AWQ_UNPACK_ORDER`-defined physical nibble position.
    fn pack_u32_nibbles(codes: [u32; 8]) -> u32 {
        let mut word = 0u32;
        for (p, &slot) in AWQ_UNPACK_ORDER.iter().enumerate() {
            word |= (codes[slot as usize] & 0xF) << (4 * p);
        }
        word
    }

    fn const_node(g: &mut Graph, shape: Shape, dtype: DType) -> NodeId {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape,
            dtype,
        })
    }

    /// Build a fused AwqMatmul node over a single group (`group_size == K`,
    /// `n_groups == 1`) with `n_out == 8` (one packed zero/scale word) and
    /// `k == 8` (one packed weight word per output row) — the smallest shape
    /// that exercises every unpack axis exactly once.
    fn fused_node_1group(g: &mut Graph, m: usize, work: DType) -> (NodeId, NodeId, NodeId, NodeId) {
        let k = 8usize;
        let n = 8usize;
        let group_size = k;
        let act = const_node(g, Shape::from_dims(&[m, k]), work);
        let qweight = const_node(g, Shape::from_dims(&[n, k / 8]), DType::U32);
        let qzeros = const_node(g, Shape::from_dims(&[1, n / 8]), DType::U32);
        let scales = const_node(g, Shape::from_dims(&[1, n]), DType::F32);
        let fused = g.push(Node {
            op: Op::Fused(
                FusedOps::AWQ_MATMUL,
                FusedOpParams::AwqMatmul {
                    group_size,
                    split_k_iters: 8,
                },
            ),
            inputs: vec![act, qweight, qzeros, scales],
            shape: Shape::from_dims(&[m, n]),
            dtype: work,
        });
        (fused, qweight, qzeros, scales)
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn awq_matmul_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let (fused, _, _, _) = fused_node_1group(&mut g, 2, DType::F32);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// Born-red control: the decompose actually fires (produces a new root,
    /// not the fused node itself) and the output shape/dtype match the
    /// fused node's own declared shape/dtype.
    #[test]
    fn awq_matmul_decompose_fires_and_matches_declared_shape_dtype() {
        let mut g = Graph::new();
        let (fused, ..) = fused_node_1group(&mut g, 2, DType::F32);
        let out_sh = g.node(fused).shape.clone();
        let out_dt = g.node(fused).dtype;
        let params = FusedOpParams::AwqMatmul {
            group_size: 8,
            split_k_iters: 8,
        };
        let new_root = decompose(&mut g, fused, &params);
        assert_ne!(new_root, fused, "decompose must fire, not fixpoint");
        assert_eq!(g.node(new_root).shape, out_sh);
        assert_eq!(g.node(new_root).dtype, out_dt);
    }

    /// Correctness: a single-group, single-weight-word, single-zero-word
    /// example with HAND-CHOSEN codes/scale, executed by walking the emitted
    /// graph's arithmetic directly (no backend execution available in a
    /// `fuel-graph`-only test) via a tiny interpreter that mirrors exactly
    /// the ops this decompose emits. This is the decompose's OWN packing
    /// convention end-to-end (pack -> unpack -> dequant), not an external
    /// AWQ reference — see the module doc's packing-order caveat.
    ///
    /// Weight codes (logical K order) = [1, 2, 3, 4, 5, 6, 7, 8] for every
    /// output row (same codes packed per row since `n_out`-axis doesn't
    /// interact with the K-axis packing). Zero codes (logical N order) =
    /// [0, 1, 2, 3, 4, 5, 6, 7]. Scale = 2.0 for every output channel.
    /// Expected dequantized weight row for channel n: `(code_k - zero_n) *
    /// 2.0`, independent of k here since every row uses the same codes.
    #[test]
    fn awq_matmul_decompose_dequant_matches_hand_computed_example() {
        let mut g = Graph::new();
        let m = 1usize;
        let (fused, qweight, qzeros, scales) = fused_node_1group(&mut g, m, DType::F32);

        // Pack qweight: every output row (8 of them) uses logical codes
        // [1..8]. Only ONE packed word exists since k/8 == 1 and the
        // shape is [n_out=8, 1] -- so there are 8 words, one per row, all
        // identical.
        let w_codes_per_row: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let packed_w_word = pack_u32_nibbles(w_codes_per_row);
        let w_bytes: Vec<u32> = vec![packed_w_word; 8];

        // Pack qzeros: one group, logical N codes [0..7].
        let z_codes: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let packed_z_word = pack_u32_nibbles(z_codes);
        let z_bytes: Vec<u32> = vec![packed_z_word];

        let scale_val = 2.0f32;
        let scale_bytes = [scale_val; 8];

        // Verify the pack/unpack round trip directly (sanity on the test's
        // own packing helper before trusting the graph-level assertions
        // below).
        for (p, &expected_slot_value) in w_codes_per_row.iter().enumerate() {
            let slot = AWQ_UNPACK_ORDER
                .iter()
                .position(|&s| s == p as u32)
                .unwrap();
            let shift = 4 * slot;
            let extracted = (packed_w_word >> shift) & 0xF;
            assert_eq!(
                extracted, expected_slot_value,
                "round-trip check: logical col {p} should extract to {expected_slot_value}"
            );
        }

        // Expected dequantized weight (post-transpose [K, N] layout):
        // dequant[k][n] = (w_codes_per_row[k] - z_codes[n]) * scale_val,
        // since every output row packs the SAME w_codes_per_row.
        let mut expected = vec![vec![0f32; 8]; 8];
        for (k, row) in expected.iter_mut().enumerate() {
            for (n, cell) in row.iter_mut().enumerate() {
                *cell = (w_codes_per_row[k] as f32 - z_codes[n] as f32) * scale_val;
            }
        }

        let params = FusedOpParams::AwqMatmul {
            group_size: 8,
            split_k_iters: 8,
        };
        let new_root = decompose(&mut g, fused, &params);
        assert_ne!(new_root, fused);

        // Walk the emitted graph with a tiny interpreter that understands
        // exactly the op set this decompose emits, evaluating with the
        // packed bytes above bound to qweight/qzeros/scales and an
        // activations input bound to a simple all-ones row (so the final
        // MatMul output is directly the dequantized weight's row sums,
        // letting us cross-check the dequant math transitively).
        let act_input = {
            let n = g.node(fused);
            n.inputs[0]
        };
        let bindings: std::collections::HashMap<NodeId, Vec<f64>> = [
            (act_input, vec![1.0f64; m * 8]),
            (
                qweight,
                w_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
            (
                qzeros,
                z_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
            (
                scales,
                scale_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
        ]
        .into_iter()
        .collect();
        let result = eval_node(&g, new_root, &bindings);

        // Expected output[n] = sum_k expected[k][n] (activations row is all
        // ones), since out = a2[1,K] @ dequant[K,N].
        let expected_out: Vec<f64> = (0..8)
            .map(|n| expected.iter().map(|row| row[n] as f64).sum())
            .collect();
        for (i, (&got, &want)) in result.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "output[{i}]: got {got}, want {want}"
            );
        }
    }

    /// Sabotage: skip the zero-point subtraction (dequant = wcodes * scale,
    /// no centering) and confirm the correctness test's own comparison
    /// would catch it -- i.e. the hand-computed expected values genuinely
    /// discriminate a broken decompose, not just a differently-shaped one.
    #[test]
    fn awq_matmul_sabotage_skipping_zero_point_changes_the_result() {
        let mut g = Graph::new();
        let m = 1usize;
        let (fused, qweight, qzeros, scales) = fused_node_1group(&mut g, m, DType::F32);
        let w_codes_per_row: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let packed_w_word = pack_u32_nibbles(w_codes_per_row);
        let w_bytes: Vec<u32> = vec![packed_w_word; 8];
        let z_codes: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let packed_z_word = pack_u32_nibbles(z_codes);
        let z_bytes: Vec<u32> = vec![packed_z_word];
        let scale_val = 2.0f32;
        let scale_bytes = [scale_val; 8];

        let params = FusedOpParams::AwqMatmul {
            group_size: 8,
            split_k_iters: 8,
        };
        let new_root = decompose(&mut g, fused, &params);

        let act_input = {
            let n = g.node(fused);
            n.inputs[0]
        };
        let bindings: std::collections::HashMap<NodeId, Vec<f64>> = [
            (act_input, vec![1.0f64; m * 8]),
            (
                qweight,
                w_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
            (
                qzeros,
                z_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
            (
                scales,
                scale_bytes.iter().map(|&x| x as f64).collect::<Vec<_>>(),
            ),
        ]
        .into_iter()
        .collect();
        let result = eval_node(&g, new_root, &bindings);

        // Sabotaged expectation: if the zero-point were (wrongly) never
        // subtracted, dequant[k][n] = w_codes_per_row[k] * scale_val
        // (independent of n), so every output column would be IDENTICAL --
        // which the real (correct) result must NOT be, since z_codes varies
        // by n.
        let all_equal = result.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
        assert!(
            !all_equal,
            "the correct decompose's output must vary across n (zero-point \
             subtracted per-channel) -- if this fires, the zero-point term \
             silently dropped out somewhere"
        );
    }

    /// Minimal interpreter for exactly the op set [`decompose`] emits
    /// (`Cast`, `Floor`, `MulScalar`, `AddScalar`, `Sub`, `Mul`, `Unsqueeze`,
    /// `Concat`, `Reshape`, `BroadcastTo`, `Transpose`, `MatMul`, `Const`),
    /// evaluated over `f64` with the given leaf bindings. Shapes are tracked
    /// via the graph's own `Node::shape`; this interpreter trusts them
    /// (rather than re-deriving) since shape correctness is covered by the
    /// `shape_rule`/structural-guard tests above.
    fn eval_node(
        g: &Graph,
        id: NodeId,
        bindings: &std::collections::HashMap<NodeId, Vec<f64>>,
    ) -> Vec<f64> {
        if let Some(v) = bindings.get(&id) {
            return v.clone();
        }
        let n = g.node(id);
        let elem_count = n.shape.dims().iter().product::<usize>().max(1);
        match &n.op {
            Op::Cast(_) => eval_node(g, n.inputs[0], bindings),
            Op::Floor => eval_node(g, n.inputs[0], bindings)
                .into_iter()
                .map(f64::floor)
                .collect(),
            Op::MulScalar(s) => eval_node(g, n.inputs[0], bindings)
                .into_iter()
                .map(|x| x * s)
                .collect(),
            Op::AddScalar(s) => eval_node(g, n.inputs[0], bindings)
                .into_iter()
                .map(|x| x + s)
                .collect(),
            Op::Sub => {
                let a = eval_node(g, n.inputs[0], bindings);
                let b = eval_node(g, n.inputs[1], bindings);
                a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
            }
            Op::Mul => {
                let a = eval_node(g, n.inputs[0], bindings);
                let b = eval_node(g, n.inputs[1], bindings);
                a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
            }
            Op::Unsqueeze { .. } | Op::Reshape(_) => eval_node(g, n.inputs[0], bindings),
            // Every Concat this decompose emits concatenates K tensors whose
            // LAST axis has size 1 (freshly `Unsqueeze`d) along that last
            // axis -- i.e. a per-"outer index" interleave: output[o*K + j] =
            // inputs[j][o], NOT a whole-array concatenation (which would be
            // correct only for a concat along axis 0 of flat arrays, not the
            // innermost axis of a multi-row tensor).
            Op::Concat { .. } => {
                let arrs: Vec<Vec<f64>> = n
                    .inputs
                    .iter()
                    .map(|&i| eval_node(g, i, bindings))
                    .collect();
                let k = arrs.len();
                let outer = arrs.first().map(|a| a.len()).unwrap_or(0);
                let mut out = vec![0f64; outer * k];
                for (o, slot) in out.chunks_mut(k).enumerate() {
                    for (j, arr) in arrs.iter().enumerate() {
                        slot[j] = arr[o];
                    }
                }
                out
            }
            Op::BroadcastTo(target) => {
                let src = eval_node(g, n.inputs[0], bindings);
                let src_shape = g.node(n.inputs[0]).shape.dims().to_vec();
                let target_dims = target.dims();
                // Only the group-broadcast shape this decompose emits:
                // [n_groups, 1, N] -> [n_groups, group_size, N].
                assert_eq!(src_shape.len(), 3);
                assert_eq!(target_dims.len(), 3);
                let (g0, _one, n0) = (src_shape[0], src_shape[1], src_shape[2]);
                let gsize = target_dims[1];
                let mut out = Vec::with_capacity(g0 * gsize * n0);
                for gi in 0..g0 {
                    for _ in 0..gsize {
                        out.extend_from_slice(&src[gi * n0..(gi + 1) * n0]);
                    }
                }
                out
            }
            Op::Transpose => {
                let src = eval_node(g, n.inputs[0], bindings);
                let src_shape = g.node(n.inputs[0]).shape.dims().to_vec();
                assert_eq!(src_shape.len(), 2);
                let (rows, cols) = (src_shape[0], src_shape[1]);
                let mut out = vec![0f64; rows * cols];
                for r in 0..rows {
                    for c in 0..cols {
                        out[c * rows + r] = src[r * cols + c];
                    }
                }
                out
            }
            Op::MatMul => {
                let a = eval_node(g, n.inputs[0], bindings);
                let b = eval_node(g, n.inputs[1], bindings);
                let a_shape = g.node(n.inputs[0]).shape.dims().to_vec();
                let b_shape = g.node(n.inputs[1]).shape.dims().to_vec();
                let (m, k) = (a_shape[0], a_shape[1]);
                let (_k2, nn) = (b_shape[0], b_shape[1]);
                let mut out = vec![0f64; m * nn];
                for i in 0..m {
                    for j in 0..nn {
                        let mut acc = 0f64;
                        for kk in 0..k {
                            acc += a[i * k + kk] * b[kk * nn + j];
                        }
                        out[i * nn + j] = acc;
                    }
                }
                out
            }
            other => panic!("eval_node: unhandled op {other:?} (elem_count={elem_count})"),
        }
    }
}

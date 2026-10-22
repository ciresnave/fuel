// SPDX-License-Identifier: MIT OR Apache-2.0
//! AwqMatmul — AWQ (mit-han-lab, asymmetric int4 weight-only GEMM)
//! quantized matrix multiply.
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   total `decompose` to the primitive dequantize→matmul recipe, and a
//!   stubbed pattern, mirroring [`super::nf4_matmul`]'s shape).
//! - [`awq_zeros_w`] — the kernel's per-row padding rule for `zeros`.
//!
//! ## Operand layout — the baracuda W4A16 kernel layout
//!
//! Inputs: `[activations, weight, zeros, scales]`. The three quantized
//! operands are in the layout baracuda's AWQ GEMM kernel reads (documented on
//! baracuda's `Int4AwqGemmArgs`, alpha.147), NOT the HuggingFace checkpoint
//! layout. `fuel_quantized::awq_repack::repack_hf_awq_for_kernel` converts a
//! checkpoint's `qweight`/`qzeros`/`scales` into it on the host. Everything is
//! OC-major (`N` = output channels, `K` = input features,
//! `groups = K / group_size`, `mult = 128 / group_size`):
//!   - `activations`: `[..., M, K]` — caller's dtype (F32/F16/BF16).
//!   - `weight`: `[N, K/8]` U32 — eight int4 codes per word in plain
//!     sequential nibble order: nibble `k % 8` of word `k / 8` in row `n`.
//!     `K` must be a multiple of 8.
//!   - `zeros`: `[N, zeros_w]` U32 — nibble `g % 8` of word `g / 8` in row `n`
//!     holds group `g`'s zero-point; `zeros_w = awq_zeros_w(K, group_size)`.
//!   - `scales`: `[N, 8 * zeros_w]` F16 — entry `g` of row `n` is group `g`'s
//!     scale.
//!
//! The tail of each `zeros`/`scales` row past `groups` is padding the kernel
//! never reads as data; this op's decompose ignores it too, whatever it holds.
//!
//! Output: `[..., M, N]` matching the activations' dtype. `group_size` must
//! be 64 or 128 (the kernel's two supported values) and evenly divide `K`.
//!
//! ## Why the operands are the kernel layout (PM ruling B, ROADMAP row 31)
//!
//! The previous operands were the HuggingFace layout with AutoAWQ's
//! interleaved nibble order. A fused kernel cannot read that layout, so every
//! launch would repack on the device. The repack is a one-time host cost at
//! load, so the op takes the kernel layout and one buffer set serves both the
//! CUDA kernel and this decompose. Changing the operand layout of an existing
//! op is a Breaking change (0.21).
//!
//! ## Architectural note — primitive decomposition (the recipe)
//!
//! Unlike NF4 (a non-linear 16-value CODEBOOK with no primitive
//! spelling — see [`super::nf4_matmul`]'s own doc), AWQ's
//! `dequant(q) = (q − zero_point) · scale` is a LINEAR affine scheme,
//! which decomposes cleanly into Fuel's existing primitive ops: unpack
//! int4 codes from each packed `U32` word → subtract the (broadcast,
//! per-group) zero-point → multiply by the (broadcast, per-group)
//! scale → standard `MatMul`. This decompose is the correctness floor: it
//! produces the right answer on every backend.
//!
//! Packed words are `U32` (values up to 2^32−1), which LOSES PRECISION if
//! cast directly to `F32` (24-bit mantissa, exact only up to 2^24). The
//! nibble-extraction arithmetic therefore runs in **F64** (53-bit mantissa,
//! exact for any `u32` value) — see step 1 of [`decompose`]'s recipe — with
//! the extracted small-integer codes (0..15) cast down to `F32` before the
//! affine dequant math, which is exact since codes are tiny integers.
//!
//! ## Why `BackwardKind::Decompose` (changed 2026-10-20, PM ruling)
//!
//! The weight is frozen, so there is no weight gradient; the activation
//! gradient is well defined and is what adapter fine-tuning over a quantized
//! base needs. Autograd lowers the node through [`decompose`] (unpack,
//! dequantize, matmul) and differentiates the primitives. This replaces the
//! earlier `NotDifferentiable` ("AWQ is an inference format"), under which a
//! backward through this op panicked. Checked against a finite difference in
//! `fuel-tensor` (`awq_matmul_activation_gradient_matches_finite_difference`).

use crate::registry::{BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, Shape};

/// Words per `zeros` row (and `scales` entries per row, divided by 8) for `k`
/// input features at `group_size`: `ceil(ceil(groups / 8) / mult) * mult`
/// with `groups = k / group_size` and `mult = 128 / group_size`. `None` when
/// `group_size` is not 64 or 128 or does not divide `k`.
///
/// The same rule as `fuel_quantized::awq_repack::awq_zeros_w` (`fuel-graph`
/// cannot depend on `fuel-quantized`, so the rule is repeated and pinned by a
/// test there).
pub fn awq_zeros_w(k: usize, group_size: usize) -> Option<usize> {
    if (group_size != 64 && group_size != 128) || !k.is_multiple_of(group_size) {
        return None;
    }
    let groups = k / group_size;
    let mult = 128 / group_size;
    Some(groups.div_ceil(8).div_ceil(mult) * mult)
}

/// Metadata-side registry entry for AwqMatmul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::AWQ_MATMUL,
        name: "AwqMatmul",
        family: FusedOpFamily::Quantized,
        pattern: crate::registry::SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::Decompose,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule: `[..., M, N]` where M is activations' second-to-last
/// dim and N is `weight`'s first dim (`weight: [N, K/8]`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(),
        4,
        "AwqMatmul takes 4 inputs (activations, weight, zeros, scales)",
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
        "AwqMatmul: weight must be rank 2 [N, K/8], got {w_dims:?}"
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
        "AwqMatmul takes 4 inputs (activations, weight, zeros, scales)",
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
/// tensors of shape `[..., W]`, one per nibble position (`result[p]` holds
/// nibble `p`, LSB first). Arithmetic runs in F64 because a `U32` value can
/// exceed F32's 24-bit-mantissa exact-integer range.
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

/// Interleave 8 nibble-position `[..., W]` F64 tensors into one
/// `[..., W*8]` F32 tensor in sequential order (nibble `p` of word `w` lands at
/// column `w*8 + p`), reshaped to `logical_shape`. `Concat` on a new trailing
/// axis followed by `Reshape` puts `positions[j]` at offset `j` within each
/// group of 8 — bitsandbytes/NF4's interleave trick, generalized from 2-way
/// to 8-way.
fn interleave_and_cast_to_f32(
    graph: &mut Graph,
    positions: &[NodeId; 8],
    half_shape: &Shape,
    logical_shape: &Shape,
) -> NodeId {
    let half_dims = half_shape.dims();
    let mut unsq_shape_dims = half_dims.to_vec();
    unsq_shape_dims.push(1);
    let unsq_shape = Shape::from_dims(&unsq_shape_dims);
    let unsqueezed: Vec<NodeId> = positions
        .iter()
        .map(|&p| {
            push(
                graph,
                Op::Unsqueeze {
                    dim: half_dims.len(),
                },
                vec![p],
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

/// Keep the first `groups` columns of a `[n_out, full]` tensor (drops the
/// kernel's row padding); a no-op when nothing is padded.
fn drop_row_padding(
    graph: &mut Graph,
    x: NodeId,
    n_out: usize,
    full: usize,
    groups: usize,
) -> NodeId {
    if full == groups {
        return x;
    }
    push(
        graph,
        Op::Slice {
            dim: 1,
            start: 0,
            len: groups,
        },
        vec![x],
        Shape::from_dims(&[n_out, groups]),
        DType::F32,
    )
}

/// Lower a fused AwqMatmul node to its
/// `dequantize(weight, zeros, scales) -> matmul` primitive subgraph and
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
    let (a_shape, w_shape, z_shape, s_shape, s_dtype, dtype) = {
        let n = graph.node(id);
        if n.inputs.len() != 4 {
            return id;
        }
        let a_shape = graph.node(n.inputs[0]).shape.clone();
        let w_shape = graph.node(n.inputs[1]).shape.clone();
        let z_shape = graph.node(n.inputs[2]).shape.clone();
        let s_node = graph.node(n.inputs[3]);
        (
            a_shape,
            w_shape,
            z_shape,
            s_node.shape.clone(),
            s_node.dtype,
            n.dtype,
        )
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
    let Some(zeros_w) = awq_zeros_w(k, group_size) else {
        return id;
    };
    let n_groups = k / group_size;
    if z_dims != [n_out, zeros_w] || s_dims != [n_out, 8 * zeros_w] {
        return id;
    }
    let k_from_act = a_dims[a_dims.len() - 1];
    if k_from_act != k {
        return id;
    }

    // --- 1. Unpack weight [N, K/8] U32 -> wcodes [N, K] F32.
    let half_w_shape = Shape::from_dims(&[n_out, k_over_8]);
    let w_positions = unpack_u32_nibbles(graph, w_id, &half_w_shape);
    let wcodes = interleave_and_cast_to_f32(
        graph,
        &w_positions,
        &half_w_shape,
        &Shape::from_dims(&[n_out, k]),
    );

    // --- 2. Unpack zeros [N, zeros_w] U32 -> [N, 8*zeros_w] F32, then drop
    //        the padding past `n_groups` -> zcodes [N, n_groups].
    let half_z_shape = Shape::from_dims(&[n_out, zeros_w]);
    let z_positions = unpack_u32_nibbles(graph, z_id, &half_z_shape);
    let zcodes_padded = interleave_and_cast_to_f32(
        graph,
        &z_positions,
        &half_z_shape,
        &Shape::from_dims(&[n_out, 8 * zeros_w]),
    );
    let zcodes = drop_row_padding(graph, zcodes_padded, n_out, 8 * zeros_w, n_groups);

    // --- 3. scales [N, 8*zeros_w] (F16) -> F32, drop padding -> [N, n_groups].
    let scales_f32 = if s_dtype == DType::F32 {
        s_id
    } else {
        push(
            graph,
            Op::Cast(DType::F32),
            vec![s_id],
            s_shape.clone(),
            DType::F32,
        )
    };
    let scales = drop_row_padding(graph, scales_f32, n_out, 8 * zeros_w, n_groups);

    // --- 4. Expand zcodes/scales [N, n_groups] -> [N, K] (repeat each group
    //        column across its group_size columns).
    let expand = |graph: &mut Graph, x: NodeId| -> NodeId {
        let unsq = push(
            graph,
            Op::Unsqueeze { dim: 2 },
            vec![x],
            Shape::from_dims(&[n_out, n_groups, 1]),
            DType::F32,
        );
        let bcast = push(
            graph,
            Op::BroadcastTo(Shape::from_dims(&[n_out, n_groups, group_size])),
            vec![unsq],
            Shape::from_dims(&[n_out, n_groups, group_size]),
            DType::F32,
        );
        push(
            graph,
            Op::Reshape(Shape::from_dims(&[n_out, k])),
            vec![bcast],
            Shape::from_dims(&[n_out, k]),
            DType::F32,
        )
    };
    let zcodes_full = expand(graph, zcodes);
    let scales_full = expand(graph, scales);

    // --- 5. dequant[N, K] = (wcodes - zcodes_full) * scales_full, then
    //        transpose to the [K, N] shape MatMul needs.
    let centered = push(
        graph,
        Op::Sub,
        vec![wcodes, zcodes_full],
        Shape::from_dims(&[n_out, k]),
        DType::F32,
    );
    let dequant_nk = push(
        graph,
        Op::Mul,
        vec![centered, scales_full],
        Shape::from_dims(&[n_out, k]),
        DType::F32,
    );
    let dequant = push(
        graph,
        Op::Transpose,
        vec![dequant_nk],
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
    use std::collections::HashMap;

    fn const_node(g: &mut Graph, shape: Shape, dtype: DType) -> NodeId {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape,
            dtype,
        })
    }

    /// A deterministic problem in the kernel layout: `n` output channels, `k`
    /// input features, `group_size` 64 or 128, `m` activation rows.
    struct Problem {
        n: usize,
        k: usize,
        gs: usize,
        m: usize,
        zeros_w: usize,
    }

    impl Problem {
        fn new(n: usize, k: usize, gs: usize, m: usize) -> Self {
            let zeros_w = awq_zeros_w(k, gs).expect("test shape is valid");
            Self {
                n,
                k,
                gs,
                m,
                zeros_w,
            }
        }
        fn groups(&self) -> usize {
            self.k / self.gs
        }
        fn wcode(&self, n: usize, k: usize) -> u32 {
            ((n * 7 + k * 3 + k / 5) % 16) as u32
        }
        fn zcode(&self, n: usize, g: usize) -> u32 {
            ((n * 5 + g * 3 + 1) % 16) as u32
        }
        /// Exact in f16 and f32.
        fn scale(&self, n: usize, g: usize) -> f64 {
            0.25 * (((n + g) % 4) + 1) as f64
        }
        fn act(&self, mi: usize, k: usize) -> f64 {
            ((mi * 3 + k) % 11) as f64 - 5.0
        }
        /// `weight [N, K/8]`, sequential nibbles.
        fn weight_words(&self) -> Vec<f64> {
            let mut out = vec![0u32; self.n * self.k / 8];
            for n in 0..self.n {
                for k in 0..self.k {
                    out[n * (self.k / 8) + k / 8] |= self.wcode(n, k) << (4 * (k % 8));
                }
            }
            out.into_iter().map(f64::from).collect()
        }
        /// `zeros [N, zeros_w]`; every nibble past `groups` is `0xF` garbage.
        fn zeros_words(&self, pad_garbage: bool) -> Vec<f64> {
            let mut out = vec![0u32; self.n * self.zeros_w];
            for n in 0..self.n {
                for g in 0..8 * self.zeros_w {
                    let code = if g < self.groups() {
                        self.zcode(n, g)
                    } else if pad_garbage {
                        0xF
                    } else {
                        0
                    };
                    out[n * self.zeros_w + g / 8] |= code << (4 * (g % 8));
                }
            }
            out.into_iter().map(f64::from).collect()
        }
        /// `scales [N, 8*zeros_w]`; the padding is a large garbage value.
        fn scale_values(&self, pad_garbage: bool) -> Vec<f64> {
            let w = 8 * self.zeros_w;
            let mut out = vec![0f64; self.n * w];
            for n in 0..self.n {
                for g in 0..w {
                    out[n * w + g] = if g < self.groups() {
                        self.scale(n, g)
                    } else if pad_garbage {
                        99.0
                    } else {
                        0.0
                    };
                }
            }
            out
        }
        /// Reference output `[M, N]` straight from the definition
        /// `sum_k a[m,k] * (code - zero) * scale`, with switches that each
        /// break one term (used only by the sabotage tests).
        fn reference(&self, skip_zero: bool, group_of: impl Fn(usize) -> usize) -> Vec<f64> {
            let mut out = vec![0f64; self.m * self.n];
            for mi in 0..self.m {
                for n in 0..self.n {
                    let mut acc = 0f64;
                    for k in 0..self.k {
                        let g = group_of(k);
                        let z = if skip_zero {
                            0.0
                        } else {
                            self.zcode(n, g) as f64
                        };
                        acc += self.act(mi, k) * (self.wcode(n, k) as f64 - z) * self.scale(n, g);
                    }
                    out[mi * self.n + n] = acc;
                }
            }
            out
        }
        fn correct_reference(&self) -> Vec<f64> {
            let gs = self.gs;
            self.reference(false, move |k| k / gs)
        }
        /// Build the fused node and evaluate its decomposition.
        fn run_decompose(&self, pad_garbage: bool) -> Vec<f64> {
            self.run_decompose_with_weight(pad_garbage, self.weight_words())
        }
        /// As [`Self::run_decompose`] with the `weight` words supplied.
        fn run_decompose_with_weight(&self, pad_garbage: bool, weight_words: Vec<f64>) -> Vec<f64> {
            let mut g = Graph::new();
            let act = const_node(&mut g, Shape::from_dims(&[self.m, self.k]), DType::F32);
            let weight = const_node(&mut g, Shape::from_dims(&[self.n, self.k / 8]), DType::U32);
            let zeros = const_node(
                &mut g,
                Shape::from_dims(&[self.n, self.zeros_w]),
                DType::U32,
            );
            let scales = const_node(
                &mut g,
                Shape::from_dims(&[self.n, 8 * self.zeros_w]),
                DType::F16,
            );
            let params = FusedOpParams::AwqMatmul {
                group_size: self.gs,
                split_k_iters: 8,
            };
            let fused = g.push(Node {
                op: Op::Fused(FusedOps::AWQ_MATMUL, params.clone()),
                inputs: vec![act, weight, zeros, scales],
                shape: Shape::from_dims(&[self.m, self.n]),
                dtype: DType::F32,
            });
            let new_root = decompose(&mut g, fused, &params);
            assert_ne!(new_root, fused, "decompose must fire, not fixpoint");
            assert_eq!(g.node(new_root).shape, Shape::from_dims(&[self.m, self.n]));
            let act_vals: Vec<f64> = (0..self.m)
                .flat_map(|mi| (0..self.k).map(move |k| (mi, k)))
                .map(|(mi, k)| self.act(mi, k))
                .collect();
            let bindings: HashMap<NodeId, Vec<f64>> = [
                (act, act_vals),
                (weight, weight_words),
                (zeros, self.zeros_words(pad_garbage)),
                (scales, self.scale_values(pad_garbage)),
            ]
            .into_iter()
            .collect();
            eval_node(&g, new_root, &bindings)
        }
    }

    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "output[{i}]: got {g}, want {w}");
        }
    }

    fn fused_node_for(g: &mut Graph, p: &Problem) -> NodeId {
        let act = const_node(g, Shape::from_dims(&[p.m, p.k]), DType::F32);
        let weight = const_node(g, Shape::from_dims(&[p.n, p.k / 8]), DType::U32);
        let zeros = const_node(g, Shape::from_dims(&[p.n, p.zeros_w]), DType::U32);
        let scales = const_node(g, Shape::from_dims(&[p.n, 8 * p.zeros_w]), DType::F16);
        g.push(Node {
            op: Op::Fused(
                FusedOps::AWQ_MATMUL,
                FusedOpParams::AwqMatmul {
                    group_size: p.gs,
                    split_k_iters: 8,
                },
            ),
            inputs: vec![act, weight, zeros, scales],
            shape: Shape::from_dims(&[p.m, p.n]),
            dtype: DType::F32,
        })
    }

    /// The padding rule, pinned to the kernel doc's formula on cases whose
    /// answers are worked by hand: 1 group at 64 -> ceil(ceil(1/8)/2)*2 = 2;
    /// 1 group at 128 -> ceil(1/1)*1 = 1; 17 groups at 64 -> ceil(3/2)*2 = 4.
    #[test]
    fn awq_zeros_w_matches_the_kernel_padding_rule() {
        assert_eq!(awq_zeros_w(64, 64), Some(2));
        assert_eq!(awq_zeros_w(128, 128), Some(1));
        assert_eq!(awq_zeros_w(17 * 64, 64), Some(4));
        // The same hand-worked cases `fuel-quantized`'s own test pins.
        assert_eq!(awq_zeros_w(128 * 9, 128), Some(2));
        assert_eq!(awq_zeros_w(64 * 8, 64), Some(2));
        assert_eq!(awq_zeros_w(128 * 16, 128), Some(2));
        assert_eq!(awq_zeros_w(64 * 24, 64), Some(4));
        assert_eq!(
            awq_zeros_w(64, 32),
            None,
            "only 64/128 are kernel group sizes"
        );
        assert_eq!(awq_zeros_w(100, 64), None, "group_size must divide k");
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn awq_matmul_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node_for(&mut g, &Problem::new(8, 64, 64, 2));
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// Totality (G2): operands whose shapes disagree with the kernel layout
    /// (here: `zeros` has the un-padded width) decline before any emission.
    #[test]
    fn awq_matmul_wrong_zeros_width_is_a_fixpoint_not_a_crash() {
        let p = Problem::new(8, 64, 64, 2);
        let mut g = Graph::new();
        let act = const_node(&mut g, Shape::from_dims(&[p.m, p.k]), DType::F32);
        let weight = const_node(&mut g, Shape::from_dims(&[p.n, p.k / 8]), DType::U32);
        let zeros = const_node(&mut g, Shape::from_dims(&[p.n, 1]), DType::U32);
        let scales = const_node(&mut g, Shape::from_dims(&[p.n, 8]), DType::F16);
        let params = FusedOpParams::AwqMatmul {
            group_size: 64,
            split_k_iters: 8,
        };
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::AWQ_MATMUL, params.clone()),
            inputs: vec![act, weight, zeros, scales],
            shape: Shape::from_dims(&[p.m, p.n]),
            dtype: DType::F32,
        });
        let before = g.len();
        assert_eq!(decompose(&mut g, fused, &params), fused);
        assert_eq!(g.len(), before);
    }

    /// Born-red control: the decompose fires and the output shape/dtype match
    /// the fused node's own declared shape/dtype.
    #[test]
    fn awq_matmul_decompose_fires_and_matches_declared_shape_dtype() {
        let p = Problem::new(8, 64, 64, 2);
        let mut g = Graph::new();
        let fused = fused_node_for(&mut g, &p);
        let out_sh = g.node(fused).shape.clone();
        let out_dt = g.node(fused).dtype;
        let params = FusedOpParams::AwqMatmul {
            group_size: 64,
            split_k_iters: 8,
        };
        let new_root = decompose(&mut g, fused, &params);
        assert_ne!(new_root, fused, "decompose must fire, not fixpoint");
        assert_eq!(g.node(new_root).shape, out_sh);
        assert_eq!(g.node(new_root).dtype, out_dt);
    }

    /// Correctness, one group (`group_size == K == 64`): activations vary with
    /// `k`, so a wrong nibble order along `K` changes the answer.
    #[test]
    fn awq_matmul_decompose_matches_reference_one_group() {
        let p = Problem::new(8, 64, 64, 3);
        assert_close(&p.run_decompose(true), &p.correct_reference());
    }

    /// Correctness, several groups with padded `zeros`/`scales` rows
    /// (`K = 640`, `group_size = 64` -> 10 groups, `zeros_w = 2`, so 6
    /// padding nibbles per row).
    #[test]
    fn awq_matmul_decompose_matches_reference_many_groups() {
        let p = Problem::new(16, 640, 64, 2);
        assert_eq!(p.groups(), 10);
        assert_eq!(p.zeros_w, 2);
        assert_close(&p.run_decompose(true), &p.correct_reference());
    }

    /// Correctness at `group_size = 128`.
    #[test]
    fn awq_matmul_decompose_matches_reference_group_128() {
        let p = Problem::new(8, 256, 128, 2);
        assert_close(&p.run_decompose(true), &p.correct_reference());
    }

    /// The kernel never reads the padding, so neither may the decompose: the
    /// result with garbage in the padding equals the result with zeros there.
    #[test]
    fn awq_matmul_decompose_ignores_row_padding() {
        let p = Problem::new(8, 640, 64, 2);
        assert_close(&p.run_decompose(true), &p.run_decompose(false));
    }

    /// Sabotage siblings: each reference breaks one term, and the decompose's
    /// output must differ from it — so the correctness tests above are
    /// sensitive to that term. (Skipping the zero-point, indexing the group
    /// as `k % groups` instead of `k / group_size`.)
    #[test]
    fn awq_matmul_sabotage_references_differ_from_the_decompose() {
        let p = Problem::new(16, 640, 64, 2);
        let got = p.run_decompose(true);
        let groups = p.groups();
        let no_zero = p.reference(true, |k| k / 64);
        let wrong_group = p.reference(false, |k| k % groups);
        for (name, bad) in [
            ("skip zero-point", no_zero),
            ("wrong group index", wrong_group),
        ] {
            let max_diff = got
                .iter()
                .zip(bad.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                max_diff > 1e-3,
                "sabotage `{name}` must change the result, max diff {max_diff}"
            );
        }
    }

    /// Sabotage: AutoAWQ's interleaved order along `K` (the previous operand
    /// layout) is a different layout; weight words packed that way and fed to
    /// the sequential-order decompose must NOT reproduce the reference.
    #[test]
    fn awq_matmul_sabotage_interleaved_nibble_order_is_not_sequential() {
        let p = Problem::new(8, 64, 64, 2);
        let order = [0usize, 2, 4, 6, 1, 3, 5, 7];
        let mut words = vec![0u32; p.n * p.k / 8];
        for n in 0..p.n {
            for k in 0..p.k {
                let phys = order.iter().position(|&o| o == k % 8).unwrap();
                words[n * (p.k / 8) + k / 8] |= p.wcode(n, k) << (4 * phys);
            }
        }
        let interleaved: Vec<f64> = words.into_iter().map(f64::from).collect();
        let got = p.run_decompose_with_weight(true, interleaved);
        let want = p.correct_reference();
        let max_diff = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(max_diff > 1e-3, "a layout mix-up must be observable");
    }

    /// Minimal interpreter for exactly the op set [`decompose`] emits
    /// (`Cast`, `Floor`, `MulScalar`, `Sub`, `Mul`, `Unsqueeze`, `Concat`,
    /// `Reshape`, `BroadcastTo`, `Slice`, `Transpose`, `MatMul`, `Const`),
    /// evaluated over `f64` with the given leaf bindings. Shapes are tracked
    /// via the graph's own `Node::shape`; this interpreter trusts them
    /// (rather than re-deriving) since shape correctness is covered by the
    /// `shape_rule`/structural-guard tests above.
    fn eval_node(g: &Graph, id: NodeId, bindings: &HashMap<NodeId, Vec<f64>>) -> Vec<f64> {
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
            // inputs[j][o].
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
            // Only the group-broadcast this decompose emits:
            // [N, groups, 1] -> [N, groups, group_size].
            Op::BroadcastTo(target) => {
                let src = eval_node(g, n.inputs[0], bindings);
                let src_shape = g.node(n.inputs[0]).shape.dims().to_vec();
                assert_eq!(src_shape.len(), 3);
                assert_eq!(src_shape[2], 1);
                let gsize = target.dims()[2];
                src.iter()
                    .flat_map(|&x| std::iter::repeat_n(x, gsize))
                    .collect()
            }
            // Only the row-padding drop this decompose emits: columns
            // [start, start+len) of a rank-2 tensor.
            Op::Slice { dim, start, len } => {
                assert_eq!(*dim, 1);
                let src = eval_node(g, n.inputs[0], bindings);
                let src_shape = g.node(n.inputs[0]).shape.dims().to_vec();
                assert_eq!(src_shape.len(), 2);
                let (rows, cols) = (src_shape[0], src_shape[1]);
                let mut out = Vec::with_capacity(rows * len);
                for r in 0..rows {
                    out.extend_from_slice(&src[r * cols + start..r * cols + start + len]);
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

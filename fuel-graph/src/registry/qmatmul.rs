//! QMatMul — quantized matrix multiply `C = A @ dequant(W_Q)`.
//! Phase 7.6 step 4 (continued — tenth op migrated; final step-4 op).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   a `decompose` that lowers **Q4_0** to a total primitive dequant→matmul
//!   recipe and surfaces the other GGUF formats as tracked gaps — see the
//!   "recipe" note below — and a stubbed pattern).
//!
//! Inputs: `[a, w_q_bytes]`.
//!   - `a`:          `[..., M, K]` F32 activations
//!   - `w_q_bytes`:  U32-typed packed block stream for the `[N, K]`
//!     weight matrix (GGUF / llama.cpp convention).
//!
//! Output: `[..., M, N]` F32.
//!
//! ## Architectural note — frozen weights, non-differentiable
//!
//! QMatMul is the inference path for quantized model weights. The
//! weight tensor is frozen (the U32 byte stream isn't a smooth
//! function of any continuous parameter), and the activation gradient
//! isn't implemented today — matches the legacy `Op::QMatMul { .. }`
//! arm in `Tensor::backward`, which panics with a clear "use a
//! dequantize + standard matmul if you need gradients" message. The
//! registry entry's `BackwardKind::NotDifferentiable` reflects this.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId, QuantType};
use fuel_ir::{DType, Shape};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

/// Metadata-side registry entry for QMatMul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::QMATMUL,
        name:       "QMatMul",
        family:     FusedOpFamily::Quantized,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output is `[..., M, N]` where `M = a.shape[-2]` and
/// `N` is the weight's output dim from `FusedOpParams::QMatMul`.
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "QMatMul takes 2 inputs (a, w_q_bytes)",
    );
    let n = match params {
        FusedOpParams::QMatMul { n, .. } => *n,
        _ => panic!("qmatmul::shape_rule got non-QMatMul params: {params:?}"),
    };
    let a_dims = input_shapes[0].dims();
    let rank = a_dims.len();
    debug_assert!(rank >= 2, "QMatMul activations must be rank ≥ 2");
    let mut out_dims: Vec<usize> = a_dims[..rank - 1].to_vec();
    out_dims.push(n);
    Shape::from_dims(&out_dims)
}

/// Dtype rule: output dtype is F32 (the activations'). The U32 w_q
/// is opaque bytes — the dtype field on its node doesn't influence
/// QMatMul's output dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "QMatMul takes 2 inputs",
    );
    input_dtypes[0]
}

// ===========================================================================
// The primitive recipe (Increment C — 2026-07-28)
// ===========================================================================
//
// QMatMul is **not** a basis gap. Its primitive recipe is
// `dequantize(w_q_bytes, quant_type) → matmul(a, ·)`, and the earlier claim
// that the dequant needs three primitives Fuel lacks was WRONG (scoped
// 2026-07-25, cross-validated with KISS/kiss-ref/Baracuda; the §7.3-0002
// necessity test came back negative from every side, so NO `Op::Bitcast` axiom
// is added — see `docs/outreach/bitcast-basis-token-design-input-ask.md`).
// Every step is expressible over the existing primitive basis:
//
//   * **Byte extraction from the U32 stream is exact F64 arithmetic.** The
//     weight is a `U32` tensor of raw GGUF bytes (`u32::from_le_bytes`). A
//     value-`Cast(U32→F64)` is exact (`u32 < 2⁵³`), then each U32 splits into
//     its four little-endian bytes by `⌊·/256ⁱ⌋ mod 256` — pure `Floor`/
//     `MulScalar`/`Sub`. (F32 can't do this: `Cast(U32→F32)` rounds above
//     `2²⁴`.) The bytes are `< 256`, so `Cast(F64→F32)` is exact and the rest
//     runs in F32.
//   * **The embedded f16 block-scale is recovered by arithmetic IEEE-754-half
//     RECONSTRUCTION, not a bitcast.** The two scale bytes give the 16-bit
//     field as an exact small integer (`b0 + 256·b1`); sign/exponent/mantissa
//     come out by `Floor`/`Sub`; the value composes as `(-1)^s · significand ·
//     2^e` with the data-dependent power of two built by a 5-bit binary
//     decomposition (a product of `2^(2^i)` selected per bit — all exact
//     multiplies). This is **bit-exact** to `f16::to_f32` for every finite f16
//     (GGUF scales are never inf/NaN), proven by
//     `f16_decode_arithmetic_matches_half_for_all_finite_bit_patterns`. It is
//     the same technique `nf4_matmul` uses for nibble unpack (read bytes as
//     small exact integers, do arithmetic), just a richer field extraction.
//   * **The block/super-block layout is `Reshape` + `Slice`** — both already
//     in the basis.
//
// Only **Q4_0** carries a recipe today (the sole format the live loader
// produces — `fuel-core/src/lazy.rs`). The other ten `QuantType`s self-return
// as surfaced gaps (never a crash), tracked on the ROADMAP ("qmatmul
// per-format decompose build-out"). This matches the `flash_attn`
// concrete-vs-symbolic precedent: a fused op counts as migrated with the
// uncovered configurations documented as gaps.
//
// The recipe is the *math* the fused kernel approximates: the CPU/Vulkan
// dequant-in-kernel arm stays the cost-preferred cover (it fuses the dequant
// into the GEMM and — for the CPU vec_dot path — also re-quantizes the
// activations, an approximation the decomposed exact form does not make), and
// the lowering is the optimizer's cost-guided alternative / the basis-map
// closure, never the executed path on a backend that has the kernel.

/// Q4_0 block geometry (llama.cpp convention).
const QK4_0: usize = 32; // weights per block
const Q4_0_BYTES_PER_BLOCK: usize = 18; // f16 scale (2) + 16 packed-nibble bytes

/// Build the total `dequant(Q4_0) → matmul` recipe as portable [`PatternNode`]
/// DATA. `w_len_u32` is the weight operand's element count (a flat U32 stream);
/// `k`/`n` are the logical weight dims `[N, K]`. Returns `None` on any
/// structural mismatch (a typed decline → the caller self-returns), never
/// panics.
///
/// Bind space: `0 = activations [.., M, K] F32`, `1 = w_q_bytes [L] U32`.
///
/// Every shape/scalar is a concrete integer at decompose time, so the shape ops
/// carry BAKED absolute `target_shape` and every constant is a BAKED pattern
/// scalar — no open slots, no rel carriers. Shared subtrees (the reshaped
/// `blocks`, `exp`, `mant`, …) are written as clones; emit's within-call
/// identity-share collapses each to one node.
fn recipe_q4_0(a_shape: &Shape, w_len_u32: usize, k: usize, n: usize) -> Option<PatternNode> {
    use OpTag as T;

    if k == 0 || n == 0 || k % QK4_0 != 0 {
        return None;
    }
    let a_dims = a_shape.dims();
    if a_dims.len() < 2 || *a_dims.last()? != k {
        return None;
    }
    let nb_per_row = k / QK4_0;
    let total_blocks = n.checked_mul(nb_per_row)?;
    let total_bytes = total_blocks.checked_mul(Q4_0_BYTES_PER_BLOCK)?;
    // The weight is a U32 stream; 4·L must reconstruct exactly the block bytes.
    if w_len_u32.checked_mul(4)? != total_bytes {
        return None;
    }
    let four_l = total_bytes; // == 4·w_len_u32
    let m_prime: usize = a_dims[..a_dims.len() - 1].iter().product();

    // --- node constructors -------------------------------------------------
    let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
    let bind = |i: u8| PatternNode::Bind { index: i };
    let shape_attr =
        |dims: &[usize]| OpAttrs { target_shape: dims.iter().map(|&d| d as i64).collect(), ..OpAttrs::default() };
    let cast_attr =
        |dt: DType| OpAttrs { cast_dtype: Some(dt.as_str().to_string()), ..OpAttrs::default() };
    let sc = |v: f64| OpAttrs { scalars: vec![v], ..OpAttrs::default() };
    let unsq = |d: u8| OpAttrs { dims: vec![d], ..OpAttrs::default() };
    let cat = |ax: i64| OpAttrs { axis: Some(ax), ..OpAttrs::default() };
    let slice = |ax: i64, start: u64, len: u64| OpAttrs {
        axis: Some(ax),
        slice_start: Some(start),
        slice_len: Some(len),
        ..OpAttrs::default()
    };

    // convenience wrappers for the most common unary/binary shapes
    let mul_s = |v: f64, x: PatternNode| op(T::MulScalar, sc(v), vec![x]);
    let add_s = |v: f64, x: PatternNode| op(T::AddScalar, sc(v), vec![x]);
    let floor = |x: PatternNode| op(T::Floor, OpAttrs::default(), vec![x]);
    let sub = |a: PatternNode, b: PatternNode| op(T::Sub, OpAttrs::default(), vec![a, b]);
    let mul = |a: PatternNode, b: PatternNode| op(T::Mul, OpAttrs::default(), vec![a, b]);
    let add = |a: PatternNode, b: PatternNode| op(T::Add, OpAttrs::default(), vec![a, b]);

    // --- 1. byte extraction: U32 stream → per-block byte matrix [tb, 18] ----
    // Exact split of each little-endian U32 into 4 bytes, done in F64.
    let wf = op(T::Cast, cast_attr(DType::F64), vec![bind(1)]); // [L] F64
    let f1 = floor(mul_s(1.0 / 256.0, wf.clone()));
    let b0 = sub(wf, mul_s(256.0, f1.clone()));
    let f2 = floor(mul_s(1.0 / 256.0, f1.clone()));
    let b1 = sub(f1, mul_s(256.0, f2.clone()));
    let f3 = floor(mul_s(1.0 / 256.0, f2.clone()));
    let b2 = sub(f2, mul_s(256.0, f3.clone()));
    let b3 = f3; // top byte: ⌊u/2²⁴⌋ < 256 since u < 2³²
    // Interleave the four byte lanes back into stream order [b0,b1,b2,b3, …].
    let lane = |x: PatternNode| op(T::Unsqueeze, unsq(1), vec![x]); // [L] → [L,1]
    let stacked = op(
        T::Concat,
        cat(1),
        vec![lane(b0), lane(b1), lane(b2), lane(b3)],
    ); // [L, 4]
    let bytes_f64 = op(T::Reshape, shape_attr(&[four_l]), vec![stacked]); // [4L] F64
    let bytes = op(T::Cast, cast_attr(DType::F32), vec![bytes_f64]); // [4L] F32, 0..255
    let blocks = op(T::Reshape, shape_attr(&[total_blocks, Q4_0_BYTES_PER_BLOCK]), vec![bytes]); // [tb,18]

    // --- 2. f16 block-scale decode (arithmetic IEEE-754-half) --------------
    // scale bytes = blocks[:, 0:2]; bits = b0 + 256·b1.
    let sb0 = op(T::Slice, slice(1, 0, 1), vec![blocks.clone()]); // [tb,1]
    let sb1 = op(T::Slice, slice(1, 1, 1), vec![blocks.clone()]); // [tb,1]
    let bits = add(sb0, mul_s(256.0, sb1)); // [tb,1] ∈ 0..65535
    let sign = floor(mul_s(1.0 / 32768.0, bits.clone()));
    let rem = sub(bits, mul_s(32768.0, sign.clone()));
    let exp = floor(mul_s(1.0 / 1024.0, rem.clone())); // 0..31
    let mant = sub(rem, mul_s(1024.0, exp.clone())); // 0..1023
    let signed = add_s(1.0, mul_s(-2.0, sign)); // 1 − 2·sign
    // 2^exp by 5-bit binary decomposition (each factor is a power of two).
    let h0 = floor(mul_s(0.5, exp.clone()));
    let bit0 = sub(exp.clone(), mul_s(2.0, h0.clone()));
    let h1 = floor(mul_s(0.5, h0.clone()));
    let bit1 = sub(h0, mul_s(2.0, h1.clone()));
    let h2 = floor(mul_s(0.5, h1.clone()));
    let bit2 = sub(h1, mul_s(2.0, h2.clone()));
    let h3 = floor(mul_s(0.5, h2.clone()));
    let bit3 = sub(h2, mul_s(2.0, h3.clone()));
    let h4 = floor(mul_s(0.5, h3.clone()));
    let bit4 = sub(h3, mul_s(2.0, h4));
    let fac0 = add_s(1.0, bit0); // 2^bit0
    let fac1 = add_s(1.0, mul_s(3.0, bit1)); // 4^bit1
    let fac2 = add_s(1.0, mul_s(15.0, bit2)); // 16^bit2
    let fac3 = add_s(1.0, mul_s(255.0, bit3)); // 256^bit3
    let fac4 = add_s(1.0, mul_s(65535.0, bit4)); // 65536^bit4
    let pow2exp = mul(mul(mul(mul(fac0, fac1), fac2), fac3), fac4); // 2^exp
    // normal:  (1024 + mant)·2^exp·2⁻²⁵ = (1 + mant/1024)·2^(exp−15)
    let normal_mag = mul_s(
        2f64.powi(-25),
        mul(add_s(1024.0, mant.clone()), pow2exp),
    );
    // subnormal: mant·2⁻²⁴
    let sub_mag = mul_s(2f64.powi(-24), mant);
    // is_sub = 1 iff exp == 0 (exp is an exact integer ≥ 0)
    let is_sub = op(T::Relu, OpAttrs::default(), vec![add_s(1.0, mul_s(-1.0, exp))]);
    let one_minus_sub = add_s(1.0, mul_s(-1.0, is_sub.clone()));
    let mag = add(mul(is_sub, sub_mag), mul(one_minus_sub, normal_mag));
    let scale = mul(signed, mag); // [tb,1] = f16 scale value

    // --- 3. quant nibble unpack + scale + assemble [N, K] ------------------
    // quant bytes = blocks[:, 2:18]  → [tb,16], values 0..255.
    let qbytes = op(T::Slice, slice(1, 2, 16), vec![blocks]); // [tb,16]
    let high = floor(mul_s(1.0 / 16.0, qbytes.clone())); // qs >> 4  (0..15)
    let low = sub(qbytes, mul_s(16.0, high.clone())); // qs & 0x0F (0..15)
    let x0 = add_s(-8.0, low); // low nibble  − 8  → [tb,16]
    let x1 = add_s(-8.0, high); // high nibble − 8  → [tb,16]
    // to_float layout: weights[0..16] = x0, weights[16..32] = x1.
    let codes = op(T::Concat, cat(1), vec![x0, x1]); // [tb,32]
    let scale_bc = op(T::BroadcastTo, shape_attr(&[total_blocks, QK4_0]), vec![scale]); // [tb,32]
    let dequant_blocks = mul(codes, scale_bc); // [tb,32] F32
    // Row-major reshape: block (row, kb) tiles column kb·32..+32 of row.
    let weight_nk = op(T::Reshape, shape_attr(&[n, k]), vec![dequant_blocks]); // [N,K]
    let weight_kn = op(T::Transpose, OpAttrs::default(), vec![weight_nk]); // [K,N]

    // --- 4. matmul (product-collapse leading activation dims) --------------
    let a2 = op(T::Reshape, shape_attr(&[m_prime, k]), vec![bind(0)]); // [M',K]
    let out2 = op(T::MatMul, OpAttrs::default(), vec![a2, weight_kn]); // [M',N]
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n);
    Some(op(T::Reshape, shape_attr(&out_dims), vec![out2]))
}

/// Lower a fused QMatMul node to its `dequantize → matmul` primitive subgraph
/// and return the new root id.
///
/// Config-branches on `quant_type`: **Q4_0** lowers via [`recipe_q4_0`] (the
/// only format the live loader produces); every other `QuantType` self-returns
/// as a surfaced gap (a tracked ROADMAP item — never a crash). Per G2 this is
/// total + never-panic: wrong params, a malformed node, or any recipe/bridge
/// decline all return `id` (the driver's fixpoint signal). No open scalar slots
/// — every constant is baked — so the bridge gets `Some(Vec::new())`.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let (quant_type, k, n) = match params {
        FusedOpParams::QMatMul { quant_type, k, n } => (*quant_type, *k, *n),
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };
    let (a_shape, w_len) = {
        let node = graph.node(id);
        // Malformed node → fixpoint self-return (never panic).
        if node.inputs.len() != 2 {
            return id;
        }
        let a_shape = graph.node(node.inputs[0]).shape.clone();
        let w_len = graph.node(node.inputs[1]).shape.elem_count();
        (a_shape, w_len)
    };

    // Only Q4_0 carries a recipe today; other formats are surfaced gaps
    // (ROADMAP: qmatmul per-format decompose build-out).
    let recipe_node = match quant_type {
        QuantType::Q4_0 => recipe_q4_0(&a_shape, w_len, k, n),
        _ => None,
    };
    let recipe_node = match recipe_node {
        Some(r) => r,
        None => return id, // typed decline → fixpoint
    };
    decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()))
}

/// Matcher stub — QMatMul nodes originate from explicit quantized
/// builders, not user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Node, Op};

    /// The f16-scale decode's arithmetic mirror — the SAME operation sequence
    /// the recipe emits as nodes (see `recipe_q4_0` step 2), in scalar f32.
    /// Because every intermediate is a small exact integer or an exact power of
    /// two, f32 evaluates it without rounding, so it equals `f16::to_f32` for
    /// every finite half. Kept beside the recipe as the spec its nodes track.
    fn decode_f16_bits_arith(bits: f32) -> f32 {
        let sign = (bits / 32768.0).floor();
        let rem = bits - 32768.0 * sign;
        let exp = (rem / 1024.0).floor();
        let mant = rem - 1024.0 * exp;
        let signed = 1.0 - 2.0 * sign;
        // 2^exp via 5-bit binary decomposition.
        let h0 = (exp * 0.5).floor();
        let bit0 = exp - 2.0 * h0;
        let h1 = (h0 * 0.5).floor();
        let bit1 = h0 - 2.0 * h1;
        let h2 = (h1 * 0.5).floor();
        let bit2 = h1 - 2.0 * h2;
        let h3 = (h2 * 0.5).floor();
        let bit3 = h2 - 2.0 * h3;
        let h4 = (h3 * 0.5).floor();
        let bit4 = h3 - 2.0 * h4;
        let pow2exp = (1.0 + bit0)
            * (1.0 + 3.0 * bit1)
            * (1.0 + 15.0 * bit2)
            * (1.0 + 255.0 * bit3)
            * (1.0 + 65535.0 * bit4);
        let normal_mag = (1024.0 + mant) * pow2exp * 2f32.powi(-25);
        let sub_mag = mant * 2f32.powi(-24);
        let is_sub = (1.0 - exp).max(0.0);
        let mag = is_sub * sub_mag + (1.0 - is_sub) * normal_mag;
        signed * mag
    }

    /// The load-bearing correctness anchor: the arithmetic f16 decode is
    /// **bit-exact** to `half::f16::to_f32` for every finite half (all
    /// non-inf/NaN bit patterns). GGUF scales are always finite, so exp==31 is
    /// out of scope and skipped. Born-red against any error in the field
    /// extraction or the power-of-two decomposition.
    #[test]
    fn f16_decode_arithmetic_matches_half_for_all_finite_bit_patterns() {
        for b in 0u16..=u16::MAX {
            let exp = (b >> 10) & 0x1F;
            if exp == 0x1F {
                continue; // inf/NaN — GGUF scales are finite
            }
            let want = half::f16::from_bits(b).to_f32();
            let got = decode_f16_bits_arith(b as f32);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "f16 bits {b:#06x}: got {got}, want {want}",
            );
        }
    }

    /// Build a fused Q4_0 QMatMul node over `activations [..leading, K]` F32 and
    /// a U32 weight stream of the right length. Returns the fused NodeId.
    fn fused_q4_0_node(g: &mut Graph, leading: &[usize], k: usize, n: usize) -> NodeId {
        let mut a_dims = leading.to_vec();
        a_dims.push(k);
        let act = g.push(Node {
            op: Op::Const, inputs: vec![], shape: Shape::from_dims(&a_dims), dtype: DType::F32,
        });
        let total_blocks = n * (k / QK4_0);
        let total_bytes = total_blocks * Q4_0_BYTES_PER_BLOCK;
        let w = g.push(Node {
            op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[total_bytes / 4]), dtype: DType::U32,
        });
        let mut out_dims = leading.to_vec();
        out_dims.push(n);
        g.push(Node {
            op: Op::Fused(
                FusedOps::QMATMUL,
                FusedOpParams::QMatMul { quant_type: QuantType::Q4_0, k, n },
            ),
            inputs: vec![act, w],
            shape: Shape::from_dims(&out_dims),
            dtype: DType::F32,
        })
    }

    /// Q4_0 decompose fires (non-fixpoint) and yields a graph whose output shape
    /// and dtype match the shape/dtype rules, across leading-dim configs (the
    /// product-collapsed `M'`). Born-red with the recipe absent (a
    /// fixpoint-returning `decompose`): `new_root == fused` trips the assert_ne.
    /// Value parity is the real-backend oracle in
    /// `fuel-core/tests/incc_qmatmul_q4_0_oracle.rs` (fuel-graph can't realize).
    #[test]
    fn qmatmul_q4_0_decompose_fires_and_is_shape_correct() {
        for leading in [vec![1usize], vec![4usize], vec![2usize, 3]] {
            let k = 64; // 2 blocks per row
            let n = 5;
            let params = FusedOpParams::QMatMul { quant_type: QuantType::Q4_0, k, n };
            let mut g = Graph::new();
            let fused = fused_q4_0_node(&mut g, &leading, k, n);
            let out_sh = g.node(fused).shape.clone();

            let new_root = decompose(&mut g, fused, &params);
            assert_ne!(new_root, fused, "Q4_0 recipe decompose fires (leading={leading:?})");
            assert_eq!(g.node(new_root).shape, out_sh, "output shape matches shape_rule");
            assert_eq!(g.node(new_root).dtype, DType::F32, "output dtype is F32");
        }
    }

    /// Totality (G2): a non-Q4_0 format is a surfaced gap — self-return
    /// fixpoint, no emission, never a crash (the ROADMAP build-out target).
    #[test]
    fn qmatmul_nonq4_0_format_is_a_surfaced_gap_fixpoint() {
        let mut g = Graph::new();
        let k = 64;
        let n = 3;
        let act = g.push(Node {
            op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2, k]), dtype: DType::F32,
        });
        let total_bytes = n * (k / 32) * 34; // Q8_0 bytes/block
        let w = g.push(Node {
            op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[total_bytes / 4]), dtype: DType::U32,
        });
        let fused = g.push(Node {
            op: Op::Fused(FusedOps::QMATMUL, FusedOpParams::QMatMul { quant_type: QuantType::Q8_0, k, n }),
            inputs: vec![act, w],
            shape: Shape::from_dims(&[2, n]),
            dtype: DType::F32,
        });
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::QMatMul { quant_type: QuantType::Q8_0, k, n });
        assert_eq!(out, fused, "non-Q4_0 => surfaced gap => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint before any
    /// emission, never a crash.
    #[test]
    fn qmatmul_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_q4_0_node(&mut g, &[2], 64, 5);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}

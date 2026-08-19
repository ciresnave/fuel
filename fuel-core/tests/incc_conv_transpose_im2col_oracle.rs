// SPDX-License-Identifier: MIT OR Apache-2.0
//! CV3 (Increment C, im2col-3): real-backend numerical parity for the migrated
//! `ConvTranspose2D` col2im (overlap-add) scatter-add recipe.
//!
//! `ConvTranspose2D` self-returned before this slice, so there is no
//! frozen-legacy structure to preserve bit-exactly, and the toy f64 parity
//! interpreter does not implement `IndexAdd`/`Iota`/`MatMul`/`Slice`. `fuel-conv`
//! has no transposed reference either, so — per the plan's RISK-B — the oracle is
//! the **native CPU ConvTranspose2D kernel** itself: the production realize path
//! (`realize_f32`) binds the fused node to that kernel and NEVER invokes the
//! decompose (lowering only fires via `RuleRegistry::lowering_only`), so the two
//! paths are independent.
//!
//! The gate: build the SAME data twice; realize one directly (native kernel =
//! oracle) and lower the other with `lowering_only` (fires the CV3 decompose) +
//! realize its primitive subgraph on the ACTUAL CPU backend (the recipe =
//! subject), then compare. The col2im matmul reorders the accumulation vs the
//! kernel's per-output `vec_dot`, so the bound is `rel<1e-5` (calibrated, not
//! bit-exact). It is sabotage-calibrated: corrupting the scatter index or the
//! matmul-arrange drives the relative error to O(1) (the oracle stays the
//! uncorrupted native kernel), and the structural assertion (no
//! `Op::Fused(CONV_TRANSPOSE2D)` survives lowering; the recipe realizes via
//! `IndexAdd` + `MatMul`) is the born-red half.

use fuel_core::Device;
use fuel_core::lazy::LazyTensor;
use fuel_core::pipelined_bridge::realize_one_as;
use fuel_graph::Op;
use fuel_graph::opt::RuleRegistry;
use fuel_graph::registry::FusedOps;
use fuel_ir::Shape;

/// Lower + realize one conv_transpose config on the CPU backend and compare to
/// the native-kernel oracle. `weight` is `[Cin, Cout/groups, Kh, Kw]`.
#[allow(clippy::too_many_arguments)]
fn check_case(
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout_per_g: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) {
    let dev = Device::cpu();
    let label = format!(
        "conv_transpose2d(n={n} cin={cin} {h}x{w} cpg={cout_per_g} k={kh}x{kw} \
         s={stride:?} p={padding:?} op={output_padding:?} d={dilation:?} g={groups})"
    );

    // Deterministic, well-scaled data (smooth sinusoids like the conv2d oracle).
    let x_data: Vec<f32> = (0..n * cin * h * w)
        .map(|i| ((i as f32) * 1.3e-3).sin())
        .collect();
    let w_len = cin * cout_per_g * kh * kw;
    let w_data: Vec<f32> = (0..w_len).map(|i| ((i as f32) * 1.7e-3).cos()).collect();

    let build = |dev: &Device| {
        let x = LazyTensor::from_f32(x_data.clone(), Shape::from_dims(&[n, cin, h, w]), dev);
        let weight = x.const_f32_like(w_data.clone(), Shape::from_dims(&[cin, cout_per_g, kh, kw]));
        x.conv_transpose2d(&weight, stride, padding, output_padding, dilation, groups)
            .unwrap_or_else(|e| panic!("{label}: conv_transpose2d build failed: {e:?}"))
    };

    // --- ORACLE: native CPU kernel via the default (non-lowering) realize path. ---
    let y_oracle = build(&dev);
    let expected = y_oracle.realize_f32();
    let out_dims = y_oracle.shape().dims().to_vec();

    // --- SUBJECT: a fresh graph, lowered to the col2im recipe, then realized. ---
    let y_sub = build(&dev);
    let t = y_sub.graph_tensor();
    let graph = t.graph().clone();
    let id = t.id();
    let roots = RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
    assert_eq!(roots.len(), 1, "{label}: lowering keeps a single root");

    // Structural (born-red): no Op::Fused(CONV_TRANSPOSE2D) survives; the recipe
    // realizes via IndexAdd (the overlap-add) + MatMul (the column stack).
    {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]];
        let mut seen = std::collections::HashSet::new();
        let (mut saw_matmul, mut saw_index_add) = (false, false);
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) {
                continue;
            }
            let node = g.node(nid);
            assert!(
                !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::CONV_TRANSPOSE2D),
                "{label}: ConvTranspose2D must lower to primitives, not remain fused",
            );
            if matches!(node.op, Op::MatMul) {
                saw_matmul = true;
            }
            if matches!(node.op, Op::IndexAdd { .. }) {
                saw_index_add = true;
            }
            for &inp in &node.inputs {
                stack.push(inp);
            }
        }
        assert!(
            saw_matmul && saw_index_add,
            "{label}: recipe realizes via MatMul + IndexAdd (matmul={saw_matmul}, index_add={saw_index_add})",
        );
    }

    let got = realize_one_as::<f32>(&graph, roots[0], &dev)
        .unwrap_or_else(|e| panic!("{label}: realize col2im recipe on CPU failed: {e:?}"));

    // Shape sanity: the lowered root keeps [N, Cout, Hout, Wout].
    let cout = cout_per_g * groups;
    assert_eq!(out_dims.len(), 4, "{label}: oracle output rank 4",);
    assert_eq!(out_dims[0], n, "{label}: N");
    assert_eq!(out_dims[1], cout, "{label}: Cout");
    assert_eq!(got.len(), expected.len(), "{label}: output length");

    let mut max_rel = 0.0f32;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let denom = g.abs().max(e.abs()).max(f32::MIN_POSITIVE);
        let rel = (g - e).abs() / denom;
        max_rel = max_rel.max(rel);
        assert!(
            rel < 1e-5,
            "{label}: col2im recipe vs native kernel mismatch at {i}: got {g}, want {e}, rel {rel}",
        );
    }
    assert!(
        max_rel < 1e-5,
        "{label}: max relative error {max_rel} exceeds 1e-5"
    );
}

/// Dense groups=1 transposed conv — the plain col2im overlap-add.
#[test]
fn conv_transpose2d_col2im_matches_native_dense() {
    check_case(1, 2, 4, 4, 3, 2, 2, (1, 1), (0, 0), (0, 0), (1, 1), 1);
    check_case(2, 3, 5, 6, 4, 3, 3, (1, 1), (0, 0), (0, 0), (1, 1), 1);
}

/// Stride / padding / output_padding exercise the full-buffer scatter positions
/// and the crop (`out = inp·stride + k·dilation`, cropped by `padding`, extended
/// by `output_padding`). Stride>1 makes overlapping columns land on distinct
/// positions; the crop drops the padding border.
#[test]
fn conv_transpose2d_col2im_matches_native_stride_pad_outpad() {
    check_case(1, 3, 5, 5, 2, 3, 3, (2, 2), (1, 1), (1, 1), (1, 1), 1); // the audio-codec upsample shape
    check_case(1, 2, 6, 5, 3, 2, 3, (2, 1), (1, 2), (1, 0), (1, 1), 1); // asymmetric everything
    check_case(2, 2, 4, 4, 2, 2, 2, (1, 1), (0, 0), (0, 0), (2, 1), 1); // dilation
}

/// Grouped + depthwise transposed conv — the rank-4 batched col2im (a distinct
/// weight arrange + matmul rank; the scatter index and crop are shared).
#[test]
fn conv_transpose2d_col2im_matches_native_grouped() {
    check_case(1, 4, 4, 4, 3, 2, 2, (1, 1), (0, 0), (0, 0), (1, 1), 2); // groups=2
    check_case(2, 4, 3, 3, 1, 2, 2, (2, 2), (0, 0), (1, 1), (1, 1), 4); // depthwise groups=Cin
    check_case(1, 6, 5, 5, 2, 3, 3, (1, 1), (1, 1), (0, 0), (1, 1), 3); // grouped, multiplier + pad
}

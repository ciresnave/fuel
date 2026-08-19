//! CV1 (Increment C, im2col-1): real-backend numerical parity for the migrated
//! `Conv2D` (groups=1) index-gather im2col recipe.
//!
//! `Conv2D` self-returned before this slice, so there is no frozen-legacy
//! structure to preserve bit-exactly, and the toy f64 parity interpreter does
//! not implement `IndexSelect`/`Iota`/`MatMul`/`Pad`. Per the plan's RISK-B, the
//! gate is a **real-backend CPU-realize** numerical test: lower the fused node
//! with `lowering_only` (which fires the new `decompose`), realize the resulting
//! primitive subgraph on the ACTUAL CPU backend, and compare to
//! `fuel_conv::conv2d_direct` — the same rel<1e-5 bound `via_gemm_matches_direct`
//! already meets (MatMul reorders the contraction vs the nested-loop reference,
//! so it is calibrated, not bit-exact).
//!
//! The structural assertion (no `Op::Fused(CONV2D)` survives lowering; the
//! recipe realizes via `IndexSelect` + `MatMul`) is the born-red half: before
//! the recipe existed, `lowering_only` left the node fused and it tripped. The
//! numeric bound is sabotage-calibrated — corrupting the gather index or the
//! matmul operands drives the relative error far above 1e-5.

use fuel_conv::{ConvShape, conv2d_direct};
use fuel_core::Device;
use fuel_core::lazy::LazyTensor;
use fuel_core::pipelined_bridge::realize_one_as;
use fuel_graph::Op;
use fuel_graph::registry::FusedOps;
use fuel_ir::Shape;

/// Lower + realize one groups=1 conv config on the CPU backend and compare to
/// the direct-conv oracle. `bias` toggles the optional broadcast-`Add` tail.
#[allow(clippy::too_many_arguments)]
fn check_case(
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    bias: bool,
) {
    let dev = Device::cpu();
    let label = format!(
        "conv2d(n={n} cin={cin} {h}x{w} cout={cout} k={kh}x{kw} s={stride:?} p={padding:?} bias={bias})"
    );

    // Deterministic, well-scaled data (smooth sinusoids like phase7b).
    let x_data: Vec<f32> = (0..n * cin * h * w)
        .map(|i| ((i as f32) * 1.3e-3).sin())
        .collect();
    let w_data: Vec<f32> = (0..cout * cin * kh * kw)
        .map(|i| ((i as f32) * 1.7e-3).cos())
        .collect();
    let b_data: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.05 - 0.1).collect();

    let x = LazyTensor::from_f32(x_data.clone(), Shape::from_dims(&[n, cin, h, w]), &dev);
    let weight = x.const_f32_like(w_data.clone(), Shape::from_dims(&[cout, cin, kh, kw]));
    let bias_t = if bias {
        Some(x.const_f32_like(b_data.clone(), Shape::from_dims(&[cout])))
    } else {
        None
    };
    let y = x
        .conv2d(&weight, bias_t.as_ref(), stride, padding, 1)
        .unwrap_or_else(|e| panic!("{label}: conv2d build failed: {e:?}"));

    // Lower the fused node to its recipe (fires the CV1 decompose).
    let t = y.graph_tensor();
    let graph = t.graph().clone();
    let id = t.id();
    let roots = fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
    assert_eq!(roots.len(), 1, "{label}: lowering keeps a single root");

    // Structural (born-red): no Op::Fused(CONV2D) survives; the recipe realizes
    // via IndexSelect (im2col) + MatMul (the GEMM).
    {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]];
        let mut seen = std::collections::HashSet::new();
        let (mut saw_matmul, mut saw_gather) = (false, false);
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) {
                continue;
            }
            let node = g.node(nid);
            assert!(
                !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::CONV2D),
                "{label}: Conv2D must lower to primitives, not remain fused",
            );
            if matches!(node.op, Op::MatMul) {
                saw_matmul = true;
            }
            if matches!(node.op, Op::IndexSelect { .. }) {
                saw_gather = true;
            }
            for &inp in &node.inputs {
                stack.push(inp);
            }
        }
        assert!(
            saw_matmul && saw_gather,
            "{label}: recipe realizes via IndexSelect + MatMul (matmul={saw_matmul}, gather={saw_gather})",
        );
    }

    let got = realize_one_as::<f32>(&graph, roots[0], &dev)
        .unwrap_or_else(|e| panic!("{label}: realize im2col recipe on CPU failed: {e:?}"));

    // Direct-conv oracle over the SAME data.
    let s = ConvShape {
        batch: n,
        c_in: cin,
        c_out: cout,
        h,
        w,
        k_h: kh,
        k_w: kw,
        stride,
        padding,
        groups: 1,
    };
    let mut expected = vec![0.0f32; s.output_len()];
    conv2d_direct(
        &x_data,
        &w_data,
        if bias { Some(b_data.as_slice()) } else { None },
        &s,
        &mut expected,
    );

    assert_eq!(got.len(), expected.len(), "{label}: output length");
    // Same rel bound as `via_gemm_matches_direct` (MatMul reorders the sum).
    let mut max_rel = 0.0f32;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let denom = g.abs().max(e.abs()).max(f32::MIN_POSITIVE);
        let rel = (g - e).abs() / denom;
        max_rel = max_rel.max(rel);
        assert!(
            rel < 1e-5,
            "{label}: im2col recipe vs direct mismatch at {i}: got {g}, want {e}, rel {rel}",
        );
    }
    // The correct recipe lands orders of magnitude below the bound; a sabotaged
    // index/matmul drives max_rel to O(1) (documented teeth).
    assert!(
        max_rel < 1e-5,
        "{label}: max relative error {max_rel} exceeds 1e-5"
    );
}

/// Dense groups=1 conv, no bias — the plain im2col + GEMM path.
#[test]
fn conv2d_im2col_recipe_matches_direct_dense() {
    check_case(1, 2, 5, 5, 3, 2, 2, (1, 1), (0, 0), false);
    check_case(2, 3, 7, 6, 4, 3, 3, (1, 1), (0, 0), false);
}

/// Padding + stride exercise the `(oh·sh + ky)·Wpad + (ow·sw + kx)` index
/// against the padded, flattened spatial axis (zero-fill windows read the
/// pad region, matching the direct-conv "contributes 0" convention).
#[test]
fn conv2d_im2col_recipe_matches_direct_stride_pad() {
    check_case(1, 4, 8, 8, 6, 3, 3, (1, 1), (1, 1), false);
    check_case(2, 3, 9, 9, 5, 3, 3, (2, 2), (1, 1), false); // stride-2, pad-1
    check_case(1, 2, 6, 7, 3, 2, 3, (2, 1), (1, 2), false); // asymmetric stride+pad
}

/// The optional bias broadcast-`Add` tail (3-input recipe).
#[test]
fn conv2d_im2col_recipe_matches_direct_with_bias() {
    check_case(1, 3, 6, 6, 4, 3, 3, (1, 1), (1, 1), true);
    check_case(2, 2, 5, 5, 3, 2, 2, (1, 1), (0, 0), true);
}

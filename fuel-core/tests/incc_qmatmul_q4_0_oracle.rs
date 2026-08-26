// SPDX-License-Identifier: MIT OR Apache-2.0
//! Increment C (qmatmul-Q4_0): real-backend numerical parity for the migrated
//! `QMatMul` Q4_0 `dequant → matmul` recipe.
//!
//! `QMatMul` self-returned before this slice, so there is no frozen-legacy
//! structure to preserve bit-exactly. Per the conv-oracle precedent the gate is
//! a **real-backend CPU-realize** numerical test: lower the fused node with
//! `lowering_only` (which fires the new `decompose`), realize the resulting
//! primitive subgraph on the ACTUAL CPU backend, and compare to an EXACT
//! dequantize-then-matmul reference built from `fuel_quantized::BlockQ4_0`.
//!
//! The recipe's dequant is **bit-exact** to `BlockQ4_0::to_float` (the f16-scale
//! decode is validated against `half::f16` for every finite bit pattern in
//! `fuel-graph`), so the only slack is the GEMM contraction order — the same
//! `rel < 1e-5` bound `conv2d_im2col_recipe_matches_direct_*` meets (MatMul
//! reorders the sum vs the reference loop, so it is calibrated, not bit-exact).
//! The bound is sabotage-calibrated: corrupting the f16 decode, the nibble
//! unpack, or the block layout drives the relative error far above 1e-5.
//!
//! The structural assertion (no `Op::Fused(QMATMUL)` survives lowering; the
//! recipe realizes via `Slice` + `MatMul`) is the born-red half: before the
//! recipe existed, `lowering_only` left the node fused and it tripped.

use fuel_core::Device;
use fuel_core::lazy::Tensor;
use fuel_core::pipelined_bridge::realize_one_as;
use fuel_graph::Op;
use fuel_graph::QuantType;
use fuel_graph::registry::FusedOps;
use fuel_ir::Shape;
use fuel_quantized::{BlockQ4_0, GgmlType};

/// Lower + realize one Q4_0 qmatmul config on the CPU backend and compare to the
/// exact `to_float` dequant → reference matmul. `leading` are the activations'
/// leading (batch) dims; the GEMM is `[M', K] @ dequant(W)^T` with `M' = ∏
/// leading`.
fn check_q4_0(leading: &[usize], k: usize, n: usize) {
    let dev = Device::cpu();
    let m: usize = leading.iter().product();
    let label = format!("qmatmul_q4_0(leading={leading:?} k={k} n={n})");

    // --- deterministic, well-scaled weight; quantize to real Q4_0 blocks. ---
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.021).sin() * 0.7)
        .collect();
    let blocks_per_row = k / BlockQ4_0::BLCK_SIZE;
    let mut w_blocks = vec![BlockQ4_0::zeros(); n * blocks_per_row];
    BlockQ4_0::from_float(&w_f32, &mut w_blocks);

    // Reinterpret the #[repr(C)] block slice as raw bytes, then as the U32
    // stream the loader/builder expect (verbatim, little-endian).
    let bytes_per_block = std::mem::size_of::<BlockQ4_0>();
    let w_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            w_blocks.as_ptr() as *const u8,
            w_blocks.len() * bytes_per_block,
        )
    }
    .to_vec();
    assert_eq!(
        w_bytes.len() % 4,
        0,
        "{label}: block bytes must pack into u32"
    );
    let w_u32: Vec<u32> = w_bytes
        .as_chunks::<4>().0.iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Exact dequantized weight [N, K] — the oracle for the recipe's dequant.
    let mut deq = vec![0f32; n * k];
    BlockQ4_0::to_float(&w_blocks, &mut deq);

    // --- build the fused qmatmul node via the Tensor builder. ----------
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let mut a_dims = leading.to_vec();
    a_dims.push(k);
    let x = Tensor::from_f32(a_data.clone(), Shape::from_dims(&a_dims), &dev);
    let w = x.const_u32_like(w_u32, Shape::from_dims(&[w_bytes.len() / 4]));
    let y = x
        .qmatmul(&w, QuantType::Q4_0, k, n)
        .unwrap_or_else(|e| panic!("{label}: qmatmul build failed: {e:?}"));

    // --- lower the fused node to its recipe (fires the decompose). ---------
    let t = y.graph_tensor();
    let graph = t.graph().clone();
    let id = t.id();
    let roots = fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[id]);
    assert_eq!(roots.len(), 1, "{label}: lowering keeps a single root");

    // Structural (born-red): no Op::Fused(QMATMUL) survives; the recipe
    // realizes via Slice (block byte extraction) + MatMul (the GEMM).
    {
        let g = graph.read().unwrap();
        let mut stack = vec![roots[0]];
        let mut seen = std::collections::HashSet::new();
        let (mut saw_matmul, mut saw_slice) = (false, false);
        while let Some(nid) = stack.pop() {
            if !seen.insert(nid) {
                continue;
            }
            let node = g.node(nid);
            assert!(
                !matches!(node.op, Op::Fused(fid, _) if fid == FusedOps::QMATMUL),
                "{label}: QMatMul must lower to primitives, not remain fused",
            );
            if matches!(node.op, Op::MatMul) {
                saw_matmul = true;
            }
            if matches!(node.op, Op::Slice { .. }) {
                saw_slice = true;
            }
            for &inp in &node.inputs {
                stack.push(inp);
            }
        }
        assert!(
            saw_matmul && saw_slice,
            "{label}: recipe realizes via Slice + MatMul (matmul={saw_matmul}, slice={saw_slice})",
        );
    }

    let got = realize_one_as::<f32>(&graph, roots[0], &dev)
        .unwrap_or_else(|e| panic!("{label}: realize Q4_0 recipe on CPU failed: {e:?}"));

    // --- exact reference: dequant(W) @ activations → [M, N]. ---------------
    // out[mi, ni] = Σ_k a[mi, k] · deq[ni, k]   (independent nested-loop sum).
    let mut expected = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a_data[mi * k + kk] * deq[ni * k + kk];
            }
            expected[mi * n + ni] = s;
        }
    }

    assert_eq!(got.len(), expected.len(), "{label}: output length");
    let mut max_rel = 0.0f32;
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let denom = g.abs().max(e.abs()).max(f32::MIN_POSITIVE);
        let rel = (g - e).abs() / denom;
        max_rel = max_rel.max(rel);
        assert!(
            rel < 1e-5,
            "{label}: recipe vs exact-dequant mismatch at {i}: got {g}, want {e}, rel {rel}",
        );
    }
    // The correct recipe lands orders of magnitude below the bound; a sabotaged
    // f16 decode / nibble unpack / layout drives max_rel to O(1) (the teeth).
    assert!(
        max_rel < 1e-5,
        "{label}: max relative error {max_rel} exceeds 1e-5"
    );
}

/// Single-block-per-row (K=32), then multi-block rows and product-collapsed
/// leading dims — exercises the f16 decode across real quantized scales, the
/// nibble unpack, the per-block broadcast, and the `M'` collapse.
#[test]
fn qmatmul_q4_0_recipe_matches_exact_dequant() {
    check_q4_0(&[1], 32, 4); // 1 block/row, single M
    check_q4_0(&[3], 64, 5); // 2 blocks/row
    check_q4_0(&[2, 3], 96, 6); // 3 blocks/row, product-collapsed M'=6
}

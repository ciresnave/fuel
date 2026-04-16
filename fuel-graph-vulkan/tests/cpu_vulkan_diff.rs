//! Differential correctness test: run the same graph through
//! `CpuBackend` and `VulkanBackend`, assert element-wise agreement.
//!
//! Validates the async-submission refactor didn't introduce memory
//! ordering races — the failure mode for a missing pipeline barrier
//! isn't a crash, it's silently-wrong output. This test would catch
//! that.
//!
//! Marked `#[ignore]` because it requires a Vulkan-capable device
//! (not available in all CI environments). Invoke explicitly with:
//!
//! ```sh
//! cargo test -p fuel-graph-vulkan --test cpu_vulkan_diff -- --ignored --nocapture
//! ```

use fuel_core_types::Shape;
use fuel_graph::Tensor;
use fuel_graph_cpu::CpuBackend;
use fuel_graph_executor::GraphExecutor;
use fuel_graph_vulkan::{DeviceSelection, VulkanBackend};

fn almost_equal(a: &[f32], b: &[f32], tol: f32) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        // Relative tolerance for non-tiny values, absolute for near-zero.
        let rel_ok = d <= tol * x.abs().max(y.abs()).max(1.0);
        let abs_ok = d <= tol;
        if !rel_ok && !abs_ok {
            return Err(format!(
                "mismatch at {i}: cpu={x} vulkan={y} diff={d}"
            ));
        }
    }
    Ok(())
}

/// Build a graph that exercises the ops with cross-op data
/// dependencies — this is specifically the case that requires
/// pipeline barriers between submits to stay correct.
fn build_graph() -> Tensor {
    // (x @ w) → silu → softmax along last dim. All tensors on one graph.
    let x = Tensor::from_f32(
        (0..24).map(|i| (i as f32) * 0.1 - 1.0).collect::<Vec<_>>(),
        Shape::from_dims(&[4, 6]),
    );
    let w = x.const_f32_like(
        (0..24).map(|i| ((i * 7) % 13) as f32 * 0.05 - 0.4).collect::<Vec<_>>(),
        Shape::from_dims(&[6, 4]),
    );
    let xw = x.matmul(&w);
    let y = xw.silu();
    y.softmax_last_dim()
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_small_mlp() {
    // Make two structurally-identical graphs (same topology) so both
    // backends see the same tensor.graph() independently.
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            return;
        }
    };

    let cpu_root = build_graph();
    let mut cpu_exec = GraphExecutor::new(CpuBackend);
    let cpu_out = cpu_exec.realize_f32(&cpu_root);
    let cpu_data = cpu_out.as_slice().to_vec();

    let vk_root = build_graph();
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_out = vk_exec.realize_f32(&vk_root);
    let vk_data = vk_out.as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-3).expect("cpu vs vulkan mismatch");

    let stats = vk_exec.backend.op_stats_snapshot();
    eprintln!("Vulkan op stats (host-side submit time):");
    for (name, s) in stats {
        let avg_us = if s.count == 0 {
            0
        } else {
            (s.total_ns / s.count as u128) / 1000
        };
        eprintln!("  {name:16} count={:>6} total={:>8}ms avg={avg_us}us",
            s.count, s.total_ns / 1_000_000);
    }
}

/// Exercises the new native per-dim reduce kernel and native
/// index_select. If the new kernels are wrong (off-by-one, missing
/// barrier, wrong shape unflatten) this test fails with a concrete
/// element mismatch rather than gibberish llama output.
fn build_rmsnorm_like_graph() -> Tensor {
    // [seq=3, hidden=8] → sum along last dim → [3] (one row sum per seq)
    // Then use that plus the original for a simple "normalize-ish"
    // shape: x - mean(x). Exercises SumDim(last) + BroadcastTo + Sub.
    let x = Tensor::from_f32(
        (0..24).map(|i| (i as f32) * 0.1 - 0.5).collect::<Vec<_>>(),
        Shape::from_dims(&[3, 8]),
    );
    // mean along last dim
    let sum = x.sum_dim(1);
    let mean = sum.mul_scalar(1.0 / 8.0);
    // broadcast back to [3, 8] and subtract
    let mean_row = mean.reshape(Shape::from_dims(&[3, 1]));
    let mean_b = mean_row.broadcast_to(Shape::from_dims(&[3, 8]));
    x.sub(&mean_b)
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_rmsnorm_like() {
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let cpu_root = build_rmsnorm_like_graph();
    let mut cpu_exec = GraphExecutor::new(CpuBackend);
    let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

    let vk_root = build_rmsnorm_like_graph();
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-4).expect("reduce_last_dim mismatch");
}

/// Mimics TinyLlama's RMSNorm shape exactly: [1, seq=5, hidden=2048],
/// sum_dim along the last axis. This is the shape that's crashing the
/// GPU in the real demo. If this test crashes too, the bug is in the
/// shader at this input scale. If this test passes, the bug is in
/// how the op integrates with other ops during a full forward.
fn build_tinyllama_scale_reduce() -> Tensor {
    let seq = 5;
    let hidden = 2048;
    let n = seq * hidden;
    let x = Tensor::from_f32(
        (0..n).map(|i| ((i as f32) * 0.001).sin()).collect::<Vec<_>>(),
        Shape::from_dims(&[1, seq, hidden]),
    );
    // Just square + sum along last dim (matches RMSNorm's inner loop).
    x.sqr().sum_dim(2)
}

/// Chains many sequential reduce_last_dim calls, with each one's
/// output feeding the next (through broadcast-back and elementwise
/// ops). TinyLlama does ~45 reduces per forward. If the bug is about
/// async-queue buildup across many reduces, this should repro.
fn build_chained_rmsnorm_graph(n_norms: usize) -> Tensor {
    let seq = 5;
    let hidden = 2048;
    let n = seq * hidden;
    let mut x = Tensor::from_f32(
        (0..n).map(|i| ((i as f32) * 0.001).sin()).collect::<Vec<_>>(),
        Shape::from_dims(&[1, seq, hidden]),
    );
    for _ in 0..n_norms {
        // RMSNorm-ish: sqrt(mean(x^2) + eps), then x / rms.
        let sqr = x.sqr();
        let sum = sqr.sum_dim(2);
        let mean = sum.mul_scalar(1.0 / hidden as f64);
        let with_eps = mean.add_scalar(1e-6);
        let rms = with_eps.sqrt();
        let rms_r = rms.reshape(Shape::from_dims(&[1, seq, 1]));
        let rms_b = rms_r.broadcast_to(Shape::from_dims(&[1, seq, hidden]));
        x = x.div(&rms_b);
    }
    x
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_chained_rmsnorms() {
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let cpu_root = build_chained_rmsnorm_graph(45);
    let mut cpu_exec = GraphExecutor::new(CpuBackend);
    let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

    let vk_root = build_chained_rmsnorm_graph(45);
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-2).expect("chained rmsnorms mismatch");
}

/// Stress test matching the real TinyLlama forward's op count
/// (~10k dispatches). The 45-iteration test passes; if the bug is
/// about op count / descriptor pool retirement / pending-list scale,
/// this should trip it. Only runs Vulkan side — CPU comparison at
/// this scale would take forever.
#[test]
#[ignore]
fn vulkan_survives_many_chained_rmsnorms() {
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    // ~1000 iterations × ~8 ops = ~8000 dispatches, comparable to
    // a full TinyLlama forward.
    let vk_root = build_chained_rmsnorm_graph(1000);
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();
    // Just confirm finite values — we're testing survival, not correctness.
    let finite_count = vk_data.iter().filter(|v| v.is_finite()).count();
    assert!(
        finite_count > vk_data.len() / 2,
        "expected most outputs finite, got {} / {}",
        finite_count, vk_data.len()
    );
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_tinyllama_scale_reduce() {
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let cpu_root = build_tinyllama_scale_reduce();
    let mut cpu_exec = GraphExecutor::new(CpuBackend);
    let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

    let vk_root = build_tinyllama_scale_reduce();
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-2).expect("reduce_last_dim@tinyllama_scale mismatch");
}

fn build_index_select_graph() -> Tensor {
    // Embedding-like: table [vocab=10, hidden=4], ids [seq=3]
    let table = Tensor::from_f32(
        (0..40).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
        Shape::from_dims(&[10, 4]),
    );
    let ids = table.const_u32_like(
        vec![2_u32, 5, 9],
        Shape::from_dims(&[3]),
    );
    table.index_select(0, &ids)
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_index_select() {
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let cpu_root = build_index_select_graph();
    let mut cpu_exec = GraphExecutor::new(CpuBackend);
    let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

    let vk_root = build_index_select_graph();
    let mut vk_exec = GraphExecutor::new(vk_backend);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-6).expect("index_select mismatch");
}

/// Train a tiny linear-regression model with SGD on Vulkan and check
/// the fitted parameters. Exercises the whole training-step loop
/// (placeholder-Const + pre_populate + backward + realize_split +
/// storage round-trip) on a real GPU backend.
/// Direct primitive-level test of `add_assign_scaled` on both CPU
/// and Vulkan. Confirms each backend's native impl produces the
/// same result.
#[test]
#[ignore]
fn add_assign_scaled_cpu_vs_vulkan_agree() {
    use fuel_core_types::HostBuffer;
    use fuel_graph_executor::GraphBackend;

    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let n = 64;
    let dst_init: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let src_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let scale: f32 = 0.25;

    // CPU reference.
    let cpu_back = CpuBackend;
    let shape = Shape::from_dims(&[n]);
    let mut cpu_dst = cpu_back.upload(&HostBuffer::F32(dst_init.clone()), &shape).unwrap();
    let cpu_src = cpu_back.upload(&HostBuffer::F32(src_vals.clone()), &shape).unwrap();
    cpu_back.add_assign_scaled(&mut cpu_dst, &cpu_src, scale).unwrap();
    let cpu_out = match cpu_back.download(&cpu_dst).unwrap() {
        HostBuffer::F32(v) => v, _ => panic!("unexpected dtype"),
    };

    // Vulkan.
    let mut vk_dst = vk_backend.upload(&HostBuffer::F32(dst_init), &shape).unwrap();
    let vk_src = vk_backend.upload(&HostBuffer::F32(src_vals), &shape).unwrap();
    vk_backend.add_assign_scaled(&mut vk_dst, &vk_src, scale).unwrap();
    let vk_out = match vk_backend.download(&vk_dst).unwrap() {
        HostBuffer::F32(v) => v, _ => panic!("unexpected dtype"),
    };

    almost_equal(&cpu_out, &vk_out, 1e-5).expect("add_assign_scaled CPU vs Vulkan");
}

#[test]
#[ignore]
fn vulkan_trains_linear_regression_sgd() {
    use fuel_core::train::{OptimizerConfig, Parameter, TrainState};
    use fuel_core::lazy::LazyTensor;
    use std::sync::Arc;

    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let xs: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 3.0).collect();
    let mut exe = GraphExecutor::new(vk_backend);
    let params = vec![
        Parameter::new_f32("w", Shape::from_dims(&[1]), vec![0.1f32]),
        Parameter::new_f32("b", Shape::from_dims(&[1]), vec![0.1f32]),
    ];
    let mut state = TrainState::new(&params, &mut exe, OptimizerConfig::sgd(0.01)).unwrap();
    let x_arc: Arc<[f32]> = xs.clone().into();
    let y_arc: Arc<[f32]> = ys.clone().into();
    for step in 0..2000 {
        let x_arc_step = x_arc.clone();
        let y_arc_step = y_arc.clone();
        let len = xs.len();
        let _ = state.step(&mut exe, move |_graph, params| {
            let w = &params["w"];
            let b = &params["b"];
            let x = w.const_f32_like(x_arc_step, Shape::from_dims(&[len]));
            let y = w.const_f32_like(y_arc_step, Shape::from_dims(&[len]));
            let w_b = w.broadcast_to(Shape::from_dims(&[len]));
            let b_b = b.broadcast_to(Shape::from_dims(&[len]));
            let y_hat = x.mul(&w_b).add(&b_b);
            let diff = y_hat.sub(&y);
            diff.sqr().sum_all().mul_scalar(1.0 / len as f64)
        }).unwrap();
        if step == 0 || step == 1999 { /* just pace */ }
    }
    let w_final = state.param_to_host("w", &exe).unwrap()[0];
    let b_final = state.param_to_host("b", &exe).unwrap()[0];
    eprintln!("vulkan SGD final: w = {w_final}, b = {b_final}");
    let _ = LazyTensor::from_f32(vec![0.0f32], Shape::from_dims(&[1])); // silence unused-import
    assert!((w_final - 2.0).abs() < 0.1, "w={w_final}");
    assert!((b_final - 3.0).abs() < 0.5, "b={b_final}");
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_with_optimization_enabled() {
    // Sanity-check that enabling the graph optimizer on both backends
    // doesn't break agreement. CSE and algebraic simplification must
    // be semantics-preserving.
    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            return;
        }
    };

    let cpu_root = build_graph();
    let mut cpu_exec = GraphExecutor::new(CpuBackend).with_optimization(true);
    let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

    let vk_root = build_graph();
    let mut vk_exec = GraphExecutor::new(vk_backend).with_optimization(true);
    let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

    almost_equal(&cpu_data, &vk_data, 1e-3).expect("cpu vs vulkan mismatch (opt on)");
}

/// Exercises the fused Op::Rope kernel against the decomposed
/// slice+neg+concat+broadcast+mul+add path. Shapes mirror LLM
/// attention (rank 4: [batch, heads, seq, head_dim]).
fn build_rope_graph(batch: usize, heads: usize, seq: usize, head_dim: usize, seed: u32, fused: bool) -> Tensor {
    let n = batch * heads * seq * head_dim;
    let x = Tensor::from_f32(
        (0..n).map(|i| (((i as u32) ^ seed) as f32 * 1e-4).sin()).collect::<Vec<_>>(),
        Shape::from_dims(&[batch, heads, seq, head_dim]),
    );
    let cos = x.const_f32_like(
        (0..(seq * head_dim))
            .map(|i| ((i as f32) * 0.01).cos())
            .collect::<Vec<_>>(),
        Shape::from_dims(&[seq, head_dim]),
    );
    let sin = x.const_f32_like(
        (0..(seq * head_dim))
            .map(|i| ((i as f32) * 0.01).sin())
            .collect::<Vec<_>>(),
        Shape::from_dims(&[seq, head_dim]),
    );
    if fused {
        x.rope_with_tables(&cos, &sin)
    } else {
        x.rope_with_tables_decomposed(&cos, &sin)
    }
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_fused_rope() {
    for (b, h, s, d, seed) in [
        (1, 32, 1, 64, 1u32),   // TinyLlama Q decode
        (1, 4, 1, 64, 2u32),    // TinyLlama K decode
        (1, 32, 5, 64, 3u32),   // Prefill
        (2, 8, 3, 32, 4u32),    // batched
    ] {
        // CPU runs the fused op natively, Vulkan runs the fused op on GPU.
        // Both must match the decomposed CPU reference.
        let cpu_ref = build_rope_graph(b, h, s, d, seed, false);
        let mut cpu_exec = GraphExecutor::new(CpuBackend);
        let cpu_data = cpu_exec.realize_f32(&cpu_ref).as_slice().to_vec();

        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("no Vulkan device; skipping: {e:?}");
                return;
            }
        };
        let vk_root = build_rope_graph(b, h, s, d, seed, true);
        let mut vk_exec = GraphExecutor::new(vk_backend);
        let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

        almost_equal(&cpu_data, &vk_data, 1e-4)
            .unwrap_or_else(|e| panic!("fused rope b={b} h={h} s={s} d={d}: {e}"));
    }
}

#[test]
#[ignore]
fn cpu_fused_rope_matches_decomposed() {
    // Sanity check: Op::Rope on the CPU backend matches the primitive
    // decomposition. If this fails we have a formula bug.
    for (b, h, s, d, seed) in [
        (1, 32, 1, 64, 1u32),
        (2, 4, 3, 32, 7u32),
    ] {
        let fused_root = build_rope_graph(b, h, s, d, seed, true);
        let decomp_root = build_rope_graph(b, h, s, d, seed, false);
        let mut exec_a = GraphExecutor::new(CpuBackend);
        let mut exec_b = GraphExecutor::new(CpuBackend);
        let a = exec_a.realize_f32(&fused_root).as_slice().to_vec();
        let b_data = exec_b.realize_f32(&decomp_root).as_slice().to_vec();
        almost_equal(&a, &b_data, 1e-5)
            .unwrap_or_else(|e| panic!("cpu fused vs decomp b={b} h={h} s={s} d={d}: {e}"));
    }
}

/// Build a graph that ends in the gradient of `x` w.r.t. a simple
/// RMSNorm-based loss. Hoists the backward pass inside the builder
/// so the returned tensor IS the grad — we can realize it through
/// either backend and compare.
fn build_rms_norm_backward_root(rows: usize, cols: usize, seed: u32) -> Tensor {
    let n = rows * cols;
    let x = Tensor::from_f32(
        (0..n).map(|i| (((i as u32) ^ seed) as f32 * 1e-3).sin() * 0.5).collect::<Vec<_>>(),
        Shape::from_dims(&[rows, cols]),
    );
    let y = x.rms_norm_last_dim(1e-6);
    let target = x.const_f32_like(
        (0..n).map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) as f32 * 1e-4).cos()).collect::<Vec<_>>(),
        Shape::from_dims(&[rows, cols]),
    );
    let weighted = y.mul(&target);
    let loss = weighted.sum_all();
    let grads = loss.backward();
    grads.get(&x).expect("grad missing for x").clone()
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_rms_norm_backward() {
    for (rows, cols, seed) in [
        (2, 8, 1u32),
        (4, 32, 2u32),
        (1, 2048, 3u32),     // TinyLlama hidden dim
        (8, 512, 4u32),
        (3, 5632, 5u32),     // TinyLlama ffn dim
    ] {
        let cpu_grad = build_rms_norm_backward_root(rows, cols, seed);
        let mut cpu_exec = GraphExecutor::new(CpuBackend);
        let cpu_grad_data = cpu_exec.realize_f32(&cpu_grad).as_slice().to_vec();

        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("no Vulkan device; skipping: {e:?}");
                return;
            }
        };
        let vk_grad = build_rms_norm_backward_root(rows, cols, seed);
        let mut vk_exec = GraphExecutor::new(vk_backend);
        let vk_grad_data = vk_exec.realize_f32(&vk_grad).as_slice().to_vec();

        almost_equal(&cpu_grad_data, &vk_grad_data, 2e-3)
            .unwrap_or_else(|e| panic!("rms_norm_backward rows={rows} cols={cols}: {e}"));
    }
}

/// Exercises the gemv pipeline (M == 1 matmul specialization).
/// Shapes mirror LLM decode: one query row against a wide weight
/// matrix. Covers K divisible by 128 (the workgroup size) and a
/// non-divisible K to flush out bounds handling.
fn build_gemv_graph(k: usize, n: usize, k_seed: u32) -> Tensor {
    let x = Tensor::from_f32(
        (0..k).map(|i| ((i as u32 ^ k_seed) as f32).sin() * 0.1).collect::<Vec<_>>(),
        Shape::from_dims(&[1, k]),
    );
    let w = x.const_f32_like(
        (0..(k * n))
            .map(|i| (((i as u32).wrapping_mul(2654435761) ^ k_seed) as i32 as f32 * 1e-9).sin())
            .collect::<Vec<_>>(),
        Shape::from_dims(&[k, n]),
    );
    x.matmul(&w)
}

#[test]
#[ignore]
fn cpu_and_vulkan_agree_on_gemv_decode_shape() {
    for (k, n, seed) in [
        (2048, 2048, 1u32),  // TinyLlama hidden->hidden projection
        (2048, 5632, 2u32),  // TinyLlama up/gate projection
        (5632, 2048, 3u32),  // TinyLlama down projection
        (2050, 1027, 4u32),  // Non-128-divisible K and N
        (127, 64, 5u32),     // K < workgroup size
    ] {
        let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("no Vulkan device; skipping: {e:?}");
                return;
            }
        };
        let cpu_root = build_gemv_graph(k, n, seed);
        let mut cpu_exec = GraphExecutor::new(CpuBackend);
        let cpu_data = cpu_exec.realize_f32(&cpu_root).as_slice().to_vec();

        let vk_root = build_gemv_graph(k, n, seed);
        let mut vk_exec = GraphExecutor::new(vk_backend);
        let vk_data = vk_exec.realize_f32(&vk_root).as_slice().to_vec();

        almost_equal(&cpu_data, &vk_data, 2e-3)
            .unwrap_or_else(|e| panic!("gemv k={k} n={n}: {e}"));
    }
}

/// Roundtrip a bf16 buffer through the Vulkan backend: upload it,
/// download it, verify every element matches. Exercises the narrow
/// piece of M1 of the bf16 project — the host ↔ device plumbing —
/// without needing any bf16-aware shader.
#[test]
#[ignore]
fn vulkan_roundtrips_bf16_host_buffer() {
    use fuel_core_types::HostBuffer;
    use fuel_graph_executor::GraphBackend;
    use half::bf16;

    let vk_backend = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            return;
        }
    };

    // Cover a handful of representative bf16 patterns: ordinary
    // magnitudes, near-zero, large, and exactly-representable integers
    // (bf16 has the same exponent range as f32 but only 8 mantissa
    // bits — these values round-trip exactly).
    let src_f32: Vec<f32> = vec![
        0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 256.0, -256.0,
        0.5, 0.25, 1.5, 3.0, 128.0, 1024.0, 65536.0, -65536.0,
        0.1, -0.1, 3.14159, -3.14159, 1e-3, -1e-3, 1e3, -1e3,
    ];
    let src_bf16: Vec<bf16> = src_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let shape = Shape::from_dims(&[src_bf16.len()]);

    let storage = vk_backend
        .upload(&HostBuffer::BF16(src_bf16.clone()), &shape)
        .expect("bf16 upload");
    let round = vk_backend.download(&storage).expect("bf16 download");

    let HostBuffer::BF16(round_bf16) = round else {
        panic!("download returned wrong dtype: {:?}", round.dtype());
    };
    assert_eq!(
        src_bf16, round_bf16,
        "bf16 roundtrip mismatch: src={:?}, got={:?}", src_bf16, round_bf16
    );
}

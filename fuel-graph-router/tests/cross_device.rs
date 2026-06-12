//! Phase 2/3 validation: the dyn-dispatch Router can realize graphs
//! on a single device, round-trip storage across devices, and honor
//! explicit `Op::Copy` / `Op::Release` nodes inside a graph.

use fuel_core_types::{Capability, DType, DeviceLocation, HostBuffer, Layout, Result, Shape};
use fuel_graph::Tensor;
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};
use fuel_graph_router::{
    apply_placement, DynBackend, Router, RuleScheduler, Scheduler, SimpleScheduler,
};
use fuel_graph_cpu::CpuBackend;

/// Phase 7.5 G2: tests need a real device for slot-populating
/// constructors. Singleton CpuBackendDevice via OnceLock.
fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
    static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}

/// For each capability a backend advertises, invoke the corresponding
/// DynBackend method with minimal valid inputs. If the backend declared
/// the capability but the impl still bails, this returns `Err` — the
/// drift guard used by the per-backend tests below.
///
/// Inputs are small (rank-1, 4-element) where possible. Correctness of
/// the numerical result is NOT checked — parity tests cover that.
/// This only validates the declared-vs-implemented contract.
fn smoke_test_capability(backend: &dyn DynBackend, cap: Capability) -> Result<()> {
    let shape4 = Shape::from_dims(&[4]);
    let layout4 = Layout::contiguous(&shape4);

    // 2D shape for matmul: [1, 4] @ [4, 1] -> [1, 1]
    let shape_1_4 = Shape::from_dims(&[1, 4]);
    let shape_4_1 = Shape::from_dims(&[4, 1]);
    let layout_1_4 = Layout::contiguous(&shape_1_4);
    let layout_4_1 = Layout::contiguous(&shape_4_1);

    let host_f32 = HostBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]);

    match cap {
        Capability::Alloc => { backend.alloc_zeros(&shape4, DType::F32)?; }
        Capability::Upload => { backend.upload(&host_f32, &shape4)?; }
        Capability::Download => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.download(&s)?;
        }
        Capability::TryClone => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.try_clone(&s, &layout4)?;
        }
        Capability::CopyStridedSrc => {
            let s = backend.upload(&host_f32, &shape4)?;
            let mut d = backend.alloc_zeros(&shape4, DType::F32)?;
            backend.copy_strided_src(&s, &mut d, 0, &layout4)?;
        }
        Capability::MatMul => {
            let a = backend.upload(&host_f32, &shape_1_4)?;
            let b = backend.upload(&host_f32, &shape_4_1)?;
            // bmnk = (batch, m, n, k)
            backend.matmul(&a, &b, (1, 1, 1, 4), &layout_1_4, &layout_4_1)?;
        }
        Capability::Unary => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.unary(UnaryOp::Neg, &s, &layout4)?;
        }
        Capability::Binary => {
            let a = backend.upload(&host_f32, &shape4)?;
            let b = backend.upload(&host_f32, &shape4)?;
            backend.binary(BinaryOp::Add, &a, &b, &layout4, &layout4)?;
        }
        Capability::Affine => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.affine(&s, &layout4, 2.0, 1.0)?;
        }
        Capability::Powf => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.powf(&s, &layout4, 2.0)?;
        }
        Capability::Cast => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.cast(&s, &layout4, DType::F32)?;
        }
        Capability::Reduce => {
            let s = backend.upload(&host_f32, &shape4)?;
            backend.reduce(fuel_core_types::op::ReduceOp::Sum, &s, &layout4, &[0])?;
        }
        Capability::SoftmaxLastDim => {
            let s = backend.upload(&host_f32, &shape_1_4)?;
            backend.softmax_last_dim(&s, &layout_1_4)?;
        }
        Capability::IndexSelect => {
            let src = backend.upload(&host_f32, &shape_1_4)?;
            let ids = backend.upload(&HostBuffer::U32(vec![0, 0]), &Shape::from_dims(&[2]))?;
            let ids_layout = Layout::contiguous(&Shape::from_dims(&[2]));
            backend.index_select(&src, &ids, &layout_1_4, &ids_layout, 0)?;
        }
        Capability::Gather => {
            // Gather wants index tensor same rank/shape as output; keep trivial.
            let src = backend.upload(&host_f32, &shape_1_4)?;
            let ids = backend.upload(&HostBuffer::U32(vec![0]), &Shape::from_dims(&[1, 1]))?;
            let ids_layout = Layout::contiguous(&Shape::from_dims(&[1, 1]));
            backend.gather(&src, &ids, &layout_1_4, &ids_layout, 1)?;
        }
        Capability::CopyTo => {
            // Same-device CopyTo is the DynBackend contract; cross-device is Router-level.
            let s = backend.upload(&host_f32, &shape4)?;
            backend.copy_to(&s, &layout4, backend.device())?;
        }
        Capability::MatMulQ4_0 => {
            // Q4_0 matmul: K=128 (4 blocks of 32), N=8. Weight is
            // N * (K/32) * 18 / 4 = 72 u32s. Values can be anything
            // valid — we're testing dispatch, not correctness.
            // Uses the smallest case that parity tests exercise.
            let k = 128;
            let n = 8;
            let n_u32 = n * (k / 32) * 18 / 4;
            let a_host = HostBuffer::F32(vec![0.1_f32; k]);
            let a_shape = Shape::from_dims(&[1, k]);
            let a = backend.upload(&a_host, &a_shape)?;
            let a_layout = Layout::contiguous(&a_shape);
            let w_host = HostBuffer::U32(vec![0_u32; n_u32]);
            let w = backend.upload(&w_host, &Shape::from_dims(&[n_u32]))?;
            let out = backend.matmul_q4_0(&a, &w, k, n, &a_layout)?;
            // Force flush so the async dispatch completes before the
            // test drops its backend. Without this, Vulkan's queue
            // still has in-flight work referring to buffers we're
            // about to drop → access violation on process exit.
            backend.download(&out)?;
        }
        Capability::MatMulQ4KM => {
            // Q4_K_M matmul: K=256 (1 super-block), N=4. Weight is
            // N * (K/256) * 144 bytes = 576 bytes = 144 u32s.
            let k = 256;
            let n = 4;
            let n_u32 = n * (k / 256) * 144 / 4;
            let a_host = HostBuffer::F32(vec![0.1_f32; k]);
            let a_shape = Shape::from_dims(&[1, k]);
            let a = backend.upload(&a_host, &a_shape)?;
            let a_layout = Layout::contiguous(&a_shape);
            let w_host = HostBuffer::U32(vec![0_u32; n_u32]);
            let w = backend.upload(&w_host, &Shape::from_dims(&[n_u32]))?;
            let out = backend.matmul_q4_km(&a, &w, k, n, &a_layout)?;
            backend.download(&out)?;
        }
        Capability::DequantizeQ4KM => {
            // 1 Q4_K_M super-block = 144 bytes = 36 u32s.
            let n_blocks = 1;
            let w_host = HostBuffer::U32(vec![0_u32; 36]);
            let w = backend.upload(&w_host, &Shape::from_dims(&[36]))?;
            // Router's GraphBackend trait doesn't expose dequantize_q4_km
            // (it's Vulkan-specific right now), so we test via the
            // matmul path above. Skip the per-op smoke by yielding Ok —
            // the MatMulQ4KM arm exercises the underlying kernel
            // through dequant-then-matmul.
            let _ = w;
        }
        Capability::QuantizeQ8_0 => {
            // 32-element input → 1 Q8_0 block.
            let n = 32;
            let a_host = HostBuffer::F32(vec![0.5_f32; n]);
            let a = backend.upload(&a_host, &Shape::from_dims(&[n]))?;
            let out = backend.quantize_q8_0(&a, n)?;
            backend.download(&out)?;
        }
        Capability::DequantizeQ8_0 => {
            // 1 Q8_0 block = 34 bytes = 9 u32s (with 2-byte pad).
            let n_blocks = 1;
            let w_host = HostBuffer::U32(vec![0_u32; 9]);
            let w = backend.upload(&w_host, &Shape::from_dims(&[9]))?;
            let out = backend.dequantize_q8_0(&w, n_blocks)?;
            backend.download(&out)?;
        }
        Capability::RmsNormLastDim => {
            let a = backend.upload(&host_f32, &shape_1_4)?;
            let out = backend.rms_norm_last_dim(&a, &layout_1_4, 1e-5)?;
            backend.download(&out)?;
        }
        Capability::ConcatAlongDim => {
            let a = backend.upload(&host_f32, &shape_1_4)?;
            let b = backend.upload(&host_f32, &shape_1_4)?;
            // Concat along dim 0: [1,4] ++ [1,4] → [2,4].
            let out = backend.concat_along_dim(&a, &b, 0, &layout_1_4, &layout_1_4)?;
            backend.download(&out)?;
        }
        Capability::Rope => {
            // RoPE expects [..., seq, head_dim] with head_dim even.
            // Minimal: x = [1, 2, 4] (batch=1, seq=2, head_dim=4),
            // cos/sin = [2, 4] (broadcast over leading dims).
            let x_shape = Shape::from_dims(&[1, 2, 4]);
            let cs_shape = Shape::from_dims(&[2, 4]);
            let x_host = HostBuffer::F32(vec![0.5_f32; 8]);
            let cs_host = HostBuffer::F32(vec![1.0_f32; 8]);
            let x = backend.upload(&x_host, &x_shape)?;
            let cos = backend.upload(&cs_host, &cs_shape)?;
            let sin = backend.upload(&cs_host, &cs_shape)?;
            let out = backend.rope(
                &x, &cos, &sin,
                &Layout::contiguous(&x_shape),
                &Layout::contiguous(&cs_shape),
                &Layout::contiguous(&cs_shape),
            )?;
            backend.download(&out)?;
        }
        Capability::AddAssignScaled => {
            let src = backend.upload(&host_f32, &shape4)?;
            let mut dst = backend.upload(&host_f32, &shape4)?;
            backend.add_assign_scaled(&mut dst, &src, 0.5)?;
            backend.download(&dst)?;
        }
        Capability::RmsNormLastDimBackward => {
            let x = backend.upload(&host_f32, &shape_1_4)?;
            let up = backend.upload(&host_f32, &shape_1_4)?;
            let out = backend.rms_norm_last_dim_backward(
                &x, &up, &layout_1_4, &layout_1_4, 1e-5,
            )?;
            backend.download(&out)?;
        }
        Capability::LayerNormLastDimBackward => {
            let x = backend.upload(&host_f32, &shape_1_4)?;
            let up = backend.upload(&host_f32, &shape_1_4)?;
            let out = backend.layer_norm_last_dim_backward(
                &x, &up, &layout_1_4, &layout_1_4, 1e-5,
            )?;
            backend.download(&out)?;
        }
        Capability::SoftmaxLastDimBackward => {
            let y = backend.upload(&host_f32, &shape_1_4)?;
            let up = backend.upload(&host_f32, &shape_1_4)?;
            let out = backend.softmax_last_dim_backward(
                &y, &up, &layout_1_4, &layout_1_4,
            )?;
            backend.download(&out)?;
        }
        // Capabilities not yet wired through DynBackend (P2.5 work)
        // and the `#[non_exhaustive]` catch-all. If a backend advertises
        // one of these, drift guard fails here until the method is also
        // dispatched through DynBackend and the smoke-test arm is
        // filled in.
        _ => {
            return Err(fuel_core_types::Error::Msg(format!(
                "smoke_test_capability: {cap:?} is declared but not yet wired \
                 through DynBackend — extend smoke_test_capability as part of P2.5"
            )));
        }
    }
    Ok(())
}

/// Assert that every capability a backend advertises has a working
/// DynBackend method behind it. Fails loud with the specific
/// (device, capability, error) triple for any mismatch.
fn assert_capabilities_match_impl(backend: &dyn DynBackend) {
    let caps = backend.capabilities();
    for &cap in caps {
        if let Err(e) = smoke_test_capability(backend, cap) {
            panic!(
                "Drift: backend {:?} declares {:?} but the impl failed: {e}",
                backend.device(), cap
            );
        }
    }
}

#[test]
fn cpu_capabilities_match_impl() {
    assert_capabilities_match_impl(&CpuBackend);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore] // Requires a Vulkan device.
fn vulkan_capabilities_match_impl() {
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    assert_capabilities_match_impl(&vk);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires a CUDA device.
fn cuda_capabilities_match_impl() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };
    let cuda = CudaBackend::new(dev);
    assert_capabilities_match_impl(&cuda);
}

#[test]
fn router_new_is_empty() {
    let r = Router::new();
    // Empty router should error on any op.
    assert!(r.alloc_zeros(&Shape::from_dims(&[4]), fuel_core_types::DType::F32).is_err());
}

#[test]
fn simple_scheduler_assigns_default_device_to_unplaced_nodes() {
    // Graph with no explicit placement hints. SimpleScheduler should
    // tag every reachable node with the Router's default device.
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let b = a.const_f32_like(vec![5.0, 6.0, 7.0, 8.0], Shape::from_dims(&[4]));
    let c = a.add(&b);
    let router = Router::new().add_cpu();

    let placement = SimpleScheduler.plan(c.graph(), &[c.id()], &router);
    // All 3 nodes (a, b, c) should be assigned Cpu (the default).
    assert_eq!(placement.len(), 3);
    for (_id, dev) in &placement {
        assert_eq!(*dev, DeviceLocation::Cpu);
    }
}

#[test]
fn simple_scheduler_preserves_explicit_placement_hints() {
    // A node with an explicit placement should keep it, not get
    // overwritten by the default.
    let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
    let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
    let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    let router = Router::new().add_cpu();

    let placement = SimpleScheduler.plan(c.graph(), &[c.id()], &router);
    // a, b get default (Cpu); c keeps its explicit Vulkan tag.
    assert_eq!(placement[&a.id()], DeviceLocation::Cpu);
    assert_eq!(placement[&b.id()], DeviceLocation::Cpu);
    assert_eq!(placement[&c.id()], DeviceLocation::Vulkan { gpu_id: 0 });
}

#[test]
fn const_pool_limit_evicts_lru_when_budget_exceeded() {
    // Three F32 constants, each 1024 bytes (256 × 4). Pool limit
    // 2100 — fits two but not three. Realize forces eviction.
    //
    // External Arc<[f32]> refs keep strong_count > 1 so eval_const
    // caches them (its "weight-like" heuristic). `_keep_*` bindings
    // hold those refs alive for the duration of the test.
    use fuel_graph_executor::GraphExecutor;
    use std::sync::Arc as StdArc;

    let a_data: StdArc<[f32]> = vec![1.0_f32; 256].into();
    let b_data: StdArc<[f32]> = vec![2.0_f32; 256].into();
    let c_data: StdArc<[f32]> = vec![3.0_f32; 256].into();
    let _keep_a = StdArc::clone(&a_data);
    let _keep_b = StdArc::clone(&b_data);
    let _keep_c = StdArc::clone(&c_data);

    let a = Tensor::from_f32(a_data, Shape::from_dims(&[256]), cpu_dev());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[256]));
    let c = a.const_f32_like(c_data, Shape::from_dims(&[256]));

    let sum_ab = a.add(&b);
    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend)
            .with_const_pool_limit(Some(2100));
    let _ = exec.realize_f32(&sum_ab);
    assert_eq!(exec.const_pool_entries(), 2);
    assert_eq!(exec.const_pool_bytes(), 256 * 4 * 2);

    let sum_bc = b.add(&c);
    let _ = exec.realize_f32(&sum_bc);
    assert_eq!(exec.const_pool_entries(), 2,
        "adding c should have evicted a (LRU)");
    assert!(exec.const_pool_bytes() <= 2100);
}

#[test]
fn const_pool_no_limit_accumulates() {
    // Without a limit, all weights stay cached (today's default).
    use fuel_graph_executor::GraphExecutor;
    use std::sync::Arc as StdArc;

    let a_data: StdArc<[f32]> = vec![1.0_f32; 256].into();
    let b_data: StdArc<[f32]> = vec![2.0_f32; 256].into();
    let c_data: StdArc<[f32]> = vec![3.0_f32; 256].into();
    let _keep_a = StdArc::clone(&a_data);
    let _keep_b = StdArc::clone(&b_data);
    let _keep_c = StdArc::clone(&c_data);

    let a = Tensor::from_f32(a_data, Shape::from_dims(&[256]), cpu_dev());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[256]));
    let c = a.const_f32_like(c_data, Shape::from_dims(&[256]));
    let sum_abc = a.add(&b).add(&c);

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let _ = exec.realize_f32(&sum_abc);
    assert_eq!(exec.const_pool_entries(), 3);
}

#[test]
fn const_pool_reupload_after_eviction_is_correct() {
    // Evict a weight, then use it again. Re-upload should produce
    // identical output to if it stayed cached. Tight limit forces
    // eviction on every realize.
    use fuel_graph_executor::GraphExecutor;
    use std::sync::Arc as StdArc;

    let a_data: StdArc<[f32]> = vec![1.0_f32, 2.0, 3.0, 4.0].into();
    let b_data: StdArc<[f32]> = vec![10.0_f32, 20.0, 30.0, 40.0].into();
    let _keep_a = StdArc::clone(&a_data);
    let _keep_b = StdArc::clone(&b_data);

    let a = Tensor::from_f32(a_data, Shape::from_dims(&[4]), cpu_dev());
    let b = a.const_f32_like(b_data, Shape::from_dims(&[4]));
    let c = a.add(&b);

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend)
            .with_const_pool_limit(Some(16));

    let r1 = exec.realize_f32(&c);
    let r2 = exec.realize_f32(&c);
    assert_eq!(r1.as_slice(), &[11.0_f32, 22.0, 33.0, 44.0]);
    assert_eq!(r2.as_slice(), &[11.0_f32, 22.0, 33.0, 44.0]);
}

#[test]
fn op_move_realizes_to_target_with_destructive_semantics() {
    // Op::Move is Copy + Release source fused. Executor should:
    //   - produce the copied tensor on target (same data as source)
    //   - evict source from cache after the move runs (destructive)
    // On a CPU-only executor both are trivial no-ops (same device),
    // but the cache eviction still happens and realize_f32 must
    // still return the moved tensor's data intact.
    use fuel_graph_executor::GraphExecutor;

    let a = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let moved = a.move_to_device(DeviceLocation::Cpu);

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let r = exec.realize_f32(&moved);
    assert_eq!(r.as_slice(), &[1.0_f32, 2.0, 3.0, 4.0],
        "Op::Move output should carry input's data to target");
}

#[test]
fn op_move_pinned_after_sibling_reader_via_derive_ordering() {
    // Graph:
    //   a = const [1, 2, 3, 4]
    //   b = relu(a)              (non-destructive reader of a)
    //   m = move(a, Cpu)         (destructive reader — fused Copy+Release)
    //   out = add(b, m)
    // derive_ordering pins m to run AFTER b. The executor cache
    // evicts a once m runs; b must have already read it.
    // Expected output: b[i] + m[i] = (i+1 if positive else 0) + (i+1).
    //   a = [1, 2, 3, 4]
    //   b = relu(a) = [1, 2, 3, 4]
    //   m = move(a) = [1, 2, 3, 4]
    //   out = b + m = [2, 4, 6, 8]
    use fuel_graph_executor::GraphExecutor;

    let a = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let b = a.relu();
    let m = a.move_to_device(DeviceLocation::Cpu);
    let out = b.add(&m);

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let r = exec.realize_f32(&out);
    assert_eq!(r.as_slice(), &[2.0_f32, 4.0, 6.0, 8.0]);
}

#[test]
fn release_with_sibling_reader_realizes_both_roots_correctly() {
    // Graph:
    //   a = const [1, 2, 3, 4]
    //   b = relu(a)         (non-destructive reader)
    //   r = release(a)      (destructive reader)
    // Realize both b and r as roots. The execution plan's derived
    // ordering pins `r` AFTER `b`, so relu's output is already cached
    // by the time release runs and destroys `a`. Both roots must
    // produce correct output; this test fails if eviction ordering
    // is wrong (e.g., `a` is removed before relu can read it).
    use fuel_graph_executor::GraphExecutor;

    let a = Tensor::from_f32(vec![-1.0_f32, 2.0, -3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let b = a.relu();
    let r = a.release();

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let results = exec.realize_many_f32(&[&b, &r]);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].as_slice(), &[0.0_f32, 2.0, 0.0, 4.0], "relu of a");
    assert_eq!(results[1].as_slice().len(), 0, "release produces zero-element marker");
}

#[test]
fn release_does_not_break_transitive_consumer() {
    // Graph:
    //   a = const [1, 2, 3, 4]
    //   b = relu(a)         (reads a)
    //   sum = sum_all(b)    (reads b — does NOT read a)
    //   r = release(a)      (destroys a)
    // Ordering pins r after b. sum only needs b. Realizing [sum, r]
    // should give sum = 0 + 2 + 0 + 4 = 6 even though a is evicted
    // from cache after release.
    use fuel_graph_executor::GraphExecutor;

    let a = Tensor::from_f32(vec![-1.0_f32, 2.0, -3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let b = a.relu();
    let sum = b.sum_all();
    let r = a.release();

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let results = exec.realize_many_f32(&[&sum, &r]);
    assert_eq!(results[0].as_slice(), &[6.0_f32]);
    assert_eq!(results[1].as_slice().len(), 0);
}

#[test]
fn op_release_realizes_as_zero_element_marker() {
    // Op::Release is a new destructive op that produces a zero-element
    // marker output. The ordering-enforcement pass (derive_ordering)
    // lands in a follow-up PR; this test just confirms that today's
    // backends can execute Release without crashing — the actual
    // destructive semantics (cache eviction + sibling-reader ordering)
    // arrives with derive_ordering.
    use fuel_graph_executor::GraphExecutor;

    let a = Tensor::from_f32(vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
    let released = a.release();

    let mut exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
        GraphExecutor::new(fuel_graph_cpu::CpuBackend);
    let r = exec.realize_f32(&released);
    assert_eq!(r.as_slice().len(), 0, "Op::Release produces zero-element marker");
}

#[test]
fn rule_scheduler_default_pipeline_matches_simple_on_flat_graph() {
    // On a graph with no placement hints, RuleScheduler's default
    // pipeline (Baseline + ConstLowering) should produce the same
    // final placement as SimpleScheduler: every node on default.
    // ConstLowering finds nothing to refine because consumers all
    // agree on default anyway.
    let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
    let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
    let c = a.add(&b);
    let router = Router::new().add_cpu();

    let simple_out = SimpleScheduler.plan(c.graph(), &[c.id()], &router);
    let rule_out = RuleScheduler::default_pipeline().plan(c.graph(), &[c.id()], &router);
    assert_eq!(simple_out, rule_out);
}

#[test]
fn rule_scheduler_lowers_const_placement_when_consumer_placed() {
    // Baseline assigns everything to default (Cpu).
    // Then we override c's placement to Vulkan via explicit hint.
    // ConstLoweringRule should pull a, b onto Vulkan to match c's
    // consumer placement — saving the Copies insert_copies would emit.
    let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
    let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
    let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    let router = Router::new().add_cpu();

    let plan = RuleScheduler::default_pipeline().plan(c.graph(), &[c.id()], &router);
    assert_eq!(plan[&c.id()], DeviceLocation::Vulkan { gpu_id: 0 });
    assert_eq!(plan[&a.id()], DeviceLocation::Vulkan { gpu_id: 0 },
               "const a should be lowered to Vulkan to match c's placement");
    assert_eq!(plan[&b.id()], DeviceLocation::Vulkan { gpu_id: 0 },
               "const b should be lowered to Vulkan to match c's placement");
}

#[test]
fn scheduler_plan_then_apply_then_insert_copies_roundtrip() {
    // Full pipeline demo: SimpleScheduler → apply → insert_copies.
    // With SimpleScheduler-on-CPU and no placement hints, everything
    // lands on Cpu and no Copies should get inserted.
    let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
    let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
    let c = a.add(&b);
    let router = Router::new().add_cpu();

    let placement = SimpleScheduler.plan(c.graph(), &[c.id()], &router);
    apply_placement(c.graph(), &placement);

    // Now every node has a Cpu placement. insert_copies should see
    // unanimous Cpu and insert zero Copies.
    let before = {
        let g = c.graph().read().unwrap();
        (0..g.len())
            .filter(|i| matches!(g.node(fuel_graph::NodeId(*i)).op, fuel_graph::Op::Copy { .. }))
            .count()
    };
    fuel_graph::opt::insert_copies(c.graph(), &[c.id()]);
    let after = {
        let g = c.graph().read().unwrap();
        (0..g.len())
            .filter(|i| matches!(g.node(fuel_graph::NodeId(*i)).op, fuel_graph::Op::Copy { .. }))
            .count()
    };
    assert_eq!(before, after, "unanimous Cpu placement → no Copies");
}

#[test]
fn cpu_router_advertises_core_capabilities_not_q4_0() {
    use fuel_core_types::Capability;
    let r = Router::new().add_cpu();
    // CPU has all 16 core ops:
    assert!(r.supports(Capability::MatMul));
    assert!(r.supports(Capability::SoftmaxLastDim));
    assert!(r.supports(Capability::CopyTo));
    // CPU has no native qmatmul kernel:
    assert!(!r.supports(Capability::MatMulQ4_0));
    assert!(!r.supports(Capability::MatMulQ8_0));
    // Devices_for is O(1):
    assert_eq!(r.devices_for(Capability::MatMul), &[DeviceLocation::Cpu]);
    assert!(r.devices_for(Capability::MatMulQ4_0).is_empty());
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore] // Requires Vulkan.
fn vulkan_router_advertises_q4_0_matmul() {
    use fuel_core_types::Capability;
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    let r = Router::new().add_cpu().add_vulkan(vk);
    // Both backends advertise core ops:
    assert_eq!(r.devices_for(Capability::MatMul).len(), 2);
    // Only Vulkan advertises native Q4_0 matmul:
    assert_eq!(
        r.devices_for(Capability::MatMulQ4_0),
        &[DeviceLocation::Vulkan { gpu_id: 0 }]
    );
    // Neither has Q8_0 gemv yet:
    assert!(!r.supports(Capability::MatMulQ8_0));
}

#[test]
fn router_cpu_only_single_device_matmul() {
    // 2x3 @ 3x2 -> 2x2
    let a = Tensor::from_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        cpu_dev(),
    );
    let b = a.const_f32_like(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Shape::from_dims(&[3, 2]),
    );
    let c = a.matmul(&b);

    let router = Router::new().add_cpu();
    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&c);
    assert_eq!(result.as_slice(), &[4.0_f32, 5.0, 10.0, 11.0]);
}

#[test]
fn router_cpu_copy_to_cpu_is_identity() {
    // Op::Copy from CPU to CPU on a CPU-only router should be a
    // pass-through (via DynBackend::copy_to's default → try_clone).
    let a = Tensor::from_f32(
        vec![7.0, 8.0, 9.0, 10.0],
        Shape::from_dims(&[4]),
        cpu_dev(),
    );
    let moved = a.copy_to_device(DeviceLocation::Cpu);

    let router = Router::new().add_cpu();
    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&moved);
    assert_eq!(result.as_slice(), &[7.0_f32, 8.0, 9.0, 10.0]);
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore] // Requires a Vulkan device.
fn router_copy_to_roundtrips_cpu_to_vulkan_to_cpu() {
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    let router = Router::new().add_cpu().add_vulkan(vk);

    let shape = Shape::from_dims(&[4]);
    let layout = Layout::contiguous(&shape);
    let src = router.upload_to(
        &fuel_core_types::HostBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
        &shape,
        DeviceLocation::Cpu,
    ).unwrap();
    assert_eq!(src.device(), DeviceLocation::Cpu);

    let on_vk = router.copy_to(&src, &layout, DeviceLocation::Vulkan { gpu_id: 0 }).unwrap();
    assert_eq!(on_vk.device(), DeviceLocation::Vulkan { gpu_id: 0 });

    let back = router.copy_to(&on_vk, &layout, DeviceLocation::Cpu).unwrap();
    assert_eq!(back.device(), DeviceLocation::Cpu);

    let host = router.download(&back).unwrap();
    match host {
        fuel_core_types::HostBuffer::F32(v) => assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]),
        _ => panic!("expected F32 buffer"),
    }
}

/// Dyn-dispatch Router with Vulkan attached realizes a graph fully
/// on Vulkan (via a vtable hop per op). Proves the pivot from enum
/// to `Vec<Arc<dyn DynBackend>>` didn't regress the single-device
/// fast path.
#[cfg(feature = "vulkan")]
#[test]
#[ignore] // Requires a Vulkan device.
fn router_vulkan_default_realizes_graph() {
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    let router = Router::new().add_cpu().add_vulkan(vk);
    // Most recent add_* becomes the default device.

    let a = Tensor::from_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        cpu_dev(),
    );
    let b = a.const_f32_like(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Shape::from_dims(&[3, 2]),
    );
    let c = a.matmul(&b);

    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&c);
    assert_eq!(result.as_slice(), &[4.0_f32, 5.0, 10.0, 11.0]);
}

/// Phase 3.5 end-to-end validation: build a graph that uses
/// placement hints (NOT explicit `copy_to_device` calls), run
/// `fuel_graph::opt::insert_copies` to auto-insert the Copy nodes,
/// then realize through the Router. The auto-inserted Copies should
/// route data CPU → Vulkan and back with no manual intervention.
#[cfg(feature = "vulkan")]
#[test]
#[ignore] // Requires Vulkan.
fn auto_insert_copies_reconciles_mixed_placement_graph() {
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    let router = Router::new().add_cpu().add_vulkan(vk);

    // 2x3 @ 3x2 → 2x2, computation placed on Vulkan. Const inputs
    // have no placement (they'll need Copies inserted to Vulkan).
    let a = Tensor::from_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        cpu_dev(),
    );
    let b = a.const_f32_like(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Shape::from_dims(&[3, 2]),
    );
    // Tag the matmul with Vulkan placement — the pass should insert
    // Copies on both const inputs.
    let c = a.matmul(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    let graph = c.graph().clone();

    // Count Copies before + after the pass to confirm insertion.
    let before = {
        let g = graph.read().unwrap();
        (0..g.len())
            .filter(|i| matches!(g.node(fuel_graph::NodeId(*i)).op, fuel_graph::Op::Copy { .. }))
            .count()
    };
    let new_roots = fuel_graph::opt::insert_copies(&graph, &[c.id()]);
    let after = {
        let g = graph.read().unwrap();
        (0..g.len())
            .filter(|i| matches!(g.node(fuel_graph::NodeId(*i)).op, fuel_graph::Op::Copy { .. }))
            .count()
    };
    assert_eq!(after - before, 2, "expected two Copies inserted for the two const inputs");

    // Realize the rewritten graph and check the result.
    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let new_root_tensor = fuel_graph::Tensor::from_existing(graph.clone(), new_roots[0]);
    let result = executor.realize_f32(&new_root_tensor);
    assert_eq!(result.as_slice(), &[4.0_f32, 5.0, 10.0, 11.0]);
}

/// End-to-end Phase 3 validation: a graph that uses `Op::Copy` to
/// shuttle data between CPU and Vulkan mid-compute. Proves the
/// Router's copy_to intercepts the op dispatch and performs the
/// cross-device host round-trip.
#[cfg(feature = "vulkan")]
#[test]
#[ignore]
fn router_graph_with_explicit_moves() {
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("Vulkan device available");
    let router = Router::new().add_cpu().add_vulkan(vk);

    // Default device is Vulkan (last attached). Upload a tensor,
    // move it to CPU, move it back to Vulkan. Compare to the direct
    // path (same tensor, no moves).
    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        Shape::from_dims(&[8]),
        cpu_dev(),
    );
    let moved_cpu = a.copy_to_device(DeviceLocation::Cpu);
    let back_vulkan = moved_cpu.copy_to_device(DeviceLocation::Vulkan { gpu_id: 0 });

    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&back_vulkan);
    // Data should survive the CPU -> Vulkan -> CPU -> Vulkan round-trip.
    assert_eq!(result.as_slice(), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

// The residency-planner Vulkan analysis test and the CUDA
// ResidencyEvictionRule correctness test moved to
// `fuel-dispatch::residency` (unit tests) +
// `fuel-dispatch/tests/residency_eviction_live.rs` (live-GPU
// evict→fault-back roundtrip) with the Session 6 port onto the
// pipelined executor.

/// CUDA single-device realize sanity check. Confirms Router routes a
/// matmul graph to the CUDA backend and produces the expected output.
/// Acts as the baseline for the more elaborate residency-rule test
/// below — if this fails, the rule test's setup is wrong, not the rule.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_router_default_realizes_graph() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;

    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };
    let cuda = CudaBackend::new(dev);
    let router = Router::new().add_cpu().add_cuda(cuda);

    let a = Tensor::from_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        cpu_dev(),
    );
    let b = a.const_f32_like(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        Shape::from_dims(&[3, 2]),
    );
    let c = a.matmul(&b);

    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&c);
    assert_eq!(result.as_slice(), &[4.0_f32, 5.0, 10.0, 11.0]);
}

/// CUDA rope kernel parity vs CPU reference, for the shapes that arise
/// in real transformer decode/prefill paths. Uses Fuel's cos/sin table
/// convention (shape `[seq, head_dim]` with duplicated halves).
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_rope_matches_cpu_reference() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    use fuel_graph_executor::{GraphBackend, GraphExecutor};
    use fuel_core_types::HostBuffer;

    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };

    for (outer, seq, head_dim, seed) in [
        (1, 1, 64, 1u32),     // TinyLlama decode, 1 token
        (1, 5, 64, 2u32),     // prefill 5 tokens
        (4, 3, 32, 3u32),     // multi-head
        (32, 1, 64, 4u32),    // many heads, 1 token
    ] {
        // Build x / cos / sin as flat const data.
        let n = outer * seq * head_dim;
        let x_data: Vec<f32> = (0..n)
            .map(|i| (((i as u32) ^ seed) as f32 * 1e-4).sin()).collect();
        let (cos_data, sin_data) =
            fuel_graph::build_rope_tables(10_000.0, 0, seq, head_dim);
        assert_eq!(cos_data.len(), seq * head_dim);
        assert_eq!(sin_data.len(), seq * head_dim);

        // CPU reference: build graph, realize via CpuBackend.
        let x_cpu = Tensor::from_f32(x_data.clone(),
            Shape::from_dims(&[outer, seq, head_dim]), cpu_dev());
        let cos_cpu = x_cpu.const_f32_like(cos_data.clone(),
            Shape::from_dims(&[seq, head_dim]));
        let sin_cpu = x_cpu.const_f32_like(sin_data.clone(),
            Shape::from_dims(&[seq, head_dim]));
        let y_cpu_t = x_cpu.rope_with_tables(&cos_cpu, &sin_cpu);
        let mut cpu_exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
            GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let cpu_ref = cpu_exec.realize_f32(&y_cpu_t).as_slice().to_vec();

        // CUDA: bypass the graph layer and call CudaBackend::rope directly
        // so failures isolate to the kernel-plumbing, not the Router.
        let cuda = CudaBackend::new(dev.clone());
        let x_shape = Shape::from_dims(&[outer, seq, head_dim]);
        let cs_shape = Shape::from_dims(&[seq, head_dim]);
        let x_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(x_data), &x_shape).unwrap();
        let cos_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(cos_data), &cs_shape).unwrap();
        let sin_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(sin_data), &cs_shape).unwrap();
        let x_layout = Layout::contiguous(&x_shape);
        let cs_layout = Layout::contiguous(&cs_shape);
        let y_cuda = <CudaBackend as GraphBackend>::rope(
            &cuda, &x_cuda, &cos_cuda, &sin_cuda,
            &x_layout, &cs_layout, &cs_layout).unwrap();
        let HostBuffer::F32(cuda_out) =
            <CudaBackend as GraphBackend>::download(&cuda, &y_cuda).unwrap()
            else { panic!("expected F32 output"); };

        if cpu_ref.len() != cuda_out.len() {
            panic!("rope outer={outer} seq={seq} head_dim={head_dim}: length mismatch: {} vs {}",
                cpu_ref.len(), cuda_out.len());
        }
        for (i, (a, b)) in cpu_ref.iter().zip(cuda_out.iter()).enumerate() {
            let d = (a - b).abs();
            let tol = 1e-5_f32;
            let rel_ok = d <= tol * a.abs().max(b.abs()).max(1.0);
            if !rel_ok && d > tol {
                panic!("rope outer={outer} seq={seq} head_dim={head_dim}: \
                       mismatch at {i}: cpu={a} cuda={b} diff={d}");
            }
        }
    }
}

/// CUDA matmul_q4_0 parity test vs CPU reference. Generates a
/// deterministic Q4_0 weight blob, computes `out = a @ dequant(w)`
/// in Rust, then exercises CUDA's dequantize_mul_mat_vec_q4_0 kernel
/// through the CudaBackend. First-cut limitation: M=1 only (decode
/// path); the kernel is a gemv.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_matmul_q4_0_matches_cpu_reference() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    use fuel_graph_executor::GraphBackend;
    use fuel_core_types::HostBuffer;
    use half::f16;

    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };

    // Deterministic Q4_0 blob generator (matches the Vulkan test).
    fn make_q4_0(n_blocks: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut blob = Vec::with_capacity(n_blocks * 18);
        let mut expected = vec![0.0_f32; n_blocks * 32];
        let mut rng = seed;
        for b in 0..n_blocks {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let s = 0.001 + ((rng >> 8) as f32 / u32::MAX as f32) * 0.099;
            let scale = f16::from_f32(s);
            blob.extend_from_slice(&scale.to_bits().to_le_bytes());
            let start = blob.len();
            for _ in 0..16 {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                blob.push((rng >> 8) as u8);
            }
            let d = scale.to_f32();
            for k in 0..16 {
                let packed = blob[start + k];
                let x0 = (packed & 0x0F) as i32 - 8;
                let x1 = ((packed >> 4) & 0x0F) as i32 - 8;
                expected[b * 32 + k] = x0 as f32 * d;
                expected[b * 32 + 16 + k] = x1 as f32 * d;
            }
        }
        (blob, expected)
    }

    for (k, n, seed) in [
        (128usize, 8usize, 1u32),
        (512, 32, 2),
        (2048, 128, 3),  // closer to real LLM proj matrix
    ] {
        let n_blocks = n * (k / 32);
        let (blob, dequant) = make_q4_0(n_blocks, seed);
        // Pack the u8 blob into u32 (4 u8s per u32, little-endian).
        assert_eq!(blob.len() % 4, 0);
        let blob_u32: Vec<u32> = blob.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        // CPU activation and expected output.
        let a_data: Vec<f32> = (0..k)
            .map(|i| (((i as u32) ^ seed) as f32 * 1e-3).sin()).collect();
        // dequant has shape [n, k]; out[j] = sum_i a[i] * dequant[j*k + i].
        let cpu_out: Vec<f32> = (0..n).map(|j| {
            (0..k).map(|i| a_data[i] * dequant[j * k + i]).sum()
        }).collect();

        // CUDA path.
        let cuda = CudaBackend::new(dev.clone());
        let a_shape = Shape::from_dims(&[1, k]);
        let a_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(a_data), &a_shape).unwrap();
        let w_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::U32(blob_u32),
            &Shape::from_dims(&[n_blocks * 18 / 4])).unwrap();
        let a_layout = Layout::contiguous(&a_shape);
        let out = <CudaBackend as GraphBackend>::matmul_q4_0(
            &cuda, &a_cuda, &w_cuda, k, n, &a_layout).unwrap();
        let HostBuffer::F32(cuda_out) =
            <CudaBackend as GraphBackend>::download(&cuda, &out).unwrap()
            else { panic!("expected F32"); };

        assert_eq!(cpu_out.len(), cuda_out.len());
        for (i, (a, b)) in cpu_out.iter().zip(cuda_out.iter()).enumerate() {
            let d = (a - b).abs();
            // Q4_0 quantization has modest precision; tolerance scaled
            // by the inner-product magnitude.
            let tol = 1e-3_f32;
            let rel_ok = d <= tol * a.abs().max(b.abs()).max(1.0);
            if !rel_ok && d > tol {
                panic!("q4_0 k={k} n={n}: mismatch at {i}: cpu={a} cuda={b} diff={d}");
            }
        }
    }
}

/// CUDA matmul_q4_km parity test. Uses `BlockQ4K::to_float` as the
/// CPU dequant reference (same approach as the Vulkan Q4_K_M test).
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_matmul_q4_km_matches_cpu_reference() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    use fuel_graph_executor::GraphBackend;
    use fuel_core_types::HostBuffer;
    use half::f16;
    use fuel_core::quantized::k_quants::{BlockQ4K, GgmlType};

    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };

    fn make_q4_km(n_blocks: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        const QK_K: usize = 256;
        const BYTES_PER_BLOCK: usize = 144;
        let mut blob = Vec::with_capacity(n_blocks * BYTES_PER_BLOCK);
        let mut rng = seed;
        let mut next = || { rng = rng.wrapping_mul(1664525).wrapping_add(1013904223); rng };
        for _ in 0..n_blocks {
            let d_f32    = 0.001 + ((next() >> 8) as f32 / u32::MAX as f32) * 0.099;
            let dmin_f32 = 0.001 + ((next() >> 8) as f32 / u32::MAX as f32) * 0.099;
            blob.extend_from_slice(&f16::from_f32(d_f32).to_bits().to_le_bytes());
            blob.extend_from_slice(&f16::from_f32(dmin_f32).to_bits().to_le_bytes());
            for _ in 0..12 { blob.push((next() >> 8) as u8); }
            for _ in 0..128 { blob.push((next() >> 8) as u8); }
        }
        assert_eq!(std::mem::size_of::<BlockQ4K>(), BYTES_PER_BLOCK);
        let blocks: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(blob.as_ptr() as *const BlockQ4K, n_blocks)
        };
        let mut expected = vec![0.0_f32; n_blocks * QK_K];
        BlockQ4K::to_float(blocks, &mut expected);
        (blob, expected)
    }

    for (k, n, seed) in [
        (256usize, 4usize, 1u32),     // 1 super-block per row
        (512, 16, 2),                 // 2 super-blocks per row
        (2048, 32, 3),                // realistic LLM-ish
    ] {
        let n_blocks = n * (k / 256);
        let (blob, dequant) = make_q4_km(n_blocks, seed);
        let u32_len = (blob.len() + 3) / 4;
        let mut padded = blob.clone();
        padded.resize(u32_len * 4, 0);
        let blob_u32: Vec<u32> = padded.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let a_data: Vec<f32> = (0..k)
            .map(|i| (((i as u32) ^ seed) as f32 * 1e-3).sin()).collect();
        let cpu_out: Vec<f32> = (0..n).map(|j| {
            (0..k).map(|i| a_data[i] * dequant[j * k + i]).sum()
        }).collect();

        let cuda = CudaBackend::new(dev.clone());
        let a_shape = Shape::from_dims(&[1, k]);
        let a_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(a_data), &a_shape).unwrap();
        let w_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::U32(blob_u32),
            &Shape::from_dims(&[u32_len])).unwrap();
        let a_layout = Layout::contiguous(&a_shape);
        let out = <CudaBackend as GraphBackend>::matmul_q4_km(
            &cuda, &a_cuda, &w_cuda, k, n, &a_layout).unwrap();
        let HostBuffer::F32(cuda_out) =
            <CudaBackend as GraphBackend>::download(&cuda, &out).unwrap()
            else { panic!("expected F32"); };

        assert_eq!(cpu_out.len(), cuda_out.len());
        for (i, (a, b)) in cpu_out.iter().zip(cuda_out.iter()).enumerate() {
            let d = (a - b).abs();
            let tol = 2e-3_f32;
            let rel_ok = d <= tol * a.abs().max(b.abs()).max(1.0);
            if !rel_ok && d > tol {
                panic!("q4_km k={k} n={n}: mismatch at {i}: cpu={a} cuda={b} diff={d}");
            }
        }
    }
}

/// CUDA rms_norm_last_dim parity test vs CPU reference.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_rms_norm_last_dim_matches_cpu_reference() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    use fuel_graph_executor::{GraphBackend, GraphExecutor};
    use fuel_core_types::HostBuffer;

    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };

    for (n_rows, n_cols, eps, seed) in [
        (1usize, 128usize, 1e-6_f64, 1u32),
        (4, 256, 1e-5, 2),
        (16, 2048, 1e-6, 3),  // transformer hidden state
    ] {
        let n = n_rows * n_cols;
        let x_data: Vec<f32> = (0..n)
            .map(|i| (((i as u32) ^ seed) as f32 * 1e-3).sin()).collect();
        let shape = Shape::from_dims(&[n_rows, n_cols]);

        // CPU reference
        let x_cpu = Tensor::from_f32(x_data.clone(), shape.clone(), cpu_dev());
        let y_cpu_t = x_cpu.rms_norm_last_dim(eps);
        let mut cpu_exec: GraphExecutor<fuel_graph_cpu::CpuBackend> =
            GraphExecutor::new(fuel_graph_cpu::CpuBackend);
        let cpu_ref = cpu_exec.realize_f32(&y_cpu_t).as_slice().to_vec();

        // CUDA direct call
        let cuda = CudaBackend::new(dev.clone());
        let x_cuda = <CudaBackend as GraphBackend>::upload(
            &cuda, &HostBuffer::F32(x_data), &shape).unwrap();
        let layout = Layout::contiguous(&shape);
        let y_cuda = <CudaBackend as GraphBackend>::rms_norm_last_dim(
            &cuda, &x_cuda, &layout, eps).unwrap();
        let HostBuffer::F32(cuda_out) =
            <CudaBackend as GraphBackend>::download(&cuda, &y_cuda).unwrap()
            else { panic!("expected F32 output"); };

        assert_eq!(cpu_ref.len(), cuda_out.len());
        for (i, (a, b)) in cpu_ref.iter().zip(cuda_out.iter()).enumerate() {
            let d = (a - b).abs();
            let tol = 1e-4_f32;
            let rel_ok = d <= tol * a.abs().max(b.abs()).max(1.0);
            if !rel_ok && d > tol {
                panic!("rms_norm n_rows={n_rows} n_cols={n_cols}: mismatch at {i}: cpu={a} cuda={b} diff={d}");
            }
        }
    }
}

/// Multi-device Router: CPU + Vulkan + CUDA all attached to one Router.
/// Proves the dyn dispatch handles three backends simultaneously and
/// that cross-device `copy_to` routes through host correctly even
/// between two discrete GPUs (Vulkan iGPU + CUDA dGPU, say).
#[cfg(all(feature = "cuda", feature = "vulkan"))]
#[test]
#[ignore]
fn router_cpu_plus_vulkan_plus_cuda_routes_per_placement() {
    use fuel_cuda_backend::CudaDevice;
    use fuel_cuda_backend::CudaBackend;
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

    let vk = match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => b,
        Err(e) => { eprintln!("no Vulkan device; skipping: {e:?}"); return; }
    };
    let cuda_dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => { eprintln!("no CUDA device; skipping: {e:?}"); return; }
    };

    let router = Router::new()
        .add_cpu()
        .add_vulkan(vk)
        .add_cuda(CudaBackend::new(cuda_dev));

    // `a` on CUDA, `b` on Vulkan. The Router must download one via
    // host and upload to the other device for the matmul. Either
    // direction (CUDA→host→Vulkan or Vulkan→host→CUDA) is valid; the
    // outcome must match the CPU reference.
    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::from_dims(&[2, 3]),
        cpu_dev(),
    ).on_device(DeviceLocation::Cuda { gpu_id: 0 });
    let b = a.const_f32_like(
        vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
        Shape::from_dims(&[3, 2]),
    ).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
    // matmul produces a tensor whose placement defaults to the first
    // input's device (CUDA); insert_copies will bridge b over.
    let c = a.matmul(&b);

    let graph = c.graph().clone();
    fuel_graph::opt::insert_copies(&graph, &[c.id()]);

    let mut executor = fuel_graph_executor::GraphExecutor::new(router);
    let result = executor.realize_f32(&c);
    assert_eq!(result.as_slice(), &[4.0_f32, 5.0, 10.0, 11.0]);
}

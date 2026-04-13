//! CUDA GPU executor for `fuel-graph` computation graphs.
//!
//! All intermediates stay in GPU memory; host↔device transfer happens
//! only at `Const` upload (H2D) and `realize_*` readback (D2H).

use fuel_core_types::{DType, DimVec, Layout, Shape};
use fuel_cuda::{CudaDevice, CudaStorage, CudaStorageSlice};
use fuel_graph::{topo_order, topo_order_multi, ConstData, NodeId, Op, Tensor};
use fuel_reference_backend::exec::AnyRefTensor as AnyRef;
use fuel_reference_backend::RefTensor;
use std::collections::HashMap;

/// Cached GPU tensor: storage + shape (CudaStorage doesn't track shape).
struct GpuTensor {
    storage: CudaStorage,
    shape: Shape,
}

impl GpuTensor {
    fn layout(&self) -> Layout {
        Layout::contiguous(&self.shape)
    }
}

/// Holds a CUDA device and a dedup cache for uploaded weight constants.
pub struct CudaGraphExecutor {
    pub device: CudaDevice,
    const_cache: HashMap<usize, GpuTensor>,
}

impl CudaGraphExecutor {
    pub fn new(device: CudaDevice) -> Self {
        Self {
            device,
            const_cache: HashMap::new(),
        }
    }

    pub fn for_device(ordinal: usize) -> fuel_core_types::Result<Self> {
        Ok(Self::new(CudaDevice::new(ordinal)?))
    }

    pub fn realize_f32(&mut self, tensor: &Tensor) -> RefTensor<f32> {
        let graph = tensor.graph().borrow();
        let order = topo_order(&graph, tensor.id());
        let mut cache: HashMap<NodeId, GpuTensor> = HashMap::new();
        for id in order {
            let node = graph.node(id);
            let gt = self.eval_node(&node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, gt);
        }
        let gt = cache.remove(&tensor.id()).expect("realize: missing root");
        gpu_to_ref_f32(gt)
    }

    pub fn realize_many_f32(&mut self, tensors: &[&Tensor]) -> Vec<RefTensor<f32>> {
        if tensors.is_empty() {
            return Vec::new();
        }
        let graph_rc = tensors[0].graph();
        let graph = graph_rc.borrow();
        let roots: Vec<NodeId> = tensors.iter().map(|t| t.id()).collect();
        let order = topo_order_multi(&graph, &roots);
        let mut cache: HashMap<NodeId, GpuTensor> = HashMap::new();
        for id in order {
            let node = graph.node(id);
            let gt = self.eval_node(&node.op, &node.inputs, &node.shape, node.dtype, &cache);
            cache.insert(id, gt);
        }
        roots
            .iter()
            .map(|id| {
                let gt = cache.get(id).expect("realize_many: missing root");
                gpu_to_ref_f32_ref(gt)
            })
            .collect()
    }

    fn eval_node(
        &mut self,
        op: &Op,
        inputs: &[NodeId],
        shape: &Shape,
        dtype: DType,
        cache: &HashMap<NodeId, GpuTensor>,
    ) -> GpuTensor {
        let result_storage = match op {
            Op::Const(data) => return self.eval_const(data, shape),

            Op::MatMul => {
                let (a, b) = (g(inputs, 0, cache), g(inputs, 1, cache));
                let ad = a.shape.dims();
                let bd = b.shape.dims();
                let rank = ad.len();
                let (m, k, n) = (ad[rank - 2], ad[rank - 1], bd[rank - 1]);
                let batch: usize = ad[..rank - 2].iter().product::<usize>().max(1);
                a.storage.matmul(&b.storage, (batch, m, n, k), &a.layout(), &b.layout())
                    .expect("MatMul")
            }

            // Unary and binary ops: CPU fallback for now. The unary/
            // binary CUDA kernels exist in fuel-cuda but are dispatched
            // via generic op-type parameters (UnaryOpT/BinaryOpT impls)
            // defined in fuel-core, which we can't depend on due to the
            // circular dep. Phase 2 will move the op types to
            // fuel-core-types to unlock native GPU dispatch.
            Op::Neg | Op::Sqr | Op::Sqrt | Op::Exp | Op::Log |
            Op::Sin | Op::Cos | Op::Tanh | Op::Sigmoid | Op::Silu |
            Op::Gelu | Op::Relu | Op::Step |
            Op::Add | Op::Sub | Op::Mul | Op::Div |
            Op::Maximum | Op::Minimum => {
                return self.cpu_fallback(inputs, shape, cache, |node_inputs, node_shape, cpu_cache| {
                    fuel_reference_backend::exec::eval_node_with_op(
                        op, node_inputs, node_shape, dtype, cpu_cache,
                    )
                });
            }

            // scalar
            Op::AddScalar(c) => {
                let a = g(inputs, 0, cache);
                a.storage.affine(&a.layout(), 1.0, *c).expect("AddScalar")
            }
            Op::MulScalar(c) => {
                let a = g(inputs, 0, cache);
                a.storage.affine(&a.layout(), *c, 0.0).expect("MulScalar")
            }
            Op::PowI(n) => {
                let a = g(inputs, 0, cache);
                a.storage.powf(&a.layout(), *n as f64).expect("PowI")
            }

            // dtype
            Op::Cast(target) => {
                let a = g(inputs, 0, cache);
                a.storage.to_dtype(&a.layout(), *target).expect("Cast")
            }

            // shape (reshape is zero-cost metadata change — just clone storage)
            Op::Reshape(_) => {
                let a = g(inputs, 0, cache);
                return GpuTensor {
                    storage: a.storage.try_clone(&a.layout()).expect("Reshape"),
                    shape: shape.clone(),
                };
            }

            Op::Transpose => {
                let a = g(inputs, 0, cache);
                let rank = a.shape.dims().len();
                let mut perm: Vec<usize> = (0..rank).collect();
                perm.swap(rank - 2, rank - 1);
                return self.do_permute(a, &perm, shape);
            }
            Op::Permute(axes) => {
                let a = g(inputs, 0, cache);
                return self.do_permute(a, axes, shape);
            }

            Op::BroadcastTo(target) => {
                let a = g(inputs, 0, cache);
                return self.do_broadcast(a, target);
            }

            // reductions
            Op::SumAll | Op::MeanAll => {
                let a = g(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                let mut r = a.storage.reduce_op(
                    fuel_core_types::op::ReduceOp::Sum, &a.layout(), &axes,
                ).expect("SumAll");
                if matches!(op, Op::MeanAll) {
                    let n = a.shape.elem_count() as f64;
                    r = r.affine(&Layout::contiguous(shape), 1.0 / n, 0.0)
                        .expect("MeanAll scale");
                }
                r
            }
            Op::MaxAll => {
                let a = g(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Max, &a.layout(), &axes)
                    .expect("MaxAll")
            }
            Op::MinAll => {
                let a = g(inputs, 0, cache);
                let axes: Vec<usize> = (0..a.shape.dims().len()).collect();
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Min, &a.layout(), &axes)
                    .expect("MinAll")
            }
            Op::SumDim(d) | Op::MeanDim(d) => {
                let a = g(inputs, 0, cache);
                let mut r = a.storage.reduce_op(
                    fuel_core_types::op::ReduceOp::Sum, &a.layout(), &[*d],
                ).expect("SumDim");
                if matches!(op, Op::MeanDim(_)) {
                    let n = a.shape.dims()[*d] as f64;
                    r = r.affine(&Layout::contiguous(shape), 1.0 / n, 0.0)
                        .expect("MeanDim scale");
                }
                r
            }
            Op::MaxDim(d) => {
                let a = g(inputs, 0, cache);
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Max, &a.layout(), &[*d])
                    .expect("MaxDim")
            }
            Op::MinDim(d) => {
                let a = g(inputs, 0, cache);
                a.storage.reduce_op(fuel_core_types::op::ReduceOp::Min, &a.layout(), &[*d])
                    .expect("MinDim")
            }

            // indexing
            Op::IndexSelect { dim } => {
                let (src, ids) = (g(inputs, 0, cache), g(inputs, 1, cache));
                src.storage.index_select(&ids.storage, &src.layout(), &ids.layout(), *dim)
                    .expect("IndexSelect")
            }
            Op::Gather { dim } => {
                let (src, ids) = (g(inputs, 0, cache), g(inputs, 1, cache));
                src.storage.gather(&src.layout(), &ids.storage, &ids.layout(), *dim)
                    .expect("Gather")
            }

            // concat / slice
            Op::Concat { dim } => return self.eval_concat(*dim, inputs, shape, cache),
            Op::Slice { dim, start, len } => {
                let a = g(inputs, 0, cache);
                return self.eval_slice(*dim, *start, *len, a, shape);
            }

            // Everything else: CPU fallback via the reference backend.
            _ => {
                return self.cpu_fallback(inputs, shape, cache, |ni, ns, cc| {
                    fuel_reference_backend::exec::eval_node_with_op(op, ni, ns, dtype, cc)
                });
            }
        };

        GpuTensor { storage: result_storage, shape: shape.clone() }
    }

    fn eval_const(&mut self, data: &ConstData, shape: &Shape) -> GpuTensor {
        let ptr = const_data_arc_ptr(data);
        if let Some(cached) = self.const_cache.get(&ptr) {
            return GpuTensor {
                storage: cached.storage.try_clone(&cached.layout()).expect("const clone"),
                shape: shape.clone(),
            };
        }
        let cpu_buf = const_data_to_host_buffer(data);
        let gpu = self.device.storage_from_cpu_storage(&cpu_buf)
            .expect("Const H2D");
        let gt = GpuTensor { storage: gpu, shape: shape.clone() };
        self.const_cache.insert(ptr, GpuTensor {
            storage: gt.storage.try_clone(&gt.layout()).expect("const cache"),
            shape: shape.clone(),
        });
        gt
    }

    fn do_permute(&self, a: &GpuTensor, axes: &[usize], out_shape: &Shape) -> GpuTensor {
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: DimVec = DimVec::from_elem(0, rank);
        let mut s = 1usize;
        for i in (0..rank).rev() {
            strides[i] = s;
            s *= in_dims[i];
        }
        let permuted_strides: DimVec = axes.iter().map(|&ax| strides[ax]).collect();
        let permuted_dims: Vec<usize> = axes.iter().map(|&ax| in_dims[ax]).collect();
        let src_layout = Layout::new(
            Shape::from_dims(&permuted_dims),
            permuted_strides,
            0,
        );
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("permute alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("permute copy");
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    fn do_broadcast(&self, a: &GpuTensor, target: &Shape) -> GpuTensor {
        let src_dims = a.shape.dims();
        let dst_dims = target.dims();
        if src_dims == dst_dims {
            return GpuTensor {
                storage: a.storage.try_clone(&a.layout()).expect("broadcast noop"),
                shape: target.clone(),
            };
        }
        let pad = dst_dims.len() - src_dims.len();
        let mut strides: DimVec = DimVec::from_elem(0, dst_dims.len());
        let mut s = 1usize;
        for i in (0..src_dims.len()).rev() {
            if src_dims[i] == dst_dims[pad + i] {
                strides[pad + i] = s;
            }
            s *= src_dims[i];
        }
        let src_layout = Layout::new(target.clone(), strides, 0);
        let mut dst = self.device.zeros_impl(target, a.storage.dtype())
            .expect("broadcast alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("broadcast copy");
        GpuTensor { storage: dst, shape: target.clone() }
    }

    fn eval_concat(
        &self,
        dim: usize,
        inputs: &[NodeId],
        out_shape: &Shape,
        cache: &HashMap<NodeId, GpuTensor>,
    ) -> GpuTensor {
        let a = g(inputs, 0, cache);
        let b = g(inputs, 1, cache);
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("concat alloc");
        a.storage.copy_strided_src(&mut dst, 0, &a.layout()).expect("concat a");
        let a_dim = a.shape.dims()[dim];
        let inner: usize = out_shape.dims()[dim + 1..].iter().product::<usize>().max(1);
        let offset = a_dim * inner;
        let outer: usize = out_shape.dims()[..dim].iter().product::<usize>().max(1);
        if outer == 1 {
            b.storage.copy_strided_src(&mut dst, offset, &b.layout()).expect("concat b");
        } else {
            let out_stride: usize = out_shape.dims()[dim..].iter().product();
            let b_stride: usize = b.shape.dims()[dim..].iter().product();
            for o in 0..outer {
                let dst_off = o * out_stride + a_dim * inner;
                let b_layout = Layout::contiguous_with_offset(
                    &Shape::from_dims(&[b_stride]),
                    o * b_stride,
                );
                b.storage.copy_strided_src(&mut dst, dst_off, &b_layout)
                    .expect("concat b slice");
            }
        }
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    fn eval_slice(
        &self,
        dim: usize,
        start: usize,
        _len: usize,
        a: &GpuTensor,
        out_shape: &Shape,
    ) -> GpuTensor {
        let in_dims = a.shape.dims();
        let rank = in_dims.len();
        let mut strides: DimVec = DimVec::from_elem(0, rank);
        let mut s = 1usize;
        for i in (0..rank).rev() {
            strides[i] = s;
            s *= in_dims[i];
        }
        let offset = start * strides[dim];
        let src_layout = Layout::new(out_shape.clone(), strides, offset);
        let mut dst = self.device.zeros_impl(out_shape, a.storage.dtype())
            .expect("slice alloc");
        a.storage.copy_strided_src(&mut dst, 0, &src_layout)
            .expect("slice copy");
        GpuTensor { storage: dst, shape: out_shape.clone() }
    }

    /// Download inputs to CPU, run a reference-backend op, re-upload.
    fn cpu_fallback(
        &self,
        inputs: &[NodeId],
        shape: &Shape,
        cache: &HashMap<NodeId, GpuTensor>,
        f: impl FnOnce(&[NodeId], &Shape, &HashMap<NodeId, AnyRef>) -> AnyRef,
    ) -> GpuTensor {
        let mut cpu_cache: HashMap<NodeId, AnyRef> = HashMap::new();
        for &id in inputs {
            let gt = cache.get(&id).expect("cpu_fallback: missing input");
            let cpu_buf = gt.storage.to_cpu_storage().expect("D2H fallback");
            cpu_cache.insert(id, host_buffer_to_any_ref(cpu_buf, &gt.shape));
        }
        let result = f(inputs, shape, &cpu_cache);
        let out_buf = any_ref_to_host_buffer(result);
        let gpu = self.device.storage_from_cpu_storage(&out_buf)
            .expect("H2D fallback");
        GpuTensor { storage: gpu, shape: shape.clone() }
    }
}

// --- free-function helpers ---

fn g<'a>(inputs: &[NodeId], idx: usize, cache: &'a HashMap<NodeId, GpuTensor>) -> &'a GpuTensor {
    cache.get(&inputs[idx]).expect("topo order missing input")
}

fn gpu_to_ref_f32(gt: GpuTensor) -> RefTensor<f32> {
    let cpu = gt.storage.to_cpu_storage().expect("D2H");
    match cpu {
        fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape),
        other => panic!("gpu_to_ref_f32: dtype {:?}", other.dtype()),
    }
}

fn gpu_to_ref_f32_ref(gt: &GpuTensor) -> RefTensor<f32> {
    let cpu = gt.storage.to_cpu_storage().expect("D2H");
    match cpu {
        fuel_core_types::HostBuffer::F32(v) => RefTensor::from_vec(v, gt.shape.clone()),
        other => panic!("gpu_to_ref_f32: dtype {:?}", other.dtype()),
    }
}

fn const_data_arc_ptr(data: &ConstData) -> usize {
    match data {
        ConstData::F32(v) => std::sync::Arc::as_ptr(v) as *const f32 as usize,
        ConstData::F64(v) => std::sync::Arc::as_ptr(v) as *const f64 as usize,
        ConstData::BF16(v) => std::sync::Arc::as_ptr(v) as *const () as usize,
        ConstData::F16(v) => std::sync::Arc::as_ptr(v) as *const () as usize,
        ConstData::U32(v) => std::sync::Arc::as_ptr(v) as *const u32 as usize,
    }
}

fn const_data_to_host_buffer(data: &ConstData) -> fuel_core_types::HostBuffer {
    use fuel_core_types::HostBuffer;
    match data {
        ConstData::F32(v) => HostBuffer::F32(v.to_vec()),
        ConstData::F64(v) => HostBuffer::F64(v.to_vec()),
        ConstData::BF16(v) => HostBuffer::BF16(v.to_vec()),
        ConstData::F16(v) => HostBuffer::F16(v.to_vec()),
        ConstData::U32(v) => HostBuffer::U32(v.to_vec()),
    }
}

fn host_buffer_to_any_ref(buf: fuel_core_types::HostBuffer, shape: &Shape) -> AnyRef {
    match buf {
        fuel_core_types::HostBuffer::F32(v) => AnyRef::F32(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F64(v) => AnyRef::F64(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::BF16(v) => AnyRef::BF16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::F16(v) => AnyRef::F16(RefTensor::from_vec(v, shape.clone())),
        fuel_core_types::HostBuffer::U32(v) => AnyRef::U32(RefTensor::from_vec(v, shape.clone())),
        _ => panic!("host_buffer_to_any_ref: unsupported dtype"),
    }
}

fn any_ref_to_host_buffer(any: AnyRef) -> fuel_core_types::HostBuffer {
    use fuel_core_types::HostBuffer;
    match any {
        AnyRef::F32(t) => HostBuffer::F32(t.into_vec()),
        AnyRef::F64(t) => HostBuffer::F64(t.into_vec()),
        AnyRef::BF16(t) => HostBuffer::BF16(t.into_vec()),
        AnyRef::F16(t) => HostBuffer::F16(t.into_vec()),
        AnyRef::U32(t) => HostBuffer::U32(t.into_vec()),
    }
}

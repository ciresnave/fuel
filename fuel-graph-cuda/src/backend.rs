//! `GraphBackend` implementation for CUDA GPUs.

use fuel_core_types::{DType, Layout, Shape};
use crate::{CudaDevice, CudaStorage};
use fuel_graph_executor::{BinaryOp, GraphBackend, UnaryOp};

/// CUDA backend: matmul via cublas, unary/binary via CUDA kernels,
/// softmax via fused reduce kernel, everything else via reference
/// backend CPU fallback (handled by the generic executor).
pub struct CudaBackend {
    pub device: CudaDevice,
}

impl CudaBackend {
    pub fn new(device: CudaDevice) -> Self {
        Self { device }
    }
}

impl GraphBackend for CudaBackend {
    type Storage = CudaStorage;

    fn alloc_zeros(&self, shape: &Shape, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        self.device.zeros_impl(shape, dtype)
    }

    fn upload(&self, buf: &fuel_core_types::HostBuffer, _shape: &Shape) -> fuel_core_types::Result<Self::Storage> {
        self.device.storage_from_cpu_storage(buf)
    }

    fn download(&self, storage: &Self::Storage) -> fuel_core_types::Result<fuel_core_types::HostBuffer> {
        storage.to_cpu_storage()
    }

    fn try_clone(&self, storage: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        storage.try_clone(layout)
    }

    fn copy_strided_src(
        &self,
        src: &Self::Storage,
        dst: &mut Self::Storage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> fuel_core_types::Result<()> {
        src.copy_strided_src(dst, dst_offset, src_layout)
    }

    fn storage_dtype(&self, storage: &Self::Storage) -> DType {
        storage.dtype()
    }

    fn matmul(
        &self,
        a: &Self::Storage, b: &Self::Storage,
        bmnk: (usize, usize, usize, usize),
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        a.matmul(b, bmnk, la, lb)
    }

    fn conv2d(
        &self,
        input:  &Self::Storage,
        weight: &Self::Storage,
        input_layout:  &Layout,
        weight_layout: &Layout,
        stride:  (usize, usize),
        padding: (usize, usize),
        groups:  usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        // Executor pre-screened: symmetric stride/padding. Grouped /
        // depthwise convolution flows through cuDNN's native group_count
        // (no per-group chunking) when the `cudnn` feature is on; the
        // im2col fallback path still requires groups==1.
        if stride.0 != stride.1 || padding.0 != padding.1 {
            fuel_core_types::bail!(
                "CudaBackend::conv2d: only symmetric stride/padding supported (got stride={stride:?} padding={padding:?})"
            );
        }
        let i_dims = input_layout.shape().dims();
        let w_dims = weight_layout.shape().dims();
        if i_dims.len() != 4 || w_dims.len() != 4 {
            fuel_core_types::bail!(
                "CudaBackend::conv2d: expected rank-4 input + weight, got {i_dims:?} and {w_dims:?}"
            );
        }
        let params = fuel_core_types::conv::ParamsConv2D {
            b_size:  i_dims[0],
            i_h:     i_dims[2],
            i_w:     i_dims[3],
            c_in:    w_dims[1],
            k_h:     w_dims[2],
            k_w:     w_dims[3],
            c_out:   w_dims[0],
            padding: padding.0,
            stride:  stride.0,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        };
        input.conv2d(input_layout, weight, weight_layout, &params)
    }

    fn unary(&self, op: UnaryOp, a: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        let kernel = match op {
            UnaryOp::Neg => "uneg",
            UnaryOp::Sqr => "usqr",
            UnaryOp::Sqrt => "usqrt",
            UnaryOp::Exp => "uexp",
            UnaryOp::Log => "ulog",
            UnaryOp::Sin => "usin",
            UnaryOp::Cos => "ucos",
            UnaryOp::Tanh => "utanh",
            UnaryOp::Sigmoid => "usigmoid",
            UnaryOp::Silu => "usilu",
            UnaryOp::Gelu => "ugelu",
            UnaryOp::Relu => "urelu",
            UnaryOp::Step => "ustep",
        };
        a.unary_by_name(kernel, layout)
    }

    fn binary(
        &self, op: BinaryOp,
        a: &Self::Storage, b: &Self::Storage,
        la: &Layout, lb: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        let kernel = match op {
            BinaryOp::Add => "badd",
            BinaryOp::Sub => "bsub",
            BinaryOp::Mul => "bmul",
            BinaryOp::Div => "bdiv",
            BinaryOp::Maximum => "bmaximum",
            BinaryOp::Minimum => "bminimum",
        };
        a.binary_by_name(b, la, lb, kernel)
    }

    fn affine(&self, a: &Self::Storage, layout: &Layout, mul: f64, add: f64) -> fuel_core_types::Result<Self::Storage> {
        a.affine(layout, mul, add)
    }

    fn powf(&self, a: &Self::Storage, layout: &Layout, exp: f64) -> fuel_core_types::Result<Self::Storage> {
        a.powf(layout, exp)
    }

    fn cast(&self, a: &Self::Storage, layout: &Layout, dtype: DType) -> fuel_core_types::Result<Self::Storage> {
        a.to_dtype(layout, dtype)
    }

    fn reduce(
        &self, op: fuel_core_types::op::ReduceOp,
        a: &Self::Storage, layout: &Layout,
        dims: &[usize],
    ) -> fuel_core_types::Result<Self::Storage> {
        a.reduce_op(op, layout, dims)
    }

    fn softmax_last_dim(&self, a: &Self::Storage, layout: &Layout) -> fuel_core_types::Result<Self::Storage> {
        a.softmax_last_dim(layout)
    }

    fn index_select(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        src.index_select(ids, src_l, ids_l, dim)
    }

    fn gather(
        &self, src: &Self::Storage, ids: &Self::Storage,
        src_l: &Layout, ids_l: &Layout, dim: usize,
    ) -> fuel_core_types::Result<Self::Storage> {
        src.gather(src_l, ids, ids_l, dim)
    }

    fn rope(
        &self,
        x: &Self::Storage,
        cos: &Self::Storage,
        sin: &Self::Storage,
        x_layout: &Layout,
        _cos_layout: &Layout,
        _sin_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        x.rope(cos, sin, x_layout)
    }

    fn rms_norm_last_dim(
        &self, a: &Self::Storage, layout: &Layout, eps: f64,
    ) -> fuel_core_types::Result<Self::Storage> {
        a.rms_norm_last_dim(layout, eps)
    }

    fn matmul_q4_0(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        a.matmul_q4_0(w_q_bytes, k, n, a_layout)
    }

    fn matmul_q4_km(
        &self,
        a: &Self::Storage,
        w_q_bytes: &Self::Storage,
        k: usize,
        n: usize,
        a_layout: &Layout,
    ) -> fuel_core_types::Result<Self::Storage> {
        a.matmul_q4_km(w_q_bytes, k, n, a_layout)
    }
}

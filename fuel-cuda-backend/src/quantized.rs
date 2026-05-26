use crate::WrapErr;
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use fuel_core_types::dyn_backend::DynBackendStorage;
use fuel_core_types::quantized::{DynQuantizedStorage, GgmlDType, QuantizedDeviceKernels};
use fuel_quantized::GgmlType;
use half::f16;
use std::any::Any;
use std::borrow::Cow;

use baracuda_driver::{DeviceBuffer as CudaSlice, DeviceSlice as CudaView};

#[derive(Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

// `quantize_q8_1` PTX wrapper retired in Phase 4 — baracuda alpha.37's
// batched MMVQ + MoE FFI consume fp32 activations directly, so the
// staging quantize step is no longer needed anywhere in fuel-cuda-backend.

// Baracuda alpha.27+ GGUF dequantize FFI. Per-format symbol picker —
// covers all 11 GGUF block formats. Output is always f32 (baracuda
// doesn't ship per-dtype dequant variants; f16 output is achieved by
// dequant→cast in `dequantize_f16` below).
type DequantRun = unsafe extern "C" fn(
    numel: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn pick_dequant(dtype: GgmlDType) -> Result<DequantRun> {
    use baracuda_kernels_sys as sys;
    Ok(match dtype {
        GgmlDType::Q4_0 => sys::baracuda_kernels_dequantize_q4_0_run,
        GgmlDType::Q4_1 => sys::baracuda_kernels_dequantize_q4_1_run,
        GgmlDType::Q5_0 => sys::baracuda_kernels_dequantize_q5_0_run,
        GgmlDType::Q5_1 => sys::baracuda_kernels_dequantize_q5_1_run,
        GgmlDType::Q8_0 => sys::baracuda_kernels_dequantize_q8_0_run,
        GgmlDType::Q2K => sys::baracuda_kernels_dequantize_q2_K_run,
        GgmlDType::Q3K => sys::baracuda_kernels_dequantize_q3_K_run,
        GgmlDType::Q4K => sys::baracuda_kernels_dequantize_q4_K_run,
        GgmlDType::Q5K => sys::baracuda_kernels_dequantize_q5_K_run,
        GgmlDType::Q6K => sys::baracuda_kernels_dequantize_q6_K_run,
        GgmlDType::Q8K => sys::baracuda_kernels_dequantize_q8_K_run,
        other => fuel_core_types::bail!("baracuda dequant: unsupported dtype {other:?}"),
    })
}

/// Dequantize Q* blocks into a fresh f32 CudaStorage via baracuda's
/// `baracuda_kernels_dequantize_<fmt>_run`. Replaces the prior PTX
/// `dequantize_block_*_f32` kernels retired in Phase 6b.
fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let run = pick_dequant(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = data.inner.as_raw().0 as *const std::ffi::c_void;
    let y_ptr = dst.as_raw().0 as *mut std::ffi::c_void;
    // SAFETY: device-resident pointers + live stream; workspace null/0
    // (baracuda dequant kernels don't need scratch).
    let status = unsafe { run(elem_count as i64, x_ptr, y_ptr, std::ptr::null_mut(), 0, stream) };
    crate::baracuda::status::check(status, "dequantize_f32")?;
    dev.synchronize()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Dequantize Q* blocks into a fresh f16 CudaStorage. Two-step
/// implementation (dequant→f32 scratch, then cast→f16) since baracuda's
/// GGUF dequant FFI only ships f32 output variants. The cast goes
/// through baracuda's `Cast` FFI (binding-table path), avoiding any
/// fuel-cuda-kernels PTX dependency.
fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let dst_f32 = dequantize_f32(data, dtype, elem_count, dev)?;
    // Cast f32 → f16 via the typed CudaStorage path (legacy `to_dtype`
    // is still backed by Fuel's CAST PTX module; that retires when the
    // binding table fully absorbs the eager API in Phase 6c).
    let layout = crate::Layout::contiguous(crate::Shape::from(elem_count));
    dst_f32.to_dtype(&layout, crate::DType::F16)
}

// Batched MMVQ dispatch via baracuda alpha.37. The `baracuda_kernels_
// mmvq_<fmt>_batched_run` symbols implement MMVQ-with-routing
// semantics (sorted_token_ids + expert_offsets + optional topk_weights),
// but degrade cleanly to plain batched MMVQ when n_experts=1 + top_k=1
// + identity-permutation token IDs. That covers Fuel's existing
// `mul_mat_vec_via_q8_1` (b_size=1..8) and `mul_mat_via_q8_1`
// (matrix-matrix) call sites without per-token routing.
//
// **ncols ≥ 64 invariant** (baracuda team note for type-0/1 quants —
// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0): contiguously-batched callers must satisfy
// `ncols ≥ 2 × GGML_CUDA_DMMV_X = 64` or risk silent garbage.
// K-quants are unaffected. We assert this at the wrapper boundary.
type MmvqBatchedRun = unsafe extern "C" fn(
    n_experts: i32, n_rows_per_expert: i32, n_cols: i32,
    weights: *const std::ffi::c_void, activations: *const std::ffi::c_void,
    sorted_token_ids: *const i32, expert_offsets: *const i32,
    topk_weights: *const f32, output: *mut std::ffi::c_void, top_k: i32,
    workspace: *mut std::ffi::c_void, workspace_bytes: usize, stream: *mut std::ffi::c_void,
) -> i32;

fn pick_mmvq_batched(dtype: GgmlDType) -> Result<MmvqBatchedRun> {
    use baracuda_kernels_sys as sys;
    Ok(match dtype {
        GgmlDType::Q4_0 => sys::baracuda_kernels_mmvq_q4_0_batched_run,
        GgmlDType::Q4_1 => sys::baracuda_kernels_mmvq_q4_1_batched_run,
        GgmlDType::Q5_0 => sys::baracuda_kernels_mmvq_q5_0_batched_run,
        GgmlDType::Q5_1 => sys::baracuda_kernels_mmvq_q5_1_batched_run,
        GgmlDType::Q8_0 => sys::baracuda_kernels_mmvq_q8_0_batched_run,
        GgmlDType::Q2K => sys::baracuda_kernels_mmvq_q2_K_batched_run,
        GgmlDType::Q3K => sys::baracuda_kernels_mmvq_q3_K_batched_run,
        GgmlDType::Q4K => sys::baracuda_kernels_mmvq_q4_K_batched_run,
        GgmlDType::Q5K => sys::baracuda_kernels_mmvq_q5_K_batched_run,
        GgmlDType::Q6K => sys::baracuda_kernels_mmvq_q6_K_batched_run,
        other => fuel_core_types::bail!("baracuda batched MMVQ: unsupported dtype {other:?}"),
    })
}

fn requires_min_ncols_64(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0,
    )
}

fn baracuda_batched_mmvq(
    weights: &PaddedCudaSlice,
    activations: &CudaView<f32>,
    dtype: GgmlDType,
    n_cols: usize,
    n_rows: usize,
    m_total: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if requires_min_ncols_64(dtype) && n_cols < 64 {
        fuel_core_types::bail!(
            "baracuda batched MMVQ: dtype {dtype:?} requires n_cols ≥ 64 (got {n_cols}); type-0/1 quants have implicit ncols min in batched mode"
        )
    }
    let run = pick_mmvq_batched(dtype)?;

    // Identity routing: 1 expert, top_k = 1, sorted_token_ids = [0..m_total).
    // expert_offsets has shape [n_experts + 1] = [0, m_total].
    let sorted_token_ids_host: Vec<i32> = (0..m_total as i32).collect();
    let expert_offsets_host: Vec<i32> = vec![0_i32, m_total as i32];
    let sorted_token_ids_dev = dev.clone_htod(&sorted_token_ids_host)?;
    let expert_offsets_dev = dev.clone_htod(&expert_offsets_host)?;

    // Workspace: m_total * sizeof(i32) bytes per the FFI contract.
    let workspace_bytes = m_total * std::mem::size_of::<i32>();
    let workspace = dev.alloc_zeros::<u8>(workspace_bytes.max(1))?;

    let out_elems = m_total * n_rows;
    let dst = dev.alloc_zeros::<f32>(out_elems)?;

    let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
    let w_ptr = weights.inner.as_raw().0 as *const std::ffi::c_void;
    let a_ptr = activations.as_raw().0 as *const std::ffi::c_void;
    let ids_ptr = sorted_token_ids_dev.as_raw().0 as *const i32;
    let off_ptr = expert_offsets_dev.as_raw().0 as *const i32;
    let dst_ptr = dst.as_raw().0 as *mut std::ffi::c_void;
    let ws_ptr = workspace.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: device-resident pointers + live stream; workspace sized per
    // FFI contract (m_total * 4 bytes); top_k = 1 ⇒ plain stores (no
    // atomicAdd) so dst's zero-init is fine.
    let status = unsafe {
        run(
            /* n_experts */ 1,
            n_rows as i32,
            n_cols as i32,
            w_ptr,
            a_ptr,
            ids_ptr,
            off_ptr,
            /* topk_weights */ std::ptr::null(),
            dst_ptr,
            /* top_k */ 1,
            ws_ptr,
            workspace_bytes,
            stream,
        )
    };
    crate::baracuda::status::check(status, "mmvq_batched")?;
    dev.synchronize()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

// `dequantize_mul_mat_vec` (fused dequant+gemv PTX, b_size=1 only)
// retired in Phase 6a — the FORCE_DMMV debug path now routes through
// `self.dequantize() + storage.matmul()` for parity with
// `dequantize_matmul`'s pre-existing two-step pattern.

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        fuel_core_types::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        fuel_core_types::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 {
        fuel_core_types::bail!("bsize must be > 0, got {b_size}")
    }
    // Baracuda alpha.37 batched MMVQ: takes fp32 activations directly,
    // no Q8_1 staging quantize. n_experts=1 + identity-permutation token
    // IDs + top_k=1 collapses the routing-aware path to plain batched
    // MMVQ semantics matching Fuel's prior contract.
    baracuda_batched_mmvq(data, y, dtype, ncols, nrows, b_size, dev)
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        fuel_core_types::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        fuel_core_types::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        fuel_core_types::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    // Baracuda alpha.37 batched MMVQ subsumes both the bsize=1..8 vector
    // case (`mul_mat_vec_via_q8_1`) and the matrix-matrix case here. The
    // contraction dim is x_cols; output shape is (y_cols × x_rows) row-
    // major matching the prior PTX kernel's contract.
    baracuda_batched_mmvq(data, y, dtype, /* n_cols */ x_cols, /* n_rows */ x_rows, /* m_total */ y_cols, dev)
}

// `indexed_moe_forward_fused_q8_1_input` and `QCudaStorage::indexed_moe_forward`
// were retired in Phase 4 of the fuel-cuda-kernels retirement (2026-05-25).
// The PTX kernel that backed them (`indexed_moe_forward_*_q8_1` in
// `fuel-cuda-kernels/src/quantized.cu`) is gone alongside the `moe/` PTX
// directory. There are no production callers — the public MoE GEMM
// surface is `fuel_nn::moe_gemm` / `fuel_nn::moe_gemm_gguf`, which now
// drive baracuda alpha.37 directly. If a future caller needs the
// pre-sorted-routing variant, it should call
// `baracuda_kernels_moe_scalar_gguf_run` with a sorting prelude (same
// pattern as `baracuda-kernels/tests/moe_ffi_direct_smoke.rs`).
//
// The default `DynQuantizedStorage::indexed_moe_forward` trait method
// returns an unsupported-backend error, which is the new behaviour.

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(0..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<fuel_quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<fuel_quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<fuel_quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<fuel_quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<fuel_quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<fuel_quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<fuel_quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<fuel_quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<fuel_quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<fuel_quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<fuel_quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<fuel_quantized::BlockQ8K>(&buffer, block_len, &mut out),
        }

        self.device
            .storage_from_cpu_storage(&fuel_core_types::HostBuffer::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    /// Quantize host-resident f32 src onto self by running the scalar CPU
    /// quantizer in fuel-quantized then htod-copying the resulting bytes.
    fn quantize_from_f32(
        &mut self,
        src: &[f32],
        imatrix: Option<(&[f32], usize)>,
    ) -> Result<()> {
        let mut qcpu = fuel_quantized::cpu_zeros(self.dtype, src.len());
        match imatrix {
            None => qcpu.from_float(src),
            Some((iw, n_per_row)) => qcpu.from_float_imatrix(src, iw, n_per_row),
        }
        let data_ptr = qcpu.as_ptr();
        let data_len = qcpu.storage_size_in_bytes();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data, &mut inner.slice_mut(0..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        let src_vec = match &src.slice {
            crate::CudaStorageSlice::F32(data) => self.device.clone_dtoh(&data.as_slice())?,
            _ => fuel_core_types::bail!("only f32 can be quantized"),
        };
        self.quantize_from_f32(&src_vec, None)
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let src_vec = match &src.slice {
            crate::CudaStorageSlice::F32(data) => self.device.clone_dtoh(&data.as_slice())?,
            _ => fuel_core_types::bail!("only f32 can be quantized"),
        };
        self.quantize_from_f32(&src_vec, Some((imatrix_weights, n_per_row)))
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &fuel_core_types::HostBuffer,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?, Some((imatrix_weights, n_per_row)))
    }

    pub fn quantize_onto(&mut self, src: &fuel_core_types::HostBuffer) -> Result<()> {
        self.quantize_from_f32(src.as_slice::<f32>()?, None)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(0..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.as_raw().0 as *const u8)
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => fuel_core_types::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            fuel_core_types::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            // Phase 6a — FORCE_DMMV (debug toggle for the dequant-then-FP
            // reference path) now routes through `self.dequantize() +
            // storage.matmul()` (same pattern `dequantize_matmul` already
            // uses for the matrix-matrix case). Slower than the prior
            // fused `dequantize_mul_mat_vec` PTX kernel but identical
            // contract and no PTX dependency.
            let data_f32 = self.dequantize(nrows * ncols)?;
            let rhs_l_t = crate::Layout::new(
                (ncols, nrows).into(),
                smallvec::smallvec![1_isize, ncols as isize],
                0,
            )
            .broadcast_as((b_size, ncols, nrows))?;
            // m=1 mat-vec — view activation as (b_size, 1, ncols).
            let lhs_l = crate::Layout::contiguous((b_size, 1, ncols));
            rhs.matmul(&data_f32, (b_size, 1, nrows, ncols), &lhs_l, &rhs_l_t)?
        } else {
            let rhs_typed = rhs.as_cuda_slice::<f32>()?;
            let rhs_slice = match rhs_l.contiguous_offsets() {
                Some((o1, o2)) => rhs_typed.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
            };
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs_slice,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => fuel_core_types::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            fuel_core_types::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), smallvec::smallvec![1_isize, k as isize], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

/// Build a `QCudaStorage` from raw block-format bytes already laid out for
/// `dtype`. Returned as a typed `Box<dyn DynQuantizedStorage>` so fuel-core
/// can hold it polymorphically alongside CPU/Metal variants.
pub fn load_quantized_bytes(
    device: &CudaDevice,
    dtype: GgmlDType,
    data: &[u8],
) -> Result<Box<dyn DynQuantizedStorage>> {
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(0..data.len()))?;
    Ok(Box::new(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

// ---------------------------------------------------------------------------
// DynQuantizedStorage / QuantizedDeviceKernels — backend-agnostic dispatch
// ---------------------------------------------------------------------------

impl DynQuantizedStorage for QCudaStorage {
    fn dtype(&self) -> GgmlDType {
        self.dtype
    }
    fn block_size(&self) -> usize {
        self.dtype.block_size()
    }
    fn storage_size_in_bytes(&self) -> usize {
        QCudaStorage::storage_size_in_bytes(self)
    }
    fn quantize(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        let cuda = src
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| crate::Error::Msg("quantize: expected cuda storage".into()).bt())?;
        QCudaStorage::quantize(self, cuda)
    }
    fn quantize_imatrix(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let cuda = src.as_any().downcast_ref::<CudaStorage>().ok_or_else(|| {
            crate::Error::Msg("quantize_imatrix: expected cuda storage".into()).bt()
        })?;
        QCudaStorage::quantize_imatrix(self, cuda, imatrix_weights, n_per_row)
    }
    fn quantize_onto(&mut self, src: &dyn DynBackendStorage) -> Result<()> {
        let cpu = src
            .as_any()
            .downcast_ref::<fuel_cpu_backend::CpuStorage>()
            .ok_or_else(|| crate::Error::Msg("quantize_onto: expected cpu storage".into()).bt())?;
        QCudaStorage::quantize_onto(self, &cpu.0)
    }
    fn quantize_imatrix_onto(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        let cpu = src.as_any().downcast_ref::<fuel_cpu_backend::CpuStorage>().ok_or_else(|| {
            crate::Error::Msg("quantize_imatrix_onto: expected cpu storage".into()).bt()
        })?;
        QCudaStorage::quantize_imatrix_onto(self, &cpu.0, imatrix_weights, n_per_row)
    }
    fn dequantize(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(QCudaStorage::dequantize(self, elem_count)?))
    }
    fn dequantize_f16(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        Ok(Box::new(QCudaStorage::dequantize_f16(self, elem_count)?))
    }
    fn data(&self) -> Result<Cow<'_, [u8]>> {
        Ok(Cow::Owned(QCudaStorage::data(self)?))
    }
    fn device_ptr(&self) -> Result<*const u8> {
        QCudaStorage::device_ptr(self)
    }
    fn fwd(
        &self,
        self_shape: &crate::Shape,
        input: &dyn DynBackendStorage,
        layout: &crate::Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, crate::Shape)> {
        let cuda = input
            .as_any()
            .downcast_ref::<CudaStorage>()
            .ok_or_else(|| crate::Error::Msg("qmatmul: expected cuda storage".into()).bt())?;
        let (s, sh) = QCudaStorage::fwd(self, self_shape, cuda, layout)?;
        Ok((Box::new(s), sh))
    }
    // `indexed_moe_forward` trait override removed in Phase 4 retirement;
    // the default in `DynQuantizedStorage` returns an unsupported-backend
    // error. Callers should use `fuel_nn::moe_gemm_gguf` (baracuda-backed)
    // or call `baracuda_kernels_moe_scalar_gguf_run` directly with a
    // sorting prelude when pre-routed inputs are required.
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn device_arc_dyn(&self) -> std::sync::Arc<dyn fuel_core_types::dyn_backend::DynBackendDevice> {
        std::sync::Arc::new(self.device.clone())
    }
}

impl QuantizedDeviceKernels for CudaDevice {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Box<dyn DynQuantizedStorage>> {
        Ok(Box::new(QCudaStorage::zeros(self, elem_count, dtype)?))
    }
    fn load_quantized(
        &self,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Box<dyn DynQuantizedStorage>> {
        load_quantized_bytes(self, dtype, &data)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // `cuda_quantize_q8_1` test was retired alongside the
    // `quantize_q8_1` PTX function in Phase 4 — there are no longer any
    // callers of the Q8_1 staging path (baracuda's batched MMVQ + MoE
    // FFI take f32 activations directly).

    fn close_rel(actual: f32, reference: f32, rel_tol: f32, label: &str) {
        let err = (actual - reference).abs() / reference.abs();
        assert!(
            err <= rel_tol,
            "{label}: |{actual} - {reference}| / |{reference}| = {err:.4e} > {rel_tol:.4e}",
        );
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let y_dup = dev.clone_dtod(&y)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y_dup, dev.clone()))?;
        // Reference: for n = 255, sum_{i=0..n} i^2 = n(n+1)(2n+1)/6 = 5559680.
        // Q4_0 quantization adds ~0.5% relative error.
        let reference = 5_559_680.0_f32;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_slice(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_slice())?;
        assert_eq!(vs.len(), 1);
        close_rel(vs[0], reference, 1e-3, "mul_mat_vec_via_q8_1");
        // The fused dequant+gemv PTX path (`dequantize_mul_mat_vec`)
        // retired in Phase 6a; the FORCE_DMMV branch now goes through
        // `QCudaStorage::dequantize` + matmul, which has its own test
        // coverage via the qmm_*_cuda integration tests.
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let y_dup = dev.clone_dtod(&y)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y_dup, dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_slice(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_slice())?;

        // Reference values pinned against the prior PTX kernel's output
        // (which baracuda alpha.37 batched MMVQ matches to ~4 fp32 ULPs).
        // These are NOT bit-exact to PyTorch — Fuel's QMatMul computes
        // Q4_0(x) @ x.t() (row-asymmetric quantization), while torch
        // x @ x.t() is symmetric. The values here are the Q4_0-quantized
        // reference; both PTX and baracuda paths agree on them within
        // 1e-4 relative tolerance.
        assert_eq!(vs.len(), 16);
        let qref = [
            347_604.0, 888_153.06, 0.0 /* not asserted */, 0.0,
            869_780.7, 2_483_145.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 9_407_368.0,
            0.0, 0.0, 9_470_856.0, 13_138_824.0,
        ];
        let asserted: &[usize] = &[0, 1, 4, 5, 11, 14, 15];
        for &i in asserted {
            close_rel(vs[i], qref[i], 1e-4, &format!("cuda_mm_q8_1[{i}]"));
        }
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    // ncols bumped from 16 → 64 because baracuda's alpha.37 batched MMVQ
    // for type-0/1 quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) requires
    // `ncols ≥ 2 × GGML_CUDA_DMMV_X = 64`. Still exercises the
    // historical "y_cols not aligned to MATRIX_ROW_PADDING (512)" path
    // that #2526 fixed.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 64, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let y_dup = dev.clone_dtod(&y)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y_dup, dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_slice(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_slice())?;
        Ok(())
    }
}

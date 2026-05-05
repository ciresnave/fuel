//! Byte-level CUDA kernels — Phase 7.5 unified-storage migration.
//!
//! These kernels operate on `CudaStorageBytes` (raw `DeviceBuffer<u8>`)
//! rather than the dtype-tagged legacy `CudaStorage` enum. Dispatch
//! to the right CUDA function happens via wrappers in
//! `fuel-storage::dispatch::register_cuda_kernels`; the typed kernel
//! functions in `fuel-cuda-kernels` are launched by passing
//! `&DeviceBuffer<u8>` as the kernel arg — at the CUDA driver level
//! the typed pointer (`f32*`, `f64*`, etc.) and the byte pointer have
//! the same value, and the kernel's compiled code interprets the
//! bytes per its declared type.
//!
//! The kernels in `fuel-cuda-kernels` (e.g. `badd_f32`) accept the
//! signature `(elem_count, ndims, dims_strides_or_null, lhs, rhs,
//! out)`. A null `dims_strides_or_null` selects the kernel's
//! contiguous fast path; the unified executor's auto-Contiguize pass
//! guarantees inputs are contiguous before kernel call, so the
//! wrappers always pass null.

use std::sync::Arc;

use fuel_core_types::Result;
use fuel_cuda_kernels as kernels;

use crate::builder_arg as barg;
use crate::byte_storage::CudaStorageBytes;
use crate::device::LaunchConfig;
use crate::error::WrapErr;
use crate::storage::SlicePtrOrNull;

/// Phase 7.5 first CUDA kernel through the unified path.
/// Element-wise add of two F32 `CudaStorageBytes`. Both inputs must
/// have the same byte length (== same element count for F32). Output
/// is freshly allocated on the same device as `lhs`; caller is
/// responsible for storing it where the unified executor expects it.
///
/// Auto-Contiguize is assumed: this wrapper passes null for the
/// dims/strides side-band, selecting the kernel's contiguous fast
/// path. Strided inputs through the unified path are an A5 follow-on
/// (Layout-on-KernelRef extension).
pub fn add_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "badd_f32")
}

/// Element-wise subtraction (lhs - rhs) of two F32 `CudaStorageBytes`.
/// Same shape as [`add_elementwise_f32`]; only the launched kernel
/// name differs.
pub fn sub_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bsub_f32")
}

/// Element-wise multiplication (lhs * rhs) of two F32 `CudaStorageBytes`.
pub fn mul_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bmul_f32")
}

/// Element-wise division (lhs / rhs) of two F32 `CudaStorageBytes`.
pub fn div_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bdiv_f32")
}

/// Element-wise maximum (max(lhs, rhs)) of two F32 `CudaStorageBytes`.
pub fn maximum_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bmaximum_f32")
}

/// Shared launch path for F32 elementwise binary ops. Validates equal
/// byte lengths, allocates a fresh device buffer, launches the
/// fuel-cuda-kernels BINARY function identified by `kernel_name`,
/// and returns the result. Synchronizes the default stream so the
/// result is observable on return (sync KernelRef per locked design
/// decision).
fn binary_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if lhs.len_bytes() != rhs.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: lhs.len_bytes={} != rhs.len_bytes={}",
            lhs.len_bytes(),
            rhs.len_bytes(),
        ))
        .bt());
    }
    if lhs.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: lhs.len_bytes={} not a multiple of f32 size",
            lhs.len_bytes(),
        ))
        .bt());
    }
    let elem_count = lhs.len_bytes() / elem;
    let device = lhs.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(lhs.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func(kernel_name, &kernels::BINARY)?;
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(lhs.buffer());
    builder.arg(rhs.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the existing legacy `Map2::f` for `BinaryOpT`, just on byte
    // buffers. Kernel-side validation is the same.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        lhs.len_bytes(),
    ))
}

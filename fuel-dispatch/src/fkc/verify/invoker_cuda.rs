//! CUDA [`KernelInvoker`] scaffold (Task 4.5). Mirrors [`super::invoker_cpu::CpuInvoker`]
//! but stages `HostTensor` bytes through a CUDA device: H2D via
//! `CudaStorageBytes::from_cpu_bytes`, D2H readback via `to_cpu_bytes` (the
//! same pair the `capture_decode_binary_chain_replays_bit_exact_cuda` live
//! test in `fuel-dispatch/src/pipelined.rs` already uses to stand up CUDA
//! `Storage` for a test). Compiled only under `--features cuda`; its
//! live-hardware test is `#[ignore]`'d (this build environment can't run
//! it as part of this task — the CPU invoker is the load-bearing path).
//!
//! Kept intentionally minimal: no output-shape/dtype validation beyond what
//! `alloc` already does, no stream-overlap tuning. A later task that
//! actually drives CUDA-kernel bit-stability verification should revisit
//! this once it has a live device to iterate against.

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_ir::{DType, Layout, Shape};

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use crate::kernel::{BindingEntry, OpParams};

/// A CUDA kernel invoker. Owns the `CudaDevice` handle used to stage every
/// probe input up and read the output back down.
pub struct CudaInvoker {
    device: CudaDevice,
    out_dtype: DType,
    out_shape: Vec<usize>,
    params: OpParams,
}

impl CudaInvoker {
    /// New invoker bound to `device`, for an op whose output is
    /// `out_dtype`/`out_shape`, with no auxiliary op params.
    pub fn new(device: CudaDevice, out_dtype: DType, out_shape: Vec<usize>) -> Self {
        Self { device, out_dtype, out_shape, params: OpParams::None }
    }

    /// Builder-style override for ops that need non-`None` `OpParams`.
    pub fn with_params(mut self, p: OpParams) -> Self {
        self.params = p;
        self
    }
}

impl KernelInvoker for CudaInvoker {
    fn invoke(&self, entry: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError> {
        // H2D: upload every probe input's bytes into fresh device storage.
        let ins: Vec<Arc<RwLock<fuel_memory::Storage>>> = inputs
            .iter()
            .map(|t| {
                let cb = CudaStorageBytes::from_cpu_bytes(&self.device, &t.bytes)
                    .map_err(|e| VerifyError::Backend(e.to_string()))?;
                Ok(Arc::new(RwLock::new(fuel_memory::Storage::new(
                    fuel_memory::BackendStorage::Cuda(cb),
                    t.dtype,
                ))))
            })
            .collect::<Result<Vec<_>, VerifyError>>()?;

        let elem_count = self.out_shape.iter().product::<usize>();
        let out_len_bytes = elem_count.saturating_mul(self.out_dtype.size_in_bytes());
        let out_cb = CudaStorageBytes::alloc(&self.device, out_len_bytes)
            .map_err(|e| VerifyError::Backend(e.to_string()))?;
        let out = Arc::new(RwLock::new(fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cuda(out_cb),
            self.out_dtype,
        )));
        let mut outs = [out.clone()];

        let layouts: Vec<Layout> = inputs
            .iter()
            .map(|t| Layout::contiguous(Shape::from_dims(&t.shape)))
            .chain(std::iter::once(Layout::contiguous(Shape::from_dims(&self.out_shape))))
            .collect();

        (entry.kernel)(&ins, &mut outs, &layouts, &self.params)
            .map_err(|e| VerifyError::Invoke(format!("{e:?}")))?;

        // D2H: read the output storage's bytes back to host.
        let guard = out
            .read()
            .map_err(|_| VerifyError::Backend("CudaInvoker: output storage RwLock poisoned".to_string()))?;
        let bytes = match &guard.inner {
            fuel_memory::BackendStorage::Cuda(c) => {
                c.to_cpu_bytes().map_err(|e| VerifyError::Backend(e.to_string()))?
            }
            #[allow(unreachable_patterns)]
            _ => return Err(VerifyError::Backend("CudaInvoker: output storage is not CUDA".to_string())),
        };

        Ok(HostTensor { dtype: self.out_dtype, shape: self.out_shape.clone(), bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DType;

    /// Live-hardware smoke test — needs a CUDA device, so it's `#[ignore]`'d
    /// (mirrors the `#[ignore = "requires a live CUDA device"]` convention
    /// used throughout `fuel-dispatch/src/pipelined.rs`). Not run as part of
    /// this task (no GPU exercised here); the CPU invoker's
    /// `cpu_invoker_runs_add_elementwise_f32_end_to_end` is the load-bearing
    /// verification for this slice.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn cuda_invoker_runs_add_elementwise_f32_end_to_end() {
        let Ok(dev) = CudaDevice::new(0) else {
            eprintln!("no CUDA device; skipping");
            return;
        };
        // The real CUDA add wrapper (`fkc::cuda_link::CUDA_BINARY_ENTRY_POINTS`'s
        // `add_f32` entry point) — NOT the CPU wrapper, which would error on
        // CUDA-resident storage (`cuda kernel wrapper called with non-CUDA
        // input`-shaped failure).
        let e = crate::kernel::BindingEntry {
            kernel: crate::baracuda_dispatch::binary::add_f32,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "baracuda",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let inv = CudaInvoker::new(dev, DType::F32, vec![3]);
        let a = HostTensor {
            dtype: DType::F32,
            shape: vec![3],
            bytes: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]).to_vec(),
        };
        let b = HostTensor {
            dtype: DType::F32,
            shape: vec![3],
            bytes: bytemuck::cast_slice(&[4.0f32, 5.0, 6.0]).to_vec(),
        };
        let out = inv.invoke(&e, &[a, b]).expect("cuda invoke");
        let got: &[f32] = bytemuck::try_cast_slice(&out.bytes)
            .expect("CudaInvoker output bytes must cast back to f32 (len/align)");
        assert_eq!(got, &[5.0, 7.0, 9.0]);
    }
}

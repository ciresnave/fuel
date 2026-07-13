//! Vulkan [`KernelInvoker`] scaffold (Task 4.5). Mirrors
//! [`super::invoker_cpu::CpuInvoker`] / [`super::invoker_cuda::CudaInvoker`]
//! but stages `HostTensor` bytes through a Vulkan device: H2D via
//! `VulkanBackend::upload_bytes_handle`, D2H readback via
//! `VulkanBackend::download_bytes`. The `_handle` upload/alloc variants
//! (not the legacy `upload_bytes`/`alloc_bytes`) are required — Vulkan's
//! binary-op wrappers (`vulkan_dispatch.rs`'s `vk_binary_f32_wrapper!`)
//! pull the `VulkanBackend` Arc back off the FIRST input's
//! `VulkanStorageBytes::backend()` to dispatch the Slang kernel, and only
//! the `_handle` constructors attach it.
//!
//! Compiled only under `--features vulkan`; its live-hardware test is
//! `#[ignore]`'d (this build environment can't run it as part of this
//! task — the CPU invoker is the load-bearing path).
//!
//! Kept intentionally minimal: no output-shape/dtype validation beyond
//! what `alloc_bytes_handle` already does, no fence/queue tuning beyond
//! what `upload_bytes_handle`/`download_bytes` already do internally.

use std::sync::{Arc, RwLock};

use fuel_ir::{DType, Layout, Shape};
use fuel_vulkan_backend::VulkanBackend;

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use crate::kernel::{BindingEntry, OpParams};

/// A Vulkan kernel invoker. Owns the `Arc<VulkanBackend>` handle used to
/// stage every probe input up and read the output back down.
pub struct VulkanInvoker {
    backend: Arc<VulkanBackend>,
    out_dtype: DType,
    out_shape: Vec<usize>,
    params: OpParams,
}

impl VulkanInvoker {
    /// New invoker bound to `backend`, for an op whose output is
    /// `out_dtype`/`out_shape`, with no auxiliary op params.
    pub fn new(backend: Arc<VulkanBackend>, out_dtype: DType, out_shape: Vec<usize>) -> Self {
        Self { backend, out_dtype, out_shape, params: OpParams::None }
    }

    /// Builder-style override for ops that need non-`None` `OpParams`.
    pub fn with_params(mut self, p: OpParams) -> Self {
        self.params = p;
        self
    }
}

impl KernelInvoker for VulkanInvoker {
    fn invoke(&self, entry: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError> {
        // H2D: upload every probe input's bytes into fresh device storage,
        // WITH the backend handle attached (`_handle` variant) so the
        // binary-op wrapper can pull it back off `input[0]` to dispatch.
        let ins: Vec<Arc<RwLock<fuel_memory::Storage>>> = inputs
            .iter()
            .map(|t| {
                let vb = self
                    .backend
                    .upload_bytes_handle(&t.bytes)
                    .map_err(|e| VerifyError::Backend(e.to_string()))?;
                Ok(Arc::new(RwLock::new(fuel_memory::Storage::new(
                    fuel_memory::BackendStorage::Vulkan(vb),
                    t.dtype,
                ))))
            })
            .collect::<Result<Vec<_>, VerifyError>>()?;

        let elem_count = self.out_shape.iter().product::<usize>();
        let out_len_bytes = elem_count.saturating_mul(self.out_dtype.size_in_bytes());
        let out_vb = self
            .backend
            .alloc_bytes_handle(out_len_bytes)
            .map_err(|e| VerifyError::Backend(e.to_string()))?;
        let out = Arc::new(RwLock::new(fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Vulkan(out_vb),
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
        let guard = out.read().map_err(|_| {
            VerifyError::Backend("VulkanInvoker: output storage RwLock poisoned".to_string())
        })?;
        let bytes = match &guard.inner {
            fuel_memory::BackendStorage::Vulkan(v) => {
                self.backend.download_bytes(v).map_err(|e| VerifyError::Backend(e.to_string()))?
            }
            #[allow(unreachable_patterns)]
            _ => return Err(VerifyError::Backend("VulkanInvoker: output storage is not Vulkan".to_string())),
        };

        Ok(HostTensor { dtype: self.out_dtype, shape: self.out_shape.clone(), bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::DType;

    /// Live-hardware smoke test — needs a Vulkan device, so it's
    /// `#[ignore]`'d. Not run as part of this task (no GPU exercised
    /// here); the CPU invoker's
    /// `cpu_invoker_runs_add_elementwise_f32_end_to_end` is the
    /// load-bearing verification for this slice.
    #[test]
    #[ignore = "requires a live Vulkan device"]
    fn vulkan_invoker_runs_add_elementwise_f32_end_to_end() {
        let Ok(backend) = VulkanBackend::new() else {
            eprintln!("no Vulkan device; skipping");
            return;
        };
        let backend = Arc::new(backend);
        // The real Vulkan add wrapper (`fkc::vulkan_link::VULKAN_BINARY_ENTRY_POINTS`'s
        // `add_f32` entry point) — NOT the CPU wrapper, which would error on
        // Vulkan-resident storage.
        let e = crate::kernel::BindingEntry {
            kernel: crate::vulkan_dispatch::binary::add_f32,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "fuel-vulkan-kernels",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let inv = VulkanInvoker::new(backend, DType::F32, vec![3]);
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
        let out = inv.invoke(&e, &[a, b]).expect("vulkan invoke");
        let got: &[f32] = bytemuck::try_cast_slice(&out.bytes)
            .expect("VulkanInvoker output bytes must cast back to f32 (len/align)");
        assert_eq!(got, &[5.0, 7.0, 9.0]);
    }
}

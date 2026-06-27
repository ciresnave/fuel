//! CUDA bridge for fuel-ug compiled kernels.
//!
//! `CudaUgIOp1` wraps a compiled `CudaFunc` produced by
//! `CudaDevice::compile` and dispatches it as an in-place unary op.
//! Implements [`fuel_backend_contract::InplaceOp1`] so it can be applied to a
//! tensor via `Tensor::inplace_op1` from fuel-core.
//!
//! Migrated out of `fuel_core::custom_op::UgIOp1` in step B2 of the
//! backend extraction; fuel-core no longer mentions
//! cuda-specific dispatch in the custom-op module.

use fuel_backend_contract::dyn_backend::DynBackendStorage;
use fuel_backend_contract::inplace_op::InplaceOp1;
use fuel_ir::{Layout, Result};

use crate::device::{CudaDevice, CudaFunc, LaunchConfig};
use crate::error::WrapErr;
use crate::storage::CudaStorage;

/// In-place unary op driven by a fuel-ug compiled CUDA kernel.
pub struct CudaUgIOp1 {
    name: &'static str,
    func: CudaFunc,
}

impl CudaUgIOp1 {
    /// Compiles `kernel` against `device` and returns a ready-to-dispatch op.
    ///
    /// fuel-ug is only available on platforms where the upstream crate
    /// builds (everything except wasm32 / iOS).
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn new(
        name: &'static str,
        kernel: fuel_ug::lang::ssa::Kernel,
        device: &CudaDevice,
    ) -> Result<Self> {
        let func = device.compile(name, kernel)?;
        Ok(Self { name, func })
    }
}

impl InplaceOp1 for CudaUgIOp1 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn fwd(&self, storage: &mut dyn DynBackendStorage, layout: &Layout) -> Result<()> {
        let sto = storage
            .as_any_mut()
            .downcast_mut::<CudaStorage>()
            .ok_or_else(|| {
                fuel_ir::Error::Msg(
                    "CudaUgIOp1: storage is not a CudaStorage".to_string(),
                )
                .bt()
            })?;

        let elem_count = layout.shape().elem_count();
        // TODO: support more dtypes.
        let sto = sto.as_cuda_slice::<f32>()?;
        let sto = match layout.contiguous_offsets() {
            None => fuel_ir::bail!("input has to be contiguous"),
            Some((o1, o2)) => sto.slice(o1..o2),
        };
        let (g, b) = if elem_count.is_multiple_of(32) {
            (elem_count / 32, 32)
        } else {
            (elem_count, 1)
        };
        let cfg = LaunchConfig {
            grid_dim: (g as u32, 1, 1),
            block_dim: (b as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = self.func.builder();
        builder.arg(&sto);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

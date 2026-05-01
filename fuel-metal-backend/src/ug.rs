//! Metal bridge for fuel-ug compiled kernels.
//!
//! `MetalUgIOp1` wraps a compiled Metal `ComputePipeline` produced by
//! `MetalDevice::compile` and dispatches it as an in-place unary op.
//! Implements [`fuel_core_types::InplaceOp1`] so it can be applied to a
//! tensor via `Tensor::inplace_op1` from fuel-core.
//!
//! Migrated out of `fuel_core::custom_op::UgIOp1` in step B2 of the
//! backend extraction; fuel-core no longer mentions
//! metal-specific dispatch in the custom-op module.

use fuel_core_types::dyn_backend::DynBackendStorage;
use fuel_core_types::inplace_op::InplaceOp1;
use fuel_core_types::{DType, Layout, Result};

use fuel_metal_kernels::metal::ComputePipeline;

use crate::device::MetalDevice;
use crate::storage::MetalStorage;

/// In-place unary op driven by a fuel-ug compiled Metal kernel.
pub struct MetalUgIOp1 {
    name: &'static str,
    func: ComputePipeline,
}

impl MetalUgIOp1 {
    /// Compiles `kernel` against `device` and returns a ready-to-dispatch op.
    ///
    /// fuel-ug is only available on platforms where the upstream crate
    /// builds (everything except wasm32 / iOS).
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios")))]
    pub fn new(
        name: &'static str,
        kernel: fuel_ug::lang::ssa::Kernel,
        device: &MetalDevice,
    ) -> Result<Self> {
        let func = device.compile(name, kernel)?;
        Ok(Self { name, func })
    }
}

impl InplaceOp1 for MetalUgIOp1 {
    fn name(&self) -> &'static str {
        self.name
    }

    fn fwd(&self, storage: &mut dyn DynBackendStorage, layout: &Layout) -> Result<()> {
        let sto = storage
            .as_any_mut()
            .downcast_mut::<MetalStorage>()
            .ok_or_else(|| {
                fuel_core_types::Error::Msg(
                    "MetalUgIOp1: storage is not a MetalStorage".to_string(),
                )
                .bt()
            })?;

        let elem_count = layout.shape().elem_count();
        if sto.dtype() != DType::F32 {
            // TODO: support more dtypes.
            fuel_core_types::bail!("input is not a f32 tensor")
        }
        let device = sto.device();
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&self.func);
        let (g, b) = if elem_count.is_multiple_of(32) {
            (elem_count / 32, 32)
        } else {
            (elem_count, 1)
        };
        let grid_dims = objc2_metal::MTLSize {
            width: g,
            height: 1,
            depth: 1,
        };
        let group_dims = fuel_metal_kernels::utils::get_block_dims(b, 1, 1);
        fuel_metal_kernels::utils::set_param(&encoder, 0, (sto.buffer(), 0usize));

        encoder.use_resource(sto.buffer(), objc2_metal::MTLResourceUsage::Write);
        encoder.dispatch_threads(grid_dims, group_dims);

        Ok(())
    }
}

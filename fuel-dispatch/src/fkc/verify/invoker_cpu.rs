//! The real CPU [`KernelInvoker`] (Task 4.5) — runs an actual registered
//! CPU kernel (`BindingEntry::kernel`) against host-resident probe inputs
//! and reads the result back to host bytes. Hardware-free (CPU always
//! runs): this is the first invoker in the Task 4.4 `KernelInvoker` trait
//! that drives a REAL kernel rather than an in-process fake, so it is the
//! producer that will feed empirically-verified ledger entries once wired
//! up (later task).
//!
//! Mirrors the shape of the CPU dispatch wrappers themselves
//! (`fuel-dispatch/src/dispatch.rs`'s `cpu_binary_wrapper!`-expanded fns):
//! wrap each [`HostTensor`]'s bytes in a CPU [`fuel_memory::Storage`],
//! allocate a zeroed output `Storage`, build contiguous [`fuel_ir::Layout`]s
//! for every operand, call the kernel fn-pointer directly, then read the
//! output bytes back out.

use std::sync::{Arc, RwLock};

use fuel_ir::{DType, Layout, Shape};

use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker, VerifyError};
use crate::kernel::{BindingEntry, OpParams};

/// A real CPU kernel invoker. Fixed output `dtype`/`shape` (the verifier
/// knows these from the contract's declared return shape/dtype — Task 4.5
/// doesn't infer them) plus whatever `OpParams` the op under test needs
/// (`OpParams::None` for elementwise ops, the default).
pub struct CpuInvoker {
    out_dtype: DType,
    out_shape: Vec<usize>,
    params: OpParams,
}

impl CpuInvoker {
    /// New invoker for an op whose output is `out_dtype`/`out_shape`,
    /// with no auxiliary op params (`OpParams::None` — elementwise unary/
    /// binary, shape-only ops, etc.).
    pub fn new(out_dtype: DType, out_shape: Vec<usize>) -> Self {
        Self { out_dtype, out_shape, params: OpParams::None }
    }

    /// Builder-style override for ops that need non-`None` `OpParams`
    /// (reductions, matmul, ...).
    pub fn with_params(mut self, p: OpParams) -> Self {
        self.params = p;
        self
    }
}

impl KernelInvoker for CpuInvoker {
    fn invoke(&self, entry: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError> {
        // Wrap each host-resident probe input in a CPU `Storage`. `from_slice`
        // is called with `T = u8` here (the byte buffer itself), which never
        // panics regardless of the logical `dtype` tag — `u8`'s size (1) and
        // alignment (1) always evenly divide any byte slice, so there is no
        // reinterpret-cast risk on this path (unlike the readback direction,
        // bytes -> a wider type, which is where `try_cast_slice` matters).
        let ins: Vec<Arc<RwLock<fuel_memory::Storage>>> = inputs
            .iter()
            .map(|t| {
                Arc::new(RwLock::new(fuel_memory::Storage::new(
                    fuel_memory::BackendStorage::Cpu(fuel_cpu_backend::CpuStorageBytes::from_slice(
                        &t.bytes,
                    )),
                    t.dtype,
                )))
            })
            .collect();

        let elem_count = self.out_shape.iter().product::<usize>();
        let out_storage = fuel_memory::alloc_cpu_zeroed(self.out_dtype, elem_count)
            .map_err(|e| VerifyError::Backend(e.to_string()))?;
        let out = Arc::new(RwLock::new(out_storage));
        let mut outs = [out.clone()];

        let layouts: Vec<Layout> = inputs
            .iter()
            .map(|t| Layout::contiguous(Shape::from_dims(&t.shape)))
            .chain(std::iter::once(Layout::contiguous(Shape::from_dims(&self.out_shape))))
            .collect();

        (entry.kernel)(&ins, &mut outs, &layouts, &self.params)
            .map_err(|e| VerifyError::Invoke(format!("{e:?}")))?;

        let guard = out
            .read()
            .map_err(|_| VerifyError::Backend("CpuInvoker: output storage RwLock poisoned".to_string()))?;
        // NOTE: deliberately NOT `fuel_memory::dispatch_storage!` here — that
        // macro expands the SAME body (`s.bytes().to_vec()`) across every
        // backend variant enabled for the crate, and only `CpuStorageBytes`
        // has a `.bytes()` method (`CudaStorageBytes`/`VulkanStorageBytes`
        // don't expose raw host-visible bytes at all — device-resident,
        // read back via `to_cpu_bytes`/`download_bytes` instead). Using
        // `dispatch_storage!` here would build fine alone but FAIL to
        // compile the instant `--features cuda` or `--features vulkan` is
        // added (found while verifying this file compiles under `vulkan`).
        // `cpu_input` is the existing, narrower accessor
        // (`fuel-dispatch/src/dispatch.rs`) built for exactly this: extract
        // `&CpuStorageBytes` from a `&Storage` known to be CPU-backed.
        let bytes = crate::dispatch::cpu_input(&guard)
            .map_err(|e| VerifyError::Backend(e.to_string()))?
            .bytes()
            .to_vec();

        Ok(HostTensor { dtype: self.out_dtype, shape: self.out_shape.clone(), bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker};
    use fuel_ir::DType;

    #[test]
    fn cpu_invoker_runs_add_elementwise_f32_end_to_end() {
        // Use the real CPU add wrapper as the KernelRef (mirror the wrapper
        // used by register.rs's `cpu_link_registry_binds_elementwise_binary_to_live_kernels`
        // test, which names it `crate::dispatch::add_elementwise_f32_cpu_wrapper`).
        let e = crate::kernel::BindingEntry {
            kernel: crate::dispatch::add_elementwise_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "portable-cpu",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let inv = CpuInvoker::new(DType::F32, vec![3]);
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
        let out = inv.invoke(&e, &[a, b]).expect("cpu invoke");
        // Readback reinterpret: bytes -> f32. This is the risky direction
        // (unlike f32 -> bytes above), so `try_cast_slice` (never-panic),
        // not `cast_slice`, per the house idiom
        // (fuel-cpu-backend/src/byte_storage.rs's `as_slice`).
        let got: &[f32] = bytemuck::try_cast_slice(&out.bytes)
            .expect("CpuInvoker output bytes must cast back to f32 (len/align)");
        assert_eq!(got, &[5.0, 7.0, 9.0]);
    }
}

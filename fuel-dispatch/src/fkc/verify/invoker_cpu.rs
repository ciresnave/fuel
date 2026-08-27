// SPDX-License-Identifier: MIT OR Apache-2.0
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
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
pub struct CpuInvoker {
    out_dtype: DType,
    out_shape: Vec<usize>,
    params: OpParams,
    /// Pre-fill for the output buffer; `None` = zeroed (the default).
    out_seed: Option<Vec<u8>>,
}

impl CpuInvoker {
    /// New invoker for an op whose output is `out_dtype`/`out_shape`,
    /// with no auxiliary op params (`OpParams::None` — elementwise unary/
    /// binary, shape-only ops, etc.).
    #[allow(
        dead_code,
        reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
    )]
    pub fn new(out_dtype: DType, out_shape: Vec<usize>) -> Self {
        Self {
            out_dtype,
            out_shape,
            params: OpParams::None,
            out_seed: None,
        }
    }

    /// Pre-fill the output buffer with `bytes` instead of zeroing it.
    ///
    /// **Required for in-place ops, and the reason is evidential rather than
    /// functional.** The executor hands an in-place target in as
    /// `outputs[0]`, so such a kernel has no inputs at all — against a zeroed
    /// buffer, `relu_inplace` reads 0 and writes 0. Sixteen repeats of that
    /// are byte-identical, so `bit_stable_on_same_hardware` comes back PASS
    /// having exercised exactly one input value and no branch. The record is
    /// true and uninformative, and `gate_precision` cannot tell it from a
    /// real verification (GAP-222) — which is why the fix belongs here, in
    /// the invoker, and not in each op's own recipe.
    ///
    /// `bytes` must be `out_dtype`-encoded and length-consistent with
    /// `out_shape`; a mismatch is rejected at invoke time rather than
    /// silently truncating, because a short seed would leave a zeroed tail
    /// and quietly reintroduce the same weak evidence over part of the
    /// buffer.
    #[allow(
        dead_code,
        reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
    )]
    pub fn with_seeded_output(mut self, bytes: Vec<u8>) -> Self {
        self.out_seed = Some(bytes);
        self
    }

    /// Builder-style override for ops that need non-`None` `OpParams`
    /// (reductions, matmul, ...).
    #[allow(
        dead_code,
        reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
    )]
    pub fn with_params(mut self, p: OpParams) -> Self {
        self.params = p;
        self
    }
}

impl KernelInvoker for CpuInvoker {
    fn invoke(
        &self,
        entry: &BindingEntry,
        inputs: &[HostTensor],
    ) -> Result<HostTensor, VerifyError> {
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
                    fuel_memory::BackendStorage::Cpu(
                        fuel_cpu_backend::CpuStorageBytes::from_slice(&t.bytes),
                    ),
                    t.dtype,
                )))
            })
            .collect();

        let elem_count = self.out_shape.iter().product::<usize>();
        let out_storage = match &self.out_seed {
            None => fuel_memory::alloc_cpu_zeroed(self.out_dtype, elem_count)
                .map_err(|e| VerifyError::Backend(e.to_string()))?,
            Some(bytes) => {
                let want = elem_count * self.out_dtype.size_in_bytes();
                if bytes.len() != want {
                    return Err(VerifyError::Backend(format!(
                        "CpuInvoker: seeded output is {} bytes, expected {want} for                          {:?} x {:?} — refusing to run rather than seed a partial                          buffer and leave a zeroed tail",
                        bytes.len(),
                        self.out_dtype,
                        self.out_shape,
                    )));
                }
                fuel_memory::Storage::new(
                    fuel_memory::BackendStorage::Cpu(
                        fuel_cpu_backend::CpuStorageBytes::from_slice(bytes),
                    ),
                    self.out_dtype,
                )
            }
        };
        let out = Arc::new(RwLock::new(out_storage));
        let mut outs = [out.clone()];

        let layouts: Vec<Layout> = inputs
            .iter()
            .map(|t| Layout::contiguous(Shape::from_dims(&t.shape)))
            .chain(std::iter::once(Layout::contiguous(Shape::from_dims(
                &self.out_shape,
            ))))
            .collect();

        (entry.kernel)(&ins, &mut outs, &layouts, &self.params)
            .map_err(|e| VerifyError::Invoke(format!("{e:?}")))?;

        let guard = out.read().map_err(|_| {
            VerifyError::Backend("CpuInvoker: output storage RwLock poisoned".to_string())
        })?;
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

        Ok(HostTensor {
            dtype: self.out_dtype,
            shape: self.out_shape.clone(),
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::verify::bit_stability::{HostTensor, KernelInvoker};
    use fuel_ir::DType;

    /// The seed must actually REACH the kernel, and a passing bit-stability
    /// run cannot show that: an in-place kernel on a zeroed buffer is
    /// perfectly bit-stable, so `with_seeded_output` could be inert and every
    /// downstream test would still be green. That is the exact shape of
    /// GAP-222 — a result that is true and uninformative — so the seeding
    /// mechanism gets a control that fails if it stops working.
    ///
    /// `relu_inplace` is chosen because it DISCRIMINATES: its output differs
    /// from its input only where the input is negative. A seed of all
    /// positives would produce the same bytes whether or not the kernel read
    /// them, so the fixture deliberately contains negatives.
    #[test]
    fn a_seeded_output_buffer_actually_reaches_an_inplace_kernel() {
        let entry = crate::kernel::BindingEntry {
            kernel: crate::dispatch::relu_inplace_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "portable-cpu",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };

        let seed: [f32; 4] = [-2.0, 1.5, -0.25, 3.0];
        let seed_bytes = bytemuck::cast_slice(&seed).to_vec();

        // In-place ops take NO inputs; the target is `outputs[0]`.
        let zeroed = CpuInvoker::new(DType::F32, vec![4])
            .invoke(&entry, &[])
            .expect("zeroed invoke");
        let seeded = CpuInvoker::new(DType::F32, vec![4])
            .with_seeded_output(seed_bytes.clone())
            .invoke(&entry, &[])
            .expect("seeded invoke");

        // Control 1: the two runs must DIFFER. If they match, the seed never
        // reached the kernel and every in-place ledger record earned through
        // this path is a measurement of zeros.
        assert_ne!(
            zeroed.bytes, seeded.bytes,
            "seeded and zeroed runs produced identical bytes — `with_seeded_output`              is inert, and every in-place verification through it is vacuous"
        );

        // Control 2: the seeded result is exactly relu(seed), which no amount
        // of accidental non-zero content would reproduce. This is what makes
        // the assertion above a statement about THIS seed rather than about
        // the buffer merely being dirty.
        let got: Vec<f32> = bytemuck::cast_slice(&seeded.bytes).to_vec();
        assert_eq!(
            got,
            vec![0.0f32, 1.5, 0.0, 3.0],
            "expected relu of the seed"
        );

        // ...and the zeroed run is relu(0) = 0, i.e. the degenerate evidence
        // GAP-222 describes: a perfectly stable result that says nothing.
        let z: Vec<f32> = bytemuck::cast_slice(&zeroed.bytes).to_vec();
        assert_eq!(z, vec![0.0f32; 4]);
    }

    /// A short or long seed is refused rather than silently padded — a
    /// truncated seed would leave a zeroed tail and reintroduce the weak
    /// evidence over part of the buffer, which is harder to spot than a
    /// wholly zeroed one.
    #[test]
    fn a_wrong_length_seed_is_refused_instead_of_partially_applied() {
        let entry = crate::kernel::BindingEntry {
            kernel: crate::dispatch::relu_inplace_f32_cpu_wrapper,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "portable-cpu",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        };
        let short: Vec<u8> = vec![0u8; 3 * 4]; // 3 f32s for a 4-element output
        let err = CpuInvoker::new(DType::F32, vec![4])
            .with_seeded_output(short)
            .invoke(&entry, &[]);
        assert!(
            err.is_err(),
            "a 3-element seed for a 4-element output was accepted"
        );
    }

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

//! Empirical bit-stability verification (`V-FKC-9`, Task 4.4).
//!
//! [`verify_bit_stability`] generalizes the worktree `gemm_dense.rs`
//! `determinism_audit` precedent (`.claude/worktrees/capturedrun-executor/
//! fuel-cuda-backend/src/baracuda/gemm_dense.rs` ~405-613): call a kernel
//! `iters` times per probe input and require byte-identical output every
//! time. This module is hardware-free — it only defines the [`KernelInvoker`]
//! trait boundary and the pure verification logic; the real CPU/CUDA
//! invokers that call actual kernels are Task 4.5. Unit tests here use a
//! fake in-process invoker.
//!
//! Never-panic: `verify_bit_stability` only ever returns `Ok(VerifyOutcome)`
//! or propagates a `VerifyError` from the invoker — it never panics on
//! malformed inputs (empty probes, zero iters) and reports the divergence
//! (or lack of one) as data.

use crate::kernel::BindingEntry;
use fuel_ir::DType;

/// A host-resident tensor snapshot: dtype + shape (informational/for the
/// invoker to interpret) + raw little-endian bytes. This is the verification
/// harness's hardware-free wire format — no `Storage`/device handle, so
/// probes and outputs can be constructed and compared in plain `cargo test`.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// The probe inputs for one verification call — one [`HostTensor`] per
/// kernel input operand.
pub type ProbeInputs = Vec<HostTensor>;

/// Errors from invoking a kernel through a [`KernelInvoker`]. Distinct from
/// [`VerifyOutcome::Fail`]: this is an *infrastructure* failure (the kernel
/// couldn't be run at all), not a verification-criterion miss.
#[derive(Debug)]
pub enum VerifyError {
    /// The invoker's underlying kernel call itself returned an error.
    Invoke(String),
    /// No reference implementation was available to compare against.
    NoReference,
    /// A backend-specific failure (device unavailable, launch failure, ...).
    Backend(String),
}

/// The result of running a verification check.
#[derive(Debug, PartialEq)]
pub enum VerifyOutcome {
    /// The claim held for every probe.
    Pass,
    /// The claim failed; `detail` names which probe/call and why.
    Fail { detail: String },
    /// No reference was available to verify against (distinct from an
    /// infrastructure [`VerifyError::NoReference`] — this is a clean,
    /// non-error "nothing to check" outcome some verifiers may choose to
    /// return instead of erroring).
    NoReference,
}

/// Runs a single registered kernel binding against host-resident inputs and
/// returns a host-resident output. Implementations are hardware-free at this
/// layer (Task 4.4); the real CPU/CUDA invokers that dispatch to actual
/// device kernels are Task 4.5. Unit tests in this module use fake in-process
/// invokers (`ConstInvoker`, `FlakyInvoker`) to exercise the verification
/// logic without any hardware.
pub trait KernelInvoker {
    fn invoke(&self, entry: &BindingEntry, inputs: &[HostTensor]) -> Result<HostTensor, VerifyError>;
}

/// Empirical bit-stability check (`bit_stable_on_same_hardware` claim, FKC
/// precision block). Generalizes the worktree `gemm_dense.rs`
/// `determinism_audit`: for each probe, call the kernel `iters` times and
/// require every output's `bytes` to be identical to the first. The FIRST
/// divergence (probe index + call index) is reported in `Fail { detail }` so
/// a caller can see exactly where determinism broke, without collecting every
/// subsequent mismatch.
///
/// `iters < 2` trivially passes (nothing to compare) — never panics on a
/// degenerate call count. An empty `probes` list also trivially passes —
/// there is nothing to falsify, which mirrors the "no verifiable claim"
/// posture the ledger gate already uses upstream (see `ledger::gate_precision`).
pub fn verify_bit_stability(
    inv: &dyn KernelInvoker,
    entry: &BindingEntry,
    probes: &[ProbeInputs],
    iters: usize,
) -> Result<VerifyOutcome, VerifyError> {
    for (probe_idx, probe) in probes.iter().enumerate() {
        let first = inv.invoke(entry, probe)?;
        for call_idx in 1..iters {
            let next = inv.invoke(entry, probe)?;
            if next.bytes != first.bytes {
                return Ok(VerifyOutcome::Fail {
                    detail: format!(
                        "probe {probe_idx} diverged at call {call_idx}: bytes differ from call 0"
                    ),
                });
            }
        }
    }
    Ok(VerifyOutcome::Pass)
}

/// Deterministic pseudo-random `f32` fill via xorshift64* — ported verbatim
/// (same constants/shifts) from the worktree `gemm_dense.rs` precedent so
/// probe generation here and there produce identical sequences for the same
/// seed. Values land in `[-0.5, 0.5)`. Never panics: `len == 0` returns an
/// empty vec; `seed == 0` degenerates to an all-zero xorshift stream (a
/// known xorshift64* fixed point), which is a valid — if uninteresting —
/// deterministic fill, not a crash.
pub fn fill_deterministic(len: usize, mut seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        let r = seed.wrapping_mul(0x2545F4914F6CDD1D);
        v.push(((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
    }
    v
}

#[cfg(test)]
mod fake_tests {
    use super::*;
    use crate::fkc::verify::ulp::{verify_precision_bound, Bound};
    use fuel_ir::DType;
    use std::sync::atomic::{AtomicU8, Ordering};

    /// Always returns the same fixed bytes — the deterministic-kernel fake.
    struct ConstInvoker(Vec<u8>);
    impl KernelInvoker for ConstInvoker {
        fn invoke(&self, _e: &BindingEntry, _i: &[HostTensor]) -> Result<HostTensor, VerifyError> {
            Ok(HostTensor { dtype: DType::F32, shape: vec![1], bytes: self.0.clone() })
        }
    }

    /// Returns different bytes on every call — the non-deterministic-kernel
    /// fake used to prove `verify_bit_stability` actually detects divergence
    /// rather than trivially passing everything.
    struct FlakyInvoker(AtomicU8);
    impl KernelInvoker for FlakyInvoker {
        fn invoke(&self, _e: &BindingEntry, _i: &[HostTensor]) -> Result<HostTensor, VerifyError> {
            let n = self.0.fetch_add(1, Ordering::Relaxed);
            Ok(HostTensor { dtype: DType::F32, shape: vec![1], bytes: vec![n] })
        }
    }

    fn probe() -> ProbeInputs {
        vec![HostTensor { dtype: DType::F32, shape: vec![1], bytes: vec![0, 0, 0, 0] }]
    }

    /// Constructs a minimal `BindingEntry` literal for verifier tests. Must
    /// track the real struct's field set (kernel.rs) exactly, including
    /// `cost_expr: None` (Task 2.1's field) — this is a compile-time
    /// tripwire: if `BindingEntry` grows a field, this helper fails to build
    /// until updated, rather than silently constructing a stale entry.
    fn dummy_entry() -> BindingEntry {
        fn k(
            _inputs: &[std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>],
            _outputs: &mut [std::sync::Arc<std::sync::RwLock<fuel_memory::Storage>>],
            _layouts: &[fuel_ir::Layout],
            _params: &crate::kernel::OpParams,
        ) -> fuel_ir::Result<()> {
            Ok(())
        }
        BindingEntry {
            kernel: k,
            caps: crate::kernel::KernelCaps::empty(),
            precision: crate::fused::PrecisionGuarantee::UNAUDITED,
            cost: crate::kernel::unknown_cost,
            kernel_source: "",
            is_generic: false,
            kernel_revision_hash: 0,
            cost_expr: None,
        }
    }

    #[test]
    fn verify_bit_stability_passes_for_a_deterministic_invoker_and_fails_for_a_flaky_one() {
        let e = dummy_entry();
        assert!(matches!(
            verify_bit_stability(&ConstInvoker(vec![1, 2, 3, 4]), &e, &[probe()], 16).unwrap(),
            VerifyOutcome::Pass
        ));
        match verify_bit_stability(&FlakyInvoker(AtomicU8::new(0)), &e, &[probe()], 16).unwrap() {
            VerifyOutcome::Fail { detail } => assert!(detail.contains("diverged"), "detail: {detail}"),
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn verify_precision_bound_flags_a_candidate_exceeding_max_absolute() {
        let e = dummy_entry();
        let reference = ConstInvoker(1.0f32.to_le_bytes().to_vec());
        let candidate = ConstInvoker(1.5f32.to_le_bytes().to_vec());
        assert!(matches!(
            verify_precision_bound(&candidate, &reference, &e, &[probe()], Bound::MaxAbsolute(0.25)).unwrap(),
            VerifyOutcome::Fail { .. }
        ));
        assert!(matches!(
            verify_precision_bound(&candidate, &reference, &e, &[probe()], Bound::MaxAbsolute(1.0)).unwrap(),
            VerifyOutcome::Pass
        ));
    }
}

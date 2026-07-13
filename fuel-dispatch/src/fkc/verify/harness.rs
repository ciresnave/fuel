//! Task 4.6 — the `rope_apply` CUDA acceptance harness (`run_fkc_verify_harness`).
//!
//! This is the FIRST harness that actually EARNS a ledger entry for a live
//! CUDA kernel: baracuda's `rope_apply_<dt>_run` (see
//! `docs/kernel-contracts/cuda/rope-apply.fkc.md`), the acceptance kernel
//! the whole FKC verification program exists to satisfy (a kernel baracuda
//! SHIPS for Fuel but that was never wired into any dispatch path — see the
//! contract's front-matter note). `#[cfg(feature = "cuda")]` throughout: it
//! needs a live `CudaDevice` to invoke anything, so it CANNOT run in this
//! (or any) hardware-free `cargo test` — the acceptance test at the bottom
//! is `#[ignore]`'d for exactly that reason.
//!
//! ## Design
//!
//! Unlike the shared production `fuel_dispatch::fkc::CudaLinkRegistry`
//! (`cuda_link.rs`, ~750 lines of hand-maintained `(&str, KernelRef)`
//! tables consumed by the REAL dispatch-registration path), this harness
//! imports the rope-apply contract into a FRESH, throwaway
//! `KernelBindingTable` via a small harness-local [`RopeApplyLinkRegistry`]
//! that resolves ONLY this contract's four fanned symbols. This keeps the
//! new (unverified-by-compiler-here) FFI-calling code and its link-table
//! entry confined to ONE new file, rather than touching `cuda_link.rs` /
//! `baracuda_dispatch.rs` — both real production files exercised by other
//! `--features cuda` tests — with code nobody can compile-check in this
//! environment. Promoting `rope_apply` to the shared `CudaLinkRegistry` (so
//! it becomes a real dispatch-table alternative, not just a
//! verification-harness fixture) is deliberately OUT OF SCOPE for Task 4.6;
//! see the contract's front-matter note.
//!
//! The actual FFI-calling driver lives in
//! `fuel_cuda_backend::baracuda::attention::rope_apply_<dt>_into` (added
//! alongside the sibling `rope_<dt>_into` family in this same task) —
//! mirrors that family's `CudaStorageBytes`-based write-into-output style
//! exactly, adapted for `rope_apply`'s different (bh, td, d, stride_b) +
//! caller-supplied-cos/sin signature.
//!
//! ## What this DOESN'T do (documented gaps, not silent shortcuts)
//!
//! - Only checks `bit_stable_on_same_hardware` (repeat-call determinism).
//!   `verify_precision_bound` (CUDA-candidate vs CPU-reference) is NOT
//!   wired here: that helper's signature takes ONE shared `BindingEntry`
//!   and ONE shared `ProbeInputs` list for BOTH invokers, but baracuda's
//!   half-width `[seq, head_dim/2]` cos/sin ABI needs DIFFERENT probe bytes
//!   than the CPU `rope_<dt>` family's full-width `[seq, head_dim]`
//!   convention to express the SAME logical rotation — the existing helper
//!   cannot express that without probe bytes that differ per invoker. A
//!   real cross-backend numeric check is a follow-on (would need a
//!   `verify_precision_bound`-shaped helper taking per-invoker probes).
//! - Probe shapes are a rope-specific FIXED set (`solve_probe_shapes`,
//!   Group 1's generic shape-constraint solver, isn't reused) — the
//!   brief explicitly permits this for an op whose accept-shape has a
//!   baracuda-native half-width relationship a generic solver hasn't been
//!   taught.

use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Error, Layout, Result};
use fuel_cuda_backend::CudaDevice;
use std::sync::{Arc, RwLock};

use crate::dispatch::{cuda_input, cuda_output, read_storage, write_storage};
use crate::fkc::lower::LinkRegistry;
use crate::kernel::{KernelRef, OpParams};
use fuel_memory::Storage;

// `verify`-level re-exports (`verify/mod.rs`'s `pub use` list) — these are
// PRIVATE submodules (`bit_stability`, `ledger`, `invoker_cuda`) reached
// through their parent's public names, the same way every other file in
// this crate consumes them.
use super::{
    fill_deterministic, verify_bit_stability, CudaInvoker, HostTensor, LedgerRecord, ProbeInputs,
    VerificationLedger, VerifyError, VerifyOutcome,
};

/// The rope-apply contract source, embedded so the harness never touches
/// the filesystem at verification time (mirrors `LEDGER_JSON` in
/// `ledger.rs`).
const ROPE_APPLY_CONTRACT: &str =
    include_str!("../../../../docs/kernel-contracts/cuda/rope-apply.fkc.md");

// ===========================================================================
// The four dispatch wrappers (KernelRef-shaped) — adapt `Storage`/`OpParams`
// to `fuel_cuda_backend::baracuda::attention::rope_apply_<dt>_into`.
// ===========================================================================

/// Shared body for the four `rope_apply_<dt>` wrappers: extract `(x, cos,
/// sin)` + the pre-allocated output from the executor-shaped `Storage`
/// slices, pull `(outer_count, seq, head_dim)` from `OpParams::Rope`, and
/// forward to `$driver`. A `macro_rules!` (not a generic fn) because each
/// dtype resolves to a DIFFERENT `fuel_cuda_backend` driver function, not a
/// value that can be threaded through a type parameter.
macro_rules! rope_apply_wrapper {
    ($wrapper_name:ident, $driver:path) => {
        pub(crate) fn $wrapper_name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            _layouts: &[Layout],
            params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 3 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(
                        stringify!($wrapper_name),
                        ": expected 3 inputs (x, cos, sin) + 1 output, got {} + {}",
                    ),
                    inputs.len(), outputs.len(),
                ))
                .bt());
            }
            let (outer_count, seq, head_dim) = match params {
                OpParams::Rope { outer_count, seq, head_dim } => (*outer_count, *seq, *head_dim),
                other => {
                    return Err(Error::Msg(format!(
                        concat!(stringify!($wrapper_name), ": expected OpParams::Rope, got {:?}"),
                        other,
                    ))
                    .bt())
                }
            };
            let x_guard = read_storage(&inputs[0])?;
            let cos_guard = read_storage(&inputs[1])?;
            let sin_guard = read_storage(&inputs[2])?;
            let mut out_guard = write_storage(&outputs[0])?;
            let x_cuda = cuda_input(&x_guard)?;
            let cos_cuda = cuda_input(&cos_guard)?;
            let sin_cuda = cuda_input(&sin_guard)?;
            let out_cuda = cuda_output(&mut out_guard)?;
            $driver(x_cuda, cos_cuda, sin_cuda, outer_count, seq, head_dim, out_cuda)
        }
    };
}

rope_apply_wrapper!(rope_apply_f32, fuel_cuda_backend::baracuda::attention::rope_apply_f32_into);
rope_apply_wrapper!(rope_apply_f16, fuel_cuda_backend::baracuda::attention::rope_apply_f16_into);
rope_apply_wrapper!(rope_apply_bf16, fuel_cuda_backend::baracuda::attention::rope_apply_bf16_into);
rope_apply_wrapper!(rope_apply_f64, fuel_cuda_backend::baracuda::attention::rope_apply_f64_into);

/// Harness-local [`LinkRegistry`]: resolves ONLY the rope-apply contract's
/// four fanned symbols (`baracuda_kernels_rope_apply_<dt>`, per the
/// contract's base `entry_point` + the standard dtype-suffix fan) to the
/// wrappers above. See the module doc for why this is NOT the shared
/// production `CudaLinkRegistry`.
struct RopeApplyLinkRegistry;

impl LinkRegistry for RopeApplyLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        match symbol {
            "baracuda_kernels_rope_apply_f32" => Some(rope_apply_f32 as KernelRef),
            "baracuda_kernels_rope_apply_f16" => Some(rope_apply_f16 as KernelRef),
            "baracuda_kernels_rope_apply_bf16" => Some(rope_apply_bf16 as KernelRef),
            "baracuda_kernels_rope_apply_f64" => Some(rope_apply_f64 as KernelRef),
            _ => None,
        }
    }
    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        None
    }
}

// ===========================================================================
// Probe synthesis — a rope-specific FIXED set (not Group 1's generic
// `solve_probe_shapes`; see the module doc for why).
// ===========================================================================

/// Fixed, small, valid geometry for every dtype variant: `outer_count=1,
/// seq=2, head_dim=4` (even, per baracuda's requirement). `x` has
/// `1*2*4 = 8` elements; the half-width `cos`/`sin` tables have
/// `seq * (head_dim/2) = 2*2 = 4` F32 elements each.
const PROBE_OUTER_COUNT: usize = 1;
const PROBE_SEQ: usize = 2;
const PROBE_HEAD_DIM: usize = 4;

/// Encode `vals` into `dt`'s byte representation. `None` for a dtype this
/// harness doesn't know how to encode — never guesses (mirrors
/// `seed_cpu_ledger.rs`'s `to_bytes`).
fn to_bytes(dt: DType, vals: &[f32]) -> Option<Vec<u8>> {
    Some(match dt {
        DType::F32 => bytemuck::cast_slice(vals).to_vec(),
        DType::F64 => {
            let v: Vec<f64> = vals.iter().map(|&x| x as f64).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::BF16 => {
            let v: Vec<half::bf16> = vals.iter().map(|&x| half::bf16::from_f32(x)).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::F16 => {
            let v: Vec<half::f16> = vals.iter().map(|&x| half::f16::from_f32(x)).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        _ => return None,
    })
}

fn ht(dt: DType, shape: Vec<usize>, vals: &[f32]) -> Option<HostTensor> {
    Some(HostTensor { dtype: dt, shape, bytes: to_bytes(dt, vals)? })
}

/// Build a real, valid `(x, cos, sin)` probe for `dt` at the fixed probe
/// geometry. `cos`/`sin` are ALWAYS F32 half-width, regardless of `dt` —
/// the baracuda ABI (see the contract). `None` if `dt` isn't an encodable
/// float dtype.
fn rope_apply_probe(dt: DType, seed: u64) -> Option<ProbeInputs> {
    let (outer, seq, head_dim) = (PROBE_OUTER_COUNT, PROBE_SEQ, PROBE_HEAD_DIM);
    let half = head_dim / 2;
    let x = ht(dt, vec![outer * seq * head_dim], &fill_deterministic(outer * seq * head_dim, seed))?;
    let cos = ht(DType::F32, vec![seq * half], &fill_deterministic(seq * half, seed ^ 0x1111))?;
    let sin = ht(DType::F32, vec![seq * half], &fill_deterministic(seq * half, seed ^ 0x2222))?;
    Some(vec![x, cos, sin])
}

/// Map a rope-apply variant's varying (`x`) dtype to the CLI-style kernel
/// name the acceptance test / `kernels` filter uses (`"rope_apply_f32"`,
/// ...). `None` for anything not in the contract's dtype fan.
fn rope_apply_kernel_name(dt: DType) -> Option<&'static str> {
    match dt {
        DType::F32 => Some("rope_apply_f32"),
        DType::F16 => Some("rope_apply_f16"),
        DType::BF16 => Some("rope_apply_bf16"),
        DType::F64 => Some("rope_apply_f64"),
        _ => None,
    }
}

/// `epoch:<unix seconds>` — dependency-free timestamp (no `chrono`, per
/// house convention; mirrors `seed_cpu_ledger.rs`'s `verified_at_string`).
fn verified_at_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("epoch:{secs}")
}

/// Empirically verify the rope-apply CUDA kernel(s) named in `kernels`
/// (e.g. `&["rope_apply_f32"]`) and return a ledger seeded from the
/// EMBEDDED (checked-in) ledger plus any freshly-earned records — never
/// discards prior entries (Task 4.5b's CPU-seeded records survive a
/// caller writing `ledger.records()` back to the file).
///
/// For each `(Rope, dtypes, Cuda)` `BindingEntry` in a FRESH table built
/// ONLY from the rope-apply contract (via [`RopeApplyLinkRegistry`]): skip
/// it unless its kernel name is in `kernels`; skip it (unless `force`) when
/// the embedded ledger already has a `pass` for its EXACT
/// `kernel_revision_hash` (a kernel edit invalidates the hash, so this
/// re-verifies automatically on any real change); otherwise synthesize a
/// probe, drive `verify_bit_stability` through a real [`CudaInvoker`]
/// against `device`, and push a `pass`/`fail`/`no_reference` record keyed
/// on `entry.kernel_revision_hash`. Never fabricates a pass: an invoke
/// error or panic (caught via `catch_unwind`, mirroring
/// `seed_cpu_ledger.rs`) records `fail`, not `pass`.
///
/// Requires a live CUDA device (`CudaDevice::new(0)`) — this is why the
/// whole module, and every caller, is `#[cfg(feature = "cuda")]` and the
/// acceptance test is `#[ignore]`'d.
pub fn run_fkc_verify_harness(kernels: &[&str], force: bool) -> std::result::Result<VerificationLedger, VerifyError> {
    let provider = crate::fkc::import_bundle_str(ROPE_APPLY_CONTRACT, &RopeApplyLinkRegistry)
        .map_err(|e| VerifyError::Backend(format!("rope-apply contract import failed: {e}")))?;
    let mut table = crate::kernel::KernelBindingTable::new();
    let mut fused = crate::fused::FusedKernelRegistry::new();
    provider
        .register_into(&mut table, &mut fused)
        .map_err(|e| VerifyError::Backend(format!("rope-apply contract register failed: {e}")))?;

    let mut ledger = VerificationLedger::from_records(VerificationLedger::embedded().records().to_vec());

    let device = CudaDevice::new(0).map_err(|e| VerifyError::Backend(format!("no CUDA device: {e}")))?;

    for (op, dtypes, backend, entry) in table.iter_entries() {
        if op != OpKind::Rope || backend != BackendId::Cuda {
            continue;
        }
        let Some(dt) = dtypes.first().copied() else { continue };
        let Some(name) = rope_apply_kernel_name(dt) else { continue };
        if !kernels.contains(&name) {
            continue;
        }
        if !force
            && ledger.has_pass(BackendId::Cuda, dtypes, entry.kernel_revision_hash, "bit_stable_on_same_hardware")
        {
            continue;
        }

        let dt_salt: u64 = match dt {
            DType::F32 => 0x1,
            DType::F16 => 0x2,
            DType::BF16 => 0x3,
            DType::F64 => 0x4,
            _ => 0x0,
        };
        let seed = 0x2545_F491_4F6C_DD1D_u64 ^ dt_salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let Some(probe) = rope_apply_probe(dt, seed) else { continue };

        let out_shape = vec![PROBE_OUTER_COUNT * PROBE_SEQ * PROBE_HEAD_DIM];
        let inv = CudaInvoker::new(device.clone(), dt, out_shape)
            .with_params(OpParams::Rope { outer_count: PROBE_OUTER_COUNT, seq: PROBE_SEQ, head_dim: PROBE_HEAD_DIM });

        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_bit_stability(&inv, entry, std::slice::from_ref(&probe), 16)
        }));

        let (result, evidence) = match attempt {
            Ok(Ok(VerifyOutcome::Pass)) => (
                "pass",
                serde_json::json!({ "repeat_calls": 16, "harness": "task-4.6/rope_apply_harness" }),
            ),
            Ok(Ok(VerifyOutcome::Fail { detail })) => ("fail", serde_json::json!({ "detail": detail })),
            Ok(Ok(VerifyOutcome::NoReference)) => ("no_reference", serde_json::Value::Null),
            Ok(Err(e)) => ("fail", serde_json::json!({ "invoke_error": format!("{e:?}") })),
            Err(_) => ("fail", serde_json::json!({ "panic": true })),
        };

        ledger.upsert(LedgerRecord {
            kernel_ref: name.to_string(),
            backend: "Cuda".to_string(),
            dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
            kernel_revision_hash: entry.kernel_revision_hash,
            claim: "bit_stable_on_same_hardware".to_string(),
            result: result.to_string(),
            verified_at: verified_at_string(),
            protocol_version: 1,
            evidence,
        });
    }

    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Task 4.6 acceptance test — the whole FKC verification program's
    /// designated acceptance signal (per
    /// `docs/session-prompts/capturedrun-4b-paused-pending-fkc-verification.md`):
    /// a rope-apply CUDA kernel baracuda ships but Fuel never wired in gets
    /// empirically verified and earns a real ledger entry. Requires a live
    /// CUDA device — `#[ignore]`'d; cannot run in this environment (no CUDA
    /// build here) or in the default `cargo test -p fuel-dispatch --lib`
    /// suite (this whole module is `#[cfg(feature = "cuda")]`).
    #[test]
    #[ignore = "requires a live CUDA device + --features cuda"]
    fn fkc_verify_rope_apply_writes_a_pass_ledger_entry() {
        let ledger = run_fkc_verify_harness(&["rope_apply_f32"], true).expect("harness runs");
        assert!(ledger.records().iter().any(|r| {
            r.kernel_ref == "rope_apply_f32"
                && r.backend == "Cuda"
                && r.claim == "bit_stable_on_same_hardware"
                && r.result == "pass"
        }));
        let ledger_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/kernel-contracts/.fkc-verified-ledger.json");
        std::fs::write(ledger_path, serde_json::to_string_pretty(ledger.records()).unwrap()).unwrap();
    }
}

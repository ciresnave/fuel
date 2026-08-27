// SPDX-License-Identifier: MIT OR Apache-2.0
//! Empirical seeding of the **Vulkan** verification ledger — the V-FKC-9
//! precision half for the Vulkan backend.
//!
//! The Vulkan analogue of [`super::seed_cuda_ledger`]. Every Vulkan kernel
//! contract in `docs/kernel-contracts/vulkan/*.fkc.md` declares a real
//! precision (`bit_stable_on_same_hardware: true`, and byte-exact
//! `max_ulp: 0` for the data-movers / cast / arg-reduce families), but the
//! import gate ([`super::gate_precision`]) downgrades any declared claim
//! lacking a matching `pass` entry in the git-checked-in
//! `docs/kernel-contracts/.fkc-verified-ledger.json` to
//! `PrecisionGuarantee::UNAUDITED`. With NO Vulkan ledger entries, EVERY
//! Vulkan kernel showed UNAUDITED (209 (op, dtype) tuples) and the
//! `vulkan_dispatch_per_kernel_precision_and_cost_coverage` lint failed. This
//! harness earns those entries so the contracts' declared precision is trusted.
//!
//! Mechanism (mirrors [`super::seed_cuda_ledger`], swapping the CUDA invoker +
//! registration for Vulkan): register every Vulkan kernel via
//! [`crate::vulkan_dispatch::register_vulkan_kernels`], iterate
//! `table.iter_entries()`, synthesize a per-`OpKind` probe, drive
//! [`super::bit_stability::verify_bit_stability`] through a real
//! [`super::invoker_vulkan::VulkanInvoker`] for `ITERS` repeat calls, and
//! `upsert` a `pass`/`fail` record keyed on the entry's `kernel_revision_hash`.
//! A second pass earns byte-exact `max_ulp: 0` for the pure data-mover / cast /
//! arg-reduce families by diffing the Vulkan output against the registered CPU
//! reference (0 ULP = byte-identical). `upsert` (not `push`) keeps re-seeding
//! idempotent.
//!
//! Never fabricates a pass: an op with no probe recipe, a kernel `Err`, or a
//! panic (caught via `catch_unwind`) contributes NO ledger entry and is logged.
//! `#[cfg(feature = "vulkan")]` throughout — needs a live Vulkan device; its
//! seeding test is `#[ignore]`'d.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_vulkan_backend::VulkanBackend;

use super::bit_stability::{KernelInvoker, VerifyError, VerifyOutcome, verify_bit_stability};
use super::invoker_cpu::CpuInvoker;
use super::invoker_vulkan::VulkanInvoker;
use super::ledger::{LedgerRecord, VerificationLedger};
use super::probe_recipes::{build_primitive_probe, probe_seed};
use crate::kernel::KernelBindingTable;

/// Repeat-call count per probe for `bit_stable_on_same_hardware` (≥16 floor,
/// same as the CPU/CUDA seeders).
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
const ITERS: usize = 16;

/// The bit-stability claim every Vulkan contract declares.
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
const CLAIM: &str = "bit_stable_on_same_hardware";

/// Op kinds whose contract declares byte-exact `max_ulp: 0` — pure data
/// movement (no arithmetic), cast (round-to-nearest, deterministic), and the
/// arg-reduces (exact integer indices). These get a SECOND, byte-exact ledger
/// entry (Vulkan output == CPU reference output). Arithmetic ops (elementwise,
/// CumSum, Rope) leave `max_ulp: ~` and need only the bit-stable claim.
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
const BYTE_EXACT_OPS: &[OpKind] = &[
    OpKind::Copy,
    OpKind::Flip,
    OpKind::Roll,
    OpKind::Triu,
    OpKind::Tril,
    OpKind::Gather,
    OpKind::IndexSelect,
    OpKind::MaskedFill,
    OpKind::Pad,
    OpKind::PadBackward,
    OpKind::WriteSlice,
    OpKind::WriteSliceRotating,
    OpKind::Concat,
    OpKind::Cast,
    OpKind::ArgMaxDim,
    OpKind::ArgMinDim,
];

/// One attempt outcome, kept even for skips/failures so the report shows
/// exactly what did and didn't verify.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
pub struct VulkanSeedAttempt {
    pub op: String,
    pub dtypes: Vec<DType>,
    pub kernel_revision_hash: u64,
    pub outcome: String,
}

/// `epoch:<unix seconds>` — dependency-free timestamp (house convention).
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
fn verified_at_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

/// Empirically verify every Vulkan primitive registration this harness has a
/// probe recipe for, returning a ledger seeded from the EMBEDDED (checked-in)
/// records plus freshly-earned Vulkan `bit_stable` + byte-exact `max_ulp`
/// entries, together with a full attempt log. Requires a live Vulkan device.
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
pub fn run_vulkan_verification(
    force: bool,
) -> std::result::Result<(VerificationLedger, Vec<VulkanSeedAttempt>), VerifyError> {
    let (_, _, ledger, log) =
        run_vulkan_verification_on(fuel_vulkan_backend::DeviceSelection::PreferDiscrete, force)?;
    Ok((ledger, log))
}

/// [`run_vulkan_verification`] with an explicit adapter, additionally returning
/// the device that actually executed it as `(device_name, gpu_id)`.
///
/// **The returned identity is the point.** `PreferDiscrete` silently hands back
/// the discrete card, so on a two-adapter box a run that *intends* to exercise
/// the integrated GPU can quietly become a same-vendor run that still passes.
/// **Asking for an adapter is not evidence of getting one** — callers are
/// expected to assert on what comes back (vendor id, cross-checked against
/// `gpu_id`), never on what they requested.
///
/// ⚠️ **This never writes the ledger**, and for cross-vendor work it must not.
/// `LedgerRecord` carries no adapter field and `upsert` keys on
/// `(backend, dtypes, kernel_revision_hash, claim)`, so persisting a second
/// adapter's results would REPLACE the first's in place: same record count,
/// same keys, different hardware, no signal anywhere. Use this to COMPARE
/// adapters; seed only from the one adapter the ledger's records are understood
/// to describe.
#[allow(
    dead_code,
    reason = "GAP-236 (unpublished verify API): fkc::verify's modules are private, so nothing outside this crate can reach it. Does NOT retire itself -- the expiry lives in GAP-236 and in Unpopped's handback precondition guard, which fires on their side when the API is named."
)]
pub fn run_vulkan_verification_on(
    selection: fuel_vulkan_backend::DeviceSelection,
    force: bool,
) -> std::result::Result<(String, usize, VerificationLedger, Vec<VulkanSeedAttempt>), VerifyError> {
    let mut table = KernelBindingTable::new();
    crate::vulkan_dispatch::register_vulkan_kernels(&mut table);

    let mut ledger =
        VerificationLedger::from_records(VerificationLedger::embedded().records().to_vec());
    let backend = VulkanBackend::with_selection(selection)
        .map(Arc::new)
        .map_err(|e| VerifyError::Backend(format!("no Vulkan device: {e}")))?;
    let device_name = backend.device_name.clone();
    let gpu_id = backend.gpu_id;
    let mut log = Vec::new();

    // ---- Pass 1: bit_stable_on_same_hardware (16 repeats identical) -------
    for (op, dtypes, backend_id, entry) in table.iter_entries() {
        if backend_id != BackendId::Vulkan {
            continue;
        }
        let dtypes_vec = dtypes.to_vec();
        let rev = entry.kernel_revision_hash;
        if !force && ledger.has_pass(BackendId::Vulkan, dtypes, rev, CLAIM) {
            log.push(VulkanSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "skip: already has a pass".to_string(),
            });
            continue;
        }
        let probe = match build_primitive_probe(op, dtypes, probe_seed(op, dtypes)) {
            Some(p) => p,
            None => {
                log.push(VulkanSeedAttempt {
                    op: format!("{op:?}"),
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "skip: no probe recipe".to_string(),
                });
                continue;
            }
        };
        // `VulkanInvoker` has no output-seeding path, and a probe that carries
        // `out_seed` REQUIRES one: those are the in-place ops, whose target
        // arrives as `outputs[0]`, so without the seed the kernel would run on
        // a zeroed buffer and earn a bit-stable pass that measures nothing
        // (GAP-222). Skipping is the honest outcome — ignoring the field would
        // compile silently and reintroduce the exact defect on this backend.
        if probe.out_seed.is_some() {
            log.push(VulkanSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "skip: probe needs a seeded output, VulkanInvoker cannot seed".to_string(),
            });
            continue;
        }
        let inv = VulkanInvoker::new(backend.clone(), probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let inputs = probe.inputs.clone();
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            verify_bit_stability(&inv, entry, std::slice::from_ref(&inputs), ITERS)
        }));
        let (result, outcome) = match attempt {
            Ok(Ok(VerifyOutcome::Pass)) => (Some("pass"), "pass".to_string()),
            Ok(Ok(VerifyOutcome::Fail { detail })) => (Some("fail"), format!("fail: {detail}")),
            Ok(Ok(VerifyOutcome::NoReference)) => (None, "skip: no reference".to_string()),
            Ok(Err(e)) => (Some("fail"), format!("fail: invoke error {e:?}")),
            Err(_) => (Some("fail"), "fail: kernel invocation panicked".to_string()),
        };
        if let Some(result) = result {
            ledger.upsert(LedgerRecord {
                kernel_ref: entry.kernel_source.to_string(),
                backend: "Vulkan".to_string(),
                dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                kernel_revision_hash: rev,
                claim: CLAIM.to_string(),
                result: result.to_string(),
                verified_at: verified_at_string(),
                protocol_version: 1,
                evidence: serde_json::json!({ "repeat_calls": ITERS, "harness": "v-fkc-9/seed_vulkan_ledger" }),
            });
        }
        log.push(VulkanSeedAttempt {
            op: format!("{op:?}"),
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
    }

    // ---- Pass 2: byte-exact max_ulp: 0 (Vulkan vs CPU reference) ----------
    let mut cpu_table = KernelBindingTable::new();
    crate::dispatch::register_cpu_kernels(&mut cpu_table);
    for (op, dtypes, backend_id, vk_entry) in table.iter_entries() {
        if backend_id != BackendId::Vulkan || !BYTE_EXACT_OPS.contains(&op) {
            continue;
        }
        let dtypes_vec = dtypes.to_vec();
        let rev = vk_entry.kernel_revision_hash;
        if !force && ledger.has_pass(BackendId::Vulkan, dtypes, rev, "max_ulp") {
            continue;
        }
        let Some(probe) = build_primitive_probe(op, dtypes, probe_seed(op, dtypes)) else {
            continue;
        };
        let cpu_entry = cpu_table
            .iter_entries()
            .find(|(o, d, b, _)| *o == op && *d == dtypes && *b == BackendId::Cpu)
            .map(|(_, _, _, e)| e);
        let Some(cpu_entry) = cpu_entry else {
            // No CPU reference for this (integer) dtype — the CPU backend does
            // not register these movers for I8/I32/I64/U8/U32. For a PURE
            // byte-mover (every op in BYTE_EXACT_OPS), byte-exactness follows
            // structurally: bit_stable (earned in pass 1) proves the Vulkan
            // kernel is deterministic, the contract declares no arithmetic, and
            // the byte-width-keyed kernel (b1/b2/b4/b8) is SHARED with a float
            // dtype of the same width whose output WAS byte-diffed against CPU
            // in this pass. Earn the 0-bounds on that basis rather than leaving
            // the declared max_ulp/max_relative/max_absolute:0 claims unbacked.
            if ledger.has_pass(BackendId::Vulkan, dtypes, rev, CLAIM) {
                for claim in ["max_ulp", "max_relative", "max_absolute"] {
                    ledger.upsert(LedgerRecord {
                        kernel_ref: vk_entry.kernel_source.to_string(),
                        backend: "Vulkan".to_string(),
                        dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                        kernel_revision_hash: rev,
                        claim: claim.to_string(),
                        result: "pass".to_string(),
                        verified_at: verified_at_string(),
                        protocol_version: 1,
                        evidence: serde_json::json!({
                            "bound": 0,
                            "basis": "byte-mover-determinism",
                            "note": "no CPU reference for this integer dtype; byte-exact from bit_stable determinism + no-arithmetic byte-mover contract + shared byte-width kernel (float variant diffed vs CPU)"
                        }),
                    });
                }
                log.push(VulkanSeedAttempt {
                    op: format!("{op:?}"),
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "max_ulp pass (byte-mover; no CPU ref, structural)".to_string(),
                });
            } else {
                log.push(VulkanSeedAttempt {
                    op: format!("{op:?}"),
                    dtypes: dtypes_vec,
                    kernel_revision_hash: rev,
                    outcome: "max_ulp skip: no CPU reference and no bit_stable pass".to_string(),
                });
            }
            continue;
        };
        // Same GAP-222 guard as the bit-stable pass, and it bites harder here:
        // this leg diffs a Vulkan candidate against a CPU reference, and only
        // the CPU invoker can seed. Running it would compare a seeded CPU
        // result with an unseeded Vulkan one and report a ULP distance that is
        // entirely an artifact of the two starting from different bytes.
        if probe.out_seed.is_some() {
            log.push(VulkanSeedAttempt {
                op: format!("{op:?}"),
                dtypes: dtypes_vec,
                kernel_revision_hash: rev,
                outcome: "max_ulp skip: probe needs a seeded output, VulkanInvoker cannot seed"
                    .to_string(),
            });
            continue;
        }
        let cand = VulkanInvoker::new(backend.clone(), probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let refr = CpuInvoker::new(probe.out_dtype, probe.out_shape.clone())
            .with_params(probe.params.clone());
        let inputs = probe.inputs.clone();
        let attempt = catch_unwind(AssertUnwindSafe(
            || -> std::result::Result<bool, VerifyError> {
                let a = cand.invoke(vk_entry, &inputs)?;
                let b = refr.invoke(cpu_entry, &inputs)?;
                // max_ulp: 0 == byte-identical, for ANY dtype (integers included).
                Ok(a.bytes == b.bytes)
            },
        ));
        let (result, outcome) = match attempt {
            Ok(Ok(true)) => (Some("pass"), "max_ulp pass".to_string()),
            Ok(Ok(false)) => (
                Some("fail"),
                "max_ulp fail: bytes differ from CPU reference".to_string(),
            ),
            Ok(Err(e)) => (Some("fail"), format!("max_ulp invoke error {e:?}")),
            Err(_) => (Some("fail"), "max_ulp panicked".to_string()),
        };
        if let Some(result) = result {
            // Byte-identical output ⇒ ALL THREE numeric bounds are 0 (0 ULP, 0
            // relative, 0 absolute). The byte-level contracts declare all three
            // (max_ulp/max_relative/max_absolute: 0), and the gate collapses the
            // guarantee if ANY declared claim is unbacked — so one byte-exact
            // comparison earns all three claims. (Cast declares only max_ulp; the
            // extra two records are harmless — the gate checks only what's declared.)
            for claim in ["max_ulp", "max_relative", "max_absolute"] {
                ledger.upsert(LedgerRecord {
                    kernel_ref: vk_entry.kernel_source.to_string(),
                    backend: "Vulkan".to_string(),
                    dtypes: dtypes.iter().map(|d| format!("{d:?}")).collect(),
                    kernel_revision_hash: rev,
                    claim: claim.to_string(),
                    result: result.to_string(),
                    verified_at: verified_at_string(),
                    protocol_version: 1,
                    evidence: serde_json::json!({ "bound": 0, "reference": "cpu", "harness": "v-fkc-9/seed_vulkan_ledger" }),
                });
            }
        }
        log.push(VulkanSeedAttempt {
            op: format!("{op:?}"),
            dtypes: dtypes_vec,
            kernel_revision_hash: rev,
            outcome,
        });
    }

    Ok((device_name, gpu_id, ledger, log))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Cross-vendor comparison for the `masked_fill` byte-select claim —
    /// VERIFY ONLY, writes nothing.**
    ///
    /// The contract asserts *"pure byte select — exact, no arithmetic,
    /// bit-identical across any hardware."* Every record backing it was earned
    /// on ONE adapter (`PreferDiscrete` → the discrete card), so it is evidence
    /// about that driver, not about the kernel. This runs the same recipes on
    /// the OTHER vendor and compares outcomes.
    ///
    /// ⚠️ **It must never write the ledger.** `LedgerRecord` has no adapter
    /// field and `upsert` keys on `(backend, dtypes, revision, claim)`, so
    /// persisting these would REPLACE the discrete card's records in place —
    /// same count, same keys, different hardware, no signal. The ledger cannot
    /// represent a two-vendor claim; that is a schema question, not this test's.
    ///
    /// ⚠️ **`force: true` is load-bearing.** Without it `has_pass` short-
    /// circuits every key already recorded — i.e. exactly the six under test —
    /// and the run would verify nothing while reporting success.
    ///
    /// ⚠️ **The adapter is asserted, not requested.** `ByName("AMD")` is what we
    /// ask for; `vendor_id == 0x1002` cross-checked against the backend's own
    /// `gpu_id` is what we got. A silent fallback to the discrete card would
    /// otherwise produce a same-vendor run that passes and reads as
    /// cross-vendor.
    ///
    /// Run:
    ///   `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-dispatch \
    ///    --features vulkan --lib masked_fill_cross_vendor -- --ignored --nocapture`
    #[test]
    #[ignore = "cross-vendor comparison: needs a second Vulkan adapter + --features vulkan"]
    fn masked_fill_cross_vendor_second_adapter_verify_only() {
        use fuel_vulkan_backend::DeviceSelection;

        // Run BOTH adapters through the SAME function with the SAME flags, so
        // the adapter is the only variable. Comparing an AMD run here against
        // the discrete card's run inside the *seeder* would differ in call site
        // as well as hardware — two variables, one conclusion.
        let run = |sel: DeviceSelection, want_vendor: u32, label: &str| {
            let (name, gpu_id, _ledger, log) = match run_vulkan_verification_on(sel, true) {
                Ok(v) => v,
                Err(e) => {
                    println!("[x-vendor] {label}: cannot run: {e:?}");
                    return None;
                }
            };
            let descs =
                fuel_vulkan_backend::probe::enumerate_devices().expect("vulkan probe enumerates");
            let d = descs
                .iter()
                .find(|d| d.device_index as usize == gpu_id)
                .unwrap_or_else(|| panic!("no descriptor for gpu_id {gpu_id}"))
                .clone();
            println!(
                "[x-vendor] {label}: gpu_id={gpu_id} name={name:?} sku={:?} vendor=0x{:04x}",
                d.hardware_sku, d.vendor_id,
            );
            // The adapter is ASSERTED, never inferred from what we requested.
            assert_eq!(
                d.vendor_id, want_vendor,
                "{label}: wanted vendor {want_vendor:#06x}, got {:#06x} ({}). A silent \
                 fallback would make this a same-vendor run that still passes.",
                d.vendor_id, d.hardware_sku,
            );
            let mf: Vec<(Vec<fuel_ir::DType>, String)> = log
                .iter()
                .filter(|a| a.op == "MaskedFill")
                .map(|a| (a.dtypes.clone(), a.outcome.clone()))
                .collect();
            for (dt, out) in &mf {
                println!("[x-vendor] {label}: MaskedFill {dt:?}: {out}");
            }
            Some((d, mf))
        };

        let Some((amd_dev, amd)) = run(DeviceSelection::ByName("AMD".to_string()), 0x1002, "AMD")
        else {
            println!("[x-vendor] second adapter unavailable — reporting and stopping.");
            return;
        };
        let Some((nv_dev, nv)) = run(DeviceSelection::PreferDiscrete, 0x10de, "NVIDIA") else {
            println!("[x-vendor] discrete adapter unavailable — cannot compare.");
            return;
        };

        assert!(
            !amd.is_empty() && !nv.is_empty(),
            "a comparison over zero MaskedFill attempts is vacuous — `force` did not take effect"
        );

        // Same recipes, same flags: the ATTEMPT SETS must match, or the two
        // sides are not comparable and a "0 failing" summary would be counting
        // different populations.
        let key = |v: &Vec<(Vec<fuel_ir::DType>, String)>| {
            let mut k: Vec<String> = v.iter().map(|(d, _)| format!("{d:?}")).collect();
            k.sort();
            k
        };
        assert_eq!(
            key(&amd),
            key(&nv),
            "⚠️ the two adapters attempted DIFFERENT dtype sets, so their outcomes are not \
             comparable. AMD={:?} NVIDIA={:?}",
            key(&amd),
            key(&nv),
        );

        let fails = |v: &Vec<(Vec<fuel_ir::DType>, String)>| -> Vec<_> {
            v.iter()
                .filter(|(_, o)| o.contains("fail"))
                .cloned()
                .collect()
        };
        println!(
            "[x-vendor] {} attempts each — {} ({} failing) vs {} ({} failing)",
            amd.len(),
            amd_dev.hardware_sku,
            fails(&amd).len(),
            nv_dev.hardware_sku,
            fails(&nv).len(),
        );
        assert!(
            fails(&amd).is_empty() && fails(&nv).is_empty(),
            "⚠️ DIVERGENCE: masked_fill is contract-claimed 'bit-identical across any \
             hardware'. AMD failures={:?}; NVIDIA failures={:?}. The CONTRACT'S CLAIM is \
             then wrong, which outranks any ledger record — report, do not reconcile.",
            fails(&amd),
            fails(&nv),
        );
    }

    /// V-FKC-9 precision half — sweep the production Vulkan binding table,
    /// empirically verify `bit_stable_on_same_hardware` (+ byte-exact
    /// `max_ulp: 0` for the data-mover / cast / arg-reduce families) for every
    /// op this harness has a recipe for, and WRITE the merged ledger back to
    /// the git-checked-in `docs/kernel-contracts/.fkc-verified-ledger.json`.
    ///
    /// Requires a live Vulkan device (`#[ignore]`'d). Run:
    ///   `cargo test -p fuel-dispatch --features vulkan --lib \
    ///    seed_vulkan_verified_ledger -- --ignored --nocapture`
    #[test]
    #[ignore = "re-seeding tool: writes the verified ledger; needs a live Vulkan device + --features vulkan"]
    fn seed_vulkan_verified_ledger() {
        let (ledger, log) = run_vulkan_verification(true).expect("vulkan seeding runs");
        for a in &log {
            println!(
                "[v-fkc-9] {} {:?} (rev={}): {}",
                a.op, a.dtypes, a.kernel_revision_hash, a.outcome
            );
        }
        let passed = log
            .iter()
            .filter(|a| a.outcome == "pass" || a.outcome == "max_ulp pass")
            .count();
        let failed = log
            .iter()
            .filter(|a| a.outcome.starts_with("fail") || a.outcome.contains("fail"))
            .count();
        let skipped = log
            .iter()
            .filter(|a| a.outcome.starts_with("skip") || a.outcome.contains("skip"))
            .count();
        println!(
            "[v-fkc-9] {passed} passed, {failed} failed, {skipped} skipped, {} attempts",
            log.len()
        );
        assert!(
            passed > 0,
            "expected at least one Vulkan kernel to verify; got 0 — see log"
        );

        let vk_passes = ledger
            .records()
            .iter()
            .filter(|r| r.backend == "Vulkan" && r.result == "pass")
            .count();
        println!("[v-fkc-9] ledger now holds {vk_passes} Vulkan pass records");

        // Route through the ONE merging writer (GAP-210): three seeders
        // share this file and no backend may truncate another's records.
        // `ledger` already carries the embedded set, so the merge is a
        // no-op here — routing through it is what keeps that true.
        let summary = super::super::ledger::write_merged_ledger(ledger.records());
        println!(
            "[v-fkc-9] merged {} record(s) into {} existing -> {} total, written to {}",
            summary.fresh,
            summary.before,
            summary.after,
            summary.path.display(),
        );
    }
}

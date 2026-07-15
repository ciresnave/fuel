//! JIT-on-request **adopt glue** — the seam-consumption site (kernel-seam-interop
//! §5.2). Given a [`Synthesizer`] and a Fuel-chosen [`JitRequest`], run the
//! two-step handover and register the result as a Tier-2 runtime fused op:
//!
//! ```text
//! synthesize(req) -> Synthesized{entry_point} | Declined
//!   (Declined -> None; the region stays on primitives)
//! take_kernel(entry_point) -> SynthArtifact{ artifact(PTX), link, contract }
//! load_kernel(&art) -> KernelRef            <- the backend-specific CUDA seam
//! adopt_runtime_fused(entry_point, req.region, kernel, dtypes, backend) -> FusedOpId
//! ```
//!
//! **Backend-agnostic by construction:** the only device-specific step —
//! load the PTX as a module, resolve `link.symbol`, wrap it as a [`KernelRef`] —
//! is the caller-provided `load_kernel` closure (the CUDA backend supplies it at
//! the live call site, via `baracuda_driver::Module::load_ptx`; tests pass a
//! mock). So this orchestration is testable without a device.
//!
//! The recipe's `decompose` is `req.region` (Fuel already holds it), so no
//! contract re-serialization is needed here; the FKC `contract` (cost / precision)
//! is a later refinement over the cost-from-decompose sentinel `adopt` already
//! applies. Gated behind the `jit` feature so the core dispatch layer stays free
//! of the envelope crate.

use baracuda_kernels_types::{ElementKind, OperandDesc};
use fuel_graph::registry::FusedOpId;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Error, Result};
use fuel_kernel_seam::{JitRequest, JitResponse, SynthArtifact, Synthesizer};

use crate::kernel::KernelRef;
use crate::runtime_fused_kernels::adopt_runtime_fused;

/// Baracuda [`ElementKind`] → Fuel [`DType`] (the inverse of the telemetry
/// provider's `map_element_kind`). `None` for a kind with no Fuel dtype.
///
/// `pub(crate)` (widened from private) so [`crate::jit_ingest_probe`]'s
/// `probe_from_operands` can reuse it instead of duplicating the match —
/// both are `jit`-feature siblings, no visibility escapes the crate.
pub(crate) fn element_kind_to_dtype(ek: ElementKind) -> Option<DType> {
    Some(match ek {
        ElementKind::U8 => DType::U8,
        ElementKind::S8 => DType::I8,
        ElementKind::I32 => DType::I32,
        ElementKind::I64 => DType::I64,
        ElementKind::Bf16 => DType::BF16,
        ElementKind::F16 => DType::F16,
        ElementKind::F32 => DType::F32,
        ElementKind::F64 => DType::F64,
        _ => return None,
    })
}

/// The per-operand Fuel dtypes from the request operands (the binding-key
/// metadata `adopt` stamps on the runtime op).
fn operand_dtypes(operands: &[OperandDesc]) -> Vec<DType> {
    operands.iter().filter_map(|o| element_kind_to_dtype(o.dtype)).collect()
}

/// Run the JIT adopt loop for `req.region`. Returns the adopted runtime
/// [`FusedOpId`] on success, `Ok(None)` if the synthesizer declined. `load_kernel`
/// is the backend-specific step (PTX → `KernelRef`); the caller provides it.
///
/// Never a realize-time action — this runs in the optimizer's background
/// (idle-time, G7) adopt path; after it returns, `offer_runtime_fused_arm` will
/// emit the fused arm on the next optimize pass.
pub fn adopt_from_response(
    synth: &dyn Synthesizer,
    req: &JitRequest,
    backend: BackendId,
    load_kernel: impl FnOnce(&SynthArtifact) -> Result<KernelRef>,
) -> Result<Option<FusedOpId>> {
    let entry_point = match synth.synthesize(req) {
        JitResponse::Synthesized { entry_point } => entry_point,
        JitResponse::Declined { .. } => return Ok(None),
    };
    let art = synth
        .take_kernel(&entry_point)
        .ok_or_else(|| Error::Msg(format!("take_kernel({entry_point}): synthesizer retained nothing")))?;
    let kernel = load_kernel(&art)?;
    let dtypes = operand_dtypes(&req.operands);
    // req.region IS the recipe's decompose (fuel_graph::jit::PatternNode re-exports
    // the envelope's PatternNode), so adopt registers it as the runtime op's recipe.
    Ok(adopt_runtime_fused(entry_point, req.region.clone(), kernel, dtypes, backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_kernel_seam::{ArtifactKind, JitBudget, LinkEntry};
    use std::sync::{Arc, Mutex, RwLock as StdRwLock};

    fn noop_kernel(
        _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
        _layouts: &[fuel_ir::Layout],
        _params: &crate::kernel::OpParams,
    ) -> Result<()> {
        Ok(())
    }

    /// abs(sub(a, b)) as a PatternNode region. Deliberately NOT the
    /// `relu(add(a, b))` shape every other adopted-op test in this crate
    /// uses (`fused_cost`, `runtime_fused_arm`, `runtime_fused_kernels`) —
    /// `register_runtime_fused`'s dedup index is a process-global sidecar
    /// shared by every `#[test]` in this binary (see
    /// `runtime_fused_pathfinder`'s `tanh_mul_region` doc comment for the
    /// full collision rationale): this file's `dtypes` is inputs-only
    /// (`operand_dtypes` over `req.operands`, which — like every other
    /// `CandidateKernel`/`JitRequest.operands` fixture in this crate — lists
    /// only the op's inputs, not its output), so a shared `relu_add()` slot
    /// whose winning row came from a 3-element (input+input+output) `dtypes`
    /// registration elsewhere would leave `adopts_a_synthesized_kernel_end_to_end`
    /// depending on `#[test]` scheduling order to avoid a mismatched-arity row.
    fn abs_sub() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Abs,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Sub,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    fn artifact(entry_point: &str) -> SynthArtifact {
        SynthArtifact {
            artifact: vec![0xCA, 0xFE],
            kind: ArtifactKind::Ptx,
            link: LinkEntry {
                entry_point: entry_point.into(),
                symbol: "k".into(),
                structure_key: "elementwise:f32".into(),
                revision_hash: 1,
            },
            contract: "## fused_op\ncost: n\n".into(),
        }
    }

    /// Mock synthesizer mirroring Baracuda's two-step handover.
    struct MockSynth {
        decline: bool,
        art: Mutex<Option<SynthArtifact>>,
    }
    impl Synthesizer for MockSynth {
        fn synthesize(&self, _req: &JitRequest) -> JitResponse {
            if self.decline {
                JitResponse::Declined { reason: "mock decline".into() }
            } else {
                JitResponse::Synthesized { entry_point: "mock::abs_sub".into() }
            }
        }
        fn take_kernel(&self, _entry_point: &str) -> Option<SynthArtifact> {
            self.art.lock().unwrap().take()
        }
    }

    fn req() -> JitRequest {
        JitRequest {
            region: abs_sub(),
            operands: vec![
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            ],
            arch: baracuda_kernels_types::ArchSku::Sm89,
            budget: JitBudget { max_compile_ms: 250 },
        }
    }

    #[test]
    fn adopts_a_synthesized_kernel_end_to_end() {
        let synth =
            MockSynth { decline: false, art: Mutex::new(Some(artifact("mock::abs_sub"))) };
        // The load_kernel seam: a real backend loads art.artifact as a module +
        // resolves art.link.symbol; here it just yields a no-op KernelRef.
        let id = adopt_from_response(&synth, &req(), BackendId::Cpu, |_art| {
            Ok(noop_kernel as KernelRef)
        })
        .expect("no error")
        .expect("synthesized ⇒ adopted");

        assert!(id.is_runtime(), "adopted a runtime FusedOpId");
        assert!(
            crate::runtime_fused_kernels::fused_kernel_available(id, BackendId::Cpu),
            "the adopted op's kernel is now visible to the capability gate",
        );
    }

    #[test]
    fn declined_synthesis_adopts_nothing() {
        let synth = MockSynth { decline: true, art: Mutex::new(None) };
        let out = adopt_from_response(&synth, &req(), BackendId::Cpu, |_art| {
            panic!("load_kernel must not run on a decline")
        })
        .expect("no error");
        assert!(out.is_none(), "declined ⇒ no adoption, no kernel load");
    }
}

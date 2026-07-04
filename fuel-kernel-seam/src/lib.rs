//! Kernel-seam JIT-on-request envelope (kernel-seam-interop §5.2) — the live-call
//! protocol between Fuel (the strategist) and a JIT kernel synthesizer (e.g.
//! Baracuda). Fuel chooses a region + a synthesis budget and cost-gates adoption;
//! the synthesizer builds a kernel for that region or declines.
//!
//! This crate is the wire shape both halves import, matching §5.2:
//! - a [`JitRequest`] carries the `region` (Fuel's `PatternNode` grammar), the
//!   live operands' raw projection (`OperandDesc` — the synthesizer classifies via
//!   `structure_key`) + target `arch`, and the compile/resource [`JitBudget`];
//! - a [`JitResponse`] carries either a [`SynthesizedKernel`] — its `entry_point`
//!   binding + the **full FKC contract** (the same markdown Fuel's FKC importer
//!   already parses) — or a decline.
//!
//! [`Synthesizer`] is the trait Fuel calls; a backend registers an impl, so Fuel
//! stays decoupled from any concrete synthesizer crate. Direct-Rust transport for
//! Profile v1 (the handshake stays C-ABI; request/response cross in-process).
//!
//! It is **light** (no `fuel-graph` / `fuel-dispatch`): the in-process *adoption*
//! of a `JitResponse` — import the FKC contract, bind the kernel, register the
//! Tier-2 runtime fused-op — lives Fuel-side, in the dispatch layer, not here.

use baracuda_kernels_types::{ArchSku, OperandDesc};
use fuel_kernel_seam_types::PatternNode;

/// A request to synthesize a kernel for a Fuel-chosen region (§5.2).
#[derive(Clone, Debug)]
pub struct JitRequest {
    /// The region (subgraph sink) to build a kernel for — the §3 grammar. Fuel
    /// owns this type; the synthesizer matches against it. This is also the
    /// recipe's `decompose` (the primitive subgraph), so the response need not
    /// send it back — Fuel already holds it.
    pub region: PatternNode,
    /// The live operands' **raw** projection, in the region's bind-index order —
    /// the `shapes`/`dtypes` half of §5.2's `target`. The synthesizer classifies
    /// these via `structure_key`; Fuel never pre-classifies (the ratified
    /// single-classifier division).
    pub operands: Vec<OperandDesc>,
    /// The target SM arch Fuel derived from the device — the `backend`/`device`
    /// half of §5.2's `target` (CUDA is implicit for a CUDA synthesizer).
    pub arch: ArchSku,
    /// The compile-time / resource budget Fuel sets (§5.2). The synthesizer
    /// SHOULD decline rather than exceed it.
    pub budget: JitBudget,
}

/// The compile-time / resource budget the requester allows for synthesis (§5.2).
/// Fuel sets it; the synthesizer treats it as a ceiling and declines
/// ([`JitResponse::Declined`]) rather than overrun. Additive per the §6 change
/// policy — new budget axes go in new fields.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JitBudget {
    /// Wall-clock ceiling (milliseconds) on synthesis + compile.
    pub max_compile_ms: u32,
}

/// The synthesizer's reply to a [`JitRequest`] (§5.2).
#[derive(Clone, Debug, PartialEq)]
pub enum JitResponse {
    /// A synthesized kernel + its full FKC contract.
    Synthesized(SynthesizedKernel),
    /// Synthesis declined — unsupported region op, not beneficial, over budget,
    /// or an arch the synthesizer can't target. Carries a human reason for
    /// telemetry; Fuel leaves the region on primitives. Never an error path.
    Declined { reason: String },
}

/// A synthesized kernel + the FKC contract that wires it into Fuel (§5.2).
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesizedKernel {
    /// The link-registry entry-point symbol the synthesized `KernelRef` is bound
    /// under; the binary/PTX lives behind it, resolved via the provider's
    /// `link_registry` (FKC §12.6). The synthesizer must have registered the
    /// binding before returning.
    pub entry_point: String,
    /// The **full FKC contract** for the synthesized kernel
    /// (accept / return / op_params / cost / precision / determinism + the
    /// re-fuse `pattern:`), as the FKC-contract markdown Fuel's importer already
    /// parses (`docs/specs/kernel-contract-format.md`). Adoption imports it
    /// exactly like a build-time FKC contract — so the JIT path reuses the FKC
    /// importer rather than a bespoke parser. The recipe's `decompose` is the
    /// [`JitRequest::region`] Fuel already holds; the `pattern:` rides in here.
    pub contract: String,
}

/// The interface Fuel calls to synthesize a kernel. A JIT-capable backend
/// registers an impl; Fuel invokes it through this trait, staying decoupled from
/// any concrete synthesizer crate. Direct-Rust transport for Profile v1 (the
/// handshake stays C-ABI; the request/response cross in-process).
pub trait Synthesizer: Send + Sync {
    /// Synthesize a kernel for `req.region` within `req.budget`, or decline.
    /// Never panics — a region the synthesizer can't build (or can't build in
    /// budget) is a [`JitResponse::Declined`], not an error.
    fn synthesize(&self, req: &JitRequest) -> JitResponse;
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernels_types::ElementKind;
    use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};

    fn relu_add() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    /// A stand-in synthesizer (the role Baracuda's live impl fills): bind a named
    /// entry point + return a minimal FKC contract for the region.
    struct EchoSynth;
    impl Synthesizer for EchoSynth {
        fn synthesize(&self, req: &JitRequest) -> JitResponse {
            if req.operands.is_empty() {
                return JitResponse::Declined { reason: "no operands".into() };
            }
            JitResponse::Synthesized(SynthesizedKernel {
                entry_point: "jit::echo::relu_add".into(),
                // A stand-in for the real FKC-contract markdown Baracuda emits.
                contract: "## fused_op: jit::echo::relu_add\ncost: n\n".into(),
            })
        }
    }

    #[test]
    fn synthesizer_round_trips_a_region_through_the_envelope() {
        let req = JitRequest {
            region: relu_add(),
            operands: vec![
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            ],
            arch: ArchSku::Sm89,
            budget: JitBudget { max_compile_ms: 250 },
        };
        match EchoSynth.synthesize(&req) {
            JitResponse::Synthesized(k) => {
                assert_eq!(k.entry_point, "jit::echo::relu_add");
                assert!(k.contract.contains("cost: n"), "carries the FKC contract markdown");
            }
            JitResponse::Declined { reason } => panic!("expected synthesis, declined: {reason}"),
        }
    }

    #[test]
    fn synthesizer_can_decline() {
        let req = JitRequest {
            region: relu_add(),
            operands: vec![],
            arch: ArchSku::Sm80,
            budget: JitBudget { max_compile_ms: 250 },
        };
        assert!(matches!(EchoSynth.synthesize(&req), JitResponse::Declined { .. }));
    }
}

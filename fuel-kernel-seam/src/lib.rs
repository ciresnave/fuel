//! Kernel-seam JIT-on-request envelope (kernel-seam-interop §5) — the live-call
//! protocol between Fuel (the strategist) and a JIT kernel synthesizer (e.g.
//! Baracuda). Fuel chooses a region + cost-gates adoption; the synthesizer
//! builds a kernel for that region or declines.
//!
//! This crate is the wire shape both halves import: a [`JitRequest`] carries the
//! region (Fuel's `PatternNode` grammar) + the live operands' raw projection
//! (the synthesizer's `OperandDesc`, which it classifies via `structure_key`) +
//! the target arch; a [`JitResponse`] carries either a [`SynthesizedKernel`] (its
//! entry-point binding + FKC re-fuse contract) or a decline. [`Synthesizer`] is
//! the trait Fuel calls — a backend registers an impl; Fuel stays decoupled from
//! any concrete synthesizer crate.
//!
//! It is **light** (no `fuel-graph`): the in-process *adoption* of a
//! `JitResponse` into a runtime fused op (register the sidecar recipe + bind the
//! kernel + auto-wire the declarative fusion rule) lives Fuel-side, in the
//! dispatch layer, not here.

use baracuda_kernels_types::{ArchSku, OperandDesc};
use fuel_kernel_seam_types::PatternNode;

/// A request to synthesize a kernel for a Fuel-chosen region.
#[derive(Clone, Debug)]
pub struct JitRequest {
    /// The region (subgraph sink) to build a kernel for — the §3 grammar. Fuel
    /// owns this type; the synthesizer matches against it.
    pub region: PatternNode,
    /// The live operands' **raw** projection, in the region's bind-index order.
    /// The synthesizer classifies these via `structure_key`; Fuel never
    /// pre-classifies (the ratified single-classifier division).
    pub operands: Vec<OperandDesc>,
    /// The target SM arch Fuel derived from the device.
    pub arch: ArchSku,
}

/// The synthesizer's reply to a [`JitRequest`].
#[derive(Clone, Debug, PartialEq)]
pub enum JitResponse {
    /// A synthesized kernel + its FKC re-fuse contract.
    Synthesized(SynthesizedKernel),
    /// Synthesis declined — unsupported region op, not beneficial, over budget,
    /// or an arch the synthesizer can't target. Carries a human reason for
    /// telemetry; Fuel leaves the region on primitives. Never an error path.
    Declined { reason: String },
}

/// A synthesized kernel + the contract that wires it back into Fuel's optimizer.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesizedKernel {
    /// The FKC link-registry entry-point symbol the synthesized `KernelRef` is
    /// bound under (§3.5). Fuel resolves it to a dispatchable kernel; the
    /// synthesizer must have registered the binding before returning.
    pub entry_point: String,
    /// The FKC `pattern:` re-fuse rule. For a JIT region this is the region
    /// itself; Fuel's matcher (`PatternKind::Declarative`) auto-wires it so the
    /// op fuses on the next optimize pass.
    pub pattern: PatternNode,
    /// The FKC cost-expression text. Fuel parses it (its `cost_expr` core) and
    /// **cost-gates adoption** — registering the op only if the fused estimate
    /// beats the region's primitive-path cost.
    pub cost: String,
}

/// The interface Fuel calls to synthesize a kernel. A JIT-capable backend
/// registers an impl; Fuel invokes it through this trait, staying decoupled from
/// any concrete synthesizer crate. Direct-Rust transport for Profile v1 (the
/// handshake stays C-ABI; the request/response cross in-process).
pub trait Synthesizer: Send + Sync {
    /// Synthesize a kernel for `req.region`, or decline. Never panics — a region
    /// the synthesizer can't build is a [`JitResponse::Declined`], not an error.
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

    /// A stand-in synthesizer (the role Baracuda's live impl fills): echo the
    /// region as the re-fuse pattern, bind a named entry point.
    struct EchoSynth;
    impl Synthesizer for EchoSynth {
        fn synthesize(&self, req: &JitRequest) -> JitResponse {
            if req.operands.is_empty() {
                return JitResponse::Declined { reason: "no operands".into() };
            }
            JitResponse::Synthesized(SynthesizedKernel {
                entry_point: "jit::echo::relu_add".into(),
                pattern: req.region.clone(),
                cost: "n".into(),
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
        };
        match EchoSynth.synthesize(&req) {
            JitResponse::Synthesized(k) => {
                assert_eq!(k.entry_point, "jit::echo::relu_add");
                assert_eq!(k.pattern, relu_add(), "the re-fuse pattern is the region");
            }
            JitResponse::Declined { reason } => panic!("expected synthesis, declined: {reason}"),
        }
    }

    #[test]
    fn synthesizer_can_decline() {
        let req = JitRequest { region: relu_add(), operands: vec![], arch: ArchSku::Sm80 };
        assert!(matches!(EchoSynth.synthesize(&req), JitResponse::Declined { .. }));
    }
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel-seam JIT-on-request envelope (kernel-seam-interop §5.2) — the live-call
//! protocol between Fuel (the strategist) and a JIT kernel synthesizer (e.g.
//! Baracuda). Fuel chooses a region + a synthesis budget and cost-gates adoption;
//! the synthesizer builds a kernel for that region or declines.
//!
//! **Two-step handover** (matches Baracuda's built `Synthesizer` impl):
//! 1. [`Synthesizer::synthesize`] returns a **light** [`JitResponse`] — a
//!    `Synthesized { entry_point }` handle or a `Declined`. No heavy artifact
//!    crosses here, so a kernel Fuel's cost-gate *declines* costs nothing.
//! 2. Once Fuel decides to adopt, [`Synthesizer::take_kernel`] hands over the
//!    retained [`SynthArtifact`] — the compiled artifact + its FKC contract + the
//!    runtime binding row — which Fuel loads, wraps as a `KernelRef`, and adopts.
//!
//! The synthesizer is `Send + Sync` and interior-mutable, so **Fuel owns the
//! concurrency**: drive `synthesize` on a background/idle-time thread (the G7
//! "JIT fusion is a background re-optimization trigger" model), never on the
//! realize hot path. The trait is Fuel's; a backend `impl`s it, so Fuel depends
//! on nothing of the synthesizer's at the type level (`&dyn Synthesizer`).
//!
//! Light by design (no `fuel-graph` / `fuel-dispatch`): the in-process adoption of
//! a `SynthArtifact` — import its FKC contract, load + bind the kernel, register
//! the Tier-2 runtime fused-op — lives Fuel-side, in the dispatch layer, not here.

use baracuda_kernels_types::{ArchSku, OperandDesc, TargetId};
use fuel_kernel_seam_types::PatternNode;

/// A request to synthesize a kernel for a Fuel-chosen region (§5.2).
///
/// `#[non_exhaustive]` (board #106 Step A, added alongside `target`): a
/// synthesizer backend (e.g. Baracuda) must construct this through
/// [`JitRequest::new`], never a struct literal, so a future field add here
/// is a compile-time nudge at every construction site rather than a value
/// silently defaulting to nothing. Field READS (`req.region`, etc.) are
/// unaffected — only construction and exhaustive destructuring are.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct JitRequest {
    /// The region (subgraph sink) to build a kernel for — the §3 grammar. Fuel
    /// owns this type; the synthesizer matches against it. Also the recipe's
    /// `decompose` (the primitive subgraph), so the artifact need not send it back.
    pub region: PatternNode,
    /// The live operands' **raw** projection, in bind-index order — the
    /// `shapes`/`dtypes` half of §5.2's `target`. The synthesizer classifies these
    /// via `structure_key`; Fuel never pre-classifies (single-classifier division).
    pub operands: Vec<OperandDesc>,
    /// The target SM arch Fuel derived from the device — the `backend`/`device`
    /// half of §5.2's `target` (CUDA implicit for a CUDA synthesizer).
    pub arch: ArchSku,
    /// The target, named by `unpopped-vocab`'s interned `namespace:capability-set`
    /// token (board #106 option C) — the vendor-neutral successor to `arch`,
    /// carried alongside it during the staging window rather than replacing it
    /// outright. Derived from `arch` via `TargetId::from` in [`JitRequest::new`]
    /// (infallible for all 4 `ArchSku` variants today).
    pub target: TargetId,
    /// The compile-time / resource budget Fuel sets (§5.2). The synthesizer
    /// SHOULD decline rather than exceed it.
    pub budget: JitBudget,
}

impl JitRequest {
    /// Construct a request, deriving `target` from `arch` so every caller
    /// states the arch once. The only public constructor — `#[non_exhaustive]`
    /// forbids the struct-literal form outside this crate, and this crate's
    /// own call sites use it too, so a future field add touches one
    /// signature instead of every construction site again.
    pub fn new(
        region: PatternNode,
        operands: Vec<OperandDesc>,
        arch: ArchSku,
        budget: JitBudget,
    ) -> Self {
        Self {
            region,
            operands,
            arch,
            target: TargetId::from(arch),
            budget,
        }
    }
}

/// The compile-time / resource budget the requester allows for synthesis (§5.2).
/// Additive per the §6 change policy — new budget axes go in new fields.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JitBudget {
    /// Wall-clock ceiling (milliseconds) on synthesis + compile. Honored today as
    /// a validated budget + typed decline (bounds the synthesizer's optimizer
    /// effort); a hard interrupt of a runaway compile is a future watchdog item.
    pub max_compile_ms: u32,
}

/// The synthesizer's **light** reply to a [`JitRequest`] (§5.2). The heavy
/// artifact is fetched separately via [`Synthesizer::take_kernel`] only if Fuel's
/// cost-gate adopts — a declined kernel transfers nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JitResponse {
    /// A kernel was synthesized and retained under `entry_point`. Fuel cost-gates,
    /// then calls [`Synthesizer::take_kernel`] with this `entry_point` to adopt.
    Synthesized { entry_point: String },
    /// Synthesis declined — unsupported region op, not beneficial, over budget, or
    /// an untargetable arch. Carries a human reason for telemetry; Fuel leaves the
    /// region on primitives. Never an error path.
    Declined { reason: String },
}

/// The heavy artifact for a synthesized kernel, handed over by
/// [`Synthesizer::take_kernel`] **after** Fuel's cost-gate decides to adopt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynthArtifact {
    /// The compiled kernel bytes — Fuel loads them as a module.
    pub artifact: Vec<u8>,
    /// Provenance of `artifact` (drives how Fuel loads it).
    pub kind: ArtifactKind,
    /// The runtime binding row: `entry_point` → the module symbol Fuel resolves,
    /// plus the FKC §12.6 metadata. Fuel resolves the symbol in the loaded module
    /// and wraps it as a `KernelRef`, so the envelope carries no live `KernelRef`.
    pub link: LinkEntry,
    /// The full FKC contract (markdown) — accept / return / op_params / cost /
    /// precision / determinism **and** the re-fuse `pattern:`. Fuel's FKC importer
    /// parses it exactly like a build-time contract, so the JIT path reuses the
    /// importer. The recipe's `decompose` is the [`JitRequest::region`] Fuel holds.
    pub contract: String,
}

/// Provenance of a [`SynthArtifact::artifact`] — always a **loadable** artifact.
/// A synthesizer that can't produce a loadable kernel returns
/// [`JitResponse::Declined`], never a `Synthesized` carrying a non-loadable
/// placeholder (Baracuda, 2026-07-04: a non-loadable/stub synth declines; no
/// unloadable sentinel crosses the seam, so a loader never has to refuse one).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArtifactKind {
    /// PTX (loaded via the driver JIT).
    Ptx,
    /// A pre-linked cubin.
    Cubin,
}

/// The runtime binding row for a synthesized kernel — the JIT analog of one row
/// in the AOT `link_registry` catalog (`(entry_point, structure_key, revision)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkEntry {
    /// The kernel's stable entry-point name (matches [`JitResponse::Synthesized`]).
    pub entry_point: String,
    /// The symbol Fuel resolves in the loaded module to get the callable.
    pub symbol: String,
    /// The FKC `structure_key` token this kernel is specialized for.
    pub structure_key: String,
    /// Kernel revision hash — cache-invalidation / drift detection.
    pub revision_hash: u64,
}

/// The interface Fuel calls to synthesize a kernel. A JIT-capable backend
/// registers an impl; Fuel invokes it through this trait via a `&dyn Synthesizer`,
/// staying decoupled from any concrete synthesizer crate. `Send + Sync` +
/// interior-mutable so Fuel can drive it from a background thread (§Q3).
pub trait Synthesizer: Send + Sync {
    /// Synthesize a kernel for `req.region` within `req.budget`, or decline.
    /// Returns a **light** handle; never panics (an unbuildable / over-budget /
    /// out-of-vocabulary region is a [`JitResponse::Declined`]).
    fn synthesize(&self, req: &JitRequest) -> JitResponse;

    /// Hand over + remove the retained [`SynthArtifact`] for `entry_point`. Called
    /// **after** the cost-gate decides to adopt, so a declined kernel's artifact is
    /// never transferred. `None` if never synthesized / already taken (single adopt).
    fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact>;

    /// Fuel refused a kernel this synthesizer produced (verify-against-decompose
    /// failed, over budget on re-offer, etc.). Default no-op; a synthesizer MAY
    /// override to stop re-offering / to log. `report` is a JSON-ish detail
    /// string (Fuel's RejectionReport rendered) so this crate stays dependency-light.
    fn on_rejected(&self, _entry_point: &str, _report: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernels_types::ElementKind;
    use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode};
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn relu_add() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
            }],
        }
    }

    /// A stand-in synthesizer mirroring Baracuda's built shape: `synthesize`
    /// inserts the artifact under its entry_point; `take_kernel` removes + returns
    /// it. Interior-mutable + `Send + Sync`, so Fuel could drive it off-thread.
    #[derive(Default)]
    struct EchoSynth {
        store: Mutex<HashMap<String, SynthArtifact>>,
    }
    impl Synthesizer for EchoSynth {
        fn synthesize(&self, req: &JitRequest) -> JitResponse {
            if req.operands.is_empty() {
                return JitResponse::Declined {
                    reason: "no operands".into(),
                };
            }
            let entry_point = "jit::echo::relu_add".to_string();
            self.store.lock().unwrap().insert(
                entry_point.clone(),
                SynthArtifact {
                    artifact: vec![0xDE, 0xAD],
                    kind: ArtifactKind::Ptx,
                    link: LinkEntry {
                        entry_point: entry_point.clone(),
                        symbol: "echo_relu_add".into(),
                        structure_key: "elementwise:f32".into(),
                        revision_hash: 0x1234,
                    },
                    contract: "## fused_op: jit::echo::relu_add\ncost: n\n".into(),
                },
            );
            JitResponse::Synthesized { entry_point }
        }
        fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact> {
            self.store.lock().unwrap().remove(entry_point)
        }
    }

    fn req() -> JitRequest {
        JitRequest::new(
            relu_add(),
            vec![
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
                OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256),
            ],
            ArchSku::Sm89,
            JitBudget {
                max_compile_ms: 250,
            },
        )
    }

    /// Board #106 Step A condition: every `ArchSku` variant, not a sample —
    /// an exhaustive `match` so a new variant upstream fails to COMPILE here
    /// rather than silently skipping coverage. `TargetId::from` is asserted
    /// infallible for each (it already panics internally if ever not, per
    /// its own doc) and produces the exact `cuda:sm<N>[a]` token the §5.2
    /// target model expects.
    #[test]
    fn jit_request_new_derives_target_for_every_arch_sku_variant() {
        for arch in [ArchSku::Sm80, ArchSku::Sm89, ArchSku::Sm90, ArchSku::Sm90a] {
            let expected_token = match arch {
                ArchSku::Sm80 => "cuda:sm80",
                ArchSku::Sm89 => "cuda:sm89",
                ArchSku::Sm90 => "cuda:sm90",
                ArchSku::Sm90a => "cuda:sm90a",
            };
            let req = JitRequest::new(relu_add(), vec![], arch, JitBudget { max_compile_ms: 1 });
            assert_eq!(
                req.target.to_string(),
                expected_token,
                "JitRequest::new's TargetId::from(ArchSku::{arch:?}) must match the \
                 §5.2 target model's cuda: token"
            );
        }
    }

    #[test]
    fn synthesize_returns_a_light_handle_then_take_kernel_hands_over_the_artifact() {
        let synth = EchoSynth::default();
        let entry_point = match synth.synthesize(&req()) {
            JitResponse::Synthesized { entry_point } => entry_point,
            JitResponse::Declined { reason } => panic!("expected synthesis, declined: {reason}"),
        };
        assert_eq!(entry_point, "jit::echo::relu_add");
        // The heavy artifact only crosses on adopt, via take_kernel.
        let art = synth
            .take_kernel(&entry_point)
            .expect("artifact retained for adopt");
        assert_eq!(art.kind, ArtifactKind::Ptx);
        assert_eq!(art.link.entry_point, entry_point);
        assert!(
            art.contract.contains("cost: n"),
            "carries the FKC contract markdown"
        );
        // Single-adopt: a second take is None.
        assert!(
            synth.take_kernel(&entry_point).is_none(),
            "take_kernel removes (single adopt)"
        );
    }

    #[test]
    fn synthesizer_can_decline_and_retains_nothing() {
        let synth = EchoSynth::default();
        let mut declined = req();
        declined.operands.clear();
        assert!(matches!(
            synth.synthesize(&declined),
            JitResponse::Declined { .. }
        ));
        assert!(
            synth.take_kernel("jit::echo::relu_add").is_none(),
            "declined → nothing retained"
        );
    }
}

//! Decode flash-arm emission — the safe, gated `Op::Branch` that makes the
//! CUDA `flash_decoding` binding (commit `f693da2c`) reachable in decode.
//!
//! ## Why this module exists
//!
//! Fuel dispatch is **fail-fast** with a key-only, shape-blind executor
//! binding lookup: a registered kernel that returns `Err` fails the whole
//! `realize`, and a wrapper *cannot* soft-decline back to the decomposed
//! base map (the finding recorded in `f693da2c`). The CUDA
//! `(FlashAttn, [f16|bf16;4], Cuda)` capacity-K bindings are live-validated,
//! but only reachable **safely** if a CUDA `FlashAttn` node is *only ever
//! created for shapes the kernel supports*: `seq_q == 1`, `head_dim ∈
//! [1, 128]`, `f16`/`bf16`, no window / ALiBi / softcap. The plan-time
//! infeasible-cost gate (`cost::cost_flash_decoding_cuda`) is defense in
//! depth for unpinned placement; the *guaranteed* clean gate is
//! **graph-construction-time arm emission** — only build the CUDA arm for
//! supported decode shapes. That is this module.
//!
//! ## Constitutional posture (04-optimization / 06-runtime)
//!
//! - **The optimizer emits/prunes arms; backends never decide.** All the
//!   strategy — the shape/dtype/config gate, the capability gate, the
//!   CUDA pin, the `Op::Branch` construction — lives here, in the dispatch
//!   (optimizer) layer, not in the model. The model layer only supplies
//!   the region's tensor handles and the live `k_len` it alone knows (the
//!   attended prefix = `cached_len + seq`), which is **data**, not a
//!   strategic choice.
//! - **The decomposed base map stays arm 0 / the correctness oracle.** The
//!   `Op::Branch`'s arm 0 is the *existing decomposed region* the decode
//!   graph already builds (`matmul → scale → mask → softmax → matmul` over
//!   the capacity KV). `finalize_branches` enforces that the merge reads
//!   arm 0, so a finalized-but-unpicked graph — and every CPU/Vulkan build,
//!   where no CUDA flash arm is ever emitted — realizes on the decomposed
//!   route, byte-identical to today.
//! - **CUDA-only.** The capacity-K prefix (`k_len != sk`) is a CUDA-only
//!   capability: `vulkan_dispatch::flash_attn` hard-rejects `k_len != sk`
//!   (fail-fast → fails realize) and the CPU flash wrapper would silently
//!   change results vs. the decomposed arm. So arm 1 is pinned to
//!   `BackendId::Cuda` and only emitted when a CUDA flash kernel is bound
//!   AND a CUDA device is in the current topology.
//!
//! ## What this module does NOT do (documented remaining slice)
//!
//! Emitting the arm makes the flash kernel **offered**. Whether it is
//! **selected** at dispatch is the runtime route picker's job — and the
//! shipped picker (`ranker::route_picker`) resolves *placement* choices
//! (it ranks arms by their `(backend, device)` keyed on arm 0's op), not a
//! **same-device kernel-variant** choice (flash vs. decomposed, both on
//! CUDA). Per 04-optimization, kernel-variant choice is "largely baked at
//! optimize time"; selecting the flash arm on a same-device basis needs
//! either a per-arm-op cost leg in the picker or an optimize-time route
//! bake. That, plus wiring this emitter into `fuel-core`'s decode builder
//! (with the persistent-decode attended-length symbol), is the remaining
//! integration — see the module's tests + the session report.

use fuel_graph::registry::{FusedOpParams, FusedOps};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DynScalar, Result, Shape};

use crate::runtime_fused_kernels::fused_kernel_available;
use crate::topology::SystemTopology;

/// The capability + topology gate for the CUDA flash decode arm: a CUDA
/// `FlashAttn` kernel must be bound **and** a CUDA device must be present
/// in the current topology. Split out (and injectable) so the gate logic is
/// unit-testable without a CUDA build, while production reads the real
/// process-global registry + topology via [`Self::production`].
#[derive(Clone, Copy, Debug)]
pub struct FlashArmCapability {
    /// A `(FlashAttn, Cuda)` kernel is bound (static or runtime-adopted).
    pub cuda_flash_kernel: bool,
    /// A CUDA device is present in the current [`SystemTopology`].
    pub cuda_in_topology: bool,
}

impl FlashArmCapability {
    /// Read the real process-global capability.
    pub fn production() -> Self {
        Self {
            cuda_flash_kernel: fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda),
            cuda_in_topology: SystemTopology::current()
                .backends()
                .contains(&BackendId::Cuda),
        }
    }

    /// A capability that always admits the arm — for unit tests that drive
    /// the shape/dtype/config gate without a CUDA build.
    #[cfg(test)]
    pub fn all_available() -> Self {
        Self { cuda_flash_kernel: true, cuda_in_topology: true }
    }

    /// The flash arm may be offered only when both halves hold.
    pub fn available(&self) -> bool {
        self.cuda_flash_kernel && self.cuda_in_topology
    }
}

/// The decode-attention region the model hands to the optimizer as a
/// flash-arm **candidate**. Arm 0 — the decomposed region — is already in
/// the graph; this describes how to build arm 1 (the flash kernel) and
/// where to splice the `Op::Branch`.
///
/// `q` is both the flash node's first input and the branch's shared
/// **diverge** point (the decomposed chain and the flash node both depart
/// from `q`). `decomposed_out` is arm 0's exit (the region's materialized
/// output, `[B, Hq, Sq, D]` — same shape/dtype as the flash output).
/// `reconverge` is the sole consumer of `decomposed_out` (the merge).
#[derive(Clone, Debug)]
pub struct DecodeFlashSpec {
    /// `q` — `[B, Hq, Sq, D]`. Also the branch diverge point.
    pub q: NodeId,
    /// `k` — the capacity KV buffer `[B, Hkv, capacity, D]`.
    pub k: NodeId,
    /// `v` — the capacity KV buffer `[B, Hkv, capacity, D]`.
    pub v: NodeId,
    /// Optional ALiBi slopes `[Hq]`. Its presence *disqualifies* the arm
    /// (the capacity-K CUDA kernel has no ALiBi param) — carried so the
    /// gate can reject it explicitly.
    pub alibi: Option<NodeId>,
    /// `1/sqrt(head_dim)` (or the model's scale).
    pub softmax_scale: f32,
    /// Causal masking. Accepted-and-ignored by the kernel (`seq_q == 1`
    /// attends the full `[0, k_len)` prefix), so it does not disqualify.
    pub causal: bool,
    /// Left sliding-window bound — disqualifies (no kernel support).
    pub window_size_left: Option<usize>,
    /// Right sliding-window bound — disqualifies (no kernel support).
    pub window_size_right: Option<usize>,
    /// Logit softcap — disqualifies (no kernel support).
    pub softcap: Option<f32>,
    /// The live attended prefix length (`cached_len + seq`), resolved via
    /// the per-pass `SymEnv` at realize. For a per-step (non-persistent)
    /// decode graph this is `DynScalar::Concrete(cached_len + seq)`; for
    /// persistent (plan-once) decode it is a `Sym` bound to the live
    /// length each token.
    pub k_len: DynScalar,
    /// Arm 0's exit: the decomposed region's output node (the oracle).
    pub decomposed_out: NodeId,
    /// The sole consumer of `decomposed_out` — the branch merge point.
    pub reconverge: NodeId,
}

/// Is a decode-attention region admissible for the CUDA flash arm?
///
/// Rejects (→ `false`, decomposed-only) anything the capacity-K CUDA
/// binding cannot do, matching `cost::cost_flash_decoding_cuda`'s
/// infeasible set and the kernel's `_can_implement` gate:
///
/// - capability unavailable (no CUDA flash kernel / no CUDA device);
/// - dtype not `f16` / `bf16`;
/// - `q` not rank-4, `seq_q != 1`, or `head_dim` `0` or `> 128`;
/// - sliding window (`window_size_{left,right}`), softcap, or ALiBi set.
///
/// `causal` is intentionally NOT a disqualifier: with `seq_q == 1` the
/// single query attends the whole `[0, k_len)` prefix, so the causal mask
/// is a no-op the kernel omits.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_admissible(
    q_shape: &Shape,
    dtype: DType,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
    softcap: Option<f32>,
    has_alibi: bool,
    cap: FlashArmCapability,
) -> bool {
    // Capability: a CUDA FlashAttn kernel bound AND CUDA present.
    if !cap.available() {
        return false;
    }
    // Dtype: the CUDA flash_decoding binding is f16 / bf16 only.
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return false;
    }
    // Shape: decode is seq_q == 1; head_dim in [1, 128].
    let dims = q_shape.dims();
    if dims.len() != 4 {
        return false;
    }
    let sq = dims[2];
    let d = dims[3];
    if sq != 1 || d == 0 || d > 128 {
        return false;
    }
    // Config the capacity-K kernel does not implement.
    if window_size_left.is_some() || window_size_right.is_some() {
        return false;
    }
    if softcap.is_some() {
        return false;
    }
    if has_alibi {
        return false;
    }
    true
}

/// Offer the CUDA flash decode arm for `spec` on `graph`.
///
/// If [`flash_decode_admissible`] holds for `spec`'s `q` shape/dtype +
/// config + `cap`, this builds arm 1 (the flash `Op::Fused(FLASH_ATTN, …)`
/// node, CUDA-pinned) and records a 2-arm `Op::Branch`:
///
/// - **arm 0** = `spec.decomposed_out` (the decomposed oracle — the route
///   an unpicked graph, and every non-CUDA build, realizes on);
/// - **arm 1** = the flash node.
///
/// Returns `Ok(Some(branch))` when the arm was emitted, `Ok(None)` when the
/// gate declined (the graph is left with the decomposed route only —
/// byte-identical to today). Surfaces a build-time `Err` only on a
/// malformed spec that fails `finalize_branches`' validation (never a
/// panic).
///
/// The flash node reads `[q, k, v, (alibi)]` and carries `k_len` in its
/// `FusedOpParams::FlashAttn` — the `flash_attn_dyn → SymEnv → OpParams`
/// path the pipelined lowering already resolves (`pipelined.rs` ~3660).
pub fn offer_decode_flash_arm(
    graph: &mut Graph,
    spec: &DecodeFlashSpec,
    cap: FlashArmCapability,
) -> Result<Option<NodeId>> {
    let (q_shape, q_dtype) = {
        let n = graph.node(spec.q);
        (n.shape.clone(), n.dtype)
    };

    if !flash_decode_admissible(
        &q_shape,
        q_dtype,
        spec.window_size_left,
        spec.window_size_right,
        spec.softcap,
        spec.alibi.is_some(),
        cap,
    ) {
        return Ok(None);
    }

    // Arm 1: the flash Op::Fused(FLASH_ATTN, { k_len }) node. Shape + dtype
    // MUST equal arm 0 (the decomposed output) — both are q's shape/dtype —
    // or finalize_branches' cast-to-uniform rule rejects the branch.
    let mut inputs = vec![spec.q, spec.k, spec.v];
    if let Some(a) = spec.alibi {
        inputs.push(a);
    }
    let flash = graph.push(Node {
        op: Op::Fused(
            FusedOps::FLASH_ATTN,
            FusedOpParams::FlashAttn {
                softmax_scale: spec.softmax_scale,
                causal: spec.causal,
                window_size_left: spec.window_size_left,
                window_size_right: spec.window_size_right,
                softcap: spec.softcap,
                k_len: Some(spec.k_len),
            },
        ),
        inputs,
        shape: q_shape,
        dtype: q_dtype,
    });
    // Pin arm 1 to CUDA — the only backend whose FlashAttn binding handles a
    // capacity-K prefix (Vulkan hard-rejects k_len != sk; the CPU flash
    // wrapper would silently change results vs. the decomposed arm).
    graph.set_target_backend(flash, BackendId::Cuda);

    // Emit the branch: arm 0 = decomposed oracle, arm 1 = flash. The diverge
    // is `q` (both arms depart from it); the merge is `reconverge`, which
    // reads arm 0 (the runnability invariant `finalize_branches` enforces).
    let mut builder = graph.open_branch(spec.q);
    builder.add_arm(spec.decomposed_out); // arm 0
    builder.add_arm(flash); // arm 1
    let branch = builder.finalize_branches(graph, spec.reconverge)?;
    Ok(branch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a Const node of the given shape/dtype (a stand-in leaf tensor).
    fn leaf(g: &mut Graph, dims: &[usize], dtype: DType) -> NodeId {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype,
        })
    }

    /// Build a synthetic decode-attention region:
    ///   q [1,H,1,D], k/v [1,H,SK,D]  (dtype)
    ///   scores  = MatMul(q, Permute(k))           [1,H,1,SK]
    ///   scaled  = MulScalar(scale)(scores)
    ///   masked  = Add(scaled, mask[1,H,1,SK])
    ///   probs   = Fused(SOFTMAX_LAST_DIM)(masked)
    ///   attn_v  = MatMul(probs, v)                [1,H,1,D]   (arm-0 exit)
    ///   merged  = Permute(attn_v)                 (reconverge / sole consumer)
    /// Returns `(graph, spec-parts)`: (q, k, v, attn_v, merged).
    fn decode_region(
        h: usize,
        d: usize,
        sk: usize,
        dtype: DType,
    ) -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let mut g = Graph::new();
        let q = leaf(&mut g, &[1, h, 1, d], dtype);
        let k = leaf(&mut g, &[1, h, sk, d], dtype);
        let v = leaf(&mut g, &[1, h, sk, d], dtype);
        let mask = leaf(&mut g, &[1, h, 1, sk], dtype);

        let kt = g.push(Node {
            op: Op::Permute(vec![0, 1, 3, 2]),
            inputs: vec![k],
            shape: Shape::from_dims(&[1, h, d, sk]),
            dtype,
        });
        let scores = g.push(Node {
            op: Op::MatMul,
            inputs: vec![q, kt],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype,
        });
        let scaled = g.push(Node {
            op: Op::MulScalar(0.5),
            inputs: vec![scores],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype,
        });
        let masked = g.push(Node {
            op: Op::Add,
            inputs: vec![scaled, mask],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype,
        });
        let probs = g.push(Node {
            op: Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim),
            inputs: vec![masked],
            shape: Shape::from_dims(&[1, h, 1, sk]),
            dtype,
        });
        let attn_v = g.push(Node {
            op: Op::MatMul,
            inputs: vec![probs, v],
            shape: Shape::from_dims(&[1, h, 1, d]),
            dtype,
        });
        // reconverge: the sole consumer of attn_v.
        let merged = g.push(Node {
            op: Op::Permute(vec![0, 2, 1, 3]),
            inputs: vec![attn_v],
            shape: Shape::from_dims(&[1, 1, h, d]),
            dtype,
        });
        (g, q, k, v, attn_v, merged)
    }

    fn spec_for(
        q: NodeId,
        k: NodeId,
        v: NodeId,
        attn_v: NodeId,
        merged: NodeId,
        sk: usize,
    ) -> DecodeFlashSpec {
        DecodeFlashSpec {
            q,
            k,
            v,
            alibi: None,
            softmax_scale: 0.5,
            causal: true,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
            k_len: DynScalar::Concrete(sk),
            decomposed_out: attn_v,
            reconverge: merged,
        }
    }

    /// GREEN target: a supported decode region (f16, seq_q==1, head_dim<=64,
    /// capacity KV) on a CUDA-capable topology gets the flash arm offered —
    /// arm 0 stays the decomposed oracle, arm 1 is a CUDA-pinned
    /// Fused(FLASH_ATTN, { k_len: Some }).
    #[test]
    fn flash_arm_offered_for_supported_decode_shape() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F16);
        let spec = spec_for(q, k, v, attn_v, merged, 37);

        let before = g.len();
        let branch = offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
            .expect("well-formed spec")
            .expect("supported shape ⇒ arm offered");

        // A Branch was recorded.
        assert!(matches!(g.node(branch).op, Op::Branch { .. }), "branch node emitted");
        let arms = g.node(branch).inputs.clone();
        assert_eq!(arms.len(), 2, "2-arm branch (decomposed + flash)");
        // arm 0 = the decomposed oracle.
        assert_eq!(arms[0], attn_v, "arm 0 is the decomposed region output (the oracle)");
        // arm 1 = a CUDA-pinned Fused(FLASH_ATTN, { k_len: Some }).
        let flash = arms[1];
        match &g.node(flash).op {
            Op::Fused(fid, FusedOpParams::FlashAttn { k_len, .. }) => {
                assert_eq!(*fid, FusedOps::FLASH_ATTN, "arm 1 is FLASH_ATTN");
                assert_eq!(*k_len, Some(DynScalar::Concrete(37)), "arm 1 carries the live k_len");
            }
            other => panic!("arm 1 must be Fused(FLASH_ATTN, FlashAttn), got {other:?}"),
        }
        assert_eq!(g.node(flash).inputs, vec![q, k, v], "flash reads q, k, v");
        assert_eq!(g.target_backend(flash), Some(BackendId::Cuda), "arm 1 pinned to CUDA");
        // Arm-0 runnability: the merge still reads the decomposed output.
        assert!(
            g.node(merged).inputs.contains(&attn_v),
            "reconverge reads arm 0 (an unpicked graph realizes decomposed)",
        );
        assert!(g.len() > before, "the flash node + branch were appended");
    }

    /// GUARD: f32 dtype ⇒ no arm (the CUDA binding is f16/bf16 only) ⇒
    /// decomposed-only, graph untouched.
    #[test]
    fn f32_dtype_gets_no_flash_arm() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F32);
        let spec = spec_for(q, k, v, attn_v, merged, 37);
        let before = g.len();
        let branch = offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
            .expect("well-formed spec");
        assert!(branch.is_none(), "f32 ⇒ no flash arm");
        assert_eq!(g.len(), before, "graph untouched (no flash node, no branch)");
    }

    /// GUARD: head_dim > 128 ⇒ no arm (outside the kernel's `_can_implement`).
    #[test]
    fn head_dim_over_128_gets_no_flash_arm() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(2, 256, 8, DType::F16);
        let spec = spec_for(q, k, v, attn_v, merged, 8);
        let before = g.len();
        let branch = offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
            .expect("well-formed spec");
        assert!(branch.is_none(), "head_dim 256 > 128 ⇒ no flash arm");
        assert_eq!(g.len(), before, "graph untouched");
    }

    /// GUARD: seq_q != 1 (prefill-shaped) ⇒ no arm. The capacity-K decode
    /// kernel is `seq_q == 1` only.
    #[test]
    fn multi_query_gets_no_flash_arm() {
        // Build a region with seq_q = 8.
        let (mut g, _q, _k, _v, _av, _m) = decode_region(4, 64, 37, DType::F16);
        // Re-make q with seq_q = 8 and rewire a fresh minimal region.
        let dtype = DType::F16;
        let q = leaf(&mut g, &[1, 4, 8, 64], dtype);
        let k = leaf(&mut g, &[1, 4, 37, 64], dtype);
        let v = leaf(&mut g, &[1, 4, 37, 64], dtype);
        let attn_v = g.push(Node {
            op: Op::MatMul,
            inputs: vec![q, v],
            shape: Shape::from_dims(&[1, 4, 8, 64]),
            dtype,
        });
        let merged = g.push(Node {
            op: Op::Permute(vec![0, 2, 1, 3]),
            inputs: vec![attn_v],
            shape: Shape::from_dims(&[1, 8, 4, 64]),
            dtype,
        });
        let spec = spec_for(q, k, v, attn_v, merged, 37);
        let branch = offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
            .expect("well-formed spec");
        assert!(branch.is_none(), "seq_q = 8 != 1 ⇒ no flash arm");
    }

    /// GUARD: capability unavailable (no CUDA flash kernel / no CUDA device)
    /// ⇒ no arm — a CPU/Vulkan build stays byte-identical to today.
    #[test]
    fn no_cuda_capability_gets_no_flash_arm() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F16);
        let spec = spec_for(q, k, v, attn_v, merged, 37);
        let before = g.len();

        // No kernel bound.
        let no_kernel = FlashArmCapability { cuda_flash_kernel: false, cuda_in_topology: true };
        assert!(offer_decode_flash_arm(&mut g, &spec, no_kernel).unwrap().is_none());
        // No CUDA device present.
        let no_device = FlashArmCapability { cuda_flash_kernel: true, cuda_in_topology: false };
        assert!(offer_decode_flash_arm(&mut g, &spec, no_device).unwrap().is_none());

        assert_eq!(g.len(), before, "no capability ⇒ decomposed-only, graph untouched");
    }

    /// GUARD: window / softcap / ALiBi each disqualify (no kernel support).
    #[test]
    fn window_softcap_alibi_get_no_flash_arm() {
        for mutate in [
            |s: &mut DecodeFlashSpec| s.window_size_left = Some(64),
            |s: &mut DecodeFlashSpec| s.window_size_right = Some(64),
            |s: &mut DecodeFlashSpec| s.softcap = Some(30.0),
        ] {
            let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F16);
            let mut spec = spec_for(q, k, v, attn_v, merged, 37);
            mutate(&mut spec);
            assert!(
                offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
                    .unwrap()
                    .is_none(),
                "window/softcap disqualifies the flash arm",
            );
        }
        // ALiBi: add a slopes leaf and point the spec at it.
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F16);
        let alibi = leaf(&mut g, &[4], DType::F16);
        let mut spec = spec_for(q, k, v, attn_v, merged, 37);
        spec.alibi = Some(alibi);
        assert!(
            offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
                .unwrap()
                .is_none(),
            "ALiBi disqualifies the flash arm",
        );
    }

    /// `causal` does NOT disqualify (seq_q == 1 attends the whole prefix, so
    /// the causal mask is a no-op the kernel omits).
    #[test]
    fn causal_flag_does_not_disqualify() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::F16);
        let mut spec = spec_for(q, k, v, attn_v, merged, 37);
        spec.causal = true;
        assert!(
            offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
                .unwrap()
                .is_some(),
            "causal is accepted-and-ignored ⇒ arm still offered",
        );
    }

    /// bf16 is supported alongside f16.
    #[test]
    fn bf16_dtype_gets_flash_arm() {
        let (mut g, q, k, v, attn_v, merged) = decode_region(4, 64, 37, DType::BF16);
        let spec = spec_for(q, k, v, attn_v, merged, 37);
        assert!(
            offer_decode_flash_arm(&mut g, &spec, FlashArmCapability::all_available())
                .unwrap()
                .is_some(),
            "bf16 is a supported decode dtype",
        );
    }
}

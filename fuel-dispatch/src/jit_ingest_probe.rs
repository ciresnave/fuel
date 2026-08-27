// SPDX-License-Identifier: MIT OR Apache-2.0
//! Candidate-kernel ingestion (Spec B), Task 3 — probe-input synthesis.
//!
//! [`probe_from_operands`] builds deterministic, sized float-fill inputs for
//! a candidate kernel's [`OperandDesc`] list, so Task 5's `verify_candidate`
//! has something real to invoke the kernel with before ever seeing live
//! graph data. Reuses [`crate::jit_adopt`]'s `element_kind_to_dtype` (Baracuda
//! `ElementKind` → Fuel `DType`) and [`crate::fkc::verify`]'s
//! `fill_deterministic` + `to_bytes` (deterministic float fill → dtype-aware
//! byte encode) rather than duplicating either — this file adds only the
//! per-operand sizing/wiring between them.
//!
//! Available under `--features jit` (no `cuda` required): unlike
//! `reference_output` (Task 4, added to this same file next), which needs a
//! live CUDA device to produce a reference, synthesizing sized deterministic
//! inputs is pure host-side arithmetic.

use baracuda_kernels_types::OperandDesc;

use crate::fkc::verify::{HostTensor, fill_deterministic, to_bytes};
use crate::jit_adopt::element_kind_to_dtype;

use crate::jit_ingest::VerifyVerdict;
#[cfg(feature = "cuda")]
use crate::pipelined::{PipelinedExecutor, StorageCache};
#[cfg(feature = "cuda")]
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
#[cfg(feature = "cuda")]
use fuel_graph::jit::PatternNode;
#[cfg(feature = "cuda")]
use fuel_graph::opt::lower_to_base_map;
#[cfg(feature = "cuda")]
use fuel_graph::registry::{FusedOpId, FusedOpParams};
#[cfg(feature = "cuda")]
use fuel_graph::runtime_fused::emit_region;
#[cfg(feature = "cuda")]
use fuel_graph::topo_order_multi;
#[cfg(feature = "cuda")]
use fuel_graph::{Graph, Node, NodeId, Op};
#[cfg(feature = "cuda")]
use fuel_ir::probe::BackendId;
#[cfg(feature = "cuda")]
use fuel_ir::{DType, Error, Result, Shape};
#[cfg(feature = "cuda")]
use std::sync::{Arc, RwLock};

/// Build one deterministic float-fill [`HostTensor`] per `operands` entry,
/// sized from that operand's `rank`/`shape` (extent = product of
/// `shape[..rank]`). Each tensor's values come from
/// `fill_deterministic(extent, seed ^ i)` (`i` = the operand's index, so
/// same-shape operands still get distinct fills) encoded via `to_bytes` for
/// the operand's dtype.
///
/// Returns `None` if any operand's dtype doesn't map to a Fuel `DType`
/// (`element_kind_to_dtype`) or isn't encodable as bytes (`to_bytes`) —
/// never fabricates a probe for an operand it can't faithfully represent.
///
/// Deterministic: the same `(operands, seed)` always produces byte-identical
/// output, so a caller (Task 5's `verify_candidate`) can re-run the probe
/// and expect the same input bytes every time.
/// Every probe the candidate-admission path runs a candidate and Fuel's
/// reference against.
///
/// **This exists so the admission path has ONE definition of "the inputs we
/// looked at".** Before it, the answer was a single seeded fill and the
/// question was never asked in one place — which is how GAP-236's hole stayed
/// invisible: nothing named the set, so nothing could observe that the set
/// could not reach the divergence.
///
/// Step A introduces it holding exactly what the path ran before, so the
/// born-red is measured against the FINAL subject rather than against a
/// stand-in that gets swapped later.
///
/// ⚠️⚠️ **THE HOLE IS NOT CLOSED YET AND THIS FUNCTION IS NOT YET CONSUMED.** GAP(GAP-236)
/// `verify_candidate_impl` (`jit_ingest.rs`, the `// (1) Probe synthesis`
/// block) still calls [`probe_from_operands`] directly, so production
/// admission still runs on the seeded fill alone. That call site is
/// `#[cfg(feature = "cuda")]`, and changing it cannot be compile-verified
/// without a full baracuda forge (~56 min through `scripts/cuda-build.ps1`),
/// which is why it is a separate increment rather than two lines appended
/// here.
///
/// **Stated loudly because a `pub fn` that nothing calls emits no warning and
/// a landed generator reads exactly like a landed fix.** What is proven today
/// is that the probe SET can reach the divergence; what is NOT proven is that
/// the gate looks at it. `gap_236_the_probe_set_is_actually_consumed` below
/// is the guard that will fail until the wiring lands.
pub fn admission_probes(operands: &[OperandDesc], seed: u64) -> Vec<Vec<HostTensor>> {
    probe_from_operands(operands, seed)
        .into_iter()
        .chain(special_value_probe_from_operands(operands))
        .collect()
}

/// Combine the per-probe verdicts of one candidate into its overall verdict.
///
/// **This is the half of the admission loop that is backend-neutral, and it is
/// here so it can be TESTED WITHOUT A GPU.** The per-probe body genuinely
/// cannot move: it takes `&CudaDevice` and performs a real CUDA invoke to
/// produce both the candidate output and Fuel's realized reference. What does
/// not need a device is the POLICY — how N per-probe outcomes become one
/// answer — and that policy is where a probe set can be silently wasted.
///
/// Policy, and each clause exists because its opposite is a real failure mode:
///
/// * **Any `Fail` fails the candidate**, and the detail NAMES the probe index.
///   A candidate that is faithful on the seeded fill and unfaithful on the
///   special values must not be admitted, and the reader must be able to see
///   WHICH input class rejected it — `fmaxf` lifted as `Max` fails only on
///   probe 1, and a detail that omits that reads as a flaky numeric miss.
/// * **Otherwise any `Inconclusive` is inconclusive**, never `Pass`. Escalation
///   outranks silence.
/// * **An EMPTY set is a `Fail`, never a `Pass`.** Nothing was verified, and a
///   verdict of `Pass` over zero evidence is precisely the shape this whole
///   program exists to eliminate.
///
/// It deliberately does NOT short-circuit on the first failure: every probe is
/// examined so the detail can report the full picture, and so that a future
/// caller cannot make "we only ran probe 0" invisible.
pub fn combine_probe_verdicts(per_probe: Vec<VerifyVerdict>) -> VerifyVerdict {
    if per_probe.is_empty() {
        return VerifyVerdict::Fail {
            claim: "probe",
            detail: "no probes were run, so no claim was verified — an empty probe set \
                     must never read as a pass"
                .to_string(),
        };
    }

    let mut failures: Vec<String> = Vec::new();
    let mut inconclusive: Option<(&'static str, String)> = None;
    for (i, v) in per_probe.iter().enumerate() {
        match v {
            VerifyVerdict::Pass => {}
            VerifyVerdict::Fail { claim, detail } => {
                failures.push(format!("probe {i} ({claim}): {detail}"));
            }
            VerifyVerdict::Inconclusive { claim, detail } => {
                if inconclusive.is_none() {
                    inconclusive = Some((claim, detail.clone()));
                }
            }
        }
    }

    if !failures.is_empty() {
        return VerifyVerdict::Fail {
            claim: "probe_set",
            detail: format!(
                "{} of {} probes failed: {}",
                failures.len(),
                per_probe.len(),
                failures.join("; ")
            ),
        };
    }
    if let Some((claim, detail)) = inconclusive {
        return VerifyVerdict::Inconclusive { claim, detail };
    }
    VerifyVerdict::Pass
}

/// The eight special float values the admission probe must reach.
///
/// **NaN is first because it is the one that matters**: `ops.md` specifies
/// NaN-propagating against NaN-suppressing as four distinct ops, and
/// `max_prop`'s primitive expansion puts the ENTIRE difference into two
/// `cmp_ne(x, x)` tests. No value in `[-0.5, 0.5)` can make that true, so
/// without NaN the four ops are one op as far as the gate can see.
///
/// The rest are the classic discriminators: both infinities, both signed
/// zeros (`-0.0` distinguishes `min`/`max` variants that `+0.0` cannot), the
/// smallest positive subnormal, and the multiplicative identity with its
/// negation.
///
/// ⚠️ **THE NaN PAYLOAD IS DELIBERATELY UNSPECIFIED AND MUST STAY THAT WAY.**
/// kiss-ref hit exactly this upstream: an assertion that pins a NaN payload is
/// STRONGER THAN THE PROPERTY and false-reds a conformant implementation that
/// produces a different payload. Verified safe on Fuel's side rather than
/// assumed — `ulp.rs` scores `x.is_nan && y.is_nan` as distance **0**, so
/// both-NaN is agreement regardless of payload, while a NaN on exactly one
/// side is `u64::MAX` and fails every bound. That asymmetry is precisely what
/// catches `fmaxf` lifted as `Max`.
const SPECIAL_F32: [f32; 8] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.0,
    -0.0,
    1.0,
    -1.0,
    // Smallest positive subnormal (1e-45); `MIN_POSITIVE` is the smallest
    // NORMAL and would not exercise the denormal path at all.
    f32::from_bits(1),
];

/// A second admission probe built from [`SPECIAL_F32`] instead of a uniform
/// fill, so the candidate and Fuel's reference are compared somewhere their
/// semantics can actually differ.
///
/// **Operands are filled as digits of a base-8 counter** — operand `i` at
/// element `k` takes `SPECIAL_F32[(k / 8^i) % 8]` — which yields the full
/// ordered CROSS PRODUCT rather than a diagonal. That distinction is
/// load-bearing: a diagonal fill (`table[(k + i) % 8]`) never produces
/// `(NaN, NaN)` or both orderings of `(NaN, finite)`, and `fmaxf` vs `Max`
/// differs on exactly those. An operand with fewer than `8^(i+1)` elements
/// gets a truncated product, which is partial coverage rather than none.
///
/// **Non-float operands fall back to the deterministic fill.** A special value
/// has no meaning for an integer operand, and `to_bytes` would silently map
/// `NaN` to `0` — a fabricated input wearing a special value's name.
///
/// Returns `None` under the same contract as [`probe_from_operands`]: any
/// operand this cannot faithfully represent yields no probe at all.
pub fn special_value_probe_from_operands(operands: &[OperandDesc]) -> Option<Vec<HostTensor>> {
    operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            let rank = operand.rank as usize;
            let shape: Vec<usize> = operand.shape[..rank].iter().map(|&d| d as usize).collect();
            let extent: usize = shape.iter().product();
            let dtype = element_kind_to_dtype(operand.dtype)?;
            let is_float = matches!(
                dtype,
                fuel_ir::DType::F32
                    | fuel_ir::DType::F64
                    | fuel_ir::DType::F16
                    | fuel_ir::DType::BF16
            );
            let vals: Vec<f32> = if is_float {
                // Base-8 digit `i` of the element index: the full ordered
                // cross product across operands, not a diagonal.
                let stride = 8usize.saturating_pow(i as u32).max(1);
                (0..extent)
                    .map(|k| SPECIAL_F32[(k / stride) % SPECIAL_F32.len()])
                    .collect()
            } else {
                fill_deterministic(extent, 0x5EED_5EC0 ^ (i as u64))
            };
            let bytes = to_bytes(dtype, &vals)?;
            Some(HostTensor {
                dtype,
                shape,
                bytes,
            })
        })
        .collect()
}

pub fn probe_from_operands(operands: &[OperandDesc], seed: u64) -> Option<Vec<HostTensor>> {
    operands
        .iter()
        .enumerate()
        .map(|(i, operand)| {
            let rank = operand.rank as usize;
            let shape: Vec<usize> = operand.shape[..rank].iter().map(|&d| d as usize).collect();
            let extent: usize = shape.iter().product();
            let dtype = element_kind_to_dtype(operand.dtype)?;
            let vals = fill_deterministic(extent, seed ^ (i as u64));
            let bytes = to_bytes(dtype, &vals)?;
            Some(HostTensor {
                dtype,
                shape,
                bytes,
            })
        })
        .collect()
}

/// Realize a candidate op's `decompose` region on the probe consts (GPU
/// primitives) and read the output bytes back to host — the **verification
/// reference** Task 5's `verify_candidate` compares a candidate kernel's
/// output against.
///
/// `decompose` is the fused op's primitive recipe as a raw [`PatternNode`]; its
/// `Bind { index }` leaves are filled, in order, by the `probe` tensors (so
/// `probe.len()` must cover every bind index the region references). Each probe
/// is uploaded H2D into fresh CUDA storage (mirroring
/// `crate::fkc::verify::invoker_cuda`), a fresh graph is built with one
/// `Op::Const` leaf per probe, the region is re-emitted onto those leaves via
/// [`fuel_graph::runtime_fused::emit_region`], every emitted primitive is
/// stamped `BackendId::Cuda`, and the sink is realized through
/// [`PipelinedExecutor::realize`]. The output storage is read back D2H into a
/// [`HostTensor`] carrying the caller-declared `out_dtype`/`out_shape`.
///
/// `scalars` for the region's open slots are empty here: a parameterless
/// (elementwise) decompose carries none. A region that does extract scalars
/// would receive them from the candidate at a higher layer.
///
/// Never panics on the production path — every device/realize/readback failure
/// is returned as `Err`. (Two panic risks live inside `emit_region`: a
/// non-re-emittable `OpTag` — a *validated* decompose never carries one — and
/// its scalar-cursor fill, `scalars.split_at(arity)`, which panics if the
/// passed `scalars` slice is shorter than the region's open-slot count;
/// `decompose_region` guards that length elsewhere, but `emit_region` is a
/// thin wrapper and deliberately does not, so the caller here passes `&[]`
/// only for a parameterless region — see the `scalars` doc above. Either way,
/// Task 5's verifier wraps the whole `reference_output` call in
/// `catch_unwind`.)
#[cfg(feature = "cuda")]
pub fn reference_output(
    decompose: &PatternNode,
    probe: &[HostTensor],
    out_dtype: DType,
    out_shape: Vec<usize>,
    device: &CudaDevice,
) -> Result<HostTensor> {
    // (a) H2D: upload every probe into fresh CUDA-resident storage.
    let mut storages: Vec<Arc<RwLock<fuel_memory::Storage>>> = Vec::with_capacity(probe.len());
    for t in probe {
        let cb = CudaStorageBytes::from_cpu_bytes(device, &t.bytes)?;
        storages.push(Arc::new(RwLock::new(fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cuda(cb),
            t.dtype,
        ))));
    }

    // (b)-(d) Build the reference graph: one Const leaf per probe (ids
    // `0..n_inputs`), re-emit the region onto them (emitted primitives take
    // ids `n_inputs..=sink`), and stamp CUDA on every emitted kernel node.
    let graph = Arc::new(RwLock::new(Graph::new()));
    let (input_ids, sink) = {
        let mut g = graph
            .write()
            .map_err(|_| Error::Msg("reference_output: graph RwLock poisoned".to_string()))?;
        let input_ids: Vec<NodeId> = probe
            .iter()
            .map(|t| {
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&t.shape),
                    dtype: t.dtype,
                })
            })
            .collect();
        let n_inputs = input_ids.len();
        let sink = emit_region(&mut g, decompose, &input_ids, &[]);
        // Input Consts are adopted from the StorageCache (no kernel); only the
        // emitted primitives `[n_inputs, sink]` need a target backend (the
        // realize precondition), matching the CPU template's single-node stamp.
        for id in n_inputs..=sink.0 {
            g.set_target_backend(NodeId(id), BackendId::Cuda);
        }
        (input_ids, sink)
    };

    // (e) Bind each probe storage to its Const node id.
    let mut cache = StorageCache::new();
    for (id, storage) in input_ids.iter().zip(storages) {
        cache.insert(*id, storage);
    }

    // (f) Realize the region sink on the device.
    let (out_arc, _layout) = PipelinedExecutor::realize(graph, sink, cache)?;

    // (g) D2H: read the CUDA output storage back to host bytes.
    let bytes = {
        let guard = out_arc.read().map_err(|_| {
            Error::Msg("reference_output: output storage RwLock poisoned".to_string())
        })?;
        match &guard.inner {
            fuel_memory::BackendStorage::Cuda(c) => c.to_cpu_bytes()?,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Msg(
                    "reference_output: realized output storage is not CUDA".to_string(),
                ));
            }
        }
    };

    Ok(HostTensor {
        dtype: out_dtype,
        shape: out_shape,
        bytes,
    })
}

/// Realize FUEL's **registered recipe** for `claimed_op` — its primitive base
/// map — on the probe consts and read the output bytes back to host: the
/// **oracle-independent** verification reference.
///
/// Where [`reference_output`] realizes a *candidate's own* `decompose`, this
/// realizes the recipe Fuel has registered for `claimed_op`, lowered to
/// primitives via [`fuel_graph::opt::lower_to_base_map`]. A candidate is thus
/// checked against what Fuel says the op computes, not against its own
/// (possibly wrong) decompose.
///
/// Mechanics: build `Op::Fused(claimed_op, params)` on one `Op::Const` leaf per
/// `probe` (ids `0..probe.len()`, bound in order); `lower_to_base_map` dissolves
/// the fused sink IN PLACE to its primitive base map (recursive fixpoint — for
/// ROPE this is Reshape/BroadcastTo/Slice/Neg/Concat/Mul/Add); stamp
/// `BackendId::Cuda` on every reachable NON-`Const` node of the lowered map;
/// bind each probe's H2D storage to its `Op::Const` id in the [`StorageCache`];
/// realize the lowered sink through [`PipelinedExecutor::realize`]; read the
/// CUDA output D2H into a [`HostTensor`] carrying the caller-declared
/// `out_dtype`/`out_shape`.
///
/// Never panics on the production path — every lock/H2D/lower/realize/D2H
/// failure is returned as `Err`. (`lower_to_base_map` is itself never-panic: a
/// self-returning `decompose` is a clean fixpoint, not a loop.)
#[cfg(feature = "cuda")]
pub fn reference_from_registered_recipe(
    claimed_op: FusedOpId,
    params: &FusedOpParams,
    probe: &[HostTensor],
    out_dtype: DType,
    out_shape: Vec<usize>,
    device: &CudaDevice,
) -> Result<HostTensor> {
    // (a) H2D: upload every probe into fresh CUDA-resident storage.
    let mut storages: Vec<Arc<RwLock<fuel_memory::Storage>>> = Vec::with_capacity(probe.len());
    for t in probe {
        let cb = CudaStorageBytes::from_cpu_bytes(device, &t.bytes)?;
        storages.push(Arc::new(RwLock::new(fuel_memory::Storage::new(
            fuel_memory::BackendStorage::Cuda(cb),
            t.dtype,
        ))));
    }

    // (b) Build the reference graph: one Const leaf per probe (ids
    // `0..n_inputs`) + the claimed fused op as the sink over those leaves.
    let graph = Arc::new(RwLock::new(Graph::new()));
    let (input_ids, fused_id) = {
        let mut g = graph.write().map_err(|_| {
            Error::Msg("reference_from_registered_recipe: graph RwLock poisoned".to_string())
        })?;
        let input_ids: Vec<NodeId> = probe
            .iter()
            .map(|t| {
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(&t.shape),
                    dtype: t.dtype,
                })
            })
            .collect();
        let fused_id = g.push(Node {
            op: Op::Fused(claimed_op, params.clone()),
            inputs: input_ids.clone(),
            shape: Shape::from_dims(&out_shape),
            dtype: out_dtype,
        });
        (input_ids, fused_id)
    };

    // (c) Lower the fused sink to FUEL's registered primitive base map. The
    // `Op::Fused` node dissolves IN PLACE (recursive fixpoint); `roots[0]` is
    // the lowered sink. No graph guard is held — lowering locks internally.
    let roots = lower_to_base_map(&graph, &[fused_id]);
    let sink = *roots.first().ok_or_else(|| {
        Error::Msg("reference_from_registered_recipe: lowering returned no roots".to_string())
    })?;

    // (d) Stamp CUDA on every reachable NON-`Const` node of the LOWERED base
    // map (not just the original fused node — after lowering the primitives are
    // what realize will dispatch). The `Op::Const` probe leaves are adopted
    // from the StorageCache (no kernel), so they need no target backend.
    {
        let mut g = graph.write().map_err(|_| {
            Error::Msg("reference_from_registered_recipe: graph RwLock poisoned".to_string())
        })?;
        for nid in topo_order_multi(&g, &roots) {
            if !matches!(g.node(nid).op, Op::Const) {
                g.set_target_backend(nid, BackendId::Cuda);
            }
        }
    }

    // (e) Bind each probe storage to its Const node id.
    let mut cache = StorageCache::new();
    for (id, storage) in input_ids.iter().zip(storages) {
        cache.insert(*id, storage);
    }

    // (f) Realize the lowered base-map sink on the device.
    let (out_arc, _layout) = PipelinedExecutor::realize(graph, sink, cache)?;

    // (g) D2H: read the CUDA output storage back to host bytes.
    let bytes = {
        let guard = out_arc.read().map_err(|_| {
            Error::Msg(
                "reference_from_registered_recipe: output storage RwLock poisoned".to_string(),
            )
        })?;
        match &guard.inner {
            fuel_memory::BackendStorage::Cuda(c) => c.to_cpu_bytes()?,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Msg(
                    "reference_from_registered_recipe: realized output storage is not CUDA"
                        .to_string(),
                ));
            }
        }
    };

    Ok(HostTensor {
        dtype: out_dtype,
        shape: out_shape,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernels_types::ElementKind;
    use fuel_ir::DType;

    /// **The whole probe set is examined — a failure in the LAST probe still
    /// fails the candidate.**
    ///
    /// This is the load-bearing half of "does the gate look at the inputs it
    /// claims to", and unlike the call-site binding it needs no GPU.
    ///
    /// **It is the exact shape GAP-236 is about.** A candidate that lifts
    /// `fmaxf` while claiming `Max` is faithful on the seeded fill (probe 0)
    /// and unfaithful only on the special values (probe 1). **An admission
    /// loop that short-circuited on the first probe's `Pass`, or that only
    /// ever ran `probes[0]`, would admit it while appearing to consult the
    /// whole set** — and every count in the harness would look identical.
    #[test]
    fn a_failure_in_the_last_probe_still_fails_the_candidate() {
        let v = combine_probe_verdicts(vec![
            VerifyVerdict::Pass,
            VerifyVerdict::Fail {
                claim: "max_ulp",
                detail: "elem 3: NaN vs 1.0".to_string(),
            },
        ]);
        match v {
            VerifyVerdict::Fail { detail, .. } => {
                // The index must be NAMED. "a probe failed" does not tell a
                // reader that it was the special-value arm, and a numeric miss
                // on the seeded fill means something entirely different.
                assert!(
                    detail.contains("probe 1"),
                    "the failing probe index is not named in: {detail}"
                );
            }
            other => panic!("a failure in the last probe must fail the candidate, got {other:?}"),
        }
    }

    /// **An empty probe set is a `Fail`, never a `Pass`.**
    ///
    /// Nothing was verified. A `Pass` over zero evidence is exactly the shape
    /// this program exists to eliminate, and it is reachable in practice:
    /// `probe_from_operands` returns `None` for an operand it cannot encode,
    /// so a candidate whose operands are all unencodable yields an EMPTY set
    /// rather than an error.
    #[test]
    fn an_empty_probe_set_is_a_fail_not_a_pass() {
        match combine_probe_verdicts(Vec::new()) {
            VerifyVerdict::Fail { claim, .. } => assert_eq!(claim, "probe"),
            other => panic!("an empty probe set must not pass; got {other:?}"),
        }
    }

    /// **`Inconclusive` outranks silence but not failure.**
    ///
    /// Escalation must survive a set where other probes passed, and must NOT
    /// mask a real failure elsewhere in the same set.
    #[test]
    fn inconclusive_survives_a_pass_and_yields_to_a_fail() {
        let esc = || VerifyVerdict::Inconclusive {
            claim: "max_ulp",
            detail: "live reference only".to_string(),
        };
        let fail = || VerifyVerdict::Fail {
            claim: "max_ulp",
            detail: "diverged".to_string(),
        };

        assert!(
            matches!(
                combine_probe_verdicts(vec![VerifyVerdict::Pass, esc()]),
                VerifyVerdict::Inconclusive { .. }
            ),
            "an inconclusive probe must not be swallowed by a passing one"
        );
        assert!(
            matches!(
                combine_probe_verdicts(vec![esc(), fail()]),
                VerifyVerdict::Fail { .. }
            ),
            "a real failure must outrank an inconclusive in the same set"
        );
    }

    /// **GAP-236 increment 2: the admission path must actually CONSUME the
    /// probe set.**
    ///
    /// ⚠️ **THIS TEST IS EXPECTED TO FAIL TODAY. That is its job, and it is
    /// `#[ignore]`d rather than deleted so the pending state is NAMED instead
    /// of implied.** A `pub fn` that nothing calls emits no warning, so a
    /// landed `admission_probes` reads exactly like a landed fix — the whole
    /// class of "which number moves if this became a no-op?", where the honest
    /// answer today is NONE.
    ///
    /// The wiring is one edit at `verify_candidate_impl`'s `// (1) Probe
    /// synthesis` block, but that call site is `#[cfg(feature = "cuda")]` and
    /// cannot be compile-verified without a full baracuda forge (~56 min via
    /// `scripts/cuda-build.ps1`). Landing an unverifiable edit to CUDA-gated
    /// code would put main's cuda build at risk for every other lane, so the
    /// wiring is a separate increment.
    ///
    /// **When the wiring lands, DELETE THE `#[ignore]`** — this becomes a
    /// permanent guard that the gate looks at the inputs it claims to.
    ///
    /// ⚠️ **WHAT REMAINS IGNORED IS NOW ONLY THE BINDING, NOT THE LOGIC.** The
    /// load-bearing half — *does the admission loop actually consume every
    /// probe in the set* — is a LIVE test at `cfg(jit)`:
    /// `a_failure_in_the_last_probe_still_fails_the_candidate`, which a
    /// short-circuit-on-first or a `probes[0]`-only loop both fail. Together
    /// with `an_empty_probe_set_is_a_fail_not_a_pass` and
    /// `inconclusive_survives_a_pass_and_yields_to_a_fail`, the POLICY is
    /// verified without a GPU and each was born-red by its own sabotage.
    ///
    /// **So the unverified surface is one call-site name.** That is deliberate:
    /// the per-probe body cannot move to the neutral side — it takes
    /// `&CudaDevice` and performs a real CUDA invoke to produce both the
    /// candidate output and Fuel's realized reference — but the POLICY it
    /// feeds can, and has.
    ///
    /// It scans `jit_ingest.rs` rather than living in it, because a
    /// source-scanning check inside the file it scans matches its own anchors
    /// — a trap this crate hit three times in one session.
    #[test]
    #[ignore = "GAP-236 increment 2: fails until the cuda-gated admission call site is wired; run explicitly to see the pending state"]
    fn gap_236_the_probe_set_is_actually_consumed() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/jit_ingest.rs");
        let src =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));

        // Positive control on the SCAN, not just the predicate: the admission
        // block must be findable at all, or "no call found" would be a wrong
        // path rather than a missing wire.
        assert!(
            src.contains("(1) Probe synthesis"),
            "cannot find the admission block in {path:?} — this scan is looking in the \
             wrong place and its verdict below would be meaningless"
        );

        assert!(
            src.contains("admission_probes("),
            "`verify_candidate_impl` does not call `admission_probes`, so candidate \
             admission still runs on the seeded `[-0.5, 0.5)` fill alone and GAP-236's \
             hole is OPEN. `special_value_probe_from_operands` exists and is proven to \
             reach the fmaxf-vs-Max divergence, but nothing in production looks at it."
        );
    }

    /// **GAP-236 BORN-RED: the admission probe cannot reach any input on which
    /// a NaN-suppressing and a NaN-propagating `Max` disagree.**
    ///
    /// This is the whole of GAP-236 reduced to the one property that makes it
    /// a hole. A candidate that lifts `fmaxf` (IEEE `maxNum`, NaN-SUPPRESSING)
    /// while honestly claiming `Max` (NaN-PROPAGATING) passes every existing
    /// layer: it claims the right op, its decompose lowers to `Max`, Fuel's
    /// own recipe is `Max` so `recipe_identity_matches` agrees, and the
    /// reference is correctly Fuel's own. **The two implementations are then
    /// compared on a probe whose inputs cannot tell them apart.**
    ///
    /// `fill_deterministic` computes `((r >> 40) as f32 / 2^24) - 0.5`, so
    /// every element lands in `[-0.5, 0.5)`: finite, normal, never NaN, never
    /// +/-inf, never denormal, never -0.0. The entire difference between the
    /// four min/max ops lives in `cmp_ne(x, x)` — a test no input in that
    /// range can ever make true.
    ///
    /// ⚠️ **POLARITY, STATED BECAUSE IT INVERTS THE OBVIOUS FRAMING.** The
    /// defect is usually described as "the unfaithful candidate is ADMITTED",
    /// which passes today and must fail later. This test is the GUARD, so it
    /// runs the other way: it asserts the probe REACHES the divergence set,
    /// which is FALSE today (born-red) and TRUE once the special-value arm
    /// lands. A guard that passed while the defect existed would be the
    /// vacuous kind this project keeps filing.
    ///
    /// It scans many seeds rather than the one seed a call site happens to
    /// use, because the claim is about the FILL, not about a draw from it.
    #[test]
    fn gap_236_the_admission_probe_reaches_the_nan_divergence_set() {
        // NaN-suppressing (IEEE maxNum / C `fmaxf`): a NaN operand is ignored.
        //
        // The first arm and the last both yield `b`, which clippy reads as a
        // duplicated branch. They are DIFFERENT CASES that happen to agree:
        // the first is NaN SUPPRESSION (`a` is NaN, so `b` wins by rule), the
        // last is ORDINARY COMPARISON (`a <= b`, so `b` wins by value).
        // Collapsing them would delete the very distinction this test exists
        // to draw against `max_prop` below.
        #[allow(clippy::if_same_then_else)]
        fn fmax_ieee(a: f32, b: f32) -> f32 {
            if a.is_nan() {
                b
            } else if b.is_nan() {
                a
            } else if a > b {
                a
            } else {
                b
            }
        }
        // NaN-propagating (what `Max` means in Fuel and in KISS `ops.md`).
        fn max_prop(a: f32, b: f32) -> f32 {
            if a.is_nan() || b.is_nan() {
                f32::NAN
            } else if a > b {
                a
            } else {
                b
            }
        }

        let od = OperandDesc::new(1, &[64], &[1], ElementKind::F32, 64 * 4);
        let mut divergences = 0usize;
        let mut specials = 0usize;
        let mut scanned = 0usize;

        for seed in 0..256u64 {
            for probe in admission_probes(&[od, od], seed ^ 0xA11CE) {
                let dec = |t: &HostTensor| -> Vec<f32> {
                    let (chunks, _rest) = t.bytes.as_chunks::<4>();
                    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
                };
                let (a, b) = (dec(&probe[0]), dec(&probe[1]));
                for (&x, &y) in a.iter().zip(b.iter()) {
                    scanned += 1;
                    if !x.is_finite() || !y.is_finite() || x == 0.0f32 && x.is_sign_negative() {
                        specials += 1;
                    }
                    // Bit-compare, so a NaN-vs-NaN "disagreement" is not counted
                    // as agreement by `==`.
                    if fmax_ieee(x, y).to_bits() != max_prop(x, y).to_bits() {
                        divergences += 1;
                    }
                }
            }
        }

        // Non-vacuity: the scan must actually have produced inputs.
        assert!(
            scanned > 10_000,
            "only {scanned} element pairs scanned — the probe builder returned \
             nothing and the assertion below would pass or fail for the wrong reason"
        );

        println!(
            "[gap-236] {scanned} element pairs scanned; {specials} non-finite/-0.0 inputs;              {divergences} fmaxf-vs-Max divergences reached"
        );
        // The specials count is a SEPARATE claim from the divergence count and
        // is asserted separately: a divergence could in principle arise from a
        // comparison quirk, whereas this says the probe genuinely carries the
        // values the four min/max ops are distinguished by.
        assert!(
            specials > 0,
            "the admission probe set contains no non-finite and no -0.0 input, so              whatever produced the divergence count is not the special-value arm"
        );
        assert!(
            divergences > 0,
            "GAP-236: across {scanned} element pairs from {} seeds, a NaN-SUPPRESSING \
             `fmaxf` and a NaN-PROPAGATING `Max` produced BIT-IDENTICAL results EVERY \
             TIME ({specials} non-finite or -0.0 inputs seen). The admission probe \
             cannot reach the divergence set, so a candidate that lifts `fmaxf` while \
             claiming `Max` agrees with Fuel's reference everywhere the probe can look \
             and is admitted. `fill_deterministic` yields `[-0.5, 0.5)`; the entire \
             difference between the four min/max ops is `cmp_ne(x, x)`, which no input \
             in that range can make true.",
            256
        );
    }

    #[test]
    fn probe_from_operands_builds_sized_float_inputs() {
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let p = probe_from_operands(&[od, od], 0x1234).expect("probe");
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].shape, vec![4]);
        assert_eq!(p[0].dtype, DType::F32);
        assert_eq!(p[0].bytes.len(), 16);
        assert_eq!(
            probe_from_operands(&[od, od], 0x1234).unwrap()[0].bytes,
            p[0].bytes
        ); // deterministic
    }

    /// Task-3 carry-forward (negative path, deferred to Task 5): an operand
    /// whose `ElementKind` DOES map to a Fuel `DType` (`element_kind_to_dtype`
    /// succeeds — `I32 → DType::I32`) but which `to_bytes` can't encode (only
    /// F32/F64/BF16/F16 are encodable) makes `probe_from_operands` return
    /// `None` — it never fabricates a probe for an operand it can't faithfully
    /// represent. Non-GPU (`--features jit`).
    #[test]
    fn probe_from_operands_rejects_an_unencodable_integer_operand() {
        // I32: element_kind_to_dtype(I32) = Some(DType::I32), but
        // to_bytes(DType::I32, ..) = None (integer dtypes aren't float-encodable).
        let int_od = OperandDesc::new(1, &[4], &[1], ElementKind::I32, 16);
        assert!(
            probe_from_operands(&[int_od], 0x1234).is_none(),
            "an unencodable-dtype operand must yield None, not a fabricated probe"
        );
        // A valid F32 operand alongside the unencodable one still fails the
        // whole probe (any un-encodable operand poisons the set).
        let f32_od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        assert!(probe_from_operands(&[f32_od, int_od], 0x1234).is_none());
    }

    /// Build a contiguous F32 `HostTensor` of shape `[vals.len()]`.
    #[cfg(feature = "cuda")]
    fn ht_f32(vals: &[f32]) -> HostTensor {
        HostTensor {
            dtype: DType::F32,
            shape: vec![vals.len()],
            bytes: bytemuck::cast_slice(vals).to_vec(),
        }
    }

    /// Reinterpret a byte buffer as `f32`s (little-endian, native).
    #[cfg(feature = "cuda")]
    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
    }

    /// Shared rope-probe builder (Task 4 acceptance test + Task 6 reuse):
    /// deterministic F32 `x` of shape `[1, seq, head_dim]` (via
    /// `fill_deterministic`) plus cos/sin tables `[seq, head_dim]` from
    /// `build_rope_tables`. Returns `(x, cos, sin, x_shape)`.
    #[cfg(feature = "cuda")]
    fn rope_probe(
        seq: usize,
        head_dim: usize,
        base: f64,
    ) -> (HostTensor, HostTensor, HostTensor, Vec<usize>) {
        use fuel_graph::build_rope_tables;
        let x_shape = vec![1usize, seq, head_dim];
        let x_vals = fill_deterministic(seq * head_dim, 0x5EED_D07);
        let (cos, sin) = build_rope_tables(base, 0, seq, head_dim);
        let x = HostTensor {
            dtype: DType::F32,
            shape: x_shape.clone(),
            bytes: bytemuck::cast_slice(&x_vals).to_vec(),
        };
        let cos_t = HostTensor {
            dtype: DType::F32,
            shape: vec![seq, head_dim],
            bytes: bytemuck::cast_slice(&cos).to_vec(),
        };
        let sin_t = HostTensor {
            dtype: DType::F32,
            shape: vec![seq, head_dim],
            bytes: bytemuck::cast_slice(&sin).to_vec(),
        };
        (x, cos_t, sin_t, x_shape)
    }

    /// Host-side rotate-half rope — the SAME formula `registry::rope::decompose`
    /// encodes: `out = x*cos + rotate_half(x)*sin`, where
    /// `rotate_half(x) = concat(-x[half:], x[:half])` along the last dim. cos/sin
    /// are `[seq, head_dim]`, broadcast over the leading batch dim of `x`.
    #[cfg(feature = "cuda")]
    fn expected_rotate_half(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        seq: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let half = head_dim / 2;
        let mut out = vec![0.0f32; seq * head_dim];
        for s in 0..seq {
            let row = s * head_dim;
            for d in 0..head_dim {
                let rot = if d < half {
                    -x[row + d + half]
                } else {
                    x[row + d - half]
                };
                out[row + d] = x[row + d] * cos[row + d] + rot * sin[row + d];
            }
        }
        out
    }

    /// Live-GPU (Spec-B Task-4 acceptance test): `reference_from_registered_recipe`
    /// builds `Op::Fused(FusedOps::ROPE, FusedOpParams::Rope)` on rope-shaped F32
    /// probes, lowers it to Fuel's registered primitive base map, realizes it on
    /// CUDA, and returns bytes equal (F32 tolerance) to a host-computed rotate-half
    /// rope. `#[ignore]`'d (needs a live CUDA device).
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn reference_from_registered_recipe_realizes_rotate_half_rope() {
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::registry::{FusedOpParams, FusedOps};

        let Ok(dev) = CudaDevice::new(0) else {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
            );
        };
        let (seq, head_dim, base) = (4usize, 8usize, 10000.0f64);
        let (x, cos_t, sin_t, x_shape) = rope_probe(seq, head_dim, base);

        // Host reference computed from the SAME rotate-half formula the recipe encodes.
        let x_vals = bytes_to_f32(&x.bytes);
        let cos_vals = bytes_to_f32(&cos_t.bytes);
        let sin_vals = bytes_to_f32(&sin_t.bytes);
        let expected = expected_rotate_half(&x_vals, &cos_vals, &sin_vals, seq, head_dim);

        let out = reference_from_registered_recipe(
            FusedOps::ROPE,
            &FusedOpParams::Rope,
            &[x, cos_t, sin_t],
            DType::F32,
            x_shape,
            &dev,
        )
        .expect("reference_from_registered_recipe should realize the lowered rope base map");

        let got = bytes_to_f32(&out.bytes);
        assert_eq!(got.len(), expected.len(), "output element count");
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "rotate-half rope mismatch at {i}: got {g}, expected {e}"
            );
        }
    }

    /// Live-GPU: `reference_output` realizes a 2-input `Add` decompose region
    /// on two F32 `[4]` probes and returns the elementwise sum. `#[ignore]`'d
    /// (needs a live CUDA device); this is the Spec-B Task-4 acceptance test.
    #[test]
    #[ignore = "requires a live CUDA device"]
    #[cfg(feature = "cuda")]
    fn reference_output_realizes_the_decompose() {
        use fuel_cuda_backend::CudaDevice;
        use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};

        let Ok(dev) = CudaDevice::new(0) else {
            return fuel_test_support::hardware::skip(
                fuel_test_support::hardware::Hardware::Cuda,
                fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
            );
        };
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![
                PatternNode::Bind { index: 0 },
                PatternNode::Bind { index: 1 },
            ],
        };
        let a = ht_f32(&[1.0, 2.0, 3.0, 4.0]);
        let b = ht_f32(&[10.0, 20.0, 30.0, 40.0]);
        let out = reference_output(&region, &[a, b], DType::F32, vec![4], &dev).unwrap();
        assert_eq!(bytes_to_f32(&out.bytes), vec![11.0, 22.0, 33.0, 44.0]);
    }
}

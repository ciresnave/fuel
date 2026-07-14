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

use crate::fkc::verify::{fill_deterministic, to_bytes, HostTensor};
use crate::jit_adopt::element_kind_to_dtype;

#[cfg(feature = "cuda")]
use std::sync::{Arc, RwLock};
#[cfg(feature = "cuda")]
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
#[cfg(feature = "cuda")]
use fuel_graph::jit::PatternNode;
#[cfg(feature = "cuda")]
use fuel_graph::runtime_fused::emit_region;
#[cfg(feature = "cuda")]
use fuel_graph::{Graph, Node, NodeId, Op};
#[cfg(feature = "cuda")]
use fuel_ir::probe::BackendId;
#[cfg(feature = "cuda")]
use fuel_ir::{DType, Error, Result, Shape};
#[cfg(feature = "cuda")]
use crate::pipelined::{PipelinedExecutor, StorageCache};

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
            Some(HostTensor { dtype, shape, bytes })
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
/// is returned as `Err`. (The only panic risk is a non-re-emittable `OpTag`
/// inside `emit_region`; a *validated* decompose never carries one, and Task
/// 5's verifier wraps the whole call in `catch_unwind`.)
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
        let guard = out_arc
            .read()
            .map_err(|_| Error::Msg("reference_output: output storage RwLock poisoned".to_string()))?;
        match &guard.inner {
            fuel_memory::BackendStorage::Cuda(c) => c.to_cpu_bytes()?,
            #[allow(unreachable_patterns)]
            _ => {
                return Err(Error::Msg(
                    "reference_output: realized output storage is not CUDA".to_string(),
                ))
            }
        }
    };

    Ok(HostTensor { dtype: out_dtype, shape: out_shape, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use baracuda_kernels_types::ElementKind;
    use fuel_ir::DType;

    #[test]
    fn probe_from_operands_builds_sized_float_inputs() {
        let od = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 16);
        let p = probe_from_operands(&[od, od], 0x1234).expect("probe");
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].shape, vec![4]);
        assert_eq!(p[0].dtype, DType::F32);
        assert_eq!(p[0].bytes.len(), 16);
        assert_eq!(probe_from_operands(&[od, od], 0x1234).unwrap()[0].bytes, p[0].bytes); // deterministic
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
            eprintln!("no CUDA device; skipping");
            return;
        };
        let region = PatternNode::Op {
            op: OpTag::Add,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
        };
        let a = ht_f32(&[1.0, 2.0, 3.0, 4.0]);
        let b = ht_f32(&[10.0, 20.0, 30.0, 40.0]);
        let out = reference_output(&region, &[a, b], DType::F32, vec![4], &dev).unwrap();
        assert_eq!(bytes_to_f32(&out.bytes), vec![11.0, 22.0, 33.0, 44.0]);
    }
}

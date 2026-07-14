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
}

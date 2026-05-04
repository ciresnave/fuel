//! Typed byte-shaped kernels — Phase 7.5 B5.
//!
//! These kernels operate on [`CpuStorageBytes`] (bytes-based CPU
//! storage). They take typed slices via `bytemuck::cast_slice` /
//! `as_slice<T>` / `as_slice_mut<T>`, do the per-element work, and
//! return.
//!
//! These are the per-T monomorphic units that the dispatch wrapper
//! in `fuel_storage::dispatch::cpu_wrappers` calls after extracting
//! the `CpuStorageBytes` from a `BackendStorage::Cpu(...)` variant.
//!
//! ## Status
//!
//! B5 ships a minimal proof-of-concept: `add_f32` only. Phase C
//! migrates the rest of the unary/binary/reduce/etc. families.

use crate::byte_storage::CpuStorageBytes;
use fuel_core_types::Result;

/// Element-wise `f32` addition: `out[i] = lhs[i] + rhs[i]`.
///
/// Caller is responsible for shape-checking before invoking; this
/// kernel just verifies the byte counts are equal across all three
/// storages.
///
/// Output Storage is pre-allocated by the caller (the dispatch
/// wrapper); the kernel writes into the pre-allocated bytes.
pub fn add_f32(
    lhs: &CpuStorageBytes,
    rhs: &CpuStorageBytes,
    out: &mut CpuStorageBytes,
) -> Result<()> {
    if lhs.len_bytes() != rhs.len_bytes() || lhs.len_bytes() != out.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "add_f32: byte length mismatch (lhs={}, rhs={}, out={})",
            lhs.len_bytes(),
            rhs.len_bytes(),
            out.len_bytes(),
        ))
        .bt());
    }
    let lhs_view: &[f32] = lhs.as_slice()?;
    let rhs_view: &[f32] = rhs.as_slice()?;
    let out_view: &mut [f32] = out.as_slice_mut()?;
    for (i, slot) in out_view.iter_mut().enumerate() {
        *slot = lhs_view[i] + rhs_view[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: add two f32 storages elementwise.
    #[test]
    fn add_f32_round_trip() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0, 40.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(16);

        add_f32(&a, &b, &mut out).expect("add");

        let result: &[f32] = out.as_slice().unwrap();
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
    }

    /// Mismatched byte counts produce an error, not a panic.
    #[test]
    fn add_f32_errors_on_size_mismatch() {
        let a = CpuStorageBytes::from_slice(&[1.0_f32, 2.0]);
        let b = CpuStorageBytes::from_slice(&[10.0_f32, 20.0, 30.0]);
        let mut out = CpuStorageBytes::from_zero_bytes(8);

        let result = add_f32(&a, &b, &mut out);
        assert!(result.is_err(), "size mismatch must error");
    }
}

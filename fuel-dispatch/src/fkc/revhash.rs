//! `kernel_revision_hash` computation (adoption plan §8 / FKC §4.7).
//!
//! A stable, endianness-stable, **non-SipHash** hash (FNV-1a, fixed) over
//! a **canonicalized** form of the parsed contract — NOT the raw YAML
//! bytes. Canonicalization means: serialize the structured fields in a
//! fixed key order with normalized scalars (a comment edit or a
//! reformat does not change the hash; a semantic change does).
//!
//! Two modes (§4.7):
//! - `"auto"` ⇒ `hash(entry_point ++ revision_base ++ canonical-block)`
//!   so editing the contract or bumping the provider build changes it.
//! - an explicit hex value ⇒ parsed directly.
//!
//! We reuse the existing [`KernelRevisionHash`] newtype from
//! [`crate::fused`] (the fused path already round-trips a revision hash),
//! so a primitive's revision and a fused op's revision are the same type.
//!
//! FNV-1a is chosen over SipHash because SipHash's default seed is
//! process-random — its output would differ run-to-run, breaking the
//! persisted-plan re-resolution key (digest §7). FNV-1a is a fixed,
//! deterministic, endianness-stable function: the same canonical input
//! always hashes to the same `u64` across processes and machines.

use crate::fkc::error::FkcError;
use crate::fkc::schema::FkcKernel;
use crate::fused::KernelRevisionHash;

/// FNV-1a 64-bit offset basis + prime (the standard fixed constants).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over a byte slice. Endianness-stable (operates byte-by-byte),
/// fixed constants, no random seed.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Build the canonical byte form of a parsed contract: the
/// dispatch-relevant structured fields in a fixed order, each scalar
/// normalized (trimmed), joined by a record separator. Comments and prose
/// never reach this function (they are not in [`FkcKernel`]), so a
/// reformat / comment edit cannot change the hash; a `flops` or dtype
/// change does.
///
/// The field order is fixed here and must not be reordered without
/// intending a hash change (it IS the canonicalization contract).
fn canonical_form(k: &FkcKernel) -> String {
    let mut out = String::new();
    // A small helper that appends `key=value` then a record separator.
    let mut push = |key: &str, val: &str| {
        out.push_str(key);
        out.push('=');
        out.push_str(val.trim());
        out.push('\x1e'); // ASCII record separator
    };

    push("op_kind", k.op_kind.as_deref().unwrap_or(""));
    push("fused_op", k.fused_op.as_deref().unwrap_or(""));
    push("backend", k.backend.as_deref().unwrap_or(""));
    push("kernel_source", k.kernel_source.as_deref().unwrap_or(""));
    push("entry_point", k.entry_point.as_deref().unwrap_or(""));
    push("determinism", k.determinism.as_deref().unwrap_or(""));

    // accept: per-operand ordered dtypes + dtype_class (semantic).
    if let Some(accept) = &k.accept {
        for (i, d) in accept.inputs.iter().enumerate() {
            push(&format!("in{i}.name"), d.name.as_deref().unwrap_or(""));
            push(&format!("in{i}.dtypes"), &d.dtypes.join(","));
            push(
                &format!("in{i}.dtype_class"),
                d.dtype_class.as_deref().unwrap_or(""),
            );
            if let Some(layout) = &d.layout {
                push(
                    &format!("in{i}.layout"),
                    &format!(
                        "{}|{}|{}|{}|{}",
                        layout.contiguous.as_deref().unwrap_or(""),
                        layout.strided.as_deref().unwrap_or(""),
                        layout.broadcast_stride0.as_deref().unwrap_or(""),
                        layout.start_offset.as_deref().unwrap_or(""),
                        layout.reverse_strides.as_deref().unwrap_or(""),
                    ),
                );
            }
            if let Some(fdx) = &d.fdx {
                if let Some(q) = &fdx.quant {
                    push(
                        &format!("in{i}.quant"),
                        &format!(
                            "{}|{}|{}|{}|{}",
                            q.family.as_deref().unwrap_or(""),
                            q.ggml_dtype.as_deref().unwrap_or(""),
                            q.granularity.as_deref().unwrap_or(""),
                            q.role.as_deref().unwrap_or(""),
                            q.scale_operand.as_deref().unwrap_or(""),
                        ),
                    );
                }
            }
        }
        if let Some(op_params) = &accept.op_params {
            push("op_params.variant", op_params.variant.as_deref().unwrap_or(""));
        }
    }

    // cost: the coefficient expressions are semantic (a `flops` edit
    // changes the hash).
    if let Some(cost) = &k.cost {
        push("cost.flops", cost.flops.as_deref().unwrap_or(""));
        push("cost.bytes_moved", cost.bytes_moved.as_deref().unwrap_or(""));
        push("cost.class", cost.class.as_deref().unwrap_or(""));
        push("cost.provenance", cost.provenance.as_deref().unwrap_or(""));
    }

    out
}

/// Compute (or parse) a kernel's [`KernelRevisionHash`] (§4.7).
///
/// - `kernel_revision_hash: auto` ⇒
///   `FNV1a(entry_point ++ revision_base ++ canonical-block)`.
/// - an explicit hex value ⇒ parsed (a leading `0x` is optional).
/// - absent ⇒ treated as `auto`.
///
/// `entry_point` and `revision_base` come from the kernel (with
/// front-matter fallback applied by the caller). They are folded in so two
/// otherwise-identical contracts on different entry points / provider
/// builds get distinct hashes.
pub fn compute_revision(
    kernel: &FkcKernel,
    entry_point: &str,
    revision_base: &str,
) -> Result<KernelRevisionHash, FkcError> {
    match kernel.kernel_revision_hash.as_deref() {
        None | Some("auto") => {
            let mut material = String::new();
            material.push_str(entry_point.trim());
            material.push('\x1f'); // unit separator between the three parts
            material.push_str(revision_base.trim());
            material.push('\x1f');
            material.push_str(&canonical_form(kernel));
            Ok(KernelRevisionHash(fnv1a(material.as_bytes())))
        }
        Some(hex) => {
            let cleaned = hex.trim().trim_start_matches("0x").trim_start_matches("0X");
            let value = u64::from_str_radix(cleaned, 16).map_err(|_| {
                FkcError::Yaml(format!(
                    "kernel `{}`: kernel_revision_hash `{hex}` is neither `auto` nor a hex u64",
                    kernel.kernel
                ))
            })?;
            Ok(KernelRevisionHash(value))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::parse_file;

    const ELEMENTWISE_BINARY: &str =
        include_str!("../../../docs/kernel-contracts/cpu/elementwise-binary.fkc.md");

    fn add_f32() -> FkcKernel {
        let file = parse_file(ELEMENTWISE_BINARY).expect("parses");
        file.kernels
            .iter()
            .find(|k| k.kernel == "add_f32")
            .expect("add_f32")
            .clone()
    }

    #[test]
    fn fnv1a_is_deterministic_frozen_fixture() {
        // FROZEN: pin the exact FNV-1a output for a fixed input so the
        // function choice cannot silently change (mirrors FDX's
        // build-time mapping test, §8).
        assert_eq!(fnv1a(b""), FNV_OFFSET_BASIS);
        // "a" → standard FNV-1a 64-bit reference value.
        assert_eq!(fnv1a(b"a"), 0xaf63dc4c8601ec8c);
        // "foobar" → standard FNV-1a 64-bit reference value.
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn auto_is_deterministic_for_fixed_inputs() {
        let k = add_f32();
        let a = compute_revision(&k, "fuel_cpu_backend::byte_kernels::add_f32", "git:f41137b4")
            .unwrap();
        let b = compute_revision(&k, "fuel_cpu_backend::byte_kernels::add_f32", "git:f41137b4")
            .unwrap();
        assert_eq!(a, b, "same canonical input ⇒ same hash");
    }

    #[test]
    fn different_entry_point_changes_hash() {
        let k = add_f32();
        let a = compute_revision(&k, "ep::one", "git:x").unwrap();
        let b = compute_revision(&k, "ep::two", "git:x").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn different_revision_base_changes_hash() {
        let k = add_f32();
        let a = compute_revision(&k, "ep", "git:aaaa").unwrap();
        let b = compute_revision(&k, "ep", "git:bbbb").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn semantic_change_changes_hash_but_reformat_does_not() {
        // Two contracts identical except whitespace/comments in the YAML
        // (which never reach FkcKernel) hash equal; a flops change differs.
        let mut k1 = add_f32();
        let mut k2 = add_f32(); // reformatted twin
        // Reformat: re-pad a token that gets trimmed in canonical_form.
        k2.op_kind = Some("  AddElementwise  ".into());
        let h1 = compute_revision(&k1, "ep", "git:x").unwrap();
        let h2 = compute_revision(&k2, "ep", "git:x").unwrap();
        assert_eq!(h1, h2, "trimmed-equivalent scalars hash equal");

        // A semantic flops edit changes the hash.
        if let Some(cost) = k1.cost.as_mut() {
            cost.flops = Some("2 * n".into());
        }
        let h3 = compute_revision(&k1, "ep", "git:x").unwrap();
        assert_ne!(h1, h3, "a flops change must change the hash");
    }

    #[test]
    fn explicit_hex_parses() {
        let mut k = add_f32();
        k.kernel_revision_hash = Some("0xdeadbeef".into());
        let h = compute_revision(&k, "ep", "git:x").unwrap();
        assert_eq!(h, KernelRevisionHash(0xdeadbeef));

        k.kernel_revision_hash = Some("ff".into());
        let h = compute_revision(&k, "ep", "git:x").unwrap();
        assert_eq!(h, KernelRevisionHash(0xff));
    }

    #[test]
    fn bad_hex_is_error() {
        let mut k = add_f32();
        k.kernel_revision_hash = Some("not-hex-zzz".into());
        assert!(compute_revision(&k, "ep", "git:x").is_err());
    }
}

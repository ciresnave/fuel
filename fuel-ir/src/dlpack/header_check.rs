//! Header↔Rust drift gate (no C compiler required).
//!
//! `include_str!`s the co-maintained C header `include/fuel_dlpack_ext.h`,
//! parses every `_Static_assert(sizeof(NAME) == N, ...)` line out of the
//! embedded text, and asserts — for each FDX / standard-DLPack struct NAME —
//! that the header's declared `N` equals the Rust struct's actual
//! `core::mem::size_of`. A same-size field re-shuffle slips through (sizeof is
//! the only signal here), but any size drift between the header and the Rust
//! `#[repr(C)]` definitions fails `cargo test`. As a bonus it also cross-checks
//! a handful of `#define FDX_* N` constants against `super::codes`.
//!
//! 64-bit LE host only (the v1 target, §5): the header asserts are written for
//! that layout, so the Rust-side comparison is likewise gated to a 64-bit
//! pointer width.

use super::abi::*;
use super::sidecar::*;
use core::mem::size_of;

/// The header text, embedded at compile time. Path is relative to THIS source
/// file (`fuel-ir/src/dlpack/header_check.rs`):
/// `../../include/fuel_dlpack_ext.h` -> `fuel-ir/include/...`.
const HEADER: &str = include_str!("../../include/fuel_dlpack_ext.h");

/// Map a struct NAME (as it appears in a header `_Static_assert(sizeof(NAME)`)
/// to the Rust `size_of` for the corresponding type. `None` for a NAME the
/// gate does not know about (the test then fails loudly — every struct asserted
/// in the header MUST be listed here).
fn rust_size_of(name: &str) -> Option<usize> {
    Some(match name {
        // standard DLPack (abi.rs)
        "DLDevice" => size_of::<DLDevice>(),
        "DLDataType" => size_of::<DLDataType>(),
        "DLTensor" => size_of::<DLTensor>(),
        "DLPackVersion" => size_of::<DLPackVersion>(),
        "DLManagedTensorVersioned" => size_of::<DLManagedTensorVersioned>(),
        // FDX (sidecar.rs)
        "FDXDTypeExt" => size_of::<FDXDTypeExt>(),
        "FDXQuant" => size_of::<FDXQuant>(),
        "FDXAffineTerm" => size_of::<FDXAffineTerm>(),
        "FDXAffine" => size_of::<FDXAffine>(),
        "FDXExtent" => size_of::<FDXExtent>(),
        "FDXTiling" => size_of::<FDXTiling>(),
        "FDXResidency" => size_of::<FDXResidency>(),
        "FDXStorage" => size_of::<FDXStorage>(),
        "FDXOutputView" => size_of::<FDXOutputView>(),
        "FDXBlockTable" => size_of::<FDXBlockTable>(),
        "FDXIndexedResidency" => size_of::<FDXIndexedResidency>(),
        "FDXBufferRef" => size_of::<FDXBufferRef>(),
        "FDXSymBinding" => size_of::<FDXSymBinding>(),
        "FDXSymEnv" => size_of::<FDXSymEnv>(),
        "FDXSidecar" => size_of::<FDXSidecar>(),
        _ => return None,
    })
}

/// Parse `_Static_assert(sizeof(NAME) == N` lines. Returns `(name, n)` pairs.
/// Tolerant of whitespace; ignores everything that is not a `sizeof` assert
/// (e.g. the `offsetof` asserts and the `sizeof(...) == sizeof(...)` one).
fn parse_sizeof_asserts(text: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("_Static_assert(") else {
            continue;
        };
        // Want: sizeof(NAME) == N , "...
        let Some(rest) = rest.trim_start().strip_prefix("sizeof(") else {
            continue;
        };
        let Some((name, after)) = rest.split_once(')') else {
            continue;
        };
        let name = name.trim();
        // The RHS must be a plain integer (skip `== sizeof(...)` / `offsetof`).
        let after = after.trim_start();
        let Some(after) = after.strip_prefix("==") else {
            continue;
        };
        // Take the integer up to the first ',' (the assert message separator).
        let Some((num, _)) = after.split_once(',') else {
            continue;
        };
        let num = num.trim();
        let Ok(n) = num.parse::<usize>() else {
            continue; // not a literal int → not a size assert we verify
        };
        out.push((name.to_string(), n));
    }
    out
}

/// THE GATE: every `sizeof` static-assert in the header must match the Rust
/// `size_of` of the same struct. Fails on any drift.
#[cfg(target_pointer_width = "64")]
#[test]
fn header_struct_sizes_match_rust() {
    let asserts = parse_sizeof_asserts(HEADER);
    assert!(
        !asserts.is_empty(),
        "no `_Static_assert(sizeof(...) == N` lines parsed from the header — \
         the include path or parser is broken"
    );

    // Every struct the gate knows about must be pinned by the header. Track
    // which knowns we saw so a struct silently dropped from the header also
    // fails (not just a mismatched one).
    let known = [
        "DLDevice",
        "DLDataType",
        "DLTensor",
        "DLPackVersion",
        "DLManagedTensorVersioned",
        "FDXDTypeExt",
        "FDXQuant",
        "FDXAffineTerm",
        "FDXAffine",
        "FDXExtent",
        "FDXTiling",
        "FDXResidency",
        "FDXStorage",
        "FDXOutputView",
        "FDXBlockTable",
        "FDXIndexedResidency",
        "FDXBufferRef",
        "FDXSymBinding",
        "FDXSymEnv",
        "FDXSidecar",
    ];
    let mut seen = std::collections::BTreeSet::new();

    for (name, header_n) in &asserts {
        let rust_n = rust_size_of(name).unwrap_or_else(|| {
            panic!(
                "header pins sizeof({name}) == {header_n} but the cross-check \
                 test has no Rust size_of mapping for `{name}` — add it to \
                 `rust_size_of`"
            )
        });
        assert_eq!(
            *header_n, rust_n,
            "header↔Rust size DRIFT: header says sizeof({name}) == {header_n}, \
             Rust size_of::<{name}>() == {rust_n}"
        );
        seen.insert(name.as_str());
    }

    for k in known {
        assert!(
            seen.contains(k),
            "struct `{k}` is no longer size-pinned by a \
             `_Static_assert(sizeof({k}) == N` line in the header — restore it"
        );
    }
}

/// Parse `_Static_assert(offsetof(STRUCT, FIELD) == N` lines. Returns
/// `(struct, field, n)`. Skips the relational `offsetof(...) == offsetof(...)`
/// assert (RHS is not a plain integer).
fn parse_offsetof_asserts(text: &str) -> Vec<(String, String, usize)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("_Static_assert(") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix("offsetof(") else {
            continue;
        };
        let Some((args, after)) = rest.split_once(')') else {
            continue;
        };
        let Some((sname, fname)) = args.split_once(',') else {
            continue;
        };
        let after = after.trim_start();
        let Some(after) = after.strip_prefix("==") else {
            continue;
        };
        let Some((num, _)) = after.split_once(',') else {
            continue;
        };
        let Ok(n) = num.trim().parse::<usize>() else {
            continue; // relational `== offsetof(...)` form → skip
        };
        out.push((sname.trim().to_string(), fname.trim().to_string(), n));
    }
    out
}

/// Cross-check the header's `offsetof` static-asserts against Rust
/// `offset_of!`. These are the load-bearing offsets (FDXExtent fields,
/// FDXIndexedResidency trailing fields, FDXSidecar.gather/reserved) the C
/// compiler would check; we replicate them here so the no-C-compiler gate also
/// catches a field-reshuffle that preserves total size.
#[cfg(target_pointer_width = "64")]
#[test]
fn header_field_offsets_match_rust() {
    use core::mem::offset_of;
    let asserts = parse_offsetof_asserts(HEADER);
    assert!(
        !asserts.is_empty(),
        "no `_Static_assert(offsetof(...) == N` lines parsed from the header"
    );

    let rust_off = |s: &str, f: &str| -> Option<usize> {
        Some(match (s, f) {
            ("FDXExtent", "kind") => offset_of!(FDXExtent, kind),
            ("FDXExtent", "_pad") => offset_of!(FDXExtent, _pad),
            ("FDXExtent", "min") => offset_of!(FDXExtent, min),
            ("FDXExtent", "capacity") => offset_of!(FDXExtent, capacity),
            ("FDXExtent", "sym_id") => offset_of!(FDXExtent, sym_id),
            ("FDXExtent", "sym_scope") => offset_of!(FDXExtent, sym_scope),
            ("FDXExtent", "_pad2") => offset_of!(FDXExtent, _pad2),
            ("FDXExtent", "cap_kind") => offset_of!(FDXExtent, cap_kind),
            ("FDXExtent", "_pad3") => offset_of!(FDXExtent, _pad3),
            ("FDXExtent", "_pad4") => offset_of!(FDXExtent, _pad4),
            ("FDXExtent", "affine") => offset_of!(FDXExtent, affine),
            ("FDXExtent", "reserved") => offset_of!(FDXExtent, reserved),
            ("FDXIndexedResidency", "logical_extents") => {
                offset_of!(FDXIndexedResidency, logical_extents)
            }
            ("FDXIndexedResidency", "context_lens_buffer") => {
                offset_of!(FDXIndexedResidency, context_lens_buffer)
            }
            ("FDXIndexedResidency", "context_len_sym") => {
                offset_of!(FDXIndexedResidency, context_len_sym)
            }
            ("FDXIndexedResidency", "context_len_scope") => {
                offset_of!(FDXIndexedResidency, context_len_scope)
            }
            ("FDXIndexedResidency", "reserved") => {
                offset_of!(FDXIndexedResidency, reserved)
            }
            ("FDXSidecar", "gather") => offset_of!(FDXSidecar, gather),
            ("FDXSidecar", "reserved") => offset_of!(FDXSidecar, reserved),
            _ => return None,
        })
    };

    for (s, f, header_n) in &asserts {
        let rust_n = rust_off(s, f).unwrap_or_else(|| {
            panic!(
                "header pins offsetof({s}, {f}) == {header_n} but the \
                 cross-check has no Rust offset_of! for `{s}.{f}`"
            )
        });
        assert_eq!(
            *header_n, rust_n,
            "header↔Rust offset DRIFT: header says offsetof({s}, {f}) == \
             {header_n}, Rust offset_of!({s}, {f}) == {rust_n}"
        );
    }
}

/// Parse `#define NAME VALUE` lines whose VALUE is a single integer literal
/// (decimal `123u`/`123` or hex `0xABCu`/`0xABC`). Bit-shift / parenthesised
/// expressions are skipped — this is a spot-check, not a full C preprocessor.
fn parse_define_ints(text: &str) -> std::collections::BTreeMap<String, u64> {
    let mut out = std::collections::BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define ") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let Some(name) = it.next() else { continue };
        let Some(val) = it.next() else { continue };
        // strip a trailing C integer suffix (u / U / ul / etc.)
        let val_trim = val.trim_end_matches(['u', 'U', 'l', 'L']);
        let parsed = if let Some(hex) = val_trim
            .strip_prefix("0x")
            .or_else(|| val_trim.strip_prefix("0X"))
        {
            u64::from_str_radix(hex, 16).ok()
        } else {
            val_trim.parse::<u64>().ok()
        };
        if let Some(v) = parsed {
            out.insert(name.to_string(), v);
        }
    }
    out
}

/// Bonus cross-check: a sample of `#define FDX_* N` header values vs the
/// normative `super::codes` constants. Cheap and catches a fat-fingered code.
#[test]
fn header_define_constants_match_codes() {
    use super::codes::*;
    let defines = parse_define_ints(HEADER);

    macro_rules! check {
        ($name:literal, $rust:expr) => {{
            let got = *defines.get($name).unwrap_or_else(|| {
                panic!("header is missing a parseable `#define {} N`", $name)
            });
            assert_eq!(
                got, $rust as u64,
                "header `#define {}` == {}, codes.rs == {}",
                $name, got, $rust as u64
            );
        }};
    }

    check!("FDX_MAGIC", FDX_MAGIC);
    check!("FDX_VERSION_1", FDX_VERSION_1);
    check!("FDX_VERSION_MAX", FDX_VERSION_MAX);
    check!("FDX_SYM_NONE", FDX_SYM_NONE);
    check!("FDX_DTYPE_NONE", FDX_DTYPE_NONE);
    check!("FDX_BUFFER_INLINE", FDX_BUFFER_INLINE);
    check!("FDX_BUFFER_NONE", FDX_BUFFER_NONE);
    check!("FDX_AFFINE_MAX_TERMS", FDX_AFFINE_MAX_TERMS);
    check!("FDX_QUANT_NONE", FDX_QUANT_NONE);
    check!("FDX_QUANT_AFFINE_BLOCK", FDX_QUANT_AFFINE_BLOCK);
    check!("FDX_DTYPE_F8E8M0", FDX_DTYPE_F8E8M0);
    check!("FDX_GGML_BF16", FDX_GGML_BF16);
    check!("FDX_BACKEND_METAL", FDX_BACKEND_METAL);
}

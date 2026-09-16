// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-314 spatial-convert POPULATION check.
//!
//! #190 deleted the vision CHANNEL asserts (`conv2d` guards them per-dim). PR2
//! CONVERTS the vision SPATIAL asserts (`dims[2]/dims[3] == image_size`) to typed
//! declines, because their only downstream guard is the patch reshape to a
//! CONFIG-derived `num_patches` — which validates the patch COUNT, not the
//! per-axis size, so a count-preserving resize would pass silently.
//!
//! A representative behavioural born-red (`lazy_vit::count_preserving_spatial_violation_declined`)
//! proves the decline is RIGHT. This test proves the change REACHED every site —
//! the disjoint failure a single representative cannot see (a missed file leaves
//! its assert AND shorts the count; a green representative says nothing about it).
//!
//! Two reconciling assertions on disjoint defects:
//!   - ZERO remaining production spatial ASSERTS (`dims[2/3], cfg.image_size` in an
//!     assert-arg position) — fires if a site was missed or one is reintroduced.
//!   - Exactly 17 spatial DECLINES (`dims[2/3] != cfg.image_size`) — a literal count
//!     committed in advance, so the check RECONCILES rather than merely reports
//!     (17 entry points across 11 files: two axes each, one logical check).
//!
//! Boundary detection is anchored on the `#[cfg(test)]` whose NEXT line declares a
//! `mod` — NOT the first `#[cfg(test)]`, which lands on a test-only `use` and would
//! scan a truncated prefix that passes vacuously. Each file's scanned extent is
//! printed so a truncated read cannot masquerade as a total.
//!
//! SABOTAGE-VERIFIED (recorded so the next reader need not re-derive it): re-introducing
//! a single production `assert_eq!(dims[2], cfg.image_size)` takes `remaining_asserts` to
//! 1 and fails this test. So the zero-remaining arm has been SEEN to be non-zero for the
//! reason it exists — an assert-zero check that has only ever passed is indistinguishable
//! from one pointed at an empty region.
//!
//! METHOD NOTE (`a-source-scan-must-not-be-inside-what-it-scans`): the assert detector is
//! deliberately assert-CONTEXT-aware. A naive operand match (`dims[N], cfg.image_size`) is
//! ALSO satisfied by the DECLINE's own `format!` args line (`dims[2], dims[3],
//! cfg.image_size,`) — the guard's own text satisfying the guard's own detector, which
//! made the first run report 17 false "asserts". The fix requires the match to be in
//! assert context (`assert` on the line, or an `assert_eq!(` opener above), which the
//! decline's format-string predecessor is not.

const FILES: &[&str] = &[
    "lazy_beit.rs",
    "lazy_blip_vision.rs",
    "lazy_clip.rs",
    "lazy_dinov2.rs",
    "lazy_dinov2reg4.rs",
    "lazy_eva2.rs",
    "lazy_moondream.rs",
    "lazy_paddleocr_vl_vision.rs",
    "lazy_pixtral.rs",
    "lazy_siglip.rs",
    "lazy_vit.rs",
];

const EXPECTED_DECLINES: usize = 17;

/// The production region: everything before the test module. The boundary is the
/// `#[cfg(test)]` immediately followed by a `mod` declaration.
fn production_prefix<'a>(src: &'a str, path: &str) -> Vec<&'a str> {
    let lines: Vec<&str> = src.lines().collect();
    let boundaries: Vec<usize> = (0..lines.len())
        .filter(|&i| {
            lines[i].contains("#[cfg(test)]")
                && lines
                    .get(i + 1)
                    .is_some_and(|n| n.trim_start().starts_with("mod "))
        })
        .collect();
    // 0 test modules => scan whole file; 1 => scan up to it; >=2 => a finding
    // about the file, never a silent fallback.
    assert!(
        boundaries.len() <= 1,
        "{path}: found {} `#[cfg(test)] mod` boundaries; expected 0 or 1 — a finding about the file, \
         not the check",
        boundaries.len()
    );
    let end = boundaries.first().copied().unwrap_or(lines.len());
    eprintln!(
        "[pop-check] {path}: scanned production lines 1..={end} of {}",
        lines.len()
    );
    lines[..end].to_vec()
}

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// Count spatial ASSERT macros in the production prefix. A line referencing a
/// spatial axis (`dims[2]`/`dims[3]`) and the config size (`image_size`/`img_size`)
/// counts as an assert only when it is in ASSERT context: `assert` on the same line
/// (single-line macro) or an `assert_eq!(` / `assert!(` opener on the line above
/// (multi-line macro). This is what distinguishes a surviving assert from the
/// DECLINE's own `format!` args line (`dims[2], dims[3], cfg.image_size,`), whose
/// line above is the format string, not an assert opener — the false positive that
/// an operand-only match produced.
fn count_spatial_asserts(prefix: &[&str]) -> usize {
    let mut n = 0;
    for i in 0..prefix.len() {
        let line = prefix[i];
        if is_comment(line) {
            continue;
        }
        let has_axis = line.contains("dims[2]") || line.contains("dims[3]");
        let has_field = line.contains("image_size") || line.contains("img_size");
        if !has_axis || !has_field {
            continue;
        }
        // Single-line assert: `assert_eq!(dims[2], cfg.image_size)`.
        if line.contains("assert") {
            n += 1;
            continue;
        }
        // Multi-line assert operand: the line above is the macro opener. The decline
        // args line's predecessor is a format string, so it is excluded here.
        if i > 0 {
            let prev = prefix[i - 1].trim();
            if prev.ends_with("assert_eq!(") || prev.ends_with("assert!(") {
                n += 1;
            }
        }
    }
    n
}

/// A spatial DECLINE line: `dims[2] != cfg.image_size` (the `if` guard).
fn is_spatial_decline(line: &str) -> bool {
    if is_comment(line) {
        return false;
    }
    line.contains("!= cfg.image_size") || line.contains("!= cfg.img_size")
}

#[test]
fn spatial_asserts_are_fully_converted_to_declines() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/");
    let mut remaining_asserts = 0usize;
    let mut declines = 0usize;

    for file in FILES {
        let path = format!("{base}{file}");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let prefix = production_prefix(&src, file);
        let file_asserts = count_spatial_asserts(&prefix);
        let file_declines = prefix.iter().filter(|l| is_spatial_decline(l)).count();
        eprintln!("[pop-check] {file}: {file_asserts} asserts, {file_declines} declines");
        remaining_asserts += file_asserts;
        declines += file_declines;
    }

    // Positive control / reconciliation: the scanner reads real content and every
    // site converted. A count committed in advance is what caught the paddleocr
    // miss (17 expected vs 16 done) before this check existed.
    assert_eq!(
        declines, EXPECTED_DECLINES,
        "expected {EXPECTED_DECLINES} spatial declines across the fixed-resolution vision entries, \
         found {declines} — a short count means a site was missed; a long count means one was \
         added or double-counted"
    );
    // Zero-remaining: fires on a missed site or a later reintroduction.
    assert_eq!(
        remaining_asserts, 0,
        "found {remaining_asserts} remaining production spatial assert(s) (`dims[2/3], cfg.image_size`); \
         every fixed-resolution vision entry must decline, not assert"
    );
}

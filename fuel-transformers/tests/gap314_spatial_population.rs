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
//! The file set is DERIVED FROM A PROPERTY, not hand-listed (see `vision_model_files`):
//! any `models/lazy_*.rs` carrying a spatial dim check is auto-included, so a 12th
//! vision model cannot drift past this test invisibly — the hand-list failure the
//! portfolio rule warns about.
//!
//! Three reconciling assertions on disjoint defects:
//!   - The derived set is NON-EMPTY (and is printed) — fires if the scanner or the
//!     models directory moved, which would make both counts vacuously satisfied.
//!   - ZERO remaining production spatial ASSERTS (`dims[2/3], cfg.image_size` in an
//!     assert-arg position) — fires if a site was missed or one is reintroduced.
//!   - Exactly 18 spatial DECLINES (`dims[2/3] != <cfg>.image_size`) — a literal count
//!     committed in advance, so the check RECONCILES rather than merely reports
//!     (18 entry points across 12 files). Was 17/11 until the derived set surfaced
//!     lazy_llava.rs, a vision model the hand-list missed — the drift class made
//!     concrete on the first run of the property derivation.
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
//! ⚠️ WHY THE SET IS DERIVED, NOT LISTED — AND WHY NO SABOTAGE CAN SUBSTITUTE: no sabotage
//! placed INSIDE a population can validate that population's BOUNDARY, because validating
//! the boundary means placing the probe where the instrument does not look. The sabotage
//! above proves the zero-arm fires for the reason it exists; it says NOTHING about whether
//! the set is complete. That is the job of `vision_model_files` deriving the set from a
//! property: a hand-list is not merely staleness-prone, it can be INCOMPLETE AT BIRTH and
//! indistinguishable from complete by every check that ranges over it — which is exactly
//! what happened here. The 11-file list was wrong when authored (it omitted lazy_llava.rs),
//! both count arms agreed, the sabotage passed, and a file was missing the whole time. The
//! derived set caught it on its first run. This paragraph is the reason nobody should ever
//! revert the derivation to a list.
//!
//! METHOD NOTE (`a-source-scan-must-not-be-inside-what-it-scans`): the assert detector is
//! deliberately assert-CONTEXT-aware. A naive operand match (`dims[N], cfg.image_size`) is
//! ALSO satisfied by the DECLINE's own `format!` args line (`dims[2], dims[3],
//! cfg.image_size,`) — the guard's own text satisfying the guard's own detector, which
//! made the first run report 17 false "asserts". The fix requires the match to be in
//! assert context (`assert` on the line, or an `assert_eq!(` opener above), which the
//! decline's format-string predecessor is not.

/// The vision models in scope, DERIVED FROM A PROPERTY rather than a hand-written list:
/// every `models/lazy_*.rs` whose PRODUCTION prefix carries a spatial `image_size` /
/// `img_size` dim check — a converted decline, or (if one ever regresses) an assert. A
/// 12th vision model added later is auto-included, so it cannot drift past this test
/// invisibly. A hand-written set can only exclude what its author thought of, and what it
/// misses is silent (portfolio CLAUDE.md); a `#[test]` reads only source it will not
/// write, so enumerating the on-disk `src/models` here carries no cross-worktree hazard.
fn vision_model_files() -> Vec<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir}: {e}")) {
        let path = entry.unwrap().path();
        let Some(name) = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(str::to_string)
        else {
            continue;
        };
        if !name.starts_with("lazy_") || !name.ends_with(".rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        if file_has_spatial_check(&name, &src) {
            out.push(name);
        }
    }
    out.sort();
    out
}

/// True when `name`'s PRODUCTION prefix carries a spatial dim check (a converted decline,
/// or an assert if one regressed). The cheap `image_size`/`img_size` text pre-filter keeps
/// `production_prefix` (which hard-fails on a malformed test-module boundary) off the ~90
/// non-vision model files.
fn file_has_spatial_check(name: &str, src: &str) -> bool {
    if !src.contains("image_size") && !src.contains("img_size") {
        return false;
    }
    let prefix = production_prefix(src, name);
    count_spatial_asserts(&prefix) > 0 || prefix.iter().any(|l| is_spatial_decline(l))
}

// 18 spatial-decline entry points across 12 files. Was 17 across 11: the property
// derivation (see `vision_model_files`) surfaced lazy_llava.rs on its first run — a 12th
// vision model the hand-list census missed, whose embedded clip vision tower carried the
// identical spatial asserts. That is the drift the derived set exists to catch, and it
// caught it: a hand-list would have stayed at 11 and 17 forever, silent.
const EXPECTED_DECLINES: usize = 18;

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

/// Count spatial ASSERT macros in the production prefix — the sites that must be ZERO
/// after conversion. Delegates the per-line decision to `is_spatial_assert_at`.
fn count_spatial_asserts(prefix: &[&str]) -> usize {
    (0..prefix.len())
        .filter(|&i| is_spatial_assert_at(prefix, i))
        .count()
}

/// A `dims[2]/[3]` vs `image_size/img_size` reference in ASSERT context at line `i`:
/// `assert` on the same line (single-line macro), or an `assert_eq!(` / `assert!(` opener
/// on the line above (multi-line macro). This is what distinguishes a surviving assert from
/// the DECLINE's own `format!` args line (`dims[2], dims[3], cfg.image_size,`), whose line
/// above is the format string, not an assert opener — the false positive an operand-only
/// match produced.
fn is_spatial_assert_at(prefix: &[&str], i: usize) -> bool {
    let line = prefix[i];
    if is_comment(line) || !mentions_spatial_axis_and_field(line) {
        return false;
    }
    line.contains("assert") || opener_above_is_assert(prefix, i)
}

/// The line above `i` opens an assert macro (`assert_eq!(` / `assert!(`).
fn opener_above_is_assert(prefix: &[&str], i: usize) -> bool {
    if i == 0 {
        return false;
    }
    let prev = prefix[i - 1].trim();
    prev.ends_with("assert_eq!(") || prev.ends_with("assert!(")
}

/// The line names a spatial axis (`dims[2]`/`dims[3]`) AND the config size field
/// (`image_size`/`img_size`). Shared by the assert counter and the decline detector.
fn mentions_spatial_axis_and_field(line: &str) -> bool {
    let has_axis = line.contains("dims[2]") || line.contains("dims[3]");
    let has_field = line.contains("image_size") || line.contains("img_size");
    has_axis && has_field
}

/// A spatial DECLINE line: the `if` guard `dims[2] != <cfg>.image_size` (or `img_size`).
/// Binding-agnostic on the config name — some towers bind it `cfg`, llava's embedded clip
/// tower binds it `v_cfg` — so the match is `!=` together with a spatial axis and the size
/// field, rather than a literal `!= cfg.image_size`. This does not collide with the assert
/// form (`assert_eq!(dims[2], cfg.image_size)` has no `!=`) nor the decline's own `format!`
/// args line (`dims[2], dims[3], cfg.image_size,` has no `!=`).
fn is_spatial_decline(line: &str) -> bool {
    !is_comment(line) && line.contains("!=") && mentions_spatial_axis_and_field(line)
}

#[test]
fn spatial_asserts_are_fully_converted_to_declines() {
    let base = concat!(env!("CARGO_MANIFEST_DIR"), "/src/models/");
    let mut remaining_asserts = 0usize;
    let mut declines = 0usize;

    let files = vision_model_files();
    // The derivation must find the population, or every downstream assertion is
    // vacuous: an empty set means BOTH counts sum to 0, `remaining_asserts == 0`
    // passes trivially, and only the `declines == 17` arm would notice — so pin the
    // set is non-empty explicitly, and print it so a wrong-but-non-empty derivation
    // (e.g. the scanner drifted) is legible rather than hidden behind a count.
    assert!(
        !files.is_empty(),
        "derived vision-model set is EMPTY — the property query (models/lazy_*.rs mentioning \
         image_size/img_size AND carrying a spatial dim check) found nothing; the scanner or the \
         models directory moved, and every count below would be vacuously satisfied"
    );
    eprintln!(
        "[pop-check] derived {} vision file(s) by property: {files:?}",
        files.len()
    );

    for file in &files {
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

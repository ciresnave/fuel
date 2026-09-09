// SPDX-License-Identifier: MIT OR Apache-2.0
//! The FDX spec's §8 numbered validator list and `validate.rs`'s `check_v*`
//! functions must be the SAME SET of V-numbers. This refuses a divergence
//! instead of reminding someone to keep them in sync.
//!
//! # Why this exists
//!
//! FDX is an **interchange format with external producers** (PyTorch / JAX /
//! CuPy build sidecars to it). `docs/specs/dlpack-extension.md` §8 is a numbered
//! list — `N. **VN — …**` — and it is the contract an external implementer
//! builds to. If `validate.rs` grows a validator that is NOT in that list, Fuel
//! rejects sidecars the spec (read literally) blesses: a conformance divergence,
//! with the spec on the producer's side. If §8 grows a clause with no validator
//! function, it is an unenforced promise. Either way the two must move together.
//!
//! Measured at `16577dc1`: spec §8 declares V1..V21 (21 distinct), and
//! `validate.rs` defines `check_v1`..`check_v21` (21 distinct — the sub-lettered
//! `check_v21{a,c,d,e}` and the private `check_v13_{base,buffers}` helper arms
//! collapse to their V-number). The sets are identical.
//!
//! # ⚠️ Why prose was not enough
//!
//! The correspondence lived only in `§3 honesty + §9.2 + a definition line` for
//! the general meaning-bearing rule — supported, but not STATED in the numbered
//! list where an implementer stands. A validator outside §8 does not dangle: it
//! rejects a real, spec-valid sidecar and reads as correct to everyone except
//! the producer it rejected. Nothing checks it; this file does.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Walk up from `CARGO_MANIFEST_DIR` until a `Cargo.toml` containing
/// `[workspace]` is found; that directory is the workspace root.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.exists()
            && let Ok(txt) = std::fs::read_to_string(&manifest)
            && txt.contains("[workspace]")
        {
            return dir;
        }
        if !dir.pop() {
            panic!("could not find a Cargo.toml containing [workspace] above CARGO_MANIFEST_DIR");
        }
    }
}

/// The distinct V-numbers declared as numbered clauses in spec §8.
///
/// Bounded by the `## 8.` heading and the next `## ` heading: `VN` tokens also
/// appear as cross-references in §3/§6/§9 prose (`(V19)`, `mirrors V8`), and a
/// document-wide match would sweep them in. Only a clause OPENER — a line that,
/// after trimming, begins `<n>. **V<m>` — counts; a mid-body `**V16**` reference
/// does not.
fn spec_v_numbers(spec: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let mut inside = false;
    for line in spec.lines() {
        if line.starts_with("## 8.") {
            inside = true;
        } else if inside && line.starts_with("## ") {
            break;
        } else if inside && let Some(n) = clause_v_number(line) {
            out.insert(n);
        }
    }
    out
}

/// The V-number if `line` OPENS a §8 clause (`<n>. **V<m> — …`), returning `m`
/// — the validator's own number, not the list index. They agree today; the
/// validator number is the one that must match a `check_v<m>` function.
fn clause_v_number(line: &str) -> Option<u32> {
    let t = line.trim_start();
    let idx_digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if idx_digits.is_empty() {
        return None;
    }
    let rest = &t[idx_digits.len()..];
    const MARKER: &str = ". **V";
    if !rest.starts_with(MARKER) {
        return None;
    }
    let after = &rest[MARKER.len()..];
    let v_digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    v_digits.parse::<u32>().ok()
}

/// The distinct V-numbers implemented as `fn check_v<N>…` in `validate.rs`.
///
/// Matches on the `fn check_v` keyword so a `// V22` comment or a doc mention is
/// not miscounted; the trailing digits after `check_v` are the V-number, so
/// `check_v21a_gather_buffers` → 21 and `check_v13_base` → 13.
fn code_v_numbers(src: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    const NEEDLE: &str = "fn check_v";
    let mut rest = src;
    while let Some(i) = rest.find(NEEDLE) {
        rest = &rest[i + NEEDLE.len()..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(n) = digits.parse::<u32>() {
            out.insert(n);
        }
    }
    out
}

/// The V-number if `call_name` is a numbered validator (`check_v<N>…`); `None`
/// for an UNNUMBERED validator such as `check_meaning_bearing_implies_ext`, and
/// `None` for `check_v` not followed by a digit. This is the discriminator arm 3
/// turns on: a wired validator that returns `None` has no place in the §8
/// numbered list and is invisible to the set-join arms (1 and 2).
fn validator_number(call_name: &str) -> Option<u32> {
    let rest = call_name.strip_prefix("check_v")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

/// Every `check_<ident>(` CALL inside the body of the orchestrator whose
/// signature begins with `sig`. The body is taken by brace-matching from the
/// first `{` after the signature — enumerating the WIRED set from the
/// orchestrator itself, never from a `check_v*` name pattern (which would define
/// the population as the numbered validators and be blind to an unnumbered one
/// wired in, the exact defect this gate exists for).
fn orchestrator_check_calls(src: &str, sig: &str) -> Vec<String> {
    let start = src
        .find(sig)
        .unwrap_or_else(|| panic!("orchestrator `{sig}` not found"));
    let after = &src[start..];
    let open = after
        .find('{')
        .unwrap_or_else(|| panic!("no body brace after `{sig}`"));
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = open;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let body = &after[open..=end];
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(i) = rest.find("check_") {
        let tail = &rest[i..];
        let ident: String = tail
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if tail[ident.len()..].starts_with('(') {
            out.push(ident.clone());
        }
        rest = &tail[ident.len()..];
    }
    out
}

fn read(root: &std::path::Path, rel: &str) -> String {
    std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

#[test]
fn spec_section8_and_validator_fns_are_the_same_set() {
    let root = workspace_root();
    let spec = read(&root, "docs/specs/dlpack-extension.md");
    let src = read(&root, "fuel-ir/src/dlpack/validate.rs");

    let spec_v = spec_v_numbers(&spec);
    let code_v = code_v_numbers(&src);

    // Foundation: a broken parser returning few/none makes this whole file
    // vacuous, and a vacuous green is the failure this gate exists to prevent.
    assert!(
        spec_v.len() >= 15,
        "only {} §8 V-clauses parsed — the spec parser is broken, not §8",
        spec_v.len()
    );
    assert!(
        code_v.len() >= 15,
        "only {} check_v* fns parsed — the source parser is broken",
        code_v.len()
    );

    let spec_only: BTreeSet<u32> = spec_v.difference(&code_v).copied().collect();
    let code_only: BTreeSet<u32> = code_v.difference(&spec_v).copied().collect();
    assert!(
        spec_only.is_empty() && code_only.is_empty(),
        "FDX §8 validator list and validate.rs check_v* fns diverged.\n  \
         §8 clauses with NO validator fn (unenforced promise — add the fn): {spec_only:?}\n  \
         validator fns with NO §8 clause (rejects spec-valid sidecars — add the clause): {code_only:?}\n\
         An interchange format's numbered §8 list is the contract external producers build to."
    );

    // ARM 3 (the load-bearing one): every validator WIRED INTO an orchestrator
    // body must be a numbered check_v<N>. Enumerated from the body, NOT a name
    // pattern — an unnumbered validator (e.g. a `check_meaning_bearing_implies_ext`
    // wired into validate()) is invisible to arms 1/2 because it is not check_v*,
    // yet it rejects sidecars §8 blesses. Born green (all 21 wired validators are
    // numbered); an unnumbered arm reddens it until it is registered under a V-number.
    let validate_calls = orchestrator_check_calls(&src, "pub fn validate(");
    assert!(
        validate_calls.len() >= 15,
        "validate() body parse found only {} check_ calls — the body extractor is broken, \
         and a broken extractor makes arm 3 vacuous",
        validate_calls.len()
    );
    let mut unnumbered: Vec<String> = Vec::new();
    for sig in ["pub fn validate(", "pub fn validate_realize("] {
        for call in orchestrator_check_calls(&src, sig) {
            if validator_number(&call).is_none() {
                unnumbered.push(call);
            }
        }
    }
    unnumbered.sort();
    unnumbered.dedup();
    assert!(
        unnumbered.is_empty(),
        "validator(s) wired into an orchestrator body with NO V-number: {unnumbered:?}\n\
         An unnumbered validator rejects sidecars §8 (read literally) blesses AND is invisible \
         to the set-join arms above. Register it under a V-number and add its §8 clause."
    );
}

// FOUNDATION CHECK, FOUR ARMS. Each asserts a DIFFERENT failure is visible on
// synthetic inputs, so the parser+join keeps discriminating on every run even
// after the real corpus changes — the retained-sabotage sibling of the born
// green above. A one-arm gate cannot tell "correct" from "asleep".
#[test]
fn the_join_discriminates() {
    // (1) a §8 clause with NO validator fn is caught (spec-only).
    let spec =
        "## 8. Validation\n1. **V1 — a**\n2. **V2 — b**\n99. **V99 — orphan clause**\n## 9. next\n";
    let src = "pub fn check_v1() {}\npub fn check_v2() {}\n";
    let sv = spec_v_numbers(spec);
    let cv = code_v_numbers(src);
    assert!(
        sv.contains(&99) && !cv.contains(&99),
        "a §8 clause with no validator fn was not detected"
    );

    // (2) a validator fn with NO §8 clause is caught (code-only) — the V22 case.
    let src2 = "pub fn check_v1() {}\npub fn check_v22_meaning_bearing() {}\n";
    let cv2 = code_v_numbers(src2);
    assert!(
        cv2.contains(&22) && !sv.contains(&22),
        "a check_v* fn with no §8 clause was not detected"
    );

    // (3) sub-lettered fns collapse to their V-number, and a private helper arm
    //     maps to a number the join already has (so it adds nothing spurious).
    let src3 = "pub fn check_v21a_gather_buffers() {}\nfn check_v13_base() {}\n";
    assert_eq!(
        code_v_numbers(src3),
        [13u32, 21].into_iter().collect::<BTreeSet<u32>>(),
        "sub-lettered / helper check_v fns did not collapse to their V-number"
    );

    // (4) the section bound holds: a mid-body `**V16**` reference is NOT a clause
    //     opener, and a numbered list in a LATER section (§9) is out of scope.
    let bounded = "## 8. Validation\n1. **V1 — a**\n   well-formedness holds via **V16** here\n## 9. Policies\n2. **V2 — not a validator**\n";
    assert_eq!(
        spec_v_numbers(bounded),
        [1u32].into_iter().collect::<BTreeSet<u32>>(),
        "the scan counted a mid-body reference or ran past the §8 boundary"
    );

    // (5) a comment mentioning a V-number is NOT a validator fn.
    assert!(
        code_v_numbers("// see V22 for the meaning-bearing rule\nlet check_v22 = 3;\n").is_empty(),
        "a non-fn `check_v` token was miscounted as a validator"
    );

    // (6) ARM 3: an UNNUMBERED validator wired into an orchestrator body is
    //     caught, and body extraction / numbering both discriminate.
    let orch = "pub fn validate(s: &S) -> R {\n    \
        check_v1_header(s)?;\n    \
        check_meaning_bearing_implies_ext(s)?;\n    \
        Ok(())\n}\n";
    assert_eq!(
        orchestrator_check_calls(orch, "pub fn validate("),
        vec![
            "check_v1_header".to_string(),
            "check_meaning_bearing_implies_ext".to_string()
        ],
        "orchestrator body call extraction is wrong"
    );
    assert!(
        validator_number("check_v1_header").is_some(),
        "a numbered validator read as unnumbered"
    );
    assert!(
        validator_number("check_meaning_bearing_implies_ext").is_none(),
        "an UNNUMBERED validator read as numbered — arm 3 would be blind to the exact defect it exists for"
    );
    assert!(
        validator_number("check_various_things").is_none(),
        "`check_v` not followed by a digit must not count as a numbered validator"
    );
}

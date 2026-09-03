//! **`docs/method-rules.md` and `CLAUDE.md` must agree, in BOTH directions.**
//!
//! WHY THIS EXISTS. Eight method-rules sections once carried no `CLAUDE.md`
//! index line at all — landed, and unreachable from the file that loads into
//! every session, which is close to not having landed. Separately, an index
//! line was nearly shipped pointing at `justification-scope-mismatch`, which is
//! a *memory file*, not a method-rules section: a dangling citation inside the
//! rule about citations.
//!
//! ## Why no single-file audit could find either
//!
//! **Neither artifact is wrong.** From `method-rules.md` every rule is present
//! and consistent; from `CLAUDE.md` nothing announces that anything is missing.
//! The defect exists only in the RELATION between them, so a reader auditing
//! either file — repeatedly, carefully — cannot see it. **A two-artifact
//! invariant needs a two-artifact check.**
//!
//! ## The two directions fail differently, which is why both arms exist
//!
//! ```text
//! section without index line  ->  the rule is UNREACHABLE. Silent.
//!                                 Costs you that one rule.
//! index line without section  ->  the rule appears to EXIST and does not.
//!                                 A reader follows a link to nothing and
//!                                 concludes the corpus is unmaintained.
//! ```
//!
//! The second is worse per instance, because it damages trust in every other
//! link — so the arm that is easier to forget is the one that costs more.
//!
//! ## Two tests, not one test with two asserts
//!
//! A test panics at its first failing assert, so a single test would let one
//! sabotage prove one arm while the other stayed undemonstrated. **Each arm is
//! its own test and each was sabotaged separately.**
//!
//! ## Anchor form
//!
//! Arm B keys on the ANCHOR IN THE LINK TARGET (`…method-rules.md#slug`), not
//! on the backticked link text. The two can drift apart, and only the target is
//! what a reader actually follows.
//!
//! Self-matching is solved by SCOPE: this scanner reads two markdown files and
//! is itself Rust, so it cannot match its own examples by construction.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file()
            && std::fs::read_to_string(&manifest).is_ok_and(|s| s.contains("[workspace]"))
        {
            return dir;
        }
        assert!(
            dir.pop(),
            "no Cargo.toml with [workspace] above CARGO_MANIFEST_DIR"
        );
    }
}

/// The slug alphabet, in ONE place.
///
/// It was written out three times -- in `sections`, `cited_anchors` and the
/// anchor scanner -- which is a divergence generator: a slug convention that
/// changes in two of three sites fails by SILENTLY NOT MATCHING, so the gate
/// would go quiet rather than red. Naming it also drops the scanner under
/// Codacy's complexity limit, but that is the smaller reason.
fn is_slug_char(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'
}

/// `## slug` headings, lowercase-kebab only (the method-rules convention).
fn sections(method_rules: &str) -> Vec<String> {
    method_rules
        .lines()
        .filter_map(|l| l.strip_prefix("## "))
        .map(str::trim)
        .filter(|s| !s.is_empty() && s.chars().all(is_slug_char))
        .map(str::to_owned)
        .collect()
}

/// Anchors CLAUDE.md points at, taken from the LINK TARGET.
fn cited_anchors(claude: &str) -> Vec<String> {
    const NEEDLE: &str = "method-rules.md#";
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = claude[from..].find(NEEDLE) {
        let at = from + rel + NEEDLE.len();
        let anchor: String = claude[at..]
            .chars()
            .take_while(|c| is_slug_char(*c))
            .collect();
        if !anchor.is_empty() {
            out.push(anchor);
        }
        from = at;
    }
    out
}

/// ARM A: a section nobody indexed is unreachable from the session-loaded file.
fn unindexed(method_rules: &str, claude: &str) -> Vec<String> {
    sections(method_rules)
        .into_iter()
        .filter(|s| !claude.contains(s.as_str()))
        .collect()
}

/// ARM C: a section with no ANCHOR LINK is reachable in prose and not by a
/// link a reader can follow.
///
/// Arm A accepts ANY mention of the name, and arm B only validates anchors that
/// already exist -- so a link whose TEXT names the section and whose TARGET
/// omits the `#anchor` satisfies both and still strands the reader at the top of
/// the file. Found 2026-09-02 with two live instances.
fn unanchored(method_rules: &str, claude: &str) -> Vec<String> {
    let cited = cited_anchors(claude);
    sections(method_rules)
        .into_iter()
        .filter(|s| !cited.contains(s))
        .collect()
}

/// ARM D: a link to the FILE rather than to a SECTION strands the reader.
///
/// Distinct from arm C and neither implies the other: arm C asks whether a
/// SECTION is reachable at all, so a rule cited five times with four anchors
/// passes it while the fifth citation still lands the reader at the top of a
/// 1,600-line file. Counts occurrences of the file path NOT followed by `#`.
///
/// KNOWN BOUND, named rather than mitigated: this also forbids a legitimate
/// WHOLE-FILE link -- `[the rules corpus](docs/method-rules.md)` -- which
/// promises no section and therefore breaks no promise. Measured 2026-09-02:
/// ZERO such links exist in any tracked `.md`, and both real instances named
/// a section in the link TEXT. Refining this to "flag only when the text
/// names a slug" would be machinery for a case that does not occur. Whoever
/// writes the first whole-file link gets a clear red and a REAL instance to
/// reason about, which is a better basis for relaxing this than a
/// hypothetical is today.
fn anchorless_links(claude: &str) -> usize {
    const NEEDLE: &str = "](docs/method-rules.md";
    let mut n = 0usize;
    let mut from = 0usize;
    while let Some(rel) = claude[from..].find(NEEDLE) {
        let at = from + rel + NEEDLE.len();
        // `](docs/method-rules.md)` -- no fragment -- is the defect.
        if claude[at..].starts_with(')') {
            n += 1;
        }
        from = at;
    }
    n
}

/// ARM B: a link to nothing damages trust in every other link.
fn dangling(method_rules: &str, claude: &str) -> Vec<String> {
    let have = sections(method_rules);
    let mut out: Vec<String> = cited_anchors(claude)
        .into_iter()
        .filter(|a| !have.contains(a))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// ARM E: an intra-document `](#anchor)` in method-rules.md pointing at no
/// section of method-rules.md.
///
/// Arm B validates CLAUDE.md -> method-rules anchors. NOTHING validated
/// method-rules -> method-rules, so the file the gate is about was the one
/// place its cross-references went unchecked. Measured 2026-09-03 at
/// `2b366804`: 36 intra-document links, 27 unique, 20 resolving (the control
/// that the query works) and SIX dangling -- every one a MEMORY-file name with
/// no section here. The hole was recorded in this corpus by the lane that found
/// it, with its measurement, and went unclosed.
///
/// # Code spans are excluded, and it is load-bearing rather than tidiness
///
/// A naive scanner reports SEVEN. The seventh is `](#x)` sitting inside an
/// INLINE CODE SPAN in the very sentence of this corpus that documents this
/// hole -- so the first false positive of a naive version is the documentation
/// of the problem it exists to catch. Measured: 0 intra links inside fenced
/// blocks, 1 inside an inline code span, 35 live.
///
/// The module header says self-matching is "solved by SCOPE" -- the scanner is
/// Rust, the subjects are markdown. That protects the scanner from its OWN
/// examples. It does not protect it from the SUBJECT's examples, which is a
/// different problem and is why this function parses code spans at all.
/// Anchors a single markdown line links to, EXCLUDING inline code spans.
///
/// Split out of `intra_dangling` so the code-span parsing is one unit with one
/// job -- and so the negative control can aim at it directly rather than at a
/// function that also walks fences.
fn anchors_in_line(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut in_code = false;
    for i in 0..chars.len() {
        if chars[i] == '`' {
            in_code = !in_code;
            continue;
        }
        if in_code || chars[i] != ']' {
            continue;
        }
        if chars.get(i + 1) != Some(&'(') || chars.get(i + 2) != Some(&'#') {
            continue;
        }
        let anchor: String = chars[i + 3..]
            .iter()
            .copied()
            .take_while(|c| is_slug_char(*c))
            .collect();
        if !anchor.is_empty() {
            out.push(anchor);
        }
    }
    out
}

fn intra_dangling(method_rules: &str) -> Vec<String> {
    let have = sections(method_rules);
    let mut out: Vec<String> = Vec::new();
    let mut fenced = false;
    for line in method_rules.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        out.extend(
            anchors_in_line(line)
                .into_iter()
                .filter(|a| !have.contains(a)),
        );
    }
    out.sort();
    out.dedup();
    out
}

fn read_pair() -> (String, String) {
    let root = workspace_root();
    let mr = std::fs::read_to_string(root.join("docs/method-rules.md")).expect("method-rules.md");
    let cm = std::fs::read_to_string(root.join("CLAUDE.md")).expect("CLAUDE.md");
    (mr, cm)
}

#[test]
fn arm_a_every_method_rule_is_reachable_from_claude_md() {
    let (mr, cm) = read_pair();
    let missing = unindexed(&mr, &cm);
    assert!(
        missing.is_empty(),
        "{} method-rules section(s) have NO mention in CLAUDE.md, so they are unreachable from the file that loads into every session — landed and invisible, which is close to not having landed. If the section carries its intended index line in a blockquote, move that across; otherwise compose one in the house style -- NOT every section has one. Missing: {missing:?}",
        missing.len(),
    );
}

#[test]
fn arm_b_every_claude_md_anchor_resolves_to_a_real_section() {
    let (mr, cm) = read_pair();
    let broken = dangling(&mr, &cm);
    assert!(
        broken.is_empty(),
        "CLAUDE.md links to {} method-rules anchor(s) that do not exist. A link to nothing is \
         worse per instance than a missing link: the rule appears to EXIST, and a reader who \
         follows it concludes the corpus is unmaintained — which damages trust in every other \
         link. Check whether the target is a MEMORY file rather than a method-rules section. \
         Dangling: {broken:?}",
        broken.len(),
    );
}

/// Positive control for ARM A — a blind scanner is indistinguishable from a
/// clean corpus, so the arm above proves nothing without this.
#[test]
fn arm_c_every_section_has_a_link_whose_target_carries_its_anchor() {
    let (mr, cm) = read_pair();
    let unanchored = unanchored(&mr, &cm);
    assert!(
        unanchored.is_empty(),
        "{} method-rules section(s) have NO link in CLAUDE.md whose TARGET carries their \
         anchor. Arm A is satisfied by a bare NAME and arm B only checks anchors that already \
         exist, so a citation like `[`…md` § `slug`](docs/method-rules.md)` -- name in the TEXT, \
         no `#anchor` in the TARGET -- passes both and strands the reader at the top of a \
         1,600-line file. It READS as a working citation and is not one. Fix the TARGET, not \
         the text. Unanchored: {unanchored:?}",
        unanchored.len(),
    );
}

#[test]
fn arm_d_every_method_rules_link_carries_an_anchor() {
    let (_mr, cm) = read_pair();
    let n = anchorless_links(&cm);
    assert_eq!(
        n, 0,
        "{n} CLAUDE.md link(s) target `docs/method-rules.md` with NO `#anchor`. Distinct from \
         arm C: a rule cited several times with anchors passes THAT while one bad citation \
         still lands the reader at the top of a 1,600-line file. Add the fragment to the \
         TARGET; the link text is not what a reader follows.",
    );
}

#[test]
fn arm_d_scanner_discriminates_anchored_from_anchorless() {
    assert_eq!(
        anchorless_links("see [`x`](docs/method-rules.md) here"),
        1,
        "arm D's scanner missed a link with no fragment",
    );
    assert_eq!(
        anchorless_links("see [`x`](docs/method-rules.md#x) here"),
        0,
        "arm D's scanner flagged a correctly-anchored link",
    );
}

#[test]
fn arm_c_scanner_can_see_a_section_cited_without_an_anchor() {
    // The exact shape found in the wild: the slug is in the link TEXT and the
    // TARGET has no fragment. A scanner keyed on the text would call this fine.
    let mr = "## only-section\n\nbody\n";
    let cm = "see [`docs/method-rules.md` § `only-section`](docs/method-rules.md) for detail\n";
    assert_eq!(
        unanchored(mr, cm),
        vec!["only-section".to_string()],
        "arm C's scanner did not flag a section whose only citation omits the #anchor from the \
         link target -- the very shape it exists to catch",
    );
    // ...and it must NOT flag the same section once the target carries the anchor.
    let fixed = "see [`only-section`](docs/method-rules.md#only-section) for detail\n";
    assert!(
        unanchored(mr, fixed).is_empty(),
        "arm C's scanner flagged a correctly-anchored citation",
    );
}

#[test]
fn arm_a_scanner_can_see_an_unindexed_section() {
    let mr = "## alpha-rule\ntext\n\n## beta-rule\ntext\n";
    let cm = "we reference alpha-rule here and nothing else";
    assert_eq!(unindexed(mr, cm), vec!["beta-rule".to_string()]);
}

/// Positive control for ARM B, and it uses the real near-miss: an index line
/// pointing at `justification-scope-mismatch`, which is a memory file.
#[test]
fn arm_b_scanner_can_see_a_dangling_anchor() {
    let mr = "## alpha-rule\ntext\n";
    let cm =
        "see [`justification-scope-mismatch`](docs/method-rules.md#justification-scope-mismatch)";
    assert_eq!(
        dangling(mr, cm),
        vec!["justification-scope-mismatch".to_string()]
    );
}

/// Negative control: Arm B must key on the LINK TARGET, not the backticked
/// text. Text saying one thing while the target says another is precisely the
/// drift the anchor-form rule exists for.
#[test]
fn arm_b_keys_on_the_link_target_not_the_link_text() {
    let mr = "## real-section\ntext\n";
    let cm = "see [`totally-made-up-name`](docs/method-rules.md#real-section)";
    assert!(
        dangling(mr, cm).is_empty(),
        "resolving target -> real-section, so not dangling"
    );
}
/// ARM E: the gate's own file had six broken cross-references and nothing looked.
#[test]
fn arm_e_every_intra_document_anchor_resolves_to_a_real_section() {
    let (mr, _cm) = read_pair();
    let bad = intra_dangling(&mr);
    assert!(
        bad.is_empty(),
        "{} intra-document anchor(s) in docs/method-rules.md point at no section OF THAT FILE. Arm B checks CLAUDE.md -> method-rules; this checks method-rules -> method-rules, which was unchecked. A reader follows one of these and lands nowhere, which damages trust in every other link in the corpus. Usually the target is a MEMORY-file name with no section here: keep the name, drop the link. Dangling: {bad:?}",
        bad.len(),
    );
}

/// Positive control for ARM E, using one of the six real instances.
#[test]
fn arm_e_scanner_can_see_a_dangling_intra_anchor() {
    let mr = "## alpha-rule
text

see [`magnitude-is-not-impossibility`](#magnitude-is-not-impossibility)
";
    assert_eq!(
        intra_dangling(mr),
        vec!["magnitude-is-not-impossibility".to_string()]
    );
}

/// NEGATIVE CONTROL for ARM E, load-bearing rather than decorative.
///
/// The corpus documents this very hole using `](#x)` inside an inline code
/// span, and shows link syntax inside fenced blocks. A scanner without these
/// exclusions reports its own documentation as the first defect -- so this is
/// what separates "the gate works" from "the gate fires on prose about the
/// gate". Both forms are asserted because different branches exclude them, and
/// the third case guards the exclusion itself: over-excluding would blind the
/// scanner entirely while every other assertion here still passed.
#[test]
fn arm_e_scanner_ignores_anchors_inside_code() {
    let inline = "## alpha-rule
text

Every `See [`x`](#no-such-section)` between sections is unchecked.
";
    assert!(
        intra_dangling(inline).is_empty(),
        "an anchor inside an INLINE CODE SPAN is an example, not a link"
    );

    let fenced = "## alpha-rule
text

```text
see [x](#no-such-section)
```
";
    assert!(
        intra_dangling(fenced).is_empty(),
        "an anchor inside a FENCED BLOCK is an example, not a link"
    );

    let mixed = "## alpha-rule
text

`code` and [real](#no-such-section) here
";
    assert_eq!(
        intra_dangling(mixed),
        vec!["no-such-section".to_string()],
        "excluding code spans must not blind the scanner to links beside them"
    );
}

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

/// `## slug` headings, lowercase-kebab only (the method-rules convention).
fn sections(method_rules: &str) -> Vec<String> {
    method_rules
        .lines()
        .filter_map(|l| l.strip_prefix("## "))
        .map(str::trim)
        .filter(|s| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        })
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
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
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
        "{} method-rules section(s) have NO mention in CLAUDE.md, so they are unreachable from \
         the file that loads into every session — landed and invisible, which is close to not \
         having landed. Each section already contains its intended index line in a blockquote; \
         move it across. Missing: {missing:?}",
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

//! **A panic that is a public contract must be documented where a consumer can read it (GAP-221).**
//!
//! `fuel-graph`'s builders that take operands from a second `Graph` panic, by
//! design: `assert_same_graph` documents itself as *"the panicking sibling for
//! builders that return `NodeHandle` rather than `Result` … a panic at the build
//! site is far better than the hang it replaces"*. The behaviour is intended and
//! it is a real part of the public API.
//!
//! It was documented **only in that private helper's comment**. Lightbulb — a
//! downstream consumer — inferred the contract from a module doc's prose,
//! correctly, and had no way to verify it; their suite pins it with a
//! `#[should_panic]` written against an inference. `fuel-graph` had **zero**
//! `# Panics` rustdoc sections in the entire crate.
//!
//! This test is the ratchet: a public builder that carries the cross-graph
//! contract must carry a `# Panics` section. It is deliberately a *source scan*
//! rather than a lint, for three reasons recorded from the day it was written:
//!
//! 1. `clippy::missing_panics_doc` has a **different population** — 95 sites in
//!    this crate (84 in `lib.rs`, 6 in `opt.rs`, 5 in `runtime_fused.rs`),
//!    against the 23 that carry *this* contract. Most of the rest
//!    are `unwrap`/indexing sites where the correct fix is to **not panic**,
//!    not to document panicking; blanket-documenting them would legitimise 95
//!    panics on production paths in a project whose rule is `Result` from day
//!    one. That is a separate registry row, deliberately not this test's job.
//! 2. The lint is **not** a superset: measured, it misses at least one builder
//!    that reaches the panic only through the private helper.
//! 3. A scan can be made to fail when a *new* undocumented builder appears,
//!    which turns an unbounded migration into a ratchet.
//!
//! # What this test can NOT see — stated because a scanner's population is its
//! whole meaning
//!
//! The population is **naming and syntax**: a builder that enforces the same
//! contract by some spelling other than [`HELPER_MARKER`] or [`INLINE_MARKER`]
//! is invisible here.
//! This is a **lower bound**, not a proof. It is still worth having: it cannot
//! certify the crate, but it can stop the count rising.

use std::path::PathBuf;

/// The two spellings that mean "this builder panics when an operand belongs to
/// another graph". They are *not* interchangeable, which is why they are two
/// constants and not a list:
///
/// - [`HELPER_MARKER`] panics on its own — its presence is sufficient.
/// - [`INLINE_MARKER`] is a plain pointer comparison, used in non-panicking
///   contexts too, so it counts **only inside an assertion**.
///
/// Keep both honest: naming a spelling the code does not use would claim a reach
/// this scan does not have — the same defect as an undocumented panic, pointed
/// at the documentation instead of the code.
const HELPER_MARKER: &str = "assert_same_graph(";

/// See [`HELPER_MARKER`]. Counts only within an assertion.
const INLINE_MARKER: &str = "Arc::ptr_eq(";

/// The private helper whose panic every `assert_same_graph` caller inherits.
///
/// **This is the ratchet's foundation, and the ratchet cannot see it being
/// removed.** If `assert_same_graph` were converted to return `Result`, every
/// builder below would stop panicking, the site count would not move by one,
/// and this test would keep passing while meaning something entirely different.
/// So the foundation is asserted separately. (Pattern adopted from Vulkane's
/// `gpu-run` scanner, which had the identical hole.)
const FOUNDATION_FN: &str = "fn assert_same_graph";

fn lib_rs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("lib.rs")
}

/// One `pub fn` and the lines of its body, plus the doc block immediately above it.
struct PubFn {
    name: String,
    line: usize,
    doc: String,
    body: String,
}

/// Split `src/lib.rs` into public functions with their preceding doc comment.
///
/// The body runs from the `pub fn` line to the **closing brace at the same
/// indentation**, which is exact for rustfmt'd source.
///
/// ⚠️ An earlier version ended a body at *the next `fn` at the same or lesser
/// indent*, and its comment claimed that over-reading "can only make this test
/// stricter, never laxer — so it errs in the safe direction." **That was wrong,
/// and it is worth keeping the correction here rather than the claim.** A free
/// function followed by an `impl` block over-reads into the block's *more*
/// indented methods: `compact` was reported as a carrier because the `get`
/// twenty lines below it — inside a following `impl` — does the assertion. The
/// consequence is not extra strictness, it is a **false attribution**, and
/// satisfying it would have added a `# Panics` section to a function that does
/// not panic. **A doc requirement enforced at the wrong site produces a false
/// claim in the documentation, which is the same defect this test exists to
/// remove, pointed the other way.**
fn public_fns(src: &str) -> Vec<PubFn> {
    let lines: Vec<&str> = src.lines().collect();
    let indent = |s: &str| s.len() - s.trim_start().len();

    let starts: Vec<usize> = (0..lines.len())
        .filter(|&i| {
            let t = lines[i].trim_start();
            t.starts_with("pub fn ") || t.starts_with("pub async fn ")
        })
        .collect();

    let mut out = Vec::new();
    for &i in &starts {
        let my_indent = indent(lines[i]);

        // Body: to the closing brace at this function's own indentation.
        let mut end = lines.len();
        for (j, line) in lines.iter().enumerate().skip(i + 1) {
            let t = line.trim_end();
            if t.len() == my_indent + 1 && t.trim_start() == "}" && indent(line) == my_indent {
                end = j + 1;
                break;
            }
        }

        // Doc block: contiguous `///` (and attributes) immediately above.
        let mut doc_start = i;
        while doc_start > 0 {
            let t = lines[doc_start - 1].trim_start();
            if t.starts_with("///")
                || t.starts_with("#[")
                || t.is_empty()
                    && doc_start > 1
                    && lines[doc_start - 2].trim_start().starts_with("///")
            {
                doc_start -= 1;
            } else {
                break;
            }
        }

        let name = lines[i]
            .trim_start()
            .trim_start_matches("pub ")
            .trim_start_matches("async ")
            .trim_start_matches("fn ")
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .next()
            .unwrap_or("")
            .to_string();

        out.push(PubFn {
            name,
            line: i + 1,
            doc: lines[doc_start..i].join("\n"),
            body: lines[i..end].join("\n"),
        });
    }
    out
}

/// Does this body enforce the cross-graph contract *by panicking*?
///
/// `Arc::ptr_eq` alone is not enough — it is also used in non-panicking
/// comparisons — so it counts only inside an assertion.
fn carries_cross_graph_panic(body: &str) -> bool {
    if body.contains(HELPER_MARKER) {
        return true;
    }
    // `Arc::ptr_eq` reached from an `assert!`/`assert_eq!`, possibly several
    // lines below it (these assertions are routinely wrapped across lines).
    let mut in_assert = 0usize;
    for line in body.lines() {
        let t = line.trim_start();
        if t.starts_with("///") {
            continue;
        }
        if t.contains("assert!(") || t.contains("assert_eq!(") {
            in_assert = 6; // window: the assertion's own line plus its wrapped tail
        }
        if in_assert > 0 {
            if line.contains(INLINE_MARKER) {
                return true;
            }
            in_assert -= 1;
        }
    }
    false
}

#[test]
fn the_panicking_helper_still_panics() {
    // FOUNDATION. Without this, a refactor of `assert_same_graph` into a
    // `Result`-returning function would leave the scan below green and
    // meaningless — the builders would no longer panic, the count would not
    // move, and nothing would say so.
    let src = std::fs::read_to_string(lib_rs()).expect("read fuel-graph/src/lib.rs");
    let idx = src
        .find(FOUNDATION_FN)
        .unwrap_or_else(|| panic!("`{FOUNDATION_FN}` is gone — this test's entire population is defined by it, so its removal invalidates `cross_graph_panics_are_documented` rather than fixing it. Re-scope both, do not delete this assertion."));

    let tail = &src[idx..];
    let body_end = tail.find("\n    }\n").unwrap_or(tail.len().min(4000));
    let body = &tail[..body_end];

    assert!(
        body.contains("assert!(") || body.contains("panic!("),
        "`assert_same_graph` no longer panics. Every builder documented with a \
         `# Panics` section for the cross-graph contract is now lying, and the \
         ratchet in this file cannot see it: the site count does not move when \
         the foundation is removed."
    );
}

#[test]
fn cross_graph_panics_are_documented() {
    let src = std::fs::read_to_string(lib_rs()).expect("read fuel-graph/src/lib.rs");
    let fns = public_fns(&src);

    let carriers: Vec<&PubFn> = fns
        .iter()
        .filter(|f| carries_cross_graph_panic(&f.body))
        .collect();

    // NON-TRIVIALITY. A scan that finds nothing passes for free; this asserts
    // the population is real before asserting anything about it. The measured
    // count when this test was written was 27; the floor is set below that so
    // that legitimate consolidation does not fail the build, while a scan that
    // silently stopped matching does.
    assert!(
        carriers.len() >= 15,
        "population collapsed to {} carriers (expected >= 15). The scan is not \
         finding what it used to; PANIC_MARKERS or `public_fns` has stopped \
         matching. A green result from this file would be vacuous.",
        carriers.len()
    );

    // DISCRIMINATION. The predicate must not simply match every public fn, or
    // the requirement below is a crate-wide docs rule wearing a contract's name.
    assert!(
        carriers.len() < fns.len() / 2,
        "the predicate matched {} of {} public fns — it is not discriminating \
         between builders that carry the cross-graph contract and those that do not.",
        carriers.len(),
        fns.len()
    );

    let undocumented: Vec<String> = carriers
        .iter()
        .filter(|f| !f.doc.contains("# Panics"))
        .map(|f| format!("  fuel-graph/src/lib.rs:{} {}", f.line, f.name))
        .collect();

    assert!(
        undocumented.is_empty(),
        "{} public builder(s) panic on cross-graph operands with no `# Panics` \
         section. This is a public contract documented where no consumer can \
         read it (GAP-221):\n{}\n\nAdd a `# Panics` section naming the \
         cross-graph case. Do NOT satisfy this by removing the assertion: the \
         panic is intended and replaces a hang.",
        undocumented.len(),
        undocumented.join("\n")
    );
}

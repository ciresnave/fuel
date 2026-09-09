// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A registry anchor must still find something, and must still point somewhere small.**
//!
//! `docs/gaps.md` cites code by prose anchor — `anchor: git grep -n '<pattern>'` — because
//! [`cite-what-cannot-move`] measured that line numbers rot and prose does not. ⚠️ **Nothing
//! has ever checked that an anchor still resolves.** `scripts/check-gaps-table.py` does not
//! read anchors at all, and the #139 conversions were verified at authoring time — **an
//! authoring-time check is a born-red that goes green and then stays green while the tree
//! moves underneath it**, which is the mechanism that produced the population #139 existed
//! to convert.
//!
//! # ⚠️ WHY THIS EXECUTES THE ANCHOR INSTEAD OF PARSING IT
//!
//! An anchor is **not a string**. It is a `git grep` COMMAND LINE, with flags (`-n`, `-A1`,
//! `-l`), regex semantics, and `-- <pathspec>` scoping. **The correct interpreter of a
//! `git grep` command is `git grep`.**
//!
//! A previous version of this file reimplemented it — literal substring search over a
//! hand-rolled corpus — and was wrong **nine times**, every one on the registry's NOTATION
//! rather than its content, and every one presenting as a finding ABOUT the registry:
//!
//! | assumed | reality |
//! |---|---|
//! | any `git grep` after `anchor:` is the anchor | a row's **control** has the same shape; reported GAP-003's anchor as 8197 hits (that is its control, `assert_eq!`) |
//! | patterns are `'single'`-quoted | **nearly half are double-quoted** (the architect measured 25 single / 23 double); the loose scan latched onto the apostrophes in *SUBJECT'S … code'S* and yielded the fragment `S SPELLING rather than the code` |
//! | a byte cap is safe | `&s[..220]` panicked inside `—`; this document has never been ASCII |
//! | the corpus may contain `gaps.md` | every anchor then matches itself, forever |
//! | the corpus may contain this test | its own doc quotes four anchors, and the negative-control sentinel matched itself |
//! | every `anchor:` names one pattern | **30 are PROSE RECIPES** — *"two `git grep -l` runs, intersected"* |
//! | patterns are literal | GAP-275's `"no .HostBuffer::F8E5M2. variant"` — the dots are **regex** |
//! | patterns are unscoped | GAP-277 and GAP-281 carry `-- <pathspec>` |
//! | `rs`/`md`/`toml`/… is enough | GAP-277 targets a **`.metal`** file; 164 such files are tracked |
//!
//! **Every "dead anchor" that version reported was the instrument. Executing the command
//! removes all nine at once**, because git owns the semantics of its own flags.
//!
//! # ⚠️ THE EXIT ARMS ARE SEPARATED, AND THAT IS NOT OPTIONAL
//!
//! `git grep` returns **1** for *no match* and non-zero for *bad regex*, *bad pathspec*,
//! *not a repo*, *binary missing*. **Reading any non-zero as DEAD would rebuild all nine
//! failures in one line** — an instrument defect wearing a finding's clothes.
//!
//! ```text
//! exit 0      RESOLVES            (and the output size is bounded)
//! exit 1      DEAD                <- the ONLY arm that is a finding
//! exit >1     INSTRUMENT FAILURE  <- reported as NOT MEASURED, never as dead
//! no git      INSTRUMENT FAILURE
//! ```
//!
//! **`separation_is_load_bearing` below is a permanent sibling that feeds a deliberately
//! malformed pathspec and requires INSTRUMENT FAILURE rather than DEAD.** A born-red proving
//! detection is not enough here; the SEPARATION is the property that matters, because
//! *"your anchor is dead"* sends a reader to re-derive working code.
//!
//! # What a green does NOT cover — printed on every run, not buried here
//!
//! ⚠️ **30 of 79 `anchor:` fields are prose RECIPES and no mechanical check can verify
//! them.** The test prints that count and names them UNCHECKABLE. **A green covering 49 of
//! 79 while reading as "anchors verified" is a coverage claim, not a result.**
//!
//! Also out of scope, by standing ruling: **the 58 rows carrying a bare `file:LINE`.** They
//! are deferred (loud-failing rot is lowest value per unit of work), and two — `GAP-004`,
//! `GAP-117` — carry a bare line ON PURPOSE as the record of what rotted. **A detector that
//! reddens on a standing ruling is a false block.** This guards the anchors that EXIST.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The anchors that were already dead when this gate was written, each with the
/// **measured replacement** for it. Baseline at `6d6299ea`.
///
/// ⚠️ **MAY ONLY SHRINK, IN BOTH DIRECTIONS.** A dead anchor NOT listed here fails (new
/// rot); a listed anchor that starts RESOLVING also fails (stale entry) — so fixing one
/// forces removing its line, and the baseline cannot quietly outlive its reason. That is
/// `gap_hedge_allowlist`'s discipline, and the property this registry has spent a night
/// discovering it lacks elsewhere.
///
/// ⚠️ **Every subject here is ALIVE. Not one of these is a closed gap** — five live
/// subjects with five dead citations, by five different mechanisms, and **none of them is
/// a rename**, which is the only death the prose-anchor convention was chosen to survive.
const DEAD_ANCHOR_BASELINE: &[(&str, &str)] = &[
    // The prose was REWORDED around the term.
    (
        "GAP-275",
        "REPLACE WITH `F8E5M2` — 15 hits in the cited file",
    ),
    // The SUBJECT MOVED FILE: lib.rs -> canonical.rs.
    (
        "GAP-287",
        "REPLACE WITH `unwrap_or` scoped to `canonical.rs` — 10 hits there",
    ),
    // A PHRASE lost a word: the concept is present, the anchor's " ops" suffix is not.
    (
        "GAP-290",
        "REPLACE WITH `Empty-schema` (drop the ` ops` suffix) — 1 hit in lib.rs",
    ),
    // ⚠️ The TOKEN BECAME A PREFIX, and `-w` made the match exact. No rename, no move, no
    // rewording — the sharpest of the five, and the one that most indicts `-w`.
    (
        "GAP-300",
        "REPLACE WITH `Truncated` (or drop `-w`) — the token grew around `Trunc`",
    ),
    // ⚠️ NOT GUESSED. The other four carry a measurement; this one carries a question, and
    // an entry that guesses would look identical to the four that do not.
    (
        "GAP-258",
        "REPLACEMENT UNDECIDED — architect ruling required: is the row's subject \
         `unsafe fn` specifically, or the `unsafe` blocks that replaced it? \
         (`unsafe` = 8 hits in the cited file; `unsafe fn` = 0)",
    ),
];

/// Anchors allowed to exceed [`LOCATES_MAX`] output lines. **May only shrink.**
const SPRAWLING_ANCHOR_CEILING: usize = 6;

/// Above this an anchor has the form of a citation and none of its function — a reader
/// running it is handed a haystack. Counts OUTPUT LINES, so `-A1`-style context flags
/// legitimately inflate it; that is the property being bounded (what the reader gets).
const LOCATES_MAX: usize = 40;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent directory")
        .to_path_buf()
}

/// Split a command line into arguments, honouring both quote styles.
fn tokenize(cmd: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for ch in cmd.chars() {
        match (quote, ch) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, c @ ('"' | '\'')) => quote = Some(c),
            (None, c) if c.is_whitespace() => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `(id, git-grep command)` for each checkable anchor, plus the ids whose `anchor:` is a
/// prose recipe rather than a command.
///
/// The command lives in a markdown code span, so the SPAN is parsed — prose apostrophes
/// outside it cannot reach the tokenizer, which is what defeated the previous version.
fn anchors(gaps: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut cmds = Vec::new();
    let mut recipes = Vec::new();
    for line in gaps.lines() {
        if !line.starts_with('|') || !line.contains("GAP-") {
            continue;
        }
        let Some(id) = line
            .split_whitespace()
            .find(|w| w.contains("GAP-"))
            .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-'))
        else {
            continue;
        };
        let lower = line.to_ascii_lowercase();
        let mut from = 0usize;
        while let Some(a) = lower[from..].find("anchor:") {
            from += a + "anchor:".len();
            let rest = &line[from..];
            let mut cur = 0usize;
            let mut got = false;
            while let Some(b1) = rest[cur..].find('`') {
                let s = cur + b1 + 1;
                let Some(b2) = rest[s..].find('`') else { break };
                let span = &rest[s..s + b2];
                cur = s + b2 + 1;
                // ⚠️ A span mentioning `git grep` is not necessarily a COMMAND. GAP-270's
                // recipe quotes the fragment `git grep -l` mid-sentence — no pattern — and
                // running it yields `fatal: no pattern given`. That surfaced as INSTRUMENT
                // FAILURE rather than as a dead anchor, which is the arm separation doing
                // its job; the classification is fixed here so it is not raised at all.
                //
                // A checkable anchor carries a QUOTED PATTERN. Anything else is a recipe.
                if span.contains("git grep") {
                    if span.contains('"') || span.contains('\'') {
                        cmds.push((id.to_string(), span.trim().to_string()));
                    } else {
                        recipes.push(id.to_string());
                    }
                    got = true;
                    break;
                }
                if cur > 400 {
                    break;
                }
            }
            if !got {
                recipes.push(id.to_string());
            }
        }
    }
    (cmds, recipes)
}

/// What running an anchor told us. **The three outcomes are deliberately distinct types,
/// not three integers**, so a caller cannot collapse a tool failure into a finding.
#[derive(Debug)]
enum Outcome {
    Resolves(usize),
    Dead,
    NotMeasured(String),
}

/// ⚠️ **`--cached`, and this is not a preference — it is the difference between a gate
/// that works and one that fires falsely on every Windows lane.**
///
/// The working tree is CRLF on a Windows checkout, so a line ending in `X` is really
/// `X\r\n` and the regex `$` — which anchors before `\n` — can NEVER match after a real
/// character. The index stores LF. Measured here:
///
/// ```text
///                          worktree   index
/// git grep '{$'  (Rust!)      0        146 files    <- thousands of lines end in `{`
/// GAP-281 `assert_eq!($`      0         75 files / 205 lines
/// CONTROL `pub fn` (no anchor) 1          1         <- identical, as it must be
/// ```
///
/// **So this gate reported GAP-281 DEAD locally while CI would have called it alive.**
/// That is *green on CI, red locally, and the disagreement reads as flakiness* — the class
/// this project has already paid for twice. **The index is also the right subject on the
/// merits: it is the canonical content, and it is what a reader on any platform sees.**
const SEARCH_THE_INDEX: &str = "--cached";

fn run_anchor(root: &Path, cmd: &str) -> Outcome {
    let args = tokenize(cmd);
    // Drop the leading `git`; keep `grep` and everything after, including `-- <pathspec>`.
    let Some(rest) = args.strip_prefix(&["git".to_string()]) else {
        return Outcome::NotMeasured(format!("not a git command: {cmd}"));
    };
    // Insert `--cached` immediately after `grep` so it precedes the pattern and pathspec.
    let mut rest: Vec<String> = rest.to_vec();
    let at = if rest.first().map(String::as_str) == Some("grep") {
        1
    } else {
        0
    };
    rest.insert(at, SEARCH_THE_INDEX.to_string());
    match Command::new("git").args(&rest).current_dir(root).output() {
        Err(e) => Outcome::NotMeasured(format!("could not invoke git: {e}")),
        Ok(o) => match o.status.code() {
            Some(0) => Outcome::Resolves(String::from_utf8_lossy(&o.stdout).lines().count()),
            Some(1) => Outcome::Dead,
            Some(n) => Outcome::NotMeasured(format!(
                "git exited {n}: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            )),
            None => Outcome::NotMeasured("git terminated by signal".into()),
        },
    }
}

#[test]
fn every_registry_anchor_still_resolves_and_still_locates() {
    let root = repo_root();
    let gaps = std::fs::read_to_string(root.join("docs/gaps.md"))
        .expect("control: docs/gaps.md must be readable, or this test checks nothing");
    let (found, recipes) = anchors(&gaps);

    // ⚠️ The exclusion is PRINTED, per the architect's ruling: a green covering 48 of 79
    // while reading as "anchors verified" is a coverage claim, not a result.
    println!(
        "checkable git-grep anchors: {}\n\
         PROSE-RECIPE anchors UNCHECKABLE BY THIS GATE: {} {:?}",
        found.len(),
        recipes.len(),
        recipes
    );

    // Non-vacuity: an extractor that stops matching passes everything below having examined
    // nothing, and is indistinguishable from a registry of perfect anchors.
    assert!(
        found.len() >= 30,
        "only {} checkable anchors extracted — the extractor is broken, not the registry. \
         (An earlier version found 24 because it assumed single quotes; 23 of 48 are \
         double-quoted.)",
        found.len()
    );

    let mut dead: std::collections::BTreeMap<String, String> = Default::default();
    let mut sprawling = Vec::new();
    let mut not_measured = Vec::new();
    for (id, cmd) in &found {
        match run_anchor(&root, cmd) {
            Outcome::Resolves(n) if n > LOCATES_MAX => sprawling.push(format!("{id} ({n} lines)")),
            Outcome::Resolves(_) => {}
            Outcome::Dead => {
                dead.insert(id.clone(), format!("{id}: `{cmd}` matched NOTHING"));
            }
            Outcome::NotMeasured(why) => not_measured.push(format!("{id}: {why}")),
        }
    }

    // ⚠️ NOT MEASURED IS NOT A FINDING, and it is reported first so it cannot be mistaken
    // for one. A tool failure and a subject property must not share an exit arm.
    assert!(
        not_measured.is_empty(),
        "INSTRUMENT FAILURE — these anchors were NOT MEASURED, and this says nothing about \
         whether they resolve:\n  {}\n\nFix the instrument, then re-read the result.",
        not_measured.join("\n  ")
    );

    // ⚠️ THE BASELINE IS BIDIRECTIONAL. Arm 1: rot that is not in the baseline.
    let baselined: std::collections::BTreeSet<&str> =
        DEAD_ANCHOR_BASELINE.iter().map(|(id, _)| *id).collect();
    let new_rot: Vec<&String> = dead
        .iter()
        .filter(|(id, _)| !baselined.contains(id.as_str()))
        .map(|(_, msg)| msg)
        .collect();
    assert!(
        new_rot.is_empty(),
        "a registry anchor no longer resolves, and it is NOT in the baseline:\n  {}\n\n\
         The row cites code by a prose string so a rename cannot break it — but a prose \
         anchor is fragile against the prose ITSELF being edited, the subject MOVING FILE, \
         a DECLARATION FORM changing, or (with `-w`) the token GROWING. All five baselined \
         entries died one of those ways, with the subject still alive.\n\n\
         RE-ANCHOR IT, or record what was searched and why it is gone. Do not delete the \
         anchor silently, and do not add it here to make this pass.",
        new_rot
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );

    // ⚠️ Arm 2, the self-retiring half: a baselined anchor that RESOLVES again must have
    // its line removed. Without this the baseline outlives its reason and is
    // indistinguishable from one still needed — the exact defect this registry keeps
    // finding in its own suppressions.
    let fixed: Vec<String> = DEAD_ANCHOR_BASELINE
        .iter()
        .filter(|(id, _)| !dead.contains_key(*id))
        .map(|(id, fix)| format!("{id} — resolves again. Remove its line. ({fix})"))
        .collect();
    assert!(
        fixed.is_empty(),
        "a BASELINED dead anchor now RESOLVES — the baseline may only shrink, so remove \
         these lines from DEAD_ANCHOR_BASELINE:\n  {}",
        fixed.join("\n  ")
    );

    assert!(
        sprawling.len() <= SPRAWLING_ANCHOR_CEILING,
        "{} anchors emit more than {LOCATES_MAX} lines, above the ceiling of {}:\n  {}\n\n\
         An anchor matching a large fraction of the tree has the FORM of a citation and none \
         of its FUNCTION — a reader running it is handed a haystack.\n\n\
         ⚠️ This ceiling may only SHRINK. It is not a budget for new sprawling anchors.",
        sprawling.len(),
        SPRAWLING_ANCHOR_CEILING,
        sprawling.join("\n  ")
    );
}

/// **The separation is the property, so it gets a permanent test rather than a one-off
/// born-red.**
///
/// ⚠️ A gate that cannot tell *bad pathspec* from *no match* reports working code as dead,
/// and sends a reader to re-derive it. Feeding a malformed pathspec must yield
/// `NotMeasured`, never `Dead`.
#[test]
fn separation_is_load_bearing() {
    let root = repo_root();

    // ⚠️ ONE BORN-RED PER FAILURE LAYER, NOT PER MECHANISM. An earlier version tested only
    // a malformed PATHSPEC and passed — while a malformed REGEX, which fails at a different
    // layer, was live and being reported as a dead anchor. **A self-test that validates the
    // mechanism does not validate every path into it.**

    // LAYER 1 — malformed pathspec.
    let bad_path = run_anchor(&root, "git grep -n \"anything\" -- :(bogusmagic)nope");
    assert!(
        matches!(bad_path, Outcome::NotMeasured(_)),
        "a malformed pathspec must classify as INSTRUMENT FAILURE, not DEAD — got {bad_path:?}"
    );

    // LAYER 2 — malformed REGEX. This is the one that got past the previous version:
    // `assert_eq!($` is VALID basic regex and INVALID extended regex, so under `-E` git
    // exits 128. Calling that DEAD accuses a live anchor.
    let bad_regex = run_anchor(&root, "git grep -nE \"assert_eq!($\" -- fuel-transformers/");
    assert!(
        matches!(bad_regex, Outcome::NotMeasured(_)),
        "a malformed regex must classify as INSTRUMENT FAILURE, not DEAD — got {bad_regex:?}"
    );

    // LAYER 3 — a path git cannot read as a pathspec at all.
    let bad_arg = run_anchor(&root, "git grep --no-such-flag \"anything\"");
    assert!(
        matches!(bad_arg, Outcome::NotMeasured(_)),
        "an unknown flag must classify as INSTRUMENT FAILURE, not DEAD — got {bad_arg:?}"
    );

    // ⚠️ LAYER 4 — the CRLF/anchor trap, asserted so it cannot regress. The SAME pattern
    // must resolve, because we search the index; against the working tree it returns 0 and
    // this gate would call a live anchor dead on every Windows checkout.
    let anchored = run_anchor(&root, "git grep -n \"assert_eq!($\" -- fuel-transformers/");
    assert!(
        matches!(anchored, Outcome::Resolves(n) if n > 0),
        "an end-of-line-anchored pattern must RESOLVE against the index — got {anchored:?}. \
         If this is Dead, the search reverted to the working tree, which is CRLF, and every \
         `$`-anchored anchor in the registry will be falsely reported dead."
    );

    // CONTROL, the other direction: a well-formed search for something that cannot exist
    // must classify as DEAD, or the arms are merged the other way and nothing is a finding.
    let genuinely_absent = run_anchor(&root, "git grep -n \"ZzNotARealStringZz\"");
    assert!(
        matches!(genuinely_absent, Outcome::Dead),
        "a well-formed search matching nothing must classify as DEAD — got {genuinely_absent:?}"
    );

    // CONTROL, third arm: something that must be found.
    let present = run_anchor(&root, "git grep -n \"pub fn\" -- fuel-ir/src");
    assert!(
        matches!(present, Outcome::Resolves(n) if n > 0),
        "a search for `pub fn` in fuel-ir/src must RESOLVE — got {present:?}"
    );
}

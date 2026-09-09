// SPDX-License-Identifier: MIT OR Apache-2.0
//! **A registry row must be able to say who owns it.**
//!
//! `docs/gaps.md` uses three table schemas. Two carry a `Status` column; one does
//! not. The project convention is that **ownership lives in the status cell** — so
//! in the third schema there is nowhere for an owner to live.
//!
//! Measured at `e3f40665` over all **301** rows:
//!
//! ```text
//! schema WITH a Status column      219 rows   171 carry an ownership token   78%
//! schema WITHOUT a Status column    82 rows     7 carry an ownership token    8%
//! ```
//!
//! ⚠️ **The 7 are the control, and they are why this is a reading rather than a
//! blind query:** the token IS expressible in the no-Status schema — those rows put
//! it in the Gap cell. Both populations can produce a hit; only one does. That
//! control is asserted below, so this test cannot go blind and report agreement.
//!
//! # ⚠️ THE 82 ARE NOT DEBT. DO NOT SHRINK THIS BY CONVERTING TABLES.
//!
//! The four-column schema is **permanently exempt from the STATUS convention, ruled
//! 2026-09-02 on measurement**: those tables are Tier C subdivided by crate, an index
//! of terse capability declines where **the gap statement IS the status**. Of the
//! rows, **67 carry no status language at all**, so a Status column would mean
//! writing OPEN into 67 cells — a field identical for 85% of its rows. **That ruling
//! is CORRECT and this test does not challenge it.**
//!
//! **What it observes is a SECOND field.** The exemption's stated criterion is *"the
//! gap statement is the status."* ⚠️ **Nothing in it says a gap statement is its own
//! OWNER.** The exemption was decided on one axis and extends silently to another
//! that nobody ruled on — and it extends invisibly, because every existing check
//! passes on those rows and therefore reports agreement.
//!
//! **A TRUE JUSTIFICATION ATTACHED TO A WIDER CLAIM THAN IT SUPPORTS.**
//!
//! # So what does this ratchet actually hold?
//!
//! **The direction, not the number.** A red-on-arrival gate over 82 pre-existing
//! rows would be useless, so this does not demand they be fixed. It fails when the
//! population **GROWS** — when a new row is filed somewhere its owner cannot be
//! written down.
//!
//! ⚠️ **The right way to shrink it is to record an owner on a row that has none.
//! Converting a table to add a Status column would shrink it too, and would do the
//! exact thing the 2026-09-02 ruling forbids.** If you are here because this test
//! went red, add the owner — do not add the column.
//!
//! # What this does NOT tell you
//!
//! - **Expressibility, not correctness.** A row carrying `Owner: UNALLOCATED` counts
//!   as expressing ownership. It says nobody owns it, which is an answer.
//! - ⚠️ **OWNERLESS and OWNER-NOT-ASKED are different and this cannot separate
//!   them.** A row owned by someone whose hold lives in a session is byte-identical
//!   here to an abandoned one. **The only detector for that is asking the lane.**
//! - It reads `docs/gaps.md` only, and does not modify it.

use std::path::{Path, PathBuf};

/// Rows under a no-Status schema carrying no ownership token, at `e3f40665`.
///
/// ⚠️ **This number is INVARIANT under the 82-vs-81 labelling disagreement that
/// `scripts/check-gaps-table.py` already documents** — *"two honest instruments
/// disagreed 82-vs-81 on 2026-09-06 … they differ by GAP-099, which HAS a status
/// cell under a 4-column header."* Measured both ways:
///
/// ```text
/// by HEADER          82 rows   7 with an ownership token   75 inexpressible
/// by ROW CELL COUNT  81 rows   6 with an ownership token   75 inexpressible
/// ```
///
/// **`GAP-099` carries `Owner: UNALLOCATED`, so it moves between numerator and
/// denominator together and the difference does not change.** The bound therefore
/// does not depend on which convention a future reader picks, which is the only
/// reason it is safe to hard-code a figure the project has already disagreed about.
///
/// This test classifies by HEADER. Reclassifying by cell count is fine and must not
/// move the ceiling; if it does, the population changed rather than the label.
/// **A ceiling, not a target.** See the module docs before changing it.
const OWNERSHIP_INEXPRESSIBLE_CEILING: usize = 75;

fn gaps_md() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-ir must have a parent directory")
        .join("docs/gaps.md")
}

fn is_row(line: &str) -> bool {
    let t = line.trim_start_matches(['|', ' ', '~', '*', '_']);
    t.starts_with("GAP-") && line.starts_with('|')
}

fn has_ownership_token(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("owner:") || l.contains("allocated")
}

/// Rows paired with whether their table's header declares a `Status` column.
fn rows_with_schema(text: &str) -> Vec<(bool, String)> {
    let mut has_status = false;
    let mut out = Vec::new();
    for line in text.replace("\r\n", "\n").lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("| ID ") || trimmed.starts_with("|ID ") {
            has_status = line.to_ascii_lowercase().contains("status");
        } else if is_row(line) {
            out.push((has_status, line.to_string()));
        }
    }
    out
}

#[test]
fn every_registry_row_can_say_who_owns_it() {
    let text = std::fs::read_to_string(gaps_md())
        .expect("control: docs/gaps.md must be readable, or this test checks nothing");
    let rows = rows_with_schema(&text);

    // Non-vacuity: a parser that stops matching passes every check below having
    // examined nothing, and is indistinguishable from a clean registry.
    assert!(
        rows.len() >= 250,
        "only {} rows parsed from docs/gaps.md — the parser is broken, not the registry. \
         (⚠️ An earlier version of this parser matched `| GAP-` and silently dropped 47 \
         STRUCK rows written `| ~~GAP-...~~`, understating the population by 16%.)",
        rows.len()
    );

    let with: Vec<_> = rows.iter().filter(|(s, _)| *s).collect();
    let without: Vec<_> = rows.iter().filter(|(s, _)| !*s).collect();
    assert!(
        !with.is_empty() && !without.is_empty(),
        "both schemas must be present, or this test compares one population with itself: \
         with-status {} / without {}",
        with.len(),
        without.len()
    );

    // ⚠️ THE CONTROL, ASSERTED RATHER THAN ASSUMED. If NO row in the no-Status schema
    // carries an ownership token, this test cannot distinguish "nobody records an
    // owner there" from "my token match does not work on those rows".
    let expressible_without = without
        .iter()
        .filter(|(_, l)| has_ownership_token(l))
        .count();
    assert!(
        expressible_without > 0,
        "no row in the no-Status schema carries an ownership token. That is not a \
         finding — it is this test going blind. The token must be demonstrably \
         findable in BOTH populations or the comparison below means nothing."
    );

    let inexpressible = without.len() - expressible_without;
    assert!(
        inexpressible <= OWNERSHIP_INEXPRESSIBLE_CEILING,
        "{inexpressible} registry rows sit under a schema with no Status column AND \
         record no owner — up from the ceiling of {OWNERSHIP_INEXPRESSIBLE_CEILING}.\n\n\
         A new row has been filed where its owner cannot be written down.\n\n\
         ⚠️ FIX BY RECORDING AN OWNER, NOT BY ADDING A COLUMN. The four-column \
         schema is exempt from the Status convention by a 2026-09-02 ruling made on \
         measurement; converting those tables would do the thing that ruling \
         forbids. Ownership is a different axis and is what this holds."
    );
}

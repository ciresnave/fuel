// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contract-maintenance tooling for the `audited:` attestation (GAP-228).
//!
//! **Deliberately NOT under `fkc/verify/`.** These tools REWRITE `.fkc.md`
//! contracts, and `verify/`'s
//! `no_seeder_writes_the_ledger_behind_this_writer` gate scans that directory
//! for ANY file-writing call. Living here keeps that gate UNBENT.
//!
//! The alternative was to teach the gate an exemption, and the first attempt
//! was already drifting: an allowance keyed on a filename, then narrowed to
//! "the file must not mention the ledger" — which failed, because the
//! seeder's own `#[ignore]` reason string legitimately names it. **Each bend
//! made the gate weaker and its claim less true.** Moving the writer out of
//! the scanned directory REMOVES the exception instead of encoding it.
//!
//! It is also the right home on the merits: `verify/` is the ledger and
//! verification seam; a contract rewriter is maintenance tooling that
//! CONSUMES the ledger rather than part of it.

use crate::fkc::verify::VerificationLedger;
use crate::fkc::verify::seed_cpu_ledger::ITERS;
use crate::fused::PrecisionGuarantee;

/// The evidence clause appended to a `notes:` field when `audited: true` is
/// backed by a LEDGER RECORD rather than by a source-level determinism
/// argument.
///
/// **Emitted, never hand-written (GAP-228).** A hand-written note passes the
/// same delta-gate whether it is honest or not: counting entries proves the
/// plumbing moved, not that the attestation was earned. Rendering the clause
/// from the harness's own constants makes honesty a data path instead of a
/// discipline, and drift becomes a diff rather than a judgement call.
///
/// **It states the BASIS and the COVERAGE, and names what it is not.** The
/// coverage half is the one that matters: a source argument generalises to
/// all inputs by construction, and this does not. Today's all-zeros finding
/// is precisely why input coverage is the axis to disclose — a claim earned
/// against a degenerate probe reads identically to one earned against real
/// data unless the note says which.
pub(crate) fn evidence_clause() -> String {
    format!(
        "[evidence: bit_stable_on_same_hardware earned EMPIRICALLY per registered dtype \
         — {ITERS} byte-identical repeat invocations of ONE probe, on the recording \
         hardware and under the pinned toolchain ({}). Not a source-level determinism \
         argument, and not evidence about other inputs, other machines, or other \
         compilers.]",
        pinned_toolchain()
    )
}

/// The channel from `rust-toolchain.toml`, for the clause to name.
///
/// ⚠️ **The clause used to disclose the HARDWARE and not the TOOLCHAIN, and
/// that was an incomplete coverage statement on the axis that matters.**
/// Float results can differ across compiler versions (FMA contraction,
/// autovectorisation), so "bit-stable on the recording hardware" is
/// implicitly also "under the recording compiler". When the clause was first
/// emitted, Fuel had NO toolchain file and the records came from the box
/// default — an unnamed nightly.
///
/// This names the PINNED channel rather than the running `rustc`, and the
/// difference is worth stating: the pin is what governs every build of this
/// repo, and it is re-derivable from the tree. It is not proof that a given
/// seeding run used it — that proof is the pin file being present and
/// `rustup` honouring it, which is a property of the repo rather than of
/// this string.
fn pinned_toolchain() -> String {
    let f = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rust-toolchain.toml");
    let text = std::fs::read_to_string(&f)
        .unwrap_or_else(|e| panic!("the clause must name a pinned toolchain; read {f:?}: {e}"));
    parse_pinned_channel(&text).unwrap_or_else(|| {
        panic!("no `channel` in {f:?}; refusing to emit a clause that names no toolchain")
    })
}

/// The parse half of [`pinned_toolchain`], split out so the REFUSAL is
/// testable.
///
/// The doc comment above claims this "refuses to emit a clause that names no
/// toolchain". That was a claim about a `panic!` I had written and never
/// executed — and verifying it in place would have meant moving the real
/// toolchain file, which changes the active compiler and forces a full
/// rebuild. **Separating I/O from parsing makes the guard's own claim cheap
/// to check**, which is the difference between a guard that is asserted and
/// one that is tested.
///
/// ⚠️ **A checkout WITHOUT the pin file resolves to the box default**, so the
/// panic is not paranoia: it is the case where a clause would otherwise name
/// a toolchain that is not the one in use. Refusing beats emitting a
/// confident wrong disclosure.
fn parse_pinned_channel(text: &str) -> Option<String> {
    for line in text.lines() {
        if let Some(rest) = line.split_once("channel")
            && let Some(v) = rest.1.split('"').nth(1) {
                return Some(format!("rust-toolchain.toml channel = {v}"));
            }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The toolchain guard REFUSES rather than emitting an unnamed compiler.
    ///
    /// Both directions, because a parser that always returns `None` would
    /// satisfy the refusal half alone and silently break every real emission.
    #[test]
    fn the_toolchain_guard_refuses_a_file_with_no_channel() {
        assert_eq!(
            parse_pinned_channel(
                "[toolchain]
channel = \"1.98.0\"
"
            )
            .as_deref(),
            Some("rust-toolchain.toml channel = 1.98.0"),
            "a real pin file must parse, or every emission breaks"
        );
        assert_eq!(
            parse_pinned_channel(
                "[toolchain]
components = [\"rustfmt\"]
"
            ),
            None,
            "a file with no channel must REFUSE — emitting a clause that names no              toolchain is worse than emitting none, because a checkout without the pin              resolves to the box default and the clause would be confidently wrong"
        );
        assert_eq!(
            parse_pinned_channel(""),
            None,
            "an empty file must refuse too"
        );
    }

    /// **GAP-228: flip the fully-backed `audited: false` sections and emit
    /// their evidence clause.** `#[ignore]`d — it rewrites checked-in
    /// contracts; run manually, review the diff.
    ///
    /// **This is 60 individually-evidenced edits that land in one commit, not
    /// one bulk assertion about 60 things.** Each section is flipped only if
    /// EVERY entry it fans to has a passing `bit_stable_on_same_hardware`
    /// record at that entry's own `kernel_revision_hash`. A section with any
    /// unbacked dtype is skipped: flipping it would hand the unbacked dtypes a
    /// populated claim the gate then downgrades, converting a silent UNAUDITED
    /// into a loud downgrade — worse than leaving it, and invisible to a
    /// section-level count.
    ///
    /// **The pre-flight measured 0 such sections**, so the hazard does not
    /// arise today; the check stays because the population changes and the
    /// cost of it is one comparison.
    ///
    /// ⚠️ **The existing `notes:` text is PRESERVED, never replaced.** Those
    /// notes carry real per-kernel reasoning ("max(0, x); exact for f32/f64.
    /// bf16/f16 widen to f32 then narrow. NaN-propagating (torch parity)").
    /// Overwriting them with a rendered clause would destroy information to
    /// make a generator's life easier. The clause is APPENDED.
    #[test]
    #[ignore = "rewrites docs/kernel-contracts/cpu/*.fkc.md; run manually via `cargo test -p fuel-dispatch gap_228_flip -- --ignored --nocapture`"]
    fn gap_228_flip_fully_backed_sections() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/kernel-contracts/cpu");
        let ledger = VerificationLedger::embedded();
        let unaudited_notes = PrecisionGuarantee::UNAUDITED.notes;
        let clause = evidence_clause();

        // Pass 1: per file, which op_kinds are FULLY backed.
        let mut flippable: Vec<(String, Vec<String>)> = Vec::new();
        let mut files: Vec<(std::path::PathBuf, String)> = Vec::new();
        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
            let path = e.expect("dir entry").path();
            if path.extension().and_then(|x| x.to_str()) != Some("md") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            files.push((path.clone(), text.clone()));
            let Ok(provider) = crate::fkc::import_bundle_str(&text, &crate::fkc::CpuLinkRegistry)
            else {
                continue;
            };
            let mut table = crate::kernel::KernelBindingTable::new();
            let mut fused = crate::fused::FusedKernelRegistry::new();
            if provider.register_into(&mut table, &mut fused).is_err() {
                continue;
            }
            let mut per_op: Vec<(String, (usize, usize))> = Vec::new();
            for (op, dtypes, backend, entry) in table.iter_entries() {
                if entry.precision.notes != unaudited_notes {
                    continue;
                }
                let backed = ledger.has_pass(
                    backend,
                    dtypes,
                    entry.kernel_revision_hash,
                    "bit_stable_on_same_hardware",
                );
                let k = format!("{op:?}");
                match per_op.iter_mut().find(|(n, _)| *n == k) {
                    Some((_, (b, u))) => {
                        if backed {
                            *b += 1
                        } else {
                            *u += 1
                        }
                    }
                    None => per_op.push((k, if backed { (1, 0) } else { (0, 1) })),
                }
            }
            let ops: Vec<String> = per_op
                .into_iter()
                .filter(|(_, (_, u))| *u == 0)
                .map(|(k, _)| k)
                .collect();
            if !ops.is_empty() {
                flippable.push((name, ops));
            }
        }

        // Pass 2: rewrite. A section runs from its `## ` heading to the next
        // one; `op_kind:` inside identifies it.
        let mut flipped = 0usize;
        for (path, text) in &files {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            // ⚠️ The per-FILE skip had the same pre-flip-state dependency as
            // the per-SECTION one, and it short-circuits first — so fixing the
            // section check alone changed nothing and the re-emission run
            // still reported success having refreshed zero clauses.
            //
            // A file is in scope if it has something to FLIP (`flippable`) or
            // something to REFRESH (a clause already in its text). The second
            // is read from the file, so it does not evaporate when the flip
            // succeeds.
            let empty_ops: Vec<String> = Vec::new();
            let ops = match flippable.iter().find(|(n, _)| *n == name) {
                Some((_, o)) => o,
                None if text.contains("[evidence: bit_stable_on_same_hardware") => &empty_ops,
                None => continue,
            };
            let lines: Vec<&str> = text.lines().collect();
            let mut out: Vec<String> = Vec::with_capacity(lines.len());
            // section bounds
            let mut starts: Vec<usize> = Vec::new();
            for (i, l) in lines.iter().enumerate() {
                if l.starts_with("## ") {
                    starts.push(i);
                }
            }
            let mut section_of = vec![usize::MAX; lines.len()];
            for (si, &st) in starts.iter().enumerate() {
                let en = starts.get(si + 1).copied().unwrap_or(lines.len());
                for x in section_of.iter_mut().take(en).skip(st) {
                    *x = si;
                }
            }
            // which sections carry a flippable op_kind
            let mut section_flip = vec![false; starts.len()];
            for (i, l) in lines.iter().enumerate() {
                let t = l.trim();
                if let Some(rest) = t.strip_prefix("op_kind:") {
                    let op = rest.split('#').next().unwrap_or("").trim();
                    if ops.iter().any(|o| o == op) && section_of[i] != usize::MAX {
                        section_flip[section_of[i]] = true;
                    }
                }
            }
            // ⚠️ PER-SECTION, not per-file. The first version tracked a
            // file-scoped `changed` flag, so once ANY section in a file
            // flipped, every later `notes:` in a flippable section received
            // the clause — INCLUDING sections that were ALREADY
            // `audited: true` by other means.
            //
            // It attached an empirical-evidence claim to two `matmul.fkc.md`
            // sections whose attestation came from somewhere else: 4 flips,
            // 6 clauses. **That is same-name-different-strength — the defect
            // this whole increment exists to avoid — committed by the
            // generator built to avoid it.** Caught by comparing flips to
            // clauses PER FILE, which is a check worth keeping precisely
            // because the entry-level delta was exactly right anyway: the
            // count that mattered for the ruling could not see this.
            // A SECTION IS ELIGIBLE FOR THE CLAUSE IF THIS RUN FLIPPED IT
            // *OR* IT ALREADY CARRIES ONE, and the second half is what makes
            // re-emission possible at all.
            //
            // The first version keyed eligibility solely on "this run flipped
            // it". Re-running after everything was already `audited: true`
            // therefore flipped 0 and refreshed 0 clauses: the generator could
            // EMIT but not RE-EMIT, which is useless in exactly the case it
            // will be needed for -- the evidence changing under a toolchain
            // pin or a re-seed.
            //
            // Keying on the existing clause is also the precise
            // discriminator: it marks the sections whose attestation rests on
            // the LEDGER. A section audited by SOURCE reasoning carries no
            // clause and must never acquire one, which is exactly the
            // misattachment this generator committed on its first run.
            let mut has_clause = vec![false; starts.len().max(1)];
            for (i, l) in lines.iter().enumerate() {
                if l.contains("[evidence: bit_stable_on_same_hardware")
                    && section_of[i] != usize::MAX
                {
                    has_clause[section_of[i]] = true;
                }
            }
            let mut flipped_sections = vec![false; starts.len().max(1)];
            let mut changed = false;
            for (i, l) in lines.iter().enumerate() {
                let sec = section_of[i];
                // ⚠️ ELIGIBILITY MUST NOT BE DERIVED SOLELY FROM THE
                // PRE-FLIP STATE, and this is the third time that shape has
                // bitten in one increment.
                //
                // `section_flip` comes from `flippable`, which lists ops whose
                // entries are currently UNAUDITED. After the flip those
                // entries are BACKED, so they vanish from that list and every
                // already-flipped section becomes ineligible — the re-emission
                // run then matched nothing and silently refreshed 0 clauses
                // while reporting success.
                //
                // `has_clause` is read from the FILE TEXT, so it survives the
                // state change that erases `flippable`.
                let in_flip = sec != usize::MAX && (section_flip[sec] || has_clause[sec]);
                let t = l.trim_start();
                if in_flip && t.starts_with("audited: false") {
                    let indent = &l[..l.len() - t.len()];
                    out.push(format!("{indent}audited: true"));
                    if sec != usize::MAX {
                        flipped_sections[sec] = true;
                    }
                    changed = true;
                    flipped += 1;
                } else if in_flip
                    && t.starts_with("notes:")
                    && sec != usize::MAX
                    && (flipped_sections[sec] || has_clause[sec])
                {
                    // Append the clause inside the existing quoted string —
                    // after STRIPPING any clause already there.
                    //
                    // Idempotence is not tidiness here. This generator will be
                    // re-run whenever the evidence changes (a toolchain pin, a
                    // re-seed), and an append-only version would stack a second
                    // clause beside a now-false first one — leaving two
                    // coverage statements where the stale one reads exactly as
                    // authoritative as the current one.
                    let mut base = l.to_string();
                    if let Some(start) = base.find("[evidence: bit_stable_on_same_hardware")
                        && let Some(rel_end) = base[start..].find(']') {
                            let end = start + rel_end + 1;
                            // also swallow one leading space, if present
                            let cut_from = if start > 0 && base.as_bytes()[start - 1] == b' ' {
                                start - 1
                            } else {
                                start
                            };
                            base.replace_range(cut_from..end, "");
                        }
                    if let Some(last) = base.rfind('"') {
                        base.insert_str(last, &format!(" {clause}"));
                    }
                    out.push(base);
                } else {
                    out.push(l.to_string());
                }
            }
            let refreshed = out.join(
                "
",
            ) != lines.join(
                "
",
            );
            if changed || refreshed {
                let mut joined = out.join("\n");
                if text.ends_with('\n') {
                    joined.push('\n');
                }
                std::fs::write(path, joined).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
                println!("[gap-228] rewrote {name}");
            }
        }
        println!(
            "[gap-228] flipped {flipped} `audited:` line(s) across {} file(s)",
            files.len()
        );
        println!("[gap-228] clause in use: {}", evidence_clause());
        // NOT `flipped > 0`: a re-emission run legitimately flips nothing
        // and refreshes every clause. Asserting on flips would make the
        // generator fail precisely when doing the job it was extended to do.
        println!("[gap-228] (0 flips is expected on a re-emission run)");
    }

    /// **GAP-228 pre-flight: which `audited: false` SECTIONS are wholesale
    /// flippable, and which are not.** `#[ignore]`d — a planning measurement,
    /// run manually.
    ///
    /// The unit of EDIT is a section; the unit of EVIDENCE is an entry, and a
    /// section fans over dtypes. **Flipping a section whose fan is only
    /// partly backed would give the unbacked dtypes a populated claim the
    /// gate then downgrades** — turning a silent UNAUDITED into a loud
    /// downgrade warning, which is worse than leaving it alone and is exactly
    /// the kind of partial move a section-level count cannot see.
    ///
    /// So this reports the sections in three buckets: fully backed (flippable
    /// now), partly backed (must not be flipped wholesale), and unbacked.
    /// **THE DECLARATION CENSUS THE ARCHITECT RULED FOR: what do the entries
    /// that are still unflipped actually DECLARE?**
    ///
    /// The ruling was *measure the declaration before I rule the semantics* —
    /// because settling accumulation-order (attention) and scan-order (SSM)
    /// for a population nobody has measured gives the ruling a scope its
    /// evidence does not cover.
    ///
    /// **The answer is that every unflipped section declares `max_ulp: ~`.**
    /// Not one declares a bound, so not one needs an exact reference, so not
    /// one needs a spec decision at all. The whole residue is reachable on
    /// GAP-228(b)'s conv pattern: write a probe recipe, earn
    /// `bit_stable_on_same_hardware`, flip with the evidence clause.
    ///
    /// ⚠️ **THE FIRST RUN OF THIS CENSUS WAS WRONG AND ITS ANSWER LOOKED
    /// CLEAN.** The predicate was `^\s*audited:\s*false\s*$`, and many
    /// sections write `audited: false     # CPU primitive-class: family
    /// default applies`. The `$` anchor dropped **7 of 42** sections —
    /// including `flash_attn_f32` — and reported a tidy, confident,
    /// undercounted result. Nothing in the output looked wrong. It was caught
    /// only because the SECTION count would not reconcile with the ENTRY
    /// count already known from `gap_226_split_of_the_entries_that_declare_nothing`:
    /// five ops appeared to be missing their f32 section while their entry
    /// counts said otherwise. **A disagreement between two constructs is what
    /// found it; nothing about the number itself could have.**
    ///
    /// Two constructs, stated separately because they differ and a single
    /// number would hide it: **42 SECTIONS** carry `audited: false`, of which
    /// **2 are `registrable: false`** describe-only chassis umbrellas that
    /// bind no `OpKind` and contribute **zero entries**. The remaining **40
    /// registrable sections** are the **60 ENTRIES** of GAP-226's residue.
    #[test]
    fn every_unflipped_cpu_section_declares_no_ulp_bound() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/kernel-contracts/cpu");
        let mut unflipped = 0usize;
        let mut describe_only = 0usize;
        let mut declares_a_bound: Vec<String> = Vec::new();
        // Vacuity control: this assertion is meaningless unless SOME section
        // somewhere declares a real bound. If every section in the tree said
        // `~`, "the unflipped ones all say `~`" would be true of nothing.
        let mut bounded_elsewhere = 0usize;
        let mut files = 0usize;

        let entries =
            std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {dir:?}: {e}"));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            files += 1;
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
            let lines: Vec<&str> = src.lines().collect();
            let starts: Vec<usize> = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| l.starts_with("## "))
                .map(|(i, _)| i)
                .collect();
            for (si, &i) in starts.iter().enumerate() {
                let end = starts.get(si + 1).copied().unwrap_or(lines.len());
                let body = &lines[i..end];
                let field = |key: &str| -> Option<String> {
                    body.iter().find_map(|l| {
                        let t = l.trim_start();
                        t.strip_prefix(key).map(|v| {
                            // Value ends at a trailing `#` comment. THE FIRST
                            // VERSION OF THIS CENSUS ANCHORED ON END-OF-LINE
                            // AND SILENTLY DROPPED EVERY COMMENTED FIELD.
                            v.split('#').next().unwrap_or("").trim().to_string()
                        })
                    })
                };
                let audited = field("audited:");
                let max_ulp = field("max_ulp:");
                let name = lines[i][3..].split("  ").next().unwrap_or("?").trim();

                if audited.as_deref() == Some("false") {
                    unflipped += 1;
                    if field("registrable:").as_deref() == Some("false") {
                        describe_only += 1;
                    }
                    match max_ulp.as_deref() {
                        Some("~") | None => {}
                        Some(v) => declares_a_bound.push(format!("{name}: max_ulp: {v}")),
                    }
                } else if matches!(max_ulp.as_deref(), Some(v) if v != "~") {
                    bounded_elsewhere += 1;
                }
            }
        }

        assert!(
            files >= 5 && unflipped > 0,
            "read {files} contract files and found {unflipped} unflipped sections — the \
             census is looking in the wrong place and would pass vacuously"
        );
        assert!(
            bounded_elsewhere > 0,
            "no CPU contract section anywhere declares a non-`~` max_ulp, so the \
             assertion below is true of nothing and this census cannot discriminate"
        );
        println!(
            "[gap-228] unflipped CPU sections: {unflipped} ({describe_only} describe-only, \
             {} registrable); sections declaring a bound elsewhere: {bounded_elsewhere}",
            unflipped - describe_only
        );
        assert!(
            declares_a_bound.is_empty(),
            "these unflipped sections DECLARE a ulp bound, so they need an exact \
             reference and therefore a semantics ruling before they can be earned: \
             {declares_a_bound:?}. Everything else in the residue needs only \
             bit-stability and is reachable on the conv pattern with no ruling at all."
        );
        assert_eq!(
            describe_only, 2,
            "expected exactly 2 `registrable: false` describe-only sections among the \
             unflipped ones (the elementwise-unary and inplace-unary-affine chassis \
             umbrellas). They bind no OpKind and contribute ZERO entries, which is why \
             42 sections correspond to 60 entries and not to 63."
        );
    }

    /// **A DELETED ASSERTION AND A SATISFIED ASSERTION ARE INDISTINGUISHABLE
    /// TO A SUITE THAT ONLY RUNS THE ASSERTIONS IT FINDS.**
    ///
    /// **Measured, not argued.** Deleting `assert_eq!(nothing, 60, ...)` from
    /// `gap_226_split_of_the_entries_that_declare_nothing` and running that
    /// gate returns `test result: ok. 1 passed` — with a real
    /// `Compiling fuel-dispatch` line, so the binary matched the edited source
    /// and this is not a warm-cache artifact. Clippy stays silent too, because
    /// `nothing` is still read by the assertions below it: even the accidental
    /// dead-code defence does not fire.
    ///
    /// **A WRONG VALUE IS COMPARED AND FAILS. A DELETED VALUE IS NEVER
    /// COMPARED AT ALL.** Every comment beside those pins defends what the
    /// number should BE. Not one of them defends the line's EXISTENCE — and
    /// the pins are the load-bearing part of this harness, because they are
    /// the only place a silent coverage loss shows up as a number.
    ///
    /// So: enumerate the pins that must be found, and go red when one is not.
    ///
    /// ⚠️ **THIS TEST DELIBERATELY DOES NOT LIVE IN THE FILE IT CHECKS, AND
    /// THAT IS LOAD-BEARING RATHER THAN TIDINESS.** An existence check written
    /// in the file it scans **counts itself**: every anchor below is a literal,
    /// so `nothing, 60,` matched twice — once in the pin and once in this list
    /// — *while the pin was deleted*. The gate reported AMBIGUOUS, and once the
    /// pin came back it would have reported PRESENT for the same reason it
    /// reported it while missing: because the list is the thing being found.
    /// Written under `fkc/` and scanning only `fkc/verify/`, the anchors are
    /// outside the scanned set and each occurrence is a real one.
    ///
    /// That was the third self-referential scan to trip on its own text in this
    /// crate in one session — after a comment naming the forbidden write
    /// spellings in `ledger.rs`, and after this test's own absence-control
    /// literal. **The rule that generalises: a source-scanning check must not
    /// be inside the set it scans.**
    ///
    /// ⚠️ **THE REGRESS IS REAL AND THIS DOES NOT CLOSE IT.** Deleting an
    /// entry from `REQUIRED_PINS` switches off the check for that pin exactly
    /// as deleting the pin switched off the check for its value. What changes
    /// is *where the deletion has to happen*: in a short list whose only
    /// purpose is to say THESE MUST EXIST, under a name that says so, instead
    /// of invisibly inside a hundred-line test among forty other lines. That
    /// is a **shorter** regress, not a terminated one, and claiming otherwise
    /// would be the same overreach as letting the `max_ulp` attachment control
    /// look like it covered bit-stability.
    #[test]
    fn the_pins_this_harness_rests_on_are_still_present() {
        // (file, anchor, what its absence would silently cost)
        const REQUIRED_PINS: &[(&str, &str, &str)] = &[
            (
                "seed_cpu_ledger.rs",
                "nothing, 32,",
                "the count of contract entries still declaring NO precision claim \
                 (GAP-226/228). Without it, a regression that un-earns entries reads \
                 as a pass.",
            ),
            (
                "seed_cpu_ledger.rs",
                "expected 591 contract-derived entries backed WITHOUT the fill",
                "the count backed by contract + record rather than by \
                 `fill_unset_cpu_precision`. This is the number the whole program \
                 exists to move; unpinned, the fill could come back unnoticed.",
            ),
            (
                "seed_cpu_ledger.rs",
                "expected 20 conv registrations to invoke",
                "the conv registrations that must actually invoke (GAP-228(b)). \
                 Without it, a probe that stops building leaves the loop covering \
                 fewer ops and still reporting no constant output.",
            ),
            (
                "seed_cpu_ledger.rs",
                "Under the old remainder",
                "the assertion that every sweep outcome lands in a NAMED bucket. Its                  predecessor was `x + (L - x) == L`, which could not fail, while its                  catch-all was printed to the reader as an invoke error.",
            ),
            (
                "seed_cpu_ledger.rs",
                "must not merge",
                "the guard that a harness gap (no recipe written) and a verifier result                  (no reference available) never share a bucket again. They did, through                  a shared `starts_with` prefix.",
            ),
            (
                "seed_cpu_ledger.rs",
                "expected 8 FlashAttn registrations to invoke",
                "the FlashAttn probes that must reach their kernel AND carry                  `k_len < sk` with a non-zero causal offset. Without it the family's                  records could silently become evidence about the static path only.",
            ),
            (
                "seed_cpu_ledger.rs",
                "expected 20 registrations across the four small surfaces",
                "the four small non-conv surfaces that must demonstrably reach their                  kernel (GAP-228(c)). Without it, a probe that stops building leaves                  the loop checking fewer ops and still reporting none inert.",
            ),
            (
                "seed_cpu_ledger.rs",
                "families_tested, EXACT_REFERENCE_FAMILIES",
                "the exact-reference families the attachment control actually \
                 poisoned. Without it, a family that stops passing is skipped with \
                 a println and the control silently covers less.",
            ),
        ];

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fkc/verify");
        let mut missing: Vec<String> = Vec::new();
        let mut duplicated: Vec<String> = Vec::new();
        let mut total_bytes = 0usize;
        for (file, anchor, why) in REQUIRED_PINS {
            let path = dir.join(file);
            let src = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path:?}: {e}"));
            total_bytes += src.len();
            match src.matches(anchor).count() {
                0 => missing.push(format!("{file}: `{anchor}` — {why}")),
                1 => {}
                n => duplicated.push(format!("{file}: `{anchor}` appears {n} times")),
            }
        }

        // Positive control on the PATH: a wrong directory reads nothing and
        // every anchor would report missing, which is loud — but a wrong
        // directory that happened to hold same-named files would not be.
        assert!(
            total_bytes > 10_000,
            "read only {total_bytes} bytes across the pinned files — the scan is \
             looking in the wrong place"
        );
        // Positive control on the PREDICATE: an anchor that is deliberately
        // not in the file must be reported. Without this, a `contains` that
        // always returned true would make every check above vacuous.
        //
        // THE ABSENT ANCHOR IS BUILT AT RUNTIME AND MUST NEVER BE A LITERAL.
        // The first version of this control WAS a literal, and it fired --
        // because a source-scanning test scans the file it is written in, so
        // the control found ITSELF, and the gate went red on its own control
        // while the real check underneath was never reached. Second time a
        // self-referential scan in this crate has been tripped by its own
        // text; the first was a comment naming the forbidden write spellings
        // in `ledger.rs`. It is also why reading a gate's MESSAGE is not
        // optional: this failure and a genuinely deleted pin are both a red X,
        // and only the message tells them apart.
        let probe = std::fs::read_to_string(dir.join("seed_cpu_ledger.rs")).expect("readable");
        let absent = format!("nothing, {},", u32::MAX);
        assert!(
            !probe.contains(&absent),
            "the absence predicate is broken: it finds `{absent}`, which is not in the              file, so every pin below would report present no matter what"
        );

        assert!(
            duplicated.is_empty(),
            "these pins are ambiguous, so a deletion of one copy would go unnoticed: \
             {duplicated:?}"
        );
        assert!(
            missing.is_empty(),
            "THESE PINS HAVE BEEN DELETED, AND THEIR GATES NOW PASS WITHOUT THEM:\n  {}\n\
             A wrong value is compared and fails; a deleted value is never compared \
             at all. Restore the line, or remove its entry from REQUIRED_PINS in the \
             same change and say why it is no longer load-bearing.",
            missing.join("\n  ")
        );
    }

    #[test]
    #[ignore = "planning measurement for GAP-228; run manually with --ignored --nocapture"]
    fn gap_228_which_sections_are_wholesale_flippable() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/kernel-contracts/cpu");
        let ledger = VerificationLedger::embedded();
        let unaudited_notes = PrecisionGuarantee::UNAUDITED.notes;

        // (file, op) -> (backed, unbacked)
        let mut per_section: Vec<((String, String), (usize, usize))> = Vec::new();

        for e in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read {dir:?}: {e}")) {
            let path = e.expect("dir entry").path();
            if path.extension().and_then(|x| x.to_str()) != Some("md") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_string();
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(provider) = crate::fkc::import_bundle_str(&text, &crate::fkc::CpuLinkRegistry)
            else {
                continue;
            };
            let mut table = crate::kernel::KernelBindingTable::new();
            let mut fused = crate::fused::FusedKernelRegistry::new();
            if provider.register_into(&mut table, &mut fused).is_err() {
                continue;
            }
            for (op, dtypes, backend, entry) in table.iter_entries() {
                if entry.precision.notes != unaudited_notes {
                    continue;
                }
                let backed = ledger.has_pass(
                    backend,
                    dtypes,
                    entry.kernel_revision_hash,
                    "bit_stable_on_same_hardware",
                );
                let key = (name.clone(), format!("{op:?}"));
                match per_section.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, (b, u))) => {
                        if backed {
                            *b += 1
                        } else {
                            *u += 1
                        }
                    }
                    None => per_section.push((key, if backed { (1, 0) } else { (0, 1) })),
                }
            }
        }
        per_section.sort();

        let (mut full, mut partial, mut none_) = (0usize, 0usize, 0usize);
        let (mut full_e, mut partial_e, mut none_e) = (0usize, 0usize, 0usize);
        println!("[gap-228] sections with UNAUDITED entries, by backing:");
        for ((file, op), (b, u)) in &per_section {
            let tag = if *u == 0 {
                full += 1;
                full_e += b;
                "FULLY BACKED  (flippable)"
            } else if *b == 0 {
                none_ += 1;
                none_e += u;
                "UNBACKED      (leave)"
            } else {
                partial += 1;
                partial_e += b + u;
                "PARTLY BACKED (DO NOT flip wholesale)"
            };
            println!("[gap-228]   {tag}  {file}  {op}  backed={b} unbacked={u}");
        }
        println!("[gap-228] SECTIONS: {full} fully backed / {partial} partly / {none_} unbacked");
        println!(
            "[gap-228] ENTRIES : {full_e} in fully-backed sections / {partial_e} in partly / {none_e} in unbacked"
        );
        println!(
            "[gap-228] PRE-DECLARED DELTA for the flip: UNAUDITED must drop by exactly {full_e}"
        );
    }
}

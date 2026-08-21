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
         — {ITERS} byte-identical repeat invocations of ONE probe on the recording \
         hardware. Not a source-level determinism argument, and not evidence about \
         other inputs or other machines.]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let Some((_, ops)) = flippable.iter().find(|(n, _)| *n == name) else {
                continue;
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
            let mut flipped_sections = vec![false; starts.len().max(1)];
            let mut changed = false;
            for (i, l) in lines.iter().enumerate() {
                let sec = section_of[i];
                let in_flip = sec != usize::MAX && section_flip[sec];
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
                    && flipped_sections[sec]
                {
                    // Append the clause inside the existing quoted string.
                    if let Some(last) = l.rfind('"') {
                        let mut s2 = l.to_string();
                        s2.insert_str(last, &format!(" {clause}"));
                        out.push(s2);
                    } else {
                        out.push(l.to_string());
                    }
                } else {
                    out.push(l.to_string());
                }
            }
            if changed {
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
        assert!(
            flipped > 0,
            "no section was flipped — the op_kind match found nothing"
        );
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

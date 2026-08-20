// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-077 — bit-stable coverage differ under a simulated GAP-058 flip.
//!
//! # The question
//!
//! GAP-058 identified **578** contract sections that assert `audited: true`
//! with **no verification-ledger entry**, **no numeric bound**, and
//! `bit_stable_on_same_hardware: true`. Flipping those to `audited: false`
//! is the honest thing to do (an unbacked claim should not read as audited),
//! but it is only safe if it does not break **FKC-4.8-0001** — the
//! always-built coverage commitment that fuel-cpu-backend provides at least
//! one bit-stable kernel per primitive op.
//!
//! So: **under that flip, does any `(op, dtypes, backend)` binding key lose
//! its LAST bit-stable candidate?**
//!
//! # Why this is a build and not a corpus scan
//!
//! Contract sections do not map 1:1 onto binding keys — a chassis section
//! fans out over dtypes — so counting `.fkc.md` text over- or under-counts
//! by an amount the counter cannot bound. This loads the **populated binding
//! table** twice (as-is, and with the qualifying sections lowered) and diffs
//! bit-stable coverage per key.
//!
//! # GAP-149 — why the controls are not optional
//!
//! A coverage differ's natural failure mode is a **silent full pass**:
//! "no key lost its last candidate" is byte-identical output to "I loaded
//! nothing" or "I classified every entry the same way". Three controls run
//! before the verdict is believed, and **each one is capable of failing**:
//!
//! * `C1` — the loaded table is non-trivial (entries and keys above a floor).
//! * `C2` — the bit-stable predicate **discriminates within** the loaded
//!   table: it selects neither all entries nor none. A count assertion alone
//!   sails straight through a predicate that matches everything.
//! * `C3` — the flip transform is non-vacuous (it rewrote > 0 sections), and
//!   it is **null-aware**: `max_ulp: ~` is YAML null and must NOT count as a
//!   numeric bound. Getting this wrong is what produced GAP-058's
//!   "1162 of 1162".
//! * `C4` — the **differ itself discriminates**: a sabotage arm that strips
//!   every bit-stable claim must make the differ report losses. A differ that
//!   reports zero losses there is broken, and its zero on the real arm means
//!   nothing.
//!
//! `C4` is the one that matters most. `C1`–`C3` check the *measurement*;
//! `C4` checks that the *instrument can register the event being looked for*.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use fuel_dispatch::fkc::{CpuLinkRegistry, import_bundle_str};
use fuel_dispatch::fused::{FusedKernelRegistry, PrecisionGuarantee};
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;

/// A `(op, dtypes, backend)` binding key, owned so it can be a map key.
type Key = (OpKind, Vec<DType>, BackendId);

/// Per-key coverage: how many alternatives, and how many are bit-stable.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Coverage {
    total: usize,
    bit_stable: usize,
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/fuel-dispatch.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fuel-dispatch has a parent")
        .to_path_buf()
}

// ===========================================================================
// Step 1 — which contracts can actually reach the binding table?
// ===========================================================================

/// Contract files reachable by the **production** registration path.
///
/// Derived rather than hand-listed on purpose: a hand-maintained list is a
/// second source of truth that drifts silently, and "the list went stale" is
/// indistinguishable from "coverage is fine" — the exact GAP-149 shape.
///
/// # Two ways this derivation can be wrong, both of which it has been
///
/// A first cut scanned every `.rs` under `src/` for the substring
/// `kernel-contracts/`. That admitted **doc-comment mentions** and, worse,
/// **test-only `include_str!`s**: `cpu/quant-matmul.fkc.md` is embedded by
/// `fkc/lower.rs` and `fkc/mod.rs` in `#[cfg(test)]` modules and is never
/// imported by `register_cpu_kernels`. Including it made the harness import a
/// contract production never touches — which fails loudly here (a
/// consumer-ahead `MxNotYetRegistrable` gate) but in the general case would
/// silently add binding keys that do not exist in production.
///
/// So the scan is narrowed twice: it requires a literal `include_str!("`, and
/// it reads only the production registration modules. It is then
/// **cross-checked against the real table** by
/// `assert_derived_set_matches_production` — a textual derivation is a model
/// of the registration path, and the model is checked against the thing.
fn live_contract_paths() -> HashSet<String> {
    const PRODUCTION_REGISTRARS: &[&str] =
        &["dispatch.rs", "baracuda_dispatch.rs", "vulkan_dispatch.rs"];
    let src = repo_root().join("fuel-dispatch").join("src");
    let mut out = HashSet::new();
    for file in PRODUCTION_REGISTRARS {
        let text = fs::read_to_string(src.join(file)).expect("readable registrar source");
        for (i, _) in text.match_indices("include_str!(\"") {
            let rest = &text[i + "include_str!(\"".len()..];
            let Some(end) = rest.find('"') else { continue };
            let literal = &rest[..end];
            let Some(pos) = literal.find("kernel-contracts/") else {
                continue;
            };
            let rel = &literal[pos + "kernel-contracts/".len()..];
            if rel.ends_with(".fkc.md") {
                out.insert(rel.to_string());
            }
        }
    }
    out
}

/// Cross-check the textually-derived contract set against the table the
/// production path actually builds.
///
/// Every binding key the contract-only baseline produces must also exist in
/// `register_cpu_kernels`' table. If the derivation admits a contract
/// production never imports, this fires. It is the control that would have
/// caught the `quant-matmul` mistake without relying on that contract
/// happening to have a loud import gate.
fn assert_derived_set_matches_production(baseline: &HashMap<Key, Coverage>) {
    let mut prod = KernelBindingTable::new();
    fuel_dispatch::dispatch::register_cpu_kernels(&mut prod);
    let prod_keys: HashSet<Key> = prod
        .iter_precision()
        .map(|(op, dt, be, _)| (op, dt.to_vec(), be))
        .collect();

    let phantom: Vec<String> = baseline
        .keys()
        .filter(|k| !prod_keys.contains(*k))
        .map(|(op, dt, be)| format!("{op:?} dtypes={dt:?} {be:?}"))
        .collect();
    assert!(
        phantom.is_empty(),
        "derived-live-set control FAILED: {} binding key(s) exist in the \
         contract-only baseline but NOT in the production table. The \
         include_str! derivation is admitting contracts production never \
         imports, so the differ is measuring a table that does not ship:\n{}",
        phantom.len(),
        phantom.join("\n"),
    );
}

/// The CPU slice of the live set.
///
/// Scoped to CPU deliberately, and the scoping is REPORTED rather than
/// silent (a hidden cap reads as "covered everything"):
///   * `FKC-4.8-0001` is a commitment about **fuel-cpu-backend**, so the CPU
///     arm is the one that can break the normative clause;
///   * `CudaLinkRegistry` / `VulkanLinkRegistry` are `#[cfg(feature = …)]`,
///     so a default-feature run cannot resolve their entry points at all.
fn live_cpu_contracts() -> Vec<(String, String)> {
    let root = repo_root().join("docs").join("kernel-contracts");
    let mut v: Vec<(String, String)> = live_contract_paths()
        .into_iter()
        .filter(|p| p.starts_with("cpu/"))
        .map(|p| {
            let text = fs::read_to_string(root.join(&p))
                .unwrap_or_else(|e| panic!("live contract {p} must be readable: {e}"));
            (p, text)
        })
        .collect();
    v.sort();
    v
}

// ===========================================================================
// Step 2 — the flip transform (GAP-058's population), applied to text
// ===========================================================================

/// Is this YAML scalar a real numeric bound?
///
/// **Null-aware by construction.** `max_ulp: ~`, `max_ulp: null`, and an
/// empty value are all "no bound". A regex like `[^\s#]` matches the `~` and
/// reports every section as bounded — that is precisely the defect that
/// produced GAP-058's "794 of 794", then "1162 of 1162".
fn is_numeric_bound(value: &str) -> bool {
    let v = value.split('#').next().unwrap_or("").trim();
    !v.is_empty() && v != "~" && v != "null" && v.parse::<f64>().is_ok()
}

/// Read `key:` out of a precision block's lines, if present.
fn field<'a>(lines: &[&'a str], key: &str) -> Option<&'a str> {
    lines.iter().find_map(|l| {
        let t = l.trim();
        t.strip_prefix(key)
            .and_then(|r| r.strip_prefix(':'))
            .map(|v| v.trim())
    })
}

/// How a section's precision block should be treated by the flip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// GAP-058's real population: `audited: true`, bit-stable, no bound.
    Gap058Flip,
    /// Sabotage arm (`C4`): strip EVERY bit-stable claim, backed or not.
    StripAllBitStable,
}

/// Rewrite a contract's text, returning `(new_text, sections_changed)`.
///
/// Operates on the `precision:` block inside each ```` ```fkc ```` fence.
fn transform(src: &str, mode: Mode) -> (String, usize) {
    let mut out: Vec<String> = Vec::new();
    let mut changed = 0usize;

    // Collect the precision block of each fkc fence, decide, then rewrite.
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        if line.trim_start().starts_with("precision:") {
            // Gather the indented block under `precision:`.
            let base_indent = line.len() - line.trim_start().len();
            let mut block_end = i + 1;
            while block_end < lines.len() {
                let l = lines[block_end];
                if l.trim().is_empty() {
                    break;
                }
                let ind = l.len() - l.trim_start().len();
                if ind <= base_indent {
                    break;
                }
                block_end += 1;
            }
            let block: Vec<&str> = lines[i + 1..block_end].to_vec();

            let audited_true = field(&block, "audited")
                .map(|v| v.split('#').next().unwrap_or("").trim() == "true")
                == Some(true);
            let bit_stable_true = field(&block, "bit_stable_on_same_hardware")
                .map(|v| v.split('#').next().unwrap_or("").trim() == "true")
                == Some(true);
            let has_bound = ["max_ulp", "max_relative", "max_absolute"]
                .iter()
                .any(|k| field(&block, k).is_some_and(is_numeric_bound));

            let qualifies = match mode {
                Mode::Gap058Flip => audited_true && bit_stable_true && !has_bound,
                Mode::StripAllBitStable => bit_stable_true,
            };

            out.push(line.to_string());
            if qualifies {
                changed += 1;
                for l in &block {
                    let t = l.trim_start();
                    let rewritten = match mode {
                        Mode::Gap058Flip if t.starts_with("audited:") => {
                            Some(l.replace("audited: true", "audited: false"))
                        }
                        Mode::StripAllBitStable
                            if t.starts_with("bit_stable_on_same_hardware:") =>
                        {
                            Some(l.replace(
                                "bit_stable_on_same_hardware: true",
                                "bit_stable_on_same_hardware: false",
                            ))
                        }
                        _ => None,
                    };
                    out.push(rewritten.unwrap_or_else(|| l.to_string()));
                }
            } else {
                for l in &block {
                    out.push(l.to_string());
                }
            }
            i = block_end;
            continue;
        }
        out.push(line.to_string());
        i += 1;
    }

    let mut text = out.join("\n");
    if src.ends_with('\n') {
        text.push('\n');
    }
    (text, changed)
}

// ===========================================================================
// Step 3 — build the populated table and project coverage per key
// ===========================================================================

fn build_table(contracts: &[(String, String)]) -> KernelBindingTable {
    let mut table = KernelBindingTable::new();
    let mut fused = FusedKernelRegistry::new();
    for (path, text) in contracts {
        let provider = import_bundle_str(text, &CpuLinkRegistry)
            .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
        provider
            .register_into(&mut table, &mut fused)
            .unwrap_or_else(|e| panic!("contract {path} must register: {e:?}"));
    }
    table
}

fn coverage(table: &KernelBindingTable) -> HashMap<Key, Coverage> {
    let mut map: HashMap<Key, Coverage> = HashMap::new();
    for (op, dtypes, backend, precision) in table.iter_precision() {
        let e = map.entry((op, dtypes.to_vec(), backend)).or_default();
        e.total += 1;
        if precision.bit_stable_on_same_hardware {
            e.bit_stable += 1;
        }
    }
    map
}

/// Keys that had ≥1 bit-stable candidate in `before` and have none in `after`.
fn keys_losing_last_bit_stable(
    before: &HashMap<Key, Coverage>,
    after: &HashMap<Key, Coverage>,
) -> Vec<String> {
    let mut lost: Vec<String> = before
        .iter()
        .filter(|(_, c)| c.bit_stable > 0)
        .filter(|(k, _)| after.get(*k).map(|c| c.bit_stable).unwrap_or(0) == 0)
        .map(|((op, dt, be), c)| {
            format!(
                "{op:?} dtypes={dt:?} {be:?} (was {}/{})",
                c.bit_stable, c.total
            )
        })
        .collect();
    lost.sort();
    lost
}

// ===========================================================================
// The harness
// ===========================================================================

#[test]
fn gap_077_bit_stable_coverage_under_simulated_gap_058_flip() {
    let contracts = live_cpu_contracts();

    // --- C1: the derived corpus is non-trivial ---------------------------
    assert!(
        contracts.len() >= 15,
        "C1 FAILED: only {} live CPU contracts discovered — the include_str! \
         scan is broken, and every downstream 'no loss' verdict would be vacuous",
        contracts.len(),
    );

    let (flipped, gap058_changed): (Vec<(String, String)>, usize) = {
        let mut v = Vec::new();
        let mut n = 0;
        for (p, t) in &contracts {
            let (text, c) = transform(t, Mode::Gap058Flip);
            n += c;
            v.push((p.clone(), text));
        }
        (v, n)
    };
    let (sabotaged, sabotage_changed): (Vec<(String, String)>, usize) = {
        let mut v = Vec::new();
        let mut n = 0;
        for (p, t) in &contracts {
            let (text, c) = transform(t, Mode::StripAllBitStable);
            n += c;
            v.push((p.clone(), text));
        }
        (v, n)
    };

    // --- C3: the flip transform is non-vacuous ---------------------------
    assert!(
        gap058_changed > 0,
        "C3 FAILED: the GAP-058 selector matched 0 sections. An unflipped \
         'flip' produces two identical tables and a guaranteed 'no loss' \
         verdict that means nothing.",
    );
    assert!(
        sabotage_changed >= gap058_changed,
        "C3 FAILED: the sabotage arm ({sabotage_changed}) must strip at least \
         as many sections as the GAP-058 arm ({gap058_changed}) — it is a \
         strict superset by construction (every qualifying section is \
         bit-stable). A smaller count means the selectors disagree.",
    );

    let base = build_table(&contracts);
    let after_flip = build_table(&flipped);
    let after_sabotage = build_table(&sabotaged);

    let cov_base = coverage(&base);
    let cov_flip = coverage(&after_flip);
    let cov_sabotage = coverage(&after_sabotage);

    let entries: usize = cov_base.values().map(|c| c.total).sum();
    let bit_stable_entries: usize = cov_base.values().map(|c| c.bit_stable).sum();

    // --- C1 (cont.): the loaded table is non-trivial ----------------------
    assert!(
        cov_base.len() >= 50 && entries >= 100,
        "C1 FAILED: loaded table is trivial ({} keys / {entries} entries). \
         A differ over an empty table reports 'no loss' unconditionally.",
        cov_base.len(),
    );

    // --- C1 (cont.): the derived live set matches what production imports -
    assert_derived_set_matches_production(&cov_base);

    // --- C4: the DIFFER discriminates — established on CONSTRUCTED inputs -
    //
    // The sabotage arm cannot serve as this control here, and *why* is the
    // finding below: every contract-derived entry already arrives UNAUDITED,
    // so an arm that strips bit-stable claims has nothing left to strip and
    // reports zero — indistinguishable from a broken differ.
    //
    // An instrument's discrimination therefore has to be established
    // independently of the corpus it is aimed at. A degenerate corpus
    // otherwise certifies the instrument by accident.
    assert_differ_discriminates();

    let lost_by_flip = keys_losing_last_bit_stable(&cov_base, &cov_flip);
    let lost_by_sabotage = keys_losing_last_bit_stable(&cov_base, &cov_sabotage);

    // --- THE FINDING: the predicate selects NONE, and that IS the answer --
    //
    // The original C2 ("neither all nor none") FAILED here, and failing was
    // correct: it stopped a vacuous verdict being reported as a clean one.
    // The vacuity is not assumed — its cause is asserted directly below.
    // --- THE CORPUS IS NO LONGER DEGENERATE, AND THAT REVERSAL IS THE POINT
    //
    // HISTORY, kept because the reversal IS the finding: until 2026-08-20 this
    // asserted `bit_stable_entries == 0`. That was correct, and it made the
    // whole differ a vacuity report — every contract-derived entry arrived
    // UNAUDITED, so no key had a bit-stable candidate to lose and GAP-058's
    // flip was inert BY ABSENCE OF ANYTHING TO STRIP. The assertion carried a
    // message written for the day it would fail. GAP-207's ledger seeding is
    // that day: 199 of 623 entries are now bit-stable from a CONTRACT, earned
    // rather than filled.
    //
    // So the direction flips. The harness now asserts the corpus is NOT
    // degenerate, because a silent return to zero would turn every verdict
    // below back into a vacuity report that reads exactly like a clean one.
    assert!(
        bit_stable_entries > 0,
        "the corpus is degenerate again: 0 of {entries} contract-derived CPU entries are bit-stable. Every verdict below would then hold for the WRONG reason — no key can lose a candidate it never had — and would be indistinguishable from a clean result. Either the ledger lost its CPU records or the query key drifted; do not relax this.",
    );

    // Cause, not correlation: the entries are UNAUDITED because the V-FKC-9
    // ledger gate (`gate_precision`, wired at register.rs:389/398) collapses
    // every unbacked machine-checkable claim at import.
    let downgrade_warnings = count_downgrade_warnings(&contracts);
    assert!(
        downgrade_warnings > 0,
        "expected the V-FKC-9 ledger gate to emit downgrade warnings — with \
         0 warnings AND 0 bit-stable entries, the contracts may simply never \
         have declared bit-stability, which is a different world with a \
         different answer.",
    );
    let unaudited_notes = PrecisionGuarantee::UNAUDITED.notes;
    let unaudited_entries = base
        .iter_precision()
        .filter(|(_, _, _, p)| p.notes == unaudited_notes)
        .count();
    assert_eq!(
        unaudited_entries + bit_stable_entries,
        entries,
        "expected the ledger gate to partition every contract-derived CPU entry \
         into exactly two classes — backed (bit-stable survives) or downgraded \
         (UNAUDITED sentinel). {unaudited_entries} + {bit_stable_entries} != \
         {entries} means a third class exists and every count here is over an \
         incomplete population",
    );

    // --- Report ------------------------------------------------------------
    eprintln!("\n=== GAP-077 bit-stable coverage differ ===");
    eprintln!("scope : CPU contracts only ({} files)", contracts.len());
    eprintln!("                   (CUDA/Vulkan link registries are feature-gated;");
    eprintln!("                    FKC-4.8-0001 is a fuel-cpu-backend commitment)");
    eprintln!("binding keys     : {}", cov_base.len());
    eprintln!("entries          : {entries}");
    eprintln!("bit-stable       : {bit_stable_entries} / {entries}");
    eprintln!("UNAUDITED        : {unaudited_entries} / {entries}");
    eprintln!("ledger downgrades: {downgrade_warnings} warnings at import");
    eprintln!("GAP-058 flipped  : {gap058_changed} sections");
    eprintln!("sabotage stripped: {sabotage_changed} sections");
    eprintln!("--- verdict ---");
    eprintln!(
        "keys losing last bit-stable candidate under the flip : {}",
        lost_by_flip.len()
    );
    eprintln!(
        "keys losing last bit-stable candidate under sabotage : {}",
        lost_by_sabotage.len()
    );
    eprintln!(
        "coverage map identical (baseline vs flipped) : {}",
        cov_base == cov_flip
    );
    // Derived from the measurement, never hard-coded. The previous version of
    // these lines printed "NO KEY HAS ONE TO LOSE. The flip is inert"
    // unconditionally — and would have gone on printing it on the very run
    // that measured 199, directly beneath the number contradicting it.
    if bit_stable_entries == 0 {
        eprintln!("READ THIS AS: no key loses its last bit-stable candidate because");
        eprintln!("NO KEY HAS ONE TO LOSE — the flip would be inert by absence.");
    } else {
        eprintln!("READ THIS AS: {bit_stable_entries} of {entries} entries are ledger-backed,");
        eprintln!("so the flip is NOT inert. Applying it would strip the last bit-stable");
        eprintln!("candidate from {} key(s).", lost_by_flip.len());
    }
    eprintln!("===========================================\n");

    // --- THE VERDICT, and it reversed on 2026-08-20 -----------------------
    //
    // This used to assert `lost_by_flip.is_empty()` and `cov_base == cov_flip`
    // — that the flip was inert. Both held only because NOTHING WAS BACKED.
    // With 199 of 623 entries now ledger-backed (GAP-207), applying GAP-058's
    // flip WOULD strip the last bit-stable candidate from every one of those
    // keys, and asserting emptiness would now be asserting something false.
    //
    // The replacement is not a relaxation. It pins the flip's blast radius to
    // EXACTLY the backed set: the flip must strip the backed keys and NOTHING
    // ELSE. Stripping FEWER would mean the GAP-058 selector and the ledger
    // disagree about which entries are backed; stripping MORE would mean it
    // reaches entries it has no business touching. Both fire here.
    //
    // FOR ANYONE ACTING ON GAP-058: this measures a HYPOTHETICAL, not a live
    // breakage — the flip is not applied. The hazard is three-way and no
    // single row states it: seeding + this flip + retiring
    // `fill_unset_cpu_precision` is when those keys actually lose coverage.
    // Any two of the three are survivable.
    // ⚠️ THIS ASSERTION WAS WRONG AND FIRED CORRECTLY TO SAY SO.
    //
    // It used to read `lost_by_flip.len() == bit_stable_entries`, tying the
    // FLIP's blast radius to the total backed set. That held only because the
    // two coincided at 199 — a COINCIDENCE asserted as an invariant. When
    // GAP-225 backed `max_ulp` for 60 more entries the numbers separated
    // (flip 219, backed 279) and it fired, with a message offering the right
    // diagnosis for the wrong situation: "the selector and the ledger
    // disagree". They do disagree, and they are SUPPOSED to — the GAP-058
    // flip is a NARROW selector (184 sections), while the sabotage arm strips
    // bit-stability wholesale (418 sections).
    //
    // The real identity is with the SABOTAGE arm, not with the backed total:
    // stripping every bit-stable claim must cost exactly the backed keys. The
    // flip is a subset of that, and what matters about it is that it is
    // non-empty — i.e. no longer inert.
    //
    // Keying an assertion to a measurement is not enough; it has to be the
    // measurement the property is actually about.
    assert_eq!(
        lost_by_sabotage.len(),
        bit_stable_entries,
        "stripping bit-stability wholesale cost {} key(s) while {bit_stable_entries} \
         entries are ledger-backed. These must be equal by construction: every backed \
         key has exactly one candidate to lose. A difference means the coverage map \
         and the differ disagree about what 'backed' means.",
        lost_by_sabotage.len(),
    );
    assert!(
        lost_by_flip.len() <= lost_by_sabotage.len(),
        "the GAP-058 flip ({}) strips MORE keys than wholesale sabotage ({}) — \
         sabotage is an upper bound by construction, so the flip is reaching entries \
         it has no business touching",
        lost_by_flip.len(),
        lost_by_sabotage.len(),
    );
    assert!(
        !lost_by_flip.is_empty(),
        "the flip is inert again: it strips the last bit-stable candidate from no key, \
         while {bit_stable_entries} entries are backed. Either the selector stopped \
         matching or the backing moved out from under it — both make every verdict \
         here vacuous for the reason this harness exists to catch."
    );
    assert_ne!(
        cov_base, cov_flip,
        "the flip no longer changes per-key coverage at all. With {bit_stable_entries} \
         backed entries that cannot be right — it would mean the flip stopped selecting \
         anything, and the verdicts above would go green for exactly the reason this \
         harness exists to catch.",
    );
}

/// Prove `keys_losing_last_bit_stable` registers the event it exists to
/// detect, using constructed maps rather than the corpus.
///
/// Both directions, because a one-sided control passes for a stuck predicate:
/// a real loss must be reported, and a non-loss must not be.
fn assert_differ_discriminates() {
    let key: Key = (OpKind::AddElementwise, vec![DType::F32], BackendId::Cpu);
    let other: Key = (OpKind::MatMul, vec![DType::F32], BackendId::Cpu);

    let before: HashMap<Key, Coverage> = HashMap::from([
        (
            key.clone(),
            Coverage {
                total: 2,
                bit_stable: 1,
            },
        ),
        (
            other.clone(),
            Coverage {
                total: 1,
                bit_stable: 1,
            },
        ),
    ]);

    // (1) A genuine loss on `key` (its only bit-stable candidate goes away)
    //     while `other` keeps one — must report exactly one key.
    let after_loss: HashMap<Key, Coverage> = HashMap::from([
        (
            key.clone(),
            Coverage {
                total: 2,
                bit_stable: 0,
            },
        ),
        (
            other.clone(),
            Coverage {
                total: 1,
                bit_stable: 1,
            },
        ),
    ]);
    let lost = keys_losing_last_bit_stable(&before, &after_loss);
    assert_eq!(
        lost.len(),
        1,
        "C4 FAILED (positive): a constructed loss of the last bit-stable \
         candidate was not reported. The differ cannot see the event it \
         exists to detect, so every 'no loss' verdict it produces is empty.",
    );

    // (2) No loss at all — must report nothing. Guards the opposite failure:
    //     a differ stuck at "everything lost" would pass (1) and be useless.
    let lost_none = keys_losing_last_bit_stable(&before, &before);
    assert!(
        lost_none.is_empty(),
        "C4 FAILED (negative): the differ reported losses between a map and \
         ITSELF, so it is stuck at yes and its verdicts carry no information.",
    );

    // (3) A partial reduction (2 bit-stable → 1) is NOT a loss of the LAST
    //     candidate. Pins the boundary the question actually asks about.
    let before_two: HashMap<Key, Coverage> = HashMap::from([(
        key.clone(),
        Coverage {
            total: 3,
            bit_stable: 2,
        },
    )]);
    let after_one: HashMap<Key, Coverage> = HashMap::from([(
        key,
        Coverage {
            total: 3,
            bit_stable: 1,
        },
    )]);
    assert!(
        keys_losing_last_bit_stable(&before_two, &after_one).is_empty(),
        "C4 FAILED (boundary): a 2→1 reduction was reported as losing the \
         LAST bit-stable candidate. The differ is answering a broader \
         question than FKC-4.8-0001 asks, and would over-report.",
    );
}

/// Total number of V-FKC-9 ledger downgrade warnings emitted while importing
/// the live CPU contracts — the direct evidence that the gate is what
/// erases the declared bit-stable claims.
/// **GAP-225 population census: WHAT actually blocks each downgraded CPU
/// entry.** Reports, never asserts a threshold.
///
/// The number "424 `max_ulp`-blocked entries" was derived by subtraction —
/// 623 contract-derived CPU entries minus the 199 that survive the ledger
/// gate — and subtraction cannot say WHY the other 424 were downgraded.
/// `gate_precision` collapses the whole guarantee if ANY of four claims is
/// unbacked (`bit_stable_on_same_hardware`, `max_ulp`, `max_relative`,
/// `max_absolute`), so a residue computed that way is a mixture, and at
/// least one component is not `max_ulp` at all: 111 of 659 CPU registrations
/// still have no probe recipe, so their BIT-STABLE claim is unearned too.
///
/// Designing a family taxonomy over an unverified population would produce a
/// well-organised description of the wrong set — and a structured wrong
/// answer invites less re-checking than the raw list it replaced.
///
/// **What this counts, named:** WARNINGS emitted at import, one per contract
/// SECTION whose declared guarantee was downgraded — not table entries. A
/// section fans over dtypes, so section-count and entry-count differ and
/// must not be quoted for each other.
#[test]
fn gap_225_what_actually_blocks_the_downgraded_cpu_entries() {
    let contracts = live_cpu_contracts();
    let mut by_claim_set: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    let mut unparsed = 0usize;

    for (path, text) in &contracts {
        let provider = import_bundle_str(text, &CpuLinkRegistry)
            .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
        for w in provider
            .warnings
            .iter()
            .filter(|w| w.message.contains("downgraded to UNAUDITED"))
        {
            total += 1;
            // `precision claim(s) ["max_ulp", ...] for kernel ...`
            let set = match (w.message.find('['), w.message.find(']')) {
                (Some(a), Some(b)) if b > a => w.message[a..=b].to_string(),
                _ => {
                    unparsed += 1;
                    continue;
                }
            };
            match by_claim_set.iter_mut().find(|(k, _)| *k == set) {
                Some((_, n)) => *n += 1,
                None => by_claim_set.push((set, 1)),
            }
        }
    }
    by_claim_set.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Is a warning per SECTION or per ENTRY? The tallies above are unusable
    // until that is settled, because a section fans over dtypes and the two
    // constructs differ by ~4x. Distinct `rev` values answer it: one per
    // fanned dtype means per-entry.
    let mut revs: Vec<String> = Vec::new();
    for (path, text) in &contracts {
        let provider = import_bundle_str(text, &CpuLinkRegistry)
            .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
        for w in provider
            .warnings
            .iter()
            .filter(|w| w.message.contains("downgraded to UNAUDITED"))
        {
            if let Some(i) = w.message.find("rev ") {
                let rest = &w.message[i + 4..];
                let end = rest.find(')').unwrap_or(rest.len());
                revs.push(rest[..end].to_string());
            }
        }
    }
    let distinct: std::collections::HashSet<&String> = revs.iter().collect();
    eprintln!(
        "[gap-225] warnings carry {} rev values, {} distinct -> {} per registration",
        revs.len(),
        distinct.len(),
        if revs.len() == distinct.len() {
            "ONE"
        } else {
            "MORE THAN ONE"
        }
    );

    eprintln!(
        "[gap-225] downgrade warnings over {} CPU contracts: {total}",
        contracts.len()
    );
    for (set, n) in &by_claim_set {
        eprintln!("[gap-225]     {n:>4}  unbacked {set}");
    }

    // Name the bit-stable-blocked class, not just its size: it is the
    // ACTIONABLE half (probe-recipe work, no oracle), so a bare count of 20
    // tells the next reader nothing about what to build.
    eprintln!("[gap-225] the bit-stable-blocked class, by kernel:");
    let mut named: Vec<String> = Vec::new();
    for (path, text) in &contracts {
        let provider = import_bundle_str(text, &CpuLinkRegistry)
            .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
        for w in provider.warnings.iter().filter(|w| {
            w.message.contains("downgraded to UNAUDITED")
                && w.message.contains("[\"bit_stable_on_same_hardware\"]")
        }) {
            let dt = w
                .message
                .find("dtypes ")
                .map(|i| {
                    let rest = &w.message[i + 7..];
                    let end = rest.find(']').map(|e| e + 1).unwrap_or(rest.len());
                    rest[..end].to_string()
                })
                .unwrap_or_default();
            named.push(format!("{path}  {dt}"));
        }
    }
    named.sort();
    for n in &named {
        eprintln!("[gap-225]     {n}");
    }

    // Derived from the live values, NOT hard-coded. The first version of
    // this line baked "104 warnings / 80 distinct revs" into the string, and
    // when the bit-stable class closed it went on printing the old numbers
    // beside the new tally — the same defect this file already carries a fix
    // for, reintroduced in the text explaining the fix.
    eprintln!(
        "[gap-225] UNMEASURED: the ENTRY-level size of each class. A warning is neither per-section nor per-registration ({} warnings / {} distinct revs), so these counts cannot be converted to entries. The subtraction-derived \"424 max_ulp-blocked entries\" is NOT this population. Left visible rather than estimated.",
        revs.len(),
        distinct.len(),
    );

    // Non-triviality: a parse that silently produced nothing would print an
    // empty table and read as "no downgrades", which is the answer this
    // census exists to disprove or confirm.
    assert!(
        total > 0,
        "no downgrade warnings at all — either the ledger now backs every declared CPU claim (which would be the finding of the program, not a quiet pass) or the warning text changed and this census is parsing nothing"
    );
    assert_eq!(
        unparsed, 0,
        "{unparsed} of {total} downgrade warnings did not contain a bracketed claim list — the message format changed and every tally above is over a subset of the population"
    );
}

/// **GAP-225 family split of the remaining `max_ulp` downgrades, by OP.**
///
/// Reports; asserts only invariants. The warning text names the
/// `kernel_source` tag (`portable-cpu`) rather than the op, so the op is
/// recovered by correlating the warning's revision hash against the imported
/// primitives' `revision` — the same key `gate_precision` used to reject it.
///
/// **RESULT (2026-08-20): the hypothesis is DISCONFIRMED, and the reason is
/// that it was applied to the wrong population.** The architect reasoned from
/// the five families of the BIT-STABLE class (`Where`, `IndexAdd`,
/// `ScatterAdd`, integer `MatMul`, `WriteSliceDoff`) to a conclusion about the
/// `max_ulp` class. **The two sets are disjoint — not one of those five
/// appears among the 84.** The `max_ulp` population is `Cast` 54, `Gather` 9,
/// `IndexSelect` 9, and 12 elementwise binaries.
///
/// **What holds instead is the architect's own third disconfirming shape:**
/// every `max_ulp` line in the three contracts owning these 84 declares
/// `max_ulp: 0`, several with an inline reason. So the claims are legitimate
/// and correctly stated, and the work is EARNING them, not correcting
/// contracts.
///
/// **And that is cheaper than either reading suggested.** `max_ulp: 0` means
/// bit-exact against the correctly-rounded result, and for these ops that
/// reference is computable in-process and exactly: a `Cast` has a defined
/// correctly-rounded value, `Gather`/`IndexSelect` must reproduce input bytes,
/// and a single IEEE `+ - * /` in f64 rounds to the f32 result without double
/// rounding. **No f64 oracle framework and no kiss-ref adapter is required for
/// this population** — which also means `fuel-kiss-ref-backend`'s feature
/// gating is not on its critical path.
///
/// **This exists to DISCONFIRM a hypothesis, not to confirm one.** The
/// architect's read is that many of the 84 declare `max_ulp` on ops where a
/// rounding bound is trivially zero (`Where`, `WriteSliceDoff` — copies) or
/// meaningless (integer `MatMul`), so the fix would be correcting contracts
/// rather than building an oracle. The disconfirming shapes are real and are
/// what this must be able to show: a float op whose bound is about
/// accumulation ORDER, or a contract that declares `max_ulp: 0` deliberately
/// to mean "must be bit-exact" — which is a legitimate claim correctly
/// stated, and would make "does not apply" wrong.
#[test]
fn gap_225_family_split_of_the_remaining_max_ulp_downgrades() {
    let contracts = live_cpu_contracts();
    let mut by_op: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    let mut unattributed = 0usize;

    for (path, text) in &contracts {
        let provider = import_bundle_str(text, &CpuLinkRegistry)
            .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
        let mut rev_to_op: HashMap<u64, String> = HashMap::new();
        for prim in &provider.primitives {
            rev_to_op.insert(
                prim.revision.0,
                format!("{:?} {:?}", prim.op, prim.dtypes.as_slice()),
            );
        }
        for w in provider.warnings.iter().filter(|w| {
            w.message.contains("downgraded to UNAUDITED") && w.message.contains("max_ulp")
        }) {
            total += 1;
            let rev = w.message.find("rev ").and_then(|i| {
                let rest = &w.message[i + 4..];
                let end = rest.find(')').unwrap_or(rest.len());
                rest[..end].parse::<u64>().ok()
            });
            let key = match rev.and_then(|r| rev_to_op.get(&r)) {
                Some(k) => k.clone(),
                None => {
                    unattributed += 1;
                    continue;
                }
            };
            let op = key.split_whitespace().next().unwrap_or("?").to_string();
            match by_op.iter_mut().find(|(k, _)| *k == op) {
                Some((_, n)) => *n += 1,
                None => by_op.push((op, 1)),
            }
        }
    }
    by_op.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    eprintln!("[gap-225-split] {total} max_ulp downgrades, by op:");
    for (op, n) in &by_op {
        eprintln!("[gap-225-split]     {n:>3}  {op}");
    }
    eprintln!("[gap-225-split] unattributed (rev not found among primitives): {unattributed}");
    eprintln!(
        "[gap-225-split] NOTE: every `max_ulp` line in the contracts owning these {total} declares `max_ulp: 0`, several with an inline reason (\"exact: f32 is a strict subset of f64\"). That is a measurement of CONTRACT TEXT LINES, not of declarations per downgraded entry — the lowered value is unavailable after `import_bundle_str`, which gates before returning. Stated as the weaker construct it is."
    );

    assert!(
        total > 0,
        "no max_ulp downgrades found — either they are all earned now (which is the finding, not a quiet pass) or this is parsing nothing"
    );
    assert_eq!(
        unattributed, 0,
        "{unattributed} of {total} warnings could not be attributed to an op by revision hash. The split above is then over a subset, and the missing ones are invisible in it"
    );
}

fn count_downgrade_warnings(contracts: &[(String, String)]) -> usize {
    contracts
        .iter()
        .map(|(path, text)| {
            let provider = import_bundle_str(text, &CpuLinkRegistry)
                .unwrap_or_else(|e| panic!("contract {path} must import: {e:?}"));
            provider
                .warnings
                .iter()
                .filter(|w| w.message.contains("downgraded to UNAUDITED"))
                .count()
        })
        .sum()
}

/// Measures how much of the CPU table's bit-stability is asserted by the
/// bulk fill rather than derived from a contract claim that survived the
/// V-FKC-9 ledger gate.
///
/// This is the second half of the GAP-077 question. `FKC-4.8-0001` can be
/// satisfied two ways: contracts declare bit-stability and the ledger backs
/// it, or `fill_unset_cpu_precision` asserts it wholesale afterwards. The
/// clause explicitly sanctions the latter, so this is a **characterisation**
/// of where the guarantee comes from, not a defect claim.
#[test]
fn gap_077_where_cpu_bit_stability_actually_comes_from() {
    let mut table = KernelBindingTable::new();
    fuel_dispatch::dispatch::register_cpu_kernels(&mut table);

    let fill_notes = PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU.notes;
    let unaudited_notes = PrecisionGuarantee::UNAUDITED.notes;

    let (mut cpu, mut bit_stable, mut from_fill, mut still_unaudited) = (0, 0, 0, 0);
    for (_, _, backend, p) in table.iter_precision() {
        if backend != BackendId::Cpu {
            continue;
        }
        cpu += 1;
        if p.bit_stable_on_same_hardware {
            bit_stable += 1;
            if p.notes == fill_notes {
                from_fill += 1;
            }
        }
        if p.notes == unaudited_notes {
            still_unaudited += 1;
        }
    }

    assert!(cpu > 0, "C1: no CPU entries loaded — nothing was measured");
    assert!(
        bit_stable > 0,
        "C2: no CPU entry is bit-stable; FKC-4.8-0001 could not hold",
    );

    // --- C2b: the `owed to bulk fill` discriminator must be able to say NO -
    //
    // This test's headline is an n-of-n number, and n-of-n is a smell rather
    // than a result: "every entry came from the fill" is what a *stuck*
    // predicate also prints. So prove the predicate can return false — a
    // registration carrying an explicit precision must be classified as
    // NOT-from-fill even after the fill pass runs over it.
    {
        use fuel_dispatch::kernel::OpParams;
        use fuel_ir::{Layout, Result};
        use fuel_memory::Storage;
        use std::sync::{Arc, RwLock};

        fn noop(
            _i: &[Arc<RwLock<Storage>>],
            _o: &mut [Arc<RwLock<Storage>>],
            _l: &[Layout],
            _p: &OpParams,
        ) -> Result<()> {
            Ok(())
        }
        let explicit = PrecisionGuarantee {
            bit_stable_on_same_hardware: true,
            max_ulp: Some(0),
            max_relative: None,
            max_absolute: None,
            notes: "control: an explicit contract-style claim, not the fill default",
        };
        let mut t = KernelBindingTable::new();
        t.register_with_precision(
            OpKind::AddElementwise,
            &[DType::F32],
            BackendId::Cpu,
            noop,
            explicit,
        );
        t.fill_unset_cpu_precision(PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU);
        let classified_from_fill = t
            .iter_precision()
            .filter(|(_, _, be, p)| *be == BackendId::Cpu && p.notes == fill_notes)
            .count();
        assert_eq!(
            classified_from_fill, 0,
            "C2b FAILED: an entry registered with an EXPLICIT precision claim \
             was classified as 'owed to the bulk fill'. The discriminator is \
             stuck at yes, so the {from_fill}-of-{bit_stable} headline below \
             would be an artefact of the predicate rather than a measurement.",
        );
    }

    eprintln!("\n=== GAP-077 provenance of CPU bit-stability ===");
    eprintln!("CPU entries            : {cpu}");
    eprintln!("bit-stable             : {bit_stable}");
    eprintln!("  ...owed to bulk fill : {from_fill}");
    eprintln!("  ...from a contract   : {}", bit_stable - from_fill);
    eprintln!("still UNAUDITED        : {still_unaudited}");
    eprintln!("===============================================\n");
}

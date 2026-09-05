// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP item-8-II MEMBERSHIP GATE.
//!
//! (II) is config-from-path built ON the config types (RULED: no shared struct —
//! the real duplication is the two cross-field rules in `fuel_core::hf_config`,
//! already extracted). Building the rule does not make anyone USE it, so this gate
//! asserts every DENSE causal-LM config that parses a `config.json` routes GQA
//! through `hf_config::num_key_value_heads` (the take-if-present-else-derive rule)
//! rather than a hand-rolled `unwrap_or(num_attention_heads)` fallback — the exact
//! fork that reads correct today and drifts silently later.
//!
//! The cluster is DERIVED mechanically (GQA + rope + vocab, non-MoE), so a NEW
//! causal-LM is auto-included and must call the rule or be exempted — it cannot
//! quietly fork.
//!
//! ⚠️ Exemptions are SELF-VERIFYING: a reason column explains, only an assertion
//! expires. Evidence this matters, from THIS train: #49 added a gate asserting no
//! exemption outlives its edge; #53 removed the `fuel-nn → fuel` edge; the stale
//! exemption reddened the day the edge went, and nothing else would have caught it.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// Cluster members that do NOT yet parse a `config.json`. Self-verified below to
/// genuinely lack a `from_hf_json_str` — the moment one gains a parser without
/// leaving this list (or calling the rule), the gate reds. This list may only SHRINK.
const NOT_YET_PARSED: &[&str] = &[
    "lazy_gemma.rs",
    "lazy_gemma3.rs",
    "lazy_gemma4_text.rs",
    "lazy_granitemoehybrid.rs",
    "lazy_lfm2.rs",
    "lazy_metavoice.rs",
    "lazy_paddleocr_vl_text.rs",
    "lazy_qwen3_moe.rs",
    "lazy_qwen3_vl_text.rs",
    "lazy_recurrent_gemma.rs",
    "lazy_voxtral.rs",
    "lazy_z_image.rs",
];

/// Cluster members STRUCTURALLY exempt: their `num_key_value_heads` is a REQUIRED
/// (non-`Option`) field, so there is no take-if-present-else-derive case to route
/// through the rule. Self-verified below — the day the field becomes `Option`, it
/// acquires a GQA default that bypasses the rule and this exemption must red.
// The two GAP-270 collapses: a serde config whose num_key_value_heads is REQUIRED usize,
// so there is no take-if-present case to route. Each self-verifies below (the day the field
// becomes Option, it acquires a GQA default that bypasses the rule and the exemption reds).
const STRUCTURAL_EXEMPT: &[&str] = &["lazy_llava.rs", "lazy_qwen2_moe.rs"];

/// GAP-279: models carrying the `num_attention_heads * head_dim == hidden_size`
/// guard whose config ALSO declares a `pub head_dim` field — so the guard's claim
/// is falsifiable by a real checkpoint that decouples head_dim.
///
/// ⚠️ WHAT THIS PINS AND WHAT IT DOES NOT. This is the class AS DOCUMENTED in
/// GAP-279 at `bbc969f4`, not a class re-derived here. `mixformer` carries the
/// guard and declares no `head_dim`, so the stated predicate makes it VACUOUS —
/// it is DELIBERATELY outside this pin, because adding one model from the outside
/// is the same act that made the row's split drift from its own census. The
/// census is to be re-derived once, completely. **Do not read a green here as
/// "28 is the complete class."**
const HEAD_DIM_AT_RISK: &[&str] = &[
    "lazy_gemma.rs",
    "lazy_gemma2.rs",
    "lazy_helium.rs",
    "lazy_lfm2.rs",
    "lazy_metavoice.rs",
    "lazy_mistral.rs",
    "lazy_mixtral.rs",
    "lazy_olmo.rs",
    "lazy_olmo2.rs",
    "lazy_paddleocr_vl_text.rs",
    "lazy_persimmon.rs",
    "lazy_phi.rs",
    "lazy_qwen3.rs",
    "lazy_qwen3_moe.rs",
    "lazy_qwen3_vl_text.rs",
    "lazy_smollm3.rs",
    "lazy_stablelm.rs",
    "lazy_starcoder2.rs",
    "lazy_yi.rs",
];

/// GAP-279: carry the same guard but declare NO `head_dim` field, so the guard
/// cannot be violated — the config has no way to express a decoupled head_dim.
///
/// The pair is a PARTITION, and the gate below asserts the SET rather than a
/// count: a model moving VACUOUS -> AT_RISK reds WITH ITS NAME. A count moving
/// 9 to 8 says something changed and not what.
const HEAD_DIM_VACUOUS: &[&str] = &[
    "lazy_bigcode.rs",
    "lazy_falcon.rs",
    "lazy_granite.rs",
    "lazy_granitemoehybrid.rs",
    "lazy_jina_bert.rs",
    "lazy_modernbert.rs",
    "lazy_musicgen.rs",
    "lazy_phi3.rs",
    "lazy_qwen2.rs",
];

const RULE_CALL: &str = "hf_config::num_key_value_heads";

/// Whether the source CALLS the shared rule in code — excludes comment lines so a
/// doc/test mention of `hf_config::num_key_value_heads` cannot fake a call. This is
/// a call-site-level backstop (the (ii) gate); the per-model behavioural tests are
/// the semantic complement (i). A direct call inside a `#[cfg(test)]` block could
/// still fool it — a known limit of the call-site form, which is why (i) exists.
fn calls_rule(src: &str) -> bool {
    src.lines()
        .any(|l| !l.trim_start().starts_with("//") && l.contains(RULE_CALL))
}

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models")
}

/// (basename, source) for every model file.
fn model_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for entry in fs::read_dir(models_dir()).expect("read models dir") {
        let p = entry.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            out.push((name, fs::read_to_string(&p).expect("read model file")));
        }
    }
    out
}

/// The `pub field:` names inside each `pub struct <name>Config { ... }` block.
fn config_struct_fields(src: &str) -> Vec<BTreeSet<String>> {
    let mut structs = Vec::new();
    let bytes = src.as_bytes();
    let mut search = 0;
    while let Some(rel) = src[search..].find("pub struct ") {
        let start = search + rel;
        // struct name must end in "Config" before the '{'
        let after = &src[start + "pub struct ".len()..];
        let brace = match after.find('{') {
            Some(b) => b,
            None => break,
        };
        let name = after[..brace].trim();
        // brace-match the body
        let body_start = start + "pub struct ".len() + brace;
        let mut depth = 0usize;
        let mut i = body_start;
        let mut body_end = body_start;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = i;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if name.ends_with("Config") {
            let body = &src[body_start..body_end];
            let fields: BTreeSet<String> = body
                .lines()
                .filter_map(|l| {
                    let l = l.trim();
                    l.strip_prefix("pub ")
                        .and_then(|r| r.split(':').next().map(|n| n.trim().to_string()))
                })
                .collect();
            structs.push(fields);
        }
        search = body_end.max(start + "pub struct ".len());
    }
    structs
}

/// Files carrying a DENSE causal-LM config: a `*Config` struct with GQA + rope +
/// vocab and NO MoE markers.
fn dense_cluster_files(models: &[(String, String)]) -> BTreeSet<String> {
    let core = ["num_key_value_heads", "rope_theta", "vocab_size"];
    let mut out = BTreeSet::new();
    for (name, src) in models {
        for fields in config_struct_fields(src) {
            let has_core = core.iter().all(|c| fields.contains(*c));
            // MoE models are IN the cluster too — config-from-path must gate them.
            // "MoE" is NOT a predicate for the kv rule (batch-1 finding): mixtral routes
            // the rule like a dense model, qwen2_moe COLLAPSED (GAP-270, exempt below),
            // qwen3_moe is not-yet-parsed. The earlier `!is_moe` filter left MoE sweep
            // targets ungated — a hole this widen closes.
            if has_core {
                out.insert(name.clone());
                break;
            }
        }
    }
    out
}

#[test]
fn every_dense_cluster_config_routes_gqa_through_the_shared_rule() {
    let models = model_sources();
    let by_name: std::collections::BTreeMap<_, _> =
        models.iter().map(|(n, s)| (n.clone(), s.clone())).collect();
    let cluster = dense_cluster_files(&models);

    // Positive control: the mechanical derivation must actually find the cluster.
    // A parser that returns nothing would make every assertion below vacuous.
    assert!(
        cluster.len() >= 30,
        "derived only {} dense-cluster files — the struct parser is broken; \
         a vacuous cluster makes this gate assert nothing",
        cluster.len()
    );

    let not_yet: BTreeSet<&str> = NOT_YET_PARSED.iter().copied().collect();
    let structural: BTreeSet<&str> = STRUCTURAL_EXEMPT.iter().copied().collect();

    let mut forks = Vec::new();
    for file in &cluster {
        let calls = calls_rule(&by_name[file]);
        let exempt = not_yet.contains(file.as_str()) || structural.contains(file.as_str());
        if !calls && !exempt {
            forks.push(file.clone());
        }
    }
    assert!(
        forks.is_empty(),
        "these dense causal-LM configs neither call {RULE_CALL} nor are exempt — \
         they fork the GQA rule (route them through it, or exempt with a \
         self-verifying reason): {forks:?}"
    );

    // No exempt entry may be a rule-caller (double-listed) or a non-cluster file (stale).
    for f in NOT_YET_PARSED.iter().chain(STRUCTURAL_EXEMPT) {
        assert!(
            cluster.contains(*f),
            "exempt entry {f} is not (or no longer) a dense-cluster member — stale; remove it"
        );
    }
}

#[test]
fn not_yet_parsed_entries_genuinely_lack_a_parser() {
    let by_name: std::collections::BTreeMap<_, _> = model_sources().into_iter().collect();
    for f in NOT_YET_PARSED {
        let src = by_name
            .get(*f)
            .unwrap_or_else(|| panic!("NOT_YET_PARSED names a missing file: {f}"));
        assert!(
            !src.contains("fn from_hf_json_str"),
            "{f} now HAS a from_hf_json_str — remove it from NOT_YET_PARSED and either \
             route it through {RULE_CALL} or move it to a self-verifying exemption. \
             A not-yet-parsed exemption that gained a parser is a fork wearing an \
             'unfinished' label."
        );
    }
}

#[test]
fn structural_exempt_kv_heads_is_a_required_field() {
    let by_name: std::collections::BTreeMap<_, _> = model_sources().into_iter().collect();
    for f in STRUCTURAL_EXEMPT {
        let src = by_name.get(*f).expect("structural-exempt file exists");
        assert!(
            src.contains("num_key_value_heads: usize"),
            "{f} is structurally exempt because kv_heads is REQUIRED (usize); that \
             field is no longer `num_key_value_heads: usize`"
        );
        assert!(
            !src.contains("num_key_value_heads: Option"),
            "{f} made num_key_value_heads Optional — it now has a GQA default that \
             bypasses the shared rule; the structural exemption is void, route it \
             through {RULE_CALL}"
        );
    }
}

#[test]
fn exempt_lists_may_only_shrink() {
    // Assert exact counts so a row cannot be ADDED silently; adjust ONLY downward
    // as models are migrated to the rule.
    assert_eq!(
        NOT_YET_PARSED.len(),
        12,
        "NOT_YET_PARSED changed — it may only SHRINK as models gain rule-routed parsers"
    );
    assert_eq!(STRUCTURAL_EXEMPT.len(), 2, "STRUCTURAL_EXEMPT changed");
}

/// GAP-279: the AT_RISK / VACUOUS split must match what the sources say.
///
/// The discriminator is `does the config declare a `pub head_dim` field`, read
/// with the same struct parser the rest of this file uses. The day a VACUOUS
/// model gains an explicit `head_dim`, it becomes falsifiable and this reds
/// naming it — which is the drift the row exists to track.
///
/// BORN-RED, observed: giving `lazy_phi3.rs` a `pub head_dim: usize` field makes
/// this fail with `phi3` named. Restored byte-identically and re-verified, with
/// the crate fingerprint cleared on BOTH transitions so neither reading came
/// from a stale binary.
#[test]
fn head_dim_dispositions_match_the_sources() {
    let models = model_sources();
    let by_name: std::collections::BTreeMap<_, _> =
        models.iter().map(|(n, s)| (n.clone(), s.clone())).collect();

    // Positive control: every pinned file must EXIST. A rename would otherwise
    // silently empty the comparison and pass.
    let pinned: Vec<&str> = HEAD_DIM_AT_RISK
        .iter()
        .chain(HEAD_DIM_VACUOUS.iter())
        .copied()
        .collect();
    assert_eq!(pinned.len(), 28, "GAP-279 documents 28 members");
    for f in &pinned {
        assert!(
            by_name.contains_key(*f),
            "pinned model {f} is not in src/models — renamed or deleted, and a              missing file would make its disposition unobservable rather than wrong"
        );
    }

    let declares_head_dim = |file: &str| -> bool {
        let src = &by_name[file];
        config_struct_fields(src)
            .iter()
            .any(|fields| fields.contains("head_dim"))
    };

    let mut wrong_side: Vec<String> = Vec::new();
    for f in HEAD_DIM_AT_RISK {
        if !declares_head_dim(f) {
            wrong_side.push(format!("{f}: pinned AT_RISK but declares no head_dim"));
        }
    }
    for f in HEAD_DIM_VACUOUS {
        if declares_head_dim(f) {
            wrong_side.push(format!("{f}: pinned VACUOUS but NOW DECLARES head_dim"));
        }
    }

    assert!(
        wrong_side.is_empty(),
        "GAP-279 disposition drift: {wrong_side:?}. A VACUOUS model that gains an explicit head_dim becomes able to express a decoupled config, so the guard's claim becomes falsifiable for it — move it to HEAD_DIM_AT_RISK and update the GAP-279 row in the same change."
    );
}

/// The two GAP-279 lists must stay a PARTITION — disjoint, and no duplicates.
///
/// Cheap, and it is the check whose ABSENCE let the row's split drift from its
/// own census while every count still agreed: 19 + 9 = 28 was true of a set
/// containing a model with no guard and missing one that had it.
#[test]
fn head_dim_lists_are_a_partition() {
    let at: BTreeSet<&str> = HEAD_DIM_AT_RISK.iter().copied().collect();
    let vac: BTreeSet<&str> = HEAD_DIM_VACUOUS.iter().copied().collect();
    assert_eq!(
        at.len(),
        HEAD_DIM_AT_RISK.len(),
        "duplicate in HEAD_DIM_AT_RISK"
    );
    assert_eq!(
        vac.len(),
        HEAD_DIM_VACUOUS.len(),
        "duplicate in HEAD_DIM_VACUOUS"
    );
    let both: Vec<&&str> = at.intersection(&vac).collect();
    assert!(
        both.is_empty(),
        "a model cannot be both AT_RISK and VACUOUS: {both:?}"
    );
}

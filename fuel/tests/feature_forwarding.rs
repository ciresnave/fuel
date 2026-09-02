//! Feature-forwarding gate for the `fuel` facade (restructure Stage 1).
//!
//! The facade's entire job in the feature dimension is to forward every
//! fuel-core feature 1:1: a consumer's `features = ["vulkan"]` on `fuel` must
//! enable exactly `fuel-core/vulkan`. Cargo already rejects a forward to a
//! NONEXISTENT fuel-core feature at build time, so this gate targets the two
//! failures cargo cannot see:
//!
//!   * CROSSING — `vulkan = ["fuel-core/telemetry"]`. Both features exist, the
//!     facade compiles, every consumer-compile passes, and a `--features vulkan`
//!     consumer silently gets no Vulkan. Caught by assertion (ii).
//!   * OMISSION — fuel-core grows a feature and the facade never forwards it.
//!     Caught by assertion (i).
//!
//! The manifest `#[test]` here (call it "a") and the `#[cfg]`-gated EFFECT tests
//! below ("b") are COMPLEMENTARY — neither alone is sufficient, so do NOT delete
//! (a) as redundant because (b) covers the case you happen to think of:
//!   * CROSSING  `vulkan = ["fuel-core/telemetry"]`
//!       -> (b) catches it (`use fuel::vulkan_backend` fails to compile);
//!          (a) also catches it.
//!   * OVER-FORWARD  `vulkan = ["fuel-core/vulkan", "fuel-core/telemetry"]`
//!       -> (b) PASSES (vulkan IS on, module reachable) — only (a) catches it,
//!          via the "forwards to EXACTLY [\"fuel-core/X\"]" length+equality check.
//!
//! Coverage split, stated so "feature forwarding is tested" is never read
//! unqualified: (a) covers all 10 forwardable features TEXTUALLY; (b) covers 2
//! (telemetry, vulkan) by EFFECT — the two runnable on this box. The other eight
//! (cuda, cudnn, nccl, metal, mkl, aocl, onemkl, accelerate) are
//! forwarding-verified and effect-UNVERIFIED (cuda forges; metal is off-platform;
//! mkl/aocl/onemkl/accelerate need absent SDKs).
//!
//! Manifests are embedded at compile time via `include_str!`, so the test is
//! hermetic and cwd-independent.
//!
//! Born-red discipline (one sabotage per assertion arm — keep it reproducible):
//!   * arm (ii): set `vulkan = ["fuel-core/telemetry"]` in fuel/Cargo.toml →
//!     `facade feature `vulkan` must forward EXACTLY to ["fuel-core/vulkan"]`.
//!   * arm (i): delete the `metal` line from fuel/Cargo.toml [features] →
//!     `facade feature set must equal fuel-core's ... Missing: ["metal"]`.
//!
//! The two `#[cfg]`-gated tests below add the EFFECT dimension (proving the
//! forward turns the fuel-core feature ON, not merely that the name resolves):
//! run `cargo test -p fuel --features telemetry` and `--features vulkan`.

use std::collections::BTreeSet;

const FACADE_MANIFEST: &str = include_str!("../Cargo.toml");
const CORE_MANIFEST: &str = include_str!("../../fuel-core/Cargo.toml");

/// Parse the `[features]` table into `(name, [array elements])` pairs, in order.
///
/// Handles multi-line array values by bracket-depth accumulation (fuel-core's
/// `cuda`/`metal`/`vulkan` span several lines) and skips comment/blank lines.
/// Feature keys always sit at the start of a line as `name = ...`; the
/// continuation lines of a multi-line array begin with `"` or `]` and never
/// match a bare `name =`, so key extraction is unambiguous. Element strings in
/// these manifests never contain `[`/`]`, so raw bracket counting is safe.
fn parse_features(manifest: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    let mut in_features = false;
    let mut lines = manifest.lines();
    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        // Section header, e.g. `[features]`, `[dependencies]`, `[[bin]]`.
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_features = trimmed == "[features]";
            continue;
        }
        if !in_features || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some(eq) = line.find('=') else { continue };
        let name = line[..eq].trim().to_string();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            continue;
        }
        // Accumulate the value until brackets balance (single-line balances at once).
        let mut value = line[eq + 1..].to_string();
        let mut depth = value.matches('[').count() as i64 - value.matches(']').count() as i64;
        while depth > 0 {
            let cont = lines
                .next()
                .expect("unterminated feature array in [features]");
            depth += cont.matches('[').count() as i64 - cont.matches(']').count() as i64;
            value.push('\n');
            value.push_str(cont);
        }
        out.push((name, extract_quoted(&value)));
    }
    out
}

/// Return every `"..."`-quoted string in `s`, in order.
fn extract_quoted(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            out.push(s[start..j].to_string());
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn facade_forwards_every_fuel_core_feature_one_to_one() {
    let facade = parse_features(FACADE_MANIFEST);
    let core = parse_features(CORE_MANIFEST);

    // Positive control: an empty parse would make every assertion below vacuous.
    // fuel has 11 features (default + 10); fuel-core has 11. Require a plausible
    // floor so a broken parser/path fails loudly instead of passing green.
    assert!(
        facade.len() >= 5,
        "parsed only {} facade feature(s) — parser or include path broken",
        facade.len()
    );
    assert!(
        core.len() >= 5,
        "parsed only {} fuel-core feature(s) — parser or include path broken",
        core.len()
    );

    let facade_names: BTreeSet<&str> = facade
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| *n != "default")
        .collect();
    let core_names: BTreeSet<&str> = core
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| *n != "default")
        .collect();

    // (i) NO OMISSION / NO EXTRA — the forwarded set is exactly fuel-core's.
    assert_eq!(
        facade_names,
        core_names,
        "facade feature set must equal fuel-core's (minus `default`). \
         Missing (in core, not forwarded): {:?}; Extra (forwarded, not in core): {:?}",
        core_names.difference(&facade_names).collect::<Vec<_>>(),
        facade_names.difference(&core_names).collect::<Vec<_>>(),
    );

    // (ii) NO CROSSING — each facade feature forwards to EXACTLY fuel-core/<same-name>.
    for (name, elems) in &facade {
        if name == "default" {
            assert!(
                elems.is_empty(),
                "facade `default` must stay empty (enable nothing), got {elems:?}"
            );
            continue;
        }
        let expected = format!("fuel-core/{name}");
        assert_eq!(
            elems.len(),
            1,
            "facade feature `{name}` must forward to a single fuel-core feature, got {elems:?}"
        );
        assert_eq!(
            elems[0], expected,
            "facade feature `{name}` must forward EXACTLY to [\"{expected}\"] (crossing?), got {:?}",
            elems[0]
        );
    }
}

// (b) EFFECT, not just text. Under each runnable feature the forwarded fuel-core
// feature must actually be ON and its gated public module reachable through the
// facade glob. Each `use` resolves iff the forward worked end-to-end; a crossing
// (`vulkan = ["fuel-core/telemetry"]`) makes fuel-core/vulkan stay off and the
// module vanish, so this fails to COMPILE under `--features vulkan`.

#[cfg(feature = "telemetry")]
#[test]
fn telemetry_forward_reaches_gated_module() {
    // `fuel::telemetry` exists only when fuel-core/telemetry is enabled.
    #[allow(unused_imports)]
    use fuel::telemetry;
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_forward_reaches_gated_module() {
    // `fuel::vulkan_backend` exists only when fuel-core/vulkan is enabled.
    #[allow(unused_imports)]
    use fuel::vulkan_backend;
}

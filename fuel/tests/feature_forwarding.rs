//! Feature-forwarding gate for the `fuel` facade.
//!
//! The facade re-exports two crates — `fuel-core` (Foundation: tensor API) and,
//! since Stage 2, `fuel-transformers` (Models tier: the model zoo). A consumer
//! enabling `fuel/cuda` must get a CUDA-capable tensor API AND CUDA-built models,
//! so the facade must forward each feature to EVERY dependency that declares it —
//! and to no dependency that does not.
//!
//! This gate DERIVES that rule from the three manifests rather than encoding a
//! snapshot ("5 shared, 5 fuel-core-only"). The snapshot is true today
//! (fuel-transformers carries accelerate/cuda/cudnn/mkl/metal, a strict subset of
//! fuel-core's ten) but a constant assertion stays green the day fuel-transformers
//! gains, say, `telemetry` and the facade is not updated — the models then build
//! without it and nothing complains. Deriving from the manifests makes that RED.
//!
//! What the compiler CANNOT see, and this test therefore must (all invisible to
//! `cargo check` because a facade with an incomplete forward still compiles):
//!   - CROSSING: `cuda = ["fuel-core/telemetry", ...]` — wrong target.
//!   - OMITTED HALF: `cuda = ["fuel-core/cuda"]` — fuel-transformers also declares
//!     `cuda` but is not forwarded to; models build without CUDA.
//!   - SPURIOUS: forwarding to a dep that does not declare the feature.
//!   - GROWTH: a feature added to a dep with no facade entry (the future case).
//!
//! Born-red directions (keep reproducible):
//!   - GROWTH (the one that will actually happen): add `foo = []` to
//!     fuel-transformers/Cargo.toml, do NOT add it to the facade -> assertion (i)
//!     fails, `Missing ... ["foo"]`.
//!   - OMITTED HALF: drop `fuel-transformers/cuda` from the facade's `cuda` ->
//!     assertion (ii) fails for `cuda`.
//!   - CROSSING: point the facade's `cuda` at `fuel-core/telemetry` ->
//!     assertion (ii) fails for `cuda`.
//!
//! Manifests are embedded at compile time via `include_str!`, so the test is
//! hermetic and cwd-independent.
//!
//! The `#[cfg]`-gated tests below add the EFFECT dimension (the forward actually
//! turns the fuel-core feature ON and its gated module is reachable through the
//! facade): run `cargo test -p fuel --features telemetry` and `--features vulkan`.

use std::collections::BTreeSet;

const FACADE_MANIFEST: &str = include_str!("../Cargo.toml");
const CORE_MANIFEST: &str = include_str!("../../fuel-core/Cargo.toml");
const TRANSFORMERS_MANIFEST: &str = include_str!("../../fuel-transformers/Cargo.toml");

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

/// Feature names in a parsed manifest, excluding `default`.
fn feature_names(parsed: &[(String, Vec<String>)]) -> BTreeSet<String> {
    parsed
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| n != "default")
        .collect()
}

#[test]
fn facade_forwards_each_feature_to_every_dep_that_declares_it() {
    let facade = parse_features(FACADE_MANIFEST);
    let core = parse_features(CORE_MANIFEST);
    let transformers = parse_features(TRANSFORMERS_MANIFEST);

    // Positive controls: an empty parse would make every assertion vacuous.
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
    assert!(
        transformers.len() >= 3,
        "parsed only {} fuel-transformers feature(s) — parser or include path broken",
        transformers.len()
    );

    let core_names = feature_names(&core);
    let tf_names = feature_names(&transformers);
    let facade_names = feature_names(&facade);

    // The dependencies the facade forwards to, each with its own declared set.
    // Derived from the manifests — see the module doc for why a hardcoded split
    // would rot silently.
    let deps: [(&str, &BTreeSet<String>); 2] =
        [("fuel-core", &core_names), ("fuel-transformers", &tf_names)];

    // (i) The facade exposes EXACTLY the union of its deps' features: no omission
    // (a dep feature the facade never surfaces) and no extra (a facade feature no
    // dep declares).
    let union: BTreeSet<String> = core_names.union(&tf_names).cloned().collect();
    assert_eq!(
        facade_names,
        union,
        "facade feature set must equal the UNION of its deps' features. \
         Missing (a dep declares it, facade doesn't): {:?}; \
         Extra (facade has it, no dep declares it): {:?}",
        union.difference(&facade_names).collect::<Vec<_>>(),
        facade_names.difference(&union).collect::<Vec<_>>(),
    );

    // (ii) Each facade feature forwards to EXACTLY the deps that declare it.
    for (name, elems) in &facade {
        if name == "default" {
            assert!(
                elems.is_empty(),
                "facade `default` must stay empty (enable nothing), got {elems:?}"
            );
            continue;
        }
        let expected: BTreeSet<String> = deps
            .iter()
            .filter(|(_, feats)| feats.contains(name))
            .map(|(dep, _)| format!("{dep}/{name}"))
            .collect();
        let got: BTreeSet<String> = elems.iter().cloned().collect();
        assert_eq!(
            got, expected,
            "facade feature `{name}` must forward to EXACTLY the deps that declare it \
             — expected {expected:?}, got {got:?}"
        );
    }
}

// EFFECT, not just text. Under each runnable feature the forwarded fuel-core
// feature must actually be ON and its gated public module reachable through the
// facade. Each `use` resolves iff the forward worked end-to-end.

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

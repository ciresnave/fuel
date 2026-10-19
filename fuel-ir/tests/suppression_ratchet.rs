// SPDX-License-Identifier: MIT OR Apache-2.0
//! A ratchet on two suppressions CireSnave called last resorts (2026-10-16):
//! `#[ignore]` on tests and `allow(dead_code)` on items.
//!
//! * `allow(dead_code)`: every tracked `.rs` file with one must be listed in
//!   [`DEAD_CODE_ALLOWED`] with its exact count and the reason it cannot be
//!   gated, deleted or underscore-named. A new one, or an unlisted file, reds
//!   the gate. A file whose count FELL also reds it, so the list shrinks with
//!   the code instead of keeping stale room for a re-introduction.
//! * `#[ignore]`: the total over tracked `.rs` files may not exceed
//!   [`IGNORE_CEILING`], and may not sit below it either (lower the constant in
//!   the same change that removes ignores). The GPU class waits on the
//!   self-hosted runner; the ceiling is what stops it growing meanwhile.
//!
//! Enumeration is `git ls-files`, and the gate FAILS rather than reporting
//! clean when git cannot answer. Matching is on code lines only (a line whose
//! trimmed text starts with `//` is ignored), and the patterns are assembled
//! from pieces so this file cannot match itself.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// Total count of lines starting `#[ignore` across tracked `.rs` files,
/// measured at origin/main e4e143e8 + the retirement PRs. Only ever lowered.
const IGNORE_CEILING: usize = 500;

/// `(path, count, reason)`. Each entry is a finding, not an exemption: the
/// reason names what blocks removal, and the entry goes when that does.
const DEAD_CODE_ALLOWED: &[(&str, usize, &str)] = &[
    (
        "fuel-cpu-backend/src/accelerate.rs",
        1,
        "module-wide allow over Apple FFI; cannot be compiled off macOS, needs a macOS check",
    ),
    (
        "fuel-cuda-backend/src/baracuda/contiguize.rs",
        1,
        "cuda config: not compilable without the forge; retire with the self-hosted CUDA check",
    ),
    (
        "fuel-cuda-backend/src/baracuda/mod.rs",
        1,
        "cuda config: module-wide, submodules land incrementally; needs the CUDA check",
    ),
    (
        "fuel-cuda-backend/src/capture.rs",
        1,
        "cuda config: needs the CUDA check",
    ),
    (
        "fuel-cuda-backend/src/flash_attn.rs",
        2,
        "cuda config: needs the CUDA check",
    ),
    (
        "fuel-cuda-backend/src/storage.rs",
        2,
        "cuda config: needs the CUDA check",
    ),
    (
        "fuel-examples/examples/siglip/main.rs",
        1,
        "serde field the example never reads but must accept from the upstream schema",
    ),
    (
        "fuel-examples/examples/whisper/main.rs",
        2,
        "result structs whose fields are only Debug-printed",
    ),
    (
        "fuel-graph/src/registry/flash_attn.rs",
        1,
        "AttnRecompute: fields read only by tests; cfg-gating them leaves the prod graph-building locals unused",
    ),
    (
        "fuel-metal-kernels/src/kernels/mlx_gemm.rs",
        2,
        "metal config: needs a macOS/Metal check",
    ),
];

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

fn tracked_rs_files(root: &PathBuf) -> Result<Vec<String>, String> {
    let out = Command::new("git")
        .args(["ls-files", "-z", "--", "*.rs"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run `git ls-files`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`git ls-files` failed with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let files: Vec<String> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect();
    if files.is_empty() {
        return Err("`git ls-files` returned nothing; refusing to report clean".into());
    }
    Ok(files)
}

/// Is this a code line (not a `//` comment) containing `needle`?
fn is_code_line_with(line: &str, needle: &str) -> bool {
    let t = line.trim_start();
    !t.starts_with("//") && t.contains(needle)
}

fn count_where(text: &str, pred: impl Fn(&str) -> bool) -> usize {
    text.lines().filter(|l| pred(l)).count()
}

fn dead_code_needle() -> String {
    format!("allow({}_{})", "dead", "code")
}

fn ignore_prefix() -> String {
    format!("#[{}", "ignore")
}

fn read(root: &PathBuf, rel: &str) -> String {
    // Lossy: a stray non-UTF-8 file must not abort the scan of the rest.
    String::from_utf8_lossy(&std::fs::read(root.join(rel)).unwrap_or_default()).into_owned()
}

#[test]
fn dead_code_allows_match_the_reasoned_list_exactly() {
    let root = workspace_root();
    let files = tracked_rs_files(&root).expect("enumerate tracked files");
    let needle = dead_code_needle();

    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for rel in &files {
        let n = count_where(&read(&root, rel), |l| is_code_line_with(l, &needle));
        if n > 0 {
            found.insert(rel.replace('\\', "/"), n);
        }
    }

    let mut problems = Vec::new();
    for (path, n) in &found {
        match DEAD_CODE_ALLOWED.iter().find(|(p, _, _)| p == path) {
            None => problems.push(format!(
                "{path}: {n} new dead-code allow(s), not in the list"
            )),
            Some((_, want, _)) if n > want => problems.push(format!("{path}: {n} > {want} listed")),
            Some((_, want, _)) if n < want => problems.push(format!(
                "{path}: {n} < {want} listed; lower the count or drop the entry"
            )),
            _ => {}
        }
    }
    for (path, want, _) in DEAD_CODE_ALLOWED {
        if !found.contains_key(*path) {
            problems.push(format!(
                "{path}: listed ({want}) but has none; drop the entry"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "dead-code suppression ratchet:\n  {}\n\nDelete, gate behind the config that uses it, or \
         underscore-name the item. Add a list entry only with the reason it cannot be.",
        problems.join("\n  ")
    );
}

#[test]
fn ignore_attribute_total_is_exactly_the_ceiling() {
    let root = workspace_root();
    let files = tracked_rs_files(&root).expect("enumerate tracked files");
    let prefix = ignore_prefix();
    let total: usize = files
        .iter()
        .map(|rel| count_where(&read(&root, rel), |l| l.trim_start().starts_with(&prefix)))
        .sum();
    assert_eq!(
        total,
        IGNORE_CEILING,
        "`#[ignore]` total over {} tracked .rs files is {total}, ceiling {IGNORE_CEILING}. \
         More: ignore is a last resort (run it on the self-hosted runner or make it hermetic). \
         Fewer: lower IGNORE_CEILING in the same change.",
        files.len()
    );
}

/// The scan must be able to see what it claims to count: a positive control
/// for the line predicate, so an always-false matcher cannot pass the gate.
#[test]
fn the_line_predicate_sees_real_code_and_skips_comments() {
    let needle = dead_code_needle();
    let code = format!("    #[{needle}]");
    let comment = format!("    // #[{needle}] mentioned in prose");
    assert!(is_code_line_with(&code, &needle));
    assert!(!is_code_line_with(&comment, &needle));
}

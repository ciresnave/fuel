// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 66 step (c): registration is the rule. A kernel registrar, or an optimizer pass,
//! with no PRODUCTION caller fails this gate unless an owned, expiring row in
//! `registrar_caller_allowlist.tsv` says why.
//!
//! Why: `register_mkl_cpu_kernels`/`register_aocl_cpu_kernels` had no caller for months, and
//! `insert_residency_evictions` had none either; a function nobody calls is untested in the way
//! that matters.
//!
//! # Population (read before trusting a green)
//! * **Registrars:** every column-0 `pub fn register_*kernels*` in a tracked `.rs` file.
//! * **Passes:** every column-0 `pub fn` whose name starts `insert_`, `fuse_`, `promote_` or
//!   `lower_` in `fuel-graph/src/opt.rs` and `fuel-dispatch/src/optimize.rs`.
//!
//! A *production caller* is an identifier token equal to the name, on a line that is not a
//! comment, `use` line or the definition, in a tracked `.rs` file outside any path component
//! `tests`/`benches`/`examples`/`target`, outside `#[cfg(test)]` regions (naive brace depth). The
//! token may be a call or a function value (`extend_global_bindings(register_x)`).
//!
//! # What it cannot see
//! * A caller that is itself never called (no transitive reachability).
//! * A caller behind a cargo feature counts: this gate does not know which features ship.
//! * A registrar or pass outside the population above (a `pub(crate) fn`, a method, another
//!   file): it is not looked at at all.
//! * Metal: there is no Metal registrar to find (ROADMAP row 113), so nothing to allowlist.
//!
//! # Allowlist teeth
//! Each row is `name TAB owner TAB expiry(YYYY-MM-DD) TAB reason`. A row fails the gate when its
//! name is no longer in the population, when the name now HAS a production caller (the cause
//! dissolved: delete the row), or when its expiry date has passed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const PASS_FILES: [&str; 2] = ["fuel-graph/src/opt.rs", "fuel-dispatch/src/optimize.rs"];
const PASS_PREFIXES: [&str; 4] = ["insert_", "fuse_", "promote_", "lower_"];

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        if let Ok(s) = std::fs::read_to_string(dir.join("Cargo.toml"))
            && s.contains("[workspace]")
        {
            return dir;
        }
        assert!(
            dir.pop(),
            "no workspace Cargo.toml above CARGO_MANIFEST_DIR"
        );
    }
}

/// Tracked `.rs` files (workspace-relative, forward slashes). Enumerates from the git index; a
/// disk walk only if git cannot answer, and it says so.
fn rs_files(root: &Path) -> Vec<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "*.rs"])
        .output();
    if let Ok(o) = out
        && o.status.success()
    {
        let v: Vec<String> = String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(str::to_string)
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    eprintln!("registrar_callers: git ls-files unavailable; WALKING the disk instead");
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if name != "target" && name != ".git" {
                walk(root, &p, out);
            }
        } else if name.ends_with(".rs")
            && let Ok(rel) = p.strip_prefix(root)
        {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

// Written as escapes so brace-counting analysers (Lizard) do not read them as scope.
const OPEN: char = '\u{7b}';
const CLOSE: char = '\u{7d}';

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn idents(line: &str) -> Vec<&str> {
    line.split(|c: char| !is_ident(c))
        .filter(|s| !s.is_empty())
        .collect()
}

fn defined_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("pub fn ")?;
    let end = rest.find(|c: char| !is_ident(c))?;
    Some(&rest[..end])
}

fn is_registrar(name: &str) -> bool {
    name.starts_with("register_") && name.contains("kernels")
}

fn is_pass(path: &str, name: &str) -> bool {
    PASS_FILES.contains(&path) && PASS_PREFIXES.iter().any(|p| name.starts_with(p))
}

fn is_test_path(path: &str) -> bool {
    path.split('/')
        .any(|c| matches!(c, "tests" | "benches" | "examples" | "target"))
}

/// Production lines of `src`: not a comment, not inside a `#[cfg(test)]` region (naive depth).
fn production_lines(src: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut armed, mut depth): (bool, i64) = (false, 0);
    for line in src.lines() {
        let t = line.trim();
        if depth > 0 || armed {
            for c in line.chars() {
                if c == OPEN {
                    depth += 1;
                    armed = false;
                } else if c == CLOSE {
                    depth -= 1;
                }
            }
            if depth <= 0 && !armed {
                depth = 0;
            }
            continue;
        }
        if t.starts_with("#[cfg(test)]") {
            armed = true;
            continue;
        }
        if t.starts_with("//") {
            continue;
        }
        out.push(line);
    }
    out
}

struct Scan {
    /// name -> defining path
    population: BTreeMap<String, String>,
    /// names with at least one production caller
    called: std::collections::BTreeSet<String>,
}

/// Scan `(path, source)` pairs.
fn scan(files: &[(String, String)]) -> Scan {
    let mut population = BTreeMap::new();
    for (path, src) in files {
        if is_test_path(path) {
            continue;
        }
        for line in production_lines(src) {
            if let Some(n) = defined_name(line)
                && (is_registrar(n) || is_pass(path, n))
            {
                population.insert(n.to_string(), path.clone());
            }
        }
    }
    let mut called = std::collections::BTreeSet::new();
    for (path, src) in files {
        if is_test_path(path) {
            continue;
        }
        for line in production_lines(src) {
            let t = line.trim_start();
            if t.starts_with("use ") || t.starts_with("pub use ") {
                continue;
            }
            let def = defined_name(line);
            for id in idents(line) {
                if population.contains_key(id) && def != Some(id) {
                    called.insert(id.to_string());
                }
            }
        }
    }
    Scan { population, called }
}

struct Row {
    name: String,
    expiry: String,
}

fn parse_allowlist(text: &str) -> Vec<Row> {
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert!(
                f.len() == 4,
                "allowlist row needs 4 tab-separated fields: {l:?}"
            );
            assert!(!f[1].trim().is_empty(), "allowlist row has no owner: {l:?}");
            assert!(
                f[2].len() == 10 && f[2].as_bytes()[4] == b'-' && f[2].as_bytes()[7] == b'-',
                "allowlist expiry must be YYYY-MM-DD: {l:?}"
            );
            Row {
                name: f[0].to_string(),
                expiry: f[2].to_string(),
            }
        })
        .collect()
}

/// Today as `YYYY-MM-DD` (UTC), from the system clock, std only.
fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Every problem with `scan` against `rows` as of `today`; empty means green.
fn problems(s: &Scan, rows: &[Row], today: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (name, path) in &s.population {
        if !s.called.contains(name) && !rows.iter().any(|r| &r.name == name) {
            out.push(format!(
                "{name} ({path}) has no production caller: wire it, or add an owned, expiring row to registrar_caller_allowlist.tsv"
            ));
        }
    }
    for r in rows {
        if !s.population.contains_key(&r.name) {
            out.push(format!(
                "STALE allowlist row {}: no such registrar or pass",
                r.name
            ));
        } else if s.called.contains(&r.name) {
            out.push(format!(
                "STALE allowlist row {}: it now has a production caller, delete the row",
                r.name
            ));
        }
        if r.expiry.as_str() < today {
            out.push(format!(
                "EXPIRED allowlist row {} (expiry {})",
                r.name, r.expiry
            ));
        }
    }
    out
}

fn f(path: &str, src: &str) -> (String, String) {
    (path.to_string(), src.to_string())
}

#[test]
fn every_registrar_and_pass_has_a_production_caller_or_an_owned_expiring_row() {
    let root = workspace_root();
    let files: Vec<(String, String)> = rs_files(&root)
        .into_iter()
        .filter_map(|p| std::fs::read_to_string(root.join(&p)).ok().map(|s| (p, s)))
        .collect();
    let s = scan(&files);
    eprintln!(
        "registrar_callers: {} files read, population {} ({} called)",
        files.len(),
        s.population.len(),
        s.called.len()
    );
    // The real-tree control: the scan must see known members of both populations, and must see
    // a caller for a registrar that is wired and a pass that is wired.
    for known in [
        "register_cpu_kernels",
        "register_mkl_cpu_kernels",
        "register_aocl_cpu_kernels",
        "insert_copies",
        "insert_cross_device_copies",
    ] {
        assert!(
            s.population.contains_key(known),
            "scan lost {known}: it is not looking where it should"
        );
    }
    for wired in [
        "register_cpu_kernels",
        "register_mkl_cpu_kernels",
        "insert_cross_device_copies",
    ] {
        assert!(
            s.called.contains(wired),
            "scan sees no production caller of {wired}, which has one"
        );
    }
    let text = std::fs::read_to_string(root.join("fuel-ir/tests/registrar_caller_allowlist.tsv"))
        .expect("registrar_caller_allowlist.tsv");
    let p = problems(&s, &parse_allowlist(&text), &today_utc());
    assert!(p.is_empty(), "\n{}\n", p.join("\n"));
}

#[test]
fn positive_control_an_unwired_registrar_and_an_unwired_pass_are_reported() {
    let files = vec![
        f("a/src/lib.rs", "pub fn register_x_kernels(t: &mut T) {}\n"),
        f(
            "fuel-graph/src/opt.rs",
            "pub fn insert_foo(g: &G) {}\npub fn insert_foo_in(g: &G) {}\nfn helper() { insert_foo_in(g); }\n",
        ),
    ];
    let p = problems(&scan(&files), &[], "2026-01-01");
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p.iter().any(|m| m.starts_with("register_x_kernels")));
    // `insert_foo_in(` must not count as a caller of `insert_foo`.
    assert!(p.iter().any(|m| m.starts_with("insert_foo ")));
}

#[test]
fn a_function_value_or_a_call_in_production_code_counts_as_a_caller() {
    let files = vec![
        f("a/src/lib.rs", "pub fn register_x_kernels(t: &mut T) {}\n"),
        f(
            "b/src/lib.rs",
            "fn init() { extend_global_bindings(register_x_kernels); }\n",
        ),
    ];
    assert!(problems(&scan(&files), &[], "2026-01-01").is_empty());
}

#[test]
fn callers_in_tests_cfg_test_regions_comments_and_use_lines_do_not_count() {
    let files = vec![
        f("a/src/lib.rs", "pub fn register_x_kernels(t: &mut T) {}\n"),
        f("a/tests/t.rs", "fn t() { register_x_kernels(&mut t); }\n"),
        f(
            "b/src/lib.rs",
            "use a::register_x_kernels;\n// register_x_kernels(t);\n/// register_x_kernels(t)\n#[cfg(test)]\nmod tests {\n    fn t() {\n        register_x_kernels(&mut t);\n    }\n}\n",
        ),
    ];
    let p = problems(&scan(&files), &[], "2026-01-01");
    assert_eq!(p.len(), 1, "{p:?}");
    // And code after the cfg(test) block is production again.
    let files = vec![
        f("a/src/lib.rs", "pub fn register_x_kernels(t: &mut T) {}\n"),
        f(
            "b/src/lib.rs",
            "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\nfn init() { register_x_kernels(t); }\n",
        ),
    ];
    assert!(problems(&scan(&files), &[], "2026-01-01").is_empty());
}

#[test]
fn allowlist_rows_cover_expire_and_go_stale() {
    let files = vec![f(
        "a/src/lib.rs",
        "pub fn register_x_kernels(t: &mut T) {}\n",
    )];
    let s = scan(&files);
    let live = parse_allowlist("register_x_kernels\tfuel lane\t2026-12-31\twhy\n");
    assert!(problems(&s, &live, "2026-06-01").is_empty());
    // Expired.
    let p = problems(&s, &live, "2027-01-01");
    assert!(p.iter().any(|m| m.starts_with("EXPIRED")), "{p:?}");
    // Stale: the name no longer exists.
    let gone = parse_allowlist("register_y_kernels\tfuel lane\t2026-12-31\twhy\n");
    let p = problems(&s, &gone, "2026-06-01");
    assert!(
        p.iter()
            .any(|m| m.contains("STALE") && m.contains("no such")),
        "{p:?}"
    );
    // Stale: the cause dissolved because the name now has a caller.
    let wired = scan(&[
        f("a/src/lib.rs", "pub fn register_x_kernels(t: &mut T) {}\n"),
        f("b/src/lib.rs", "fn init() { register_x_kernels(t); }\n"),
    ]);
    let p = problems(&wired, &live, "2026-06-01");
    assert!(
        p.iter().any(|m| m.contains("now has a production caller")),
        "{p:?}"
    );
}

#[test]
fn today_utc_is_a_plausible_date() {
    let t = today_utc();
    assert!(
        t.as_str() > "2026-10-01" && t.as_str() < "2100-01-01",
        "{t}"
    );
}

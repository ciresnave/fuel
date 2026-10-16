// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hugging Face Hub prerequisites for tests, as a [`Hardware::Hub`] family.
//!
//! A test that needs a real checkpoint used to be `#[ignore]`d because it hit
//! the Hub. That hid it from every run, including the ones on a machine that
//! HAS the checkpoint. This module lets the test run wherever its prerequisite
//! is present and say so, loudly, where it is not:
//!
//! - [`require_cached`]: the files are already in the local HF cache. It NEVER
//!   downloads (a plain `cargo test` must not pull a multi-GB model); on a
//!   miss it declares a skip with the cause. On the self-hosted runner the
//!   cache is pre-populated and `FUEL_REQUIRE_HUB=1` turns a miss into a
//!   FAILURE, so the evidence cannot silently stop being produced.
//! - [`require_reachable`]: the test genuinely calls the Hub API (it lists a
//!   repo's files) so a cache cannot satisfy it; the prerequisite is a TCP
//!   connection to `huggingface.co:443`.
//!
//! Deliberately dependency-free (the crate is): the cache layout is read with
//! `std` alone.

use std::path::{Path, PathBuf};

use crate::hardware::{Hardware, Missing, skip};

/// What kind of Hub repo a cache entry belongs to (the directory prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoKind {
    Model,
    Dataset,
}

impl RepoKind {
    const fn prefix(self) -> &'static str {
        match self {
            RepoKind::Model => "models",
            RepoKind::Dataset => "datasets",
        }
    }
}

/// The HF hub cache directory, resolved the way `hf-hub` does: `HF_HUB_CACHE`,
/// then `HUGGINGFACE_HUB_CACHE`, then `$HF_HOME/hub`, then
/// `~/.cache/huggingface/hub`.
pub fn hf_cache_dir() -> Option<PathBuf> {
    for var in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Some(v) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(v));
        }
    }
    if let Some(v) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(v).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Whether every one of `files` exists in some snapshot of `repo_id` under
/// `cache_root`. Pure over a path so it is testable without the real cache.
pub fn is_cached_in(cache_root: &Path, kind: RepoKind, repo_id: &str, files: &[&str]) -> bool {
    let dir = cache_root.join(format!("{}--{}", kind.prefix(), repo_id.replace('/', "--")));
    let Ok(snapshots) = std::fs::read_dir(dir.join("snapshots")) else {
        return false;
    };
    snapshots
        .flatten()
        .any(|snap| files.iter().all(|f| snap.path().join(f).exists()))
}

/// `true` if `files` of `repo_id` are in the local HF cache; otherwise declares
/// a [`Hardware::Hub`] skip (fatal under `FUEL_REQUIRE_HUB`) and returns
/// `false`. The caller returns on `false`.
pub fn require_cached(kind: RepoKind, repo_id: &str, files: &[&str]) -> bool {
    let Some(root) = hf_cache_dir() else {
        skip(
            Hardware::Hub,
            Missing::device("cannot locate the HF cache directory (no HF_HOME/HOME)"),
        );
        return false;
    };
    if is_cached_in(&root, kind, repo_id, files) {
        return true;
    }
    skip(
        Hardware::Hub,
        Missing::device(format!(
            "{repo_id} {files:?} not in the HF cache at {} (never downloaded implicitly; \
             populate it, e.g. `hf download {repo_id}`)",
            root.display()
        )),
    );
    false
}

/// `true` if `huggingface.co:443` accepts a connection within a few seconds;
/// otherwise declares a [`Hardware::Hub`] skip and returns `false`.
pub fn require_reachable() -> bool {
    use std::net::ToSocketAddrs;
    let reachable = "huggingface.co:443"
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next())
        .is_some_and(|addr| {
            std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)).is_ok()
        });
    if !reachable {
        skip(
            Hardware::Hub,
            Missing::device("huggingface.co:443 is not reachable"),
        );
    }
    reachable
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fuel-hub-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(root: &Path, repo_dir: &str, rev: &str, file: &str) {
        let p = root.join(repo_dir).join("snapshots").join(rev);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join(file), b"x").unwrap();
    }

    #[test]
    fn a_present_file_is_cached_and_an_absent_one_is_not() {
        let root = scratch("present");
        put(
            &root,
            "models--timm--repvgg_a0.rvgg_in1k",
            "abc",
            "model.safetensors",
        );
        let id = "timm/repvgg_a0.rvgg_in1k";
        assert!(is_cached_in(
            &root,
            RepoKind::Model,
            id,
            &["model.safetensors"]
        ));
        // Sabotage siblings: wrong file, wrong repo, wrong kind each read as a miss.
        assert!(!is_cached_in(&root, RepoKind::Model, id, &["config.json"]));
        assert!(!is_cached_in(
            &root,
            RepoKind::Model,
            "timm/other",
            &["model.safetensors"]
        ));
        assert!(!is_cached_in(
            &root,
            RepoKind::Dataset,
            id,
            &["model.safetensors"]
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn every_file_must_be_in_one_snapshot_not_spread_across_snapshots() {
        let root = scratch("spread");
        put(&root, "models--a--b", "rev1", "model.safetensors");
        put(&root, "models--a--b", "rev2", "config.json");
        assert!(!is_cached_in(
            &root,
            RepoKind::Model,
            "a/b",
            &["model.safetensors", "config.json"]
        ));
        put(&root, "models--a--b", "rev2", "model.safetensors");
        assert!(is_cached_in(
            &root,
            RepoKind::Model,
            "a/b",
            &["model.safetensors", "config.json"]
        ));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_cache_root_is_a_miss_not_a_panic() {
        let root = std::env::temp_dir().join("fuel-hub-test-does-not-exist-xyz");
        assert!(!is_cached_in(&root, RepoKind::Model, "a/b", &["f"]));
    }
}

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Verify the workspace's `.crate` tarballs build for a consumer, from the tarball alone.

A published crate contains only its own directory. A file that reaches outside it
(`include_str!("../../docs/x")`) builds in the workspace and fails for every consumer;
`cargo package --no-verify` does not notice. This script closes that:

  1. SCAN   every `.rs` file inside every `.crate`: each `include_str!`/`include_bytes!`/
            `include!`/`#[path]` target must be a member of the SAME tarball. An escape in
            `src/` or `build.rs` is a hard failure with no allowlist. An escape in `tests/`,
            `benches/` or `examples/` (those cannot compile from a tarball anyway: workspace
            dev-dependencies are path-only and `cargo publish` strips them) must be listed in
            `scripts/packaged-escapes-allowlist.txt`. Runtime `CARGO_MANIFEST_DIR/..` reads are
            counted the same way. The allowlist is a RATCHET: an unlisted escape fails, and a
            listed entry that no longer exists fails too, so it can shrink and never grow.
  2. BUILD  extract every `.crate` into a throwaway workspace, point each crate at the others
            through `[patch.crates-io]`, and `cargo check` it: what a consumer builds, from
            only what was packaged.

Usage:
  verify-packaged-crates.py selftest
  verify-packaged-crates.py scan  <dir-of-.crate-files> [--allowlist FILE]
  verify-packaged-crates.py build <dir-of-.crate-files> <lockfile> [--exclude CRATE ...]
"""
import os
import posixpath
import re
import shutil
import subprocess  # nosec B404 -- only used with fixed argument lists and no shell
import sys
import tarfile
import tempfile

ALLOWLIST_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packaged-escapes-allowlist.txt")

# `include_str!("p")`, `include_bytes!("p")`, `include!("p")`, `#[path = "p"]`; whitespace may wrap.
INCLUDE = re.compile(r'\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"|#\s*\[\s*path\s*=\s*"([^"]+)"')
# A runtime read that leaves the crate: CARGO_MANIFEST_DIR joined with a ".." component.
RUNTIME = re.compile(r'env!\(\s*"CARGO_MANIFEST_DIR"\s*\)[^;]{0,200}?"\.\.(?:/[^"]*)?"', re.S)


def strip_line_comments(text):
    """Drop whole-line `//` comments (incl. doc comments), which quote the macros they discuss."""
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("//"))


def crate_members(path):
    """Return (name, {relative member path: bytes-or-None for non-.rs}) for one .crate."""
    with tarfile.open(path, "r:gz") as tf:
        files = {}
        root = None
        for m in tf.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/", 1)
            root = parts[0]
            rel = parts[1] if len(parts) > 1 else parts[0]
            data = tf.extractfile(m).read() if rel.endswith(".rs") else None
            files[rel] = data
    return root, files


def classify(rel):
    first = rel.split("/", 1)[0]
    if first in ("tests", "benches", "examples"):
        return "aux"
    return "src"  # src/, build.rs, anything else compiled for a consumer


def scan_crate(path):
    """Yield (crate, file, kind, target) for each escape; kind is 'src', 'aux' or 'runtime'."""
    root, files = crate_members(path)
    name = root.rsplit("-", 1)[0]
    members = set(files)
    for rel, data in sorted(files.items()):
        if data is None:
            continue
        text = strip_line_comments(data.decode("utf8", "replace"))
        for m in INCLUDE.finditer(text):
            p = m.group(1) or m.group(2)
            tgt = posixpath.normpath(posixpath.join(posixpath.dirname(rel), p))
            if tgt.startswith("..") or tgt not in members:
                yield (name, rel, classify(rel), p)
        for m in RUNTIME.finditer(text):
            yield (name, rel, "runtime", " ".join(m.group(0).split())[:80])


def read_allowlist(path):
    """Lines: `crate<TAB>file<TAB>kind<TAB>count<TAB>reason`. `#` comments."""
    entries = {}
    if not os.path.exists(path):
        return entries
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        crate, rel, kind, count, reason = line.split("\t", 4)
        entries[(crate, rel, kind)] = (int(count), reason)
    return entries


def collect_escapes(crate_dir, crates):
    """Return (hard src escapes, {(crate, file, kind): count} of allowlistable escapes)."""
    found = {}
    hard = []
    for f in crates:
        for name, rel, kind, tgt in scan_crate(os.path.join(crate_dir, f)):
            if kind == "src":
                hard.append((name, rel, tgt))
            else:
                found[(name, rel, kind)] = found.get((name, rel, kind), 0) + 1
    return hard, found


def judge_escape(key, n, want):
    """Print one found escape against its allowlist entry; return 1 if it fails, else 0."""
    allowed = want[0] if want else 0
    reason = f"  -- {want[1]}" if want else ""
    print(f"  {'ok ' if allowed == n else 'NEW'} {key[0]}/{key[1]} [{key[2]}] x{n}{reason}")
    if allowed == n:
        return 0
    print(f"FAIL unlisted or grown escape: {key} found {n}, allowed {allowed}")
    return 1


def check_allowlist(found, allow):
    """Count unlisted, grown and stale allowlist entries, printing each."""
    bad = sum(judge_escape(key, found[key], allow.get(key)) for key in sorted(found))
    stale = [key for key in sorted(allow) if key not in found]
    for key in stale:
        print(f"FAIL allowlist entry no longer exists (remove it): {key}")
    return bad + len(stale)


def scan(crate_dir, allow_path):
    crates = sorted(f for f in os.listdir(crate_dir) if f.endswith(".crate"))
    hard, found = collect_escapes(crate_dir, crates)
    allow = read_allowlist(allow_path)
    print(f"scanned {len(crates)} .crate files")
    for name, rel, tgt in hard:
        print(f"FAIL src escape (no allowlist for src/): {name}/{rel} -> {tgt}")
    print(f"allowlisted escapes found: {sum(found.values())} in {len(found)} files "
          f"(allowlist entries: {len(allow)})")
    return len(hard) + check_allowlist(found, allow)


def make_crate(dirpath, name, files):
    path = os.path.join(dirpath, f"{name}-0.0.0.crate")
    with tarfile.open(path, "w:gz") as tf:
        for rel, body in files.items():
            full = os.path.join(dirpath, "_stage", rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "w", encoding="utf8").write(body)
            tf.add(full, arcname=f"{name}-0.0.0/{rel}")
    return path


def selftest():
    """Known-bad controls: each planted case must be flagged, each clean case must not."""
    fails = 0

    def expect(label, got, want):
        nonlocal fails
        ok = got == want
        print(f"selftest {'PASS' if ok else 'FAIL'}: {label} (got {got}, want {want})")
        fails += 0 if ok else 1

    with tempfile.TemporaryDirectory() as d:
        make_crate(d, "bad_src", {"src/lib.rs": 'const X: &str = include_str!("../x.md");\n'})
        make_crate(d, "bad_wrapped", {"src/lib.rs": 'const X: &[u8] = include_bytes!(\n    "../../y.bin"\n);\n'})
        make_crate(d, "good", {"src/lib.rs": 'const X: &str = include_str!("data.txt");\n', "src/data.txt": "x"})
        make_crate(d, "aux_only", {"src/lib.rs": "", "tests/t.rs": 'const X: &str = include_str!("../../z");\n'})
        make_crate(d, "runtime", {"src/lib.rs": 'fn f() { let _ = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../q"); }\n'})
        empty = os.path.join(d, "empty.txt")
        open(empty, "w").close()

        def count(name, kind):
            return sum(1 for e in scan_crate(os.path.join(d, f"{name}-0.0.0.crate")) if e[2] == kind)

        expect("planted src include('../x') is a src escape", count("bad_src", "src"), 1)
        expect("planted multi-line include_bytes! is a src escape", count("bad_wrapped", "src"), 1)
        expect("an include inside the tarball is not an escape", sum(1 for _ in scan_crate(os.path.join(d, "good-0.0.0.crate"))), 0)
        expect("a tests/ escape is classed aux, not src", (count("aux_only", "aux"), count("aux_only", "src")), (1, 0))
        expect("a runtime CARGO_MANIFEST_DIR/.. read is counted", count("runtime", "runtime"), 1)
        # Whole-job controls: the planted src escape fails the scan; an unlisted aux escape fails;
        # a listed one passes; a stale entry fails.
        only = lambda names: _dir_with(d, names)
        expect("scan FAILS on a planted src escape", scan(only(["bad_src"]), empty) > 0, True)
        expect("scan FAILS on an unlisted aux escape", scan(only(["aux_only"]), empty) > 0, True)
        listed = os.path.join(d, "listed.txt")
        open(listed, "w").write("aux_only\ttests/t.rs\taux\t1\tcontrol\n")
        expect("scan PASSES a listed aux escape", scan(only(["aux_only"]), listed), 0)
        stale = os.path.join(d, "stale.txt")
        open(stale, "w").write("aux_only\ttests/t.rs\taux\t1\tcontrol\ngone\tsrc/x.rs\truntime\t1\tstale\n")
        expect("scan FAILS on a stale allowlist entry", scan(only(["aux_only"]), stale) > 0, True)
        expect("scan PASSES a clean crate", scan(only(["good"]), empty), 0)
    print("selftest:", "OK" if fails == 0 else f"{fails} FAILED")
    return fails


def _dir_with(src_dir, names):
    """Copy the named planted crates into a fresh directory so each case scans alone."""
    d = tempfile.mkdtemp(dir=src_dir)
    for n in names:
        shutil.copy(os.path.join(src_dir, f"{n}-0.0.0.crate"), d)
    return d


def safe_extract(tf, dest):
    """Extract a tarball, refusing absolute paths, `..` components and links.

    The tarballs are ours (`cargo package`), but this script is the guard, so it does
    not trust them: members are validated, then extracted with the `data` filter.
    """
    root = os.path.realpath(dest)
    for m in tf.getmembers():
        target = os.path.realpath(os.path.join(dest, m.name))
        if m.issym() or m.islnk() or os.path.isabs(m.name) or os.path.commonpath([root, target]) != root:
            raise ValueError(f"unsafe tar member: {m.name}")
    tf.extractall(dest, filter="data")


def write_workspace(ws, names):
    """Write the throwaway workspace manifest: every extracted crate patched in by path."""
    patch = [f'{d.rsplit("-", 1)[0]} = {{ path = "{d}" }}' for d in names]
    root = (
        "[workspace]\nresolver = \"2\"\nmembers = [\n"
        + "".join(f'    "{d}",\n' for d in names)
        + "]\n\n[patch.crates-io]\n" + "\n".join(patch) + "\n"
    )
    with open(os.path.join(ws, "Cargo.toml"), "w", encoding="utf8") as fh:
        fh.write(root)


CRATE_NAME = re.compile(r"[A-Za-z0-9_-]+")


def extract_all(crate_dir, crates, ws):
    """Extract every tarball into `ws`; return the name-version directory names."""
    for f in crates:
        with tarfile.open(os.path.join(crate_dir, f), "r:gz") as tf:
            safe_extract(tf, ws)
    return [f[: -len(".crate")] for f in crates]


def cargo_check_argv(excludes):
    """argv for `cargo check --workspace`: constants plus crate names validated as plain identifiers."""
    for e in excludes:
        if not CRATE_NAME.fullmatch(e):
            raise ValueError(f"not a crate name: {e!r}")
    return ["cargo", "check", "--workspace"] + [x for e in excludes for x in ("--exclude", e)]


def build(crate_dir, lockfile, excludes):
    crates = sorted(f for f in os.listdir(crate_dir) if f.endswith(".crate"))
    argv = cargo_check_argv(excludes)
    ws = tempfile.mkdtemp(prefix="packaged-ws-")
    names = extract_all(crate_dir, crates, ws)
    write_workspace(ws, names)
    if lockfile and os.path.exists(lockfile):
        shutil.copy(lockfile, os.path.join(ws, "Cargo.lock"))
    print(f"BUILD throwaway workspace {ws}: {len(names)} extracted crates; {' '.join(argv)}")
    print("NAMED HOLES: DOCS_RS is set by the caller, so the CUDA/forge (nvcc) compile of "
          "baracuda-kernels-sys is NOT verified; excluded crates are not built at all.")
    # argv is constants plus identifier-validated crate names; no shell; cwd is our own temp dir.
    return subprocess.run(argv, cwd=ws).returncode  # NOSONAR # nosec B603


def cmd_selftest(argv):
    return 1 if selftest() else 0


def cmd_scan(argv):
    allow = ALLOWLIST_DEFAULT
    if "--allowlist" in argv:
        allow = argv[argv.index("--allowlist") + 1]
    return 1 if scan(argv[2], allow) else 0


def cmd_build(argv):
    ex = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude"]
    return build(argv[2], argv[3], ex)


COMMANDS = {"selftest": cmd_selftest, "scan": cmd_scan, "build": cmd_build}


def main(argv):
    handler = COMMANDS.get(argv[1]) if len(argv) > 1 else None
    if handler is None:
        print(__doc__)
        return 2
    return handler(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

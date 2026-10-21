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
import subprocess
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


def scan(crate_dir, allow_path):
    crates = sorted(f for f in os.listdir(crate_dir) if f.endswith(".crate"))
    found = {}
    hard = []
    for f in crates:
        for name, rel, kind, tgt in scan_crate(os.path.join(crate_dir, f)):
            if kind == "src":
                hard.append((name, rel, tgt))
            else:
                found[(name, rel, kind)] = found.get((name, rel, kind), 0) + 1
    allow = read_allowlist(allow_path)
    bad = 0
    print(f"scanned {len(crates)} .crate files")
    for name, rel, tgt in hard:
        print(f"FAIL src escape (no allowlist for src/): {name}/{rel} -> {tgt}")
        bad += 1
    print(f"allowlisted escapes found: {sum(found.values())} in {len(found)} files "
          f"(allowlist entries: {len(allow)})")
    for key in sorted(found):
        want = allow.get(key)
        tag = "ok " if want and want[0] == found[key] else "NEW"
        print(f"  {tag} {key[0]}/{key[1]} [{key[2]}] x{found[key]}" + (f"  -- {want[1]}" if want else ""))
        if not want or want[0] != found[key]:
            print(f"FAIL unlisted or grown escape: {key} found {found[key]}, allowed {want[0] if want else 0}")
            bad += 1
    for key, (count, reason) in sorted(allow.items()):
        if key not in found:
            print(f"FAIL allowlist entry no longer exists (remove it): {key}")
            bad += 1
    return bad


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


def build(crate_dir, lockfile, excludes):
    crates = sorted(f for f in os.listdir(crate_dir) if f.endswith(".crate"))
    ws = tempfile.mkdtemp(prefix="packaged-ws-")
    names = []
    for f in crates:
        with tarfile.open(os.path.join(crate_dir, f), "r:gz") as tf:
            tf.extractall(ws)
        names.append(f[: -len(".crate")])  # name-version directory
    patch = []
    members = []
    for d in names:
        name = d.rsplit("-", 1)[0]
        members.append(d)
        patch.append(f'{name} = {{ path = "{d}" }}')
    root = (
        "[workspace]\nresolver = \"2\"\nmembers = [\n"
        + "".join(f'    "{m}",\n' for m in members)
        + "]\n\n[patch.crates-io]\n" + "\n".join(patch) + "\n"
    )
    open(os.path.join(ws, "Cargo.toml"), "w", encoding="utf8").write(root)
    if lockfile and os.path.exists(lockfile):
        shutil.copy(lockfile, os.path.join(ws, "Cargo.lock"))
    cmd = ["cargo", "check", "--workspace"] + [x for e in excludes for x in ("--exclude", e)]
    print(f"BUILD throwaway workspace {ws}: {len(members)} extracted crates; {' '.join(cmd)}")
    print("NAMED HOLES: DOCS_RS is set by the caller, so the CUDA/forge (nvcc) compile of "
          "baracuda-kernels-sys is NOT verified; excluded crates are not built at all.")
    return subprocess.run(cmd, cwd=ws).returncode


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "selftest":
        return 1 if selftest() else 0
    if cmd == "scan":
        allow = ALLOWLIST_DEFAULT
        if "--allowlist" in argv:
            allow = argv[argv.index("--allowlist") + 1]
        return 1 if scan(argv[2], allow) else 0
    if cmd == "build":
        ex = [argv[i + 1] for i, a in enumerate(argv) if a == "--exclude"]
        return build(argv[2], argv[3], ex)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

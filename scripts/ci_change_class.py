#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Classify a change as `light` (documentation plus version-number edits) or heavy.

CI uses the answer to skip the NON-REQUIRED slow legs (Windows and macOS test runs) on a
pull request that cannot change what they test. The required Linux legs always run in full,
so a wrong `light=true` costs one skipped cross-platform run on that PR; `main` runs every
leg on every push, which is the backstop.

A change is light only if EVERY path in `base..head` is one of:
  * documentation: `docs/**`, `*.md`, `LICENSE*`;
  * `Cargo.toml` / `*/Cargo.toml`, when each changed line is a workspace version number: a lone
    `version = "N"` line, or a dependency line carrying `path = "..."`, and the removed and added
    lines are identical once their version numbers are blanked (an external dependency bump has no
    `path =` and is therefore heavy);
  * `Cargo.lock`, when the removed and added lines are identical once every `N.N.N` is blanked (a
    registry crate changes its `checksum`, which does not blank, so any dependency change is heavy).

Anything else, an empty diff, or a git failure is heavy. Usage:
`ci_change_class.py [--base REF] [--head REF]`; prints `light=true|false` and, when
`GITHUB_OUTPUT` is set, appends the same line there. Reasons go to stderr.
"""
import collections
import os
import re
import shutil
import subprocess  # nosec B404 - runs `git` only, with a fixed argument list and no shell
import sys

DOC = re.compile(r"^(docs/.*|.*\.md|LICENSE.*)$")
VERSION = re.compile(r"\d+\.\d+\.\d+")
LONE_VERSION = re.compile(r'^version = "\d+\.\d+\.\d+"$')


def git(*args):
    # Fixed argument list, no shell; the only inputs are ref names and paths from `git diff` itself.
    exe = shutil.which("git")
    if exe is None:
        raise OSError("git not found on PATH")
    return subprocess.run([exe, *args], capture_output=True, text=True, check=True).stdout  # nosec B603


def changed_lines(base, head, path):
    """(removed, added) content lines of one file, +/- markers stripped."""
    out = git("diff", "-U0", "--no-color", base, head, "--", path).splitlines()
    removed, added = [], []
    for line in out:
        if line.startswith(("+++", "---")) or not line or line[0] not in "+-":
            continue
        (removed if line[0] == "-" else added).append(line[1:].rstrip("\r"))
    return removed, added


def blank(line):
    return VERSION.sub("N", line)


def same_but_versions(removed, added):
    return collections.Counter(map(blank, removed)) == collections.Counter(map(blank, added))


def manifest_is_version_only(removed, added):
    ok_line = lambda l: bool(LONE_VERSION.match(l)) or 'path = "' in l
    return all(map(ok_line, removed + added)) and same_but_versions(removed, added)


def heavy_reason(base, head, name):
    """Why `name` makes the change heavy, or None when it is documentation or version-only."""
    if DOC.match(name):
        return None
    base_name = name.rsplit("/", 1)[-1]
    if base_name not in ("Cargo.toml", "Cargo.lock"):
        return f"{name}: not documentation or a version file"
    removed, added = changed_lines(base, head, name)
    if not removed and not added:
        return f"{name}: no content lines (mode or rename change)"
    check = same_but_versions if base_name == "Cargo.lock" else manifest_is_version_only
    if not check(removed, added):
        return f"{name}: changes more than workspace version numbers"
    return None


def classify(base, head):
    """Return (light, reasons)."""
    names = git("diff", "--name-only", base, head).splitlines()
    if not names:
        return False, ["empty diff"]
    for name in names:
        reason = heavy_reason(base, head, name)
        if reason:
            return False, [reason]
    return True, [f"{len(names)} path(s), all documentation or version-only"]


def main(argv):
    base, head = "HEAD^1", "HEAD"
    it = iter(argv)
    for a in it:
        if a == "--base":
            base = next(it)
        elif a == "--head":
            head = next(it)
        else:
            sys.exit(f"unknown argument {a}")
    try:
        light, reasons = classify(base, head)
    except (subprocess.CalledProcessError, OSError) as e:
        light, reasons = False, [f"git failed: {e}"]
    line = f"light={'true' if light else 'false'}"
    print(line)
    print("; ".join(reasons), file=sys.stderr)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail if a workspace dev-dependency on another workspace crate carries a version.

`cargo publish` strips a dev-dependency that has a `path` and NO `version`, but
keeps -- and must resolve on crates.io -- one that has both. A versioned
dev-dependency on a workspace crate therefore (a) forces that crate to be
published first, which breaks any publish order computed from normal/build
dependencies alone ("no matching package named ..."), and (b) is an
unpublishable cycle when the target reaches the package. Neither shows until
publish time, so this check makes the property a CI failure.

Rule: every dev-dependency of a workspace package that names another workspace
member must be path-only (`{ path = "../x" }`), never `workspace = true`.

Usage: python3 scripts/check-dev-dep-publish-cycles.py [path/to/metadata.json]
(without an argument it runs `cargo metadata --no-deps --format-version 1`).
"""
import json
import subprocess
import sys


def load():
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            return json.load(f)
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True, capture_output=True,
    ).stdout
    return json.loads(out)


def main():
    meta = load()
    names = {p["name"] for p in meta["packages"]}
    bad = []
    checked = 0
    for p in sorted(meta["packages"], key=lambda p: p["name"]):
        for d in p["dependencies"]:
            if d["kind"] != "dev" or d["name"] not in names:
                continue
            checked += 1
            if d["req"] != "*":
                bad.append((p["name"], d["name"], d["req"]))
    print(f"workspace packages: {len(names)}; workspace dev-deps checked: {checked}")
    for pkg, dev, req in bad:
        print(f"VERSIONED DEV-DEP: {pkg} [dev-dependencies] {dev} = {req!r}; "
              f"make it path-only (no version, not `workspace = true`)")
    if bad:
        print(f"{len(bad)} versioned workspace dev-dependenc(ies)")
        return 1
    print("no versioned workspace dev-dependencies")
    return 0


if __name__ == "__main__":
    sys.exit(main())

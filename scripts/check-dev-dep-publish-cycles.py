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

Usage: cargo metadata --no-deps --format-version 1 > metadata.json
       python3 scripts/check-dev-dep-publish-cycles.py metadata.json
"""
import json
import sys


def versioned_dev_deps(meta):
    """Return (checked, bad): workspace dev-deps seen, and those carrying a version."""
    names = {p["name"] for p in meta["packages"]}
    checked, bad = 0, []
    for p in sorted(meta["packages"], key=lambda p: p["name"]):
        for d in p["dependencies"]:
            if d["kind"] == "dev" and d["name"] in names:
                checked += 1
                if d["req"] != "*":
                    bad.append((p["name"], d["name"], d["req"]))
    return checked, bad


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], "rb") as f:
        meta = json.load(f)
    checked, bad = versioned_dev_deps(meta)
    print(f"workspace packages: {len(meta['packages'])}; "
          f"workspace dev-deps checked: {checked}")
    for pkg, dev, req in bad:
        print(f"VERSIONED DEV-DEP: {pkg} [dev-dependencies] {dev} = {req!r}; "
              f"make it path-only (no version, not `workspace = true`)")
    print(f"{len(bad)} versioned workspace dev-dependenc(ies)" if bad
          else "no versioned workspace dev-dependencies")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

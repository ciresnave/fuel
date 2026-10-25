#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Copy the root LICENSE-MIT, LICENSE-APACHE and THIRD-PARTY-NOTICES.md into every publishable
workspace crate, so a crates.io tarball carries the licence texts and the third-party notices
(candle's licences require both to travel with derived code).

    python scripts/sync-packaged-notices.py          # write the copies
    python scripts/sync-packaged-notices.py --check  # exit 1 if any copy is missing or differs

Publishable = a workspace member whose Cargo.toml has no `publish = false`. The same rule is
asserted by fuel-ir/tests/spdx_header_ratchet.rs (packaged_licence_copies_are_listed_and_match_the_root_files)
and, on the built tarballs, by `scripts/verify-packaged-crates.py notices`.
"""
import os
import re
import shutil
import sys
import tomllib

FILES = ("LICENSE-MIT", "LICENSE-APACHE", "THIRD-PARTY-NOTICES.md")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def publishable_members():
    with open(os.path.join(ROOT, "Cargo.toml"), "rb") as f:
        members = tomllib.load(f)["workspace"]["members"]
    out = []
    for m in members:
        with open(os.path.join(ROOT, m, "Cargo.toml"), "rb") as f:
            pkg = tomllib.load(f)["package"]
        if pkg.get("publish", True) is not False:
            out.append(m)
    return out


def main(argv):
    check = "--check" in argv
    members = publishable_members()
    bad = []
    for m in members:
        for name in FILES:
            src = os.path.join(ROOT, name)
            dst = os.path.join(ROOT, m, name)
            same = os.path.exists(dst) and open(src, "rb").read() == open(dst, "rb").read()
            if same:
                continue
            bad.append(f"{m}/{name}")
            if not check:
                shutil.copyfile(src, dst)
    print(f"{len(members)} publishable crates x {len(FILES)} files; {'differing/missing' if check else 'written'}: {len(bad)}")
    for b in bad:
        print("  " + b)
    return 1 if (check and bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""GAP-314 shared-op assert census (property-derived, test-excluding).

Finds PRODUCTION `assert!/assert_eq!/assert_ne!/debug_assert*!` macros whose CALL TEXT
references caller-supplied SHAPE (dims/shape/rank/ndim/numel/elem_count/dim indices).
Excludes `#[cfg(test)] mod {...}` blocks by brace-matching. Prints file:line + text so
each hit can be classified (build-time typed-Err belongs vs a real panic-on-caller-shape).

Re-derivation aid for GAP-314 (shared-op enumeration), GAP-326 (inlined per-model copies)
and GAP-327 (realize-time dtype panics — those are NOT asserts, grep 'root dtype is'
separately). Roots are derived from this file's location, so it runs from any checkout.
"""
import re, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOTS = [REPO / "fuel-core" / "src", REPO / "fuel-nn" / "src", REPO / "fuel-transformers" / "src"]
SHAPE = re.compile(r"\.dims\(\)|\.shape\(\)|\.rank\(\)|\bdims\[|dims\.len\(\)|\.ndim|numel|elem_count|\.dim\(|shape\.dims")
ASSERT = re.compile(r"\b(debug_assert|assert)(_eq|_ne)?!\s*\(")


def strip_test_blocks(lines):
    """Return set of line indices INSIDE a `#[cfg(test)] mod ... {` block (brace-matched)."""
    inside = set()
    i, n = 0, len(lines)
    while i < n:
        if "#[cfg(test)]" in lines[i]:
            j = i + 1
            while j < n and lines[j].lstrip().startswith("#["):
                j += 1
            if j < n and re.match(r"\s*(pub\s+)?mod\s+\w+", lines[j]):
                depth, started, k = 0, False, j
                while k < n:
                    depth += lines[k].count("{") - lines[k].count("}")
                    if "{" in lines[k]:
                        started = True
                    if started and depth <= 0:
                        break
                    k += 1
                for x in range(i, min(k, n - 1) + 1):
                    inside.add(x)
                i = k + 1
                continue
        i += 1
    return inside


def macro_call_text(lines, start):
    """Accumulate from an assert opener line until parens balance (cap 12 lines)."""
    depth, parts = 0, []
    for k in range(start, min(start + 12, len(lines))):
        parts.append(lines[k])
        depth += lines[k].count("(") - lines[k].count(")")
        if depth <= 0 and "(" in "".join(parts):
            break
    return " ".join(p.strip() for p in parts)


def main():
    total = 0
    for root in ROOTS:
        for path in sorted(pathlib.Path(root).rglob("*.rs")):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            test_lines = strip_test_blocks(lines)
            for idx, line in enumerate(lines):
                if idx in test_lines or line.lstrip().startswith("//"):
                    continue
                if not ASSERT.search(line):
                    continue
                call = macro_call_text(lines, idx)
                if SHAPE.search(call):
                    total += 1
                    rel = path.relative_to(REPO).as_posix()
                    print(f"{rel}:{idx + 1}: {re.sub(r'\\s+', ' ', call)[:180]}")
    print(f"\n=== TOTAL production shape-asserts: {total} ===", file=sys.stderr)


if __name__ == "__main__":
    main()

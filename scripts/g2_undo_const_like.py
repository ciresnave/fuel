#!/usr/bin/env python3
"""
Undo the const_*_like device-arg insertion the main g2_sweep added.
We later decided that const_*_like derives device from self's graph
and doesn't need an explicit device parameter — only from_* (root
constructors that need to choose where to allocate fresh) take one.

Removes the trailing `, <DEV>` (or trailing-comma multi-line variant)
from .const_f32_like / .const_f64_like / .const_bf16_like /
.const_f16_like / .const_u32_like / .const_like calls.

Usage: python g2_undo_const_like.py <file> <DEV>
"""

import re
import sys

METHOD_FNS = {
    'const_f32_like', 'const_f64_like', 'const_bf16_like', 'const_f16_like',
    'const_u32_like', 'const_like',
}

METHOD_PATTERN = re.compile(
    r'\.(?P<fn>' + '|'.join(METHOD_FNS) + r')\s*\('
)


def find_balanced_close(src: str, open_paren_idx: int) -> int:
    depth = 1
    i = open_paren_idx + 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"':
            i += 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            j = i + 1
            if j < n and src[j] == '\\':
                k = j + 1
                while k < n and src[k] != "'":
                    k += 1
                i = k + 1
                continue
            elif j < n and src[j].isalpha() and (j + 1 >= n or src[j + 1] != "'"):
                k = j + 1
                while k < n and (src[k].isalnum() or src[k] == '_'):
                    k += 1
                i = k
                continue
            elif j + 1 < n and src[j + 1] == "'":
                i = j + 2
                continue
            else:
                i += 1
                continue
        if c == '/' and i + 1 < n:
            if src[i + 1] == '/':
                while i < n and src[i] != '\n':
                    i += 1
                continue
            if src[i + 1] == '*':
                i += 2
                while i + 1 < n and not (src[i] == '*' and src[i + 1] == '/'):
                    i += 1
                i += 2
                continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced parens starting at {open_paren_idx}")


def undo_inserts(src: str, dev_expr: str) -> tuple[str, int]:
    matches = []
    for m in METHOD_PATTERN.finditer(src):
        matches.append(m.end() - 1)
    matches.sort()

    out = src
    count = 0
    for open_idx in reversed(matches):
        try:
            close_idx = find_balanced_close(out, open_idx)
        except ValueError as e:
            print(f"WARN: skipping at {open_idx}: {e}", file=sys.stderr)
            continue
        # Look at the args between ( and ) and check whether the last
        # arg is the dev_expr (with optional trailing comma).
        inside = out[open_idx + 1:close_idx]
        # Strip trailing whitespace.
        inside_rstrip = inside.rstrip()
        had_trailing_comma = False
        if inside_rstrip.endswith(','):
            inside_rstrip = inside_rstrip[:-1].rstrip()
            had_trailing_comma = True
        if not inside_rstrip.endswith(dev_expr):
            continue
        # Find where dev_expr starts.
        dev_start_in_stripped = len(inside_rstrip) - len(dev_expr)
        # Walk backward from there to find the `,` that introduced this arg.
        # We want to remove everything from that comma onward (including
        # any whitespace before it).
        rel_close = close_idx  # absolute
        # The end of dev_expr in absolute terms:
        # inside_rstrip ends at some absolute index ≤ close_idx; let's
        # locate it by re-scanning the actual string.
        # Walk from close_idx-1 backward, skipping ws and trailing commas.
        end = close_idx - 1
        while end > open_idx and out[end] in ' \t\n':
            end -= 1
        if had_trailing_comma:
            assert out[end] == ','
            end -= 1
            while end > open_idx and out[end] in ' \t\n':
                end -= 1
        # Now end is the last char of dev_expr.
        dev_end_excl = end + 1
        dev_start = dev_end_excl - len(dev_expr)
        if out[dev_start:dev_end_excl] != dev_expr:
            # mismatch — bail to be safe
            continue
        # Find the `,` (or `(`) that precedes dev_expr. We want to keep
        # the args up to and including the previous arg's closing chars,
        # so look for a comma.
        cut_start = dev_start - 1
        while cut_start > open_idx and out[cut_start] in ' \t\n':
            cut_start -= 1
        if out[cut_start] == ',':
            # Remove from this comma to the end of dev_expr (and any
            # trailing whitespace+newline up to ')').
            removal_start = cut_start
            removal_end = close_idx  # don't remove ')'
            # If there's whitespace+newline between dev_end_excl and
            # close_idx, strip it too — unless that would collapse a
            # multi-line call awkwardly. Keep it simple: strip trailing
            # whitespace including newlines.
            out = out[:removal_start] + out[removal_end:]
            count += 1
        elif out[cut_start] == '(':
            # dev_expr was the only arg — leave the parens as-is and
            # just remove the dev expression.
            out = out[:dev_start] + out[close_idx:]
            count += 1
    return out, count


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    dev_expr = sys.argv[2]
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    new, n = undo_inserts(src, dev_expr)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new)
    print(f"{path}: {n} const_*_like callsites reverted", file=sys.stderr)


if __name__ == '__main__':
    main()

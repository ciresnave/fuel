#!/usr/bin/env python3
"""Phase 7.5 B3 step 1 migration helper.

Adds `?` propagation to call sites of the storage seam after they
become Result-returning. Pure mechanical sed-like replacement on
specific call patterns. Skip lines that already have `?` after the
call.
"""
import re
import sys
from pathlib import Path

# The patterns we migrate. Order matters: most specific first so an
# earlier pattern doesn't eat a substring that the next one needs.
SUBS = [
    # Method calls followed by another method call.
    (r'\.realized_storage\(\)(\.[a-zA-Z_])', r'.realized_storage()?\1'),
    (r'\.storage_and_layout\(\)(\.[a-zA-Z_])', r'.storage_and_layout()?\1'),
    (r'\.storage_mut_and_layout\(\)(\.[a-zA-Z_])', r'.storage_mut_and_layout()?\1'),
    (r'\.same_storage\((.*?)\)(\.[a-zA-Z_])', r'.same_storage(\1)?\2'),
    (r'\.storage\(\)\.read\(\)', r'.storage()?.read()'),
    (r'\.storage\(\)\.write\(\)', r'.storage()?.write()'),
    (r'\.storage_mut\(\)\.read\(\)', r'.storage_mut()?.read()'),
    (r'\.storage_mut\(\)\.write\(\)', r'.storage_mut()?.write()'),

    # Statement-terminating calls (followed by `;` or `,` or `)`).
    (r'\.realized_storage\(\)([;,)\s])', r'.realized_storage()?\1'),
    (r'\.storage\(\)([;,)])', r'.storage()?\1'),
    (r'\.storage_mut\(\)([;,)])', r'.storage_mut()?\1'),
    (r'\.storage_and_layout\(\)([;,)])', r'.storage_and_layout()?\1'),
    (r'\.storage_mut_and_layout\(\)([;,)])', r'.storage_mut_and_layout()?\1'),
    # same_storage in `if foo.same_storage(&bar) { ... }`
    (r'\.same_storage\((.*?)\)(\s*\{)', r'.same_storage(\1)?\2'),
    (r'\.same_storage\((.*?)\)([;,)])', r'.same_storage(\1)?\2'),
]

# Lines that we MUST NOT rewrite — the seam definitions themselves
# already return Result. They sit in tensor.rs lines 4540-4625 area.
# Safer: skip lines that contain the function signature pattern.
SKIP_PATTERNS = [
    re.compile(r'pub(\(crate\))?\s+fn\s+(realized_storage|storage|storage_mut|storage_and_layout|storage_mut_and_layout|same_storage)\b'),
    re.compile(r'fn\s+(realized_storage|storage|storage_mut|storage_and_layout|storage_mut_and_layout|same_storage)\b'),
]


def already_propagated(text: str, pos: int) -> bool:
    """Cheap check: is the next non-whitespace char after position `pos` a `?`?"""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos < len(text) and text[pos] == '?'


def migrate_text(text: str) -> tuple[str, int]:
    out_lines = []
    changes = 0
    for line in text.splitlines(keepends=True):
        if any(p.search(line) for p in SKIP_PATTERNS):
            out_lines.append(line)
            continue
        new_line = line
        for pat, rep in SUBS:
            # Only replace where there's no existing `?` immediately after.
            def smart_sub(m):
                # Recover the position of `)` we just matched.
                # If text right after the )-match starts with `?`, skip.
                end_of_call = m.end(0) - len(m.group(1)) if m.lastindex else m.end(0)
                if m.lastindex and m.group(1).startswith('?'):
                    return m.group(0)
                # Also detect existing `?` already in the line right after the call.
                # The simplest signal: if `?` is the char immediately following the
                # closing `)` of the call we matched. The replacement `(...)\?\1`
                # naturally fails to match again because the char between `)` and
                # the captured group is now `?`, not the captured group's first char.
                return re.sub(pat, rep, m.group(0))
            try:
                # Direct re.sub is fine because the SUBS already encode the
                # "followed by something specific" requirement.
                new_line2 = re.sub(pat, rep, new_line)
            except Exception as e:
                print(f"regex error: {e} on line {line!r}", file=sys.stderr)
                new_line2 = new_line
            if new_line2 != new_line:
                changes += new_line2.count('?') - new_line.count('?')
                new_line = new_line2
        out_lines.append(new_line)
    return ''.join(out_lines), changes


def main(paths):
    total = 0
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        text = p.read_text(encoding='utf-8')
        new_text, changes = migrate_text(text)
        if new_text != text:
            p.write_text(new_text, encoding='utf-8')
            print(f"{path}: {changes} additions")
            total += changes
    print(f"total: {total} additions")


if __name__ == '__main__':
    main(sys.argv[1:])

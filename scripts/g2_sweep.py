#!/usr/bin/env python3
"""
Phase 7.5 G2 step 2 sweep: append a device argument to every call site
of fuel-graph's slot-populating constructors.

Mechanically rewrites:
  Tensor::from_f32(data, shape)            -> Tensor::from_f32(data, shape, <DEV>)
  Tensor::from_f64(...)                    -> ... (same)
  Tensor::from_bf16(...)                   -> ... (same)
  Tensor::from_f16(...)                    -> ... (same)
  Tensor::from_u32(...)                    -> ... (same)
  Tensor::from_const(data, shape)          -> Tensor::from_const(data, shape, <DEV>)
  receiver.const_f32_like(data, shape)     -> receiver.const_f32_like(data, shape, <DEV>)
  receiver.const_f64_like(...)             -> ... (same)
  receiver.const_bf16_like(...)            -> ... (same)
  receiver.const_f16_like(...)             -> ... (same)
  receiver.const_u32_like(...)             -> ... (same)
  receiver.const_like(data, shape)         -> receiver.const_like(data, shape, <DEV>)

Walks balanced parens so nested tuples like (2, 3) are handled correctly.
Also handles multi-line function calls.

Usage:
    python g2_sweep.py <file> <DEV>

Where <DEV> is the device-argument expression to append, e.g.:
    "&cpu_dev"
    "device.as_dyn()"
    "&Device::cpu().as_dyn()"
"""

import re
import sys

# Constructors invoked as ::Tensor::name (or just Tensor::name) — bare-function form.
TYPE_FNS = {
    'from_f32', 'from_f64', 'from_bf16', 'from_f16', 'from_u32', 'from_const',
}

# Method-form receivers — need a leading `.`.
# G2 design refinement: const_*_like methods derive the device from
# self's graph internally, so they DON'T take a device parameter.
# Only the root constructors (from_*) need explicit device threading.
METHOD_FNS: set[str] = set()

# Build a regex that finds the opening of any matching call.
TYPE_PATTERN = re.compile(
    r'\b(?:Tensor|fuel_graph::Tensor|LazyTensor)::(?P<fn>' +
    '|'.join(TYPE_FNS) + r')\s*\('
)
METHOD_PATTERN = (
    re.compile(r'\.(?P<fn>' + '|'.join(METHOD_FNS) + r')\s*\(')
    if METHOD_FNS else None
)


def find_balanced_close(src: str, open_paren_idx: int) -> int:
    """Given the index of an opening '(' in src, return the index of its
    matching ')'. Tracks string and char literals so we don't count parens
    inside them. Comments outside strings are ignored too."""
    depth = 1
    i = open_paren_idx + 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == '"':
            # Skip string literal.
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
            # Skip char literal. Could be label, lifetime, or char literal —
            # all of these are short and end at non-ident char.
            # If `'a` (lifetime/label), no closing quote — bail after one
            # ident char. If `'x'` or `'\n'`, find closing quote.
            j = i + 1
            if j < n and src[j] == '\\':
                # `'\n'`, `'\\'`, `'\x41'`, etc. Find closing '.
                k = j + 1
                while k < n and src[k] != "'":
                    k += 1
                i = k + 1
                continue
            elif j < n and src[j].isalpha() and (j + 1 >= n or src[j + 1] != "'"):
                # Lifetime / label like `'a`.
                k = j + 1
                while k < n and (src[k].isalnum() or src[k] == '_'):
                    k += 1
                i = k
                continue
            elif j + 1 < n and src[j + 1] == "'":
                # `'x'`
                i = j + 2
                continue
            else:
                # Not a recognized form — skip the quote and keep going.
                i += 1
                continue
        if c == '/' and i + 1 < n:
            if src[i + 1] == '/':
                # Line comment — skip to newline.
                while i < n and src[i] != '\n':
                    i += 1
                continue
            if src[i + 1] == '*':
                # Block comment — skip to */.
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


def insert_device_arg(src: str, dev_expr: str) -> tuple[str, int]:
    """Walk `src` and insert `, dev_expr` before the closing paren of every
    matching constructor call. Returns the modified text + count of
    insertions."""
    # Collect all match starts, including the ( position of each.
    matches = []
    for m in TYPE_PATTERN.finditer(src):
        matches.append((m.end() - 1, m.group('fn')))  # m.end()-1 is index of '('
    if METHOD_PATTERN is not None:
        for m in METHOD_PATTERN.finditer(src):
            matches.append((m.end() - 1, m.group('fn')))
    matches.sort()

    # Walk in reverse so insertions don't perturb earlier indices.
    out = src
    count = 0
    for open_idx, fn in reversed(matches):
        try:
            close_idx = find_balanced_close(out, open_idx)
        except ValueError as e:
            print(f"WARN: skipping {fn} at {open_idx}: {e}", file=sys.stderr)
            continue
        # Skip if the call already has the dev arg (idempotent re-runs).
        # Heuristic: look at the last 80 chars before close_idx for dev_expr.
        window_start = max(open_idx, close_idx - 80)
        if dev_expr in out[window_start:close_idx]:
            continue
        # Determine if we need ", " prefix or just "" (zero-arg call).
        between = out[open_idx + 1:close_idx].strip()
        if between == "":
            insertion = dev_expr
        else:
            # Preserve existing whitespace style at insertion point.
            # If the call is multi-line and ends in ",\n        )" already,
            # we want to add the dev on a new line. Otherwise inline.
            # Simple heuristic: look at the char immediately before `)`.
            pre_close = out[close_idx - 1]
            if pre_close == '\n' or pre_close == ' ':
                # Multi-line with whitespace before ): place arg on its own line.
                # Find the indentation of the line containing close_idx.
                line_start = out.rfind('\n', 0, close_idx) + 1
                indent = ''
                k = line_start
                while k < close_idx and out[k] in ' \t':
                    indent += out[k]
                    k += 1
                # Trailing comma?
                # Look back from close_idx, skipping whitespace, for last non-WS char.
                j = close_idx - 1
                while j > open_idx and out[j] in ' \t\n':
                    j -= 1
                if out[j] == ',':
                    insertion = f'\n{indent}    {dev_expr},'
                    # Insert before the trailing whitespace+newline+indent.
                    # We'll insert right after the trailing comma.
                    out = out[:j + 1] + f'\n{indent}    {dev_expr},' + out[j + 1:close_idx] + out[close_idx:]
                    count += 1
                    continue
                else:
                    insertion = f',\n{indent}    {dev_expr}'
            else:
                insertion = f', {dev_expr}'
        out = out[:close_idx] + insertion + out[close_idx:]
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
    new, n = insert_device_arg(src, dev_expr)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new)
    print(f"{path}: {n} call sites updated", file=sys.stderr)


if __name__ == '__main__':
    main()

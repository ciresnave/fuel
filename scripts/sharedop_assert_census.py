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

GAP-326 modes (see the block near the end):
  --gap326-derive <ref> [--write]   record the functions that held an inlined
                                    rank/channel guard at <ref>, from git blobs
  --gap326-gate                     PROPERTY + MEMBERSHIP check against that record
  --gap326-self-test                constructed fixtures for both checks
"""
import re, subprocess, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOTS = [REPO / "fuel-core" / "src", REPO / "fuel-nn" / "src", REPO / "fuel-transformers" / "src"]
SHAPE = re.compile(r"\.dims\(\)|\.shape\(\)|\.rank\(\)|\bdims\[|dims\.len\(\)|\.ndim|numel|elem_count|\.dim\(|shape\.dims")
ASSERT = re.compile(r"\b(debug_assert|assert)(_eq|_ne)?!\s*\(")


def _test_mod_open_at(lines, i):
    """If lines[i] is `#[cfg(test)]` (possibly followed by more attrs) heading a
    `mod ... {`, return the index of that `mod` line; else None."""
    if "#[cfg(test)]" not in lines[i]:
        return None
    j = i + 1
    while j < len(lines) and lines[j].lstrip().startswith("#["):
        j += 1
    if j < len(lines) and re.match(r"\s*(pub\s+)?mod\s+\w+", lines[j]):
        return j
    return None


def _brace_block_end(lines, start):
    """Index of the line at which the brace block opening at/after `start` closes
    (depth returns to 0); the last line if it never closes."""
    depth, started = 0, False
    for k in range(start, len(lines)):
        depth += lines[k].count("{") - lines[k].count("}")
        if "{" in lines[k]:
            started = True
        if started and depth <= 0:
            return k
    return len(lines) - 1


def strip_test_blocks(lines):
    """Return set of line indices INSIDE a `#[cfg(test)] mod ... {` block (brace-matched)."""
    inside = set()
    i = 0
    while i < len(lines):
        j = _test_mod_open_at(lines, i)
        if j is None:
            i += 1
            continue
        k = _brace_block_end(lines, j)
        inside.update(range(i, k + 1))
        i = k + 1
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


# ---------------------------------------------------------------------------
# GAP-326: the validated-entry gate.
#
# Two checks, because either alone is satisfied by the wrong fix:
#   PROPERTY   -- no inlined rank/channel guard of the three GAP-326 classes
#                 remains in fuel-transformers production code. Deleting a
#                 guard satisfies this, so it cannot stand alone.
#   MEMBERSHIP -- every function that held such a guard at the baseline ref
#                 still exists and calls the validated-entry op for its class.
#                 A guard "fixed" by deleting it fails here.
#
# The membership list is DERIVED from git blobs at the baseline ref, never
# typed by hand: `--gap326-derive <ref> --write`.
# ---------------------------------------------------------------------------

GAP326_DIR = "fuel-transformers/src/models"
GAP326_LIST = REPO / "scripts" / "gap326_guarded_functions.txt"

# Non-debug `assert_eq!` only. `\b` does not match inside `debug_assert_eq!`,
# which keeps out lazy_z_image's `debug_assert_eq!(axes_dims.len(), 3)`: a
# config slice in a function that returns no Result (ruled OUT of GAP-326).
GAP326_CLASSES = {
    "rank4": re.compile(r"\bassert_eq!\(\s*\w*dims\.len\(\),\s*4\s*[,)]"),
    "rank3": re.compile(r"\bassert_eq!\(\s*\w*dims\.len\(\),\s*3\s*[,)]"),
    "chan3": re.compile(r"\bassert_eq!\(\s*dims\[1\],\s*3\s*[,)]"),
}
# The op each class must be routed through. A channel guard needs the helper,
# because `dims4()` alone would silently drop the channel check.
GAP326_REQUIRED = {
    "rank4": ("image_nchw(", ".dims4()"),
    "rank3": (".dims3()",),
    "chan3": ("image_nchw(",),
}
GAP326_NOT_COVERED = (
    "NOT covered by this gate, deliberately (other rows own them): batch == 1 guards, "
    "fixed spatial sizes (hiera 224x224), vgg square and /32, codebook and channel "
    "counts, paddleocr tiles (GAP-328), config-vs-config agreement (GAP-314), and "
    "lazy_z_image's config-slice debug_assert (GAP-315)."
)

FN_RE = re.compile(r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)")


def gap326_sites(text):
    """[(line_index, class)] for production guards of the three classes."""
    lines = text.splitlines()
    test_lines = strip_test_blocks(lines)
    out = []
    for idx, line in enumerate(lines):
        if idx in test_lines or line.lstrip().startswith("//"):
            continue
        if not ASSERT.search(line):
            continue
        call = macro_call_text(lines, idx)
        for cls, pat in GAP326_CLASSES.items():
            if pat.search(call):
                out.append((idx, cls))
    return out


def _impl_header(lines, start):
    """The whitespace-collapsed `impl ...` header whose first line is `start`."""
    parts = []
    for k in range(start, min(start + 8, len(lines))):
        parts.append(lines[k].split("{", 1)[0])
        if "{" in lines[k]:
            break
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def enclosing_fn(lines, idx):
    """(impl header or '-', fn name) of the function containing line `idx`."""
    for k in range(idx, -1, -1):
        m = FN_RE.match(lines[k])
        if not m:
            continue
        if not m.group(1):
            return "-", m.group(2)
        for j in range(k, -1, -1):
            if lines[j].startswith("impl"):
                return _impl_header(lines, j), m.group(2)
        return "-", m.group(2)
    return "-", "<none>"


def _skip_literal(line, i):
    """Index of the last character of a string or char literal starting at `i`,
    or `i` itself when `line[i]` does not open one (a lifetime, say)."""
    n = len(line)
    if line[i] == '"':
        i += 1
        while i < n and line[i] != '"':
            i += 2 if line[i] == "\\" else 1
        return i
    if line[i] == "'" and i + 2 < n:
        if line[i + 1] == "\\":
            end = line.find("'", i + 2)
            return end if end != -1 else i
        if line[i + 2] == "'":
            return i + 2
    return i


def _code_braces(line):
    """Net `{` minus `}` outside string/char literals and `//` comments."""
    net, i = 0, 0
    while i < len(line) and not line.startswith("//", i):
        j = _skip_literal(line, i)
        if j == i:
            net += {"{": 1, "}": -1}.get(line[i], 0)
        i = j + 1
    return net


def _block(lines, start):
    """Lines from `start` through the line where its brace block closes."""
    depth, opened = 0, False
    for k in range(start, len(lines)):
        d = _code_braces(lines[k])
        depth += d
        opened = opened or d > 0
        if opened and depth <= 0:
            return lines[start:k + 1]
    return lines[start:]


def _scopes(lines, impl):
    """The line lists to search: the whole file for a free fn, else each
    column-0 `impl` block whose header matches."""
    if impl == "-":
        return [lines]
    return [_block(lines, j) for j, line in enumerate(lines)
            if line.startswith("impl") and _impl_header(lines, j) == impl]


def fn_body(text, impl, name):
    """The text of `fn name` inside `impl` (or at top level), or None; raises if ambiguous."""
    top_level = impl == "-"
    hits = []
    for scope in _scopes(text.splitlines(), impl):
        for k, line in enumerate(scope):
            m = FN_RE.match(line)
            if m and m.group(2) == name and not (top_level and m.group(1)):
                hits.append("\n".join(_block(scope, k)))
    if len(hits) > 1:
        raise ValueError(f"{impl} :: {name} is ambiguous ({len(hits)} definitions)")
    return hits[0] if hits else None


def _git(*args, stdin=None):
    return subprocess.run(["git", *args], cwd=REPO, input=stdin, capture_output=True, check=True).stdout


def blobs_at(ref, directory):
    """{path: text} for every .rs blob under `directory` at `ref`."""
    names = _git("ls-tree", "-r", "--name-only", ref, "--", directory).decode().split()
    out = {}
    for name in names:
        if name.endswith(".rs"):
            out[name] = _git("show", f"{ref}:{name}").decode("utf-8", errors="replace")
    return out


def index_files(directory):
    """{path: working-tree text} for every .rs file the INDEX lists under `directory`."""
    names = _git("ls-files", "-z", "--", directory).decode().split("\0")
    return {
        n: (REPO / n).read_text(encoding="utf-8", errors="replace")
        for n in names if n.endswith(".rs")
    }


def gap326_membership(files):
    """{(path, impl, fn): sorted classes} for the guarded functions in `files`."""
    found = {}
    for path, text in files.items():
        lines = text.splitlines()
        for idx, cls in gap326_sites(text):
            key = (path,) + enclosing_fn(lines, idx)
            found.setdefault(key, set()).add(cls)
    return {k: sorted(v) for k, v in found.items()}


def read_list():
    entries = {}
    for line in GAP326_LIST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        path, impl, name, classes = line.split("\t")
        entries[(path, impl, name)] = classes.split(",")
    return entries


def property_problems(files):
    """Every inlined guard of the three classes still present."""
    return [f"{path}:{idx + 1}: inlined {cls} guard"
            for path, text in sorted(files.items())
            for idx, cls in gap326_sites(text)]


def _entry_problems(files, path, impl, name, classes):
    """Why one recorded function fails MEMBERSHIP (empty if it passes)."""
    text = files.get(path)
    if text is None:
        return [f"{path}: file is gone; re-point {impl} :: {name} with a reason"]
    try:
        body = fn_body(text, impl, name)
    except ValueError as e:
        return [f"{path}: {e}"]
    if body is None:
        return [f"{path}: {impl} :: {name} is gone; re-point it with a reason"]
    return [f"{path}: {impl} :: {name} held a {cls} guard and calls none of "
            f"{' / '.join(GAP326_REQUIRED[cls])}; a deleted guard is not a fix"
            for cls in classes
            if not any(op in body for op in GAP326_REQUIRED[cls])]


def gap326_check(files, entries):
    """(property problems, membership problems) for `files` against `entries`."""
    memb = []
    for (path, impl, name), classes in sorted(entries.items()):
        memb += _entry_problems(files, path, impl, name, classes)
    return property_problems(files), memb


def gap326_gate():
    files = index_files(GAP326_DIR)
    entries = read_list()
    prop, memb = gap326_check(files, entries)
    print(f"GAP-326 gate: {len(entries)} recorded functions, {len(files)} files")
    for p in prop:
        print(f"  PROPERTY   {p}")
    for m in memb:
        print(f"  MEMBERSHIP {m}")
    if prop or memb:
        print(f"GAP-326 gate FAILED: {len(prop)} property, {len(memb)} membership problem(s).")
        print(GAP326_NOT_COVERED)
        return 1
    print("GAP-326 gate OK: no inlined rank/channel guards; every recorded function is routed.")
    print(GAP326_NOT_COVERED)
    return 0


def all_locatable(files, entries, ref):
    """Every recorded function must be LOCATABLE at the ref it was derived
    from, or the gate would later report it "gone" for a reason that is the
    locator's, not the code's."""
    _, memb = gap326_check(files, entries)
    lost = [m for m in memb if "is gone" in m or "ambiguous" in m]
    for m in lost:
        print(f"  NOT LOCATABLE AT {ref}: {m}", file=sys.stderr)
    return not lost


def gap326_derive(ref, write):
    files = blobs_at(ref, GAP326_DIR)
    entries = gap326_membership(files)
    sites = sum(len(gap326_sites(t)) for t in files.values())
    if not all_locatable(files, entries, ref):
        return 1
    rows = [f"{p}\t{i}\t{n}\t{','.join(c)}" for (p, i, n), c in sorted(entries.items())]
    header = [
        f"# GAP-326 membership: functions that held an inlined rank/channel guard at {ref}.",
        f"# {sites} guard sites in {len(rows)} functions. DERIVED, do not edit by hand:",
        f"#   python scripts/sharedop_assert_census.py --gap326-derive {ref} --write",
        "# A recorded function that moves or is renamed fails the gate until it is",
        "# re-pointed here, with the reason in the commit.",
        "# path<TAB>impl header (- for a free fn)<TAB>fn<TAB>classes",
    ]
    text = "\n".join(header + rows) + "\n"
    if write:
        GAP326_LIST.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {GAP326_LIST.relative_to(REPO).as_posix()}: {sites} sites in {len(rows)} functions")
    else:
        sys.stdout.write(text)
    return 0


def gap326_self_test():
    guarded = (
        "impl Foo {\n"
        "    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {\n"
        "        let dims = image.shape().dims().to_vec();\n"
        "        assert_eq!(dims.len(), 4, \"image must be rank 4\");\n"
        "        assert_eq!(dims[1], 3, \"3 channels\");\n"
        "        let s = format!(\"{x}\");\n"
        "        Ok(image.clone())\n"
        "    }\n"
        "}\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    fn t() { assert_eq!(dims.len(), 4); }\n"
        "}\n"
    )
    routed = guarded.replace(
        "        assert_eq!(dims.len(), 4, \"image must be rank 4\");\n"
        "        assert_eq!(dims[1], 3, \"3 channels\");\n",
        "        let (_, _, h, w) = image_nchw(image, 3, \"Foo::forward\")?;\n")
    deleted = guarded.replace(
        "        assert_eq!(dims.len(), 4, \"image must be rank 4\");\n"
        "        assert_eq!(dims[1], 3, \"3 channels\");\n", "")
    rank_only = guarded.replace(
        "        assert_eq!(dims.len(), 4, \"image must be rank 4\");\n"
        "        assert_eq!(dims[1], 3, \"3 channels\");\n",
        "        let (_, _, h, w) = image.shape().dims4()?;\n")
    debug_only = guarded.replace("assert_eq!(dims.len(), 4, \"image must be rank 4\");",
                                 "debug_assert_eq!(axes_dims.len(), 3);").replace(
        "        assert_eq!(dims[1], 3, \"3 channels\");\n", "")
    renamed = routed.replace("pub fn forward(", "pub fn forward_image(")
    path = "fuel-transformers/src/models/foo.rs"
    entries = gap326_membership({path: guarded})
    key = (path, "impl Foo", "forward")
    arms = {
        "derive finds one fn, both classes (test mod excluded)": entries == {key: ["chan3", "rank4"]},
        "a routed fn passes both checks": gap326_check({path: routed}, entries) == ([], []),
        "an inlined guard fails PROPERTY": bool(gap326_check({path: guarded}, entries)[0]),
        "a deleted guard passes PROPERTY but fails MEMBERSHIP": (
            not gap326_check({path: deleted}, entries)[0]
            and bool(gap326_check({path: deleted}, entries)[1])),
        "dims4 without the channel helper fails MEMBERSHIP": bool(gap326_check({path: rank_only}, entries)[1]),
        "a renamed fn fails MEMBERSHIP": bool(gap326_check({path: renamed}, entries)[1]),
        "a deleted file fails MEMBERSHIP": bool(gap326_check({}, entries)[1]),
        "a debug_assert on a config slice is not a site": gap326_sites(debug_only) == [],
    }
    for name, ok in arms.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(arms.values()) else 1


if __name__ == "__main__":
    ARGS = sys.argv[1:]
    if "--gap326-self-test" in ARGS:
        sys.exit(gap326_self_test())
    if "--gap326-gate" in ARGS:
        sys.exit(gap326_gate())
    if "--gap326-derive" in ARGS:
        sys.exit(gap326_derive(ARGS[ARGS.index("--gap326-derive") + 1], "--write" in ARGS))
    main()

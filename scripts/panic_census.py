#!/usr/bin/env python3
"""Production-panic census: assert/panic-family macros in fuel-transformers/src/models/.

WHY THIS EXISTS
    GAP-281 surveyed ONE relation (`num_attention_heads * head_dim == hidden_size`)
    and EXPLICITLY declared the rest of the crate's production panics out of scope:
    "Other production `assert!`s exist in these files ... and are NOT surveyed here.
    The count 8 is over ONE relation, not over the crate." This is the instrument
    that surveys the population GAP-281 set aside — every production panic-family
    macro, per SITE, excluding the head_dim relation (which is its own row).

    It lives in the repo ON PURPOSE. The GAP-302 doc-name census was measured with a
    tool that did NOT live in the repository, so nobody but its author could re-run it
    and it could not be a gate's foundation. That absence was the finding. A partition
    whose derivation lives in a scratchpad is a NUMBER; a committed script makes the
    row's headline a RE-DERIVATION.

WHAT IT IS AND IS NOT
    It is a per-SITE PRINTER. It is NOT a count-asserting born-red: the number 475
    (measured at origin/main 1b8d3b1e) is a snapshot that rots as the zoo grows, and
    a gate pinned to it would enforce a stale count. The DEFAULT prints the partition
    and exits 0.

    The one invariant worth gating is not the total but that GAP-281 STAYS CONVERTED:
    the head_dim relation is 0 production ASSERTS at 1b8d3b1e (they became typed
    declines). `--gate` exits non-zero if any head_dim-relation ASSERT reappears — a
    never-panic regression on the relation GAP-281 closed.

CENSUS RULES (stated so the census is auditable)
    - population: the macros assert! / assert_eq! / assert_ne! / debug_assert(_eq|_ne)!
      / panic! / unreachable! / todo! / unimplemented! in *.rs under
      fuel-transformers/src/models/
    - production only: `#[cfg(test)]` blocks are brace-matched and masked. This
      over-counts a panic in a non-cfg-gated helper that is only called from tests
      (safe direction) and cannot see a custom or aliased assert macro (under-count).
    - PER SITE, never per file: GAP-281's partition census stopped at the first match
      per file and undercounted twice in one day. Every macro invocation is a site.
    - the argument parser is STRING-LITERAL AWARE: a naive `;`-terminated or
      paren-counting scan splits on a `;` or unbalances on a `(` inside a message
      string. `--self-test` proves the parser survives both.
    - EXCLUDES the GAP-281 head_dim relation (`heads * head_dim` vs a width) — that is
      GAP-281's territory and is reported separately as HEADDIM_RELATION (0 = converted).

STRUCTURAL PARTITION (what each site's SHAPE is; the DISPOSITION — Result vs .expect —
    is a per-site read and is the CONVERSION, not this census):
      TENSOR_SHAPE_LEN   a tensor/buffer .len()/.dims()/.shape() check
      SCALAR_OTHER       a scalar guard (seq>0, !is_empty, batch==1, ordering)
      CONFIG_FIELD       an assert touching cfg./config.
      PRODUCT_NONTENSOR  a product compared to a non-tensor bound
      UNCONDITIONAL      panic!/unreachable!/todo!/unimplemented!
      HEADDIM_RELATION   the GAP-281 relation (excluded from the crate census)

USAGE
    python scripts/panic_census.py               # print the partition, exit 0
    python scripts/panic_census.py --self-test    # two-arm proof the instrument works
    python scripts/panic_census.py --gate         # exit 1 if a head_dim assert reappears
"""

import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "fuel-transformers", "src", "models")

MACRO = re.compile(
    r"\b(assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne"
    r"|panic|unreachable|todo|unimplemented)\s*!\s*\("
)
WIDTH = re.compile(r"\b(hidden_size|embed_dim|n_embd|d_model|model_dim)\b")
HEADCOUNT = re.compile(r"\b(num_attention_heads|num_heads|n_heads|n_head|nh)\b")
HEADDIM = re.compile(r"\b(head_dim|hpa|hd)\b")
TENSORISH = re.compile(
    r"\.len\(\)|\.dims\(\)|\.dim\(|\.shape|\.rank\(|dims\[|\.elem_count\(|\.numel\("
    r"|\.rows\(|\.cols\("
)
CONFIGISH = re.compile(r"\bcfg\.|config\.|\.config\b")


def cfg_test_spans(src):
    """Byte spans of every brace-matched `#[cfg(test)] { ... }` block."""
    spans = []
    for m in re.finditer(r"#\[cfg\(test\)\]", src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        spans.append((m.start(), j + 1))
    return spans


def in_test(pos, spans):
    return any(a <= pos < b for a, b in spans)


def _skip_string_literal(src, j):
    """`j` points at the opening '"'. Return the index just past the closing '"'."""
    n = len(src)
    j += 1
    while j < n:
        if src[j] == "\\":
            j += 2
            continue
        if src[j] == '"':
            return j + 1
        j += 1
    return j


def _skip_char_literal(src, j):
    """`j` points at a "'". Return the index past a char literal, or `j+1` for a
    lifetime tick (consumed as an ordinary char, matching the original scan)."""
    n = len(src)
    if j + 1 < n and src[j + 1] == "\\":
        k = j + 2
        while k < n and src[k] != "'":
            k += 1
        return k + 1
    if j + 2 < n and src[j + 2] == "'":
        return j + 3
    return j + 1


def parse_args(src, open_idx):
    """Args of a macro call, string-literal aware. `open_idx` points at the '('.

    A naive scan that stops at the first ';' or counts parens is defeated by a ';'
    or a '(' inside a message string; the string/char skips keep those literals from
    splitting an argument or unbalancing the parens. Args are returned as source
    slices between top-level commas, so a literal's bytes ride inside the slice.
    """
    n = len(src)
    j = open_idx + 1
    depth = 1
    start = j
    args = []
    while j < n:
        c = src[j]
        if c == '"':
            j = _skip_string_literal(src, j)
        elif c == "'":
            j = _skip_char_literal(src, j)
        elif c in "([{":
            depth += 1
            j += 1
        elif c in ")]}":
            if c == ")" and depth == 1:
                args.append(src[start:j])
                break
            depth -= 1
            j += 1
        elif c == "," and depth == 1:
            args.append(src[start:j])
            start = j + 1
            j += 1
        else:
            j += 1
    return [a.strip() for a in args]


def _relation_blob(name, args):
    if name in ("panic", "unreachable", "todo", "unimplemented"):
        return None
    if name in ("assert", "debug_assert"):
        return args[0] if args else ""
    a0 = args[0] if len(args) > 0 else ""
    a1 = args[1] if len(args) > 1 else ""
    return a0 + " @@ " + a1


def _is_headdim_relation(blob):
    """The GAP-281 relation: a `heads * head_dim` product against a width term."""
    return bool(
        HEADDIM.search(blob)
        and (HEADCOUNT.search(blob) or WIDTH.search(blob))
        and "*" in blob
    )


def classify(name, args):
    blob = _relation_blob(name, args)
    if blob is None:
        return "UNCONDITIONAL"
    if _is_headdim_relation(blob):
        return "HEADDIM_RELATION"
    if TENSORISH.search(blob):
        return "TENSOR_SHAPE_LEN"
    if CONFIGISH.search(blob):
        return "CONFIG_FIELD"
    if "*" in blob:
        return "PRODUCT_NONTENSOR"
    return "SCALAR_OTHER"


def is_buffer_len(name, args):
    """A `something.len()` compared to a product — the architect's control class."""
    blob = _relation_blob(name, args)
    return bool(blob) and ".len()" in blob and "*" in blob


def model_files():
    return sorted(
        os.path.join(MODELS_DIR, f)
        for f in os.listdir(MODELS_DIR)
        if f.endswith(".rs")
    )


def census(files=None):
    """Every production panic site: list of (base, line, macro, bucket, buffer_len)."""
    if files is None:
        files = model_files()
    sites = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        spans = cfg_test_spans(src)
        base = os.path.basename(path)
        for m in MACRO.finditer(src):
            if in_test(m.start(), spans):
                continue
            name = m.group(1)
            args = parse_args(src, m.end() - 1)
            line = src.count("\n", 0, m.start()) + 1
            sites.append((base, line, name, classify(name, args), is_buffer_len(name, args)))
    return sites


def adjacent_floor(files=None):
    """.unwrap()/.expect( — a SEPARATE panic class, named as a floor, not in the census."""
    if files is None:
        files = model_files()
    unwrap = expect = 0
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
        spans = cfg_test_spans(src)
        for m in re.finditer(r"\.unwrap\(\)", src):
            if not in_test(m.start(), spans):
                unwrap += 1
        for m in re.finditer(r"\.expect\(", src):
            if not in_test(m.start(), spans):
                expect += 1
    return unwrap, expect


def git_ref():
    """Best-effort HEAD label for the checkout ROOT lives in (filesystem only).

    No child process: a census header is a nicety, and shelling out to `git` trips
    the B603/B607 lints and adds a PATH dependency. Detached HEAD -> short sha; on a
    branch -> the branch name.
    """
    try:
        gitdir = os.path.join(ROOT, ".git")
        if os.path.isfile(gitdir):  # linked worktree: .git is a "gitdir: <path>" file
            with open(gitdir, encoding="utf-8") as fh:
                gitdir = fh.read().split(":", 1)[1].strip()
        with open(os.path.join(gitdir, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
    except OSError:
        return "unknown"
    if head.startswith("ref:"):
        return head.rsplit("/", 1)[-1]
    return head[:8]


def _summarize(sites):
    """The numbers print_report shows, computed in one place (keeps print_report flat)."""
    crate = [s for s in sites if s[3] != "HEADDIM_RELATION"]
    return {
        "total": len(sites),
        "headdim": sum(1 for s in sites if s[3] == "HEADDIM_RELATION"),
        "crate": len(crate),
        "partition": Counter(s[3] for s in crate).most_common(),
        "buf_sites": sum(1 for s in crate if s[4]),
        "buf_files": len({s[0] for s in crate if s[4]}),
    }


def print_report():
    s = _summarize(census())
    print("PRODUCTION PANIC CENSUS @ %s | fuel-transformers/src/models/" % git_ref())
    print("population: assert*/panic!/unreachable!/todo! ; per-SITE ; #[cfg(test)] masked")
    print("total production panic sites: %d" % s["total"])
    print("  of which head_dim relation (GAP-281, excluded from crate census): %d" % s["headdim"])
    print("  CRATE CENSUS (excl. head_dim relation): %d" % s["crate"])
    print("\nSTRUCTURAL PARTITION:")
    for bucket, count in s["partition"]:
        print("  %-18s %d" % (bucket, count))
    print("\nbuffer-length control class (X.len() == product): %d sites in %d files"
          % (s["buf_sites"], s["buf_files"]))
    unwrap, expect = adjacent_floor()
    print("\nADJACENT FLOOR (SEPARATE class, macro family only, NOT in the census):")
    print("  .unwrap()  %d" % unwrap)
    print("  .expect(   %d" % expect)


# The buffer-length asserts the census was born to reproduce (the architect's
# hand-list). Named by FILE, not line: line numbers rot, file membership does not.
CONTROL_FILES = [
    "lazy_bert.rs", "lazy_csm.rs", "lazy_mimi_seanet.rs", "lazy_snac.rs",
    "lazy_t5.rs", "lazy_voxtral.rs", "lazy_whisper.rs", "lazy_yolov8.rs",
]


def _arm_a():
    """The parser survives a ';' and a '(' inside a message string (a naive one fails)."""
    probe = 'assert_eq!(a, "b;c(d)e", f);'
    args = parse_args(probe, probe.index("("))
    return args == ["a", '"b;c(d)e"', "f"], args


def _arm_b():
    """The census sees the corpus: the buffer-length control class is present."""
    sites = census()
    buf_files = {s[0] for s in sites if s[4]}
    missing = [f for f in CONTROL_FILES if f not in buf_files]
    ok = not missing and len([s for s in sites if s[4]]) >= 8
    return ok, missing


def self_test():
    """Two arms: the parser survives adversarial strings, and it sees the corpus."""
    print("SELF-TEST @ %s" % git_ref())
    arm_a, a_args = _arm_a()
    print("  ARM A (string-aware parse survives ';' and '(' in a message): %s  got=%r"
          % ("PASS" if arm_a else "FAIL", a_args))
    arm_b, missing = _arm_b()
    print("  ARM B (>=8 buffer-length sites, all control files present):        %s%s"
          % ("PASS" if arm_b else "FAIL", "" if not missing else "  missing=%r" % missing))
    ok = arm_a and arm_b
    print("SELF-TEST: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def gate():
    """Regression guard: GAP-281's head_dim relation must stay 0 production ASSERTS."""
    head = [s for s in census() if s[3] == "HEADDIM_RELATION"]
    if head:
        print("GATE FAIL: %d head_dim-relation production assert(s) reappeared — GAP-281 "
              "was closed by converting these to typed declines. Convert, do not assert:"
              % len(head))
        for s in head:
            print("  %s:%d  %s!" % (s[0], s[1], s[2]))
        return 1
    print("GATE OK: 0 head_dim-relation production asserts (GAP-281 stays converted).")
    return 0


def main(argv):
    if "--self-test" in argv:
        return self_test()
    if "--gate" in argv:
        return gate()
    print_report()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

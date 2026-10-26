#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Classify every tracked file against candle, the project this repository forked from.

    python scripts/provenance.py generate   # rewrite docs/provenance/FILES.tsv and the summary in README.md
    python scripts/provenance.py summary    # print the per-directory class counts

Needs only this repository's own history: the fork point is the last pure-candle commit,
FORK_TREE_COMMIT, whose tree is identical (by tree hash) to upstream huggingface/candle
097655a29e17ad4891bb1014ec7f3aee53e64e6d.
`fuel-ir/tests/spdx_header_ratchet.rs` (candle_provenance_table_covers_every_tracked_file) is the CI side:
it needs no candle tree, it fails when a tracked file has no row (or a row has no file).

What the classes MEAN (a class is evidence, not a verdict on every idea in a file):
  candle           same path or a rename/copy of a candle file, and >= 90% of its lines (>25 chars) appear verbatim
  candle-modified  lineage to a candle path (same path, or git rename/copy at >= 30% similarity), or >= 30% verbatim
  ggml             derived from llama.cpp / ggml (THIRD-PARTY-NOTICES.md)
  not-traced       no lineage, but >= 10% (or >= 20 lines) of its lines also appear in candle, or it names candle
  data             binary / data file with no match; provenance is its directory's PROVENANCE.md where present
  copy             byte-identical packaged copy of a root file (scripts/sync-packaged-notices.py)
  own              no lineage and no meaningful verbatim overlap with candle. NOT a proof of independent authorship
"""
import collections
import os
import shutil
import subprocess  # nosec B404 -- only used with fixed argument lists and no shell
import sys
from typing import NamedTuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIT = shutil.which("git") or "git"
FORK_TREE_COMMIT = "b4b15281b"  # last pure-candle commit in this history; tree == upstream 097655a29e
TABLE = "docs/provenance/FILES.tsv"
README = "docs/provenance/README.md"
META = {TABLE, README}
MARK_BEGIN = "<!-- provenance-summary:begin -->"
MARK_END = "<!-- provenance-summary:end -->"
PACKAGED = ("LICENSE-MIT", "LICENSE-APACHE", "THIRD-PARTY-NOTICES.md")
GGML = {
    "fuel-quantized/src/iq4.rs",
    "fuel-quantized/src/iq_grid_types.rs",
    "fuel-quantized/src/iq_grids.rs",
    "fuel-quantized/src/tq_mx_types.rs",
}
SOURCE_EXT = {"rs", "metal", "cu", "cuh", "slang", "glsl", "py", "ps1", "sh", "h", "c", "cpp", "proto3", "proto"}
CLASSES = ["candle", "candle-modified", "not-traced", "ggml", "own", "data", "copy"]


def git(*a, text=True):
    out = subprocess.check_output([GIT, *a], cwd=ROOT)  # NOSONAR # nosec B603 -- fixed argv, no shell
    return out.decode("utf-8", "surrogateescape") if text else out


def blob_map(rev):
    ents = []
    for e in git("ls-tree", "-r", "-z", rev, text=False).split(b"\0"):
        if e:
            meta, path = e.split(b"\t", 1)
            ents.append((path.decode("utf-8", "surrogateescape"), meta.split()[2]))
    p = subprocess.Popen(  # NOSONAR # nosec B603 -- fixed argv, no shell
        [GIT, "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    out = {}
    for path, oid in ents:
        p.stdin.write(oid + b"\n")
        p.stdin.flush()
        size = int(p.stdout.readline().split()[2])
        out[path] = p.stdout.read(size)
        p.stdout.read(1)
    p.stdin.close()
    p.wait()
    return out


def lines_of(b):
    """Stripped lines longer than 25 characters, or None for a binary file."""
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return [ln.strip() for ln in t.replace("\r\n", "\n").split("\n") if len(ln.strip()) > 25]


def renamed(line):
    """The fork's global candle -> fuel rename, applied to a candle line so renamed lines still match."""
    return line.replace("candle", "fuel").replace("Candle", "Fuel").replace("CANDLE", "FUEL")


# Directories that are, by the fork's own design, re-expressions of candle's: every source file in them is
# presumed a port until a person judges it (the candle directory it mirrors is the basis).
PORT_DIRS = {
    "fuel-transformers/src/models/": "candle-transformers/src/models/",
    "fuel-lazy-examples/src/bin/": "candle-examples/examples/",
}
# Single files that implement a model candle also implements (the file name is generic, so the stem rule misses them).
MODEL_PORTS = {
    "fuel-model-llama/src/lib.rs": "candle-transformers/src/models/llama.rs",
    "fuel-model-phi/src/lib.rs": "candle-transformers/src/models/phi.rs",
}
# Files a person read (Fuel lane, 2026-10-25) after the automatic rule flagged them: the class and why.
JUDGED = {
    "fuel-ir/tests/no_utf8_bom.rs": ("own", "judged: names candle only to describe the fork rename"),
    "fuel-ir/tests/spdx_header_ratchet.rs": ("own", "judged: names candle only to describe the fork and its licences"),
    "scripts/provenance.py": ("own", "judged: names candle only to describe the fork"),
    "scripts/sync-packaged-notices.py": ("own", "judged: names candle only to describe its licences"),
    "scripts/verify-packaged-crates.py": ("own", "judged: names candle only to describe its licences"),
    "fuel-kernels-source/kernels/index_select.slang": (
        "own", "judged: a comment cites candle's index_select semantics; Slang kernel, no shared lines"),
    "fuel-loaders/src/model_progress.rs": (
        "own", "judged: 'candlelight' in a doc comment is a different project, not candle"),
    "THIRD-PARTY-NOTICES.md": (
        "own", "judged: the ledger; it reproduces upstream licence texts, which is why git sees a copy of LICENSE-APACHE"),
}
GENERIC_STEMS = {"lib", "mod", "main", "utils", "error", "config", "tests", "test", "build", "kernel", "common"}


def stem(path):
    """File stem with the fork's lazy_/quantized_ prefixes dropped, for same-name matching."""
    name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower().replace("-", "_")
    while name.startswith(("lazy_", "quantized_")):
        name = name.split("_", 1)[1]
    return name


def candle_stems(paths):
    """(extension, stem) -> one candle path, for non-test, non-bench candle source files."""
    idx = {}
    for p in sorted(paths):
        ext = p.rsplit(".", 1)[-1]
        if ext in SOURCE_EXT and "/benches/" not in p and "/tests/" not in p:
            idx.setdefault((ext, stem(p)), p)
    return idx


def lineage():
    """path -> (status, source path, similarity) from git's rename/copy detection vs the fork tree."""
    out = git("-c", "diff.renameLimit=999999", "diff", "--find-copies-harder", "-M30%", "-C30%", "-l0",
              "--name-status", FORK_TREE_COMMIT, "HEAD")
    st = {}
    for ln in out.splitlines():
        f = ln.split("\t")
        if f[0][0] in "RC":
            st[f[2]] = (f[0][0], f[1], int(f[0][1:] or 0))
        elif f[0] in ("A", "M"):
            st[f[-1]] = (f[0], f[-1] if f[0] == "M" else "", 0)
    return st


class Facts(NamedTuple):
    """Everything the rules read about one file."""

    p: str
    status: str
    src: str
    sim: int
    ls: object  # list of long lines, or None for a binary file
    hit: list
    top: str
    says: bool
    same_name: str

    @property
    def n(self):
        return len(self.ls) if self.ls else 0

    @property
    def cov(self):
        return len(self.hit) / self.n if self.n else 0.0

    @property
    def pct(self):
        return f"{round(self.cov * 100)}% of {self.n} lines verbatim"

    @property
    def is_src(self):
        ext = self.p.rsplit(".", 1)[-1] if "." in self.p.rsplit("/", 1)[-1] else ""
        return self.ls is not None and ext in SOURCE_EXT


def rule_judged(f):
    return (JUDGED[f.p][0], "", JUDGED[f.p][1]) if f.p in JUDGED else None


def rule_copy(f):
    name = f.p.partition("/")[2]
    if "/" in f.p and name in PACKAGED:
        return "copy", "", f"byte-identical packaged copy of the root {name}"
    return None


def rule_ggml(f):
    return ("ggml", "", "llama.cpp / ggml (MIT), THIRD-PARTY-NOTICES.md") if f.p in GGML else None


def rule_lineage(f):
    if not (f.status in ("=", "M") or (f.status in "RC" and f.sim >= 30)):
        return None
    how = {"=": "identical path", "M": "same path"}.get(f.status, f"rename/copy {f.sim}%")
    near_identical = f.status == "=" or (f.cov >= 0.9 and (f.status == "M" or f.sim >= 90))
    return ("candle" if near_identical else "candle-modified"), f.src, f"{how}; {f.pct}"


def rule_overlap_modified(f):
    if f.is_src and f.n >= 10 and f.cov >= 0.30:
        return "candle-modified", f.top, f"no git lineage; {len(f.hit)} of {f.n} lines verbatim in candle (mostly {f.top})"
    return None


def rule_overlap_untraced(f):
    if f.is_src and ((f.n >= 10 and f.cov >= 0.10) or len(f.hit) >= 20):
        return "not-traced", f.top, f"{len(f.hit)} of {f.n} lines also appear in candle (mostly {f.top}); not judged"
    return None


def rule_names_candle(f):
    return ("not-traced", "", "names candle in its text; not judged") if f.is_src and f.says else None


def rule_model_port(f):
    if f.is_src and f.p in MODEL_PORTS:
        return "not-traced", MODEL_PORTS[f.p], f"implements the model candle has at {MODEL_PORTS[f.p]}; a port is presumed until judged"
    return None


def rule_mirror_dir(f):
    mirror = next((v for k, v in PORT_DIRS.items() if f.p.startswith(k)), "")
    if f.is_src and mirror and not f.same_name:
        return "not-traced", "", f"lives in a directory that mirrors {mirror}; no same-name file; not judged"
    return None


def rule_same_name(f):
    if f.is_src and f.same_name:
        return "not-traced", f.same_name, f"same file name as candle {f.same_name}; a port or re-expression is presumed until judged"
    return None


def rule_binary(f):
    return ("data", "", "binary file; no path or similarity match to a candle file") if f.ls is None else None


def rule_own(f):
    return "own", "", f"no lineage; {len(f.hit)} of {f.n} lines verbatim in candle ({round(f.cov * 100)}%)"


# The order is the precedence: the first rule that answers decides the class.
RULES = (
    rule_judged, rule_copy, rule_ggml, rule_lineage, rule_overlap_modified, rule_overlap_untraced,
    rule_names_candle, rule_model_port, rule_mirror_dir, rule_same_name, rule_binary, rule_own,
)


def verdict(f):
    """(class, candle_source, basis) for one file."""
    return next(r for r in (rule(f) for rule in RULES) if r is not None)


def candle_corpus():
    """(line -> candle files containing it, (extension, stem) -> candle path) from the fork tree."""
    corp = collections.defaultdict(set)
    blobs = blob_map(FORK_TREE_COMMIT)
    for p, b in blobs.items():
        for ln in set(lines_of(b) or []):
            corp[renamed(ln)].add(p)
    return corp, candle_stems(blobs.keys())


def top_source(hit, corp):
    """The candle file that most of the matching lines come from (ties by name: deterministic)."""
    votes = collections.Counter(f for ln in hit for f in corp[ln])
    return min(votes, key=lambda f: (-votes[f], f)) if votes else ""


def same_name_source(p, stems):
    """The candle file with the same (extension, stem), unless the stem is too generic to mean it."""
    s = stem(p)
    if s in GENERIC_STEMS or len(s) < 4:
        return ""
    return stems.get((p.rsplit(".", 1)[-1], s), "")


def facts_for(p, b, st, corp, stems):
    status, src, sim = st.get(p, ("=", p, 100))
    ls = lines_of(b)
    hit = [ln for ln in (ls or []) if ln in corp]
    says = ls is not None and "candle" in b.decode("utf-8", "ignore").lower()
    return Facts(p, status, src, sim, ls, hit, top_source(hit, corp), says, same_name_source(p, stems))


def classify():
    tracked = [p for p in git("ls-files", "-z").split("\0") if p and p not in META]
    corp, stems = candle_corpus()
    cur = blob_map("HEAD")
    st = lineage()
    return [(p, *verdict(facts_for(p, cur[p], st, corp, stems))) for p in sorted(tracked)]


def write_table(rows):
    body = "path\tclass\tcandle_source\tbasis\n" + "".join("\t".join(r) + "\n" for r in rows)
    with open(os.path.join(ROOT, TABLE), "w", encoding="utf-8", newline="\n") as f:
        f.write(body)


def summary(rows):
    per = collections.defaultdict(collections.Counter)
    for p, cls, _, _ in rows:
        per[p.split("/")[0] if "/" in p else "(root)"][cls] += 1
    out = ["| directory | files | " + " | ".join(CLASSES) + " |", "|---|--:|" + "--:|" * len(CLASSES)]
    for d in sorted(per):
        c = per[d]
        out.append(f"| `{d}` | {sum(c.values())} | " + " | ".join(str(c[k]) if c[k] else "" for k in CLASSES) + " |")
    tot = collections.Counter(r[1] for r in rows)
    out.append(f"| **all** | **{len(rows)}** | " + " | ".join(f"**{tot[k]}**" for k in CLASSES) + " |")
    return "\n".join(out)


def write_readme(rows):
    p = os.path.join(ROOT, README)
    with open(p, encoding="utf-8", newline="") as f:
        s = f.read().replace("\r\n", "\n")
    a, b = s.index(MARK_BEGIN) + len(MARK_BEGIN), s.index(MARK_END)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s[:a] + "\n" + summary(rows) + "\n" + s[b:])


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd not in ("generate", "summary"):
        print(__doc__)
        return 2
    rows = classify()
    if cmd == "summary":
        print(summary(rows))
        return 0
    write_table(rows)
    write_readme(rows)
    print(f"{len(rows)} rows -> {TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

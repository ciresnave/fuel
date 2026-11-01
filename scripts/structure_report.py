#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ROADMAP row 136: measure Fuel's structure before touching it.

Run from the repository root:  python scripts/structure_report.py [--window N] [--ref REF]

Every number is computed from `git ls-files` (never a disk walk), `cargo metadata --no-deps` and
`git log --first-parent REF`, and every definition is in the docstring of the function that computes
it, so another tool (CodeRipper's architecture and modularity rules) can be validated against the
same definitions. A failed control prints `CONTROL FAILED` and exits non-zero.

Definitions, in one place (the same text is in docs/design-notes/2026-11-01-structure-measurement.md):
  member        a package in the workspace (`cargo metadata --no-deps`).
  edge          member A depends on member B in A's manifest. kind = normal | dev. `optional` marks a
                normal edge that exists only behind a feature of A.
  tier          `fuel-ir/tests/crate_dependency_tiers.txt` (normative). A judged edge has a tier at
                both ends. A DIRECTION VIOLATION is a judged normal edge A -> B with tier(A) < tier(B).
  backend crate a member whose tier-file layer is `backends`, `vendor backends` or `reference backend`.
  kernel crate  layer `kernels`.
  generic crate every other member (neither backend nor kernel).
  backend leak  in a generic crate, in a `.rs` file under `src/` (tests/ counted separately), after
                removing `//` comments: a line with `feature = "cuda"|"vulkan"|"metal"|"mkl"|"aocl"`
                (a cfg line), or the path `BackendId::X` or `DeviceLocation::{Cuda,Vulkan,Metal}` (a
                backend-path use), or a source file or directory whose name contains cuda, vulkan or
                metal. `#[cfg(test)]` code inside `src/` is NOT separated (stated limitation).
  commit-crates the set of members with at least one changed file in a commit, ignoring files named
                Cargo.toml, Cargo.lock, *.md, and anything outside member directories. Commits are the
                first-parent commits of REF (one per squash-merged PR).
  sweep         a commit touching more than SWEEP crates (default 12); reported with and without.
  co-change     for crates A and B: c(A,B) commits touching both; J = c / (n(A) + n(B) - c).
  seam census   for a LEFT group L and a RIGHT group R: every path in L's `.rs` files that starts with
                the lib name of a crate in R (`fuel_x::...`), comments removed. `use` statements are
                brace-expanded so each imported leaf is one site; other occurrences are one site each.
                A glob import is counted separately and not as an item. An ITEM is the path after the
                crate name cut after its first segment that starts with an uppercase letter, else cut
                to 3 segments. A site is UP-TIER when the referring crate's tier is lower than the
                named crate's (a source-level back-edge by the tier file's own order).
"""

import argparse
import collections
import itertools
import json
import re
import shutil
import subprocess  # nosec B404 - runs git and cargo by argv list, no shell; developer-run measurement script
import sys
from pathlib import Path

REF_OK = re.compile(r"^[A-Za-z0-9._/@^~-]+$")
GIT = shutil.which("git") or "git"
CARGO = shutil.which("cargo") or "cargo"


def run(argv, cwd=None):
    """The one place a subprocess starts. argv is a list of fixed tool paths (resolved with
    shutil.which) and arguments built here or validated (`--ref` against REF_OK); no shell."""
    # nosec B603 - argv list, no shell, tools resolved by shutil.which
    # nosemgrep - the only caller-supplied argument is the ref, validated by REF_OK in main()
    done = subprocess.run(argv, check=True, capture_output=True, text=True,  # nosec B603  # nosemgrep
                          encoding="utf-8", errors="replace", cwd=cwd)
    return done.stdout


ROOT = Path(run([GIT, "rev-parse", "--show-toplevel"]).strip())


def git(*args):
    return run([GIT, *args], cwd=ROOT)


def require(ok, message):
    """A control. Unlike `assert` it survives `python -O`."""
    if not ok:
        print(f"CONTROL FAILED: {message}")
        sys.exit(2)


def code_only(txt):
    """Source text without `//` comments (line and doc), so paths in prose are not counted. A `//`
    inside a string literal or after `:` (a URL) is kept; block comments are not handled (rare here)."""
    out = []
    for ln in txt.splitlines():
        if ln.lstrip().startswith("//"):
            continue
        out.append(cut_trailing_comment(ln))
    return chr(10).join(out)


def cut_trailing_comment(ln):
    i = ln.find("//")
    while i != -1:
        if (i == 0 or ln[i - 1] != ":") and ln[:i].count('"') % 2 == 0:
            return ln[:i]
        i = ln.find("//", i + 2)
    return ln


def read_code(path):
    return code_only((ROOT / path).read_text(encoding="utf-8", errors="replace"))


# ----------------------------------------------------------------------------------------------- load
def load_workspace():
    meta = json.loads(run([CARGO, "metadata", "--no-deps", "--format-version", "1", "--offline"], cwd=ROOT))
    members = {}
    for p in meta["packages"]:
        mdir = Path(p["manifest_path"]).parent.resolve().relative_to(ROOT.resolve()).as_posix()
        members[p["name"]] = {"dir": mdir if mdir != "." else "", "deps": p["dependencies"]}
    edges = [(n, d["name"], d["kind"] or "normal", bool(d.get("optional")))
             for n, m in members.items() for d in m["deps"] if d["name"] in members]
    return members, edges


def load_tiers():
    tiers = {}
    text = (ROOT / "fuel-ir/tests/crate_dependency_tiers.txt").read_text(encoding="utf-8")
    for ln in text.splitlines():
        parts = ln.split("\t")
        if ln.strip() and not ln.lstrip().startswith("#") and len(parts) >= 3:
            tiers[parts[0]] = (int(parts[1]), parts[2])
    return tiers


def member_dirs(members):
    return sorted(((m["dir"], n) for n, m in members.items() if m["dir"]), key=lambda t: -len(t[0]))


def crate_of(path, dirs):
    for d, n in dirs:
        if path.startswith(d + "/"):
            return n, path[len(d) + 1:]
    return None, path


def tracked_rs(members):
    """{crate: {"src": [paths], "other": [paths]}} of tracked .rs files, from the index."""
    dirs = member_dirs(members)
    by_crate = collections.defaultdict(lambda: {"src": [], "other": []})
    for f in filter(None, git("ls-files", "-z", "--", "*.rs").split("\0")):
        crate, rel = crate_of(f, dirs)
        if crate:
            by_crate[crate]["src" if rel.startswith("src/") else "other"].append(f)
    return by_crate


# ----------------------------------------------------------------------------------------------- A
def violations(edges, tiers, kind):
    return [e for e in edges if e[2] == kind and e[0] in tiers and e[1] in tiers and tiers[e[0]][0] < tiers[e[1]][0]]


def names(pairs):
    return ", ".join(f"{a}->{b}" for a, b, *_ in pairs) or "none"


def top(counter, n=8):
    return ", ".join(f"{k} {v}" for k, v in counter.most_common(n))


def section_graph(members, edges, tiers):
    normal = [e for e in edges if e[2] == "normal"]
    judged = [e for e in normal if e[0] in tiers and e[1] in tiers]
    unplaced = sorted(n for n in members if n not in tiers)
    stale = sorted(n for n in tiers if n not in members)
    backend = {n for n, (_, layer) in tiers.items() if layer in ("backends", "vendor backends", "reference backend")}
    opt_to_backend = sorted((a, b) for a, b, _, o in normal if o and b in backend and a not in backend)
    require(("fuel-tensor", "fuel-ir", "normal", False) in edges, "fuel-tensor -> fuel-ir edge must exist")
    return [
        f"members {len(members)}; edges normal {len(normal)} (optional {sum(1 for e in normal if e[3])}), dev {len(edges) - len(normal)}",
        f"tier file places {len(tiers) - len(stale)} of {len(members)} members; unplaced: {', '.join(unplaced) or 'none'}",
        f"tier-file entries naming a non-member (stale): {', '.join(stale) or 'none'}",
        f"judged normal edges {len(judged)} of {len(normal)}; DIRECTION VIOLATIONS {len(violations(edges, tiers, 'normal'))}: "
        + names(violations(edges, tiers, "normal")),
        f"dev-edge direction violations (not gated): {len(violations(edges, tiers, 'dev'))}: " + names(violations(edges, tiers, "dev")),
        "top fan-in (normal): " + top(collections.Counter(b for _, b, *_ in normal)),
        "top fan-out (normal): " + top(collections.Counter(a for a, *_ in normal)),
        "optional edges from a non-backend crate to a backend crate: " + names(opt_to_backend),
        "control ok: fuel-tensor -> fuel-ir edge present",
    ]


# ----------------------------------------------------------------------------------------------- C
LEAK_CFG = re.compile(r'feature\s*=\s*"(cuda|vulkan|metal|mkl|aocl)"')
LEAK_ID = re.compile(r"\b(BackendId::(Cuda|Vulkan|Metal|Mkl|Aocl)|DeviceLocation::(Cuda|Vulkan|Metal))\b")
LEAK_NAME = re.compile(r"(cuda|vulkan|metal)", re.I)


def named_for_backend(path):
    parts = Path(path).parts
    return bool(LEAK_NAME.search(parts[-1])) or any(LEAK_NAME.search(p) for p in parts[:-1] if p != "tests")


def count_leaks(files, with_names):
    cfg = ids = 0
    named = []
    for f in files:
        txt = read_code(f)
        cfg += len(LEAK_CFG.findall(txt))
        ids += len(LEAK_ID.findall(txt))
        if with_names and named_for_backend(f):
            named.append(f)
    return cfg, ids, named


def leak_row(crate, rs):
    cfg_s, id_s, named = count_leaks(rs[crate]["src"], True)
    cfg_t, id_t, _ = count_leaks(rs[crate]["other"], False)
    return (crate, cfg_s, id_s, len(named), cfg_t, id_t)


def top_leak_files(crate, rs):
    per = []
    for f in rs[crate]["src"]:
        t = read_code(f)
        per.append((len(LEAK_CFG.findall(t)) + len(LEAK_ID.findall(t)), f))
    per.sort(reverse=True)
    return f"   top {crate} src files by backend-path+cfg lines: " + "; ".join(
        f"{f.split('/', 1)[1]} {n}" for n, f in per[:5] if n)


def section_leaks(members, tiers, rs):
    skip = {n for n in members if tiers.get(n, (0, ""))[1] in ("backends", "vendor backends", "reference backend", "kernels")}
    rows = [leak_row(n, rs) for n in sorted(set(members) - skip)]
    rows = [r for r in rows if any(r[1:])]
    rows.sort(key=lambda r: -(r[1] + r[2]))
    tot_cfg, tot_id = sum(r[1] for r in rows), sum(r[2] for r in rows)
    require(tot_cfg + tot_id > 0, "the scan must find the known backend modules")
    out = ["generic crate | src cfg(feature=backend) | src BackendId/DeviceLocation::X | src files/dirs named for a backend | tests cfg | tests ids"]
    out += [" | ".join(map(str, r)) for r in rows]
    out += [top_leak_files(c, rs) for c in ("fuel-dispatch", "fuel-tensor")]
    out.append(f"TOTAL in generic crates' src: {tot_cfg} cfg lines, {tot_id} backend-path uses, across {sum(1 for r in rows if r[1] or r[2])} crates")
    out.append("control ok: scan is non-empty")
    return out


# ----------------------------------------------------------------------------------------------- B
def parse_commits(members, ref, window):
    dirs = member_dirs(members)
    commits, cur = [], None
    for ln in git("log", ref, "--first-parent", f"-n{window}", "--name-only", "--format=@@%H %cs").split("\n"):
        if ln.startswith("@@"):
            cur = {"date": ln.split()[1], "crates": set()}
            commits.append(cur)
        elif ln.strip() and cur is not None and counts_as_code(ln):
            crate, _ = crate_of(ln, dirs)
            if crate:
                cur["crates"].add(crate)
    return [c for c in commits if c["crates"]]


def counts_as_code(path):
    base = path.rsplit("/", 1)[-1]
    return base not in ("Cargo.toml", "Cargo.lock") and not base.endswith(".md")


def cochange_matrix(commits):
    n, c = collections.Counter(), collections.Counter()
    for x in commits:
        n.update(x["crates"])
        c.update(itertools.combinations(sorted(x["crates"]), 2))
    return n, c


def pair_summary(n, c):
    pairs = [(a, b, v, v / (n[a] + n[b] - v)) for (a, b), v in c.items() if v >= 5]
    pairs.sort(key=lambda t: -t[3])
    return "; ".join(f"{a}+{b} {v} {j:.2f}" for a, b, v, j in pairs[:12])


def section_cochange(members, ref, window, sweep):
    commits = parse_commits(members, ref, window)
    out = [f"window: {len(commits)} first-parent commits touching at least one member's code, {commits[-1]['date']} to {commits[0]['date']} (ref {ref}, first {window} commits)"]
    out.append(f"sweeps (> {sweep} crates in one commit): {sum(1 for c in commits if len(c['crates']) > sweep)}; shown with and without")
    for label, cs in (("ALL commits", commits), ("WITHOUT sweeps", [c for c in commits if len(c["crates"]) <= sweep])):
        n, c = cochange_matrix(cs)
        out.append(f"-- {label}: {len(cs)} commits; most-changed crates: " + top(n, 10))
        out.append("   pairs with c>=5 by Jaccard (A, B, c, J): " + pair_summary(n, c))
    return out, commits


def seam_line(name, commits, left, right):
    l = sum(1 for c in commits if c["crates"] & left and not c["crates"] & right)
    r = sum(1 for c in commits if c["crates"] & right and not c["crates"] & left)
    both = sum(1 for c in commits if c["crates"] & left and c["crates"] & right)
    share = f"{both / (l + both):.0%}" if l + both else "n/a"
    return f"{name}: left-only {l}, right-only {r}, both {both} (of {l + r + both}); share of left commits that also touch right = {share}"


def seam_cochange(commits, groups, sweep):
    use = [c for c in commits if len(c["crates"]) <= sweep]
    return [seam_line(name, use, left, right) for name, (left, right) in groups.items()]


# ----------------------------------------------------------------------------------------------- E
USE_STMT = re.compile(r"\buse\s+(fuel_[a-z0-9_]+(?:::[^;]*)?);", re.S)
PATH_TOK = re.compile(r"\b(fuel_[a-z0-9_]+)((?:::[A-Za-z_][A-Za-z0-9_]*)+)")


def split_top(body):
    """Split `a, b::{c, d}, e` at top-level commas."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        depth += (ch == "{") - (ch == "}")
        if ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p for p in parts if p]


def expand(path):
    """Brace-expand a use path into leaf paths; '*' leaves stay as a trailing '::*'."""
    path = re.sub(r"\s+", "", path)
    open_at = path.find("{")
    if open_at < 0:
        return [path]
    depth, close = 0, len(path) - 1
    for j in range(open_at, len(path)):
        depth += (path[j] == "{") - (path[j] == "}")
        if depth == 0:
            close = j
            break
    prefix, suffix = path[:open_at], path[close + 1:]
    return [leaf for part in split_top(path[open_at + 1:close]) for leaf in expand(prefix + part + suffix)]


def item_of(segs):
    out = []
    for s in segs[:3]:
        out.append(s)
        if s[:1].isupper():
            break
    return "::".join(out)


def refs_in(txt):
    """Every `fuel_x::...` path in a source text, as a list of segment lists."""
    refs = [[s for s in leaf.split("::") if s] for m in USE_STMT.finditer(txt) for leaf in expand(m.group(1))]
    rest = USE_STMT.sub(" ", txt)
    refs += [[m.group(1)] + [s for s in m.group(2).split("::") if s] for m in PATH_TOK.finditer(rest)]
    return refs


def record(segs, crate, path, sites, globs, nfiles):
    if segs[-1] in ("*", "self") and len(segs) <= 2:
        globs[(crate, segs[0])] += 1
        return
    if segs[-1] == "*":
        globs[(crate, segs[0])] += 1
        segs = segs[:-1]
    key = (crate, segs[0], item_of(segs[1:]) if len(segs) > 1 else "(crate)")
    sites[key] += 1
    nfiles[key].add(path)


def census(files, target_libs):
    """files: list of (referring crate, path). Returns sites[(referrer, lib, item)], globs, files-per-key."""
    sites, globs, nfiles = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for crate, path in files:
        for segs in refs_in(read_code(path)):
            if segs[0] in target_libs:
                record(segs, crate, path, sites, globs, nfiles)
    return sites, globs, nfiles


def lib_name(crate):
    return crate.replace("-", "_")


def up_tier(sites, tiers, lib2crate):
    n = 0
    for (ref, lib, _), c in sites.items():
        t1, t2 = tiers.get(ref), tiers.get(lib2crate.get(lib, ""))
        if t1 and t2 and t1[0] < t2[0]:
            n += c
    return n


def tier_range(group, tiers):
    ts = sorted({tiers[n][0] for n in group if n in tiers})
    return f"tiers {ts[0]}-{ts[-1]}" if ts else "no tier"


def merged(sites):
    d = collections.Counter()
    for (_, lib, it), c in sites.items():
        d[(lib, it)] += c
    return d


def files_of(group, rs):
    return [(n, f) for n in sorted(group) for part in ("src", "other") for f in rs[n][part]]


def seam_census_lines(name, left, right, rs, tiers, lib2crate):
    s1, g1, nf1 = census(files_of(left, rs), {lib_name(n) for n in right})
    s2, g2, _ = census(files_of(right, rs), {lib_name(n) for n in left})
    d1, d2 = merged(s1), merged(s2)
    nfiles = len({f for fs in nf1.values() for f in fs})
    out = [f"{name} [LEFT {tier_range(left, tiers)}; RIGHT {tier_range(right, tiers)}]: "
           f"LEFT->RIGHT {sum(s1.values())} sites ({up_tier(s1, tiers, lib2crate)} up-tier), {len(d1)} distinct items, {sum(g1.values())} globs, in {nfiles} files"
           f"; RIGHT->LEFT {sum(s2.values())} sites ({up_tier(s2, tiers, lib2crate)} up-tier), {len(d2)} distinct items, {sum(g2.values())} globs",
           "   top items LEFT->RIGHT: " + ", ".join(f"{lib}::{it} {c}" for (lib, it), c in d1.most_common(6))]
    if d2:
        out.append("   top items RIGHT->LEFT: " + ", ".join(f"{lib}::{it} {c}" for (lib, it), c in d2.most_common(6)))
    return out


def section_census(members, rs, groups, tiers):
    lib2crate = {lib_name(n): n for n in members}
    out = []
    for name, (left, right) in groups.items():
        out += seam_census_lines(name, left, right, rs, tiers, lib2crate)
    s, _, _ = census([("fuel-tensor", f) for f in rs["fuel-tensor"]["src"]], {"fuel_ir"})
    require(sum(s.values()) > 0, "fuel-tensor must reference fuel_ir")
    out.append(f"control ok: fuel-tensor src references fuel_ir at {sum(s.values())} sites")
    return out


# ----------------------------------------------------------------------------------------------- main
CORE = {"fuel-ir", "fuel-graph", "fuel-memory", "fuel-dispatch", "fuel-tensor", "fuel-hardware", "fuel-backend-contract"}
BACKENDS = {"fuel-cpu-backend", "fuel-cuda-backend", "fuel-vulkan-backend", "fuel-metal-backend",
            "fuel-aocl-cpu-backend", "fuel-mkl-cpu-backend", "fuel-kiss-ref-backend"}
MODELS = {"fuel-nn", "fuel-training", "fuel-posttrain", "fuel-datasets", "fuel-model-core", "fuel-model-llama",
          "fuel-model-phi", "fuel-transformers", "fuel-inference", "fuel-parallel"}
KERNELS = {"fuel-kernels-source", "fuel-vulkan-kernels", "fuel-metal-kernels", "fuel-cpu-kernels"}
SEAMS = {
    "S1 formats+interchange vs core": ({"fuel-formats", "fuel-loaders", "fuel-onnx"}, CORE),
    "S2 hardware vs dispatch+tensor": ({"fuel-hardware"}, {"fuel-dispatch", "fuel-tensor"}),
    "S3 executor (tensor+dispatch) vs backends": ({"fuel-tensor", "fuel-dispatch"}, BACKENDS),
    "S4 models+nn+training vs core": (MODELS, CORE),
    "S5 kernels vs backends": (KERNELS, BACKENDS),
}


def build_groups(members):
    return {k: ({x for x in left if x in members}, {x for x in right if x in members}) for k, (left, right) in SEAMS.items()}


def emit(title, lines):
    print(f"\n## {title}")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--sweep", type=int, default=12)
    a = ap.parse_args()
    require(REF_OK.match(a.ref) is not None, f"--ref {a.ref!r} is not a plain git ref")
    members, edges = load_workspace()
    tiers, rs = load_tiers(), tracked_rs(members)
    groups = build_groups(members)
    print("# structure report: tree =", git("rev-parse", "--short", "HEAD").strip(),
          "(the checked-out worktree; sources must equal the ref), co-change ref =", a.ref, git("rev-parse", "--short", a.ref).strip())
    emit("A. crate graph and tiers", section_graph(members, edges, tiers))
    emit("C. backend leaks in generic crates", section_leaks(members, tiers, rs))
    lines, commits = section_cochange(members, a.ref, a.window, a.sweep)
    emit("B. co-change", lines)
    emit("B2. per-seam co-change (sweeps excluded)", seam_cochange(commits, groups, a.sweep))
    emit("E. seam census", section_census(members, rs, groups, tiers))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""ROADMAP row 136: measure Fuel's structure before touching it.

Run from the repository root:  python scripts/structure_report.py [--window N] [--ref REF]

Every number is computed from `git ls-files` (never a disk walk), `cargo metadata --no-deps` and
`git log --first-parent REF`, and every definition is in the docstring of the function that computes
it, so another tool (CodeRipper's architecture and modularity rules) can be validated against the
same definitions. Positive controls are asserted at the end of each section.

Definitions, in one place:
  member        a package in the workspace (`cargo metadata --no-deps`).
  edge          member A depends on member B in A's manifest. kind = normal | dev. `optional` marks a
                normal edge that exists only behind a feature of A.
  tier          `fuel-ir/tests/crate_dependency_tiers.txt` (normative). A judged edge has a tier at
                both ends. A DIRECTION VIOLATION is a judged normal edge A -> B with tier(A) < tier(B).
  backend crate a member whose tier-file layer is `backends`, `vendor backends` or `reference backend`.
  kernel crate  layer `kernels`.
  generic crate every other member (neither backend nor kernel).
  backend leak  in a generic crate, in a `.rs` file under `src/` (tests/ counted separately): a line
                with `feature = "cuda"|"vulkan"|"metal"|"mkl"|"aocl"` inside a `cfg`, or the path
                `BackendId::X` or `DeviceLocation::{Cuda,Vulkan,Metal}`, or a source file or module
                directory whose name contains cuda, vulkan or metal. Lines are counted as written;
                `#[cfg(test)]` code inside `src/` is NOT separated (stated limitation).
  commit-crates the set of members with at least one changed file in a commit, ignoring files named
                Cargo.toml, Cargo.lock, *.md, and anything outside member directories (version-bump
                and docs noise). Commits are the first-parent commits of REF (one per squash-merged PR).
  sweep         a commit touching more than SWEEP crates (default 12); reported with and without.
  co-change     for crates A and B: c(A,B) commits touching both; J = c / (n(A) + n(B) - c).
  seam census   for a LEFT group L and a RIGHT group R: every path in L's `.rs` files that starts with
                the lib name of a crate in R (`fuel_x::...`). `use` statements are brace-expanded so
                each imported leaf is one site; other occurrences are one site each. A glob import
                (`use fuel_x::*` or `::*`) is counted separately and not as an item. An ITEM is the
                path after the crate name truncated after its first segment that starts with an
                uppercase letter, else truncated to 3 segments. Reported both ways (L uses R, and R
                uses L = back-edges at source level).
"""

import argparse
import collections
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def sh(*args, **kw):
    return subprocess.check_output(args, text=True, cwd=ROOT, encoding="utf-8", errors="replace", **kw)



def code_only(txt):
    """Source text without `//` comments (line and doc), so paths in prose are not counted. A `//`
    inside a string literal or after `:` (a URL) is kept; block comments are not handled (rare here)."""
    out = []
    for ln in txt.splitlines():
        if ln.lstrip().startswith("//"):
            continue
        i = ln.find("//")
        while i != -1:
            if (i == 0 or ln[i - 1] != ":") and ln[:i].count('"') % 2 == 0:
                ln = ln[:i]
                break
            i = ln.find("//", i + 2)
        out.append(ln)
    return chr(10).join(out)


# ----------------------------------------------------------------------------------------------- load
def load_workspace():
    meta = json.loads(sh("cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"))
    members = {}
    for p in meta["packages"]:
        mdir = Path(p["manifest_path"]).parent.resolve().relative_to(ROOT.resolve()).as_posix()
        members[p["name"]] = {"dir": mdir if mdir != "." else "", "deps": p["dependencies"]}
    edges = []
    for name, m in members.items():
        for d in m["deps"]:
            if d["name"] in members:
                edges.append((name, d["name"], d["kind"] or "normal", bool(d.get("optional"))))
    return members, edges


def load_tiers():
    tiers = {}
    for ln in (ROOT / "fuel-ir/tests/crate_dependency_tiers.txt").read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 3:
            tiers[parts[0]] = (int(parts[1]), parts[2])
    return tiers


def tracked_rs(members):
    """{crate: [paths]} of tracked .rs files, from the index, split by src/ vs elsewhere."""
    files = sh("git", "ls-files", "-z", "--", "*.rs").split("\0")
    by_crate = collections.defaultdict(lambda: {"src": [], "other": []})
    dirs = sorted(((m["dir"], n) for n, m in members.items() if m["dir"]), key=lambda t: -len(t[0]))
    for f in files:
        if not f:
            continue
        for d, n in dirs:
            if f.startswith(d + "/"):
                rel = f[len(d) + 1:]
                by_crate[n]["src" if rel.startswith("src/") else "other"].append(f)
                break
    return by_crate


# ----------------------------------------------------------------------------------------------- A
def section_graph(members, edges, tiers):
    out = []
    normal = [e for e in edges if e[2] == "normal"]
    dev = [e for e in edges if e[2] == "dev"]
    opt = [e for e in normal if e[3]]
    judged = [e for e in normal if e[0] in tiers and e[1] in tiers]
    viol = [e for e in judged if tiers[e[0]][0] < tiers[e[1]][0]]
    dev_viol = [e for e in dev if e[0] in tiers and e[1] in tiers and tiers[e[0]][0] < tiers[e[1]][0]]
    unplaced = sorted(n for n in members if n not in tiers)
    stale = sorted(n for n in tiers if n not in members)
    out.append(f"members {len(members)}; edges normal {len(normal)} (optional {len(opt)}), dev {len(dev)}")
    out.append(f"tier file places {len(tiers) - len(stale)} of {len(members)} members; unplaced: {', '.join(unplaced) or 'none'}")
    out.append(f"tier-file entries naming a non-member (stale): {', '.join(stale) or 'none'}")
    out.append(f"judged normal edges {len(judged)} of {len(normal)}; DIRECTION VIOLATIONS {len(viol)}: "
               + (", ".join(f"{a}->{b}" for a, b, *_ in viol) or "none"))
    out.append(f"dev-edge direction violations (not gated): {len(dev_viol)}: "
               + (", ".join(f"{a}->{b}" for a, b, *_ in dev_viol) or "none"))
    fin = collections.Counter(b for a, b, *_ in normal)
    fout = collections.Counter(a for a, b, *_ in normal)
    out.append("top fan-in (normal): " + ", ".join(f"{n} {c}" for n, c in fin.most_common(8)))
    out.append("top fan-out (normal): " + ", ".join(f"{n} {c}" for n, c in fout.most_common(8)))
    be = {n for n, (t, layer) in tiers.items() if layer in ("backends", "vendor backends", "reference backend")}
    opt_to_backend = sorted((a, b) for a, b, k, o in normal if o and b in be and a not in be)
    out.append("optional edges from a non-backend crate to a backend crate: " + (", ".join(f"{a}->{b}" for a, b in opt_to_backend) or "none"))
    assert ("fuel-tensor", "fuel-ir", "normal", False) in edges, "control: fuel-tensor -> fuel-ir must exist"
    out.append("control ok: fuel-tensor -> fuel-ir edge present")
    return out


# ----------------------------------------------------------------------------------------------- C
LEAK_CFG = re.compile(r'feature\s*=\s*"(cuda|vulkan|metal|mkl|aocl)"')
LEAK_ID = re.compile(r"\b(BackendId::(Cuda|Vulkan|Metal|Mkl|Aocl)|DeviceLocation::(Cuda|Vulkan|Metal))\b")
LEAK_NAME = re.compile(r"(cuda|vulkan|metal)", re.I)


def section_leaks(members, tiers, rs):
    out = []
    kinds = {n: tiers.get(n, (0, ""))[1] for n in members}
    backend = {n for n, k in kinds.items() if k in ("backends", "vendor backends", "reference backend")}
    kernel = {n for n, k in kinds.items() if k == "kernels"}
    generic = [n for n in members if n not in backend and n not in kernel]
    rows = []
    for n in sorted(generic):
        cfg_s = id_s = cfg_t = id_t = 0
        name_files = []
        for part in ("src", "other"):
            for f in rs[n][part]:
                txt = code_only((ROOT / f).read_text(encoding="utf-8", errors="replace"))
                cfg = len(LEAK_CFG.findall(txt))
                ids = len(LEAK_ID.findall(txt))
                if part == "src":
                    cfg_s += cfg
                    id_s += ids
                    if LEAK_NAME.search(Path(f).name) or any(LEAK_NAME.search(p) for p in Path(f).parts[:-1] if p not in ("tests",)):
                        name_files.append(f)
                else:
                    cfg_t += cfg
                    id_t += ids
        if cfg_s or id_s or cfg_t or id_t or name_files:
            rows.append((n, cfg_s, id_s, len(name_files), cfg_t, id_t))
    out.append("generic crate | src cfg(feature=backend) | src BackendId/DeviceLocation::X | src files/dirs named for a backend | tests cfg | tests ids")
    for r in sorted(rows, key=lambda r: -(r[1] + r[2])):
        out.append(" | ".join(map(str, r)))
    for crate in ("fuel-dispatch", "fuel-tensor"):
        per = []
        for f in rs[crate]["src"]:
            t = code_only((ROOT / f).read_text(encoding="utf-8", errors="replace"))
            per.append((len(LEAK_CFG.findall(t)) + len(LEAK_ID.findall(t)), f))
        per.sort(reverse=True)
        out.append(f"   top {crate} src files by backend-path+cfg lines: " + "; ".join(f"{f.split('/', 1)[1]} {n}" for n, f in per[:5] if n))
    tot_cfg = sum(r[1] for r in rows)
    tot_id = sum(r[2] for r in rows)
    out.append(f"TOTAL in generic crates' src: {tot_cfg} cfg lines, {tot_id} backend-path uses, across {sum(1 for r in rows if r[1] or r[2])} crates")
    # positive control: a backend crate must match its own name patterns when scanned the same way
    ctl = sum(len(LEAK_ID.findall(code_only((ROOT / f).read_text(encoding="utf-8", errors="replace"))))
              for f in rs["fuel-dispatch"]["src"])
    assert tot_cfg + tot_id > 0, "control: the scan must find the known fuel-tensor backend modules"
    out.append(f"control ok: scan is non-empty (fuel-dispatch src alone has {ctl} backend-path uses)")
    return out


# ----------------------------------------------------------------------------------------------- B
def section_cochange(members, ref, window, sweep):
    out = []
    dirs = sorted(((m["dir"], n) for n, m in members.items() if m["dir"]), key=lambda t: -len(t[0]))
    raw = sh("git", "log", ref, "--first-parent", f"-n{window}", "--name-only", "--format=@@%H %cs").split("\n")
    commits = []
    cur = None
    for ln in raw:
        if ln.startswith("@@"):
            cur = {"sha": ln[2:10], "date": ln.split()[1], "crates": set()}
            commits.append(cur)
        elif ln.strip() and cur is not None:
            base = ln.rsplit("/", 1)[-1]
            if base in ("Cargo.toml", "Cargo.lock") or base.endswith(".md"):
                continue
            for d, n in dirs:
                if ln.startswith(d + "/"):
                    cur["crates"].add(n)
                    break
    commits = [c for c in commits if c["crates"]]
    out.append(f"window: {len(commits)} first-parent commits touching at least one member's code, {commits[-1]['date']} to {commits[0]['date']} (ref {ref}, first {window} commits)")
    sweeps = [c for c in commits if len(c["crates"]) > sweep]
    out.append(f"sweeps (> {sweep} crates in one commit): {len(sweeps)}; shown with and without")

    def matrix(cs):
        n = collections.Counter()
        c = collections.Counter()
        for x in cs:
            for a in x["crates"]:
                n[a] += 1
            for a, b in itertools.combinations(sorted(x["crates"]), 2):
                c[(a, b)] += 1
        return n, c

    for label, cs in (("ALL commits", commits), (f"WITHOUT sweeps", [c for c in commits if len(c["crates"]) <= sweep])):
        n, c = matrix(cs)
        out.append(f"-- {label}: {len(cs)} commits; most-changed crates: " + ", ".join(f"{k} {v}" for k, v in n.most_common(10)))
        pairs = [(a, b, v, v / (n[a] + n[b] - v)) for (a, b), v in c.items() if v >= 5]
        pairs.sort(key=lambda t: -t[3])
        out.append("   pairs with c>=5 by Jaccard (A, B, c, J): " + "; ".join(f"{a}+{b} {v} {j:.2f}" for a, b, v, j in pairs[:12]))
    return out, commits


def seam_cochange(commits, groups, sweep):
    out = []
    use = [c for c in commits if len(c["crates"]) <= sweep]
    for name, (L, R) in groups.items():
        l = sum(1 for c in use if c["crates"] & L and not c["crates"] & R)
        r = sum(1 for c in use if c["crates"] & R and not c["crates"] & L)
        both = sum(1 for c in use if c["crates"] & L and c["crates"] & R)
        tot = l + r + both
        out.append(f"{name}: left-only {l}, right-only {r}, both {both} (of {tot}); share of left commits that also touch right = "
                   + (f"{both / (l + both):.0%}" if l + both else "n/a"))
    return out


# ----------------------------------------------------------------------------------------------- E
USE_STMT = re.compile(r"\buse\s+(fuel_[a-z0-9_]+(?:::[^;]*)?);", re.S)
PATH_TOK = re.compile(r"\b(fuel_[a-z0-9_]+)((?:::[A-Za-z_][A-Za-z0-9_]*)+)")


def expand(path):
    """Brace-expand a use path into leaf paths; '*' leaves stay as a trailing '::*'."""
    path = re.sub(r"\s+", "", path)
    m = re.search(r"\{", path)
    if not m:
        return [path]
    depth, i = 0, m.start()
    parts, start = [], m.start() + 1
    for j in range(m.start(), len(path)):
        ch = path[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parts.append(path[start:j])
                end = j
                break
        elif ch == "," and depth == 1:
            parts.append(path[start:j])
            start = j + 1
    prefix, suffix = path[:m.start()], path[end + 1:]
    res = []
    for p in parts:
        if p:
            res.extend(expand(prefix + p + suffix))
    return res


def item_of(segs):
    out = []
    for s in segs[:3]:
        out.append(s)
        if s[:1].isupper():
            break
    return "::".join(out)


def census(files, target_libs):
    """files: list of (referring crate, path). Returns sites[(referrer, lib, item)], globs, files-per-key."""
    sites = collections.Counter()
    globs = collections.Counter()
    nfiles = collections.defaultdict(set)
    for crate, f in files:
        txt = code_only((ROOT / f).read_text(encoding="utf-8", errors="replace"))
        txt_nouse = USE_STMT.sub(" ", txt)
        refs = []
        for m in USE_STMT.finditer(txt):
            for leaf in expand(m.group(1)):
                refs.append([s for s in leaf.split("::") if s])
        for m in PATH_TOK.finditer(txt_nouse):
            refs.append([m.group(1)] + [s for s in m.group(2).split("::") if s])
        for segs in refs:
            lib = segs[0]
            if lib not in target_libs:
                continue
            if segs[-1] in ("*", "self") and len(segs) <= 2:
                globs[(crate, lib)] += 1
                continue
            if segs[-1] == "*":
                globs[(crate, lib)] += 1
                segs = segs[:-1]
            key = (crate, lib, item_of(segs[1:]) if len(segs) > 1 else "(crate)")
            sites[key] += 1
            nfiles[key].add(f)
    return sites, globs, nfiles


def lib_name(crate):
    return crate.replace("-", "_")


def section_census(members, rs, groups, tiers):
    out = []
    lib2crate = {lib_name(n): n for n in members}

    def up(sites):
        """Sites where the referring crate has a LOWER tier than the crate it names (a source-level
        back-edge by the tier file's own order). Unplaced crates are not judged."""
        n = 0
        for (ref, lib, _), c in sites.items():
            t1, t2 = tiers.get(ref), tiers.get(lib2crate.get(lib, ""))
            if t1 and t2 and t1[0] < t2[0]:
                n += c
        return n

    for name, (L, R) in groups.items():
        def files_of(group):
            return [(n, f) for n in sorted(group) for part in ("src", "other") for f in rs[n][part]]
        s1, g1, nf1 = census(files_of(L), {lib_name(n) for n in R})
        s2, g2, nf2 = census(files_of(R), {lib_name(n) for n in L})
        files_l = {f for fs in nf1.values() for f in fs}

        def trange(g):
            ts = sorted({tiers[n][0] for n in g if n in tiers})
            return f"tiers {ts[0]}-{ts[-1]}" if ts else "no tier"

        d1 = collections.Counter()
        for (_, lib, it), c in s1.items():
            d1[(lib, it)] += c
        d2 = collections.Counter()
        for (_, lib, it), c in s2.items():
            d2[(lib, it)] += c
        out.append(f"{name} [LEFT {trange(L)}; RIGHT {trange(R)}]: LEFT->RIGHT {sum(s1.values())} sites ({up(s1)} up-tier), {len(d1)} distinct items, {sum(g1.values())} globs, in {len(files_l)} files"
                   f"; RIGHT->LEFT {sum(s2.values())} sites ({up(s2)} up-tier), {len(d2)} distinct items, {sum(g2.values())} globs")
        out.append("   top items LEFT->RIGHT: " + ", ".join(f"{lib}::{it} {c}" for (lib, it), c in d1.most_common(6)))
        if d2:
            out.append("   top items RIGHT->LEFT: " + ", ".join(f"{lib}::{it} {c}" for (lib, it), c in d2.most_common(6)))
        ups = [(ref, lib, it, c) for (ref, lib, it), c in list(s1.items()) + list(s2.items())
               if tiers.get(ref) and tiers.get(lib2crate.get(lib, "")) and tiers[ref][0] < tiers[lib2crate[lib]][0]]
        if ups:
            out.append("   up-tier sites by (referrer -> crate): " + ", ".join(
                f"{r}->{l.replace('_', '-')} {c}" for (r, l), c in collections.Counter(
                    {(r, l): sum(x[3] for x in ups if x[0] == r and x[1] == l) for r, l, _, _ in ups}).most_common(8)))
    s, _, _ = census([("fuel-tensor", f) for f in rs["fuel-tensor"]["src"]], {"fuel_ir"})
    assert sum(s.values()) > 0, "control: fuel-tensor must reference fuel_ir"
    out.append(f"control ok: fuel-tensor src references fuel_ir at {sum(s.values())} sites")
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--sweep", type=int, default=12)
    a = ap.parse_args()
    members, edges = load_workspace()
    tiers = load_tiers()
    rs = tracked_rs(members)
    core = {"fuel-ir", "fuel-graph", "fuel-memory", "fuel-dispatch", "fuel-tensor", "fuel-hardware", "fuel-backend-contract"}
    backends = {"fuel-cpu-backend", "fuel-cuda-backend", "fuel-vulkan-backend", "fuel-metal-backend",
                "fuel-aocl-cpu-backend", "fuel-mkl-cpu-backend", "fuel-kiss-ref-backend"}
    groups = {
        "S1 formats+interchange vs core": ({"fuel-formats", "fuel-loaders", "fuel-onnx"}, core),
        "S2 hardware vs dispatch+tensor": ({"fuel-hardware"}, {"fuel-dispatch", "fuel-tensor"}),
        "S3 executor (tensor+dispatch) vs backends": ({"fuel-tensor", "fuel-dispatch"}, backends),
        "S4 models+nn+training vs core": ({"fuel-nn", "fuel-training", "fuel-posttrain", "fuel-datasets", "fuel-model-core",
                                           "fuel-model-llama", "fuel-model-phi", "fuel-transformers", "fuel-inference", "fuel-parallel"}, core),
        "S5 kernels vs backends": ({"fuel-kernels-source", "fuel-vulkan-kernels", "fuel-metal-kernels", "fuel-cpu-kernels"}, backends),
    }
    present = {m for m in members}
    groups = {k: ({x for x in L if x in present}, {x for x in R if x in present}) for k, (L, R) in groups.items()}
    print("# structure report: tree =", sh("git", "rev-parse", "--short", "HEAD").strip(), "(the checked-out worktree; sources must equal the ref), co-change ref =", a.ref, sh("git", "rev-parse", "--short", a.ref).strip())
    print("\n## A. crate graph and tiers"); print("\n".join(section_graph(members, edges, tiers)))
    print("\n## C. backend leaks in generic crates"); print("\n".join(section_leaks(members, tiers, rs)))
    lines, commits = section_cochange(members, a.ref, a.window, a.sweep)
    print("\n## B. co-change"); print("\n".join(lines))
    print("\n## B2. per-seam co-change (sweeps excluded)"); print("\n".join(seam_cochange(commits, groups, a.sweep)))
    print("\n## E. seam census"); print("\n".join(section_census(members, rs, groups, tiers)))


if __name__ == "__main__":
    sys.exit(main())

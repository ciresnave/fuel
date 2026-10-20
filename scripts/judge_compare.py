#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare two Judge reports: winner agreement per (op, dtype, size_class) cell.

usage: python scripts/judge_compare.py base.json variant.json
Winner = the entry with the lowest recorded latency in the cell (correctness
verdicts are not consulted). Prints cell count, winner agreement, up to 20
disagreements, and a per-entry latency drift summary.
"""
import json
import sys


def load(p):
    d = json.load(open(p))
    return d["entries"] if isinstance(d, dict) and "entries" in d else d


def key(e):
    return (str(e.get("op")), str(e.get("dtype")), json.dumps(e.get("size_class"), sort_keys=True))


def label(e):
    return (str(e.get("backend")), e.get("device_index"), str(e.get("kernel_source")))


def winners(entries):
    cells = {}
    for e in entries:
        cells.setdefault(key(e), []).append(e)
    out = {}
    for k, es in cells.items():
        best = min(es, key=lambda e: e.get("latency_ns", 1 << 62))
        out[k] = (label(best), best.get("latency_ns"), len(es))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
wa, wb = winners(a), winners(b)
common = sorted(set(wa) & set(wb))
dis = [k for k in common if wa[k][0] != wb[k][0]]
print(f"entries: base {len(a)} variant {len(b)}; cells: base {len(wa)} variant {len(wb)} common {len(common)}")
print(f"winner agreement: {len(common) - len(dis)}/{len(common)}")
for k in dis[:20]:
    print("  DISAGREE", k, "base", wa[k][0], wa[k][1], "variant", wb[k][0], wb[k][1])
drift = []
ia = {(key(e), label(e)): e.get("latency_ns") for e in a}
ib = {(key(e), label(e)): e.get("latency_ns") for e in b}
for k in set(ia) & set(ib):
    if ia[k] and ib[k]:
        drift.append(abs(ib[k] - ia[k]) / ia[k])
drift.sort()
if drift:
    print(f"per-entry latency |variant-base|/base over {len(drift)} entries: median {drift[len(drift)//2]:.3f} p90 {drift[int(len(drift)*0.9)]:.3f} max {drift[-1]:.3f}")

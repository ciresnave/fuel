#!/usr/bin/env python3
"""Doc-name census: backticked CamelCase names in docs/architecture/ vs *.rs.

WHY THIS EXISTS
    The GAP-302 doc-name drift census was measured with an instrument that did
    not live in the repository, so nobody but its author could re-run it and it
    could not be a gate's foundation. That absence WAS the finding. This is the
    instrument, in-repo, re-runnable by anyone.

WHAT IT IS AND IS NOT
    It is a PRINTER, not a count-asserting born-red. A gate born on a population
    that contains known false positives would enforce them. Instead every absent
    name carries a REASONED DISPOSITION (below); the count falls out of the
    dispositions, it is not an input to them.

    The one invariant worth gating is NOT the count but "every absent name has a
    disposition" -- i.e. a NEWLY-absent backticked name (real drift someone just
    introduced) is UNCLASSIFIED and should be looked at. Run with --gate to exit
    non-zero on any unclassified absence; the default prints and exits 0.

EXTRACTOR RULES (stated so the census is auditable)
    - source: inline `backtick` spans in docs/architecture/*.md
    - fenced ``` blocks are stripped first (they are code samples, not prose refs)
    - each span is split on non-identifier chars, and EVERY segment is tested --
      so a qualified form like `Concurrency::Auto` tests BOTH `Concurrency` (the
      head) AND `Auto` (the tail). This matters: the earlier out-of-repo tool
      took only the TAIL, so a type appearing solely in qualified form (e.g.
      `Concurrency`, which occurs only as `Concurrency::{Auto,Required,Forbidden}`)
      was invisible to it -- which is why the published "25 absent" UNDERCOUNTS
      (it missed `Concurrency` and `ErrorBound`; this tool finds them).
    - a segment counts as a doc name iff it matches ^[A-Z][A-Za-z0-9]*$ AND has
      >=1 lowercase AND len >= 2 (the lowercase rule drops all-caps prose words
      like MINOR, MAJOR, PATCH)
    - PRESENT iff the token appears as a CamelCase word-token in any *.rs under the
      repo (the `target/` build dir and `.git/` are excluded)

    This differs from the earlier out-of-repo tool two ways, which is exactly why
    its published "25 absent" figure and this tool's 26 differ (settled by a
    membership diff, not a total): (1) this tool splits qualified forms and tests
    head AND tail, so it CATCHES `Concurrency`/`ErrorBound` (which occur only as
    `Type::Variant`) -- the earlier tool took only the tail and UNDERCOUNTS by
    those two; (2) the >=1-lowercase rule drops `MINOR`, a semver level in prose
    the earlier tool OVER-counts as a type. So published-25 = this-26 minus
    {Concurrency, ErrorBound} plus {MINOR}; it closes with no third term.

CONTROLS
    Every run reasserts two positive controls -- NodeHandle and FusedOpRegistry
    must both be present in *.rs. A run where a control is missing means the *.rs
    scan itself is broken and the ABSENT set is meaningless (a malformed query
    returning a plausible population is this project's signature failure).

Usage:
    python scripts/doc_name_census.py             # print the census + dispositions
    python scripts/doc_name_census.py --gate      # additionally exit 1 on any
                                                  # UNCLASSIFIED absent name
    python scripts/doc_name_census.py --self-test # two-arm proof the fixture
                                                  # exclusion is load-bearing
"""

import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Disposition table: one reasoned entry per name currently absent from *.rs.
# Six classes. The count is derived from this table, never asserted against it.
#
#   EXTERNAL   an external library / API type; allowlisted WITH REASON
#   DRIFT      a real Fuel type under an abbreviated/stale doc spelling; the
#              `code` field names what it should become (a doc rename)
#   PROPOSED   an UNBUILT future concept; stays absent, anchored to a roadmap
#              item. NEVER allowlisted -- an allowlist entry for a thing that
#              will exist later is an expiring decline with no detector.
#   CITED      the sentence's POINT is the absence (rejected / removed /
#              never-existed correction), asserted at ONE site; leave it, it
#              will never map to code
#   FALSEHOOD  falsehood-with-trailing-correction: the name is asserted to
#              EXIST at one site and corrected at a LATER one. NOT cited (that
#              class has one site whose point IS the absence). Remedy: an inline
#              supersession marker AT the stale passage, never only a later fix.
#   CONCEPT    a concept, not a type; the doc should un-backtick it
#
# CITED vs FALSEHOOD is decided by ENUMERATING EVERY occurrence, not the first:
# one site is not the population (Fmin was mis-read as CITED from a single site
# while a 1468-char line asserted the opposite at char ~1350).
#
# Every disposition is stamped against origin/main 56bf08bb. A `code` count is a
# BOUND-shaped fact (">=1 in *.rs"), re-measured live by this script, not a pin.
# ---------------------------------------------------------------------------
DISPOSITIONS = {
    # -- EXTERNAL TYPE ------------------------------------------------------
    "ArrowDeviceArray": ("EXTERNAL", "Apache Arrow C-data-interface struct; 13-interchange backlog leaf I/O format. Never Fuel code."),
    "MemGetInfo":       ("EXTERNAL", "CUDA driver API (cuMemGetInfo); 05-backend-contract. Never a Fuel type."),

    # -- REAL DRIFT (doc rename proposed; code name in `->`) ---------------

    # -- PROPOSED-FUTURE (unbuilt; roadmap anchor named) -------------------
    "Concurrency":            ("PROPOSED", "concurrent-execute realize knob enum {Auto,Required,Forbidden}; 04-optimization. Unbuilt (RuleFamily != this)."),
    "Forbidden":              ("PROPOSED", "variant of the unbuilt Concurrency enum; 04-optimization."),
    "WholeGraph":             ("PROPOSED", "frontier-compat class (Concurrent|WholeGraph); 04-optimization. Unbuilt; 04 present-tense framing is a separate doc-defect."),
    "FusionMissRecord":       ("PROPOSED", "G5 missing-fusion telemetry, closed-world; 08-pattern-harvest. decisions-log G5: no signal today."),
    "SequenceRecord":         ("PROPOSED", "G5 missing-fusion telemetry, open-world 'deferred'; 08-pattern-harvest:27."),
    "NoBackendKernel":        ("PROPOSED", "a reason value on the unbuilt FusionMissRecord (G5); 05-backend-contract:94."),
    "ErrorBound":             ("PROPOSED", "tolerance v1 annotation type; 07-tolerance:55 'v1: best-effort annotations'. Unbuilt."),
    "Aggressive":             ("PROPOSED", "Tolerance:: profile, 'reasonable extensions'; 07-tolerance:23. Unbuilt."),
    "Mild":                   ("PROPOSED", "Tolerance:: profile, 'reasonable extensions'; 07-tolerance:23. Unbuilt."),
    "AddInplace":             ("PROPOSED", "optimizer in-place op, 'Optional in v1'; 05-backend-contract:590. Unbuilt (unary in-place family exists; binary add/sub does not)."),
    "SubInplace":             ("PROPOSED", "optimizer in-place op, 'Optional in v1'; 05-backend-contract:590. Unbuilt."),
    "MemoryPressureSelector": ("PROPOSED", "planned RuntimeSelector sibling; 05-backend-contract:625. Unbuilt (RuntimeSelector + VramPressureSelector exist)."),
    "LoadAwareSelector":      ("PROPOSED", "explicitly 'future' RuntimeSelector; 05-backend-contract:625. Unbuilt."),
    "GraphInvoker":           ("PROPOSED", "'the missing piece' under decisions-log header 'What does NOT exist'; :1544. Unbuilt."),
    "RuntimeHook":            ("PROPOSED", "'the unbuilt Phase-9 RuntimeHook'; decisions-log:956."),

    # -- CITED-AS-ABSENT (leave; will never map to code) -------------------
    "CostRegistry":     ("CITED", "'a sibling CostRegistry ... rejected'; decisions-log:450 (sole occurrence)."),
    "NodeKind":         ("CITED", "'no separate NodeKind discriminator'; 5 sites (03-ir:3,:38; 04-optimization:3; decisions-log:83 x2), all absence/rejected-option."),
    "OpEntry":          ("CITED", "sole occurrence (11-persistence:80) IS the correction-at-the-site: 'this said OpEntry, which does not exist ... FusedOpEntry'. Worked example of the FALSEHOOD remedy done right."),
    "ReferenceFactory": ("CITED", "'ReferenceFactory is removed'; 3 sites (05-backend-contract:3,:163,:185) all assert the TYPE removed. (:185's flagged self-contradiction is about the CRATE, not this type.)"),

    # -- FALSEHOOD-WITH-TRAILING-CORRECTION (class 7; needs inline supersession) --
    "Fmin":             ("FALSEHOOD", "was asserted to EXIST at decisions-log:493 ('remains available ... unchanged') and corrected at :499. Resolved by an inline supersession marker at :493 (the stale passage), so a reader who never reaches :499 cannot believe the false clause."),

    # -- CONCEPT-NOT-TYPE (doc should un-backtick) -------------------------
}

CLASS_ORDER = ["DRIFT", "FALSEHOOD", "PROPOSED", "CITED", "CONCEPT", "EXTERNAL", "UNCLASSIFIED"]

CONTROLS = ["NodeHandle", "FusedOpRegistry"]

# ---------------------------------------------------------------------------
def is_camel(tok):
    return (
        len(tok) >= 2
        and re.match(r"^[A-Z][A-Za-z0-9]*$", tok) is not None
        and any(c.islower() for c in tok)
    )


def doc_names():
    """{name: first (file, line) it was seen at} over docs/architecture/*.md."""
    seen = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "docs", "architecture", "*.md"))):
        base = os.path.basename(path)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        # strip fenced code, preserving line numbers so citations stay honest
        stripped = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), raw, flags=re.DOTALL)
        for lineno, line in enumerate(stripped.split("\n"), 1):
            for span in re.findall(r"`([^`]+)`", line):
                for tok in re.split(r"[^A-Za-z0-9_]+", span):
                    if is_camel(tok):
                        seen.setdefault(tok, (base, lineno))
    return seen


def _rust_paths():
    """Every `*.rs` path under ROOT, with `target/` and `.git/` pruned."""
    for dirpath, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != "target" and d != ".git"]
        for fn in files:
            if fn.endswith(".rs"):
                yield os.path.join(dirpath, fn)


def _read_text(path):
    """The file's text, or None when it cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return None


def code_tokens():
    """Every CamelCase word-token appearing in any *.rs (target/ and .git/ excluded).

    NO FILE IS EXCLUDED, and that is the point. A fixture exclusion used to live
    here because `fuel-ir/tests/doc_block_scope.rs` named the drifted
    identifiers verbatim as test fixtures -- the census went from ~25 absent to
    0 the moment #147 merged. That gate now keeps its population in a NON-RUST
    data file, so there is nothing in `*.rs` to exclude. Dissolving the coupling
    beats maintaining an exclusion list: no filename to rot, and it works for
    tools that do not exist yet.
    """
    toks = set()
    for path in _rust_paths():
        txt = _read_text(path)
        if txt is None:
            continue
        toks.update(t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", txt) if is_camel(t))
    return toks


def absent_count():
    docs = set(doc_names())
    code = code_tokens()
    return sum(1 for n in docs if n not in code)


def self_test():
    """Two arms, both required; either alone passes on a broken tool.

    ARM A (population integrity). RANGES OVER `DISPOSITIONS`, NEVER OVER THE
      DERIVED `absent` SET -- `absent` IS `docs - code`, so "no absent name is
      in code" is a TAUTOLOGY that passes on any tool, including one that reads
      zero files. `DISPOSITIONS` is a claim about the world: each entry was
      adjudicated ABSENT at a point in time, and this asks whether that still
      holds.

      It REPLACES a fixture-exclusion arm that asserted the exclusion was
      load-bearing. That arm was load-bearing only WHILE a fixture was
      poisoning the corpus -- its evidence WAS the defect -- so fixing the
      defect would have made it vacuous with nothing saying so. A gate cannot
      source its negative case from the thing it exists to catch.

      TWO EVENTS, ONE PREDICATE, AND ONLY ONE IS A DEFECT:
        `n in code`         the name acquired a referent -- it got BUILT.
                            FAILS, and the message says RECLASSIFY, not drift.
        `n not in docs`     the doc mention was renamed or un-backticked --
                            the fix landing. PRINTS, never fails. An arm that
                            reddens when the fix lands teaches suppression.

    ARM B (still-sees-corpus). UNCHANGED, and it is the arm that catches the
      NEXT total-poisoning fixture: a new file naming the whole population
      drives `absent` to 0 and reds here. Keep both.
    """
    docs, code = set(doc_names()), code_tokens()
    reappeared = sorted(n for n in DISPOSITIONS if n in code)
    left_docs = sorted(n for n in DISPOSITIONS if n not in docs and n not in code)
    absent = absent_count()

    print("SELF-TEST @ %s" % git_ref())
    print("  DISPOSITIONS entries              : %d" % len(DISPOSITIONS))
    print("  entries that now RESOLVE in *.rs  : %d" % len(reappeared))
    print("  entries whose DOC mention is gone : %d  (benign: renamed/un-backticked)"
          % len(left_docs))
    print("  ABSENT population                 : %d" % absent)
    for n in left_docs:
        print("      %s -- doc mention removed; retire the table entry" % n)
    for n in reappeared:
        cls = DISPOSITIONS[n][0]
        print("      %s [%s] NOW RESOLVES in *.rs -- re-examine its disposition." % (n, cls))
        print("          A PROPOSED-FUTURE name resolving means it was BUILT. That is"
              " GOOD NEWS and the signal to move this row out of PROPOSED --"
              " it is not drift and must not be suppressed.")

    arm_a = not reappeared
    arm_b = absent > 0
    print("  ARM A (no disposition entry resolves in *.rs) : %s" % ("PASS" if arm_a else "FAIL"))
    print("  ARM B (absent > 0, tool still sees corpus)    : %s" % ("PASS" if arm_b else "FAIL"))
    ok = arm_a and arm_b
    print("  => %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def git_ref():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main():
    if "--self-test" in sys.argv[1:]:
        return self_test()

    gate = "--gate" in sys.argv[1:]

    docs = doc_names()
    code = code_tokens()

    # controls first: a broken *.rs scan makes the whole census meaningless
    control_ok = all(c in code for c in CONTROLS)

    absent = sorted(n for n in docs if n not in code)
    present_ct = len(docs) - len(absent)

    ref = git_ref()
    print("doc-name census @ %s  (working tree)" % ref)
    print("  distinct doc names : %d" % len(docs))
    print("  present in *.rs    : %d" % present_ct)
    print("  ABSENT             : %d" % len(absent))
    print("  controls           : %s" % (
        "OK (%s)" % ", ".join(CONTROLS) if control_ok
        else "BROKEN -- %s missing from *.rs; ABSENT set is UNRELIABLE" % (
            ", ".join(c for c in CONTROLS if c not in code))))
    print()

    # group absent names by disposition class
    buckets = {cls: [] for cls in CLASS_ORDER}
    for name in absent:
        cls, reason = DISPOSITIONS.get(name, ("UNCLASSIFIED", "NO DISPOSITION -- newly-absent backticked name; classify it."))
        buckets[cls].append((name, reason))

    for cls in CLASS_ORDER:
        rows = buckets[cls]
        if not rows:
            continue
        print("[%s] (%d)" % (cls, len(rows)))
        for name, reason in rows:
            src = docs.get(name, ("?", 0))
            print("  %-24s %s:%s  %s" % (name, src[0], src[1], reason))
        print()

    # names in the table that are no longer absent -> disposition satisfied
    resolved = sorted(n for n in DISPOSITIONS if n not in docs or n in code)
    if resolved:
        print("[RESOLVED SINCE DISPOSITION] (%d) -- no longer absent; table entry now moot:" % len(resolved))
        for name in resolved:
            print("  %s" % name)
        print()

    unclassified = buckets["UNCLASSIFIED"]
    if unclassified:
        print("=> %d UNCLASSIFIED absent name(s). Each is either new drift or a new concept; disposition required." % len(unclassified))

    if not control_ok:
        # a broken scan is always a hard failure, gate or not
        print("=> CONTROLS BROKEN: refusing to certify this census.")
        return 2
    if gate and unclassified:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

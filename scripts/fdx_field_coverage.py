#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""FDX wire-struct fields MINUS the fields `validate()` reads.

GAP-286. A field that appears on the wire and is read by no validator arm is a
field a producer may emit anything into. `lanes` and `sub_byte_bit_order` were
each found by hand, one at a time; this is the instrument that finds the rest,
so the population stops being "the ones we happened to look at".

WHY NOT GREP THE FIELD NAMES
----------------------------
A field can be reached by destructuring, through a helper, or by matching on the
parent struct -- none of which contain the string `parent.field`. And a bare name
is ambiguous ACROSS structs: `lanes` is a field of BOTH `FDXDTypeExt` and
DLPack's `DLDataType`, and a bare `lanes` grep in `validate.rs` returns 2 hits
that are the WRONG struct's. The qualified `dtype_ext.lanes` returns 0, which is
the true answer -- but only because someone knew to qualify it.

THE METHOD: ASK THE COMPILER
----------------------------
Rename every field of ONE struct in its DEFINITION, compile, and read rustc's
`E0609 no field 'X' on type 'Y'` diagnostics. The compiler resolves types, so
the answer survives every alias a text search cannot follow. A field named in a
diagnostic whose span is `validate.rs` IS read by the validator.

One `cargo check` per struct; ~3s each warm, ~1 minute for the whole sweep.

WHAT IT DOES NOT DO
-------------------
It does not add validator arms and must not be read as proposing any. A
validator arm is a behaviour change on a PUBLISHED wire format and needs its own
born-red plus a per-field ruling. This ENUMERATES.

CLASSIFYING THE RESULT -- and the trap to avoid
-----------------------------------------------
A field the spec declares IGNORABLE is CONFORMANCE, not a hole. GAP-286's own
history is the worked example: its `HAS_TILING`-is-a-defect claim was RETRACTED
because §6.5 declares tiling an optional hint, and "no enforcement" and "no
obligation to enforce" have an identical code signature. `_pad*` and `reserved`
are the same shape by construction. READ THE SPEC PER FIELD before calling
anything a hole, and report the spec-exempt bucket separately.

USAGE
    python scripts/fdx_field_coverage.py             # the sweep
    python scripts/fdx_field_coverage.py --self-test # calibrate the instrument
"""

import io
import json
import os
import re
import shutil
import subprocess  # nosec B404 - see SUPPRESSION NOTE below
import sys

SUFFIX = "_FDXPROBE"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR = "fuel-ir/src/dlpack/validate.rs"
SOURCES = ["fuel-ir/src/dlpack/sidecar.rs", "fuel-ir/src/dlpack/abi.rs"]

# DLPack's own ABI structs. Their unread fields are a DLPack-conformance
# question, not FDX validator coverage, and folding them into one number would
# answer neither.
DLPACK_ABI = {
    "DLDevice",
    "DLDataType",
    "DLTensor",
    "DLPackVersion",
    "DLManagedTensorVersioned",
}


def structs_in(src):
    return [m.group(1) for m in re.finditer(r"pub struct (\w+) \{", src)]


def fields_of(src, struct):
    m = re.search(r"pub struct %s \{(.*?)\n\}" % re.escape(struct), src, re.S)
    return re.findall(r"^\s*pub (\w+):", m.group(1), re.M) if m else []


def _rename(src, struct, on):
    m = re.search(r"(pub struct %s \{)(.*?)(\n\})" % re.escape(struct), src, re.S)
    head, body, tail = m.group(1), m.group(2), m.group(3)
    if on:
        body = re.sub(r"^(\s*pub )(\w+)(:)", r"\1\2" + SUFFIX + r"\3", body, flags=re.M)
    return src[: m.start()] + head + body + tail + src[m.end() :]


def probe(rel_path, struct):
    """{field: {files that read it}}, via the compiler. Always restores."""
    # ⚠️ BINARY I/O, AND THIS IS NOT PEDANTRY. Text-mode round-tripping
    # NORMALISES LINE ENDINGS: on a CRLF checkout the restore returns identical
    # CONTENT while leaving every probed file "modified", and a careless commit
    # then sweeps in a whole-file ending change. Restoring the BYTES is what
    # makes this probe safe to run on a tree someone else is working in.
    path = os.path.join(ROOT, rel_path)
    raw = io.open(path, "rb").read()
    io.open(path, "wb").write(
        _rename(raw.decode("utf-8"), struct, True).encode("utf-8")
    )
    try:
        return _e0609_hits(_cargo_check_json(), struct)
    finally:
        io.open(path, "wb").write(raw)
        # ⚠️ NOT `assert`. Under `python -O` an assertion is COMPILED AWAY, so
        # the guard that catches a bad restore would silently not exist -- and
        # the failure it guards against (content restored, bytes not) is exactly
        # the one that leaves no trace. A guard a compiler flag can delete is a
        # guard with an off switch nobody can see in the source.
        if io.open(path, "rb").read() != raw:
            raise RuntimeError(
                "RESTORE FAILED -- %s differs from the original bytes" % rel_path
            )


def _cargo_check_json():
    """rustc's JSON diagnostics for the perturbed tree.

    SUPPRESSION NOTE -- `# nosec B404, B603`, ruled by the Fuel architect
    2026-09-09 rather than decided by whoever wanted the gate green.

    1. WHY IT IS INHERENT. The method IS invoking the compiler. `cargo check`'s
       `E0609 no field 'X' on type 'Y'` output is the oracle, and the whole
       reason this tool exists is that no text search answers the question --
       a field reached by destructuring, a helper, or a match on the parent
       contains no `parent.field` string, and a bare field name is ambiguous
       across structs. There is no filesystem answer to read instead, so the
       remedy that cleared these lints elsewhere in this repo -- dropping
       subprocess and reading a file -- has no analogue here.

    2. WHAT THE INPUT ACTUALLY IS, since B603 is about untrusted input. The
       argv is a fixed literal list. Nothing from outside the repository
       reaches it. The only value this tool ever interpolates anywhere is a
       FIELD NAME it read out of Fuel's own source moments earlier, and that
       goes into a regex, never into a command line.

    3. ⚠️ WHAT WOULD HAVE TO CHANGE FOR THIS TO COME OUT, which is the part
       that makes it a suppression and not an exemption: if the
       "is this field ever read" question ever becomes answerable WITHOUT
       compiling -- a rustc lint, an analysis API, a MIR dump -- then the
       subprocess is avoidable and this note is stale and must go. A
       suppression that records no expiry condition is permanently
       unfalsifiable, which is the same defect as a prohibition that records
       no precondition.

    4. ⚠️⚠️ THE TWO RULES ARE OPPOSED AND NO ARGV FORM SATISFIES BOTH. DO NOT
       "FIX" THIS BACK. Measured:

           literal "cargo"             -> B607 "partial executable path"  WARNING
           shutil.which("cargo")       -> "run without a static string"   FAILURE
           which as an existence check -> B607 returns                    WARNING

       and the check's threshold is `0 new issues`, where a WARNING blocks
       exactly as a FAILURE does. Full-pathing the executable is the honest fix
       for B607 and is what CREATED the static-string failure. There is no code
       form that reaches green, so this suppression is FORCED rather than
       chosen -- a measurement, not a judgement.

       The consequence is a property of the GATE, not of this file: a
       `0 new issues` gate over a MULTI-ANALYZER check has no
       guaranteed-reachable green state. Unlike two THRESHOLDS on one axis
       (satisfy the smaller and you satisfy both), opposed rules admit no
       ordering and no "just be stricter".

    ⚠️ AND NOTE WHICH OF THIS FILE'S TWO CODACY FAILURES WAS WHICH, because the
    diff does not say and they are not the same kind of thing: `_tally`'s
    cyclomatic 13 was a REAL DEFECT and was fixed by splitting. This one is a
    GATE ARTIFACT and could only be suppressed. A red check can be a fact about
    your change or a fact about the gate, and nothing in the annotation
    distinguishes them.
    """
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError("cargo not found on PATH -- the probe needs a compiler")
    # SUPPRESSION NOTE point 4 applies here: the two rules are opposed and no
    # argv form satisfies both. `nosec` on the call line is bandit's, for B603;
    # the bare `nosemgrep` below is the other analyzer's. TWO ANALYZERS, TWO
    # SUPPRESSION SYNTAXES, ONE CHECK-RUN -- a comment written for one does not
    # reach the other, which is the same discovery as the two cyclomatic limits.
    #
    # ⚠️ THE BARE TOKEN MUST BE ON THE LINE IMMEDIATELY ABOVE THE FINDING. The
    # first attempt put it at the TOP of this five-line block, five lines from
    # the call, and it did not take -- an inline suppression is LINE-ANCHORED,
    # so prose between the token and its target silently disarms it. Keep the
    # explanation above the token, never between it and the call.
    # nosemgrep
    proc = subprocess.run(  # nosec B603 - fixed argv, no external input; see above
        [cargo, "check", "-p", "fuel-ir", "--features", "dlpack",
         "--all-targets", "-j", "4", "--message-format", "json"],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return proc.stdout or ""


def _e0609_hits(stdout, struct):
    """{field: {files}} from `no field 'X' on type 'STRUCT'` diagnostics."""
    want = re.compile(r"no field `(\w+)` on type `[&\w:]*%s" % re.escape(struct))
    hits = {}
    for line in stdout.split("\n"):
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line).get("message") or {}
        except ValueError:
            continue
        m = want.search(msg.get("message", ""))
        if not m:
            continue
        for span in msg.get("spans", []):
            f = span.get("file_name", "").replace(chr(92), "/")
            hits.setdefault(m.group(1), set()).add(f)
    return hits


def classify(files):
    if VALIDATOR in files:
        return "VALIDATOR"
    return "ELSEWHERE" if files else "NEVER"


def sweep():
    rows = []
    for rel in SOURCES:
        src = io.open(os.path.join(ROOT, rel), encoding="utf-8").read()
        for struct in structs_in(src):
            names = fields_of(src, struct)
            if not names:
                continue
            hits = probe(rel, struct)
            for name in names:
                files = sorted(hits.get(name, []))
                rows.append((struct, name, classify(files),
                             ",".join(f for f in files if f != VALIDATOR)))
    return rows


def is_padding(field):
    return field.startswith("_pad") or field == "reserved"


def _by_tag(semantic):
    """{VALIDATOR|ELSEWHERE|NEVER: rows} in one pass."""
    buckets = {"VALIDATOR": [], "ELSEWHERE": [], "NEVER": []}
    for row in semantic:
        buckets[row[2]].append(row)
    return buckets


def _split_fdx(rows):
    """(fdx rows, DLPack ABI rows, the semantic subset of fdx)."""
    fdx, dlpack = [], []
    for row in rows:
        (dlpack if row[0] in DLPACK_ABI else fdx).append(row)
    return fdx, dlpack, [r for r in fdx if not is_padding(r[1])]


def _tally(rows):
    fdx, dlpack, semantic = _split_fdx(rows)
    tagged = _by_tag(semantic)
    return {
        "fdx": fdx,
        "semantic": semantic,
        "validator": tagged["VALIDATOR"],
        "elsewhere": tagged["ELSEWHERE"],
        "never": tagged["NEVER"],
        "dlpack": dlpack,
    }


def _print_counts(t):
    print("FDX FIELD COVERAGE -- fields on the wire MINUS fields validate() reads")
    print("  FDX struct fields            : %d" % len(t["fdx"]))
    print("    padding / reserved         : %d  SPEC-EXEMPT by construction"
          % (len(t["fdx"]) - len(t["semantic"])))
    print("    semantic                   : %d" % len(t["semantic"]))
    print("      read by validate.rs      : %d" % len(t["validator"]))
    print("      read ONLY elsewhere      : %d" % len(t["elsewhere"]))
    print("      NEVER read anywhere      : %d" % len(t["never"]))
    print("  DLPack ABI struct fields     : %d  (not FDX -- reported, not counted)"
          % len(t["dlpack"]))


def _print_lists(t):
    print()
    print("READ ONLY OUTSIDE THE VALIDATOR -- exercised, but not enforced:")
    for st, f, _, where in t["elsewhere"]:
        print("  %-22s %-22s %s" % (st, f, where))
    print()
    print("NEVER READ -- candidates, each needing its SPEC read before it is a hole:")
    for st, f, _, _ in t["never"]:
        print("  %-22s %s" % (st, f))


def main():
    t = _tally(sweep())
    _print_counts(t)
    _print_lists(t)
    return 0


def self_test():
    """Calibrate the instrument against a struct whose answer is already known.

    `FDXDTypeExt` was measured BY HAND during GAP-286: `logical_dtype`,
    `bit_width` and `packing` are read by `validate.rs`; `lanes` is not (it is
    set only by a test fixture); `sub_byte_bit_order` is read nowhere.

    ⚠️ THIS ARM ALREADY EARNED ITS PLACE. The first version of `classify()`
    tested `"/dlpack/validate" in path`, which matches `validate/tests.rs` as
    well as `validate.rs`, and it reported `lanes` as VALIDATOR-read. The known
    answer is what exposed it. A sweep calibrated against nothing would have
    shipped that.
    """
    expect = {
        "logical_dtype": "VALIDATOR",
        "bit_width": "VALIDATOR",
        "packing": "VALIDATOR",
        "lanes": "ELSEWHERE",
        "sub_byte_bit_order": "NEVER",
        "_pad": "NEVER",
        "reserved": "NEVER",
    }
    hits = probe(SOURCES[0], "FDXDTypeExt")
    bad = []
    for field, want in expect.items():
        got = classify(sorted(hits.get(field, [])))
        print("  %-22s want %-10s got %-10s %s"
              % (field, want, got, "ok" if got == want else "MISMATCH"))
        if got != want:
            bad.append(field)
    print("  => %s" % ("PASS" if not bad else "FAIL %s" % bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(self_test() if "--self-test" in sys.argv else main())

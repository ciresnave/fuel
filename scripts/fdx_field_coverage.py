#!/usr/bin/env python3
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
import subprocess
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
        proc = subprocess.run(
            ["cargo", "check", "-p", "fuel-ir", "--features", "dlpack",
             "--all-targets", "-j", "4", "--message-format", "json"],
            cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        hits = {}
        for line in (proc.stdout or "").split("\n"):
            if not line.startswith("{"):
                continue
            try:
                msg = json.loads(line).get("message") or {}
            except ValueError:
                continue
            m = re.search(
                r"no field `(\w+)` on type `[&\w:]*%s" % re.escape(struct),
                msg.get("message", ""),
            )
            if not m:
                continue
            for span in msg.get("spans", []):
                f = span.get("file_name", "").replace(chr(92), "/")
                hits.setdefault(m.group(1), set()).add(f)
        return hits
    finally:
        io.open(path, "wb").write(raw)
        assert io.open(path, "rb").read() == raw, (
            "RESTORE FAILED -- bytes differ from the original"
        )


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


def main():
    rows = sweep()
    fdx = [r for r in rows if r[0] not in DLPACK_ABI]
    semantic = [r for r in fdx if not is_padding(r[1])]
    never = [r for r in semantic if r[2] == "NEVER"]
    elsewhere = [r for r in semantic if r[2] == "ELSEWHERE"]

    print("FDX FIELD COVERAGE -- fields on the wire MINUS fields validate() reads")
    print("  FDX struct fields            : %d" % len(fdx))
    print("    padding / reserved         : %d  SPEC-EXEMPT by construction"
          % (len(fdx) - len(semantic)))
    print("    semantic                   : %d" % len(semantic))
    print("      read by validate.rs      : %d"
          % len([r for r in semantic if r[2] == "VALIDATOR"]))
    print("      read ONLY elsewhere      : %d" % len(elsewhere))
    print("      NEVER read anywhere      : %d" % len(never))
    print("  DLPack ABI struct fields     : %d  (not FDX -- reported, not counted)"
          % len([r for r in rows if r[0] in DLPACK_ABI]))
    print()
    print("READ ONLY OUTSIDE THE VALIDATOR -- exercised, but not enforced:")
    for st, f, _, where in elsewhere:
        print("  %-22s %-22s %s" % (st, f, where))
    print()
    print("NEVER READ -- candidates, each needing its SPEC read before it is a hole:")
    for st, f, _, _ in never:
        print("  %-22s %s" % (st, f))
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

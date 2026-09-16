#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Intra-doc-link census: every `unresolved link to X` rustdoc emits, dispositioned.

WHY THIS EXISTS
    Two reasons, both paid for.

    (1) The 2026-09 rustdoc programme (#179, #186) drove broken intra-doc links
        from 462 to 21 using an instrument that DID NOT LIVE IN THE REPOSITORY.
        Nobody but its author could re-run it, and it could not be a gate's
        foundation. `scripts/doc_name_census.py` had already recorded that exact
        finding for the doc-NAME census -- "that absence WAS the finding" -- and
        the doc-LINK census repeated it in the same directory.

    (2) GAP-319. That out-of-repo tool bucketed 234 broken links under
        `RESOLVABLE-IN-PRINCIPLE (item or variant exists) -- path/scope defect`,
        whose predicate is `the item exists somewhere => the path is wrong`.
        THAT PREDICATE CONFLATES EXISTENCE WITH REACHABILITY. It was true of 230
        of 234. The other four were CORRECT links whose targets are simply not
        reachable in a default-features build, and the repair destroyed them:
        `avx`, `neon`, `fuel_cuda_backend::CudaDevice`,
        `fuel_metal_backend::MetalDevice`.

⚠️ THE LESSON THAT SHAPES THIS FILE, AND IT IS NOT "WRITE A BETTER CLASSIFIER"
    The defect was AN AUTOMATIC CLASSIFIER TRUSTED AS A VERDICT. A sharper
    automatic classifier fails the same way -- it just needs a subtler input to
    do it. So this tool does NOT decide why a link is broken. It identifies the
    POPULATION mechanically (rustdoc says so) and requires a HUMAN DISPOSITION
    per entry. The reasoning lives in `DISPOSITIONS`, written by someone who
    looked at the site.

    `doc_name_census.py` states the rule this file obeys:
        "It is a PRINTER, not a count-asserting born-red. A gate born on a
         population that contains known false positives would enforce them."
    That is GAP-319 written down BEFORE it happened, in this repo. It reached
    nobody because it lived in a docstring.

WHAT IT GATES ON, AND WHAT IT DELIBERATELY DOES NOT
    ⚠️ IT DOES NOT GATE ON A COUNT. A count-gate is what enforced the false
    positives, and a count can be satisfied BY DESTROYING EVIDENCE -- de-link a
    correct link and the number improves. An instrument that counts what still
    exists cannot distinguish "fixed" from "destroyed", and both read as
    progress.

    It gates on: EVERY BROKEN LINK CARRIES A DISPOSITION. A newly-broken link
    nobody has classified is UNDISPOSITIONED and reddens `--gate`. That is a
    condition destroying evidence cannot satisfy: delete a link and its entry
    becomes STALE, which also reports.

THE CLASSES
    DEFECT              the link is wrong and should be repaired. Either the
                        target exists nowhere, or it exists AND IS REACHABLE in
                        this build and the path is simply mistaken.
    CORRECT-UNREACHABLE the link is RIGHT. The target exists but this build
                        cannot see it -- a `cfg(feature)` or
                        `cfg(target_feature)` module, an `optional = true`
                        dependency, a CI-excluded crate, a private `mod`, or a
                        cross-crate target under `--no-deps`.
                        ⚠️ DO NOT "FIX" THESE. De-linking one destroys a working
                        reference permanently and silently: the link is gone,
                        nothing renders wrong, and NO FUTURE CENSUS CAN SEE IT,
                        because a census only reports links that still exist.
    NOT-A-LINK          notation rustdoc reads as a link and is not one
                        (array indexing, optional-argument syntax). Fix by
                        putting it in a code span.

⚠️ THE 462 -> 21 FIGURES ARE NOT RE-DERIVABLE BY THIS TOOL
    They were produced by the scratchpad instrument described above, with its own
    extraction rules, at commits that have since moved. Those numbers appear in
    merged PR titles (#179, #186) and in peer summaries. THIS TOOL'S FIRST RUN IS
    NOT A RECONCILIATION OF THEM. If it reports a different number that is TWO
    DIFFERENT INSTRUMENTS, not a regression -- settle any disagreement by a
    MEMBERSHIP DIFF, never by comparing totals.

A DISPOSITION IS KEYED ON THE TARGET, NOT THE SITE
    The same target broken at two sites gets one entry. If a target ever needs
    DIFFERENT dispositions at different sites, that is itself a finding and this
    table must be split by (target, file) -- do not paper over it by choosing the
    more convenient class.

Usage:
    python scripts/doc_link_census.py                # build docs, print census
    python scripts/doc_link_census.py --from F.json  # use a captured
                                                     # `cargo doc --message-format json`
    python scripts/doc_link_census.py --gate         # additionally exit 1 on any
                                                     # UNDISPOSITIONED link
    python scripts/doc_link_census.py --self-test    # two-arm proof the gate
                                                     # discriminates
"""

import io
import json
import os
import re
import subprocess
import sys

NL = chr(10)

# Crates CI excludes from the workspace build. Kept beside the command that uses
# them so the two cannot drift apart.
EXCLUDED = [
    "fuel-mkl-cpu-backend",
    "fuel-aocl-cpu-backend",
    "fuel-cuda-backend",
    "fuel-metal-backend",
    "fuel-metal-kernels",
]

DEFECT = "DEFECT"
CORRECT = "CORRECT-UNREACHABLE"
NOT_A_LINK = "NOT-A-LINK"

# Mechanism vocabulary. A CORRECT-UNREACHABLE reason MUST name one of these, and
# `--gate` enforces it.
#
# WHY ENFORCED RATHER THAN CONVENTIONAL: a count-gate is satisfied by DESTROYING
# EVIDENCE; a disposition-gate is satisfied by MIS-DISPOSITIONING. Label a real
# DEFECT `CORRECT-UNREACHABLE` and this gate goes green forever with a reason
# nobody re-reads. Naming the mechanism does not PREVENT a wrong disposition --
# nothing can -- but it makes a wrong one DISCOVERABLE IN ONE GREP, which is the
# property the original `path/scope defect` bucket lacked over 234 entries.
#
#     neon   CORRECT-UNREACHABLE  cfg(target_feature = "neon") ...   <- checkable
#     <bad>  CORRECT-UNREACHABLE  "it is gated"                      <- unfalsifiable
MECHANISMS = (
    "cfg(feature",
    "cfg(target_feature",
    "optional = true",
    "CI-excluded crate",
    "private mod",
    "cross-crate under --no-deps",
)

# THE CLASSIFICATION ASYMMETRY, LEARNED ON THIS VERY TABLE.
# CORRECT-UNREACHABLE is the SAFE verdict -- it means "leave it alone". DEFECT is
# the DANGEROUS one, because it licenses de-linking, and a wrongly de-linked
# reference is destroyed permanently and silently.
#
# So a DEFECT needs evidence the link cannot resolve under ANY plausible
# configuration. A SOURCE READ SAYING "THIS LOOKS REACHABLE" IS NOT SUFFICIENT.
# Measured here: `crate::vulkan_dispatch::register_vulkan_kernels` is a `pub fn`
# in an UNGATED `pub mod` with no cfg above it -- every source signal says
# reachable -- and it resolves ONLY under `--features vulkan` (two-arm: plain 1
# unresolved, --features vulkan 0). Dispositioning from the source read would
# have marked a CORRECT link as a DEFECT and invited the next author to destroy
# it, which is GAP-319 recurring inside the instrument built to prevent it.
#
# target -> (CLASS, reason). Every entry was looked at; the reason says what was
# seen, not what was assumed.
DISPOSITIONS = {
    # ---- CORRECT-UNREACHABLE: the link is RIGHT. Do not de-link. ----
    # The two SIMD mods are SYMMETRIC, and this table said they were not until
    # CI proved otherwise. See `avx` below for how that was measured.
    "neon": (CORRECT,
             'cfg(target_feature = "neon") on `pub mod neon`, fuel-quantized/src/lib.rs. '
             "aarch64-only. MEASURED: resolves on aarch64+neon, not on x86. NO cfg fixes "
             "it cross-arch -- cfg(any(doc, ...)) compiles the module on x86, where its "
             "NEON types do not exist (E0425 int8x16_t). Its sibling `avx` fails the same "
             "way in the opposite direction; neither can be cfg-fixed."),
    "avx": (CORRECT,
             'cfg(target_feature = "avx2") on `pub mod avx`, fuel-quantized/src/lib.rs. '
             "x86-only, and the mirror image of `neon` above. MEASURED, and I got this "
             "one WRONG FIRST: I shipped cfg(any(doc, target_feature = \"avx2\")) on this "
             "mod, having verified it on x86, and CI rejected it on aarch64-apple-darwin "
             "with E0425 `cannot find type __m256i` -- the same error class, the same "
             "cause, the opposite architecture. I had run a two-architecture cross-over "
             "to prove these two LINKS were correct, then applied a FIX tested on one "
             "architecture. The cfg is reverted; this entry replaces it."),
    "SType::to_fdx": (CORRECT,
             'cfg(feature = "dlpack") on the impl block at fuel-ir/src/stype.rs:140; '
             "`dlpack = []` is a non-default feature. The doc prose itself says "
             '"(step 3, `dlpack` feature)".'),
    "crate::vulkan_dispatch::register_vulkan_kernels": (CORRECT,
             "the `vulkan` feature of fuel-dispatch, which turns on an optional = true "
             "dep (fuel-vulkan-backend). MEASURED TWO-ARM, not read: `cargo doc -p "
             "fuel-dispatch --no-deps` leaves it unresolved; `--features vulkan` resolves "
             "it. EVERY SOURCE SIGNAL SAYS REACHABLE -- `pub fn` in an UNGATED `pub mod`, "
             "no cfg above it -- so a source read would have mis-classified this DEFECT."),
    "fuel_cuda_backend::CudaDevice": (CORRECT,
             "optional = true dep (fuel-cuda-backend, behind fuel-core's `cuda` feature) "
             "and also a CI-excluded crate. The containing `pub mod cuda_backend` is "
             "UNGATED, so the doc is always rendered while the target is conditionally "
             "present. STRUCTURAL, not build-verified: confirming needs a --features cuda "
             "doc build, i.e. the 56-93 minute baracuda forge."),
    "fuel_metal_backend::MetalDevice": (CORRECT,
             "optional = true dep (fuel-metal-backend, behind fuel-core's `metal` feature) "
             "and also a CI-excluded crate; `pub mod metal_backend` is UNGATED. "
             "STRUCTURAL, not build-verified: confirming needs an Apple target."),

    # ---- NOT-A-LINK: notation rustdoc misreads. Fix with a code span. ----
    "Layout::contiguous(shape)": (NOT_A_LINK,
             "a CALL EXPRESSION, not a path -- the trailing `(shape)` makes it notation. "
             "fuel-ir/src/storage.rs:73."),
    "axis": (NOT_A_LINK,
             "a bare prose word in brackets at fuel-vulkan-backend/src/lib.rs:11054, not "
             "an item reference."),

    # ---- DEFECT: repair the link. ----
    "crate::dispatch::KernelBindingTable": (DEFECT,
             "`pub struct KernelBindingTable` is at fuel-dispatch/src/kernel.rs:944 and is "
             "re-exported at the crate root (lib.rs:154). There is NO `dispatch` module "
             "holding it, so the path names something that does not exist. Repoint to "
             "`crate::KernelBindingTable`."),
    "crate::Tensor": (DEFECT,
             "STALE NAME. `Tensor` was renamed to `NodeHandle` (cf861588) and the eager "
             "type is gone. fuel-core/src/shape.rs:5."),
    "fuel_dispatch::pipelined::CapturedDecodeSession::capture": (DEFECT,
             "no declaration of `CapturedDecodeSession` anywhere in the workspace -- a "
             "DEAD REFERENCE, not a path defect."),
    "BindingEntry": (DEFECT,
             "`pub struct BindingEntry` exists at fuel-dispatch/src/kernel.rs:838, but the "
             "bare shorthand is not in scope at compiled.rs:116 / ranker/candidate.rs:6. "
             "Needs a path."),
    "CostEstimate": (DEFECT,
             "`pub struct CostEstimate` at fuel-dispatch/src/fused.rs:88; bare shorthand "
             "not in scope at ranker/cost_vector.rs:107. Needs a path."),
    "RESERVED_DTYPE_TOKENS": (DEFECT,
             "`pub const` at fuel-ir/src/dtype.rs:453; bare shorthand not in scope at "
             "token_kind.rs:72. Needs a path."),
    "Op::Conv1D": (DEFECT,
             "`Conv1D` is a `pub struct` in fuel-cpu-backend, NOT an `Op` variant -- the "
             "same class as the QMatMul/Conv2D/FlashAttn sites #179 repaired, where the "
             "SENTENCE is false rather than merely unlinked. Name the real construct."),
    "Op::NonZeroIndices": (DEFECT,
             "the variant exists but the link is cross-crate from fuel-nn; needs "
             "`fuel_graph::Op::NonZeroIndices`."),
    "Op::WriteSlice": (DEFECT,
             "cross-crate from fuel-transformers; needs `fuel_graph::Op::WriteSlice`."),
    "WorkItemKind::Alloc": (DEFECT,
             "the `Alloc` variant is at fuel-dispatch/src/pipelined.rs:667; the link is "
             "cross-crate from fuel-core/src/pipelined_bridge.rs. Needs the full path."),
    "Self::cast": (DEFECT,
             "`pub fn cast` lives on `NodeHandle` at fuel-graph/src/lib.rs:6609. `Self` at "
             "fuel-core/src/lazy.rs:1868 is a different type, so `Self::` names the wrong "
             "one."),
    "NodeHandle::flash_attn_dyn": (DEFECT,
             "`pub fn flash_attn_dyn` is at fuel-graph/src/lib.rs:5202; the link at "
             "registry.rs:264 does not resolve against `NodeHandle`. Verify the enclosing "
             "impl before repointing -- a pub-mod chain reaches a MODULE, not an item "
             "inside an impl."),
    "PrecisionGuarantee::UNAUDITED": (DEFECT,
             "`pub const UNAUDITED` at fuel-dispatch/src/fused.rs:208; the link at "
             "kernel.rs:981 does not resolve. An associated-const link needs the exact "
             "owning type path."),
    "LlamaModel::forward_paged_step": (DEFECT,
             "`pub fn forward_paged_step` at fuel-core/src/lazy.rs:9160; the link at "
             "inference_context.rs:1547/1553 does not resolve against `LlamaModel`. Check "
             "the enclosing impl before repointing."),
    "LlamaModel::forward_paged_step_persistent": (DEFECT,
             "same shape as its sibling above, at inference_context.rs:1527."),
}


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def git_ref(root):
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                       capture_output=True, text=True)
    return r.stdout.strip() or "<unknown>"


class CensusUnusable(Exception):
    """The doc build did not produce a census. NOT the same as a clean census."""


def build_docs(root):
    """Run cargo doc and return the captured json, or RAISE.

    The first version of this function discarded the exit code and sent stderr
    to DEVNULL. A failed or empty doc build then yielded ZERO broken links,
    which this tool would have printed as a perfectly clean census.

    That is the portfolio rule `never discard stderr from a command whose
    output you are about to count`, and `a tool that fails early reports a
    smaller population than the truth, and the smaller number looks like good
    news` -- committed inside the instrument built to catch exactly that. A
    reviewer reading the code found it; no run of it ever could, because its
    failure mode IS a clean-looking pass.

    So: keep stderr, check the exit code, and require a POSITIVE ARTIFACT --
    at least one `compiler-artifact` record proving rustdoc actually ran. An
    empty file and a successful build of nothing are indistinguishable by size
    alone.
    """
    out = os.path.join(root, "target", "doc-link-census.json")
    err = os.path.join(root, "target", "doc-link-census.err")
    cmd = ["cargo", "doc", "--workspace", "--no-deps", "-j", "4",
           "--message-format", "json"]
    for c in EXCLUDED:
        cmd += ["--exclude", c]
    with io.open(out, "w", encoding="utf-8") as fh, io.open(err, "w", encoding="utf-8") as eh:
        rc = subprocess.run(cmd, cwd=root, stdout=fh, stderr=eh, check=False).returncode
    if rc != 0:
        raise CensusUnusable(
            "cargo doc exited %d -- stderr kept at %s. A failed doc build yields zero\n"
            "broken links, which is NOT a clean census. Fix the build, then re-run." % (rc, err))
    assert_census_is_real(out)
    return out


def assert_census_is_real(path):
    """A census with no rustdoc artifacts is not a census.

    Zero broken links is only meaningful if rustdoc RAN. This is the
    compile-artifact rule: an exit code is evidence about the harness until an
    artifact proves the tool ran on the code you think it did.
    """
    artifacts = 0
    for line in io.open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("reason") == "compiler-artifact":
            artifacts += 1
    if artifacts == 0:
        raise CensusUnusable(
            "no `compiler-artifact` records in %s -- rustdoc did not run, so a report of\n"
            "zero broken links would be a statement about nothing." % path)
    return artifacts


BROKEN = re.compile(r"unresolved link to `([^`]+)`")


def broken_links(path):
    """(target, file, line) for every broken_intra_doc_links diagnostic."""
    rows = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("reason") != "compiler-message":
            continue
        m = o.get("message", {})
        if ((m.get("code") or {}).get("code")) != "rustdoc::broken_intra_doc_links":
            continue
        hit = BROKEN.search(m.get("message") or "")
        spans = [s for s in m.get("spans", []) if s.get("is_primary")]
        if hit and spans:
            rows.append((hit.group(1), spans[0]["file_name"], spans[0]["line_start"]))
    return rows


def mechanism_violations():
    """CORRECT-UNREACHABLE entries whose reason names no MECHANISM.

    This is a check on the FORM of a reason, not its TRUTH -- a mechanical check
    cannot tell whether the named cfg is the real one. That division is the point:
    the machine enforces that a mechanism was NAMED, a human supplies which. An
    unnamed mechanism is unfalsifiable, and unfalsifiable is how a wrong
    disposition survives.
    """
    bad = []
    for target, (cls, reason) in sorted(DISPOSITIONS.items()):
        if cls != CORRECT:
            continue
        if not any(mech in reason for mech in MECHANISMS):
            bad.append(target)
    return bad


def classify(rows):
    """(dispositioned, undispositioned, stale) -- no automatic verdicts."""
    seen = set(t for t, _, _ in rows)
    dispositioned, undispositioned = [], []
    for target, f, ln in sorted(rows):
        if target in DISPOSITIONS:
            cls, why = DISPOSITIONS[target]
            dispositioned.append((target, f, ln, cls, why))
        else:
            undispositioned.append((target, f, ln))
    stale = sorted(t for t in DISPOSITIONS if t not in seen)
    return dispositioned, undispositioned, stale


def run_gate(dispositions, targets):
    """Exit code of the REAL `main()` under `--gate`, on a constructed capture.

    `dispositions` replaces the table for the call; `targets` become broken-link
    records. Output is swallowed -- the exit code is the assertion.
    """
    global DISPOSITIONS
    import contextlib
    import tempfile

    recs = ['{"reason":"compiler-artifact","target":{"name":"x"}}']
    for t in targets:
        recs.append('{"reason":"compiler-message","message":{"code":'
                    '{"code":"rustdoc::broken_intra_doc_links"},"message":'
                    '"unresolved link to `%s`","spans":[{"is_primary":true,'
                    '"file_name":"a.rs","line_start":1}]}}' % t)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    io.open(tmp, "w", encoding="utf-8").write("\n".join(recs) + "\n")
    saved, saved_argv = DISPOSITIONS, sys.argv
    try:
        DISPOSITIONS = dispositions
        sys.argv = ["doc_link_census.py", "--from", tmp, "--gate"]
        with contextlib.redirect_stdout(io.StringIO()):
            return main()
    finally:
        DISPOSITIONS, sys.argv = saved, saved_argv
        os.unlink(tmp)


def integration_arms():
    """The arms that run the REAL `--gate`, one per term of its condition.

    Each failing case makes exactly ONE term non-empty. The first version of
    this arm fed an undispositioned link through a fixture table that ALSO
    held a mechanism-less CORRECT entry, so the gate reddened on the wrong term
    and deleting the undispositioned check would have left the arm green.
    """
    named = (CORRECT, 'cfg(feature = "x") on mod y')
    arms = {
        "F0": run_gate({"T": named}, ["T"]) == 0,
        "F1": run_gate({}, ["T"]) == 1,
        "F2": run_gate({"T": (CORRECT, "it is gated")}, ["T"]) == 1,
        "F3": run_gate({"T": named, "Gone": (DEFECT, "was here")}, ["T"]) == 1,
    }

    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    io.open(tmp, "w", encoding="utf-8").write("")
    try:
        assert_census_is_real(tmp)
        arms["G"] = False
    except CensusUnusable:
        arms["G"] = True
    finally:
        os.unlink(tmp)
    return arms


def self_test():
    """Two arms, because this instrument's whole purpose is telling them apart.

    ARM A (born-red): a broken link with no disposition MUST be reported.
    ARM B (born-green): a link dispositioned CORRECT-UNREACHABLE must NOT be
    reported as needing attention -- otherwise the gate pressures a future
    author into de-linking a correct link, which is the GAP-319 defect arriving
    through the instrument built to prevent it.
    """
    global DISPOSITIONS
    saved = DISPOSITIONS
    try:
        DISPOSITIONS = {"known_gated": (CORRECT, "test fixture")}
        rows = [("known_gated", "a.rs", 1), ("brand_new", "b.rs", 2)]
        d, u, s = classify(rows)
        arm_a = [t for t, _, _ in u] == ["brand_new"]
        arm_b = any(t == "known_gated" and c == CORRECT for t, _, _, c, _ in d)
        arm_c = s == []
        # stale arm: a disposition whose subject is gone
        _, _, s2 = classify([("brand_new", "b.rs", 2)])
        arm_d = s2 == ["known_gated"]

        # ARM E: a CORRECT-UNREACHABLE reason naming no mechanism must be caught.
        # This is the arm that makes mis-disposition discoverable -- the gate's
        # own failure mode.
        DISPOSITIONS["vague"] = (CORRECT, "it is gated")
        caught = "vague" in mechanism_violations()
        DISPOSITIONS["precise"] = (CORRECT, 'cfg(feature = "x") on mod y')
        clean = "precise" not in mechanism_violations()
        arm_e = caught and clean

        arms = integration_arms()

        print("SELF-TEST")
        print("  ARM A  undispositioned link is REPORTED      : %s" % ("PASS" if arm_a else "FAIL"))
        print("  ARM B  CORRECT-UNREACHABLE is NOT flagged    : %s" % ("PASS" if arm_b else "FAIL"))
        print("  ARM C  no false stale on a full population   : %s" % ("PASS" if arm_c else "FAIL"))
        print("  ARM D  a vanished subject reports STALE      : %s" % ("PASS" if arm_d else "FAIL"))
        print("  ARM E  unnamed mechanism is CAUGHT           : %s" % ("PASS" if arm_e else "FAIL"))
        labels = (
            ("F0", "real --gate PASSES a clean population  "),
            ("F1", "real --gate reddens on UNDISPOSITIONED "),
            ("F2", "real --gate reddens on NO MECHANISM    "),
            ("F3", "real --gate reddens on STALE           "),
            ("G", "a census with no artifacts is REFUSED  "),
        )
        for key, text in labels:
            print("  ARM %-2s %s: %s" % (key, text, "PASS" if arms[key] else "FAIL"))
        return 0 if all([arm_a, arm_b, arm_c, arm_d, arm_e] + list(arms.values())) else 1
    finally:
        DISPOSITIONS = saved


def print_by_class(dispositioned):
    """Group the dispositioned links by class and print each with its reason."""
    by_class = {}
    for target, f, ln, cls, why in dispositioned:
        by_class.setdefault(cls, []).append((target, f, ln, why))
    for cls in sorted(by_class):
        print("[%s] (%d)" % (cls, len(by_class[cls])))
        for target, f, ln, why in by_class[cls]:
            print("   %-44s %s:%d" % (target[:42], f, ln))
            print("        %s" % why)
        print()


def print_problems(stale, undispositioned, vague):
    """The three ways this census asks for attention, each with its reason."""
    if stale:
        print("[STALE DISPOSITION] (%d) -- no longer broken; entry is now moot." % len(stale))
        print("   A disposition whose subject vanished may mean it was FIXED -- or")
        print("   that a correct link was DESTROYED. Read the site before deleting.")
        for t in stale:
            print("   %-44s %s" % (t[:42], DISPOSITIONS[t][0]))
        print()

    if undispositioned:
        print("[UNDISPOSITIONED] (%d) -- each needs a REASONED class, not a guess."
              % len(undispositioned))
        for target, f, ln in undispositioned:
            print("   %-44s %s:%d" % (target[:42], f, ln))
        print()
        print("   Before classifying one as %s, check whether the target EXISTS but is" % DEFECT)
        print("   simply UNREACHABLE in this build (cfg / optional dep / excluded crate /")
        print("   private mod). If so it is %s and DE-LINKING IT IS HARM." % CORRECT)

    if vague:
        print("[NO MECHANISM NAMED] (%d) -- a %s entry whose reason names none of"
              % (len(vague), CORRECT))
        print("   %s" % ", ".join(MECHANISMS))
        print("   An unnamed mechanism is UNFALSIFIABLE, which is how a wrong")
        print("   disposition survives. Name the cfg, the feature, the optional dep,")
        print("   the excluded crate, or the private mod.")
        for t in vague:
            print("   %s" % t)
        print()


def resolve_source(root, args):
    """Where the census json comes from, or raise CensusUnusable."""
    if "--from" in args:
        src = args[args.index("--from") + 1]
        assert_census_is_real(src)
        return src
    print("building docs (cargo doc --workspace --no-deps, %d exclusions) ..." % len(EXCLUDED))
    return build_docs(root)


def print_headline(root, src, rows, dispositioned, undispositioned):
    """The counts, plus the non-comparability warning that must ride with them."""
    print("doc-link census at %s   (source: %s)" % (git_ref(root), os.path.basename(src)))
    print("  broken intra-doc link diagnostics : %d" % len(rows))
    print("  distinct targets                  : %d" % len(set(t for t, _, _ in rows)))
    print("  dispositioned                     : %d" % len(dispositioned))
    print("  UNDISPOSITIONED                   : %d" % len(undispositioned))
    print()
    print("  these figures are NOT comparable with the 462->21 of #179/#186,")
    print("  which came from a different, out-of-repo instrument. Settle any")
    print("  disagreement by a MEMBERSHIP DIFF, never by comparing totals.")
    print()


def main():
    args = sys.argv[1:]
    if "--self-test" in args:
        return self_test()

    root = repo_root()
    gate = "--gate" in args
    try:
        src = resolve_source(root, args)
    except CensusUnusable as exc:
        print()
        print("CENSUS UNUSABLE -- refusing to report:")
        print("   %s" % exc)
        return 2

    rows = broken_links(src)
    d, u, stale = classify(rows)

    print_headline(root, src, rows, d, u)

    print_by_class(d)

    vague = mechanism_violations()
    print_problems(stale, u, vague)

    if gate and (u or vague or stale):
        print()
        print("GATE: %d undispositioned, %d without a named mechanism, %d stale."
              % (len(u), len(vague), len(stale)))
        if stale:
            print("   A STALE entry fails the gate DELIBERATELY. This row's own finding is")
            print("   that an instrument counting what still exists cannot distinguish FIXED")
            print("   from DESTROYED -- and a gate that could not fail on `stale` was that")
            print("   same hole one level up: the disposition rots and the gate keeps saying")
            print("   yes. Read the site, then either delete the entry (it was fixed) or")
            print("   restore the link (it was destroyed).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

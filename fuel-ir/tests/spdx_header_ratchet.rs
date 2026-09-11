// SPDX-License-Identifier: MIT OR Apache-2.0
//! **Every tracked SOURCE file carries an SPDX licence identifier, and no
//! tracked file carries an unaccounted-for third-party copyright notice.**
//!
//! WHY THIS IS A GATE AND NOT A SWEEP, AND THE REASON IS MEASURED RATHER THAN
//! ASSUMED. `62b0e36b` stamped 795 files on 2026-08-19. By 2026-09-10 ten files
//! carried no identifier — and **not one of them was missed by that sweep**:
//! every one was ADDED between 2026-08-20 and 2026-09-02. The gap was not a hole
//! in the sweep, it was REGROWTH, and 98% was on its way back down.
//!
//! **A SWEEP IS NOT THE FIX FOR A PROPERTY THAT REGROWS.** A one-time cleanup of
//! a property that every new file must satisfy decays at exactly the rate the
//! repository grows, and nothing reports the decay. This test is the ratchet the
//! sweep needed.
//!
//! WHAT AN ABSENT IDENTIFIER ACTUALLY COSTS, so it is not taken on faith: the
//! root manifest declares `license = "MIT OR Apache-2.0"` and every crate takes
//! it with `license.workspace = true`, so an unstamped file is not *unlicensed* —
//! it is **licensed with no in-file evidence of it**, which is what automated
//! licence scanners, SBOM generators and downstream vendoring tools read. The
//! manifest binds the crate; the header is what survives a file being copied out
//! of it.
//!
//! # THE GATE HAS THREE PARTS AND THEY MUST NOT BE COLLAPSED
//!
//! ## Part 1 — the UNSCOPED notice survey
//!
//! Reads **every tracked file whole**, of every extension, and reddens on any
//! `copyright` not named in [`COPYRIGHT_ACCOUNTED_FOR`] with a reason.
//!
//! ⚠️ **THE ARGUMENT FOR THIS PART EXISTING SEPARATELY, IN ONE MEASUREMENT:
//! `quantized.metal`'s third-party notice is at LINE 2844.** Part 2 reads
//! [`HEAD_BYTES`] = 512 bytes. **A head-scoped check cannot see that notice and
//! never could** — it is far past where Part 2 stops looking. Collapse Part 1
//! into Part 2 and a vendored file carrying somebody else's copyright deep in
//! its body becomes invisible to the only instrument that was going to look.
//!
//! The two parts ask different questions. Part 2 asks *"does this file declare
//! OUR licence?"*. Part 1 asks *"does this file carry SOMEBODY ELSE'S claim?"*.
//! A file can fail either independently, and the second has legal consequences.
//!
//! ## Part 2 — the header requirement, over an EXPLICIT extension list
//!
//! [`SOURCE_EXTENSIONS`] is written out rather than inferred. An inferred list
//! ("anything that looks like code") silently acquires and loses members; an
//! explicit one is a claim that Part 3 then audits.
//!
//! ⚠️ **AND THE CLASSIFICATION IS BY NAMED EXTENSION, NEVER BY A MARKER STRING.**
//! Exempting on something like an `@generated` match reads as obviously correct
//! and is not: this repository contains hand-written files that *describe*
//! generated output, and a string match cannot tell a file that IS generated
//! from one that TALKS ABOUT generation. An extension cannot be written into a
//! file by accident; a marker string can.
//!
//! ## Part 3 — the extension-ABSENCE check
//!
//! Every extension present must appear in [`SOURCE_EXTENSIONS`] **or** in
//! [`NON_SOURCE_EXTENSIONS`] with a reason. **Without this part, Part 2's list
//! is unfalsifiable**: a new `.cu`, `.wgsl` or `.c` file would be covered by no
//! rule at all, and the gate would stay green while the repository grew a whole
//! unstamped language.
//!
//! Part 3 also flags STALE declines. ⚠️ **The subtraction is against EVERY
//! present extension, never against the source ones only.** Declining a
//! non-source extension (`.spv`, `.jpg`, `.txt`) is legitimate and must be
//! recordable without reddening; subtract only the source set and every such
//! entry reads as stale, **so the gate would punish you for writing the reason
//! down — and the reason then goes back into a code comment, which is the exact
//! artifact class this gate exists to replace.**
//!
//! # EXEMPTIONS ASSERT THE PROPERTY THEY WERE EXEMPTED FOR
//!
//! An exemption that only NAMES its subject decays into a hole. Every exemption
//! here asserts the condition that justified it, so it reddens when its own
//! premise stops being true.
//!
//! ⚠️ **AND THE PROPERTY IS NOT THE SAME FOR ALL OF THEM, WHICH A SINGLE BLANKET
//! ASSERTION WOULD HAVE GOT BACKWARDS.** The third-party files fall into THREE
//! classes with different — in two cases OPPOSITE — obligations:
//!
//! * [`INHERITED_WITHOUT_NOTICE`] (21) — carry no notice and no identifier.
//!   Assert they carry **neither**: stamping our dual licence on them would
//!   assert a grant we are not entitled to make.
//! * [`INHERITED_WITH_NOTICE`] (3) — carry a real third-party notice. Assert the
//!   notice **is still there**. The hazard is the exact inverse: a tidy-up that
//!   deletes Apple's copyright line out of a vendored kernel.
//! * [`APACHE_ONLY_FILES`] (2) — upstream DECLARED a licence, and it is not ours.
//!   Assert the identifier line still says `Apache-2.0` and never acquires `MIT`.
//!
//! Measured: of 16 inherited `.metal` files, **3 carry notices and 13 do not**.
//! A single "still carries no notice" assertion over all of them would have been
//! false for three files on the day it was written.
//!
//! ⚠️ **AND THE THIRD CLASS EXISTS BECAUSE THIS GATE CAUGHT ITS OWN AUTHOR.**
//! `onnx.proto3` was originally filed in the first list, which asserts a file
//! carries no identifier — **an assumption made without measuring it.** It has
//! carried an upstream `Apache-2.0` identifier since before the fork, so the
//! exemption arm went red on its first run. The lists are not decoration: the
//! one that fired was checking a claim nobody had verified.
//!
//! # ⚠️ EVERY ARM HERE IS LATENT, SO ITS NEGATIVE CASE IS CONSTRUCTED
//!
//! Nothing in this tree makes the absence arm, the stale-decline arm or the
//! exemption arms fail. **A latent defect has no failing arm, so a fix for it
//! cannot be verified by execution** — "applied" and "not applied" are
//! observationally identical from any green run. The predicates are therefore
//! written as pure functions over injected tables, and the `constructed_*` tests
//! below feed each one a case built to fail. **An exemption whose failure mode
//! has never been observed is a hole with a comment on it.**
//!
//! ## ⚠️ THE ONE FILE STAMPED DIFFERENTLY, AND IT DEFENDS ITSELF
//!
//! `fuel-examples/src/bs1770.rs` carries `Apache-2.0` **only**. It is a verbatim
//! copy of a third-party Apache-2.0 work whose author never granted an MIT
//! option, and the ORIGINAL sweep stamped the dual licence on it before
//! `ef59fc23` corrected it — that commit's own subject reads *"my SPDX sweep
//! asserted a grant that does not exist"*. **That is a legal claim, not a
//! formatting one.** So this gate does not merely SKIP that file: it asserts the
//! identifier line still says `Apache-2.0` and still does not say `MIT`.
//!
//! # ENUMERATION IS `git ls-files`, DELIBERATELY
//!
//! Same population the census measured, respects `.gitignore`, and cannot pick
//! up untracked scratch files. **If git is unavailable the test FAILS rather
//! than passing over an empty list** — a gate that cannot enumerate must never
//! report clean, which is the `0 passed` trap.
//!
//! # SELF-MATCHING
//!
//! This file carries a real identifier on line 1 and is scanned like any other
//! tracked file: **the gate is subject to itself.** It is also named in
//! [`COPYRIGHT_ACCOUNTED_FOR`], because the prose above necessarily contains the
//! word this gate searches for — **a source scan must not be inside what it
//! scans without declaring itself**, or it reports its own vocabulary as a
//! finding.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Bytes of each file inspected by Part 2. Generous enough for a shebang-like
/// preamble or a short block comment above the identifier, small enough that the
/// token can not be satisfied by something buried in the body.
///
/// ⚠️ Part 1 deliberately does NOT use this bound. See the module docs:
/// `quantized.metal`'s notice is at line 2844.
const HEAD_BYTES: usize = 512;

const TOKEN: &str = "SPDX-License-Identifier";

/// Third-party files carrying `Apache-2.0` **only**, with the reason each does.
/// Named here so the reason travels with the file instead of living in a commit
/// message.
///
/// These are NOT exemptions from Part 2 — they carry a real identifier and are
/// required to. What they are exempt from is the *dual* licence: their
/// identifier line must keep saying `Apache-2.0` and must never acquire `MIT`.
///
/// ⚠️ `onnx.proto3` was found by this gate's own exemption arm going red. It had
/// been filed as "inherited, carries no identifier" **without that being
/// measured** — it has carried an upstream `Apache-2.0` identifier since before
/// the fork. An inherited file whose upstream DECLARED its licence is the
/// strongest provenance available and the opposite of an open question; filing
/// it with the undeclared ones would have buried that.
const APACHE_ONLY_FILES: &[(&str, &str)] = &[
    (
        "fuel-examples/src/bs1770.rs",
        "verbatim copy of a third-party Apache-2.0 work whose author never granted an MIT option",
    ),
    (
        "fuel-onnx/src/onnx.proto3",
        "generated from upstream ONNX, which declares Apache-2.0 in the file itself",
    ),
];

const THIS_FILE: &str = "fuel-ir/tests/spdx_header_ratchet.rs";

// ---------------------------------------------------------------------------
// PART 2 — what must carry an identifier
// ---------------------------------------------------------------------------

/// Extensions whose files must carry an SPDX identifier in their first
/// [`HEAD_BYTES`] bytes. Audited for completeness by Part 3.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "slang", "glsl", "py", "metal", "ps1", "sh", "h", "proto3",
];

/// Source files carrying no extension at all. An extension-keyed rule cannot see
/// these, so they are named.
const SOURCE_PATHS_WITHOUT_EXTENSION: &[&str] = &["Makefile", "scripts/hooks/pre-commit"];

// ---------------------------------------------------------------------------
// PART 3 — everything else, DECLINED with a reason rather than ignored
// ---------------------------------------------------------------------------

/// Extensions that carry no licence header, each with the reason. A reason is
/// required so that "not source" is a judgement on the record rather than an
/// omission — the difference between a decision and a gap.
const NON_SOURCE_EXTENSIONS: &[(&str, &str)] = &[
    ("md", "prose"),
    ("txt", "prose and plain-text fixtures"),
    ("org", "prose"),
    (
        "toml",
        "cargo/config manifests; the licence is DECLARED in them, not stamped on them",
    ),
    ("lock", "generated by cargo; edits are not authored"),
    ("json", "data and fixtures"),
    ("yml", "CI configuration"),
    ("yaml", "CI configuration"),
    ("cfg", "configuration"),
    ("code-workspace", "editor configuration"),
    (
        "spv",
        "compiled SPIR-V: a build OUTPUT of the .slang/.glsl sources, which ARE stamped",
    ),
    ("safetensors", "binary model weights"),
    ("pt", "binary torch fixture"),
    ("pth", "binary torch fixture"),
    ("npy", "binary numpy fixture"),
    ("bytes", "binary fixture"),
    ("jpg", "binary image asset"),
    ("png", "binary image asset"),
    ("gif", "binary image asset"),
    ("mp4", "binary video asset"),
    (
        "ttf",
        "binary font asset; its own notice is asserted via COPYRIGHT_ACCOUNTED_FOR",
    ),
];

/// Extensionless files that are not source, each with the reason.
const NON_SOURCE_PATHS_WITHOUT_EXTENSION: &[(&str, &str)] = &[
    ("LICENSE-APACHE", "licence text"),
    ("LICENSE-MIT", "licence text"),
    ("fuel-core/LICENSE", "licence text"),
    (".gitignore", "git configuration"),
    (".gitattributes", "git configuration"),
    (
        "fuel-dispatch/fixtures/kiss-corpus/.gitattributes",
        "git configuration",
    ),
    (
        ".git-blame-ignore-revs",
        "git configuration; prose ABOUT the stamping sweep",
    ),
];

// ---------------------------------------------------------------------------
// EXEMPTIONS — split by the property each was exempted FOR
// ---------------------------------------------------------------------------

/// Inherited from Candle before the `815abd04` fork rename, carrying NO
/// third-party notice of their own.
///
/// **They must stay unstamped.** Stamping `MIT OR Apache-2.0` on a file whose
/// provenance we did not author asserts a licence grant nobody made — the exact
/// error `ef59fc23` exists to record. Their licensing is an open question routed
/// to the copyright holder, not a gap to be closed by a sweep.
///
/// ⚠️ **PROVENANCE NOTE, because the obvious query gets this backwards: a global
/// rename makes every file it touched look like a bulk import, so a "single
/// adding commit" is a vendoring signature ONLY in a repository that never
/// renamed.** These were classified with `git log --follow` against the
/// `815abd04` boundary, not from an unfollowed `--diff-filter=A`.
///
/// RETIREMENT CONDITION, so this is not a permanent exemption wearing a
/// temporary one's clothes: when the licensing question is answered, each file
/// is either stamped with the identifier that answer produces, or removed.
/// Either way it leaves this list.
const INHERITED_WITHOUT_NOTICE: &[&str] = &[
    "Makefile",
    "fuel-core/tests/npy.py",
    "fuel-core/tests/pth.py",
    "fuel-examples/examples/flux/t5_tokenizer.py",
    "fuel-examples/examples/marian-mt/python/convert_slow_tokenizer.py",
    "fuel-examples/examples/resnet/export_models.py",
    "fuel-examples/examples/whisper/extract_weights.py",
    "fuel-examples/examples/yolo-v3/extract-weights.py",
    "fuel-metal-kernels/src/metal_src/affine.metal",
    "fuel-metal-kernels/src/metal_src/binary.metal",
    "fuel-metal-kernels/src/metal_src/cast.metal",
    "fuel-metal-kernels/src/metal_src/conv.metal",
    "fuel-metal-kernels/src/metal_src/fill.metal",
    "fuel-metal-kernels/src/metal_src/indexing.metal",
    "fuel-metal-kernels/src/metal_src/random.metal",
    "fuel-metal-kernels/src/metal_src/reduce.metal",
    "fuel-metal-kernels/src/metal_src/scaled_dot_product_attention.metal",
    "fuel-metal-kernels/src/metal_src/sort.metal",
    "fuel-metal-kernels/src/metal_src/ternary.metal",
    "fuel-metal-kernels/src/metal_src/unary.metal",
    "fuel-metal-kernels/src/metal_src/utils.metal",
];

/// Inherited files that DO carry a third-party copyright notice:
/// `(path, who, a substring of the notice that must still be present)`.
///
/// ⚠️ **The obligation here is the INVERSE of the list above.** These must KEEP
/// their notice. Deleting an upstream author's copyright line while "tidying" a
/// vendored kernel is a licence violation that nothing else in this repository
/// would report: not the compiler, not the formatter, and not a review diff that
/// is not read line by line.
const INHERITED_WITH_NOTICE: &[(&str, &str, &str)] = &[
    (
        "fuel-metal-kernels/src/metal_src/mlx_gemm.metal",
        "Apple Inc. (MLX)",
        "Apple Inc.",
    ),
    (
        "fuel-metal-kernels/src/metal_src/mlx_sort.metal",
        "Apple Inc. (MLX)",
        "Apple Inc.",
    ),
    (
        "fuel-metal-kernels/src/metal_src/quantized.metal",
        "Jeffrey Quesnelle and Bowen Peng (MIT), at LINE 2844",
        "Jeffrey Quesnelle",
    ),
];

// ---------------------------------------------------------------------------
// PART 1 — every `copyright` in the repository, accounted for
// ---------------------------------------------------------------------------

/// Every tracked file allowed to contain the word this gate searches for, with
/// the reason. Anything else reddens Part 1.
const COPYRIGHT_ACCOUNTED_FOR: &[(&str, &str)] = &[
    ("LICENSE-APACHE", "the licence text itself"),
    ("LICENSE-MIT", "the licence text itself"),
    ("fuel-core/LICENSE", "the licence text itself"),
    (
        "fuel-examples/src/bs1770.rs",
        "vendored Apache-2.0 work; asserted separately by its own arm",
    ),
    (
        "fuel-metal-kernels/src/metal_src/mlx_gemm.metal",
        "inherited notice; asserted by INHERITED_WITH_NOTICE",
    ),
    (
        "fuel-metal-kernels/src/metal_src/mlx_sort.metal",
        "inherited notice; asserted by INHERITED_WITH_NOTICE",
    ),
    (
        "fuel-metal-kernels/src/metal_src/quantized.metal",
        "inherited notice at line 2844; asserted by INHERITED_WITH_NOTICE",
    ),
    (
        "fuel-examples/examples/yolo-v8/roboto-mono-stripped.ttf",
        "vendored Roboto Mono; notice is UTF-16 inside the font's name table",
    ),
    (
        ".git-blame-ignore-revs",
        "prose ABOUT the stamping sweep, not a notice",
    ),
    (
        "docs/gaps.md",
        "prose ABOUT the provenance incident (GAP-237), not a notice",
    ),
    (
        THIS_FILE,
        "this gate necessarily names the token it searches for",
    ),
];

// ===========================================================================
// PURE PREDICATES
//
// Every rule below takes its tables as PARAMETERS rather than reading the
// consts directly. That is what lets the `constructed_*` tests drive each one
// with a case built to fail — see the module docs on latency.
// ===========================================================================

fn extension_of(rel: &str) -> Option<String> {
    Path::new(rel)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
}

fn is_source(rel: &str, source_ext: &[&str], source_paths: &[&str]) -> bool {
    match extension_of(rel) {
        Some(ext) => source_ext.contains(&ext.as_str()),
        None => source_paths.contains(&rel),
    }
}

/// A lowercase view of a file's text for Part 1.
///
/// ⚠️ **A UTF-8 read alone is not sufficient and the miss is silent.** The
/// tracked `roboto-mono-stripped.ttf` stores its copyright in the font's name
/// table as UTF-16BE, where a UTF-8 reader sees NUL-interleaved bytes and finds
/// nothing at all. A survey that could not see it would report a clean
/// repository while a vendored third-party notice sat inside it.
fn searchable_text(bytes: &[u8]) -> String {
    let mut s = String::from_utf8_lossy(bytes).to_lowercase();
    if bytes.contains(&0) {
        let wide: Vec<u16> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes(*c))
            .collect();
        s.push('\n');
        s.push_str(&String::from_utf16_lossy(&wide).to_lowercase());
    }
    s
}

/// Part 3, arm A: paths whose extension is classified neither way.
fn unclassified_paths(
    files: &[String],
    source_ext: &[&str],
    source_paths: &[&str],
    non_source_ext: &[(&str, &str)],
    non_source_paths: &[(&str, &str)],
) -> Vec<String> {
    let mut out: Vec<String> = files
        .iter()
        .filter_map(|f| match extension_of(f) {
            Some(ext) => (!source_ext.contains(&ext.as_str())
                && !non_source_ext.iter().any(|(e, _)| *e == ext))
            .then(|| format!(".{ext}  (e.g. {f})")),
            None => (!source_paths.contains(&f.as_str())
                && !non_source_paths.iter().any(|(p, _)| *p == f.as_str()))
            .then(|| format!("(no extension)  {f}")),
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Part 3, arm B: declines naming an extension no longer present in the tree.
///
/// ⚠️ The subtraction is against EVERY present extension. Subtracting only the
/// source ones would mark every legitimate non-source decline as stale.
fn stale_declines<'a>(files: &[String], non_source_ext: &[(&'a str, &str)]) -> Vec<&'a str> {
    let present: Vec<String> = files.iter().filter_map(|f| extension_of(f)).collect();
    non_source_ext
        .iter()
        .map(|(e, _)| *e)
        .filter(|e| !present.iter().any(|p| p == e))
        .collect()
}

/// The bare-inherited exemption property, over `(path, contents)` pairs.
fn bare_exemption_violations(items: &[(&str, Vec<u8>)]) -> Vec<String> {
    let token = TOKEN.to_lowercase();
    let mut out = Vec::new();
    for (rel, bytes) in items {
        let text = searchable_text(bytes);
        if text.contains(&token) {
            out.push(format!("{rel}: now carries an SPDX identifier"));
        }
        if text.contains("copyright") {
            out.push(format!("{rel}: now carries a copyright notice"));
        }
    }
    out
}

/// The with-notice exemption property, over `(path, who, needle, contents)`.
fn notice_exemption_violations(items: &[(&str, &str, &str, Vec<u8>)]) -> Vec<String> {
    let token = TOKEN.to_lowercase();
    let mut out = Vec::new();
    for (rel, who, needle, bytes) in items {
        let text = searchable_text(bytes);
        if !text.contains(&needle.to_lowercase()) {
            out.push(format!(
                "{rel}: lost its {who} notice (looked for {needle:?})"
            ));
        }
        if text.contains(&token) {
            out.push(format!(
                "{rel}: now carries an SPDX identifier over somebody else's copyright"
            ));
        }
    }
    out
}

// ===========================================================================
// ENUMERATION
// ===========================================================================

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file()
            && std::fs::read_to_string(&manifest).is_ok_and(|s| s.contains("[workspace]"))
        {
            return dir;
        }
        assert!(
            dir.pop(),
            "no Cargo.toml with [workspace] above CARGO_MANIFEST_DIR"
        );
    }
}

/// One tracked file: its path and the object id of its INDEX content.
struct Entry {
    path: String,
    oid: String,
}

/// Every tracked file, enumerated from the INDEX.
///
/// ⚠️ **ENUMERATION AND CONTENT MUST COME FROM THE SAME SOURCE, AND THE FIRST
/// VERSION OF THIS GATE GOT THAT WRONG IN A WAY THAT WAS INVISIBLE LOCALLY.**
/// It enumerated with `git ls-files` (the index) and then read each file from
/// the DISK. Those agree in a normal working tree and disagree the moment
/// anything removes a tracked file without removing it from the index — which
/// `rust-ci.yml` does deliberately: `rm -f .cargo/config.toml`, guarded
/// `if: runner.os == 'macOS'`, because that file carries `-C target-cpu=native`
/// and is wrong for a runner. The survey died on `No such file or directory`
/// for a file git had just told it was tracked.
///
/// ⚠️ Note the manifestation was macOS-only and the DEFECT is not: any step on
/// any platform that removes a tracked file without removing it from the index
/// lands here. Reading blobs out of the object store removes the disagreement
/// instead of naming one platform's instance of it — a file CI deleted from the
/// working tree is still surveyed, because its content is in the index either
/// way.
///
/// ⚠️ **AND THE FIX IS DELIBERATELY NOT A SKIP.** Skipping files absent from
/// the working tree would let a REAL deletion hide inside the same branch, and
/// noticing what is inside files is this gate's whole job.
///
/// Fails loudly if git cannot answer — an empty list must never be mistaken for
/// a clean repository.
fn tracked_entries(root: &PathBuf) -> Vec<Entry> {
    let out = Command::new("git")
        .args(["ls-files", "-s", "-z"])
        .current_dir(root)
        .output()
        .expect("git ls-files could not be run; this gate cannot enumerate and must not pass");
    assert!(
        out.status.success(),
        "git ls-files failed ({}); a gate that cannot enumerate must never report clean",
        out.status
    );

    let entries: Vec<Entry> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|r| !r.is_empty())
        .map(|rec| {
            let s = String::from_utf8_lossy(rec).into_owned();
            let (meta, path) = s
                .split_once('\t')
                .unwrap_or_else(|| panic!("unparsable `git ls-files -s` record: {s:?}"));
            let oid = meta
                .split_whitespace()
                .nth(1)
                .unwrap_or_else(|| panic!("no object id in record: {s:?}"))
                .to_string();
            Entry {
                path: path.to_string(),
                oid,
            }
        })
        .collect();

    assert!(
        entries.len() > 1000,
        "git ls-files returned only {} tracked files, which is far below this workspace's \
         known size — the enumeration is broken, not the repository",
        entries.len()
    );
    entries
}

/// Read ONE `git cat-file --batch` record: `<oid> SP <type> SP <size> LF`,
/// then `<size>` bytes, then a terminating LF.
///
/// Extracted from [`blobs`] purely for size. ⚠️ **The stdin-writing thread was
/// deliberately NOT extracted with it:** that thread is the deadlock avoidance,
/// not a tidiness choice, and it is coupled to the spawn and the read loop it
/// feeds. Record parsing carries no such coupling, which is why this is the
/// seam that is safe to cut.
///
/// Every failure here PANICS rather than yielding empty content: a gate that
/// cannot read a file it was told exists must never report clean.
fn read_batch_record<R: BufRead>(reader: &mut R, e: &Entry) -> Vec<u8> {
    let mut header = String::new();
    let n = reader
        .read_line(&mut header)
        .unwrap_or_else(|err| panic!("cat-file header for {}: {err}", e.path));
    assert!(n > 0, "cat-file ended early at {}", e.path);
    assert!(
        !header.contains(" missing"),
        "object {} for {} is missing from the store; the index and the object database \
         disagree, which is a repository problem rather than a licence one",
        e.oid,
        e.path
    );

    let size: usize = header
        .trim_end()
        .rsplit(' ')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("unparsable cat-file header for {}: {header:?}", e.path));

    let mut buf = vec![0u8; size];
    reader
        .read_exact(&mut buf)
        .unwrap_or_else(|err| panic!("cat-file body for {}: {err}", e.path));
    let mut lf = [0u8; 1];
    reader
        .read_exact(&mut lf)
        .unwrap_or_else(|err| panic!("cat-file terminator for {}: {err}", e.path));
    buf
}

/// Contents of the given entries, read from the object store in ONE
/// `git cat-file --batch` pass.
///
/// Input is written from a separate thread: `cat-file` blocks once its output
/// pipe fills, so a single thread that writes every id before reading any
/// content deadlocks on a repository this size.
fn blobs(root: &PathBuf, entries: &[Entry]) -> Vec<(String, Vec<u8>)> {
    if entries.is_empty() {
        return Vec::new();
    }

    let mut child = Command::new("git")
        .args(["cat-file", "--batch"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("git cat-file could not be run; this gate cannot read content and must not pass");

    let mut stdin = child.stdin.take().expect("cat-file stdin");
    let ids: Vec<String> = entries.iter().map(|e| e.oid.clone()).collect();
    let writer = std::thread::spawn(move || {
        for id in &ids {
            if writeln!(stdin, "{id}").is_err() {
                break;
            }
        }
        // Dropping stdin closes the pipe, which is what ends the batch.
    });

    let mut reader = BufReader::new(child.stdout.take().expect("cat-file stdout"));
    let mut out = Vec::with_capacity(entries.len());
    for e in entries {
        out.push((e.path.clone(), read_batch_record(&mut reader, e)));
    }
    let _ = writer.join();
    let _ = child.wait();

    assert_eq!(
        out.len(),
        entries.len(),
        "read {} blobs for {} tracked entries; a partial read must never be surveyed as a whole",
        out.len(),
        entries.len()
    );
    out
}

/// Blob contents for a named subset, in the order requested.
fn blobs_for(root: &PathBuf, entries: &[Entry], want: &[&str]) -> Vec<(String, Vec<u8>)> {
    let picked: Vec<Entry> = want
        .iter()
        .map(|w| {
            entries
                .iter()
                .find(|e| e.path == *w)
                .map(|e| Entry {
                    path: e.path.clone(),
                    oid: e.oid.clone(),
                })
                .unwrap_or_else(|| {
                    panic!("{w} is named in this gate's tables but is not tracked by git")
                })
        })
        .collect();
    blobs(root, &picked)
}

fn head_of_bytes(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(HEAD_BYTES)]).into_owned()
}

// ===========================================================================
// PART 2 — the population
// ===========================================================================

#[test]
fn every_tracked_source_file_carries_an_spdx_identifier() {
    let root = workspace_root();
    let entries = tracked_entries(&root);

    let exempt: Vec<&str> = INHERITED_WITHOUT_NOTICE
        .iter()
        .copied()
        .chain(INHERITED_WITH_NOTICE.iter().map(|(p, _, _)| *p))
        .collect();

    let source: Vec<Entry> = entries
        .iter()
        .filter(|e| is_source(&e.path, SOURCE_EXTENSIONS, SOURCE_PATHS_WITHOUT_EXTENSION))
        .map(|e| Entry {
            path: e.path.clone(),
            oid: e.oid.clone(),
        })
        .collect();
    assert!(
        source.len() > 1000,
        "only {} tracked source files found; the source classification is broken, not the \
         repository",
        source.len()
    );

    let missing: Vec<String> = blobs(&root, &source)
        .into_iter()
        .filter(|(path, _)| !exempt.contains(&path.as_str()))
        .filter(|(_, bytes)| !head_of_bytes(bytes).contains(TOKEN))
        .map(|(path, _)| path)
        .collect();

    assert!(
        missing.is_empty(),
        "{} tracked source file(s) carry no `{TOKEN}` in their first {HEAD_BYTES} bytes.\n\
         The root manifest licenses this workspace, so these files are not unlicensed — they \
         carry no IN-FILE evidence of it, which is what licence scanners, SBOM generators and \
         downstream vendoring read.\n\
         Add the identifier as the first line, in that file's comment syntax. Keep it on LINE 2 \
         where line 1 is a directive the interpreter reads there: a `#!` shebang, a GLSL \
         `#version`, or a PowerShell `#Requires`. In a PowerShell script leave a BLANK LINE \
         between the identifier and a following `<#` block, or the comment displaces the \
         script's comment-based help.\n\
         If a file is VENDORED from elsewhere, do NOT stamp the dual licence on it: that asserts \
         a grant its author may never have made. Add it to INHERITED_WITHOUT_NOTICE instead.\n\
         Missing: {missing:#?}",
        missing.len()
    );
}

// ===========================================================================
// EXEMPTION ARMS
// ===========================================================================

#[test]
fn inherited_files_still_carry_no_identifier_and_no_notice() {
    let root = workspace_root();
    let entries = tracked_entries(&root);
    let fetched = blobs_for(&root, &entries, INHERITED_WITHOUT_NOTICE);
    let items: Vec<(&str, Vec<u8>)> = fetched
        .iter()
        .map(|(path, bytes)| (path.as_str(), bytes.clone()))
        .collect();

    let wrong = bare_exemption_violations(&items);
    assert!(
        wrong.is_empty(),
        "{} inherited file(s) no longer match the property they were exempted FOR.\n\
         These predate the `815abd04` fork rename and were NOT authored here. They are exempt \
         precisely because stamping `MIT OR Apache-2.0` on them would assert a licence grant \
         nobody made.\n\
         If the licensing question has been ANSWERED, that is good news — stamp the identifier \
         the answer produces and remove the file from INHERITED_WITHOUT_NOTICE. Do not silence \
         this arm without doing that.\n\
         {wrong:#?}",
        wrong.len()
    );
}

/// ⚠️ The INVERSE obligation: these files must KEEP their notice.
#[test]
fn inherited_files_with_a_third_party_notice_still_carry_it() {
    let root = workspace_root();
    let entries = tracked_entries(&root);
    let want: Vec<&str> = INHERITED_WITH_NOTICE.iter().map(|(p, _, _)| *p).collect();
    let fetched = blobs_for(&root, &entries, &want);

    let items: Vec<(&str, &str, &str, Vec<u8>)> = INHERITED_WITH_NOTICE
        .iter()
        .zip(fetched.iter())
        .map(|((rel, who, needle), (_, bytes))| (*rel, *who, *needle, bytes.clone()))
        .collect();

    let lost = notice_exemption_violations(&items);
    assert!(
        lost.is_empty(),
        "{} vendored file(s) lost the third-party notice that must travel with them.\n\
         Removing an upstream author's copyright line is a licence violation, and nothing else \
         here would report it.\n\
         Restore the notice. If a file was genuinely rewritten from scratch and no longer derives \
         from that work, say so in the commit and move it out of INHERITED_WITH_NOTICE.\n\
         {lost:#?}",
        lost.len()
    );
}

/// ⚠️ THE ASSERTION READS THE IDENTIFIER LINE, NOT THE HEAD. The first version of
/// this arm asserted `!head.contains("MIT")` and **failed at rest** — because the
/// file explains its own exemption in prose directly beneath the identifier:
/// *"stamping `MIT OR Apache-2.0` here would assert a licence grant that does not
/// exist."* **The explanation of why the file is not MIT contains the string
/// MIT**, and a head-wide search cannot tell an ASSERTION from an EXPLANATION of
/// the same token.
#[test]
fn the_vendored_files_are_still_apache_only() {
    let root = workspace_root();
    let entries = tracked_entries(&root);
    let want: Vec<&str> = APACHE_ONLY_FILES.iter().map(|(p, _)| *p).collect();
    let fetched = blobs_for(&root, &entries, &want);

    for ((rel, why), (_, bytes)) in APACHE_ONLY_FILES.iter().zip(fetched.iter()) {
        let head = head_of_bytes(bytes);
        let line = head
            .lines()
            .find(|l| l.contains(TOKEN))
            .unwrap_or_else(|| panic!("{rel} carries no `{TOKEN}` line at all ({why})"))
            .trim()
            .to_string();

        assert!(
            line.contains("Apache-2.0"),
            "{rel} must carry `Apache-2.0` ({why}). Its identifier line reads: {line:?}"
        );
        assert!(
            !line.contains("MIT"),
            "{rel}'s identifier line now claims MIT: {line:?}\n\
             Reason it is Apache-only: {why}.\n\
             Commit ef59fc23 exists precisely because an earlier sweep made this mistake, with \
             the subject \"my SPDX sweep asserted a grant that does not exist\". This is a LEGAL \
             claim, not a formatting one. Revert it."
        );
    }
}

// ===========================================================================
// PART 1 — unscoped, whole-file, every extension
// ===========================================================================

#[test]
fn no_tracked_file_carries_an_unaccounted_copyright_notice() {
    let root = workspace_root();
    let entries = tracked_entries(&root);

    let unexpected: Vec<String> = blobs(&root, &entries)
        .into_iter()
        .filter(|(path, _)| {
            !COPYRIGHT_ACCOUNTED_FOR
                .iter()
                .any(|(p, _)| *p == path.as_str())
        })
        .filter_map(|(path, bytes)| {
            let n = searchable_text(&bytes).matches("copyright").count();
            (n > 0).then(|| format!("{path} ({n} occurrence(s))"))
        })
        .collect();

    assert!(
        unexpected.is_empty(),
        "{} tracked file(s) carry a copyright notice that is not accounted for.\n\
         A third-party notice means the file is NOT ours to license under the workspace default, \
         and the identifier we would otherwise stamp on it would be a grant nobody made.\n\
         Establish the file's provenance FIRST, then either derive its SPDX identifier from its \
         actual upstream LICENSE, or add it to INHERITED_WITH_NOTICE so the notice is ASSERTED \
         rather than merely tolerated. If the match is prose ABOUT licensing rather than a \
         notice, name it in COPYRIGHT_ACCOUNTED_FOR with that reason.\n\
         Unaccounted: {unexpected:#?}",
        unexpected.len()
    );
}

// ===========================================================================
// PART 3 — the extension nobody thought about
// ===========================================================================

#[test]
fn every_tracked_extension_is_classified() {
    let root = workspace_root();
    let files: Vec<String> = tracked_entries(&root).into_iter().map(|e| e.path).collect();

    let unclassified = unclassified_paths(
        &files,
        SOURCE_EXTENSIONS,
        SOURCE_PATHS_WITHOUT_EXTENSION,
        NON_SOURCE_EXTENSIONS,
        NON_SOURCE_PATHS_WITHOUT_EXTENSION,
    );
    assert!(
        unclassified.is_empty(),
        "{} tracked extension(s)/path(s) are classified neither SOURCE nor NON-SOURCE.\n\
         This is the arm that fires on the language nobody thought about. Part 2's requirement is \
         keyed on an EXPLICIT list, which means a new extension is covered by NO rule until \
         somebody classifies it — the gate would otherwise stay green while the repository grew \
         an entire unstamped file type.\n\
         Decide: if these carry authored source, add the extension to SOURCE_EXTENSIONS and stamp \
         the files. If they do not, add them to NON_SOURCE_EXTENSIONS WITH A REASON, so the \
         decision is on the record rather than an omission.\n\
         Unclassified: {unclassified:#?}",
        unclassified.len()
    );
}

#[test]
fn no_decline_names_an_extension_that_no_longer_exists() {
    let root = workspace_root();
    let files: Vec<String> = tracked_entries(&root).into_iter().map(|e| e.path).collect();

    let dead = stale_declines(&files, NON_SOURCE_EXTENSIONS);
    assert!(
        dead.is_empty(),
        "NON_SOURCE_EXTENSIONS declares {} extension(s) no longer present in the repository: \
         {dead:?}.\n\
         Remove them. A classification table that outlives its subjects is a list of permissions \
         nobody is checking, and it silently widens what this gate tolerates the day a file with \
         that extension comes back.",
        dead.len()
    );
}

// ===========================================================================
// CONSTRUCTED NEGATIVE CASES
//
// ⚠️ Every arm above is LATENT: this tree supplies no failing input for any of
// them, so none can be shown to work by running it. Each test below builds the
// failure the real arm exists to catch, and asserts the same predicate reports
// it. Without these, "the arm is wired" and "the arm is absent" are
// indistinguishable from any green run.
// ===========================================================================

#[test]
fn constructed_an_unclassified_extension_reddens_part_3() {
    let files: Vec<String> = vec![
        "src/lib.rs".into(),
        "docs/readme.md".into(),
        "kernels/gemm.cu".into(), // the language nobody thought about
    ];

    let found = unclassified_paths(
        &files,
        SOURCE_EXTENSIONS,
        SOURCE_PATHS_WITHOUT_EXTENSION,
        NON_SOURCE_EXTENSIONS,
        NON_SOURCE_PATHS_WITHOUT_EXTENSION,
    );
    assert_eq!(
        found.len(),
        1,
        "the absence arm must report exactly the unclassified extension; got {found:?}"
    );
    assert!(found[0].starts_with(".cu"), "got {found:?}");

    // Control: the same population WITHOUT the new extension is clean, so the
    // arm is reporting `.cu` rather than reporting everything.
    let clean = unclassified_paths(
        &files[..2],
        SOURCE_EXTENSIONS,
        SOURCE_PATHS_WITHOUT_EXTENSION,
        NON_SOURCE_EXTENSIONS,
        NON_SOURCE_PATHS_WITHOUT_EXTENSION,
    );
    assert!(clean.is_empty(), "control: classified paths must not flag");
}

#[test]
fn constructed_an_unclassified_extensionless_path_reddens_part_3() {
    let files: Vec<String> = vec!["Dockerfile".into()];
    let found = unclassified_paths(
        &files,
        SOURCE_EXTENSIONS,
        SOURCE_PATHS_WITHOUT_EXTENSION,
        NON_SOURCE_EXTENSIONS,
        NON_SOURCE_PATHS_WITHOUT_EXTENSION,
    );
    assert_eq!(found.len(), 1, "got {found:?}");
    assert!(found[0].contains("Dockerfile"), "got {found:?}");
}

#[test]
fn constructed_a_stale_decline_reddens_and_a_live_one_does_not() {
    let files: Vec<String> = vec!["a/b.png".into(), "c/d.rs".into()];

    // A decline naming an extension that IS present must NOT be reported. This
    // is the half that breaks if the subtraction is against the source set
    // instead of against every present extension.
    let live: &[(&str, &str)] = &[("png", "binary image asset")];
    assert!(
        stale_declines(&files, live).is_empty(),
        "a decline for a PRESENT non-source extension must not read as stale; otherwise the gate \
         punishes recording the reason"
    );

    let stale: &[(&str, &str)] = &[("png", "binary image asset"), ("bmp", "gone from the tree")];
    let dead = stale_declines(&files, stale);
    assert_eq!(dead, vec!["bmp"], "the stale arm must name exactly `bmp`");
}

#[test]
fn constructed_a_stamped_inherited_file_reddens_its_exemption() {
    let planted = format!("// {TOKEN}: MIT OR Apache-2.0\nkernel void f() {{}}\n");
    let items = vec![("fake/inherited.metal", planted.into_bytes())];
    let v = bare_exemption_violations(&items);
    assert!(
        v.iter().any(|m| m.contains("SPDX identifier")),
        "planting an identifier in an inherited file must redden its exemption; got {v:?}"
    );

    let clean = vec![("fake/inherited.metal", b"kernel void f() {}\n".to_vec())];
    assert!(
        bare_exemption_violations(&clean).is_empty(),
        "control: an untouched inherited file must not flag, or the arm flags everything"
    );
}

#[test]
fn constructed_a_notice_appearing_in_an_inherited_file_reddens_its_exemption() {
    let items = vec![(
        "fake/inherited.py",
        b"# Copyright (c) 2024 Somebody Else\n".to_vec(),
    )];
    let v = bare_exemption_violations(&items);
    assert!(
        v.iter().any(|m| m.contains("copyright notice")),
        "a notice appearing in a file exempted for having none must redden; got {v:?}"
    );
}

#[test]
fn constructed_a_deleted_notice_reddens_the_inverse_exemption() {
    // The hazard this arm exists for: a "tidy-up" that strips the upstream
    // author's line out of a vendored kernel.
    let stripped = vec![(
        "fake/mlx_gemm.metal",
        "Apple Inc. (MLX)",
        "Apple Inc.",
        b"// tidied\nkernel void gemm() {}\n".to_vec(),
    )];
    let v = notice_exemption_violations(&stripped);
    assert!(
        v.iter().any(|m| m.contains("lost its")),
        "deleting a vendored notice must redden; got {v:?}"
    );

    let intact = vec![(
        "fake/mlx_gemm.metal",
        "Apple Inc. (MLX)",
        "Apple Inc.",
        b"// Copyright (c) 2024 Apple Inc.\nkernel void gemm() {}\n".to_vec(),
    )];
    assert!(
        notice_exemption_violations(&intact).is_empty(),
        "control: an intact notice must not flag"
    );
}

#[test]
fn constructed_stamping_over_a_third_party_notice_reddens() {
    let both = vec![(
        "fake/mlx_gemm.metal",
        "Apple Inc. (MLX)",
        "Apple Inc.",
        format!("// {TOKEN}: MIT OR Apache-2.0\n// Copyright (c) 2024 Apple Inc.\n").into_bytes(),
    )];
    let v = notice_exemption_violations(&both);
    assert!(
        v.iter().any(|m| m.contains("over somebody else's")),
        "claiming our licence on top of a retained third-party notice must redden; got {v:?}"
    );
}

// ===========================================================================
// PREDICATE CONTROLS
// ===========================================================================

/// Proof the scanner can SEE an absent identifier. Without this, a gate that
/// silently matched everything would pass forever and look identical to a clean
/// repository — the failure mode the `0 passed` trap describes.
#[test]
fn the_scanner_can_see_a_missing_identifier() {
    let with = format!("// {TOKEN}: MIT OR Apache-2.0\nfn main() {{}}\n");
    let without = "fn main() {}\n".to_string();

    assert!(with.contains(TOKEN), "control: a stamped head must match");
    assert!(
        !without.contains(TOKEN),
        "control: an unstamped head must NOT match — if this fires, the predicate matches \
         everything and the population test above is vacuous"
    );
}

/// Proof Part 1's predicate can see a notice INCLUDING past [`HEAD_BYTES`] and
/// INCLUDING UTF-16 — the two ways this survey could have been silently blind.
#[test]
fn the_copyright_scanner_can_see_a_notice() {
    assert!(
        !searchable_text(b"fn main() {}\n").contains("copyright"),
        "control: a clean file must NOT match, or Part 1 flags everything"
    );

    let mut deep = vec![b' '; HEAD_BYTES * 8];
    deep.extend_from_slice(b"// Copyright (c) 2023 Somebody Else.");
    assert!(
        searchable_text(&deep).contains("copyright"),
        "control: a notice {} bytes in must still be found — this is the quantized.metal case \
         (line 2844) and the whole reason Part 1 is not head-scoped",
        HEAD_BYTES * 8
    );

    let wide: Vec<u8> = "Copyright 2015 The Roboto Mono Project Authors"
        .encode_utf16()
        .flat_map(|u| u.to_be_bytes())
        .collect();
    assert!(
        searchable_text(&wide).contains("copyright"),
        "control: a UTF-16BE notice must be found — a UTF-8-only read sees NUL-interleaved bytes \
         and reports a clean file"
    );
}

/// A permanent guard against the WRONG fix for the index/disk mismatch above.
///
/// The tempting repair when a tracked file is missing from the working tree is
/// to skip it. ⚠️ **A skip is green forever and hides a real deletion inside
/// the same branch** — the survey would report a clean repository having never
/// looked at the file. This asserts Part 1 reads exactly as many blobs as git
/// enumerated, so introducing a skip reddens here instead of going quiet.
#[test]
fn part_1_surveys_every_tracked_file_and_skips_none() {
    let root = workspace_root();
    let entries = tracked_entries(&root);
    let surveyed = blobs(&root, &entries);

    assert_eq!(
        surveyed.len(),
        entries.len(),
        "Part 1 surveyed {} of {} tracked files. A survey that silently drops files reports a \
         clean repository it never read, which is exactly the failure this gate exists to \
         prevent. If a file cannot be read, FAIL on it — do not skip it.",
        surveyed.len(),
        entries.len()
    );
}

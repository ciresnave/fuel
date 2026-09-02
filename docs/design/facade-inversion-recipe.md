# Facade-inversion repoint — the recipe

**Status:** MEASURED on `fuel-datasets`, 2026-09-02. One crate, landed and green.
**Purpose:** de-risk the remaining repoints, of which `fuel-nn` is the expensive one:
**213 code lines carrying 215 symbol references**, plus 6 doc-comment lines
(219 lines contain `fuel::` in total). Those three numbers are the same crate
counted three ways — see "What surprised me" below. This recipe is the pilot's actual product; the crate itself
was the cheapest available subject, not the point.

Stage 1 made `fuel` a real crate that facades `fuel-core`. A repoint rewrites
`fuel::X` to `fuel_core::X` in a consumer and swaps that consumer's dependency
edge, so it no longer reaches the tensor API through the facade.

---

## ⚠️ What transfers from this pilot, and what does not

**A pilot's PROCEDURAL output generalises. Its MEASUREMENTS do not.** State which
half you are citing.

**TRANSFERS** — the steps, the prerequisites, the traps, the gate set: the
workspace-dependency prerequisite, the `lazy` / `lazy_` delimiter trap, the
requirement for two compile gates plus a text sweep, reading artifacts instead of
progress lines, and gating optional consumers under their features.

**DOES NOT TRANSFER** — any ratio measured here. `fuel-datasets` is **8/8 fenced,
the only 100% in the set**, which makes it the *least* representative crate for
the fenced-vs-prose question, and a claim generalised from it was wrong within
hours. ⚠️ **The pilot was chosen for being cheapest; cheapest is not
representative, and the two are unrelated properties.**

**The structural part worth remembering: this failure is invisible from inside the
pilot, because everything the pilot measured was correct.** No control on the
pilot's own work can catch it — only the population can, and avoiding measuring
the population is the entire point of a pilot. **The cheap defence is not a second
pilot: when a pilot produces a claim about a PROPERTY, spot-check that property on
ONE other member of the population before generalising.** One extra crate here
would have shown 1/5 prose on `fuel-nn` and killed the generalisation on the spot.

---

## The procedure, in the order that works

### 0. Decide whether the crate is eligible AT ALL

Count MODEL references (`fuel::lazy_<model>`) separately from TENSOR references
(everything else). A repoint is **wrong** for a model reference until Stage 2
completes, because `fuel::lazy_bert` keeps working through the facade's glob
while `fuel_core::lazy_bert` stops existing once the module moves.

⚠️ **The discriminator is a delimiter trap and it runs the expensive way.**

- `fuel::lazy::` — the 25,926-line TENSOR module, which **stays**
- `fuel::lazy_bert::` — a MODEL module, which **moves**

`lazy` is a **prefix** of `lazy_bert`, so a naive `fuel::lazy` matches both,
misclassifies the tensor API as models, and inflates "must wait". Match
`fuel::lazy_[a-z]` and positive-control it against a file known to contain one
(`fuel-examples/examples/bert/main.rs` has `use fuel::lazy_bert::...`). A zero
from an unverified pattern is the shape of a broken query, and here it is a null
that would license action.

### 1. Check the dependency exists as a workspace dependency

⚠️ **This is the step nobody predicts, and it fails at manifest-parse time.**
`fuel-core` is a workspace **member** (it appears in `members` and
`default-members`) but was **not** a workspace **dependency** — only the facade
depended on it, by direct path. `fuel-core = { workspace = true }` therefore
fails until the root manifest declares it.

Added once, in the pilot, and every later repoint inherits it:

    # root Cargo.toml, [workspace.dependencies]
    fuel-core = { path = "./fuel-core", version = "0.10.3" }

Chosen over a direct path dependency because `workspace = true` is the dominant
convention — 4 of the 5 crates declaring `fuel-ir` use it. The one precedent for
a direct path dep is the facade itself, which is a special case.

### 2. Confirm every symbol you are repointing TO actually exists

Measured for this crate: `fuel_core::Error` and `fuel_core::Result` at
`fuel-core/src/lib.rs:309`, re-exported from `fuel_ir::error`; `bail!` at
`fuel-core/src/error.rs:16`, `#[macro_export]`.

⚠️ **Macros do not travel through a glob re-export.** `fuel/src/lib.rs` carries
`pub use fuel_core::*` AND a separate `pub use fuel_core::bail;`, with a comment
saying exactly why. A repoint to `fuel_core::bail!` is fine because that is where
the macro is defined — but do not assume a symbol reachable through the facade is
reachable the same way, and do not assume the reverse either.

### 3. Swap the edge, then the references

In the crate manifest, `fuel = { workspace = true }` becomes
`fuel-core = { workspace = true }`.

Then replace `fuel::` with `fuel_core::`. A plain string replace is safe here and
a regex is not needed: `fuel::` cannot occur inside `fuel_core::`, so the
substitution cannot compound. Guard it anyway with an assertion that every
occurrence is preceded by a non-identifier character, so a hypothetical
`myfuel::` cannot be silently corrupted.

---

## The two gates, and why one is not enough

⚠️⚠️ **`cargo check --all-targets` DOES NOT COMPILE DOCTESTS.** For this crate
that is 8 of 13 references — the majority — sitting inside `no_run` fences as
hidden lines such as `# Ok::<(), fuel::Error>(())`.

**This was MEASURED, not assumed.** One doctest reference was reverted to
`fuel::Error` and both gates were run against the identical tree:

| gate | sabotaged tree | artifact |
|---|---|---|
| `cargo check -p <crate> --all-targets` | **GREEN** | emitted `Checking fuel-datasets v0.10.3`, so the crate genuinely recompiled and still did not see it |
| `cargo test --doc -p <crate>` | **RED** | `error[E0433]: cannot find module or crate 'fuel'` |

⚠️ **Two gates were enough for THIS crate because all 8 of its doc references are
fenced. They are not enough in general** — see "What surprised me" below, where the
prose portion of the doc column has no compile gate at all and needs a text sweep.
For `fuel-nn`, 5 of 6 doc references are prose.

So the required compile gate is **both**, as two invocations:

    cargo check -p <crate> --all-targets -j 4
    cargo test  --doc -p <crate>         -j 4

The restore was re-verified with a required `Compiling <crate>` line, because a
byte-identical restore does not always invalidate cargo's fingerprint, and a
sabotage that never recompiled looks exactly like a passing control.

### Gate the consumers too, under their features

`fuel-datasets` has one consumer, `fuel-examples`, declared `optional = true`
with `required-features = ["fuel-datasets"]` — so no default build gates it:

    cargo check -p fuel-examples --features fuel-datasets --all-targets -j 4

⚠️ **Read artifacts, not progress lines.** That run printed `Checking
fuel-datasets` and **no** `Checking fuel-examples`, which is indistinguishable
between "warm cache" and "never attempted". `--message-format json` separates
them: 93 `example` + 2 `lib` + 1 `custom-build` artifacts for `fuel-examples`,
16 `lib` for `fuel-datasets`, 0 errors.

---

## What surprised me

**The doc population is PART executable doctest, part prose — and the ratio
varies per crate.** It was briefed as compiler-blind. For `fuel-datasets` that is
wrong: all 8 doc references sit inside `no_run` fences, so they are executable
code with a gate of their own. ⚠️ **But I first wrote this section as "the doc
population is not prose", which is a claim about every crate derived from the one
in front of me — and the pilot crate turns out to be the MOST extreme fenced case
in the set, i.e. the least representative crate for exactly this question.** Fuel 3
refuted it; the split below is my independent reproduction of their measurement.

| crate | doc lines | inside a fence | prose |
|---|---|---|---|
| `fuel-datasets` | 8 | 8 | 0 |
| `fuel-training` | 11 | 7 | 4 |
| `fuel-onnx` | 5 | 3 | 2 |
| `fuel-inference` | 17 | 6 | 11 |
| `fuel-parallel` | 5 | 1 | 4 |
| `fuel-nn` | 6 | 1 | 5 |

⚠️⚠️ **SO THERE ARE THREE GATES OVER THAT COLUMN, NOT TWO:**

- `cargo check --all-targets` — sees **neither** portion
- `cargo test --doc` — sees the **fenced** portion, and nothing else does
- **nothing at all** — sees the **prose** portion; it needs a text sweep

**A repoint therefore needs the doctest run AND a text sweep, and which one
carries the majority flips between crates** — 8/0 fenced for `fuel-datasets`,
1/5 prose for `fuel-nn`. Reporting a doc count without splitting it says nothing
about which gate would catch it.

For `fuel-training`, which has 6 model references in doc comments against 4 in
code, this is the difference between a repoint that works and one whose majority
population is silently wrong.

⚠️ **"References" is three different constructs and they disagree.** For this
crate: **5 lines** contain `fuel::`, there are **5 string occurrences**, and there
are **7 symbol references** — because `use fuel::{Error, Result};` is one line,
one occurrence, two symbols. The applier asserted 7, measured 5, and **failed
correctly**: the edit was right and the expected number named a construct that had
never been stated. Name the construct in the same breath as the number, or a later
reader cannot tell a real disagreement from a vocabulary mismatch.

---

## Parked, with the measurement recorded so nobody re-derives it

Measured 2026-09-02. MODEL = `fuel::lazy_<model>`; TENSOR = every other `fuel::`;
MIXED = one grouped `use` carrying both kinds. **MODEL and TENSOR count SYMBOL
REFERENCES** (grouped imports expanded to their members); **the doc columns count
LINES.** Stated because the two do not agree and the difference is not visible in
the table.

| crate | MODEL | TENSOR | MIXED | doc-MODEL | doc-TENSOR | disposition |
|---|---|---|---|---|---|---|
| `fuel-datasets` | 0 | 7 | 0 | 0 | 8 | **DONE — this pilot** |
| `fuel-onnx` | 0 | 32 | 0 | 0 | 5 | PARKED — eligible, no consumer waiting |
| `fuel-parallel` | 0 | 22 | 0 | 0 | 5 | PARKED — eligible, no consumer waiting |
| `fuel-training` | 4 | 15 | 0 | 6 | 5 | BLOCKED until Stage 2 |
| `fuel-inference` | 9 | 63 | 0 | 2 | 17 | BLOCKED until Stage 2 |
| `fuel-nn` | 1 | 214 | 0 | 0 | 6 | BLOCKED — see below |

**Zero MIXED grouped imports across all six.** The expensive case — a single
`use fuel::{...}` carrying both kinds, which must be split rather than
repointed — does not occur anywhere.

`fuel-nn` is blocked by exactly one production line,
`fuel-nn/src/modules/two_proj_attention.rs:42`. The stay-or-move question for that
module is settled by the dependency graph rather than by whether it reads as a
model: `fuel-transformers` depends on `fuel-nn` and not the reverse, so moving it
would create a cycle.

⚠️ **`fuel-onnx` needs `protoc` on PATH** or it fails at **exit 101 before
compiling a line** — its `build.rs` calls `prost_build::compile_protos` and
prost-build 0.13.5 bundles no `protoc`. CLAUDE.md carries the installed directory.
This is a harness failure that reads exactly like the repoint breaking the crate.

**Sequencing note.** The three eligible crates were originally ordered as "cheap
independents first". That order is right and its stated reason is not the
load-bearing one: the split is on model-reference count, and cheapness coincides
with it here **by accident**. A crate that is both cheap and model-referencing
would be sent early by the cheapness heuristic and break. Order on the property
that decides it.

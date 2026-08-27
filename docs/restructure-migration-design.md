# Fuel restructure — migration design

**Status**: draft for review, 2026-08-26. Measured at `1b6ae698`.
**Reviewers requested**: Vulkane (live external backend), Lightbulb (external consumer of `fuel-core`), Unpopped (provider, if the FKC surface publishes), the portfolio PM (claims-against-refs).
**Author**: Fuel architect. Corrected four times before landing; see §10.

---

## 0. What this document is, and what it is not

**It is not a redesign.** The destination is already ratified in
[`docs/architecture/02-layers.md`](architecture/02-layers.md) v0.5 — Foundation
boundaries, the Models tier, the format/interchange tiers, the stopping rule for
new crates, and the `fuel-core` dissolution trajectory. That document is the
constitution and this one does not amend it.

**It is the path.** 02-layers pins *where things go*; nothing until now pinned
*how we get there from a 184,000-line `fuel-core`*, in what order, what breaks at
each step, and what each external consumer has to do. Two lanes are stopped
waiting on exactly that.

Where this document proposes something 02-layers does not already say, it is
marked **NEW** and carries its own justification against the stopping rule.

---

## 1. What is already decided, what has landed, what is new

### Ratified in 02-layers v0.5 (not re-opened here)

- Foundation crate boundaries: `fuel-ir`, `fuel-graph`, `fuel-memory`,
  `fuel-dispatch`, `fuel-hardware`, `fuel-backend-contract`, plus post-fission
  `fuel-tensor` + `fuel-autograd`.
- Models tier: `fuel-model-core` (the `Model` trait, the registry,
  `AutoModel::from_path`, link-time distributed registration) + `fuel-model-*`
  leaves, with `fuel-transformers` as an optional umbrella.
- Generic building blocks (RoPE, RMSNorm, GQA attention, SwiGLU MLP) stay in
  **`fuel-nn`**, not in per-family crates.
- **The stopping rule**: a new crate is justified only when a class of consumer
  uses one side and not the other; *the split must be in the dependency graph,
  not just the file layout*; speculative splits are rejected.
- **The reason `fuel-core` must go at all**: the name collides with an existing
  crates.io crate. This is a publish blocker, not an aesthetic preference.
- The stopping rule applied to models: **high-demand architectures split now,
  the long tail extracted lazily.** Big-banging every model into its own crate
  is explicitly rejected.

### Landed (B0.1–B0.5)

`fuel-core-types` → **`fuel-ir`** · hardware discovery → **`fuel-hardware`** ·
`fuel-storage` → **`fuel-memory`** · backend-contract traits →
**`fuel-backend-contract`**.

### Not landed — and it is the whole remaining job

**`fuel-core` itself has not dissolved.** 02-layers describes it as "an umbrella
facade [that] dissolves into the crates it re-exports." It *is* a facade — 192
`pub use` / `pub mod` lines in `lib.rs` — but it is a facade that also **contains
184,432 lines of implementation**.

### NEW, from CireSnave (2026-08-26)

1. **`fuel-model`** — a *model-authoring surface*: a text language a model author
   writes to create a model. **See §4: this name collides with the ratified
   `fuel-model-core` / `fuel-model-*` tier, which means something else.**
2. **`fuel` becomes a real facade crate** exporting the public surface.
   **See §3 — this is the migration's central lever, and it is nearly free.**
3. **`fuel-error`** as a shared crate every other crate needs, open to being
   merged with other genuine must-haves.

---

## 2. Measured current state

All figures at `1b6ae698`, counted rather than recalled.

```
36 workspace members, 26 default-members

fuel-core      184,432 lines / 220 files      <- the crate that must dissolve
fuel-dispatch  138,772 lines / 125 files
fuel-graph      48,410 lines /  32 files
fuel-ir         15,755 lines /  36 files
```

### `fuel-core` is not 184k lines of framework

```
134,800  (73%)  model definitions — 147 `lazy_*` files
 25,515  (14%)  lazy.rs — the user-facing tensor API
  8,825   (5%)  runtime bridge + the Judge
  8,111   (4%)  serving / decode (kv_block_pool, persistent_decode, inference_context)
  5,548   (3%)  everything else (31 files: nf4, device, decode_shape, telemetry, …)
  1,633   (1%)  training
```

**Three-quarters of the "dissolving crate" is a model zoo, not framework.** That
single fact does more to make this tractable than any design decision below.

### The blast radius is six crates, not thirty-six

In-tree dependents of `fuel-core`: `fuel-datasets`, `fuel-examples`, `fuel-nn`,
`fuel-onnx`, `fuel-tensor-tools`, `fuel-transformers`. Plus **Lightbulb**,
externally.

### Corrections to CLAUDE.md found while measuring

Three claims in the working agreement are false at head. Recorded here because
the working agreement is loaded into every session:

| CLAUDE.md says | Measured at `1b6ae698` |
|---|---|
| `fuel-nn` — "**Never existed.** The NN surface lives in `fuel-core`" | **Exists**: 23 files, 8,859 lines, a workspace member, and 02-layers assigns it the generic building blocks |
| `fuel-tensor-tools` "was **archived** 2026-08-01" | **Exists**: 626 lines, still in `members` |
| the `fuel-wasm-examples/*` glob "is present in both `members` and `default-members`" | **Absent**: `grep -c wasm Cargo.toml` → 0 |

Per the working agreement's own rule — *when one entry in a list you wrote is
corrected, re-measure the whole list* — the crate-name glossary needs a pass.

---

## 3. The central lever: `fuel` is already the name everyone imports

```
Root manifest:  fuel = { path = "./fuel-core", package = "fuel-core" }

Consumer imports, excluding fuel-core itself:
    fuel::         1018
    fuel_core::      18      (98.3% / 1.7%)

  fuel-examples  553 : 2      fuel-nn  219 : 0      fuel-onnx  21 : 0
  fuel-datasets   13 : 0   fuel-transformers 3 : 0  fuel-tensor-tools 2 : 1
```

**Nobody depends on the name `fuel-core`. They depend on the *paths* under
`fuel::`.**

So the dissolution does not have to be a breaking change:

> **Promote `fuel` from a manifest alias to a real facade crate that re-exports
> the same paths from the fissioned crates. `fuel-core` then empties out
> underneath it, and no consumer edits an import.**

This turns the riskiest-looking part of the restructure into the safest, and it
gives a **mechanical, checkable gate**: the facade's public path set must be
identical before and after each move. That is diffable — `cargo public-api` or an
equivalent path dump — and it is the single gate that makes every later stage
safe to attempt.

**The 18 `fuel_core::` sites are a one-line sed and should be converted first**,
so the alias is the only route in before anything moves.

---

## 4. ⚠️ A name collision to resolve before either name is fixed

CireSnave approved **`fuel-model`** for *"a text language that model authors can
enter as input to create a model."*

02-layers v0.5 already specifies **`fuel-model-core`** and **`fuel-model-*`** —
the `Model` trait, the architecture registry, `AutoModel::from_path`, and one
crate per model architecture. **That is a different thing**: the *definitions and
registry*, not an authoring language.

`fuel-model` and `fuel-model-core` sitting side by side, meaning
"authoring surface" and "model registry", is the kind of thing that reads fine in
a design document and is unusable in an import list.

**Options, cheapest first:**

| | Authoring surface | Registry + definitions |
|---|---|---|
| **A** | `fuel-author` | `fuel-model-core` / `fuel-model-*` *(unchanged, ratified)* |
| **B** | `fuel-model` | `fuel-arch-core` / `fuel-arch-*` *(renames a ratified tier)* |
| **C** | `fuel-model-lang` | `fuel-model-core` / `fuel-model-*` *(both keep the prefix; still adjacent)* |

**Recommendation: A.** It leaves a ratified tier alone, and *author* names the
act rather than the artifact — which is what distinguishes the two things.
**This is CireSnave's call and it should be made before either name is written
into a manifest**, because a crate name is the one decision that is expensive to
revisit after publish.

**Registry checked 2026-08-26** (raised by the portfolio PM, re-verified here
against `index.crates.io` with HTTP status codes and a negative control —
`fuel-nonexistent-xyzzy-control` returns 404, the same as the free names, so the
instrument discriminates in both directions):

```
TAKEN   fuel         0.1.0,    1 version,  live     <- see the warning below
TAKEN   fuel-core    0.48.2, 125 versions, live     <- the unrelated blockchain client
FREE    fuel-author · fuel-model · fuel-model-core · fuel-tensor · fuel-error
```

**Every candidate name in §4 is free, so availability does not decide the
collision — the argument there is purely semantic.**

### ⚠️ But `fuel` itself is taken, and §3's lever is named after it

`fuel 0.1.0`, one version, no successor, unyanked — the classic squat shape, and
crates.io does not reclaim those on request.

**This does not break the lever; it splits it in two, and the distinction is the
PM's:**

- **In-tree, the lever is untouched.** `fuel = { path = "./fuel-core", package =
  "fuel-core" }` is a *local rename*; the registry is irrelevant to it. Every
  claim in §3 about imports not changing holds exactly as measured. Lightbulb
  consumes Fuel **by git rev**, not from the registry, so their guarantee holds
  too.
- **At publish, the facade cannot be called `fuel` on crates.io.** The published
  name and the path root can differ, and **the consumer controls the path root
  themselves**: `fuel = { package = "<published-name>", version = "…" }` in their
  manifest keeps every `use fuel::…` working. So this costs a name, not the
  design.

**And note what the 1018-vs-18 measurement does and does not license.** It is a
fact about **paths**. `fuel` as a *registry name* is a different axis, and the
ratio speaks to neither its availability nor its publishability. **A true
measurement attached to a wider claim than it supports** — the defect this
document's own §6 is full of, arriving in its headline argument. The lever
survives; the sentence needed splitting.

---

## 5. The ordering

Four stages. **Each is independently valuable and independently revertible**, and
no stage begins before its predecessor's gate is green.

### Stage 1 — Make `fuel` real (small, unblocks everything)

Convert the 18 `fuel_core::` sites to `fuel::`. Create `fuel` as a real crate
whose `lib.rs` re-exports exactly what `fuel-core/src/lib.rs` exports today.
`fuel-core` becomes its only dependency.

**Gate**: the public path set is byte-identical before and after.
**Consumer impact**: none.

### Stage 2 — Move the model zoo out (73% of the problem, mechanical)

Move the 147 `lazy_*` files into the Models tier. **As a group, behind the
existing `fuel-transformers` umbrella — not into 147 crates.** 02-layers'
stopping rule rejects that big-bang explicitly; leaves get extracted lazily when a
real single-model consumer appears.

`fuel-core` goes from 184k to ~50k lines. These files are leaf consumers of the
tensor API; they are moved, not rewritten.

**Gate**: facade path set unchanged; `fuel-examples` (553 `fuel::` imports, the
heaviest consumer) compiles untouched.

**And this is the first real test of the §4 authoring surface**: whatever
`fuel-author` ends up being, it has to be able to express what already exists in
`lazy_flux.rs` and `lazy_mmdit.rs`. **147 worked examples is a better spec than
any greenfield design**, and if the surface cannot express them, the surface is
wrong.

### Stage 3 — Fission the ~50k remainder

Now a normal-sized refactor, and only now are the boundaries visible:

| Content | Destination | Note |
|---|---|---|
| `lazy.rs` (25.5k) | **`fuel-tensor`** | ratified; Lightbulb is the named consumer |
| runtime bridge (`pipelined_bridge`, `factories`, `scheduling`) | **`fuel-dispatch`** | joins the ranker/executor already there |
| the **Judge** (4.4k) | **open — see §6.3** | a profiler, not a dispatcher |
| serving / decode (8.1k) | **open** | plausibly `fuel-inference` (exists) |
| `train.rs` | **`fuel-training`** (exists) | |
| the rest (5.5k, 31 files) | distribute | `nf4`→`fuel-quantized`, `device`→`fuel-hardware`, … |

**No `fuel-autograd` split is proposed here.** 02-layers names it, but the
stopping rule requires a consumer that wants one side and not the other, and that
pressure should be demonstrated at the time rather than assumed now.

### Stage 4 — Retire `fuel-core`

The crate is empty. Delete it, drop the manifest alias, publish under `fuel`.

**`fuel-error` (NEW)** lands wherever it is first genuinely needed by two crates
that do not otherwise depend on each other — **not up front.** The stopping rule
applies to it exactly as to everything else, and "every crate needs it" is a
prediction until two crates need it and cannot share a nearer home.

---

## 6. Where does a trait go when its crate dissolves?

Three instances arrived independently, which is why this is a section and not
three cases.

### 6.1 `Realizer` — the internal instance

`pub trait Realizer` lives in `fuel-core/src/factories.rs:53`. Two implementors
(`BridgeRealizer`, plus one test stub), **zero external implementors across all
12 portfolio repos**, and it is not named outside `fuel-core`.

**It follows its implementor into `fuel-dispatch` at Stage 3.** `pub` here is a
visibility keyword, not a distribution fact, so this is a move and not a
publication event.

**Live interaction with ROADMAP item 7**: that work is adding
`last_kernel_revision()` to this trait *now*. The lane has been told explicitly
not to let the restructure change what they build — a lane optimising for a design
that does not exist yet would be guessing, and the guess would land in code the
design then has to accommodate.

### 6.2 Lightbulb's `impl From<CudaDevice> for Device` — the external instance

**Lightbulb's dependency on `fuel-core` is a trait impl, not a function call.**
Their port question is *"where does that impl live in the new shape?"*, and the
design must **answer** it rather than leave it derivable.

**Answer**: `Device` moves to `fuel-hardware` (`device.rs`, 493 lines, Stage 3);
`CudaDevice` is `fuel-cuda-backend`'s. The impl is a cross-crate bridge, so it
lives with whichever side is closer to being the orphan — **`fuel-cuda-backend`**,
which already depends on the hardware layer.

**Under §3's lever Lightbulb's import does not change at all** if they reach it
through `fuel::`. If they name `fuel_core::` directly, they are in the 1.7% and
we owe them a one-line change and advance notice. **They should confirm which,
against their own tree, rather than us inferring it.**

### 6.3 `fkc::verify`'s two device roles — the provider instance

Vulkane measured the seam at `7954f73c`: `KernelInvoker` and `HostTensor` are
already backend-neutral; the coupling is `&CudaDevice` at the entry points, plus
`fkc::verify` being private. `Storage` is a separate, larger coupling that must
not be conflated with this one.

**This is GAP-236 and it is CireSnave's decision, not this design's**, but the
restructure changes its shape: publishing a provider-facing surface is a very
different act when it lives in a crate about to be renamed and republished.
**Whatever is decided should be decided before Stage 4, not after.**

And the trap Vulkane caught stands regardless: **a trait with one implementor
used on both sides of a comparison launders a dependency as an abstraction.** If
the verify seam is published, the two device roles need *structural or observable*
independence — different traits, or a verdict that records which implementation
realized the reference.

---

## 7. Gates

Each stage is gated, and the gates are chosen so that a false green is hard:

1. **Facade path-set diff** — the public path set under `fuel::` is byte-identical
   before and after every move. This is the load-bearing gate and it is
   mechanical.

   ⚠️ **AND A PATH-SET DIFF IS NOT SUFFICIENT — Lightbulb's catch, 2026-08-26, on
   a gate that would have encoded a false claim.** A path diff compares **names**.
   A re-export resolving to a **different type** at the same path passes it:
   `fuel::Device` could point at `fuel_core::device::Device` before and a
   newly-written `fuel_hardware::Device` after, and **Fuel compiles while a
   downstream `impl From<CudaDevice> for Device` silently targets a different
   type.** The gate as first written claimed *API stability* and delivered *name
   stability*.

   **Closing it, cheapest first:**

   - **Type-identity assertions for every type an external consumer names** — one
     line each, compile-time, no runtime cost:
     ```rust
     const _: () = { fn _same(x: fuel::Device) -> fuel_hardware::Device { x } };
     ```
     Compiles **only** if the two are literally the same type. Add one per
     externally-named type at Stage 1, before anything moves.
   - **Diff rendered item signatures with fully-qualified defining paths**
     (`cargo public-api`), not the bare set of paths.
   - **A downstream compile fixture is what actually closes it.** The mechanical
     gates narrow the hole; only compiling a consumer's usage against the facade
     shuts it. Saying that plainly is better than claiming the diff covers it.
2. **`fuel-examples` compiles untouched** — 553 `fuel::` imports, the heaviest
   consumer in the tree, and per the working agreement it is a *constructing*
   crate for most model weights structs, so it catches field-level breakage a
   `--lib` gate cannot see.
3. **`--all-targets` per moved crate**, not `--lib`: a move that compiles the
   library and breaks `tests/` is the documented failure mode.
4. **CI stays green across all 10 jobs.** It went green for the first time in 995
   runs tonight; a restructure that reds it is not "temporarily" red.
5. **No `cargo check --workspace` claim without naming the exclusions** —
   `fuel-metal-backend`, `fuel-metal-kernels` (wrong platform),
   `fuel-mkl-cpu-backend`, `fuel-aocl-cpu-backend` (missing SDKs),
   `fuel-cuda-backend` (unconditional kernel forge).

**Not a gate, deliberately**: "the tests pass." Every stage here is a move, and a
move that preserves behaviour while breaking a boundary passes its tests. The
path-set diff is what catches that; the test suite is not.

---

## 8. What this deliberately does not propose

- **No `fuel-autograd` split** — no demonstrated consumer (§5, Stage 3).
- **No 147-crate model explosion** — rejected by the ratified stopping rule.
- **No `fuel-error` up front** — it lands when two crates need it and cannot
  share a nearer home.
- **No changes to `fuel-dispatch`'s 138k lines.** It is the second-largest crate
  and dissolving it is not on any critical path. The stopping rule cuts both
  ways: absence of a consumer boundary is a reason *not* to split.
- **No trait-only crates.** CireSnave corrected this explicitly: *trait-centered*
  and *trait-only* are not the same thing, and nothing here proposes the latter.

---

## 9. Open decisions

| # | Decision | Owner | Blocking |
|---|---|---|---|
| 1 | `fuel-model` name collision — §4, recommend **A** (`fuel-author`) | CireSnave | any manifest change |
| 2 | Registry availability for every new crate name | — | Stage 4 publish |
| 3 | Judge's destination (§5, Stage 3) | architect | Stage 3 only |
| 4 | Serving/decode destination (§5, Stage 3) | architect | Stage 3 only |
| 5 | GAP-236 / publish `fkc::verify` | CireSnave | Stage 4 |
| 6 | ~~Does Lightbulb import via `fuel::` or `fuel_core::`?~~ **ANSWERED 2026-08-26 — `fuel_core::` = 0, `fuel::` = 71, positive-controlled (the same `git grep -c` returns 71, so the zero is real). Lightbulb is in the 98.3% and has NO PORT under the facade lever.** Their other direct roots, measured rather than assumed: `fuel_inference::` 2 (tests only), `fuel_cuda_backend::` 2 (1 production, `device.rs:25`), and `fuel_graph::`/`fuel_ir::`/`fuel_dispatch::` **comment-only — 5 sites, zero code**, named explicitly because they would inflate a census. | Lightbulb | ✅ closed |

**Only #1 blocks starting**, and #6 is now closed by measurement. Stages 1 and 2 — the lever and 73% of the mass —
depend on nothing in this table except the name of a crate that does not yet
exist, and Stage 1 does not even need that.

---

## 10. Provenance

This document's four inputs each came from someone measuring rather than
agreeing, and each made it smaller:

- **Vulkane** measured the verify seam instead of asking about it, and caught that
  the symmetric-trait fix does not achieve independence.
- **Lightbulb's** dependency being a trait impl rather than a call was the PM's
  catch, and it converted a derivable detail into a question the design must
  answer.
- **The `fuel-core` blast radius** turned out to be 6 crates, and **73% of the
  dissolving crate is content, not framework** — both measured here, both
  contrary to how the job had been described.
- **Three CLAUDE.md claims are false at head** (§2), found incidentally while
  establishing the current state.

- **Lightbulb** answered §9 #6 from measurement with a positive control, and
  caught that the load-bearing gate compared names rather than types.
- **The portfolio PM** checked registry availability for every name and found
  `fuel` itself occupied — the one name the central lever is built on.

The single most consequential fact in this document — **1018 `fuel::` versus 18
`fuel_core::`** — was a two-second query that nobody had run, and it turns the
riskiest-looking stage into the safest one.

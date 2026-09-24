# Fuel — method rules (the long form)

**This file is the EVIDENCE, not the rule.** Every entry below is indexed by a
one-line rule in [`CLAUDE.md`](../CLAUDE.md), which is what loads into every
session. The one-liner is the operative instruction; this file is why it exists,
what it cost to learn, and the measurements behind it.

**Split out 2026-08-14** because CLAUDE.md had reached ~19,600 words in 66
bullets — one of them 2,074 words — and a working agreement nobody can scan is a
working agreement nobody follows. Nothing was rewritten or summarised in the
move: every entry is verbatim, so no evidence was lost and the incident detail
stays greppable. **The 2026-08-14 criterion was METHOD bullets of >=300 words; operational
rules stayed in CLAUDE.md in full. That changed on 2026-09-16:** on CireSnave's
ruling (*"Aggressive trim please."*, relayed verbatim by the portfolio PM),
CLAUDE.md was cut from 141,563 bytes to under 35,000. Every rule keeps a
one-line operative statement there, and every removed line — operational ones
included — moved here VERBATIM: either as a new section, or under a
`### Former CLAUDE.md index text` heading at the end of its existing section.
The PR that did it (base `c4936cfc`) carries the line-by-line ledger.

⚠️ **If you are acting on a rule, read the entry — the one-liner is a pointer, not
a summary you can safely reason from.** Several of these rules exist precisely
because someone acted on a compressed version and got the scope wrong.

---

## injectivity-and-collapsed-mappings

> **Index line (in CLAUDE.md):** **Exhaustiveness gives completeness; nobody checks INJECTIVITY.** Two inputs mapping to one output is silent all the way down, and a false agreement gets FILED where a disagreement gets investigated. Demand injectivity where the output is an IDENTITY (a wire token), not where it is a CLASSIFICATION. For a decline reason: a distinction stated in prose MUST be carried in the type.

**EXHAUSTIVENESS GIVES YOU COMPLETENESS; NOBODY CHECKS INJECTIVITY — AND A COLLAPSED MAPPING AGREES WITH A REFERENCE FOR THE WRONG REASON (2026-08-12, KISS architect's framing).** A new variant falling through a catch-all is loud once you look. **Two variants mapped to the SAME output is silent all the way down: green build, complete report, two verdicts collapsed into one — and strictly worse than a disagreement, because a disagreement gets investigated and a false agreement gets filed.** **FUEL'S LIVE INSTANCE IS GAP-161:** `fuel-cuda-backend/src/storage.rs:3023-3039` spells **three distinct decline reasons** in comments and returns the **identical** `UnsupportedDtype` for all three. The precise name for that is not *"reasons erased into prose"* — it is **the reason→value mapping is not injective**, which is better because it says what instrument catches it. **⚠️ THE DISTINCTION THAT DECIDES WHERE THE INSTRUMENT IS REQUIRED: INJECTIVITY IS MANDATORY WHERE THE OUTPUT IS AN *IDENTITY*, OPTIONAL WHERE IT IS A *CLASSIFICATION*.** `sk4_token` is an identity — it NAMES the dtype on the wire, so a collision means two operand shapes share a `structure_key`, a correctness bug. `map_element_kind(DType) -> Option<ElementKind>` is a classification: **two Fuel dtypes legitimately COULD map to one backend kind if the backend does not distinguish them**, so requiring injectivity there would encode a false constraint. **Demand it of mappings onto a wire identity; do not demand it universally.** **⚠️ AND A DECLINE REASON IS A THIRD CASE THAT FITS NEITHER — many causes legitimately share one verdict, so the rule for diagnostics is NOT injectivity but: A DISTINCTION STATED IN PROSE MUST BE CARRIED IN THE TYPE** (KISS architect, KISS #167). The author knew the distinctions, wrote them down, and shipped an interface that cannot express them — **the comment claims a resolution the type does not have.** **Greppable instrument: if a comment says "distinct from X" / "NOT because", check the returned value.** Measured on Fuel: **137 candidates in production Rust → 6 hits / 5 sites within 5 lines of a decline return, positive control (`fuel-cuda-backend/src/storage.rs:3027`) surviving the narrowing, 96% exclusion; the 5-line window is a LOWER bound** because a comment can precede a long `match`. **⚠️⚠️ "ONE RETURN VALUE" IS NOT THE TEST — AND NEITHER IS "IS THE DISTINGUISHED DIMENSION A FIELD", WHICH WAS MY REFINEMENT AND WAS REFUTED BY MY OWN MEASUREMENT: both backends carry `dtype`+`op` and NEITHER carries a reason, so that test flags the CORRECT site too. THE TEST IS WHAT THE COMMENT CLAIMS ABOUT ITS OWN OUTPUT.** `fuel-metal-backend/src/storage.rs:128` joins six dtypes into one decline arm and is **CORRECT**, because `UnsupportedDTypeForOp(self.dtype, op)` is **dtype-parameterised** — six joined arms, six distinct values — and its comment claims only that the join *"asserts nothing false about the format"*, which is the weaker and honest claim. Both backends carry `dtype` and `op`; **NEITHER carries a REASON**, and GAP-161's comment distinguishes *why*. **So the defect is: CUDA asserts three reasons as facts about three declines that are ONE value — a resolution the type does not have — while Metal claims only that its join "asserts nothing false", which is weaker and TRUE. LOSSY-AND-HONEST IS FINE; LOSSY-WHILE-ASSERTING-OTHERWISE IS THE DEFECT.** **PRACTICAL SHAPE, and its bound: TWO STAGES — FILTER BY MACHINE (the grep, as measured), JUDGE BY READER (does the comment claim the output distinguishes?). Stage 2 is not mechanizable and should not pretend to be; mechanization is the floor here, not the ceiling.** **A comment that DOCUMENTS its own loss instead of denying it is rarer than one that is simply right, and more useful — it tells the next reader what they may NOT conclude.** **AND NOTE HOW FUEL'S sk4 CASE IS ACTUALLY COVERED, because it is fragile: `token_kind.rs` asserts a token **`BTreeSet`** has cardinality 14 (`:172`) while separately counting the **dtypes** that produce `Some` as 14 (`:216`) — set-cardinality on one side, population-count on the other, which IS an injectivity check. But nobody wrote "assert injective," so a future refactor can delete the coverage without knowing it existed. NAME THE PROPERTY THE ASSERTION PROTECTS.** ⚠️⚠️ **AND THE SAME FAILURE ONE LEVEL OUT, FOUND BY KISS 2026-08-14 AND WORTH MORE THAN THE BUG IT CAME FROM: AGREEMENT BETWEEN TWO *CONSUMERS* IS NOT AGREEMENT WITH THE *AUTHORITY*.** KISS's reference artifact publishes `vulkan:sg64.ops-abr.arith-f16.cm-none` — **four fields — while `spec/namespaces/vulkan.md` went to Vocabulary v4 four weeks earlier and rule V-1 requires exactly FIVE**, none omissible. Fuel's sk4 byte-match leg compared that vector and reported **20/20 green**. **THE BYTE-MATCH DID NOT FAIL — IT AGREED. Two implementations matched each other while both disagreed with the document of the maintainer who owns the vocabulary, because NEITHER INSTRUMENT WAS POINTED AT `vulkan.md`.** **A disagreement gets investigated; A FALSE AGREEMENT GETS FILED** — same asymmetry as the collapsed mapping above, and the same reason a null result reading *clean* is the dangerous direction. **THE OPERATIVE SENTENCE: A BYTE-MATCH'S ENTIRE STRENGTH IS THAT ITS EXPECTED VALUE HAS AN AUTHOR WHO IS NOT YOU — AND THAT HOLDS ONLY WHILE THE AUTHOR'S DOCUMENT AND THE ARTIFACT ARE THE SAME AGE. A match against something you did not write is only as fresh as the thing you did not write.** ⚠️⚠️⚠️ **AMENDED HOURS LATER — THE RULE ABOVE UNDERSTATES IT, AND THE KISS ARCHITECT FOUND THE STRONGER FORM BY TRACING THE STRING. FOR THE `target` FIELD THERE WERE NEVER *TWO DERIVATIONS TO AGREE*.** That token traces to a **single hand-written Rust literal**, authored once (`conformance/src/reference_vectors.rs:202`); the artifact is generated from it, the spec quotes it, and Fuel's fixture was copied from it. **Fuel interpolates the target verbatim. So the byte-match over that field COMPARED A STRING AGAINST COPIES OF ITSELF.** No instrument on either side reads the namespace doc's `Vocabulary version` header or validates a target against its grammar — **so a token that was malformed THE DAY IT WAS WRITTEN is equally invisible, and a version stamp catches the bump but not that.** **THE OPERATIVE TEST, AND IT IS PER-FIELD RATHER THAN PER-VECTOR: FOR EACH FIELD, DOES EACH PARTY *DERIVE* IT OR *COPY* IT? A conformance claim's strength is not uniform across the row it is reported for.** Here **19 of 20 vectors were byte-match evidence and the 20th was PROVENANCE evidence wearing the same shirt** — and *"20/20"* reads as twenty independent agreements. **This is the same defect as *the construct is invisible in the number*, applied to a CONFORMANCE score: the arithmetic is right, and what each unit MEASURED differs.** **AND THE INVERSION WORTH MEMORISING, because it is counter-intuitive and it decides where to spend effort: VULKANE'S SINGLE PINNED EXPECTED VALUE BEAT FOUR INDEPENDENT LEGS — because it was the only one comparing against a DOCUMENT instead of against us. ONE COMPARISON AGAINST THE AUTHORITY IS WORTH MORE THAN FOUR AGAINST EACH OTHER, and adding a fifth implementation would not have found this.** **Corollary for Fuel's own tests: a passthrough field's byte-match proves only *nobody mangles it* — a real and worth-having property (our truncation sabotage proves it discriminates) — but it is NOT a well-formedness check on the value, and must never be cited as one.** **PRACTICE: a conformance artifact must record the version of every FOREIGN vocabulary it embeds, not just the spec commit it was generated from ⚠️⚠️ **AND A PROVENANCE STAMP PROVES *BINDING*, NOT *CURRENCY* — MEASURED ON MY OWN TEST 2026-08-14, WHICH HAD BEEN PASSING THROUGHOUT.** `corpus_is_the_artifact_this_leg_was_bound_to` asserts `source_commit == KISS_SOURCE_COMMIT` plus schema, counts and set sizes. **The constant and the vendored blob were updated together, so they agree — and the test proves THEY AGREE, not that either is current.** Fuel's corpus sat at `decline_vectors: 10` while upstream had moved to 15 and then 17: **seven declines behind, green the entire time.** **SO THE ASSERTION GUARDS AGAINST SWAPPING THE FILE WITHOUT UPDATING THE CLAIM, AND IS STRUCTURALLY INCAPABLE OF DETECTING BEING BEHIND UPSTREAM.** That is the two-consumers-agreeing failure turned one step inward — **the two consumers being ME AND MY OWN CONSTANT.** **A LOCAL CHECK CAN ONLY EVER PROVE INTERNAL CONSISTENCY; CURRENCY REQUIRES COMPARING AGAINST THE SOURCE.** Note also that a per-namespace *vocabulary version* would NOT have caught this one — it was a decline-SET drift, not a vocabulary drift — so the two axes need separate detectors. **PRACTICE: for any vendored conformance artifact, either a CI step fetches the published sha and compares, or the currency question is OPEN and must be stated as open. "Our provenance assertion passes" is not an answer to "is our copy current?", and it reads exactly like one.** ⚠️⚠️ **AND THE SHARPEST FORM OF THE TEST CAME BACK FROM MLMF, WHO GENERALISED THE INCIDENT FURTHER THAN I HAD: *THE ASSERTION MUST BE "RE-HASH THE SOURCE AND COMPARE", WHICH IS FALSIFIABLE. FIELD EQUALITY IS NOT.*** Two stored constants compared against each other can only ever restate what someone wrote down; **a hash of content you can go re-read is the version that survives the incident**, because the source is a party to the check. **Use it as the discriminator on any provenance mechanism: CAN THIS ASSERTION FAIL WITHOUT A HUMAN CHANGING A CONSTANT? If not, it is a binding record wearing a currency check's clothes.** **AND THEIR EXTENSION IS THE ONE THAT SHOULD WORRY US, BECAUSE FUEL HAS THE SAME SHAPE IN MORE THAN ONE PLACE: A CONFORMANCE CORPUS FROZEN AT TODAY'S ARTIFACTS STAYS GREEN INDEFINITELY WHILE SILENTLY CEASING TO DESCRIBE THE WORLD.** Their instance: a checkpoint corpus asserting *"MLMF understands everything here"* will still pass in a year while failing to understand every metadata key shipped in the meantime — **the green decays into meaninglessness and nothing announces it.** Ours is the same object one layer in: vendored conformance vectors, golden fixtures, recorded model configs. **PRACTICE: CORPUS CURRENCY IS A SEPARATE OBLIGATION FROM CORPUS AGREEMENT AND MUST BE ABLE TO FAIL ON ITS OWN. Record where each entry came from and WHEN; make staleness its own failing check rather than something inferred from a passing one.** A green agreement test and an absent currency test are not two thirds of a guarantee — **the second one's absence silently caps what the first one can mean.** — a single `source_commit` DETECTS a change and cannot DIAGNOSE it (one value standing for many independent axes), and a vocabulary owned by someone else moves on its own clock.** **AND RECORD THE VERSION EVEN WHEN IT IS UNCHANGED: a field that appears only when something moves is indistinguishable from a field nobody remembered to write.** ⚠️ ****⚠️ THE ATTRIBUTION HERE WAS WRONG IN ITS FIRST VERSION AND THE CORRECTION IS THE MORE USEFUL RULE. I originally wrote that the exclusion was an error in MY notes which the KISS architect INHERITED FROM ME. Not so — the architect pressed the point rather than accepting a graceful offer, with three independent confirmations: my own measurement (Fuel CONSTRUCTS and compares that vector, `kiss_structure_key_byte_match.rs:290`, ZERO vulkan exclusions — Fuel was 20/20 with no exclusions at all); their dated memory file recording the exclusion against **Unpopped** BY NAME; and Unpopped owning and closing it. THE PROPAGATION WAS ENTIRELY WITHIN KISS. I VOLUNTEERED FOR A LINK IN A CHAIN I WAS NEVER IN — and was about to set it in a working-agreement file, which is precisely the artifact class this whole exchange was about: ACCURATE ABOUT THE LESSON, WRONG ABOUT THE PROVENANCE, AND NOTHING DOWNSTREAM RE-DERIVES IT.** **MY ACTUAL ERROR WAS THE MIRROR OF THE ONE I CLAIMED, AND IT IS WORTH MORE: A PEER DESCRIBED MY OWN CODE TO ME, I AGREED IT MATCHED MY NOTES, AND ONLY MEASURED LATER. Accepting someone else's characterisation OF YOUR OWN PROJECT and restating it as your own record is how a foreign error acquires local corroboration — and it is the cheapest possible thing to check, because the code is right there.** **AND THE MECHANISM THE ARCHITECT NAMED, WHICH IS SHARPER THAN 'WATCH YOUR PROPAGATION' AND APPLIES DIRECTLY TO THIS REGISTRY: A CORRECTION THAT LIVES IN A CONVERSATION AND A FACT THAT LIVES IN A FILE DIVERGE SILENTLY, AND THE FILE WINS.** Unpopped corrected them once, in conversation; the correction was never written back; the FILE is what got re-read, and the stale figure travelled from there. **NO CONTROL COULD FIRE — nothing in the repo changed at the transition, so there was no artifact to disagree with.** It was caught the only way this class ever is: **CONTRADICTION WITH AN INDEPENDENT FINDING**, when Unpopped re-measured. **THE FIX IS NOT TO BE MORE CAREFUL — IT IS TO EDIT THE FILE IN THE SAME TURN AS THE CORRECTION.**

⚠️⚠️ **THE SAME ASYMMETRY IN QUANTIFIERS, AND IT IS WHY THE INDEX-JOIN GUARD NEEDED TWO MORE ARMS (2026-09-02, self-inflicted, and only building the second arm exposed it).** Exhaustiveness is an **∃** claim — *every section is reachable from SOMEWHERE* — and ONE good instance satisfies it. **The obligation a reader actually depends on is a ∀ claim: every CITATION works.** In `fuel-ir/tests/method_rules_index_join.rs`, arm C asks the ∃ question per SECTION (*is there at least one link whose TARGET carries this anchor?*) and arm D asks the ∀ question per LINK (*does every link carry an anchor?*). **NEITHER IMPLIES THE OTHER, AND IT FAILS IN BOTH DIRECTIONS: a rule cited five times with four anchors PASSES arm C while the fifth citation still drops the reader at the top of a 1,600-line file; a section carrying no link at all PASSES arm D vacuously.** **THE MEASUREMENT ERROR THIS PRODUCED IS THE POINT: I reported "2 instances" of the anchor defect. There were 2 unanchored LINKS and 1 unanchored SECTION — right about one construct, wrong about the other, and I had named neither.** That is CLAUDE.md's *"the number was correct and the construct it counted was invisible in it"* arriving through a quantifier: **an aggregate satisfied by ONE good member is structurally unable to count its bad ones, so an ∃-shaped gate reports a figure that is not about the population anyone is relying on.** **PRACTICE: when a gate covers a RELATION, state its QUANTIFIER — per-what, and ∃ or ∀ — then ask whether the obligation you actually hold is the other one. Where both matter they are TWO ARMS, not one, and must be reported separately because they fail on disjoint defects.** **⚠️ ARM D CARRIES A NAMED KNOWN BOUND, recorded here rather than in a commit message because a bound nobody can find is a bound nobody honours: it forbids a link to `docs/method-rules.md` AS A DOCUMENT — a deliberate whole-file reference with no section intended. ZERO exist today, which is what makes it safe NOW and not safe FOREVER. If one is ever legitimately wanted, AMEND ARM D; do not contort the link to satisfy the gate.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Exhaustiveness gives completeness; nobody checks INJECTIVITY.** Two inputs mapping to one output is silent all the way down, and a false agreement gets FILED where a disagreement gets investigated. Demand injectivity where the output is an IDENTITY (a wire token), not where it is a CLASSIFICATION. For a decline reason: a distinction stated in prose MUST be carried in the type. → [`injectivity-and-collapsed-mappings`](#injectivity-and-collapsed-mappings)

## lib-does-not-build-tests

> **Index line (in CLAUDE.md):** **`--lib` does not build `tests/` ⚠️ NOR AN IN-FILE `#[cfg(test)] mod tests`, so a `--lib` gate is blind to both.** Pick the gate from where the CHANGE lives, not from what the CRATE is. The tell in the failure is `could not compile <crate> (lib test)`. When reporting a tree healthy, state the TARGET KINDS as well as the features — an unqualified "green" is the direction nobody questions. ⚠️ And `--all-targets` does NOT include doctests.

**⚠️ `--lib` DOES NOT BUILD `tests/`, SO A `--lib` GATE IS STRUCTURALLY BLIND TO EVERY INTEGRATION TEST — AND "MAIN IS HEALTHY" REPORTED OFF ONE IS A CLAIM ABOUT A TARGET KIND WEARING THE LABEL OF THE WHOLE TREE (2026-08-13, coordinator, self-inflicted).** Measured on `main`: `-p fuel-dispatch --features telemetry --lib` reported **806 passed, green**, while `--all-targets` on the same commit reported **`fuel_emits_only_recognized_sk4_dtype_spellings ... FAILED`**. The merge that introduced it had been reviewed and the tree pronounced healthy across four feature configurations — **every one of them `--lib`.** **The features were named in the report; the TARGETS never were, which is the construct-invisible-in-the-number defect committed by the person who keeps catching it in others.** **PRACTICE: gate with `--all-targets`, and when reporting a tree healthy, state the target kinds as well as the features ⚠️⚠️ **— AND `--all-targets` DOES NOT INCLUDE DOCTESTS, WHICH IS A NAMED HOLE IN THE GATE THIS PROJECT TRUSTS MOST (measured 2026-08-14: `cargo test -p fuel-inference --all-targets` emits **ZERO** `Doc-tests` sections).** `--all-targets` expands to `--lib --bins --tests --benches --examples` — **`--doc` is not in that list and cannot be added to it.** So every `--all-targets` gate in this repo is **structurally incapable of seeing doctest breakage**, and doctests are compiled code: a ` ```no_run ` fence **still compiles**, it merely does not execute. **LIVE INSTANCE, AND IT IS THE THIRD TIME THE SAME COMPLETION CLAIM WAS WRONG: `fuel-inference`'s module docs carry ` ```no_run ` examples doing `use fuel::{Device, Tensor};` — the eager API DELETED IN B6. `cargo test -p fuel-inference --doc` exits 101 with 2 FAILED.** **B6 was declared complete three times, and each miss was a TARGET KIND NOBODY ENUMERATED: first two whole crates, then feature-gated examples, now doctests.** **THE RULE THAT GENERALISES IS NOT "remember doctests" — IT IS: A COMPLETION CLAIM MUST NAME THE TARGET KINDS IT SEARCHED, because every miss so far has been a KIND, never a missed instance within a kind.** The enumeration to run against any "X is fully retired" claim: **lib · bins · tests · benches · examples · DOCTESTS · feature-gated variants of each · and non-default-member crates.** `--all-targets --workspace --all-features` still misses the sixth. **⚠️ CORRECTED 2026-08-15 BY THE COVERAGE-MATRIX MEASUREMENT — THE CORRECTION MOVES THE BLAME FROM CI TO OUR OWN HABIT. CI ALREADY RUNS DOCTESTS: its test job is `cargo test --workspace --no-fail-fast $EXCL` with **NO `--all-targets`**, and plain `cargo test` runs doctests by default. So *"nothing runs them"* was FALSE at CI level, and a new `--doc` step is NOT the recurrence fix.** **THE BLIND SPOT IS THE LOCAL VERIFICATION HABIT: both B6 closes were verified with local `--all-targets`, the one instrument that never sees doctests — AND CI, which would have caught it, WAS DOWN for the entire window (dead at dependency resolution). TWO FAILURES COMPOUNDING; neither alone would have hidden it.** `cargo test --workspace --doc` stays useful for **isolation and speed**, not because doctests are otherwise uncovered. — an unqualified "green" is the direction that does not get questioned.** ⚠️⚠️ **AND A GATE MUST MATCH CI'S *TOOLCHAIN* AS WELL AS ITS FEATURES AND TARGETS — 2026-08-14, and the instructive part is that the FIRST FIX TO THE GATE WAS ITSELF INSUFFICIENT AND ONLY A SABOTAGE OF THE REPAIR FOUND IT.** `scripts/aarch64-cross-check.ps1` existed, ran, and passed, while `fuel-quantized` carried an `E0658 stdarch_neon_dotprod` that killed both macOS CI jobs. **Two independent causes, and fixing one left the gate still green on a live defect:** **(a) IT WAS POINTED AT THE OTHER BRANCH.** `.cargo/config.toml` sets `target-cpu=generic` for `aarch64-apple-darwin` — ARMv8.0-A, **no dotprod** — so the script compiled the SOFTWARE arm, **while macOS CI DELETES `.cargo/config.toml` (the ring workaround) and compiles the HARDWARE arm.** ⚠️ **And the config's own justification was a TRUE STATEMENT WITH TOO-WIDE SCOPE — *"NEON is baseline-mandatory in ARMv8, so `generic` still compiles the NEON paths this gate exists to reach"* is true of BASELINE NEON and false of DOTPROD-GATED NEON.** The justification-scope-mismatch pattern, living in a config comment. **(b) IT RAN ON THE DEFAULT TOOLCHAIN (nightly on this box) WHILE CI RUNS STABLE — and an unstable-feature error is INVISIBLE ON NIGHTLY.** **⚠️⚠️ THE TRANSFERABLE PART: the lane fixed (a), RE-RAN THE SABOTAGE, AND THE GATE STILL SAID PASS — which is the only reason (b) surfaced. SABOTAGE-VALIDATE YOUR REPAIR TO THE GATE, NOT JUST THE GATE.** A fix to an instrument is a claim about the instrument, and it earns the same red-then-green proof as anything else; **without it a half-fix ships looking complete, and the next reader trusts a gate that now covers one of its two blind spots.** **AND THE SEVERITY ORDERING WORTH KEEPING: A GATE THAT EXISTS AND PASSES IS WORSE THAN NO GATE, BECAUSE IT GETS CITED AS COVERAGE.** An absent check is a known hole; a check aimed at the wrong branch, on the wrong toolchain, is a hole wearing a green badge. **Ask of any local gate: does it compile the same BRANCH, on the same TOOLCHAIN, with the same CONFIG-FILE state, as the CI job it stands in for? `.cargo/config.toml` presence is part of that state — CI deleting it is a configuration difference no feature flag or target triple records.** ⚠️⚠️ **AND `--lib` IS BLIND TO AN IN-FILE `#[cfg(test)] mod tests` FOR THE SAME REASON, WHICH THIS RULE DID NOT SAY AND WHICH COST A RED CI ON 2026-09-03.** The rule above is phrased around `tests/` DIRECTORIES, so a lane whose change lives entirely inside a `#[cfg(test)]` module in a `src/` file does not recognise it as applying. **`--lib` builds the library target WITHOUT `--cfg test`, so that module is never compiled at all.** **MEASURED, both invocations on the same commit:** `cargo clippy -p fuel-transformers --no-deps --lib` — **exit 0**; `cargo clippy -p fuel-transformers --no-deps --all-targets` — **exit 101**, `error: using contains() instead of iter().any()` at `fuel-transformers/src/models/lazy_qwen3_vl_text.rs:830`, `-D clippy::manual-contains` implied by `-D warnings`. **The lane’s gate returned a REAL exit 0 on a REAL compile of a tree that did not contain their change** — the `exit=0 + artifact` row of the truth table reporting genuine success about the wrong target set. ⚠️⚠️ **THE ARTIFACT THAT NAMES THIS CLASS IS IN THE FAILURE LINE ITSELF AND IS WORTH MORE THAN THE RULE: `could not compile <crate> (lib test)`.** **`(lib test)` IS THE TARGET `--lib` DOES NOT BUILD.** Anyone seeing that token after a green `--lib` gate has the diagnosis in front of them without knowing this section exists. ⚠️ **AND A SECOND, INDEPENDENT TRAP FROM THE SAME INCIDENT, WHICH IS THE ONE THAT WOULD HAVE LET THE LANE ARGUE WITH CI: `cargo test` WAS NEVER FOOLED.** It compiled and ran both new tests, 7 passed. **A green TEST run is not evidence about a LINT gate, because they build different target sets** — the lane held an artifact proving their code ran and it said nothing about the question clippy was being asked. **THE LANE’S OWN FORMULATION IS THE RULE AND IT IS BETTER THAN A LIST OF TARGET KINDS: *I chose `--lib` because the change is in a LIB CRATE, not because I asked which TARGET SET CONTAINS MY CODE.*** **PRACTICE: pick the gate from where the CHANGE lives, never from what the CRATE is; and for a lint gate specifically, match CI’s invocation — here `cargo clippy --workspace --tests --examples --benches ... -- -D warnings` (`rust-ci.yml`), for which the per-crate stand-in is `--no-deps --all-targets`.** *(Positive control that the distinction is real rather than a story about one lint: the two `fuel-hardware` warnings from GAP-267 appear in BOTH invocations, so they cannot explain the difference — and they were available as a comfortable wrong answer, which is why the lane was asked to confirm the lint was theirs BEFORE fixing anything.)*

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`--lib` does not build `tests/` ⚠️ NOR AN IN-FILE `#[cfg(test)] mod tests`, so a `--lib` gate is blind to both.** Pick the gate from where the CHANGE lives, not from what the CRATE is. The tell in the failure is `could not compile <crate> (lib test)`. When reporting a tree healthy, state the TARGET KINDS as well as the features — an unqualified "green" is the direction nobody questions. ⚠️ And `--all-targets` does NOT include doctests. → [`lib-does-not-build-tests`](#lib-does-not-build-tests)

## sabotage-calibrated-tolerances

> **Index line (in CLAUDE.md):** **A numeric oracle's tolerance must be SABOTAGE-CALIBRATED, never inherited.** Report (a) correct-path drift and (b) sabotaged divergence; set the threshold BETWEEN them; record both in-file. Poor separation is a FINDING, not a threshold to tune. A relative oracle is blind to defects in shared code; a golden nobody has seen fail is a constant.

**⚠️ A NUMERIC ORACLE'S TOLERANCE MUST BE SABOTAGE-CALIBRATED, NEVER INHERITED — AND THE AMBIENT DECODE TEMPLATE IS MEASURED NOT TO SEE A RoPE cos/sin SWAP (2026-08-13).** The decode suite's natural template asserts **`diff < 5e-3 || rel < 1e-2`**, and it is **too loose to be an oracle for the defects these ports actually make**. Two independent measurements: (1) the windowed-vs-single-mask divergence a whole GAP-029 sub-scope was priced on is **7.9e-3 at prefill / 7.04e-3–7.95e-3 at decode** — *1.6x* the abs threshold, and the `||` means the `rel` arm alone passes it outright; (2) under a deliberate sabotage, `forward_with_kv_context_decode_matches_non_cached_forward` — an **absolute** oracle at that tolerance — **PASSED with a RoPE cos/sin swap in place**, which is not a marginal defect. **THE FAILURE ARRIVES THROUGH THE TOLERANCE RATHER THAN THE ASSERTION TARGET, so it survives every check aimed at *what* is asserted.** **PRACTICE: report BOTH numbers — (a) the correct implementation's drift against the oracle, (b) the sabotaged/old-fabrication variant's divergence — and set the threshold BETWEEN them, with both recorded in the test file. If (a) and (b) are not comfortably separated, THAT IS A FINDING: report it rather than picking a threshold that makes the test pass.** For scale, the golden that did catch the rope swap sits at **1e-6**, and a behaviour-preserving extraction of a shipped decode path held to it exactly. **⚠️ AND TWO ORACLE SHAPES THAT NO TOLERANCE FIXES, because they are category errors: a RELATIVE oracle (D2 vs D1, or any A-vs-B over shared code) is STRUCTURALLY BLIND to a defect in the code both sides run — it passed under a sabotage corrupting every logit it read; and a GOLDEN NOBODY HAS SEEN FAIL IS A CONSTANT, NOT AN ORACLE, so a golden owes a red before it is cited as evidence.** ⚠️⚠️ **AND THE STRONGEST INSTANCE ARRIVED 2026-08-14 ON GEMMA3, WHERE THE DEFECT SITS *INSIDE* THE TEMPLATE RATHER THAN 1.6x OVER IT: the RoPE axis (per-layer base collapsed to a single global base) diverges at `[1.235e-3, 8.673e-4, 1.031e-3]` — about 100x the 1e-5 oracle used, and COMFORTABLY UNDER BOTH ARMS of the natural template. A DUAL-BASE DEFECT PASSES IT OUTRIGHT, with no arithmetic to notice.** The mask axis on the same model is `[5.876e-2, …]`, i.e. **40x larger**, so a single "windowing works" number would have been read as covering both. **AND THE AXIS HAD TWO INDEPENDENT ROUTES TO A SILENT GREEN — a DEGENERATE FIXTURE collapsing the input (`rope_local_base_freq == rope_theta`) and the TOLERANCE swallowing the output — so closing only one produces a test that LOOKS rigorous and proves nothing. Both must be closed, and the fixture guard must assert the DERIVED quantity (`decode_rope_plan().n_variants() == 2`), not that the config fields differ, WITH a sibling test pinning that the derived quantity tracks the fixture (equal bases → 1) — otherwise the guard is a tautology waiting for a later edit.** ⚠️⚠️ **SEPARATE AND MORE PORTABLE, FROM THE SAME LANDING — A TRUE PIECE OF EVIDENCE THAT WAS BEING CARRIED AS A PORTABLE PROOF SHAPE AND IS ACTUALLY CONFIG-DEPENDENT.** Qwen2's born-red `[0.0, 7.04e-3, 7.95e-3]` was repeatedly cited as proof of discrimination — *"position 3 is clean under BOTH bodies, so a degenerate oracle would have shown three zeros and this showed one."* **On Gemma3 all three positions diverge, and that is EQUALLY CORRECT: Qwen2's window of 4 cannot exclude anything until absolute position 4; Gemma3's window of 3 bites at position 3. THE LEADING ZERO IS A PROPERTY OF THE CONFIG (window width vs prefill length), NOT OF THE SEAM, THE PORT, OR WINDOWING.** **A CONFIG-DEPENDENT PIECE OF EVIDENCE THAT LOOKS PORTABLE IS MORE DANGEROUS THAN A WRONG ONE: the next family showing three zeros gets investigated as a degenerate oracle when it is fine, and one showing a leading zero for an unrelated reason gets read as confirmation.** **PRACTICE: before reusing a proof SHAPE across cases, derive it from the config that produced it — and if it depends on the config, say so at the site rather than in the report, because the report is not what the next family's author reads.** Caught by the lane that had been making the inference, mid-program, unprompted — the third self-caught false generalisation in one day from three different people (the others: a grep called a "lower bound" that was exact, and a positive control asserted into a spec that was never measured).

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A numeric oracle's tolerance must be SABOTAGE-CALIBRATED, never inherited.** Report (a) correct-path drift and (b) sabotaged divergence; set the threshold BETWEEN them; record both in-file. Poor separation is a FINDING, not a threshold to tune. A relative oracle is blind to defects in shared code; a golden nobody has seen fail is a constant. → [`sabotage-calibrated-tolerances`](#sabotage-calibrated-tolerances)

## vacuous-oracle-four-routes

> **Index line (in CLAUDE.md):** **A vacuous oracle arrives by FOUR routes: the tolerance, the assertion target, the FIXTURE (input collapses the axis), and the SHORT-CIRCUIT (an earlier guard answers first).** When a test asserts something is REJECTED, ask which guard did the rejecting.

**⚠️ A VACUOUS ORACLE ARRIVES BY THREE DISTINCT ROUTES, AND THE THIRD IS THE FIXTURE — WHERE THE TOLERANCE IS FINE, THE ASSERTION IS FINE, AND THE *INPUT* COLLAPSES THE AXIS UNDER TEST (2026-08-13).** Measured instances, all in one program: **(1) the TOLERANCE** — `5e-3 || rel 1e-2` sitting above a `7.9e-3` real defect, and passing a RoPE cos/sin swap under sabotage; **(2) the ASSERTION TARGET** — sampled-token equality, vacuous for a KV-dependent tiny model, where only logits discriminate; **(3) the FIXTURE** — `lazy_gemma3.rs:540` sets `rope_local_base_freq: 10_000.0` and `rope_theta: 10_000.0`, *deliberately*, with the comment *"same as global for the tables match test"*. **The two RoPE bases are EQUAL in the only Gemma3 configs the repo has, so a decode test written on that fixture PASSES UNDER A SINGLE-TABLE PORT — the dual-base defect is invisible because the input has no dual base.** **THE FIRST TWO ARE ABOUT HOW YOU MEASURE AND CAN BE CAUGHT BY CARE IN THE ASSERTION; THE THIRD IS ABOUT WHAT YOU MEASURE IT ON, AND NO CARE IN THE ASSERTION REACHES IT.** **PRACTICE: assert NON-VACUITY OF THE INPUT inside the test itself** — `assert!(rope_local_base_freq != rope_theta)`, the way the mask families assert `seq > window` and `window_wider_than_capacity_is_byte_identical_to_the_dense_mask` guards its own trap. **A born-red on a degenerate fixture is a born-green wearing the right label** ⚠️⚠️ **AND A *FOURTH* ROUTE, FOUND 2026-08-14 BY THE GAP-194 LANE WHILE DESIGNING ITS OWN CUDA TEST — THE SHORT-CIRCUIT, WHERE THE TOLERANCE, THE ASSERTION AND THE FIXTURE ARE ALL FINE AND THE PREDICATE UNDER TEST IS NEVER REACHED.** The flash-decode arm's CPU test asserts *"a windowed layer is DECLINED"* and gets a decline — **from the capability and dtype checks, which run BEFORE the window is ever consulted.** So the green is **equally consistent with the window being ignored entirely**, and the decline half of that test was **vacuous while passing**, which is the state nobody investigates. **On CUDA with a bf16 cache the earlier guards pass, and a decline becomes attributable TO THE WINDOW — the first configuration in which that assertion means what its name says.** **THE GENERAL SHAPE: WHEN A TEST ASSERTS THAT SOMETHING IS REJECTED, ASK WHICH GUARD DID THE REJECTING.** A rejection is the cheapest possible outcome to produce accidentally — **every unrelated failure produces one** — so a passing negative-case test is far weaker evidence than a passing positive-case one unless the *reason* is pinned. **Distinct from the wrong-site failure (where the trigger cannot physically reach the code): here the code IS reached, the assertion IS about the right property, and an EARLIER GUARD ANSWERS FIRST.** Practical form: **construct the case so every guard upstream of the one under test PASSES**, and if that is impossible in the cheap configuration, say so in the file rather than letting the cheap green stand in for the expensive one. **Companion decision from the same lane, and it is the right default: `expect` RATHER THAN SKIP on a missing device — a silent skip is INDISTINGUISHABLE FROM A GREEN ADMIT HALF, which is the single outcome such a test exists to rule out.** The pre-existing tests in that file take the skip route; the lane deliberately did not copy them., and a shared fixture tuned to make some *other* test convenient is exactly where this hides — the comment that collapsed the axis was itself explaining a different test's needs.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A vacuous oracle arrives by FOUR routes: the tolerance, the assertion target, the FIXTURE (input collapses the axis), and the SHORT-CIRCUIT (an earlier guard answers first).** When a test asserts something is REJECTED, ask which guard did the rejecting. → [`vacuous-oracle-four-routes`](#vacuous-oracle-four-routes)

## inherited-config-provenance

> **Index line (in CLAUDE.md):** **Ask of inherited config: has anything ever COMPILED this?** Fork-inherited lines look like decisions this project made. Provenance tells you who a line was written for; compile coverage tells you whether anyone was ever positioned to notice it was wrong. A "routine dep bump" is high-risk to feature-gated code specifically.

**Ask of any inherited config: "was this written for a repo that isn't this one?"** Two defects on 2026-08-13, same shape, found hours apart: `.gitignore:10` ignoring `Cargo.lock` (correct for Candle-the-library, costly for Fuel with a `[[bin]]`, CI, and a 30–56 minute forge — and four checkouts had silently drifted to two resolutions ~1904 lines apart), and `ci_cuda.yaml` requesting HuggingFace's runner group (correct upstream, unobtainable here). **Fork-inherited configuration is the hardest kind to see, because nothing about it looks wrong — it looks like a decision this project made, and `git log` on the line says "initial commit", which reads as settled rather than unexamined.** Both files even *documented* their own condition — the `.gitignore` comment literally said to remove the entry for an executable. **Check the provenance of a config line before defending it; "it has always been that way" and "nobody chose it" are the same observation.** ⚠️ **AND THE TRIGGER FOR THIS RULE CANNOT BE "WHEN SOMETHING LOOKS WRONG", BECAUSE NEITHER DEFECT DID — each was invisible in the direction that gets ignored.** The dead workflow presented as *a red X in a repo where red is normal*; the `Cargo.lock` ignore presented as *nothing at all*. **Neither ever looked like a question.** So the trigger has to be positional — **when you touch inherited infrastructure, ask who it was written for** — because no amount of attentiveness fires on a thing that is not displaying anything. ⚠️⚠️ **AND A THIRD INSTANCE ON 2026-08-14 SHOWS THE QUESTION ABOVE IS NOT SUFFICIENT, BECAUSE THIS ONE WAS WRONG *UPSTREAM TOO* AND WE INHERITED IT ANYWAY.** `fuel-examples`' `audio` module is written against **rubato 0.15** while the manifest requires **`"1"`**, so five example targets do not compile (`E0433 rubato::FftFixedInOut`). `git log -L` on the requirement line returns **exactly two commits in all of history, both upstream Candle**: `d365ef32` (#1865) added the resampler against `0.15.0`, and `8d5873bf` — a routine ***"Update deps"*** — bumped the requirement **`0.15.0` → `"1"` and did not touch the code.** **So the config was not "correct for a repo that isn't this one"; it was BROKEN WHERE IT WAS WRITTEN, shipped, and crossed the fork boundary intact.** ***"Was this written for a different repo?"* would have cleared it.** **THE MECHANISM IS THE SAME ONE THAT HID IT AFTERWARD, IN BOTH REPOS: `rubato` is `optional` and every target using it sits behind `required-features`, so no `cargo check`, no `--all-targets`, and no CI job in EITHER project ever compiled it. A dependency bump silently broke code upstream, and the identical structural hole downstream kept it silent here** — found only by compiling all 15 feature-gated targets one at a time (GAP-199). **THE BETTER QUESTION, WHICH SUBSUMES THE ORIGINAL: *HAS ANYTHING EVER COMPILED THIS?*** Provenance tells you who a line was written for; **compile coverage tells you whether anyone has ever been in a position to notice it was wrong** — and for `optional` deps and `required-features` targets the answer is routinely *no*, in every repo in the chain. **Corollary: a "routine dep bump" commit is a HIGH-RISK change to feature-gated code specifically, because the bump is validated by exactly the build that cannot see it.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Ask of inherited config: has anything ever COMPILED this?** Fork-inherited lines look like decisions this project made. Provenance tells you who a line was written for; compile coverage tells you whether anyone was ever positioned to notice it was wrong. A "routine dep bump" is high-risk to feature-gated code specifically. → [`inherited-config-provenance`](#inherited-config-provenance)

## validating-a-gate-means-reading-it

> **Index line (in CLAUDE.md):** **Validating a gate means reading its MESSAGE, not just its exit code** — a gate can return the right verdict with the wrong diagnosis. And a check that only PRINTS is not a gate: put it in the `&&` chain, never pipe it, never route its status through `echo`.

**VALIDATING A GATE MEANS READING ITS MESSAGE, NOT JUST ITS EXIT CODE (2026-08-08).** The sabotage discipline as carried asks only *did it go RED* — and a gate can return **the right verdict with the wrong diagnosis**, which a pass/fail-only check can never surface. Observed: a coverage gate checked for build ARTIFACTS first, and since **a FAILED compile and a NEVER-ATTEMPTED compile both produce no artifact**, it reported a genuine `E0004` as *"the compiler did not reach these crates"* — **true about artifacts, and a diagnosis pointing the reader at the harness instead of at their own code.** Companion failure from the same gate: it grepped for `Checking <crate>` lines, which **cargo does not reprint when every unit is fresh**, so it went **FALSE RED on a warm cache** — a gate nobody trusts, via a mechanism unrelated to the code under test. **Both directions must be validated (sabotage -> red, revert -> green) AND the message read in each.** **⚠️ AND THE EXACT COMPLEMENT, LEARNED SELF-INFLICTED (2026-08-12): A CHECK THAT PRODUCES A MESSAGE BUT NO EXIT CODE IS NOT A GATE AT ALL.** A registry table-integrity check printed `rows NOT in {5,6}: [('GAP-168', 11)]` **immediately before a commit** — and because it only *printed*, the `&&` chain sailed past it and pushed the corrupted row. **The instrument was correct, it fired, and it was ignored because it did not block.** So the pair is: *a gate's message must be read* (that rule) **and** *a check must `exit` nonzero or its message is decoration* (this one). **Generalises past markdown: an informational check inside an automated chain is INDISTINGUISHABLE FROM NO CHECK at the moment it matters, because its output scrolls past in exactly the place a human would have stopped.** Same family as the correctly-named validator that is never invoked — existence is not enforcement. **⚠️⚠️ AND THE SECOND HALF, LEARNED BY DEFEATING THIS EXACT FIX FOUR HOURS AFTER WRITING IT (2026-08-13, coordinator, self-inflicted): A GATE THAT *DOES* EXIT NONZERO IS NEUTRALISED JUST AS COMPLETELY BY CAPTURING ITS EXIT CODE FOR DISPLAY.** The shape was `gate.py | tail -8; echo "GATE: ${PIPESTATUS[0]}" && git commit && git push` — **the gate exited 1, printed `GATE: 1`, and the `&&` chain proceeded from the `echo`'s success, pushing a corrupted table row.** Two compounding errors, each individually survivable: **`| tail -8` truncated away the one line naming the violation**, and **`echo`ing the status turned a gate back into a print** — the precise defect the rule above exists to prevent, reintroduced through a different door. **A pipe also destroys the exit code by default** (hence `PIPESTATUS`), which is what invites the echo in the first place. **PRACTICE: put the gate in the `&&` chain ITSELF (`python gate.py && git commit && git push`) so a nonzero exit stops the chain; never pipe it; never route its status through `echo`. If you want to see the status, run it AGAIN afterwards — reporting and enforcing are different jobs and must not share a command.** The corruption here was caught one command later only because the *next* thing run happened to be the same gate, unpiped.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Validating a gate means reading its MESSAGE, not just its exit code** — a gate can return the right verdict with the wrong diagnosis. And a check that only PRINTS is not a gate: put it in the `&&` chain, never pipe it, never route its status through `echo`. → [`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it)

## sha-is-not-a-stable-name

> **Index line (in CLAUDE.md):** **A sha is not a stable name for a change in a rebase workflow.** Anchor a cross-lane state check on CONTENT or the commit subject. `git log HEAD --not --remotes` answers "is this OBJECT on a remote", not "is this WORK landed" — branch reaping makes those diverge.

**⚠️ A SHA IS NOT A STABLE NAME FOR A CHANGE IN A REBASE WORKFLOW — FOR A STATE CHECK ACROSS A REBASING LANE, ANCHOR ON CONTENT OR ON THE COMMIT MESSAGE, NEVER ON A SHA HANDED TO YOU EARLIER (2026-08-13).** The coordinator reported a lane's work "not landed" from *"everything above `4f05f848` is mine"* — **`4f05f848` was the lane's PRE-REBASE sha, and `git rebase origin/main` rewrote it.** `merge-base --is-ancestor` correctly answered **not on main**, because that object genuinely is not in main's history, **while its CONTENT was, under a new sha.** **The query was well-formed, the answer was TRUE, and the conclusion was WRONG.** **This is a DIFFERENT mechanism from the other absence-claim failures in this file: those were the wrong instrument (one visibility mechanism of two; a file's gate standing in for a guarantee's coverage). THIS was the RIGHT instrument on an object that had been REPLACED — so it also bounds the positive-control rule further: a positive control proves the query can find its target; it cannot tell you the target is still the right NAME for the thing.** **Practice: `git show origin/main:<path> | grep <the actual construct>`, or match on the commit subject. This is specifically a COORDINATOR hazard — cross-lane state checks are SHA-anchored by habit — and a lane's peer SUMMARY rots the same way (it can name a branch that no longer exists on origin, and then answer a state check with it). **⚠️⚠️ AND THE SAME TRAP RUNS THE OTHER DIRECTION, INTO THE HANDOFF GUARD ITSELF — WHERE IT PRODUCES A FALSE REPORT OF UNPUSHED WORK, AND THE COORDINATOR CAUSES IT (2026-08-13, reported by the lane on its way out).** The standing handoff check is `git log HEAD --not --remotes` = 0. A lane returned **1** and correctly refused to read it as local-only work: **its local commit was the PRE-MERGE sha; the coordinator had rebased it onto main and then REAPED THE BRANCH, so the object lost its remote counterpart while its CONTENT shipped.** Verified: the sha is **not** an ancestor of `origin/main`, and the construct it added appears **5 times** in `git show origin/main:<path>`, under a new sha with the same subject. **`--not --remotes` answers *"is this OBJECT on a remote"*, NOT *"is this WORK landed"* — and branch reaping makes those diverge.** **A lane following the guard literally, after a reap, reports unpushed work that shipped an hour earlier — and the natural response to that report is to re-push it.** **PRACTICE: verify a nonzero count by CONTENT or SUBJECT before believing it; and as coordinator, either do not reap a branch until its lane has finished its handoff, or tell the lane you reaped it — the false positive is created by the cleanup, not by the lane.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A sha is not a stable name for a change in a rebase workflow.** Anchor a cross-lane state check on CONTENT or the commit subject. `git log HEAD --not --remotes` answers "is this OBJECT on a remote", not "is this WORK landed" — branch reaping makes those diverge. → [`sha-is-not-a-stable-name`](#sha-is-not-a-stable-name)

## uninformative-signals-both-directions

> **Index line (in CLAUDE.md):** **An uninformative signal does not become informative by being read pessimistically.** Absence-as-proof-of-absence and absence-as-proof-of-a-defect are the same error; in a registry the pessimistic one is more expensive. ⚠️ Includes the FALSE RECOMMEND-AGAINST: a null that confirms a pattern you were recently rewarded for is the one that most needs a second query.

**AN UNINFORMATIVE SIGNAL DOES NOT BECOME INFORMATIVE BY BEING READ PESSIMISTICALLY (2026-08-08).** The positive-control rule as usually carried is **one-sided** — *an empty grep is not evidence of absence* — which quietly implies the safe move is to assume the defect is present. **It is not.** Absence-as-proof-of-absence and absence-as-proof-of-a-defect are the **same error**; one lane made both in a single day, in opposite directions. In a registry the pessimistic one is **more expensive**: a wrong OPEN row costs someone a day **and looks responsible while doing it**. ⚠️⚠️ **AND THE 2026-08-14 VARIANT, WHICH IS THE MOST EXPENSIVE FORM BECAUSE IT ARRIVES DISGUISED AS THE DISCIPLINE ITSELF: A FALSE *RECOMMEND-AGAINST*.** Three lanes that day produced measured recommend-against reports and all three were RIGHT (LFM2 unportable, rubato not worth fixing, GAP-194's own premise false) — **so the shape acquired credibility.** A fourth then searched `fuel-*/src/**` *excluding* `lazy_`, found no training stack, and drafted *"no lazy training stack — recommend against, like LFM2."* **Wrong: the exclusion had cut out the answer. Continuing to look surfaced `Tensor::backward()`, the `fuel-training` crate, and a WORKING lazy training binary.** The port was feasible and would have been killed. **THE RULE: A NULL RESULT THAT CONFIRMS A PATTERN YOU HAVE RECENTLY BEEN REWARDED FOR IS THE ONE THAT MOST NEEDS A SECOND QUERY — because it is the one nobody, including you, wants to re-check.** A recommend-against is the highest-value report this project produces **and therefore the highest-value thing to get wrong**: it inherits the credibility of every correct one before it, and — unlike a wrong OPEN row — **it CLOSES work instead of creating it, so nothing later trips over the mistake.** Caught only by not stopping at the first grep. **Companion instance the same hour, same shape: that lane cited a `src/bin/` binary with ZERO `#[test]` fns as proof that a loss-convergence oracle 'works as a test'. A REAL ARTIFACT, CORRECTLY FOUND, CITED FOR A PROPERTY IT DOES NOT HAVE** — which is the [[a-precise-citation-spends-skepticism]] failure pointed at a file's TARGET KIND rather than its contents. **Corollary that retired an instrument here: the presence of a wildcard says NOTHING until you read what it does.** Two rows (GAP-075, GAP-158) were scoped off wildcard counts; one count was mostly real, the other **entirely artifact**, and **nothing about the count could distinguish them.** The replacement shape: **classify every arm by RHS, then check each arm's SCRUTINEE** — the scrutinee step is the one that gets skipped and the one that decides the answer.


**⚠️ AND A THIRD DIRECTION, WHICH IS NOT ABOUT ABSENCE AT ALL: A MAGNITUDE READ AS AN IMPOSSIBILITY (2026-08-20, self-retracted by the lane that wrote it).**

A lane reported *"`cargo fmt --all -- --check` **cannot be green** — `-p fuel-dispatch` alone has ~3818 diff hunks"*. Measured at head: **9 hunks across 6 files, all owned by one lane already committed to fixing them — one commit from green**, on a gate that has never passed in this repo's history.

**Their own diagnosis is the durable one, and it is sharper than "the number was stale":**

> **A hunk count measures DISTANCE FROM GREEN, not POSSIBILITY OF GREEN.** *"3818 hunks"* licenses *"far from green"*. *"Cannot be green"* is a claim **no hunk count can support at any magnitude** — it would have been wrong even if the number had still been accurate.

**So the number was right and the CONCLUSION CHANGED ITS TYPE**, from a quantity to a modality. That is a different failure from a stale measurement, and staleness is the half that gets checked.

**AND THE COST IS ASYMMETRIC IN THE SAME WAY THIS SECTION IS ABOUT: A FALSE "UNFIXABLE" RETIRES A FIX; A FALSE "FIXABLE" COSTS AN HOUR.** An impossibility claim ends work permanently and is almost never re-tested, because nobody re-attempts what they have been told cannot be done — the same ratchet as an unfalsifiable prohibition, arriving through a measurement instead of a rule.

**Two details worth keeping.** The lane had made the CORRECT call on the same axis four hours earlier, refusing to write *"COMPLETE"* in a doc where the evidence supported only *"the use cases shipped"* — **then made the mirror error on a number of their own.** And their later framing (*"mechanical and bounded, the cheapest lever"*) **contradicted their own earlier "cannot be green" without either of them noticing the two were in tension** — a self-contradiction inside one program, invisible because the two statements were about the same thing under different descriptions.

**PRACTICE: when a measurement licenses a verdict, check the verdict's TYPE against the measurement's. A count supports "far", "many", "worse than"; it never supports "cannot". If you want an impossibility claim, you need an argument, not a bigger number.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **An uninformative signal does not become informative by being read pessimistically.** Absence-as-proof-of-absence and absence-as-proof-of-a-defect are the same error; in a registry the pessimistic one is more expensive. ⚠️ Includes the FALSE RECOMMEND-AGAINST: a null that confirms a pattern you were recently rewarded for is the one that most needs a second query. → [`uninformative-signals-both-directions`](#uninformative-signals-both-directions)

## one-feature-is-not-two

> **Index line (in CLAUDE.md):** **One feature is not two — a gate covering each feature separately covers no INTERSECTION.** When a change must be exhaustive, enumerate the feature COMBINATIONS that parse each site, not the features.

**ONE FEATURE IS NOT TWO — A GATE THAT COVERS EACH FEATURE SEPARATELY COVERS NO INTERSECTION (2026-08-12).** `fuel-dispatch/src/telemetry/baracuda_provider.rs` is reachable only under a FEATURE PAIR, never a single feature: `mod telemetry` is cfg'd on `telemetry`, and `mod baracuda_provider` is cfg'd *inside it*. **⚠️ CORRECTED 2026-08-13 — THIS RULE HAD ROTTED ON THE EXACT FEATURE COMBINATION IT NAMES, AND THE ROT MADE THE PRESCRIBED GATE FAR MORE EXPENSIVE THAN NECESSARY.** It used to read "`mod baracuda_provider` is cfg'd on `cuda`", which was true when written and was **fixed by `f1d475d2` ("gate the provider on what it needs, not on what implies it")**: the module is now cfg'd on **`baracuda-types`**, and only the `pub use BaracudaStructureKeyProvider` re-export still needs `cuda`. **So the minimal gate for that match is `--features telemetry,baracuda-types` — no CUDA SDK, no `cuda-build.ps1` slot, seconds on any machine — whereas this text sent readers to a GPU-class build for the same answer. A stale rule about feature gating cost more than a wrong fact: it made the cheap gate look impossible.** The incident below is historical and its lesson stands unchanged. So a wildcard-free `match` there missing `DType::F8E5M2` sat on `main` as a **latent E0004 that no gate we run could see** — including GAP-097's `--features telemetry` gate, which is *strictly stronger* than a default build and **still does not parse the file.** Confirmed by the compiler through `scripts/cuda-build.ps1` with both required artifacts (`Environment initialized for: 'x64'` + `Checking fuel-dispatch`), exit 101 → fixed → exit 0. **So a dtype was added across 20 sites in 7 crates and missed a 21st that is structurally unreachable by every gate in the repo.** Practice: when a change must be exhaustive, enumerate the FEATURE COMBINATIONS that parse each site, not the features. **Population caveat, and it is the honest residual: a scan for `#[cfg(feature=…)]` immediately preceding a `mod` finds exactly one such nesting workspace-wide — but it is a LOWER BOUND by construction, because it sees gated MODULES and not `#[cfg(feature=…)]` on individual items or impls inside ungated modules, which are equally invisible and are where most exhaustive matches actually live.** See docs/gaps.md GAP-097, GAP-171.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **One feature is not two — a gate covering each feature separately covers no INTERSECTION.** When a change must be exhaustive, enumerate the feature COMBINATIONS that parse each site, not the features. → [`one-feature-is-not-two`](#one-feature-is-not-two)

## a-workflow-that-never-starts

> **Index line (in CLAUDE.md):** **A workflow that never starts is indistinguishable from one that runs and fails** — the discriminator is `steps: 0`, never the conclusion field. ⚠️ And `steps > 0` does not mean the gate reached YOUR code: read the failing step's log.

**A workflow that never starts is indistinguishable from one that runs and fails — the discriminator is `steps: 0` / `runner_group_name: None` from the jobs API, never the conclusion field.** Both render as the same red X, and a repo with known-red CI trains everyone to read one more X as more of the same. This is the CI-shaped instance of the standing rule that **an exit code is evidence about the harness until an artifact proves the work ran**: at job level the artifact is *executed steps*, and a red conclusion is equally consistent with "the gate caught something" and "nothing ever ran". Before citing any CI job as coverage, confirm it has ever executed a step. ⚠️⚠️ **AND THE `steps: 0` SIGNATURE RECURRED SELF-INFLICTED HOURS AFTER THIS RULE WAS WRITTEN, VIA A VALIDATOR THAT PASSED BECAUSE ITS *LIBRARY* WAS PERMISSIVE (2026-08-14).** A workflow edit orphaned a job's `with:` block, **GitHub ran ZERO JOBS**, and the pre-flight check said *"YAML OK"* — because **`yaml.safe_load` TOLERATES DUPLICATE KEYS**, silently letting the last one win. **The parser was answering *"can I load this?"* while the author was asking *"is this correct?"*, and those two questions diverge exactly where the defect lives.** **GENERALISE PAST YAML: A PARSE-SUCCEEDS CHECK IS NOT A VALIDITY CHECK, AND EVERY FORGIVING PARSER IS A VALIDATOR THAT AGREES WITH YOU.** JSON parsers that accept trailing garbage, TOML readers that ignore unknown keys, `serde(deny_unknown_fields)` left off, a Markdown table "checked" by rendering rather than by counting delimiters — same shape. **ASK WHAT THE PARSER IS PERMITTED TO IGNORE, and use the STRICTEST available mode (a duplicate-key-rejecting loader, `deny_unknown_fields`, a schema) — then SABOTAGE IT, because a strict mode you have never seen reject anything is indistinguishable from the permissive one.** **AND NOTE WHICH RULE SAVED IT: the defect was found by the `steps: 0` discriminator recorded that same morning. The check that failed and the check that caught it were both about CI, hours apart — which is the argument for writing these down rather than remembering them.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A workflow that never starts is indistinguishable from one that runs and fails** — the discriminator is `steps: 0`, never the conclusion field. ⚠️ And `steps > 0` does not mean the gate reached YOUR code: read the failing step's log. ⚠️⚠️ **AND EVERY LAYER THAT CAN STOP EARLY MUST BE TOLD NOT TO — THERE ARE AT LEAST TWO, NEITHER REPORTS WHAT IT SKIPPED, AND THEY WERE FOUND ONE AT A TIME (2026-08-14).** **(1) CI matrix `fail-fast`:** a failing OS cancels its siblings and reports `cancelled`, which is **not a result in either direction** — it hid a genuinely-broken defect that the failing OS *never runs*. **(2) `cargo test` STOPS AFTER THE FIRST FAILING TEST BINARY:** a failing binary cancels every binary after it. **Measured — fixing one defect took a macOS run from 21 executed suites to 72: fifty-one suites CI had never reached, revealed by an unrelated fix.** **THE COMMON PROPERTY IS THE RULE: NEITHER SAYS "AND N MORE WERE NOT RUN" — each reports a smaller number that looks like an answer, so a triage can be honestly scoped to what executed and still be an undercount of what exists.** Set `fail-fast: false` on the matrix AND `--no-fail-fast` on the invocation, or each fix reveals exactly one more layer, one run at a time. **And when reporting a failure list, SAY WHETHER IT IS A LIST OR A PREFIX — they are indistinguishable when the prefix has one entry.** → [`a-workflow-that-never-starts`](#a-workflow-that-never-starts)

## delimiter-traps-have-two-ends

> **Index line (in CLAUDE.md):** **A delimiter trap has two ends, and learning one does not protect you from the other.** Match at word boundaries at BOTH ends. "I've already learned this one" is not a defence — the last field before a closing brace is the end that keeps being forgotten.

**A DELIMITER TRAP HAS TWO ENDS, AND LEARNING ONE DOES NOT PROTECT YOU FROM THE OTHER (2026-08-08).** One worker hit both in eight hours. **Morning, matched at the PREFIX:** a spelling guard keyed on `e4m3fn` also fires inside `f8e4m3fn`. **Afternoon, matched at the SUFFIX:** a string-replace of `l.stride(),` also fires inside `t_l.stride(),` / `src_l.stride(),` — producing `t_&l.stride_unsigned()` and 18 fresh errors. **The rule was written down between the two, by the person who then broke it in the mirror direction.** Their conclusion is the durable one: ***"I've already learned this one" is not a defence.*** **State it as delimiters at BOTH ends** (`\b`-anchored regex / word-boundary match), or you re-derive the missing half by breaking something. **⚠️ AND A THIRD END THE `\b` FIX DOES NOT REACH — THE *TYPE* BOUNDARY (2026-08-13).** A lane scripted a six-family field rename with `^\s*rope_base: (.+),$` — perfectly anchored, both ends, no substring hazard — **and it rewrote 33 `LlamaConfig` literals in `lazy.rs`, because those configs have a field with the SAME NAME.** 34 changes, 1 legitimate. Caught by the compiler (`LlamaConfig has no field embed_scale`) and repaired from `git diff` **by count, not by eye**. **A REGEX KEYED ON A FIELD NAME HITS EVERY STRUCT THAT HAPPENS TO HAVE THAT FIELD — AND "RENAME A FIELD IN ONE STRUCT" IS PRECISELY THE CHANGE WHERE THAT IS MOST LIKELY, because related structs share vocabulary.** The pattern cannot see types; anchoring harder does not help, because the text really is identical. **PRACTICE: do struct-field edits per-file with anchored inserts, or drive them from the compiler; and when a scripted sweep touches more sites than the population you enumerated, RECONCILE BY COUNT before trusting the diff.** Same family as the prefix/suffix traps above — the boundary the pattern cannot express is just a type instead of a word. Caught immediately only because the **error count went UP** — a fast, unambiguous signal is what makes a self-inflicted sweep error cheap instead of expensive.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A delimiter trap has two ends, and learning one does not protect you from the other.** Match at word boundaries at BOTH ends. "I've already learned this one" is not a defence — the last field before a closing brace is the end that keeps being forgotten. → [`delimiter-traps-have-two-ends`](#delimiter-traps-have-two-ends)

## target-crate-compile-line

> **Index line (in CLAUDE.md):** **For a scoped check the required artifact is the TARGET CRATE's compile line, not any compile line** — a build that dies in deps emits every other positive artifact. ⚠️ And a crate-level line proves nothing about a `cfg`'d MODULE, nor about a crate's TESTS (that needs the `(lib test)` unit).

**FOR A SCOPED CHECK, THE REQUIRED ARTIFACT IS THE *TARGET CRATE'S* COMPILE LINE — NOT *ANY* COMPILE LINE (2026-08-08). This AMENDS the artifact rule below, which discriminates less than it appears to.** A `-p fuel-cuda-backend` check was terminated mid-flight while still building deps. Its log contained **`Environment initialized for: 'x64'`, 32 `Checking` lines, zero errors, and zero `E0004`** — **every positive artifact the rule below requires** — and it proved nothing about `fuel-cuda-backend`, which was **never reached**. **"The compiler ran" and "the compiler reached the code under test" diverge exactly when a build dies in deps, which on this box is most of the wall-clock.** ⚠️⚠️ **AND THE `Checking <crate>` LINE IS ITSELF NOT ENOUGH WHEN THE QUESTION IS ABOUT A CRATE'S *TESTS* — MEASURED 2026-08-14 BY THE GAP-157 LANE, AND IT SHARPENS THIS RULE'S OWN PRESCRIPTION.** `cargo check -p fuel-core --features cuda` builds `fuel-cuda-backend` **as a LIB DEPENDENCY** and never compiles its test targets — **yet it emits a perfectly good `Checking fuel-cuda-backend v0.10.3` line**, which this rule as written accepts as proof the target crate was reached. It WAS reached; **its `#[cfg(test)]` code was not**, so that leg structurally could not verify three `probe.rs` conversions while looking like full coverage. **THE DISCRIMINATING ARTIFACT IS THE `(lib test)` UNIT: `warning: fuel-cuda-backend (lib test) generated 42 warnings` — the `(lib test)` suffix is what proves the test targets compiled.** Quote that, not the bare `Checking` line, whenever the claim involves a crate's tests. Same family as `--lib` not building `tests/`: **the target KIND is a second dimension the feature list never mentions, and a crate-level compile line collapses it.** Grep for the target crate's own `Checking <crate> v<version>` line, and **name it in the report**. **This is a FIFTH build state, and the most deceptive:** *unrun* (chose not to), *unrunnable* (machine can't), *invisible* (non-default feature — see below), *invisible-and-unrunnable* (mkl/accelerate/aocl/onemkl), and now **ran, looked green, terminated before reaching the target.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **For a scoped check the required artifact is the TARGET CRATE's compile line, not any compile line** — a build that dies in deps emits every other positive artifact. ⚠️ And a crate-level line proves nothing about a `cfg`'d MODULE, nor about a crate's TESTS (that needs the `(lib test)` unit). → [`target-crate-compile-line`](#target-crate-compile-line)

## claim-shape-decides-n

> **Index line (in CLAUDE.md):** **Choose N from the claim's SHAPE, not from a habit.** Falsifying claims cost ONE counterexample; confirmatory and comparative claims are expensive. And `n/n` never establishes determinism — by the rule of three it only bounds the miss rate (20/20 ⇒ under ~14%).

**CHOOSE N FROM THE SHAPE OF THE CLAIM (2026-08-19, GAP-001 lane, self-reported against their own run).** Three shapes, three costs:

- **Falsifying** — *"the sync fixes it"*, *"alpha.78 introduced this"*. **One counterexample ends it.** The lane ran 20 repeats of a sync experiment that predicted 0/20; **the first failure settled it and runs 2–20 were confirmation of something already dead.** Retroactively the same was true of the control arm: *"alpha.78 introduced this defect"* entails *"alpha.77 does not have it"*, which **one** failure on `.77` falsifies. They ran eleven.
- **Comparative** — *"`.77` is worse than `.78`"*. **Expensive, and usually not worth it.** 5/20 vs 11/20 gives Fisher exact **p = 0.105** — not distinguishable at n=20/arm. The lane correctly refused to publish it, which is the right call: **a point estimate running opposite to a premise is rhetorically striking and statistically weak, and saying so is cheaper than being quoted.**
- **Confirmatory** — *"this is fixed"*, *"this is deterministic"*. **The expensive one, and the one people default to without noticing.**

**⚠️ AND `n/n` NEVER ESTABLISHES DETERMINISM — IT BOUNDS A MISS RATE.** By the rule of three: **20/20 bounds the miss rate below ~14% at 95% confidence**; under 5% needs ~60 runs; under 1%, ~300. **A row that records `20/20` as proof something never happens is overstating by roughly an order of magnitude.** State it as *"miss rate not large"*, with the bound.

**WHY THIS EARNED A RULE RATHER THAN A NOTE.** The same day produced **two independent ~25% intermittents** across the portfolio (Lightbulb's mutation surviving 4/15; Fuel's GAP-001 at 5/20, later 11/20 on the *pinned* version) — **both of which a single trial reports as clean.** Fuel's lane had a clean 1/1 validation run in hand; reporting it would have un-pinned a dependency and declared a sibling project's crate innocent, with a correct instrument, correct prefill and correct controls. **Everything was right except the sample size, and no amount of care with the instrument catches that.** The reciprocal failure is real too — being over-powered for a falsifying claim wastes an hour of forge — so the rule is *match N to the shape*, not *always run 20*.

**PRACTICE: before choosing N, say out loud whether the claim is falsifying, comparative, or confirmatory. Then state the repeat count in the result** — a claim that says `1/1` stays admissible; it just stops being indistinguishable from one that says `20/20`.


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Choose N from the claim's SHAPE, not from a habit.** Falsifying claims cost ONE counterexample; confirmatory and comparative claims are expensive. And `n/n` never establishes determinism — by the rule of three it only bounds the miss rate (20/20 ⇒ under ~14%). → [`claim-shape-decides-n`](#claim-shape-decides-n)

## docs-are-not-code-and-a-sweep-cannot-tell

> **Index line (in CLAUDE.md):** **A mechanical rename rewrites re-derivation commands embedded in prose, silently — no compiler, test or CI job reads markdown.** Exclude `docs/**` from tree-wide identifier sweeps, or re-run every embedded command afterwards. Anchor on identifiers that were RETIRED, not RENAMED. **AND ANCHOR THE CONTROL ON STRUCTURE, NOT ON ANY IDENTIFIER — the claim anchors on something RETIRED and retired things can only stay absent, but the CONTROL must anchor on something PRESENT and anything present can be renamed. The control is the fragile half, and the retired-identifier rule protects the half that was never at risk.** Use a file count (`git ls-files '<crate>/src/*.rs' | wc -l`), which breaks only on a restructure that moves files loudly.

**A MECHANICAL RENAME CORRUPTS DOC-EMBEDDED VERIFICATION COMMANDS, AND NOTHING DETECTS IT (2026-08-20, architect, self-inflicted, caught by the lane whose program it was about to invalidate).**

The `Lazy`-prefix sweep (`18c29ad0`) swept `docs/**` along with `*.rs`. Hours earlier the same architect had written a B6 evidence block:

```
pub struct Tensor  in fuel-core/src/*.rs  ->  0 matches      <- claim
(control) pub struct LazyTensor           ->  1 match        <- control
```

The sweep rewrote `LazyTensor` -> `Tensor` **inside the fenced code block**, leaving **the claim and its control as the same string, asserted to return both 0 and 1** — and leaving the claim independently FALSE, because `pub struct Tensor` in `fuel-core/src/` now matches the renamed lazy type. **A reader running it gets 1 and concludes B6 regressed.**

**THE FACT SURVIVED AND THE EVIDENCE DIED, WHICH IS THE WORSE FAILURE** — a fact is recoverable; a corrupted control is invisible, and it fails in the direction that MANUFACTURES a false alarm.

**Why this class is specific rather than obvious: a re-derivation command is the artifact designed to keep documentation honest, and it is made of source identifiers.** So the better a doc is at being checkable, the more surface it exposes to ordinary maintenance. **No compiler, no test, and no CI job reads markdown**, so there is no instrument anywhere in the repo that would have flagged it.

**PRACTICE, in order of strength:**

1. **EXCLUDE `docs/**` from mechanical identifier sweeps.** A rename rewriting a `git grep` string inside a fenced code block is a rename doing something nobody asked for. Update prose deliberately, as its own reviewed change.
2. **ANCHOR ON WHAT A RENAME CANNOT TOUCH** — file paths, `git ls-files`, directory counts, and identifiers that were **RETIRED** rather than renamed. `BackpropOp` and `fuel-core/src/op.rs` are good anchors *because they were deleted*. `pub struct Tensor` was a bad one *because it was renamed into*.
3. **Where an identifier is unavoidable, say what it was renamed FROM**, so a later sweep's damage is legible rather than silent.
4. **After any tree-wide rename, re-run every doc-embedded command.** The lane that found this did exactly that after rebasing, and caught three more of their own that had broken identically — control -> 0 (dead), claim -> 1 (reads as regression). **Had they pushed before rebasing they would have landed three amendments whose controls were already dead.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A RENAME THAT COMPILES IS NOT A RENAME THAT IS DONE — AND THE REMEDY IS NOT A SWEEP.** The compiler checks call sites and nothing checks prose: **a bare `` `Foo` `` has no referent, so a rustdoc gate closes only the LINKED half** (`[Foo]` fails when `Foo` dies; `` `Foo` `` does not). After a rename, grep the old names across `*.rs` and `*.md`. **The grep is mechanical; the DISPOSITION IS NOT** — an old name in prose is **STALE** (rename it), **HISTORICAL** (prose *about* the rename — renaming destroys the record and self-erases), or **PINNED** (documents a revision — renaming makes it FALSE). **A missed stale mention misleads and is findable later; a swept historical or pinned one DESTROYS EVIDENCE and reads as correct.** Fuel's 296-file `Lazy` sweep measured **0 stale**, all 12 markdown hits historical. → [`docs-are-not-code-and-a-sweep-cannot-tell`](#docs-are-not-code-and-a-sweep-cannot-tell)

## checked-then-didnt-look

> **Index line (in CLAUDE.md):** **Performing a verification and then reporting something other than its output is a distinct defect from failing to verify — and remembering to check does not fix it.** Report from the artifact, not from the fact that a step completed.

**TWO INSTANCES IN ONE NIGHT, TWO PEOPLE, DIFFERENT TOOLS (2026-08-20).**

- The architect ran `git log -1` on five shas *specifically to verify them*, and their terminal printed `docs(outreach): baracuda-seam SEAM_MAGIC lockstep ask`. They then wrote **"Spec-B candidate-kernel ingestion"** — the label from the document under review — into a report that reached CireSnave.
- The doc-currency lane resolved a rebase conflict, confirmed the rebase **completed**, and reported *"resolved in favour of yours."* The push had in fact overwritten the other party's file. **They never read the file; they read that the step succeeded.**

**Why this is not "forgot to check":** the check ran. Its output existed. What failed was the step between the output and the claim — a prior belief supplied the answer, and the measurement was treated as a formality that had been satisfied rather than as a source of information. **A rule that says "verify before claiming" is already satisfied by both of these.**

**Practice:** state the claim in the form of the artifact — quote the subject line, `cat` the file, paste the count — so the report *is* the output rather than a summary of it. If you cannot quote it, you did not read it. And be most suspicious when the measurement is expected to confirm something: **a number that agrees with the prose gets shipped; a number that disagrees gets re-measured** (see `gap-029-persistent-decode-trait.md`, where the naive grep returns exactly the stale figure the document claims).

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Performing a verification and then reporting something other than its output is a distinct defect from failing to verify — and remembering to check does not fix it.** Report from the artifact, not from the fact that a step completed. → [`checked-then-didnt-look`](#checked-then-didnt-look)

## git-rebase-inverts-ours-and-theirs

> **Index line (in CLAUDE.md):** **In a REBASE, `--theirs` is the commit being APPLIED (yours) and `--ours` is upstream — the exact inverse of a merge.** Taking "theirs" to mean "the other side's version" silently keeps your own and discards theirs.

**A footgun with no warning and a plausible-sounding name in both directions (2026-08-20).** A lane resolving a conflict against a peer's committed repair ran `git checkout --theirs <file>` meaning *"take main's version"*. During a rebase, upstream is checked out first and each commit is replayed **onto** it, so **`--ours` is upstream and `--theirs` is the commit under replay.** The resolve silently kept the lane's own version, the rebase reported success, and the subsequent push overwrote the peer's work — while the lane reported the opposite in good faith.

**Practice:** do not use `--ours`/`--theirs` during a rebase at all. Name the source explicitly — `git checkout origin/main -- <path>` — which is unambiguous under both operations. Then **read the file** before reporting what it contains (see `checked-then-didnt-look`; these two combined to produce the incident).


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **In a REBASE, `--theirs` is the commit being APPLIED (yours) and `--ours` is upstream — the exact inverse of a merge.** Taking "theirs" to mean "the other side's version" silently keeps your own and discards theirs. → [`git-rebase-inverts-ours-and-theirs`](#git-rebase-inverts-ours-and-theirs)

## a-local-branch-goes-stale-too

> **Index line (in CLAUDE.md):** **`git checkout main` in a worktree lands you in the PAST.** The stale-tree rule is usually stated about the shared checkout; the local `main` BRANCH rots the same way and is measured the same way. Verified 2026-08-20: local `main` was **150 commits** behind `origin/main`.

**THE STALE-TREE HAZARD HAS A SECOND FORM AND THE USUAL PHRASING MISSES IT (2026-08-20, reported by the precision lane after it bit them).**

CLAUDE.md leads with *establish facts with `git show origin/main:<path>`, never by reading a working tree* — and everyone reads that as being about **the shared checkout** `C:\Projects\fuel`. **The local `main` BRANCH is a separate object and rots independently.**

Measured the day it was reported:

```
git rev-list --count main..origin/main   ->  150
local main   2699fbad  2026-08-13
origin/main  1be77f05  2026-08-20
```

**A lane that branches from `origin/main` explicitly is fine. A lane that runs `git checkout main` first is silently seven days and 150 commits in the past**, in a repo taking 40+ commits a day — and every fact it then establishes is about last week's code.

**Why this is worse than the shared-checkout case: `main` is the name people reach for.** The shared checkout at least has a suspicious path; `main` reads as authoritative by its name alone.

**PRACTICE: branch from `origin/main`, never from local `main`; and when a measurement disagrees with something you believe, check `git rev-list --count main..origin/main` before re-deriving the claim.** Same instrument as the shared-tree case, pointed at a ref instead of a directory.

⚠️ **THAT PRACTICE IS REACTIVE, AND A 2026-09-02 INSTANCE DEFEATED IT: *"when a measurement disagrees with something you believe"* REQUIRES A DISAGREEMENT.** A behaviour-preserving refactor of the gaps gate's pipe-splitter was verified by a differential over every line of `docs/gaps.md`, before and after. **It returned IDENTICAL — and it was IDENTICAL ABOUT THE WRONG FILE**, because the local branch was one commit behind its own remote. **Measured: the differential recorded 350 lines where the real file has 353.** ⚠️ **A reassuring result, and reassuring results are not audited.** Git had said so — `git checkout` printed *"use git pull to update your local branch"* — but that line is **noise-shaped**, and it appears whether or not it matters.

⚠️ **THE DETECTOR THAT ACTUALLY CAUGHT IT IS PROACTIVE, CHEAP, AND UNLIKE `git status` CANNOT BE READ AS NOISE: GREP THE TREE FOR THE NEWEST ARTIFACT YOU KNOW LANDED.** Here, a conflict-marker check written into that very file two PRs earlier was **absent from the working tree** — and it could only be absent if the tree predated a commit known to have merged. **The tell was a thing that SHOULD have been there and was not, which is the opposite of the usual stale-tree tell and is why it survives a clean result.** **State it as: *before trusting any measurement taken from a checkout, grep for the newest thing you know is in it.*** ⚠️ **AND THE CAVEAT WITHOUT WHICH IT FALSE-POSITIVES ON EVERY UN-REBASED BRANCH IN THE REPO: AN ARTIFACT'S ABSENCE IS EVIDENCE OF STALENESS ONLY WHEN YOU ALSO KNOW IT SHOULD HAVE ARRIVED BY THEN.** Measured 2026-09-02 on its first deliberate use: the same grep returned **0 before a rebase and 1 after**, and the 0 was CORRECT — the artifact had landed on `main`, not on the branch, so its absence meant *not yet rebased* rather than *stale*. **The detector distinguishes those two only if you supply the missing premise, which is knowledge about the branch's expected base and not about the file.** It costs one grep, it works when the answer is flattering, and **it only works if you pick an artifact distinctive enough that its absence is unambiguous** — a recently-landed check, a named constant, a new function — never something a merge could plausibly have renamed.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`git checkout main` in a worktree lands you in the PAST.** The stale-tree rule is usually stated about the shared checkout; the local `main` BRANCH rots the same way and is measured the same way. Verified 2026-08-20: local `main` was **150 commits** behind `origin/main`. → [`a-local-branch-goes-stale-too`](#a-local-branch-goes-stale-too)

## a-297-byte-log-has-three-causes

> **Index line (in CLAUDE.md):** **A detached CUDA build that dies at ~297 bytes with vcvarsall's banner and NO marker has THREE known causes, all in the script, none in the launcher:** a missing `call` before vcvarsall, **LF-only line endings in the `.bat`**, and (for `.ps1`) **UTF-8 without a BOM plus any non-ASCII character**, which Windows PowerShell 5.1 reads as ANSI.

**THE 297-BYTE SIGNATURE IS A SCRIPT DEFECT, AND KNOWING THAT IS ONLY HALF — IT HAS AT LEAST THREE DIFFERENT CAUSES (2026-08-20, third cause found by Fuel 1).**

CLAUDE.md already records that the launcher is exonerated by an isolating control (`Start-Process` + broken bat → died; WMI + **the same** broken bat → died identically, 297 bytes; WMI + fixed bat → survived). **That correctly sends you to the script. It does not tell you which defect.**

**Cause 1 — missing `call`.** `<...>vcvarsall.bat amd64` without `call` *terminates* the parent bat. The banner appears; nothing after it runs.

**Cause 1b — `call` PRESENT, but combined with a per-line `>>` redirect.** **vcvarsall halts a `call`-with-redirect.** This is the inverse of cause 1 and it defeats a reader who has been told to check for a *missing* `call`: the `call` is there, correct, and is part of the problem **in combination with** the redirect. The recorded working recipe chains instead — `vcvarsall && cargo`, **one** redirect for the whole chain, **no** `call`. Verified foreground: `cl.exe` resolves (14.51.36231), exit 0.

**⚠️ CAUSE 1b IS WIDER THAN THE REDIRECT, AND THE WIDENING WAS ESTABLISHED BY AN OPPOSITE-OUTCOME CONTROL (2026-08-20, Fuel 1, correcting their own first report).** The claim as first offered was that a **parenthesized-group redirect** — `( … ) > log` — holds the log handle so **the line after the group never executes**, even when that line writes to a *different* file. It was offered as a new mechanism. **It is not. The group redirect is innocent, and two repros separate the variables:**

```
( echo x ) > log             then  echo y > separate-file   ->  line-after RUNS
( "vcvarsall" amd64 ) > log  then  echo y > separate-file   ->  line-after does NOT run
```

**Same group, same redirect, opposite outcomes — so the differentiator is vcvarsall, not the redirect.** `vcvarsall.bat` terminates any *subsequent statement* in the batch file, whether that statement follows a `call vcvarsall >>` line (cause 1b) or a `( vcvarsall … )` group. **The ONLY thing that survives is a SAME-STATEMENT `&&` continuation** — which is exactly why the recorded working recipe (`vcvarsall && cargo`) works: cargo sits inside vcvarsall's own statement.

**PRACTICE: nothing runs after vcvarsall in the same batch file except a same-statement chain. If you need to do anything afterwards, ISOLATE VCVARSALL IN ITS OWN SUBPROCESS** — `cmd /c "vcvarsall amd64 && set"` to harvest the environment, then `Set-Item env:` per line in the calling shell. The termination is then contained in a throwaway subprocess and the caller survives it.

**AND KEEP THE SHAPE OF THE CORRECTION, WHICH IS THE TRANSFERABLE PART: two candidate mechanisms were CO-PRESENT in every failing run (a group redirect, and vcvarsall), so no amount of staring at the failures could separate them. What settled it was the SAME construct with the OTHER variable removed** — a group redirect around `echo`. **The reporter built that control against their own published claim and refuted it.**

**Cause 2 — LF-only line endings in a `.bat`.** `cmd.exe` wants CRLF and mis-parses an LF-only batch file: it executes the early lines, then chokes mid-file. **The Write tool emits LF.** Write bats with PowerShell, or convert, and **measure** (`grep -c $'\r'`) rather than assume.

**Cause 3 — a `.ps1` that is UTF-8 WITHOUT a BOM and contains any non-ASCII byte.** Windows PowerShell 5.1 reads BOM-less files as **ANSI**, so a single em-dash (`—`, `e2 80 94`) becomes three garbage characters *inside a string literal* and the parse fails several lines later with a message naming an innocent token.

**THE DIAGNOSTIC TELL THAT SEPARATES THEM FROM A PATH FAILURE:** `Environment initialized for: 'x64'` **present** in the log means the bat started and vcvarsall ran, so execution stopped *after* the banner. **A missing-interpreter/PATH failure dies BEFORE the banner.** Read the banner's presence before forming a hypothesis.

**⚠️ AND CAUSE 3 PRODUCED A FALSE DIRECTIVE THAT SURVIVED IN GPU-SAFETY INFRASTRUCTURE.** `scripts/cuda-build.ps1` and `scripts/gpu-run.ps1` both declared `#Requires -Version 5.1` while failing to parse under 5.1 (**10 and 4 errors**). **The directive was TRUE ABOUT THE LANGUAGE AND FALSE ABOUT THE FILE** — every construct is 5.1-compatible; the encoding is not. That is a nastier variant of a false guard, because reading the code confirms the claim and only running it refutes it.

**Fixed by making the claim true rather than by narrowing it** — a UTF-8 BOM was added to both, so 5.1 now parses them clean (0 errors) and pwsh 7 is unaffected (0 errors). **Weakening `#Requires` to `7.0` would also have been honest, and was rejected: the author evidently intended 5.1 support and the content delivers it.** Prefer repairing the artifact to lowering the claim, where the claim was achievable.

**PRACTICE: for any `.ps1` that must run under Windows PowerShell, write it UTF-8 WITH BOM, or keep it strictly ASCII. Test with `[System.Management.Automation.Language.Parser]::ParseFile` under `powershell.exe`, not just `pwsh` — the version you run is not the version your `#Requires` promises.**


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **THE "vcvarsall recipe is PowerShell-tool-ONLY" CONSTRAINT IS DISSOLVABLE — put the quotes in a `.bat` (2026-08-08).** That rule exists because Git Bash's MSYS layer mangles **both** the leading `/c` argument **and** the inner double quotes around the `.bat` path. **The quotes only matter if they are on the command line.** So write the recipe into a `.bat` (`cd /d <worktree>` / `call "<...>vcvarsall.bat" amd64` / `set BARACUDA_FORGE_THREADS=6` / `cargo check …`) and invoke it as `cmd //c <path-to-bat>` — **`//c` survives the path rewrite and there are no quotes left to mangle.** Positive-controlled when adopted (`Environment initialized for: 'x64'` x1, the VS Developer Command Prompt banner, 60 KB of output, live processes). **This does NOT solve the 10-minute kill above — it solves the quoting constraint only.** Use both. **⚠️ SUPERSEDED 2026-08-20 BY A CHEAPER DISSOLUTION — AND THE OLD TEXT ABOVE STAYS, because it is the record of what Git Bash does to `/c` and to quotes, and every word of it is still TRUE.** The `.bat`-plus-`cmd //c` recipe works, and it is no longer the cheapest path. **Isolate vcvarsall in a throwaway subprocess instead of ever putting it on a command line:** `cmd /c "vcvarsall amd64 && set"` to harvest the environment, `Set-Item env:` per line in the calling PowerShell session, then run cargo from a shell that never had vcvarsall in it — cmd reduced to a trivial CRLF launcher. **The mechanism is measured and lives in [`docs/method-rules.md` § `a-297-byte-log-has-three-causes`](#a-297-byte-log-has-three-causes) (cause 1b, widened); do not restate it here — that file is the evidence, this line is the pointer.** The controlling fact, with an opposite-outcome control: `( echo x ) > log` then a separate line RUNS, while `( vcvarsall amd64 ) > log` then a separate line does **NOT** — **nothing runs after vcvarsall in a batch file except a same-statement `&&` chain**, which is exactly why `vcvarsall && cargo` works and why everything else silently doesn't. **⚠️ AND THIS IS THE WORKED EXAMPLE OF ITS OWN DEFECT CLASS — [`staleness-by-workaround`](#staleness-by-workaround): a rule that is TRUE when written, REMAINS true as stated, and still routes the reader down a far more expensive path than the one that now works.** Nothing above needed correcting, so no re-reading of it would ever have raised a flag; the only detector was somebody re-attempting the forbidden thing and finding the obstacle gone. **When you write a workaround, record what would have to change for it to stop being necessary — otherwise it outlives its obstacle and nothing can tell you.**

## the-marker-can-be-eaten-by-its-own-log

> **Index line (in CLAUDE.md):** **Write the completion marker to a SEPARATE FILE from the build log.** If the log is captured by a group/whole-script redirect, that redirect owns the handle for the script's lifetime and a marker appended to the same file is lost — producing "no marker" on a build that finished, which the discipline reads as "no result".

**THE EXIT-CODE MARKER DISCIPLINE HAS A FAILURE MODE OF ITS OWN, AND IT IS INDISTINGUISHABLE FROM THE THING THE MARKER EXISTS TO DETECT (2026-08-20, found by Fuel 1 while rebuilding a detached forge launcher).**

The standing rule for long CUDA builds is: launch detached, have the script write its own `<TAG>_DONE_EXITCODE=<code>` as its last output, and **treat "no marker" as NO RESULT rather than as failure**. That rule is right and stays.

**But if the log is captured with a parenthesized-group or whole-script redirect — `( … ) > log.txt` — that redirect holds the file handle for the group's lifetime.** A marker line appended to **the same file** after or inside the group is silently lost. **So a build that ran to completion reports exactly the signature of one that died: no marker.**

**The rule and its own failure mode produce the same observation**, which is the worst property a diagnostic can have. **The `Environment initialized for: 'x64'` banner and the target crate's `Checking <crate>` line are still in the log — so a log with real content and no marker should raise this, not the death hypothesis.**

**⚠️ MEASURED 2026-08-20 — THE MECHANISM IS A SHARING VIOLATION, AND IT LEAVES A TELL THAT IS NOT SILENCE.** Two runs of one CRLF `.bat` that appends a marker to the log AND to a separate file:

```
A  cmd /c "bat > mlog.txt 2>&1"     mlog.txt:   BODY_LINE_1
                                                The process cannot access the file because it is
                                                being used by another process.   <- IN THE MARKER'S PLACE
                                                BODY_LINE_2
                                    marker.txt: MARKER_SEP_FILE=0                <- separate file SURVIVED
                                    cmd exit=0                                   <- FAILURE DID NOT PROPAGATE

B  cmd /c "bat"      (control)      mlog.txt:   MARKER_SAME_FILE=0               <- same append SUCCEEDS
```

**Three things worth having exactly:** (1) the same-file append fails with a **sharing violation**, because the whole-script redirect owns the handle for the script's lifetime; (2) **the error text lands in the log AT THE MARKER'S POSITION**, so the log is not silent — it carries a file-lock complaint that reads as unrelated noise, while a reader grepping for `_DONE_EXITCODE=` still finds nothing; (3) **`cmd` exits 0** — the failed append does not propagate, so the script reports success while its completion marker never lands. **Run B is what proves the redirect is the cause rather than the append being wrong.**

**PRACTICE: the marker goes in a SEPARATE FILE.** `echo TAG_DONE_EXITCODE=%ERRORLEVEL% > marker.txt`, distinct from the log's redirect target. Then "no marker" means what it is supposed to mean. **And keep the three checks that were already required — liveness (pid), completion (marker), verdict (the target crate's own compile line) — because any two of them leave a state indistinguishable.**


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Write the completion marker to a SEPARATE FILE from the build log.** If the log is captured by a group/whole-script redirect, that redirect owns the handle for the script's lifetime and a marker appended to the same file is lost — producing "no marker" on a build that finished, which the discipline reads as "no result". → [`the-marker-can-be-eaten-by-its-own-log`](#the-marker-can-be-eaten-by-its-own-log)

## a-stale-tool-is-a-wrong-action

> **Index line (in CLAUDE.md):** The stale-shared-tree rule covers reading stale CODE and getting a clean wrong ANSWER. **Executing a stale TOOL is a different and worse case: a tool carries machine-wide state, so a stale copy is not a wrong answer but a wrong ACTION — and no artifact records which version ran.**

**FOUND 2026-08-20 (docs/gaps.md GAP-223), AND THE WELL-KNOWN STALE PATH IS NOT THE ONE THAT BIT.**

`5db8e9af` fixed `scripts/gpu-run.ps1` and `scripts/cuda-build.ps1`, which declared `#Requires -Version 5.1` while being UTF-8 **without BOM** with em-dashes — so they could not parse under the version they demanded. Measured with 5.1's own parser (host `5.1.26100.9168`), positive control a deliberately-unbalanced `.ps1` returning `PARSE_ERRORS=1`:

```
origin/main          gpu-run.ps1  0    cuda-build.ps1   0     (also 0 under pwsh 7.6.5)
C:\Projects\fuel     gpu-run.ps1  4    cuda-build.ps1  10     (0 under pwsh 7.6.5)
```

**TWO INDEPENDENT STALE PATHS reach the same broken tool, and the one everybody is warned about is not the one that bit:**

1. **The shared checkout** — `C:\Projects\fuel`, where every lane's Bash tool defaults its cwd, sitting 162 commits behind. **It is also the ONLY natural path for a cross-project caller** (Baracuda, Vulkane) that has no Fuel worktree of its own.
2. **A lane's OWN worktree at a pre-fix commit.** This is what actually happened: the lane invoked an absolute path into their own tree and never touched the shared checkout — they ran the test before rebasing.

**THREE PROPERTIES MAKE THIS SURVIVE RATHER THAN GET CAUGHT:**

- **It is shell-dependent, so it is intermittent ACROSS callers.** Under `pwsh` the stale copy parses clean; under `powershell` it dies. **Two lanes disagree, both are right, and neither can see why.** A defect that manufactures disagreement between honest reporters.
- **The failure direction removes the guard.** `gpu-run.ps1` is the machine-wide GPU mutex — the only thing preventing a repeat of the 2026-07-31 host-aperture kernel bugcheck. **The natural reaction to "the wrapper won't even parse" is to run the GPU command directly.** A guard whose failure mode is to delete itself.
- **It manufactures a FALSE STANDING CONSTRAINT.** The lane concluded *"5.1 is out; pwsh 7 required"* and was about to write it down. **A false constraint born of a stale path is durable precisely because it makes the working shape look mandatory** — nobody re-tests a requirement that the thing they already built appears to satisfy.

**PRACTICE: invoke a shared TOOL from a head worktree by absolute path, and confirm the worktree is at head first. When a tool misbehaves, check the version you EXECUTED before forming any hypothesis about the language, the shell, or the tool's design.** Note what this does NOT admit of: a version check *inside* the tool cannot help, because the stale copy is the thing that would run it.

**Corollary on attribution.** The architect filed the row asserting the lane hit it via the shared checkout, inferred from their `from_cwd`. **That was a hypothesis published as a fact, and the lane refuted it.** The measured hazard stood — it had been measured directly rather than inferred from the report — **and the correction WIDENED the class rather than shrinking it.** Verify the path a symptom came through, not merely that the symptom is real.

**⚠️ AND THE DEEPER FINDING, WHICH ARRIVED FROM A PEER WHO NEVER HIT THE PARSE ERROR AT ALL: THE BYPASS DOES NOT NEED A BROKEN WRAPPER AS ITS EXCUSE.** Vulkane reported, unprompted, that they had run `cargo test --workspace --features …,kiss-target` **with no wrapper at all** — a GPU run, since `kiss_target_live.rs` creates an instance and enumerates physical devices — while chained behind a `cargo fmt`, having *"`cargo test --workspace` IS a GPU run"* written in their own durable memory. Nothing went wrong; **the only reason anyone knows is that they said so.**

**Their framing is the one to keep: THE GUARD'S ABSENCE IS INDISTINGUISHABLE FROM ITS SUCCESS.** The run completes, the tests pass, and the sole difference is a mutex nobody observes. Same family as every other invisible-null in this file, with a worse blast radius — a host-aperture bugcheck rather than a wrong token.

**So "use `pwsh`" and "remember the wrapper" are BOTH instructions that decay, and decay was just demonstrated in someone who had the rule recorded.** The durable form is to make the guarded thing detect the guard's absence: **`gpu-run.ps1` already exports `GPU_RUN_HELD=1` into the child environment** (for nested-invocation passthrough), and **zero Rust code observes it** — so a live-GPU test helper that refuses to proceed without it costs almost nothing and converts a silent success into a named refusal at the point of use.


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A STALE TOOL IS A WRONG ACTION, NOT A WRONG ANSWER.** The stale-shared-tree rule covers READING stale code; EXECUTING a stale tool is worse, because a tool carries machine-wide state and no artifact records which version ran. Measured: `gpu-run.ps1` — the GPU mutex — has **4 parse errors under PowerShell 5.1 in the shared checkout** and **0 at head**, so the guard's failure mode is to delete itself. **It is SHELL-DEPENDENT and therefore manufactures disagreement between honest reporters.** Two stale paths reach it: the shared checkout AND your own worktree at a pre-fix commit — **the second is the one that bit.** ⚠️ **And the bypass does not need a broken wrapper as its excuse: a peer ran a GPU workload with no wrapper at all, with the rule in their own durable memory. THE GUARD'S ABSENCE IS INDISTINGUISHABLE FROM ITS SUCCESS.** → [`a-stale-tool-is-a-wrong-action`](#a-stale-tool-is-a-wrong-action)

## staleness-by-workaround

> **Index line (in CLAUDE.md):** A rule can be TRUE when written, remain TRUE as stated, and still send the reader to a far more expensive path than the one that now works. **"Is this claim still true?" passes on every one of these.** Ask the second question: **is the path it prescribes still the cheapest one?**

**NAMED 2026-08-20 by the doc-currency lane, who observed that their entire program would sail past this class untouched.**

Doc-currency auditing tests one predicate: **is the claim still true?** That catches staleness by contradiction — the doc says X, the code says not-X. **It cannot catch a rule whose every sentence remains true while the obstacle it was written to route around has been removed.** Nothing in the rule is false, so no amount of re-reading it produces a flag.

**THE WORKED EXAMPLE IS IN THIS REPO'S OWN `CLAUDE.md`.** The vcvarsall recipe is documented as *"PowerShell-tool-ONLY"*, because Git Bash's MSYS layer mangles both the leading `/c` and the inner quotes. **Every word of that is still true.** And it is now avoidable: put the quotes in a `.bat` and invoke `cmd //c <bat>`; better still, harvest vcvarsall's environment in a throwaway subprocess and never put it on a command line at all. **The rule survives its own audit and costs the reader the cheap path.**

**A sibling instance, same shape, higher cost:** the gate for a `cfg`'d module was documented as `--features telemetry,cuda` — a GPU-class build — after the module's gate had been split onto `baracuda-types`. **The stated rule was about feature gating and was not obviously false; what it did was make the CHEAP gate look impossible**, sending two people to a 30-56 minute forge for an answer available in seconds.

**PRACTICE: a currency checklist needs BOTH questions.**

- **Is the claim still true?** — staleness by contradiction. Detectable by re-measuring the claim.
- **Is the path it prescribes still the cheapest one?** — staleness by workaround. **Detectable ONLY by re-testing the obstacle**, which nobody does, because the rule exists to stop them hitting it.

**So the detector is not a re-read but a deliberate re-attempt of the forbidden thing, on a schedule. When a rule says "you cannot do X, do Y instead", the maintenance question is not whether Y still works — it is whether X still fails.**

**⚠️ AND THE BOUND ON THAT DETECTOR, WHICH IS ALSO A GROWTH MECHANISM FOR THIS VERY FILE (2026-08-20, the doc-currency lane, checking back on the rule it had just been handed).** **A schedule of re-attempts does not exist for this repo's expensive prohibitions.** Re-attempting the vcvarsall recipe costs one command. Re-attempting *"only one `--features cuda` build at a time"* means starting a second and possibly killing a peer's 30-56 minute forge with a `ptxas` allocation failure. Re-attempting *"ALL GPU-touching runs go through `gpu-run`"* means **not** doing that, and the recorded cost of being wrong is a **host-aperture kernel bugcheck**.

**THE RULES MOST LIKELY TO BE OBSOLETE ARE THE ONES WE CAN LEAST AFFORD TO TEST.** So the requirement splits:

- **`X` cheap and safe** → schedule the re-attempt.
- **`X` expensive or destructive** → the rule **MUST RECORD THE MEASURED PRECONDITION** that makes `X` fail, **because a precondition is testable without triggering the failure.** The CUDA-concurrency rule already does this correctly — *"~16 concurrent nvcc survive; an allocation failed near 22"* lets a reader **count processes** instead of causing an OOM. **That clause was doing work nobody had credited it for.**

**AND THE COROLLARY EXPLAINS WHY THIS FILE AND `CLAUDE.md` ONLY EVER GROW: a prohibition with no safe re-attempt AND no recorded precondition is PERMANENTLY UNFALSIFIABLE.** No evidence against it can be gathered without doing the forbidden thing, so it survives on its own authority for as long as the file does — **which is indistinguishable, from the reader's side, from being correct.** Such rules can only accumulate.

**PRACTICE, one sentence: when you write a prohibition, record the measurement that would have to change for it to stop applying.** It is the only thing that lets a later reader retire your rule **without first getting hurt by it.**

**⚠️ THREE MORE SHAPES FROM THE SAME PROGRAM, EACH A DIFFERENT KIND OF STALE (2026-08-20, doc-currency, 35/35 session-prompts + 7/7 specs).**

**(a) A PARENT CONTRADICTED BY ITS OWN CHILDREN — the worst-placed staleness there is.** `step-e-async-execution` reads *"design / scoping — no executor code until Phase A is reviewed"*, while `step-e-a4b-async-completion` says **SHIPPED**, `step-e-phase-c-design` says **SHIPPED**, and `pipelined.rs` is **15,739 lines**. **A stale leaf misinforms whoever reaches it; a stale INDEX misinforms whoever is orienting — which is everyone who does not already know the answer.** The lane's sentence is the durable form: *"a reader who starts at the parent — which is what a parent is for — gets the one stale answer in the set."* **PRACTICE: check parents against their CHILDREN'S STATUS, not only against code.**

**⚠️ AND A RENAME THAT COMPILES IS NOT A RENAME THAT IS DONE — BUT THE REMEDY IS NOT A SWEEP, BECAUSE AN OLD SYMBOL NAME IN PROSE HAS THREE CORRECT TREATMENTS AND TWO OF THEM ARE "DO NOT TOUCH IT" (2026-08-20; class raised by Unpopped, refined by measuring it on Fuel's own 296-file `Lazy` rename).**

Unpopped renamed six gates; **the compiler caught every call site and none of the 31 mentions in doc comments** — `cargo build`, `clippy` and `cargo doc` all clean with all 31 present. **The reason no gate catches it: unlike a broken intra-doc link, a bare name in prose has no referent to check.** `[Foo]` fails when `Foo` dies; `` `Foo` `` does not. **So a rustdoc gate closes the LINKED half and is structurally blind to the UNLINKED half — which is the larger half, because most prose names things without linking them.**

**Measured on Fuel's `Lazy` sweep: ZERO stale mentions.** Rust comments contain only `LazyKvCache`, `LazyPadMode`, `LazyConvTranspose`, `LazyConv` — the four names that legitimately **kept** the prefix (a bare-name collision in their own crate) — and the positive control is that each appears in code (46 / 26 / 21 / 15 occurrences). Markdown holds `LazyTensor` ×10 and `LazyCommunicator` ×2, **neither of which exists in code**, which looks exactly like the predicted residue. **It is not.**

**Every one is prose ABOUT the rename** — including this file's own account of the sweep corrupting a verification control inside a fenced block. **A mechanical rename would have been WRONG in all twelve.** Hence the three-way split:

- **STALE** — the prose means the current thing and names the old one. **Rename it.**
- **HISTORICAL** — the prose is *about* the rename, or about behaviour before it. **Renaming DESTROYS the record, and it self-erases**: the sentence explaining why an old name mattered becomes a sentence about the new name, which was never true.
- **PINNED** — the document describes the code as of a revision. **Renaming makes it FALSE.** Lightbulb's `fuel-api-surface.md` documents Fuel as of `13279179`, where `Tensor` was the **eager** type — a distinct thing, since deleted — so it carries a do-not-rename banner naming the pin.

**THE FAILURE DIRECTIONS ARE NOT SYMMETRIC, WHICH IS WHY THE SWEEP IS THE DANGEROUS MOVE: a missed STALE mention misleads a reader and is findable later; a swept HISTORICAL or PINNED mention DESTROYS EVIDENCE and cannot be detected afterwards, because the result reads as correct.** **PRACTICE: after a rename, grep the old names across `*.rs` and `*.md` — the grep is mechanical, the DISPOSITION IS NOT. A zero is a clean result worth two minutes; a non-zero is a reading task, never a sed.**

**✅ AND THE PREVENTION HALF, WHICH IS A DRAFTING CHOICE MADE BEFORE ANY TOOL RUNS (2026-08-20, Baracuda's observation, relayed).** Baracuda renamed `multi_reduced_panic`/`multi_coord_panic` to `_decline` and produced **zero** candidates — not because the rename was small, but because **they wrote the doc comment as CURRENT BEHAVIOUR (*"why the name is `_decline`"*) rather than as A RECORD OF THE CHANGE.** The two drafts carry the same information to a reader and differ entirely downstream:

- *"`X` was renamed to `Y` because Z"* — names `X` forever, generates a candidate forever.
- *"`Y` is called `Y` because Z"* — names nothing dead, generates nothing.

**So HISTORICAL is partly self-inflicted, and ledger size is partly a choice about how you write rather than only a fact about your history.**

**BUT NOT EVERY HISTORICAL RECORD IS AVOIDABLE, AND THE TEST IS WHETHER THE OLD SYMBOL IS THE SUBJECT.** Ten of Fuel's twelve are the account of a sweep corrupting a verification control — **that record is meaningless without naming what was swept**, and it is the record of why a sweep must not sweep. **PRACTICE: draft as current behaviour by DEFAULT; name the old symbol only when the old symbol is what the sentence is ABOUT.** That is a one-question test at writing time, and it is far cheaper than a disposition at audit time.

**(b) A BROKEN ANCHOR IS NOT A STALE STATUS, AND THE REMEDY IS DIFFERENT.** `lazy-multi-process-inference`'s goal is reviving `_llama_multiprocess_retired`, **which matches 0 files.** **A work item whose OBJECT was deleted needs RE-SCOPING, not resuming** — and it will read as merely *paused* forever, because nothing about a status line reveals that its subject is gone.

**(c) SELF-ASSERTED CURRENCY IS A CLAIM A DOCUMENT MAKES ABOUT ITSELF AND NOTHING CHECKS.** `load-time-incremental-planner` calls itself a **"live spec"**, last reconciled 2026-06-15 — **the same phrase that cost the eager master plan two months.** *"Live spec"*, *"last reconciled <date>"*, and *"current as of"* are all unverified assertions that READ as verification. **Treat them as the opposite of evidence: a document confident about its own freshness is one nobody has had to re-derive.**

**⚠️ AND THE REPORTING RULE THAT MAKES SUCH AN AUDIT MEAN ANYTHING: REPORT THE CLEAN RESULTS BY NAME.** Five of fourteen documents in the final batch needed nothing, and the lane insisted that count as a result. The obvious reason is trust — *"a program that only ever finds rot produces a reader who distrusts everything"* — but the harder reason is the one that matters: **AN AUDIT THAT REPORTS ONLY DEFECTS MAKES "AUDITED AND CLEAN" INDISTINGUISHABLE FROM "NEVER AUDITED."** Same absence-is-invisible family as every other entry in this file: **the clean result is the one that gets dropped, and dropping it destroys the ability to tell COVERAGE from SILENCE.** The strongest of the five was a doc claiming a TODO remained *"with explicit code markers"* — **verified rather than assumed, and the markers were there** (10 in `lazy_mmdit.rs` alone, across 6 files): a claim that could have been waved through in either direction.

**⚠️⚠️ AND WHEN THE AUDIT IS OF PROHIBITIONS SPECIFICALLY, THE CLASSIFICATION MUST FORCE THE CHEAP CASES TO ACTUALLY BE TESTED.** The natural three buckets — *carries a precondition* / *precondition measurable, here's the command* / *needs a ruling* — **let the audit label a rule falsifiable without ever falsifying one**, which is the failure the whole class exists to prevent. **Split the middle: `X` CHEAP AND SAFE → RE-ATTEMPT IT AND REPORT THE RESULT (not the command); `X` EXPENSIVE → give the command and the number it should produce, and do NOT run the forbidden thing.** **A rule you could cheaply test and did not test is still unfalsified, and a tidy classification containing zero new facts is worse than no audit, because it looks like one.**

**⚠️ THAT DETECTOR ONLY EXISTS FOR RULES WHOSE `X` IS CHEAP AND SAFE TO
RE-ATTEMPT — AND THE MOST ENTRENCHED RULES ARE PRECISELY THE ONES WHERE IT
ISN'T.** Re-attempting the vcvarsall recipe from Bash costs one command. But
this repo's expensive prohibitions cannot be re-attempted at all without
inflicting the failure they forbid: *"only one `--features cuda` build at a
time"* is re-tested by starting a second one and possibly killing a peer's
30-56 minute forge with a `ptxas` allocation failure; *"ALL GPU-touching runs
go through `gpu-run`"* is re-tested by not doing that, and the recorded cost of
being wrong is a host-aperture kernel bugcheck. **A schedule of re-attempts is
not available for either.**

**So the requirement splits, and the second half is the one that keeps a rule
retirable:**

- **`X` cheap and safe to re-attempt** → schedule the re-attempt. The rule stays
  falsifiable by doing the forbidden thing on purpose.
- **`X` expensive or destructive** → the rule MUST record the measured
  **precondition** that makes `X` fail, not merely the prohibition — because a
  precondition can be tested without triggering the failure. The CUDA-concurrency
  rule does this correctly: it records *"~16 concurrent nvcc survive; an
  allocation failed near 22"*, and a reader can count processes and cores
  instead of causing an OOM.

**COROLLARY, and it is why a rules file grows monotonically: a prohibition with
no safe re-attempt AND no recorded precondition is permanently unfalsifiable.**
It cannot be retired by evidence, because no evidence against it can be gathered
without doing the forbidden thing. Such a rule survives on its own authority for
as long as the file does — **which is indistinguishable, from the reader's side,
from being correct.**

**⚠️ AND SORT THE UNFALSIFIABLE ONES BY COST-OF-COMPLIANCE, WHICH IS THE DIMENSION THE
CLASSIFICATION ITSELF LACKS (2026-08-20, architect).** Whether a rule can be
falsified is only half the question; the other half is **what obeying it costs
while nobody knows.**

- **Cheap to obey, expensive to violate** → a missing precondition barely
  matters. `-j 4` is here: if it is over-conservative we lose minutes; if it is
  wrong the other way we lose hours to misattributed ICEs. **Retain it, and say
  plainly that it rests on one known-good value rather than a bisected
  boundary** — which is a real thing to stand on and is not the same claim.
- **Expensive to obey AND unfalsifiable** → **the dangerous cell.** A rule that
  taxes every session forever with no way to discover it has stopped being true.
  *"`--features cuda` builds must be launched detached"* and *"one CUDA build at
  a time"* live here. These deserve a measured precondition even when the
  precondition is hard to get.

**So the audit's output is the ORDERING, not the classification.** A `C` that is
cheap to obey is a footnote; a `C` that costs an hour per session is the one to
spend effort on.

**AND SOME MEASUREMENTS MUST NOT BE TAKEN AT ALL.** Measuring `-j 4`'s true
boundary means deliberately racing rustc on a shared box, whose symptom is
**nondeterministic ICEs naming different dependency crates each run** — which
every other lane blames on the branch they are building. **That is not a
measurement, it is an injection into someone else's evidence.** A precondition
you can only obtain by inflicting the failure on people who did not ask for it
is one you decline to obtain, and record as declined.

**PRACTICE when writing a prohibition: record the measurement that would have to
change for it to stop applying.** One clause, written while you still know it —
and it is the only thing that lets a later reader retire your rule without
first getting hurt by it.

---

### THE FAMILY — four variants, four detectors, and a currency audit catches NONE of them

**Consolidated 2026-08-26.** All four share the property that made this class worth naming: ***"is this claim still true?" passes.*** Every sentence in the artifact remains true. **What has changed is outside the sentence**, so re-reading cannot find it. They differ in WHERE the change happened, and therefore in what would detect it.

| # | variant | what changed | the only detector |
|---|---------|--------------|-------------------|
| 1 | **PROHIBITION whose forbidden path stopped failing** | the obstacle went away | **RE-ATTEMPT the forbidden thing** — or, when that is expensive, record the measured PRECONDITION so it can be checked without triggering the failure |
| 2 | **PRESCRIPTION whose recommended path stopped working for a SUBSET of arguments** | the instrument's DOMAIN narrowed | **state the argument shapes the instrument does NOT cover** — a re-attempt does not find it either, because it succeeds on almost everything |
| 3 | **REMEDY verified on the axis it was CHOSEN for and never on the axis it was REPLACING** | nothing — the remedy was always half-wrong | **re-verify the replacement on the property the ORIGINAL was chosen for** |
| 4 | **STATUS whose REFERENT dissolved** | the thing it defines itself against ENDED | **notice the referent is gone** — nothing inside the artifact can tell you |
| 5 | **RULING whose DOMAIN drifted** | nothing — the claim is still true and its referent still alive; it is being APPLIED to a question it never answered | **ask what question the ruling was ANSWERING**, not whether it is still true |

**(1)** is the base case above. **(2)** and **(3)** are the MSYS path-conversion incident: `git show origin/main:.github/…` silently returns zero bytes for a slashed-ref-plus-leading-dot path, and the obvious fix — drop the slash, use `main:` — **cures the mangling by reintroducing the staleness the rule existed to prevent**, since `main:` reads the stale local branch. *"It does not mangle"* and *"it is not current"* are both true, and only the first is tested when you check whether the workaround works.

**(4) IS THE NEWEST AND HAS THE WORST DETECTOR PROBLEM.** Two registry rows carried the status *"not blocking GAP-229's remaining legs."* **GAP-229 finished.** Neither cell was wrong — every claim in both remained true — **and both had stopped saying anything while still READING as a considered position rather than an expired one.** A status expressed as a RELATIONSHIP goes vacuous the moment its referent ends, **and nothing about the cell changes to show it.** Surfaced only because a reader checked the status cells against the world rather than against themselves.

**(5) IS THE NEWEST AND THE ONLY ONE WHERE NOTHING CHANGED AT ALL.** A ruling held that GAP-234 stay unallocated: *the case for taking it was PERISHABLE CONTEXT, and the context was RECORDED, so it does not perish.* **Every word of that stayed true.** But it was an argument against allocating **AGAINST A CLOCK**, and days later it was standing as an argument against allocating **TO AN IDLE LANE** — a question it had never been asked. **The claim did not rot; its DOMAIN drifted out from under it, and the author was the one who moved it.** **Distinguish it from (4) carefully, because they look identical from outside: in (4) the REFERENT dies and the sentence goes vacuous; in (5) the referent is alive, the sentence is sound, and it is simply ANSWERING SOMETHING ELSE.** Neither is visible to a currency audit — in (4) every claim is still true, and in (5) the claim is not merely true but correct. **The only detector is re-deriving what the ruling was FOR, which nobody does for a ruling they agree with.** *(Named by the portfolio PM, 2026-08-26, off the architect's own self-reversal: “I answered the question I had been asked and then let the answer stand for a different one.”)*

**THE PRACTICAL FORM: when you write a status, prefer one that is FALSIFIABLE ON ITS OWN TERMS over one that is true-by-reference.** *"Unallocated; hand-review per site, ~119 sites"* survives its neighbours dying. *"Not blocking X"* does not — **and it degrades into a sentence that passes every check and informs nobody.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ STALENESS BY WORKAROUND: a rule can be TRUE when written, remain TRUE as stated, and still send you to a far more expensive path than the one that now works.** *"Is this claim still true?"* passes on every one of these, so a currency audit cannot see them. Ask the second question: **is the path it prescribes still the cheapest one?** **The detector is not a re-read but a deliberate RE-ATTEMPT of the forbidden thing — when a rule says "you cannot do X, do Y", the maintenance question is whether X still FAILS.** ⚠️ **But the rules most likely to be obsolete are the ones we can least afford to test, so: `X` cheap → schedule the re-attempt; `X` expensive or destructive → THE RULE MUST RECORD THE MEASURED PRECONDITION that makes `X` fail, since a precondition is testable without triggering the failure. **A prohibition with no safe re-attempt AND no recorded precondition is PERMANENTLY UNFALSIFIABLE** — indistinguishable from correct, and it can only accumulate. **When you write a prohibition, record the measurement that would have to change for it to stop applying.** → [`staleness-by-workaround`](#staleness-by-workaround)

## which-number-moves-if-it-became-a-no-op

> **Index line (in CLAUDE.md):** Ask of any mechanism you ship: **"if this silently became a no-op, which number would move?"** If the honest answer is *none*, the metric is not measuring the mechanism — and the mechanism's own greenness is the thing least able to tell you.

**FORMULATED 2026-08-20 by the precision lane, after two independent instances in one afternoon stopped looking like a coincidence.**

**They are the same family and differ in one way worth keeping, because the REMEDIES differ:**

**(a) A FIX WHOSE INERTNESS IS INVISIBLE IN THE METRIC MEASURING IT.** `CpuInvoker::with_seeded_output` was written to stop in-place kernels being verified against an all-zeros target. **An in-place kernel on a zeroed buffer is perfectly bit-stable, so the seeding could have been inert and every downstream test would still have been green** — GAP-222 one level up, inside its own fix. **REMEDY: A DISCRIMINATING FIXTURE.** The test uses `relu_inplace` *because* its output differs from its input only where the input is negative, and the fixture deliberately carries negatives; **a seed of all positives produces identical bytes whether or not the kernel read them.**

**(b) A METRIC THAT IS INVARIANT UNDER TOTAL FAILURE OF WHAT IT MEASURES.** A ratchet counts live-GPU sites that do not call a guard. **If the guard itself stopped refusing — or were refactored to return `Result` and quietly ignored — every "guarded" site would be guarded by nothing and THE COUNT WOULD NOT MOVE BY ONE.** The ratchet stays green, complete, and meaningless. **REMEDY: A FOUNDATION CHECK ON THE GUARD ITSELF**, as a separate assertion, because no amount of care in the counting can reach it. (Vulkane's observation, made against their own scanner.)

**So one question — *if this mechanism silently became a no-op, which number would move?* — and the answer names the remedy: a fixture that discriminates (a), or an assertion about the foundation (b). "None" means you have not instrumented the mechanism at all, only its surroundings.**



**⚠️ (d) READ THE SABOTAGE'S KILL **COUNT**, NOT JUST ITS COLOUR — A PARTIAL RED IS THE ONLY SIGNAL THAT SEPARATES A LIVE SUITE FROM A SUITE WITH DEAD INPUTS (2026-08-20, and it found the largest evidence-quality defect of the day).**

The standing discipline is *sabotage the mechanism, confirm the gate goes RED*. **That is not enough.** A `Gather` reference was sabotaged to ignore its indices and **failed 4 of 9 registrations.** A correct sabotage of a live suite fails **9 of 9**. **The five that survived were the INTEGER dtypes — and they survived because every integer probe tensor was ALL ZEROS**, so a reference that had stopped gathering still agreed with a kernel that gathered: a permutation of zeros equals a copy of zeros.

**Cause: `fill_deterministic` produces floats in `[-0.5, 0.5)` and the integer arms convert with `as`, which truncates toward zero. Measured: U8, I8, I16, U32, I32, I64 each collapse all four probe values to ONE byte pattern.** Every integer-dtype ledger record — bit-stability and bound alike — had been earned against a degenerate input. **Probes ran, comparisons passed, records were written, and the tensor they all agreed about was zeros.**

**THE PORTFOLIO PM'S FORMULATION IS THE RULE: a sabotage that kills EVERYTHING tells you the suite is alive; one that kills only SOME of what it should is the only signal that distinguishes a live suite from a suite with DEAD INPUTS.** A full red and a partial red are both "the gate fired", and only the second carries the information.

**AND THE TAXONOMY IS WORTH KEEPING: every other entry in this file is an INSTRUMENT problem — a wrong gate, a stale set, a truncated query, an assertion that cannot see. THIS ONE IS THE MATERIAL THE INSTRUMENT WAS FED.** The gate was correct, the comparison was correct, the record was true. **The input was degenerate, and nothing downstream of an input can detect that.**

**Corroborating consequence, which is how you know the fix was real: non-degenerate probes immediately exposed 16 disagreements, ALL of them the REFERENCE rather than the kernel** (Rust int→int truncates where float→int saturates). **Zero is in range for every target, so saturation and truncation AGREE on it** — one defect had been perfectly concealing another, and the concealed one was in the instrument.

**PRACTICE: state the expected kill count before running a sabotage, and treat a SHORTFALL as a finding about the inputs. `n of m` where `n < m` is not a weaker red; it is a different red.**

**⚠️ (c) A THIRD SHAPE, AND THE ARCHITECT PRAISED AN INSTANCE OF IT BEFORE IT WAS CAUGHT (2026-08-20): AN ASSERTION KEYED TO A MESSAGE **STRING** CANNOT DETECT A BEHAVIOUR CHANGE THAT REWRITES THE MESSAGE.**

A characterisation test pinned the *current wrong* behaviour of a verifier and carried a "stale detector" — `assert!(!detail.contains("candidate 1 vs reference 2"))` — whose job was to fail once the verifier was fixed, forcing the fix to arrive with its own assertion. **The fixed verifier emitted a DIFFERENT sentence, so the negative assertion held VACUOUSLY. The test passed before AND after.** It was called *"the best-designed test I have seen today"* by the architect roughly an hour before its author withdrew it.

**Same family as a report line that stopped depending on its measurement — aimed at a test instead of at a log.** A negative assertion over prose is satisfied by any rewording, **including the rewording the fix itself performs**, so it is at its weakest exactly when it is supposed to fire.

**REMEDY: assert on NUMBERS the wording cannot carry.** The replacement pins that `1.0f16` (`0x3C00`) and `2.0f16` (`0x4000`) are **exactly 1024 f16 ULP** apart — **from both sides** (passes 1024, fails 1023), plus an identical-buffers control so a verifier that always reported a large distance cannot satisfy it. **1024 is unreachable by reading those 8 bytes as two f32s: it discriminates by construction rather than by wording.**

**AND THE SABOTAGE OF THE REPLACEMENT WAS ITSELF INVALID ON THE FIRST TRY** — widening a width field did not reproduce the defect, because the decoder reads by dtype and merely compared fewer elements; the test passed, and reading that as *"the replacement doesn't discriminate either"* would have been a second wrong conclusion in the opposite direction. **The valid sabotage decodes every output as `F32`.** **A positive control can itself be inert — twice in one fix.**

**Companion rule from the same change, on the opposite failure: REFUSING A CASE THE OLD CODE ANSWERED CORRECTLY IS A REGRESSION, NOT A SAFETY MEASURE.** A first draft refused two bound kinds as "not expressible from a total-order key"; they were expressible from the *value*, which the author had not carried yet. It broke a pre-existing test, which is how it was found. **"Refuse rather than approximate" is a rule about UNIMPLEMENTED things, not a licence for not implementing them.**

**Why this is not merely "write better tests": in both instances the suite was green, the count was correct, and the work was real. Nothing in the output was wrong. The defect is that THE OUTPUT WOULD HAVE BEEN IDENTICAL HAD THE MECHANISM DONE NOTHING** — the same property as an unheld mutex, a zeroed probe target, and an unrun CI job. **Much of this file is one defect wearing different clothes: a result that cannot distinguish success from absence.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ ASK OF ANY MECHANISM YOU SHIP: "IF THIS SILENTLY BECAME A NO-OP, WHICH NUMBER WOULD MOVE?" If the honest answer is NONE, the metric is not measuring the mechanism.** Two shapes, two remedies: **(a) a fix whose inertness is invisible in its own metric** (an in-place kernel on a zeroed buffer is perfectly bit-stable, so the seeding fix could have been inert and every test still green) → needs a **DISCRIMINATING FIXTURE**; **(b) a metric invariant under total failure of what it measures** (if the guard stopped refusing, the count of unguarded sites would not move by one) → needs a **FOUNDATION CHECK on the guard itself**. In both, the suite is green, the count is correct, and **the output would have been IDENTICAL had the mechanism done nothing**. → [`which-number-moves-if-it-became-a-no-op`](#which-number-moves-if-it-became-a-no-op)

## the-fingerprint-filename-is-not-the-fingerprint

> **Index line (in CLAUDE.md):** A cargo fingerprint is keyed on **features + flags**, not just version — so `.fingerprint/<crate>-*` **existing** does not mean the cache is warm for *your* invocation. **"Same pin ⇒ warm" is the wrong inference; warmth is PER FEATURE SET.** Checking for the file reads the **filename**; the hash is what decides.

**FOUND 2026-08-20, self-reported by the lane it cost, and it is the mirror of a rule already in this repo.**

A lane paid a **66m26s** cold forge, then rebased 26 commits and checked that `target/debug/.fingerprint/baracuda-kernels-sys-*` still existed. **It did, and they reported "warm forge, best case".** The next build re-forged from scratch — `Compiling baracuda-kernels-sys v0.0.1-alpha.79`, 12 live `nvcc`.

**The pin matched. The FEATURE SET did not.** The 66-minute build compiled `baracuda-kernels-sys` as a dependency of plain `fuel-cuda-backend` under **default features**; the new one pulls it under **`fuel-dispatch/cuda` + `fuel-core/cuda`**. Different feature unification → different fingerprint hash → different directory → **a full re-forge, with a file present the whole time that looked like proof of the opposite.**

**THIS IS THE EXACT MIRROR OF THE CLEARING RULE ALREADY RECORDED HERE:** *clearing a fingerprint by guessed path can match nothing, leaving you warm while believing you forced a rebuild.* **Same object, opposite direction:**

- **Clearing:** you believe you went COLD and you are WARM → you read a stale green as a fresh verification.
- **Checking:** you believe you are WARM and you go COLD → you budget minutes and spend an hour, and a build that looks hung is just building.

**In both, the mistake is treating the fingerprint DIRECTORY NAME as the fingerprint.** It is a hash of features and flags, and neither `ls` nor a glob can see that.

**PRACTICE: never infer warmth from a version pin or from a file's presence. If warmth matters (on this box a cold baracuda forge is 30-56 minutes), warm it with the SAME invocation you intend to run** — same `-p` set, same `--features`, same target kinds — **or budget cold.** And when allocating work on "whoever has a warm forge" grounds, **name the FEATURE SET, not just the crate**: an allocation that says "warm forge" is underspecified in precisely the way that produces this.

**Corollary worth keeping about the cost: one cold forge under the RIGHT feature set is cheaper than a warm cache under the wrong one, which never helps and hides that it never helped.** The lane's own conclusion — *"my forge-verify cache was for the wrong feature set and never would have helped increments anyway"* — is the useful form: **the warm cache they thought they had was not merely stale, it was for a different question.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ THE FINGERPRINT FILENAME IS NOT THE FINGERPRINT — a cargo fingerprint is keyed on FEATURES + FLAGS, not just version.** `.fingerprint/<crate>-*` **existing** does not mean warm for YOUR invocation: a lane's 66-minute forge under plain `fuel-cuda-backend` default features left a file present while `fuel-dispatch/cuda` + `fuel-core/cuda` re-forged from scratch. **"Same pin ⇒ warm" is the wrong inference; warmth is PER FEATURE SET.** This is the exact MIRROR of the clearing rule above: **clearing** leaves you warm believing you went cold (stale green read as fresh); **checking** leaves you cold believing you were warm (minutes budgeted, an hour spent, a build that looks hung). **Warm with the SAME invocation you intend to run, or budget cold — and when allocating on "warm forge" grounds, NAME THE FEATURE SET, not just the crate.** → [`the-fingerprint-filename-is-not-the-fingerprint`](#the-fingerprint-filename-is-not-the-fingerprint)

## evidence-that-is-not-independent

> **Index line (in CLAUDE.md):** Two artifacts agreeing is not two pieces of evidence **if one was written from the other** — and data that arrives ADJACENT to a question is not thereby the question's population. **And two INDEPENDENT implementations can be wrong the SAME way when the failure mode is inherent to the TASK — independence of authorship does not give independence of error.** All three feel like corroboration and none is.

**Two mechanisms found 2026-08-20, by two different people, both producing a confident wrong answer from something that looked like support.**

### (a) DERIVED CORROBORATION — the second copy was written from the first

`CLAUDE.md` and `.github/workflows/rust-ci.yml:14-18` carried the **identical** false premise: that both metal crates are *"deliberately kept OFF `default-members` so a plain build works without Apple toolchains"*. Measured: **`fuel-metal-backend` IS in `default-members` and builds clean on Windows**; only `fuel-metal-kernels` is killed by `objc2`.

**The doc-currency lane's framing is the rule: a reader who checks one against the other finds agreement and stops.** The agreement is worth nothing, because the second copy was derived from the first. **THAT IS STRICTLY WORSE THAN A SINGLE UNCORROBORATED CLAIM** — a lone claim invites verification; a matching pair closes the question.

**Third instance in one day of a fact living in two places with one maintainer's attention.** The others: the `baracuda-kernelgen` pin (retraction in `Cargo.toml`, accusation left standing in `CLAUDE.md`), and a rename that updated a panic's own MESSAGE while leaving an allowlist entry quoting that message.

**PRACTICE: when two artifacts agree, ask whether one was written FROM the other before counting it as confirmation. Independent corroboration means independently DERIVED, not merely separately STORED.**

### (b) ADJACENT DATA ADOPTED AS THE POPULATION

A lane reported five op families for the **bit-stable-blocked** class in the same message as a question about the **`max_ulp`-blocked** class. **The architect reasoned from the five to a conclusion about the 84 — and the two sets are DISJOINT.** One is the ops nothing could *probe*; the other is the ops whose declared *bound* is unbacked. **There was never a reason they should overlap; they arrived adjacent.**

**This is specifically a COORDINATOR hazard, and worth naming as one: someone who does not run the measurements receives a SELECTION, and a selection presented together reads as one population.** The numbers in a lane's report were chosen for that report, not for the question you are about to ask of them.

**PRACTICE: name the population before answering a question about it — out loud, in the reply. "The 84 are ops whose declared bound is unbacked" would have caught this before the reasoning started.** And when handing someone data plus a question, say which data the question ranges over.

**Note the shape both share with the rest of this file: nothing in the OUTPUT looked wrong.** Two agreeing documents look like verification; two adjacent numbers look like one dataset. **The defect is in the provenance, which is not visible in the artifact.**

---

### (c) CONVERGENT ERROR — two INDEPENDENT implementations, wrong the same way, because the failure mode is inherent to the TASK

**2026-08-26.** Two people, working separately and neither reading the other's code, each wrote a counter for **unescaped** `|` characters in `docs/gaps.md` — one in `awk` with a `gsub`, one in a heredoc-mangled Python regex. **Both were wrong. Both were wrong IN THE SAME DIRECTION** (inflating the count), and **their two wrong answers agreed with each other.**

**That agreement was far more persuasive than either answer alone, and it was one message away from being filed as a confirmed defect in a gate that was working correctly.** The gate's `NONE` was right; every flagged row carried exactly six unescaped pipes, the excess being escaped `\|` inside prose, which the gate deliberately skips.

**THIS IS NOT DERIVED CORROBORATION — the distinction matters because the defence differs.** In (a) the second artifact was *copied from* the first, so checking provenance exposes it. **Here the implementations were genuinely independent and the provenance is clean.** What they shared was the TASK, and **the task has a failure mode that is easy to hit and hard to see: forgetting the escape case is precisely what is easy to get wrong about counting delimiters.** So two competent independent attempts converge on it, *because* they are competent attempts at the same hard thing.

> **INDEPENDENCE OF AUTHORSHIP DOES NOT GIVE INDEPENDENCE OF ERROR.**

**And the self-diagnosis was wrong too, one level down** — the author first explained their mangled regex as *"matches empty and removes nothing"*. Measured, it is an alternation of *(literal backslash)* OR *(empty)*, so it **strips the backslash and LEAVES the pipe**, which then counts as a separator. **The count was INFLATED, not unchanged.** A wrong diagnosis of a real error, corrected by printing it rather than reasoning about it.

**THE TEST THAT SEPARATES THE THREE MECHANISMS: ask where the shared blind spot LIVES.** In a common SOURCE → derived, and provenance exposes it. In a common TASK → convergent, and **provenance is clean while the agreement is still worthless.** In a common ADJACENCY → the data merely arrived together. **All three feel like corroboration and none is.**

**PRACTICE: when two instruments agree, ask what they SHARE before treating it as confirmation — and for a fiddly primitive, EXTRACT IT ONCE AND CALL IT EVERYWHERE, with the incident in its docstring.** The fix here was one function replacing every ad-hoc reimplementation, and the reason is written at the site **so nobody later "tidies" it back into two.** *Fix the generator, not the output* — where the generator is the temptation to re-derive a primitive that already exists.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ TWO ARTIFACTS AGREEING IS NOT TWO PIECES OF EVIDENCE IF ONE WAS WRITTEN FROM THE OTHER — and data that arrives ADJACENT to a question is not thereby the question's population.** A reader who checks `CLAUDE.md` against a CI comment finds agreement and stops; **a matching pair CLOSES the question where a lone claim would have invited verification.** Independent corroboration means independently DERIVED, not separately STORED. And a coordinator receives a SELECTION — numbers chosen for the report you are reading, not for the question you are about to ask of them — so **name the population before answering.** → [`evidence-that-is-not-independent` · **third mechanism added 2026-08-26: CONVERGENT ERROR — two INDEPENDENT implementations wrong the SAME way because the failure mode is inherent to the TASK; their agreement nearly filed a false defect against a working gate.**](#evidence-that-is-not-independent)

## a-reference-must-be-able-to-indict-itself

> **Index line (in CLAUDE.md):** **A truth-reference must be ABLE to come out wrong — a harness that can only ever blame the kernel is an agreement check wearing a bound's name.** And keying an assertion to a *measurement* is not enough: **it has to be the measurement the property is ABOUT.** Two measurements that coincide today are indistinguishable as anchors until they diverge, so **a coincidence gets recorded as an invariant.**

**Both found 2026-08-20 by the precision lane, in the same change, and each is worth more than the coverage number it came with (84 → 24 downgrades, 134 claims, 60 entries).**

### (a) The reference was wrong before the kernel was

Four `Cast [T, Bool]` comparisons failed — candidate `0x01`, reference `0x00`. **The KERNEL was right.** The independently-written reference truncated the float before testing against zero, so `0.5` became `false`.

**That is the outcome a truth-reference must be CAPABLE of.** A comparison harness whose only possible verdict is "the kernel is wrong" is not measuring against truth — **it is an agreement check that has been given a bound's name**, the same defect as recording a differential as `max_ulp`. **The reference indicting itself is the evidence that it is a reference at all.**

**PRACTICE: when a reference and a candidate disagree, the FIRST hypothesis is that the reference is wrong — and if that has never once happened, ask whether the harness could express it.**

### (b) An assertion anchored to the wrong measurement

Earlier the same day this lane taught **key an assertion to the MEASUREMENT, not to the number the measurement produced** — their flip harness followed `199 → 219` untouched while a registry row did not. **Then they keyed one to the wrong measurement.**

`lost_by_flip.len() == bit_stable_entries` tied the **flip's blast radius** to the **total backed set**. **It held only because the two coincided at 199 — A COINCIDENCE RECORDED AS AN INVARIANT.** Backing a second claim for 60 more entries separated them (flip 219, backed 279) and it fired, reporting that the selector and the ledger disagree. **They do, and they are supposed to**: the flip is a narrow selector (184 sections), sabotage strips wholesale (418). **The real identity is with the SABOTAGE arm (279 == 279); what matters about the flip is only that it is non-empty.**

**PRACTICE: name the property, then ask which measurement is ABOUT it — not which one currently equals it.** Two quantities that agree today are indistinguishable as anchors, and **the wrong one gives a green that is right by accident and a red that diagnoses the wrong situation confidently.**

### Two smaller ones from the same change, both about labels

**A FIXTURE THE WORK UNDER TEST IS ACTIVELY REMOVING IS NOT A FIXTURE.** Two tests named one entry as their *downgraded-entry* fixture; the change earned both of that entry's claims, so it stopped being downgraded and neither test could observe its property. **Both now LOCATE a downgraded entry, and if none exists anywhere they report that as the finding rather than passing.**

**A REFUSAL WEARING AN ERROR'S LABEL.** 22 casts failed deep inside `encode` as `invoke error: no width for F8E4M3`. **Same outcome for the ledger, different thing entirely for a reader** — an unsupported format is a DECLINE and belongs up front, not a failure surfacing from the middle of a call stack.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A TRUTH-REFERENCE MUST BE ABLE TO COME OUT WRONG** — a harness whose only possible verdict is that the kernel is wrong is an **agreement check wearing a bound's name**. When reference and candidate disagree, the FIRST hypothesis is that the reference is wrong; **if that has never once happened, ask whether the harness could express it.** ⚠️ **And keying an assertion to a MEASUREMENT is not enough — it must be the measurement the property is ABOUT.** `lost_by_flip == bit_stable_entries` held only because both were 199: **a coincidence recorded as an invariant**, which later fired with a confident diagnosis of the wrong situation. **Two quantities that agree today are indistinguishable as anchors.** → [`a-reference-must-be-able-to-indict-itself`](#a-reference-must-be-able-to-indict-itself)

## a-gate-cannot-source-its-negative-case-from-the-defect

> **Index line (in CLAUDE.md):** **A gate's “it fires” test must not source its negative case from data the work is actively removing** — a fixture, a search SET, or a named unsupported case all expire the same way, and they expire *by the program succeeding*. **Construct the negative case instead: an empty ledger backs nothing, permanently.** ⚠️ And a **WRONG** assertion that fires can settle a question the right one could not ask.

**FOUR instruments fired on the program SUCCEEDING in one increment (2026-08-20/21), all belonging to the lane that wrote them, after THREE prior rewrites of the same two tests.**

- a refusal test named `F8E4M3` as its unsupported dtype — **and `F8E4M3` became supported**;
- two register-gate tests searched for a downgraded entry — **and no downgraded entry existed anywhere**;
- two censuses asserted a **non-empty residue** — and the residue reached zero.

**The progression is the lesson, because each rewrite was a smaller version of one mistake.** First a hand-picked FIXTURE (`add_f32`) expired when the work earned its claims. Rewritten to *locate* one — then the hand-picked SEARCH SET (two contracts) expired the same way. Rewritten to read every contract from disk — then the assumption that a residue EXISTS expired. **A fixture, a search set, and an existence assumption are the same defect at three scopes: all three source the negative case from the thing being fixed.**

**PRACTICE: CONSTRUCT the negative case.** An empty ledger backs nothing and always will; a synthetic entry declaring an unearnable claim is downgraded forever. **And where a census must assert on real data, assert ZERO rather than non-empty** — which is strictly stronger and fires the moment a contract declares a claim nobody earned, **or a kernel revision changes and silently un-earns every claim keyed to it.**

### And a wrong assertion can be an instrument

Replacing an older cause-assertion, the lane wrote a **biconditional** — *UNAUDITED **iff** downgraded* — and **it fired immediately.** The failure is the finding: **downgrade is not the only route to UNAUDITED.** An entry that declares no machine-checkable claim arrives UNAUDITED **without ever being rejected**, so only one direction is an invariant (a downgrade implies UNAUDITED; the converse is false by construction).

**That measured something the registry row had explicitly declined to decide.** The row listed two readings that fit the arithmetic equally well and refused to pick between them; **the wrong assertion picked one, by failing.** Measured at entry level: **303 backed, 320 UNAUDITED, 0 downgrades** — the 320 declare nothing at all.

**So: an assertion strong enough to be WRONG is worth more than one weak enough to be safe** — it can only fail informatively, and a biconditional fails by naming which half is false. **This is the constructive twin of *a coincidence recorded as an invariant*: there, two quantities that agreed were anchored together and the tie was silent; here, two that were ASSUMED to agree were asserted equal and the disagreement spoke.**

### A footnote worth its own line

**A failure message that carries only a DISTANCE cannot diagnose the disagreement.** The lane's own encoder lost precision *inside a precision harness* — a brute-force nearest search computed distance in f64 where the f64 ULP at ~4.5e18 is ~512, so two candidates **rounded to the same distance**, a strict ordering became a spurious tie, and the tie-break took the wrong pattern. **It was diagnosable only because the message had just been changed to carry RAW BYTES: *“2 ULP apart” names the SIZE of a disagreement, not its CONTENT.***

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A GATE'S “IT FIRES” TEST MUST NOT SOURCE ITS NEGATIVE CASE FROM DATA THE WORK IS ACTIVELY REMOVING.** A hand-picked FIXTURE, a hand-picked SEARCH SET, and an assumption that a RESIDUE EXISTS are the same defect at three scopes — **all three expire by the program SUCCEEDING** (four instruments did, in one increment, after three rewrites). **CONSTRUCT the negative case: an empty ledger backs nothing, permanently** — and where a census must assert on real data, **assert ZERO rather than non-empty**, which also fires when a kernel revision silently un-earns every claim keyed to it. ⚠️ **And an assertion strong enough to be WRONG can settle what a safe one cannot ask** — a biconditional fired and thereby picked between two readings a registry row had declined to choose. → [`a-gate-cannot-source-its-negative-case-from-the-defect`](#a-gate-cannot-source-its-negative-case-from-the-defect)

## a-correct-total-can-hide-a-wrong-distribution

> **Index line (in CLAUDE.md):** **A pre-declared delta that holds EXACTLY is necessary and NOT sufficient — a correct total can contain a wrong distribution, and the count that justifies a change is structurally unable to see a defect that conserves it.** Check the delta AND its shape. ⚠️ And when a gate fires on your tooling, **MOVE THE TOOL, do not teach the gate an exception** — every exemption makes the gate weaker and its claim less true.

**Both found 2026-08-21 by the precision lane, inside the change that closed GAP-228(a) — 240 CPU entries moved from a bulk fill to contract + earned record (UNAUDITED 320→80, backed 303→543).**

### (a) The delta held exactly, and the defect was inside it

The architect's gate was a **pre-declared count**: *the fill must have exactly 240 fewer entries afterwards; a shortfall is a finding rather than a smaller success.* **It held. Exactly 240.**

**And on its first run the generator had attached the EMPIRICAL basis clause to 2 `matmul` sections that were already `audited: true` by SOURCE reasoning** — attaching one kind of evidence to an attestation earned another way. **Same-name-different-strength, produced by the tool built to avoid it.**

**The entry-level delta was 240 regardless, because the defect CONSERVED the quantity being counted.** The two mis-clothed sections were already audited, so they never entered or left the UNAUDITED set. **The number that justified the whole change was structurally unable to see the flaw the change introduced.**

**Caught by comparing flips to clauses PER FILE — 4 flips, 6 clauses.** That comparison is now part of the run.

**PRACTICE: a pre-declared total is necessary and not sufficient. Pair it with a SHAPE check — per-file, per-op, per-class — chosen so that a defect which conserves the total cannot conserve the shape.** The delta answers *did the right AMOUNT move*; only the shape answers *did the right THINGS move*.

### (b) A gate fired on the tooling, and the first two fixes were BENDS

The single-writer gate (GAP-210) correctly refused a second writer into the ledger's directory. **The gate's CLAIM is narrower than its PREDICATE** — a contract rewriter is not the ledger — so the temptation is to encode the difference as an exception.

**Two were tried and both were bends:** an exemption keyed on a **filename**, then a narrower *"the file must not mention the ledger"* — **which failed because the seeder's own `#[ignore]` string legitimately names it.** **Each bend made the gate weaker and its claim less true**, and the second failed on a *correct* usage, which is how you know the predicate was being tortured rather than refined.

**Resolution: the WRITER MOVED** — out of the scanned directory entirely, to `fkc/contract_audit_flip.rs`. **The exception is REMOVED rather than ENCODED**, and it turned out to be the right home on the merits anyway: `verify/` is the verification seam, and this tool *consumes* the ledger rather than producing it.

**PRACTICE: when a gate fires on your own tooling, the first question is whether the TOOL is in the wrong place, not whether the GATE needs an exception.** An exemption is permanent, invisible in the gate's stated claim, and compounds — **a moved file is none of those things.**

### A footnote: the born-red was available in a stronger form than the obvious one

The obvious born-red for *"these entries no longer depend on the fill"* is **delete the fill and see what survives** — destructive, and it perturbs everything else. **The contract-derived table is built by `register_into`, which never applies the fill at all** — so the backed count measured there **IS** the post-retirement number, obtained by reading a path that already excludes the fill rather than by removing it. **Look for a code path that already lacks the thing you were about to delete.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A PRE-DECLARED DELTA THAT HOLDS EXACTLY IS NECESSARY AND NOT SUFFICIENT — a correct TOTAL can contain a wrong DISTRIBUTION.** A generator attached the empirical basis clause to 2 sections already audited by SOURCE reasoning, **and the delta was exactly 240 anyway, because the defect CONSERVED the quantity being counted** — the number justifying the change could not see the flaw it introduced. Caught by flips-vs-clauses **per file** (4 vs 6). **Pair every pre-declared total with a SHAPE check chosen so a total-conserving defect cannot conserve the shape.** ⚠️ **And when a gate fires on your own tooling, MOVE THE TOOL rather than teach the gate an exception** — two bends were tried (a filename exemption, then "must not mention the ledger", which failed on a legitimate `#[ignore]` string); **each made the gate weaker and its claim less true.** An exemption is permanent and invisible in the gate's claim; a moved file is neither. → [`a-correct-total-can-hide-a-wrong-distribution`](#a-correct-total-can-hide-a-wrong-distribution)

## eliminating-one-hypothesis-does-not-support-the-next

> **Index line (in CLAUDE.md):** **A test that correctly ELIMINATES one hypothesis does not thereby SUPPORT the next one.** Refuting *“it is X”* leaves *“it is Y”* exactly as unevidenced as before — and the relief of having ruled something out is what makes the unearned conclusion feel earned. **Two people made this error on the same artifact, hours apart, from two different correct tests.**

**2026-08-21, `fuel-ci-fix`: 655 uncommitted tracked files, five days old, 120 commits behind main, claimed by nobody — the only uncommitted work anywhere in the portfolio.** The question was whether it was lost authored work.

**The PM's test (01:58): does the tree MATCH `origin/main`?** It differed by 897 files → concluded *real edits*. **A binary match test cannot see “an OLDER main” — it renders it identically to “someone's edits.”** Correct test, correct result, wrong inference.

**The architect's test (03:50): is the diff WHITESPACE-ONLY?** Hypothesis was *an abandoned rustfmt sweep, already superseded.* `git diff -w --ignore-blank-lines` → **650 of 655 files still carried real changes**, killing it. **And that refutation was then read as support for “real authored edits” — which was also wrong.**

**What it actually was: a working tree brought forward to a newer commit without the branch ref moving. Its content was main MINUS that day's commits.**

### The discriminator that worked is not a match test

**(1) THE DIRECTION OF THE DIFF.** `main` had **~21k lines MORE** than the tree, and the largest gaps were *that day's work missing from it* — the verified ledger `+3825/-17396` in main's favour, `probe_recipes.rs` −827, `seed_cpu_ledger.rs` −759. **Content flowing the wrong way is not what authored work looks like.**

**(2) THE SHAPE OF THE SMALL DIFFS.** **415 of 906 files differed from main by 1-3 lines, and the sampled difference was a single line: an SPDX header added to main that day.** That is **the fingerprint of one commit's absence**, not of anybody editing.

**Replacement check, computable in one pass and immune to the older-main collapse: *is the MASS of the diff in main's favour, and do MOST files differ by a handful of lines?*** — the PM's formulation, adopted.

### Why this is worth a rule and not just an anecdote

**Both tests were correct. Both results were correct. Both inferences were wrong, in the same direction, from opposite evidence** — one from *“it does not match”*, one from *“it is not whitespace”*. **The common step is treating the elimination of a hypothesis as evidence FOR whatever you were going to conclude next**, and it is seductive precisely because a refutation feels like progress. **Ruling out X narrows the space; it does not populate it.**

**PRACTICE: after a test refutes a hypothesis, state the NEXT hypothesis as a hypothesis and ask what would distinguish it — do not let the refutation carry it.** And prefer a discriminator that **measures a direction or a distribution** over one that asks **match / no-match**: a binary test collapses every not-matching state into one answer, and the states you most need to tell apart are all on that side.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A TEST THAT CORRECTLY ELIMINATES ONE HYPOTHESIS DOES NOT THEREBY SUPPORT THE NEXT ONE.** Two people, hours apart, on one artifact: *“does it match main?”* (no → concluded real edits) and *“is it whitespace-only?”* (no → concluded real edits). **Both tests correct, both results correct, both inferences wrong** — the tree was **an OLDER main**, and a binary match test renders that identically to somebody's edits. **Ruling out X narrows the space; it does not populate it**, and a refutation FEELS like progress. **Prefer a discriminator that measures a DIRECTION or a DISTRIBUTION over match/no-match** — here, main having ~21k lines MORE plus 415 files differing by a single SPDX line was the answer. → [`eliminating-one-hypothesis-does-not-support-the-next`](#eliminating-one-hypothesis-does-not-support-the-next)

## a-zero-match-filter-satisfies-the-born-red

> **Index line (in CLAUDE.md):** **`cargo test <filter>` reports `ok` and exits 0 when the filter matches NOTHING** — so **every “run it and watch it fail” step is satisfied by a run in which the test does not exist.** A module never registered, a name misspelled, a forgotten `#[test]` — all produce the same green. **The born-red discipline has a hole exactly at its entry point.** Read the COUNTS, never the word `ok`.

**Found 2026-08-21 by MLMF, live, on a task whose brief predicted a compile error for an unregistered module and got a green run with zero tests. VERIFIED IN FUEL with a positive control:**

```
cargo test -p fuel-ir --test gap_hedges this_test_does_not_exist_anywhere
  test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out     exit 0
(control) cargo test -p fuel-ir --test gap_hedges no_new_prose_hedge
  test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 11 filtered out     exit 0
```

**This repo already refuses `running 0 tests` as a PASS. The sharper form is that it also satisfies a RED**: an implementer can ship tests that never ran **and truthfully report the red-before-green step as performed.** The step that is supposed to prove a test CAN fail is the step the hole lives in.

> **A test runner reports on the tests it FOUND, and has no opinion about the ones it did not.**

### The false premise underneath, and why nobody had extended the rule this way

**The reason this repo's existing rule stopped at *“refuse a zero-test PASS”* is an unstated premise: **A RED RESULT IS SELF-VALIDATING.** A green is understood to need scrutiny — it might be vacuous, filtered, cached, or measuring the wrong thing. **A red is assumed to have earned itself, because what would fake a failure?** The answer is: a run that found no tests, which reports neither pass nor fail and is read as whichever one the reader came for.

✅ **TWO PROJECTS FOUND THAT PREMISE FALSE THE SAME NIGHT, FROM OPPOSITE ENDS, WITHOUT CONTACT.** KISS found it **stated outright in one of its own conventions and refuted by that convention's own later text**. MLMF found **the mechanism that makes it false, in a tool everyone runs**. **Neither knew about the other.**

⚠️ **THAT IS WHAT REAL CORROBORATION LOOKS LIKE, AND IT IS WORTH CONTRASTING WITH THE FALSE KIND RECORDED IN [`evidence-that-is-not-independent`](#evidence-that-is-not-independent) EARLIER THE SAME NIGHT.** There, two artifacts agreed because **one had been written from the other** — and the agreement closed the question instead of opening it. **Here, two agree because they were derived independently, by different methods, about the same premise.** **The test is not whether two sources agree; it is whether either could have been produced without the other.**

**PRACTICE: a born-red report must carry the counts and the assertion — *“1 failed, 52 filtered out, at assertion X”*. *“It failed as expected”* is a VERDICT, and a verdict is exactly what the zero-match run also produces.**

✅ **FUEL'S BORN-REDS TONIGHT WERE NOT EXPOSED, and the reason is worth keeping rather than the relief:** every reported pair carried a nonzero count — GAP-214 inc 2 `FAILED. 0 passed; 1 failed; 2 filtered out`, inc 3 `0 passed; 1 failed; 3 filtered out`, the greens `1 passed`. **A `1 failed` cannot come from a zero-match run.** The defence was already standing because the architect kept asking for the `test result:` line instead of the exit code — **an instrument requirement that turned out to close a hole nobody had named yet.**

**Two adjacent findings from the same MLMF task, both worth carrying:**

**A FALSE COMMENT WITH A COMPLETELY DEAD CONTROL.** A brief claimed that deleting an allocation bound would abort on a `u32::MAX` count. Measured: `try_reserve(4294967295)` for a `Vec<u64>` **returns `Ok` and commits 34 GB** — the abort could never happen, because `try_reserve` exists to avoid it. **So the explicit bytes-remaining bound is the entire defence and the suite as briefed could not detect its removal.** And their own earlier code already documented the correct behaviour, from a sabotage that had corrected the same misconception — **a mistake re-made against a fix its author had already written down.**

**A CONTROL THAT WAS STRUCTURALLY IMPOSSIBLE RATHER THAN WEAK.** A predicted `u32`/`u64` field swap would kill two tests — **except the swap consumes the same twelve bytes in either order, so no position assertion anywhere can see it.** That is *a correct total hides a wrong distribution* with **POSITION** as the projection the defect leaves invariant: **second domain the rule has covered without modification.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ `cargo test <filter>` REPORTS `ok` AND EXITS 0 WHEN THE FILTER MATCHES NOTHING — so a zero-match run SATISFIES THE BORN-RED'S “watch it fail” STEP.** Verified here: a nonexistent filter gives `ok. 0 passed; …; 12 filtered out`, exit 0. **A module never registered, a misspelled name, a forgotten `#[test]` — all green.** The discipline's hole is at its entry point. **READ THE COUNTS, NEVER THE WORD `ok`: a born-red report must say “1 failed, N filtered out, at assertion X”.** *“It failed as expected”* is a VERDICT, and a verdict is what the zero-match run also produces. **A test runner reports on the tests it FOUND and has no opinion about the ones it did not.** → [`a-zero-match-filter-satisfies-the-born-red`](#a-zero-match-filter-satisfies-the-born-red)

## a-restated-reason-is-never-re-derived

> **Index line (in CLAUDE.md):** **A stale MEASUREMENT gets caught because someone eventually re-runs the query. A STALE JUSTIFICATION NEVER GETS RE-RUN — restating it costs nothing and reads as identical to having checked.** And distinguish the two failures, because they need different fixes: **an EXPIRED reason is a claim that WAS true (wants a re-check schedule); a reason that NEVER APPLIED is a claim that never was (wants the justification DERIVED at the moment it is stated).**

**2026-08-21, and the architect's own instance is the worked example.** GAP-227 (the last red CI gate) was held on the stated ground that fixing it required `rustup update stable`, **which would invalidate a warm CUDA forge while a lane was mid-verify on a UB fix.**

**The PM raised it as a hold whose reason had EXPIRED — the lane had finished. The truth was worse: the reason NEVER APPLIED.** `1.98.0` was already installed as a **named toolchain**, so the fix never required touching the `stable` alias at all — **and the architect had measured that personally, hours before giving the reason.**

**The mechanism, which explains every other instance in this file:**

> **I re-stated a reason instead of re-deriving it.**

**A number invites re-measurement; a justification does not.** Repeating *“we can't, because X”* is free, produces no artifact, and is indistinguishable from having just checked X. **That is why stale prohibitions, stale holds and stale blockers all outlive stale numbers** — not because people are more careless with reasons, but because **nothing about a reason prompts a re-run.**

**PRACTICE: when you state a reason for NOT doing something, derive it in that moment or say you are quoting it.** *“Held — because a forge is warm (measured now)”* and *“held — because a forge was warm (as of four hours ago)”* are different claims, and only the second is honest when you have not looked.

### And ANNOUNCE CLOSURES, not only allocations

**The standing fix for invisible allocations — announce at the point of allocation — has a half that was missing: announce at the point of CLOSURE too.** Twice in one night a lane proposed work from a candidate list containing an item the architect had already **closed and not mentioned**, costing them a slot on a three-item list they had measured before proposing. **A closed item is as invisible as an unannounced allocation, and it wastes the lane's planning rather than the coordinator's.**

### Footnote: a prefix that is truncated BY CONSTRUCTION

*“13 clippy sites”* was not a count — **it was where the compiler gave up.** CI dies at the first failing crate, which stops its dependents, so the rest **cannot** be reported. **Distinct from the other population failures recorded here** (a wrong complement, a two-mechanism grep, an op set minus a family): **those are bad queries and can be fixed by asking better. This one is truncated by the instrument's own semantics and no phrasing repairs it** — only running past the failure does.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A STALE JUSTIFICATION NEVER GETS RE-RUN.** A stale NUMBER is caught because someone re-runs the query; **restating a REASON costs nothing, produces no artifact, and is indistinguishable from having just checked it.** That is why stale holds, prohibitions and blockers outlive stale measurements. **And separate the two: an EXPIRED reason WAS true (wants a re-check schedule); a reason that NEVER APPLIED never was (wants deriving at the moment it is stated).** Worked example: a gate held because updating a toolchain would kill a warm forge — when the required toolchain was already installed under its own name, measured personally hours earlier. **When you state a reason for NOT doing something, derive it now or say you are quoting it.** ⚠️ **And ANNOUNCE CLOSURES, not only allocations** — twice in one night a lane proposed work already closed and unmentioned. → [`a-restated-reason-is-never-re-derived`](#a-restated-reason-is-never-re-derived)

## where-a-process-starts-is-not-where-it-acts

> **Index line (in CLAUDE.md):** **`from_cwd`, a peer summary, a branch name in a handoff — all describe where a process STARTED or what was true at a MOMENT, never where it acted.** Inferring a peer's staleness from launch metadata is an absence claim off the wrong instrument, and it is the **coordinator's** version of that error, with the cost landing on someone else. **Send the discriminating probe; do NOT attach your conclusion to it.**

**2026-08-21, the architect's own instance, twice in one night against the same lane.** Both times the inference was *"this lane measured in the shared checkout, which is stale."* Both times it was **wrong**, and both times the lane refuted it with content.

The first is GAP-223 (a parse failure attributed to the shared tree; their own worktree was pre-rebase — a **different** stale path, so the correction *widened* the class). The second: a seven-crate coverage triage reported *"at head 237e4264"* from `from_cwd=C:\Projects\fuel`, and the shared checkout was measured sitting **46 commits behind** that sha, with **three of the seven triaged crates** having files inside the window. The suspicion was cheap to form and entirely unfounded — **the lane never runs cargo in the shared checkout**, a discipline they had already stated once.

**MECHANISM.** `from_cwd` is where a shell was *spawned*. It says nothing about where `cargo`, `git`, or an editor subsequently ran, and a disciplined peer's whole practice is to act somewhere else. Same family as *reachability is a `pub mod` chain, not a root re-export* and *a file's feature gate is not a guarantee's coverage* — **one mechanism checked, absence concluded** — but with a distinguishing feature that makes it worse:

> **The other instances are wrong about CODE. This one is wrong about a PERSON, and they pay the round-trip to disprove it.**

**THE ASYMMETRY THAT DECIDES THE PRACTICE.** A discriminating probe costs the recipient one command. The *conclusion attached to it* costs them a rebuttal, and — compounding — trains them to read incoming probes as accusations. **The probe was correct both times; the framing was the defect both times.** Separate them explicitly: *"run this, it distinguishes X from Y"* is a request. *"You're probably stale, run this"* is a verdict wearing a request's clothing, and the verdict is the part that was never checked.

**AND THE DISCRIMINATOR IS NOT "IS IT STALE".** Staleness is only a defect if it touches the measured population. The right question is **does the changed set intersect what was measured** — here, `git diff --name-only HEAD <sha>` against the crate list, which is one command and answers the question that actually matters. A tree can be 46 commits behind and the measurement still be exactly right.

**WHAT THE LANE DID BETTER, and it is a reusable upgrade.** Asked *"does file X exist in your tree?"* — a **presence** question — they answered with their run log showing `Running tests\<X> ... 4 passed`: the file was **compiled and executed inside the measurement**. Presence proves the tree; **execution proves the tree AND that the reported count includes it.** When asked to prove an anchor, prefer evidence that the thing PARTICIPATED over evidence that it was merely THERE.

**PRACTICE: verify a peer's anchor by content or execution, never from launch metadata; ask what the staleness would have to touch before deciding it matters; and send the probe naked — if you cannot state the check without the accusation, you have not designed the check.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ `from_cwd`, A PEER SUMMARY, A BRANCH NAME IN A HANDOFF — ALL SAY WHERE A PROCESS *STARTED*, NEVER WHERE IT *ACTED* (2026-08-21).** Twice in one night the architect inferred *“this lane measured in the stale shared checkout”* from a peer's `from_cwd`, and was **wrong both times** — the lane runs cargo in its own worktree, as a disciplined peer does. Same family as *reachability is a `pub mod` chain* and *a file's gate is not a guarantee's coverage* — **one mechanism checked, absence concluded** — but wrong about a **PERSON**, who then pays the round-trip to disprove it. **AND THE DISCRIMINATOR IS NOT “IS IT STALE” BUT “DOES THE CHANGED SET INTERSECT WHAT WAS MEASURED”** — one `git diff --name-only HEAD <sha>` against the crate list; **a tree can be 46 commits behind and the measurement still be exactly right.** **SEND THE PROBE NAKED: if you cannot state the check without the accusation attached, you have not designed the check.** And prefer evidence that the artifact **PARTICIPATED** (a run log showing it compiled and ran) over evidence it was merely **PRESENT**. → [`where-a-process-starts-is-not-where-it-acts`](#where-a-process-starts-is-not-where-it-acts)

## enumerate-the-divergence-input

> **Index line (in CLAUDE.md):** **To decide whether a test suite catches a proposed rewrite, do not run the suite — ENUMERATE THE CALLERS FOR THE INPUT ON WHICH THE OLD AND NEW FORMS DIVERGE.** If no caller supplies it, the suite **provably cannot** catch the rewrite, and reading the callers told you more than running 1,471 tests would have. **Suite SIZE is not coverage of the FIX SURFACE.**

**2026-08-21, GAP-229.** A per-crate coverage triage established that `fuel-core` runs **1,471 tests** under default features on every platform — a strong gate by any aggregate measure, and the basis for a proposed *"safe to auto-fix here"* verdict across ~1,460 clippy findings.

**The aggregate was true and licensed nothing.** Worked example, in production training code:

```
clippy::neg_cmp_op_on_partial_ord  wants   !(max_norm > 0.0)  ->  max_norm <= 0.0
```

**The two forms are identical on `0.0`, on every negative, and on every finite value. They diverge on `NaN` alone** — `!(NaN > 0.0)` is `true` and **rejects**; `NaN <= 0.0` is `false` and **admits**, after which `max_norm / total_norm` silently scales **every gradient** to `NaN`.

So the question *"does the suite catch this?"* has a **closed-form** answer requiring no build: enumerate the callers, and ask whether any supplies `NaN`. The complete caller set was **two tests** (`5.0`/`2.0` and `6.5`/`2.0`), **one production site** forwarding an unvalidated builder field, and a builder called **from doc comments only**. **None supplies `NaN`. Therefore the divergence is unreachable from the test set and the suite CANNOT catch the rewrite** — not *probably does not*.

**Confirmed by sabotage afterwards, which is the weaker instrument and was run only as a check on the reasoning:** guards flipped to `<= 0.0` gave `3 passed; 2 failed; 1389 filtered out`, and **both pre-existing tests went GREEN with the guard sabotaged.**

**WHY THE ENUMERATION BEATS THE SABOTAGE HERE.** A sabotage run answers *"did these tests catch THIS mutation"* and costs a build. The enumeration answers *"can any test in the population reach the divergence at all"*, costs a grep, and returns **provably** rather than **empirically** — a green sabotage leaves *"maybe I mutated the wrong line"* open, while an empty caller set does not. Sabotage remains correct when the divergence input is hard to characterise; **when the two forms differ on a nameable set of inputs, enumerate instead.**

**AND IT GENERALISES PAST LINTS.** Any behaviour-preserving-looking edit — a guard rewrite, a widened `Option` domain, a swapped comparison, a default changed — has a **divergence set**. Name it, then ask which callers can produce a member of it. **A suite of any size is blind to a divergence its inputs cannot reach**, and the size is what makes people stop asking.

**PRACTICE: state the divergence set explicitly (*"these differ only on NaN"*), enumerate the callers for it, and report the caller set — not the suite total — as the coverage claim.** *"1,471 tests"* is a fact about the crate; *"zero callers supply the divergence input"* is a fact about the change.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ TO DECIDE WHETHER A SUITE CATCHES A REWRITE, DO NOT RUN THE SUITE — ENUMERATE THE CALLERS FOR THE INPUT ON WHICH THE TWO FORMS DIVERGE (2026-08-21, GAP-229).** `!(x > 0.0)` and `x <= 0.0` are identical on `0.0`, on negatives, and on every finite value; **they diverge on `NaN` alone.** `clip_grad_norm`'s complete caller set supplies no `NaN`, so a **1,471-test** `fuel-core` suite **provably cannot** catch clippy's `neg_cmp_op_on_partial_ord` rewrite — confirmed after the fact by sabotage (`3 passed; 2 failed`, **both pre-existing tests GREEN under the sabotage**). **SUITE SIZE IS NOT COVERAGE OF THE FIX SURFACE, and the size is what makes people stop asking.** The enumeration is also the STRONGER instrument: a green sabotage leaves *“maybe I mutated the wrong line”* open; an empty caller set does not. → [`enumerate-the-divergence-input`](#enumerate-the-divergence-input)

## line-numbers-rot-and-nothing-else-does

> **Index line (in CLAUDE.md):** In a doc-citation sweep of CLAUDE.md, **file paths and GAP ids were 0% defective and line numbers were 67% defective.** A citation that already names its target does not need the number — and the number is the only half that rots. **Strip it; keep the name.**

**MEASURED 2026-08-21 across CLAUDE.md's bullets (L70 excluded, in flight):**

```
file-path citations   19 checked   0 wrong    (0%)
GAP-id citations      25 checked   0 wrong    (0%)   control: 217 ids known
line-anchored         9 in-repo    6 wrong   (67%)   3 more cross-repo, unverifiable here
```

**The decay is entirely in line numbers.** Paths survive because a file move is a
loud event someone notices; ids survive because a registry row is never
renumbered. **A line number rots on any insertion above it, silently, and nothing
in the artifact shows it.**

**Drift observed:** off-by-1 x4, off-by-8 x1, **off-by-26 x1**. Severity tracks
distance: off-by-one lands the reader adjacent and self-corrects; **off-by-8 landed
in a function signature where a reader would conclude the cited panic had been
FIXED**, which is the harmful direction; off-by-26 lands in unrelated code.

**⚠️ THE CONTROLLED COMPARISON IS THE STRONGEST PART, AND IT IS ONE BULLET AGAINST
ITSELF.** The `never panic` bullet carries BOTH a line number (`lib.rs:~3931`) and
a prose anchor (`git grep 'does not match shape element count'`), added **on the
same day by the same person**. One day later the **line number had moved to 3939**
and **the prose anchor still resolved both sites**. Same claim, same file, same
author, same hour — the identifier-free anchor survived and the number did not.

**PRACTICE: do not update a drifted line number — DELETE it.** Every one of the six
here already named its target (`pub mod telemetry;`, the `cfg_attr` derive, the
quoted sentence), so the number was **redundant AND the only rot-prone half**.
Updating it resets a clock that will run out again; removing it ends the class.
Keep a line number only where nothing else identifies the target, and expect to
re-verify it.

**BOUND, because the obvious inference is wrong:** this does **not** show stamped
claims (`(checked)`, `measured`) drift more. The drifted six span stamped and
unstamped bullets alike, and at n=9 the split does not separate. **The
stamped-is-higher-risk inversion is right on mechanism** (a verification stamp
suppresses re-derivation — CLAUDE.md's `no rust-toolchain.toml (checked)` went
false in a day) **but this sweep does not measure it as a rate, and should not be
cited as if it did.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- In a doc-citation sweep of CLAUDE.md, **file paths and GAP ids were 0% defective and line numbers were 67% defective.** A citation that already names its target does not need the number — and the number is the only half that rots. **Strip it; keep the name.** → [`line-numbers-rot-and-nothing-else-does`](#line-numbers-rot-and-nothing-else-does)

## a-count-has-no-rename-resistant-form-so-bound-it

> **Index line (in CLAUDE.md):** Paths and symbols can be made rot-proof by naming something stable. **A COUNT CANNOT — it is inherently a claim about a moment.** Its analogue is a **BOUND** (`>= 20`), a **date+ref**, or **the property the number is doing work for** (usually just "non-zero"). Write the point value only when the exact figure IS the claim, and date it.

**MEASURED ON MY OWN TEXT, FOUR DAYS APART (2026-08-21 -> 2026-08-25), which is
the shortest interval any claim in this file has been tested over:**

```
form                                     written   now    survived
>= 20  fkc .rs files          (HEDGED)      >=20     35      YES
>= 10  telemetry .rs files    (HEDGED)      >=10     10      YES
190    fuel-core/src/*.rs     (CONTROL)      190    190      YES
210    gaps.md live rows      (EXACT)        210    217      NO
175    rows carrying an owner (EXACT)        175    180      NO
```

**Both hedged claims survived; both exact claims failed. 2/2 against 0/2.**

**This completes the citation taxonomy and the fix differs by KIND, not degree:**

- **paths / symbols / GAP ids** — 0% defective; rot-proof by *naming something
  stable*.
- **line numbers** — 67% defective; fix by *deleting the number and keeping the
  name*, because the name was already there.
- **counts** — no stable form exists. **You cannot name your way out of a count.**
  Fix by *weakening the claim to what the number is actually doing*.

**PRACTICE — ask what work the number is doing, then write the weakest form that
does it:**

- proving a query is not broken -> **"non-zero"** is the whole claim; the value is
  decoration. (My structural controls said `-> 190`; what they NEEDED was
  `-> non-zero`.)
- establishing a magnitude ("the registry is substantial") -> **a floor**.
- the exact figure IS the finding ("6 of 9 drifted") -> **exact, and dated with a
  ref**, because it is a measurement rather than a fact.

**AND THE FAILURE IS SILENT IN THE FLATTERING DIRECTION.** A rotted count still
reads as precise, still carries whatever authority precision confers, and its
wrongness is invisible without re-running the measurement. **210 and 175 were
written as evidence that a registry mechanism was real — a claim that survives
being off by seven — but they were written as point values, so a reader checking
them finds a discrepancy and doubts the whole clause.** Over-precision does not
merely decay; it converts a durable argument into a fragile one.


---
## a-defence-can-outlive-its-defect

> **Index line (in CLAUDE.md):** **A REMEMBERED CONTROL THAT HAS BEEN REPLACED BY A STRUCTURAL ONE DOES NOT GO NEUTRAL — IT COMPETES WITH ITS REPLACEMENT.** Worse than ordinary staleness: it does not merely stop being useful, **it acquires the OPPOSITE effect at a future date nobody is watching for, and it looks like compliance while doing it.** When a discipline becomes structural, RETIRE the remembered version explicitly and state the mechanism.

**2026-08-25, the architect's own standing order.** *"Use `+1.98.0`, never `+stable`"* was **correct and load-bearing** when written: nothing pinned the toolchain, this box's ambient default was **1.99.0-nightly**, `+stable` was **1.97.1**, and CI resolved `stable` **fresh per run**. Three compilers, and the order was the only thing between a lane and a wrong-compiler measurement. It worked.

Then `rust-toolchain.toml` landed, and the order **inverted**, on a fact nobody had needed before:

```
rustup precedence:  1  explicit +toolchain on the command line   <- WINS
                    3  directory override
                    4  rust-toolchain.toml
                    5  rustup default
```

**An explicit `+toolchain` BEATS the pin.** So from the day the pin is bumped, every lane still obeying the order stays silently on the old compiler **while CI moves** — local greens diverging from CI, **the exact condition the order existed to eliminate.** Each such lane is *following the documented rule*, so the divergence presents as diligence.

**WHY THIS IS ITS OWN CLASS.** Its neighbours describe rules that go **inert**: `staleness-by-workaround` stays TRUE and stops being CHEAPEST; a plain stale fact stops being true and misleads once. **This one stays grammatical, stays obeyed, and REVERSES SIGN.** Its harm is *created by compliance*, so the usual detector — noticing the rule is wrong — never fires, because the rule is not wrong, it is inverted.

**AND NOTE WHAT DID NOT CATCH IT.** A full doc-currency sweep had passed **35/35 session-prompts and 7/7 specs** and was **CORRECT when it ran**. The line expired *afterwards*, at the pin commit. **A complete currency audit does not immunise a corpus against a line that expires after the audit.**

**PRACTICE: when a discipline moves from REMEMBERED to STRUCTURAL, the retirement is part of landing the structure.** Name the old rule, say it is retired, state the mechanism that replaced it. **A rule stated without its reason gets re-broken by the next person with a good reason to type the forbidden thing** — and this one reads as harmless to them.

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A DEFENCE CAN OUTLIVE ITS DEFECT AND THEN BECOME ONE - a remembered control replaced by a STRUCTURAL one does not go neutral, IT COMPETES WITH ITS REPLACEMENT (2026-08-25).** *Use +1.98.0, never +stable* was correct while nothing pinned and the box default was nightly. Post-pin **an explicit +toolchain BEATS rust-toolchain.toml (rustup precedence 1 over 4)**, so the day the pin bumps every obedient lane stays silently on the old compiler while CI moves - **the exact condition the order existed to eliminate, arriving THROUGH compliance with it.** Worse than ordinary staleness: the rule stays grammatical and REVERSES SIGN, so *is this still true?* never fires. **And a full doc-currency sweep (35/35, 7/7) had passed and was CORRECT - the line expired AFTER the audit.** **When a discipline becomes structural, retiring the remembered version IS part of landing the structure, and the retirement must state the MECHANISM.** -> [`a-defence-can-outlive-its-defect`](#a-defence-can-outlive-its-defect)

## cite-what-cannot-move

> **Index line (in CLAUDE.md):** **THREE CITATION FORMS, THREE DECAY RATES, AND THREE DIFFERENT FIXES — measured: paths/symbols/GAP ids 0% defective, LINE NUMBERS 67%, COUNTS no stable form at all.** Fix a path by NAMING something stable; fix a line number by DELETING it and keeping the name; **you cannot name your way out of a COUNT — its only durable forms are a BOUND, a date+ref, or the PROPERTY the number was doing work for.**

**2026-08-25, measured by a lane sweeping the ~25 normative bullets' embedded citations, then extended to its own text.**

```
paths / symbols / GAP ids   0% defective   -> rot-proof by NAMING something stable
line numbers               67% defective   -> fix by DELETING the number, keep the name
counts                     no stable form  -> fix by WEAKENING the claim
```

**The line-number mechanism is obvious once separated and invisible while aggregated:** a path survives because editing a file does not rename it; **a line number rots on every insertion ABOVE it**, which is most commits — it decays on a schedule set by unrelated work.

**THE COUNT HALF IS NOT RESTATED HERE** — it is measured and owned by [`a-count-has-no-rename-resistant-form-so-bound-it`](#a-count-has-no-rename-resistant-form-so-bound-it), which carries the controlled comparison (hedged **2/2** survived, exact **0/2**, four days apart), the three-case practice for choosing a count's weakest sufficient form, and the corollary that **a positive control asserts a PREDICATE, not a VALUE.** Deliberately a POINTER AND NOT A SUMMARY: **this file's own rule is that a second copy is a divergence generator**, and that finding is the lane's measurement rather than mine. The one line that belongs here is the taxonomy it completes — **you cannot NAME your way out of a count, so its fix differs in KIND from the two forms above, not in degree.**

**INDEPENDENTLY CONFIRMED on a bullet nobody had flagged.** The `never panic on production paths` rule cited its own violation and was wrong in **three particulars at once** — type renamed, line moved, and the construct was an `assert_eq!` not the `.expect()` named. **The rule was eternally true; only every detail of how to FIND it was wrong.** Re-anchoring on a **prose string from the panic message** — unreachable by any rename — fixed it, **and immediately found a SECOND violation site the bullet had never named**, against a positive control (437 `assert_eq!` in that file) proving the query could find things where they exist.

**WHY THE ASYMMETRY MATTERS MORE THAN THE RATE.** A wrong path fails loudly — no such file. **A wrong line number silently points at REAL CODE that is not the code meant**, and a reader finding something plausible concludes the citation is fine. **The failure mode of the 67% is not a broken link, it is confident mis-reading.**

**PRACTICE: cite a symbol, a distinctive string, or a test name, and give the grep; give a line number only as a convenience beside a durable anchor, never as the anchor. For a count, write the BOUND you actually rely on, or the property, or a date+ref — and treat a normative rule's citation as the likeliest to have rotted, precisely because the principle above it is unimpeachable and stops anyone checking.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ THREE CITATION FORMS, THREE DECAY RATES, THREE DIFFERENT FIXES (2026-08-25, measured).** paths/symbols/GAP ids **0% defective**; LINE NUMBERS **67%**; COUNTS **no stable form at all**. Fix a path by NAMING something stable; fix a line number by DELETING it and keeping the name; **you cannot NAME your way out of a COUNT - its durable forms are a BOUND, a date+ref, or the PROPERTY the number was doing work for.** Controlled on one author over four days: **hedged claims 2/2 survived, exact counts 0/2**. **And OVER-PRECISION IS WORSE THAN IMPRECISION: a count written as a point converts a durable argument into a fragile one, and it rots in the FLATTERING direction because it still reads precise.** Corollary reaching every control here: **a positive control asserts a PREDICATE (non-zero), never a VALUE (190).** Asymmetry that decides the practice: a wrong path fails LOUDLY, **a wrong line number silently points at REAL CODE that is not the code meant** - confident mis-reading, not a broken link. -> [`cite-what-cannot-move`](#cite-what-cannot-move) · **counts specifically:** [`a-count-has-no-rename-resistant-form-so-bound-it`](#a-count-has-no-rename-resistant-form-so-bound-it)

## an-anchor-buys-immunity-and-pays-in-blindness

> **Index line (in CLAUDE.md):** **A RENAME-RESISTANT PROSE ANCHOR IS BLIND TO EVERY PANIC AT THE SITE THAT DOES NOT CARRY ITS STRING — the property that makes it rot-proof is the property that makes it narrow.** Measured: the never-panic rule's anchor finds **1 of 3** panic constructs at each of its two sites. **A citation states a LOCATION; a rule needs a POPULATION, and the two are different queries.**

**2026-09-06, found by a lane claiming GAP-003 — the one violation in this project cited by an anchor chosen specifically for durability.**

[`cite-what-cannot-move`](#cite-what-cannot-move) established that a normative rule's citation should anchor on a distinctive prose string, because no identifier rename can reach one. That is correct and it is not free.

```
anchor 'does not match shape element count'   -> 2 hits: fuel-graph/src/lib.rs :3939, :4147
panic-capable constructs within 12 lines of EACH hit:
    assert_eq!(...)          <- the anchor SEES this one
    .expect("NodeHandle::...")   <- BLIND, ~5-8 lines below
    .write().unwrap()            <- BLIND, and a DIFFERENT CLASS

CONTROL: both sites test TRUE for all three buckets, so the blindness is not
         a property of one unlucky site.
```

**The bullet carrying that anchor said "TWO hits, and both are the violation."** ⚠️ **True about the SITES and silent about the SIBLINGS — and phrased so that it reads as completeness.** A later reader re-deriving the violation from the prescribed grep gets two lines and a clean feeling.

⚠️ **THE MECHANISM, WHICH IS THE PART THAT TRANSFERS: A PROSE ANCHOR IS EXACTLY AS NARROW AS IT IS DURABLE.** A symbol anchor (`from_host_buffer_on`) selects a *scope* — the whole function, every panic in it — and rots on the next rename. A message anchor selects a *line* — the one construct that happens to carry that string — and survives every rename forever. **The trade is not a defect in either form; what is a defect is presenting the narrow one as if it enumerated.**

**And the two failure directions are unequal.** A rotted symbol anchor fails **loudly**: the grep returns nothing and someone re-derives. **A durable message anchor fails SILENTLY and FLATTERINGLY** — it returns real hits, at the right file, at real violations, and simply omits the neighbours. **Nothing in the output says it is a sample.**

**THE GENERAL SHAPE: a citation answers "where is it?" and a normative rule needs "what is all of it?".** Those are different queries and the same string cannot serve both. The same confusion appears elsewhere in this file as the `enumerate-the-population-not-the-strings` memory note — **here it arrives through a citation-hygiene rule doing its job correctly.**

**PRACTICE:**

- **Anchor on prose for DURABILITY; enumerate by CONSTRUCT for COMPLETENESS. Give both, and say which is which** — *"grep `<string>` to find it; the population is every `assert!`/`.expect`/`panic!` in `<fn>`"*.
- ⚠️ **When a rule's citation returns N hits, state whether N is a COUNT or a LOWER BOUND.** The never-panic bullet's "two hits" was a lower bound written as a count, which is the [count-has-no-rename-resistant-form](#a-count-has-no-rename-resistant-form-so-bound-it) rule arriving at a citation.
- ⚠️ **Classify the neighbours before sweeping them in.** At these sites the third construct is a poisoned-lock `.unwrap()` — **198 of them in that one file** — and a poisoned lock means another thread has *already* panicked, so converting it to `Result` propagates an unrecoverable condition and buys the caller nothing. **It is a different obligation with a different remedy and belongs in its own row.** Proximity is not membership: **the anchor's blindness and the sweep's over-reach are the same error with opposite signs.**

---
## state-the-link-not-the-proxy

> **Index line (in CLAUDE.md):** **A CONDITIONAL LICENCE MUST STATE THE CAUSAL LINK, NOT THE PROXY** — *"if X, proceed"* is only as good as the unstated connection between X and the thing being licensed. **State the link and the grantee can check it; state only the proxy and the licence is UNFALSIFIABLE — the grantee cannot test it and the grantor never learns it was wrong.** The failure is silent in the direction where the proxy HOLDS.

**2026-08-26, the architect's own delegation, caught by the lane it was handed to.** The licence: *"if the four small precision surfaces turn out to be as cheap as conv, take attention immediately and do not wait for me."*

**COST was the proxy. The actual variable was WHETHER A KNOWN-GOOD SHAPE EXISTED TO MIRROR.** The small surfaces were cheap *because* all five families already carried a fused arm in the probe builder that could be mirrored into the primitive builder. **Attention has none** — measured with a positive control so the zero is absence rather than a broken query: **10** references for the families that had arms, **0** for FlashAttn/PagedAttn.

**So the proxy held perfectly and the mechanism under it did not transfer at all.** The lane reported instead of spending the licence.

**WHY THIS IS WORSE THAN AN ORDINARY WRONG INSTRUCTION.** A wrong instruction can be checked against its own terms. **A licence conditioned on a proxy cannot** — the grantee can verify the proxy (cost was low, truthfully) and still be authorised to do the wrong thing, because the sentence never mentioned the property that actually decides it. **Nothing in the licence is available to be contradicted.**

**AND THE ORDERING HAZARD, WHICH IS THE LANE'S OWN CORRECTION AND THE SHARPEST PART:** they only examined the causal link *after* the small surfaces came in cheap. **Had those surfaces come in EXPENSIVE, they would have reported cost, the licence would never have been exercised, and nobody would ever have discovered it was unfalsifiable as written.** The defect is only observable on the branch where the proxy is satisfied — so a licence like this can sit unspent and undetected indefinitely, and be reissued.

**PRACTICE:**
- **Write the link, not the trigger.** *"Take attention if a mirrorable arm exists for it"* is checkable. *"Take attention if the small ones were cheap"* is not.
- **When you cannot name the link, say the licence is provisional and ask for a report** rather than pre-authorising.
- **As grantee: check the link BEFORE the trigger fires**, not after. Verifying a proxy is not verifying a licence.
- **Corollary for verdicts: when a yes/no question is malformed, return the STRUCTURE rather than the answer.** Asked *"do the backwards mirror FlashAttn?"*, "no" would have bought a fresh-surface estimate for work that reuses the params struct outright, and "yes" a copy-paste estimate for work that adds an operand and a per-variant output shape. **The delta was the thing being asked for; neither branch of the binary carried it.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A CONDITIONAL LICENCE MUST STATE THE CAUSAL LINK, NOT THE PROXY (2026-08-26, the architect delegating).** *Take attention if the small surfaces were as cheap as conv* used COST as the trigger; the real variable was **whether a mirrorable arm existed** (measured 10 vs 0, positive-controlled). **The proxy held and the mechanism did not transfer.** Worse than an ordinary wrong instruction: the grantee can VERIFY the proxy and still be authorised to do the wrong thing, because **the sentence never names the property that decides it — so nothing in the licence is available to be contradicted.** And the ordering hazard makes it self-concealing: the link is only examined on the branch where the trigger FIRES, so a licence like this can sit unspent and undetected indefinitely. **Write the link, not the trigger; as grantee, check the link BEFORE the trigger.** Corollary: **when a yes/no question is malformed, return the STRUCTURE, not the answer** — asked whether three backward families mirrored a template, both branches of the binary would have bought the wrong estimate; the DELTA was what was actually being asked for. -> [`state-the-link-not-the-proxy`](#state-the-link-not-the-proxy)

## a-partial-exclusion-list-implies-coverage

> **Index line (in CLAUDE.md):** **AN ENUMERATION THAT NAMES THREE EXCLUSIONS AND OMITS A FOURTH DOES NOT READ AS SILENT ON THE FOURTH — IT READS AS COVERING IT.** A scope statement is not merely incomplete when it under-enumerates; **it makes a positive claim it was never intended to make.** An exclusion list must be exhaustive, or must say that it is not.

**2026-08-26, found by the precision lane while confirming a scope clause the architect had challenged for being in the wrong PLACE.** The real defect was worse than placement.

The emitted evidence clause read:

> *"...not evidence about other inputs, other machines, or other compilers."*

`softcap`, `window_size_*` and `causal` are **parameters, not operands.** So *"other inputs"* never reached them — the clause was silent on parameterisation. **And silence, inside a three-item enumeration, is not neutral.** A reader who sees inputs / machines / compilers carved out concludes the author enumerated the axes of variation, and that **anything unnamed is inside the attestation.** The clause did not merely fail to exclude parameter configurations; **it implied they were covered.**

Fixed to: *"...other inputs, OTHER PARAMETER CONFIGURATIONS (one probe fixes one; branches such as causal/softcap/window go untaken), other machines, or other compilers."*

**WHY THE GENERIC FIX WAS RIGHT AND THE SPECIFIC ONE WOULD HAVE BEEN A TRAP.** The lane made the new limb **generic rather than attention-specific**, because *every* probe fixes one parameterisation — the hole was **in the clause, not in FlashAttn's row.** An attention-specific sentence would have closed one family and left the identical implicature standing in the other nineteen, while *looking* like the problem was solved. **Fix the generator, not the instance** — and here the "generator" is a sentence emitted 148 times.

**THE RE-EMISSION IS ITSELF THE CONTROL.** 11 files, **148 insertions / 148 deletions** — a pure swap, verified as one: **148** clauses carry the new limb, **0** carry the old, and attention's own count is **unchanged at 4**. That last check is the misattachment defect from GAP-228(a) pointed backwards: a count that MOVED would have meant the clause had been re-attached to the wrong sections. **A pure-swap edit has a pre-declarable shape, and checking the shape is what distinguishes it from a rewrite.**

**AND A MECHANISM BUILT FOR ONE TRIGGER CAUGHT A DIFFERENT ONE.** Clause eligibility had been re-keyed on `has_clause` rather than *"this run flipped it"*, justified at the time as *"useless for exactly the case it will be needed in — the evidence changing under a toolchain pin or a re-seed."* **It changed under a CORRECTION instead, and the mechanism did not care which.** Worth keeping as evidence for building the general re-emission path rather than the predicted one: **the trigger you name when justifying a mechanism is rarely the trigger that fires it.**

**Scope of that correction, stated plainly because it is the kind of thing that gets over-read: the RECORDS were unaffected.** Every one was earned by 16 byte-identical repeat invocations and still is. **What changed is what the contract SAYS they cover** — no re-seed, no re-verification, no downgrade.

**PRACTICE: when writing a scope limitation, enumerate the AXES OF VARIATION, not the ones you happen to have thought of — and if the list is not exhaustive, say so in the list.** Related but distinct from the **"a true justification attached to a wider claim than it supports"** rule in `CLAUDE.md` (GAP-166) — *cited to the file that actually carries it: there is no `justification-scope-mismatch` section here, and a link to one would have been a dangling citation inside the rule about citations*: there a true reason silently licenses a wider claim; **here an explicit list manufactures the wider claim by omission.** The first is an overreach nobody wrote down; the second is written down, in the very sentence intended to constrain it.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ AN ENUMERATION THAT NAMES THREE EXCLUSIONS AND OMITS A FOURTH DOES NOT READ AS SILENT ON THE FOURTH — IT READS AS COVERING IT (2026-08-26).** An emitted evidence clause said *not evidence about other inputs, other machines, or other compilers*; `softcap`/`window_size`/`causal` are **params, not operands**, so *other inputs* never reached them. **The clause did not merely fail to exclude parameter configurations — by enumerating three axes it IMPLIED the unnamed fourth was inside the attestation.** **An exclusion list must be exhaustive or say that it is not.** Fixed GENERICALLY, not per-family: every probe fixes one parameterisation, so the hole was in the CLAUSE, not in one row — an attention-specific sentence would have closed one family and left the identical implicature in nineteen others while looking solved. **Verified as a PURE SWAP with a pre-declarable shape: 148 in / 148 out, 148 clauses carrying the new limb, 0 the old, and the per-family count UNCHANGED** — a count that moved would have meant re-attachment to the wrong sections. Distinct from *a true justification attached to a wider claim*: there the overreach is unwritten; **here the constraining sentence itself manufactures it by omission.** -> [`a-partial-exclusion-list-implies-coverage`](#a-partial-exclusion-list-implies-coverage)

## commands-dont-rot-by-breaking

> **Index line (in CLAUDE.md):** Third and last citation population. **Cited commands were 0% dead (10/10 alive).** They do not rot by breaking — breakage is loud and gets fixed. They rot by **SCOPE**: still running, still exiting 0, no longer answering the question they were cited for.

**MEASURED 2026-08-25, completing the taxonomy:**

```
paths / symbols / GAP ids     0% defective  (19, 25)   rot-proof by NAMING something stable
line numbers                 67% defective  (9)        fix by DELETING the number
counts                    exact 0/2 survived, hedged 2/2  no stable form: BOUND it
commands                      0% dead       (10/10)    fix is not about liveness at all
```

**47 distinct commands cited; 13 are templates with placeholders and cannot be run
as cited; 34 runnable, of which 7 are expensive (workspace / CUDA / forge). Every
one of the 10 cheap read-only ones ran.**

**So the fix for commands is not "check it still runs" — that check passes.** The
failure mode is a command that succeeds and misleads, and CLAUDE.md already
documents three of them: `cargo check --workspace` **is a CUDA forge**;
`git log HEAD --not --remotes` **reports false unpushed work after a branch
reap**; a warm cache **suppresses the `Checking` line and the warnings both**.
Each runs perfectly. **PRACTICE: cite what the command must SHOW, not just the
command** — the expected output, or the property being established.

**⚠️ AND THE INSTRUMENT WAS WRONG MORE OFTEN THAN THE CORPUS — FOUR TIMES IN ONE
SWEEP, EVERY TIME TOWARD A MORE INTERESTING NUMBER:**

1. **4 "missing" paths** — 3 were shorthand, 1 a legitimate cross-repo citation.
   Would have reported a **21% path-defect rate that was entirely mine**.
2. **A phrase grep missed a hedge that WRAPPED ACROSS TWO LINES**, reading as
   *"the cited sentence is gone"* when it was merely folded.
3. **4 commands classified DEAD on `rc=124`** — which is the *timeout firing*, not
   a failure. They are cargo invocations that pull the kernel forge. **I had
   written the rule "classify by what a command DOES, not what it looks like" two
   days earlier and then built a filter that classified by what it looks like.**
4. **A `grep` with no file argument inside a `while read` loop CONSUMED THE LOOP'S
   STDIN**, silently eating 3 of 10 inputs — the run reported 7 and I only noticed
   because `wc -l` said 10. Fix: `< /dev/null` on the inner command.

**Every one of the four inflated the finding.** An instrument error that makes the
corpus look worse is the one least likely to be questioned, because it agrees with
why the sweep was commissioned. **Validate the instrument before reporting through
it, and reconcile any count that disagrees with itself.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- Third and last citation population. **Cited commands were 0% dead (10/10 alive).** They do not rot by breaking — breakage is loud and gets fixed. They rot by **SCOPE**: still running, still exiting 0, no longer answering the question they were cited for. → [`commands-dont-rot-by-breaking`](#commands-dont-rot-by-breaking)

## measure-the-gate-before-building-it

> **Index line (in CLAUDE.md):** Asked whether the citation findings could be made into a gate, the answer is **split, and only measurement separates the halves**: a **counts** gate is INFEASIBLE (~90% false positives, 157 flags), a **line-number** gate is FEASIBLE (6 flags, syntactically unambiguous), and **commands** need no gate because their rot is semantic.

**MEASURED 2026-08-26 BEFORE BUILDING ANYTHING, on the architect's instruction
after three unmeasured mechanisms were specified and caught the same night.**

**COUNTS GATE — INFEASIBLE, and the reason is structural rather than fixable.**
A rule of the form *"an exact count in normative text must be dated or a bound"*
raises **157 flags** in CLAUDE.md. A 16-item random sample contains: three dates
(`2026`), two exit codes (`101`), a `_MSC_VER` value (`1959`), a CUTLASS warning
code (`#177-D`), a GPU model number (`4070`), a Visual Studio version (`18`), a
model dimension (`130`), a filename (`10-decisions-log.md`), a line number
already covered by the other rule (`3934`), and **at most two** actual rot-prone
counts. **~90% false positive.**

**And it cannot be fixed by a better regex**, which is the load-bearing part: the
discriminating property is *"is this a count of a CURRENT repo state"*, and the
token `13` is a lint count, an exit code, and a version number depending on a
noun the pattern cannot resolve. **An exact number is sometimes exactly right —
`6 of 9 drifted` IS a finding — and nothing syntactic separates that from a
rotting point-value.**

**LINE-NUMBER GATE — FEASIBLE.** `path.ext:NNN` is syntactically distinctive and
raises **6 flags total**, every one genuinely a line-anchored citation: 3 verified
correct, 3 cross-repo. **An allowlist of six entries each carrying a reason is
tractable; an allowlist of 157 is a shredder** — it degrades to noise and takes
the guard's signal with it, which is GAP-141's prose-hedge guard exactly.

**COMMANDS — NO GATE POSSIBLE OR NEEDED.** 0% dead; they rot by SCOPE (still
runs, still exits 0, no longer answers the cited question). **A syntactic gate
cannot see semantics, and a gate that only checks liveness would pass every real
instance.**

**THE GENERAL SHAPE: a finding being true does not make it gateable, and
feasibility varies by POPULATION rather than by how good the finding was.** The
counts result is the strongest of the three (a controlled 2/2-vs-0/2 on the same
author's text in four days) and it is the one that cannot be enforced. **Measure
the false-positive rate against a real corpus before proposing the mechanism —
"no gate is possible here, and here is why" is a complete result, and cheaper
than a guard that reds on honest numbers and trains reflexive allowlisting.**


---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- Asked whether the citation findings could be made into a gate, the answer is **split, and only measurement separates the halves**: a **counts** gate is INFEASIBLE (~90% false positives, 157 flags), a **line-number** gate is FEASIBLE (6 flags, syntactically unambiguous), and **commands** need no gate because their rot is semantic. → [`measure-the-gate-before-building-it`](#measure-the-gate-before-building-it)

## a-rule-bound-to-an-instrument-does-not-transfer

> **Index line (in CLAUDE.md):** **A DISCIPLINE STATED ABOUT AN INSTRUMENT DOES NOT GENERALISE TO OTHER INSTRUMENTS WITH THE SAME FAILURE MODE — even for the person who wrote it, on the same day, having applied it correctly a dozen times.** *"Never pipe the GATE through `head`"* is obeyed faithfully while the identical truncation is committed on a comment thread. **State rules in terms of the PROPERTY that fails (never act on a truncated document) rather than the TOOL you first met it on.**

**2026-08-26. Three instances, two people, one night.**

**(1) THE WORKED EXAMPLE.** The portfolio PM's standing instructions carry, verbatim: *"Never pipe the gate's output through `head`/`tail` — truncating it is how a merge got hidden from me once already."* They applied it to the merge gate **all night, faithfully**. Then, surveying a PR thread, they ran `.body[0:700]` across a **3,474-character** architect ruling and dispatched off the fragment. **The disposition — *"Recorded as: Baracuda cosigned; Fuel non-responsive … cosign requirement DISCHARGED"* — sat in the 79% they discarded**, and a five-day-old "blocker" was dispatched that had been closed six hours earlier. **They did not read a stale document; they read a live document badly.**

**(2) THE SAME PERSON, TWO HOURS EARLIER, SAME DEFECT.** A `grep -c` returned `1` for a retracted phrase and was read as *live* — when the `1` was **the retraction quoting the sentence it deleted.** A count standing in for the document. **The lesson did not survive two hours, because it had been filed against `head`/`tail` rather than against *substituting a fragment for the whole*.**

**(3) A DIFFERENT PERSON, DIFFERENT SURFACE.** A lane wrote *"classify by what a command DOES, not what it looks like"* and **two days later built a filter that classified by what a command looks like** — misreading `rc=124` (a timeout) as a dead command. **Having the rule, in their own words, in a file they had just edited, did not prevent applying its inverse.**

**(4) 2026-09-02 — THREE IN ONE NIGHT, ONE PERSON, AND EVERY RULE WAS ONE THEY HAD WRITTEN THAT SAME NIGHT.** A lane wrote the heredoc-backslash hazard into a commit message, then an hour later lost a `\'` to a quoted heredoc while editing a decision table. They landed a gate requiring parent→child backlinks, having just measured the score at 0 of 10 — and filed the next row without one, inside the hour. They wrote *count UNIQUE `file:line:col`, not warning lines* into `scripts/check-gaps-table.py`, then read a cargo log by filename histogram and reported **48** accesses where there were **24**, because `--all-targets` emits every error twice. ⚠️ **All three rules were correctly stated about the DEFECT. All three were filed against the MEDIUM they were met in — a shell script, a registry convention, a markdown table — and none fired in the medium the violation happened in: a Python edit, a filing action, a compiler log.** **So the prescription above is necessary and NOT sufficient: this section is correctly defect-indexed and still did not fire, three times, for someone who could quote it.** **That is the argument for a GATE rather than a better sentence, and it is the same conclusion the mechanism below reaches by a different route.**

**THE MECHANISM.** A rule learned from an incident gets filed under **the tool the incident happened on**, because that is the salient detail while it stings. Recall is then keyed on the tool: pipe a gate → rule fires; slice a comment body → nothing fires, **even though the failure is identical and the person is the same person.** **The rule is not forgotten; it is not INDEXED under the situation.**

**WHY THIS IS THE STRONGEST AVAILABLE ARGUMENT FOR STRUCTURE OVER VIGILANCE**, and it is stronger than any specific defect: these three were committed by people **actively applying** the very rule they were violating, on the same day, with the text in front of them. **Vigilance did not fail through inattention — it failed through correct application to the wrong index.** A gate does not need the situation to remind it of itself.

**PRACTICE: when you write a rule from an incident, name the PROPERTY that failed and then ask which OTHER surfaces have it.** *"Never truncate a document you are about to act on"* covers gate output, comment bodies, `grep -c` results, PR descriptions, and summary lines. *"Never pipe the gate through `head`"* covers one. **And when a rule proves it does not transfer, that is the moment to make it structural — the failures above are exactly the ones a check would have caught and a memory did not.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A DISCIPLINE STATED ABOUT AN INSTRUMENT DOES NOT GENERALISE TO OTHER INSTRUMENTS WITH THE SAME FAILURE MODE — even for the person who wrote it, on the same day, having applied it correctly a dozen times.** *"Never pipe the GATE through `head`"* is obeyed faithfully while the identical truncation is committed on a comment thread. **State rules in terms of the PROPERTY that fails (never act on a truncated document) rather than the TOOL you first met it on.** → [`a-rule-bound-to-an-instrument-does-not-transfer`](#a-rule-bound-to-an-instrument-does-not-transfer)

## fixing-a-thing-can-make-the-next-fix-dearer

**Two correct fixes, and landing them in the wrong order makes the second one
cost more than it does today. The usual intuition — defer the bigger change,
it will keep — is backwards whenever the first fix makes broken data VALID.**

**Worked example, 2026-08-27, caught before it landed.** Fuel's Vulkan backend
stamps `kernel_source = "vulkan-slang"`, a tag absent from
`kernel_source_intern`'s closed allowlist. Two fixes were queued:

```
PR 1  MECHANISM   allowlist -> interner; an unknown tag is preserved,
                  never silently coerced to ""
PR 2  PRODUCER    vulkan-slang -> slang  (21 FKC sites)
```

**Measured before PR 1: the rename was PROVABLY FREE.** `kernel_source` is part
of `ProfileEntry`'s key, so a rename normally orphans persisted profiles — but
every `vulkan-slang` profile was *already* broken, because the tag collapsed to
`""` or tripped a `debug_assert` at `DispatchTable`-build time. **Nothing valid
keyed on it, so renaming orphaned nothing.**

**PR 1 changes that.** Once the interner preserves the tag, profiles written
under it become valid — **and from that moment the rename starts orphaning real
data.** The window is empty today and opens the instant the mechanism lands.

**And the second-order version is worse than the accounting one: A FIX THAT
MAKES A WRONG THING *FUNCTION* REMOVES THE PRESSURE TO MAKE IT RIGHT.** While
`vulkan-slang` is broken it is a forcing function — somebody trips over it.
Afterwards it works, and **a category error that works is one nobody comes back
to.**

**PRACTICE.** When two fixes queue against one defect, ask **which of them makes
the other's precondition disappear**, and land that one *second* — or land them
back-to-back and say so, so a slip is visible rather than silent. Concretely:

- **Ask what becomes VALID after fix 1** that is currently broken. Anything in
  that set is data fix 2 will now have to migrate.
- **Do not fold the cheap-today change into a deferred umbrella row** because it
  shares a blast radius with it. Shared *code* is not shared *timing*, and the
  umbrella's schedule is what makes the cheap change expensive.
- **Re-verify the free-ness at LAND time, not at ruling time.** The subject
  changes between the two, which is
  `reverify-differential-after-rebase-before-push`
  with the mutation coming from your own queue rather than a peer's.

**The tell that you are in this situation:** the deferred change is described as
*"free"* or *"orphans nothing"* **on the strength of the current broken state.**
That freeness is a property of the bug, not of the change — and the other fix is
about to remove the bug.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Two correct fixes can be ordered wrongly: ask which one makes the OTHER's precondition disappear, and land that one second.** A change described as *"free"* or *"orphans nothing"* on the strength of the CURRENT BROKEN STATE has a freeness that belongs to the bug, not to the change — and the other fix is about to remove the bug. ⚠️ And a fix that makes a wrong thing FUNCTION removes the pressure to make it right. → [`fixing-a-thing-can-make-the-next-fix-dearer`](#fixing-a-thing-can-make-the-next-fix-dearer)

## a-sabotage-that-never-applied

**A perturbation that FAILS TO APPLY reports absence-of-sensitivity as
presence. The green is real, the code under it was never sabotaged, and the
conclusion — "this test does not discriminate" — is a finding that is wrong.**

**This is the INVERSE of the usual sabotage failure and it is quieter.** The
known one is a passing sabotage caused by a warm cache
([`sabotage-calibrated-tolerances`](#sabotage-calibrated-tolerances)): the source
changed, the binary did not. **This one is worse — the SOURCE never changed
either**, so a `git status` is clean, a recompile is honest, and every artifact
agrees.

**Worked example, 2026-08-27, item 8 (II).** A perturbation hard-wiring
`rope_scaling` to `None` reported **14 passed, 0 failed** — read naively, *the
tolerance test is vacuous*. It was not: **`cargo fmt` had rewrapped the target
across four lines, so the exact-string anchor matched 0 occurrences.**

⚠️ **AND THE GATE WORKED. THE SHELL IGNORED IT.** The script asserted
`count == 1` and **that assertion FIRED** — then bash carried on and ran the
tests anyway, because a non-zero exit from the perturbation step was not fatal to
the surrounding shell. **A correct gate, correctly failing, with its verdict
discarded by the thing that called it** — the same defect as piping a gate or
echoing its exit code into an `&&` chain
([`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it)),
arriving through the harness rather than the invocation.

**Re-run with a wrap-tolerant regex and `set -e`: 11 passed, 3 FAILED**, the
tolerance test among them.

**PRACTICE, and the third is the standing form:**

- **`set -e`, or check the perturbation's exit status before believing the test
  run.** A harness that continues past a failed perturbation **cannot
  distinguish *"the test is vacuous"* from *"the sabotage never happened"* —
  both present as a pass.**
- **Anchor perturbations with a formatting-tolerant regex, or perturb BEFORE
  running `cargo fmt`.** Any exact-string anchor spanning a method chain is
  fragile by construction, and a formatter is entitled to rewrap it.
- ⚠️ **A PASSING SABOTAGE IS A RED FLAG, NOT A RESULT. Suspect the harness
  before the subject — the base rate favours it and the check costs one grep.**

**What caught it was domain judgement, not an instrument:** `rope_scaling: None`
*must* break an assertion reading `.is_some()`, so the pass was implausible on
its face. **That is the same detector as disbelieving a config with a documented
`-1` sentinel that scored zero cross-field defaults — and in both cases every
automated check agreed with the wrong answer.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A PASSING sabotage is a red flag, not a result — suspect the HARNESS before the subject.** A perturbation that fails to apply reports absence-of-sensitivity as PRESENCE, and unlike the warm-cache variant the SOURCE never changed either, so `git status` is clean and every artifact agrees. ⚠️ Use `set -e`: an assertion can fire correctly and the shell carry on regardless. → [`a-sabotage-that-never-applied`](#a-sabotage-that-never-applied)

## a-report-is-not-a-gate

**`&&` only helps if the gate's EXIT CODE encodes its verdict. A step that
COUNTS and PRINTS is a report; a step that EXITS NON-ZERO is a gate — and in a
chain they are indistinguishable until one of them lets something through.**

**This is the second half of
[`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it),
and the half that was missing.** That rule says: put the gate IN the `&&` chain,
never pipe it, never route its status through `echo`. **Necessary, and not
sufficient.**

**Worked example, 2026-08-27, item 8 (II).** A lane chained
`cargo clippy … && git commit && git push`, **saw the gate print
`fuel-core clippy: 1`, and pushed the regression anyway.**

**The chain was correct. The gate was not.** `cargo clippy` **exits 0 on
warnings** unless given `-D warnings` — so the verdict went to stdout and the
exit status said *fine*. **Every structural rule was obeyed and the defect
shipped.**

```
the recorded failure   gate OUTSIDE the chain             -> put it in the chain
this one               gate INSIDE the chain, but its
                       exit code does not carry its       -> make it FAIL,
                       verdict                               not merely REPORT
```

**PRACTICE:**

- **For a counting gate, convert the count into an exit status** — `-D warnings`,
  or an explicit `[ "$n" -eq 0 ] || exit 1` after it. **Do not rely on reading
  the number**, because the whole point of a chain is that nobody has to.
- **Ask of any step you put in an `&&` chain: what makes this exit non-zero?**
  If the answer is *"the tool crashing"* rather than *"the condition I care
  about"*, it is a report wearing a gate's position.
- **The tell is a tool with a `-D` / `--strict` / `--check` flag you did not
  pass.** Formatters, linters and validators overwhelmingly default to
  reporting; the strict flag is what turns them into gates, and its absence is
  invisible in a chain.

**Shares a root with
[`a-sabotage-that-never-applied`](#a-sabotage-that-never-applied) — a correct
verdict that nothing acts on — and has a different cause and a different fix.**
There, an assertion fired and the shell continued. Here, nothing fired at all.
**Grouping them under "put the gate in the chain" would fix neither.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`&&` only helps if the gate's EXIT CODE carries its verdict.** A step that COUNTS and PRINTS is a report; one that EXITS NON-ZERO is a gate, and in a chain they are indistinguishable until something gets through. `cargo clippy` exits 0 on warnings — a lane watched its own gate print `1` and pushed the regression. The tell is a `-D` / `--strict` / `--check` flag you did not pass. → [`a-report-is-not-a-gate`](#a-report-is-not-a-gate)

## a-new-lens-does-not-re-audit-old-findings

**Discovering a defect class does not retroactively re-audit the findings you
already made. A claim formed BEFORE the lens exists never meets the lens unless
you deliberately re-run it — and the claims most likely to have escaped are your
earliest and most confident ones.**

**Worked example, 2026-08-27, the doc-vs-code audit.** A lane auditing 15
architecture documents found, in its **second** pass, a defect class it had not
been given: **a satisfied non-goal** — an absence that CONFIRMS the constitution,
which a *named-but-missing* grep reports as a violation.

**They wrote the class down, swept it FORWARD over the remaining documents, and
never swept it BACKWARD over the four they had already done.** Their flagship
finding — reported to the architect as the audit's strongest result, and filed by
the architect as a roadmap headline — **was an instance of exactly that class.**
The document declared the surface unbuilt, **in bold, twice**, named its
prerequisites, and pointed at a sequencing plan.

**Two failures compounded, and they are separable:**

- **Forward-only sweep.** The lens was applied from the moment of discovery
  onward. **Nothing connects a new class to old conclusions except a deliberate
  re-run**, and the earlier work is where the confident claims live.
- **Filing ahead of the evidence.** The architect made it a headline and
  requested the supporting passages *afterwards*. **The passages requested were
  the passages that refuted it.** One message, in the other order, would have
  prevented it.

**PRACTICE:**

- **When you discover a defect class mid-task, re-run it over everything you have
  already concluded, and say that you did.** "Swept forward" and "swept" are
  different claims.
- **Re-check the FLAGSHIP finding first.** It was formed earliest, with the least
  calibration, and it is the one already in flight to someone else.
- **Never file a claim whose supporting passage you have not read.** A relayed
  reading is a claim about a passage, not the passage.

**COROLLARY — A RETRACTION BOUNDS NOTHING. Finding one instance of a class does
not tell you the class's SIZE, and it is most misleading when the instance was
found by the very sweep that had not finished.**

**Worked example, same incident, two hops downstream.** After the retraction, a
count of **21** was already on the project owner's desk. Both the architect and
the portfolio PM independently reached for **`21 − 1`**, and the PM caught it:
*"that is arithmetic, not a measurement, and the other 20 have not been
re-dispositioned."* **Then it turned out to be worse — the remaining ~14 had not
even been polarity-checked**, because the lens had been applied to the file group
only.

**So `21 − 1` does not merely substitute arithmetic for measurement. It assumes
EXACTLY ONE instance of a class nobody has finished looking for, and it claims
completeness on behalf of the incomplete pass that produced the retraction.**

⚠️ **The same fallacy hides in prose. *"Stale by one"* names a size for a class
nobody has counted.** The honest form is **"stale by an unknown amount, bounded
below by one"** — and the remedy is to carry the ORIGINAL figure marked
provisional, with the reason, until the sweep that would bound it actually runs.
**A number corrected by subtraction is a new claim, not a repair.**

**Related: [`a-defence-can-outlive-its-defect`](#a-defence-can-outlive-its-defect)
— there a remedy survives its cause; here a CONCLUSION survives the arrival of the
thing that would have refuted it.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Discovering a defect class does not retroactively re-audit the findings you already made.** A claim formed BEFORE the lens exists never meets it unless you deliberately re-run — and "swept forward" is not "swept". ⚠️ Re-check the FLAGSHIP finding first: it was formed earliest, with the least calibration, and it is the one already in flight to someone else. **And never file a claim whose supporting passage you have not read — a relayed reading is a claim ABOUT a passage, not the passage.** → [`a-new-lens-does-not-re-audit-old-findings`](#a-new-lens-does-not-re-audit-old-findings)

## born-red-the-aim-not-the-shape

**Pointing a gate at something known-broken proves the STEP SHAPE reddens. It
is blind to a gate aimed at nothing. For a `cfg`-gated cell the proof is a SEED
IN THE CELL'S OWN GATED REGION, and it needs TWO arms: the leg goes RED, and a
DEFAULT build stays GREEN.**

**The second arm is the load-bearing one. Without it, a red leg is
indistinguishable from a merely broken crate.**

**Worked example, 2026-08-27.** Four CI legs were added over non-default feature
cells. The architect prescribed the weak form — re-point each invocation at a
crate known to fail, confirm red. **All four went red. One of them compiled NONE
of its feature's code.**

`cargo check -p fuel-dispatch --features baracuda-types` compiles nothing that
the feature gates: **every `baracuda-types`-gated line lives inside `telemetry/`,
and `pub mod telemetry` is itself `#[cfg(feature = "telemetry")]`.** The fix is
`--features telemetry,baracuda-types` — **[`one-feature-is-not-two`](#one-feature-is-not-two),
committed while writing the comment that warned against the analogous defect.**

**It would have shipped as a permanent green gating nothing, and the prescribed
born-red certified it.**

**THE STRONG FORM:**

```
seed   #[cfg(feature = X)] const _SEED: u8 = <undefined>;
       ...into the crate's OWN gated region

(a) the leg must go RED        <- proves it reaches the cell
(b) a DEFAULT build must stay  <- proves the red came from the CELL and not
    GREEN                          from the crate being broken generally
```

**Both arms, or the result is uninformative in the direction that looks like
success.**

⚠️ **AND THE SEED ITSELF CAN BE MALFORMED IN A WAY THAT READS AS A FINDING.** The
lane's first attempt spliced the seed **directly after an existing `#[cfg]`
attribute, stealing it from the item below and un-gating that item** — so the
DEFAULT build broke too, and the result read INCONCLUSIVE *in a way that looked
like a discovery about the crate.* **Append a COMPLETE gated item; never splice
between an attribute and what it modifies.** **An inconclusive control reads like
evidence**, which is why it belongs in the record rather than being quietly
rerun.

**WHERE THE STRONG FORM IS UNAVAILABLE, SAY SO.** A feature that only adds a
dependency (`onnx = ["fuel-onnx"]`) gates no code and has no region to seed —
**that leg has the weak form only, and the report must say it rather than let a
reader assume parity across the set.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Pointing a gate at something known-broken proves the STEP SHAPE reddens — it is BLIND to a gate aimed at nothing.** For a `cfg`-gated cell, seed the cell's OWN gated region and require BOTH that the leg goes RED **and that a DEFAULT build stays GREEN**; the second arm is load-bearing, because without it a red leg is indistinguishable from a merely broken crate. ⚠️ And never splice a seed between an attribute and the item it modifies — it steals the `#[cfg]` and the malformed control reads like a finding. → [`born-red-the-aim-not-the-shape`](#born-red-the-aim-not-the-shape)

## marking-one-representation-does-not-mark-the-others

**ONE CLAIM CAN EXIST IN SEVERAL REPRESENTATIONS INSIDE ONE DOCUMENT — A
DIAGRAM, PROSE, A BULLET LIST, TWICE IN A SENTENCE — AND EACH IS INVISIBLE
FROM THE ONE YOU ARE LOOKING AT. A FIX SCOPED TO THE REPRESENTATION THAT
PROMPTED IT LEAVES THE CLAIM STANDING EVERYWHERE ELSE, AND LOOKS COMPLETE.**

**Worked example, `docs/architecture/02-layers.md`, 2026-08-27/28. FOUR
deliberate, increasingly-correct remedies. The claim survived all four:**

```
2026-07-29  an as-built NOTE under the diagram        -> went stale itself
v0.8        remedy = another diagram-adjacent note    -> too weak, and the
                                                         file's OWN TEXT said so
a6b5476d    remedy = mark inside the DIAGRAM          -> right, and the PROSE
                                                         went on making the claim
v0.10       remedy = mark the prose leaves too        -> right, and one name
                                                         appeared TWICE ON ONE LINE;
                                                         a first-occurrence replace
                                                         marked one of them
```

**Every remedy was stronger than the last. Every one was correct about the
surface it addressed. NOTHING ENUMERATED THE SURFACES.**

⚠️ **THE SHARPEST INSTANCE IS THE SECOND ROW: v0.8 EXISTS *BECAUSE* THE EARLIER
NOTE WENT STALE, AND ITS OWN TEXT READS *"the remedy for a stale diagram entry
was itself a diagram-adjacent prose claim."* THE AUTHOR WROTE THAT SENTENCE AND
APPLIED THE SAME WEAK REMEDY TO FOUR MORE NAMES IN THE SAME FILE HOURS LATER.**
A document diagnosing a failure mode does not inoculate the next edit to it.

**AND THE FOURTH ROW IS THE ONE THAT ENDS THE ARGUMENT: it was caught by
RE-COUNTING, not by reading.** A careful reader had just read that line while
editing it. **So the honest close is to write NOT PROVABLY COMPLETE into the
commit** — a fifth confident fix would have taught the wrong lesson.

**PRACTICE: when a claim is wrong, do not fix the instance — ENUMERATE ITS
REPRESENTATIONS FIRST** (grep the name, count occurrences, and check whether any
share a line). **If you cannot enumerate them, say the pass is not complete and
reach for a MECHANICAL GUARD rather than a further careful read** — a grep does
not have the scoped-to-what-prompted-it failure mode. Related:
`verify-the-population-not-the-instance`,
`fix-the-generator-not-the-output`.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **ONE CLAIM CAN EXIST IN SEVERAL REPRESENTATIONS IN ONE DOCUMENT — DIAGRAM, PROSE, BULLET LIST, TWICE IN A SENTENCE — AND EACH IS INVISIBLE FROM THE ONE YOU ARE LOOKING AT.** A fix scoped to the representation that prompted it leaves the claim standing everywhere else **and looks complete**. Worked example: FOUR increasingly-correct remedies to `02-layers.md` and the claim survived all four — including one where the file's own text had already diagnosed the weak remedy being applied. **Enumerate the representations BEFORE fixing; if you cannot, write NOT PROVABLY COMPLETE into the commit and reach for a mechanical guard rather than a further careful read.** → [`marking-one-representation-does-not-mark-the-others`](#marking-one-representation-does-not-mark-the-others)

## a-pre-stated-blocker-is-a-prediction

**A BLOCKER YOU NAME BEFORE MEASURING IS A PREDICTION, AND THESE FAIL
OVERWHELMINGLY IN ONE DIRECTION: THE DIRECTION THAT STOPS THE WORK. PHRASE A
STOP-CONDITION AS *STOP AND REPORT*, NEVER AS THE OUTCOME — AN
OUTCOME-PHRASED FENCE LICENSES FILING THE PREDICTION AS A RESULT.**

**Worked example: `fuel-cpu-backend --features mkl`, 2026-08-27. THREE
predictions in one thread, from two people, ALL WRONG, ALL IN THE STOPPING
DIRECTION:**

```
predicted                            measured
----------------------------------   -----------------------------------------
"5 errors blocked on f16             a MISSING QUALIFICATION. `half` was already
 stabilisation (rust#116909)"        a dep; `f16::ONE` resolved to Rust's
                                     unstable PRIMITIVE because the file
                                     imported `half` nowhere. 19 of 19 fixed,
                                     none blocked.

"no CI leg -- the cell needs a       VIABLE. ocipkg fetched MKL on a clean
 preinstalled SDK"                   runner. Step green in 39s.

"the leg may pull a large archive    COMPARABLE: onnx 25s, mkl 39s, against
 -- read the duration before         0s/3s/6s for the others.
 treating it as cheap"
```

**Each was individually reasonable. Each dissolved on contact with a
measurement. And each would have PREVENTED the work rather than misdirected
it** — which is why the class is expensive and quiet: **the output of a
wrong stopping-prediction is an ABSENCE, and nobody audits work that was never
attempted.**

⚠️ **THE MOST DURABLE WRONG ANSWER AVAILABLE IS A BLOCKED ITEM WITH A NAMED
UPSTREAM ISSUE.** *"14 fixed, 5 blocked on rust#116909"* would have been
accepted by everyone, filed, and never revisited — **it LOOKS resolved.**
And the error message named a REAL upstream limitation that was not the one
being hit, **so the plausible reading was also the well-evidenced one.**

**THE FENCE DESIGN THAT SAVED IT, named by the lane that went past it:**

```
OUTCOME-PHRASED   "if they are blocked on f16, file them as blocked"
                  -> licenses the PREDICTED OUTCOME. Stops the looking.

STOP-AND-REPORT   "if they are blocked, STOP AND REPORT WHAT YOU FOUND"
                  -> licenses only STOPPING. The report requires looking at
                     the thing, which is what dissolved it.
```

**PRACTICE: name the predicted blocker so it can be RECOGNISED, but make the
deliverable a REPORT ON WHAT WAS FOUND, never the predicted disposition. And
when a fence fires, that is the moment to LOOK HARDEST** — a fence firing
is the least-audited event in a task, because it feels like the process
working. Related:
[`uninformative-signals-both-directions`](#uninformative-signals-both-directions),
`magnitude-is-not-impossibility`.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A BLOCKER NAMED BEFORE MEASURING IS A PREDICTION, AND THEY FAIL IN THE DIRECTION THAT STOPS THE WORK.** Three in one thread on `fuel-cpu-backend --features mkl`, all wrong: *"5 blocked on `f16` stabilisation"* was a missing qualification (19/19 fixed), *"needs a preinstalled SDK"* was a 39-second green leg, *"may pull a large archive"* was comparable to the others. **The output of a wrong stopping-prediction is an ABSENCE, and nobody audits work never attempted** — **a blocked item with a named upstream issue is the most durable wrong answer available, because it LOOKS resolved.** **Phrase a stop-condition as STOP AND REPORT, never as the outcome: an outcome-phrased fence licenses filing the prediction as a result. When a fence fires, look hardest.** → [`a-pre-stated-blocker-is-a-prediction`](#a-pre-stated-blocker-is-a-prediction)

## the-instrument-nearest-to-hand

**THE COMMONEST SOURCING ERROR IS NOT USING A BAD INSTRUMENT. IT IS REACHING
FOR THE ONE NEAREST TO HAND INSTEAD OF THE ONE THAT ANSWERS THE QUESTION** —
**and it feels like diligence, because you DID consult a source.**

**Three instances on 2026-08-28, three different parties, one shape. Each was
caught by a different mechanism and none by review:**

```
PROXIMITY    Baracuda was closest to their own build's death and was WRONG
             about its cause -- they reported a cross-project collision; it was
             self-inflicted. Caught by their own retry succeeding after they
             capped their OWN threads.

AUTHORSHIP   The party that IMPLEMENTED `NUM_JOBS` support wrote a durable
             config comment saying their forge does not read it. Caught by two
             review bots on the comment, by accident.

RECENCY      A coordinator read the CLAUDE.md SNAPSHOT loaded at session start
             and reported it as the file's current content. It had been
             corrected hours earlier. Caught by the file's owner.
```

**Being NEAR a thing, having AUTHORED a thing, and having RECENTLY READ a thing
all feel like standing, and none of them is evidence.** The victim, the
implementer and the reader each had the strongest available claim to know, and
each was wrong in the direction their position made comfortable.

⚠️ **RECENCY IS THE WORST OF THE THREE IN AN AGENT SESSION, because a cached
snapshot is INDISTINGUISHABLE FROM HEAD at the point of use.** A stale working
tree at least sits at a path you could question; **a context snapshot has no
path and no timestamp in the moment you read it.** The remedy is the one this
file already mandates for trees — **`git show <ref>:<path>`, at a NAMED ref** —
and it applies to your own project's files, not only to other people's.

**PRACTICE: before citing a source, ask what makes it AUTHORITATIVE for THIS
question rather than merely CLOSE.** For a file's current content that is a ref,
never a memory. For a build's cause it is a log, never a proximity. **For a
crate's behaviour it is the version you RESOLVE, never the version you
authored** — a version boundary is exactly where authorship stops being
evidence. Related:
[`a-stale-tool-is-a-wrong-action`](#a-stale-tool-is-a-wrong-action),
`go-to-the-artefact-not-the-rendering`.

⚠️ **AND THE COMPANION INSTRUMENT DEFECT, CAUGHT WITHIN AN HOUR OF THIS RULE
LANDING, BY THE LANE APPLYING IT: YOUR *DEFINITION* QUERY'S KEYWORD SET IS
ITSELF A POPULATION CLAIM.**

Checking every name in a mixed list (as this rule demands), a lane's query
reported **`KernelRef`: ZERO definitions, 54 files mentioning it** — which
reads exactly like a second false name in the sentence being corrected.
**It is real: `pub type KernelRef = fn(...)`, a TYPE ALIAS.** The query was
keyed on `struct|enum|trait|mod` and Rust declares things with `type`, `const`,
`static`, `fn`, and macros as well.

**A query keyed on SOME declaration keywords misses the others SILENTLY, and
its output is indistinguishable from a real absence.** Had it been trusted, the
correction of one false name would have introduced another **into the very
sentence being fixed.**

**PRACTICE: when a definition query returns zero, name the declaration FORMS it
covers before reporting the absence** — and note that a high mention-count
beside a zero definition-count is the signature of this defect, not of a
phantom.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **THE COMMONEST SOURCING ERROR IS REACHING FOR THE INSTRUMENT NEAREST TO HAND RATHER THAN THE ONE THAT ANSWERS THE QUESTION** — and it feels like diligence, because you DID consult a source. Three instances in one day, three parties: **PROXIMITY** (the project closest to its own build's death was wrong about the cause) · **AUTHORSHIP** (the party that IMPLEMENTED `NUM_JOBS` wrote a comment denying their forge reads it) · **RECENCY** (a coordinator read a session-start CLAUDE.md snapshot and reported it as head, hours after it was corrected). ⚠️ **RECENCY is the worst in an agent session: a cached snapshot is INDISTINGUISHABLE FROM HEAD at the point of use** — a stale tree at least sits at a path you could question. **Use `git show <ref>:<path>` for your OWN project's files too, not only other people's.** → [`the-instrument-nearest-to-hand`](#the-instrument-nearest-to-hand)

## a-true-half-vouches-for-the-false-half

**A SENTENCE NAMING SEVERAL ARTIFACTS, WHERE SOME EXIST AND SOME DO NOT, READS
AS VERIFIED. The real names lend their credibility to the absent ones, and a
reader spot-checking ONE name is likelier to hit a real one.**

**Two instances, 2026-08-28, found by the doc-vs-code audit:**

```
02-layers:77       "...`Node`, `Graph`, `FusedOpRegistry` metadata types,
                    `OptimizationMap` rules..."
                   FusedOpRegistry EXISTS. OptimizationMap does NOT.

04-optimization    "A rule's FAMILY, COST CONTRIBUTION, and FRONTIER
                    COMPATIBILITY are part of its identity"
                   family()/RuleFamily EXIST on `pub trait Rule`.
                   frontier compatibility: NO method, NO enum, 0 occurrences.
                   -> nothing can DECLARE it, so nothing can READ it, and the
                      doc's closing present-tense claim that "the optimizer
                      reads the declaration" cannot be true.
```

**Why these survive review specifically: the sentence PASSES a spot check.**
Verification effort scales with the number of names, attention does not, and
**the first name a reader tries is the one that vouches for the rest.** A list
of three where two are real is more dangerous than a single false claim,
because a single false claim has nothing standing next to it.

⚠️ **AND EVERY LIST IS AN OPPORTUNITY FOR THIS** — the failure needs no
carelessness, only a doc that outlived one of its members. Both instances above
are lists that were true when written.

**PRACTICE: check EVERY name in a list, or state which ones you checked.** A
report of the form *"spot-checked, looks right"* over a multi-name sentence is
an unbounded claim from a bounded measurement. **When writing such a sentence,
prefer separate clauses over a comma list** — a list invites the reader to
sample, and separate claims each have to stand on their own. Related:
`verify-the-population-not-the-instance`,
`enumerate-the-population-not-the-strings`.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A SENTENCE NAMING SEVERAL ARTIFACTS, SOME REAL AND SOME NOT, READS AS VERIFIED — THE TRUE HALF VOUCHES FOR THE FALSE HALF.** Two instances: `02-layers` named `FusedOpRegistry` (exists) beside `OptimizationMap` (does not); `04-optimization` named a rule's *family* (real: `family()`/`RuleFamily` on `pub trait Rule`) beside *frontier compatibility* (**no method, no enum, 0 occurrences** — so nothing can declare it and the doc's present-tense *"the optimizer reads the declaration"* cannot be true). **These survive review because the sentence PASSES A SPOT CHECK: the first name a reader tries is likelier to be a real one.** **Check EVERY name in a list or state which you checked** — and when writing, prefer separate clauses to a comma list, because a list invites sampling. → [`a-true-half-vouches-for-the-false-half`](#a-true-half-vouches-for-the-false-half)

## a-guard-exists-is-not-the-guard-protects-this

**"A GUARD EXISTS" AND "THE GUARD PROTECTS *THIS* PROPERTY" ARE DIFFERENT
CLAIMS. The guard is correct in both instances below — the error is in the
READER, which is why no amount of hardening the guard prevents it.**

**Two instances, 2026-08-28, hours apart, one caught and one nearly committed:**

```
NEON      "the CI matrix HAS an aarch64 runner"
          inferred: "so the neon-dotprod cell is LIVE there"
          FALSE -- the cell is triply gated and its intrinsic is nightly-only.
          The runner exists; the cell still compiles itself away.
          Cost: a coordinator corrected a lane's TRUE statement into a false
          one, which the lane then propagated into their own file.

SLOT      "cuda-build.ps1 serialises CUDA builds"
          inferred: "so it protects Baracuda's in-flight measurement"
          FALSE -- SlotCount = 2, so a second build is ADMITTED, not blocked.
          It prevents OVERSUBSCRIPTION. It does not protect a MEASUREMENT.
          Caught before running; the lane's own words: "I know the wrapper
          serialises CUDA builds; I would have reasoned that the slot
          protects me."
```

**In both, the guard's own contract is accurate and narrower than the use it
was put to.** Reading the guard would have shown it; **reading that the guard
EXISTS showed nothing** — and existence is what a hurried reader checks.

⚠️ **THE SLOT CASE IS THE MORE DANGEROUS SHAPE, because the guard fires
CORRECTLY and admits you.** A blocked build tells you something; **an admitted
one tells you the system is content, and the system IS content — it was never
asked about measurement integrity.**

**PRACTICE: name the PROPERTY you need protected, then read the guard's
contract for that property specifically.** *"A resource guard is not a
measurement guard"* is the worked form; the general form is that **a guard
protects the invariant it was written for and NOTHING adjacent, however
similar the adjacent thing looks.** Related:
[`the-instrument-nearest-to-hand`](#the-instrument-nearest-to-hand),
[`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it).

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **"A GUARD EXISTS" IS NOT "THE GUARD PROTECTS *THIS* PROPERTY"** — and in both instances the guard was CORRECT, so the error is the reader's and no hardening prevents it. *"The matrix has an aarch64 runner"* → *"so the `neon-dotprod` cell is live there"* (false: triply gated, nightly-only intrinsic). *"`cuda-build.ps1` serialises CUDA builds"* → *"so it protects an in-flight measurement"* (false: `SlotCount = 2` ADMITS a second build — **it prevents OVERSUBSCRIPTION, not contamination; a resource guard is not a measurement guard**). ⚠️ **The slot shape is the worse one: the guard fires CORRECTLY and lets you through, so a green admission reads as approval of a question nobody asked it.** **Name the property you need protected, then read that guard's contract FOR THAT PROPERTY.** → [`a-guard-exists-is-not-the-guard-protects-this`](#a-guard-exists-is-not-the-guard-protects-this)

## a-control-must-vary-along-the-claims-axis

> **Index line (in CLAUDE.md):** **When a claim is scoped to a CONFIGURATION — a version, a feature, a target, a host — the control must VARY along that axis or it cannot see the error.** The instrument runs, the control passes, and the configuration under test is never exercised.

**WHEN A CLAIM IS SCOPED TO A CONFIGURATION, A CONTROL THAT DOES NOT VARY ALONG THAT AXIS IS A CONTROL FOR A DIFFERENT CLAIM (2026-09-02).** This corpus already carries the bound that *a positive control proves the query CAN FIND the thing it looks for; it does not prove the query LOOKS FOR THE RIGHT THING*. **This is the sharper case, and it is nastier: the query is right, the instrument is right, the control genuinely passes — and the CONFIGURATION the claim is scoped to was never exercised.** The three instances below come from three unrelated tools and were carried as three separate lessons. **They are one lesson about controls.**

| axis | the claim | the control that PASSED | what it actually proved | what it could not see |
|---|---|---|---|---|
| **VERSION** | `gpu-run.ps1` / `cuda-build.ps1` fail to parse under **PowerShell 5.1** — they parse fine under 7 | a deliberately-unbalanced `.ps1`, returning 1 parse error | the parser API works and reports errors | **WHICH parser.** An unbalanced script errors under 5.1 *and* under 7, so the control is **version-insensitive by construction** |
| **FEATURE** | an exhaustive `match` in `baracuda_provider.rs` is parsed only under a feature **pair** | `--features telemetry` — *strictly stronger* than a default build | the crate compiles with telemetry on | **the intersection.** The module is cfg'd *inside* another cfg, so no single feature reaches it |
| **CFG / TARGET KIND** | a `cfg`'d module's code, or a crate's `#[cfg(test)]` code, was compiled | `Checking fuel-cuda-backend v0.10.3` — a real target-crate artifact | the crate was reached | **the MODULE, and the TEST TARGETS.** A crate-level compile line collapses both dimensions; the discriminating artifact is `(lib test)` |

**In every row: the instrument RAN, the control PASSED, and the configuration under test was never exercised.** One row is an anecdote. Three, across a shell, a compiler and a build system, is a class.

**THE OPERATIONAL TEST, and it is one question:** ***does my control VARY along the axis the claim is scoped to?*** If it does not, it validates the **instrument** and says nothing about the **configuration**. For the version case the control that was needed is not "a script that fails to parse" but **a script that parses under 7 and FAILS under 5.1** — the control has to be able to tell the two hosts apart, because that distinction *is* the claim.

**THE LEAD EXAMPLE IS WORTH THE SPACE BECAUSE IT NEARLY SUCCEEDED.** Re-measuring a closed row (`GAP-223`) to decide whether its defect had recurred, the first run used `[Parser]::ParseFile` under the host that happened to be attached — **pwsh 7.6.5** — with the unbalanced-script control returning **1**, and reported **0 parse errors** for both scripts. **Correct API, passing control, clean answer, wrong conclusion available for free.** It was caught only because `GAP-223`'s own text named **both** hosts, quoting *"0 under pwsh 7.6.5"* beside *"shared-checkout 10 / 4"* under 5.1. ⚠️ **The row was the instrument that validated the instrument — and a row that had merely said "the scripts fail to parse" would have let the clean zero stand.** Re-run under 5.1 gave the same zeros, **so the conclusion survived by luck rather than by method**, which is the part that makes it teach rather than merely warn.

**⚠️ COROLLARY — THE FORCING MECHANISM IS SUBJECT TO THE SAME AXIS AS THE CONTROL.** Measuring `E0133` across target-features, the script forced a cold rebuild by touching the alphabetically-first source file — `avx.rs` — **which under default features cargo never reads**, so the fingerprint stayed valid and the run was warm. **A cold-forcing touch on a file the axis gates away is silently a no-op, and it produces a warm run reporting zero for the honest reason that nothing compiled.** The rule above says the CONTROL must vary along the claim's axis; this says the thing that makes the measurement HAPPEN must not be subject to that axis at all. **Touch a file the configuration always reads (`lib.rs`), never one the axis can gate out.** Caught only by a harness guard requiring the target crate's own `Checking` line — `Checking=0` reads as *about the HARNESS, not the code* — which is why this is a corollary and not a wrong number in a registry.

**WHAT THIS SUBSUMES — cross-referenced, not replaced, because three orphaned lessons is worse than one duplicated one.** [`one-feature-is-not-two`](#one-feature-is-not-two) is the **feature** row; [`target-crate-compile-line`](#target-crate-compile-line) and [`lib-does-not-build-tests`](#lib-does-not-build-tests) are the **cfg / target-kind** row. Each remains correct and each carries detail this rule does not. **Read them as instances; read this as the axis they share.** Closest neighbours: [`a-rule-bound-to-an-instrument-does-not-transfer`](#a-rule-bound-to-an-instrument-does-not-transfer) (the instrument changes, the rule does not follow) and [`a-guard-exists-is-not-the-guard-protects-this`](#a-guard-exists-is-not-the-guard-protects-this) (the guard is real and aimed elsewhere). **The difference from both: here nothing is stale and nothing is misaimed. The control is correct for a claim that is one configuration away from the one being made.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **When a claim is scoped to a CONFIGURATION — a version, a feature, a target, a host — the control must VARY along that axis or it cannot see the error.** The instrument runs, the control passes, and the configuration under test is never exercised. → [`a-control-must-vary-along-the-claims-axis`](#a-control-must-vary-along-the-claims-axis)

## an-incomplete-decoder-produces-false-equality

**AN INCOMPLETE DECODER DOES NOT PRODUCE OBVIOUSLY-MISSING OUTPUT. IT PRODUCES
FALSE EQUALITY — distinct inputs collapse to identical renderings, and the
collapse is invisible because the output is WELL-FORMED.**

**Measured 2026-09-02 on GAP-260** (`fuel-vulkan-backend`, host-visible memory
layout). The diagnostic decodes `VkMemoryPropertyFlags` bits to names. vulkane's
`MemoryPropertyFlags` constants cover five bits; this box's AMD Radeon 610M sets
**`0xc0`** on **eight of its sixteen** memory types — `vk.xml` bitpos 6/7,
`DEVICE_COHERENT_BIT_AMD` and `DEVICE_UNCACHED_BIT_AMD`.

**Had the decoder rendered only the bits it could name, those eight types would
have printed IDENTICALLY to four others:**

```
  [3]  0x000e  ->  HOST_VISIBLE + HOST_COHERENT + HOST_CACHED
  [7]  0x00ce  ->  HOST_VISIBLE + HOST_COHERENT + HOST_CACHED   <- SAME RENDERING
```

**Measured collapse, counting DISTINCT FLAG VALUES rather than rows:** the
adapter's sixteen types carry **8 distinct flag values**, which a dropping
decoder renders as **4 distinct strings**. Restricted to the host-visible types
that GAP-260 is about: **6 distinct values render as 3**. Exactly halved, both
ways. **The reader does not see a gap — the reader sees rows that agree, and
concludes the device has half the memory types it has.**

**WHY THIS HIDES WHERE OTHER COLLAPSES DO NOT.** This is the same *mechanism* as
[`injectivity-and-collapsed-mappings`](#injectivity-and-collapsed-mappings) — two
inputs, one output, and a false agreement filed where a disagreement would be
investigated. **But every instrument in that rule assumes a RETURNED VALUE**:
*check the returned value*, *demand injectivity where the output is an identity*,
the greppable five-line window around a decline. **A decoder has none of those.
There is no assertion, no comparison, no verdict — it is a `println!` read by a
human, and the only thing that could catch it is noticing that two rows which
should differ do not.**

**REMEDY, and it is a different prescription: PRESERVE WHAT YOU CANNOT NAME.**
Render unrecognised input as itself — `<unknown 0xc0>` — rather than dropping it.
The output stops being pretty and starts being injective. **A decoder's contract
is not "name everything"; it is "never make two different things look the same".**

⚠️ **AND THE FAILURE IS FORWARD-DATED, WHICH IS WHY "our decoder covers the
spec" is not a defence.** The bits were unnamed because vulkane's constants
predate the AMD extension, not because anyone was careless. **Any decoder over a
vendor-extensible enum — Vulkan flags, `cpuid` leaves, ELF section types, HTTP
status classes, a wire vocabulary someone else owns — is guaranteed to meet a
value it does not know, on hardware or a peer that ships after it.** The
preservation is what makes the diagnostic survive that; naming the bits is only
what makes it comfortable today. **In this instance the preservation paid off on
the FIRST run.**

**Practice: when writing any value-to-text decoder, ask what the renderer does
with an input outside its table. If the answer is "drops it", two distinct inputs
already render identically and nothing in the program will ever say so.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **An incomplete decoder produces FALSE EQUALITY, not obviously-missing output** — distinct inputs collapse to identical renderings behind well-formed text. The remedy is not "name everything" but **preserve what you cannot name**. → [`an-incomplete-decoder-produces-false-equality`](#an-incomplete-decoder-produces-false-equality)

## grep-o-discards-the-context-that-dispositions-the-match

**`grep -o` PRINTS ONLY THE MATCHED FRAGMENT, SO IT STRIPS THE EVIDENCE THAT
SAYS WHETHER THE MATCH IS CODE, A COMMENT, OR PROSE.** A hit printed without
its line cannot be dispositioned at all — and it reads exactly like a live
declaration.

**Measured 2026-09-02.** Auditing manifest aliases:

```
grep -rhoE 'package *= *"fuel-[a-z-]+"' --include=Cargo.toml .
    ->  package = "fuel-core"          <- looks like a live rename
```

Reported as a live hazard to another lane. It is **not** live. Without `-o`:

```
./fuel/Cargo.toml:4:# package = "fuel-core" }`. This crate replaces that alias…
                   ^ THE LINE STARTS WITH `#`
```

The alias was deleted by Stage 1; the comment exists to record what was
**replaced**. `#` is not part of the match, so `-o` discarded it. The lane
nearly shipped a warning about a hazard that does not exist — which is worse
than no warning, because it spends the reader's attention and teaches them to
distrust the surrounding text.

**⚠️ TWO TRAPS, ONE FLAG, OPPOSITE DIRECTIONS — and the pairing is what makes
the FLAG the hazard rather than either misuse.** The same session:

- **`-o` did TOO MUCH** — stripped the `#`, manufacturing a live declaration
  out of a historical note.
- **`-o` did NOTHING AT ALL** — `grep -co` silently ignores `-o` and counts
  **lines**. Reported as 51; occurrences were 53; the question wanted 50
  distinct anchors. Three constructs, one command, right for none of them.

**Both produced clean, confident, wrong output.**

**⚠️ AND THIS REPO IS UNUSUALLY EXPOSED, WHICH IS THE PART THAT MAKES IT A RULE
RATHER THAN A GREP TIP.** Fuel *deliberately* preserves historical mentions —
[`docs-are-not-code-and-a-sweep-cannot-tell`](#docs-are-not-code-and-a-sweep-cannot-tell)
requires it, because sweeping a historical mention destroys the record it
exists to keep. So the corpus is **full of true statements about the past**,
and `-o` is precisely the flag that hides which tense a line is in. **A
convention that preserves history and a flag that strips context are
individually reasonable and jointly produce confident false positives.**

**PRACTICE: never disposition a match from `-o` output. Use `-n` and read the
line.** Reserve `-o` for counting a construct you have *already* dispositioned
— and even then, not with `-c`, which ignores it.

**Related but distinct:** the existing rule says the grep is mechanical and the
disposition is not, which tells a reader to classify each hit. **This one is
the operational half: with `-o` the classification is physically unreachable,
so the instruction cannot be followed even by someone trying.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`grep -o` DISCARDS THE CONTEXT THAT DISPOSITIONS THE MATCH** — a hit printed without its line cannot be classified as code, comment or prose, and reads as live. Two traps, one flag, opposite directions: `-o` stripped a leading `#` and manufactured a false hazard; `-co` silently ignored `-o` and counted lines. → [`grep-o-discards-the-context-that-dispositions-the-match`](#grep-o-discards-the-context-that-dispositions-the-match)

## a-fixer-can-reproduce-the-corruption-it-fixes

> **Index line (in CLAUDE.md):** **A FIXER WRITTEN IN A LANGUAGE WITH ESCAPE SEQUENCES CAN REPRODUCE THE EXACT CORRUPTION IT IS FIXING** — and the tool, the diff and the exit code all agree it worked. The compiler warns about the escape that is HARMLESS and is silent about the one that corrupts. Assert the PROPERTY, not the ACTION: *"the replacement ran"* is satisfied by a no-op replacement.

**⚠️ A DIAGNOSTIC CAN BE REAL, ACTIONABLE, AND ABOUT THE WRONG CHARACTER. A READER WHO FIXES WHAT THE WARNING NAMES HAS FIXED NOTHING AND NOW HAS A CLEAN RUN (2026-09-02).** Repairing a form feed in `docs/method-rules.md` — a Windows path whose `\f` had collapsed into `0x0C` — the fixer wrote its replacement as an ordinary Python literal. **In Python that literal IS the corruption**: `\f` is a valid escape for form feed, so the replacement string was **byte-identical to its target**. It would have rewritten the file unchanged and printed success.

**AND BOTH ESCAPE BEHAVIOURS OCCURRED IN THE SAME LITERAL, ON THE SAME LINE, WHICH IS WHAT MAKES THIS A SECTION RATHER THAN A CURIOSITY.** Python emitted `SyntaxWarning: invalid escape sequence '\P'` — **loud, and harmless**, because an invalid escape stays literal. The valid escape in the same string silently became a form feed and **was the entire bug, with no diagnostic at all**. ⚠️ **THE ASYMMETRY IS THE RULE: INVALID ESCAPES ARE LOUD AND HARMLESS; VALID ESCAPES ARE SILENT AND CORRUPTING. The warning you receive is evidence about the character that is fine.**

**⚠️ THERE ARE AT LEAST THREE ESCAPE LAYERS, NOT ONE — `JSON (tool call)` → `bash (heredoc)` → `Python (string literal)` — AND EACH CAN CONSUME OR TRANSFORM INDEPENDENTLY.** A `\\s` written into a tool call arrives at bash as `\s` before a quoted heredoc ever sees it; `\s` is invalid in Python, so it warns and survives, while `\f` is valid and does not. **Do not reason about "the escaping" as one step, and do not assume a quoted heredoc protects you — the mangling can happen before the heredoc exists.**

**THE REMEDY IS MECHANICAL AND IMMUNE TO ALL OF THEM, HOWEVER MANY THERE TURN OUT TO BE: BUILD EVERY SPECIAL CHARACTER WITH `chr()`.** `chr(92)` for a backslash, `chr(12)` for a form feed — nothing in any layer has an escape left to reinterpret. The same argument is why the conflict-marker guard BUILDS its patterns with `printf` rather than writing them literally: a literal would make the file flag itself.

**⚠️⚠️ AND THE GENERAL LESSON, WELL PAST CONTROL CHARACTERS: ASSERT THE PROPERTY, NOT THE ACTION.** *"The replacement ran"* is satisfied by a no-op replacement; *"the property now holds"* is not. **The only thing that caught this was a post-condition asserting ZERO CONTROL CHARACTERS REMAIN** — not that the edit had been applied, which it had been, vacuously. Same family as [`which-number-moves-if-it-became-a-no-op`](#which-number-moves-if-it-became-a-no-op), and **sharper, because here the no-op is produced BY THE FIX ITSELF rather than by a mechanism silently failing.** ⚠️⚠️ **AND THE APPLIER FOR THIS VERY SECTION PROVED THE POINT ON ITS WAY IN.** Its section scanner used `^## (.+)$`; the file is CRLF, and Python's `$` matches before the `
`, so every captured name kept a trailing carriage return. The assertion compared the new slug against itself-plus-``, disagreed, and **refused to write** — so a line ending, the same class of invisible character this section is about, broke the tool applying it. **It failed SAFE for exactly one reason: the assertion ran BEFORE the write.** Had the check been *"did the replace run"* it would have written a malformed file and reported success. **Order the post-condition ahead of the side effect and a wrong tool produces no output instead of wrong output.**

**⚠️ KNOWING ABOUT THE TRAP IS NOT THE DEFENCE, AND THE EVIDENCE IS UNUSUALLY CLEAN.** The architect hit the identical defect **while transcribing this finding, quoting it, an hour after reading it** — writing a census command into a memory file through a Python literal produced **six control characters in one line, in the memory about not leaving control characters around.** Their post-condition caught it and nothing else would have. **Two independent instances, one by a person who had just been told, which is why the remedy must be mechanical rather than attentional.**

**⚠️⚠️ AND THE STRONGER FORM, WHICH GENERALISES PAST ESCAPES ENTIRELY: PROXIMITY TO THE WARNING IS NOT PROTECTION, AND MAY BE THE REVERSE — HAVING WRITTEN IT, YOU STOP CHECKING WHETHER YOU ARE DOING IT.** Three instances in one day, none of them careless people: the portfolio PM **fixed a stale-ref hole in the portfolio `CLAUDE.md` and corrupted a path in the same edit**, then treated an exit code as a gate result **one line below their own comment warning against exactly that**; the architect **prescribed `ls-tree` as the cure for a hand-written file list and then ran it non-recursively**; and I put a second copy of the cell-boundary rule in `scripts/check-gaps-table.py` **fourteen lines under a docstring reading "Hence: one function, called everywhere"** — a docstring written *because* two people had already reimplemented it wrongly. **The warning was true, present, and in view in all three.** ⚠️ **So "I read it an hour ago" and "I wrote it myself" fail IDENTICALLY, and that pair is the whole argument: a defence that degrades with familiarity is not a defence. Put the check in the program.**

**⚠️⚠️ A FOURTH INSTANCE CLOSES THE SERIES, AND IT IS THE WORST ONE: I DID IT WHILE DRAFTING THIS SECTION.** A peer had just reported that `head -12` cut a `test result:` summary out of their run, so they counted the visible lines and published a test count that was one short. **Hours later, writing THIS paragraph, I ran two `gh pr merge` commands as `… | head -N ; echo ---` — piped through `head`, and separated with `;` rather than `&&`, so `$?` was `head`'s or `echo`'s and never the command's.** I discarded my own exit codes with a display choice, twice, and only discovered it when asked to contribute them as evidence. **The tally an architect had built on my two reports had to be corrected from five samples to three.** ⚠️ **So the series runs: I READ the warning · I WROTE the warning · I WAS DRAFTING THE WARNING. All four failed identically, which is as complete a refutation of attentional defence as the material allows — and note that the discipline did work, but only at the point where the claim had to be SOURCED, not at the point where it was made.**

**⚠️ AND THE PROSE FORM OF THE SAME THING, WHERE THE COUNT IS WHAT MAKES IT A CLAIM ABOUT AUTHORSHIP RATHER THAN ABOUT CARELESSNESS: A RULE STATED IN A DOCUMENT DOES NOT PROTECT THAT DOCUMENT.** `docs/design/facade-inversion-recipe.md` argues that *"references"* is three disagreeing constructs — lines, string occurrences, symbol references — and then violated that **three separate times in its own body**: a bare *"219 references"* in the purpose line, a table whose header never said whether its columns counted lines or symbols, and *"8 of 13 references"* opening the section that makes the argument. ⚠️ **All three were found by the author auditing his own deliverable, none by a reader, and each was written AFTER the rule it breaks.** **One violation is an oversight; three in one document, by the person making the argument, is evidence that stating a rule and applying it are separate acts that do not reinforce each other.** **The practical form: when a document argues for a discipline, audit the document AGAINST ITSELF as a distinct pass — not while writing it, because that is the pass that produced the violations.**

**⚠️ A NEGATIVE CONTROL WEARING A POSITIVE CONTROL'S JOB IS THE CHEAPEST VERSION OF THE SAME MISTAKE, AND IT PRODUCES A CONFIDENT ZERO.** While writing this entry I "controlled" a grep by searching for a section name I knew was **absent** — which proves nothing about whether the query works, and returned 0 exactly as the real query did. It was caught only because **both** numbers came back 0 and the coincidence looked wrong. **A control must be something you know is PRESENT; a check that cannot fail is not a check.** See [`uninformative-signals-both-directions`](#uninformative-signals-both-directions).

**⚠️⚠️ AND THE THIRD LEVEL, WHICH IS WHERE "ADD A POSITIVE CONTROL" STOPS BEING TERMINAL ADVICE: A REMEMBERED CONTROL IS EXACTLY AS UNVERIFIED AS THE THING IT WAS ADDED TO VERIFY — AND A STALE ONE FAILS IN THE DIRECTION THAT LOOKS LIKE SUCCESS.** Verifying that a merged branch had really been deleted, a lane picked a control branch *from memory* and it returned **"absent"** — which is **the same answer a successful deletion gives**. The branch had in fact been deleted hours earlier by an unrelated merge, so the control could not distinguish *the query works* from *the query is broken*: **the control and the defect it guarded against shared a failure mode.** Re-picked from live output, one branch came back PRESENT (proving discrimination) and the subject came back ABSENT (now evidence). ⚠️ **Note the layering — the merge, the branch deletion, and the control on the deletion were THREE levels of one check, and each needed its own verification.** **Practice: choose a control by LISTING WHAT EXISTS NOW (`git ls-remote --heads origin | head`), never from memory, and verify it in the same breath as the subject** — a control's freshness is part of the control. ⚠️⚠️ **AND A THIRD ROUTE TO THE SAME DEAD CONTROL, WHICH NEEDS NO MISTAKE BY ANYONE: THE SUBJECT CAN BE CHANGED BY SOMEONE ELSE, OR BY THE ACT OF RECORDING IT.** A branch serving as one lane's deletion-control was deleted by another lane's merge — correctly chosen, correctly verified, and dead an hour later through nobody's error. And a coverage metric over `02-layers` counted *crates named nowhere in the document*, reported **11**, and was fixed by LISTING those eleven — **which named them, so re-running the query returns 0.** ⚠️ **A later reader takes that 0 for the gap having closed: `every crate genuinely PLACED` and `the eleven merely LISTED as unplaced` both render as 0.** It was replaced with **PLACED (20 of 40)**, which is stable under the document's own edits. **PRACTICE: ask of any metric whether RECORDING its result changes the population it counts — a measurement that its own write-up falsifies is not a measurement, and it fails toward the reassuring value.**

**⚠️ THE PROPERTY SHARED BY THE FAILURES IN THIS SECTION, STATED SO IT CAN BE TESTED RATHER THAN FELT: TWO DISTINCT STATES PRODUCE ONE IDENTICAL, WELL-FORMED RENDERING.** *Fixed* and *unfixed* both give a clean diff and a success message. *Deleted* and *never-existed* both give `absent`. **A correctly-written `\f` and a corrupted one both give SILENCE** — which is the honest locus, and worth stating precisely: the `\P` warning is *not* the false equality, it is a separate defect (a true diagnostic about the wrong subject). **The collapse is on the `\f` axis, where the compiler says nothing either way.** That is [`an-incomplete-decoder-produces-false-equality`](#an-incomplete-decoder-produces-false-equality) arriving through a WRITE rather than through a decoder.

⚠️⚠️ **AND THE DISCIPLINE THIS PARAGRAPH COST, WHICH IS WORTH MORE THAN THE PARAGRAPH: NAMING A SHARED PROPERTY IS NOT ENOUGH — TEST IT AGAINST EVERY CASE ON BOTH SIDES.** I first linked this section elsewhere on *"an instrument silently removing the thing that answered the question"*, withdrew it because it covered a minority of my cases, then asserted a **replacement** relationship that was wrong in exactly the same way — and it was disproved by someone running my own test on my own claim, case by case. **Two states collapsing to one rendering is a testable predicate; *these feel similar* is not, and a link is a claim about a relationship.** ⚠️ **The mechanism of the first failure and the mechanism of its correction were identical, which is the whole reason a correction needs the verification you would demand of an original claim.**

**⚠️ THE REUSABLE HALF, AND THE PART THAT IS NOT OBVIOUS: A TAXONOMY CLAIM HAS A MECHANICAL TEST.** *"X is a special case of Y"*, *"these two rules are parent and child"*, *"this is an instance of that"** — these are cheap to assert, feel like insight, and read as structural facts about a corpus, so nobody runs anything. **They are checkable in three lines: take the PARENT'S LITERAL PREDICATE, run it against EACH of the child's instances, and count.** Here the parent's predicate was *two distinct inputs → one identical well-formed rendering*; the candidate child's three instances scored **1 of 3**, and the two failures were a different rule (an unnamed construct in a count) and a third thing again (truncation — information LOST, not two things MERGED). **Related, not nested.** ⚠️ **A claim that two things are the same phenomenon is a claim about EVERY member on both sides, and the cost of testing it is proportional to the number of instances, not to the confidence of the assertion.**

**AND THE SCOPE HALF, because the survivor is what started this: A REPAIR NEEDS A POPULATION ENUMERATION EXACTLY LIKE A COUNT DOES.** The earlier sweep fixed **seven** control characters in **two** files and was reported done — the files the author happened to be holding, with no enumeration of where else the class occurred. **One survivor sat in the rules corpus for a week**, found by someone tripping over it while writing an unrelated entry; a tree-wide census, positive-controlled with a planted form feed, then confirmed it was the last. **Enumerate the population for REPAIRS, not just for measurements — a partially-completed sweep reads exactly like a finished one.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **A FIXER WRITTEN IN A LANGUAGE WITH ESCAPE SEQUENCES CAN REPRODUCE THE EXACT CORRUPTION IT IS FIXING** — the tool, the diff and the exit code all agree it worked. The compiler warns about the escape that is HARMLESS and is silent about the one that corrupts; there are at least THREE escape layers. Build special characters with `chr()`, and assert the PROPERTY, not the ACTION. → [`a-fixer-can-reproduce-the-corruption-it-fixes`](#a-fixer-can-reproduce-the-corruption-it-fixes)

## an-insertion-before-an-item-steals-the-attribute-above-it

**Inserting a test immediately before `fn target` splices your block BETWEEN that function's `#[test]` and the function it modifies. The attribute binds to YOUR item, so yours registers twice and the pre-existing test silently STOPS RUNNING.** Measured 2026-09-02 in `fuel-cpu-backend/src/byte_kernels.rs`: a boundary test anchored on `    fn cast_f32_f8e4m3_round_trip_exact_for_representable() {` — a unique, exact, verified anchor — landed under that function's `#[test]`, and `cargo test` reported **`2 passed`** for a filter matching **one** name.

**Both directions are silent and both look like good news.** The suite stays green; the total stays plausible (my duplicate exactly replaced the test I had disabled, so the count did not move); and the disabled test is not *failing*, it is *absent*, which no result line mentions. **CLAUDE.md already warns never to splice a seed between an attribute and its item — but it says so about `#[cfg]` in born-red sabotage, and this arrived through ordinary test authoring.** The stolen attribute can be `#[test]`, `#[cfg]`, `#[ignore]`, `#[should_panic]` or a derive; the rule is about INSERTION POINTS, not about sabotage.

⚠️ **THE COMPILER REPORTED IT AND MY INSTRUMENT ATE THE MESSAGE.** `rustc` emitted `function cast_f32_f8e4m3_round_trip_exact_for_representable is never used` — the exact finding, unprompted. I was running `cargo test -- --list 2>&1 | grep -c '<name>'`, so **stderr was merged into the stream I was counting and the warning lines were tallied as if they were list entries**, returning `2` for a function that was registered `0` times. **A count over a stream carrying diagnostics is not a count of the thing you asked about** — see [`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it) and [`grep-o-discards-the-context-that-dispositions-the-match`](#grep-o-discards-the-context-that-dispositions-the-match). Reading the two matching LINES instead of counting them ended the investigation immediately.

✅ **THE MECHANICAL DETECTOR IS ONE LINE AND NEEDS NO SUSPICION: `--list` LISTED MUST EQUAL UNIQUE.** A duplicate fully-qualified path cannot come from one `fn`, so `cargo test -p <crate> --lib -- --list 2>/dev/null | grep ': test$'` piped to `sort -u | wc -l` must match the raw count — **and drop stderr for this one, precisely because the diagnostics you want to READ are the ones that corrupt a COUNT.** Pair it with naming the neighbour: after inserting near an existing test, assert that test still appears in the listing. **Anchor on the start of the item's attributes or doc comment, never on its `fn` line.**

⚠️ **AND THE SECOND ANCHOR FAILURE FROM THE SAME HOUR, BECAUSE IT INVERTED A SABOTAGE RESULT: `cargo fmt` RUNS BETWEEN AUTHORING A FIXTURE AND SABOTAGING IT, AND IT INVALIDATES THE TEXT ANCHOR YOU JUST WROTE.** To prove a vacuity guard fired I trimmed the fixture table by splitting on `'(449.0'` — a token I had typed myself minutes earlier. `rustfmt` had since exploded the longer tuples across lines, so the token was gone; **Python's `str.split` on an absent separator returns the whole string unchanged, so the "trim" was a no-op, nothing recompiled, and the test passed.** I very nearly recorded that pass as *the guard does not fire*. **A sabotage that never applied reports absence-of-sensitivity as presence** ([`a-sabotage-that-never-applied`](#a-sabotage-that-never-applied)); the new specifics are that **a formatter is a mutation between your write and your read**, and that **the split/replace family fails SILENTLY and IDENTITY-PRESERVINGLY.** **Assert the anchor is present and that the edit SHRANK the text, before running anything — and require the `Compiling` line, which is what would have caught it here.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ AN INSERTION ANCHORED ON A `fn` LINE SPLICES BETWEEN THE NEIGHBOURING `#[test]` AND ITS FUNCTION — YOURS REGISTERS TWICE AND THE PRE-EXISTING TEST SILENTLY STOPS RUNNING.** The anchor was unique, exact and verified; the splice still landed under the previous item’s attribute. **The total does not move**, because the duplicate exactly replaces what you disabled — a conserved total hiding a swapped distribution. CLAUDE.md warns of this for `#[cfg]` seeds in born-red sabotage; it arrives just as easily through ordinary test authoring, and the stolen attribute can be `#[test]`, `#[ignore]`, `#[should_panic]` or a derive. ⚠️ `rustc` reports it as `function … is never used`, but a `grep -c` over `2>&1` **counts that warning as a list entry**. **Detector: `--list` LISTED must equal UNIQUE** — a duplicate fully-qualified path cannot come from one `fn`. **Anchor on the start of the item’s attributes or doc comment, never on its `fn` line.** → [`an-insertion-before-an-item-steals-the-attribute-above-it`](#an-insertion-before-an-item-steals-the-attribute-above-it)

## a-confession-is-the-claim-nobody-audits

> **Index line (in CLAUDE.md):** **AN OVERSHOOTING CORRECTION CAN TAKE THE FORM OF INVENTING A DEFECT CLASS — the most credible possible disguise for an error, because it arrives with the author's name attached asking to make it PERMANENT.** A confession has no apparent motive to overstate, so it is the one claim nobody audits; **overstating against yourself reads as rigour and is exactly as wrong.** Verify a correction you ACCEPT with the same measurement you would demand of the claim it corrects.

**2026-09-02, Fuel 3 and the architect, and it came within one PR of entering this file with both names on it.**

Fuel 3 reported that `fuel-hardware/src/transfer_cost.rs` carries two clippy errors in its default config while CI stays green, and offered a mechanism **explicitly labelled as a hypothesis**: `cargo clippy --workspace` unifies features across the selected graph, so the `#[cfg]`'d arm goes live and the lints vanish.

The architect refuted it: *"`fuel-core` and `fuel-dispatch` both enable `fuel-hardware/cuda`, but only under their own non-default `cuda` feature — so your unification theory would not have worked anyway."* **The refutation was accepted.** They were also right that GAP-267 already tracks the file, right that `KNOWN_FAILING`/`unexpected_pass` fences it, and right that the `--no-deps` comment is about attribution — **all three verified at head.**

**Then, to explain an error that had not happened, a defect class was invented:**

> *"I lifted a TRUE, ADJACENT, DOCUMENTED mechanism from eighteen lines away and attached it to the wrong subject — and that is more durable than invention, because grepping for the mechanism CONFIRMS it. My positive control would have passed while I was wrong about the subject."*

**It is a good rule. It is also about an error nobody made.** The architect called it *"the sharpest thing anyone produced tonight"*, asked for it in this file, and **wrote it into their own durable notes within minutes.**

**THEN IT WAS MEASURED:**

```text
cargo tree -e features --workspace <CI excludes>
  fuel-hardware feature "vulkan"   PRESENT      <- the arm that rescues it
  fuel-hardware feature "cuda"     ABSENT       <- the only one either party checked

cargo clippy -p fuel-hardware --no-deps                    -> 3 errors, exit 101
cargo clippy -p fuel-hardware --no-deps --features vulkan  -> exit 0, CLEAN
```

`measure_h2d_d2h` has **two** gated arms. **The original mechanism was correct; the refutation checked the wrong feature.** The architect verified this independently and deleted the memory file. Their own account of the miss: *"My grep printed FOUR relevant lines — two `cuda`, two `vulkan`. I read the two `cuda` lines and concluded about the crate. And I had personally quoted the `#[cfg(feature = "vulkan")]` arm from the source BEFORE asserting the match has only `_ => return None`"* — a mixed list read as verified, which is [`a-true-half-vouches-for-the-false-half`](#a-true-half-vouches-for-the-false-half) occurring *inside a correction*.

**WHY THIS IS ITS OWN CLASS AND NOT A VARIANT OF "VERIFY THE PREMISE".** The neighbouring rules describe corrections that are *wrong*. **This one describes a correction that was ACCEPTED and then ELABORATED — and the elaboration is what made it dangerous.** A plain wrong claim invites checking. **A self-critical one does not: the author has no apparent motive to overstate, so a confession reads as already-audited.** The memory note `a-precise-citation-spends-skepticism` says precision consumes a reader's skepticism budget; **penitence spends it the same way, and from a direction nobody guards.**

⚠️ **THE ESCALATION IS THE PART TO FEAR, BECAUSE IT IS FAST AND STRUCTURAL.** A loose hypothesis became an accepted refutation, became a fabricated defect class, became *"write it into `method-rules.md`"*, became a peer's durable memory file — **in under fifteen minutes, with every step performed by someone acting correctly on the previous one.** Nothing in that chain re-derives the original measurement, and the artifact at the end of it would have been permanent.

⚠️ **NOTE WHAT DID NOT CATCH IT.** Two independent parties said the unification read was right — Fuel 2 by measurement, then the architect by re-verification — **and agreement was not what settled it.** The opposite position had already been agreed to, on the same evidence. **The only thing that discriminated was running `cargo tree` and `cargo clippy --features vulkan` directly.** Per [`evidence-that-is-not-independent`](#evidence-that-is-not-independent), peers agreeing is a weak instrument; here it was weaker than the wrong conclusion deference had already produced.

**PRACTICE, three parts, and the third is the one that is easy to skip:**

1. **A correction you ACCEPT is a claim you now hold.** Measure it to the standard you would demand of the thing it corrected. The memory note `a-correction-that-contradicts-your-measurement` covers REFUSING a correction that contradicts something you measured; **this covers ACCEPTING one where you had not measured — the commoner case, since a labelled hypothesis is exactly the place you have no measurement to defend.**
2. **Do not build theory on an accepted correction until you have measured it.** The theory inherits the correction's truth value and then disguises it, because a mechanism argued at length reads as a mechanism verified.
3. **When you SUPPLY a correction, name your own half if it turns out wrong.** The architect required this of themselves here: *"a rule about unaudited confessions that omits the party who failed to audit is missing its own mechanism."* **A confession that names only the confessor is the same defect one level up** — it audits the cheapest party and leaves the correction's author unexamined. **Approval by a coordinator is what would have made this permanent, so the approval is part of the mechanism, not context around it.**

**AND THE DISPOSITION THAT SURVIVED ALL OF IT, because right-answer-false-reason is the combination nobody re-checks:** CI is green on that crate for a **measured** reason, GAP-267 tracks it, and the fence is real. **The architect's conclusion was right and the mechanism they gave for it was wrong.** Without the measurement the row would have carried a fence with no explanation — precisely the pair that never gets re-opened, because the answer is correct and only the reason is missing.

⚠️⚠️ **AND A GATE HOLE FOUND WHILE WRITING THIS SECTION, RECORDED WITH ITS MEASUREMENT SO WHOEVER CLOSES IT NEED NOT REDISCOVER IT.** Drafting the citations above, three intended references turned out to be **MEMORY files with no `method-rules.md` section at all** — `a-precise-citation-spends-skepticism`, `a-correction-that-contradicts-your-measurement`, `enumerate-the-population-not-the-strings` (`grep -c '^## <name>$'` → **0** for each; controls `a-true-half-vouches-for-the-false-half`, `evidence-that-is-not-independent`, `a-defence-can-outlive-its-defect` → **1** each, so the query finds sections where they exist). **Two of the three had already been written as bracketed links in the first draft.**

**`fuel-ir/tests/method_rules_index_join.rs` would NOT have caught them.** Its arm B validates that every anchor in **CLAUDE.md** resolves to a real section; **nothing validates anchors that live INSIDE `method-rules.md` itself.** The invariant is bidirectional and **the gate guards one direction — the one that accumulates cross-references more slowly.** Every `See [`x`](#x)` between sections here is currently unchecked, and a section renamed or a citation typed from memory of a *memory file* lands as a dead in-page anchor that renders as ordinary text.

**Recorded as an observation rather than fixed, deliberately:** the arm is small but adding it at the end of a long night, in a PR whose subject is a fabricated defect class, is how the next entry in this section gets written. **The measurement is the deliverable; the arm is somebody's next cheap win.** ⚠️ **And note the detection route, because it is the transferable half: this was found by OBEYING a warning about dangling citations, not by testing the gate.** The gate was green throughout and is still green — **a hole in coverage is invisible to the thing that has the hole.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ AN OVERSHOOTING CORRECTION CAN TAKE THE FORM OF INVENTING A DEFECT CLASS — the most credible possible disguise for an error, because it arrives with the author's name attached asking to make it PERMANENT.** A confession has no apparent motive to overstate, so it is the one claim nobody audits; **overstating against yourself reads as rigour and is exactly as wrong.** A hypothesis was labelled as such, "refuted" on the wrong feature (`cuda` checked, `vulkan` was the live arm), accepted, and then explained by a fabricated class that a coordinator praised, approved for this file, and wrote into durable notes — **in under fifteen minutes, every step correct given the last.** Measurement settled it, not two peers agreeing. **Verify a correction you ACCEPT with the same measurement you would demand of the claim it corrects; and when you SUPPLY one, name your own half.** → [`a-confession-is-the-claim-nobody-audits`](#a-confession-is-the-claim-nobody-audits)

## an-allowlist-entry-that-reddens-when-its-reason-dissolves

> **Index line (in CLAUDE.md):** **AN EXCLUSION, SUPPRESSION OR KNOWN-FAILING ENTRY MUST CARRY A DETECTOR FOR ITS OWN CAUSE DISSOLVING** — otherwise it outlives the defect it records and is indistinguishable from one that still applies. It fires on the **FIX**, not on a date: an expiry needs an event that will occur, and *"the reason stopped being true"* is exactly that event, with no false alarms and no silence when the world moves. Two polarities: an entry that reddens when a listed thing starts PASSING, and one that reddens when the gap it cites CLOSES.

**Reached independently by two parties on 2026-09-02, on unrelated work, with no contact — which is why it is a construct and not a coincidence.**

```text
CI, rust-ci.yml:776,829   KNOWN_FAILING="fuel-hardware"
                          clippy PASSES && crate is listed -> unexpected_pass, RED
                          "PASSES but is listed KNOWN_FAILING -- delete the entry (GAP-267)"

#72, kiss_structure_key_byte_match.rs
                          the_i4_exclusion_still_has_its_reason
                          asserts "i4" is still in RECOGNIZED_UNSUPPORTED_DTYPE_TOKENS
                          -> closing GAP-097 turns the exclusion RED
```

**Same construct, opposite polarity.** CI's fires when a listed *failure* stops failing. The corpus exclusion fires when the *reason for excluding* stops being true. **Neither can quietly outlive its cause.**

**AND THE SHAPE ALREADY RECURS HERE, which makes this a missing NAME rather than a new idea.** Measured: `expiring-decline` appears **4 times** and `allow(dead_code)` **5 times** in `docs/gaps.md`, and **neither appears in this file at all** *(control: `staleness-by-workaround` appears twice here, so the query finds sections where they exist)*. **Nine instances in the registry, zero named sections — the construct was being re-argued from scratch each time.**

**WHY A DEADLINE IS THE WEAKER MECHANISM.** CLAUDE.md already requires that an expiry fire on a checkpoint that WILL occur rather than an event that MAY NOT. A date satisfies that and is still weak: **it fires whether or not anything changed, so it trains people to push it forward.** *"The reason dissolved"* is guaranteed-detectable **and only fires when there is something to do.**

⚠️ **THE FAILURE IT PREVENTS IS INVISIBLE BY CONSTRUCTION, WHICH IS THE WHOLE ARGUMENT.** A stale exclusion costs exactly one vector of coverage; a stale `KNOWN_FAILING` entry costs exactly one crate's lint gate — **and both are SILENT: the suite is green, the count is correct, and the output is byte-identical to the healthy case.** Nothing distinguishes *"still needed"* from *"was needed in August"* except re-deriving it, and **nobody re-derives an entry that is not complaining.**

**PRACTICE.** When adding to any allowlist, exclusion list, `KNOWN_FAILING` set or `#[allow]`:

1. **State the CAUSE, not the symptom.** *"Not constructed"* is a symptom. *"Fuel has no `DType::I4`, so no token for this cell can exist"* is a cause — **it names something checkable that can later become false.** A symptom cannot be a detector's subject.
2. **Write the detector in the same change.** One assertion on the cause. **If the cause is not expressible as an assertion, that is itself the finding:** you have recorded a feeling, not a reason.
3. **Home the residue at an owned row.** The detector says *when*; the row says *who* and *what next*. `the_i4_exclusion_still_has_its_reason` points at GAP-097; CI's message points at GAP-267. **A site comment is read only by someone already standing there, and the entry exists precisely for the case where nobody comes.**

**Related:** [`a-defence-can-outlive-its-defect`](#a-defence-can-outlive-its-defect) is the mirror image — there a control becomes ACTIVELY HARMFUL once its replacement lands, where here it merely goes INERT while still looking live. **Both are cured by making the entry able to fail.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ AN EXCLUSION, SUPPRESSION OR KNOWN-FAILING ENTRY MUST CARRY A DETECTOR FOR ITS OWN CAUSE DISSOLVING** — otherwise it outlives the defect it records and is **indistinguishable from one that still applies**, silently, with the suite green and the count correct. It fires on the **FIX**, not a date: a deadline fires whether or not anything changed and trains people to push it forward. Two polarities, reached independently the same night: an entry that reddens when a listed thing starts PASSING (CI's `unexpected_pass`), and one that reddens when the gap it cites CLOSES (`the_i4_exclusion_still_has_its_reason`). **State the CAUSE not the symptom, write the detector in the same change, and home the residue at an owned row.** → [`an-allowlist-entry-that-reddens-when-its-reason-dissolves`](#an-allowlist-entry-that-reddens-when-its-reason-dissolves)

## a-new-file-that-git-does-not-list-is-a-finding

**An ignore rule can swallow a file you just wrote, and the failure is LOCAL-GREEN / CI-RED with the error pointing somewhere else.**

Measured 2026-09-09 in this repo. A test fixture written to `fuel-ir/tests/data/gap302_known_absent.txt` produced **no `git status` entry at all**:

```
git check-ignore -v fuel-ir/tests/data/x.txt  ->  .gitignore:4:data/   IGNORED
control: fuel-ir/tests/doc_block_scope.rs     ->  exit 1, NOT ignored
```

**A bare `data/` matches a directory of that name at ANY DEPTH.** In this `.gitignore` it sits among `debug/`, `dist/` and `target/` — build-output rules — so it reads as a Cargo artifact rule and is nowhere near anything about test fixtures. Nobody adding a fixture directory would think to look.

⚠️ **THE DIAGNOSTIC POINTS AT THE WRONG PLACE.** `include_str!("data/x.txt")` compiles locally, because the file is on disk. CI checks out a tree where it was never committed and fails **on the `include_str!` line**, so the error names the include and not the ignore. Every local gate is green and says nothing.

**PRACTICE: after writing any new file, read `git status` before `git add`, and treat a MISSING untracked entry as a FINDING rather than as nothing to do.** A file you just created failing to appear is the signal; there is no other one.

**And prefer moving the file to fighting the ignore.** A non-`.rs` file directly in `tests/` is not a cargo test target (`tests/*.rs` are), so no subdirectory is needed. A `.gitignore` negation would encode one crate's fixture layout in a repo-wide file, which is the wrong home for it.

The class is *things this working copy has that a clean checkout does not*, and an uncommitted file is the cheapest possible instance. ⚠️ **NOTE ON THE CITATION THAT IS NOT HERE:** this section was handed over citing `local-green-is-not-evidence-about-ci`, which is a **personal-memory file, not a section of this document** — the anchor would have dangled and `arm_e` would have caught it. **A slug that exists in one ledger reads as a citable anchor in the other, and only one of the two ledgers has a gate.**

---

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A NEW FILE THAT `git status` DOES NOT LIST IS A FINDING, NOT A NON-EVENT** — a bare `data/` in `.gitignore` matches at ANY DEPTH and sits among the build-output rules, so a test fixture written under `tests/data/` is swallowed silently. `include_str!` then compiles LOCALLY and fails in CI **on the include line**, naming the wrong thing. Read `git status` before `git add`, and move the file rather than fight the ignore. → [`a-new-file-that-git-does-not-list-is-a-finding`](#a-new-file-that-git-does-not-list-is-a-finding)

## a-sabotage-can-redden-the-wrong-assert

**The one-sabotage-per-arm rule is not enough when a single arm has an ORDERED PAIR of assertions. An earlier assert absorbs the sabotage and the arm's actual claim is never exercised — while the arm goes RED, which is the outcome nobody questions.**

Measured 2026-09-09 on `kiss_ops_619_divergence.rs`. Each arm PINS the encoder's bytes (`assert_eq!`) and then asserts DIVERGENCE from a spec-derived vector (`assert_ne!`).

```
sabotage 1  make the encoder conformant
            -> 3 arms RED, EVERY ONE on the PIN ("row moved")
            -> the assert_ne! carrying the file's claim NEVER RAN

sabotage 2  realign the pin to the conformant bytes, so assert_eq! passes
            -> "assertion `left != right` failed: gather now MATCHES 6.19-0027"
```

⚠️ **A RED RESULT IS THE LEAST-QUESTIONED OUTCOME IN A BORN-RED DISCIPLINE.** The first sabotage went red, named the right file, and proved only that the pin discriminates. It is [`vacuous-oracle-four-routes`](#vacuous-oracle-four-routes) route 4 — an earlier guard answering first — **inside a single test rather than across tests, which is where nobody looks for it.**

**PRACTICE: for an arm with more than one assertion, ask WHICH assertion the sabotage reaches. If an earlier one absorbs it, run a second sabotage that SATISFIES the earlier assertion so the later one is exercised. Then write the ordering into the file** — otherwise the next reader runs one sabotage, sees red, and stops exactly where the first author nearly did.

**AND SABOTAGE THE INSTRUMENT SEPARATELY FROM THE SUBJECT — they fail different assertions.** The mirror case, same day, same increment: an arm asserting *no population member appears in this file* was sabotaged by making the tokenizer return an empty vector — a BLIND READER. **The offender assertion passed VACUOUSLY (0 found), and only the positive control caught it** (*"the tokenizer cannot see a token known to be in this file, so the emptiness above is a reader defect, not a finding"*). **Subject-sabotage proves the assertion discriminates; instrument-sabotage is what proves the control earns its place.** One of the two alone ships a gate that can go permanently green on a broken reader.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A SABOTAGE CAN REDDEN THE WRONG ASSERT — one sabotage per ARM is not enough when an arm has an ORDERED PAIR of assertions.** An earlier `assert_eq!` pin absorbs the perturbation and the later `assert_ne!` carrying the file's actual claim never runs, **while the arm goes RED — the least-questioned outcome in a born-red discipline.** Run a second sabotage that SATISFIES the earlier assertion. And sabotage the INSTRUMENT separately from the SUBJECT: subject-sabotage proves the assertion discriminates, instrument-sabotage proves the CONTROL earns its place. → [`a-sabotage-can-redden-the-wrong-assert`](#a-sabotage-can-redden-the-wrong-assert)

## a-formatted-repo-splits-multi-token-constructs

**⚠️ IN A FORMATTED REPO, A LINE-ORIENTED GREP FOR A MULTI-TOKEN CONSTRUCT IS BLIND BY CONSTRUCTION — NOT BY BAD LUCK.** `rustfmt` (and any formatter with a line width) splits every construct longer than the width across lines: `assert_eq!(` lands on one line and its argument on the next; a long string wraps; a `Type::Variant` chain breaks at the `::`. A `grep`/`git grep` pattern that requires two tokens ON THE SAME LINE therefore misses exactly the constructs long enough to be split — and the longer the construct, the more certain the split. The failure is a property of the repository's formatting, so it RECURS; it is not a one-off.

**FIVE INSTANCES IN ONE SESSION (2026-09-09). Four produced, nearly produced, or held up a wrong action; the fifth returned the RIGHT answer for the wrong reason, which is the most dangerous of the five.**

1. **`not baked` split across a doc-comment wrap** — nearly sent a wrong correction to a sibling project.
2. **A hard-wrapped sentence in a spec** — a lane nearly filed a citation as ROTTED when it had only wrapped.
3. **`Fmax`/`Fmin` at char ~1350 of a 1,468-char line** — a supersession marker took four review rounds because the falsifying clause sat past every reader's and every `grep`'s effective horizon.
4. **`assert_eq!(` with its argument on the next line** — `git grep 'assert_eq!(FKC_SUPPORTED_VERSIONS'` returned **1**; the multiline-aware count returned **2**. The missed assertion was a live regression guard, and the undercount **held a fully-resolved `gaps.md` row (GAP-292) out of a closure batch** as a suspected dropped born-red. Reading the two test BODIES — not grepping the symbol — settled it.
5. **A single-line query for GAP-281's production relation-guard returned `0`.** The guard is `head_count * head_dim == width` — exactly the multi-token construct `rustfmt` wraps — so the query could not have seen one had it existed. Here the true answer WAS zero (the asserts had long since been converted to typed declines), so the blind instrument was right — **by luck, not by sight.** **NAMING THE HAND: this was the architect's own query, run roughly twenty minutes after ruling on this very hazard, while sizing GAP-281.** A right-by-luck reading from a blind method is worse than a wrong one: it banks confidence in a method that will be wrong the next time the answer is not zero.

Cases 1–3 are the mechanism in **prose** (an author's wrap width); cases 4–5 are its **code** twin (the formatter's). One rule covers both.

**THE RULE:** any `grep`/`git grep` whose pattern spans **more than one token** must be **multiline-aware** (`git grep -U`/`--multiline`, `rg -U`, or a `python` read over the whole file), **or be replaced by reading the construct.** A single-line pattern is safe only for a single token.

**THE DISCRIMINATOR IS FREE — anchor on ONE token plus a POSITION, not on the multi-token shape.** `git grep -c 'assert_eq!($'` (the macro at end-of-line) counts the very assertions the same-line pattern structurally cannot see; a count that rises against your line-pattern's is the tell. `wc -l` on a file you know is large is the cheapest control for the long-line case, and the file's own byte length against your reader's horizon is the control for case 3.

**AND BECAUSE IT IS FREE, IT HAS TO BE THE *DEFAULT* REACH — not a remedy applied after being burned.** Instance 5 is the proof: the person who wrote this rule reached for the blind single-line form twenty minutes later. A discipline invoked only once you already suspect a split will never fire, because the split's whole signature is that nothing looks wrong — the count is plausible, the command exits 0, no error appears. The multiline-aware form has to be the one you type first, before there is any reason for suspicion.

**THE COMPOUNDING HAZARD (why this one is expensive):** a blind grep returns a **plausible number**, not an error, and a plausible undercount that **agrees with an honest caution** — *"I didn't deep-verify this"* — reads as corroboration when the caution was correct only by luck. GAP-292 (instance 4): a lane flagged uncertainty, a grep returned the undercount, the two matched, and the row was held out of the closure batch. Had the deep read gone the other way, the wrong number would have shipped with a lane's hedge apparently backing it. **A defective instrument that happens to match a hedge is not confirmation of the hedge — read the construct.** This is the false-corroboration failure recorded in [`evidence-that-is-not-independent`](#evidence-that-is-not-independent), reached here from the INSTRUMENT side rather than the two-artifacts side, and it is the same defect the memory rules name as *agreement-is-not-corroboration*, *long-lines-defeat-line-oriented-instruments*, and *construct-invisible-in-the-number*.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ IN A FORMATTED REPO A SINGLE-LINE GREP FOR A MULTI-TOKEN CONSTRUCT IS BLIND BY CONSTRUCTION** — `rustfmt` wraps any construct past the line width, so `assert_eq!(` and its argument, a long string, or a `Type::Variant` chain land on different lines and a same-line pattern misses exactly the ones long enough to split. Five instances in one session, including a `git grep 'assert_eq!(FKC_…'` returning **1** where the multiline count returned **2** — holding a fully-resolved GAP-292 row out of a closure batch — and the rule's own author reaching for the blind form twenty minutes after ruling on it. The discriminator is FREE and must be the DEFAULT reach: anchor on ONE token plus a POSITION (`git grep -c 'assert_eq!($'`), go multiline-aware (`-U` / `rg -U`), or read the construct. A blind undercount that happens to match an honest hedge reads as corroboration and is not. → [`a-formatted-repo-splits-multi-token-constructs`](#a-formatted-repo-splits-multi-token-constructs)

## an-empty-collection-is-not-a-null-result

**⚠️ FIVE INSTRUMENTS ANSWER "DOES THIS REPO HAVE BRANCH PROTECTION?" AND THREE OF THEM ARE WRONG — TWO BY RETURNING AN EMPTY LIST INSTEAD OF AN ERROR, AND ONE BY RETURNING A 404 THAT AN UNPROTECTED REPO RETURNS TOO (2026-09-09, `ciresnave/fuel`, one token; A–D measured by Fuel 2, E measured by the architect).**

```text
A. GraphQL branchProtectionRules(first:5)         -> 0 nodes     reads as NO PROTECTION   WRONG
B. REST    /repos/<nwo>/rulesets                  -> 0 entries   reads as NO PROTECTION   WRONG
C. REST    /repos/<nwo>/branches/main .protected  -> true        PROTECTED                RIGHT
D. GraphQL isRequired(pullRequestNumber: N)       -> 5 required  PROTECTED                RIGHT
E. REST    /repos/<nwo>/branches/main/protection  -> HTTP 404    reads as NO PROTECTION   WRONG
```

**C and D are correct.** Protection demonstrably exists: D names the five required checks (`Check (ubuntu-latest)`, `Clippy`, `Rustfmt`, `Test Suite (ubuntu-latest)`, `trufflehog`) — **out of fourteen contexts on the PR, so nine of the fourteen checks cannot block a merge**, which is the operational reason anyone runs this query at all.

**A and B are permission-blind, and they do not say so.** They return HTTP 200 with an empty collection. **A `[]` and a `403` are different facts and only one of them is reported.**

**NEGATIVE CONTROL, so this is not a story about one repo:** `ciresnave/synapse` `.protected` -> **false** against fuel's **true**. C discriminates; it is not returning `true` for everything.

### ⚠️ E fails the same way, and the status code is what disguises it

**Measured first-hand for this section, both arms, same token, same minute:**

```text
GET /repos/ciresnave/fuel/branches/main/protection      -> 404   repo IS protected     (C: true)
GET /repos/ciresnave/synapse/branches/main/protection   -> 404   repo is NOT protected (C: false)
```

**The two responses are BYTE-IDENTICAL** — 138 bytes of stdout and 25 of stderr each, `diff` clean on both streams:

```text
{"message":"Not Found","documentation_url":"https://docs.github.com/rest/branches/branch-protection#get-branch-protection","status":"404"}
```

**⚠️ IT IS TEMPTING TO SAY A 404 AT LEAST ANNOUNCES THAT SOMETHING WENT WRONG. IT DOES NOT, AND THE MEASUREMENT ABOVE IS WHY.** A genuinely unprotected branch returns that same 404 — so on this endpoint the 404 is not an error wearing a null, it is **the correct answer for one of the two states, returned identically for the other.** The status code separates E from A and B while carrying no information about which state you are in. *(This paragraph replaces the reading the draft of this section carried, which was that the 404 was the better-behaved of the three. Without the unprotected arm there was nothing to contradict it — **an absence claim about an API needs a repo that genuinely has the absence.**)*

**So E belongs WITH A and B, not against them. Three instruments, one direction of failure: all three report NONE, and NONE is the answer that prompts a WRITE.**

⚠️ **THAT DIRECTION IS THE WHOLE POINT.** **Writing protection onto an already-protected repo is a different act from configuring an unprotected one, and A, B and E cannot tell you which one you are about to do.** The instruments fail in the direction that causes the mutation.

### ⚠️ D's blind spot is CORRELATED with the condition D detects

*(the portfolio PM's, and it is sharper than the probe that prompted it)*

**D needs an OPEN PULL REQUEST**, because `isRequired` is scoped to one. So it **cannot run on a quiet repo — and a quiet repo is exactly where missing protection goes unnoticed.** Four of thirteen repos in the portfolio census had no open PR and could not be measured with D at all.

**An instrument whose UNAVAILABILITY correlates with the condition it would detect contributes nothing to the cases you most need it for**, while looking like a second opinion on the cases you do not.

### Practice

- **For "is this branch protected", use C.** It is readable without admin and it discriminates.
- **Never use E — `branches/<b>/protection` — to answer it.** Without admin it returns the same 404 for a protected branch and an unprotected one, and that 404 is a legitimate negative answer rather than a permission error, so no amount of reading the response body will separate them.
- **Corroborate with D only where an open PR exists**, and say which repos it could not reach rather than reporting a two-instrument figure over a population where one instrument was silent. The census that prompted this reported **9 repos C+D agreeing 9/9, and 4 where D could not run** — which is a smaller and truer claim than "13 repos, two instruments."
- **Never read an empty collection from a permissioned API as a null result** until you have a positive control showing the same call returns non-empty for a case you know is populated. A/B here had no such control, and would have passed any check that did not have one.
- **And for a null shaped like an error, the control is a NEGATIVE one: a subject that genuinely has the absence.** A positive control proves the query can see the thing; only the unprotected repo proves the query can tell you when the thing is gone. E passes the first and fails the second.

Related: [`uninformative-signals-both-directions`](#uninformative-signals-both-directions) · [`a-guard-exists-is-not-the-guard-protects-this`](#a-guard-exists-is-not-the-guard-protects-this) · and the portfolio working agreement's *give every absence a positive control*, which has no section here — it is a `C:\Projects\CLAUDE.md` rule, and is named rather than linked so the reference cannot rot into a dead anchor.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ AN EMPTY COLLECTION FROM A PERMISSIONED API IS NOT A NULL RESULT — AND NEITHER IS A 404 THAT THE GENUINELY-ABSENT CASE ALSO RETURNS.** Five instruments answer *"is `main` protected?"* on `ciresnave/fuel`; **three are wrong, and all three fail toward NONE — the answer that prompts a WRITE.** `branchProtectionRules` → **0 nodes** and `/rulesets` → **0 entries** return **HTTP 200 with an empty collection**, while `.protected` → **true** and `isRequired` → **5 required of 14 contexts** (so a Windows-only or macOS-only failure cannot block a merge here). And `branches/main/protection` **404s without admin BYTE-IDENTICALLY for a protected repo and an unprotected one** — 138 B stdout, 25 B stderr, `diff` clean on both streams — so its status code disguises a legitimate negative answer as an error. **A positive control proves a query can SEE the thing; only a subject that genuinely HAS the absence proves the query can report it MISSING.** → [`an-empty-collection-is-not-a-null-result`](#an-empty-collection-is-not-a-null-result)

## a-repoint-can-make-the-link-green-and-the-sentence-false

**⚠️ A FIX THAT MOVES THE METRIC IN THE APPROVING DIRECTION HAS NO DETECTOR EXCEPT READING THE SENTENCE (2026-09-09, Fuel 3, during the rustdoc-census repair; the classification, the counter-example and the inverse case are all theirs).**

A broken intra-doc link is a defect a gate can see. **Repairing it by REPOINTING the link at a live item that happens to share the name is not always a repair — sometimes it converts a true sentence into a false one, and the gate goes GREEN on the way.**

```
fuel-nn/src/conv_transpose.rs:5   //! Mirrors the eager [`fuel_nn::ConvTranspose1d`]
                                  // the struct at :117 is the LAZY item
repoint to crate::ConvTranspose1d ->  "the lazy X mirrors the [lazy X]"
                                       LINK RESOLVES.  SENTENCE IS NOW FALSE.
```

**A crate cannot name itself in an intra-doc link, so `fuel_nn::X` inside `fuel-nn` is always broken — and the mechanical repair is always `crate::X`.** It looks like the freest class of repair there is.

⚠️ **AND THE POPULATION EXPLAINS WHY IT IS THE MOST DANGEROUS ONE: 8 of the 9 such sites carried *eager* / *retired* / *former* language.** That is not bad luck. **The commonest reason a doc names its own crate is to contrast the current item with a FORMER one that lived in a crate which no longer exists** — so the structural class *"a crate naming itself"* and the historical class *"this names something retired"* coincide **for a reason**, and the coincidence is exactly what makes the mechanical fix wrong.

**THE RULE: STRUCTURE TELLS YOU WHETHER A LINK CAN RESOLVE; ONLY THE SENTENCE TELLS YOU WHETHER IT SHOULD.**

### The eleven-character proof, and why the unit of work is the SENTENCE and not the line

```diff
- //! [`Var`] is the lazy equivalent of eager [`crate::Var`]: a
+ //! [`Var`] is the lazy equivalent of eager `crate::Var`: a
```

**Two references on ONE LINE with OPPOSITE dispositions** — the live lazy `Var` stays linked; the retired eager `crate::Var` is de-linked. **Any line-scoped or file-scoped sweep takes both.** There is no pattern that separates them, because what separates them is the word *eager*.

### ⚠️ THE COUNTER-EXAMPLE THAT KEEPS THIS FROM BECOMING SUPERSTITION

**"Never repoint" is WRONG.** Measured in the same pass: `Recorder::submit_batch` had **already been repointed** at base, by someone else, to `VulkanBackend::submit_pending` — and there the public sibling genuinely carried the meaning. **That is what a real repoint looks like when one is available.**

### ⚠️ AND THE INVERSE CASE, MEASURED THE SAME NIGHT — WHERE DE-LINKING IS THE WRONG FIX

Six sites read *"Append a [`Op::QMatMul`] node"*. **`QMatMul` is not an `Op` variant at all** — it is a `FusedOpParams` variant, and the real node is `Op::Fused(FusedOpId, FusedOpParams)`. **So the sentence is ALREADY FALSE, and the broken link is the only thing advertising it.**

⚠️ **De-linking there would REMOVE THE ADVERTISEMENT AND LEAVE THE FALSEHOOD** — the worst of the three options, and the one a mechanical de-link sweep takes.

**THE DISCRIMINATOR IS NOT THE SHAPE OF THE LINK. IT IS WHETHER THE CODE SAYS WHAT THE THING BECAME — AND THE SOURCES RANK (Fuel 3, measured in the same program):**

```text
1  A NEIGHBOURING CORRECT REFERENCE   strongest, and it is free
2  a migration comment                the code stating what it became
3  a live sibling elsewhere           a public equivalent that carries the meaning
4  nothing                            -> DE-LINK; a repoint manufactures a falsehood
```

⚠️ **RANK 1 IS THE ONE NOBODY LOOKS FOR, AND IT IS SITTING IN THE SAME SENTENCE.** Worked example, `lazy_quantized_gemma3.rs`:

```text
//!   over [`Gemma3Weights::load_from_mmapped`] + [`Self::from_f32_bake`].
           ^^^^ correctly qualified                  ^^^^ broken
```

**The correct form is four words from the broken one, in the same doc block, naming the same type.** Not a comment, not a convention — **the same author getting it right about the same thing in the same sentence.** Before reaching for any other evidence, read the rest of the block.

**And the general form:**

```
the code says (a migration comment, a live sibling)  ->  REPOINT; the sentence gets truer
the code does NOT say                                ->  DE-LINK; a repoint manufactures a falsehood
the sentence is already false and the link is the tell -> REWRITE the sentence, never just the link
```

**And the third row is prose work, one site at a time, with the evidence cited per site** — `// Phase 7.6 step 4 (final): emits Op::Fused(QMATMUL, _)` is the licence, and it belongs in the commit for each site rather than once for the batch.

Related: [`docs-are-not-code-and-a-sweep-cannot-tell`](#docs-are-not-code-and-a-sweep-cannot-tell) · [`a-sweep-must-report-applied-over-population`](#a-sweep-must-report-applied-over-population) · [`marking-one-representation-does-not-mark-the-others`](#marking-one-representation-does-not-mark-the-others)

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A REPOINT CAN MAKE THE LINK GREEN AND THE SENTENCE FALSE — a fix that moves the metric in the APPROVING direction has no detector except reading the sentence.** A crate cannot name itself in an intra-doc link, so `fuel_nn::X` inside `fuel-nn` is always broken and the mechanical repair is always `crate::X` — **but 8 of 9 such sites said *"Mirrors the EAGER [`fuel_nn::X`]"*, and the repoint yields *"the lazy X mirrors the [lazy X]"*.** That is not bad luck: the commonest reason a doc names its own crate is to contrast the current item with a FORMER one, so the structural class and the historical class coincide **for a reason**. **STRUCTURE TELLS YOU WHETHER A LINK CAN RESOLVE; ONLY THE SENTENCE TELLS YOU WHETHER IT SHOULD** — and the unit of work is the sentence, not the line: one line carried a live `[`Var`]` and a retired `[`crate::Var`]` with opposite dispositions. ⚠️ Not superstition — a genuine repoint existed elsewhere (`Recorder::submit_batch` → `VulkanBackend::submit_pending`), and the INVERSE case exists too: six sites whose sentence is ALREADY false (`QMatMul` is a `FusedOpParams` variant, not an `Op` variant), where de-linking would remove the advertisement and leave the falsehood. **The discriminator is whether the CODE says what the thing became, never the shape of the link.** → [`a-repoint-can-make-the-link-green-and-the-sentence-false`](#a-repoint-can-make-the-link-green-and-the-sentence-false)

## a-sweep-must-report-applied-over-population

**⚠️ "NO ERRORS" AND "14 OF 25" ARE THE SAME RUN WITH DIFFERENT REPORTING, AND ONLY THE SECOND IS A RESULT (2026-09-09, Fuel 3, on their own applier).**

A sweep over a measured population ran with two instruments pointed at it, and **they were asking different questions:**

```
the ANCHOR check :  is the target PRESENT?                    passed 25/25
the EDIT patterns:  is it present IN A SHAPE I CAN REWRITE?   matched 14/25
```

**Only the second can fail silently.** The anchor check answers a strictly weaker question, it passes, and **a sweep that printed *"no errors"* would have shipped a half-done pass with a clean log** — 11 sites untouched, the population count unchanged, and nothing anywhere saying so.

**What caught it was that the applier reports WHAT IT APPLIED against WHAT IT WAS GIVEN.** That comparison is the entire difference.

### The blind spot had a shape, and it is worth knowing

```
[`Op::WriteSlice`](fuel_graph::Op::WriteSlice)
 ^^^^^^^^^^^^^^^  the link TEXT is an INTERMEDIATE-length path
```

The patterns covered the **full target** and the **bare leaf**. **The middle of a range is where a two-ended pattern set is blind, and nothing about a two-ended set announces that it has a middle.** Anchor on the exact target instead, so a sibling link on the same line still cannot be caught by it.

### THE GENERAL FORM

**Any sweep, in any tool, must report `applied / population`, never `success / failure`.** A pass rate is a measurement; an absence of errors is a statement about the sweep's own error handling and says nothing about coverage.

⚠️ **AND WHEN TWO INSTRUMENTS RUN OVER ONE POPULATION, ASK WHICH QUESTION EACH LITERALLY ANSWERS AND WHICH OF THEM IS STRICTLY WEAKER.** A passing weak instrument is routinely read as corroboration for the strong one. It is not corroboration; it is a different, easier question that happened to be asked at the same time.

Related: [`a-report-is-not-a-gate`](#a-report-is-not-a-gate) · [`evidence-that-is-not-independent`](#evidence-that-is-not-independent) · [`a-repoint-can-make-the-link-green-and-the-sentence-false`](#a-repoint-can-make-the-link-green-and-the-sentence-false)

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A SWEEP MUST REPORT `applied / population`, NEVER `success / failure`** — *"no errors"* and *"14 of 25"* are the same run with different reporting and only the second is a result. An anchor check asking *is the target PRESENT?* passed **25/25** while the edit patterns asking *is it present in a shape I can REWRITE?* matched **14/25**; only the second can fail silently, and a sweep printing "no errors" ships a half-done pass with a clean log. The blind spot had a shape worth knowing: the link TEXT was an **intermediate-length path**, and the patterns covered the full target and the bare leaf — **the middle of a range is where a two-ended pattern set is blind, and nothing about a two-ended set announces it has a middle.** ⚠️ When two instruments run over one population, ask which question each LITERALLY answers and which is strictly weaker — **a passing weak instrument reads as corroboration for the strong one and is not.** → [`a-sweep-must-report-applied-over-population`](#a-sweep-must-report-applied-over-population)

## a-correction-is-a-claim

**⚠️ A CORRECTION IS A CLAIM AND NEEDS THE SAME VERIFICATION AS THE THING IT CORRECTS — THE SECOND ERROR ARRIVES WEARING THE COSTUME OF THE FIX FOR THE FIRST (2026-08-13; KISS architect's formulation, off a Fuel instance where the coordinator was wrong twice running, in opposite directions).** Worked example, GAP-188: the coordinator filed a row asserting two op→family mappings with nothing checking they agree; **flagged on handover that the row might be malformed — correct**; then **overcorrected**, predicting family assignment therefore had *"no independent check at all"*, and told the lane it would be relayed cross-project if true. **Also false** — the conformance path is a **formatter with no production consumer** (so there is nothing to check), and the production assignment that does exist **is** unit-tested. **The lane disconfirmed a hypothesis handed to it by its own coordinator, which is the only reason a wrong correction did not go upstream.** **WHY THIS IS DANGEROUS RATHER THAN MERELY COMMON: a correction FEELS like rigour, so it is the claim least likely to be re-checked — and it inherits the credibility of having caught the first error.** **PRACTICE: verify a correction at head with the same positive controls you would demand of an original claim; and when handing a lane a hypothesis, say explicitly that DISCONFIRMING IT IS THE DELIVERABLE — a lane that stops at "your hypothesis is confirmed" has done the easy half.** Corollary: **an artifact corrected three times by its own owner is in better shape than one never corrected — the second is indistinguishable from the first until someone looks.** ⚠️⚠️ **AND THE TRIGGER, WHICH THIS RULE LACKED UNTIL 2026-08-28 AND WITHOUT WHICH IT IS UNACTIONABLE: YOUR OWN MEASUREMENT OUTRANKS A HANDED-DOWN CORRECTION, AND THE CONTRADICTION IS THE SIGNAL TO RE-CHECK RATHER THAN THE SIGNAL TO DEFER.** **Worked example with a real cost: a lane measured `fuel-quantized/neon-dotprod` vacuous and wrote a CORRECT sentence into `rust-ci.yml`. The architect "corrected" them — having confirmed the CI matrix HAS an `ubuntu-24.04-arm` runner and inferred the cell was LIVE there WITHOUT CHECKING THE GATE SET (the arm is TRIPLY gated and its `vdotq_s32` intrinsic is nightly-only, so it compiles itself away there too). The lane, in good faith, REWROTE THEIR TRUE SENTENCE INTO A FALSE ONE and cited the architect as the source.** **A TRUE COMMENT EDITED INTO A FALSE ONE BY A FIX** — undetectable by review, because the wrong ruling AND the rewrite both look like diligence. **Caught only because a THIRD lane pre-flighted the follow-on allocation instead of implementing it.** **The lane's own formulation is the rule and it is better than the one it corrects: *"verify a coordinator's correction before propagating it into a file I own — especially when it contradicts something I measured. I had measured that cell. That should have outranked being told otherwise."*** **Note the direction of authority makes this WORSE, not better: deference to a coordinator is the behaviour the structure asks for, so the failure mode is authority WORKING AS DESIGNED.** **And note the ranking of the three rulings, because the middle one is the durable hazard: the lane was RIGHT; the architect's first ruling was RIGHT-BY-ACCIDENT WITH A WRONG REASON; the architect's correction was WRONG. Right-by-accident would have survived indefinitely — it agrees with the truth, so nothing ever contradicts it.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A CORRECTION IS A CLAIM AND NEEDS THE SAME VERIFICATION AS THE THING IT CORRECTS — the second error arrives wearing the costume of the fix for the first.** ⚠️ **MECHANISM: a correction FEELS like rigour, so it is the claim LEAST likely to be re-checked — and it inherits the credibility of having caught the first error.** ⚠️ **TRIGGER, without which the rule is unactionable: YOUR OWN MEASUREMENT OUTRANKS A HANDED-DOWN CORRECTION, and a contradiction is the signal to RE-CHECK rather than to DEFER** — deference to a coordinator is the behaviour the structure asks for, so this failure mode is authority WORKING AS DESIGNED. **PRACTICE: verify a correction at head with the positive controls you would demand of an original claim; and when handing a lane a hypothesis, say explicitly that DISCONFIRMING IT IS THE DELIVERABLE.** ⚠️ Corollary worth keeping in the head: **an artifact corrected three times by its own owner is in better shape than one never corrected** — the second is indistinguishable from the first until someone looks. → [`a-correction-is-a-claim`](#a-correction-is-a-claim)

## name-the-construct-with-the-number

**THE NUMBER WAS CORRECT AND THE CONSTRUCT IT COUNTED WAS INVISIBLE IN IT — NAME THE CONSTRUCT IN THE SAME BREATH AS THE NUMBER (2026-08-12; four instances in one day, four different constructs).** **(i) EXTRACTION BOUNDARY:** a per-function line count that ran to the next `fn` swept the *following* function's doc comments, so `decode_shape_key` read as **3.1x** (15 vs 47) when it is **14 vs 11 lines of code** — and an architectural scope was ruled off that ratio. ⚠️ **THE CORRECT METHOD, ADDED 2026-09-02 BECAUSE THIS ENTRY NAMED THE DEFECT WITHOUT NAMING THE REMEDY AND WAS THEREFORE REPRODUCED BY SOMEONE WHO HAD READ IT: measure a body by BRACE MATCHING from the signature to its closing brace, never `fn`-to-next-`fn`.** Recurrence: a per-function count on `fuel-ir/tests/crate_dependency_direction.rs` reported an arm at **52 lines when its body is 23**, and a function was about to be restructured to satisfy a number that did not exist. **Knowing the defect is not the defence — re-measuring with the right instrument is, and an entry that describes a trap without giving the tool that avoids it leaves the reader to reinvent it under time pressure.** **(ii) INERT POSITIVE CONTROL:** the symbol chosen to prove a test could see referencedness was itself unreferenced on the path under test. **(iii) DENOMINATOR SWITCH:** `86/208` (41%) reported against a go/no-go rule written about `86/143` (60%) — both true, different questions. **(iv) LINE-IDENTITY vs UNIT OF MEANING:** two `decode_shape_key` bodies are **81% textually identical** while the unit that matters is the **field SET**, and the sets differ 10 vs 7 — so *"81% overlap"* is a true statement about lines and a false one about duplication. **In all four the arithmetic was right; what the count RANGED OVER was never stated, and therefore never checked.** **The defence is NOT skepticism about numbers — it is naming the construct in the same clause: "143 lines byte-identical, out of Phi's 178-line body, blank lines excluded, extracted `fn`-start to closing brace." That costs one clause and would have caught all four.** Corollary for percentages: **a bare percentage is unreadable — always give the denominator.** Related: [[enumerate-the-population-not-the-strings]] (count the construct, not the strings) is the same defect one level up, about choosing the population rather than describing it.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **THE NUMBER WAS CORRECT AND THE CONSTRUCT IT COUNTED WAS INVISIBLE IN IT — NAME THE CONSTRUCT IN THE SAME BREATH AS THE NUMBER.** ⚠️ **MECHANISM: in every instance the ARITHMETIC was right; what the count RANGED OVER was never stated, and therefore never checked.** ⚠️ **THE ONE OPERATIVE REMEDY, KEPT HERE BECAUSE THIS ENTRY ONCE NAMED THE DEFECT WITHOUT THE REMEDY AND WAS REPRODUCED BY SOMEONE WHO HAD READ IT: measure a function body by BRACE MATCHING from the signature to its closing brace, NEVER `fn`-to-next-`fn`** — the latter sweeps the following function's doc comments and once reported a 23-line body as 52. **PRACTICE: name the construct in the same clause — *"143 lines byte-identical, out of Phi's 178-line body, blank lines excluded, brace-matched"*. And a bare percentage is unreadable: always give the denominator.** → [`name-the-construct-with-the-number`](#name-the-construct-with-the-number)

## a-compile-line-licenses-only-the-phases-that-ran

**A COMPILE LINE LICENSES A CONCLUSION ONLY ABOUT THE PHASES THAT RAN — AN EARLY-PHASE ERROR SUPPRESSES EVERY LATER PHASE SILENTLY (2026-08-08). This AMENDS the target-crate rule below, which it DEFEATS as written.** A cross-check of `fuel-metal-backend` produced **`Checking fuel-metal-backend v0.10.3`** — the exact artifact that rule demands — and **`0 x E0004`**, and *that combination licensed nothing*: the 3 errors were `E0432`/`E0433`, **name-resolution** failures, and **resolution runs BEFORE exhaustiveness checking**, so no `DType` match was ever type-checked. **The rule was written for "the compiler ran" vs "the compiler reached the code under test"; this is one level deeper — it reached the CRATE but not the PHASE.** **Practical form: an absence of late-phase diagnostics (E0004 exhaustiveness, type errors, borrowck) is MEANINGLESS in a compile that emitted ANY resolution or macro-expansion error.** Fix the early errors and re-run before reading the absence of the late ones — **and when the question IS "does this exhaustiveness hold", sabotage the check (plant a non-exhaustive match, confirm it is REJECTED) rather than trusting a clean run.** ⚠️⚠️ **AND THE RESTORE STEP IS THE ONE THAT SILENTLY NO-OPS — A BYTE-IDENTICAL RESTORE MAY NOT INVALIDATE CARGO'S FINGERPRINT, SO YOU CAN BE READING THE SABOTAGED BINARY WITH CLEAN SOURCE AND A CLEAN `git status` (2026-08-14, GAP-203 lane).** Observed: after restoring a sabotaged file the suite still reported the **sabotage value** (`6.7964196e-5`), source clean, `git status` clean. The lane nearly read that as *"my bound is wrong."* **⚠️ AND THE OTHER DIRECTION IS WORSE, WHICH IS WHY THIS IS A RULE AND NOT A CURIOSITY: A SABOTAGE THAT NEVER RECOMPILED LOOKS EXACTLY LIKE A PASSING CONTROL — a validation that proves nothing while looking rigorous.** The whole sabotage discipline rests on the binary matching the source at two separate moments, and **`git status` attests to the SOURCE, never to the BINARY.** **PRACTICE: clear the crate fingerprint on BOTH transitions and require a `Compiling <crate>` artifact before believing either the sabotage or the restore.** Same family as the warm-cache rules above, pointed specifically at sabotage testing, where the cheap step — putting the file back — is the one nobody instruments.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A COMPILE LINE LICENSES A CONCLUSION ONLY ABOUT THE PHASES THAT RAN — AN EARLY-PHASE ERROR SUPPRESSES EVERY LATER PHASE SILENTLY.** This AMENDS the target-crate rule below, which it DEFEATS as written. ⚠️ **MECHANISM: name resolution runs BEFORE exhaustiveness checking, so `Checking <crate>` plus `0 x E0004` licenses NOTHING if the compile emitted any `E0432`/`E0433`.** **PRACTICE: an absence of late-phase diagnostics (E0004, type errors, borrowck) is MEANINGLESS in a compile that emitted ANY resolution or macro-expansion error — fix the early ones and re-run before reading the absence of the late ones. And when the question IS *does this exhaustiveness hold*, SABOTAGE it (plant a non-exhaustive match, confirm it is REJECTED) rather than trusting a clean run.** ⚠️⚠️ **AND THE RESTORE STEP SILENTLY NO-OPS: a byte-identical restore may not invalidate cargo's fingerprint, so you can be reading the SABOTAGED binary with clean source and a clean `git status` — and A SABOTAGE THAT NEVER RECOMPILED LOOKS EXACTLY LIKE A PASSING CONTROL. `git status` attests to the SOURCE, never the BINARY. Clear the crate fingerprint on BOTH transitions and require a `Compiling <crate>` artifact before believing either.** → [`a-compile-line-licenses-only-the-phases-that-ran`](#a-compile-line-licenses-only-the-phases-that-ran)

## adding-a-binding-is-gated-by-the-lint

**⚠️ ADDING A BINDING IS A CHANGE-KIND WHOSE GATE IS THE LINT, NOT THE TEST — AND A PASSING SUITE IS CORRECT AND UNINFORMATIVE, WHICH IS THE WORST COMBINATION (2026-09-02, the GAP-265 dependency-direction gate).** A vacuity hole was found in a cross-check arm, the fix was written up in the arm's doc comment, the counter variable was added — **and the `assert!` reading it never was.** `cargo test` passed, honestly: **a test CANNOT fail on an unused variable.** CI caught it because CI runs `-D warnings` while locally it is a warning — `error: variable 'compared' is assigned to, but never used`. **So the local green was real and about a different question than the one being asked.** The gate matching this change-kind is **`cargo clippy -p <crate> --tests -- -D warnings`** — note `--tests`, because the binding lived in a test target and a lib-scoped clippy never parses it. **This is the FOURTH entry in the gate-must-match-the-change-kind family, after enum variants (workspace scope), struct fields (every CONSTRUCTING crate) and `cfg`'d modules (the feature that compiles them) — and it is the one where the WRONG gate reports SUCCESS rather than silence.** ⚠️ **The second half generalises past Rust and is worse than the gate error: THE DOC COMMENT VOUCHED FOR THE ASSERTION.** Re-reading the change, the prose said the guard existed, so it read as done — **and the author is the worst possible reviewer of that difference, because they know what they meant. A comment describing a guard is not weak evidence that the guard exists; it is ZERO evidence, and it reads as strong.** Same family as a justification that is true about intent and silent about existence. **PRACTICE: watch a guard FAIL before believing it exists — the sabotage is not polish, it is the only thing separating an assertion from a sentence about one.** The instance is worth keeping for its proximity: **the arm whose stated argument is *a guard must assert the quantity that MOVES* had a variable that moved and no assertion reading it — vacuous in precisely the manner it existed to detect.** And note what surfaced it: **two readers and a green local suite agreed on something false, and one stricter flag disagreed.** That is the argument for `-D warnings` in CI over any amount of care.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ ADDING A BINDING IS A CHANGE-KIND WHOSE GATE IS THE LINT, NOT THE TEST — AND A PASSING SUITE IS CORRECT AND UNINFORMATIVE, WHICH IS THE WORST COMBINATION.** ⚠️ **MECHANISM: a test CANNOT fail on an unused variable, so a counter added without the `assert!` that reads it leaves `cargo test` honestly green about a different question.** **THE GATE FOR THIS CHANGE-KIND: `cargo clippy -p <crate> --tests -- -D warnings` — note `--tests`, because a lib-scoped clippy never parses a test target.** Fourth entry in the gate-must-match-the-change-kind family (after enum variants → workspace scope, struct fields → every CONSTRUCTING crate, `cfg`'d modules → the feature that compiles them), and the one where the WRONG gate reports SUCCESS rather than silence. ⚠️ **AND THE HALF THAT GENERALISES PAST RUST AND IS WORSE: A COMMENT DESCRIBING A GUARD IS NOT WEAK EVIDENCE THAT THE GUARD EXISTS — IT IS ZERO EVIDENCE, AND IT READS AS STRONG.** **PRACTICE: watch a guard FAIL before believing it exists; the sabotage is not polish, it is the only thing separating an assertion from a sentence about one.** → [`adding-a-binding-is-gated-by-the-lint`](#adding-a-binding-is-gated-by-the-lint)

## a-same-signature-repoint-is-a-behaviour-change

**⚠️ A REFACTOR THAT REPOINTS A CALL AT A *SAME-SIGNATURE* HELPER IS A BEHAVIOUR CHANGE THE COMPILER CANNOT SEE — AND "PURE MOVE" IS THE LABEL UNDER WHICH IT TRAVELS (2026-08-15, GAP-209).** Two functions typed `fn(DType, &[f32]) -> Option<Vec<u8>>` encoded **11 dtypes** and **4**. An extraction bound the call to the 4-dtype one; everything type-checked, the commit said *pure move*, the module doc said *"Extracted VERBATIM ... no behaviour change"*, and **228 of 530 earned ledger records silently became non-re-earnable.** `Option` returning `None` more often is a **domain narrowing with no type signature**, and a pure-move refactor verified by compiling is STRUCTURALLY BLIND to it — the label was not careless; the gate could not have contradicted it. **⚠️⚠️ AND THE PRESCRIBED FIX WAS WRONG IN THE SAME FAMILY, WHICH IS THE HALF WORTH KEEPING. The architect ruled *"dedup to the superset, delete the loser"*; the lane REFUTED it with measurement: the narrow encoder is re-exported and consumed by JIT ingest, where a test ASSERTS `to_bytes(DType::I32) == None`, because an unencodable operand must yield NO probe rather than a fabricated one. Widening it would have committed the mirror image of the bug. SO: TWO FUNCTIONS WITH ONE SIGNATURE AND DIFFERENT DOMAINS ARE EITHER (a) AN ACCIDENTAL DIVERGENCE — dedup to the superset — OR (b) TWO DISTINCT OBLIGATIONS SHARING A SHAPE, WHICH MUST NOT BE MERGED AT ALL. NOTHING IN THE CODE DISTINGUISHES THEM; ONLY THE CALLERS' REQUIREMENTS DO — so READ THE CALLERS BEFORE DEDUPLICATING.** **AND THE DEDUP HALF THAT STANDS: the module's own doc comment warned that a second copy "would be a divergence generator" and cited the incident proving it — then resolved the divergence toward the NARROWER implementation. Deduplication picks a winner; nothing checks WHICH, and a correct warning fourteen lines above the violation does not fire.** **PRACTICE: when a refactor changes WHICH implementation a call reaches, diff the two DOMAINS and read both call sets, never the signatures. Where the domain is data-driven, gate it with a POPULATION test over the real corpus — and prefer an UNCONDITIONAL weaker assertion over a feature-gated stronger one, because a gate nobody's build runs is not a gate.** Final note on aim: the architect warned about `DType::Bool`, which was handled correctly; the regression was its sibling, and **Bool turned out to be missing from BOTH encoders** — **a hazard you have already seen is the one you are least likely to be hit by.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **⚠️ A REFACTOR THAT REPOINTS A CALL AT A *SAME-SIGNATURE* HELPER IS A BEHAVIOUR CHANGE THE COMPILER CANNOT SEE — AND "PURE MOVE" IS THE LABEL UNDER WHICH IT TRAVELS.** ⚠️ **MECHANISM: an `Option` that returns `None` more often is a DOMAIN NARROWING WITH NO TYPE SIGNATURE, so a pure-move refactor verified by compiling is structurally blind to it — the label was not careless, the gate could not have contradicted it.** ⚠️ **AND THE OBVIOUS FIX IS WRONG HALF THE TIME: two functions with ONE signature and DIFFERENT domains are either (a) an ACCIDENTAL DIVERGENCE — dedup to the superset — or (b) TWO DISTINCT OBLIGATIONS SHARING A SHAPE, which must NOT be merged at all. NOTHING IN THE CODE DISTINGUISHES THEM; ONLY THE CALLERS' REQUIREMENTS DO — so READ THE CALLERS BEFORE DEDUPLICATING.** **PRACTICE: when a refactor changes WHICH implementation a call reaches, diff the two DOMAINS and read both call sets, never the signatures. Where the domain is data-driven, gate it with a POPULATION test over the real corpus, and prefer an UNCONDITIONAL weaker assertion over a feature-gated stronger one — a gate nobody's build runs is not a gate.** → [`a-same-signature-repoint-is-a-behaviour-change`](#a-same-signature-repoint-is-a-behaviour-change)

## cap-build-parallelism-at-j4

**CAP BUILD PARALLELISM: use `-j 4`. One cargo invocation at full `-j` is enough to break the build on this box (2026-08-07).** The rule above says nothing about `-j` *within* a single invocation, and that is the gap: this machine gives cargo **~22 jobs**, and at that width concurrent rustc invocations race. Symptoms are **nondeterministic and blame the wrong thing** — rustc names *different* dependency crates as "required to be available in rlib format" on each run, plus intermittent ICEs leading with `delayed bug: no resolution for an import` (cascading to `Res::Err but no error emitted`) — while **every target builds and passes individually**, so it reads as a code defect on the branch under test. Diagnosed by Lightbulb across a 5-hypothesis elimination (sccache, toolchain, stale artifacts, disk, project-specific all falsified); the project-specific leg fell because **Fuel hit the identical ICE signature the same day**. `cargo test --tests -j 4` passes clean on a tree that failed three times at default width. **`-j 4` is known-good, NOT a bisected boundary** — the true threshold is unmeasured. See GAP-019. ⚠️ **AND THE RULE ASSUMES THE DUMP EXISTS — MEASURED 2026-08-27, IT MAY NOT, AND THE ABSENCE READS THE WRONG WAY.** A concurrent-link race on this box produced the full documented signature — `Res::Err but no error emitted`, `no errors encountered even though delayed bugs were created`, a **randomly-named** crate (`test "phase_c_rotating_kv"`) — **and wrote NO `rustc-ice-*.txt` at all.** **A reader who looks for the dump, finds none, and concludes "not an ICE" lands on the code under test — which is the exact misdiagnosis this rule exists to prevent, arriving through the rule's own assumption.** **THE DUMP-INDEPENDENT DISCRIMINATOR, and it is cleaner than the dump: `exit != 0` together with ZERO `level:error` messages in `--message-format json`.** A real compile failure has errors; this had none. **Pair it with the randomly-named crate and it is unambiguous without any artifact on disk.** (Found by the GAP-243 verification lane, who had forgotten the `-j 4` cap above — which is also the reminder that this rule and the `-j 4` rule are the same incident seen from two ends.) **And do not delete `rustc-ice-*.txt`**: a crash artifact is evidence until something explains it, so hygiene comes *after* the explanation. Five ICEs occurred on this box that day and only three survived, because "untracked crash dump at a repo root" is a category people tidy without thinking.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **ONE cargo invocation at a time.** The build-dir lock serializes; parallel invocations thrash. Long builds: background + wait.

- **CAP BUILD PARALLELISM: use `-j 4`.** One cargo invocation at full `-j` is enough to break the build on this box: the machine gives cargo **~22 jobs**, and at that width concurrent rustc invocations race. ⚠️ **MECHANISM — why it blames the wrong thing: the symptoms are NONDETERMINISTIC and every target builds and passes INDIVIDUALLY, so it reads as a code defect on the branch under test.** rustc names *different* dependency crates as "required to be available in rlib format" on each run, plus intermittent ICEs leading with `delayed bug: no resolution for an import`. **`-j 4` is known-good, NOT a bisected boundary — the true threshold is unmeasured (GAP-019).** ⚠️⚠️ **THE DUMP-INDEPENDENT DISCRIMINATOR, KEPT HERE BECAUSE THE DUMP MAY NOT EXIST AND ITS ABSENCE READS THE WRONG WAY: `exit != 0` together with ZERO `level:error` messages in `--message-format json`.** A real compile failure has errors; a concurrent-link race has none. Pair it with the randomly-named crate and it is unambiguous with no artifact on disk — **a reader who looks for `rustc-ice-*.txt`, finds none and concludes "not an ICE" lands on the code under test, which is the exact misdiagnosis this rule exists to prevent, arriving through the rule's own assumption.** **And do not delete `rustc-ice-*.txt`: a crash artifact is evidence until something explains it, so hygiene comes AFTER the explanation.** → [`cap-build-parallelism-at-j4`](#cap-build-parallelism-at-j4)

## background-tasks-are-killed-at-ten-minutes

**BACKGROUND TASKS ARE KILLED AT ~10 MINUTES — TOOL-AGNOSTIC — SO ANY CUDA BUILD MUST BE FULLY DETACHED (2026-08-08).** The kernel forge takes **~56 minutes at `BARACUDA_FORGE_THREADS=6`** on this box, i.e. **5x over the limit**. **⚠️ RE-MEASURED 2026-08-27: 92m49s on a loaded box, against this ~56m figure — 65% over. Box load, not a hang: the lane's ~8-minute polling windows are the only reason the PM's hourly stall timer did not read live work as dead. TREAT ~56m AS A FLOOR MEASURED ON A QUIET BOX, NOT AN ESTIMATE — size a slot at 90m+ and expect an under-budgeted slot to LOOK LIKE A HANG to anyone watching it.** Two lanes lost three builds to this before it was diagnosed; both initially suspected the *tool* (PowerShell vs Bash) because their PowerShell runs happened to be the only ones exceeding ten minutes — **a confound resolved only by comparing two lanes' reports.** **Working pattern, proven on this box (a 59.4-minute build to completion):** launch **fully detached** so no harness timeout owns it — **`Invoke-CimMethod -ClassName Win32_Process -MethodName Create` is CONFIRMED working** (a WMI-created process is parented to the WMI service, not to the calling shell). **`Start-Process -WindowStyle Hidden -PassThru` is CONFIRMED-NOT-IMPLICATED — both launchers work; the CAUSE was the script.** An isolating control settles it: three attempts ran, not two — **(A)** `Start-Process` + broken bat -> died; **(B)** WMI + **the same broken bat** -> died identically (297 bytes, no marker); **(C)** WMI + fixed bat -> survived. **B is the control: same script, different launcher, same death — so the launcher was never the differentiator.** That reconciles the two lanes (one completed a 59.4-minute build on `Start-Process`) instead of leaving their evidence in apparent conflict. **Residual confound, stated rather than smoothed:** vcvarsall reported **v18.0 on one run and v18.8.2 on another minutes apart**, so no two of A/B/C shared a toolchain — it does not disturb the bat-vs-launcher conclusion (A and B died alike) but **no two runs are strictly comparable.** **THREE checks, because any two leave a state indistinguishable:** **liveness** = PID alive (running vs died); **completion** = a marker carrying **cargo's OWN exit code**; **verdict** = the **target crate's** compile line; have the script write an explicit **`..._DONE_EXITCODE=` marker**; and **pair the marker with a pid check**, because marker-absent is consistent with BOTH *still running* and *died* — the marker alone is not a liveness test. Completion is then confirmed by the **target crate's own compile line** (see the rule below), not by artifacts-in-general.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **BACKGROUND TASKS ARE KILLED AT ~10 MINUTES — TOOL-AGNOSTIC — SO ANY CUDA BUILD MUST BE FULLY DETACHED.** ⚠️ **BUDGET FROM THE MEASURED FLOOR, NOT THE ESTIMATE: the kernel forge is ~56 min at `BARACUDA_FORGE_THREADS=6` on a QUIET box and was measured at 92m49s under load — treat ~56m as a FLOOR, size a slot at 90m+, and expect an under-budgeted slot to LOOK LIKE A HANG to anyone watching it.** **LAUNCH DETACHED so no harness timeout owns it: `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` (a WMI-created process is parented to the WMI service, not to the calling shell) or `Start-Process -WindowStyle Hidden -PassThru`.** ⚠️ **BOTH LAUNCHERS WORK — the launcher was never the differentiator; the cause was the script.** ⚠️ **THREE CHECKS, BECAUSE ANY TWO LEAVE A STATE INDISTINGUISHABLE: LIVENESS = PID alive · COMPLETION = a marker carrying CARGO's OWN exit code · VERDICT = the TARGET CRATE's compile line.** **Marker-absent is consistent with BOTH still-running and died, so the marker alone is not a liveness test — pair it with a pid check.** → [`background-tasks-are-killed-at-ten-minutes`](#background-tasks-are-killed-at-ten-minutes)

## scan-for-both-crate-aliases

**Scan for BOTH crate aliases, or your search is wrong by construction.** The root manifest aliases fuel-core: `fuel = { path = "./fuel-core", package = "fuel-core" }`. Consumers therefore write `use fuel::…` and essentially never `fuel_core::`, so **any grep keyed on `fuel_core::` returns near-zero no matter what is really there** — during B6 this was the difference between ~5 files and 137. Grouped imports (`use fuel::{DType, Tensor}`) also defeat a naive `fuel::Tensor` pattern. Two rules follow: search `fuel(_core)?::` with the symbol matched separately, and **give every "I found nothing" a positive control** — a query you know should hit — before reporting absence. The same defect produced two other false claims recorded above. **⚠️ AND THE BOUND ON THAT DISCIPLINE, ESTABLISHED THE HARD WAY (2026-08-12, three instances in ONE night, all by the coordinator, all caught unprompted): A POSITIVE CONTROL PROVES THE QUERY CAN FIND THE THING IT LOOKS FOR. IT DOES NOT PROVE THE QUERY LOOKS FOR THE RIGHT THING.** All three claims below had a passing positive control and were still wrong: **(1)** *"Fuel is not a reader"* — true of `structure_key`s, unqualified as stated, and `.fkc.md` contracts DO parse dtype tokens; **(2)** the wire-boundary exhaustiveness guarantee reported as unenforced — the **file's** feature gate was read and the **guarantee's** coverage concluded, after personally approving the change that moved the match out of that file; **(3)** *"the key types are crate-internal"* — from *"is it re-exported from `lib.rs`?"*, when **reachability is a `pub mod` chain** and the type is published by path with 7 pub fields. **OPERATIONAL FORM: state THE QUESTION THE QUERY LITERALLY ANSWERS, and where a property has MORE THAN ONE MECHANISM, name which mechanisms you examined and which you did not** — visibility by re-export **and** by module path; enforcement by CI **and** by a local script; parsing at a wire boundary **and** in a format you own. **Instance (3) is the cleanest: the right answer needed TWO visibility queries, and one returning empty was reported as the property being absent.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Scan for BOTH crate aliases, or your search is wrong by construction.** The root manifest aliases fuel-core: `fuel = { path = "./fuel-core", package = "fuel-core" }`, so consumers write `use fuel::…` and essentially never `fuel_core::` — **any grep keyed on `fuel_core::` returns near-zero no matter what is really there** (during B6 that was ~5 files against 137). Grouped imports (`use fuel::{DType, Tensor}`) also defeat a naive `fuel::Tensor` pattern. **SEARCH `fuel(_core)?::` WITH THE SYMBOL MATCHED SEPARATELY, and give every "I found nothing" a positive control** — a query you know should hit — before reporting absence. ⚠️⚠️ **AND THE BOUND ON THAT DISCIPLINE: A POSITIVE CONTROL PROVES THE QUERY CAN FIND THE THING IT LOOKS FOR. IT DOES NOT PROVE THE QUERY LOOKS FOR THE RIGHT THING.** Three claims in one night had a PASSING positive control and were still wrong. **OPERATIONAL FORM: state THE QUESTION THE QUERY LITERALLY ANSWERS, and where a property has MORE THAN ONE MECHANISM, name which mechanisms you examined and which you did not** — visibility by re-export **and** by `pub mod` chain; enforcement by CI **and** by a local script; parsing at a wire boundary **and** in a format you own. → [`scan-for-both-crate-aliases`](#scan-for-both-crate-aliases)

## the-shared-tree-and-the-shared-hook

**Multiple agent sessions share ONE working tree + `.git/index` here — never mutate git state in the shared checkout concurrently.** Several Claude sessions run in `C:\Projects\fuel` at once (the claude-peers channel lists them); they share one working directory *and* one git index, so concurrent `git add`/`commit`/`checkout`/`reset`/`rebase`/`stash` clobber each other silently — a `git add` in one session gets swept into another session's `commit` (observed 2026-07-20). Rules: **(1)** treat the shared checkout's `main` as **read-only** — never develop, `git add`, or commit on it while other sessions may be active; **(2)** do all commit-producing work — **code AND docs** — in an isolated **`git worktree`**: `git worktree add ../fuel-<task> -b <branch> origin/main`, edit/commit *there*, then `git push origin HEAD:main` (or open a PR). Sibling path-deps `../aocl`/`../vulkane` still resolve from a `C:\Projects`-sibling worktree, so the workspace parses. **(3)** Branch from **pushed** `origin/main`, never a stale local base; **re-fetch right before pushing** ⚠️⚠️ **AND THE MIRROR OF THAT RULE, WHICH IS THE DIRECTION THAT LOOKS LIKE NOTHING HAPPENED: `origin/main` CAN MOVE **FORWARD** UNDER YOU MID-TASK, INSIDE THE SHARED `.git`.** Linked worktrees share the refs directory (`.git` in a worktree is a one-line `gitdir:` pointer into the main repo), **so ANY peer fetching advances `origin/main` for EVERY lane at once.** Measured 2026-09-02: a lane read `origin/main` and then ran `git checkout -B <branch> origin/main`, and **the branch was created at a DIFFERENT, NEWER commit than the one just measured** — a peer had fetched in between. **Harmless there (a newer base is a better base) and NOT harmless in general: a rebase or a `--is-ancestor` check then silently answers about a commit you never measured.** **The stale-ref rule covers reading a ref that is BEHIND; this is the same name being AHEAD, and nothing in the output marks it.** ⚠️ **PRACTICE: capture the SHA ONCE and branch from the CONSTANT, not from the moving name** — `BASE=$(git rev-parse origin/main)` then `git checkout -B <branch> $BASE` — **so a peer’s fetch cannot re-point your base under a name you are treating as fixed.** Caught only because a branch tip disagreed with expectation and the lane went to the reflog instead of assuming they had fumbled. — a peer may have advanced `main` under you between fetch and push. **(4)** If two sessions genuinely must share the tree, coordinate over the peers channel so only one performs git ops at a time. ⚠️ **AND `gh pr merge --delete-branch` IS A GIT MUTATION OF THE LOCAL REPO, NOT JUST A FORGE CALL — RUN IT FROM A THROWAWAY WORKTREE, NEVER FROM THE SHARED CHECKOUT (2026-08-26).** After merging, `gh` performs a local `git checkout <base>` and deletes the local branch. **In the shared tree that moves `main` under every lane whose cwd points there, including any holding a long build** — the exact class this bullet exists to prevent, arriving through a command that reads as a remote operation. **Observed: it ERRORED and the shared checkout was verified unchanged afterwards (still `88997178`, still on `main`, no rebase/merge state) — so the near-miss cost nothing, and it cost nothing BY LUCK rather than by design.** Verify the shared tree's `HEAD` after any `gh` command that can touch refs, and prefer running them from a worktree you own. This supersedes the looser "WIP goes on a branch" note below for the multi-session case. ⚠️⚠️ **AND THE TRAP THAT SURVIVES A WORKTREE, FOUND 2026-09-02 BY A LANE WHO STOPPED RATHER THAN EDITING: `core.hooksPath` IS SET TO THE ABSOLUTE PATH `C:\Projects\fuel\.git\hooks`, SO A LINKED WORKTREE RUNS THE **SHARED** HOOK FILE, NOT A COPY.** Every other isolation a worktree gives you stops here: **hand-editing `.git/hooks/pre-commit` is a LIVE, MACHINE-WIDE MUTATION landing on every fuel lane's NEXT COMMIT, mid-flight, including a peer landing an unrelated PR.** ⚠️ **And that file is UNTRACKED** (`git ls-files --error-unmatch .git/hooks/pre-commit` → *does not match any file(s) known to git*), **so nothing reviews it and nothing records who changed it.** Its own header names the tracked sources: **`scripts/install-hooks.sh` + `scripts/check-gaps-table.py`, and THOSE are where a hook change belongs.** **Practice: never edit the hook; edit the tracked script it is generated from.**

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Multiple agent sessions share ONE working tree + `.git/index` here — never mutate git state in the shared checkout concurrently.** Concurrent `add`/`commit`/`checkout`/`reset`/`rebase`/`stash` clobber each other SILENTLY — a `git add` in one session gets swept into another session's `commit`. **(1)** treat the shared checkout's `main` as **read-only**; **(2)** do all commit-producing work — code AND docs — in an isolated **`git worktree`** (`git worktree add ../fuel-<task> -b <branch> origin/main`), commit there, then push or open a PR; **(3)** branch from **pushed** `origin/main`, never a stale local base, and re-fetch right before pushing; **(4)** if two sessions genuinely must share the tree, coordinate over the peers channel so only one does git ops at a time. ⚠️⚠️ **AND `origin/main` CAN MOVE *FORWARD* UNDER YOU MID-TASK: linked worktrees share the refs directory, so ANY peer fetching advances `origin/main` for EVERY lane at once — the stale-ref rule covers a ref that is BEHIND, this is the same name being AHEAD, and nothing in the output marks it. PRACTICE: CAPTURE THE SHA ONCE AND BRANCH FROM THE CONSTANT** — `BASE=$(git rev-parse origin/main)` then `git checkout -B <branch> $BASE` — **so a peer's fetch cannot re-point a base you are treating as fixed.** ⚠️ **`gh pr merge --delete-branch` IS A LOCAL GIT MUTATION, NOT JUST A FORGE CALL — run it from a throwaway worktree, never from the shared checkout**, and verify the shared tree's `HEAD` after any `gh` command that can touch refs. ⚠️⚠️ **AND THE TRAP THAT SURVIVES A WORKTREE: `core.hooksPath` IS THE ABSOLUTE PATH `C:\Projects\fuel\.git\hooks`, SO A LINKED WORKTREE RUNS THE *SHARED* HOOK FILE, NOT A COPY.** Hand-editing `.git/hooks/pre-commit` is a LIVE, MACHINE-WIDE mutation landing on every fuel lane's NEXT commit, mid-flight — **and the file is UNTRACKED, so nothing reviews it and nothing records who changed it. NEVER EDIT THE HOOK; edit the tracked sources it is generated from (`scripts/install-hooks.sh` + `scripts/check-gaps-table.py`).** → [`the-shared-tree-and-the-shared-hook`](#the-shared-tree-and-the-shared-hook)

## lazy-only-and-recorded-deferrals

**Lazy-only. New features land lazy-only** — permanent, and not conditional on anything: there is no eager path left to land on. **⚠️ AMENDED 2026-08-20 — this bullet used to read "Lazy-only / no deferrals UNTIL eager is fully retired", and that condition has been MET** (`git grep -l BackpropOp -- '*.rs'` → **0**; `git ls-files fuel-core/src/op.rs` → **gone**; control `git ls-files 'fuel-core/src/*.rs' | wc -l` → **190**, so the zeros are absence rather than a broken query). **An authorization that names its own expiry, and whose expiry has arrived, does not silently continue.** **The two halves separate, and only one survives.** *New features land lazy-only* stands on independent grounds and is restated above without the expiry clause, so it stops looking conditional. ***No deferrals* LAPSES — and is SUPERSEDED BY A MECHANISM, not merely retired:** its justification was avoiding eager-shaped debt while a retirement was in flight, and that program has finished. What replaces it is `docs/gaps.md`'s own rule — **a deferral is fine if it is RECORDED, with a tier and an owner** (**over 200** live rows. ⚠️⚠️ **THE CLAUSE THAT STOOD HERE — *"most carrying an owner field"* — IS FALSE AND WAS THE JUSTIFICATION FOR THIS RETIREMENT. MEASURED 2026-09-06 at `f10bdb79`: 236 open rows, **11** `**Owner:` markers, and THERE IS NO OWNER COLUMN — all 16 header shapes are `| ID | File:Line | Tier | Gap | Status |` or the 4-column variant.** **So ownership is not mechanically answerable for 225 of 236 open rows.** ⚠️ **THE ARGUMENT DEPENDED ON IT: the retirement is justified above by *"a registry row is greppable"*. THE ROW IS GREPPABLE; THE OWNER IS NOT — the replacement mechanism is real but WEAKER than this sentence claimed.** **The ruling stands (it is CireSnave’s); only its stated evidence was overclaimed.** ⚠️ **AND NOTE WHERE THE CARE WENT: this sentence was scrupulous about the NUMBER’S PRECISION (*"stated as a floor rather than a point because the exact figures rotted in FOUR DAYS"*) and wrong about the PROPERTY. Hedging a figure does not verify the predicate it quantifies.** **Found by the Claim Auditor, who stopped one keystroke from filing "225 rows have no owner" because a 10x disagreement with the working agreement was too large to file, and then read the table HEADER to ask which construct carries the property.** *(Earlier figures 210→217 and 175→180 counted a different construct and are retired with the claim.)* — stated as a floor rather than a point because the exact figures rotted in FOUR DAYS: 210→217 and 175→180, measured 2026-08-21 and re-measured 2026-08-25. A count has no rename-resistant form; a BOUND is its analogue). That is strictly stronger than the prohibition it replaces, because a blanket ban is unenforceable and a registry row is greppable. **Do not read this as "deferrals are now unremarked"** — read it as: the ban moved from prose into a registry that can be queried, and an unrecorded deferral is still the thing being forbidden.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Lazy-only. New features land lazy-only** — permanent and not conditional on anything: there is no eager path left to land on. **A DEFERRAL IS FINE IF IT IS RECORDED in `docs/gaps.md` with a tier and an owner** — the old blanket *no deferrals* ban lapsed when the eager retirement finished, and was SUPERSEDED BY THAT MECHANISM rather than merely dropped. ⚠️⚠️ **BUT THE MECHANISM IS WEAKER THAN THE RETIREMENT CLAIMED, AND THE CAVEAT IS LIVE: measured 2026-09-06 at `f10bdb79`, 236 open rows carry ELEVEN `**Owner:` markers and THERE IS NO OWNER COLUMN — every header shape is `| ID | File:Line | Tier | Gap | Status |` or its 4-column variant. So ownership is NOT mechanically answerable for 225 of 236 open rows. THE ROW IS GREPPABLE; THE OWNER IS NOT.** **The ruling stands — it is CireSnave's — and only its stated evidence was overclaimed.** ⚠️ **AND THE LESSON THE MISS CARRIES: the sentence that got this wrong was scrupulous about the NUMBER'S PRECISION and wrong about the PROPERTY — HEDGING A FIGURE DOES NOT VERIFY THE PREDICATE IT QUANTIFIES.** **Do not read any of this as "deferrals are now unremarked": the ban moved from prose into a registry that can be queried, and an UNRECORDED deferral is still the thing being forbidden.** → [`lazy-only-and-recorded-deferrals`](#lazy-only-and-recorded-deferrals)

## the-baracuda-dependency-and-the-cuda-build-recipe

`baracuda` (CUDA kernels) comes from **crates.io**, requested at **`0.0.1-alpha.78`** in the root manifest — ⚠️ **BUT THAT IS A MINIMUM, NOT A PIN, AND EVERY BARACUDA CRATE ACTUALLY RESOLVES TO `alpha.79` (measured in `Cargo.lock` 2026-08-26: ALL TWENTY-TWO).** A bare `version = "0.0.1-alpha.78"` is a CARET requirement — `>=0.0.1-alpha.78, <0.0.2` — so a newer pre-release satisfies it silently. **Only `baracuda-cuda-emit` uses an EXACT `=0.0.1-alpha.79`.** ⚠️ **CONSEQUENCE, AND IT IS THE ONE THAT BURNS: ANYONE REASONING ABOUT BARACUDA BEHAVIOUR FROM THE MANIFEST IS READING A VERSION THE BUILD DOES NOT USE — for all 22 crates, not one.** That is exactly how the `NUM_JOBS` accusation below survived 17 days. **Read `Cargo.lock`, never the manifest, when the question is "what does Fuel actually build".** — **⚠️ this line said `alpha.72` until 2026-08-15 while the bullet below it said `alpha.78`, i.e. CLAUDE.md contradicted itself on the pin and THE STALE HALF READ FIRST. Reported independently by the portfolio PM and the Lightbulb architect before anyone here noticed.** **⚠️ RETIRED 2026-08-20 — this bullet used to carry a live accusation against `baracuda-kernelgen`, and BOTH halves of it were dead. (a) The pin is GONE: `grep -c 'baracuda-kernelgen' Cargo.toml` → **0**; `14b7b4aa` migrated to `baracuda-cuda-emit = "=0.0.1-alpha.79"` (control: `baracuda-nvrtc`'s pin IS found, so the zero is absence rather than a broken query). (b) The defect was RETRACTED: **GAP-001 is closed and the all-zero output was never a version regression** — it was a `FusedOpId` collision plus nondeterministic selection, measured **11/20 failures at `.77` against 5/20 at `.78`**, i.e. the pinned "safe" version failed MORE. See Baracuda PR #17 `docs/gap-001-emitter-exoneration.md`. **The accusation outlived its own retraction by days, in the file that loads into every session** — if Baracuda's lane had read our working agreement they would have found us still blaming them. Deleted rather than softened.** A local `../baracuda` checkout is reference-only; a local `../baracuda` checkout is reference-only. **To check whether baracuda has a kernel, grep `baracuda-kernels-sys` (the FFI surface), NOT the plan facade.** (2026-07-03: `--features cuda` builds from a plain shell fail with `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` — NOT a CUDA-13.3/CUTLASS issue. Workaround until the next baracuda alpha (which self-resolves via vswhere): **build through `vcvarsall amd64` — do NOT hand-set `NVCC_CCBIN`.** Known-good **from the PowerShell tool**: `cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 && cargo build -p <crate> --features cuda'`. **From the Bash tool this exact command silently does nothing — use `cmd //c` instead (2026-08-05).** **Correction (2026-08-07): `cmd //c` is NOT a sufficient fix — treat this recipe as PowerShell-tool-ONLY.** `//c` fixes the path rewrite described next, but Bash *also* strips/mangles the inner double quotes around the `.bat` path, so the command exits **1** with ``'\"C:\Program Files\…\vcvarsall.bat\"' is not recognized as an internal or external command`` and **cargo is never invoked at all**. Two independent defects in one command line; fixing the first does not fix the second. The resulting bare `exit 1` reads exactly like "my change broke the CUDA build", which is how it cost a full diagnostic cycle. Git Bash's MSYS layer rewrites a leading `/c` argument to a Windows path before handing it to a native binary, so `cmd /c '<script> && cargo …'` arrives as `cmd C:/ '<script> && cargo …'` and opens an **interactive** cmd. Positive control, run in this environment: `python -c "import sys; print(sys.argv[1:])" /c x` prints `['C:/', 'x']`, while `//c` prints `['/c', 'x']`. Two failure modes observed on this machine the same day: a `--features cuda` check that **returned exit 0 having compiled nothing** (132 bytes of cmd banner, no `Checking` line), and an interactive cmd **holding the machine-wide `gpu-run` mutex indefinitely** — held, not abandoned, so recovery correctly never fires. **Never accept an exit code as evidence a CUDA build ran**: require a positive artifact only the real run emits (vcvarsall's `Environment initialized for: 'x64'` plus cargo's `Checking <crate>`). **⚠️ CORRECTED 2026-08-12 — THE `Checking <crate>` HALF OF THAT IS WRONG ON A WARM CACHE, and the error direction is "disbelieve a real success".** Cargo **omits** `Checking`/`Compiling` lines when everything is already built, so a genuinely-successful warm run emits **no** artifact and this rule as written verdicts it *the compiler never ran*. **Read artifacts from `--message-format json` instead** (found while building `scripts/feature-combination-check.ps1`, which had to solve exactly this). **⚠️ AND THE SIBLING SYMPTOM OF THE SAME ROOT, which bites any lint or warning MEASUREMENT: A WARM CACHE SUPPRESSES **WARNINGS** TOO — so counting diagnostics on a cached build measures NEAR-ZERO AND READS AS CLEAN.** Clear the crate fingerprint (`target[/<triple>]/debug/.fingerprint/<crate>-*`) before any warning census. Measured 2026-08-12 on GAP-176: with the fingerprint cleared, `E0133` came back **107 (x86)** and **21 (aarch64)**; without it the census would have reported almost nothing. **A warm cache hides the progress artifacts AND the diagnostics — the same failure direction both times: the wrong answer wearing good news.** **AND CHECK THE EXIT CODE BEFORE THE ARTIFACTS, not after — artifact-first misdiagnoses a real compile ERROR as a harness failure, because a failed compile and a never-attempted compile both produce no artifact.** The full truth table — **memorise the table, not the ordering rule; the table is what survives being half-remembered: `exit≠0 + artifact` = REAL FAILURE · `exit≠0 + no artifact` = HARNESS, never ran · `exit=0 + artifact` = REAL SUCCESS · `exit=0 + no artifact` = WARM CACHE **or** nothing attempted, and ONLY `--message-format json` separates those two.** Artifact-first collapses the top row — the same misdiagnosis the rule exists to prevent, pointed the other way. The hazard is the path-conversion layer, not the shell as such — the same `cmd /c` through the PowerShell tool is fine. **This rule applies to NONZERO exits too, and that half was being missed (2026-08-07).** It reads as a rule about false *greens*, so a red gate gets believed on sight — but every failure mode above produces a *red* gate that says nothing about the code, and the reflex is to blame the change under test. **An exit code of either polarity is evidence about the harness until an artifact proves the compiler ran on the code you think it did.** Related, and its own trap: **never pipe a background build through a filter** (`| Select-String …`) — the task's captured output file then holds the FILTERED stream, so the errors you need are discarded and unrecoverable, leaving an exit 101 with a two-line log. Capture unfiltered to a file; filter on read. And do not edit sources while a background build is reading them. **Corrected 2026-07-30:** this previously read "set `NVCC_CCBIN=<path-to-cl.exe>` **or** build from a VS Developer shell", and the first option is a trap whenever *more than one* MSVC toolset is installed. `NVCC_CCBIN` sets the *compiler* but not a matching `INCLUDE`/`LIB`, so aiming it at one toolset's `cl.exe` while `INCLUDE` still resolves another's headers gives a **mixed-toolset** environment and nvcc's front-end dies deep in the stdlib (`result_of<CallableType> is invalid`, `tuple index out of bounds`) — which reads like a CUDA/CUTLASS bug and is not one. `vcvarsall` sets compiler *and* headers *and* libs consistently, which is the actual requirement. **Toolset state churns — verify, never assume:** in one day this box went 14.29.30133 + 14.51.36231 → VS 2022 replaced by **VS 18 Community** carrying 14.44.35207 + 14.51.36231 + 14.52.36520 → an uninstall plus an unrequested Windows restart left **only 14.51.36231**. One toolset makes the mismatch temporarily impossible, but the next VS update re-adds toolsets, so the `vcvarsall` prescription stands unconditionally. Check with `Get-ChildItem '<VS root>\VC\Tools\MSVC'` rather than trusting a report of how many are installed — a claim that "only one remains, so the trap is structurally impossible" was relayed across sessions today while three were in fact present. Note also that the VS *root* moved (`...\Microsoft Visual Studio\2022\<ed>\` → `...\18\Community\`), so a path search rooted at the old location returns empty, and empty reads as "toolchain absent" when the truth is "toolchain moved". CUDA 13.3's `host_config.h` accepts `_MSC_VER` 1920–1959; 14.51 is 1951, so it is in range.) **SET `BARACUDA_FORGE_THREADS=6` ON ANY `--features cuda` BUILD *OR ANY `--workspace` INVOCATION* (2026-08-08). The `--workspace` half is NOT obvious and is the one that has actually bitten: `baracuda-kernels-sys`'s build script forges kernels UNCONDITIONALLY — it does not consult the `cuda` feature — and `--workspace` pulls in `fuel-cuda-backend`. So `cargo check --workspace --all-targets` with DEFAULT features IS a CUDA build**, and produced ~32 nvcc on a lane that had written CUDA nowhere on its command line. **A rule phrased as "on any `--features cuda` build" exempts exactly that invocation**, which is how the first version of this rule was wrong. **Note the bind: the workspace-scope gate that enum-variant additions are REQUIRED to run is itself one of these builds.** Related trap worth stating once: **"doesn't touch the GPU" and "is cheap on this box" are INDEPENDENT properties** — that enumeration needed no `gpu-run` and touched the 4070 not at all, while being the heaviest thing on the machine.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

  - `baracuda` (CUDA kernels) comes from **crates.io**. ⚠️ **READ `Cargo.lock`, NEVER THE MANIFEST, when the question is "what does Fuel actually build": the root manifest REQUESTS `0.0.1-alpha.78`, which is a CARET requirement (`>=0.0.1-alpha.78, <0.0.2`), and ALL TWENTY-TWO baracuda crates actually RESOLVE to `alpha.79`. Anyone reasoning about baracuda behaviour from the manifest is reading a version the build does not use — that is how a false accusation against a sibling project survived 17 days in the file that loads into every session.** A local `../baracuda` checkout is **reference-only**. **To check whether baracuda has a kernel, grep `baracuda-kernels-sys` (the FFI surface), NOT the plan facade.** ⚠️ **`--features cuda` FROM A PLAIN SHELL FAILS `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` — build through `vcvarsall amd64`, and do NOT hand-set `NVCC_CCBIN`**: it sets the COMPILER but not a matching `INCLUDE`/`LIB`, so on a multi-toolset box nvcc's front-end dies deep in the stdlib and reads like a CUDA/CUTLASS bug. **CHEAPEST FORM, AND IT NEEDS NO QUOTING AT ALL — isolate vcvarsall in a throwaway subprocess instead of putting it on a command line: `cmd /c "vcvarsall amd64 && set"` to harvest the environment, `Set-Item env:` per line, then run cargo from a shell that never had vcvarsall in it.** ⚠️ **IF YOU DO PUT IT ON A COMMAND LINE IT IS POWERSHELL-TOOL-ONLY: Git Bash's MSYS layer mangles BOTH the leading `/c` AND the inner quotes around the `.bat` path — two independent defects in one command line — and the result is `exit 1` WITH CARGO NEVER INVOKED, which reads exactly like *my change broke the CUDA build*.** **SET `BARACUDA_FORGE_THREADS=6` ON ANY `--features cuda` BUILD *OR ANY `--workspace` INVOCATION*** — `baracuda-kernels-sys`'s build script forges UNCONDITIONALLY and does not consult the `cuda` feature, so `cargo check --workspace` with DEFAULT features **IS a CUDA build**, and a rule phrased as *on any `--features cuda` build* exempts exactly that invocation. ⚠️⚠️ **NEVER ACCEPT AN EXIT CODE AS EVIDENCE A CUDA BUILD RAN, IN EITHER POLARITY — AND MEMORISE THE TABLE, NOT THE ORDERING RULE, BECAUSE THE TABLE IS WHAT SURVIVES BEING HALF-REMEMBERED: `exit≠0 + artifact` = REAL FAILURE · `exit≠0 + no artifact` = HARNESS, never ran · `exit=0 + artifact` = REAL SUCCESS · `exit=0 + no artifact` = WARM CACHE **or** nothing attempted, and ONLY `--message-format json` separates those two.** **A WARM CACHE SUPPRESSES `Checking` LINES *AND* WARNINGS, so a diagnostic census on a cached build measures near-zero and reads as CLEAN — clear the crate fingerprint before any warning census.** **And never pipe a background build through a filter: the task's captured output file then holds the FILTERED stream, so the errors are discarded and unrecoverable.** → [`the-baracuda-dependency-and-the-cuda-build-recipe`](#the-baracuda-dependency-and-the-cuda-build-recipe)

## all-targets-is-not-all-features-and-the-converse

**`--all-targets` DOES NOT IMPLY `--all-features`, and a non-default feature can hide an exhaustive `DType`/`Scalar` match COMPLETELY (2026-08-08).** ⚠️⚠️ **AND THE CONVERSE BITES TOO — `cargo test --workspace` UNIFIES FEATURES ACROSS THE SELECTED GRAPH *INCLUDING DEV-DEPENDENCIES*, SO A `#[cfg(feature = "…")]` GATE IS NOT EVIDENCE THAT CI SKIPS A TEST (2026-08-14).** A lane predicted its `vulkan`-gated test could not run in CI — the test's own comment said *"a CI machine without a GPU never builds it"* — and it ran on every macOS job. The chain is `fuel-vulkan-backend` -> `fuel-dispatch/vulkan` -> `fuel-core/vulkan`, **pulled in through a DEV-dependency**, so selecting the workspace turns the feature on for everyone. **Measured with `cargo tree -e features --workspace`, which is the instrument — and the lane ran it AFTER being contradicted rather than before asserting.** **READING A `cfg` GATE AND CONCLUDING COVERAGE IS THE SAME ERROR IN BOTH DIRECTIONS: it hides code you think is built, and builds code you think is skipped.** **AND THE PRODUCTION INSTANCE IS THE ARGUMENT FOR GAP-157 IN ONE SENTENCE: that test had been COMPILED AND RUN on every macOS CI job, found no Vulkan loader, returned early, and reported `ok` HAVING ASSERTED NOTHING — a green that lied, in CI, for an unknown window, and nothing would ever have surfaced it BECAUSE IT PASSED.** Converting the silent early return to a hard fail is what made it visible. **A test that cannot run must SAY SO (declared gate / `ignored`), never return `ok`.** ⚠️⚠️ **AND A GUARD AGAINST PROSE HEDGING CANNOT LEXICALLY DISTINGUISH A HEDGE FROM A BOUNDARY STATEMENT — THEY SHARE VOCABULARY, AND THE SECOND IS SOMETHING THIS WORKING AGREEMENT REQUIRES (2026-08-14).** `fuel-ir/tests/gap_hedges.rs` (GAP-141) fires on 14 prose patterns. It flagged two lines, and **neither was a hedge**: `mnist_train.rs`'s module doc (*"behaviour across platforms/backends is NOT yet established and must not be assumed"* — a sentence the ARCHITECT REQUIRED, so a reader cannot over-cite the convergence oracle) and `persistent_decode.rs`'s `per_layer` comment (*"a scaled multi-base model would need per-variant inv_freq, and no such model exists yet"* — a documented scope boundary with a stated trigger, followed by an invariant claim about what actually reaches the code). **THE DISTINCTION: A HEDGE SUBSTITUTES PROSE FOR WORK (*"should probably be fine"*) AND HIDES THAT NOBODY MEASURED. A BOUNDARY STATEMENT RECORDS A MEASURED LIMIT AND EXISTS *BECAUSE* SOMEBODY MEASURED — it tells you where the measurement stops.** A pattern match sees only the words. **SO THE DISTINCTION NEEDS A DURABLE HOME, WHICH MAKES THE CHOICE OF MECHANISM LOAD-BEARING RATHER THAN ADMINISTRATIVE.** ⚠️⚠️ **CORRECTED 2026-09-02 — THIS RULE USED TO SAY THE ALLOWLIST IS *"THE ONLY PLACE THAT DISTINCTION CAN LIVE"* AND THAT *"EVERY ENTRY MUST CARRY A REASON"*. BOTH ARE MEASURABLY FALSE, AND THE SECOND IS AN INSTRUCTION THAT CANNOT BE OBEYED.** `fuel-ir/tests/gap_hedge_allowlist.txt` is bare `<rel-path>` TAB `<trimmed comment text>` rows — **209 entries, ZERO comment lines, and a `#` line would be parsed as a KEY and then reported STALE, reddening the gate.** There is nowhere for a reason to go. **Found by Fuel 2 while COMPLYING with the rule, which is the only way an unobeyable instruction gets discovered.** ⚠️ **AND THE SAME SENTENCE MISDESCRIBED THE FORMAT IN THE FLATTERING-TO-ITSELF DIRECTION: it warned that *"an allowlist of bare LINE references degrades into noise"*. The entries are NOT line references.** The key is path plus comment TEXT (`gap_hedges.rs`, the `key` helper and its `format!("{rel_path}\t{trimmed}")` site), so it survives code moving above it, and any key no longer found in the tree FAILS the gate as STALE — **which is the rename-resistant, self-retiring form this working agreement elsewhere prescribes. The design was already right and the rule gave it no credit.** ⚠️⚠️ **THE RULING (2026-09-02): A BOUNDARY STATEMENT TAKES `// GAP(GAP-NNN)`, NOT THE ALLOWLIST. The allowlist stays a shrink-only baseline of genuine PRE-EXISTING HEDGE DEBT.** Three reasons, and the third is why it reconciles rather than picks a side: **(1)** the GAP path is the only one of the guard's three outs where a REASON can live durably, because a registry row can hold one and a TAB-separated key cannot; **(2)** the guard's own header calls the allowlist *"a checked-in baseline of every hedge that ALREADY EXISTED at authoring time"* with "may only shrink" teeth making its size *"a real, monotonically-shrinking metric of outstanding prose debt"* — **so adding a NEW boundary statement to it corrupts the metric, and the header and this file were pointing opposite ways**; **(3)** the hedge-vs-boundary distinction then lives in WHICH MECHANISM WAS USED — allowlisted means "a real hedge, tracked as debt", GAP-referenced means "a boundary statement with a home" — **and both are greppable without any comment column.** ⚠️ **NOTE THE TRACKED PATH IS UNVALIDATED: the guard tests `line.contains("GAP(GAP-")`, a plain substring, so `GAP(GAP-99999)` passes too. Convenient (you may cite a row before its PR lands) and a real weakness — the reference is only as good as the discipline that files the row.** **Worked example: GAP-274 exists precisely so a measured E4M3 encoding boundary in `dyn_impl.rs` could be TRACKED rather than allowlisted.** **STILL TRUE AND UNCHANGED: an allowlist that accumulates entries nobody can classify degrades into noise and takes the guard's signal with it — the very thing it was built to prevent, arriving through its own escape hatch.** And **if honest boundary statements need allowlisting ROUTINELY, the pattern set is too broad and that is a finding, not a chore.** **Second-order hazard, observed the same day: DELETING CODE ORPHANS ALLOWLIST ENTRIES.** Retiring `fuel-wasm-examples` left 4 entries pointing at files that no longer exist, and the guard went red for reasons unrelated to any hedge. **A deletion sweep must include the allowlists and baselines that reference the deleted paths** — same class as the dangling FKC mdBook references the same deletion left behind. ⚠️⚠️ **SEPARATELY, A CI-MATRIX RULE WORTH ITS OWN LINE: SET `fail-fast: false` ON ANY TEST MATRIX. Fuel's `check` job had it and `test` did not, so on the first run where the suite could execute at all, macOS failed and ubuntu + windows were CANCELLED MID-RUN — reporting `cancelled`, which is NOT A RESULT IN EITHER DIRECTION.** **Test failures are usually platform-specific, so cancelling a failure's siblings discards THE SINGLE MOST INFORMATIVE COMPARISON AVAILABLE: did this break everywhere, or only here?** A triage scoped to one OS because the other two were killed is honest but a third of the answer, **and the missing two-thirds is exactly the part that distinguishes a platform quirk from a real defect.** A dtype addition was verified with `--all-targets` across six crates and reported clean; **`fuel-dispatch`'s `telemetry` feature is non-default**, so `telemetry/structure_key_derive.rs` — which carries an exhaustive `DType` match and the sk4 wire codec — **was never compiled by any of it.** It was a 20th `E0004` site **structurally invisible to the enumeration**, not a leg anyone skipped. **Caught only because a test filter printed `running 0 tests; 737 filtered out` and the worker refused to read that as a pass.** **The feature space that can gate per-dtype code (enumerate it, do not rediscover it):** `fuel-dispatch` — **telemetry**, jit, jit-synth, cuda, vulkan, metal; `fuel-ir` — **dlpack**, ug, serde; `fuel-memory` — cuda, vulkan, metal, dlpack; `fuel-cpu-backend` — **mkl**, **accelerate**; `fuel-core` — telemetry, cuda, cudnn, nccl, mkl, aocl, onemkl, accelerate, metal, vulkan, ug; `fuel-correctness-fixtures` — serde, capture. ⚠️⚠️ **CORRECTED 2026-08-27 — THIS CLAIM WAS FALSE FOR `mkl` AND `aocl`, AND THE FALSE CLAIM IS WHAT LET TWO CRATES ROT.** It read: *"`mkl`/`accelerate`/`aocl`/`onemkl` are BOTH invisible AND unbuildable here (DLLs absent, GAP-016)."* **MEASURED: BOTH BUILD ON THIS BOX.** oneAPI MKL is installed at `C:\Program Files (x86)\Intel\oneAPI\mkl\latest`; AOCL builds from the `../aocl` sibling. **The claim is right about CI and wrong about this machine — and the distinction is not pedantic, it is the whole mechanism: A WRONG "UNBUILDABLE" CLAIM HIDES ROT INDEFINITELY, BECAUSE NOBODY TRIES.** **Two crates were found rotted the same day, both by lanes who needed the feature for an unrelated reason** — `fuel-aocl-cpu-backend` (3 test targets uncompilable since the never-panic migration, GAP-247) and **`fuel-mkl-cpu-backend`, ENTIRELY uncompilable INCLUDING ITS LIBRARY since `m_compute` was added to `OpParams::Matmul`** (**`E0027`** — a non-exhaustive `match` — in `binding_table.rs`'s **lib**, and **`E0063`** — a missing field in a struct literal — in its **lib test**; both on `m_compute`, both greppable by error code without a line number). **Neither is a `default-member`; both are CI-excluded; nothing had compiled either.** **And the MKL breakage was not cosmetic:** `matmul_f32_mkl` handed `m` straight to a BLAS gemm with no row-limit concept, **so for any `MatmulM` but `All` it would compute rows past the valid data of a sparse-MoE batch and write plausible-looking garbage.** It now declines loudly, exhaustively matched. **I would not assume there are only two.** **THE CORRECT CATEGORIES: `mkl`/`aocl` — INVISIBLE TO CI, BUILDABLE HERE (so cheap local gates, and rot-prone precisely because the old claim said not to bother) · `accelerate` — Apple-only, genuinely unbuildable here · CUDA — buildable, expensive, unrun · Metal — wrong platform.** **"Unbuildable in CI" and "unbuildable anywhere" have different lifetimes and different remedies, and collapsing them cost two crates.** **One negative result worth keeping so it is not re-derived:** `serde` is NOT a hidden `E0004` site for `DType` — `fuel-ir/src/dtype.rs`'s `DType` declaration uses `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]`, a **derive**, which absorbs a new variant silently. That is a *wire-compatibility* concern, not an enumeration gap.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`--all-targets` DOES NOT IMPLY `--all-features`, and a non-default feature can hide an exhaustive `DType`/`Scalar` match COMPLETELY.** ⚠️⚠️ **AND THE CONVERSE BITES TOO: `cargo test --workspace` UNIFIES FEATURES ACROSS THE SELECTED GRAPH *INCLUDING DEV-DEPENDENCIES*, SO A `#[cfg(feature = "…")]` GATE IS NOT EVIDENCE THAT CI SKIPS A TEST.** **READING A `cfg` GATE AND CONCLUDING COVERAGE IS THE SAME ERROR IN BOTH DIRECTIONS: it hides code you think is built, and builds code you think is skipped.** **THE INSTRUMENT IS `cargo tree -e features --workspace`** — run it BEFORE asserting, not after being contradicted. ⚠️ **AND A TEST THAT CANNOT RUN MUST SAY SO (declared gate / `ignored`) AND MUST NEVER RETURN `ok`**: a `vulkan`-gated test compiled and ran on every macOS job, found no loader, returned early and reported `ok` HAVING ASSERTED NOTHING — a green that lied, in CI, for an unknown window, and nothing would ever have surfaced it BECAUSE IT PASSED. ⚠️ **BOUNDARY STATEMENTS TAKE `// GAP(GAP-NNN)`, NEVER THE HEDGE ALLOWLIST** — the allowlist is a SHRINK-ONLY baseline of pre-existing hedge DEBT, so adding a new boundary statement to it corrupts the metric; a registry row can hold a REASON and a TAB-separated key cannot; and the hedge-vs-boundary distinction then lives in WHICH MECHANISM WAS USED, greppable either way. **A HEDGE SUBSTITUTES PROSE FOR WORK AND HIDES THAT NOBODY MEASURED; A BOUNDARY STATEMENT RECORDS A MEASURED LIMIT AND EXISTS *BECAUSE* SOMEBODY MEASURED — a pattern match sees only the words, so no guard can separate them.** (For `fail-fast`, see the workflow bullet below, which carries BOTH layers — the matrix flag AND `--no-fail-fast` — where this bullet used to carry only the first.) **THE FEATURE SPACE THAT CAN GATE PER-DTYPE CODE — ENUMERATE IT, DO NOT REDISCOVER IT:** `fuel-dispatch` — **telemetry**, jit, jit-synth, cuda, vulkan, metal; `fuel-ir` — **dlpack**, ug, serde; `fuel-memory` — cuda, vulkan, metal, dlpack; `fuel-cpu-backend` — **mkl**, **accelerate**; `fuel-core` — telemetry, cuda, cudnn, nccl, mkl, aocl, onemkl, accelerate, metal, vulkan, ug; `fuel-correctness-fixtures` — serde, capture. ⚠️⚠️ **AND THE CATEGORIES MATTER MORE THAN THE LIST, BECAUSE A WRONG "UNBUILDABLE" CLAIM HIDES ROT INDEFINITELY — NOBODY TRIES: `mkl`/`aocl` are INVISIBLE TO CI BUT BUILDABLE HERE (cheap local gates, and rot-prone precisely because this file once said not to bother — two crates were found entirely uncompilable that way) · `accelerate` is Apple-only and genuinely unbuildable here · CUDA is buildable, expensive, unrun · Metal is the wrong platform. "UNBUILDABLE IN CI" AND "UNBUILDABLE ANYWHERE" HAVE DIFFERENT LIFETIMES AND DIFFERENT REMEDIES.** **One negative result kept so it is not re-derived: `serde` is NOT a hidden `E0004` site for `DType` — the declaration uses `#[cfg_attr(feature = "serde", derive(…))]`, a DERIVE, which absorbs a new variant silently. That is a wire-compatibility concern, not an enumeration gap.** → [`all-targets-is-not-all-features-and-the-converse`](#all-targets-is-not-all-features-and-the-converse)

## the-workspace-check-and-its-exclusions

**`cargo check --workspace` CANNOT RUN on this box, and each way it fails looks EXACTLY like success (2026-08-08).** Three consecutive aborts each reported **`E0004: 0`** with jobs still queued: **(1) `objc2` — "only works on Apple platforms"**, because `fuel-metal-backend` is pulled in by `--workspace` on a non-Apple runner **⚠️ (CORRECTED 2026-08-15: this previously said it is "deliberately kept OFF `default-members`" — MEASURED FALSE. It IS listed in `default-members`, so a bare local `cargo check` tries to build it too. What makes CI work is the explicit `--exclude`, NOT a membership choice — and I cited the wrong fact while rewriting the workspace-gate rule the same day.)**; **(2) `onemkl-sys`** build-script failure (`fuel-mkl-cpu-backend`, oneMKL absent); (3) only with both excluded did the real list appear. **An aborted enumeration and a completed one are byte-identical at the top.** So: **read the HALT LINE, never the error count** — a toolchain gap otherwise reads as "nothing left to fix." **⚠️ RE-MEASURED 2026-08-20 — THE EXCLUSION LIST AND ITS STATED CAUSES ARE BOTH WRONG AT HEAD. The exclusions were correct when written; two of the four factual components have since dissolved.** Measured one crate at a time, each with its own `Checking <crate>` line: **`fuel-metal-kernels` — EXCLUDE, measured: fails `could not compile objc2`. THIS is the crate the `objc2` obstacle belongs to.** **`fuel-metal-backend` — DOES NOT NEED EXCLUDING: builds clean on Windows, exit 0.** And the rule's stated premise — *"deliberately kept OFF `default-members` so a plain build works without Apple toolchains"* — is **FALSE at head**: it IS in `default-members` (27 entries, parsed from the manifest). **The premise is the part a future reader would otherwise re-derive, so it is named as false rather than quietly dropped.** **`fuel-mkl-cpu-backend` — STATUS UNMEASURED (timed out at 7 min), and the stated reason is wrong:** oneAPI **is** installed at `C:/Program Files (x86)/Intel/oneAPI` and the build script SUCCEEDED, emitting `cargo:rustc-link-lib=…mkl_core_dll`. **"oneMKL absent" is retired; "builds" is NOT claimed.** **`fuel-aocl-cpu-backend` — a workspace member the rule never mentioned, unmeasured.** A silent omission in a rule whose whole job is enumerating exclusions. **Completing this needs a `--workspace` check, which IS a CUDA forge — a C, not a B. Ride it along with someone's next forge; do not justify one.** **⚠️ THE SAME FALSE PREMISE IS ALSO IN `.github/workflows/rust-ci.yml`'s header comment**, which says both metal crates are *"deliberately kept OFF `default-members`"*. Not corrected here — flagged, since that file is operational config. **The general lesson: a rule written because a tool was broken, kept after the tool was fixed, reads identically to one written because the thing is dangerous — and what made THIS one auditable is that it stated its reasons with unusual precision. A vaguely-right rule ("workspace checks are problematic here") would have survived this audit untouched while being just as wrong. Write the precondition even though it will rot.** The gate as previously written was: `cargo check --workspace --all-targets --exclude fuel-metal-backend --exclude fuel-metal-kernels --exclude fuel-mkl-cpu-backend`, **and it must be reported together with the statement that the Metal and MKL arms are UNVERIFIABLE on this machine** (Metal needs a Mac or `--target aarch64-apple-darwin`). This matters most for **enum-variant additions**, where the workspace-scope rule exists precisely to enumerate consumers: `fuel-metal-backend/src/storage.rs` does match on dtypes, so a variant addition almost certainly breaks a Metal site nobody here can see, let alone gate. **Never claim "workspace-clean" without naming the exclusions. ⚠️⚠️⚠️ **AND THE INCANTATION ABOVE IS INCOMPLETE — IT WAS MEASURED ON THIS BOX, WHICH IS THE ONE PLACE IT CANNOT BE MEASURED (2026-08-14, found by restoring CI). It names `fuel-metal-backend`, `fuel-metal-kernels` and `fuel-mkl-cpu-backend`, and OMITS `fuel-aocl-cpu-backend` FOR NO REASON EXCEPT THAT THE AUTHOR'S MACHINE HAD AOCL INSTALLED** (the `../aocl` sibling), so that crate compiles here and panics in a clean environment — `aocl-build/src/lib.rs`'s `"Could not locate AOCL"` panic. **A LOCAL `--workspace` GREEN IS NOT EVIDENCE ABOUT CI.** **THE MEASURED-FROM-CI SET, GROUPED BY *CAUSE* BECAUSE THE CAUSES HAVE DIFFERENT LIFETIMES: WRONG-PLATFORM = `fuel-metal-backend` + `fuel-metal-kernels` (need an Apple target; must stay INCLUDED on macOS or nothing builds Metal anywhere). MISSING-SDK = `fuel-mkl-cpu-backend` (oneMKL) + `fuel-aocl-cpu-backend` (AOCL) — excluded on EVERY runner including macOS, and that half must stay in sync across all lists. COST, NOT MISSING-TOOLCHAIN — ⚠️ **RELABELLED 2026-08-26; THE OLD LABEL SAID `MISSING-TOOLCHAIN` AND ITS OWN JUSTIFICATION SAID OTHERWISE.** = `fuel-cuda-backend`, because `baracuda-kernels-sys`'s build script FORGES UNCONDITIONALLY (it does not consult the `cuda` feature), so mere workspace MEMBERSHIP drags the forge into a build that enables no CUDA feature at all — one `--exclude` removes it, and the consequence, STATED IN THE WORKFLOW HEADER RATHER THAN HIDDEN BEHIND A GREEN BADGE, IS THAT NO CI JOB COMPILES FUEL'S CUDA CODE.** ⚠️ **THE CUDA TOOLKIT IS INSTALLABLE ON AN UBUNTU RUNNER (measured by Baracuda: nvcc via apt or a setup action), SO "we cannot" WAS NEVER THE REASON. The reason is ~56 MINUTES OF FORGE PER RUN at `BARACUDA_FORGE_THREADS=6` (**93m measured under load 2026-08-27, so the CI cost argument is if anything understated**), which is a COST decision with a different remedy and a DIFFERENT EXPIRY than an availability one** — a missing SDK is closed by installing it; a cost is closed by the cost changing, or by someone deciding to pay it. **INSTALLABLE AND AFFORDABLE ARE NOT THE SAME LIFETIME, and the old label asserted the wrong one — flagged by the portfolio PM when a sibling project's coverage-matrix convention was about to inherit the mislabel from us.** ⚠️ **AND THE "COMPILE-ONLY CUDA JOB" HOPE THIS FILE RECORDED AS *possibly viable* IS NOW MEASURED AND IT IS **NOT** VIABLE FOR `fuel-cuda-backend`: `baracuda-kernels-sys` is a MANDATORY dependency of that crate (only `cudnn`/`cudnn-sys`/`nccl` are `optional = true`), and `cargo check` RUNS BUILD SCRIPTS — so there is no feature combination that compiles Fuel's CUDA backend without forging.** **The narrower case remains open and UNDEMONSTRATED and must not be confused with it: `fuel-dispatch --features telemetry,baracuda-types` is a DIFFERENT TARGET that does not pull `baracuda-kernels-sys` at all.** Those two were being discussed as one thing; **they are separate questions with separate answers, and only the second is still live.** **⚠️ THE CLASS, AND IT IS THE REUSABLE PART: *THINGS THIS MACHINE HAS THAT A CLEAN RUNNER DOES NOT.* FIVE CONSECUTIVE FIXES WERE UNVERIFIABLE LOCALLY BECAUSE THE FAILURE COULD NOT REPRODUCE HERE — a private git rev (warm `~/.cargo/git` + credentials), AOCL, nvcc, protoc, and the `objc2` case in reverse. THE ONLY INSTRUMENT FOR THAT CLASS IS CI ITSELF, and a fix must be REPORTED as not-locally-gated rather than as checked.** The one local approximation that works is a **cold `CARGO_HOME` with `GIT_TERMINAL_PROMPT=0` and an empty global gitconfig** — reproducing the environment instead of caveating past it — but it only covers the *resolution* half. **AND THE COUNTER-PRESSURE THAT KEEPS THE GATE HONEST, worth more than the list: PREFER INSTALLING A MISSING TOOL OVER EXCLUDING A CRATE WHENEVER THE TOOL IS ACTUALLY INSTALLABLE.** `protoc` is a small package available on every runner, so installing it KEEPS A REAL CRATE COMPILED for a few seconds of CI time, where excluding `fuel-onnx` would have cost coverage permanently. **Without that rule an exclusion list drifts into *"workspace minus everything that needs a toolchain"* — and NOBODY NOTICES, BECAUSE EACH INDIVIDUAL STEP IS REASONABLE.** Excluding is for platform-impossible or heavyweight-SDK cases only, and **every exclusion costs real coverage that must be named where a reader of the green will see it.****

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **`cargo check --workspace` RUNS ON THIS BOX ONLY WITH A NAMED EXCLUSION SET, AND EACH WAY IT FAILS LOOKS EXACTLY LIKE SUCCESS.** ⚠️ **READ THE HALT LINE, NEVER THE ERROR COUNT: an aborted enumeration and a completed one are BYTE-IDENTICAL AT THE TOP** — three consecutive aborts each reported `E0004: 0` with jobs still queued, and a toolchain gap otherwise reads as *nothing left to fix*. ⚠️⚠️ **THE MEASURED-FROM-CI EXCLUSION SET, GROUPED BY *CAUSE* BECAUSE THE CAUSES HAVE DIFFERENT LIFETIMES AND DIFFERENT REMEDIES — a missing SDK is closed by installing it, a COST is closed only by the cost changing or by someone deciding to pay it:** **WRONG-PLATFORM** = `fuel-metal-backend` + `fuel-metal-kernels` (need an Apple target; must stay INCLUDED on macOS or nothing builds Metal anywhere). **MISSING-SDK** = `fuel-mkl-cpu-backend` (oneMKL) + `fuel-aocl-cpu-backend` (AOCL) — excluded on EVERY runner including macOS, and that half must stay in sync across all lists. **COST, NOT MISSING-TOOLCHAIN** = `fuel-cuda-backend`, because `baracuda-kernels-sys`'s build script FORGES UNCONDITIONALLY and does not consult the `cuda` feature, so mere workspace MEMBERSHIP drags the forge into a build enabling no CUDA feature at all. **The CUDA toolkit IS installable on an Ubuntu runner, so *we cannot* was never the reason — the reason is ~56 min of forge per run (93m under load), and the CONSEQUENCE, which belongs in the workflow header rather than behind a green badge, is that NO CI JOB COMPILES FUEL'S CUDA CODE.** ⚠️ **A compile-only CUDA job is MEASURED NOT VIABLE for `fuel-cuda-backend` — `baracuda-kernels-sys` is a MANDATORY dep and `cargo check` RUNS BUILD SCRIPTS, so no feature combination compiles that backend without forging. The narrower `fuel-dispatch --features telemetry,baracuda-types` case is a DIFFERENT TARGET that pulls none of it, and is still open and undemonstrated — do not conflate them.** ⚠️⚠️ **A LOCAL `--workspace` GREEN IS NOT EVIDENCE ABOUT CI, AND THE CLASS IS THE REUSABLE PART: *THINGS THIS MACHINE HAS THAT A CLEAN RUNNER DOES NOT.* Five consecutive fixes were unverifiable locally because the failure could not reproduce here (a warm `~/.cargo/git` plus credentials, AOCL, nvcc, protoc, and the `objc2` case in reverse). THE ONLY INSTRUMENT FOR THAT CLASS IS CI ITSELF, and such a fix must be REPORTED AS NOT-LOCALLY-GATED rather than as checked.** The one local approximation is a cold `CARGO_HOME` with `GIT_TERMINAL_PROMPT=0` and an empty global gitconfig, and it covers only the RESOLUTION half. ⚠️ **AND THE COUNTER-PRESSURE THAT KEEPS THE GATE HONEST, WORTH MORE THAN THE LIST: PREFER INSTALLING A MISSING TOOL OVER EXCLUDING A CRATE WHENEVER THE TOOL IS ACTUALLY INSTALLABLE.** `protoc` is a small package on every runner, so installing it KEEPS A REAL CRATE COMPILED, where excluding `fuel-onnx` would have cost coverage permanently. **Without that rule an exclusion list drifts into *workspace minus everything that needs a toolchain* — AND NOBODY NOTICES, BECAUSE EACH INDIVIDUAL STEP IS REASONABLE.** **NEVER CLAIM "WORKSPACE-CLEAN" WITHOUT NAMING THE EXCLUSIONS: every exclusion costs real coverage, and that cost must be named where a reader of the green will see it.** **This matters most for ENUM-VARIANT ADDITIONS, where the workspace-scope rule exists precisely to enumerate consumers — `fuel-metal-backend/src/storage.rs` matches on dtypes, so a variant addition almost certainly breaks a Metal site nobody here can see, let alone gate.** → [`the-workspace-check-and-its-exclusions`](#the-workspace-check-and-its-exclusions)

## never-panic-and-what-closing-one-family-did-not-close

**Never panic on production paths.** `Result` from day one. (⚠️ **THE STANDING VIOLATION THIS PARENTHETICAL WAS BUILT AROUND — `NodeHandle::from_*` panicking on a length mismatch — IS CLOSED as of 2026-09-09, `62ac2991`, GAP-003 / PR #173. The history is kept because its CITATION-ROT lessons are the durable part; read every present-tense claim inside it as dated, not as current.** Citation history — **corrected 2026-08-20, it was stale in THREE particulars at once**: the type is now `NodeHandle` not `Tensor` (renamed by `cf861588`), the construct is an **`assert_eq!`** not an `.expect()`, and it lives in **`from_host_buffer_on`, `fuel-graph/src/lib.rs`** — not `~2256`, where there is no panic at all. `NodeHandle::from_f32` and siblings still return `Self` and delegate there, so **the violation itself stood at that date**; only every detail of how to find it was wrong. Note the panic's own MESSAGE was updated to say `NodeHandle::from_*` by the rename sweep — **the code was maintained and the reference to it was not**, which is why a normative rule's embedded citation needs checking even when the rule is eternal. **⚠⚠ THE VIOLATION IS FIXED AS OF 2026-09-09, `62ac2991` (GAP-003, PR #173 — 20 commits, 100 files, 10 crates). THIS SENTENCE PREVIOUSLY ASSERTED THE VIOLATION WAS REAL AND UNFIXED, AND IS REPLACED RATHER THAN ANNOTATED (the old wording is deliberately NOT quoted verbatim: quoting a retired claim to explain it leaves it greppable, so anyone checking whether the claim is gone would hit the correction), because an annotation leaves the falsehood on the page for a skimmer.** MEASURED at that commit: `NodeHandle::from_f32` and `from_f64` return `Result<Self, fuel_ir::Error>`; the prose anchor's two PRODUCTION hits are now `if n != shape.elem_count() { return Err(…) }` — one in `from_host_buffer_on`, one in the `const_*_like` family — and its third hit is a RETAINED born-red, `gap003_from_f32_declines_a_length_mismatch`, asserting the decline BY MESSAGE. ⚠️ **NO LINE NUMBERS, AND THE FIRST DRAFT OF THIS CORRECTION CARRIED THREE: they were already off by two before the commit was an hour old, in the one bullet that measures line anchors at 67% defective — and `fuel-ir/tests/claude_md_line_anchors.rs` would have reddened on the single one of the three that carried a filename. The gate caught its own author inside the sentence correcting a stale citation.** ⚠️ **THE PROSE ANCHOR `does not match shape element count` WAS DELIBERATELY PRESERVED THROUGH THE CONVERSION** — the source says *“rename-proof, so changing the words here would orphan two citations that no identifier rename could have touched”* — so this citation and `docs/gaps.md`'s both still resolve. **A conversion free to reword its own message and choosing not to, because two documents anchor on it, is `cite-what-cannot-move` applied in the direction almost nobody does.** ⚠️ **AND THIS CLOSES ONE CONSTRUCTOR FAMILY, NOT THE PRINCIPLE: GAP-308 measures 475 production panics in `fuel-transformers/src/models/`, and GAP-309 records two `Result`-returning public builders that still inherit 8 unguarded dtype/shape panics each. Do not read this correction as the rule being satisfied.** ⚠️ **UNTIL `62ac2991` this sentence read that `NodeHandle::from_f32`/`from_f64`/`from_bf16`/… all return `Self`, delegate to `from_host_buffer_on`, and that `from_host_buffer_on` panics on a length mismatch. **It is kept as a dated HISTORICAL statement rather than deleted — sweeping a historical mention destroys the record of what the bullet used to assert — but it is no longer phrased as an assertion.** It is the FOURTH representation of the retired claim in this one bullet, and the first pass of this correction replaced only the most prominent one — leaving three statements of a false claim behind a sentence announcing it false. ONE CLAIM CAN LIVE IN AN OPENING FRAME, A PRESENT-TENSE ASIDE, A HEADLINE AND A CLOSING RESTATEMENT; ENUMERATE THE REPRESENTATIONS BEFORE FIXING ANY.** The structural half is still true and is why the anchor discussion below works: the `from_*` family delegates to `from_host_buffer_on`, so ONE decline covers the whole family. **Re-derive with a RENAME-RESISTANT anchor, not a symbol name** — the citation above broke precisely because it named a type: `git grep -n 'does not match shape element count' -- fuel-graph/src/lib.rs` → **TWO PRODUCTION hits at the time — AND THAT WAS A LOWER BOUND ON THE PANIC POPULATION, NOT A COUNT OF IT**: one in `from_host_buffer_on` (behind `NodeHandle::from_f32`/`from_f64`/…) and one in the `const_*_like` family. **At `62ac2991` the same anchor returns THREE, the third being the born-red that asserts the decline — so a bare count off this anchor now mixes production sites with their own regression guard.** ⚠️⚠️ **MEASURED 2026-09-06 BY THE GAP-003 LANE — THIS ANCHOR FINDS ONE OF THREE PANIC CONSTRUCTS AT EACH SITE.** It sees the `assert_eq!` and is **blind** to an `.expect("NodeHandle::…")` five-to-eight lines below and to a `.write().unwrap()` — the latter a **poisoned-lock panic, a DIFFERENT obligation with a different remedy (198 in that one file), which must NOT be swept into this row**: a poisoned lock means another thread has *already* panicked, so a `Result` there propagates an unrecoverable condition and buys the caller nothing. ⚠️ **THE PROSE ANCHOR IS EXACTLY AS NARROW AS IT IS DURABLE, and it fails FLATTERINGLY — real hits, right file, real violations, neighbours omitted, and nothing in the output says it is a sample.** A *rotted symbol* anchor fails loudly and gets re-derived; **a durable message anchor fails silently.** → [`an-anchor-buys-immunity-and-pays-in-blindness`](docs/method-rules.md#an-anchor-buys-immunity-and-pays-in-blindness). ⚠️ **AND THE TWO LINE NUMBERS THIS SENTENCE CARRIED UNTIL NOW (`:3934`, `:4142`) HAD *BOTH* ROTTED BY FIVE — measured `:3939` and `:4147` at `3e49610f`. The bullet documenting that line numbers are 67% defective was itself carrying two. STRIPPED rather than updated, per its own rule.** **The bullet only ever named one of them.** I wrote "exactly one hit" here and the command returned two on the next line — which is the second site being found by the very anchor added to stop the first from getting lost. That phrase is prose inside the panic message and no identifier rename can reach it, whereas `from_host_buffer_on` and `NodeHandle` both can. *(Control: `git grep -c 'assert_eq!' -- fuel-graph/src/lib.rs` → **437**, so the query finds asserts where they exist.)*) The three ex-"panicking" `decompose`s are **resolved** (2026-07-03; a prior G2 pass had already converted the panics to self-returns — the real work was recipes, not crashes): **`nf4_matmul`** now carries a total primitive recipe (nibble-unpack + indicator-sum NF4 codebook + per-block scale → matmul); **`flash_attn`** decomposes its concrete-`k_len` decode (static `Slice` + bottom-right-aligned SDPA) — only symbolic (`Sym`) `k_len` stays a documented registry-layer gap (no `DynScalar`-length `Slice`; the symbolic oracle is the `decode_flash` optimizer arm, which holds the `SymEnv`); **`selective_scan`** is the constitution's canonical basis gap (G3 — needs a higher-order `Scan` `Op`; the CumSum closed-form overflows for `a<0`), kept as a never-crash surfaced gap. Parity + gap-posture tests in `fuel-core/src/lazy.rs`.

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **Never panic on production paths.** `Result` from day one. ⚠️⚠️ **CLOSING ONE CONSTRUCTOR FAMILY DID NOT SATISFY THE PRINCIPLE, AND THE CLOSURE READS LIKE IT DID.** `NodeHandle::from_*` — the standing violation this rule carried for months — was converted to `Result` at `62ac2991` (GAP-003, PR #173). **STILL OPEN: GAP-308 measures 475 production panics in `fuel-transformers/src/models/`, and GAP-309 records two `Result`-returning public builders that each inherit 8 unguarded dtype/shape panics. DO NOT READ THE GAP-003 CLOSURE AS THE RULE BEING MET.** ⚠️⚠️ **AND IF YOU ARE COUNTING PANICS, THE ANCHOR YOU REACH FOR FINDS ONE OF THREE CONSTRUCTS AT EACH SITE AND FAILS FLATTERINGLY: a prose-message anchor sees the `assert_eq!` and is BLIND to an `.expect(…)` five lines below and to a `.write().unwrap()` — real hits, right file, real violations, neighbours omitted, and NOTHING IN THE OUTPUT SAYS IT IS A SAMPLE. A rotted SYMBOL anchor fails loudly and gets re-derived; a durable MESSAGE anchor fails silently.** ⚠️ **AND `.write().unwrap()` IS A DIFFERENT OBLIGATION THAT MUST NOT BE SWEPT INTO THIS ONE — a poisoned lock means another thread has ALREADY panicked, so returning `Result` there propagates an unrecoverable condition and buys the caller nothing** (198 in one file alone, and folding them in would have inflated the population with work that should not be done). → [`an-anchor-buys-immunity-and-pays-in-blindness`](#an-anchor-buys-immunity-and-pays-in-blindness) · → [`never-panic-and-what-closing-one-family-did-not-close`](#never-panic-and-what-closing-one-family-did-not-close) **The three ex-"panicking" `decompose`s are RESOLVED and none was ever a crash — the real work was recipes:** `nf4_matmul` carries a total primitive recipe (nibble-unpack + indicator-sum NF4 codebook + per-block scale → matmul); `flash_attn` decomposes its concrete-`k_len` decode (static `Slice` + bottom-right-aligned SDPA), leaving **symbolic `k_len` as a documented registry-layer gap** (no `DynScalar`-length `Slice`; the symbolic oracle is the `decode_flash` optimizer arm, which holds the `SymEnv`); `selective_scan` is the constitution's canonical basis gap **G3** — it needs a higher-order `Scan` `Op`, the CumSum closed-form overflows for `a<0`, and it is kept as a never-crash SURFACED gap.

## one-cuda-build-at-a-time-and-the-misleading-red

**ONLY ONE `--features cuda` BUILD ON THIS BOX AT A TIME, and the failure mode is a MISLEADING RED IN A THIRD-PARTY BUILD SCRIPT (2026-08-08).** Two concurrent CUDA builds do not fit. `baracuda-kernels-sys` compiles **426 kernels through a single build script**, and under contention `ptxas` dies with `ptxas fatal : Memory allocation failure` — **while ~22 GB of system RAM is free**, so free-RAM headroom is NOT the resource that binds and measuring it does not tell you a second build will fit. **`gpu-run` does NOT cover this: it serialises GPU *runs*, not *builds*.** Before starting a CUDA build, check for peer CUDA processes and match ownership **by the build-script's target path or by parentage — never by process count or start time** (both are non-discriminating; a count cannot attribute). ⚠️⚠️ **AND UPGRADING THE INSTRUMENT DOES NOT FIX AIMING IT AT THE UNASKED QUESTION — 2026-08-15, self-reported by the lane that had quoted this very rule at the architect hours earlier.** Mid-forge they reported *"254 CPU-seconds in 20 seconds wall — the forge is genuinely working."* **The number was correct and about the WRONG POPULATION: no `baracuda-kernels-sys` build dir in ANY worktree had a write dated that day, so those 32 CPU-burning processes were not producing their kernels, and their build produced nothing at all.** **THEY UPGRADED FROM *COUNT* TO *CPU-SECONDS* — a strictly better instrument for "is anything working" — AND STILL NEVER ATTRIBUTED THE PROCESSES TO THEIR OWN WORKTREE. A SHARPER INSTRUMENT AIMED AT THE SAME UNASKED QUESTION.** The failure is not low resolution, it is **the missing ATTRIBUTION STEP** — and improving resolution feels like progress while leaving the defect exactly where it was. **⚠️ AND THE TWO ERRORS COMPOUNDED INTO A CONFIDENT WRONG PICTURE, WHICH IS THE PART WORTH FEARING: their pre-flight "zero peer CUDA processes" check was ALSO malformed (`tasklist | grep -icE` output read as a clean zero), so they may have launched INTO a peer's forge and then read THAT PEER'S CPU as evidence of their own progress. NEITHER ERROR ALONE WAS FATAL; TOGETHER THEY PRODUCED A COHERENT, WHOLLY FALSE ACCOUNT OF A BUILD THAT NEVER STARTED.** **⚠️⚠️ RETIRED 2026-08-26 — THIS PARAGRAPH CARRIED A FALSE ACCUSATION AGAINST A SIBLING PROJECT FOR 17 DAYS, IN THE FILE THAT LOADS INTO EVERY SESSION, AND IT IS THE SECOND TIME AGAINST THE SAME PROJECT.** It read: *"Cargo does export `NUM_JOBS` to build scripts, but **`baracuda-forge` never reads it**… `cargo -j N` limits how many CRATES compile at once and does not touch nvcc concurrency at all… **The control that works is DRAINING**"*, plus *"raised with Baracuda as a propose-first ask to honour `NUM_JOBS`"*. **MEASURED AT THE VERSION FUEL ACTUALLY BUILDS — `Cargo.lock` resolves `baracuda-forge 0.0.1-alpha.79` from crates.io, and its vendored source reads `NUM_JOBS` in NINE places** (`lib.rs` 2, `parallel.rs` 7), with `parallel.rs`'s `resolve_thread_count` reading it via `std::env::var("NUM_JOBS")`, documented as an **UPPER CAP, never a floor**, and **unit-tested** (`resolve_thread_count(16, Some(100)) == 8` — a `-j` above the 50% default does not raise it). **The precedence chain in alpha.79 is `BARACUDA_FORGE_THREADS` (checked first, absolute) → `RAYON_NUM_THREADS` → `NUM_JOBS` as a cap → the 50% default.** **The ask was DELIVERED THE SAME DAY IT WAS RAISED** (`62182fce`, an ancestor of the `alpha.79` release commit), so the "propose-first ask" was outstanding only in this file. ⚠️ **THE COST WAS IN THE PRESCRIPTION, NOT THE PROSE: this file FORBADE the cheap control and mandated the expensive one.** Draining — waiting for every peer CUDA process to exit — was presented as the only thing that works, when **`cargo -j N` lowers the forge pool**. ⚠️ **STILL UNMEASURED HERE AND STATED AS SUCH: that `-j 4` caps nvcc END-TO-END on this box has NOT been re-run** — what is measured is that the code reads `NUM_JOBS`, that Fuel resolves the version containing it, and that the ORIGINAL CLAIM'S MECHANISM ("never reads it") IS FALSE. **`BARACUDA_FORGE_THREADS=<n>` remains the preferred knob** — absolute rather than a cap, and it perturbs only forge's pool instead of every rayon consumer in the build. ⚠️ **THE CLASS, AND IT IS THIS FILE'S OWN, TWICE OVER: `staleness-by-workaround` — a PROHIBITION whose forbidden path stopped failing.** *"Is this claim still true?"* passes on the rule text; **only a re-attempt of the forbidden thing detects it**, and nobody re-attempts a thing the working agreement calls useless. **The `baracuda-kernelgen` bullet was retired on 2026-08-20 with the note *"the accusation outlived its own retraction by days, in the file that loads into every session — if Baracuda's lane had read our working agreement they would have found us still blaming them."* SAME DEFECT, SAME PROJECT, SAME FILE, SIX DAYS LATER — and again it was a peer who found it, not us.** **WHEN YOU RECORD THAT AN UPSTREAM LACKS SOMETHING, RECORD THE VERSION YOU MEASURED AND RE-CHECK IT AT EVERY BUMP: a dependency claim is pinned to a version whether or not you write the version down, and ours moved to `alpha.79` in the lockfile while the prose stayed at the old behaviour.** (Traced in the **reference-only** `C:\Projects\baracuda` checkout; **Fuel resolves `alpha.79`, NOT the `alpha.78` the manifest requests — see the version note above — so check `Cargo.lock` before relying on anything traced there.** ⚠️ **AND THE SENTENCE THAT USED TO END THIS PARENTHETICAL — *“Raised with Baracuda as a propose-first ask to honour `NUM_JOBS`”* — WAS DELETED 2026-08-26 AS FALSE: the ask was DELIVERED THE SAME DAY IT WAS RAISED and was outstanding only in this file.** It survived the retraction above **because it is a parenthetical about PROVENANCE attached to a different sentence, so it did not read as part of the mechanism being retracted** — and the retraction's own `git grep` missed it **on capitalisation alone**.) **AND GREP THE LOG FOR `ptxas fatal` / `nvcc fatal` BEFORE ATTRIBUTING ANY `--features cuda` BUILD-SCRIPT FAILURE.** The whole failure presents as cargo **exit 101 with exactly one `error:` line** — `failed to run custom build command for baracuda-kernels-sys` — with the real cause buried under a wall of harmless CUTLASS `#177-D "declared but never referenced"` warnings. The available wrong conclusions are all right there and all false: *"the hypothesis under test is confirmed"*, *"baracuda alpha.NN is broken"*, *"my code doesn't compile"*. **An allocator failure inside a code generator is not a fact about the code being generated.** Note this is a *sub-case* of the never-trust-an-exit-code rule above, and a nastier one: there the error text was **absent**, here it is **present and misleading**. **Runtime PATH (2026-07-04):** the `fuel-core` cuda test exe directly imports `cudnn64_9.dll` (+ `cublas64_13.dll`); cuDNN's CUDA-13.3-matched build must be on PATH to *launch* it or the process dies `0xc0000135 STATUS_DLL_NOT_FOUND`. Prepend `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64` (installed there, not on the default PATH). `fuel-dispatch` cuda tests don't hit this (no cuDNN link).

### Former CLAUDE.md index text (moved verbatim 2026-09-16, base c4936cfc)

- **ONLY ONE `--features cuda` BUILD ON THIS BOX AT A TIME, AND THE FAILURE MODE IS A MISLEADING RED IN A THIRD-PARTY BUILD SCRIPT.** `baracuda-kernels-sys` compiles **426 kernels through a single build script**, and under contention `ptxas` dies with `ptxas fatal : Memory allocation failure` — ⚠️ **while ~22 GB of system RAM is FREE, so free-RAM headroom is NOT the resource that binds and measuring it does not tell you a second build will fit.** ⚠️ **`gpu-run` DOES NOT COVER THIS: it serialises GPU *runs*, not *builds*.** ⚠️⚠️ **ATTRIBUTE PEER CUDA PROCESSES BY THE BUILD-SCRIPT'S TARGET PATH OR BY PARENTAGE — NEVER BY PROCESS COUNT OR START TIME.** Both are non-discriminating: a count cannot attribute, and a sharper instrument aimed at the same unasked question is not an improvement — a lane upgraded from *count* to *CPU-seconds*, reported *254 CPU-seconds in 20 seconds wall, the forge is genuinely working*, and **no `baracuda-kernels-sys` build dir in ANY worktree had been written that day**. The number was right and about the wrong population. **THE MISSING STEP IS ATTRIBUTION, AND IMPROVING RESOLUTION FEELS LIKE PROGRESS WHILE LEAVING THE DEFECT EXACTLY WHERE IT WAS.** **THE KNOBS: `BARACUDA_FORGE_THREADS=<n>` is PREFERRED — absolute rather than a cap, and it perturbs only forge's pool instead of every rayon consumer. `cargo -j N` ALSO lowers it, because forge reads `NUM_JOBS` as an UPPER CAP (never a floor).** ⚠️ **DRAINING — waiting for every peer CUDA process to exit — is NOT required, and this file once mandated it while forbidding the cheap control. THE CLASS IS `staleness-by-workaround`: A PROHIBITION WHOSE FORBIDDEN PATH STOPPED FAILING. *Is this claim still true?* passes on the rule text, so only a RE-ATTEMPT detects it — and nobody re-attempts a thing the working agreement calls useless. WHEN YOU RECORD THAT AN UPSTREAM LACKS SOMETHING, RECORD THE VERSION YOU MEASURED AND RE-CHECK AT EVERY BUMP: a dependency claim is pinned to a version whether or not you write the version down.** ⚠️⚠️ **AND GREP THE LOG FOR `ptxas fatal` / `nvcc fatal` BEFORE ATTRIBUTING ANY `--features cuda` BUILD-SCRIPT FAILURE.** The whole failure presents as cargo **exit 101 with exactly one `error:` line** — *failed to run custom build command for baracuda-kernels-sys* — with the real cause buried under a wall of harmless CUTLASS `#177-D "declared but never referenced"` warnings. **The available wrong conclusions are all right there and all false: *the hypothesis under test is confirmed*, *baracuda alpha.NN is broken*, *my code doesn't compile*. AN ALLOCATOR FAILURE INSIDE A CODE GENERATOR IS NOT A FACT ABOUT THE CODE BEING GENERATED.** **RUNTIME PATH: the `fuel-core` cuda test exe directly imports `cudnn64_9.dll` (+ `cublas64_13.dll`), so cuDNN's CUDA-13.3-matched build must be on PATH to *launch* it or the process dies `0xc0000135 STATUS_DLL_NOT_FOUND`** — prepend `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64`, which is not on the default PATH. **`fuel-dispatch` cuda tests do not hit this (no cuDNN link).** → [`one-cuda-build-at-a-time-and-the-misleading-red`](#one-cuda-build-at-a-time-and-the-misleading-red)

---

<!-- The sections below were split out of CLAUDE.md on 2026-09-16 (aggressive trim, base c4936cfc). -->

## prefer-p-crate-and-the-root-check

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **Prefer `-p <crate>`; a root-wide `cargo check` now WORKS but costs ~31 minutes.** This rule used to read "NEVER run `cargo check`/`cargo test` workspace-wide" because the root build *failed*. **As of 2026-07-31 it does not: `cargo check` at the workspace root returns exit 0** (measured, 1851s). Two things changed — `fuel-wasm-tests`, the broken default-member that actually caused the failure, was archived; and a cross-crate `Result` fallout in `fuel-lazy-examples` was fixed. So the prohibition is retired and replaced by a cost argument: 31 minutes of full-machine build, on a box shared with several concurrent sessions, is antisocial when `-p <crate>` answers the same question in seconds. Run it deliberately (e.g. to re-verify this claim), not reflexively. **The `cargo test` half of the old rule still stands for a different reason: the suite includes live-GPU and network-dependent tests.**

  Two earlier stated reasons for this rule were WRONG and are recorded so they are not re-derived: (1) "`tensor-tools` has a standing `Device::Cpu` break and is a default-member" — the crate was `fuel-tensor-tools`, it was in `members`, NOT `default-members`, so it could never have been the cause; it contained no `Device::Cpu` at all, and it was **archived 2026-08-01** in B6 (see ROADMAP §2) ⚠️⚠️ **AND THAT ARCHIVAL CLAIM IS FALSE — MEASURED 2026-09-02 AT `ffc5f25a`, THE THIRD CORRECTION TO THIS ONE BULLET AND THE FIRST IN THIS DIRECTION.** `fuel-tensor-tools` **is listed in `[workspace.members]` in the root `Cargo.toml`, its directory exists, and it COMPILES** — a lane built it as one of five safetensors consumers (`Checking fuel-tensor-tools v0.10.3`). *(Control: `fuel-wasm-tests`, a genuinely archived crate, returns **0** files to the same query, so the instrument discriminates.)* **The other two corrections here narrowed a false accusation; this one reverses a false RETIREMENT — a bullet can be wrong in both directions about the same crate, and being corrected twice is no evidence the third claim is sound.**; (2) "no `fuel-wasm-examples` directory exists" — it does, with 11 crates and 100 files. Both were asserted from searches that returned nothing, without a positive control showing the search could have found the thing. The default member that *did* break is **`fuel-wasm-tests`** (unresolved `fuel::quantized::k_quants` — no `k_quants` in `quantized`, plus a `.round()` trait-scope error). **`fuel-wasm-tests` was ARCHIVED 2026-07-31** (removed from the workspace and deleted; retrievable from git — see ROADMAP §2), so re-test whether a bare root `cargo check` now succeeds before assuming this rule still bites; if it does, the rule can finally be retired.

**Correction (2026-07-31) — a claim added here on 2026-07-30 was FALSE.** It read: "`default-members` lists `fuel-wasm-examples/*`, and **no `fuel-wasm-examples` directory exists**." The directory **does** exist and holds **11 real crates / 100 files**, each with its own `Cargo.toml` (bert, blip, chat-template, llama2-c, moondream, phi, quant-qwen3, segment-anything, t5, whisper, yolo). The claim came from an empty search result treated as evidence of absence, with no positive control to show the search would have found the directory had it been there — the same defect this file's own process rules exist to catch. Note also that ROADMAP §2 says the WASM tree is "quarantined out of the workspace (removed from `[workspace.members]`)", which is **also stale**: the glob is present in both `members` and `default-members`, and the sampled crates import no retired crate (`phi` uses `fuel-transformers::generation::LogitsProcessor`, which still exists). Whether they actually build is UNVERIFIED — check before trusting either statement.

**Always `-p <crate>`.**

## an-enum-variant-needs-a-workspace-gate

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **EXCEPTION — adding an ENUM VARIANT needs a WORKSPACE-scope check, not `-p`.** A variant added to an enum marked `// EXHAUSTIVE-BY-DESIGN` (grep the marker; `Scalar`, `DType`, … qualify) breaks every *consumer* crate's exhaustive `match` — and `-p <the-enum's-own-crate>` is the one gate guaranteed **NOT** to see it, because the defining crate stays green while consumers go red. Gate it across consumers: a root `cargo check` (covers default-member consumers — that alone would have caught GAP-049) **plus** each feature-gated consumer under its feature (`--features cuda`, `--features vulkan`). GAP-049: `feff38ed` added `Scalar::F8E8M0` and left `main` RED on a default build for an unknown window because it was verified with `-p fuel-ir` (the leaf where the variant *lives*, the one crate that couldn't hear the complaint). The compile break is the *intended* enforcement — a wildcard arm would silently mis-encode the new dtype — but it only works if the gate is pointed where the compiler is speaking. A marker-integrity test (`fuel-ir/tests/exhaustive_by_design_marker.rs`) keeps `EXHAUSTIVE-BY-DESIGN` and `#[non_exhaustive]` mutually exclusive, but it does **not** verify you ran the right gate — that part is this rule. This is "gate must match the change kind" for the enum-variant change-kind.

## all-gpu-runs-go-through-gpu-run

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **ALL GPU-touching runs MUST go through the lock: `pwsh scripts/gpu-run.ps1 -Project <name> -- <cmd>`.** Never invoke live-GPU tests, `--features cuda`/`--features vulkan` runs, capture/replay, sanitizers, or GPU benches directly. `gpu-run` is a machine-wide named mutex (`Global\gpu-run`) enforcing one GPU-touching run at a time across ALL sessions/projects, with abandoned-holder recovery that survives hard-kill/bugcheck (validated 2026-08-01, incl. a real contended multi-process kill). It **replaces** the old "one live-GPU suite at a time" *convention*, which did not hold with many sessions and contributed to the 2026-07-31 host-aperture kernel bugcheck ([docs/postmortems/2026-07-31-gpu-host-aperture-crash.md](postmortems/2026-07-31-gpu-host-aperture-crash.md)) — routing through it is not optional; it is the only thing preventing recurrence. Still `#[ignore]` live-GPU tests so a plain `cargo test` skips them; run them via `gpu-run`. (Caveat: `gpu-run` serializes *between processes*; it does NOT fix *within-process* fan-out — e.g. K threads racing `vkCreateInstance` on a cold `SystemTopology` cache — a separate memoization fix.) **Corrected 2026-07-30: the card is an RTX 4070 *Laptop* GPU with 8 GB (8188 MiB), not 12 GB.** Measured directly — `nvidia-smi --query-gpu=name,memory.total` returns `NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB`. The rule is therefore *more* binding than it read, not less: a budget written against 12 GB overcommits the card by 50%, and the failure mode is an OOM abort — bare nonzero exit, no panic, nothing in the log — which mimics a harness defect and costs far more to diagnose than it should.

## sibling-checkouts-are-reference-only

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ SIBLING CHECKOUTS ARE *NOT* PATH DEPS — THIS BULLET'S HEADLINE WAS FALSE FOR BOTH CRATES IT LISTS, MEASURED 2026-08-15.** It used to read *"Sibling path deps must exist for the workspace to parse"*. **Measured at head: `vulkane = "0.9.0"` (in `fuel-vulkan-backend/Cargo.toml`) and `aocl-blas`/`aocl-types = "0.1"` (root `:103-104`) all resolve from CRATES.IO with lockfile checksums; `[patch.crates-io]` contains ONLY `fuel-kernel-seam` + `fuel-kernel-seam-types`; and `.cargo/config.toml` has NO `paths` override.** So a fresh clone parses with no sibling checkouts at all, and the sibling directories are **reference-only**, exactly like `../baracuda`. **HOW IT WAS CAUGHT, AND THE LESSON IS THE SECOND HALF: the Vulkane maintainer corrected the vulkane half from source after the portfolio PM relayed this bullet to them — and they were about to adopt the WRONG WARNING TRIGGER because of it (publish-time, not merge-time, is when a crates.io consumer is affected). CHECKING THE NEIGHBOUR OF A REPORTED ERROR FOUND THE SAME DEFECT IN THE aocl HALF, WHICH NOBODY HAD REPORTED.** A wrong fact in a working agreement does not stay inside the project: it got relayed to another project and nearly changed their process. **Practice: when a peer corrects one entry in a list you wrote, re-measure the WHOLE list — the error was in how the list was compiled, not in one row.** The directories below are still useful to have locally for reading source:

  - `../aocl` (github.com/ciresnave/aocl) — `aocl-blas`, `aocl-types` path deps. Never enable the `aocl` cargo feature in tests (AMD's DLLs aren't on PATH).

  - `../vulkane/vulkane` (github.com/ciresnave/vulkane) — Vulkan FFI, used by `fuel-vulkan-backend` (a default-member).

## backend-crates-have-no-backend-feature

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ `--features cuda` / `--features vulkan` DO NOT EXIST ON THE BACKEND CRATES THEMSELVES — THEY *ARE* THE BACKEND (2026-08-13). One lane made this mistake on `fuel-vulkan-backend`, did not generalise it, and made the identical mistake on `fuel-cuda-backend` two hours later.** `fuel-cuda-backend`'s features are `cudnn` / `nccl` / `ug` only; the `cuda` feature lives on **consumers** (`fuel-core`, `fuel-dispatch`). **The correct gate for a backend crate is a PLAIN `cargo check -p fuel-<x>-backend --all-targets`.** **The failure presents as `exit 101` with `error: the package '…' does not contain this feature` and ZERO compile artifacts — i.e. the HARNESS-NEVER-RAN row of the truth table, which reads exactly like a broken gate on your own code.** *"I've already learned this one"* was not a defence here either.

## gate-several-crates-in-one-invocation

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ GATE MULTIPLE CRATES IN **ONE** CARGO INVOCATION USING `pkg/feature` SYNTAX — SPLITTING BY PACKAGE SILENTLY DOUBLES A 77-MINUTE BUILD (2026-08-13, measured; and this rule REPLACES a wrong version of itself that stood here for an hour).** `cargo check -p fuel-core --features cuda` and `cargo check -p fuel-cuda-backend` resolve features **differently**, so they share **no cache**: two legs, **77 minutes each**, ~2.5 hours of forge. **The same coverage in ONE invocation — `cargo check -p fuel-core -p fuel-cuda-backend --features fuel-core/cuda --all-targets` — resolves once, HIT THE FIRST LEG'S CACHE ENTIRELY (`forge = 0`), and finished in 34 SECONDS.** **⚠️ THE VERSION OF THIS RULE THAT STOOD HERE FIRST SAID *"a two-leg CUDA gate is ~2 hours, budget for both forges"* — TRUE ABOUT THE SPLIT AND ACTIVELY MISLEADING, because it told the reader to PAY the cost rather than that the cost was self-inflicted. A rule that correctly describes a bad path, without naming the good one, is worse than no rule: it makes the waste look like a property of the toolchain.** The lane that reported the original measurement found the combined form and corrected it. **STILL TRUE and the reason two crates are needed at all: `-p fuel-core --features cuda` does NOT compile `fuel-cuda-backend`'s own test targets, so it cannot verify `#[cfg(test)]` code there — the artifact that proves it did is `fuel-cuda-backend (lib test)` in the log, not merely `Checking fuel-cuda-backend`.**

## a-warning-string-is-not-a-finding

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **DO NOT PROMOTE A WARNING STRING TO A FINDING WITHOUT CHECKING ITS SCOPE (2026-08-08).** A `f32: From<f64>` fallback warning seen in an aarch64 cross-check was relayed as *"the one substantive item — the numeric path silently degrades on aarch64"*. **It appears identically on x86**; it is a future-compat lint whose fallback produces the **correct value**. **The claim was manufactured from the warning's TEXT plus the CONTEXT it was noticed in, which is not evidence of either scope or severity** — and a plausible defect sentence, once written into a registry, gets repeated as fact. **A hypothesis handed to someone else must be labelled as one** (see [[fact-vs-consequence-labeling]]).

## a-long-registry-row-reads-its-stalest-claim-first

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️⚠️ A LONG REGISTRY ROW ACCUMULATES BY APPENDING, SO THE MOST SUPERSEDED CLAIM IS THE ONE YOU READ FIRST — AND IT IS USUALLY THE MOST FORCEFULLY WORDED, BECAUSE IT WAS WRITTEN WHEN IT SEEMED NEWLY ESTABLISHED (2026-08-14, coordinator, and the SECOND time this row mis-informed the same external consumer in the same direction).** GAP-007 is **6,092 characters**. Its opening carries *"ANSWERED — it is BROKEN, not unoptimised"* in caps; its **re-tiering by measurement — "NOT ON THE PREFILL PATH, and no longer Lightbulb's blocker" — lives ~4,000 characters later**, and the STATUS CELL says *"REAL, but off the prefill path"*. **I printed that status cell while composing the answer and did not reconcile it against the body I was quoting.** The consumer caught it by refusing to average two conflicting statements — the same refusal that caught the FIRST mis-relay of this row. **PRACTICE: (1) READ THE STATUS CELL FIRST AND TREAT IT AS THE VERDICT — the body is a history, not a conclusion; (2) if the body contradicts the status, the STATUS wins and the body needs a superseded marker AT THE STALE PASSAGE, not only a correction at the end; (3) never quote a long row's opening as current without searching it for a later reversal.** **A row that has to be read to the end to be read correctly is a format defect, not a reader defect** — the fix is inline supersession, and GAP-007 now carries one.

## a-positive-control-is-a-claim-too

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ A POSITIVE CONTROL IS A CLAIM TOO — AND THE WORST PLACE TO PUT A FALSE ONE IS INSIDE A REQUIREMENT, BECAUSE A REQUIREMENT GETS BUILT (2026-08-13, coordinator, self-inflicted, caught by the lane it was handed to).** Specifying a tripwire (*"assert zero production constructors of `Encoding::AffineBlock`"*), the coordinator **required** a control: *"sibling `GgmlBlock` IS found in production, so the zero is falsifiable."* **It came from `grep -c 'GgmlBlock'` returning nonzero — which counts OCCURRENCES, not CONSTRUCTIONS, and does not separate test from production.** Measured properly: **every construction site of BOTH variants is test code**, and both "production" hits sit within five lines of a `#[test]`. **So the prescribed test would have shipped with a control asserting something false, and its zero would have been unfalsifiable — the exact failure the requirement existed to prevent.** **THE ESCALATION IS THE LESSON: a loose measurement became a report ("has real consumers"), the report hardened into a claim ("is found in production"), and the claim became a REQUIREMENT — at which point nobody re-derives it, because it arrives as a spec.** **PRACTICE: measure a positive control before REQUIRING it, to the same standard as the thing it controls; and for a Rust enum variant, note that occurrence ≠ construction ≠ pattern — the same token serves all three, so `grep -c` cannot answer "is this constructed in production".** Corollary for anyone handed a control in a spec: **verify it — a control you were told to use is the one nobody checked.**

## a-formatter-and-a-classifier-conform-identically

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ A FORMATTER AND A CLASSIFIER PRODUCE IDENTICAL CONFORMANCE RESULTS WHEN THE TEST VECTORS SUPPLY THE THING BEING CLASSIFIED (2026-08-13).** Fuel's sk4 byte-match passes identically whether `derive_structure_key_token` **assigns** an op family or is **given** one, because every vector supplies it — so a green leg cannot distinguish the two, and the leg's independence is real but **scoped to FORMATTING**. Generalises well past sk4: **whenever a conformance test feeds in a value the production path would have to derive, the test says nothing about that derivation** — and the gap is invisible in the result, the artifact, and the pass rate. **Ask of any conformance claim: which inputs does the vector SUPPLY that production would have to COMPUTE?** Adopted upstream as a required leg-report field (*state whether the deriver assigns the family or is given it*).

## a-true-justification-for-a-wider-claim

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A TRUE JUSTIFICATION ATTACHED TO A WIDER CLAIM THAN IT SUPPORTS (2026-08-12).** Ask what a stated reason actually *licenses*, not whether it is *true*. `DecodeModel`'s geometry methods assert a uniform KV shape, justified in-doc by *"all sessions share one model, so this is uniform"*. **That reason is CORRECT — for uniformity across SESSIONS. The methods use it to assert uniformity across LAYERS and across STATE KINDS, which it does not establish** — and MLA (`DeepSeek2Model`, shipped, decode-capable) has neither a `KvCache` nor a per-layer-uniform state, yet the methods are **syntactically returnable** for it, so a scheduler trusting them allocates the wrong state and everything type-checks (docs/gaps.md GAP-166). **Why this defect is durable where a plain wrong comment is not: the citation is REAL, so the claim reads as sourced, and nothing in review flags it. A false reason gets caught; a true reason with the wrong scope does not.** Distinct from fact-vs-consequence labelling — there the two parts are separable and one is unsourced; **here there is ONE sentence whose evidence covers a strict subset of what it is being used to claim.** **Practice: when a doc comment justifies an invariant, state the invariant's scope and the reason's scope SEPARATELY and check they match.** Same shape as the expiring-decline coupling (GAP-161): the justification lives in a different scope from the thing it licenses.

## a-test-at-the-wrong-site

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A TEST AT THE WRONG SITE IS NOT WEAKER EVIDENCE THAN A TEST AT THE RIGHT ONE — IT IS EVIDENCE ABOUT A DIFFERENT QUESTION WEARING THE RIGHT LABEL (2026-08-12, KISS architect's framing).** A lane was hours from landing a KISS reserved-dtype recognition fix — single source of truth, matchable typed error, three tests, all green — on `fuel-ir/src/dtype.rs::from_str`, **which parses Fuel's INTERNAL token vocabulary (`"f8e4m3"`), reachable only by Fuel's own graph-attr deserialization. The tokens it was written to recognise (`"f8e4m3fnuz"`) cannot arrive there.** It would have gone green and its greenness would have been cited as conformance evidence (docs/gaps.md GAP-167, GAP-155). **This is NOT the usual instrument failure: the instrument was perfect. It was AIMED WHERE THE PROPERTY DOES NOT LIVE**, so no control on sensitivity or discrimination catches it — those validate the instrument, and the instrument is fine. **The one question that does catch it: WHICH SURFACE DOES THE OBLIGATION BIND, and can the trigger physically REACH the code under test?** For a conformance duty, ask whose act it is (reader / emitter / provider / consumer) before writing the assertion — Fuel turned out to have **no** token-reader surface at all (zero dtype-token parse arms against a 116-arm positive control in `fkc/lower.rs`), so the clause had no site here to land on. Same family as the inert audited flip (GAP-077, 0 live triggers) and counting the wrong construct: **the artifact was verified, and what it was verified ABOUT was assumed.**

## a-positive-control-can-be-inert

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **YOUR POSITIVE CONTROL CAN ITSELF BE INERT — CHOOSE ITS REFERENCE POINT FROM THE PATH UNDER TEST, NOT FROM THE TYPE (2026-08-12).** To prove a test could detect whether a symbol is *referenced*, a lane perturbed a sibling it believed was known-referenced (`cached_len_sym`) — and the output did not move, because on the device-offset path the KV write offset rides a device-resident **buffer** (`Op::WriteSliceDoff` reads `DecodeTokenData::offset` at launch), not the symbol. **The proof-of-sight was itself invisible on that path, so the verdict would have been VACUOUS — a meaningless green produced by a test that HAD a positive control.** A sibling that is load-bearing *somewhere* is not thereby load-bearing *here*. **Fix: one control that must ALWAYS move the output (perturb the input itself, proving the pipeline is live at all), plus a control whose asserted direction MATCHES the path the run actually takes rather than an assumed one.** Corollary from the same run: **two green arms are not two confirmations** — the same test over two models measured **different code paths** (`PhiModel` SymEnv-live vs `LlamaModel` device-offset SymEnv-inert), so one arm carried the evidence and the other established strictly less. **Write which arm proves what into the file, or a later reader counts greens.** See docs/gaps.md GAP-029.

## a-structural-guarantee-needs-compile-coverage

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A STRUCTURAL GUARANTEE IS ONLY AS STRONG AS THE COMPILE COVERAGE OF THE CODE CARRYING IT — AN EXHAUSTIVE MATCH IN CODE NOTHING BUILDS IS A COMMENT THAT HAPPENS TO BE WRITTEN IN RUST (2026-08-12, KISS architect's formulation, and Fuel is the worked example).** **AND ASK THE PLAIN QUESTION BEFORE THE EXOTIC ONE: WHICH *SINGLE* NON-DEFAULT FEATURES DOES NO GATE BUILD?** Fuel went hunting for the obscure `cuda`+`telemetry` combination — and underneath it sat the more ordinary hiding place: `fuel-dispatch` has **`default = []`**, `pub mod telemetry` is `#[cfg(feature = "telemetry")]`, and **`telemetry` appears in NO CI job** ⚠️ **— RETIRED 2026-08-20: it does now.** `.github/workflows/rust-ci.yml` carries a NAMED step, `Check telemetry-gated dispatch (sk4 wire codec)`, running `cargo check -p fuel-dispatch --features telemetry --all-targets`. The "unenforced everywhere" conclusion no longer holds., so `telemetry/structure_key_derive.rs::dtype_token` — **the wildcard-free match that makes "a new Fuel dtype is a compile error BEFORE it can reach the wire" true** — was **unenforced everywhere**, and was reported cross-project as a will-not-drift structural guarantee before that was checked. `cargo check -p fuel-dispatch --features telemetry` costs **seconds, no CUDA, no SDK, no slot** and covers eight-plus files. **Practice: when you claim a compile-time guarantee, name the configuration that compiles it and confirm some gate builds that configuration** — and remember that *"enforced by a script someone must remember to run"* is weaker than *"enforced by CI"* and must be stated as the weaker thing. See docs/gaps.md GAP-173.

## an-approved-change-can-invalidate-your-claim

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A CHANGE YOU APPROVED CAN INVALIDATE A CLAIM YOU LATER MAKE — CHECK WHERE THE THING *IS*, NOT JUST WHETHER ITS OLD HOME IS GATED (2026-08-12).** The architect reported cross-project that Fuel's wire-boundary dtype exhaustiveness was unenforced, having read `telemetry/mod.rs`, confirmed the module is feature-gated, and **inferred the guarantee's coverage from the FILE's gate without verifying the MATCH was still in that file.** It wasn't: hours earlier a lane had consolidated two sk4 token tables, leaving `structure_key_derive.rs` a two-line delegation and moving the match to **ungated** `fuel-ir/src/token_kind.rs` — so it is compiled by a bare `cargo check -p fuel-ir` and by CI's `--workspace`. **The architect had RULED ON and RECORDED that consolidation personally; the information was in their own registry entry.** Distinct from *the construct is invisible in the number* — this is **a claim about a LOCATION, made after authorising the move.** **Practice: when asserting that code is or isn't covered, grep for the CODE at head, never reason from the gate on the file you remember it living in** — and treat your own recent rulings as the most likely source of staleness in your own claims. **Corollary worth keeping: the fix here was a SIDE EFFECT** — nobody was addressing compile coverage; a deduplication done because *"a diverging copy does not fail loudly, it emits a retired spelling under a current version prefix"* closed the hole before anyone knew it was open.

## rust-reachability-is-a-pub-mod-chain

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **RUST REACHABILITY IS A `pub mod` CHAIN, NOT A ROOT RE-EXPORT — CHECKING ONE VISIBILITY MECHANISM AND CONCLUDING "INTERNAL" IS AN ABSENCE CLAIM OFF THE WRONG INSTRUMENT (2026-08-12).** Asked whether Fuel's `structure_key` key types are a downstream source surface, the architect checked for a re-export in `fuel-dispatch/src/lib.rs` (**nothing**) plus references outside the crate (**zero**), with a positive control confirming the pattern finds the type where it lives — **which reads cleanly as "crate-internal."** It is wrong: `lib.rs`'s `pub mod telemetry;` → `telemetry/mod.rs`'s `pub mod structure_key;` → `pub struct FdxOperandDesc` with **7 pub fields**, so `fuel_dispatch::telemetry::structure_key::FdxOperandDesc` **is** published and downstream-constructible **by path**. **Practice: to decide whether a type is published, walk the `pub mod` chain from the crate root; a missing `pub use` proves nothing.** Same shape as checking a file's feature gate and assuming it still holds the match — **one mechanism checked, absence concluded.**

## a-checklist-of-axes-is-a-population-claim

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ A CHECKLIST OF AXES IS ITSELF A POPULATION CLAIM — POSITIVE CONTROLS VALIDATE EACH AXIS, NOTHING VALIDATES THE AXIS SET (2026-08-13).** The coordinator approved a **six-axis** check as the step that turns *"asserted LLaMA-shaped"* into *"measured"* before a six-family port. **All six axes are MODEL-SCALAR** (partial rotary · GQA · final norm · output bias · layer-weights type) **and none of them can see PER-LAYER variation — the property the design spec itself says decides the trait's shape.** Four of the six families vary per layer: three switch mask by `layer_idx < max_window_layers`, and SmolLm3 **skips RoPE** on listed layers. **Each axis was measured correctly; the SET did not span.** Distinct from the other failures in this file — those are a construct invisible *in* a measurement, this is a set of constructs never checked for **coverage**. **AND NAME AN AXIS BY THE BEHAVIOUR THAT VARIES, NOT BY THE FIRST MECHANISM YOU FIND:** anyone grepping `sliding_window` would have **cleared SmolLm3**, whose variation is RoPE-on/off — a different payload on the same axis. *"Per-layer attention behaviour"* catches both; *"sliding window"* catches three of four. **Practice: before trusting a checklist, ask what DIMENSION each entry ranges over and name one thing it provably cannot see.** See docs/gaps.md GAP-029.

## an-expiry-must-fire-on-a-checkpoint

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ AN EXPIRY MUST FIRE ON A CHECKPOINT THAT WILL OCCUR, NOT AN EVENT THAT MAY NOT — AND PROSE AT THE SITE IS NOT WHERE A DEADLINE LIVES (2026-08-13).** *"Delete this `#[allow(dead_code)]` the moment the first caller lands"* is **event-conditional with no detector**: if the caller never lands, nothing fires, and the suppression outlives its reason silently. **THE DEFECT CLASS DISGUISES ITSELF AS GOOD PRACTICE — it wears the costume of a conscientious TODO, which is exactly why it survives review.** The lane that wrote this one was **carrying the expiring-decline review in its own memory**, flagged the `allow` to the coordinator as something it would question in someone else's branch, **and still did not recognise that its own remedy WAS the defect class.** **CONVERSION: tie it to a checkpoint that is GUARANTEED to happen** — *"if X has not landed by the next handoff, the suppression comes out and the code goes with it."* Handoffs occur; the triggering event may not. **AND THE PLACEMENT IS HALF THE FIX: prose at the site tells a FUTURE CALLER what to do; only an OWNED REGISTRY ROW carries a deadline — because the doc comment is read solely by someone already standing there, and the deadline exists precisely for the case where NOBODY GOES.** Keep both: the site comment for the caller, the row for the clock. **Keep-vs-revert turns on BOUNDED-AND-NAMED versus OPEN-ENDED** — a short risk window with a scheduled check is a different object from an unbounded decline, and collapsing the two reverts things that are fine. See docs/gaps.md GAP-029, GAP-161, GAP-171.

## establish-facts-at-origin-main

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ THE SHARED WORKING TREES ARE STALE — ESTABLISH FACTS WITH `git show origin/main:<path>`, NEVER BY READING A WORKING TREE. THREE INCIDENTS IN ONE NIGHT, ACROSS TWO PROJECTS, ALL PRODUCING CLEAN ANSWERS (2026-08-12).** Measured that night: `C:\Projects\fuel` sat at `efa71c39` while `origin/main` was `f84f3329` — **164 commits ahead, measured with `rev-list --count`, not estimated (my first guess was ~40)**, and `C:\Projects\KISS` sat at `1c0c9a6` while its `origin/main` was `19c3ad7` (sk3 vs sk4 — **22 tokens, `s8`, `e4m3fn`, `c32`/`c64`, all wrong**). Every lane's Bash tool **defaults its cwd to the shared checkout**, so a grep run without thinking reads stale code. The three incidents: a Fuel-vs-KISS dtype divergence computed off the sk3 table (wrong in both directions); a KISS `cfg(windows)` survey that returned **zero** gates because `harness/` postdated the checkout; and a check that appeared to **REFUTE a true statement** because the shared Fuel tree still held a since-moved function. **Do not fetch or check out in a shared tree** — other sessions' Bash cwd points there and several hold long builds. Read `git show origin/main:<path>`; `git ls-remote` is a safe read-only tip check. **AND THE REASON THIS CLASS IS EXPENSIVE RATHER THAN MERELY ANNOYING: A STALE TREE PRODUCES A *CLEAN* ANSWER, AND CLEAN IS THE DIRECTION THAT DOES NOT GET QUESTIONED — A FALSE POSITIVE GETS INVESTIGATED, A FALSE NEGATIVE GETS FILED** (KISS architect's formulation). **So a null result needs a stated reason to be believed: the positive control, the commit it ran against, and any independent finding it agrees or disagrees with — and a null that CONTRADICTS a previously-reported finding must be reconciled before it is recorded, not after.** Two of the three were caught by exactly that: the result disagreed with something already known, not by anything in the output looking wrong.

## git-show-mangles-a-leading-dot-path

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️⚠️ AND THE PRESCRIBED INSTRUMENT SILENTLY RETURNS AN EMPTY FILE FOR ONE ARGUMENT SHAPE — WHICH IS EXACTLY THE SHAPE THIS RULE TELLS YOU TO USE (2026-08-26; found by the PM, narrowed twice, and BOTH narrowings changed the fix).** `git show origin/main:.github/workflows/rust-ci.yml` returns **0 bytes, exit 128**: MSYS reads the argument as a colon-separated PATH LIST and rewrites it to `origin\main;.github\workflows\rust-ci.yml`. **The fatal goes to STDERR, so the usual `2>/dev/null | grep -c` wrapper eats it AND the pipeline's `$?` is grep's, not git's — WHAT COMES OUT IS A CLEAN ZERO.** **THE MECHANISM, WHICH MAKES THE FIX MEMORABLE RATHER THAN ARBITRARY: MSYS REWRITES ONLY WHEN *BOTH* SIDES OF THE `:` LOOK PATH-LIKE.** `origin/main` looks path-like (it has a slash); `.github/workflows/…` looks path-like (leading dot plus slashes). **`./.github/…` reads as an EXPLICIT RELATIVE PATH and passes through untouched — so `./` removes the RIGHT half of the condition while KEEPING THE REF HONEST. `main:` removes the LEFT half and silently changes which commit you read.** Both “work”; only one is safe here. ✅ **THE FIX IS TWO CHARACTERS AND NEEDS NO ENV VAR — PREFIX THE PATH WITH `./`:** `git show origin/main:./.github/workflows/rust-ci.yml` → **19,398 bytes**; `./.gitignore` → 1,533; and it is **harmless on paths that already work** (`./Cargo.toml` → 20,602). `MSYS_NO_PATHCONV=1 git show …` also works; `git grep <pat> <ref> -- <path>` is the form for testing a STRING (path after `--`, cannot mangle) — **but a safe form that does not cover what you are doing offers no protection: the PM had only the string form and fell back to the broken one the moment they needed CONTENT.** ⚠️ **SCOPE, NARROWED TWICE AND BOTH TIMES IT MATTERED: it is a SLASHED REF plus a path whose *LEADING COMPONENT* starts with `.` — NOT "dot-prefixed paths".** Measured: `origin/main:.github/…` **BROKEN** · `origin/main:.gitignore` **BROKEN** · `origin/main:docs/kernel-contracts/.fkc-verified-ledger.json` **666,881 B, FINE** (dot-prefixed FILENAME, non-leading) · `origin/main:docs/gaps.md` fine · `origin/main:Cargo.toml` fine. **The over-broad version would have warned about the precision ledger — the file this project depends on most — and AN OVER-WARNING RULE GETS DISCOUNTED WHOLESALE.** ⚠️⚠️ **DO *NOT* "FIX" THIS BY DROPPING THE SLASH FROM THE REF. `main:` AND `HEAD:` DO NOT MANGLE AND RETURN THE STALE LOCAL BRANCH — measured in the shared checkout: `main`/`HEAD` = `88997178`, `FETCH_HEAD`/`origin/main` = `9d6a6959`, 18,261 bytes of the WRONG FILE against 19,398.** **THAT IS THE EXACT STALENESS THIS RULE EXISTS TO PREVENT, AND IT IS THE QUIETER FAILURE: mangling gives you ZERO BYTES, which is detectable; a stale ref gives you a PLAUSIBLE, WELL-FORMED, COMPLETE, WRONG FILE.** ⚠️ **AND THEY ARE *DIFFERENTLY* STALE PER CHECKOUT, WHICH IS WORSE THAN CONSISTENTLY WRONG: the shared tree's `HEAD` measured `88997178` while a lane's worktree measured `747a39ba` and was CORRECT BY LUCK OF THEIR BRANCH STATE.** So a *“works”* claim about `main:`/`HEAD:` **is not stable between lanes and cannot be tested once and relied on — it survives exactly the spot-check someone would run before trusting it, then fails tomorrow, or in someone else's tree today.** The safe slash-free forms are **`FETCH_HEAD` after `git fetch origin main`** (verified identical to `origin/main`) **or a bare sha** — never `main` or `HEAD`. ⚠️ **THE CLASS — NOW FOUR VARIANTS WITH FOUR DIFFERENT DETECTORS, CONSOLIDATED IN [`staleness-by-workaround` § THE FAMILY](#staleness-by-workaround) RATHER THAN RESTATED HERE, because a general taxonomy buried in a Windows-path-mangling bullet is unfindable by anyone who does not already have the problem.** Two of the four are worked examples from THIS incident: [`staleness-by-workaround`](#staleness-by-workaround) asks whether a PROHIBITION's forbidden path still fails. **(2) A PRESCRIPTION whose recommended path silently stopped working for a SUBSET of its arguments** — *"is this claim still true?"* passes, because the rule text is fine and only the instrument's DOMAIN is narrower than its phrasing; neither a currency audit nor a re-attempt sees it, because it succeeds on almost everything. **(3) A REMEDY WHOSE CORRECTNESS WAS VERIFIED ON THE AXIS IT WAS CHOSEN FOR AND NEVER ON THE AXIS IT WAS REPLACING** — the PM's formulation, and the `main:` trap is the worked example: *it does not mangle* and *it is not current* are both true, and only the first is what you test when you check whether the workaround works. **When you prescribe an instrument, state the argument shapes it does NOT cover; when you prescribe a REPLACEMENT, re-verify it on the property the original was chosen for.**

## a-guard-must-not-encode-a-false-claim

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A GUARD MUST NOT ENCODE A CLAIM THAT IS FALSE, EVEN WHEN THE BEHAVIOUR IT PRODUCES IS RIGHT (2026-08-08). Three lanes, three unrelated surfaces, one rule — recorded as a principle rather than three coincidences.** (a) `masked_fill`'s allowlist produced correct behaviour for every dtype it listed **while asserting a closed set that was not closed** — replaced by an exhaustive match. (b) The `dummy_dtype` stubs implemented `WithDType` **for types that have no numeric meaning**, so the *type system* stated a capability the *body* answered with `panic!` — deleted, not documented. (c) `fuel-cuda-backend`'s `unreachable!` arms are **unreachable in fact** — but they assert impossibility about a scrutinee (`CudaStorageSlice`) rather than about the dtype the reader assumes, so the guard is right by accident of a different fact. **The test: read the guard as a SENTENCE and ask whether the sentence is true — not whether the program behaves.** A guard is documentation the compiler enforces; **a false one is a lie with a green test suite behind it**, and it is believed precisely because the behaviour is correct. Prefer *exhaustive match* > *typed decline* > *allowlist*, and **delete a guard whose claim you cannot make true.**

## a-conformance-report-states-its-command

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **EVERY BYTE-MATCH / CONFORMANCE REPORT MUST STATE THE EXACT COMMAND AND FEATURE FLAGS IT WAS PRODUCED BY (2026-08-08).** Not "gates green", not a summary — **the command line.** Requested by the KISS architect as a standing requirement across all four sk4 token-derivers, and adopted here: **an unattributable green is indistinguishable from an unearned one.** A byte-match is a **four-way** claim, so **one leg whose provenance cannot be reconstructed contaminates the whole result** and the other parties cannot tell which leg it was. **The worked example is the rule below it:** a dtype-addition leg reported green across six crates, **none of which compiled the file carrying the wire codec**, because the feature gating it is non-default.

## a-long-cuda-build-runs-detached-with-a-marker

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A LONG CUDA BUILD MUST BE LAUNCHED **DETACHED**, WITH A SELF-WRITTEN COMPLETION MARKER — A HARNESS-TRACKED BUILD CAN BE STOPPED BY THE HARNESS (2026-08-12; two lanes paid for this 17 minutes apart, and it was already recorded in docs/gaps.md GAP-029 but NOT here, which is why it recurred).** `cargo test --features cuda` **LINKS**, so it builds the whole baracuda kernel forge (**~30–56 min quiet, 93m measured under load 2026-08-27**); `cargo check` only wants metadata and finishes in minutes. **A build that exceeds the agent's foreground window gets moved to background and can then be stopped — leaving no `test result:` line and a truncated log.** ⚠️ **A KILLED FORGE BUILD IS INDISTINGUISHABLE FROM A FAILING ONE, so this is the harness-defect-reported-as-code-defect trap in its most expensive form.** **REQUIRED SHAPE: (1) launch DETACHED so no agent-side timeout can reach it — `Start-Process`, or `Win32_Process Create`, both validated on this box against a 56-minute forge; (2) THE SCRIPT WRITES ITS OWN EXIT-CODE MARKER as its last output (`<TAG>_BUILD_DONE_EXITCODE=<code>`), because WITHOUT IT A TRUNCATED LOG CANNOT BE TOLD FROM A FINISHED BUILD; (3) capture UNFILTERED to a file and filter on read — never pipe a background build through `Select-String`, or the captured stream IS the filtered one and the errors are gone; (4) still go through `scripts/cuda-build.ps1` for the slot.** **TREAT "NO MARKER" AS "NO RESULT", NOT AS FAILURE ⚠️⚠️ **AND THE HOLE IN THIS VERY RECIPE, FOUND 2026-08-14 BY THE GAP-194 LANE AFTER FOLLOWING IT CORRECTLY: DETACHING PROTECTS THE *BUILD*. IT DOES NOT PROTECT THE *EVIDENCE*.** The wrapper `cmd.exe` can be killed while its descendants live — and **BOTH the exit-code marker line AND the `>>` log redirect live in the WRAPPER.** Observed: the harness killed the watcher, the wrapper died with it, and **34 nvcc/cicc/cargo processes kept compiling normally into a CLOSED HANDLE** — log frozen at **397 bytes**, and the final `echo <TAG>_DONE_EXITCODE=` line **now physically impossible to run.** A full forge consumed, producing no output and no completion signal. **⚠️ AND IT DEFEATS THIS RULE'S OWN PURPOSE: the marker exists so a TRUNCATED LOG cannot be mistaken for a finished build — but here the truncation and the missing marker have a COMMON CAUSE, so the marker carries no information at exactly the moment it was designed for. The resulting state is indistinguishable from STILL RUNNING, indefinitely.** **A COMPLETION SIGNAL WRITTEN BY A PROCESS THAT CAN DIE INDEPENDENTLY OF THE WORK IS NOT A COMPLETION SIGNAL.** **FIX: DERIVE COMPLETION FROM AN ARTIFACT THE BUILD ITSELF WRITES** — cargo's own `--message-format json` output file, or the test binary's mtime — **produced by the surviving process, never appended by the killable parent.** Same family as the orphaned-child hazard (killing a wrapper does not kill its cargo child, and the child holds the log handle), but pointed at the COMPLETION SIGNAL rather than at stale content. **AND THE PRE-FLIGHT FAILED THE SAME DAY IN THE SAME SHAPE: `tasklist | grep -icE` returned MALFORMED OUTPUT that READ AS A CLEAN ZERO, so a "zero peer CUDA processes" check could not rule out starting a second concurrent forge — the documented ptxas-OOM path. POSITIVE-CONTROL THE PRE-FLIGHT: run it while a process you KNOW matches is alive and confirm it reports non-zero. A CHECK THAT HAS NEVER BEEN SEEN TO RETURN NON-ZERO IS NOT A CHECK.** Fourth instance that day of a malformed instrument returning something that parses as a clean answer. — a partial log supports no conclusion in EITHER direction.** **⚠️ AND THE HALF THAT IS EASY TO MISS: THE DETACHED BUILD SURVIVES, BUT A HARNESS-TRACKED **WATCHER** OVER IT DOES NOT.** Measured 2026-08-12: a `Win32_Process Create` forge survived **37+ minutes and three watcher deaths**, while background `until`-loop watchers armed to wait for its marker were killed — one within about a minute of arming. **So: launch detached, then POLL ON YOUR OWN TURN. Do not delegate the waiting to a background task.** **⚠️⚠️ AND THE TRAP THAT MISLEADS IN THE OPPOSITE DIRECTION TO EVERYTHING ELSE IN THIS BULLET: A `killed` NOTIFICATION IDENTIFIES A TASK, NOT THE WORK.** The same word covers *the build died while the log still looked alive* and *the watcher died while the build was perfectly healthy* — **and those demand opposite responses.** The only reliable evidence is **the log's completion marker plus the process table** (`nvcc`/`ptxas`/`cicc`/`cargo` counts, and the `cuda-build` slot JSON's pid): a live pid means the work is fine no matter what the notification implied. Before relaunching, verify **zero orphaned `nvcc`/`ptxas`/`cargo`**: a leaked forge process makes both the next build's timing and the slot accounting unreadable. **Corollary observed the same day: `cuda-build.ps1`'s stale-slot reclaim fired correctly on a REAL dead holder in the wild** (*"uncontended crash — no `AbandonedMutexException` is raised when the mutex object dies with its last handle"*) — **which is exactly why it uses N mutexes and not a `Semaphore`: a Semaphore has no abandoned-holder recovery, so a dead holder would wedge the slot permanently.** ⚠️⚠️ **AND THE SELF-WRITTEN MARKER IS NOT A CUDA RULE — IT IS A HARNESS RULE, AND THIS BULLET'S CUDA FRAMING MADE IT READ AS OPTIONAL FOR EVERYTHING ELSE (2026-08-26, plain HOST clippy run, no forge, no slot).** A lane's background task notification reported **`completed (exit code 0)`** while its own marker read **`..._DONE_EXITCODE=101`**. **THE HARNESS REPORTS THE WRAPPER'S EXIT, NOT CARGO'S** — so a wrapper that launches cargo, survives, and exits cleanly reports success no matter what cargo did. **The lane had already been burned once by a truncated census and would have shipped the wrong number a second time on the strength of that notification.** **PRACTICE: write the marker on ANY detached or backgrounded build whose exit code you intend to believe, host or CUDA, ten seconds or ten minutes — the marker is the only thing that carries CARGO's exit code. A task notification is a statement about a process you did not care about.** *(Filed here rather than as its own bullet because everything above about markers, truncation and killed watchers applies unchanged; only the SCOPE was wrong — and a rule shelved under an expensive special case is a rule nobody applies to the cheap general one.)*

## fuel-onnx-needs-protoc

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **`fuel-onnx` needs `protoc`, which is installed but NOT on the inherited PATH (2026-08-01).** Its `build.rs` calls `prost_build::compile_protos` and prost-build 0.13.5 does not bundle a `protoc` (dropped in 0.11), so without one `cargo check -p fuel-onnx` fails at **exit 101 before compiling a single line**. `protoc` (libprotoc 35.1) was installed via WinGet to `C:\Users\cires\AppData\Local\Microsoft\WinGet\Packages\Google.Protobuf_Microsoft.Winget.Source_8wekyb3d8bbwe\bin` and is in the persisted **User** PATH — but a shell spawned from a long-running parent inherits a **stale environment**, so prepend that directory (or set `PROTOC`) in any cargo invocation that touches fuel-onnx until the parent process is restarted. **⚠️ AMENDED 2026-08-20 — CHECK THE CONDITION BEFORE APPLYING THE WORKAROUND; it is currently NOT needed.** Measured: `command -v protoc` → **found**, `PROTOC` unset and unnecessary. **This rule names its own expiry — *"until the parent process is restarted"* — and that expiry has arrived, while the rule still reads as live.** It is kept rather than deleted because **the obstacle genuinely recurs**: a shell spawned from a long-running parent inherits a stale environment, so the next session may hit it again. **The check is one command; run it instead of assuming either way.** `fuel-onnx` is not a `default-member`, so a root `cargo check` stays green while it is broken; check it explicitly.

## run-bare-cargo-the-toolchain-is-pinned

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ `+stable` IS A MOVING ALIAS, SO "VERIFY WITH `+stable` TO MATCH CI" IS UNDERSPECIFIED — TWO MACHINES' `stable` DIFFER (2026-08-20).** This box's `stable` is **rustc 1.97.1 (2026-07-14)**; CI uses `dtolnay/rust-toolchain@stable`, which resolves **fresh at run time**. CORRECTED 2026-08-21 (`97c8addd`) — THIS SENTENCE IS NOW FALSE: a pinned `rust-toolchain.toml` (`channel = "1.98.0"` + components) EXISTS and makes box and CI agree, confirmed effective on CI (rustup names the override in its own log). The "(checked)" stamp is exactly what let this absence claim outlive its truth — a verification stamp decays with the claim while looking re-verified. The `+stable`/`+1.98.0` GUIDANCE in the rest of this bullet is superseded by the pin (post-pin, run bare `cargo` with NO `+toolchain` — an explicit `+toolchain` OVERRIDES the pin at rustup precedence 1 > 4) — **REWRITE DISCHARGED 2026-08-25 by the architect, and the operative rule is this sentence, not the reasoning below: RUN BARE `cargo` / `cargo fmt` / `cargo clippy`, WITH NO `+toolchain` EVER. `rust-toolchain.toml` is the single source of truth; an explicit `+toolchain` BEATS it (rustup precedence 1 over 4), so `+1.98.0` LOOKS LIKE DILIGENCE WHILE SILENTLY PINNING YOU TO A STALE COMPILER THE DAY THE FILE BUMPS — with CI moving and your local greens diverging from it, which is precisely the condition the old order existed to eliminate. THE VERSION LIVES IN THE FILE AND MUST NEVER BE REPEATED HERE.** ⚠️ **CONFIRM THE FILE IS ON DISK BEFORE TRUSTING BARE `cargo`** — a checkout that predates `97c8addd` does not have it, and there the box default is **1.99.0-nightly**, not stable and not the pin. ⚠️⚠️ **AND THE CLASS THIS BELONGS TO IS WORTH MORE THAN THE RULE: A DEFENCE CAN OUTLIVE ITS DEFECT AND THEN BECOME ONE.** *“Use `+1.98.0`, never `+stable`”* was correct and load-bearing while nothing pinned and the ambient default was nightly — it was the only thing standing between a lane and a wrong-compiler measurement. **The pin made it STRUCTURAL, and a remembered control that has been replaced by a structural one does not go neutral: IT COMPETES WITH ITS REPLACEMENT.** **This is sharper than an ordinary stale rule, because it does not merely stop being useful — it ACQUIRES THE OPPOSITE EFFECT, at a future date nobody will be watching for, and it will look like compliance while doing it.** **Note also what did NOT catch it: a full doc-currency sweep had passed 35/35 and 7/7, and was CORRECT when run. The line went stale afterwards, at `97c8addd`.** A complete currency audit does not immunise a corpus against a line that expires after the audit — see [`staleness-by-workaround`](#staleness-by-workaround), of which this is the mirror image: there a rule stays true and stops being cheapest; **here a rule stays grammatical and starts being harmful.** Live instance: **CI's Clippy fails on 13 instances of a lint that postdates the local toolchain**, so `cargo +stable clippy` returning **exit 0 here is not evidence about CI**. **This is the `--edition` trap one level out** — there, bare `rustfmt` defaulted to edition 2015 and manufactured diffs a `cargo fmt` never sees; here, a toolchain alias resolves to different compilers on different machines. **In both cases an invocation that LOOKS like the gate is a different gate.** When a rule says "match CI", name the property that must match (the toolchain DATE, the edition) — an alias is not a specification.

- **⚠️ THE ONE REAL EXCEPTION TO "NEVER `+toolchain`": WHEN THE TOOLCHAIN DIFFERENCE IS THE THING BEING MEASURED (2026-09-24).** Codacy flagged fuel's own "never `+toolchain`" bullet as an absolute rule with no escape hatch after this file's `CLAUDE.md` follow-up split it out on its own — and it was RIGHT, because auth-framework's lane had, the same night, run `cargo +stable clippy --all-targets` **and** `cargo +beta clippy --all-targets` side by side, deliberately, to compare them, and found **92 `double_must_use` errors that exist only on beta** — a real defect invisible to the pinned toolchain, undiscoverable WITHOUT overriding the pin. It then verified a `clippy.toml` fix on **both** toolchains before that PR was merged. **Overriding the pin is exactly what you want when the question is "does the pin hide something?" — there the override is the instrument, not a mistake.** **THE OBLIGATION THAT MAKES IT SAFE: say so at the call site.** A `+toolchain` invocation with no stated reason is indistinguishable from the careless kind this rule exists to forbid; one made as a DECLARED cross-toolchain comparison, with both results reported, is not the same act even though the command line looks identical. **The rule is therefore stronger with the exception named than without it: "never, unless the toolchain difference is what you are measuring — and then say so" cannot be broken silently by someone who genuinely needs the comparison, where a bare "never" can only be followed or quietly violated.**

## this-box

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- Environment (as of 2026-06-13; GPU line re-measured 2026-07-30): Windows 11, **RTX 4070 Laptop GPU (8 GB)** + an AMD integrated GPU reachable via Vulkan, CUDA 13 + Vulkan SDK installed, Rust 1.96 / edition 2024. `fuel-dispatch` checks clean (warnings only). The iGPU matters for tri-backend work: `fuel::vulkan_backend::new_device()` uses `PreferDiscrete` and hands back the **NVIDIA** card, so a test that wants CPU+CUDA+Vulkan on three *distinct* vendors must select the Vulkan adapter explicitly or it silently becomes a same-vendor test that still passes.

## tdd-and-the-retained-sabotage-sibling

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **Test-driven development is the default.** Write the failing test first, watch it go red, then make it green. "Born-red" tests are the *goal*, not an accident to apologize for. ⚠️⚠️ **AND A BORN-RED EXPIRES THE MOMENT IT GOES GREEN — KEEP THE SABOTAGE AS A PERMANENT SIBLING TEST. Reached INDEPENDENTLY BY TWO LANES ON THE SAME DAY (2026-08-14), on unrelated rows and with no contact, which is why it is a rule and not a preference: `mlp_loss_is_flat_when_learning_rate_is_zero` (GAP-198, MNIST convergence) and Gemma3's `n_variants()`-tracks-the-fixture sibling (GAP-029).** **A born-red proves the gate discriminated ONCE, AT AUTHORING TIME — it is an observation about a moment. A RETAINED SABOTAGE SIBLING PROVES IT STILL DISCRIMINATES ON EVERY RUN.** The failure it catches is specific and silent: **a later edit that detaches the graph, collapses the fixture, or removes the variation makes the ORIGINAL test go green for the wrong reason, and nothing reports it** — the test still passes, the suite is still green, and the only thing that changed is that the assertion stopped being about anything. **The sibling must FAIL if the pipeline is inert, so a vacuous green becomes a red somewhere.** Cheap to write while the sabotage is already in your hands; nearly impossible to reconstruct later, because by then nobody remembers what the gate was supposed to be sensitive to. The historical failure mode — batch commits verified with `cargo check` only, shipping tests that never ran — is banned. A change that touches behavior ships with the test that exercises it, and that test must have been observed to run.

## the-recipe-principle-is-a-build-time-invariant

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **The recipe principle / total `decompose` is a build-time invariant (G1/G2/G3).** Every fused op ships with BOTH a `decompose` (fused → primitive subgraph) and a `pattern` (re-fuse); `decompose` is total + never-panic + primitive→self (base map = its fixpoint); the primitive `Op` basis is build-time-closed. A non-basis op that won't decompose is a surfaced opaque-op gap (telemetry), never a crash — and it breaks the optimizer itself (optimization = lower-to-base-map + find-best-cover). See [docs/architecture/10-decisions-log.md](architecture/10-decisions-log.md) (2026-06-20 "Adaptive runtime fusion").

## wip-on-a-branch-and-no-gpu-ci

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **WIP/unverified work goes on a branch, not `main`.** With CI currently red, "main builds" is only a convention — keep it true. **Corrected 2026-08-13: "CI is red" names TWO DIFFERENT PROBLEMS and this line used to hide the worse one.** `rust-ci.yml` genuinely *runs* — 6–12 steps execute per job — and fails on content (`cargo fmt --check`, `clippy -D warnings`, real check/test steps). `ci_cuda.yaml` **never executed a single step in its entire history** (16 runs, 2026-04-10 → 2026-08-13, `steps: 0` on every one) because `runs-on: group: aws-g5-4xlarge-cache` is **HuggingFace's** runner group, inherited from the Candle fork. **It was DELETED 2026-08-13 on CireSnave's ruling — *"if a Workflow designed to run in CI requires CUDA, it will never run; remove it"* — so that a red X could no longer stand in for coverage that never existed.** **`cargo test --features cuda` has therefore never run in CI, and every claim of the form "CUDA is tested in CI" is false: CUDA verification here is local-only through `scripts/gpu-run.ps1`.** ⚠️ **Corollary that outlives the workflow: DO NOT ADD A CI JOB THAT NEEDS A GPU — there is no GPU runner. A job needing only the CUDA *SDK* (compile-only, e.g. `--features telemetry,baracuda-types`) is a different and possibly viable case, since `baracuda-cuda-sys`'s build script forges nothing; but it is UNDEMONSTRATED and must be seen to fail on a seeded error before being described as coverage.** See GAP-181.

## a-crate-compile-line-is-not-a-module-compile-line

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ THE TARGET-CRATE `Checking` LINE PROVES THE *CRATE* WAS REACHED — NEVER THAT A `cfg`'d *MODULE* WAS IN THE BUILD (2026-08-13). THIS DEFEATS THE ARTIFACT RULE AS WRITTEN, AND THE GREEN IS BYTE-IDENTICAL TO A CORRECT ONE.** A lane verified a `baracuda_provider.rs` change with `--features telemetry,cuda` and got **exit 0 with a real `Checking fuel-dispatch` line** — proving nothing, because `mod baracuda_provider` is cfg'd on **`baracuda-types`** (GAP-173 split it off `cuda`), so the crate compiled and **the module was never parsed**. A latent `E0004` sat under it. **The artifact rule's own worked example is this file, and the rule still does not catch this.** **PRACTICE: for a `cfg`'d module, the required artifact is something ONLY THAT MODULE CAN EMIT — a diagnostic from it, or a test that exercises it. A crate-level compile line is not evidence about a module. And read the `cfg` AT HEAD rather than from a remembered description: the correct gate here is `--features telemetry,baracuda-types` (pure host, no forge, seconds), and a stale "telemetry AND cuda" sent two people to a GPU-class build for an answer it could not give.**

## cargo-fmt-does-not-touch-doc-comments

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ `cargo fmt` DOES NOT TOUCH DOC-COMMENT CONTENT, so a doc-markup lint passes the local fmt hook and fails ONLY in CI.** An overindented list, a tab, or a lazy continuation inside a `///` / `//!` block is invisible to `rustfmt` and caught by `clippy -- -D warnings`. **Measured from the hook's own source: `scripts/hooks/pre-commit` invokes `rustfmt` 12 times and `clippy` ZERO times**, so the local gate is STRUCTURALLY blind to this class rather than merely lenient — **fmt-clean is not clippy-clean for docs.** Run `cargo clippy -p <crate> --tests` on any crate whose doc comments you edited before pushing. *Found and reported by Fuel 1, PR #35 commit `dbd354d2`: a two-column comparison in a module doc comment, indented 7-10 spaces, passed the fmt hook and failed CI on `doc_overindented_list_items`; fixed by fencing it as a `text` block, since it was preformatted content rather than a list.*

## a-clean-rebase-says-nothing-about-a-data-gate

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A CLEAN REBASE SAYS NOTHING ABOUT A GATE WHOSE SUBJECT IS *DATA* — AND IT IS WORSE THAN THE CODE CASE, BECAUSE THERE IS NOTHING TO CONFLICT (2026-08-13, self-reported by the lane that did it).** "Clean rebase ≠ compatibility" is already understood for code. **The sharper case: a DATA-DIFFERENTIAL gate — parse a corpus two ways and diff — is a claim about the CORPUS AT A COMMIT, and a rebase can replace the corpus with ZERO conflicts, because changed input files merge cleanly by construction.** Observed on GAP-182: a push fired immediately after a clean rebase that had brought in **11 changed `.fkc.md` contracts plus `lower.rs`** — precisely the inputs the 114-file differential had been run against — so the evidence licensing the push was, at the moment of the push, about a corpus that no longer existed. **Re-verified after pushing and it was still identical, so no harm; the point is that the ordering was luck.** **PRACTICE: if a gate's subject is a set of files rather than the code under test, list those paths, and re-run the gate BEFORE pushing whenever a rebase touched any of them. A rebase that reports no conflicts has told you about mergeability, not about whether your measurement still stands.** Same family as *gate must match the change kind*, one level out: here the gate was right, the change kind was right, and **the gate's INPUTS moved underneath a result that had already been computed.**

## a-log-is-evidence-only-with-its-writer

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **A LOG IS NOT EVIDENCE UNTIL YOU KNOW WHICH PROCESS WROTE IT AND WHEN (2026-08-13, self-inflicted, and it produced the same wrong verdict twice).** A lane killed a wrapper `cmd`, which did **not** kill its `cargo` child; the child kept the log handle open, so `Remove-Item` on the log **silently failed**, and the next run appended — leaving **old content with new output beneath it**. Reading the top, the lane twice concluded *"same failure"* from what were in fact **stale results with identical timings**. Caught by checking the log's **mtime** and the **PID in the panic line**. This is the companion to *a `killed` notification names a task, not the work*: there, the process outlived the notification; **here, the OUTPUT outlived the process, which is the more deceptive direction because the file is right where you left it and looks fresh.** **PRACTICE: before reading a captured log as evidence, confirm mtime moved and that the process identity in it matches the run you think you launched — and never assume a delete succeeded on Windows while any handle is open.**

## a-struct-field-add-breaks-constructing-crates

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ A STRUCT-FIELD ADD BREAKS *CONSTRUCTING* CRATES, AND THE DEFINING CRATE'S GATE IS STRUCTURALLY INCAPABLE OF SEEING IT — TWICE IN TWO DAYS, TWO DIFFERENT LANES, AND THE COORDINATOR VERIFIED THE SECOND ONE WITH THE BLIND INSTRUMENT (2026-08-13).** `66c09ec4` added `pub instance` to `SmolLm3Weights` and did not update `fuel-examples/examples/smollm3/main.rs`, which constructs it: **`-p fuel-examples --all-targets` failed `E0063 missing field` at head, while `-p fuel-core --lib` — what the architect ran and reported as *"verified at head"* — was green and says nothing about it.** The same trap hit an earlier family port via `gte-qwen` / `stella-en-v5`. **PRACTICE: for a field add, gate `--all-targets` on every CONSTRUCTING crate, not the defining one — find them with a grep for the struct name, not by intuition — UNLESS PRIVACY DISCHARGES IT, WHICH IS A STRONGER ARGUMENT THAN THE GREP AND SHOULD BE CHECKED FIRST (2026-08-21). If EVERY field is private, an external struct literal, an external exhaustive pattern, and external functional-update syntax are all **impossible by Rust's visibility rules** — outside crates can only call constructors — so the defining crate's own gate covers the change COMPLETELY. Worked example: `CudaDevice` gaining `Arc<DeviceLiveness>`; all fields private (`blas` is `pub(crate)`), the sole literal is `Self { … }` in its own `new_from`. **And note the grep was the WEAKER instrument twice over: its hits were false matches (`-> CudaDevice {`, `impl CudaDevice {`), so an enumeration would have produced work AND missed the reason none was needed. A language-level guarantee beats an enumeration — ask what the compiler FORBIDS before asking what a search FINDS.** Positive control that the search could see the type at all: `CudaDevice::new` calls exist widely; it is construction-BY-LITERAL that privacy forbids — and note that `fuel-examples` is a constructor for most model weights structs.**

## check-is-not-a-test-build

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ `cargo check` BUILDS METADATA; `cargo test` BUILDS OBJECT CODE AND LINKS — SO A "WARM" `check` VERIFICATION DOES NOT PREDICT A `test` BUILD (2026-08-21, GAP-214).** A lane verified increments with `cargo check -p fuel-dispatch --features jit,cuda --lib` in **50 seconds**, then the live `cargo test` run needed **31m45s of codegen the check had skipped entirely** — and the first attempt died in the foreground on that gap, leaving a stale `gpu-run` lock (which the mutex then correctly reclaimed, the best evidence yet that its crash recovery is live). **The codegen is INVISIBLE in every prior green**, so "it compiles fast" is not evidence about how long the test will take. **PRACTICE: budget a live `--features cuda` TEST run from the codegen cost, never from a `check` timing; and launch it DETACHED from the first attempt rather than after a foreground timeout.** Related and larger: **commit WIP before ending a turn even mid-verification** — a pushed branch is visible, a dirty worktree is a single point of loss, and a portfolio sweep the same night found a **five-day-old, 655-file uncommitted worktree that no agent claimed.**

## clearing-a-fingerprint-by-guessed-path

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ CLEARING A FINGERPRINT BY GUESSED PATH CAN MATCH NOTHING, LEAVING YOU WARM WHILE BELIEVING YOU FORCED A REBUILD (2026-08-13).** `rm -rf target/debug/.fingerprint/fuel-examples-*` matched **zero** directories, so the follow-up `cargo check` returned `exit 0` with **zero `Checking` lines** — the warm-cache row — and would have been read as a clean verification of a crate that had just been red. **The reliable forcing function is to `touch` a SOURCE FILE in the crate** (ideally the one under suspicion): the rebuild is then guaranteed and the `Checking <crate>` line is a real artifact. **If you clear a fingerprint, confirm the glob actually matched something before trusting the result.**

## a-numeric-golden-cannot-see-node-growth

> Moved verbatim from `CLAUDE.md` on 2026-09-16 (base `c4936cfc`). The operative rule is the one-line `CLAUDE.md` bullet that links here; this is its full text.

- **⚠️ A NUMERIC GOLDEN CANNOT SEE NODE GROWTH — A BEHAVIOUR-PRESERVING SEAM CHANGE NEEDS A STRUCTURAL WITNESS TOO (2026-08-13).** `mul_scalar(1.0)`, or a slice+reshape a single-variant family should never emit, is **numerically invisible and structurally real**: it costs a node in **every held decode plan for the life of the session**, and a 1e-6 logits golden sails straight past it. **INSTRUMENT: capture GRAPH NODE COUNTS for representative carriers BEFORE the change and assert them after.** Worked example, adding `RopePlan` + `embed_scale` to the decode seam: `llama 186 -> 186`, `qwen2 150 -> 150`, `phi3 130 -> 130`. **Choose the STRICTEST witness deliberately: Phi3 is single-variant on BOTH axes and has no embed scale, so it is the one that moves if `None` emits a multiply or `n == 1` emits a slice+reshape — a carrier that already has the feature cannot witness its accidental addition.** Pair it with the numerical golden; they fail on disjoint defects. **This is why `embed_scale` is `Option<f64>` rather than an `f64` defaulting to `1.0`: `Some(1.0)` would still emit a node and break the counts, so the type carries the no-op case instead of the arithmetic.**

# Fuel — method rules (the long form)

**This file is the EVIDENCE, not the rule.** Every entry below is indexed by a
one-line rule in [`CLAUDE.md`](../CLAUDE.md), which is what loads into every
session. The one-liner is the operative instruction; this file is why it exists,
what it cost to learn, and the measurements behind it.

**Split out 2026-08-14** because CLAUDE.md had reached ~19,600 words in 66
bullets — one of them 2,074 words — and a working agreement nobody can scan is a
working agreement nobody follows. Nothing was rewritten or summarised in the
move: every entry is verbatim, so no evidence was lost and the incident detail
stays greppable. **The criterion was METHOD bullets (verification epistemics,
not commands/paths/hardware) of >=300 words — operational rules stayed in
CLAUDE.md IN FULL**, because their detail is needed to run a command correctly
and a one-line index would not substitute for it.

⚠️ **If you are acting on a rule, read the entry — the one-liner is a pointer, not
a summary you can safely reason from.** Several of these rules exist precisely
because someone acted on a compressed version and got the scope wrong.

---

## injectivity-and-collapsed-mappings

> **Index line (in CLAUDE.md):** **Exhaustiveness gives completeness; nobody checks INJECTIVITY.** Two inputs mapping to one output is silent all the way down, and a false agreement gets FILED where a disagreement gets investigated. Demand injectivity where the output is an IDENTITY (a wire token), not where it is a CLASSIFICATION. For a decline reason: a distinction stated in prose MUST be carried in the type.

**EXHAUSTIVENESS GIVES YOU COMPLETENESS; NOBODY CHECKS INJECTIVITY — AND A COLLAPSED MAPPING AGREES WITH A REFERENCE FOR THE WRONG REASON (2026-08-12, KISS architect's framing).** A new variant falling through a catch-all is loud once you look. **Two variants mapped to the SAME output is silent all the way down: green build, complete report, two verdicts collapsed into one — and strictly worse than a disagreement, because a disagreement gets investigated and a false agreement gets filed.** **FUEL'S LIVE INSTANCE IS GAP-161:** `fuel-cuda-backend/src/storage.rs:3023-3039` spells **three distinct decline reasons** in comments and returns the **identical** `UnsupportedDtype` for all three. The precise name for that is not *"reasons erased into prose"* — it is **the reason→value mapping is not injective**, which is better because it says what instrument catches it. **⚠️ THE DISTINCTION THAT DECIDES WHERE THE INSTRUMENT IS REQUIRED: INJECTIVITY IS MANDATORY WHERE THE OUTPUT IS AN *IDENTITY*, OPTIONAL WHERE IT IS A *CLASSIFICATION*.** `sk4_token` is an identity — it NAMES the dtype on the wire, so a collision means two operand shapes share a `structure_key`, a correctness bug. `map_element_kind(DType) -> Option<ElementKind>` is a classification: **two Fuel dtypes legitimately COULD map to one backend kind if the backend does not distinguish them**, so requiring injectivity there would encode a false constraint. **Demand it of mappings onto a wire identity; do not demand it universally.** **⚠️ AND A DECLINE REASON IS A THIRD CASE THAT FITS NEITHER — many causes legitimately share one verdict, so the rule for diagnostics is NOT injectivity but: A DISTINCTION STATED IN PROSE MUST BE CARRIED IN THE TYPE** (KISS architect, KISS #167). The author knew the distinctions, wrote them down, and shipped an interface that cannot express them — **the comment claims a resolution the type does not have.** **Greppable instrument: if a comment says "distinct from X" / "NOT because", check the returned value.** Measured on Fuel: **137 candidates in production Rust → 6 hits / 5 sites within 5 lines of a decline return, positive control (`fuel-cuda-backend/src/storage.rs:3027`) surviving the narrowing, 96% exclusion; the 5-line window is a LOWER bound** because a comment can precede a long `match`. **⚠️⚠️ "ONE RETURN VALUE" IS NOT THE TEST — AND NEITHER IS "IS THE DISTINGUISHED DIMENSION A FIELD", WHICH WAS MY REFINEMENT AND WAS REFUTED BY MY OWN MEASUREMENT: both backends carry `dtype`+`op` and NEITHER carries a reason, so that test flags the CORRECT site too. THE TEST IS WHAT THE COMMENT CLAIMS ABOUT ITS OWN OUTPUT.** `fuel-metal-backend/src/storage.rs:128` joins six dtypes into one decline arm and is **CORRECT**, because `UnsupportedDTypeForOp(self.dtype, op)` is **dtype-parameterised** — six joined arms, six distinct values — and its comment claims only that the join *"asserts nothing false about the format"*, which is the weaker and honest claim. Both backends carry `dtype` and `op`; **NEITHER carries a REASON**, and GAP-161's comment distinguishes *why*. **So the defect is: CUDA asserts three reasons as facts about three declines that are ONE value — a resolution the type does not have — while Metal claims only that its join "asserts nothing false", which is weaker and TRUE. LOSSY-AND-HONEST IS FINE; LOSSY-WHILE-ASSERTING-OTHERWISE IS THE DEFECT.** **PRACTICAL SHAPE, and its bound: TWO STAGES — FILTER BY MACHINE (the grep, as measured), JUDGE BY READER (does the comment claim the output distinguishes?). Stage 2 is not mechanizable and should not pretend to be; mechanization is the floor here, not the ceiling.** **A comment that DOCUMENTS its own loss instead of denying it is rarer than one that is simply right, and more useful — it tells the next reader what they may NOT conclude.** **AND NOTE HOW FUEL'S sk4 CASE IS ACTUALLY COVERED, because it is fragile: `token_kind.rs` asserts a token **`BTreeSet`** has cardinality 14 (`:172`) while separately counting the **dtypes** that produce `Some` as 14 (`:216`) — set-cardinality on one side, population-count on the other, which IS an injectivity check. But nobody wrote "assert injective," so a future refactor can delete the coverage without knowing it existed. NAME THE PROPERTY THE ASSERTION PROTECTS.** ⚠️⚠️ **AND THE SAME FAILURE ONE LEVEL OUT, FOUND BY KISS 2026-08-14 AND WORTH MORE THAN THE BUG IT CAME FROM: AGREEMENT BETWEEN TWO *CONSUMERS* IS NOT AGREEMENT WITH THE *AUTHORITY*.** KISS's reference artifact publishes `vulkan:sg64.ops-abr.arith-f16.cm-none` — **four fields — while `spec/namespaces/vulkan.md` went to Vocabulary v4 four weeks earlier and rule V-1 requires exactly FIVE**, none omissible. Fuel's sk4 byte-match leg compared that vector and reported **20/20 green**. **THE BYTE-MATCH DID NOT FAIL — IT AGREED. Two implementations matched each other while both disagreed with the document of the maintainer who owns the vocabulary, because NEITHER INSTRUMENT WAS POINTED AT `vulkan.md`.** **A disagreement gets investigated; A FALSE AGREEMENT GETS FILED** — same asymmetry as the collapsed mapping above, and the same reason a null result reading *clean* is the dangerous direction. **THE OPERATIVE SENTENCE: A BYTE-MATCH'S ENTIRE STRENGTH IS THAT ITS EXPECTED VALUE HAS AN AUTHOR WHO IS NOT YOU — AND THAT HOLDS ONLY WHILE THE AUTHOR'S DOCUMENT AND THE ARTIFACT ARE THE SAME AGE. A match against something you did not write is only as fresh as the thing you did not write.** ⚠️⚠️⚠️ **AMENDED HOURS LATER — THE RULE ABOVE UNDERSTATES IT, AND THE KISS ARCHITECT FOUND THE STRONGER FORM BY TRACING THE STRING. FOR THE `target` FIELD THERE WERE NEVER *TWO DERIVATIONS TO AGREE*.** That token traces to a **single hand-written Rust literal**, authored once (`conformance/src/reference_vectors.rs:202`); the artifact is generated from it, the spec quotes it, and Fuel's fixture was copied from it. **Fuel interpolates the target verbatim. So the byte-match over that field COMPARED A STRING AGAINST COPIES OF ITSELF.** No instrument on either side reads the namespace doc's `Vocabulary version` header or validates a target against its grammar — **so a token that was malformed THE DAY IT WAS WRITTEN is equally invisible, and a version stamp catches the bump but not that.** **THE OPERATIVE TEST, AND IT IS PER-FIELD RATHER THAN PER-VECTOR: FOR EACH FIELD, DOES EACH PARTY *DERIVE* IT OR *COPY* IT? A conformance claim's strength is not uniform across the row it is reported for.** Here **19 of 20 vectors were byte-match evidence and the 20th was PROVENANCE evidence wearing the same shirt** — and *"20/20"* reads as twenty independent agreements. **This is the same defect as *the construct is invisible in the number*, applied to a CONFORMANCE score: the arithmetic is right, and what each unit MEASURED differs.** **AND THE INVERSION WORTH MEMORISING, because it is counter-intuitive and it decides where to spend effort: VULKANE'S SINGLE PINNED EXPECTED VALUE BEAT FOUR INDEPENDENT LEGS — because it was the only one comparing against a DOCUMENT instead of against us. ONE COMPARISON AGAINST THE AUTHORITY IS WORTH MORE THAN FOUR AGAINST EACH OTHER, and adding a fifth implementation would not have found this.** **Corollary for Fuel's own tests: a passthrough field's byte-match proves only *nobody mangles it* — a real and worth-having property (our truncation sabotage proves it discriminates) — but it is NOT a well-formedness check on the value, and must never be cited as one.** **PRACTICE: a conformance artifact must record the version of every FOREIGN vocabulary it embeds, not just the spec commit it was generated from ⚠️⚠️ **AND A PROVENANCE STAMP PROVES *BINDING*, NOT *CURRENCY* — MEASURED ON MY OWN TEST 2026-08-14, WHICH HAD BEEN PASSING THROUGHOUT.** `corpus_is_the_artifact_this_leg_was_bound_to` asserts `source_commit == KISS_SOURCE_COMMIT` plus schema, counts and set sizes. **The constant and the vendored blob were updated together, so they agree — and the test proves THEY AGREE, not that either is current.** Fuel's corpus sat at `decline_vectors: 10` while upstream had moved to 15 and then 17: **seven declines behind, green the entire time.** **SO THE ASSERTION GUARDS AGAINST SWAPPING THE FILE WITHOUT UPDATING THE CLAIM, AND IS STRUCTURALLY INCAPABLE OF DETECTING BEING BEHIND UPSTREAM.** That is the two-consumers-agreeing failure turned one step inward — **the two consumers being ME AND MY OWN CONSTANT.** **A LOCAL CHECK CAN ONLY EVER PROVE INTERNAL CONSISTENCY; CURRENCY REQUIRES COMPARING AGAINST THE SOURCE.** Note also that a per-namespace *vocabulary version* would NOT have caught this one — it was a decline-SET drift, not a vocabulary drift — so the two axes need separate detectors. **PRACTICE: for any vendored conformance artifact, either a CI step fetches the published sha and compares, or the currency question is OPEN and must be stated as open. "Our provenance assertion passes" is not an answer to "is our copy current?", and it reads exactly like one.** ⚠️⚠️ **AND THE SHARPEST FORM OF THE TEST CAME BACK FROM MLMF, WHO GENERALISED THE INCIDENT FURTHER THAN I HAD: *THE ASSERTION MUST BE "RE-HASH THE SOURCE AND COMPARE", WHICH IS FALSIFIABLE. FIELD EQUALITY IS NOT.*** Two stored constants compared against each other can only ever restate what someone wrote down; **a hash of content you can go re-read is the version that survives the incident**, because the source is a party to the check. **Use it as the discriminator on any provenance mechanism: CAN THIS ASSERTION FAIL WITHOUT A HUMAN CHANGING A CONSTANT? If not, it is a binding record wearing a currency check's clothes.** **AND THEIR EXTENSION IS THE ONE THAT SHOULD WORRY US, BECAUSE FUEL HAS THE SAME SHAPE IN MORE THAN ONE PLACE: A CONFORMANCE CORPUS FROZEN AT TODAY'S ARTIFACTS STAYS GREEN INDEFINITELY WHILE SILENTLY CEASING TO DESCRIBE THE WORLD.** Their instance: a checkpoint corpus asserting *"MLMF understands everything here"* will still pass in a year while failing to understand every metadata key shipped in the meantime — **the green decays into meaninglessness and nothing announces it.** Ours is the same object one layer in: vendored conformance vectors, golden fixtures, recorded model configs. **PRACTICE: CORPUS CURRENCY IS A SEPARATE OBLIGATION FROM CORPUS AGREEMENT AND MUST BE ABLE TO FAIL ON ITS OWN. Record where each entry came from and WHEN; make staleness its own failing check rather than something inferred from a passing one.** A green agreement test and an absent currency test are not two thirds of a guarantee — **the second one's absence silently caps what the first one can mean.** — a single `source_commit` DETECTS a change and cannot DIAGNOSE it (one value standing for many independent axes), and a vocabulary owned by someone else moves on its own clock.** **AND RECORD THE VERSION EVEN WHEN IT IS UNCHANGED: a field that appears only when something moves is indistinguishable from a field nobody remembered to write.** ⚠️ ****⚠️ THE ATTRIBUTION HERE WAS WRONG IN ITS FIRST VERSION AND THE CORRECTION IS THE MORE USEFUL RULE. I originally wrote that the exclusion was an error in MY notes which the KISS architect INHERITED FROM ME. Not so — the architect pressed the point rather than accepting a graceful offer, with three independent confirmations: my own measurement (Fuel CONSTRUCTS and compares that vector, `kiss_structure_key_byte_match.rs:290`, ZERO vulkan exclusions — Fuel was 20/20 with no exclusions at all); their dated memory file recording the exclusion against **Unpopped** BY NAME; and Unpopped owning and closing it. THE PROPAGATION WAS ENTIRELY WITHIN KISS. I VOLUNTEERED FOR A LINK IN A CHAIN I WAS NEVER IN — and was about to set it in a working-agreement file, which is precisely the artifact class this whole exchange was about: ACCURATE ABOUT THE LESSON, WRONG ABOUT THE PROVENANCE, AND NOTHING DOWNSTREAM RE-DERIVES IT.** **MY ACTUAL ERROR WAS THE MIRROR OF THE ONE I CLAIMED, AND IT IS WORTH MORE: A PEER DESCRIBED MY OWN CODE TO ME, I AGREED IT MATCHED MY NOTES, AND ONLY MEASURED LATER. Accepting someone else's characterisation OF YOUR OWN PROJECT and restating it as your own record is how a foreign error acquires local corroboration — and it is the cheapest possible thing to check, because the code is right there.** **AND THE MECHANISM THE ARCHITECT NAMED, WHICH IS SHARPER THAN 'WATCH YOUR PROPAGATION' AND APPLIES DIRECTLY TO THIS REGISTRY: A CORRECTION THAT LIVES IN A CONVERSATION AND A FACT THAT LIVES IN A FILE DIVERGE SILENTLY, AND THE FILE WINS.** Unpopped corrected them once, in conversation; the correction was never written back; the FILE is what got re-read, and the stale figure travelled from there. **NO CONTROL COULD FIRE — nothing in the repo changed at the transition, so there was no artifact to disagree with.** It was caught the only way this class ever is: **CONTRADICTION WITH AN INDEPENDENT FINDING**, when Unpopped re-measured. **THE FIX IS NOT TO BE MORE CAREFUL — IT IS TO EDIT THE FILE IN THE SAME TURN AS THE CORRECTION.**

⚠️⚠️ **THE SAME ASYMMETRY IN QUANTIFIERS, AND IT IS WHY THE INDEX-JOIN GUARD NEEDED TWO MORE ARMS (2026-09-02, self-inflicted, and only building the second arm exposed it).** Exhaustiveness is an **∃** claim — *every section is reachable from SOMEWHERE* — and ONE good instance satisfies it. **The obligation a reader actually depends on is a ∀ claim: every CITATION works.** In `fuel-ir/tests/method_rules_index_join.rs`, arm C asks the ∃ question per SECTION (*is there at least one link whose TARGET carries this anchor?*) and arm D asks the ∀ question per LINK (*does every link carry an anchor?*). **NEITHER IMPLIES THE OTHER, AND IT FAILS IN BOTH DIRECTIONS: a rule cited five times with four anchors PASSES arm C while the fifth citation still drops the reader at the top of a 1,600-line file; a section carrying no link at all PASSES arm D vacuously.** **THE MEASUREMENT ERROR THIS PRODUCED IS THE POINT: I reported "2 instances" of the anchor defect. There were 2 unanchored LINKS and 1 unanchored SECTION — right about one construct, wrong about the other, and I had named neither.** That is CLAUDE.md's *"the number was correct and the construct it counted was invisible in it"* arriving through a quantifier: **an aggregate satisfied by ONE good member is structurally unable to count its bad ones, so an ∃-shaped gate reports a figure that is not about the population anyone is relying on.** **PRACTICE: when a gate covers a RELATION, state its QUANTIFIER — per-what, and ∃ or ∀ — then ask whether the obligation you actually hold is the other one. Where both matter they are TWO ARMS, not one, and must be reported separately because they fail on disjoint defects.** **⚠️ ARM D CARRIES A NAMED KNOWN BOUND, recorded here rather than in a commit message because a bound nobody can find is a bound nobody honours: it forbids a link to `docs/method-rules.md` AS A DOCUMENT — a deliberate whole-file reference with no section intended. ZERO exist today, which is what makes it safe NOW and not safe FOREVER. If one is ever legitimately wanted, AMEND ARM D; do not contort the link to satisfy the gate.**

---

## lib-does-not-build-tests

> **Index line (in CLAUDE.md):** **`--lib` does not build `tests/`, so a `--lib` gate is blind to every integration test.** When reporting a tree healthy, state the TARGET KINDS as well as the features — an unqualified "green" is the direction nobody questions. ⚠️ And `--all-targets` does NOT include doctests.

**⚠️ `--lib` DOES NOT BUILD `tests/`, SO A `--lib` GATE IS STRUCTURALLY BLIND TO EVERY INTEGRATION TEST — AND "MAIN IS HEALTHY" REPORTED OFF ONE IS A CLAIM ABOUT A TARGET KIND WEARING THE LABEL OF THE WHOLE TREE (2026-08-13, coordinator, self-inflicted).** Measured on `main`: `-p fuel-dispatch --features telemetry --lib` reported **806 passed, green**, while `--all-targets` on the same commit reported **`fuel_emits_only_recognized_sk4_dtype_spellings ... FAILED`**. The merge that introduced it had been reviewed and the tree pronounced healthy across four feature configurations — **every one of them `--lib`.** **The features were named in the report; the TARGETS never were, which is the construct-invisible-in-the-number defect committed by the person who keeps catching it in others.** **PRACTICE: gate with `--all-targets`, and when reporting a tree healthy, state the target kinds as well as the features ⚠️⚠️ **— AND `--all-targets` DOES NOT INCLUDE DOCTESTS, WHICH IS A NAMED HOLE IN THE GATE THIS PROJECT TRUSTS MOST (measured 2026-08-14: `cargo test -p fuel-inference --all-targets` emits **ZERO** `Doc-tests` sections).** `--all-targets` expands to `--lib --bins --tests --benches --examples` — **`--doc` is not in that list and cannot be added to it.** So every `--all-targets` gate in this repo is **structurally incapable of seeing doctest breakage**, and doctests are compiled code: a ` ```no_run ` fence **still compiles**, it merely does not execute. **LIVE INSTANCE, AND IT IS THE THIRD TIME THE SAME COMPLETION CLAIM WAS WRONG: `fuel-inference`'s module docs carry ` ```no_run ` examples doing `use fuel::{Device, Tensor};` — the eager API DELETED IN B6. `cargo test -p fuel-inference --doc` exits 101 with 2 FAILED.** **B6 was declared complete three times, and each miss was a TARGET KIND NOBODY ENUMERATED: first two whole crates, then feature-gated examples, now doctests.** **THE RULE THAT GENERALISES IS NOT "remember doctests" — IT IS: A COMPLETION CLAIM MUST NAME THE TARGET KINDS IT SEARCHED, because every miss so far has been a KIND, never a missed instance within a kind.** The enumeration to run against any "X is fully retired" claim: **lib · bins · tests · benches · examples · DOCTESTS · feature-gated variants of each · and non-default-member crates.** `--all-targets --workspace --all-features` still misses the sixth. **⚠️ CORRECTED 2026-08-15 BY THE COVERAGE-MATRIX MEASUREMENT — THE CORRECTION MOVES THE BLAME FROM CI TO OUR OWN HABIT. CI ALREADY RUNS DOCTESTS: its test job is `cargo test --workspace --no-fail-fast $EXCL` with **NO `--all-targets`**, and plain `cargo test` runs doctests by default. So *"nothing runs them"* was FALSE at CI level, and a new `--doc` step is NOT the recurrence fix.** **THE BLIND SPOT IS THE LOCAL VERIFICATION HABIT: both B6 closes were verified with local `--all-targets`, the one instrument that never sees doctests — AND CI, which would have caught it, WAS DOWN for the entire window (dead at dependency resolution). TWO FAILURES COMPOUNDING; neither alone would have hidden it.** `cargo test --workspace --doc` stays useful for **isolation and speed**, not because doctests are otherwise uncovered. — an unqualified "green" is the direction that does not get questioned.** ⚠️⚠️ **AND A GATE MUST MATCH CI'S *TOOLCHAIN* AS WELL AS ITS FEATURES AND TARGETS — 2026-08-14, and the instructive part is that the FIRST FIX TO THE GATE WAS ITSELF INSUFFICIENT AND ONLY A SABOTAGE OF THE REPAIR FOUND IT.** `scripts/aarch64-cross-check.ps1` existed, ran, and passed, while `fuel-quantized` carried an `E0658 stdarch_neon_dotprod` that killed both macOS CI jobs. **Two independent causes, and fixing one left the gate still green on a live defect:** **(a) IT WAS POINTED AT THE OTHER BRANCH.** `.cargo/config.toml` sets `target-cpu=generic` for `aarch64-apple-darwin` — ARMv8.0-A, **no dotprod** — so the script compiled the SOFTWARE arm, **while macOS CI DELETES `.cargo/config.toml` (the ring workaround) and compiles the HARDWARE arm.** ⚠️ **And the config's own justification was a TRUE STATEMENT WITH TOO-WIDE SCOPE — *"NEON is baseline-mandatory in ARMv8, so `generic` still compiles the NEON paths this gate exists to reach"* is true of BASELINE NEON and false of DOTPROD-GATED NEON.** The justification-scope-mismatch pattern, living in a config comment. **(b) IT RAN ON THE DEFAULT TOOLCHAIN (nightly on this box) WHILE CI RUNS STABLE — and an unstable-feature error is INVISIBLE ON NIGHTLY.** **⚠️⚠️ THE TRANSFERABLE PART: the lane fixed (a), RE-RAN THE SABOTAGE, AND THE GATE STILL SAID PASS — which is the only reason (b) surfaced. SABOTAGE-VALIDATE YOUR REPAIR TO THE GATE, NOT JUST THE GATE.** A fix to an instrument is a claim about the instrument, and it earns the same red-then-green proof as anything else; **without it a half-fix ships looking complete, and the next reader trusts a gate that now covers one of its two blind spots.** **AND THE SEVERITY ORDERING WORTH KEEPING: A GATE THAT EXISTS AND PASSES IS WORSE THAN NO GATE, BECAUSE IT GETS CITED AS COVERAGE.** An absent check is a known hole; a check aimed at the wrong branch, on the wrong toolchain, is a hole wearing a green badge. **Ask of any local gate: does it compile the same BRANCH, on the same TOOLCHAIN, with the same CONFIG-FILE state, as the CI job it stands in for? `.cargo/config.toml` presence is part of that state — CI deleting it is a configuration difference no feature flag or target triple records.**

---

## sabotage-calibrated-tolerances

> **Index line (in CLAUDE.md):** **A numeric oracle's tolerance must be SABOTAGE-CALIBRATED, never inherited.** Report (a) correct-path drift and (b) sabotaged divergence; set the threshold BETWEEN them; record both in-file. Poor separation is a FINDING, not a threshold to tune. A relative oracle is blind to defects in shared code; a golden nobody has seen fail is a constant.

**⚠️ A NUMERIC ORACLE'S TOLERANCE MUST BE SABOTAGE-CALIBRATED, NEVER INHERITED — AND THE AMBIENT DECODE TEMPLATE IS MEASURED NOT TO SEE A RoPE cos/sin SWAP (2026-08-13).** The decode suite's natural template asserts **`diff < 5e-3 || rel < 1e-2`**, and it is **too loose to be an oracle for the defects these ports actually make**. Two independent measurements: (1) the windowed-vs-single-mask divergence a whole GAP-029 sub-scope was priced on is **7.9e-3 at prefill / 7.04e-3–7.95e-3 at decode** — *1.6x* the abs threshold, and the `||` means the `rel` arm alone passes it outright; (2) under a deliberate sabotage, `forward_with_kv_context_decode_matches_non_cached_forward` — an **absolute** oracle at that tolerance — **PASSED with a RoPE cos/sin swap in place**, which is not a marginal defect. **THE FAILURE ARRIVES THROUGH THE TOLERANCE RATHER THAN THE ASSERTION TARGET, so it survives every check aimed at *what* is asserted.** **PRACTICE: report BOTH numbers — (a) the correct implementation's drift against the oracle, (b) the sabotaged/old-fabrication variant's divergence — and set the threshold BETWEEN them, with both recorded in the test file. If (a) and (b) are not comfortably separated, THAT IS A FINDING: report it rather than picking a threshold that makes the test pass.** For scale, the golden that did catch the rope swap sits at **1e-6**, and a behaviour-preserving extraction of a shipped decode path held to it exactly. **⚠️ AND TWO ORACLE SHAPES THAT NO TOLERANCE FIXES, because they are category errors: a RELATIVE oracle (D2 vs D1, or any A-vs-B over shared code) is STRUCTURALLY BLIND to a defect in the code both sides run — it passed under a sabotage corrupting every logit it read; and a GOLDEN NOBODY HAS SEEN FAIL IS A CONSTANT, NOT AN ORACLE, so a golden owes a red before it is cited as evidence.** ⚠️⚠️ **AND THE STRONGEST INSTANCE ARRIVED 2026-08-14 ON GEMMA3, WHERE THE DEFECT SITS *INSIDE* THE TEMPLATE RATHER THAN 1.6x OVER IT: the RoPE axis (per-layer base collapsed to a single global base) diverges at `[1.235e-3, 8.673e-4, 1.031e-3]` — about 100x the 1e-5 oracle used, and COMFORTABLY UNDER BOTH ARMS of the natural template. A DUAL-BASE DEFECT PASSES IT OUTRIGHT, with no arithmetic to notice.** The mask axis on the same model is `[5.876e-2, …]`, i.e. **40x larger**, so a single "windowing works" number would have been read as covering both. **AND THE AXIS HAD TWO INDEPENDENT ROUTES TO A SILENT GREEN — a DEGENERATE FIXTURE collapsing the input (`rope_local_base_freq == rope_theta`) and the TOLERANCE swallowing the output — so closing only one produces a test that LOOKS rigorous and proves nothing. Both must be closed, and the fixture guard must assert the DERIVED quantity (`decode_rope_plan().n_variants() == 2`), not that the config fields differ, WITH a sibling test pinning that the derived quantity tracks the fixture (equal bases → 1) — otherwise the guard is a tautology waiting for a later edit.** ⚠️⚠️ **SEPARATE AND MORE PORTABLE, FROM THE SAME LANDING — A TRUE PIECE OF EVIDENCE THAT WAS BEING CARRIED AS A PORTABLE PROOF SHAPE AND IS ACTUALLY CONFIG-DEPENDENT.** Qwen2's born-red `[0.0, 7.04e-3, 7.95e-3]` was repeatedly cited as proof of discrimination — *"position 3 is clean under BOTH bodies, so a degenerate oracle would have shown three zeros and this showed one."* **On Gemma3 all three positions diverge, and that is EQUALLY CORRECT: Qwen2's window of 4 cannot exclude anything until absolute position 4; Gemma3's window of 3 bites at position 3. THE LEADING ZERO IS A PROPERTY OF THE CONFIG (window width vs prefill length), NOT OF THE SEAM, THE PORT, OR WINDOWING.** **A CONFIG-DEPENDENT PIECE OF EVIDENCE THAT LOOKS PORTABLE IS MORE DANGEROUS THAN A WRONG ONE: the next family showing three zeros gets investigated as a degenerate oracle when it is fine, and one showing a leading zero for an unrelated reason gets read as confirmation.** **PRACTICE: before reusing a proof SHAPE across cases, derive it from the config that produced it — and if it depends on the config, say so at the site rather than in the report, because the report is not what the next family's author reads.** Caught by the lane that had been making the inference, mid-program, unprompted — the third self-caught false generalisation in one day from three different people (the others: a grep called a "lower bound" that was exact, and a positive control asserted into a spec that was never measured).

---

## vacuous-oracle-four-routes

> **Index line (in CLAUDE.md):** **A vacuous oracle arrives by FOUR routes: the tolerance, the assertion target, the FIXTURE (input collapses the axis), and the SHORT-CIRCUIT (an earlier guard answers first).** When a test asserts something is REJECTED, ask which guard did the rejecting.

**⚠️ A VACUOUS ORACLE ARRIVES BY THREE DISTINCT ROUTES, AND THE THIRD IS THE FIXTURE — WHERE THE TOLERANCE IS FINE, THE ASSERTION IS FINE, AND THE *INPUT* COLLAPSES THE AXIS UNDER TEST (2026-08-13).** Measured instances, all in one program: **(1) the TOLERANCE** — `5e-3 || rel 1e-2` sitting above a `7.9e-3` real defect, and passing a RoPE cos/sin swap under sabotage; **(2) the ASSERTION TARGET** — sampled-token equality, vacuous for a KV-dependent tiny model, where only logits discriminate; **(3) the FIXTURE** — `lazy_gemma3.rs:540` sets `rope_local_base_freq: 10_000.0` and `rope_theta: 10_000.0`, *deliberately*, with the comment *"same as global for the tables match test"*. **The two RoPE bases are EQUAL in the only Gemma3 configs the repo has, so a decode test written on that fixture PASSES UNDER A SINGLE-TABLE PORT — the dual-base defect is invisible because the input has no dual base.** **THE FIRST TWO ARE ABOUT HOW YOU MEASURE AND CAN BE CAUGHT BY CARE IN THE ASSERTION; THE THIRD IS ABOUT WHAT YOU MEASURE IT ON, AND NO CARE IN THE ASSERTION REACHES IT.** **PRACTICE: assert NON-VACUITY OF THE INPUT inside the test itself** — `assert!(rope_local_base_freq != rope_theta)`, the way the mask families assert `seq > window` and `window_wider_than_capacity_is_byte_identical_to_the_dense_mask` guards its own trap. **A born-red on a degenerate fixture is a born-green wearing the right label** ⚠️⚠️ **AND A *FOURTH* ROUTE, FOUND 2026-08-14 BY THE GAP-194 LANE WHILE DESIGNING ITS OWN CUDA TEST — THE SHORT-CIRCUIT, WHERE THE TOLERANCE, THE ASSERTION AND THE FIXTURE ARE ALL FINE AND THE PREDICATE UNDER TEST IS NEVER REACHED.** The flash-decode arm's CPU test asserts *"a windowed layer is DECLINED"* and gets a decline — **from the capability and dtype checks, which run BEFORE the window is ever consulted.** So the green is **equally consistent with the window being ignored entirely**, and the decline half of that test was **vacuous while passing**, which is the state nobody investigates. **On CUDA with a bf16 cache the earlier guards pass, and a decline becomes attributable TO THE WINDOW — the first configuration in which that assertion means what its name says.** **THE GENERAL SHAPE: WHEN A TEST ASSERTS THAT SOMETHING IS REJECTED, ASK WHICH GUARD DID THE REJECTING.** A rejection is the cheapest possible outcome to produce accidentally — **every unrelated failure produces one** — so a passing negative-case test is far weaker evidence than a passing positive-case one unless the *reason* is pinned. **Distinct from the wrong-site failure (where the trigger cannot physically reach the code): here the code IS reached, the assertion IS about the right property, and an EARLIER GUARD ANSWERS FIRST.** Practical form: **construct the case so every guard upstream of the one under test PASSES**, and if that is impossible in the cheap configuration, say so in the file rather than letting the cheap green stand in for the expensive one. **Companion decision from the same lane, and it is the right default: `expect` RATHER THAN SKIP on a missing device — a silent skip is INDISTINGUISHABLE FROM A GREEN ADMIT HALF, which is the single outcome such a test exists to rule out.** The pre-existing tests in that file take the skip route; the lane deliberately did not copy them., and a shared fixture tuned to make some *other* test convenient is exactly where this hides — the comment that collapsed the axis was itself explaining a different test's needs.

---

## inherited-config-provenance

> **Index line (in CLAUDE.md):** **Ask of inherited config: has anything ever COMPILED this?** Fork-inherited lines look like decisions this project made. Provenance tells you who a line was written for; compile coverage tells you whether anyone was ever positioned to notice it was wrong. A "routine dep bump" is high-risk to feature-gated code specifically.

**Ask of any inherited config: "was this written for a repo that isn't this one?"** Two defects on 2026-08-13, same shape, found hours apart: `.gitignore:10` ignoring `Cargo.lock` (correct for Candle-the-library, costly for Fuel with a `[[bin]]`, CI, and a 30–56 minute forge — and four checkouts had silently drifted to two resolutions ~1904 lines apart), and `ci_cuda.yaml` requesting HuggingFace's runner group (correct upstream, unobtainable here). **Fork-inherited configuration is the hardest kind to see, because nothing about it looks wrong — it looks like a decision this project made, and `git log` on the line says "initial commit", which reads as settled rather than unexamined.** Both files even *documented* their own condition — the `.gitignore` comment literally said to remove the entry for an executable. **Check the provenance of a config line before defending it; "it has always been that way" and "nobody chose it" are the same observation.** ⚠️ **AND THE TRIGGER FOR THIS RULE CANNOT BE "WHEN SOMETHING LOOKS WRONG", BECAUSE NEITHER DEFECT DID — each was invisible in the direction that gets ignored.** The dead workflow presented as *a red X in a repo where red is normal*; the `Cargo.lock` ignore presented as *nothing at all*. **Neither ever looked like a question.** So the trigger has to be positional — **when you touch inherited infrastructure, ask who it was written for** — because no amount of attentiveness fires on a thing that is not displaying anything. ⚠️⚠️ **AND A THIRD INSTANCE ON 2026-08-14 SHOWS THE QUESTION ABOVE IS NOT SUFFICIENT, BECAUSE THIS ONE WAS WRONG *UPSTREAM TOO* AND WE INHERITED IT ANYWAY.** `fuel-examples`' `audio` module is written against **rubato 0.15** while the manifest requires **`"1"`**, so five example targets do not compile (`E0433 rubato::FftFixedInOut`). `git log -L` on the requirement line returns **exactly two commits in all of history, both upstream Candle**: `d365ef32` (#1865) added the resampler against `0.15.0`, and `8d5873bf` — a routine ***"Update deps"*** — bumped the requirement **`0.15.0` → `"1"` and did not touch the code.** **So the config was not "correct for a repo that isn't this one"; it was BROKEN WHERE IT WAS WRITTEN, shipped, and crossed the fork boundary intact.** ***"Was this written for a different repo?"* would have cleared it.** **THE MECHANISM IS THE SAME ONE THAT HID IT AFTERWARD, IN BOTH REPOS: `rubato` is `optional` and every target using it sits behind `required-features`, so no `cargo check`, no `--all-targets`, and no CI job in EITHER project ever compiled it. A dependency bump silently broke code upstream, and the identical structural hole downstream kept it silent here** — found only by compiling all 15 feature-gated targets one at a time (GAP-199). **THE BETTER QUESTION, WHICH SUBSUMES THE ORIGINAL: *HAS ANYTHING EVER COMPILED THIS?*** Provenance tells you who a line was written for; **compile coverage tells you whether anyone has ever been in a position to notice it was wrong** — and for `optional` deps and `required-features` targets the answer is routinely *no*, in every repo in the chain. **Corollary: a "routine dep bump" commit is a HIGH-RISK change to feature-gated code specifically, because the bump is validated by exactly the build that cannot see it.**

---

## validating-a-gate-means-reading-it

> **Index line (in CLAUDE.md):** **Validating a gate means reading its MESSAGE, not just its exit code** — a gate can return the right verdict with the wrong diagnosis. And a check that only PRINTS is not a gate: put it in the `&&` chain, never pipe it, never route its status through `echo`.

**VALIDATING A GATE MEANS READING ITS MESSAGE, NOT JUST ITS EXIT CODE (2026-08-08).** The sabotage discipline as carried asks only *did it go RED* — and a gate can return **the right verdict with the wrong diagnosis**, which a pass/fail-only check can never surface. Observed: a coverage gate checked for build ARTIFACTS first, and since **a FAILED compile and a NEVER-ATTEMPTED compile both produce no artifact**, it reported a genuine `E0004` as *"the compiler did not reach these crates"* — **true about artifacts, and a diagnosis pointing the reader at the harness instead of at their own code.** Companion failure from the same gate: it grepped for `Checking <crate>` lines, which **cargo does not reprint when every unit is fresh**, so it went **FALSE RED on a warm cache** — a gate nobody trusts, via a mechanism unrelated to the code under test. **Both directions must be validated (sabotage -> red, revert -> green) AND the message read in each.** **⚠️ AND THE EXACT COMPLEMENT, LEARNED SELF-INFLICTED (2026-08-12): A CHECK THAT PRODUCES A MESSAGE BUT NO EXIT CODE IS NOT A GATE AT ALL.** A registry table-integrity check printed `rows NOT in {5,6}: [('GAP-168', 11)]` **immediately before a commit** — and because it only *printed*, the `&&` chain sailed past it and pushed the corrupted row. **The instrument was correct, it fired, and it was ignored because it did not block.** So the pair is: *a gate's message must be read* (that rule) **and** *a check must `exit` nonzero or its message is decoration* (this one). **Generalises past markdown: an informational check inside an automated chain is INDISTINGUISHABLE FROM NO CHECK at the moment it matters, because its output scrolls past in exactly the place a human would have stopped.** Same family as the correctly-named validator that is never invoked — existence is not enforcement. **⚠️⚠️ AND THE SECOND HALF, LEARNED BY DEFEATING THIS EXACT FIX FOUR HOURS AFTER WRITING IT (2026-08-13, coordinator, self-inflicted): A GATE THAT *DOES* EXIT NONZERO IS NEUTRALISED JUST AS COMPLETELY BY CAPTURING ITS EXIT CODE FOR DISPLAY.** The shape was `gate.py | tail -8; echo "GATE: ${PIPESTATUS[0]}" && git commit && git push` — **the gate exited 1, printed `GATE: 1`, and the `&&` chain proceeded from the `echo`'s success, pushing a corrupted table row.** Two compounding errors, each individually survivable: **`| tail -8` truncated away the one line naming the violation**, and **`echo`ing the status turned a gate back into a print** — the precise defect the rule above exists to prevent, reintroduced through a different door. **A pipe also destroys the exit code by default** (hence `PIPESTATUS`), which is what invites the echo in the first place. **PRACTICE: put the gate in the `&&` chain ITSELF (`python gate.py && git commit && git push`) so a nonzero exit stops the chain; never pipe it; never route its status through `echo`. If you want to see the status, run it AGAIN afterwards — reporting and enforcing are different jobs and must not share a command.** The corruption here was caught one command later only because the *next* thing run happened to be the same gate, unpiped.

---

## sha-is-not-a-stable-name

> **Index line (in CLAUDE.md):** **A sha is not a stable name for a change in a rebase workflow.** Anchor a cross-lane state check on CONTENT or the commit subject. `git log HEAD --not --remotes` answers "is this OBJECT on a remote", not "is this WORK landed" — branch reaping makes those diverge.

**⚠️ A SHA IS NOT A STABLE NAME FOR A CHANGE IN A REBASE WORKFLOW — FOR A STATE CHECK ACROSS A REBASING LANE, ANCHOR ON CONTENT OR ON THE COMMIT MESSAGE, NEVER ON A SHA HANDED TO YOU EARLIER (2026-08-13).** The coordinator reported a lane's work "not landed" from *"everything above `4f05f848` is mine"* — **`4f05f848` was the lane's PRE-REBASE sha, and `git rebase origin/main` rewrote it.** `merge-base --is-ancestor` correctly answered **not on main**, because that object genuinely is not in main's history, **while its CONTENT was, under a new sha.** **The query was well-formed, the answer was TRUE, and the conclusion was WRONG.** **This is a DIFFERENT mechanism from the other absence-claim failures in this file: those were the wrong instrument (one visibility mechanism of two; a file's gate standing in for a guarantee's coverage). THIS was the RIGHT instrument on an object that had been REPLACED — so it also bounds the positive-control rule further: a positive control proves the query can find its target; it cannot tell you the target is still the right NAME for the thing.** **Practice: `git show origin/main:<path> | grep <the actual construct>`, or match on the commit subject. This is specifically a COORDINATOR hazard — cross-lane state checks are SHA-anchored by habit — and a lane's peer SUMMARY rots the same way (it can name a branch that no longer exists on origin, and then answer a state check with it). **⚠️⚠️ AND THE SAME TRAP RUNS THE OTHER DIRECTION, INTO THE HANDOFF GUARD ITSELF — WHERE IT PRODUCES A FALSE REPORT OF UNPUSHED WORK, AND THE COORDINATOR CAUSES IT (2026-08-13, reported by the lane on its way out).** The standing handoff check is `git log HEAD --not --remotes` = 0. A lane returned **1** and correctly refused to read it as local-only work: **its local commit was the PRE-MERGE sha; the coordinator had rebased it onto main and then REAPED THE BRANCH, so the object lost its remote counterpart while its CONTENT shipped.** Verified: the sha is **not** an ancestor of `origin/main`, and the construct it added appears **5 times** in `git show origin/main:<path>`, under a new sha with the same subject. **`--not --remotes` answers *"is this OBJECT on a remote"*, NOT *"is this WORK landed"* — and branch reaping makes those diverge.** **A lane following the guard literally, after a reap, reports unpushed work that shipped an hour earlier — and the natural response to that report is to re-push it.** **PRACTICE: verify a nonzero count by CONTENT or SUBJECT before believing it; and as coordinator, either do not reap a branch until its lane has finished its handoff, or tell the lane you reaped it — the false positive is created by the cleanup, not by the lane.**

---

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

## one-feature-is-not-two

> **Index line (in CLAUDE.md):** **One feature is not two — a gate covering each feature separately covers no INTERSECTION.** When a change must be exhaustive, enumerate the feature COMBINATIONS that parse each site, not the features.

**ONE FEATURE IS NOT TWO — A GATE THAT COVERS EACH FEATURE SEPARATELY COVERS NO INTERSECTION (2026-08-12).** `fuel-dispatch/src/telemetry/baracuda_provider.rs` is reachable only under a FEATURE PAIR, never a single feature: `mod telemetry` is cfg'd on `telemetry`, and `mod baracuda_provider` is cfg'd *inside it*. **⚠️ CORRECTED 2026-08-13 — THIS RULE HAD ROTTED ON THE EXACT FEATURE COMBINATION IT NAMES, AND THE ROT MADE THE PRESCRIBED GATE FAR MORE EXPENSIVE THAN NECESSARY.** It used to read "`mod baracuda_provider` is cfg'd on `cuda`", which was true when written and was **fixed by `f1d475d2` ("gate the provider on what it needs, not on what implies it")**: the module is now cfg'd on **`baracuda-types`**, and only the `pub use BaracudaStructureKeyProvider` re-export still needs `cuda`. **So the minimal gate for that match is `--features telemetry,baracuda-types` — no CUDA SDK, no `cuda-build.ps1` slot, seconds on any machine — whereas this text sent readers to a GPU-class build for the same answer. A stale rule about feature gating cost more than a wrong fact: it made the cheap gate look impossible.** The incident below is historical and its lesson stands unchanged. So a wildcard-free `match` there missing `DType::F8E5M2` sat on `main` as a **latent E0004 that no gate we run could see** — including GAP-097's `--features telemetry` gate, which is *strictly stronger* than a default build and **still does not parse the file.** Confirmed by the compiler through `scripts/cuda-build.ps1` with both required artifacts (`Environment initialized for: 'x64'` + `Checking fuel-dispatch`), exit 101 → fixed → exit 0. **So a dtype was added across 20 sites in 7 crates and missed a 21st that is structurally unreachable by every gate in the repo.** Practice: when a change must be exhaustive, enumerate the FEATURE COMBINATIONS that parse each site, not the features. **Population caveat, and it is the honest residual: a scan for `#[cfg(feature=…)]` immediately preceding a `mod` finds exactly one such nesting workspace-wide — but it is a LOWER BOUND by construction, because it sees gated MODULES and not `#[cfg(feature=…)]` on individual items or impls inside ungated modules, which are equally invisible and are where most exhaustive matches actually live.** See docs/gaps.md GAP-097, GAP-171.

---

## a-workflow-that-never-starts

> **Index line (in CLAUDE.md):** **A workflow that never starts is indistinguishable from one that runs and fails** — the discriminator is `steps: 0`, never the conclusion field. ⚠️ And `steps > 0` does not mean the gate reached YOUR code: read the failing step's log.

**A workflow that never starts is indistinguishable from one that runs and fails — the discriminator is `steps: 0` / `runner_group_name: None` from the jobs API, never the conclusion field.** Both render as the same red X, and a repo with known-red CI trains everyone to read one more X as more of the same. This is the CI-shaped instance of the standing rule that **an exit code is evidence about the harness until an artifact proves the work ran**: at job level the artifact is *executed steps*, and a red conclusion is equally consistent with "the gate caught something" and "nothing ever ran". Before citing any CI job as coverage, confirm it has ever executed a step. ⚠️⚠️ **AND THE `steps: 0` SIGNATURE RECURRED SELF-INFLICTED HOURS AFTER THIS RULE WAS WRITTEN, VIA A VALIDATOR THAT PASSED BECAUSE ITS *LIBRARY* WAS PERMISSIVE (2026-08-14).** A workflow edit orphaned a job's `with:` block, **GitHub ran ZERO JOBS**, and the pre-flight check said *"YAML OK"* — because **`yaml.safe_load` TOLERATES DUPLICATE KEYS**, silently letting the last one win. **The parser was answering *"can I load this?"* while the author was asking *"is this correct?"*, and those two questions diverge exactly where the defect lives.** **GENERALISE PAST YAML: A PARSE-SUCCEEDS CHECK IS NOT A VALIDITY CHECK, AND EVERY FORGIVING PARSER IS A VALIDATOR THAT AGREES WITH YOU.** JSON parsers that accept trailing garbage, TOML readers that ignore unknown keys, `serde(deny_unknown_fields)` left off, a Markdown table "checked" by rendering rather than by counting delimiters — same shape. **ASK WHAT THE PARSER IS PERMITTED TO IGNORE, and use the STRICTEST available mode (a duplicate-key-rejecting loader, `deny_unknown_fields`, a schema) — then SABOTAGE IT, because a strict mode you have never seen reject anything is indistinguishable from the permissive one.** **AND NOTE WHICH RULE SAVED IT: the defect was found by the `steps: 0` discriminator recorded that same morning. The check that failed and the check that caught it were both about CI, hours apart — which is the argument for writing these down rather than remembering them.**

---

## delimiter-traps-have-two-ends

> **Index line (in CLAUDE.md):** **A delimiter trap has two ends, and learning one does not protect you from the other.** Match at word boundaries at BOTH ends. "I've already learned this one" is not a defence — the last field before a closing brace is the end that keeps being forgotten.

**A DELIMITER TRAP HAS TWO ENDS, AND LEARNING ONE DOES NOT PROTECT YOU FROM THE OTHER (2026-08-08).** One worker hit both in eight hours. **Morning, matched at the PREFIX:** a spelling guard keyed on `e4m3fn` also fires inside `f8e4m3fn`. **Afternoon, matched at the SUFFIX:** a string-replace of `l.stride(),` also fires inside `t_l.stride(),` / `src_l.stride(),` — producing `t_&l.stride_unsigned()` and 18 fresh errors. **The rule was written down between the two, by the person who then broke it in the mirror direction.** Their conclusion is the durable one: ***"I've already learned this one" is not a defence.*** **State it as delimiters at BOTH ends** (`\b`-anchored regex / word-boundary match), or you re-derive the missing half by breaking something. **⚠️ AND A THIRD END THE `\b` FIX DOES NOT REACH — THE *TYPE* BOUNDARY (2026-08-13).** A lane scripted a six-family field rename with `^\s*rope_base: (.+),$` — perfectly anchored, both ends, no substring hazard — **and it rewrote 33 `LlamaConfig` literals in `lazy.rs`, because those configs have a field with the SAME NAME.** 34 changes, 1 legitimate. Caught by the compiler (`LlamaConfig has no field embed_scale`) and repaired from `git diff` **by count, not by eye**. **A REGEX KEYED ON A FIELD NAME HITS EVERY STRUCT THAT HAPPENS TO HAVE THAT FIELD — AND "RENAME A FIELD IN ONE STRUCT" IS PRECISELY THE CHANGE WHERE THAT IS MOST LIKELY, because related structs share vocabulary.** The pattern cannot see types; anchoring harder does not help, because the text really is identical. **PRACTICE: do struct-field edits per-file with anchored inserts, or drive them from the compiler; and when a scripted sweep touches more sites than the population you enumerated, RECONCILE BY COUNT before trusting the diff.** Same family as the prefix/suffix traps above — the boundary the pattern cannot express is just a type instead of a word. Caught immediately only because the **error count went UP** — a fast, unambiguous signal is what makes a self-inflicted sweep error cheap instead of expensive.

---

## target-crate-compile-line

> **Index line (in CLAUDE.md):** **For a scoped check the required artifact is the TARGET CRATE's compile line, not any compile line** — a build that dies in deps emits every other positive artifact. ⚠️ And a crate-level line proves nothing about a `cfg`'d MODULE, nor about a crate's TESTS (that needs the `(lib test)` unit).

**FOR A SCOPED CHECK, THE REQUIRED ARTIFACT IS THE *TARGET CRATE'S* COMPILE LINE — NOT *ANY* COMPILE LINE (2026-08-08). This AMENDS the artifact rule below, which discriminates less than it appears to.** A `-p fuel-cuda-backend` check was terminated mid-flight while still building deps. Its log contained **`Environment initialized for: 'x64'`, 32 `Checking` lines, zero errors, and zero `E0004`** — **every positive artifact the rule below requires** — and it proved nothing about `fuel-cuda-backend`, which was **never reached**. **"The compiler ran" and "the compiler reached the code under test" diverge exactly when a build dies in deps, which on this box is most of the wall-clock.** ⚠️⚠️ **AND THE `Checking <crate>` LINE IS ITSELF NOT ENOUGH WHEN THE QUESTION IS ABOUT A CRATE'S *TESTS* — MEASURED 2026-08-14 BY THE GAP-157 LANE, AND IT SHARPENS THIS RULE'S OWN PRESCRIPTION.** `cargo check -p fuel-core --features cuda` builds `fuel-cuda-backend` **as a LIB DEPENDENCY** and never compiles its test targets — **yet it emits a perfectly good `Checking fuel-cuda-backend v0.10.3` line**, which this rule as written accepts as proof the target crate was reached. It WAS reached; **its `#[cfg(test)]` code was not**, so that leg structurally could not verify three `probe.rs` conversions while looking like full coverage. **THE DISCRIMINATING ARTIFACT IS THE `(lib test)` UNIT: `warning: fuel-cuda-backend (lib test) generated 42 warnings` — the `(lib test)` suffix is what proves the test targets compiled.** Quote that, not the bare `Checking` line, whenever the claim involves a crate's tests. Same family as `--lib` not building `tests/`: **the target KIND is a second dimension the feature list never mentions, and a crate-level compile line collapses it.** Grep for the target crate's own `Checking <crate> v<version>` line, and **name it in the report**. **This is a FIFTH build state, and the most deceptive:** *unrun* (chose not to), *unrunnable* (machine can't), *invisible* (non-default feature — see below), *invisible-and-unrunnable* (mkl/accelerate/aocl/onemkl), and now **ran, looked green, terminated before reaching the target.**

---

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

## checked-then-didnt-look

> **Index line (in CLAUDE.md):** **Performing a verification and then reporting something other than its output is a distinct defect from failing to verify — and remembering to check does not fix it.** Report from the artifact, not from the fact that a step completed.

**TWO INSTANCES IN ONE NIGHT, TWO PEOPLE, DIFFERENT TOOLS (2026-08-20).**

- The architect ran `git log -1` on five shas *specifically to verify them*, and their terminal printed `docs(outreach): baracuda-seam SEAM_MAGIC lockstep ask`. They then wrote **"Spec-B candidate-kernel ingestion"** — the label from the document under review — into a report that reached CireSnave.
- The doc-currency lane resolved a rebase conflict, confirmed the rebase **completed**, and reported *"resolved in favour of yours."* The push had in fact overwritten the other party's file. **They never read the file; they read that the step succeeded.**

**Why this is not "forgot to check":** the check ran. Its output existed. What failed was the step between the output and the claim — a prior belief supplied the answer, and the measurement was treated as a formality that had been satisfied rather than as a source of information. **A rule that says "verify before claiming" is already satisfied by both of these.**

**Practice:** state the claim in the form of the artifact — quote the subject line, `cat` the file, paste the count — so the report *is* the output rather than a summary of it. If you cannot quote it, you did not read it. And be most suspicious when the measurement is expected to confirm something: **a number that agrees with the prose gets shipped; a number that disagrees gets re-measured** (see `gap-029-persistent-decode-trait.md`, where the naive grep returns exactly the stale figure the document claims).

## git-rebase-inverts-ours-and-theirs

> **Index line (in CLAUDE.md):** **In a REBASE, `--theirs` is the commit being APPLIED (yours) and `--ours` is upstream — the exact inverse of a merge.** Taking "theirs" to mean "the other side's version" silently keeps your own and discards theirs.

**A footgun with no warning and a plausible-sounding name in both directions (2026-08-20).** A lane resolving a conflict against a peer's committed repair ran `git checkout --theirs <file>` meaning *"take main's version"*. During a rebase, upstream is checked out first and each commit is replayed **onto** it, so **`--ours` is upstream and `--theirs` is the commit under replay.** The resolve silently kept the lane's own version, the rebase reported success, and the subsequent push overwrote the peer's work — while the lane reported the opposite in good faith.

**Practice:** do not use `--ours`/`--theirs` during a rebase at all. Name the source explicitly — `git checkout origin/main -- <path>` — which is unambiguous under both operations. Then **read the file** before reporting what it contains (see `checked-then-didnt-look`; these two combined to produce the incident).


---

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
  [`reverify-differential-after-rebase-before-push`](#reverify-differential-after-rebase-before-push)
  with the mutation coming from your own queue rather than a peer's.

**The tell that you are in this situation:** the deferred change is described as
*"free"* or *"orphans nothing"* **on the strength of the current broken state.**
That freeness is a property of the bug, not of the change — and the other fix is
about to remove the bug.

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
[`verify-the-population-not-the-instance`](#verify-the-population-not-the-instance),
[`fix-the-generator-not-the-output`](#fix-the-generator-not-the-output).

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
[`magnitude-is-not-impossibility`](#magnitude-is-not-impossibility).

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
[`go-to-the-artefact-not-the-rendering`](#go-to-the-artefact-not-the-rendering).

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
[`verify-the-population-not-the-instance`](#verify-the-population-not-the-instance),
[`enumerate-the-population-not-the-strings`](#enumerate-the-population-not-the-strings).

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

## an-insertion-before-an-item-steals-the-attribute-above-it

**Inserting a test immediately before `fn target` splices your block BETWEEN that function's `#[test]` and the function it modifies. The attribute binds to YOUR item, so yours registers twice and the pre-existing test silently STOPS RUNNING.** Measured 2026-09-02 in `fuel-cpu-backend/src/byte_kernels.rs`: a boundary test anchored on `    fn cast_f32_f8e4m3_round_trip_exact_for_representable() {` — a unique, exact, verified anchor — landed under that function's `#[test]`, and `cargo test` reported **`2 passed`** for a filter matching **one** name.

**Both directions are silent and both look like good news.** The suite stays green; the total stays plausible (my duplicate exactly replaced the test I had disabled, so the count did not move); and the disabled test is not *failing*, it is *absent*, which no result line mentions. **CLAUDE.md already warns never to splice a seed between an attribute and its item — but it says so about `#[cfg]` in born-red sabotage, and this arrived through ordinary test authoring.** The stolen attribute can be `#[test]`, `#[cfg]`, `#[ignore]`, `#[should_panic]` or a derive; the rule is about INSERTION POINTS, not about sabotage.

⚠️ **THE COMPILER REPORTED IT AND MY INSTRUMENT ATE THE MESSAGE.** `rustc` emitted `function cast_f32_f8e4m3_round_trip_exact_for_representable is never used` — the exact finding, unprompted. I was running `cargo test -- --list 2>&1 | grep -c '<name>'`, so **stderr was merged into the stream I was counting and the warning lines were tallied as if they were list entries**, returning `2` for a function that was registered `0` times. **A count over a stream carrying diagnostics is not a count of the thing you asked about** — see [`validating-a-gate-means-reading-it`](#validating-a-gate-means-reading-it) and [`grep-o-discards-the-context-that-dispositions-the-match`](#grep-o-discards-the-context-that-dispositions-the-match). Reading the two matching LINES instead of counting them ended the investigation immediately.

✅ **THE MECHANICAL DETECTOR IS ONE LINE AND NEEDS NO SUSPICION: `--list` LISTED MUST EQUAL UNIQUE.** A duplicate fully-qualified path cannot come from one `fn`, so `cargo test -p <crate> --lib -- --list 2>/dev/null | grep ': test$'` piped to `sort -u | wc -l` must match the raw count — **and drop stderr for this one, precisely because the diagnostics you want to READ are the ones that corrupt a COUNT.** Pair it with naming the neighbour: after inserting near an existing test, assert that test still appears in the listing. **Anchor on the start of the item's attributes or doc comment, never on its `fn` line.**

⚠️ **AND THE SECOND ANCHOR FAILURE FROM THE SAME HOUR, BECAUSE IT INVERTED A SABOTAGE RESULT: `cargo fmt` RUNS BETWEEN AUTHORING A FIXTURE AND SABOTAGING IT, AND IT INVALIDATES THE TEXT ANCHOR YOU JUST WROTE.** To prove a vacuity guard fired I trimmed the fixture table by splitting on `'(449.0'` — a token I had typed myself minutes earlier. `rustfmt` had since exploded the longer tuples across lines, so the token was gone; **Python's `str.split` on an absent separator returns the whole string unchanged, so the "trim" was a no-op, nothing recompiled, and the test passed.** I very nearly recorded that pass as *the guard does not fire*. **A sabotage that never applied reports absence-of-sensitivity as presence** ([`a-sabotage-that-never-applied`](#a-sabotage-that-never-applied)); the new specifics are that **a formatter is a mutation between your write and your read**, and that **the split/replace family fails SILENTLY and IDENTITY-PRESERVINGLY.** **Assert the anchor is present and that the edit SHRANK the text, before running anything — and require the `Compiling` line, which is what would have caught it here.**

---

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

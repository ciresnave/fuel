# The seam crates' version line (proposal, no code)

**Status:** proposal for the PM and CireSnave. Measured at `origin/main@66948f82` (0.24.4) and on crates.io on 2026-10-24.
**Asked by:** the PM, 2026-10-24: "the fuel<->baracuda-cuda-seam version coupling (every fuel breaking bump forces a seam republish): propose a stable seam-types crate version policy."

## 1. The problem, with the numbers

`fuel-kernel-seam`, `fuel-kernel-seam-types` and `fuel-kernel-seam-announce` are the interface between Fuel and a kernel synthesizer (`baracuda-cuda-seam`). They carry the workspace version, because the portfolio rule is one version number per project. `baracuda-cuda-seam` depends on them by caret, and a caret on a `0.y` version pins `y`. So **every Fuel release that moves the second number is a breaking release of the seam crates as far as Cargo can tell, whether or not a single seam type changed.**

- `baracuda-cuda-seam 0.0.1-alpha.161` required `fuel-kernel-seam{,-types} ^0.23.0`; `0.1.0-alpha.2` requires `^0.24.0` (read from crates.io). Fuel's own 0.24.0 broke `jit-synth` for the days between the two, and the next breaking Fuel release (0.25.0, MLMF slice D) will do it again until baracuda republishes on `^0.25`.
- The seam crates were published 13 times between 2026-07-04 and 2026-10-24 (`0.10.3`, `0.11.2`, `0.19.1`, `0.20.0` ... `0.23.0`, `0.24.0`). Since 2026-10-14 alone that is five different second numbers (0.19, 0.20, 0.22, 0.23, 0.24) in ten days, four forced consumer-visible steps.
- **What actually changed in them is small.** Since 2026-10-01 three commits touched the source of the three crates: `bc9b49ff9` (`JitRequest` gains `target: TargetId`, the one change that is a real seam break), `e575f7a02` (a bytemuck refactor that changed no seam type) and `5aaad145f` (today: the dependency moved from `baracuda-kernels-types` to `unpopped-vocab`). One real break in the window, against four second-number steps the consumer had to follow.

## 2. What the rule says, and what it was for

CireSnave's rule (`C:\Projects\CLAUDE.md` section 9): *"I always want all crates within a project to use the same version number so developers consuming them know which go with which with one exception where a crate in one project should keep a version that matches another project such as an emitter crate in Baracuda designed to work with Unpopped keeping its version matching the version of Unpopped it works with."* The exception applies "only when your number answers 'which version of the *other* project does this work with?'"

The seam crates sit exactly on that line. Their number is meant to answer "which seam protocol does this speak?", which Baracuda needs to know, and not "which Fuel release is this?", which it does not.

## 3. Options

| | Option | Effect on `baracuda-cuda-seam` | Cost |
|---|---|---|---|
| A | **Status quo**: seam crates track the Fuel workspace number | republish on every Fuel second-number bump (4 in 10 days) | none now; recurring coordination, and a window of breakage each time |
| B | **Own version line for the three seam crates** (an exception under the clause above): the number moves only when a seam type or the handshake changes; a Fuel release that does not touch them leaves them alone | republish only on a real seam break (1 since 2026-10-01) | breaks "one number per project" for these 3 crates; the seam crates stop being published in every Fuel release; the workspace `[patch]` still redirects registry to path, so it must apply to the seam line's version |
| C | **Widen baracuda's requirement** (`>=0.24, <0.30`) | none, until a real break | Cargo cannot know which Fuel bump was breaking for the seam, so this is a promise made by hand; the one real break in the sample (`TargetId`) would have gone through silently |
| D | **Take the seam crates to 1.0** | republish only on a major | needs the types stable first: `PatternNode` and `OpAttrs` are `EXHAUSTIVE-BY-DESIGN` (GAP-049/076) and the grammar is still moving (Op::Scan, row 40's Gated DeltaNet) |

## 4. Recommendation

**Option B**, the version line of the seam crates, starting from the current number so nothing already published is orphaned:

1. The three crates get their own `version` (not `version.workspace = true`), initially `0.24.x`, and `0.25.0` only when a seam type, the `Synthesizer` trait or the `SeamHello` handshake changes in a way a consumer must notice. `fuel-kernel-seam-announce` already exists to carry the handshake; its version is the natural "protocol" number.
2. Fuel's own crates depend on them by path plus the seam line's version, as today.
3. A CI check (small) that fails when a PR changes a file under `fuel-kernel-seam*/src` without changing the seam line's version, and the reverse (the version moved but no source did), because the old failure was exactly an unnoticed coupling.
4. `baracuda-cuda-seam` then needs a republish only on a real seam break, which is the signal both projects want.

This needs the PM and CireSnave to accept one named exception to the one-number rule (it fits the clause quoted above, but it is not the clause's own example, which is an *emitter* matching its *host*), so it is a decision, not something to slip into a PR.

## 5. What I did not do

I did not check how `baracuda` itself versions `baracuda-cuda-seam` (its own number is `0.1.0-alpha.2`), nor whether `unpopped`'s consumers face the same coupling. The counts above are `git log` and crates.io listings, over the paths and dates stated; "one real break" is my reading of the three commits' subjects and diffs, not a type-by-type API diff.

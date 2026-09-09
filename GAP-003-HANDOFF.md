# GAP-003 — handoff

**State at this commit.** Not a summary of intent; a description of what is
measured, what is left, and which of my instruments you should not trust.

## Where it stands

```
ALL TEN crates in the constructing set: their LIBS compile
  fuel-graph · fuel-core · fuel-transformers · fuel-dispatch · fuel-nn
  fuel-onnx · fuel-inference · fuel-parallel · fuel-examples · fuel-lazy-examples

fuel-graph + fuel-core   --all-targets green, 1053 tests, clippy 0
fuel-transformers        LIB green; TEST TARGETS at 116 errors  <- THE REMAINING WORK
```

`fuel-inference`, `fuel-parallel` and `fuel-lazy-examples` were **structurally
unmeasurable** until `fuel-transformers` and `fuel-nn` compiled — they sit
downstream, and cargo stops at the first failing crate.

**The branch is honestly RED workspace-wide and must not merge alone.** Ruling:
one atomic pass, and NOT behind a CI change that narrows the workspace check.

## ⚠️ The remaining 116 — and why I stopped rather than finishing them

They are `fuel-transformers` test-target errors and they are mechanical. **The
sweep that would clear them has re-broken its own earlier fixes three times:**

| # | what happened | caught by |
|---|---|---|
| 1 | `?` welded onto three hand-written `.expect(PROOF)` sites | the compiler |
| 2 | stray `?` on the type-invariant proof, just written | the compiler |
| 3 | `.unwrap()` added to SEVEN production tail positions | **the guard** |

**The mechanism:** a constructor in **tail position** of a `-> Result` fn is
already correct and needs nothing. A line-local sweep sees "no handler" and adds
one. Looking *one* line ahead is insufficient — when the call spans lines the
handler can be several lines down.

⚠️ **The guard is one-for-three on this. And on TEST targets a stray `.unwrap()`
type-checks and the test still passes — so the compiler, which caught two of the
three, is a WEAKER backstop there than it was on production.** That combination
is why this is a fresh-pass job, not a tired-pass job.

## ⚠️ Two of my instruments are wrong. Do not re-derive them.

**1. The `Checking`-line detector has a false-negative mode.**
I used "does each target crate emit its own `Checking <crate> v...` line?" to
detect crates a multi-`-p` run never reached. **A WARM CACHE emits no `Checking`
line either**, so `NOT REACHED` is ambiguous between *cached* and *never ran*.
It produced **seven false NOT-REACHED readings in one run**.

**The cache-proof form** is `--message-format json` and a check for each crate's
`package_id` appearing at all:

```
cargo check -p <...> --all-targets --message-format json
# then: does each crate's package_id appear in any emitted record?
```

**2. The lib-only guard is a DELTA instrument, not an absolute one.**
After a bulk edit, `cargo build -p <crate>` (no `--all-targets`) compiles no
`#[cfg(test)]`, so a non-zero result means the sweep escaped into production —
**but only if you know the lib was 0 beforehand.** Mid-row a non-zero count is
ambiguous with "not finished". I misread an 8 as an escape when it was simply
remaining work.

## The proof taxonomy this row produced

The line is **where the proof lives**, never the return type. Two functions with
identical shape (a constructor in a `-> Self` fn) get opposite answers.

| flavour | holds for | example |
|---|---|---|
| **type invariant** ⬅ strongest | **every state the value can be in** | `fuel-nn` `Var::tensor` — `data` private, both write paths check `len == shape.elem_count()` |
| proof by preceding guard | this call, by an explicit check | `lazy_metavoice::anchor_speaker` |
| local proof | this call, by construction | mask builders, `eye`, `arange_step`, `elu` |
| partially local | half of it | `rope_tables_const_batched` — `b` IS `positions.len()`; the per-call length is not local |
| proof elsewhere | somewhere, not here | `rope` — `build_rope_tables` owns the length |
| no proof, no channel | nothing | `rope_delta_rotate_block_f32` — the signature is the real fix |
| **genuinely external** | nothing, and that is correct | mimi conv weights, `rwkv7` mix — **these take `Result`** |
| design boundary | — | `make_leaf` — the caller's `catch_unwind` IS the error channel |

⚠️ **A site that LOOKS like loaded-data-vs-config-shape may be fully proven by an
invariant one type away.** Sizing category 3 from call sites alone over-counts.

## ⚠️ And the sizing was wrong three times, in three directions

```
~1,020 sites   the original grant's framing        -- a name-match count
   4 judgement  read off the error log             -- UNDERCOUNT: downstream
                                                      errors MASK the signatures
                                                      that cause them
  21 judgement  after the mechanical sweep         -- OVERCOUNT: 13 were my own
                                                      sweep's artifacts, and they
                                                      present as `fn signature`
                                                      errors because the compiler
                                                      points at the ENCLOSING
                                                      signature when a CLOSURE's
                                                      return type is wrong
   8 judgement  after reading all 21               -- the actual number
```

**An error count OVERSTATES the sites and UNDERSTATES the judgements.**

## Message discipline for whoever continues

An `.expect()` message may claim **only what the adjacent lines prove**. Three
tailored messages were needed for the 19 settled sites because one message would
have been **false** for two of the three families. A false `.expect()` message is
the lie-with-a-green-suite this row exists to delete.

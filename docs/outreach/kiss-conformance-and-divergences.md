# Fuel ↔ KISS: conformance, divergences, and standard-improvement RFCs

**Date:** 2026-07-14 · **Status:** living record · **Scope:** the kernel seam only
(advertise / describe / discover / provision / verify), never the optimizer/executor/IR internals.

Fuel's kernel seam (FDX + FKC + `fuel-kernel-seam*` + the `SeamHello` handshake) is the named
reference **seed** for the public **KISS — Kernel Interface Standards Suite**
(github.com/ThinkersJournal/KISS, CC0, pre-1.0 draft), co-developed by the same author. So "will
KISS work for Fuel" is near-tautological — it *is* Fuel's architecture lifted into a public
standard. This doc records the genuine deltas found in a dimension-by-dimension comparison against
the now-drafted 9 sub-standards + umbrella, the better/worse-for-Fuel calls, and the recommendations
Fuel is feeding back as RFCs.

**Method / confidence.** Ten extractor passes (one per KISS sub-standard) each read both sides in
full; ten adversarial verifier passes re-checked every claim against the files. **102 of 113 claimed
divergences CONFIRMED, 9 PARTIAL, 2 REFUTED.** The six highest-stakes findings were independently
re-derived from the primary sources. This was analysis + targeted TDD fixes; no full workspace build
was run (per the one-crate build discipline).

---

## 1. Latent-bug fixes SHIPPED (KISS's discipline caught these)

Each was invisible only because Fuel and baracuda are same-endian, same-backend, co-developed peers —
the exact reader class KISS's foreign-reader freeze gate exists to protect. All four are TDD-verified
(born-red observed, then green).

| Fix | Site | KISS clause | Verified |
|---|---|---|---|
| `SEAM_MAGIC` "MAES" → "SEAM" byte-order | `fuel-kernel-seam-announce/src/lib.rs`; C-header mirror in `docs/specs/kernel-seam-interop.md` | KISS-ANNOUNCE §6.1-0004 | `seam_magic_wire_bytes_spell_seam` |
| `SeamHello.reserved1` padding made explicit + zeroed + validated | `fuel-kernel-seam-announce/src/lib.rs` | §6.2-0011 (reserved hard-reject) | `validate_rejects_nonzero_reserved`, `advertise_zeroes_all_reserved_padding` |
| Raw-bits ULP distance → IEEE total-order mapping (de-duped) | `fuel-dispatch/src/fkc/verify/ulp.rs` + `seed_cuda_ledger.rs` | — (correctness) | `ulp_distance_signed_zero_is_one`, `…_straddling_zero_is_small` |
| `relu` `-0.0` preservation (`select(x<0,0,x)`, not `max(x,0)`) | `fuel-cpu-backend/src/chassis/unary.rs` | KISS-OPS-6.15-0002 | `relu_f32_preserves_negative_zero`, `relu_f64_…` |

> **`SEAM_MAGIC` needs a baracuda lockstep.** `baracuda-seam` must adopt `0x4D41_4553` in the same
> change so the two seeds stay byte-identical. No live handshake exists today, so the brief skew is
> latent; this is a propose-first cross-project ask, not a unilateral edit.

---

## 2. Confirmed divergences (grouped)

### A. Vocabulary-ownership forks — KISS better at the boundary, Fuel fine internally

- **No Semantics op-DAG; op identity is a private enum.** FKC carries no mandatory Semantics/`op_dag`
  section (KISS-Contract §6.4-0001) and identifies ops by a private `OpKind`/`FusedOpId` (§6.3-0003
  forbids a private op enum; Grammar §6.1-0002 forbids a forkable advertisable-op set). *The single
  biggest interchange-boundary gap.* Fix path: a registered `external-op-ref` escape + a published
  `OpTag ↔ KISS-Ops-name` mapping table (keeps Fuel's build-time-closed basis interior).
- **`OpAttrs` is an interpreted struct, not an opaque byte channel.** KISS-Ops owns `OpAttrs` as a
  canonical, default-resolved, little-endian, no-elision byte blob Grammar carries uninterpreted; Fuel
  defines it as a typed struct **in the Grammar crate** with no canonical serialization.
- **Determinism enum re-spelling.** Fuel `{bitwise, same_hardware_bitwise, nondeterministic}` +
  `PrecisionGuarantee` + 5-rung `AccuracyClass`; KISS pins a 3-value comparator enum + an orthogonal
  `MathPrecision` axis. Fuel's `same_hardware_bitwise` captures a reproducibility-scope axis KISS lacks
  (see §4 rec 8).
- **Dtype set & target-capability.** Fuel's MX dtypes (`F6E2M3/F6E3M2/F8E8M0`/F4) aren't in
  KISS-CLASSIFY-6.1-0001's closed 17-token set; baracuda's `ArchSku {Sm80,Sm89,Sm90a}` can't name a
  `cpu:`/`vulkan:` target (every non-CUDA operand gets no `structure_key`). See §4 rec 7 + 9.

### B. Missing wire/protocol surface — KISS ahead, correctly gated behind the roadmap

`SeamHello` is envelope + `negotiate()` only — no availability list (§6.4), no contract-query frames
CYRQ/CRSP/CDEC (§6.5); no region-byte codec, no contract framing/checksum (→ adopt-from-KISS goal 8:
KISC self-delimiting framing as the single import frame); `SEAM_CAP_JIT_ON_REQUEST`
sits at bit 16 (KISS's EXT-experimental range — KISS puts provider features at FEAT bits 32/33);
`revision_hash` is a `u64` where KISS pins 32 bytes. All bite the first cross-process/foreign reader.

### C. Where Fuel is deliberately and rightly different

- **Dispatch geometry stays out of the FKC contract** (FKC §4.10: a `BackendCapabilities` fact, not a
  per-kernel section) because `plan-is-the-graph` gives the executor runtime arm-selection. **KISS does
  NOT force a change to how Fuel *handles* dispatch** — the umbrella explicitly excludes in-ecosystem
  load/dispatch. KISS-Contract §6.6's Dispatch section is consumer-facing *documentation of how to
  launch a specific kernel*; adopting it is mostly an asset for Fuel-as-consumer (it would replace the
  `_scalar`-suffix ABI sniffing), plus a small additive requirement for Fuel-as-provider.
- **Structured accept-block over an opaque `structure_key`** — FKC's parsed 5-flag `LayoutSpec` lets
  layout/dtype facts flow to the optimizer where a byte-matched opaque token would hide them.

---

## 3. Adopt-from-KISS goals (tracked in ROADMAP "KISS interop-standard alignment")

1. Clause↔test traceability + build-fail gate (port `tools/kiss_trace.py`).
2. Conformance + foreign-reader freeze gate (dissimilar second impl + cross-endian byte read).
3. Reference/decomposition semantic oracle for pinned op edge-cases.
4. Determinism-class-selected verify comparators + declared-ULP ceiling.
5. Oracle independence + edge-case corpus in verification.
6. `MathPrecision` reduced-mantissa axis.
7. Named `reference_function` + derived `audited_status`.
8. **KISC self-delimiting framing as the *single* kernel-import frame** (KISS-Contract §2.8 /
   §6.11) — one `KISC` magic + version + `len` + `crc32` envelope for **all** kernel data, the
   in-repo `.fkc.md` corpus *and* the future contract-query wire, so the corpus continuously
   dogfoods the foreign-reader/freeze-gate discipline (goal 2) and no local-vs-wire framing can
   drift. Build-stamp `len`/`crc32` (never hand-maintain); one kernel per KISC document, a file =
   an ordered bundle of N; a `{abort_batch | isolate}` failure knob — local build **fail-fast**,
   wire **isolate** — so framing is unified without weakening the build-time-loud invariant. The
   existing [`parse.rs`](../../fuel-dispatch/src/fkc/parse.rs) section/fence scanner survives as
   the *inner-body* parser (KISC is the outer envelope). The silent-drop/orphan half is already
   shipped (`FkcError::OrphanFkcBlock`). Cutover negotiated behind a cap bit in the KISS **FEAT
   range** (not bit 16). Full position: [`baracuda-kisc-framing-reply.md`](baracuda-kisc-framing-reply.md).

Deferred latent-bug items: `OpTag::Gelu → GeluTanh` seam rename; `op_to_attrs` load-bearing-attr
projection (frozen `OpAttrs` schema change); integer wrapping path (confirm reachability first); MKL
`vs_max` NaN-suppression guard (currently dormant).

---

## 4. Where Fuel is ahead / standard-improvement RFCs → KISS issues

Filed as issues on github.com/ThinkersJournal/KISS (index kept current below). Each benefits any
third-party kernel/compute orchestrator, ML library, or inference engine — not just Fuel.

1. **In-process / same-language binding profile** — map CYRQ/PRSP/CDEC + region/contract semantics
   onto an ABI-level trait/vtable with the same identity/typed-decline/never-panic obligations, instead
   of requiring byte framing. The biggest adoption barrier for orchestrators that own their dispatch.
2. **Novel-region provision** — an optional `OpDef`/Semantics payload on the build-on-miss request (or
   a wired KISS-Emit hand-off) so a runtime-discovered fusion with no pre-existing `structure_key` can
   be synthesized. Fuel's `JitRequest` already carries the op-DAG; every runtime JIT frontend hits this.
3. **Consumer-side artifact-binding + `abi_model` token** `{pointer-abi | descriptor-set |
   buffer-index | host-closure}` so non-C-ABI providers (Vulkan/Metal/WebGPU/SYCL/host-closure) can
   declare a conformant Interface; plus a typed decline when a loader can't handle the artifact kind.
4. **Normative consumer-side empirical verification** that verifies the emitter's *declared*
   determinism class before trusting a bit-identity claim, with a persistent revision-hash-keyed
   evidence ledger + a defined downgrade path (Fuel's working reference resolves Conform Appendix D Q1).
5. **Pin the ULP-distance function** (sign-magnitude total-order before subtraction) + ship a reference
   comparator with a sign-boundary golden vector + a mandatory edge-case corpus.
6. **Unknown-enum-value policy distinct from unknown-bit** — an admissibility-affecting value is a
   typed decline; only advisory fields warn-and-default (lift FKC §11.1's table).
7. **Resolve the MX/microscaling dtype hole** — add I16 + the 4/6-bit MX floats + the E8M0 scale to the
   dtype set, or bless the quant-sidecar path normatively (unblocks the OCP-MXFP ecosystem).
8. **Two orthogonal determinism axes** — the per-op comparator class KISS has, PLUS a per-kernel
   reproducibility-scope axis `{portable-bitwise, same-device-bitwise, nondeterministic}`; also lift
   Fuel's `Negotiated{capabilities = local & remote}` into a normative clause. Required by any
   captured-graph-replay / KV-cache-replay / persistent-cache engine.
9. **Namespaced all-hardware `target_capability`** (`cuda:sm89` / `vulkan:spirv1.6` / `cpu:`) +
   fold the output operand into the `structure_key`.
10. **Reference wire codec + a per-op identity-bearing OpAttrs table** — the reference seed is a shared
    Rust-type crate with no serializer, so the foreign-reader freeze gate has nothing to read.
11. **Consumer-side never-panic obligation** (KISS loads it entirely on the provider).
12. **Rejection-feedback backchannel** (KISS-Synth is strictly one-shot; no home for `on_rejected`).

### Issue index (filed 2026-07-14, github.com/ThinkersJournal/KISS)

**New issues (no prior home):**

| # | Item | KISS # |
|---|---|---|
| 1 | In-process / same-language binding profile | [#18](https://github.com/ThinkersJournal/KISS/issues/18) |
| 3 | Artifact→callable binding + `abi_model` token | [#19](https://github.com/ThinkersJournal/KISS/issues/19) |
| 5 | Pin the ULP-distance function + edge corpus | [#20](https://github.com/ThinkersJournal/KISS/issues/20) |
| 6 | Unknown-enum-value policy distinct from unknown-bit | [#21](https://github.com/ThinkersJournal/KISS/issues/21) |
| 9 | Namespaced all-hardware `target_capability` + output-operand-in-key | [#22](https://github.com/ThinkersJournal/KISS/issues/22) |
| 10 | Reference wire codec + per-op identity-bearing OpAttrs table | [#23](https://github.com/ThinkersJournal/KISS/issues/23) |
| 11 | Consumer-side never-panic obligation | [#24](https://github.com/ThinkersJournal/KISS/issues/24) |
| 8b | Lift `Negotiated{caps = local & remote}` into a normative clause | [#25](https://github.com/ThinkersJournal/KISS/issues/25) |

**Commented on existing issues (Fuel's POV added):**

| Item | Existing KISS issue |
|---|---|
| 2 Novel-region provision + single-classifier division | [#11](https://github.com/ThinkersJournal/KISS/issues/11) |
| 4 Consumer-side empirical verification (oracle independence + verify declared class + downgrade ledger) | [#16](https://github.com/ThinkersJournal/KISS/issues/16) |
| 7 MX/microscaling dtype hole (E8M0 scale, quant-sidecar) | [#9](https://github.com/ThinkersJournal/KISS/issues/9) |
| 8 Same-device reproducibility-scope determinism axis | [#13](https://github.com/ThinkersJournal/KISS/issues/13) |
| 12 Rejection-feedback backchannel (`on_rejected`) | [#12](https://github.com/ThinkersJournal/KISS/issues/12) |
| — Two-step light-offer/heavy-fetch handover | [#5](https://github.com/ThinkersJournal/KISS/issues/5) |

---

## 5. Corrections to prior internal beliefs (from the adversarial pass)

- FKC cost expressions **are** wired to the ranker (`ranker/cost.rs` `compute_static_costs` →
  `fkc::cost_estimate`). The FKC-verification-gap memory is stale on the cost-trampoline point.
- The verification ledger **does** check correctness (`max_ulp` vs a CPU reference), not
  determinism-only.
- Scalar CPU `Maximum`/`Minimum` already propagate NaN; MKL `vs_max`/`vd_max` NaN-suppression is
  dormant (defined, not wired to `Op::Maximum`).
- KISS-Contract §6.5/§6.6 (Interface/Dispatch) *do* exist — so several "KISS has no home for X"
  observations are narrower than first stated (see §2.C).

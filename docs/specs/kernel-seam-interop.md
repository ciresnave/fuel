# Kernel-Seam Interop Contract — Profile v1

**Status: RATIFIED — Profile v1 (2026-06-20), branch `feat/kernel-contracts-dlpack`.** Ratified by all
three parties: **Fuel**, **Baracuda** (its A1/A2/E1 fusion-patterns conditions resolved in rev 3/4 + the
`SeamHello` C-ABI pinned, §3.1), and **Vulkane** (confirmed the named BDA surface, §7.2). This is the
single, ratifiable description of how software on the two sides of Fuel's **kernel seam** communicate:
which party implements which subset, how a connection agrees on a spec version at runtime, and what each
side may rely on. It is the **cross-project contract**; the [FDX spec](dlpack-extension.md) and the
[FKC spec](kernel-contract-format.md) (+ [FKC fusion patterns](fkc-fusion-patterns.md)) are its
**normative annexes** — they carry the byte-level and field-level detail, this document pins *who speaks
what, at which version*.

**Audience:** the maintainers of Fuel, Baracuda (CUDA kernel provider), Vulkane (Vulkan FFI), and any
future ecosystem that wants to exchange tensors with — or ship kernels to — Fuel. Reviewable in one
sitting; ratify here, implement against the annexes.

**Why this exists.** Fuel's kernel-announcement and -contracting design moved onto FDX/FKC. The specs are
authoritative and Fuel-side, and the parties have exchanged point-in-time acceptances — but there was no
*living* artifact that says, in one place, what each party must implement, at what version, and how a
connection negotiates that version. This document is that artifact, and **Profile v1** is its first
ratified version (per the process in §8). See [10-decisions-log §2026-06-20](../architecture/10-decisions-log.md).

---

## 1. The seam parties and their subsets

The two sides of the seam do **not** speak an identical dialect; they speak **disjoint, well-defined
subsets** by role. "Conformance" means implementing *your* subset, not all of it.

| Party | Role | FDX | FKC | JIT-on-request | Telemetry |
|---|---|---|---|---|---|
| **Fuel** | the planner / consumer / strategist | full (producer + consumer) | importer (registration surface) | **requester** (strategist) | consumer + emitter |
| **Baracuda** | CUDA kernel provider | full (consumer of Fuel-produced tensors; producer on return) | **provider** (ships kernel contracts) | **synthesizer** (optional, capability-gated) | reads Fuel's telemetry |
| **Vulkane** | Vulkan FFI (no kernels) | **FDX-only**, BDA subset | **N-A** (ships no kernels; Fuel's Vulkan kernels are internal Slang) | N-A | N-A |
| **Future FKC provider** | any backend shipping kernels | full | provider | optional | optional |

Two consequences worth stating plainly:

- **Vulkane's surface is small and almost entirely settled** — it is the FDX BDA handoff (§4.1's FDX-core
  rows + the byte-level path in FDX §3.3.1; reconciled in §7.2) and nothing else. If Vulkane ever exposes its *own* compute entry points Fuel could dispatch to, those
  would carry FKC contracts *at that point*; until then Vulkane is FDX-only.
- **Baracuda's surface is the large one** — full FDX + FKC + (newly) the JIT-on-request layer (§5) — and
  it is where ratification effort concentrates.

The contract is **open**: a future ecosystem implements Profile v1's published subset and negotiates
(§3) without a bilateral deal with Fuel.

---

## 2. Profile v1 — the bundled version

The seam has several independently-drafted axes (FDX, FKC, the JIT protocol, the telemetry vocabulary)
that would, if versioned separately, produce a combinatorial version matrix nobody can reason about.
Instead a **Seam Profile bundles one pinned combination**, and the handshake (§3) negotiates a single
profile integer. **Profile v1 pins:**

| Axis | Pinned version | Annex / anchor |
|---|---|---|
| FDX schema | `FDX_VERSION_1` (= 1; major-only, independent of the DLPack ABI version axis) | [dlpack-extension.md §5.2](dlpack-extension.md), `FDX_MAGIC` / `FDX_VERSION_MAX = 1` |
| FKC contract format | `fkc_version: 1` (`FKC_VERSION_MAX = 1`) | [kernel-contract-format.md §0/§11](kernel-contract-format.md) |
| FKC fusion patterns | rev 4 (declarative `pattern:` grammar) | [fkc-fusion-patterns.md](fkc-fusion-patterns.md) |
| JIT-on-request protocol | v1 (defined in §5; implementation pending both sides) | this doc §5 |
| Telemetry vocabulary | v1 (`ImplId` 5-field basis tuple; `StructureKey`; `DispatchRecord`/`FusionMissRecord`/`SequenceRecord`) | [kernel-contract-format.md §4](kernel-contract-format.md), `baracuda-telemetry-plan.md §9` |

A party "supports Profile v1" iff it implements the **role-relevant** rows of the conformance matrix (§4)
for these pinned versions. Earlier *drafts* of any axis are **retired** — Profile v1 is the floor; nothing
needs to support a pre-ratification draft. (Versioning is *not* abandoned — the version fields stay so
Profile v2 can be negotiated later, per §3 and §6.)

---

## 3. Versioning & Handshake (first-class)

A connection **negotiates a profile before any tensor or kernel crosses the seam**. This is the piece
that lets Fuel, Baracuda, Vulkane, and future ecosystems support *multiple* profile versions over time
without another lockstep flag-day. It is designed in from Profile v1 even though v1 negotiation is
trivial — a handshake retro-fitted at v2 is the classic "everyone silently assumed v1" bug.

### 3.1 The frozen envelope

The wire format by which each side **advertises its supported profiles** must outlive every profile it
negotiates — otherwise a v1 party and a v5 party can't even exchange version lists ("you'd need a
handshake to agree on the handshake"). So the envelope is deliberately tiny and **frozen forever**:

```c
/* The negotiation envelope — a FIXED-SIZE POD, STABLE FOR ALL TIME. */
#define SEAM_MAGIC            0x4D414553u   /* on-wire bytes 53 45 41 4D = "SEAM" (LE u32); never changes */
#define SEAM_ENVELOPE_VERSION 1u            /* the envelope's OWN version; designed never to bump */
#define SEAM_MAX_PROFILES     16u           /* fixed cap on simultaneously-advertised profiles    */

typedef struct {
  uint32_t magic;             /* == SEAM_MAGIC (LE bytes "SEAM", offset 0)                  */
  uint8_t  envelope_version;  /* == SEAM_ENVELOPE_VERSION                                   */
  uint8_t  reserved[3];       /* == 0                                                       */
  uint16_t profiles_len;      /* number of valid entries in profiles[] (<= SEAM_MAX_PROFILES) */
  uint16_t profiles[SEAM_MAX_PROFILES];  /* ascending; entries [profiles_len ..] are 0      */
  uint8_t  reserved1[6];      /* == 0; alignment padding made explicit (offset 42), zeroed + validated */
  uint64_t capabilities;      /* optional-feature bitset within the selected profile (§3.4) */
} SeamHello;                   /* fixed 56 bytes; offsets frozen, asserted like FDX structs  */
```

> **Wire-magic pin (KISS-ANNOUNCE §6.1-0004).** `SEAM_MAGIC` is the *numeric* constant
> `0x4D414553`, chosen so a little-endian `u32` write places the ASCII bytes `53 45 41 4D`
> (`"SEAM"`) at offset 0. Do **not** use `0x5345414D` (the big-endian spelling of "SEAM"),
> which serializes LE to `4D 41 45 53` = `"MAES"`. Both Fuel (`fuel-kernel-seam-announce`)
> and any provider (e.g. `baracuda-seam`) MUST use `0x4D414553` so their envelopes are
> byte-identical; this pairing was flipped from the earlier inverted value in lockstep.

Everything mutable lives *inside* a negotiated profile; the envelope around it is immutable. New
negotiable information goes into a *profile*, not the envelope.

**The C realization is pinned (no variable-length return).** The `profiles` list is a **fixed-max array**
(`SEAM_MAX_PROFILES = 16`) with a `profiles_len` count — *not* a variable-length member — so `SeamHello`
is a fixed-size POD with frozen field offsets, and the calling convention is an **out-param** (callee
fills caller-allocated storage; never allocates, never returns by value across the ABI):

```c
/* Provider exposes exactly this. Returns 0 on success; never panics/aborts. */
int  baracuda_seam_hello(SeamHello* out);
```

16 simultaneously-advertised profiles is absurdly generous in practice — profiles *retire* as the floor
advances (§3.6), so the live set stays tiny. If that cap were ever reached, raising it is the one change
that bumps `SEAM_ENVELOPE_VERSION` (additive, negotiated like anything else) — which is exactly what that
field exists for. The struct layout (size, offsets, `SEAM_MAX_PROFILES`) is **frozen** and cross-checked
by `offset_of!`/size asserts on both sides, the same discipline FDX uses for its `#[repr(C)]` structs.

### 3.2 The negotiation algorithm

```
selected = max( local.profiles ∩ remote.profiles )
if selected is empty:  → SeamVersionMismatch   (a hard, typed error; the connection does NOT proceed)
else:                  → operate at Profile `selected`, with capabilities = local.capabilities & remote.capabilities
```

- **Highest mutually-supported wins** (TLS-style), not per-axis negotiation.
- **Hard-fail on disjoint, never proceed on an assumed version.** Operating a seam on an unnegotiated or
  guessed profile is forbidden — the same build-time-validation / never-silent-coercion discipline FDX
  already mandates ([dlpack-extension.md §9](dlpack-extension.md), P10/G6), applied to the wire.
- At Profile v1 this is trivial: both advertise `[1]` → select 1, or hard-fail. The *value* is that the
  advertise → select → record → hard-fail path exists and is exercised from day one.

### 3.3 Session-negotiated profile vs. per-artifact version tags

Two version layers, kept distinct:

- The **session-negotiated profile** (the §3.2 result) — one per connection.
- **Per-artifact version tags** that already exist in the annexes: `FDXSidecar.version`
  ([dlpack-extension.md §5.3](dlpack-extension.md)) on each tensor, and `fkc_version` in each FKC bundle's
  front-matter ([kernel-contract-format.md §0](kernel-contract-format.md)).

The handshake sets the session floor/ceiling; each artifact then carries its own tag that the receiver
validates **against the negotiated profile**. A tensor or contract tagged outside the negotiated profile's
range is a clean, detected error — *not* silently coerced. (FDX already does the per-tensor half: an
unrecognized `FDXSidecar.version` is treated as sidecar-absent, [§9.2](dlpack-extension.md); FKC already
does the per-bundle half: `fkc_version > FKC_VERSION_MAX` is rejected, [§10 rule 1](kernel-contract-format.md).
Profile v1 adds the *session* layer above them.)

### 3.4 The capability-flag layer (optional features within a profile)

A profile defines the *floor everyone agrees on*; **capability flags** carry the optional features *within*
it. FDX already has this layer — the `BackendProbe` capability tokens
([dlpack-extension.md §12](dlpack-extension.md)): `DlpackExtV1`, `DlpackExtMx`, `DlpackExtGgml`,
`DlpackExtAffine`, `DlpackExtSymbolic`, `DlpackExtGather`. Profile v1 **adopts those tokens as the
low bits of `SeamHello.capabilities`** and reserves additional bits for FKC-/JIT-level optional features
(e.g. `SeamCapJitOnRequest` — a party supports Profile v1's FDX+FKC but has not yet implemented the JIT
endpoint, §5). The negotiated capability set is `local & remote`; a feature neither side flags is simply
not used on that connection (the planner routes around it, never trial-and-falls-back —
[§12 / §9.3](dlpack-extension.md)).

### 3.5 Per-party surface

The handshake surfaces differently by role:

- **Baracuda (FKC provider, FFI):** the out-param C-ABI negotiation entry point `int
  baracuda_seam_hello(SeamHello* out)` (§3.1; Fuel reads it and runs §3.2) — **plus** each FKC bundle
  declares `seam_profiles: [1]` in its front-matter, so an imported contract states which profiles it
  targets and the importer rejects a contract outside the negotiated profile.
- **Vulkane (FDX-only):** **no new FFI — nothing on the wire originates from Vulkane.** The Vulkane FFI
  exposes no `BackendCapabilities`/`BackendProbe`; those are *Fuel-side* FDX abstractions. In practice
  **`fuel-vulkan-backend` (Fuel-side glue) advertises the FDX version, derived from the linked `vulkane`
  crate version** — Vulkane itself does literally nothing to version or break. The "profile" for Vulkane is
  exactly *FDX v1, BDA subset*; the §3.1 envelope is the FKC-provider form, and FDX-only parties use the
  lighter FDX-version negotiation [FDX §12](dlpack-extension.md) already incorporates.
- **Fuel:** advertises its supported profile set + capabilities, runs the negotiation, records the result
  per connection, and hard-fails on mismatch.

### 3.6 Floor advance & retirement

A profile is **retired** when the mutually-agreed *floor advances past it* — not by another flag-day. Each
party drops the old profile integer from its advertised set once all active peers support a newer one; the
negotiation then naturally selects the newer floor. Profile v1's clean break (retire the pre-ratification
*drafts*) and all future retirements use the *same* mechanism: advertised-set membership + the
highest-common rule.

---

## 4. Conformance matrix

What each party MUST / MAY / does-not implement, for Profile v1. Legend: **MUST** = required to claim
Profile v1 conformance in that role; **MAY** = optional, capability-gated (§3.4); **N-A** = not in that
party's role; **PROD** = producer-side obligation (when *emitting* a tensor). Anchors cite the FDX/FKC
annexes.

### 4.1 FDX — core (the honesty invariant; non-negotiable for any FDX participant)

| Feature | Fuel | Baracuda | Vulkane | Anchor |
|---|---|---|---|---|
| Base `DLTensor` is always valid standard DLPack (sub-byte/quant ride as opaque `uint8`) | MUST | MUST | MUST | FDX §3, V3 |
| Explicit strides on versioned export (never NULL) | PROD | PROD | N-A¹ | FDX §3.2, V11 |
| 256-byte `data` alignment on boundary-(b) export | PROD | PROD | N-A¹ | FDX §3.3, V12 |
| Signed-stride OOB range check (negative strides described first-class) | MUST | MUST | N-A¹ | FDX §3.2.1, V13 |
| Typed errors, never panic, never silent coercion; producer refuse-or-materialize (`IS_COPIED`) | MUST | MUST | MUST | FDX §9, P10/G6 |
| No raw pointers in serialized form (buffer refs are capability-relative indices) | MUST | MUST | MUST | FDX P7 |

¹ Vulkane never sees `strides`/host alignment — strides live in Fuel's Slang indexing; under BDA the
descriptor-offset constraint is moot ([vulkane-reply](../outreach/vulkane-reply.md)). Listed N-A for
Vulkane's role, MUST for Fuel-as-producer of the tensors Vulkane's path consumes.

### 4.2 FDX — optional features (capability-gated via §3.4 tokens)

| Feature | Fuel | Baracuda | Vulkane | Token / Anchor |
|---|---|---|---|---|
| Sub-byte / microscaling dtype (`FDXDTypeExt`) | MAY | MAY | N-A | `DlpackExtV1`; FDX §6.1, V4 |
| Quant family GGML_BLOCK (baked inline scales) | MAY | MAY | N-A | `DlpackExtGgml`; FDX §6.2, V5 |
| Quant family MX (F8E8M0 per-block) | MAY | MAY | N-A | `DlpackExtMx`; FDX §6.2 |
| Quant family AFFINE_INT/FLOAT/**AFFINE_BLOCK** (nf4/QLoRA, separate block scale) | MAY | MAY | N-A | `DlpackExtAffine`; FDX §6.2, V5 |
| Symbolic extents (Scalar/Range/**Affine**, live-vs-capacity) + `FDXSymEnv` | MAY | MAY | N-A | `DlpackExtSymbolic`; FDX §6.4, V7/V16/V17 |
| Gather / paged-blocks residency (vLLM KV cache) | MAY | MAY | N-A | `DlpackExtGather`; FDX §6.9, V18–V21 |
| Residency tier / substrate class / storage class | MAY | MAY | MAY² | FDX §6.6/§6.7 |
| Multi-output bundle (`FDX_FLAG_IS_BUNDLE`) | MAY | MAY | MAY² | FDX §6.8 |
| Negative-stride *consumer* acceptance | — | MAY (per-kernel FKC `layout.reverse_strides`) | N-A | FKC §4 / FDX §3.2.1 |

² Vulkane allocates per buffer-table role (DATA/SCALE/ZERO_POINT/POOL/BLOCK_TABLE/CONTEXT_LENS/bundle
slots) and binds plural buffers; it carries these structurally but does not *interpret* them.

### 4.3 FKC — kernel advertisement (provider obligations; N-A for Vulkane)

| Obligation | Fuel | Baracuda | Vulkane | Anchor |
|---|---|---|---|---|
| Front-matter: `fkc_version`, `provider.{name,backend,kernel_source,link_registry,revision_base}`, `seam_profiles:[1]` | importer | MUST | N-A | FKC §0, §3.1; this doc §3.5 |
| Per-kernel: `kernel`, `op_kind` XOR `fused_op`, `blurb`, `entry_point`, `kernel_revision_hash` | importer | MUST | N-A | FKC §3.3, §10 rule 2 |
| `accept`/`return` contract, `op_params`, `caps`, `cost.{provenance}`, `precision`, `determinism` | importer | MUST | N-A | FKC §3.3, §4 |
| Every dtype/quant token in a contract is in FDX's normative table (FDX = the shared vocabulary) | enforce | MUST | N-A | FKC §10 rule 16 |
| `entry_point` resolved via provider `link_registry` (`&[(symbol, KernelRef)]`); unresolved → typed error | resolve | MUST expose | N-A | FKC §12.6 |
| `ImplId` = `(backend, op, dtypes, kernel_source, kernel_revision_hash)` — five separable wire fields, never a hash | emit | MUST | N-A | FKC §4.11 (Baracuda-locked) |
| `pattern:` (declarative) for a pattern-recognized fused op | importer³ | MUST when applicable | N-A | fkc-fusion-patterns §3 |

³ Fuel's declarative-pattern engine is a stub today (`PatternKind::Declarative => false`,
`fuel-graph/src/opt.rs:434`); implementing it is the Tier-2 prerequisite (§5, [10-decisions-log G4](../architecture/10-decisions-log.md)).

### 4.4 JIT-on-request & telemetry (the new Profile v1 layer; §5)

| Item | Fuel | Baracuda | Vulkane | Anchor |
|---|---|---|---|---|
| Telemetry vocabulary: `StructureKey` (provider computes, Fuel calls — never reimplements), `ImplId`, `DispatchRecord` | call + emit | MUST provide `structure_key` | N-A | FKC §4; baracuda-telemetry-plan §9 |
| Missing-fusion signal: closed-world `FusionMissRecord{NoBackendKernel}` (v1 headline); open-world `SequenceRecord{fused_as:None}` (deferred) | emit | reads | N-A | this doc §5.3 |
| JIT-on-request endpoint (Fuel sends a partial base map + budget; provider returns kernel + FKC contract + recipe) | requester (strategist) | MAY synthesize (`SeamCapJitOnRequest`) | N-A | this doc §5 |

---

## 5. The JIT-on-request contract (new in Profile v1)

This is the contract surface neither party has agreed yet (decided Fuel-side 2026-06-20; **Baracuda has not
seen it** — §7). It is **defined** in Profile v1 and **capability-gated** (`SeamCapJitOnRequest`) so a
Profile-v1 Baracuda that hasn't built it yet is still fully conformant; the bit lights up when both sides
implement it.

### 5.1 The division of labor (constitution-preserving)

- **Fuel is the STRATEGIST.** It decides *which* sub-region of the base map to ask for, *when* (idle-time,
  whole-machine resource-aware — Fuel is the only layer that sees the host + every device), and whether to
  *adopt* the result (cost-gated, via the route picker). Choosing the region **is** the fusion decision.
- **The provider (Baracuda) is the SYNTHESIZER.** It builds the best kernel for the **Fuel-chosen** region,
  applying its hardware knowledge *within* it. **No backend-side opportunity-finding** — this is not
  backend-internal fusion ([09-non-goals](../architecture/09-non-goals.md)); the constitution holds (the
  optimizer that reads the DAG keeps the intelligence; backends advertise and synthesize, never decide
  strategy).
  - **Clarification (on the record, at Baracuda's request).** "No backend-side opportunity-finding" bounds
    *region selection*, not *optimization within the region*. The synthesizer **MAY** use arbitrarily
    powerful fusion machinery — including an e-graph / equality-saturation optimizer — to produce the best
    kernel for the region Fuel handed it. That is precisely "synthesize the best kernel for this subgraph,"
    and it is fully compatible with the constitution: the provider's optimizer is **pointed only inward** at
    a Fuel-chosen region, never scanning Fuel's graph to *pick* regions. What stays Fuel's alone is choosing
    *which* sub-base-map to request and whether to *adopt* the result.

### 5.2 The request / response shape (protocol v1)

The handover is **two-step** — a light wire response, then a lazy artifact fetch at adopt — so a kernel
Fuel's cost-gate declines transfers nothing (revised 2026-07-04 to Baracuda's built impl + frozen):

```
JitRequest  {  region:   partial base map (primitive subgraph — also the recipe's decompose),
               operands: [OperandDesc]  (shapes/dtypes of `target`; raw, synthesizer-classified),
               arch:     ArchSku         (the backend/device of `target`),
               budget:   JitBudget { max_compile_ms }  (Fuel sets it; coarse — bounds optimizer effort) }

// step 1 — synthesize → a LIGHT handle (or decline); no heavy artifact crosses here:
JitResponse ::= Synthesized { entry_point }  |  Declined { reason }

// step 2 — after Fuel's cost-gate adopts, take_kernel(entry_point) → the artifact:
SynthArtifact { artifact: bytes,  kind: Ptx | Cubin  (always loadable — non-loadable ⇒ Declined),
                link:     LinkEntry { entry_point, symbol, structure_key, revision_hash },
                contract: full FKC contract markdown  (accept/return/op_params/cost/precision/determinism
                          + the re-fuse `pattern:`) }
```

`Synthesizer` is Fuel's trait (the synthesizer `impl`s it, so Fuel is type-decoupled):
`synthesize(&JitRequest) -> JitResponse` + `take_kernel(&str) -> Option<SynthArtifact>`. The methods are
**sync**; the impl is `Send + Sync` + interior-mutable, so Fuel drives `synthesize` on a
**background / idle-time thread** (the G7 background-re-optimization trigger), never the realize path.

**No `recipe` on the wire** — it is byte-reconstructable from `contract`: `recipe.pattern` *is* the
contract's embedded `pattern:` block, and `recipe.decompose` is that block with `pattern:`→`decompose:`
swapped. Fuel derives `decompose` by the **string swap on `contract.pattern`**, not by re-serializing the
region (which risks serializer drift between Fuel's serializer and the synthesizer's). The kernel *body* is
optimized before codegen, but the contract's `pattern:` is the **un-optimized** region — exactly the
primitive subgraph the decompose must expand back to.

Adoption is **cost-gated**: the returned kernel enters the binding table as one more multi-sibling
alternative; the route picker adopts it only if it *wins*. A kernel that never wins is never used (and the
cache-pruning policy, [10-decisions-log G8](../architecture/10-decisions-log.md), governs eviction:
drive-space-capped, loses-across-every-model only). The build-time-closed primitive basis still holds — a
JIT kernel decomposes into Fuel's *existing* primitives or prompts a Fuel-side build-time `Op`-enum
extension; it never introduces a new primitive ([G3](../architecture/10-decisions-log.md)).

### 5.3 The telemetry work-order feed

The request feed is the missing-fusion telemetry: a ranked stream of fusion opportunities Fuel detected and
lacks a kernel for. **None of this exists yet** (Fuel sees fusions it *performed*, not *wanted-but-lacked*;
the graph-layer hook + the `structure_key`/`DispatchRecord` base-emission seam are unbuilt). Sequencing:
the **closed-world `FusionMissRecord{NoBackendKernel}`** (a recognized fusion-eligible chain realized as N
primitives because the kernel was absent, against a *known* fused-op id) is the v1 headline — its consumer
is the already-extensible Tier-1 binding table (append a `BindingEntry`). The **open-world
`SequenceRecord{fused_as:None}`** (a frequent realized chain matching *no* known id — discovered by
*observation*, not subgraph enumeration) is **deferred**, because its consumer is Tier-2 runtime declarative
registration. We never enumerate the subgraph space or seek a whole-model fusion.

---

## 6. Compatibility & change policy

- **Additive within a profile.** New facts go into reserved fields / capability bits; older readers use the
  `(version, flags, struct_bytes)` joint detection FDX already mandates ([§5.2](dlpack-extension.md)) and
  ignore unknown trailing bytes. No profile bump for additive change.
- **A profile bump is required only** when a field's meaning changes incompatibly, a new *mandatory* block
  is introduced, or an axis's pinned version changes incompatibly. A bump cuts a new Profile integer (v2);
  parties advertise `[1, 2]` during the overlap, then drop `1` once all peers carry `2` (§3.6).
- **The envelope never changes** (§3.1). If it ever must, that is `envelope_version`'s job — and the design
  exists to make that never happen.
- **Retirement is by floor advance** (§3.6), never a flag-day.

---

## 7. Sync reconciliation — what each party last accepted vs. Profile v1

This is the drift audit the contract exists to make durable.

### 7.1 Baracuda — accepted against the **~2026-06-19 draft**

**Confirmed / locked** (from `baracuda-reply.md`, `baracuda-reply-2.md`): FDX + FKC core; the honesty
invariant; negative-stride capability-gating; gather + affine extents; the **`ImplId` 5-field basis tuple**
(*"ready to freeze"*, separable wire fields, round-trip test added); **`StructureKey`** computed by
Baracuda and *called* by Fuel via a minimal `FdxOperandDesc` projection (strides, dtype, alignment, quant,
symbolic extent) — Fuel never re-derives it; Judge per-`(op,dtype,size_class,backend,device,kernel_source)`
retention incl. losers (Open-Q-1 = **YES**, `candidates[]` feasible); miss = "best admissible match is a
generic contract"; the dtype codes `I4=0x0102 / U4=0x0103 / B1=0x0104`; `F32Strict` as a precision *mode*
not a wire dtype; FKC as a *generated* projection of Baracuda's `KernelSku`/OP-matrix.

**Drift since (must be re-confirmed for Profile v1):**
- **The version handshake itself** (§3) — new; Baracuda needs the `baracuda_seam_hello()` entry point +
  `seam_profiles` front-matter.
- **The adaptive runtime-fusion / JIT-on-request layer** (§5) — **not seen**; a new contract surface.
- **Tier-2 declarative registration** + the declarative `pattern:` being its mechanism — not seen.
- Any FDX/FKC change after 2026-06-19 on the branch (the recipe-principle / two-tier re-scope of
  2026-06-20 is wording, not a wire change, but should be acknowledged).
- **Baracuda's build state (corrected 2026-06-20 per Baracuda's reply — the earlier draft understated it):**
  `structure_key` **and** `baracuda-kernelgen` are **built and GPU-validated on Baracuda PR #2** — the
  kernelgen carries an IR, three schedules, `f32/f16/bf16/f64`, and `derive_pattern` now emitting
  `AddScalar`/`MulScalar` patterns with `extract:`. What genuinely remains for a *conforming publish*: the
  **full FKC contract emitter** (`pattern:` is emitted today; `accept`/`return`/`cost`/`precision` next),
  the **`link_registry`**, **`baracuda_seam_hello()`**, and packaging the **`structure_key` callable**.

### 7.2 Vulkane — **frozen 2026-06-19** (FDX-only, BDA)

**Confirmed by Vulkane 2026-06-20** (conformant as-is). The Vulkane contract is stated as **behavior +
named surface**, not a crate-version number (Vulkane bumps for reasons unrelated to this seam; pinning a
`≥ x.y.z` floor would silently keep asserting conformance across an unrelated major break):

- **Behavior (normative):** on `kDLVulkan`, FDX `data` is a `VkDeviceAddress`; `byte_offset` stays a
  separate wire field; **`fuel-vulkan-backend` folds `data + byte_offset` at dispatch**. Realized via the
  named surface `AllocatorOptions::buffer_device_address` (`new_with_options`) + `Buffer::device_address()`
  + per-buffer `BufferUsage::SHADER_DEVICE_ADDRESS` + the `bufferDeviceAddress` device feature.
- **Version (informative):** `vulkane 0.8.2` (2026-06-19, crates.io) is the *first* version exposing that
  named surface — the `≥ 0.8.2` floor is a convenience pointer, not the contract. **A Vulkane major bump
  triggers a re-check of the named surface above**, not a silent pass.

- **`SHADER_DEVICE_ADDRESS` is a producer-side obligation, not a Vulkane auto-guarantee** (Vulkane's #1
  fix): Vulkane never auto-applies it. For `Buffer::device_address()` to return a valid address,
  **`fuel-vulkan-backend` must satisfy all three together**, or it silently gets a wrong/erroring address —
  exactly the "never silent coercion" discipline this contract champions:
  1. construct the allocator via `new_with_options(.., buffer_device_address: true)` (else pooled-buffer
     addresses are invalid on strict drivers — the bug 0.8.2 fixed);
  2. create **each** buffer-table buffer with `BufferUsage::SHADER_DEVICE_ADDRESS` (per-buffer usage, not
     an allocator-wide property);
  3. enable the `bufferDeviceAddress` device feature at device creation
     (`DeviceFeatures::with_buffer_device_address()`, Vulkan 1.2 core) — else `vkGetBufferDeviceAddress`
     isn't loaded and `device_address()` returns `Error::MissingFunction`, not a bad value.
- **Final-address alignment is Fuel's to own** (Vulkane's minor clarification): Vulkane honors the
  buffer's `VkMemoryRequirements.alignment` (`minStorageBufferOffsetAlignment`, often 16–256) — *not* a
  guaranteed 256-aligned base address. Because `byte_offset` is folded at dispatch and Fuel owns it, only
  the **final `data + byte_offset`** must meet the kernel's load alignment; **`fuel-vulkan-backend` ensures
  final-address alignment** (the producer-side 256-byte rule, FDX §3.3, is Fuel's obligation, not a
  Vulkane base-address guarantee).
- The rest, all from the frozen 2026-06-19 design: push-constant transport in buffer-table **role → index**
  order (plural buffer table, descriptor-set-free); sidecar stays Fuel-side (no Vulkane ABI slot); signed
  strides a non-issue (never reach a binding); 6-dim shape/stride cap ignored on Vulkane's side. **No FKC,
  no JIT, no kernels** — and the §4.2 buffer-table roles are `fuel-vulkan-backend`'s to map; Vulkane is
  role-agnostic (footnote² stands).

**Drift since:** none on the wire — Vulkane confirmed the BDA subset maps to **FDX v1 unchanged** and the
light FDX-version path (§3.5) is acceptable.

### 7.3 Fuel — implementation state (the "Fuel" conformance column today)

| Piece | State |
|---|---|
| `fuel-core-types::dlpack` (abi, codes, sidecar, validate V1–V21, convert) | **Complete**, behind the `dlpack` feature (default-off) |
| `fuel-dispatch::fkc` importer (parse, schema, lower, register, validate) | **Complete**, behind the `fkc` feature; **not yet called from dispatch init** |
| DLPack/FDX view layer (`fuel-memory::dlpack_view::view` / `view_with_quant`) | **Complete**; **not yet wired at the kernel-call boundary** |
| `LinkRegistry` trait + test stubs | Trait done; **provider impls pending** |
| `extend_global_bindings` (Tier-1 runtime-extensible binding table) | **Complete** (`dispatch.rs:5098`) |
| Declarative pattern engine (`PatternKind::Declarative`) | **Stub** (returns `false`); the Tier-2 prerequisite |
| FKC contract corpus | ~68 bundle files / ~856 kernel sections, lint-clean |
| JIT-on-request + missing-fusion telemetry + `structure_key`/`DispatchRecord` emission | **Not built** (the base-emission seam is the prerequisite) |

#### 7.3.1 `OpAttrs` full first-order coverage (Convergence Increment A, 2026-07-16)

`OpAttrs` (`fuel-kernel-seam-types`) gained the dependency-free carriers the
full first-order re-emit vocabulary needs but the F1 set (`scalars`/`axis`/
`perm`/`target_shape`/`dims`) could not express. **Additive, optional →
backward-compatible; Fuel-led, Baracuda to mirror** (same convention as the F1
`perm`/`target_shape`/`dims` additions). Conforms to the pinned KISS §6.19
positional-blob grammar (the canonical byte serialization lands in Task 7 —
see the §6.19 schema table added there).

| Field | Type | Carries |
|---|---|---|
| `cast_dtype` | `Option<String>` | `Cast` target dtype as `DType::as_str()` (dep-free; mapped back via `FromStr`); also `MaskedFill`'s value dtype |
| `slice_start` / `slice_len` | `Option<u64>` | `Slice` window (its `dim` rides `axis`) |
| `roll_shift` | `Option<i64>` | `Roll` signed shift (its `dim` rides `axis`) |
| `pad_amounts` | `Vec<(u64, u64)>` | `Pad` per-axis `(before, after)` |
| `pad_mode` | `Option<u8>` | `Pad` mode code `0=Constant, 1=Reflect, 2=Replicate` (mirrors Fuel `PadMode` order) |
| `pad_value` | `Option<f64>` | `Pad` constant fill value |
| `keepdim` | `Option<bool>` | §6.19 reduce-schema conformance (serialized only; Fuel reduce Ops encode keepdim structurally) |

`Iota`'s `len` and `MaskedFill`'s value reuse existing fields (`target_shape` =
`[len]`; value on `scalars[0]`) — the `OpTag` disambiguates, as it already does
for `target_shape` serving both `BroadcastTo` and `Reshape`.

#### 7.3.2 `OpAttrs` §6.19 canonical positional-blob serialization (Convergence Increment A, 2026-07-16)

`OpAttrs::to_canonical_bytes(op) -> Vec<u8>` (`fuel-kernel-seam-types`) emits the
KISS **§6.19 canonical positional blob**: a per-op **positional** little-endian
body (no field names, **no elision** — the `OpTag` fixes the schema),
length-prefixed with a `u32` LE byte count. It is the canonical serialization
*onto* Fuel's internal `OpAttrs` struct, which stays a struct. Registered under
the recipe-import capability **`SEAM_CAP_RECIPE_IMPORT = FEAT bit 35`** (the KISS
FEAT range; see the co-design reply-2). std-only, dependency-free, deterministic.
**Baracuda to mirror the encoding.**

**Conformance scope (accurate claim, per the review of Increment A).** The blob
is **byte-comparable with a Baracuda-emitted one for the positionally-conformant
ops** — elementwise, `cast`, `slice`, `concat`, `roll`, `pad`, `flip`, `iota`,
`permute`, `(un)squeeze`, and the shape-target ops. It is **not yet fully
§6.19.3-conformant for `reduce`/`gather`/`scatter`**, and that is partly by
design under the pinned node schema `Op{op_name, op_attrs, child_edges}`:
- **`reduce{monoid, reduce_axes, keepdim}`** — Fuel emits single-axis
  `{axis, keepdim}`. `monoid` is carried by `op_name` (Fuel's distinct
  `SumDim`/`MaxDim`/`MinDim`/`ReduceSumTo`/`ReduceMaxTo` tags), so it does **not**
  belong in `op_attrs`. A **multi-axis `reduce_axes` list is DEFERRED** — Fuel
  currently models single-axis reduce (no consumer yet).
- **`gather{axis, oob_policy, index_operand, index_dtype}` /
  `scatter{axis, scatter_combine, oob_policy, index_operand, index_dtype}`** —
  Fuel emits `{axis}`. `scatter_combine` rides `op_name` (`IndexAdd` vs
  `ScatterAdd`), `index_operand` rides `child_edges`, and `index_dtype` rides
  that operand node — so those three legitimately are **not** `op_attrs` fields.
  **`oob_policy` is a genuine DEFERRAL** — a known-unwired slot with no carrier
  yet (tracked with the rest of the `oob_policy` seam work).

These deferrals are consumer-gated (Increment C and beyond), not Increment-A
scope; the §2.A conformance gap is closed for the conformant ops and the
remaining fields have a pinned home (`op_name`/`child_edges`) or a named
deferral (`oob_policy`, multi-axis `reduce_axes`).

Envelope: `u32` LE `body.len()` ++ `body`. **Empty-schema op** (`Add`, `Neg`,
`MatMul`, `Where`, comparisons, unary math, scalar reductions, `LogSoftmaxLastDim`,
…): `body` empty → the single canonical form `[0,0,0,0]`.

Per-op `body` field order (all little-endian; lists are `u32` count then elements):

| Op(s) | Positional body |
|---|---|
| `Reshape` / `BroadcastTo` / `ReduceSumTo` / `ReduceMaxTo` / `Iota` | `target_shape`: `i64` list (Iota = `[len]`) |
| `Permute` / `Transpose` | `perm`: `u32` list (absolute axis order) |
| `Unsqueeze` / `Squeeze` | `dims`: `u32` list |
| `Slice` | `axis: u32`, `start: u64`, `len: u64` |
| `Concat` / `Flip` / `Triu` / `Tril` / `IndexSelect` / `Gather` / `IndexAdd` / `ScatterAdd` | `axis: i64` |
| `Roll` | `axis: i64`, `shift: i64` |
| `SumDim` / `MeanDim` / `CumSum` | `axis: i64`, `keepdim: u8` |
| `Cast` | `dtype_name`: length-prefixed UTF-8 (`DType::as_str()`) |
| `Pad` | `amounts`: `u32` count then `(before:u64, after:u64)` pairs; `mode: u8`; `value: f64` |
| `AddScalar` / `MulScalar` / `Clamp` / `PowI` | `scalars`: `f64` list |
| `MaskedFill` | `scalars`: `f64` list; `dtype_name`: length-prefixed UTF-8 |
| all others (empty schema) | *(empty)* → `[0,0,0,0]` |

The `scan` / `scan_placeholder` / `runtime_scalar` tokens Fuel proposed are
higher-order / leaf ops handled outside this first-order `OpAttrs` set.

---

## 8. Ratification & implementation plan

> **Circulation rule (a party must never have to read Fuel's repo to review).** This contract pins specific
> versions of its annexes, so a recipient cannot ratify a profile that pins a spec version they have not
> seen. Therefore **circulation is self-contained**: the contract travels *with* the full text of every
> annex version it pins, in one bundle. The **Profile v1 circulation manifest** is:
> 1. this contract (`kernel-seam-interop.md`);
> 2. the FDX spec (`dlpack-extension.md`, FDX v1);
> 3. the FKC spec (`kernel-contract-format.md`, `fkc_version: 1`);
> 4. the FKC fusion-patterns spec (`fkc-fusion-patterns.md`, **rev 4** — the version §2 pins);
> 5. the rev-2 → rev-4 fusion-patterns delta (`baracuda-fusion-patterns-rev4-delta.md`) — so a reviewer who
>    last saw rev 2 can confirm their findings were resolved without diffing;
> 6. the role-specific cover note (`baracuda-seam-v1-roundtrip.md` / `vulkane-seam-v1-confirm.md`).
>
> (This is the lesson from the first round: Baracuda reviewed fusion-patterns **rev 2** and Profile v1 pins
> **rev 4** — they correctly refused to ratify a pinned version they had not seen. Bundling the annexes
> closes that gap permanently.)

1. **Circulate the Profile v1 bundle** (the manifest above) to Baracuda and Vulkane — self-contained, no
   references into Fuel's repo a reviewer would have to chase.
2. **Feedback round.** Baracuda reviews the full surface (esp. the handshake §3 and the JIT layer §5);
   Vulkane confirms the BDA subset + the light handshake. Fuel's in-flight vertical slice (a first kernel
   end-to-end through the seam, on a Fuel-*internal* kernel) runs *in parallel* and feeds any "this doesn't
   actually work" findings back into the contract — so ratification is implementation-true, not paper.
3. **Ratify Profile v1** when all three agree. Stamp it: *Profile v1, ratified `<date>`* = the §2 bundle.
   Retire the pre-ratification drafts.
4. **Implement in lockstep, now (not eventually):** Baracuda publishes a conforming crate version
   (`baracuda_seam_hello()`, the `link_registry` + FKC generator, `structure_key`); Vulkane confirms the
   **named BDA surface is present** (first exposed in `0.8.2`, per §7.2); **Fuel bumps the pins** — the same
   mechanism as the vulkane 0.8.2 BDA bump.
5. **Complete the Fuel side** (the integration points in §7.3: import-into-dispatch-init, the view at the
   kernel boundary, provider `LinkRegistry`s, the declarative engine for Tier-2, and the base-emission
   telemetry seam). The vertical slice from step 2 de-risks this before the 390-kernel fanout.

Once all three speak Profile v1, the seam is live and version-aware, and further development proceeds on a
ratified foundation.

---

## See also

- [dlpack-extension.md](dlpack-extension.md) — **FDX** (the normative tensor-interchange annex).
- [kernel-contract-format.md](kernel-contract-format.md) — **FKC** (the normative kernel-contract annex).
- [fkc-fusion-patterns.md](fkc-fusion-patterns.md) — the declarative `pattern:` grammar (Profile v1, FKC fusion-patterns rev 4).
- [10-decisions-log §2026-06-20](../architecture/10-decisions-log.md) — the adaptive-runtime-fusion decision (recipe principle, two-tier extensibility, the JIT loop) this contract operationalizes across the seam.
- `docs/session-prompts/baracuda-telemetry-plan.md` §9 — the missing-fusion / co-occurrence telemetry the JIT work-order feed rides on.
- `docs/outreach/{baracuda-seam-v1-roundtrip,vulkane-seam-v1-confirm}.md` — the circulation cover notes (drafts; the owner sends).

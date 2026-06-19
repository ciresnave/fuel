# Baracuda telemetry / miss-reporting emission — implementation plan

**Status: PLAN — not started. Branch: `feat/kernel-contracts-dlpack` (unmerged; `main` untouched).**
**Audience: a fresh instance executing this end-to-end, TDD, one crate at a time.**

This plan builds the **emission layer** for the Baracuda dispatch-telemetry / miss-reporting
feed. It is **not** a retention rebuild. The crucial grounding (verified this session) is that the
Judge **already retains** the per-candidate timings the feed needs:

- **Persistent JSON** — [`ProfileReport`](../../fuel-core-types/src/dispatch.rs) /
  [`ProfileEntry`](../../fuel-core-types/src/dispatch.rs) at `fuel-core-types/src/dispatch.rs:653-692`.
  One `ProfileEntry` **per measured alternative including losers**; fields verified:
  `op: OpKind`, `dtype: DType`, `size_class: SizeClass`, `backend: BackendId`,
  `device_index: u32`, `latency_ns: u64`, `iterations: u32`, `max_rel_error: f32`,
  `kernel_source: String` (dispatch.rs:656-682). Latencies are **`u64` nanoseconds**, not f32
  squares (corrects the stale memory).
- **In-memory oracle** — [`HashMapJudge`](../../fuel-dispatch/src/ranker/judge.rs) keyed
  `(OpKind, DType, SizeClass, BackendId, String)` → `u64` ns
  (`fuel-dispatch/src/ranker/judge.rs:73-75`), behind the
  [`JudgeOracle`](../../fuel-dispatch/src/ranker/judge.rs) trait
  (`judge.rs:41-61`, `measured_latency_ns`).
- **The adapter** — [`ProfileJudgeOracle::from_report`](../../fuel-core/src/judge/oracle.rs)
  (`fuel-core/src/judge/oracle.rs:65-92`) indexes **every** entry (losing alternatives included,
  module docs `oracle.rs:10-23`), keeps the min across multi-device duplicates, and is the
  per-candidate read surface this plan extends. The sibling-non-collision invariant is tested:
  `sibling_kernel_sources_do_not_collide` (`oracle.rs:170-195`).

**Therefore Baracuda Open-Q-1 ("per-(shape,impl) timings retained, or winner-only?") is answered
YES.** `candidates[]` is feasible by reading the oracle. What remains to build is: the
`DispatchRecord`/`MissRecord` JSONL writer, the `ImplId`/`StructureKey` join, the
best-admissible-match-is-generic miss signal, and the opt-in flag.

**Read-with (do not duplicate — point at these):**
- Outbound proposal: [`docs/outreach/baracuda-dlpack-fkc-ask.md`](../outreach/baracuda-dlpack-fkc-ask.md)
- FKC spec §4.11 (`ImplId` basis) / §4.12 (structural miss): [`docs/specs/kernel-contract-format.md`](../specs/kernel-contract-format.md)
- FDX spec §4.1 (operand description = `structure_key` input): [`docs/specs/dlpack-extension.md`](../specs/dlpack-extension.md)
- Inbound Baracuda ask (sibling repo): `baracuda/docs/fuel-ask-telemetry-2026-06-17.md` +
  companion `baracuda/docs/design/kernel-specialization.md` (canonical `structure_key`/`OperandKey`).
- Adjacent Fuel plans (reference, don't re-cover):
  [`dlpack-comm-layer-plan.md`](dlpack-comm-layer-plan.md),
  [`kernel-contract-adoption-plan.md`](kernel-contract-adoption-plan.md),
  [`internal-kernel-dlpack-conversion-plan.md`](internal-kernel-dlpack-conversion-plan.md),
  [`quantize-as-graph-op.md`](quantize-as-graph-op.md),
  [`symbolic-extents-and-persistent-decode.md`](symbolic-extents-and-persistent-decode.md).

---

## 0. Guardrails (every step obeys these)

- **NEVER run workspace-wide `cargo check`/`cargo test`.** `tensor-tools` has a standing
  `Device::Cpu` break and is a default-member — bare root `cargo check` fails. **Always
  `-p <crate>`.**
- **ONE cargo invocation at a time** (the build-dir lock serializes). Long builds → background + wait.
- **ONE live-GPU suite at a time** (RTX 4070, 12 GB — two concurrent live suites OOM). Telemetry
  emission has **no live-GPU test** in v1 (it reads a `ProfileReport`/oracle; both are populated
  from synthetic `ProfileEntry` fixtures in unit tests). Keep it that way.
- **TDD, born-red.** Write the failing test first, run it, **observe it go red**, then make it
  green. Record the red→green transition in the commit body.
- **Docs in the same change as behavior.** When this lands real emission, bump FKC §4.11 from
  `[consumer-ahead: deferred Baracuda telemetry feed]` to the as-built state and add a
  `docs/architecture/10-decisions-log.md` entry; update `docs/outreach/baracuda-dlpack-fkc-ask.md`
  §6 (move telemetry from DEFERRED → committed) and §4 Open-Q-1 (DEFERRED → answered YES).
- **WIP on the branch, not `main`.** Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Opt-in, off by default.** No record is ever written unless the flag is explicitly enabled
  (step 6). This matches FKC §1 non-goals (the static contract never carries live telemetry) and
  the outbound proposal §4 (privacy bullet).

**Crate placement decision (load-bearing for the whole plan):** the emission layer lives in a
**new module `fuel-dispatch/src/telemetry/`** behind a **new cargo feature `telemetry`** (mirror
of how `fkc` and `dlpack` are gated). Rationale: the `ImplId` basis fields all originate in
`fuel-dispatch` (`BindingEntry.kernel_source` at `kernel.rs:760`; `ResolvedPrimitive` carries
`op`/`dtypes`/`backend`/`kernel_source`/`revision` at `fkc/lower.rs:61-84`), the
`JudgeOracle`/`HashMapJudge` read surface is in `fuel-dispatch/src/ranker/`, and `fuel-dispatch`
**cannot** depend on `fuel-core` (cycle — `judge.rs:11-14`). The *file writer* that needs the
concrete `ProfileJudgeOracle` + the on-disk report path lives in **`fuel-core/src/telemetry.rs`**
(it already depends on `fuel-dispatch`), reusing the `default_report_path` sibling pattern
(`fuel-core/src/judge/mod.rs:161-164`). Split: **types + key derivation in `fuel-dispatch`;
JSONL sink + path resolution + oracle read in `fuel-core`.**

---

## 1. Step 1 — `DispatchRecord` / `MissRecord` JSONL schema (one record per line)

**Goal:** the wire structs that mirror Baracuda's `DispatchRecord` / `MissRecord` shapes, with a
serde JSONL encoding (one compact JSON object per line, newline-terminated — **not** a JSON
array, **not** pretty-printed; the persistent `ProfileReport` uses pretty JSON via
`to_vec_pretty` at `dispatch.rs:698`, but the telemetry feed is append-friendly JSONL so a long
run streams without rewriting).

**Files:**
- New: `fuel-dispatch/src/telemetry/mod.rs` (module root, `#[cfg(feature = "telemetry")]`).
- New: `fuel-dispatch/src/telemetry/record.rs` (the structs + serde).
- Edit: `fuel-dispatch/src/lib.rs` — `#[cfg(feature = "telemetry")] pub mod telemetry;`.
- Edit: `fuel-dispatch/Cargo.toml` — add `telemetry = ["serde", "dep:serde_json"]` feature
  (verify whether `serde_json` is already a dep of this crate; if not, add it dev+feature-gated).

**Schema (mirror the inbound ask's shapes; `ImplId` / `StructureKey` are opaque join tokens
defined in steps 2-3 and embedded here):**

```rust
/// One emitted dispatch decision. Mirrors Baracuda's `DispatchRecord`
/// (baracuda/docs/fuel-ask-telemetry-2026-06-17.md). Serialized as one
/// compact JSON object per line (JSONL).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DispatchRecord {
    /// Schema version of the telemetry wire format (NOT PROFILE_REPORT_VERSION).
    pub schema: u32,
    /// Baracuda's structure key for this dispatch site (step 3). Opaque
    /// string/u64 token Fuel obtains by CALLING Baracuda's structure_key;
    /// Fuel never derives it. `None` until the structure_key callable is
    /// linked (v1 may emit miss-key histograms without it — see step 4).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structure_key: Option<StructureKeyToken>,
    /// The implementation that won this dispatch (step 2).
    pub chosen: ImplId,
    /// Every admitted alternative + its measured latency, read from the
    /// Judge oracle (step 5). Empty when detailed mode is off (step 6).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
    /// Aggregated hit count for this (structure_key, chosen) cell since the
    /// last flush. Coarse mode emits counts only; detailed adds timings.
    pub count: u64,
}

/// One admitted alternative + its empirical latency (the loser rows).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Candidate {
    pub impl_id: ImplId,
    /// Median ns from the Judge oracle; `None` = unmeasured cell
    /// (oracle MISS — Layer-1 static estimate stood; never fabricate 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ns: Option<u64>,
}

/// A structural miss: the tightest admissible contract at this dispatch
/// key was a GENERIC one — a specialized cell would have fit but none is
/// registered. Mirrors Baracuda's `MissRecord`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MissRecord {
    pub schema: u32,
    /// The desired specialized cell (step 4). The MissRecord.wanted demand
    /// signal — what structure-specialized kernel would have fit here.
    pub wanted: StructureKeyToken,
    /// The generic contract the planner actually fell back to (step 2).
    pub fallback: ImplId,
    /// Aggregated count of this miss since last flush.
    pub count: u64,
    // NOTE: `est_speedup` is deliberately OMITTED in v1. Per the outbound
    // proposal §4 item 3, it is inferable from the fallback's DispatchRecord
    // rather than estimated at miss time; we drop the field rather than hold
    // the dataset for it.
}
```

**Failing test first** (`fuel-dispatch/src/telemetry/record.rs` `#[cfg(test)]`):

```rust
#[test]
fn dispatch_record_round_trips_as_one_jsonl_line() {
    let rec = DispatchRecord { /* fixture with chosen ImplId, 2 candidates */ };
    let line = serde_json::to_string(&rec).unwrap();      // compact, single line
    assert!(!line.contains('\n'), "JSONL record must be one line");
    let back: DispatchRecord = serde_json::from_str(&line).unwrap();
    assert_eq!(rec, back);
}

#[test]
fn miss_record_has_no_est_speedup_field() {
    let line = serde_json::to_string(&MissRecord { /* fixture */ }).unwrap();
    assert!(!line.contains("est_speedup"));
}
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::record` — red first (structs
  don't exist), then green.
- Done when both tests pass and `serde_json::to_string` produces a newline-free line.

**Where it is written:** the JSONL **sink** is `fuel-core` (step 7). This step defines only the
*records*; nothing writes them yet.

---

## 2. Step 2 — `ImplId` basis tuple → `{Baracuda|Vendor|FuelNative}` mapping (no new identifier)

**Goal:** the `ImplId` type whose **basis tuple is FKC kernel identity** — `(BackendId, op,
dtypes, kernel_source, kernel_revision_hash)` — projected onto Baracuda's discriminated union.
**No new identifier is invented** (FKC §4.11, kernel-contract-format.md:1191-1229). Every field
already exists:
- `kernel_source: &'static str` on `BindingEntry` (`fuel-dispatch/src/kernel.rs:760`) and as
  `String` on `ResolvedPrimitive`/`ResolvedFused` (`fkc/lower.rs:81,108`).
- `revision: KernelRevisionHash` on the same resolved structs (`fkc/lower.rs:83,110`); the
  newtype is `KernelRevisionHash(pub u64)` (`fuel-dispatch/src/fused.rs:250`).
- `op` / `dtypes` / `backend` are the binding-table key axes (`fkc/lower.rs:62-67`).

**Mapping (FKC §4.11, kernel-contract-format.md:1208-1214 — the discriminant is `kernel_source`,
no reconciliation table):**

```rust
/// The stable, pointer-free implementation id. Basis tuple = FKC kernel
/// identity (FKC §4.11); the discriminated form maps onto Baracuda's enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImplId {
    pub backend: BackendId,
    pub op: OpKind,                 // or a fused-op tag for fused contracts
    pub dtypes: Vec<DType>,         // inputs-in-order then outputs (KernelDTypes)
    pub kernel_source: String,      // the discriminant
    pub kernel_revision_hash: u64,  // KernelRevisionHash.0; pins the revision
}

/// Baracuda's wire form. `ImplId::classify` is the ONLY place the mapping
/// lives — co-defined with Baracuda, encoding frozen jointly (outbound §3.2).
pub enum ImplClass<'a> {
    Baracuda { symbol: &'a str },   // BackendId::Cuda + kernel_source == "baracuda"
    Vendor   { which: &'a str },    // "cublas" | "cudnn" | "mkl" | "aocl" | ...
    FuelNative { which: &'a str },  // portable CPU/native ("portable-cpu", "slang", ...)
}
```

- `BackendId::Cuda` + `kernel_source == "baracuda"` → `Baracuda { symbol }` (the `entry_point`
  IS the symbol — FKC §12.6, kernel-contract-format.md:1993-2001; in v1 the symbol may be the
  `kernel_source`-derived tag if the `entry_point` string isn't threaded to the dispatch site
  yet — **note this divergence to resolve**, see below).
- `kernel_source` in the vendor set → `Vendor { which }`.
- everything else → `FuelNative { which }`.

**DIVERGENCE NOTE (verified, must be resolved here):** the basis-tuple field
`kernel_revision_hash` is fully present on the **FKC-imported** path (`ResolvedPrimitive.revision`,
`fkc/lower.rs:83`), but `BindingEntry` (the *runtime* dispatch entry, `kernel.rs:743-760`) carries
**`kernel_source` but NOT a `revision` field** as of this writing. So an `ImplId` built at a live
dispatch site from a `BindingEntry` can supply `(backend, op, dtypes, kernel_source)` directly but
must source `kernel_revision_hash` from the importer-side `ResolvedPrimitive`, or default to
`KernelRevisionHash::UNTRACKED` (`= KernelRevisionHash(0)`, `fused.rs:255`) for non-FKC kernels.
**Step 2 must add the resolution: thread `revision` onto `BindingEntry` (preferred — append a
field, append-only register is already the pattern) OR carry an `ImplId`-keyed side table from the
FKC import.** Threading the field onto `BindingEntry` is the clean fix and unblocks v2 live
emission; do it here and update `register_full_with_source` (`kernel.rs:895`) + the FKC
`register_into` call site (`fkc/register.rs:208-217`) to pass `p.revision`.

**Files:**
- New: `fuel-dispatch/src/telemetry/impl_id.rs` (`ImplId`, `ImplClass`, `classify`,
  `from_binding`, `from_resolved_primitive`).
- Edit: `fuel-dispatch/src/kernel.rs` — add `pub kernel_revision_hash: u64` to `BindingEntry`
  (default `KernelRevisionHash::UNTRACKED.0`) + thread through `register_full_with_source`.
- Edit: `fuel-dispatch/src/fkc/register.rs` — pass `p.revision.0` at `register_into` (lines
  208-217).

**Failing test first:**

```rust
#[test]
fn baracuda_cuda_kernel_classifies_as_baracuda() {
    let id = ImplId { backend: BackendId::Cuda, op: OpKind::MatMul,
        dtypes: vec![DType::F16; 3], kernel_source: "baracuda".into(),
        kernel_revision_hash: 0xabc };
    assert!(matches!(id.classify(), ImplClass::Baracuda { .. }));
}
#[test]
fn cublas_classifies_as_vendor_and_revision_round_trips() {
    // kernel_source "cublas" -> Vendor; revision survives the basis tuple
}
#[test]
fn portable_cpu_classifies_as_fuel_native() { /* "portable-cpu" -> FuelNative */ }
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::impl_id` — red, then green.
- Done when all three classify cases pass AND `BindingEntry` carries a revision (verify the
  importer call site compiles: `cargo build -p fuel-dispatch --features fkc,telemetry`).

---

## 3. Step 3 — `StructureKey` join: Fuel CALLS Baracuda's `structure_key`, never reimplements it

**Goal:** the trampoline that obtains a `StructureKeyToken` for a dispatch site by **calling
Baracuda's shipped `structure_key(op_class, operands, arch) -> StructureKey`** with **FDX operand
descriptions** as input. Fuel **never reimplements** the key (FDX §4.1,
dlpack-extension.md:520-549; FKC §4.12 `[consumer-ahead]`, kernel-contract-format.md:1268-1273;
outbound §3.1).

**The boundary in v1:** Baracuda's callable is not yet linked into Fuel (it ships from the
`baracuda` crate / `baracuda-kernels-sys` FFI; cross-project, propose-first). So step 3 defines
**only the Fuel-side trait + the FDX operand-description adapter**, with a no-op/identity default
provider. The real callable is wired when Baracuda ships it (deferred, v2-adjacent).

```rust
/// Opaque token. Baracuda owns the encoding (string or u64); Fuel treats
/// it as bytes for the join. NEVER derived by Fuel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StructureKeyToken(pub String);

/// The seam to Baracuda's shipped structure_key. Fuel calls this; an impl
/// is provided BY Baracuda (FFI) or is the v1 no-op `None`.
pub trait StructureKeyProvider: Send + Sync {
    /// `operands` are FDX operand descriptions (the canonical input,
    /// FDX §4.1). Fuel builds them from (Storage, Layout[, SymEnv]) via the
    /// existing DlpackView comm-layer (fuel-memory/src/dlpack_view.rs).
    fn structure_key(&self, op_class: &str, operands: &[FdxOperandDesc], arch: &str)
        -> StructureKeyToken;
}
```

The **FDX operand description** is produced by the existing comm-layer
(`fuel-memory/src/dlpack_view.rs`, DlpackView slice 1 — borrowed `(Storage, Layout[, SymEnv])` →
`DLTensor` + sidecar). Step 3 adds a thin `FdxOperandDesc` projection (strides → contig class,
stride-0 → bcast, **stride sign → flipped**, dtype + sub-byte/quant) per the FDX §4.1 mapping
table (dlpack-extension.md:530). It does **not** add a `structure_key` field to FDX (P3 —
description, not decision; dlpack-extension.md:541-543).

**Files:**
- New: `fuel-dispatch/src/telemetry/structure_key.rs` (`StructureKeyToken`,
  `StructureKeyProvider`, `FdxOperandDesc`, the FDX-desc → provider-input adapter).
- The `FdxOperandDesc` projection reuses `fuel-memory`'s comm-layer; if `fuel-dispatch` doesn't
  depend on `fuel-memory`, define `FdxOperandDesc` as a plain data struct in `fuel-dispatch` and
  have the `fuel-core` writer (step 7) populate it from the live `DlpackView`. **Verify the dep
  direction before choosing.**

**Failing test first:**

```rust
#[test]
fn flipped_operand_sets_flipped_axis_in_fdx_desc() {
    // a Layout with a NEGATIVE inner stride -> FdxOperandDesc.flipped == true.
    // Load-bearing: negative-strides-first-class keeps this axis visible.
}
#[test]
fn fuel_never_derives_the_key_identity_provider_is_passthrough() {
    // With a recording provider, assert structure_key is CALLED with the
    // FDX operand descriptions, and Fuel returns the provider's token verbatim.
}
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::structure_key` — red, then green.
- Done when (a) a negative stride surfaces as `flipped == true` in `FdxOperandDesc`, and (b) the
  token is the provider's output verbatim (no Fuel-side derivation).

**Negative-strides note (load-bearing — outbound §3.4):** because Fuel reversed the
normalize-everything rule (2026-06-17, negative strides first-class), a flipped operand reaches
the dispatch site **as flipped**. `FdxOperandDesc.flipped` therefore tracks a **live demand
axis** — the one structure-key axis that is load-bearing *today* (`flipped` ↔ `reverse_strides`,
FKC §4.12, kernel-contract-format.md:1253). Had the old rule stood, every flip would be laundered
into a contiguous copy and `flipped` would be permanently `false`. Keep a test asserting the
flip survives to the descriptor.

---

## 4. Step 4 — the miss signal = "best admissible FKC match is a GENERIC contract" (no separate detector)

**Goal:** produce a `MissRecord` **without any bolt-on miss detector**, by reading the planner's
own contract-matching outcome. A **structural miss** is *definitionally* "at this dispatch key,
the tightest admissible contract is the **generic** one" (FKC §4.12,
kernel-contract-format.md:1261-1266). A structure-specialized kernel imports as a **tight-predicate
contract**; a generic strided kernel imports with `any`/floor predicates and is admissible
everywhere. The miss is exactly `MissRecord.wanted`.

**The mechanism:** at the matching site, the candidate set already carries each admitted
contract's predicate tightness (the §4.2 structure predicates: `inner_div`, `vec_width`,
`inner_contiguous`, `reverse_strides`). Step 4 adds a classifier:
`is_generic_contract(&Candidate) -> bool` (a contract whose admissibility is all `any`/floor /
`strided`+`scalar`+`Any`), and emits a `MissRecord` iff the **best admissible match is generic**
while the live operand structure (the `FdxOperandDesc` from step 3) would have keyed a *tighter*
cell. `wanted` is the structure key (step 3) of the live operands; `fallback` is the generic
contract's `ImplId` (step 2).

**Files:**
- New: `fuel-dispatch/src/telemetry/miss.rs` (`is_generic_contract`, `detect_miss(best_match,
  operands, provider) -> Option<MissRecord>`).
- Read-only against the candidate/matching surface: `fuel-dispatch/src/ranker/candidate.rs`
  (`Candidate.kernel_source` at `candidate.rs:89`) and the FKC predicate flags retained on
  `ResolvedPrimitive.layouts` (`fkc/lower.rs:71-73`, `ResolvedLayout` five-flag sets). **No new
  detection path** — read what matching already computed.

**Failing test first:**

```rust
#[test]
fn generic_only_match_emits_a_miss_record() {
    // candidate set = [generic strided contract]; live operands are
    // inner-contiguous + vec-able -> detect_miss returns Some, wanted = the
    // tight structure key, fallback = the generic ImplId.
}
#[test]
fn tight_specialized_match_emits_no_miss() {
    // a tight-predicate contract is admissible -> detect_miss returns None.
}
#[test]
fn flipped_operand_with_only_non_flip_kernels_is_a_miss() {
    // reverse_strides demand visible (negative-strides-first-class) ->
    // best match is generic-normalizing -> Some(MissRecord).
}
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::miss` — red, then green.
- Done when generic-only → `Some`, tight → `None`, and the flipped case surfaces a miss (proving
  the load-bearing axis flows end-to-end). **No `MissDetector` type exists** — the miss is read
  out of ordinary matching.

---

## 5. Step 5 — populate `candidates[]` from per-candidate Judge timings (now feasible)

**Goal:** fill `DispatchRecord.candidates[]` by reading the **per-candidate** measured latency
for each admitted `ImplId` out of the Judge oracle. This is the step the Judge-retention finding
unblocks: `ProfileJudgeOracle` indexes **every** measured alternative including losers
(`oracle.rs:10-23, 65-92`), so each candidate gets its **own** number, never the winner's.

**The read:** for each admitted candidate, call
`oracle.measured_latency_ns(op, dtype, size_class, backend, kernel_source)`
(`judge.rs:53-60`). `None` = unmeasured cell → `Candidate.latency_ns = None` (do **not**
fabricate 0; the sibling-miss invariant is already tested at `oracle.rs:191-194`).

**THE COVERAGE CAVEAT (must be in the doc + a test):** the Judge's *retention* is complete, but its
*populated coverage* is currently a bounded profiling matrix — **F32 only**, an offline-profiled
**square-matmul size ladder** (no GEMV / decode-shaped cells), a fixed primitive set, no online
exploration. So today non-F32 / GEMV / quantized cells **miss** the oracle and emit
`latency_ns: None` (never a fabricated 0). This is **transient**, not a retention gap: the Judge is
slated for extensive expansion (more dtypes — it will not be F32-only for long — plus judging every
op lacking a declared cost, and flash-vs-decomposed arm comparison). **Build this step
coverage-agnostic:** it reads whatever the oracle holds, so when the Judge's matrix grows, **no
telemetry code changes** — the cells simply start hitting. So in v1, `candidates[]` carries real
timings for the profiled (F32 square-matmul) cells and `None` for the rest, densifying over time.
**Assert the limitation in a test so it is visible, not silent**, and document — in the emitted
schema and in the outbound reply — that the feed starts sparse and fills in.

**Files:**
- New: `fuel-dispatch/src/telemetry/candidates.rs` (`fill_candidates(admitted: &[ImplId], oracle:
  &dyn JudgeOracle, dtype, size_class) -> Vec<Candidate>`). Note: `size_class` is derived via
  `SizeClass::from_elem_count` (`fuel-core-types/src/dispatch.rs:635-639`).

**Failing test first:**

```rust
#[test]
fn candidates_get_own_latency_loser_not_winner() {
    // HashMapJudge with two siblings at one cell, different latencies.
    // fill_candidates -> each Candidate carries ITS measured number.
}
#[test]
fn unmeasured_cell_yields_none_never_zero() {
    // oracle miss -> Candidate.latency_ns == None.
}
#[test]
fn non_f32_dtype_misses_today_f32_axis_only_caveat() {
    // F32 cell hits; F64 same op/size/backend/source -> None (documents the
    // current axis limitation; flips to a hit when the axis is extended).
}
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::candidates` — red, then green.
- Done when loser-keeps-own-number passes, unmeasured → `None`, and the F32-only caveat test
  passes (documenting the limit).

---

## 6. Step 6 — the opt-in flag (off by default; coarse vs detailed mode)

**Goal:** **no record is written unless explicitly enabled.** A single config controlling
`{ Off | Coarse | Detailed }`:
- `Off` (**default**) — emission disabled; zero overhead, no file opened.
- `Coarse` — `(structure_key, chosen)` + aggregated `count`; **no** `candidates[]` (no oracle
  reads). The miss-key histogram (step 4) alone — enough to start ranking Baracuda's build matrix
  per the inbound ask, and it does **not** depend on Judge timings at all.
- `Detailed` — Coarse **plus** `candidates[]` with timings (step 5).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TelemetryMode {
    #[default] Off,
    Coarse,
    Detailed,
}

#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub mode: TelemetryMode,          // default Off
    pub out_path: Option<PathBuf>,    // None -> default_telemetry_path() (step 7)
}
```

The flag is read once at plan/dispatch start (mirror `PlanOptions::with_judge`, exercised at
`oracle.rs:299`). Default-constructed config is `Off`, so a build that never sets it never emits.

**Files:**
- New: `fuel-dispatch/src/telemetry/config.rs` (`TelemetryMode`, `TelemetryConfig`).
- Wire point: a `with_telemetry(&TelemetryConfig)` on the existing `PlanOptions`
  (`fuel-dispatch/src/plan.rs` — verify the struct; `with_judge` is the sibling pattern). Edit
  `plan.rs` to carry an optional telemetry config; gate the field on `#[cfg(feature =
  "telemetry")]`.

**Failing test first:**

```rust
#[test]
fn default_mode_is_off_and_emits_nothing() {
    assert_eq!(TelemetryConfig::default().mode, TelemetryMode::Off);
    // a dispatch under Off produces zero records (assert the sink is never touched).
}
#[test]
fn coarse_omits_candidates_detailed_includes_them() {
    // build a DispatchRecord under each mode; Coarse -> candidates empty,
    // Detailed -> candidates populated.
}
```

**Build / done-check:**
- `cargo test -p fuel-dispatch --features telemetry telemetry::config` — red, then green.
- Done when default is `Off` with zero emission, and Coarse/Detailed differ exactly in
  `candidates[]`.

---

## 7. Step 7 — v1 batch/offline JSONL sink at release cadence (v2 live = forward-compat only)

**Goal:** the **writer** that drains accumulated records to a JSONL file at **release/run
cadence** (batch/offline), reusing the Judge persistence pattern. **v2 (live, per-dispatch
streaming) is forward-compat only — designed for, not built.**

**v1 is batch/offline** (outbound §4): a process accumulates aggregated `DispatchRecord` /
`MissRecord` counts in memory (keyed by `(structure_key, chosen)` and `wanted` respectively),
then flushes them to a JSONL artifact at process end / explicit `flush()` — mirroring how
`populate_dispatch_table` writes the `ProfileReport` once
(`fuel-core/src/judge/cache.rs:158-163` via `ProfileReport::save`, `dispatch.rs:697-706`,
atomic tmp+rename). The telemetry path uses the **same atomic write**, but **append-friendly
JSONL** (one record per line) rather than a single pretty JSON blob, so a long run can append
without rewriting.

**The sink lives in `fuel-core`** (it has the concrete `ProfileJudgeOracle` and the on-disk path;
`fuel-dispatch` cannot — cycle):
- New: `fuel-core/src/telemetry.rs`:
  - `default_telemetry_path() -> Option<PathBuf>` — sibling of
    `judge::default_report_path` (`fuel-core/src/judge/mod.rs:161-164`); place the JSONL next to
    the profile report (same hardware-keyed cache dir).
  - `TelemetrySink { records, misses }` — in-memory aggregation; `record_dispatch`,
    `record_miss`, `flush(&Path) -> Result<()>` (atomic tmp+rename, one JSON line per record).
  - `flush` reads the process-wide `ProfileJudgeOracle` (via `cache::cached()`,
    `cache.rs`) to fill `candidates[]` in Detailed mode (step 5).
- Edit: `fuel-core/Cargo.toml` — `telemetry = ["fuel-dispatch/telemetry"]`.

**v2 forward-compat note (design only, DO NOT build):** the JSONL-one-record-per-line shape is
*already* the live wire format — a v2 live emitter appends a line per dispatch instead of
aggregating. The `schema: u32` field (step 1) versions the line so v1 batch and v2 live are
distinguishable. **No v2 code in this plan**; the only obligation is that the v1 schema is a
strict subset a live emitter can extend (it is — counts collapse to 1, candidates stay the same).

**Files:**
- New: `fuel-core/src/telemetry.rs`.
- Edit: `fuel-core/src/lib.rs` — `#[cfg(feature = "telemetry")] pub mod telemetry;`.

**Failing test first** (`fuel-core/src/telemetry.rs` `#[cfg(test)]`):

```rust
#[test]
fn flush_writes_one_json_object_per_line() {
    let mut sink = TelemetrySink::new();
    sink.record_dispatch(/* rec A */);
    sink.record_dispatch(/* rec A again -> count aggregates to 2 */);
    sink.record_miss(/* miss */);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("telemetry.jsonl");
    sink.flush(&path).unwrap();
    let body = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<_> = body.lines().collect();
    // each line parses independently as a DispatchRecord or MissRecord
    for l in &lines { assert!(serde_json::from_str::<serde_json::Value>(l).is_ok()); }
    // aggregation: rec A appears once with count == 2
}
#[test]
fn detailed_mode_fills_candidates_from_oracle() {
    // build a sink from a ProfileReport fixture; Detailed flush -> candidates
    // carry per-sibling latencies (step 5), F32 cells hit, non-F32 None.
}
```

**Build / done-check:**
- `cargo test -p fuel-core --features telemetry telemetry` — red first (module absent), then
  green. (`tempfile` is likely already a dev-dep; verify.)
- Done when (a) the file is valid JSONL (each line parses standalone), (b) duplicate dispatches
  aggregate into one record with `count`, (c) Detailed flush fills `candidates[]` from the
  oracle. **No live-GPU test** — the sink reads a synthetic `ProfileReport`.

---

## 8. Closeout — docs in the same change

When step 7 is green, in the **same** branch:
- FKC spec §4.11 (`kernel-contract-format.md:1220-1229`): change the
  `[consumer-ahead: deferred Baracuda telemetry feed]` admonition to as-built; bump the spec
  version header; add a `docs/architecture/10-decisions-log.md` entry if the bump is MAJOR.
- FDX spec §4.1 `[consumer-ahead]` (`dlpack-extension.md:549`): note the `FdxOperandDesc`
  projection now exists (the call into Baracuda's `structure_key` is still gated on Baracuda
  shipping the callable — keep that half deferred).
- `docs/outreach/baracuda-dlpack-fkc-ask.md`: §6 move "the actual telemetry subsystem" from
  DEFERRED → committed; §4 Open-Q-1 DEFERRED → **answered YES** (`candidates[]` is fed from
  `ProfileJudgeOracle`, F32-axis caveat noted); §4 Open-Q-2 granularity → aggregated histograms
  (step 7); confirm the `est_speedup`-dropped decision (step 1).
- `ROADMAP.md`: advance the frontier (Baracuda telemetry emission shipped; formal reply now
  unblocked).
- **Cross-project (propose-first, do NOT edit):** the `ImplId` enum wire encoding and the
  `structure_key` FFI signature are co-defined with Baracuda. Draft the formal reply pointing at
  this as-built emission; do **not** touch the `baracuda` repo.

---

## Dependency order (one cargo invocation at a time)

1 (record) → 2 (`ImplId` + `BindingEntry` revision) → 3 (`structure_key` seam) → 4 (miss) →
5 (candidates) → 6 (flag) → 7 (sink in `fuel-core`) → 8 (docs). Steps 1-6 are all
`-p fuel-dispatch --features telemetry`; step 7 is `-p fuel-core --features telemetry`. Steps
3 and 5 are independent of each other and could swap, but keep the listed order so each
`DispatchRecord` field has its producer before the sink assembles it.

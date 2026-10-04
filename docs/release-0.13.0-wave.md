# 0.13.0 release wave — breaking and behavior-changing items

Tracks every breaking or behavior-changing item bundled into the next MAJOR
bump (pre-1.0, so the SECOND number per CireSnave's versioning rule: 0.12.x →
0.13.0). One line per item, linking its PR and/or gap row. Keep this current
as items land — it becomes the 0.13.0 release notes, not a separate writeup
done at release time.

Adding an item here does not promise it ships in 0.13.0; it records that an
item IS a 0.13.0-shaped change wherever one lands. Strike a line (`~~...~~`)
when its PR merges, rather than deleting it, so the final notes are a
filtered read of this same list.

## Items

- **arch.rs: a declared-but-unrecognized `general.architecture` now
  classifies as `Unknown`, never guessed from tensor names.**
  Behavior change (same signature, same enum). PR #303.
- **`fuel_formats::gguf::Value` gains a `Bytes(Vec<u8>)` variant; `read_string`
  stops its lossy UTF-8 conversion.** Breaking: `Value` is not
  `#[non_exhaustive]`, so a new variant breaks every exhaustive external
  match (the workspace-wide enum-variant gate applies). GAP-344.
- **`Content::open`/`Content::read` metadata parity:** four differences
  (duplicate-key first-vs-last, trailing-NUL stripping, `Bool` byte
  `2..=255` acceptance, non-UTF-8 key/name rejection) must be resolved or
  explicitly accepted before any real consumer is repointed from `read` to
  `open`. GAP-345.
- **`fuel-core` removed** (dissolution complete): every module relocated to
  its destination crate; the facade (`fuel`) keeps re-exporting the moved
  paths, but the `fuel-core` crate itself goes away. Board item 9.
- ~~**`Tensor` moves into a new `fuel-tensor` crate** (mechanical move per
  CireSnave's ruling on #109; the semantic split, if any, is a later,
  separate decision). Board #109, PR #305 (one PR for the whole
  mutually-referencing cluster: `lazy`/`device`/the 3 GPU bridges/`dtype`'s
  trait trio/`pipelined_bridge`/`judge`/`factories`/`planner`/
  `decode_shape`/`lazy_latent_cache`/`test_utils`/`scheduling`/
  `inference_context`/`kv_block_pool_device`/`persistent_decode`/`nf4` —
  a full crate::-reference census found they cannot split into the
  originally-planned smaller PRs without a dependency cycle).~~ Merged
  2026-10-03 (`3e3833a0`).
- ~~**Two duplicated items, each with a named single-home follow-up, from
  PR #305's move:** fuel-tensor carries its own local copy of
  `fuel_core::bail!` (`$crate::Error` resolves to whichever crate invokes
  it, and `fuel-tensor` depending on `fuel-core` for the macro would cycle
  with `fuel-core`'s own need for `fuel-tensor`) — single-home candidate:
  alongside `Error`/`Result` in `fuel_ir::error`. `metal_backend`'s
  `metal_is_available()` is also a local copy of
  `fuel_core::utils::metal_is_available` (the `cfg!(feature = "metal")`
  check must read whichever crate it's compiled into) — single-home
  candidate: wherever `utils.rs`'s own already-flagged
  accelerate/mkl/metal feature-check functions eventually land (that
  destination is itself still "Final home TBD" per `utils.rs`'s own doc,
  independent of this move).~~ Merged with #305 (2026-10-03,
  `3e3833a0`); the two duplicates themselves are NOT yet resolved to a
  single home — that follow-up is still open and un-ticketed (no GAP row
  yet), tracked only by this struck bullet until one is filed.
- **`fuel-kernel-seam::JitRequest` gains `pub target: TargetId` and becomes
  `#[non_exhaustive]`**, with a `JitRequest::new(...)` constructor; baracuda's
  5 construction sites move to `::new(...)` in the same release so they
  break once, not twice. Board #106 Step A.
- **`fuel-compression` renamed to `fuel-posttrain`** (mechanical: directory
  `git mv`, package name, the 2 repo sites). Not a model-definition crate
  (the `fuel-model-*` family is reserved for those); names the lifecycle
  stage it actually performs, pairing with `fuel-training`.
- **Facade path changes:** any `pub use` the above items add, move, or
  remove in `fuel/src/lib.rs` — tracked per-item above rather than
  separately; this bullet exists so a reader checking "did anything in the
  facade move" knows to read every item's own note, not search for a
  facade-specific list that doesn't exist.
- ~~**`fuel-inference`'s multi-session scheduler widens `eos_id: Option<u32>`
  to `eos_ids: Option<Vec<u32>>`** across `SessionState::new`,
  `SessionScheduler::add_session`, `PagedSessionScheduler::add_session`, and
  `PagedSessionScheduler::add_session_sharing_prefix` — a session now stops
  on ANY of several configured EOS ids (LLaMA-3-instruct-style checkpoints
  declare multiple, via `fuel-transformers`'s existing
  `LlamaEosToks::Single`/`Multiple`; the scheduler couldn't express that).
  Breaking: 4 public signatures change shape. `Some(vec![])` normalizes to
  `None` at construction, so "no EOS" has one representation. No shim —
  callers (lightbulb) adapt their call sites after this merges.~~ Merged
  2026-10-04 (PR #307). Lightbulb notified of the exact new signature.
- **PR-D (board #109 follow-up): `judge/cache.rs`'s storage/lookup half
  moves to `fuel_dispatch::judge_cache`** (`cached`, `cached_oracle`,
  `invalidate`, the process-wide slot) — the one Tensor-dependent function,
  `populate_dispatch_table`, stays as a thin fn in `fuel-tensor` that runs
  the Judge (needs `Tensor`) and hands the finished report down via a new
  `fuel_dispatch::judge_cache::store(report)`, avoiding the
  `fuel-dispatch → fuel-tensor` cycle a bare move would have recreated (see
  `docs/restructure-migration-design.md` §5.1 Row 3's amended note for why
  the original 2026-09-02 ruling undercounted this). Behavior-preserving —
  no public signature changes, same `cached()`/`cached_oracle()`/
  `populate_dispatch_table()` call sites. Not a breaking item on its own.
- **PR-D: `fuel-core/src/backend.rs` deleted** — a pure
  `pub use fuel_backend_contract::backend::HostStorage;` compat shim with
  zero consumers via `fuel_core::backend::`/`fuel::backend::` module path
  anywhere in the repo (verified with a positive-control grep: the same
  query finds `fuel-backend-contract`'s and `fuel-ir`'s own, unrelated,
  local `backend` modules, so a zero-hit-for-fuel-core result isn't a
  broken query). `utils.rs`'s `has_accelerate`/`has_mkl`/
  `metal_is_available` single-home follow-up (tracked since #305) is
  explicitly OUT of this item's scope — it needs a feature-flag semantics
  question answered first (does linking `fuel-metal-backend` alone imply
  availability, or does it need its own `metal` feature) that wasn't
  resolved here.

## Allocation

The PM allocates the actual version number at gate time (CireSnave's
ruling, 2026-09-23) — per-PR bumps do not compose across parallel PRs. This
doc does not carry a version bump itself.

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
- **`Tensor` moves into a new `fuel-tensor` crate** (mechanical move per
  CireSnave's ruling on #109; the semantic split, if any, is a later,
  separate decision). Board #109, PR #305 (one PR for the whole
  mutually-referencing cluster: `lazy`/`device`/the 3 GPU bridges/`dtype`'s
  trait trio/`pipelined_bridge`/`judge`/`factories`/`planner`/
  `decode_shape`/`lazy_latent_cache`/`test_utils`/`scheduling`/
  `inference_context`/`kv_block_pool_device`/`persistent_decode`/`nf4` —
  a full crate::-reference census found they cannot split into the
  originally-planned smaller PRs without a dependency cycle).
- **Two duplicated items, each with a named single-home follow-up, from
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
  independent of this move).
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

## Allocation

The PM allocates the actual version number at gate time (CireSnave's
ruling, 2026-09-23) — per-PR bumps do not compose across parallel PRs. This
doc does not carry a version bump itself.

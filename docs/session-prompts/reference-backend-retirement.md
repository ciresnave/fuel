# Session prompt — Reference backend retirement

## What this session is for

Retire `fuel-reference-backend` as a privileged oracle. Replace
its role in Judge correctness comparison with pairwise cross-
backend consensus + a distributable captured-output fixture file.
Delete the crate, the `BackendId::Reference` enum variant, the
`realize_f32_reference()` API, and the ReferenceFactory.

Architecture v0.2 of [`docs/architecture/05-backend-contract.md`](../architecture/05-backend-contract.md)
(2026-05-09) already demoted Reference architecturally:

> "The fuel-reference-backend crate, where it exists today, becomes
> 'the backend whose entire kernel set has `bit_stable_on_same_hardware: true`.'
> ... architecturally its role is no longer special. Its kernels
> could equivalently live as `bit_stable`-tagged kernels inside
> fuel-cpu-backend; the choice is implementation convenience, not
> architectural commitment."

The Judge code never followed. This session aligns code with
architectural intent.

## Why now

1. **Maintenance burden is real and growing.** Every new IR op
   (most recent: `Op::Contiguize` in commit `6a860777`) needs a
   Reference impl that does nothing but mirror Reshape. Pure
   duplicate code added for match-exhaustiveness, zero new
   validation value.

2. **Reference's bit-stability is per-hardware anyway.** Its
   kernels are textbook scalar loops but IEEE rounding on different
   CPUs gives different bits. "Reference produces THE answer" was
   always a small lie — it produces "this CPU's answer with
   textbook math."

3. **fuel-cpu-backend already commits to bit-stable coverage** for
   every primitive op (the v0.2 "always-built coverage commitment"
   in §The always-built coverage commitment). So the bit-stable
   oracle exists redundantly — once in Reference, once in
   fuel-cpu-backend.

4. **The PrecisionGuarantee framework already does the heavy
   lifting** Reference used to do. Kernels declare their precision
   properties; the optimizer reasons about them; calibration
   tooling picks comparators from `bit_stable_on_same_hardware: true`
   kernels — which can live on any backend, not just Reference.

5. **Pairwise consensus is more honest.** When CPU + AOCL + MKL +
   CUDA all produce numerically close outputs for matmul, that's
   stronger evidence than "matches Reference." When they disagree,
   the disagreement names exactly which cells drift, which is what
   the Judge cares about.

## Architectural shape

### Pairwise consensus correctness logic

For each `(op, dtype, size)` cell during Judge profiling:

1. Run every backend's kernel that registered for that cell.
2. Collect outputs `O_1, O_2, ..., O_N`.
3. Cluster by mutual `rel_err < epsilon` (epsilon from the
   `PrecisionGuarantee::max_relative` of the strictest-tier kernel
   among the N, falling back to `1e-5` for unspecified).
4. Largest cluster = consensus group. If ties (e.g. 2-vs-2 with
   N=4), the cluster containing the highest-tier-precision kernel
   wins.
5. Outliers (kernels outside the consensus cluster) flagged in the
   ProfileReport for human review.
6. After human review, the consensus group's median output (or any
   representative member) becomes the captured ground truth for
   that cell.

### Distributable correctness fixtures

`tools/fuel-capture-fixtures` binary that:

1. Walks every `(op, dtype, size_class)` cell in Judge's profile
   scope.
2. Generates a deterministic input (seeded RNG; the seed becomes
   part of the fixture key).
3. Runs all locally-available backends' kernels for that cell.
4. Applies pairwise consensus to identify the agreed-upon output.
5. Writes `(op_id, dtype, size_class, seed, input_hash, expected_output_bytes, tolerance)` tuples to a fixture file.

Fixture file format:

```text
fuel-correctness-fixtures/
  v1/
    f32/
      matmul/
        size_64.bin      # (input_hash, expected_output, tolerance) for size_class 64
        size_128.bin
        ...
      add/
      ...
    f16/
    ...
```

Distributed with Fuel via `fuel-correctness-fixtures` crate;
shipped as a package-internal data file (`include_bytes!` or
read at runtime from a known path).

### Judge consumption

Subsequent Judge runs on new systems / new backends:

1. Locate the matching fixture for `(op, dtype, size_class, seed)`.
2. Run the local backend's kernel with the fixture's input.
3. Assert `rel_err(output, expected_output) < tolerance`.
4. No inter-backend comparison needed if the fixture is available.
5. When no fixture exists (new op, new size class), fall back to
   inline pairwise consensus across locally-available backends.

### Tolerance bands (not bit-exact)

Captured outputs include a tolerance band, not a single bit-exact
expected value. The band is derived from:

- The strictest `PrecisionGuarantee::max_relative` among the
  consensus group's kernels.
- Plus a `2x` safety margin to absorb platform rounding drift
  (ARM vs x86 vs Apple Silicon).

Same fixture file works everywhere; the tolerance band absorbs
platform rounding. Kernels that exceed the tolerance are flagged
as drift.

## Scope

| Step | Where | Effort |
|------|-------|--------|
| 1. Define fixture file format + crate scaffolding | `fuel-correctness-fixtures` (new crate) | small |
| 2. Capture tool — multi-backend pairwise consensus + fixture writer | `tools/fuel-capture-fixtures` (new binary) | medium |
| 3. Initial fixture capture on dev box (Windows + RTX 4070) | run + human review of outliers | medium (human review is the bottleneck) |
| 4. Judge consumes fixtures instead of Reference | `fuel-core::judge::mod` | medium |
| 5. Remove `realize_f32_reference()` | `fuel-core::lazy` | trivial |
| 6. Remove `ReferenceFactory` from `factories.rs` registry | `fuel-core::factories` | trivial |
| 7. Remove `BackendId::Reference` enum variant + all consumers | `fuel-core-types::probe` + `fuel-dispatch` + Judge cache | medium (~10 sites) |
| 8. Delete `fuel-reference-backend` crate | workspace + dep stripping | trivial |
| 9. Update tests + documentation | various | small |

## Step 1 — Fixture file format

Crate `fuel-correctness-fixtures` (new) with:

```rust
pub struct CorrectnessFixture {
    pub op: OpKind,
    pub dtype: DType,
    pub size_class: SizeClass,
    pub input_seed: u64,
    pub input_hash: u64,  // BLAKE3 of input bytes, for sanity
    pub expected_output: Vec<u8>,  // raw bytes
    pub tolerance: ToleranceBand,
}

pub struct ToleranceBand {
    pub max_relative: f64,
    pub max_absolute: f64,
}

pub fn load_fixture(
    op: OpKind, dtype: DType, size_class: SizeClass, seed: u64,
) -> Option<&'static CorrectnessFixture>;

pub fn validate_against_fixture(
    fixture: &CorrectnessFixture,
    actual_output: &[u8],
) -> Result<(), CorrectnessDrift>;
```

Fixtures distributed as `include_bytes!`-baked data files inside
the crate (or read from `$FUEL_FIXTURES_DIR` at runtime for
development workflows that update fixtures often).

## Step 2 — Capture tool

`cargo run --bin fuel-capture-fixtures` binary that walks
Judge's profile scope, runs all available backends, applies
pairwise consensus, surfaces outliers for human review, writes
the fixture files.

CLI:

```
fuel-capture-fixtures
  --op all                     # or specific op
  --dtype all                  # or specific dtype
  --size-class all             # or specific size
  --backends cpu,cuda,vulkan   # which to compare (must be ≥2)
  --tolerance-mode strict|generous
  --output fuel-correctness-fixtures/v1/
```

For each cell:

1. Generate input from seed.
2. Run every backend's kernel that registered for the cell.
3. Cluster outputs by `rel_err < epsilon`.
4. If consensus group ≥ ⌈N/2⌉, write fixture with consensus output
   + tolerance band.
5. If no consensus, write a report file listing the outputs +
   flag for human review.

## Step 3 — Initial capture

Run the capture tool on the dev box (Windows + RTX 4070; backends:
CPU portable + AOCL [if installed] + CUDA via baracuda + Vulkan
via vulkane). Review any outlier reports. Hand-check that the
consensus group's output is numerically correct for at least a
few representative cells (matmul, RMS norm, softmax) before
trusting the bulk capture.

## Step 4 — Judge consumes fixtures

Today's Judge ([`fuel-core/src/judge/mod.rs:391`](../../fuel-core/src/judge/mod.rs#L391)):

```rust
let reference_out = tensor.realize_f32_reference();
let backend_out = realizer.realize_f32(&tensor);
let rel_err = max_rel_err(&backend_out, &reference_out);
```

Becomes:

```rust
let fixture = fuel_correctness_fixtures::load_fixture(op, dtype, size_class, seed);
let backend_out = realizer.realize_f32(&tensor);
let rel_err = match fixture {
    Some(f) => max_rel_err_vs_fixture(&backend_out, &f.expected_output),
    None => {
        // No fixture — fall back to pairwise consensus inline
        // (slower but covers new ops before fixtures get captured)
        max_rel_err_via_pairwise_consensus(op, dtype, size, &backend_out)
    }
};
```

## Step 5 — Remove `realize_f32_reference()`

Trivial: delete the method from `fuel-core/src/lazy.rs:1332-1334`.
Update any callers (search shows just judge/mod.rs:391 — already
addressed in step 4).

## Step 6 — Remove `ReferenceFactory`

Delete from `fuel-core/src/factories.rs:105-123` + the `&ReferenceFactory`
in `registry()`'s `v.push` list.

## Step 7 — Remove `BackendId::Reference`

The enum variant lives at `fuel-core-types/src/probe.rs:67`.
Consumers visible from `grep -rn 'BackendId::Reference' --include='*.rs'`:

- `fuel-core/src/judge/cache.rs:185, 189, 221, 225` — test fixtures
  for cache priority; rewrite to use another backend.
- `fuel-core/src/judge/mod.rs:838-895` — Reference-disagreement
  tests; rewrite for pairwise consensus.
- `fuel-dispatch/src/plan.rs:847` — test fixture; remove Reference
  from the iteration list.
- Probe's `as_str`, `as_str_lower` match arms.

Strip the variant.

## Step 8 — Delete `fuel-reference-backend` crate

After step 7, nothing references it. Remove from:

- `Cargo.toml` workspace members (lines 39, 68, 137 per grep).
- `fuel-core/Cargo.toml` dependency list.
- `fuel-aocl-cpu-backend/Cargo.toml` (if listed; unlikely).
- Delete the `fuel-reference-backend/` directory.

## Step 9 — Documentation update

- `docs/architecture/05-backend-contract.md` — bump to v0.4 noting
  Reference retirement complete; replace "may continue to exist
  for clarity" language with "retired in commit XYZ; pairwise
  consensus + captured fixtures replace it."
- `docs/architecture/07-tolerance.md` — update §Calibration to
  describe the fixture-based path.
- README or top-level docs — strip any mention of "reference
  backend" as a user-facing concept.

## Risks

- **Initial fixture capture quality.** Captured outputs become the
  ground truth for all future runs. A bug at capture time
  propagates. Mitigation: human review of outlier reports + spot
  checks on representative cells before merging the fixture file.
- **Cross-platform drift.** A fixture captured on x86_64 may not
  bit-match on aarch64. The tolerance band absorbs this, but the
  band's size matters. Document the rationale in the fixture
  format header.
- **Bootstrapping new ops.** When a new op lands without fixtures,
  pairwise consensus needs ≥2 backends locally to validate. On
  single-backend dev boxes this is a blocker. Mitigation: Judge
  reports "no consensus possible, single backend" as a warning,
  not an error.
- **Diverged-Reference legacy.** Existing Judge profile reports
  on disk have Reference as a column. Migration tool needed to
  strip / archive them.

## Concrete deliverables

1. New crate `fuel-correctness-fixtures` with format + loader API.
2. New binary `fuel-capture-fixtures` for the consensus + capture
   workflow.
3. Initial fixture set (`fuel-correctness-fixtures/v1/`) captured
   on the dev box.
4. Judge migrated to fixtures + inline-consensus fallback.
5. Commits retiring `realize_f32_reference`, `ReferenceFactory`,
   `BackendId::Reference`, and the `fuel-reference-backend` crate.
6. Doc update to v0.4 of `05-backend-contract.md`.

## Coordination with backend-extensions Phase 2

Both this session and [`backend-extensions-phase-2.md`](backend-extensions-phase-2.md)
require Judge refactor work (alternative-walk + per-kernel-source
measurement). The two sessions naturally compose:

- Do this session's Judge changes first (consumes fixtures).
- Then backend-extensions Phase 2's Judge changes (walks
  alternatives) on top.

Or combine into one large Judge-refactor commit if scoping allows.

## Scope estimate

~2 focused sessions:

- Session A: fixture format + capture tool + initial capture +
  Judge consumes fixtures.
- Session B: enum removal + crate deletion + test migration +
  doc update.

If combined with backend-extensions Phase 2's Judge work, ~3
sessions total.

## Pointers

- Architecture v0.3: [`docs/architecture/05-backend-contract.md`](../architecture/05-backend-contract.md)
- Reference's current role: [fuel-reference-backend/src/lib.rs](../../fuel-reference-backend/src/lib.rs)
- Judge today: [fuel-core/src/judge/mod.rs](../../fuel-core/src/judge/mod.rs)
- LazyTensor::realize_f32_reference: [fuel-core/src/lazy.rs:1332](../../fuel-core/src/lazy.rs#L1332)
- BackendId enum: [fuel-core-types/src/probe.rs:63](../../fuel-core-types/src/probe.rs#L63)
- Related: [`backend-extensions-phase-2.md`](backend-extensions-phase-2.md)

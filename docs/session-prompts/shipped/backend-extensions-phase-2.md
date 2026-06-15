# Session prompt — Backend-extensions Phase 2: BackendId::Aocl/Mkl retirement

## What this session is for

Complete the architectural retirement of `BackendId::Aocl` and
`BackendId::Mkl` as distinct backends. Phase 1 (commit `c68dc212`,
2026-06-07) added the `kernel_source: &'static str` field to
`BindingEntry` and `Candidate`, and migrated the AOCL + MKL crates
to register their kernels under `BackendId::Cpu` with `"aocl"` /
`"mkl"` source tags. That left the architectural change complete at
the kernel-dispatch layer but with three vestigial scaffolds still
in place:

1. `realize_f32_aocl` / `realize_f32_mkl` in `fuel-core::lazy` —
   delegation stubs that just call `realize_f32`.
2. `AoclFactory` / `MklFactory` in `fuel-core::factories` —
   load-bearing for the Judge today (Judge uses
   `factories::factory_for(backend)` to make a realizer per
   discovered device).
3. `BackendId::Aocl` / `BackendId::Mkl` enum variants in
   `fuel-core-types::probe` — surfaced by AOCL / MKL probes;
   tested by `tests/phase7b_*_anchor.rs` files; named in
   `fuel-dispatch` test fixtures (cast_fusion, plan, ranker/cost).

This session ships the retirement in one focused pass.

## Scope

| Step | Where | Effort |
|------|-------|--------|
| 1. Judge: walk alternatives, not factories | `fuel-core::judge::mod` | medium |
| 2. AOCL + MKL probes report `BackendId::Cpu` with source descriptor | `fuel-aocl-cpu-backend::probe`, `fuel-mkl-cpu-backend::probe`, `fuel-core-types::DeviceDescriptor` | small (add `kernel_source: Option<&'static str>` to `DeviceDescriptor`) |
| 3. Remove `realize_f32_aocl` / `realize_f32_mkl` | `fuel-core::lazy` | trivial |
| 4. Remove `AoclFactory` / `MklFactory` | `fuel-core::factories` | small |
| 5. Update `phase7b_*` integration tests | `fuel-core::tests::phase7b_*` | small (replace with kernel_source-aware checks) |
| 6. Update `fuel-dispatch` test fixtures | `cast_fusion.rs`, `plan.rs`, `ranker/cost.rs`, `ranker/enumerate.rs`, `ranker/runtime_selector.rs` | medium (~10 sites) |
| 7. Remove `BackendId::Aocl` + `BackendId::Mkl` variants | `fuel-core-types::probe` | trivial after the above |
| 8. Update `fuel-graph-router` consumers | `fuel-graph-router::lib`, `tests::dispatch_routing*` | medium |
| 9. Update scheduling + topology consumers | `fuel-core::scheduling`, `fuel-core::topology` | small |

## Step 1 — Judge walks alternatives

Today's [`fuel-core/src/judge/mod.rs:333`](../../fuel-core/src/judge/mod.rs#L333):

```rust
let factory = match crate::factories::factory_for(device.backend) {
    Some(f) => f,
    None => return None,  // backend not compiled in
};
let mut realizer = factory.try_make_realizer(device.device_index)?;
```

The factory step is what tied Judge to `BackendId::Aocl/Mkl` —
it looked up an AOCL-or-MKL-specific factory. With kernel-source
tagging, the alternative-walk is the right shape: for each binding
table entry at the cell being profiled, time it directly via its
`KernelRef`.

Shape proposal:

```rust
fn profile_cell(
    &self,
    op: OpKind,
    dtype: DType,
    size: &OpSize,
    device: &DeviceDescriptor,
) -> Vec<ProfileEntry> {
    let bindings = global_bindings();
    let dtypes = canonical_dtypes_for(op, dtype);
    let key = (op, KernelDTypes::from_slice(&dtypes), device.backend);
    let alternatives = bindings.lookup_alternatives(...);

    let mut out = Vec::with_capacity(alternatives.len());
    for entry in alternatives {
        // entry.kernel + entry.kernel_source — measure each
        let timed = time_kernel_directly(entry, ...);
        out.push(ProfileEntry {
            op, dtype, size_class,
            backend: device.backend,
            kernel_source: entry.kernel_source,  // NEW field
            median_ns: timed.median,
            max_rel_error: timed.rel_err_vs_oracle,
            ...
        });
    }
    out
}
```

`ProfileEntry` gains a `kernel_source: &'static str` field. The
Judge cache's lookup key extends from `(op, dtype, size, backend)`
to `(op, dtype, size, backend, kernel_source)` so Judge can answer
"how fast is the AOCL matmul" vs "how fast is the portable matmul"
even though they share `BackendId::Cpu`.

## Step 2 — Probes consolidate under BackendId::Cpu

`DeviceDescriptor` gets a new optional field:

```rust
pub struct DeviceDescriptor {
    pub backend: BackendId,
    pub device_index: u32,
    pub hardware_sku: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: String,
    pub total_memory_bytes: u64,
    pub location: DeviceLocation,
    // NEW: identifies the kernel-source extension this descriptor
    // refers to. `None` for primary backends; `Some("aocl")` /
    // `Some("mkl")` for kernel-library extensions on a CPU substrate.
    pub kernel_source: Option<&'static str>,
}
```

AOCL probe returns:

```rust
DeviceDescriptor {
    backend: BackendId::Cpu,             // was Aocl
    device_index: 0,
    kernel_source: Some("aocl"),         // NEW
    hardware_sku: "...".to_string(),
    ...
}
```

`ProbeReport::collect_all()` now produces multiple descriptors
with the same `(backend, device_index) = (Cpu, 0)` but different
`kernel_source` values — one entry per (substrate × kernel-source)
combination that's loadable on the host. The Judge then iterates
these descriptors and measures each kernel-source individually.

## Step 3 — Remove `realize_f32_aocl` / `realize_f32_mkl`

These are delegation stubs at `fuel-core/src/lazy.rs:1387-1403`.
Just delete them. Update the doc comments on `realize_f32` to
mention: "When AOCL or MKL extensions are loaded, the picker
selects among them as alternatives to the portable CPU kernels."

## Step 4 — Remove `AoclFactory` / `MklFactory`

After step 1, Judge no longer depends on these factories. After
step 3, nothing else does. Delete from `fuel-core/src/factories.rs`,
remove from `registry()`'s `v.push` list.

## Step 5 — Migrate phase7b integration tests

`fuel-core/tests/phase7b_aocl_anchor.rs`,
`phase7b_mkl_anchor.rs`, `phase7b_conv2d_oracle.rs` test the
behavioral intent "AOCL/MKL kernels produce correct output and the
dispatch table prefers them on appropriate hardware." After
retirement, the intent translates to:

- "AOCL/MKL kernels register and appear in the binding table at
  `(MatMul, [F32x3], Cpu)`."
- "The Judge report distinguishes their measurements via the
  `kernel_source` field."
- "On Zen hardware, AOCL's median latency is below the portable
  kernel's, and the picker's cost ranking surfaces AOCL as winner."

Rewrite each test to query the binding table directly (via
`global_bindings().lookup_alternatives(...)` filtered on
`kernel_source == Some("aocl")`) instead of calling
`realize_f32_aocl`.

## Step 6 — Update fuel-dispatch test fixtures

`fuel-dispatch/src/cast_fusion.rs:75-76`, `plan.rs` (many sites),
`ranker/cost.rs`, `ranker/enumerate.rs`, `ranker/runtime_selector.rs`
all reference `BackendId::Aocl` / `BackendId::Mkl` in test code.
Replace with `BackendId::Cpu` + the `kernel_source` discriminator.
Each test that needed two distinct backends to test
"cross-backend alternatives" now uses two distinct kernel-source
strings under the same backend.

## Step 7 — Remove `BackendId::Aocl` + `BackendId::Mkl` enum variants

After steps 1–6, no consumer references the variants. Strip them
from `fuel-core-types/src/probe.rs:79-81`. Update `as_str`,
`from_str`-like helpers if any. The enum's `#[non_exhaustive]`
attribute means downstream users that matched on these variants
won't break (they had to handle `_ => ...` already).

## Step 8 — fuel-graph-router consumers

`fuel-graph-router/src/lib.rs` + `tests/dispatch_routing.rs` +
`tests/dispatch_routing_mkl.rs` reference the variants for routing
fixtures. Update to use `BackendId::Cpu` + kernel_source.

## Step 9 — scheduling + topology

`fuel-core/src/scheduling.rs` and `fuel-core/src/topology.rs`
reference the variants for backend enumeration. Update to query
the binding table for available kernel sources instead.

## Risks

- **Probe identity collision**: with the kernel_source field on
  DeviceDescriptor, two descriptors can share
  `(backend, device_index) = (Cpu, 0)` while having different
  kernel_source. Audit `ProbeReport`'s deduplication logic + the
  Judge cache key shape.
- **Test ordering**: phase7b tests were written assuming AoclBackend
  was a distinct type. After retirement, they assert behavioral
  intent (alternatives present + ranking correct) rather than
  type identity. Verify on a host with AOCL installed.
- **Behavioral change on AOCL machines**: today's
  `realize_f32_aocl(&mut exe)` always dispatched via the AoclBackend
  trait dispatch (which bypasses the picker). After retirement,
  the same workload goes through the standard `realize_f32` path
  → picker → AOCL kernel (when available). This SHOULD be a
  performance no-op (same kernel runs either way), but verify on
  Zen hardware.

## Out of scope (separate session)

- **Reference retirement** — separate concern, separate session.
  Architecture v0.3 already demoted Reference; the code retirement
  is parallel work on a different axis.
- **Vendor BLAS Tier 2 expansion** — int GEMM, quantized GEMM
  follow-ups for AOCL/MKL. Independent of contract retirement.

## Test sweep targets

- `cargo test -p fuel-dispatch --lib` — must hold steady (251/252
  pass currently with pre-existing FlashAttnBackwardV gap).
- `cargo test -p fuel-core --lib` — lazy + judge tests pass on
  default features.
- `cargo test -p fuel-core --features aocl` — phase7b_aocl_anchor
  rewritten to assert binding-table presence + dispatch preference.
- `cargo test -p fuel-core --features onemkl` — same for MKL.
- `cargo test -p fuel-graph-router --lib --tests` — routing tests
  use kernel_source.

## Scope estimate

~1 focused session if no probe-collision surprises in step 2; ~2
sessions if the Judge cache key shape change cascades into
profile-data file format changes.

## Pointers

- Phase 1 commit: `c68dc212`
- Backend contract: [`docs/architecture/05-backend-contract.md`](../architecture/05-backend-contract.md) §Trait surface
- Related: [`docs/session-prompts/reference-backend-retirement.md`](reference-backend-retirement.md) (separate axis, may be authored in parallel)

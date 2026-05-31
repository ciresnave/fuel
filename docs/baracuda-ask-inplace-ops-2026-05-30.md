# Baracuda ask — in-place op family on CUDA (2026-05-30)

Single coordinated ask for the in-place ops infrastructure shipped
in Fuel commits `dd2b7158`..`7a158a10` (Phases 1-5) plus the
dtype-expansion follow-up landing in this session. All paths absolute
on the Fuel author's machine; the Fuel repo's root is at
`c:/Users/cires/OneDrive/Documents/projects/fuel/`.

Background docs:

- `c:/Users/cires/OneDrive/Documents/projects/fuel/docs/session-prompts/in-place-ops-infrastructure.md`
  — the design memo behind the infrastructure.
- `c:/Users/cires/OneDrive/Documents/projects/fuel/docs/session-prompts/inplace-unary-dtype-expansion.md`
  — the dtype-coverage session.

The architectural intent on Fuel's side: in-place ops are a
first-class IR family (one `Op::*Inplace` variant per kind, plus
`Op::Fused(INPLACE_AFFINE, _)` for the scalar-affine form). The
executor's `WorkItemKind::InplaceKernel` arm adopts the target
Storage Arc as `outputs[0]` with `inputs=[]`; the dispatch wrapper
acquires the write lock and hands the kernel a single buffer (or
same-pointer dispatch over the forward kernel where the ABI allows
it). v1 contract: target is always contiguous + zero-offset (the
executor rejects strided targets up front), so there's no strided
in-place variant to ship.

Fuel currently covers in-place ops as follows (post 2026-05-30
dtype expansion):

| Op family                        | CPU                  | CUDA (baracuda alpha.60)                                |
|----------------------------------|----------------------|---------------------------------------------------------|
| `Op::ReluInplace`                | f32/f64/bf16/f16     | f32/f64/bf16/f16 (via forward `unary_relu_*_run` reuse) |
| `Op::SiluInplace`                | f32/f64/bf16/f16     | f32/f64/bf16/f16 (via forward `unary_silu_*_run` reuse) |
| `Op::GeluInplace`                | f32/f64/bf16/f16     | f32/f64/bf16/f16 (via forward `unary_gelu_*_run` reuse) |
| `Op::TanhInplace`                | f32/f64/bf16/f16     | f32/f64/bf16/f16 (via forward `unary_tanh_*_run` reuse) |
| `Op::SigmoidInplace`             | f32/f64/bf16/f16     | f32/f64/bf16/f16 (via forward `unary_sigmoid_*_run` reuse) |
| `Op::Fused(INPLACE_AFFINE, _)`   | f32/f64/bf16/f16     | **f32/f64 only** (dedicated `affine_inplace_{f32,f64}_run`) |

The "via forward reuse" pattern: for elementwise unary ops where each
output slot depends only on the matching input slot, Fuel calls the
forward `baracuda_kernels_unary_*_run` symbol passing the same
pointer for both `x` and `y`. This works for the 5 unary activations
shipped today and will work for every additional elementwise unary
in-place variant Fuel adds — no new baracuda symbols needed for that
class.

The affine case is the asymmetric one: baracuda alpha.60 ships
`baracuda_kernels_affine_inplace_{f32,f64}_run` with a different ABI
(single-pointer; `y = mul * y + add`) than the forward
`baracuda_kernels_affine_*_run` (dual-pointer; `y[i] = mul * x[i] +
add`). bf16 + f16 don't have in-place counterparts in alpha.60.

---

## Item 1 — bf16 + f16 affine in-place exposure

**Ask:** add `baracuda_kernels_affine_inplace_bf16_run` and
`baracuda_kernels_affine_inplace_f16_run` to `baracuda-kernels-sys`,
matching the existing f32 + f64 ABI:

```c
// Existing in alpha.60:
cudaError_t baracuda_kernels_affine_inplace_f32_run(
    int64_t numel, float mul, float add,
    void* y_ptr, void* scratch, size_t scratch_bytes, void* stream);

cudaError_t baracuda_kernels_affine_inplace_f64_run(
    int64_t numel, double mul, double add,
    void* y_ptr, void* scratch, size_t scratch_bytes, void* stream);

// Asked for:
cudaError_t baracuda_kernels_affine_inplace_bf16_run(
    int64_t numel, /* mul */ float mul, /* add */ float add,
    void* y_ptr, void* scratch, size_t scratch_bytes, void* stream);

cudaError_t baracuda_kernels_affine_inplace_f16_run(
    int64_t numel, /* mul */ float mul, /* add */ float add,
    void* y_ptr, void* scratch, size_t scratch_bytes, void* stream);
```

(Scalar params pivot through f32 for half-precision, matching how
Fuel's CPU `affine_inplace_{bf16,f16}` kernels handle them and
matching baracuda's likely internal pattern for forward `affine_bf16`.
If baracuda's preferred convention is to take bf16/f16 scalars
directly, that's fine too — Fuel will adapt the wrapper.)

**Why:** completes the 4-dtype matrix that every other in-place
variant in Fuel already covers, and unblocks `Op::AddScalar` /
`Op::MulScalar` in-place rewrites on bf16/f16 model weights (e.g.,
weight-decay scaling in optimizer steps). Today Fuel's bf16/f16
INPLACE_AFFINE on CUDA falls back to the non-inplace `affine_*`
cousin (Cast → Affine → Cast), which defeats in-place semantics and
allocates a scratch buffer per call.

**Severity:** straightforward request. baracuda's f32 + f64
in-place affine kernels are presumably one-line specializations of
the forward kernels (drop the `x` pointer; rewrite `y[i] = mul *
x[i] + add` to `y[i] = mul * y[i] + add`); the bf16 + f16 variants
should follow the same template using the same dtype-specific math
the forward `affine_{bf16,f16}_run` already use.

**Fuel-side commitment if accepted:** wire the 2 new symbols into
`fuel-cuda-backend/src/baracuda/affine.rs` (one `affine_inplace_kernel!`
invocation per dtype, ~5 lines each); add the 2 new
`(InplaceAffine, [bf16, bf16], Cuda)` and `(InplaceAffine, [f16,
f16], Cuda)` registrations in `fuel-storage/src/baracuda_dispatch.rs`;
flip the dtype-coverage table in
`c:/Users/cires/.claude/projects/c--Users-cires-OneDrive-Documents-projects-fuel/memory/project_inplace_ops_complete.md`
to "f32/f64/bf16/f16" for the INPLACE_AFFINE row.

---

## Item 2 — Confirmation: forward-kernel same-pointer reuse contract

**Ask:** confirm in `baracuda-kernels-sys`' docs (or in a short reply
that Fuel can quote in the binding-table comments) that calling the
forward `baracuda_kernels_unary_*_run` family with `x_ptr == y_ptr`
is **supported and safe** as the long-term in-place dispatch path
for elementwise unary ops.

**Why:** Fuel currently uses this trick for all 20 (op × dtype)
in-place unary entries shipped today. It works empirically (live
RTX 4070 tests pass), but Fuel is depending on an undocumented
guarantee — if a future baracuda kernel author introduces a
two-pass implementation (e.g., a temporal-coupling optimization
that reads all of `x` before writing any of `y`), same-pointer
dispatch would silently corrupt data on tail-elements.

The same-pointer contract is the lever that lets Fuel add new
in-place op families WITHOUT requesting dedicated baracuda symbols
each time. We're planning to extend the in-place op surface in
the next few sessions (see Item 3); knowing whether to plan around
"reuse forward symbol with same pointer" or "dedicated in-place
symbol per kind" is a load-bearing architectural choice.

**Three possible answers**, ranked by Fuel's preference:

1. **Yes, supported across all `unary_*_run` (and ideally `binary_*_run`,
   `clamp_*_run`, `powi_*_run`).** Fuel ships in-place dispatch for
   any new elementwise op family without coordinating with baracuda
   for each one. Document the contract in a header comment; Fuel
   will cite that comment in its binding-table registrations.
2. **Yes for unary, no for binary/multi-input families** (because a
   binary kernel might read both `x1` and `x2` before writing `y` to
   support a `y = f(x1, x2) where x2 might overlap with y` permutation
   case). Fuel will request dedicated `binary_inplace_*_run` symbols
   as needed.
3. **Add a dedicated `unary_inplace_*_run` family** with the
   `y_ptr` single-pointer ABI (mirror of the affine pattern). Fuel
   would switch all 20 existing unary in-place wrappers over and
   ask for the corresponding `binary_inplace_*_run` family later
   when it materializes that op family. More baracuda surface area
   but eliminates the same-pointer-contract ambiguity entirely.

**Severity:** documentation / confirmation, not new code. If the
answer is (1), no baracuda work; Fuel just updates its comments.
If (2), Fuel adjusts its forward planning. If (3), Fuel files a
followup ask once the binary in-place op family materializes (no
urgency this quarter).

---

## Item 3 — Forward-look: in-place op families Fuel plans to add

**Update 2026-05-30 (post-ask):** Fuel shipped the 16 unary +
`Op::ClampInplace` + `Op::PowIInplace` families immediately after
filing this ask, using same-pointer dispatch over baracuda alpha.60's
forward symbols. All 80 (16 unary × 4 dtypes) + 4 (powi × 4 dtypes)
+ 4 (clamp × 4 dtypes) entries land green on live RTX 4070 tests.
The same-pointer-contract assumption (Item 2) is therefore now
load-bearing in production; the formal confirmation from the
baracuda team is still wanted as defensive documentation. Listed
below for completeness:

**Shipped (no new baracuda symbols)** — same-pointer reuse over
existing forward kernels:

- `Op::NegInplace`, `Op::AbsInplace`, `Op::SqrInplace`,
  `Op::SqrtInplace`, `Op::RsqrtInplace`, `Op::RecipInplace`,
  `Op::ExpInplace`, `Op::LogInplace`, `Op::SinInplace`,
  `Op::CosInplace`, `Op::SignInplace`, `Op::FloorInplace`,
  `Op::CeilInplace`, `Op::RoundInplace`, `Op::ErfInplace`,
  `Op::GeluErfInplace` — every unary in
  `baracuda_kernels_unary_*_run`, mirrored 4 dtypes.
- `Op::ClampInplace` — reuses
  `baracuda_kernels_ternary_clamp_*_strided_run` with `a_ptr == y_ptr`
  and 1-element broadcast bound buffers.
- `Op::PowIInplace` — reuses `baracuda_kernels_unary_powi_*_run` with
  `x_ptr == y_ptr`, integer `exp` via the `p0: f32` param.

**Still planned, not yet shipped** (same-pointer reuse over forward,
contingent on Item 2 answer for binary):

- `Op::AddInplace`, `Op::SubInplace`, `Op::MulInplace`,
  `Op::DivInplace` (binary in-place, `x1 += x2` shape) — would reuse
  `baracuda_kernels_binary_*_run` with `x1_ptr == y_ptr`. (See
  Item 2 option 2 — if baracuda answers "no same-pointer reuse for
  binary," Fuel will file a `binary_inplace_*_run` ask.)

**Likely needs dedicated baracuda symbols** (asymmetric ABI like
affine):

- *None forecast today.* Fuel doesn't see another op family with
  the affine-style single-pointer asymmetry. Listing this here so
  the baracuda team can speak up if they're aware of a kernel family
  where same-pointer reuse would be unsafe or where a dedicated
  in-place kernel would be substantially more efficient.

**Out of scope for this ask:** fused multi-op in-place kernels
(e.g., `Op::Fused(ADD_RELU_INPLACE)`), in-place softmax / layer
norm / RMS norm (would mutate input with normalized output —
needs careful save-for-backward analysis), and any in-place op that
requires reading the pre-mutation values for backward (Fuel's
Phase 4a view-aware ordering handles save-for-backward for the
elementwise unary case but the policy for stateful kernels needs
design work).

---

## Sequencing

Items 1 and 2 are independent and can land in either order. Item 3
is informational; no action required unless the baracuda team wants
to flag concerns about the planned trajectory.

Fuel's read on urgency: Item 2's answer is the higher-leverage one
(unblocks an entire op-family expansion plan), but Item 1 is the
smaller / more concrete deliverable. If both fit in a single alpha
bump, great; if not, Item 2's answer can come as a docs / Discord
reply ahead of Item 1's kernel addition.

# FDX addition — AFFINE symbolic extents (`FDXExtent` v1 generalization)

> **SUPERSEDED / FOLDED INTO `docs/specs/dlpack-extension.md` as of 2026-06-17 — retained as
> rationale only; do NOT re-integrate.** The parent spec is the single source of truth and already
> contains this addition in full (§5.2 bit 8, §6.4/§6.4.2/§6.4.3, validators V16/V17 + V7/V14
> arms, Appendix A, example §13.7, plus the 2026-06-17 critique fixes folded after integration —
> the i128-widening eval, the V17-before-V14 ordering, the negative-coeff V16 cross-check). Read
> the parent for the authoritative form; this file explains the *why*.

**Status:** COMPANION RATIONALE — FOLDED INTO the parent (2026-06-17, rev. 2). **Design pass — no
code yet.** The post-integration authoritative form lives in `docs/specs/dlpack-extension.md`.
This document is a *self-contained addition* that specifies the AFFINE extent variant: its
struct shapes (C + Rust), field semantics, realize-time evaluation through `SymEnv`, the
bounds/OOB guard, backward-compat with the as-built `Scalar`/`Range`, the validator extensions,
and a worked persistent-decode example.

> **Reconciliation note (rev. 2).** This addition was originally drafted *independently of* the
> gather addition (`fdx-addition-gather.md`); both first claimed flag bit 7 and both proposed a
> trailing append to the same structs. The integration record now lives in the parent spec:
> **gather keeps bit 7, affine moved to bit 8**, and the `FDXExtent` layout is governed by a
> *load-bearing field-offset-freeze rule* (§3.0 below). This rev. brings the standalone affine
> draft into byte-exact agreement with the integrated `dlpack-extension.md` (§5.2 flag table,
> §6.4–§6.4.3, §8 V7/V14/V16/V17, §13.7, Appendix A). Where this draft and the integrated spec
> disagree, the **integrated spec wins**; this draft is the rationale + self-contained companion.

**USER DECISION recorded:** FDX v1 `FDXExtent` **will carry AFFINE symbolic-extent
expressions** so persistent decode (`k_len = cached_len + new_tokens` — the current ROADMAP
frontier) is expressed *symbolically*, planned once, and served every token **without
per-pass recompute of a derived symbol**. This *resolves* FDX open question §17.3
("Affine sym expressions … v1 keeps it a single `sym_id`") in favor of carrying a small
affine form **in v1**, not deferring to a producer-precomputed composite sym.

**Authoritative inputs:** FDX spec §3.1 (capacity-honesty corollary), §3.1.1 (live-prefix is a
COPY), §5.2 (flag-bit allocation table), §5.3 (sidecar layout), §5.4 (size discipline),
§6.0 (FDX-owned codes), **§6.4 (`FDXExtent`)**, §7.3 (`FDXSymEnv` call surface), §8 (validators
V7/V8/V14), §13.4 (single-`Range` KV example), §14 (versioning incl. the array-element-growth
caveat), §17.3 (the deferred affine question); the as-built core types
`fuel-core-types/src/shape.rs` (`Extent {Scalar | Range}`, `DynAxis`, `Shape::resolve`) and
`fuel-core-types/src/symbol.rs` (`SymId(u32)`, `SymEnv` (`SymId -> usize`, write-once),
`DynScalar`); the symbolic-extents design
`docs/session-prompts/symbolic-extents-and-persistent-decode.md` (§5 "the decode application":
`flash` `k_len = cached_len + seq`; §9 "Affine sym expressions"). When this draft and the
constitution conflict, the constitution wins.

---

## 0. Why affine, and why in v1

### 0.1 The frontier need (persistent decode)

The ROADMAP frontier is the persistent decode graph: build the decode-step graph **once**,
re-bind data + a `SymEnv` per token, re-realize the *same* graph, skip `optimize_graph`. The
blocker is that **one dimension's size differs every token**: `total_seq = cached_len + seq`
drives the K/V live extent and the flash `k_len` (session-prompt §1, §5). The as-built
`Extent::Range { min, max, sym }` can carry **one** `SymId`. So `k_len = cached_len + seq` today
must be expressed by either:

- **(a) a single composite `SymId`** that the *binder recomputes and re-binds every pass*
  (`bind(k_len_sym, cached_len + seq)`), or
- **(b)** baking `seq` into the structure (loses the persistent-graph property when `seq` itself
  is symbolic, e.g. prefill chunks).

Option (a) is exactly the "per-pass recompute of a derived symbol" the USER DECISION rules out
for the interchange form: the FDX sidecar must express the *relationship* (`k_len = cached_len +
new_tokens`) so a consumer evaluates it from the **base** symbols already in the `SymEnv`
(`cached_len`, and — when symbolic — `new_tokens`). One description, evaluated per pass from the
base bindings, no producer-side recompute, no extra derived `SymId` to keep coherent.

### 0.2 What "affine" buys (and its bound)

An **affine combination** over the `SymEnv` is the smallest closed form that covers the decode
need and the recorded futures (session-prompt §9: ragged batch lengths, MoE/chunked prefill)
while staying **bounded + POD-serializable**:

```text
value = c0 + Σ_i ( c_i * resolve(sym_i) )            (integer coefficients c0, c_i ∈ i64)
```

- It **subsumes** `Scalar` (a constant: zero terms, `c0 = v`) and `Range` (one term, `c_i = 1`,
  `sym_i = sym`, `c0 = 0`) — see §4, so existing §13.1/§13.4 examples and the as-built
  `Extent::{Scalar,Range}` validate byte-for-byte unchanged.
- It is **not** general arithmetic (no `sym*sym`, no `max`, no `floordiv`). Those are *kernel/op*
  concerns (a `DynScalar` producer computes them and binds a fresh sym), not a *tensor extent*
  description. Affine is the deliberate ceiling: linear in the base symbols, integer-exact,
  fixed-size, trivially evaluable, trivially bounds-checkable.
- It is **bounded**: a small fixed-capacity term list (cap chosen below), with an **explicit
  typed error on overflow** of the term count — never a heap pointer, never a variable-length
  blob in the POD struct.

### 0.3 Term-count cap (chosen)

**`FDX_AFFINE_MAX_TERMS = 4`.** Rationale:

- The decode need is **2 terms** (`cached_len + 1·new_tokens`), often **1 base term + constant**
  (`cached_len + seq` with `seq` a build-time constant ⇒ one term + `c0=seq`).
- 4 leaves headroom for ragged/chunked/MoE composites (e.g. `prefix + chunk·chunks + tail`)
  without unbounding the struct.
- 4 × `(i64 coeff, u32 sym, u32 pad)` = 4 × 16 B = **64 B** of terms; the whole `FDXAffine` is
  **80 B**, keeping `FDXExtent` POD and `#[repr(C)]`-stable (§3.0 size pins).

A producer needing > 4 terms gets a typed `AffineTooManyTerms` error at **build time** (graph
construction / sidecar authoring), per the never-panic / validate-at-build-time rules. It then
falls back to a producer-precomputed composite sym (the option-(a) path), which remains legal —
affine is an *additional* expressive option, not a mandate.

---

## 1. Where this slots into the FDX spec (exact sections)

| FDX section | Change |
|---|---|
| **§5.2** (magic/version/flags) | Add `FDX_FLAG_HAS_AFFINE_EXTENT (1u << 8)` to the *authoritative flag-bit allocation table* — bit **8** (bit 7 is `FDX_FLAG_HAS_GATHER`; the two 2026-06-17 additions collided on bit 7 and affine moved). "≥1 extent is `kind=AFFINE`"; advisory/fast-reject (the per-extent `kind` byte is authoritative). The no-two-bits-collide build test covers it. |
| **§6.0** (FDX-owned codes) | Add the `FDXExtentKind` code (0=Scalar, 1=Range, **2=Affine**), the `cap_kind` values (0=EXPLICIT, 1=AFFINE_MAX reserved), and `FDX_AFFINE_MAX_TERMS` to the owned-constant table + the build-time pin test. |
| **§6.4** (`FDXExtent`) | **Generalize** the struct under the *field-offset-freeze rule* (§3.0): keep `kind`@0, `min`@8, `capacity`@16, `sym_id`@24, **`sym_scope`@28** unchanged; append `cap_kind`@32 + the `affine` sub-block@40 into the old `reserved[u32;2]` region + the struct's additive growth. Add `FDXAffineTerm` (16 B) + `FDXAffine` (80 B). This is the primary edit. |
| **§5.4** (size discipline) | Pin `size_of::<FDXAffineTerm>()==16`, `size_of::<FDXAffine>()==80`, `size_of::<FDXExtent>()` **and** an `offset_of!`-per-field assertion on `FDXExtent` (it is an *array element* — its `sizeof` is the `extents[]` stride; a field-order edit must break the build, not the ABI). |
| **§7.3** (`FDXSymEnv`) | No struct change. Add the affine-evaluation contract: a consumer computes `value = c0 + Σ c_i·lookup(env, sym_i)` over the binding array; **every** referenced `sym_i` must be bound (unbound ⇒ `UnboundSymbol`); checked-i128 accumulate + narrowing rules (§3.4). |
| **§8** (validators) | Extend **V7** (extents) for `kind=Affine` (incl. the `cap_kind==0` guard on **every** kind, closing cross-version `cap_kind` poisoning); extend **V14** (realize-time bounds) to evaluate the affine value before `min ≤ value ≤ capacity`; add **V16** (affine well-formedness) + **V17** (affine evaluation: checked-i128 overflow per step + per-host narrowing, runs **before** V14). V8 (capacity backing) unchanged; its capacity input for an affine axis is `capacity` (EXPLICIT, §3.3). |
| **§13** (examples) | Add **§13.7** "Persistent-decode KV with an AFFINE live extent" (below). §13.4 stays valid (single-sym `Range` degenerate case). |
| **§14** (versioning) | Note affine is additive within v1 under the **array-element-growth caveat**: `FDXExtent` grows by freezing leading offsets + appending into its own reserved, with the `offset_of!` build assertion; an older (pre-affine) reader that meets `kind=2` hits the unknown-`kind` typed-error rule (never silent mis-stride/truncate). A genuine incompatible element-layout change would need a version bump. |
| **§17.3** | Mark **RESOLVED** by this addition (affine carried in v1); list the three residual deferrals (§8 here). |
| **Appendix A** | Add `FDX_AFFINE_MAX_TERMS`, `FDXExtentKind` values, the `cap_kind` values, `FDX_FLAG_HAS_AFFINE_EXTENT (1u << 8)`. |
| **Appendix B** | Add row: composite `k_len = cached_len + new_tokens` ⇒ `FDXExtent{kind=2, cap_kind=EXPLICIT, affine={c0, terms[(coeff,sym)]}}`. |

---

## 2. Struct definitions

### 2.1 New owned constants (§6.0 / Appendix A)

```c
/* Term-count cap for an FDXAffine. Overflow -> typed AffineTooManyTerms. */
#define FDX_AFFINE_MAX_TERMS   4u

/* FDXExtent.kind (FDX-owned, §6.0 pinned table). */
#define FDX_EXTENT_SCALAR      0u   /* concrete; == base shape[i]                     */
#define FDX_EXTENT_RANGE       1u   /* single bounded symbol (as-built Extent::Range) */
#define FDX_EXTENT_AFFINE      2u   /* affine combo c0 + sum c_i*sym_i                */

/* FDXExtent.cap_kind (affine only): how `capacity` is determined (§3.3). */
/*   0 = EXPLICIT   (the v1 path; `capacity` is the concrete bound)               */
/*   1 = AFFINE_MAX (RESERVED, consumer-ahead; rejected in v1)                    */

/* FDXSidecar.flags (§5.2 authoritative table) — additive; bit 7 is HAS_GATHER. */
#define FDX_FLAG_HAS_AFFINE_EXTENT (1u << 8) /* >=1 extent is kind=AFFINE (advisory) */
/* bits 9..63 reserved (0). A build-time test asserts no two FDX_FLAG_* share a bit. */
```

### 2.2 The affine term and combination — C

```c
/* One affine term: coeff * sym. POD, fixed size (16 bytes). */
typedef struct FDXAffineTerm {  /* sizeof == 16 */
  int64_t  coeff;     /* signed integer coefficient (i64). Negatives allowed       */
                      /* (e.g. capacity - cached_len); they do NOT relax a term's  */
                      /* OWN per-symbol bound (§3.6 "negative-coeff note").        */
  uint32_t sym_id;    /* matches SymId(u32). Must be a BASE symbol bound in the    */
                      /* FDXSymEnv (NOT itself an affine result — no nesting).     */
                      /* FDX_SYM_NONE (0xFFFFFFFF) marks an UNUSED slot (coeff=0). */
  uint32_t _pad;      /* zero on write.                                            */
} FDXAffineTerm;

/* An affine combination over the SymEnv:
 *   value = c0 + sum_{i<term_count} terms[i].coeff * resolve(terms[i].sym_id).
 * Fixed-capacity, inline, POD, serializable. */
typedef struct FDXAffine {      /* sizeof == 8 + 8 + 4*16 == 80 */
  int64_t       c0;                          /* constant term (i64).               */
  uint8_t       term_count;                  /* 0..=FDX_AFFINE_MAX_TERMS.          */
  uint8_t       _pad[7];                     /* zero on write.                     */
  FDXAffineTerm terms[FDX_AFFINE_MAX_TERMS]; /* slots >= term_count are zeroed     */
                                             /* (sym_id=FDX_SYM_NONE, coeff=0).    */
} FDXAffine;
```

### 2.3 The generalized `FDXExtent` — C (replaces the §6.4 struct)

> **Layout rule — LOAD-BEARING (cross-version byte compatibility).** The leading bytes of
> `FDXExtent` are **frozen at the original §6.4 field order**: `kind`@0 (`_pad[3]`@1..3),
> `min`@8, `capacity`@16, `sym_id`@24, **`sym_scope`@28** (`_pad2[3]`@29..31). The new affine
> machinery (`cap_kind`@32, then the 8-byte-aligned `affine` sub-block@40) is appended **strictly
> after offset 32**, into what was the original `reserved[u32;2]` region and the struct's additive
> growth. `sym_scope` does **NOT** move — an earlier (pre-affine) extent and an affine-aware
> reader agree on every pre-affine field offset. Because `FDXExtent` is an **array element**
> (`extents[]` stride is its `sizeof`), §5.4 MUST pin `size_of::<FDXExtent>()` *and* `offset_of!`
> for **every** field so a future field-order edit breaks the build, not the ABI.

```c
typedef struct FDXExtent {
  uint8_t   kind;        /* FDXExtentKind: 0=Scalar, 1=Range, 2=Affine. Offset 0.  */
  uint8_t   _pad[3];
  uint64_t  min;         /* live lower bound. Range/Scalar as-built; Affine: the   */
                         /* producer-asserted guaranteed minimum (V14 lower).      */
                         /* Offset 8.                                              */
  uint64_t  capacity;    /* == base.shape[i]; strides keyed here (P4). For Affine  */
                         /* the concrete bound the realized value is checked vs    */
                         /* (cap_kind). Offset 16.                                 */
  uint32_t  sym_id;      /* RANGE only: the single live symbol. Scalar/Affine:     */
                         /* FDX_SYM_NONE (Affine symbols live in affine.terms[]).  */
                         /* Offset 24.                                             */
  uint8_t   sym_scope;   /* 0=InputDetermined, 1=DataDetermined, 2=SessionScoped.  */
                         /* Advisory. Affine: most-constrained scope of its syms.  */
                         /* FROZEN at offset 28 (do NOT move).                     */
  uint8_t   _pad2[3];    /* offsets 29..31                                         */
  uint8_t   cap_kind;    /* AFFINE only: 0=EXPLICIT (v1), 1=AFFINE_MAX (reserved). */
                         /* MUST be 0 for Scalar/Range (V7). Occupies the original */
                         /* reserved[0] low byte. Offset 32.                       */
  uint8_t   _pad3[3];    /* offsets 33..35                                         */
  uint32_t  _pad4;       /* 8-byte-align the affine sub-block. Offsets 36..39.     */
  FDXAffine affine;      /* AFFINE only; all-zero (term_count==0) otherwise.       */
                         /* Inline, no pointer. Offset 40.                         */
  uint32_t  reserved[2];
} FDXExtent;
```

### 2.4 Rust mirror

```rust
/// One affine term `coeff * sym_id`. `#[repr(C)]` POD, EXACTLY 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FDXAffineTerm {
    /// Signed integer coefficient (i64). Negative coeffs are allowed (e.g.
    /// `capacity - cached_len`); they do NOT relax the term's own per-symbol
    /// bound (§3.6).
    pub coeff: i64,
    /// Base `SymId(u32)` bound in the `FDXSymEnv`. `FDX_SYM_NONE` ⇒ unused slot
    /// (then `coeff == 0`). MUST be a BASE symbol — never another affine result
    /// (no nesting).
    pub sym_id: u32,
    pub _pad: u32,
}

/// `value = c0 + Σ_{i<term_count} terms[i].coeff * resolve(terms[i].sym_id)`.
/// Fixed-capacity (`FDX_AFFINE_MAX_TERMS = 4`), inline, POD. EXACTLY 80 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FDXAffine {
    /// Constant term (i64).
    pub c0: i64,
    /// Active term count, `0..=FDX_AFFINE_MAX_TERMS`.
    pub term_count: u8,
    pub _pad: [u8; 7],
    /// Slots `>= term_count` are zeroed (`sym_id = FDX_SYM_NONE`, `coeff = 0`).
    pub terms: [FDXAffineTerm; FDX_AFFINE_MAX_TERMS as usize],
}

/// Per-axis live-vs-capacity extent. Generalizes the as-built
/// `fuel-core-types::shape::Extent` ({Scalar | Range}) with an Affine kind that
/// carries a bounded affine combination over the `SymEnv` (§2). `#[repr(C)]` POD.
/// LEADING FIELD OFFSETS ARE FROZEN (§3.0): `sym_scope` stays at offset 28; the
/// affine machinery is appended after offset 32. It is an ARRAY ELEMENT, so §5.4
/// pins `size_of::<FDXExtent>()` and `offset_of!` per field.
#[repr(C)]
pub struct FDXExtent {
    /// 0 = Scalar, 1 = Range, 2 = Affine (`FDXExtentKind`, §6.0). Offset 0.
    pub kind: u8,
    pub _pad: [u8; 3],
    /// Live lower bound. Scalar: == capacity. Range: `Extent::min`. Affine: the
    /// producer-asserted guaranteed minimum (V14 lower bound). Offset 8.
    pub min: u64,
    /// Capacity (== base `shape[i]`, strides keyed here, P4). For Affine the
    /// concrete bound the realized value is checked against (`cap_kind`, §3.3).
    /// Offset 16.
    pub capacity: u64,
    /// Range only: the single live symbol; else `FDX_SYM_NONE`. Offset 24.
    pub sym_id: u32,
    /// Symbol scope hint (advisory). For Affine: the most-constrained scope of
    /// its referenced symbols. FROZEN at offset 28 (cross-version byte compat).
    pub sym_scope: u8,
    pub _pad2: [u8; 3],
    /// Affine only: how `capacity` is determined. 0 = EXPLICIT (v1), 1 =
    /// AFFINE_MAX (reserved). MUST be 0 for Scalar/Range (V7). Offset 32.
    pub cap_kind: u8,
    pub _pad3: [u8; 3],
    pub _pad4: u32,
    /// Affine only: the combination (`term_count == 0` & all-zero for
    /// Scalar/Range). Carried inline (POD, no pointer). Offset 40.
    pub affine: FDXAffine,
    pub reserved: [u32; 2],
}
```

These map onto a future generalized `fuel-core-types::shape::Extent` as a new variant
`Extent::Affine { min, capacity, c0, terms }` (or an `AffineExpr` helper), but the as-built
`{Scalar, Range}` need not change for FDX to subsume them — the FDX encoding is wider than the
source enum on purpose, exactly as §6.0 already establishes the FDX code tables are
wider/independent of the source enums.

---

## 3. Field semantics & rules (load-bearing — extends §6.4)

### 3.0 Field-offset freeze (the array-element-growth discipline, §14)

`FDXExtent` is the element of a **variable-length array** (`FDXSidecar.extents`, stride =
`sizeof(FDXExtent)`). The §14 `struct_bytes` tail-ignore rule does **not** cover array elements:
growing `sizeof(FDXExtent)` would make a pre-affine reader (striding with its *smaller* element
size) misalign *every* entry after index 0 — silent corruption, not a clean tail-ignore.
Therefore FDX handles `FDXExtent` evolution by:

1. **Freezing every pre-affine field offset** (`kind`@0, `min`@8, `capacity`@16, `sym_id`@24,
   `sym_scope`@28). The original `reserved[u32;2]`@32 is repurposed for `cap_kind`@32 +
   `_pad4`@36; the new `affine`@40 is the additive growth.
2. **Pinning `size_of::<FDXExtent>()` + `offset_of!`-per-field** at build time (§5.4), so a future
   reorder breaks the build instead of the ABI.
3. **Routing an unknown `kind` through the typed-error rule** (§14): a pre-affine reader that
   meets `kind=2` errors (`UnsupportedVersion`-class), never silently mis-strides or truncates.

An affine-aware producer and an affine-aware consumer agree on `sizeof(FDXExtent)` and every
offset; a genuine *incompatible* element-layout change would require a version bump, never an
"additive-within-v1" claim.

### 3.1 Kind discriminant

- **`kind = SCALAR (0)`** — concrete axis, `min == capacity == base.shape[i]`,
  `sym_id == FDX_SYM_NONE`, `cap_kind == 0`, `affine.term_count == 0`. Identical to the as-built
  `Extent::Scalar(v)` / current §6.4 Scalar.
- **`kind = RANGE (1)`** — single bounded symbol, `min ≤ capacity == base.shape[i]`,
  `sym_id != FDX_SYM_NONE`, `cap_kind == 0`, `affine.term_count == 0`. Identical to the as-built
  `Extent::Range { min, max, sym }` / current §6.4 Range. **The default for one-symbol axes** — a
  producer SHOULD emit `Range`, not a one-term affine, to keep the simple path simple (§4
  canonicalization).
- **`kind = AFFINE (2)`** — the live value is the affine combination in `affine` (§2.2),
  evaluated through the `SymEnv` at realize. `sym_id == FDX_SYM_NONE` (symbols are in
  `affine.terms[]`), `cap_kind == EXPLICIT (0)` in v1, `capacity == base.shape[i]` and is the
  bound the realized value is checked against (§3.3, V14).

### 3.2 Affine evaluation (realize-time, §7.3 contract) — overflow-CHECKED per step

Given the call-time `FDXSymEnv` (the boundary form of `SymEnv`, §7.3), the live value of an
affine axis is computed with `checked_*` arithmetic at **every** step (not a wrapping `+=`,
not a single final check):

```text
value: i128 = affine.c0
for i in 0 .. affine.term_count:
    s = lookup(env, affine.terms[i].sym_id)              // typed UnboundSymbol if absent
    prod  = i128::checked_mul(affine.terms[i].coeff, s)  ?: AffineOverflow
    value = i128::checked_add(value, prod)               ?: AffineOverflow
// then narrow (§3.4) and bound-check (V14)
```

- **i128 is NOT unconditionally overflow-free.** With `FDX_AFFINE_MAX_TERMS = 4` and pathological
  i64 coeffs × u64 bindings, a single term can reach ~`2^126` and four summed can exceed `2^127`.
  So the **running accumulation MUST use `checked_add`/`checked_mul` at each step**; any overflow
  ⇒ typed `AffineOverflow` (V17), never a wrap, never deferred to one final check. (With realistic
  decode magnitudes it cannot overflow; the check is the never-silent-coercion guarantee, not a
  hot path. The earlier rev. of this draft said "accumulate then narrow once" — corrected here to
  match V17.)
- **Every referenced `sym_i` MUST be bound** in the `FDXSymEnv`. An unbound symbol is the typed
  `UnboundSymbol` error (matching `SymEnv`'s write-once / presence contract, §7.3), **never a
  silent 0** (mirrors `Shape::resolve` erroring on an unbound dynamic axis, `shape.rs::resolve`).
- **Determinism / unification.** Two axes that must move together carry **the same affine
  expression over the same base syms** (K-length and V-length both `cached_len + new_tokens`), so
  they resolve to the same value by construction — the as-built `Extent`/`SymEnv`
  unification-by-id lifted to expressions. An affine expression unifies when its term set + coeffs
  + `c0` match; the V16 no-duplicate-sym rule + a canonical sort by `sym_id` make `Hash`/`Eq`
  order-independent for plan-cache keying (§8 open item 4).

### 3.3 Capacity determination for an affine axis (the bounds key — load-bearing)

Strides and allocation remain keyed to a **concrete capacity** (P4, §3.1). For an affine axis,
how that concrete capacity is obtained is named by `cap_kind`:

- **`cap_kind = EXPLICIT (0)` — the v1 path, the only `cap_kind` a producer emits.** The
  `capacity` field *is* the concrete bound and MUST equal `base.shape[i]` (V7). For decode this is
  the KV buffer's fixed capacity `K` (e.g. `max_seq_len`): the buffer is physically allocated for
  `K` slots, strides keyed to `K`, and `k_len = cached_len + new_tokens` is checked
  `min ≤ k_len ≤ K` at realize. The honesty invariant (§3, §3.1) requires `base.shape[i] ==
  capacity` so a sidecar-blind consumer reads a correctly-sized, fully-backed `[…, K, …]` tensor
  (V8). The affine expression describes the **live prefix length**, never the capacity; the
  §3.1.1 "live-prefix export is a COPY when the symbolic axis is not leading" reasoning is
  unchanged (the affine `k_len` is just the prefix length on the capacity-strided axis).
- **`cap_kind = AFFINE_MAX (1)` — RESERVED (consumer-ahead).** Capacity would be the affine value
  evaluated at each symbol's per-binding maximum (growable/ragged buffers where capacity itself is
  a function of session bounds). **Not emitted in v1**; validators reject it as
  `UnsupportedVersion`-class until a consumer exists (V7/V16). The field is present so adding it
  later is additive (§14). The decode frontier does not need it — the KV capacity is the
  build-time `max_seq_len`.

### 3.4 u64 ↔ usize narrowing & overflow (extends §6.4 narrowing policy, §7.3)

The as-built `SymEnv` binds `SymId -> usize`; `FDXSymBinding.value` is `u64` (§7.3). The affine
math is signed (`i64` coeffs, `i64 c0`), so:

- **Resolve each `sym_i`** to its `u64` binding, **widen to i128**.
- **Accumulate `c0 + Σ coeff_i · s_i` in i128 with `checked_*` per step** (§3.2); an i128 overflow
  ⇒ typed `AffineOverflow`.
- **Narrow once at the end.** The final value must be `>= 0` (a negative affine live length is a
  producer bug ⇒ `ExtentOutOfRange`, and it is then also `< min` since `min: u64 >= 0`); on a
  **32-bit host** (`usize == u32`), `value > usize::MAX` ⇒ typed `ExtentOutOfRange` — **never
  truncated** (the §6.4 narrowing rule extended to the affine result).
- Then the V14 bound check `min ≤ value ≤ capacity` runs on the narrowed value (§3.5).

### 3.5 Realize-time OOB guard (extends §6.4, V14)

For **every** axis, regardless of kind, after computing the live `value` (Scalar: `capacity`;
Range: `lookup(env, sym)`; **Affine: evaluate §3.2/§3.4 — V17 runs first**): the consumer MUST
verify `min ≤ value ≤ capacity`. A value outside `[min, capacity]` is the typed
`ExtentOutOfRange` error, **not** silently clamped (§6.4). For affine this is the defining OOB
guard for `k_len`: a `cached_len + new_tokens` resolving above `K` would walk past the allocation
— V14 rejects it before any kernel touches memory (the OOB guard the single-`Range` form could
not express without a producer recompute).

### 3.6 Per-symbol bounds are NOT relaxed by the affine guard (no-OOB is compositional)

V14/V17 bound the affine **result** only. Any base symbol that is **also** used elsewhere as an
extent or as an offset into the same buffer — e.g. `cached_len` used both in
`k_len = cached_len + new_tokens` *and* as the persistent-decode write offset — MUST carry its
**own** `FDXExtent`/bound, most naturally as its own `Range` extent on the consumed-prefix axis, or as a
bounded `DynScalar` in the `SymEnv`. The unification-by-`sym_id` rule makes this automatic: the
same `cached_len` symbol resolves identically wherever it appears.

A **negative-coefficient** affine (e.g. `capacity - cached_len` remaining-space) does **not**
relax the per-symbol bounds of its terms — a binding that keeps the *sum* in `[min, capacity]`
while a base term is itself out of its own range is rejected by that term's own bound, not by the
sum. This keeps the no-OOB property compositional, not just a final-sum check.

---

## 4. Backward-compat: Scalar/Range are the degenerate cases (subsumption)

The encoding **subsumes** the as-built `Extent::{Scalar, Range}` and the current §6.4
Scalar/Range so **every existing example still validates byte-for-byte where the producer keeps
emitting `kind ∈ {0,1}`** (the leading offsets are frozen, §3.0), and is *re-expressible* as
affine where useful:

| as-built / pre-affine §6.4 | canonical FDX kind | equivalent affine form — *math only, NOT a legal encoding* |
|---|---|---|
| `Extent::Scalar(v)` | `kind=0` (`min=capacity=v`, `sym=NONE`) | `affine{ c0=v, term_count=0 }` — **V16 rejects; emit Scalar** |
| `Extent::Range{min,max,sym}` | `kind=1` (`min, capacity=max, sym`) | `affine{ c0=0, term_count=1, [{coeff=1, sym}] }` — **V16 rejects; emit Range** |
| `k_len = cached_len + seq` (seq const) | `kind=2` | `affine{ c0=seq, term_count=1, [{1, cached_len}] }` — **legal** (non-zero `c0`) |
| `k_len = cached_len + new_tokens` (both sym) | `kind=2` | `affine{ c0=0, term_count=2, [{1,cached_len},{1,new_tokens}] }` — **legal** |

> The "equivalent affine form" column is **mathematical equivalence for understanding, NOT a legal
> encoding**: the first two rows are exactly the degenerate forms V16 **rejects** (a constant is
> always `Scalar`; a bare coeff-1 zero-`c0` symbol is always `Range`). Only genuinely-composite
> rows (`c0 != 0`, or `term_count >= 2`, or a non-unit coeff) are legal `kind=2` encodings.

**Canonicalization rule (producer policy, V16-checked):** a producer MUST emit the *lowest*
sufficient kind — `Scalar` for a constant, `Range` for a single coeff-1 zero-`c0` symbol,
`Affine` only for a genuinely multi-term / non-unit-coeff / non-zero-`c0` combination. V16
**rejects** the degenerate affine forms so the two encodings never diverge for the same fact,
keeping the simple consumer path (`Scalar`/`Range`) untouched. **No existing §13 example
changes** (they are all `Scalar`/`Range`); affine is purely additive for the
genuinely-composite case.

The two **independent versioning axes** are unaffected (§5.2): affine is an FDX-schema additive
change within FDX v1 (new `kind` value, new flag bit **8**, appended sub-block under the §3.0
array-element discipline), independent of the DLPack ABI `DLPackVersion`.

---

## 5. Validators (extends §8)

Renumber-free additions; **V7** and **V14** gain affine arms, plus new **V16/V17**.

- **V7 — extents (extended).** `extents_count ∈ {0, base.ndim}`; each `capacity == base.shape[i]`;
  `min ≤ capacity`; **`cap_kind == 0 (EXPLICIT)` for EVERY kind** (so a stray nonzero byte at the
  `cap_kind` offset from a mis-versioned blob is caught, not silently ignored — closes the
  cross-version `cap_kind` poisoning). Per kind:
  - `kind=Scalar` ⇒ `sym_id == FDX_SYM_NONE`, `min == capacity`, `affine.term_count == 0`.
  - `kind=Range` ⇒ `sym_id != FDX_SYM_NONE`, `affine.term_count == 0`.
  - `kind=Affine` ⇒ `sym_id == FDX_SYM_NONE`, `cap_kind == EXPLICIT (0)` in v1
    (`AFFINE_MAX` ⇒ `UnsupportedVersion`-class until a consumer exists), and **V16**
    well-formedness holds; `FDX_FLAG_HAS_AFFINE_EXTENT` set iff ≥1 axis is `kind=Affine`.
  - Any other `kind` value ⇒ typed `UnsupportedVersion`-class error (no guess, §14).
  - (The same arms apply to `gather.logical_extents[]` keyed to `logical_shape`/`max_seq_capacity`
    instead of `base.shape` — the gather addition's V21d; affine extents may appear there too.)

- **V14 — realize-time symbol bounds (extended).** For every axis, compute the live `value`
  (Scalar: `capacity`; Range: `lookup(env, sym)`; **Affine: evaluate §3.2/§3.4 — V17 runs
  first**), then enforce `min ≤ value ≤ capacity` ⇒ `ExtentOutOfRange` otherwise. An unbound term
  sym ⇒ `UnboundSymbol`. On a 32-bit host, a narrowed `value > usize::MAX` ⇒ `ExtentOutOfRange`
  (no truncation). V14 bounds the affine **result** only; a base symbol used elsewhere as an
  extent/offset carries its own bound (§3.6).

- **V16 — affine well-formedness (new, build/boundary time).** For `kind=Affine`:
  `1 ≤ term_count ≤ FDX_AFFINE_MAX_TERMS`; each active term (`i < term_count`) has
  `sym_id != FDX_SYM_NONE`; each inactive slot (`i ≥ term_count`) is zeroed
  (`sym_id == FDX_SYM_NONE && coeff == 0`); **no duplicate `sym_id`** across active terms
  (a producer combines repeats into one coeff — keeps the form canonical and Hash/Eq
  order-independent); **not degenerate** (reject `term_count==1 && c0==0 && coeff==1` ⇒ must be
  `Range`; reject the all-constant `term_count==0` form ⇒ must be `Scalar`) per §4;
  `cap_kind ∈ {EXPLICIT}` in v1 (`AFFINE_MAX` ⇒ `UnsupportedVersion`-class). Failure ⇒ typed
  `AffineMalformed` / `AffineTooManyTerms` / `AffineDegenerate`.

- **V17 — affine evaluation safety (new, realize time, runs BEFORE V14).** The i128 accumulation
  (§3.2/§3.4) uses `checked_mul`/`checked_add` at **every** step; any overflow ⇒ `AffineOverflow`
  (i128 is NOT unconditionally safe for 4 terms — a single final check is insufficient). The final
  value is `>= 0`; the host narrowing succeeds (32-bit overflow ⇒ `ExtentOutOfRange`). V17 is the
  evaluation-time complement to V16's structural check; V14 then does the bound comparison on the
  safe, narrowed value.

New typed error variants added to the §8 `FDXError` set: `AffineTooManyTerms`, `AffineMalformed`,
`AffineDegenerate`, `AffineOverflow`. (`UnboundSymbol`, `ExtentOutOfRange` already exist.)
**Validator order:** V16 (structural) at author/boundary time; at realize, **V17 (eval-safe) then
V14 (bounds)**, so a malformed expr is caught before evaluation and an unsafe evaluation before
the bound check.

> **Description-only re-affirmation (P3/G7).** None of V7/V14/V16/V17 priced anything or chose a
> path. The cost of materializing a dense copy for a blind consumer, and the
> contiguize/strided/materialize choice, remain the FKC cost model's job (§6.5/§9.3). FDX only
> *describes* the affine extent; the kernel that consumes `k_len` declares its acceptance in FKC.

---

## 6. Worked example — §13.7 Persistent-decode KV with an AFFINE live extent

> Add as FDX **§13.7**. This is the affine analogue of §13.4 (single-`Range` KV); §13.4 stays as
> the simple one-sym case.

A per-layer K cache `[n_heads=32, K_capacity=4096, head_dim=128]` in F16, in the **persistent
decode graph** (built once, re-realized per token). The live length is
**`k_len = cached_len + new_tokens`**, where:

- `cached_len = SymId(7)` — input-determined, bound up-front each pass (tokens already in cache).
- `new_tokens = SymId(8)` — the tokens being processed this pass (1 in pure decode; the chunk
  size in chunked prefill). Carried as a base sym so the *same* graph serves decode **and**
  prefill chunks without re-baking `seq`.

The buffer is a dense, fully-committed `K_capacity=4096` allocation.

- **base `DLTensor`:** `dtype={kDLFloat,16,1}`, `ndim=3`, `shape=[32, 4096, 128]` (**capacity**),
  `strides=[4096*128, 128, 1]` (explicit, keyed to capacity, non-negative — §3.2/§3.1), `data`
  256-aligned. A sidecar-blind consumer sees an honest, fully-backed F16 `[32,4096,128]` tensor.
- **sidecar:** `flags = HAS_SYMBOLIC | HAS_AFFINE_EXTENT` (note: **`MEANING_REQUIRES_EXT` clear**
  — the base is a faithful F16 tensor and the allocation backs the full capacity, so reading the
  tail is harmless; V8 satisfied by the dense backing, exactly as §13.4).
  - `extents` (count = 3):
    - axis 0: `kind=Scalar`, `min=32`, `capacity=32`, `sym_id=NONE`, `cap_kind=0`.
    - **axis 1 (the live length): `kind=Affine`**, `min=1`, `capacity=4096`, `sym_id=NONE`,
      `cap_kind=EXPLICIT (0)`, `sym_scope=InputDetermined`,
      `affine = { c0=0, term_count=2, terms=[ {coeff=1, sym_id=7 /*cached_len*/},
      {coeff=1, sym_id=8 /*new_tokens*/} ] }`.
    - axis 2: `kind=Scalar`, `min=128`, `capacity=128`, `sym_id=NONE`, `cap_kind=0`.
  - `storage`: `class=Session`, `session_id=<this session>` (KV cache is session-state).
  - `residency`: `tier=Device`, `substrate=CudaUntyped`, `backend_id=Cuda`, `device_index=0`,
    `is_mmap_view=0`.
  - `buffers`: index 0, `size_bytes = 32*4096*128*2` (full-capacity backing ⇒ V8 satisfied).

**Realize (per token), no recompute of a derived sym:** the call passes
`FDXSymEnv { 7 → cached_len, 8 → new_tokens }` (both base symbols already in the pass's
`SymEnv` — `cached_len` bound up-front, `new_tokens` the prompt/decode width). A flash-attention
kernel:

1. Evaluates `k_len = c0 + 1·lookup(env,7) + 1·lookup(env,8) = 0 + cached_len + new_tokens`
   with i128 **checked** accumulate then narrow (§3.2/§3.4 — V17).
2. Applies the realize-time bound `1 ≤ k_len ≤ 4096` (V14) — a `cached_len + new_tokens > 4096`
   is `ExtentOutOfRange` *before* the kernel touches memory (the OOB guard the single-`Range`
   form could not express without a producer recompute).
3. Walks the live prefix using **capacity strides + the affine live extent** — stride = how far
   per element (keyed to 4096), extent = `k_len` live elements (the two halves a kernel needs,
   session-prompt §3.4).

The **same sidecar serves every token**: re-supply the data ptr + the `FDXSymEnv` (`cached_len`
advances, `new_tokens` is 1), reuse the description. **No producer-side recompute of a composite
`k_len` symbol, no per-pass re-bind of a derived sym** — the relationship lives in the sidecar
and is evaluated from the base bindings each pass. This is the persistent-decode property
expressed in the interchange form (the USER DECISION).

**Per-symbol bound (no-OOB compositional, §3.6).** Because `cached_len (SymId 7)` is *also* the
persistent-decode write offset, it carries its **own** bound — either as the consumed-prefix
axis's `Range` extent or as a bounded `DynScalar` in the `SymEnv` — so the affine `k_len` guard
(which bounds only the sum) does not let `cached_len` itself walk OOB.

**Unification:** the V cache for the same layer carries the *identical* affine expression
(`cached_len + new_tokens` over `SymId(7)`, `SymId(8)`), so K-length and V-length resolve to the
same value by construction (§3.2) — the as-built K≡V `SymId` unification (session-prompt §6)
lifted to the affine form.

**Cross-runtime export to a generic consumer (boundary b):** unchanged from §13.4 — because the
live region on the **middle** axis is non-contiguous (§3.1.1), the producer exports a
**materialized dense copy** `[32, k_len, 128]` (with `k_len` evaluated from the env) as a standard
dense F16 tensor with `DLPACK_FLAG_BITMASK_IS_COPIED` set (§9.1). The affine expression is
*resolved away* at the boundary (the generic consumer gets the concrete length); affine identity
is preserved only across a Fuel→Fuel managed export (§15), like any sym.

> **Degenerate-`seq`-constant variant:** in pure single-token decode where `new_tokens` is a
> build-time constant 1, the same axis is `kind=Affine`,
> `affine={ c0=1, term_count=1, terms=[{1, sym=7}] }` (`k_len = cached_len + 1`). Still affine
> (non-zero `c0` ⇒ not reducible to `Range` per §4/V16), still no recompute.

---

## 7. Integration points (code, when built)

Implementation-side (behind the `dlpack` feature, per FDX §0), in dependency order:

1. **`fuel-core-types::dlpack`** — add `FDXAffineTerm`, `FDXAffine`, the generalized `FDXExtent`,
   the `FDXExtentKind` / `FDX_AFFINE_MAX_TERMS` / `cap_kind` / `FDX_FLAG_HAS_AFFINE_EXTENT (1u <<
   8)` constants; `#[repr(C)]` size-assertion **and `offset_of!`-per-field** tests
   (`size_of::<FDXAffineTerm>()==16`, `FDXAffine==80`, `FDXExtent` size + every field offset
   pinned, §5.4); the §6.0 build-time mapping/pin test for `FDXExtentKind`; the no-two-flags-share-
   a-bit test now covers bit 8.
2. **Affine evaluator** — `fn eval_affine(a: &FDXAffine, env: &FDXSymEnv) -> Result<u64>`
   (i128 **checked** accumulate per step, unbound ⇒ `UnboundSymbol`, overflow ⇒ `AffineOverflow`,
   narrow per host with the 32-bit guard). Sibling of `Extent::resolve` / `Shape::resolve`
   (`shape.rs`).
3. **Validators** — V7 affine arm (incl. the `cap_kind==0`-on-every-kind guard), V16, V17, V14
   affine arm in the reference validator (§8); `FDXError` gains the four affine variants.
4. **Generalized source `Extent` (optional, lazy-only)** — when a graph-level producer needs to
   *author* an affine extent (vs. only transport it), add `Extent::Affine { … }` to
   `fuel-core-types::shape` + `Shape::with_affine_axis(axis, min, AffineExpr)` +
   `Extent::resolve`/`Shape::resolve` affine arms (born-red test mirroring
   `symbolic_extent_foundation` in `shape.rs`). Until then FDX can transport affine extents
   authored directly in the sidecar from the binder (the decode binder already computes
   `cached_len + seq`, session-prompt §5).
5. **Realize bridge** — the `FDXSymEnv` already threads through realize (commits `f0679d97` /
   step-1d, `pipelined_bridge` + `InferenceContext::realize_*_with_env`); affine axes resolve
   through the **same** env at the same point as the single-sym `Range` axes — no new plumbing,
   only the evaluator at the resolve site.

No sibling-project (baracuda/vulkane) change is required to *describe* affine extents; consuming
`k_len` from the env is already the flash kernel's interface (session-prompt §0 step 2:
`flash_sdpa_run` takes `k_len`). FDX only changes how `k_len` is *derived* (affine eval) before
it is handed to the kernel.

---

## 8. Open items (this addition; mirrors FDX §17.3 residuals)

1. **Source-`Extent` generalization timing.** Whether to add `Extent::Affine` to
   `fuel-core-types::shape` now or transport-only first (§7.4). Lazy-only norm favors building the
   primitive; sequence behind the persistent-decode consumer (the first real user).
2. **`cap_kind = AFFINE_MAX` consumer.** Reserved for growable/ragged buffers; confirm the
   ragged-batch program (session-prompt §9) is the consumer before un-reserving it.
3. **Negative coefficients.** Type-allowed by i64 (e.g. `capacity - cached_len` remaining space);
   bounded only by the V14 `min ≤ value ≤ capacity` sum check and each term's own per-symbol bound
   (§3.6). Confirm no decode use needs them beyond the bound check (none does today).
4. **Hash/Eq of an affine `Extent`** for plan-cache keying: must be order-independent over terms
   (canonical sort by `sym_id`, per V16's no-duplicate rule) so two equal expressions key the
   same plan. Mirror the `Shape` `Eq`-includes-`dynamic` discipline (`shape.rs`).

---

## Summary (12 lines)

1. USER DECISION applied: FDX v1 `FDXExtent` carries AFFINE expressions; resolves FDX §17.3.
2. New `kind=2 (AFFINE)` on `FDXExtent`, alongside the as-built `Scalar(0)`/`Range(1)`.
3. Value = `c0 + Σ cᵢ·SymIdᵢ`, integer (i64) coeffs, evaluated through the `SymEnv` at realize.
4. Bounded + POD: `FDXAffine` = `c0:i64` + `term_count:u8` + `[FDXAffineTerm;4]` (cap=4 terms, 80 B; term 16 B).
5. Overflow of the term cap ⇒ typed `AffineTooManyTerms` at build time (never panic).
6. Capacity stays a CONCRETE bound via `cap_kind=EXPLICIT` (`capacity==base.shape[i]`, strides keyed there); `AFFINE_MAX` reserved.
7. Realize-time eval is i128 **checked per step** (i128 not safe for 4 terms), unbound sym ⇒ `UnboundSymbol`, narrow once (32-bit host overflow ⇒ typed error, no truncation).
8. V14 extended (eval then `min ≤ value ≤ capacity` ⇒ `ExtentOutOfRange`); new V16 (well-formed) + V17 (eval-safe, runs before V14); V7 also guards `cap_kind==0` on every kind.
9. Subsumes Scalar (0 terms, `c0=v`) and Range (1 term, coeff 1, `c0=0`); V16 rejects degenerate affines so encodings never diverge — no existing §13 example changes (leading offsets frozen, §3.0).
10. Layout is FROZEN: `sym_scope`@28 unchanged, `cap_kind`@32 + `affine`@40 appended; `FDXExtent` is an array element so §5.4 pins size + `offset_of!` per field; flag is bit **8** (gather took bit 7).
11. Worked §13.7: KV `[32,4096,128]`, `k_len = cached_len(SymId 7) + new_tokens(SymId 8)`, capacity K=4096; same sidecar every token, no derived-sym recompute; realize bridge already threads `FDXSymEnv` (step-1d), affine adds only the evaluator at the existing resolve site.
12. FDX sections to touch: §5.2 (flag bit 8), §5.4 (size+offset pins), §6.0+AppA (codes/cap/cap_kind), §6.4 (struct+rules — primary, offset-frozen), §7.3 (eval contract), §8 (V7/V14 extend, add V16/V17), §13.7 (example), §14 (array-element-growth additive note), §17.3 (RESOLVED), AppB (mapping row).

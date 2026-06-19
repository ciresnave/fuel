# Session prompt — Self-describing storage: `SType` / `Encoding`

**Status:** Plan (2026-06-18). Not started. WIP lands on `feat/kernel-contracts-dlpack`
(the same branch the FDX/FKC specs and the DLPack comm layer live on), **never `main`**.

**Goal.** Make *how a tensor's bytes are encoded* (the SCHEME) self-describing **on the
tensor** — so any op holding the tensor knows it is e.g. NF4 block-affine without consulting
op-params — while the scale **VALUES** (bulk data) stay a sibling graph operand. Adds a new
`SType` (a stack of `Encoding` layers) field to `Storage`, default-empty so every existing
single-output `Storage` stays byte-identical in behaviour, plus the `SType::to_fdx()`
projection that fills the **deferred** quant sidecar in
[`fuel-memory/src/dlpack_view.rs`](../../fuel-memory/src/dlpack_view.rs).

This is the LOCKED DECISION (2026-06-18) below, expanded into ordered, TDD-first steps.

---

## LOCKED DECISION (2026-06-18) — Self-describing Storage: DType + SType/Encoding

**THESIS.** Today "how to interpret a tensor's bytes" is split between the tensor view and
op-params (e.g. quant scales passed as op parameters). Goal: make the ENCODING SCHEME
self-describing ON the tensor, so any op holding the tensor knows how its bytes are encoded
without consulting op-params. The scale VALUES (bulk data) stay a sibling operand; only the
SCHEME moves onto the tensor.

**CORE TYPES.**

- **`DType`** (EXISTING, [`fuel-core-types/src/dtype.rs:14`](../../fuel-core-types/src/dtype.rs))
  = the LOGICAL element type ("what is a value"). UNCHANGED, stays logical: an NF4 weight's
  `DType` is the logical float it represents (`F16`/`F32`), not the 4-bit storage.
- **`SType`** (NEW) = an ordered stack of encoding layers describing HOW logical elements are
  physically stored. A NAMED NEWTYPE (not a bare field), shaped:
  `pub struct SType(pub SmallVec<[Encoding; 1]>);` — empty = plain (dense `DType`, no extra
  interpretation). Named newtype because it needs: a home for the layer-ordering invariant, a
  `to_fdx()` projection method, construction invariants, and room for representation evolution.
  Default = empty = plain.
- **`Encoding`** (NEW) = ONE layer. Holds ONLY static descriptors (geometry, scheme, dtype
  codes, scale REQUIREMENTS) — NEVER bulk data (weights) and NEVER scale VALUES. This keeps
  `Encoding` small and `Eq+Hash` so it can feed structure keys / plan caches.
- **`ScaleSpec`** (NEW) = a REQUIREMENT descriptor (scale dtype + granularity; per-block shape
  DERIVED from `base.shape` + `block_shape`), NOT an operand pointer. It says "I need an absmax
  operand of this dtype/granularity"; the consuming OP binds the actual operand; FDX fills the
  concrete `scale_buffer` index at projection.

**ATTACHMENT.** `Storage` gains `stype: SType` (default empty). Today
`Storage = { inner, dtype, bundle: Option<Arc<[OutputView]>> }`
([`fuel-memory/src/lib.rs:89`](../../fuel-memory/src/lib.rs) and
[`fuel-core-types/src/storage.rs:216`](../../fuel-core-types/src/storage.rs)). After: add
`stype: SType`. Default-empty keeps every existing single-output `Storage` byte-identical in
behaviour. **v1: `SType` lives on the PRIMARY `Storage` only**; bundle slots (`OutputView`) keep
`dtype` only. Per-slot `SType` is a FUTURE addition (note it, do not build it).

**THE LOAD-BEARING ARCHITECTURE DECISION** (how scale DATA is carried) — decided AGAINST
embedding the scale buffer inside the weight's `Storage`/`Encoding` ("composite-by-reference",
model A). Decided FOR:

- **(B) GRAPH LAYER = SIBLING OPERANDS.** The per-block scale (absmax) is a SEPARATE first-class
  tensor / graph edge, an operand of the consuming op (dequant / matmul). The weight's `Encoding`
  declares only the REQUIREMENT (`AffineBlock` + `ScaleSpec`); the OP binds the actual scale operand.
- **PLUS KERNEL BOUNDARY = FDX SIDECAR COMPOSITE PROJECTION.** The `DlpackView` / FDX sidecar
  re-unites `{weight scheme, scale-buffer reference}` into ONE self-describing descriptor for the
  kernel (FDX `AFFINE_BLOCK`: `scale_placement = SEPARATE_BUFFER`, `scale_buffer` = buffer-table index).

**WHY B (not A) — verified facts:**

1. Multi-output graph machinery is ONE-BUFFER ONLY. A multi-output node allocates ONE bundled
   `Storage` (one alloc, one Arc) plus `OutputView` offset-slots; `Op::View{slot}` clones the Arc
   and bakes `byte_offset` into `Layout.start_offset` (zero-copy WINDOW into the ONE buffer);
   `Op::ViewOwned{slot}` MEMCPYs a slot into a fresh alloc.
   (`fuel-graph/src/lib.rs:933-976`; `fuel-dispatch/src/pipelined.rs:3542-3668`;
   `docs/architecture/12-multi-output.md`.) So the graph supports "many NODES sharing ONE
   allocation", NOT "one node owning many separate allocations". Folding separate-source scales
   into a bundle would force a load-time merge-copy and kill zero-copy.
2. FDX ALREADY specifies the scale as a SEPARATE operand: `AFFINE_BLOCK` has `scale_present=1`,
   `scale_placement = SEPARATE_BUFFER` (never `INLINE`), `scale_buffer` = a real buffer-table
   index; `GGML_BLOCK` is baked inline with no separate operand.
   ([`docs/specs/dlpack-extension.md:958, 969-980`](../../docs/specs/dlpack-extension.md).) So
   model B at the graph layer is already what the boundary spec assumes — A would contradict the
   shipped FDX.
3. B is cheaper AND honest: scales become a normal placeable/transferable/costable operand (the
   planner already handles operands — NO recursive `Storage`, NO new planner introspection).
   Weight+scale co-location falls out automatically (both feed one op, so both land where it
   runs). Matches external convention (GPTQ / HF / bitsandbytes all pass scales as separate
   tensors). Keeps `Encoding` a small `Eq`/`Hash` POD.
4. The self-describing property that MATTERS is still delivered: the SCHEME is self-describing on
   the tensor (no op-param needed to know it is NF4 block-affine); the scale VALUES are bulk data,
   correctly a sibling operand; FDX re-unites them at the boundary where the kernel needs one
   descriptor.

**GGML STAYS INLINE (forced, not a choice):** GGUF on-disk is interleaved struct-packed (Q4_0 =
`{f16 d; u8 qs[16]}` = 18 bytes/block; see
[`fuel-core-types/src/quantized.rs:87-113`](../../fuel-core-types/src/quantized.rs)
`type_size`/`block_size`); the format, k_quants, and ~40 quantized kernels assume it; zero-copy
mmap requires it. `Encoding::GgmlBlock` = inline. Do NOT generalize interleaving to NF4 (it would
force a repack on load from bnb's separate-tensor format, killing zero-copy, for no kernel-locality
win — the absmax array is tiny and block-indexed).

**EFFICIENCY RULE (general):** match the SOURCE format's native layout to preserve zero-copy on
load. GGML → interleaved; NF4/GPTQ/bnb → separate. There is no universal winner; layout follows
source.

**DYNAMIC vs STATIC:** static scheme/block-size = `Encoding` parameters; dynamic block sizes /
dynamic scales = additional operands (or runtime-produced tensors feeding the op). All expressible
under B with no new `Storage` capability.

---

## Ground-truth reconciliation (read the code, not the sketch)

Every fact below was read from the live tree on 2026-06-18 (`feat/kernel-contracts-dlpack`).

| Sketch claim | Ground truth | Divergence / note |
|---|---|---|
| `Storage = { inner, dtype, bundle }` | [`fuel-memory/src/lib.rs:89-101`](../../fuel-memory/src/lib.rs) — fields `inner: BackendStorage`, `dtype: DType`, `bundle: Option<Arc<[OutputView]>>` (all `pub`). | **Matches.** There are TWO distinct `Storage` types (see next row). |
| One `Storage` | TWO: (a) [`fuel-core-types/src/storage.rs:216`](../../fuel-core-types/src/storage.rs) `Storage { inner: Box<dyn DynBackendStorage>, bundle }` (note: **no `dtype` field** — dtype comes from `inner.dtype_dyn()`); (b) [`fuel-memory/src/lib.rs:89`](../../fuel-memory/src/lib.rs) `Storage { inner: BackendStorage, dtype, bundle }`. | **Divergence vs sketch.** The sketch's "`Storage = { inner, dtype, bundle }`" describes the `fuel-memory` one. The `fuel-core-types` one is a *separate* trait-object wrapper with **no `dtype` field**. BOTH get `stype` (step 2), but the `fuel-core-types` one's constructors are `new`/`from_dyn`/`from_dyn_bundled`/`with_bundle` (storage.rs:234, 242, 256, 283), NOT `new`/`new_bundled`/`with_bundle`. |
| `Encoding` variants from FDX vocabulary | `FDX_QUANT_GGML_BLOCK=0`, `FDX_QUANT_AFFINE_BLOCK=4` ([`fuel-core-types/src/dlpack/codes.rs:79, 86`](../../fuel-core-types/src/dlpack/codes.rs)). `GgmlDType` enum at [`quantized.rs:26`](../../fuel-core-types/src/quantized.rs) (already `Eq+Hash`). | **Matches.** `Encoding::GgmlBlock` carries `GgmlDType`; `Encoding::AffineBlock` maps to family 4. |
| NF4/F4 packed sub-byte code | There is **no `FDX_DTYPE_NF4`**. The sub-byte 4-bit code is `FDX_DTYPE_F4 = 13` ([codes.rs:176](../../fuel-core-types/src/dlpack/codes.rs)); `DType::F4` ([dtype.rs:44](../../fuel-core-types/src/dtype.rs)) is the logical 4-bit float; `dtype_to_fdx(DType::F4)` is in [`convert.rs:44`](../../fuel-core-types/src/dlpack/convert.rs). | **Divergence vs sketch's "NF4/F4".** Use `DType::F4` (→ `FDX_DTYPE_F4`) as the packed sub-byte code in v1. "NF4" is a *normalization variant* of 4-bit affine; FDX has no distinct code for it yet — model it as `packed = DType::F4` + `AffineBlock`. Note this in the deferred list. |
| `ScaleSpec` granularity reuses existing type | `ScaleGranularity { PerTensor, PerToken, PerChannel }` at [`quant_scale.rs:38`](../../fuel-core-types/src/quant_scale.rs) (`Clone+Copy+Debug+PartialEq+Eq+Hash`), re-exported [`lib.rs:57`](../../fuel-core-types/src/lib.rs). **No `PerBlock`** in the Fuel enum (that code is FDX-MX-only). | **Matches the FDX rule** (`AFFINE_BLOCK` grain rides `block_shape`, not a granularity byte — [spec §6.2, lines 982-985](../../docs/specs/dlpack-extension.md)). `ScaleSpec` carries `ScaleGranularity` + a `scale_dtype: DType`; block geometry lives in `Encoding::AffineBlock::block_shape`. |
| `smallvec` available in fuel-core-types | `smallvec = { workspace = true }` in [`fuel-core-types/Cargo.toml:19`](../../fuel-core-types/Cargo.toml). | **Matches** — no new dep. |
| `FDXQuant` fields for projection | [`sidecar.rs:51-96`](../../fuel-core-types/src/dlpack/sidecar.rs): `family, ggml_dtype, block_ndim, block_shape:[u32;4], block_axes:[i32;4], pack_order, scale_present, scale_dtype, scale_placement, scale_granularity, scale_buffer, zp_present, zp_dtype, zp_buffer, ...`. `quant_none()` builder at [`dlpack_view.rs:271`](../../fuel-memory/src/dlpack_view.rs). | **Matches.** `to_fdx()` writes exactly these fields. `FDX_FLAG_HAS_QUANT = 1<<1` ([codes.rs:31](../../fuel-core-types/src/dlpack/codes.rs)). |
| quant sidecar deferred in `view()` | [`dlpack_view.rs:485-491`](../../fuel-memory/src/dlpack_view.rs): "quant … sidecar needs the consuming op's quant params … deliberately deferred (`[consumer-ahead]`)". `view()` always writes `quant: quant_none()` at [dlpack_view.rs:665](../../fuel-memory/src/dlpack_view.rs). | **This is exactly what SType unblocks.** The scheme now travels on `storage.stype`; `view()` reads it instead of needing op-context. The scale BUFFER reference is still op-context (step 4). |

**The one thing the sketch gets wrong that changes the plan:** there are **two** `Storage`
structs (`fuel-core-types` and `fuel-memory`), with **different constructor sets** and the
`fuel-core-types` one has **no `dtype` field**. Step 2 must touch both, with the right
constructor names for each.

---

## Discipline (guardrails — these are hard rules)

- **NEVER run workspace-wide `cargo check`/`cargo test`** — `tensor-tools` has a standing
  `Device::Cpu` break and is a default-member, so even bare `cargo check` at the root fails.
  **Always `-p <crate>`.**
- **ONE cargo invocation at a time** (the build-dir lock serializes). Long builds: background + wait.
- **TDD born-red:** write the failing test FIRST, run it, observe it fail (red), then make it
  green. Do not write the implementation before the test. Report the red→green transition.
- **Docs in the SAME change as behaviour.** When `Encoding`/`SType` lands, add a
  `docs/architecture/` note or a `10-decisions-log.md` entry if a core claim/interface changes.
  Update this plan's checkboxes as steps complete.
- **Never panic on production paths** — `Result`-returning constructors; no `try_*` siblings.
- **WIP on `feat/kernel-contracts-dlpack`, not `main`.**
- Commit messages end with the line:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Step 1 — `Encoding` + `ScaleSpec` + `SType` as a new module in `fuel-core-types`

**Files to touch:**
- NEW `fuel-core-types/src/stype.rs` — the module.
- `fuel-core-types/src/lib.rs` — add `pub mod stype;` (alongside the other `pub mod` lines at
  [lib.rs:22-45](../../fuel-core-types/src/lib.rs)) and a `pub use stype::{SType, Encoding, ScaleSpec};`
  (alongside the re-exports at [lib.rs:49-63](../../fuel-core-types/src/lib.rs)).
- `fuel-core-types/Cargo.toml` — **no change** (`smallvec` already present at line 19).

**Failing test FIRST** (put in `stype.rs` under `#[cfg(test)] mod tests`). Write these, run,
watch them fail to compile / fail assertion (born-red):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::quantized::GgmlDType;
    use crate::quant_scale::ScaleGranularity;

    /// Default SType is empty = plain (the byte-identical default).
    #[test]
    fn default_stype_is_empty_plain() {
        let s = SType::default();
        assert!(s.is_plain(), "default SType must be plain (empty layer stack)");
        assert_eq!(s.layers().len(), 0);
    }

    /// SType is Eq + Hash so it can feed structure keys / plan caches.
    #[test]
    fn stype_is_eq_and_hash() {
        use std::collections::HashSet;
        let a = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 });
        let b = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 });
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b), "equal STypes must hash equal");
    }

    /// AffineBlock carries the static descriptor only (geometry + scale REQUIREMENT),
    /// never the scale values. ScaleSpec is a requirement, not a pointer.
    #[test]
    fn affine_block_holds_static_scale_requirement() {
        let enc = Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        };
        let s = SType::from_layer(enc.clone());
        assert!(!s.is_plain());
        assert_eq!(s.layers()[0], enc);
        // The packed sub-byte storage is F4; the LOGICAL dtype is NOT here — it
        // lives on Storage.dtype (step 2). Encoding never names the logical float.
        match &s.layers()[0] {
            Encoding::AffineBlock { packed, block_shape, .. } => {
                assert_eq!(*packed, DType::F4);
                assert_eq!(block_shape.as_slice(), &[64u32]);
            }
            _ => panic!("expected AffineBlock"),
        }
    }

    /// GgmlBlock is a single self-contained inline layer (no scale sibling).
    #[test]
    fn ggml_block_is_inline_single_layer() {
        let s = SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4K });
        assert_eq!(s.layers().len(), 1);
        assert!(!s.requires_scale_sibling(),
            "GGML scale is baked inline; no sibling operand required");
    }

    /// AffineBlock requires a scale sibling operand (the absmax).
    #[test]
    fn affine_block_requires_scale_sibling() {
        let s = SType::from_layer(Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        });
        assert!(s.requires_scale_sibling());
    }
}
```

**Implementation** (after the tests are red):

```rust
//! Self-describing storage encoding (`SType` / `Encoding` / `ScaleSpec`).
//!
//! `DType` (see [`crate::dtype`]) is the LOGICAL element type — "what is a
//! value". `SType` is orthogonal: it describes HOW those logical elements are
//! physically encoded (block-quantized, sub-byte-packed, …). An empty `SType`
//! means "plain": the bytes are a dense array of `DType`, no extra interpretation.
//!
//! Design: `docs/session-prompts/self-describing-storage-plan.md` (LOCKED
//! DECISION 2026-06-18). The SCHEME is self-describing on the tensor; the scale
//! VALUES are a sibling graph operand (model B); FDX re-unites them at the kernel
//! boundary (`SType::to_fdx`, behind the `dlpack` feature — step 3).

use smallvec::SmallVec;

use crate::dtype::DType;
use crate::quant_scale::ScaleGranularity;
use crate::quantized::GgmlDType;

/// A REQUIREMENT for a sibling per-block / per-axis scale operand — NOT a
/// pointer to one. Says "I need an absmax operand of this dtype + granularity";
/// the consuming op binds the actual operand, and FDX fills the concrete
/// `scale_buffer` index at projection (step 4). The per-block scale SHAPE is
/// DERIVED from the base shape + the layer's `block_shape`, not stored here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScaleSpec {
    /// Element dtype of the required scale operand (commonly `F32`).
    pub dtype: DType,
    /// Granularity of the required scale operand. For `AffineBlock` the block
    /// grain rides the layer's `block_shape`; this is the coarse dispatch-key
    /// granularity (FDX keeps `PerBlock` MX-only — see spec §6.2).
    pub granularity: ScaleGranularity,
}

/// ONE encoding layer. Holds ONLY static descriptors (geometry, scheme, dtype
/// codes, scale REQUIREMENTS) — NEVER bulk data and NEVER scale VALUES. Small,
/// `Eq + Hash` so it can feed structure keys / plan caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// GGUF / ggml block format. Scale is baked INLINE in each block struct;
    /// one self-contained buffer, no separate scale operand. Maps to FDX
    /// `GGML_BLOCK` (family 0). `ggml_dtype` IS the format.
    GgmlBlock { ggml_dtype: GgmlDType },

    /// NF4 / QLoRA-style block-grained affine. Maps to FDX `AFFINE_BLOCK`
    /// (family 4): low-bit packed data + a SEPARATE per-block absmax scale
    /// operand (model B). `packed` is the sub-byte storage code (`DType::F4`
    /// for 4-bit; FDX has no distinct NF4 code in v1 — see plan deferred list).
    AffineBlock {
        /// Sub-byte packed storage code (e.g. `DType::F4`).
        packed: DType,
        /// Block extent along each quantized axis (QLoRA default `[64]`).
        block_shape: SmallVec<[u32; 2]>,
        /// The REQUIREMENT for the sibling per-block absmax operand.
        scale: ScaleSpec,
        /// Asymmetric affine zero-point requirement; `None` for symmetric
        /// (NF4 is symmetric → `None`).
        zero_point: Option<ScaleSpec>,
    },

    /// RESERVED placeholder for MX block-scaled (FDX `MX`, family 1). Declared
    /// for shape only; NOT wired in v1 (see plan deferred list).
    Mx,
    // Reserved for LATER (do NOT implement now): AffineInt, AffineFloat, Compressed.
}

impl Encoding {
    /// Whether this layer needs a sibling scale operand bound by the consuming
    /// op (true for `AffineBlock`; false for inline GGML).
    pub fn requires_scale_sibling(&self) -> bool {
        matches!(self, Encoding::AffineBlock { .. })
    }
}

/// An ordered stack of [`Encoding`] layers describing how a `Storage`'s bytes
/// are physically encoded. EMPTY = plain (dense `DType`, no extra interpretation)
/// — the default, byte-identical to pre-SType behaviour.
///
/// Named newtype (not a bare field) because it owns: the layer-ordering
/// invariant, the [`SType::to_fdx`] projection (step 3, `dlpack` feature),
/// construction invariants, and room for representation evolution.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct SType(pub SmallVec<[Encoding; 1]>);

impl SType {
    /// The plain (empty) SType — dense `DType`, no extra interpretation.
    pub fn plain() -> Self { SType(SmallVec::new()) }

    /// A single-layer SType.
    pub fn from_layer(e: Encoding) -> Self {
        let mut v = SmallVec::new();
        v.push(e);
        SType(v)
    }

    /// True iff there are no encoding layers (plain dense `DType`).
    pub fn is_plain(&self) -> bool { self.0.is_empty() }

    /// The layer stack (outermost-first ordering invariant TBD as layers grow;
    /// v1 has at most one layer).
    pub fn layers(&self) -> &[Encoding] { &self.0 }

    /// Whether ANY layer needs a sibling scale operand.
    pub fn requires_scale_sibling(&self) -> bool {
        self.0.iter().any(Encoding::requires_scale_sibling)
    }
}
```

Note: `smallvec::smallvec![...]` macro requires `use smallvec::smallvec;` in tests, or the
fully-qualified `smallvec::smallvec!`. Confirm the workspace `smallvec` exposes the `smallvec!`
macro (it does by default).

**Build command:** `cargo test -p fuel-core-types stype`
(scoped to the module; never workspace-wide).

**Done-check:**
- The 5 tests above run and pass.
- `cargo test -p fuel-core-types stype` is green; no new warnings on the new file.
- `SType`, `Encoding`, `ScaleSpec` are re-exported from the crate root (`fuel_core_types::SType`).
- `Encoding` derives `Eq + Hash` (compiler-enforced by the `HashSet` test).

---

## Step 2 — add `stype: SType` to both `Storage` structs (default empty, byte-identical)

There are **two** `Storage` structs (see ground-truth table). Both gain `stype: SType`, defaulting
to empty so every existing single-output `Storage` is behaviourally identical.

**Files to touch:**
- [`fuel-core-types/src/storage.rs`](../../fuel-core-types/src/storage.rs) — the
  `Box<dyn DynBackendStorage>` wrapper (`Storage { inner, bundle }` at line 216; **no `dtype`
  field**). Constructors: `new` (234), `from_dyn` (242), `from_dyn_bundled` (256), `with_bundle`
  (283), and `try_clone` (385).
- [`fuel-memory/src/lib.rs`](../../fuel-memory/src/lib.rs) — the `BackendStorage` enum wrapper
  (`Storage { inner, dtype, bundle }` at line 89). Constructors: `new` (140), `new_bundled` (149),
  `with_bundle` (162). Plus the free fns `alloc_cpu_zeroed` (233) and `from_slice_cpu` (243).

**Failing test FIRST.**

In [`fuel-memory/src/lib.rs`](../../fuel-memory/src/lib.rs) `mod tests`:

```rust
/// Born-red: a plain Storage has an empty (plain) SType by default.
#[test]
fn plain_storage_has_empty_stype() {
    let s = alloc_cpu_zeroed(DType::F32, 4).expect("alloc");
    assert!(s.stype().is_plain(), "default Storage must carry a plain SType");
    assert_eq!(s.stype().layers().len(), 0);
}

/// Born-red: attaching an SType is preserved and does not disturb dtype/bytes.
#[test]
fn storage_with_stype_round_trips() {
    use fuel_core_types::{SType, Encoding, GgmlDType};
    let s = alloc_cpu_zeroed(DType::F32, 8).expect("alloc")
        .with_stype(SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 }));
    assert!(!s.stype().is_plain());
    assert_eq!(s.dtype(), DType::F32);   // dtype unchanged
    assert_eq!(s.len_bytes(), 32);       // bytes unchanged
}
```

In [`fuel-core-types/src/storage.rs`](../../fuel-core-types/src/storage.rs) `mod multi_output_specs`
(or a new `mod stype_attach`), add a parallel born-red test asserting `Storage::new(b).stype().is_plain()`
on the trait-object wrapper (construct a dummy `DynBackendStorage` — reuse whatever the existing
storage.rs tests use; if none, gate this assertion behind the `fuel-memory` test which exercises a
real CPU backend, and assert at minimum that the field defaults via a unit construction).

**Implementation** (after red):

`fuel-memory/src/lib.rs` — add the field and thread it through every constructor with a default:

```rust
#[derive(Debug)]
pub struct Storage {
    pub inner: BackendStorage,
    pub dtype: DType,
    pub bundle: Option<Arc<[OutputView]>>,
    /// How the bytes are ENCODED (orthogonal to `dtype`, which is the logical
    /// element type). Empty = plain dense `dtype`. v1: PRIMARY storage only —
    /// bundle slots keep `dtype` only (per-slot SType is a future addition).
    pub stype: SType,
}
```

Update each constructor so existing behaviour is byte-identical:
- `new(inner, dtype)` → `Self { inner, dtype, bundle: None, stype: SType::default() }`.
- `new_bundled(inner, dtype, bundle)` → `... stype: SType::default() ...` (after validation).
- `with_bundle(self, bundle)` → unchanged except the `stype` field is already on `self` (moves through).
- `alloc_cpu_zeroed` / `from_slice_cpu` → call `Storage::new(...)` (already default-empty); **no
  signature change**.

Add a builder + accessor:

```rust
impl Storage {
    /// Attach an encoding scheme to this storage (consuming builder). Does not
    /// touch the bytes or the logical dtype.
    pub fn with_stype(mut self, stype: SType) -> Self {
        self.stype = stype;
        self
    }
    /// The encoding scheme. Empty = plain dense `dtype`.
    pub fn stype(&self) -> &SType { &self.stype }
}
```

`fuel-core-types/src/storage.rs` — same field + accessor on the trait-object wrapper. Because this
one has **no `dtype` field**, the struct becomes `Storage { inner, bundle, stype }`. Thread the
default through `new`, `from_dyn`, `from_dyn_bundled`, `with_bundle`. **Critical:** `try_clone`
(storage.rs:385) currently does `Storage::from_dyn(self.inner.try_clone_dyn(layout)?)` — that drops
any attached `stype`. Decide and document: v1 a clone of a plain storage stays plain (correct,
since `from_dyn` defaults empty); if cloning an *encoded* storage must preserve `stype`, change
`try_clone` to `Ok(Storage::from_dyn(...).with_stype(self.stype.clone()))`. **For v1, preserve it**
(the cheap correct choice) and add a one-line test that `try_clone` carries `stype` forward.

Import `SType` in both files (`use fuel_core_types::SType;` in fuel-memory; `use crate::stype::SType;`
in storage.rs).

**Build command (two invocations, ONE AT A TIME):**
1. `cargo test -p fuel-core-types storage` (the trait-object wrapper + stype attach).
2. `cargo test -p fuel-memory` (the `BackendStorage` wrapper, including the two new born-red tests).

**Done-check:**
- `plain_storage_has_empty_stype` and `storage_with_stype_round_trips` pass.
- ALL pre-existing `storage.rs` / `fuel-memory` tests still pass (byte-identical: `compose_bundle_*`,
  `cpu_storage_basic_shape`, `from_slice_cpu_round_trip`, etc. — listed at
  [storage.rs:744-836](../../fuel-core-types/src/storage.rs) and
  [lib.rs:252-310](../../fuel-memory/src/lib.rs)).
- No constructor's public *signature* changed (only internal field init).
- `try_clone` preserves `stype` (new one-line test green).

---

## Step 3 — `SType::to_fdx()` → fill the DEFERRED quant sidecar in `dlpack_view.rs`

This is the payoff: the quant sidecar that `view()` deliberately deferred
([`dlpack_view.rs:485-491`](../../fuel-memory/src/dlpack_view.rs), "needs the consuming op's quant
params … deliberately deferred") is now UNBLOCKED — **because the SCHEME travels on
`storage.stype`**, `view()` no longer needs op-context to know the tensor is NF4 block-affine. It
reads `storage.stype.to_fdx(...)` and writes `FDXQuant` + `FDX_FLAG_HAS_QUANT` directly. The scale
**buffer index** is still op-context (the consuming op binds the operand and supplies the
buffer-table slot — step 4); v1 emits the scheme with `scale_buffer = FDX_BUFFER_NONE` as a
placeholder when no buffer table is supplied, OR takes an optional `scale_buffer: Option<u32>`
argument that step 4 fills.

**Files to touch:**
- `fuel-core-types/src/stype.rs` — add `to_fdx()` behind `#[cfg(feature = "dlpack")]`.
- [`fuel-memory/src/dlpack_view.rs`](../../fuel-memory/src/dlpack_view.rs) — `view()` reads
  `storage.stype` and replaces the hardcoded `quant: quant_none()`
  ([dlpack_view.rs:665](../../fuel-memory/src/dlpack_view.rs)) with the projected `FDXQuant` when
  `!storage.stype.is_plain()`; set `flags |= FDX_FLAG_HAS_QUANT` and
  `need_sidecar |= !storage.stype.is_plain()`.

**The projection (`stype.rs`, `dlpack` feature):**

```rust
#[cfg(feature = "dlpack")]
impl SType {
    /// Project this encoding scheme into an `FDXQuant` for the kernel boundary.
    /// The scale BUFFER reference (`scale_buffer`) is op-context: pass
    /// `scale_buffer = Some(idx)` once the consuming op has bound the sibling
    /// scale operand into the buffer table (step 4); `None` ⇒ `FDX_BUFFER_NONE`
    /// placeholder (scheme is described, buffer not yet bound).
    ///
    /// Returns `None` for a plain SType (no quant sidecar needed) and for the
    /// `Mx` placeholder (not wired in v1).
    pub fn to_fdx(&self, scale_buffer: Option<u32>) -> Option<crate::dlpack::sidecar::FDXQuant> {
        use crate::dlpack::codes::*;
        use crate::dlpack::convert::{dtype_to_fdx, ggml_to_fdx};
        // v1: at most one layer.
        let layer = self.0.first()?;
        match layer {
            Encoding::GgmlBlock { ggml_dtype } => {
                // GGML: scale baked INLINE; no separate operand, no granularity.
                Some(FDXQuant {
                    family: FDX_QUANT_GGML_BLOCK,
                    ggml_dtype: ggml_to_fdx(*ggml_dtype),
                    block_ndim: 0,
                    _pad0: [0; 3],
                    block_shape: [0; 4],
                    block_axes: [-1; 4],
                    pack_order: 0, // GGML_NATIVE is per-format via ggml_dtype
                    _pad1: [0; 3],
                    scale_present: 0,
                    scale_dtype: FDX_DTYPE_NONE,
                    scale_placement: FDX_SCALE_PLACEMENT_INLINE,
                    scale_granularity: 0,
                    _pad2: [0; 3],
                    scale_buffer: FDX_BUFFER_INLINE,
                    zp_present: 0,
                    zp_dtype: FDX_DTYPE_NONE,
                    _pad3: 0,
                    zp_buffer: FDX_BUFFER_NONE,
                    scale_pair_act: 0,
                    scale_pair_weight: 0,
                    role: 0,
                    _pad4: 0,
                    reserved: [0; 6],
                })
            }
            Encoding::AffineBlock { packed: _, block_shape, scale, zero_point } => {
                let mut bshape = [0u32; 4];
                let mut baxes = [-1i32; 4];
                let n = block_shape.len().min(4);
                for i in 0..n {
                    bshape[i] = block_shape[i];
                    baxes[i] = i as i32; // v1: blocks tile leading axes; refine in step 4
                }
                let (zp_present, zp_dtype, zp_buffer) = match zero_point {
                    Some(zp) => (1u8, dtype_to_fdx(zp.dtype), FDX_BUFFER_NONE),
                    None => (0u8, FDX_DTYPE_NONE, FDX_BUFFER_NONE),
                };
                Some(FDXQuant {
                    family: FDX_QUANT_AFFINE_BLOCK,
                    ggml_dtype: FDX_DTYPE_NONE, // not GGML
                    block_ndim: n as u8,
                    _pad0: [0; 3],
                    block_shape: bshape,
                    block_axes: baxes,
                    pack_order: 0,
                    _pad1: [0; 3],
                    scale_present: 1,
                    scale_dtype: dtype_to_fdx(scale.dtype),
                    scale_placement: FDX_SCALE_PLACEMENT_SEPARATE_BUFFER, // never INLINE
                    // AFFINE_BLOCK grain rides block_shape, NOT a granularity byte
                    // (spec §6.2 lines 982-985). Keep the dispatch-key granularity
                    // for the planner but it is NOT PerBlock.
                    scale_granularity: fdx_gran(scale.granularity),
                    _pad2: [0; 3],
                    scale_buffer: scale_buffer.unwrap_or(FDX_BUFFER_NONE),
                    zp_present,
                    zp_dtype,
                    _pad3: 0,
                    zp_buffer,
                    scale_pair_act: 0, // stored weight format, not a dynamic matmul pairing
                    scale_pair_weight: 0,
                    role: 0,
                    _pad4: 0,
                    reserved: [0; 6],
                })
            }
            Encoding::Mx => None, // RESERVED, not wired in v1
        }
    }
}
```

`fdx_gran(ScaleGranularity) -> u8` maps `PerTensor/PerToken/PerChannel` →
`FDX_SCALE_GRAN_PER_TENSOR/_TOKEN/_CHANNEL` ([codes.rs:89-91](../../fuel-core-types/src/dlpack/codes.rs)).
Check whether [`convert.rs`](../../fuel-core-types/src/dlpack/convert.rs) already has a
`ScaleGranularity → FDX` helper (the comm-layer plan §1.3 promised one); if so, reuse it instead of
re-defining `fdx_gran`. Also confirm `FDX_BUFFER_NONE` / `FDX_DTYPE_NONE` exist in `codes.rs` (the
`quant_none()` builder at [dlpack_view.rs:271](../../fuel-memory/src/dlpack_view.rs) uses both, so
they do).

**`view()` wiring** ([`dlpack_view.rs:492`](../../fuel-memory/src/dlpack_view.rs) onward):

```rust
// NEW: the encoding scheme travels on the tensor (SType). It UNBLOCKS the
// quant sidecar — no op-context needed to know the SCHEME. The scale BUFFER
// is still op-context (step 4): view() emits scale_buffer = FDX_BUFFER_NONE.
let quant = storage.stype.to_fdx(None);
let has_quant = quant.is_some();
let need_sidecar = sub_byte || symbolic || bundled || has_quant;
// ... when assembling FDXSidecar, replace `quant: quant_none()` with:
quant: quant.unwrap_or_else(quant_none),
// ... and `flags |= FDX_FLAG_HAS_QUANT;` when has_quant.
```

**Failing test FIRST** ([`fuel-memory/src/dlpack_view/tests.rs`](../../fuel-memory/src/dlpack_view/tests.rs)
— match the existing T7-style fixtures, e.g. `plain_f32_contiguous_is_faithful_no_sidecar` at line 45):

```rust
/// SType drives the quant sidecar: an AffineBlock NF4 weight projects to FDX
/// AFFINE_BLOCK with scale_placement=SEPARATE_BUFFER and HAS_QUANT set.
#[test]
fn affine_block_stype_emits_quant_sidecar() {
    use fuel_core_types::{SType, Encoding, ScaleSpec, DType, ScaleGranularity};
    use fuel_core_types::dlpack::codes::*;
    let storage = cpu_bytes(DType::F4, 64).with_stype(SType::from_layer(
        Encoding::AffineBlock {
            packed: DType::F4,
            block_shape: smallvec::smallvec![64],
            scale: ScaleSpec { dtype: DType::F32, granularity: ScaleGranularity::PerChannel },
            zero_point: None,
        }));
    let layout = Layout::contiguous(Shape::from_dims(&[64]));
    let v = view(&storage, &layout, None).expect("view");
    let sc = v.sidecar.as_ref().expect("AffineBlock must emit a sidecar");
    assert_ne!(sc.flags & FDX_FLAG_HAS_QUANT, 0, "HAS_QUANT must be set");
    assert_eq!(sc.quant.family, FDX_QUANT_AFFINE_BLOCK);
    assert_eq!(sc.quant.scale_present, 1);
    assert_eq!(sc.quant.scale_placement, FDX_SCALE_PLACEMENT_SEPARATE_BUFFER);
    assert_eq!(sc.quant.scale_buffer, FDX_BUFFER_NONE); // op binds it later (step 4)
    assert_eq!(sc.quant.block_shape[0], 64);
    v.validate().expect("AFFINE_BLOCK sidecar must pass FDX validators");
}

/// A GGML-block SType projects to inline-scale FDX GGML_BLOCK (no sibling).
#[test]
fn ggml_block_stype_emits_inline_quant() {
    use fuel_core_types::{SType, Encoding, GgmlDType, DType};
    use fuel_core_types::dlpack::codes::*;
    let storage = cpu_bytes(DType::F32, 18) // one Q4_0 block (18 bytes)
        .with_stype(SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 }));
    let layout = Layout::contiguous(Shape::from_dims(&[18]));
    let v = view(&storage, &layout, None).expect("view");
    let sc = v.sidecar.as_ref().expect("GGML sidecar");
    assert_eq!(sc.quant.family, FDX_QUANT_GGML_BLOCK);
    assert_eq!(sc.quant.scale_present, 0);
    assert_eq!(sc.quant.scale_buffer, FDX_BUFFER_INLINE);
    v.validate().expect("GGML_BLOCK sidecar must pass FDX validators");
}

/// A plain Storage still emits NO sidecar (byte-identical to today).
#[test]
fn plain_storage_still_no_quant_sidecar() {
    let storage = cpu_f32(12);
    let layout = Layout::contiguous(Shape::from_dims(&[3, 4]));
    let v = view(&storage, &layout, None).expect("view");
    assert!(v.sidecar.is_none(), "plain storage must stay sidecar-free");
}
```

**Watch the validators.** `v.validate()` runs the FDX V-checks
([dlpack_view.rs:159](../../fuel-memory/src/dlpack_view.rs)). The `AFFINE_BLOCK` arm has spec rules
(scale_placement must be SEPARATE_BUFFER, not PerBlock, etc. — [spec §6.2](../../docs/specs/dlpack-extension.md)).
If a validator rejects `scale_buffer = FDX_BUFFER_NONE` (because `scale_present == 1` requires a real
buffer-table index), that is the **born-red signal that step 3 and step 4 cannot fully separate**:
either (a) relax the validator to allow `FDX_BUFFER_NONE` as "scheme described, buffer not yet
bound" with a documented flag, or (b) make `affine_block_stype_emits_quant_sidecar` use the
`view_with_quant(storage, layout, env, scale_buffer)` extension entry that step 4 introduces and
supply a 1-entry buffer table. **Prefer (b)** — it keeps the validator honest and proves the
end-to-end path. Decide this when you see the validator output; do not pre-guess.

**Build command:**
1. `cargo test -p fuel-core-types --features dlpack stype` (the `to_fdx` projection unit tests —
   add a couple in `stype.rs` asserting the family/placement codes directly, no `view()` needed).
2. `cargo test -p fuel-memory --features dlpack dlpack_view` (the view-integration tests).

**Done-check:**
- `to_fdx()` returns the right `family` / `scale_placement` / `scale_buffer` for each `Encoding`.
- The three view tests pass; `v.validate()` is green for both quant arms.
- Plain storage is unchanged (`plain_storage_still_no_quant_sidecar`).
- `view()`'s deferral comment ([dlpack_view.rs:485-491](../../fuel-memory/src/dlpack_view.rs)) is
  updated: the **scheme** half is now wired via `storage.stype`; only the scale **buffer binding**
  remains `[consumer-ahead]` (step 4).

---

## Step 4 — consuming-op wiring: declare the scale sibling operand + read the weight's `ScaleSpec`

Now connect the graph layer (model B). A dequant / quantized-matmul op:
1. reads the weight operand's `Encoding` (via `storage.stype()`) to learn it needs a scale sibling
   (`SType::requires_scale_sibling()` / the `ScaleSpec` requirement);
2. declares the per-block absmax scale as a **separate operand** (a normal graph edge / tensor);
3. the planner co-locates them automatically (both feed one op → both land where it runs — no new
   planner introspection);
4. at the kernel boundary, the op supplies the scale operand's buffer-table index, and the FDX
   projection fills `scale_buffer` (the `view_with_quant(...)` extension from step 3).

**Files to touch (investigate first — these are the likely homes):**
- `fuel-graph/src/registry.rs` — `FusedOpParams` / operand declarations
  (the comm-layer plan cites `registry.rs:104,108,133,159` for `FusedOp.shape_rule`/`dtype_rule`/
  `output_views` and `FusedOpParams { QMatMul, ... }`). The quantized-matmul / dequant op params
  are where the scale sibling operand index is declared. **Grep `QMatMul` / `Dequantize` in
  `fuel-graph/src/` to find the exact op.**
- `fuel-memory/src/dlpack_view.rs` — add `pub fn view_with_quant(storage, layout, env,
  scale_buffer: Option<u32>) -> Result<DlpackView>` (or thread an extra arg) that forwards
  `scale_buffer` into `storage.stype.to_fdx(scale_buffer)` and adds the scale operand as a
  buffer-table entry (role `FDX_BUFFER_ROLE_SCALE = 1`, [codes.rs:105](../../fuel-core-types/src/dlpack/codes.rs)).
  Mirror the `buffers` Vec assembly already at [dlpack_view.rs:630-644](../../fuel-memory/src/dlpack_view.rs).
- The consuming op's kernel-call site (wherever it builds the `DlpackView` for its operands) —
  bind the weight's `DlpackView` via `view_with_quant`, passing the scale operand's buffer index.

**Design note for the implementer (model B, verbatim from the LOCKED DECISION):** the weight's
`Encoding` declares ONLY the REQUIREMENT (`AffineBlock` + `ScaleSpec`); the OP binds the actual
scale operand. Do NOT embed the scale buffer inside the weight's `Storage`/`Encoding` (that is model
A, explicitly rejected — it would force a load-time merge-copy and kill zero-copy, and contradict
FDX which specifies the scale as a separate operand). The scale is a normal
placeable/transferable/costable operand; weight+scale co-location falls out automatically.

**Failing test FIRST** (graph-level, in the op's crate — likely `fuel-graph` or `fuel-dispatch`):
Write a test that builds a quantized-matmul/dequant node whose weight operand carries an
`Encoding::AffineBlock`, asserts the op declares exactly ONE additional scale operand, asserts the
scale operand's declared shape equals the per-block count derived from `base.shape + block_shape`
(the `ScaleSpec` derivation), and — at the boundary — asserts `view_with_quant` fills
`quant.scale_buffer` with the bound index (role `SCALE`). Run it red; it will fail because the op
does not yet declare the sibling.

**Build command:** `cargo test -p fuel-graph <test-name>` (or `-p fuel-dispatch` if the op lives
there — confirm by grep). One invocation; never workspace-wide.

**Done-check:**
- The consuming op declares the scale sibling operand from the weight's `ScaleSpec` (no op-param
  carries the scheme — it is read from `storage.stype()`).
- `view_with_quant` fills `quant.scale_buffer` with the bound buffer-table index (role `SCALE`),
  and `v.validate()` passes with a real scale buffer present (this is where step 3's option-(b)
  validator path becomes fully green).
- The scale operand's derived shape matches `ScaleSpec` + `block_shape` + `base.shape`.
- No new planner introspection added (co-location is automatic).

---

## Step 5 — GGML path stays inline (no behaviour change; described via `Encoding`)

GGML is FORCED inline (GGUF on-disk is interleaved struct-packed: Q4_0 = `{f16 d; u8 qs[16]}` = 18
bytes/block; see [`quantized.rs:87-113`](../../fuel-core-types/src/quantized.rs) `type_size` /
`block_size`). The format, k_quants, and ~40 quantized kernels assume it; zero-copy mmap requires
it. **No code path changes** — GGML quantized storage continues to be the existing
`DynQuantizedStorage` ([quantized.rs:124](../../fuel-core-types/src/quantized.rs)) block math.

The ONLY addition is *descriptive*: a GGUF-loaded tensor's `Storage` MAY carry
`SType::from_layer(Encoding::GgmlBlock { ggml_dtype })` so its scheme is self-describing at the FDX
boundary (step 3's `ggml_block_stype_emits_inline_quant` proves the projection). This is opt-in and
inline — `scale_placement = INLINE`, `scale_buffer = FDX_BUFFER_INLINE`, no sibling operand. **Do
NOT** generalize the interleaving to NF4.

**Files to touch:** none required for behaviour. Optionally, where GGUF tensors are wrapped into
`Storage` (grep `load_quantized` / `GgmlDType` in the loader path — likely `fuel-formats` /
`fuel-core/src/quantized/`), attach the descriptive `Encoding::GgmlBlock`. **Sequence this behind a
real consumer** — if nothing reads the GGML `Encoding` yet, the step-3 projection test is sufficient
coverage and the loader annotation can wait.

**Failing test:** none new beyond step 3's `ggml_block_stype_emits_inline_quant` (the inline
projection is the contract). If you do annotate the loader, add one test that a GGUF-loaded tensor's
`storage.stype()` reports `GgmlBlock` with the right `GgmlDType`.

**Build command:** `cargo test -p fuel-core-types --features dlpack stype` (projection already
covered). If annotating the loader: `cargo test -p <loader-crate> <test-name>`.

**Done-check:** GGML numerics/dispatch are unchanged (no kernel touched); the GGML scheme is
expressible as `Encoding::GgmlBlock`; projection stays inline (no sibling operand).

---

## Step 6 — loader path note (NF4/bnb → separate Storage zero-copy; GGUF → inline)

This step is a **note + a small loader annotation**, not a numerics change. The EFFICIENCY RULE:
match the source format's native layout to preserve zero-copy on load.

- **NF4 / GPTQ / bitsandbytes:** the source ships the packed weights and the absmax scales as
  **separate tensors**. Load each into its OWN `Storage`, zero-copy (no repack). Tag the weight
  `Storage` with `SType::from_layer(Encoding::AffineBlock { packed: DType::F4, block_shape, scale,
  zero_point })`. The scale tensor is a plain separate `Storage` (the sibling operand the consuming
  op binds in step 4). Do NOT merge them into a bundle (that would force a load-time merge-copy and
  break zero-copy — and the bundle machinery is one-buffer-only, see the LOCKED DECISION fact 1).
- **GGUF:** interleaved on disk → load as the existing inline GGML block storage (step 5),
  optionally tagged `Encoding::GgmlBlock`. Zero-copy mmap requires the interleaving stays.

**Files to touch (investigate, annotate only where a consumer exists):**
- The NF4 / bnb loader path (grep `NF4` / `nf4` / `bnb` / `AffineBlock` / `absmax` in
  `fuel-formats/` and `fuel-core/`). When such a loader lands or exists, tag the weight `Storage`
  per above.
- The GGUF loader ([`fuel-formats/src/gguf.rs`](../../fuel-formats/src/gguf.rs) — appears in the
  grep set) for the optional `GgmlBlock` annotation.

**Failing test:** if an NF4 loader exists, write a born-red test that loading an NF4 weight yields a
`Storage` whose `stype()` is `AffineBlock` with the source's `block_shape`, AND a *separate* scale
`Storage` (assert two distinct allocations — zero-copy, not bundled). If no NF4 loader exists yet,
this step is documentation-only: record the rule here and in the loader crate's module docs, and
defer the annotation to when the loader lands (note it in the deferred list).

**Build command:** `cargo test -p fuel-formats <test-name>` (or the relevant loader crate), one
invocation.

**Done-check:** the loader rule is documented; where a loader exists, NF4 → separate zero-copy
`Storage` + `AffineBlock` tag, GGUF → inline. No merge-copy introduced on any load path.

---

## Step 7 — tests round-up + the gather / quant sidecar follow-on

**Consolidated test gate** (run each `-p` scoped, one at a time):
1. `cargo test -p fuel-core-types stype` — `SType`/`Encoding`/`ScaleSpec` units (step 1) +
   `try_clone` stype-preservation (step 2, the storage.rs half).
2. `cargo test -p fuel-core-types --features dlpack stype` — `to_fdx()` projection units (step 3).
3. `cargo test -p fuel-memory` — `Storage` stype field + byte-identical pre-existing tests (step 2).
4. `cargo test -p fuel-memory --features dlpack dlpack_view` — view quant-sidecar integration +
   validators (step 3) + `view_with_quant` buffer binding (step 4).
5. `cargo test -p fuel-graph <op-test>` (or `-p fuel-dispatch`) — consuming-op scale-sibling
   declaration (step 4).

**Follow-on (sequence after this program, do NOT build here):**
- **Gather sidecar.** The gather (`FDX_FLAG_HAS_GATHER` / `FDXIndexedResidency`) sidecar is the
  OTHER deferral in [`dlpack_view.rs:485-491`](../../fuel-memory/src/dlpack_view.rs). It is NOT
  unblocked by SType (it needs paged-pool geometry / block-table op-context, not an encoding
  scheme). It follows the SAME model-B pattern (the block table + context-lens are sibling operands
  the consuming attention op binds) but is a separate program — see
  [`docs/specs/_drafts/fdx-addition-gather.md`](../../docs/specs/_drafts/fdx-addition-gather.md) and
  the comm-layer plan. Note it; do not implement.
- **The quant graph op** ([`quantize-as-graph-op.md`](./quantize-as-graph-op.md)) — `Op::Quantize` /
  `Op::Dequantize` for DYNAMIC quant (FP8/int8 with runtime scales) is the `AffineInt`/`AffineFloat`
  families (reserved in `Encoding`, deferred below). SType describes STATIC schemes on weights;
  dynamic-quant ops produce scales at runtime as sibling operands — same model B, future `Encoding`
  variants.

**Done-check:** all five gates green; the two follow-ons are documented as next programs with
pointers, not started.

---

## What is deliberately deferred (do NOT build in this program)

- **Per-slot `SType` on bundle `OutputView`s.** v1: `SType` lives on the PRIMARY `Storage` only;
  bundle slots ([`OutputView`, storage.rs:46](../../fuel-core-types/src/storage.rs)) keep `dtype`
  only. A multi-output node whose slots have *different* encodings is a future addition.
- **`Encoding::Mx` wiring.** The `Mx` variant is a RESERVED placeholder (FDX `MX` family 1,
  F8E8M0 per-block scale, the sole `PerBlock` user). Declared for shape; `to_fdx()` returns `None`
  for it. Not wired.
- **A distinct NF4 dtype code.** FDX has no `FDX_DTYPE_NF4`; v1 models NF4 as `packed = DType::F4`
  (`FDX_DTYPE_F4 = 13`) + `AffineBlock`. A dedicated NF4 normalization code (and the
  `AffineInt`/`AffineFloat`/`Compressed` reserved families) are future `Encoding`/FDX additions.
- **Dynamic-scale runtime path.** Dynamic block sizes / runtime-produced scales (the
  `Op::Quantize`/`Op::Dequantize` dynamic-quant path) are model-B-expressible but a SEPARATE
  program ([`quantize-as-graph-op.md`](./quantize-as-graph-op.md)). SType v1 covers STATIC,
  load-time schemes only.
- **Gather sidecar** (see step 7 follow-on) — not unblocked by SType; separate program.
- **Layer-ordering invariant enforcement for multi-layer stacks.** v1 `SType` has at most one
  layer; the ordering invariant (outermost-first) and stacked-encoding projection (`to_fdx` over
  >1 layer) are future work — `to_fdx()` reads `self.0.first()` in v1.

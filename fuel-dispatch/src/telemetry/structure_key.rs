//! The `StructureKey` join token + the provider seam Fuel CALLS.
//!
//! Baracuda owns the structure-key encoding and ships the callable
//! `structure_key(op_class, operands, arch) -> StructureKey`. Fuel **calls** it
//! with FDX operand descriptions as input and **never derives the key itself**
//! (K1 opacity). Here the token is treated as opaque bytes for the join; the
//! [`StructureKeyProvider`] trait is the seam.
//!
//! **Environment note (2026-07-03):** Baracuda's callable ships from its FFI
//! (`baracuda-kernels-sys`) and is `#[cfg(feature = "cuda")]`-gated — it is NOT
//! compiled or tested in this environment (nvcc fails under CUDA 13.3). So the
//! only provider built here is [`NullStructureKeyProvider`] (returns `None`);
//! the Baracuda-backed impl is documented, not compiled. Tests use an in-test
//! stub provider returning canned tokens.

use fuel_ir::{DType, Layout};

/// Opaque structure-key token. Baracuda owns the encoding (a string or a `u64`
/// rendered as a string); Fuel treats it as bytes for the `(structure_key,
/// chosen)` join and never derives it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StructureKeyToken(pub String);

/// The contiguity class of one operand as the structure-key input sees it.
/// A thin two-state projection of the live [`Layout`]; the richer classes
/// (inner-div / vec-width) Baracuda keys on are derived provider-side from the
/// full descriptor, not here (Fuel never derives the key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contiguity {
    /// Row-major C-contiguous (dense inner walk).
    Contiguous,
    /// Non-contiguous — arbitrary explicit strides.
    Strided,
}

/// An FDX operand description — the canonical input to Baracuda's
/// `structure_key` (FDX §4.1). Fuel projects a live `(Layout, DType)` into this
/// backend-agnostic description; the packed-quant / sub-byte axis rides `dtype`
/// for now (its `SType`/`FDXQuant` refinement is a later step).
///
/// It carries BOTH the raw descriptor (`shape` / `strides` / `align_bytes` /
/// `dtype`) that a provider hands to Baracuda's `structure_key` verbatim, AND a
/// few pre-derived boolean/enum axes (`contiguity` / `broadcast` / `flipped`)
/// that Fuel's own miss detector reads. The provider re-derives the key from the
/// RAW fields (K1: Fuel never derives the key) — the derived booleans are for
/// Fuel-internal consumers, never fed back into the key.
///
/// The `flipped` axis is **load-bearing today**: Fuel made negative strides
/// first-class (2026-06-17), so an `Op::Flip`ped operand reaches the dispatch
/// site AS flipped rather than laundered into a contiguous copy. It is the one
/// derived axis with a live Fuel consumer, so the projection must preserve it —
/// and the raw negative stride survives in `strides` for Baracuda too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdxOperandDesc {
    /// The operand's logical element dtype.
    pub dtype: DType,
    /// Contiguity class (row-major vs arbitrary strided).
    pub contiguity: Contiguity,
    /// A stride-0 broadcast axis (with extent > 1) is present.
    pub broadcast: bool,
    /// A negative (reverse) stride axis is present — the `Op::Flip` view.
    /// Kept first-class so the flip survives to the descriptor (see type docs).
    pub flipped: bool,
    /// Per-axis logical extents (the RAW shape, outer→inner). Carried so a
    /// provider can hand Baracuda's `structure_key` the FULL descriptor it
    /// derives the key from — Fuel never derives the key itself (K1 opacity).
    pub shape: Vec<i64>,
    /// Per-axis SIGNED element strides (raw, outer→inner). `0` = broadcast axis,
    /// `< 0` = flipped/reversed view. Baracuda re-derives contiguity / vec-width
    /// / divisibility / flip from these; Fuel supplies them, never the key.
    pub strides: Vec<i64>,
    /// Base-pointer alignment ESTIMATE in bytes — drives the structure key's
    /// vec-width axis. The live base pointer does not exist at plan time on a
    /// lazy DAG, so this MODELS Fuel's ≥256-byte base allocation alignment
    /// refined by the view's byte offset (see [`estimate_align_bytes`]). It is a
    /// documented estimate that informs only the key's vec-width class — never
    /// dispatch — so an imperfect value costs a coarser join, never a wrong pick.
    pub align_bytes: u32,
}

impl FdxOperandDesc {
    /// Project a live `(Layout, DType)` into an FDX operand description.
    ///
    /// Derived axes (Fuel-internal consumers):
    /// - `flipped`  ← any negative stride (the `Op::Flip` demand axis).
    /// - `broadcast`← any stride-0 axis whose extent is > 1.
    /// - `contiguity` ← [`Layout::is_contiguous`] (row-major) else `Strided`.
    ///
    /// Raw descriptor (fed to Baracuda's `structure_key` verbatim):
    /// - `shape`   ← [`Layout::dims`] as `i64`.
    /// - `strides` ← [`Layout::stride`] as `i64` (signed — 0 bcast, < 0 flip).
    /// - `align_bytes` ← [`estimate_align_bytes`] over the view's byte offset.
    pub fn from_layout(layout: &Layout, dtype: DType) -> Self {
        let raw_strides = layout.stride();
        let dims = layout.dims();
        let flipped = raw_strides.iter().any(|&s| s < 0);
        let broadcast = raw_strides
            .iter()
            .zip(dims.iter())
            .any(|(&s, &d)| s == 0 && d > 1);
        let contiguity = if layout.is_contiguous() {
            Contiguity::Contiguous
        } else {
            Contiguity::Strided
        };
        Self {
            dtype,
            contiguity,
            broadcast,
            flipped,
            shape: dims.iter().map(|&d| d as i64).collect(),
            strides: raw_strides.iter().map(|&s| s as i64).collect(),
            align_bytes: estimate_align_bytes(layout.start_offset(), dtype),
        }
    }
}

/// Fuel's device/host base allocations are at least 256-byte aligned (matches
/// CUDA `cudaMalloc` and the pooled allocator), which is also the base alignment
/// Baracuda's own `OperandDesc::from_tensor_ref` documents.
const BASE_ALLOC_ALIGN_BYTES: u32 = 256;

/// Estimate an operand's base-pointer byte alignment from its view offset.
///
/// The live pointer does not exist at plan time (lazy DAG), so this MODELS the
/// ≥256-byte base allocation alignment ([`BASE_ALLOC_ALIGN_BYTES`]) and refines
/// it by the view's byte offset: a sub-view at element offset `o` starts at
/// `base + o·elem`, whose alignment is `gcd(256, o·elem)`. Deterministic and
/// honest for views; sub-byte dtypes (no whole-byte element) yield `1`.
pub fn estimate_align_bytes(start_offset_elems: usize, dtype: DType) -> u32 {
    let elem = dtype.size_in_bytes();
    if elem == 0 {
        return 1; // sub-byte dtype: no meaningful whole-byte alignment
    }
    let offset_bytes = start_offset_elems.saturating_mul(elem);
    if offset_bytes == 0 {
        return BASE_ALLOC_ALIGN_BYTES;
    }
    // gcd(256, offset_bytes) = 2^min(8, trailing_zeros(offset_bytes)).
    let tz = offset_bytes.trailing_zeros().min(8);
    (1u32 << tz).min(BASE_ALLOC_ALIGN_BYTES)
}

/// The seam to Baracuda's shipped `structure_key(op_class, operands, arch)`.
/// Fuel CALLS this and returns the provider's token verbatim; it never derives
/// the token (K1 opacity, FKC §4.12 / FDX §4.1).
///
/// The `operands` are [`FdxOperandDesc`]s (the canonical input, FDX §4.1). A
/// `None` return means "no key available" — the provider is not linked (the v1
/// default [`NullStructureKeyProvider`]), so a dispatch site simply keys `None`
/// and no miss demand signal is formed. The Baracuda-backed impl is
/// `#[cfg(feature = "cuda")]` FFI and is NOT compiled in this environment.
pub trait StructureKeyProvider: Send + Sync {
    /// Obtain the structure-key token for a dispatch site's live operands.
    /// `op_class` names the op family (e.g. `"matmul"`), `arch` the target
    /// architecture tag (e.g. `"sm_89"`). Returns `None` when unlinked.
    fn structure_key(
        &self,
        op_class: &str,
        operands: &[FdxOperandDesc],
        arch: &str,
    ) -> Option<StructureKeyToken>;
}

/// The v1 default provider: Baracuda's callable is not linked (its FFI is
/// cuda-gated and absent here), so no token is available and every dispatch
/// keys `None`. A build that never installs a real provider therefore emits
/// dispatch records without a structure key and forms no miss demand signal —
/// the honest "unlinked" posture, never a fabricated token.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullStructureKeyProvider;

impl StructureKeyProvider for NullStructureKeyProvider {
    fn structure_key(
        &self,
        _op_class: &str,
        _operands: &[FdxOperandDesc],
        _arch: &str,
    ) -> Option<StructureKeyToken> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{Layout, Shape, StrideVec};

    /// A NEGATIVE inner stride surfaces as `flipped == true` in the FDX
    /// descriptor. Load-bearing: negative-strides-first-class keeps this axis
    /// visible instead of laundering the flip into a contiguous copy.
    #[test]
    fn flipped_operand_sets_flipped_axis_in_fdx_desc() {
        // shape [4, 3]; contiguous stride is [3, 1]. Flip dim0 → stride
        // [-3, 1], start_offset = 3 * (4 - 1) = 9 (an `Op::Flip` view).
        let shape = Shape::from(vec![4usize, 3]);
        let stride: StrideVec = [-3isize, 1].into_iter().collect();
        let layout = Layout::new(shape, stride, 9);

        let desc = FdxOperandDesc::from_layout(&layout, DType::F32);
        assert!(desc.flipped, "negative stride must set flipped");
        assert_eq!(desc.contiguity, Contiguity::Strided, "a flip is not contiguous");
        assert!(!desc.broadcast, "no stride-0 axis here");
    }

    /// A plain contiguous operand is `Contiguous`, not flipped, not broadcast.
    #[test]
    fn contiguous_operand_projects_cleanly() {
        let layout = Layout::contiguous(Shape::from(vec![8usize, 16]));
        let desc = FdxOperandDesc::from_layout(&layout, DType::F16);
        assert_eq!(desc.contiguity, Contiguity::Contiguous);
        assert!(!desc.flipped);
        assert!(!desc.broadcast);
        assert_eq!(desc.dtype, DType::F16);
    }

    /// A stride-0 axis with extent > 1 sets `broadcast`.
    #[test]
    fn broadcast_axis_sets_broadcast() {
        // shape [4, 3] with inner stride 0 → a broadcast along the inner axis.
        let shape = Shape::from(vec![4usize, 3]);
        let stride: StrideVec = [1isize, 0].into_iter().collect();
        let layout = Layout::new(shape, stride, 0);
        let desc = FdxOperandDesc::from_layout(&layout, DType::F32);
        assert!(desc.broadcast, "stride-0 extent-3 axis is a broadcast");
    }

    /// The v1 default provider is unlinked: every call yields `None` (no
    /// fabricated token). This is the honest "Baracuda callable not linked"
    /// posture the whole miss path degrades to.
    #[test]
    fn null_provider_yields_none() {
        let p = NullStructureKeyProvider;
        let operands = [FdxOperandDesc {
            dtype: DType::F32,
            contiguity: Contiguity::Contiguous,
            broadcast: false,
            flipped: false,
            shape: vec![8, 16],
            strides: vec![16, 1],
            align_bytes: 256,
        }];
        assert!(p.structure_key("matmul", &operands, "sm_89").is_none());
    }

    /// `from_layout` carries the RAW shape/strides/align the provider hands to
    /// Baracuda — not just the derived booleans. A contiguous [4,3] f32 keeps
    /// its extents + row-major strides and estimates the base alignment.
    #[test]
    fn from_layout_carries_raw_shape_strides_and_align() {
        let layout = Layout::contiguous(Shape::from(vec![4usize, 3]));
        let desc = FdxOperandDesc::from_layout(&layout, DType::F32);
        assert_eq!(desc.shape, vec![4i64, 3]);
        assert_eq!(desc.strides, vec![3i64, 1]);
        // offset 0 ⇒ the modelled base allocation alignment.
        assert_eq!(desc.align_bytes, 256);
        // A negative inner stride survives raw (not laundered).
        let flipped = Layout::new(
            Shape::from(vec![4usize, 3]),
            [3isize, -1].into_iter().collect(),
            2,
        );
        let fdesc = FdxOperandDesc::from_layout(&flipped, DType::F32);
        assert_eq!(fdesc.strides, vec![3i64, -1]);
        assert!(fdesc.flipped);
    }

    /// The alignment estimate refines the 256-byte base by the view's byte
    /// offset: `gcd(256, offset·elem)`.
    #[test]
    fn estimate_align_bytes_refines_by_offset() {
        // offset 0 ⇒ full base alignment.
        assert_eq!(estimate_align_bytes(0, DType::F32), 256);
        // offset 1 f32 = 4 bytes ⇒ gcd(256, 4) = 4.
        assert_eq!(estimate_align_bytes(1, DType::F32), 4);
        // offset 2 f32 = 8 bytes ⇒ gcd(256, 8) = 8.
        assert_eq!(estimate_align_bytes(2, DType::F32), 8);
        // offset 64 f32 = 256 bytes ⇒ capped at the 256 base.
        assert_eq!(estimate_align_bytes(64, DType::F32), 256);
        // sub-byte dtype ⇒ no whole-byte alignment.
        assert_eq!(estimate_align_bytes(3, DType::F4), 1);
    }
}

//! Multi-output bundle **data** types.
//!
//! As of B0.3 the type-erased `Storage` handle (`Box<dyn DynBackendStorage>`
//! + eager-dispatch ops) and `allocate_bundled_storage` moved to the
//! `fuel-backend-contract` crate — they name the backend-contract traits, which
//! sit above this vocabulary crate. What stays here is the bundle **data**:
//! [`OutputView`] (per-slot description), [`OutputViewSpec`] (author-side spec),
//! and [`compose_bundle`] (the pure byte-offset/alignment composer) — none of
//! which name a backend trait, so consumers like `fuel-graph` and `fuel-memory`
//! keep importing them straight from `fuel_ir::storage`.

use crate::{DType, Error, Layout, Result, Shape};

/// Per-slot description of one output inside a multi-output bundled
/// `Storage`. The slot's bytes live at
/// `[byte_offset .. byte_offset + len_elements * dtype.size_in_bytes()]`
/// inside the bundle's inner buffer.
///
/// Each slot carries its own dtype, shape, and [`Layout`] — they are
/// independent. Two slots may have different dtypes (e.g. an F32 `y`
/// alongside an I64 `argmax_idx`) and different ranks.
#[derive(Debug, Clone)]
pub struct OutputView {
    /// Byte offset into the bundle's inner buffer where this slot
    /// starts. Must satisfy the slot's dtype alignment.
    pub byte_offset:  usize,
    /// Number of dtype-sized elements this slot covers. Must equal
    /// `shape.elem_count()` for contiguous slots; for strided slots,
    /// it bounds the slot's reachable byte range (typically equal to
    /// the contiguous element count of the shape).
    pub len_elements: usize,
    /// The slot's element dtype. Independent of every other slot's
    /// dtype and of the bundle's "primary" dtype.
    pub dtype:        DType,
    /// The slot's logical shape.
    pub shape:        Shape,
    /// The slot's logical layout (strides, contiguity, start offset
    /// *within the slot*). The `Layout::start_offset` is element-
    /// counted within the slot, NOT the bundle — it composes with
    /// `byte_offset` at access time.
    pub layout:       Layout,
    /// Optional debugging name (`Some("y")`, `Some("last_state")`).
    /// Not load-bearing; the slot index is the dispatch key.
    pub name:         Option<&'static str>,
}

impl OutputView {
    /// Total byte size of this slot inside the bundle, including any
    /// strided/non-contiguous padding implied by the layout. For a
    /// contiguous slot this is `len_elements * dtype.size_in_bytes()`.
    pub fn len_bytes(&self) -> usize {
        self.len_elements.saturating_mul(self.dtype.size_in_bytes())
    }
}

/// Author-side per-slot output spec for a multi-output fused op.
///
/// Compared to [`OutputView`], this drops the byte-offset and
/// element-count fields — the allocator derives them from the
/// dtype / shape / layout when it composes a bundle. Lets op authors
/// (via `FusedOpEntry::output_views`) describe their outputs purely
/// in terms of "what does each output look like" without thinking
/// about packing order.
#[derive(Debug, Clone)]
pub struct OutputViewSpec {
    /// The slot's element dtype.
    pub dtype:  DType,
    /// The slot's logical shape.
    pub shape:  Shape,
    /// The slot's logical layout. For a freshly allocated slot the
    /// caller typically passes [`Layout::contiguous(shape)`]; strided
    /// slots are permitted but currently uncommon (the kernel would
    /// need to honour them on writes).
    pub layout: Layout,
    /// Optional debugging name.
    pub name:   Option<&'static str>,
}

impl OutputViewSpec {
    /// Convenience: contiguous slot with the standard
    /// `Layout::contiguous(shape)` and no name.
    pub fn contiguous(dtype: DType, shape: Shape) -> Self {
        let layout = Layout::contiguous(shape.clone());
        Self { dtype, shape, layout, name: None }
    }

    /// Element count of this slot — `shape.elem_count()` for
    /// contiguous slots; for strided slots this is still the logical
    /// element count (which bounds the slot's byte footprint).
    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Byte footprint of this slot, ignoring inter-slot alignment.
    pub fn len_bytes(&self) -> usize {
        self.elem_count().saturating_mul(self.dtype.size_in_bytes())
    }
}

/// Compose a slot-spec list into the inputs needed by the bundled
/// allocator: a list of resolved [`OutputView`] entries (with
/// `byte_offset` + `len_elements` filled in) and the total byte size
/// of the bundle.
///
/// Per-slot alignment policy: each slot's `byte_offset` is rounded up
/// to the next multiple of the slot's `dtype.size_in_bytes()`. That
/// keeps every slot naturally aligned for typed loads / stores,
/// without padding ever exceeding `align - 1` bytes per boundary.
///
/// Rejects:
/// - empty spec list (a "multi-output" with zero slots is a contract
///   bug — use single-output);
/// - any slot whose `layout.shape()` disagrees with its `shape`
///   (mirrors the `Storage::with_bundle` / `Graph::set_output_views`
///   coherence rule).
pub fn compose_bundle(
    specs: &[OutputViewSpec],
) -> Result<(usize, Vec<OutputView>)> {
    if specs.is_empty() {
        return Err(Error::Msg(
            "compose_bundle: spec list must be non-empty".into(),
        ).bt());
    }
    let mut views = Vec::with_capacity(specs.len());
    let mut cursor: usize = 0;
    for (i, spec) in specs.iter().enumerate() {
        if spec.layout.shape() != &spec.shape {
            return Err(Error::Msg(format!(
                "compose_bundle: slot {i} layout.shape() = {:?} \
                 disagrees with spec shape {:?}",
                spec.layout.shape(), spec.shape,
            )).bt());
        }
        let align = spec.dtype.size_in_bytes().max(1);
        let rem = cursor % align;
        if rem != 0 {
            cursor += align - rem;
        }
        let len_elements = spec.elem_count();
        views.push(OutputView {
            byte_offset:  cursor,
            len_elements,
            dtype:        spec.dtype,
            shape:        spec.shape.clone(),
            layout:       spec.layout.clone(),
            name:         spec.name,
        });
        cursor = cursor.saturating_add(
            len_elements.saturating_mul(spec.dtype.size_in_bytes()),
        );
    }
    Ok((cursor, views))
}

#[cfg(test)]
mod multi_output_specs {
    use super::*;

    /// Helper: contiguous F32 spec of the given shape.
    fn f32_spec(dims: &[usize]) -> OutputViewSpec {
        OutputViewSpec::contiguous(DType::F32, Shape::from_dims(dims))
    }

    /// Helper: contiguous F64 spec of the given shape.
    fn f64_spec(dims: &[usize]) -> OutputViewSpec {
        OutputViewSpec::contiguous(DType::F64, Shape::from_dims(dims))
    }

    /// compose_bundle composes a single-slot spec into one OutputView
    /// with byte_offset 0 and total_bytes equal to the slot's
    /// footprint.
    #[test]
    fn compose_bundle_single_slot() {
        let specs = vec![f32_spec(&[2, 3])];
        let (total, views) = compose_bundle(&specs).expect("single slot composes");
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].byte_offset, 0);
        assert_eq!(views[0].len_elements, 6);
        assert_eq!(views[0].dtype, DType::F32);
        assert_eq!(total, 24); // 6 * 4 bytes
    }

    /// compose_bundle stacks slots with per-slot dtype alignment.
    /// Slot 0 = F32[6] = 24 bytes (aligned to 4).
    /// Slot 1 = F64[3] = 24 bytes (aligned to 8). cursor at 24 is
    /// already aligned to 8, so byte_offset = 24.
    #[test]
    fn compose_bundle_two_slot_aligned() {
        let specs = vec![f32_spec(&[2, 3]), f64_spec(&[3])];
        let (total, views) = compose_bundle(&specs).expect("aligned compose");
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].byte_offset, 0);
        assert_eq!(views[0].len_elements, 6);
        assert_eq!(views[1].byte_offset, 24);
        assert_eq!(views[1].len_elements, 3);
        assert_eq!(views[1].dtype, DType::F64);
        assert_eq!(total, 48); // 24 (slot 0) + 24 (slot 1)
    }

    /// compose_bundle pads when slot 1's alignment requires it.
    /// Slot 0 = F32[1] = 4 bytes, cursor at 4.
    /// Slot 1 = F64[1], alignment 8; 4 % 8 = 4, pad to 8. byte_offset = 8.
    #[test]
    fn compose_bundle_pads_for_alignment() {
        let specs = vec![f32_spec(&[1]), f64_spec(&[1])];
        let (total, views) = compose_bundle(&specs).expect("padded compose");
        assert_eq!(views[0].byte_offset, 0);
        assert_eq!(views[1].byte_offset, 8); // padded from 4 to next-8-multiple
        assert_eq!(total, 16); // 8 (start of slot 1) + 8 (slot 1)
    }

    /// compose_bundle rejects an empty spec list.
    #[test]
    fn compose_bundle_rejects_empty() {
        let err = compose_bundle(&[]).err()
            .expect("empty spec list must error");
        assert!(format!("{err}").contains("non-empty"));
    }

    /// compose_bundle rejects a spec whose layout.shape() disagrees
    /// with its declared shape (mirrors with_bundle's invariant).
    #[test]
    fn compose_bundle_rejects_shape_layout_mismatch() {
        let s = Shape::from_dims(&[2, 3]);
        let bogus_layout = Layout::contiguous(Shape::from_dims(&[3, 2]));
        let bad = OutputViewSpec {
            dtype:  DType::F32,
            shape:  s,
            layout: bogus_layout,
            name:   None,
        };
        let err = compose_bundle(&[bad]).err()
            .expect("shape/layout mismatch must error");
        assert!(format!("{err}").contains("disagrees"));
    }

    /// OutputViewSpec::contiguous wires the default layout correctly.
    #[test]
    fn output_view_spec_contiguous_helper() {
        let s = f32_spec(&[4, 5]);
        assert_eq!(s.dtype, DType::F32);
        assert_eq!(s.shape, Shape::from_dims(&[4, 5]));
        assert_eq!(s.layout.shape(), &Shape::from_dims(&[4, 5]));
        assert_eq!(s.elem_count(), 20);
        assert_eq!(s.len_bytes(), 80);
        assert!(s.name.is_none());
    }
}

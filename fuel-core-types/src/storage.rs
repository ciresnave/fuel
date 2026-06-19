//! Type-erased tensor storage wrapper (Phase 7.5 work item G fix-up).
//!
//! `Storage` was originally `fuel-core::Storage` and lived alongside the
//! eager-dispatch op methods that mutate it. Phase 7.5 work item G moves
//! the struct and the methods that depend only on fuel-core-types here so
//! that `fuel_graph::Graph` can own a NodeId-keyed map of
//! `Arc<RwLock<Storage>>` slots without inverting the dependency graph.
//!
//! The eager-dispatch methods that need fuel-core types (`CustomOp1/2/3`)
//! stay in fuel-core via the `StorageApplyOps` trait extension. They are
//! scheduled for removal in Phase 7.5 work item B6 (drop eager dispatch).
//!
//! `Storage::device()` returns `Arc<dyn DynBackendDevice>` rather than
//! the `Device` wrapper (which still lives in fuel-core); callers wrap as
//! needed. This is the one API change vs. the pre-G surface.
//!
//! ## Multi-output bundles (Option C, Session 1)
//!
//! A `Storage` may optionally carry [`OutputView`] side-table entries
//! describing how the inner byte buffer is partitioned into multiple
//! logically independent outputs. This is the substrate-side half of
//! the multi-output-node infrastructure: a multi-output op allocates one
//! bundled `Storage` whose `bundle` slice describes per-slot dtype/shape/
//! layout at byte offsets within the same underlying buffer. The
//! backend trait surface ([`DynBackendStorage`]) is intentionally
//! unchanged — backends still produce single typed-byte buffers; bundle
//! metadata lives only on the `Storage` newtype.

use crate::dyn_backend::{DynBackendDevice, DynBackendStorage};
use crate::stype::SType;
use crate::op::{self, BinaryOp, CmpOp, ReduceOp};
use crate::scalar::Scalar;
use crate::{
    conv, DType, Error, HostBuffer, InplaceOp1, InplaceOp2, InplaceOp3, Layout, Result, Shape,
};
use std::sync::Arc;

/// Per-slot description of one output inside a multi-output bundled
/// [`Storage`]. The slot's bytes live at
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
///   (mirrors the [`Storage::with_bundle`] / [`Graph::set_output_views`]
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

/// Allocate a bundled [`Storage`] on `device` covering all of `specs`.
///
/// Sizes the underlying allocation to `total_bytes / primary_size`
/// elements of the primary (slot 0) dtype, rounding up so the buffer
/// is guaranteed to hold every slot's bytes. Zero-initialised — the
/// kernel can overwrite each slot on first write.
///
/// Returns a [`Storage`] whose [`Storage::bundle()`] is the resolved
/// slot-view list and whose primary dtype equals slot 0's dtype.
pub fn allocate_bundled_storage(
    device: &dyn DynBackendDevice,
    specs: &[OutputViewSpec],
) -> Result<Storage> {
    let (total_bytes, views) = compose_bundle(specs)?;
    let primary_dtype = specs[0].dtype;
    let primary_size = primary_dtype.size_in_bytes().max(1);
    // Round up so the allocation holds every slot's bytes even when
    // the bundle's total isn't a clean multiple of the primary dtype's
    // size (e.g. F32 primary + F64 secondary at an odd boundary).
    let flat_elems = total_bytes.div_ceil(primary_size).max(1);
    let inner = device
        .zeros_impl_dyn(&Shape::from_dims(&[flat_elems]), primary_dtype)?;
    Storage::from_dyn_bundled(inner, Arc::from(views.into_boxed_slice()))
}

/// Owns a typed contiguous buffer on one device. The boxed
/// `DynBackendStorage` is the actual byte holder; `Storage` is a thin
/// wrapper that gives the eager-dispatch op methods (matmul, conv,
/// unary, binary, …) somewhere to live.
///
/// Optionally carries a `bundle` side-table describing how the inner
/// buffer is partitioned into multiple logically independent outputs.
/// See [`OutputView`] and the module-level "Multi-output bundles" doc
/// for the contract.
///
/// We do not implement `Clone` because cloning storage may fail on
/// out-of-memory; use [`Self::try_clone`] for the fallible version.
#[derive(Debug)]
pub struct Storage {
    pub(crate) inner:  Box<dyn DynBackendStorage>,
    /// `None` for single-output storage (today's default). `Some(_)`
    /// for multi-output bundles: a shared Arc'd slice of per-slot
    /// [`OutputView`] entries, one per logical output. `Op::View`
    /// nodes share this Arc so the bundle stays alive as long as any
    /// view holds a reference.
    pub(crate) bundle: Option<Arc<[OutputView]>>,
    /// How the bytes are ENCODED (orthogonal to the logical element type
    /// reported by `inner.dtype_dyn()`). Empty = plain dense dtype,
    /// byte-identical to pre-SType behaviour. v1: PRIMARY storage only —
    /// bundle slots keep `dtype` only (per-slot SType is a future
    /// addition). See [`crate::SType`] and `docs/specs/storage-encoding.md`.
    pub(crate) stype: SType,
}

impl Storage {
    /// Construct storage from any concrete `DynBackendStorage` implementor.
    ///
    /// This is the backend-agnostic entry point — backends provide a type
    /// implementing `DynBackendStorage`, and `Storage::new` boxes it.
    /// Single-output (no bundle metadata); use
    /// [`Storage::new_bundled`] / [`Storage::with_bundle`] for the
    /// multi-output case.
    pub fn new<B: DynBackendStorage + 'static>(b: B) -> Self {
        Storage { inner: Box::new(b), bundle: None, stype: SType::default() }
    }

    /// Wrap an already-boxed `dyn DynBackendStorage`. Used by callers
    /// (notably the quantized fast-paths) that produce a `Box<dyn ..>`
    /// directly from trait dispatch. Single-output; use
    /// [`Storage::from_dyn_bundled`] for the multi-output case.
    pub fn from_dyn(b: Box<dyn DynBackendStorage>) -> Self {
        Storage { inner: b, bundle: None, stype: SType::default() }
    }

    /// Wrap an already-boxed `dyn DynBackendStorage` together with a
    /// per-slot bundle side-table. Used by multi-output op authors:
    /// the backend allocates one bundled byte buffer (whose
    /// `dtype_dyn()` is the primary/slot-0 dtype), and the caller
    /// attaches the slot metadata here.
    ///
    /// Validates at construction time that
    /// `bundle[0].dtype == inner.dtype_dyn()` so callers can trust
    /// `Storage::dtype()` ≡ `Storage::primary_dtype()` ≡
    /// `Storage::slot_dtype(0)` on a bundled storage.
    pub fn from_dyn_bundled(
        b:      Box<dyn DynBackendStorage>,
        bundle: Arc<[OutputView]>,
    ) -> Result<Self> {
        if bundle.is_empty() {
            return Err(Error::Msg(
                "Storage::from_dyn_bundled: bundle slice must be non-empty"
                    .into(),
            )
            .bt());
        }
        let primary = bundle[0].dtype;
        let inner_dtype = b.dtype_dyn();
        if primary != inner_dtype {
            return Err(Error::Msg(format!(
                "Storage::from_dyn_bundled: slot 0 dtype {primary:?} must \
                 match inner backend dtype {inner_dtype:?}",
            ))
            .bt());
        }
        Ok(Storage { inner: b, bundle: Some(bundle), stype: SType::default() })
    }

    /// Attach a bundle side-table to an existing single-output
    /// `Storage`. The dtype check from [`Storage::from_dyn_bundled`]
    /// applies. Panics in `debug_assertions` mode if a bundle is
    /// already attached — re-bundling silently is a contract bug.
    pub fn with_bundle(mut self, bundle: Arc<[OutputView]>) -> Result<Self> {
        debug_assert!(
            self.bundle.is_none(),
            "Storage::with_bundle: bundle already attached",
        );
        if bundle.is_empty() {
            return Err(Error::Msg(
                "Storage::with_bundle: bundle slice must be non-empty"
                    .into(),
            )
            .bt());
        }
        let primary = bundle[0].dtype;
        let inner_dtype = self.inner.dtype_dyn();
        if primary != inner_dtype {
            return Err(Error::Msg(format!(
                "Storage::with_bundle: slot 0 dtype {primary:?} must \
                 match inner backend dtype {inner_dtype:?}",
            ))
            .bt());
        }
        self.bundle = Some(bundle);
        Ok(self)
    }

    /// Whether this storage carries multi-output bundle metadata.
    pub fn is_bundled(&self) -> bool {
        self.bundle.is_some()
    }

    /// Number of logical output slots in this storage. `1` for
    /// single-output storage; for a bundled storage, the number of
    /// [`OutputView`] entries.
    pub fn slot_count(&self) -> usize {
        match &self.bundle {
            Some(b) => b.len(),
            None => 1,
        }
    }

    /// Borrow the full bundle slice, or `None` if this is a
    /// single-output storage.
    pub fn bundle(&self) -> Option<&[OutputView]> {
        self.bundle.as_deref()
    }

    /// Clone the `Arc<[OutputView]>` handle. Used by `Op::View`'s
    /// realization path: a View's output storage shares this Arc so
    /// the bundle stays alive as long as any view holds a reference.
    pub fn bundle_arc(&self) -> Option<Arc<[OutputView]>> {
        self.bundle.clone()
    }

    /// Per-slot view for `idx`. Returns `None` for out-of-range
    /// indices or for single-output storage with `idx != 0`. For
    /// single-output storage with `idx == 0` this returns `None` as
    /// well — the slot's dtype/shape live in the (`Storage::dtype`,
    /// caller-supplied Layout) pair, NOT in a synthetic `OutputView`.
    pub fn slot_view(&self, idx: usize) -> Option<&OutputView> {
        self.bundle.as_deref().and_then(|b| b.get(idx))
    }

    /// Primary slot's dtype — for a bundled storage this is slot 0's
    /// dtype (enforced equal to the inner backend dtype at construction
    /// time); for single-output storage this is just the inner dtype.
    /// Use [`Self::slot_dtype`] when the consumer is talking about a
    /// specific bundle slot, not the primary one.
    pub fn primary_dtype(&self) -> DType {
        self.inner.dtype_dyn()
    }

    /// Dtype of a specific bundle slot. Returns `None` for
    /// out-of-range indices or for single-output storage. For
    /// single-output storage, callers should use [`Self::dtype`]
    /// directly.
    pub fn slot_dtype(&self, idx: usize) -> Option<DType> {
        self.slot_view(idx).map(|v| v.dtype)
    }

    /// Borrow the inner storage as a `DynBackendStorage` trait object.
    ///
    /// Backends that need to peel back to their concrete storage type can
    /// downcast via `storage.as_dyn().as_any().downcast_ref::<MyStorage>()`.
    pub fn as_dyn(&self) -> &dyn DynBackendStorage {
        &*self.inner
    }

    /// Mutable variant of [`as_dyn`].
    pub fn as_dyn_mut(&mut self) -> &mut dyn DynBackendStorage {
        &mut *self.inner
    }

    /// Downcast the inner storage to a concrete backend type.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }

    /// Mutable variant of [`downcast_ref`](Self::downcast_ref).
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner.as_any_mut().downcast_mut::<T>()
    }

    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        // Preserve the encoding scheme across a clone (v1: cloning a plain
        // storage stays plain since `from_dyn` defaults empty; cloning an
        // *encoded* storage must carry its `stype` forward).
        Ok(Storage::from_dyn(self.inner.try_clone_dyn(layout)?).with_stype(self.stype.clone()))
    }

    /// Return an `Arc` to the owning device as a trait object.
    /// fuel-core wraps this in its `Device` newtype; other consumers
    /// (fuel-graph, tests) can use the trait object directly.
    pub fn device(&self) -> Arc<dyn DynBackendDevice> {
        self.inner.device_arc_dyn()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype_dyn()
    }

    /// Attach an encoding scheme to this storage (consuming builder).
    /// Does not touch the bytes or the logical dtype — only describes HOW
    /// the bytes are encoded. See [`crate::SType`].
    pub fn with_stype(mut self, stype: SType) -> Self {
        self.stype = stype;
        self
    }

    /// The encoding scheme. Empty = plain dense dtype.
    pub fn stype(&self) -> &SType {
        &self.stype
    }

    /// Pre-G this method consulted `Device::same_device` for the Metal
    /// pointer-identity check; post-G it goes through the
    /// `DynBackendDevice::same_device_dyn` trait method which has the
    /// same semantics.
    pub fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs_dev = self.inner.device_dyn();
        let rhs_dev = rhs.inner.device_dyn();
        let lhs_loc = lhs_dev.location_dyn();
        let rhs_loc = rhs_dev.location_dyn();
        let same = if matches!(lhs_loc, crate::DeviceLocation::Metal { .. }) {
            // On metal, require physical identity (matches pre-G behaviour).
            lhs_dev.same_device_dyn(rhs_dev)
        } else {
            lhs_loc == rhs_loc
        };
        if !same {
            Err(Error::DeviceMismatchBinaryOp { lhs: lhs_loc, rhs: rhs_loc, op }.bt())
        } else {
            Ok(())
        }
    }

    pub fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub fn const_set(&mut self, v: Scalar, l: &Layout) -> Result<()> {
        self.inner.const_set_dyn(v, l)
    }

    pub fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.affine_dyn(layout, mul, add)?))
    }

    pub fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.powf_dyn(layout, e)?))
    }

    pub fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.elu_dyn(layout, alpha)?))
    }

    pub fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.cmp_dyn(op, &*rhs.inner, lhs_layout, rhs_layout)?))
    }

    pub fn reduce_op(
        &self,
        op: ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.reduce_op_dyn(op, layout, reduce_dims)?))
    }

    pub fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.to_dtype_dyn(layout, dtype)?))
    }

    pub fn to_cpu_storage(&self) -> Result<HostBuffer> {
        self.inner.to_host_buffer_dyn()
    }

    pub fn inplace_op1(&mut self, l: &Layout, c: &dyn InplaceOp1) -> Result<()> {
        c.fwd(&mut *self.inner, l)
    }

    pub fn inplace_op2(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn InplaceOp2,
    ) -> Result<()> {
        self.same_device(t2, c.name())?;
        c.fwd(&mut *self.inner, l1, &*t2.inner, l2)
    }

    pub fn inplace_op3(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn InplaceOp3,
    ) -> Result<()> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        c.fwd(&mut *self.inner, l1, &*t2.inner, l2, &*t3.inner, l3)
    }

    // -----------------------------------------------------------------------
    // Unary / Binary dispatch
    // -----------------------------------------------------------------------

    pub fn unary_impl<B: op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op = op::UnaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown unary op '{}'", B::NAME))
        })?;
        Ok(Storage::from_dyn(self.inner.unary_op_dyn(layout, op)?))
    }

    pub fn binary_impl<B: op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        let op = BinaryOp::from_name(B::NAME).ok_or_else(|| {
            Error::Msg(format!("unknown binary op '{}'", B::NAME))
        })?;
        Ok(Storage::from_dyn(self.inner.binary_op_dyn(&*rhs.inner, lhs_layout, rhs_layout, op)?))
    }

    // -----------------------------------------------------------------------
    // Convolutions, pooling, upsampling
    // -----------------------------------------------------------------------

    pub fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConv1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv1d")?;
        self.same_dtype(kernel, "conv1d")?;
        Ok(Storage::from_dyn(self.inner.conv1d_dyn(l, &*kernel.inner, kernel_l, params)?))
    }

    pub fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv-transpose1d")?;
        self.same_dtype(kernel, "conv-transpose1d")?;
        Ok(Storage::from_dyn(self.inner.conv_transpose1d_dyn(l, &*kernel.inner, kernel_l, params)?))
    }

    pub fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConv2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv2d")?;
        self.same_dtype(kernel, "conv2d")?;
        Ok(Storage::from_dyn(self.inner.conv2d_dyn(l, &*kernel.inner, kernel_l, params)?))
    }

    pub fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv_transpose2d")?;
        self.same_dtype(kernel, "conv_transpose2d")?;
        Ok(Storage::from_dyn(self.inner.conv_transpose2d_dyn(l, &*kernel.inner, kernel_l, params)?))
    }

    pub fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.avg_pool2d_dyn(layout, kernel_size, stride)?))
    }

    pub fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.max_pool2d_dyn(layout, kernel_size, stride)?))
    }

    pub fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.upsample_nearest1d_dyn(layout, sz)?))
    }

    pub fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.upsample_nearest2d_dyn(layout, h, w)?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        Ok(Storage::from_dyn(self.inner.upsample_bilinear2d_dyn(layout, h, w, align_corners, scale_h, scale_w)?))
    }

    // -----------------------------------------------------------------------
    // Gather / Scatter / Index
    // -----------------------------------------------------------------------

    pub fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        self.same_device(t, "where")?;
        self.same_device(f, "where")?;
        t.same_dtype(f, "where")?;
        Ok(Storage::from_dyn(self.inner.where_cond_dyn(layout, &*t.inner, layout_t, &*f.inner, layout_f)?))
    }

    pub fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        Ok(Storage::from_dyn(self.inner.gather_dyn(l, &*indexes.inner, indexes_l, d)?))
    }

    pub fn scatter_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-set")?;
        self.same_device(source, "scatter-set")?;
        self.inner.scatter_set_dyn(l, &*source.inner, source_l, &*indexes.inner, indexes_l, d)
    }

    pub fn scatter_add(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-add")?;
        self.same_device(source, "scatter-add")?;
        self.inner.scatter_add_set_dyn(l, &*source.inner, source_l, &*indexes.inner, indexes_l, d)
    }

    pub fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        self.same_device(source, "index-add")?;
        Ok(Storage::from_dyn(self.inner.index_add_dyn(l, &*indexes.inner, indexes_l, &*source.inner, source_l, d)?))
    }

    pub fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        Ok(Storage::from_dyn(self.inner.index_select_dyn(&*rhs.inner, lhs_l, rhs_l, d)?))
    }

    // -----------------------------------------------------------------------
    // Matmul and copy
    // -----------------------------------------------------------------------

    pub fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        Ok(Storage::from_dyn(self.inner.matmul_dyn(&*rhs.inner, bmnk, lhs_layout, rhs_layout)?))
    }

    /// `self`, the source, can be strided whereas `dst` is contiguous.
    pub fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        self.inner.copy_strided_src_dyn(&mut *dst.inner, dst_offset, src_l)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.inner.copy2d_dyn(&mut *dst.inner, d1, d2, src_s, dst_s, src_o, dst_o)
    }
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

#[cfg(test)]
mod stype_attach {
    //! Step 2 born-red coverage for the trait-object `Storage`'s `stype`
    //! field. There is no concrete `DynBackendStorage` impl in this crate
    //! (backends live downstream and would cycle as a dev-dep), so the test
    //! uses a minimal mock whose only real method is `try_clone_dyn`.
    use super::*;
    use crate::op::UnaryOp;
    use std::any::Any;

    /// Minimal `DynBackendStorage`: an empty unit struct. Only
    /// `try_clone_dyn` / `dtype_dyn` / `as_any*` are exercised; every other
    /// method panics (never called by these tests).
    #[derive(Debug)]
    struct MockStorage;

    #[allow(unused_variables)]
    impl DynBackendStorage for MockStorage {
        fn try_clone_dyn(&self, _layout: &Layout) -> Result<Box<dyn DynBackendStorage>> {
            Ok(Box::new(MockStorage))
        }
        fn dtype_dyn(&self) -> DType { DType::F32 }
        fn as_any(&self) -> &dyn Any { self }
        fn as_any_mut(&mut self) -> &mut dyn Any { self }

        fn device_dyn(&self) -> &dyn DynBackendDevice { unimplemented!() }
        fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice> { unimplemented!() }
        fn to_host_buffer_dyn(&self) -> Result<HostBuffer> { unimplemented!() }
        fn affine_dyn(&self, l: &Layout, mul: f64, add: f64) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn powf_dyn(&self, l: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn elu_dyn(&self, l: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn reduce_op_dyn(&self, op: ReduceOp, l: &Layout, axes: &[usize]) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn cmp_dyn(&self, op: CmpOp, rhs: &dyn DynBackendStorage, ll: &Layout, rl: &Layout) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn to_dtype_dyn(&self, l: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn unary_op_dyn(&self, l: &Layout, op: UnaryOp) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn binary_op_dyn(&self, rhs: &dyn DynBackendStorage, ll: &Layout, rl: &Layout, op: BinaryOp) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn where_cond_dyn(&self, cl: &Layout, t: &dyn DynBackendStorage, tl: &Layout, f: &dyn DynBackendStorage, fl: &Layout) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn conv1d_dyn(&self, l: &Layout, k: &dyn DynBackendStorage, kl: &Layout, p: &conv::ParamsConv1D) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn conv_transpose1d_dyn(&self, l: &Layout, k: &dyn DynBackendStorage, kl: &Layout, p: &conv::ParamsConvTranspose1D) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn conv2d_dyn(&self, l: &Layout, k: &dyn DynBackendStorage, kl: &Layout, p: &conv::ParamsConv2D) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn conv_transpose2d_dyn(&self, l: &Layout, k: &dyn DynBackendStorage, kl: &Layout, p: &conv::ParamsConvTranspose2D) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn avg_pool2d_dyn(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn max_pool2d_dyn(&self, l: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn upsample_nearest1d_dyn(&self, l: &Layout, t: usize) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn upsample_nearest2d_dyn(&self, l: &Layout, h: usize, w: usize) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn upsample_bilinear2d_dyn(&self, l: &Layout, h: usize, w: usize, ac: bool, sh: Option<f64>, sw: Option<f64>) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn gather_dyn(&self, sl: &Layout, ids: &dyn DynBackendStorage, il: &Layout, dim: usize) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn scatter_set_dyn(&mut self, sl: &Layout, src: &dyn DynBackendStorage, srl: &Layout, ids: &dyn DynBackendStorage, il: &Layout, dim: usize) -> Result<()> { unimplemented!() }
        fn scatter_add_set_dyn(&mut self, sl: &Layout, src: &dyn DynBackendStorage, srl: &Layout, ids: &dyn DynBackendStorage, il: &Layout, dim: usize) -> Result<()> { unimplemented!() }
        fn index_select_dyn(&self, ids: &dyn DynBackendStorage, sl: &Layout, il: &Layout, dim: usize) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn index_add_dyn(&self, sl: &Layout, ids: &dyn DynBackendStorage, il: &Layout, src: &dyn DynBackendStorage, srl: &Layout, dim: usize) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn matmul_dyn(&self, rhs: &dyn DynBackendStorage, bmnk: (usize, usize, usize, usize), ll: &Layout, rl: &Layout) -> Result<Box<dyn DynBackendStorage>> { unimplemented!() }
        fn copy_strided_src_dyn(&self, dst: &mut dyn DynBackendStorage, off: usize, sl: &Layout) -> Result<()> { unimplemented!() }
        fn copy2d_dyn(&self, dst: &mut dyn DynBackendStorage, d1: usize, d2: usize, ss1: usize, ds1: usize, so: usize, dofs: usize) -> Result<()> { unimplemented!() }
        fn const_set_dyn(&mut self, value: Scalar, l: &Layout) -> Result<()> { unimplemented!() }
    }

    /// Born-red: a freshly constructed trait-object Storage carries a plain
    /// (empty) SType by default.
    #[test]
    fn trait_object_storage_defaults_plain() {
        let s = Storage::new(MockStorage);
        assert!(s.stype().is_plain(), "default Storage must carry a plain SType");
        assert_eq!(s.stype().layers().len(), 0);
    }

    /// Born-red: `try_clone` carries an attached `stype` forward (the cheap,
    /// correct v1 choice — cloning an encoded storage preserves its scheme).
    #[test]
    fn try_clone_preserves_stype() {
        use crate::stype::Encoding;
        use crate::quantized::GgmlDType;
        let s = Storage::new(MockStorage)
            .with_stype(SType::from_layer(Encoding::GgmlBlock { ggml_dtype: GgmlDType::Q4_0 }));
        let layout = Layout::contiguous(Shape::from_dims(&[4]));
        let cloned = s.try_clone(&layout).expect("clone");
        assert_eq!(cloned.stype(), s.stype(), "try_clone must preserve stype");
        assert!(!cloned.stype().is_plain());
    }
}

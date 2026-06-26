//! Shapes describe the dimensionality of tensors.
//!
//! A [`Shape`] holds an ordered list of dimension sizes. It is used to create tensors,
//! verify operation compatibility, and query tensor structure.
#![allow(clippy::redundant_closure_call)]
use crate::symbol::{SymEnv, SymId};
use crate::{DimVec, Error, Result};
use smallvec::smallvec;

/// A single dimension's **extent**: a build-time constant, or a bounded
/// runtime symbol.
///
/// `Shape::dims()` reports the *bound* (a `Scalar`'s value, or a `Range`'s
/// `max`/capacity) — which is what sizing/striding/iteration want. The *live*
/// value of a `Range` is resolved per forward pass through a [`SymEnv`].
/// `Scalar` carries no symbol: two build-time constants that must match
/// already match by being equal, and a *runtime* dimension always has a
/// capacity bound, so it is a `Range`, never a `Scalar`.
///
/// Phase D step 1b. See
/// `docs/session-prompts/symbolic-extents-and-persistent-decode.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extent {
    /// A dimension fixed at graph-construction time.
    Scalar(usize),
    /// A bounded runtime dimension: live value in `[min, max]`, resolved via
    /// the `SymEnv` under `sym`. `max` is the capacity (what `dims()` reports).
    Range { min: usize, max: usize, sym: SymId },
}

impl Extent {
    /// The capacity bound: a `Scalar`'s value, or a `Range`'s `max`. This is
    /// what `Shape::dims()` reports for the axis.
    pub fn bound(&self) -> usize {
        match self {
            Extent::Scalar(v) => *v,
            Extent::Range { max, .. } => *max,
        }
    }

    /// The lower bound: a `Scalar`'s value, or a `Range`'s `min`.
    pub fn min(&self) -> usize {
        match self {
            Extent::Scalar(v) => *v,
            Extent::Range { min, .. } => *min,
        }
    }

    /// Whether this is a runtime (`Range`) extent rather than a constant.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Extent::Range { .. })
    }

    /// The symbol of a `Range`, or `None` for a `Scalar`.
    pub fn sym(&self) -> Option<SymId> {
        match self {
            Extent::Scalar(_) => None,
            Extent::Range { sym, .. } => Some(*sym),
        }
    }

    /// The live value this pass: a `Scalar`'s value, or the `Range`'s `sym`
    /// resolved through `env`. `None` if a `Range`'s symbol is unbound.
    pub fn resolve(&self, env: &SymEnv) -> Option<usize> {
        match self {
            Extent::Scalar(v) => Some(*v),
            Extent::Range { sym, .. } => env.get(*sym),
        }
    }
}

/// A sparse record marking one axis of a [`Shape`] as a bounded-symbolic
/// `Range`. `max` is the corresponding `Shape::dims()` bound, so only `min`
/// and `sym` are stored here. Phase D step 1b.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynAxis {
    pub axis: usize,
    pub min: usize,
    pub sym: SymId,
}

/// Sparse list of a shape's dynamic axes, kept sorted by `axis` so equality is
/// structural. Empty/`None` for the common all-concrete shape.
type DynAxisVec = smallvec::SmallVec<[DynAxis; 2]>;

/// A shape represents the dimensions of a tensor.
///
/// Internally backed by a [`SmallVec`](smallvec::SmallVec) that stores up to 6 dimensions
/// on the stack without heap allocation. A scalar has rank 0 (no dimensions), a vector has
/// rank 1, a matrix has rank 2, and so on.
///
/// `Shape` can be created from tuples, slices, `Vec<usize>`, or the [`Shape::from_dims`]
/// constructor.
/// `dims` holds the per-axis **bounds** (a `Scalar`'s value, or a `Range`'s
/// `max`/capacity); `dynamic` sparsely marks which axes are bounded-symbolic.
/// `dims()` borrows `dims` unchanged; the symbolic view is `extent()`. The
/// common all-concrete shape has `dynamic: None`. Equality includes `dynamic`
/// (a symbolic shape is distinct from a concrete one with the same bounds), so
/// `DynAxis` entries are kept sorted by axis.
#[derive(Clone, PartialEq, Eq)]
pub struct Shape {
    dims: DimVec,
    dynamic: Option<DynAxisVec>,
}

/// A constant representing a scalar shape (rank 0, no dimensions).
pub const SCALAR: Shape = Shape {
    dims: smallvec::SmallVec::new_const(),
    dynamic: None,
};

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", &self.dims())?;
        if let Some(dyn_axes) = &self.dynamic {
            if !dyn_axes.is_empty() {
                write!(f, "{{dyn: {dyn_axes:?}}}")?;
            }
        }
        Ok(())
    }
}

impl<const C: usize> From<&[usize; C]> for Shape {
    fn from(dims: &[usize; C]) -> Self {
        Self { dims: DimVec::from_slice(dims), dynamic: None }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self { dims: DimVec::from_slice(dims), dynamic: None }
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        Self { dims: shape.dims.clone(), dynamic: shape.dynamic.clone() }
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self { dims: smallvec![], dynamic: None }
    }
}

impl From<usize> for Shape {
    fn from(d1: usize) -> Self {
        Self { dims: smallvec![d1], dynamic: None }
    }
}

macro_rules! impl_from_tuple {
    ($tuple:ty, $($index:tt),+) => {
        impl From<$tuple> for Shape {
            fn from(d: $tuple) -> Self {
                Self { dims: smallvec![$(d.$index,)+], dynamic: None }
            }
        }
    }
}

impl_from_tuple!((usize,), 0);
impl_from_tuple!((usize, usize), 0, 1);
impl_from_tuple!((usize, usize, usize), 0, 1, 2);
impl_from_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_from_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_from_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self { dims: DimVec::from(dims), dynamic: None }
    }
}

impl From<DimVec> for Shape {
    fn from(dims: DimVec) -> Self {
        Self { dims, dynamic: None }
    }
}

/// Macro that generates dimension extraction functions, Shape methods, and TryInto impls.
///
/// For each invocation, creates:
/// - A free function `$fn_name(dims: &[usize]) -> Result<$out_type>`
/// - An `impl Shape` method with the same name
/// - A `TryInto<$out_type> for Shape` impl
#[macro_export]
macro_rules! extract_dims {
    ($(#[$fn_meta:meta])* $fn_name:ident, $cnt:tt, $dims:expr, $out_type:ty,
     $(#[$shape_meta:meta])* shape) => {
        $(#[$fn_meta])*
        pub fn $fn_name(dims: &[usize]) -> $crate::Result<$out_type> {
            if dims.len() != $cnt {
                Err($crate::Error::UnexpectedNumberOfDims {
                    expected: $cnt,
                    got: dims.len(),
                    shape: $crate::Shape::from(dims),
                }
                .bt())
            } else {
                Ok($dims(dims))
            }
        }

        impl $crate::Shape {
            $(#[$shape_meta])*
            pub fn $fn_name(&self) -> $crate::Result<$out_type> {
                $fn_name(self.dims())
            }
        }

        impl std::convert::TryInto<$out_type> for $crate::Shape {
            type Error = $crate::Error;
            fn try_into(self) -> std::result::Result<$out_type, Self::Error> {
                self.$fn_name()
            }
        }
    };
}

impl Shape {
    /// Creates a shape from a slice of dimension sizes.
    pub fn from_dims(dims: &[usize]) -> Self {
        Self { dims: DimVec::from_slice(dims), dynamic: None }
    }

    /// The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Consumes the shape and returns its dimensions as a `Vec<usize>`.
    pub fn into_dims(self) -> Vec<usize> {
        self.dims.into_vec()
    }

    /// The dimensions as a slice of `usize` — the per-axis **bounds** (a
    /// scalar's value, or a dynamic axis's capacity/`max`). Unchanged contract:
    /// this is what sizing, striding, and iteration want. The live value of a
    /// dynamic axis is obtained separately via [`Shape::extent`] +
    /// [`Extent::resolve`] (or [`Shape::resolve`]); it is never folded in here.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The [`Extent`] of `axis`: a `Range` if the axis is dynamic (carrying its
    /// `min`, the `dims()` bound as `max`, and its `sym`), else a `Scalar` of
    /// the bound. The symbolic *view* over `dims()`. Phase D step 1b.
    pub fn extent(&self, axis: usize) -> Extent {
        if let Some(dyn_axes) = &self.dynamic {
            if let Some(da) = dyn_axes.iter().find(|d| d.axis == axis) {
                return Extent::Range { min: da.min, max: self.dims[axis], sym: da.sym };
            }
        }
        Extent::Scalar(self.dims[axis])
    }

    /// The [`Extent`] of every axis, in order.
    pub fn extents(&self) -> impl Iterator<Item = Extent> + '_ {
        (0..self.dims.len()).map(move |i| self.extent(i))
    }

    /// Whether any axis is a bounded-symbolic `Range`.
    pub fn has_dynamic(&self) -> bool {
        self.dynamic.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// Mark `axis` as a bounded-symbolic `Range` `[min, dims[axis]]` resolved
    /// via `sym`. `dims()[axis]` stays the capacity bound; `extent(axis)` now
    /// reports the `Range`. Replaces any prior dynamic mark on `axis`. The
    /// KV-cache length axis uses this; K-length and V-length pass the *same*
    /// `sym` to unify.
    pub fn with_dynamic_axis(mut self, axis: usize, min: usize, sym: SymId) -> Self {
        assert!(
            axis < self.dims.len(),
            "with_dynamic_axis: axis {axis} out of range for rank {}",
            self.dims.len(),
        );
        let da = DynAxis { axis, min, sym };
        let v = self.dynamic.get_or_insert_with(DynAxisVec::new);
        // Keep sorted by axis so equality is structural.
        match v.binary_search_by_key(&axis, |d| d.axis) {
            Ok(i) => v[i] = da,
            Err(i) => v.insert(i, da),
        }
        self
    }

    /// Resolve every dynamic axis through `env`, returning a fully-concrete
    /// `Shape` (all `Scalar`). Reads `env` at call time, so it always reflects
    /// the current bindings; never caches. Errors if a dynamic axis's symbol
    /// is unbound. A shape with no dynamic axes resolves to a clone.
    pub fn resolve(&self, env: &SymEnv) -> Result<Shape> {
        let Some(dyn_axes) = &self.dynamic else {
            return Ok(self.clone());
        };
        let mut dims = self.dims.clone();
        for da in dyn_axes {
            match env.get(da.sym) {
                Some(v) => dims[da.axis] = v,
                None => {
                    return Err(Error::Msg(format!(
                        "Shape::resolve: dynamic axis {} symbol {:?} is unbound",
                        da.axis, da.sym,
                    ))
                    .bt());
                }
            }
        }
        Ok(Shape { dims, dynamic: None })
    }

    /// The dimension size for a specified dimension index.
    ///
    /// Supports both forward indexing (`usize`) and backward indexing via [`D`].
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self, "dim")?;
        Ok(self.dims()[dim])
    }

    /// The total number of elements, this is the product of all dimension sizes.
    ///
    /// For a scalar (rank 0), the element count is 1.
    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// The strides given in number of elements for a contiguous n-dimensional
    /// array using this shape. Returns signed strides
    /// ([`crate::StrideVec`]) — a contiguous layout's strides are
    /// always non-negative, but the type is signed so callers can
    /// uniformly handle negative-stride view ops (Flip, etc.).
    pub fn stride_contiguous(&self) -> crate::StrideVec {
        let mut stride: crate::StrideVec = self
            .dims
            .iter()
            .rev()
            .scan(1_isize, |prod, &u| {
                let prod_pre_mult = *prod;
                *prod *= u as isize;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    /// Returns true if the strides are C contiguous (aka row major).
    pub fn is_contiguous(&self, stride: &[isize]) -> bool {
        if self.dims.len() != stride.len() {
            return false;
        }
        let mut acc: isize = 1;
        for (&stride, &dim) in stride.iter().zip(self.dims.iter()).rev() {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim as isize;
        }
        true
    }

    /// Returns true if the strides are Fortran contiguous (aka column major).
    pub fn is_fortran_contiguous(&self, stride: &[isize]) -> bool {
        if self.dims.len() != stride.len() {
            return false;
        }
        let mut acc: isize = 1;
        for (&stride, &dim) in stride.iter().zip(self.dims.iter()) {
            if dim > 1 && stride != acc {
                return false;
            }
            acc *= dim as isize;
        }
        true
    }

    /// Modifies the shape by adding a list of additional dimensions at the end of the existing
    /// dimensions.
    pub fn extend(mut self, additional_dims: &[usize]) -> Self {
        // Appending axes leaves existing axis indices (and thus `dynamic`)
        // valid; the new trailing axes are concrete.
        self.dims.extend_from_slice(additional_dims);
        self
    }

    /// Check whether the two shapes are compatible for broadcast, and if it is the case return the
    /// broadcasted shape. This is to be used for binary pointwise ops.
    pub fn broadcast_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = smallvec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp {
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    op,
                }
                .bt())?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    /// Returns the broadcasted shapes for matrix multiplication.
    pub fn broadcast_shape_matmul(&self, rhs: &Self) -> Result<(Shape, Shape)> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
            crate::bail!("only 2d matrixes are supported {lhs:?} {rhs:?}")
        }
        let (m, lhs_k) = (lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]);
        let (rhs_k, n) = (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]);
        if lhs_k != rhs_k {
            crate::bail!("different inner dimensions in broadcast matmul {lhs:?} {rhs:?}")
        }

        let lhs_b = Self::from(&lhs_dims[..lhs_dims.len() - 2]);
        let rhs_b = Self::from(&rhs_dims[..rhs_dims.len() - 2]);
        let bcast = lhs_b.broadcast_shape_binary_op(&rhs_b, "broadcast_matmul")?;
        let bcast_dims = bcast.dims();

        let bcast_lhs = [bcast_dims, &[m, lhs_k]].concat();
        let bcast_rhs = [bcast_dims, &[rhs_k, n]].concat();
        Ok((Shape::from(bcast_lhs), Shape::from(bcast_rhs)))
    }
}

/// A trait for types that can be used as dimension indices.
///
/// This is implemented for `usize` (forward indexing) and [`D`] (backward indexing).
pub trait Dim {
    /// Converts this dimension reference to a concrete index.
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize>;
    /// Converts this dimension reference to an index, allowing one-past-the-end for insertion.
    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim > shape.dims().len() {
            Err(Error::DimOutOfRange {
                shape: shape.clone(),
                dim: dim as i32,
                op,
            }
            .bt())?
        } else {
            Ok(dim)
        }
    }
}

/// Dimension indexing from the right (end) of a shape.
///
/// `D::Minus1` refers to the last dimension, `D::Minus2` to the second-to-last, and
/// `D::Minus(n)` to the n-th from the end.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    /// The last dimension.
    Minus1,
    /// The second-to-last dimension.
    Minus2,
    /// The n-th dimension from the end.
    Minus(usize),
}

impl D {
    fn out_of_range(&self, shape: &Shape, op: &'static str) -> Error {
        let dim = match self {
            Self::Minus1 => -1,
            Self::Minus2 => -2,
            Self::Minus(u) => -(*u as i32),
        };
        Error::DimOutOfRange {
            shape: shape.clone(),
            dim,
            op,
        }
        .bt()
    }
}

impl Dim for D {
    fn to_index(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 if rank >= 1 => Ok(rank - 1),
            Self::Minus2 if rank >= 2 => Ok(rank - 2),
            Self::Minus(u) if *u > 0 && rank >= *u => Ok(rank - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }

    fn to_index_plus_one(&self, shape: &Shape, op: &'static str) -> Result<usize> {
        let rank = shape.rank();
        match self {
            Self::Minus1 => Ok(rank),
            Self::Minus2 if rank >= 1 => Ok(rank - 1),
            Self::Minus(u) if *u > 0 && rank + 1 >= *u => Ok(rank + 1 - *u),
            _ => Err(self.out_of_range(shape, op)),
        }
    }
}

/// A trait for types representing multiple dimension indices.
///
/// Used by operations that act on multiple axes simultaneously.
pub trait Dims: Sized {
    /// Converts the dimension references to concrete indices, checking for duplicates.
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>>;

    fn to_indexes(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dims = self.to_indexes_internal(shape, op)?;
        for (i, &dim) in dims.iter().enumerate() {
            if dims[..i].contains(&dim) {
                Err(Error::DuplicateDimIndex {
                    shape: shape.clone(),
                    dims: dims.clone(),
                    op,
                }
                .bt())?
            }
            if dim >= shape.rank() {
                Err(Error::DimOutOfRange {
                    shape: shape.clone(),
                    dim: dim as i32,
                    op,
                }
                .bt())?
            }
        }
        Ok(dims)
    }
}

impl Dims for Vec<usize> {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self)
    }
}

impl<const N: usize> Dims for [usize; N] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for &[usize] {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(self.to_vec())
    }
}

impl Dims for () {
    fn to_indexes_internal(self, _: &Shape, _: &'static str) -> Result<Vec<usize>> {
        Ok(vec![])
    }
}

impl<D: Dim + Sized> Dims for D {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D: Dim> Dims for (D,) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let dim = self.0.to_index(shape, op)?;
        Ok(vec![dim])
    }
}

impl<D1: Dim, D2: Dim> Dims for (D1, D2) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        Ok(vec![d0, d1])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim> Dims for (D1, D2, D3) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        Ok(vec![d0, d1, d2])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim> Dims for (D1, D2, D3, D4) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim> Dims for (D1, D2, D3, D4, D5) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3, d4])
    }
}

impl<D1: Dim, D2: Dim, D3: Dim, D4: Dim, D5: Dim, D6: Dim> Dims for (D1, D2, D3, D4, D5, D6) {
    fn to_indexes_internal(self, shape: &Shape, op: &'static str) -> Result<Vec<usize>> {
        let d0 = self.0.to_index(shape, op)?;
        let d1 = self.1.to_index(shape, op)?;
        let d2 = self.2.to_index(shape, op)?;
        let d3 = self.3.to_index(shape, op)?;
        let d4 = self.4.to_index(shape, op)?;
        let d5 = self.5.to_index(shape, op)?;
        Ok(vec![d0, d1, d2, d3, d4, d5])
    }
}

extract_dims!(
    /// Validates 0 dimensions (scalar).
    dims0, 0, |_: &[usize]| (), (),
    /// Extracts dimensions from a scalar (rank-0) shape.
    shape);

extract_dims!(
    /// Validates 1 dimension and returns its size.
    dims1, 1, |d: &[usize]| d[0], usize,
    /// Extracts the single dimension from a rank-1 shape.
    shape);

extract_dims!(
    /// Validates 2 dimensions and returns them as a tuple.
    dims2, 2, |d: &[usize]| (d[0], d[1]), (usize, usize),
    /// Extracts the two dimensions from a rank-2 shape.
    shape);

extract_dims!(
    /// Validates 3 dimensions and returns them as a tuple.
    dims3, 3, |d: &[usize]| (d[0], d[1], d[2]), (usize, usize, usize),
    /// Extracts the three dimensions from a rank-3 shape.
    shape);

extract_dims!(
    /// Validates 4 dimensions and returns them as a tuple.
    dims4, 4, |d: &[usize]| (d[0], d[1], d[2], d[3]), (usize, usize, usize, usize),
    /// Extracts the four dimensions from a rank-4 shape.
    shape);

extract_dims!(
    /// Validates 5 dimensions and returns them as a tuple.
    dims5, 5, |d: &[usize]| (d[0], d[1], d[2], d[3], d[4]), (usize, usize, usize, usize, usize),
    /// Extracts the five dimensions from a rank-5 shape.
    shape);

// Stride destructure helpers — sister set to dims2/3/4/5 above, but
// for [`crate::Layout::stride()`] which returns `&[isize]`. These cast
// to `usize` at the destructure boundary with a debug_assert that
// every stride is non-negative, since current kernels are written
// against unsigned-stride byte arithmetic. The Layout-side change to
// signed strides unlocks negative-stride view ops (e.g. Op::Flip);
// when those land, kernels that consume the resulting layouts
// directly should iterate via [`crate::StridedIndex`] (which handles
// signed) rather than reading raw stride bytes through these helpers.

/// Validates 2 strides and returns them as `(usize, usize)`.
pub fn stride_dims2(stride: &[isize]) -> Result<(usize, usize)> {
    if stride.len() != 2 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 2, got: stride.len(), shape: Shape::from(&[][..]),
        }.bt());
    }
    debug_assert!(stride.iter().all(|&s| s >= 0), "stride_dims2: negative stride");
    Ok((stride[0] as usize, stride[1] as usize))
}

/// Validates 3 strides and returns them as `(usize, usize, usize)`.
pub fn stride_dims3(stride: &[isize]) -> Result<(usize, usize, usize)> {
    if stride.len() != 3 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 3, got: stride.len(), shape: Shape::from(&[][..]),
        }.bt());
    }
    debug_assert!(stride.iter().all(|&s| s >= 0), "stride_dims3: negative stride");
    Ok((stride[0] as usize, stride[1] as usize, stride[2] as usize))
}

/// Validates 4 strides and returns them as `(usize, usize, usize, usize)`.
pub fn stride_dims4(stride: &[isize]) -> Result<(usize, usize, usize, usize)> {
    if stride.len() != 4 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 4, got: stride.len(), shape: Shape::from(&[][..]),
        }.bt());
    }
    debug_assert!(stride.iter().all(|&s| s >= 0), "stride_dims4: negative stride");
    Ok((stride[0] as usize, stride[1] as usize, stride[2] as usize, stride[3] as usize))
}

/// Validates 5 strides and returns them as `(usize, usize, usize, usize, usize)`.
pub fn stride_dims5(stride: &[isize]) -> Result<(usize, usize, usize, usize, usize)> {
    if stride.len() != 5 {
        return Err(Error::UnexpectedNumberOfDims {
            expected: 5, got: stride.len(), shape: Shape::from(&[][..]),
        }.bt());
    }
    debug_assert!(stride.iter().all(|&s| s >= 0), "stride_dims5: negative stride");
    Ok((stride[0] as usize, stride[1] as usize, stride[2] as usize, stride[3] as usize, stride[4] as usize))
}

/// A trait for shape specifications that may contain one unknown dimension marked with `()`.
///
/// When reshaping, you can use `()` as a placeholder for one dimension and the
/// library will infer its size from the total element count.
pub trait ShapeWithOneHole {
    /// Resolves the shape given the total element count, filling in the `()` hole.
    fn into_shape(self, el_count: usize) -> Result<Shape>;
}

impl<S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape> {
        Ok(self.into())
    }
}

impl ShapeWithOneHole for ((),) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        Ok(el_count.into())
    }
}

fn hole_size(el_count: usize, prod_d: usize, s: &dyn std::fmt::Debug) -> Result<usize> {
    if prod_d == 0 {
        crate::bail!("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if !el_count.is_multiple_of(prod_d) {
        crate::bail!("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    Ok(el_count / prod_d)
}

impl ShapeWithOneHole for ((), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1) = self;
        Ok((hole_size(el_count, d1, &self)?, d1).into())
    }
}

impl ShapeWithOneHole for (usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, ()) = self;
        Ok((d1, hole_size(el_count, d1, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2) = self;
        Ok((hole_size(el_count, d1 * d2, &self)?, d1, d2).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2) = self;
        Ok((d1, hole_size(el_count, d1 * d2, &self)?, d2).into())
    }
}

impl ShapeWithOneHole for (usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, ()) = self;
        Ok((d1, d2, hole_size(el_count, d1 * d2, &self)?).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d, d1, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d, d2, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d, d3).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3, &self)?;
        Ok((d1, d2, d3, d).into())
    }
}

impl ShapeWithOneHole for ((), usize, usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let ((), d1, d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d, d1, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, (), usize, usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, (), d2, d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d, d2, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, (), usize, usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, (), d3, d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d, d3, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, (), usize) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, (), d4) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d, d4).into())
    }
}

impl ShapeWithOneHole for (usize, usize, usize, usize, ()) {
    fn into_shape(self, el_count: usize) -> Result<Shape> {
        let (d1, d2, d3, d4, ()) = self;
        let d = hole_size(el_count, d1 * d2 * d3 * d4, &self)?;
        Ok((d1, d2, d3, d4, d).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride() {
        let shape = Shape::from(());
        assert_eq!(shape.stride_contiguous().to_vec(), Vec::<isize>::new());
        let shape = Shape::from(42);
        assert_eq!(shape.stride_contiguous().to_vec(), [1_isize]);
        let shape = Shape::from((42, 1337));
        assert_eq!(shape.stride_contiguous().to_vec(), [1337_isize, 1]);
        let shape = Shape::from((299, 792, 458));
        assert_eq!(shape.stride_contiguous().to_vec(), [458_isize * 792, 458, 1]);
    }

    #[test]
    fn test_from_tuple() {
        let shape = Shape::from((2,));
        assert_eq!(shape.dims(), &[2]);
        let shape = Shape::from((2, 3));
        assert_eq!(shape.dims(), &[2, 3]);
        let shape = Shape::from((2, 3, 4));
        assert_eq!(shape.dims(), &[2, 3, 4]);
        let shape = Shape::from((2, 3, 4, 5));
        assert_eq!(shape.dims(), &[2, 3, 4, 5]);
        let shape = Shape::from((2, 3, 4, 5, 6));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6]);
        let shape = Shape::from((2, 3, 4, 5, 6, 7));
        assert_eq!(shape.dims(), &[2, 3, 4, 5, 6, 7]);
    }

    /// PR step-1b: the symbolic-extent foundation. `dims()` returns the
    /// capacity bounds unchanged; `extent()` is the symbolic view;
    /// `resolve()` substitutes live values (erroring when unbound); two shapes
    /// sharing a `sym` resolve together; equality includes `dynamic`; `Layout`
    /// inherits `Extent` and resolves while keeping capacity strides. Born-red:
    /// fails if any of those contracts is wrong.
    #[test]
    fn symbolic_extent_foundation() {
        use crate::{Layout, SymEnv, SymId};

        // A capacity-shaped KV buffer with a dynamic length axis (axis 2).
        let sym = SymId(0);
        let shape = Shape::from_dims(&[1, 32, 4096, 128]).with_dynamic_axis(2, 0, sym);

        // dims() returns the BOUNDS (axis 2 = capacity 4096), contract unchanged.
        assert_eq!(shape.dims(), &[1, 32, 4096, 128]);
        assert!(shape.has_dynamic());

        // extent() is the symbolic view.
        assert_eq!(shape.extent(0), Extent::Scalar(1));
        assert_eq!(shape.extent(2), Extent::Range { min: 0, max: 4096, sym });
        assert_eq!(shape.extent(2).bound(), 4096);
        assert!(shape.extent(2).is_dynamic());
        assert_eq!(shape.extent(2).sym(), Some(sym));
        assert_eq!(shape.extent(0).sym(), None);
        assert_eq!(shape.extents().count(), 4);

        // resolve() substitutes the live value; unbound is an error.
        let mut env = SymEnv::new();
        assert!(shape.resolve(&env).is_err(), "unbound symbol must error");
        env.bind(sym, 53).unwrap();
        let concrete = shape.resolve(&env).unwrap();
        assert_eq!(concrete.dims(), &[1, 32, 53, 128]);
        assert!(!concrete.has_dynamic());
        assert_eq!(shape.extent(2).resolve(&env), Some(53));

        // Two shapes sharing a sym resolve together (unification via id).
        let other = Shape::from_dims(&[1, 32, 4096, 128]).with_dynamic_axis(2, 0, sym);
        assert_eq!(other.resolve(&env).unwrap().dims(), &[1, 32, 53, 128]);

        // Equality includes `dynamic`: symbolic != concrete-with-same-bounds.
        let concrete_caps = Shape::from_dims(&[1, 32, 4096, 128]);
        assert_ne!(shape, concrete_caps);
        assert_eq!(shape, other);
        assert!(!concrete_caps.has_dynamic());
        // from_dims stays the all-Scalar constructor.
        assert_eq!(concrete_caps.extent(2), Extent::Scalar(4096));

        // elem_count() is the capacity product (bound), not the live count.
        assert_eq!(shape.elem_count(), 32 * 4096 * 128);

        // Layout inherits Extent via its embedded Shape; resolve keeps strides.
        let layout = Layout::contiguous_with_offset(shape.clone(), 0);
        assert!(layout.has_dynamic());
        let rlayout = layout.resolve(&env).unwrap();
        assert_eq!(rlayout.dims(), &[1, 32, 53, 128]);
        assert_eq!(rlayout.stride(), layout.stride(), "strides stay capacity-based");
    }
}

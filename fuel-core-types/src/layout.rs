//! Tensor memory layout: shape, strides, and start offset.
//!
//! A [`Layout`] describes how a tensor's elements are stored in memory.
//! Contiguous layouts use row-major strides; non-contiguous layouts arise from
//! operations like [`transpose`](Layout::transpose) or [`narrow`](Layout::narrow).
use crate::{DimVec, Error, Result, Shape};
use smallvec::smallvec;

/// Describes the memory layout of a tensor: shape, strides, and start offset.
///
/// Strides are in units of elements (not bytes). A contiguous (row-major) tensor
/// of shape `[a, b, c]` has strides `[b*c, c, 1]`.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: DimVec,
    start_offset: usize,
}

impl Layout {
    /// Creates a layout with explicit shape, strides, and start offset.
    pub fn new(shape: Shape, stride: DimVec, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    /// Creates a contiguous (row-major) layout with the given start offset.
    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    /// Creates a contiguous (row-major) layout with zero start offset.
    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    /// Returns the dimension sizes as a slice.
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Returns the size of a specific dimension.
    pub fn dim<D: crate::shape::Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    /// Returns a reference to the [`Shape`].
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides as a slice (in elements, not bytes).
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Returns the start offset into the underlying storage.
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Returns `(start, end)` offsets if the layout is contiguous, `None` otherwise.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            let start_o = self.start_offset;
            Some((start_o, start_o + self.shape.elem_count()))
        } else {
            None
        }
    }

    /// Returns `true` if the layout is C-contiguous (row-major).
    ///
    /// This does not require the start offset to be 0.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    /// Returns `true` if the layout is Fortran-contiguous (column-major).
    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_fortran_contiguous(&self.stride)
    }

    /// Returns a narrowed layout selecting `len` elements starting at `start` on dimension `dim`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op: "narrow",
            }
            .bt())?
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            }
            .bt())?
        }
        let mut dims = DimVec::from_slice(dims);
        dims[dim] = len;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    /// Returns a layout with two dimensions swapped.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let mut stride = DimVec::from_slice(self.stride());
        let mut dims = DimVec::from_slice(self.shape().dims());
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            stride,
            start_offset: self.start_offset,
        })
    }

    /// Returns a layout with dimensions reordered according to `idxs`.
    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation =
            idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            crate::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                idxs
            )
        }
        let stride = self.stride();
        let dims = self.shape().dims();
        let mut perm_stride = DimVec::from_slice(stride);
        let mut perm_dims = DimVec::from_slice(dims);
        for (i, &idx) in idxs.iter().enumerate() {
            perm_stride[i] = stride[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self {
            shape: Shape::from(perm_dims),
            stride: perm_stride,
            start_offset: self.start_offset,
        })
    }

    /// Returns a layout broadcast to the given shape.
    ///
    /// Dimensions of size 1 in `self` are stretched to match the target shape.
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            }
            .bt());
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride: DimVec = smallvec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                }
                .bt());
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }

    pub fn strided_index(&self) -> crate::StridedIndex<'_> {
        crate::StridedIndex::from_layout(self)
    }

    pub fn strided_blocks(&self) -> crate::StridedBlocks<'_> {
        let mut block_len = 1;
        let mut contiguous_dims = 0; // These are counted from the right.
        for (&stride, &dim) in self.stride().iter().zip(self.dims().iter()).rev() {
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.dims().len() - contiguous_dims;
        if index_dims == 0 {
            crate::StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            }
        } else {
            let block_start_index = crate::StridedIndex::new(
                &self.dims()[..index_dims],
                &self.stride[..index_dims],
                self.start_offset,
            );
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            }
        }
    }

    // Returns the contiguous offsets with broadcast if applicable.
    pub fn offsets_b(&self) -> Option<ContiguousOffsetsWithBroadcast> {
        let mut left_broadcast = 1;
        let mut right_broadcast = 1;
        let strides = self.stride();
        let dims = self.dims();
        let mut start_cont = 0;
        let mut end_cont = dims.len();
        for (&s, &d) in strides.iter().zip(dims.iter()) {
            if s != 0 {
                break;
            }
            start_cont += 1;
            left_broadcast *= d;
        }
        if start_cont == dims.len() {
            return Some(ContiguousOffsetsWithBroadcast {
                start: self.start_offset,
                len: 1,
                left_broadcast,
                right_broadcast: 1,
            });
        }
        for (&s, &d) in strides.iter().zip(dims.iter()).rev() {
            if s != 0 {
                break;
            }
            end_cont -= 1;
            right_broadcast *= d;
        }
        // Check that the inner dims are contiguous
        let strides = &strides[start_cont..end_cont];
        let dims = &dims[start_cont..end_cont];
        let mut len = 1;
        for (&stride, &dim) in strides.iter().zip(dims.iter()).rev() {
            if stride != len {
                return None;
            }
            len *= dim;
        }
        Some(ContiguousOffsetsWithBroadcast {
            start: self.start_offset,
            len,
            left_broadcast,
            right_broadcast,
        })
    }
}

/// Offsets for a layout that is contiguous in its inner dimensions but may be broadcast
/// on the left or right outer dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousOffsetsWithBroadcast {
    /// Start offset of the contiguous block.
    pub start: usize,
    /// Length of the contiguous block in elements.
    pub len: usize,
    /// Number of times the block is repeated on the left (outer) dimensions.
    pub left_broadcast: usize,
    /// Number of times the block is repeated on the right (inner) dimensions.
    pub right_broadcast: usize,
}

use crate::{DimVec, Layout};
use smallvec::smallvec;

/// An iterator over offset position for items of an N-dimensional arrays stored in a
/// flat buffer using some potential strides.
///
/// `next_storage_index` is held as `isize` internally so negative
/// strides (e.g. from `Op::Flip`) work uniformly: each step adds
/// `stride[k]` (signed) to the current offset, and the iteration
/// invariant is that the cumulative offset stays in
/// `[0, storage.len())` — a property the producing Layout's
/// `start_offset` guarantees by construction. Yielded indices are
/// cast to `usize` at the boundary.
#[derive(Debug)]
pub struct StridedIndex<'a> {
    next_storage_index: Option<isize>,
    multi_index: DimVec,
    dims: &'a [usize],
    stride: &'a [isize],
    remaining: usize,
}

impl<'a> StridedIndex<'a> {
    pub fn new(dims: &'a [usize], stride: &'a [isize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(start_offset as isize)
        };
        StridedIndex {
            next_storage_index,
            multi_index: smallvec![0; dims.len()],
            dims,
            stride,
            remaining: elem_count,
        }
    }

    pub fn from_layout(l: &'a Layout) -> Self {
        Self::new(l.dims(), l.stride(), l.start_offset())
    }
}

impl Iterator for StridedIndex<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = self.next_storage_index?;
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= (*multi_i as isize) * stride_i;
                *multi_i = 0
            }
        }
        self.remaining -= 1;
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        // Cast to usize at the boundary; the Layout's construction
        // invariant guarantees the offset is non-negative.
        Some(storage_index as usize)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIndex<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

#[derive(Debug)]
pub enum StridedBlocks<'a> {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: StridedIndex<'a>,
        block_len: usize,
    },
}

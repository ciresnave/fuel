//! Safetensors support — transport convenience over the upstream
//! parser.
//!
//! Unlike the other modules in `fuel-formats`, the byte-level
//! safetensors parser is **not implemented here**. The upstream
//! [`safetensors`] crate already provides a complete, transport-
//! independent parser ([`SafeTensors::deserialize`] over `&[u8]`,
//! [`TensorView`], [`Dtype`], [`View`]). Re-implementing it would be
//! wasted effort.
//!
//! What this module *does* provide:
//!
//! - **Re-exports** of the parser surface so transport-only consumers
//!   can depend on `fuel-formats` instead of pulling in upstream
//!   `safetensors` directly.
//! - [`MmapedFile`] — a low-level memory-mapped file handle that
//!   produces a [`SafeTensors`] view via [`MmapedFile::deserialize`].
//!   Transport-independent in the same sense as the rest of
//!   `fuel-formats`: it owns the bytes, and the parser surface
//!   above sits on top.
//!
//! Tensor-construction wrappers ([`fuel_core::safetensors::load`],
//! [`fuel_core::safetensors::MmapedSafetensors`],
//! [`fuel_core::safetensors::SliceSafetensors`],
//! [`fuel_core::safetensors::BufferedSafetensors`], the [`Load`]
//! trait, dtype materializers, save path) stay in `fuel-core`
//! because each calls `Tensor::from_*` or `Storage::*` constructors.
//! When work item E lands and `Tensor` moves to `fuel-tensor`,
//! those wrappers migrate to `fuel-loaders`.

use std::path::{Path, PathBuf};

use fuel_core_types::{Error, Result};

pub use safetensors::SafeTensors;
pub use safetensors::tensor::{Dtype, TensorView, View};

/// A low-level memory-mapped file handle for a safetensors file.
///
/// This does not eagerly deserialize the header. Call
/// [`MmapedFile::deserialize`] to obtain a [`SafeTensors`] view that
/// can be iterated or queried for individual tensors.
///
/// # Safety
///
/// Construction is `unsafe` because it relies on memory-mapped I/O.
/// The underlying file must not be modified while this handle is
/// alive.
pub struct MmapedFile {
    path: PathBuf,
    inner: memmap2::Mmap,
}

impl MmapedFile {
    /// Open `p` and memory-map its contents.
    ///
    /// # Safety
    ///
    /// Inherits the safety contract of [`memmap2::MmapOptions`]: the
    /// caller must ensure the file is not concurrently modified or
    /// truncated for the lifetime of the returned handle.
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let inner = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::from(e).with_path(p))?
        };
        Ok(Self {
            inner,
            path: p.to_path_buf(),
        })
    }

    /// Deserialize the safetensors header and return a borrowed
    /// [`SafeTensors`] view.
    ///
    /// The returned view borrows from the memory-mapped region and
    /// can be used to iterate over tensor names or load individual
    /// tensor views.
    pub fn deserialize(&self) -> Result<SafeTensors<'_>> {
        let st = SafeTensors::deserialize(&self.inner)
            .map_err(|e| Error::from(e).with_path(&self.path))?;
        Ok(st)
    }

    /// Path this handle was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Borrow the underlying mmapped bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

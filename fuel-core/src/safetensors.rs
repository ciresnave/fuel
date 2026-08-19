// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors file reading — the mmap/view surface.
//!
//! - [`MmapedSafetensors`] — memory-maps one or more files and hands out
//!   `safetensors::TensorView`s via `get`/`tensors`. This is what the lazy
//!   stack uses: it reads `view.data()` + `view.dtype()` and decodes the bytes
//!   itself (fuel-core/src/lazy.rs:9802-9810).
//! - [`BufferedSafetensors`] — the same, over an owned `Vec<u8>`.
//!
//! The EAGER half was removed in B6: `impl st::View for Tensor`, the `Load`
//! trait and its `.load(device)` method, every `convert*` helper, the free
//! `load`/`load_buffer`/`save` functions, and `SliceSafetensors`. All of them
//! produced or consumed the eager `Tensor`, and nothing lazy called them —
//! verified: zero `.load(` call sites workspace-wide.
use crate::{Error, Result};
use safetensors::tensor as st;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

#[derive(yoke::Yokeable)]
struct SafeTensors_<'a>(SafeTensors<'a>);

/// Memory-mapped access to one or more safetensors files.
///
/// This is the recommended way to load large models because tensors are read from disk on
/// demand via `mmap` rather than being fully loaded into memory upfront. When multiple files
/// are provided (via [`MmapedSafetensors::multi`]), a routing table maps tensor names to the
/// correct file.
///
/// # Safety
///
/// Construction is `unsafe` because it relies on memory-mapped I/O
/// ([`memmap2::MmapOptions`]). The caller must ensure the underlying files are not modified
/// or truncated while the `MmapedSafetensors` is alive.
///
/// # Example
///
/// ```no_run
/// use fuel_core::safetensors::MmapedSafetensors;
/// // SAFETY: the file must not be modified while the mapping is alive.
/// let st = unsafe { MmapedSafetensors::new("weights.safetensors")? };
/// let view = st.get("weight")?;   // raw TensorView; caller decodes the bytes
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub struct MmapedSafetensors {
    safetensors: Vec<yoke::Yoke<SafeTensors_<'static>, memmap2::Mmap>>,
    routing: Option<HashMap<String, usize>>,
}

impl MmapedSafetensors {
    /// Creates a wrapper around a memory mapped file and deserialize the safetensors header.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
        let file = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| Error::from(e).with_path(p))?
        };
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
            file,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)
                    .map_err(|e| Error::from(e).with_path(p))?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self {
            safetensors: vec![safetensors],
            routing: None,
        })
    }

    /// Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
    ///
    /// If a tensor name appears in multiple files, the last entry is returned.
    ///
    /// # Safety
    ///
    /// The unsafe is inherited from [`memmap2::MmapOptions`].
    pub unsafe fn multi<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let mut routing = HashMap::new();
        let mut safetensors = vec![];
        for (index, p) in paths.iter().enumerate() {
            let p = p.as_ref();
            let file = std::fs::File::open(p).map_err(|e| Error::from(e).with_path(p))?;
            let file = unsafe {
                memmap2::MmapOptions::new()
                    .map(&file)
                    .map_err(|e| Error::from(e).with_path(p))?
            };
            let data = yoke::Yoke::<SafeTensors_<'static>, memmap2::Mmap>::try_attach_to_cart(
                file,
                |data: &[u8]| {
                    let st = safetensors::SafeTensors::deserialize(data)
                        .map_err(|e| Error::from(e).with_path(p))?;
                    Ok::<_, Error>(SafeTensors_(st))
                },
            )?;
            for k in data.get().0.names() {
                routing.insert(k.to_string(), index);
            }
            safetensors.push(data)
        }
        Ok(Self {
            safetensors,
            routing: Some(routing),
        })
    }

    /// Return metadata (name, dtype, shape) for every tensor across all mapped files.
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        let mut tensors = vec![];
        for safetensors in self.safetensors.iter() {
            tensors.push(safetensors.get().0.tensors())
        }
        tensors.into_iter().flatten().collect()
    }

    /// Retrieve the raw `TensorView` for a tensor by name without loading it onto a device.
    ///
    /// This is useful for inspecting tensor metadata (dtype, shape) before deciding
    /// whether to materialize it.
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        let index = match &self.routing {
            None => 0,
            Some(routing) => {
                let index = routing.get(name).ok_or_else(|| {
                    Error::CannotFindTensor {
                        path: name.to_string(),
                    }
                    .bt()
                })?;
                *index
            }
        };
        Ok(self.safetensors[index].get().0.tensor(name)?)
    }
}

/// Owning wrapper around a `Vec<u8>` containing safetensors data.
///
/// Use this when the data source (e.g. a file read or a network download) hands
/// you an owned `Vec<u8>` rather than a path. For the memory-mapped equivalent,
/// see [`MmapedSafetensors`].
///
/// # Example
///
/// ```no_run
/// use fuel_core::safetensors::BufferedSafetensors;
/// let bytes: Vec<u8> = std::fs::read("weights.safetensors")?;
/// let st = BufferedSafetensors::new(bytes)?;
/// let view = st.get("weight")?;   // raw TensorView; caller decodes the bytes
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub struct BufferedSafetensors {
    safetensors: yoke::Yoke<SafeTensors_<'static>, Vec<u8>>,
}

impl BufferedSafetensors {
    /// Creates a wrapper around a binary buffer and deserialize the safetensors header.
    pub fn new(buffer: Vec<u8>) -> Result<Self> {
        let safetensors = yoke::Yoke::<SafeTensors_<'static>, Vec<u8>>::try_attach_to_cart(
            buffer,
            |data: &[u8]| {
                let st = safetensors::SafeTensors::deserialize(data)?;
                Ok::<_, Error>(SafeTensors_(st))
            },
        )?;
        Ok(Self { safetensors })
    }

    /// Return metadata for every tensor in the buffer.
    pub fn tensors(&self) -> Vec<(String, st::TensorView<'_>)> {
        self.safetensors.get().0.tensors()
    }

    /// Retrieve the raw `TensorView` for a tensor by name without loading it onto a device.
    pub fn get(&self, name: &str) -> Result<st::TensorView<'_>> {
        Ok(self.safetensors.get().0.tensor(name)?)
    }
}

/// A low-level memory-mapped safetensors file handle.
///
/// Re-exported from [`fuel_formats::safetensors`] — the
/// transport-independent layer that owns the mmap surface. Use
/// [`MmapedFile::deserialize`] to obtain a `SafeTensors` view, then read
/// each `TensorView`'s bytes directly.
///
/// # Example
///
/// ```no_run
/// use fuel_core::safetensors::MmapedFile;
/// // SAFETY: the file must not be modified while the mapping is alive.
/// let file = unsafe { MmapedFile::new("weights.safetensors")? };
/// let st = file.deserialize()?;
/// for (name, view) in st.tensors() {
///     println!("tensor: {name}  dtype={:?}  bytes={}", view.dtype(), view.data().len());
/// }
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub use fuel_formats::safetensors::MmapedFile;

#[cfg(test)]
mod tests {
    use super::*;

    /// The view surface must hand back raw bytes without ever materializing a
    /// tensor — this is the exact path the lazy stack takes (`get` → `data()`).
    #[test]
    fn buffered_get_returns_raw_view() -> Result<()> {
        // A minimal hand-built safetensors buffer: one U8 tensor `x` = [1, 3].
        let bytes: &[u8] =
            b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
        let st = BufferedSafetensors::new(bytes.to_vec())?;

        let view = st.get("x")?;
        assert_eq!(view.shape(), &[2]);
        assert_eq!(view.dtype(), st::Dtype::U8);
        assert_eq!(view.data(), &[1, 3]);

        let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["x".to_string()]);
        Ok(())
    }

    /// A missing name is an error, not a panic.
    #[test]
    fn buffered_get_missing_name_is_an_error() {
        let bytes: &[u8] =
            b"8\0\0\0\0\0\0\0{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   \x01\x03";
        let st = BufferedSafetensors::new(bytes.to_vec()).unwrap();
        assert!(st.get("nope").is_err());
    }
}

//! Tensor-construction wrappers around [`fuel_formats::pickle`].
//!
//! Wire-format parsing (OpCode interpreter, `Stack`, `Object`,
//! `TensorInfo`, archive walker `read_pth_tensor_info`) lives in
//! `fuel-formats`. This file holds the two pieces that need a
//! `Tensor`: [`PthTensors::get`] (loads raw storage bytes from the
//! zip and reshapes/transposes into a `Tensor`) and the high-level
//! [`read_all`] / [`read_all_with_key`] entry points.

use std::collections::HashMap;

use crate::{Result, Tensor};

pub use fuel_formats::pickle::{
    OpCode, Object, Stack, TensorInfo, read_pth_tensor_info,
};

/// Lazy tensor loader for PyTorch `.pth` checkpoint files.
///
/// On construction the pickle metadata is parsed to build a name-to-[`TensorInfo`] index.
/// Individual tensors can then be loaded on demand via [`PthTensors::get`], which re-opens
/// the zip archive for each read (the archive is not held open between calls).
///
/// # Example
///
/// ```rust,no_run
/// use fuel_core::pickle::PthTensors;
/// let loader = PthTensors::new("model.pth", None)?;
/// for name in loader.tensor_infos().keys() {
///     println!("{name}");
/// }
/// # Ok::<(), fuel_core::Error>(())
/// ```
pub struct PthTensors {
    tensor_infos: HashMap<String, TensorInfo>,
    path: std::path::PathBuf,
}

impl PthTensors {
    /// Open a `.pth` file and parse its tensor metadata.
    ///
    /// If the archive contains a nested dict (e.g. `{"state_dict": {...}}`), pass the
    /// appropriate `key` to drill into it.
    pub fn new<P: AsRef<std::path::Path>>(path: P, key: Option<&str>) -> Result<Self> {
        let tensor_infos = read_pth_tensor_info(path.as_ref(), false, key)?;
        let tensor_infos = tensor_infos
            .into_iter()
            .map(|ti| (ti.name.to_string(), ti))
            .collect();
        let path = path.as_ref().to_owned();
        Ok(Self { tensor_infos, path })
    }

    /// Return the tensor metadata index built during construction.
    pub fn tensor_infos(&self) -> &HashMap<String, TensorInfo> {
        &self.tensor_infos
    }

    /// Load a single tensor by name, returning `Ok(None)` if the name is not in the index.
    ///
    /// Each call re-opens the zip archive to extract the raw storage data.
    pub fn get(&self, name: &str) -> Result<Option<Tensor>> {
        use std::io::Read;
        let tensor_info = match self.tensor_infos.get(name) {
            None => return Ok(None),
            Some(tensor_info) => tensor_info,
        };
        // We hope that the file has not changed since first reading it.
        let zip_reader = std::io::BufReader::new(std::fs::File::open(&self.path)?);
        let mut zip = zip::ZipArchive::new(zip_reader)?;
        let mut reader = zip.by_name(&tensor_info.path)?;
        let is_fortran_contiguous = tensor_info.layout.is_fortran_contiguous();
        let rank = tensor_info.layout.shape().rank();

        // Reading the data is a bit tricky as it can be strided; for now only support the basic
        // case and when the tensor is fortran contiguous.
        if !tensor_info.layout.is_contiguous() && !is_fortran_contiguous {
            crate::bail!(
                "cannot retrieve non-contiguous tensors {:?}",
                tensor_info.layout
            )
        }
        let start_offset = tensor_info.layout.start_offset();
        if start_offset > 0 {
            std::io::copy(
                &mut reader.by_ref().take(start_offset as u64),
                &mut std::io::sink(),
            )?;
        }
        let tensor = Tensor::from_reader(
            tensor_info.layout.shape().clone(),
            tensor_info.dtype,
            &mut reader,
        )?;

        if rank > 1 && is_fortran_contiguous {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            let shape_reversed: Vec<_> = tensor_info.layout.dims().iter().rev().cloned().collect();
            let tensor = tensor.reshape(shape_reversed)?;
            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            let dim_indices_reversed: Vec<_> = (0..rank).rev().collect();
            let tensor = tensor.permute(dim_indices_reversed)?;
            Ok(Some(tensor))
        } else {
            Ok(Some(tensor))
        }
    }
}

/// Read all the tensors from a PyTorch pth file with a given key.
///
/// Sometimes the pth file contains multiple objects and the
/// `state_dict` is the one of interest; pass that key to drill into
/// it.
pub fn read_all_with_key<P: AsRef<std::path::Path>>(
    path: P,
    key: Option<&str>,
) -> Result<Vec<(String, Tensor)>> {
    let pth = PthTensors::new(path, key)?;
    let tensor_names = pth.tensor_infos.keys();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        if let Some(tensor) = pth.get(name)? {
            tensors.push((name.to_string(), tensor))
        }
    }
    Ok(tensors)
}

/// Read all the tensors from a PyTorch pth file.
pub fn read_all<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<(String, Tensor)>> {
    read_all_with_key(path, None)
}

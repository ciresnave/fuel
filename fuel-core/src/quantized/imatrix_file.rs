//! Thin wrapper preserving the historical `load_imatrix` entry point.
//!
//! Format-parsing logic lives in [`fuel_formats::imatrix`]. This file
//! exists only for back-compat — callers who imported
//! `fuel_core::quantized::imatrix_file::load_imatrix` continue to
//! work. New code should depend on `fuel-formats` directly.

use std::collections::HashMap;
use std::path::Path;

use crate::Result;

/// Load an imatrix file and return `name -> normalized activations`.
///
/// Delegates to [`fuel_formats::imatrix::load_path`].
pub fn load_imatrix<P: AsRef<Path>>(fname: P) -> Result<HashMap<String, Vec<f32>>> {
    fuel_formats::imatrix::load_path(fname)
}

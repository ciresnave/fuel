//! `LazyVarMap` — a name → [`LazyVar`] registry with safetensors
//! save/load.
//!
//! Companion to [`crate::lazy_nn_optim::LazyVar`]. Mirrors the eager
//! `fuel_nn::VarMap` surface so checkpoint code that used to call
//! `varmap.save(path)` / `varmap.load(path)` can swap to the lazy
//! equivalent without architectural change.
//!
//! # Storage format
//!
//! Each parameter is written as an F32 tensor with its shape and name
//! intact. On load, only parameters whose name is already present in
//! the map have their host buffers updated; unknown tensor names in
//! the file are silently ignored (matches the eager `VarMap::load`
//! semantics).
//!
//! Files use the standard safetensors layout (header + concatenated
//! little-endian F32 bytes), so they are interoperable with HF Hub
//! checkpoints and the `MmapedSafetensors` loader.

use crate::Result;
use crate::lazy_nn_optim::LazyVar;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Name-keyed registry of [`LazyVar`] parameters with serialize / load
/// helpers backed by the `safetensors` format.
#[derive(Clone, Debug, Default)]
pub struct LazyVarMap {
    vars: Arc<RwLock<HashMap<String, LazyVar>>>,
}

impl LazyVarMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `var` keyed by its name. Replaces any existing entry with
    /// the same name.
    pub fn insert(&self, var: LazyVar) {
        let name = var.name().to_string();
        self.vars.write().unwrap().insert(name, var);
    }

    /// Look up a parameter by name. Returns `None` if absent.
    pub fn get(&self, name: &str) -> Option<LazyVar> {
        self.vars.read().unwrap().get(name).cloned()
    }

    /// Snapshot of all registered parameters in arbitrary order.
    pub fn all_vars(&self) -> Vec<LazyVar> {
        self.vars.read().unwrap().values().cloned().collect()
    }

    /// Number of registered parameters.
    pub fn len(&self) -> usize {
        self.vars.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.vars.read().unwrap().is_empty()
    }

    /// Serialize every registered [`LazyVar`] as an F32 safetensors
    /// tensor at `path`. Shapes are preserved.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let snapshots = {
            let guard = self.vars.read().unwrap();
            let mut out: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::with_capacity(guard.len());
            for (name, var) in guard.iter() {
                out.push((name.clone(), var.to_vec(), var.shape().dims().to_vec()));
            }
            out
        };
        let byte_bufs: Vec<(String, Vec<u8>, Vec<usize>)> = snapshots
            .into_iter()
            .map(|(name, floats, shape)| {
                let mut bytes = Vec::with_capacity(floats.len() * 4);
                for v in &floats {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (name, bytes, shape)
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = byte_bufs
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice())
                    .map_err(|e| {
                        crate::Error::Msg(format!(
                            "LazyVarMap::save: TensorView for {name}: {e}"
                        ))
                        .bt()
                    });
                view.map(|v| (name.clone(), v))
            })
            .collect::<Result<Vec<_>>>()?;
        safetensors::tensor::serialize_to_file(views.into_iter(), None, path.as_ref())
            .map_err(|e| crate::Error::Msg(format!("LazyVarMap::save: {e}")).bt())?;
        Ok(())
    }

    /// Load parameter values from a safetensors file at `path`. Only
    /// parameters whose name is already present in this map are
    /// updated; unknown tensor names in the file are silently ignored.
    /// Errors if a registered parameter's stored shape does not match
    /// the on-disk shape or if its dtype is not F32.
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            crate::Error::Msg(format!("LazyVarMap::load: read {}: {e}", path.as_ref().display()))
                .bt()
        })?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|e| crate::Error::Msg(format!("LazyVarMap::load: deserialize: {e}")).bt())?;
        let guard = self.vars.read().unwrap();
        for (name, view) in tensors.tensors() {
            let Some(var) = guard.get(&name) else {
                continue;
            };
            if view.dtype() != Dtype::F32 {
                return Err(crate::Error::Msg(format!(
                    "LazyVarMap::load: tensor {name} has dtype {:?}, expected F32",
                    view.dtype()
                ))
                .bt());
            }
            let on_disk_shape = view.shape();
            let var_shape = var.shape().dims();
            if on_disk_shape != var_shape {
                return Err(crate::Error::Msg(format!(
                    "LazyVarMap::load: tensor {name} has shape {on_disk_shape:?}, registered LazyVar shape is {var_shape:?}",
                ))
                .bt());
            }
            let raw = view.data();
            if raw.len() % 4 != 0 {
                return Err(crate::Error::Msg(format!(
                    "LazyVarMap::load: tensor {name} has byte length {} not divisible by 4",
                    raw.len(),
                ))
                .bt());
            }
            let mut values = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            var.set(values)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::Shape;

    #[test]
    fn insert_and_get_roundtrips() -> Result<()> {
        let map = LazyVarMap::new();
        let v = LazyVar::new("alpha", Shape::from_dims(&[2]), vec![1.0_f32, 2.0])?;
        map.insert(v.clone());
        let got = map.get("alpha").expect("alpha should be present");
        assert_eq!(got.name(), "alpha");
        assert_eq!(got.to_vec(), vec![1.0_f32, 2.0]);
        assert_eq!(map.len(), 1);
        Ok(())
    }

    #[test]
    fn save_and_load_roundtrips() -> Result<()> {
        let tmp = std::env::temp_dir().join("fuel_lazy_varmap_save_load.safetensors");
        let _ = std::fs::remove_file(&tmp);

        let src = LazyVarMap::new();
        src.insert(LazyVar::new(
            "a",
            Shape::from_dims(&[3]),
            vec![1.0_f32, 2.0, 3.0],
        )?);
        src.insert(LazyVar::new(
            "b",
            Shape::from_dims(&[2, 2]),
            vec![10.0_f32, 20.0, 30.0, 40.0],
        )?);
        src.save(&tmp)?;

        let dst = LazyVarMap::new();
        dst.insert(LazyVar::zeros("a", Shape::from_dims(&[3]))?);
        dst.insert(LazyVar::zeros("b", Shape::from_dims(&[2, 2]))?);
        dst.load(&tmp)?;

        assert_eq!(dst.get("a").unwrap().to_vec(), vec![1.0_f32, 2.0, 3.0]);
        assert_eq!(
            dst.get("b").unwrap().to_vec(),
            vec![10.0_f32, 20.0, 30.0, 40.0]
        );
        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn load_skips_unknown_names() -> Result<()> {
        let tmp = std::env::temp_dir().join("fuel_lazy_varmap_skip_unknown.safetensors");
        let _ = std::fs::remove_file(&tmp);
        let src = LazyVarMap::new();
        src.insert(LazyVar::new("known", Shape::from_dims(&[1]), vec![42.0_f32])?);
        src.insert(LazyVar::new(
            "extra",
            Shape::from_dims(&[1]),
            vec![99.0_f32],
        )?);
        src.save(&tmp)?;

        // Destination only knows about "known"; "extra" must be silently ignored.
        let dst = LazyVarMap::new();
        dst.insert(LazyVar::zeros("known", Shape::from_dims(&[1]))?);
        dst.load(&tmp)?;
        assert_eq!(dst.get("known").unwrap().to_vec(), vec![42.0_f32]);
        assert!(dst.get("extra").is_none());
        std::fs::remove_file(&tmp).ok();
        Ok(())
    }

    #[test]
    fn load_rejects_shape_mismatch() -> Result<()> {
        let tmp = std::env::temp_dir().join("fuel_lazy_varmap_shape_mismatch.safetensors");
        let _ = std::fs::remove_file(&tmp);
        let src = LazyVarMap::new();
        src.insert(LazyVar::new("x", Shape::from_dims(&[4]), vec![1.0, 2.0, 3.0, 4.0])?);
        src.save(&tmp)?;
        let dst = LazyVarMap::new();
        dst.insert(LazyVar::zeros("x", Shape::from_dims(&[2, 2]))?);
        assert!(dst.load(&tmp).is_err());
        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}

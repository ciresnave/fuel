//! A thread-safe store for named trainable variables.
//!
//! A [`VarMap`] holds a collection of [`candle::Var`] instances keyed by name. It is the
//! primary building block for training workflows: you create a `VarMap`, build your model
//! via [`VarBuilder::from_varmap`](crate::VarBuilder::from_varmap), and then pass the
//! variables to an optimizer.
//!
//! `VarMap` also supports serialization to and from the safetensors format via [`VarMap::save`]
//! and [`VarMap::load`], making it easy to checkpoint and resume training.
use candle::{DType, Device, Result, Shape, Tensor, Var};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A thread-safe store of named trainable variables ([`candle::Var`]).
///
/// New variables are created on demand when accessed through a [`VarBuilder`](crate::VarBuilder)
/// backed by this map. If a variable with the requested name already exists, the existing one is
/// returned (enabling weight sharing). The map can be serialized to safetensors via [`save`](Self::save)
/// and loaded back via [`load`](Self::load).
///
/// All access is protected by a [`Mutex`], so the map can be shared across
/// threads (e.g. for data-parallel training).
///
/// # Example
///
/// ```rust
/// use candle_nn::{VarMap, VarBuilder};
/// use candle::{DType, Device};
///
/// let vm = VarMap::new();
/// assert_eq!(vm.all_vars().len(), 0);
///
/// // Build a model using a VarBuilder that writes into the map:
/// // let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
/// // let w = vb.get((4, 4), "weight")?;
/// // vm.save("checkpoint.safetensors")?;
/// # Ok::<(), candle::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct VarMap {
    data: Arc<Mutex<HashMap<String, Var>>>,
}

impl VarMap {
    /// Create a new empty `VarMap`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let data = Arc::new(Mutex::new(HashMap::new()));
        Self { data }
    }

    /// Retrieve all the variables currently stored in the map.
    pub fn all_vars(&self) -> Vec<Var> {
        let tensor_data = self.data.lock().unwrap();
        #[allow(clippy::map_clone)]
        tensor_data.values().map(|c| c.clone()).collect::<Vec<_>>()
    }

    /// Save the map in the safetensors format.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let data = tensor_data.iter().map(|(k, v)| (k, v.as_tensor()));
        safetensors::tensor::serialize_to_file(data, None, path.as_ref())?;
        Ok(())
    }

    /// Load some values from a safetensors file and modify the existing variables to have these
    /// values.
    ///
    /// Note that values for variables that are currently not in the map are not kept.
    pub fn load<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        let data = unsafe { candle::safetensors::MmapedSafetensors::new(path)? };
        let mut tensor_data = self.data.lock().unwrap();
        for (name, var) in tensor_data.iter_mut() {
            let data = data.load(name, var.device())?;
            if let Err(err) = var.set(&data) {
                candle::bail!("error setting {name} using data from {path:?}: {err}",)
            }
        }
        Ok(())
    }

    /// Set a named variable to some value.
    pub fn set_one<K: AsRef<str>, V: AsRef<Tensor>>(&mut self, name: K, value: V) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        let name = name.as_ref();
        match tensor_data.get(name) {
            None => candle::bail!("cannot find {name} in VarMap"),
            Some(var) => {
                if let Err(err) = var.set(value.as_ref()) {
                    candle::bail!("error setting {name}: {err}",)
                }
            }
        }
        Ok(())
    }

    /// Set some named variables to some values.
    ///
    /// If an error is returned, some of the variables might have already been set to their new
    /// values.
    pub fn set<I: Iterator<Item = (K, V)>, K: AsRef<str>, V: AsRef<Tensor>>(
        &mut self,
        iter: I,
    ) -> Result<()> {
        let tensor_data = self.data.lock().unwrap();
        for (name, value) in iter {
            let name = name.as_ref();
            match tensor_data.get(name) {
                None => candle::bail!("cannot find {name} in VarMap"),
                Some(var) => {
                    if let Err(err) = var.set(value.as_ref()) {
                        candle::bail!("error setting {name}: {err}",)
                    }
                }
            }
        }
        Ok(())
    }

    /// Retrieves an existing variable or creates a new one.
    ///
    /// If a variable named `path` already exists in the map, it is returned (after a shape
    /// check). Otherwise, a new variable is created using the provided `init` strategy,
    /// inserted into the map, and returned.
    pub fn get<S: Into<Shape>>(
        &self,
        shape: S,
        path: &str,
        init: crate::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let mut tensor_data = self.data.lock().unwrap();
        if let Some(tensor) = tensor_data.get(path) {
            let tensor_shape = tensor.shape();
            if &shape != tensor_shape {
                candle::bail!("shape mismatch on {path}: {shape:?} <> {tensor_shape:?}")
            }
            return Ok(tensor.as_tensor().clone());
        }
        let var = init.var(shape, dtype, device)?;
        let tensor = var.as_tensor().clone();
        tensor_data.insert(path.to_string(), var);
        Ok(tensor)
    }

    /// Returns a reference to the underlying mutex-protected variable map.
    ///
    /// This is useful for advanced use cases such as iterating over all variables or
    /// implementing custom serialization logic.
    pub fn data(&self) -> &Mutex<HashMap<String, Var>> {
        &self.data
    }
}

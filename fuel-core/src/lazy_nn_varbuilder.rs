//! `LazyVarBuilder` — a name-prefixing wrapper around [`LazyVarMap`]
//! for constructing named lazy parameters.
//!
//! Mirrors the surface of the retired eager `fuel_nn::VarBuilder`: a
//! dot-separated path prefix is maintained so individual layers can
//! request parameters by local name (`"weight"`) while the builder
//! resolves the full qualified name (`"encoder.layer0.weight"`).
//!
//! # Differences from the eager `VarBuilder`
//!
//! The lazy variant returns a [`LazyVar`] (not a `Tensor`) — i.e. a
//! handle to a host-resident parameter that can be spliced into a
//! lazy graph via [`LazyVar::tensor`]. The underlying storage is the
//! [`LazyVarMap`]; the builder is purely a path-prefix + default
//! `DType`/`Device` carrier with no backend-trait machinery.
//!
//! # Path semantics
//!
//! The full key is formed by joining the prefix with the local name
//! using `'.'`. When the prefix is empty the local name is used
//! verbatim.
//!
//! ```text
//! prefix = ""        + name = "w"  ⇒  "w"
//! prefix = "outer"   + name = "w"  ⇒  "outer.w"
//! prefix = "a.b"     + name = "w"  ⇒  "a.b.w"
//! ```

use crate::Result;
use crate::lazy_nn_optim::LazyVar;
use crate::lazy_nn_varmap::LazyVarMap;
use crate::{DType, Device};
use fuel_ir::Shape;

/// A name-prefixing handle over a [`LazyVarMap`] for constructing
/// named lazy parameters. Cheap to clone — internally an Arc-shared
/// map plus a fresh path prefix.
#[derive(Clone, Debug)]
pub struct LazyVarBuilder {
    map: LazyVarMap,
    prefix: String,
    dtype: DType,
    device: Device,
}

impl LazyVarBuilder {
    /// Build a new var-builder rooted at the empty prefix, backed by
    /// `map`. `default_dtype` and `default_device` are carried so
    /// downstream layer factories can pick them up without an extra
    /// argument.
    pub fn from_varmap(map: LazyVarMap, default_dtype: DType, default_device: Device) -> Self {
        Self {
            map,
            prefix: String::new(),
            dtype: default_dtype,
            device: default_device,
        }
    }

    /// Push a new path segment onto the prefix and return the resulting
    /// builder. Conceptually similar to `cd`-ing into a directory:
    ///
    /// ```text
    /// vs.pp("outer").pp("inner").get(shape, "w")  ⇒  "outer.inner.w"
    /// ```
    pub fn pp(&self, name: impl AsRef<str>) -> Self {
        let segment = name.as_ref();
        let new_prefix = if self.prefix.is_empty() {
            segment.to_string()
        } else {
            format!("{}.{}", self.prefix, segment)
        };
        Self {
            map: self.map.clone(),
            prefix: new_prefix,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }

    /// Look up — or zero-initialize — the parameter at `prefix.name`
    /// with the given `shape`. If a [`LazyVar`] with the resolved key
    /// already exists, it is returned as-is.
    pub fn get(&self, shape: impl Into<Shape>, name: &str) -> Result<LazyVar> {
        self.get_with(shape, name, |s| vec![0.0_f32; s.elem_count()])
    }

    /// Look up — or initialize via `init_fn` — the parameter at
    /// `prefix.name`. The init closure receives the resolved [`Shape`]
    /// and must return a `Vec<f32>` of length `shape.elem_count()`.
    ///
    /// If a parameter with the resolved key already exists, the
    /// existing handle is returned and `init_fn` is **not** called.
    pub fn get_with(
        &self,
        shape: impl Into<Shape>,
        name: &str,
        init_fn: impl FnOnce(&Shape) -> Vec<f32>,
    ) -> Result<LazyVar> {
        let shape = shape.into();
        let key = self.path(name);
        if let Some(existing) = self.map.get(&key) {
            if existing.shape().dims() != shape.dims() {
                return Err(crate::Error::Msg(format!(
                    "LazyVarBuilder::get_with: parameter {key} already \
                     registered with shape {:?}, requested {:?}",
                    existing.shape().dims(),
                    shape.dims(),
                ))
                .bt());
            }
            return Ok(existing);
        }
        let data = init_fn(&shape);
        let var = LazyVar::new(key, shape, data)?;
        self.map.insert(var.clone());
        Ok(var)
    }

    /// Access the underlying [`LazyVarMap`].
    pub fn map(&self) -> &LazyVarMap {
        &self.map
    }

    /// The current path prefix (dot-separated, no trailing dot).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The default dtype carried by this builder. New layer factories
    /// pick this up when they don't otherwise specify a dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The default device carried by this builder.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pp_chains_produce_dot_joined_keys() -> Result<()> {
        let map = LazyVarMap::new();
        let vs = LazyVarBuilder::from_varmap(map.clone(), DType::F32, Device::cpu());
        let v = vs.pp("outer").pp("inner").get(Shape::from_dims(&[2, 3]), "w")?;
        assert_eq!(v.name(), "outer.inner.w");
        assert_eq!(v.shape().dims(), &[2, 3]);
        assert!(map.get("outer.inner.w").is_some());

        // Empty prefix => verbatim local name.
        let v2 = vs.get(Shape::from_dims(&[4]), "bare")?;
        assert_eq!(v2.name(), "bare");
        Ok(())
    }

    #[test]
    fn repeated_get_returns_same_var() -> Result<()> {
        let map = LazyVarMap::new();
        let vs = LazyVarBuilder::from_varmap(map, DType::F32, Device::cpu());
        let first = vs.get_with(Shape::from_dims(&[3]), "w", |s| {
            (0..s.elem_count()).map(|i| i as f32 + 1.0).collect()
        })?;
        // Second call with a different init_fn must NOT overwrite the
        // existing values — the original handle is returned verbatim.
        let second = vs.get_with(Shape::from_dims(&[3]), "w", |_| vec![99.0; 3])?;
        assert_eq!(first.name(), second.name());
        assert_eq!(first.shape().dims(), second.shape().dims());
        assert_eq!(first.to_vec(), vec![1.0_f32, 2.0, 3.0]);
        assert_eq!(second.to_vec(), vec![1.0_f32, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn shape_mismatch_on_existing_var_errors() {
        let map = LazyVarMap::new();
        let vs = LazyVarBuilder::from_varmap(map, DType::F32, Device::cpu());
        vs.get(Shape::from_dims(&[2, 2]), "w").unwrap();
        let err = vs.get(Shape::from_dims(&[3, 3]), "w");
        assert!(err.is_err());
    }

    #[test]
    fn dtype_and_device_accessors_round_trip() {
        let map = LazyVarMap::new();
        let vs = LazyVarBuilder::from_varmap(map, DType::F32, Device::cpu());
        assert_eq!(vs.dtype(), DType::F32);
        assert!(vs.device().is_cpu());
        let nested = vs.pp("layer0");
        assert_eq!(nested.dtype(), DType::F32);
        assert!(nested.device().is_cpu());
        assert_eq!(nested.prefix(), "layer0");
    }

    #[test]
    fn save_then_load_round_trips_through_var_builder() -> Result<()> {
        let tmp = std::env::temp_dir().join("fuel_lazy_varbuilder_save_load.safetensors");
        let _ = std::fs::remove_file(&tmp);

        // Source: register two prefixed parameters with known values
        // through a LazyVarBuilder + LazyVarMap.
        let src_map = LazyVarMap::new();
        let src_vs = LazyVarBuilder::from_varmap(src_map.clone(), DType::F32, Device::cpu());
        let src_w = src_vs
            .pp("layer0")
            .get_with(Shape::from_dims(&[2, 2]), "weight", |_| {
                vec![1.0_f32, 2.0, 3.0, 4.0]
            })?;
        let src_b = src_vs
            .pp("layer0")
            .get_with(Shape::from_dims(&[2]), "bias", |_| vec![10.0_f32, 20.0])?;
        assert_eq!(src_w.name(), "layer0.weight");
        assert_eq!(src_b.name(), "layer0.bias");
        src_map.save(&tmp)?;

        // Destination: register placeholders with the same shapes
        // through a fresh LazyVarBuilder, then load — values must
        // overwrite the placeholder zeros.
        let dst_map = LazyVarMap::new();
        let dst_vs = LazyVarBuilder::from_varmap(dst_map.clone(), DType::F32, Device::cpu());
        let dst_w = dst_vs.pp("layer0").get(Shape::from_dims(&[2, 2]), "weight")?;
        let dst_b = dst_vs.pp("layer0").get(Shape::from_dims(&[2]), "bias")?;
        assert_eq!(dst_w.to_vec(), vec![0.0_f32; 4]);
        assert_eq!(dst_b.to_vec(), vec![0.0_f32; 2]);
        dst_map.load(&tmp)?;
        assert_eq!(dst_w.to_vec(), vec![1.0_f32, 2.0, 3.0, 4.0]);
        assert_eq!(dst_b.to_vec(), vec![10.0_f32, 20.0]);

        std::fs::remove_file(&tmp).ok();
        Ok(())
    }
}

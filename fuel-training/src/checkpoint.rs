//! Training checkpoint management (lazy).
//!
//! Extends [`fuel::lazy_nn_varmap::LazyVarMap`]'s basic save/load with
//! metadata tracking for resumable training. A [`Checkpoint`] bundles
//! model weights with the epoch, step, and an optional best metric so
//! training can resume from exactly where it was interrupted.
//!
//! # Example
//!
//! ```rust,no_run
//! use fuel::lazy_nn_optim::LazyVar;
//! use fuel::lazy_nn_varmap::LazyVarMap;
//! use fuel_training::checkpoint::Checkpoint;
//!
//! # fn main() -> fuel::Result<()> {
//! let varmap = LazyVarMap::new();
//! varmap.insert(LazyVar::zeros("w", fuel::Shape::from_dims(&[8]))?);
//!
//! let dir = std::env::temp_dir().join("fuel_ckpt_example");
//! let ckpt = Checkpoint::new(5, 1000).with_metric("val_loss", 0.42);
//! # let _ = std::fs::remove_dir_all(&dir);
//! ckpt.save(&dir, &varmap)?;
//!
//! let loaded = Checkpoint::load_latest(&dir)?;
//! assert_eq!(loaded.epoch(), 5);
//! # std::fs::remove_dir_all(&dir).ok();
//! # Ok(())
//! # }
//! ```

use fuel::Result;
use fuel::lazy_nn_varmap::LazyVarMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Metadata stored alongside model weights.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    epoch: usize,
    step: usize,
    metrics: Vec<(String, f64)>,
}

impl Checkpoint {
    /// Create a new checkpoint with the given epoch and global step.
    pub fn new(epoch: usize, step: usize) -> Self {
        Self {
            epoch,
            step,
            metrics: Vec::new(),
        }
    }

    /// Attach a named metric value (e.g. `"val_loss"`, `"accuracy"`).
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.push((name.to_string(), value));
        self
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn metrics(&self) -> &[(String, f64)] {
        &self.metrics
    }

    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }

    /// Save model weights and metadata to `dir`. Creates `dir` if it
    /// does not exist. Writes:
    /// - `weights.safetensors` — model parameters
    /// - `metadata.json` — epoch, step, metrics
    pub fn save<P: AsRef<Path>>(&self, dir: P, varmap: &LazyVarMap) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(|e| {
            fuel::Error::Msg(format!(
                "Checkpoint::save: create_dir_all {}: {e}",
                dir.display()
            ))
            .bt()
        })?;

        varmap.save(dir.join("weights.safetensors"))?;

        let meta_json = serde_json::to_string_pretty(self).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::save: serialize metadata: {e}")).bt()
        })?;
        std::fs::write(dir.join("metadata.json"), meta_json).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::save: write metadata.json: {e}")).bt()
        })?;
        Ok(())
    }

    /// Save to a timestamped subdirectory inside `base_dir`.
    /// The subdirectory is named `epoch_{epoch}_step_{step}`.
    pub fn save_named<P: AsRef<Path>>(&self, base_dir: P, varmap: &LazyVarMap) -> Result<PathBuf> {
        let name = format!("epoch_{}_step_{}", self.epoch, self.step);
        let dir = base_dir.as_ref().join(name);
        self.save(&dir, varmap)?;
        Ok(dir)
    }

    /// Load model weights and metadata from `dir`, applying weights to `varmap`.
    pub fn load<P: AsRef<Path>>(dir: P, varmap: &LazyVarMap) -> Result<Self> {
        let dir = dir.as_ref();
        let weights_path = dir.join("weights.safetensors");
        varmap.load(&weights_path)?;

        let meta_bytes = std::fs::read(dir.join("metadata.json")).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::load: read metadata.json: {e}")).bt()
        })?;
        let ckpt: Self = serde_json::from_slice(&meta_bytes).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::load: parse metadata.json: {e}")).bt()
        })?;
        Ok(ckpt)
    }

    /// Load metadata only (no weights) from `dir`.
    pub fn load_metadata<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let meta_bytes = std::fs::read(dir.join("metadata.json")).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::load_metadata: read: {e}")).bt()
        })?;
        let ckpt: Self = serde_json::from_slice(&meta_bytes).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::load_metadata: parse: {e}")).bt()
        })?;
        Ok(ckpt)
    }

    /// Load the latest checkpoint metadata from `base_dir`.
    ///
    /// Scans immediate subdirectories for `metadata.json` files and
    /// returns the one with the highest step count.
    pub fn load_latest<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        let direct_meta = base_dir.join("metadata.json");
        if direct_meta.exists() {
            return Self::load_metadata(base_dir);
        }
        let entries = std::fs::read_dir(base_dir).map_err(|e| {
            fuel::Error::Msg(format!("Checkpoint::load_latest: read_dir: {e}")).bt()
        })?;
        let mut best: Option<Self> = None;
        for entry in entries {
            let entry = entry.map_err(|e| {
                fuel::Error::Msg(format!("Checkpoint::load_latest: entry: {e}")).bt()
            })?;
            let meta_path = entry.path().join("metadata.json");
            if meta_path.exists() {
                let ckpt = Self::load_metadata(entry.path())?;
                if best.as_ref().is_none_or(|b| ckpt.step > b.step) {
                    best = Some(ckpt);
                }
            }
        }
        best.ok_or_else(|| {
            fuel::Error::Msg(format!(
                "no checkpoint found in {}",
                base_dir.display()
            ))
            .bt()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_metadata() {
        let ckpt = Checkpoint::new(3, 5000)
            .with_metric("val_loss", 0.123)
            .with_metric("accuracy", 0.95);
        assert_eq!(ckpt.epoch(), 3);
        assert_eq!(ckpt.step(), 5000);
        assert!((ckpt.metric("val_loss").unwrap() - 0.123).abs() < 1e-12);
        assert!((ckpt.metric("accuracy").unwrap() - 0.95).abs() < 1e-12);
        assert!(ckpt.metric("nonexistent").is_none());
    }

    #[test]
    fn save_and_load() -> fuel::Result<()> {
        let dir = std::env::temp_dir().join("fuel_training_lazy_ckpt_test");
        let _ = std::fs::remove_dir_all(&dir);

        let varmap = LazyVarMap::new();
        let ckpt = Checkpoint::new(2, 400).with_metric("loss", 1.5);
        ckpt.save(&dir, &varmap)?;

        let loaded = Checkpoint::load_metadata(&dir)?;
        assert_eq!(loaded.epoch(), 2);
        assert_eq!(loaded.step(), 400);
        assert!((loaded.metric("loss").unwrap() - 1.5).abs() < 1e-12);

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[test]
    fn load_latest_picks_highest_step() -> fuel::Result<()> {
        let base = std::env::temp_dir().join("fuel_training_lazy_ckpt_latest_test");
        let _ = std::fs::remove_dir_all(&base);

        let varmap = LazyVarMap::new();
        Checkpoint::new(1, 100).save_named(&base, &varmap)?;
        Checkpoint::new(2, 200).save_named(&base, &varmap)?;
        Checkpoint::new(1, 150).save_named(&base, &varmap)?;

        let latest = Checkpoint::load_latest(&base)?;
        assert_eq!(latest.step(), 200);

        std::fs::remove_dir_all(&base).ok();
        Ok(())
    }
}

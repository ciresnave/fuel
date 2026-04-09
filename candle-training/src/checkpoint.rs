//! Training checkpoint management.
//!
//! Extends `VarMap`'s basic save/load with metadata tracking for resumable
//! training. A [`Checkpoint`] bundles model weights with the epoch, step, and
//! an optional best metric so training can resume from exactly where it was
//! interrupted.
//!
//! # Example
//!
//! ```rust
//! use candle::{DType, Device, Tensor, Var};
//! use candle_nn::VarMap;
//! use candle_training::checkpoint::Checkpoint;
//!
//! # fn main() -> candle::Result<()> {
//! let mut varmap = VarMap::new();
//! // ... create model using varmap ...
//!
//! // Save a checkpoint
//! let dir = std::env::temp_dir().join("candle_ckpt_example");
//! let ckpt = Checkpoint::new(5, 1000)
//!     .with_metric("val_loss", 0.42);
//!
//! # let _ = std::fs::remove_dir_all(&dir); // clean
//! ckpt.save(&dir, &varmap)?;
//!
//! // Load it back
//! let loaded = Checkpoint::load_latest(&dir)?;
//! assert_eq!(loaded.epoch(), 5);
//! assert_eq!(loaded.step(), 1000);
//! # std::fs::remove_dir_all(&dir).ok();
//! # Ok(())
//! # }
//! ```

use candle::Result;
use candle_nn::VarMap;
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

    /// Return the epoch at which this checkpoint was saved.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Return the global step at which this checkpoint was saved.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Return all stored metrics as `(name, value)` pairs.
    pub fn metrics(&self) -> &[(String, f64)] {
        &self.metrics
    }

    /// Return the value of a named metric, if stored.
    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }

    /// Save model weights and metadata to `dir`.
    ///
    /// Creates `dir` if it does not exist. Writes:
    /// - `weights.safetensors` — model parameters
    /// - `metadata.json` — epoch, step, metrics
    pub fn save<P: AsRef<Path>>(&self, dir: P, varmap: &VarMap) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir).map_err(candle::Error::wrap)?;

        varmap.save(dir.join("weights.safetensors"))?;

        let meta_json =
            serde_json::to_string_pretty(self).map_err(candle::Error::wrap)?;
        std::fs::write(dir.join("metadata.json"), meta_json)
            .map_err(candle::Error::wrap)?;
        Ok(())
    }

    /// Save to a timestamped subdirectory inside `base_dir`.
    ///
    /// The subdirectory is named `epoch_{epoch}_step_{step}`, making it easy
    /// to keep multiple checkpoints.
    pub fn save_named<P: AsRef<Path>>(&self, base_dir: P, varmap: &VarMap) -> Result<PathBuf> {
        let name = format!("epoch_{}_step_{}", self.epoch, self.step);
        let dir = base_dir.as_ref().join(name);
        self.save(&dir, varmap)?;
        Ok(dir)
    }

    /// Load model weights and metadata from `dir`, applying weights to `varmap`.
    pub fn load<P: AsRef<Path>>(dir: P, varmap: &mut VarMap) -> Result<Self> {
        let dir = dir.as_ref();
        let weights_path = dir.join("weights.safetensors");
        varmap.load(&weights_path)?;

        let meta_bytes =
            std::fs::read(dir.join("metadata.json")).map_err(candle::Error::wrap)?;
        let ckpt: Self =
            serde_json::from_slice(&meta_bytes).map_err(candle::Error::wrap)?;
        Ok(ckpt)
    }

    /// Load metadata only (no weights) from `dir`.
    pub fn load_metadata<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let meta_bytes =
            std::fs::read(dir.join("metadata.json")).map_err(candle::Error::wrap)?;
        let ckpt: Self =
            serde_json::from_slice(&meta_bytes).map_err(candle::Error::wrap)?;
        Ok(ckpt)
    }

    /// Load the latest checkpoint from `base_dir` (by metadata only — does not
    /// load weights into a VarMap).
    ///
    /// Scans immediate subdirectories for `metadata.json` files and returns
    /// the one with the highest step count.
    pub fn load_latest<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref();

        // Check for direct metadata.json (single checkpoint directory)
        let direct_meta = base_dir.join("metadata.json");
        if direct_meta.exists() {
            return Self::load_metadata(base_dir);
        }

        // Scan subdirectories
        let entries =
            std::fs::read_dir(base_dir).map_err(candle::Error::wrap)?;
        let mut best: Option<Self> = None;
        for entry in entries {
            let entry = entry.map_err(candle::Error::wrap)?;
            let meta_path = entry.path().join("metadata.json");
            if meta_path.exists() {
                let ckpt = Self::load_metadata(entry.path())?;
                if best.as_ref().is_none_or(|b| ckpt.step > b.step) {
                    best = Some(ckpt);
                }
            }
        }

        best.ok_or_else(|| {
            candle::Error::Msg(format!(
                "no checkpoint found in {}",
                base_dir.display()
            ))
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
    fn save_and_load() -> candle::Result<()> {
        let dir = std::env::temp_dir().join("candle_training_ckpt_test");
        let _ = std::fs::remove_dir_all(&dir);

        let varmap = VarMap::new();
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
    fn load_latest_picks_highest_step() -> candle::Result<()> {
        let base = std::env::temp_dir().join("candle_training_ckpt_latest_test");
        let _ = std::fs::remove_dir_all(&base);

        let varmap = VarMap::new();
        Checkpoint::new(1, 100).save_named(&base, &varmap)?;
        Checkpoint::new(2, 200).save_named(&base, &varmap)?;
        Checkpoint::new(1, 150).save_named(&base, &varmap)?;

        let latest = Checkpoint::load_latest(&base)?;
        assert_eq!(latest.step(), 200);

        std::fs::remove_dir_all(&base).ok();
        Ok(())
    }
}

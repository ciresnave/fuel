//! Progress reporting for model loading.
//!
//! Loading a multi-GB model takes seconds-to-minutes depending on disk
//! and decode cost. Without feedback, callers can't tell a slow load
//! from a hang. This module gives loaders a tiny, dependency-free
//! callback mechanism so they can announce what they're doing without
//! the loaders themselves having to know about terminals or UIs.
//!
//! Design notes:
//! - Events are opaque to Fuel — the caller decides whether to print,
//!   plot, log, or ignore them. This keeps `fuel-core` from depending
//!   on indicatif or any TUI crate.
//! - The callback is `Fn(&ProgressEvent) + Send + Sync` so it can be
//!   shared across threads (rayon parallel decode, future async
//!   shard downloads).
//! - Passing `None` (or using the default `ProgressReporter::silent()`)
//!   compiles to effectively zero overhead: one `Option::is_some`
//!   check per event site.
//!
//! Inspired by MLMF's `ProgressFn` / `ProgressEvent` design
//! (`mlmf/src/progress.rs`), cherry-picked to avoid pulling in
//! candlelight as a transitive dependency.

use std::path::PathBuf;
use std::sync::Arc;

/// Discrete events a loader can emit. New variants can be added later;
/// callers should match non-exhaustively.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ProgressEvent {
    /// About to open and mmap a single file.
    OpeningFile { path: PathBuf, format: &'static str },
    /// Finished parsing the header; known how many tensors are coming.
    MetadataParsed { format: &'static str, tensor_count: usize },
    /// Architecture was detected (from metadata or tensor names).
    ArchitectureDetected { architecture: String },
    /// A single tensor has just been materialized.
    TensorLoaded { name: String, index: usize, total: usize },
    /// A whole file has been fully processed.
    FileComplete { path: PathBuf, tensor_count: usize },
    /// Everything across all files is done.
    Complete { tensor_count: usize },
    /// A non-fatal diagnostic; callers may log or ignore.
    Warning(String),
}

/// Callback signature. `Arc` lets callers share the reporter across
/// threads without the loader having to know about any cloning rules.
pub type ProgressFn = Arc<dyn Fn(&ProgressEvent) + Send + Sync>;

/// Optional progress sink. Pass to loaders that support it; cheap to
/// clone (just Arc refcount). Construct with [`ProgressReporter::new`]
/// to attach a callback, or [`ProgressReporter::silent`] for no-op.
#[derive(Clone, Default)]
pub struct ProgressReporter {
    cb: Option<ProgressFn>,
}

impl ProgressReporter {
    /// Reporter that drops every event. Zero-overhead default.
    pub fn silent() -> Self {
        Self { cb: None }
    }

    /// Reporter backed by any `Fn(&ProgressEvent) + Send + Sync`. Use
    /// for custom sinks — logging, TUI bars, metrics, tests.
    pub fn new<F>(cb: F) -> Self
    where
        F: Fn(&ProgressEvent) + Send + Sync + 'static,
    {
        Self { cb: Some(Arc::new(cb)) }
    }

    /// Emit an event if a callback is attached.
    pub fn emit(&self, event: &ProgressEvent) {
        if let Some(cb) = &self.cb {
            cb(event);
        }
    }

    /// Convenience for the common case of "I just loaded tensor N/M".
    pub fn tensor(&self, name: &str, index: usize, total: usize) {
        if self.cb.is_some() {
            self.emit(&ProgressEvent::TensorLoaded {
                name: name.to_string(),
                index,
                total,
            });
        }
    }

    /// True if any callback is attached. Lets loaders skip expensive
    /// string formatting when nobody is listening.
    pub fn is_enabled(&self) -> bool {
        self.cb.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn silent_reporter_drops_events() {
        let r = ProgressReporter::silent();
        assert!(!r.is_enabled());
        // Does not panic:
        r.emit(&ProgressEvent::Warning("ignored".into()));
        r.tensor("w", 1, 2);
    }

    #[test]
    fn callback_receives_events_in_order() {
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_cb = Arc::clone(&log);
        let r = ProgressReporter::new(move |e| match e {
            ProgressEvent::TensorLoaded { name, .. } => log_cb.lock().unwrap().push(name.clone()),
            ProgressEvent::Complete { .. } => log_cb.lock().unwrap().push("done".into()),
            _ => {}
        });
        r.tensor("a", 0, 2);
        r.tensor("b", 1, 2);
        r.emit(&ProgressEvent::Complete { tensor_count: 2 });
        assert_eq!(&*log.lock().unwrap(), &["a", "b", "done"]);
    }
}

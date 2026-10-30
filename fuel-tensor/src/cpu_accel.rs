// SPDX-License-Identifier: MIT OR Apache-2.0
//! One-time install of the vendor CPU BLAS kernels (oneMKL behind the `onemkl` feature, AOCL behind `aocl`) into the
//! global binding table (ROADMAP row 66).
//!
//! `fuel_mkl_cpu_backend::register_mkl_cpu_kernels` had no production caller: building with
//! `--features onemkl` linked the backend and then never offered its matmul to the judge. The
//! first lazy realize now probes the library and, if it loads, extends the global bindings once.
//!
//! A library that is NOT loadable is not an error: nothing is registered, the scalar CPU kernels
//! stay in force, and [`cpu_accel_report`] says why, so a user can ask "why not MKL".
//!
//! Status: wired, NOT run on hardware by CI (no hosted runner has oneMKL or AOCL). The once-only logic in
//! `install_once` is feature-independent and unit-tested in every build.

#[cfg(any(feature = "onemkl", feature = "aocl", test))]
use std::sync::OnceLock;

/// What happened when a backend's kernels were considered for installation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccelStatus {
    /// The probe passed and the kernels were added to the global binding table.
    Registered,
    /// The probe failed; nothing was registered. Carries the probe's error text.
    NotLoadable(String),
}

/// Run `probe` then, if it passed, `register`, at most once per `cell`; later calls return the
/// recorded status without calling either closure again.
#[cfg(any(feature = "onemkl", feature = "aocl", test))]
pub(crate) fn install_once(
    cell: &OnceLock<AccelStatus>,
    probe: impl FnOnce() -> Result<(), String>,
    register: impl FnOnce(),
) -> AccelStatus {
    cell.get_or_init(|| match probe() {
        Ok(()) => {
            register();
            AccelStatus::Registered
        }
        Err(why) => AccelStatus::NotLoadable(why),
    })
    .clone()
}

#[cfg(feature = "onemkl")]
static MKL: OnceLock<AccelStatus> = OnceLock::new();
#[cfg(feature = "aocl")]
static AOCL: OnceLock<AccelStatus> = OnceLock::new();

/// Install every enabled vendor CPU backend, once. Cheap after the first call.
pub(crate) fn ensure_installed() {
    #[cfg(feature = "onemkl")]
    {
        install_once(
            &MKL,
            || fuel_mkl_cpu_backend::probe_mkl_loadable().map_err(|e| format!("{e:?}")),
            || {
                fuel_dispatch::dispatch::extend_global_bindings(
                    fuel_mkl_cpu_backend::register_mkl_cpu_kernels,
                )
            },
        );
    }
    #[cfg(feature = "aocl")]
    {
        install_once(
            &AOCL,
            || fuel_aocl_cpu_backend::probe_aocl_loadable().map_err(|e| format!("{e:?}")),
            || {
                fuel_dispatch::dispatch::extend_global_bindings(
                    fuel_aocl_cpu_backend::register_aocl_cpu_kernels,
                )
            },
        );
    }
}

/// The install result per vendor backend that this build enables, as `(name, status)`. A backend
/// whose feature is off is absent, and so is one before the first lazy realize has run.
pub fn cpu_accel_report() -> Vec<(&'static str, AccelStatus)> {
    #[allow(unused_mut)]
    let mut out: Vec<(&'static str, AccelStatus)> = Vec::new();
    #[cfg(feature = "onemkl")]
    if let Some(s) = MKL.get() {
        out.push(("mkl", s.clone()));
    }
    #[cfg(feature = "aocl")]
    if let Some(s) = AOCL.get() {
        out.push(("aocl", s.clone()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn a_loadable_library_registers_exactly_once_across_calls() {
        let cell = OnceLock::new();
        let registered = AtomicUsize::new(0);
        for _ in 0..3 {
            let s = install_once(
                &cell,
                || Ok(()),
                || {
                    registered.fetch_add(1, Ordering::SeqCst);
                },
            );
            assert_eq!(s, AccelStatus::Registered);
        }
        assert_eq!(registered.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_unloadable_library_registers_nothing_and_says_why() {
        let cell = OnceLock::new();
        let registered = AtomicUsize::new(0);
        let s = install_once(
            &cell,
            || Err("mkl_rt.dll not found".to_string()),
            || {
                registered.fetch_add(1, Ordering::SeqCst);
            },
        );
        assert_eq!(s, AccelStatus::NotLoadable("mkl_rt.dll not found".into()));
        assert_eq!(registered.load(Ordering::SeqCst), 0);
        // A second call does not re-probe: the recorded failure stands.
        let again = install_once(&cell, || Ok(()), || registered.store(99, Ordering::SeqCst));
        assert_eq!(again, s);
        assert_eq!(registered.load(Ordering::SeqCst), 0);
    }

    #[cfg(any(feature = "onemkl", feature = "aocl"))]
    #[test]
    fn ensure_installed_records_a_status_once_with_the_features_on() {
        ensure_installed();
        let first = cpu_accel_report();
        ensure_installed();
        let expected = usize::from(cfg!(feature = "onemkl")) + usize::from(cfg!(feature = "aocl"));
        assert_eq!(first.len(), expected);
        assert_eq!(cpu_accel_report(), first);
    }
}

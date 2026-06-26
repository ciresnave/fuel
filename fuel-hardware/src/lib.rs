//! Hardware discovery for fuel — device enumeration + probe reports.
//! (Transfer calibration and the topology discovery-half follow in later
//! B0.2 sub-steps of the fuel-core retirement.)
//!
//! Split out of `fuel-core` so discovery is reusable independent of the
//! lazy/dispatch machinery. Depends only on `fuel-ir` (vocabulary) + the
//! backend crates' device enumerators. It must NOT depend on `fuel-dispatch`
//! or `fuel-core` — the dependency points the other way:
//! `fuel-dispatch` / `fuel-core` → `fuel-hardware` → `fuel-ir`.

pub mod enumerate;
pub mod probe;

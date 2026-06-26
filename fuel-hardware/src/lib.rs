//! Hardware discovery for fuel — device enumeration, probe reports, and
//! transfer (bandwidth) calibration. (The topology discovery-half follows in
//! a later B0.2 sub-step of the fuel-core retirement.)
//!
//! Split out of `fuel-core` so discovery is reusable independent of the
//! lazy/dispatch machinery. Depends only on `fuel-ir` (vocabulary) + the
//! backend crates' device enumerators / byte-storage APIs. It must NOT depend
//! on `fuel-dispatch` or `fuel-core` — the dependency points the other way:
//! `fuel-dispatch` / `fuel-core` → `fuel-hardware` → `fuel-ir`.

pub mod enumerate;
pub mod probe;
pub mod transfer_cost;

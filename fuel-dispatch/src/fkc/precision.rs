//! [`PrecisionBlock`] → [`PrecisionGuarantee`] lowering (adoption plan
//! §2.1 / FKC §4.8).
//!
//! The mapping (exactly the plan's rule):
//! - `audited: false` + all-null bounds  ⇒ [`PrecisionGuarantee::UNAUDITED`].
//! - `audited: true`  + all-null bounds  ⇒ [`PrecisionGuarantee::none(notes)`]
//!   (audited, "no static bound applies", the reason in `notes`).
//! - any bound present                   ⇒ a populated struct
//!   (`max_ulp` / `max_relative` / `max_absolute` /
//!   `bit_stable_on_same_hardware` mapped through).
//! - a bare placeholder (no `audited`, no bounds, no notes) ⇒
//!   [`FkcError::PlaceholderPrecision`] (there is nothing to lower).
//!
//! `notes` lowers to a `&'static str` via an intern leak (process-lifetime,
//! bounded by the number of distinct contract notes — the same posture the
//! plan §1.3 takes for `kernel_source`). The lifetime mismatch is real:
//! `PrecisionGuarantee.notes` is `&'static str`; a contract's notes are
//! read from a file. We intern through a small `OnceLock<Mutex<HashSet>>`.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::fkc::error::FkcError;
use crate::fkc::schema::PrecisionBlock;
use crate::fused::PrecisionGuarantee;

/// Process-lifetime string interner for precision `notes`. Returns a
/// `&'static str` for the supplied string, leaking it on first sighting.
/// Bounded by the number of distinct notes across imported contracts.
fn intern(s: &str) -> &'static str {
    static POOL: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = pool.lock().expect("precision notes interner poisoned");
    if let Some(existing) = guard.get(s) {
        return existing;
    }
    let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
    guard.insert(leaked);
    leaked
}

/// Read a bound field that the schema carries as a raw `serde_yml::Value`
/// (it may be `~`/null, an integer, or a float). Returns `None` for
/// null/absent, `Some(f64)` for a numeric, and an error for a non-numeric.
fn read_f64(
    v: &Option<serde_yml::Value>,
    section: &str,
    field: &str,
) -> Result<Option<f64>, FkcError> {
    match v {
        None => Ok(None),
        Some(serde_yml::Value::Null) => Ok(None),
        Some(serde_yml::Value::Number(n)) => Ok(n.as_f64()),
        Some(other) => Err(FkcError::Yaml(format!(
            "section `{section}`: precision field `{field}` must be a number or null, got {other:?}"
        ))),
    }
}

/// Same as [`read_f64`] but coerces to `u32` for `max_ulp`.
fn read_u32(
    v: &Option<serde_yml::Value>,
    section: &str,
    field: &str,
) -> Result<Option<u32>, FkcError> {
    Ok(read_f64(v, section, field)?.map(|f| f as u32))
}

/// Lower a [`PrecisionBlock`] into a [`PrecisionGuarantee`].
///
/// `section` names the kernel for error context. A `None` precision block
/// (the operand omitted it entirely) lowers to [`PrecisionGuarantee::UNAUDITED`]
/// — the conservative "no claim" default the binding table already uses.
pub fn lower_precision(
    block: Option<&PrecisionBlock>,
    section: &str,
) -> Result<PrecisionGuarantee, FkcError> {
    let Some(b) = block else {
        // No precision block at all → UNAUDITED (same as a plain
        // `register(...)` call's default).
        return Ok(PrecisionGuarantee::UNAUDITED);
    };

    let max_ulp = read_u32(&b.max_ulp, section, "max_ulp")?;
    let max_relative = read_f64(&b.max_relative, section, "max_relative")?;
    let max_absolute = read_f64(&b.max_absolute, section, "max_absolute")?;
    let has_bound = max_ulp.is_some() || max_relative.is_some() || max_absolute.is_some();
    let bit_stable = b.bit_stable_on_same_hardware.unwrap_or(false);
    let has_notes = b.notes.as_deref().is_some_and(|n| !n.trim().is_empty());

    // A bare placeholder: nothing declared at all.
    if b.audited.is_none() && !has_bound && !has_notes && b.bit_stable_on_same_hardware.is_none() {
        return Err(FkcError::PlaceholderPrecision {
            section: section.to_string(),
        });
    }

    // Bounds present (or a bit-stable flag) ⇒ a populated struct. We map
    // every declared field through; `notes` is interned to `&'static`.
    if has_bound {
        let notes: &'static str = match b.notes.as_deref() {
            Some(n) if !n.trim().is_empty() => intern(n),
            _ => "",
        };
        return Ok(PrecisionGuarantee {
            bit_stable_on_same_hardware: bit_stable,
            max_ulp,
            max_relative,
            max_absolute,
            notes,
        });
    }

    // No bound. Branch on `audited`.
    match b.audited {
        Some(false) => Ok(PrecisionGuarantee::UNAUDITED),
        Some(true) => {
            // audited:true + all-null + (a bit_stable flag and/or notes).
            // If a bit_stable claim is made with no static bound, that is a
            // real (populated) claim, not "no bound applies" — emit it.
            if bit_stable {
                let notes: &'static str = match b.notes.as_deref() {
                    Some(n) if !n.trim().is_empty() => intern(n),
                    _ => "",
                };
                Ok(PrecisionGuarantee {
                    bit_stable_on_same_hardware: true,
                    max_ulp: None,
                    max_relative: None,
                    max_absolute: None,
                    notes,
                })
            } else {
                // audited, concluded no static bound applies → none(reason).
                let reason: &'static str = match b.notes.as_deref() {
                    Some(n) if !n.trim().is_empty() => intern(n),
                    _ => "audited; no static precision bound applies",
                };
                Ok(PrecisionGuarantee::none(reason))
            }
        }
        None => {
            // No `audited` flag, but a bit_stable claim and/or notes were
            // present (otherwise we'd have hit the placeholder branch).
            if bit_stable {
                let notes: &'static str = match b.notes.as_deref() {
                    Some(n) if !n.trim().is_empty() => intern(n),
                    _ => "",
                };
                Ok(PrecisionGuarantee {
                    bit_stable_on_same_hardware: true,
                    max_ulp: None,
                    max_relative: None,
                    max_absolute: None,
                    notes,
                })
            } else {
                // Only notes, no bound, no bit-stable, no audited flag:
                // treat as an unaudited claim carrying a reason.
                let reason: &'static str = match b.notes.as_deref() {
                    Some(n) if !n.trim().is_empty() => intern(n),
                    _ => unreachable!("placeholder branch already returned"),
                };
                Ok(PrecisionGuarantee::none(reason))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(v: f64) -> Option<serde_yml::Value> {
        Some(serde_yml::Value::Number(v.into()))
    }

    #[test]
    fn audited_false_all_null_is_unaudited() {
        let block = PrecisionBlock {
            bit_stable_on_same_hardware: None,
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            audited: Some(false),
            notes: Some("not yet audited".into()),
        };
        let p = lower_precision(Some(&block), "k").unwrap();
        assert_eq!(p.notes, PrecisionGuarantee::UNAUDITED.notes);
        assert!(!p.bit_stable_on_same_hardware);
    }

    #[test]
    fn audited_true_all_null_no_bitstable_is_none_with_reason() {
        let block = PrecisionBlock {
            bit_stable_on_same_hardware: Some(false),
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            audited: Some(true),
            notes: Some("scheduler-dependent FADD order".into()),
        };
        let p = lower_precision(Some(&block), "k").unwrap();
        assert!(!p.bit_stable_on_same_hardware);
        assert_eq!(p.notes, "scheduler-dependent FADD order");
        assert!(p.max_ulp.is_none());
    }

    #[test]
    fn bounds_present_populates_struct() {
        // The real add_f32 contract: bit_stable true, max_ulp 0, audited true.
        let block = PrecisionBlock {
            bit_stable_on_same_hardware: Some(true),
            max_ulp: num(0.0),
            max_relative: None,
            max_absolute: None,
            audited: Some(true),
            notes: Some("Exact IEEE-754 f32 addition.".into()),
        };
        let p = lower_precision(Some(&block), "add_f32").unwrap();
        assert!(p.bit_stable_on_same_hardware);
        assert_eq!(p.max_ulp, Some(0));
        assert_eq!(p.notes, "Exact IEEE-754 f32 addition.");
    }

    #[test]
    fn bitstable_true_all_null_is_a_real_claim_not_none() {
        // The real qmatmul contract: bit_stable true, all bounds ~, audited true.
        let block = PrecisionBlock {
            bit_stable_on_same_hardware: Some(true),
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            audited: Some(true),
            notes: Some("f32 accumulate; deterministic.".into()),
        };
        let p = lower_precision(Some(&block), "qmatmul").unwrap();
        assert!(p.bit_stable_on_same_hardware);
        assert!(p.max_ulp.is_none());
        assert_eq!(p.notes, "f32 accumulate; deterministic.");
    }

    #[test]
    fn bare_placeholder_is_error() {
        let block = PrecisionBlock {
            bit_stable_on_same_hardware: None,
            max_ulp: None,
            max_relative: None,
            max_absolute: None,
            audited: None,
            notes: None,
        };
        let err = lower_precision(Some(&block), "k").expect_err("placeholder errors");
        assert!(matches!(err, FkcError::PlaceholderPrecision { .. }), "got {err:?}");
    }

    #[test]
    fn absent_block_is_unaudited() {
        let p = lower_precision(None, "k").unwrap();
        assert_eq!(p.notes, PrecisionGuarantee::UNAUDITED.notes);
    }
}

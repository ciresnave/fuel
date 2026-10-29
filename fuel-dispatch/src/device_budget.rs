// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where a pinned device's byte budget comes from (ROADMAP row 67, increment I2).
//!
//! `PlanOptions::device_budget_bytes` (I1) takes a number. This module is the policy that produces
//! one, and says which rule produced it so a plan report never hides a guessed budget:
//!
//! 1. an **explicit** value from the caller wins;
//! 2. else the [`DEVICE_BUDGET_ENV`] override (tests, or a box with a shared GPU);
//! 3. else the **probed** free memory of the device times a reserve fraction
//!    ([`DEFAULT_RESERVE_FRACTION`]).
//!
//! Nothing here is on by default: `PlanOptions` still carries no budget until a caller resolves one
//! and attaches it with `PlanOptions::with_resolved_device_budget`. The eviction pass only moves a
//! reload, it does not yet delay it (row 67 I3), so a default-on budget would add copies without
//! lowering the analysed peak.

use fuel_backend_contract::backend::BackendRuntime;

/// Environment variable holding an explicit budget in bytes (decimal).
pub const DEVICE_BUDGET_ENV: &str = "FUEL_DEVICE_BUDGET_BYTES";

/// Share of the probed free memory a probed budget keeps (the rest is headroom for the driver, other
/// processes, and allocations the analysis does not see).
pub const DEFAULT_RESERVE_FRACTION: f64 = 0.9;

/// Which rule produced a budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetSource {
    /// The caller passed the number.
    Explicit,
    /// [`DEVICE_BUDGET_ENV`] supplied it.
    Env,
    /// `available_bytes * fraction`, from the device's live probe.
    Probed {
        /// The device's free bytes when probed.
        available: u64,
        /// The fraction of `available` kept as budget.
        fraction: f64,
    },
}

/// A budget and the rule that produced it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedBudget {
    /// The budget, in bytes.
    pub bytes: usize,
    /// Which rule produced `bytes`.
    pub source: BudgetSource,
}

/// Why a budget could not be resolved. A bad override is an error, never a silent fallback to the
/// probe: the caller asked for a specific number.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BudgetError {
    /// [`DEVICE_BUDGET_ENV`] was set but is not a decimal byte count.
    #[error("{DEVICE_BUDGET_ENV}={0:?} is not a decimal byte count")]
    BadEnv(String),
    /// The reserve fraction was outside `(0, 1]` or not finite.
    #[error("reserve fraction {0} must be finite and in (0, 1]")]
    BadFraction(f64),
}

/// Resolve a budget from its three sources. `env` is the raw value of [`DEVICE_BUDGET_ENV`]
/// (`None` when unset); `available` is the device's free bytes (`None` when it cannot say).
///
/// Returns `Ok(None)` when no rule applies: nothing explicit, no override, and the device reports
/// no usable figure (`None`, or `u64::MAX`, the "unbounded" convention of reference backends).
/// `fraction` is validated even when an earlier rule wins, so a bad constant fails in every test
/// run instead of only on a box that reaches the probe.
pub fn resolve_device_budget(
    explicit: Option<usize>,
    env: Option<&str>,
    available: Option<u64>,
    fraction: f64,
) -> Result<Option<ResolvedBudget>, BudgetError> {
    if !(fraction.is_finite() && fraction > 0.0 && fraction <= 1.0) {
        return Err(BudgetError::BadFraction(fraction));
    }
    if let Some(bytes) = explicit {
        return Ok(Some(ResolvedBudget {
            bytes,
            source: BudgetSource::Explicit,
        }));
    }
    if let Some(raw) = env {
        let bytes = raw
            .trim()
            .parse::<usize>()
            .map_err(|_| BudgetError::BadEnv(raw.to_string()))?;
        return Ok(Some(ResolvedBudget {
            bytes,
            source: BudgetSource::Env,
        }));
    }
    Ok(match available {
        Some(avail) if avail != u64::MAX => Some(ResolvedBudget {
            bytes: (avail as f64 * fraction) as usize,
            source: BudgetSource::Probed {
                available: avail,
                fraction,
            },
        }),
        _ => None,
    })
}

/// [`resolve_device_budget`] reading the process environment and a live runtime handle.
pub fn resolve_device_budget_for(
    explicit: Option<usize>,
    runtime: &dyn BackendRuntime,
    fraction: f64,
) -> Result<Option<ResolvedBudget>, BudgetError> {
    let env = std::env::var(DEVICE_BUDGET_ENV).ok();
    resolve_device_budget(
        explicit,
        env.as_deref(),
        runtime.available_bytes(),
        fraction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const F: f64 = DEFAULT_RESERVE_FRACTION;

    #[test]
    fn explicit_beats_env_and_probe() {
        let r = resolve_device_budget(Some(7), Some("9"), Some(1000), F).unwrap();
        assert_eq!(
            r,
            Some(ResolvedBudget {
                bytes: 7,
                source: BudgetSource::Explicit
            })
        );
    }

    #[test]
    fn env_beats_probe() {
        let r = resolve_device_budget(None, Some(" 4096 "), Some(1000), F).unwrap();
        assert_eq!(
            r,
            Some(ResolvedBudget {
                bytes: 4096,
                source: BudgetSource::Env
            })
        );
    }

    #[test]
    fn probe_keeps_the_reserve_fraction_and_records_its_inputs() {
        let r = resolve_device_budget(None, None, Some(1000), 0.9)
            .unwrap()
            .unwrap();
        assert_eq!(r.bytes, 900);
        assert_eq!(
            r.source,
            BudgetSource::Probed {
                available: 1000,
                fraction: 0.9
            }
        );
    }

    #[test]
    fn a_bad_env_value_is_an_error_not_a_fallback_to_the_probe() {
        // Sabotage sibling of `env_beats_probe`: with a probe available, a fallback would return Some.
        assert_eq!(
            resolve_device_budget(None, Some("8GB"), Some(1000), F),
            Err(BudgetError::BadEnv("8GB".into()))
        );
        assert!(resolve_device_budget(None, Some(""), Some(1000), F).is_err());
        assert!(resolve_device_budget(None, Some("-1"), Some(1000), F).is_err());
    }

    #[test]
    fn no_usable_probe_and_no_override_means_no_budget() {
        assert_eq!(resolve_device_budget(None, None, None, F), Ok(None));
        // Reference backends advertise "unbounded" as u64::MAX; that is not a budget.
        assert_eq!(
            resolve_device_budget(None, None, Some(u64::MAX), F),
            Ok(None)
        );
        // Control: a finite figure on the same path does resolve.
        assert!(
            resolve_device_budget(None, None, Some(1), F)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn the_fraction_is_validated_even_when_an_earlier_rule_wins() {
        for bad in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    resolve_device_budget(Some(1), None, None, bad),
                    Err(BudgetError::BadFraction(_))
                ),
                "fraction {bad}"
            );
        }
        assert!(resolve_device_budget(Some(1), None, None, 1.0).is_ok());
    }
}

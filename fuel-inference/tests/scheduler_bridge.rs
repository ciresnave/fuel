//! Phase 6d Track 4 integration test: `MemoryPressureRule` from the
//! fuel-inference / planner bridge correctly biases placement under
//! pressure and is a no-op otherwise.
//!
//! Lives as an integration test (not a `#[cfg(test)] mod tests`)
//! because fuel-inference's lib-test target has unrelated pre-existing
//! compile errors in `prefix_cache` / `speculative` (Device::Cpu API
//! drift) that would block the whole lib-test run.
//!
//! Session 6 (2026-06-11): fuel-graph-router retired; the rule is now
//! self-contained (plain `Placement` map, no Router / SchedulerRule).

use fuel_core_types::{DeviceLocation, Shape};
use fuel_graph::Tensor;
use fuel_inference::scheduler::{MemoryScheduler, Priority, RequestInfo};
use fuel_inference::scheduler_bridge::{
    MemoryPressureRule, MemoryPressureSnapshot, Placement,
};

/// Phase 7.5 G2: tests need a real device for slot-populating
/// constructors. Singleton CpuBackendDevice via OnceLock.
fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
    static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}

#[test]
fn rule_no_op_when_not_under_pressure() {
    let a = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
    let b = a.const_f32_like(vec![2.0_f32; 4], Shape::from_dims(&[4]));
    let c = a.add(&b);

    let snapshot = MemoryPressureSnapshot { under_pressure: false, usage_fraction: 0.1 };
    let rule = MemoryPressureRule::new(snapshot);
    let mut placement = Placement::new();
    rule.apply(c.graph(), &[c.id()], &mut placement);
    assert!(placement.is_empty(),
        "rule must not touch placement when under_pressure=false");
}

#[test]
fn rule_inherits_first_input_placement_under_pressure() {
    let a = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
    let b = a.relu();
    let c = b.relu();

    let snapshot = MemoryPressureSnapshot { under_pressure: true, usage_fraction: 0.95 };
    let rule = MemoryPressureRule::new(snapshot);
    let mut placement = Placement::new();
    placement.insert(a.id(), DeviceLocation::Cpu);

    rule.apply(c.graph(), &[c.id()], &mut placement);
    assert_eq!(placement.get(&a.id()), Some(&DeviceLocation::Cpu));
    assert_eq!(placement.get(&b.id()), Some(&DeviceLocation::Cpu),
        "b should inherit a's placement under pressure");
    assert_eq!(placement.get(&c.id()), Some(&DeviceLocation::Cpu),
        "c should inherit b's (= a's) placement under pressure");
}

#[test]
fn snapshot_round_trips_through_memory_scheduler() {
    let mut s = MemoryScheduler::new(1000);
    let admitted = s.try_admit(RequestInfo::new("req-a", 950, Priority::High));
    assert!(admitted.is_some());
    let snap = MemoryPressureSnapshot::from(&s);
    // 950/1000 = 0.95 > default threshold (0.9) → under pressure.
    assert!(snap.under_pressure);
    assert!((snap.usage_fraction - 0.95).abs() < 1e-9);
}

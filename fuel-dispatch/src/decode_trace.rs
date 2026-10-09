// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-realize host-side timing breakdown, gated by `FUEL_DECODE_TRACE`
//! (any non-empty value enables it; unset is the default, byte-identical
//! fast path — every call here is a no-op check against a cached bool).
//!
//! Built to answer one question the steady-state Qwen3-0.6B CUDA decode
//! number (2.76 tok/s, fuel#321) left open: where does the per-step time
//! go? [`PipelinedExecutor::realize_inner`]'s work-item loop never issues a
//! CUDA-graph replay (no [`crate::pipelined::PipelinedExecutor::capture_decode`]
//! call on this path) — it dispatches every node's kernel individually, so a
//! single decode token for a ~28-layer model walks several hundred
//! `execute_work_item` calls. This module buckets the host-side wall time of
//! each one into h2d copy / kernel / gemm-or-qmatmul / everything-else, plus
//! the realize-end device sync, and reports one line per realize call.
//!
//! Host wall time around an async CUDA launch call measures *dispatch*
//! (host-side driver round-trip to queue the launch), not device execution
//! time — that is the quantity "kernel launch overhead" names, and it is
//! exactly the thing a few-hundred-tiny-ops-per-token decode path is
//! hypothesized to be bound by. The realize-end `sync_active_cuda_devices`
//! call is where any outstanding device work (including real GEMM compute
//! time this breakdown cannot see directly) actually gets waited on, so its
//! reported duration is the floor on "device busy, not yet accounted for by
//! any async dispatch above."
use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Default)]
pub(crate) struct StepTrace {
    /// `insert_safety_copies` + `compiler_work_and_wait_set_for`: the
    /// per-call graph-walk `realize_inner` does BEFORE the compiler thread
    /// is even spawned — unconditionally, every call, including on the
    /// plan-once/prebuilt fast path (that path skips `optimize_graph`, not
    /// this walk).
    pub setup_time: Duration,
    /// The `insert_safety_copies` call alone, within `setup_time`.
    pub setup_safety_copies_time: Duration,
    /// The `compiler_work_and_wait_set_for` call alone, within `setup_time`.
    pub setup_compiler_work_time: Duration,
    /// Time the executor thread spends blocked in `rx.recv()` waiting for
    /// the compiler thread to resolve + emit the NEXT `WorkItem`. High recv
    /// time means the bottleneck is the compiler thread's per-node
    /// binding-table resolution, not kernel dispatch.
    pub recv_time: Duration,
    pub h2d_count: u64,
    pub h2d_bytes: u64,
    pub h2d_time: Duration,
    pub kernel_count: u64,
    pub kernel_time: Duration,
    pub gemm_count: u64,
    pub gemm_time: Duration,
    pub other_count: u64,
    pub other_time: Duration,
    pub sync_time: Duration,
}

thread_local! {
    static TRACE: RefCell<Option<StepTrace>> = const { RefCell::new(None) };
    static STEP_COUNTER: RefCell<u64> = const { RefCell::new(0) };
}

/// Cached once per thread-unaware process lifetime (env vars don't change
/// mid-run in any caller here) so the hot loop pays one atomic-free
/// `OnceLock` read, not a `std::env::var` syscall per work item.
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FUEL_DECODE_TRACE").is_some())
}

/// Starts a fresh accumulator for one `realize_inner` call. A no-op, and
/// cheap even when checked, when tracing is disabled.
pub(crate) fn begin_step() {
    if !enabled() {
        return;
    }
    TRACE.with(|t| *t.borrow_mut() = Some(StepTrace::default()));
}

pub(crate) fn record_setup(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.setup_time += dt;
        }
    });
}

pub(crate) fn record_setup_safety_copies(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.setup_safety_copies_time += dt;
        }
    });
}

pub(crate) fn record_setup_compiler_work(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.setup_compiler_work_time += dt;
        }
    });
}

pub(crate) fn record_recv(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.recv_time += dt;
        }
    });
}

pub(crate) fn record_h2d(bytes: u64, dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.h2d_count += 1;
            s.h2d_bytes += bytes;
            s.h2d_time += dt;
        }
    });
}

pub(crate) fn record_kernel(is_gemm: bool, dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            if is_gemm {
                s.gemm_count += 1;
                s.gemm_time += dt;
            } else {
                s.kernel_count += 1;
                s.kernel_time += dt;
            }
        }
    });
}

pub(crate) fn record_other(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.other_count += 1;
            s.other_time += dt;
        }
    });
}

pub(crate) fn record_sync(dt: Duration) {
    if !enabled() {
        return;
    }
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().as_mut() {
            s.sync_time += dt;
        }
    });
}

/// Ends the current accumulator and prints it to stderr tagged with a
/// per-thread monotonic step counter — the caller has no per-decode-step
/// label to hand in (that would mean threading a new parameter through
/// every `realize_inner` caller for a diagnostic-only feature), so the
/// counter is the correlation key: callers reconstruct "which realize was
/// this" from call order (e.g. `m1_cuda_speed_harness`'s own prefill /
/// first-decode / steady-state split).
pub(crate) fn end_step_and_report() {
    if !enabled() {
        return;
    }
    let step = STEP_COUNTER.with(|c| {
        let mut c = c.borrow_mut();
        *c += 1;
        *c
    });
    TRACE.with(|t| {
        if let Some(s) = t.borrow_mut().take() {
            eprintln!(
                "[decode-trace] step={step} \
                 setup: t={:.6}s (safety_copies={:.6}s compiler_work={:.6}s) | \
                 recv: t={:.6}s | \
                 h2d: n={} bytes={} t={:.6}s | \
                 kernel: n={} t={:.6}s | \
                 gemm: n={} t={:.6}s | \
                 other: n={} t={:.6}s | \
                 sync: t={:.6}s",
                s.setup_time.as_secs_f64(),
                s.setup_safety_copies_time.as_secs_f64(),
                s.setup_compiler_work_time.as_secs_f64(),
                s.recv_time.as_secs_f64(),
                s.h2d_count,
                s.h2d_bytes,
                s.h2d_time.as_secs_f64(),
                s.kernel_count,
                s.kernel_time.as_secs_f64(),
                s.gemm_count,
                s.gemm_time.as_secs_f64(),
                s.other_count,
                s.other_time.as_secs_f64(),
                s.sync_time.as_secs_f64(),
            );
        }
    });
}

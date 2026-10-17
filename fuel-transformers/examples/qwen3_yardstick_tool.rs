// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3-0.6B CPU yardstick tools (needs network on an empty HF cache).
//!
//! ```text
//! cargo run -p fuel-transformers --features dev-tools --example qwen3_yardstick_tool -- regenerate
//! cargo run -p fuel-transformers --features dev-tools --release --example qwen3_yardstick_tool -- speed
//! ```
//!
//! `regenerate` rewrites `tests/fixtures/qwen3_0_6b_cpu_reference.json` (after
//! the plan's reduction-order calibration check); `speed` prints prefill/decode
//! tok/s. These were `#[ignore]`d tests; a tool that is not an assertion is a
//! program, not a skipped test.

use fuel_transformers::dev_tools::qwen3_yardstick as y;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("regenerate") => y::regenerate_qwen3_cpu_reference_fixture(),
        Some("speed") => y::qwen3_cpu_speed_harness(),
        other => {
            eprintln!("usage: qwen3_yardstick_tool <regenerate|speed> (got {other:?})");
            std::process::exit(2);
        }
    }
}

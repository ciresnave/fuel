// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3-0.6B decode on the Vulkan backend: step timing and a per-op host-time
//! profile (ROADMAP 17 H12 follow-up; see
//! `docs/research/qwen3-vulkan-decode-profile.md`).
//!
//! ```text
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-transformers --features dev-tools,vulkan --example qwen3_vulkan_profile \
//!     -- [runs=5] [steady_steps=10]
//! ```
//!
//! Needs the Hugging Face cache (or network) for the Q4_K_M GGUF and the
//! tokenizer, as `qwen3_yardstick_tool` does. A program, not an `#[ignore]`
//! test (the ignore total sits at its ceiling).
//!
//! The per-op table is `fuel_dispatch::compiled::op_profile`: HOST wall time of
//! each kernel launch, summed per `(OpKind, backend)`. A kernel that only
//! enqueues work is charged its enqueue cost, not its device time; the rest of a
//! step's wall time (the part no launch accounts for) is reported as `other`.

use fuel_dispatch::compiled::op_profile;
use fuel_ir::DType;
use fuel_tensor::inference_context::{DecodeSession, InferenceContext, KvCache};
use fuel_transformers::dev_tools::qwen3_yardstick::{PROMPTS, load_model};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One run's measurements.
struct Run {
    prefill: Duration,
    first_decode: Duration,
    /// Wall time of each profiled steady decode step.
    steps: Vec<Duration>,
    /// `(op, backend) -> (launches, host time)` summed over the steady steps.
    ops: BTreeMap<(String, String), (u64, Duration)>,
}

fn main() {
    let arg = |i: usize, d: usize| {
        std::env::args()
            .nth(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or(d)
    };
    let (runs, steady) = (arg(1, 5), arg(2, 10));
    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let dev = fuel_tensor::vulkan_backend::new_device().expect("vulkan device");
    let formatted = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        PROMPTS[0]
    );
    let prompt_tokens = tokenizer
        .encode(formatted, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let max_seq_len = prompt_tokens.len() + steady + 2;

    let run_once = || -> Run {
        let mut cache = KvCache::with_capacity(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq_len,
            DType::F32,
            &dev,
        )
        .expect("KvCache::with_capacity");
        let mut ctx = InferenceContext::new(dev.clone());
        let mut session: Option<DecodeSession> = None;
        let mut step = |tokens: &[u32]| {
            let t = Instant::now();
            let logits = model
                .forward_with_kv_context_persistent(tokens, &mut cache, &mut ctx, &mut session)
                .expect("forward");
            (argmax(&logits), t.elapsed())
        };
        let (mut next, prefill) = step(&prompt_tokens);
        let (n, first_decode) = step(&[next]);
        next = n;
        op_profile::set_enabled(true);
        let mut steps = Vec::new();
        for _ in 0..steady {
            let (n, dt) = step(&[next]);
            next = n;
            steps.push(dt);
        }
        op_profile::set_enabled(false);
        let ops = op_profile::take()
            .into_iter()
            .map(|(op, be, n, d)| ((op, be), (n, d)))
            .collect();
        Run {
            prefill,
            first_decode,
            steps,
            ops,
        }
    };

    let _ = run_once(); // warm-up, discarded (and clears nothing it needs)
    let all: Vec<Run> = (0..runs).map(|_| run_once()).collect();

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let mut prefill: Vec<f64> = all.iter().map(|r| ms(r.prefill)).collect();
    let mut first: Vec<f64> = all.iter().map(|r| ms(r.first_decode)).collect();
    let mut step_med: Vec<f64> = all
        .iter()
        .map(|r| median(&mut r.steps.iter().map(|d| ms(*d)).collect::<Vec<_>>()))
        .collect();
    let spread = |v: &[f64]| {
        let (lo, hi) = v
            .iter()
            .fold((f64::MAX, 0f64), |(l, h), x| (l.min(*x), h.max(*x)));
        (lo, hi)
    };
    println!(
        "prompt {} tokens; {runs} runs x {steady} steady decode steps (+1 warm-up run)",
        prompt_tokens.len()
    );
    let (lo, hi) = spread(&prefill);
    println!(
        "prefill ms: median {:.0} (min {lo:.0}, max {hi:.0})",
        median(&mut prefill)
    );
    let (lo, hi) = spread(&first);
    println!(
        "first decode call ms: median {:.0} (min {lo:.0}, max {hi:.0})",
        median(&mut first)
    );
    let (lo, hi) = spread(&step_med);
    let step_all = median(&mut step_med);
    println!(
        "steady decode step ms (per-run medians): median {step_all:.0} (min {lo:.0}, max {hi:.0})"
    );

    // Per-op: per-run, per-step average host ms; then median / spread across runs.
    let mut keys: Vec<(String, String)> = all.iter().flat_map(|r| r.ops.keys().cloned()).collect();
    keys.sort();
    keys.dedup();
    let mut rows: Vec<(f64, String, f64, f64, f64, u64)> = Vec::new();
    for k in keys {
        let mut per_step: Vec<f64> = all
            .iter()
            .map(|r| {
                r.ops
                    .get(&k)
                    .map_or(0.0, |(_, d)| ms(*d) / r.steps.len() as f64)
            })
            .collect();
        let launches = all[0]
            .ops
            .get(&k)
            .map_or(0, |(n, _)| n / all[0].steps.len() as u64);
        let (lo, hi) = spread(&per_step);
        let m = median(&mut per_step);
        rows.push((
            m,
            format!("{}/{}", k.0, k.1),
            lo,
            hi,
            m / step_all * 100.0,
            launches,
        ));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "\nper-op HOST time per decode step (median over runs; min-max; share of step wall; launches/step):"
    );
    let mut accounted = 0.0;
    for (m, name, lo, hi, pct, n) in &rows {
        accounted += m;
        println!("  {name:<28} {m:>9.2} ms  ({lo:.2}-{hi:.2})  {pct:>5.1}%  {n:>5}");
    }
    println!(
        "  {:<28} {:>9.2} ms  {:>5.1}%   (step wall minus the launches above: planning, waits, flushes, readback)",
        "other",
        step_all - accounted,
        (step_all - accounted) / step_all * 100.0
    );
}

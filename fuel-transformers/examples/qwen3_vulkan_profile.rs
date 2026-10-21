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

use fuel_dispatch::compiled::{op_profile, op_profile::LaunchRow};
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
    /// Every launch of one extra (unprofiled-by-the-table) decode step, in order.
    trace: Vec<LaunchRow>,
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
    // What the planner believes a Vulkan<->host copy costs (the numbers it
    // prices placement with), to set beside the measured copy times.
    {
        use fuel_ir::DeviceLocation;
        let topo = fuel_dispatch::topology::SystemTopology::current();
        let vk = DeviceLocation::Vulkan { gpu_id: 0 };
        for bytes in [4u64, 4096, 4 << 20, 12_582_912, 622_329_856] {
            println!(
                "planner estimate {bytes:>10} B: Vulkan->Cpu {:>10.3} ms, Cpu->Vulkan {:>10.3} ms",
                topo.estimate_transfer_ns(vk, DeviceLocation::Cpu, bytes) as f64 / 1e6,
                topo.estimate_transfer_ns(DeviceLocation::Cpu, vk, bytes) as f64 / 1e6,
            );
        }
    }
    let formatted = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        PROMPTS[0]
    );
    let prompt_tokens = tokenizer
        .encode(formatted, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let max_seq_len = prompt_tokens.len() + steady + 4;

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
        let mut step = |tokens: &[u32], reset: bool| {
            if reset {
                session = None;
            }
            let t = Instant::now();
            let logits = model
                .forward_with_kv_context_persistent(tokens, &mut cache, &mut ctx, &mut session)
                .expect("forward");
            (argmax(&logits), t.elapsed())
        };
        let (mut next, prefill) = step(&prompt_tokens, false);
        let (n, first_decode) = step(&[next], false);
        next = n;
        op_profile::set_enabled(true);
        let mut steps = Vec::new();
        for _ in 0..steady {
            let (n, dt) = step(&[next], false);
            next = n;
            steps.push(dt);
        }
        op_profile::set_enabled(false);
        let ops = op_profile::take()
            .into_iter()
            .map(|(op, be, n, d)| ((op, be), (n, d)))
            .collect();
        // QWEN3_REPLAN=1: drop the decode session and rebuild it (re-plan) with
        // the same context, so the planner sees whatever residency the first
        // session left behind; the traced step below is then the re-planned one.
        if std::env::var_os("QWEN3_REPLAN").is_some() {
            let (n, _) = step(&[next], true);
            next = n;
        }
        op_profile::set_trace(true);
        let (_, _) = step(&[next], false);
        op_profile::set_trace(false);
        let trace = op_profile::take_trace();
        let _ = op_profile::take();
        Run {
            prefill,
            first_decode,
            steps,
            ops,
            trace,
        }
    };

    let _ = run_once(); // warm-up, discarded (and clears nothing it needs)
    let all: Vec<Run> = (0..runs).map(|_| run_once()).collect();

    copy_anatomy(&all.last().expect("at least one run").trace);
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

fn bucket(bytes: usize) -> &'static str {
    match bytes {
        0..=64 => "<=64 B",
        65..=4096 => "<=4 KB",
        4097..=65_536 => "<=64 KB",
        65_537..=1_048_576 => "<=1 MB",
        1_048_577..=16_777_216 => "<=16 MB",
        _ => ">16 MB",
    }
}

fn med(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Direction, size and trigger of every `Copy` launch in one traced decode step.
fn copy_anatomy(trace: &[LaunchRow]) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let copies: Vec<&LaunchRow> = trace.iter().filter(|r| r.op == "Copy").collect();
    println!(
        "\ncopy anatomy, ONE traced decode step: {} launches, {} are Copy",
        trace.len(),
        copies.len()
    );
    let mut dir: BTreeMap<String, (u64, f64, usize)> = BTreeMap::new();
    for c in &copies {
        let e = dir
            .entry(format!("{}->{} (kernel {})", c.src, c.dst, c.backend))
            .or_default();
        e.0 += 1;
        e.1 += ms(c.dt);
        e.2 += c.bytes;
    }
    println!("  by direction:");
    for (k, (n, t, b)) in &dir {
        println!("    {k:<34} {n:>4} copies  {t:>9.1} ms  {b:>12} bytes");
    }
    let mut sizes: BTreeMap<(String, &'static str), Vec<f64>> = BTreeMap::new();
    for c in &copies {
        sizes
            .entry((format!("{}->{}", c.src, c.dst), bucket(c.bytes)))
            .or_default()
            .push(ms(c.dt));
    }
    println!("  by direction and size (per-copy ms: median / max):");
    for ((d, b), v) in &mut sizes {
        let max = v.iter().cloned().fold(0.0, f64::max);
        println!(
            "    {d:<14} {b:<9} {:>4} copies  total {:>8.1} ms  median {:>7.2}  max {max:>7.2}",
            v.len(),
            v.iter().sum::<f64>(),
            med(v)
        );
    }
    // What launched just before each Copy (the nearest earlier non-Copy launch).
    let mut prev: BTreeMap<String, (u64, f64)> = BTreeMap::new();
    let mut last_non_copy = "(start)".to_string();
    for r in trace {
        if r.op == "Copy" {
            let e = prev.entry(last_non_copy.clone()).or_default();
            e.0 += 1;
            e.1 += ms(r.dt);
        } else {
            last_non_copy = format!("{}/{}", r.op, r.backend);
        }
    }
    println!("  copies by the nearest earlier non-Copy launch:");
    let mut p: Vec<_> = prev.into_iter().collect();
    p.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
    for (k, (n, t)) in p.iter().take(12) {
        println!("    after {k:<28} {n:>4} copies  {t:>9.1} ms");
    }
    println!("  first 70 launches in order (op/backend src>dst bytes ms):");
    for r in trace.iter().take(70) {
        println!(
            "    {:<20} {:<7} {}>{} {:>9} B {:>8.3} ms",
            r.op,
            r.backend,
            r.src,
            r.dst,
            r.bytes,
            ms(r.dt)
        );
    }
}

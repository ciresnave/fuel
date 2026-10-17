// SPDX-License-Identifier: MIT OR Apache-2.0
//! M0 of the joint GPU milestone plan (Unpopped PR #39,
//! `docs/joint-gpu-milestone-plan.md`): "the yardstick (CPU only, no GPU
//! needed)". Fuel owns M0. This file is the whole of it:
//!
//! - A real-model CPU reference fixture (`fixtures/qwen3_0_6b_cpu_reference.json`),
//!   generated once by `regenerate_qwen3_cpu_reference_fixture` (run through
//!   `examples/qwen3_yardstick_tool.rs`, feature `dev-tools`; the code lives in
//!   `fuel_transformers::dev_tools::qwen3_yardstick`)
//!   below (needs network, to fetch the GGUF via `hf_hub`; cached after the
//!   first run) and checked into the repo so every other test here needs no
//!   network and no GPU.
//! - `compare_runs`, the model-level compare harness M1 (sm_89 through fuel)
//!   will reuse against a CUDA-backend run: logit tolerance per the
//!   already-decided dtype tier, plus greedy-token identity over the first
//!   `DECODE_LEN` tokens with at most one excused near-tie per prompt.
//! - Two required negative controls (`perturbed_logit_is_rejected`,
//!   `token_swap_is_rejected`) proving `compare_runs` is not vacuous: each
//!   mutates the checked-in fixture and asserts the comparison fails. These
//!   run in default `cargo test`, need no network, and are the actual
//!   regression protection this milestone exists to establish.
//! - The speed harness (`qwen3_cpu_speed_harness`, the same example's `speed` mode: it
//!   needs the real model) — prefill/decode tok/s with a warm-up pass,
//!   recording the device string. M1 reuses the same timing shape on CUDA.
//!
//! Model: Qwen3-0.6B, Q4_K_M GGUF (`unsloth/Qwen3-0.6B-GGUF`) — the smallest
//! Qwen3.x *dense* checkpoint, per the plan's Q1 decision ("the first model
//! is the smallest Qwen3 dense model that exercises the real op set").
//! Loading mirrors `fuel-examples/examples/quantized-qwen3/main.rs` exactly
//! (same GGUF-metadata-to-`Qwen3Config` mapping), minus the CLI.
//!
//! Tolerance table (PM-decided 2026-10-07, §5.1 of the plan — written down
//! before any GPU run, so the GPU cannot set its own bar):
//!
//! | tier                                  | bound                          |
//! |----------------------------------------|---------------------------------|
//! | f32 / f32                              | `max|Δ| ≤ 1e-3 · max|ref|`      |
//! | f16/bf16 storage, f32 accumulate        | `max|Δ| ≤ 2e-2 · max|ref|`      |
//! | quantized, vs a CPU reference using the SAME quantized weights | same as the row above |
//!
//! This fixture's model is itself quantized (Q4_K_M), so every comparison
//! against it runs under the quantized-tier bound.

use fuel_transformers::dev_tools::qwen3_yardstick::*;
use fuel_transformers::models::lazy_quantized_qwen3::QuantizedQwen3Model;
use fuel_transformers::models::lazy_qwen3::Qwen3Config;

/// Happy path, no network needed: the checked-in fixture must compare
/// equal to itself under the quantized tier — proves `compare_runs` doesn't
/// reject a run against itself (a prerequisite for the negative controls
/// below to mean anything).
#[test]
fn fixture_compares_equal_to_itself() {
    let fixture = load_fixture();
    for run in &fixture.runs {
        compare_runs(run, run, DtypeTier::F16OrQuantized)
            .unwrap_or_else(|e| panic!("fixture must compare equal to itself: {e}"));
    }
}

/// Required negative control #1: a perturbed logit must be REJECTED.
/// Flips the top logit at one step far enough to both change the argmax
/// (so the token-identity check also fires) and exceed the tolerance bound
/// outright — proving `compare_runs` is not vacuously permissive.
#[test]
fn perturbed_logit_is_rejected() {
    let fixture = load_fixture();
    let reference = &fixture.runs[0];
    let mut corrupted = reference.clone();
    let step = &mut corrupted.steps[DECODE_LEN / 2];
    // Push the top logit far below the second-best: changes the argmax
    // AND blows the tolerance bound (this is not a near-tie).
    let second_best = step.top_logits[1].1;
    step.top_logits[0].1 = second_best - 10.0;
    step.token = step.top_logits[1].0;

    let result = compare_runs(&corrupted, reference, DtypeTier::F16OrQuantized);
    assert!(
        result.is_err(),
        "a corrupted logit must be rejected by compare_runs, but it was accepted"
    );
}

/// Required negative control #2: a swapped greedy token must be REJECTED
/// (a corruption that keeps logits untouched but changes which token was
/// actually sampled — e.g. a sampling-path bug downstream of correct
/// logits). Picks a step with a clear top1/top2 margin so the near-tie
/// exception cannot mask it.
#[test]
fn token_swap_is_rejected() {
    let fixture = load_fixture();
    let reference = &fixture.runs[0];
    let mut corrupted = reference.clone();

    let step_idx = corrupted
        .steps
        .iter()
        .position(|s| s.top_logits.len() >= 2 && (s.top_logits[0].1 - s.top_logits[1].1) > 1.0)
        .expect("fixture must contain at least one step with a clear top1/top2 margin");
    corrupted.steps[step_idx].token = corrupted.steps[step_idx].top_logits[1].0;

    let result = compare_runs(&corrupted, reference, DtypeTier::F16OrQuantized);
    assert!(
        result.is_err(),
        "a swapped greedy token at a non-near-tie step must be rejected, but it was accepted"
    );
}

/// Required negative control #3 (Unpopped, 2026-10-07): a reference top-K
/// vocab id that is simply ABSENT from the candidate's reported top-K must
/// be REJECTED, not silently skipped. Replaces one non-top-1 reference id
/// with an unrelated id in the candidate's list — top-1 is untouched, so
/// the token-identity check alone would pass this — and relies entirely on
/// `CandidateLogits::TopK`'s `NEG_INFINITY`-on-miss behaviour to catch it.
/// This is the exact scenario the pre-fix `compare_runs` missed: a real
/// divergence that drops a vocab id out of the candidate's top-K without
/// changing the argmax.
#[test]
fn missing_reference_id_in_candidate_topk_is_rejected() {
    let fixture = load_fixture();
    let reference = &fixture.runs[0];
    let mut corrupted = reference.clone();

    let step_idx = 10;
    let step = &mut corrupted.steps[step_idx];
    assert!(
        step.top_logits.len() > 2,
        "fixture step must have more than 2 top-K entries for this control to be meaningful"
    );
    // Replace a non-top-1, non-top-2 entry's vocab id (never read by the
    // token-identity check) with one not present anywhere else in this
    // step's list — the reference's original id at this slot now has NO
    // representative in the candidate's top-K at all.
    let replaced_logit = step.top_logits[2].1;
    let unused_id = step.top_logits.iter().map(|&(id, _)| id).max().unwrap() + 1;
    step.top_logits[2] = (unused_id, replaced_logit);

    let result = compare_runs(&corrupted, reference, DtypeTier::F16OrQuantized);
    assert!(
        result.is_err(),
        "a reference top-K id entirely missing from the candidate's top-K must be rejected \
         (NEG_INFINITY on miss), but it was accepted"
    );
}

/// Positive control, paired with the negative control above (Unpopped,
/// 2026-10-07): a LIVE candidate (full logit row — the shape M1 actually
/// produces) whose ranking reorders near the top-K cutoff, by an amount
/// WITHIN tolerance, must stay GREEN. This is what the missing-id fix
/// above must NOT break: `run_kv_context_decode_and_compare`'s
/// `CandidateLogits::Full` looks up every reference id by its REAL value in the full row,
/// regardless of what rank that id happens to hold in the candidate, so a
/// harmless reorder can never be mistaken for a missing id.
#[test]
fn live_candidate_near_cutoff_reorder_stays_accepted() {
    // A tiny synthetic reference step: 4 top-K entries over an 8-entry
    // vocabulary, standing in for one real step. No model, no fixture —
    // this tests `compare_step`'s Full-vs-TopK distinction in isolation.
    let reference_step = StepRecord {
        token: 0,
        max_abs_logit: 10.0,
        top_logits: vec![(0, 10.0), (1, 9.0), (2, 1.02), (3, 1.0)],
    };
    let bound = DtypeTier::F16OrQuantized.relative_bound() * reference_step.max_abs_logit;
    assert!(
        1.02 - 1.0 < bound,
        "this control's whole point is a reorder WITHIN tolerance; widen the gap above if \
         the tolerance table changes and this assertion starts failing"
    );

    // The live candidate's FULL row: ids 2 and 3 have swapped RANK (3 is
    // now nominally "better" than 2) but both values are within `bound`
    // of their reference counterparts — a harmless reorder, not a real
    // divergence. Ids 4-7 are unrelated filler so the row has a real
    // vocab size larger than the reference's top-K.
    let candidate_full_logits = vec![10.0, 9.0, 1.0, 1.02, -5.0, -5.0, -5.0, -5.0];
    let mut excused = 0usize;
    let result = compare_step(
        0,
        0, // candidate's argmax token, unchanged
        &CandidateLogits::Full(&candidate_full_logits),
        &reference_step,
        DtypeTier::F16OrQuantized.relative_bound(),
        &mut excused,
    );
    assert!(
        result.is_ok(),
        "a within-tolerance reorder near the top-K cutoff must stay accepted on a Full \
         (live-run-shaped) candidate, but it was rejected: {result:?}"
    );
}

/// Result of running the persistent-KV-context decode path on some device
/// and comparing it, step by step, against the CPU reference fixture.
/// Diagnostic fields (`first_divergent_step`, per-prompt breakdown) exist
/// so a failure can be root-caused — prefill (step 0) wrong implicates
/// kernels/dequant; prefill right but decode drifting implicates the KV
/// cache, RoPE position offset, or session persistence — without having
/// to re-run anything.
struct KvContextRunResult {
    device_label: String,
    max_abs_logit_diff: f32,
    total_steps: usize,
    total_mismatched_tokens: usize,
    /// `(prompt_index, step_index)` of the first step whose `compare_step`
    /// result was `Err`, if any.
    first_divergent_step: Option<(usize, usize)>,
}

/// Runs M0's fixed prompt set through the persistent-KV-context decode
/// path (`forward_with_kv_context_persistent` — plain `forward()` is
/// hardcoded to `Device::cpu()` and cannot target an arbitrary device) on
/// `dev`, and compares every step against the CPU reference fixture inline
/// via `compare_step`/`CandidateLogits::Full`. Shared by `m1_cuda_matches_cpu_reference` and
/// `kv_context_cpu_matches_forward_cpu_reference` — the latter is the
/// cheap (no GPU, no `cuda` feature) diagnostic for isolating "is this a
/// CUDA-specific bug, or a bug in this API / this harness, full stop?"
/// (PM ruling, 2026-10-07: run the SAME harness on CPU before trusting a
/// CUDA divergence as a backend finding).
fn run_kv_context_decode_and_compare(
    model: &QuantizedQwen3Model,
    tokenizer: &tokenizers::Tokenizer,
    cfg: &Qwen3Config,
    fixture: &ReferenceFixture,
    dev: fuel_tensor::Device,
    device_label: &str,
) -> KvContextRunResult {
    use fuel_ir::DType;
    use fuel_tensor::{
        inference_context::DecodeSession, inference_context::InferenceContext,
        inference_context::KvCache,
    };

    let mut max_abs_logit_diff = 0f32;
    let mut total_steps = 0usize;
    let mut total_mismatched_tokens = 0usize;
    let mut first_divergent_step = None;

    for (prompt_idx, reference_run) in fixture.runs.iter().enumerate() {
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            reference_run.prompt
        );
        let encoding = tokenizer.encode(formatted, true).expect("encode");
        let prompt_tokens = encoding.get_ids().to_vec();

        let max_seq_len = prompt_tokens.len() + DECODE_LEN;
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

        let mut candidate_steps: Vec<(u32, Vec<f32>)> = Vec::with_capacity(DECODE_LEN);

        // Prefill: full prompt in one call. FOUND 2026-10-07 running M1 on
        // the real CUDA backend: unlike plain `forward()` (which returns a
        // flattened `(seq, vocab)` tensor needing a last-position slice),
        // `forward_with_kv_context_persistent` returns exactly `vocab_size`
        // elements — the next-token prediction only — REGARDLESS of how
        // many tokens were in the input slice. The precedent test this
        // file's doc comment cites
        // (`quantized_qwen3_windowed_decode_matches_quantized_forward` in
        // `lazy_quantized_qwen3.rs`) never exercised this: it called
        // `forward_with_kv_context_persistent` for prefill too, but never
        // asserted anything about that call's OWN return shape, only
        // about later decode steps. The original slicing code here
        // (`off = (prompt_tokens.len() - 1) * vocab_size`) assumed the
        // plain-`forward()` shape and panicked:
        // "range start index 3038720 out of range for slice of length
        // 151936" (3038720 = 20 × 151936, prompt_tokens.len() == 21).
        let logits = model
            .forward_with_kv_context_persistent(&prompt_tokens, &mut cache, &mut ctx, &mut session)
            .expect("prefill forward_with_kv_context_persistent");
        let vocab_size = cfg.vocab_size;
        assert_eq!(
            logits.len(),
            vocab_size,
            "forward_with_kv_context_persistent's prefill return must be exactly \
             vocab_size elements, not seq*vocab_size — if this assertion fires, the \
             API shape changed and this test's decode loop below needs revisiting too"
        );
        let mut logits = logits;
        let mut next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        candidate_steps.push((next_token, logits.clone()));

        // Decode: one token at a time, `forward_with_kv_context_persistent`
        // returns exactly `vocab_size` elements per call once `session`
        // holds a built graph — no slicing needed, same shape as prefill.
        for _ in 1..DECODE_LEN {
            logits = model
                .forward_with_kv_context_persistent(
                    &[next_token],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                )
                .expect("decode forward_with_kv_context_persistent");
            next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            candidate_steps.push((next_token, logits.clone()));
        }

        let mut excused = 0usize;
        for (step, (candidate, reference)) in candidate_steps
            .iter()
            .zip(reference_run.steps.iter())
            .enumerate()
        {
            for &(vocab_id, r_logit) in &reference.top_logits {
                let c_logit = candidate.1[vocab_id as usize];
                max_abs_logit_diff = max_abs_logit_diff.max((c_logit - r_logit).abs());
            }
            if candidate.0 != reference.token {
                total_mismatched_tokens += 1;
            }
            let step_result = compare_step(
                step,
                candidate.0,
                &CandidateLogits::Full(&candidate.1),
                reference,
                DtypeTier::F16OrQuantized.relative_bound(),
                &mut excused,
            );
            if step_result.is_err() && first_divergent_step.is_none() {
                first_divergent_step = Some((prompt_idx, step));
            }
            total_steps += 1;
        }

        println!(
            "device={device_label} prompt={:?} max_abs_logit_diff_so_far={max_abs_logit_diff} \
             mismatched_tokens_so_far={total_mismatched_tokens}/{total_steps} \
             first_divergent_step_so_far={first_divergent_step:?}",
            reference_run.prompt,
        );
    }

    KvContextRunResult {
        device_label: device_label.to_string(),
        max_abs_logit_diff,
        total_steps,
        total_mismatched_tokens,
        first_divergent_step,
    }
}

impl KvContextRunResult {
    fn assert_passes(&self) {
        assert!(
            self.first_divergent_step.is_none(),
            "{} run diverged from the CPU reference at (prompt, step)={:?}: \
             max_abs_logit_diff={} mismatched_tokens={}/{}",
            self.device_label,
            self.first_divergent_step,
            self.max_abs_logit_diff,
            self.total_mismatched_tokens,
            self.total_steps
        );
        println!(
            "PASS: device={} max_abs_logit_diff={} mismatched_tokens={}/{}",
            self.device_label,
            self.max_abs_logit_diff,
            self.total_mismatched_tokens,
            self.total_steps
        );
    }
}

/// Diagnostic (PM ruling, 2026-10-07): runs the EXACT SAME
/// `forward_with_kv_context_persistent` decode path `m1_cuda_matches_cpu_reference`
/// uses, but on `Device::cpu()` instead of CUDA. Isolates "is a CUDA
/// divergence a real backend bug, or a bug in this API / this harness
/// regardless of device?" — if this test ALSO fails, the fault is here or
/// in `forward_with_kv_context_persistent` itself, not CUDA-specific. If
/// it passes, a CUDA failure is a real CUDA-path finding. No `cuda`
/// feature or GPU needed — cheap to run before trusting any CUDA result.
#[test]
#[ignore = "needs network (hf_hub) to fetch the real Qwen3-0.6B GGUF on first run"]
fn kv_context_cpu_matches_forward_cpu_reference() {
    let fixture = load_fixture();
    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let result = run_kv_context_decode_and_compare(
        &model,
        &tokenizer,
        &cfg,
        &fixture,
        fuel_tensor::Device::cpu(),
        "cpu (via forward_with_kv_context_persistent)",
    );
    result.assert_passes();
}

/// M1 of the joint GPU milestone plan: "sm_89 end to end through fuel" —
/// runs M0's model on the RTX 4070 through fuel's CUDA backend, via the
/// persistent-KV-cache decode path, then compares against M0's checked-in
/// CPU reference fixture. `#[ignore]`d: needs network (first run) AND a
/// real CUDA device AND the `cuda` feature — take a CUDA build slot per
/// `scripts/cuda-build.ps1` to compile this, and run it through
/// `scripts/gpu-run.ps1` (exclusive device access), never bare `cargo test`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs network, a real CUDA device, and --features cuda; run via scripts/gpu-run.ps1"]
fn m1_cuda_matches_cpu_reference() {
    let fixture = load_fixture();
    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let dev =
        fuel_tensor::cuda_backend::new_device(0).expect("fuel_tensor::cuda_backend::new_device(0)");
    let result =
        run_kv_context_decode_and_compare(&model, &tokenizer, &cfg, &fixture, dev, "cuda:0");
    result.assert_passes();
}

/// Timed CUDA run for M1's reporting requirement (PM, 2026-10-07): prefill
/// and decode tok/s, measured separately, warm (a discarded warm-up run
/// pays for first-touch CUDA context / allocator init so the timed run
/// isn't charged for it). Mirrors `qwen3_cpu_speed_harness`'s shape but
/// over `forward_with_kv_context_persistent` on `cuda:0` (plain `forward()`
/// is hardcoded to `Device::cpu()` and cannot reach CUDA at all). Not a
/// correctness assertion — a recorded number, like its CPU counterpart.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs network, a real CUDA device, and --features cuda; run via scripts/gpu-run.ps1"]
fn m1_cuda_speed_harness() {
    use fuel_ir::DType;
    use fuel_tensor::{
        inference_context::DecodeSession, inference_context::InferenceContext,
        inference_context::KvCache,
    };

    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let dev =
        fuel_tensor::cuda_backend::new_device(0).expect("fuel_tensor::cuda_backend::new_device(0)");
    let prompt = PROMPTS[0];
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer.encode(formatted, true).expect("encode");
    let prompt_tokens = encoding.get_ids().to_vec();
    let max_seq_len = prompt_tokens.len() + DECODE_LEN;

    // Reports THREE durations rather than two (Sourcery + PM, 2026-10-07):
    // `forward_with_kv_context_persistent` builds/optimizes the persistent
    // decode graph on the FIRST `seq == 1` call per session
    // (`build_and_realize_first_decode_token`), then skips optimize on
    // every later call. Each `run_once` starts a FRESH session, so a
    // separate throwaway "warm-up" call does NOT warm this cost away — it
    // only warms CUDA context / allocator / first-touch costs, which is a
    // DIFFERENT one-time cost from the per-session graph build. Folding
    // the graph-build latency into "decode tok/s" (the original form of
    // this harness) silently mixed a one-time setup cost into a
    // steady-state throughput number. Split here: `first_decode_dt` is
    // that one call alone; `steady_decode_dt` is every call after it.
    let run_once = || -> (std::time::Duration, std::time::Duration, std::time::Duration) {
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

        let prefill_start = std::time::Instant::now();
        let logits = model
            .forward_with_kv_context_persistent(&prompt_tokens, &mut cache, &mut ctx, &mut session)
            .expect("prefill forward_with_kv_context_persistent");
        let prefill_dt = prefill_start.elapsed();
        let mut next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        let first_decode_start = std::time::Instant::now();
        let logits = model
            .forward_with_kv_context_persistent(&[next_token], &mut cache, &mut ctx, &mut session)
            .expect("first decode forward_with_kv_context_persistent");
        let first_decode_dt = first_decode_start.elapsed();
        next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        let steady_decode_start = std::time::Instant::now();
        for _ in 2..DECODE_LEN {
            let logits = model
                .forward_with_kv_context_persistent(
                    &[next_token],
                    &mut cache,
                    &mut ctx,
                    &mut session,
                )
                .expect("decode forward_with_kv_context_persistent");
            next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
        }
        let steady_decode_dt = steady_decode_start.elapsed();
        (prefill_dt, first_decode_dt, steady_decode_dt)
    };

    // Warm-up: discarded, so the timed run isn't paying for CUDA context /
    // allocator / first-touch page-fault costs. Does NOT warm the
    // persistent-decode graph build (see above) -- that is measured
    // separately below, every run, because it is per-session.
    let _ = run_once();

    let (prefill_dt, first_decode_dt, steady_decode_dt) = run_once();
    let steady_decode_steps = (DECODE_LEN - 2) as f64;
    println!(
        "device=cuda:0 model={MODEL_REPO}/{MODEL_FILE} \
         prefill: {} tokens in {:.3}s ({:.2} tok/s) \
         first_decode_call (session build + 1 token): {:.3}s \
         steady_state_decode: {steady_decode_steps} tokens in {:.3}s ({:.2} tok/s)",
        prompt_tokens.len(),
        prefill_dt.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_dt.as_secs_f64(),
        first_decode_dt.as_secs_f64(),
        steady_decode_dt.as_secs_f64(),
        steady_decode_steps / steady_decode_dt.as_secs_f64(),
    );
}

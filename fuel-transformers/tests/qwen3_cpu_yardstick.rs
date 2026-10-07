// SPDX-License-Identifier: MIT OR Apache-2.0
//! M0 of the joint GPU milestone plan (Unpopped PR #39,
//! `docs/joint-gpu-milestone-plan.md`): "the yardstick (CPU only, no GPU
//! needed)". Fuel owns M0. This file is the whole of it:
//!
//! - A real-model CPU reference fixture (`fixtures/qwen3_0_6b_cpu_reference.json`),
//!   generated once by the `#[ignore]`d `regenerate_qwen3_cpu_reference_fixture`
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
//! - The speed harness (`qwen3_cpu_speed_harness`, also `#[ignore]`d: it
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

use std::collections::HashMap;
use std::path::PathBuf;

use fuel_core::quantized::gguf_file::Value as GgufValue;
use fuel_core::quantized::gguf_mmap::MmapedContent;
use fuel_transformers::models::lazy_quantized_qwen3::QuantizedQwen3Model;
use fuel_transformers::models::lazy_qwen3::Qwen3Config;
use serde::{Deserialize, Serialize};

/// First `DECODE_LEN` greedily generated tokens are compared — the plan's
/// "greedy-token identity" window.
const DECODE_LEN: usize = 32;
/// Per-step logit record width. Storing the full ~151936-entry vocab per
/// step per prompt would make the checked-in fixture tens of megabytes;
/// the top-`TOP_K` entries (by reference logit) are enough to check the
/// tolerance bound on every candidate that could plausibly be sampled,
/// and to compute the top1/top2 margin the near-tie exception needs.
const TOP_K: usize = 16;
/// Fixed prompt set. Two short, distinct prompts: enough to catch a
/// position- or content-dependent regression without a large fixture.
const PROMPTS: &[&str] = &[
    "Write a Rust function to calculate the factorial of a given number.",
    "The capital of France is",
];

const MODEL_REPO: &str = "unsloth/Qwen3-0.6B-GGUF";
const MODEL_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";
const MODEL_REVISION: &str = "main";
const TOKENIZER_REPO: &str = "Qwen/Qwen3-0.6B";

const FIXTURE_PATH: &str = "tests/fixtures/qwen3_0_6b_cpu_reference.json";

/// The dtype tier a run was produced under — selects the tolerance bound.
/// Mirrors the plan's §5.1 table exactly; see the module doc above.
#[derive(Clone, Copy, Debug)]
enum DtypeTier {
    #[allow(dead_code)] // exercised once CUDA f32 compute lands in M1
    F32F32,
    /// Also covers "quantized vs a same-quantized-weights CPU reference" —
    /// the plan gives it the identical bound and rationale (rounding of
    /// intermediate activations), so it is not a separate variant here.
    F16OrQuantized,
}

impl DtypeTier {
    fn relative_bound(self) -> f32 {
        match self {
            DtypeTier::F32F32 => 1e-3,
            DtypeTier::F16OrQuantized => 2e-2,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StepRecord {
    token: u32,
    /// `max|logit|` over the FULL vocab at this step — the tolerance
    /// bound's denominator (`max|logit_ref|`), computed before truncating
    /// to the top-`TOP_K` entries below.
    max_abs_logit: f32,
    /// Top-`TOP_K` (vocab_id, logit) pairs, sorted descending by logit.
    top_logits: Vec<(u32, f32)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PromptRun {
    prompt: String,
    steps: Vec<StepRecord>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct ReferenceFixture {
    model_repo: String,
    model_file: String,
    decode_len: usize,
    top_k: usize,
    /// Spread observed between two same-process CPU reruns of the SAME
    /// prompt, as `max|Δ|` over every logit both runs stored. 0.0 if the
    /// two reruns were bit-identical. Calibration note from the plan's
    /// §5.1: "the bound must be at least 4x the spread that measures, or
    /// it is too tight to be stable" — recorded here so a tightened bound
    /// later can be checked against it without rerunning the model.
    rerun_spread_max_abs_delta: f32,
    runs: Vec<PromptRun>,
}

/// Build a `Qwen3Config` from GGUF metadata. Exact copy of
/// `quantized-qwen3/main.rs`'s `qwen3_cfg_from_gguf`, with the CLI stripped —
/// kept in sync by hand since the example is a binary, not a library fn.
fn qwen3_cfg_from_gguf(mc: &MmapedContent) -> anyhow::Result<Qwen3Config> {
    let md = mc.metadata();
    let get = |k: &str| -> anyhow::Result<&GgufValue> {
        md.get(k)
            .ok_or_else(|| anyhow::anyhow!("gguf metadata: missing key {k:?}"))
    };
    let num_attention_heads = get("qwen3.attention.head_count")?.to_u32()? as usize;
    let num_key_value_heads = get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
    let head_dim = get("qwen3.attention.key_length")?.to_u32()? as usize;
    let num_hidden_layers = get("qwen3.block_count")?.to_u32()? as usize;
    let hidden_size = get("qwen3.embedding_length")?.to_u32()? as usize;
    let intermediate_size = get("qwen3.feed_forward_length")?.to_u32()? as usize;
    let max_position_embeddings = get("qwen3.context_length")?.to_u32()? as usize;
    let rms_norm_eps = get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_theta = get("qwen3.rope.freq_base")?.to_f32()? as f64;
    let vocab_size = match md.get("qwen3.vocab_size") {
        Some(v) => v.to_u32()? as usize,
        None => {
            let info = mc
                .content()
                .tensor_infos
                .get("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("gguf: missing token_embd.weight"))?;
            let dims = info.shape.dims();
            if dims.is_empty() {
                anyhow::bail!("gguf: token_embd.weight has empty shape");
            }
            dims[0]
        }
    };
    let tie_word_embeddings = !mc.content().tensor_infos.contains_key("output.weight");
    Ok(Qwen3Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
        sliding_window: None,
        max_window_layers: 0,
        use_sliding_window: false,
        rope_theta,
        rms_norm_eps,
        attention_bias: false,
        tie_word_embeddings,
    })
}

/// Fetches (and caches, via `hf_hub`) the GGUF weights and tokenizer, and
/// builds the model. Needs network on an empty cache — only called from
/// `#[ignore]`d tests.
fn load_model() -> anyhow::Result<(QuantizedQwen3Model, tokenizers::Tokenizer, Qwen3Config)> {
    let api = hf_hub::api::sync::Api::new()?;
    let model_path: PathBuf = api
        .repo(hf_hub::Repo::with_revision(
            MODEL_REPO.to_string(),
            hf_hub::RepoType::Model,
            MODEL_REVISION.to_string(),
        ))
        .get(MODEL_FILE)?;
    let tokenizer_path = api
        .model(TOKENIZER_REPO.to_string())
        .get("tokenizer.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    let mmaped = MmapedContent::from_path(&model_path)?;
    let cfg = qwen3_cfg_from_gguf(&mmaped)?;
    drop(mmaped);
    let model = QuantizedQwen3Model::from_gguf(&model_path, &cfg)
        .map_err(|e| anyhow::anyhow!("from_gguf: {e}"))?;
    Ok((model, tokenizer, cfg))
}

/// Runs greedy decode for `DECODE_LEN` steps, recording the top-`TOP_K`
/// logits and `max_abs_logit` at every step. Pure argmax — no sampling, no
/// seed, no temperature — so the only source of nondeterminism is the CPU
/// backend's own reduction order (what `rerun_spread_max_abs_delta` checks).
fn run_greedy(
    model: &QuantizedQwen3Model,
    tokenizer: &tokenizers::Tokenizer,
    cfg: &Qwen3Config,
    prompt: &str,
) -> anyhow::Result<PromptRun> {
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer
        .encode(formatted, true)
        .map_err(anyhow::Error::msg)?;
    let prompt_tokens = encoding.get_ids();
    let vocab_size = cfg.vocab_size;

    let slice_last = |flat: Vec<f32>, seq: usize| -> Vec<f32> {
        let off = (seq - 1) * vocab_size;
        flat[off..off + vocab_size].to_vec()
    };

    let record_step = |logits: &[f32]| -> StepRecord {
        let token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits must not be NaN"))
            .map(|(idx, _)| idx as u32)
            .expect("logits must be non-empty");
        let max_abs_logit = logits.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("logits must not be NaN"));
        indexed.truncate(TOP_K);
        StepRecord {
            token,
            max_abs_logit,
            top_logits: indexed,
        }
    };

    let logits_flat = model
        .forward(prompt_tokens, 0)
        .map_err(|e| anyhow::anyhow!("forward: {e}"))?
        .realize_f32();
    let mut logits = slice_last(logits_flat, prompt_tokens.len());
    let mut steps = Vec::with_capacity(DECODE_LEN);
    let first = record_step(&logits);
    let mut next_token = first.token;
    steps.push(first);

    for index in 1..DECODE_LEN {
        let logits_flat = model
            .forward(&[next_token], prompt_tokens.len() + index - 1)
            .map_err(|e| anyhow::anyhow!("forward: {e}"))?
            .realize_f32();
        logits = slice_last(logits_flat, 1);
        let step = record_step(&logits);
        next_token = step.token;
        steps.push(step);
    }

    Ok(PromptRun {
        prompt: prompt.to_string(),
        steps,
    })
}

/// Compares a candidate run (e.g. a future M1 CUDA-backend run) against a
/// reference `PromptRun` under the given tier's tolerance. Mirrors
/// `fuel-model-llama/tests/paged_decode_parity.rs`'s `assert_close` shape,
/// generalized for the plan's per-dtype relative bound plus the
/// greedy-token-identity-with-one-excused-near-tie rule.
///
/// Only vocab ids present in the REFERENCE's top-`TOP_K` are checked for
/// logit closeness — a candidate whose own top-K omits one of those ids
/// cannot be tolerance-checked on it from this fixture, which is a
/// fixture-size tradeoff (see `TOP_K`'s doc comment), not a pass given by
/// default: such an omission almost always co-occurs with a token-identity
/// mismatch, which IS checked unconditionally below.
fn compare_runs(
    candidate: &PromptRun,
    reference: &PromptRun,
    tier: DtypeTier,
) -> Result<(), String> {
    if candidate.steps.len() != reference.steps.len() {
        return Err(format!(
            "step count mismatch: candidate={} reference={}",
            candidate.steps.len(),
            reference.steps.len()
        ));
    }
    let rel_bound = tier.relative_bound();
    let mut excused_near_ties = 0usize;

    for (i, (c, r)) in candidate
        .steps
        .iter()
        .zip(reference.steps.iter())
        .enumerate()
    {
        let bound = rel_bound * r.max_abs_logit.max(f32::MIN_POSITIVE);

        let c_by_id: HashMap<u32, f32> = c.top_logits.iter().copied().collect();
        for &(vocab_id, r_logit) in &r.top_logits {
            if let Some(&c_logit) = c_by_id.get(&vocab_id) {
                let diff = (c_logit - r_logit).abs();
                if diff > bound {
                    return Err(format!(
                        "step {i}: vocab {vocab_id} logit diff {diff} exceeds bound {bound} \
                         (candidate={c_logit} reference={r_logit})"
                    ));
                }
            }
        }

        if c.token != r.token {
            let margin = if r.top_logits.len() >= 2 {
                r.top_logits[0].1 - r.top_logits[1].1
            } else {
                f32::INFINITY
            };
            if margin < bound && excused_near_ties == 0 {
                excused_near_ties += 1;
                continue;
            }
            return Err(format!(
                "step {i}: greedy token mismatch: candidate={} reference={} \
                 (top1/top2 margin={margin}, bound={bound}, already excused={excused_near_ties})",
                c.token, r.token
            ));
        }
    }
    Ok(())
}

fn load_fixture() -> ReferenceFixture {
    let bytes = std::fs::read(FIXTURE_PATH).unwrap_or_else(|e| {
        panic!(
            "{FIXTURE_PATH} is missing or unreadable ({e}) — run \
             `cargo test -p fuel-transformers --test qwen3_cpu_yardstick -- --ignored \
             regenerate_qwen3_cpu_reference_fixture` once (needs network) to produce it"
        )
    });
    serde_json::from_slice(&bytes).expect("fixture JSON must deserialize as ReferenceFixture")
}

/// Generates `fixtures/qwen3_0_6b_cpu_reference.json`. Manual-run only —
/// needs network on an empty `hf_hub` cache. Also measures same-process
/// rerun determinism (the plan's §5.1 calibration step: the tolerance
/// bound must be ≥4x this spread) and panics before writing the fixture if
/// that check fails, so a non-deterministic build can't silently produce a
/// fixture the tolerance bound doesn't actually cover.
#[test]
#[ignore = "needs network (hf_hub) to fetch the real Qwen3-0.6B GGUF on first run"]
fn regenerate_qwen3_cpu_reference_fixture() {
    let (model, tokenizer, cfg) = load_model().expect("load_model");

    let runs: Vec<PromptRun> = PROMPTS
        .iter()
        .map(|p| run_greedy(&model, &tokenizer, &cfg, p).expect("run_greedy"))
        .collect();

    // Calibration: rerun the first prompt in the same process and measure
    // the spread against the just-recorded run.
    let rerun = run_greedy(&model, &tokenizer, &cfg, PROMPTS[0]).expect("run_greedy rerun");
    let mut spread = 0f32;
    for (a, b) in runs[0].steps.iter().zip(rerun.steps.iter()) {
        let a_by_id: HashMap<u32, f32> = a.top_logits.iter().copied().collect();
        for &(vid, b_logit) in &b.top_logits {
            if let Some(&a_logit) = a_by_id.get(&vid) {
                spread = spread.max((a_logit - b_logit).abs());
            }
        }
    }
    let bound = DtypeTier::F16OrQuantized.relative_bound() * runs[0].steps[0].max_abs_logit;
    assert!(
        spread * 4.0 <= bound,
        "same-process rerun spread {spread} is not ≤ 1/4 of the tolerance bound {bound} — \
         the tolerance table in docs/joint-gpu-milestone-plan.md §5.1 needs re-measuring \
         before this fixture can be trusted, per the plan's own calibration rule"
    );

    let fixture = ReferenceFixture {
        model_repo: MODEL_REPO.to_string(),
        model_file: MODEL_FILE.to_string(),
        decode_len: DECODE_LEN,
        top_k: TOP_K,
        rerun_spread_max_abs_delta: spread,
        runs,
    };
    let json = serde_json::to_string_pretty(&fixture).expect("serialize fixture");
    std::fs::write(FIXTURE_PATH, json).expect("write fixture");
    println!(
        "wrote {FIXTURE_PATH}: {} prompts x {DECODE_LEN} steps, rerun spread {spread}",
        fixture.runs.len()
    );
}

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

/// Speed harness: prefill and decode tok/s, with a warm-up pass before
/// timing. Manual-run only (needs the real model). M1 reuses this exact
/// shape against the CUDA backend; the device/driver/toolkit strings this
/// prints are the per-card record the plan's M0/M1 rows ask for.
#[test]
#[ignore = "needs network (hf_hub) and real CPU inference; not a CI assertion, a recorded number"]
fn qwen3_cpu_speed_harness() {
    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let prompt = PROMPTS[0];

    // Warm-up: one full run, discarded, so the timed run isn't paying for
    // first-touch page faults / allocator warm-up / lazy init.
    let _ = run_greedy(&model, &tokenizer, &cfg, prompt).expect("warm-up run_greedy");

    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer.encode(formatted, true).expect("encode");
    let prompt_tokens = encoding.get_ids().to_vec();

    let prefill_start = std::time::Instant::now();
    let logits_flat = model
        .forward(&prompt_tokens, 0)
        .expect("forward")
        .realize_f32();
    let prefill_dt = prefill_start.elapsed();
    let vocab_size = cfg.vocab_size;
    let off = (prompt_tokens.len() - 1) * vocab_size;
    let mut logits = logits_flat[off..off + vocab_size].to_vec();
    let mut next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();

    let decode_start = std::time::Instant::now();
    for index in 1..DECODE_LEN {
        let logits_flat = model
            .forward(&[next_token], prompt_tokens.len() + index - 1)
            .expect("forward")
            .realize_f32();
        logits = logits_flat;
        next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
    }
    let decode_dt = decode_start.elapsed();
    let decode_steps = (DECODE_LEN - 1) as f64;

    println!(
        "device=cpu model={MODEL_REPO}/{MODEL_FILE} \
         prefill: {} tokens in {:.3}s ({:.2} tok/s) \
         decode: {decode_steps} tokens in {:.3}s ({:.2} tok/s)",
        prompt_tokens.len(),
        prefill_dt.as_secs_f64(),
        prompt_tokens.len() as f64 / prefill_dt.as_secs_f64(),
        decode_dt.as_secs_f64(),
        decode_steps / decode_dt.as_secs_f64(),
    );
}

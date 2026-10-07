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
    /// Spread observed between two same-process CPU runs of the SAME
    /// prompt at DIFFERENT thread counts (`RAYON_NUM_THREADS=1` vs the
    /// default), as `max|Δ|` over every logit both runs stored — a real
    /// reduction-order probe, not a repeatability one (see
    /// `regenerate_qwen3_cpu_reference_fixture`'s doc for why same-config
    /// reruns don't measure this). Calibration note from the plan's §5.1:
    /// "the bound must be at least 4x the spread that measures, or it is
    /// too tight to be stable" — recorded here so a tightened bound later
    /// can be checked against it without rerunning the model.
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

/// A candidate step's logits, in one of two shapes. Unifies the
/// stored-fixture self-tests (which only ever have a top-K view of BOTH
/// sides) with a live run's comparison (M1 onward: the candidate is a
/// real forward pass and has every vocab id, not just its own top-K).
///
/// Lookup-by-id is the operation both call sites need, and the two shapes
/// give it different answers for an id OUTSIDE the stored top-K: `Full`
/// returns the real value (so a harmless rank reorder near the cutoff
/// reads correctly and does not false-positive); `TopK` returns
/// `NEG_INFINITY` (so a vocab id the candidate's reported top-K omits
/// entirely reads as a real, large divergence — FAILS the bound check —
/// rather than silently passing because "we have no data on it"). Treating
/// "no data" as "no problem" is exactly the bug this enum exists to close:
/// found 2026-10-07 by Unpopped, reading this function at `fuel@de4d370`
/// (`qwen3_cpu_yardstick.rs:313-314`) — "a real deviation can pass the
/// compare unchecked... make a missing id a failure, never a skip."
enum CandidateLogits<'a> {
    /// A live run's full per-step logit vector (index == vocab id).
    Full(&'a [f32]),
    /// A stored/synthetic top-K view (what `fixture_compares_equal_to_itself`
    /// and the negative controls below work with — there is no "full row"
    /// for historical fixture data, by design; see `TOP_K`'s doc comment).
    TopK(&'a [(u32, f32)]),
}

impl CandidateLogits<'_> {
    fn get(&self, vocab_id: u32) -> f32 {
        match self {
            Self::Full(row) => row
                .get(vocab_id as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY),
            Self::TopK(pairs) => pairs
                .iter()
                .find(|&&(id, _)| id == vocab_id)
                .map(|&(_, logit)| logit)
                .unwrap_or(f32::NEG_INFINITY),
        }
    }
}

/// Compares one candidate step against one reference step under the given
/// tier's tolerance, plus the greedy-token-identity-with-one-excused-near-tie
/// rule. Every reference top-K id is looked up in the candidate's `logits`
/// — for a `Full` candidate this can never miss (every id resolves to its
/// real value); for a `TopK` candidate (self-tests only) a genuinely absent
/// id resolves to `NEG_INFINITY`, which always exceeds the bound and so
/// always FAILS rather than being silently skipped.
fn compare_step(
    i: usize,
    candidate_token: u32,
    candidate_logits: &CandidateLogits<'_>,
    reference: &StepRecord,
    rel_bound: f32,
    excused_near_ties: &mut usize,
) -> Result<(), String> {
    let bound = rel_bound * reference.max_abs_logit.max(f32::MIN_POSITIVE);

    for &(vocab_id, r_logit) in &reference.top_logits {
        let c_logit = candidate_logits.get(vocab_id);
        let diff = (c_logit - r_logit).abs();
        if diff > bound {
            return Err(format!(
                "step {i}: vocab {vocab_id} logit diff {diff} exceeds bound {bound} \
                 (candidate={c_logit} reference={r_logit})"
            ));
        }
    }

    if candidate_token != reference.token {
        let margin = if reference.top_logits.len() >= 2 {
            reference.top_logits[0].1 - reference.top_logits[1].1
        } else {
            f32::INFINITY
        };
        if margin < bound && *excused_near_ties == 0 {
            *excused_near_ties += 1;
            return Ok(());
        }
        return Err(format!(
            "step {i}: greedy token mismatch: candidate={} reference={} \
             (top1/top2 margin={margin}, bound={bound}, already excused={excused_near_ties})",
            candidate_token, reference.token
        ));
    }
    Ok(())
}

/// Compares a stored/synthetic candidate `PromptRun` (top-K only) against a
/// reference `PromptRun`. Used by the self-consistency test and the
/// negative controls below, all of which work from the checked-in fixture
/// or a deliberately mutated copy of it — never a live run (see
/// `compare_live_run` for that). Mirrors
/// `fuel-model-llama/tests/paged_decode_parity.rs`'s `assert_close` shape.
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
        compare_step(
            i,
            c.token,
            &CandidateLogits::TopK(&c.top_logits),
            r,
            rel_bound,
            &mut excused_near_ties,
        )?;
    }
    Ok(())
}

/// Compares a LIVE candidate run — one full per-step logit vector plus its
/// argmax token, e.g. from a real `model.forward(...)` call on a CUDA
/// backend (M1 onward) — against a stored reference `PromptRun`. Unlike
/// `compare_runs`, there is no "missing id" case here: every reference
/// top-K id resolves to its REAL value in the candidate's full row, so a
/// harmless near-cutoff rank reorder cannot false-positive, and a real
/// divergence that moves a value far from the reference cannot hide behind
/// truncation either way.
///
/// Only called from `m1_cuda_matches_cpu_reference` below, which is
/// `#[cfg(feature = "cuda")]` — gated the same way so a default (no-cuda)
/// build has no dead-code warning for a function whose only consumer is
/// conditionally compiled.
#[cfg(feature = "cuda")]
fn compare_live_run(
    candidate_steps: &[(u32, Vec<f32>)],
    reference: &PromptRun,
    tier: DtypeTier,
) -> Result<(), String> {
    if candidate_steps.len() != reference.steps.len() {
        return Err(format!(
            "step count mismatch: candidate={} reference={}",
            candidate_steps.len(),
            reference.steps.len()
        ));
    }
    let rel_bound = tier.relative_bound();
    let mut excused_near_ties = 0usize;
    for (i, ((token, logits), r)) in candidate_steps
        .iter()
        .zip(reference.steps.iter())
        .enumerate()
    {
        compare_step(
            i,
            *token,
            &CandidateLogits::Full(logits),
            r,
            rel_bound,
            &mut excused_near_ties,
        )?;
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
/// needs network on an empty `hf_hub` cache. Also measures the plan's
/// §5.1 calibration step — "run the CPU reference twice with a different
/// reduction order (or thread count)" — and panics before writing the
/// fixture if the bound isn't ≥4x that spread, so a build whose real
/// nondeterminism the tolerance table doesn't actually cover can't
/// silently produce a fixture.
///
/// FIXED 2026-10-07 (found by Unpopped, reading the first version of this
/// function): the original rerun used the SAME `RAYON_NUM_THREADS` as the
/// first run (both unset, both defaulting to `num_cpus::get()`), so it
/// measured same-process repeatability, not reduction-order sensitivity —
/// spread=0.0 passed `bound ≥ 4x spread` for ANY bound, including a far
/// too tight one, and would have kept passing even if the real sensitivity
/// were large. `fuel-cpu-backend::ops::get_num_threads()` reads
/// `RAYON_NUM_THREADS` fresh on every call (no caching) and feeds it
/// straight into `gemm::Parallelism::Rayon(n)` as a per-call matmul
/// parameter, so setting it to `"1"` for one run and unsetting it for the
/// other genuinely forces two different reduction orders through the same
/// process, the same weights, the same prompt.
#[test]
#[ignore = "needs network (hf_hub) to fetch the real Qwen3-0.6B GGUF on first run"]
fn regenerate_qwen3_cpu_reference_fixture() {
    let (model, tokenizer, cfg) = load_model().expect("load_model");

    // SAFETY (env mutation in a test): this binary runs this one test
    // under `--ignored`, never alongside any other test that reads
    // RAYON_NUM_THREADS, so there is no cross-test race.
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", "1");
    }
    let single_threaded =
        run_greedy(&model, &tokenizer, &cfg, PROMPTS[0]).expect("run_greedy (RAYON_NUM_THREADS=1)");
    unsafe {
        std::env::remove_var("RAYON_NUM_THREADS");
    }
    let runs: Vec<PromptRun> = PROMPTS
        .iter()
        .map(|p| run_greedy(&model, &tokenizer, &cfg, p).expect("run_greedy (default threads)"))
        .collect();

    // Calibration: single_threaded vs runs[0], same prompt, different
    // thread count — a real reduction-order probe, not a repeatability one.
    let mut spread = 0f32;
    for (a, b) in runs[0].steps.iter().zip(single_threaded.steps.iter()) {
        for &(vid, b_logit) in &b.top_logits {
            let a_logit = CandidateLogits::TopK(&a.top_logits).get(vid);
            if a_logit.is_finite() {
                spread = spread.max((a_logit - b_logit).abs());
            }
        }
    }
    let bound = DtypeTier::F16OrQuantized.relative_bound() * runs[0].steps[0].max_abs_logit;
    assert!(
        spread * 4.0 <= bound,
        "reduction-order spread (RAYON_NUM_THREADS=1 vs default) {spread} is not ≤ 1/4 of \
         the tolerance bound {bound} — the tolerance table in \
         docs/joint-gpu-milestone-plan.md §5.1 needs re-measuring before this fixture can \
         be trusted, per the plan's own calibration rule"
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
/// above must NOT break: `compare_live_run`'s `CandidateLogits::Full`
/// looks up every reference id by its REAL value in the full row,
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

/// M1 of the joint GPU milestone plan: "sm_89 end to end through fuel" —
/// runs M0's model on the RTX 4070 through fuel's CUDA backend, via the
/// persistent-KV-cache decode path (`forward_with_kv_context_persistent`;
/// plain `forward()` is hardcoded to `Device::cpu()` in
/// `run_backbone`/`run_backbone_embeds` and cannot reach CUDA at all), then
/// runs it through `compare_live_run` against M0's checked-in CPU
/// reference fixture. `#[ignore]`d: needs network (first run) AND a real
/// CUDA device AND the `cuda` feature — take a CUDA build slot per
/// `scripts/cuda-build.ps1` to compile this, and run it through
/// `scripts/gpu-run.ps1` (exclusive device access), never bare `cargo test`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs network, a real CUDA device, and --features cuda; run via scripts/gpu-run.ps1"]
fn m1_cuda_matches_cpu_reference() {
    use fuel_core::Device;
    use fuel_core::inference_context::{DecodeSession, InferenceContext, KvCache};
    use fuel_ir::DType;

    let fixture = load_fixture();
    let (model, tokenizer, cfg) = load_model().expect("load_model");
    let dev =
        fuel_core::cuda_backend::new_device(0).expect("fuel_core::cuda_backend::new_device(0)");

    let mut max_abs_logit_diff = 0f32;
    let mut total_steps = 0usize;
    let mut total_mismatched_tokens = 0usize;

    for reference_run in &fixture.runs {
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

        // Prefill: full prompt in one call, slice the LAST position's
        // vocab_size-wide row out of the flattened (seq, vocab) result.
        let prefill_logits_flat = model
            .forward_with_kv_context_persistent(&prompt_tokens, &mut cache, &mut ctx, &mut session)
            .expect("prefill forward_with_kv_context_persistent");
        let vocab_size = cfg.vocab_size;
        let off = (prompt_tokens.len() - 1) * vocab_size;
        let mut logits = prefill_logits_flat[off..off + vocab_size].to_vec();
        let mut next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        candidate_steps.push((next_token, logits));

        // Decode: one token at a time, `forward_with_kv_context_persistent`
        // returns exactly `vocab_size` elements per call once `session`
        // holds a built graph — no slicing needed, unlike prefill.
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
            total_steps += 1;
            let _ = step;
        }

        let result = compare_live_run(&candidate_steps, reference_run, DtypeTier::F16OrQuantized);
        println!(
            "prompt={:?} max_abs_logit_diff_so_far={max_abs_logit_diff} \
             mismatched_tokens_so_far={total_mismatched_tokens}/{total_steps} compare_result={result:?}",
            reference_run.prompt,
        );
        result.unwrap_or_else(|e| {
            panic!(
                "M1: CUDA (sm_89) run diverged from the CPU reference for prompt {:?}: {e}",
                reference_run.prompt
            )
        });
    }

    println!(
        "M1 PASS: device=cuda:0 model={MODEL_REPO}/{MODEL_FILE} \
         max_abs_logit_diff={max_abs_logit_diff} \
         mismatched_tokens={total_mismatched_tokens}/{total_steps}"
    );
}

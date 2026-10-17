// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Qwen3-0.6B CPU yardstick's shared machinery (M0 of the joint GPU
//! milestone plan, Unpopped PR #39): model loading, greedy decode, the reference
//! fixture types, the compare harness, and the two manual tools (fixture
//! regeneration, speed harness). It lives here, behind the `dev-tools` feature,
//! so the tools are real programs (`examples/qwen3_yardstick_tool.rs`) instead of
//! `#[ignore]`d tests, and `tests/qwen3_cpu_yardstick.rs` imports the same items.
//! `#[doc(hidden)]`: not API.

use std::path::PathBuf;

use crate::models::lazy_quantized_qwen3::QuantizedQwen3Model;
use crate::models::lazy_qwen3::Qwen3Config;
use fuel_loaders::quantized::gguf_file::Value as GgufValue;
use fuel_loaders::quantized::gguf_mmap::MmapedContent;
use serde::{Deserialize, Serialize};

/// First `DECODE_LEN` greedily generated tokens are compared — the plan's
/// "greedy-token identity" window.
pub const DECODE_LEN: usize = 32;
/// Per-step logit record width. Storing the full ~151936-entry vocab per
/// step per prompt would make the checked-in fixture tens of megabytes;
/// the top-`TOP_K` entries (by reference logit) are enough to check the
/// tolerance bound on every candidate that could plausibly be sampled,
/// and to compute the top1/top2 margin the near-tie exception needs.
pub const TOP_K: usize = 16;
/// Fixed prompt set. Two short, distinct prompts: enough to catch a
/// position- or content-dependent regression without a large fixture.
pub const PROMPTS: &[&str] = &[
    "Write a Rust function to calculate the factorial of a given number.",
    "The capital of France is",
];

pub const MODEL_REPO: &str = "unsloth/Qwen3-0.6B-GGUF";
pub const MODEL_FILE: &str = "Qwen3-0.6B-Q4_K_M.gguf";
pub const MODEL_REVISION: &str = "main";
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3-0.6B";

pub const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/qwen3_0_6b_cpu_reference.json"
);

/// The dtype tier a run was produced under — selects the tolerance bound.
/// Mirrors the plan's §5.1 table exactly; see the module doc above.
#[derive(Clone, Copy, Debug)]
pub enum DtypeTier {
    F32F32,
    /// Also covers "quantized vs a same-quantized-weights CPU reference" —
    /// the plan gives it the identical bound and rationale (rounding of
    /// intermediate activations), so it is not a separate variant here.
    F16OrQuantized,
}

impl DtypeTier {
    pub fn relative_bound(self) -> f32 {
        match self {
            DtypeTier::F32F32 => 1e-3,
            DtypeTier::F16OrQuantized => 2e-2,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StepRecord {
    pub token: u32,
    /// `max|logit|` over the FULL vocab at this step — the tolerance
    /// bound's denominator (`max|logit_ref|`), computed before truncating
    /// to the top-`TOP_K` entries below.
    pub max_abs_logit: f32,
    /// Top-`TOP_K` (vocab_id, logit) pairs, sorted descending by logit.
    pub top_logits: Vec<(u32, f32)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PromptRun {
    pub prompt: String,
    pub steps: Vec<StepRecord>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReferenceFixture {
    pub model_repo: String,
    pub model_file: String,
    pub decode_len: usize,
    pub top_k: usize,
    /// Spread observed between two same-process CPU runs of the SAME
    /// prompt at DIFFERENT thread counts (`RAYON_NUM_THREADS=1` vs the
    /// default), as `max|Δ|` over every logit both runs stored — a real
    /// reduction-order probe, not a repeatability one (see
    /// `regenerate_qwen3_cpu_reference_fixture`'s doc for why same-config
    /// reruns don't measure this). Calibration note from the plan's §5.1:
    /// "the bound must be at least 4x the spread that measures, or it is
    /// too tight to be stable" — recorded here so a tightened bound later
    /// can be checked against it without rerunning the model.
    pub rerun_spread_max_abs_delta: f32,
    pub runs: Vec<PromptRun>,
}

/// Build a `Qwen3Config` from GGUF metadata. Exact copy of
/// `quantized-qwen3/main.rs`'s `qwen3_cfg_from_gguf`, with the CLI stripped —
/// kept in sync by hand since the example is a binary, not a library fn.
pub fn qwen3_cfg_from_gguf(mc: &MmapedContent) -> anyhow::Result<Qwen3Config> {
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
/// the `qwen3_yardstick_tool` example and the cache-gated tests.
pub fn load_model() -> anyhow::Result<(QuantizedQwen3Model, tokenizers::Tokenizer, Qwen3Config)> {
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
pub fn run_greedy(
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

    // FOUND 2026-10-07 (M1 investigation, reviewer-identified): plain
    // `forward()` has NO KV cache — `Qwen3Model` holds only `config` +
    // `weights` (no session state), so `model.forward(&[next_token], pos)`
    // embeds and self-attends over a length-1 sequence, attending ONLY to
    // itself, with zero context from the prompt or earlier generated
    // tokens. The ORIGINAL code here called exactly that for every decode
    // step, making this "CPU reference" context-free from step 1 onward —
    // wrong, not a tolerance issue. The fix re-feeds the GROWING full
    // token sequence every step (expensive — O(n^2) total — but correct)
    // and takes the last position's logits, matching the pattern
    // `fuel-model-llama/tests/paged_decode_parity.rs` and this project's
    // other decode-vs-forward precedent tests already use for their
    // reference side.
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let logits_flat = model
        .forward(&all_tokens, 0)
        .map_err(|e| anyhow::anyhow!("forward: {e}"))?
        .realize_f32();
    let mut logits = slice_last(logits_flat, all_tokens.len());
    let mut steps = Vec::with_capacity(DECODE_LEN);
    let first = record_step(&logits);
    let mut next_token = first.token;
    steps.push(first);

    for _ in 1..DECODE_LEN {
        all_tokens.push(next_token);
        let logits_flat = model
            .forward(&all_tokens, 0)
            .map_err(|e| anyhow::anyhow!("forward: {e}"))?
            .realize_f32();
        logits = slice_last(logits_flat, all_tokens.len());
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
pub enum CandidateLogits<'a> {
    /// A live run's full per-step logit vector (index == vocab id).
    Full(&'a [f32]),
    /// A stored/synthetic top-K view (what `fixture_compares_equal_to_itself`
    /// and the negative controls below work with — there is no "full row"
    /// for historical fixture data, by design; see `TOP_K`'s doc comment).
    TopK(&'a [(u32, f32)]),
}

impl CandidateLogits<'_> {
    pub fn get(&self, vocab_id: u32) -> f32 {
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
pub fn compare_step(
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
/// `run_kv_context_decode_and_compare` for that, which inlines the
/// equivalent full-logit comparison via `CandidateLogits::Full`). Mirrors
/// `fuel-model-llama/tests/paged_decode_parity.rs`'s `assert_close` shape.
pub fn compare_runs(
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

pub fn load_fixture() -> ReferenceFixture {
    let bytes = std::fs::read(FIXTURE_PATH).unwrap_or_else(|e| {
        panic!(
            "{FIXTURE_PATH} is missing or unreadable ({e}) — run \
             `cargo run -p fuel-transformers --features dev-tools --example \
             qwen3_yardstick_tool -- regenerate` once (needs network) to produce it"
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
pub fn regenerate_qwen3_cpu_reference_fixture() {
    let (model, tokenizer, cfg) = load_model().expect("load_model");

    // SAFETY (env mutation): the example is single-threaded at this point and
    // nothing else in the process reads RAYON_NUM_THREADS concurrently.
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

/// Speed harness: prefill and decode tok/s, with a warm-up pass before
/// timing. Manual-run only (needs the real model). M1 reuses this exact
/// shape against the CUDA backend; the device/driver/toolkit strings this
/// prints are the per-card record the plan's M0/M1 rows ask for.
pub fn qwen3_cpu_speed_harness() {
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

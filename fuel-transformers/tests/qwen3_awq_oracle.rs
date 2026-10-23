// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 31c: a real `*-AWQ` checkpoint, run end to end through the
//! loader and the fused `AWQ_MATMUL` op, against an INDEPENDENT reference.
//!
//! The reference is `fixtures/qwen3_0_6b_awq_greedy.json`: the greedy token ids
//! HuggingFace transformers + AutoAWQ produced for the same checkpoint and
//! prompt (see `fixtures/qwen3_0_6b_awq_greedy.PROVENANCE.md`). It is sixteen
//! integers, not weights. Nothing here compares Fuel against Fuel.
//!
//! The checkpoint (`Orion-zhen/Qwen3-0.6B-AWQ`, ~540 MB, licence GPL-3.0) is
//! NOT in the repository. The test reads it from `FUEL_QWEN3_AWQ_DIR` and
//! declares a [`Hardware::Hub`] skip when it is absent, which is fatal under
//! `FUEL_REQUIRE_HUB` (what the self-hosted runner arms). The fixture-only
//! tests below run everywhere.
//!
//! Run the real-checkpoint test:
//! `FUEL_QWEN3_AWQ_DIR=C:\Models\Qwen3-0.6B-AWQ cargo test -p fuel-transformers
//! --release --test qwen3_awq_oracle -- --nocapture`
//!
//! What the real-checkpoint test establishes, and what it does not:
//! - It runs the CPU/decompose path with F32 activations (the dequantise-then-
//!   matmul lowering of `AWQ_MATMUL`). The CUDA F16 kernel is covered
//!   separately by `fuel-dispatch/tests/baracuda_awq_live.rs`, against its own
//!   reference; this test does not exercise it.
//! - Greedy token identity over 16 tokens is a coarse oracle: it can pass with
//!   small logit differences. The sabotage arm below shows the oracle CAN fail.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fuel_tensor::lazy::WeightStorage;
use fuel_test_support::hardware::{Hardware, Missing, skip};
use fuel_transformers::models::lazy_qwen3::{Qwen3Config, Qwen3Model, Qwen3Weights};
use half::f16;

const FIXTURE: &str = include_str!("fixtures/qwen3_0_6b_awq_greedy.json");

struct Reference {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
}

fn reference() -> Reference {
    let v: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture is JSON");
    let ids = |key: &str| -> Vec<u32> {
        v[key]
            .as_array()
            .unwrap_or_else(|| panic!("fixture key {key:?} is not an array"))
            .iter()
            .map(|x| x.as_u64().expect("token id") as u32)
            .collect()
    };
    Reference {
        prompt_ids: ids("prompt_token_ids"),
        generated_ids: ids("generated_token_ids"),
    }
}

/// Index of the first position at which `got` and `want` differ, or `None` when
/// they agree over the whole of `want`. A shorter `got` diverges where it ends.
fn first_divergence(got: &[u32], want: &[u32]) -> Option<usize> {
    (0..want.len()).find(|&i| got.get(i) != Some(&want[i]))
}

#[test]
fn the_fixture_is_the_sixteen_token_reference() {
    let r = reference();
    assert_eq!(r.prompt_ids, vec![785, 6722, 315, 9625, 374]);
    assert_eq!(r.generated_ids.len(), 16);
    assert_eq!(r.generated_ids[0], 12095, "' Paris'");
}

/// The comparator is not vacuous: it accepts the reference, and rejects a
/// single flipped token, a truncation, and an empty candidate.
#[test]
fn the_comparator_rejects_what_it_should() {
    let want = reference().generated_ids;
    assert_eq!(first_divergence(&want, &want), None);
    let mut flipped = want.clone();
    flipped[9] ^= 1;
    assert_eq!(first_divergence(&flipped, &want), Some(9));
    assert_eq!(first_divergence(&want[..5], &want), Some(5));
    assert_eq!(first_divergence(&[], &want), Some(0));
}

fn checkpoint_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("FUEL_QWEN3_AWQ_DIR").map(PathBuf::from);
    let Some(dir) = dir else {
        skip(
            Hardware::Hub,
            Missing::device(
                "FUEL_QWEN3_AWQ_DIR is not set (Orion-zhen/Qwen3-0.6B-AWQ, ~540 MB, GPL-3.0, \
                 never committed)",
            ),
        );
        return None;
    };
    for f in ["config.json", "model.safetensors"] {
        if !dir.join(f).is_file() {
            skip(
                Hardware::Hub,
                Missing::device(format!("{} is missing {f}", dir.display())),
            );
            return None;
        }
    }
    Some(dir)
}

/// `quantization_config.group_size`, after refusing every quantization this
/// loader does not read (a different method, bit width, packing or zero-point
/// convention would load as plausible garbage).
fn awq_group_size(config_json: &str) -> usize {
    let v: serde_json::Value = serde_json::from_str(config_json).expect("config.json");
    let q = &v["quantization_config"];
    assert_eq!(q["quant_method"], "awq", "not an AWQ checkpoint: {q}");
    assert_eq!(q["bits"], 4, "only 4-bit AWQ is read: {q}");
    assert_eq!(q["version"], "gemm", "only the GEMM packing is read: {q}");
    assert_eq!(q["zero_point"], true, "only zero-point AWQ is read: {q}");
    q["group_size"].as_u64().expect("group_size") as usize
}

fn load(dir: &Path) -> Qwen3Model {
    let config_json = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let group = awq_group_size(&config_json);
    let config = Qwen3Config::from_hf_json_str(&config_json).expect("Qwen3Config");
    // SAFETY: the file is read-only for the lifetime of the test process.
    let st =
        unsafe { fuel_loaders::safetensors::MmapedSafetensors::new(dir.join("model.safetensors")) }
            .expect("mmap model.safetensors");
    let weights = Qwen3Weights::load_awq_from_mmapped(&st, &config, group).expect("load AWQ");
    Qwen3Model { config, weights }
}

/// Realize `t` on the CPU after lowering the fused ops that have no CPU
/// binding. `AWQ_MATMUL` has none yet (ROADMAP row 31 lists the CPU binding as
/// not done), and production realize does not auto-lower an unmapped fused op,
/// so the decompose (dequantise, then matmul) is applied explicitly here, the
/// same way `fuel-tensor`'s AWQ unit tests do.
fn realize_lowered(t: &fuel_tensor::lazy::Tensor) -> Vec<f32> {
    let g = t.graph_tensor();
    let graph = g.graph().clone();
    let roots =
        fuel_graph::opt::RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[g.id()]);
    fuel_tensor::pipelined_bridge::realize_one_as::<f32>(
        &graph,
        roots[0],
        &fuel_tensor::Device::cpu(),
    )
    .expect("realize the lowered graph")
}

/// Greedy decode by full recompute (no KV cache). Stops early once the output
/// diverges from `stop_at_divergence_from`, when given, so a sabotaged model
/// does not pay for tokens that cannot change the verdict.
fn greedy(
    model: &Qwen3Model,
    prompt: &[u32],
    new_tokens: usize,
    stop_at_divergence_from: Option<&[u32]>,
) -> Vec<u32> {
    let vocab = model.config.vocab_size;
    let mut tokens = prompt.to_vec();
    let mut out = Vec::new();
    for step in 0..new_tokens {
        let t0 = std::time::Instant::now();
        let logits = realize_lowered(&model.forward(&tokens, 0).expect("forward"));
        let seq = tokens.len();
        assert_eq!(
            logits.len(),
            seq * vocab,
            "logits must be [1, {seq}, {vocab}]"
        );
        let last = &logits[(seq - 1) * vocab..];
        // First maximum wins, as torch.argmax does.
        let mut best = 0usize;
        for (i, &v) in last.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite logit at step {step}, vocab id {i}"
            );
            if v > last[best] {
                best = i;
            }
        }
        eprintln!("step {step}: token {best} ({:.1?})", t0.elapsed());
        tokens.push(best as u32);
        out.push(best as u32);
        if let Some(want) = stop_at_divergence_from
            && first_divergence(&out, &want[..out.len()]).is_some()
        {
            break;
        }
    }
    out
}

/// Same weights, but every layer's `o_proj` scales are zeroed, so attention
/// contributes nothing to the residual stream.
fn without_attention_output(model: &Qwen3Model) -> Qwen3Model {
    let mut weights = model.weights.clone();
    for layer in &mut weights.layers {
        let WeightStorage::Awq(w) = &layer.attn_o else {
            panic!("attn_o is not an AWQ weight");
        };
        let mut cut = (**w).clone();
        cut.scales = Arc::from(vec![f16::ZERO; cut.scales.len()]);
        layer.attn_o = WeightStorage::Awq(Arc::new(cut));
    }
    Qwen3Model {
        config: model.config.clone(),
        weights,
    }
}

#[test]
fn real_awq_checkpoint_reproduces_the_reference_greedy_tokens() {
    let Some(dir) = checkpoint_dir() else {
        return;
    };
    let want = reference();
    let model = load(&dir);

    // Every projection really is the fused AWQ op, not a dense fallback: if the
    // loader quietly dequantised to F32, the run below would still match and
    // prove nothing about row 31.
    for (i, l) in model.weights.layers.iter().enumerate() {
        for (name, w) in [
            ("q", &l.attn_q),
            ("k", &l.attn_k),
            ("v", &l.attn_v),
            ("o", &l.attn_o),
            ("gate", &l.ffn_gate),
            ("up", &l.ffn_up),
            ("down", &l.ffn_down),
        ] {
            assert!(
                matches!(w, WeightStorage::Awq(_)),
                "layer {i} {name}_proj is not WeightStorage::Awq"
            );
        }
    }

    let got = greedy(&model, &want.prompt_ids, want.generated_ids.len(), None);
    assert_eq!(
        first_divergence(&got, &want.generated_ids),
        None,
        "greedy tokens diverge from the HF/AutoAWQ reference.\n got: {got:?}\nwant: {:?}",
        want.generated_ids
    );

    // The oracle can fail: with attention's output projection zeroed the same
    // run must leave the reference within the 16 tokens.
    let sabotaged = without_attention_output(&model);
    let bad = greedy(
        &sabotaged,
        &want.prompt_ids,
        want.generated_ids.len(),
        Some(&want.generated_ids),
    );
    assert!(
        first_divergence(&bad, &want.generated_ids).is_some(),
        "a model with every o_proj zeroed still reproduced the reference: the oracle is vacuous"
    );
}

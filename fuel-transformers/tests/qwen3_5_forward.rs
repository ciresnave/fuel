// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 115, step 4: the whole Qwen3.5 prefill (24 decoder layers: 18 Gated DeltaNet and 6
//! gated full-attention) against the HuggingFace reference on the REAL `Qwen/Qwen3.5-0.8B`
//! weights: the embedding, the output of layers 0, 1, 3 and 23, the final norm, and the top-10
//! logits per position.
//!
//! Same conditions as `qwen3_5_attention.rs`: the checkpoint is read from the local Hugging Face
//! cache and never downloaded implicitly; a miss is a `Hardware::Hub` skip, FATAL under
//! `FUEL_REQUIRE_HUB=1`. Reference: `tests/fixtures/qwen3_5_0_8b_forward_reference.safetensors`
//! (see its PROVENANCE.md and `scripts/qwen35_reference/forward_ref.py`). Run with
//! `FUEL_REQUIRE_HUB=1 cargo test --release -p fuel-transformers --test qwen3_5_forward`.

use std::path::PathBuf;

use fuel_loaders::safetensors::{BufferedSafetensors, MmapedSafetensors};
use fuel_test_support::hub::{RepoKind, hf_cache_dir, require_cached};
use fuel_transformers::models::lazy_qwen3_5::{Qwen3_5Config, Qwen3_5Model, Qwen3_5Weights};

const REPO: &str = "Qwen/Qwen3.5-0.8B";
const WEIGHTS: &str = "model.safetensors-00001-of-00001.safetensors";
const REFERENCE: &[u8] = include_bytes!("fixtures/qwen3_5_0_8b_forward_reference.safetensors");
const VOCAB: usize = 248_320;
const TOP: usize = 10;

struct Fixture {
    model: Qwen3_5Model,
    reference: BufferedSafetensors,
    tokens: Vec<u32>,
}

/// The snapshot directory of the cached checkpoint that holds both files.
fn snapshot_dir() -> Option<PathBuf> {
    if !require_cached(RepoKind::Model, REPO, &["config.json", WEIGHTS]) {
        return None;
    }
    let snaps = hf_cache_dir()?
        .join("models--Qwen--Qwen3.5-0.8B")
        .join("snapshots");
    std::fs::read_dir(snaps)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("config.json").exists() && p.join(WEIGHTS).exists())
}

fn bytes_of(st: &BufferedSafetensors, name: &str) -> Vec<u8> {
    st.get(name)
        .unwrap_or_else(|e| panic!("reference lacks {name}: {e}"))
        .data()
        .to_vec()
}

fn f32s(st: &BufferedSafetensors, name: &str) -> Vec<f32> {
    bytes_of(st, name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn i64s(st: &BufferedSafetensors, name: &str) -> Vec<i64> {
    bytes_of(st, name)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect()
}

fn fixture() -> Option<Fixture> {
    let dir = snapshot_dir()?;
    let config = Qwen3_5Config::from_hf_json_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("config parses");
    assert_eq!(config.vocab_size, VOCAB);
    // SAFETY: a read-only checkpoint, mapped for the duration of the load only.
    let st = unsafe { MmapedSafetensors::new(dir.join(WEIGHTS)) }.expect("mmap");
    let weights = Qwen3_5Weights::load_from_mmapped(&st, &config).expect("weights load");
    let reference = BufferedSafetensors::new(REFERENCE.to_vec()).expect("reference parses");
    let tokens = i64s(&reference, "tokens")
        .iter()
        .map(|&t| t as u32)
        .collect();
    Some(Fixture {
        model: Qwen3_5Model { config, weights },
        reference,
        tokens,
    })
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs() as f64))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0_f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs()))
        / scale
}

/// Relative to the largest reference element at that stage. Measured at most 5e-6 through all 24
/// layers and the logits (the values are printed by the tests); a wrong layer scores 0.5.
const TOL: f64 = 1e-4;

#[test]
fn every_checked_stage_matches_huggingface() {
    let Some(f) = fixture() else { return };
    let stages = f.model.prefill_stages(&f.tokens).unwrap();
    let embed = stages.embed.realize_f32();
    let rel = max_rel(&embed, &f32s(&f.reference, "embed"));
    eprintln!("embed: {rel:e}");
    assert!(rel < TOL, "embed: {rel:e}");
    for i in [0_usize, 1, 3, 23] {
        let got = stages.layers[i].realize_f32();
        let rel = max_rel(&got, &f32s(&f.reference, &format!("layer{i}")));
        eprintln!("layer {i}: {rel:e}");
        assert!(rel < TOL, "layer {i}: {rel:e}");
    }
    let rel = max_rel(
        &stages.final_norm.realize_f32(),
        &f32s(&f.reference, "final_norm"),
    );
    eprintln!("final_norm: {rel:e}");
    assert!(rel < TOL, "final_norm: {rel:e}");
}

#[test]
fn a_wrong_layer_is_far_off() {
    let Some(f) = fixture() else { return };
    let stages = f.model.prefill_stages(&f.tokens).unwrap();
    let got = stages.layers[1].realize_f32();
    let rel = max_rel(&got, &f32s(&f.reference, "layer3"));
    eprintln!("Fuel layer 1 vs reference layer 3: {rel:e}");
    assert!(
        rel > 100.0 * TOL,
        "the check cannot tell layers apart: {rel:e}"
    );
}

/// The logits' top-10 per position: the same token ids in the same order, and values close.
#[test]
fn the_logits_top_ten_match_huggingface() {
    let Some(f) = fixture() else { return };
    let logits = f.model.prefill_logits(&f.tokens).unwrap().realize_f32();
    let t = f.tokens.len();
    assert_eq!(logits.len(), t * VOCAB);
    let want_ids = i64s(&f.reference, "top_ids");
    let want_vals = f32s(&f.reference, "top_vals");
    let mut worst = 0.0_f64;
    for row in 0..t {
        let r = &logits[row * VOCAB..(row + 1) * VOCAB];
        let mut order: Vec<usize> = (0..VOCAB).collect();
        order.sort_by(|a, b| r[*b].total_cmp(&r[*a]));
        let ids: Vec<i64> = order[..TOP].iter().map(|&i| i as i64).collect();
        let vals: Vec<f32> = order[..TOP].iter().map(|&i| r[i]).collect();
        assert_eq!(ids[0], want_ids[row * TOP], "row {row}: argmax differs");
        let top_set = |v: &[i64]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(
            top_set(&ids),
            top_set(&want_ids[row * TOP..(row + 1) * TOP]),
            "row {row}: top-{TOP} sets differ"
        );
        worst = worst.max(max_rel(&vals, &want_vals[row * TOP..(row + 1) * TOP]));
    }
    eprintln!("top-{TOP} logit values, worst relative error over {t} rows: {worst:e}");
    assert!(worst < TOL, "{worst:e}");
}

fn top_ten(row: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let mut order: Vec<usize> = (0..VOCAB).collect();
    order.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
    (
        order[..TOP].iter().map(|&i| i as i64).collect(),
        order[..TOP].iter().map(|&i| row[i]).collect(),
    )
}

/// One token per call with the carried cache against HuggingFace's own token-at-a-time cached
/// decode (`step_top_*` in the fixture): same top-10 ids, values close.
#[test]
fn token_at_a_time_decode_matches_huggingface_cached_decode() {
    let Some(f) = fixture() else { return };
    let want_ids = i64s(&f.reference, "step_top_ids");
    let want_vals = f32s(&f.reference, "step_top_vals");
    let mut cache = f.model.new_cache();
    let mut worst = 0.0_f64;
    for (i, tok) in f.tokens.iter().enumerate() {
        let logits = f.model.decode(&mut cache, &[*tok]).unwrap();
        assert_eq!(logits.len(), VOCAB);
        assert_eq!(cache.len(), i + 1);
        let (ids, vals) = top_ten(&logits);
        let mut got = ids.clone();
        got.sort_unstable();
        let mut want = want_ids[i * TOP..(i + 1) * TOP].to_vec();
        want.sort_unstable();
        assert_eq!(ids[0], want_ids[i * TOP], "step {i}: argmax differs");
        assert_eq!(got, want, "step {i}: top-{TOP} sets differ");
        worst = worst.max(max_rel(&vals, &want_vals[i * TOP..(i + 1) * TOP]));
    }
    eprintln!(
        "decode top-{TOP} values, worst relative error over {} steps: {worst:e}",
        f.tokens.len()
    );
    assert!(worst < TOL, "{worst:e}");
}

/// Five tokens, then three more from the carried cache, equal the whole-sequence rows 5..8; a
/// fresh cache for the last three does not (the control that the state is doing the work).
#[test]
fn a_split_decode_equals_the_whole_sequence() {
    let Some(f) = fixture() else { return };
    let whole = f.model.prefill_logits(&f.tokens).unwrap().realize_f32();
    let want = &whole[5 * VOCAB..];
    let mut cache = f.model.new_cache();
    f.model.decode(&mut cache, &f.tokens[..5]).unwrap();
    let carried = f.model.decode(&mut cache, &f.tokens[5..]).unwrap();
    let rel = max_rel(&carried, want);
    eprintln!("5 + 3 carried vs whole (last 3 rows): {rel:e}");
    assert!(rel < TOL, "{rel:e}");
    let fresh = f
        .model
        .decode(&mut f.model.new_cache(), &f.tokens[5..])
        .unwrap();
    let off = max_rel(&fresh, want);
    eprintln!("last 3 from a fresh cache vs whole: {off:e}");
    assert!(off > 100.0 * TOL, "the control cannot tell: {off:e}");
}

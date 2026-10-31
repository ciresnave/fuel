// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 115: the gated full-attention token mixer against the HuggingFace reference, on the
//! REAL `Qwen/Qwen3.5-0.8B` weights (layers 3 and 7; the model has 6 full-attention layers).
//!
//! Same conditions as `qwen3_5_gdn_mixer.rs`: the checkpoint is read from the local Hugging Face
//! cache and never downloaded implicitly; a miss is a `Hardware::Hub` skip, FATAL under
//! `FUEL_REQUIRE_HUB=1`. The committed reference activations are
//! `tests/fixtures/qwen3_5_0_8b_attn_reference.safetensors` (see its PROVENANCE.md and
//! `scripts/qwen35_reference/attn_ref.py`). Run with
//! `FUEL_REQUIRE_HUB=1 cargo test --release -p fuel-transformers --test qwen3_5_attention`.

use std::path::PathBuf;

use fuel_ir::Shape;
use fuel_loaders::safetensors::{BufferedSafetensors, MmapedSafetensors};
use fuel_tensor::{Device, lazy::Tensor};
use fuel_test_support::hub::{RepoKind, hf_cache_dir, require_cached};
use fuel_transformers::models::lazy_qwen3_5::{
    FullAttentionWeights, Qwen3_5Config, Qwen3_5Mixer, Qwen3_5Weights,
};
use fuel_transformers::models::lazy_qwen3_5_attn::gated_full_attention;

const T: usize = 8;

struct Fixture {
    cfg: Qwen3_5Config,
    layers: Vec<FullAttentionWeights>,
    reference: BufferedSafetensors,
}

const REPO: &str = "Qwen/Qwen3.5-0.8B";
const WEIGHTS: &str = "model.safetensors-00001-of-00001.safetensors";
const REFERENCE: &[u8] = include_bytes!("fixtures/qwen3_5_0_8b_attn_reference.safetensors");

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

fn fixture() -> Option<Fixture> {
    let dir = snapshot_dir()?;
    let cfg = Qwen3_5Config::from_hf_json_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("config parses");
    let file = dir.join(WEIGHTS);
    // SAFETY: a read-only checkpoint, mapped for the duration of the test only.
    let st = unsafe { MmapedSafetensors::new(file) }.expect("mmap");
    let weights = Qwen3_5Weights::load_from_mmapped(&st, &cfg).expect("weights load");
    let layers = weights
        .layers
        .iter()
        .filter_map(|l| match &l.mixer {
            Qwen3_5Mixer::FullAttention(a) => Some(a.clone()),
            Qwen3_5Mixer::GatedDeltaNet(_) => None,
        })
        .collect();
    let reference = BufferedSafetensors::new(REFERENCE.to_vec()).expect("reference parses");
    Some(Fixture {
        cfg,
        layers,
        reference,
    })
}

fn tensor_f32(st: &BufferedSafetensors, name: &str) -> Vec<f32> {
    st.get(name)
        .unwrap_or_else(|e| panic!("reference lacks {name}: {e}"))
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
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

fn input(x: &[f32], rows: usize, hidden: usize) -> Tensor {
    Tensor::from_f32(
        x.to_vec(),
        Shape::from_dims(&[rows, hidden]),
        &Device::cpu(),
    )
    .unwrap()
}

/// Relative to the largest reference element; the measured value is printed by the test.
const TOL: f64 = 1e-4;

/// Position of the full-attention layers among the model's layers is 3, 7, 11, ...: index `n` of
/// `Fixture::layers` is model layer `4 * n + 3`.
const LAYERS: [(usize, &str); 2] = [(0, "attn3"), (1, "attn7")];

#[test]
fn attention_layers_match_huggingface() {
    let Some(f) = fixture() else { return };
    for (idx, name) in LAYERS {
        let x = tensor_f32(&f.reference, &format!("{name}_in"));
        let want = tensor_f32(&f.reference, &format!("{name}_out"));
        let y =
            gated_full_attention(&input(&x, T, f.cfg.hidden_size), &f.layers[idx], &f.cfg).unwrap();
        let rel = max_rel(&y.realize_f32(), &want);
        eprintln!("{name}: max |diff| / max |ref| = {rel:e}");
        assert!(rel < TOL, "{name}: {rel:e}");
    }
}

#[test]
fn a_different_layers_weights_are_far_off() {
    let Some(f) = fixture() else { return };
    let x = tensor_f32(&f.reference, "attn3_in");
    let want = tensor_f32(&f.reference, "attn3_out");
    let y = gated_full_attention(&input(&x, T, f.cfg.hidden_size), &f.layers[1], &f.cfg).unwrap();
    let rel = max_rel(&y.realize_f32(), &want);
    eprintln!("layer 3 input through layer 7 weights vs layer 3 reference: {rel:e}");
    assert!(
        rel > 100.0 * TOL,
        "the check cannot tell layers apart: {rel:e}"
    );
}

/// Causality: the first 3 rows of an 8-token run equal a 3-token run. Fails if a later token
/// leaks into an earlier row or the rotary positions are off.
#[test]
fn a_prefix_run_reproduces_the_first_rows() {
    let Some(f) = fixture() else { return };
    let h = f.cfg.hidden_size;
    let x = tensor_f32(&f.reference, "attn3_in");
    let full = gated_full_attention(&input(&x, T, h), &f.layers[0], &f.cfg)
        .unwrap()
        .realize_f32();
    let part = gated_full_attention(&input(&x[..3 * h], 3, h), &f.layers[0], &f.cfg)
        .unwrap()
        .realize_f32();
    let rel = max_rel(&part, &full[..3 * h]);
    eprintln!("3-token prefix vs first 3 rows of 8: {rel:e}");
    assert!(rel < TOL, "{rel:e}");
}

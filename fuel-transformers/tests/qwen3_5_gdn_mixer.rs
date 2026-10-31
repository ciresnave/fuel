// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 115 step 2: one Gated DeltaNet token mixer against the HuggingFace reference, on
//! the REAL `Qwen/Qwen3.5-0.8B` weights.
//!
//! The checkpoint (~1.7 GB, never committed) is read from the local Hugging Face cache and is
//! never downloaded implicitly: when it is absent the tests declare a `Hardware::Hub` skip, which
//! is FATAL under `FUEL_REQUIRE_HUB=1`. The reference activations are committed
//! (`tests/fixtures/qwen3_5_0_8b_gdn_mixer_reference.safetensors`, 131 KB, activations only; see
//! its PROVENANCE.md and `scripts/qwen35_reference/mixer_ref.py`).
//!
//! WHERE THESE RUN: GitHub-hosted CI cannot hold the checkpoint, so there they SKIP. They are run
//! with `FUEL_REQUIRE_HUB=1` (so a skip is a failure) on a machine whose HF cache has
//! `Qwen/Qwen3.5-0.8B`: `hf download Qwen/Qwen3.5-0.8B`, then
//! `FUEL_REQUIRE_HUB=1 cargo test --release -p fuel-transformers --test qwen3_5_gdn_mixer`.
//!
//! The reference ran HF's chunked prefill path; Fuel runs the token-by-token recurrence, so the
//! two share the maths but not the arithmetic order.

use std::path::PathBuf;

use fuel_ir::Shape;
use fuel_loaders::safetensors::{BufferedSafetensors, MmapedSafetensors};
use fuel_tensor::{Device, lazy::Tensor};
use fuel_test_support::hub::{RepoKind, hf_cache_dir, require_cached};
use fuel_transformers::models::lazy_qwen3_5::{
    GatedDeltaNetWeights, Qwen3_5Config, Qwen3_5Mixer, Qwen3_5Weights,
};
use fuel_transformers::models::lazy_qwen3_5_gdn::{GdnState, gated_delta_net_mixer};

const T: usize = 8;

struct Fixture {
    cfg: Qwen3_5Config,
    layers: Vec<GatedDeltaNetWeights>,
    reference: BufferedSafetensors,
}

const REPO: &str = "Qwen/Qwen3.5-0.8B";
const WEIGHTS: &str = "model.safetensors-00001-of-00001.safetensors";
const REFERENCE: &[u8] = include_bytes!("fixtures/qwen3_5_0_8b_gdn_mixer_reference.safetensors");

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
            Qwen3_5Mixer::GatedDeltaNet(g) => Some(g.clone()),
            Qwen3_5Mixer::FullAttention(_) => None,
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

/// Relative to the largest reference element. Measured (see the test output) well under this;
/// `a_different_layers_reference_is_far_off` shows what a wrong layer scores.
const TOL: f64 = 1e-4;

#[test]
fn gdn_layers_match_huggingface() {
    let Some(f) = fixture() else { return };
    for (idx, name) in [(0_usize, "gdn0"), (1, "gdn1")] {
        let x = tensor_f32(&f.reference, &format!("{name}_in"));
        let want = tensor_f32(&f.reference, &format!("{name}_out"));
        let xt = input(&x, T, f.cfg.hidden_size);
        let (y, _) = gated_delta_net_mixer(&xt, &f.layers[idx], &f.cfg, None).unwrap();
        let got = y.realize_f32();
        let rel = max_rel(&got, &want);
        eprintln!("{name}: max |diff| / max |ref| = {rel:e}");
        assert!(rel < TOL, "{name}: {rel:e}");
    }
}

#[test]
fn a_different_layers_reference_is_far_off() {
    let Some(f) = fixture() else { return };
    let x = tensor_f32(&f.reference, "gdn0_in");
    let wrong = tensor_f32(&f.reference, "gdn1_out");
    let xt = input(&x, T, f.cfg.hidden_size);
    let (y, _) = gated_delta_net_mixer(&xt, &f.layers[0], &f.cfg, None).unwrap();
    let rel = max_rel(&y.realize_f32(), &wrong);
    eprintln!("layer 0 output vs layer 1 reference: {rel:e}");
    assert!(
        rel > 100.0 * TOL,
        "the check cannot tell layers apart: {rel:e}"
    );
}

/// Prefill of the first 5 tokens, then the last 3 from the carried `(S, conv window)`, must equal
/// the 8-token run on those last 3 rows: the state is complete.
#[test]
fn carried_state_reproduces_the_whole_sequence() {
    let Some(f) = fixture() else { return };
    let h = f.cfg.hidden_size;
    let x = tensor_f32(&f.reference, "gdn0_in");
    let full = {
        let (y, _) = gated_delta_net_mixer(&input(&x, T, h), &f.layers[0], &f.cfg, None).unwrap();
        y.realize_f32()
    };
    let (_, st) =
        gated_delta_net_mixer(&input(&x[..5 * h], 5, h), &f.layers[0], &f.cfg, None).unwrap();
    // Realise the carried state and rebuild it on the next graph (tensors are graph-affine).
    let s_data = st.recurrent.realize_f32();
    let w_data = st.conv_window.realize_f32();
    let tail = input(&x[5 * h..], 3, h);
    let carried = GdnState {
        recurrent: tail
            .const_f32_like(
                s_data,
                Shape::from_dims(&[
                    f.cfg.linear_num_value_heads,
                    f.cfg.linear_key_head_dim,
                    f.cfg.linear_value_head_dim,
                ]),
            )
            .unwrap(),
        conv_window: tail
            .const_f32_like(
                w_data,
                Shape::from_dims(&[1, f.cfg.linear_conv_dim(), f.cfg.linear_conv_kernel_dim - 1]),
            )
            .unwrap(),
    };
    let (y, _) = gated_delta_net_mixer(&tail, &f.layers[0], &f.cfg, Some(&carried)).unwrap();
    let rel = max_rel(&y.realize_f32(), &full[5 * h..]);
    eprintln!("carried vs whole (last 3 rows): {rel:e}");
    assert!(rel < TOL, "{rel:e}");
}

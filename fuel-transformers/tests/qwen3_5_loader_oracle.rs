// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 40 stage 1: the Qwen3.5 loader against the REAL `Qwen/Qwen3.5-0.8B` checkpoint.
//!
//! The checkpoint (~1.7 GB, Apache-2.0, never committed) is read from `FUEL_QWEN35_DIR` and a
//! [`Hardware::Hub`] skip is declared when it is absent, which is fatal under `FUEL_REQUIRE_HUB`.
//!
//! Run: `FUEL_QWEN35_DIR=C:\Models\Qwen3.5-0.8B cargo test -p fuel-transformers --release
//! --test qwen3_5_loader_oracle -- --nocapture`
//!
//! What this establishes: the loader reads exactly the language-model tensors the checkpoint
//! has, each at the shape the config implies, and applies the zero-centred norm convention. The
//! tensor-name set is enumerated here from the config, independently of the loader, and compared
//! with the names in the file. The spot values were read from the same file by a separate
//! script (Python, byte-range reads of the safetensors header and data), not by this loader.
//! What it does NOT establish: that the model computes anything (there is no forward pass yet).

use std::collections::BTreeSet;
use std::path::PathBuf;

use fuel_test_support::hardware::{Hardware, Missing, skip};
use fuel_transformers::models::lazy_qwen3_5::{
    Qwen3_5Config, Qwen3_5LayerKind, Qwen3_5Mixer, Qwen3_5Model, Qwen3_5Weights,
};

const PREFIX: &str = "model.language_model.";

fn checkpoint_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("FUEL_QWEN35_DIR").map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device("FUEL_QWEN35_DIR is not set (Qwen/Qwen3.5-0.8B, ~1.7 GB, Apache-2.0)"),
        );
        return None;
    };
    if !dir.join("config.json").is_file() || safetensors_file(&dir).is_none() {
        skip(
            Hardware::Hub,
            Missing::device(format!(
                "{} lacks config.json or a .safetensors file",
                dir.display()
            )),
        );
        return None;
    }
    Some(dir)
}

fn safetensors_file(dir: &std::path::Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
}

/// Every language-model tensor name the config implies, enumerated independently of the loader.
fn expected_names(cfg: &Qwen3_5Config) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert(format!("{PREFIX}embed_tokens.weight"));
    names.insert(format!("{PREFIX}norm.weight"));
    for (i, kind) in cfg.layer_types.iter().enumerate() {
        let p = format!("{PREFIX}layers.{i}");
        for n in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.insert(format!("{p}.{n}"));
        }
        let mixer: &[&str] = match kind {
            Qwen3_5LayerKind::LinearAttention => &[
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ],
            Qwen3_5LayerKind::FullAttention => &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ],
        };
        for n in mixer {
            names.insert(format!("{p}.{n}"));
        }
    }
    names
}

fn mean(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len() as f64
}

fn load_config(dir: &std::path::Path) -> Qwen3_5Config {
    Qwen3_5Config::from_hf_json_str(
        &std::fs::read_to_string(dir.join("config.json")).expect("config.json"),
    )
    .expect("config parses")
}

fn open_checkpoint(dir: &std::path::Path) -> fuel_loaders::safetensors::MmapedSafetensors {
    let file = safetensors_file(dir).expect("checked by checkpoint_dir");
    // SAFETY: the file is a read-only checkpoint this test does not modify, and it is mapped for
    // the duration of the test only.
    unsafe { fuel_loaders::safetensors::MmapedSafetensors::new(file) }.expect("mmap")
}

/// Independent accounting: the names the config implies vs the names in the file.
fn assert_tensor_accounting(
    st: &fuel_loaders::safetensors::MmapedSafetensors,
    cfg: &Qwen3_5Config,
) {
    let in_file: BTreeSet<String> = st
        .tensors()
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.starts_with(PREFIX))
        .collect();
    let expected = expected_names(cfg);
    let missing: Vec<_> = expected.difference(&in_file).collect();
    let unread: Vec<_> = in_file.difference(&expected).collect();
    println!(
        "language-model tensors: {} in file, {} implied by the config, {} missing, {} unread",
        in_file.len(),
        expected.len(),
        missing.len(),
        unread.len()
    );
    assert!(missing.is_empty(), "implied but absent: {missing:?}");
    assert!(unread.is_empty(), "present but not implied: {unread:?}");
    // Positive control: the sets are not both empty, and the visual tower is in the file but
    // outside the prefix (so the filter is doing something).
    assert!(in_file.len() > 100);
    assert!(
        st.tensors()
            .iter()
            .any(|(n, _)| n.starts_with("model.visual."))
    );
}

/// Spot values read from the same file by an independent script (means over the whole tensor):
/// layers.0.input_layernorm mean 0.238 (so 1.238 folded), final norm 3.309 (4.309),
/// linear_attn.A_log mean -1.979 (kept as is), linear_attn.norm mean 0.952 (not folded).
fn assert_spot_values(model: &Qwen3_5Model) {
    let l0 = &model.weights.layers[0];
    assert!((mean(&l0.input_norm_gain) - 1.238).abs() < 0.005);
    assert!((mean(&model.weights.final_norm_gain) - 4.309).abs() < 0.005);
    let Qwen3_5Mixer::GatedDeltaNet(g) = &l0.mixer else {
        panic!("layer 0 is a Gated DeltaNet layer");
    };
    assert!((mean(&g.a_log) - (-1.979)).abs() < 0.005);
    assert!((mean(&g.norm_gain) - 0.952).abs() < 0.005);
    assert_eq!(g.conv1d_weight.len(), 6144 * 4);
}

#[test]
fn the_loader_reads_exactly_the_language_model_tensors_of_the_real_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let cfg = load_config(&dir);
    let st = open_checkpoint(&dir);
    assert_tensor_accounting(&st, &cfg);

    let weights = Qwen3_5Weights::load_from_mmapped(&st, &cfg).expect("loads, shapes checked");
    let model = Qwen3_5Model {
        config: cfg.clone(),
        weights,
    };
    assert_eq!(model.weights.layers.len(), 24);
    assert_eq!(model.num_linear_layers(), 18);
    for (layer, kind) in model.weights.layers.iter().zip(&cfg.layer_types) {
        let ok = matches!(
            (&layer.mixer, kind),
            (
                Qwen3_5Mixer::GatedDeltaNet(_),
                Qwen3_5LayerKind::LinearAttention
            ) | (
                Qwen3_5Mixer::FullAttention(_),
                Qwen3_5LayerKind::FullAttention
            )
        );
        assert!(ok, "mixer variant disagrees with layer_types");
    }
    assert_spot_values(&model);

    // The forward pass is a typed refusal, not a wrong answer.
    let err = model.forward(&[1, 2, 3], 0).unwrap_err().to_string();
    assert!(err.contains("GAP-368"), "{err}");
}

/// Sabotage sibling: a config that disagrees with the file by one dimension must be refused by
/// the shape check, so the green above cannot come from a loader that never looks at shapes.
#[test]
fn a_config_that_disagrees_with_the_file_is_refused() {
    let Some(dir) = checkpoint_dir() else { return };
    let mut cfg = load_config(&dir);
    cfg.intermediate_size += 1;
    let st = open_checkpoint(&dir);
    let err = Qwen3_5Weights::load_from_mmapped(&st, &cfg)
        .expect_err("must be refused")
        .to_string();
    assert!(err.contains("shape"), "{err}");
}

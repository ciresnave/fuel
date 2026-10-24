// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, slice A (`docs/design-notes/2026-10-24-mlmf-move-survey.md`): AWQ configuration and
//! packed-layer discovery now come from `mlmf-awq`, through `fuel_loaders::awq`.
//!
//! Equivalence, on the REAL `Orion-zhen/Qwen3-0.6B-AWQ` checkpoint (read from
//! `FUEL_QWEN3_AWQ_DIR`; a `Hardware::Hub` skip when absent), against the OLD path:
//!
//! - the group size equals what the oracle used to read by hand from `config.json`
//!   (`quantization_config.group_size`, read here with a plain `serde_json` lookup);
//! - the set of located layers equals the set of projections `Qwen3Weights::load_awq_from_mmapped`
//!   reads, enumerated here from the model config (7 projections per layer), and each located
//!   layer's `(out, in)` equals the pair the loader passes for that projection;
//! - for every located layer the three tensors exist in the old reader (`MmapedSafetensors`) with
//!   the shapes `mlmf-awq` derived from.
//!
//! Positive control: the located set is not empty and is the size the config implies. Sabotage
//! sibling: a configuration the old oracle refused is still refused.

use std::collections::BTreeMap;
use std::path::PathBuf;

use fuel_loaders::awq::{awq_group_size, locate_awq_layers};
use fuel_loaders::safetensors::MmapedSafetensors;
use fuel_test_support::hardware::{Hardware, Missing, skip};
use fuel_transformers::models::lazy_qwen3::Qwen3Config;

fn checkpoint_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("FUEL_QWEN3_AWQ_DIR").map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device("FUEL_QWEN3_AWQ_DIR is not set (Orion-zhen/Qwen3-0.6B-AWQ, ~540 MB)"),
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

/// The `(prefix, out_features, in_features)` of every projection `load_awq_from_mmapped` reads,
/// from the model config: the same table `Qwen3Weights::load_attn_projections` and
/// `load_ffn_projections` build.
fn loader_projections(cfg: &Qwen3Config) -> BTreeMap<String, (usize, usize)> {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let mut out = BTreeMap::new();
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        for (name, shape) in [
            ("self_attn.q_proj", (q_dim, h)),
            ("self_attn.k_proj", (kv_dim, h)),
            ("self_attn.v_proj", (kv_dim, h)),
            ("self_attn.o_proj", (h, q_dim)),
            ("mlp.gate_proj", (cfg.intermediate_size, h)),
            ("mlp.up_proj", (cfg.intermediate_size, h)),
            ("mlp.down_proj", (h, cfg.intermediate_size)),
        ] {
            out.insert(format!("{p}.{name}"), shape);
        }
    }
    out
}

#[test]
fn mlmf_awq_agrees_with_the_loader_on_the_real_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let config_bytes = std::fs::read(dir.join("config.json")).expect("config.json");
    let st_bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");

    // Old path, group size: a plain JSON lookup, as the oracle did.
    let v: serde_json::Value = serde_json::from_slice(&config_bytes).expect("config.json");
    let old_group_size = v["quantization_config"]["group_size"]
        .as_u64()
        .expect("group_size") as usize;

    let layout = locate_awq_layers(&config_bytes, &st_bytes).expect("mlmf-awq locates the layers");
    assert_eq!(layout.group_size, old_group_size);
    assert_eq!(awq_group_size(&config_bytes).unwrap(), old_group_size);

    // Old path, layer set and geometry: what the loader reads.
    let cfg = Qwen3Config::from_hf_json_str(std::str::from_utf8(&config_bytes).unwrap())
        .expect("model config");
    let expected = loader_projections(&cfg);
    let located: BTreeMap<String, (usize, usize)> = layout
        .layers
        .iter()
        .map(|l| {
            (
                l.prefix.clone(),
                (l.out_features as usize, l.in_features as usize),
            )
        })
        .collect();
    println!(
        "{} layers located by mlmf-awq, {} projections read by the loader, group_size {}",
        located.len(),
        expected.len(),
        layout.group_size
    );
    assert_eq!(located, expected);
    // Positive control: 28 layers x 7 projections is not an empty comparison.
    assert_eq!(expected.len(), cfg.num_hidden_layers * 7);
    assert!(expected.len() > 100);

    // Old reader: the three tensors exist with the shapes mlmf-awq derived its geometry from.
    // SAFETY: a read-only checkpoint this test does not modify, mapped for the test's duration.
    let st = unsafe { MmapedSafetensors::new(dir.join("model.safetensors")) }.expect("mmap");
    for l in &layout.layers {
        let shape = |name: &str| st.get(name).expect(name).shape().to_vec();
        let groups = l.group_count as usize;
        assert_eq!(
            shape(&l.qweight),
            vec![l.in_features as usize, l.out_features as usize / 8]
        );
        assert_eq!(shape(&l.qzeros), vec![groups, l.out_features as usize / 8]);
        assert_eq!(shape(&l.scales), vec![groups, l.out_features as usize]);
    }
}

/// Sabotage sibling: the located set must move when the input moves. Asking for a group size the
/// checkpoint's geometry cannot satisfy makes every layer malformed, which is an error.
#[test]
fn a_wrong_group_size_makes_the_scan_fail_not_pass() {
    let Some(dir) = checkpoint_dir() else { return };
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let st_bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let right = awq_group_size(config.as_bytes()).unwrap();
    let wrong = config.replace(
        &format!("\"group_size\": {right}"),
        &format!("\"group_size\": {}", right * 3 + 1),
    );
    assert_ne!(wrong, config, "the sabotage must apply");
    assert!(locate_awq_layers(wrong.as_bytes(), &st_bytes).is_err());
}

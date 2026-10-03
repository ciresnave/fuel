// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model architecture detection from GGUF metadata or tensor names.
//!
//! GGUF files carry a `general.architecture` metadata key which is the
//! authoritative source (llama.cpp sets it; ggml-quantized exports set
//! it). When that key is missing — e.g. on older files or safetensors
//! dumps converted to GGUF by non-standard tools — we fall back to
//! pattern-matching on the tensor name table.
//!
//! This is intentionally lean: we classify into families (LLaMA-like,
//! Qwen-like, Phi, Gemma, …) rather than every specific variant.
//! Variant-specific config still comes from the model's own config
//! (context length, RoPE base, head dims, etc.). Arch detection only
//! tells the caller "which loader should I hand this file to?".
//!
//! **Why detect?** Fuel has ~10 `quantized_*` model loaders; picking
//! the right one currently means the caller either hard-codes it per
//! example or parses `config.json` separately. With detection, a
//! generic "load any GGUF" entry point is possible.

use super::gguf_file::{Content, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    Llama,
    Qwen2,
    Qwen3,
    Qwen3Moe,
    Phi,
    Phi3,
    Gemma,
    Gemma3,
    Glm4,
    Lfm2,
    SmolLm3,
    Gpt2,
    GptNeoX,
    Unknown,
}

impl Architecture {
    /// Human-readable label, matching the GGUF `general.architecture`
    /// convention where applicable.
    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen2 => "qwen2",
            Architecture::Qwen3 => "qwen3",
            Architecture::Qwen3Moe => "qwen3moe",
            Architecture::Phi => "phi2",
            Architecture::Phi3 => "phi3",
            Architecture::Gemma => "gemma",
            Architecture::Gemma3 => "gemma3",
            Architecture::Glm4 => "glm4",
            Architecture::Lfm2 => "lfm2",
            Architecture::SmolLm3 => "smollm3",
            Architecture::Gpt2 => "gpt2",
            Architecture::GptNeoX => "gptneox",
            Architecture::Unknown => "unknown",
        }
    }

    fn from_metadata_string(s: &str) -> Self {
        // Normalize to lowercase and strip hyphens/underscores for
        // robustness against casing variants like "Qwen3-MoE" or
        // "qwen3_moe".
        let norm: String = s
            .chars()
            .flat_map(|c| c.to_lowercase())
            .filter(|c| *c != '-' && *c != '_')
            .collect();
        match norm.as_str() {
            "llama" => Architecture::Llama,
            "qwen2" => Architecture::Qwen2,
            "qwen3" => Architecture::Qwen3,
            "qwen3moe" => Architecture::Qwen3Moe,
            "phi" | "phi2" => Architecture::Phi,
            "phi3" => Architecture::Phi3,
            "gemma" => Architecture::Gemma,
            "gemma3" => Architecture::Gemma3,
            "glm4" | "chatglm4" => Architecture::Glm4,
            "lfm2" => Architecture::Lfm2,
            "smollm3" => Architecture::SmolLm3,
            "gpt2" => Architecture::Gpt2,
            "gptneox" => Architecture::GptNeoX,
            _ => Architecture::Unknown,
        }
    }
}

/// Primary detection path for GGUF: read `general.architecture`. Falls
/// back to tensor-name pattern matching ONLY when the key is genuinely
/// absent.
///
/// **A declared value -- recognized or not, string or not -- is never
/// overridden by a tensor-name guess** (adversarial-review finding,
/// 2026-10-03, ported from mlmf-gguf-arch's same fix): llama.cpp writes
/// `blk.N.*` tensor names for nearly every architecture, not only the
/// ones this table lists, so the fallback fires on almost any file --
/// including one that explicitly declared a real, just-unlisted
/// architecture like `nomic-bert-moe` or `bert`. Silently answering
/// `Qwen3Moe`/`Llama` for those is indistinguishable from a file that
/// genuinely declared them. `content.metadata` is an eager, COMPLETE map
/// by the time this runs (both `Content::read` and `Content::open`
/// build it fully or return a typed `Err` first), so "key not in the
/// map" here means genuine absence, never "not read yet" -- unlike
/// mlmf-core's lazily-backed `MetadataSource`, fuel needs no
/// `index_complete()` guard to make that distinction.
pub fn detect_from_gguf(content: &Content) -> Architecture {
    match content.metadata.get("general.architecture") {
        Some(Value::String(s)) => Architecture::from_metadata_string(s),
        // Declared but not a string -- a real value, not an absence, so
        // this must decline too, same discipline as an unrecognized
        // string (a real writer bug would otherwise silently route
        // through the tensor-name guess instead of surfacing as unknown).
        Some(_) => Architecture::Unknown,
        // Genuinely absent: fall back to the tensor-name heuristic.
        None => detect_from_tensor_names(content.tensor_infos.keys().map(|s| s.as_str())),
    }
}

/// Fallback for files missing `general.architecture`. Looks at the set
/// of tensor names for family-distinctive patterns. Deliberately
/// narrow — this only disambiguates the major families GGUF'd tensor
/// layouts actually differ on.
pub fn detect_from_tensor_names<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Architecture {
    let mut has_blk_any = false;
    let mut has_expert = false; // MoE marker
    let mut has_gpt2_style = false; // transformer.h.N.attn.c_attn.weight
    let mut has_neox_style = false; // gpt_neox.layers.N.attention
    for n in names {
        if n.starts_with("blk.") {
            has_blk_any = true;
            if n.contains(".ffn_gate_exps") || n.contains(".ffn_down_exps") {
                has_expert = true;
            }
        }
        if n.contains("transformer.h.") && n.contains(".attn.c_attn") {
            has_gpt2_style = true;
        }
        if n.contains("gpt_neox.layers.") {
            has_neox_style = true;
        }
    }
    if has_expert {
        return Architecture::Qwen3Moe;
    }
    if has_blk_any {
        // Llama-family tensor layout (also used by Qwen2/3, Phi3, Gemma,
        // Mistral, etc. in GGUF). Without metadata we can't pin down
        // which variant, so return the family root.
        return Architecture::Llama;
    }
    if has_gpt2_style {
        return Architecture::Gpt2;
    }
    if has_neox_style {
        return Architecture::GptNeoX;
    }
    Architecture::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_architecture_names() {
        assert_eq!(
            Architecture::from_metadata_string("Qwen3-MoE"),
            Architecture::Qwen3Moe
        );
        assert_eq!(
            Architecture::from_metadata_string("qwen3_moe"),
            Architecture::Qwen3Moe
        );
        assert_eq!(
            Architecture::from_metadata_string("LLaMA"),
            Architecture::Llama
        );
        assert_eq!(
            Architecture::from_metadata_string("phi2"),
            Architecture::Phi
        );
        assert_eq!(
            Architecture::from_metadata_string("something-new"),
            Architecture::Unknown
        );
    }

    #[test]
    fn tensor_name_fallback_picks_moe() {
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ];
        assert_eq!(
            detect_from_tensor_names(names.iter().copied()),
            Architecture::Qwen3Moe
        );
    }

    #[test]
    fn tensor_name_fallback_picks_llama_family() {
        let names = ["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        assert_eq!(
            detect_from_tensor_names(names.iter().copied()),
            Architecture::Llama
        );
    }

    #[test]
    fn tensor_name_fallback_picks_gpt2() {
        let names = ["transformer.h.0.attn.c_attn.weight"];
        assert_eq!(
            detect_from_tensor_names(names.iter().copied()),
            Architecture::Gpt2
        );
    }

    #[test]
    fn tensor_name_fallback_unknown_when_empty() {
        let names: [&str; 0] = [];
        assert_eq!(
            detect_from_tensor_names(names.iter().copied()),
            Architecture::Unknown
        );
    }

    /// Builds a `Content` with the given declared `general.architecture`
    /// value (or none) and tensor names, for `detect_from_gguf` tests --
    /// `detect_from_gguf` only reads `content.metadata` and
    /// `content.tensor_infos.keys()`, so the other fields are placeholders.
    fn fake_content(declared_architecture: Option<Value>, tensor_names: &[&str]) -> Content {
        let mut metadata = std::collections::HashMap::new();
        if let Some(v) = declared_architecture {
            metadata.insert("general.architecture".to_string(), v);
        }
        let mut tensor_infos = std::collections::HashMap::new();
        for name in tensor_names {
            tensor_infos.insert(
                name.to_string(),
                super::super::gguf_file::TensorInfo {
                    ggml_dtype: fuel_ir::GgmlDType::F32,
                    shape: fuel_ir::Shape::from(vec![1usize]),
                    offset: 0,
                },
            );
        }
        Content {
            magic: super::super::gguf_file::VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 0,
        }
    }

    /// BORN-RED (adversarial-review finding, 2026-10-03): a file that
    /// DECLARES `general.architecture = "nomic-bert-moe"` -- a real,
    /// just-unlisted architecture, not absence -- with MoE-style expert
    /// tensor names (`blk.N.ffn_gate_exps` / `.ffn_down_exps`, which
    /// llama.cpp writes for several real MoE architectures, not only
    /// Qwen3-MoE) must NOT be silently reported as `Qwen3Moe`. Before the
    /// fix, `detect_from_gguf` fell through to the tensor-name guess for
    /// ANY unrecognized declared string, indistinguishable from a file
    /// that genuinely declared Qwen3Moe.
    #[test]
    fn declared_unrecognized_architecture_is_not_overridden_by_tensor_guess() {
        let content = fake_content(
            Some(Value::String("nomic-bert-moe".to_string())),
            &["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"],
        );
        assert_eq!(
            detect_from_gguf(&content),
            Architecture::Unknown,
            "a declared-but-unrecognized architecture must report Unknown, \
             never silently adopt whatever the tensor-name heuristic guesses"
        );
    }

    /// Same defect, the `bert` case from the PM's handoff: a plain blk.N
    /// tensor layout (no expert markers) with `general.architecture =
    /// "bert"` must not be reported as `Llama` just because BERT-family
    /// GGUF exports commonly use llama.cpp's generic `blk.N.*` naming.
    #[test]
    fn declared_bert_is_not_overridden_by_llama_family_tensor_guess() {
        let content = fake_content(
            Some(Value::String("bert".to_string())),
            &["blk.0.attn_q.weight", "blk.0.attn_k.weight"],
        );
        assert_eq!(detect_from_gguf(&content), Architecture::Unknown);
    }

    /// A recognized declared string wins even when the tensor names would
    /// suggest something else entirely (conflicting-evidence case) -- the
    /// declared value is authoritative and tensor names are never even
    /// consulted once it resolves.
    #[test]
    fn declared_recognized_architecture_wins_over_conflicting_tensor_names() {
        let content = fake_content(
            Some(Value::String("qwen3".to_string())),
            &["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"], // MoE markers
        );
        assert_eq!(detect_from_gguf(&content), Architecture::Qwen3);
    }

    /// A declared value that is present but NOT a string (a real writer
    /// bug, e.g. a U32 under `general.architecture`) is a DECLARED value,
    /// not an absence -- the PM's correction to the initial plan. It must
    /// decline (`Unknown`), never fall through to a tensor-name guess, the
    /// same discipline as the unrecognized-string case.
    #[test]
    fn declared_non_string_value_does_not_fall_through_to_tensor_guess() {
        let content = fake_content(
            Some(Value::U32(42)),
            &["blk.0.attn_q.weight", "blk.0.attn_k.weight"], // would guess Llama
        );
        assert_eq!(detect_from_gguf(&content), Architecture::Unknown);
    }

    /// A genuinely absent key (never declared at all) is the ONLY case
    /// that still falls through to the tensor-name heuristic -- this is
    /// the behavior the fix preserves, not removes.
    #[test]
    fn absent_architecture_key_falls_through_to_tensor_guess() {
        let content = fake_content(None, &["blk.0.attn_q.weight", "blk.0.attn_k.weight"]);
        assert_eq!(detect_from_gguf(&content), Architecture::Llama);
    }
}

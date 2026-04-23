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
/// back to tensor-name pattern matching if the key is absent or
/// resolves to `Unknown`.
pub fn detect_from_gguf(content: &Content) -> Architecture {
    if let Some(Value::String(s)) = content.metadata.get("general.architecture") {
        let a = Architecture::from_metadata_string(s);
        if a != Architecture::Unknown {
            return a;
        }
    }
    detect_from_tensor_names(content.tensor_infos.keys().map(|s| s.as_str()))
}

/// Fallback for files missing `general.architecture`. Looks at the set
/// of tensor names for family-distinctive patterns. Deliberately
/// narrow — this only disambiguates the major families GGUF'd tensor
/// layouts actually differ on.
pub fn detect_from_tensor_names<'a, I: IntoIterator<Item = &'a str>>(names: I) -> Architecture {
    let mut has_blk_any = false;
    let mut has_expert = false;       // MoE marker
    let mut has_gpt2_style = false;   // transformer.h.N.attn.c_attn.weight
    let mut has_neox_style = false;   // gpt_neox.layers.N.attention
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
        assert_eq!(Architecture::from_metadata_string("Qwen3-MoE"), Architecture::Qwen3Moe);
        assert_eq!(Architecture::from_metadata_string("qwen3_moe"), Architecture::Qwen3Moe);
        assert_eq!(Architecture::from_metadata_string("LLaMA"), Architecture::Llama);
        assert_eq!(Architecture::from_metadata_string("phi2"), Architecture::Phi);
        assert_eq!(Architecture::from_metadata_string("something-new"), Architecture::Unknown);
    }

    #[test]
    fn tensor_name_fallback_picks_moe() {
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ];
        assert_eq!(detect_from_tensor_names(names.iter().copied()), Architecture::Qwen3Moe);
    }

    #[test]
    fn tensor_name_fallback_picks_llama_family() {
        let names = ["blk.0.attn_q.weight", "blk.0.attn_k.weight"];
        assert_eq!(detect_from_tensor_names(names.iter().copied()), Architecture::Llama);
    }

    #[test]
    fn tensor_name_fallback_picks_gpt2() {
        let names = ["transformer.h.0.attn.c_attn.weight"];
        assert_eq!(detect_from_tensor_names(names.iter().copied()), Architecture::Gpt2);
    }

    #[test]
    fn tensor_name_fallback_unknown_when_empty() {
        let names: [&str; 0] = [];
        assert_eq!(detect_from_tensor_names(names.iter().copied()), Architecture::Unknown);
    }
}

//! Phase 6c validation: DP planner vs recommend_placement on real
//! anchor forward graphs.
//!
//! USAGE
//!
//!     cargo run --release --bin dp_diff
//!     cargo run --release --bin dp_diff --features cuda
//!     cargo run --release --bin dp_diff --features "cuda vulkan"
//!
//! For each Phase 6a anchor, builds a synthetic forward graph and
//! runs both planners on it. Reports per-anchor:
//!
//!   - total node count
//!   - number of nodes where the two planners disagree
//!   - the breakdown of disagreements (which DeviceLocation each
//!     planner picks)
//!
//! The disagreements are where the DP planner's transfer-cost
//! awareness changed the answer — those are the cases Phase 6b's
//! single-op winner-only model would have routed sub-optimally.

use fuel::dispatch::{Criterion, DispatchTable};
use fuel::lazy::LazyTensor;
use fuel::scheduling::{dp_plan, prepare_dp_inputs, recommend_placement, ScheduleOptions};
use fuel_core_types::{probe::BackendId, DeviceLocation};
use fuel_graph::{NodeId, Op};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Probing + judging + measuring bandwidth (cached if hardware unchanged)...");
    let t0 = std::time::Instant::now();
    let (probe, profile, bandwidth) = prepare_dp_inputs(ScheduleOptions::default())?;
    eprintln!("  done in {:.2?}", t0.elapsed());
    eprintln!(
        "  {} device(s); {} profile entries; {} bandwidth entries.",
        probe.devices.len(),
        profile.entries.len(),
        bandwidth.entries.len(),
    );
    eprintln!();

    let mut backends_seen: HashSet<BackendId> = Default::default();
    let mut available_backends: Vec<BackendId> = Vec::new();
    for d in &probe.devices {
        if backends_seen.insert(d.backend) {
            available_backends.push(d.backend);
        }
    }
    let table = DispatchTable::build(&profile);

    eprintln!("Anchors:");
    eprintln!("{:<20}  {:>8}  {:>10}  {:>14}", "anchor", "nodes", "disagreed", "agreement %");
    eprintln!("{}", "-".repeat(60));

    let mut all_reports = Vec::new();
    for anchor in build_anchors() {
        let report = compare_planners(
            &anchor.label,
            &anchor.outputs,
            &table,
            &profile,
            &bandwidth,
            &available_backends,
        );
        let pct = if report.total_nodes == 0 {
            100.0
        } else {
            100.0 * (1.0 - report.disagreements as f64 / report.total_nodes as f64)
        };
        eprintln!(
            "{:<20}  {:>8}  {:>10}  {:>13.1}%",
            anchor.label, report.total_nodes, report.disagreements, pct,
        );
        all_reports.push((anchor.label, report));
    }

    eprintln!();
    let any_disagree = all_reports.iter().any(|(_, r)| r.disagreements > 0);
    if any_disagree {
        eprintln!("Disagreement breakdown (DP placement differs from recommend_placement):");
        for (label, report) in &all_reports {
            if report.disagreements == 0 { continue; }
            eprintln!("  [{label}]");
            for ((op, transition), count) in &report.breakdown {
                eprintln!("    {op:<10} {transition}  ×{count}");
            }
        }
    } else {
        eprintln!("All planners agreed on every node. Likely because the dispatch");
        eprintln!("table picked the same backend (typically CPU) for every op at");
        eprintln!("the synthetic test sizes — every op below the CPU↔GPU crossover.");
        eprintln!("Run on production-scale anchors (e.g. BERT-base hidden=1024)");
        eprintln!("to see DP and recommend diverge.");
    }

    eprintln!();
    eprintln!("Where they disagree, the DP planner's transfer-cost awareness");
    eprintln!("flipped the answer relative to the per-op winner.");
    Ok(())
}

struct AnchorBuild {
    label: String,
    outputs: Vec<LazyTensor>,
}

#[derive(Debug)]
struct DiffReport {
    total_nodes: usize,
    disagreements: usize,
    breakdown: BTreeMap<(String, String), usize>,
}

fn compare_planners(
    label: &str,
    outputs: &[LazyTensor],
    table: &DispatchTable,
    profile: &fuel::judge::ProfileReport,
    bandwidth: &fuel::transfer_cost::BandwidthMatrix,
    available_backends: &[BackendId],
) -> DiffReport {
    let _ = label;
    let graph = outputs[0].graph_tensor().graph();
    let g = graph.borrow();

    let roots: Vec<NodeId> = outputs.iter().map(|t| t.graph_tensor().id()).collect();

    let placements_recommend = recommend_placement(
        &g, table, Criterion::Fastest, DeviceLocation::Cpu,
    );

    let placements_dp = dp_plan(
        &g, &roots, profile, bandwidth, available_backends, DeviceLocation::Cpu,
    );

    let mut breakdown: BTreeMap<(String, String), usize> = BTreeMap::new();
    let mut disagreements = 0;
    let mut total = 0;
    // Iterate the union of node IDs from both placements (DP only
    // visits roots' transitive ancestors; recommend_placement visits
    // every node). Use the recommend keys as the canonical set —
    // they cover the whole graph.
    for (id, &rec_loc) in &placements_recommend {
        total += 1;
        let dp_loc = placements_dp.get(id).copied().unwrap_or(rec_loc);
        if dp_loc != rec_loc {
            disagreements += 1;
            let op_short = op_short_name(&g.node(*id).op).to_string();
            let key = (op_short, format!("{rec_loc:?} → {dp_loc:?}"));
            *breakdown.entry(key).or_default() += 1;
        }
    }
    DiffReport { total_nodes: total, disagreements, breakdown }
}

fn op_short_name(op: &Op) -> &'static str {
    match op {
        Op::Const(_)        => "Const",
        Op::Add             => "Add",
        Op::Sub             => "Sub",
        Op::Mul             => "Mul",
        Op::Div             => "Div",
        Op::MatMul          => "MatMul",
        Op::Conv2D{..}      => "Conv2D",
        Op::Permute(_)      => "Permute",
        Op::Reshape(_)      => "Reshape",
        Op::BroadcastTo(_)  => "BroadcastTo",
        Op::Concat{..}      => "Concat",
        Op::Slice{..}       => "Slice",
        Op::Silu            => "Silu",
        Op::Gelu            => "Gelu",
        Op::Relu            => "Relu",
        Op::Sigmoid         => "Sigmoid",
        Op::SoftmaxLastDim  => "Softmax",
        Op::LayerNormLastDim{..} => "LayerNorm",
        Op::RmsNormLastDim{..}   => "RmsNorm",
        Op::Rope            => "Rope",
        _ => "Other",
    }
}

fn build_anchors() -> Vec<AnchorBuild> {
    let mut out = Vec::new();
    out.push(build_bert());
    out.push(build_clip());
    out.push(build_qwen2_moe());
    out.push(build_whisper_decoder());
    out.push(build_convnext());
    out.push(build_yolov8());
    // SD VAE adds Conv2D-heavy graphs but its forward is more
    // involved (full decoder); skip for v1.

    // Stress configuration: BERT at production hidden_size to get
    // matmul shapes large enough to cross the CPU↔GPU dispatch
    // crossover. This is where DP and recommend_placement can
    // diverge on real workloads — the planners agree by accident
    // when every op falls below the CPU-wins threshold.
    out.push(build_bert_stress());
    out
}

fn build_bert_stress() -> AnchorBuild {
    use fuel::lazy_bert::{BertConfig, BertLayerWeights, BertModel, BertWeights};
    // BERT-large dimensions with long sequence, sized so the FFN
    // matmul output (seq * intermediate = 256 * 8192 = 2M elements)
    // crosses the dispatch table's CPU↔GPU crossover (~1M
    // elements / size_class 20) and gets routed to CUDA. With
    // smaller dims every op stays in the CPU-wins regime and the
    // two planners agree by accident.
    let cfg = BertConfig {
        vocab_size: 100, hidden_size: 2048, num_hidden_layers: 2,
        num_attention_heads: 16, intermediate_size: 8192,
        max_position_embeddings: 256, type_vocab_size: 2, layer_norm_eps: 1e-12,
    };
    let h = cfg.hidden_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = BertWeights {
        word_embeddings: z(cfg.vocab_size * h),
        position_embeddings: z(cfg.max_position_embeddings * h),
        token_type_embeddings: z(cfg.type_vocab_size * h),
        emb_ln_gamma: o(h), emb_ln_beta: z(h),
        layers: (0..cfg.num_hidden_layers).map(|_| BertLayerWeights {
            attn_q_w: z(h*h), attn_q_b: z(h),
            attn_k_w: z(h*h), attn_k_b: z(h),
            attn_v_w: z(h*h), attn_v_b: z(h),
            attn_out_w: z(h*h), attn_out_b: z(h),
            attn_ln_gamma: o(h), attn_ln_beta: z(h),
            ffn_in_w: z(h*cfg.intermediate_size), ffn_in_b: z(cfg.intermediate_size),
            ffn_out_w: z(cfg.intermediate_size*h), ffn_out_b: z(h),
            ffn_ln_gamma: o(h), ffn_ln_beta: z(h),
        }).collect(),
    };
    let model = BertModel { config: cfg, weights };
    let ids: Vec<u32> = (0..256).collect();
    AnchorBuild { label: "BERT (stress)".into(), outputs: vec![model.forward(&ids)] }
}

fn build_bert() -> AnchorBuild {
    use fuel::lazy_bert::{BertConfig, BertLayerWeights, BertModel, BertWeights};
    let cfg = BertConfig {
        vocab_size: 100, hidden_size: 32, num_hidden_layers: 2,
        num_attention_heads: 4, intermediate_size: 64,
        max_position_embeddings: 16, type_vocab_size: 2, layer_norm_eps: 1e-12,
    };
    let h = cfg.hidden_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = BertWeights {
        word_embeddings: z(cfg.vocab_size * h),
        position_embeddings: z(cfg.max_position_embeddings * h),
        token_type_embeddings: z(cfg.type_vocab_size * h),
        emb_ln_gamma: o(h), emb_ln_beta: z(h),
        layers: (0..cfg.num_hidden_layers).map(|_| BertLayerWeights {
            attn_q_w: z(h*h), attn_q_b: z(h),
            attn_k_w: z(h*h), attn_k_b: z(h),
            attn_v_w: z(h*h), attn_v_b: z(h),
            attn_out_w: z(h*h), attn_out_b: z(h),
            attn_ln_gamma: o(h), attn_ln_beta: z(h),
            ffn_in_w: z(h*cfg.intermediate_size), ffn_in_b: z(cfg.intermediate_size),
            ffn_out_w: z(cfg.intermediate_size*h), ffn_out_b: z(h),
            ffn_ln_gamma: o(h), ffn_ln_beta: z(h),
        }).collect(),
    };
    let model = BertModel { config: cfg, weights };
    let ids: Vec<u32> = (0..8).collect();
    let out = model.forward(&ids);
    AnchorBuild { label: "BERT".into(), outputs: vec![out] }
}

fn build_clip() -> AnchorBuild {
    use fuel::lazy_sd_text_encoder::{ClipLayerWeights, ClipTextConfig, ClipTextWeights, SdTextEncoder};
    let cfg = ClipTextConfig {
        vocab_size: 100, hidden_size: 16, num_hidden_layers: 2,
        num_attention_heads: 4, intermediate_size: 32, max_position_embeddings: 8,
        layer_norm_eps: 1e-5, bos_token_id: 0, eos_token_id: 2, pad_token_id: 1,
    };
    let h = cfg.hidden_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = ClipTextWeights {
        token_embedding: z(cfg.vocab_size * h),
        position_embedding: z(cfg.max_position_embeddings * h),
        layers: (0..cfg.num_hidden_layers).map(|_| ClipLayerWeights {
            ln1_g: o(h), ln1_b: z(h),
            q_w: z(h*h), q_b: z(h),
            k_w: z(h*h), k_b: z(h),
            v_w: z(h*h), v_b: z(h),
            out_w: z(h*h), out_b: z(h),
            ln2_g: o(h), ln2_b: z(h),
            fc1_w: z(h*cfg.intermediate_size), fc1_b: z(cfg.intermediate_size),
            fc2_w: z(cfg.intermediate_size*h), fc2_b: z(h),
        }).collect(),
        final_ln_g: o(h), final_ln_b: z(h),
    };
    let model = SdTextEncoder { config: cfg.clone(), weights };
    let tokens: Vec<u32> = (0..cfg.max_position_embeddings as u32).collect();
    AnchorBuild { label: "SD CLIP".into(), outputs: vec![model.forward(&tokens)] }
}

fn build_qwen2_moe() -> AnchorBuild {
    use fuel::lazy_qwen2_moe::{ExpertWeights, Qwen2MoeConfig, Qwen2MoeLayerWeights, Qwen2MoeModel, Qwen2MoeWeights};
    let cfg = Qwen2MoeConfig {
        vocab_size: 32, hidden_size: 8, num_hidden_layers: 1,
        num_attention_heads: 2, num_key_value_heads: 2,
        moe_intermediate_size: 12, shared_expert_intermediate_size: 16,
        num_experts: 3, num_experts_per_tok: 2,
        max_position_embeddings: 32, rope_theta: 10_000.0,
        rms_norm_eps: 1e-6, norm_topk_prob: false,
    };
    let h = cfg.hidden_size;
    let mi = cfg.moe_intermediate_size;
    let si = cfg.shared_expert_intermediate_size;
    let z = |n: usize| Arc::<[f32]>::from(vec![0.0_f32; n]);
    let o = |n: usize| Arc::<[f32]>::from(vec![1.0_f32; n]);
    let weights = Qwen2MoeWeights {
        token_embedding: z(cfg.vocab_size * h),
        layers: vec![Qwen2MoeLayerWeights {
            input_ln: o(h),
            q_w: z(h*h), q_b: z(h),
            k_w: z(h*h), k_b: z(h),
            v_w: z(h*h), v_b: z(h),
            o_w: z(h*h),
            post_attn_ln: o(h),
            gate_w: z(h * cfg.num_experts),
            experts: (0..cfg.num_experts).map(|_| ExpertWeights {
                gate_w: z(h*mi), up_w: z(h*mi), down_w: z(mi*h),
            }).collect(),
            shared_gate_w: z(h*si), shared_up_w: z(h*si), shared_down_w: z(si*h),
            shared_expert_gate_w: z(h),
        }],
        final_ln: o(h),
        lm_head: z(h * cfg.vocab_size),
    };
    let model = Qwen2MoeModel { config: cfg, weights };
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    AnchorBuild { label: "Qwen2-MoE".into(), outputs: vec![model.forward(&tokens)] }
}

fn build_whisper_decoder() -> AnchorBuild {
    use fuel::lazy_whisper::{tiny_cfg, zero_weights, WhisperModel};
    let cfg = tiny_cfg();
    let weights = zero_weights(&cfg);
    let model = WhisperModel { config: cfg.clone(), weights };
    let mel = vec![0.0_f32; cfg.num_mel_bins * 32];
    let enc = model.forward_encoder(&mel, 32);
    let tokens: Vec<u32> = vec![1, 2, 3, 4];
    let logits = model.forward_decoder(&tokens, &enc);
    AnchorBuild { label: "Whisper decoder".into(), outputs: vec![logits] }
}

fn build_convnext() -> AnchorBuild {
    use fuel::lazy_convnext::{tiny_cfg, zero_weights, ConvNextModel};
    let cfg = tiny_cfg();
    let weights = zero_weights(&cfg);
    let model = ConvNextModel { weights, config: cfg.clone() };
    let image = vec![0.0_f32; cfg.in_channels * cfg.image_size * cfg.image_size];
    AnchorBuild { label: "ConvNeXt".into(), outputs: vec![model.forward(&image)] }
}

fn build_yolov8() -> AnchorBuild {
    use fuel::lazy_yolov8::{YoloV8Config, YoloV8Model, YoloV8Weights};
    let mut cfg = YoloV8Config::v8n();
    cfg.image_size = 64;
    let weights = YoloV8Weights::zeros(&cfg);
    let model = YoloV8Model { config: cfg.clone(), weights };
    let image = vec![0.0_f32; 3 * cfg.image_size * cfg.image_size];
    let raw = model.forward(&image);
    AnchorBuild { label: "YOLOv8".into(), outputs: vec![raw.cls_logits, raw.reg_dists] }
}

#[allow(dead_code)]  // placeholder for future per-op breakdown printing
fn print_breakdown(b: &BTreeMap<(String, String), usize>) {
    let mut by_op: HashMap<&str, Vec<(&str, usize)>> = HashMap::new();
    for ((op, transition), count) in b {
        by_op.entry(op.as_str()).or_default().push((transition.as_str(), *count));
    }
    for (op, transitions) in &by_op {
        for (transition, count) in transitions {
            println!("    {op:<10} {transition}  ×{count}");
        }
    }
}

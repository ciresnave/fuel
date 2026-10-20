// SPDX-License-Identifier: MIT OR Apache-2.0
//! Proxy measurement for model-free n-gram drafting (ROADMAP item 17, H24).
//!
//! usage: ngram_draft_probe <tokenizer.json> <file A (the prompt)> <file C (unrelated text)>
//!
//! The "target model" is replaced by ground truth: the output B is assumed to be
//! reproduced token for token, so a draft token is accepted exactly when it
//! equals the next token of B. That makes every number here an **upper bound on
//! acceptance for that scenario** (a real model sometimes diverges from the text
//! it is copying), and it says nothing about quality or about a model step's
//! cost. What it does measure is how often the lookup finds the right
//! continuation on real tokens, and what the lookup costs.
//!
//! Scenarios, all with the history starting as the tokens of file A:
//!   edit-N%  B = A with N% of its lines deleted, replaced, or preceded by an
//!            inserted line taken from file C (deterministic, seeded)
//!   control  B = file C, which shares nothing with A (no gain expected)

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use fuel_inference::ngram_draft::NgramDrafter;
use tokenizers::Tokenizer;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

fn edit(a: &str, c_lines: &[&str], pct: f64, seed: u64) -> String {
    let mut rng = Lcg(seed);
    let mut out = String::new();
    for line in a.lines() {
        if rng.unit() < pct {
            match rng.next() % 3 {
                0 => continue, // delete
                1 => out.push_str(&format!(
                    "{}\n",
                    c_lines[(rng.next() as usize) % c_lines.len()]
                )),
                _ => {
                    out.push_str(&format!(
                        "{}\n",
                        c_lines[(rng.next() as usize) % c_lines.len()]
                    ));
                    out.push_str(line);
                    out.push('\n');
                }
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

struct Outcome {
    tokens: usize,
    steps: usize,
    steps_with_draft: usize,
    drafted: usize,
    accepted: usize,
    draft_ns: u128,
}

fn simulate(prompt: &[u32], b: &[u32], drafter: &NgramDrafter) -> Outcome {
    let mut history = prompt.to_vec();
    let mut o = Outcome {
        tokens: b.len(),
        steps: 0,
        steps_with_draft: 0,
        drafted: 0,
        accepted: 0,
        draft_ns: 0,
    };
    let mut t = 0;
    while t < b.len() {
        let t0 = Instant::now();
        let draft = drafter.draft(&history);
        o.draft_ns += t0.elapsed().as_nanos();
        let mut k = 0;
        while k < draft.len() && t + k < b.len() && draft[k] == b[t + k] {
            k += 1;
        }
        if !draft.is_empty() {
            o.steps_with_draft += 1;
            o.drafted += draft.len();
            o.accepted += k;
        }
        // The verify pass emits the accepted tokens plus one token of the
        // target's own (the bonus / correction), capped at the end of B.
        let emitted = (k + 1).min(b.len() - t);
        history.extend_from_slice(&b[t..t + emitted]);
        t += emitted;
        o.steps += 1;
    }
    o
}

/// Drafter settings swept: (max n-gram, min n-gram, max draft length).
const CONFIGS: [(usize, usize, usize); 5] =
    [(4, 2, 3), (4, 2, 5), (4, 2, 10), (3, 2, 5), (2, 2, 5)];

#[derive(Parser)]
#[command(about = "Proxy measurement for n-gram drafting (ground-truth target)")]
struct Cli {
    /// A Hugging Face tokenizer.json
    tokenizer: PathBuf,
    /// File A: the prompt, and the base text the edited outputs are made from
    prompt_file: PathBuf,
    /// File C: unrelated text used for the inserted lines and the control
    other_file: PathBuf,
}

fn print_header() {
    println!(
        "{:>10} {:>14} {:>9} {:>9} {:>12} {:>10} {:>11}",
        "scenario",
        "ngram(max,min,draft)",
        "tok/step",
        "draft%",
        "accept/prop",
        "us/draft",
        "B tokens"
    );
}

fn print_row(name: &str, cfg: (usize, usize, usize), o: &Outcome) {
    let ratio = |num: usize, den: usize| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    println!(
        "{:>10} {:>14} {:>9.3} {:>8.1}% {:>12.3} {:>10.1} {:>11}",
        name,
        format!("({},{},{})", cfg.0, cfg.1, cfg.2),
        ratio(o.tokens, o.steps),
        100.0 * ratio(o.steps_with_draft, o.steps),
        ratio(o.accepted, o.drafted),
        o.draft_ns as f64 / o.steps as f64 / 1e3,
        o.tokens
    );
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let tok = Tokenizer::from_file(&cli.tokenizer).map_err(anyhow::Error::msg)?;
    let a = std::fs::read_to_string(&cli.prompt_file)?;
    let c = std::fs::read_to_string(&cli.other_file)?;
    let enc = |s: &str| -> anyhow::Result<Vec<u32>> {
        Ok(tok
            .encode(s, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec())
    };
    let a_tokens = enc(&a)?;
    let c_lines: Vec<&str> = c.lines().filter(|l| !l.trim().is_empty()).collect();
    println!(
        "A: {} bytes, {} tokens; C: {} lines",
        a.len(),
        a_tokens.len(),
        c_lines.len()
    );
    print_header();
    let scenarios = [
        ("edit-2%", edit(&a, &c_lines, 0.02, 11)),
        ("edit-10%", edit(&a, &c_lines, 0.10, 12)),
        ("edit-30%", edit(&a, &c_lines, 0.30, 13)),
        ("control", c.clone()),
    ];
    for (name, b_text) in scenarios {
        let b = enc(&b_text)?;
        for cfg in CONFIGS {
            let drafter = NgramDrafter::new(cfg.0, cfg.1, cfg.2);
            print_row(name, cfg, &simulate(&a_tokens, &b, &drafter));
        }
    }
    Ok(())
}

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::{Parser, ValueEnum};

use fuel::lazy_marian::{MarianConfig, MarianModel, MarianWeights};
use fuel_examples::token_output_stream::TokenOutputStream;

use tokenizers::Tokenizer;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum Which {
    Base,
    Big,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum LanguagePair {
    #[value(name = "fr-en")]
    FrEn,
    #[value(name = "en-zh")]
    EnZh,
    #[value(name = "en-hi")]
    EnHi,
    #[value(name = "en-es")]
    EnEs,
    #[value(name = "en-fr")]
    EnFr,
    #[value(name = "en-ru")]
    EnRu,
}

// TODO: Maybe add support for the conditional prompt.
#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    tokenizer_dec: Option<String>,

    /// Choose the variant of the model to run.
    #[arg(long, default_value = "big")]
    which: Which,

    // Choose which language pair to use
    #[arg(long, default_value = "fr-en")]
    language_pair: LanguagePair,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized version of the model.
    #[arg(long)]
    quantized: bool,

    /// Text to be translated
    #[arg(long)]
    text: String,
}

/// Special token IDs not currently mirrored on the lazy `MarianConfig`. The
/// lazy port ships only the `opus_mt_tc_big_fr_en` preset; other presets are
/// kept here for reference but will be rejected at model-construction time.
struct MarianSpecials {
    decoder_start_token_id: u32,
    eos_token_id: u32,
    forced_eos_token_id: u32,
}

fn config_and_specials(
    which: Which,
    pair: LanguagePair,
) -> anyhow::Result<(MarianConfig, MarianSpecials)> {
    match (which, pair) {
        (Which::Big, LanguagePair::FrEn) => Ok((
            MarianConfig::opus_mt_tc_big_fr_en(),
            MarianSpecials {
                decoder_start_token_id: 53016,
                eos_token_id: 43311,
                forced_eos_token_id: 43311,
            },
        )),
        (Which::Big, lp) => {
            anyhow::bail!("big is not supported for language pair {lp:?}")
        }
        (Which::Base, lp) => {
            anyhow::bail!(
                "lazy_marian only ships the opus_mt_tc_big_fr_en preset; \
                 the base variant for {lp:?} is not yet ported"
            )
        }
    }
}

pub fn main() -> anyhow::Result<()> {
    use hf_hub::api::sync::Api;
    let args = Args::parse();

    if args.quantized {
        anyhow::bail!(
            "--quantized is not supported by the lazy port (no lazy_quantized_marian module)"
        );
    }

    let (config, specials) = config_and_specials(args.which, args.language_pair)?;
    let tokenizer_default_repo = match args.language_pair {
        LanguagePair::FrEn => "lmz/fuel-marian",
        LanguagePair::EnZh
        | LanguagePair::EnHi
        | LanguagePair::EnEs
        | LanguagePair::EnFr
        | LanguagePair::EnRu => "KeighBee/fuel-marian",
    };
    let tokenizer = {
        let tokenizer = match args.tokenizer {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let filename = match (args.which, args.language_pair) {
                    (Which::Base, LanguagePair::FrEn) => "tokenizer-marian-base-fr.json",
                    (Which::Big, LanguagePair::FrEn) => "tokenizer-marian-fr.json",
                    (Which::Base, LanguagePair::EnZh) => "tokenizer-marian-base-en-zh-en.json",
                    (Which::Base, LanguagePair::EnHi) => "tokenizer-marian-base-en-hi-en.json",
                    (Which::Base, LanguagePair::EnEs) => "tokenizer-marian-base-en-es-en.json",
                    (Which::Base, LanguagePair::EnFr) => "tokenizer-marian-base-en-fr-en.json",
                    (Which::Base, LanguagePair::EnRu) => "tokenizer-marian-base-en-ru-en.json",
                    (Which::Big, lp) => {
                        anyhow::bail!("big is not supported for language pair {lp:?}")
                    }
                };
                Api::new()?
                    .model(tokenizer_default_repo.to_string())
                    .get(filename)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };

    let tokenizer_dec = {
        let tokenizer = match args.tokenizer_dec {
            Some(tokenizer) => std::path::PathBuf::from(tokenizer),
            None => {
                let filename = match (args.which, args.language_pair) {
                    (Which::Base, LanguagePair::FrEn) => "tokenizer-marian-base-en.json",
                    (Which::Big, LanguagePair::FrEn) => "tokenizer-marian-en.json",
                    (Which::Base, LanguagePair::EnZh) => "tokenizer-marian-base-en-zh-zh.json",
                    (Which::Base, LanguagePair::EnHi) => "tokenizer-marian-base-en-hi-hi.json",
                    (Which::Base, LanguagePair::EnEs) => "tokenizer-marian-base-en-es-es.json",
                    (Which::Base, LanguagePair::EnFr) => "tokenizer-marian-base-en-fr-fr.json",
                    (Which::Base, LanguagePair::EnRu) => "tokenizer-marian-base-en-ru-ru.json",
                    (Which::Big, lp) => {
                        anyhow::bail!("big is not supported for language pair {lp:?}")
                    }
                };
                Api::new()?
                    .model(tokenizer_default_repo.to_string())
                    .get(filename)?
            }
        };
        Tokenizer::from_file(&tokenizer).map_err(E::msg)?
    };
    let mut tokenizer_dec = TokenOutputStream::new(tokenizer_dec);

    let model_path = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => {
            let api = Api::new()?;
            let api = match (args.which, args.language_pair) {
                (Which::Base, LanguagePair::FrEn) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-fr-en".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/4".to_string(),
                )),
                (Which::Big, LanguagePair::FrEn) => {
                    api.model("Helsinki-NLP/opus-mt-tc-big-fr-en".to_string())
                }
                (Which::Base, LanguagePair::EnZh) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-zh".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/13".to_string(),
                )),
                (Which::Base, LanguagePair::EnHi) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-hi".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/3".to_string(),
                )),
                (Which::Base, LanguagePair::EnEs) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-es".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/4".to_string(),
                )),
                (Which::Base, LanguagePair::EnFr) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-fr".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/9".to_string(),
                )),
                (Which::Base, LanguagePair::EnRu) => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-ru".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/7".to_string(),
                )),
                (Which::Big, lp) => {
                    anyhow::bail!("big is not supported for language pair {lp:?}")
                }
            };
            api.get("model.safetensors")?
        }
    };

    let filenames = vec![model_path];
    let st = unsafe { fuel::safetensors::MmapedSafetensors::multi(&filenames) }
        .map_err(|e| E::msg(format!("mmap safetensors: {e}")))?;
    let weights = MarianWeights::load_from_mmapped(&st, &config)
        .map_err(|e| E::msg(format!("load marian weights: {e}")))?;
    let model = MarianModel {
        config: config.clone(),
        weights,
    };

    // Encode the source sentence once. The lazy port's
    // `forward_encoder` returns the encoder output as a `LazyTensor`
    // which can be reused as the cross-attention K/V source on every
    // decode step.
    let src_tokens: Vec<u32> = {
        let mut tokens = tokenizer
            .encode(args.text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        tokens.push(specials.eos_token_id);
        tokens
    };
    let encoder_xs = model
        .forward_encoder(&src_tokens)
        .map_err(|e| E::msg(format!("encoder forward: {e}")))?;

    let target_vocab = config.target_vocab_size();
    let mut token_ids = vec![specials.decoder_start_token_id];
    for _index in 0..1000 {
        // The lazy decoder is non-cached: it takes the full target
        // prefix every step and produces logits of shape
        // `(1, tgt_len, target_vocab)`.
        let logits = model
            .forward_decoder(&token_ids, &encoder_xs)
            .map_err(|e| E::msg(format!("decoder forward: {e}")))?;
        let logits_data = logits.realize_f32();
        let tgt_len = token_ids.len();
        let last_off = (tgt_len - 1) * target_vocab;
        let last_logits = &logits_data[last_off..last_off + target_vocab];

        // Greedy argmax — matches `LogitsProcessor::new(seed, None, None)`
        // in the eager binary.
        let mut best_i = 0usize;
        let mut best = last_logits[0];
        for (i, &v) in last_logits.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                best_i = i;
            }
        }
        let token = best_i as u32;
        token_ids.push(token);
        if let Some(t) = tokenizer_dec.next_token(token)? {
            use std::io::Write;
            print!("{t}");
            std::io::stdout().flush()?;
        }
        if token == specials.eos_token_id || token == specials.forced_eos_token_id {
            break;
        }
    }
    if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    println!();
    Ok(())
}

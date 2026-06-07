//! Whisper language-detection helper — lazy port.
//!
//! Runs one encoder pass + a single-token decoder pass on the supplied
//! mel slice, then softmaxes the first-row logits over the 99 language-
//! token IDs to pick the most likely language. Mirrors the OpenAI
//! reference's `detect_language`.

use crate::{token_id, Model, SOT_TOKEN, N_FRAMES};
use anyhow::{Error as E, Result};
use tokenizers::Tokenizer;

const LANGUAGES: [(&str, &str); 99] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
];

/// Returns the token id for the most-likely language given a mel-window.
///
/// `mel` is a flat row-major `(num_mel_bins, mel_time)` spectrogram.
/// The slice is truncated to `max_source_positions` mel frames before
/// being run through the encoder.
pub fn detect_language(
    model: &Model,
    tokenizer: &Tokenizer,
    mel: &[f32],
    mel_time: usize,
) -> Result<u32> {
    let cfg = model.config();
    let num_mel_bins = cfg.num_mel_bins;
    let max_mel_time = usize::min(
        mel_time,
        usize::min(2 * cfg.max_source_positions, N_FRAMES),
    );
    // Even mel_time required (stride-2 conv).
    let mel_time_used = max_mel_time - (max_mel_time % 2);
    if mel_time_used == 0 {
        anyhow::bail!("detect_language: empty mel after trimming to even length");
    }
    // Narrow the time axis to `mel_time_used` columns. The mel buffer
    // is row-major `(num_mel_bins, mel_time)` — copy the first
    // `mel_time_used` columns out of each row.
    let mel_view = crate::narrow_time_axis(mel, num_mel_bins, mel_time, 0, mel_time_used);

    let sot_token = token_id(tokenizer, SOT_TOKEN)?;
    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| token_id(tokenizer, &format!("<|{t}|>")))
        .collect::<fuel::Result<Vec<u32>>>()
        .map_err(E::msg)?;

    let audio_features = model
        .encoder_forward(&mel_view, mel_time_used)
        .map_err(|e| E::msg(format!("encoder: {e}")))?;
    // forward_decoder returns logits of shape [1, seq, vocab]. With a
    // single SOT token, that's a [1, 1, vocab] tensor.
    let tokens = [sot_token];
    let logits = model
        .decoder_forward(&tokens, &audio_features)
        .map_err(|e| E::msg(format!("decoder: {e}")))?;
    let flat = logits.realize_f32();
    let vocab = cfg.vocab_size;
    if flat.len() != vocab {
        anyhow::bail!(
            "detect_language: unexpected logits length {} expected {vocab}",
            flat.len()
        );
    }
    // Gather logits at the language token ids and softmax across them.
    let gathered: Vec<f32> = language_token_ids
        .iter()
        .map(|&id| flat[id as usize])
        .collect();
    let row_max = gathered.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = gathered
        .iter()
        .map(|v| ((*v - row_max) as f64).exp() as f32)
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for p in exps.iter_mut() {
            *p /= sum;
        }
    }
    let mut probs = LANGUAGES.iter().zip(exps.iter()).collect::<Vec<_>>();
    probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs.iter().take(5) {
        println!("{language}: {p}")
    }
    let language = token_id(tokenizer, &format!("<|{}|>", probs[0].0 .0)).map_err(E::msg)?;
    Ok(language)
}

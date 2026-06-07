use anyhow::Result;
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

/// Detect the spoken language using the encoder + a single decoder
/// step. Operates on a flat `[n_mels, total_frames]` mel buffer; clips
/// the time axis to `max_source_positions` (the encoder's mel_time
/// must be even for the stride-2 conv stem).
pub fn detect_language(
    model: &super::Model,
    tokenizer: &Tokenizer,
    mel: &[f32],
    n_mels: usize,
    total_frames: usize,
) -> Result<u32> {
    let max_src = model.config().max_source_positions;
    // mel_time = 2 * encoder_input_seq; the encoder downsamples by 2.
    // Use min(total_frames, max_src*2) and round down to even.
    let cap = 2 * max_src;
    let mut seg_size = usize::min(total_frames, cap);
    if !seg_size.is_multiple_of(2) {
        seg_size -= 1;
    }
    // Extract the prefix segment from the row-major mel buffer.
    let mut mel_segment = Vec::with_capacity(n_mels * seg_size);
    for m in 0..n_mels {
        let row = &mel[m * total_frames..m * total_frames + seg_size];
        mel_segment.extend_from_slice(row);
    }

    let encoder_out = model.encoder_forward(&mel_segment, seg_size)?;
    let language_token_ids = LANGUAGES
        .iter()
        .map(|(t, _)| crate::token_id(tokenizer, &format!("<|{t}|>")))
        .collect::<fuel::Result<Vec<_>>>()
        .map_err(|e| anyhow::Error::msg(format!("language token id: {e}")))?;
    let sot_token = crate::token_id(tokenizer, crate::SOT_TOKEN)
        .map_err(|e| anyhow::Error::msg(format!("sot token: {e}")))?;
    let tokens = vec![sot_token];
    let logits_flat = model.decoder_logits(&tokens, &encoder_out, seg_size)?;
    let vocab = model.config().vocab_size;
    // First-row (the only row) last-token logits — `[vocab]`.
    let last_off = (tokens.len() - 1) * vocab;
    let logits = &logits_flat[last_off..last_off + vocab];
    // Softmax over the picked language token ids only.
    let picked: Vec<f32> = language_token_ids.iter().map(|&i| logits[i as usize]).collect();
    let probs = softmax_vec(&picked);
    let mut probs_lang: Vec<((&str, &str), f32)> =
        LANGUAGES.iter().copied().zip(probs.into_iter()).collect();
    probs_lang.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    for ((_, language), p) in probs_lang.iter().take(5) {
        println!("{language}: {p}")
    }
    let language = crate::token_id(tokenizer, &format!("<|{}|>", probs_lang[0].0 .0))
        .map_err(|e| anyhow::Error::msg(format!("language token id: {e}")))?;
    Ok(language)
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    let s = if s == 0.0 { 1.0 } else { s };
    exps.into_iter().map(|v| v / s).collect()
}

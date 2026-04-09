pub mod audio;
pub mod model;
pub mod voxtral_llama;

pub use audio::extract_features;
pub use model::{
    VoxtralCache, VoxtralConfig, VoxtralEncoder, VoxtralEncoderConfig,
    VoxtralForConditionalGeneration, VoxtralGenerationConfig, VoxtralMultiModalProjector,
};
pub use voxtral_llama::{VoxtralLlama, VoxtralLlamaCache, VoxtralLlamaConfig};

/// FFT window size for Voxtral audio feature extraction.
pub const N_FFT: usize = 400;
/// Hop length (stride) for Voxtral short-time Fourier transform.
pub const HOP_LENGTH: usize = 160;
/// Number of mel filter banks for Voxtral audio feature extraction.
pub const N_MELS: usize = 128;

// SD 1.5 VAE decoder runner. Second component of Phase 6a anchor #6.
//
// USAGE
//
//     cargo run --release --bin sd-vae-lazy
//     cargo run --release --bin sd-vae-lazy -- [REPO_ID] [LATENT_SIZE]
//
// Defaults:
//     REPO_ID     = stable-diffusion-v1-5/stable-diffusion-v1-5
//     LATENT_SIZE = 8   (→ 8×8×8 = 64×64 output; fast)
//
// LATENT_SIZE scales as square:
//     8  → 64×64 output   (seconds, smoke test)
//     16 → 128×128 output (tens of seconds)
//     32 → 256×256 output (minute+)
//     64 → 512×512 output (SD's native resolution; several minutes)
//
// Runs a zero latent through the decoder and prints summary stats of
// the output image tensor. With zero input + real weights, the output
// is a "mean image" — whatever the VAE's post-quant + decoder bias
// chain produces in the absence of conditioning. A real pipeline
// would sample a Gaussian noise latent or take a UNet-denoised
// latent; this binary's purpose is to confirm the decoder's forward
// pass wires up correctly against real weights.

use fuel::lazy_sd_vae::SdVaeDecoder;
use std::io::Write;
use std::time::Instant;

const DEFAULT_REPO: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";
const DEFAULT_LATENT_SIZE: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let repo_id = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| DEFAULT_REPO.to_string());
    let lat_size: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LATENT_SIZE);

    eprintln!("=== fuel sd-vae-lazy ===");
    eprintln!("Repo:        {repo_id}");
    eprintln!("Latent size: {lat_size}×{lat_size}  (output {} × {})", lat_size * 8, lat_size * 8);
    eprintln!();

    eprint!("Downloading + loading VAE weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let decoder = SdVaeDecoder::from_hub(&repo_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  dims={:?}  norm_groups={}  layers_per_block={}",
        decoder.config.dims,
        decoder.config.norm_num_groups,
        decoder.config.layers_per_block,
    );
    eprintln!();

    let lc = decoder.config.latent_channels;
    let oc = decoder.config.out_channels;
    let latent = vec![0.0_f32; lc * lat_size * lat_size];
    eprintln!("Running decoder...");
    let t0 = Instant::now();
    let out_t = decoder.decode(&latent, lat_size, lat_size);
    let out = out_t.realize_f32();
    let elapsed = t0.elapsed();
    eprintln!("Forward done in {:.2?}", elapsed);
    eprintln!();

    let h_out = lat_size * 8;
    let w_out = lat_size * 8;
    assert_eq!(out.len(), oc * h_out * w_out);

    // Per-channel min/mean/max stats over the output.
    for c in 0..oc {
        let start = c * h_out * w_out;
        let ch = &out[start..start + h_out * w_out];
        let min = ch.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = ch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = ch.iter().sum::<f32>() / ch.len() as f32;
        let var: f32 = ch.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / ch.len() as f32;
        println!(
            "channel {c}:  min={:+.4}  mean={:+.4}  max={:+.4}  std={:.4}",
            min, mean, max, var.sqrt()
        );
    }
    let finite = out.iter().all(|v| v.is_finite());
    println!();
    println!("All finite: {finite}");
    Ok(())
}

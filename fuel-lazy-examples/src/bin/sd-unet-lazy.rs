// SD 1.5 UNet runner. Third + final component of Phase 6a anchor #6.
//
// USAGE
//
//     cargo run --release --bin sd-unet-lazy
//     cargo run --release --bin sd-unet-lazy -- [REPO] [LAT_H] [LAT_W]
//
// Defaults: REPO=stable-diffusion-v1-5/stable-diffusion-v1-5  LAT=8×8.
//
// SCOPE
//
// Runs a single denoising step: (zero latent, timestep=0, zero text
// embedding) → noise prediction. Validates the architectural wiring
// end-to-end against real weights. A real diffusion loop would iterate
// this ~20-50 times with a scheduler, then pass the final latent to the
// VAE decoder. That loop isn't implemented here — scheduler math lives
// outside the network.
//
// At LAT=8 the forward pass runs at latent resolution 8×8 (= 64×64
// image), traverses to 1×1 at the deepest stage, and back up to 8×8.
// Much smaller than SD's native 64×64 latent (= 512×512 image) but
// exercises every layer. Expect several minutes on CPU.

use fuel::lazy_sd_unet::SdUnet;
use std::io::Write;
use std::time::Instant;

const DEFAULT_REPO: &str = "stable-diffusion-v1-5/stable-diffusion-v1-5";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let repo_id = args.get(1).cloned().unwrap_or_else(|| DEFAULT_REPO.to_string());
    let h_lat: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let w_lat: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    eprintln!("=== fuel sd-unet-lazy ===");
    eprintln!("Repo: {repo_id}");
    eprintln!("Latent: {h_lat}×{w_lat}");
    eprintln!();

    eprint!("Downloading + loading UNet weights... ");
    std::io::stderr().flush().ok();
    let t0 = Instant::now();
    let unet = SdUnet::from_hub(&repo_id)?;
    eprintln!("done in {:.2?}", t0.elapsed());
    eprintln!(
        "  channels={:?}  time_dim={}  cross_dim={}",
        unet.config.block_out_channels,
        unet.config.time_embed_dim,
        unet.config.cross_attention_dim,
    );
    eprintln!();

    let lc = unet.config.in_channels;
    let latent = vec![0.0_f32; lc * h_lat * w_lat];
    let text = vec![0.0_f32; 1 * 77 * unet.config.cross_attention_dim];

    eprintln!("Running one denoising step...");
    let t0 = Instant::now();
    let out_t = unet.forward(&latent, 0.0, &text, h_lat, w_lat);
    let out = out_t.realize_f32();
    eprintln!("Forward done in {:.2?}", t0.elapsed());

    let expected = unet.config.out_channels * h_lat * w_lat;
    assert_eq!(out.len(), expected);
    let min = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    eprintln!();
    println!("Output noise prediction [1, {}, {h_lat}, {w_lat}]:",
        unet.config.out_channels);
    println!("  min={min:+.4}  mean={mean:+.4}  max={max:+.4}");
    println!("  all finite: {}", out.iter().all(|v| v.is_finite()));
    Ok(())
}

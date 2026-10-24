// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools that rewrite committed files. Not a test, not part of the
//! supported API. Run: `cargo run -p fuel-dispatch --features dev-tools --bin
//! fuel-dev-tools -- <tool>`.
//!
//! Tools: `seed-cpu`, `seed-cuda` (`--features cuda`, live device), `seed-vulkan`
//! (`--features vulkan`, live device), `seed-vulkan-accuracy` and `check-vulkan-accuracy`
//! (`--features vulkan`, live device), `seed-cuda-accuracy` and `check-cuda-accuracy`
//! (`--features cuda`, live device), `gap-228-flip`, `gap-228-report`.

use fuel_dispatch::dev_tools;

fn main() {
    let tool = std::env::args().nth(1).unwrap_or_default();
    match tool.as_str() {
        "seed-cpu" => dev_tools::seed_cpu_ledger(),
        #[cfg(feature = "cuda")]
        "seed-cuda" => dev_tools::seed_cuda_ledger(),
        #[cfg(feature = "cuda")]
        "seed-cuda-accuracy" => dev_tools::seed_cuda_accuracy(),
        #[cfg(feature = "cuda")]
        "check-cuda-accuracy" => dev_tools::check_cuda_accuracy(),
        #[cfg(feature = "vulkan")]
        "seed-vulkan" => dev_tools::seed_vulkan_ledger(),
        #[cfg(feature = "vulkan")]
        "seed-vulkan-accuracy" => dev_tools::seed_vulkan_matmul_accuracy(),
        #[cfg(feature = "vulkan")]
        "check-vulkan-accuracy" => dev_tools::check_vulkan_accuracy(),
        "gap-228-flip" => dev_tools::gap_228_flip(),
        "gap-228-report" => dev_tools::gap_228_sections_report(),
        other => {
            eprintln!(
                "unknown tool {other:?}; expected seed-cpu | seed-cuda | seed-vulkan | seed-vulkan-accuracy | check-vulkan-accuracy | seed-cuda-accuracy | check-cuda-accuracy | gap-228-flip | gap-228-report"
            );
            std::process::exit(2);
        }
    }
}

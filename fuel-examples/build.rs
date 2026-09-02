// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused)]
mod buildtime_downloader;
use buildtime_downloader::download_model;

struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "examples/custom-ops/kernels/*.cu",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
}];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::{Path, PathBuf};
        // Added: Get the safe output directory from the environment.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        for kdir in KERNEL_DIRS.iter() {
            // Changed: This now writes to a safe path inside $OUT_DIR.
            let safe_target = out_dir.join(
                Path::new(kdir.rust_target)
                    .file_name()
                    .expect("Failed to get filename from rust_target"),
            );

            let bindings = cudaforge::KernelBuilder::new()
                .source_glob(kdir.kernel_glob)
                .build_ptx()
                .expect("Failed to build ptx");
            bindings
                .write(safe_target)
                .expect("Failed to write ptx bindings");
        }
    }

    // Download config, tokenizer, and model files from hf at build time.
    // option_env! automatically detects changes in the env var and trigger rebuilds correctly.
    // Example value:
    // FUEL_BUILDTIME_MODEL_REVISION="sentence-transformers/all-MiniLM-L6-v2:c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if let Some(model_rev) = core::option_env!("FUEL_BUILDTIME_MODEL_REVISION") {
        buildtime_downloader::download_model(model_rev).expect("Model download failed!");
    } else if std::env::var_os("CARGO_FEATURE_BUILDTIME_DOWNLOAD").is_some() {
        // The `buildtime-download` feature (bert_single_file_binary) is enabled
        // but no revision was given, so the download never ran and the three
        // `FUEL_BUILDTIME_MODEL_*` env vars are unset. Fail with a clear,
        // actionable message here instead of the cryptic downstream
        // `env!("FUEL_BUILDTIME_MODEL_CONFIG") not defined`. The revision is a
        // required build-time input — no default is invented (there is no
        // sensible default model to download).
        println!(
            "cargo::error=the `buildtime-download` feature (the \
             bert_single_file_binary example) requires FUEL_BUILDTIME_MODEL_REVISION \
             to be set at build time. Example: FUEL_BUILDTIME_MODEL_REVISION=\
             \"sentence-transformers/all-MiniLM-L6-v2:<revision-sha>\" cargo build \
             --example bert_single_file_binary --release --features buildtime-download"
        );
    }
}

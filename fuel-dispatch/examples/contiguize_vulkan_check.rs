// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-GPU check of the native Vulkan contiguize (`vulkan_dispatch::contiguize_vulkan`)
//! against the CPU reference (`fuel_cpu_backend::byte_kernels::contiguize_cpu`).
//!
//! For element widths 2, 4 and 8 bytes and a set of layouts (transpose, 3-D
//! permute, broadcast, narrowed with a start offset, flip = negative stride,
//! already contiguous), the native result must be byte-identical to the CPU walk.
//! Widths 1 and 16 must report `None` (the caller's host fallback). Exits non-zero
//! on any difference.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-dispatch --features vulkan --example contiguize_vulkan_check
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling).

use std::sync::Arc;

use fuel_cpu_backend::byte_kernels::contiguize_cpu;
use fuel_cpu_backend::byte_storage::CpuStorageBytes;
use fuel_dispatch::vulkan_dispatch::contiguize_vulkan;
use fuel_ir::{Layout, Shape};
use fuel_vulkan_backend::VulkanBackend;

fn pattern(n_bytes: usize) -> Vec<u8> {
    (0..n_bytes)
        .map(|i| (i.wrapping_mul(131).wrapping_add(i >> 8) % 251) as u8)
        .collect()
}

/// `(name, layout over a 6 x 8 base unless the layout carries its own extent)` and the base element count.
fn cases() -> Vec<(&'static str, Layout, usize)> {
    let base = Layout::contiguous(Shape::from(vec![6usize, 8]));
    let cube = Layout::contiguous(Shape::from(vec![3usize, 4, 5]));
    vec![
        ("contiguous", base.clone(), 48),
        ("transpose", base.transpose(0, 1).unwrap(), 48),
        ("permute3", cube.permute(&[2, 0, 1]).unwrap(), 60),
        (
            "narrow+offset",
            base.narrow(1, 2, 4).unwrap().narrow(0, 1, 3).unwrap(),
            48,
        ),
        (
            "broadcast",
            Layout::contiguous(Shape::from(vec![1usize, 8]))
                .broadcast_as(Shape::from(vec![5usize, 8]))
                .unwrap(),
            8,
        ),
        ("flip (negative stride)", base.flip(1).unwrap(), 48),
        (
            "flip rows + transpose",
            base.flip(0).unwrap().transpose(0, 1).unwrap(),
            48,
        ),
    ]
}

fn main() {
    let be = Arc::new(VulkanBackend::new().expect("vulkan backend"));
    let mut failures = 0usize;
    let mut checked = 0usize;
    for width in [2usize, 4, 8] {
        for (name, layout, base_elems) in cases() {
            let host = pattern(base_elems * width);
            let src = be.upload_bytes_handle(&host).expect("upload");
            let want = contiguize_cpu(&CpuStorageBytes::from_bytes(&host), &layout, width)
                .expect("cpu contiguize");
            let got = contiguize_vulkan(&src, &layout, width)
                .expect("contiguize_vulkan")
                .expect("native path should apply");
            let got_bytes = be.download_bytes(&got).expect("download");
            let ok = got_bytes.as_slice() == want.bytes();
            checked += 1;
            if !ok {
                failures += 1;
            }
            println!(
                "width {width} {name:<24} {}",
                if ok { "ok" } else { "MISMATCH" }
            );
        }
    }
    // Widths the kernels do not cover must decline, not guess.
    for width in [1usize, 16] {
        let layout = Layout::contiguous(Shape::from(vec![2usize, 3]));
        let src = be.upload_bytes_handle(&pattern(6 * width)).expect("upload");
        let none = contiguize_vulkan(&src, &layout, width)
            .expect("contiguize_vulkan")
            .is_none();
        checked += 1;
        if !none {
            failures += 1;
        }
        println!(
            "width {width} declines (host fallback): {}",
            if none { "ok" } else { "WRONG" }
        );
    }
    println!("{checked} checks, {failures} failures");
    if failures != 0 {
        std::process::exit(1);
    }
}

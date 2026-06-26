//! Live-CUDA tests for baracuda-kernels-sys-backed Contiguize
//! (Phase 7.6 step 9c E.3.2.4). Exercises every layout shape the
//! executor's auto_contiguize pass must handle: transpose (positive
//! strides, non-canonical order), broadcast (zero strides), slice
//! (non-zero offset), flip (negative strides).
//!
//! These tests construct strided source storages directly and call
//! the contiguize wrapper without going through the pipelined
//! executor — the executor's auto_contiguize loop is tested
//! indirectly by every kernel-op live test (which exercises non-
//! contig inputs through realize_*).

#![cfg(feature = "cuda")]

use fuel_ir::{DimVec, Layout, Shape, StrideVec};
use fuel_cuda_backend::{baracuda::contiguize::contiguize_to_fresh, CudaDevice, CudaStorageBytes};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> CudaStorageBytes {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d")
}

fn download_f32(s: &CudaStorageBytes) -> Vec<f32> {
    let bytes = s.to_cpu_bytes().expect("d2h");
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

/// Transpose: source [2, 3] stored row-major, layout reflects column-
/// major view. Contiguize should materialize the column-major order
/// as a fresh contiguous buffer.
#[test]
#[ignore]
fn baracuda_contiguize_f32_transpose_2d() {
    let Some(dev) = dev_or_skip() else { return };
    // Source row-major [2, 3]:
    //   [[1, 2, 3],
    //    [4, 5, 6]]
    let src_host = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let src = upload_f32(&dev, &src_host);

    // Transpose layout: shape [3, 2], strides [1, 3] over the same buffer.
    let layout = Layout::new(
        Shape::from(DimVec::from_slice(&[3, 2])),
        StrideVec::from_slice(&[1, 3]),
        0,
    );

    let contig = contiguize_to_fresh(&src, &layout, 4).expect("contiguize");
    let got = download_f32(&contig);
    // Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]] = [1,4,2,5,3,6].
    assert_eq!(got, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// Broadcast: source [3] becomes shape [2, 3] via stride [0, 1].
/// Contiguize should duplicate the source row.
#[test]
#[ignore]
fn baracuda_contiguize_f32_broadcast_zero_stride() {
    let Some(dev) = dev_or_skip() else { return };
    let src_host = vec![10.0_f32, 20.0, 30.0];
    let src = upload_f32(&dev, &src_host);

    // BroadcastTo [2, 3] with strides [0, 1] over the 3-element source.
    let layout = Layout::new(
        Shape::from(DimVec::from_slice(&[2, 3])),
        StrideVec::from_slice(&[0, 1]),
        0,
    );

    let contig = contiguize_to_fresh(&src, &layout, 4).expect("contiguize");
    let got = download_f32(&contig);
    assert_eq!(got, vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
}

/// Slice: source [5] with view starting at offset=2, length 3.
/// Layout shape [3], stride [1], offset 2.
#[test]
#[ignore]
fn baracuda_contiguize_f32_slice_offset() {
    let Some(dev) = dev_or_skip() else { return };
    let src_host = vec![100.0_f32, 200.0, 300.0, 400.0, 500.0];
    let src = upload_f32(&dev, &src_host);

    let layout = Layout::new(
        Shape::from(DimVec::from_slice(&[3])),
        StrideVec::from_slice(&[1]),
        2, // element offset
    );

    let contig = contiguize_to_fresh(&src, &layout, 4).expect("contiguize");
    let got = download_f32(&contig);
    assert_eq!(got, vec![300.0, 400.0, 500.0]);
}

/// Flip: source [4], layout with negative stride (-1) and offset
/// pointing at the last element. Contiguize should materialize the
/// reversed sequence.
#[test]
#[ignore]
fn baracuda_contiguize_f32_flip_negative_stride() {
    let Some(dev) = dev_or_skip() else { return };
    let src_host = vec![1.0_f32, 2.0, 3.0, 4.0];
    let src = upload_f32(&dev, &src_host);

    // Flip layout: shape [4], stride [-1], offset 3 (last element).
    // source[3 + i * (-1)] for i in 0..4 = source[3], source[2], source[1], source[0]
    //                                    = 4, 3, 2, 1.
    let layout = Layout::new(
        Shape::from(DimVec::from_slice(&[4])),
        StrideVec::from_slice(&[-1]),
        3,
    );

    let contig = contiguize_to_fresh(&src, &layout, 4).expect("contiguize");
    let got = download_f32(&contig);
    assert_eq!(got, vec![4.0, 3.0, 2.0, 1.0]);
}

/// Already-contiguous + zero offset: baracuda's fast path should
/// reduce to a single cuMemcpyDtoDAsync. Verify byte-correctness.
#[test]
#[ignore]
fn baracuda_contiguize_f32_already_contig_fast_path() {
    let Some(dev) = dev_or_skip() else { return };
    let src_host: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let src = upload_f32(&dev, &src_host);

    let layout = Layout::contiguous(Shape::from(DimVec::from_slice(&[3, 4])));
    let contig = contiguize_to_fresh(&src, &layout, 4).expect("contiguize");
    let got = download_f32(&contig);
    assert_eq!(got, src_host);
}

/// b8 byte-width dispatch: f64 contiguize through the same wrapper.
/// Tests the byte-width selection inside contiguize_to_fresh.
#[test]
#[ignore]
fn baracuda_contiguize_f64_transpose_2d() {
    let Some(dev) = dev_or_skip() else { return };
    let src_host = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes: &[u8] = bytemuck::cast_slice(&src_host);
    let src = CudaStorageBytes::from_cpu_bytes(&dev, bytes).expect("h2d");

    let layout = Layout::new(
        Shape::from(DimVec::from_slice(&[3, 2])),
        StrideVec::from_slice(&[1, 3]),
        0,
    );

    let contig = contiguize_to_fresh(&src, &layout, 8).expect("contiguize");
    let host_bytes = contig.to_cpu_bytes().expect("d2h");
    let got: Vec<f64> = bytemuck::cast_slice::<u8, f64>(&host_bytes).to_vec();
    assert_eq!(got, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 34: `load_transposed_matrix_preserve_dtype` keeps an F16 checkpoint half-width
//! (`WeightStorage::F16`), transposed `[out, in]` -> `[in, out]`, bit-for-bit; BF16 stays BF16 and
//! an F32 file stays on the f32 path (the control that the dtype branch, not the loader, decides).

use fuel_ir::DType;
use fuel_loaders::safetensors::MmapedSafetensors;
use fuel_tensor::lazy::{WeightStorage, load_transposed_matrix_preserve_dtype};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, TensorView};

const OUT: usize = 3;
const IN: usize = 4;

/// Row-major `[OUT, IN]` source values, all exactly representable in f16 and bf16.
fn source() -> Vec<f32> {
    (0..OUT * IN).map(|i| i as f32 * 0.5 - 3.0).collect()
}

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    let len = std::mem::size_of_val(v);
    // SAFETY: `v` is a live slice of plain `Copy` numeric types; reading its bytes is sound.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), len) }.to_vec()
}

fn load(dtype: Dtype, data: &[u8]) -> WeightStorage {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("w.safetensors");
    let view = TensorView::new(dtype, vec![OUT, IN], data).expect("view");
    safetensors::serialize_to_file([("w", view)], None, &path).expect("serialize");
    // SAFETY: the file is private to this test and not modified while mapped.
    let st = unsafe { MmapedSafetensors::new(&path) }.expect("mmap");
    load_transposed_matrix_preserve_dtype(&st, "w", OUT, IN).expect("load")
}

#[test]
fn f16_file_stays_f16_and_transposes() {
    let src = source();
    let halves: Vec<f16> = src.iter().map(|&x| f16::from_f32(x)).collect();
    let ws = load(Dtype::F16, &bytes_of(&halves));
    assert_eq!(ws.dtype(), DType::F16);
    let WeightStorage::F16(got) = ws else {
        panic!("expected WeightStorage::F16");
    };
    // Target layout `[IN, OUT]`: got[j * OUT + i] == W[i, j] == src[i * IN + j].
    for i in 0..OUT {
        for j in 0..IN {
            assert_eq!(got[j * OUT + i].to_bits(), halves[i * IN + j].to_bits());
        }
    }
    // Sabotage sibling: an un-transposed copy differs, so the check above can fail.
    assert_ne!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        halves.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn bf16_file_stays_bf16() {
    let halves: Vec<bf16> = source().iter().map(|&x| bf16::from_f32(x)).collect();
    assert_eq!(load(Dtype::BF16, &bytes_of(&halves)).dtype(), DType::BF16);
}

#[test]
fn f32_file_takes_the_f32_path() {
    let ws = load(Dtype::F32, &bytes_of(&source()));
    assert_eq!(ws.dtype(), DType::F32);
}

/// The half-width weight is usable: `apply_linear` on an F16 weight with f32 activations equals the
/// f64 matmul over the same widened weight (small integers and halves, so f32 accumulate is exact).
#[test]
fn f16_weight_applies_linear_on_the_cpu() {
    use fuel_ir::Shape;
    use fuel_tensor::lazy::Tensor;
    let (in_f, out_f, batch) = (IN, OUT, 2);
    let w: Vec<f16> = (0..in_f * out_f)
        .map(|i| f16::from_f32(i as f32 * 0.25 - 1.0))
        .collect();
    let x: Vec<f32> = (0..batch * in_f).map(|i| i as f32 * 0.5 + 1.0).collect();
    let anchor = Tensor::from_f32(
        x.clone(),
        Shape::from_dims(&[batch, in_f]),
        &fuel_tensor::Device::cpu(),
    )
    .expect("x");
    let y = WeightStorage::F16(w.clone().into())
        .apply_linear(&anchor, in_f, out_f)
        .expect("apply_linear")
        .realize_f32()
        .to_vec();
    for b in 0..batch {
        for j in 0..out_f {
            let want: f32 = (0..in_f)
                .map(|k| x[b * in_f + k] * w[k * out_f + j].to_f32())
                .sum();
            assert_eq!(y[b * out_f + j], want, "row {b} col {j}");
        }
    }
}

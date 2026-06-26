//! PyTorch-oracle parity for `LazyTensor::conv_transpose1d`.
//!
//! The lazy primitive is built by lifting the rank-3 input + weight
//! into rank-4 and dispatching through `conv_transpose2d`. This
//! test confirms the lift produces bit-equivalent output to a
//! direct PyTorch `nn.functional.conv_transpose1d` reference (same
//! oracle values used by the eager `conv_tests.rs`).
//!
//! Unblocks: audio codec ports (DAC, EnCodec, SNAC, Mimi,
//! Parler-TTS, MetaVoice, CSM) which all need transposed-conv
//! upsampling on quantized latents to reconstruct waveforms.

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

const T_DATA: [f32; 20] = [
    0.4056, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
    1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
];

const W_FLAT: [f32; 24] = [
    -0.8404, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
    -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
    -0.6451, -0.0840, -1.4247, 0.5512,
];

/// Transpose `[2, 4, 3]` to `[4, 2, 3]` along the first two axes —
/// matches what eager `w.transpose(0, 1)` does.
fn transposed_weight() -> Vec<f32> {
    let (a, b, c) = (2_usize, 4_usize, 3_usize);
    let mut out = vec![0.0_f32; a * b * c];
    for i in 0..a {
        for j in 0..b {
            for k in 0..c {
                let src = i * (b * c) + j * c + k;
                let dst = j * (a * c) + i * c + k;
                out[dst] = W_FLAT[src];
            }
        }
    }
    out
}

#[test]
fn conv_transpose1d_groups_1_matches_pytorch() {
    let dev = fuel_core::Device::cpu();
    let t = LazyTensor::from_f32(T_DATA.to_vec(), Shape::from_dims(&[1, 4, 5]), &dev);
    let wt = transposed_weight();
    let w = t.const_f32_like(wt, Shape::from_dims(&[4, 2, 3]));

    let res = t.conv_transpose1d(&w, 1, 0, 0, 1, 1).unwrap();
    let shape = res.shape();
    let dims = shape.dims();
    assert_eq!(dims, &[1, 2, 7]);

    let out = res.realize_f32();
    let expected: [f32; 14] = [
        0.0699, -1.2899, 8.3018, 5.5873, 2.4572, -2.6143, -0.0706,
        1.8765, 4.8318, 1.1538, 4.7076, -5.9745, -0.8276, 1.6210,
    ];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-3,
            "groups=1 mismatch at {i}: got {a}, expected {e}");
    }
}

#[test]
fn conv_transpose1d_groups_2_matches_pytorch() {
    let dev = fuel_core::Device::cpu();
    let t = LazyTensor::from_f32(T_DATA.to_vec(), Shape::from_dims(&[1, 4, 5]), &dev);
    let wt = transposed_weight();
    let w = t.const_f32_like(wt, Shape::from_dims(&[4, 2, 3]));

    let res = t.conv_transpose1d(&w, 1, 0, 0, 1, 2).unwrap();
    let shape = res.shape();
    let dims = shape.dims();
    assert_eq!(dims, &[1, 4, 7]);

    let out = res.realize_f32();
    let expected: [f32; 28] = [
        -1.5596, -1.8099, 2.0407, 4.8764, -0.1743, -0.7350, -0.7819,
        0.7816, 3.8152, -0.5926, 2.2515, -5.1844, -0.3157, 1.4721,
        1.6295, 0.5200, 6.2611, 0.7109, 2.6315, -1.8793, 0.7113,
        1.0949, 1.0166, 1.7464, 2.4561, -0.7900, -0.5119, 0.1488,
    ];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-3,
            "groups=2 mismatch at {i}: got {a}, expected {e}");
    }
}

/// Stride=2 with output_padding=1: the typical audio-codec
/// upsampling shape. Verifies the lift handles non-unit stride
/// + output_padding correctly.
#[test]
fn conv_transpose1d_stride_2_out_pad_1_shape() {
    let dev = fuel_core::Device::cpu();
    let t = LazyTensor::from_f32(vec![0.5_f32; 1 * 1 * 4], Shape::from_dims(&[1, 1, 4]), &dev);
    let w = t.const_f32_like(vec![0.3_f32; 1 * 1 * 3], Shape::from_dims(&[1, 1, 3]));
    // Lout = (4-1)*2 + (3-1) + 1 + 1 - 2 = 8.
    let res = t.conv_transpose1d(&w, 2, 1, 1, 1, 1).unwrap();
    let shape = res.shape();
    let dims = shape.dims();
    assert_eq!(dims, &[1, 1, 8]);
    for &v in &res.realize_f32() {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
}

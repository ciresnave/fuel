// SPDX-License-Identifier: MIT OR Apache-2.0
use enterpolation::Generator;
use enterpolation::linear::ConstEquidistantLinear;
use palette::LinSrgb;

pub struct SpectralRColormap {
    gradient: ConstEquidistantLinear<f32, LinSrgb, 9>,
}

impl SpectralRColormap {
    pub(crate) fn new() -> Self {
        // Define a colormap similar to 'Spectral_r' by specifying key colors.
        // got the colors from ChatGPT-4o
        let gradient = ConstEquidistantLinear::<f32, _, 9>::equidistant_unchecked([
            LinSrgb::new(0.3686, 0.3098, 0.6353), // Dark blue
            LinSrgb::new(0.1961, 0.5333, 0.7412), // Blue
            LinSrgb::new(0.4000, 0.7608, 0.6471), // Cyan
            LinSrgb::new(0.6706, 0.8667, 0.6431), // Green
            LinSrgb::new(0.9020, 0.9608, 0.5961), // Yellow
            LinSrgb::new(0.9961, 0.8784, 0.5451), // Orange
            LinSrgb::new(0.9922, 0.6824, 0.3804), // Red
            LinSrgb::new(0.9569, 0.4275, 0.2627), // Dark red
            LinSrgb::new(0.8353, 0.2431, 0.3098), // Dark purple
        ]);
        Self { gradient }
    }

    fn get_color(&self, value: f32) -> LinSrgb {
        // `gen` is a reserved keyword in edition 2024 — the raw identifier is
        // required to call enterpolation's `Generator::gen`.
        self.gradient.r#gen(value)
    }

    /// Map a `height * width` grayscale plane to a `3 * height * width`
    /// **CHW** RGB buffer.
    ///
    /// B6 note: this took and returned an eager `Tensor`, but the colormap
    /// lookup was always host arithmetic — the tensor only supplied the
    /// HWC→CHW `permute`, which is done here directly.
    pub fn gray2color(&self, gray: &[f32], height: usize, width: usize) -> Vec<f32> {
        debug_assert_eq!(gray.len(), height * width);
        let plane = height * width;
        let mut out = vec![0f32; 3 * plane];
        for (i, g) in gray.iter().enumerate() {
            let rgb = self.get_color(*g);
            out[i] = rgb.red;
            out[plane + i] = rgb.green;
            out[2 * plane + i] = rgb.blue;
        }
        out
    }
}

// Re-export the lazy SAM types so the wasm binary can use them through
// this crate (`use fuel_wasm_example_sam as sam; sam::SamModel; …`).
pub use fuel::lazy_sam::{SamModel, SamModelConfig};
/// SAM input image side (Meta default = 1024).
pub const IMAGE_SIZE: usize = 1024;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(s: &str);
}

#[macro_export]
macro_rules! console_log {
    // Note that this is using the `log` function imported above during
    // `bare_bones`
    ($($t:tt)*) => ($crate::log(&format_args!($($t)*).to_string()))
}

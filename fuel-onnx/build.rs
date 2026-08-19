// SPDX-License-Identifier: MIT OR Apache-2.0
use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["src/onnx.proto3"], &["src/"])?;
    Ok(())
}

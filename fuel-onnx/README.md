# fuel-onnx

This crate adds ONNX support to fuel

## FAQ

#### Missing protoc installation when compiling fuel-onnx

The fuel-onnx dependency prost-build no longer comes bundled with prost
binaries. This could cause the following error when attempting to compile
fuel-onnx:

```
error: failed to run custom build command for `fuel-onnx`
Caused by: // (...)
  Could not find `protoc` installation and this build crate cannot proceed without this knowledge.
```

To fix this issue install protoc on your system and make it available in your
system `PATH`. See the [protoc
documentation](https://grpc.io/docs/protoc-installation/) for more information.

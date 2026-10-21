# Vulkan SPIR-V validation errors (2026-10-21)

Running the Qwen3-0.6B decode (`qwen3_vulkan_profile`, 1 run, 1 step) under the Khronos
validation layer (core checks on) printed **131 validation errors**, identical in two runs. They
are pipeline- and shader-module-creation errors: the committed SPIR-V declares capabilities that
the device creation in `fuel-vulkan-backend/src/lib.rs` never enabled, plus two real shader
issues. The messages carry no shader name, so `scripts/spirv_audit.py` parses every module in
`fuel-vulkan-kernels/spv` (178) and lists which shaders declare each capability.

## Counts by VUID

| VUID | meaning | before | after the device-feature change |
|---|---|---|---|
| `VkShaderModuleCreateInfo-pCode-08740` | capability declared, feature not enabled (`UniformAndStorageBuffer16BitAccess`, `Int16`, `VulkanMemoryModel`) | 50 | **0** |
| `RuntimeSpirv-uniformAndStorageBuffer16BitAccess-06332` | 16-bit `Uniform` variable | 50 | **0** |
| `RuntimeSpirv-None-06278` | 64-bit integer atomics on buffers (`shaderBufferInt64Atomics`) | 20 | **0** |
| `RuntimeSpirv-Workgroup-06530` | shared memory above the reported `maxComputeSharedMemorySize` (32,768) | 6 | 6 |
| `VkShaderModuleCreateInfo-pCode-08737` | `spirv-val`: GLSL.std.450 `Exp` with an f64 result type | 5 | 5 |
| (other) | | 0 | 0 |
| total | | 131 | **11** |

The `after` column is one run, one step, with the layer settings used for the synchronization
check (`validate_sync = true`, core checks on). The pre-change 131 is the same measurement taken
twice. The two counts per class that do not divide into the audit's shader counts (50 vs 19
modules for 16-bit access, 6 vs 3 for shared memory) are not reconciled: the decode creates
modules and pipelines more than once per shader and the layer reports per creation. Only the
direction and the classes are claims here.

During the fix the layer revealed classes that were hidden behind the first one (the `Int16`
capability behind 16-bit access, `VulkanMemoryModel` and then its device scope
(`VUID-RuntimeSpirv-vulkanMemoryModel-06265`) behind the cooperative-matrix shaders); each was
found by re-running, so the list above is what one decode run exercises, not every shader.

## What changed

`fuel-vulkan-backend/src/lib.rs` now probes the device (`probe_optional_features`, through
`vkGetPhysicalDeviceFeatures2`; the 64-bit-atomics and memory-model structs are Vulkan 1.2 core,
so an older device reports them absent) and enables each of `uniformAndStorageBuffer16BitAccess`,
`shaderBufferInt64Atomics` and `shaderInt16` only if reported, and `vulkanMemoryModel` with
`vulkanMemoryModelDeviceScope` only on the cooperative-matrix path and only if reported. Enabling
a feature the SPIR-V already declares changes no instruction a kernel executes. Check: the
CPU-fixture comparison (`QWEN3_VERIFY=1`) on the unconditional version of this change is max
logit difference 9.2e-5 with 0 of 64 greedy tokens different, PASS, the same as before (9.9e-5 /
9.2e-5, one run each); the probed version gave 11 layer errors, the same 11, on the RTX 4070.

**Probe output** (`cargo run -p fuel-vulkan-backend --example declared_features_probe`, through
`gpu-run`): the RTX 4070 Laptop GPU and the AMD Radeon 610M iGPU both report all five features
supported, `declared_features_absent` is empty, and both backends are created.

**On absence** (not exercised: no device here lacks one): the feature is not enabled, the backend
is still created, a `tracing::warn!` names the missing features, and they are listed in
`VulkanBackend::declared_features_absent`. The shaders that declare them still load, because
`Pipelines::new` builds every pipeline eagerly, so they run unvalidated on that device, exactly as
before this change. Marking those kernels unavailable would mean making each affected pipeline
optional and teaching the dispatcher to decline them; that is not done here.

## Left, and not fixed here

* **`VUID-RuntimeSpirv-Workgroup-06530` (3 shaders):** `flash_attention` (33,920 bytes),
  `flash_attn_backward_k_f32` and `flash_attn_backward_q_f32` (35,852 bytes each) exceed 32,768.
  The layer reported the limit as 32,768 on the device used; the device value was not read back
  separately. Fixing it means shrinking tiles, which changes the reduction order, so it is a
  numerics decision and is reported, not made.
* **`VUID-VkShaderModuleCreateInfo-pCode-08737` (1 shader):** `softmax_f64` calls GLSL.std.450
  `Exp` on f64, which the Vulkan environment does not allow (16/32-bit only). Behaviour on a
  driver that accepts it today is undefined by the spec; the fix (compute `exp` in f32, or write an
  f64 `exp`) changes the numerics, so it is reported, not made.

## Per-shader classes (`python scripts/spirv_audit.py --markdown`, 178 modules)

| shader | classes |
|---|---|
| `affine_f16` | uniformAndStorageBuffer16BitAccess |
| `arg_reduce_any_dim_f16` | shaderInt16 |
| `arg_reduce_last_dim_f16` | uniformAndStorageBuffer16BitAccess |
| `binary_f16` | uniformAndStorageBuffer16BitAccess |
| `cast_f16_to_f32` | shaderInt16 |
| `cast_f16_to_f8e4m3` | shaderInt16 |
| `cast_f32_to_f16` | shaderInt16 |
| `cast_f8e4m3_to_f16` | shaderInt16 |
| `concat_along_dim_f16` | uniformAndStorageBuffer16BitAccess |
| `conv2d_im2col_bf16` | shaderInt16 |
| `cumsum_bf16` | uniformAndStorageBuffer16BitAccess, shaderInt16 |
| `cumsum_f16` | uniformAndStorageBuffer16BitAccess |
| `dequant_q4_0` | shaderInt16 |
| `dequant_q4_km` | shaderInt16 |
| `dequant_q8_0` | shaderInt16 |
| `flash_attention` | shared 33920 B |
| `flash_attn_backward_k_f32` | shared 35852 B |
| `flash_attn_backward_q_f32` | shared 35852 B |
| `flash_attn_bf16` | shaderInt16 |
| `flash_attn_combine_bf16` | shaderInt16 |
| `flash_attn_split_bf16` | shaderInt16 |
| `flip_b2` | uniformAndStorageBuffer16BitAccess, shaderInt16 |
| `index_add_f16` | shaderInt16 |
| `index_add_f64` | shaderBufferInt64Atomics |
| `index_select_f16` | uniformAndStorageBuffer16BitAccess |
| `layer_norm_last_dim_backward_f16` | uniformAndStorageBuffer16BitAccess |
| `layer_norm_last_dim_f16` | uniformAndStorageBuffer16BitAccess |
| `matmul_coop` | vulkanMemoryModel |
| `matmul_coop_bf16_bf16` | shaderInt16, vulkanMemoryModel |
| `matmul_coop_bf16_bf16_bf16` | shaderInt16, vulkanMemoryModel |
| `matmul_coop_f16_f16` | vulkanMemoryModel |
| `matmul_coop_f16_f16_f16` | shaderInt16, vulkanMemoryModel |
| `matmul_q4_0_tiled` | shaderInt16 |
| `matmul_small_bf16_bf16_bf16` | shaderInt16 |
| `matmul_small_bf16_bf16_f32` | shaderInt16 |
| `pad_backward_reflect_f16` | shaderInt16 |
| `pad_backward_reflect_f64` | shaderBufferInt64Atomics |
| `pad_backward_replicate_f16` | shaderInt16 |
| `pad_backward_replicate_f64` | shaderBufferInt64Atomics |
| `qmatvec_q4_0` | shaderInt16 |
| `quantize_q8_0` | shaderInt16 |
| `reduce_f16` | uniformAndStorageBuffer16BitAccess |
| `reduce_last_dim_f16` | uniformAndStorageBuffer16BitAccess |
| `rms_norm_last_dim_f16` | uniformAndStorageBuffer16BitAccess |
| `roll_b2` | uniformAndStorageBuffer16BitAccess, shaderInt16 |
| `rope_f16` | uniformAndStorageBuffer16BitAccess |
| `scatter_add_f16` | shaderInt16 |
| `scatter_add_f64` | shaderBufferInt64Atomics |
| `softmax_f16` | uniformAndStorageBuffer16BitAccess |
| `softmax_f64` | f64 Exp |
| `softmax_last_dim_backward_f16` | uniformAndStorageBuffer16BitAccess |
| `strided_copy_signed_b2` | uniformAndStorageBuffer16BitAccess, shaderInt16 |
| `unary_f16` | uniformAndStorageBuffer16BitAccess |

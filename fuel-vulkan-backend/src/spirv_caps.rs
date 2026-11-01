// SPDX-License-Identifier: MIT OR Apache-2.0
//! Which optional device features a SPIR-V module needs, read from its
//! `OpCapability` words, and the typed reason a kernel is unavailable when the
//! device lacks one.

use crate::OptionalFeatureSupport;

const OP_CAPABILITY: u32 = 17;
const CAP_INT16: u32 = 22;
const CAP_INT64_ATOMICS: u32 = 12;
const CAP_UNIFORM_AND_STORAGE_BUFFER_16BIT: u32 = 4434;
const CAP_VULKAN_MEMORY_MODEL: u32 = 5345;
const CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE: u32 = 5346;

/// The optional features one module needs (the same five as [`OptionalFeatureSupport`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RequiredFeatures {
    pub uniform_and_storage_buffer16_bit_access: bool,
    pub shader_buffer_int64_atomics: bool,
    pub shader_int16: bool,
    pub vulkan_memory_model: bool,
    pub vulkan_memory_model_device_scope: bool,
}

impl RequiredFeatures {
    pub fn any(&self) -> bool {
        *self != Self::default()
    }
}

/// Scan a SPIR-V module's `OpCapability` instructions. A truncated or non-SPIR-V
/// input yields whatever was read before the break (an empty set for non-SPIR-V).
pub fn required_features(spirv: &[u8]) -> RequiredFeatures {
    let mut out = RequiredFeatures::default();
    let words: Vec<u32> = spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    if words.first() != Some(&0x0723_0203) {
        return out;
    }
    let mut i = 5;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = words[i] & 0xFFFF;
        if wc == 0 {
            break;
        }
        // OpCapability instructions precede every other instruction that follows
        // the header, so stop at the first one that is not a capability.
        if op != OP_CAPABILITY {
            if i > 5 {
                break;
            }
        } else if let Some(&cap) = words.get(i + 1) {
            match cap {
                CAP_INT16 => out.shader_int16 = true,
                CAP_INT64_ATOMICS => out.shader_buffer_int64_atomics = true,
                CAP_UNIFORM_AND_STORAGE_BUFFER_16BIT => {
                    out.uniform_and_storage_buffer16_bit_access = true
                }
                CAP_VULKAN_MEMORY_MODEL => out.vulkan_memory_model = true,
                CAP_VULKAN_MEMORY_MODEL_DEVICE_SCOPE => out.vulkan_memory_model_device_scope = true,
                _ => {}
            }
        }
        i += wc;
    }
    out
}

impl OptionalFeatureSupport {
    /// The first feature `req` needs that this device lacks (device-feature name),
    /// or `None` when every needed feature is present.
    pub fn first_missing(&self, req: &RequiredFeatures) -> Option<&'static str> {
        [
            (
                req.uniform_and_storage_buffer16_bit_access,
                self.uniform_and_storage_buffer16_bit_access,
                "uniformAndStorageBuffer16BitAccess",
            ),
            (
                req.shader_buffer_int64_atomics,
                self.shader_buffer_int64_atomics,
                "shaderBufferInt64Atomics",
            ),
            (req.shader_int16, self.shader_int16, "shaderInt16"),
            (
                req.vulkan_memory_model,
                self.vulkan_memory_model,
                "vulkanMemoryModel",
            ),
            (
                req.vulkan_memory_model_device_scope,
                self.vulkan_memory_model_device_scope,
                "vulkanMemoryModelDeviceScope",
            ),
        ]
        .into_iter()
        .find(|(needed, present, _)| *needed && !*present)
        .map(|(_, _, name)| name)
    }
}

/// Typed reason a Vulkan kernel is not offered on this device: its shader needs a
/// device feature the device does not report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelUnavailable {
    pub kernel: &'static str,
    pub missing_feature: &'static str,
}

impl std::fmt::Display for KernelUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "vulkan kernel `{}` is unavailable on this device: it needs the `{}` feature, \
             which the device does not report",
            self.kernel, self.missing_feature
        )
    }
}

impl std::error::Error for KernelUnavailable {}

impl From<KernelUnavailable> for fuel_ir::Error {
    fn from(e: KernelUnavailable) -> Self {
        fuel_ir::Error::wrap(e)
    }
}

/// The kernels among `modules` (name, SPIR-V bytes) that `support` cannot run.
pub fn unavailable_for<'a>(
    support: OptionalFeatureSupport,
    modules: impl IntoIterator<Item = &'a (&'static str, &'static [u8])>,
) -> Vec<KernelUnavailable> {
    modules
        .into_iter()
        .filter_map(|(name, spv)| {
            support
                .first_missing(&required_features(spv))
                .map(|missing_feature| KernelUnavailable {
                    kernel: name,
                    missing_feature,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_present() -> OptionalFeatureSupport {
        OptionalFeatureSupport {
            uniform_and_storage_buffer16_bit_access: true,
            shader_buffer_int64_atomics: true,
            shader_int16: true,
            vulkan_memory_model: true,
            vulkan_memory_model_device_scope: true,
        }
    }

    fn names(v: &[KernelUnavailable]) -> Vec<&'static str> {
        v.iter().map(|u| u.kernel).collect()
    }

    fn count_requiring(f: impl Fn(&RequiredFeatures) -> bool) -> usize {
        fuel_vulkan_kernels::EMBEDDED
            .iter()
            .filter(|(_, spv)| f(&required_features(spv)))
            .count()
    }

    /// Positive control for the scanner: the per-feature counts match
    /// `python scripts/spirv_audit.py` (19 / 30 / 4 / 5), a parser written separately.
    #[test]
    fn scanner_counts_match_the_static_audit() {
        assert_eq!(
            count_requiring(|r| r.uniform_and_storage_buffer16_bit_access),
            19
        );
        assert_eq!(count_requiring(|r| r.shader_int16), 30);
        assert_eq!(count_requiring(|r| r.shader_buffer_int64_atomics), 4);
        assert_eq!(count_requiring(|r| r.vulkan_memory_model), 5);
    }

    #[test]
    fn a_device_with_every_feature_declines_nothing() {
        assert!(unavailable_for(all_present(), fuel_vulkan_kernels::EMBEDDED).is_empty());
    }

    /// Each feature forced absent declines exactly the kernels that need it, and the
    /// sabotage sibling (the same feature present) offers them again.
    #[test]
    fn each_absent_feature_declines_its_kernels_and_enabling_it_offers_them() {
        type Probe = (
            &'static str,
            fn(&mut OptionalFeatureSupport),
            &'static str,
            &'static str,
        );
        let probes: [Probe; 4] = [
            (
                "shaderInt16",
                |s| s.shader_int16 = false,
                "cast_f32_to_f16",
                "cast_f32_to_bf16",
            ),
            (
                "uniformAndStorageBuffer16BitAccess",
                |s| s.uniform_and_storage_buffer16_bit_access = false,
                "unary_f16",
                "unary",
            ),
            (
                "shaderBufferInt64Atomics",
                |s| s.shader_buffer_int64_atomics = false,
                "scatter_add_f64",
                "scatter_add_f32",
            ),
            (
                "vulkanMemoryModel",
                |s| s.vulkan_memory_model = false,
                "matmul_coop",
                "matmul_tiled",
            ),
        ];
        for (feature, remove, declined, kept) in probes {
            let mut lacking = all_present();
            remove(&mut lacking);
            let un = unavailable_for(lacking, fuel_vulkan_kernels::EMBEDDED);
            assert!(
                names(&un).contains(&declined),
                "{feature} absent must decline {declined}; declined = {:?}",
                names(&un)
            );
            assert!(
                !names(&un).contains(&kept),
                "{feature} absent must not decline {kept}"
            );
            let reason = un.iter().find(|u| u.kernel == declined).unwrap();
            assert_eq!(reason.missing_feature, feature);
            // Sabotage sibling: with the feature present the kernel is offered.
            let un = unavailable_for(all_present(), fuel_vulkan_kernels::EMBEDDED);
            assert!(!names(&un).contains(&declined));
        }
    }

    /// Every module needing a feature must be a `gated` or `coop` row of the `pipelines!`
    /// table (its pipeline field is an `Option`); a new such shader as a `required` row
    /// would fail device creation on a device lacking the feature.
    #[test]
    fn every_module_needing_a_feature_is_loaded_through_the_gate() {
        // rustfmt may wrap table rows, so compare with all whitespace removed.
        let src: String = include_str!("pipelines.rs")
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        let gated_rows = src.matches(",gated;").count() + src.matches(",coop;").count();
        assert_eq!(
            gated_rows,
            count_requiring(RequiredFeatures::any),
            "modules needing an optional feature vs gated/coop rows of the pipelines! table"
        );
    }

    #[test]
    fn the_decline_is_a_typed_error_naming_kernel_and_feature() {
        let e = KernelUnavailable {
            kernel: "cast_f32_to_f16",
            missing_feature: "shaderInt16",
        };
        let msg = fuel_ir::Error::from(e).to_string();
        assert!(
            msg.contains("cast_f32_to_f16") && msg.contains("shaderInt16"),
            "{msg}"
        );
    }

    #[test]
    fn a_non_spirv_blob_requires_nothing() {
        assert!(!required_features(b"not a spirv module at all!").any());
    }
}

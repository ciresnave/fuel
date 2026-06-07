import re

path = "c:/Users/cires/OneDrive/Documents/projects/fuel/fuel-cpu-backend/src/dyn_impl.rs"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

original_len = len(content)

# 1. Update module-level doc comment
old_doc = (
    "//! `DynBackendStorage` and `DynBackendDevice` implementations for the CPU backend.\n"
    "//!\n"
    "//! This module defines newtype wrappers `CpuBackendStorage` and `CpuBackendDevice`\n"
    "//! that implement the object-safe `DynBackend*` traits from `fuel-core-types`.\n"
    "//!\n"
    "//! These newtypes exist because of Rust's orphan rule: both `DynBackendStorage`\n"
    "//! and `CpuStorage` are defined in `fuel-core-types`, but the impl needs access\n"
    "//! to the computation infrastructure in this crate (`fuel-cpu-backend`). The\n"
    "//! newtype is zero-cost — it's just a transparent wrapper around `CpuStorage`."
)
new_doc = (
    "//! `DynBackendStorage` and `DynBackendDevice` implementations for the CPU backend.\n"
    "//!\n"
    "//! `CpuStorage` (defined here) owns raw tensor data as a typed `HostBuffer` and\n"
    "//! implements `DynBackendStorage` directly. `CpuBackendDevice` is the stateless\n"
    "//! device handle. `CpuBackendStorage` is a backward-compat alias for `CpuStorage`."
)
content = content.replace(old_doc, new_doc)

# 2. Fix import
content = content.replace(
    "use fuel_core_types::{CpuStorage, DType, DeviceLocation, Error, Layout, Result,\n"
    "                         Scalar, Shape};",
    "use fuel_core_types::{CpuStorage as HostBuffer, DType, DeviceLocation, Error, Layout, Result,\n"
    "                         Scalar, Shape};"
)

# 3. Change the struct + its doc
old_struct = (
    "/// Newtype wrapper around [`CpuStorage`] implementing [`DynBackendStorage`].\n"
    "///\n"
    "/// This wrapper is zero-cost (transparent) and exists solely to satisfy the\n"
    "/// orphan rule: `CpuStorage` and `DynBackendStorage` are both defined in\n"
    "/// `fuel-core-types`, so their impl must live in that crate or in a crate\n"
    "/// that defines one of them.  Since `CpuBackendStorage` is defined *here*,\n"
    "/// we can implement the trait here.\n"
    "#[derive(Debug, Clone)]\n"
    "pub struct CpuBackendStorage(pub CpuStorage);"
)
new_struct = (
    "/// CPU backend storage: owns raw tensor data as a typed `HostBuffer`.\n"
    "///\n"
    "/// Defined in `fuel-cpu-backend` so the orphan rule allows implementing\n"
    "/// `DynBackendStorage` here with full access to CPU kernels.\n"
    "#[derive(Debug, Clone)]\n"
    "pub struct CpuStorage(pub HostBuffer);\n"
    "\n"
    "/// Backward-compat alias.\n"
    "pub type CpuBackendStorage = CpuStorage;"
)
content = content.replace(old_struct, new_struct)

# 4. Fix impl CpuBackendStorage
old_impl = (
    "impl CpuBackendStorage {\n"
    "    /// Unwrap the inner `CpuStorage`.\n"
    "    pub fn into_inner(self) -> CpuStorage {\n"
    "        self.0\n"
    "    }\n"
    "\n"
    "    /// Borrow the inner `CpuStorage`.\n"
    "    pub fn inner(&self) -> &CpuStorage {\n"
    "        &self.0\n"
    "    }\n"
    "}"
)
new_impl = (
    "impl CpuStorage {\n"
    "    /// Unwrap the inner `HostBuffer`.\n"
    "    pub fn into_inner(self) -> HostBuffer {\n"
    "        self.0\n"
    "    }\n"
    "\n"
    "    /// Borrow the inner `HostBuffer`.\n"
    "    pub fn inner(&self) -> &HostBuffer {\n"
    "        &self.0\n"
    "    }\n"
    "}"
)
content = content.replace(old_impl, new_impl)

# 5. Fix From impls
old_from = (
    "impl From<CpuStorage> for CpuBackendStorage {\n"
    "    fn from(s: CpuStorage) -> Self {\n"
    "        Self(s)\n"
    "    }\n"
    "}\n"
    "\n"
    "impl From<CpuBackendStorage> for CpuStorage {\n"
    "    fn from(s: CpuBackendStorage) -> Self {\n"
    "        s.0\n"
    "    }\n"
    "}"
)
new_from = (
    "impl From<HostBuffer> for CpuStorage {\n"
    "    fn from(s: HostBuffer) -> Self {\n"
    "        Self(s)\n"
    "    }\n"
    "}\n"
    "\n"
    "impl From<CpuStorage> for HostBuffer {\n"
    "    fn from(s: CpuStorage) -> Self {\n"
    "        s.0\n"
    "    }\n"
    "}"
)
content = content.replace(old_from, new_from)

# 6. Fix HostStorage impl
content = content.replace(
    "impl fuel_core_types::backend::HostStorage for CpuBackendStorage {",
    "impl fuel_core_types::backend::HostStorage for CpuStorage {"
)

# 7. Fix CpuBackendDevice doc
content = content.replace(
    "/// Newtype wrapper around [`CpuDevice`] implementing [`DynBackendDevice`].\n"
    "///\n"
    "/// Same orphan-rule motivation as [`CpuBackendStorage`].",
    "/// CPU device handle (stateless) implementing [`DynBackendDevice`]."
)

# 8. Fix downcast signatures
content = content.replace(
    "/// Downcast a `&dyn DynBackendStorage` to `&CpuBackendStorage`.\n"
    "fn downcast(s: &dyn DynBackendStorage) -> Result<&CpuBackendStorage> {\n"
    "    s.as_any()\n"
    "        .downcast_ref::<CpuBackendStorage>()",
    "/// Downcast a `&dyn DynBackendStorage` to `&CpuStorage`.\n"
    "fn downcast(s: &dyn DynBackendStorage) -> Result<&CpuStorage> {\n"
    "    s.as_any()\n"
    "        .downcast_ref::<CpuStorage>()"
)
content = content.replace(
    "/// Downcast a `&mut dyn DynBackendStorage` to `&mut CpuBackendStorage`.\n"
    "fn downcast_mut(s: &mut dyn DynBackendStorage) -> Result<&mut CpuBackendStorage> {\n"
    "    let loc = s.device_dyn().location_dyn();\n"
    "    s.as_any_mut()\n"
    "        .downcast_mut::<CpuBackendStorage>()",
    "/// Downcast a `&mut dyn DynBackendStorage` to `&mut CpuStorage`.\n"
    "fn downcast_mut(s: &mut dyn DynBackendStorage) -> Result<&mut CpuStorage> {\n"
    "    let loc = s.device_dyn().location_dyn();\n"
    "    s.as_any_mut()\n"
    "        .downcast_mut::<CpuStorage>()"
)

# 9. Fix impl DynBackendStorage comment + impl header
content = content.replace(
    "// impl DynBackendStorage for CpuBackendStorage\n"
    "// ---------------------------------------------------------------------------\n"
    "\n"
    "impl DynBackendStorage for CpuBackendStorage {",
    "// impl DynBackendStorage for CpuStorage\n"
    "// ---------------------------------------------------------------------------\n"
    "\n"
    "impl DynBackendStorage for CpuStorage {"
)

# 10. Change all CpuStorage:: (enum variant) to HostBuffer::
content = content.replace("CpuStorage::", "HostBuffer::")

# 11. Fix function signatures that use CpuStorage as a type
content = re.sub(r'\bs: &CpuStorage\b', 's: &HostBuffer', content)
content = re.sub(r'\bdst: &mut CpuStorage\b', 'dst: &mut HostBuffer', content)
content = re.sub(r'\bsrc: &CpuStorage\b', 'src: &HostBuffer', content)
content = re.sub(r'\bstorage: &mut CpuStorage\b', 'storage: &mut HostBuffer', content)
content = re.sub(r'-> Result<CpuStorage>', '-> Result<HostBuffer>', content)

# 12. Fix to_host_buffer_dyn
content = content.replace(
    "    fn to_host_buffer_dyn(&self) -> Result<CpuStorage> {",
    "    fn to_host_buffer_dyn(&self) -> Result<HostBuffer> {"
)

# 13. Fix DynBackendDevice host buffer method signatures
content = content.replace(
    "        buf: &CpuStorage,\n"
    "    ) -> Result<Box<dyn DynBackendStorage>> {\n"
    "        Ok(Box::new(CpuBackendStorage(buf.clone())))",
    "        buf: &HostBuffer,\n"
    "    ) -> Result<Box<dyn DynBackendStorage>> {\n"
    "        Ok(Box::new(CpuStorage(buf.clone())))"
)
content = content.replace(
    "        buf: CpuStorage,\n"
    "    ) -> Result<Box<dyn DynBackendStorage>> {\n"
    "        Ok(Box::new(CpuBackendStorage(buf)))",
    "        buf: HostBuffer,\n"
    "    ) -> Result<Box<dyn DynBackendStorage>> {\n"
    "        Ok(Box::new(CpuStorage(buf)))"
)

# 14. Replace remaining CpuBackendStorage( constructor calls
content = content.replace("CpuBackendStorage(", "CpuStorage(")

# Check remaining CpuBackendStorage occurrences
remaining = [(m.start(), content[max(0,m.start()-40):m.start()+60]) for m in re.finditer(r'CpuBackendStorage', content)]
print(f"Remaining CpuBackendStorage occurrences: {len(remaining)}")
for pos, ctx in remaining[:15]:
    print(f"  [{pos}] {repr(ctx)}")

# Check for any accidental self-referential issues
remaining_cpu = [(m.start(), content[max(0,m.start()-20):m.start()+40]) for m in re.finditer(r'\bCpuStorage\b', content)]
print(f"\nCpuStorage occurrences (new newtype): {len(remaining_cpu)}")

print(f"\nOriginal length: {original_len}, New length: {len(content)}")

# Write out
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Written successfully.")

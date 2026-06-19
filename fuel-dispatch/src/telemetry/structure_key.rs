//! The `StructureKey` join token.
//!
//! Baracuda owns the structure-key encoding and ships the callable
//! `structure_key(op_class, operands, arch) -> StructureKey`. Fuel **calls** it
//! with FDX operand descriptions as input and **never derives the key itself**.
//! Here the token is treated as opaque bytes for the join; the provider seam
//! (the trait Fuel calls, fed FDX operand descriptions) lands in step 3.

/// Opaque structure-key token. Baracuda owns the encoding (a string or a `u64`
/// rendered as a string); Fuel treats it as bytes for the `(structure_key,
/// chosen)` join and never derives it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct StructureKeyToken(pub String);

//! CPU storage types for tensor data.

use crate::{DType, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};

/// CPU-backed tensor storage holding a typed vector of elements.
///
/// Each variant holds the data for the corresponding [`DType`].
#[derive(Debug, Clone)]
pub enum CpuStorage {
    U8(Vec<u8>),
    U32(Vec<u32>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    F8E4M3(Vec<F8E4M3>),
    // Dummy types that store raw bytes
    F6E2M3(Vec<u8>),
    F6E3M2(Vec<u8>),
    F4(Vec<u8>),
    F8E8M0(Vec<u8>),
}

/// A borrowed reference to CPU tensor storage.
#[derive(Debug, Clone)]
pub enum CpuStorageRef<'a> {
    U8(&'a [u8]),
    U32(&'a [u32]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    BF16(&'a [bf16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
    F64(&'a [f64]),
    F8E4M3(&'a [F8E4M3]),
    // Dummy types that store raw bytes
    F6E2M3(&'a [u8]),
    F6E3M2(&'a [u8]),
    F4(&'a [u8]),
    F8E8M0(&'a [u8]),
}

/// A CPU device handle (unit struct — the CPU needs no state).
#[derive(Debug, Clone)]
pub struct CpuDevice;

impl CpuStorage {
    /// Returns the [`DType`] of the elements stored.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::U32(_) => DType::U32,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F6E2M3(_) => DType::F6E2M3,
            Self::F6E3M2(_) => DType::F6E3M2,
            Self::F4(_) => DType::F4,
            Self::F8E8M0(_) => DType::F8E8M0,
        }
    }

    /// Returns a typed slice of the stored data.
    pub fn as_slice<D: WithDType>(&self) -> crate::Result<&[D]> {
        D::cpu_storage_as_slice(self)
    }

    /// Concatenates multiple storages of the same dtype into one.
    pub fn concat(storages: &[CpuStorage]) -> crate::Result<CpuStorage> {
        let storage0 = &storages[0];
        let s = match storage0 {
            Self::U8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U8(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::U8(storages.concat())
            }
            Self::U32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::U32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::U32(storages.concat())
            }
            Self::I16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::I16(storages.concat())
            }
            Self::I32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::I32(storages.concat())
            }
            Self::I64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::I64(storages.concat())
            }
            Self::BF16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::BF16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::BF16(storages.concat())
            }
            Self::F16(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F16(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F16(storages.concat())
            }
            Self::F32(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F32(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F32(storages.concat())
            }
            Self::F64(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F64(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F64(storages.concat())
            }
            Self::F8E4M3(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F8E4M3(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F8E4M3(storages.concat())
            }
            Self::F6E2M3(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F6E2M3(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F6E2M3(storages.concat())
            }
            Self::F6E3M2(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F6E3M2(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F6E3M2(storages.concat())
            }
            Self::F4(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F4(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F4(storages.concat())
            }
            Self::F8E8M0(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::F8E8M0(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::F8E8M0(storages.concat())
            }
        };
        Ok(s)
    }
}

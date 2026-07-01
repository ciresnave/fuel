//! Host-side tensor storage types.
//!
//! The canonical type is [`HostBuffer`] — a dtype-tagged owned `Vec<T>`
//! representing contiguous tensor data in host-addressable RAM. The
//! type alias [`CpuStorage`] = [`HostBuffer`] preserves backwards
//! compatibility so existing code doesn't need to change.

use crate::{DType, Error, Result, WithDType};
use float8::F8E4M3;
use half::{bf16, f16};

/// Host-addressable tensor storage holding a typed vector of elements.
///
/// This is the universal interchange format for moving tensor data
/// between backends: every `BackendStorage` can produce one via
/// `to_host_buffer()`, and every `BackendDevice` can ingest one via
/// `storage_from_host_buffer()`.
///
/// Previously named `CpuStorage` — that name remains as a type alias.
#[derive(Debug, Clone)]
pub enum HostBuffer {
    U8(Vec<u8>),
    I8(Vec<i8>),
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

/// Backwards-compatibility alias. New code should use [`HostBuffer`].
pub type CpuStorage = HostBuffer;

/// A borrowed reference to host tensor storage.
#[derive(Debug, Clone)]
pub enum HostBufferRef<'a> {
    U8(&'a [u8]),
    I8(&'a [i8]),
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

/// Backwards-compatibility alias. New code should use [`HostBufferRef`].
pub type CpuStorageRef<'a> = HostBufferRef<'a>;

/// A CPU device handle (unit struct — the CPU needs no state).
#[derive(Debug, Clone)]
pub struct CpuDevice;

impl HostBuffer {
    /// Returns the [`DType`] of the elements stored.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
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
    pub fn as_slice<D: HostDType>(&self) -> crate::Result<&[D]> {
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
            Self::I8(_) => {
                let storages = storages
                    .iter()
                    .map(|s| match s {
                        Self::I8(s) => Ok(s.as_slice()),
                        _ => crate::bail!("dtype mismatch"),
                    })
                    .collect::<crate::Result<Vec<_>>>()?;
                Self::I8(storages.concat())
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

    /// Borrow this buffer as a [`HostBufferRef`] (zero-copy).
    ///
    /// The ref form is the common shape for `HostStorage::as_host_buffer_ref`;
    /// owned-buffer implementors delegate through here.
    pub fn as_ref(&self) -> HostBufferRef<'_> {
        match self {
            Self::U8(v) => HostBufferRef::U8(v),
            Self::I8(v) => HostBufferRef::I8(v),
            Self::U32(v) => HostBufferRef::U32(v),
            Self::I16(v) => HostBufferRef::I16(v),
            Self::I32(v) => HostBufferRef::I32(v),
            Self::I64(v) => HostBufferRef::I64(v),
            Self::BF16(v) => HostBufferRef::BF16(v),
            Self::F16(v) => HostBufferRef::F16(v),
            Self::F32(v) => HostBufferRef::F32(v),
            Self::F64(v) => HostBufferRef::F64(v),
            Self::F8E4M3(v) => HostBufferRef::F8E4M3(v),
            Self::F6E2M3(v) => HostBufferRef::F6E2M3(v),
            Self::F6E3M2(v) => HostBufferRef::F6E3M2(v),
            Self::F4(v) => HostBufferRef::F4(v),
            Self::F8E8M0(v) => HostBufferRef::F8E8M0(v),
        }
    }
}

impl<'a> HostBufferRef<'a> {
    /// The [`DType`] of the borrowed data.
    pub fn dtype(&self) -> DType {
        match self {
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
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

    /// Number of elements in the borrowed slice.
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::F8E4M3(v) => v.len(),
            Self::F6E2M3(v) => v.len(),
            Self::F6E3M2(v) => v.len(),
            Self::F4(v) => v.len(),
            Self::F8E8M0(v) => v.len(),
        }
    }

    /// `true` if the borrowed slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize into an owned [`HostBuffer`] (copies the slice).
    ///
    /// Use when a caller needs ownership and the storage backend is not
    /// itself buffer-owning (mmap-backed, pinned, remote, …). For
    /// `CpuBackendStorage` the cheaper path is `into_host_buffer`, which
    /// transfers the existing `Vec<T>` without allocating.
    pub fn to_owned(&self) -> HostBuffer {
        match self {
            Self::U8(v) => HostBuffer::U8(v.to_vec()),
            Self::I8(v) => HostBuffer::I8(v.to_vec()),
            Self::U32(v) => HostBuffer::U32(v.to_vec()),
            Self::I16(v) => HostBuffer::I16(v.to_vec()),
            Self::I32(v) => HostBuffer::I32(v.to_vec()),
            Self::I64(v) => HostBuffer::I64(v.to_vec()),
            Self::BF16(v) => HostBuffer::BF16(v.to_vec()),
            Self::F16(v) => HostBuffer::F16(v.to_vec()),
            Self::F32(v) => HostBuffer::F32(v.to_vec()),
            Self::F64(v) => HostBuffer::F64(v.to_vec()),
            Self::F8E4M3(v) => HostBuffer::F8E4M3(v.to_vec()),
            Self::F6E2M3(v) => HostBuffer::F6E2M3(v.to_vec()),
            Self::F6E3M2(v) => HostBuffer::F6E3M2(v.to_vec()),
            Self::F4(v) => HostBuffer::F4(v.to_vec()),
            Self::F8E8M0(v) => HostBuffer::F8E8M0(v.to_vec()),
        }
    }
}

/// Host-buffer conversion contract for a tensor element type.
///
/// Split off `WithDType` in B0.4 (the weld break): `WithDType` stays pure
/// vocabulary, while these `HostBuffer`/`HostBufferRef` conversions live here,
/// next to the buffer types they produce. Generic code that needs to materialize
/// a `&[T]` / `Vec<T>` into host storage bounds on `T: HostDType` (which implies
/// `WithDType`). This breaks the old `dtype <-> cpu_storage` dependency cycle.
pub trait HostDType: WithDType {
    fn cpu_storage_ref(data: &[Self]) -> HostBufferRef<'_>;
    fn to_cpu_storage_owned(data: Vec<Self>) -> HostBuffer;

    fn to_cpu_storage(data: &[Self]) -> HostBuffer {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &HostBuffer) -> Result<&[Self]>;
    fn cpu_storage_data(s: HostBuffer) -> Result<Vec<Self>>;
}

macro_rules! host_dtype {
    ($ty:ty, $dtype:ident) => {
        impl HostDType for $ty {
            fn cpu_storage_ref(data: &[Self]) -> HostBufferRef<'_> {
                HostBufferRef::$dtype(data)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> HostBuffer {
                HostBuffer::$dtype(data)
            }

            fn cpu_storage_data(s: HostBuffer) -> Result<Vec<Self>> {
                match s {
                    HostBuffer::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &HostBuffer) -> Result<&[Self]> {
                match s {
                    HostBuffer::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }
        }
    };
}

host_dtype!(u8, U8);
host_dtype!(i8, I8);
host_dtype!(u32, U32);
host_dtype!(i16, I16);
host_dtype!(i32, I32);
host_dtype!(i64, I64);
host_dtype!(f16, F16);
host_dtype!(bf16, BF16);
host_dtype!(f32, F32);
host_dtype!(f64, F64);
host_dtype!(F8E4M3, F8E4M3);

/// Dummy sub-byte float markers carry no real host storage — their conversions
/// panic / error (parity with the pre-B0.4 `WithDType` dummy impls).
macro_rules! host_dtype_dummy {
    ($ty:ty, $dtype:ident) => {
        impl HostDType for $ty {
            fn cpu_storage_ref(_data: &[Self]) -> HostBufferRef<'_> {
                panic!("{} is a dummy type and does not support storage", stringify!($ty))
            }
            fn to_cpu_storage_owned(_data: Vec<Self>) -> HostBuffer {
                panic!("{} is a dummy type and does not support storage", stringify!($ty))
            }
            fn cpu_storage_data(_s: HostBuffer) -> Result<Vec<Self>> {
                Err(Error::UnsupportedDTypeForOp(DType::$dtype, "cpu_storage_data").bt())
            }
            fn cpu_storage_as_slice(_s: &HostBuffer) -> Result<&[Self]> {
                Err(Error::UnsupportedDTypeForOp(DType::$dtype, "cpu_storage_as_slice").bt())
            }
        }
    };
}

host_dtype_dummy!(crate::F6E2M3, F6E2M3);
host_dtype_dummy!(crate::F6E3M2, F6E3M2);
host_dtype_dummy!(crate::F4, F4);
host_dtype_dummy!(crate::F8E8M0, F8E8M0);

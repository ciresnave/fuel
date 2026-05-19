/// Helper functions to plug cuda kernels in fuel.
use fuel_core_types::dtype::WithDType;
use fuel_core_types::{Layout, Result};
use baracuda_driver::DeviceBuffer;
use baracuda_types::{DeviceRepr, ValidAsZeroBits};

use crate::{CudaDevice, CudaError, WrapErr};

pub type S = crate::CudaStorageSlice;

pub trait Map1 {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &DeviceBuffer<T>,
        dev: &CudaDevice,
        layout: &Layout,
    ) -> Result<DeviceBuffer<T>>;

    fn map(&self, s: &S, d: &CudaDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::I8(s) => S::I8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I16(s) => S::I16(self.f(s, d, l)?),
            S::I32(s) => S::I32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
            S::F8E4M3(s) => S::F8E4M3(self.f(s, d, l)?),
            S::F4(_) | S::F6E2M3(_) | S::F6E3M2(_) | S::F8E8M0(_) => {
                fuel_core_types::bail!("Map1 does not uspport this dtype.");
            }
        };
        Ok(out)
    }
}

pub trait Map2 {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &DeviceBuffer<T>,
        layout1: &Layout,
        src2: &DeviceBuffer<T>,
        layout2: &Layout,
        dev: &CudaDevice,
    ) -> Result<DeviceBuffer<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &CudaDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => S::U8(self.f(s1, l1, s2, l2, d)?),
            (S::I8(s1), S::I8(s2)) => S::I8(self.f(s1, l1, s2, l2, d)?),
            (S::U32(s1), S::U32(s2)) => S::U32(self.f(s1, l1, s2, l2, d)?),
            (S::I16(s1), S::I16(s2)) => S::I16(self.f(s1, l1, s2, l2, d)?),
            (S::I32(s1), S::I32(s2)) => S::I32(self.f(s1, l1, s2, l2, d)?),
            (S::I64(s1), S::I64(s2)) => S::I64(self.f(s1, l1, s2, l2, d)?),
            (S::BF16(s1), S::BF16(s2)) => S::BF16(self.f(s1, l1, s2, l2, d)?),
            (S::F16(s1), S::F16(s2)) => S::F16(self.f(s1, l1, s2, l2, d)?),
            (S::F32(s1), S::F32(s2)) => S::F32(self.f(s1, l1, s2, l2, d)?),
            (S::F64(s1), S::F64(s2)) => S::F64(self.f(s1, l1, s2, l2, d)?),
            (S::F8E4M3(s1), S::F8E4M3(s2)) => S::F8E4M3(self.f(s1, l1, s2, l2, d)?),
            _ => Err(CudaError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}

pub trait Map3 {
    #[allow(clippy::too_many_arguments)]
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &DeviceBuffer<T>,
        layout1: &Layout,
        src2: &DeviceBuffer<T>,
        layout2: &Layout,
        src3: &DeviceBuffer<T>,
        layout3: &Layout,
        dev: &CudaDevice,
    ) -> Result<DeviceBuffer<T>>;

    #[allow(clippy::too_many_arguments)]
    fn map(
        &self,
        s1: &S,
        l1: &Layout,
        s2: &S,
        l2: &Layout,
        s3: &S,
        l3: &Layout,
        d: &CudaDevice,
    ) -> Result<S> {
        let out = match (s1, s2, s3) {
            (S::U8(s1), S::U8(s2), S::U8(s3)) => S::U8(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::U32(s1), S::U32(s2), S::U32(s3)) => S::U32(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::I64(s1), S::I64(s2), S::I64(s3)) => S::I64(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::BF16(s1), S::BF16(s2), S::BF16(s3)) => S::BF16(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F16(s1), S::F16(s2), S::F16(s3)) => S::F16(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F32(s1), S::F32(s2), S::F32(s3)) => S::F32(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F64(s1), S::F64(s2), S::F64(s3)) => S::F64(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F8E4M3(s1), S::F8E4M3(s2), S::F8E4M3(s3)) => {
                S::F8E4M3(self.f(s1, l1, s2, l2, s3, l3, d)?)
            }
            _ => Err(CudaError::InternalError("dtype mismatch in ternary op"))?,
        };
        Ok(out)
    }
}

pub trait Map2InPlace {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut DeviceBuffer<T>,
        dst_l: &Layout,
        src: &DeviceBuffer<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()>;

    fn map(
        &self,
        dst: &mut S,
        dst_l: &Layout,
        src: &S,
        src_l: &Layout,
        d: &CudaDevice,
    ) -> Result<()> {
        match (dst, src) {
            (S::U8(dst), S::U8(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::U32(dst), S::U32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I16(dst), S::I16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I32(dst), S::I32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F8E4M3(dst), S::F8E4M3(src)) => self.f(dst, dst_l, src, src_l, d),
            _ => Err(CudaError::InternalError("dtype mismatch in binary op"))?,
        }
    }
}

pub trait Map1Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(DeviceBuffer<T>) -> S>(
        &self,
        src: &DeviceBuffer<T>,
        dev: &CudaDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &CudaDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::I8(s) => self.f(s, d, l, S::I8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I16(s) => self.f(s, d, l, S::I16)?,
            S::I32(s) => self.f(s, d, l, S::I32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
            S::F8E4M3(s) => self.f(s, d, l, S::F8E4M3)?,
            S::F4(_) | S::F6E2M3(_) | S::F6E3M2(_) | S::F8E8M0(_) => {
                fuel_core_types::bail!("Map1 does not uspport this dtype.");
            }
        };
        Ok(out)
    }
}

pub trait Map2Any {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src1: &DeviceBuffer<T>,
        layout1: &Layout,
        src2: &DeviceBuffer<T>,
        layout2: &Layout,
        dev: &CudaDevice,
    ) -> Result<S>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &CudaDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::U32(s1), S::U32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::I64(s1), S::I64(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::BF16(s1), S::BF16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F16(s1), S::F16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F32(s1), S::F32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F64(s1), S::F64(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F8E4M3(s1), S::F8E4M3(s2)) => self.f(s1, l1, s2, l2, d)?,
            _ => Err(CudaError::InternalError("dtype mismatch in binary op")).w()?,
        };
        Ok(out)
    }
}

use crate::{Result, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Vec<u32> {
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        sort_indexes
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use baracuda_driver::DeviceBuffer as CudaSlice;
    use baracuda_types::{DeviceRepr, ValidAsZeroBits};
    use fuel_cuda_backend::baracuda_kernels_sys as sys;
    use crate::cuda_backend::CudaStorageSlice as S;
    use crate::CudaDevice;
    use fuel_core_types::dtype::{DType, WithDType};

    /// Baracuda `argsort_<dtype>_run` (block-bitonic, row_len ≤ 1024)
    /// and `argsort_<dtype>_big_run` (multi-block radix, row_len > 1024)
    /// share this FFI shape; `big_run` additionally requires a workspace
    /// buffer sized via `argsort_<dtype>_big_workspace_size`.
    type ArgsortRun = unsafe extern "C" fn(
        batch: i32,
        row_len: i32,
        descending: i32,
        x: *const std::ffi::c_void,
        y_idx: *mut std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    type ArgsortBigWorkspaceSize = unsafe extern "C" fn(batch: i32, row_len: i32) -> usize;

    fn pick_argsort_small(dt: DType) -> Option<ArgsortRun> {
        Some(match dt {
            DType::F32 => sys::baracuda_kernels_argsort_f32_run,
            DType::F64 => sys::baracuda_kernels_argsort_f64_run,
            DType::I32 => sys::baracuda_kernels_argsort_i32_run,
            DType::I64 => sys::baracuda_kernels_argsort_i64_run,
            DType::U8  => sys::baracuda_kernels_argsort_u8_run,
            DType::I8  => sys::baracuda_kernels_argsort_i8_run,
            DType::U32 => sys::baracuda_kernels_argsort_u32_run,
            DType::I16 => sys::baracuda_kernels_argsort_i16_run,
            DType::BF16 => sys::baracuda_kernels_argsort_bf16_run,
            DType::F16  => sys::baracuda_kernels_argsort_f16_run,
            DType::F8E4M3 => sys::baracuda_kernels_argsort_fp8e4m3_run,
            _ => return None,
        })
    }

    fn pick_argsort_big(dt: DType) -> Option<(ArgsortRun, ArgsortBigWorkspaceSize)> {
        Some(match dt {
            DType::F32 => (
                sys::baracuda_kernels_argsort_f32_big_run,
                sys::baracuda_kernels_argsort_f32_big_workspace_size,
            ),
            DType::F64 => (
                sys::baracuda_kernels_argsort_f64_big_run,
                sys::baracuda_kernels_argsort_f64_big_workspace_size,
            ),
            DType::I32 => (
                sys::baracuda_kernels_argsort_i32_big_run,
                sys::baracuda_kernels_argsort_i32_big_workspace_size,
            ),
            _ => return None,
        })
    }

    impl crate::cuda_backend::Map1Any for ArgSort {
        fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
            &self,
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &crate::Layout,
            _wrap: W,
        ) -> Result<S> {
            let slice = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let ncols = self.last_dim;
            let nrows = elem_count / ncols;
            let descending = if self.asc { 0 } else { 1 };
            // Baracuda writes i32 indices into y_idx; we expose u32 to
            // Fuel callers (same byte width, same bit pattern for
            // values in [0, ncols) since ncols always fits i32).
            let dst = unsafe { dev.alloc::<u32>(elem_count)? };
            let stream = dev.stream().as_raw() as *mut std::ffi::c_void;
            let x_ptr = slice.as_raw().0 as *const std::ffi::c_void;
            let y_ptr = dst.as_raw().0 as *mut std::ffi::c_void;

            if ncols <= 1024 {
                let run = pick_argsort_small(T::DTYPE).ok_or_else(|| {
                    crate::Error::Msg(format!(
                        "argsort: unsupported dtype {:?}", T::DTYPE
                    ))
                })?;
                let status = unsafe {
                    run(nrows as i32, ncols as i32, descending,
                        x_ptr, y_ptr,
                        std::ptr::null_mut(), 0, stream)
                };
                fuel_cuda_backend::baracuda::status::check(status, "argsort_small")?;
            } else {
                // Multi-block radix path for row_len > 1024.
                let (run, ws_size) = pick_argsort_big(T::DTYPE).ok_or_else(|| {
                    crate::Error::Msg(format!(
                        "argsort: row_len {ncols} > 1024 needs `_big` variant; \
                         baracuda alpha.54 ships F32/F64/I32 for `_big`, got {:?}",
                        T::DTYPE
                    ))
                })?;
                let ws_bytes = unsafe { ws_size(nrows as i32, ncols as i32) };
                let ws_buf = if ws_bytes > 0 {
                    Some(unsafe { dev.alloc::<u8>(ws_bytes)? })
                } else {
                    None
                };
                let ws_ptr = ws_buf
                    .as_ref()
                    .map(|b| b.as_raw().0 as *mut std::ffi::c_void)
                    .unwrap_or(std::ptr::null_mut());
                let status = unsafe {
                    run(nrows as i32, ncols as i32, descending,
                        x_ptr, y_ptr,
                        ws_ptr, ws_bytes, stream)
                };
                fuel_cuda_backend::baracuda::status::check(status, "argsort_big")?;
            }
            dev.synchronize()?;
            Ok(S::U32(dst))
        }
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn fwd(
        &self,
        storage: &dyn crate::dyn_backend::DynBackendStorage,
        layout: &crate::Layout,
    ) -> Result<(Box<dyn crate::dyn_backend::DynBackendStorage>, crate::Shape)> {
        if let Some(cpu) = storage
            .as_any()
            .downcast_ref::<fuel_cpu_backend::dyn_impl::CpuStorage>()
        {
            let sort_indexes = match &cpu.0 {
                crate::HostBuffer::U8(vs) => self.asort(vs, layout),
                crate::HostBuffer::I8(vs) => self.asort(vs, layout),
                crate::HostBuffer::U32(vs) => self.asort(vs, layout),
                crate::HostBuffer::I16(vs) => self.asort(vs, layout),
                crate::HostBuffer::I32(vs) => self.asort(vs, layout),
                crate::HostBuffer::I64(vs) => self.asort(vs, layout),
                crate::HostBuffer::BF16(vs) => self.asort(vs, layout),
                crate::HostBuffer::F16(vs) => self.asort(vs, layout),
                crate::HostBuffer::F32(vs) => self.asort(vs, layout),
                crate::HostBuffer::F64(vs) => self.asort(vs, layout),
                crate::HostBuffer::F8E4M3(vs) => self.asort(vs, layout),
                // Dummy types don't support sorting
                crate::HostBuffer::F6E2M3(_) => {
                    return Err(
                        crate::Error::UnsupportedDTypeForOp(crate::DType::F6E2M3, "argsort").bt(),
                    )
                }
                crate::HostBuffer::F6E3M2(_) => {
                    return Err(
                        crate::Error::UnsupportedDTypeForOp(crate::DType::F6E3M2, "argsort").bt(),
                    )
                }
                crate::HostBuffer::F4(_) => {
                    return Err(
                        crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt(),
                    )
                }
                crate::HostBuffer::F8E8M0(_) => {
                    return Err(
                        crate::Error::UnsupportedDTypeForOp(crate::DType::F8E8M0, "argsort").bt(),
                    )
                }
            };
            let out = crate::HostBuffer::U32(sort_indexes);
            return Ok((
                Box::new(fuel_cpu_backend::dyn_impl::CpuStorage(out)),
                layout.shape().into(),
            ));
        }

        #[cfg(feature = "cuda")]
        if let Some(cuda) = storage
            .as_any()
            .downcast_ref::<fuel_cuda_backend::CudaStorage>()
        {
            use crate::cuda_backend::Map1Any;
            let dev = cuda.device();
            let slice = self.map(&cuda.slice, dev, layout)?;
            let dst = crate::cuda_backend::CudaStorage {
                slice,
                device: dev.clone(),
            };
            return Ok((
                Box::new(dst),
                layout.shape().clone(),
            ));
        }

        #[cfg(feature = "metal")]
        if let Some(metal) = storage
            .as_any()
            .downcast_ref::<fuel_metal_backend::MetalStorage>()
        {
            use crate::DType;

            let name = {
                if self.asc {
                    match metal.dtype() {
                        DType::BF16 => "asort_asc_bf16",
                        DType::F16 => "asort_asc_f16",
                        DType::F32 => "asort_asc_f32",
                        DType::F64 => "asort_asc_f64",
                        DType::U8 => "asort_asc_u8",
                        DType::U32 => "asort_asc_u32",
                        DType::I16 => "asort_asc_i16",
                        DType::I32 => "asort_asc_i32",
                        DType::I64 => "asort_asc_i64",
                        DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                            return Err(crate::Error::UnsupportedDTypeForOp(
                                metal.dtype(),
                                "argsort",
                            )
                            .bt())
                        }
                    }
                } else {
                    match metal.dtype() {
                        DType::BF16 => "asort_desc_bf16",
                        DType::F16 => "asort_desc_f16",
                        DType::F32 => "asort_desc_f32",
                        DType::F64 => "asort_desc_f64",
                        DType::U8 => "asort_desc_u8",
                        DType::U32 => "asort_desc_u32",
                        DType::I16 => "asort_desc_i16",
                        DType::I32 => "asort_desc_i32",
                        DType::I64 => "asort_desc_i64",
                        DType::F8E4M3 => crate::bail!("Metal device does not yet support F8E4M3."),
                        DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                            return Err(crate::Error::UnsupportedDTypeForOp(
                                metal.dtype(),
                                "argsort",
                            )
                            .bt())
                        }
                    }
                }
            };
            let device = metal.device();
            let kernels = device.kernels();
            let command_encoder = device.command_encoder()?;
            let el = layout.shape().elem_count();
            let ncols = self.last_dim;
            let nrows = el / ncols;
            let src = crate::metal_backend::buffer_o(metal.buffer(), layout, metal.dtype());
            let dst = device.new_buffer(el, DType::U32, "asort")?;
            let mut ncols_pad = 1;
            while ncols_pad < ncols {
                ncols_pad *= 2;
            }
            fuel_metal_kernels::call_arg_sort(
                device.metal_device(),
                &command_encoder,
                kernels,
                name,
                nrows,
                ncols,
                ncols_pad,
                src,
                &dst,
            )
            .map_err(crate::Error::wrap)?;
            let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
            return Ok((
                Box::new(dst),
                layout.shape().clone(),
            ));
        }

        Err(crate::Error::Msg("argsort: unsupported backend".into()).bt())
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort { asc, last_dim })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}

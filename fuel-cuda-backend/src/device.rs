use fuel_core_types::HostBufferRef;
use fuel_core_types::dtype::WithDType;
use fuel_core_types::{HostBuffer, DType, Layout, Result, Shape};
use fuel_cuda_kernels as kernels;
use baracuda_curand::RngKind;
use baracuda_driver::{DeviceBuffer, Dim3, Function, LaunchBuilder};
use float8::F8E4M3;
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

// ---------------------------------------------------------------------------
// cudarc-compatible launch shim
// ---------------------------------------------------------------------------
//
// baracuda's [`LaunchBuilder`] is a value-style builder: every call consumes
// `self` and returns `Self`. Fuel's CUDA code (inherited from candle-cuda)
// uses cudarc's mutation-style `LaunchArgs` where calls take `&mut self` —
// ~100 launch sites across storage.rs / dyn_impl.rs / downstream crates
// depend on that shape.
//
// The `LaunchArgs` wrapper below preserves the mutation semantics on top of
// baracuda's value-style builder: an internal `Option<LaunchBuilder>`
// lets each `arg()` call `take()` the current builder, call its `.arg()`,
// and stash the returned builder back. Call sites read exactly like the
// cudarc original:
//
// ```ignore
// let mut builder = func.builder();
// builder.arg(&src);
// builder.arg(&mut out);
// builder_arg!(builder, n_cols as i32);
// unsafe { builder.launch(cfg) }.w()?;
// ```

/// cudarc-shaped launch-args builder layered over baracuda's
/// `LaunchBuilder`. See module doc for rationale.
pub struct LaunchArgs<'f> {
    inner: Option<LaunchBuilder<'f>>,
}

impl<'f> LaunchArgs<'f> {
    pub(crate) fn new(b: LaunchBuilder<'f>) -> Self {
        Self { inner: Some(b) }
    }

    /// Append an argument; preserves cudarc's `&mut self` return for
    /// chained or statement usage.
    pub fn arg<K: baracuda_types::KernelArg>(&mut self, arg: K) -> &mut Self {
        let b = self.inner.take().expect("LaunchArgs already launched");
        self.inner = Some(b.arg(arg));
        self
    }

    /// Submit the kernel. Consumes the builder (matches cudarc's
    /// drop-after-launch behavior — don't reuse the `LaunchArgs` after
    /// calling this).
    ///
    /// # Safety
    ///
    /// Same obligations as `baracuda_driver::LaunchBuilder::launch`:
    /// argument count / types must match the kernel signature,
    /// pointer-valued args must be live for the duration of
    /// submission, and grid/block dimensions must fit the device.
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> baracuda_driver::Result<()> {
        let b = self
            .inner
            .take()
            .expect("LaunchArgs already launched")
            .grid(Dim3 { x: cfg.grid_dim.0, y: cfg.grid_dim.1, z: cfg.grid_dim.2 })
            .block(Dim3 { x: cfg.block_dim.0, y: cfg.block_dim.1, z: cfg.block_dim.2 })
            .shared_mem_bytes(cfg.shared_mem_bytes);
        unsafe { b.launch() }
    }
}

/// cudarc-shaped launch config. Populated with the same `grid_dim /
/// block_dim / shared_mem_bytes` fields fuel's launch sites used
/// against cudarc. `LaunchArgs::launch` translates into baracuda's
/// `.grid().block().shared_mem_bytes()` chain.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Grid = ceil(n / 256), block = 256. Matches cudarc's
    /// `for_num_elems` helper for 1-D elementwise kernels.
    pub fn for_num_elems(n: u32) -> Self {
        const BLOCK: u32 = 256;
        let grid = n.div_ceil(BLOCK).max(1);
        Self {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(baracuda_curand::Generator);
unsafe impl Send for CudaRng {}

pub struct ModuleStore {
    mdls: [Option<Arc<baracuda_driver::Module>>; kernels::ALL_IDS.len()],
}

/// cuBLAS handle wrapper that is `Sync` via an unsafe promise that the caller
/// serialises concurrent use (per NVIDIA's per-thread handle contract).
/// Fuel's graph executor serialises GPU work onto a single dispatch thread,
/// so this holds at the Fuel layer; the wrapper exists because baracuda's
/// own `Handle` is `Send` but intentionally `!Sync`.
pub struct CublasHandle(pub baracuda_cublas::Handle);
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}
impl std::ops::Deref for CublasHandle {
    type Target = baracuda_cublas::Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone)]
pub struct CudaDevice {
    // Field order matters for Drop. Rust drops struct fields in
    // declaration order, and baracuda's Stream / cuBLAS Handle / cuRAND
    // Generator / loaded Modules all hold raw CUDA resources that must
    // be destroyed *before* the owning Context is torn down (otherwise
    // the driver access-violates on process exit). Keep `context` last.
    id: DeviceId,
    seed_value: Arc<RwLock<u64>>,
    curand: Arc<Mutex<CudaRng>>,
    pub(crate) blas: Arc<CublasHandle>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<baracuda_driver::Module>>>>,
    stream: Arc<baracuda_driver::Stream>,
    context: Arc<baracuda_driver::Context>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    /// Allocate a new async device buffer (uninitialized).
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: baracuda_types::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::new_async(&self.context, len, &self.stream).w()
    }

    /// Allocate a new device buffer, zeroed. (Baracuda's `zeros` is
    /// synchronous; the result is usable on any stream afterward.)
    pub fn alloc_zeros<T: baracuda_types::DeviceRepr + baracuda_types::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::zeros(&self.context, len).w()
    }

    /// Host → device copy, async on this device's default stream.
    /// Destination can be a DeviceBuffer or a DeviceSliceMut — the shared
    /// behaviour is "a contiguous device-side region with a raw pointer and
    /// a length", which we reach via the raw sys API.
    pub fn memcpy_htod<T: baracuda_types::DeviceRepr>(
        &self,
        src: &[T],
        dst: &mut baracuda_driver::DeviceSliceMut<T>,
    ) -> Result<()> {
        use baracuda_cuda_sys::{driver, CUresult};
        assert_eq!(src.len(), dst.len());
        let bytes = src.len() * std::mem::size_of::<T>();
        let d = driver().map_err(|_| CudaError::InternalError("cuda driver load")).w()?;
        let cu = d
            .cu_memcpy_htod_async()
            .map_err(|_| CudaError::InternalError("cuMemcpyHtoDAsync not available"))
            .w()?;
        let r = unsafe {
            cu(
                dst.as_raw(),
                src.as_ptr() as *const std::ffi::c_void,
                bytes,
                self.stream.as_raw(),
            )
        };
        if r == CUresult::SUCCESS {
            Ok(())
        } else {
            Err(CudaError::InternalError("cuMemcpyHtoDAsync failed").into())
        }
    }

    /// Device → host blocking copy — returns an owned Vec.
    pub fn clone_dtoh<T: baracuda_types::DeviceRepr + Default + Clone>(
        &self,
        src: &baracuda_driver::DeviceSlice<T>,
    ) -> Result<Vec<T>> {
        use baracuda_cuda_sys::{driver, CUresult};
        let mut out = vec![T::default(); src.len()];
        let bytes = src.len() * std::mem::size_of::<T>();
        let d = driver().map_err(|_| CudaError::InternalError("cuda driver load")).w()?;
        let cu = d
            .cu_memcpy_dtoh()
            .map_err(|_| CudaError::InternalError("cuMemcpyDtoH not available"))
            .w()?;
        let r = unsafe { cu(out.as_mut_ptr() as *mut std::ffi::c_void, src.as_raw(), bytes) };
        if r == CUresult::SUCCESS {
            Ok(out)
        } else {
            Err(CudaError::InternalError("cuMemcpyDtoH failed").into())
        }
    }

    /// Device → device copy, async on this device's default stream.
    /// Accepts any pair of baracuda device slices (buffer, slice, slice_mut).
    pub fn memcpy_dtod<T: baracuda_types::DeviceRepr>(
        &self,
        src: &baracuda_driver::DeviceSlice<T>,
        dst: &mut baracuda_driver::DeviceSliceMut<T>,
    ) -> Result<()> {
        use baracuda_cuda_sys::{driver, CUresult};
        assert_eq!(src.len(), dst.len());
        let bytes = src.len() * std::mem::size_of::<T>();
        let d = driver().map_err(|_| CudaError::InternalError("cuda driver load")).w()?;
        let cu = d
            .cu_memcpy_dtod_async()
            .map_err(|_| CudaError::InternalError("cuMemcpyDtoDAsync not available"))
            .w()?;
        let r = unsafe { cu(dst.as_raw(), src.as_raw(), bytes, self.stream.as_raw()) };
        if r == CUresult::SUCCESS {
            Ok(())
        } else {
            Err(CudaError::InternalError("cuMemcpyDtoDAsync failed").into())
        }
    }

    /// Device → device copy into a freshly allocated buffer on this device.
    pub fn clone_dtod<T: baracuda_types::DeviceRepr + baracuda_types::ValidAsZeroBits>(
        &self,
        src: &DeviceBuffer<T>,
    ) -> Result<DeviceBuffer<T>> {
        let dst = DeviceBuffer::<T>::zeros(&self.context, src.len()).w()?;
        src.copy_to_device_async(&dst, &self.stream).w()?;
        Ok(dst)
    }

    /// Device → host blocking copy (dst must have `len == src.len()`).
    pub fn memcpy_dtoh<T: baracuda_types::DeviceRepr>(
        &self,
        src: &baracuda_driver::DeviceSlice<T>,
        dst: &mut [T],
    ) -> Result<()> {
        use baracuda_cuda_sys::{driver, CUresult};
        assert_eq!(src.len(), dst.len());
        let bytes = src.len() * std::mem::size_of::<T>();
        let d = driver().map_err(|_| CudaError::InternalError("cuda driver load")).w()?;
        let cu = d
            .cu_memcpy_dtoh()
            .map_err(|_| CudaError::InternalError("cuMemcpyDtoH not available"))
            .w()?;
        let r = unsafe { cu(dst.as_mut_ptr() as *mut std::ffi::c_void, src.as_raw(), bytes) };
        if r == CUresult::SUCCESS {
            Ok(())
        } else {
            Err(CudaError::InternalError("cuMemcpyDtoH failed").into())
        }
    }

    /// Host → device (new buffer): allocate + copy in one call.
    pub fn clone_htod<T: baracuda_types::DeviceRepr>(
        &self,
        src: &[T],
    ) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::from_slice(&self.context, src).w()
    }
}

pub struct CudaFunc {
    func: Function,
    stream: Arc<baracuda_driver::Stream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = Function;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> Function {
        self.func
    }
}

/// Push one or more args onto a `LaunchArgs` as a statement. Handles
/// the "bind temporary to extend lifetime, then push by reference"
/// idiom that cudarc's PushKernelArg requires.
#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    /// Start a launch builder pre-bound to this function's stream.
    /// Returns a cudarc-shaped `LaunchArgs` layer — see its docs.
    pub fn builder(&self) -> LaunchArgs<'_> {
        LaunchArgs::new(self.func.launch().stream(&self.stream))
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<baracuda_driver::Stream> {
        self.stream.clone()
    }

    /// Event-tracking toggle. Was a cudarc-specific knob that disabled
    /// per-tensor event bookkeeping on shared-stream workloads. Baracuda
    /// doesn't expose an equivalent — its `DeviceBuffer` lifetime model
    /// doesn't rely on events the same way. This is now a no-op kept
    /// only for source compatibility with existing callers
    /// (`fuel-examples/llama2-c/main.rs`); remove once the caller drops
    /// the API call.
    ///
    /// # Safety
    ///
    /// No longer does anything; marked `unsafe` purely to preserve the
    /// old signature.
    pub unsafe fn disable_event_tracking(&self) {}

    /// Always returns `true` for the same reason `disable_event_tracking`
    /// is a no-op now — baracuda's `DeviceBuffer` model is implicitly
    /// stream-ordered and doesn't expose the flag.
    pub fn is_event_tracking(&self) -> bool { true }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: fuel_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        fuel_ug::cuda::code_gen::r#gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = baracuda_nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        // `compile_with` returns the PTX text directly (String).
        let ptx = baracuda_nvrtc::Program::compile_with(&cuda_code, func_name, &opts).w()?;
        let module = baracuda_driver::Module::load_ptx(&self.context, &ptx).w()?;
        let func = module.get_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.get_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = Arc::new(baracuda_driver::Module::load_ptx(&self.context, ptx).w()?);
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.get_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.get_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = Arc::new(baracuda_driver::Module::load_ptx(&self.context, mdl.ptx()).w()?);
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.get_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<CublasHandle> {
        self.blas.clone()
    }

    /// Borrow the device's default stream. Used by CUTLASS plan launches
    /// (and any future safe-API CUDA library) that take `&Stream`. The
    /// stream is shared across all kernel launches for this device.
    pub fn stream(&self) -> &baracuda_driver::Stream {
        &self.stream
    }

    /// Borrow the underlying baracuda [`Context`](baracuda_driver::Context).
    ///
    /// Used by crates like [`crate::pinned`] that build CUDA host
    /// allocations (pinned memory) or other context-scoped resources on
    /// top of the device's existing context.
    pub fn context_ref(&self) -> &baracuda_driver::Context {
        &self.context
    }
}

impl CudaDevice {
    /// Construct a CudaDevice with a freshly-created stream.
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let device = baracuda_driver::Device::get(ordinal as u32).w()?;
        let context = baracuda_driver::Context::new(&device).w()?;
        let stream = baracuda_driver::Stream::new(&context).w()?;
        Self::new_from(context, stream)
    }

    /// Construct a CudaDevice with the default stream on a fresh
    /// context. Baracuda doesn't expose a "default stream getter" — we
    /// create one explicitly per context, which is what cudarc's
    /// default-stream path effectively did underneath.
    pub fn new(ordinal: usize) -> Result<Self> {
        Self::new_with_stream(ordinal)
    }

    fn new_from(
        context: baracuda_driver::Context,
        stream: baracuda_driver::Stream,
    ) -> Result<Self> {
        let blas = baracuda_cublas::Handle::new().w()?;
        blas.set_stream(&stream).w()?;
        let mut curand = baracuda_curand::Generator::new(RngKind::Default).w()?;
        curand.seed(299792458).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context: Arc::new(context),
            stream: Arc::new(stream),
            blas: Arc::new(CublasHandle(blas)),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }

    pub fn set_seed(&self, seed: u64) -> Result<()> {
        // Baracuda's Generator has a direct `set_seed` — no need to
        // rebuild the generator the way cudarc required.
        let mut curand = self.curand.lock().unwrap();
        curand.0.seed(seed).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    pub fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    pub fn location(&self) -> fuel_core_types::DeviceLocation {
        fuel_core_types::DeviceLocation::Cuda {
            gpu_id: self.context.device().ordinal() as usize,
        }
    }

    pub fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    pub fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::I8 => {
                let data = self.alloc_zeros::<i8>(elem_count)?;
                CudaStorageSlice::I8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::I8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.uniform_f64(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use crate::utils::Map1;
            let layout = Layout::contiguous(shape);
            crate::storage::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/fuel/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::I8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.normal_f64(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = unsafe { self.alloc::<u8>(elem_count) }?;
                CudaStorageSlice::U8(data)
            }
            DType::I8 => {
                let data = unsafe { self.alloc::<i8>(elem_count) }?;
                CudaStorageSlice::I8(data)
            }
            DType::U32 => {
                let data = unsafe { self.alloc::<u32>(elem_count) }?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = unsafe { self.alloc::<i16>(elem_count) }?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = unsafe { self.alloc::<i32>(elem_count) }?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = unsafe { self.alloc::<i64>(elem_count) }?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = unsafe { self.alloc::<bf16>(elem_count) }?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = unsafe { self.alloc::<f16>(elem_count) }?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = unsafe { self.alloc::<f32>(elem_count) }?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = unsafe { self.alloc::<f64>(elem_count) }?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = unsafe { self.alloc::<F8E4M3>(elem_count) }?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn storage_from_slice<T: WithDType>(&self, s: &[T]) -> Result<CudaStorage> {
        let slice = match T::cpu_storage_ref(s) {
            HostBufferRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            HostBufferRef::I8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I8(data)
            }
            HostBufferRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            HostBufferRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            HostBufferRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            HostBufferRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            HostBufferRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            HostBufferRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            HostBufferRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            HostBufferRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            HostBufferRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            HostBufferRef::F4(_)
            | HostBufferRef::F6E2M3(_)
            | HostBufferRef::F6E3M2(_)
            | HostBufferRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn storage_from_cpu_storage(&self, storage: &HostBuffer) -> Result<CudaStorage> {
        let slice = match storage {
            HostBuffer::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            HostBuffer::I8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I8(data)
            }
            HostBuffer::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            HostBuffer::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            HostBuffer::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            HostBuffer::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            HostBuffer::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            HostBuffer::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            HostBuffer::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            HostBuffer::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            HostBuffer::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            HostBuffer::F4(_)
            | HostBuffer::F6E2M3(_)
            | HostBuffer::F6E3M2(_)
            | HostBuffer::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn storage_from_cpu_storage_owned(&self, storage: HostBuffer) -> Result<CudaStorage> {
        let slice = match storage {
            HostBuffer::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            HostBuffer::I8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I8(data)
            }
            HostBuffer::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            HostBuffer::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            HostBuffer::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            HostBuffer::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            HostBuffer::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            HostBuffer::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            HostBuffer::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            HostBuffer::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            HostBuffer::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            HostBuffer::F4(_)
            | HostBuffer::F6E2M3(_)
            | HostBuffer::F6E3M2(_)
            | HostBuffer::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(fuel_core_types::Error::wrap)?;
        Ok(())
    }
}

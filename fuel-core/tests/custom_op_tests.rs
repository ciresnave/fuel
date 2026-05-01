//! Integration tests for the Phase-7 [`CustomOp1`] / [`InplaceOp1`] /
//! [`UgIOp1`] public API.
//!
//! These exercise the full surface an external consumer of `fuel-core`
//! has to implement: the `fwd(&dyn DynBackendStorage, &Layout)` entry
//! point, downcasting through `fuel_cpu_backend::dyn_impl::
//! CpuBackendStorage`, and the forward + backward + inplace variants.
//! For the downcast pattern in a real fuel-core op, see
//! [`fuel_core::sort::ArgSort`].

use fuel_core::dyn_backend::DynBackendStorage;
use fuel_core::test_utils::to_vec1_round;
use fuel_core::{HostBuffer, CustomOp1, DType, Device, Error, InplaceOp1, Layout, Result, Shape, Tensor};
use fuel_cpu_backend::dyn_impl::CpuStorage as CpuBackendStorage;

fn fwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        v
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        (v.exp() - T::one()) * alpha
    }
}

struct Elu {
    alpha: f64,
}

impl CustomOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn fwd(
        &self,
        storage: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let cpu = storage
            .as_any()
            .downcast_ref::<CpuBackendStorage>()
            .ok_or_else(|| Error::Msg(format!("{}: CPU storage required", CustomOp1::name(self))).bt())?;
        let alpha = self.alpha;
        let out = match &cpu.0 {
            HostBuffer::F8E4M3(s) => HostBuffer::F8E4M3(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| fwd(v, alpha)),
            ),
            HostBuffer::BF16(s) => HostBuffer::BF16(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| fwd(v, alpha)),
            ),
            HostBuffer::F16(s) => HostBuffer::F16(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| fwd(v, alpha)),
            ),
            HostBuffer::F32(s) => HostBuffer::F32(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| fwd(v, alpha)),
            ),
            HostBuffer::F64(s) => HostBuffer::F64(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| fwd(v, alpha)),
            ),
            s => return Err(Error::UnsupportedDTypeForOp(s.dtype(), CustomOp1::name(self)).bt()),
        };
        Ok((Box::new(CpuBackendStorage(out)), layout.shape().clone()))
    }
}

#[test]
fn custom_op1_no_backward() -> Result<()> {
    let cpu = &Device::cpu();
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    let elu_t = t.apply_op1_no_bwd(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&elu_t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

/// Derivative of [`fwd`]: `1` on the positive branch, `alpha * exp(v)`
/// on the negative branch. Used by [`EluBackward`] for the custom
/// backward pass of [`EluWithBackward`].
fn bwd<T: num_traits::Float>(v: T, alpha: f64) -> T {
    if v.is_sign_positive() {
        T::one()
    } else {
        let alpha = T::from(alpha).unwrap_or(T::nan());
        v.exp() * alpha
    }
}

struct EluBackward {
    alpha: f64,
}

impl CustomOp1 for EluBackward {
    fn name(&self) -> &'static str {
        "elu-bwd"
    }

    fn fwd(
        &self,
        storage: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        let cpu = storage
            .as_any()
            .downcast_ref::<CpuBackendStorage>()
            .ok_or_else(|| Error::Msg(format!("{}: CPU storage required", self.name())).bt())?;
        let alpha = self.alpha;
        let out = match &cpu.0 {
            HostBuffer::F8E4M3(s) => HostBuffer::F8E4M3(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| bwd(v, alpha)),
            ),
            HostBuffer::BF16(s) => HostBuffer::BF16(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| bwd(v, alpha)),
            ),
            HostBuffer::F16(s) => HostBuffer::F16(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| bwd(v, alpha)),
            ),
            HostBuffer::F32(s) => HostBuffer::F32(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| bwd(v, alpha)),
            ),
            HostBuffer::F64(s) => HostBuffer::F64(
                fuel_cpu_backend::utils::unary_map(s, layout, |v| bwd(v, alpha)),
            ),
            s => return Err(Error::UnsupportedDTypeForOp(s.dtype(), self.name()).bt()),
        };
        Ok((Box::new(CpuBackendStorage(out)), layout.shape().clone()))
    }
}

struct EluWithBackward(Elu);

impl EluWithBackward {
    fn new(alpha: f64) -> Self {
        Self(Elu { alpha })
    }
}

impl CustomOp1 for EluWithBackward {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn fwd(
        &self,
        storage: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        CustomOp1::fwd(&self.0, storage, layout)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        let alpha = self.0.alpha;
        let bwd = arg.apply_op1(EluBackward { alpha })?;
        Ok(Some(grad_res.mul(&bwd)?))
    }
}

#[test]
fn custom_op1_with_backward() -> Result<()> {
    let cpu = &Device::cpu();
    let t = fuel_core::Var::new(&[-2f32, 0f32, 2f32], cpu)?;
    let elu_t = t.apply_op1(EluWithBackward::new(2.))?;
    assert_eq!(to_vec1_round(&elu_t, 4)?, &[-1.7293, 0.0, 2.0]);

    let grads = elu_t.backward()?;
    let grad_x = grads.get(&t).unwrap();
    assert_eq!(to_vec1_round(grad_x, 4)?, [0.2707, 1.0, 1.0]);

    Ok(())
}

impl InplaceOp1 for Elu {
    fn name(&self) -> &'static str {
        "elu"
    }

    fn fwd(&self, storage: &mut dyn DynBackendStorage, _layout: &Layout) -> Result<()> {
        let cpu = storage
            .as_any_mut()
            .downcast_mut::<CpuBackendStorage>()
            .ok_or_else(|| Error::Msg(format!("{}: CPU storage required", <Self as InplaceOp1>::name(self))).bt())?;
        let alpha = self.alpha;
        match &mut cpu.0 {
            HostBuffer::F8E4M3(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            HostBuffer::BF16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            HostBuffer::F16(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            HostBuffer::F32(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            HostBuffer::F64(s) => s.iter_mut().for_each(|v| *v = fwd(*v, alpha)),
            s => fuel_core::bail!("unsupported dtype {:?} for inplace elu", s.dtype()),
        }
        Ok(())
    }
}

#[test]
fn inplace_op1() -> Result<()> {
    let cpu = &Device::cpu();
    let t = Tensor::arange(0u32, 12u32, cpu)?.to_dtype(DType::F32)?;
    let t = (t - 5.)?;
    t.inplace_op1(&Elu { alpha: 1. })?;
    assert_eq!(
        to_vec1_round(&t, 4)?,
        &[-0.9933, -0.9817, -0.9502, -0.8647, -0.6321, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    Ok(())
}

#[cfg(all(feature = "ug", any(feature = "cuda", feature = "metal")))]
#[allow(clippy::approx_constant)]
#[test]
fn ug_op() -> Result<()> {
    let kernel = {
        use fuel_ug::lang::op;

        let layout = fuel_ug::Layout::from_shape(&[12]);
        let ptr = op::Arg::ptr(fuel_ug::DType::F32);
        let src = op::load(ptr.id(), layout.clone(), fuel_ug::DType::F32)?;
        let src = op::unary(op::UnaryOp::Exp, src)?;
        let st = op::store(ptr.id(), layout, src)?;
        let kernel = op::Kernel::new("exp".to_string(), vec![ptr], vec![st]);
        let opts: fuel_ug::lower_op::Opts = Default::default();
        kernel.lower(&opts)?
    };
    let device = if fuel_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else if fuel_core::utils::metal_is_available() {
        Device::new_metal(0)?
    } else {
        fuel_core::bail!("metal/cuda is mandatory for this test")
    };
    let op = fuel_core::UgIOp1::new("test", kernel, &device)?;
    let t = Tensor::arange(0u32, 12u32, &device)?.to_dtype(DType::F32)?;
    t.inplace_op1(&op)?;
    assert_eq!(
        to_vec1_round(&t, 2)?,
        &[
            1.0, 2.72, 7.39, 20.09, 54.6, 148.41, 403.43, 1096.63, 2980.96, 8103.08, 22026.47,
            59874.13
        ]
    );
    Ok(())
}

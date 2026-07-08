use fuel_core::{DType, Result, Tensor};

struct TmpFile(std::path::PathBuf);

impl TmpFile {
    fn create(base: &str) -> TmpFile {
        let filename = std::env::temp_dir().join(format!(
            "fuel-{}-{}-{:?}",
            base,
            std::process::id(),
            std::thread::current().id(),
        ));
        TmpFile(filename)
    }
}

impl std::convert::AsRef<std::path::Path> for TmpFile {
    fn as_ref(&self) -> &std::path::Path {
        self.0.as_path()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap()
    }
}

#[test]
fn npy() -> Result<()> {
    let npy = Tensor::read_npy("tests/test.npy")?;
    assert_eq!(
        npy.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    Ok(())
}

#[test]
fn npz() -> Result<()> {
    // `tests/test.npz` was lost in an old project rename, so this round-trips the
    // fixture through `write_npz`/`read_npz` instead of relying on a checked-in file.
    let tmp_file = TmpFile::create("npz");
    let x = Tensor::arange(0f32, 10f32, &fuel_core::Device::cpu())?;
    let x_plus_one = Tensor::arange(1f32, 11f32, &fuel_core::Device::cpu())?;
    Tensor::write_npz(&[("x", &x), ("x_plus_one", &x_plus_one)], &tmp_file)?;

    let npz = Tensor::read_npz(&tmp_file)?;
    assert_eq!(npz.len(), 2);
    assert_eq!(npz[0].0, "x");
    assert_eq!(npz[1].0, "x_plus_one");
    assert_eq!(
        npz[1].1.to_dtype(DType::U8)?.to_vec1::<u8>()?,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
    Ok(())
}

#[test]
fn safetensors() -> Result<()> {
    use fuel_core::safetensors::Load;

    let tmp_file = TmpFile::create("st");
    let t = Tensor::arange(0f32, 24f32, &fuel_core::Device::cpu())?;
    t.save_safetensors("t", &tmp_file)?;
    // Load from file.
    let st = fuel_core::safetensors::load(&tmp_file, &fuel_core::Device::cpu())?;
    let t2 = st.get("t").unwrap();
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    // Load from bytes.
    let bytes = std::fs::read(tmp_file)?;
    let st = fuel_core::safetensors::SliceSafetensors::new(&bytes)?;
    let t2 = st.get("t").unwrap().load(&fuel_core::Device::cpu());
    let diff = (&t - t2)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff, 0f32);
    Ok(())
}

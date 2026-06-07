use fuel::Result;
use fuel_transformers::generation::LogitsProcessor;

#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    let logits = [-1.0f32, 0.0, 0.2, 1.0];
    // Host-side softmax over the f32 slice for the reference distribution.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&x| ((x - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let sm: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}

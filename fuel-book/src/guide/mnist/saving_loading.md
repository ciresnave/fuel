# Fuel MNIST Tutorial

## Saving and Loading Models

After training a model, it is useful to save and subsequently load the model parameters. In Fuel, this functionality is managed through the `LazyVarMap` data structure, with parameters stored on disk using the [safetensors](https://huggingface.co/docs/safetensors/index) format.

### Saving Model Parameters

Let's modify our `training_loop` function to include functionality for saving weights:

```rust
fn training_loop(
    m: fuel_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    let dev = fuel::cuda_backend::device_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Initialize a LazyVarMap for trainable parameters
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, dev.clone());
    let model = Model::new(&vs)?;

    let learning_rate = 0.05;
    let epochs = 10;

    // Initialize stochastic gradient descent optimizer
    let mut sgd = LazySgd::new(varmap.all_vars(), learning_rate)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for epoch in 1..epochs {
        // Standard MNIST forward pass
        let logits = model.forward(&train_images)?;
        let log_sm = logits.log_softmax(D::Minus1)?;

        // Compute Negative Log Likelihood loss
        let loss = loss::nll(&log_sm, &train_labels, loss::Reduction::Mean)?;

        // Backward + step in a single call.
        sgd.backward_step(&loss)?;

        // Evaluate model on test set
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            test_accuracy
        );
    }

    // Save model weights to disk
    varmap.save("model_weights.safetensors")?;
    Ok(())
}
```

```bash
$ cargo run --release

> 1 train loss:  2.40485 test acc:  0.11%
> 2 train loss:  2.34161 test acc:  0.14%
> 3 train loss:  2.28841 test acc:  0.17%
> 4 train loss:  2.24158 test acc:  0.19%
> 5 train loss:  2.19898 test acc:  0.23%
> 6 train loss:  2.15927 test acc:  0.26%
> 7 train loss:  2.12161 test acc:  0.29%
> 8 train loss:  2.08549 test acc:  0.32%
> 9 train loss:  2.05053 test acc:  0.35%
```

### Loading Model Parameters

Now that we have saved our model parameters, we can modify the code to load them. `LazyVarMap::load` mutates parameter buffers in place through interior mutability, so the binding does not need to be `mut`:

```rust
fn training_loop(
    m: fuel_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    let dev = fuel::cuda_backend::device_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Create a LazyVarMap for trainable parameters
    let varmap = LazyVarMap::new();
    let vs = LazyVarBuilder::from_varmap(varmap.clone(), DType::F32, dev.clone());
    let model = Model::new(&vs)?;

    // Load pre-trained weights from file (matches parameters by name —
    // `Model::new` must run first so the VarMap knows the expected names
    // and shapes).
    varmap.load("model_weights.safetensors")?;

    let learning_rate = 0.05;
    let epochs = 10;

    // Initialize stochastic gradient descent optimizer
    let mut sgd = LazySgd::new(varmap.all_vars(), learning_rate)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    for epoch in 1..epochs {
        // Standard MNIST forward pass
        let logits = model.forward(&train_images)?;
        let log_sm = logits.log_softmax(D::Minus1)?;

        // Compute Negative Log Likelihood loss
        let loss = loss::nll(&log_sm, &train_labels, loss::Reduction::Mean)?;

        // Backward + step in a single call.
        sgd.backward_step(&loss)?;

        // Evaluate model on test set
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            test_accuracy
        );
    }

    // Save updated weights back to disk
    varmap.save("model_weights.safetensors")?;
    Ok(())
}
```

```bash
$ cargo run --release

> 1 train loss:  2.01645 test acc:  0.38%
> 2 train loss:  1.98300 test acc:  0.41%
> 3 train loss:  1.95008 test acc:  0.44%
> 4 train loss:  1.91754 test acc:  0.47%
> 5 train loss:  1.88534 test acc:  0.50%
> 6 train loss:  1.85349 test acc:  0.53%
> 7 train loss:  1.82198 test acc:  0.56%
> 8 train loss:  1.79077 test acc:  0.59%
> 9 train loss:  1.75989 test acc:  0.61%
```

Note that loading the weights will fail if the specified file does not exist or is incompatible with the current model architecture. Implementing file existence checks and appropriate error handling is left to the user.
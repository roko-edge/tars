use std::error::Error;
use std::time::Instant;
use tars::*;
// argmax/hardmax algorithm
pub fn evaluate_accuracy(model: &Sequential, dataset: &[Data]) -> f32 {
    let mut correct = 0;

    for sample in dataset {
        let pred = model.forward(&sample.input);

        let pred_digit = (0..10)
            .max_by(|&a, &b| pred.get(&[a]).total_cmp(&pred.get(&[b])))
            .unwrap();

        let target_digit = (0..10)
            .max_by(|&a, &b| sample.target.get(&[a]).total_cmp(&sample.target.get(&[b])))
            .unwrap();

        if pred_digit == target_digit {
            correct += 1;
        }
    }

    (correct as f32 / dataset.len() as f32) * 100.0
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut train = experiments::mnist::train();
    let use_softmax = matches!(
        &train.experiment_type,
        experiments::experiment_type::ExperimentType::Mnist
    );

    let mut optimizer = Adam::new(&train.model, train.lr);

    let mut loader = DataMiniBatches::new(&train.dataset, 32, 42);

    println!(
        "============================================================================================"
    );
    println!("Starting Training: {:?}", train.experiment_type);
    println!(
        "Samples per Dataset: {} | Epochs: {} | Learning Rate: {}",
        train.dataset.len(),
        train.epochs,
        train.lr
    );
    println!(
        "============================================================================================"
    );

    let training_start = Instant::now();
    let loss_fn: &dyn Loss = if use_softmax { &SoftmaxCE } else { &MSE };
    for i in 1..=train.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in loader.iter_epoch() {
            let (effective_model, _) = train.model.quantize();
            let cache = effective_model.forward_cache(&batch.input);
            let grad = backward(&effective_model, &cache, &batch.target, loss_fn);

            //let cache = train.model.forward_cache(&batch.input);
            //let grad = backward(&train.model, &cache, &batch.target, loss_fn);
            optimizer.step(&mut train.model, &grad);

            let pred = cache.output();
            if let (Some(p_slice), Some(t_slice)) = (pred.as_slice(), batch.target.as_slice()) {
                let batch_loss: f32 = p_slice
                    .iter()
                    .zip(t_slice.iter())
                    .map(|(&p, &t)| {
                        let diff = p - t;
                        diff * diff
                    })
                    .sum::<f32>()
                    / p_slice.len() as f32;
                epoch_loss += batch_loss;
                batch_count += 1;
            }
        }

        let epoch_duration = epoch_start.elapsed();
        let samples_per_sec = (train.dataset.len() as f64) / epoch_duration.as_secs_f64();
        let avg_loss = if batch_count > 0 {
            epoch_loss / batch_count as f32
        } else {
            0.0
        };
        println!(
            "Epoch [{:04}/{:04}] - Time: {:>6.2?} | Throughput: {:>8.0} img/s | Loss (MSE): {:.6}",
            i, train.epochs, epoch_duration, samples_per_sec, avg_loss
        );
    }
    let training_duration = training_start.elapsed();
    println!(
        "============================================================================================"
    );
    println!("Training completed in: {:.2?}", training_duration);
    println!(
        "============================================================================================"
    );

    let f32_accuracy = evaluate_accuracy(&train.model, &train.dataset);
    println!(
        "F32 final accuracy: {:.2}%  | Average throughput: {:>8.0} img/s",
        f32_accuracy,
        (train.dataset.len() * train.epochs) as f64 / training_duration.as_secs_f64()
    );

    let scales;
    (train.model, scales) = train.model.quantize();

    let t3_accuracy = evaluate_accuracy(&train.model, &train.dataset);
    let (sparsity, _) = train.model.sparsity(&scales);
    println!(
        "Ternary PTQ accuracy: {:.2}% | Sparsity: {:.2}% | Performance degradation f32->T3: {:.2}% ",
        t3_accuracy,
        sparsity * 100.0,
        f32_accuracy - t3_accuracy
    );
    Ok(())
}

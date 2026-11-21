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

    let mut loader = DataMiniBatches::new(&train.dataset, 512, 42);

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

    for i in 1..=train.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for batch in loader.iter_epoch() {
            let cache = train.model.forward_cache(&batch.input);
            let logits = cache.output();

            let target: Tensor<f32> = if use_softmax {
                let sm_delta = SoftmaxCE.backward(logits, &batch.target);
                let b = logits.shape()[0] as f32;
                let out_sz = logits.shape()[1] as f32;
                let mut adjusted = Tensor::zeros(batch.target.shape().to_vec());
                for r in 0..logits.shape()[0] {
                    for c in 0..logits.shape()[1] {
                        adjusted.set(
                            &[r, c],
                            logits.get(&[r, c]) - (out_sz * b / 2.0) * sm_delta.get(&[r, c]),
                        );
                    }
                }
                adjusted
            } else {
                batch.target.clone()
            };

            let grad = backward(&train.model, &cache, &target, &MSE);
            optimizer.step(&mut train.model, &grad);
            // Telemetria zero-overhead:
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

    println!(
        "Final accuracy: {:.2}%  | Average throughput: {:>8.0} img/s",
        evaluate_accuracy(&train.model, &train.dataset),
        (train.dataset.len() * train.epochs) as f64 / training_duration.as_secs_f64()
    );

    Ok(())
}

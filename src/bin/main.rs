use std::error::Error;
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
    train.dataset.truncate(600);
    train.epochs = 100;
    let use_softmax = matches!(
        &train.experiment_type,
        experiments::experiment_type::ExperimentType::Mnist
    );

    println!("Starting {:?} experiment...", train.experiment_type);

    let mut optimizer = Adam::new(&train.model, train.lr);
    let initial_cost = cost(&train.model, &train.dataset);
    println!("initial cost: {initial_cost:014.8}");

    let mut loader = DataMiniBatches::new(&train.dataset, train.dataset.len(), 42);

    for _ in 1..=train.epochs {
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
        }
    }

    let final_cost = cost(&train.model, &train.dataset);
    println!("final cost: {final_cost:014.8}");
    println!(
        "total improvement: {:.3}%",
        ((initial_cost - final_cost) * 100.0) / initial_cost
    );

    //let scales;
    //(train.model, scales) = train.model.quantize();
    //export_model(&train.model, &scales)?;
    //export_inference(&train.model, &train.dataset)?;

    println!(
        "Accuracy: {:.2}%",
        evaluate_accuracy(&train.model, &train.dataset)
    );

    Ok(())
}

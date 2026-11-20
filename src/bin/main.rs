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

    println!("Starting {:?} experiment...", train.experiment_type);

    let step = (train.epochs / 20).max(1);
    let optimizer = BGD::new(train.lr);
    let mut prev_cost = cost(&train.model, &train.dataset);
    println!("epoch: 000000, cost is:{:014.8}", prev_cost);

    let mut loader = DataMiniBatches::new(&train.dataset, 256, 42);

    for i in 1..=train.epochs {
        for batch in loader.iter_epoch() {
            let cache = train.model.forward_cache(&batch.input);
            let grad = backward(&train.model, &cache, &batch.target);
            optimizer.step(&mut train.model, &grad);
        }

        if i % step == 0 {
            let curr_cost = cost(&train.model, &train.dataset);

            println!(
                "epoch: {:06.0}, cost is:{:014.8}, {:07.3}% better",
                i,
                curr_cost,
                ((prev_cost - curr_cost) * 100.0) / prev_cost
            );
            prev_cost = curr_cost;
        }
    }

    let scales;
    (train.model, scales) = train.model.quantize();
    export_model(&train.model, &scales)?;
    export_inference(&train.model, &train.dataset)?;

    println!(
        "Accuracy: {:.2}%",
        evaluate_accuracy(&train.model, &train.dataset)
    );

    Ok(())
}

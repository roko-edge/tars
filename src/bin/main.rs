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

    //training
    let optimizer = Adam::new(&train.model, train.lr);
    let mut trainer = Trainer::new(&mut train.model, optimizer, train.loss).with_qat(true);
    trainer.fit(&train.dataset[..50000], train.epochs);

    //evaluation
    let f32_accuracy = evaluate_accuracy(&train.model, &train.dataset[50000..]);
    println!("F32 final accuracy: {:.2}%", f32_accuracy,);

    let scales;
    (train.model, scales) = train.model.quantize();

    let t3_accuracy = evaluate_accuracy(&train.model, &train.dataset[50000..]);
    let (sparsity, _) = train.model.sparsity(&scales);
    println!(
        "Ternary PTQ accuracy: {:.2}% | Sparsity: {:.2}% | Performance degradation f32->T3: {:.2}% ",
        t3_accuracy,
        sparsity * 100.0,
        f32_accuracy - t3_accuracy
    );
    export_quantized_model(&train.model, &scales)?;
    export_inference(&train.model, &train.dataset[59900..])?;
    Ok(())
}

use crate::Data;
use crate::Module;
use crate::Sequential;

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

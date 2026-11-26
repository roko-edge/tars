use tars::*;

fn main() {
    let dataset = vec![
        Data::new(&[0.0, 0.0], &[0.0]),
        Data::new(&[0.0, 1.0], &[1.0]),
        Data::new(&[1.0, 0.0], &[1.0]),
        Data::new(&[1.0, 1.0], &[0.0]),
    ];

    let mut model = Sequential::new(2, 4).sigmoid().linear(1).sigmoid();
    let optimizer = Adam::new(&model, 1.0);
    let epochs = 10000;
    let summary = Trainer::new(&mut model, optimizer, MSE)
        .batch_size(4)
        .log_every(epochs / 20)
        .fit(&dataset, epochs);

    let report = ModelReport::new(&model).with_train(summary).with_size();
    println!("{report}");

    for sample in &dataset {
        let pred = model.forward(&sample.input);
        println!(
            "[{:.0}, {:.0}] -> target={:.1} | pred={:.4}",
            sample.input.get(&[0]),
            sample.input.get(&[1]),
            sample.target.get(&[0]),
            pred.get(&[0])
        );
    }
}

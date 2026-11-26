use mnist_reader::MnistReader;
use std::error::Error;
use tars::*;

fn encode_split(images: Vec<Vec<f32>>, labels: Vec<u8>) -> Vec<Data> {
    images
        .into_iter()
        .zip(labels)
        .map(|(image, label)| {
            let mut target = vec![0.0f32; 10];
            target[label as usize] = 1.0;
            Data::new(&image, &target)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut reader = MnistReader::new("mnist-data");
    reader.load()?;

    let data = DatasetSplit {
        train: encode_split(reader.train_data, reader.train_labels),
        eval: encode_split(reader.test_data, reader.test_labels),
    };

    let mut model = Sequential::new(784, 64).relu().linear(10);
    let optimizer = Adam::new(&model, 0.003);
    let epochs = 10;
    let summary = Trainer::new(&mut model, optimizer, SoftmaxCE)
        .with_qat(true)
        .batch_size(32)
        .log_every(epochs / 20)
        .fit(&data.train, epochs);

    let report = ModelReport::new(&model)
        .with_train(summary)
        .with_size()
        .with_accuracy(&data.eval)
        .with_ternary_accuracy(&data.eval)
        .with_sparsity();

    println!("{report}");

    if let Some(q) = &report.quantized {
        export_quantized_model(&q.model, &q.scales)?;
        export_inference(&q.model, &data.eval[..100])?;
    }
    Ok(())
}

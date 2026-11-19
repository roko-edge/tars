use crate::{experiment_type::ExperimentType, *};
use mnist_reader::MnistReader;

pub fn train() -> Train {
    let mut mnist = MnistReader::new("mnist-data");
    mnist.load().expect("Falha ao carregar os dados do MNIST");

    let dataset: Vec<Data> = mnist
        .train_data
        .into_iter()
        .zip(mnist.train_labels)
        .map(|(image, label)| {
            let mut target = vec![0.0f32; 10];
            target[label as usize] = 1.0;
            Data::new(&image, &target)
        })
        .collect();

    Train {
        model: Sequential::new(784).linear(64).relu().linear(10),

        epochs: 10,

        lr: 0.003,

        dataset,

        experiment_type: ExperimentType::Mnist,
    }
}

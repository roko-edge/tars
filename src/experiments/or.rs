use crate::{experiment_type::ExperimentType, *};

pub fn train() -> Train<MSE> {
    Train {
        model: Sequential::new(2, 1).sigmoid(),

        epochs: 100_000,
        lr: 20.0,

        dataset: vec![
            Data::new(&[0.0, 0.0], &[0.0]),
            Data::new(&[0.0, 1.0], &[1.0]),
            Data::new(&[1.0, 0.0], &[1.0]),
            Data::new(&[1.0, 1.0], &[1.0]),
        ],

        experiment_type: ExperimentType::Or,
        loss: MSE,
    }
}

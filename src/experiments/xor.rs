use crate::{experiment_type::ExperimentType, *};

pub fn train() -> Train {
    Train {
        model: Sequential::new(2).linear(4).sigmoid().linear(1).sigmoid(),
        epochs: 1000000,
        lr: 1.0,

        dataset: vec![
            Data::new(&[0.0, 0.0], &[0.0]),
            Data::new(&[0.0, 1.0], &[1.0]),
            Data::new(&[1.0, 0.0], &[1.0]),
            Data::new(&[1.0, 1.0], &[0.0]),
        ],

        experiment_type: ExperimentType::Xor,
    }
}

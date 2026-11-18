pub use crate::tensor_core::Tensor;

#[derive(Clone, Debug)]
pub struct Data {
    pub input: Tensor<f32>,
    pub target: Tensor<f32>,
}

impl Data {
    pub fn new(input: &[f32], target: &[f32]) -> Self {
        Self {
            input: Tensor::from_vec(input.to_vec(), vec![input.len()]),
            target: Tensor::from_vec(target.to_vec(), vec![target.len()]),
        }
    }
}

pub struct Batch {
    pub input: Tensor<f32>,
    pub target: Tensor<f32>,
}

impl Batch {
    fn from_samples(samples: &[Data]) -> Self {
        let b = samples.len();
        let in_sz = samples[0].input.shape()[0];
        let out_sz = samples[0].target.shape()[0];

        let mut inputs = Vec::with_capacity(b * in_sz);
        let mut targets = Vec::with_capacity(b * out_sz);

        for sample in samples {
            for j in 0..in_sz {
                inputs.push(sample.input.get(&[j]));
            }
            for j in 0..out_sz {
                targets.push(sample.target.get(&[j]));
            }
        }

        Self {
            input: Tensor::from_vec(inputs.to_vec(), vec![b, in_sz]),
            target: Tensor::from_vec(targets.to_vec(), vec![b, out_sz]),
        }
    }
    pub fn len(&self) -> usize {
        self.input.shape()[0]
    }
}

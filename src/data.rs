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

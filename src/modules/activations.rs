use crate::act_functions::{relu, sigmoidf};
use crate::modules::Module;
use crate::tensor_core::Tensor;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
}

impl Module for Activation {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        input.map(|z| match self {
            Activation::Relu => relu(z),
            Activation::Sigmoid => sigmoidf(z),
        })
    }
}

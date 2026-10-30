use crate::Tensor;
use crate::math::{relu, sigmoidf};
use crate::modules::Module;
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
}
impl Module for Activation {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.map(|z| match self {
            Activation::Relu => relu(z),
            Activation::Sigmoid => sigmoidf(z),
        })
    }
}

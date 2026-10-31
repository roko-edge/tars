use crate::act_functions::{relu, sigmoidf};
use crate::modules::Module;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Sigmoid,
}

impl Module for Activation {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for i in 0..input.len() {
            output[i] = match self {
                Activation::Relu => relu(input[i]),
                Activation::Sigmoid => sigmoidf(input[i]),
            }
        }
        output
    }
}

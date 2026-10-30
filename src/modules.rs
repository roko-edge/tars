pub mod activations;
pub mod linear;
pub mod sequential;
pub use crate::Tensor;
pub use activations::*;
pub use linear::*;
pub use sequential::*;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
}

#[derive(Clone, Debug)]
pub enum AnyModule {
    Linear(Linear),
    Activation(Activation),
}

impl AnyModule {
    pub fn as_linear(&self) -> &Linear {
        match self {
            Self::Linear(linear) => linear,
            _ => panic!(""),
        }
    }

    pub fn as_mut_linear(&mut self) -> &mut Linear {
        match self {
            Self::Linear(linear) => linear,
            _ => panic!(""),
        }
    }
}

impl Module for AnyModule {
    fn forward(&self, input: &Tensor) -> Tensor {
        match self {
            AnyModule::Linear(linear) => linear.forward(input),
            AnyModule::Activation(activation) => activation.forward(input),
        }
    }
}

use crate::grad::Grad;
use crate::sequential::Sequential;
pub use crate::tensor_core::Tensor;
use crate::{AnyModule, LinearGrad};

#[derive(Clone, Copy, Debug)]
pub struct BGD {
    lr: f32,
}

impl BGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
    pub fn step(&self, model: &mut Sequential, grad: &Grad) {
        for (module, grad_opt) in model.modules.iter_mut().zip(&grad.modules) {
            match (module, grad_opt) {
                (AnyModule::Linear(linear), Some(linear_grad)) => {
                    for o in 0..linear.out_sz {
                        for w in 0..linear.in_sz {
                            let old_w = linear.weights.get(&[o, w]);

                            linear.weights.set(
                                &[o, w],
                                old_w - (linear_grad.weights.get(&[o, w]) * self.lr),
                            );
                        }

                        let old_b = linear.bias.get(&[o]);

                        linear
                            .bias
                            .set(&[o], old_b - (linear_grad.bias.get(&[o]) * self.lr));
                    }
                }
                _ => {}
            }
        }
    }
}

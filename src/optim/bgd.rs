use crate::AnyModule;
use crate::grad::Grad;
use crate::sequential::Sequential;
pub use crate::tensor_core::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct BGD {
    lr: f32,
}

impl BGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
    pub fn step(&self, model: &mut Sequential, grad: &Grad) {
        for m in 0..model.modules.len() {
            match &grad.modules[m] {
                AnyModule::Linear(linear) => {
                    for o in 0..linear.out_sz {
                        for w in 0..linear.in_sz {
                            let weight = model.modules[m].as_linear().weights.get(&[o, w])
                                - grad.modules[m].as_linear().weights.get(&[o, w]) * self.lr;
                            model.modules[m]
                                .as_mut_linear()
                                .weights
                                .set(&[o, w], weight);
                        }
                        let bias = model.modules[m].as_linear().bias.get(&[o])
                            - grad.modules[m].as_linear().bias.get(&[o]) * self.lr;
                        model.modules[m].as_mut_linear().bias.set(&[o], bias);
                    }
                }
                AnyModule::Activation(_) => {}
            }
        }
    }
}

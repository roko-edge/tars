use crate::AnyModule;
use crate::grad::Grad;
use crate::sequential::Sequential;
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
                            model.modules[m].as_mut_linear().weights[o][w] -=
                                grad.modules[m].as_linear().weights[o][w] * self.lr;
                        }
                        model.modules[m].as_mut_linear().bias[o] -=
                            grad.modules[m].as_linear().bias[o] * self.lr;
                    }
                }
                AnyModule::Activation(_) => {}
            }
        }
    }
}

use crate::modules::{AnyModule, Sequential};
use crate::optim::grad::Grad;
use crate::tensor_core::Tensor;

pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: usize,
    pub m_weights: Vec<Option<Tensor<f32>>>,
    pub v_weights: Vec<Option<Tensor<f32>>>,
    pub m_bias: Vec<Option<Tensor<f32>>>,
    pub v_bias: Vec<Option<Tensor<f32>>>,
}

impl Adam {
    pub fn new(model: &Sequential, lr: f32) -> Self {
        let len = model.modules.len();
        let mut m_weights = Vec::with_capacity(len);
        let mut v_weights = Vec::with_capacity(len);
        let mut m_bias = Vec::with_capacity(len);
        let mut v_bias = Vec::with_capacity(len);

        for module in &model.modules {
            match module {
                AnyModule::Linear(linear) => {
                    m_weights.push(Some(Tensor::zeros(vec![linear.out_sz, linear.in_sz])));
                    v_weights.push(Some(Tensor::zeros(vec![linear.out_sz, linear.in_sz])));
                    m_bias.push(Some(Tensor::zeros(vec![linear.out_sz])));
                    v_bias.push(Some(Tensor::zeros(vec![linear.out_sz])));
                }
                _ => {
                    m_weights.push(None);
                    v_weights.push(None);
                    m_bias.push(None);
                    v_bias.push(None);
                }
            }
        }

        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_weights,
            v_weights,
            m_bias,
            v_bias,
        }
    }

    pub fn step(&mut self, model: &mut Sequential, grad: &Grad) {
        self.t += 1;

        let inv_bc1 = 1.0 / (1.0 - self.beta1.powi(self.t as i32));
        let inv_bc2 = 1.0 / (1.0 - self.beta2.powi(self.t as i32));
        let one_minus_b1 = 1.0 - self.beta1;
        let one_minus_b2 = 1.0 - self.beta2;

        for i in 0..model.modules.len() {
            let (
                AnyModule::Linear(linear),
                Some(linear_grad),
                Some(mw),
                Some(vw),
                Some(mb),
                Some(vb),
            ) = (
                &mut model.modules[i],
                &grad.modules[i],
                &mut self.m_weights[i],
                &mut self.v_weights[i],
                &mut self.m_bias[i],
                &mut self.v_bias[i],
            )
            else {
                continue;
            };

            for o in 0..linear.out_sz {
                for w in 0..linear.in_sz {
                    let g = linear_grad.weights.get(&[o, w]);

                    let m = self.beta1 * mw.get(&[o, w]) + one_minus_b1 * g;
                    let v = self.beta2 * vw.get(&[o, w]) + one_minus_b2 * (g * g);

                    mw.set(&[o, w], m);
                    vw.set(&[o, w], v);

                    let m_hat = m * inv_bc1;
                    let v_hat = v * inv_bc2;
                    let old_w = linear.weights.get(&[o, w]);

                    linear
                        .weights
                        .set(&[o, w], old_w - self.lr * m_hat / (v_hat.sqrt() + self.eps));
                }

                let gb = linear_grad.bias.get(&[o]);

                let m = self.beta1 * mb.get(&[o]) + one_minus_b1 * gb;
                let v = self.beta2 * vb.get(&[o]) + one_minus_b2 * (gb * gb);

                mb.set(&[o], m);
                vb.set(&[o], v);

                let m_hat = m * inv_bc1;
                let v_hat = v * inv_bc2;
                let old_b = linear.bias.get(&[o]);

                linear
                    .bias
                    .set(&[o], old_b - self.lr * m_hat / (v_hat.sqrt() + self.eps));
            }
        }
    }
}

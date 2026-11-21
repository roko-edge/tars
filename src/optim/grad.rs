use crate::Activation;
use crate::AnyModule;
use crate::ForwardCache;
use crate::Loss;
use crate::Sequential;
use crate::act_functions::*;
use crate::tensor_core::{MatMul, Tensor};

#[derive(Clone, Debug)]
pub struct LinearGrad {
    pub weights: Tensor<f32>,
    pub bias: Tensor<f32>,
}

#[derive(Clone, Debug)]
pub struct Grad {
    pub modules: Vec<Option<LinearGrad>>,
}

pub fn backward(
    model: &Sequential,
    cache: &ForwardCache,
    target: &Tensor<f32>,
    loss: &dyn Loss,
) -> Grad {
    let pred = cache.output();
    let mut delta = loss.backward(pred, target);

    let b = pred.shape()[0];

    let mut grad = Grad {
        modules: model
            .modules
            .iter()
            .map(|m| match m {
                AnyModule::Linear(l) => Some(LinearGrad {
                    weights: Tensor::zeros(vec![l.out_sz, l.in_sz]),
                    bias: Tensor::zeros(vec![l.out_sz]),
                }),
                AnyModule::Activation(_) => None,
            })
            .collect(),
    };

    for m in (0..model.modules.len()).rev() {
        match &model.modules[m] {
            AnyModule::Linear(linear) => {
                let a_prev = cache.module_input(m);
                let dw = delta.transpose(0, 1).matmul(a_prev);

                // Fazer: Implementar um método de add entre tensores que suporte as operações [B,out] + [out] para operações com min-batches com o bias, pra ficar só uma linha
                let mut db = Tensor::zeros(vec![linear.out_sz]);
                for i in 0..b {
                    for o in 0..linear.out_sz {
                        db.set(&[o], db.get(&[o]) + delta.get(&[i, o]));
                    }
                }
                grad.modules[m] = Some(LinearGrad {
                    weights: dw,
                    bias: db,
                });
                delta = delta.matmul(&linear.weights);
            }

            AnyModule::Activation(activation) => {
                let deriv = match activation {
                    Activation::Relu => cache.module_input(m).map(relu_deriv_from_z),
                    Activation::Sigmoid => cache.module_output(m).map(sigmoidf_deriv_from_a),
                };
                delta = delta.mul(&deriv);
            }
        }
    }

    grad
}

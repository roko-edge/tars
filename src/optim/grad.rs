use crate::Activation;
use crate::AnyModule;
use crate::ForwardCache;
use crate::Sequential;
use crate::act_functions::*;
use crate::tensor_core::Tensor;

#[derive(Clone, Debug)]
pub struct LinearGrad {
    pub weights: Tensor<f32>,
    pub bias: Tensor<f32>,
}

#[derive(Clone, Debug)]
pub struct Grad {
    pub modules: Vec<Option<LinearGrad>>,
}

pub fn backward(model: &Sequential, cache: &ForwardCache, target: &Tensor<f32>) -> Grad {
    let pred = cache.output();
    let mut delta = Tensor::zeros(pred.shape().to_vec());

    let b = pred.shape()[0];
    let out_sz = pred.shape()[1];

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

    for i in 0..b {
        for j in 0..out_sz {
            delta.set(
                &[i, j],
                cost_derivative(pred.get(&[i, j]), target.get(&[i, j]), out_sz) / b as f32,
            );
        }
    }

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

/*
pub fn backward(model: &Sequential, data: &[Data]) -> Grad {
    let mut grad = Grad {
        modules: model.modules.clone(),
    };

    for m in 0..grad.modules.len() {
        match &model.modules[m] {
            AnyModule::Linear(l) => {
                grad.modules[m].as_mut_linear().weights = Tensor::zeros(vec![l.in_sz, l.out_sz]);
                grad.modules[m].as_mut_linear().bias = Tensor::zeros(vec![l.out_sz]);
            }

            AnyModule::Activation(_) => {}
        }
    }

    for sample in data {
        let acts = forward_memoria(model, &sample.input);

        let mut delta = vec![0.0; sample.target.len()];

        for i in 0..sample.target.len() {
            delta[i] = cost_derivative(
                acts.last().unwrap()[i],
                sample.target[i],
                sample.target.len(),
            );
        }

        for m in (0..model.modules.len()).rev() {
            match &model.modules[m] {
                AnyModule::Linear(l) => {
                    let a_prev = &acts[m];
                    let g = grad.modules[m].as_mut_linear();

                    for o in 0..l.out_sz {
                        for w in 0..l.in_sz {
                            g.weights[o][w] += delta[o] * a_prev[w];
                        }
                        g.bias[o] += delta[o];
                    }

                    let mut prev = vec![0.0; l.in_sz];

                    for w in 0..l.in_sz {
                        for o in 0..l.out_sz {
                            prev[w] += l.weights[o][w] * delta[o];
                        }
                    }

                    delta = prev;
                }
                AnyModule::Activation(a) => {
                    let a_in = &acts[m]; // z
                    let a_out = &acts[m + 1]; // a

                    for i in 0..delta.len() {
                        let d = match a {
                            Activation::Relu => relu_deriv_from_z(a_in[i]),
                            Activation::Sigmoid => sigmoidf_deriv_from_a(a_out[i]),
                        };

                        delta[i] *= d;
                    }
                }
            }
        }
    }

    let n = data.len() as f32;

    for m in 0..grad.modules.len() {
        match &mut grad.modules[m] {
            AnyModule::Linear(linear) => {
                for output_neuron in &mut linear.weights {
                    for weight in output_neuron {
                        *weight /= n;
                    }
                }

                for bias in &mut linear.bias {
                    *bias /= n;
                }
            }
            AnyModule::Activation(_) => {}
        }
    }

    grad
}
*/

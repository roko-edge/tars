use crate::Activation;
use crate::AnyModule;
use crate::Data;
use crate::Module;
use crate::Sequential;
use crate::act_functions::*;
use crate::cost;

#[derive(Clone, Debug)]
pub struct Grad {
    pub modules: Vec<AnyModule>,
}

pub fn forward_memoria(model: &Sequential, input: &[f32]) -> Vec<Vec<f32>> {
    let mut acts: Vec<Vec<f32>> = vec![input.to_vec()];
    let mut cur = input.to_vec();

    for mod_ in &model.modules {
        cur = mod_.forward(&cur);
        acts.push(cur.clone());
    }

    acts
}

pub fn backward(model: &Sequential, data: &[Data]) -> Grad {
    let mut grad = Grad {
        modules: model.modules.clone(),
    };

    for m in 0..grad.modules.len() {
        match &model.modules[m] {
            AnyModule::Linear(l) => {
                grad.modules[m].as_mut_linear().weights = vec![vec![0.0; l.in_sz]; l.out_sz];
                grad.modules[m].as_mut_linear().bias = vec![0.0; l.out_sz];
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

pub fn num_grad(model: &Sequential, data: &[Data]) -> Grad {
    let mut temp_model = model.clone();
    let mut grad = Grad {
        modules: model.modules.clone(),
    };
    let h = 1e-3;
    for m in 0..model.modules.len() {
        match &model.modules[m] {
            AnyModule::Linear(linear) => {
                for o in 0..linear.out_sz {
                    for w in 0..linear.in_sz {
                        temp_model.modules[m].as_mut_linear().weights[o][w] += h;
                        let costp = cost(&temp_model, data);
                        temp_model.modules[m].as_mut_linear().weights[o][w] -= 2.0 * h;
                        let costm = cost(&temp_model, data);
                        grad.modules[m].as_mut_linear().weights[o][w] = (costp - costm) / (2.0 * h);
                        temp_model.modules[m].as_mut_linear().weights[o][w] += h;
                    }

                    temp_model.modules[m].as_mut_linear().bias[o] += h;
                    let costp = cost(&temp_model, data);
                    temp_model.modules[m].as_mut_linear().bias[o] -= 2.0 * h;
                    let costm = cost(&temp_model, data);
                    grad.modules[m].as_mut_linear().bias[o] = (costp - costm) / (2.0 * h);
                    temp_model.modules[m].as_mut_linear().bias[o] += h;
                }
            }
            AnyModule::Activation(_) => {}
        }
    }
    grad
}

use crate::modules::Module;
use crate::quant::*;
pub use crate::rng_gen::random_mat;
use crate::tensor_core::{MatMul, Tensor};

#[derive(Clone, Debug)]
pub struct Linear {
    pub in_sz: usize,
    pub out_sz: usize,
    pub weights: Tensor<f32>,
    pub bias: Tensor<f32>,
}

impl Linear {
    pub fn new(in_sz: usize, out_sz: usize, weights: Tensor<f32>, bias: Tensor<f32>) -> Self {
        assert_eq!(weights.shape().len(), 2, "weights is not a matrix");
        assert_eq!(bias.shape().len(), 1, "bias is not a vector");
        assert_eq!(
            weights.shape(),
            [out_sz, in_sz],
            "Dimensionas of weights is not correct"
        );
        assert_eq!(bias.shape(), [out_sz], "Dimensionas of bias is not correct");
        Self {
            in_sz,
            out_sz,
            weights,
            bias,
        }
    }
    pub fn zeros(in_sz: usize, out_sz: usize) -> Self {
        let weights = Tensor::zeros(vec![out_sz, in_sz]);
        let bias = Tensor::zeros(vec![out_sz]);

        Self {
            in_sz,
            out_sz,
            weights,
            bias,
        }
    }
    pub fn random(in_sz: usize, out_sz: usize) -> Self {
        // Acanaiado, trocar essa função por uma que já use vetor unidimensional;
        let weights_random = random_mat(in_sz, out_sz);
        let weights = Tensor::from_vec(
            weights_random.into_iter().flatten().collect(),
            vec![out_sz, in_sz],
        );
        let bias = Tensor::zeros(vec![out_sz]);
        Self {
            in_sz,
            out_sz,
            weights,
            bias,
        }
    }
    pub fn quantize(&self) -> (Self, f32) {
        let mut l = self.clone();
        let mut w0 = 0.0;
        for o in 0..l.out_sz {
            for w in 0..l.in_sz {
                w0 += l.weights.get(&[o, w]).abs();
            }
        }
        if w0 > 1e-7 {
            w0 /= (l.in_sz * l.out_sz) as f32;
        }

        for o in 0..l.out_sz {
            for w in 0..l.in_sz {
                let trit = quantize_scalar(l.weights.get(&[o, w]), w0);
                l.weights.set(&[o, w], trit.as_f32() * w0);
            }
        }
        (l, w0)
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let rank = input.shape().len();

        match rank {
            1 => {
                let mut out = self.weights.matvec(input);
                out.add_mut(&self.bias);

                out
            }
            2 => {
                let mut out = input.matmul(&self.weights.transpose(0, 1));
                let dim1 = out.shape()[0];
                let dim2 = out.shape()[1];

                for b in 0..dim1 {
                    for o in 0..dim2 {
                        let v = out.get(&[b, o]) + self.bias.get(&[o]);
                        out.set(&[b, o], v);
                    }
                }
                out
            }
            _ => {
                panic!("The rank used is neither 1 or 2");
            }
        }
    }
}

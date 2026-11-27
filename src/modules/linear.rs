use crate::modules::Module;
use crate::quant::*;
pub use crate::rng_gen::random_mat;
use crate::tensor_core::{MatMul, Tensor};
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

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
        let mut rng = ChaCha8Rng::seed_from_u64(67);
        Self::random_with_rng(in_sz, out_sz, &mut rng)
    }

    pub fn random_with_rng(in_sz: usize, out_sz: usize, rng: &mut ChaCha8Rng) -> Self {
        let weights_random = random_mat(rng, in_sz, out_sz);

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

    fn with_contiguous_weights<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        match self.weights.as_slice() {
            Some(s) => f(s),
            None => f(self
                .weights
                .make_contiguous()
                .as_slice()
                .expect("make_contiguous must yield a contiguous slice")),
        }
    }
    pub fn quantize(&self) -> (Self, f32) {
        let (q_weights, w0) = self.with_contiguous_weights(|src_w| {
            let mut q = vec![0.0; src_w.len()];
            let w0 = quantize_into_f32(src_w, &mut q);
            (q, w0)
        });
        (
            Self {
                in_sz: self.in_sz,
                out_sz: self.out_sz,
                weights: Tensor::from_vec(q_weights, self.weights.shape().to_vec()),
                bias: self.bias.clone(),
            },
            w0,
        )
    }
    pub fn quantize_into(&self, other: &mut Self) -> f32 {
        assert_eq!(
            self.in_sz, other.in_sz,
            "input size mismatch in quantize_into"
        );
        assert_eq!(
            self.out_sz, other.out_sz,
            "output size mismatch in quantize_into"
        );

        if let (Some(src_b), Some(dst_b)) = (self.bias.as_slice(), other.bias.as_mut_slice()) {
            dst_b.copy_from_slice(src_b);
        } else {
            other.bias = self.bias.make_contiguous();
        }

        if let (Some(src_w), Some(dst_w)) = (self.weights.as_slice(), other.weights.as_mut_slice())
        {
            quantize_into_f32(src_w, dst_w)
        } else {
            let (q, w0) = self.quantize();
            other.weights = q.weights;
            w0
        }
    }
    pub fn sparsity(&self, w0: f32) -> (usize, f32) {
        self.with_contiguous_weights(|slice| {
            let zeros = slice.iter().filter(|w| w.abs() < w0).count();
            (zeros, zeros as f32 / slice.len() as f32)
        })
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

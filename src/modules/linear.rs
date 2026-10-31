use crate::modules::Module;
pub use crate::rng_gen::random_mat;
use crate::tensor_core::Tensor;

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
}
impl Module for Linear {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(input.shape().len(), 1, "Forward uses rank 1");

        let mut out = self.weights.matvec(input);
        out.add_mut(&self.bias);

        out
    }
}

use crate::modules::Module;
pub use crate::rng_gen::random_mat;

#[derive(Clone, Debug)]
pub struct Linear {
    pub in_sz: usize,
    pub out_sz: usize,
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl Linear {
    pub fn new(in_sz: usize, out_sz: usize, weights: Vec<Vec<f32>>, bias: Vec<f32>) -> Self {
        Self {
            in_sz,
            out_sz,
            weights,
            bias,
        }
    }
    pub fn zeros(in_sz: usize, out_sz: usize) -> Self {
        Self {
            in_sz,
            out_sz,
            weights: vec![vec![0.0; in_sz]; out_sz],
            bias: vec![0.0; out_sz],
        }
    }
    pub fn random(in_sz: usize, out_sz: usize) -> Self {
        Self {
            in_sz,
            out_sz,
            weights: random_mat(in_sz, out_sz),
            bias: vec![0.0; out_sz],
        }
    }
}
impl Module for Linear {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.out_sz];
        for i in 0..self.out_sz {
            output[i] = self.bias[i];
            for j in 0..self.in_sz {
                output[i] += input[j] * self.weights[i][j];
            }
        }
        output
    }
}

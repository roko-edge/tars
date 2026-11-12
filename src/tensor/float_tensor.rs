use crate::tensor_core::*;
use std::sync::Arc;

impl Element for f32 {
    type Storage = Arc<Vec<f32>>;
    const DTYPE: DType = DType::F32;
}

impl MatMul<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn matmul(&self, other: &Tensor<f32>) -> Self::Output {
        assert_eq!(self.shape().len(), 2);
        assert_eq!(other.shape().len(), 2);
        assert_eq!(self.shape()[1], other.shape()[0]);
        let mut out = Tensor::zeros(vec![self.shape()[0], other.shape()[1]]);
        let mut acc;
        for k in 0..other.shape()[1] {
            for i in 0..self.shape()[0] {
                acc = 0.0;
                for j in 0..self.shape()[1] {
                    acc += self.get(&[i, j]) * other.get(&[j, k]);
                }
                out.set(&[i, k], acc);
            }
        }
        out
    }
}

impl Tensor<f32> {
    pub fn mul_scalar(&self, s: f32) -> Self {
        self.map(|x| x * s)
    }

    pub fn mul_scalar_mut(&mut self, s: f32) {
        self.map_mut(|x| x * s);
    }

    pub fn div_scalar(&self, s: f32) -> Self {
        assert!(s != 0.0);
        self.map(|x| x / s)
    }

    pub fn div_scalar_mut(&mut self, s: f32) {
        assert!(s != 0.0);
        self.map_mut(|x| x / s);
    }
}

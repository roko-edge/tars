use crate::tensor_core::*;
use std::sync::Arc;

impl Element for f32 {
    type Storage = Arc<Vec<f32>>;
    const DTYPE: DType = DType::F32;
}
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
#[inline]
fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    y.iter_mut()
        .zip(x)
        .for_each(|(dst, &xv)| *dst += alpha * xv);
}
impl MatMul<Tensor<f32>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn matmul(&self, other: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(self.shape().len(), 2);
        assert_eq!(other.shape().len(), 2);
        assert_eq!(self.shape()[1], other.shape()[0]);

        let (m, k, n) = (self.shape()[0], self.shape()[1], other.shape()[1]);
        let mut out = vec![0.0f32; m * n];
        if m == 0 || k == 0 || n == 0 {
            return Tensor::from_vec(out, vec![m, n]);
        }

        match (
            self.as_slice(),
            self.as_transposed_slice(),
            other.as_slice(),
            other.as_transposed_slice(),
        ) {
            // A * B^T
            (Some(a), _, _, Some(b_t)) => {
                for (a_row, out_row) in a.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
                    for (dst, b_col) in out_row.iter_mut().zip(b_t.chunks_exact(k)) {
                        *dst = dot(a_row, b_col);
                    }
                }
            }
            // A * B
            (Some(a), _, Some(b), _) => {
                for (a_row, out_row) in a.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
                    for (&a_ik, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
                        axpy(a_ik, b_row, out_row);
                    }
                }
            }
            // A^T * B
            (_, Some(a_t), Some(b), _) => {
                for (at_row, b_row) in a_t.chunks_exact(m).zip(b.chunks_exact(n)) {
                    for (&a_ik, out_row) in at_row.iter().zip(out.chunks_exact_mut(n)) {
                        axpy(a_ik, b_row, out_row);
                    }
                }
            }
            // Fallback
            _ => {
                for (i, out_row) in out.chunks_exact_mut(n).enumerate() {
                    for (j, dst) in out_row.iter_mut().enumerate() {
                        *dst = (0..k)
                            .map(|kk| self.get(&[i, kk]) * other.get(&[kk, j]))
                            .sum();
                    }
                }
            }
        }
        Tensor::from_vec(out, vec![m, n])
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

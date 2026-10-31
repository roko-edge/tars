use crate::tensor_core::*;
use std::sync::Arc;

pub struct T3 {
    pckg: Vec<u8>,
}

pub trait Mul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

impl Element for T3 {
    type Storage = Arc<Vec<u64>>;
    const DTYPE: DType = DType::T3;
}

impl T3 {
    pub fn new(data: Vec<u8>) -> Self {
        Self { pckg: data }
    }
}

// impl Mul<Tensor<T3>> for Tensor<f32> {
//     type Output = Tensor<f32>;
//
//     fn matmul(&self, rhs: &Tensor<T3>) -> Self::Output {
//         assert_eq!(self.shape().len(), 2);
//         assert_eq!(rhs.shape().len(), 2);
//         assert_eq!(self.shape()[1], rhs.shape()[0]);
//
//         let mut out = Tensor::<f32>::zeros(vec![self.shape()[0], rhs.shape[1]]);
//         let mut acc;
//
//         for k in 0..rhs.shape()[1] {
//             for i in 0..self.shape()[0] {
//                 acc = 0.0;
//                 for j in 0..self.shape[1] {
//                     acc += self.get(&[i, j]) * rhs.get(&[j, k]);
//                 }
//                 out.set(&[i, k], acc);
//             }
//         }
//         out
//     }
// }
//
// impl Tensor<T3> {}

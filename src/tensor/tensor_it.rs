// use crate::tensor;
//
// use super::tensor::*;
//
// struct LogicalIter<'a> {
//     tensor: &'a Tensor,
//     idx: &'a [usize],
// }
//
// impl<'a> Iterator for LogicalIter<'a> {
//     type Item = LogicalIter<'a>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         Tensor::step_logical(&(self.tensor.shape().to_vec()), &mut self.idx);
//         Some(self)
//     }
// }
//
// struct PhysicalIter<'a> {
//     tensor: &'a Tensor,
//     idx: &'a [usize],
// }
//
// impl<'a> Iterator for PhysicalIter<'a> {
//     type Item = PhysicalIter<'a>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         Tensor::step_logical((self.tensor.shape().clone()), self.idx);
//         Some(self.tensor.get(self.idx))
//     }
// }
//
// #[test]
// fn tensor_it_test() {
//     let t = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
// }

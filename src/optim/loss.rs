pub trait Loss {
    fn foward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32;
    fn backward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32>;
}

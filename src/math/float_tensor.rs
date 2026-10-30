use crate::tensor::*;

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

use crate::tensor::*;

impl Tensor<f32> {
    pub fn mul_scalar(&self, s: f32) -> Self {
        let mut out = Tensor::zeros(self.shape.clone());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) * s);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn mul_scalar_mut(&mut self, s: f32) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) * s);
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn div_scalar(&self, s: f32) -> Self {
        assert!(s != 0.0);
        let mut out = Tensor::zeros(self.shape.clone());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) / s);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn div_scalar_mut(&mut self, s: f32) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) / s);
            Self::step_logical(&self.shape, &mut idx);
        }
    }
}

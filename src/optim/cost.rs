use crate::data::Data;
use crate::modules::Module;
use crate::sequential::Sequential;
pub use crate::tensor_core::Tensor;

pub fn cost(model: &Sequential, data: &[Data]) -> f32 {
    let mut loss = 0.0;
    for sample in data {
        let mut erro = 0.0;
        let target = &sample.target;
        let pred = model.forward(&sample.input);
        for i in 0..sample.target.numel() {
            erro += (target.get(&[i]) - pred.get(&[i])) * (target.get(&[i]) - pred.get(&[i]));
        }
        loss += erro / (sample.target.numel() as f32);
    }
    loss / data.len() as f32
}

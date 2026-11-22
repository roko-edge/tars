pub mod adam;
pub mod bgd;
pub mod cost;
pub mod grad;
pub mod loss;

use crate::Sequential;
pub use adam::*;
pub use bgd::*;
pub use cost::*;
pub use grad::*;
pub use loss::*;
pub trait Optimizer {
    fn lr(&self) -> f32;
    fn step(&mut self, model: &mut Sequential, grad: &Grad);
}

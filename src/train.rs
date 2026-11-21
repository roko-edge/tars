use crate::data::*;
use crate::experiments::experiment_type::ExperimentType;
use crate::sequential::*;

#[derive(Debug, Clone)]
pub struct Train {
    pub model: Sequential,
    pub epochs: usize,
    pub lr: f32,
    pub dataset: Vec<Data>,
    pub experiment_type: ExperimentType,
}

impl Train {
    pub fn view_metrics() {}
}

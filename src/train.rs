use crate::Loss;
use crate::Optimizer;
use crate::backward;
use crate::data::*;
use crate::experiments::experiment_type::ExperimentType;
use crate::sequential::*;
use std::time::Instant;
#[derive(Debug, Clone)]
pub struct Train<L: Loss> {
    pub model: Sequential,
    pub epochs: usize,
    pub lr: f32,
    pub dataset: Vec<Data>,
    pub experiment_type: ExperimentType,
    pub loss: L,
}

impl<L: Loss> Train<L> {
    pub fn view_metrics() {}
}

#[derive(Debug)]
pub struct Trainer<'a, L: Loss, O: Optimizer> {
    pub model: &'a mut Sequential,
    pub optimizer: O,
    pub loss_fn: L,
    with_qat: bool,
    silent: bool,
    batch_size: usize,
}

impl<'a, L: Loss, O: Optimizer> Trainer<'a, L, O> {
    pub fn new(model: &'a mut Sequential, optimizer: O, loss_fn: L) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            with_qat: false,
            silent: false,
            batch_size: 32,
        }
    }
    pub fn with_qat(&mut self, value: bool) {
        self.with_qat = value
    }
    pub fn silent(&mut self, value: bool) {
        self.silent = value
    }
    pub fn batch_size(&mut self, value: usize) {
        self.batch_size = value
    }
    pub fn fit(&mut self, train_dataset: &[Data], epochs: usize) {
        let mut loader = DataMiniBatches::new(train_dataset, 64, 42);

        println!(
            "============================================================================================"
        );
        println!(
            "Samples per Dataset: {} | Epochs: {} | Learning Rate: {}",
            train_dataset.len(),
            epochs,
            self.optimizer.lr()
        );
        println!(
            "============================================================================================"
        );

        let training_start = Instant::now();
        for i in 1..=epochs {
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for batch in loader.iter_epoch() {
                let cache;
                let grad;
                if self.with_qat {
                    let (effective_model, _) = self.model.quantize();
                    cache = effective_model.forward_cache(&batch.input);
                    grad = backward(&effective_model, &cache, &batch.target, &self.loss_fn);
                } else {
                    cache = self.model.forward_cache(&batch.input);
                    grad = backward(self.model, &cache, &batch.target, &self.loss_fn);
                }
                self.optimizer.step(self.model, &grad);

                let pred = cache.output();
                if let (Some(p_slice), Some(t_slice)) = (pred.as_slice(), batch.target.as_slice()) {
                    let batch_loss: f32 = p_slice
                        .iter()
                        .zip(t_slice.iter())
                        .map(|(&p, &t)| {
                            let diff = p - t;
                            diff * diff
                        })
                        .sum::<f32>()
                        / p_slice.len() as f32;
                    epoch_loss += batch_loss;
                    batch_count += 1;
                }
            }

            let epoch_duration = epoch_start.elapsed();
            let samples_per_sec = (train_dataset.len() as f64) / epoch_duration.as_secs_f64();
            let avg_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };
            println!(
                "Epoch [{:04}/{:04}] - Time: {:>6.2?} | Throughput: {:>8.0} img/s | Loss (MSE): {:.6}",
                i, epochs, epoch_duration, samples_per_sec, avg_loss
            );
        }
        let training_duration = training_start.elapsed();
        println!(
            "============================================================================================"
        );
        println!(
            "Training completed in: {:.2?} | Average throughput: {:>8.0} img/s",
            training_duration,
            (train_dataset.len() * epochs) as f64 / training_duration.as_secs_f64()
        );
        println!(
            "============================================================================================"
        );
        println!();
    }
}

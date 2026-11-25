use crate::Loss;
use crate::Optimizer;
use crate::backward;
use crate::data::*;
use crate::experiments::experiment_type::ExperimentType;
use crate::sequential::*;
use std::fmt::Debug;
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

impl<'a, L: Loss + Debug, O: Optimizer> Trainer<'a, L, O> {
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
        let mut loader = DataMiniBatches::new(train_dataset, self.batch_size, 42);

        println!(
            "==============================================================================================="
        );

        println!(
            "Samples per Dataset: {} | Epochs: {} | Learning Rate: {} | Batch size: {} | QAT: {}",
            train_dataset.len(),
            epochs,
            self.optimizer.lr(),
            self.batch_size,
            if self.with_qat { "Enabled" } else { "Disabled" }
        );

        println!(
            "==============================================================================================="
        );

        let training_start = Instant::now();
        for i in 1..=epochs {
            let epoch_start = Instant::now();
            let mut epoch_loss_sum = 0.0;
            let mut processed_samples = 0;

            for batch in loader.iter_epoch() {
                let cache;

                let grad = if self.with_qat {
                    let (effective_model, _) = self.model.quantize();
                    cache = effective_model.forward_cache(&batch.input);
                    backward(&effective_model, &cache, &batch.target, &self.loss_fn)
                } else {
                    cache = self.model.forward_cache(&batch.input);
                    backward(self.model, &cache, &batch.target, &self.loss_fn)
                };

                let batch_loss = self.loss_fn.forward(cache.output(), &batch.target);
                let batch_samples = batch.len();

                epoch_loss_sum += (batch_loss) as f64 * batch_samples as f64;
                processed_samples += batch_samples;

                self.optimizer.step(self.model, &grad);
            }

            let epoch_duration = epoch_start.elapsed();
            let samples_per_sec = (train_dataset.len() as f64) / epoch_duration.as_secs_f64();
            let avg_loss = epoch_loss_sum / processed_samples as f64;

            println!(
                "Epoch [{:04}/{:04}] - Time: {:>6.2?} | Throughput: {:>8.0} img/s | Loss ({:?}): {:.6}",
                i, epochs, epoch_duration, samples_per_sec, self.loss_fn, avg_loss
            );
        }

        let training_duration = training_start.elapsed();

        println!(
            "==============================================================================================="
        );

        println!(
            "Training completed in: {:.2?} | Average throughput: {:>8.0} img/s",
            training_duration,
            (train_dataset.len() * epochs) as f64 / training_duration.as_secs_f64()
        );

        println!(
            "==============================================================================================="
        );

        println!();
    }
}

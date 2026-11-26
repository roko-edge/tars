use crate::Loss;
use crate::Optimizer;
use crate::backward;
use crate::data::*;
use crate::format::{pretty, print_rule};
use crate::sequential::*;
use std::fmt::Debug;
use std::time::{Duration, Instant};
#[derive(Debug, Clone, Copy)]
pub struct TrainSummary {
    pub epochs: usize,
    pub duration: Duration,
    pub final_loss: f32,
    pub avg_throughput: f64,
    pub qat: bool,
}

#[derive(Debug)]
pub struct Trainer<'a, L: Loss, O: Optimizer> {
    pub model: &'a mut Sequential,
    pub optimizer: O,
    pub loss_fn: L,
    qat_model: Option<Sequential>,
    silent: bool,
    batch_size: usize,
    log_every: usize,
}

fn throughput(samples: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        samples as f64 / secs
    } else {
        0.0
    }
}
impl<'a, L: Loss + Debug, O: Optimizer> Trainer<'a, L, O> {
    pub fn new(model: &'a mut Sequential, optimizer: O, loss_fn: L) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            qat_model: None,
            silent: false,
            batch_size: 32,
            log_every: 1,
        }
    }
    pub fn with_qat(mut self, value: bool) -> Self {
        if value {
            self.qat_model
                .get_or_insert_with(|| self.model.zeros_like());
        } else {
            self.qat_model = None;
        }
        self
    }
    pub fn silent(mut self, value: bool) -> Self {
        self.silent = value;
        self
    }
    pub fn batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }
    pub fn log_every(mut self, value: usize) -> Self {
        self.log_every = value.max(1);
        self
    }
    fn print_training_init(&self, epochs: usize, train_dataset_len: usize) {
        if self.silent {
            return;
        }

        print_rule("=");
        println!(
            "Samples per Dataset: {} | Epochs: {} | Learning Rate: {} | Batch size: {} | QAT: {}",
            pretty(train_dataset_len),
            pretty(epochs),
            pretty(self.optimizer.lr()),
            self.batch_size,
            if self.qat_model.is_some() {
                "Enabled"
            } else {
                "Disabled"
            }
        );
        print_rule("=");
    }
    fn print_epoch(
        &self,
        epochs: usize,
        i: usize,
        avg_loss: f64,
        epoch_width: usize,
        window_start: &mut Instant,
        window_samples: &mut usize,
    ) {
        if self.silent {
            return;
        }
        if i.is_multiple_of(self.log_every) || i == epochs {
            let elapsed = window_start.elapsed();
            let samples_per_sec = throughput(*window_samples, elapsed);
            println!(
                "Epoch [{i:>width$}/{epochs}] - Time: {:>6.2?} | Throughput: {} samples/s | Loss ({:?}): {}",
                elapsed,
                pretty(samples_per_sec.max(0.0).round() as usize),
                self.loss_fn,
                pretty(avg_loss),
                width = epoch_width,
            );
            *window_start = Instant::now();
            *window_samples = 0;
        }
    }
    fn print_training_end(&self, training_duration: Duration, avg_throughput: f64) {
        if self.silent {
            return;
        }

        print_rule("=");

        println!(
            "Training completed in: {:.2?} | Average throughput: {} samples/s",
            training_duration,
            pretty(avg_throughput.max(0.0).round() as usize)
        );

        print_rule("=");

        println!();
    }
    pub fn fit(&mut self, train_dataset: &[Data], epochs: usize) -> TrainSummary {
        let mut loader = DataMiniBatches::new(train_dataset, self.batch_size, 42);

        self.print_training_init(epochs, train_dataset.len());
        let mut scales = Vec::new();

        let epoch_width = epochs.max(1).to_string().len();
        let training_start = Instant::now();
        let mut final_loss = 0.0;

        let mut window_start = Instant::now();
        let mut window_samples = 0usize;

        for i in 1..=epochs {
            let mut epoch_loss_sum = 0.0;
            let mut processed_samples = 0;
            for batch in loader.iter_epoch() {
                let cache;
                let grad = if let Some(effective_model) = self.qat_model.as_mut() {
                    self.model.quantize_into(effective_model, &mut scales);
                    cache = effective_model.forward_cache(&batch.input);
                    backward(effective_model, &cache, &batch.target, &self.loss_fn)
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
            window_samples += processed_samples;
            let avg_loss = epoch_loss_sum / processed_samples as f64;
            final_loss = avg_loss as f32;
            self.print_epoch(
                epochs,
                i,
                avg_loss,
                epoch_width,
                &mut window_start,
                &mut window_samples,
            );
        }

        let training_duration = training_start.elapsed();
        let avg_throughput = throughput(train_dataset.len() * epochs, training_duration);
        self.print_training_end(training_duration, avg_throughput);
        TrainSummary {
            epochs,
            duration: training_duration,
            final_loss,
            avg_throughput,
            qat: self.qat_model.is_some(),
        }
    }
}

use crate::format::{begin_section, pretty, write_banner, write_field, write_rule};
use crate::{AnyModule, Data, Sequential, TrainSummary};
use std::fmt;

#[derive(Debug, Clone)]
pub struct QuantizedMetrics {
    pub model: Sequential,
    pub scales: Vec<f32>,
    pub accuracy: Option<f32>,
    pub sparsity: Option<(f32, Vec<f32>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct SizeMetrics {
    pub weights: usize,
    pub biases: usize,
    pub layers: usize,
    pub f32_bytes: usize,
    pub packed_t3_bytes: usize,
}

impl SizeMetrics {
    pub fn from_model(model: &Sequential) -> Self {
        let weights = model.num_weights();
        let params = model.num_parameters();
        let biases = params - weights;
        let layers = model
            .modules
            .iter()
            .filter(|m| matches!(m, AnyModule::Linear(_)))
            .count();

        let f32_bytes = params * size_of::<f32>();
        let packed_t3_bytes = weights.div_ceil(4) + (layers + biases) * size_of::<u32>();

        Self {
            weights,
            biases,
            layers,
            f32_bytes,
            packed_t3_bytes,
        }
    }

    pub fn compression_ratio(&self) -> f32 {
        if self.packed_t3_bytes == 0 {
            0.0
        } else {
            self.f32_bytes as f32 / self.packed_t3_bytes as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelReport<'a> {
    model: &'a Sequential,
    pub train: Option<TrainSummary>,
    pub f32_accuracy: Option<f32>,
    pub quantized: Option<QuantizedMetrics>,
    pub size: Option<SizeMetrics>,
}

impl<'a> ModelReport<'a> {
    pub fn new(model: &'a Sequential) -> Self {
        Self {
            model,
            train: None,
            f32_accuracy: None,
            quantized: None,
            size: None,
        }
    }

    fn ensure_quantized(&mut self) -> &mut QuantizedMetrics {
        self.quantized.get_or_insert_with(|| {
            let (model, scales) = self.model.quantize();
            QuantizedMetrics {
                model,
                scales,
                accuracy: None,
                sparsity: None,
            }
        })
    }

    pub fn with_train(mut self, summary: TrainSummary) -> Self {
        self.train = Some(summary);
        self
    }

    pub fn with_accuracy(mut self, dataset: &[Data]) -> Self {
        self.f32_accuracy = Some(self.model.evaluate_accuracy(dataset));
        self
    }

    pub fn with_ternary_accuracy(mut self, dataset: &[Data]) -> Self {
        let q = self.ensure_quantized();
        q.accuracy = Some(q.model.evaluate_accuracy(dataset));
        self
    }

    pub fn with_sparsity(mut self) -> Self {
        let q = self.ensure_quantized();
        q.sparsity = Some(q.model.sparsity(&q.scales));
        self
    }

    pub fn with_size(mut self) -> Self {
        self.size = Some(SizeMetrics::from_model(self.model));
        self
    }

    pub fn ternary_accuracy(&self) -> Option<f32> {
        self.quantized.as_ref()?.accuracy
    }
    pub fn delta(&self) -> Option<f32> {
        Some(self.ternary_accuracy()? - self.f32_accuracy?)
    }
    fn ternary_mode(&self) -> &'static str {
        if self.train.as_ref().is_some_and(|t| t.qat) {
            "QAT"
        } else {
            "PTQ"
        }
    }
}

impl ModelReport<'_> {
    fn fmt_train(&self, f: &mut fmt::Formatter<'_>, t: &TrainSummary) -> fmt::Result {
        let throughput = pretty(t.avg_throughput.max(0.0).round() as usize);
        write_field(f, "Mode", if t.qat { "QAT" } else { "F32" })?;
        write_field(f, "Epochs", pretty(t.epochs))?;
        write_field(
            f,
            "Duration",
            format_args!("{:.2?} ({throughput} samples/s)", t.duration),
        )?;
        write_field(f, "Final Loss", pretty(t.final_loss))
    }

    fn fmt_accuracy(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(acc) = self.f32_accuracy {
            write_field(f, "F32", format_args!("{acc:.2}%"))?;
        }
        if let Some(t3) = self.ternary_accuracy() {
            let label = match self.ternary_mode() {
                "QAT" => "Ternary (QAT)",
                _ => "Ternary (PTQ)",
            };
            match self.delta() {
                Some(delta) => {
                    write_field(f, label, format_args!("{t3:.2}% ({delta:+.2}% vs F32)"))?
                }
                None => write_field(f, label, format_args!("{t3:.2}%"))?,
            }
        }
        Ok(())
    }
    fn fmt_sparsity(f: &mut fmt::Formatter<'_>, total_sp: f32, layer_sp: &[f32]) -> fmt::Result {
        write_field(f, "Total", format_args!("{:.2}%", total_sp * 100.0))?;
        write_field(
            f,
            "Per Layer",
            fmt::from_fn(|f| {
                for (i, sp) in layer_sp.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" | ")?;
                    }
                    write!(f, "L{i} {:.2}%", sp * 100.0)?;
                }
                Ok(())
            }),
        )
    }
    fn fmt_size(f: &mut fmt::Formatter<'_>, s: &SizeMetrics) -> fmt::Result {
        write_field(
            f,
            "Parameters",
            format_args!(
                "{} weights + {} biases ({} layers)",
                pretty(s.weights),
                pretty(s.biases),
                s.layers
            ),
        )?;
        write_field(
            f,
            "Footprint",
            format_args!(
                "{:.2} KB (F32) -> {:.2} KB (Packed T3 + Q8.24) [{:.2}x smaller]",
                s.f32_bytes as f32 / 1024.0,
                s.packed_t3_bytes as f32 / 1024.0,
                s.compression_ratio()
            ),
        )
    }
}

impl fmt::Display for ModelReport<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_banner(f, "TARS EVALUATION REPORT")?;

        let mut has_section = false;

        if let Some(t) = &self.train {
            begin_section(f, "Train", &mut has_section)?;
            self.fmt_train(f, t)?;
        }
        if self.f32_accuracy.is_some() || self.ternary_accuracy().is_some() {
            begin_section(f, "Accuracy", &mut has_section)?;
            self.fmt_accuracy(f)?;
        }
        if let Some((total_sp, layer_sp)) =
            self.quantized.as_ref().and_then(|q| q.sparsity.as_ref())
        {
            begin_section(f, "Sparsity", &mut has_section)?;
            Self::fmt_sparsity(f, *total_sp, layer_sp)?;
        }
        if let Some(s) = &self.size {
            begin_section(f, "Size", &mut has_section)?;
            Self::fmt_size(f, s)?;
        }

        write_rule(f, "=")
    }
}

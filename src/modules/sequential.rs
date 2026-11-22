use crate::Activation;
use crate::AnyModule;
use crate::Linear;
use crate::Module;
use crate::tensor_core::Tensor;

#[derive(Clone, Debug)]
pub struct Sequential {
    pub modules: Vec<AnyModule>,
    input_size: usize,
}

#[derive(Clone, Debug)]
pub struct ForwardCache {
    trace: Vec<Tensor<f32>>,
}

impl ForwardCache {
    pub fn input(&self) -> &Tensor<f32> {
        &self.trace[0]
    }

    pub fn output(&self) -> &Tensor<f32> {
        &self.trace[self.trace.len() - 1]
    }

    pub fn module_input(&self, i: usize) -> &Tensor<f32> {
        &self.trace[i]
    }

    pub fn module_output(&self, i: usize) -> &Tensor<f32> {
        &self.trace[i + 1]
    }

    pub fn len(&self) -> usize {
        self.trace.len()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            trace: Vec::with_capacity(capacity),
        }
    }
}

impl Sequential {
    pub fn new(input_size: usize, prox_size: usize) -> Self {
        assert_ne!(input_size, 0);
        assert_ne!(prox_size, 0);
        let mut initial_modules = Vec::new();
        let linear = Linear::random(input_size, prox_size);
        initial_modules.push(AnyModule::Linear(linear));

        Self {
            modules: initial_modules,
            input_size: prox_size,
        }
    }

    pub fn linear(mut self, output_size: usize) -> Self {
        assert_ne!(output_size, 0);
        let linear = Linear::random(self.input_size, output_size);
        self.modules.push(AnyModule::Linear(linear));
        self.input_size = output_size;
        self
    }

    pub fn relu(mut self) -> Self {
        //TODO: apply random function in previous linear
        self.modules.push(AnyModule::Activation(Activation::Relu));
        self
    }

    pub fn sigmoid(mut self) -> Self {
        //TODO: apply random function in previou:s linear
        self.modules
            .push(AnyModule::Activation(Activation::Sigmoid));
        self
    }
    pub fn forward_cache(&self, input: &Tensor<f32>) -> ForwardCache {
        let mut cache = ForwardCache::with_capacity(self.modules.len() + 1);
        cache.trace.push(input.clone());

        for module in &self.modules {
            let prev = cache.trace.last().unwrap();
            let out = module.forward(prev);
            cache.trace.push(out);
        }

        cache
    }
    pub fn num_weights(&self) -> usize {
        self.modules
            .iter()
            .map(|m| match m {
                AnyModule::Linear(l) => l.weights.numel(),
                AnyModule::Activation(_) => 0,
            })
            .sum()
    }
    pub fn num_parameters(&self) -> usize {
        self.modules
            .iter()
            .map(|m| match m {
                AnyModule::Linear(l) => l.weights.numel() + l.bias.numel(),
                AnyModule::Activation(_) => 0,
            })
            .sum()
    }

    pub fn quantize(&self) -> (Sequential, Vec<f32>) {
        let mut out = self.clone();
        let mut scales = Vec::new();
        for m in &mut out.modules {
            match m {
                AnyModule::Linear(l) => {
                    let w0;
                    (*l, w0) = l.quantize();
                    scales.push(w0);
                }

                AnyModule::Activation(_) => (),
            }
        }
        (out, scales)
    }
    pub fn sparsity(&self, scales: &[f32]) -> (f32, Vec<f32>) {
        let mut sparsities = Vec::new();
        let mut total_zeros = 0;
        let mut pos = 0;
        for m in &self.modules {
            match m {
                AnyModule::Linear(l) => {
                    let zeros;
                    let s0;
                    (zeros, s0) = l.sparsity(scales[pos]);
                    total_zeros += zeros;

                    sparsities.push(s0);
                    pos += 1;
                }
                _ => (),
            }
        }

        (total_zeros as f32 / self.num_weights() as f32, sparsities)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut output = input.clone();

        for module in &self.modules {
            output = module.forward(&output);
        }

        output
    }
}

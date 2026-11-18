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
    pub fn new(input_size: usize) -> Self {
        Self {
            modules: Vec::new(),
            input_size,
        }
    }

    pub fn linear(mut self, output_size: usize) -> Self {
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

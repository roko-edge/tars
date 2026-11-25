pub use crate::tensor_core::Tensor;

#[derive(Clone, Debug)]
pub struct Data {
    pub input: Tensor<f32>,
    pub target: Tensor<f32>,
}

impl Data {
    pub fn new(input: &[f32], target: &[f32]) -> Self {
        Self {
            input: Tensor::from_vec(input.to_vec(), vec![input.len()]),
            target: Tensor::from_vec(target.to_vec(), vec![target.len()]),
        }
    }
}

pub struct Batch {
    pub input: Tensor<f32>,
    pub target: Tensor<f32>,
}

impl Batch {
    fn from_samples(samples: &[Data]) -> Self {
        let b = samples.len();
        let in_sz = samples[0].input.shape()[0];
        let out_sz = samples[0].target.shape()[0];

        let mut inputs = Vec::with_capacity(b * in_sz);
        let mut targets = Vec::with_capacity(b * out_sz);

        for sample in samples {
            for j in 0..in_sz {
                inputs.push(sample.input.get(&[j]));
            }
            for j in 0..out_sz {
                targets.push(sample.target.get(&[j]));
            }
        }

        Self {
            input: Tensor::from_vec(inputs.to_vec(), vec![b, in_sz]),
            target: Tensor::from_vec(targets.to_vec(), vec![b, out_sz]),
        }
    }

    pub fn len(&self) -> usize {
        self.input.shape()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct DataMiniBatches {
    data: Vec<Data>,
    batch_size: usize,
    seed: u64,
    epoch: u64,
}

pub struct XorRandomizer {
    rng: u64,
}

impl XorRandomizer {
    pub fn new(seed: u64) -> Self {
        assert_ne!(seed, 0, "seed must be diferent than 0");
        Self { rng: seed }
    }

    pub fn next_rng(&mut self) -> u64 {
        let mut rng = self.rng;

        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        self.rng = rng;

        rng
    }
}

impl DataMiniBatches {
    pub fn new(data: &[Data], batch_size: usize, seed: u64) -> Self {
        assert!(batch_size > 0);
        assert!(!data.is_empty());

        Self {
            data: data.to_vec(),
            batch_size,
            seed,
            epoch: 0,
        }
    }

    pub fn iter_epoch(&mut self) -> impl Iterator<Item = Batch> + '_ {
        let mut rng = XorRandomizer::new(self.seed.wrapping_add(self.epoch));
        self.epoch += 1;

        let mut indices: Vec<usize> = (0..self.data.len()).collect();

        for i in (1..self.data.len()).rev() {
            let j = (rng.next_rng() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let chunks: Vec<Vec<usize>> = indices
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let data = &self.data;

        chunks.into_iter().map(move |chunk| {
            let samples: Vec<Data> = chunk.into_iter().map(|i| data[i].clone()).collect();

            Batch::from_samples(&samples)
        })
    }
}

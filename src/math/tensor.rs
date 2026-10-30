use core::panic;
use std::{sync::Arc, vec};

#[derive(Debug)]
pub enum DType {
    F32,
    I8,
    T3,
}

#[derive(Debug, Clone)]
pub enum Storage {
    F32(Arc<Vec<f32>>),
    I8 { data: Arc<Vec<i8>>, scale: f32 },
    T3 { data: Arc<Vec<u8>>, scale: f32 },
}

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl Tensor {
    pub fn new(storage: Storage, shape: Vec<usize>) -> Self {
        Self::from(storage, shape, 0)
    }

    fn from(storage: Storage, shape: Vec<usize>, offset: usize) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            storage,
            shape,
            strides,
            offset,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product(), "");
        Self::new(Storage::F32(Arc::new(data)), shape)
    }

    pub fn from_slice(data: &[f32], shape: Vec<usize>) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut actual_stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = actual_stride;
            actual_stride *= shape[i];
        }
        strides
    }

    pub fn step_logical(shape: &Vec<usize>, v: &mut Vec<usize>) {
        assert_eq!(shape.len(), 0, "Vector size can't be zero");

        v[shape.len() - 1] += 1;

        let mut i = shape.len() - 1;

        while v[i] >= shape[i] {
            v[i] = 0;

            if i > 0 {
                v[i - 1] += 1;
                i -= 1;
            }
        }
    }

    pub fn make_contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            let mut tensor = Self::zeros(self.shape.clone(), self.dtype());
            let mut idx = vec![0usize; self.shape.len()];

            for _ in 0..self.numel() {
                tensor.set(&idx, self.get(&idx));
                Self::step_logical(&self.shape, &mut idx);
            }

            tensor
        }
    }

    // fn make_contiguous_mut(&self) -> Self {}

    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::contiguous_strides(&self.shape)
            && self.offset == 0
            && match &self.storage {
                Storage::F32(storage) => storage.len() != 0,
                Storage::I8 { .. } => panic!(),
                Storage::T3 { .. } => panic!(),
            }
    }

    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let numel = shape.iter().product();

        let storage = match dtype {
            DType::F32 => Storage::F32(Arc::new(vec![0.0; numel])),
            DType::I8 => Storage::I8 {
                data: Arc::new(vec![0i8; numel]),
                scale: 0.0,
            },
            DType::T3 => Storage::T3 {
                data: Arc::new(vec![0u8; (numel + 3).div_ceil(4)]),
                scale: 0.0,
            },
        };

        Self::from(storage, shape, 0)
    }

    fn storage_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "indices must match with shape len"
        );
        let idx: usize = indices
            .iter()
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            .map(|((&index, &dim), &stride)| {
                assert!(index < dim, "tensor index out of bounds!");
                index * stride
            })
            .sum();
        self.offset + idx
    }

    pub fn get(&self, index: &[usize]) -> f32 {
        let idx = self.storage_index(index);
        match &self.storage {
            Storage::F32(storage) => storage[idx],
            Storage::I8 { .. } => panic!(),
            Storage::T3 { .. } => panic!(),
        }
    }

    pub fn set(&mut self, index: &[usize], value: f32) {
        let idx = self.storage_index(index);
        let storage = match &mut self.storage {
            Storage::F32(storage) => Arc::make_mut(storage),
            Storage::I8 { .. } => panic!(),
            Storage::T3 { .. } => panic!(),
        };
        storage[idx] = value;
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn dtype(&self) -> DType {
        match &self.storage {
            Storage::F32(_) => DType::F32,
            Storage::I8 { .. } => DType::I8,
            Storage::T3 { .. } => DType::T3,
        }
    }

    pub fn storage(&self) -> Arc<Vec<f32>> {
        match &self.storage {
            Storage::F32(storage) => storage.clone(),
            Storage::I8 { .. } => panic!(),
            Storage::T3 { .. } => panic!(),
        }
    }

    pub fn transpose_mut(&mut self, dim0: usize, dim1: usize) {
        assert!(
            dim0 < self.shape.len() && dim1 < self.shape.len(),
            "dimension/s out of bounds"
        );

        self.shape.swap(dim0, dim1);
        self.strides.swap(dim0, dim1);
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(
            dim0 < self.shape.len() && dim1 < self.shape.len(),
            "dimension/s out of bounds"
        );

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "");
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) + other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn add_mut(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "");
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) + other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "");
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) - other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn sub_mut(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "");
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) - other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "");
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) * other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn mul_mut(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "");
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) * other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn div(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "");
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) / other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn div_mut(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape, "");
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) / other.get(&idx));
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn mul_scalar(&self, s: f32) -> Self {
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) * s);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn mul_scalar_mut(&mut self, s: f32) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) * s);
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn div_scalar(&self, s: f32) -> Self {
        assert!(s != 0.0);
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, self.get(&idx) / s);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn div_scalar_mut(&mut self, s: f32) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, self.get(&idx) / s);
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let mut out = Tensor::zeros(self.shape.clone(), self.dtype());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, f(self.get(&idx)));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn map_mut(&mut self, f: impl Fn(f32) -> f32) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, f(self.get(&idx)));
            Self::step_logical(&self.shape, &mut idx);
        }
    }

    pub fn sum(&self) -> f32 {
        let mut out = 0.0;
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out += self.get(&idx);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }

    pub fn outer(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 1);
        assert_eq!(other.shape.len(), 1);

        let mut out = Tensor::zeros(vec![self.shape[0], other.shape[0]], self.dtype());
        for i in 0..self.numel() {
            for j in 0..other.numel() {
                out.set(&[i, j], self.get(&[i]) * other.get(&[j]));
            }
        }
        out
    }

    pub fn matvec(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 1);
        assert_eq!(self.shape[1], other.shape[0]);
        let mut out = Tensor::zeros(vec![self.shape[0]], self.dtype());
        let mut acc;
        for i in 0..self.shape[0] {
            acc = 0.0;
            for j in 0..self.shape[1] {
                acc += self.get(&[i, j]) * other.get(&[j]);
            }
            out.set(&[i], acc);
        }
        out
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(self.shape[1], other.shape[0]);
        let mut out = Tensor::zeros(vec![self.shape[0], other.shape[1]], self.dtype());
        let mut acc;
        for k in 0..other.shape[1] {
            for i in 0..self.shape[0] {
                acc = 0.0;
                for j in 0..self.shape[1] {
                    acc += self.get(&[i, j]) * other.get(&[j, k]);
                }
                out.set(&[i, k], acc);
            }
        }
        out
    }
    fn add_bias(&mut self, bias: &[f32]) {
        let b = self.shape[0];
        let out = self.shape[1];
        for i in 0..b {
            for j in 0..out {
                let val = self.get(&[i, j]) + bias[j];
                self.set(&[i, j], val);
            }
        }
    }
}

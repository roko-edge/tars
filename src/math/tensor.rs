use core::panic;
use num_traits::Zero;
use std::ops::{Add, Mul};
use std::{sync::Arc, vec};

// pub trait DType: Copy + Default + Add<Output = Self> + Mul<Output = Self> {}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    storage: Arc<Vec<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl<T> Tensor<T>
where
    T: Clone + Default + Zero,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::from(data, shape, 0)
    }

    fn from(data: Vec<T>, shape: Vec<usize>, offset: usize) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            storage: Arc::new(data),
            shape,
            strides,
            offset,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Self::from(data, shape, 0)
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
            let mut tensor = Self::zeros(self.shape.clone());
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
            && self.storage.len() != 0
    }

    // pub fn dtype(&self) -> DType {
    //     match &self.storage {
    //         Storage::F32(_) => DType::F32,
    //         Storage::I8 { .. } => DType::I8,
    //         Storage::T3 { .. } => DType::T3,
    //     }
    // }

    pub fn storage(&self) -> Arc<Vec<T>> {
        self.storage.clone()
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

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Self {
        let mut out = Tensor::zeros(self.shape.clone());
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

    pub fn sum(&self) -> T {
        let mut out = Zero::zero();
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

        let mut out = Tensor::zeros(vec![self.shape[0], other.shape[0]]);
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
        let mut out = Tensor::zeros(vec![self.shape[0]]);
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
        let mut out = Tensor::zeros(vec![self.shape[0], other.shape[1]]);
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
}

impl<T: Add + Mul> Tensor<T> {
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape, "");
        let mut out = Tensor::zeros(self.shape.clone());
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
        let mut out = Tensor::zeros(self.shape.clone());
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
        let mut out = Tensor::zeros(self.shape.clone());
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
        let mut out = Tensor::zeros(self.shape.clone());
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

    pub fn mul_scalar(&self, s<T>) -> Self {
        let mut out = Tensor::zeros(self.shape.clone());
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
        let mut out = Tensor::zeros(self.shape.clone());
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
}

impl<T: Clone> Tensor<T> {
    fn storage_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            indices.len(),
            self.shape.len(),
            "indices must match with shape len"
        ); let idx: usize = indices
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

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product(), "");
        Self::new(data, shape)
    }

    pub fn from_slice(data: &[T], shape: Vec<usize>) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn get(&self, index: &[usize]) -> T {
        let idx = self.storage_index(index);
        self.storage[idx].clone()
    }

    pub fn set(&mut self, index: &[usize], value: T) {
        let idx = self.storage_index(index);
        let storage = Arc::make_mut(&mut self.storage);
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
}

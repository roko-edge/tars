use num_traits::Zero;
use std::clone::Clone;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Sub};
use std::{sync::Arc, vec};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    T3,
}

// Backend operations used by Tensor<T> instead of indexing a Vec directly.
// Indices refer to logical elements
pub trait StorageOps<T>: Clone + Debug {
    fn from_vec(data: Vec<T>) -> Self;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> T;

    fn set(&mut self, index: usize, value: T);

    // Allocate storage containing 'len' copies of 'value'.
    fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::from_vec(vec![value; len])
    }
}

// Copy-on-write backend for regular elements such as f32.
impl<T: Clone + Debug> StorageOps<T> for Arc<Vec<T>> {
    fn from_vec(data: Vec<T>) -> Self {
        Arc::new(data)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn get(&self, index: usize) -> T {
        self[index].clone()
    }

    fn set(&mut self, index: usize, value: T) {
        Arc::make_mut(self)[index] = value;
    }
}

// "Sized" keywork means that the type size should be known at compile time
pub trait Element: Sized {
    type Storage: StorageOps<Self>;
    const DTYPE: DType;
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Element> {
    storage: <T as Element>::Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

// rhs = right hand side
pub trait MatMul<Rhs> {
    type Output;
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

// Geometry
impl<T: Element> Tensor<T> {
    pub fn new(data: T::Storage, shape: Vec<usize>) -> Self {
        let required: usize = shape.iter().product();
        assert!(data.len() >= required);
        Self::from(data, shape, 0)
    }

    fn from(data: T::Storage, shape: Vec<usize>, offset: usize) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            storage: data,
            shape,
            strides,
            offset,
        }
    }

    pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut actual_stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = actual_stride;
            actual_stride *= shape[i];
        }
        strides
    }

    pub fn step_logical(shape: &Vec<usize>, v: &mut Vec<usize>) {
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

    pub fn storage_idx(&self, indices: &[usize]) -> usize {
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

    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::contiguous_strides(&self.shape) && self.offset == 0
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

    pub fn shape(&self) -> &[usize] {
        &self.shape
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

    pub fn storage(&self) -> T::Storage {
        self.storage.clone()
    }

    pub fn storage_ref(&self) -> &T::Storage {
        &self.storage
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        let required: usize = shape.iter().product();
        assert_eq!(data.len(), required);
        Self::new(<T::Storage as StorageOps<T>>::from_vec(data), shape)
    }
}

// Copy and acess
impl<T> Tensor<T>
where
    T: Clone + Element,
{
    pub fn get(&self, index: &[usize]) -> T {
        self.storage.get(self.storage_idx(index))
    }

    pub fn set(&mut self, index: &[usize], value: T) {
        let idx = self.storage_idx(index);
        self.storage.set(idx, value);
    }

    pub fn from_slice(data: &[T], shape: Vec<usize>) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn map_mut(&mut self, f: impl Fn(T) -> T) {
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            self.set(&idx, f(self.get(&idx)));
            Self::step_logical(&self.shape, &mut idx);
        }
    }
}

// Zeroes and Mapping
impl<T> Tensor<T>
where
    T: Clone + Zero + Element,
{
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let storage = <T::Storage as StorageOps<T>>::filled(numel, T::zero());
        Self::from(storage, shape, 0)
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

    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let mut out = Tensor::zeros(self.shape.clone());
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out.set(&idx, f(self.get(&idx)));
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }
}

// Numerical operations

impl<T> Tensor<T>
where
    T: AddAssign + Element + Zero + Clone,
{
    pub fn sum(&self) -> T {
        let mut out = T::zero();
        let mut idx = vec![0usize; self.shape.len()];
        for _ in 0..self.numel() {
            out += self.get(&idx);
            Self::step_logical(&self.shape, &mut idx);
        }
        out
    }
}

impl<T> Tensor<T>
where
    T: Add<Output = T> + Sub<Output = T> + Element + Zero + Clone,
{
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
}

impl<T> Tensor<T>
where
    T: Clone + Mul<Output = T> + Div<Output = T> + Zero + Element,
{
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
}

impl<T> Tensor<T>
where
    T: Clone + Mul<Output = T> + Zero + Element,
{
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
}

impl<T> Tensor<T>
where
    T: Clone + AddAssign + Mul<Output = T> + Zero + Element,
{
    pub fn matvec(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 1);
        assert_eq!(self.shape[1], other.shape[0]);
        let mut out = Tensor::zeros(vec![self.shape[0]]);
        let mut acc;
        for i in 0..self.shape[0] {
            acc = T::zero();
            for j in 0..self.shape[1] {
                acc += self.get(&[i, j]) * other.get(&[j]);
            }
            out.set(&[i], acc);
        }
        out
    }
}

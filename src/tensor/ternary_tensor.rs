use crate::tensor_core::*;
use std::sync::Arc;

#[repr(i8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T3 {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl T3 {
    // converts from t3 to binary representation
    const fn bits(self) -> u64 {
        match self {
            Self::Zero => 0b00,
            Self::Pos => 0b01,
            Self::Neg => 0b10,
        }
    }

    // converts from bit representation to T3
    const fn from_bits(bits: u64) -> Self {
        match bits {
            0b00 => Self::Zero,
            0b01 => Self::Pos,
            0b10 => Self::Neg,
            0b11 => Self::Zero,
            _ => panic!("invalid packed T3 encoding"),
        }
    }

    // converts from t3 to f32
    pub const fn as_f32(self) -> f32 {
        self as i8 as f32
    }
}

// converts from i8 to T3
impl TryFrom<i8> for T3 {
    type Error = &'static str;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(Self::Neg),
            0 => Ok(Self::Zero),
            1 => Ok(Self::Pos),
            _ => Err("T3 must be -1, 0, or 1"),
        }
    }
}

/// 32 ternary elements per 64-bit word, with copy-on-write sharing.
#[derive(Clone, Debug)]
pub struct PackedT3Storage {
    words: Arc<Vec<u64>>,
    // number of logical ternary elements
    len: usize,
}

impl PackedT3Storage {
    const PER_WORD: usize = 32;
    const BITS_PER_ELEMENT: usize = 2;

    pub fn words(&self) -> &[u64] {
        self.words.as_slice()
    }
}

impl StorageOps<T3> for PackedT3Storage {
    fn from_vec(data: Vec<T3>) -> Self {
        let len = data.len();

        // Calculates the number of u64 words needed to store 'len' ternary values.
        // Each word holds PER_WORD elements. Round up to allocate an extra word
        // when the elements don't fit exactly into complete words.
        let word_count = len / Self::PER_WORD + usize::from(len % Self::PER_WORD != 0);
        let mut words = vec![0_u64; word_count];

        for (index, value) in data.into_iter().enumerate() {
            let word = index / Self::PER_WORD;
            let shift = (index % Self::PER_WORD) * Self::BITS_PER_ELEMENT;
            words[word] |= value.bits() << shift;
        }

        Self {
            words: Arc::new(words),
            len,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> T3 {
        assert!(index < self.len, "ternary storage index out of bounds");
        let word = index / Self::PER_WORD;
        let shift = (index % Self::PER_WORD) * Self::BITS_PER_ELEMENT;
        T3::from_bits((self.words[word] >> shift) & 0b11)
    }

    fn set(&mut self, index: usize, value: T3) {
        assert!(index < self.len, "ternary storage index out of bounds");
        let word = index / Self::PER_WORD;
        let shift = (index % Self::PER_WORD) * Self::BITS_PER_ELEMENT;
        let mask = 0b11_u64 << shift;
        let words = Arc::make_mut(&mut self.words);
        words[word] = (words[word] & !mask) | (value.bits() << shift);
    }
}

impl Element for T3 {
    type Storage = PackedT3Storage;
    const DTYPE: DType = DType::T3;
}

// Standard matrix multiplication: lhs [m, n] x rhs [n, p] -> [m, p].
impl MatMul<Tensor<T3>> for Tensor<f32> {
    type Output = Tensor<f32>;

    fn matmul(&self, rhs: &Tensor<T3>) -> Self::Output {
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();
        assert_eq!(lhs_shape.len(), 2);
        assert_eq!(rhs_shape.len(), 2);
        assert_eq!(lhs_shape[1], rhs_shape[0]);

        let (m, n, p) = (lhs_shape[0], lhs_shape[1], rhs_shape[1]);
        let mut out = Tensor::<f32>::zeros(vec![m, p]);

        for i in 0..m {
            for k in 0..p {
                let mut acc = 0.0_f32;
                for j in 0..n {
                    acc += self.get(&[i, j]) * rhs.get(&[j, k]).as_f32();
                }
                out.set(&[i, k], acc);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t3_to_binary() {
        let neg = T3::Neg;
        let zero = T3::Zero;
        let pos = T3::Pos;

        assert_eq!(neg.bits(), 0b10);
        assert_eq!(zero.bits(), 0b00);
        assert_eq!(pos.bits(), 0b01);
    }

    #[test]
    fn t3_to_f32() {
        let neg = T3::Neg;
        let zero = T3::Zero;
        let pos = T3::Pos;

        assert_eq!(neg.as_f32(), -1.0);
        assert_eq!(zero.as_f32(), 0.0);
        assert_eq!(pos.as_f32(), 1.0);
    }

    #[test]
    fn bin_to_t3() {
        assert_eq!(T3::from_bits(0b10), T3::Neg);
        assert_eq!(T3::from_bits(0b00), T3::Zero);
        assert_eq!(T3::from_bits(0b01), T3::Pos);
        assert_eq!(T3::from_bits(0b01), T3::Pos);
        assert_eq!(T3::from_bits(0b11), T3::Zero);
    }
}

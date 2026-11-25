use crate::tensor::ternary_tensor::T3;

const EPS: f32 = 1e-7;
const THRESHOLD_FACTOR: f32 = 0.9999999;

#[inline]
fn threshold(w0: f32) -> f32 {
    if w0 <= EPS {
        f32::INFINITY
    } else {
        THRESHOLD_FACTOR * w0
    }
}
#[inline]
pub fn absmean(weights: &[f32]) -> f32 {
    let sum: f32 = weights.iter().map(|w| w.abs()).sum();
    if sum > EPS {
        sum / weights.len() as f32
    } else {
        sum
    }
}
#[inline]
pub fn quantize_scalar(weight: f32, w0: f32) -> T3 {
    let t = threshold(w0);
    if weight > t {
        T3::Pos
    } else if weight < -t {
        T3::Neg
    } else {
        T3::Zero
    }
}
#[inline]
pub fn quantize_into_f32(src: &[f32], dst: &mut [f32]) -> f32 {
    assert_eq!(src.len(), dst.len());
    let w0 = absmean(src);
    let t = threshold(w0);
    dst.iter_mut().zip(src).for_each(|(d, &w)| {
        *d = if w > t {
            w0
        } else if w < -t {
            -w0
        } else {
            0.0
        };
    });
    w0
}

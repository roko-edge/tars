use crate::tensor::ternary_tensor::T3;
pub fn quantize_scalar(weight: f32, w0: f32) -> T3 {
    if w0 <= 1e-7 {
        return T3::Zero;
    }
    let threshold = 0.9999999 * w0;
    if weight > threshold {
        T3::Pos
    } else if weight < -threshold {
        T3::Neg
    } else {
        T3::Zero
    }
}

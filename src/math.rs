use rand::RngExt;

pub mod tensor;
pub use tensor::*;

pub mod tensor_it;

pub mod float_tensor;

pub fn random_vec(len: usize) -> Vec<f32> {
    let mut vector = vec![0.0; len];
    for e in &mut vector {
        *e = rand::random::<f32>() * 2.0 - 1.0;
    }
    vector
}

// rand::random_range() function generates a distribution with mean 1/2 and variance 1/12. But,
// our goal is to produce a pseudo-rng with mean 0 and variance 1 - according to xavier glorot. So,
// we apply the linear transformation T = srqt(3) * (2*u -1).
// T stretches u E [0, 1) to [-1, 1], making the mean 0.
// considering [-1, 1], var(x) = (sqrt(3) * sqrt(3)*1/2) / 12 = (2 * srqt(3)*1/2)
// / 12 = 12/12 = 1
// So, prng fn generates a distribution with mean 0 and var 1 - exactly what we want.

pub fn prng(low: f32, high: f32) -> f32 {
    let u: f32 = rand::random_range(low..high);
    3.0_f32.sqrt() * (2.0 * u - 1.0)
}

// This function returns a vector of n random weights sampled as prng(0, 1) / sqrt(n).
// This scaling is necessary because of the variance property: Var(aX) = a^2 * Var(X).
// Since Var(X) = 1 for prng(0, 1):
// => Var(X / sqrt(n)) = (1 / n) * Var(X) = (1 / n) * 1 = 1 / n.
// A variance of 1/n preserves forward-pass variance (LeCun initialization, or Xavier/Glorot
// initialization when n_in == n_out) - see Glorot & Bengio (2010):
// [https://www.researchgate.net/publication/215616968_Understanding_the_difficulty_of_training_deep_feedforward_neural_networks]

pub fn xavier(n: usize) -> Vec<f32> {
    (0..n).map(|_| prng(0.0, 1.0) / (n as f32).sqrt()).collect()
}

// This function returns a random number between from the discrete interval [-1, 0, 1]
// By standard, this strategy generates a distribution with with mean 0 and variance = 2/3.
// The mean is ok, but we need a variance = 1. To fix that, we use the fact that var(a*x) = a^2 *
// var(x). => c^2*(2/3) = 1 => c = sqrt(3/2). It means that we should multi the rng by the factor
// a = sqrt(3/2)

// Basis used rn: (Consider sqrt() as s())
// {-s(3/2), 0, s(3, 2)}

pub fn ternary_rng() -> f32 {
    let mut rng = rand::rng();
    let u = rng.random_range(0..3);

    let scale = (3.0_f32 / 2.0).sqrt();

    match u {
        0 => -scale,
        1 => 0.0,
        _ => scale,
    }
}

pub fn random_mat(lenx: usize, leny: usize) -> Vec<Vec<f32>> {
    let mut mat = vec![vec![0.0; lenx]; leny];
    for l in &mut mat {
        *l = random_vec(lenx);
    }
    mat
}

pub fn sigmoidf(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoidf_deriv_from_a(a: f32) -> f32 {
    a * (1.0 - a)
}

pub fn cost_derivative(p: f32, t: f32, out_sz: usize) -> f32 {
    2.0 * (p - t) / (out_sz as f32)
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn relu_deriv_from_z(z: f32) -> f32 {
    if z > 0.0 { 1.0 } else { 0.0 }
}

use rand::{SeedableRng, rngs::ChaCha8Rng};
use tars::rng_gen::*;

#[test]
fn prng_test() {
    let n = 1000000;

    let mut rng = ChaCha8Rng::seed_from_u64(67);

    let values: Vec<f32> = (0..n).map(|_| prng(&mut rng)).collect();

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    assert!(mean.abs() < 0.02);

    assert!((variance - 1.0).abs() < 0.02);

    let limit = 3.0_f32.sqrt();

    assert!(values.iter().all(|x| *x >= -limit && *x < limit));
}

#[test]
fn xavier_test() {
    use rand::SeedableRng;
    use rand::rngs::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(67);

    let fan_in = 1_000_000;
    let fan_out = 512;

    let values = xavier(&mut rng, fan_in, fan_out);

    let n = values.len() as f32;

    let mean = values.iter().sum::<f32>() / n;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

    let expected_variance = 2.0 / (fan_in + fan_out) as f32;

    let limit = (3.0 * expected_variance).sqrt();

    assert_eq!(values.len(), fan_in);

    assert!(mean.abs() < 0.01);

    assert!((variance - expected_variance).abs() < expected_variance * 0.05);

    assert!(values.iter().all(|x| x.abs() <= limit));
}

#[test]
fn ternary_distribution_test() {
    let n = 1000000;

    let mut rng = ChaCha8Rng::seed_from_u64(67);

    let values: Vec<f32> = (0..n).map(|_| prng(&mut rng)).collect();

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    println!("Variance: {}", variance);

    assert_eq!(values.len(), n);

    assert!(mean.abs() < 0.02);

    assert!((variance - 1.0).abs() < 0.02);
}

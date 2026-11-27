use rand::{SeedableRng, rngs::ChaCha8Rng};
use tars::Sequential;
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

    let values: Vec<f32> = (0..n).map(|_| ternary_rng(&mut rng)).collect();

    let scale = (3.0_f32 / 2.0).sqrt();

    let mut neg = 0usize;
    let mut zero = 0usize;
    let mut pos = 0usize;

    // counting the frequency of the random generated numbers
    for &v in &values {
        if v == -scale {
            neg += 1;
        } else if v == 0.0 {
            zero += 1;
        } else if v == scale {
            pos += 1;
        } else {
            panic!("unexpected ternary sample: {v}");
        }
    }

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    assert_eq!(values.len(), n);

    assert!(mean.abs() < 0.02);

    assert!((variance - 1.0).abs() < 0.02);

    let expected_freq = 1.0 / 3.0;
    for count in [neg, zero, pos] {
        let freq = count as f64 / n as f64;
        // the frequency of each value should be proportional
        assert!((freq - expected_freq).abs() < 0.01);
    }
}

#[test]
fn ternary_rng_is_deterministic() {
    let mut rng_a = ChaCha8Rng::seed_from_u64(67);
    let mut rng_b = ChaCha8Rng::seed_from_u64(67);

    let seq_a: Vec<f32> = (0..64).map(|_| ternary_rng(&mut rng_a)).collect();
    let seq_b: Vec<f32> = (0..64).map(|_| ternary_rng(&mut rng_b)).collect();

    assert_eq!(seq_a, seq_b);
}

#[test]
fn sequential_init_advances_rng_and_is_reproducible() {
    let m1 = Sequential::with_seed(4, 4, 123).linear(4);
    let m2 = Sequential::with_seed(4, 4, 123).linear(4);
    let m3 = Sequential::with_seed(4, 4, 124).linear(4);

    assert_eq!(m1.init_seed(), 123);

    let l0_w = m1.modules[0].as_linear().weights.as_slice().unwrap();
    let l1_w = m1.modules[1].as_linear().weights.as_slice().unwrap();
    let m2_l0_w = m2.modules[0].as_linear().weights.as_slice().unwrap();
    let m2_l1_w = m2.modules[1].as_linear().weights.as_slice().unwrap();
    let m3_l0_w = m3.modules[0].as_linear().weights.as_slice().unwrap();

    assert_eq!(l0_w, m2_l0_w);
    assert_eq!(l1_w, m2_l1_w);
    assert_ne!(l0_w, m3_l0_w);
    assert_ne!(l0_w, l1_w);
    assert_ne!(&l0_w[0..4], &l0_w[4..8]);
}

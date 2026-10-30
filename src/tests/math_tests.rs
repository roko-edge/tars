use crate::math::*;

// The highest precision we've got was:
// mean => 3 points precision,
// variation => 8 points precision.

// TODO
// Those current tests are just checking if the mean is around 0 and the variance is around 1, but
// they still don't check they specificities. For example, a ternary distribution should only
// generate -1, 0 or 1, or whatever basis we're using.

#[test]
fn prng_test() {
    let n = 1000000;

    let values: Vec<f32> = (0..n).map(|_| prng(0.0, 1.0)).collect();

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    assert!(mean.abs() < 0.02);

    assert!((variance - 1.0).abs() < 0.02);

    let limit = 3.0_f32.sqrt();

    assert!(values.iter().all(|x| *x >= -limit && *x < limit));
}

#[test]
fn xavier_test() {
    let n = 1000000;

    let values = xavier(n);

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    let expected_variance = 1.0 / n as f32;

    let limit = (3.0 / n as f32).sqrt();

    assert_eq!(values.len(), n);

    assert!(mean.abs() < 0.01);

    assert!((variance - expected_variance).abs() < expected_variance * 0.05);

    assert!(values.iter().all(|x| *x >= -limit && *x < limit));
}

#[test]
fn ternary_distribution_test() {
    let n = 1000000;

    let values: Vec<f32> = (0..n).map(|_| prng(0.0, 1.0)).collect();

    let mean = values.iter().sum::<f32>() / n as f32;

    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    println!("Variance: {}", variance);

    assert_eq!(values.len(), n);

    assert!(mean.abs() < 0.02);

    assert!((variance - 1.0).abs() < 0.02);

    assert!(variance <= 1.0);
}

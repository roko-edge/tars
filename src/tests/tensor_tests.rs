use crate::Tensor;

#[test]
fn mul_scalar_test() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let u = t.mul_scalar(2.0);
    assert_eq!(u.storage().as_slice(), &[2.0, 4.0, 6.0, 8.0]);

    let u = t.mul_scalar(5.0);
    assert_eq!(u.storage().as_slice(), &[5.0, 10.0, 15.0, 20.0]);

    let u = t.mul_scalar(0.0);
    assert_eq!(u.storage().as_slice(), &[0.0, 0.0, 0.0, 0.0]);

    let u = t.mul_scalar(1.0);
    assert_eq!(u.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let u = t.mul_scalar(-2.0);
    assert_eq!(u.storage().as_slice(), &[-2.0, -4.0, -6.0, -8.0]);

    let u = t.mul_scalar(0.5);
    assert_eq!(u.storage().as_slice(), &[0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn mul_scalar_mut_test() {
    let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    t.mul_scalar_mut(2.0);
    assert_eq!(t.storage().as_slice(), &[2.0, 4.0, 6.0, 8.0]);

    t.mul_scalar_mut(0.5);
    assert_eq!(t.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    t.mul_scalar_mut(-1.0);
    assert_eq!(t.storage().as_slice(), &[-1.0, -2.0, -3.0, -4.0]);

    t.mul_scalar_mut(3.0);
    assert_eq!(t.storage().as_slice(), &[-3.0, -6.0, -9.0, -12.0]);

    t.mul_scalar_mut(0.0);
    assert_eq!(t.storage().as_slice(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn div_scalar_test() {
    let t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);

    let u = t.div_scalar(2.0);
    assert_eq!(u.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let u = t.div_scalar(1.0);
    assert_eq!(u.storage().as_slice(), &[2.0, 4.0, 6.0, 8.0]);

    let u = t.div_scalar(4.0);
    assert_eq!(u.storage().as_slice(), &[0.5, 1.0, 1.5, 2.0]);

    let u = t.div_scalar(-2.0);
    assert_eq!(u.storage().as_slice(), &[-1.0, -2.0, -3.0, -4.0]);

    let u = t.div_scalar(0.5);
    assert_eq!(u.storage().as_slice(), &[4.0, 8.0, 12.0, 16.0]);

    let u = t.div_scalar(8.0);
    assert_eq!(u.storage().as_slice(), &[0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn div_scalar_mut_test() {
    let mut t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);

    t.div_scalar_mut(2.0);
    assert_eq!(t.storage().as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    t.div_scalar_mut(0.5);
    assert_eq!(t.storage().as_slice(), &[2.0, 4.0, 6.0, 8.0]);

    t.div_scalar_mut(-2.0);
    assert_eq!(t.storage().as_slice(), &[-1.0, -2.0, -3.0, -4.0]);

    t.div_scalar_mut(4.0);
    assert_eq!(t.storage().as_slice(), &[-0.25, -0.5, -0.75, -1.0]);

    t.div_scalar_mut(-1.0);
    assert_eq!(t.storage().as_slice(), &[0.25, 0.5, 0.75, 1.0]);
}

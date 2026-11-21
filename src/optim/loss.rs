use crate::tensor_core::Tensor;

pub trait Loss {
    fn forward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32;
    fn backward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32>;
}

pub struct MSE;
pub struct SoftmaxCE;

impl Loss for MSE {
    fn forward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "predictions and targets dont have the same shape"
        );
        let mut sum = 0.0;

        match predictions.shape() {
            [n] => {
                for i in 0..*n {
                    let diff = targets.get(&[i]) - predictions.get(&[i]);
                    sum += diff * diff;
                }
            }

            [b, out_sz] => {
                for i in 0..*b {
                    for j in 0..*out_sz {
                        let diff = targets.get(&[i, j]) - predictions.get(&[i, j]);
                        sum += diff * diff;
                    }
                }
            }
            _ => panic!("MSE only support 1d or 2d tensors"),
        }

        sum / (predictions.numel() as f32)
    }

    fn backward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "predictions and targets dont have the same shape"
        );

        match predictions.shape() {
            [n] => {
                let scale = 2.0 / (*n as f32);
                let mut grad = Tensor::zeros(vec![*n]);

                for i in 0..*n {
                    let p = predictions.get(&[i]);
                    let t = targets.get(&[i]);
                    grad.set(&[i], scale * (p - t));
                }

                grad
            }

            [b, out_sz] => {
                let scale = 2.0 / ((*b * *out_sz) as f32);
                let mut grad = Tensor::zeros(vec![*b, *out_sz]);

                for i in 0..*b {
                    for j in 0..*out_sz {
                        let p = predictions.get(&[i, j]);
                        let t = targets.get(&[i, j]);
                        grad.set(&[i, j], scale * (p - t));
                    }
                }

                grad
            }

            _ => panic!("MSE only support 1d or 2d tensors"),
        }
    }
}

impl SoftmaxCE {
    fn softmax(elements: &Tensor<f32>) -> Tensor<f32> {
        match elements.shape() {
            [n] => {
                let mut max_element = f32::NEG_INFINITY;
                let mut probs = Tensor::zeros(vec![*n]);

                for i in 0..*n {
                    let v = elements.get(&[i]);

                    if v > max_element {
                        max_element = v;
                    }
                }

                let mut sum = 0.0;

                for i in 0..*n {
                    let exp_val = (elements.get(&[i]) - max_element).exp();
                    probs.set(&[i], exp_val);
                    sum += exp_val;
                }

                for i in 0..*n {
                    let v = probs.get(&[i]) / sum;
                    probs.set(&[i], v);
                }

                probs
            }

            [b, out_sz] => {
                let mut probs = Tensor::zeros(vec![*b, *out_sz]);

                for i in 0..*b {
                    let mut max_element = f32::NEG_INFINITY;

                    for j in 0..*out_sz {
                        let v = elements.get(&[i, j]);

                        if v > max_element {
                            max_element = v;
                        }
                    }

                    let mut sum = 0.0;

                    for j in 0..*out_sz {
                        let exp_val = (elements.get(&[i, j]) - max_element).exp();
                        probs.set(&[i, j], exp_val);
                        sum += exp_val;
                    }

                    for j in 0..*out_sz {
                        let v = probs.get(&[i, j]) / sum;
                        probs.set(&[i, j], v);
                    }
                }

                probs
            }
            _ => panic!("Softmax only support 1d or 2d tensors"),
        }
    }
}

impl Loss for SoftmaxCE {
    fn forward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "logits and targets dont have the same shape"
        );
        let probs = SoftmaxCE::softmax(predictions);

        match predictions.shape() {
            [n] => {
                let mut loss = 0.0;

                for i in 0..*n {
                    let y = targets.get(&[i]);

                    if y > 0.0 {
                        let p = probs.get(&[i]).max(1e-7);
                        loss -= y * p.ln();
                    }
                }

                loss
            }

            [b, out_sz] => {
                let mut loss = 0.0;

                for i in 0..*b {
                    for j in 0..*out_sz {
                        let y = targets.get(&[i, j]);

                        if y > 0.0 {
                            let p = probs.get(&[i, j]).max(1e-7);
                            loss -= y * p.ln();
                        }
                    }
                }

                loss / (*b as f32)
            }

            _ => {
                panic!("Softmax only supports 1D or 2D tensors")
            }
        }
    }

    fn backward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32> {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "predictions and targets dont have the same shape"
        );
        let probs = SoftmaxCE::softmax(predictions);

        match predictions.shape() {
            [n] => {
                let mut grad = Tensor::zeros(vec![*n]);

                for i in 0..*n {
                    grad.set(&[i], probs.get(&[i]) - targets.get(&[i]));
                }

                grad
            }

            [b, out_sz] => {
                let mut grad = Tensor::zeros(vec![*b, *out_sz]);
                let scale = 1.0 / (*b as f32);

                for i in 0..*b {
                    for j in 0..*out_sz {
                        let diff = probs.get(&[i, j]) - targets.get(&[i, j]);
                        grad.set(&[i, j], diff * scale);
                    }
                }

                grad
            }

            _ => {
                panic!("Softmax only supports 1D or 2D tensors")
            }
        }
    }
}

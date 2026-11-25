# Specification: Training Stack and Execution Contracts

**Document type:** Engineering specification (normative contract).
**Status:** Normative target contract. The [public API](../../API.md#3-training-operations)
describes the implemented training interface and known deviations.
The [delivery plan](../../project/V0_DELIVERY_PLAN.md#351-how-should-the-public-training-path-work-proposed)
tracks proposed refinements to the implemented high-level trainer.
**Revision date:** 2026-11-25 (maintainer decision: updated with Section 1.7; Sections 1.1–1.6 frozen 2026-11-02).
**Owns:** Frozen software training interfaces across model evaluation, loss computation, gradient tracking, optimizer step execution, batch loading, and the wire export boundary.
**Does not cover:** Mathematical quantization operators ([`QUANT.md`](QUANT.md)), tensor memory strides ([`TENSOR.md`](TENSOR.md)), or hardware-side FSM execution ([`ARITHMETIC_Q8_24.md`](ARITHMETIC_Q8_24.md)).

---

## 1. Frozen Training Interfaces

To eliminate architectural rework during parallel development across tensor migration, hardware export, and QAT, the six training interfaces are frozen as normative contracts.

### 1.1 Module and Composite Forward

Every neural network layer implements the `Module` trait over float tensors:

```rust
pub trait Module {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32>;
}
```

The composite model (`Sequential`) executes forward evaluation while recording intermediate layer inputs in a temporary forward trace:

```rust
pub struct ForwardTrace {
    pub inputs: Vec<Tensor<f32>>,
    pub output: Tensor<f32>,
}

impl Sequential {
    pub fn forward_record(&self, input: &Tensor<f32>) -> ForwardTrace;
    pub fn backward(&self, trace: &ForwardTrace, grad_output: &Tensor<f32>) -> Grad;
}
```

`backward` consumes the trace recorded during `forward_record` and never re-evaluates forward activations (conforming to [`MODEL_TRAIT.md`](MODEL_TRAIT.md), Section 2.2).

### 1.2 Loss Function Contract

All training objectives implement the unified `Loss` trait:

```rust
pub trait Loss {
    fn forward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32;
    fn backward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> Tensor<f32>;
}
```

The contract for `Loss::backward` specifies that the returned tensor is the analytical gradient of the scalar loss with respect to the pre-loss predictions or logits:

$$\nabla_{\hat{y}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

Supported implementations:

1. **Mean Squared Error (`MSE`):** Retained for OR and XOR baseline regressions:
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2, \qquad \frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \hat{y}_i} = \frac{2}{N} (\hat{y}_i - y_i)$$
2. **Softmax Cross-Entropy (`SoftmaxCE`):** Evaluated over raw unnormalized logits for multi-class classification (MNIST):
   $$p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad \mathcal{L}_{\text{CE}} = - \sum_{k} y_k \ln(p_k), \qquad \frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = p_k - y_k$$

### 1.3 Optimizer Contract

Optimizers update composite model parameters in-place given computed gradients:

```rust
pub trait Optimizer {
    fn lr(&self) -> f32;
    fn step(&mut self, model: &mut Sequential, grad: &Grad);
}
```

`lr()` reports the configured learning rate. Batch Gradient Descent (`BGD`)
is the release reference. Adam implements the same interface and keeps its
moment state in the optimizer instance.

### 1.4 Seeded Batch Loader

Batch creation and dataset iteration require reproducible deterministic state:

```rust
pub struct Batch {
    pub inputs: Tensor<f32>,
    pub targets: Tensor<f32>,
}

pub struct DataLoader {
    pub batch_size: usize,
    pub seed: u64,
}

impl DataLoader {
    pub fn new(data: &[Data], batch_size: usize, seed: u64) -> Self;
    pub fn iter_epoch(&mut self) -> impl Iterator<Item = Batch>;
}
```

The random seed is an explicit recorded parameter. Two training runs configured with identical seeds MUST generate bit-for-bit identical batch sequences and gradient updates.

### 1.5 Stable Wire Export Boundary

The wire format is independent of the storage used by `Tensor<T>`. The
[memory format reference](../MEM_FORMAT.md) records the current artifact
names, encodings, producers, and consumers. v0.1.0 uses an assembled wrapper
and independent integer expected outputs for release parity. `program.mem`
belongs to the programmable top in v0.1.1.
[Scope decision](../../project/research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026)

### 1.6 Quantizer and Straight-Through Estimation

The quantizer maps latent continuous weights to discrete ternary states and computes layer scales $W_0$ (one per layer in v0.1.0):

$$W_0 = \frac{1}{mn} \sum_{ij} |W_{ij}|, \qquad Q(W) = \text{clip}\left(\text{round}\left(\frac{W}{W_0}\right), -1, +1\right)$$

```rust
pub struct QuantizedWeights {
    pub codes: Vec<u8>, // packed 2-bit codes, canonical emission (00 for
                        // zero), row-major — the exact wt.mem byte layout
    pub scale: f32,     // W0 — one per layer (v0.1.0)
}

pub trait Quantizer {
    /// absmean over the whole tensor (per-layer W0).
    fn compute_scale(&self, weights: &Tensor<f32>) -> f32;
    /// codes + scale; the single source for export and dequantization.
    fn quantize(&self, weights: &Tensor<f32>) -> QuantizedWeights;
    /// W0 * Q(W) as f32 — the QAT forward weights.
    fn dequantize(&self, q: &QuantizedWeights) -> Tensor<f32>;
}
```

- Encoders emit canonical `00` for zero.
- Backward evaluation applies the Straight-Through Estimator (identity pass-through): gradients flow to latent weights `W`, never to `codes`.
- Full specification in [`QUANT.md`](QUANT.md).

### 1.7 Decision Rules and Accuracy Evaluation `[DECIDED]`

Discrete classification scoring is decoupled from differentiable training objectives (`Loss`, Section 1.2). Because `Hardmax` (`argmax`) is non-differentiable and cannot supply an analytical `Loss::backward` gradient for `Trainer`, `Hardmax` MUST NOT implement `Loss`:

```rust
pub trait Decision {
    fn is_correct(&self, pred: &Tensor<f32>, target: &Tensor<f32>) -> bool;
}

pub struct Hardmax;             // multi-class argmax (e.g., MNIST)
pub struct Threshold(pub f32);  // single-output binary decision (e.g., OR, XOR)
```

*Implementation status:* `Sequential::evaluate_accuracy(&self, dataset: &[Data]) -> f32` currently hardcodes a 10-class `argmax` loop. Generalizing it to accept `&impl Decision` (`Hardmax` / `Threshold`) is tracked as `[PENDING]` in [`ROADMAP.md`](../../project/ROADMAP.md#what-is-the-current-v010-work).

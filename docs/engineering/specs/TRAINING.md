# Specification: Training Stack and Execution Contracts

**Document type:** Engineering specification (normative contract).
**Status:** Normative contract.
**Revision date:** 2026-11-02 (maintainer decision).
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
    fn step(&mut self, model: &mut Sequential, grad: &Grad);
}
```

Initial optimizer: Batch Gradient Descent (`BGD`). The signature supports drop-in replacement by `Adam` without changes to model training loops.

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

The wire format exported to hardware memory artifacts is frozen and decoupled from internal tensor representations:

| Artifact | Format | Producer | Consumer |
|---|---|---|---|
| `wt.mem` | 2-bit dense ternary (`v = b0 - b1`) | Exporter (`export_model`) | NPU weight memories |
| `w0.mem` | Q8.24 hexadecimal scalar | Exporter (`export_model`) | NPU epilogue |
| `program.mem` | 4-word layer descriptor | Exporter (`export_model`) | NPU sequencer |
| `activations.mem` | Q8.24 hexadecimal words | Exporter (`export_data`) | NPU testbench input |
| `target.mem` | Q8.24 hexadecimal words | Exporter (`export_data`) | NPU testbench assertion |

Internal refactoring of `Tensor<T>` never alters this wire serialization contract.

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

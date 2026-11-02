# Engineering Specifications

These documents define internal technical contracts. Each specification declares
whether it describes current behavior, a target architecture, or a mixture with
explicit deviations.

| Specification | Subject | Status |
|---|---|---|
| [`ARITHMETIC_Q8_24.md`](ARITHMETIC_Q8_24.md) | Fixed-point encoding, arithmetic, and parity protocol | Target contract with documented current deviations |
| [`MODEL_TRAIT.md`](MODEL_TRAIT.md) | Tensor-based module hierarchy and model container | Target architecture |
| [`QUANT.md`](QUANT.md) | Weight quantization, $W_0$ scaling, canonical emitter, and STE | Normative contract |
| [`TENSOR.md`](TENSOR.md) | Tensor storage, views, mutation, and broadcasting | Target contract with a per-operation implementation status |
| [`TRAINING.md`](TRAINING.md) | Training stack interfaces: modules, loss, optimizer, batch, and wire format | Normative contract |

Release acceptance criteria are project records under
[`../../project/V0_CONTRACT.md`](../../project/V0_CONTRACT.md), not engineering API
documentation.

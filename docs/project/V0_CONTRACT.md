# Normative Specification: TARS v0.1.0 Release Contract (V0 Scope)

**Document type:** Project release target. The capabilities in this contract are not
claims about the current public library.

This document defines the normative scope, benchmark criteria, hardware-software capabilities,
visualization requirements, and research logging protocol for TARS Release v0.1.0 (v0).

---

## 1. Primary Benchmark Goal

**The flagship milestone of TARS v0.1.0 is solving the MNIST Handwritten Digit Recognition task.**

- **Input Dimension:** 784 features ($28 \times 28$ normalized grayscale pixels).
- **Output Dimension:** 10 classes ($0$ through $9$ softmax/one-hot logits).
- **Target Accuracy (f32 reference):** $\ge 95\%$ classification accuracy on the MNIST validation set for the floating-point model in Rust simulation.
- **Target Accuracy (ternary release gate):** $> 90\%$ for the ternary-quantized model, measured on the bit-exact inference path (identical to the NPU output by parity), reported with a 95% binomial confidence interval. The 90% threshold is pre-registered by this contract before any measurement.
- **Hardware Target:** Full inference of the trained and ternary-quantized MNIST model executing on the SystemVerilog NPU with bit-exact numeric parity.

---

## 2. Core Library Architecture and Functional Capabilities

TARS v0.1.0 must be structured as an elegant, clean Rust crate (`tars`) with a public API designed for future extensibility (CNNs, Closed-Form Continuous / CfC networks, Neural ODEs).

```text
tars/
├── src/
│   ├── lib.rs                   # Public crate re-exports and prelude
│   ├── math/                    # Tensor engine (strides, views, broadcasting, matmul)
│   ├── modules/                 # Module trait, Linear, Activations, Sequential
│   ├── model/                   # Model<M> container and IR graph extraction
│   ├── optim/                   # Tensorized Backprop, BGD, Adam, Xavier/He init
│   ├── quant/                   # Quantization engine (Q8.24 fixed-point & ternary {-1, 0, +1})
│   ├── export/                  # Hardware manifest & memory artifact generators (.mem)
│   ├── observe/                 # Observer trait, telemetry event stream, metrics
│   └── view/                    # Graphviz topology & node activation visualization
```

The `model/` container is deferred until a non-sequential topology exists (Section 6.1 and [`engineering/specs/MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md)); manifest extraction is served by `Sequential` in the interim.

### 2.1 Complete Tensor Engine (`src/math/tensor.rs`)
- Full N-dimensional contiguous and strided `Tensor` implementation with `Arc` storage.
- Compile-time element types on `Tensor<T>` (`f32` in the current phase; `T3` in the v0.1.0 arc; `i8` deferred), with `DType` as the one-byte wire tag derived from the element type, matching [`engineering/specs/TENSOR.md`](../engineering/specs/TENSOR.md). Q8.24 is strictly a serialization format of the exporter and is not an element type.
- Zero-copy view transformations: `slice`, `transpose`, `reshape`, `permute`.
- Multidimensional NumPy-compliant broadcasting.
- High-performance row-major matrix multiplication (`matmul`) for float tensors, and mixed-precision matrix multiplication for ternary weight tensors.
- Axis reductions (`sum`, `mean`).
- Copy-on-Write (COW) mutation semantics via `make_contiguous_mut`.

### 2.2 Quantization & Ternary Engine (`src/quant/`)
- **Q8.24 Fixed-Point Engine:** Quantization of continuous floats into signed 32-bit Q8.24 integers.
- **Ternary Quantization Engine ($\{-1, 0, +1\}$):** Weight quantization mapping parameters to ternary states $\{-W_0, 0, +W_0\}$ with scaling factor $W_0$.
- **Ternary Compute Operations:** Tensor-level mixed-precision dot products reducing continuous/fixed-point activations against ternary weights via conditional add/sub/zero accumulation.
- **Training Modes:** Training operates on latent `F32` master weights in all modes. The ternary training mode is quantization-aware: the forward pass evaluates against the quantized weight representation and the backward pass applies the straight-through estimator to the quantizer. Integer-only training is out of scope for v0.1.0.
- **Quantization-Aware Export:** Export pipeline emitting the layer manifest, dense ternary weight memories (`wt.mem`), per-layer scale parameters (`w0.mem`), and the layer program (`program.mem`), per LOG-007, Section 4.2, Step 1. The legacy Q8.24 artifacts (`weights.mem`, `bias.mem`) remain the development-substrate export.

### 2.3 Respective NPU Hardware (`npu/`)
- Multi-cycle **Layer Engine FSM** in SystemVerilog capable of processing $784 \to H \to 10$ layer shapes.
- **Ternary-only release engine:** 2-bit dense ternary weights with conditional add/sub/zero accumulation, a Q8.24 activation substrate, and the per-layer scale $W_0$ applied in the epilogue before the bias. The dense Q8.24 weight mode is not implemented in the release engine; the existing single-layer Q8.24 prototype remains as the development substrate and as the synthesis baseline for resource and energy comparisons (maintainer decision, October 2026).
- Layer-descriptor modes are reserved: `mode = 0` (dense Q8.24, unused), `mode = 1` (dense ternary, release), `mode = 2` (zero-skipping, v0.2.0), `mode = 3` (int8, future phase).
- 64-bit saturating accumulator to prevent overflow during 784-element dot products.
- Self-checking simulation testbench (`tb.sv`) verifying NPU execution against Rust golden vectors with $0$ bit errors.

### 2.4 Tensorized Analytical Backpropagation
- Pure tensor-based **analytical** gradient computation (`backward()`), eliminating all `Vec<Vec<f32>>` allocations. Each `Module` implements its own `backward` under the composite trait contract; no operation-graph autodiff engine is included in v0.1.0 (deferred to v1.0+, Section 5).
- Quantizer gradients follow the straight-through estimator; the numerical gradient check (`num_grad`) applies to the `F32` graph with quantizers disabled.
- Dual-mode verification: `num_grad` (finite differences) executes in regression test suites to guarantee analytical gradient correctness on the `F32` path.

---

## 3. Visualization and Observability Specifications

The visualization subsystem (`src/view/`) must enhance the topology renderer:

1. **Topology Graph Rendering:** Render neurons organized in layer columns (Input 784 $\to$ Hidden $H$ $\to$ Output 10, with configurable subsampling for display).
2. **Dynamic Node Activation Color Intensity:**
   - Each neuron vertex in the graph must visually encode its activation level $a \in [0, 1]$ (or normalized range).
   - The fill color hue is chosen per-network (e.g., deep cobalt blue or emerald green), while the **color intensity / saturation / alpha** scales dynamically with $a$:
     $$\text{Color}(a) = \text{RGBA}(\text{R}_{\text{base}}, \text{G}_{\text{base}}, \text{B}_{\text{base}}, \alpha = a)$$
3. **Edge Weight Rendering:** Edge stroke width and color polarity scale with $w_{ij}$ magnitude and sign.

---

## 4. Academic Research Logging Protocol (Paper Prerequisites)

To support the subsequent publication of an academic research paper on the TARS architecture and hardware-software co-design, all engineering discoveries, design trade-offs, and empirical results must be formally recorded.

### 4.1 Research Artifact Directory (`docs/project/research/`)

Every non-trivial engineering decision or discovery must produce a numbered log in
`docs/project/research/`:

```text
docs/project/research/
├── README.md                      # Index of research logs and paper outline
├── LOG-001-q8_24_truncation_error.md
├── LOG-002-ternary_quantization_loss.md
├── LOG-003-npu_accumulator_overflow.md
└── LOG-004-mnist_accuracy_vs_bitwidth.md
```

### 4.2 Research Log Requirements
Each log entry must follow a standardized academic structure:
1. **Title and Context:** Problem statement or hypothesis.
2. **Theoretical Formulation:** Mathematical equations or hardware state transition models.
3. **Empirical Benchmarks:** Numerical tables, accuracy graphs, or RTL cycle counts comparing alternatives.
4. **Conclusion & Architectural Implication:** Final decision and impact on TARS v0.1.0.

---

## 5. Future-Proofing Invariants (Preparing for v1.0+)

To ensure v0.1.0 abstractions seamlessly scale to CNNs, Closed-Form Continuous (CfC) networks, and Neural ODEs:

1. **Graph Agnosticism:** The `Module` trait must enforce `forward(&self, input: &Tensor) -> Tensor` without assuming 1D or 2D vector shapes.
2. **Tensor Dimensionality:** `Tensor` must support arbitrary rank $N \ge 1$ (e.g., $B \times C \times H \times W$ for CNNs, $B \times T \times D$ for sequence/CfC networks).
3. **Intermediate Representation (IR):** The `HardwareManifest` exported by `Model` must represent layers as a directed acyclic graph (DAG) of tensor operations, not hard-coded linear chains.

---

## 6. Release Boundaries and Terminology

### 6.1 Version Boundaries

| Version | Scope |
|---|---|
| **v0.1.0** | MNIST with the f32 reference model ($\ge 95\%$) and the ternary release gate ($> 90\%$ on the bit-exact path); ternary-only NPU engine with bit-exact parity; exporter formats: Q8.24 serialization and dense ternary (`mode = 1`); visualization (Section 3) and research logging (Section 4). |
| **v0.2.0** | `mode = 2` zero-skipping ternary inference ([`research/LOG-006`](research/LOG-006-ternary_zero_skipping_encoding.md)), admitted under the compatibility rules of LOG-007, Section 2.4. The int8 (`I8`/`mode = 3`) capability may enter this phase, with embedded and microcontroller inference as its prospective consumer. |
| **v0.3.0** | GPU training via cubecl (Vulkan) with custom ternary kernels, per [`engineering/FUTURE_TARGETS.md`](../engineering/FUTURE_TARGETS.md). The device dimension remains compile-time and feature-gated; the golden reference stays CPU-deterministic. |
| **v1.0+** | Architectural expansion era (Section 5): CNNs, CfC networks, Neural ODEs, DAG topologies, arbitrary-rank tensors, and an operation-level autodiff engine. |

The optimizer of record for v0.1.0 is BGD; Adam is admissible but is not required by any gate.

### 6.2 Terminology

| Term | Meaning |
|---|---|
| Training mode | `F32` training or quantization-aware ternary training; both operate on latent `F32` master weights |
| Element type | The compile-time `T` of `Tensor<T>`: `f32` in the current phase, `T3` in the v0.1.0 arc, `i8` deferred; `DType` is the one-byte wire tag derived from `T` |
| Export format | Q8.24 words or dense ternary words written under `npu/data/` from the layer manifest |
| NPU mode | Layer-descriptor field: 0 (dense Q8.24, reserved), 1 (dense ternary, release), 2 (zero-skipping, v0.2.0), 3 (int8, future) |

An element type without an implementation cannot be constructed: absence is enforced
at compile time rather than diagnosed at runtime. The `i8` wire tag is retained for
artifact stability, under the same rule as the reserved descriptor `mode = 3`.

### 6.3 Research Claim Discipline

Published results must separate measured values, tool-derived estimates, and cited constants. Accuracy claims carry a 95% binomial confidence interval over the MNIST validation set and a paired comparison (McNemar) between the f32 and ternary classifiers over identical inputs. Energy figures are estimates: exact operation counts and cycle counts from simulation, synthesis reports under a declared flow, and per-operation energy classes cited from the literature. Cycle-count parity between dense modes must not be presented as a latency advantage of ternary inference; the latency claim belongs to zero-skipping in v0.2.0.

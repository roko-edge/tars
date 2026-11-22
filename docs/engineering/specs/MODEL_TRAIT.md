# Specification: Model and Module Interface Hierarchy

**Status:** Target architecture. The current runtime still uses `Sequential` with
`Vec<AnyModule>` and does not provide the `Model<M>` container specified here. The
`Module` trait, the `Sequential` composite, and the parameter surfaces are the
v0.1.0 scope; `Model<M>` is deferred until a non-sequential topology exists
(maintainer decision, October 2026), with parameter tracking and manifest extraction
served by `Sequential` in the interim. Fluent construction of `Sequential`
is part of the v0.1.0 scope (Section 2.2). The `backward` operation is a pending
contract addition under the same composite design: recursive like `forward`,
returning parameter gradients aligned with `parameters()` and applying the
straight-through estimator to quantizers. Its data dependency is fixed by the
maintainer position of October 2026: the composite records, during the forward
evaluation, the intermediates that the backward clauses require, and `backward`
consumes the recorded trace; a second evaluation of the forward chain inside the
backward pass is not admitted.

**Implementation [CONFLICT], 2026-11-21:** `Sequential::forward_cache` stores a
flat trace of `Tensor<f32>` values. The free `backward` function consumes that
trace in reverse module order. The trait currently exposes `forward` alone.
The [delivery plan](../../project/V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition)
tracks the difference between the current pass and the target composition.

**Construction [CONFLICT]:** At `4961a9e`, a zero output width can bypass
the initializer's dimension check because the check runs once per output
row. The inspected working tree adds runtime checks for zero widths in
`Sequential::new` and `Sequential::linear` before layer allocation.
Checked allocation sizes and encapsulation of the topology and parameter
fields remain pending. [Sequential constructor](../../../src/modules/sequential.rs) ·
[Initializer](../../../src/math/rng_gen.rs)

The [training API proposal](../../project/V0_DELIVERY_PLAN.md#351-how-should-the-public-training-path-work-proposed)
tracks validation and initialization for this constructor.

This document specifies the structural interface contract for neural network
layers, composite blocks, and execution containers in TARS.

---

## 1. Motivation and Architectural Scope

The initial TARS runtime coupled model topology, layer storage, and training state
inside a monolithic `Sequential` struct storing `Vec<AnyModule>`. This structure exhibits
several limitations:

1. **Topology Rigidity:** Cannot represent branching graphs, skip-connections (e.g., ResNet),
   or recurrent/continuous-time architectures (e.g., Liquid/CfC networks).
2. **Type Erasure Overhead:** Relies on manual enum dispatch (`AnyModule`) rather than
   trait-based polymorphism or statically typed pipelines.
3. **Hardware Coupling:** Prevents clean separation between internal Rust execution
   and external hardware graph compilation.

This specification decouples these concerns into three abstractions: `Module`,
`Sequential` (as a composite `Module`), and `Model<M>`.

---

## 2. Core Abstractions

### 2.1 The `Module` Trait Interface

Every differentiable layer, activation, or composite block must implement the `Module` trait interface.

```rust
pub trait Module: std::fmt::Debug {
    fn forward(&self, input: &Tensor<f32>) -> Tensor<f32>;
    fn parameters(&self) -> Vec<&Tensor<f32>>;
}
```

The training element type is `f32` in all modes (latent master weights), so the trait
stays concrete over the element type in v0.1.0. A device dimension, if admitted later,
is a mechanical generalization under [`FUTURE_TARGETS.md`](../FUTURE_TARGETS.md),
Section 4.

#### Contract Expectations
- `forward`: Accepts an input tensor (contiguous or strided; the element type is verified statically) and evaluates the transformation, returning the resulting tensor without mutating internal layer state. The intermediates required by the backward pass are recorded by the composite evaluation (Section 2.2), never through mutation of the submodule.
- `parameters`: Collects immutable references to all float parameter tensors owned by the module in execution order.
- **Mutable parameter access [PENDING]:** Optimizer access uses the same order and preserves the validated tensor shapes. The public signature for shape-preserving mutation or checked replacement is pending.

---

### 2.2 `Sequential` as a Composite Module

`Sequential` is re-architected as a composite `Module` that owns an ordered collection of submodules (`Vec<Box<dyn Module>>`).

#### Structural Requirements
- **Encapsulation:** Holds a private, nonempty sequence of modules. Parameter updates preserve the validated weight and bias shapes.
- **Forward Composition:** Evaluating `forward` on `Sequential` recursively pipes the output tensor of module $k$ as the input tensor of module $k+1$.
- **Parameter Aggregation:** Evaluating `parameters` collects parameters from the inner submodules in execution order. Optimizer access uses the same order.
- **Intermediate Recording:** The composite exposes an evaluation that produces, beside the output tensor, the trace of the intermediates required by backward: the input consumed by each submodule in evaluation order, together with the composite output. The trace is owned by the caller for the duration of the training step and is the sole data source of the backward composition; `backward` consumes the trace and does not re-evaluate the forward chain. The plain `forward` of the trait remains a pure evaluation with no hidden state, and recording never mutates module state (Section 2.1). In the v0.1.0 scope the trace covers the backward requirements of the shipped modules: `Linear` consumes its submodule input, `ReLU` its pre-activation input, and `Sigmoid` its activation output, held as the input of the following submodule or as the composite output. Each `backward` clause drafted under this contract names the trace entries it consumes.
- **Fluent Construction [DECIDED]:** `Sequential::new(input_size, first_output_size)` creates an executable model with its first affine layer and returns `Self`. `linear(output_size)`, `relu()`, and `sigmoid()` also return `Self`. `linear` infers its input width from the current output width. Activations preserve that width. `ReLU` and `Sigmoid` are distinct structs implementing `Module`.
- **Construction Preconditions:** All feature widths must be positive and the weight and bias allocation sizes must be representable. `new` and `linear` check these preconditions before allocating a layer and panic with a diagnostic on violation. Entry points that read widths from configuration or files validate them before invoking the direct construction methods. Affine initialization advances the same seeded stream across layers. Any custom initialization seed is selected before the first layer is created.

---

### 2.3 The `Model<M>` Execution Container

`Model<M>` encapsulates a root `Module` `M`, serving as the top-level container for training execution, serialization, and hardware compilation.

#### Responsibilities
- **Root Ownership:** Holds the top-level `Module` instance representing the entire network topology.
- **Parameter Management:** Delegates parameter access to the root module using the parameter contract in §2.1.
- **Hardware IR Compilation:** Exposes a compilation interface (`compile_to_manifest(&self) -> HardwareManifest`) that traverses the module topology and parameters to construct the flat layer descriptors consumed by the NPU.

---

## 3. Extensibility Design

By restricting `Module` to an abstract trait interface, future architectural expansions
require no changes to the training container or optimizer:

- **Convolutional Layers (`Conv2d`):** Implements `Module`, exposing kernel weights and bias parameters.
- **Closed-Form Continuous Cells (`CfcCell`):** Implements `Module`, handling state transition and input concatenation.
- **Branching Networks (`DagModel`):** Implements `Module`, executing a directed acyclic graph of submodules.
- **Static Composition:** A type-level composition, in which the layer sequence is a static structure and the trait-object hop of `Vec<Box<dyn Module>>` disappears, is a future direction consistent with the compile-time-first stance of the project; it is outside the v0.1.0 scope.

# Architecture

**Audience:** TARS maintainers and engineering reviewers.

**Document type:** Current implementation record. Planned replacements belong in
specifications or proposals and are identified explicitly.

The repository contains one Rust package and one SystemVerilog prototype. They share
the Q8.24 numeric representation and the `npu/data/` artifact paths, but they do not yet
share a model topology format or an automated parity check.

## Repository structure

```text
Cargo.toml
src/
  lib.rs                  Public module exports and re-exports
  train.rs                Shared Train configuration
  experiments.rs          Public experiment-module declarations
  experiments/
    experiment_type.rs    OR/XOR identifiers
    or.rs                 OR configuration factory
    xor.rs                XOR configuration factory
  data.rs                 Training samples
  math.rs                 Scalar activations and random initialization
  math/tensor.rs          Tensor core: storage dispatch, views, kernels, products
  modules.rs              Module trait and concrete-module enum
  modules/
    linear.rs             Dense layer
    activations.rs        ReLU and sigmoid modules
    sequential.rs         Ordered module composition
  optim.rs                Optimizer module exports
  optim/
    cost.rs               Mean squared error
    grad.rs               Finite-difference gradients
    bgd.rs                Batch gradient descent
  view.rs                 Visualization module export
  view/graph.rs           petgraph and Graphviz integration
  bin/
    main.rs               Training executable with source-selected experiment
    visualization.rs      Hard-coded graph visualization executable
npu/
  src/npu_pkg.sv          Shared package: activation/state types and Q8.24 constants
  src/t3_dec.sv           2-bit ternary weight decode ($v = b_0 - b_1$)
  src/pe.sv               Ternary conditional add/subtract processing element
  src/epilogue.sv         $W_0$ scale, saturations, bias, and activation
  src/layer_engine.sv     Parameterized single-layer engine controller (`IN`, `OUT`)
  tests/tb.sv             Simulation testbench
  data/                    Memory files consumed by the testbench
  files.f                 Single source RTL file list (Makefile and LSP)
  Makefile                Simulation, test, waveform, and lint targets
```

## Rust model

`Module` defines the inference interface:

```rust
pub trait Module {
    fn forward(&self, input: &[f32]) -> Vec<f32>;
}
```

`AnyModule` provides runtime dispatch over the implemented module types:

```rust
pub enum AnyModule {
    Linear(Linear),
    Activation(Activation),
}
```

`Sequential` stores an ordered `Vec<AnyModule>`. Its builder methods update the
expected feature count when a linear layer is appended.

```rust
impl Sequential {
    pub fn new(input_size: usize) -> Self;
    pub fn linear(self, output_size: usize) -> Self;
    pub fn relu(self) -> Self;
    pub fn sigmoid(self) -> Self;
}
```

`Linear` stores weights as `Vec<Vec<f32>>`, indexed by output feature and then input
feature. Biases are stored as `Vec<f32>`. `Sequential::linear` constructs layers
through the random constructor, which draws weights from the unseeded global
generator and initializes biases to zero; the zeros constructor is not used by the
built-in experiments.

```rust
impl Linear {
    pub fn new(
        in_sz: usize,
        out_sz: usize,
        weights: Vec<Vec<f32>>,
        bias: Vec<f32>,
    ) -> Self;
    pub fn zeros(in_sz: usize, out_sz: usize) -> Self;
    pub fn random(in_sz: usize, out_sz: usize) -> Self;
}
```

The remaining training interfaces are:

```rust
impl Data {
    pub fn new(input: &[f32], target: &[f32]) -> Self;
}

pub fn cost(model: &Sequential, data: &[Data]) -> f32;
pub fn num_grad(model: &Sequential, data: &[Data]) -> Grad;

impl BGD {
    pub fn new(lr: f32) -> Self;
    pub fn step(&self, model: &mut Sequential, grad: &Grad);
}
```

`cost` computes the mean squared error across output elements and samples.
`num_grad` computes centered finite differences with `h = 1e-3` for every weight and
bias. `BGD::step` applies the resulting gradient to each linear layer.

## Training executable

`src/bin/main.rs` performs the following operations:

1. Calls `experiments::or::train()` to construct the selected `Train` configuration.
2. Prints its `ExperimentType` identifier and initial MSE.
3. Trains with full-dataset analytical backpropagation and BGD, reporting progress.
4. Prints predictions for the configured dataset.
5. Skips the Q8.24 export: the `export_model` and `export_data` calls are commented
   out in the source.
6. Builds a `NetGraph` from the trained model, writes `src/view/artifacts/graph.dot`,
   and renders `src/view/artifacts/network.png` through Graphviz.

`Train` is defined in `src/train.rs` and re-exported as `tars::Train`. Experiment
factories provide the model, epochs, learning rate, dataset, and identifier. Selection
is fixed by the factory call in the executable source; there is no interactive
selector. The model and dataset export calls are commented out; the graph artifact
generation runs after training.

See [`../EXPERIMENTS.md`](../EXPERIMENTS.md) for the configuration contract,
built-in experiments, reporting behavior, and manual extension workflow.

## Visualization

The visualization code is part of the `tars` package and is exported through
`tars::view`. `NetGraph` wraps `petgraph::graph::DiGraph<Perceptron, f32>` and exposes
the following interface:

```rust
impl NetGraph {
    pub fn new(model: Sequential, input_data: &[f32]) -> Self;
    pub fn network_to_graph(&mut self) -> Result<(), Box<dyn std::error::Error>>;
    pub fn add_node(&mut self, p: Perceptron);
    pub fn add_edge(&mut self, from: usize, to: usize, w: f32);
    pub fn plot(&self);
}
```

Serialization and rendering are reached through `save_dot` and `compile_dot`, provided
by the private `graph::dot` module.

Node indices are retained in insertion order. `plot` invokes the external `dot`
command with fixed input and output paths. The visualization executable constructs
its own hard-coded `Sequential` model and is not connected to the model trained by
the main executable.

## SystemVerilog Hardware Subsystem

`npu/src/layer_engine.sv` defines the `layer_engine` module (successor of the
former `npu` top level), parameterized by vector lengths `IN` and `OUT`:

```systemverilog
module layer_engine #(
    parameter int IN  = 2,
    parameter int OUT = 2
) (
    input  logic                     clk,
    input  logic                     rst,
    input  logic                     start,
    input  npu_pkg::activation_t     act_fn,
    input  logic [1:0]               w[IN*OUT],
    input  logic signed [31:0]       a[IN],
    input  logic signed [31:0]       b[OUT],
    input  logic signed [31:0]       w0,
    output logic signed [31:0]       result[OUT],
    output logic                     done
);
```

The engine uses a three-state machine (`npu_pkg::state_t`: `IDLE`, `RUN`,
`WRITE`) that sequences one output neuron at a time. In `RUN` it walks the
input vector, feeding each activation and its 2-bit weight code to `t3_dec`
(`src/t3_dec.sv`), which decodes $v = b_0 - b_1$, and to the Processing
Element (`pe.sv`), which conditionally adds or subtracts into a signed 64-bit
accumulator (negation evaluated at 33 bits). Weight addressing uses the linear
stream pointer `w_ptr` (rule R1 of LOG-007, Section 2.4), not the product
form $j \cdot IN + i$. In `WRITE` the engine commits `result[j]` from the
`epilogue` (`src/epilogue.sv`), which evaluates the 96-bit $W_0$ product, the
arithmetic shift by 24, two saturations, the bias, and the activation, per
[`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md), Section 3.3.
Shared types and constants live in `src/npu_pkg.sv`; the RTL file order is
declared once in `npu/files.f`.

`npu/tests/tb.sv` instantiates the DUT (`IN = 2`, `OUT = 1`), loads
`data/bias.mem`, `data/weights.mem`, and `data/activations.mem` into
testbench-level arrays that drive the DUT ports, sets `w0` to $1.0$ in Q8.24,
executes four sequential evaluations with reset between them, logs trace
outputs via `$dumpfile` / `$dumpvars`, and displays formatted Q8.24 results
without automated assertions. The checked-in `data/weights.mem` holds legacy
dense Q8.24 words while the engine consumes 2-bit ternary codes; the ternary
weight export arrives with implementation Step 1.

## Integration Boundary

The Rust export module (`src/export.rs`) converts floating-point weights, biases, inputs, and targets into Q8.24 fixed-point hexadecimal words (`to_q8_24` with truncation):

| Artifact File | Format | Source Generator | Consumer |
|---|---|---|---|
| `npu/data/weights.mem` | Q8.24 Hex (legacy dense) | `export_model()` | `$readmemh` into the testbench weight array (2-bit codes expected by the engine; mismatch until Step 1) |
| `npu/data/bias.mem` | Q8.24 Hex | `export_model()` | `$readmemh` into the testbench bias array |
| `npu/data/activations.mem` | Q8.24 Hex | `export_data()` | `$readmemh` into the testbench data array |
| `npu/data/target.mem` | Q8.24 Hex | `export_data()` | Not loaded by the current testbench |

The export calls in the training binary are commented out, so the checked-in memory
files consumed by `make sim` remain from an earlier export and are not overwritten by
a training run. The shared encoding and paths do not provide topology or activation
information, `target.mem` holds dataset labels rather than golden inference outputs,
and the testbench performs no bit-exact assertions. Topology compilation and
golden-vector parity remain project targets recorded in
[`../project/ROADMAP.md`](../project/ROADMAP.md).

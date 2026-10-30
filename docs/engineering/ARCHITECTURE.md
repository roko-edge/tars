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
  src/layer_engine.sv     Single-layer engine controller (`IN`, `OUT`)
  src/npu.sv              Per-network wrapper: two chained `layer_engine` instances (784→64→10 defaults)
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

### What is `npu.sv`?

`npu/src/npu.sv` is the **per-network wrapper**: a thin top that instantiates
two `layer_engine` instances and wires them into one MLP. The default
parameters shape the MNIST release network ($784 \to 64 \to 10$); the
topology is configuration (elaboration parameters plus the weight, bias, and
$W_0$ ports), not code — a network with different dimensions is the same file
with different parameters. The wrapper chains the layers through the handshake:
`u_l1.start` is driven by `u_l0.done`, so layer 1 begins when layer 0 commits
its last result. Layer 0 applies `RELU`; layer 1 applies `IDENTITY`.

This wrapper is the manual-composition reference of Layer 2 in
[`ASSEMBLY_AND_TEST_MODEL.md`](ASSEMBLY_AND_TEST_MODEL.md), Section 1. It
does not contain the sequencer, program memory, or activation banks of the
programmable top; those arrive with LOG-007 Steps 3–4 and will replace the
hand-wired chain, not the engines.

### The engine: `layer_engine`

`npu/src/layer_engine.sv` defines the `layer_engine` module, parameterized by
vector lengths `IN` and `OUT`:

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

### Handshake contract (`start` / `done`)

The engine and the wrapper exchange `start` (request pulse) and `done`
(completion pulse); the wrapper additionally arbitrates requests while data is
in flight. The full contract — the seven clauses, the responsibility
boundaries, and the test gates — is owned by
[`PROTOCOL.md`](PROTOCOL.md). Summary: both signals are one-cycle events,
`result[]` holds from `done` until the next accepted request, and a held
`start` (level) is a contract violation that the engine tolerates but does not
protect against.

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

## Glossary

Canonical names for every hardware concept. Use these terms in code, docs, and
discussion; the aliases column lists retired or informal names that must not
re-enter the documentation.

| Term | Definition | Aliases (retired) |
|---|---|---|
| Primitive (Layer 0) | A frozen building block with a golden-vector contract: `npu_pkg`, `t3_dec`, `pe`, `epilogue` | — |
| Engine (Layer 1) | A computation unit built from primitives behind the `start`/`done` contract; today only `layer_engine` (dense ternary, `mode = 1`) | — |
| Wrapper (Layer 2, current) | `npu.sv`: the per-network top that instantiates and chains engines for one specific network; topology is parameters and ports, not code | "the NPU", "top-level NPU" (ambiguous) |
| Assembled NPU | The `npu.sv` wrapper for a specific network (defaults $784 \to 64 \to 10$); a synonym of *wrapper* used when contrasting with the programmable top | — |
| Programmable top (Layer 2, target) | The sequencer + `program.mem` + activation banks of LOG-007 Steps 3–4; will replace the wrapper's hand-wired chain | "shell" |
| `layer_engine` | The single-layer engine: `IDLE`/`RUN`/`WRITE` FSM over one `pe` and one `epilogue` | "npu module" (pre-October 2026 name) |
| Epilogue | The per-neuron post-processing block: 96-bit $W_0$ product, arithmetic shift by 24, two saturations, bias, activation | "post-op", "output stage" |
| $W_0$ | Per-layer Q8.24 scale applied by the epilogue before the bias; one scalar per layer in v0.1.0 | "the scale" (ambiguous with quantization step $\Delta$) |
| Ternary decode | The map $v = b_0 - b_1$ over a 2-bit weight code; all four codes valid (`2'b11` is the redundant zero) | "active flag + sign" (retired encoding) |
| Dense ternary (`mode = 1`) | The release weight stream: one 2-bit code per weight, no skip tokens | — |
| Sparse tokens (`mode = 2`) | The five-bit `(skip, sign)` stream of LOG-006; v0.2.0, decoded by a future `sparse_frontend` | "zero-skip mode" (fine), "reserved dense code" (retired) |
| Handshake | The `start`/`done` pair; the full contract (seven clauses, boundaries, gates) is owned by [`PROTOCOL.md`](PROTOCOL.md) | "busy" (the pre-October 2026 two-state FSM used `BUSY`; retired) |
| Request event | A rising `start` edge: one inference request; consumers MUST emit one-cycle pulses (PROTOCOL.md, clause 1) | — |
| Done pulse | The one-cycle completion pulse aligned with the last `result[]` write; `result[]` is valid on that cycle and holds until the next accepted request (PROTOCOL.md, clauses 2–3) | "done level" (retired: caused re-trigger loops) |
| Tier ① / ② / ③ | The three verification tiers of [`ASSEMBLY_AND_TEST_MODEL.md`](ASSEMBLY_AND_TEST_MODEL.md): core, golden parity, system shell | — |
| Q8.24 | The signed 32-bit fixed-point substrate: 8 integer bits, 24 fractional bits | — |

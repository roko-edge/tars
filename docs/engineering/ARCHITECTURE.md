# Architecture

**Audience:** TARS maintainers and engineering reviewers.

**Document type:** Current implementation record. Planned replacements belong in
specifications or proposals and are identified explicitly.

TARS combines a typed Rust tensor and training stack with a dense ternary
SystemVerilog accelerator. Explicit exporter calls connect them through
hexadecimal memory files in `npu/data/`. The assembled parity bench can read
these files. Exact comparison with an independent integer reference is a
release gate. [Artifact reference](MEM_FORMAT.md)

## Repository structure

| Area | Responsibility |
| --- | --- |
| `src/tensor/` | Typed `Tensor<T>`, geometry and views, float kernels, and experimental packed T3 storage |
| `src/modules/` | Dense layers, activations, and the sequential model and forward cache |
| `src/optim/` | Losses, analytical parameter gradients, BGD, and Adam |
| `src/data.rs` | Sample, `DatasetSplit`, batch assembly, and seeded shuffling |
| `src/train.rs` and `src/report.rs` | Reusable float/QAT training loop (`TrainSummary`) and structured evaluation reporting (`ModelReport`) |
| `src/quant/` and `src/export.rs` | Ternary scalar quantization and conversion to the Rust–RTL memory files |
| `examples/` | Standalone runnable examples (`mnist`, `or`, `xor`) consuming the public crate API |
| `npu/src/` and `npu/tests/` | Ternary primitives, layer engine, assembled wrapper, and simulation benches |

## Rust model

`Module::forward` maps `&Tensor<f32>` to `Tensor<f32>`. `AnyModule`
dispatches between `Linear` and `Activation`. `Sequential` stores a
`Vec<AnyModule>`. Its constructor creates the first affine layer and the
consuming builder appends layers while tracking the current feature width.
`Linear` stores `[out, in]` weights and `[out]` biases as `Tensor<f32>`.
Each random layer initializes from a fixed seed and starts with zero biases.

`Sequential::forward_cache` records its input and each module output. The
free `backward` function walks those entries in reverse and returns gradients
indexed by module. BGD and Adam update the model parameters. Full-model finite
differences are a verification task. [Current public API](../API.md) ·
[Target training interfaces](specs/TRAINING.md)

## Runnable examples

Standalone examples under `examples/` exercise the public library API without
coupling datasets to `src/lib.rs`. [`examples/mnist.rs`](../../examples/mnist.rs)
trains `784 → 64 → 10` on `train_data` with `Adam` and `.with_qat(true)`,
evaluates latent `f32` and scaled ternary projections on `test_data` via
`ModelReport`, and invokes `export_quantized_model` and `export_inference` on
the first 100 test samples. `examples/or.rs` and `examples/xor.rs` exercise
`Adam`/`MSE` on the two-input logic gates.
[Experiment behavior](../EXPERIMENTS.md#selection-and-execution)

## Visualization

Graph helpers under `src/view/` remain outside the public module tree, and no
visualization binary is currently built in `src/bin/`.
[Visualization status](../VISUALIZATION.md)

## SystemVerilog Hardware Subsystem

### What is `npu.sv`?

`npu/src/npu.sv` is the **per-network wrapper**: a thin top that instantiates
two `layer_engine` instances and wires them into one MLP. The default
parameters shape the MNIST release network ($784 \to 64 \to 10$); the
topology is configuration (elaboration parameters plus the weight, bias, and
$W_0$ ports), not code — a network with different dimensions is the same file
with different parameters. The wrapper chains the layers through the handshake:
`u_l1.start` is driven by `u_l0.done`, so layer 1 begins when layer 0 commits
its last result. For MNIST parity the bench supplies `RELU` for layer 0 and
`IDENTITY` for layer 1.

This wrapper is the manual-composition reference of Layer 2 in
[`ASSEMBLY_AND_TEST_MODEL.md`](ASSEMBLY_AND_TEST_MODEL.md), Section 1. It
does not contain the sequencer, program memory, or activation banks of the
programmable top; those arrive in v0.1.1 (LOG-008) and will replace the
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

The legacy smoke bench drives a single layer with small input files. The
Tier 1 bench exercises the assembled wrapper with hand-calculated data.
The Tier 2 bench instantiates the release-shaped wrapper and reads exported
weights, scales, biases, and inference data. [Simulation commands](../../npu/README.md#tests)

## Integration Boundary

`export_quantized_model` writes dense ternary weights and Q8.24 scales and biases.
`export_inference` writes Q8.24 inputs and float-derived predictions.
`npu.sv` receives parameters through ports, and elaboration configures the
layer dimensions. The current Tier 2 comparator allows one LSB. The release
requires expected words from the independent Rust integer reference and an
exact comparison. [Artifact encoding and consumers](MEM_FORMAT.md) ·
[Parity delivery plan](../project/V0_DELIVERY_PLAN.md#24-how-do-export-and-rtl-establish-exact-parity-pending)

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
| Programmable top (Layer 2, target) | The sequencer + `program.mem` + activation banks (deferred to v0.1.1, LOG-008); will replace the wrapper's hand-wired chain | "shell" |
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

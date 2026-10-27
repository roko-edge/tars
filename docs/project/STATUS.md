# Implementation Status

**Document type:** Project status record. User instructions are maintained in
[`docs/`](../README.md).

This document reflects the current state of the repository as of October 2026.

---

## Rust Package

| Component | Status | Implementation Details |
|---|---|---|
| Sequential Model | Implemented | `Sequential` stores `Vec<AnyModule>`; the tensor-based replacement is specified under `docs/engineering/` |
| Dense Layer | Implemented | `Linear` uses `Vec<Vec<f32>>` weights and `Vec<f32>` biases |
| Activations | Implemented | ReLU and Sigmoid modules |
| Tensor Core | In Progress | `Tensor` in `src/math/tensor.rs` implements container-level `Storage` dispatch (`F32`, `I8`, `T3`) and the `DType` tag. Active over `F32`: geometry accessors, `get`/`set` with COW (`Arc::make_mut`), zero-copy `transpose` and its in-place variant, parameterized `zeros`, `make_contiguous`, the element-wise kernels (`add`, `sub`, tensor `mul`/`div`, `mul_scalar`, `div_scalar`, `map`, with in-place variants), the full reduction `sum`, and the products `matmul`, `matvec`, `outer`. Open clauses of `docs/engineering/specs/TENSOR.md`: construction capacity validation, one-byte copyable type tag, rank-zero-safe logical iteration, non-empty precondition diagnostics, zero-divisor rejection in `div_scalar`, stateful `map` callables, the bulk access surface, and the `I8`/`T3` access paths. The typed-element revision of the specification (compile-time `Tensor<T>`, October 2026) is pending the source migration |
| Loss Function | Implemented | Mean Squared Error (MSE) across outputs and dataset samples |
| Analytical Backpropagation | Implemented | Full analytical backprop in `src/optim/grad.rs` (`backward()`) |
| Finite-Difference Gradient | Implemented | Centered finite differences in `src/optim/grad.rs` (`num_grad()`), retained for verification |
| Optimizer | Implemented | Batch Gradient Descent (BGD) applying computed gradients |
| Experiments | Implemented | Built-in OR and XOR experiment factories |
| Fixed-Point Export | Implemented | `export_model()` and `export_data()` convert values to Q8.24 fixed-point hex files under `npu/data/` (`weights.mem`, `bias.mem`, `activations.mem`, `target.mem`). The training binary's export calls are commented out, so the checked-in files are not overwritten (limitation 6) |
| Automated Tests | Unstable | `cargo test` compiles; four tests exist and only one exercises `Tensor`. The distribution tests sample the unseeded global generator, and the ternary-distribution test asserts a variance bound at the true value of its distribution, so the suite fails intermittently. The tensor contracts, the module hierarchy, and the optimizer have no coverage |

The Rust implementation uses `f32`, `std`, and dynamically allocated vectors. Analytical backpropagation is present, but operates on nested vectors rather than contiguous tensors.

---

## Visualization

The package exports a `view` module backed by `petgraph`. `NetGraph::network_to_graph` extracts graph topology directly from a `Sequential` model and input slice. The visualization binary saves Graphviz DOT files (`graph_to_dot` / `save_dot`) and generates PNG diagrams via `plot()`. `plotters` and `petgraph` are required package dependencies.

---

## SystemVerilog NPU Prototype

| Component | Status | Implementation Details |
|---|---|---|
| Processing Element | Implemented | `pe.sv` in `npu/src/` multiplies signed 32-bit Q8.24 values and accumulates reduced products in 64 bits |
| NPU Top-Level | Implemented | `npu.sv` in `npu/src/` implements the single-layer affine engine: `IDLE`/`BUSY` FSM with `start`/`done`, sequential computation of `OUT` outputs from `IN` activations, per-output bias, and final signed 64-to-32-bit saturation |
| Layer Dimensions | Parameterized | Parameters `IN` and `OUT` (testbench instantiates `IN=2`, `OUT=1`) |
| Internal Storage | Implemented | Internal `b[OUT]`, `w[IN*OUT]`, and `a[IN]` arrays loaded via `$readmemh` |
| Control Logic | Implemented | `clk`, `rst`, `start`, and `done` handshaking signals |
| Testbench | Partial | `tests/tb.sv` instantiates DUT, enables `$dumpfile`/`$dumpvars`, but lacks automated pass/fail assertions and non-zero exit codes |

---

## Command Status

| Command | Status | Notes |
|---|---|---|
| `cargo build` | Available | Compiles the Rust package and binaries |
| `cargo run --bin main` | Partial | Trains the source-selected OR model and prints predictions. The export calls are commented out, so the files under `npu/data/` are not overwritten. The run continues into graph artifact generation and terminates with a file error when `src/view/artifacts/` is absent |
| `cargo run --bin visualization` | Available | Requires Graphviz; generates graph DOT and PNG artifacts |
| `cargo test` | Unstable | Compiles four tests; the suite fails intermittently because the ternary-distribution test samples the unseeded generator and asserts a variance bound at the true value of its distribution. Coverage of the tensor contracts and of the training path is not yet established |
| `make sim` (from `npu/`) | Partial | Runs in the sanctioned `hardware` development shell, whose `flake.nix` provides the Python 3 interpreter required by the Verilator `--binary --timing --trace` build. Compiles `src/pe.sv`, `src/npu.sv`, and `tests/tb.sv`, prints the four legacy evaluations, and dumps waveforms to `build/dump.vcd`. Lacks pass/fail checks and a non-zero exit code on failure |
| `make test` / `make all` | Partial | Aliases `sim` |
| `make wave` | Available | Runs GTKWave against `build/dump.vcd` generated from simulation |
| `make lint` | Partial | Runs Verilator lint-only checks on `src/` and `tests/` sources; currently exits non-zero with two index-width warnings (`WIDTHTRUNC`) in `npu/src/npu.sv` under Verilator 5.052 |
| `make clean` | Available | Cleans NPU build outputs |

---

## Known Technical Limitations

1. **Testbench Self-Checking:** The NPU testbench (`tests/tb.sv`) prints result signals but does not assert equality against expected values or return a non-zero exit code on failure.
2. **Q8.24 Rounding Strategy:** Rust export (`src/export.rs`) and SystemVerilog arithmetic (`npu/src/npu.sv`, `npu/src/pe.sv`) perform truncation (`as i32` / bit slicing) rather than round-half-to-even.
3. **NPU RTL Lint:** Verilator reports index-width warnings on the `j` index expressions in `npu/src/npu.sv` (two `WIDTHTRUNC` findings under Verilator 5.052), so `make lint` exits non-zero.
4. **Vector Tensor Storage:** Linear layers and gradients currently allocate nested vectors (`Vec<Vec<f32>>`), requiring migration to contiguous `Tensor` storage. The tensor core must first satisfy the surface enumerated in `docs/engineering/specs/TENSOR.md`, Section 4.2.
5. **Gradient Verification:** The analytical and finite-difference gradients share the same scale, but the finite-difference estimator evaluates the loss difference in single precision with a step of $10^{-3}$, which places the attainable precision near the tolerance of $\epsilon \le 10^{-4}$ required by the project roadmap. Accumulation in double precision and a relative tolerance are required before the check is authoritative.
6. **Artifact Integration:** The export calls in the training binary are commented out, so the checked-in `npu/data/` files remain from an earlier export and are not overwritten by a training run. Exports omit activation functions and topology, and `target.mem` contains dataset labels rather than golden Q8.24 inference outputs.
7. **Determinism:** The PRNG and the layer initializer draw from the unseeded global generator, so training runs and the distribution tests are not reproducible; the project invariant requiring seeded generation is not yet met.
8. **RTL Activation Unit:** The activation unit implements ReLU only; selecting sigmoid routes the saturated sum unchanged (identity), so the sigmoid output layer of the built-in experiments has no hardware equivalent under the current sources.
9. **Numerical Gradient Check:** `num_grad` has no callers; the finite-difference verification required by the roadmap is not exercised by any test or binary.

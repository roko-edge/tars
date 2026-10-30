# Implementation Status

**Document type:** Project status record. User instructions are maintained in
[`docs/`](../README.md).

This document reflects the current state of the repository as of October 2026.

---

## State in Ten Lines

1. The Rust package trains small MLPs (OR/XOR) with `f32` and analytical backprop; the tensor core is mid-migration and its suite is not deterministic yet.
2. The NPU release engine is **dense ternary**: 2-bit weights, conditional add/subtract, a 64-bit accumulator, and a per-neuron $W_0$ epilogue (96-bit product, floor shift, two saturations).
3. The ternary encoding is $v = b_0 - b_1$; all four 2-bit codes are valid. Signed off by the maintainer in October 2026.
4. RTL is organised in three layers: primitives (`t3_dec`, `pe`, `epilogue`), the single-layer engine (`layer_engine`), and a per-network wrapper (`npu.sv`, 784→64→10 by default).
5. `make lint` (top `npu`) exits zero; `make test` runs the testbench and propagates its exit code.
6. The testbench prints results but asserts nothing; golden vectors and the Rust bit-exact model are the next gate (LOG-007 Step 1).
7. The checked-in `npu/data/` files are legacy dense Q8.24; the ternary weight export (`wt.mem`, `w0.mem`) does not exist yet.
8. The programmable top (sequencer, `program.mem`, ping-pong banks) is not implemented; the wrapper is the manual stand-in.
9. Release target: MNIST $784 \to 64 \to 10$ bit-exact against the Rust model, ternary accuracy $> 90\%$; tag v0.1.0.
10. Verification is three-tiered (`tb_core` / `tb_parity` / `tb_shell`); today only the core tier's testbench exists, without assertions.

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
| Fixed-Point Export | Implemented | `export_model()` and `export_data()` convert values to Q8.24 fixed-point hex files under `npu/data/` (`weights.mem`, `bias.mem`, `activations.mem`, `target.mem`). The training binary's export calls are commented out, so the checked-in files are not overwritten (limitation 5) |
| Automated Tests | Unstable | `cargo test` compiles; four tests exist and only one exercises `Tensor`. The distribution tests sample the unseeded global generator, and the ternary-distribution test asserts a variance bound at the true value of its distribution, so the suite fails intermittently. The tensor contracts, the module hierarchy, and the optimizer have no coverage |

The Rust implementation uses `f32`, `std`, and dynamically allocated vectors. Analytical backpropagation is present, but operates on nested vectors rather than contiguous tensors.

---

## Visualization

The package exports a `view` module backed by `petgraph`. `NetGraph::network_to_graph` extracts graph topology directly from a `Sequential` model and input slice. The visualization binary saves Graphviz DOT files (`graph_to_dot` / `save_dot`) and generates PNG diagrams via `plot()`. `plotters` and `petgraph` are required package dependencies.

---

## SystemVerilog NPU Prototype

| Component | Status | Implementation Details |
|---|---|---|
| Ternary Decode | Implemented | `t3_dec.sv` in `npu/src/` decodes the 2-bit dense ternary code $v = b_0 - b_1$ (`w_en = w[1] ^ w[0]`, `w_s = w[1]`); all four codes are valid (`2'b11` is the redundant zero) |
| Processing Element | Implemented | `pe.sv` in `npu/src/` conditionally adds or subtracts the signed 32-bit Q8.24 activation into a signed 64-bit accumulator; negation is evaluated at 33 bits before extension |
| Epilogue | Implemented | `epilogue.sv` in `npu/src/` computes the 96-bit $\mathrm{acc} \cdot W_0$ product, the arithmetic shift by 24 (floor), the two saturations, the bias, and the activation (`RELU`; `IDENTITY` fallback) |
| Layer Engine | Implemented | `layer_engine.sv` in `npu/src/`: `IDLE`/`RUN`/`WRITE` FSM, sequential computation of `OUT` outputs from `IN` activations, linear weight-stream pointer (rule R1), instantiating one `pe` and one `epilogue`; `done` is a one-cycle registered pulse |
| Network Wrapper | Implemented, untested | `npu.sv` in `npu/src/`: two `layer_engine` instances chained (defaults $784 \to 64 \to 10$; layer 0 `RELU`, layer 1 `IDENTITY`); `u_l1.start` driven by `u_l0.done`. Topology is parameters and ports. Compiles under `make lint` (top module); no tier exercises it against golden vectors |
| Shared Package | Implemented | `npu_pkg.sv` in `npu/src/` declares `activation_t` (`IDENTITY`, `RELU`; `SIGMOID` reserved), `state_t` (`IDLE`, `RUN`, `WRITE`), and the `Q8_24_MAX`/`Q8_24_MIN` constants |
| Layer Dimensions | Parameterized | Elaboration parameters `IN`/`OUT` per engine (testbench instantiates `IN=2`, `OUT=1`; wrapper defaults 784/64/10); the 16-bit runtime shape registers are Step 2 of LOG-007, Section 4.2 |
| Internal Storage | Refactored | Activation, weight, and bias arrays are explicit top-level ports driven by the testbench (`$readmemh` into testbench-level arrays) |
| Control Logic | Implemented | `clk`, `rst`, `start`, and `done` handshaking signals |
| Testbench | Partial | `tests/tb.sv` instantiates `layer_engine`, enables `$dumpfile`/`$dumpvars`, and drives all ports, but lacks automated pass/fail assertions and non-zero exit codes |

---

## Command Status

| Command | Status | Notes |
|---|---|---|
| `cargo build` | Available | Compiles the Rust package and binaries |
| `cargo run --bin main` | Partial | Trains the source-selected OR model and prints predictions. The export calls are commented out, so the files under `npu/data/` are not overwritten. The run continues into graph artifact generation and terminates with a file error when `src/view/artifacts/` is absent |
| `cargo run --bin visualization` | Available | Requires Graphviz; generates graph DOT and PNG artifacts |
| `cargo test` | Unstable | Compiles four tests; the suite fails intermittently because the ternary-distribution test samples the unseeded generator and asserts a variance bound at the true value of its distribution. Coverage of the tensor contracts and of the training path is not yet established |
| `make sim` (from `npu/`) | Partial | Runs in the sanctioned `hardware` development shell, whose `flake.nix` provides the Python 3 interpreter required by the Verilator `--binary --timing --assert --trace` build. Compiles the RTL list of `files.f` together with `tests/tb.sv`, prints the four legacy evaluations, and dumps waveforms to `build/dump.vcd`. The build tolerates warnings (`-Wno-fatal`); Lacks pass/fail checks in the testbench itself |
| `make test` / `make all` | Partial | Alias of `sim`; propagates the testbench exit code, so a `$fatal` mismatch fails the target once the testbench is self-checking |
| `make wave` | Available | Runs GTKWave against `build/dump.vcd` generated from simulation |
| `make lint` | Available | Runs `verilator --lint-only -Wall --top-module npu -f files.f` (top `npu` covers the wrapper, engines, and primitives); exits zero with no `WIDTHTRUNC`/`WIDTHEXPAND`/`UNDRIVEN` findings. The two `WIDTHEXPAND` findings on the `safe_log(IN)`/`safe_log(OUT)` calls were resolved with explicit `64'` casts (October 2026); verified under Verilator 5.048 and 5.052 |
| `make clean` | Available | Cleans NPU build outputs |

---

## Known Technical Limitations

1. **Testbench Self-Checking:** The NPU testbench (`tests/tb.sv`) prints result signals but does not assert equality against expected values. `make test` would propagate a non-zero exit code from a `$fatal` mismatch, but no assertion exists yet; the Rust bit-exact golden model and the ternary export artifacts are implementation Step 1.
2. **Q8.24 Rounding Strategy:** Rust export (`src/export.rs`) truncates toward zero (`as i32`), while the NPU epilogue reduces the 96-bit $W_0$ product with an arithmetic right shift (floor toward negative infinity, `npu/src/epilogue.sv`). The golden model must replicate the floor semantics for bit-exact parity; neither side uses round-half-to-even.
3. **Vector Tensor Storage:** Linear layers and gradients currently allocate nested vectors (`Vec<Vec<f32>>`), requiring migration to contiguous `Tensor` storage. The tensor core must first satisfy the surface enumerated in `docs/engineering/specs/TENSOR.md`, Section 4.2.
4. **Gradient Verification:** The analytical and finite-difference gradients share the same scale, but the finite-difference estimator evaluates the loss difference in single precision with a step of $10^{-3}$, which places the attainable precision near the tolerance of $\epsilon \le 10^{-4}$ required by the project roadmap. Accumulation in double precision and a relative tolerance are required before the check is authoritative.
5. **Artifact Integration:** The export calls in the training binary are commented out, so the checked-in `npu/data/` files remain from an earlier export and are not overwritten by a training run. Exports omit activation functions and topology, and `target.mem` contains dataset labels rather than golden Q8.24 inference outputs.
6. **Determinism:** The PRNG and the layer initializer draw from the unseeded global generator, so training runs and the distribution tests are not reproducible; the project invariant requiring seeded generation is not yet met.
7. **RTL Activation Unit:** The activation unit implements ReLU only; `SIGMOID` is declared in `npu_pkg::activation_t` as reserved and falls through to the identity path, so selecting it routes the saturated sum unchanged. The sigmoid output layer of the built-in experiments has no hardware equivalent under the current sources, and the testbench must keep `act_fn != SIGMOID` until it is implemented.
8. **Numerical Gradient Check:** `num_grad` has no callers; the finite-difference verification required by the roadmap is not exercised by any test or binary.

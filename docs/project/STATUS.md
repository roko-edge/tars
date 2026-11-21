# Implementation Status

**Document type:** Project status record. User instructions are maintained in
[`docs/`](../README.md).

This source snapshot covers `e7b7f64`, inspected on 2026-11-21.
The command table records available entry points. Release evidence is tied
to its tested revision in the [delivery plan](V0_DELIVERY_PLAN.md).

---

## State in Ten Lines

1. The Rust training path uses `Tensor<f32>` and `DataMiniBatches` seeded shuffling. Packed T3 storage is available independently. The [delivery plan](V0_DELIVERY_PLAN.md#11-what-did-the-training-and-tensor-audit-find-on-2026-11-21) records the current loss, MNIST, and quantization gaps.
2. The NPU release engine uses dense 2-bit ternary weights, conditional add/subtract, a 64-bit accumulator, and a per-neuron epilogue with one scale $W_0$ per layer.
3. The ternary encoding is $v = b_0 - b_1$ ($00=0, 01=+1, 10=-1, 11=0$). The canonical encoder emits `00` for zero. [Encoding contract](../engineering/specs/QUANT.md#2-canonical-hardware-code-serialization)
4. RTL is organised in three layers: primitives (`t3_dec`, `pe`, `epilogue`), the single-layer engine (`layer_engine`), and a per-network wrapper (`npu.sv`, 784→64→10 by default).
5. CI runs RTL lint and `test-core`. The default `make test` runs the Tier 1 assertions.
6. Tier 2 (`tb_parity`) loads memory files for the assembled wrapper. The independent Rust reference and exact word comparator remain [PENDING].
7. Dense ternary weight export (`wt.mem`) and per-layer scales (`w0.mem`) are implemented in `src/export.rs` (PR #30); checked-in legacy `npu/data/` files remain retained for `tb.sv`.
8. The manual wrapper is the v0.1.0 assembly. The programmable top is scheduled for v0.1.1. [Scope decision](research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026)
9. Release target: MNIST $784 \to 64 \to 10$ bit-exact against the Rust model, ternary accuracy $> 90\%$; tag v0.1.0.
10. The MNIST runner uses Adam and reports accuracy on the training split. The official float baseline and verified QAT path remain [PENDING].

---

## Rust Package

| Component | Status | Implementation Details |
|---|---|---|
| Sequential Model | Implemented | `Sequential` stores `Vec<AnyModule>`; implements forward caching and `.quantize()` extracting layer-wise scales $W_0$ |
| Dense Layer | Implemented | `Linear` uses contiguous `Tensor<f32>` weights and `Tensor<f32>` biases (PR #26); implements `.quantize()` returning quantized layer and scale $W_0$ (PR #30) |
| Activations | Implemented | ReLU and Sigmoid modules |
| Tensor Core | Implemented | Typed `Tensor<T>` with strided geometry and copy-on-write storage. The float backend uses `Arc<Vec<f32>>`. The experimental T3 backend packs 32 trits per `u64` |
| Loss Function | [CONFLICT] | MSE and SoftmaxCE implement `Loss`. SoftmaxCE forward clamps probabilities while its backward uses the unclamped gradient |
| Analytical Backpropagation | [CONFLICT] | The free `backward` function traverses `ForwardCache` in reverse and overwrites the selected delta with the legacy MSE derivative. The module-local composition contract remains open. |
| Finite-Difference Gradient | [PENDING] | The current source tree has no `num_grad` implementation or full-model finite-difference verification |
| Optimizer | Implemented | `BGD` and `Adam` update float parameters. The MNIST runner uses Adam while the release baseline specifies BGD |
| Experiments | Implemented | Built-in OR, XOR, and MNIST factories. The MNIST runner evaluates against training data |
| Fixed-Point & Ternary Export | Partial | `export_model` writes dense 2-bit ternary codes (`wt.mem`), Q8.24 scales (`w0.mem`), and Q8.24 biases (`bias.mem`). `export_inference` writes inputs and float-derived expected outputs under `npu/data/` |
| QAT and fixed-point reference | [PENDING] | The release QAT runner, independent Q8.24 evaluator, and STE verification are tracked in the delivery plan. |

The Rust implementation uses contiguous `Tensor<f32>` storage and matrix multiplication. Analytical backpropagation operates through `ForwardCache`. Quantization maps latent weights to ternary states scaled by layer-wise $W_0$.

---

## Visualization

**[PENDING]** The visualization executable is a stub. The graph helpers
under `src/view/` sit outside the public module tree.
[Visualization status](../VISUALIZATION.md)

---

## SystemVerilog NPU Prototype

| Component | Status | Implementation Details |
|---|---|---|
| Ternary Decode | Implemented | `t3_dec.sv` in `npu/src/` decodes the 2-bit dense ternary code $v = b_0 - b_1$ (`w_en = w[1] ^ w[0]`, `w_s = w[1]`); all four codes are valid (`2'b11` is the redundant zero) |
| Processing Element | Implemented | `pe.sv` in `npu/src/` conditionally adds or subtracts the signed 32-bit Q8.24 activation into a signed 64-bit accumulator; negation is evaluated at 33 bits before extension |
| Epilogue | Implemented | `epilogue.sv` in `npu/src/` computes the 96-bit $\mathrm{acc} \cdot W_0$ product, the arithmetic shift by 24 (floor), the two saturations, the bias, and the activation (`RELU`; `IDENTITY` fallback) |
| Layer Engine | Implemented | `layer_engine.sv` in `npu/src/`: `IDLE`/`RUN`/`WRITE` FSM, sequential computation of `OUT` outputs from `IN` activations, linear weight-stream pointer (rule R1), instantiating one `pe` and one `epilogue`; `done` is a one-cycle registered pulse |
| Network Wrapper | Implemented | `npu.sv` chains two `layer_engine` instances through `u_l0.done`. The wrapper defaults to $784 \to 64 \to 10$. Tier 1 exercises it with small inputs and fixed expectations, and Tier 2 supplies the MNIST shape. |
| Shared Package | Implemented | `npu_pkg.sv` in `npu/src/` declares `activation_t` (`IDENTITY`, `RELU`; `SIGMOID` reserved), `state_t` (`IDLE`, `RUN`, `WRITE`), and the `Q8_24_MAX`/`Q8_24_MIN` constants |
| Layer Dimensions | Parameterized | Each engine uses elaboration parameters `IN` and `OUT`. Runtime shape registers belong to the programmable top in v0.1.1. |
| Internal Storage | Refactored | Activation, weight, and bias arrays are explicit top-level ports driven by the testbench (`$readmemh` into testbench-level arrays) |
| Control Logic | Implemented | `clk`, `rst`, `start`, and `done` handshaking signals |
| Testbench (Tier 1) | Implemented | `tests/tb_core.sv` instantiates `npu`, asserts deterministic hand-calculated Q8.24 outputs, and calls `$fatal` on mismatch; runs via `make test-core` / `make test` |
| Testbench (Legacy Smoke) | Implemented | `tests/tb.sv` instantiates `layer_engine`, prints outputs, and dumps waveforms without automated assertions; runs via `make sim` |

---

## Command Status

| Command | Status | Notes |
|---|---|---|
| `cargo build` | Available | Compiles the Rust package and binaries |
| `cargo run --bin main` | Available | The runner selects the MNIST factory and trains with a forward cache and `DataMiniBatches` |
| `cargo run --bin visualization` | Stub | Completes without generating artifacts |
| `cargo test` | Available | Runs the Rust unit and integration suites. CI invokes this command. |
| `make sim` (from `npu/`) | Available | Runs legacy smoke simulation (`tests/tb.sv`, prints results, dumps waveforms to `build/dump.vcd`). Uses `-Wno-fatal` |
| `make test` / `make all` | Available | Executes Tier 1 testbench `test-core` (`tests/tb_core.sv`) and propagates its exit status |
| `make test-core` | Available | Compiles and executes `tests/tb_core.sv` in `build/core/tb_core_sim`; asserts deterministic hand-calculated Q8.24 outputs |
| `make test-parity` | Available | Runs `tb_parity.sv`. The comparator allows a one-LSB difference and loads `target.mem`, which the current exporter fills with float-derived predictions |
| `make wave` | Available | Runs GTKWave against `build/dump.vcd` generated from simulation |
| `make lint` | Available | Runs `verilator --lint-only -Wall --top-module npu -f files.f` over the wrapper, engines, and primitives |
| `make clean` | Available | Cleans NPU build outputs |

---

## Known Technical Limitations

1. **Testbench Self-Checking (Tier 1 vs Tier 2):** Tier 1 core assertions run through `make test`. The Tier 2 bench is present. Exact comparison with the independent Rust integer reference remains the release gate.
2. **Q8.24 Rounding Strategy:** Rust export (`src/export.rs`) truncates toward zero (`as i32`), while the NPU epilogue reduces the 96-bit $W_0$ product with an arithmetic right shift (floor toward negative infinity, `npu/src/epilogue.sv`). The golden model must replicate the floor semantics for bit-exact parity; neither side uses round-half-to-even.
3. **Vector Tensor Storage (Resolved in PR #26):** Linear layers and backward passes have migrated to contiguous `Tensor<f32>` storage and matrix multiplication (`matmul`), eliminating `Vec<Vec<f32>>` in model execution. Residual legacy vector dependencies in `export.rs` have been updated to the tensor access interface.
4. **Gradient Verification:** The release gate requires finite differences on the float graph. Verify the loss reduction and both weight and bias gradients on a deterministic multilayer fixture. [Delivery plan](V0_DELIVERY_PLAN.md#22-how-do-we-restore-trustworthy-training-pending)
5. **Artifact Integration:** The training runner leaves hardware export inactive. The exporter writes parameter and inference files. Topology and activation selectors are supplied by the assembled wrapper and its consumer. `export_inference` writes float-derived predictions to `target.mem`. The independent Q8.24 reference must supply release expectations.
6. **Determinism:** The initializer advances a seeded generator within each layer and restarts from a fixed seed for every layer. The batch loader maintains its own epoch seed. A configured initialization stream across layers remains pending.
7. **RTL Activation Unit:** The activation unit implements ReLU only; `SIGMOID` is declared in `npu_pkg::activation_t` as reserved and falls through to the identity path, so selecting it routes the saturated sum unchanged. The sigmoid output layer of the built-in experiments has no hardware equivalent under the current sources, and the testbench must keep `act_fn != SIGMOID` until it is implemented.

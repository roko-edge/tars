# Project Roadmap and Engineering Organization

**Document type:** Internal project plan. Planned work is not part of the current
public API unless separately documented under [`docs/`](../README.md).

This document records active assignments, cross-functional responsibilities,
and integration rules for TARS development.

---

## What is the current v0.1.0 work?

The [v0.1.0 delivery plan](V0_DELIVERY_PLAN.md) records release evidence,
proposed hardening, and conflicts that need a maintainer disposition. The
[release contract](V0_CONTRACT.md) governs acceptance. This snapshot uses the
existing ownership assignments.

| Owner | Current assignment | Definition of done | Unblocks |
|---|---|---|---|
| Gustavo | [PENDING] Optimize tensor operations (`map` and `zip_map`), implement `Decision` rules (`Hardmax`, `Threshold`) separated from `Loss` per [`TRAINING.md`](../engineering/specs/TRAINING.md#17-decision-rules-and-accuracy-evaluation-decided), generalize `Sequential::evaluate_accuracy` to accept a `Decision` rule instead of hardcoded 10-class argmax, implement finite-difference gradient tests (`tests/grad_check.rs`), compute release statistics (McNemar and 95% CI), and implement topology graph visualization in Graphviz (`src/view/` and a planned visualization entry point). | Fast tensor map/zip_map applied across hot paths; `Decision` (`Hardmax`, `Threshold`) separated from differentiable `Loss`; `model.evaluate_accuracy(&dataset, &decision)` working for multi-class and binary tasks; numerical gradient check ($\epsilon \le 10^{-4}$) passing; statistical interval and paired comparison generated; visualization entry point producing valid diagrams; math and evaluation tests pass. | Ergonomic task-agnostic model evaluation, tensor execution speed, mathematical gradient proof, research telemetry, and architectural visualization |
| Gildo | [PENDING] Refactor `backward` to match composite `forward`, streamline the batch loader and Adam moment management, settle the proposed training API, and implement `Display` for `Sequential`. The current `Trainer` already uses fluent builders and returns `TrainSummary`. [Current training API](../API.md#3-training-operations) | Backward descent across composite modules, verified gradients, streamlined batching and optimizer updates, a finalized training API, and an architecture table rendered by `Display`. | Extensible composite backpropagation, clean training data flow, and high-level framework UX |
| Arthur | [PENDING] Deliver synthesizable NPU targeting the Urbana board (Spartan-7 / Vivado), close the quantizer contract (TS-04), implement the independent Q8.24 bit-exact integer reference (TS-05), connect the high-level `model.export_npu("npu/data/")` method, and verify the assembled wrapper against exact expected words in `tb_parity.sv`. | Synthesizable RTL achieving timing closure on Urbana; bit-exact integer reference running in Rust; `model.export_npu()` emitting valid `.mem` artifacts; small fixtures and trained MNIST model passing exact parity with zero mismatched output words in `tb_parity.sv`; held-out fixed-point score reaching the ternary threshold (>90%). | Synthesizable hardware NPU on Urbana FPGA, verified Rust–RTL deployment artifacts, and release parity |

Gildo owns closure of tensor packages WP-01, WP-03 through WP-05, the remaining
WP-07 clauses, and WP-10 through WP-13. The package dependencies and test
matrix define closure. [Tensor packages](TENSOR_PHASE_B.md#3-work-package-decomposition) ·
[Test matrix](TENSOR_PHASE_B.md#5-test-matrix)

---

## 1. System Vision and Boundaries

TARS is a TinyML co-design stack: quantization-first models with ternary weights,
inference on microcontroller-class CPUs and the ternary NPU, and training on
commodity GPUs. Capabilities that can be resolved at compile time — element types,
operation availability, and mixed-precision signatures — are resolved statically.
TARS unifies an experimental deep learning engine written in Rust with a synthesizable
hardware neural processing unit (NPU) specified in SystemVerilog.

```text
┌─────────────────────────────────────────────────────────────┐
│                        RUST RUNTIME                         │
│                                                             │
│   Model<M> ──► Module Composition ──► Tensor Engine         │
│        │                                     ▲              │
│        ▼ (Train)                             │ (Verify)     │
│   Analytic Backprop ──► Optimizer (BGD) ─────┘              │
│        │                                                    │
│        ▼ (Export)                                           │
│   Hardware Manifest + Q8.24 Memory Artifacts (.mem)         │
└──────────────────────────────┬──────────────────────────────┘
                               │ File / IPC Boundary
┌──────────────────────────────▼──────────────────────────────┐
│                    SYSTEMVERILOG NPU (DUT)                  │
│                                                             │
│   Memory Controller ──► Layer Engine FSM ──► MAC Array       │
│                                                   │         │
│   Self-Checking Testbench (Bit-Exact Parity) ◄────┘         │
└─────────────────────────────────────────────────────────────┘
```

The system requires strict parity between software inference and RTL simulation.
To achieve this without blocking parallel velocity, work is distributed across three
core engineers with explicit cross-domain intersections.

---

## 2. Engineering Matrix and Cross-Functional Intersections

Work is structured around five primary intersections where each domain features an
**Owner** (responsible for design and delivery) and a **Peer Reviewer** (responsible
for review and verification).

```text
                [ GUSTAVO ]
      Stochasticity, Numerics, Testing & Analytics
                  /         \
   Intersection 1 /           \ Intersection 5
 (Network Arch.)/             \(Stochastics & Benchmarks)
               /               \
        [ GILDO ] ─────────── [ ARTHUR ]
 Engine, Backprop &           NPU, Hardware Tests &
 NPU RTL Reviewer             Systems Integration
        \                       /
   Intersection 2 \           / Intersection 4
  (Tensor Core)  (Owner:     (NPU & Parity)
                  Gildo)    /
               \           /
            Intersection 3
        (Compute & Backprop)
```

### Intersections Specification

| Intersection | Subsystem | Lead (Owner) | Peer Reviewer | Joint Scope & Deliverables |
| --- | --- | --- | --- | --- |
| **1** | **Network Architecture (`Model` vs `Sequential`)** | Arthur | Gustavo | `Module` trait definition, parameter tracking, topologically ordered IR extraction, and deprecation of monolithic `Sequential`. |
| **2** | **Tensor Core, Layout & Data Types** | Gildo (Layout) / Gustavo (DType) | Gustavo / Gildo | Strided memory layout, zero-copy views (`strides`, `offset`), multidimensional broadcasting rules, and copy-on-write semantics (Gildo); the typed element layer (`Tensor<T>`: `f32` in the current phase, `T3` in the v0.1.0 arc), the `DType` wire tag, ternary storage representations, and mixed-precision operations (Gustavo). The `i8` element type is deferred beyond v0.1.0, and Q8.24 remains an export-only serialization format. |
| **3** | **Compute Engine, Tensorized Backprop & Quantization** | Gildo (Float) / Gustavo (Ternary) | Arthur / Gildo | Batched tensor float arithmetic (`matmul`, element-wise), elimination of `Vec<Vec<f32>>`, and tensorized analytical backprop (Gildo); ternary weight kernels, mixed-precision `matmul` ($\text{act} \times \text{ternary}$), and ternary quantizer logic (Gustavo). |
| **4** | **NPU Co-Processor, Parity & Ternary RTL** | Arthur | Gildo | Dense ternary layer engine over Q8.24 activations, signed 64-bit accumulation and epilogue saturation, Tier 1 hand-calculated checks, and exact Rust–RTL parity in Tier 2. The original Q8.24 weight prototype remains a historical synthesis baseline. |
| **5** | **Stochasticity, Numerical Controls & Benchmark Analytics** | Gustavo | Gildo | Deterministic PRNG harness, random weight initialization algorithms (Xavier/Glorot, He/Kaiming, ternary), Rust test suite, dataset loading, metric computation, and empirical research analysis. |

### Domain Responsibility Summary

- **Arthur** owns NPU microarchitecture (`layer_engine.sv`, `pe.sv`, `epilogue.sv`, `t3_dec.sv`, `npu_pkg.sv`), the synthesizable Urbana FPGA target (Spartan-7 / Vivado synthesis, pin constraints, clocking, timing closure), hardware testing (`npu/tests/tb.sv`, `tb_core.sv`), Verilator linting, VCD tracing, `Module` trait design, the high-level `model.export_npu()` method and layer-manifest format contract, the Track T quantizer and QAT engine (TS-04), the bit-exact golden inference model and golden vector export (TS-05), and the deferred `Model<M>` container. He supplies raw hardware metrics (cycle latency, resource utilization, parity outcomes) to Gustavo for benchmark aggregation.
- **Gustavo** owns stochastic weight generation (seeded PRNG, Xavier/He, ternary sampling), numerical controls (gradient validation with $\epsilon \le 10^{-4}$ in `tests/grad_check.rs`), automated Rust test suite for the stochasticity surface (`tests/`, `cargo test`), MNIST dataset ingestion, statistical evaluation harness (bootstrap 95% CI, confusion matrix, paired McNemar), benchmark analytics aggregation for research publications (`tars-research`), fast tensor iteration via `map` and `zip_map`, discrete `Decision` rules (`Hardmax`, `Threshold`) separated from `Loss`, model evaluation methods on `Sequential`/`Module`, and the planned topology graph visualization in Graphviz.
- **Gildo** owns the float tensor engine, recursive `backward` matching `forward` composite descent, losses and optimizers (including Adam moment streamlining), the streamlined batch loader, the high-level `Trainer` ergonomics, the `Display` implementation for `Sequential`, the MNIST runner (`examples/mnist.rs`), and the exporter implementation against the maintainer's contract. He reviews the NPU FSM, saturation logic, and parity co-simulation, and supplies profiling data to Gustavo.

---

## What are the training work packages?

The work-package IDs identify assignments. The delivery plan owns their
verification steps and the release contract owns acceptance.
[Training verification](V0_DELIVERY_PLAN.md#22-how-do-we-restore-trustworthy-training-pending) ·
[Release contract](V0_CONTRACT.md)

The training stack delivers the software capabilities required to train, quantize, and verify the $784 \to 64 \to 10$ MNIST network on the release path. Interfaces are frozen in [`TRAINING.md`](../engineering/specs/TRAINING.md) and [`QUANT.md`](../engineering/specs/QUANT.md).

| WP | Scope | Owner | Exit condition |
|---|---|---|---|
| TS-01 | `Loss` trait: `forward(&logits,&target)->f32` + `backward()->Tensor<f32>` (grad wrt logits). Impls: MSE (OR/XOR kept) + SoftmaxCE (MNIST; grad $p - y$) | Gildo | both losses pass tests; OR/XOR regression green |
| TS-02 | Batch loader: `Data` → batches with seeded shuffle; batch size parameter; epoch loop | Gildo | two runs with same seed give identical trajectory |
| TS-03 | `Optimizer::step(&mut Sequential,&Grad)` with BGD as the v0.1.0 optimizer of record | Gildo | BGD on `Tensor<f32>` is functional and tested. Adam is an optional follow-up. |
| TS-04 | Quantizer (absmean + threshold + STE) + QAT engine (quantized-weight forward, STE backward) | Arthur | PTQ baseline measured; QAT converges on ternary OR/XOR |
| TS-05 | Bit-exact integer reference, validated export, and `tb_parity` | Arthur | Zero mismatched words on the trained MNIST fixture. The release revision passes RTL lint, Tier 1 and Tier 2. |
| TS-06 | MNIST release evidence: held-out evaluation (Gustavo), float accuracy $\ge 95\%$ (Gildo), fixed-point ternary accuracy $> 90\%$ and parity (Arthur), 95% binomial confidence interval and paired comparison (Gustavo) | all three | The evidence satisfies the [release contract](V0_CONTRACT.md). |

---

## 4. Engineering Invariants and Anti-Regression Policy

1. **Continuous Verification:** The legacy training path must remain functional until XOR convergence on the tensor backend is verified against the golden characterization fixtures.
2. **Deterministic Reproducibility:** Every automated test, benchmark, and experiment must initialize PRNG state with a hard-coded seed.
3. **Interface Authority:** Design standards and modifications to public structs or traits in `src/math/` or `src/modules/` are set by the maintainer, recorded with date in research logs, with zero sign-off ceremony.
4. **Hardware-Software Synchronization:** Standards are defined by the maintainer. Design decisions are recorded with date in the research log; implementation and documentation move in one revision. Any modification to numeric representation, scaling, or rounding behavior requires simultaneous updates to `src/export.rs`, documentation, and the NPU RTL (`npu/src/epilogue.sv`, `npu/src/layer_engine.sv`, `npu/src/t3_dec.sv`). No sign-off ceremony: the maintainer decides, the history records.
5. **Empirical Data Provenance:** All published benchmarks, convergence metrics, and research claims must be generated from version-controlled fixtures managed under `docs/project/research/` with documented random seeds.

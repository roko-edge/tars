# Project Roadmap and Engineering Organization

**Document type:** Internal project plan. Planned work is not part of the current
public API unless separately documented under [`docs/`](../README.md).

This document establishes the technical milestones, cross-functional responsibilities,
and integration contracts for TARS development.

---

## What is the current v0.1.0 work?

The [v0.1.0 delivery plan](V0_DELIVERY_PLAN.md) records release evidence,
proposed hardening, and conflicts that need a maintainer disposition. The
[release contract](V0_CONTRACT.md) governs acceptance. This snapshot uses the
existing ownership assignments.

| Owner | Current assignment | Definition of done | Unblocks |
|---|---|---|---|
| Gustavo | [PENDING] Optimize tensor operations (`map` and `zip_map`), add hardmax to loss/decision options, attach evaluation methods (`evaluate_accuracy`) directly to `Sequential`/`Module`, implement finite-difference gradient tests (`tests/grad_check.rs`), compute release statistics (McNemar and 95% CI), and implement topology graph visualization in Graphviz (`src/view/` and `src/bin/visualization.rs`). | Fast tensor map/zip_map applied across hot paths; hardmax available; `model.evaluate_accuracy(&dataset)` callable directly; numerical gradient check ($\epsilon \le 10^{-4}$) passing; statistical interval and paired comparison generated; visualization binary producing valid diagrams; math and evaluation tests pass. | Ergonomic model evaluation, tensor execution speed, mathematical gradient proof, research telemetry, and architectural visualization |
| Gildo | [PENDING] Refactor `backward` to be recursive matching `forward`, streamline and simplify the batch loader, improve `Adam` ergonomics and moment management, implement `Trainer` fluent builder (`&mut Self`) and `TrainResults`, and implement `Display` for `Sequential` (`println!("{model}")`). | Recursive backward descent across modules; simplified batching pipeline; clean Adam optimizer interface; fluent `Trainer` returning structured `TrainResults`; visual architecture table displayed by `Display`; gradient checks pass. | Extensible composite backpropagation, clean training data flow, and high-level framework UX |
| Arthur | [PENDING] Deliver synthesizable NPU targeting the Urbana board (Spartan-7 / Vivado), close the quantizer contract (TS-04), implement the independent Q8.24 bit-exact integer reference (TS-05), connect the high-level `model.export_npu("npu/data/")` method, and verify the assembled wrapper against exact expected words in `tb_parity.sv`. | Synthesizable RTL achieving timing closure on Urbana; bit-exact integer reference running in Rust; `model.export_npu()` emitting valid `.mem` artifacts; small fixtures and trained MNIST model passing exact parity with zero mismatched output words in `tb_parity.sv`; held-out fixed-point score reaching the ternary threshold (>90%). | Synthesizable hardware NPU on Urbana FPGA, verified Rust–RTL deployment artifacts, and release parity |

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
- **Gustavo** owns stochastic weight generation (seeded PRNG, Xavier/He, ternary sampling), numerical controls (gradient validation with $\epsilon \le 10^{-4}$ in `tests/grad_check.rs`), automated Rust test suite for the stochasticity surface (`tests/`, `cargo test`), raw MNIST dataset ingestion (`src/data/mnist.rs`), statistical evaluation harness (bootstrap 95% CI, confusion matrix, paired McNemar), benchmark analytics aggregation for research publications (`tars-research`), fast tensor iteration via `map` and `zip_map`, hardmax loss/decision functions, model evaluation methods attached directly to `Sequential`/`Module`, and the topology graph visualization in Graphviz (`src/view/` and `src/bin/visualization.rs`).
- **Gildo** owns the float tensor engine, recursive `backward` matching `forward` composite descent, losses and optimizers (including Adam moment streamlining), the streamlined batch loader, the high-level `Trainer` ergonomics (fluent builder and `TrainResults`), the `Display` implementation for `Sequential`, the MNIST runner (`src/bin/main.rs`), and the exporter implementation against the maintainer's contract. He reviews the NPU FSM, saturation logic, and parity co-simulation, and supplies profiling data to Gustavo.

---

## 3. Milestone Execution Plan (Sequential Phases)

Phases are ordered by dependency. The checklists retain the original work
packages. Use [current assignments](#what-is-the-current-v010-work) and
the [delivery plan](V0_DELIVERY_PLAN.md) for active status and release gates.

Deliverables annotated **[done]** are present in the current repository. The annotation
records implementation presence only; phase gates remain closed until their verification
conditions are met and peer reviews are recorded.

### Phase 1: Contracts and Deterministic Baselines

- **Arthur:**
  - [x] Author normative specifications: `docs/engineering/specs/MODEL_TRAIT.md` and `docs/engineering/specs/ARITHMETIC_Q8_24.md`.
  - [x] Sanitize `npu/Makefile`: eliminate broken simulation/ternary targets, wire up automated Verilator linting (`make lint`).
  - [x] Upgrade `npu/tests/tb.sv` to support `$dumpfile` and `$dumpvars` for GTKWave tracing.

- **Gustavo:**
  - [x] Author normative specification: `docs/engineering/specs/TENSOR.md`.
  - [ ] Expose a recorded initialization seed and advance one stream across layers. The current initializer restarts a fixed seed for each layer.
  - [ ] Construct baseline characterization tests: capture loss trajectories and canonical weights for OR and XOR models using current engine. No regression fixture exists.
  - [x] Implement the baseline `cargo test` harness. Integration tests live under `tests/`.

- **Gildo:** The tensor foundation, the memory operations, and the products were delivered by the maintainer; the open clauses below constitute the current single closure package (WP-01, WP-03, WP-04, WP-05, the WP-07 gaps, WP-10, WP-11, WP-12, WP-13), with the acceptance criteria of [`TENSOR_PHASE_B.md`](TENSOR_PHASE_B.md), Section 5, as the definition of done.
  - [x] Review `docs/engineering/specs/TENSOR.md` for algorithm suitability. The outcome is the revision of the specification and the delivery plan at `docs/project/TENSOR_PHASE_B.md`, which decomposes the phase into the work packages WP-01 through WP-13.
  - [ ] Deliver the foundation of the `F32` tensor core: a one-byte copyable and comparable type tag, separation of the public `contiguous` operation from the internal stride helper, rank-zero-safe logical index iteration, and a descriptive diagnostic on every precondition violation.
  - [ ] Deliver the geometry and the views: validation of the element capacity at construction, logical-element addressing per Section 2.3 of the specification, and the axis swap in both variants under the contract of Section 4.
  - [x] Deliver the memory operations and the element-wise kernels: the public `contiguous` operation, the pointwise kernels `add`, `sub`, `mul_scalar`, `div_scalar`, and `map`, and the full reduction `sum`, each honouring the stride vector of the operand.
  - [x] Deliver the products `matmul`, `matvec`, and `outer` over `F32`, reproducing the forward pass of the linear layer in the orientation it consumes.
  - [ ] Deliver the bulk access surface required by the exporter, the graph renderer, and the later migration of the linear layer.
  - [ ] Establish the tensor test suite: geometry, construction preconditions, contiguity, views, kernels, reduction, products, bulk access, and the failure path of every precondition of Section 4.1 of the specification.
  - [ ] Perform initial peer review of NPU RTL files (`npu/src/layer_engine.sv`, `npu/src/pe.sv`, `npu/src/epilogue.sv`, `npu/src/t3_dec.sv`, `npu/tests/tb.sv`, `npu/Makefile`). Recorded findings: the testbench asserts no expected value; and the package roles declared in `flake.nix` do not correspond to the flow driven by `npu/Makefile`. The earlier findings against `npu.sv` (four index-width warnings) and the missing Python 3 interpreter were resolved by the file's replacement and by `flake.nix`; the two `WIDTHEXPAND` findings on the `safe_log` calls in `layer_engine.sv` were resolved with explicit `64'` casts, and `make lint` exits zero (October 2026).

**Phase 1 Gate:** All specification documents merged into `main`. Canonical XOR loss curves locked in regression fixtures via `cargo test`. NPU testbench compiles and traces cleanly under Verilator and passes initial RTL review.

**Gate state:** [PENDING] Deterministic OR/XOR regressions and remaining
tensor contract checks are tracked in the delivery plan. The current Tier 1
bench asserts hand-calculated results and CI runs it alongside RTL lint.
[Implementation status](STATUS.md) · [Training verification](V0_DELIVERY_PLAN.md#22-how-do-we-restore-trustworthy-training-pending)

---

### Phase 2: Core Abstractions and Hardware Upgrades

- **Gustavo:**
  - [ ] Finish the experimental `T3` contract in `src/tensor/ternary_tensor.rs`. Packed storage and the mixed product exist. Scaled reads and the packed-view rules remain open. Typed T3 storage is decoupled from the release pipeline by LOG-008.
  - [ ] Implement weight initialization algorithms (Xavier/Glorot, He/Kaiming, and ternary $\{-1, 0, +1\}$ distribution) with deterministic seed support.
  - [ ] Extend `tests/` for seeded initialization, ternary sampling, and OR/XOR regression fixtures. Gildo owns the tensor operation suite under WP-13 of [`TENSOR_PHASE_B.md`](TENSOR_PHASE_B.md).
  - [ ] Design benchmark harness for tracking loss curves, training convergence rates, and memory allocations.
  - [ ] Implement dataset loading and batching pipeline for MNIST.
- **Gildo:**
  - [ ] Deliver the deferred zero-copy tensor views (`slice`, `permute`, `reshape`) via strided index projection, each of which requires a contract clause in `docs/engineering/specs/TENSOR.md` before implementation. The axis swap is delivered in Phase 1.
  - [ ] Implement NumPy-compliant multidimensional broadcasting resolution and safe copy-on-write (COW) mutation semantics (`Arc::make_mut`).
  - [ ] Implement cache-friendly row-major matrix multiplication (`matmul`) with dimension assertions.
  - [ ] Review Arthur's NPU microarchitecture updates, including FSM transitions, signed 64-bit accumulation, both epilogue saturations, and signed ReLU.
- **Arthur:**
  - [ ] Refactor `Sequential` to implement the new `Module` trait: `ReLU` and `Sigmoid` become distinct `Module` structs, no activation enumeration remains in the public surface, and the consuming builder (`new`, `linear`, `relu`, `sigmoid`, with running-dimension tracking and the seeded initializer) is preserved ([`MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md), Section 2.2).
  - [ ] Introduce `Model<M>` container owning parameter lifetime management. Deferred until a non-sequential topology exists (maintainer decision, October 2026); parameter tracking is served by the `Module` parameter surfaces in the interim.
  - [x] Upgrade the NPU microarchitecture with 64-bit accumulation, epilogue saturation, and signed ReLU.
  - [ ] Maintain hardware testbench in `npu/tests/tb.sv` and supply raw cycle count metrics to Gustavo.

**Phase 2 Gate:** Matrix multiplication, tensor views, and `DType` abstractions pass automated unit tests (`cargo test`). NPU RTL compiles cleanly under Verilator without warnings and passes Gildo's peer review.

---

### Phase 3: Engine Convergence, Tensorized Backpropagation, and Ternary Support

- **Gildo (Lead) & Arthur (Pair):**
  - [ ] Migrate `Linear` layer to store weights and biases as contiguous `Tensor` instances.
  - [ ] Re-engineer `backward()`: perform full backward pass using batched matrix multiplications, eliminating `Vec<Vec<f32>>` allocations; the pass consumes the intermediates recorded by the composite forward ([`MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md), Section 2.2) and does not re-evaluate the forward chain.
  - [ ] Eliminate redundant model cloning in gradient structures (`Grad`).
  - [ ] Deliver the reductions along arbitrary axes and the `mean` reduction, each of which requires a contract clause in `docs/engineering/specs/TENSOR.md` before implementation; the full reduction `sum` is delivered in Phase 1.
  - [ ] Review NPU multi-cycle Layer Engine FSM implementation.
  - [ ] Complete release export with validated model metadata, dense ternary parameters, and exact inference fixtures. The programmable `program.mem` artifact belongs to v0.1.1. [Artifact reference](../engineering/MEM_FORMAT.md)
- **Gustavo:**
  - [ ] Implement ternary weight compute kernels: mixed-precision `matmul` ($\text{act} \times \text{ternary}$) with conditional accumulation and scaling by $W_0$.
  - [ ] Execute numerical gradient validation: verify analytical tensor backprop against finite differences (`num_grad`) with tolerance $\epsilon \le 10^{-4}$, on the `F32` graph with quantizers disabled; quantizer gradients follow the straight-through estimator.
  - [ ] Conduct quantitative analysis of quantization error ($f32 \to \text{Q8.24}$ and ternary loss bounds) and saturation effects.
  - [ ] Run benchmark suite on XOR and multi-layer configurations; log performance metrics, convergence curves, and memory profiles into `docs/project/research/`.
- **Arthur:**
  - [x] Design the multi-cycle layer engine for sequential output computation. The assembled wrapper chains two engines and the epilogue supports Identity and ReLU.
  - [x] Implement the ternary multiply-accumulate execution mode of the release engine: 2-bit dense weights (decode `v = b_0 - b_1` in `t3_dec.sv`), conditional accumulation in `pe.sv`, and the per-layer $W_0$ epilogue in `epilogue.sv` (RTL committed October 2026; golden-vector validation remains gated below). The dense Q8.24 MAC is retired from the tree and retained in repository history as the development substrate and the synthesis baseline (maintainer decision, October 2026). The MAC obligations of [`NPU_SPECIALIZATION_PROPOSAL.md`](../engineering/NPU_SPECIALIZATION_PROPOSAL.md), Section 3, apply: the processing element is an isolated module, negation is exact at N+1 bits, and a decode stage decouples the packing format from the accumulate path.
  - [ ] Close the release artifact contract for the assembled wrapper and integer reference. The programmable descriptor belongs to v0.1.1. The reserved sparse mode uses the LOG-006 token stream. [Assembly model](../engineering/ASSEMBLY_AND_TEST_MODEL.md)
  - [x] Implement mathematical quantization algorithms in `src/quant/` (TS-04: absmean scale, threshold $\Delta$, PTQ engine in `src/quant/ternary.rs` landed in PR #30).
  - [ ] Implement the bit-exact ternary inference simulator and upgrade the existing `tb_parity.sv` to exact word comparison (TS-05).
  - [ ] Maintain hardware simulation suite and deliver cycle timing data to Gustavo.

**Phase 3 Gate:** Training loop converges on XOR using pure `Tensor` backend. Numerical gradient check passes ($\epsilon \le 10^{-4}$). Ternary mixed-precision operations pass verification against golden software vectors. NPU Layer Engine FSM passes simulation and Gildo's RTL review.

---

### Track T: Training Stack (v0.1.0 Release Path)

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

### Phase 4: Integrated System Verification

- **Arthur (Lead) & Gildo (RTL Reviewer):**
  - [ ] Export trained XOR model from Rust runtime into unified `.mem` format with manifest.
  - [ ] Implement self-checking testbench in `npu/tests/tb.sv`: reads input activations, computes hardware output, and asserts equivalence against exported golden vectors with non-zero exit codes on failure.
  - [ ] Document bit-exact parity proof across all 4 XOR input combinations.
- **Gustavo:**
  - [ ] Lead final evaluation, benchmark reporting, and scientific analysis for the v0.1.0 release.
  - [ ] Consolidate software accuracy (f32 MNIST $\ge 95\%$; ternary $> 90\%$ on the bit-exact path), hardware cycle latency, and quantization loss metrics into `docs/project/research/`.
  - [ ] Validate end-to-end regression fixtures ensuring bit-exact alignment between software Q8.24 inference and RTL simulation.
- **Gildo:**
  - [ ] Deprecate and remove legacy vector math code paths.
  - [ ] Audit finalized NPU testbench results and verify RTL hardware verification outcomes.

**Phase 4 Gate:** End-to-end demonstration: Rust train ──► Export ──► NPU simulation PASS with zero bit mismatches. Full benchmark report and research logs consolidated in `docs/project/research/`.

---

## 4. Engineering Invariants and Anti-Regression Policy

1. **Continuous Verification:** The legacy training path must remain functional until Phase 3 convergence is verified against the golden characterization fixtures.
2. **Deterministic Reproducibility:** Every automated test, benchmark, and experiment must initialize PRNG state with a hard-coded seed.
3. **Interface Authority:** Design standards and modifications to public structs or traits in `src/math/` or `src/modules/` are set by the maintainer, recorded with date in research logs, with zero sign-off ceremony.
4. **Hardware-Software Synchronization:** Standards are defined by the maintainer. Design decisions are recorded with date in the research log; implementation and documentation move in one revision. Any modification to numeric representation, scaling, or rounding behavior requires simultaneous updates to `src/export.rs`, documentation, and the NPU RTL (`npu/src/epilogue.sv`, `npu/src/layer_engine.sv`, `npu/src/t3_dec.sv`). No sign-off ceremony: the maintainer decides, the history records.
5. **Empirical Data Provenance:** All published benchmarks, convergence metrics, and research claims must be generated from version-controlled fixtures managed under `docs/project/research/` with documented random seeds.

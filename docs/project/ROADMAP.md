# Project Roadmap and Engineering Organization

**Document type:** Internal project plan. Planned work is not part of the current
public API unless separately documented under [`docs/`](../README.md).

This document establishes the technical milestones, cross-functional responsibilities,
and integration contracts for TARS development.

---

## Current Assignments

This section is the live snapshot that answers what each member must do next. The
numbered sections below remain the normative plan. The snapshot is updated whenever
a package closes or a gate passes. When a member is idle or blocked, the next
assignment of the same phase comes first; starting a work package owned by another
member requires an explicit reassignment by the lead.

| Owner | Current assignment | Definition of done | Unblocks |
|---|---|---|---|
| Gustavo | P0, determinism and fixtures: replace the unseeded global generator with an explicitly seeded PRNG; correct `ternary_distribution_test` (call `ternary_rng`, assert the variance around one with a tolerance, check the ternary support); lock the OR and XOR loss trajectories and canonical weights into `cargo test`. Next assignment of the phase: the typed-element migration, porting `src/math/tensor.rs` to the revised `Tensor<T>` contract of [`TENSOR.md`](../engineering/specs/TENSOR.md) (the `Element` trait, per-type storage, per-type implementation blocks) | Phase 1 Gate, condition 2; the suite becomes deterministic and green; the typed migration lands with the `f32` kernel behavior unchanged and the suite green, with the operation methods as thin dispatchers over the per-type kernel modules (the kernel seam of [`TENSOR.md`](../engineering/specs/TENSOR.md), Section 4) | The regression backbone of the migration phase and every downstream benchmark; Gildo's continued closure package on the revised shape and the migration phase |
| Gildo | F32 closure package: WP-01, WP-03, WP-04, WP-05, the WP-07 gaps, WP-10, WP-11, WP-12, WP-13; author the `backward` clause draft in [`MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md) under the Module trait sign-off, drafted against the intermediates recorded by the composite forward (Section 2.2 of the specification), as the parallel assignment while the typed migration is pending | The test matrix of [`TENSOR_PHASE_B.md`](TENSOR_PHASE_B.md), Section 5, fully green, including every failure path; the `MODEL_TRAIT` revision merged with sign-off | The `Linear` migration, the exporter (WP-12 bulk access), and the packed `T3` access (WP-14 depends on WP-04 and WP-11) |
| Arthur | NPU execution and hardware contracts: begin LOG-007 Step 2 (runtime shape registers, the linear weight-stream pointer, the `layer_engine.sv` extraction); author the layer-manifest format contract (LOG-007, Section 4.2, Step 1) and retain its sign-off; retain the Module trait sign-off of the `MODEL_TRAIT` `backward` clause; adjust the module forwards and the composite evaluation to record the intermediates that `backward` requires ([`MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md), Section 2.2); the v0.1.0 obligations of [`NPU_SPECIALIZATION_PROPOSAL.md`](../engineering/NPU_SPECIALIZATION_PROPOSAL.md), Section 3, ride with the engine steps | The Step 2 gate: legacy single-layer results identical and zero width warnings; the manifest contract recorded with sign-off; the training step evaluates the forward chain once and `backward` consumes the recorded trace | NPU Steps 3 through 5 and the Phase 3 exporter implementation |
| Gustavo (subsequent; blocked by Gildo's closure) | Packed `T3` access, WP-14 T3 scope: the ternary encoding of [`TENSOR.md`](../engineering/specs/TENSOR.md), Section 3.1, dequantization, and byte alignment (Section 2.2, Invariant 7). The `I8` variant is out of scope for v0.1.0. While waiting, the parallel work is the quantizer mathematics ($\Delta$, $W_0$) over latent `F32` weights and the bit-exact simulator specification, which do not depend on the tensor core | The subsequent-phase exit condition of [`TENSOR.md`](../engineering/specs/TENSOR.md), Section 4.2, for the `T3` variant | The quantizer chain, QAT, and the golden-vector path |

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
for contract sign-off and verification).

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
| **4** | **NPU Co-Processor, Parity & Ternary RTL** | Arthur | Gildo | Multi-neuron Layer Engine FSM, dense ternary MAC (2-bit weights, conditional accumulation, per-layer $W_0$ epilogue) over the Q8.24 activation substrate, 64-bit saturating accumulator, self-checking testbench (`tb.sv`), and golden-model bit-exact test suite. The Q8.24 prototype remains the development substrate and synthesis baseline. |
| **5** | **Stochasticity, Numerical Controls & Benchmark Analytics** | Gustavo | Gildo | Deterministic PRNG harness, random weight initialization algorithms (Xavier/Glorot, He/Kaiming, ternary), Rust test suite, dataset loading, metric computation, and empirical research analysis. |

### Domain Responsibility Summary

- **Arthur** owns NPU microarchitecture (`npu.sv`, `pe.sv`), the ternary execution mode of the release engine (with the Q8.24 prototype as the development substrate and synthesis baseline), hardware testing (`npu/tests/tb.sv`), Verilator linting, VCD tracing, `Module` trait design, the layer-manifest format contract and its sign-off (the exporter implementation is delegated to Gildo), and the deferred `Model<M>` container. He supplies raw hardware metrics (cycle latency, resource utilization, parity outcomes) to Gustavo for benchmark aggregation.
- **Gustavo** owns the `DType` enum on `Tensor`, ternary tensor operations (mixed-precision matmul, conditional additions/subtractions with scale $W_0$), mathematical quantization algorithms in `src/quant/` (threshold $\Delta$ and scale $W_0$), stochastic weight generation (seeded PRNG, Xavier/He, ternary sampling), the bit-exact ternary inference simulator and its golden vectors (consumed by the NPU testbench and by the ternary release-gate accuracy measurement), numerical controls (gradient validation with $\epsilon \le 10^{-4}$, Q8.24 and ternary quantization error analysis, saturation checks), the automated Rust test suite for the stochasticity surface (`src/tests/`, `cargo test`), dataset loaders (MNIST), and benchmark harnesses.
- **Gildo** owns the continuous tensor execution engine (float `matmul`, element-wise float kernels, axis reductions), tensor memory layout (strides, views, broadcasting, COW), `Linear` layer migration to contiguous `Tensor`, tensorized continuous `backward()`, the exporter implementation over the maintainer's manifest contract (`program.mem`, `wt.mem`, `w0.mem`, golden vectors), and serves as official RTL reviewer for the NPU (`npu.sv`, `pe.sv`, `tb.sv`), auditing FSM correctness, saturation logic, and parity co-simulation. He supplies engine profiling data to Gustavo for benchmark aggregation.

---

## 3. Milestone Execution Plan (Sequential Phases)

Phases are strictly ordered by dependency, not by calendar. Each phase is completed when its gate condition is met.

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
  - [ ] Implement deterministic pseudo-random number generator (PRNG) with explicit seed support (replacing unseeded uniform randoms). The current implementation draws from the unseeded global generator.
  - [ ] Construct baseline characterization tests: capture loss trajectories and canonical weights for OR and XOR models using current engine. No regression fixture exists.
  - [x] Implement baseline `cargo test` harness in `src/tests/` for automated Rust regression testing.

- **Gildo:** The tensor foundation, the memory operations, and the products were delivered by the maintainer; the open clauses below constitute the current single closure package (WP-01, WP-03, WP-04, WP-05, the WP-07 gaps, WP-10, WP-11, WP-12, WP-13), with the acceptance criteria of [`TENSOR_PHASE_B.md`](TENSOR_PHASE_B.md), Section 5, as the definition of done.
  - [x] Review `docs/engineering/specs/TENSOR.md` for algorithm suitability. The outcome is the revision of the specification and the delivery plan at `docs/project/TENSOR_PHASE_B.md`, which decomposes the phase into the work packages WP-01 through WP-13.
  - [ ] Deliver the foundation of the `F32` tensor core: a one-byte copyable and comparable type tag, separation of the public `contiguous` operation from the internal stride helper, rank-zero-safe logical index iteration, and a descriptive diagnostic on every precondition violation.
  - [ ] Deliver the geometry and the views: validation of the element capacity at construction, logical-element addressing per Section 2.3 of the specification, and the axis swap in both variants under the contract of Section 4.
  - [x] Deliver the memory operations and the element-wise kernels: the public `contiguous` operation, the pointwise kernels `add`, `sub`, `mul_scalar`, `div_scalar`, and `map`, and the full reduction `sum`, each honouring the stride vector of the operand.
  - [x] Deliver the products `matmul`, `matvec`, and `outer` over `F32`, reproducing the forward pass of the linear layer in the orientation it consumes.
  - [ ] Deliver the bulk access surface required by the exporter, the graph renderer, and the later migration of the linear layer.
  - [ ] Establish the tensor test suite: geometry, construction preconditions, contiguity, views, kernels, reduction, products, bulk access, and the failure path of every precondition of Section 4.1 of the specification.
  - [ ] Perform initial peer review of NPU RTL files (`npu/src/npu.sv`, `npu/src/pe.sv`, `npu/tests/tb.sv`, `npu/Makefile`). Recorded findings: four index-width warnings at `npu/src/npu.sv` lines 39, 40, 47, and 87; the testbench asserts no expected value and returns no non-zero exit code; the simulation flow requires a Python 3 interpreter that the `hardware` development shell does not provide; and the package roles declared in `flake.nix` do not correspond to the flow driven by `npu/Makefile`.

**Phase 1 Gate:** All specification documents merged into `main`. Canonical XOR loss curves locked in regression fixtures via `cargo test`. NPU testbench compiles and traces cleanly under Verilator and passes initial RTL review.

**Gate state:** Open. The specification revision and the delivery plan are merged into `main`, closing the first condition. The `cargo test` suite holds four tests, of which one exercises the tensor core; the suite is not deterministic because the distribution tests sample the unseeded global generator, and no regression fixture captures the loss trajectories of the OR and XOR models, so the second condition of the gate is unmet. The testbench compiles and traces under Verilator in the sanctioned `hardware` shell, whose `flake.nix` now provides the Python 3 interpreter required by the Verilator binary build; the initial RTL review recorded findings that remain open, and lint still exits non-zero with two index-width warnings in `npu/src/npu.sv` under Verilator 5.052, so the third condition is unmet as well. The tensor foundation, the memory operations, the element-wise kernels, and the products of the `F32` core are present in the repository; the quality clauses of the delivery plan (capacity validation, rank-zero-safe iteration, non-empty diagnostics, the bulk access surface, and the test suite) remain open.

---

### Phase 2: Core Abstractions and Hardware Upgrades

- **Gustavo:**
  - [ ] Deliver the `T3` element type of the typed tensor core (`Tensor<T>`) in `src/math/tensor.rs`: the associated packed storage, the ternary encoding of Section 3.1, and the dequantizing read. The `i8` element type and the scaling of Section 3.3 are deferred beyond v0.1.0 (maintainer decision, October 2026). The one-byte copyable and comparable wire tag is delivered in Phase 1.
  - [ ] Implement weight initialization algorithms (Xavier/Glorot, He/Kaiming, and ternary $\{-1, 0, +1\}$ distribution) with deterministic seed support.
  - [ ] Build unit tests in `src/tests/` for the stochasticity surface (seeded PRNG, Xavier/He, ternary sampling) and the OR/XOR regression fixtures. The tensor operation suite is owned by Gildo under WP-13 of [`TENSOR_PHASE_B.md`](TENSOR_PHASE_B.md).
  - [ ] Design benchmark harness for tracking loss curves, training convergence rates, and memory allocations.
  - [ ] Implement dataset loading and batching pipeline for MNIST.
- **Gildo:**
  - [ ] Deliver the deferred zero-copy tensor views (`slice`, `permute`, `reshape`) via strided index projection, each of which requires a contract clause in `docs/engineering/specs/TENSOR.md` before implementation. The axis swap is delivered in Phase 1.
  - [ ] Implement NumPy-compliant multidimensional broadcasting resolution and safe copy-on-write (COW) mutation semantics (`Arc::make_mut`).
  - [ ] Implement cache-friendly row-major matrix multiplication (`matmul`) with dimension assertions.
  - [ ] Review Arthur's NPU microarchitecture updates: audit FSM transitions, 64-bit saturating accumulator arithmetic, and signed ReLU activation logic.
- **Arthur:**
  - [ ] Refactor `Sequential` to implement the new `Module` trait: `ReLU` and `Sigmoid` become distinct `Module` structs, no activation enumeration remains in the public surface, and the consuming builder (`new`, `linear`, `relu`, `sigmoid`, with running-dimension tracking and the seeded initializer) is preserved ([`MODEL_TRAIT.md`](../engineering/specs/MODEL_TRAIT.md), Section 2.2).
  - [ ] Introduce `Model<M>` container owning parameter lifetime management. Deferred until a non-sequential topology exists (maintainer decision, October 2026); parameter tracking is served by the `Module` parameter surfaces in the interim.
  - [x] Upgrade NPU microarchitecture: expand the accumulator to 64 bits with final saturation logic (signed ReLU activation unit remains pending).
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
  - [ ] (Led by Gildo; format contract and review by the maintainer) Implement the exporter against the maintainer's manifest contract: layer descriptors, the layer program (`program.mem`), dense ternary weights (`wt.mem`), per-layer scale parameters (`w0.mem`), and golden inference vectors, consuming the bulk access surface and the quantizer output.
- **Gustavo:**
  - [ ] Implement ternary weight compute kernels: mixed-precision `matmul` ($\text{act} \times \text{ternary}$) with conditional accumulation and scaling by $W_0$.
  - [ ] Implement mathematical quantization algorithms in `src/quant/` mapping continuous floats to ternary states $\{-W_0, 0, +W_0\}$.
  - [ ] Implement the bit-exact ternary inference simulator: the Rust reference reproducing the NPU datapath (truncation, arithmetic shift by 24, saturation, per-layer $W_0$ epilogue), producing the golden vectors consumed by `npu/tests/tb.sv` and measuring the ternary release-gate accuracy on the bit-exact path.
  - [ ] Execute numerical gradient validation: verify analytical tensor backprop against finite differences (`num_grad`) with tolerance $\epsilon \le 10^{-4}$, on the `F32` graph with quantizers disabled; quantizer gradients follow the straight-through estimator.
  - [ ] Conduct quantitative analysis of quantization error ($f32 \to \text{Q8.24}$ and ternary loss bounds) and saturation effects.
  - [ ] Run benchmark suite on XOR and multi-layer configurations; log performance metrics, convergence curves, and memory profiles into `docs/project/research/`.
- **Arthur:**
  - [x] Design multi-cycle Layer Engine FSM in `npu/src/npu.sv` for sequential computation of $M$ output neurons from $N$ inputs (multi-layer chaining and activation units remain pending).
  - [ ] Implement the ternary multiply-accumulate execution mode of the release engine: 2-bit dense weights, conditional accumulation, and the per-layer $W_0$ epilogue. The Q8.24 MAC remains as the development substrate and the synthesis baseline (maintainer decision, October 2026). The MAC obligations of [`NPU_SPECIALIZATION_PROPOSAL.md`](../engineering/NPU_SPECIALIZATION_PROPOSAL.md), Section 3, apply: the processing element is an isolated module, negation is exact at N+1 bits, and a decode stage decouples the packing format from the accumulate path.
  - [ ] Author the layer-manifest format contract (four-word descriptor, mode semantics, `program.mem`/`wt.mem`/`w0.mem` layout, per LOG-007 Section 4.2 Step 1) and retain its sign-off; the contract names the consumers of the format — the NPU testbench, the bit-exact simulator, and the embedded inference runtime load the same artifacts. The sparse stream-entry decision (the mode 2 token stream versus a reserved dense-stream code) is the open item of [`NPU_SPECIALIZATION_PROPOSAL.md`](../engineering/NPU_SPECIALIZATION_PROPOSAL.md), Section 3. The exporter implementation is delegated to Gildo.
  - [ ] Maintain hardware simulation suite and deliver cycle timing data to Gustavo.

**Phase 3 Gate:** Training loop converges on XOR using pure `Tensor` backend. Numerical gradient check passes ($\epsilon \le 10^{-4}$). Ternary mixed-precision operations pass verification against golden software vectors. NPU Layer Engine FSM passes simulation and Gildo's RTL review.

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
  - [ ] Audit finalized NPU testbench results and sign off on RTL hardware verification.

**Phase 4 Gate:** End-to-end demonstration: Rust train ──► Export ──► NPU simulation PASS with zero bit mismatches. Full benchmark report and research logs consolidated in `docs/project/research/`.

---

## 4. Engineering Invariants and Anti-Regression Policy

1. **Continuous Verification:** The legacy training path must remain functional until Phase 3 convergence is verified against the golden characterization fixtures.
2. **Deterministic Reproducibility:** Every automated test, benchmark, and experiment must initialize PRNG state with a hard-coded seed.
3. **Interface Sign-Off:** No modifications to public structs or traits in `src/math/` or `src/modules/` are permitted without approval from the respective intersection owners.
4. **Hardware-Software Dual Sign-Off:** Any modification to numeric representation, scaling, or rounding behavior requires simultaneous updates to both `src/export.rs` and `npu/src/npu.sv`, with sign-off from Arthur (Hardware Owner), Gildo (NPU Reviewer), and Gustavo (Numerical Controls).
5. **Empirical Data Provenance:** All published benchmarks, convergence metrics, and research claims must be generated from version-controlled fixtures managed under `docs/project/research/` with documented random seeds.

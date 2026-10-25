# Proposal: NPU Specialization and Per-Network Assembly

**Status:** Proposed. The generator, the engine-selection flow, and the
elaboration interface described here are not implemented. Section 3 records
obligations that bind the v0.1.0 hardware steps so that the specialization
model is a bounded later extension rather than a redesign. The operative
v0.1.0 hardware plan remains
[`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md).

This proposal defines the specialization model of the TARS NPU — **generic
components, bespoke assembly**: universal, vendor-agnostic engine modules whose
sizing, selection, and interconnection are derived per network from the hardware
manifest, with the layer program as the only per-network runtime artifact.

---

## 1. Model and Degrees of Specialization

The NPU is neither a general-purpose accelerator nor a spatial unrolling of the
network. The engines are universal RTL components, reusable in any third-party
project; the assembly is the per-network specialization. Per-network silicon
cost is data — the layer program and the memory streams — never structure.

| Degree | Resolves | Mechanism | v0.1.0 state |
|---|---|---|---|
| Elaboration-time sizing | Datapath width (computation lanes), bank depths, engine count | Parameters set from the manifest at build | The lane count is a named parameter (value one); banks sized for the MNIST shapes |
| Structural selection | Which engine modules exist: dense ternary (`mode = 1`), zero-skipping (`mode = 2`), int8 (`mode = 3`), future convolution and CfC engines | Engine modules instantiated per manifest; the layer-descriptor `mode` field is the dispatch seed | Mode 1 ships; modes 2 and 3 are reserved in the descriptor |
| Runtime programming | Execution of the layer sequence: shapes, scales, streams | The layer program (`program.mem`) read by the top sequencer | The v0.1.0 release mechanism (LOG-007, Section 4.2, Steps 2 through 5) |

The input of a future generator is the hardware manifest — the directed acyclic
graph intermediate representation required by
[`V0_CONTRACT.md`](../project/V0_CONTRACT.md), Section 5.3. The specialization
chain is therefore: trained network, manifest extraction, sizing analysis,
elaboration defines plus packed program and memories, synthesizable top. The
generator is not built in the current phase: a generator with a single engine
type abstracts from one data point, under the same discipline that defers the
device axis ([`FUTURE_TARGETS.md`](FUTURE_TARGETS.md), Section 4).

---

## 2. Prior-Art Positioning

The prior art brackets the model on a specialization spectrum.

| System | Specialization mechanism | Cost of the model | Transferable finding |
|---|---|---|---|
| FINN ([project page](https://xilinx.github.io/finn/)) | Generation-time spatial dataflow: one streaming engine per layer, weights baked at synthesis — "a physical representation of the implemented custom network in space... not a generic DNN acceleration solution" | Sum across all layers; build times from hours to days; measured average BRAM utilization of roughly 22% ([Fraser et al.](https://doi.org/10.1145/3029580.3029586)); routing congestion can strand up to half of the device | The spatial pole. FINN itself converged away from it: the 2025 FINN Loop and multi-layer-offload nodes reuse engines across iterations with weights streamed over AXI-MM ([finn sources](https://github.com/Xilinx/finn/blob/38c42bf95abbdad493db0b0f5053060074698908/src/finn/custom_op/fpgadataflow/matrixvectoractivation.py)) |
| FINN-R ([DOI](https://doi.org/10.1145/3242897), [arXiv](https://arxiv.org/abs/1809.04570)) | Names both endpoints: DF (spatial dataflow) versus MO (multilayer offload) | MO cost is the maximum across scheduled layers rather than the sum; MO is beneficial "when the minimal footprint of the fully unrolled DF architecture exceeds the target device capabilities" | The TARS engine — reused engines, weights in memory, sequenced execution — is the MO operating point, stated by FINN's own authors in 2018 |
| Gemmini ([paper](https://arxiv.org/abs/1911.09925), [ISA sources](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/GemminiISA.scala)) | Elaboration-time parameters plus a small runtime instruction set; a configuration instruction selects dataflow, strides, and scale per matmul | One array fixed at elaboration; utilization collapses when problem shapes misalign | The two-plane pattern: architecture at elaboration, execution by a tiny program — the TARS top sequencer in miniature |
| VTA ([paper](https://arxiv.org/abs/1807.04188), [repository](https://github.com/apache/tvm-vta)) | Runtime micro-code: four task instructions with hardware dependency counters; micro-ops execute in two-level nested affine loops ([configurable-stack paper](https://arxiv.org/abs/2111.15024)) | One fixed tile shape bounds all layers; the whole TVM compiler stack is a co-requisite; extracted from TVM in 2024 and dormant since | Program words carrying (base, stride, iteration) triples amortize the program to the layer count, not the network size |
| NVDLA ([primer](https://nvdla.org/primer.html), [scalability](https://nvdla.org/hw/v2/scalability.html)) | A spec file through a preprocessor yields defines for both elaborated RTL and software headers ([full-configuration spec](https://github.com/nvdla/hw/blob/8e06b1b9d85aab65b40d43d08eec5ea4681ff715/spec/defs/nv_full.spec)); per-layer register programming; a ConfigROM exposes the build configuration | Utilization cliffs on shape misalignment; the configuration surface "has only been tested for the nv_small configuration" | One manifest artifact must feed the elaboration defines and the program packer; the configuration surface equals the verification surface |
| Ternary MACs ([ISPASS 2026 DSE](https://arxiv.org/abs/2604.25183), [TeLLMe](https://arxiv.org/abs/2504.16266), [TernaryCore RTL](https://github.com/Ternarycore/ternarycore/blob/c05b562586cb966cc31f3ee82ed0f9e0dcd0e6d2/rtl/ternary_mac.v)) | Conditional add/sub (grouping factor one) versus LUT grouping of activation combinations | For integer activations, the conditional add/sub is measured near-optimal; LUT grouping pays only for wide (FP16) activations, where it reports a 2.23-fold area advantage; DSP blocks are "highly inefficient" for 1.58-bit computation | The shipped MAC primitive is the validated point. The crossover evidence is INT8-based and must be re-derived against the 32-bit Q8.24 adder cost before any redesign |
| Sparse weight streams ([EIE](https://arxiv.org/abs/1602.01528), [retrospective](https://arxiv.org/abs/2306.09552), [Bit-Pragmatic](https://arxiv.org/abs/1610.06920)) | Encoded skip distance — a four-bit relative index — carried in the weight stream | Skipped cycles and skipped bandwidth, not only adder toggling; profitable in dedicated hardware up to roughly 50% density | Energy is won in the encoding; runtime zero detection wins almost nothing. The LOG-006 five-bit token (sign plus four-bit distance) is the same construction |

The differentiator of TARS is the combination over the substrate: the FINN-R
maximum-cost model, the NVDLA manifest-to-defines flow, the Gemmini
configuration-instruction economy, and the VTA loop-amortized program, on
hand-written SystemVerilog — measured at a tenfold synthesis-time advantage
over the equivalent HLS flow ([RTL FINN backend](https://arxiv.org/abs/2201.11409))
— keeping per-network elaboration in the minutes range. The defensible claim is
the substrate and the co-design (bit-exactness with the Rust golden reference),
not the concept.

---

## 3. v0.1.0 Obligations

These bind the current phase so that the generator phase is mechanical.

1. **The layer program is the only per-network artifact.** Dimensions, base
   addresses, scales, and modes are program data, never silicon.
2. **Program-word economy.** Per-layer descriptors carry (base, stride,
   iteration-count) triples for the weight, activation, and accumulator streams,
   with per-layer configuration fields — dimensions, scale, mode — in the manner
   of the Gemmini configuration instruction and the VTA affine loops. The
   program memory scales with the layer count, not with the network size.
3. **Engine modularity and memory discipline.** `layer_engine.sv` is
   self-contained; the ternary processing element is an isolated module;
   weight and activation memories are inference-friendly primitives with
   synchronous read and no reset of the stored array; a decode stage decouples
   the weight packing format from the accumulate path. Rules R1 and R2 of
   LOG-007 hold: a linear weight-stream pointer and a single
   activation-index update point.
4. **MAC exactness.** Negation is performed at N+1 bits, because the
   two's-complement negation of the most-negative N-bit value wraps within N
   bits. The conditional add/sub is the shipped primitive; the LUT-grouping
   alternative is re-derived against a 32-bit adder cost model before any
   redesign is considered.
5. **Runtime-elaboration separation.** The runtime shape registers of Step 2
   and the sizing parameters of the future generator are distinct, named
   surfaces; a generator sets the latter without touching the former.
6. **Open decision — sparse stream entry** (owner: the layer-manifest format
   contract, LOG-007 Step 1). Sparse runs enter either as the `mode = 2`
   five-bit token stream of
   [`LOG-006`](../project/research/LOG-006-ternary_zero_skipping_encoding.md)
   (the recorded default) or as a skip-run reserved code within the `mode = 1`
   dense stream (the `2'b10` word is reserved). The decision rides with the
   manifest contract. The principle is encoding-first, detection-never.
7. **Recorded, not implemented.** One manifest artifact feeds the elaboration
   defines and the program packer (the NVDLA `project.h`/`project.vh` pattern),
   and a ConfigROM-style readable register exposes the build configuration to
   software.

---

## 4. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| The configuration surface equals the verification surface | NVDLA: only the nv_small configuration is tested to the stated quality | One specialization axis per release, each gated by golden vectors |
| A programmable accelerator dies with its software stack | VTA: extracted from TVM in 2024, dormant since | The Rust bit-exact reference is the stack; the CPU parity path keeps the hardware optional |
| Parameterized engines suffer utilization cliffs | Gemmini and NVDLA shape-misalignment losses | Per-network sizing is the generator's purpose; the runtime shape registers already decouple shapes from silicon |
| Drift toward the spatial pole reintroduces the dataflow walls | FINN: sum-of-layers cost, hours-to-days builds, roughly 22% BRAM utilization | The maximum-cost (MO) model is the recorded architecture; this proposal pins it |
| LUT grouping adopted on the wrong substrate | The grouping crossover evidence is INT8-based | Re-derive with the 32-bit Q8.24 adder cost model first |

---

## 5. Boundary with Other Records

| Record | Relationship |
|---|---|
| [`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md) | The operative v0.1.0 ladder; the obligations of Section 3 ride with its Steps 2 through 5 |
| [`V0_CONTRACT.md`](../project/V0_CONTRACT.md), Sections 2.3 and 5 | Release-engine scope and the manifest IR invariant |
| [`FUTURE_TARGETS.md`](FUTURE_TARGETS.md) | Invariance rules and phase placement; the generator phase extends the v1.0+ expansion |
| [`LOG-006`](../project/research/LOG-006-ternary_zero_skipping_encoding.md) | The `mode = 2` token encoding; the open stream-entry decision of Section 3 |

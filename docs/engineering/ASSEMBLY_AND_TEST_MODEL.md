# NPU Assembly and Test Model

**Status:** Proposal. The three-layer assembly model, the engine contract, and the
three-tier verification structure recorded here are the maintainer's target
organization. Nothing beyond the cited current-state rows is implemented. This
document binds the v0.1.0 hardware steps of
[`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md) so that
per-network specialization extends the reference design instead of forking it.

---

## 1. Three-Layer Assembly Model

A network-specific NPU is data plus a minimal assembly top. Hardware changes only
when a network needs a new kind of computation (a new engine) or different sizing
(elaboration parameters) — never because the network is "a different network".

| Layer | Content | Location (current state) | Change policy |
|---|---|---|---|
| 0 — Primitives | `npu_pkg.sv`, `t3_dec.sv`, `pe.sv`, `epilogue.sv` | `npu/src/`, committed | Frozen contracts; change only with a golden-vector regression |
| 1 — Engines | `layer_engine.sv` (dense ternary, `mode = 1`); future `sparse_frontend` (`mode = 2`), convolution or im2col engines | `npu/src/`, committed | One engine per computation type; every engine obeys the contract of Section 2 |
| 2 — Per-network top | `npu.sv` (manual two-engine wrapper, committed); sequencer, program memory, activation banks, memory interface (not implemented, LOG-007 Steps 3–4); board glue (`npu/synthesis/`, git-ignored) | Wrapper: `npu/src/npu.sv`; sequencer: pending; board lab: git-ignored | Per network: program data and elaboration parameters only |

The three layers are the operational form of the three specialization degrees of
[`NPU_SPECIALIZATION_PROPOSAL.md`](NPU_SPECIALIZATION_PROPOSAL.md), Section 1:
elaboration-time sizing (Layer 2 parameters), structural selection (the layer
descriptor `mode` field dispatching to Layer 1 engines), and runtime programming
(`program.mem`, Layer 2 data).

---

## 2. Engine Contract

Every Layer 1 engine exposes the same boundary; an engine that does not is a
separate project, not a block.

| Interface group | Contents | Source of truth |
|---|---|---|
| Configuration | Input/output dimensions, per-layer scale `w0`, activation selector, `mode` | Layer descriptor fields (four-word descriptor, Section 7.1); runtime shape registers are LOG-007 Step 2 |
| Streams | Packed weight codes, activations, bias | Memory artifacts (`wt.mem`, `w0.mem`, `bias.mem`) delivered by the top memory interface |
| Handshake | `start` / `done` with the sequencer | Single protocol across engines; `done` is a one-cycle registered pulse, `start` is sampled only in `IDLE` (contract recorded in [`ARCHITECTURE.md`](ARCHITECTURE.md), *Handshake contract*); `layer_engine.sv` implements the elaboration-parameter form today |
| Internals | Instances of Layer 0 primitives | An engine never reimplements decode, accumulate, or epilogue arithmetic |

`npu_pkg.sv` is the shared type book (`activation_t`, `state_t`, the Q8.24
saturation constants). New engine types extend this package; they do not redefine
its contents locally.

### 2.1 Zero-Skip as a Frontend, Not a Rewrite

The `mode = 2` zero-skipping stream (five-bit tokens, `LOG-006`) enters as a
separate `sparse_frontend` block that decodes `(skip, sign)` tokens into
activation-index advances and signed enable pulses over the same `pe` and
`epilogue` instances. The dense decoder (`t3_dec.sv`) and the fault-tolerant
dense encoding are unaffected: stream entry for sparse runs is exclusively the
`mode = 2` token stream (NPU_SPECIALIZATION_PROPOSAL, Section 3, obligation 6).
The remaining open question is memory organization, recorded in Section 7.2.

---

## 3. Per-Network Assembly Recipe

| Step | Decision | Code effect |
|---|---|---|
| 1 | Which engines the network needs | Instantiate the corresponding Layer 1 engines in the top |
| 2 | Sizing | Elaboration parameters: lane count, maximum dimensions, bank depths |
| 3 | Top composition | One thin file following the reference template (debounce, sequencer, banks, engines, display) |
| 4 | Program | `program.mem`, `wt.mem`, `w0.mem` produced by the exporter — data only |
| 5 | Tests | Tier ① is untouched; tier ② gains the network's golden vectors; tier ③ gains the program |

A reference MLP with different dimensions and a different program is the same
hardware plus data — zero new RTL. Automatic generation of the top from the
hardware manifest is deferred (NPU_SPECIALIZATION_PROPOSAL, Section 1); assembly
in the current phase is manual composition of the reference blocks.

---

## 4. Top Placement and Version Control

Current state: the board lab under `npu/synthesis/` is git-ignored
([`BOARD_INTEGRATION.md`](BOARD_INTEGRATION.md), Section 1); the versioned tree
contains only the generic core and its simulation flow. The manual wrapper
`npu.sv` (two chained engines) exists; the programmable multi-layer top of
LOG-007 Steps 3–4 (sequencer, program memory, banks) does not exist yet.

Target policy recorded by this proposal:

| Artifact | Placement | Rationale |
|---|---|---|
| Generic top (sequencer, `prog_mem`, activation banks, `mem_if`) | Versioned, with the core | Vendor-agnostic synthesizable RTL; it is the assembly template |
| Reference network example (program data, parameters) | Versioned, as data | Demonstrates the export/program contract |
| Board wrappers, constraints, Vivado scripts | Git-ignored lab | Board-specific; unchanged policy from BOARD_INTEGRATION |

The move of the generic top from the lab into the versioned tree is recorded as a
pending decision in Section 7.4.

---

## 5. Three-Tier Verification Model

Each tier is anchored to one assembly layer and has a distinct golden source and
change trigger.

| Tier | Scope | Golden source | Changes when | Gate |
|---|---|---|---|---|
| ① `tb_core` | Layer 0 primitives and a Layer 1 engine in isolation | Hand-computed reference values embedded in the testbench (spec-as-code) | Only on an RTL contract change — never per network | Every revision |
| ② `tb_parity` | Layer 1 engine against the Rust golden model | `exp.mem`-class vectors produced by the exporter (Rust bit-exact path) | Gains vectors when a network is added | Release parity gate (LOG-007 Steps 3–5) |
| ③ `tb_shell` | Layer 2 top: sequencer, memories, I/O | The same golden vectors, observed at the system level, plus the network's `program.mem` | Gains the program of each network | Pre-bitstream gate |

Current state: a single testbench (`npu/tests/tb.sv`) drives `layer_engine`
with `IN = 2`, `OUT = 1`, loads the checked-in legacy `.mem` files, prints
results, and dumps waveforms. It has no assertions and no golden comparison
(STATUS.md, Known Limitation 1). The exporter does not yet emit ternary
artifacts (LOG-007 Step 1), so tier ② has no input. The tiers above are the
target structure for the steps that close these gaps.

System-level behaviors belong to tier ③ only: button debounce, synchronous
memory read latency, ping-pong bank flips between layers, the fetch–run–commit
flow, and reset during execution.

---

## 6. Make Target Surface

| Target | Tier | State |
|---|---|---|
| `test` (alias of `sim`) | ① | Implemented; propagates the testbench exit code once assertions exist |
| `lint` | ① | Implemented; `verilator --lint-only -Wall --top-module npu`, exits zero; covers wrapper, engines, and primitives |
| `vectors` | ② input | Not implemented; requires the Step 1 exporter |
| `test-parity` | ② | Not implemented |
| `sim-shell` | ③ | Not implemented; requires the Step 3–4 top |
| `bit` | board | Exists in the local lab (`make synt-npu` / `flash-npu`, git-ignored) |
| `test-all` | ① + ② + ③ | Not implemented |

---

## 7. Pending Decision Briefs

### 7.1 Four-Word Layer Descriptor Layout

LOG-007 Step 1 requires a four-word layer descriptor documented in
[`ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md) and
[`MEM_FORMAT.md`](MEM_FORMAT.md). The open choice is the word width and field
packing: four 32-bit words versus four 64-bit words, and the bit assignment of
dimensions, base addresses, scale, and mode. This decision fixes the program
memory width and the descriptor type consumed by the sequencer; it gates the
Step 1 exporter contract.

### 7.2 `mode = 2` Memory Organization

Stream entry is closed (obligation 6): the dense `mode = 1` stream uses all four
2-bit codes and never carries skip tokens; the sparse stream is the five-bit
token format of LOG-006. The open question is physical organization: memories
sized per mode versus a dedicated token memory, and how the `sparse_frontend`
of Section 2.1 attaches without changing the dense decode path. The conflict
between the dense active flag and the token sign bit on bit 0 is resolved by
never sharing a stream, not by extending either encoding.

### 7.3 Golden-Vector Integrity

The checked-in `npu/data/` files are from an earlier export and are not
regenerated by training (MEM_FORMAT, Section "Size and Configuration"; STATUS.md
limitation 5). A stale `.mem` that no longer matches its generating model fails
as a silent parity mismatch. A per-case checksum or manifest digest recorded
next to the vectors would fail fast on stale artifacts. Scope: the tier ②
protocol, decided together with the Step 1 exporter.

### 7.4 Generic Top Placement

Section 4 records the target: the generic multi-layer top joins the versioned
tree when LOG-007 Steps 3–4 land; the board lab stays ignored. Confirming this
split (and the directory layout of the versioned top and its reference example)
is a maintainer decision before Step 3 starts.

### 7.5 Automatic Generation Scope

Manifest-to-top automatic generation is out of the current phase
(NPU_SPECIALIZATION_PROPOSAL, Section 1: a generator with a single engine type
abstracts from one data point). This brief records that assembly in v0.1.0 is
manual by decision, not by omission, and that the generator question reopens
when a second engine type exists.

---

## 8. Boundary with Other Records

| Record | Relationship |
|---|---|
| [`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md) | Operative implementation ladder; the tiers and recipes above attach to its Steps 1–5 |
| [`NPU_SPECIALIZATION_PROPOSAL.md`](NPU_SPECIALIZATION_PROPOSAL.md) | Specialization degrees and generator scope; this document is their assembly and verification operationalization |
| [`V0_CONTRACT.md`](../project/V0_CONTRACT.md) | Release gates the tiers must satisfy (bit-exact parity, self-checking testbench) |
| [`MEM_FORMAT.md`](MEM_FORMAT.md), [`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md) | Artifact and numeric contracts consumed by tiers ② and ③; Section 7.1 extends them |
| [`BOARD_INTEGRATION.md`](BOARD_INTEGRATION.md) | Board-lab boundary; Section 4 preserves it and scopes the versioned-top exception |

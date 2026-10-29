# Research Log LOG-007: v0 Execution-Mode Phasing and Implementation Sequence

**Document type:** Engineering research log under the protocol defined in
[`V0_CONTRACT.md`](../V0_CONTRACT.md), Section 4. Companion to
[`LOG-006-ternary_zero_skipping_encoding.md`](LOG-006-ternary_zero_skipping_encoding.md).

**State:** Accepted scope decision and release plan (maintainer position).
The single-layer ternary core (decode, processing element, epilogue, layer
engine) is implemented; the multi-layer scope below remains gated by
Section 4, and all quantitative figures are projections pending those gates.

This log records the v0.1.0 hardware scope — multi-layer layer engine,
ping-pong activation banks, execution modes 0 and 1, with mode 2 reserved in
the layer descriptor for v0.2.0 — and defines the four-step implementation
sequence with acceptance gates that isolate each new failure surface.

---

## 1. Title and Context

**Decision.** Release v0.1.0 ships the complete multi-layer engine with
descriptor `mode = 0` (dense Q8.24) and `mode = 1` (dense ternary, 2-bit).
`mode = 2` (zero-skipping, specified in
[LOG-006](LOG-006-ternary_zero_skipping_encoding.md)) is reserved in the
descriptor and implemented in v0.2.0.

**Problem statement.** Migrating the NPU from one fixed layer to a
multi-layer ternary engine introduces three new failure surfaces at once:

1. Activation-bank switching (ping-pong flip between layers).
2. Ternary epilogue arithmetic ($W_0$ scaling order, truncation,
   saturation).
3. Weight and activation addressing under runtime layer shapes.

If zero-skipping enters in the same step, a mismatching bit in a layer-2
output cannot be attributed to a specific surface: bank flip, epilogue
truncation, and skip arithmetic become indistinguishable causes. The phasing
below isolates each surface behind a verifiable gate.

---

## 2. Theoretical Formulation

### 2.1 Failure-Surface Isolation

Dense ternary addressing (`mode = 1`) is identical to Q8.24 addressing: the
activation and neuron counters traverse exactly the same values in both
modes; only the weight word width and the accumulate operation differ.
Shipping `mode = 1` in v0.1.0 therefore validates surfaces 1 and 2 on a
well-understood addressing substrate, and leaves zero-skipping as the single
new variable of v0.2.0.

### 2.2 Ternary Epilogue Contract (Maintainer Sign-Off; Remaining Owners Pending)

The per-neuron epilogue for ternary modes. Normative formula:

$$z = \mathrm{sat}_{32}\big( (\mathrm{acc} \cdot W_0) \ggg 24 \big), \qquad
\mathrm{out} = \mathrm{act}\big( \mathrm{sat}_{32}(z + b_j) \big)$$

Normative properties, by order of evaluation:

1. **96-bit product.** $\mathrm{acc}$ is a 64-bit signed accumulator and $W_0$
   is a 32-bit signed Q8.24 word; the product is evaluated at 96 bits and
   never wraps. The $Q16.48$ framing of
   [`ARITHMETIC_Q8_24.md`](../../engineering/specs/ARITHMETIC_Q8_24.md),
   Section 3.1, covers only 32-by-32-bit operands and does not describe this
   product.
2. **Arithmetic shift of 24 (floor).** $\ggg$ denotes the arithmetic right
   shift, which floors toward negative infinity. This truncation is
   normative; the Rust golden model must reproduce it with an arithmetic
   shift, not with a cast that truncates toward zero.
3. **Two saturations, fixed order.** One saturation clamps the scaled product
   to 32 bits before the bias is added; the second clamps the biased sum.
   $W_0$ is applied before the bias — the bias lives in the scaled domain.
4. **$W_0$ per layer (scalar).** v0.1.0 stores one Q8.24 $W_0$ value per
   layer ($w0.mem$). A per-channel $W_0[j]$ is a compatible future extension
   that does not change the hardware structure.
5. **Multiplier outside the array.** The $W_0$ multiplier lives in the
   epilogue, one per epilogue, not in the processing elements; the zero-DSP
   obligation of the PE is unaffected.

**Sign-off status.** The maintainer (Arthur) confirmed this contract,
including the 96-bit product and the floor-shift correction, in October 2026.
Sign-off by the remaining intersection owners required by Engineering
Invariant 4 of the [roadmap](../ROADMAP.md) is pending. Per that invariant,
any revision to this formula requires simultaneous updates to
[`ARITHMETIC_Q8_24.md`](../../engineering/specs/ARITHMETIC_Q8_24.md),
`src/export.rs`, and the NPU RTL in one revision.

### 2.3 Dense Ternary Weight Encoding (2 Bits)

**Amended, October 2026 (maintainer decision).** Standard for `mode = 1`
weight words: the weight is the difference of the two bits,
$v = b_0 - b_1$. All four codes are valid; there is no illegal state.

| Code ($b_1 b_0$) | $v = b_0 - b_1$ | Weight |
|---|---|---|
| `2'b00` | $0 - 0$ | $0$ |
| `2'b01` | $1 - 0$ | $+1$ |
| `2'b10` | $0 - 1$ | $-1$ |
| `2'b11` | $1 - 1$ | $0$ (redundant zero, self-cancelling) |

Rationale:

- $\pm 1$ codes sit at Hamming distance 2, so a single-bit flip can never
  flip the sign of a weight.
- An all-ones word (erased flash, `0xFF`) decodes as all-zero — fail-safe.
- Negation is the bitwise complement of the code.
- The safe zero `2'b11` of the original Tars contract is preserved.

This encoding supersedes the previous table of this section
(`2'b01` $=-1$, `2'b11` $=+1$, `2'b10` reserved). The reserved code
`2'b10` no longer exists in the dense stream: the sparse mode therefore
enters exclusively as the 5-bit token stream of
[LOG-006](LOG-006-ternary_zero_skipping_encoding.md) under `mode = 2`,
which closes the open stream-entry decision of
[`NPU_SPECIALIZATION_PROPOSAL.md`](../../engineering/NPU_SPECIALIZATION_PROPOSAL.md),
Section 3, obligation 6.

The dense encoding is a per-mode stream format, not bit-compatible with the
5-bit token scheme of LOG-006, whose bit 0 is the sign and whose bits 4:1
are the skip distance. The two streams are never decoded
interchangeably. The Rust ternary encoder in `src/export.rs` does not exist
yet and must be born with this encoding (implementation Step 1); the RTL
decode is `src/t3_dec.sv`.

### 2.4 Design Rules for v0.2.0 Compatibility

Two rules applied while building the v0.1.0 engine confine the future sparse
mode to a localized change:

- **R1 — Linear weight-stream pointer.** The layer engine must not compute
  weight addresses as $j \cdot n_{in} + i$. A pointer register initialized
  at the layer weight base and advanced by one per consumed word serves dense
  and sparse modes alike, since both streams are strictly sequential.
- **R2 — Isolated activation-index increment.** The activation-index update
  must remain a single identified update point: dense modes advance
  $i \leftarrow i + 1$; `mode = 2` will substitute
  $i \leftarrow i + 1 + \mathrm{skip}$ at that point alone. All other engine
  state is mode-invariant.

---

## 3. Empirical Benchmarks

The figures below are projections; no measurement has been attached to this
log yet. Benchmark network: MNIST MLP
$784 \to 128 \to 10$. Reproducible commands, seeds, and measurements attach to
this log at the gate that produces them.

| Metric | Projection | Producing gate |
|---|---|---|
| Weight storage, dense Q8.24 | ~3.25 Mb (~400 KiB) — exceeds the XC7S50 total BRAM (1.8 Mb) | Step 3 comparison record |
| Weight storage, dense ternary | ~203 kb (~24.8 KiB) — about 11% of device BRAM | Step 4 memory accounting |
| Inference latency, modes 0 and 1 | ~102,000 cycles per inference at one product per cycle | Step 3 and Step 4 cycle counters |
| Inference latency, mode 2 (v0.2.0 projection) | ~12,300 cycles at 90% zero density ([LOG-006](LOG-006-ternary_zero_skipping_encoding.md), Section 3) | out of v0.1.0 scope |

Failure-surface attribution — the empirical purpose of the phasing:

| Failure surface | Introduced by | Verified by gate |
|---|---|---|
| Addressing under runtime shapes | Step 2 | Zero lint warnings and identical legacy single-layer results |
| Bank switching (ping-pong) | Step 3 | Two-layer `mode = 0` end-to-end match against the Rust model |
| Epilogue arithmetic ($W_0$) | Step 4 | Ternary MNIST bit-exact against the Rust golden model |

---

## 4. Conclusion and Architectural Implication

### 4.1 Decision

v0.1.0 is tagged when the Step 4 gate passes. Dense ternary already delivers
the board-enabling memory footprint (about 11% of the Spartan-7 BRAM), so
v0.1.0 does not depend on zero-skipping; `mode = 2` is a performance
optimization admitted later under the compatibility rules of Section 2.4.

### 4.2 Phased Implementation Plan

Each step is a self-contained, reviewable revision; a step starts only after
the previous gate passes.

| Step | Scope | Acceptance gate |
|---|---|---|
| 1. Contract and golden model (Rust and docs) | Epilogue sign-off (Section 2.2); four-word layer descriptor and mode semantics documented in [`ARITHMETIC_Q8_24.md`](../../engineering/specs/ARITHMETIC_Q8_24.md) and [`MEM_FORMAT.md`](../../engineering/MEM_FORMAT.md); `src/export.rs` emits `program.mem`, `wt.mem`, `w0.mem` | A small test network (for example $4 \to 3 \to 2$) exports with bit-exact expected vectors |
| 2. Runtime signals and engine extraction | Elaboration parameters `IN`/`OUT` replaced by 16-bit runtime registers; product-form weight addressing replaced by the linear pointer (R1); compute loop migrated from `npu.sv` into `layer_engine.sv` | Zero `WIDTHTRUNC`/`WIDTHEXPAND` warnings under Verilator lint; the legacy single-layer testbench reproduces identical numerical results |
| 3. Top sequencer and ping-pong | `prog_mem`; activation memory as two banks; top FSM (idle, fetch, run, commit) with per-layer bank inversion | A two-layer `mode = 0` network executes end to end and matches the Rust model exactly |
| 4. Dense ternary mode and release | Ternary weight memory and per-layer $W_0$ memory; conditional accumulation selected by the active and sign bits; epilogue applies $W_0$ scaling before the bias (Section 2.2) | MNIST $784 \to 128 \to 10$ executes in `mode = 1` bit-exact against the Rust model; tag v0.1.0 |

Step 2 has no dependency on the epilogue sign-off and may proceed
immediately: it touches only counter widths and module boundaries while
preserving the current Q8.24 arithmetic.

### 4.3 v0.1.0 Design Obligations

The layer engine delivered by Step 2 must implement rules R1 and R2
(Section 2.4). These obligations are what confine v0.2.0 `mode = 2`
insertion to the token decoder and the activation-index update, per
[LOG-006](LOG-006-ternary_zero_skipping_encoding.md), Section 2.4. The mode
field must be present in the v0.1.0 descriptor with `mode = 2` reserved;
retrofitting encodings later costs descriptor format revisions.

Normative document updates ride with their steps and are maintainer-owned:
the research index in [`README.md`](README.md), the arithmetic contract and
memory format at Step 1, and `npu/README.md` after the RTL changes.

---

## 5. Revision (October 2026): Ternary-Only Release Engine

**Superseding maintainer decision.** The release NPU implements `mode = 1` (dense
ternary) only. The dense Q8.24 weight mode is not a shipped mode: the single-layer
Q8.24 prototype was replaced in the source tree by the ternary engine (October
2026) and is retained in repository history as the development substrate and as
the synthesis baseline for resource and energy comparisons. The int8 variant
(`mode = 3`) is reserved in the descriptor and postponed beyond v0.1.0. The
accuracy release gate is $> 90\%$ for the ternary-quantized model on the bit-exact
path; the $\ge 95\%$ figure remains the target of the f32 software reference. The
sections above record the superseded dual-mode plan and retain the failure-surface
isolation argument, which the revised ladder below preserves.

Consequences for the implementation ladder of Section 4.2:

| Step | Revised scope | Acceptance gate |
|---|---|---|
| 1 | Contract and golden model: epilogue sign-off, four-word layer descriptor, export of `program.mem`, `wt.mem`, `w0.mem` | Unchanged |
| 2 | Runtime signals and engine extraction over the existing Q8.24 arithmetic | Unchanged: legacy single-layer results identical, zero width warnings |
| 3 | MAC swap to conditional accumulation; single-layer ternary validated on a small network | Small ternary network bit-exact against the Rust golden model |
| 4 | Top sequencer and ping-pong activation banks | Two-layer ternary network bit-exact against the Rust golden model |
| 5 | MNIST release run | MNIST $784 \to 128 \to 10$ executes in `mode = 1` bit-exact against the Rust model; tag v0.1.0 |

Step 3 preserves the failure-surface isolation of Section 1: the epilogue and the
conditional accumulator are first observed on a network small enough to verify by
hand, before the bank-switching surface of Step 4 is admitted. Dense Q8.24 and
dense ternary share addressing and cycle counts (Section 2.1), so the release
claims of dense ternary inference are memory footprint and per-operation energy,
not latency; the latency claim belongs to `mode = 2` in v0.2.0.

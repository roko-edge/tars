# Specification: Q8.24 Fixed-Point Arithmetic and Bit-Exact Parity

**Status:** Target contract with current deviations identified in Sections 2.2 and
3.4. It does not claim that automated parity verification is implemented. The
release NPU executes dense ternary weights over the Q8.24 activation substrate;
the ternary arithmetic and epilogue are specified in
[`LOG-007`](../../project/research/LOG-007-v0_execution_mode_phasing.md), Section
2.2 (maintainer sign-off recorded, October 2026; sign-off by the remaining
intersection owners pending).

This document specifies the shared numeric contract between the Rust training runtime
and the SystemVerilog NPU. Compliance is mandatory for bit-exact hardware parity.

> **V0 Scope:** The full v0.1.0 release requirements, including MNIST benchmark targets,
> ternary quantization, and research logging protocol, are defined in
> [`docs/project/V0_CONTRACT.md`](../../project/V0_CONTRACT.md).

---

## 1. Numeric Format Definition

| Property | Value |
|---|---|
| Total width | 32 bits |
| Sign representation | Two's complement, signed |
| Integer bits (incl. sign) | 8 |
| Fractional bits | 24 |
| Value interpretation | $v = \frac{w}{2^{24}}$ where $w$ is the signed 32-bit word |

Valid representable range: $[-128.0, +127.999999940\ldots]$.

---

## 2. Conversion Contract (Rust, `src/export.rs`)

### 2.1 Encoding (float to fixed)

The canonical encoding function in the Rust runtime must satisfy:

$$w = \text{clip}_{\text{int32}} \left( \text{round}(x \cdot 2^{24}) \right)$$

where:

- `round` is round-half-to-even (ties to even), matching IEEE-754 default rounding.
- `clip_int32` clamps the value to $[-2^{31}, 2^{31} - 1]$ to prevent silent overflow.

### 2.2 Current Deviation (Normative Warning)

The existing `to_q8_24(x: f32) -> u32` implementation scales by $2^{24}$ and
performs **truncation** through the Rust numeric cast.

The `as` cast truncates toward zero. Until both the exporter and the NPU arithmetic
specification agree on rounding mode, **truncation remains the normative behavior**
for v1 of this contract. Any change requires updating this specification, `src/export.rs`,
and the NPU RTL (`npu/src/epilogue.sv`, `npu/src/layer_engine.sv`) in the same
revision (Engineering Invariant 4 of the ROADMAP). The ternary encoder in
`src/export.rs` does not exist yet and must be born with the dense ternary encoding
of [`LOG-007`](../../project/research/LOG-007-v0_execution_mode_phasing.md),
Section 2.3 (implementation Step 1).

---

## 3. Arithmetic Contract (SystemVerilog, `npu/src/`)

### 3.1 Multiplication

The release datapath (`mode = 1`) evaluates a single multiplication, in the
epilogue:

$$p = \mathrm{acc}_{64} \cdot W_0$$

| Property | Value |
|---|---|
| Left operand | $\mathrm{acc}$, signed 64-bit accumulator |
| Right operand | $W_0$, signed 32-bit Q8.24 scale |
| Product width | $64 + 32 = 96$ bits, evaluated without wrapping |
| Reduction | arithmetic right shift by 24, then saturation to 32-bit Q8.24 (Section 3.3) |

A 64-bit intermediate is not sufficient for this product: the maximum
magnitude of $\mathrm{acc}$ over 784 inputs approaches $2^{40}$, and
multiplying by a full-scale $W_0$ word overflows 64 bits. The $Q16.48$
designation ($8 + 24 + 8 + 24 = 64$ bits) applies only to products of two
32-bit Q8.24 words and must not be used to size the epilogue product.

The dense product of two 32-bit Q8.24 words defined the superseded dense
processing element; the release engine performs conditional addition and
subtraction of activations instead, so no dense product occurs in the
datapath. Dense $Q16.48$ semantics remain the reference for any future dense
mode and for the historical development substrate.

### 3.2 Accumulation

The accumulator is signed 64-bit and sums the conditionally added or
subtracted Q8.24 activations:

$$\mathrm{acc}_{64} = \sum_{i=0}^{N-1} s_i \cdot a_i, \qquad s_i \in \{-1, 0, +1\}$$

For $N = 784$ and $|a_i| < 2^{31}$ the magnitude stays below $2^{41}$, within
the 64-bit range. The accumulator is not narrowed and **not saturated during
or after accumulation**: narrowing to Q8.24 happens only inside the epilogue,
after the $W_0$ product (Section 3.3), because the full 64-bit accumulator is
an operand of the 96-bit scale product.

### 3.3 Ternary Epilogue (Normative)

Per output neuron, with sign-off recorded in
[`LOG-007`](../../project/research/LOG-007-v0_execution_mode_phasing.md),
Section 2.2:

$$z = \mathrm{sat}_{32}\big( (\mathrm{acc} \cdot W_0) \ggg 24 \big), \qquad
\mathrm{out} = \mathrm{act}\big( \mathrm{sat}_{32}(z + b_j) \big)$$

where $\ggg$ denotes the arithmetic right shift. Normative properties, by
order of evaluation:

1. The $\mathrm{acc} \cdot W_0$ product is evaluated at 96 bits and never
   wraps (Section 3.1).
2. The reduction by 24 bits is an arithmetic right shift, which floors toward
   negative infinity. This floor behavior is normative; truncation toward
   zero (for example a numeric cast) is not equivalent for negative operands.
3. Two saturations are mandatory, in this order: one on the scaled product
   before the bias, one on the biased sum (evaluated at 64 bits). $W_0$ is
   applied before the bias — the bias lives in the scaled domain.
4. $W_0$ is one Q8.24 scalar per layer in v0.1.0 ($w0.mem$). A per-channel
   $W_0[j]$ is a compatible future extension that does not change the
   hardware structure.
5. The $W_0$ multiplier lives in the epilogue, one per epilogue, outside the
   processing-element array; the zero-DSP obligation of the PE is unaffected.
6. The activation $\mathrm{act}$ is applied after the second saturation;
   `SIGMOID` is reserved and not implemented in the release engine.

The implementation of record is `npu/src/epilogue.sv`, instantiated once per
layer engine by `npu/src/layer_engine.sv`.

### 3.4 Current Deviation (Normative Warning)

Two deviations from bit-exact parity remain:

1. **Floor reduction is not yet reproduced in Rust.** The epilogue reduces
   with an arithmetic right shift (floor toward negative infinity), while the
   existing `to_q8_24` conversion truncates toward zero. The Rust bit-exact
   golden model for the ternary path does not exist yet; when written it must
   replicate the floor reduction and both saturations exactly — a cast-based
   truncation diverges from the hardware for negative scaled products.
2. **Golden vectors and assertion are absent.** The exported `target.mem`
   contains dataset labels rather than golden Q8.24 inference outputs, and the
   testbench implements no assertion; the parity protocol of Section 4
   remains unimplemented. The ternary export artifacts (`program.mem`,
   `wt.mem`, `w0.mem`) do not exist yet either — they arrive with
   implementation Step 1 together with the golden model.

---

## 4. Parity Test Protocol

Bit-exact parity between Rust golden inference and NPU simulation is verified through
golden vector exchange:

```text
Rust Runtime                              NPU Simulation
────────────                              ──────────────
1. train model (seeded PRNG)
2. run inference on dataset
3. quantize activations & weights
   (to_q8_24, truncation)
4. write:
   ├── activations.mem                   ├── $readmemh(activations)
   ├── weights.mem                       ├── $readmemh(weights)
   ├── bias.mem                          ├── $readmemh(bias)
   └── target.mem    ─────────────────►  ├── compute (dot product + bias)
                                          └── assert result == expected
                                          (PASS/FAIL + $fatal on mismatch)
```

This protocol is not implemented by the current testbench. The target behavior is to
terminate with a non-zero exit status on any mismatch; the current `make test` target
is only an alias for simulation.

# Memory Files

**Audience:** Engineers working on the Rust-to-RTL boundary.

**Document type:** Current artifact-format reference. This is not a user guide for
the Rust library.

The NPU prototype uses hexadecimal text files accepted by SystemVerilog
`$readmemh`. Artifacts are generated from the Rust runtime (`src/export.rs`)
in Q8.24 fixed-point format with truncation.

The exporter and the testbench operate on the same files under `npu/data/`. The calls
to the exporter are commented out in the training binary, so the checked-in files
consumed by simulation remain from an earlier export and are not overwritten by a
training run. The shared encoding and paths do not constitute an automated integration
path.

## `activations.mem`

`npu/tests/tb.sv` loads input activations into an array for batch evaluation:

```systemverilog
$readmemh("data/activations.mem", data);
```

Each 32-bit hexadecimal word represents a Q8.24 fixed-point value ($1.0 \times 2^{24} = \text{0x01000000}$). For a 2-input network evaluating the 4 XOR/OR cases, the file contains 8 words (4 pairs of $x_1, x_2$):

```text
00000000
00000000
00000000
01000000
01000000
00000000
01000000
01000000
```

The NPU processes these entries as signed 32-bit Q8.24 words.

## `weights.mem` and `bias.mem`

`src/export.rs` exports trained weights and biases converted through
`to_q8_24(x: f32) -> u32` as eight-digit hexadecimal words.

The SystemVerilog testbench loads these files into testbench-level arrays that drive the DUT ports:

```systemverilog
$readmemh("data/bias.mem", b);
$readmemh("data/weights.mem", w);
```

Values are loaded directly as signed 32-bit Q8.24 integers (bias), while the
engine's weight port `w` is a 2-bit ternary code array decoded as
$v = b_0 - b_1$. The checked-in `weights.mem` still holds legacy dense Q8.24
words; reading it into the 2-bit array truncates each word to its two low
bits. The ternary weight export (`wt.mem`) arrives with implementation Step 1
of [`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md),
Section 4.2.

The engine accumulates the conditionally added or subtracted activations in a
64-bit accumulator. The epilogue then evaluates the 96-bit product of the
accumulator and the per-layer scale `w0`, reduces it with an arithmetic right
shift by 24 bits (floor), saturates to 32 bits, adds the per-output bias,
saturates again, and applies the activation
([`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md), Section 3.3).

## `target.mem`

`export_data()` also exports the dataset target labels in Q8.24 hex format (`0x00000000` or `0x01000000`). These are training labels, not golden inference outputs; automated bit-exact assertion against golden Q8.24 vectors remains a project target defined in [`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md).

## Size and Configuration

The testbench configures the layer engine with `IN = 2` and `OUT = 1`. The module supports sequential computation of `OUT` output neurons from `IN` input activations within a single affine layer. Multi-layer sequencing (top sequencer, `program.mem`, ping-pong banks) is scheduled as Steps 4 and 5 of [`LOG-007`](../project/research/LOG-007-v0_execution_mode_phasing.md), Section 4.2.

## Current Limitations

- **Layer Manifest:** Model topology, layer counts, and per-layer dimensions are not yet encapsulated in a formal manifest. Manifest extraction from `Sequential` is the v0.1.0 scope; the `Model<M>` container is deferred until a non-sequential topology exists.
- **Weight File Format:** `weights.mem` holds legacy dense Q8.24 words while the engine consumes 2-bit ternary codes; the ternary export (`wt.mem`) arrives with implementation Step 1.
- **Rounding Mode:** Rust export truncates toward zero while the epilogue reduces with an arithmetic right shift (floor); the golden model must replicate the floor semantics. Neither side uses round-half-to-even ([`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md), Section 3.4).
- **Automated Assertions:** `npu/tests/tb.sv` logs outputs and dumps waveforms but does not yet assert bit-exact equality against golden vectors. `make test` propagates the testbench exit code, so the target fails automatically once `$fatal` assertions exist.

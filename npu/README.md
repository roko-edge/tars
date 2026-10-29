# SystemVerilog NPU Prototype

This directory contains the ternary neural processing unit (NPU) RTL of the
v0.1.0 release engine and its testbench. The toolchain is Verilator-only; Icarus
Verilog is not used by any target in this directory.

## Files

| File / Directory | Purpose |
|---|---|
| `src/npu_pkg.sv` | Shared package: `activation_t` (`IDENTITY`, `RELU`; `SIGMOID` reserved), `state_t` (`IDLE`, `RUN`, `WRITE`), and the `Q8_24_MAX` / `Q8_24_MIN` constants |
| `src/t3_dec.sv` | Ternary weight decode: $v = b_0 - b_1$ over the 2-bit code (`w_en = w[1] ^ w[0]`, `w_s = w[1]`) |
| `src/pe.sv` | Ternary processing element: conditional add/subtract accumulate into a signed 64-bit accumulator, negation at 33 bits before extension |
| `src/epilogue.sv` | Per-neuron post-processing: 96-bit $W_0$ product, arithmetic shift by 24, two saturations, bias, activation |
| `src/layer_engine.sv` | Layer engine (successor of the former `npu.sv`): `IDLE`/`RUN`/`WRITE` FSM driving one `pe` and one `epilogue`, with the linear weight-stream pointer (rule R1) |
| `tests/tb.sv` | Testbench: clock, reset, `$readmemh` of the data files, waveform dump, result display. Self-checking assertions are pending |
| `data/activations.mem` | Q8.24 hexadecimal activation words (8 entries for 4 evaluation pairs) |
| `data/weights.mem` | Legacy dense Q8.24 hexadecimal weight words. **Format mismatch:** the engine consumes 2-bit ternary codes; see Current Limitations |
| `data/bias.mem` | Q8.24 hexadecimal bias word |
| `data/target.mem` | Q8.24 hexadecimal target outputs (dataset labels, not golden vectors) |
| `files.f` | Single source list of RTL files, shared by the Makefile and the LSP |
| `Makefile` | Simulation (`sim`), test (`test`), waveform inspection (`wave`), linting (`lint`), and cleanup |

## Interface and Arithmetic

The testbench instantiates `layer_engine` with parameters `IN = 2` and
`OUT = 1`. Ports are explicit at the top level: activation array `a`, 2-bit
ternary weight array `w`, bias array `b`, the per-layer scale `w0` (Q8.24), the
`act_fn` selector, and the `result` / `done` handshake.

The engine walks each output neuron sequentially: the `pe` accumulates the
signed activations selected by the decode of `t3_dec` into a 64-bit
accumulator; on the `WRITE` state the `epilogue` computes, per the normative
contract of [`LOG-007`](../docs/project/research/LOG-007-v0_execution_mode_phasing.md),
Section 2.2:

```text
z   = sat32((acc · w0) >>> 24)   // 96-bit product, arithmetic shift (floor)
out = act(sat32(z + b[j]))        // two saturations, w0 before bias
```

The full numeric contract, including the floor-shift and saturation
obligations, is specified in
[`ARITHMETIC_Q8_24.md`](../docs/engineering/specs/ARITHMETIC_Q8_24.md),
Section 3.3.

## Environment

From the repository root:

```bash
nix develop .#hardware
```

From this directory:

```bash
nix develop ..#hardware
```

The shell provides GNU Make, Verilator, GTKWave, Verible, and the Python 3
interpreter required by the Verilator binary build. The Makefile automatically
invokes the parent hardware shell when `verilator` is not already available on
`PATH`.

## Commands

Run these commands from `npu/`:

```bash
make lint    # verilator --lint-only -Wall over files.f (top: layer_engine)
make sim     # Compiles (files.f + tests/tb.sv, --binary --timing --assert) and runs
make test    # Alias of sim; propagates the testbench exit code
make wave    # Opens GTKWave with build/dump.vcd
make clean   # Removes the build/ directory
```

`files.f` is the single source of the RTL file list (`npu_pkg.sv` first); the
Makefile and the language server consume the same list.

The build tolerates warnings so the simulation always runs; `make lint` is the
quality gate and must exit zero with no `WIDTHTRUNC`/`WIDTHEXPAND`/`UNDRIVEN`
findings. `make test` propagates the testbench exit code, so a future
self-checking testbench that calls `$fatal` on a golden mismatch fails the
target automatically. The current testbench asserts no expected value; golden
vectors and assertions are pending the Rust bit-exact model (implementation
Step 1).

## Current Limitations

- **Weight file format mismatch:** `data/weights.mem` holds legacy dense
  Q8.24 32-bit words, while `layer_engine` consumes 2-bit ternary codes
  ($v = b_0 - b_1$). Reading the legacy file into the 2-bit array truncates
  each word to its two low bits. The ternary weight export (`wt.mem`) arrives
  with implementation Step 1; until then the testbench needs hand-written
  ternary codes or literal weights.
- **No assertions:** the testbench prints results but does not compare them
  against golden vectors; the parity protocol of
  [`ARITHMETIC_Q8_24.md`](../docs/engineering/specs/ARITHMETIC_Q8_24.md),
  Section 4 is unimplemented.
- **Elaboration-time shapes:** `IN` and `OUT` are still elaboration
  parameters. The 16-bit runtime shape registers are Step 2 (LOG-007,
  Section 4.2).

## Memory Representation

`$readmemh` loads memory files directly into signed 32-bit Q8.24 arrays (the
2-bit weight array excepted). The Rust export module (`src/export.rs`) writes
the Q8.24 hexadecimal values. See
[`MEM_FORMAT.md`](../docs/engineering/MEM_FORMAT.md) for full format details.

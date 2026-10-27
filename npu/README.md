# SystemVerilog NPU Prototype

This directory contains a synthesizable SystemVerilog neural processing unit (NPU)
prototype and its simulation testbench.

## Files

| File / Directory | Purpose |
|---|---|
| `src/npu.sv` | Top-level `npu` module with FSM control (`IDLE`, `BUSY`) parameterized by inputs `IN` and outputs `OUT` |
| `src/pe.sv` | Processing Element (PE) performing multiply-accumulate with 64-bit accumulator in Q8.24 |
| `tests/tb.sv` | Testbench with clock generation, reset, memory loading, waveform dumping, and display |
| `data/activations.mem` | Q8.24 hexadecimal activation words (8 entries for 4 evaluation pairs) |
| `data/weights.mem` | Q8.24 hexadecimal weight words |
| `data/bias.mem` | Q8.24 hexadecimal bias word |
| `data/target.mem` | Q8.24 hexadecimal target outputs |
| `Makefile` | Simulation (`sim`), waveform inspection (`wave`), linting (`lint`), and cleanup |

## Interface and Arithmetic

The testbench instantiates `npu` with parameters `IN = 2` and `OUT = 1`. The top-level module coordinates with `pe.sv` using an `init` and `en` handshake over the input elements. The processing element computes 64-bit products ($Q16.48$) and arithmetic-shifts right by 24 bits to accumulate in Q8.24 format. On vector completion for each output neuron, the top level adds the per-output bias `b[j]`, saturates the sum to the signed 32-bit Q8.24 range, and commits `result[j]`.

## Environment

From the repository root:

```bash
nix develop .#hardware
```

From this directory:

```bash
nix develop ..#hardware
```

The shell provides GNU Make, Verilator, GTKWave, Verible, the Python 3 interpreter
required by the Verilator binary build, and Icarus Verilog, an alternative simulator
not used by the current Makefile flow. The Makefile automatically invokes the parent
hardware shell when `verilator` is not already available on `PATH`.

## Commands

Run these commands from `npu/`:

```bash
make sim     # Compiles and runs simulation using Verilator
make wave    # Opens GTKWave with build/dump.vcd
make lint    # Runs Verilator lint-only checks on src/ and tests/
make clean   # Removes the build/ directory
```

`make sim` compiles `src/pe.sv`, `src/npu.sv`, and `tests/tb.sv` with the Verilator `--binary --timing --trace` flow and prints output. Waveforms are dumped to `build/dump.vcd`. The testbench asserts no expected value and returns no non-zero exit code; automated golden assertion checks are scheduled for Phase 4. `make lint` currently reports two index-width warnings (`WIDTHTRUNC`) in `src/npu.sv` and therefore exits non-zero.

## Memory Representation

`$readmemh` loads memory files directly into signed 32-bit Q8.24 arrays. The Rust export module (`src/export.rs`) writes matching Q8.24 hexadecimal values. See [`../docs/MEM_FORMAT.md`](../docs/MEM_FORMAT.md) for full format details.

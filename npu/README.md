# SystemVerilog NPU Prototype

This directory contains the dense ternary neural processing unit (NPU) RTL
for v0.1.0 and the Verilator simulation targets. Tier 1 runs hand-calculated
self-checks. The assembled Tier 2 bench is present and exact Rust–RTL parity
remains a release gate.

## Files

| File / Directory | Purpose |
|---|---|
| `src/npu_pkg.sv` | Shared package: `activation_t` (`IDENTITY`, `RELU`; `SIGMOID` reserved), `state_t` (`IDLE`, `RUN`, `WRITE`), and the `Q8_24_MAX` / `Q8_24_MIN` constants |
| `src/t3_dec.sv` | Ternary weight decode: $v = b_0 - b_1$ over the 2-bit code (`w_en = w[1] ^ w[0]`, `w_s = w[1]`) |
| `src/pe.sv` | Ternary processing element: conditional add/subtract accumulate into a signed 64-bit accumulator, negation at 33 bits before extension |
| `src/epilogue.sv` | Per-neuron post-processing: 96-bit $W_0$ product, arithmetic shift by 24, two saturations, bias, activation |
| `src/layer_engine.sv` | Single-layer engine: `IDLE`/`RUN`/`WRITE` FSM driving one `pe` and one `epilogue`, with the linear weight-stream pointer (rule R1) |
| `src/npu.sv` | Per-network wrapper: two `layer_engine` instances chained into an MLP (defaults 784→64→10); see "What is `npu.sv`?" below |
| `tests/tb.sv` | Legacy single-layer smoke test with waveform output and displayed results |
| `tests/tb_core.sv` | Tier 1 self-checking tests with hand-calculated results |
| `tests/tb_parity.sv` | Tier 2 assembled MNIST-sized comparison bench |
| `data/` | Legacy smoke inputs and memory files for the assembled bench. [Artifact formats](../docs/engineering/MEM_FORMAT.md) |
| `files.f` | Single source list of RTL files, shared by the Makefile and the LSP |
| `Makefile` | Simulation (`sim`), test (`test`), waveform inspection (`wave`), linting (`lint`), and cleanup |

## What is `npu.sv`?

`npu.sv` is the **per-network wrapper** — a thin top that instantiates two
`layer_engine` blocks and wires them into one MLP:

```text
start ──► u_l0 (784→64, RELU) ──l0_done──► u_l1 (64→10, IDENTITY) ──► done
             ▲                                  ▲
          l0_w, l0_b, l0_w0                  l1_w, l1_b, l1_w0
```

The default parameters shape the MNIST release network ($784 \to 64 \to 10$).
Elaboration parameters change the dimensions of this two-layer topology.
Weights, biases, scales, and activation selections arrive over ports. The
`done` pulse of layer 0 starts layer 1. [Handshake contract](../docs/engineering/PROTOCOL.md)

This wrapper is the manual-composition reference of Layer 2 in
[`ASSEMBLY_AND_TEST_MODEL.md`](../docs/engineering/ASSEMBLY_AND_TEST_MODEL.md),
Section 1. It does **not** contain the sequencer, program memory, or
activation banks of the programmable top; those arrive in v0.1.1 (LOG-008)
and will replace the hand-wired chain, not the engines.

**Protocol.** `start` and `done` are events (one-cycle pulses); the wrapper
arbitrates requests while data is in flight. See
[`PROTOCOL.md`](../docs/engineering/PROTOCOL.md).

## Interface and Arithmetic

The legacy smoke test instantiates `layer_engine` with parameters `IN = 2` and
`OUT = 1`. The engine exposes an activation array `a`, 2-bit
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
make lint        # verilator --lint-only -Wall over files.f (top: npu — wrapper + engines + primitives)
make sim         # Legacy smoke: compiles (files.f + tests/tb.sv) and runs without assertions
make test        # Runs Tier 1 testbench (test-core); propagates exit code ($fatal on mismatch)
make test-core   # Compiles and runs tests/tb_core.sv with self-checking assertions
make test-parity # Runs the assembled comparison bench with matching memory files
make wave        # Opens GTKWave with build/dump.vcd
make clean       # Removes the build/ directory
```

`files.f` is the single source of the RTL file list (`npu_pkg.sv` first); the
Makefile and the language server consume the same list.

The simulation build uses `-Wno-fatal` for warnings. `make lint` is the
separate strict lint gate. CI runs lint and Tier 1. `make test` defaults to
`test-core`, which calls `$fatal` on a mismatch.
[CI workflow](../.github/workflows/ci.yml)

## Tests

Verification is organized in three tiers, each with its own golden source and
change trigger. The full model — scope, golden sources, gates, and the make
targets that will host each tier — is recorded in
[`ASSEMBLY_AND_TEST_MODEL.md`](../docs/engineering/ASSEMBLY_AND_TEST_MODEL.md),
Section 5:

| Tier | Scope | Golden source | State |
|---|---|---|---|
| ① `tb_core` | Ternary primitives and the assembled wrapper | Hand-computed values in the testbench | Implemented under `make test-core` / `make test` |
| ② `tb_parity` | Assembled `784 → 64 → 10` wrapper | Current development inputs and float-derived expectations. Release requires the independent Rust integer reference. | [PENDING] Bench and command exist. The comparator allows one LSB and must become an exact word comparison. |
| ③ `tb_shell` | Programmable top: sequencer, memories, I/O | System-level golden + `program.mem` | Deferred to v0.1.1 (LOG-008) |

Tier 1 runs one two-layer inference and checks both output words against
hand-calculated constants. The protocol checks for repeated requests and
requests while busy are listed in the [protocol contract](../docs/engineering/PROTOCOL.md#test-gates).

## Current Limitations

- The legacy `weights.mem` contains dense Q8.24 words. The smoke bench reads
  their low two bits into its ternary array. The assembled bench uses `wt.mem`.
  [Current and legacy artifacts](../docs/engineering/MEM_FORMAT.md)
- `IN`, `H`, and `OUT` are elaboration parameters. Runtime shapes and the
  programmable sequencer are deferred to v0.1.1.
  [Scope decision](../docs/project/research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026)

## Memory Representation

The assembled bench loads 2-bit ternary weight codes and signed 32-bit Q8.24
scales, biases, inputs, and expected outputs with `$readmemh`.
[Memory encodings and file order](../docs/engineering/MEM_FORMAT.md)

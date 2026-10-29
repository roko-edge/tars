# Getting Started

This guide describes the commands and behavior available in the current repository.
TARS is experimental: public interfaces may change, and some subsystems are incomplete.

---

## 1. Environment

The Nix flake provides development shells for `x86_64-linux`:

```bash
nix develop
nix develop .#rust
nix develop .#hardware
```

The default shell contains the Rust and hardware toolchains. The Rust shell includes
Graphviz and the visualization dependencies. The hardware shell includes GNU Make,
Verilator, GTKWave, Verible, the Python 3 interpreter required by the Verilator
binary build invoked by `npu/Makefile`, and Icarus Verilog, an alternative simulator
not used by the current Makefile flow.

Without Nix, provide a Rust 2024 toolchain and the corresponding system tools manually.

---

## 2. Build the Rust Package

Run commands from the repository root:

```bash
cargo build
```

The package contains the library and two binaries, `main` and `visualization`.

`cargo test` compiles. The suite contains four tests; one of them exercises the
tensor core, and the module hierarchy, the optimizer, and the export path have no
coverage. The suite is not deterministic: the distribution tests sample the unseeded
global generator, and the ternary-distribution test asserts a variance bound at the
true value of its distribution, so it fails intermittently.

---

## 3. Run the Training Binary

```bash
cargo run --bin main
```

The current binary:

1. Selects the OR experiment in source; it has no prompt or command-line selector.
2. Trains for the configured 100,000 epochs. Weights initialize through the unseeded
   random constructor and biases to zero, so runs are not reproducible.
3. Prints periodic mean squared error values and final floating-point predictions.
4. Skips the export calls: `export_model` and `export_data` are commented out in the
   source, so the checked-in `npu/data/` files are not overwritten.
5. Extracts the network topology and writes `src/view/artifacts/graph.dot`, then
   renders `src/view/artifacts/network.png` through Graphviz. The directory is not
   created automatically and is absent in a fresh clone, so the run terminates with a
   file error after the predictions are printed.

The files under `npu/data/` are the ones the NPU testbench reads; because the export
calls are disabled, they remain unchanged by a training run.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the experiment contract and selection
limitations.

---

## 4. Run the Visualization Binary

```bash
cargo run --bin visualization
```

The visualization binary requires Graphviz. It builds its own hard-coded random model;
it does not consume the model trained by `main`. It writes:

| Artifact | Path |
|---|---|
| Graphviz source | `src/view/artifacts/graph.dot` |
| Rendered image | `src/view/artifacts/network.png` |

---

## 5. Run the Hardware Prototype

Hardware setup, simulation, waveform inspection, and lint commands are documented in
[`npu/README.md`](../npu/README.md). The current prototype computes a single affine
layer with dense ternary weights (testbench: `IN = 2`, `OUT = 1`): conditional
add/subtract accumulation, the per-layer $W_0$ epilogue with two saturations and
per-output bias, and prints results without automatic target assertions. The flow is
Verilator-only.

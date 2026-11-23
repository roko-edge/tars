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
Verilator, GTKWave, Verible, and the Python 3 interpreter required by the Verilator
binary build invoked by `npu/Makefile`. The NPU flow is strictly Verilator-only.

Without Nix, provide a Rust 2024 toolchain and the corresponding system tools manually.

---

## 2. Build the Rust Package

Run commands from the repository root:

```bash
cargo build --lib --bins
cargo test
```

The package contains the Rust library, the `main` training binary, the `load_ds`
MNIST reader example, and the `visualization` stub. The CI workflow builds the
library and binaries and runs the Rust test suite. [CI workflow](../.github/workflows/ci.yml)

---

## 3. Run the Training Binary

```bash
cargo run --release --locked --bin main
```

`main` selects the MNIST factory in source. It reads the data under
`mnist-data/` and downloads missing files through the reader. The runner prints
training progress, then evaluates the float model and its PTQ projection on a
reserved suffix of the official training split. OR and XOR are library factories.
[Experiment behavior](EXPERIMENTS.md#selection-and-execution)

The training command reports results to the terminal. Hardware memory files
come from explicit calls to the export functions. [Experiment behavior and benchmark capture](EXPERIMENTS.md#selection-and-execution) ·
[Export API](API.md#4-export-and-visualization)

---

## 4. Run the Visualization Binary

The `visualization` binary is a stub. Network diagram generation is tracked
under [visualization status](VISUALIZATION.md).

---

## 5. Run the Hardware Prototype

From the repository root, use `make -C npu test` for the Tier 1 hand-calculated
checks and `make -C npu lint` for RTL lint. Use `make -C npu sim` for the legacy
single-layer smoke test. The assembled two-layer parity bench has a separate
`test-parity` target and requires matching inputs and expected outputs.
[Hardware commands and fixtures](../npu/README.md#commands)

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
cargo build --lib --examples
cargo test --all-targets
```

The package contains the Rust library (`src/lib.rs`) and runnable examples
under `examples/` (`mnist`, `or`, `xor`), using `mnist_reader` from
`[dev-dependencies]`. The CI workflow compiles the library, any binary targets,
and the examples, and runs the Rust test suite. [CI workflow](../.github/workflows/ci.yml)

---

## 3. Run the MNIST Example

```bash
cargo run --release --locked --example mnist
```

`examples/mnist.rs` reads MNIST data under `mnist-data/` (downloading missing
files through `mnist_reader`). It trains `784 → 64 → 10` on `train_data` with
QAT enabled (`Trainer::with_qat(true)`), prints per-epoch progress, evaluates
the latent `f32` model, scaled ternary projection, sparsity, and model footprint
on `test_data` via `ModelReport`, and writes `.mem` artifacts under `npu/data/`
via `export_quantized_model` and `export_inference`. Smaller OR and XOR examples
run via `cargo run --example or` and `cargo run --example xor`.
[Experiment behavior](EXPERIMENTS.md#selection-and-execution) ·
[Export API](API.md#4-export-and-visualization)

---

## 4. Network Visualization

Network diagram generation under `src/view/` is currently outside `src/lib.rs`
and tracked under [visualization status](VISUALIZATION.md).

---

## 5. Run the Hardware Prototype

From the repository root, use `make -C npu test` for the Tier 1 hand-calculated
checks and `make -C npu lint` for RTL lint. Use `make -C npu sim` for the legacy
single-layer smoke test. The assembled two-layer parity bench has a separate
`test-parity` target and requires matching inputs and expected outputs.
[Hardware commands and fixtures](../npu/README.md#commands)

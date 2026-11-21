<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tars-logo-transparent.png" />
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/tars-logo-transparent-dark.png" />
    <img src="docs/assets/tars-logo-transparent-dark.png" alt="Tars — geometric robot logo" width="360" height="459" />
  </picture>
</h1>

<p align="center">
  Experimental neural networks in Rust.<br />
    A ternary inference prototype in SystemVerilog.
</p>

<p align="center">
  <a href="#requirements">Getting started</a> &middot;
  <a href="#rust">Rust</a> &middot;
  <a href="#systemverilog">SystemVerilog</a> &middot;
  <a href="#documentation">Documentation</a>
</p>

---

TARS pairs a Rust training library with a dense ternary SystemVerilog inference
prototype. The library provides typed float tensors, sequential dense models,
ReLU and sigmoid, analytical backpropagation, MSE and SoftmaxCE, and BGD and
Adam. The training binary selects MNIST in source.

The explicit exporter writes ternary weight codes and Q8.24 scales, biases,
and inference data under `npu/data/`. The two-layer RTL wrapper is assembled
at elaboration. Exact parity with an independent Rust integer reference is a
v0.1.0 release gate. [Memory format](docs/engineering/MEM_FORMAT.md) ·
[Release gates](docs/project/V0_DELIVERY_PLAN.md#21-which-results-close-the-release-gate)

## Requirements

The Nix flake defines development shells for `x86_64-linux`:

```bash
nix develop             # Rust, NPU, Nix, and visualization tools
nix develop .#rust      # Rust and visualization tools
nix develop .#hardware  # Verilator, GTKWave, Verible (the NPU flow is Verilator-only)
```

Without Nix, use a Rust toolchain supporting edition 2024. Install GNU Make
and Verilator for the RTL flow, and GTKWave for waveform inspection.
[Environment setup](docs/GETTING_STARTED.md#1-environment)

## Rust

```bash
cargo build --lib --bins
cargo run --release --locked --bin main
cargo test
```

`main` reports training progress and evaluates on the MNIST training split.
Experiment configuration comes from the selected factory. Hardware export and
visualization are separate from this command. [Training guide](docs/GETTING_STARTED.md#3-run-the-training-binary)

## SystemVerilog

From `npu/`:

```bash
make test    # Runs Tier 1 self-checking testbench (test-core); exits zero on pass
make sim     # Runs legacy smoke testbench (prints result signals)
make wave    # Opens GTKWave with build/dump.vcd
make lint    # Verilator lint-only quality gate (top: npu)
make clean   # Cleans build artifacts
```

`make test` runs the Tier 1 hand-calculated checks. The separate `test-parity`
target runs the assembled MNIST-sized bench and requires matching memory
files. Its current comparison is a development check. [RTL commands and test states](npu/README.md#tests)

## Documentation

See [`docs/README.md`](docs/README.md) for user documentation and navigation by task.

| Key Reference | Scope |
|---|---|
| [`docs/README.md`](docs/README.md) | User documentation entry point |
| [`docs/project/V0_CONTRACT.md`](docs/project/V0_CONTRACT.md) | v0.1.0 release target and acceptance criteria |
| [`docs/project/ROADMAP.md`](docs/project/ROADMAP.md) | Project milestones and ownership |
| [`docs/project/STATUS.md`](docs/project/STATUS.md) | Current implementation state and known limitations |
| [`AGENTS.md`](AGENTS.md) | Repository automation policy for AI tools |

## Licensing

Copyright (c) 2026 The TARS Authors.

The Rust package, the SystemVerilog NPU prototype, and this documentation are
distributed under the Apache License, Version 2.0. The license text is available in
the [`LICENSE`](LICENSE) file and the authorship record in the
[`AUTHORS`](AUTHORS) file. Apache-2.0 governs all use of this project, commercial
use included; no separate commercial license is required.

TARS is an independent open-source project: its copyright is held by the individual
authors listed in [`AUTHORS`](AUTHORS). The authors are affiliated with the
Universidade Federal do Cariri (UFCA), where TARS is studied and evaluated in
academic research (see the companion
[tars-research](https://github.com/roko-edge/tars-research) repository). UFCA is
credited as the authors' affiliation and research home; it is not a copyright
holder of this repository.

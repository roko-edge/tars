# Experiments

**Audience:** Users running the OR, XOR, and MNIST configurations.

This page describes the experiment API and training flow currently implemented in
TARS. Experiment definitions configure a run; the executable owns the training loop.

## Module structure

| Source | Responsibility |
|---|---|
| [`src/train.rs`](../src/train.rs) | Shared `Train` configuration |
| [`src/experiments.rs`](../src/experiments.rs) | Public experiment modules |
| [`src/experiments/experiment_type.rs`](../src/experiments/experiment_type.rs) | `ExperimentType` identifiers |
| [`src/experiments/or.rs`](../src/experiments/or.rs) | OR configuration factory |
| [`src/experiments/xor.rs`](../src/experiments/xor.rs) | XOR configuration factory |
| [`src/experiments/mnist.rs`](../src/experiments/mnist.rs) | MNIST configuration factory |
| [`src/bin/main.rs`](../src/bin/main.rs) | Selection, training, and accuracy reporting |

`Train` was moved out of `lib.rs` into `train.rs`. The library declares `pub mod
train` and re-exports its contents, so callers can use either `tars::Train` or
`tars::train::Train`. It also exposes the `experiments` module and re-exports its
public modules.

## Configuration contract

`Train` is a struct deriving `Debug` and `Clone`, with these public fields:

| Field | Type | Purpose |
|---|---|---|
| `model` | `Sequential` | Network whose parameters are updated during training |
| `epochs` | `usize` | Number of passes over the training data |
| `lr` | `f32` | Learning rate supplied to the optimizer |
| `dataset` | `Vec<Data>` | Input and target tensors in `f32` |
| `experiment_type` | `ExperimentType` | Identifier printed when the run starts |

Each experiment exposes `pub fn train() -> Train` to construct a fresh model
and dataset. `Sequential::linear` calls `Linear::random`, which starts a fixed
seeded generator for each layer and initializes biases to zero.

`ExperimentType` provides `Or`, `Xor`, `Mnist`, and `Custom` identifiers. Each
factory supplies the identifier in its configuration.

## Built-in experiments

| Factory | Network | Epochs | Learning rate | Identifier |
|---|---|---:|---:|---|
| `tars::experiments::or::train()` | `2 -> Linear(1) -> Sigmoid` | 100,000 | 20.0 | `ExperimentType::Or` |
| `tars::experiments::xor::train()` | `2 -> Linear(4) -> Sigmoid -> Linear(1) -> Sigmoid` | 1,000,000 | 1.0 | `ExperimentType::Xor` |
| `tars::experiments::mnist::train()` at `e7b7f64` | `784 → 10`, one affine layer | 10 | 0.003 | `ExperimentType::Mnist` |

The OR and XOR datasets contain the complete two-input truth table.

| Input | OR target | XOR target |
|---|---|---|
| `[0.0, 0.0]` | `[0.0]` | `[0.0]` |
| `[0.0, 1.0]` | `[1.0]` | `[1.0]` |
| `[1.0, 0.0]` | `[1.0]` | `[1.0]` |
| `[1.0, 1.0]` | `[1.0]` | `[0.0]` |

## Selection and execution

The executable selects `experiments::mnist::train()` in `src/bin/main.rs`.
The factory takes images and labels from the MNIST training split and creates
one-hot targets with ten classes. The runner at `e7b7f64` uses Adam and
batches of 512.
[MNIST factory](../src/experiments/mnist.rs) · [Runner](../src/bin/main.rs)

Selection is fixed in the source compiled into the executable. The small-model
factories supply training data and topology for OR and XOR. Their MSE regression
checks are part of the release verification plan.

The training executable runs from the repository root.

```bash
cargo run --release --locked --bin main
```

Each optimization step reads a batch from `DataMiniBatches`, records the forward
trace, computes analytical parameter gradients, and updates the model. The
runner derives synthetic MSE targets from the SoftmaxCE gradient and reports
argmax accuracy on the training samples.
Associate each report with its exact source. [API reference](API.md#3-training-operations)

## Exports and constraints

Export is inactive in the normal training run. Explicit export writes the
artifacts described in the [API reference](API.md#4-export-and-visualization).
The batch loader requires a nonempty dataset and a positive batch size. Samples
in a batch must have consistent input and target shapes.

## Adding an experiment manually

A human maintainer can extend the same pattern:

1. Define a module under `src/experiments/` with a `pub fn train() -> Train` factory.
2. Supply all five configuration fields, including a matching experiment identifier.
3. Add the identifier to `ExperimentType` and expose the module in
   `src/experiments.rs`.
4. Select the factory in `src/bin/main.rs`, rebuild, and run the executable.
5. Document the topology, dataset, and hyperparameters alongside this page.

For the broader module structure, see
[`engineering/ARCHITECTURE.md`](engineering/ARCHITECTURE.md).

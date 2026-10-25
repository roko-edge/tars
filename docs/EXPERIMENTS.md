# Experiments

**Audience:** Users running the implemented OR and XOR configurations.

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
| [`src/bin/main.rs`](../src/bin/main.rs) | Selection, training, predictions, export, and graph construction |

`Train` was moved out of `lib.rs` into `train.rs`. The library declares `pub mod
train` and re-exports its contents, so callers can use either `tars::Train` or
`tars::train::Train`. It also exposes the `experiments` module and re-exports its
public modules.

## Configuration contract

`Train` is a struct deriving `Debug` and `Clone`, with these public fields:

| Field | Type | Purpose |
|---|---|---|
| `model` | `Sequential` | Network whose parameters are updated during training |
| `epochs` | `usize` | Number of full-dataset optimization steps |
| `lr` | `f32` | Learning rate supplied to BGD |
| `dataset` | `Vec<Data>` | Input vectors and expected target vectors |
| `experiment_type` | `ExperimentType` | Identifier printed when the run starts |

Each experiment exposes `pub fn train() -> Train`. Calling it constructs a fresh
model and dataset; it does not execute training. `Sequential::linear` constructs
layers through `Linear::random`: weights are drawn from the unseeded global
generator and biases initialize to zero. Two runs of the same configuration are not
reproducible.

`ExperimentType` is a Rust enum deriving `Debug` and `Clone`, with variants `Or` and
`Xor`. It identifies the configuration for logging. It does not select a factory,
build a model, or dispatch the training loop; the factory explicitly supplies it.

## Built-in experiments

| Factory | Network | Epochs | Learning rate | Identifier |
|---|---|---:|---:|---|
| `tars::experiments::or::train()` | `2 -> Linear(1) -> Sigmoid` | 100,000 | 20.0 | `ExperimentType::Or` |
| `tars::experiments::xor::train()` | `2 -> Linear(4) -> Sigmoid -> Linear(1) -> Sigmoid` | 100,000 | 1.0 | `ExperimentType::Xor` |

Both datasets contain the complete two-input truth table in the following order:

| Input | OR target | XOR target |
|---|---|---|
| `[0.0, 0.0]` | `[0.0]` | `[0.0]` |
| `[0.0, 1.0]` | `[1.0]` | `[1.0]` |
| `[1.0, 0.0]` | `[1.0]` | `[1.0]` |
| `[1.0, 1.0]` | `[1.0]` | `[0.0]` |

## Selection and execution

The executable currently selects the OR factory in `src/bin/main.rs`.

Selection is fixed in the source compiled into the executable. To select XOR, a
human maintainer changes the selected factory to `experiments::xor::train()` and
rebuilds. There
is no stdin prompt, command-line selector, or Cargo feature selecting an experiment.
The factory itself constructs the configuration at runtime.

From the repository root, run the configured experiment with:

```bash
cargo run --bin main
```

```text
Experiment factory -> Train -> Initial MSE
                               |
                               v
                 backward over the full dataset
                               |
                               v
                   BGD updates weights and biases
                               |
                               v
                  Recompute MSE and report progress
                     (repeat for every epoch)
                               |
                               v
                  Print predictions -> Graph artifacts
```

1. Print the experiment identifier and calculate the initial mean squared error
   with `cost(&train.model, &train.dataset)` at epoch zero.
2. Create `BGD::new(train.lr)` and set the reporting interval to
   `(train.epochs / 20).max(1)`.
3. For each epoch from `1` through `train.epochs`, call `backward` on the full
   dataset. It stores forward activations, propagates loss derivatives through the
   modules in reverse order, and averages parameter gradients across samples.
4. Apply one BGD step: each weight and bias is updated by subtracting its gradient
   multiplied by the learning rate. Recalculate the full-dataset MSE.
5. At each reporting interval, print the epoch, cost, and percentage improvement
   relative to the previous reported cost, then update that reference cost. With
   the built-in configurations, reports occur every 5,000 epochs.
6. After training, print each sample's input, network output, and expected target.
   Outputs remain floating-point predictions; the executable does not threshold
   them into binary labels or calculate classification accuracy.
7. Create a `NetGraph` and call `network_to_graph` with the trained model and the
   first sample's input. This extracts the neurons and weighted edges, writes
   `src/view/artifacts/graph.dot`, and renders `src/view/artifacts/network.png`
   through Graphviz. Activation modules are skipped by graph extraction, and the
   artifacts directory must already exist.

The loop uses analytical backpropagation (`backward`). The numerical
finite-difference helper `num_grad` remains available but is not used by `main`.
Training runs for the configured epoch count without early stopping.

## Exports and constraints

The calls to `export_model` and `export_data` are commented out in
`src/bin/main.rs`. A normal run does not write the files under `npu/data/`; the
checked-in artifacts remain from an earlier export and are the ones the NPU
testbench reads when running `make sim`. For the artifact-writing behavior that does
run at the end of `main`, see [`VISUALIZATION.md`](VISUALIZATION.md).

Configurations must provide a nonempty dataset because graph extraction accesses
`train.dataset[0]`. Inputs, targets, and layer dimensions must be consistent; the
configuration API does not validate them. Percentage improvement divides by the
previous reported cost without a zero guard.

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

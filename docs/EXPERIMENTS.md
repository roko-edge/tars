# Experiments

**Audience:** Users running the OR, XOR, and MNIST configurations.

Experiment definitions configure a run. At `4961a9e` the executable owns the
training loop. The latest inspected working tree moves it into `Trainer::fit`.
[Working-tree training API status](API.md#3-training-operations)

## Module structure

| Source | Responsibility |
|---|---|
| [`src/train.rs`](../src/train.rs) | Shared `Train` configuration and working-tree `Trainer::fit` loop |
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

The working-tree `Train<L: Loss>` groups these public fields:

| Field | Type | Purpose |
|---|---|---|
| `model` | `Sequential` | Network whose parameters are updated during training |
| `epochs` | `usize` | Number of passes over the training data |
| `lr` | `f32` | Learning rate supplied to the optimizer |
| `dataset` | `Vec<Data>` | Input and target tensors in `f32` |
| `experiment_type` | `ExperimentType` | Experiment identifier |
| `loss` | `L` | Training objective passed to `Trainer::new` |

Each working-tree factory returns `Train` parameterized by its loss. MNIST
uses `SoftmaxCE`, and OR and XOR use `MSE`. `Sequential::linear` calls
`Linear::random`, which starts a fixed seeded generator for each layer and
initializes biases to zero.

`ExperimentType` provides `Or`, `Xor`, `Mnist`, and `Custom` identifiers. Each
factory supplies the identifier in its configuration.

## Built-in experiments

| Factory | Network | Epochs | Learning rate | Identifier |
|---|---|---:|---:|---|
| `tars::experiments::or::train()` | `2 -> Linear(1) -> Sigmoid` | 100,000 | 20.0 | `ExperimentType::Or` |
| `tars::experiments::xor::train()` | `2 -> Linear(4) -> Sigmoid -> Linear(1) -> Sigmoid` | 1,000,000 | 1.0 | `ExperimentType::Xor` |
| `tars::experiments::mnist::train()` at `e7b7f64` | `784 → 10`, one affine layer | 10 | 0.003 | `ExperimentType::Mnist` |
| `tars::experiments::mnist::train()` at `4961a9e` | `784 → 128 → 64 → 10`, ReLU after the hidden layers | 5 | 0.001 | `ExperimentType::Mnist` |
| Earlier MNIST working-tree snapshot on 2026-11-22 | `784 → 64 → 10`, ReLU after the hidden layer | 100 | 0.001 | `ExperimentType::Mnist` |
| MNIST working tree with `Trainer::fit` | `784 → 64 → 10`, ReLU after the hidden layer | 50 | 0.001 | `ExperimentType::Mnist` |

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
one-hot targets with ten classes. The baseline at `e7b7f64` uses Adam and
batches of 512. The QAT experiment at `4961a9e` uses Adam and batches of 32
with scaled ternary weights represented in `f32`. An earlier working-tree snapshot
uses float training with Adam and batches of 64, fits the first 50,000 samples
from the official training split, and evaluates on the remaining training
samples. The reader exposes the official test split separately.
[MNIST factory](../src/experiments/mnist.rs) · [Runner](../src/bin/main.rs)

The working tree with `Trainer::fit` uses the same split and batch size on
the assembled `784 → 64 → 10` topology. It trains in float mode with Adam and
evaluates the final float model and its scaled float PTQ projection on the
reserved suffix. This is an experimental host workflow. The release comparison
uses the official test split and the independent integer inference path.
[Working-tree API status](API.md#3-training-operations) ·
[Release plan](project/V0_DELIVERY_PLAN.md#23-how-do-we-measure-and-improve-quantized-accuracy-pending)

Selection is fixed in the source compiled into the executable. The small-model
factories supply training data and topology for OR and XOR. Their MSE regression
checks are part of the release verification plan.

The training executable runs from the repository root.

```bash
cargo run --release --locked --bin main
```

Each optimization step reads a batch from `DataMiniBatches`, records the forward
trace, computes analytical parameter gradients, and updates the model. The
baseline at `e7b7f64` derives synthetic MSE targets from the SoftmaxCE gradient.
The QAT path at `4961a9e` uses the SoftmaxCE gradient directly and updates
latent float weights. Those two committed snapshots report argmax accuracy on
the samples used for training. The earlier working-tree snapshot uses a disjoint
suffix of the official training split. Release accuracy
requires a frozen checkpoint and independent test data. Associate each
report with its exact source, including local configuration overrides.
[Held-out evaluation protocol](project/V0_DELIVERY_PLAN.md#231-how-do-we-keep-the-final-mnist-evaluation-independent-proposed) ·
[API reference](API.md#3-training-operations)

## Exports and constraints

Export is inactive in the normal training run. Explicit export writes the
artifacts described in the [API reference](API.md#4-export-and-visualization).
The batch loader requires a nonempty dataset and a positive batch size. Samples
in a batch must have consistent input and target shapes.

## Adding an experiment manually

A human maintainer can extend the same pattern:

1. Define a module under `src/experiments/` with a `train()` factory returning `Train` parameterized by the chosen loss type.
2. Supply the configuration fields, including the loss and a matching experiment identifier.
3. Add the identifier to `ExperimentType` and expose the module in
   `src/experiments.rs`.
4. Select the factory in `src/bin/main.rs`, rebuild, and run the executable.
5. Document the topology, dataset, and hyperparameters alongside this page.

For the broader module structure, see
[`engineering/ARCHITECTURE.md`](engineering/ARCHITECTURE.md).

## How should benchmark runs be recorded? [PROPOSED]

Create one record per invocation and keep its raw log with the source snapshot,
configuration and evaluation artifacts. A terminal-only log is usable as an
exploratory observation with unknown fields recorded explicitly. A controlled
comparison needs matched workloads and reproducible inputs. Release evidence
also needs the held-out predictions and the exact inference artifact.
[Release contract](project/V0_CONTRACT.md#63-research-claim-discipline)

### What belongs in a run record?

| Record | Fields to retain |
| --- | --- |
| Run and comparison | Unique `run_id`, UTC start, operator, purpose, comparison ID or parent run, question, intended change, primary metric, and outcome or exit status. Keep failed and interrupted attempts under their own IDs. |
| Source and machine | Exact command, commit, dirty diff or archived source including any untracked build inputs, `Cargo.lock` hash, compiler version and target, profile, features, flags, CPU, OS and thread count. For timing comparisons record CPU affinity and frequency or power settings when controlled. |
| Data | Dataset and file hashes, preprocessing, train/development/test roles, sample counts, exact split IDs or deterministic derivation, and the sample-order algorithm. Use the development split for model choices and freeze the final test protocol before evaluating. |
| Model and training | Topology and activations, parameter counts, starting checkpoint hash if any, float or QAT training, objective and reduction, optimizer and hyperparameters, batch policy, epochs and actual optimizer steps. Record initialization and shuffle seeds and algorithms separately. |
| Quantization and evaluation | Checkpoint hash, split, evaluation path, argmax tie rule, quantizer scale, threshold, tie and cutoff rules, bias treatment, and wire revision where applicable. Link artifacts and raw prediction files by hash. |

For existing logs, preserve the original units and labels in the raw log. In
the summary, label any reinterpretation and leave unavailable fields blank or
mark them `unknown`. In particular, a legacy line marked `Ternary PTQ` can
describe the projected evaluation of a QAT run. Confirm the training mode from
the runner snapshot or operator notes. Rounded percentages are insufficient to
reconstruct exact counts. Use a new run ID for each rerun and keep revisions to
the record traceable.

[CONFLICT] The `4961a9e` runner and both inspected working-tree snapshots train
with SoftmaxCE and report an unweighted average of batch MSE on raw logits as
`Loss (MSE)`. In `4961a9e`, `Ternary PTQ` labels the projected evaluation of a
QAT checkpoint. In the float working-tree run, it labels PTQ on the float
checkpoint, evaluated through a scaled `f32` projection. The earlier working-tree
throughput uses the full dataset length even though fit consumes 50,000
samples per epoch. `Trainer::fit` uses the length of the fit slice for
throughput. `Performance degradation` is the accuracy difference in
percentage points. Record the objective, diagnostic, actual processed count
and evaluation path independently until the runner reports them directly.
[Runner](../src/bin/main.rs)

### How can a maintainer capture the current runner today?

From the project root, with the existing MNIST files present, the command below
captures the source and raw output locally. Add the model, seeds, optimizer,
machine and quantizer metadata from the configuration actually built. Archive
untracked build inputs if any were used. Keep the resulting directory outside
the public repository and reference its run ID from the engineering log.

```bash
(
    set -e
    run_root="${XDG_STATE_HOME:-$HOME/.local/state}/tars/benchmarks"
    mkdir -p "$run_root"
    run_dir=$(mktemp -d "$run_root/$(date -u +%Y%m%dT%H%M%SZ)-mnist-XXXXXX")
    printf '%s\n' "$run_dir"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_at.txt"
    git log -1 --format=%H > "$run_dir/commit.txt"
    git status --short --branch --untracked-files=all > "$run_dir/status.txt"
    git diff --binary HEAD > "$run_dir/source.diff"
    rustc -Vv > "$run_dir/rustc.txt"
    cargo -V > "$run_dir/cargo.txt"
    uname -srm > "$run_dir/platform.txt"
    if command -v lscpu > /dev/null
    then
        lscpu > "$run_dir/cpu.txt"
    fi
    sha256sum Cargo.lock mnist-data/{train-images-idx3,train-labels-idx1,t10k-images-idx3,t10k-labels-idx1}-ubyte.gz > "$run_dir/input-sha256.txt"
    printf '%s\n' 'cargo run --release --locked --bin main' > "$run_dir/command.txt"
    run_exit=0
    cargo run --release --locked --bin main > "$run_dir/stdout.log" 2> "$run_dir/stderr.log" || run_exit=$?
    printf '%s\n' "$run_exit" > "$run_dir/exit_status.txt"
    exit "$run_exit"
)
```

The current runner can supply terminal logs immediately. Structured epoch
counters, saved checkpoints, held-out predictions and exact evaluator outputs
depend on the corresponding runner and evaluator work. Use the proposed CSV
headers when that data becomes available.
[Runner](../src/bin/main.rs) · [Delivery plan](project/V0_DELIVERY_PLAN.md)

### Which raw records make the metrics reproducible?

| File | Proposed CSV header or contents |
| --- | --- |
| `epochs.csv` | `run_id,epoch,samples_seen,optimizer_steps,elapsed_ns,loss_name,loss_sum,loss_count` with per-epoch, noncumulative counts. |
| `predictions.csv` | `run_id,evaluation_id,split,sample_id,target_class,predicted_class` with one row per sample per evaluation. Record the checkpoint hash and evaluation path for each `evaluation_id` in the run manifest. |
| `performance.csv` | `run_id,scope,phase,iteration,samples,elapsed_ns,allocations,peak_temporary_bytes` with raw observations and an explicit warm-up or measured phase. |
| `quantization.csv` | `run_id,evaluation_id,layer_index,weight_count,negative_count,zero_count,positive_count,scale_f32_bits,scale_q8_24_word` with one row per affine layer. |
| `parity.csv` | `run_id,evaluation_id,artifact_sha256,expected_words,compared_words,mismatched_words` with simulator or board details in the run manifest. Retain the first mismatch's sample ID, output index and exact words in the parity log. |

Store counts as integers and durations in nanoseconds when available. Define
time boundaries alongside the CSV. Keep unavailable values missing. Derived
tables come from raw counts and timings. Keep research conclusions and result
promotion in the [project research process](project/research/README.md).

### How should the metrics be calculated and checked?

| Metric | Calculation and scope |
| --- | --- |
| Training loss | Accumulate stable SoftmaxCE from logits as a sum over samples and divide by the number of samples. For MSE, accumulate squared errors and divide by the number of evaluated elements. Use the actual batch sizes and name any separate diagnostic, such as `logit_mse`. |
| Accuracy | Record integer `correct` and `total`, then compute `correct / total` for the declared split. Use the same sample IDs, labels and deterministic tie rule across paths. Check each ID occurs exactly once per evaluation, sample counts reconcile, and predictions are valid. |
| Accuracy difference | Report `100 × (accuracy_candidate − accuracy_reference)` in percentage points, with accuracies expressed as fractions. Keep the comparison direction in the header or caption. |
| Accuracy uncertainty | For a fixed checkpoint and held-out evaluation, a proposed method is a two-sided 95% exact binomial interval (Clopper–Pearson) from `correct` and `total`. Summarize between-seed variation separately. [NIST exact binomial limits](https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm) |
| Paired comparison | On the predeclared pair evaluated over the same IDs, record both correct, both wrong, reference only correct ($b$), and candidate only correct ($c$). Verify that these counts sum to $n$ and the signed difference is $100(c-b)/n$ percentage points. For the proposed two-sided exact McNemar test, use $\min(1, 2\Pr[\operatorname{Bin}(b+c, 1/2) \le \min(b,c)])$, with $p=1$ when there are no discordances. Record the variant alongside the effect. [Fagerland et al., exact conditional McNemar](https://doi.org/10.1186/1471-2288-13-91) |
| Ternary sparsity | Divide emitted zero trits by total weight trits. Report each layer and the ratio of summed counts across layers. Verify negative, zero and positive counts add up per layer. Exclude bias and wire padding. |
| Training throughput | Divide the sample count from batches actually processed by total wall time in the declared training window. Use the actual length of each batch, including a short final batch. Report samples per step, optimizer steps and the window's inclusions. Summarize repeated step timings with median and p95 and retain the raw observations. |
| Time to target quality | Predeclare a development-set target, evaluation cadence and budget. Record wall time and samples or steps to the first observed crossing, including scheduled evaluations in the timer. Record runs that reach the budget without crossing. [MLPerf Training benchmark paper](https://arxiv.org/abs/1910.01500) |
| Bit-exact parity | Check that the artifact identity and output counts match and that every compared word equals its reference word. Report comparisons and mismatches. Keep parity, inference cycles and end-to-end latency as independently scoped results. |

### Which comparisons answer different questions?

| Question | Comparison |
| --- | --- |
| What does PTQ cost? | Train a float baseline and evaluate it in float. Quantize that same checkpoint and evaluate its exported ternary form on the same inputs. |
| What does QAT recover? | Compare exported PTQ and QAT models using the same integer evaluator and held-out inputs. Declare starting checkpoints, update budgets, optimizer and quantizer settings. A float projection of a QAT checkpoint can be recorded as a host diagnostic, alongside the latent float evaluation. |
| Is a code change faster? | Run baseline and candidate with matched model, mode, mini-batch, inputs, initial weights, toolchain and timing scope. Repeat whole runs or matched step measurements. Use the same warm-up, evaluation and logging cadence. [Optimization acceptance](project/V0_DELIVERY_PLAN.md#34-how-do-we-accept-an-optimization-proposed) |
| Is it release evidence? | Apply the release thresholds to the correct held-out float and bit-exact ternary paths and associate the candidate with its parity log. Keep evaluation of the final checkpoint independent of hyperparameter selection. [Release gates](project/V0_DELIVERY_PLAN.md#21-which-results-close-the-release-gate) |

## How should single-thread CPU performance be compared? [PROPOSED]

Compare TARS, Burn, Candle and PyTorch on the same CPU with a single compute
thread. Start with dense `f32` on shared input and parameter fixtures. Record
the backend, version, features, math libraries, internal precision and whether
each row measures eager or compiled execution. The existing QAT training
runner includes batch gathering, weight projection and training. Its throughput
is an end-to-end result for that scope.

| Framework | Initial comparison configuration |
| --- | --- |
| TARS | Optimized build using the existing CPU tensor implementation. Record compiler and CPU target flags. Time a dense float training step separately from the full QAT step. |
| Burn | Use the Flex CPU backend with its SIMD feature and a single compute thread. Record CubeCL CPU separately if tested. [Burn backend and feature documentation](https://burn.dev/docs/burn/#feature-flags) |
| Candle | Use the standard CPU backend with one thread. Record MKL or Accelerate builds as separate configurations when available. [Candle CPU matrix multiplication](https://github.com/huggingface/candle/blob/main/candle-core/src/cpu_backend/mod.rs#L1368-L1454) · [Candle thread controls](https://github.com/huggingface/candle/blob/main/candle-core/src/utils.rs#L373-L391) |
| PyTorch | Use CPU eager execution and `float32` with full internal precision. Set `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` before doing tensor work. Record the linked math libraries and precision settings. [Intra-op threads](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html) · [Inter-op threads](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html) · [oneDNN precision](https://docs.pytorch.org/docs/2.14/notes/mkldnn.html) |

On Linux, choose the same allowed logical CPU for all processes. Keep the SMT
sibling idle when it can be controlled. Pin the already-built executable or
the Python benchmark process. Set worker limits before launch and verify the
backend's thread settings separately from affinity. The example runs the
existing TARS QAT workload and assumes `benchmark_cpu` contains the chosen CPU
index. Use the same prefix for the external benchmark executables or scripts.
[Linux `taskset`](https://man7.org/linux/man-pages/man1/taskset.1.html)

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1 \
    taskset -c "$benchmark_cpu" ./target/release/main
```

| Scope | Workload and units |
| --- | --- |
| Matrix kernels | Start at batch 32 with the first affine layer of the development model. Time `[32,784] × [784,128]`, `[128,32] × [32,784]`, and `[32,128] × [128,784]` in forward, weight-gradient and input-gradient layouts respectively. Run the other layers and repeat with the release model `784 → 64 → 10`. Keep transposes as views where the API permits and account for any explicit packing. Report time per call and, optionally, `2 × M × N × K / elapsed_seconds` for a product of `[M,K]` and `[K,N]`. |
| Ready-model inference | Measure the affine and activation chain with input and weights already loaded. Report batch-1 latency and batch-32 throughput. If rank-1 input takes a different kernel, record it separately from a rank-2 batch of one. |
| Dense float training | Use shared initial parameters and preloaded batches. Time forward, cross-entropy, backward and one matched optimizer update. Request the same gradients and use the same loss reduction and optimizer hyperparameters. Check one-step logits, loss, gradients and parameter updates with a declared float tolerance before timing. Report milliseconds per batch and images per second. |
| QAT and deployment | Compare QAT steps when all paths implement the same effective weights and gradient rule. Measure bit-exact ternary inference on one fixed exported artifact as its own workload. |

Warm up and repeat at each scope. Record raw timing samples and their sampling
unit with the run. Report p95 from individual call or step observations when
available. Report a median of per-call block averages as such when using a
blocked microbenchmark timer. Use the matched end-to-end timing to judge
engine changes and the kernel timing to localize the difference.
[PyTorch benchmark timers](https://docs.pytorch.org/docs/2.15/benchmark_utils.html)

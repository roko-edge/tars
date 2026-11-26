# How does TARS deliver and harden v0.1.0?

**Status: [PROPOSED]** Execution plan for maintainer review. The
[v0.1.0 release contract](V0_CONTRACT.md) owns release acceptance. The
[roadmap](ROADMAP.md) owns work-package assignments. Engineering specifications
own interfaces, arithmetic, and formats.

First freeze a reproducible engineering baseline. Use it to build a candidate
and measure the effect of hardening changes. The release evidence must be
reproducible on the exact revision the maintainer chooses to tag.

| Milestone | Completion criterion |
| --- | --- |
| **Internal baseline** | Pin the existing entry point, objective, optimizer, seeds, data roles and starting model. Verify small-model training and integer fixtures and record development results and remaining gates. Use the pinned run to judge later changes. |
| **Release candidate** | Reproducible float and QAT training, held-out MNIST accuracy at the contractual thresholds, exact Rust–RTL outputs, and closure of the published v0.1.0 scope. |
| **Hardened release** | The supported public entry points are verified against the baseline. Accepted performance and deployment changes have paired evidence. The final revision closes all release gates before the v0.1.0 tag. |

---

## 1. What is the starting point?

The audit covered the local branch at `2a136a6` and the public MNIST merge at
`a49af36`. The following findings define the starting point for this plan.

| Area | Evidence at the audited revisions |
| --- | --- |
| Model execution | `Sequential`, the pure forward pass, the caller-owned `ForwardCache`, analytical backward, and BGD are present. The float training loop still uses MSE. |
| MNIST | PR #35 introduced a loader and runner. The runner trains on the training split, then evaluates the quantized float model on that same split. The loss files are disconnected stubs. |
| Initialization | The MNIST merge seeds a new generator for each draw. A layer receives repeated values. All three math tests fail in the published Rust job. The RTL lint and Tier 1 job passed on the same commit. |
| Quantization | PTQ is present. Export re-quantizes dequantized float weights and consumes a positional scale list. QAT and an independent fixed-point Rust reference remain pending. |
| Parity | The MNIST-sized bench is present. Its expected outputs are float-derived, it allows one LSB, and its signed 32-bit difference can overflow. The stock CI job runs Tier 1. |
| Board and visualization | The audited local lab lacks its expected NPU board wrapper. The public visualization binary is a stub. |

[MNIST merge](https://github.com/roko-edge/tars/commit/a49af36),
[Rust job](https://github.com/roko-edge/tars/actions/runs/44623526976/job/135911185366),
[RTL job](https://github.com/roko-edge/tars/actions/runs/44623526976/job/135911184583)

### 1.1 What did the training and tensor audit find on 2026-11-21?

This inspection covers `e7b7f64` and the MNIST QAT experiment at `4961a9e`.
The table records source observations. Release test and performance evidence
is pending.

| Status | Area | Observed behavior |
| --- | --- | --- |
| [DECIDED] | Float training boundary | Data, `Module::forward`, parameters, the forward cache, parameter gradients, and optimizer state use `Tensor<f32>`. LOG-008 decouples typed T3 storage from the release training, export, and parity path. [Training specification](../engineering/specs/TRAINING.md), [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026) |
| [PENDING] | Packed ternary consumer | `PackedT3Storage` stores 32 trits per `u64`. The mixed product converts each trit to `f32` for multiplication and carries no scale. QAT, PTQ, and export use the scalar ternary encoder. The packed tensor and mixed product have no callers in those paths. [`ternary_tensor.rs`](../../src/tensor/ternary_tensor.rs), [`export.rs`](../../src/export.rs) |
| [CONFLICT] | Backward and loss | `Sequential::forward_cache` records a flat trace and the free backward pass walks it in reverse. The baseline runner uses synthetic MSE targets to compensate for the legacy delta overwrite. The QAT runner passes the selected loss's output gradient directly into backpropagation. The target module-local interface remains open. [`sequential.rs`](../../src/modules/sequential.rs), [`grad.rs`](../../src/optim/grad.rs), [baseline runner](https://github.com/roko-edge/tars/blob/e7b7f64/src/bin/main.rs), [QAT runner](https://github.com/roko-edge/tars/blob/4961a9e/src/bin/main.rs) |
| [CONFLICT] | SoftmaxCE forward and backward | The loss averages over the batch and its backward returns $(p-y)/B$ for normalized targets. Forward clamps probabilities before taking the logarithm, so the reported scalar and gradient disagree for sufficiently unfavorable logits. Use log-sum-exp and verify the pair on a deterministic float fixture. [`loss.rs`](../../src/optim/loss.rs), [training specification](../engineering/specs/TRAINING.md#12-loss-function-contract) |
| [CONFLICT] | MNIST baseline | Both inspected paths use Adam and evaluate on training data. The baseline factory uses a single affine layer and the QAT factory uses `784 → 128 → 64 → 10`. The release calls for BGD, held-out evaluation, and the assembled `784 → 64 → 10` topology. The QAT runner reports MSE on raw logits and averages the batch means. [Historical runner](https://github.com/roko-edge/tars/blob/4961a9e/src/bin/main.rs), [historical factory](https://github.com/roko-edge/tars/blob/4961a9e/src/experiments/mnist.rs), [release contract](V0_CONTRACT.md) |
| [PENDING] | Independent MNIST evaluation | At `4961a9e`, the factory builds its dataset from `train_data` and `train_labels`. The runner passes that same dataset to batching and to both final accuracy calls. The printed accuracies are training diagnostics. The release requires the [held-out evaluation protocol](#231-how-do-we-keep-the-final-mnist-evaluation-independent-proposed). [Historical factory](https://github.com/roko-edge/tars/blob/4961a9e/src/experiments/mnist.rs), [historical runner](https://github.com/roko-edge/tars/blob/4961a9e/src/bin/main.rs) |
| [PENDING] | QAT and exact parity | QAT runs forward and backward with scaled ternary weights represented in `f32` and applies the gradient to latent weights. The independent fixed-point reference and release validation remain pending. `export_inference` serializes float predictions, and Tier 2 permits a one-LSB difference. [Historical runner](https://github.com/roko-edge/tars/blob/4961a9e/src/bin/main.rs), [`export.rs`](../../src/export.rs), [`tb_parity.sv`](../../npu/tests/tb_parity.sv) |
| [PENDING] | Dense tensor fast paths | The committed core has logical dense slices and an `FnMut` mapping path with strided fallback. `is_contiguous` allocates stride metadata, and consumers such as `sum` use a logical iterator that panics for rank zero. [`tensor_core.rs`](../../src/tensor/tensor_core.rs) |

### 1.2 What changed in the inspected MNIST working tree on 2026-11-22?

In the earlier local snapshot over `4961a9e`, the factory builds `784 → 64 → 10`
and the runner uses float training with Adam. It passes the first 50,000
samples from the MNIST training files to the loader and evaluates the float
checkpoint and its PTQ projection on the remaining samples from those files.
These are source observations. The official test split and release run are
still [PENDING].

The subsequent trainer integration is recorded in §1.3.

| Status | Finding |
| --- | --- |
| [CONFLICT] | The runner logs an unweighted mean of batch MSE values on logits while optimizing SoftmaxCE. Its throughput numerator includes the full factory dataset, including the suffix reserved for evaluation. With 50,000 fit samples and batch size 64, a complete epoch has 782 updates and a final batch of 16. |
| [PENDING] | Initialization advances the generator across draws and neuron rows within a layer. Each layer restarts from the same fixed seed. The pending change is a configurable model-wide initialization stream. The ternary distribution test still calls the float sampler. |
| [PENDING] | The compiled QAT branch at `4961a9e` and the active float branch in the working tree are represented by successive source snapshots. A single runnable entry point that selects either mode is part of the [proposed API refinement](#351-how-should-the-public-training-path-work-proposed). |

[Historical runner](https://github.com/roko-edge/tars/blob/4961a9e/src/bin/main.rs) · [Historical factory](https://github.com/roko-edge/tars/blob/4961a9e/src/experiments/mnist.rs) ·
[`linear.rs`](../../src/modules/linear.rs) · [`math_tests.rs`](../../tests/math_tests.rs)

### 1.3 What changed with the reusable trainer?

| Revision or snapshot | MNIST preset and runner behavior |
| --- | --- |
| `e7b7f64` | Single affine `784 → 10`, 10 epochs, learning rate 0.003, Adam with batches of 512. The runner derives synthetic MSE targets from the SoftmaxCE gradient. |
| `4961a9e` | `784 → 128 → 64 → 10` with hidden ReLUs, 5 epochs, learning rate 0.001, Adam with batches of 32. QAT updates latent float weights. The `Ternary PTQ` label describes the float projection of a QAT checkpoint. |
| Earlier local inspection on 2026-11-22 | `784 → 64 → 10` with hidden ReLU, 100 epochs, learning rate 0.001. Float training with Adam, batches of 64, and a reserved evaluation suffix. |
| `7dd519d` | Same topology and split, 50 epochs, learning rate 0.001. `Trainer::fit` hardcodes batches of 64. The earlier API record reports `cargo check --all-targets` succeeded for this snapshot. |
| `f72bca4` (`origin/main` at review) | Integrates the reusable trainer, effective batch-size configuration, selected-loss reporting, and updated preset. |
| `6cb1a0b` (reviewed 2026-11-26) | Moves the runners to `examples/`. MNIST trains `784 → 64 → 10` with Adam and QAT, uses the official training and test splits, and reports the latent and projected models in float arithmetic. `Trainer::fit` returns `TrainSummary` and reuses the QAT projection buffer. [Current API](../API.md#3-training-operations) · [Current preset](../EXPERIMENTS.md#built-in-examples) |

The pre-trainer QAT and float snapshots logged MSE on logits while optimizing
SoftmaxCE. The local snapshot in §1.2 also included evaluation samples in its
throughput numerator. Interpret old logs against their own source and run inputs.
[Trainer integration](https://github.com/roko-edge/tars/pull/44)

---

## 2. What must be true to deliver v0.1.0?

### 2.1 Which results close the release gate?

| Status | Gate | Required evidence |
| --- | --- | --- |
| [DECIDED] | Float MNIST accuracy **≥ 95%** | Correct and total counts for the held-out evaluation split, tied to the float checkpoint and run configuration. |
| [DECIDED] | Ternary MNIST accuracy **> 90%** | Correct and total counts from the bit-exact fixed-point path, using the same held-out sample IDs. Report the 95% binomial confidence interval and the paired McNemar comparison specified by the contract. |
| [DECIDED] | Rust–RTL parity with **zero mismatched output words** | Tier 2 log for the trained dense ternary `784 → 64 → 10` model, exact fixture identity and output count, and successful Tier 1 and RTL lint on the release revision. |
| [DECIDED] | QAT and analytical gradient verification | A run that updates latent float parameters through the specified STE, MSE regression for OR/XOR, and finite-difference verification on the float graph with quantization disabled. |
| [DECIDED] | Visualization and research evidence | The visualization scope and engineering evidence in §§3–4 of the release contract, or a maintainer-approved amendment to that scope. |
| [PENDING] | Remaining capability clauses in the release contract | Resolve the conflicts in §4 and verify every clause in the final scope on the candidate revision. |

The model can hit both accuracy targets before the full release contract closes.
The confidence interval accompanies the ternary result. The contract applies
its threshold to the measured accuracy. [Release contract](V0_CONTRACT.md#1-primary-benchmark-goal)

### 2.2 How do we restore trustworthy training? [PENDING]

| Order | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | **Complete RNG ownership.** Extend the existing per-layer stream to the model so initialization advances across layers. Give initialization and shuffling explicit, recorded seeds. Fix the ternary distribution test to exercise the ternary sampler. | The same algorithm and seed reproduce the sequence, successive draws vary, and the initializer produces distinct neuron rows and distinct layers of equal shape on a fixed fixture. The Rust gate passes without retries. Record the locked dependencies and execution environment for byte-level reproducibility. |
| 2 | **Pin the small-model baseline.** Preserve sigmoid/MSE behavior on OR and XOR. Keep their training commands independent of MNIST downloads and export side effects. | Deterministic fixture for predictions, loss, and a parameter update. Repeated runs reproduce the expected result under the recorded toolchain. |
| 3 | **Finish loss-boundary validation.** The analytical pass already calls `Loss::backward` for the initial delta. Align the SoftmaxCE scalar and gradient at extreme logits, define matching shapes and batch reductions, and diagnose invalid inputs. [Current implementation](../API.md#3-training-operations) | MSE divides by the number of elements. For one-hot targets, batch-mean SoftmaxCE accepts raw logits, uses a stable max-shift/log-sum-exp computation, and returns `(p - y) / B`. Tests cover large logits, the final short batch, mismatched inputs, and finite differences. |
| 4 | **Verify the model gradient and optimizer.** Use a deterministic multilayer float fixture with ReLU inputs away from zero. Check both weight and bias gradients. Validate the model–gradient pairing before applying BGD updates. | The roadmap sets a finite-difference tolerance of $\epsilon \le 10^{-4}$. Record the perturbation and error metric alongside the result. An invalid gradient is rejected before changing parameters. |
| 5 | **Finish the MNIST run path.** Use the existing reader, enforce the split roles in Section 2.3.1, and validate image and label counts, dimensions and ranges before pairing. Record the training mode and label the actual loss, split and inference path in the metrics. | The runner records dataset identity, topology, seeds, batch size, learning rate, optimizer, epochs, checkpoint identity, and integer accuracy counts. It can evaluate a saved checkpoint without retraining. Keep exported parity samples identifiable. |

For the first valid baseline, keep BGD and make initialization and shuffle fixes
independent of optimizer or kernel changes. A later initializer or optimizer
experiment should compare against that baseline on the same data protocol.
[Training specification](../engineering/specs/TRAINING.md)

### 2.3 How do we measure and improve quantized accuracy? [PENDING]

| Stage | What runs | Evidence to retain |
| --- | --- | --- |
| Float baseline | Train the `784 → 64 → 10` ReLU/identity model with SoftmaxCE on logits. Choose hyperparameters using a documented development split or predeclared configuration. | Float checkpoint, settings, and predictions keyed by held-out sample ID. Define the argmax tie rule and use it for each evaluation. |
| PTQ baseline | Extract canonical dense trits and one $W_0$ per layer from the float checkpoint. Run the fixed-point evaluator on the serialized-domain model. | Per-layer scales, checkpoint and artifact identities, saturation diagnostics when needed, and per-sample predictions on the same split. |
| QAT | Start from a known float checkpoint. Use the same float forward and backward kernels with effective weights $\widehat W = W_0 Q(W)$ materialized as `Tensor<f32>`. Keep the effective weights available for the layer input gradient. Route the effective weight gradient to the latent `Tensor<f32>` through the identity STE, then update the latent weights. | A deterministic small fixture verifies the input gradient against the effective weights and the optimizer update against the latent weights. Verify the full analytical gradient with finite differences on the float graph. Record the QAT checkpoint and evaluate the compiled result with the integer reference. |
| Final comparison | Evaluate the frozen float and selected QAT checkpoints on the same held-out sample IDs. | Counts and accuracy, ternary confidence interval, paired disagreement counts for McNemar, and the exact script, command, and checkpoint IDs. A confusion matrix can explain class-level changes. |

Each layer should produce one canonical codes-and-scale value. QAT
dequantization, hardware export, and fixed-point evaluation derive from that
value or its serialized representation. Resolve the half-step and small-scale
rules in §4 before treating encoded weights as reference evidence.
[Quantization specification](../engineering/specs/QUANT.md)

[PROPOSED] Keep strides, views, and copy-on-write in the float training path.
Treat packed ternary storage and a mixed-precision product as inference
implementation options. Select them when a runtime consumer and a measured
memory or latency requirement justify them. The independent integer reference
can consume canonical codes and Q8.24 words directly. Evaluate any removal of the
experimental tensor API as a compatibility decision for its callers.

#### 2.3.1 How do we keep the final MNIST evaluation independent? [PROPOSED]

Split the official MNIST training data into disjoint fit and development
partitions using a deterministic rule. Reserve the official test split for
the final release evaluation. Record dataset hashes and partition membership
or its exact derivation.

| Partition | Allowed use |
| --- | --- |
| Fit | Parameter updates and fitting any data-dependent preprocessing or quantizer calibration. |
| Development | Select the architecture, quantizer settings and checkpoint or stopping rule. |
| Final held-out test | Evaluate the frozen float checkpoint and exported PTQ and QAT artifacts on identical sample IDs and labels. Use the serialized-domain integer evaluator for PTQ and QAT. |

Verify split membership and exactly one prediction per held-out sample and
path. Paired comparisons require the same sample IDs in every evaluated path.
Freeze the selection protocol and checkpoints before final evaluation.
Record training-set accuracy separately as a diagnostic. Keep the Tier 2
parity fixture separately identified because it verifies exact words on a
declared subset. [Run records and metrics](../EXPERIMENTS.md#how-should-benchmark-runs-be-recorded-proposed)

### 2.4 How do export and RTL establish exact parity? [PENDING]

| Order | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | **Validate an assembled model before emission.** Check supported activations, connected layer dimensions, weight order, one scale per layer, exact bias sizes, and input shape. Export the layer manifest and data to an explicit run directory. | Invalid input fails before publishing a set of artifacts. The manifest identifies the assembled topology and activation choices. XOR exports and MNIST parity fixtures have independent destinations. The existing wire format changes only through Engineering Invariant 4. |
| 2 | **Implement the independent integer reference.** Consume signed Q8.24 inputs, biases, and scales plus the canonical ternary codes. Apply signed accumulation, full-width multiplication, floor shift, pre-bias saturation, bias addition, post-bias saturation, and activation in the specified order. | Hand-calculated fixtures cover negative fractional rescaling, the two distinct saturations, the minimum signed activation, both encodings of zero, and intermediate quantization across two layers. The same reference evaluates held-out accuracy and generates expected parity words. |
| 3 | **Make the testbench strict.** Compare the complete 32-bit words. Check missing, malformed, short, and extra inputs in the fixture reader or preflight. Drive stimulus away from the sampling edge and require one completion per accepted request. | A deliberately corrupted expected word fails. Missing or inconsistent data fails before comparison. Logs identify the sample and output index on mismatch and report total comparisons and mismatches. |
| 4 | **Run exact parity at two scales.** Use an inexpensive deterministic integration fixture in CI as proposed below, then run the trained-model MNIST fixture for release. | A clean checkout can reproduce the small fixture. The release bundle pins selected sample IDs, model words, expected output words, and simulator version. Lint, Tier 1, and Tier 2 results are attached to the tested revision. |

Tier 1 should check the [start/done contract](../engineering/PROTOCOL.md)
with consecutive requests, a request while busy, and a request in the completion
cycle. Record the accepted and completed request count and check that the result
is complete on `done`. Tier 2 should check the assembled model end to end.
Keep one reproducible generation command for its exact fixture. The legacy
smoke inputs need a replacement consumer and a verified fixture before their
tracked copies can be retired.

The current MNIST bench fixes 100 samples. With that configuration, preflight
expects 50,816 weight codes, two scales, 74 biases, 78,400 activation words, and
1,000 expected output words. The accuracy evaluation is a separate, larger run
on the held-out split. A useful diagnostic fixture has a raw accumulator of
$-1$ and a scale of $0.5$. The epilogue floor gives raw $-1$, while float
serialization of that product gives zero. [Parity bench](../../npu/tests/tb_parity.sv),
[arithmetic contract](../engineering/specs/ARITHMETIC_Q8_24.md),
[memory format](../engineering/MEM_FORMAT.md)

### 2.5 What do we do when a result falls short? [PROPOSED]

| Observation | Diagnose next |
| --- | --- |
| Float accuracy is below threshold | Check initialization diversity, image/label alignment, loss normalization, and finite differences. Change training settings only after the small fixtures give expected results. |
| Float passes and ternary accuracy drops sharply | Compare PTQ and QAT on identical samples. Inspect scales, trit distribution, saturation and the first layer where outputs diverge. Hold the wire contract fixed during training experiments. |
| Rust and RTL diverge | Freeze the failing artifact. Locate the first differing sample, layer and neuron. Check artifact ordering, then accumulation, rescale, saturation, bias and activation in that order. Reduce the case to an independently calculable fixture. |
| A rerun produces different results | Compare artifact and dataset identities, both seeds, batch order, toolchain and feature flags. Establish the baseline before tuning. |
| Accuracy and parity pass while published requirements remain open | Record the result as a candidate and close the remaining capability or scope decisions in §4 before tagging. |

---

## 3. What makes the candidate ready for long-term use?

Keep the public library small, make the supported operations predictable, and
remove work from the full training step. Performance work can start from a
reproducible baseline while release validation proceeds. The final candidate
must preserve the release evidence and include paired measurements for any
performance claims.

### 3.1 How should the tensor core handle views and bulk operations? [PROPOSED]

| Sequence | Design | Acceptance evidence |
| --- | --- | --- |
| 1 | Fix logical traversal before adding fast paths. A rank-zero tensor has one element. An empty extent has zero. Compare the stored strides with a running expected stride, from the last axis to the first. Retain the current offset check. | Tests cover scalars, empty extents, rectangular transposes, round-trip materialization, and storage larger than `numel()`. A borrowed dense slice contains exactly the logical extent. The contiguity query makes no heap allocation. |
| 2 | Verify and benchmark the existing `map` paths over a borrowed dense slice and a stride-aware fallback. Preserve logical row-major order, direct construction of the canonical output, and support for stateful callbacks. | Dense and strided results agree with hand-calculated examples. A stateful closure observes the same order. The fallback has no temporary packed input and the output avoids an initial zero fill. |
| 3 | Introduce crate-internal `zip_map` for equal-shape element-wise kernels. Zip borrowed slices when both operands are dense and traverse the logical coordinates otherwise. Reuse it in the existing arithmetic API. Express row-bias addition and batch reduction as named operations. | Shape mismatch fails before the callback, including different shapes with equal element counts. Test both-contiguous, mixed-layout, and both-transposed pairs. Shape, traversal order, and results agree with the scalar reference. |
| 4 | Reuse existing mutable methods and add an internal `zip_map_mut` where backward or BGD can actually reuse a destination. Acquire mutable dense storage once. Preserve layout and copy-on-write behavior. | Tests confirm unchanged sibling views and retained destination strides. Measurements distinguish unique and shared storage and account for the full update, including any temporary allocation. |
| 5 | Preserve metadata-only `transpose` for tensor data. Use `make_contiguous` at a consumer that needs canonical storage or after measurement shows that reuse pays for packing. Give packed T3 its own storage access capability. | The dense fast path is available only for storage that can expose a real element slice. The strided path works without forced input copies. Packed and wire representations have their own fixtures. |

Keep bulk storage access within the engine, exporter, and renderer boundaries
defined in the project architecture. Arbitrary Rust closures are a host CPU
convenience. Keep model operations behind named kernels with stable semantics
for later execution targets. Measure a simple safe loop before adding axis
coalescing, manual SIMD, or further layout types.
[Tensor specification](../engineering/specs/TENSOR.md),
[tensor delivery plan](TENSOR_PHASE_B.md),
[future execution targets](../engineering/FUTURE_TARGETS.md)

### 3.2 How should training and matrix kernels get faster? [PROPOSED]

Remove discarded products and full-buffer copies first. Use the results from
those changes to profile and prioritize affine kernels, QAT scratch reuse,
and the smaller element-wise operations. Land changes with one measurable
hypothesis per revision.

| Order | Change | Evidence required |
| --- | --- | --- |
| 1 | Build each parameter gradient once. Stop propagating at the first trainable layer when backward returns only parameter gradients. Apply each activation derivative directly to the owned delta. Use an explicit batch reduction for `db`. | The same weight and bias gradients for OR, XOR and multilayer float fixtures. Check rank 1 and batch use against the accepted API, and test an explicit input-gradient request if exposed. Measure complete backward time and allocations. [`grad.rs`](../../src/optim/grad.rs) |
| 2 | Gather directly from borrowed samples using slices of one shuffled index buffer. Transfer the completed input and target buffers into the batch tensors. Flatten parameter initialization in a separate change and preserve draw order during that refactor. | The same seeded sample order, values and shapes, including the final short batch. Measure gather time, allocation count and temporary bytes. Compare changes to the random stream separately. [`data.rs`](../../src/data.rs), [`linear.rs`](../../src/modules/linear.rs) |
| 3 | Specialize matrix products for the actual access patterns below. Validate geometry once per kernel and resolve strides outside the inner loop. Add row bias after the dot product to preserve the current operation order. | Match a stride-aware scalar reference and test the effect on the full step. Account for any packing, allocation and setup. [`float_tensor.rs`](../../src/tensor/float_tensor.rs) |
| 4 | In QAT, populate effective float weights directly from the latent parameters. Reuse one buffer per linear layer when ownership allows. Keep that batch's effective weights for the corresponding input gradient. The current quantizer clones the layer and detaches shared weight storage on the first write. | Identical trits, scales and effective weights under the selected quantizer rules. A fixed batch gives the same logits, gradients and latent update. Measure projection and the complete QAT step. [`sequential.rs`](../../src/modules/sequential.rs), [`linear.rs`](../../src/modules/linear.rs) |
| 5 | After fixing the SoftmaxCE scalar, fuse loss and initial delta over each logits row so they share one exponentiation pass. Weight telemetry by the objective's denominator. Use named in-place update kernels for BGD and, if retained as a supported optimizer, Adam. | Extreme and translated logits, normalized targets and short batches agree with a reference. Optimizer fixtures cover weight and bias updates, multi-step Adam state and rejection of mismatches before mutation. Measure telemetry at a matched cadence. [`loss.rs`](../../src/optim/loss.rs), [`bgd.rs`](../../src/optim/bgd.rs), [`adam.rs`](../../src/optim/adam.rs) |

| Affine operation | First kernel to compare |
| --- | --- |
| Forward $XW^T$ | Dot products over contiguous rows of $X$ and $W$ when both layouts permit it. Use the strided path for general views. Keep the accumulation over input features in order. |
| Weight gradient $D^TX$ | Accumulate into blocks of output rows with contiguous segments of $X$. Advance the batch index in order for each output element. |
| Input gradient $DW$ | Walk each row of $W$ contiguously while accumulating into a row or a block of the output. Compare blocking sizes on both release layers. Keep this product at layer boundaries that consume it. |

For a first affine layer with $K$ inputs, $N$ outputs and batch size $B$,
skipping its unused input gradient removes $BKN$ multiply-accumulate terms
and the corresponding $[B,K]$ output buffer. This operation count informs
priority. The paired step benchmark determines the elapsed-time gain.

Use a straightforward safe implementation as the comparison point. SIMD,
external matrix libraries, alternative accumulation orders and kernel-local
packing are follow-up experiments when the profile identifies a remaining
bottleneck. Fusion and scratch reuse should keep the live range of saved
activations and effective weights explicit.

#### 3.2.1 Which training optimizations belong in v0.1.0? [DECIDED]

**Maintainer decision (2026-11-26):** CPU training throughput is not a v0.1.0 release gate. Only loss correctness and backward-pass cleanup tied to open release conflicts are required for v0.1.0; slice-based data, optimizer, and bias cleanups are optional non-blocking improvements; `MatMul` tiling, multi-threading, and manual SIMD kernels are deferred beyond v0.1.0.

| Scope | Target | Disposition for v0.1.0 | Notes |
| --- | --- | --- | --- |
| Required (conflict closure) | `SoftmaxCE` (`src/optim/loss.rs`, `src/train.rs`) | **In scope** | Reconcile `forward` and `backward` via stable log-sum-exp (`[CONFLICT]`). Operating on contiguous slices and avoiding duplicate `softmax` passes belong in the same change. |
| Required (conflict closure) | `backward` (`src/optim/grad.rs`) | **In scope** | Paired with resolving the free vs. module-local `backward` `[CONFLICT]`. Drop redundant `Tensor::zeros` pre-allocation, use slice reduction for `db`, and skip unused input-gradient `delta.matmul(&linear.weights)` at layer `0`. |
| Optional (non-blocking) | `DataMiniBatches` / `Batch::from_samples` (`src/data.rs`) | **Optional if time permits** | Gather via `as_slice()` / `extend_from_slice` without per-epoch `Vec<Data>` clones or strided `get` lookups; does not block v0.1.0. |
| Optional (non-blocking) | `Adam::step` (`src/optim/adam.rs`) | **Optional if time permits** | Iterate `as_slice()` / `as_mut_slice()` instead of per-weight strided `get`/`set`; does not block v0.1.0 (BGD remains baseline optimizer). |
| Optional (non-blocking) | `Linear::forward` batch bias (`src/modules/linear.rs`) | **Optional if time permits** | Replace per-element `get`/`set` bias addition with row-wise `chunks_exact_mut` slice updates; does not block v0.1.0. |
| Deferred (post-v0.1.0) | `MatMul` micro-kernels (`src/tensor/float_tensor.rs`) | **Deferred** | No cache tiling, custom SIMD micro-kernels, or multi-threading in v0.1.0; revisit only after cleanups 1–4 if CPU profiling still warrants it (GPU training lands in v0.3.0). |

### 3.3 What should Roko be able to consume? [PROPOSED]

Use the versioned inference artifact as the integration boundary. The public
v0.1.0 design should identify the topology, supported activations, ternary
weights, Q8.24 scales and biases, and wire-format revision. The same validated
data feeds the Rust reference and the assembled RTL wrapper. Run metadata can
carry provenance and hashes alongside the existing files. Adopt any new
serialized field only through the format revision process.

| Work | Evidence of readiness |
| --- | --- |
| Isolate host tooling from the execution path. | The evaluator consumes verified model data and activations through a small interface. Dataset downloading, optimization, plotting, and CLI handling stay outside it. |
| Demonstrate the real board path when a physical deployment claim is in scope. | The local wrapper loads and supplies the exported weights and activations, the canonical RTL source order is used, and known inputs match expected outputs on the board. Record tool, part, constraints, post-route slack and utilization for that exact build. |
| Report deployment costs with their boundaries. | Record artifact bytes, parameter payload bytes, peak working memory, and inference latency or cycles. Distinguish core-only results from end-to-end results that include data movement. Keep simulation, synthesis estimates, and measurements separately labeled. |
| Preserve the iteration path beyond v0.1.0. | The assembled wrapper ships with the v0.1.0 model. The v0.1.1 sequencer can consume the same layer semantics. Later CPU or GPU paths can be compared against the deterministic reference without changing artifact meaning. |

The documented 100 MHz lab clock is a constraint for a board build. Timing
closure and maximum frequency require evidence for the actual routed design.
Dense ternary's resource and weight representation claims use matched baselines.
The contract reserves the zero-skipping latency claim for v0.2.0.
[Board integration](../engineering/BOARD_INTEGRATION.md),
[release claim discipline](V0_CONTRACT.md#63-research-claim-discipline)

### 3.4 How do we accept an optimization? [PROPOSED]

| Check | Required comparison |
| --- | --- |
| Correctness first | Pass the same reference fixtures, deterministic training checks and relevant RTL checks before interpreting timing. For a work-removal refactor with unchanged arithmetic order, compare float outputs, gradients and updated parameters bit for bit on a pinned CPU fixture. Evaluate any deliberate floating-point change separately. Changes affecting the selected checkpoint or exported words regenerate candidate evidence. |
| Compare one change | Pin baseline and candidate revisions, dataset and seeds, compiler and target, feature flags, release profile, and machine. Hold the model, optimizer, arithmetic and logging settings fixed unless one is the subject of the comparison. For local kernels include small XOR shapes and MNIST training shapes. |
| Measure the whole cost | Warm up and repeat. Record median and p95 step latency, samples per second, allocation count and peak temporary bytes. Measure isolated kernels and the full step. Count packing and buffer preparation in end-to-end timings and report reuse separately. |
| Decide before merging | State the primary metric and reference workloads before the run. Use repeated baseline runs to establish ordinary variance. Require a reproducible improvement in the target metric for a performance claim and check the protected workloads for regressions beyond that variance. Keep the simpler implementation when a more complex kernel has no repeatable gain. |
| Keep the run reproducible | Retain commands, seed, input hashes and raw measurements under the project evidence policy. Generated `.mem`, waveforms, bitstreams and build products stay outside version control. [Run record and metrics](../EXPERIMENTS.md#how-should-benchmark-runs-be-recorded-proposed) |

| Workload | Comparisons to run |
| --- | --- |
| Element-wise | Dense versus transposed and mixed-layout inputs. Compare direct traversal with one-time explicit packing over one use and repeated reuse. Benchmark unique and shared destinations separately for mutation. |
| Affine layer | For each inner/output pair $(K,N)$ in $\{(784,64),(64,10)\}$, measure forward $[B,K]\times[K,N]$, weight gradient $[N,B]\times[B,K]$, and input gradient $[B,N]\times[N,K]$. Use batch size 1, the configured training batch size, and the final short batch when present. |
| Full training step | Use matched starting parameters and a seeded sequence of batches in float and QAT. Measure batch gather, weight projection when active, cached forward, loss, backward and optimizer together. Report telemetry with the same cadence and include the last short batch. |
| Deployment | Measure artifact read and validation, inference with a ready model, and the full input-to-output path as separate timings. Report clock and target with cycle counts. |

### 3.5 What makes the Rust API small and complete? [PROPOSED]

Give each operation one documented home and make common operations usable
through the public library. Exercise the public API with an external-crate
example or integration fixture for the supported model, training and deployment
paths. For a signature change, reconcile the affected specification conflicts
recorded in Section 4.

| Boundary | Design and acceptance |
| --- | --- |
| Composition | Keep fluent construction and a pure `forward`. Fold a flat sequence in forward order and walk the saved trace in reverse for backward. Apply each layer's calculus in that layer. Introduce actual recursive descent when an admitted composite owns other composites. The traced and plain forwards agree for the same input and parameters. |
| Training | The loss owns its reduction and output gradient. The composite owns trace order. The optimizer owns parameter updates and any moments tied to the model's parameter order. Validate shapes and topology at construction or step entry so the numeric loop can assume them. Exercise the public calls from a deterministic small model through one complete update, including invalid shape and state cases. |
| Tensor engine | Keep `map` and internal `zip_map` available for CPU implementation work. Expose named tensor operations for model code so storage, indexing and CPU kernels stay behind the operation boundary. Describe logical order, accepted ranks and layouts, shape errors and allocation behavior in the API documentation. |
| Public exports | Inventory `lib.rs` re-exports against the documented entry points. Make deliberate exports for the supported surface and use the existing module paths during a compatibility transition. Resolve duplicate entry points, empty diagnostics and stubs in the area being changed. Public API examples should compile using the documented paths. [`lib.rs`](../../src/lib.rs), [API reference](../API.md) |
| Scope | Favor a complete, tested path for `Tensor<f32>`, sequential affine layers and activations, the supported losses and optimizer, and ternary export with exact CPU evaluation. Add a new public abstraction when a supported operation or a documented consumer needs it. Keep generalized graphs and devices on their existing milestones. [Release boundaries](V0_CONTRACT.md#61-version-boundaries), [execution targets](../engineering/FUTURE_TARGETS.md#3-invariance-rules) |

#### 3.5.1 How should the public training path work? [PROPOSED]

Keep the fluent sequential chain and expose one reusable training loop. Prove
float, PTQ, and QAT with the existing execution paths first. Use those fixtures
to verify the public API during hardening, before tagging v0.1.0. The proposed
signatures below are subject to maintainer review.

| Boundary | Proposed behavior and trade-off |
| --- | --- |
| Model construction [DECIDED] | Preserve `Sequential::new(784, 64).relu().linear(10)` as a complete construction chain. The constructor creates the first affine layer and every call returns `Sequential`. Direct calls check positive widths and allocation sizes before creating a layer. Configuration or import entry points return errors for invalid external dimensions. The initialization seed API remains [PENDING]. [Construction contract](../engineering/specs/MODEL_TRAIT.md#22-sequential-as-a-composite-module) |
| Model invariants | Keep topology and parameter shapes private. Public mutation preserves those shapes or validates replacements. A caller can currently clear `Sequential::modules` or replace a `Linear` weight tensor directly. |
| Training configuration | `Trainer::new(model, loss, optimizer)` binds the objective and update rule explicitly. `.batch_size(...)`, `.shuffle_seed(...)`, and `.mode(TrainingMode::Qat)` set run options. Use float mode by default and document any batch and seed defaults. Pass the epoch count to `fit` so the run budget is visible. The release preset explicitly selects `BGD`, which applies one update per supplied batch. Keep Adam available for comparison after its state and step are verified. |
| Ownership and errors [PROPOSED] | `fit(self, fit_data, epochs)` borrows samples and returns a `Result` containing the trained latent model and a report on success. The run consumes the trainer on an error as well. If the release needs repeated calls that continue the same session, consider `fit(&mut self, ...)` with explicit ownership of the model, optimizer moments, and shuffle position. Bind optimizer state to the model and preflight layout, configuration, and target shape before updating weights. A numerical failure during a run identifies the epoch and batch. |
| Manual training | Use the same forward trace, analytical backward pass, and optimizer step in the trainer and in the manual path. A manual caller can inspect the gradients before stepping. Reuse the optimizer interface specified in the [training contract](../engineering/specs/TRAINING.md#13-optimizer-contract). |
| Quantization-aware training | Select `TrainingMode::Qat` explicitly. The release QAT run starts from the frozen float checkpoint. Use effective weights from the current batch for forward and input gradients and apply the STE update to latent weights. Record starting checkpoint, run mode, and optimizer-state policy. A fresh trainer starts with fresh optimizer state. Keep float training, PTQ evaluation of the float baseline, and QAT as separately identified runs or evaluation paths. |
| Reporting and evaluation | The report includes the actual loss, reduction, processed sample and update counts, epoch history and defined timing windows. Aggregate loss using the objective's denominator. Provide classification evaluation with `correct`, `total`, a deterministic tie rule and explicit split and inference path. The generic trainer can also run MSE on OR and XOR. The release runner calls the final evaluation on frozen models under the protocol in Section 2.3.1. Keep terminal rendering at the caller. |
| Model summary | Use `Display` for the topology, output widths, weight and bias counts, and logical float payload. Derive ternary sparsity from emitted zero trits, with per-layer and count-weighted totals. For `784 → 64 → 10`, there are 50,816 weights and 74 biases. Logical float parameters occupy 203,560 bytes. Ideal packed weight codes occupy 12,704 bytes, plus 304 bytes for the two scales and 74 biases in Q8.24. Report text `.mem` size and runtime footprint separately. |
| Deployment | `compile_npu()` produces one checked artifact for the assembled target. Confirm layer count, dimensions, activation mapping, canonical trits and Q8.24 words. Use the same artifact for integer evaluation and file emission. Offer `export_npu(dir)` as a convenience that calls the same compiler and writer. Publish the artifact set after all writes succeed. Return errors for unsupported models or I/O failures. [Export and parity work](#24-how-do-export-and-rtl-establish-exact-parity-pending) |

Keras illustrates inferred intermediate widths, explicit training setup and a
structured `fit` history. Candle demonstrates binding optimizer configuration
to parameters before stepping.
[Keras Sequential](https://keras.io/api/models/sequential/) ·
[Keras training API](https://keras.io/api/models/model_training_apis/) ·
[Candle optimizer](https://docs.rs/candle-nn/latest/candle_nn/optim/trait.Optimizer.html)

#### 3.5.2 Which cases should the API fixtures cover? [PROPOSED]

| Area | Cases to verify |
| --- | --- |
| Construction and import | The model can run a forward pass immediately after creating its first affine layer. Appending a linear layer infers its input width. Verify that direct calls diagnose zero widths and size overflow before allocation and that import returns errors for invalid dimensions and parameter shapes. Forward results match between traced and plain execution. |
| Data and training | Reject empty fit data, zero epochs or batch size, an invalid learning rate, mismatched counts or widths, invalid class targets, and non-finite values. A batch size above the sample count produces one update. A short final batch contributes its actual weight to the epoch metric. Preserve bias updates, repeated seeds, and OR/XOR MSE behavior. |
| Loss and optimizer | Verify extreme and translated logits and both analytical gradients. Keep the existing dense one-hot MNIST targets for the first path. A future class-index loss can eliminate those target buffers if a benchmark justifies the API change. A gradient or optimizer layout mismatch fails before any parameter mutation. |
| Manual and trainer paths | With matching initial parameters and optimizer state, one float batch produces the same updated parameters through the trainer and the manual path. Verify the QAT step separately against its effective-weight and latent-update fixture. |
| Evaluation | Validate the data and handle ties consistently across float and fixed-point paths. A saturated or zero-scale model can produce ties. Empty data must yield an explicit error. Use identical reserved IDs for the release comparison. |
| Export | Reject a host-valid but unsupported NPU topology, activation, or non-finite parameter. Verify the all-zero layer and the agreed Q8.24 encoding edge cases. Exercise an unwritable destination and make incomplete artifact sets unusable by the consumer. Compare the complete emitted words with the configured wrapper. |
| Performance | Measure allocations after warm-up over complete float and QAT steps, with equal starting parameters and batch order. First remove discarded gradients and gather copies per Section 3.2. Extend reuse through activations, deltas, loss scratch, and effective weights where the measurements justify it. Include the short batch and check copy-on-write detachment. Count a zero-allocation step only after observing it across that entire path. |

---

## 4. Which decisions need an explicit disposition?

| Status | Issue | Canonical record |
| --- | --- | --- |
| [CONFLICT] | **Quantizer ties and tiny scales.** LOG-008 specifies ties away from zero and the spec separately gives strict comparisons. The implementation uses strict comparisons and a small-scale cutoff. Confirm the exact behavior, including $W_0 = 0$ and positive scales near the cutoff, before freezing release fixtures. | [Quantization specification](../engineering/specs/QUANT.md) and [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [CONFLICT] | **External Q8.24 encoding versus internal rescaling.** The arithmetic document records both a target ties-to-even encoder and current normative truncate-toward-zero serialization. State the selected release encoder separately from the mandatory arithmetic floor inside the epilogue. The release contract specifies a signed 64-bit accumulator and two saturations in the epilogue. | [Q8.24 arithmetic specification](../engineering/specs/ARITHMETIC_Q8_24.md) · [Release contract §2.3](V0_CONTRACT.md#23-respective-npu-hardware-npu) |
| [DECIDED] | **Program and wrapper scope.** v0.1.0 uses the manually assembled two-layer wrapper. `program.mem` and runtime shape controls belong to the v0.1.1 programmable top. | [Release contract](V0_CONTRACT.md#61-version-boundaries) · [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [PENDING] | **Optimizer baseline.** The current MNIST runner uses Adam. BGD remains the optimizer of record for the release baseline. | [Current runner](../EXPERIMENTS.md#selection-and-execution) · [Release contract](V0_CONTRACT.md#61-version-boundaries) |
| [CONFLICT] | **Statistics.** The roadmap's responsibility summary mentions a bootstrap interval. The release contract specifies a 95% binomial interval and a paired comparison for published accuracy claims. | [Roadmap](ROADMAP.md#domain-responsibility-summary) · [Release contract](V0_CONTRACT.md#63-research-claim-discipline) |
| [PENDING] | **Board and export API scope.** The roadmap assigns Urbana timing closure and `model.export_npu` alongside exact parity. Confirm their release scope with the maintainer. The release contract specifies the NPU inference evidence and its artifacts. | [Assignments](ROADMAP.md#what-is-the-current-v010-work) · [Release contract §§2.2–2.3](V0_CONTRACT.md#22-quantization--ternary-engine-srcquant) |
| [CONFLICT] | **Tensor surface and packed views.** The release contract lists generalized views, broadcasting, axis reductions and mutation capabilities. The implementation is partial and the later decision decouples typed storage from the release path. The tensor spec also names byte-backed T3 storage and byte-aligned views, while the code stores words and uses generic transpose. Record which surface must ship and resolve packing representation and view rules in their specification before extending the API. | [Release contract](V0_CONTRACT.md), [tensor specification](../engineering/specs/TENSOR.md), and [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [CONFLICT] | **Backward composition and loss reduction.** The release contract calls for module-local backward and the model spec describes recursive composition. The training spec declares a composite method consuming an explicit output gradient. The implementation uses a free reverse pass over `Vec<AnyModule>` and takes a loss and targets. The SoftmaxCE formula in the training spec describes one example. State the batch reduction explicitly when reconciling the interface. Preserve the forward trace and a single source of the initial delta. | [Release contract §2.4](V0_CONTRACT.md#24-tensorized-analytical-backpropagation), [model specification](../engineering/specs/MODEL_TRAIT.md), [training specification](../engineering/specs/TRAINING.md) |
| [PENDING] | **Public capability scope.** Confirm the release disposition of He initialization and the observer interface sketched in Section 2. Distinguish the assembled sequential layer manifest required for v0.1.0 from the DAG manifest in the future-proofing section. Record the supported API in the public reference. | [Release contract §§2, 5 and 6](V0_CONTRACT.md), [public API reference](../API.md) |
| [PENDING] | **Public training entry point.** Fluent `Sequential` construction returns `Self`. Select the initialization seed and shape-preserving parameter access APIs. Confirm the explicit optimizer, one-shot `fit` ownership and compiler entry point before changing public training and deployment signatures. | [Model specification](../engineering/specs/MODEL_TRAIT.md#22-sequential-as-a-composite-module) · [Training API proposal](#351-how-should-the-public-training-path-work-proposed) |
| [PENDING] | **Visualization.** The active release contract includes subsampled layer layout, activation intensity, and signed weight styling. Deliver that surface using a pinned model and trace, or record a maintainer-approved scope amendment. The graph helpers sit outside the public module tree, with no runnable visualization entry point. | [Release contract §3](V0_CONTRACT.md#3-visualization-and-observability-specifications) · [Current visualization status](../VISUALIZATION.md) |
| [PROPOSED] | **Artifact fixtures and naming.** Make run metadata and fixture provenance explicit. Document how the expected-output role is distinguished from dataset labels and legacy smoke inputs. Any change to `.mem` names or semantics must move the exporter, RTL consumer, and format docs together. | [Memory format](../engineering/MEM_FORMAT.md) |
| [DECIDED] | **Training optimization scope.** Release gating covers only `SoftmaxCE` log-sum-exp consistency and `backward` conflict closure; slice cleanups in `data.rs`, `adam.rs`, and `linear.rs` bias are optional non-blocking work; `MatMul` tiling/SIMD/threading is deferred beyond v0.1.0 (2026-11-26). | [Training optimization triage](#321-which-training-optimizations-belong-in-v010-decided) |
| [PROPOSED] | **Phase boundary.** Keep generalized graph execution, GPU training, int8, and zero-skipping on their documented milestones. Treat Adam, more aggressive kernel tuning, and deployment tooling as candidates justified by evidence and concrete consumers. | [Release boundaries](V0_CONTRACT.md#61-version-boundaries) and [future targets](../engineering/FUTURE_TARGETS.md) |

Record each maintainer disposition in its canonical contract or decision log
before the dependent implementation. Numeric or format changes follow
Engineering Invariant 4. Work that already satisfies the active contract can
proceed while broader scope decisions are open.

---

## 5. How can work proceed in parallel?

| Track | First merge-sized change | Next dependency | Completion evidence |
| --- | --- | --- | --- |
| Stochasticity, data and evaluation | Fix the seeded stream and its tests. | Freeze split, preprocessing and evaluation settings. | Reproducible data ordering, checkpoint evaluation, and paired metrics. |
| Training and tensor engine | Verify the loss-to-backward path and add MSE regressions. | Consume the verified data path and close numerical gradient checks. | Stable float MNIST baseline. Tensor cleanup follows correctness and benchmark fixtures. |
| Quantization and RTL | Build hand-calculated integer cases and specify the exact comparator behavior. | Agree on quantizer edge cases, derive the shared artifact and connect it to the exact evaluator and testbench. | Small exact fixture first, trained MNIST parity second. |
| Release integration | Track the open scope decisions and reproduce documented entry points. | Collect final held-out metrics and parity from frozen inputs. | One named candidate revision, complete evidence and a maintainer tag decision. |

Use existing [roadmap ownership](ROADMAP.md#2-engineering-matrix-and-cross-functional-intersections).
The small parity model allows RTL work to finish ahead of the trained MNIST
checkpoint. Seed and loss work can land without waiting for tensor
microbenchmarks. Avoid mixing arithmetic, artifact-format and loop-order
changes in one experiment unless the contract requires the changes to move
together.

### How is the candidate cut? [PROPOSED]

1. Confirm the release scope and select the revision to evaluate. Generate the
   artifact and expected outputs from the selected checkpoint and record their identities.
2. Run the correctness gates and collect held-out metrics. Keep the evaluated
   artifact fixed while interpreting comparisons.
3. Apply accepted hardening changes as independently reviewable revisions. An
   optimization should demonstrate which costs fell and which correctness
   evidence stayed valid. Regenerate affected evidence on the final revision.
4. Close the release gates on the final revision and hand the completed checklist
   to the maintainer for the tag decision.

### What evidence travels with the candidate?

- [ ] The tested revision and locked software and RTL toolchains are identified.
- [ ] OR/XOR regressions, float finite differences, the QAT/STE step fixture,
  and the Rust gate are green.
- [ ] Fit, development and final evaluation data are identified. The frozen
  float checkpoint and exported PTQ and QAT artifacts have per-sample
  predictions and counts on the same reserved IDs. Float accuracy is at
  least 95% and bit-exact ternary accuracy is above 90%. Record the 95%
  binomial interval for the ternary result and record the paired comparison
  for publication.
- [ ] The layer manifest, parameters, inputs and integer expectations are
  traceable to the candidate. Exact Tier 2 comparisons report zero
  mismatched words. RTL lint, Tier 1 and Tier 2 results are recorded.
- [ ] Visualization, the tensor and public API surface, and the other
  [open release clauses](#4-which-decisions-need-an-explicit-disposition)
  have an implemented or ratified disposition. Numbered research logs
  point to the reproducible evidence required by the
  [release contract](V0_CONTRACT.md#4-academic-research-logging-protocol-paper-prerequisites).
- [ ] Each selected hardening change has its before/after evidence and repeatable command. Unselected work has an explicit phase or disposition.
- [ ] Public docs describe working entry points and distinguish implemented capabilities, measurements, and future targets.

# How does TARS deliver and harden v0.1.0?

**Status: [PROPOSED]** Execution plan for maintainer review. The
[v0.1.0 release contract](V0_CONTRACT.md) owns release acceptance. The
[roadmap](ROADMAP.md) owns work-package assignments. Engineering specifications
own interfaces, arithmetic, and formats.

First produce a reproducible release candidate that meets the MNIST and Rust–RTL
targets. Then harden the same candidate through measured, independently
reviewable changes. The release evidence must be reproducible on the exact
revision the maintainer chooses to tag.

| Milestone | Completion criterion |
| --- | --- |
| **Release candidate** | Reproducible float and QAT training, held-out MNIST accuracy at the contractual thresholds, exact Rust–RTL outputs, and closure of the published v0.1.0 scope. |
| **Hardened candidate** | Correctness evidence survives the changes. Measured improvements land in tensor kernels, training data flow, and deployment diagnostics. Each remaining proposal has an explicit disposition. |

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

This inspection covers source revision `e7b7f64`. The table records source
observations. Release test and performance evidence is pending.

| Status | Area | Observed behavior |
| --- | --- | --- |
| [DECIDED] | Float training boundary | Data, `Module::forward`, parameters, the forward cache, parameter gradients, and optimizer state use `Tensor<f32>`. LOG-008 decouples typed T3 storage from the release training, export, and parity path. [Training specification](../engineering/specs/TRAINING.md), [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026) |
| [PENDING] | Packed ternary consumer | `PackedT3Storage` stores 32 trits per `u64`. The mixed product converts each trit to `f32` for multiplication and carries no scale. PTQ and export use the scalar ternary encoder. [`ternary_tensor.rs`](../../src/tensor/ternary_tensor.rs), [`export.rs`](../../src/export.rs) |
| [CONFLICT] | Backward and loss | `Sequential::forward_cache` records a flat trace and the free backward pass walks it in reverse. The runner uses synthetic MSE targets to compensate for the legacy delta overwrite. The target module-local interface remains open. [`sequential.rs`](../../src/modules/sequential.rs), [`grad.rs`](../../src/optim/grad.rs), [`main.rs`](../../src/bin/main.rs) |
| [CONFLICT] | SoftmaxCE forward and backward | The loss averages over the batch and its backward returns $(p-y)/B$ for normalized targets. Forward clamps probabilities before taking the logarithm, so the reported scalar and gradient disagree for sufficiently unfavorable logits. Use log-sum-exp and verify the pair on a deterministic float fixture. [`loss.rs`](../../src/optim/loss.rs), [training specification](../engineering/specs/TRAINING.md#12-loss-function-contract) |
| [CONFLICT] | MNIST baseline | The runner uses Adam, evaluates on training data, and uses a single affine layer. The release calls for BGD, held-out evaluation, and the assembled `784 → 64 → 10` topology. Telemetry reports MSE on raw logits and averages the batch means. [`main.rs`](../../src/bin/main.rs), [`mnist.rs`](../../src/experiments/mnist.rs), [release contract](V0_CONTRACT.md) |
| [PENDING] | QAT and exact parity | The official QAT step and the independent fixed-point reference remain pending. `export_inference` serializes float predictions, and Tier 2 permits a one-LSB difference. [`main.rs`](../../src/bin/main.rs), [`export.rs`](../../src/export.rs), [`tb_parity.sv`](../../npu/tests/tb_parity.sv) |
| [PENDING] | Dense tensor fast paths | The committed core has logical dense slices and an `FnMut` mapping path with strided fallback. `is_contiguous` allocates stride metadata, and consumers such as `sum` use a logical iterator that panics for rank zero. [`tensor_core.rs`](../../src/tensor/tensor_core.rs) |

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
| 1 | **Restore RNG state.** Own a seeded generator for the initialization stream and advance it across draws and layers. Give initialization and shuffling explicit, recorded seeds. Fix the ternary distribution test to exercise the ternary sampler. | The same algorithm and seed reproduce the sequence, successive draws vary, and the initializer produces distinct neuron rows on a fixed fixture. The Rust gate passes without retries. Record the locked dependencies and execution environment for byte-level reproducibility. |
| 2 | **Pin the small-model baseline.** Preserve sigmoid/MSE behavior on OR and XOR. Keep their training commands independent of MNIST downloads and export side effects. | Deterministic fixture for predictions, loss, and a parameter update. Repeated runs reproduce the expected result under the recorded toolchain. |
| 3 | **Finish the loss boundary.** Register MSE and SoftmaxCE and let the loss produce the initial delta consumed by the analytical backward pass. Define identical shapes, consistent batch reductions, and diagnostics for invalid inputs. | MSE divides by the number of elements. For one-hot targets, batch-mean SoftmaxCE accepts raw logits, uses a stable max-shift/log-sum-exp computation, and returns `(p - y) / B`. Tests cover large logits, the final short batch, mismatched inputs, and finite differences. |
| 4 | **Verify the model gradient and optimizer.** Use a deterministic multilayer float fixture with ReLU inputs away from zero. Check both weight and bias gradients. Validate the model–gradient pairing before applying BGD updates. | The roadmap sets a finite-difference tolerance of $\epsilon \le 10^{-4}$. Record the perturbation and error metric alongside the result. An invalid gradient is rejected before changing parameters. |
| 5 | **Finish the MNIST run path.** Use the existing reader, retain separate train and held-out splits, validate image and label counts and ranges before pairing, and keep exported parity samples explicit. | The runner records dataset identity, topology, seeds, batch size, learning rate, optimizer, epochs, checkpoint identity, and the accuracy denominator. It can run the same evaluation from a saved checkpoint without retraining. |

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

### 2.4 How do export and RTL establish exact parity? [PENDING]

| Order | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | **Validate an assembled model before emission.** Check supported activations, connected layer dimensions, weight order, one scale per layer, exact bias sizes, and input shape. Export to an explicit run directory. | Invalid input fails before publishing a set of artifacts. XOR exports and MNIST parity fixtures have independent destinations. The existing wire format changes only through Engineering Invariant 4. |
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

This milestone is the proposed quality bar for a polished v0.1.0. It begins from
the passing evidence above. Performance-only work can be developed in parallel
after its baseline is captured. Changes merged before the tag must preserve the
release evidence. Later changes follow normal versioned releases.

### 3.1 How should the tensor core handle views and bulk operations? [PROPOSED]

| Sequence | Design | Acceptance evidence |
| --- | --- | --- |
| 1 | Fix logical traversal before adding fast paths. A rank-zero tensor has one element. An empty extent has zero. Check canonical contiguity without allocating stride metadata. | Tests cover scalars, empty extents, rectangular transposes, round-trip materialization, and storage larger than `numel()`. A borrowed dense slice contains exactly the logical extent. |
| 2 | Implement `map` with a borrowed dense slice and a stride-aware fallback. Visit elements in logical row-major order and construct the canonical output directly. Accept a stateful callback. | Dense and strided results agree with hand-calculated examples. A stateful closure observes the same order. The fallback has no temporary packed input and the output avoids an initial zero fill. |
| 3 | Introduce crate-internal `zip_map` for equal-shape element-wise kernels and migrate arithmetic to it. Keep broadcasting and reductions as explicit operations. | Shape mismatch fails before the callback, including different shapes with equal element counts. Test both-contiguous, mixed-layout, and both-transposed pairs. |
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

| Work | Expected improvement | Evidence required |
| --- | --- | --- |
| Draw parameters into a flat buffer and gather batches from one shuffled index buffer. Move completed buffers into tensors. | Less allocation and copying during initialization and batching. | Sample order matches the fixture for each seed. Include the final short batch. Measure allocation count and peak temporary memory. |
| Build each parameter gradient once. Express row-bias addition and batch reduction with explicit kernels. Fuse activation derivative application and the BGD update where lifetimes allow. | Fewer temporary tensors and a shorter backward path that follows the layer math. | Gradient fixtures, independent mutation tests, and full-step latency and allocation measurements. |
| Keep `matmul` as its own benchmarked change. Start from direct stride-aware indexing and preserve each output's reduction order on the first pass. Compare loop order and optional kernel-local packing. | Locality improvements for the real forward and backward layouts. | Compare `X × Wᵀ`, `deltaᵀ × X`, and `delta × W` for both layers and representative batches. Include packing time and memory. Check output against a simple reference, then rerun end-to-end evidence. |
| Narrow parameter and training boundaries with real callers. Preserve the consuming builder and forward trace. Keep layer-specific calculus local and validate dimensions at construction and replacement. | A layer can be tested independently and the optimizer sees an unambiguous parameter order. | OR/XOR and MNIST entry points still work. Parameter access, trace use, and mismatch diagnostics have focused tests. Introduce wider abstractions when a second concrete consumer needs them. |
| Report each epoch's training objective from cached outputs and aggregate by the objective's denominator. Measure telemetry with the same batches and logging cadence. | Reported loss follows the objective used for updates, including on a short final batch. | Check a full and a short batch against a reference. Report throughput with telemetry enabled and disabled. |

On the inspected path, batch gathering clones indexed samples and copies both
completed buffers into tensors. Backward allocates zeroed parameter gradients
before replacing them, and still computes the model input gradient after the
first affine layer even though the current API returns parameter gradients
alone. Activation backward allocates a derivative tensor before multiplying by
the delta. These are candidates for the work above. The input gradient is needed
between layers and whenever a caller requests it.

The module tree compiles the SoftmaxCE implementation in `optim/loss.rs`.
The disconnected `optim/softmax.rs` stub is a cleanup candidate.

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
| Correctness first | Pass the same reference fixtures, deterministic training checks and relevant RTL checks before interpreting timing. Changes affecting the selected checkpoint or exported words regenerate candidate evidence. |
| Compare one change | Pin baseline and candidate revisions, dataset and seeds, compiler and target, feature flags, release profile, and machine. For local kernels include small XOR shapes and MNIST training shapes. |
| Measure the whole cost | Warm up, repeat, and record latency distribution, allocations, and peak temporary memory. Measure both isolated kernels and a full training step. Include cold packing and separately measured packing reuse. |
| Decide before merging | State the workload, baseline, expected improvement and acceptable variance beforehand. Keep simpler code when differences are within noise. Record a meaningful gain together with any memory or code complexity cost. |
| Keep the run reproducible | Retain commands, seed, input hashes and raw measurements under the project evidence policy. Generated `.mem`, waveforms, bitstreams and build products stay outside version control. |

| Workload | Comparisons to run |
| --- | --- |
| Element-wise | Dense versus transposed and mixed-layout inputs. Compare direct traversal with one-time explicit packing over one use and repeated reuse. Benchmark unique and shared destinations separately for mutation. |
| Affine layer | For each inner/output pair $(K,N)$ in $\{(784,64),(64,10)\}$, measure forward $[B,K]\times[K,N]$, weight gradient $[N,B]\times[B,K]$, and input gradient $[B,N]\times[N,K]$. Use batch size 1, the configured training batch size, and the final short batch when present. |
| Full training step | Use one seeded batch and identical parameters. Measure batch gather, cached forward, loss, backward and BGD together. Report the same scope for every revision. |
| Deployment | Measure artifact read and validation, inference with a ready model, and the full input-to-output path as separate timings. Report clock and target with cycle counts. |

---

## 4. Which decisions need an explicit disposition?

| Status | Issue | Canonical record |
| --- | --- | --- |
| [CONFLICT] | **Quantizer ties and tiny scales.** LOG-008 specifies ties away from zero and the spec separately gives strict comparisons. The implementation uses strict comparisons and a small-scale cutoff. Confirm the exact behavior, including $W_0 = 0$ and positive scales near the cutoff, before freezing release fixtures. | [Quantization specification](../engineering/specs/QUANT.md) and [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [CONFLICT] | **External Q8.24 encoding versus internal rescaling.** The arithmetic document records both a target ties-to-even encoder and current normative truncate-toward-zero serialization. State the selected release encoder separately from the mandatory arithmetic floor inside the epilogue. The contract summary also calls the accumulator saturating, while the detailed arithmetic contract and PE keep a wide accumulator and saturate in the epilogue. | [Q8.24 arithmetic specification](../engineering/specs/ARITHMETIC_Q8_24.md) |
| [DECIDED] | **Program and wrapper scope.** v0.1.0 uses the manually assembled two-layer wrapper. `program.mem` and runtime shape controls belong to the v0.1.1 programmable top. | [Release contract](V0_CONTRACT.md#61-version-boundaries) · [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [CONFLICT] | **Optimizer and statistics.** The older roadmap mentions Adam and a bootstrap interval. The release contract makes BGD the optimizer of record, specifies a 95% binomial interval, and requires the paired comparison for published accuracy claims. | [Release contract](V0_CONTRACT.md) |
| [CONFLICT] | **Tensor surface and packed views.** The release contract lists generalized views, broadcasting, axis reductions and mutation capabilities. The implementation is partial and the later decision decouples typed storage from the release path. The tensor spec also names byte-backed T3 storage and byte-aligned views, while the code stores words and uses generic transpose. Record which surface must ship and resolve packing representation and view rules in their specification before extending the API. | [Release contract](V0_CONTRACT.md), [tensor specification](../engineering/specs/TENSOR.md), and [LOG-008](research/LOG-008-v0_governance_and_release_decisions.md) |
| [CONFLICT] | **Backward composition and loss reduction.** The release contract calls for module-local backward and the model spec describes recursive composition. The training spec declares a composite method consuming an explicit output gradient. The implementation uses a free reverse pass over `Vec<AnyModule>` and takes a loss and targets. The SoftmaxCE formula in the training spec describes one example. State the batch reduction explicitly when reconciling the interface. Preserve the forward trace and a single source of the initial delta. | [Release contract §2.4](V0_CONTRACT.md#24-tensorized-analytical-backpropagation), [model specification](../engineering/specs/MODEL_TRAIT.md), [training specification](../engineering/specs/TRAINING.md) |
| [PENDING] | **Visualization.** The active release contract includes subsampled layer layout, activation intensity, and signed weight styling. Deliver that surface using a pinned model and trace, or record a maintainer-approved scope amendment. The current binary stub is insufficient. | [Release contract §3](V0_CONTRACT.md#3-visualization-and-observability-specifications) |
| [PROPOSED] | **Artifact fixtures and naming.** Make run metadata and fixture provenance explicit. Document how the expected-output role is distinguished from dataset labels and legacy smoke inputs. Any change to `.mem` names or semantics must move the exporter, RTL consumer, and format docs together. | [Memory format](../engineering/MEM_FORMAT.md) |
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
| Training and tensor engine | Finish loss-to-backward integration and MSE regression. | Consume the verified data path and close numerical gradient checks. | Stable float MNIST baseline. Tensor cleanup follows correctness and benchmark fixtures. |
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
- [ ] OR/XOR regressions, float finite differences, and the Rust gate are green.
- [ ] The selected float, PTQ and QAT configurations, model identities and held-out results are recorded.
- [ ] Exact Tier 2 comparisons use a matching artifact set. RTL lint, Tier 1 and Tier 2 results are recorded.
- [ ] Visualization, tensor scope, and the other open release clauses have an implemented or ratified disposition.
- [ ] Each selected hardening change has its before/after evidence and repeatable command. Unselected work has an explicit phase or disposition.
- [ ] Public docs describe working entry points and distinguish implemented capabilities, measurements, and future targets.

# Current Public API

This document identifies the implemented public surface of the TARS Rust
package. Internal design targets are maintained under
[`engineering/`](engineering/README.md).

---

## 1. Data and Example Presets

| Resource | Current role | Important constraint |
|---|---|---|
| `Data` | Stores one input and one target as rank-one `Tensor<f32>` values | Dimensions are not validated by the constructor |
| `DatasetSplit` | Pairs `train: Vec<Data>` and `eval: Vec<Data>` | Defined in `src/data.rs`; separates training and evaluation slices |
| `DataMiniBatches` | Shuffles samples and builds rank-two input and target tensors | Requires a positive batch size and a nonempty data set. The current shuffle requires a nonzero effective epoch seed |

The library crate does not embed built-in dataset presets; runnable OR, XOR, and
MNIST configurations live in [`examples/`](../examples/mnist.rs) using
`mnist_reader` from `[dev-dependencies]`. See [`EXPERIMENTS.md`](EXPERIMENTS.md)
for topologies, datasets, and execution commands.

---

## 2. Models and Modules

| Resource | Implemented operations | Important constraint |
|---|---|---|
| `Module` | Forward evaluation from `&Tensor<f32>` to `Tensor<f32>` | The trait currently exposes `forward` |
| `Sequential` | `new(input_size, first_output_size)` creates the first dense layer. `new`, `linear`, `relu` and `sigmoid` return `Sequential`. Each appended linear layer infers its input width | The builder rejects zero widths. Public mutation of `modules` bypasses construction checks. [Constructor source](../src/modules/sequential.rs) |
| `Sequential::quantize`, `Sequential::quantize_into`, `Sequential::zeros_like` | Project weights to scaled ternary `f32` values (`quantize` allocates a new model; `zeros_like` and `quantize_into` reuse a pre-allocated buffer and `scales` vector) | Biases and activations are carried over. Scales follow linear layer order |
| `Sequential::num_weights`, `Sequential::num_parameters` | Count weights and, for parameters, biases | Activation modules contribute no parameters |
| `Sequential::sparsity`, `Linear::sparsity` | Report weight sparsity (`Sequential::sparsity` counts weights in a single pass with zero-division guard) | Apply to projected weights with positive scales in linear layer order. The strict comparison reports zero sparsity for an all-zero layer with a zero scale |
| `Sequential::evaluate_accuracy` | Evaluates classification accuracy (%) on `&[Data]` in `Tensor<f32>` | Currently hardcodes 10-class `argmax` `(0..10)` (panics on single-output models like OR/XOR); generalizing via `Decision` is tracked in [`ROADMAP.md`](project/ROADMAP.md#what-is-the-current-v010-work) |
| `absmean`, `quantize_scalar`, `quantize_into_f32` | Ternary absmean scale, scalar `T3` lookup, and slice-level `f32` ternary projection | Defined in [`src/quant.rs`](../src/quant.rs) and shared by `Linear::quantize` and `Linear::quantize_into` |
| `Linear` | Construction from explicit parameters, zeros, or seeded initialization; slice-based `quantize` and in-place `quantize_into` | The explicit constructor checks weight and bias shapes. Public fields allow unchecked replacements. Forward accepts one sample or a rank-two batch |
| `Activation` | ReLU and sigmoid evaluation | No additional activation variants are implemented |
| `AnyModule` | Runtime dispatch over linear and activation modules | Typed accessors panic when used with the wrong variant |

The [model specification](engineering/specs/MODEL_TRAIT.md) describes the target
hierarchy and parameter access. The [delivery plan](project/V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition)
records the differences in backward composition.

---

## 3. Training Operations

| Resource | Current role | Important constraint |
|---|---|---|
| `cost` | Computes mean squared error over outputs and samples | Empty or dimensionally inconsistent data is not guarded |
| `Sequential::forward_cache` | Records the input and each module output as `Tensor<f32>` | The caller owns the cache for the training step |
| `Loss`, `MSE`, `SoftmaxCE` | Evaluate float objectives and produce an output gradient | SoftmaxCE expects raw logits. See the delivery plan for the forward and gradient discrepancy at extreme logits |
| `backward` | Walks the flat module list in reverse using a rank-two batch and its `ForwardCache`. Uses the selected loss's output gradient and returns parameter gradients indexed by module | Match the model and cache to the same forward pass |
| `Optimizer`, `BGD`, `Adam` | `Optimizer` exposes `lr()` and `step()`. Both implementations update float parameters | Pair compatible model, optimizer state, and gradients. [Source](../src/optim.rs) |
| `Trainer::new(model, optimizer, loss)` | Holds a mutable borrow of the model and owns the optimizer and loss | Default mode is float. `with_qat`, `silent`, `batch_size`, and `log_every` are fluent builders returning `Self`; `with_qat(true)` allocates a reusable QAT projection buffer once |
| `Trainer::fit` | Trains on the supplied slice, prints progress according to `silent`/`log_every`, and returns `TrainSummary` | Projects weights into the pre-allocated QAT buffer via `quantize_into` when enabled. `batch_size` defaults to 32 and `log_every` defaults to 1 (clamped to `>= 1`). Each call starts the shuffle at seed 42; when `silent` is false, prints multiples of `log_every` and the final epoch with window-level elapsed time and throughput. [Source](../src/train.rs) |
| `ModelReport`, `SizeMetrics`, `QuantizedMetrics` | Fluent evaluation and footprint report (`with_train`, `with_size`, `with_accuracy`, `with_ternary_accuracy`, `with_sparsity`) with `Display` formatting | Quantizes at most once lazily via `ensure_quantized` when ternary accuracy or sparsity is requested. Accuracy evaluation currently requires `&[Data]` compatible with `Sequential::evaluate_accuracy`. [Source](../src/report.rs) |

The epoch loss weights each pre-update batch's `Loss::forward` by its sample
count, including the last short batch. This aggregates SoftmaxCE and MSE with
fixed output width. The executable's preset is documented separately.
[Experiment behavior](EXPERIMENTS.md#selection-and-execution) ·
[Verification work](project/V0_DELIVERY_PLAN.md#22-how-do-we-restore-trustworthy-training-pending) ·
[Proposed training API](project/V0_DELIVERY_PLAN.md#351-how-should-the-public-training-path-work-proposed)

---

## 4. Export and Visualization

| Resource | Current role | Important constraint |
|---|---|---|
| `to_q8_24` | Converts an `f32` value to the current Q8.24 word representation | Uses truncation rather than round-half-to-even |
| `export_quantized_model` | Writes dense ternary codes and Q8.24 scales and biases | Accepts scales in linear layer order. The artifacts omit topology and activations. [Memory format](engineering/MEM_FORMAT.md) |
| `export_inference` | Writes Q8.24 inputs and serialized float predictions | Uses fixed paths under `npu/data/`. The predictions come from float execution |
| Visualization | [PENDING] No `visualization` binary is currently present and `src/view/` sits outside the exported library surface | [Visualization status](VISUALIZATION.md) |

Exact numeric parity is tracked in the [delivery plan](project/V0_DELIVERY_PLAN.md#24-how-do-export-and-rtl-establish-exact-parity-pending).

---

## 5. Experimental Tensor Surface

`Tensor` is public but incomplete. The current implementation provides the following
surface:

| Resource | Implemented operations | Important constraint |
|---|---|---|
| `Tensor::new` | Wraps storage in a canonical row-major view | Requires storage for at least the shape's element count |
| `Tensor::zeros` | Allocates a zeroed float buffer for a shape | The method requires the element type to implement `Zero` |
| `Tensor::get`, `Tensor::set` | Index lookup and mutation over typed storage, including `f32` and `T3` | Mutation detaches the backing buffer if another view shares it |
| `Tensor::transpose`, `Tensor::transpose_mut` | Swap shape and stride entries while sharing existing element storage | Both axes must lie within the tensor rank |
| `Tensor::shape`, `Tensor::strides`, `Tensor::offset`, `Tensor::numel`, `Tensor::as_transposed_slice` | Expose geometry and zero-copy transposed 2D slice view. `Element::DTYPE` supplies the one-byte type tag | Shape and strides describe logical coordinates |
| `Tensor::is_contiguous` | Checks canonical row-major strides and zero offset | Zero-allocation check across `shape` and `strides` in reverse order |
| `Tensor::make_contiguous` | Re-packs a non-contiguous view into a canonical buffer; shares the storage of an already contiguous source | Carries an identifier distinct from the `contiguous` operation of the specification; no test coverage |
| `Tensor::add`, `sub`, `mul`, `div` and their in-place variants | Pointwise operations over two tensors, honouring the stride vectors | Shape-mismatch violations are reported through empty diagnostics |
| `Tensor::mul_scalar`, `div_scalar` and their in-place variants | Pointwise float scalar multiplication and division | Scalar division rejects zero |
| `Tensor::map`, `map_mut` | Apply a callable in logical element order | The [delivery plan](project/V0_DELIVERY_PLAN.md#31-how-should-the-tensor-core-handle-views-and-bulk-operations-proposed) tracks dense slices and stateful callbacks for the bulk path |
| `Tensor::sum` | Full reduction across all logical coordinates | [CONFLICT] Rank-zero tensors are admitted by the spec. The logical iterator used by `sum` currently panics on a rank-zero operand. |
| `MatMul`, `Tensor::matvec`, `Tensor::outer` | Float products (`MatMul` uses contiguous and transposed 2D slices; `matvec` uses contiguous slices with generic fallback). A mixed float/ternary matrix product is also available | The mixed product uses raw trits and has no layer scale |
| `Tensor::storage`, `Tensor::storage_ref` | Clone or borrow the associated backing storage | The raw buffer follows physical storage order |

Ternary storage is experimental and has no consumer in the built-in training and
export paths. The [tensor specification](engineering/specs/TENSOR.md) defines the
target operations, including broadcasting.

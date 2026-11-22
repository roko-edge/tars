# Current Public API

This document identifies the implemented public surface of the TARS Rust
package. Internal design targets are maintained under
[`engineering/`](engineering/README.md).

---

## 1. Data and Experiment Configuration

| Resource | Current role | Important constraint |
|---|---|---|
| `Data` | Stores one input and one target as rank-one `Tensor<f32>` values | Dimensions are not validated by the constructor |
| `DataMiniBatches` | Shuffles samples and builds rank-two input and target tensors | Requires a positive batch size and a nonempty data set. The current shuffle requires a nonzero effective epoch seed |
| `Train` | Groups a model, epoch count, learning rate, dataset, and experiment identifier. The working-tree `Train<L: Loss>` also carries the objective | `Trainer::fit` executes training |
| `ExperimentType` | Identifies OR, XOR, MNIST, or a custom configuration | Callers select the factory explicitly |
| `experiments::or::train()` | Creates the built-in OR configuration | Each linear constructor starts the same fixed seed and initializes biases to zero |
| `experiments::xor::train()` | Creates the built-in XOR configuration | The main binary does not select it currently |
| `experiments::mnist::train()` | Creates a model and one-hot targets from the MNIST training split | The topology and hyperparameters come from the experiment factory and can change by revision. [Experiment configurations](EXPERIMENTS.md#built-in-experiments) |

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for topologies, datasets, and execution behavior.

---

## 2. Models and Modules

| Resource | Implemented operations | Important constraint |
|---|---|---|
| `Module` | Forward evaluation from `&Tensor<f32>` to `Tensor<f32>` | The trait currently exposes `forward` |
| `Sequential` | `new(input_size, first_output_size)` creates the first dense layer. `new`, `linear`, `relu` and `sigmoid` return `Sequential`. Each appended linear layer infers its input width | Records a training trace through `forward_cache`. Public mutation of `modules` bypasses construction checks. [Committed and working-tree dimension checks](engineering/specs/MODEL_TRAIT.md#specification-model-and-module-interface-hierarchy) |
| `Sequential::quantize` | Returns a model with scaled ternary weights represented as `f32`, plus one scale per linear layer | Biases and activations are carried over. Scales follow linear layer order |
| `Sequential::num_weights`, `Sequential::num_parameters` | Count weights and, for parameters, biases | Activation modules contribute no parameters |
| `Sequential::sparsity`, `Linear::sparsity` | Report weight sparsity | Apply to projected weights with positive scales in linear layer order. The strict comparison reports zero sparsity for an all-zero layer with a zero scale |
| `Linear` | Construction from explicit parameters, zeros, or seeded initialization | The explicit constructor checks weight and bias shapes. Public fields allow unchecked replacements. Forward accepts one sample or a rank-two batch |
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
| `BGD`, `Adam` | Apply gradient updates to float parameters | Callers must pair compatible model, optimizer state, and gradient structures |

At `4961a9e`, the executable owns the training loop. The finite-difference
verification gate and the QAT training step are tracked in the
[delivery plan](project/V0_DELIVERY_PLAN.md#22-how-do-we-restore-trustworthy-training-pending).
The [proposed training API](project/V0_DELIVERY_PLAN.md#351-how-should-the-public-training-path-work-proposed)
describes the public entry point under consideration.

[PENDING] At `7dd519d`, `Trainer::fit` updates the borrowed
model in float mode by default. QAT selects projected weights for forward
and backward and updates the latent model. BGD and Adam implement
`Optimizer::lr()` and `step()`. The loop uses a batch size of 64 and a shuffle
seed of 42. The batch-size and silent setters remain disconnected from the
loop, which prints metrics and returns `()`. The executable selects float
training with Adam and evaluates the float model and its PTQ projection.
`cargo check --all-targets` succeeds on the inspected source. Training results
and parity evidence remain pending. [Trainer](../src/train.rs) ·
[Optimizer trait](../src/optim.rs) · [Executable](../src/bin/main.rs)

[PENDING] The updated trainer uses the configured `batch_size` with a
default of 32 and logs the effective batch size and QAT mode at startup.
The epoch loss uses the selected `Loss::forward` on the cached
predictions before each update and weights the batch means by actual sample
count. This aggregation covers SoftmaxCE and MSE with a fixed output width.
`silent` remains disconnected. [Trainer](../src/train.rs)

---

## 4. Export and Visualization

| Resource | Current role | Important constraint |
|---|---|---|
| `to_q8_24` | Converts an `f32` value to the current Q8.24 word representation | Uses truncation rather than round-half-to-even |
| `export_model` | Writes dense ternary codes and Q8.24 scales and biases | Accepts scales in linear layer order. The artifacts omit topology and activations |
| `export_inference` | Writes Q8.24 inputs and serialized float predictions | Uses fixed paths under `npu/data/`. The predictions come from float execution |
| Visualization | [PENDING] The `visualization` binary is a stub and `view` is outside the exported library surface | [Visualization status](VISUALIZATION.md) |

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
| `Tensor::shape`, `Tensor::strides`, `Tensor::offset`, `Tensor::numel` | Expose geometry. `Element::DTYPE` supplies the one-byte type tag | Shape and strides describe logical coordinates |
| `Tensor::is_contiguous` | Checks canonical row-major strides and zero offset | The current comparison constructs a stride vector |
| `Tensor::make_contiguous` | Re-packs a non-contiguous view into a canonical buffer; shares the storage of an already contiguous source | Carries an identifier distinct from the `contiguous` operation of the specification; no test coverage |
| `Tensor::add`, `sub`, `mul`, `div` and their in-place variants | Pointwise operations over two tensors, honouring the stride vectors | Shape-mismatch violations are reported through empty diagnostics |
| `Tensor::mul_scalar`, `div_scalar` and their in-place variants | Pointwise float scalar multiplication and division | Scalar division rejects zero |
| `Tensor::map`, `map_mut` | Apply a callable in logical element order | The [delivery plan](project/V0_DELIVERY_PLAN.md#31-how-should-the-tensor-core-handle-views-and-bulk-operations-proposed) tracks dense slices and stateful callbacks for the bulk path |
| `Tensor::sum` | Full reduction across all logical coordinates | [CONFLICT] Rank-zero tensors are admitted by the spec. The logical iterator used by `sum` currently panics on a rank-zero operand. |
| `MatMul`, `Tensor::matvec`, `Tensor::outer` | Float products honoring strides. A mixed float/ternary matrix product is also available | The mixed product uses raw trits and has no layer scale |
| `Tensor::storage`, `Tensor::storage_ref` | Clone or borrow the associated backing storage | The raw buffer follows physical storage order |

Ternary storage is experimental and has no consumer in the built-in training and
export paths. The [tensor specification](engineering/specs/TENSOR.md) defines the
target operations, including broadcasting.

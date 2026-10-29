# Specification: Tensor Storage, Strides, Data Types, and Memory Layout

**Status:** Target contract for the core tensor subsystem. This specification defines
the normative architecture, minimum viable surface, memory layout, striding rules,
type representations, and linear algebra contracts for the typed tensor core in
`src/math/tensor.rs`. Section 4.2 records the delivery phase in progress and the
surface that remains deferred.

**Revision (October 2026):** the element type is a compile-time parameter
(`Tensor<T>`), replacing the container-level storage dispatch of the preceding
revision. The wire tag `DType` is retained for serialization and for the hardware
descriptor. The source implements the preceding container-level design until the
typed migration lands; Section 4.2 governs the transition.

---

## 1. Architectural Foundations and Design Rationale

The tensor abstraction in TARS decouples structural indexing, numerical representation,
and physical storage. The element type is a compile-time parameter; the BitNet b1.58
reference implementations (`microsoft/BitNet` and `llama.cpp` TQ2_0) anchor the packed
ternary representation, and the per-element-type implementation model keeps every
execution context of this project statically typed.

### 1.1 Structural Decoupling: Element Type vs. Storage vs. Layout

The tensor decouples three concerns. The numerical type of an element is a
compile-time parameter; the physical storage representation is defined per element
type; the geometry (`shape`, `strides`, `offset`) is shared and type-independent.

| Layer | Responsibility | Rationale |
| --- | --- | --- |
| `Tensor<T>` | One concrete tensor type per element type `T: Element`; geometrical metadata is shared | The execution contexts of this project are statically typed: training is `f32`-latent in all modes, the bit-exact reference is ternary, and the export path is generic over `T`. Mixed-precision kernels (activations by ternary weights) have a static type shape that belongs in signatures, not in runtime checks |
| `Element` | The per-type contract: the associated `Storage`, the wire `DType`, the element capacity of a buffer, and the dequantizing element read | Encodes the per-(operation, type) status matrix of this specification as per-type implementation blocks: an operation absent from a type's block is absent at compile time instead of being reported at runtime. Planned GPU kernels (cubecl, v0.3.0) are typed per element by construction |
| `DType` | One-byte copyable wire tag for serialization, the layer manifest, and the hardware descriptor `mode` field | The tag names a representation in artifacts; it is derived from the element type and never stored in the tensor |

`DType` remains a fieldless enumeration with a one-byte representation. It derives
`Clone`, `Copy`, `PartialEq`, `Eq`, and `Debug`, so that the tag is returned by value
and is directly comparable. The numeric value of each discriminant is an implementation
detail and is not part of this contract.

Fixed-point arithmetic (`Q8_24`) is strictly excluded from the element types. It is
retained solely as a serialization format within `src/export.rs` for generating SystemVerilog
`$readmemh` test artifacts under `npu/data/`. Simulating fixed-point inside the tensor
operator graph imposes artificial software overhead without parity benefit.

### 1.2 Implementation Status Vocabulary

Section 4 annotates each member of the minimum viable surface with an implementation
status, recorded per (element type, operation) pair. The vocabulary is normative:

| Status | Meaning |
| --- | --- |
| `Implemented` | The behavior described by this specification is present and exercised by the test suite |
| `Partial` | The operation exists but does not yet satisfy every clause of its contract |
| `Not implemented` | The operation is specified here and absent from the runtime |
| `Deferred` | The operation is outside the scope of the current delivery phase |

---

## 2. Data Structure Invariants and Representations

### 2.1 Type Representations

```text
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    I8,
    T3,
}

pub trait Element {
    type Storage;

    const DTYPE: DType;

    // Per-type contracts, specified by the sections below:
    // - the element capacity of a storage buffer (Section 2.2, Invariant 5)
    // - the dequantizing element read into f32 (Section 2.3)
    // - the zeroed storage allocation for a shape (Section 4)
}

pub struct Tensor<T: Element> {
    storage: T::Storage,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

// Operations are delivered as per-type implementation blocks:
//   impl Tensor<f32>  { ... }  // add, sub, scalar kernels, map, sum, matmul, matvec, outer
//   impl Tensor<T> { ... }
//   impl Tensor<Trit> { ... }  // pack, unpack, dequantizing read, mixed matmul (subsequent phase)
// An operation absent from a type's block does not exist for that type.
```

The `f32` storage is `Arc<Vec<f32>>`. The `Trit` storage is packed `Arc<Vec<u8>>`
with a per-tensor scale (Sections 3.1 and 3.2). The `i8` element type is deferred
beyond v0.1.0 (maintainer decision, October 2026): no implementation exists until
it is implemented, which is the reserved-variant rule of the v0.1.0 layer descriptor
applied to the element axis.

The device-neutral data record of the crate boundary — the exchange format of the
exporter, the graph renderer, and, in later phases, the device engines — is:

```text
pub struct TensorData {
    pub bytes: Vec<u8>,     // raw packed representation of Section 2.3, logical order
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub scale: Option<f32>, // the per-storage scale: W0 for `Trit`; absent for `f32`
}
```

The record owns its bytes and shares no storage with any tensor. The wire tag of a
record is validated against the static element type at the construction boundary
(Section 4.1).

### 2.2 Invariants

1. `shape.len() == strides.len()`.
2. Total logical elements: $N = \prod_{i=0}^{D-1} \text{shape}[i]$. For scalar tensors ($D=0$), $N=1$.
3. Storage bounds: for all multi-indices $(i_0, \dots, i_{D-1})$ with $0 \le i_k < \text{shape}[k]$, the physical index
   $$\text{idx} = \text{offset} + \sum_{k=0}^{D-1} i_k \cdot \text{strides}[k]$$
   is a **logical element index**, not a byte offset, and must satisfy
   $0 \le \text{idx} < \text{element\_capacity}$ of the backing buffer.
4. A tensor is **contiguous (row-major / C-order)** if and only if both of the following clauses hold:
   $$\text{strides}[D-1] = 1 \quad \text{and} \quad \text{strides}[k] = \text{strides}[k+1] \cdot \text{shape}[k+1] \quad \forall k \in [0, D-2]$$
   $$\text{offset} = 0$$
   The first clause establishes the row-major element order. The second clause establishes that the tensor owns a packed buffer that begins at element zero, which is the precondition of every kernel that addresses the buffer directly. A tensor whose physical offset is non-zero is therefore not contiguous, even when its stride vector is canonical. A tensor whose logical extent is empty is contiguous whenever both clauses hold; the vacuous case carries no behavioral consequence, because such a tensor has no element to transfer.
5. Buffer capacity constraints per element type, expressed as a logical element capacity:
    - The `f32` storage: at least $N$ elements (`f32`).
    - The `Trit` storage: at least $N$ logical elements, materialized as $\lceil N / 4 \rceil$ bytes (`u8`), packing four trits per byte.
    A buffer whose element capacity is below $N$ is a precondition violation at construction.
6. `DType` occupies exactly one byte in memory and is copyable by value.
7. A `Trit` tensor is **byte-aligned** when its `offset` and every entry of its
   `strides` are multiples of four. A ternary view that is not byte-aligned is not a
   valid view, because the four trits sharing a byte cannot be addressed
   independently. Such a view is not produced by the view operations of Section 4 and
   is not eligible for in-place mutation.

### 2.3 Index and Addressing Model

The logical element index of Section 2.2, Invariant 3, is resolved into a physical
address according to the element type. This model is the single normative
statement of how the packing of Section 3.1 reaches the byte buffer.

| Element type | Byte address of element `idx` | Bit position within byte |
| --- | --- | --- |
| `f32` | $\text{idx} \cdot 4$ | Not applicable |
| `i8` (deferred) | $\text{idx}$ | Not applicable |
| `Trit` | $\lfloor \text{idx} / 4 \rfloor$ | $(\text{idx} \bmod 4) \cdot 2$ |

`offset` is always expressed in logical element units, for every element type.

---

## 3. Ternary and Packed Encoding Contracts

TARS supports ternary weights $W \in \{-W_0, 0, +W_0\}$. The design supports a standard
two-phase evolution to prevent future architectural rework.

### 3.1 Canonical Bit-Packed Ternary (BitNet b1.58 / TQ2_0 Standard)

Each trit is encoded using biased 2-bit unsigned integers ($\text{uint2} = \text{trit} + 1$):

| Trit Value | Code (Binary) | Semantics |
| :---: | :---: | :--- |
| $-1$ | `00` | Negative weight: subtract activation |
| $0$ | `01` | Zero: skip (sparsity gating) |
| $+1$ | `10` | Positive weight: add activation |
| Reserved | `11` | Invalid / padding |

- **Packing Density:** 4 trits per byte (`u8`). Byte index: $\lfloor \text{idx} / 4 \rfloor$. Bit shift: $(\text{idx} \pmod 4) \times 2$.
- **Memory Compression:** Consumes $2.0$ bits per element ($16\times$ reduction relative to `f32`, $4\times$ relative to `i8`).
- **Scale Factor:** $W_0 = \frac{1}{mn} \sum_{ij} |W_{ij}|$ is stored per tensor (or per output channel).
- **Dequantization:** The value represented by a trit code $c$ is $\text{trit}(c) \cdot W_0$, where $W_0$ is the scale held by the storage.
- **Reserved Encoding:** The code `11` is an invalid encoding. Reading an element whose code is `11` is a precondition violation and terminates the operation. A ternary buffer allocated by the runtime never contains the reserved code, and a buffer intended to hold zero-valued trits is filled with the code `01` in every bit pair.

### 3.2 Ternary Affine Evaluation

When evaluating an affine projection with ternary weights, inner products eliminate floating-point multiplications:
$$y_i = b_i + W_0 \left( \sum_{j \in S_+} a_j - \sum_{j \in S_-} a_j \right)$$
where $S_+ = \{j \mid w_{ij} = +1\}$ and $S_- = \{j \mid w_{ij} = -1\}$. The scale factor $W_0$ is multiplied once per output accumulation.

The bias term $b_i$ belongs to the forward contract of the linear layer, not to the
tensor product. `Tensor::matmul` is a pure product and returns no bias contribution;
the ternary evaluation above is the specialization that a ternary-aware linear layer
applies to the result of that product.

### 3.3 I8 Scaling Semantics

The scale held by the `i8` storage is a multiplicative factor. A stored integer $q$
represents the value $q \cdot \text{scale}$. The scale is a property of the storage
buffer and is not altered by element access. In the current scope the scale is held
per tensor; a per-channel scale is a subsequent extension. The `i8` element type is
deferred beyond v0.1.0 (maintainer decision, October 2026).

### 3.4 Extensibility to Succinct Mask + Sign Vectors (POPCNT)

The `Trit` storage encapsulation guarantees that migrating internally to a Position Bitmap ($N$ bits) plus a Compact Sign Vector ($K$ bits) evaluated via `POPCNT` requires zero changes to the public `Tensor<T>` API, `Linear` layer, or `backward()` routines.

---

## 4. Minimum Viable Surface for Repository Convergence

To unblock the migration of `Linear`, `backward()`, activations, and optimizers without
premature optimization, the tensor core must provide strictly the following surface.
The surface is organized per element type: an operation is delivered as a member of the
implementation block of its element type, and an operation absent from a block does not
exist for that type.

**Operation structure (kernel seam).** Every method of this surface is a thin
dispatcher: it validates the preconditions of Section 4.1 and delegates the arithmetic
to a kernel function of the per-type kernel module of its element type. A `Tensor`
method contains no arithmetic of its own, and the `f32` kernel module is the CPU
reference implementation of the project. This clause operationalizes Invariance Rule 1
of [`FUTURE_TARGETS.md`](../FUTURE_TARGETS.md): a later device engine replaces kernel
implementations, never the method signatures.

| Method Category | Method Signature | Operational Behavior | Status |
| --- | --- | --- | --- |
| **Constructors** | `new(Storage, Vec<usize>) -> Self` | Wraps storage in canonical C-order strided view at offset 0 | Partial |
| | `zeros(Vec<usize>) -> Self` (per element type) | Allocates the zeroed buffer of the element's storage | Partial |
| **Geometry** | `shape(&self) -> &[usize]` | Borrows shape vector | Implemented |
| | `strides(&self) -> &[usize]` | Borrows strides vector | Implemented |
| | `offset(&self) -> usize` | Returns physical offset | Implemented |
| | `numel(&self) -> usize` | Computes logical product $\prod \text{shape}[i]$ | Implemented |
| | `is_contiguous(&self) -> bool` | Validates the contiguity definition of Section 2.2, Invariant 4 | Partial |
| | `dtype(&self) -> DType` | Returns the one-byte wire tag of the element type | Partial |
| **Element Access** | `get(&self, &[usize]) -> f32` | Logical view: dequantizes value to `f32` for generic consumption | Partial |
| | `set(&mut self, &[usize], f32)` | Mutates storage in-place via `Arc::make_mut` (COW) | Partial |
| **Views & Memory** | `transpose(&self, usize, usize) -> Self` | Zero-copy axis swap modifying `shape` and `strides` | Partial |
| | `transpose_mut(&mut self, usize, usize)` | In-place axis swap of `shape` and `strides`; alters metadata only and does not mutate the storage buffer | Partial |
| | `contiguous(&self) -> Self` | Returns a tensor whose strides are canonical row-major and whose physical offset is zero; shares the source storage when the source already satisfies Section 2.2, Invariant 4, and re-packs the elements by coordinate mapping otherwise | Partial |
| **Linear Algebra** | `matmul(&self, &Tensor) -> Self` | Computes 2D matrix multiplication ($[M, K] \times [K, N] \to [M, N]$) with specialized ternary fast-path | Partial |
| | `matvec(&self, &Tensor) -> Self` | Multiplies matrix $[M, K]$ by vector $[K]$ | Partial |
| | `outer(&self, &Tensor) -> Self` | Outer product of 1D vectors $[M] \times [N] \to [M, N]$ | Partial |
| **Element-wise** | `add(&self, &Tensor) -> Self` | Pointwise logical addition respecting strides | Partial |
| | `sub(&self, &Tensor) -> Self` | Pointwise logical subtraction respecting strides | Partial |
| | `mul_scalar(&self, f32) -> Self` | Pointwise scalar multiplication | Partial |
| | `div_scalar(&self, f32) -> Self` | Pointwise scalar division | Partial |
| | `map(&self, F) -> Self` | Applies a scalar function to logical elements, where $F$ is any callable over `f32` that may capture state | Partial |
| | `sum(&self) -> f32` | Full reduction across all logical coordinates | Partial |
| **Data Transfer** | `into_data(&self) -> TensorData` | Materializes the record of Section 2.1: the logical elements as the raw packed representation of Section 2.3 in logical coordinate order, the shape, the wire tag, and the per-storage scale; the record owns its bytes | Not implemented |
| | `from_data(TensorData) -> Self` | Constructs a contiguous tensor from the record; the wire tag and the byte length are validated against the static element type and the shape | Not implemented |

#### 4.0.1 Gap Notes

- `new` does not yet validate the element capacity of the supplied storage against the shape product, as required by Section 2.2, Invariant 5.
- `zeros` does not yet materialize the ternary fill encoding of Section 3.1 for the `Trit` type; the `Trit` zero-allocation path belongs to the subsequent phase. The `f32` zeroed allocation exists.
- Element access exists for implemented element types only. A type without an implemented access path cannot be constructed, so there is no runtime unsupported-variant path; this supersedes the former empty-diagnostic termination of the container-level design.
- `is_contiguous` applies a third clause, a non-empty backing buffer, which is not part of the definition of Section 2.2, Invariant 4.
- The in-place axis swap alters the shape and stride vectors only. It does not write to the storage buffer and therefore does not trigger the buffer duplication of Section 5. Its diagnostic, shared with the non-in-place variant, is not yet worded according to the policy of Section 4.1, and the operation has no test coverage.
- The stride computation that establishes canonical row-major strides carries an identifier distinct from the re-packing operation, so the two entities coexist without collision. The re-packing operation is delivered under the identifier `make_contiguous`; its reconciliation with the `contiguous` name of this table remains open. The operation has no test coverage.
- The element-wise kernels and the products are implemented over `F32` and honour the stride vectors of their operands. Their shape and dimension preconditions are reported through default assertion text or empty diagnostics that do not name the violated precondition; `div_scalar` does not reject a zero divisor; `map` accepts non-mutating callables only; the kernels and the products have no test coverage.
- The full reduction `sum` is implemented and honours the stride vector. It terminates on a rank-zero operand through the logical iteration helper (defect D-06 of the delivery plan) and has no test coverage.

### 4.1 Preconditions and Error Policy

The tensor core reports precondition violations by terminating the operation with a
descriptive diagnostic. It does not return a recoverable error value in the current
delivery phase. A diagnostic is never empty, and a precondition violation is never
reported as a successful operation.

| Operation | Precondition |
| --- | --- |
| `new` | The element capacity of the storage is at least the shape product (Section 2.2, Invariant 5) |
| `is_contiguous` | The element type is within the active delivery phase (Section 4.2) |
| `get`, `set` | The index tuple has one entry per dimension, and each entry is within the corresponding shape extent |
| `get`, `set` | The element type implements the access path of the active delivery phase (compile time) |
| `set` | The buffer is not shared with a view that requires exclusive ownership; otherwise the buffer is duplicated before mutation (Section 5) |
| `transpose` | Both axis identifiers are within the rank of the tensor; a ternary result is byte-aligned (Section 2.2, Invariant 7) |
| `transpose_mut` | Both axis identifiers are within the rank of the tensor; a ternary result is byte-aligned (Section 2.2, Invariant 7) |
| `contiguous` | The element type supports element transfer in the active delivery phase |
| `matmul` | The left operand has rank 2, the right operand has rank 2, and the inner dimensions agree |
| `matvec` | The left operand has rank 2, the right operand has rank 1, and the inner dimensions agree |
| `outer` | Both operands have rank 1 |
| `add`, `sub` | Both operands have identical shapes; broadcasting is not provided in the current phase |
| `mul_scalar`, `div_scalar`, `map` | The operand is a valid tensor; `div_scalar` requires a non-zero divisor |
| `sum` | The operand is a valid tensor |
| `from_data` | The wire tag of the record equals the tag of the static element type; the byte length of the record satisfies the element capacity of Section 2.2, Invariant 5, for the shape |

### 4.2 Delivery Phases

The minimum viable surface of this section is delivered incrementally. The phases
below partition the surface and state the exit condition of each.

| Phase | Scope | Exit Condition |
| --- | --- | --- |
| **Current: `f32` core** | Constructors, geometry accessors, element access over `f32`, the axis swap in both variants, `contiguous`, the element-wise kernels (`add`, `sub`, `mul_scalar`, `div_scalar`, `map`), `sum`, the products `matmul`, `matvec`, `outer` over `f32`, and the data-transfer surface of Section 2.1 (`into_data`, `from_data`) | Every operation of the phase satisfies Section 4.1 and is covered by the tensor test suite |
| **Subsequent: `Trit` element type** | The `Trit` implementation block: packing and unpacking (Section 3.1), the dequantizing read, the ternary fast path of Section 3.2 (the mixed matmul of activations by ternary weights), and the alignment constraint of Section 2.2, Invariant 7. The `i8` element type and the scaling of Section 3.3 are deferred beyond v0.1.0 (maintainer decision, October 2026) | `Trit` round trips and dequantized values are covered by the tensor test suite |
| **Deferred** | Reductions along arbitrary axes, `mean`, `slice`, `permute`, `reshape`, and multidimensional broadcasting | Each deferred operation receives its own contract in this specification before implementation |

The deferral of axis reductions and of the view and broadcasting operations is
consistent with the delivery gates recorded in the project roadmap. The
corresponding roadmap entries and this table must be reviewed together whenever
either is amended.

---

## 5. Storage Mutation and Copy-on-Write (COW)

1. Storage buffers are guarded by atomic reference counting (`Arc`).
2. In-place mutation via `set` must invoke `Arc::make_mut`, guaranteeing that shared
   buffers (`strong_count > 1`) are cloned before modification without panic.
3. Direct `Arc::get_mut().expect(...)` calls on shared or view-derived tensors are strictly prohibited.
4. Mutation of a view that is not byte-aligned under Section 2.2, Invariant 7, is
   prohibited in the current delivery phase.
5. Copy-on-write is a property of the CPU storage implementations. Other devices
   define their own synchronization model
   (see [`FUTURE_TARGETS.md`](../FUTURE_TARGETS.md), Section 4).

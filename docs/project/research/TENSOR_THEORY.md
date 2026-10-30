# Tensor Theory: Storage, Shape, Strides, Offset, and Views

**Document type:** Theoretical reference and supporting research analysis.

This document explains how a dense strided tensor represents multidimensional
values over linear storage. It covers coordinate addressing, views, memory
layout, copying, aliasing, and numerical representation through equations and
worked examples. It is not an implementation guide, an empirical benchmark, or a
claim that every described operation is available in TARS. The normative project
contract remains the [tensor specification](../../engineering/specs/TENSOR.md).

## Reading Guide

| Question | Sections |
|---|---|
| What are `data`, `shape`, `strides`, and `offset`? | 1–4 |
| How does a coordinate select a stored value? | 4–5 |
| Why can a transpose or slice avoid copying? | 5–6 |
| What do contiguous, reshape, and materialize mean? | 7–8 |
| When do coordinates overlap or exceed storage? | 9–10 |
| What changes with ownership, dtype, or packed elements? | 11–12 |
| How do layout and storage affect computation and export? | 13–15 |
| Which distinctions matter specifically to TARS? | 16 |

---

## 1. The Tensor as Values and as a Descriptor

In machine-learning software, a tensor is commonly a multidimensional collection
of values. A scalar has no axes, a vector has one axis, and a matrix has two.
Higher-dimensional tensors use the same indexing principle with more axes.
Here, **rank** means the number of axes, not matrix rank or tensor decomposition
rank. The mathematical definition of a tensor includes transformation properties
that an array container alone does not establish.

A dense strided tensor separates two concerns:

- **Storage:** the linear region containing element representations.
- **Descriptor:** metadata defining which logical coordinates refer to which
  storage positions and how those representations are interpreted.

A matrix does not require a separate allocation for every row. Its rows and
columns can be an interpretation of one linear buffer. Multiple tensor
descriptors can also interpret the same buffer differently.

The central distinction is therefore between **logical organization** and
**physical representation**. Shape describes the former; strides and offset
connect it to the latter.

This model concerns affine strided layouts. Sparse, ragged, tiled, and compressed
formats can require additional indexing structures and are not fully described
by one shape, one stride vector, and one offset.

---

## 2. Metadata Vocabulary

| Term | Meaning | Unit or example |
|---|---|---|
| `data` or storage | The backing element representations, not necessarily only those visible through the tensor | A linear buffer of twelve `f32` values |
| `shape` | The number of valid coordinates along each axis | `(3, 4)` means three rows and four columns |
| Rank or `ndim` | The number of axes; the length of `shape` | Rank 2 for `(3, 4)` |
| `numel` | The number of logical elements | Twelve for `(3, 4)` |
| `strides` | The storage displacement caused by advancing one coordinate along each axis | `(4, 1)` in element units |
| `offset` | The storage position corresponding to the all-zero coordinate | Five means the logical origin is storage element five |
| `dtype` | The numerical representation and interpretation of elements | `f32`, integer, or packed ternary |
| Ownership | The rules governing storage lifetime, sharing, and mutation | Shared allocation, exclusive allocation, or copy-on-write |
| Device | The execution and memory domain containing the storage | CPU or an accelerator |

The word `data` is not standardized across APIs. It can denote a whole storage
object, a pointer to the tensor's first logical element, or an exported copy of
the logical values. These meanings are not interchangeable. This document uses
**storage** for the backing allocation and states explicitly when a logical
sequence or a pointer is intended.

Device placement, gradient history, and axis names can belong to a tensor
descriptor, but none changes the basic coordinate-to-storage equation.

---

## 3. Shape, Rank, and Logical Size

For a rank-$D$ tensor, let its shape be

$$
d=(d_0,d_1,\ldots,d_{D-1}).
$$

A valid coordinate $i=(i_0,\ldots,i_{D-1})$ satisfies

$$
0\leq i_k<d_k\quad\text{for every axis }k.
$$

The logical element count is

$$
N=\prod_{k=0}^{D-1}d_k.
$$

| Shape | Rank | `numel` | Interpretation |
|---|---|---|---|
| `()` | 0 | 1 | One scalar; its coordinate tuple is also empty |
| `(1,)` | 1 | 1 | A vector with one element |
| `(3, 4)` | 2 | 12 | A matrix |
| `(2, 3, 4)` | 3 | 24 | Two groups, each containing a three-by-four matrix |
| `(0, 4)` | 2 | 0 | An empty tensor, with no valid element coordinate |

The empty product is one, so a rank-zero scalar is not an empty tensor. An axis
of length one also differs from an absent axis, even when both descriptions
contain the same number of values.

Shape alone does not identify the storage layout or semantic meaning of an axis.
For example, a four-axis image tensor needs an additional convention to identify
batch, channel, height, and width. Shape is neither the backing buffer length nor
its allocation capacity.

---

## 4. Coordinate Addressing

### 4.1 From a Coordinate to a Storage Position

Let $s_k$ be the stride of axis $k$ and $o$ the offset, both in **element units**.
The coordinate-to-storage map is

$$
\boxed{p(i)=o+\sum_{k=0}^{D-1}i_ks_k.}
$$

Holding every other coordinate fixed, advancing axis $k$ by one changes $p$ by
$s_k$. A stride is therefore a displacement, not an axis length and not a count
of dimensions.

For a scalar, the sum is empty and $p(())=o$. For an empty tensor there is no valid
coordinate to address.

### 4.2 From a Storage Position to a Byte Address

For ordinary fixed-width elements of $b$ bytes and storage base address $B$,

$$
\operatorname{address}(i)=B+b\,p(i).
$$

For `f32`, $b=4$. A stride of four elements therefore corresponds to sixteen
bytes, not four bytes. Packed sub-byte elements require a separate byte-and-bit
mapping, described in Section 12.

NumPy reports strides in bytes. PyTorch reports strides in element units and
defines `storage_offset` in storage elements. For the same `f32` row-major matrix
of shape `(3, 4)`, these stride vectors are `(16, 4)` and `(4, 1)`, respectively.
The units must be known before comparing metadata between libraries.

There are also two possible pointer origins. If a pointer $P$ already addresses
the first logical element, then $P=B+bo$ and

$$
\operatorname{address}(i)=P+b\sum_k i_ks_k.
$$

Adding the offset again would count it twice. A descriptor or foreign-function
interface must distinguish a storage-base pointer from a logical-origin pointer.

### 4.3 Logical Linear Position Is Not Storage Position

In C-order, the last logical coordinate varies fastest. Its logical linear
position is

$$
\ell(i)=\sum_{k=0}^{D-1}i_k\prod_{j=k+1}^{D-1}d_j.
$$

This is a numbering of logical coordinates, not a general storage address.
Equality $p(i)=\ell(i)$ holds for canonical C-order strides with offset zero,
but does not hold for an arbitrary view.

---

## 5. Worked Example: One Buffer, Several Tensors

Consider storage containing the twelve values from zero through eleven in
ascending order. Let tensor $A$ have shape `(3, 4)`, strides `(4, 1)`, and offset
zero. Its logical matrix is

$$
A=\begin{bmatrix}
0&1&2&3\\
4&5&6&7\\
8&9&10&11
\end{bmatrix}.
$$

For coordinate $(2,1)$,

$$
p_A(2,1)=0+2\cdot4+1\cdot1=9.
$$

The selected value is storage element nine. With `f32` storage, it begins
thirty-six bytes after the storage base.

### 5.1 Transpose Without Moving Values

Swapping the axes produces $T$ with shape `(4, 3)`, strides `(1, 4)`, and the same
offset and storage:

$$
T=\begin{bmatrix}
0&4&8\\
1&5&9\\
2&6&10\\
3&7&11
\end{bmatrix}.
$$

The equation becomes $p_T(i,j)=i+4j$, so $T(i,j)=A(j,i)$. The backing buffer still
contains the original ascending sequence. Only the interpretation changed.

### 5.2 Subsets of the Same Storage

Every descriptor below refers to the original buffer. Slice notation denotes
half-open intervals and is mathematical shorthand, not a claim about TARS APIs.

| Logical selection | Shape | Strides | Offset | Values in C-order |
|---|---|---|---|---|
| All of $A$ | `(3, 4)` | `(4, 1)` | 0 | $(0,1,2,3,4,5,6,7,8,9,10,11)$ |
| Row 1 | `(4,)` | `(1,)` | 4 | $(4,5,6,7)$ |
| Column 2 | `(3,)` | `(4,)` | 2 | $(2,6,10)$ |
| Rows `[1:3]`, columns `[1:4]` | `(2, 3)` | `(4, 1)` | 5 | $(5,6,7,9,10,11)$ |
| All rows, columns `[0:4:2]` | `(3, 2)` | `(4, 2)` | 0 | $(0,2,4,6,8,10)$ |
| Transpose $T$ | `(4, 3)` | `(1, 4)` | 0 | $(0,4,8,1,5,9,2,6,10,3,7,11)$ |

The two-row submatrix has six logical elements, but occupies positions spanning
storage elements five through eleven. Position eight lies inside that span and
is not part of the view. A span, a logical element count, and a buffer length are
three different quantities.

### 5.3 A View of a View

Transposing that submatrix produces shape `(3, 2)`, strides `(1, 4)`, and offset
five. Its coordinate $(2,1)$ addresses

$$
p(2,1)=5+2\cdot1+1\cdot4=11.
$$

The offset is still relative to the original storage base. A chain of affine
views can be described by a composed descriptor; element access does not
conceptually require following every intermediate view.

---

## 6. Views and Metadata Transformations

A **view** provides a different logical interpretation of existing storage.
Creating a view does not copy its element buffer, although creating or adjusting
metadata can require work proportional to the number of axes.

| Operation | Descriptor interpretation | Storage effect |
|---|---|---|
| Transpose or axis permutation | Rearrange shape and stride entries together | None |
| Select one coordinate along an axis | Move the origin along that axis and remove its shape and stride entries | None |
| Regular slice | Move the origin, adjust the selected extent, and scale its stride | None |
| Squeeze | Remove axes whose extent is one | None |
| Unsqueeze | Insert an axis whose extent is one | None |
| Matrix diagonal | Advance both matrix coordinates together | None when the representation permits the view |

For a normalized nonempty slice starting at coordinate $a$ with positive step
$q$ along axis $k$, the descriptor relation is

$$
o'=o+a s_k,\qquad s'_k=q s_k.
$$

The new extent is the number of selected coordinates; the other axes keep their
extents and strides. For multiple sliced axes, their starting displacements add
to the original offset.

Selecting a fixed coordinate $a$ uses the same origin displacement, but removes
that axis rather than retaining a length-one axis. Squeezing a length-one axis
does not change any address because its only valid coordinate is zero.

For a rank-two tensor, the main diagonal has length $\min(d_0,d_1)$ and stride
$s_0+s_1$, since successive diagonal elements advance both coordinates.

An arbitrary gather is different. Selecting vector coordinates $(0,3,1)$ does
not yield one constant stride, so it cannot generally be represented by this
descriptor alone. A library can materialize those values or use a richer indexed
representation. NumPy and PyTorch return copies for advanced-indexing reads.

---

## 7. Contiguity and Memory Order

### 7.1 Canonical Row-Major and Column-Major Strides

Canonical C-order, or row-major, element strides are

$$
s_k^{\mathrm C}=\prod_{j=k+1}^{D-1}d_j.
$$

The final stride is one. For `(2, 3, 4)`, the canonical strides are `(12, 4, 1)`.
Advancing the final coordinate selects the next element; advancing the first
coordinate skips one complete three-by-four group.

Canonical Fortran-order, or column-major, element strides are

$$
s_k^{\mathrm F}=\prod_{j=0}^{k-1}d_j.
$$

For `(3, 4)`, C-order strides are `(4, 1)` and F-order strides are `(1, 3)`.
These are distinct interpretations of axis traversal, not distinct numerical
types. C-order and F-order are also not synonymous with semantic image formats
such as channel-first and channel-last.

### 7.2 Dense Storage Does Not Imply C-Contiguous Traversal

For nonempty shapes without length-one axes, canonical C-order strides satisfy

$$
s_{D-1}=1,\qquad s_k=s_{k+1}d_{k+1}.
$$

In general array terminology, a C-contiguous view traverses consecutive storage
elements in increasing order during C-order logical traversal. Its offset need
not be zero. Row 1 in Section 5 is C-contiguous even though it starts at storage
element four and shares a larger allocation.

The transpose in Section 5 covers every original storage element without gaps,
but is not C-contiguous. It is F-contiguous. The column view has gaps between
elements and is neither C-contiguous nor F-contiguous when treated as a vector.

Contiguity does not imply exclusive ownership, and offset zero does not imply
exclusive ownership either. Conversely, an exclusively owned allocation can have
a noncontiguous tensor descriptor.

### 7.3 Singleton and Empty Axes

For an axis of extent one, its stride does not affect any valid address. Shape
`(2, 1, 3)` with strides `(3, 99, 1)` describes the same addresses as canonical
strides `(3, 3, 1)`. General contiguity predicates can ignore that singleton
stride rather than requiring exact equality with the canonical vector.

Empty tensors have no valid addresses, so their contiguity is a convention.
NumPy considers empty arrays both C-contiguous and F-contiguous. Scalar handling
also requires a rank-zero convention rather than indexing a nonexistent final
stride. Library predicates can deliberately use stricter definitions.

The TARS-specific requirement for canonical strides and offset zero is discussed
in Section 16; it must not be substituted for the general definition above.

---

## 8. Reshape, Flatten, and Materialization

### 8.1 Reshape Preserves an Element Sequence

A reshape changes the grouping of a logical element sequence into axes.
Preserving the element count is necessary:

$$
\prod_k d_k=\prod_j d'_j.
$$

It is not sufficient to guarantee a zero-copy result. For a chosen logical order,
the new coordinates must address that same sequence through a valid stride map.

The canonical `(3, 4)` tensor can be reshaped to `(2, 6)` with strides `(6, 1)`
without moving storage. This is not equivalent to transposing it. Reshape
preserves the chosen logical sequence; transpose changes how axes select values.

Full C-contiguity is sufficient for a C-order reshape, but is not always
necessary. A length-six vector with stride two can be described as shape
`(3, 2)` with strides `(4, 2)`, preserving its sequence without copying. Its
addresses remain spaced by two elements.

For adjacent nonsingleton axes being merged, the relation
$s_k=d_{k+1}s_{k+1}$ identifies a compatible traversal block. Singleton axes and
empty shapes require their own conventions. API names such as `view`, `reshape`,
and `flatten` do not establish identical copy guarantees across libraries.

### 8.2 Materialization Preserves Values, Not the Original Layout

To materialize a tensor in canonical C-order is to place its logical values in
that order into suitable storage and give the result canonical strides and
offset zero. For the transpose $T$, the materialized element sequence is

$$
(0,4,8,1,5,9,2,6,10,3,7,11).
$$

Its shape remains `(4, 3)`, but its new strides are `(3, 1)`. Copying the original
buffer unchanged and merely replacing the strides would produce different
logical values.

The same transposed sequence cannot be flattened into a one-axis affine view in
C-order: successive storage differences are not constant. A contiguous flattened
result therefore requires materialization for this example.

A request for contiguous layout is not necessarily a request for independent
storage. An API may reuse already suitable storage. A deep copy requires a
separate ownership guarantee even when no layout conversion is needed.

---

## 9. Broadcasting, Reversal, and Overlapping Addresses

### 9.1 Broadcasting and Zero Strides

Broadcasting aligns axes from the right. Corresponding extents must agree or one
must equal one; absent leading axes are treated as length one. For example,
`(3, 4)` and `(4,)` are compatible, whereas `(3, 4)` and `(3,)` are not.

Conceptually expanding a vector $(10,20,30,40)$ to shape `(3, 4)` can use strides
`(0, 1)` and offset zero:

$$
p(i,j)=0i+j=j.
$$

The expanded tensor has twelve logical elements but refers to only four distinct
storage positions. No physical duplication is needed for reads. A runtime may
instead implement this mapping inside the consuming operation without creating
a persistent broadcast view.

Writing through expanded axes is ambiguous when different logical results
target the same physical position. Such writes require an explicit policy;
they are commonly restricted or require materialization.

### 9.2 Negative Strides

In a representation supporting signed strides, reversing a length-four vector
can use shape `(4,)`, stride `(-1,)`, and offset three. The addressed positions
are $(3,2,1,0)$.

This shows that offset identifies the **logical origin**, not necessarily the
smallest addressed position. Signed-stride support is library-dependent; it is
not implied by the strided model and cannot be represented by unsigned stride
fields alone.

### 9.3 Overlap Is More General Than Broadcasting

Shape `(2, 2)`, strides `(1, 1)`, and offset zero address positions
$(0,1,1,2)$ in C-order. All strides are positive, but two coordinates alias the
same element. Sliding-window views can produce similar overlap.

Bounds safety and absence of overlap are independent properties. A read can be
within bounds while an in-place operation remains order-dependent because of
aliasing. Copy-on-write between tensor objects does not remove overlapping
coordinates inside one descriptor.

---

## 10. Descriptor Validity and Storage Bounds

A valid descriptor must have equal shape and stride lengths, supported extents
and stride signs, and a representation capable of addressing every valid
coordinate. Arithmetic used for sizes and displacements must also remain within
the integer representation; overflow can invalidate an apparently valid bound.

Let $L$ be the number of accessible, initialized storage elements, not the
allocation's spare capacity. For a nonempty tensor with nonnegative strides,

$$
p_{\max}=o+\sum_k(d_k-1)s_k,
\qquad 0\leq o\leq p_{\max}<L.
$$

For a nonempty tensor supporting signed strides, the address extrema are

$$
p_{\min}=o+\sum_k\min\bigl(0,(d_k-1)s_k\bigr),
$$

$$
p_{\max}=o+\sum_k\max\bigl(0,(d_k-1)s_k\bigr).
$$

The bounds requirement becomes $0\leq p_{\min}$ and $p_{\max}<L$. For an empty
tensor, these expressions involving $d_k-1$ must not be used as element-access
requirements; there is no coordinate to dereference.

The column view in Section 5 has $N=3$ but reaches position ten. Three backing
elements would not suffice. Conversely, an overlapping view can have $N>L$
without any read exceeding storage bounds. Thus checking only $N\leq L$ neither
describes arbitrary views nor proves their validity.

A constructor can impose stronger conditions, such as exact storage length for
a newly owned tensor or a prohibition on overlapping views. Packed
representations can add alignment, padding, and code-validity requirements.
Metadata must remain mutually consistent after any permitted modification.

---

## 11. Storage Lifetime, Sharing, and Mutation

### 11.1 Shared Lifetime Is Not a Mutation Policy

Views must retain a valid relationship to their backing storage. Reference
counting is one possible lifetime mechanism: storage remains allocated while
any owner retains it. Copying a tensor descriptor can then share values instead
of duplicating them.

This has a memory-retention consequence. A small view can keep a much larger
buffer alive. Materializing only the selected values can release that dependency
once no other owner retains the original buffer.

Reference counting alone does not define whether writes are visible through
other tensors and does not make concurrent writes safe.

### 11.2 Shared Mutation and Copy-on-Write Are Different Contracts

| Policy | Effect of writing through a tensor that initially shares storage |
|---|---|
| Shared mutable views | Other descriptors observe changes at the same storage positions |
| Read-only views | Mutation is rejected or unavailable |
| Copy-on-write | Shared storage is duplicated before a permitted write; other descriptors keep the original values |

NumPy and PyTorch views ordinarily expose shared mutation. TARS specifies
copy-on-write for its CPU storage. These are different observable semantics even
when the same metadata equations apply.

Changing shape and strides is distinct from changing the buffer. An in-place
axis swap can modify only one descriptor's metadata and need not duplicate its
storage. A subsequent element write may trigger copy-on-write.

Copy-on-write can duplicate the whole backing buffer rather than only the
view's $N$ logical values. Its allocation cost can therefore scale with storage
length $L$. Neither descriptor cloning nor making a tensor contiguous inherently
guarantees a deep copy, and neither operation alone defines synchronization for
an accelerator.

---

## 12. DType, Bytes, and Packed Elements

### 12.1 Numerical Interpretation Is Separate from Geometry

Shape, strides, and offset identify logical storage positions. Dtype determines
how a selected position represents a value. Equal geometry does not make two
buffers numerically equivalent when their dtypes differ.

A numerical conversion changes representations to preserve values as far as the
target type permits. A byte reinterpretation preserves bits and changes their
meaning. Reinterpreting `f32` bytes as integers is not a floating-point-to-integer
conversion and can impose different size and alignment constraints.

For ordinary fixed-width storage, the logical payload is $Nb$ bytes. This does
not measure retained storage for a view, allocation overhead, or the span between
its first and last addresses. Quantized representations may additionally need
scale, zero-point, or other decoding metadata.

### 12.2 Four Two-Bit Elements per Byte

Consider a ternary representation using two bits per logical element, with four
elements packed per byte. The coordinate map still produces an **element**
position $p$. The representation then selects

$$
\operatorname{byte}(p)=\left\lfloor\frac p4\right\rfloor,
\qquad
\operatorname{bit}(p)=2(p\bmod4).
$$

Position seven selects byte one and the bit pair beginning at bit six. A stride
of one advances one ternary element, not one byte. Fractional byte strides are
not a substitute for this element-to-bit mapping.

A newly packed logical sequence of $N$ elements needs $\lceil N/4\rceil$ bytes,
plus any numerical metadata. Unused bit pairs in the final byte are padding,
not extra logical elements; shape determines the logical count.

The TARS target encoding uses `00` for $-1$, `01` for zero, `10` for $+1$, and
reserves `11`. Consequently, zero-filled bytes would represent negative trits,
not zero-valued trits. A full byte of four zero-valued trits is `01010101`.
With per-tensor scale $W_0$, a decoded trit $t$ represents $tW_0$.

Individual bit pairs can be selected mathematically, but updating one pair also
accesses a byte containing neighboring elements. A runtime may therefore impose
stricter view or mutation alignment rules. Such rules are representation and
concurrency contracts, not consequences of the affine coordinate equation alone.

---

## 13. Layout and Numerical Operations

An operation is defined over logical coordinates. Its operands need not place
corresponding coordinates at the same physical positions.

For pointwise addition without broadcasting,

$$
C(i)=A(i)+B(i),
$$

but $p_A(i)$ and $p_B(i)$ may differ. Adding backing buffers position by position
is correct only when the participating layouts satisfy that assumption. Equal
shape alone does not establish it.

For matrix multiplication,

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj},
\qquad
(M,K)\times(K,N)\longrightarrow(M,N).
$$

The formula is independent of storage order. A stride-aware kernel may consume
views directly; a restricted kernel may require materialized operands. Neither
choice changes the mathematical operation.

A full reduction counts **logical elements**, not distinct backing positions.
Summing a zero-stride expansion of one scalar across five coordinates contributes
that scalar five times. A reduction over one axis removes that axis or retains
it at extent one, according to the operation's shape contract.

Accumulation dtype and evaluation order are separate numerical choices.
Floating-point addition is not associative, so layout-dependent execution orders
can produce different rounding without changing the real-number expression.

For differentiable systems, views also have logical reverse mappings: transpose
gradients reverse the permutation, and repeated selections or broadcast uses
require combining gradient contributions. This is a statement about automatic
differentiation semantics, not a claim that TARS provides these general view
gradient operations.

---

## 14. Data Export and Serialization

A raw backing buffer does not, by itself, describe the values of an arbitrary
view. Exporting the transpose from Section 5 as an ascending buffer with shape
`(4, 3)` would lose its logical meaning unless the receiver also recovered the
correct layout.

Two distinct interchange contracts are possible:

| Contract | Required information | Consequence |
|---|---|---|
| Storage-preserving descriptor | Backing storage, shape, strides, offset, dtype, and relevant representation and lifetime rules | Can preserve layout and sharing when the receiver supports them |
| Materialized value record | Logical values in a declared order, shape, dtype, and decoding metadata | Does not need the original strides or offset; sharing is not preserved |

The target TARS `TensorData` record follows the second model: it owns the
logical-order bytes and records shape, dtype, and an optional scale. This is a
specified boundary contract, not evidence that the export methods are complete.

A persistent wire format must additionally define byte order, packing order,
padding validity, and numerical encoding. An in-memory pointer cannot be
serialized as a replacement for its values. Device transfer also requires
explicit lifetime and synchronization rules beyond this indexing model.

---

## 15. Performance Consequences Without Benchmark Claims

| Property | Theoretical consequence |
|---|---|
| Metadata-only transpose or slice | No element-buffer transfer; descriptor work can scale with rank |
| Materializing $N$ fixed-width values | Work and destination payload scale with $N$ |
| Copy-on-write of a shared backing buffer | First-write duplication can scale with $L$, not the selected view size |
| Unit stride in the actively traversed axis | Adjacent accesses can improve cache locality and support vectorized transfers |
| Large gaps between accesses | More transferred cache-line or device data can be unused by that traversal |
| Reusing a small view | Avoids immediate copying but may retain the whole original allocation |
| Packed elements | Reduce payload size while requiring representation-aware access and mutation |

A zero-copy operation can defer cost to later arithmetic. Conversely,
materializing once can benefit repeated dense operations. The result depends on
tensor dimensions, access order, hardware, kernel design, and reuse count.
Contiguity alone neither proves optimal performance nor replaces measurement.
This document reports no timing, memory-bandwidth, or accelerator measurements.

---

## 16. Relationship to TARS

The [tensor specification](../../engineering/specs/TENSOR.md) defines project
contracts. The [tensor delivery plan](../TENSOR_PHASE_B.md) tracks convergence.
The observations below distinguish those contracts from source structure visible
during this document's preparation in October 2026; they are not test results.

| Topic | TARS context | Distinction from the general theory |
|---|---|---|
| Geometry | [`src/math/tensor.rs`](../../../src/math/tensor.rs) declares shape, strides, and offset as `usize` metadata | The stored strides cannot express negative-stride reversal |
| Addressing | The source uses offset plus the coordinate–stride products to select storage elements | These displacements are not byte counts |
| Storage | The inspected tensor field is `Arc<Vec<T>>`; `Element` also declares an associated storage type | The generic field alone does not establish the specification's packed per-type storage representation |
| Sharing | The source shares the `Arc` for transpose and uses `Arc::make_mut` for element writes | A TARS write need not be visible through another previously shared descriptor |
| Contiguity | The target contract requires canonical C-order strides and offset zero; the inspected predicate also requires nonempty backing storage | This is stricter than common layout-only contiguity and does not prove exclusive ownership |
| Ternary representation | The specification describes packed `T3`; [`ternary_tensor.rs`](../../../src/math/ternary_tensor.rs) contains a preliminary type and packing notes | Section 12 explains the target representation, not completed packed tensor access |
| Ternary views | The target contract adds multiples-of-four alignment constraints on offset and strides | Mathematically addressable packed elements are not automatically valid project views |
| Extended operations | General slicing, permutation, reshape, axis reductions, and multidimensional broadcasting are deferred in the specification | Their theoretical descriptions here do not extend the supported API |
| Data exchange | `TensorData` and its transfer methods are specified in the target contract | A theoretical exchange model does not certify implemented export behavior |

This document neither changes those contracts nor resolves differences between
the specification and an evolving source tree. In particular, offset zero,
canonical strides, storage sharing, and numerical packing must be evaluated as
separate properties.

---

## 17. Common Misinterpretations

| Statement | Correction |
|---|---|
| Shape determines where every value is stored | Addressing also requires strides and offset |
| A tensor is necessarily a nested collection of row allocations | One linear buffer can represent all axes |
| `numel` equals backing buffer length | A view can select a subset, skip positions, or repeat positions |
| Strides always count bytes | Their unit is API-dependent; the formulas here use logical elements |
| Offset must be zero for a generally contiguous view | Offset locates the view's origin; TARS imposes a separate stricter contract |
| Transpose physically rearranges the original buffer | A strided transpose can change metadata only |
| Reshape and transpose are interchangeable | Reshape preserves a chosen sequence; transpose changes the coordinate permutation |
| Equal element counts guarantee a zero-copy reshape | The address sequence must also admit the requested stride description |
| Positive strides imply unique addresses | Positive-stride descriptors can overlap |
| Shared lifetime implies shared-write visibility | Mutation policy determines whether writes propagate, copy, or fail |
| Contiguous implies independently owned | A contiguous descriptor can still share backing storage |
| Packed zero-valued elements require zero-filled bytes | The encoding determines the bit pattern representing numerical zero |

---

## 18. References

The external references establish general array behavior; they do not define
TARS contracts. NumPy and PyTorch documentation was consulted in October 2026.

| Reference | Relevant material |
|---|---|
| [NumPy: The N-dimensional array](https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray) | Linear storage, byte strides, memory orders, singleton and empty-array contiguity |
| [NumPy: Copies and views](https://numpy.org/doc/stable/user/basics.copies.html) | Metadata-only views, advanced-indexing copies, and reshape behavior |
| [PyTorch: Tensor Views](https://docs.pytorch.org/docs/2.15/tensor_view.html) | Shared storage, transpose, contiguous conversion, and copy-dependent reshape operations |
| [PyTorch: Tensor.stride](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.stride.html) | Per-axis displacement in element units |
| [PyTorch: Tensor.storage_offset](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.storage_offset.html) | Offset measured in storage elements rather than bytes |
| [PyTorch: Tensor.view](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.view.html) | Shape–stride compatibility for zero-copy reshaping |
| [TARS tensor specification](../../engineering/specs/TENSOR.md) | Normative geometry, storage, packed ternary, mutation, and transfer contracts |

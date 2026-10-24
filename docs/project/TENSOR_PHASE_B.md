# Tensor Core Delivery Plan: F32 Phase

**Document type:** Planning record. The work packages described here are not part of
the current public library until the corresponding contract clauses are implemented
and verified.

This document decomposes the delivery phase defined in
[`docs/engineering/specs/TENSOR.md`](../engineering/specs/TENSOR.md), Section 4.2, into
ordered work packages with testable exit conditions. It addresses the deviations
between the published tensor contract and the current implementation of
`src/math/tensor.rs`.

---

## 1. Objective and Exit Condition

The objective of this phase is to make the `F32` path of the tensor core satisfy the
minimum viable surface of the specification, so that the migration of the linear
layer, the analytical backward pass, and the optimizer off nested vectors becomes
unblocked.

The phase is complete when every clause of the following holds:

1. Every operation of the `F32` scope of the specification satisfies the preconditions
   of Section 4.1 of the specification.
2. Every operation of the scope is covered by the tensor test suite, including the
   failure paths of the preconditions.
3. No diagnostic emitted by the tensor core is empty.
4. The compiler reports no warning attributable to the tensor module.
5. The documentation under `docs/` describes the resulting public surface, and the
   status column of the specification matches the implementation.

---

## 2. Entry Conditions

| Condition | State |
|---|---|
| The package compiles without error | Satisfied |
| The published contract enumerates the required surface and its preconditions | Satisfied by the current revision of the specification |
| The tensor test suite exists | Partial; one test covers the construction and the contiguity check |
| The tensor core is consumed outside its own module | Not satisfied; no other module constructs a tensor, so no downstream regression constrains the redesign |
| The linear layer, the optimizer, and the exporter use nested vectors | Satisfied; the migration is the objective of the later phase |

### 2.1 Delivery State (October 2026)

The surface of WP-02, WP-06, WP-07, WP-08, and WP-09 is present in the repository. A
package closes only when its acceptance criteria are verified by the test suite,
which does not yet exist in the required scope, so the delivered packages remain open
in the milestone sense.

| Packages | State |
|---|---|
| WP-02 | Closed |
| WP-06, WP-07, WP-08, WP-09 | Implemented; pending verification by the test suite |
| WP-04, WP-05 | Partially implemented; the addressing remains variant-coupled and the view contract clauses are open |
| WP-01, WP-03, WP-10, WP-11, WP-12, WP-13 | Open |

### 2.2 Revision (October 2026): Typed-Element Design

The specification was revised to compile-time element types (`Tensor<T>`), replacing
the container-level storage dispatch against which this plan was originally written.
The work packages survive the revision in substance; the mapping is recorded below.

| Item | Status under the revised specification |
|---|---|
| WP-01 | Unchanged: the one-byte `Copy`/`PartialEq` wire tag |
| WP-02, WP-03, WP-05, WP-06, WP-07, WP-08, WP-09, WP-10, WP-13 | Unchanged in substance; the acceptance criteria apply to the `f32` implementation block |
| WP-12 | Reshaped (October 2026): the public surface is the data round trip of Section 2.1 of the specification (`into_data`/`from_data`); raw buffer views are internal to the crate, per Invariance Rule 2 of [`FUTURE_TARGETS.md`](../engineering/FUTURE_TARGETS.md) |
| WP-04 | Strengthened: the per-element-type physical addressing rule becomes the associated `Storage` contract of the `Element` trait; defect D-03 is enforced by the type system |
| WP-11 | Reduced: the unsupported-variant diagnostics (D-02, D-15) are superseded by compile-time absence; the remaining scope is the precondition diagnostics (capacity, indices, zero divisor, alignment) |
| WP-14 | Reframed as the `Trit` implementation block: packing, the dequantizing read, the mixed matmul, and byte alignment |

The migration of the source to the typed contract precedes the continued closure work
and is recorded in the current assignments of [`ROADMAP.md`](ROADMAP.md).

---

## 3. Work Package Decomposition

The packages are ordered by dependency. A package may start only after the packages
it depends on are closed. No package may be closed while its acceptance criteria are
unverified by the test suite.

### 3.1 Milestone Sequencing

The packages are grouped into milestones. A milestone is closed when every package it
contains is closed and its exit condition is verified. The milestone boundaries are
advisory for the execution order; the package dependencies remain the normative
constraint.

| Milestone | Packages | Exit condition |
|---|---|---|
| M0 — Foundation | WP-01, WP-02, WP-10, WP-11 | The package compiles without a warning attributable to the tensor module; the type tag occupies one byte; every reachable precondition violation reports a non-empty diagnostic; the public contiguous operation is separated from the internal stride helper |
| M1 — Geometry and views | WP-03, WP-04, WP-05 | No precondition violation is reported on a valid input; the constructor rejects an insufficient element capacity; the index resolution yields a logical element index; the axis swap and its in-place variant satisfy the contract of Section 4 of the specification |
| M2 — Memory and kernels | WP-06, WP-07, WP-08 | The re-packed elements of a non-contiguous source preserve the source element order; each kernel reproduces a hand-computed reference on a contiguous and on a non-contiguous operand; the reduction reproduces a hand-computed reference |
| M3 — Products | WP-09 | Each product reproduces a hand-computed reference; the products reproduce the forward pass of the linear layer element by element, in the orientation recorded in Section 6 of this document |
| M4 — Access and coverage | WP-12, WP-13 | The bulk access surface is available to the downstream consumers; the test matrix of Section 5 of this document is entirely green, including the failure path of every precondition |
| M5 — Subsequent phase | WP-14 | Recorded as the subsequent phase of Section 4.2 of the specification; listed here because the work of the current phase constrains its design |

### WP-01 — Type Tag Normalization

| Field | Content |
|---|---|
| Objective | Reduce the numerical type tag to a one-byte copyable value with equality, as required by Section 1.1 and Section 2.2, Invariant 6 of the specification |
| Deliverables | The type tag carries a one-byte representation, derives copying and comparison, and the accessor that reports the tag returns it by value |
| Acceptance criteria | The tag occupies one byte; it is copyable without cloning; two tags of the same variant compare equal; the accessor compiles for a caller that holds the result by value |
| Dependencies | None |
| Risk | Low; the tag is not consumed by any other module |

### WP-02 — Contiguous Operation Identifier Separation

| Field | Content |
|---|---|
| Objective | Expose the public `contiguous` operation of Section 4 of the specification, whose identifier is already separated from the internal stride helper |
| Deliverables | The public `contiguous` operation is declared with the signature of Section 4 of the specification; the internal helper that derives canonical row-major strides retains its own identifier and remains private |
| Acceptance criteria | The public operation is reachable from outside the module; the internal helper remains private; no diagnostic refers to a public contiguous operation as a stride helper |
| Dependencies | WP-01 |
| Risk | Low; the identifier separation is already in place |

### WP-03 — Construction Validation

| Field | Content |
|---|---|
| Objective | Satisfy Section 2.2, Invariant 5 of the specification, under which a buffer whose element capacity is below the shape product is a precondition violation at construction |
| Deliverables | The constructor validates the element capacity of the supplied storage against the shape product and reports a descriptive diagnostic on violation; the shape and stride vectors always have equal length; a rank-zero shape is admitted and denotes a scalar |
| Acceptance criteria | A storage smaller than the shape product is rejected with a non-empty diagnostic; a storage at least as large as the shape product is accepted; a rank-zero tensor reports one element; a rank-zero tensor answers element access without panic |
| Dependencies | WP-01, WP-02 |
| Risk | Medium; the current test suite builds a storage larger than the shape product, which must remain admissible |

### WP-04 — Addressing Model Conformance

| Field | Content |
|---|---|
| Objective | Conform the element address resolution to Section 2.3 of the specification, under which the resolved index is a logical element index rather than a byte offset |
| Deliverables | The index resolution yields a logical element index for every variant; the resolution is expressed independently of the storage variant, so that the physical address rule of Section 2.3 becomes the only place where packing is applied |
| Acceptance criteria | The resolved index of a contiguous tensor matches the canonical row-major order; the resolved index of a transposed view matches the permuted element order; the index resolution is not specialized per variant |
| Dependencies | WP-02 |
| Risk | Medium; the current resolution is used by every element access |

### WP-05 — View Integrity

| Field | Content |
|---|---|
| Objective | Make the axis swap conform to Section 4.1 of the specification, which requires the axis identifiers to be within the rank and requires a ternary result to be byte-aligned |
| Deliverables | The axis swap rejects an identifier outside the rank with a descriptive diagnostic; the resulting view preserves the physical offset; the resulting view shares the storage without copying; the in-place variant is brought under the contract of Section 4 of the specification, which records that it alters metadata only and does not write to the storage buffer |
| Acceptance criteria | An out-of-range identifier is rejected; a swap over a rank-two tensor exchanges both the shape and the stride entries; the storage is shared rather than duplicated; the physical offset is unchanged by the swap; a rank-zero tensor rejects the operation with a descriptive diagnostic; the in-place variant exchanges the same entries and leaves the storage buffer unchanged |
| Dependencies | WP-02, WP-04 |
| Risk | Low; the operation has no test coverage today and no external consumer |

### WP-06 — Contiguous Operation

| Field | Content |
|---|---|
| Objective | Deliver the public `contiguous` operation of Section 4 of the specification |
| Deliverables | The operation returns a tensor whose strides are canonical row-major; when the source is already contiguous the result shares the source storage and preserves the physical offset; otherwise the elements are transferred by coordinate mapping into a newly allocated buffer whose physical offset is zero; the element transfer respects the stride vector of the source |
| Acceptance criteria | A contiguous source yields a result with the same shape, the same strides, and the same physical offset; a source whose strides are canonical but whose physical offset is non-zero is re-packed, because it does not satisfy Section 2.2, Invariant 4; a non-contiguous source yields a result whose element order matches the source element order; the result of a re-packed source reports a physical offset of zero; the transfer is correct for a source whose stride vector was produced by an axis swap |
| Dependencies | WP-02, WP-03, WP-04, WP-05 |
| Risk | Medium; the coordinate transfer depends on the correctness of the index iteration helper, covered by WP-10 |

### WP-07 — Element-Wise Kernels

| Field | Content |
|---|---|
| Objective | Deliver the pointwise operations of Section 4 of the specification: `add`, `sub`, `mul_scalar`, `div_scalar`, and `map` |
| Deliverables | Each operation traverses the logical coordinates of its operand and reads through the stride vector rather than over the raw buffer; `add` and `sub` require identical shapes and reject a mismatch with a descriptive diagnostic; `map` accepts any callable over the element type and permits captured state; `div_scalar` rejects a zero divisor |
| Acceptance criteria | Each operation reproduces the expected result for a contiguous operand against a hand-computed reference; each operation reproduces the expected result for a non-contiguous operand, proving that the stride vector is honoured; `add` and `sub` reject operands of different shapes; `map` accepts a callable that captures mutable state and observes the mutation; `div_scalar` rejects a zero divisor; the result of each operation is contiguous with a physical offset of zero |
| Dependencies | WP-03, WP-04, WP-10 |
| Risk | Medium; the kernels are the first operations that traverse strided data, and they establish the traversal pattern reused by the products |

### WP-08 — Reduction

| Field | Content |
|---|---|
| Objective | Deliver the full reduction `sum` of Section 4 of the specification |
| Deliverables | The reduction accumulates over all logical coordinates through the stride vector; the result is a scalar value; the reduction over a rank-zero tensor returns the single element; the reduction over an empty logical extent returns the additive identity |
| Acceptance criteria | The reduction of a contiguous operand matches a hand-computed reference; the reduction of a non-contiguous operand matches the reduction of the equivalent contiguous operand; the reduction of a rank-zero tensor returns its element |
| Dependencies | WP-04, WP-10 |
| Risk | Low |

### WP-09 — Products

| Field | Content |
|---|---|
| Objective | Deliver the linear algebra operations of Section 4 of the specification: `matmul`, `matvec`, and `outer` |
| Deliverables | `matmul` accepts two rank-two operands, requires the inner dimensions to agree, and produces a rank-two result; `matvec` accepts a rank-two left operand and a rank-one right operand and produces a rank-one result; `outer` accepts two rank-one operands and produces a rank-two result; each operation rejects an incompatible rank or a mismatched inner dimension with a descriptive diagnostic; the products traverse the operands through their stride vectors |
| Acceptance criteria | Each product reproduces a hand-computed reference for contiguous operands; each product reproduces the same reference for non-contiguous operands; the inner-dimension mismatch is rejected; the rank precondition of each product is rejected; the result of each product is contiguous |
| Dependencies | WP-04, WP-07, WP-10 |
| Risk | Medium; the inner dimension of the left operand is traversed in the non-contiguous case, which is the most common layout produced by an axis swap |

### WP-10 — Logical Index Iteration

| Field | Content |
|---|---|
| Objective | Remove the defect by which the logical index iteration underflows on a rank-zero tensor, and make the iteration correct for every rank |
| Deliverables | The iteration helper advances the coordinate vector over all logical positions of a shape, including the rank-zero shape, and terminates without an out-of-range access; the helper is shared by the element transfer, the element-wise kernels, the reduction, and the products |
| Acceptance criteria | The iteration visits every logical position exactly once for shapes of rank one, two, and three; the iteration terminates on a rank-zero shape; the iteration does not access an out-of-range position for a shape containing an empty extent |
| Dependencies | None |
| Risk | Low; the helper is currently private and unused |

### WP-11 — Diagnostic Quality

| Field | Content |
|---|---|
| Objective | Satisfy the error policy of Section 4.1 of the specification, under which a precondition violation is reported with a non-empty descriptive diagnostic and is never reported as a successful operation |
| Deliverables | Every violation reachable from the public surface reports a diagnostic that names the violated precondition; no violation is reported through an empty diagnostic; no violation is silently ignored; the operations that are outside the active phase report an explicit unsupported-variant diagnostic rather than terminating without context |
| Acceptance criteria | Each reachable violation produces a diagnostic that is not empty; the failure paths of every precondition of Section 4.1 are exercised by the test suite; an access to a packed variant under the current phase reports a diagnostic that names the variant |
| Dependencies | WP-03, WP-07, WP-09 |
| Risk | Low |

### WP-12 — Bulk Access for Downstream Consumers

This package belongs to the current phase of Section 4.2 of the specification, which
places the data-transfer surface inside the scope of the `F32` core.

| Field | Content |
|---|---|
| Objective | Provide the access surface required by the exporter, the graph renderer, and the later migration of the linear layer, as the device-neutral data round trip of Section 2.1 of the specification |
| Deliverables | The public surface is `into_data` and `from_data` over the data record: the logical elements as the raw packed representation in logical order, the shape, the wire tag, and the per-storage scale; the record owns its bytes and shares no storage with any tensor; raw buffer views are internal to the crate and serve the exporter, the graph renderer, and the kernel implementations only |
| Acceptance criteria | The data round trip of a contiguous tensor preserves the encoded state; `into_data` of a non-contiguous tensor lists the elements in logical order; the record of a rank-zero tensor contains the single element; the record of a tensor whose logical extent is empty is empty; `from_data` rejects a wire tag that does not match the static element type with a non-empty diagnostic; `from_data` rejects a byte length below the element capacity of the shape |
| Dependencies | WP-03, WP-04, WP-06 |
| Risk | Low; no consumer exists yet, and the design remains open |

### WP-13 — Tensor Test Suite

| Field | Content |
|---|---|
| Objective | Establish the coverage required by the exit condition of Section 1 of this document |
| Deliverables | The suite covers the geometry accessors, the construction preconditions, the contiguity check, the axis swap and its in-place variant, the public contiguous operation, the element-wise kernels, the reduction, the products, the bulk access surface, the logical index iteration, the copy-on-write behaviour, and the failure path of every precondition of Section 4.1 of the specification |
| Acceptance criteria | Every operation of the phase is exercised; every precondition has at least one failure-path assertion; the copy-on-write behaviour is proven by mutating a tensor that shares its buffer with another tensor and observing that the other tensor is unaffected; the in-place axis swap is proven to leave the storage buffer unchanged; the suite passes deterministically and without a randomized seed |
| Dependencies | WP-01 through WP-12 |
| Risk | Low |

### WP-14 — Packed Variants

This package belongs to the subsequent phase of Section 4.2 of the specification and is
recorded here because the work of the current phase constrains its design.

| Field | Content |
|---|---|
| Objective | Deliver the element access over the `T3` variant, the ternary dequantization of Section 3.1, and the ternary evaluation of Section 3.2; the `I8` variant and the scaling of Section 3.3 are deferred beyond v0.1.0 (maintainer decision, October 2026) |
| Deliverables | The packed variants answer element access with a dequantized value; the `I8` scale is multiplicative as defined by Section 3.3; the ternary codes follow the encoding table of Section 3.1, in which the code reserved for an invalid state is never produced and never read; a buffer intended to hold zero-valued trits is materialized with the zero code in every bit pair; the axis swap rejects a ternary result that is not byte-aligned, as required by Section 2.2, Invariant 7 |
| Acceptance criteria | A packed round trip preserves the encoded state; the dequantized value of a ternary element equals the trit multiplied by the scale; a zero-filled ternary buffer decodes to zero-valued trits; a scale of zero is rejected where the specification requires a non-zero scale |
| Dependencies | WP-04, WP-11, WP-13, and the closure of the alignment constraint of Section 2.2, Invariant 7 |
| Risk | High; the bit-level mutation of a shared byte interacts with the copy-on-write semantics of Section 5 of the specification |

---

## 4. Defect Backlog

The defects observed during the assessment are mapped to the work package that
resolves them. The severity follows the impact on the contract of the specification.

| Identifier | Defect | Severity | Work package |
|---|---|---|---|
| D-01 | The internal stride helper occupied the identifier required by the public contiguous operation | Resolved | Closed; the helper now carries a distinct identifier |
| D-02 | Element access terminates on the packed variants without a diagnostic that names the violated precondition | Blocking | WP-11, WP-14 |
| D-03 | The index resolution is expressed in element units and is not separated from the physical addressing rule, leaving the packing of Section 3.1 unrealized | Blocking | WP-04, WP-14 |
| D-04 | A zero-filled ternary buffer is materialized with the code that encodes the negative trit, contradicting Section 3.1 of the specification | High | WP-14 |
| D-05 | The packed variants are allocated with a degenerate zero scale, contradicting Sections 3.1 and 3.3 | High | WP-14 |
| D-06 | The logical index iteration underflows on a rank-zero shape | High | WP-10 |
| D-07 | The constructor does not validate the element capacity against the shape product | Medium | WP-03 |
| D-08 | The public contiguous operation is absent; the internal helper is unreachable and reports as unused | Resolved | Closed; the public operation is delivered under the identifier `make_contiguous`; the identifier reconciliation with the specification remains open |
| D-09 | The relationship between the contiguity predicate, the stride vector, and the physical offset was not stated by the specification | Resolved | Closed; Section 2.2, Invariant 4 of the specification now states the two clauses; the remaining coupling with the storage variant is tracked by D-15 |
| D-10 | The type tag is neither copyable nor comparable | Low | WP-01 |
| D-11 | The module imports the panic module and the vector module in forms that are not required | Low | WP-01 |
| D-12 | The diagnostic of the axis swap contains a typographic defect and does not name the violated precondition | Low | WP-05, WP-11 |
| D-13 | The tensor exposes no bulk access surface for the exporter and the graph renderer | Low | WP-12 |
| D-14 | The axis swap and its in-place variant have no test coverage | Low | WP-05, WP-13 |
| D-15 | The contiguity check terminates the operation on the packed variants without a diagnostic, contrary to the error policy of Section 4.1 of the specification | High | WP-11, WP-14 |
| D-16 | The contiguity check reports a tensor whose logical extent is empty as not contiguous | Open | The specification no longer carries a non-empty extent clause; the implementation retains the clause until WP-05 closes |

---

## 5. Test Matrix

| Area | Cases |
|---|---|
| Construction | Valid storage, oversized storage, undersized storage, rank-zero shape, shape with an empty extent, shape with an empty extent in a non-leading position |
| Geometry | Shape, stride, offset, and element count for rank one, rank two, and rank three; agreement between the length of the shape and the length of the stride vector |
| Contiguity | Canonical construction, construction from a storage with a non-zero offset, view produced by an axis swap, view produced by a second axis swap, tensor whose logical extent is empty, reported as contiguous under the two clauses of Section 2.2, Invariant 4 |
| Element access | Read and write on a contiguous tensor, read and write on a non-contiguous tensor, out-of-range index, index tuple of the wrong length, access to a packed variant under the current phase |
| Copy-on-write | Mutation of a tensor that shares its buffer with a clone; mutation of a tensor that shares its buffer with a view; the shared tensor remains unchanged in both cases |
| Views | Axis swap over a rank-two tensor, axis swap over a rank-three tensor, axis swap with a repeated identifier, axis swap with an identifier outside the rank, axis swap over a rank-zero tensor, in-place axis swap leaving the storage buffer unchanged |
| Contiguous operation | Contiguous source, non-contiguous source, non-contiguous source with a non-zero physical offset, comparison of the materialized elements against the source |
| Element-wise kernels | Each kernel on a contiguous operand, each kernel on a non-contiguous operand, shape mismatch, zero divisor, stateful callable |
| Reduction | Reduction of a contiguous operand, reduction of a non-contiguous operand, reduction of a rank-zero operand |
| Products | Each product against a hand-computed reference, each product on non-contiguous operands, inner-dimension mismatch, rank precondition violation |
| Iteration | Enumeration of all logical positions for rank one, rank two, and rank three, enumeration of a rank-zero shape |
| Bulk access | Data round trip of a contiguous tensor, of a non-contiguous tensor, of a rank-zero tensor, and of a tensor whose logical extent is empty; `from_data` construction; `from_data` rejection of a mismatched wire tag and of an undersized byte length |

---

## 6. Constraints Carried from the Current Implementation

| Constraint | Consequence for this phase |
|---|---|
| The linear layer, the optimizer, the exporter, and the graph renderer consume nested vectors rather than tensors | No regression is observable from those consumers; the phase may redesign the tensor surface without a compatibility obligation |
| The gradient structure clones the module tree to carry gradients | The tensor surface must support in-place parameter mutation under copy-on-write so that the later migration does not require a second redesign |
| The forward pass of the linear layer produces a row-major result and the backward pass accumulates over a batch | The products of WP-09 must accept a rank-one right operand and a rank-two left operand in the orientation used by the exporter, so that the migration does not require a transposition at every call site |
| The export path writes the weights of a linear layer row by row | The bulk access of WP-12 must expose the elements in the order expected by the export format |

---

## 7. Items Outside This Phase

| Item | Reason |
|---|---|
| The `I8` and `T3` access paths | The `T3` access path belongs to the v0.1.0 arc and is recorded as WP-14; the `I8` access path is deferred beyond v0.1.0 (maintainer decision, October 2026) |
| The ternary fast path of the product | Belongs to the subsequent phase; depends on WP-14 |
| Reductions along arbitrary axes, the mean reduction | Deferred by Section 4.2 of the specification; each requires its own contract clause |
| `slice`, `permute`, `reshape`, and multidimensional broadcasting | Deferred by Section 4.2 of the specification; each requires its own contract clause |
| The migration of the module trait, the linear layer, the gradient structure, the exporter, and the graph renderer | The objective of the phase that follows this one; this phase is the prerequisite |
| The correction of the single-precision finite-difference estimator | Belongs to the gradient verification phase, recorded as a known limitation in the project status record |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| The stride vector is reinterpreted between a logical element index and a physical address | The addressing model of Section 2.3 of the specification is the single normative statement; WP-04 isolates the resolution from the variant |
| The copy-on-write duplication of a shared buffer is defeated by a kernel that writes through a stale reference | Every mutation reaches the buffer through the copy-on-write path; WP-13 asserts the isolation of a shared buffer |
| The cost of a strided traversal grows with the rank and defeats the objective of the contiguous operation | The kernels and the products are specified to traverse logical coordinates; the profiling of the traversal is out of scope of this phase and is recorded in the research log when the benchmark harness exists |
| The public surface grows before the consumers exist | The scope of this phase is the minimum viable surface of Section 4 of the specification; no operation outside that surface is admitted |

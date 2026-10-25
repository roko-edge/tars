# Future Execution Targets and Invariance Rules

**Document type:** Target architecture record. Nothing in this document is implemented.
It records the execution targets of later phases and the rules the current phase must
hold so that those phases do not force rework.

TARS is a TinyML co-design stack. The inference surface is quantization-first and
embedded-bound — ternary weights, static shapes, microcontroller-class CPUs alongside
the ternary NPU — and training runs on commodity GPUs through cubecl. The typing
stance is compile-time-first: element types, operation availability, and
mixed-precision signatures are static ([`specs/TENSOR.md`](specs/TENSOR.md),
[`specs/MODEL_TRAIT.md`](specs/MODEL_TRAIT.md)); runtime values appear only where an
execution boundary requires them — the wire tag at serialization, the device at the
GPU phase.

---

## 1. Target Matrix

| Target | Phase | Role |
|---|---|---|
| NPU inference | v0.1.0 (release) | Bit-exact execution of the ternary model |
| CPU inference, embedded and microcontroller | v0.2.0+ | Host-side and MCU execution of the quantized model through the bit-exact simulator (a pure-Rust, `no_std` with `alloc` runtime); the first prospective consumer of the reserved `I8` variant |
| GPU training (Vulkan through cubecl, custom ternary kernels) | v0.3.0 (proposed) | Training acceleration and research iteration on commodity GPUs |
| CPU golden reference | Permanent | The bit-exact simulator never runs on GPU; determinism is the product |

---

## 2. GPU Path

Training on GPU will use cubecl (tracel-ai). Two integration paths exist: consuming
cubecl directly, or training through burn's module layer over cubecl. The choice is
deferred to the GPU-phase kickoff and recorded there; the invariance rules of
Section 3 are held so that either path is a bounded addition rather than a redesign.

---

## 3. Invariance Rules

These rules bind the current phase.

1. **Operation bodies live behind a kernel seam.** `Tensor` methods are thin; the
   arithmetic is implemented in a kernel surface (f32-CPU today; cubecl kernels in
   the GPU phase). The GPU port replaces kernel implementations, not APIs.
2. **Modules touch tensors only through the operation set.** Model code performs no
   storage-buffer access; bulk and raw access are confined to the exporter, the
   renderer, and kernel implementations.
3. **The tensor core is allocation-only.** No filesystem, threading, or environment
   access inside the tensor engine, so an embedded inference crate is a crate split
   under `no_std` with `alloc`, not a rewrite.
4. **Wire formats are device-independent.** The packed-trit format, the layer
   descriptor, and the manifest are identical across CPU, GPU, and NPU.
5. **Quantization and the golden path stay CPU-deterministic.** Quantizer
   mathematics is pure `f32` over latent weights; the bit-exact reference never runs
   on GPU; GPU training reproducibility is driver-dependent and will be scoped in
   the GPU-phase contract. Regression fixtures remain CPU.

---

## 4. Device Axis

The device dimension is deliberately not abstracted in the current phase: the device
is static per execution phase in this project, and designing a device trait before
the cubecl API is in hand risks the wrong abstraction. With the rules of Section 3
held, a later generalization is mechanical.

The consequences worth recording now:

- Copy-on-write through `Arc::make_mut` is a CPU-storage concern. GPU storage has
  its own synchronization model and must not inherit the CPU mechanism.
- Element types remain compile-time. cubecl kernels are typed per element by
  construction, so the typed-tensor design is the native shape of the GPU path.
- The device enters as a value, not as a generic parameter: the GPU storage of an
  element type replaces the CPU buffer inside the tensor, and public signatures do
  not change. This is the erased-client shape of the current cubecl, validated
  against the cubecl and burn sources in October 2026.
- The tensor carries no parallel runtime dtype identity: the wire tag derives from
  the static element type (`Element::DTYPE`) at the points where the GPU stack
  requires a value. This is a deliberate deviation from runtime-tagged tensor
  containers, held by the compile-time-first stance.
- The embedded path never runs cubecl: every cubecl execution runtime is
  standard-bound and no microcontroller target exists upstream. Microcontroller
  inference is the bit-exact simulator — a pure-Rust, `no_std` with `alloc`
  evaluation of the device-independent wire formats — isolated from the GPU phase
  by the crate split of Invariance Rule 3.
- The GPU phase is budgeted against one pinned cubecl release (the upstream
  declares alpha status with breaking changes between minor versions). The kernel
  library of TARS depends on cubecl with no runtime feature enabled; the executable
  — or a thin alias crate — enables `vulkan` (the wgpu/SPIR-V pin) and, where
  required, `cpu`; the crate that owns the GPU engines re-exports cubecl so that
  custom kernels compile against the same version.

---

## 5. Phase Placement

| Phase | Scope |
|---|---|
| v0.1.0 | Unchanged: MNIST ternary release, bit-exact parity, tag gate |
| v0.2.0 | Unchanged: zero-skipping; `I8` may enter with the embedded-inference consumer |
| v0.3.0 (proposed) | GPU training via cubecl with custom ternary kernels; device dimension admitted as a compile-time, feature-gated axis |
| v1.0+ | Unchanged: architectural expansion (CNNs, CfC, Neural ODEs, DAG, autodiff); per-network NPU assembly per [`NPU_SPECIALIZATION_PROPOSAL.md`](NPU_SPECIALIZATION_PROPOSAL.md) |

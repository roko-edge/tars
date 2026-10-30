# Internal Engineering Records

This area contains architecture descriptions, implemented decisions, technical
contracts, and proposals for TARS maintainers. It is not user documentation and does
not establish that a planned capability is available.

---

## Navigation by Task

| Objective | Document | State |
|---|---|---|
| Understand the current repository structure | [`ARCHITECTURE.md`](ARCHITECTURE.md) | Current implementation record |
| Review implemented architectural decisions | [`DECISIONS.md`](DECISIONS.md) | Implemented decisions only |
| Inspect the Rust-to-RTL artifact format | [`MEM_FORMAT.md`](MEM_FORMAT.md) | Current format with shared `npu/data/` paths |
| Review board integration and synthesis architecture | [`BOARD_INTEGRATION.md`](BOARD_INTEGRATION.md) | Board integration architecture and local lab specification |
| Evaluate the observability design | [`OBSERVABILITY_PROPOSAL.md`](OBSERVABILITY_PROPOSAL.md) | Proposal, not implemented |
| Review future execution targets and invariance rules | [`FUTURE_TARGETS.md`](FUTURE_TARGETS.md) | Target architecture record for the GPU and embedded phases |
| Review the NPU specialization and generator model | [`NPU_SPECIALIZATION_PROPOSAL.md`](NPU_SPECIALIZATION_PROPOSAL.md) | Proposal, not implemented; v0.1.0 obligations bind the engine steps |
| Review the assembly layers, engine contract, and test tiers | [`ASSEMBLY_AND_TEST_MODEL.md`](ASSEMBLY_AND_TEST_MODEL.md) | Proposal; records pending decision briefs for the v0.1.0 steps |
| Review the `start`/`done` protocol and wrapper contract | [`PROTOCOL.md`](PROTOCOL.md) | Engineering contract (normative); owns the engine and wrapper handshake |
| Review normative and target contracts | [`specs/README.md`](specs/README.md) | Mixed current and target contracts with explicit status |

---

## Navigation by Engineering Domain

| Domain | Primary records |
|---|---|
| Rust model architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md), [`specs/MODEL_TRAIT.md`](specs/MODEL_TRAIT.md) |
| Tensor engine | [`specs/TENSOR.md`](specs/TENSOR.md) |
| NPU arithmetic and integration | [`MEM_FORMAT.md`](MEM_FORMAT.md), [`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md), [`BOARD_INTEGRATION.md`](BOARD_INTEGRATION.md), [`NPU_SPECIALIZATION_PROPOSAL.md`](NPU_SPECIALIZATION_PROPOSAL.md) |
| NPU handshake and wrapper protocol | [`PROTOCOL.md`](PROTOCOL.md), [`ASSEMBLY_AND_TEST_MODEL.md`](ASSEMBLY_AND_TEST_MODEL.md) |
| Visualization and telemetry | [`OBSERVABILITY_PROPOSAL.md`](OBSERVABILITY_PROPOSAL.md) |

Current project progress and scheduling belong in
[`../project/`](../project/README.md).

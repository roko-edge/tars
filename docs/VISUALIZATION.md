# Network Visualization

**Status** [PENDING]

No `visualization` binary is currently present in `src/bin/`. The graph helpers
under `src/view/` are retained in the source tree and `src/lib.rs` currently
leaves them outside the public module exports.
[Public modules](../src/lib.rs)

## What is planned for the release?

The v0.1.0 contract calls for a network diagram with a subsampled layer
layout, activation intensity, and signed weight styling. The delivery plan
tracks the implementation against a pinned model and forward trace.
[Visualization requirements](project/V0_CONTRACT.md#3-visualization-and-observability-specifications) ·
[Delivery plan](project/V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition)

The engineering proposal records the graph representation and observer
design. The Rust development shell includes Graphviz for that work.
[Observability proposal](engineering/OBSERVABILITY_PROPOSAL.md) ·
[Development shells](GETTING_STARTED.md#1-environment)

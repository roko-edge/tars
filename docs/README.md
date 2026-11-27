# TARS Documentation

Use this index to find current Rust and NPU behavior, engineering contracts, and
project plans. The scope of each destination identifies which question it answers.

---

## How does the code work?

| Question | Start here |
|---|---|
| How do I build and run TARS? | [Getting started](GETTING_STARTED.md) |
| How do training, batching, and QAT work? | [Implemented Rust API](API.md#3-training-operations) |
| What do MNIST, OR, and XOR run? | [Experiment configurations](EXPERIMENTS.md#built-in-examples) |
| How do I simulate the NPU? | [RTL commands and checks](../npu/README.md#commands) |
| What crosses the Rust–RTL boundary? | [Memory artifacts](engineering/MEM_FORMAT.md) · [Q8.24 arithmetic](engineering/specs/ARITHMETIC_Q8_24.md) |
| How does `start`/`done` work? | [Handshake contract](engineering/PROTOCOL.md) |
| Where is the rest of the engineering reference? | [Architecture and specifications](engineering/README.md) |
| Can I render a model diagram? | [Visualization status](VISUALIZATION.md) |

## What is the project working toward?

| Question | Start here |
|---|---|
| What must v0.1.0 verify? | [Release acceptance](project/V0_CONTRACT.md) · [Evidence and conflicts](project/V0_DELIVERY_PLAN.md) |
| Who owns the work and what is pending? | [Project records](project/README.md) |
| How are benchmark runs recorded? | [Proposed run protocol](EXPERIMENTS.md#how-should-benchmark-runs-be-recorded-proposed) |

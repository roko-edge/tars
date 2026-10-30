# TARS User Documentation

This directory documents resources that are implemented and available to users of
the TARS Rust package and SystemVerilog prototype.

Project plans and internal engineering contracts are maintained separately. Their
presence in this repository does not imply that a feature is available.

---

## Where to Start

Pick the path that matches your role; each is an ordered reading list, not a
requirement to read everything.

| If you are… | Read in this order |
|---|---|
| New to the project | [`GETTING_STARTED.md`](GETTING_STARTED.md) (environment and commands) → [`../npu/README.md`](../npu/README.md) (what the NPU is) → [`project/STATUS.md`](project/STATUS.md) (where things stand) |
| Going to touch the RTL | [`../npu/README.md`](../npu/README.md) → [`engineering/ARCHITECTURE.md`](engineering/ARCHITECTURE.md) (glossary and handshake contract) → [`engineering/specs/ARITHMETIC_Q8_24.md`](engineering/specs/ARITHMETIC_Q8_24.md) → [`project/research/LOG-007-v0_execution_mode_phasing.md`](project/research/LOG-007-v0_execution_mode_phasing.md) (decisions and steps) |
| Going to touch the Rust side or the file formats | [`engineering/MEM_FORMAT.md`](engineering/MEM_FORMAT.md) → [`project/V0_CONTRACT.md`](project/V0_CONTRACT.md) (release criteria) → [`project/ROADMAP.md`](project/ROADMAP.md) (ownership and gates) |

---

## Navigation by Task

| Objective | Document | Scope |
|---|---|---|
| Prepare the environment and run TARS | [`GETTING_STARTED.md`](GETTING_STARTED.md) | Requirements, commands, current behavior, and operational limitations |
| Run the OR or XOR configurations | [`EXPERIMENTS.md`](EXPERIMENTS.md) | Experiment factories, training flow, outputs, and manual selection |
| Inspect the current public surface | [`API.md`](API.md) | Implemented modules, operations, and stability boundaries |
| Generate network diagrams | [`VISUALIZATION.md`](VISUALIZATION.md) | Graph extraction, DOT export, Graphviz rendering, and manual graph construction |
| Simulate the NPU prototype | [`../npu/README.md`](../npu/README.md) | Hardware environment, Make targets, and current simulation behavior |

---

## Other Repository Records

| Area | Entry Point | Purpose |
|---|---|---|
| Internal engineering | [`engineering/README.md`](engineering/README.md) | Architecture, implemented decisions, technical contracts, and proposals |
| Project operations | [`project/README.md`](project/README.md) | Status, roadmap, release targets, and research process |

Internal engineering and project records are not user documentation. A capability is
considered available only when it is described in this user section and agrees with
the current source tree.

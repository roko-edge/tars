# Proposal: Observability and Frontend Interface

**Status:** Proposed. The telemetry interfaces, persisted run layout, frontend, and
streaming server described here are not current library features.

This document proposes the telemetry architecture, interchange formats, and
implementation phases for training observability and interactive visualization in
TARS.

## Scope and boundaries

The core mathematical engine (`tars`) must not depend on frontend libraries,
HTTP/WebSocket servers, or rendering runtimes. Observability is decoupled through an
event-driven telemetry interface:

```text
Training Loop (src/train.rs)
    │
    ▼
Observer Trait (src/observe/mod.rs)
    ├── TerminalReporter (stdout logging)
    ├── FileLogger (run artifacts: JSON / JSONL)
    └── StreamingBroadcaster (bounded channel -> WebSocket server)
```

1. Training execution does not block on visualization.
2. The core library exports only abstract telemetry events and static serialization.
3. Telemetry communication uses typed, versioned JSON records.
4. Graphviz DOT generation is preserved for static documentation and checkpoints.

## Implementation phases

Development proceeds in bounded milestones to isolate mathematical core changes from
user-interface development.

| Phase | Target | Scope | Dependencies |
|---|---|---|---|
| **Phase 1** | Model graph extraction | Inspect `Sequential` and convert layers/weights to `NetGraph` | `petgraph` |
| **Phase 2** | Telemetry and disk artifacts | Define `Observer` trait and export `run.json`, `metrics.jsonl`, `topology.json` | `serde`, `serde_json` |
| **Phase 3** | Offline frontend prototype | Web client parsing static JSON artifacts to display loss curves and topology | Svelte/React, Cytoscape, ECharts |
| **Phase 4** | Local streaming server | Local Axum/Tokio server forwarding events via WebSocket to browser | `axum`, `tokio` |
| **Phase 5** | Dynamic topology and hardware | Stable node identities (`NodeId`), topological mutation events, and NPU metrics | Core dynamic graph engine |

---

## Phase 1: Model Graph Extraction

The visualization pipeline in `src/view/graph.rs` extracts topology from a
`Sequential` model through `network_to_graph`. Phase 1 of this proposal generalizes
that extraction into the telemetry topology export described below.

### Interface additions

```rust
impl NetGraph {
    /// Constructs a directed graph directly from an existing sequential model.
    pub fn from_sequential(model: &Sequential) -> Self;
}
```

### Graph mapping rules

1. For each `Linear(in_features, out_features)`:
   - Input dimension defines the source layer vertices.
   - Output dimension defines the destination layer vertices.
   - Directed edges store parameter weights from `Linear::weights` (`f32`).
   - Vertices store parameter biases from `Linear::bias` (`f32`).
2. For each `Activation`:
   - Annotates target neuron nodes with the activation function variant (`ReLU`, `Sigmoid`).
3. Output produces valid DOT representations and renders via Graphviz to PNG/SVG.

---

## Phase 2: Telemetry and Disk Artifacts

Decouples progress reporting from `println!` invocations in training loops.

### Rust abstractions

```rust
pub enum TrainEvent {
    RunStarted {
        run_id: String,
        timestamp: u64,
        model_topology: TopologySnapshot,
    },
    StepFinished {
        epoch: usize,
        step: usize,
        loss: f32,
        lr: f32,
    },
    EpochFinished {
        epoch: usize,
        loss: f32,
    },
    TopologyChanged {
        revision: u64,
        snapshot: TopologySnapshot,
    },
    RunFinished {
        final_loss: f32,
        total_steps: usize,
    },
}

pub trait Observer: Send + Sync {
    fn on_event(&mut self, event: &TrainEvent);
}
```

### Storage layout

Runs persist to disk under `.tars/runs/<run_id>/`:

```text
.tars/runs/019ad3f0-4a8b-7000/
├── run.json           Model hyperparameters, input/output dims, metadata
├── metrics.jsonl      Append-only loss, step, epoch, learning rate records
└── topology.json      Initial topology snapshot
```

---

## Phase 3: Offline Frontend Prototype

The frontend client operates independently of the Rust runtime. It consumes the
JSON and JSONL artifacts produced in Phase 2.

### Technology stack

- **Runtime:** Node.js / Vite development server.
- **Framework:** Svelte or React (managed in separate `ui/` directory).
- **Topology layout:** Cytoscape.js (DAG / breadthfirst layout).
- **Metrics plotting:** Apache ECharts.

### Functional requirements

1. **Topology panel:** Renders neurons arranged by layer columns. Edge stroke width
   scales with `abs(weight)`. Edge color indicates sign (positive / negative).
2. **Loss panel:** Real-time line plot of `step` versus `loss`.
3. **Inspector drawer:** Clicking a node or edge displays its identifier, layer index,
   bias, weight, and activation function.

---

## Phase 4: Local Streaming Server

Bridges Rust training execution to the web dashboard in real time over local WebSockets.

```text
Training Thread ──(try_send)──► Bounded Channel ──► Axum WebSocket Server ──► Browser
```

### Protocol

- Server binds to `127.0.0.1:7347`.
- Endpoint `GET /ws` establishes a WebSocket connection.
- Messages are dispatched as JSON serialized `TrainEvent` records.
- If the client lags or disconnects, the channel drops pending frames; training
  execution is never delayed.

---

## Data Interchange Contracts

All telemetry serialization follows fixed schemas.

### 1. Topology snapshot (`topology.json`)

```json
{
  "revision": 0,
  "nodes": [
    {
      "id": 0,
      "layer": 0,
      "kind": "input",
      "label": "x0",
      "bias": null,
      "activation": null
    },
    {
      "id": 2,
      "layer": 1,
      "kind": "hidden",
      "label": "h1_0",
      "bias": 0.1245,
      "activation": "sigmoid"
    },
    {
      "id": 4,
      "layer": 2,
      "kind": "output",
      "label": "y0",
      "bias": -0.0412,
      "activation": "sigmoid"
    }
  ],
  "edges": [
    {
      "id": 0,
      "source": 0,
      "target": 2,
      "weight": 1.4521
    },
    {
      "id": 1,
      "source": 2,
      "target": 4,
      "weight": -0.8912
    }
  ]
}
```

### 2. Metrics stream record (`metrics.jsonl`)

Each line contains a standalone JSON object:

```json
{"epoch": 240, "step": 4800, "loss": 0.00241, "lr": 0.01, "timestamp_ms": 1727721600120}
```

### 3. Run metadata (`run.json`)

```json
{
  "run_id": "019ad3f0-4a8b-7000",
  "model_name": "xor",
  "optimizer": "BGD",
  "initial_lr": 0.01,
  "epochs": 10000,
  "created_at": 1727721600000
}
```

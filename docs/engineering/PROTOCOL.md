# Start/Done Protocol — Engine & Wrapper Contract

**Document type:** Engineering contract (normative).
**Owns:** the `start`/`done` handshake between `layer_engine`, the `npu` wrapper,
and any consumer.
**Does not cover:** memory formats ([`MEM_FORMAT.md`](MEM_FORMAT.md)), arithmetic
([`specs/ARITHMETIC_Q8_24.md`](specs/ARITHMETIC_Q8_24.md)), weight packing
formats (planned).

---

## Summary

**`start` and `done` are events, not states.** A rising `start` requests exactly one
inference; `done` is a one-cycle pulse aligned with the last `result[]` write; and
`result[]` stays valid until the next accepted request. The `npu` wrapper orchestrates
multi-layer inference and protects layer-to-layer data in flight — it does **not**
do button conditioning: consumers drive clean pulses.

---

## What is the contract? [DECIDIDO]

| # | Clause |
|---|---|
| 1 | `start` is a **request event**. Consumers MUST drive one-cycle pulses. The engine is level-sensitive while idle: a held `start` yields repeated back-to-back inferences — documented behavior, not protected. |
| 2 | `done` is a **completion event**: a one-cycle pulse. On the `done` cycle, `result[]` is complete and valid. |
| 3 | `result[]` holds from the `done` pulse until the next accepted request. Nothing clears it in between. |
| 4 | `done` is driven ONLY by the last `layer_engine` (single driver). Never assign `done` in the wrapper — pulses self-clear. |
| 5 | The wrapper IGNORES requests while an inference is in flight (`state == RUN`): no queue, no cancel. |
| 6 | A request in the same cycle as `done` is lost (the FSM is still `RUN` that cycle). Resubmit on the next cycle. |
| 7 | **Invariant:** the condition that launches the motor MUST equal the condition of the FSM transition. In the current wrapper both are `start && state == IDLE`. |

## Timing (2-layer wrapper)

```text
clk        : _/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\_/¯\
start      : __/¯¯¯\____________________________________________   (1-cycle pulse)
u_l0 runs  : ____/¯¯¯¯¯¯¯¯¯¯\__________________________________
l0_done    : ________/¯\_______________________________________   (pulse → L1 starts)
u_l1 runs  : __________/¯¯¯¯¯¯¯¯\______________________________
done       : ________________/¯\_______________________________   (1 cycle; result[] valid)
result[]   : .................└─ v0..vn (held until next request)
```

## Who does what? (responsibility boundaries)

| Layer | Does | Does NOT |
|---|---|---|
| `layer_engine` | compute; `start`/`done` events; hold `result[]` | know the network or the board |
| `npu` wrapper | layer chaining (pure wires); request arbitration while data is in flight | button conditioning, math, display |
| `synthesis/` | debounce → one-shot pulse; latch `done` (pulse → level) for LED/status | re-implement the handshake |
| testbenches | drive one-cycle pulses; count `done` pulses | hold `start` high (contract violation) |

## What is the wrapper? (scope)

`npu.sv` is the **assembled NPU** for one specific network — `784→64→10` today.
Topology is *parameters + data*, not code: a new network means new sizing and new
`.mem` files, zero redesign. Layer chaining is a pure wire (`L1.start = L0.done`);
the two-state FSM (`IDLE`/`RUN`) is protocol arbitration (one bit of state), not
sequencing. It merges into the future `sequencer` (LOG-007 Step 4) as
`state != IDLE` — no rework.

## Why are the clauses shaped this way? (lessons)

| Lesson | Clause |
|---|---|
| `done` as a level caused re-trigger loops; a pulse removed the class | 2 |
| FSM condition ≠ motor condition → deadlock (FSM enters `RUN`, motor never fires, `done` never arrives) | 7 |
| A 2nd request between L0 finish and L1 finish corrupted `l0_result` mid-read | 5 |
| `done <= 0` in the FSM created a second driver (MULTIDRIVEN); pulses self-clear | 4 |

## Test gates

| Gate | Expectation | Proves |
|---|---|---|
| Smoke, 2 layers (`IN=2, H=2, OUT=1`) | result `= 10.0` (hand-computed) | chain + math |
| Two clean `start` pulses | two `done` pulses, independent results | clauses 1–2 |
| Pulse while L1 in flight | ignored; result = first inference | clause 5 |
| Pulse coincident with `done` | lost (documented) | clause 6 |
| `verilator --lint-only -Wall` | clean (`MULTIDRIVEN`/`UNDRIVEN` = 0) | clause 4 |

## Status

- [DECIDIDO] — the seven clauses above. Aligned with the RTL as of the
  October 2026 wrapper commit: the arbitration FSM (`start && state == IDLE`
  launch, `state == RUN && done` return) and the single-driver `done` are
  implemented in `npu/src/npu.sv`; `make lint` (top `npu`) exits zero.
- [DECIDIDO] — `layer_engine.sv` drives `done` as a one-cycle registered pulse
  (`done <= (state == WRITE) && (j == OUT-1)`); no `done <= 0` assignment
  remains in the engine or the wrapper.
- [PROPOSTO] cosmetics: wrapper FSM type (own `wrap_state_t` vs reuse
  `npu_pkg::state_t`); declarations before instances; drop redundant `npu_pkg::`
  prefixes (the header wildcard import covers the whole module).

# Research Logs and Academic Publication Notes

**Document type:** Project research process and publication planning.

This directory holds empirical research logs, mathematical derivations,
hardware/software co-design trade-offs, and experimental benchmarks for TARS.

The contents of this directory serve as the primary source material for compiling academic publications on TARS (e.g., hardware-efficient NPU acceleration, ternary quantization loss bounds, and bit-exact co-simulation).

---

## Log Index

| ID | Planned title | Domain | State |
|---|---|---|---|
| `LOG-000` | Research Logging Protocol and Paper Outline | Methodology | Planned |
| `LOG-001` | Zero-Copy Strided Tensor Engine | Software tensor runtime | Planned |
| `LOG-002` | Q8.24 vs. Ternary Loss Bounds | Quantization | Planned |
| `LOG-003` | Layer Engine FSM and Accumulator Width | Hardware | Planned |
| `LOG-004` | Accuracy vs. Quantization Bit-Width | Benchmark | Planned |
| `LOG-005` | Extension to CNNs and CfC Networks | Future work | Planned |
| `LOG-006` | Ternary Zero-Skipping Weight Encoding | Hardware and quantization | Accepted design; scheduled for v0.2.0 |
| `LOG-007` | v0 Execution-Mode Phasing and Implementation Sequence | Hardware and release plan | Accepted maintainer position |
| `LOG-008` | NPU Resource and Energy Measurement Protocol | Hardware benchmark methodology | Planned |

Supporting analyses:

| Document | Scope | State |
|---|---|---|
| [`TENSOR_THEORY.md`](TENSOR_THEORY.md) | Dense strided tensor theory: storage, shape, strides, offset, views, contiguity, ownership, and packed elements | Theoretical reference; not an implementation or benchmark report |
| [`Glorot_Bengio_2010_Summary.md`](Glorot_Bengio_2010_Summary.md) | Summary of Glorot and Bengio (2010) and its relation to the TARS initialization functions | Complete |
| [`Xavier_Analysis.md`](Xavier_Analysis.md) | Notes on weight initialization | One-line draft stub |

---

## Paper Structural Mapping

Planned logs map to the proposed academic paper sections:

```text
Paper Section                      Corresponding Research Logs
─────────────────────────          ───────────────────────────────────────────
1. Introduction & Background       LOG-000 (Motivation, Co-Design Philosophy)
2. Software Tensor Runtime         LOG-001 (Zero-Copy Strided Tensor Engine)
3. Quantization & Ternary Models   LOG-002 (Q8.24 vs. Ternary Loss Bounds)
4. SystemVerilog NPU Architecture  LOG-003 (Layer Engine FSM & Accumulator Width)
5. Experimental Results (MNIST)    LOG-004 (Accuracy vs. Quantization Bit-Width)
6. Conclusion & Future Work        LOG-005 (Extension to CNNs and CfC Networks)
```

---

## Logging Guidelines

1. **Self-Contained Empirical Proofs:** Every log must contain exact reproducible commands, seeds, and quantitative tables.
2. **Mathematical Rigor:** Equations must be typeset in LaTeX math syntax ($y = f(Wx + b)$).
3. **Hardware Waveforms & RTL Metrics:** RTL changes must include cycle counts, LUT/register utilization estimates, or GTKWave waveform analysis.

# Memory Files

**Audience** Engineers working on the Rust-to-RTL boundary.

**Document type** Current artifact-format reference.

The exporter writes dense 2-bit ternary weight codes and 32-bit Q8.24 words
as hexadecimal text under `npu/data/`. The assembled Tier 2 bench reads these
files through `$readmemh`. The training binary leaves export to explicit
calls. [Exporter](../../src/export.rs) · [Tier 2 bench](../../npu/tests/tb_parity.sv)

## Which files does the current exporter write?

| File | Writer | Encoding and order | Consumer |
| --- | --- | --- | --- |
| `wt.mem` | `export_model` | One 2-bit code per line, printed as two hex digits. Affine layers in model order, then output row, then input feature. | Tier 2 weight arrays |
| `w0.mem` | `export_model` | One signed Q8.24 word per supplied scale, printed as eight hex digits. Supply one scale per affine layer in model order. | One scale for each Tier 2 layer |
| `bias.mem` | `export_model` | One signed Q8.24 word per output neuron, in affine-layer and output order. | Tier 2 bias arrays |
| `activations.mem` | `export_inference` | Signed Q8.24 inputs in sample order and then feature order. | Tier 2 inputs and legacy smoke inputs |
| `target.mem` | `export_inference` | Signed Q8.24 words derived from the model's float predictions, in sample order and then output order. | Tier 2 development expectations |

The canonical encoder emits `00` for zero, `01` for +1, and `10` for −1.
The decoder also accepts `11` as zero. `to_q8_24` scales by $2^{24}$, truncates
toward zero and saturates through the signed integer cast before preserving
the word bits. The RTL epilogue instead uses an arithmetic right shift when
rescaling the wide accumulator product. [Quantization encoding](specs/QUANT.md#2-canonical-hardware-code-serialization) ·
[Q8.24 arithmetic](specs/ARITHMETIC_Q8_24.md)

## Which artifacts are legacy?

The legacy `tb.sv` smoke test reads `weights.mem`, `bias.mem` and
`activations.mem` for a single affine layer with two inputs and one output.
`weights.mem` contains older dense Q8.24 words. The smoke bench's 2-bit
weight array keeps their low bits and the bench displays results without
assertions. Treat its input files as a matched set. The two benches share
the `bias.mem` and `activations.mem` names. [Legacy smoke bench](../../npu/tests/tb.sv)

## What remains pending for release parity?

- **[PENDING] Exact expectations.** The Tier 2 comparator currently allows
  a one-LSB difference. Release expectations must come from the independent
  Rust integer reference using the same encoded inputs, weights, scales and
  biases. The release comparison requires equality of every output word.
- **[PENDING] Artifact validation.** The exporter accepts scales by position
  and uses fixed output paths. Validate the supplied model, file lengths and
  fixture identity against the assembled wrapper before parity testing.
- **[DECIDED] Wrapper scope.** The v0.1.0 wrapper assembles two engines with
  elaboration-time dimensions. The files above contain parameter and
  inference data. `program.mem` and the programmable sequencer belong to
  v0.1.1. [LOG-008](../project/research/LOG-008-v0_governance_and_release_decisions.md#2-maintainer-decisions-november-2026)

[Parity delivery plan](../project/V0_DELIVERY_PLAN.md#24-how-do-export-and-rtl-establish-exact-parity-pending) ·
[RTL test tiers](ASSEMBLY_AND_TEST_MODEL.md#5-three-tier-verification-model)

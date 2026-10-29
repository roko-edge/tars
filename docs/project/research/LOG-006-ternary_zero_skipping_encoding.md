# Research Log LOG-006: Ternary Zero-Skipping Weight Encoding

**Document type:** Engineering research log under the protocol defined in
[`V0_CONTRACT.md`](../V0_CONTRACT.md), Section 4.

**State:** Design analysis. The encoding is reserved as layer-descriptor
`mode = 2` and scheduled for v0.2.0. No exporter or RTL support exists; all
quantitative figures in Section 3 are projections of the token model.

This log specifies a sparse weight-stream format for ternary layers, analyzes its
feasibility across target architectures, and defines a five-bit token scheme that
removes run-length and row-termination ambiguities without auxiliary tables.

---

## 1. Title and Context

Ternary quantization maps weights to $\{-1, 0, +1\}$ with an empirically high zero
fraction (typically 75-90% for thresholded trained networks). Under dense storage
and single-PE traversal, every zero weight consumes one memory slot and one MAC
cycle while contributing nothing to the accumulator.

Hypothesis: a relative zero-skipping stream — the architectural basis of EIE
(Han et al., ISCA 2016) and Cambricon-X (Zhang et al., ISCA 2016) — is well suited
to the TARS sequential layer engine, provided two ambiguities are resolved
in-band: zero runs longer than the skip field, and row termination.

---

## 2. Theoretical Formulation

### 2.1 Position Model

The stream stores only active (non-zero) weights. The position of the $k$-th
active weight of a row is recovered incrementally, where $\mathrm{skip}_k$ is the
number of zeros preceding it:

$$i_k = i_{k-1} + \mathrm{skip}_k + 1$$

The encoding is relative and unstructured: an absolute position exists only after
summing all preceding skips. This property is what makes the format cheap on
sequential engines and expensive on lockstep and gather-based architectures
(Section 2.2).

### 2.2 Platform Viability

| Target architecture | Viability | Mechanism |
|---|---|---|
| Single-PE sequential NPU (TARS v0/v1, Spartan-7) | High | One token consumed per cycle; position recovery is one addition; the activation is fetched by direct indexed read. Projected gain approaches 10x in cycles and memory at high sparsity. |
| Multi-PE, neuron-parallel (EIE-style) | High | Each PE owns one output neuron and its token stream, provided concurrent activation reads do not serialize (Section 4). |
| Classic systolic array (TPU / Gemmini class) | Low | Lockstep distribution of a shared activation column breaks when PEs skip to divergent positions; zeros would have to flow as bubbles. |
| GPU (CUDA / tensor cores) | Low | Divergent skips cause warp divergence; scattered activation reads are uncoalesced. Ampere-and-later GPUs expose dedicated 2:4 structured-sparsity hardware, which mandates aligned block or bitmask formats instead. |
| CPU (AVX2 / NEON) | Marginal | Scalar execution is adequate; vector registers would require slow gather instructions. |

### 2.3 Failure Modes of a Naive $\{\mathrm{skip}[3{:}0], \mathrm{sign}[0]\}$

1. **Run-length overflow:** a four-bit skip encodes at most 15 consecutive zeros,
   while rows at 90% sparsity commonly contain runs of 30-50 zeros. Padding runs
   with phantom $\pm 1$ weights corrupts the model.
2. **Row termination:** dense iteration ends at $i = n_{in}$; a sparse stream has
   no per-row endpoint. A per-row active-count or pointer table would reintroduce
   the overhead the format exists to remove.

### 2.4 Five-Bit Token Scheme (32 States)

Token layout: $t[4{:}1] = \mathrm{skip}$, $t[0] = \mathrm{sign}$.

| Token | Class | Decode action (one cycle) |
|---|---|---|
| 0-29 ($\mathrm{skip} \in [0,14]$, either sign) | Active weight | $i^* = i + \mathrm{skip}$; $\mathrm{acc} \leftarrow \mathrm{acc} \pm a[i^*]$ selected by sign; $i \leftarrow i^* + 1$. |
| 30 (`11110`) | Long jump (NOP) | $i \leftarrow i + 15$; accumulator unchanged. |
| 31 (`11111`) | End of row (EOR) | Fire the per-neuron epilogue; $i \leftarrow 0$; $j \leftarrow j + 1$. |

### 2.5 Properties

1. **No pointer table.** Row termination is in-band: the EOR token replaces a
   per-row active-count table.
2. **Trailing zeros are free.** A row whose last active weight is at position 720
   of 784 terminates with one EOR token; the remaining 63 zeros cost one cycle.
3. **Unbounded runs by composition.** A run of $L$ zeros followed by an active
   weight costs $\max(0, \lceil (L-14)/15 \rceil) + 1$ tokens (for example, 35
   zeros: two long jumps plus one active token with skip 5).
4. **Monotonic addressing.** The weight-stream pointer advances by one per token;
   address generation contains no multiplier in any mode.

### 2.6 Export Strategy: Model Separated from Physical Format

The Rust-internal representation remains a plain ternary matrix. The exporter
selects the physical format per target: the NPU receives the five-bit token
stream; a future GPU backend receives 2:4-structured blocks or aligned 32-bit
bitmasks. Training and quantization are independent of the target format.

### 2.7 Layer-Descriptor Mode Assignment

| mode | Weight stream | Intended use |
|---|---|---|
| 0 | Dense Q8.24, 32-bit | Baseline and parity reference. |
| 1 | Dense ternary, 2-bit | Sparsity below ~60%, where token overhead would exceed dense storage. |
| 2 | Sparse ternary tokens, 5-bit | Sparsity above ~60-90%. |

Mode is selected per layer from measured sparsity; empirical calibration is
pending and relates to the planned quantization-loss log (LOG-002 per the
[research index](README.md)).

### 2.8 Open Implementation Points

1. **Memory width and stream separation.** Mode 1 (2-bit) and mode 2 (5-bit)
   assign incompatible meanings to bit 0 — the low operand of the dense
   difference $v = b_0 - b_1$ in mode 1 (activity is $b_1 \oplus b_0$, not a
   dedicated flag), sign in mode 2 (Section 2.4 here versus
   [LOG-007](LOG-007-v0_execution_mode_phasing.md), Section 2.3). Separate
   memories or a width-parameterized stream is decided at v0.2.0 implementation.
2. **Stream integrity.** EOR-only termination desynchronizes on a corrupted or
   misaligned stream; mitigation is the golden-vector parity protocol
   ([`ARITHMETIC_Q8_24.md`](../../engineering/specs/ARITHMETIC_Q8_24.md),
   Section 4), scheduled with the verification phase.
3. **Epilogue interaction.** The accumulation feeding an EOR commit must follow
   the signed-off ternary epilogue
   ([LOG-007](LOG-007-v0_execution_mode_phasing.md), Section 2.2).

---

## 3. Empirical Benchmarks

No RTL support exists; the figures below are projections of the
one-token-per-cycle model and must be replaced by measurements when mode 2 is
implemented in v0.2.0. Benchmark network: MNIST MLP $784 \to 128 \to 10$;
sparsity model: uniform 90% zeros.

| Configuration | Weight storage | Layer-1 row cost | Total inference cycles |
|---|---|---|---|
| Dense Q8.24 (mode 0) | ~3.25 Mb | 785 cycles | ~102,000 |
| Dense ternary 2-bit (mode 1) | ~203 kb (~24.8 KiB) | 785 cycles | ~102,000 |
| Sparse tokens 5-bit (mode 2) | ~62 kb (~8 KiB) | ~95 tokens | ~12,300 |

Derivation: at 90% uniform sparsity a 784-wide row holds ~78 active weights; gaps
longer than 14 zeros add ~16 long-jump tokens in expectation; plus one EOR token.
Row cost ~95 tokens against 785 dense cycles (~8.3x); the ~10x bound is
approached as sparsity exceeds 90% or zeros cluster.

Validation to record here when implemented: per-layer measured sparsity of the
trained network, simulated cycle counts for modes 1 and 2 per layer,
post-synthesis BRAM utilization, and exact reproducible commands per the
[logging guidelines](README.md).

---

## 4. Conclusion and Architectural Implication

Decision: adopt the five-bit token scheme as descriptor mode 2; reserve the mode
field in the v0.1.0 layer descriptor; implement in v0.2.0 (phasing rationale in
[LOG-007](LOG-007-v0_execution_mode_phasing.md)).

Implications:

1. **Multi-PE scaling rule.** Neuron-parallel expansion with replicated
   activation banks: one $784 \times 32$-bit bank is ~3 KiB, so four replicas
   cost ~12 KiB of Spartan-7 BRAM and multiply throughput by four with the
   encoding intact. A lockstep systolic array is not an admissible scaling path
   under high sparsity.
2. **GPU rule.** GPU targets never receive the token stream; they receive 2:4 or
   bitmask exports generated from the same trained model.
3. **v0.1.0 obligations.** The layer engine must implement the linear
   weight-stream pointer and the isolated activation-index increment
   ([LOG-007](LOG-007-v0_execution_mode_phasing.md), Section 4.3), so that mode 2
   insertion in v0.2.0 is confined to token decode and the index step.

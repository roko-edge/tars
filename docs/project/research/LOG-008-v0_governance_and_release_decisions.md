# Research Log LOG-008: v0.1.0 Governance, Engineering Invariants, and Release Decisions

**Document type:** Engineering research log under the protocol defined in
[`V0_CONTRACT.md`](../V0_CONTRACT.md), Section 4.

**State:** Accepted maintainer decisions. Authoritative on commit.
**Date:** 2026-11-01.

This log records the formal maintainer decisions hardening project governance, canonical numeric representations, verification thresholds, and release-path ownership for the v0.1.0 milestone.

---

## 1. Context and Problem Statement

As parallel development accelerates across the Rust tensor migration, hardware RTL synthesis, and training pipeline implementation, multi-party consensus ceremonies and uncoordinated document copies create integration friction. This log records the definitive architectural decisions aligning all repository contracts to a single source of truth.

---

## 2. Maintainer Decisions (November 2026)

1. **Canonical Ternary Encoding Table and Emitter Invariant (2026-11-01):** The canonical hardware identity is $v = b_0 - b_1$ ($w = \{b_1, b_0\}$) across RTL (`t3_dec.sv`), export routines (`export.rs`), and tensor representations (`TENSOR.md` §3.1). The states are `00` = $0$, `01` = $+1$, `10` = $-1$, and `11` = $0$ (redundant zero, auto-cancelling fail-safe). Software encoders MUST emit `00` for zero; hardware and software decoders accept all four codes without error.
2. **Maintainer-Authority Governance Model (2026-10-30):** Following PR #20, all design standards are defined by the repository maintainer (Arthur). Design decisions are recorded with date in research logs. Implementation and documentation move synchronously in a single revision per Engineering Invariant 4. Multi-party sign-off ceremonies are abolished: the maintainer decides, the git history records.
3. **Continuous Integration Auto-Merge Threshold (2026-10-31):** The automated CI merge threshold in `.github/workflows/ci.yml` is set to 200 lines for PRs with green gates (`rust`, `lint`). PRs from `agent/*` branches and PRs exceeding 200 lines are preserved for explicit human review.
4. **Release-Path Decoupling of `T3` Tensor Storage (2026-11-01):** The v0.1.0 release path (quantizer, export pipeline, golden model simulator, and MNIST verification) executes over the current contiguous model representation. The generic `Tensor<T>` typed storage migration proceeds as a parallel engineering refactoring; bit-exact hardware parity serves as the non-regression guard.
5. **Programmable Top Sequencer Deferred to v0.1.1 (Option B, 2026-11-01):** The v0.1.0 release engine executes the target MNIST $784 \to 64 \to 10$ network using the assembled multi-layer wrapper (`npu.sv`). The generic programmable top (instruction sequencer, `program.mem`, ping-pong activation banks, and runtime descriptor execution) is scheduled for the v0.1.1 milestone.
6. **Quantizer and Golden-Model Ownership (2026-11-01):** Mathematical quantization logic, QAT engine implementation (TS-04), and the bit-exact golden reference simulator with `wt.mem`/`w0.mem` export (TS-05) are owned and delivered by Arthur. Gustavo focuses on deterministic test fixtures, typed tensor migration, and benchmark evaluation. Gildo owns the forward-trace cache, `Loss` trait, and batch data loaders (TS-01, TS-02).

---

## 3. Authority and Confirmation

Confirmed by the maintainer (authoritative by governance model of 2026-10-30).

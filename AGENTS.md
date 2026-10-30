# AGENTS.md — Automation & Git Policy

**Status:** [DECIDIDO] by the repository owner (2026-10-28). Effective on commit.
**Scope:** this repository. The former "human-only code and commits" rule is
superseded **here**; human-only remains in force for measured data and academic
text (see tars-research/AGENTS.md — that divergence is intentional).

---

## 1. What agents may do here

Agents may edit code (`.rs`, `.sv`), docs (`docs/`, `**/README.md`), and build
files (`Makefile`, `files.f`, `.gitignore`, CI), and may run the git workflow
**within the limits of §2**. Agents never push to `main`, never rewrite history,
and never touch `AUTHORS`, `LICENSE`, `CITATION*`, or generated artifacts.

## 2. Git operations — allowlist (not listed = ask first)

**Read-only (anytime):** `git status` · `git diff` · `git log` · `git show` ·
`git fetch` · `git branch --show-current`

| Operation | Policy |
|---|---|
| `checkout -b agent/<type>-<slug>` / `switch` | ✅ `<type>` ∈ feat, fix, docs, test, refactor, build, chore |
| `add <path>` | ✅ after reviewing `git diff --cached` |
| `restore --staged <path>` · `git rm --cached <path>` | ✅ `rm --cached` is how generated files leave the index |
| `git commit` | ✅ on `agent/*` only, format + trailer per §3 |
| `git revert <sha>` | ✅ recovery = new commits, never rewrites |
| `git pull --ff-only` · `git stash` / `git stash pop` | ✅ branch upkeep |
| `git merge origin/main` | ✅ **into `agent/*` only** — never into `main` |
| `git push -u origin agent/<name>` | ✅ never `--force`. **Push is publication** — the repo is public and branches are not private: run the §4 scan before pushing, not only before committing |
| push to `main` · merge/rebase **into** `main` | ❌ human only (branch protection) |
| `push --force` / `--all` / `--tags` · `reset --hard` · `clean -fd` | ❌ never |
| `commit --amend` · rebase of pushed branches · `tag`/`release` · branch deletion | ❌ human only |
| `git config`, credentials, hooks, submodules | ❌ never |
| anything not listed above | ❌ ask first |

## 3. Commit protocol

- **One logical change per commit.** Tests with the code; docs in the same
  revision whenever the Invariant-4 trio (docs / exporter / RTL) moves together.
- **Workflow order:** `add` → review `git diff --cached` → `commit` → final
  review of the series → `push`.
- **Message format** — the body explains *why*; the diff already shows *what*.
  Template and example (indent as shown in the commit editor):

        <type>(<scope>): <imperative summary, max 72 chars>

        <why this changed — reasoning, contract references, trade-offs>

        Assisted-By: <agent-name>

  Example of a complete message:

        fix(epilogue): correct saturation polarity in safe_*_to_32

        The sign-bit branches returned MAX for negative overflow and MIN for
        positive. Aligns the RTL with the two-saturation contract (LOG-007 §2.2).

        Assisted-By: MiMo

  `<type>` ∈ feat, fix, docs, test, refactor, build, chore.
- **Attribution is mandatory**: the git author is the human running `git commit`;
  the `Assisted-By:` trailer (or `Co-Authored-By:`) names the agent that prepared
  the change. **Neither field may ever be empty** — this is how the repository
  records who did what.
- **Never commit generated artifacts** — `obj_dir/`, `build/`, `*.bit`, `*.vcd`,
  Vivado projects, generated `*.mem`, `tests/vectors/`, editor/LSP state
  (`.nvim/`, `.vscode/` workspace indexes). If one is tracked, `git rm --cached`
  and extend `.gitignore`.

## 4. Checklist — before every commit AND before every push

1. `git diff --cached` matches the message exactly — no drive-by changes.
2. **Leak scan:** no credentials/tokens/keys; no internal URLs or hostnames; no
   client names, pricing, or commercial strategy; no content copied from other
   repositories; no proprietary engine internals. Any doubt → stop and ask.
3. **Litmus:** *would a competitor with a good LLM gain meaningful time reading
   this diff or message?* If yes → stop and ask.
4. **Gates green** for the area touched (there is **no root Makefile**):

   | Area | Gate |
   |---|---|
   | Rust (`*.rs`) | `cargo test` |
   | RTL (`npu/`) | `make -C npu lint` and `make -C npu test` |
   | Docs / build files | checklist only |

5. **Atomic:** two unrelated concerns in one diff → split before committing.

## 5. Authorship & integrity

- **Every commit carries attribution** (git author = human; `Assisted-By` trailer
  = agent). Never forge authorship, never remove existing trailers.
- `AUTHORS`, `LICENSE`, `CITATION*`: human-only, never edited by agents.
- Never fabricate metrics, results, or experiment data. Numbers require a real
  run: CSV + seed + reproducible command.
- Keep the `Assisted-By` trail on everything you touch — AI-assistance
  disclosure is part of the project's academic integrity policy.

## 6. Project layout

**On disk today:**

    npu/src/        RTL core: npu_pkg, t3_dec, pe, epilogue, layer_engine, npu wrapper
    npu/tests/      testbenches (currently tb.sv)
    npu/synthesis/  board-only (Urbana) — currently untracked [PENDENTE: versioning]
    npu/data/       generated *.mem — never commit
    docs/           one decision, one home; status labels on every decision

**Target test model [PROPOSTO]** — tiers, not files that exist yet:
① `tb_core` self-checking (every commit) · ② `tb_parity` against golden vectors
from the Rust exporter · ③ `tb_shell` system-level. Do not claim these exist
until they land.

- `files.f`: `npu_pkg.sv` **always first**.
- Docs: English; conclusion first; question-style headings; tables over prose;
  labels `[DECIDIDO]` / `[PENDENTE]` / `[PROPOSTO]` / `[CONFLITO]`.
- Contradictions between docs and code → mark `[CONFLITO]` and report. Never
  resolve silently (Invariant 4: docs, exporter and RTL move together).

## 7. Who edits what

| Target | Policy |
|---|---|
| `docs/`, `**/README.md`, code, build files | agents (per §1) |
| **This file (`AGENTS.md`)** | **human commit only** — agents may draft changes, but need explicit approval before staging |
| `.gitignore` | agents may **add generated paths**; never edit it to hide artifacts from the "never commit" rules or to bypass this policy |
| `AUTHORS`, `LICENSE`, `CITATION*` | human only |

## 8. Escalation — stop and ask when

- a change touches the format contract (`.mem`, descriptor, golden vectors)
  without the matching docs/RTL revision;
- §4.2/§4.3 raise any doubt about leakage;
- a git operation is not explicitly allowed in §2;
- a task seems to reference another repository's internals.

A stopped commit costs minutes. A leaked one costs the project. Report — don't improvise.

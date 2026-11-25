# AGENTS.md — Automation & Git Policy

**Status:** [DECIDED] by the repository owner (2026-10-28, amended 2026-10-30, 2026-11-19, and 2026-11-22).
Effective on commit. The 2026-11-19 amendment authorizes agents to execute the
git workflow (staging, commit, push, PR creation) for human-written code, while
maintaining strict human ownership of all code files (.rs, .sv, .svh, testbenches):
agents never write or edit code files directly, acting exclusively by proposing
diffs in chat, teaching, and researching implementation methods.
The 2026-11-22 amendment requires agents to run the applicable verification
gates and inspect their results before committing or pushing changes.
**Scope:** this repository. Human-only: code writing and editing (§1), measured data and academic
text (see tars-research/AGENTS.md — that divergence is intentional).

---

## 1. What agents may do here

Agents may edit docs (`docs/`, `**/README.md`) and build files (`Makefile`,
`files.f`, `.gitignore`, CI), and may run the git workflow (`git add`,
`git commit`, `git push`, `gh pr create`) **within the limits of §2**.

**Code is strictly human-authored.** All `.rs`, `.sv`, `.svh` files and testbenches
are written and edited by humans only. Agents **never** directly edit, write,
generate to disk, or format code files — directly **or indirectly** (formatters,
code generators, scripts, symlinks, or alternate paths that write into code files
are equally forbidden). When assisting with code, the agent acts exclusively
through chat: proposing diffs, explaining architectural and mathematical
principles, researching methods, and reviewing code. Once the human has written
or applied the code into the file, the agent is authorized to automate the git
workflow (staging, commit, push, PR creation) adhering strictly to the attribution
and commit protocols of §3.

Agents never push directly to `main`, never rewrite history, and never touch
`AUTHORS`, `LICENSE`, or `CITATION*`. Verification commands may produce their
normal build outputs. The generated-artifact exclusions in §3 apply.

## 2. Git operations — allowlist (not listed = ask first)

**Read-only (anytime):** `git status` · `git diff` · `git log` · `git show` ·
`git fetch` · `git branch --show-current`

| Operation | Policy |
|---|---|
| `checkout -b <type>/<slug>` / `switch` | ✅ `<type>` ∈ feat, fix, docs, test, refactor, build, chore, agent |
| `add <path>` | ✅ after reviewing `git diff --cached` — authorized for all repository files (including code files written/applied by the human) |
| `restore --staged <path>` · `git rm --cached <path>` | ✅ index management and untracking generated artifacts |
| `git commit` | ✅ on working branches. Format + trailer per §3 (git author = human; `Assisted-By:` mandatory) |
| `git revert <sha>` | ✅ recovery = new commits, never rewrites |
| `git pull --ff-only` · `git stash` / `git stash pop` | ✅ branch upkeep |
| `git merge origin/main` | ✅ into working branches only — never into `main` |
| `git push -u origin <branch>` | ✅ never `--force`. **Push is publication** — the repo is public: run §4 scan before pushing |
| push to `main` · merge/rebase **into** `main` | ❌ human only (branch protection) |
| `push --force` / `--all` / `--tags` · `reset --hard` · `clean -fd` | ❌ never |
| `commit --amend` · rebase of pushed branches · `tag`/`release` · branch deletion | ❌ human only |
| `git config`, credentials, hooks, submodules | ❌ never |
| anything not listed above | ❌ ask first |

For human-led changes, name the branch for its primary purpose. Use
`feat/<slug>` for new behavior, `fix/<slug>` for a bug fix, and `refactor/`,
`docs/`, `test/`, `build/`, or `chore/` as appropriate. Reserve
`agent/<slug>` for work prepared entirely or predominantly by an agent
within its permitted scope, such as documentation maintenance. Review,
testing, or Git automation alone does not make a human-led change an
`agent/` branch. Record assistance in the commit message per §3.

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

        docs(protocol): add start/done protocol contract

        Seven normative clauses for the engine/wrapper handshake; owns what
        ARCHITECTURE.md previously implied inline.

        Assisted-By: MiMo

  `<type>` ∈ feat, fix, docs, test, refactor, build, chore.
- **Attribution is mandatory**: the git author is the human running `git commit`;
  the `Assisted-By:` trailer (or `Co-Authored-By:`) names the agent that prepared
  the change. **Neither field may ever be empty** on agent-prepared commits.
- **Code commits are human-authored.** When a code change originated from an
  agent-proposed diff (in chat) and the human applied it, the commit message
  carries `Assisted-By:` — assistance is disclosed; authorship stays human.
  A code commit never claims sole human origin when an agent produced the change.
- **Never commit generated artifacts** — `obj_dir/`, `build/`, `*.bit`, `*.vcd`,
  Vivado projects, generated `*.mem`, `tests/vectors/`, editor/LSP state
  (`.nvim/`, `.vscode/` workspace indexes). If one is tracked, `git rm --cached`
  and extend `.gitignore`.
- **Shell-safe message application:** PR bodies and commit messages are written
  to a file and applied with `gh pr edit --body-file` / `git commit -F`, or via
  the GitHub web editor. They are NEVER passed through unquoted shell strings:
  `$` and backticks are shell syntax and will be expanded. Second occurrence,
  2026-11-01 (PRs #20/#21).

## 4. Checklist — before every commit AND before every push

1. `git diff --cached` matches the message exactly — no drive-by changes.
2. **Leak scan:** no credentials/tokens/keys; no internal URLs or hostnames; no
   client names, pricing, or commercial strategy; no content copied from other
   repositories; no proprietary engine internals. Any doubt → stop and ask.
3. **Litmus:** *would a competitor with a good LLM gain meaningful time reading
   this diff or message?* If yes → stop and ask.
4. **Gates green** for the area touched. Agents run the applicable commands
   and inspect their results before committing or pushing. Report failures
   or unavailable checks. Record a pass only for an observed run tied to the
   tested revision. There is **no root Makefile**.

   | Area | Gate |
   |---|---|
   | Rust (`*.rs`) | `cargo test` |
   | RTL (`npu/`) | `make -C npu lint` and `make -C npu test` |
   | Docs / build files | checklist only |

5. **Atomic:** two unrelated concerns in one diff → split before committing.

## 5. Authorship & integrity

- **Code is written by human hands.** Agents assist by proposing diffs in chat,
  teaching, diagnosing, and reviewing — never by writing code directly into the repository.
  Git operations (staging, commit, push, PR creation) on non-main branches may be executed
  by agents on behalf of the human author after verification gates pass.
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
    npu/tests/      legacy smoke, Tier 1 core checks, and Tier 2 parity bench
    npu/synthesis/  board-only lab (Urbana), git-ignored
    npu/data/       generated *.mem — never commit
    docs/           one decision, one home; status labels on every decision

**Test model:** ① `tb_core` runs hand-calculated self-checks. ② `tb_parity`
exists, with the independent bit-exact reference and exact comparison pending.
③ `tb_shell` is planned for the programmable top. Record a pass only for an
observed run tied to the tested revision.

- `files.f`: `npu_pkg.sv` **always first**.
- Docs: English; conclusion first; question-style headings; tables over prose;
  labels `[DECIDED]` / `[PENDING]` / `[PROPOSED]` / `[CONFLICT]`.
- **Keep documentation current.** When an agent verifies that a document
  presented as current is stale, the agent must update the canonical document
  and affected links in the same task. Check claims against the inspected
  source, identify working-tree changes separately from committed behavior,
  and distinguish supported commands, observed results, and planned work.
  Preserve dated historical records as history.
- If the implemented behavior conflicts with a contract, record `[CONFLICT]`
  with source references and report it in the handoff. Apply recorded maintainer
  decisions when reconciling docs. Unresolved contract decisions retain their
  status (Invariant 4: coordinated changes to docs, exporter, and RTL).

## 7. Who edits what

| Target | Policy |
|---|---|
| **Code: `.rs`, `.sv`, `.svh`, testbenches** | **human-authored** — agents propose diffs in chat, teach, and research; never write or edit code files directly. Staging/committing code is authorized for agents after human writes |
| `docs/`, `**/README.md`, build files | agents (per §1) |
| **This file (`AGENTS.md`)** | **human commit only** — agents may draft changes, but need explicit approval before staging |
| `.gitignore` | agents may **add generated paths**; never edit it to hide artifacts from the "never commit" rules or to bypass this policy |
| `AUTHORS`, `LICENSE`, `CITATION*` | human only |

## 8. Escalation — stop and ask when

- **asked to edit, generate, or format a code file** → the answer is a proposed
  diff in chat, never a write;
- a change touches the format contract (`.mem`, descriptor, golden vectors)
  without the matching docs/RTL revision;
- §4.2/§4.3 raise any doubt about leakage;
- a git operation is not explicitly allowed in §2;
- a task seems to reference another repository's internals.

A stopped commit costs minutes. A leaked one costs the project. Report — don't improvise.

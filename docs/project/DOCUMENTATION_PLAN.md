# How should TARS maintain its documentation?

**Status [PROPOSED].** Organize documentation around two questions. What does
this checkout do? What must a future release deliver? Use [the documentation
index](../README.md) to route each question to one authoritative section. Keep
the current answer short, with links to the contract, source, test, or decision
needed for deeper work.

| Start here for | Section |
| --- | --- |
| Navigation and what belongs where | [Documentation tree](#how-should-readers-find-an-answer) |
| Conflicting claims and evidence | [Authority](#which-source-answers-each-question) |
| When a change needs a documentation update | [Maintenance](#what-keeps-pages-current) |
| How an agent checks claims and proposes corrections | [Documentation audit](DOCUMENTATION_AUDIT.md) |
| Order of work and completion | [Migration](#how-should-tars-migrate) · [Acceptance](#how-will-the-team-measure-success) |

## What does the current tree reveal?

The 2026-11-23 inspection started from `a51a193` and compared the training
sources with `origin/main` at `f72bca4`. The reusable trainer was already
merged. Current-facing pages still described the older QAT runner or called
the trainer a working-tree change. The pilot gives the implemented API and
runner preset one current reference each and keeps earlier snapshots in the
delivery record. [Training API](../API.md#3-training-operations) ·
[Runner behavior](../EXPERIMENTS.md#selection-and-execution) ·
[Snapshot history](V0_DELIVERY_PLAN.md#13-what-changed-with-the-reusable-trainer)

| Priority | Evidence | Migration action |
| --- | --- | --- |
| High | `QUANT.md` gives inconsistent tie rules, and `quantize_scalar` uses a different threshold. The Q8.24 contract also records an unresolved external rounding choice. | Keep the current behavior and intended rule side by side in each specification. Keep the release disposition in one register. [Ternary rules](../engineering/specs/QUANT.md#1-mathematical-formulation) · [Dispositions](V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition) |
| High | Tier 1 checks hand-calculated values. Tier 2 currently allows one LSB and consumes float-derived expectations. Exact parity requires independent integer expectations. | Route simulation commands to the NPU guide, wire semantics to the format and arithmetic references, and release proof to the delivery record. [Bench and evidence](../engineering/MEM_FORMAT.md#what-remains-pending-for-release-parity) |
| High | The roadmap mentions bootstrap intervals and assigns Urbana timing closure and a high-level export method. The release contract specifies a binomial interval, exact parity and artifact requirements. | Cite acceptance criteria from the contract. Keep assignments in the roadmap and record any unresolved scope decisions with the maintainer. [Assignments](ROADMAP.md#what-is-the-current-v010-work) · [Dispositions](V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition) |
| Medium | The baseline delivery plan, roadmap, and tensor plan together exceeded 15,000 words. The old roadmap phases and advisory tensor milestones duplicated active planning. | The current tree drops both schedules. Work-package dependencies and release acceptance keep their own references. Git preserves the older schedules at `a51a193`. [Roadmap](ROADMAP.md) · [Tensor packages](TENSOR_PHASE_B.md#3-work-package-decomposition) |

## How should readers find an answer?

Retain established paths. The tree shows the reading order after migration.

```text
README.md
└── docs/README.md
    ├── How does the code work?
    │   ├── GETTING_STARTED.md           build and run
    │   ├── API.md                       supported Rust surface and source links
    │   ├── EXPERIMENTS.md               current presets and runner behavior
    │   ├── ../npu/README.md             simulation commands and test tiers
    │   └── engineering/README.md       architecture and contracts by domain
    └── What is the project working toward?
        └── project/README.md
            ├── ROADMAP.md               active work and ownership
            ├── V0_CONTRACT.md           release acceptance
            ├── V0_DELIVERY_PLAN.md      dependencies, evidence, and conflicts
            ├── TENSOR_PHASE_B.md        tensor work-package details
            └── research/README.md       decision history and evidence process
```

Code documentation covers guides, current API, architecture and technical
contracts. An accepted target contract names its target version and any
observed deviation. Planning covers future work, release gates, decisions,
and evidence. An engineering proposal can keep its existing URL, with the
planning index routing to it. Treat `project/STATUS.md` as a short entry point
to current code references and open release gates. Keep compatibility links
out of normal reading paths and point each directly to its destination.

Start with a question from the index and open the linked section. Follow the
source on the same revision when an answer depends on an exact default, API
signature, or numeric rule. Follow the run when reporting a verification
result. Use exact terminology such as `Trainer::fit`, `start`/`done`,
`wt.mem`, and `Q8.24` in link text or nearby summaries. For a local checkout,
`rg` can locate a symbol or heading before opening the relevant line range.
Keep `AGENTS.md` for execution policy and a pointer to the same index. Align
agent-specific instructions with the repository policy when tooling is updated.

Start each leaf with its answer and scope. Give guides a command, prerequisites,
and expected observable output. Give references the relevant types, units,
shapes, errors, and invariants, with source and verification links. Give a
planning item an owner, state, dependency, and acceptance evidence. Link to the
clause that defines a reused rule. Split a long page when its questions have
different readers or update triggers. Keep a long specification intact when
its clauses need to be read together, with direct links to those clauses.

## Which source answers each question?

| Question | Authority | How other records refer to it |
| --- | --- | --- |
| What executes in this checkout? | Rust and RTL sources for behavior, Makefile and CI workflow for entry points | Code docs summarize observed behavior and link to the relevant source. |
| What interface or boundary must implementations preserve? | Engineering specification for that boundary | Link to [handshake](../engineering/PROTOCOL.md), [artifact layout](../engineering/MEM_FORMAT.md), [arithmetic](../engineering/specs/ARITHMETIC_Q8_24.md), or [quantization](../engineering/specs/QUANT.md) by clause. |
| What is required to tag a release? | [Release contract](V0_CONTRACT.md) | Plans cite the clause and link to revision-bound evidence. The roadmap owns assignment and priority. |
| Who decided and why? | Accepted dated decision record | The maintainer's disposition is applied to the affected canonical clause. Link the decision and any superseded record. [Decision history](research/README.md) |
| What passed on a specific revision? | Observed run with command, exact revision, result and artifacts | Cite the run and its scope. For claims involving benchmarks, record the protocol, input identities, and seeds. [Evidence process](research/README.md) |

Use `[DECIDED]` for an accepted decision, `[PROPOSED]` for an option under
review, `[PENDING]` for unfinished work or missing evidence, and `[CONFLICT]`
for a material disagreement. Implementation present and verification observed
are independent facts. Describe local changes as working-tree observations
with their base commit. Use relative links for checkout-specific docs and
revision-pinned links for historical source. A completed feature may still have
open release evidence.

When code and a contract disagree, document observed behavior in the current
reference and preserve the normative clause. Put both citations and the
required disposition in the conflict register. The maintainer decides whether
to change the code, the contract, or the release scope. Apply the decision to
the owning clause and its incoming links in the same revision. Changes to
numeric representation, rounding, or wire semantics follow Engineering
Invariant 4 across the contract, exporter, and RTL. [Conflict register](V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition) ·
[Engineering invariants](ROADMAP.md#4-engineering-invariants-and-anti-regression-policy)

## What keeps pages current?

The proposed workflow scales with the effect of the change. Use the existing
delivery workflow for internal changes. Update the owning section when a
change affects how someone uses TARS, a technical contract, or release scope.

| Trigger | Change and review |
| --- | --- |
| Internal refactor, formatting, or equivalent optimization | Use the existing code checks and review. A changed path can suggest a reference to consult. |
| Public API, default, command, or documented behavior changes | Update the owning guide or reference in the same PR. Include any source or evidence necessary to substantiate the change. The usual reviewer checks the relevant text with the code. |
| Rust–RTL boundary changes | The affected domain owners review the canonical clause and verification fixture with the maintainer. The same revision carries the exporter, RTL and documentation change required by Invariant 4. |
| Scope or assignment changes | The work owner updates the roadmap and dependency links. The maintainer records any scope decision and updates release acceptance in its contract. |
| A run completes | The operator records the revision, command, scope and observed result. The delivery plan links the evidence for the matching gate. |
| Release candidate | The release owner focuses on affected references, open conflicts, and release evidence. Verify claims against the exact candidate. |

### What goes in the documentation review?

Keep the edit proportional to the changed fact. A default changing from 32 to
64 may need one line in its reference. Guides that link to the reference
already lead to the new value. Use the ordinary PR description when the impact
needs explanation. A documentation-only correction to a technical claim cites
the source or test that supports it.

### What can CI check cheaply?

| Check | Scope |
| --- | --- |
| Navigation | The documentation links workflow checks local links and anchors on PRs to `main`, pushes to `main`, and manual runs. The Actions summary shows any broken targets. Use the navigation trial below to check active-page reachability. Consider a merge requirement after observing the workflow in use. |
| Review hints | Suggest owning sections for relevant source paths. The same `src/train.rs` diff can be a refactor or a change to a public default. Treat path matches as advisory. |
| Evidence | Use the existing Rust and RTL gates for their stated scopes. Prefer executable Rust examples where maintainers add them. Check cited results against the revision and fixture they cover. |

If the maintainer adopts a required navigation check, integrate that result
with the current auto-merge job and let out-of-scope diffs complete through
the existing workflow. Keep external link availability and length budgets
advisory.
[Documentation links workflow](../../.github/workflows/docs-links.yml) ·
[CI workflow](../../.github/workflows/ci.yml)

Pilot a broken local anchor, a changed default with a stale API reference,
and an internal refactor in the same source file. The navigation check
reports the broken anchor, the ordinary review catches the stale default,
and the refactor proceeds through the existing checks and review workflow.

### How is drift corrected after a merge?

When a contributor finds stale text, correct the owning section and affected
links in the same task using the inspected revision. For a code–contract
disagreement, follow the conflict procedure above. At release review, confirm
that each affected guide works on the candidate, planned items still have
owners, and test claims point to observed evidence. Keep unpublished planning
and private inputs in the team's authorized records.

Use an agent-assisted audit on demand for changes spanning several references,
known drift, or release claims that need reconciliation. A scheduled report
can catch drift between reviews. The audit follows the same index and
evidence rules as human review.
[Audit procedure and Antigravity pilot](DOCUMENTATION_AUDIT.md)

## How should TARS migrate?

Deliver scoped revisions in dependency order. During migration, land each new
owner section together with the links that route to it. Preserve public paths
and inbound anchors where practical. When moving a section, provide a
one-hop pointer with its old anchor for established inbound links. Retain
dated decisions and measured evidence under the existing provenance rules.

| Order | Scope and responsible role | Exit condition |
| --- | --- | --- |
| 1 | **Navigation and training pilot.** Documentation editor and Rust owner align the index, API, experiments, guide, and status with one checkout. | The training and evaluation questions resolve to current sections and source. Earlier snapshots are explicitly dated. |
| 2 | **NPU boundary.** Rust export and RTL owners map artifact names and shapes, quantization, arithmetic, handshake, and tier evidence to their owning clauses. | A reader can trace each artifact from producer to consumer. Known rounding and parity discrepancies retain their status and links. |
| 3 | **Active planning.** Work owners reduce the roadmap to release scope links, active packages, ownership and blockers. The release owner moves acceptance references and evidence to their authority. | Each open work package has an owner and evidence criterion. The dated delivery snapshots and tensor work-package IDs remain reachable. Obsolete schedules live in Git history. Benchmark collection procedures have a planning entry point. |
| 4 | **Automation and policy alignment.** Pilot navigation checks and optional review hints. | Confirm that the check finds broken routes and that internal changes still follow the existing workflow. Integrate with auto-merge if a check becomes required. |
| 5 | **Candidate review.** The release owner measures navigation and verifies the transition on the candidate. | The acceptance checks below pass. The project index leads to the maintained workflow and the migration record is dated. |

For each existing page, record its current purpose, canonical destination,
inbound links, and action in the migration PR. Choose keep, shorten, link, or
preserve as dated history. Use Git history for superseded prose that needs no
stable destination. Keep scope and authorship records under their existing
maintainers. If a migration breaks discovery, restore the previous pointer
and update the affected links in a corrective revision.

## How will the team measure success?

Use five fixed questions before and after migration. A human reviewer and
an agent start at `docs/README.md` with the task and checkout revision.
Give each the same questions and record the answer, cited section, and any
wrong turn. The human trial checks that the answer makes sense without
prior repository knowledge and that a chosen guide has the prerequisites,
command, and expected outcome needed to use it. The agent trial also records
files opened and words actually read. Compare tokens only across the same
agent configuration and tasks.

| Question | Expected destination |
| --- | --- |
| What mode, batch size, and evaluation data does the MNIST runner use today? | [Experiment behavior](../EXPERIMENTS.md#selection-and-execution) and [trainer defaults](../API.md#3-training-operations) |
| Can I submit a new request while the NPU is busy? | [Protocol clauses](../engineering/PROTOCOL.md#what-is-the-contract-decided) |
| What writes `target.mem` today and what evidence closes parity? | [Memory format](../engineering/MEM_FORMAT.md) and [parity gate](V0_DELIVERY_PLAN.md#24-how-do-export-and-rtl-establish-exact-parity-pending) |
| What does v0.1.0 require and who owns the remaining work? | [Release contract](V0_CONTRACT.md) and [active assignments](ROADMAP.md#what-is-the-current-v010-work) |
| Is the ternary threshold settled and what does the current exporter use? | [Quantization specification](../engineering/specs/QUANT.md#1-mathematical-formulation) and [conflict register](V0_DELIVERY_PLAN.md#4-which-decisions-need-an-explicit-disposition) |

The initial target is five correct, sourced answers for each reader. At
least four agent answers should take at most three documents after the index
and 1,500 words of selected reading. Record any full-file reads and the
reason for them. Use the result to fix the route or rewrite the answer.
Audit every migrated active page for reachability, source links where
relevant, and one owner per claim. Require zero broken local links in
migrated pages.

The first pilot compared the complete index and selected answer sections
against `a51a193`. These are whitespace-delimited Markdown word counts for
two fixed reading paths. The document counts include the index.

| Question | Before | Pilot |
| --- | --- | --- |
| MNIST mode, batch size, and evaluation split | 3 documents, 1,015 words | 3 documents, 606 words |
| NPU behavior when `start` arrives while busy | 3 documents, 640 words | 2 documents, 375 words |

## Which practices inform this plan?

- Diátaxis distinguishes learning, tasks, reference and explanation. Apply
  those distinctions inside the code documentation as content warrants.
  [Diátaxis](https://diataxis.fr/start-here/)
- Kubernetes links to the canonical source to reduce maintained copies.
  TARS can do the same for boundaries shared by Rust and RTL.
  [Kubernetes content guide](https://kubernetes.io/docs/contribute/style/content-guide/#dual-sourced-content)
- GitLab encourages documentation to travel with the code change. Rustdoc
  supports executable API examples. Use these for integration review and
  claims that can be tested directly.
  [GitLab workflow](https://docs.gitlab.com/development/documentation/workflow/#including-documentation-with-code) ·
  [Rustdoc tests](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html)

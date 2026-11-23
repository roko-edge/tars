# How should an agent audit TARS documentation?

**Status [PROPOSED].** Use a scoped audit as part of the
[documentation maintenance workflow](DOCUMENTATION_PLAN.md#what-keeps-pages-current).
Each finding names a claim, the revision it describes, and the evidence for the
proposed action. The ordinary contributor or release reviewer can verify the
finding from the same references.

## What should the audit read?

1. Pin the revision and scope. Use the public commit for a baseline, the base
   and head for a proposed change, or the exact candidate for a release claim.
   Describe working-tree observations separately and record their base.
2. Read [repository policy](../../AGENTS.md) and start with the affected
   question in [the documentation index](../README.md). Open the owning
   section, then the relevant source or evidence. Use the changed paths and
   exact symbols to find related claims and summaries. Read a full file when
   the answer depends on several clauses.
3. Identify what the claim asserts. Current behavior needs source at the
   inspected revision. A technical or release requirement needs its accepted
   contract. A test or measurement result needs an observed run. Apply the
   [authority rules](DOCUMENTATION_PLAN.md#which-source-answers-each-question).

For each inspected area, report the revision, scope, and outcome. A finding
contains the affected section, a source or run reference, the type of
discrepancy, and the proposed action. List any scope that lacked evidence or
access. A clean result applies to the claims that were actually inspected.

## Which findings can be corrected?

| Finding | Action |
| --- | --- |
| Current documentation disagrees with unambiguous source | Correct the owning Markdown section and affected links or summaries in the same task. Cite the source and keep the edit focused. |
| Accepted contract and implementation disagree | Preserve the contract and record the observed behavior and `[CONFLICT]` with both references. The maintainer decides the disposition. Apply Engineering Invariant 4 for numeric or format changes. |
| A claimed run or result lacks matching evidence | Identify the missing revision, command, outcome, or fixture. Keep verification pending until evidence is observed. Run applicable gates during publication review as specified by repository policy. |
| The claim matches its authority | Preserve the text. Record the checked scope in the review when relevant. |

For an externally assisted audit, have the CLI return Markdown edits as a
proposed diff. The normal contributor applies the verified corrections to
the canonical sections. Recheck the revision, source lines, and target
sections before application if the working tree has advanced. Validate
links and inspect the resulting diff. The normal reviewer checks any
changed technical claim against its evidence. Existing rules govern
authorship, attribution, and publication.

## What should reach the public main branch?

Keep accepted status in [STATUS.md](STATUS.md) and put a confirmed technical
correction in the section that owns the claim. Update the status summary when
the confirmed change affects a capability, limitation, or release gate. Use
the [delivery plan](V0_DELIVERY_PLAN.md) for an open release conflict or missing
release evidence. Keep historical observations tied to their original revisions.

The Antigravity conversation holds each semantic audit report. GitHub Actions
holds each structural link check and its summary. A reviewed correction belongs
in the repository through a PR with the inspected SHA, the affected section,
and the evidence for the change. Include evidence needed for future review
directly in the PR or canonical section with stable source links.
[Documentation links workflow](../../.github/workflows/docs-links.yml) ·
[GitHub job summaries](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary)

When preparing an agent-assisted correction PR, use an `agent/*` working
branch, recheck the finding against the target revision if `main` advanced,
and follow the attribution and public-repository checks. The current TARS CI
excludes `agent/*` from its auto-merge job. The maintainer reviews and merges
the PR. Keep unresolved evidence gaps in the run report or an existing
release tracking record as appropriate.
[Repository policy](../../AGENTS.md) · [CI workflow](../../.github/workflows/ci.yml)

## How does this work with Google AI Pro?

As of 2026-11-23, Google AI Pro includes Antigravity access with increased
usage limits. Run the Antigravity CLI (`agy`) from the root of an isolated copy
of the published TARS revision and sign in with the personal Google Account
that has the subscription. The CLI discovers the root `AGENTS.md`. Its rule
discovery also includes global rules and rules from installed plugins.
Explicitly inspect any applicable instructions in subdirectories.
[Google AI Pro benefits](https://support.google.com/googleone/answer/14534406) ·
[CLI authentication](https://antigravity.google/docs/install?tab=cli) ·
[Rules](https://antigravity.google/docs/rules/)

Use a public-only copy without private attachments, private remotes, local
research assets, untracked inputs, or credentials in the workspace. Local
execution gives the agent access to the local files and tools it needs for
the review. Keep that access scoped to the approved snapshot. For the first
run, use a read-only review configuration with file writes, shell commands,
web access and MCP tools disabled. Supply the revision and any relevant diff
as approved inputs. Confirm that the effective permissions allow the intended
reads and block writes on a scratch file before trusting the setup.
[CLI permissions](https://antigravity.google/docs/permissions?tab=cli)

Start with one interactive session. Ask for evidence and a proposed Markdown
diff, then verify the citations locally. A starting prompt is below.

> Audit the listed documentation sections for the supplied TARS revision.
> Read AGENTS.md and the documentation index first. Check current behavior
> against source, requirements against their accepted contract, and results
> against the observed run. Output the inspected scope, evidence and any
> proposed Markdown diff. List disagreements and missing inputs separately.

When an automated invocation would save work, the local CLI supports
`agy -p`, JSON output and
`--json-schema`. Pin the execution mode, permissions and timeout. Check
both the terminal result and evidence coverage. The CLI can return exit
code zero after denying a tool call that required approval. Treat a missing
required input, invalid output or interrupted run as an incomplete audit.
The CLI documentation describes separate API-key authentication for
unattended environments and programmatic usage has its own billing terms.
[Headless mode](https://antigravity.google/docs/cli/headless/) ·
[Gemini API billing](https://ai.google.dev/gemini-api/docs/pricing)

Google also offers Remote Agents in a Google-managed environment. The
documented remote session lacks CLI print mode, so the headless procedure
above uses the local CLI. [Remote Agents](https://antigravity.google/docs/remote-agents)

## How should the Antigravity App schedule an audit?

Use **Automations** in the Antigravity App. A local automation starts a
fresh conversation for each run. Keep the instructions self-contained and
select the dedicated public checkout as its workspace. **Create with Prompt**
also offers a guided test run and permission review before activation.
[Automation setup](https://antigravity.google/docs/commands/automation/)

Start with a paused, weekly, report-only automation. The schedule is a
backstop for issues that slip through ordinary review. Use the available
Flash model for the pilot and change models only if the observed reports
justify it. Keep the report and proposed Markdown diff in Antigravity.
Correct confirmed issues through the [existing review process](DOCUMENTATION_PLAN.md#what-keeps-pages-current).

| Step | Expected behavior |
| --- | --- |
| Refresh | Fetch only the public `main` in the isolated checkout and pin `FETCH_HEAD` to a full SHA. Use that SHA for every source citation. Mark the audit incomplete if refresh or evidence access fails. |
| Scope | Verify the three pilot cases below and route relevant changes from a bounded recent window through the documentation index. State the actual revision, window, and sections reviewed. |
| Permissions | Review effective permissions for the audit project. Allow the scoped public fetch and Git reads. Restrict workspace writes, unrelated commands, other workspaces, web and MCP access. Unattended requests that require approval can be denied. |
| Output | Return evidence and a minimal Markdown proposal for each proven error. Report contract conflicts and missing evidence explicitly. A clean report applies to the inspected scope. |

Run a pilot in the dedicated checkout with the same prompt and permissions.
Check the three expected decisions and verify that the checkout is clean.
Activate recurrence after a result with usable citations. A later change to
the model, scope, or permissions warrants a rerun of these cases.
[Automations and scoped permissions](https://antigravity.google/docs/sidecars/#automations)

## What cases should qualify the pilot?

Use the following cases before relying on the agent's corrections. They
exercise three different decisions at the same public source baseline.

| Case | Expected finding | Evidence |
| --- | --- | --- |
| Stale API text in the published baseline | The reusable trainer is in the committed source. Its default batch size is 32. Propose the current API wording. | Inspect `git show 495cc9c:docs/API.md`. [Trainer implementation](../../src/train.rs) · [Corrected reference in this working tree](../API.md#3-training-operations) |
| Quantizer threshold | The current scalar threshold and the documented rules disagree. Preserve the open conflict and cite both. | [Scalar quantizer](../../src/quant/ternary.rs) · [Specification](../engineering/specs/QUANT.md#1-mathematical-formulation) |
| NPU test guidance | `make test` selects Tier 1. Preserve that guidance. Exact Tier 2 parity needs its own evidence and the current comparator allows one LSB. | [Make targets](../../npu/Makefile) · [Test commands](../../npu/README.md#commands) · [Comparator](../../npu/tests/tb_parity.sv) |

Manual source inspection at public commit `495cc9c` confirmed these
expectations. The Gemini pilot passes
when Antigravity independently produces all three dispositions with valid
citations, leaves the working tree intact, and reports the scope it checked.
Compare review time and the reported tokens if available. Use the
[navigation trial](DOCUMENTATION_PLAN.md#how-will-the-team-measure-success)
to measure whether both humans and agents can retrieve the same answers.

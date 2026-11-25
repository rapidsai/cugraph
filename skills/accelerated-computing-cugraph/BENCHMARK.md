# Skill Benchmark: accelerated-computing-cugraph

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `accelerated-computing-cugraph`
- Validation status: `passed`
- Evaluation date: 2026-11-24
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 11 evaluation tasks (10 positive, 1 negative)
- Dataset digest: `sha256:04569661f39bd2d6f5c54ea1fb3eda6db8d6dec7ebf38ef52e2ddcb85d166426` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| accelerated-computing-cugraph | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| accelerated-computing-cugraph | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| accelerated-computing-cugraph | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| accelerated-computing-cugraph | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| accelerated-computing-cugraph | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `accelerated-computing-cugraph`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 78.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 81.8% → 81.8% (±0.0 points) |
| Correctness | 61.8% → 100.0% (+38.2 points) | 54.6% → 96.4% (+41.8 points) |
| Discoverability | 99.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 29.4% → 69.4% (+40.0 points) | 32.5% → 56.1% (+23.6 points) |
| Efficiency | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 67.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.8% → 94.5% (-2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.3% → 83.8% (-2.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,986,294 | 1,648,864 | +337,430 | +20.46% | skill 11/11; base 11/11 |
| claude-code | cugraph-direct-multi-node | 161,202 | 111,585 | +49,617 | +44.47% | skill 1/1; base 1/1 |
| claude-code | cugraph-empty-rank | 115,906 | 143,404 | -27,498 | -19.18% | skill 1/1; base 1/1 |
| claude-code | cugraph-existing-dask | 164,248 | 106,741 | +57,507 | +53.88% | skill 1/1; base 1/1 |
| claude-code | cugraph-gnn-default | 150,332 | 71,138 | +79,194 | +111.32% | skill 1/1; base 1/1 |
| claude-code | cugraph-gnn-distributed | 156,492 | 144,649 | +11,843 | +8.19% | skill 1/1; base 1/1 |
| claude-code | cugraph-local-spawn | 299,044 | 117,226 | +181,818 | +155.10% | skill 1/1; base 1/1 |
| claude-code | cugraph-memory-speed | 74,030 | 71,500 | +2,530 | +3.54% | skill 1/1; base 1/1 |
| claude-code | cugraph-negative | 35,375 | 70,342 | -34,967 | -49.71% | skill 1/1; base 1/1 |
| claude-code | cugraph-networkx-migration | 250,829 | 313,874 | -63,045 | -20.09% | skill 1/1; base 1/1 |
| claude-code | cugraph-scorer-semantics | 242,566 | 187,113 | +55,453 | +29.64% | skill 1/1; base 1/1 |
| claude-code | cugraph-sg-arrays | 336,270 | 311,292 | +24,978 | +8.02% | skill 1/1; base 1/1 |
| codex | All cases | 1,474,756 | 1,402,067 | +72,689 | +5.18% | skill 11/11; base 11/11 |
| codex | cugraph-direct-multi-node | 139,708 | 322,283 | -182,575 | -56.65% | skill 1/1; base 1/1 |
| codex | cugraph-empty-rank | 72,383 | 54,769 | +17,614 | +32.16% | skill 1/1; base 1/1 |
| codex | cugraph-existing-dask | 71,776 | 45,217 | +26,559 | +58.74% | skill 1/1; base 1/1 |
| codex | cugraph-gnn-default | 69,419 | 30,443 | +38,976 | +128.03% | skill 1/1; base 1/1 |
| codex | cugraph-gnn-distributed | 67,040 | 27,185 | +39,855 | +146.61% | skill 1/1; base 1/1 |
| codex | cugraph-local-spawn | 285,866 | 480,400 | -194,534 | -40.49% | skill 1/1; base 1/1 |
| codex | cugraph-memory-speed | 36,638 | 13,388 | +23,250 | +173.66% | skill 1/1; base 1/1 |
| codex | cugraph-negative | 13,290 | 13,217 | +73 | +0.55% | skill 1/1; base 1/1 |
| codex | cugraph-networkx-migration | 332,638 | 44,323 | +288,315 | +650.49% | skill 1/1; base 1/1 |
| codex | cugraph-scorer-semantics | 123,845 | 131,644 | -7,799 | -5.92% | skill 1/1; base 1/1 |
| codex | cugraph-sg-arrays | 262,153 | 239,198 | +22,955 | +9.60% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,461,050 | 3,050,931 | +410,119 | +13.44% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 27 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/accelerated-computing-cugraph/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/accelerated-computing-cugraph/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/accelerated-computing-cugraph/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in distributed-dask.md (`skills/accelerated-computing-cugraph/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/accelerated-computing-cugraph/SKILL.md`)
- 22 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.

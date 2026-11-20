<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# cuGraph skill benchmark

## Status and provenance

A single-GPU PageRank benchmark was [reported by 9Olive on 2026-11-19](https://github.com/NVIDIA/skills/pull/888#issuecomment-7014482688) during review of NVIDIA/skills#888. The report is reproduced below with attribution; its author labels it "Written by Codex." These measurements were not rerun during this PR move.

**NVSkills publication verdict: pending.** The report does not provide a full NVSkills evaluation verdict or signature. It also does not identify the evaluated commit, exact model/package versions, raw traces, or reproducible benchmark commands. Do not infer that it validates the modified submission or its distributed scripts.

## Report by 9Olive

### Task performed

This PageRank task was used to benchmark the cuGraph skill.

The agent wrote and ran a Python program that read a directed, unweighted edge list, built a five-vertex graph on a GPU, computed PageRank, and wrote one score per vertex using the original vertex IDs.

### Method

Codex and Claude each ran five paired trials with the cuGraph repo alone and with the repo plus the skill, for 20 runs on the same prepared A100 environment.
An independent CPU reference checked the supplied graph, reordered edges, and a graph with a vertex that had no outgoing edges.

### Results

Elapsed time includes agent work and independent verification.
Paired slowdowns are medians of within-pair differences.

| Agent | Repo only, median | Repo + skill, median | Median paired slowdown | Correct runs, repo / skill |
|---|---:|---:|---:|---:|
| Codex | 56.9 s | 68.7 s | 12.9 s | 5/5 / 5/5 |
| Claude | 28.5 s | 30.2 s | 1.6 s | 5/5 / 5/5 |

On this PageRank task, all 60 fixture checks passed.
The skill eliminated Claude’s graph-storage performance warning (5/5 runs to 0/5).
Codex already used the recommended setting.
The slowdown occurred primarily before submission, where the traces show extra reference reads and diagnostic checks.

*Written by Codex.*

## Interpretation and remaining validation

Both agents achieved the same correctness with and without the skill on this task. The skill removed Claude's graph-storage warning, but added elapsed time for both agents. The reported paired slowdowns are medians of within-pair differences, not differences between the aggregate medians.

This is an agent-task timing benchmark on a small graph, not a graph-kernel speedup measurement. It does not establish multi-GPU/multi-node correctness, scaling, or performance, nor coverage of the eleven skill evaluation cases. Raw evidence and a full NVSkills run on the final content are still required before a publication PASS or signing. Preserve this report as attributed evidence when replacing the benchmark with the final generated NVSkills artifact.

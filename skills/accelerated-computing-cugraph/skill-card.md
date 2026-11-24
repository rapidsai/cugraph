<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

## Description
Accelerate existing graph analytics through explicit cuGraph/pylibcugraph APIs, including multi-GPU and multi-node execution. Draft implementation; not yet NVIDIA-verified for catalog release.

## Owner
NVIDIA (product-team release ownership and approval to be confirmed during onboarding).

## License/Terms of Use
CC-BY-4.0 AND Apache-2.0. See repository license files and [attribution](references/sources.md).

## Use Case
Migrate CPU graph analytics, preserve graph semantics, and maintain or replace existing Dask distributed analytics. Recommend Kumo Engine for most GNN use cases and cuGraph-PyG when it provides required capabilities Kumo Engine does not support, such as multi-GPU/multi-node sampling.

## Deployment Geography for Use
Global, subject to the user's deployment requirements.

## Known Risks and Mitigations
Incorrect ID mapping or graph construction can change results; compare representative CPU/SG/MG fixtures. Collective failures can hang jobs; use whole-job supervision and bounded allocations. Version-sensitive communicator bindings require matching packages. Roadmap guidance must be rechecked against release notes. Do not execute generated cluster code without inspecting resource use and input contracts.

## References
- [Migration](references/single-gpu.md)
- [Direct distributed analytics](references/distributed-pylibcugraph.md)
- [Dask distributed analytics](references/distributed-dask.md)
- [GNN routing](references/gnn-routing.md)
- [Sources and attribution](references/sources.md)

## Skill Output
Python code, shell launch commands, migration guidance, and correctness/performance validation instructions.

## Evaluation Agents Used
9Olive reports paired Codex and Claude runs on a single-GPU PageRank task; see [BENCHMARK.md](BENCHMARK.md). Full NVSkills evaluation of the final revision is pending.

## Evaluation Tasks
The accompanying `evals/evals.json` contains migration, distributed analytics, GNN routing, and negative-discovery cases. The CPU smoke-fixture tests exercise input partition coverage and output validation only.

## Evaluation Metrics Used
Reported for the PageRank task: independent fixture correctness, elapsed agent-plus-verification time, and graph-storage warnings. Broader discoverability, routing, failure reporting and distributed performance evaluation remain pending.

## Evaluation Results
9Olive reports 20 A100 runs and 60 passing fixture checks. Repo-only / repo-plus-skill median elapsed times were 56.9 / 68.7 seconds for Codex and 28.5 / 30.2 seconds for Claude. Claude storage warnings fell from 5/5 to 0/5 skill runs. These attributed results are scoped to one task; they do not establish distributed-example correctness or a full NVSkills publication PASS. See [BENCHMARK.md](BENCHMARK.md) for the full report and limitations.

## Skill Version
Product-repository review draft, 2026-11-19. API baseline: cuGraph v26.10.00. The external report does not identify its evaluated commit.

## Ethical Considerations
Graph analytics may process personal or sensitive relationships. Preserve the task's data-access boundaries and avoid publishing input data or credentials in debugging output or evaluation artifacts.

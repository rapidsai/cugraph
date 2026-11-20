<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# cuGraph skill development and catalog handoff

The product-owned skill lives at `skills/accelerated-computing-cugraph/`. This submission replaces [NVIDIA/skills#888](https://github.com/NVIDIA/skills/pull/888), following the [reviewer's source-repository guidance](https://github.com/NVIDIA/skills/pull/888#issuecomment-6980433165). NVIDIA/skills is a catalog that synchronizes signed skills from public product repositories; its eventual onboarding PR should contain only the component registration.

## Evaluation evidence

[BENCHMARK.md](../../skills/accelerated-computing-cugraph/BENCHMARK.md) preserves 9Olive's reported single-GPU PageRank analysis, with source attribution and limitations. Both agents passed all task checks; the skill removed Claude's storage warning but increased elapsed time. These results are not a full NVSkills PASS and do not validate the distributed examples. The [validation record](VALIDATION.md) separates local checks from reported GPU evidence.

## Before catalog onboarding

1. Confirm owning-team release clearance, roadmap statements, licensing and dependency review. The documentation retains its CC-BY-4.0 attribution and code retains Apache-2.0 licensing; do not interpret this submission as approval of a new license for the whole cuGraph repository.
2. Complete Linux GPU validation: single-GPU semantics, 2-GPU direct and Dask fixtures, 2-node/4-GPU fixtures, empty partitions and coordinated failure handling. Add a nonuniform weighted graph with dangling/isolated vertices and compare CPU/SG/MG results. Record package locks, GPU/network hardware, commands, logs and tolerances.
3. Use `.github/workflows/request-nvskills-ci.yml`, adapted from the official [request template](https://github.com/NVIDIA/skills/blob/main/.github/workflows/request-nvskills-ci.yml). The NVSkills maintainers must enable this source repository and fork support; follow the current [contribution instructions](https://github.com/NVIDIA/skills/blob/main/CONTRIBUTING.md) for service-account access and the maintainer `/nvskills-ci` trigger. The reusable workflow is pinned to NVIDIA/skills commit `b0694f6d55b4037ffd414aba5759b59f38380c5e` to meet cuGraph's workflow policy. This PR does not create secrets or grant collaborator access. An issue-comment workflow must exist on the default branch to receive comments, so a newly added workflow may require maintainer bootstrap before that trigger works.
4. Run the eleven agent evaluation cases through NVSkills. Update the card and generated benchmark with measured results, then obtain a real `skill.oms.sig` over the final files. Do not reuse signatures or claim a publication PASS from the narrow external benchmark.
5. Once this source is published and signed, open a separate NVIDIA/skills PR adding only `components.d/cugraph.yml`, with `repo: rapidsai/cugraph`, the approved ref, `path: skills/accelerated-computing-cugraph/`, and `catalog_dir: accelerated-computing-cugraph`. Ensure the product mapping matches the catalog schema. Let catalog automation generate README, marketplace and plugin content.

The source workflow is the standard request template, not the full NVSkills evaluation implementation. Maintainers own enablement and final publication approvals.

## Local validation

From the cuGraph repository root:

```bash
python3 -m unittest discover -s docs/cugraph-development/tests -v
python3 -m compileall -q skills/accelerated-computing-cugraph/scripts
python3 skills/accelerated-computing-cugraph/scripts/pagerank_mg.py --help
python3 skills/accelerated-computing-cugraph/scripts/pagerank_dask.py --help
```

Run cuGraph's pre-commit hooks on changed files before submission. The skill-creator `quick_validate.py` can also check frontmatter when available. Local CPU checks do not validate CUDA, NCCL or distributed runtime behavior; follow both distributed references for hardware checks.

## Scope

Explicit cuGraph/pylibcugraph analytics, Kumo Engine routing for GNNs with a distributed-training exception, and transitional Dask migration. The walkthroughs remain scoped to the 26.10 source API, including the optional-edge-ID MGGraph constructor workaround. No cuGraph runtime implementation is changed by this PR.

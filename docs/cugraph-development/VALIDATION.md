<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Validation record — 2026-11-06

Base NVIDIA/skills commit: `c663e5799366d945a70a40b1ab18cedfc817495c`.
API source: cuGraph v26.10.00, `8ff922fabf4521cf540a7ac3cfae857f7edc6f10`.

Passed locally on macOS (CPU only):

- Four unittest cases: disjoint fixture coverage at 1/2/4/16 ranks including empty ranks; order-independent correct result acceptance; missing/duplicate/nonfinite/wrong-result rejection; existing input directory protection.
- All three Python scripts compile; both GPU runners expose help without importing GPU packages.
- Python code blocks parse; all local Markdown links resolve.
- Eleven evaluation records parse and have unique IDs (ten positive, one negative).
- Skill frontmatter validation and repository skill-card unresolved-marker check.
- Whitespace/diff checks.

Not run: Linux CUDA execution, SG/MG numerical equivalence, Dask runtime, NCCL transport, multi-node launch, failure injection, performance measurement, agent effectiveness benchmark, or NVSkills signing. The cycle fixture checker was tested with synthetic expected outputs; this is not a claim that GPU PageRank produced those outputs.

Initial validation encountered missing PyYAML and a system Python cache path outside the writable workspace. PyYAML was installed in an isolated workspace venv; compilation was rerun with a temporary cache prefix. No system Python configuration was changed.

At the time of this initial validation, source placement, release clearance, measured benchmark PASS and a final signed manifest were pending. See the later entries for updated evidence. See README.md for the handoff.

## Draft PR preparation — 2026-11-17

Rebased the implementation onto NVIDIA/skills `3e2ed2d8` without changing the skill or GPU examples. Reran the four CPU tests, script compilation, frontmatter validation, skill-card marker check, local link and Python code-block checks, and evaluation JSON validation. All passed. GPU and agent evaluation status remains unchanged. Updated onboarding notes for the current product metadata requirement and the public review draft.

## Product-repository submission — 2026-11-19

Moved the skill and CPU fixture tests to rapidsai/cugraph main at `420baac7b191dba4ed04817c7f662553e2f7d0c0`, following [the catalog review](https://github.com/NVIDIA/skills/pull/888#issuecomment-6980433165). Removed private-source references and added the standard NVSkills CI request workflow. The original examples remain scoped to the 26.10 API.

[9Olive's reported benchmark](https://github.com/NVIDIA/skills/pull/888#issuecomment-7014482688) now appears in the skill's BENCHMARK.md, with attribution and the distinction between that single-GPU task and untested distributed examples. These are externally reported results, not a new run performed during this move. Raw traces, exact environment/model versions, and the evaluated commit are not attached to the source comment. Full NVSkills validation/signing and multi-GPU/multi-node testing remain pending.

Local rerun for the product-repository submission: all four CPU tests, script compilation, skill frontmatter, Markdown links, Python examples, evaluation JSON, private-source-reference scan, and all applicable cuGraph pre-commit hooks passed. The hooks normalized copyright headers and formatting; the NVSkills reusable workflow was SHA-pinned as required by cuGraph's zizmor policy. No new GPU benchmark or distributed integration run was performed.

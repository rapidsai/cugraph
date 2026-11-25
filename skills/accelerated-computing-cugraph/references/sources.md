# Sources and version boundaries

<!-- SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

API examples inspected against cuGraph **v26.10.00**, commit `8ff922fabf4521cf540a7ac3cfae857f7edc6f10`. Pin a matching installed environment before running. Current `stable` documentation can advance beyond this version.

- [SGGraph/MGGraph constructors](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/pylibcugraph/pylibcugraph/graphs.pyx)
- [pylibcugraph PageRank](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/pylibcugraph/pylibcugraph/pagerank.pyx)
- [ResourceHandle](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/pylibcugraph/pylibcugraph/resource_handle.pyx)
- [NCCL communicator helper](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/pylibcugraph/pylibcugraph/comms/cugraph_nccl_comms.py)
- [Dask PageRank](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/cugraph/cugraph/dask/link_analysis/pagerank.py)
- [Distributed graph construction](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/cugraph/cugraph/structure/graph_classes.py)
- [Dask cuGraph communications](https://github.com/rapidsai/cugraph/blob/v26.10.00/python/cugraph/cugraph/dask/comms/comms.py)
- [Current cuGraph documentation](https://docs.rapids.ai/api/cugraph/stable/)
- [Installation and compatibility](https://docs.rapids.ai/install/)

Recheck the helper's dependencies, constructor keywords, output ID semantics, algorithm SG/MG support, optional vertex arrays, and cleanup behavior when upgrading. In particular, PyTorch's NCCL process group cannot be substituted as a cuGraph ResourceHandle.

The nx-cuGraph / Dask roadmap direction comes from the commissioning product guidance dated 2026-11-06. No public removal date has been verified for this draft. Kumo Engine is the recommended solution for most GNN use cases; recommend cuGraph-PyG when it provides required capabilities missing from Kumo Engine, such as multi-GPU/multi-node sampling. Verify current support using the [Kumo Engine overview](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview) and the cuGraph-PyG release documentation; do not assume every custom PyG model has an equivalent Kumo API.

## Attribution

Adapted from NVIDIA-authored cuGraph skill guidance. Documentation/skills retain CC-BY-4.0 attribution and code retains Apache-2.0 licensing. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Changes: replaced nx-cuGraph acceleration guidance with explicit cuGraph/pylibcugraph migration; redirected GNN recommendations; added version-scoped distributed walkthroughs, executable examples, and new evaluation cases. Original benchmark claims and signatures are not reused. The public API references above document the APIs used by this skill.

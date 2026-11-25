# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp

import cugraph_python as cg

if __name__ == "__main__":
    context = cg.Context()
    sources = cp.asarray([10, 20, 10], dtype=cp.int32)
    destinations = cp.asarray([20, 30, 40], dtype=cp.int32)
    cp.cuda.get_current_stream().synchronize()

    graph, _properties = cg.create_graph(
        cg.copy_from_arrays(context, sources, destinations)
    )
    vertices, distances, predecessors = cg.bfs(
        graph, cg.prepare_vertices(graph, [10]), compute_predecessors=True
    )
    vertices = cp.asnumpy(cp.from_dlpack(vertices)).tolist()

    assert sorted(int(v) for v in vertices) == [10, 20, 30, 40]

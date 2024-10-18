# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy

from pylibcugraph import (
    SGGraph,
    ResourceHandle,
    GraphProperties,
    EdgeIdLookupTable,
)


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================


def test_lookup_table():
    # Vertex id array
    vtcs = cupy.arange(6, dtype="int64")

    # Edge ids are unique per edge type and start from 0
    # Each edge type has the same src/dst vertex type here,
    # just as it would in a GNN application.
    srcs = cupy.array([0, 1, 5, 4, 3, 2, 2, 0, 5, 4, 4, 5])
    dsts = cupy.array([1, 5, 0, 3, 2, 1, 3, 3, 2, 3, 1, 4])
    etps = cupy.array([0, 2, 6, 7, 4, 3, 4, 1, 7, 7, 6, 8], dtype="int32")
    eids = cupy.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0])

    wgts = cupy.ones((len(srcs),), dtype="float32")

    graph = SGGraph(
        resource_handle=ResourceHandle(),
        graph_properties=GraphProperties(is_symmetric=False, is_multigraph=True),
        src_or_offset_array=srcs,
        dst_or_index_array=dsts,
        vertices_array=vtcs,
        weight_array=wgts,
        edge_id_array=eids,
        edge_type_array=etps,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
    )

    table = EdgeIdLookupTable(ResourceHandle(), graph)

    assert table is not None

    found_edges = table.lookup_vertex_ids(cupy.array([0, 1, 2, 3, 4]), 7)
    assert (found_edges["sources"] == cupy.array([4, 5, 4, -1, -1])).all()
    assert (found_edges["destinations"] == cupy.array([3, 2, 3, -1, -1])).all()

    found_edges = table.lookup_vertex_ids(cupy.array([0]), 5)
    assert (found_edges["sources"] == cupy.array([-1])).all()
    assert (found_edges["destinations"] == cupy.array([-1])).all()

    found_edges = table.lookup_vertex_ids(cupy.array([3, 1, 0, 5]), 6)
    assert (found_edges["sources"] == cupy.array([-1, 4, 5, -1])).all()
    assert (found_edges["destinations"] == cupy.array([-1, 1, 0, -1])).all()

    # call __dealloc__()
    del table

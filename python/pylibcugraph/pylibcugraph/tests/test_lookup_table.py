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
import numpy as np

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

    vtcs = cupy.arange(6, dtype="int64")
    vtps = np.array([0, 0, 1, 1, 2, 2])

    e_lookup = {
        (0, 0): [0, 0],
        (0, 1): [1, 0],
        (0, 2): [2, 0],
        (1, 0): [3, 0],
        (1, 1): [4, 0],
        (1, 2): [5, 0],
        (2, 0): [6, 0],
        (2, 1): [7, 0],
        (2, 2): [8, 0],
    }

    srcs = np.array([0, 1, 5, 4, 3, 2, 2, 0, 5, 4, 4, 5])
    dsts = np.array([1, 5, 0, 3, 2, 1, 3, 3, 2, 3, 1, 4])
    wgts = cupy.ones((len(srcs),), dtype="float32")

    eids = []
    etps = []
    for i in range(len(srcs)):
        key = (int(vtps[srcs[i]]), int(vtps[dsts[i]]))
        etps.append(e_lookup[key][0])
        eids.append(e_lookup[key][1])

        e_lookup[key][1] += 1

    eids = cupy.array(eids)
    etps = cupy.array(etps, dtype="int32")

    graph = SGGraph(
        resource_handle=ResourceHandle(),
        graph_properties=GraphProperties(is_symmetric=False, is_multigraph=True),
        src_or_offset_array=cupy.array(srcs),
        dst_or_index_array=cupy.array(dsts),
        vertices_array=vtcs,
        weight_array=wgts,
        edge_id_array=eids,
        edge_type_array=etps,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
    )

    # call __dealloc__()
    table = EdgeIdLookupTable(ResourceHandle(), graph)

    assert table is not None

    found_edges = table.find(cupy.array([0, 1, 2, 3, 4]), 7)
    assert (found_edges["sources"] == cupy.array([4, 5, 4, -1, -1])).all()
    assert (found_edges["destinations"] == cupy.array([3, 2, 3, -1, -1])).all()

    found_edges = table.find(cupy.array([0]), 5)
    assert (found_edges["sources"] == cupy.array([-1])).all()
    assert (found_edges["destinations"] == cupy.array([-1])).all()

    found_edges = table.find(cupy.array([3, 1, 0, 5]), 6)
    assert (found_edges["sources"] == cupy.array([-1, 4, 5, -1])).all()
    assert (found_edges["destinations"] == cupy.array([-1, 1, 0, -1])).all()

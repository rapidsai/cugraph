# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import pytest


# =============================================================================
# Pytest fixtures
# =============================================================================
# fixtures used in this test module are defined in conftest.py


# =============================================================================
# Tests
# =============================================================================
def test_graph_properties():
    from pylibcugraph import GraphProperties

    gp = GraphProperties()
    assert gp.is_symmetric is False
    assert gp.is_multigraph is False

    gp.is_symmetric = True
    assert gp.is_symmetric is True
    gp.is_symmetric = 0
    assert gp.is_symmetric is False
    with pytest.raises(TypeError):
        gp.is_symmetric = "foo"

    gp.is_multigraph = True
    assert gp.is_multigraph is True
    gp.is_multigraph = 0
    assert gp.is_multigraph is False
    with pytest.raises(TypeError):
        gp.is_multigraph = "foo"

    gp = GraphProperties(is_symmetric=True, is_multigraph=True)
    assert gp.is_symmetric is True
    assert gp.is_multigraph is True

    gp = GraphProperties(is_multigraph=True, is_symmetric=False)
    assert gp.is_symmetric is False
    assert gp.is_multigraph is True

    with pytest.raises(TypeError):
        gp = GraphProperties(is_symmetric="foo", is_multigraph=False)

    with pytest.raises(TypeError):
        gp = GraphProperties(is_multigraph=[])


def test_resource_handle():
    from pylibcugraph import ResourceHandle

    # This type has no attributes and is just defined to pass a struct from C
    # back in to C. In the future it may take args to acquire specific
    # resources, but for now just make sure nothing crashes.
    rh = ResourceHandle()
    del rh


def test_sg_graph(graph_data):
    from pylibcugraph import (
        SGGraph,
        ResourceHandle,
        GraphProperties,
    )

    # is_valid will only be True if the arrays are expected to produce a valid
    # graph. If False, ensure SGGraph() raises the proper exception.
    (device_srcs, device_dsts, device_weights, ds_name, is_valid) = graph_data

    graph_props = GraphProperties(is_symmetric=False, is_multigraph=False)
    resource_handle = ResourceHandle()

    if is_valid:
        g = SGGraph(  # noqa:F841
            resource_handle,
            graph_props,
            device_srcs,
            device_dsts,
            device_weights,
            store_transposed=False,
            renumber=False,
            do_expensive_check=False,
        )
        # call SGGraph.__dealloc__()
        del g

    else:
        with pytest.raises(ValueError):
            SGGraph(
                resource_handle,
                graph_props,
                device_srcs,
                device_dsts,
                device_weights,
                store_transposed=False,
                renumber=False,
                do_expensive_check=False,
            )


def test_SGGraph_create_from_cudf():
    """
    Smoke test to ensure an SGGraph can be created from a cuDF DataFrame
    without raising exceptions, crashing, etc. This currently does not assert
    correctness of the graph in any way.
    """
    # FIXME: other PLC tests are using cudf so this does not add a new dependency,
    # however, PLC tests should consider having fewer external dependencies, meaning
    # this and other tests would be changed to not use cudf.
    import cudf

    # Importing this cugraph class seems to cause a crash more reliably (2023-01-22)
    # from cugraph.structure.graph_implementation import simpleGraphImpl
    from pylibcugraph import (
        ResourceHandle,
        GraphProperties,
        SGGraph,
    )

    print("get edgelist...", end="", flush=True)
    edgelist = cudf.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [1, 2, 4],
            "wgt": [0.0, 0.1, 0.2],
        }
    )

    print("edgelist = ", edgelist)
    print("done", flush=True)
    print("create Graph...", end="", flush=True)

    graph_props = GraphProperties(is_multigraph=False, is_symmetric=False)

    plc_graph = SGGraph(
        resource_handle=ResourceHandle(),
        graph_properties=graph_props,
        src_or_offset_array=edgelist["src"],
        dst_or_index_array=edgelist["dst"],
        weight_array=edgelist["wgt"],
        edge_id_array=None,
        edge_type_array=None,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
        input_array_format="COO",
    )
    print("done", flush=True)
    print(f"created SGGraph {plc_graph=}", flush=True)

# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#
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

import os
from pathlib import Path
import pickle

import pytest

from . import data


###############################################################################
# fixtures


@pytest.fixture(scope="module")
def mg_handler():
    """
    Creates a cugraph_service handler that uses a dask client.
    """
    from cugraph_service_server.cugraph_handler import CugraphHandler

    handler = CugraphHandler()
    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")

    if dask_scheduler_file is None:
        handler.initialize_dask_client()

    else:
        dask_scheduler_file = Path(dask_scheduler_file)
        if not dask_scheduler_file.exists():
            raise FileNotFoundError(
                "env var SCHEDULER_FILE is set to "
                f"{dask_scheduler_file}, which does not "
                "exist."
            )

        handler.initialize_dask_client(dask_scheduler_file)

    return handler


# Make this a function-level fixture so it cleans up the mg_handler after each
# test, allowing other tests to use mg_handler without graphs loaded.
@pytest.fixture(scope="function")
def handler_with_karate_edgelist_loaded(mg_handler):
    """
    Loads the karate CSV into the default graph in the handler.
    """
    from cugraph_service_client import defaults

    test_data = data.edgelist_csv_data["karate"]

    # Ensure the handler starts with no graphs in memory
    for gid in mg_handler.get_graph_ids():
        mg_handler.delete_graph(gid)

    mg_handler.load_csv_as_edge_data(
        test_data["csv_file_name"],
        delimiter=" ",
        dtypes=test_data["dtypes"],
        header=None,
        vertex_col_names=["0", "1"],
        type_name="",
        property_columns=[],
        names=[],
        edge_id_col_name="",
        graph_id=defaults.graph_id,
    )
    assert mg_handler.get_graph_ids() == [0]

    yield (mg_handler, test_data)

    for gid in mg_handler.get_graph_ids():
        mg_handler.delete_graph(gid)


###############################################################################
# tests

# FIXME: consolidate this with the SG version of this test.
def test_get_graph_data_large_vertex_ids(
    mg_handler,
    graph_creation_extension_big_vertex_ids,
):
    """
    Test that graphs with large vertex ID values (>int32) are handled.
    """
    handler = mg_handler
    extension_dir = graph_creation_extension_big_vertex_ids

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function_vert_and_edge_data_big_vertex_ids", "()", "{}"
    )

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 0

    large_vert_id = (2**32) + 1
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=large_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 1

    invalid_edge_id = (2**32) + 1
    edge_data = handler.get_graph_edge_data(
        id_or_ids=invalid_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 0

    small_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=small_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 1


# FIXME: consolidate this with the SG version of this test.
def test_get_graph_data_empty_graph(
    mg_handler,
    graph_creation_extension_empty_graph,
):
    """
    Tests that get_graph_*_data() handles empty graphs correctly.
    """
    handler = mg_handler
    extension_dir = graph_creation_extension_empty_graph

    # Load the extension and ensure it can be called.
    handler.load_graph_creation_extensions(extension_dir)
    new_graph_id = handler.call_graph_creation_extension(
        "graph_creation_function", "()", "{}"
    )

    invalid_vert_id = 2
    vert_data = handler.get_graph_vertex_data(
        id_or_ids=invalid_vert_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(vert_data)) == 0

    invalid_edge_id = 2
    edge_data = handler.get_graph_edge_data(
        id_or_ids=invalid_edge_id,
        null_replacement_value=0,
        property_keys=None,
        types=None,
        graph_id=new_graph_id,
    )

    assert len(pickle.loads(edge_data)) == 0


def test_get_edge_IDs_for_vertices(handler_with_karate_edgelist_loaded):
    from cugraph_service_client import defaults

    (handler, test_data) = handler_with_karate_edgelist_loaded

    # Use the test/debug API to ensure the correct type was created
    assert "MG" in handler.get_graph_type(defaults.graph_id)

    extracted_graph_id = handler.extract_subgraph(
        create_using=None,
        selection=None,
        edge_weight_property=None,
        default_edge_weight=1.0,
        check_multi_edges=True,
        renumber_graph=True,
        add_edge_data=True,
        graph_id=defaults.graph_id,
    )

    # FIXME: this assumes these are always the first 3 edges in karate, which
    # may not be a safe assumption.
    eIDs = handler.get_edge_IDs_for_vertices([1, 2, 3], [0, 0, 0], extracted_graph_id)
    assert eIDs == [0, 1, 2]


def test_get_graph_info(handler_with_karate_edgelist_loaded):
    """
    get_graph_info() for specific args.
    """
    from cugraph_service_client import defaults

    (handler, test_data) = handler_with_karate_edgelist_loaded

    # A common use of get_graph_info() is to get the "shape" of the data,
    # meaning the number of vertices/edges by the number of properites per
    # edge/vertex.
    info = handler.get_graph_info(
        ["num_edges", "num_edge_properties"], defaults.graph_id
    )
    # info is a dictionary containing cugraph_service_client.types.ValueWrapper
    # objs, so access the int32 member directly for easy comparison.
    shape = (
        info["num_edges"].get_py_obj(),
        info["num_edge_properties"].get_py_obj(),
    )
    assert shape == (156, 1)  # The single edge property is the weight

    info = handler.get_graph_info(
        ["num_vertices_from_vertex_data", "num_vertex_properties"], defaults.graph_id
    )
    shape = (
        info["num_vertices_from_vertex_data"].get_py_obj(),
        info["num_vertex_properties"].get_py_obj(),
    )
    assert shape == (0, 0)


def test_get_graph_info_defaults(mg_handler):
    """
    Ensure calling get_graph_info() with no args returns the info dict with all
    keys present for an empty default graph.
    """
    from cugraph_service_client import defaults

    handler = mg_handler

    info = handler.get_graph_info([], graph_id=defaults.graph_id)

    expected = {
        "is_multi_gpu": True,
        "num_vertices": 0,
        "num_vertices_from_vertex_data": 0,
        "num_edges": 0,
        "num_vertex_properties": 0,
        "num_edge_properties": 0,
    }
    actual = {key: val.get_py_obj() for (key, val) in info.items()}

    assert expected == actual


def test_uniform_neighbor_sampling(handler_with_karate_edgelist_loaded):
    from cugraph_service_client import defaults

    (handler, test_data) = handler_with_karate_edgelist_loaded

    start_list = [1, 2, 3]
    fanout_vals = [2, 2, 2]
    with_replacement = True

    # FIXME: add test coverage for specifying the edge ID as the
    # edge_weight_property, then ensuring the edge ID is returned properly with
    # the uniform_neighbor_sample results.
    # See: https://github.com/rapidsai/cugraph/issues/2654
    extracted_gid = handler.extract_subgraph(
        create_using=None,
        selection=None,
        edge_weight_property=None,
        default_edge_weight=1.0,
        check_multi_edges=True,
        renumber_graph=True,
        add_edge_data=True,
        graph_id=defaults.graph_id,
    )

    # Ensure call can be made, assume results verified in other tests
    handler.uniform_neighbor_sample(
        start_list=start_list,
        fanout_vals=fanout_vals,
        with_replacement=with_replacement,
        graph_id=extracted_gid,
        result_host=None,
        result_port=None,
    )

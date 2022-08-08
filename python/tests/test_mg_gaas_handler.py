# Copyright (c) 2022, NVIDIA CORPORATION.
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

import pytest

from . import data

###############################################################################
## fixtures

@pytest.fixture(scope="module")
def mg_handler():
    """
    Creates a GaaS handler that uses a dask client.
    """
    from gaas_server.gaas_handler import GaasHandler

    dask_scheduler_file = os.environ.get("SCHEDULER_FILE")
    if dask_scheduler_file is None:
        raise EnvironmentError("Environment variable SCHEDULER_FILE must be set"
                               "to the path to a dask scheduler json file")
    dask_scheduler_file = Path(dask_scheduler_file)
    if not dask_scheduler_file.exists():
        raise FileNotFoundError("env var SCHEDULER_FILE is set to "
                                f"{dask_scheduler_file}, which does not exist.")

    handler = GaasHandler()
    handler.initialize_dask_client(dask_scheduler_file)
    return handler


# Make this a function-level fixture so it cleans up the mg_handler after each
# test, allowing other tests to use mg_handler without graphs loaded.
@pytest.fixture(scope="function")
def handler_with_edgelist_csv_loaded(mg_handler):
    """
    Loads the karate CSV into the default graph in the handler.
    """
    from gaas_client import defaults

    test_data = data.edgelist_csv_data["karate"]

    mg_handler.load_csv_as_edge_data(test_data["csv_file_name"],
                                     delimiter=" ",
                                     dtypes=test_data["dtypes"],
                                     header=None,
                                     vertex_col_names=["0", "1"],
                                     type_name="",
                                     property_columns=[],
                                     names=[],
                                     graph_id=defaults.graph_id,
                                     )
    assert mg_handler.get_graph_ids() == [0]

    yield (mg_handler, test_data)

    for gid in mg_handler.get_graph_ids():
        mg_handler.delete_graph(gid)


###############################################################################
## tests

def test_get_edge_IDs_for_vertices(handler_with_edgelist_csv_loaded):
    """
    """
    from gaas_client import defaults

    (handler, test_data) = handler_with_edgelist_csv_loaded

    # Use the test/debug API to ensure the correct type was created
    assert "MG" in handler.get_graph_type(defaults.graph_id)

    extracted_graph_id = handler.extract_subgraph(create_using=None,
                                                  selection=None,
                                                  edge_weight_property="",
                                                  default_edge_weight=1.0,
                                                  allow_multi_edges=True,
                                                  graph_id=defaults.graph_id,
                                                  )

    # FIXME: this assumes these are always the first 3 edges in karate, which
    # may not be a safe assumption.
    eIDs = handler.get_edge_IDs_for_vertices([1, 2, 3],
                                             [0, 0, 0],
                                             extracted_graph_id)
    assert eIDs == [0, 1, 2]


def test_get_graph_edge_dataframe_shape(handler_with_edgelist_csv_loaded):
    """
    """
    from gaas_client import defaults
    from gaas_client.types import ValueWrapper

    (handler, test_data) = handler_with_edgelist_csv_loaded

    info = handler.get_graph_info(["num_edges", "num_edge_properties"],
                                  defaults.graph_id)
    # info is a dictionary containing gaas_client.types.Value objs, so access
    # the int32 member directly for easy comparison.
    shape = (ValueWrapper(info["num_edges"]).get_py_obj(),
             ValueWrapper(info["num_edge_properties"]).get_py_obj())
    assert shape == (156, 3)


def test_get_graph_vertex_dataframe_shape(handler_with_edgelist_csv_loaded):
    """
    A graph created from an edgelist CSV will not have vertex data, so ensure
    the shape is returned correctly.
    """
    from gaas_client import defaults
    from gaas_client.types import ValueWrapper

    (handler, test_data) = handler_with_edgelist_csv_loaded

    info = handler.get_graph_info(["num_vertices",
                                   "num_vertex_properties"],
                                  defaults.graph_id)
    # info is a dictionary containing gaas_client.types.Value objs, so access
    # the int32 member directly for easy comparison.
    shape = (ValueWrapper(info["num_vertices"]).get_py_obj(),
             ValueWrapper(info["num_vertex_properties"]).get_py_obj())
    assert shape == (0, 0)


def test_get_graph_info_defaults(mg_handler):
    """
    Ensure calling get_graph_info() with no args returns the info dict with all
    keys present for an empty default graph.
    """
    from gaas_client import defaults
    from gaas_client.types import ValueWrapper

    handler = mg_handler

    info = handler.get_graph_info([], graph_id=defaults.graph_id)

    expected = {"num_vertices": 0,
                "num_edges": 0,
                "num_vertex_properties": 0,
                "num_edge_properties": 0,
                }
    actual = {key:ValueWrapper(val).get_py_obj() for (key, val) in info.items()}

    assert expected == actual

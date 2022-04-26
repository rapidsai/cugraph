# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import time
import gc
import cugraph.dask as dcg
import dask_cudf
import pytest
import pandas as pd
import cudf
from cudf.testing import assert_frame_equal, assert_series_equal
from cugraph.tests.utils import RAPIDS_DATASET_ROOT_DIR_PATH

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.

import cugraph
from cugraph.tests import utils


# Placeholder for a directed Graph instance. This is not constructed here in
# order to prevent cuGraph code from running on import, which would prevent
# proper pytest collection if an exception is raised. See setup_function().
DiGraph_inst = None


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    global DiGraph_inst

    gc.collect()
    # Set the global DiGraph_inst. This is used for calls that require a Graph
    # type or instance to be provided for tests that use a directed graph.
    DiGraph_inst = cugraph.Graph(directed=True)  # noqa: F841


# =============================================================================
# Pytest fixtures
# =============================================================================
df_types = [dask_cudf.core.DataFrame]


def df_type_id(dataframe_type):
    """
    Return a string that describes the dataframe_type, used for test output.
    """
    s = "df_type="
    if dataframe_type == cudf.DataFrame:
        return s+"cudf.DataFrame"
    if dataframe_type == pd.DataFrame:
        return s+"pandas.DataFrame"
    if dataframe_type == dask_cudf.core.DataFrame:
        return s+"dask_cudf.core.DataFrame"
    return s+"?"

df_types_fixture_params = utils.genFixtureParamsProduct((df_types, df_type_id))


@pytest.fixture(scope="module", params=df_types_fixture_params)
def net_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the netscience.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import PropertyGraph

    dataframe_type = request.param[0]
    netscience_csv = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"netscience.csv"
    source_col_name = "srcip"
    dest_col_name = "dstip"

    if dataframe_type is pd.DataFrame:
        read_csv = pd.read_csv
    else:
        read_csv = cudf.read_csv
    df = read_csv(netscience_csv, delimiter=",",
                  dtype={"idx": "int32",
                         source_col_name: "str",
                         dest_col_name: "str"},
                  header=0)

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    return pG

@pytest.fixture(scope="module", params=df_types_fixture_params)
def net_Dask_PropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the netscience.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import PropertyGraph
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "netscience.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    def modify_dataset(df):
        temp_df = cudf.DataFrame()
        temp_df['src'] = df['src']+1000
        temp_df['dst'] = df['dst']+1000
        temp_df['value'] = df['value']
        return cudf.concat([df, temp_df])

    meta = ddf._meta
    ddf = ddf.map_partitions(modify_dataset, meta=meta)

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = modify_dataset(df)

    dpG = PropertyGraph()
    dpG.add_edge_data(ddf, ("src", "dst"))
    return dpG


def test_select_vertices_from_previous_selection(net_Dask_PropertyGraph):
    """
    Ensures that the intersection of vertices of multiple types (only vertices
    that are both type A and type B) can be selected.
    """
    from cugraph.experimental import PropertyGraph


def test_extract_subgraph(net_Dask_PropertyGraph):
    """
    Valid query that only matches a single vertex.
    """
    pG = net_Dask_PropertyGraph
    print(pG.num_vertices)


def test_extract_subgraph_no_query(net_Dask_PropertyGraph):
    """
    Call extract with no args, should result in the entire property graph.
    """
    pG = net_Dask_PropertyGraph
    print(pG.num_vertices)
    print(pG.num_edges)
    subGraph = pG.extract_subgraph()
    breakpoint()
    assert type(subGraph.edgelist.edgelist_df) == type(pG.edgelist.edgelist_df)
    
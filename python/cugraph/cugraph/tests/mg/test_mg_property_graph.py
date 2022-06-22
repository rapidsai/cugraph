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

import gc
import cugraph.dask as dcg
import dask_cudf
import pytest
import pandas as pd
import cudf
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH
from cugraph.testing import utils

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.

import cugraph

# =============================================================================
# Test data
# =============================================================================

dataset1 = {
    "merchants": [
        ["merchant_id", "merchant_location", "merchant_size", "merchant_sales",
         "merchant_num_employees", "merchant_name"],
        [(11, 78750, 44, 123.2, 12, "north"),
         (4, 78757, 112, 234.99, 18, "south"),
         (21, 44145, 83, 992.1, 27, "east"),
         (16, 47906, 92, 32.43, 5, "west"),
         (86, 47906, 192, 2.43, 51, "west"),
         ]
     ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [(89021, 78757, 0),
         (32431, 78750, 1),
         (89216, 78757, 1),
         (78634, 47906, 0),
         ]
     ],
    "taxpayers": [
        ["payer_id", "amount"],
        [(11, 1123.98),
         (4, 3243.7),
         (21, 8932.3),
         (16, 3241.77),
         (86, 789.2),
         (89021, 23.98),
         (78634, 41.77),
         ]
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [(89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
         (89216, 4, None, 1639085163.481217, 8832, "CASH"),
         (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
         (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
         ]
     ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [(89216, 89021, 9),
         (89216, 32431, 9),
         (32431, 78634, 8),
         (78634, 89216, 8),
         ]
     ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [(89216, 78634, 11, 5),
         (89021, 89216, 4, 4),
         (89021, 89216, 21, 3),
         (89021, 89216, 11, 3),
         (89021, 78634, 21, 4),
         (78634, 32431, 11, 4),
         ]
     ],
}


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
df_types = [cudf.DataFrame]


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
    source_col_name = "src"
    dest_col_name = "dst"

    if dataframe_type is pd.DataFrame:
        read_csv = pd.read_csv
    else:
        read_csv = cudf.read_csv
    df = read_csv(netscience_csv,
                  delimiter=" ",
                  names=["src", "dst", "value"],
                  dtype=["int32", "int32", "float32"])

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    return pG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def dataset1_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    pG.add_vertex_data(dataframe_type(columns=merchants[0],
                                      data=merchants[1]),
                       type_name="merchants",
                       vertex_col_name="merchant_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=users[0],
                                      data=users[1]),
                       type_name="users",
                       vertex_col_name="user_id",
                       property_columns=None)
    pG.add_vertex_data(dataframe_type(columns=taxpayers[0],
                                      data=taxpayers[1]),
                       type_name="taxpayers",
                       vertex_col_name="payer_id",
                       property_columns=None)

    pG.add_edge_data(dataframe_type(columns=transactions[0],
                                    data=transactions[1]),
                     type_name="transactions",
                     vertex_col_names=("user_id", "merchant_id"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=relationships[0],
                                    data=relationships[1]),
                     type_name="relationships",
                     vertex_col_names=("user_id_1", "user_id_2"),
                     property_columns=None)
    pG.add_edge_data(dataframe_type(columns=referrals[0],
                                    data=referrals[1]),
                     type_name="referrals",
                     vertex_col_names=("user_id_1",
                                       "user_id_2"),
                     property_columns=None)
    return pG


@pytest.fixture(scope="module")
def dataset1_MGPropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = cudf.DataFrame
    (merchants, users, taxpayers,
     transactions, relationships, referrals) = dataset1.values()
    from cugraph.experimental import MGPropertyGraph
    mpG = MGPropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    sg_df = dataframe_type(columns=merchants[0], data=merchants[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_vertex_data(mg_df,
                        type_name="merchants",
                        vertex_col_name="merchant_id",
                        property_columns=None)

    sg_df = dataframe_type(columns=users[0], data=users[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_vertex_data(mg_df,
                        type_name="users",
                        vertex_col_name="user_id",
                        property_columns=None)

    sg_df = dataframe_type(columns=taxpayers[0], data=taxpayers[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_vertex_data(mg_df,
                        type_name="taxpayers",
                        vertex_col_name="payer_id",
                        property_columns=None)

    sg_df = dataframe_type(columns=transactions[0], data=transactions[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(mg_df,
                      type_name="transactions",
                      vertex_col_names=("user_id", "merchant_id"),
                      property_columns=None)

    sg_df = dataframe_type(columns=relationships[0], data=relationships[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(mg_df,
                      type_name="relationships",
                      vertex_col_names=("user_id_1", "user_id_2"),
                      property_columns=None)

    sg_df = dataframe_type(columns=referrals[0], data=referrals[1])
    mg_df = dask_cudf.from_cudf(sg_df, npartitions=2)
    mpG.add_edge_data(mg_df,
                      type_name="referrals",
                      vertex_col_names=("user_id_1", "user_id_2"),
                      property_columns=None)

    return mpG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def net_MGPropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the netscience.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import MGPropertyGraph
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

    dpG = MGPropertyGraph()
    dpG.add_edge_data(ddf, ("src", "dst"))
    return dpG


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_extract_subgraph_no_query(net_MGPropertyGraph, net_PropertyGraph):
    """
    Call extract with no args, should result in the entire property graph.
    """
    dpG = net_MGPropertyGraph
    pG = net_PropertyGraph
    assert pG.num_edges == dpG.num_edges
    assert pG.num_vertices == dpG.num_vertices
    # tests that the edges are the same in the sg and mg property graph
    sg_df = \
        pG.edges.sort_values(by=['_SRC_', '_DST_']).reset_index(drop=True)
    mg_df = dpG.edges.compute().sort_values(by=['_SRC_', '_DST_'])
    mg_df = mg_df.reset_index(drop=True)
    assert (sg_df.equals(mg_df))
    subgraph = pG.extract_subgraph(allow_multi_edges=False)
    dask_subgraph = dpG.extract_subgraph(allow_multi_edges=False)
    sg_subgraph_df = \
        subgraph.edge_data.sort_values(by=list(subgraph.edge_data.columns))
    sg_subgraph_df = sg_subgraph_df.reset_index(drop=True)
    mg_subgraph_df = dask_subgraph.edge_data.compute()
    mg_subgraph_df = \
        mg_subgraph_df.sort_values(by=list(mg_subgraph_df.columns))
    mg_subgraph_df = mg_subgraph_df.reset_index(drop=True)
    assert (sg_subgraph_df[['_SRC_', '_DST_']]
            .equals(mg_subgraph_df[['_SRC_', '_DST_']]))


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_adding_fixture(dataset1_PropertyGraph, dataset1_MGPropertyGraph):
    sgpG = dataset1_PropertyGraph
    mgPG = dataset1_MGPropertyGraph
    subgraph = sgpG.extract_subgraph(allow_multi_edges=True)
    dask_subgraph = mgPG.extract_subgraph(allow_multi_edges=True)
    sg_subgraph_df = \
        subgraph.edge_data.sort_values(by=list(subgraph.edge_data.columns))
    sg_subgraph_df = sg_subgraph_df.reset_index(drop=True)
    mg_subgraph_df = dask_subgraph.edge_data.compute()
    mg_subgraph_df = \
        mg_subgraph_df.sort_values(by=list(mg_subgraph_df.columns))
    mg_subgraph_df = mg_subgraph_df.reset_index(drop=True)
    assert (sg_subgraph_df[['_SRC_', '_DST_']]
            .equals(mg_subgraph_df[['_SRC_', '_DST_']]))


@pytest.mark.skip(reason="Skipping tests because it is a work in progress")
def test_frame_data(dataset1_PropertyGraph, dataset1_MGPropertyGraph):
    sgpG = dataset1_PropertyGraph
    mgpG = dataset1_MGPropertyGraph

    edge_sort_col = ['_SRC_', '_DST_', '_TYPE_']
    vert_sort_col = ['_VERTEX_', '_TYPE_']
    # vertex_prop_dataframe
    sg_vp_df = sgpG._vertex_prop_dataframe.\
        sort_values(by=vert_sort_col).reset_index(drop=True)
    mg_vp_df = mgpG._vertex_prop_dataframe.compute()\
        .sort_values(by=vert_sort_col).reset_index(drop=True)
    assert (sg_vp_df['_VERTEX_'].equals(mg_vp_df['_VERTEX_']))

    # get_edge_prop_dataframe
    sg_ep_df = sgpG._edge_prop_dataframe\
        .sort_values(by=edge_sort_col).reset_index(drop=True)
    mg_ep_df = mgpG._edge_prop_dataframe\
        .compute().sort_values(by=edge_sort_col).reset_index(drop=True)
    assert (sg_ep_df['_SRC_'].equals(mg_ep_df['_SRC_']))

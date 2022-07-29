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

import dask_cudf
import pytest
import pandas as pd
import cudf
from cudf.testing import assert_frame_equal

import cugraph.dask as dcg
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


dataset2 = {
    "simple": [
        ["src", "dst", "some_property"],
        [(99, 22, "a"),
         (98, 34, "b"),
         (97, 56, "c"),
         (96, 88, "d"),
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

    (merchants, users,
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
    return (pG, dataset1)


@pytest.fixture(scope="module")
def dataset1_MGPropertyGraph(dask_client):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = cudf.DataFrame
    (merchants, users,
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

    return (mpG, dataset1)


@pytest.fixture(scope="module")
def dataset2_simple_MGPropertyGraph(dask_client):
    from cugraph.experimental import MGPropertyGraph

    dataframe_type = cudf.DataFrame
    simple = dataset2["simple"]
    mpG = MGPropertyGraph()

    sg_df = dataframe_type(columns=simple[0], data=simple[1])
    mgdf = dask_cudf.from_cudf(sg_df, npartitions=2)

    mpG.add_edge_data(mgdf,
                      vertex_col_names=("src", "dst"))

    return (mpG, simple)


@pytest.fixture(scope="module")
def dataset2_MGPropertyGraph(dask_client):
    from cugraph.experimental import MGPropertyGraph

    dataframe_type = cudf.DataFrame
    simple = dataset2["simple"]
    mpG = MGPropertyGraph()

    sg_df = dataframe_type(columns=simple[0], data=simple[1])
    mgdf = dask_cudf.from_cudf(sg_df, npartitions=2)

    mpG.add_edge_data(mgdf,
                      vertex_col_names=("src", "dst"))

    return (mpG, simple)


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
    assert pG.get_num_edges() == dpG.get_num_edges()
    assert pG.get_num_vertices() == dpG.get_num_vertices()
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
    (sgpG, _) = dataset1_PropertyGraph
    (mgPG, _) = dataset1_MGPropertyGraph
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
    (sgpG, _) = dataset1_PropertyGraph
    (mgpG, _) = dataset1_MGPropertyGraph

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


def test_property_names_attrs(dataset1_MGPropertyGraph):
    """
    Ensure the correct number of user-visible properties for vertices and edges
    are returned. This should exclude the internal bookkeeping properties.
    """
    (pG, data) = dataset1_MGPropertyGraph

    expected_vert_prop_names = ["merchant_id", "merchant_location",
                                "merchant_size", "merchant_sales",
                                "merchant_num_employees", "merchant_name",
                                "user_id", "user_location", "vertical"]
    expected_edge_prop_names = ["user_id", "merchant_id", "volume", "time",
                                "card_num", "card_type", "user_id_1",
                                "user_id_2", "relationship_type", "stars"]

    # Extracting a subgraph with weights has/had a side-effect of adding a
    # weight column, so call extract_subgraph() to ensure the internal weight
    # column name is not present.
    pG.extract_subgraph(default_edge_weight=1.0, allow_multi_edges=True)

    actual_vert_prop_names = pG.vertex_property_names
    actual_edge_prop_names = pG.edge_property_names

    assert sorted(actual_vert_prop_names) == sorted(expected_vert_prop_names)
    assert sorted(actual_edge_prop_names) == sorted(expected_edge_prop_names)


def test_extract_subgraph_nonrenumbered_noedgedata(
        dataset2_simple_MGPropertyGraph):
    """
    Ensure a subgraph can be extracted that contains no edge_data.  Also ensure
    renumber cannot be False since that is currently not allowed for MG.
    """
    from cugraph import Graph

    (pG, data) = dataset2_simple_MGPropertyGraph

    # renumber=False is currently not allowed for MG.
    with pytest.raises(ValueError):
        G = pG.extract_subgraph(create_using=Graph(directed=True),
                                renumber_graph=False,
                                add_edge_data=False)

    G = pG.extract_subgraph(create_using=Graph(directed=True),
                            add_edge_data=False)

    actual_edgelist = G.edgelist.edgelist_df.compute()

    src_col_name = pG.src_col_name
    dst_col_name = pG.dst_col_name

    # create a DF without the properties (ie. the last column)
    expected_edgelist = cudf.DataFrame(columns=[src_col_name, dst_col_name],
                                       data=[(i, j) for (i, j, k) in data[1]])

    assert_frame_equal(expected_edgelist.sort_values(by=src_col_name,
                                                     ignore_index=True),
                       actual_edgelist.sort_values(by=src_col_name,
                                                   ignore_index=True))
    assert hasattr(G, "edge_data") is False


def test_num_vertices_with_properties(dataset2_simple_MGPropertyGraph):
    """
    Checks that the num_vertices_with_properties attr is set to the number of
    vertices that have properties, as opposed to just num_vertices which also
    includes all verts in the graph edgelist.
    """
    (pG, data) = dataset2_simple_MGPropertyGraph

    # assume no repeated vertices
    assert pG.get_num_vertices() == len(data[1]) * 2
    assert pG.get_num_vertices(include_edge_data=False) == 0

    df = cudf.DataFrame({"vertex": [98, 97],
                         "some_property": ["a", "b"],
                         })
    mgdf = dask_cudf.from_cudf(df, npartitions=2)
    pG.add_vertex_data(mgdf, vertex_col_name="vertex")

    # assume no repeated vertices
    assert pG.get_num_vertices() == len(data[1]) * 2
    assert pG.get_num_vertices(include_edge_data=False) == 2


def test_edges_attr(dataset2_simple_MGPropertyGraph):
    """
    Ensure the edges attr returns the src, dst, edge_id columns properly.
    """
    (pG, data) = dataset2_simple_MGPropertyGraph

    # create a DF without the properties (ie. the last column)
    expected_edges = cudf.DataFrame(columns=[pG.src_col_name, pG.dst_col_name],
                                    data=[(i, j) for (i, j, k) in data[1]])
    actual_edges = pG.edges[[pG.src_col_name, pG.dst_col_name]].compute()

    assert_frame_equal(expected_edges.sort_values(by=pG.src_col_name,
                                                  ignore_index=True),
                       actual_edges.sort_values(by=pG.src_col_name,
                                                ignore_index=True))
    edge_ids = pG.edges[pG.edge_id_col_name].compute()
    expected_num_edges = len(data[1])
    assert len(edge_ids) == expected_num_edges
    assert edge_ids.nunique() == expected_num_edges


def test_get_vertex_data(dataset1_MGPropertyGraph):
    """
    Ensure PG.get_vertex_data() returns the correct data based on vertex IDs
    passed in.
    """
    (pG, data) = dataset1_MGPropertyGraph

    # Ensure the generated vertex IDs are unique
    all_vertex_data = pG.get_vertex_data()
    assert all_vertex_data[pG.vertex_col_name].nunique().compute() == \
        len(all_vertex_data)

    # Test with specific columns and types
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(types=[vert_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # vert/type + specified columns
    standard_vert_columns = [pG.vertex_col_name, pG.type_col_name]
    assert len(some_vertex_data) == len(data[vert_type][1])
    assert (
        sorted(some_vertex_data.columns) ==
        sorted(columns + standard_vert_columns)
    )

    # Test with all params specified
    vert_ids = [11, 4, 21]
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(vertex_ids=vert_ids,
                                          types=[vert_type],
                                          columns=columns)
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_vertex_data) == len(vert_ids)
    assert set(columns) - set(some_vertex_data.columns) == set()


def test_get_edge_data(dataset1_MGPropertyGraph):
    """
    Ensure PG.get_edge_data() returns the correct data based on edge IDs passed
    in.
    """
    (pG, data) = dataset1_MGPropertyGraph

    # Ensure the generated edge IDs are unique
    all_edge_data = pG.get_edge_data()
    assert all_edge_data[pG.edge_id_col_name].nunique().compute() == \
        len(all_edge_data)

    # Test with specific edge IDs
    edge_ids = [4, 5, 6]
    some_edge_data = pG.get_edge_data(edge_ids)
    actual_edge_ids = some_edge_data[pG.edge_id_col_name].compute()
    if hasattr(actual_edge_ids, "values_host"):
        actual_edge_ids = actual_edge_ids.values_host
    assert sorted(actual_edge_ids) == sorted(edge_ids)

    # Create a list of expected column names from the three input tables
    expected_columns = set([pG.src_col_name, pG.dst_col_name,
                            pG.edge_id_col_name, pG.type_col_name])
    for d in ["transactions", "relationships", "referrals"]:
        for name in data[d][0]:
            expected_columns.add(name)

    actual_columns = set(some_edge_data.columns)

    assert actual_columns == expected_columns

    # Test with specific columns and types
    edge_type = "transactions"
    columns = ["card_num", "card_type"]

    some_edge_data = pG.get_edge_data(types=[edge_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # src/dst/id/type + specified columns
    standard_edge_columns = [pG.src_col_name, pG.dst_col_name,
                             pG.edge_id_col_name, pG.type_col_name]
    assert len(some_edge_data) == len(data[edge_type][1])
    assert (
        sorted(some_edge_data.columns) ==
        sorted(columns + standard_edge_columns)
    )

    # Test with all params specified
    # FIXME: since edge IDs are generated, assume that these are correct based
    # on the intended edges being the first three added.
    edge_ids = [0, 1, 2]
    edge_type = "transactions"
    columns = ["card_num", "card_type"]
    some_edge_data = pG.get_edge_data(edge_ids=edge_ids,
                                      types=[edge_type],
                                      columns=columns)
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_edge_data) == len(edge_ids)
    assert set(columns) - set(some_edge_data.columns) == set()


def test_get_data_empty_graphs(dask_client):
    """
    Ensures that calls to pG.get_*_data() on an empty pG are handled correctly.
    """
    from cugraph.experimental import MGPropertyGraph

    pG = MGPropertyGraph()

    assert pG.get_vertex_data() is None
    assert pG.get_vertex_data([0, 1, 2]) is None
    assert pG.get_edge_data() is None
    assert pG.get_edge_data([0, 1, 2]) is None

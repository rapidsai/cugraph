# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import pytest
import pandas as pd
import numpy as np
import cudf
import cupy as cp
from cudf.testing import assert_frame_equal, assert_series_equal
from pylibcugraph.testing.utils import gen_fixture_params_product

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

# FIXME: remove when fully-migrated to pandas 1.5.0
try:
    # pandas 1.5.0
    from pandas.errors import SettingWithCopyWarning as pandas_SettingWithCopyWarning
except ImportError:
    # pandas 1.4
    from pandas.core.common import (
        SettingWithCopyWarning as pandas_SettingWithCopyWarning,
    )

import cugraph
from cugraph.generators import rmat
from cugraph.experimental.datasets import cyber


def type_is_categorical(pG):
    return (
        pG._vertex_prop_dataframe is None
        or pG._vertex_prop_dataframe.dtypes[pG.type_col_name] == "category"
    ) and (
        pG._edge_prop_dataframe is None
        or pG._edge_prop_dataframe.dtypes[pG.type_col_name] == "category"
    )


# =============================================================================
# Test data
# =============================================================================

dataset1 = {
    "merchants": [
        [
            "merchant_id",
            "merchant_location",
            "merchant_size",
            "merchant_sales",
            "merchant_num_employees",
            "merchant_name",
        ],
        [
            (11, 78750, 44, 123.2, 12, "north"),
            (4, 78757, 112, 234.99, 18, "south"),
            (21, 44145, 83, 992.1, 27, "east"),
            (16, 47906, 92, 32.43, 5, "west"),
            (86, 47906, 192, 2.43, 51, "west"),
        ],
    ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [
            (89021, 78757, 0),
            (32431, 78750, 1),
            (89216, 78757, 1),
            (78634, 47906, 0),
        ],
    ],
    "taxpayers": [
        ["payer_id", "amount"],
        [
            (11, 1123.98),
            (4, 3243.7),
            (21, 8932.3),
            (16, 3241.77),
            (86, 789.2),
            (89021, 23.98),
            (78634, 41.77),
        ],
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num", "card_type"],
        [
            (89021, 11, 33.2, 1639084966.5513437, 123456, "MC"),
            (89216, 4, None, 1639085163.481217, 8832, "CASH"),
            (78634, 16, 72.0, 1639084912.567394, 4321, "DEBIT"),
            (32431, 4, 103.2, 1639084721.354346, 98124, "V"),
        ],
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [
            (89216, 89021, 9),
            (89216, 32431, 9),
            (32431, 78634, 8),
            (78634, 89216, 8),
        ],
    ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [
            (89216, 78634, 11, 5),
            (89021, 89216, 4, 4),
            (89021, 89216, 21, 3),
            (89021, 89216, 11, 3),
            (89021, 78634, 21, 4),
            (78634, 32431, 11, 4),
        ],
    ],
}


dataset2 = {
    "simple": [
        ["src", "dst", "some_property"],
        [
            (99, 22, "a"),
            (98, 34, "b"),
            (97, 56, "c"),
            (96, 88, "d"),
        ],
    ],
}


# CSV file contents used for testing various CSV-based use cases.
# These are to be used for test_single_csv_multi_vertex_edge_attrs()
edges_edgeprops_vertexprops_csv = """
src dst edge_attr1 edge_attr2 src_attr1 src_attr2 dst_attr1 dst_attr2
0 1 87 "a" 3.1 "v0" 1.3 "v1"
0 2 88 "b" 3.2 "v0" 1.1 "v2"
2 1 89 "c" 2.3 "v2" 1.9 "v1"
"""

vertexprops_csv = """
vertex attr1 attr2
0 32 dog
1 54 fish
2 87 cat
3 12 snake
4 901 gecko
"""

edgeprops_csv = """
v_src v_dst edge_id
0 1 123
0 2 432
2 1 789
"""

edgeid_edgeprops_csv = """
edge_id attr1 attr2
123 'PUT' 21.32
432 'POST' 21.44
789 'GET' 22.03
"""


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
    DiGraph_inst = cugraph.Graph(directed=True)


# =============================================================================
# Pytest fixtures
# =============================================================================
@pytest.fixture(scope="function", autouse=True)
def raise_on_pandas_warning():
    """Raise when pandas gives SettingWithCopyWarning warning"""
    # Perhaps we should put this in pytest.ini, pyproject.toml, or conftest.py
    import warnings

    filters = list(warnings.filters)
    warnings.filterwarnings("error", category=pandas_SettingWithCopyWarning)
    yield
    warnings.filters = filters


df_types = [cudf.DataFrame, pd.DataFrame]


def df_type_id(dataframe_type):
    """
    Return a string that describes the dataframe_type, used for test output.
    """
    s = "df_type="
    if dataframe_type == cudf.DataFrame:
        return s + "cudf.DataFrame"
    if dataframe_type == pd.DataFrame:
        return s + "pandas.DataFrame"
    return s + "?"


df_types_fixture_params = gen_fixture_params_product((df_types, df_type_id))


@pytest.fixture(scope="function", params=df_types_fixture_params)
def dataset1_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    (
        merchants,
        users,
        taxpayers,
        transactions,
        relationships,
        referrals,
    ) = dataset1.values()

    pG = PropertyGraph()

    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    pG.add_vertex_data(
        dataframe_type(columns=merchants[0], data=merchants[1]),
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=None,
    )
    pG.add_vertex_data(
        dataframe_type(columns=users[0], data=users[1]),
        type_name="users",
        vertex_col_name="user_id",
        property_columns=None,
    )
    # Do not add taxpayers since that may now be considered invalid input (it
    # adds the same vertices under different types, which leads to the same
    # vertex ID appearing in the internal vertex prop table.
    #
    # FIXME: determine if this should be allowed or not then either remove
    # "taxpayers" or uncomment it.
    """
    pG.add_vertex_data(dataframe_type(columns=taxpayers[0],
                                      data=taxpayers[1]),
                       type_name="taxpayers",
                       vertex_col_name="payer_id",
                       property_columns=None)
    """
    pG.add_edge_data(
        dataframe_type(columns=transactions[0], data=transactions[1]),
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )
    pG.add_edge_data(
        dataframe_type(columns=relationships[0], data=relationships[1]),
        type_name="relationships",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )
    pG.add_edge_data(
        dataframe_type(columns=referrals[0], data=referrals[1]),
        type_name="referrals",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    assert type_is_categorical(pG)
    return (pG, dataset1)


@pytest.fixture(scope="module", params=df_types_fixture_params)
def dataset2_simple_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with only edge
    data added from dataset2, parameterized for different DataFrame types.
    """
    dataframe_type = request.param[0]
    from cugraph.experimental import PropertyGraph

    dataframe_type = cudf.DataFrame
    simple = dataset2["simple"]
    pG = PropertyGraph()
    df = dataframe_type(columns=simple[0], data=simple[1])

    pG.add_edge_data(df, vertex_col_names=("src", "dst"))

    assert type_is_categorical(pG)
    return (pG, simple)


@pytest.fixture(scope="module", params=df_types_fixture_params)
def cyber_PropertyGraph(request):
    """
    Fixture which returns an instance of a PropertyGraph with vertex and edge
    data added from the cyber.csv dataset, parameterized for different
    DataFrame types.
    """
    from cugraph.experimental import PropertyGraph

    dataframe_type = request.param[0]
    source_col_name = "srcip"
    dest_col_name = "dstip"

    df = cyber.get_edgelist()
    if dataframe_type is pd.DataFrame:
        df = df.to_pandas()

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    assert type_is_categorical(pG)
    return pG


@pytest.fixture(scope="module", params=df_types_fixture_params)
def rmat_PropertyGraph():
    """
    Fixture which uses the RMAT generator to generate a cuDF DataFrame
    edgelist, then uses it to add vertex and edge data to a PropertyGraph
    instance, then returns the (PropertyGraph, DataFrame) instances in a tuple.
    """
    from cugraph.experimental import PropertyGraph

    source_col_name = "src"
    dest_col_name = "dst"
    weight_col_name = "weight"
    scale = 20
    edgefactor = 16
    seed = 42
    df = rmat(
        scale,
        (2**scale) * edgefactor,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed,
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=None,  # None == return edgelist
        mg=False,
    )
    rng = np.random.default_rng(seed)
    df[weight_col_name] = rng.random(size=len(df))

    pG = PropertyGraph()
    pG.add_edge_data(df, (source_col_name, dest_col_name))

    assert type_is_categorical(pG)
    return (pG, df)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data(df_type):
    """
    add_vertex_data() on "merchants" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0], data=merchants[1])

    pG = PropertyGraph()
    pG.add_vertex_data(
        merchants_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=None,
    )
    assert pG.get_num_vertices() == 5
    assert pG.get_num_vertices("merchants") == 5
    assert pG.get_num_edges() == 0
    expected_props = set(merchants[0].copy()) - {"merchant_id"}
    assert sorted(pG.vertex_property_names) == sorted(expected_props)
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_num_vertices(df_type):
    """
    Ensures get_num_vertices is correct after various additions of data.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0], data=merchants[1])

    pG = PropertyGraph()
    assert pG.get_num_vertices() == 0
    assert pG.get_num_vertices("unknown_type") == 0
    assert pG.get_num_edges("unknown_type") == 0
    pG.add_vertex_data(
        merchants_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=None,
    )

    # Test caching - the second retrieval should always be faster
    st = time.time()
    assert pG.get_num_vertices() == 5
    compute_time = time.time() - st
    assert pG.get_num_edges() == 0

    st = time.time()
    assert pG.get_num_vertices() == 5
    cache_retrieval_time = time.time() - st
    assert cache_retrieval_time < compute_time

    users = dataset1["users"]
    users_df = df_type(columns=users[0], data=users[1])

    pG.add_vertex_data(
        users_df, type_name="users", vertex_col_name="user_id", property_columns=None
    )

    assert pG.get_num_vertices() == 9
    assert pG.get_num_vertices("merchants") == 5
    assert pG.get_num_vertices("users") == 4
    assert pG.get_num_edges() == 0

    # The taxpayers table does not add new unique vertices, it only adds
    # properties to vertices already present in the merchants and users
    # tables.
    taxpayers = dataset1["taxpayers"]
    taxpayers_df = df_type(columns=taxpayers[0], data=taxpayers[1])

    pG.add_vertex_data(
        taxpayers_df,
        type_name="taxpayers",
        vertex_col_name="payer_id",
        property_columns=None,
    )

    assert pG.get_num_vertices() == 9
    assert pG.get_num_vertices("merchants") == 5
    assert pG.get_num_vertices("users") == 4
    assert pG.get_num_vertices("unknown_type") == 0
    assert pG.get_num_edges() == 0
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_type_names(df_type):
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()
    assert pG.edge_types == set()
    assert pG.vertex_types == set()

    df = df_type(
        {
            "src": [99, 98, 97],
            "dst": [22, 34, 56],
            "some_property": ["a", "b", "c"],
        }
    )
    pG.add_edge_data(df, vertex_col_names=("src", "dst"))
    assert pG.edge_types == set([""])
    assert pG.vertex_types == set([""])

    df = df_type(
        {
            "vertex": [98, 97],
            "some_property": ["a", "b"],
        }
    )
    pG.add_vertex_data(df, type_name="vtype", vertex_col_name="vertex")
    assert pG.edge_types == set([""])
    assert pG.vertex_types == set(["", "vtype"])

    df = df_type(
        {
            "src": [199, 98, 197],
            "dst": [22, 134, 56],
            "some_property": ["a", "b", "c"],
        }
    )
    pG.add_edge_data(df, type_name="etype", vertex_col_names=("src", "dst"))
    assert pG.edge_types == set(["", "etype"])
    assert pG.vertex_types == set(["", "vtype"])
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_num_vertices_include_edge_data(df_type):
    """
    Ensures get_num_vertices is correct after various additions of data.
    """
    from cugraph.experimental import PropertyGraph

    (
        merchants,
        users,
        taxpayers,
        transactions,
        relationships,
        referrals,
    ) = dataset1.values()

    pG = PropertyGraph()
    assert pG.get_num_vertices(include_edge_data=False) == 0
    assert pG.get_num_vertices("", include_edge_data=False) == 0

    pG.add_edge_data(
        df_type(columns=transactions[0], data=transactions[1]),
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )

    assert pG.get_num_vertices(include_edge_data=False) == 0
    assert pG.get_num_vertices("", include_edge_data=False) == 0
    assert pG.get_num_vertices(include_edge_data=True) == 7
    assert pG.get_num_vertices("", include_edge_data=True) == 7
    pG.add_vertex_data(
        df_type(columns=merchants[0], data=merchants[1]),
        # type_name="merchants",  # Use default!
        vertex_col_name="merchant_id",
        property_columns=None,
    )
    assert pG.get_num_vertices(include_edge_data=False) == 5
    assert pG.get_num_vertices("", include_edge_data=False) == 5
    assert pG.get_num_vertices(include_edge_data=True) == 9
    assert pG.get_num_vertices("", include_edge_data=True) == 9
    pG.add_vertex_data(
        df_type(columns=users[0], data=users[1]),
        type_name="users",
        vertex_col_name="user_id",
        property_columns=None,
    )
    assert pG.get_num_vertices(include_edge_data=False) == 9
    assert pG.get_num_vertices("", include_edge_data=False) == 5
    assert pG.get_num_vertices("users", include_edge_data=False) == 4
    # All vertices now have vertex data, so this should match
    assert pG.get_num_vertices(include_edge_data=True) == 9
    assert pG.get_num_vertices("", include_edge_data=True) == 5
    assert pG.get_num_vertices("users", include_edge_data=True) == 4


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_num_vertices_with_properties(df_type):
    """
    Checks that the num_vertices_with_properties attr is set to the number of
    vertices that have properties, as opposed to just num_vertices which also
    includes all verts in the graph edgelist.
    """
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()
    df = df_type(
        {
            "src": [99, 98, 97],
            "dst": [22, 34, 56],
            "some_property": ["a", "b", "c"],
        }
    )
    pG.add_edge_data(df, vertex_col_names=("src", "dst"))

    assert pG.get_num_vertices() == 6
    assert pG.get_num_vertices(include_edge_data=False) == 0

    df = df_type(
        {
            "vertex": [98, 97],
            "some_property": ["a", "b"],
        }
    )
    pG.add_vertex_data(df, vertex_col_name="vertex")

    assert pG.get_num_vertices() == 6
    assert pG.get_num_vertices(include_edge_data=False) == 2


def test_edges_attr(dataset2_simple_PropertyGraph):
    """
    Ensure the edges attr returns the src, dst, edge_id columns properly.
    """
    (pG, data) = dataset2_simple_PropertyGraph

    # create a DF without the properties (ie. the last column)
    expected_edges = cudf.DataFrame(
        columns=[pG.src_col_name, pG.dst_col_name],
        data=[(i, j) for (i, j, k) in data[1]],
    )
    actual_edges = pG.edges[[pG.src_col_name, pG.dst_col_name]]

    assert_frame_equal(
        expected_edges.sort_values(by=pG.src_col_name, ignore_index=True),
        actual_edges.sort_values(by=pG.src_col_name, ignore_index=True),
    )
    edge_ids = pG.edges[pG.edge_id_col_name]
    expected_num_edges = len(data[1])
    assert len(edge_ids) == expected_num_edges
    assert edge_ids.nunique() == expected_num_edges


def test_get_vertex_data(dataset1_PropertyGraph):
    """
    Ensure PG.get_vertex_data() returns the correct data based on vertex IDs
    passed in.
    """
    (pG, data) = dataset1_PropertyGraph

    # Ensure the generated vertex IDs are unique
    all_vertex_data = pG.get_vertex_data()
    assert all_vertex_data[pG.vertex_col_name].nunique() == len(all_vertex_data)

    # Test getting a subset of data
    # Use the appropriate series type based on input
    # FIXME: do not use the debug _vertex_prop_dataframe to determine type
    if isinstance(pG._vertex_prop_dataframe, cudf.DataFrame):
        vert_ids = cudf.Series([11, 4, 21])
    else:
        vert_ids = pd.Series([11, 4, 21])

    some_vertex_data = pG.get_vertex_data(vert_ids)
    actual_vertex_ids = some_vertex_data[pG.vertex_col_name]
    if hasattr(actual_vertex_ids, "values_host"):
        actual_vertex_ids = actual_vertex_ids.values_host
    if hasattr(vert_ids, "values_host"):
        vert_ids = vert_ids.values_host
    assert sorted(actual_vertex_ids) == sorted(vert_ids)

    expected_columns = set([pG.vertex_col_name, pG.type_col_name])
    for d in ["merchants", "users"]:
        for name in data[d][0]:
            expected_columns.add(name)
    expected_columns -= {"merchant_id", "user_id"}
    actual_columns = set(some_vertex_data.columns)
    assert actual_columns == expected_columns

    # Test with specific columns and types
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(types=[vert_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # vert/type + specified columns
    standard_vert_columns = [pG.vertex_col_name, pG.type_col_name]
    assert len(some_vertex_data) == len(data[vert_type][1])
    assert sorted(some_vertex_data.columns) == sorted(columns + standard_vert_columns)

    # Test with all params specified
    vert_ids = [11, 4, 21]
    vert_type = "merchants"
    columns = ["merchant_location", "merchant_size"]

    some_vertex_data = pG.get_vertex_data(
        vertex_ids=vert_ids, types=[vert_type], columns=columns
    )
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_vertex_data) == len(vert_ids)
    assert set(columns) - set(some_vertex_data.columns) == set()

    # Allow a single vertex type and single vertex id to be passed in
    df1 = pG.get_vertex_data(vertex_ids=[11], types=[vert_type])
    df2 = pG.get_vertex_data(vertex_ids=11, types=vert_type)
    assert len(df1) == 1
    assert df1.shape == df2.shape
    # assert_frame_equal(df1, df2, check_like=True)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_get_vertex_data_repeated(df_type):
    from cugraph.experimental import PropertyGraph

    df = df_type({"vertex": [2, 3, 4, 1], "feat": np.arange(4)})
    pG = PropertyGraph()
    pG.add_vertex_data(df, "vertex")
    df1 = pG.get_vertex_data(vertex_ids=[2, 1, 3, 1], columns=["feat"])
    expected = df_type(
        {
            pG.vertex_col_name: [2, 1, 3, 1],
            pG.type_col_name: ["", "", "", ""],
            "feat": [0, 3, 1, 3],
        }
    )
    df1[pG.type_col_name] = df1[pG.type_col_name].astype(str)  # Undo category
    if df_type is cudf.DataFrame:
        afe = assert_frame_equal
    else:
        afe = pd.testing.assert_frame_equal
    expected["feat"] = expected["feat"].astype("Int64")
    afe(df1, expected)


def test_get_edge_data(dataset1_PropertyGraph):
    """
    Ensure PG.get_edge_data() returns the correct data based on edge IDs passed
    in.
    """
    (pG, data) = dataset1_PropertyGraph

    # Ensure the generated edge IDs are unique
    all_edge_data = pG.get_edge_data()
    assert all_edge_data[pG.edge_id_col_name].nunique() == len(all_edge_data)

    # Test with specific edge IDs
    edge_ids = [4, 5, 6]
    some_edge_data = pG.get_edge_data(edge_ids)
    actual_edge_ids = some_edge_data[pG.edge_id_col_name]
    if hasattr(actual_edge_ids, "values_host"):
        actual_edge_ids = actual_edge_ids.values_host
    assert sorted(actual_edge_ids) == sorted(edge_ids)

    # Create a list of expected column names from the three input tables
    expected_columns = set(
        [pG.src_col_name, pG.dst_col_name, pG.edge_id_col_name, pG.type_col_name]
    )
    for d in ["transactions", "relationships", "referrals"]:
        for name in data[d][0]:
            expected_columns.add(name)
    expected_columns -= {"user_id", "user_id_1", "user_id_2"}

    actual_columns = set(some_edge_data.columns)

    assert actual_columns == expected_columns

    # Test with specific columns and types
    edge_type = "transactions"
    columns = ["card_num", "card_type"]

    some_edge_data = pG.get_edge_data(types=[edge_type], columns=columns)
    # Ensure the returned df is the right length and includes only the
    # src/dst/id/type + specified columns
    standard_edge_columns = [
        pG.src_col_name,
        pG.dst_col_name,
        pG.edge_id_col_name,
        pG.type_col_name,
    ]
    assert len(some_edge_data) == len(data[edge_type][1])
    assert sorted(some_edge_data.columns) == sorted(columns + standard_edge_columns)

    # Test with all params specified
    # FIXME: since edge IDs are generated, assume that these are correct based
    # on the intended edges being the first three added.
    edge_ids = [0, 1, 2]
    edge_type = "transactions"
    columns = ["card_num", "card_type"]
    some_edge_data = pG.get_edge_data(
        edge_ids=edge_ids, types=[edge_type], columns=columns
    )
    # Ensure the returned df is the right length and includes at least the
    # specified columns.
    assert len(some_edge_data) == len(edge_ids)
    assert set(columns) - set(some_edge_data.columns) == set()

    # Allow a single edge type and single edge id to be passed in
    df1 = pG.get_edge_data(edge_ids=[1], types=[edge_type])
    df2 = pG.get_edge_data(edge_ids=1, types=edge_type)
    assert len(df1) == 1
    assert df1.shape == df2.shape
    # assert_frame_equal(df1, df2, check_like=True)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_get_edge_data_repeated(df_type):
    from cugraph.experimental import PropertyGraph

    df = df_type({"src": [1, 1, 1, 2], "dst": [2, 3, 4, 1], "edge_feat": np.arange(4)})
    pG = PropertyGraph()
    pG.add_edge_data(df, vertex_col_names=["src", "dst"])
    df1 = pG.get_edge_data(edge_ids=[2, 1, 3, 1], columns=["edge_feat"])
    expected = df_type(
        {
            pG.edge_id_col_name: [2, 1, 3, 1],
            pG.src_col_name: [1, 1, 2, 1],
            pG.dst_col_name: [4, 3, 1, 3],
            pG.type_col_name: ["", "", "", ""],
            "edge_feat": [2, 1, 3, 1],
        }
    )
    df1[pG.type_col_name] = df1[pG.type_col_name].astype(str)  # Undo category
    if df_type is cudf.DataFrame:
        afe = assert_frame_equal
    else:
        afe = pd.testing.assert_frame_equal
    for col in ["edge_feat", pG.src_col_name, pG.dst_col_name]:
        expected[col] = expected[col].astype("Int64")
    afe(df1, expected)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_null_data(df_type):
    """
    test for null data
    """
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()

    assert pG.get_num_vertices() == 0
    assert pG.get_num_edges() == 0
    assert sorted(pG.vertex_property_names) == sorted([])
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_vertex_data_prop_columns(df_type):
    """
    add_vertex_data() on "merchants" table, subset of properties.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = df_type(columns=merchants[0], data=merchants[1])
    expected_props = ["merchant_name", "merchant_sales", "merchant_location"]

    pG = PropertyGraph()
    pG.add_vertex_data(
        merchants_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        property_columns=expected_props,
    )

    assert pG.get_num_vertices() == 5
    assert pG.get_num_vertices("merchants") == 5
    assert pG.get_num_edges() == 0
    assert sorted(pG.vertex_property_names) == sorted(expected_props)
    assert type_is_categorical(pG)


def test_add_vertex_data_bad_args():
    """
    add_vertex_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph.experimental import PropertyGraph

    merchants = dataset1["merchants"]
    merchants_df = cudf.DataFrame(columns=merchants[0], data=merchants[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_vertex_data(
            42,
            type_name="merchants",
            vertex_col_name="merchant_id",
            property_columns=None,
        )
    with pytest.raises(TypeError):
        pG.add_vertex_data(
            merchants_df,
            type_name=42,
            vertex_col_name="merchant_id",
            property_columns=None,
        )
    with pytest.raises(ValueError):
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="bad_column_name",
            property_columns=None,
        )
    with pytest.raises(ValueError):
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            property_columns=["bad_column_name", "merchant_name"],
        )
    with pytest.raises(TypeError):
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            property_columns="merchant_name",
        )


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data(df_type):
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0], data=transactions[1])

    pG = PropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )

    assert pG.get_num_vertices() == 7
    # 'transactions' is edge type, not vertex type
    assert pG.get_num_vertices("transactions") == 0
    assert pG.get_num_edges() == 4
    assert pG.get_num_edges("transactions") == 4
    # Original SRC and DST columns no longer include "merchant_id", "user_id"
    expected_props = ["volume", "time", "card_num", "card_type"]
    assert sorted(pG.edge_property_names) == sorted(expected_props)
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data_prop_columns(df_type):
    """
    add_edge_data() on "transactions" table, subset of properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0], data=transactions[1])
    expected_props = ["card_num", "card_type"]

    pG = PropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=expected_props,
    )

    assert pG.get_num_vertices() == 7
    # 'transactions' is edge type, not vertex type
    assert pG.get_num_vertices("transactions") == 0
    assert pG.get_num_edges() == 4
    assert pG.get_num_edges("transactions") == 4
    assert sorted(pG.edge_property_names) == sorted(expected_props)
    assert type_is_categorical(pG)


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_edge_data_with_ids(df_type):
    """
    add_edge_data() on "transactions" table, all properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = df_type(columns=transactions[0], data=transactions[1])
    transactions_df["edge_id"] = list(range(10, 10 + len(transactions_df)))

    pG = PropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        edge_id_col_name="edge_id",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )

    assert pG.get_num_vertices() == 7
    # 'transactions' is edge type, not vertex type
    assert pG.get_num_vertices("transactions") == 0
    assert pG.get_num_edges() == 4
    assert pG.get_num_edges("transactions") == 4
    # Original SRC and DST columns no longer include "merchant_id", "user_id"
    expected_props = ["volume", "time", "card_num", "card_type"]
    assert sorted(pG.edge_property_names) == sorted(expected_props)

    relationships = dataset1["relationships"]
    relationships_df = df_type(columns=relationships[0], data=relationships[1])

    # user-provided, then auto-gen (not allowed)
    with pytest.raises(NotImplementedError):
        pG.add_edge_data(
            relationships_df,
            type_name="relationships",
            vertex_col_names=("user_id_1", "user_id_2"),
            property_columns=None,
        )

    relationships_df["edge_id"] = list(range(30, 30 + len(relationships_df)))

    pG.add_edge_data(
        relationships_df,
        type_name="relationships",
        edge_id_col_name="edge_id",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    if df_type is cudf.DataFrame:
        ase = assert_series_equal
    else:
        ase = pd.testing.assert_series_equal
    df = pG.get_edge_data(types="transactions")
    ase(
        df[pG.edge_id_col_name].sort_values().reset_index(drop=True),
        transactions_df["edge_id"],
        check_names=False,
    )
    df = pG.get_edge_data(types="relationships")
    ase(
        df[pG.edge_id_col_name].sort_values().reset_index(drop=True),
        relationships_df["edge_id"],
        check_names=False,
    )

    # auto-gen, then user-provided (not allowed)
    pG = PropertyGraph()
    pG.add_edge_data(
        transactions_df,
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )
    with pytest.raises(NotImplementedError):
        pG.add_edge_data(
            relationships_df,
            type_name="relationships",
            edge_id_col_name="edge_id",
            vertex_col_names=("user_id_1", "user_id_2"),
            property_columns=None,
        )


def test_add_edge_data_bad_args():
    """
    add_edge_data() with various bad args, checks that proper exceptions are
    raised.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    transactions_df = cudf.DataFrame(columns=transactions[0], data=transactions[1])

    pG = PropertyGraph()
    with pytest.raises(TypeError):
        pG.add_edge_data(
            42,
            type_name="transactions",
            vertex_col_names=("user_id", "merchant_id"),
            property_columns=None,
        )
    with pytest.raises(TypeError):
        pG.add_edge_data(
            transactions_df,
            type_name=42,
            vertex_col_names=("user_id", "merchant_id"),
            property_columns=None,
        )
    with pytest.raises(ValueError):
        pG.add_edge_data(
            transactions_df,
            type_name="transactions",
            vertex_col_names=("user_id", "bad_column"),
            property_columns=None,
        )
    with pytest.raises(ValueError):
        pG.add_edge_data(
            transactions_df,
            type_name="transactions",
            vertex_col_names=("user_id", "merchant_id"),
            property_columns=["bad_column_name", "time"],
        )
    with pytest.raises(TypeError):
        pG.add_edge_data(
            transactions_df,
            type_name="transactions",
            vertex_col_names=("user_id", "merchant_id"),
            property_columns="time",
        )
    with pytest.raises(TypeError):
        pG.add_edge_data(
            transactions_df,
            type_name="transactions",
            edge_id_col_name=42,
            vertex_col_names=("user_id", "merchant_id"),
            property_columns=None,
        )
    with pytest.raises(ValueError):
        pG.add_edge_data(
            transactions_df,
            type_name="transactions",
            edge_id_col_name="MISSING",
            vertex_col_names=("user_id", "merchant_id"),
            property_columns=None,
        )


def test_extract_subgraph_vertex_prop_condition_only(dataset1_PropertyGraph):

    (pG, data) = dataset1_PropertyGraph

    # This should result in two users: 78634 and 89216
    selection = pG.select_vertices(
        f"({pG.type_col_name}=='users') "
        "& ((user_location<78750) | ((user_location==78757) & (vertical==1)))"
    )
    G = pG.extract_subgraph(
        selection=selection,
        create_using=DiGraph_inst,
        edge_weight_property="relationship_type",
        default_edge_weight=99,
    )
    # Should result in two edges, one a "relationship", the other a "referral"
    expected_edgelist = cudf.DataFrame(
        {"src": [89216, 78634], "dst": [78634, 89216], "weights": [99, 8]}
    )
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    # check_like=True ignores differences in column/index ordering
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_vertex_edge_prop_condition(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_vertices("(user_location==47906) | " "(user_location==78750)")
    selection += pG.select_edges(f"{tcn}=='referrals'")
    G = pG.extract_subgraph(
        selection=selection, create_using=DiGraph_inst, edge_weight_property="stars"
    )

    expected_edgelist = cudf.DataFrame({"src": [78634], "dst": [32431], "weights": [4]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_edge_prop_condition_only(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn} =='transactions'")
    G = pG.extract_subgraph(selection=selection, create_using=DiGraph_inst)

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts) = zip(*[(t[0], t[1]) for t in transactions])
    expected_edgelist = cudf.DataFrame({"src": srcs, "dst": dsts})
    expected_edgelist = expected_edgelist.sort_values(by="src", ignore_index=True)

    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)
    actual_edgelist = actual_edgelist.sort_values(by="src", ignore_index=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_unweighted(dataset1_PropertyGraph):
    """
    Ensure a subgraph is unweighted if the edge_weight_property is None.
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn} == 'transactions'")
    G = pG.extract_subgraph(selection=selection, create_using=DiGraph_inst)

    assert G.is_weighted() is False


def test_extract_subgraph_specific_query(dataset1_PropertyGraph):
    """
    Graph of only transactions after time 1639085000 for merchant_id 4 (should
    be a graph of 2 vertices, 1 edge)
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # _DST_ below used to be referred to as merchant_id
    selection = pG.select_edges(
        f"({tcn}=='transactions') & " "(_DST_==4) & " "(time>1639085000)"
    )
    G = pG.extract_subgraph(
        selection=selection, create_using=DiGraph_inst, edge_weight_property="card_num"
    )

    expected_edgelist = cudf.DataFrame({"src": [89216], "dst": [4], "weights": [8832]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_select_vertices_from_previous_selection(dataset1_PropertyGraph):
    """
    Ensures that the intersection of vertices of multiple types (only vertices
    that are both type A and type B) can be selected.
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # Select referrals from only users 89216 and 78634 using an intentionally
    # awkward query with separate select calls to test from_previous_selection
    selection = pG.select_vertices(f"{tcn} == 'users'")
    selection = pG.select_vertices(
        "((user_location == 78757) & (vertical == 1)) " "| (user_location == 47906)",
        from_previous_selection=selection,
    )
    selection += pG.select_edges(f"{tcn} == 'referrals'")
    G = pG.extract_subgraph(create_using=DiGraph_inst, selection=selection)

    expected_edgelist = cudf.DataFrame({"src": [89216], "dst": [78634]})
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_graph_without_vert_props():
    """
    Ensure a subgraph can be extracted from a PropertyGraph that does not have
    vertex properties.
    """
    from cugraph.experimental import PropertyGraph

    transactions = dataset1["transactions"]
    relationships = dataset1["relationships"]

    pG = PropertyGraph()

    pG.add_edge_data(
        cudf.DataFrame(columns=transactions[0], data=transactions[1]),
        type_name="transactions",
        vertex_col_names=("user_id", "merchant_id"),
        property_columns=None,
    )
    pG.add_edge_data(
        cudf.DataFrame(columns=relationships[0], data=relationships[1]),
        type_name="relationships",
        vertex_col_names=("user_id_1", "user_id_2"),
        property_columns=None,
    )

    scn = PropertyGraph.src_col_name
    G = pG.extract_subgraph(
        selection=pG.select_edges(f"{scn} == 89216"),
        create_using=DiGraph_inst,
        edge_weight_property="relationship_type",
        default_edge_weight=0,
    )

    expected_edgelist = cudf.DataFrame(
        {"src": [89216, 89216, 89216], "dst": [4, 89021, 32431], "weights": [0, 9, 9]}
    )
    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_no_edges(dataset1_PropertyGraph):
    """
    Valid query that only matches a single vertex.
    """
    (pG, data) = dataset1_PropertyGraph

    # "merchant_id" column is no longer saved; use as "_VERTEX_"
    with pytest.raises(NameError, match="merchant_id"):
        selection = pG.select_vertices("(_TYPE_=='merchants') & (merchant_id==86)")

    selection = pG.select_vertices("(_TYPE_=='merchants') & (_VERTEX_==86)")
    G = pG.extract_subgraph(selection=selection)
    assert G.is_directed()

    assert len(G.edgelist.edgelist_df) == 0


def test_extract_subgraph_no_query(dataset1_PropertyGraph):
    """
    Call extract with no args, should result in the entire property graph.
    """
    (pG, data) = dataset1_PropertyGraph

    G = pG.extract_subgraph(create_using=DiGraph_inst, check_multi_edges=False)

    num_edges = (
        len(dataset1["transactions"][-1])
        + len(dataset1["relationships"][-1])
        + len(dataset1["referrals"][-1])
    )
    # referrals has 3 edges with the same src/dst, so subtract 2 from
    # the total count since this is not creating a multigraph..
    num_edges -= 2
    assert len(G.edgelist.edgelist_df) == num_edges


def test_extract_subgraph_multi_edges(dataset1_PropertyGraph):
    """
    Ensure an exception is thrown if a graph is attempted to be extracted with
    multi edges.
    NOTE: an option to allow multi edges when create_using is
    MultiGraph will be provided in the future.
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # referrals has multiple edges
    selection = pG.select_edges(f"{tcn} == 'referrals'")

    # FIXME: use a better exception
    with pytest.raises(RuntimeError):
        pG.extract_subgraph(
            selection=selection, create_using=DiGraph_inst, check_multi_edges=True
        )


def test_extract_subgraph_bad_args(dataset1_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    # non-PropertySelection selection
    with pytest.raises(TypeError):
        pG.extract_subgraph(
            selection=78750,
            create_using=DiGraph_inst,
            edge_weight_property="stars",
            default_edge_weight=1.0,
        )

    selection = pG.select_edges(f"{tcn}=='referrals'")
    # bad create_using type
    with pytest.raises(TypeError):
        pG.extract_subgraph(
            selection=selection,
            create_using=pytest,
            edge_weight_property="stars",
            default_edge_weight=1.0,
        )
    # invalid column name
    with pytest.raises(ValueError):
        pG.extract_subgraph(
            selection=selection,
            edge_weight_property="bad_column",
            default_edge_weight=1.0,
        )
    # column name has None value for all results in subgraph and
    # default_edge_weight is not set.
    with pytest.raises(ValueError):
        pG.extract_subgraph(selection=selection, edge_weight_property="card_type")


def test_extract_subgraph_default_edge_weight(dataset1_PropertyGraph):
    """
    Ensure the default_edge_weight value is added to edges with missing
    properties used for weights.
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    tcn = PropertyGraph.type_col_name

    selection = pG.select_edges(f"{tcn}=='transactions'")
    G = pG.extract_subgraph(
        create_using=DiGraph_inst,
        selection=selection,
        edge_weight_property="volume",
        default_edge_weight=99,
    )

    # last item is the DataFrame rows
    transactions = dataset1["transactions"][-1]
    (srcs, dsts, weights) = zip(*[(t[0], t[1], t[2]) for t in transactions])
    # replace None with the expected value (convert to a list to replace)
    weights_list = list(weights)
    weights_list[weights.index(None)] = 99.0
    weights = tuple(weights_list)
    expected_edgelist = cudf.DataFrame({"src": srcs, "dst": dsts, "weights": weights})
    expected_edgelist = expected_edgelist.sort_values(by="src", ignore_index=True)

    actual_edgelist = G.unrenumber(G.edgelist.edgelist_df, "src", preserve_order=True)
    actual_edgelist = G.unrenumber(actual_edgelist, "dst", preserve_order=True)
    actual_edgelist = actual_edgelist.sort_values(by="src", ignore_index=True)

    assert G.is_directed()
    assert_frame_equal(expected_edgelist, actual_edgelist, check_like=True)


def test_extract_subgraph_default_edge_weight_no_property(dataset1_PropertyGraph):
    """
    Ensure default_edge_weight can be used to provide an edge value when a
    property for the edge weight is not specified.
    """
    (pG, data) = dataset1_PropertyGraph
    edge_weight = 99.2
    G = pG.extract_subgraph(default_edge_weight=edge_weight)
    assert (G.edgelist.edgelist_df["weights"] == edge_weight).all()


def test_extract_subgraph_nonrenumbered_noedgedata():
    """
    Ensure a subgraph can be extracted that is not renumbered and contains no
    edge_data.
    """
    from cugraph.experimental import PropertyGraph
    from cugraph import Graph

    pG = PropertyGraph()
    df = cudf.DataFrame(
        {
            "src": [99, 98, 97],
            "dst": [22, 34, 56],
            "some_property": ["a", "b", "c"],
        }
    )
    pG.add_edge_data(df, vertex_col_names=("src", "dst"))

    G = pG.extract_subgraph(
        create_using=Graph(directed=True), renumber_graph=False, add_edge_data=False
    )

    expected_edgelist = cudf.DataFrame(
        {
            "src": [99, 98, 97],
            "dst": [22, 34, 56],
        }
    )
    assert_frame_equal(
        expected_edgelist.sort_values(by="src", ignore_index=True),
        G.edgelist.edgelist_df.sort_values(by="src", ignore_index=True),
    )
    assert hasattr(G, "edge_data") is False


def test_graph_edge_data_added(dataset1_PropertyGraph):
    """
    Ensures the subgraph returned from extract_subgraph() has the edge_data
    attribute added which contains the proper edge IDs.
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    eicn = PropertyGraph.edge_id_col_name

    expected_num_edges = (
        len(dataset1["transactions"][-1])
        + len(dataset1["relationships"][-1])
        + len(dataset1["referrals"][-1])
    )

    assert pG.get_num_edges() == expected_num_edges
    assert pG.get_num_edges("transactions") == len(dataset1["transactions"][-1])
    assert pG.get_num_edges("relationships") == len(dataset1["relationships"][-1])
    assert pG.get_num_edges("referrals") == len(dataset1["referrals"][-1])
    assert pG.get_num_edges("unknown_type") == 0

    # extract_subgraph() should return a directed Graph object with additional
    # meta-data, which includes edge IDs.
    G = pG.extract_subgraph(create_using=DiGraph_inst, check_multi_edges=False)

    # G.edge_data should be set to a DataFrame with rows for each graph edge.
    assert len(G.edge_data) == expected_num_edges
    edge_ids = sorted(G.edge_data[eicn].values)

    assert edge_ids[0] == 0
    assert edge_ids[-1] == (expected_num_edges - 1)


def test_annotate_dataframe(dataset1_PropertyGraph):
    """
    FIXME: Add tests for:
    properties list
    properties list with 1 or more bad props
    copy=False
    invalid args raise correct exceptions
    """
    (pG, data) = dataset1_PropertyGraph

    selection = pG.select_edges("(_TYPE_ == 'referrals') & (stars > 3)")
    G = pG.extract_subgraph(selection=selection, create_using=DiGraph_inst)

    df_type = type(pG._edge_prop_dataframe)
    # Create an arbitrary DataFrame meant to represent an algo result,
    # containing vertex IDs present in pG.
    #
    # Drop duplicate edges since actual results from a Graph object would not
    # have them.
    (srcs, dsts, mids, stars) = zip(*(dataset1["referrals"][1]))
    algo_result = df_type({"from": srcs, "to": dsts, "result": range(len(srcs))})
    algo_result.drop_duplicates(subset=["from", "to"], inplace=True, ignore_index=True)

    new_algo_result = pG.annotate_dataframe(
        algo_result, G, edge_vertex_col_names=("from", "to")
    )
    expected_algo_result = df_type(
        {
            "from": srcs,
            "to": dsts,
            "result": range(len(srcs)),
            "merchant_id": mids,
            "stars": stars,
        }
    )
    # The integer dtypes of annotated properties are nullable integer dtypes,
    # so convert for proper comparison.
    expected_algo_result["merchant_id"] = expected_algo_result["merchant_id"].astype(
        "Int64"
    )
    expected_algo_result["stars"] = expected_algo_result["stars"].astype("Int64")

    expected_algo_result.drop_duplicates(
        subset=["from", "to"], inplace=True, ignore_index=True
    )

    if df_type is cudf.DataFrame:
        ase = assert_series_equal
    else:
        ase = pd.testing.assert_series_equal
    # For now, the result will include extra columns from edge types not
    # included in the df being annotated, so just check for known columns.
    for col in ["from", "to", "result", "merchant_id", "stars"]:
        ase(new_algo_result[col], expected_algo_result[col])


def test_different_vertex_edge_input_dataframe_types():
    """
    Ensures that a PropertyGraph initialized with one DataFrame type cannot be
    extended with another.
    """
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()
    pG.add_vertex_data(df, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_edge_data(pdf, type_name="bar", vertex_col_names=("a", "b"))

    pG = PropertyGraph()
    pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))

    # Different order
    pG = PropertyGraph()
    pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))
    with pytest.raises(TypeError):
        pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")

    # Same API call, different types
    pG = PropertyGraph()
    pG.add_vertex_data(df, type_name="foo", vertex_col_name="a")
    with pytest.raises(TypeError):
        pG.add_vertex_data(pdf, type_name="foo", vertex_col_name="a")

    pG = PropertyGraph()
    pG.add_edge_data(df, type_name="bar", vertex_col_names=("a", "b"))
    with pytest.raises(TypeError):
        pG.add_edge_data(pdf, type_name="bar", vertex_col_names=("a", "b"))


def test_get_vertices(dataset1_PropertyGraph):
    """
    Test that get_vertices() returns the correct set of vertices without
    duplicates.
    """
    (pG, data) = dataset1_PropertyGraph

    (
        merchants,
        users,
        taxpayers,
        transactions,
        relationships,
        referrals,
    ) = dataset1.values()

    expected_vertices = set(
        [t[0] for t in merchants[1]]
        + [t[0] for t in users[1]]
        + [t[0] for t in taxpayers[1]]
    )

    assert sorted(pG.get_vertices().values) == sorted(expected_vertices)


def test_get_edges(dataset1_PropertyGraph):
    """
    Test that get_edges() returns the correct set of edges (as src/dst
    columns).
    """
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph

    (
        merchants,
        users,
        taxpayers,
        transactions,
        relationships,
        referrals,
    ) = dataset1.values()

    expected_edges = (
        [(src, dst) for (src, dst, _, _, _, _) in transactions[1]]
        + [(src, dst) for (src, dst, _) in relationships[1]]
        + [(src, dst) for (src, dst, _, _) in referrals[1]]
    )

    actual_edges = pG.edges

    assert len(expected_edges) == len(actual_edges)
    for i in range(len(expected_edges)):
        src = actual_edges[PropertyGraph.src_col_name].iloc[i]
        dst = actual_edges[PropertyGraph.dst_col_name].iloc[i]
        assert (src, dst) in expected_edges


def test_property_names_attrs(dataset1_PropertyGraph):
    """
    Ensure the correct number of user-visible properties for vertices and edges
    are returned. This should exclude the internal bookkeeping properties.
    """
    (pG, data) = dataset1_PropertyGraph

    # _VERTEX_ columns: "merchant_id", "user_id"
    expected_vert_prop_names = [
        "merchant_location",
        "merchant_size",
        "merchant_sales",
        "merchant_num_employees",
        "user_location",
        "merchant_name",
        "vertical",
    ]
    # _SRC_ and _DST_ columns: "user_id", "user_id_1", "user_id_2"
    # Note that "merchant_id" is a property in for type "transactions"
    expected_edge_prop_names = [
        "merchant_id",
        "volume",
        "time",
        "card_num",
        "card_type",
        "relationship_type",
        "stars",
    ]

    # Extracting a subgraph with weights has/had a side-effect of adding a
    # weight column, so call extract_subgraph() to ensure the internal weight
    # column name is not present.
    pG.extract_subgraph(default_edge_weight=1.0)

    actual_vert_prop_names = pG.vertex_property_names
    actual_edge_prop_names = pG.edge_property_names

    assert sorted(actual_vert_prop_names) == sorted(expected_vert_prop_names)
    assert sorted(actual_edge_prop_names) == sorted(expected_edge_prop_names)


@pytest.mark.skip(reason="unfinished")
def test_extract_subgraph_with_vertex_ids():
    """
    FIXME: add a PropertyGraph API that makes it easy to support the common use
    case of extracting a subgraph containing only specific vertex IDs. This is
    currently done in the bench_extract_subgraph_for_* tests below, but could
    be made easier for users to do.
    """
    raise NotImplementedError


def test_get_data_empty_graphs():
    """
    Ensures that calls to pG.get_*_data() on an empty pG are handled correctly.
    """
    from cugraph.experimental import PropertyGraph

    pG = PropertyGraph()

    assert pG.get_vertex_data() is None
    assert pG.get_vertex_data([0, 1, 2]) is None
    assert pG.get_edge_data() is None
    assert pG.get_edge_data([0, 1, 2]) is None


@pytest.mark.parametrize("prev_id_column", [None, "prev_id"])
def test_renumber_vertices_by_type(dataset1_PropertyGraph, prev_id_column):
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    with pytest.raises(ValueError, match="existing column"):
        pG.renumber_vertices_by_type("merchant_size")
    df_id_ranges = pG.renumber_vertices_by_type(prev_id_column)
    expected = {
        "merchants": [0, 4],  # stop is inclusive
        "users": [5, 8],
    }
    for key, (start, stop) in expected.items():
        assert df_id_ranges.loc[key, "start"] == start
        assert df_id_ranges.loc[key, "stop"] == stop
        df = pG.get_vertex_data(types=[key])
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()
        assert len(df) == stop - start + 1

        assert df["_VERTEX_"].tolist() == list(range(start, stop + 1))
        if prev_id_column is not None:
            cur = df[prev_id_column].sort_values()
            expected = sorted(x for x, *args in data[key][1])
            assert cur.tolist() == expected
    # Make sure we renumber vertex IDs in edge data too
    df = pG.get_edge_data()
    assert 0 <= df[pG.src_col_name].min() < df[pG.src_col_name].max() < 9
    assert 0 <= df[pG.dst_col_name].min() < df[pG.dst_col_name].max() < 9

    empty_pG = PropertyGraph()
    assert empty_pG.renumber_vertices_by_type(prev_id_column) is None

    # Test when vertex IDs only exist in edge data
    df = type(df)({"src": [99998], "dst": [99999]})
    empty_pG.add_edge_data(df, ["src", "dst"])
    with pytest.raises(NotImplementedError, match="only exist in edge"):
        empty_pG.renumber_vertices_by_type(prev_id_column)


@pytest.mark.parametrize("prev_id_column", [None, "prev_id"])
def test_renumber_edges_by_type(dataset1_PropertyGraph, prev_id_column):
    from cugraph.experimental import PropertyGraph

    (pG, data) = dataset1_PropertyGraph
    with pytest.raises(ValueError, match="existing column"):
        pG.renumber_edges_by_type("time")
    df_id_ranges = pG.renumber_edges_by_type(prev_id_column)
    expected = {
        "transactions": [0, 3],  # stop is inclusive
        "relationships": [4, 7],
        "referrals": [8, 13],
        # Results are no longer alphabetical b/c use of categoricals for types
        # "referrals": [0, 5],  # stop is inclusive
        # "relationships": [6, 9],
        # "transactions": [10, 13],
    }
    for key, (start, stop) in expected.items():
        assert df_id_ranges.loc[key, "start"] == start
        assert df_id_ranges.loc[key, "stop"] == stop
        df = pG.get_edge_data(types=[key])
        if isinstance(df, cudf.DataFrame):
            df = df.to_pandas()

        assert len(df) == stop - start + 1
        assert df[pG.edge_id_col_name].tolist() == list(range(start, stop + 1))
        if prev_id_column is not None:
            assert prev_id_column in df.columns

    empty_pG = PropertyGraph()
    assert empty_pG.renumber_edges_by_type(prev_id_column) is None


def test_renumber_vertices_edges_dtypes():
    from cugraph.experimental import PropertyGraph

    edgelist_df = cudf.DataFrame(
        {
            "src": cp.array([0, 5, 2, 3, 4, 3], dtype="int32"),
            "dst": cp.array([2, 4, 4, 5, 1, 2], dtype="int32"),
            "eid": cp.array([8, 7, 5, 2, 9, 1], dtype="int32"),
        }
    )

    vertex_df = cudf.DataFrame(
        {"v": cp.array([0, 1, 2, 3, 4, 5], dtype="int32"), "p": [5, 10, 15, 20, 25, 30]}
    )

    pG = PropertyGraph()
    pG.add_vertex_data(vertex_df, vertex_col_name="v", property_columns=["p"])
    pG.add_edge_data(
        edgelist_df, vertex_col_names=["src", "dst"], edge_id_col_name="eid"
    )

    pG.renumber_vertices_by_type()
    vd = pG.get_vertex_data()
    assert vd.index.dtype == cp.int32

    pG.renumber_edges_by_type()
    ed = pG.get_edge_data()
    assert ed.index.dtype == cp.int32


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_add_data_noncontiguous(df_type):
    from cugraph.experimental import PropertyGraph

    df = df_type(
        {
            "src": [0, 0, 1, 2, 2, 3, 3, 1, 2, 4],
            "dst": [1, 2, 4, 3, 3, 1, 2, 4, 4, 3],
            "edge_type": [
                "pig",
                "dog",
                "cat",
                "pig",
                "cat",
                "pig",
                "dog",
                "pig",
                "cat",
                "dog",
            ],
        }
    )
    counts = df["edge_type"].value_counts()

    pG = PropertyGraph()
    for edge_type in ["cat", "dog", "pig"]:
        pG.add_edge_data(
            df[df.edge_type == edge_type],
            vertex_col_names=["src", "dst"],
            type_name=edge_type,
        )
    if df_type is cudf.DataFrame:
        ase = assert_series_equal
    else:
        ase = pd.testing.assert_series_equal
    for edge_type in ["cat", "dog", "pig"]:
        cur_df = pG.get_edge_data(types=edge_type)
        assert len(cur_df) == counts[edge_type]
        ase(
            cur_df[pG.type_col_name].astype(str),
            cur_df["edge_type"],
            check_names=False,
        )

    df["vertex"] = (
        100 * df["src"]
        + df["dst"]
        + df["edge_type"].map({"pig": 0, "dog": 10, "cat": 20})
    )
    pG = PropertyGraph()
    for edge_type in ["cat", "dog", "pig"]:
        pG.add_vertex_data(
            df[df.edge_type == edge_type], vertex_col_name="vertex", type_name=edge_type
        )
    for edge_type in ["cat", "dog", "pig"]:
        cur_df = pG.get_vertex_data(types=edge_type)
        assert len(cur_df) == counts[edge_type]
        ase(
            cur_df[pG.type_col_name].astype(str),
            cur_df["edge_type"],
            check_names=False,
        )


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_vertex_ids_different_type(df_type):
    """Getting the number of vertices requires combining vertex ids from
    multiple columns.

    This test ensures combining these columns works even if they are different types.
    """
    from cugraph.experimental import PropertyGraph

    if df_type is pd.DataFrame:
        series_type = pd.Series
    else:
        series_type = cudf.Series
    pg = PropertyGraph()
    node_df = df_type()
    node_df["node_id"] = series_type([0, 1, 2]).astype("int32")
    pg.add_vertex_data(node_df, "node_id", type_name="_N")

    edge_df = df_type()
    edge_df["src"] = series_type([0, 1, 2]).astype("int32")
    edge_df["dst"] = series_type([0, 1, 2]).astype("int64")
    pg.add_edge_data(edge_df, ["src", "dst"], type_name="_E")

    assert pg.get_num_vertices() == 3


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_vertex_vector_property(df_type):
    from cugraph.experimental import PropertyGraph

    (
        merchants,
        users,
        taxpayers,
        transactions,
        relationships,
        referrals,
    ) = dataset1.values()
    if df_type is cudf.DataFrame:
        assert_array_equal = cp.testing.assert_array_equal
        zeros = cp.zeros
    else:
        assert_array_equal = np.testing.assert_array_equal
        zeros = np.zeros

    pG = PropertyGraph()
    merchants_df = df_type(columns=merchants[0], data=merchants[1])
    with pytest.raises(ValueError):
        # Column doesn't exist
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": ["merchant_location", "BAD_NAME"]},
        )
    with pytest.raises(ValueError):
        # Using reserved name
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={
                pG.type_col_name: ["merchant_location", "merchant_size"]
            },
        )
    with pytest.raises(TypeError):
        # String value invalid
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": "merchant_location"},
        )
    with pytest.raises(ValueError):
        # Length-0 vector not allowed
        pG.add_vertex_data(
            merchants_df,
            type_name="merchants",
            vertex_col_name="merchant_id",
            vector_properties={"vec1": []},
        )
    pG.add_vertex_data(
        merchants_df,
        type_name="merchants",
        vertex_col_name="merchant_id",
        vector_properties={
            "vec1": ["merchant_location", "merchant_size", "merchant_num_employees"]
        },
    )
    df = pG.get_vertex_data()
    expected_columns = {
        pG.vertex_col_name,
        pG.type_col_name,
        "merchant_sales",
        "merchant_name",
        "vec1",
    }
    assert set(df.columns) == expected_columns
    expected = merchants_df[
        ["merchant_location", "merchant_size", "merchant_num_employees"]
    ].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort

    vec1 = pG.vertex_vector_property_to_array(df, "vec1")
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    vec1 = pG.vertex_vector_property_to_array(df, "vec1", missing="error")
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    with pytest.raises(ValueError):
        pG.vertex_vector_property_to_array(df, "BAD_NAME")

    users_df = df_type(columns=users[0], data=users[1])
    with pytest.raises(ValueError):
        # Length doesn't match existing vector
        pG.add_vertex_data(
            users_df,
            type_name="users",
            vertex_col_name="user_id",
            property_columns=["vertical"],
            vector_properties={"vec1": ["user_location", "vertical"]},
        )
    with pytest.raises(ValueError):
        # Can't assign property to existing vector column
        pG.add_vertex_data(
            users_df.assign(vec1=users_df["user_id"]),
            type_name="users",
            vertex_col_name="user_id",
            property_columns=["vec1"],
        )

    pG.add_vertex_data(
        users_df,
        type_name="users",
        vertex_col_name="user_id",
        property_columns=["vertical"],
        vector_properties={"vec2": ["user_location", "vertical"]},
    )
    expected_columns.update({"vec2", "vertical"})
    df = pG.get_vertex_data()
    assert set(df.columns) == expected_columns
    vec1 = pG.vertex_vector_property_to_array(df, "vec1")
    vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec1)
    with pytest.raises(RuntimeError):
        pG.vertex_vector_property_to_array(df, "vec1", missing="error")

    pGusers = PropertyGraph()
    pGusers.add_vertex_data(
        users_df,
        type_name="users",
        vertex_col_name="user_id",
        vector_property="vec3",
    )
    vec2 = pG.vertex_vector_property_to_array(df, "vec2")
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    df2 = pGusers.get_vertex_data()
    assert set(df2.columns) == {pG.vertex_col_name, pG.type_col_name, "vec3"}
    vec3 = pGusers.vertex_vector_property_to_array(df2, "vec3")
    vec3 = vec3[np.lexsort(vec3.T)]  # may be jumbled, so sort
    assert_array_equal(vec2, vec3)

    vec1filled = pG.vertex_vector_property_to_array(df, "vec1", 0, missing="error")
    vec1filled = vec1filled[np.lexsort(vec1filled.T)]  # may be jumbled, so sort
    expectedfilled = np.concatenate([zeros((4, 3), int), expected])
    assert_array_equal(expectedfilled, vec1filled)

    vec1filled = pG.vertex_vector_property_to_array(df, "vec1", [0, 0, 0])
    vec1filled = vec1filled[np.lexsort(vec1filled.T)]  # may be jumbled, so sort
    assert_array_equal(expectedfilled, vec1filled)

    with pytest.raises(ValueError, match="expected 3"):
        pG.vertex_vector_property_to_array(df, "vec1", [0, 0])

    vec2 = pG.vertex_vector_property_to_array(df, "vec2")
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    expected = users_df[["user_location", "vertical"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort
    assert_array_equal(expected, vec2)
    with pytest.raises(TypeError):
        # Column is wrong type to be a vector
        pG.vertex_vector_property_to_array(
            df.rename(columns={"vec1": "vertical", "vertical": "vec1"}), "vec1"
        )
    with pytest.raises(ValueError):
        # Vector column doesn't exist in dataframe
        pG.vertex_vector_property_to_array(df.rename(columns={"vec1": "moved"}), "vec1")
    with pytest.raises(TypeError):
        # Bad type
        pG.vertex_vector_property_to_array(42, "vec1")


@pytest.mark.parametrize("df_type", df_types, ids=df_type_id)
def test_edge_vector_property(df_type):
    from cugraph.experimental import PropertyGraph

    if df_type is cudf.DataFrame:
        assert_array_equal = cp.testing.assert_array_equal
    else:
        assert_array_equal = np.testing.assert_array_equal
    df1 = df_type(
        {
            "src": [0, 1],
            "dst": [1, 2],
            "feat_0": [1, 2],
            "feat_1": [10, 20],
            "feat_2": [10, 20],
        }
    )
    df2 = df_type(
        {
            "src": [2, 3],
            "dst": [1, 2],
            "feat_0": [0.5, 0.2],
            "feat_1": [1.5, 1.2],
        }
    )
    pG = PropertyGraph()
    pG.add_edge_data(
        df1, ("src", "dst"), vector_properties={"vec1": ["feat_0", "feat_1", "feat_2"]}
    )
    df = pG.get_edge_data()
    expected_columns = {
        pG.edge_id_col_name,
        pG.src_col_name,
        pG.dst_col_name,
        pG.type_col_name,
        "vec1",
    }
    assert set(df.columns) == expected_columns
    expected = df1[["feat_0", "feat_1", "feat_2"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort

    pGalt = PropertyGraph()
    pGalt.add_edge_data(df1, ("src", "dst"), vector_property="vec1")
    dfalt = pG.get_edge_data()

    for cur_pG, cur_df in [(pG, df), (pGalt, dfalt)]:
        vec1 = cur_pG.edge_vector_property_to_array(cur_df, "vec1")
        vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
        assert_array_equal(vec1, expected)
        vec1 = cur_pG.edge_vector_property_to_array(cur_df, "vec1", missing="error")
        vec1 = vec1[np.lexsort(vec1.T)]  # may be jumbled, so sort
        assert_array_equal(vec1, expected)

    pG.add_edge_data(
        df2, ("src", "dst"), vector_properties={"vec2": ["feat_0", "feat_1"]}
    )
    df = pG.get_edge_data()
    expected_columns.add("vec2")
    assert set(df.columns) == expected_columns
    expected = df2[["feat_0", "feat_1"]].values
    expected = expected[np.lexsort(expected.T)]  # may be jumbled, so sort
    vec2 = pG.edge_vector_property_to_array(df, "vec2")
    vec2 = vec2[np.lexsort(vec2.T)]  # may be jumbled, so sort
    assert_array_equal(vec2, expected)
    with pytest.raises(RuntimeError):
        pG.edge_vector_property_to_array(df, "vec2", missing="error")


@pytest.mark.skip(reason="feature not implemented")
def test_single_csv_multi_vertex_edge_attrs():
    """
    Read an edgelist CSV that contains both edge and vertex attrs
    """
    pass


def test_fillna_vertices():
    from cugraph.experimental import PropertyGraph

    df_edgelist = cudf.DataFrame(
        {
            "src": [0, 7, 2, 0, 1, 3, 1, 4, 5, 6],
            "dst": [1, 1, 1, 3, 2, 1, 6, 5, 6, 7],
            "val": [1, None, 2, None, 3, None, 4, None, 5, None],
        }
    )

    df_props = cudf.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "a": [0, 1, None, 2, None, 4, 1, 8],
            "b": [None, 1, None, 2, None, 3, 8, 9],
        }
    )

    pG = PropertyGraph()
    pG.add_edge_data(df_edgelist, vertex_col_names=["src", "dst"])
    pG.add_vertex_data(df_props, vertex_col_name="id")

    pG.fillna_vertices({"a": 2, "b": 3})

    assert not pG.get_vertex_data(columns=["a", "b"]).isna().any().any()
    assert pG.get_edge_data(columns=["val"]).isna().any().any()

    expected_values_prop_a = [
        0,
        1,
        2,
        2,
        2,
        4,
        1,
        8,
    ]
    assert pG.get_vertex_data(columns=["a"])["a"].values_host.tolist() == (
        expected_values_prop_a
    )

    expected_values_prop_b = [
        3,
        1,
        3,
        2,
        3,
        3,
        8,
        9,
    ]
    assert pG.get_vertex_data(columns=["b"])["b"].values_host.tolist() == (
        expected_values_prop_b
    )


def test_fillna_edges():
    from cugraph.experimental import PropertyGraph

    df_edgelist = cudf.DataFrame(
        {
            "src": [0, 7, 2, 0, 1, 3, 1, 4, 5, 6],
            "dst": [1, 1, 1, 3, 2, 1, 6, 5, 6, 7],
            "val": [1, None, 2, None, 3, None, 4, None, 5, None],
        }
    )

    df_props = cudf.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "a": [0, 1, None, 2, None, 4, 1, 8],
            "b": [None, 1, None, 2, None, 3, 8, 9],
        }
    )

    pG = PropertyGraph()
    pG.add_edge_data(df_edgelist, vertex_col_names=["src", "dst"])
    pG.add_vertex_data(df_props, vertex_col_name="id")

    pG.fillna_edges(2)

    assert not pG.get_edge_data(columns=["val"]).isna().any().any()
    assert pG.get_vertex_data(columns=["a", "b"]).isna().any().any()

    expected_values_prop_val = [
        1,
        2,
        2,
        2,
        3,
        2,
        4,
        2,
        5,
        2,
    ]
    assert pG.get_edge_data(columns=["val"])["val"].values_host.tolist() == (
        expected_values_prop_val
    )


def test_types_from_numerals():
    from cugraph.experimental import PropertyGraph

    df_edgelist_cow = cudf.DataFrame(
        {
            "src": [0, 7, 2, 0, 1],
            "dst": [1, 1, 1, 3, 2],
            "val": [1, 3, 2, 3, 3],
        }
    )

    df_edgelist_pig = cudf.DataFrame(
        {
            "src": [3, 1, 4, 5, 6],
            "dst": [1, 6, 5, 6, 7],
            "val": [5, 4, 5, 5, 2],
        }
    )

    df_props_duck = cudf.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "a": [0, 1, 6, 2],
            "b": [2, 1, 2, 2],
        }
    )

    df_props_goose = cudf.DataFrame(
        {
            "id": [4, 5, 6, 7],
            "a": [5, 4, 1, 8],
            "b": [2, 3, 8, 9],
        }
    )

    pG = PropertyGraph()

    pG.add_edge_data(df_edgelist_cow, vertex_col_names=["src", "dst"], type_name="cow")
    pG.add_edge_data(df_edgelist_pig, vertex_col_names=["src", "dst"], type_name="pig")

    pG.add_vertex_data(df_props_duck, vertex_col_name="id", type_name="duck")
    pG.add_vertex_data(df_props_goose, vertex_col_name="id", type_name="goose")

    assert pG.vertex_types_from_numerals(
        cudf.Series([0, 1, 0, 0, 1, 0, 1, 1])
    ).values_host.tolist() == [
        "duck",
        "goose",
        "duck",
        "duck",
        "goose",
        "duck",
        "goose",
        "goose",
    ]
    assert pG.edge_types_from_numerals(
        cudf.Series([1, 1, 0, 1, 1, 0, 0, 1, 1])
    ).values_host.tolist() == [
        "pig",
        "pig",
        "cow",
        "pig",
        "pig",
        "cow",
        "cow",
        "pig",
        "pig",
    ]


# =============================================================================
# Benchmarks
# =============================================================================
def bench_num_vertices(gpubenchmark, dataset1_PropertyGraph):
    (pG, data) = dataset1_PropertyGraph

    assert gpubenchmark(pG.get_num_vertices) == 9


def bench_get_vertices(gpubenchmark, dataset1_PropertyGraph):
    (pG, data) = dataset1_PropertyGraph

    gpubenchmark(pG.get_vertices)


def bench_extract_subgraph_for_cyber(gpubenchmark, cyber_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    pG = cyber_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    # Create a Graph containing only specific src or dst vertices
    verts = ["10.40.182.3", "10.40.182.255", "59.166.0.9", "59.166.0.8"]
    selected_edges = pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")
    gpubenchmark(
        pG.extract_subgraph,
        create_using=cugraph.Graph(directed=True),
        selection=selected_edges,
        default_edge_weight=1.0,
        check_multi_edges=False,
    )


def bench_extract_subgraph_for_cyber_detect_duplicate_edges(
    gpubenchmark, cyber_PropertyGraph
):
    from cugraph.experimental import PropertyGraph

    pG = cyber_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    # Create a Graph containing only specific src or dst vertices
    verts = ["10.40.182.3", "10.40.182.255", "59.166.0.9", "59.166.0.8"]
    selected_edges = pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")

    def func():
        with pytest.raises(RuntimeError):
            pG.extract_subgraph(
                create_using=cugraph.Graph(directed=True),
                selection=selected_edges,
                default_edge_weight=1.0,
                check_multi_edges=True,
            )

    gpubenchmark(func)


def bench_extract_subgraph_for_rmat(gpubenchmark, rmat_PropertyGraph):
    from cugraph.experimental import PropertyGraph

    (pG, generated_df) = rmat_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    verts = []
    for i in range(0, 10000, 10):
        verts.append(generated_df["src"].iloc[i])

    selected_edges = pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")
    gpubenchmark(
        pG.extract_subgraph,
        create_using=cugraph.Graph(directed=True),
        selection=selected_edges,
        default_edge_weight=1.0,
        check_multi_edges=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize("n_rows", [15_000_000, 30_000_000, 60_000_000, 120_000_000])
def bench_add_edge_data(gpubenchmark, n_rows):
    from cugraph.experimental import PropertyGraph

    def func():
        pg = PropertyGraph()
        src = cp.arange(n_rows)
        dst = src - 1
        df = cudf.DataFrame({"src": src, "dst": dst})
        pg.add_edge_data(df, ["src", "dst"], type_name="('_N', '_E', '_N')")

    gpubenchmark(func)


# This test runs for *minutes* with the current implementation, and since
# benchmarking can call it multiple times per run, the overall time for this
# test can be ~20 minutes.
@pytest.mark.slow
def bench_extract_subgraph_for_rmat_detect_duplicate_edges(
    gpubenchmark, rmat_PropertyGraph
):
    from cugraph.experimental import PropertyGraph

    (pG, generated_df) = rmat_PropertyGraph
    scn = PropertyGraph.src_col_name
    dcn = PropertyGraph.dst_col_name

    verts = []
    for i in range(0, 10000, 10):
        verts.append(generated_df["src"].iloc[i])

    selected_edges = pG.select_edges(f"{scn}.isin({verts}) | {dcn}.isin({verts})")

    def func():
        with pytest.raises(RuntimeError):
            pG.extract_subgraph(
                create_using=cugraph.Graph(directed=True),
                selection=selected_edges,
                default_edge_weight=1.0,
                check_multi_edges=True,
            )

    gpubenchmark(func)


@pytest.mark.slow
@pytest.mark.parametrize("N", [1, 3, 10, 30])
def bench_add_edges_cyber(gpubenchmark, N):
    from cugraph.experimental import PropertyGraph

    # Partition the dataframe to add in chunks
    cyber_df = cyber.get_edgelist()
    chunk = (len(cyber_df) + N - 1) // N
    dfs = [cyber_df.iloc[i * chunk : (i + 1) * chunk] for i in range(N)]

    def func():
        pG = PropertyGraph()
        for df in dfs:
            pG.add_edge_data(df, ("srcip", "dstip"))
        df = pG.get_edge_data()
        assert len(df) == len(cyber_df)

    gpubenchmark(func)


# @pytest.mark.slow
@pytest.mark.parametrize("n_rows", [10_000, 100_000, 1_000_000, 10_000_000])
@pytest.mark.parametrize("n_feats", [32, 64, 128])
def bench_add_vector_features(gpubenchmark, n_rows, n_feats):
    from cugraph.experimental import PropertyGraph

    df = cudf.DataFrame(
        {
            "src": cp.arange(0, n_rows, dtype=cp.int32),
            "dst": cp.arange(0, n_rows, dtype=cp.int32) + 1,
        }
    )
    for i in range(n_feats):
        df[f"feat_{i}"] = cp.ones(len(df), dtype=cp.int32)

    vector_properties = {"feat": [f"feat_{i}" for i in range(n_feats)]}

    def func():
        pG = PropertyGraph()
        pG.add_edge_data(
            df, vertex_col_names=["src", "dst"], vector_properties=vector_properties
        )

    gpubenchmark(func)


@pytest.mark.parametrize("n_rows", [1_000_000])
@pytest.mark.parametrize("n_feats", [128])
def bench_get_vector_features_cp_array(benchmark, n_rows, n_feats):
    from cugraph.experimental import PropertyGraph

    df = cudf.DataFrame(
        {
            "src": cp.arange(0, n_rows, dtype=cp.int32),
            "dst": cp.arange(0, n_rows, dtype=cp.int32) + 1,
        }
    )
    for i in range(n_feats):
        df[f"feat_{i}"] = cp.ones(len(df), dtype=cp.int32)

    vector_properties = {"feat": [f"feat_{i}" for i in range(n_feats)]}
    pG = PropertyGraph()
    pG.add_edge_data(
        df, vertex_col_names=["src", "dst"], vector_properties=vector_properties
    )
    benchmark(pG.get_edge_data, edge_ids=cp.arange(0, 100_000))


@pytest.mark.parametrize("n_rows", [1_000_000])
@pytest.mark.parametrize("n_feats", [128])
def bench_get_vector_features_cudf_series(benchmark, n_rows, n_feats):
    from cugraph.experimental import PropertyGraph

    df = cudf.DataFrame(
        {
            "src": cp.arange(0, n_rows, dtype=cp.int32),
            "dst": cp.arange(0, n_rows, dtype=cp.int32) + 1,
        }
    )
    for i in range(n_feats):
        df[f"feat_{i}"] = cp.ones(len(df), dtype=cp.int32)

    vector_properties = {"feat": [f"feat_{i}" for i in range(n_feats)]}
    pG = PropertyGraph()
    pG.add_edge_data(
        df, vertex_col_names=["src", "dst"], vector_properties=vector_properties
    )
    benchmark(pG.get_edge_data, edge_ids=cudf.Series(cp.arange(0, 100_000)))

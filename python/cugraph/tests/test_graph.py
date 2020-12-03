# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import time

import pandas as pd
import pytest

import scipy
import cudf
from cudf.tests.utils import assert_eq
import cugraph
from cugraph.tests import utils

# MG
import cugraph.dask as dcg
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def compare_series(series_1, series_2):
    assert len(series_1) == len(series_2)
    df = cudf.DataFrame({"series_1": series_1, "series_2": series_2})
    diffs = df.query('series_1 != series_2')

    if len(diffs) > 0:
        print("diffs:\n", diffs)

    assert len(diffs) == 0


# This function returns True if two graphs are identical (bijection between the
# vertices in one graph to the vertices in the other graph is identity AND two
# graphs are automorphic; no permutations of vertices are allowed).
def compare_graphs(nx_graph, cu_graph):
    edgelist_df = cu_graph.view_edge_list().reset_index(drop=True)

    df = cudf.DataFrame()
    df["source"] = edgelist_df["src"]
    df["target"] = edgelist_df["dst"]
    if len(edgelist_df.columns) > 2:
        df["weight"] = edgelist_df["weights"]
        cu_to_nx_graph = nx.from_pandas_edgelist(
            df.to_pandas(),
            source="source",
            target="target",
            edge_attr=["weight"],
            create_using=nx.DiGraph(),
        )
    else:
        cu_to_nx_graph = nx.from_pandas_edgelist(
            df.to_pandas(), create_using=nx.DiGraph()
        )

    # first compare nodes
    ds0 = pd.Series(list(nx_graph.nodes)).sort_values(ignore_index=True)
    ds1 = pd.Series(list(cu_to_nx_graph.nodes)).sort_values(ignore_index=True)

    if not ds0.equals(ds1):
        print('ds0 != ds1')
        return False

    # second compare edges
    diff = nx.difference(nx_graph, cu_to_nx_graph)

    if diff.number_of_edges() > 0:
        print('diff.number_of_edges = ', diff.number_of_edges())
        return False

    diff = nx.difference(cu_to_nx_graph, nx_graph)
    if diff.number_of_edges() > 0:
        print('2: diff.number_of_edges = ', diff.number_of_edges())
        return False

    if len(edgelist_df.columns) > 2:
        df0 = cudf.from_pandas(nx.to_pandas_edgelist(nx_graph))
        merge = df.merge(df0, on=["source", "target"],
                         suffixes=("_cugraph", "_nx"))
        print("merge = \n", merge)
        print(merge[merge.weight_cugraph != merge.weight_nx])
        if not merge["weight_cugraph"].equals(merge["weight_nx"]):
            print('weights different')
            print(merge[merge.weight_cugraph != merge.weight_nx])
            return False

    return True


def find_two_paths(df, M):
    for i in range(len(df)):
        start = df["first"][i]
        end = df["second"][i]
        foundPath = False
        for idx in range(M.indptr[start], M.indptr[start + 1]):
            mid = M.indices[idx]
            for innerIdx in range(M.indptr[mid], M.indptr[mid + 1]):
                if M.indices[innerIdx] == end:
                    foundPath = True
                    break
            if foundPath:
                break
        if not foundPath:
            print("No path found between " + str(start) + " and " + str(end))
        assert foundPath


def has_pair(first_arr, second_arr, first, second):
    for i in range(len(first_arr)):
        firstMatch = first_arr[i] == first
        secondMatch = second_arr[i] == second
        if firstMatch and secondMatch:
            return True
    return False


def check_all_two_hops(df, M):
    num_verts = len(M.indptr) - 1
    first_arr = df["first"].to_array()
    second_arr = df["second"].to_array()
    for start in range(num_verts):
        for idx in range(M.indptr[start], M.indptr[start + 1]):
            mid = M.indices[idx]
            for innerIdx in range(M.indptr[mid], M.indptr[mid + 1]):
                end = M.indices[innerIdx]
                if start != end:
                    assert has_pair(first_arr, second_arr, start, end)


def test_version():
    cugraph.__version__


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_add_edge_list_to_adj_list(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    M = utils.read_csv_for_nx(graph_file)
    N = max(max(M["0"]), max(M["1"])) + 1
    M = scipy.sparse.csr_matrix((M.weight, (M["0"], M["1"])), shape=(N, N))
    offsets_exp = M.indptr
    indices_exp = M.indices

    # cugraph add_egde_list to_adj_list call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=False)
    offsets_cu, indices_cu, values_cu = G.view_adj_list()
    compare_series(offsets_cu, offsets_exp)
    compare_series(indices_cu, indices_exp)
    assert values_cu is None


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_add_adj_list_to_edge_list(graph_file):
    Mnx = utils.read_csv_for_nx(graph_file)
    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    Mcsr = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )

    offsets = cudf.Series(Mcsr.indptr)
    indices = cudf.Series(Mcsr.indices)

    Mcoo = Mcsr.tocoo()
    sources_exp = cudf.Series(Mcoo.row)
    destinations_exp = cudf.Series(Mcoo.col)

    # cugraph add_adj_list to_edge_list call
    G = cugraph.DiGraph()
    G.from_cudf_adjlist(offsets, indices, None)
    edgelist = G.view_edge_list()
    sources_cu = edgelist["src"]
    destinations_cu = edgelist["dst"]
    compare_series(sources_cu, sources_exp)
    compare_series(destinations_cu, destinations_exp)


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_view_edge_list_from_adj_list(graph_file):
    Mnx = utils.read_csv_for_nx(graph_file)
    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    Mcsr = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )

    offsets = cudf.Series(Mcsr.indptr)
    indices = cudf.Series(Mcsr.indices)
    G = cugraph.DiGraph()
    G.from_cudf_adjlist(offsets, indices, None)
    edgelist_df = G.view_edge_list()
    Mcoo = Mcsr.tocoo()
    src1 = Mcoo.row
    dst1 = Mcoo.col
    compare_series(src1, edgelist_df["src"])
    compare_series(dst1, edgelist_df["dst"])


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_delete_edge_list_delete_adj_list(graph_file):
    Mnx = utils.read_csv_for_nx(graph_file)
    df = cudf.DataFrame()
    df["src"] = cudf.Series(Mnx["0"])
    df["dst"] = cudf.Series(Mnx["1"])

    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    Mcsr = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )
    offsets = cudf.Series(Mcsr.indptr)
    indices = cudf.Series(Mcsr.indices)

    # cugraph delete_adj_list delete_edge_list call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    G.delete_edge_list()
    with pytest.raises(Exception):
        G.view_adj_list()

    G.from_cudf_adjlist(offsets, indices, None)
    G.delete_adj_list()
    with pytest.raises(Exception):
        G.view_edge_list()


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_add_edge_or_adj_list_after_add_edge_or_adj_list(graph_file):
    Mnx = utils.read_csv_for_nx(graph_file)
    df = cudf.DataFrame()
    df["src"] = cudf.Series(Mnx["0"])
    df["dst"] = cudf.Series(Mnx["1"])

    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    Mcsr = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )

    offsets = cudf.Series(Mcsr.indptr)
    indices = cudf.Series(Mcsr.indices)

    G = cugraph.DiGraph()

    # If cugraph has at least one graph representation, adding a new graph
    # should fail to prevent a single graph object storing two different
    # graphs.

    # If cugraph has a graph edge list, adding a new graph should fail.
    G.from_cudf_edgelist(df, source="src", destination="dst")
    with pytest.raises(Exception):
        G.from_cudf_edgelist(df, source="src", destination="dst")
    with pytest.raises(Exception):
        G.from_cudf_adjlist(offsets, indices, None)
    G.delete_edge_list()

    # If cugraph has a graph adjacency list, adding a new graph should fail.
    G.from_cudf_adjlist(offsets, indices, None)
    with pytest.raises(Exception):
        G.from_cudf_edgelist(df, source="src", destination="dst")
    with pytest.raises(Exception):
        G.from_cudf_adjlist(offsets, indices, None)
    G.delete_adj_list()


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_edges_for_Graph(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    # Create nx Graph
    pdf = cu_M.to_pandas()[['0', '1']]
    nx_graph = nx.from_pandas_edgelist(pdf, source='0',
                                       target='1',
                                       create_using=nx.Graph)
    nx_edges = nx_graph.edges()

    # Create Cugraph Graph from DataFrame
    # Force it to use renumber_from_cudf
    G = cugraph.from_cudf_edgelist(cu_M, source=['0'],
                                   destination=['1'],
                                   create_using=cugraph.Graph)
    cu_edge_list = G.edges()

    # Check if number of Edges is same
    assert len(nx_edges) == len(cu_edge_list)
    assert nx_graph.number_of_edges() == G.number_of_edges()

    # Compare nx and cugraph edges when viewing edgelist
    edges = []
    for edge in nx_edges:
        if edge[0] > edge[1]:
            edges.append([edge[1], edge[0]])
        else:
            edges.append([edge[0], edge[1]])
    nx_edge_list = cudf.DataFrame(list(edges), columns=['src', 'dst'])
    assert_eq(
        nx_edge_list.sort_values(by=['src', 'dst']).reset_index(drop=True),
        cu_edge_list.sort_values(by=['src', 'dst']).reset_index(drop=True),
        check_dtype=False
    )


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_view_edge_list_for_Graph(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    # Create nx Graph
    pdf = cu_M.to_pandas()[["0", "1"]]
    nx_graph = nx.from_pandas_edgelist(
        pdf, source="0", target="1", create_using=nx.Graph
    )
    nx_edges = nx_graph.edges()

    # Create Cugraph Graph from DataFrame
    G = cugraph.from_cudf_edgelist(
        cu_M, source="0", destination="1", create_using=cugraph.Graph
    )
    cu_edge_list = G.view_edge_list().sort_values(["src", "dst"])

    # Check if number of Edges is same
    assert len(nx_edges) == len(cu_edge_list)
    assert nx_graph.number_of_edges() == G.number_of_edges()

    # Get edges as upper triangle of matrix
    edges = []
    for edge in nx_edges:
        if edge[0] > edge[1]:
            edges.append([edge[1], edge[0]])
        else:
            edges.append([edge[0], edge[1]])
    edges = list(edges)
    edges.sort()
    nx_edge_list = cudf.DataFrame(edges, columns=["src", "dst"])

    # Compare nx and cugraph edges when viewing edgelist
    # assert cu_edge_list.equals(nx_edge_list)
    assert (
        cu_edge_list["src"].to_array() == nx_edge_list["src"].to_array()
    ).all()
    assert (
        cu_edge_list["dst"].to_array() == nx_edge_list["dst"].to_array()
    ).all()


# Test
@pytest.mark.parametrize('graph_file', utils.DATASETS)
def test_consolidation(graph_file):
    cluster = LocalCUDACluster()
    client = Client(cluster)
    chunksize = dcg.get_chunksize(graph_file)

    M = utils.read_csv_for_nx(graph_file)

    df = pd.DataFrame()
    df['source'] = pd.Series(M['0'])
    df['target'] = pd.Series(M['1'])

    ddf = dask_cudf.read_csv(graph_file, chunksize=chunksize,
                             delimiter=' ',
                             names=['source', 'target', 'weight'],
                             dtype=['int32', 'int32', 'float32'], header=None)

    Gnx = nx.from_pandas_edgelist(df, source='source', target='target',
                                  create_using=nx.DiGraph)
    G = cugraph.from_cudf_edgelist(ddf, source='source', destination='target',
                                   create_using=cugraph.DiGraph)

    t1 = time.time()
    assert compare_graphs(Gnx, G)
    t2 = time.time() - t1
    print('compare_graphs time: ', t2)

    Gnx.clear()
    G.clear()
    client.close()
    cluster.close()


# Test
@pytest.mark.parametrize('graph_file', utils.DATASETS_SMALL)
def test_two_hop_neighbors(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    df = G.get_two_hop_neighbors()
    Mnx = utils.read_csv_for_nx(graph_file)
    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    Mcsr = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )

    find_two_paths(df, Mcsr)
    check_all_two_hops(df, Mcsr)


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_degree_functionality(graph_file):
    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    df_in_degree = G.in_degree()
    df_out_degree = G.out_degree()
    df_degree = G.degree()

    nx_in_degree = Gnx.in_degree()
    nx_out_degree = Gnx.out_degree()
    nx_degree = Gnx.degree()

    err_in_degree = 0
    err_out_degree = 0
    err_degree = 0
    for i in range(len(df_degree)):
        in_deg = df_in_degree["degree"][i]
        out_deg = df_out_degree["degree"][i]
        if in_deg != nx_in_degree[df_in_degree["vertex"][i]]:
            err_in_degree = err_in_degree + 1
        if out_deg != nx_out_degree[df_out_degree["vertex"][i]]:
            err_out_degree = err_out_degree + 1
        if df_degree["degree"][i] != nx_degree[df_degree["vertex"][i]]:
            err_degree = err_degree + 1
    assert err_in_degree == 0
    assert err_out_degree == 0
    assert err_degree == 0


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_degrees_functionality(graph_file):
    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    df = G.degrees()

    nx_in_degree = Gnx.in_degree()
    nx_out_degree = Gnx.out_degree()

    err_in_degree = 0
    err_out_degree = 0

    for i in range(len(df)):
        if df["in_degree"][i] != nx_in_degree[df["vertex"][i]]:
            err_in_degree = err_in_degree + 1
        if df["out_degree"][i] != nx_out_degree[df["vertex"][i]]:
            err_out_degree = err_out_degree + 1

    assert err_in_degree == 0
    assert err_out_degree == 0


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_number_of_vertices(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    M = utils.read_csv_for_nx(graph_file)
    if M is None:
        raise TypeError("Could not read the input graph")

    # cugraph add_edge_list
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )
    assert G.number_of_vertices() == Gnx.number_of_nodes()


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
def test_to_directed(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    cu_M = cu_M[cu_M["0"] <= cu_M["1"]].reset_index(drop=True)
    M = utils.read_csv_for_nx(graph_file)
    M = M[M["0"] <= M["1"]]
    assert len(cu_M) == len(M)

    # cugraph add_edge_list
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )

    DiG = G.to_directed()
    DiGnx = Gnx.to_directed()

    assert DiG.number_of_nodes() == DiGnx.number_of_nodes()
    assert DiG.number_of_edges() == DiGnx.number_of_edges()

    for index, row in cu_M.to_pandas().iterrows():
        assert G.has_edge(row['0'], row['1'])
        assert G.has_edge(row['1'], row['0'])


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS_SMALL)
def test_to_undirected(graph_file):
    # Read data and then convert to directed by dropped some edges
    cu_M = utils.read_csv_file(graph_file)
    cu_M = cu_M[cu_M["0"] <= cu_M["1"]].reset_index(drop=True)

    M = utils.read_csv_for_nx(graph_file)
    M = M[M["0"] <= M["1"]]
    assert len(cu_M) == len(M)

    # cugraph add_edge_list
    DiG = cugraph.DiGraph()
    DiG.from_cudf_edgelist(cu_M, source="0", destination="1")

    DiGnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    for index, row in cu_M.to_pandas().iterrows():
        assert DiG.has_edge(row['0'], row['1'])
        assert not DiG.has_edge(row['1'], row['0'])

    G = DiG.to_undirected()
    Gnx = DiGnx.to_undirected()

    assert G.number_of_nodes() == Gnx.number_of_nodes()
    assert G.number_of_edges() == Gnx.number_of_edges()

    for index, row in cu_M.to_pandas().iterrows():
        assert G.has_edge(row['0'], row['1'])
        assert G.has_edge(row['1'], row['0'])


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_has_edge(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    cu_M = cu_M[cu_M["0"] <= cu_M["1"]].reset_index(drop=True)

    # cugraph add_edge_list
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    for index, row in cu_M.to_pandas().iterrows():
        assert G.has_edge(row['0'], row['1'])
        assert G.has_edge(row['1'], row['0'])


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_has_node(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    nodes = cudf.concat([cu_M["0"], cu_M["1"]]).unique()

    # cugraph add_edge_list
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    for n in nodes.values_host:
        assert G.has_node(n)


def test_invalid_has_node():
    df = cudf.DataFrame([[1, 2]], columns=["src", "dst"])
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source="src", destination="dst")
    assert not G.has_node(-1)
    assert not G.has_node(0)
    assert not G.has_node(G.number_of_nodes() + 1)


@pytest.mark.parametrize('graph_file', utils.DATASETS)
def test_bipartite_api(graph_file):
    # This test only tests the functionality of adding set of nodes and
    # retrieving them. The datasets currently used are not truly bipartite.
    cu_M = utils.read_csv_file(graph_file)
    nodes = cudf.concat([cu_M['0'], cu_M['1']]).unique()

    # Create set of nodes for partition
    set1_exp = cudf.Series(nodes[0:int(len(nodes)/2)])
    set2_exp = cudf.Series(set(nodes.values_host)
                           - set(set1_exp.values_host))

    G = cugraph.Graph()
    assert not G.is_bipartite()

    # Add a set of nodes present in one partition
    G.add_nodes_from(set1_exp, bipartite='set1')
    G.from_cudf_edgelist(cu_M, source='0', destination='1')

    # Check if Graph is bipartite. It should return True since we have
    # added the partition in add_nodes_from()
    assert G.is_bipartite()

    # Call sets() to get the bipartite set of nodes.
    set1, set2 = G.sets()

    # assert if the input set1_exp is same as returned bipartite set1
    assert set1.equals(set1_exp)
    # assert if set2 is the remaining set of nodes not in set1_exp
    assert set2.equals(set2_exp)


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_neighbors(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    nodes = cudf.concat([cu_M["0"], cu_M["1"]]).unique()
    M = utils.read_csv_for_nx(graph_file)

    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  create_using=nx.Graph())
    for n in nodes.values_host:
        cu_neighbors = G.neighbors(n).to_arrow().to_pylist()
        nx_neighbors = [i for i in Gnx.neighbors(n)]
        cu_neighbors.sort()
        nx_neighbors.sort()
        assert cu_neighbors == nx_neighbors

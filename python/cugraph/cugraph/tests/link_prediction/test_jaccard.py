# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import pytest

import cudf
from cudf.testing import assert_series_equal, assert_frame_equal

import cugraph
from cugraph.testing import utils
from cugraph.experimental import jaccard_coefficient as exp_jaccard_coefficient
from cugraph.experimental import jaccard as exp_jaccard
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, netscience

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def compare_jaccard_two_hop(G, Gnx, edgevals=True):
    """
    Compute both cugraph and nx jaccard after extracting the two hop neighbors
    from G and compare both results
    """
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    nx_pairs = list(pairs.to_records(index=False))
    preds = nx.jaccard_coefficient(Gnx, nx_pairs)
    nx_coeff = []
    for u, v, p in preds:
        # print(u, " ", v, " ", p)
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=["first", "second"]).reset_index(drop=True)
    if not edgevals:
        # experimental jaccard currently only supports unweighted graphs
        df_exp = exp_jaccard(G, pairs)
        df_exp = df_exp.sort_values(by=["first", "second"]).reset_index(drop=True)
        assert_frame_equal(df, df_exp, check_dtype=False, check_like=True)

    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df["jaccard_coeff"].iloc[i])
        assert diff < 1.0e-6


def cugraph_call(benchmark_callable, graph_file, edgevals=False, input_df=None):
    G = cugraph.Graph()
    G = graph_file.get_graph(ignore_weights=not edgevals)

    # If no vertex_pair is passed as input, 'cugraph.jaccard' will
    # compute the 'jaccard_similarity' with the two_hop_neighbor of the
    # entire graph while nx compute with the one_hop_neighbor. For better
    # comparaison, get the one_hop_neighbor of the entire graph for 'cugraph.jaccard'
    # and pass it as vertex_pair
    vertex_pair = input_df.rename(columns={"0": "first", "1": "second"})
    vertex_pair = vertex_pair[["first", "second"]]

    # cugraph Jaccard Call
    df = benchmark_callable(cugraph.jaccard, G, vertex_pair=vertex_pair)

    df = df.sort_values(["first", "second"]).reset_index(drop=True)

    return (
        df["first"].to_numpy(),
        df["second"].to_numpy(),
        df["jaccard_coeff"].to_numpy(),
    )


def networkx_call(M, benchmark_callable=None):

    sources = M["0"]
    destinations = M["1"]
    edges = []
    for i in range(len(M)):
        edges.append((sources[i], destinations[i]))
        edges.append((destinations[i], sources[i]))
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print("Format conversion ... ")

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    # Networkx Jaccard Call
    print("Solving... ")
    if benchmark_callable is not None:
        preds = benchmark_callable(nx.jaccard_coefficient, Gnx, edges)
    else:
        preds = nx.jaccard_coefficient(Gnx, edges)
    src = []
    dst = []
    coeff = []
    for u, v, p in preds:
        src.append(u)
        dst.append(v)
        coeff.append(p)
    return src, dst, coeff


# =============================================================================
# Pytest Fixtures
# =============================================================================
@pytest.fixture(scope="module", params=DATASETS_UNDIRECTED)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    graph_file = request.param
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)

    return M_cu, M, graph_file


def test_jaccard(read_csv, gpubenchmark):

    M_cu, M, graph_file = read_csv
    cu_src, cu_dst, cu_coeff = cugraph_call(gpubenchmark, graph_file, input_df=M_cu)
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


def test_directed_graph_check(read_csv):
    _, M, _ = read_csv

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = cugraph.Graph(directed=True)
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]
    with pytest.raises(ValueError):
        cugraph.jaccard(G1, vertex_pair)


def test_nx_jaccard_time(read_csv, gpubenchmark):

    _, M, _ = read_csv
    nx_src, nx_dst, nx_coeff = networkx_call(M, gpubenchmark)


@pytest.mark.parametrize("graph_file", [netscience])
def test_jaccard_edgevals(gpubenchmark, graph_file):
    dataset_path = netscience.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)
    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, netscience, edgevals=True, input_df=M_cu
    )
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


def test_jaccard_two_hop(read_csv):

    _, M, graph_file = read_csv

    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())
    G = graph_file.get_graph(ignore_weights=True)

    compare_jaccard_two_hop(G, Gnx)


def test_jaccard_two_hop_edge_vals(read_csv):

    _, M, graph_file = read_csv

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    G = graph_file.get_graph()

    compare_jaccard_two_hop(G, Gnx, edgevals=True)


def test_jaccard_nx(read_csv):

    M_cu, M, _ = read_csv
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())

    nx_j = nx.jaccard_coefficient(Gnx)
    nv_js = sorted(nx_j, key=len, reverse=True)

    ebunch = M_cu.rename(columns={"0": "first", "1": "second"})
    ebunch = ebunch[["first", "second"]]
    cg_j = cugraph.jaccard_coefficient(Gnx, ebunch=ebunch)
    cg_j_exp = exp_jaccard_coefficient(Gnx, ebunch=ebunch)

    assert len(nv_js) > len(cg_j)
    assert len(nv_js) > len(cg_j_exp)

    # FIXME:  Nx does a full all-pair Jaccard.
    # cuGraph does a limited 1-hop Jaccard
    # assert nx_j == cg_j


def test_jaccard_multi_column(read_csv):

    _, M, _ = read_csv

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]

    df_res = cugraph.jaccard(G1, vertex_pair)
    df_plc_exp = exp_jaccard(G1, vertex_pair)

    df_plc_exp = df_plc_exp.rename(
        columns={
            "0_src": "0_source",
            "0_dst": "0_destination",
            "1_src": "1_source",
            "1_dst": "1_destination",
        }
    )

    jaccard_res = df_res["jaccard_coeff"].sort_values().reset_index(drop=True)
    jaccard_plc_exp = df_plc_exp["jaccard_coeff"].sort_values().reset_index(drop=True)
    assert_series_equal(jaccard_res, jaccard_plc_exp)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")
    df_exp = cugraph.jaccard(G2, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_first").reset_index()
    expected = df_exp.sort_values("first").reset_index()
    assert_series_equal(actual["jaccard_coeff"], expected["jaccard_coeff"])


def test_weighted_exp_jaccard():
    karate = DATASETS_UNDIRECTED[0]
    G = karate.get_graph()
    with pytest.raises(ValueError):
        exp_jaccard(G)

    G = karate.get_graph(ignore_weights=True)
    use_weight = True
    with pytest.raises(ValueError):
        exp_jaccard(G, use_weight=use_weight)

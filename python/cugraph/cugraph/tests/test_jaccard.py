# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
from cudf.testing import assert_series_equal

import cugraph
from cugraph.tests import utils

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
def compare_jaccard_two_hop(G, Gnx):
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
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=["source", "destination"]).reset_index(drop=True)
    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df["jaccard_coeff"].iloc[i])
        assert diff < 1.0e-6


def cugraph_call(benchmark_callable, cu_M, edgevals=False):
    G = cugraph.Graph()
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")

    # cugraph Jaccard Call
    df = benchmark_callable(cugraph.jaccard, G)

    df = df.sort_values(["source", "destination"]).reset_index(drop=True)

    return (
        df["source"].to_numpy(),
        df["destination"].to_numpy(),
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
@pytest.fixture(scope="module", params=utils.DATASETS_UNDIRECTED)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    M = utils.read_csv_for_nx(request.param)
    cu_M = utils.read_csv_file(request.param)

    return M, cu_M


def test_jaccard(read_csv, gpubenchmark):

    M, cu_M = read_csv
    cu_src, cu_dst, cu_coeff = cugraph_call(gpubenchmark, cu_M)
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


def test_nx_jaccard_time(read_csv, gpubenchmark):

    M, _ = read_csv
    nx_src, nx_dst, nx_coeff = networkx_call(M, gpubenchmark)


@pytest.mark.parametrize(
    "graph_file",
    [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"netscience.csv"]
)
def test_jaccard_edgevals(gpubenchmark, graph_file):

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_src, cu_dst, cu_coeff = cugraph_call(gpubenchmark, cu_M, edgevals=True)
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

    M, cu_M = read_csv

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    compare_jaccard_two_hop(G, Gnx)


def test_jaccard_two_hop_edge_vals(read_csv):

    M, cu_M = read_csv

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    compare_jaccard_two_hop(G, Gnx)


def test_jaccard_nx(read_csv):

    M, _ = read_csv
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )

    nx_j = nx.jaccard_coefficient(Gnx)
    nv_js = sorted(nx_j, key=len, reverse=True)

    cg_j = cugraph.jaccard_coefficient(Gnx)

    assert len(nv_js) > len(cg_j)

    # FIXME:  Nx does a full all-pair Jaccard.
    # cuGraph does a limited 1-hop Jaccard
    # assert nx_j == cg_j


def test_jaccard_multi_column(read_csv):

    M, _ = read_csv

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(cu_M, source=["src_0", "src_1"],
                          destination=["dst_0", "dst_1"])

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]

    df_res = cugraph.jaccard(G1, vertex_pair)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0",
                          destination="dst_0")
    df_exp = cugraph.jaccard(G2, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_source").reset_index()
    expected = df_exp.sort_values("source").reset_index()
    assert_series_equal(actual["jaccard_coeff"], expected["jaccard_coeff"])

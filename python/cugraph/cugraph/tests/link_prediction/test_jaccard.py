# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

# FIXME: Can we use global variables for column names instead of hardcoded ones?

import gc

import pytest
import networkx as nx

import cudf
import cugraph
from cugraph.datasets import netscience
from cugraph.testing import utils, UNDIRECTED_DATASETS
from cudf.testing import assert_series_equal

# from cugraph import jaccard_coefficient


print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================


def compare_jaccard_two_hop(G, Gnx, use_weight=False):
    """
    Compute both cugraph and nx jaccard after extracting the two hop neighbors
    from G and compare both results
    """
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=["first", "second"]).reset_index(drop=True)

    if not use_weight:
        nx_pairs = list(pairs.to_records(index=False))
        preds = nx.jaccard_coefficient(Gnx, nx_pairs)
        nx_coeff = []
        for u, v, p in preds:
            nx_coeff.append(p)

        assert len(nx_coeff) == len(df)
        for i in range(len(df)):
            diff = abs(nx_coeff[i] - df["jaccard_coeff"].iloc[i])
            assert diff < 1.0e-6
    else:
        # FIXME: compare results against resultset api
        pass


def cugraph_call(benchmark_callable, graph_file, input_df=None, use_weight=False):
    G = cugraph.Graph()
    G = graph_file.get_graph(ignore_weights=not use_weight)

    # If no vertex_pair is passed as input, 'cugraph.jaccard' will
    # compute the 'jaccard_similarity' with the two_hop_neighbor of the
    # entire graph while nx compute with the one_hop_neighbor. For better
    # comparaison, get the one_hop_neighbor of the entire graph for 'cugraph.jaccard'
    # and pass it as vertex_pair
    if isinstance(input_df, cudf.DataFrame):
        vertex_pair = input_df.rename(columns={"0": "first", "1": "second"})
        vertex_pair = vertex_pair[["first", "second"]]
    else:
        vertex_pair = cudf.DataFrame(
            columns=["first", "second"], dtype=G.edgelist.edgelist_df["src"].dtype
        )

    # cugraph Jaccard Call
    df = benchmark_callable(
        cugraph.jaccard, G, vertex_pair=vertex_pair, use_weight=use_weight
    )

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
@pytest.fixture(scope="module", params=UNDIRECTED_DATASETS)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    graph_file = request.param
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)

    return M_cu, M, graph_file


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_jaccard(read_csv, gpubenchmark, use_weight):
    M_cu, M, graph_file = read_csv
    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, graph_file, input_df=M_cu, use_weight=use_weight
    )
    if not use_weight:
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
    else:
        # FIXME: compare weighted jaccard results against resultset api
        pass


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_directed_graph_check(read_csv, use_weight):
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
        cugraph.jaccard(G1, vertex_pair, use_weight)


@pytest.mark.sg
def test_nx_jaccard_time(read_csv, gpubenchmark):
    _, M, _ = read_csv
    nx_src, nx_dst, nx_coeff = networkx_call(M, gpubenchmark)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", [netscience])
@pytest.mark.parametrize("use_weight", [False, True])
def test_jaccard_edgevals(gpubenchmark, graph_file, use_weight):
    dataset_path = netscience.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)
    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, netscience, input_df=M_cu, use_weight=use_weight
    )
    if not use_weight:
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
    else:
        # FIXME: compare results against resultset api
        pass


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_jaccard_two_hop(read_csv, use_weight):
    _, M, graph_file = read_csv

    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())
    G = graph_file.get_graph(ignore_weights=not use_weight)

    compare_jaccard_two_hop(G, Gnx, use_weight)


@pytest.mark.sg
def test_jaccard_nx(read_csv):
    M_cu, M, _ = read_csv
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())

    nx_j = nx.jaccard_coefficient(Gnx)
    nv_js = sorted(nx_j, key=len, reverse=True)

    ebunch = M_cu.rename(columns={"0": "first", "1": "second"})
    ebunch = ebunch[["first", "second"]]
    cg_j = cugraph.jaccard_coefficient(Gnx, ebunch=ebunch)

    assert len(nv_js) > len(cg_j)

    # FIXME:  Nx does a full all-pair Jaccard.
    # cuGraph does a limited 1-hop Jaccard
    # assert nx_j == cg_j


@pytest.mark.sg
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

    df_multi_col_res = cugraph.jaccard(G1, vertex_pair)
    # jaccard_multi_col_res = (
    #     df_multi_col_res["jaccard_coeff"].sort_values().reset_index(drop=True)
    # )

    print(df_multi_col_res)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")
    df_single_col_res = cugraph.jaccard(G2, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_multi_col_res.sort_values("0_src").reset_index()
    expected = df_single_col_res.sort_values("first").reset_index()

    assert_series_equal(actual["jaccard_coeff"], expected["jaccard_coeff"])

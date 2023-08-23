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

import gc

import pytest
import networkx as nx
import numpy as np

import cudf
import cugraph
from cugraph.testing import utils

from cudf.testing import assert_series_equal, assert_frame_equal

from cugraph.datasets import karate
from cugraph.experimental import jaccard as exp_jaccard
from cugraph.experimental import jaccard_coefficient as exp_jaccard_coefficient

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


def cugraph_call(
    benchmark_callable,
    graph_dataset,
    input_df,
    edgevals=False,
):
    G = graph_dataset.get_graph(ignore_weights=not edgevals)

    # If no vertex_pair is passed as input, 'cugraph.jaccard' will
    # compute the 'jaccard_similarity' with the two_hop_neighbor of the
    # entire graph while nx compute with the one_hop_neighbor. For better
    # comparaison, we use the one-hop edges list

    # cugraph Jaccard Call
    df = benchmark_callable(cugraph.jaccard, G, vertex_pair=input_df)
    df = df.sort_values(["first", "second"]).reset_index(drop=True)

    return (
        df["first"].to_numpy(),
        df["second"].to_numpy(),
        df["jaccard_coeff"].to_numpy(),
    )


def networkx_call(M, is_symmetric=False, benchmark_callable=None):

    sources = M["src"]
    destinations = M["dst"]
    edges = []
    for i in range(len(M)):
        edges.append((sources[i], destinations[i]))
        if is_symmetric is False:
            edges.append((destinations[i], sources[i]))

    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)

    Gnx = nx.from_pandas_edgelist(
        M, source="src", target="dst", edge_attr="wgt", create_using=nx.Graph()
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
# Pytest
# =============================================================================
@pytest.mark.sg
def test_jaccard(undirected_datasets, gpubenchmark):
    M_cu, M, graph_dataset = undirected_datasets
    symmetric = graph_dataset.metadata.get("is_symmetric")
    cu_src, cu_dst, cu_coeff = cugraph_call(gpubenchmark, graph_dataset, input_df=M_cu)
    nx_src, nx_dst, nx_coeff = networkx_call(M, is_symmetric=symmetric)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


@pytest.mark.sg
def test_directed_graph(undirected_datasets):
    M_cu, M, graph_file = undirected_datasets

    G1 = cugraph.Graph(directed=True)
    G1.from_cudf_edgelist(M_cu, source=["src"], destination=["dst"])

    with pytest.raises(ValueError):
        cugraph.jaccard(G1, M)


@pytest.mark.sg
def test_nx_jaccard_time(undirected_datasets, gpubenchmark):
    M_cu, M, graph_dataset = undirected_datasets
    symmetric = graph_dataset.metadata.get("is_symmetric")
    nx_src, nx_dst, nx_coeff = networkx_call(
        M, is_symmetric=symmetric, benchmark_callable=gpubenchmark
    )


@pytest.mark.sg
def test_jaccard_edgevals(undirected_datasets, gpubenchmark):
    M_cu, M, graph_dataset = undirected_datasets

    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, graph_dataset, input_df=M_cu, edgevals=True
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


@pytest.mark.sg
def test_jaccard_two_hop(undirected_datasets):
    M_cu, M, graph_file = undirected_datasets

    Gnx = nx.from_pandas_edgelist(
        M, source="src", target="dst", create_using=nx.Graph()
    )
    G = graph_file.get_graph(ignore_weights=True)

    compare_jaccard_two_hop(G, Gnx)


@pytest.mark.sg
def test_jaccard_two_hop_edge_vals(undirected_datasets):
    _, M, graph_file = undirected_datasets

    Gnx = nx.from_pandas_edgelist(
        M, source="src", target="dst", edge_attr="wgt", create_using=nx.Graph()
    )

    G = graph_file.get_graph()

    compare_jaccard_two_hop(G, Gnx, edgevals=True)


@pytest.mark.sg
def test_jaccard_nx(undirected_datasets):

    M_cu, M, _ = undirected_datasets
    Gnx = nx.from_pandas_edgelist(
        M, source="src", target="dst", create_using=nx.Graph()
    )

    nx_j = nx.jaccard_coefficient(Gnx)
    nv_js = sorted(nx_j, key=len, reverse=True)

    ebunch = M_cu.rename(columns={"src": "first", "dst": "second"})
    ebunch = ebunch[["first", "second"]]
    cg_j = cugraph.jaccard_coefficient(Gnx, ebunch=ebunch)
    cg_j_exp = exp_jaccard_coefficient(Gnx, ebunch=ebunch)

    assert len(nv_js) > len(cg_j)
    assert len(nv_js) > len(cg_j_exp)

    # FIXME:  Nx does a full all-pair Jaccard.
    # cuGraph does a limited 1-hop Jaccard
    # assert nx_j == cg_j


@pytest.mark.sg
def test_jaccard_multi_column(undirected_datasets):

    _, M, _ = undirected_datasets

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["src"])
    cu_M["dst_0"] = cudf.Series(M["dst"])
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


@pytest.mark.sg
def test_weighted_exp_jaccard():
    G = karate.get_graph(ignore_weights=False)
    with pytest.raises(ValueError):
        exp_jaccard(G)

    G = karate.get_graph(ignore_weights=True)
    use_weight = True
    with pytest.raises(ValueError):
        exp_jaccard(G, use_weight=use_weight)


@pytest.mark.sg
def test_invalid_datasets_jaccard():
    df = karate.get_edgelist()
    df = df.add(1)
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(df, source="src", destination="dst")
    with pytest.raises(ValueError):
        cugraph.jaccard(G)


TYPES = [np.dtype(str), np.dtype(float)]


@pytest.mark.sg
@pytest.mark.parametrize("type", TYPES)
def test_str_datasets_jaccard(type):
    df = karate.get_edgelist()
    df2 = utils.convert_edges_to(df, type, inplace=False)

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(df2, source="src", destination="dst", renumber=True)

    cugraph.jaccard(G, df, do_expensive_check=False)

# Copyright (c) 2022, NVIDIA CORPORATION.
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

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import (
    toy_graph,
    karate,
    DATASETS_UNDIRECTED,
    DATASETS,
)

import networkx as nx

# This toy graph is used in multiple tests throughout libcugraph_c and pylib.
TOY = toy_graph


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def topKVertices(eigen, col, k):
    top = eigen.nlargest(n=k, columns=col)
    top = top.sort_values(by=col, ascending=False)
    return top["vertex"]


def calc_eigenvector(graph_file):
    dataset_path = graph_file.get_path()
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=True), ignore_weights=True
    )

    k_df = cugraph.eigenvector_centrality(G, max_iter=1000)
    k_df = k_df.sort_values("vertex").reset_index(drop=True)

    NM = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(NM, create_using=nx.DiGraph(), source="0", target="1")
    nk = nx.eigenvector_centrality(Gnx)
    pdf = [nk[k] for k in sorted(nk.keys())]
    k_df["nx_eigen"] = pdf
    k_df = k_df.rename(columns={"eigenvector_centrality": "cu_eigen"}, copy=False)
    return k_df


@pytest.mark.parametrize("graph_file", DATASETS)
def test_eigenvector_centrality(graph_file):
    eigen_scores = calc_eigenvector(graph_file)

    topKNX = topKVertices(eigen_scores, "nx_eigen", 10)
    topKCU = topKVertices(eigen_scores, "cu_eigen", 10)

    assert topKNX.equals(topKCU)


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_eigenvector_centrality_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)

    Gnx = nx.from_pandas_edgelist(
        NM,
        create_using=nx.DiGraph(),
        source="0",
        target="1",
    )

    nk = nx.eigenvector_centrality(Gnx)
    ck = cugraph.eigenvector_centrality(Gnx)

    # Calculating mismatch
    nk = sorted(nk.items(), key=lambda x: x[0])
    ck = sorted(ck.items(), key=lambda x: x[0])
    err = 0
    assert len(ck) == len(nk)
    for i in range(len(ck)):
        if abs(ck[i][1] - nk[i][1]) > 0.1 and ck[i][0] == nk[i][0]:
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.1 * len(ck))


# TODO: Uncomment this test when/if nstart is supported for eigen centrality
"""
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_eigenvector_centrality_multi_column(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    cu_M.rename(columns={'0': 'src_0', '1': 'dst_0'}, inplace=True)
    cu_M['src_1'] = cu_M['src_0'] + 1000
    cu_M['dst_1'] = cu_M['dst_0'] + 1000

    G1 = cugraph.Graph(directed=True)
    G1.from_cudf_edgelist(cu_M, source=["src_0", "src_1"],
                          destination=["dst_0", "dst_1"],
                          store_transposed=True)

    G2 = cugraph.Graph(directed=True)
    G2.from_cudf_edgelist(
        cu_M, source="src_0", destination="dst_0", store_transposed=True)

    k_df_exp = cugraph.eigenvector_centrality(G2)
    k_df_exp = k_df_exp.sort_values("vertex").reset_index(drop=True)

    nstart = cudf.DataFrame()
    nstart['vertex_0'] = k_df_exp['vertex']
    nstart['vertex_1'] = nstart['vertex_0'] + 1000
    nstart['values'] = k_df_exp['eigenvector_centrality']

    k_df_res = cugraph.eigenvector_centrality(G1, nstart=nstart)
    k_df_res = k_df_res.sort_values("0_vertex").reset_index(drop=True)
    k_df_res.rename(columns={'0_vertex': 'vertex'}, inplace=True)

    top_res = topKVertices(k_df_res, "eigenvector_centrality", 10)
    top_exp = topKVertices(k_df_exp, "eigenvector_centrality", 10)

    assert top_res.equals(top_exp)
"""


@pytest.mark.parametrize("graph_file", [TOY])
def test_eigenvector_centrality_toy(graph_file):
    # This test is based off of libcugraph_c and pylibcugraph tests
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))

    tol = 1e-6
    max_iter = 200
    centralities = [0.236325, 0.292055, 0.458457, 0.60533, 0.190498, 0.495942]

    ck = cugraph.eigenvector_centrality(G, tol=tol, max_iter=max_iter)

    ck = ck.sort_values("vertex")
    for vertex in ck["vertex"].to_pandas():
        expected_score = centralities[vertex]
        actual_score = ck["eigenvector_centrality"].iloc[vertex]
        assert pytest.approx(expected_score, abs=1e-4) == actual_score, (
            f"Eigenvector centrality score is {actual_score}, should have"
            f" been {expected_score}"
        )


def test_eigenvector_centrality_transposed_false():
    G = karate.get_graph(create_using=cugraph.Graph(directed=True))
    warning_msg = (
        "Eigenvector centrality expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        cugraph.eigenvector_centrality(G)

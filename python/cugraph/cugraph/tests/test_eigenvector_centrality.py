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

import cudf
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

# This toy graph is used in multiple tests throughout libcugraph_c and pylib.
TOY_DATASET = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"toy_graph.csv"


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
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    k_df = cugraph.eigenvector_centrality(G, max_iter=1000)
    k_df = k_df.sort_values("vertex").reset_index(drop=True)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, create_using=nx.DiGraph(), source="0", target="1"
    )
    nk = nx.eigenvector_centrality(Gnx)
    pdf = [nk[k] for k in sorted(nk.keys())]
    k_df["nx_eigen"] = pdf
    k_df = k_df.rename(columns={"eigenvector_centrality": "cu_eigen"},
                       copy=False)
    return k_df


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_eigenvector_centrality(graph_file):
    eigen_scores = calc_eigenvector(graph_file)

    topKNX = topKVertices(eigen_scores, "nx_eigen", 10)
    topKCU = topKVertices(eigen_scores, "cu_eigen", 10)

    assert topKNX.equals(topKCU)


# def test_katz_centrality_nx(graph_file):

# def test_katz_centrality_multi_column(graph_file):

@pytest.mark.parametrize("graph_file", [TOY_DATASET])
def test_eigenvector_centrality_toy(graph_file):
    # This test is based off of libcugraph_c and pylibcugraph tests
    df = cudf.read_csv(TOY_DATASET, delimiter=' ',
                       dtype=['int32', 'int32', 'float32'], header=None)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source='0', destination='1', edge_attr='2')

    tol = 0.000001
    max_iter = 1000
    centralities = [0, 0, 0, 0, 0, 0]

    ck = cugraph.eigenvector_centrality(G, tol=tol, max_iter=max_iter)

    ck = ck.sort_values("vertex")
    for vertex in ck["vertex"].to_pandas():
        expected_score = centralities[vertex]
        actual_score = ck["eigenvector_centrality"].iloc[vertex]
        assert pytest.approx(expected_score, abs=1e-2) == actual_score, \
            f"Eigenvector centrality score is {actual_score}, should have" \
            f"been {expected_score}"

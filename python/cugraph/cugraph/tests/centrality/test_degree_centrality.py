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
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED

import networkx as nx


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def topKVertices(degree, col, k):
    top = degree.nlargest(n=k, columns=col)
    top = top.sort_values(by=col, ascending=False)
    return top["vertex"]


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_degree_centrality_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)
    Gnx = nx.from_pandas_edgelist(
        NM,
        create_using=nx.DiGraph(),
        source="0",
        target="1",
    )

    G = cugraph.utilities.convert_from_nx(Gnx)

    nk = nx.degree_centrality(Gnx)
    ck = cugraph.degree_centrality(G)

    # Calculating mismatch
    nk = sorted(nk.items(), key=lambda x: x[0])
    ck = ck.sort_values("vertex")
    ck.index = ck["vertex"]
    ck = ck["degree_centrality"]
    err = 0

    assert len(ck) == len(nk)
    for i in range(len(ck)):
        if abs(ck[i] - nk[i][1]) > 0.1 and ck.index[i] == nk[i][0]:
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.1 * len(ck))


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_degree_centrality_multi_column(graph_file):
    dataset_path = graph_file.get_path()
    cu_M = utils.read_csv_file(dataset_path)
    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G1 = cugraph.Graph(directed=True)
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    G2 = cugraph.Graph(directed=True)
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")

    k_df_exp = cugraph.degree_centrality(G2)
    k_df_exp = k_df_exp.sort_values("vertex").reset_index(drop=True)

    nstart = cudf.DataFrame()
    nstart["vertex_0"] = k_df_exp["vertex"]
    nstart["vertex_1"] = nstart["vertex_0"] + 1000
    nstart["values"] = k_df_exp["degree_centrality"]

    k_df_res = cugraph.degree_centrality(G1)
    k_df_res = k_df_res.sort_values("0_vertex").reset_index(drop=True)
    k_df_res.rename(columns={"0_vertex": "vertex"}, inplace=True)

    top_res = topKVertices(k_df_res, "degree_centrality", 10)
    top_exp = topKVertices(k_df_exp, "degree_centrality", 10)

    assert top_res.equals(top_exp)

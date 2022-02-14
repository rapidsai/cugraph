
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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


print("Networkx version : {} ".format(nx.__version__))


def topKVertices(katz, col, k):
    top = katz.nlargest(n=k, columns=col)
    top = top.sort_values(by=col, ascending=False)
    return top["vertex"]


def calc_katz(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    largest_out_degree = G.degrees().nlargest(n=1, columns="out_degree")
    largest_out_degree = largest_out_degree["out_degree"].iloc[0]
    katz_alpha = 1 / (largest_out_degree + 1)

    k_df = cugraph.katz_centrality(G, alpha=None, max_iter=1000)
    k_df = k_df.sort_values("vertex").reset_index(drop=True)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, create_using=nx.DiGraph(), source="0", target="1"
    )
    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    pdf = [nk[k] for k in sorted(nk.keys())]
    k_df["nx_katz"] = pdf
    k_df = k_df.rename(columns={"katz_centrality": "cu_katz"}, copy=False)
    return k_df


# FIXME: the default set of datasets includes an asymmetric directed graph
# (email-EU-core.csv), which currently produces different results between
# cugraph and Nx and fails that test. Investigate, resolve, and use
# utils.DATASETS instead.
#
# https://github.com/rapidsai/cugraph/issues/1042
#
# @pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_katz_centrality(graph_file):
    gc.collect()

    katz_scores = calc_katz(graph_file)

    topKNX = topKVertices(katz_scores, "nx_katz", 10)
    topKCU = topKVertices(katz_scores, "cu_katz", 10)

    assert topKNX.equals(topKCU)


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_katz_centrality_nx(graph_file):
    gc.collect()

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, create_using=nx.DiGraph(), source="0", target="1"
    )

    G = cugraph.utilities.convert_from_nx(Gnx)
    largest_out_degree = G.degrees().nlargest(n=1, columns="out_degree")
    largest_out_degree = largest_out_degree["out_degree"].iloc[0]
    katz_alpha = 1 / (largest_out_degree + 1)

    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    ck = cugraph.katz_centrality(Gnx, alpha=None, max_iter=1000)

    # Calculating mismatch
    nk = sorted(nk.items(), key=lambda x: x[0])
    ck = sorted(ck.items(), key=lambda x: x[0])
    err = 0
    assert len(ck) == len(nk)
    for i in range(len(ck)):
        if (
            abs(ck[i][1] - nk[i][1]) > 0.1
            and ck[i][0] == nk[i][0]
        ):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.1 * len(ck))


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_katz_centrality_multi_column(graph_file):
    gc.collect()

    cu_M = utils.read_csv_file(graph_file)
    cu_M.rename(columns={'0': 'src_0', '1': 'dst_0'}, inplace=True)
    cu_M['src_1'] = cu_M['src_0'] + 1000
    cu_M['dst_1'] = cu_M['dst_0'] + 1000

    G1 = cugraph.DiGraph()
    G1.from_cudf_edgelist(cu_M, source=["src_0", "src_1"],
                          destination=["dst_0", "dst_1"])

    G2 = cugraph.DiGraph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")

    k_df_exp = cugraph.katz_centrality(G2, alpha=None, max_iter=1000)
    k_df_exp = k_df_exp.sort_values("vertex").reset_index(drop=True)

    nstart = cudf.DataFrame()
    nstart['vertex_0'] = k_df_exp['vertex']
    nstart['vertex_1'] = nstart['vertex_0'] + 1000
    nstart['values'] = k_df_exp['katz_centrality']

    k_df_res = cugraph.katz_centrality(G1, nstart=nstart,
                                       alpha=None, max_iter=1000)
    k_df_res = k_df_res.sort_values("0_vertex").reset_index(drop=True)
    k_df_res.rename(columns={'0_vertex': 'vertex'}, inplace=True)

    top_res = topKVertices(k_df_res, "katz_centrality", 10)
    top_exp = topKVertices(k_df_exp, "katz_centrality", 10)

    assert top_res.equals(top_exp)

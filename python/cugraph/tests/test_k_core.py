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

import pytest

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


def calc_k_cores(graph_file, directed=True):
    # directed is used to create either a Graph or DiGraph so the returned
    # cugraph can be compared to nx graph of same type.
    cu_M = utils.read_csv_file(graph_file)
    NM = utils.read_csv_for_nx(graph_file)
    if directed:
        G = cugraph.DiGraph()
        Gnx = nx.from_pandas_edgelist(
            NM, source="0", target="1", create_using=nx.DiGraph()
        )
    else:
        G = cugraph.Graph()
        Gnx = nx.from_pandas_edgelist(
            NM, source="0", target="1", create_using=nx.Graph()
        )
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    ck = cugraph.k_core(G)
    nk = nx.k_core(Gnx)
    return ck, nk


def compare_edges(cg, nxg):
    edgelist_df = cg.view_edge_list()
    src, dest = edgelist_df["src"], edgelist_df["dst"]

    assert cg.edgelist.weights is False
    assert len(src) == nxg.size()
    for i in range(len(src)):
        assert nxg.has_edge(src[i], dest[i])
    return True


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_core_number_Graph(graph_file):
    gc.collect()

    cu_kcore, nx_kcore = calc_k_cores(graph_file, False)

    assert compare_edges(cu_kcore, nx_kcore)


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_core_number_Graph_nx(graph_file):
    gc.collect()

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        NM, source="0", target="1", create_using=nx.Graph()
    )
    nc = nx.k_core(Gnx)
    cc = cugraph.k_core(Gnx)

    assert nx.is_isomorphic(nc, cc)

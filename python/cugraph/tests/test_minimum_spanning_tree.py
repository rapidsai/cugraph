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
from cugraph.utilities import check_nx_graph
from cugraph.utilities import cugraph_to_nx

import time
import numpy as np

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


def compare_mst(mst_cugraph, mst_nx):
    mst_nx = nx.to_pandas_edgelist(mst_nx)

    edgelist_df = mst_cugraph.view_edge_list()
    src = edgelist_df["src"]
    dst = edgelist_df["dst"]
    wgt = edgelist_df["weights"]
    assert len(edgelist_df) == len(mst_nx)
    for i in range(len(src)):
        has_edge = (
            (mst_nx["source"] == src[i])
            & (mst_nx["target"] == dst[i])
            & np.isclose(mst_nx["weights"], wgt[i])
        ).any()
        has_opp_edge = (
            (mst_nx["source"] == dst[i])
            & (mst_nx["target"] == src[i])
            & np.isclose(mst_nx["weights"], wgt[i])
        ).any()
        assert has_edge or has_opp_edge
    return True


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_minimum_spanning_tree_nx(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    # Just for getting relevant timing
    cuG, isNx = check_nx_graph(G)
    cuG.view_adj_list()

    t1 = time.time()
    cugraph_mst = cugraph.minimum_spanning_tree(cuG)
    t2 = time.time() - t1
    print("CuGraph time : " + str(t2))
    df = nx.to_pandas_edgelist(cugraph_mst)

    t1 = time.time()
    mst_nx = nx.minimum_spanning_tree(G)
    t2 = time.time() - t1
    print("Nx Time : " + str(t2))
    print()

    assert len(df) == len(nx_df)
    compare_mst(cugraph_mst, mst_nx)

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
    mst_nx_df = nx.to_pandas_edgelist(mst_nx)
    edgelist_df = mst_cugraph.view_edge_list()
    cg_sum = edgelist_df["weights"].sum()
    nx_sum = mst_nx_df["weight"].sum()
    print(cg_sum)
    print(nx_sum)
    assert np.isclose(cg_sum, nx_sum)


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED_WEIGHTS)
def test_minimum_spanning_tree_nx(graph_file):
    gc.collect()

    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    cuG = utils.read_csv_file(graph_file, read_weights_in_sp=True)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cuG, source="0", destination="1", edge_attr="2")

    # Just for getting relevant timing
    G.view_adj_list()

    t1 = time.time()
    cugraph_mst = cugraph.minimum_spanning_tree(G)
    t2 = time.time() - t1
    print("CuGraph time : " + str(t2))

    t1 = time.time()
    mst_nx = nx.minimum_spanning_tree(Gnx)
    t2 = time.time() - t1
    print("Nx Time : " + str(t2))
    print()

    compare_mst(cugraph_mst, mst_nx)

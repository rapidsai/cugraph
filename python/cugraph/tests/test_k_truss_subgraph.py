# Copyright (c) 2019, NVIDIA CORPORATION.
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

print('Networkx version : {} '.format(nx.__version__))


# These ground truth files have been created by running the networkx ktruss
# function on reference graphs. Currently networkx ktruss has an error such
# that nx.k_truss(G,k-2) gives the expected result for running ktruss with
# parameter k. This fix (https://github.com/networkx/networkx/pull/3713) is
# currently in networkx master and will hopefully will make it to a release
# soon.
def ktruss_ground_truth(graph_file):
    G = nx.read_edgelist(graph_file, nodetype=int, data=(('weights', float),))
    df = nx.to_pandas_edgelist(G)
    return df


def cugraph_k_truss_subgraph(graph_file, k, directed):
    # directed is used to create either a Graph or DiGraph so the returned
    # cugraph can be compared to nx graph of same type.
    cu_M = utils.read_csv_file(graph_file)
    if directed:
        G = cugraph.DiGraph()
    else:
        G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2')
    k_subgraph = cugraph.ktruss_subgraph(G, k)
    return k_subgraph


def compare_k_truss(graph_file, k, ground_truth_file, directed=True):
    k_truss_cugraph = cugraph_k_truss_subgraph(graph_file, k, directed)
    k_truss_nx = ktruss_ground_truth(ground_truth_file)

    edgelist_df = k_truss_cugraph.view_edge_list()
    src = edgelist_df['src']
    dst = edgelist_df['dst']
    wgt = edgelist_df['weights']
    if not directed:
        assert len(edgelist_df) == len(k_truss_nx)
    for i in range(len(src)):
        has_edge = ((k_truss_nx['source'] == src[i]) &
                    (k_truss_nx['target'] == dst[i]) &
                    np.isclose(k_truss_nx['weights'], wgt[i])).any()
        has_opp_edge = ((k_truss_nx['source'] == dst[i]) &
                        (k_truss_nx['target'] == src[i]) &
                        np.isclose(k_truss_nx['weights'], wgt[i])).any()
        assert(has_edge or has_opp_edge)
    return True


DATASETS = [('../datasets/polbooks.csv',
             '../datasets/ref/ktruss/polbooks.csv'),
            ('../datasets/netscience.csv',
             '../datasets/ref/ktruss/netscience.csv')]


@pytest.mark.parametrize('graph_file, nx_ground_truth', DATASETS)
def test_ktruss_subgraph_DiGraph(graph_file, nx_ground_truth):
    gc.collect()

    compare_k_truss(graph_file, 5, nx_ground_truth)


@pytest.mark.parametrize('graph_file, nx_ground_truth', DATASETS)
def test_ktruss_subgraph_Graph(graph_file, nx_ground_truth):
    gc.collect()

    compare_k_truss(graph_file, 5, nx_ground_truth, False)

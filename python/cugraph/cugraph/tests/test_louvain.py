# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
import time

import pytest

import cugraph
from cugraph.tests import utils

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import community and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

try:
    import community
except ModuleNotFoundError:
    pytest.exit("community module not found\n"
                "The python-louvain module needs to be installed\n"
                "please run `pip install python-louvain`")


print("Networkx version : {} ".format(nx.__version__))


def cugraph_call(cu_M, edgevals=False):

    G = cugraph.Graph()
    if edgevals:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    return parts, mod


def networkx_call(M):
    # z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    # Networkx louvain Call
    print("Solving... ")
    t1 = time.time()
    parts = community.best_partition(Gnx)
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))
    return parts


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_louvain_with_edgevals(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_parts, cu_mod = cugraph_call(cu_M, edgevals=True)

    nx_parts = networkx_call(M)
    # Calculating modularity scores for comparison
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1",
        edge_attr="weight", create_using=nx.Graph()
    )

    cu_parts = cu_parts.to_pandas()
    cu_map = dict(zip(cu_parts["vertex"], cu_parts["partition"]))

    assert set(nx_parts.keys()) == set(cu_map.keys())

    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)

    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (0.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < 0.0001


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_louvain(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_parts, cu_mod = cugraph_call(cu_M)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1",
        edge_attr="weight", create_using=nx.Graph()
    )

    cu_parts = cu_parts.to_pandas()
    cu_map = dict(zip(cu_parts["vertex"], cu_parts["partition"]))

    assert set(nx_parts.keys()) == set(cu_map.keys())

    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)

    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (0.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < 0.0001

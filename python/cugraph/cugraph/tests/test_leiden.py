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

import networkx as nx
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


def cugraph_leiden(G, edgevals=False):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.leiden(G)
    t2 = time.time() - t1
    print("Cugraph Leiden Time : " + str(t2))

    return parts, mod


def cugraph_louvain(G, edgevals=False):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Louvain Time : " + str(t2))

    return parts, mod


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_leiden(graph_file):
    gc.collect()
    edgevals = True

    cu_M = utils.read_csv_file(graph_file)

    G = cugraph.Graph()
    if edgevals:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")

    leiden_parts, leiden_mod = cugraph_leiden(G, edgevals=True)
    louvain_parts, louvain_mod = cugraph_louvain(G, edgevals=True)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_leiden_nx(graph_file):
    gc.collect()
    edgevals = True

    NM = utils.read_csv_for_nx(graph_file)

    if edgevals:
        G = nx.from_pandas_edgelist(
            NM, create_using=nx.Graph(), source="0", target="1"
        )
    else:
        G = nx.from_pandas_edgelist(
            NM, create_using=nx.Graph(), source="0", target="1", edge_attr="2"
        )

    leiden_parts, leiden_mod = cugraph_leiden(G, edgevals=True)
    louvain_parts, louvain_mod = cugraph_louvain(G, edgevals=True)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)

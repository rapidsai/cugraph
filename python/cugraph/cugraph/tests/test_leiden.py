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
import time

import pytest

import networkx as nx
import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS, karate_asymmetric

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import community and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_leiden(G):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.leiden(G)
    t2 = time.time() - t1
    print("Cugraph Leiden Time : " + str(t2))

    return parts, mod


def cugraph_louvain(G):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Louvain Time : " + str(t2))

    return parts, mod


@pytest.mark.parametrize("graph_file", DATASETS)
def test_leiden(graph_file):
    edgevals = True

    G = graph_file.get_graph(ignore_weights=not edgevals)
    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)


@pytest.mark.parametrize("graph_file", DATASETS)
def test_leiden_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)

    G = nx.from_pandas_edgelist(
        NM, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )

    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)


def test_leiden_directed_graph():

    edgevals = True
    G = karate_asymmetric.get_graph(
        create_using=cugraph.Graph(directed=True), ignore_weights=not edgevals
    )

    with pytest.raises(ValueError):
        parts, mod = cugraph_leiden(G)

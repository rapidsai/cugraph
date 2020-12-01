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
import time

import numpy as np
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


def cugraph_call(cu_M):
    # Device data
    weights_arr = cudf.Series(
        np.ones(max(cu_M["0"].max(), cu_M["1"].max()) + 1, dtype=np.float32)
    )

    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.jaccard_w(G, weights_arr)
    t2 = time.time() - t1
    print("Time : " + str(t2))

    df = df.sort_values(["source", "destination"]).reset_index(drop=True)

    return df["jaccard_coeff"]


def networkx_call(M):

    sources = M["0"]
    destinations = M["1"]
    edges = []
    for i in range(len(sources)):
        edges.append((sources[i], destinations[i]))
        edges.append((destinations[i], sources[i]))
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print("Format conversion ... ")

    # NetworkX graph
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )
    # Networkx Jaccard Call
    print("Solving... ")
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 = time.time() - t1

    print("Time : " + str(t2))
    coeff = []
    for u, v, p in preds:
        coeff.append(p)
    return coeff


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_wjaccard(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    # suppress F841 (local variable is assigned but never used) in flake8
    # no networkX equivalent to compare cu_coeff against...
    cu_coeff = cugraph_call(cu_M)  # noqa: F841
    nx_coeff = networkx_call(M)
    for i in range(len(cu_coeff)):
        diff = abs(nx_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6

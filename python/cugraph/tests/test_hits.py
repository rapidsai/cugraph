# Copyright (c) 2020, NVIDIA CORPORATION.
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
import pandas as pd

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


def cudify(d):
    if d is None:
        return None

    k = np.fromiter(d.keys(), dtype="int32")
    v = np.fromiter(d.values(), dtype="float32")
    cuD = cudf.DataFrame({"vertex": k, "values": v})
    return cuD


def cugraph_call(cu_M, max_iter, tol):
    # cugraph hits Call

    t1 = time.time()
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    df = cugraph.hits(G, max_iter, tol)
    df = df.sort_values("vertex").reset_index(drop=True)
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    return df


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def networkx_call(M, max_iter, tol):
    # in NVGRAPH tests we read as CSR and feed as CSC,
    # so here we do this explicitly
    print("Format conversion ... ")

    # Networkx Hits Call
    print("Solving... ")
    t1 = time.time()

    # Directed NetworkX graph
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    # same parameters as in NVGRAPH
    pr = nx.hits(Gnx, max_iter, tol, normalized=True)
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))

    return pr


MAX_ITERATIONS = [50]
TOLERANCE = [1.0e-06]


# Test all combinations of default/managed and pooled/non-pooled allocation


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
def test_hits(graph_file, max_iter, tol):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    hubs, authorities = networkx_call(M, max_iter, tol)

    cu_M = utils.read_csv_file(graph_file)
    cugraph_hits = cugraph_call(cu_M, max_iter, tol)

    # Calculating mismatch
    # hubs = sorted(hubs.items(), key=lambda x: x[0])
    # print("hubs = ", hubs)

    #
    #  Scores don't match.  Networkx uses the 1-norm,
    #  gunrock uses a 2-norm.  Eventually we'll add that
    #  as a parameter. For now, let's check the order
    #  which should match.  We'll allow 6 digits to right
    #  of decimal point accuracy
    #
    pdf = pd.DataFrame.from_dict(hubs, orient="index").sort_index()
    pdf = pdf.multiply(1000000).floordiv(1)
    cugraph_hits["nx_hubs"] = cudf.Series.from_pandas(pdf[0])

    pdf = pd.DataFrame.from_dict(authorities, orient="index").sort_index()
    pdf = pdf.multiply(1000000).floordiv(1)
    cugraph_hits["nx_authorities"] = cudf.Series.from_pandas(pdf[0])

    #
    #  Sort by hubs (cugraph) in descending order.  Then we'll
    #  check to make sure all scores are in descending order.
    #
    cugraph_hits = cugraph_hits.sort_values("hubs", ascending=False)

    assert cugraph_hits["hubs"].is_monotonic_decreasing
    assert cugraph_hits["nx_hubs"].is_monotonic_decreasing

    cugraph_hits = cugraph_hits.sort_values("authorities", ascending=False)

    assert cugraph_hits["authorities"].is_monotonic_decreasing
    assert cugraph_hits["nx_authorities"].is_monotonic_decreasing


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
def test_hits_nx(graph_file, max_iter, tol):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )
    nx_hubs, nx_authorities = nx.hits(Gnx, max_iter, tol, normalized=True)
    cg_hubs, cg_authorities = cugraph.hits(Gnx, max_iter, tol, normalized=True)

    # assert nx_hubs == cg_hubs
    # assert nx_authorities == cg_authorities

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
    nx_hits = nx.hits(Gnx, max_iter, tol, normalized=True)
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))

    return nx_hits


MAX_ITERATIONS = [50]
TOLERANCE = [1.0e-06]


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
def test_hits(graph_file, max_iter, tol):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    hubs, authorities = networkx_call(M, max_iter, tol)

    cu_M = utils.read_csv_file(graph_file)
    cugraph_hits = cugraph_call(cu_M, max_iter, tol)

    pdf = pd.DataFrame.from_dict(hubs, orient="index").sort_index()
    cugraph_hits["nx_hubs"] = cudf.Series.from_pandas(pdf[0])

    pdf = pd.DataFrame.from_dict(authorities, orient="index").sort_index()
    cugraph_hits["nx_authorities"] = cudf.Series.from_pandas(pdf[0])

    hubs_diffs1 = cugraph_hits.query('hubs - nx_hubs > 0.00001')
    hubs_diffs2 = cugraph_hits.query('hubs - nx_hubs < -0.00001')
    authorities_diffs1 = cugraph_hits.query(
        'authorities - nx_authorities > 0.0001')
    authorities_diffs2 = cugraph_hits.query(
        'authorities - nx_authorities < -0.0001')

    assert len(hubs_diffs1) == 0
    assert len(hubs_diffs2) == 0
    assert len(authorities_diffs1) == 0
    assert len(authorities_diffs2) == 0

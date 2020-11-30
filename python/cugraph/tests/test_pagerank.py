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


def cudify(d):
    if d is None:
        return None

    k = np.fromiter(d.keys(), dtype="int32")
    v = np.fromiter(d.values(), dtype="float32")
    cuD = cudf.DataFrame({"vertex": k, "values": v})
    return cuD


def cugraph_call(G, max_iter, tol, alpha, personalization, nstart):
    # cugraph Pagerank Call
    t1 = time.time()
    df = cugraph.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=personalization,
        nstart=nstart,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    # Sort Pagerank values
    sorted_pr = []

    df = df.sort_values("vertex").reset_index(drop=True)

    pr_scores = df["pagerank"].to_array()
    for i, rank in enumerate(pr_scores):
        sorted_pr.append((i, rank))

    return sorted_pr


# need a different function since the Nx version returns a dictionary
def cugraph_nx_call(G, max_iter, tol, alpha, personalization, nstart):
    # cugraph Pagerank Call
    t1 = time.time()
    pr = cugraph.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=personalization,
        nstart=nstart,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    return pr


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def networkx_call(Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx):

    personalization = None
    if personalization_perc != 0:
        personalization = {}
        personalization_count = int(
            (nnz_vtx.size * personalization_perc) / 100.0
        )
        nnz_vtx = np.random.choice(
            nnz_vtx, min(nnz_vtx.size, personalization_count), replace=False
        )

        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

    z = {k: 1.0 / Gnx.number_of_nodes() for k in Gnx.nodes()}

    # Networkx Pagerank Call
    t1 = time.time()

    pr = nx.pagerank(
        Gnx,
        alpha=alpha,
        nstart=z,
        max_iter=max_iter * 2,
        tol=tol * 0.01,
        personalization=personalization,
    )
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))

    return pr, personalization


MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]
PERSONALIZATION_PERC = [0, 10, 50]
HAS_GUESS = [0, 1]


# FIXME: the default set of datasets includes an asymmetric directed graph
# (email-EU-core.csv), which currently produces different results between
# cugraph and Nx and fails that test. Investigate, resolve, and use
# utils.DATASETS instead.
#
# https://github.com/rapidsai/cugraph/issues/533
#
# @pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
@pytest.mark.parametrize("alpha", ALPHA)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
def test_pagerank(
    graph_file, max_iter, tol, alpha, personalization_perc, has_guess
):
    gc.collect()

    # NetworkX PageRank
    M = utils.read_csv_for_nx(graph_file)
    nnz_vtx = np.unique(M[['0', '1']])
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    networkx_pr, networkx_prsn = networkx_call(
        Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx
    )

    cu_nstart = None
    if has_guess == 1:
        cu_nstart = cudify(networkx_pr)
        max_iter = 5
    cu_prsn = cudify(networkx_prsn)

    # cuGraph PageRank
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")

    cugraph_pr = cugraph_call(G, max_iter, tol, alpha, cu_prsn, cu_nstart)

    # Calculating mismatch
    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)
    for i in range(len(cugraph_pr)):
        if (
            abs(cugraph_pr[i][1] - networkx_pr[i][1]) > tol * 1.1
            and cugraph_pr[i][0] == networkx_pr[i][0]
        ):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01 * len(cugraph_pr))


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
@pytest.mark.parametrize("alpha", ALPHA)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
def test_pagerank_nx(
    graph_file, max_iter, tol, alpha, personalization_perc, has_guess
):
    gc.collect()

    # NetworkX PageRank
    M = utils.read_csv_for_nx(graph_file)
    nnz_vtx = np.unique(M[['0', '1']])
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.DiGraph()
    )

    networkx_pr, networkx_prsn = networkx_call(
        Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx
    )

    cu_nstart = None
    if has_guess == 1:
        cu_nstart = cudify(networkx_pr)
        max_iter = 5
    cu_prsn = cudify(networkx_prsn)

    # cuGraph PageRank with Nx Graph
    cugraph_pr = cugraph_nx_call(Gnx, max_iter, tol, alpha, cu_prsn, cu_nstart)

    # Calculating mismatch
    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    cugraph_pr = sorted(cugraph_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)

    for i in range(len(cugraph_pr)):
        if (
            abs(cugraph_pr[i][1] - networkx_pr[i][1]) > tol * 1.1
            and cugraph_pr[i][0] == networkx_pr[i][0]
        ):
            err = err + 1
            print(f"{cugraph_pr[i][1]} and {cugraph_pr[i][1]}")
    print("Mismatches:", err)
    assert err < (0.01 * len(cugraph_pr))

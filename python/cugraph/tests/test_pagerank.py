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
from itertools import product
import time
import numpy as np

import pytest

import cudf
import cugraph
from cugraph.tests import utils
import rmm

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


def cudify(d):
    if d is None:
        return None

    k = np.fromiter(d.keys(), dtype='int32')
    v = np.fromiter(d.values(), dtype='float32')
    cuD = cudf.DataFrame({'vertex': k, 'values': v})
    return cuD


def cugraph_call(cu_M, max_iter, tol, alpha, personalization, nstart):
    # cugraph Pagerank Call
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', target='1')
    t1 = time.time()
    df = cugraph.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol,
                          personalization=personalization, nstart=nstart)
    t2 = time.time() - t1
    print('Cugraph Time : '+str(t2))

    # Sort Pagerank values
    sorted_pr = []
    pr_scores = df['pagerank'].to_array()
    for i, rank in enumerate(pr_scores):
        sorted_pr.append((i, rank))

    return sorted_pr


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def networkx_call(M, max_iter, tol, alpha, personalization_perc):
    nnz_per_row = {r: 0 for r in range(M.get_shape()[0])}
    for nnz in range(M.getnnz()):
        nnz_per_row[M.row[nnz]] = 1 + nnz_per_row[M.row[nnz]]
    for nnz in range(M.getnnz()):
        M.data[nnz] = 1.0/float(nnz_per_row[M.row[nnz]])

    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    personalization = None
    if personalization_perc != 0:
        personalization = {}
        nnz_vtx = np.unique(M.nonzero())
        personalization_count = int((nnz_vtx.size *
                                     personalization_perc)/100.0)
        nnz_vtx = np.random.choice(nnz_vtx,
                                   min(nnz_vtx.size, personalization_count),
                                   replace=False)
        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val/sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

    # should be autosorted, but check just to make sure
    if not M.has_sorted_indices:
        print('sort_indices ... ')
        M.sort_indices()

    # in NVGRAPH tests we read as CSR and feed as CSC,
    # so here we do this explicitly
    print('Format conversion ... ')

    # Directed NetworkX graph
    Gnx = nx.DiGraph(M)

    z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

    # Networkx Pagerank Call
    print('Solving... ')
    t1 = time.time()

    # same parameters as in NVGRAPH
    pr = nx.pagerank(Gnx, alpha=alpha, nstart=z, max_iter=max_iter*2,
                     tol=tol*0.01, personalization=personalization)
    t2 = time.time() - t1

    print('Networkx Time : ' + str(t2))

    return pr, personalization


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv']

MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]
PERSONALIZATION_PERC = [0, 10, 50]
HAS_GUESS = [0, 1]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('max_iter', MAX_ITERATIONS)
@pytest.mark.parametrize('tol', TOLERANCE)
@pytest.mark.parametrize('alpha', ALPHA)
@pytest.mark.parametrize('personalization_perc', PERSONALIZATION_PERC)
@pytest.mark.parametrize('has_guess', HAS_GUESS)
def test_pagerank(managed, pool, graph_file, max_iter, tol, alpha,
                  personalization_perc, has_guess):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())
    M = utils.read_csv_for_nx(graph_file)
    networkx_pr, networkx_prsn = networkx_call(M, max_iter, tol, alpha,
                                               personalization_perc)

    cu_nstart = None
    if has_guess == 1:
        cu_nstart = cudify(networkx_pr)
        max_iter = 5
    cu_prsn = cudify(networkx_prsn)
    cu_M = utils.read_csv_file(graph_file)
    cugraph_pr = cugraph_call(cu_M, max_iter, tol, alpha, cu_prsn, cu_nstart)

    # Calculating mismatch

    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)
    for i in range(len(cugraph_pr)):
        if(abs(cugraph_pr[i][1]-networkx_pr[i][1]) > tol*1.1
           and cugraph_pr[i][0] == networkx_pr[i][0]):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01*len(cugraph_pr))

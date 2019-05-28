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

import pytest
from scipy.io import mmread

import cudf
import cugraph
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg

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


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def read_csv_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return cudf.read_csv(mm_file, delimiter=' ',
                         dtype=['int32', 'int32', 'float32'], header=None)


def cugraph_call(cu_M, max_iter, tol, alpha):
    # Device data
    sources = cu_M['0']
    destinations = cu_M['1']
    # values = cudf.Series(np.ones(len(sources), dtype = np.float64))

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)
    t1 = time.time()
    df = cugraph.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Sort Pagerank values
    sorted_pr = []
    pr_scores = df['pagerank'].to_array()
    for i, rank in enumerate(pr_scores):
        sorted_pr.append((i, rank))

    return sorted(sorted_pr, key=lambda x: x[1], reverse=True)


def networkx_call(M, max_iter, tol, alpha):
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
                     tol=tol*0.01)
    t2 = time.time() - t1

    print('Time : ' + str(t2))

    # return Sorted Pagerank values
    return sorted(pr.items(), key=lambda x: x[1], reverse=True)


DATASETS = ['../datasets/dolphins',
            '../datasets/karate',
            '../datasets/netscience']

MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('max_iter', MAX_ITERATIONS)
@pytest.mark.parametrize('tol', TOLERANCE)
@pytest.mark.parametrize('alpha', ALPHA)
def test_pagerank(managed, pool, graph_file, max_iter, tol, alpha):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    M = read_mtx_file(graph_file+'.mtx')
    networkx_pr = networkx_call(M, max_iter, tol, alpha)

    cu_M = read_csv_file(graph_file+'.csv')
    cugraph_pr = cugraph_call(cu_M, max_iter, tol, alpha)

    # Calculating mismatch

    err = 0
    assert len(cugraph_pr) == len(networkx_pr)
    for i in range(len(cugraph_pr)):
        if(abs(cugraph_pr[i][1]-networkx_pr[i][1]) > tol*1.1
           and cugraph_pr[i][0] == networkx_pr[i][0]):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01*len(cugraph_pr))

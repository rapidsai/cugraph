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

import time

import numpy as np
import pytest
from scipy.io import mmread

import cudf
import cugraph

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


def cugraph_call(M, source):

    # Device data
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    print('sources size = ' + str(len(sources)))
    print('destinations size = ' + str(len(destinations)))

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)

    print('cugraph Solving... ')
    t1 = time.time()

    dist = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print('Time : '+str(t2))

    distances = []
    dist_np = dist['distance'].to_array()
    for i, d in enumerate(dist_np):
        distances.append((i, d))

    return distances


def networkx_call(M, source):

    print('Format conversion ... ')
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Directed NetworkX graph
    Gnx = nx.DiGraph(M)

    print('NX Solving... ')
    t1 = time.time()

    path = nx.single_source_shortest_path(Gnx, source)

    t2 = time.time() - t1

    print('Time : ' + str(t2))

    return path


DATASETS = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']

SOURCES = [1]


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('source', SOURCES)
def test_sssp(graph_file, source):

    M = read_mtx_file(graph_file)
    cu_paths = cugraph_call(M, source)
    nx_paths = networkx_call(M, source)

    # Calculating mismatch
    err = 0

    for i in range(len(cu_paths)):
        if (cu_paths[i][1] != np.finfo(np.float32).max):
            if(cu_paths[i][1] != (len(nx_paths[cu_paths[i][0]])-1)):
                err = err + 1
        else:
            if (cu_paths[i][0] in nx_paths.keys()):
                err = err + 1

    assert err == 0

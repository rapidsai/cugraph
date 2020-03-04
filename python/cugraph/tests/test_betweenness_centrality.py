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

import pytest

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


def calc_betweenness_centrality(graph_file, normalized=True):
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')

    df = cugraph.betweenness_centrality(G, normalized=normalized)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(NM, create_using=nx.DiGraph(),
                                  source='0', target='1')
    nb = nx.betweenness_centrality(Gnx, normalized=normalized)
    pdf = [nb[k] for k in sorted(nb.keys())]
    df['nx'] = pdf
    df = df.rename({'betweenness_centrality': 'cu'})
    return df


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/netscience.csv']


@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_betweenness_centrality(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality(graph_file)

    err = 0
    epsilon = 0.0001

    for i in range(len(scores)):
        if (scores['cu'][i] < (scores['nx'][i] * (1 - epsilon)) or
           scores['cu'][i] > (scores['nx'][i] * (1 + epsilon))):
            err = err + 1
            print('ERROR: cu = {}, nx = {}'.format(scores['cu'][i],
                                                   scores['nx'][i]))

    assert err == 0

@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_betweenness_centrality_unnormalized(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    scores = calc_betweenness_centrality(graph_file, False)

    err = 0
    epsilon = 0.0001

    for i in range(len(scores)):
        if (scores['cu'][i] < (scores['nx'][i] * (1 - epsilon)) or
           scores['cu'][i] > (scores['nx'][i] * (1 + epsilon))):
            err = err + 1
            print('ERROR: cu = {}, nx = {}'.format(scores['cu'][i],
                                                   scores['nx'][i]))

    assert err == 0

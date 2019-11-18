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

import pandas as pd
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


def calc_core_number(graph_file):
    M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(M, source='0', target='1')

    cn = cugraph.core_number(G)

    NM = utils.read_csv_for_nx(graph_file)
    NM = NM.tocsr()
    Gnx = nx.Graph(NM)
    nc = nx.core_number(Gnx)
    pdf = pd.DataFrame(nc, index=[0]).T
    cn['nx_core_number'] = pdf[0]
    cn = cn.rename({'core_number': 'cu_core_number'})
    return cn


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/netscience.csv']


@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_core_number(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    cn = calc_core_number(graph_file)

    assert cn['cu_core_number'].equals(cn['nx_core_number'])

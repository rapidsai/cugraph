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

# This file test the Renumbering features

import gc
from itertools import product

import pytest

import cudf
import cugraph
from cugraph.tests import utils
import rmm

DATASETS = ['../datasets/karate.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_multi_column_unrenumbering(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    translate = 100
    cu_M = utils.read_csv_file(graph_file)
    cu_M['00'] = cu_M['0'] + translate
    cu_M['11'] = cu_M['1'] + translate

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, ['0', '00'], ['1', '11'])
    result_multi = cugraph.pagerank(G).sort_values(by='0').\
        reset_index(drop=True)

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, '0', '1')
    result_single = cugraph.pagerank(G)

    result_exp = cudf.DataFrame()
    result_exp['0'] = result_single['vertex']
    result_exp['1'] = result_single['vertex'] + translate
    result_exp['pagerank'] = result_single['pagerank']

    assert result_multi.equals(result_exp)

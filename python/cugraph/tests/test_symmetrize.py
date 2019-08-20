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

import cudf
import cugraph
from cugraph.tests import utils
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg


def test_version():
    gc.collect()
    cugraph.__version__


DATASETS = ['../datasets/karate',
            '../datasets/email-Eu-core']


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sources = cu_M['0']
    destinations = cu_M['1']

    sym_sources, sym_destinations = cugraph.symmetrize(sources, destinations)

    #
    #  Check to see if all pairs in sources/destinations exist in
    #  both directions
    #
    #  Try this with join logic.  Note that if we create data frames
    #  we can join the data frames (using the DataFrame.merge function).
    #  The symmetrize function should contain every edge that was contained
    #  in the input data.  So if we join the input data with the output
    #  the length of the data frames should be equal.
    #
    sym_df = cudf.DataFrame()
    sym_df['src_s'] = sym_sources
    sym_df['dst_s'] = sym_destinations

    orig_df = cudf.DataFrame()
    orig_df['src'] = sources
    orig_df['dst'] = destinations

    #
    #  Make sure all of the original data is present
    #
    join = orig_df.merge(sym_df, left_on=['src', 'dst'],
                         right_on=['src_s', 'dst_s'])
    assert len(orig_df) == len(join)

    #
    #  Now check the symmetrized edges are present
    #
    join = orig_df.merge(sym_df, left_on=['src', 'dst'],
                         right_on=['dst_s', 'src_s'])
    assert len(orig_df) == len(join)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    join = sym_df.merge(orig_df, left_on=['src_s', 'dst_s'],
                        right_on=['src', 'dst'])
    join1 = sym_df.merge(orig_df, left_on=['src_s', 'dst_s'],
                         right_on=['dst', 'src'])
    joinM = join.merge(join1, how='outer', on=['src_s', 'dst_s'])

    assert len(sym_df) == len(joinM)

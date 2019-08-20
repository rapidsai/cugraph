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
def test_symmetrize_simple(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')

    sym_sources, sym_destinations = cugraph.symmetrize(cu_M['0'], cu_M['1'])

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
    orig_df['src'] = cu_M['0']
    orig_df['dst'] = cu_M['1']

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


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize_weighted(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')

    sym_src, sym_dst, sym_w = cugraph.symmetrize(cu_M['0'],
                                                 cu_M['1'],
                                                 cu_M['2'])

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
    sym_df = cudf.DataFrame([('src', sym_src),
                             ('dst', sym_dst),
                             ('weight', sym_w)])

    #  TODO:  How to test weights...

    #
    #  Make sure all of the original data is present
    #
    join = cu_M.merge(sym_df, left_on=['0', '1'],
                      right_on=['src', 'dst'])
    assert len(cu_M) == len(join)

    #
    #  Now check the symmetrized edges are present
    #
    join = cu_M.merge(sym_df, left_on=['0', '1'],
                      right_on=['dst', 'src'])
    assert len(cu_M) == len(join)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    join = sym_df.merge(cu_M, left_on=['src', 'dst'],
                        right_on=['0', '1'])
    join1 = sym_df.merge(cu_M, left_on=['src', 'dst'],
                         right_on=['1', '0'])
    joinM = join.merge(join1, how='outer', on=['src', 'dst'])

    assert len(sym_df) == len(joinM)


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize_df(managed, pool, graph_file):
    gc.collect()

    rmm.finalize()
    rmm_cfg.use_managed_memory = managed
    rmm_cfg.use_pool_allocator = pool
    rmm.initialize()

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sym_df = cugraph.symmetrize_df(cu_M, '0', '1')

    #
    # NOTE:  rename is necessary due to
    #    https://github.com/rapidsai/cudf/issues/2594
    # which should be fixed soon
    #
    sym_df.rename(columns={'0': '0_sym', '1': '1_sym'}, inplace=True)

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

    #  TODO:  How to test weights...

    #
    #  Make sure all of the original data is present
    #
    join = cu_M.merge(sym_df, left_on=['0', '1'],
                      right_on=['0_sym', '1_sym'])
    assert len(cu_M) == len(join)

    #
    #  Now check the symmetrized edges are present
    #
    join = cu_M.merge(sym_df, left_on=['0', '1'],
                      right_on=['1_sym', '0_sym'])
    assert len(cu_M) == len(join)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    join = sym_df.merge(cu_M, left_on=['0_sym', '1_sym'],
                        right_on=['0', '1'])
    join1 = sym_df.merge(cu_M, left_on=['0_sym', '1_sym'],
                         right_on=['1', '0'])
    joinM = join.merge(join1, how='outer', on=['0_sym', '1_sym'])

    assert len(sym_df) == len(joinM)

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
import cudf
import cugraph
from cugraph.tests import utils
import rmm


def test_version():
    gc.collect()
    cugraph.__version__


DATASETS = ['../datasets/karate',
            '../datasets/email-Eu-core']


def compare(src1, dst1, val1, src2, dst2, val2):
    #
    #  We will do comparison computations by using dataframe
    #  merge functions (essentially doing fast joins).  We
    #  start by making two data frames
    #
    df1 = cudf.DataFrame()
    df1['src1'] = src1
    df1['dst1'] = dst1
    if val1 is not None:
        df1['val1'] = val1

    df2 = cudf.DataFrame()
    df2['src2'] = src2
    df2['dst2'] = dst2
    if val2 is not None:
        df2['val2'] = val2

    #
    #  Check to see if all pairs in the original data frame
    #  still exist in the new data frame.  If we join (merge)
    #  the data frames where (src1[i]=src2[i]) and (dst1[i]=dst2[i])
    #  then we should get exactly the same number of entries in
    #  the data frame if we did not lose any data.
    #
    join = df1.merge(df2,
                     left_on=['src1', 'dst1'],
                     right_on=['src2', 'dst2'])
    assert len(df1) == len(join)

    if val1 is not None:
        #
        #  Check the values.  In this join, if val1 and val2 are
        #  the same then we are good.  If they are different then
        #  we need to check if the value is selected from the opposite
        #  direction, so we'll merge with the edges reversed and
        #  check to make sure that the values all match
        #
        diffs = join.query('val1 != val2')
        diffs_check = diffs.merge(df1,
                                  left_on=['src1', 'dst1'],
                                  right_on=['dst1', 'src1'])
        query = diffs_check.query('val1_y != val2')
        if len(query) > 0:
            print('differences: ')
            print(query)
            assert 0 == len(query)

    #
    #  Now check the symmetrized edges are present.  If the original
    #  data contains (u,v) we want to make sure that (v,u) is present
    #  in the new data frame.
    #
    #  We can accomplish this by doing the join (merge) where
    #  (src1[i] = dst2[i]) and (dst1[i] = src2[i]), and verifying
    #  that we get exactly the same number of entries in the data frame.
    #
    join = df1.merge(df2,
                     left_on=['src1', 'dst1'],
                     right_on=['dst2', 'src2'])
    assert len(df1) == len(join)

    if val1 is not None:
        #
        #  Check the values.  In this join, if val1 and val2 are
        #  the same then we are good.  If they are different then
        #  we need to check if the value is selected from the opposite
        #  direction, so we'll merge with the edges reversed and
        #  check to make sure that the values all match
        #
        diffs = join.query('val1 != val2')
        diffs_check = diffs.merge(df1,
                                  left_on=['src2', 'dst2'],
                                  right_on=['src1', 'dst1'])
        query = diffs_check.query('val1_y != val2')
        if len(query) > 0:
            print('differences: ')
            print(query)
            assert 0 == len(query)

    #
    #  Finally, let's check (in both directions) backwards.
    #  We want to make sure that no edges were created in
    #  the symmetrize logic that didn't already exist in one
    #  direction or the other.  This is a bit more complicated.
    #
    #  The complication here is that the original data could,
    #  for some edge (u,v) ALREADY contain the edge (v,u).  The
    #  symmetrized graph will not duplicate any edges, so the edge
    #  (u,v) will only be present once.  So we can't simply check
    #  counts of df2 joined with df1.
    #
    #  join1 will contain the join (merge) of df2 to df1 in the
    #        forward direction
    #  join2 will contain the join (merge) of df2 to df1 in the
    #        reverse direction
    #
    #  Finally, we'll do an outer join of join1 and join2, which
    #  will combine any (u,v)/(v,u) pairs that might exist into
    #  a joined row while keeping any (u,v) pairs that don't exist
    #  in both data frames as single rows.  This gives us a data frame
    #  with the same number of rows as the symmetrized data.
    #
    join1 = df2.merge(df1,
                      left_on=['src2', 'dst2'],
                      right_on=['src1', 'dst1'])
    join2 = df2.merge(df1,
                      left_on=['src2', 'dst2'],
                      right_on=['dst1', 'src1'])
    joinM = join1.merge(join2, how='outer', on=['src2', 'dst2'])

    assert len(df2) == len(joinM)

    #
    #  Note, we don't need to check the reverse values... we checked
    #  them in both directions earlier.
    #


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize_unweighted(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

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

    compare(orig_df['src'], orig_df['dst'], None,
            sym_df['src_s'], sym_df['dst_s'], None)


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize_weighted(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')

    sym_src, sym_dst, sym_w = cugraph.symmetrize(cu_M['0'],
                                                 cu_M['1'],
                                                 cu_M['2'])

    compare(cu_M['0'], cu_M['1'], cu_M['2'], sym_src, sym_dst, sym_w)


# Test all combinations of default/managed and pooled/non-pooled allocation
# NOTE: see https://github.com/rapidsai/cudf/issues/2636
#       drop_duplicates doesn't work well with the pool allocator
#                        list(product([False, True], [False, True])))
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_symmetrize_df(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool
    )

    assert(rmm.is_initialized())

    cu_M = utils.read_csv_file(graph_file+'.csv')
    sym_df = cugraph.symmetrize_df(cu_M, '0', '1')

    compare(cu_M['0'], cu_M['1'], cu_M['2'],
            sym_df['0'], sym_df['1'], sym_df['2'])


def test_symmetrize_bad_weights():
    src = [0, 0, 0, 0, 1, 2]
    dst = [1, 2, 3, 4, 0, 3]
    val = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0]

    df = pd.DataFrame({
        'src': src,
        'dst': dst,
        'val': val
    })

    gdf = cudf.DataFrame.from_pandas(df[['src', 'dst', 'val']])
    sym_df = cugraph.symmetrize_df(gdf, 'src', 'dst')

    compare(gdf['src'], gdf['dst'], gdf['val'],
            sym_df['src'], sym_df['dst'], sym_df['val'])

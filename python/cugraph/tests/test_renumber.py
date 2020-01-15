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

import pandas as pd
import pytest

import cudf
import cugraph
from cugraph.tests import utils
import rmm

DATASETS = ['../datasets/karate.csv',
            '../datasets/dolphins.csv',
            '../datasets/netscience.csv']


def test_renumber_ips():

    source_list = ['192.168.1.1',
                   '172.217.5.238',
                   '216.228.121.209',
                   '192.16.31.23']
    dest_list = ['172.217.5.238',
                 '216.228.121.209',
                 '192.16.31.23',
                 '192.168.1.1']

    pdf = pd.DataFrame({
            'source_list': source_list,
            'dest_list': dest_list
            })

    gdf = cudf.from_pandas(pdf)

    gdf['source_as_int'] = gdf['source_list'].str.ip2int()
    gdf['dest_as_int'] = gdf['dest_list'].str.ip2int()

    src, dst, numbering = cugraph.renumber(gdf['source_as_int'],
                                           gdf['dest_as_int'])

    for i in range(len(gdf)):
        assert numbering[src[i]] == gdf['source_as_int'][i]
        assert numbering[dst[i]] == gdf['dest_as_int'][i]


def test_renumber_ips_cols():

    source_list = ['192.168.1.1',
                   '172.217.5.238',
                   '216.228.121.209',
                   '192.16.31.23']
    dest_list = ['172.217.5.238',
                 '216.228.121.209',
                 '192.16.31.23',
                 '192.168.1.1']

    pdf = pd.DataFrame({
            'source_list': source_list,
            'dest_list': dest_list
            })

    gdf = cudf.from_pandas(pdf)

    gdf['source_as_int'] = gdf['source_list'].str.ip2int()
    gdf['dest_as_int'] = gdf['dest_list'].str.ip2int()

    src, dst, number_df = cugraph.renumber_from_cudf(
        gdf, ['source_as_int'], ['dest_as_int'])

    for i in range(len(gdf)):
        assert number_df['0'][src[i]] == gdf['source_as_int'][i]
        assert number_df['0'][dst[i]] == gdf['dest_as_int'][i]


def test_renumber_ips_str_cols():

    source_list = ['192.168.1.1',
                   '172.217.5.238',
                   '216.228.121.209',
                   '192.16.31.23']
    dest_list = ['172.217.5.238',
                 '216.228.121.209',
                 '192.16.31.23',
                 '192.168.1.1']

    pdf = pd.DataFrame({
            'source_list': source_list,
            'dest_list': dest_list
            })

    gdf = cudf.from_pandas(pdf)

    src, dst, number_df = cugraph.renumber_from_cudf(
        gdf, ['source_list'], ['dest_list'])

    for i in range(len(gdf)):
        assert number_df['0'][src[i]] == gdf['source_list'][i]
        assert number_df['0'][dst[i]] == gdf['dest_list'][i]


def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({
        'source_list': source_list,
        'dest_list': dest_list,
    })

    gdf = cudf.DataFrame.from_pandas(df[['source_list', 'dest_list']])

    src, dst, numbering = cugraph.renumber(gdf['source_list'],
                                           gdf['dest_list'])

    for i in range(len(source_list)):
        assert source_list[i] == numbering[src[i]]
        assert dest_list[i] == numbering[dst[i]]


def test_renumber_negative_col():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({
        'source_list': source_list,
        'dest_list': dest_list,
    })

    gdf = cudf.DataFrame.from_pandas(df[['source_list', 'dest_list']])

    src, dst, numbering = cugraph.renumber_from_cudf(
        gdf, ['source_list'], ['dest_list'])

    for i in range(len(source_list)):
        assert source_list[i] == numbering['0'][src[i]]
        assert dest_list[i] == numbering['0'][dst[i]]


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_renumber_files(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    source_translated = cudf.Series([x + translate for x in sources])
    dest_translated = cudf.Series([x + translate for x in destinations])

    src, dst, numbering = cugraph.renumber(source_translated, dest_translated)

    for i in range(len(sources)):
        assert sources[i] == (numbering[src[i]] - translate)
        assert destinations[i] == (numbering[dst[i]] - translate)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_renumber_files_col(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src'] = cudf.Series([x + translate for x in sources])
    gdf['dst'] = cudf.Series([x + translate for x in destinations])

    src, dst, numbering = cugraph.renumber_from_cudf(gdf, ['src'], ['dst'])

    for i in range(len(gdf)):
        assert sources[i] == (numbering['0'][src[i]] - translate)
        assert destinations[i] == (numbering['0'][dst[i]] - translate)

# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
def test_renumber_files_multi_col(managed, pool, graph_file):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src_old'] = sources
    gdf['dst_old'] = destinations
    gdf['src'] = sources + translate
    gdf['dst'] = destinations + translate

    src, dst, numbering = cugraph.renumber_from_cudf(
        gdf, ['src', 'src_old'], ['dst', 'dst_old'])

    for i in range(len(gdf)):
        assert sources[i] == (numbering['0'][src[i]] - translate)
        assert destinations[i] == (numbering['0'][dst[i]] - translate)

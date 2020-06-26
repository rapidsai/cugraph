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

import pandas as pd
import pytest

import cudf
import cugraph
from cugraph.structure.number_map import NumberMap
from cugraph.tests import utils

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf
import dask


@pytest.mark.skip(reason='debug')
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

    numbering = NumberMap()
    numbering.from_series(gdf['source_as_int'], gdf['dest_as_int'])
    src = numbering.to_vertex_id(gdf['source_as_int'])
    dst = numbering.to_vertex_id(gdf['dest_as_int'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['source_as_int'])
    assert check_dst.equals(gdf['dest_as_int'])


@pytest.mark.skip(reason='debug')
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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['source_as_int'], ['dest_as_int'])
    src = numbering.to_vertex_id(gdf['source_as_int'])
    dst = numbering.to_vertex_id(gdf['dest_as_int'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['source_as_int'])
    assert check_dst.equals(gdf['dest_as_int'])


@pytest.mark.skip(reason='temporarily dropped string support')
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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['source_list'], ['dest_list'])
    src = numbering.to_vertex_id(gdf['source_list'])
    dst = numbering.to_vertex_id(gdf['dest_list'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['source_list'])
    assert check_dst.equals(gdf['dest_list'])


@pytest.mark.skip(reason='debug')
def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({
        'source_list': source_list,
        'dest_list': dest_list,
    })

    gdf = cudf.DataFrame.from_pandas(df[['source_list', 'dest_list']])

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['source_list'], ['dest_list'])
    src = numbering.to_vertex_id(gdf['source_list'])
    dst = numbering.to_vertex_id(gdf['dest_list'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['source_list'])
    assert check_dst.equals(gdf['dest_list'])


@pytest.mark.skip(reason='debug')
def test_renumber_negative_col():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({
        'source_list': source_list,
        'dest_list': dest_list,
    })

    gdf = cudf.DataFrame.from_pandas(df[['source_list', 'dest_list']])

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['source_list'], ['dest_list'])
    src = numbering.to_vertex_id(gdf['source_list'])
    dst = numbering.to_vertex_id(gdf['dest_list'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['source_list'])
    assert check_dst.equals(gdf['dest_list'])


# Test all combinations of default/managed and pooled/non-pooled allocation

@pytest.mark.parametrize('graph_file', utils.DATASETS)
@pytest.mark.skip(reason='debug')
def test_renumber_files(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    source_translated = cudf.Series([x + translate for x in sources])
    dest_translated = cudf.Series([x + translate for x in destinations])

    numbering = NumberMap()
    numbering.from_series(source_translated, dest_translated)
    src = numbering.to_vertex_id(source_translated)
    dst = numbering.to_vertex_id(dest_translated)

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(source_translated)
    assert check_dst.equals(dest_translated)


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('graph_file', utils.DATASETS)
@pytest.mark.skip(reason='debug')
def test_renumber_files_col(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src'] = cudf.Series([x + translate for x in sources])
    gdf['dst'] = cudf.Series([x + translate for x in destinations])

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['src'], ['dst'])
    src = numbering.to_vertex_id(gdf['src'])
    dst = numbering.to_vertex_id(gdf['dst'])

    check_src = numbering.from_vertex_id(src)['0']
    check_dst = numbering.from_vertex_id(dst)['0']

    assert check_src.equals(gdf['src'])
    assert check_dst.equals(gdf['dst'])


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('graph_file', utils.DATASETS)
@pytest.mark.skip(reason='debug')
def test_renumber_files_multi_col(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src_old'] = sources
    gdf['dst_old'] = destinations
    gdf['src'] = sources + translate
    gdf['dst'] = destinations + translate

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ['src', 'src_old'], ['dst', 'dst_old'])
    src = numbering.to_vertex_id(gdf, ['src', 'src_old'])
    dst = numbering.to_vertex_id(gdf, ['dst', 'dst_old'])

    check_src = numbering.from_vertex_id(src)
    check_dst = numbering.from_vertex_id(dst)

    assert check_src['0'].equals(gdf['src'])
    assert check_src['1'].equals(gdf['src_old'])
    assert check_dst['0'].equals(gdf['dst'])
    assert check_dst['1'].equals(gdf['dst_old'])

        
@pytest.fixture()
def client_setup():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    client


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('graph_file', utils.DATASETS)
def test_opg_renumber(graph_file, client_setup):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M['0'])
    destinations = cudf.Series(M['1'])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src_old'] = sources
    gdf['dst_old'] = destinations
    gdf['src'] = sources + translate
    gdf['dst'] = destinations + translate

    #ddf = dask_cudf.dataframe.from_pandas(gdf, npartitions=2)
    ddf = dask.dataframe.from_pandas(gdf, npartitions=2)

    numbering = NumberMap()
    numbering.from_dataframe(ddf, ['src', 'src_old'], ['dst', 'dst_old'])
    src = numbering.to_vertex_id(ddf, ['src', 'src_old'])
    dst = numbering.to_vertex_id(ddf, ['dst', 'dst_old'])

    check_src = numbering.from_vertex_id(src).compute()
    check_dst = numbering.from_vertex_id(dst).compute()

    print('gdf = ', gdf)
    print('check_src (type: ', type(check_src), ') = ', check_src)

    assert check_src['0'].to_pandas().equals(gdf['src'].to_pandas())
    assert check_src['1'].to_pandas().equals(gdf['src_old'].to_pandas())
    assert check_dst['0'].to_pandas().equals(gdf['dst'].to_pandas())
    assert check_dst['1'].to_pandas().equals(gdf['dst_old'].to_pandas())

# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import pytest

import cudf
import cugraph
from cugraph.tests import utils


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


def cugraph_call(M, edgevals=False):
    G = cugraph.Graph()
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])
    if edgevals is True:
        cu_M["weights"] = cudf.Series(M["weight"])
        G.from_cudf_edgelist(
            cu_M, source="src", destination="dst", edge_attr="weights"
        )
    else:
        G.from_cudf_edgelist(cu_M, source="src", destination="dst")
    return cugraph.triangles(G)


def networkx_call(M):
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )
    dic = nx.triangles(Gnx)
    print(dic)
    count = 0
    for i in dic.keys():
        count += dic[i]
    return count


# FIXME: the default set of datasets includes an asymmetric directed graph
# (email-EU-core.csv), which currently produces different results between
# cugraph and Nx and fails that test. Investigate, resolve, and use
# utils.DATASETS instead.
#
# https://github.com/rapidsai/cugraph/issues/1043
#
# @pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_triangles(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_count = cugraph_call(M)
    nx_count = networkx_call(M)
    assert cu_count == nx_count


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_triangles_edge_vals(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_count = cugraph_call(M, edgevals=True)
    nx_count = networkx_call(M)
    assert cu_count == nx_count


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_triangles_nx(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )

    cu_count = cugraph.triangles(G)
    dic = nx.triangles(G)
    nx_count = 0
    for i in dic.keys():
        nx_count += dic[i]

    assert cu_count == nx_count

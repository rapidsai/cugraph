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

import pytest

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


print('Networkx version : {} '.format(nx.__version__))


def topKVertices(katz, col, k):
    top = katz.nlargest(n=k, columns=col)
    top = top.sort_values(by=col, ascending=False)
    return top['vertex']


def calc_katz(graph_file):
    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')

    largest_out_degree = G.degrees().nlargest(n=1, columns='out_degree')
    largest_out_degree = largest_out_degree['out_degree'].iloc[0]
    katz_alpha = 1/(largest_out_degree + 1)

    k_df = cugraph.katz_centrality(G, None, max_iter=1000)

    NM = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(NM, create_using=nx.DiGraph(),
                                  source='0', target='1')
    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    pdf = [nk[k] for k in sorted(nk.keys())]
    k_df['nx_katz'] = pdf
    k_df = k_df.rename({'katz_centrality': 'cu_katz'})
    return k_df


DATASETS = ['../datasets/dolphins.csv',
            '../datasets/netscience.csv']


@pytest.mark.parametrize('graph_file', DATASETS)
def test_katz_centrality(graph_file):
    gc.collect()

    katz_scores = calc_katz(graph_file)

    topKNX = topKVertices(katz_scores, 'nx_katz', 10)
    topKCU = topKVertices(katz_scores, 'cu_katz', 10)

    assert topKNX.equals(topKCU)

# Copyright (c) 2022, NVIDIA CORPORATION.
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

LIBCUGRAPH_C_DATASET = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"small_graph.csv"


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_katz_centrality_2(graph_file):
    # Creating the networkx version alongside cugraph version
    gc.collect()
    NM = utils.read_csv_for_nx(graph_file)
    nx_G = nx.from_pandas_edgelist(
        NM, create_using=nx.DiGraph(), source="0", target="1",
        edge_attr="weight"
    )

    G = cugraph.utilities.convert_from_nx(nx_G)
    largest_out_degree = G.degrees().nlargest(n=1, columns="out_degree")
    largest_out_degree = largest_out_degree["out_degree"].iloc[0]
    katz_alpha = 1 / (largest_out_degree + 1)

    nk = nx.katz_centrality(nx_G, alpha=katz_alpha)
    ck = cugraph.centrality.katz_centrality(G, alpha=katz_alpha,
                                            max_iter=1000)
    ck2 = cugraph.centrality.katz_centrality_2(G, alpha=katz_alpha,
                                               max_iter=1000)
    print(nk)
    print(ck)
    print(ck2)
    # breakpoint()


@pytest.mark.parametrize("graph_file", [LIBCUGRAPH_C_DATASET])
def test_katz_centrality_toy(graph_file):
    # This test is based off of libcugraph_c and pylibcugraph tests
    gc.collect()

    df = cudf.read_csv(LIBCUGRAPH_C_DATASET, delimiter=' ',
                       dtype=['int32', 'int32', 'float32'], header=None)
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source='0', destination='1', edge_attr='2')

    alpha = 0.01
    beta = 1.0
    tol = 0.000001
    max_iter = 1000
    centralities = [0.410614, 0.403211, 0.390689, 0.415175, 0.395125,
                    0.433226]

    ck = cugraph.centrality.katz_centrality_2(G, alpha=alpha, beta=beta,
                                              tol=tol, max_iter=max_iter)
    # breakpoint()
    # for vertex in ck["vertices"]:
    ck = ck.sort_values("vertex")
    for vertex in ck["vertex"].to_pandas():
        expected_score = centralities[vertex]
        actual_score = ck["katz_centrality"][vertex]
        if pytest.approx(expected_score, 1e-4) != actual_score:
            # raise ValueError(f"Actual: {ck["katz_centrality"]}"
            #                  f", expected: {centralities}")
            raise ValueError(f"Katz centrality score is {actual_score}"
                             f", should have been {expected_score}")

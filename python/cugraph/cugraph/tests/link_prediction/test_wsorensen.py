# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np
import pytest

import cudf
from cudf.testing import assert_series_equal

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_call(benchmark_callable, graph_file):
    # Device data
    cu_M = graph_file.get_edgelist()
    weight_arr = cudf.Series(
        np.ones(max(cu_M["src"].max(), cu_M["dst"].max()) + 1, dtype=np.float32)
    )
    weights = cudf.DataFrame()
    weights["vertex"] = np.arange(len(weight_arr), dtype=np.int32)
    weights["weight"] = weight_arr

    G = graph_file.get_graph(ignore_weights=True)

    # cugraph Sorensen Call
    df = benchmark_callable(cugraph.sorensen_w, G, weights)

    df = df.sort_values(["first", "second"]).reset_index(drop=True)

    return df["sorensen_coeff"]


def networkx_call(M, benchmark_callable=None):

    sources = M["0"]
    destinations = M["1"]
    edges = []
    for i in range(len(sources)):
        edges.append((sources[i], destinations[i]))
        edges.append((destinations[i], sources[i]))
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print("Format conversion ... ")

    # NetworkX graph
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.Graph())
    # Networkx Jaccard Call
    print("Solving... ")
    if benchmark_callable is not None:
        preds = benchmark_callable(nx.jaccard_coefficient, Gnx, edges)
    else:
        preds = nx.jaccard_coefficient(Gnx, edges)
    coeff = []
    for u, v, p in preds:
        # FIXME: Use known correct values of WSorensen for few graphs,
        # hardcode it and compare to Cugraph WSorensen
        # to get a more robust test

        # Conversion from Networkx Jaccard to Sorensen
        coeff.append((2 * p) / (1 + p))
    return coeff


# =============================================================================
# Pytest Fixtures
# =============================================================================
@pytest.fixture(scope="module", params=DATASETS_UNDIRECTED)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    graph_file = request.param
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)

    return M, graph_file


def test_wsorensen(gpubenchmark, read_csv):

    M, graph_file = read_csv

    cu_coeff = cugraph_call(gpubenchmark, graph_file)
    nx_coeff = networkx_call(M)
    for i in range(len(cu_coeff)):
        diff = abs(nx_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6


def test_nx_wsorensen_time(gpubenchmark, read_csv):

    M, _ = read_csv
    networkx_call(M, gpubenchmark)


def test_wsorensen_multi_column_weights(gpubenchmark, read_csv):

    M, cu_M = read_csv

    cu_coeff = cugraph_call(gpubenchmark, cu_M)
    nx_coeff = networkx_call(M)
    for i in range(len(cu_coeff)):
        diff = abs(nx_coeff[i] - cu_coeff[i])
        assert diff < 1.0e-6


def test_wsorensen_multi_column(read_csv):

    M, _ = read_csv

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"]
    )

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]

    weight_arr = cudf.Series(np.ones(G2.number_of_vertices(), dtype=np.float32))
    weights = cudf.DataFrame()
    weights["vertex"] = G2.nodes()
    weights["vertex_"] = weights["vertex"] + 1000
    weights["weight"] = weight_arr

    df_res = cugraph.sorensen_w(G1, weights, vertex_pair)

    weights = weights[["vertex", "weight"]]
    df_exp = cugraph.sorensen_w(G2, weights, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_first").reset_index()
    expected = df_exp.sort_values("first").reset_index()
    assert_series_equal(actual["sorensen_coeff"], expected["sorensen_coeff"])

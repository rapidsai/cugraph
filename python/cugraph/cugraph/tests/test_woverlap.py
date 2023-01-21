# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import scipy
import numpy as np
import cudf
from cudf.testing import assert_series_equal
from cugraph.experimental.datasets import DATASETS_UNDIRECTED

import cugraph
from cugraph.testing import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_call(benchmark_callable, graph_file, pairs):
    # Device data
    cu_M = graph_file.get_edgelist()
    weights_arr = cudf.Series(
        np.ones(max(cu_M["src"].max(), cu_M["dst"].max()) + 1, dtype=np.float32)
    )
    weights = cudf.DataFrame()
    weights["vertex"] = np.arange(len(weights_arr), dtype=np.int32)
    weights["weight"] = weights_arr

    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))

    # cugraph Overlap Call
    df = benchmark_callable(cugraph.overlap_w, G, weights, pairs)

    df = df.sort_values(by=["first", "second"])
    return df["overlap_coeff"].to_numpy()


def intersection(a, b, M):
    count = 0
    a_idx = M.indptr[a]
    b_idx = M.indptr[b]

    while (a_idx < M.indptr[a + 1]) and (b_idx < M.indptr[b + 1]):
        a_vertex = M.indices[a_idx]
        b_vertex = M.indices[b_idx]

        if a_vertex == b_vertex:
            count += 1
            a_idx += 1
            b_idx += 1
        elif a_vertex < b_vertex:
            a_idx += 1
        else:
            b_idx += 1

    return count


def degree(a, M):
    return M.indptr[a + 1] - M.indptr[a]


def overlap(a, b, M):
    b_sum = degree(b, M)
    if b_sum == 0:
        return float("NaN")

    i = intersection(a, b, M)
    a_sum = degree(a, M)
    total = min(a_sum, b_sum)
    return i / total


def cpu_call(M, first, second):
    result = []
    for i in range(len(first)):
        result.append(overlap(first[i], second[i], M))
    return result


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_woverlap(gpubenchmark, graph_file):
    dataset_path = graph_file.get_path()
    Mnx = utils.read_csv_for_nx(dataset_path)
    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    M = scipy.sparse.csr_matrix((Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N))

    G = graph_file.get_graph(ignore_weights=True)
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    cu_coeff = cugraph_call(gpubenchmark, graph_file, pairs)
    cpu_coeff = cpu_call(M, pairs["first"], pairs["second"])
    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        if np.isnan(cpu_coeff[i]):
            assert np.isnan(cu_coeff[i])
        elif np.isnan(cu_coeff[i]):
            assert cpu_coeff[i] == cu_coeff[i]
        else:
            diff = abs(cpu_coeff[i] - cu_coeff[i])
            assert diff < 1.0e-6


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_woverlap_multi_column(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)

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

    df_res = cugraph.overlap_w(G1, weights, vertex_pair)

    weights = weights[["vertex", "weight"]]
    df_exp = cugraph.overlap_w(G2, weights, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_first").reset_index()
    expected = df_exp.sort_values("first").reset_index()
    assert_series_equal(actual["overlap_coeff"], expected["overlap_coeff"])

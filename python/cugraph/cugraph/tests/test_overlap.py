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
import numpy as np
import scipy

import cudf
from cudf.testing import assert_series_equal

import cugraph
from cugraph.tests import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def compare_overlap(cu_coeff, cpu_coeff):

    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        if np.isnan(cpu_coeff[i]):
            assert np.isnan(cu_coeff[i])
        elif np.isnan(cu_coeff[i]):
            assert cpu_coeff[i] == cu_coeff[i]
        else:
            diff = abs(cpu_coeff[i] - cu_coeff[i])
            assert diff < 1.0e-6


def cugraph_call(benchmark_callable, cu_M, pairs, edgevals=False):
    G = cugraph.DiGraph()
    # Device data
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")
    # cugraph Overlap Call
    df = benchmark_callable(cugraph.overlap, G, pairs)
    df = df.sort_values(by=["source", "destination"])
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

    a_sum = degree(a, M)

    i = intersection(a, b, M)
    total = min(a_sum, b_sum)
    return i / total


def cpu_call(M, first, second):
    result = []
    for i in range(len(first)):
        result.append(overlap(first[i], second[i], M))
    return result


# =============================================================================
# Pytest Fixtures
# =============================================================================
@pytest.fixture(scope="module", params=utils.DATASETS_UNDIRECTED)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """

    Mnx = utils.read_csv_for_nx(request.param)
    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    M = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N)
    )

    cu_M = utils.read_csv_file(request.param)
    print("cu_M is \n", cu_M)
    return M, cu_M


@pytest.fixture(scope="module")
def extract_two_hop(read_csv):
    """
    Build graph and extract two hop neighbors
    """
    G = cugraph.Graph()
    _, cu_M = read_csv
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )
    return pairs


# Test
def test_overlap(gpubenchmark, read_csv, extract_two_hop):

    M, cu_M = read_csv
    pairs = extract_two_hop

    cu_coeff = cugraph_call(gpubenchmark, cu_M, pairs)
    cpu_coeff = cpu_call(M, pairs["first"], pairs["second"])

    compare_overlap(cu_coeff, cpu_coeff)


# Test
def test_overlap_edge_vals(gpubenchmark, read_csv, extract_two_hop):

    M, cu_M = read_csv
    pairs = extract_two_hop

    cu_coeff = cugraph_call(gpubenchmark, cu_M, pairs, edgevals=True)
    cpu_coeff = cpu_call(M, pairs["first"], pairs["second"])

    compare_overlap(cu_coeff, cpu_coeff)


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_overlap_multi_column(graph_file):

    M = utils.read_csv_for_nx(graph_file)

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    G1 = cugraph.Graph()
    G1.from_cudf_edgelist(cu_M, source=["src_0", "src_1"],
                          destination=["dst_0", "dst_1"])

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]

    df_res = cugraph.overlap(G1, vertex_pair)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0",
                          destination="dst_0")
    df_exp = cugraph.overlap(G2, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_source").reset_index()
    expected = df_exp.sort_values("source").reset_index()
    assert_series_equal(actual["overlap_coeff"], expected["overlap_coeff"])

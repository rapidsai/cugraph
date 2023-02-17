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
import numpy as np
import scipy

import cudf
from cudf.testing import assert_series_equal, assert_frame_equal

from cugraph.experimental import overlap as exp_overlap

import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED


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


def cugraph_call(benchmark_callable, graph_file, pairs, edgevals=False):
    # Device data
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=False), ignore_weights=not edgevals
    )
    # cugraph Overlap Call
    df = benchmark_callable(cugraph.overlap, G, pairs)
    df = df.sort_values(by=["first", "second"]).reset_index(drop=True)
    if not edgevals:
        # experimental overlap currently only supports unweighted graphs
        df_exp = exp_overlap(G, pairs)
        df_exp = df_exp.sort_values(by=["first", "second"]).reset_index(drop=True)
        assert_frame_equal(df, df_exp, check_dtype=False, check_like=True)

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
@pytest.fixture(scope="module", params=DATASETS_UNDIRECTED)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    graph_file = request.param
    dataset_path = graph_file.get_path()
    Mnx = utils.read_csv_for_nx(dataset_path)

    N = max(max(Mnx["0"]), max(Mnx["1"])) + 1
    M = scipy.sparse.csr_matrix((Mnx.weight, (Mnx["0"], Mnx["1"])), shape=(N, N))

    return M, graph_file


@pytest.fixture(scope="module")
def extract_two_hop(read_csv):
    """
    Build graph and extract two hop neighbors
    """
    _, graph_file = read_csv
    G = graph_file.get_graph(ignore_weights=True)
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    return pairs


# Test
def test_overlap(gpubenchmark, read_csv, extract_two_hop):

    M, graph_file = read_csv
    pairs = extract_two_hop

    cu_coeff = cugraph_call(gpubenchmark, graph_file, pairs)
    cpu_coeff = cpu_call(M, pairs["first"], pairs["second"])

    compare_overlap(cu_coeff, cpu_coeff)


# Test
def test_overlap_edge_vals(gpubenchmark, read_csv, extract_two_hop):

    M, graph_file = read_csv
    pairs = extract_two_hop

    cu_coeff = cugraph_call(gpubenchmark, graph_file, pairs, edgevals=True)
    cpu_coeff = cpu_call(M, pairs["first"], pairs["second"])

    compare_overlap(cu_coeff, cpu_coeff)


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_overlap_multi_column(graph_file):
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

    vertex_pair = cu_M[["src_0", "src_1", "dst_0", "dst_1"]]
    vertex_pair = vertex_pair[:5]

    df_res = cugraph.overlap(G1, vertex_pair)
    df_plc_exp = exp_overlap(G1, vertex_pair)

    df_plc_exp = df_plc_exp.rename(
        columns={
            "0_src": "0_source",
            "0_dst": "0_destination",
            "1_src": "1_source",
            "1_dst": "1_destination",
        }
    )
    overlap_res = df_res["overlap_coeff"].sort_values().reset_index(drop=True)
    overlap_plc_exp = df_plc_exp["overlap_coeff"].sort_values().reset_index(drop=True)
    assert_series_equal(overlap_res, overlap_plc_exp)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(cu_M, source="src_0", destination="dst_0")
    df_exp = cugraph.overlap(G2, vertex_pair[["src_0", "dst_0"]])

    # Calculating mismatch
    actual = df_res.sort_values("0_first").reset_index()
    expected = df_exp.sort_values("first").reset_index()
    assert_series_equal(actual["overlap_coeff"], expected["overlap_coeff"])


def test_weighted_exp_overlap():
    karate = DATASETS_UNDIRECTED[0]
    G = karate.get_graph()
    with pytest.raises(ValueError):
        exp_overlap(G)

    G = karate.get_graph(ignore_weights=True)
    use_weight = True
    with pytest.raises(ValueError):
        exp_overlap(G, use_weight=use_weight)

# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
import cugraph
from cugraph.testing import utils, UNDIRECTED_DATASETS
from cudf.testing import assert_series_equal, assert_frame_equal

SRC_COL = "0"
DST_COL = "1"
VERTEX_PAIR_FIRST_COL = "first"
VERTEX_PAIR_SECOND_COL = "second"
OVERLAP_COEFF_COL = "overlap_coeff"
EDGE_ATT_COL = "weight"
MULTI_COL_SRC_0_COL = "src_0"
MULTI_COL_DST_0_COL = "dst_0"
MULTI_COL_SRC_1_COL = "src_1"
MULTI_COL_DST_1_COL = "dst_1"


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


def cugraph_call(benchmark_callable, graph_file, pairs, use_weight=False):
    # Device data
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=False), ignore_weights=not use_weight
    )
    # cugraph Overlap Call
    df = benchmark_callable(cugraph.overlap, G, pairs, use_weight=use_weight)
    df = df.sort_values(by=[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]).reset_index(
        drop=True
    )

    return df[OVERLAP_COEFF_COL].to_numpy()


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
@pytest.fixture(scope="module", params=UNDIRECTED_DATASETS)
def read_csv(request):
    """
    Read csv file for both networkx and cugraph
    """
    graph_file = request.param
    dataset_path = graph_file.get_path()
    Mnx = utils.read_csv_for_nx(dataset_path)

    N = max(max(Mnx[SRC_COL]), max(Mnx[DST_COL])) + 1
    M = scipy.sparse.csr_matrix(
        (Mnx.weight, (Mnx[SRC_COL], Mnx[DST_COL])), shape=(N, N)
    )

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
        .sort_values([VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL])
        .reset_index(drop=True)
    )

    return pairs


# Test
@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_overlap(gpubenchmark, read_csv, extract_two_hop, use_weight):
    M, graph_file = read_csv
    pairs = extract_two_hop

    cu_coeff = cugraph_call(gpubenchmark, graph_file, pairs, use_weight=use_weight)
    cpu_coeff = cpu_call(M, pairs[VERTEX_PAIR_FIRST_COL], pairs[VERTEX_PAIR_SECOND_COL])

    compare_overlap(cu_coeff, cpu_coeff)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
@pytest.mark.parametrize("use_weight", [False, True])
def test_directed_graph_check(graph_file, use_weight):
    M = utils.read_csv_for_nx(graph_file.get_path())
    cu_M = cudf.DataFrame()
    cu_M[SRC_COL] = cudf.Series(M[SRC_COL])
    cu_M[DST_COL] = cudf.Series(M[DST_COL])
    if use_weight:
        cu_M[EDGE_ATT_COL] = cudf.Series(M[EDGE_ATT_COL])

    G1 = cugraph.Graph(directed=True)
    weight = EDGE_ATT_COL if use_weight else None
    G1.from_cudf_edgelist(cu_M, source=SRC_COL, destination=DST_COL, weight=weight)

    vertex_pair = cu_M[[SRC_COL, DST_COL]]

    vertex_pair = vertex_pair[:5]
    with pytest.raises(ValueError):
        cugraph.overlap(
            G1, vertex_pair, do_expensive_check=False, use_weight=use_weight
        )


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
@pytest.mark.parametrize("use_weight", [False, True])
def test_overlap_multi_column(graph_file, use_weight):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)

    cu_M = cudf.DataFrame()
    cu_M[MULTI_COL_SRC_0_COL] = cudf.Series(M[SRC_COL])
    cu_M[MULTI_COL_DST_0_COL] = cudf.Series(M[DST_COL])
    cu_M[MULTI_COL_SRC_1_COL] = cu_M[MULTI_COL_SRC_0_COL] + 1000
    cu_M[MULTI_COL_DST_1_COL] = cu_M[MULTI_COL_DST_0_COL] + 1000
    if use_weight:
        cu_M[EDGE_ATT_COL] = cudf.Series(M[EDGE_ATT_COL])

    G1 = cugraph.Graph()
    weight = EDGE_ATT_COL if use_weight else None
    G1.from_cudf_edgelist(
        cu_M,
        source=[MULTI_COL_SRC_0_COL, MULTI_COL_SRC_1_COL],
        destination=[MULTI_COL_DST_0_COL, MULTI_COL_DST_1_COL],
        weight=weight,
    )

    vertex_pair = cu_M[
        [
            MULTI_COL_SRC_0_COL,
            MULTI_COL_SRC_1_COL,
            MULTI_COL_DST_0_COL,
            MULTI_COL_DST_1_COL,
        ]
    ]
    vertex_pair = vertex_pair[:5]

    df_multi_col_res = cugraph.overlap(G1, vertex_pair, use_weight=use_weight)
    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(
        cu_M, source=MULTI_COL_SRC_0_COL, destination=MULTI_COL_DST_0_COL, weight=weight
    )
    df_single_col_res = cugraph.overlap(
        G2, vertex_pair[[MULTI_COL_SRC_0_COL, MULTI_COL_DST_0_COL]]
    )

    # Calculating mismatch
    actual = df_multi_col_res.sort_values("0_src").reset_index()
    expected = df_single_col_res.sort_values(VERTEX_PAIR_FIRST_COL).reset_index()
    assert_series_equal(actual[OVERLAP_COEFF_COL], expected[OVERLAP_COEFF_COL])


@pytest.mark.sg
def test_weighted_overlap():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)
    with pytest.raises(ValueError):
        cugraph.overlap(G, use_weight=True)


@pytest.mark.sg
def test_all_pairs_overlap():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Overlap
    overlap_results = cugraph.overlap(G)

    # Remove self loop
    overlap_results = overlap_results[
        overlap_results["first"] != overlap_results["second"]
    ].reset_index(drop=True)

    all_pairs_overlap_results = cugraph.all_pairs_overlap(G)

    assert_frame_equal(
        overlap_results.head(),
        all_pairs_overlap_results.head(),
        check_dtype=False,
        check_like=True,
    )


# FIXME
@pytest.mark.sg
@pytest.mark.skip(reason="Inaccurate results returned by all-pairs similarity")
def test_all_pairs_overlap_with_vertices():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Overlap
    overlap_results = cugraph.overlap(G)

    # Remove self loop
    overlap_results = overlap_results[
        overlap_results["first"] != overlap_results["second"]
    ].reset_index(drop=True)

    vertices = [0, 1, 2]

    mask_first = overlap_results["first"].isin(vertices)
    mask_second = overlap_results["second"].isin(vertices)
    # mask = [v in vertices for v in (overlap_results['first'].to_pandas()
    # or overlap_results['second'].to_pandas())]
    mask = [f or s for (f, s) in zip(mask_first.to_pandas(), mask_second.to_pandas())]

    overlap_results = overlap_results[mask].reset_index(drop=True)

    # Call all-pairs Overlap
    all_pairs_overlap_results = cugraph.all_pairs_overlap(
        G, vertices=cudf.Series(vertices, dtype="int32")
    )

    assert_frame_equal(
        overlap_results, all_pairs_overlap_results, check_dtype=False, check_like=True
    )


@pytest.mark.sg
def test_all_pairs_overlap_with_topk():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Overlap
    overlap_results = cugraph.overlap(G)

    topk = 4

    # Remove self loop
    overlap_results = (
        overlap_results[overlap_results["first"] != overlap_results["second"]]
        .sort_values(["overlap_coeff", "first", "second"], ascending=False)
        .reset_index(drop=True)[:topk]
    )

    # Call all-pairs overlap
    all_pairs_overlap_results = (
        cugraph.all_pairs_overlap(G, topk=topk)
        .sort_values(["first", "second"], ascending=False)
        .reset_index(drop=True)
    )

    assert_frame_equal(
        overlap_results, all_pairs_overlap_results, check_dtype=False, check_like=True
    )

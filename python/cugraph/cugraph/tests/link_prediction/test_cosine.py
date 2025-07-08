# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import pandas as pd

SRC_COL = "0"
DST_COL = "1"
VERTEX_PAIR_FIRST_COL = "first"
VERTEX_PAIR_SECOND_COL = "second"
COSINE_COEFF_COL = "cosine_coeff"
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
def compare_cosine(cu_coeff, cpu_coeff, epsilon=1.0e-6):
    assert len(cu_coeff) == len(cpu_coeff)
    for i in range(len(cu_coeff)):
        if np.isnan(cpu_coeff[i]):
            assert np.isnan(cu_coeff[i])
        elif np.isnan(cu_coeff[i]):
            assert cpu_coeff[i] == cu_coeff[i]
        else:
            diff = abs(cpu_coeff[i] - cu_coeff[i])
            # Properly handle floating-point arithmetic
            assert diff <= abs(cu_coeff[i]) if \
                (abs(cpu_coeff[i]) < abs(cu_coeff[i])) else \
                    (abs(cpu_coeff[i]) * epsilon)

def cugraph_call(benchmark_callable, graph_file, pairs, use_weight=False):
    # Device data
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=False), ignore_weights=not use_weight
    )
    # cugraph Cosine Call
    df = benchmark_callable(cugraph.cosine, G, pairs, use_weight=use_weight)
    df = df.sort_values(by=[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]).reset_index(
        drop=True
    )

    return df[COSINE_COEFF_COL].to_numpy()



def cosine(a, b, M):
    # Retieve the out-degree of a
    out_degree_a = M.indices[M.indptr[a]:M.indptr[a+1]]
    # Retieve the out-degree of a
    out_degree_b = M.indices[M.indptr[b]:M.indptr[b+1]]
    
    # Find the intersection of a and b
    intersection_a_b = np.intersect1d(out_degree_a, out_degree_b)

    norm_a = 0
    norm_b = 0
    a_dot_b = 0

    for dst in intersection_a_b:
        norm_a += pow(M[a, dst], 2)
        norm_b += pow(M[b, dst], 2)
        a_dot_b += (pow(M[a, dst], 2) * pow(M[b, dst], 2))

    return a_dot_b / (pow(norm_a, 0.5) * pow(norm_b, 0.5)) if \
        (norm_a * norm_b) != 0 else 0


def cpu_call(M, first, second):
    result = []
    for i in range(len(first)):
        result.append(cosine(first[i], second[i], M))
    return result


def compare(src1, dst1, val1, src2, dst2, val2):
    #
    #  We will do comparison computations by using dataframe
    #  merge functions (essentially doing fast joins).  We
    #  start by making two data frames
    #
    df1 = cudf.DataFrame()
    df1["src1"] = src1
    df1["dst1"] = dst1
    if val1 is not None:
        df1["val1"] = val1

    df2 = cudf.DataFrame()
    df2["src2"] = src2
    df2["dst2"] = dst2
    if val2 is not None:
        df2["val2"] = val2

    #
    #  Check to see if all pairs in the original data frame
    #  still exist in the new data frame.  If we join (merge)
    #  the data frames where (src1[i]=src2[i]) and (dst1[i]=dst2[i])
    #  then we should get exactly the same number of entries in
    #  the data frame if we did not lose any data.
    #
    join = df1.merge(df2, left_on=["src1", "dst1"], right_on=["src2", "dst2"])

    if len(df1) != len(join):
        join2 = df1.merge(
            df2, how="left", left_on=["src1", "dst1"], right_on=["src2", "dst2"]
        )
        pd.set_option("display.max_rows", 500)
        print("df1 = \n", df1.sort_values(["src1", "dst1"]))
        print("df2 = \n", df2.sort_values(["src2", "dst2"]))
        print(
            "join2 = \n",
            join2.sort_values(["src1", "dst1"])
            .to_pandas()
            .query("src2.isnull()", engine="python"),
        )

    assert len(df1) == len(join)


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
def test_cosine(benchmark, read_csv, extract_two_hop, use_weight):
    M, graph_file = read_csv
    pairs = extract_two_hop

    if use_weight:
        dataset_path = graph_file.get_path()
        Mnx = utils.read_csv_for_nx(dataset_path)

        Mnx["weight"] = np.random.choice(
            np.arange(0, 1, 0.001), size=len(Mnx), replace=False)

        N = max(max(Mnx[SRC_COL]), max(Mnx[DST_COL])) + 1
        M = scipy.sparse.csr_matrix(
            (Mnx.weight, (Mnx[SRC_COL], Mnx[DST_COL])), shape=(N, N)
        )

        Mnx = Mnx.rename(columns={'0': 'src', '1': 'dst'})

        df = cudf.from_pandas(Mnx)

        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(
            df, source="src", destination="dst", edge_attr="weight")

        # cugraph Cosine Call
        cu_coeff = benchmark(cugraph.cosine, G, pairs, use_weight=True)
        cu_coeff = cu_coeff.sort_values(
            by=[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]).reset_index(
            drop=True
        )[COSINE_COEFF_COL].to_numpy()

    else:
        cu_coeff = cugraph_call(benchmark, graph_file, pairs, use_weight=use_weight)
    
    cpu_coeff = cpu_call(M, pairs[VERTEX_PAIR_FIRST_COL], pairs[VERTEX_PAIR_SECOND_COL])

    compare_cosine(cu_coeff, cpu_coeff)

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
        cugraph.cosine(
            G1, vertex_pair, use_weight=use_weight
        )

@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
@pytest.mark.parametrize("use_weight", [False, True])
def test_cosine_multi_column(graph_file, use_weight):
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

    df_multi_col_res = cugraph.cosine(G1, vertex_pair, use_weight=use_weight)
    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(
        cu_M, source=MULTI_COL_SRC_0_COL, destination=MULTI_COL_DST_0_COL, weight=weight
    )
    df_single_col_res = cugraph.cosine(
        G2, vertex_pair[[MULTI_COL_SRC_0_COL, MULTI_COL_DST_0_COL]]
    )

    # Calculating mismatch
    actual = df_multi_col_res.sort_values("0_src").reset_index()
    expected = df_single_col_res.sort_values(VERTEX_PAIR_FIRST_COL).reset_index()
    assert_series_equal(actual[COSINE_COEFF_COL], expected[COSINE_COEFF_COL])

@pytest.mark.sg
def test_weighted_cosine():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)
    with pytest.raises(ValueError):
        # input_graph' must be weighted if 'use_weight=True'
        cugraph.cosine(G, use_weight=True)

@pytest.mark.sg
def test_all_pairs_cosine():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Cosine
    cosine_results = cugraph.cosine(G)

    # Remove self loop
    cosine_results = cosine_results[
        cosine_results["first"] != cosine_results["second"]
    ].reset_index(drop=True)

    all_pairs_cosine_results = cugraph.all_pairs_cosine(G)

    assert_frame_equal(
        cosine_results.head(),
        all_pairs_cosine_results.head(),
        check_dtype=False,
        check_like=True,
    )

@pytest.mark.sg
def test_all_pairs_cosine_with_topk():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Cosine
    cosine_results = cugraph.cosine(G)

    topk = 10

    # Remove self loop
    cosine_results = (
        cosine_results[cosine_results["first"] != cosine_results["second"]]
        .sort_values(["cosine_coeff", "first", "second"], ascending=False)
        .reset_index(drop=True)  # [:topk]
    )

    # Call all-pairs cosine
    all_pairs_cosine_results = (
        cugraph.all_pairs_cosine(G, topk=topk)
        .sort_values(["first", "second"], ascending=False)
        .reset_index(drop=True)
    )

    # 1. All pair similarity might return different top pairs k pairs
    # which are still valid hence, ensure the pairs returned by all-pairs
    # exists.

    compare(
        all_pairs_cosine_results["first"],
        all_pairs_cosine_results["second"],
        all_pairs_cosine_results["cosine_coeff"],
        cosine_results["first"],
        cosine_results["second"],
        cosine_results["cosine_coeff"],
    )

    # 2. Ensure the coefficient scores are still the highest
    assert_series_equal(
        all_pairs_cosine_results["cosine_coeff"],
        cosine_results["cosine_coeff"][:topk],
    )

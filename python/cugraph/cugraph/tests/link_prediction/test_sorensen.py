# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
import networkx as nx

import cudf
import cugraph
from cugraph.testing import utils, UNDIRECTED_DATASETS
from cugraph.datasets import netscience
from cudf.testing import assert_series_equal, assert_frame_equal

SRC_COL = "0"
DST_COL = "1"
VERTEX_PAIR_FIRST_COL = "first"
VERTEX_PAIR_SECOND_COL = "second"
SORENSEN_COEFF_COL = "sorensen_coeff"
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
def compare_sorensen_two_hop(G, Gnx, use_weight=False):
    """
    Compute both cugraph and nx sorensen after extracting the two hop neighbors
    from G and compare both results
    """
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values([VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL])
        .reset_index(drop=True)
    )

    # print(f'G = {G.edgelist.edgelist_df}')

    df = cugraph.sorensen(G, pairs, use_weight=use_weight)
    df = df.sort_values(by=[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]).reset_index(
        drop=True
    )

    nx_pairs = list(pairs.to_records(index=False))

    # print(f'nx_pairs = {len(nx_pairs)}')

    preds = nx.jaccard_coefficient(Gnx, nx_pairs)

    # FIXME: Use known correct values of Sorensen for few graphs,
    # hardcode it and compare to Cugraph Sorensen to get a more robust test

    # Conversion from Networkx Jaccard to Sorensen
    # No networkX equivalent

    nx_coeff = list(map(lambda x: (2 * x[2]) / (1 + x[2]), preds))

    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df[SORENSEN_COEFF_COL].iloc[i])
        assert diff < 1.0e-6


def cugraph_call(benchmark_callable, graph_file, input_df=None, use_weight=False):
    G = cugraph.Graph()
    G = graph_file.get_graph(ignore_weights=not use_weight)

    # If no vertex_pair is passed as input, 'cugraph.sorensen' will
    # compute the 'sorensen_similarity' with the two_hop_neighbor of the
    # entire graph while nx compute with the one_hop_neighbor. For better
    # comparaison, get the one_hop_neighbor of the entire graph for 'cugraph.sorensen'
    # and pass it as vertex_pair
    if isinstance(input_df, cudf.DataFrame):
        vertex_pair = input_df.rename(
            columns={SRC_COL: VERTEX_PAIR_FIRST_COL, DST_COL: VERTEX_PAIR_SECOND_COL}
        )
        vertex_pair = vertex_pair[[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]]
    else:
        vertex_pair = cudf.DataFrame(
            columns=[VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL],
            dtype=G.edgelist.edgelist_df["src"].dtype,
        )

    # cugraph Sorensen Call
    df = benchmark_callable(cugraph.sorensen, G, vertex_pair=vertex_pair)

    df = df.sort_values([VERTEX_PAIR_FIRST_COL, VERTEX_PAIR_SECOND_COL]).reset_index(
        drop=True
    )

    return (
        df[VERTEX_PAIR_FIRST_COL].to_numpy(),
        df[VERTEX_PAIR_SECOND_COL].to_numpy(),
        df[SORENSEN_COEFF_COL].to_numpy(),
    )


def networkx_call(M, benchmark_callable=None):
    sources = M[SRC_COL]
    destinations = M[DST_COL]
    edges = []
    for i in range(len(M)):
        edges.append((sources[i], destinations[i]))
        edges.append((destinations[i], sources[i]))
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print("Format conversion ... ")

    Gnx = nx.from_pandas_edgelist(
        M,
        source=SRC_COL,
        target=DST_COL,
        edge_attr=EDGE_ATT_COL,
        create_using=nx.Graph(),
    )

    # Networkx Jaccard Call
    print("Solving... ")
    if benchmark_callable is not None:
        preds = benchmark_callable(nx.jaccard_coefficient, Gnx, edges)
    else:
        preds = nx.jaccard_coefficient(Gnx, edges)

    src = []
    dst = []
    coeff = []
    for u, v, p in preds:
        src.append(u)
        dst.append(v)
        # Conversion from Networkx Jaccard to Sorensen
        # No networkX equivalent
        coeff.append((2 * p) / (1 + p))
    return src, dst, coeff


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
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)

    return M_cu, M, graph_file


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_sorensen(gpubenchmark, read_csv, use_weight):
    M_cu, M, graph_file = read_csv
    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, graph_file, input_df=M_cu, use_weight=use_weight
    )
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


@pytest.mark.sg
def test_nx_sorensen_time(gpubenchmark, read_csv):
    _, M, _ = read_csv
    nx_src, nx_dst, nx_coeff = networkx_call(M, gpubenchmark)


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_directed_graph_check(read_csv, use_weight):
    _, M, _ = read_csv

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
        cugraph.sorensen(
            G1, vertex_pair, do_expensive_check=False, use_weight=use_weight
        )


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", [netscience])
@pytest.mark.parametrize("use_weight", [False, True])
@pytest.mark.skip(reason="Skipping because this datasets is unrenumbered")
def test_sorensen_edgevals(gpubenchmark, graph_file, use_weight):
    dataset_path = netscience.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    M_cu = utils.read_csv_file(dataset_path)
    cu_src, cu_dst, cu_coeff = cugraph_call(
        gpubenchmark, netscience, input_df=M_cu, use_weight=use_weight
    )
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_sorensen_two_hop(read_csv, use_weight):
    _, M, graph_file = read_csv

    Gnx = nx.from_pandas_edgelist(
        M, source=SRC_COL, target=DST_COL, create_using=nx.Graph()
    )
    G = graph_file.get_graph(ignore_weights=not use_weight)

    compare_sorensen_two_hop(G, Gnx, use_weight=use_weight)


@pytest.mark.sg
@pytest.mark.parametrize("use_weight", [False, True])
def test_sorensen_two_hop_edge_vals(read_csv, use_weight):
    _, M, graph_file = read_csv

    Gnx = nx.from_pandas_edgelist(
        M,
        source=SRC_COL,
        target=DST_COL,
        edge_attr=EDGE_ATT_COL,
        create_using=nx.Graph(),
    )

    G = graph_file.get_graph(ignore_weights=not use_weight)

    compare_sorensen_two_hop(G, Gnx, use_weight=use_weight)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
@pytest.mark.parametrize("use_weight", [False, True])
def test_sorensen_multi_column(graph_file, use_weight):
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

    df_multi_col_res = cugraph.sorensen(G1, vertex_pair)

    G2 = cugraph.Graph()
    G2.from_cudf_edgelist(
        cu_M, source=MULTI_COL_SRC_0_COL, destination=MULTI_COL_DST_0_COL, weight=weight
    )
    df_single_col_res = cugraph.sorensen(
        G2, vertex_pair[[MULTI_COL_SRC_0_COL, MULTI_COL_DST_0_COL]]
    )

    # Calculating mismatch
    actual = df_multi_col_res.sort_values("0_src").reset_index()
    expected = df_single_col_res.sort_values(VERTEX_PAIR_FIRST_COL).reset_index()
    assert_series_equal(actual[SORENSEN_COEFF_COL], expected[SORENSEN_COEFF_COL])


@pytest.mark.sg
def test_weighted_sorensen():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)
    with pytest.raises(ValueError):
        cugraph.sorensen(G, use_weight=True)


@pytest.mark.sg
def test_all_pairs_sorensen():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Sorensen
    sorensen_results = cugraph.sorensen(G)
    
    # Remove self loop
    sorensen_results = sorensen_results[sorensen_results['first'] != sorensen_results['second']].reset_index(drop=True)
    
    all_pairs_sorensen_results = cugraph.all_pairs_sorensen(G)

    assert_frame_equal(sorensen_results.head(), all_pairs_sorensen_results.head(), check_dtype=False, check_like=True)


# FIXME
@pytest.mark.sg
@pytest.mark.skip(reason="Inaccurate results returned by all-pairs similarity")
def test_all_pairs_sorensen_with_vertices():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Sorensen
    sorensen_results = cugraph.sorensen(G)
    
    # Remove self loop
    sorensen_results = sorensen_results[sorensen_results['first'] != sorensen_results['second']].reset_index(drop=True)

    vertices = [0, 1, 2]

    mask_first = sorensen_results['first'].isin(vertices)
    mask_second = sorensen_results['second'].isin(vertices)
    # mask = [v in vertices for v in (sorensen_results['first'].to_pandas() or sorensen_results['second'].to_pandas())]
    mask = [f or s for (f, s) in zip(mask_first.to_pandas(), mask_second.to_pandas())]

    sorensen_results = sorensen_results[mask].reset_index(drop=True)

    # Call all-pairs Sorensen
    all_pairs_sorensen_results = cugraph.all_pairs_sorensen(G, vertices=cudf.Series(vertices, dtype="int32"))

    assert_frame_equal(sorensen_results, all_pairs_sorensen_results, check_dtype=False, check_like=True)


@pytest.mark.sg
def test_all_pairs_sorensen_with_topk():
    karate = UNDIRECTED_DATASETS[0]
    G = karate.get_graph(ignore_weights=True)

    # Call Sorensen
    sorensen_results = cugraph.sorensen(G)

    topk = 4
    
    # Remove self loop
    sorensen_results = sorensen_results[sorensen_results['first'] != sorensen_results['second']].\
        sort_values(["sorensen_coeff", "first", "second"], ascending=False).reset_index(drop=True)[:topk]

    # Call all-pairs sorensen
    all_pairs_sorensen_results = cugraph.all_pairs_sorensen(G, topk=topk).sort_values(["first", "second"], ascending=False).reset_index(drop=True)

    assert_frame_equal(sorensen_results, all_pairs_sorensen_results, check_dtype=False, check_like=True)

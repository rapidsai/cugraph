# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
import cupy as cp
import numpy as np
from scipy.sparse import coo_matrix as sp_coo_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
from scipy.sparse import csc_matrix as sp_csc_matrix

import cudf
import cugraph
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from pylibcugraph.testing.utils import gen_fixture_params_product
from cugraph.testing import (
    utils,
    get_resultset,
    load_resultset,
    DEFAULT_DATASETS,
    SMALL_DATASETS,
)


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [True, False]

SUBSET_SEED_OPTIONS = [42]

DATASET_STARTS = {
    "dolphins": 16,
    "karate": 7,
    "karate-disjoint": 19,
    "netscience": 1237,
}

DEFAULT_EPSILON = 1e-6

DEPTH_LIMITS = [None, 1, 5, 18]

# Map of cuGraph input types to the expected output type for cuGraph
# connected_components calls.
cuGraph_input_output_map = {
    cugraph.Graph: cudf.DataFrame,
    cp_coo_matrix: tuple,
    cp_csr_matrix: tuple,
    cp_csc_matrix: tuple,
    sp_coo_matrix: tuple,
    sp_csr_matrix: tuple,
    sp_csc_matrix: tuple,
}
cupy_types = [cp_coo_matrix, cp_csr_matrix, cp_csc_matrix]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Helper functions
# =============================================================================
def convert_output_to_cudf(input_G_or_matrix, cugraph_result):
    """
    Convert cugraph_result to a cudf DataFrame. The conversion is based on the
    type of input_G_or_matrix, since different input types result in different
    cugraph_result types (see cugraph_input_output_map).
    """
    input_type = type(input_G_or_matrix)
    expected_return_type = cuGraph_input_output_map[type(input_G_or_matrix)]
    assert type(cugraph_result) is expected_return_type

    if expected_return_type is cudf.DataFrame:
        return cugraph_result

    # A CuPy/SciPy input means the return value will be a 2-tuple of:
    #   distance: cupy.ndarray
    #      ndarray of shortest distances between source and vertex.
    #   predecessor: cupy.ndarray
    #      ndarray of predecessors of a vertex on the path from source, which
    #      can be used to reconstruct the shortest paths.
    # or a 3-tuple of the above 2 plus
    #   sp_counter: cupy.ndarray
    #      for the i'th position in the array, the number of shortest paths
    #      leading to the vertex at position i in the (input) vertex array.
    elif expected_return_type is tuple:
        if input_type in cupy_types:
            assert type(cugraph_result[0]) is cp.ndarray
            assert type(cugraph_result[1]) is cp.ndarray
            if len(cugraph_result) == 3:
                assert type(cugraph_result[2]) is cp.ndarray
        else:
            assert type(cugraph_result[0]) is np.ndarray
            assert type(cugraph_result[1]) is np.ndarray
            if len(cugraph_result) == 3:
                assert type(cugraph_result[2]) is np.ndarray

        # Get unique verts from input since they are not incuded in output
        if type(input_G_or_matrix) in [
            cp_csr_matrix,
            cp_csc_matrix,
            sp_csr_matrix,
            sp_csc_matrix,
        ]:
            coo = input_G_or_matrix.tocoo(copy=False)
        else:
            coo = input_G_or_matrix
        verts = sorted(set([n.item() for n in coo.col] + [n.item() for n in coo.row]))
        dists = [n.item() for n in cugraph_result[0]]
        preds = [n.item() for n in cugraph_result[1]]
        assert len(verts) == len(dists) == len(preds)

        d = {"vertex": verts, "distance": dists, "predecessor": preds}

        if len(cugraph_result) == 3:
            counters = [n.item() for n in cugraph_result[2]]
            assert len(counters) == len(verts)
            d.update({"sp_counter": counters})

        return cudf.DataFrame(d)

    else:
        raise RuntimeError(f"unsupported return type: {expected_return_type}")


# NOTE: We need to use relative error, the values of the shortest path
# counters can reach extremely high values 1e+80 and above
def compare_single_sp_counter(result, expected, epsilon=DEFAULT_EPSILON):
    return np.isclose(result, expected, rtol=epsilon)


def compare_bfs(benchmark_callable, G, golden_values, start_vertex, depth_limit):
    """
    Generate both cugraph and reference bfs traversal.
    """

    if isinstance(start_vertex, int):
        result = benchmark_callable(cugraph.bfs_edges, G, start_vertex)
        cugraph_df = convert_output_to_cudf(G, result)
        compare_func = _compare_bfs

        # NOTE: We need to take 2 different path for verification as the nx
        #       functions used as reference return dictionaries that might
        #       not contain all the vertices while the cugraph version return
        #       a cudf.DataFrame with all the vertices, also some verification
        #       become slow with the data transfer
        compare_func(cugraph_df, golden_values, start_vertex)

    elif isinstance(start_vertex, list):  # For other Verifications
        all_golden_values = golden_values
        all_cugraph_distances = []

        def func_to_benchmark():
            for sv in start_vertex:
                cugraph_df = cugraph.bfs_edges(G, sv, depth_limit=depth_limit)
                all_cugraph_distances.append(cugraph_df)

        benchmark_callable(func_to_benchmark)

        compare_func = _compare_bfs
        for (i, sv) in enumerate(start_vertex):
            cugraph_df = convert_output_to_cudf(G, all_cugraph_distances[i])

            compare_func(cugraph_df, all_golden_values[i], sv)

    else:  # Unknown type given to seed
        raise NotImplementedError("Invalid type for start_vertex")


def _compare_bfs(cugraph_df, golden_distances, source):
    # This call should only contain 3 columns:
    # 'vertex', 'distance', 'predecessor'
    # It also confirms wether or not 'sp_counter' has been created by the call
    # 'sp_counter' triggers atomic operations in BFS, thus we want to make
    # sure that it was not the case
    # NOTE: 'predecessor' is always returned while the C++ function allows to
    # pass a nullptr
    assert len(cugraph_df.columns) == 3, (
        "The result of the BFS has an invalid " "number of columns"
    )
    cu_distances = {
        vertex: dist
        for vertex, dist in zip(
            cugraph_df["vertex"].to_numpy(), cugraph_df["distance"].to_numpy()
        )
    }
    cu_predecessors = {
        vertex: dist
        for vertex, dist in zip(
            cugraph_df["vertex"].to_numpy(), cugraph_df["predecessor"].to_numpy()
        )
    }

    # FIXME: The following only verifies vertices that were reached
    #       by cugraph's BFS.
    # We assume that the distances are given back as integers in BFS
    # max_val = np.iinfo(df['distance'].dtype).max
    # Unreached vertices have a distance of max_val
    missing_vertex_error = 0
    distance_mismatch_error = 0
    invalid_predecessor_error = 0
    for vertex in golden_distances:
        if vertex in cu_distances:
            result = cu_distances[vertex]
            expected = golden_distances[vertex]
            if result != expected:
                print(
                    "[ERR] Mismatch on distances: "
                    "vid = {}, cugraph = {}, golden = {}".format(
                        vertex, result, expected
                    )
                )
                distance_mismatch_error += 1
            if vertex not in cu_predecessors:
                missing_vertex_error += 1
            else:
                pred = cu_predecessors[vertex]
                if vertex != source and pred not in golden_distances:
                    invalid_predecessor_error += 1
                else:
                    # The graph is unweighted thus, predecessors are 1 away
                    if vertex != source and (
                        (golden_distances[pred] + 1 != cu_distances[vertex])
                    ):
                        print(
                            "[ERR] Invalid on predecessors: "
                            "vid = {}, cugraph = {}".format(vertex, pred)
                        )
                        invalid_predecessor_error += 1
        else:
            missing_vertex_error += 1
    assert missing_vertex_error == 0, "There are missing vertices"
    assert distance_mismatch_error == 0, "There are invalid distances"
    assert invalid_predecessor_error == 0, "There are invalid predecessors"


def get_cu_graph_and_params(dataset, directed):
    """
    Helper for fixtures returning a cuGraph obj and params.
    """
    # create graph
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed))
    dataset_path = dataset.get_path()
    dataset_name = dataset.metadata["name"]
    return (G, dataset_path, dataset_name, directed)


def get_cu_graph_golden_results_and_params(
    depth_limit, G, dataset_path, dataset_name, directed, _
):
    """
    Helper for fixtures returning golden results and params.
    """
    start_vertex = DATASET_STARTS[dataset_name]
    golden_values = get_resultset(
        resultset_name="traversal",
        algo="single_source_shortest_path_length",
        cutoff=str(depth_limit),
        graph_dataset=dataset_name,
        graph_directed=str(directed),
        start_vertex=str(start_vertex),
    )

    golden_values = cudf.Series(
        golden_values.distance.values, index=golden_values.vertex
    ).to_dict()

    return (G, dataset_path, directed, golden_values, start_vertex, depth_limit)


# =============================================================================
# Pytest Fixtures
# =============================================================================
SEEDS = [pytest.param(s) for s in SUBSET_SEED_OPTIONS]
DIRECTED = [pytest.param(d) for d in DIRECTED_GRAPH_OPTIONS]
DATASETS = [pytest.param(d) for d in DEFAULT_DATASETS]
SMALL_DATASETS = [pytest.param(d) for d in SMALL_DATASETS]
DEPTH_LIMIT = [pytest.param(d) for d in DEPTH_LIMITS]

# Call gen_fixture_params_product() to caluculate the cartesian product of
# multiple lists of params. This is required since parameterized fixtures do
# not do this automatically (unlike multiply-parameterized tests). The 2nd
# item in the tuple is a label for the param value used when displaying the
# full test name.
algo_test_fixture_params = gen_fixture_params_product((DEPTH_LIMIT, "depth_limit"))

graph_fixture_params = gen_fixture_params_product(
    (DATASETS, "ds"), (DIRECTED, "dirctd")
)

small_graph_fixture_params = gen_fixture_params_product(
    (SMALL_DATASETS, "ds"), (DIRECTED, "dirctd")
)


# The single param list variants are used when only 1 param combination is
# needed (eg. testing non-native input types where tests for other combinations
# was covered elsewhere).
single_algo_test_fixture_params = gen_fixture_params_product(
    ([DEPTH_LIMIT[0]], "depth_limit")
)

single_small_graph_fixture_params = gen_fixture_params_product(
    ([SMALL_DATASETS[0]], "ds"), (DIRECTED, "dirctd")
)


# Fixture that loads all golden results necessary to run cugraph tests if the
# tests are not already present in the designated results directory. Most of the
# time, this will only check if the module-specific mapping file exists.
@pytest.fixture(scope="module")
def load_traversal_results():
    load_resultset(
        "traversal", "https://data.rapids.ai/cugraph/results/resultsets.tar.gz"
    )


# Fixtures that result in a test-per (dataset X directed/undirected)
# combination. These return the path to the dataset, a bool indicating if a
# directed graph is being used, and the Nx graph object.
@pytest.fixture(scope="module", params=graph_fixture_params)
def dataset_golden_results(request):
    return get_cu_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=small_graph_fixture_params)
def small_dataset_golden_results(request):
    return get_cu_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=single_small_graph_fixture_params)
def single_small_dataset_golden_results(request):
    return get_cu_graph_and_params(*request.param)


# Fixtures that result in a test-per (dataset_nx_graph combinations X algo_test
# param combinations) combination. These run Nx BFS on the Nx graph obj and
# return the path to the dataset, if a directed graph is being used, the Nx BFS
# results, the starting vertex for BFS, and flag if shortes path counting was
# used.
@pytest.fixture(scope="module", params=algo_test_fixture_params)
def dataset_goldenresults_startvertex_spc(
    dataset_golden_results, load_traversal_results, request
):
    return get_cu_graph_golden_results_and_params(
        *request.param, *dataset_golden_results, load_traversal_results
    )


@pytest.fixture(scope="module", params=single_algo_test_fixture_params)
def single_dataset_goldenresults_startvertex_spc(
    single_small_dataset_golden_results, load_traversal_results, request
):
    return get_cu_graph_golden_results_and_params(
        *request.param, *single_small_dataset_golden_results, load_traversal_results
    )


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.sg
@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs(gpubenchmark, dataset_goldenresults_startvertex_spc, cugraph_input_type):
    """
    Test BFS traversal on random source with distance and predecessors
    """
    (
        G,
        dataset,
        directed,
        golden_values,
        start_vertex,
        depth_limit,
    ) = dataset_goldenresults_startvertex_spc

    if directed:
        if isinstance(cugraph_input_type, cugraph.Graph):
            cugraph_input_type = cugraph.Graph(directed=True)

    if not isinstance(cugraph_input_type, cugraph.Graph):
        G_or_matrix = utils.create_obj_from_csv(dataset, cugraph_input_type)
    else:
        G_or_matrix = G

    compare_bfs(gpubenchmark, G_or_matrix, golden_values, start_vertex, depth_limit)


@pytest.mark.sg
@pytest.mark.parametrize("cugraph_input_type", utils.MATRIX_INPUT_TYPES)
def test_bfs_nonnative_inputs_matrix(
    gpubenchmark, single_dataset_goldenresults_startvertex_spc, cugraph_input_type
):
    test_bfs(
        gpubenchmark, single_dataset_goldenresults_startvertex_spc, cugraph_input_type
    )


@pytest.mark.sg
def test_bfs_nonnative_inputs_nx(
    single_dataset_goldenresults_startvertex_spc,
):
    (
        _,
        _,
        directed,
        golden_values,
        start_vertex,
        _,
    ) = single_dataset_goldenresults_startvertex_spc

    cugraph_df = get_resultset(
        resultset_name="traversal",
        algo="bfs_edges",
        graph_dataset="karate",
        graph_directed=str(directed),
        source=str(start_vertex),
    )

    compare_func = _compare_bfs
    compare_func(cugraph_df, golden_values, start_vertex)


@pytest.mark.sg
@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs_invalid_start(dataset_goldenresults_startvertex_spc, cugraph_input_type):
    (G, _, _, _, start_vertex, depth_limit) = dataset_goldenresults_startvertex_spc

    el = G.view_edge_list()

    newval = max(el.src.max(), el.dst.max()) + 1
    start_vertex = newval

    with pytest.raises(ValueError):
        cugraph.bfs(G, start_vertex, depth_limit=depth_limit)


@pytest.mark.sg
def test_scipy_api_compat():
    graph_file = DEFAULT_DATASETS[0]
    dataset_path = graph_file.get_path()

    input_cugraph_graph = graph_file.get_graph(ignore_weights=True)

    input_coo_matrix = utils.create_obj_from_csv(
        dataset_path, cp_coo_matrix, edgevals=True
    )
    # Ensure scipy-only options are rejected for cugraph inputs
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph, start=0, directed=False)
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph)  # required arg missing

    # Ensure cugraph-compatible options work as expected
    cugraph.bfs(input_cugraph_graph, i_start=0)
    cugraph.bfs(input_cugraph_graph, i_start=0)
    # cannot have start and i_start
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph, start=0, i_start=0)

    # Ensure SciPy options for matrix inputs work as expected
    cugraph.bfs(input_coo_matrix, i_start=0)
    cugraph.bfs(input_coo_matrix, i_start=0, directed=True)
    cugraph.bfs(input_coo_matrix, i_start=0, directed=False)
    result = cugraph.bfs(input_coo_matrix, i_start=0)
    assert type(result) is tuple
    assert len(result) == 2

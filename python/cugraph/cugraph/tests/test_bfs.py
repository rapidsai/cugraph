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
import random

import pytest
import pandas as pd
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from scipy.sparse import coo_matrix as sp_coo_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
from scipy.sparse import csc_matrix as sp_csc_matrix
import cudf
from pylibcugraph.testing.utils import gen_fixture_params_product

import cugraph
from cugraph.testing import utils
from cugraph.experimental import datasets

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx
    import networkx.algorithms.centrality.betweenness as nxacb


# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [True, False]

SUBSET_SEED_OPTIONS = [42]

DEFAULT_EPSILON = 1e-6

DEPTH_LIMITS = [None, 1, 5, 18]

# Map of cuGraph input types to the expected output type for cuGraph
# connected_components calls.
cuGraph_input_output_map = {
    cugraph.Graph: cudf.DataFrame,
    nx.Graph: pd.DataFrame,
    nx.DiGraph: pd.DataFrame,
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

    elif expected_return_type is pd.DataFrame:
        return cudf.from_pandas(cugraph_result)

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


def compare_bfs(benchmark_callable, G, nx_values, start_vertex, depth_limit):
    """
    Genereate both cugraph and reference bfs traversal.
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
        compare_func(cugraph_df, nx_values, start_vertex)

    elif isinstance(start_vertex, list):  # For other Verifications
        all_nx_values = nx_values
        all_cugraph_distances = []

        def func_to_benchmark():
            for sv in start_vertex:
                cugraph_df = cugraph.bfs_edges(G, sv, depth_limit=depth_limit)
                all_cugraph_distances.append(cugraph_df)

        benchmark_callable(func_to_benchmark)

        compare_func = _compare_bfs
        for (i, sv) in enumerate(start_vertex):
            cugraph_df = convert_output_to_cudf(G, all_cugraph_distances[i])

            compare_func(cugraph_df, all_nx_values[i], sv)

    else:  # Unknown type given to seed
        raise NotImplementedError("Invalid type for start_vertex")


def _compare_bfs(cugraph_df, nx_distances, source):
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
    for vertex in nx_distances:
        if vertex in cu_distances:
            result = cu_distances[vertex]
            expected = nx_distances[vertex]
            if result != expected:
                print(
                    "[ERR] Mismatch on distances: "
                    "vid = {}, cugraph = {}, nx = {}".format(vertex, result, expected)
                )
                distance_mismatch_error += 1
            if vertex not in cu_predecessors:
                missing_vertex_error += 1
            else:
                pred = cu_predecessors[vertex]
                if vertex != source and pred not in nx_distances:
                    invalid_predecessor_error += 1
                else:
                    # The graph is unweighted thus, predecessors are 1 away
                    if vertex != source and (
                        (nx_distances[pred] + 1 != cu_distances[vertex])
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


def get_cu_graph_nx_graph_and_params(dataset, directed):
    """
    Helper for fixtures returning a Nx graph obj and params.
    """
    # create graph
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed))
    dataset_path = dataset.get_path()

    return (
        G,
        dataset_path,
        directed,
        utils.generate_nx_graph_from_file(dataset_path, directed),
    )


def get_cu_graph_nx_results_and_params(seed, depth_limit, G, dataset, directed, Gnx):
    """
    Helper for fixtures returning Nx results and params.
    """
    random.seed(seed)
    start_vertex = random.sample(list(Gnx.nodes()), 1)[0]

    nx_values = nx.single_source_shortest_path_length(
        Gnx, start_vertex, cutoff=depth_limit
    )

    return (G, dataset, directed, nx_values, start_vertex, depth_limit)


# =============================================================================
# Pytest Fixtures
# =============================================================================
SEEDS = [pytest.param(s) for s in SUBSET_SEED_OPTIONS]
DIRECTED = [pytest.param(d) for d in DIRECTED_GRAPH_OPTIONS]
DATASETS = [pytest.param(d) for d in datasets.DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in datasets.DATASETS_SMALL]
DEPTH_LIMIT = [pytest.param(d) for d in DEPTH_LIMITS]

# Call gen_fixture_params_product() to caluculate the cartesian product of
# multiple lists of params. This is required since parameterized fixtures do
# not do this automatically (unlike multiply-parameterized tests). The 2nd
# item in the tuple is a label for the param value used when displaying the
# full test name.
algo_test_fixture_params = gen_fixture_params_product(
    (SEEDS, "seed"), (DEPTH_LIMIT, "depth_limit")
)

graph_fixture_params = gen_fixture_params_product(
    (DATASETS, "ds"), (DIRECTED, "dirctd")
)

small_graph_fixture_params = gen_fixture_params_product(
    (DATASETS_SMALL, "ds"), (DIRECTED, "dirctd")
)

# The single param list variants are used when only 1 param combination is
# needed (eg. testing non-native input types where tests for other combinations
# was covered elsewhere).
single_algo_test_fixture_params = gen_fixture_params_product(
    ([SEEDS[0]], "seed"), ([DEPTH_LIMIT[0]], "depth_limit")
)

single_small_graph_fixture_params = gen_fixture_params_product(
    ([DATASETS_SMALL[0]], "ds"), (DIRECTED, "dirctd")
)


# Fixtures that result in a test-per (dataset X directed/undirected)
# combination. These return the path to the dataset, a bool indicating if a
# directed graph is being used, and the Nx graph object.
@pytest.fixture(scope="module", params=graph_fixture_params)
def dataset_nx_graph(request):
    return get_cu_graph_nx_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=small_graph_fixture_params)
def small_dataset_nx_graph(request):
    return get_cu_graph_nx_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=single_small_graph_fixture_params)
def single_small_dataset_nx_graph(request):
    return get_cu_graph_nx_graph_and_params(*request.param)


# Fixtures that result in a test-per (dataset_nx_graph combinations X algo_test
# param combinations) combination. These run Nx BFS on the Nx graph obj and
# return the path to the dataset, if a directed graph is being used, the Nx BFS
# results, the starting vertex for BFS, and flag if shortes path counting was
# used.
@pytest.fixture(scope="module", params=algo_test_fixture_params)
def dataset_nxresults_startvertex_spc(dataset_nx_graph, request):
    return get_cu_graph_nx_results_and_params(*request.param, *dataset_nx_graph)


@pytest.fixture(scope="module", params=single_algo_test_fixture_params)
def single_dataset_nxresults_startvertex_spc(single_small_dataset_nx_graph, request):
    return get_cu_graph_nx_results_and_params(
        *request.param, *single_small_dataset_nx_graph
    )


@pytest.fixture(scope="module")
def dataset_nxresults_allstartvertices_spc(small_dataset_nx_graph):

    dataset, directed, Gnx = small_dataset_nx_graph
    use_spc = True

    start_vertices = [start_vertex for start_vertex in Gnx]

    all_nx_values = []
    for start_vertex in start_vertices:
        _, _, nx_sp_counter = nxacb._single_source_shortest_path_basic(
            Gnx, start_vertex
        )
        nx_values = nx_sp_counter
        all_nx_values.append(nx_values)

    return (dataset, directed, all_nx_values, start_vertices, use_spc)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs(gpubenchmark, dataset_nxresults_startvertex_spc, cugraph_input_type):
    """
    Test BFS traversal on random source with distance and predecessors
    """
    (
        G,
        dataset,
        directed,
        nx_values,
        start_vertex,
        depth_limit,
    ) = dataset_nxresults_startvertex_spc

    # special case: ensure cugraph and Nx Graph types are DiGraphs if
    # "directed" is set, since the graph type parameterization is currently
    # independent of the directed parameter. Unfortunately this does not
    # change the "id" in the pytest output. Ignore for nonnative inputs
    if directed:
        if isinstance(cugraph_input_type, cugraph.Graph):
            cugraph_input_type = cugraph.Graph(directed=True)
        elif cugraph_input_type is nx.Graph:
            cugraph_input_type = nx.DiGraph

    if not isinstance(cugraph_input_type, cugraph.Graph):
        G_or_matrix = utils.create_obj_from_csv(dataset, cugraph_input_type)
    else:
        G_or_matrix = G

    compare_bfs(gpubenchmark, G_or_matrix, nx_values, start_vertex, depth_limit)


@pytest.mark.parametrize(
    "cugraph_input_type", utils.NX_INPUT_TYPES + utils.MATRIX_INPUT_TYPES
)
def test_bfs_nonnative_inputs(
    gpubenchmark, single_dataset_nxresults_startvertex_spc, cugraph_input_type
):
    test_bfs(gpubenchmark, single_dataset_nxresults_startvertex_spc, cugraph_input_type)


@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs_invalid_start(
    gpubenchmark, dataset_nxresults_startvertex_spc, cugraph_input_type
):
    (
        G,
        dataset,
        directed,
        nx_values,
        start_vertex,
        depth_limit,
    ) = dataset_nxresults_startvertex_spc

    el = G.view_edge_list()

    newval = max(el.src.max(), el.dst.max()) + 1
    start_vertex = newval

    with pytest.raises(ValueError):
        cugraph.bfs(G, start_vertex, depth_limit=depth_limit)


def test_scipy_api_compat():
    graph_file = datasets.DATASETS[0]
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

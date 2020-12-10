# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import pandas
import cupy
import numpy as np
import cudf
import pytest
import cugraph
from cugraph.tests import utils
import random

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

import pandas as pd
import cupy as cp
from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse.csr import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.csc import csc_matrix as cp_csc_matrix
from scipy.sparse.coo import coo_matrix as sp_coo_matrix
from scipy.sparse.csr import csr_matrix as sp_csr_matrix
from scipy.sparse.csc import csc_matrix as sp_csc_matrix

# =============================================================================
# Parameters
# =============================================================================
DIRECTED_GRAPH_OPTIONS = [True, False]

SUBSET_SEED_OPTIONS = [42]

DEFAULT_EPSILON = 1e-6


# Map of cuGraph input types to the expected output type for cuGraph
# connected_components calls.
cuGraph_input_output_map = {
    cugraph.Graph: cudf.DataFrame,
    cugraph.DiGraph: cudf.DataFrame,
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
        if type(input_G_or_matrix) in [cp_csr_matrix, cp_csc_matrix,
                                       sp_csr_matrix, sp_csc_matrix]:
            coo = input_G_or_matrix.tocoo(copy=False)
        else:
            coo = input_G_or_matrix
        verts = sorted(set([n.item() for n in coo.col] +
                           [n.item() for n in coo.row]))
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


def compare_bfs(benchmark_callable, G, nx_values, start_vertex,
                return_sp_counter=False):
    """
    Genereate both cugraph and reference bfs traversal.
    """
    if isinstance(start_vertex, int):
        result = benchmark_callable(cugraph.bfs_edges, G, start_vertex,
                                    return_sp_counter=return_sp_counter)
        cugraph_df = convert_output_to_cudf(G, result)

        if return_sp_counter:
            # This call should only contain 3 columns:
            # 'vertex', 'distance', 'predecessor', 'sp_counter'
            assert len(cugraph_df.columns) == 4, (
                "The result of the BFS has an invalid " "number of columns"
            )

        if return_sp_counter:
            compare_func = _compare_bfs_spc

        else:
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
                cugraph_df = cugraph.bfs_edges(
                    G, sv, return_sp_counter=return_sp_counter)
                all_cugraph_distances.append(cugraph_df)

        benchmark_callable(func_to_benchmark)

        compare_func = _compare_bfs_spc if return_sp_counter else _compare_bfs
        for (i, sv) in enumerate(start_vertex):
            cugraph_df = convert_output_to_cudf(G, all_cugraph_distances[i])
            if return_sp_counter:
                assert len(cugraph_df.columns) == 4, (
                    "The result of the BFS has an invalid " "number of columns"
                )
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
            cugraph_df["vertex"].to_array(), cugraph_df["distance"].to_array()
        )
    }
    cu_predecessors = {
        vertex: dist
        for vertex, dist in zip(
                cugraph_df["vertex"].to_array(),
                cugraph_df["predecessor"].to_array()
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
                    "vid = {}, cugraph = {}, nx = {}".format(
                        vertex, result, expected
                    )
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


def _compare_bfs_spc(cugraph_df, nx_sp_counter, unused):
    """
    Compare BFS with shortest path counters.
    """
    sorted_nx = [nx_sp_counter[key] for key in sorted(nx_sp_counter.keys())]
    # We are not checking for distances / predecessors here as we assume
    # that these have been checked  in the _compare_bfs tests
    # We focus solely on shortest path counting

    # cugraph return a dataframe that should contain exactly one time each
    # vertex
    # We could us isin to filter only vertices that are common to both
    # But it would slow down the comparison, and in this specific case
    # nxacb._single_source_shortest_path_basic is a dictionary containing all
    # the vertices.
    # There is no guarantee when we get `df` that the vertices are sorted
    # thus we enforce the order so that we can leverage faster comparison after
    sorted_df = cugraph_df.sort_values("vertex").rename(
        columns={"sp_counter": "cu_spc"}, copy=False
    )

    # This allows to detect vertices identifier that could have been
    # wrongly present multiple times
    cu_vertices = set(sorted_df['vertex'].values_host)
    nx_vertices = nx_sp_counter.keys()
    assert len(cu_vertices.intersection(nx_vertices)) == len(
        nx_vertices
    ), "There are missing vertices"

    # We add the nx shortest path counter in the cudf.DataFrame, both the
    # the DataFrame and `sorted_nx` are sorted base on vertices identifiers
    sorted_df["nx_spc"] = sorted_nx

    # We could use numpy.isclose or cupy.isclose, we can then get the entries
    # in the cudf.DataFrame where there are is a mismatch.
    # numpy / cupy allclose would get only a boolean and we might want the
    # extra information about the discrepancies
    shortest_path_counter_errors = sorted_df[
        ~cupy.isclose(
            sorted_df["cu_spc"], sorted_df["nx_spc"], rtol=DEFAULT_EPSILON
        )
    ]
    if len(shortest_path_counter_errors) > 0:
        print(shortest_path_counter_errors)
    assert len(shortest_path_counter_errors) == 0, (
        "Shortest path counters " "are too different"
    )


def get_nx_graph_and_params(dataset, directed):
    """
    Helper for fixtures returning a Nx graph obj and params.
    """
    return (dataset, directed,
            utils.generate_nx_graph_from_file(dataset, directed))


def get_nx_results_and_params(seed, use_spc, dataset, directed, Gnx):
    """
    Helper for fixtures returning Nx results and params.
    """
    random.seed(seed)
    start_vertex = random.sample(Gnx.nodes(), 1)[0]

    if use_spc:
        _, _, nx_sp_counter = \
            nxacb._single_source_shortest_path_basic(Gnx, start_vertex)
        nx_values = nx_sp_counter
    else:
        nx_values = nx.single_source_shortest_path_length(Gnx, start_vertex)

    return (dataset, directed, nx_values, start_vertex, use_spc)


# =============================================================================
# Pytest Fixtures
# =============================================================================
SEEDS = [pytest.param(s) for s in SUBSET_SEED_OPTIONS]
DIRECTED = [pytest.param(d) for d in DIRECTED_GRAPH_OPTIONS]
DATASETS = [pytest.param(d) for d in utils.DATASETS]
DATASETS_SMALL = [pytest.param(d) for d in utils.DATASETS_SMALL]
USE_SHORTEST_PATH_COUNTER = [pytest.param(False), pytest.param(True)]

# Call genFixtureParamsProduct() to caluculate the cartesian product of
# multiple lists of params. This is required since parameterized fixtures do
# not do this automatically (unlike multiply-parameterized tests). The 2nd
# item in the tuple is a label for the param value used when displaying the
# full test name.
algo_test_fixture_params = utils.genFixtureParamsProduct(
    (SEEDS, "seed"),
    (USE_SHORTEST_PATH_COUNTER, "spc"))

graph_fixture_params = utils.genFixtureParamsProduct(
    (DATASETS, "ds"),
    (DIRECTED, "dirctd"))

small_graph_fixture_params = utils.genFixtureParamsProduct(
    (DATASETS_SMALL, "ds"),
    (DIRECTED, "dirctd"))

# The single param list variants are used when only 1 param combination is
# needed (eg. testing non-native input types where tests for other combinations
# was covered elsewhere).
single_algo_test_fixture_params = utils.genFixtureParamsProduct(
    ([SEEDS[0]], "seed"),
    ([USE_SHORTEST_PATH_COUNTER[0]], "spc"))

single_small_graph_fixture_params = utils.genFixtureParamsProduct(
    ([DATASETS_SMALL[0]], "ds"),
    (DIRECTED, "dirctd"))


# Fixtures that result in a test-per (dataset X directed/undirected)
# combination. These return the path to the dataset, a bool indicating if a
# directed graph is being used, and the Nx graph object.
@pytest.fixture(scope="module", params=graph_fixture_params)
def dataset_nx_graph(request):
    return get_nx_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=small_graph_fixture_params)
def small_dataset_nx_graph(request):
    return get_nx_graph_and_params(*request.param)


@pytest.fixture(scope="module", params=single_small_graph_fixture_params)
def single_small_dataset_nx_graph(request):
    return get_nx_graph_and_params(*request.param)


# Fixtures that result in a test-per (dataset_nx_graph combinations X algo_test
# param combinations) combination. These run Nx BFS on the Nx graph obj and
# return the path to the dataset, if a directed graph is being used, the Nx BFS
# results, the starting vertex for BFS, and flag if shortes path counting was
# used.
@pytest.fixture(scope="module", params=algo_test_fixture_params)
def dataset_nxresults_startvertex_spc(dataset_nx_graph, request):
    return get_nx_results_and_params(*request.param, *dataset_nx_graph)


@pytest.fixture(scope="module", params=single_algo_test_fixture_params)
def single_dataset_nxresults_startvertex_spc(single_small_dataset_nx_graph,
                                             request):
    return get_nx_results_and_params(*request.param,
                                     *single_small_dataset_nx_graph)


@pytest.fixture(scope="module")
def dataset_nxresults_allstartvertices_spc(small_dataset_nx_graph):

    dataset, directed, Gnx = small_dataset_nx_graph
    use_spc = True

    start_vertices = [start_vertex for start_vertex in Gnx]

    all_nx_values = []
    for start_vertex in start_vertices:
        _, _, nx_sp_counter = \
            nxacb._single_source_shortest_path_basic(Gnx, start_vertex)
        nx_values = nx_sp_counter
        all_nx_values.append(nx_values)

    return (dataset, directed, all_nx_values, start_vertices, use_spc)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs(gpubenchmark, dataset_nxresults_startvertex_spc,
             cugraph_input_type):
    """
    Test BFS traversal on random source with distance and predecessors
    """
    (dataset, directed, nx_values, start_vertex, use_spc) = \
        dataset_nxresults_startvertex_spc

    # special case: ensure cugraph and Nx Graph types are DiGraphs if
    # "directed" is set, since the graph type parameterization is currently
    # independent of the directed parameter. Unfortunately this does not
    # change the "id" in the pytest output.
    if directed:
        if cugraph_input_type is cugraph.Graph:
            cugraph_input_type = cugraph.DiGraph
        elif cugraph_input_type is nx.Graph:
            cugraph_input_type = nx.DiGraph

    G_or_matrix = utils.create_obj_from_csv(dataset, cugraph_input_type)

    compare_bfs(
        gpubenchmark,
        G_or_matrix, nx_values, start_vertex, return_sp_counter=use_spc
    )


@pytest.mark.parametrize("cugraph_input_type",
                         utils.NX_INPUT_TYPES + utils.MATRIX_INPUT_TYPES)
def test_bfs_nonnative_inputs(gpubenchmark,
                              single_dataset_nxresults_startvertex_spc,
                              cugraph_input_type):
    test_bfs(gpubenchmark,
             single_dataset_nxresults_startvertex_spc,
             cugraph_input_type)


@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_INPUT_TYPES)
def test_bfs_spc_full(gpubenchmark, dataset_nxresults_allstartvertices_spc,
                      cugraph_input_type):
    """
    Test BFS traversal on every vertex with shortest path counting
    """
    (dataset, directed, all_nx_values, start_vertices, use_spc) = \
        dataset_nxresults_allstartvertices_spc

    # use_spc is currently always True

    # special case: ensure cugraph and Nx Graph types are DiGraphs if
    # "directed" is set, since the graph type parameterization is currently
    # independent of the directed parameter. Unfortunately this does not
    # change the "id" in the pytest output.
    if directed:
        if cugraph_input_type is cugraph.Graph:
            cugraph_input_type = cugraph.DiGraph
        elif cugraph_input_type is nx.Graph:
            cugraph_input_type = nx.DiGraph

    G_or_matrix = utils.create_obj_from_csv(dataset, cugraph_input_type)

    compare_bfs(
        gpubenchmark,
        G_or_matrix, all_nx_values, start_vertex=start_vertices,
        return_sp_counter=use_spc
    )


def test_scipy_api_compat():
    graph_file = utils.DATASETS[0]

    input_cugraph_graph = utils.create_obj_from_csv(graph_file, cugraph.Graph,
                                                    edgevals=True)
    input_coo_matrix = utils.create_obj_from_csv(graph_file, cp_coo_matrix,
                                                 edgevals=True)
    # Ensure scipy-only options are rejected for cugraph inputs
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph, start=0, directed=False)
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph)  # required arg missing

    # Ensure cugraph-compatible options work as expected
    cugraph.bfs(input_cugraph_graph, i_start=0)
    cugraph.bfs(input_cugraph_graph, i_start=0, return_sp_counter=True)
    # cannot have start and i_start
    with pytest.raises(TypeError):
        cugraph.bfs(input_cugraph_graph, start=0, i_start=0)

    # Ensure SciPy options for matrix inputs work as expected
    cugraph.bfs(input_coo_matrix, i_start=0)
    cugraph.bfs(input_coo_matrix, i_start=0, directed=True)
    cugraph.bfs(input_coo_matrix, i_start=0, directed=False)
    result = cugraph.bfs(input_coo_matrix, i_start=0,
                         return_sp_counter=True)
    assert type(result) is tuple
    assert len(result) == 3

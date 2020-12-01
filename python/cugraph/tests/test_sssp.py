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
import time

import numpy as np
import pytest
import pandas as pd
import cupy as cp
from cupyx.scipy.sparse.coo import coo_matrix as cp_coo_matrix
from cupyx.scipy.sparse.csr import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse.csc import csc_matrix as cp_csc_matrix
from scipy.sparse.coo import coo_matrix as sp_coo_matrix
from scipy.sparse.csr import csr_matrix as sp_csr_matrix
from scipy.sparse.csc import csc_matrix as sp_csc_matrix

import cudf
import cugraph
from cugraph.tests import utils

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
def cugraph_call(gpu_benchmark_callable, input_G_or_matrix,
                 source, edgevals=False):
    """
    Call cugraph.sssp on input_G_or_matrix, then convert the result to a
    standard format (dictionary of vertex IDs to (distance, predecessor)
    tuples) for easy checking in the test code.
    """
    result = gpu_benchmark_callable(cugraph.sssp, input_G_or_matrix, source)

    input_type = type(input_G_or_matrix)
    expected_return_type = cuGraph_input_output_map[type(input_G_or_matrix)]
    assert type(result) is expected_return_type

    # Convert cudf and pandas: DF of 3 columns: (vertex, distance, predecessor)
    if expected_return_type in [cudf.DataFrame, pd.DataFrame]:
        if expected_return_type is pd.DataFrame:
            result = cudf.from_pandas(result)

        if np.issubdtype(result["distance"].dtype, np.integer):
            max_val = np.iinfo(result["distance"].dtype).max
        else:
            max_val = np.finfo(result["distance"].dtype).max
        verts = result["vertex"].to_array()
        dists = result["distance"].to_array()
        preds = result["predecessor"].to_array()

    # A CuPy/SciPy input means the return value will be a 2-tuple of:
    #   distance: cupy.ndarray
    #      ndarray of shortest distances between source and vertex.
    #   predecessor: cupy.ndarray
    #      ndarray of predecessors of a vertex on the path from source, which
    #      can be used to reconstruct the shortest paths.
    elif expected_return_type is tuple:
        if input_type in cupy_types:
            assert type(result[0]) is cp.ndarray
            assert type(result[1]) is cp.ndarray
        else:
            assert type(result[0]) is np.ndarray
            assert type(result[1]) is np.ndarray

        if np.issubdtype(result[0].dtype, np.integer):
            max_val = np.iinfo(result[0].dtype).max
        else:
            max_val = np.finfo(result[0].dtype).max

        # Get unique verts from input since they are not incuded in output
        if type(input_G_or_matrix) in [cp_csr_matrix, cp_csc_matrix,
                                       sp_csr_matrix, sp_csc_matrix]:
            coo = input_G_or_matrix.tocoo(copy=False)
        else:
            coo = input_G_or_matrix
        verts = sorted(set([n.item() for n in coo.col] +
                           [n.item() for n in coo.row]))
        dists = [n.item() for n in result[0]]
        preds = [n.item() for n in result[1]]
        assert len(verts) == len(dists) == len(preds)

    else:
        raise RuntimeError(f"unsupported return type: {expected_return_type}")

    result_dict = dict(zip(verts, zip(dists, preds)))
    return result_dict, max_val


def networkx_call(graph_file, source, edgevals=False):
    M = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    # Directed NetworkX graph
    edge_attr = "weight" if edgevals else None
    Gnx = nx.from_pandas_edgelist(
        M,
        source="0",
        target="1",
        edge_attr=edge_attr,
        create_using=nx.DiGraph(),
    )
    print("NX Solving... ")
    t1 = time.time()

    if edgevals is False:
        nx_paths = nx.single_source_shortest_path_length(Gnx, source)
    else:
        nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)

    t2 = time.time() - t1
    print("NX Time : " + str(t2))

    return (graph_file, source, nx_paths, Gnx)


# =============================================================================
# Pytest fixtures
# =============================================================================

# Call genFixtureParamsProduct() to caluculate the cartesian product of
# multiple lists of params. This is required since parameterized fixtures do
# not do this automatically (unlike multiply-parameterized tests). The 2nd
# item in the tuple is a label for the param value used when displaying the
# full test name.
DATASETS = [pytest.param(d) for d in utils.DATASETS]
SOURCES = [pytest.param(1)]
fixture_params = utils.genFixtureParamsProduct((DATASETS, "ds"),
                                               (SOURCES, "src"))
fixture_params_single_dataset = \
    utils.genFixtureParamsProduct(([DATASETS[0]], "ds"), (SOURCES, "src"))


# These fixtures will call networkx BFS algos and save the result. The networkx
# call is only made only once per input param combination.
@pytest.fixture(scope="module", params=fixture_params)
def dataset_source_nxresults(request):
    # request.param is a tuple of params from fixture_params. When expanded
    # with *, will be passed to networkx_call() as args (graph_file, source)
    return networkx_call(*(request.param))


@pytest.fixture(scope="module", params=fixture_params_single_dataset)
def single_dataset_source_nxresults(request):
    return networkx_call(*(request.param))


@pytest.fixture(scope="module", params=fixture_params)
def dataset_source_nxresults_weighted(request):
    return networkx_call(*(request.param), edgevals=True)


@pytest.fixture(scope="module", params=fixture_params_single_dataset)
def single_dataset_source_nxresults_weighted(request):
    return networkx_call(*(request.param), edgevals=True)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_DIR_INPUT_TYPES)
def test_sssp(gpubenchmark, dataset_source_nxresults, cugraph_input_type):
    # Extract the params generated from the fixture
    (graph_file, source, nx_paths, Gnx) = dataset_source_nxresults

    input_G_or_matrix = utils.create_obj_from_csv(graph_file,
                                                  cugraph_input_type)
    cu_paths, max_val = cugraph_call(gpubenchmark, input_G_or_matrix, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + 1 = current dist (since unweighted)
            pred = cu_paths[vid][1]
            if vid != source and cu_paths[pred][0] + 1 != cu_paths[vid][0]:
                err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.parametrize("cugraph_input_type",
                         utils.NX_DIR_INPUT_TYPES + utils.MATRIX_INPUT_TYPES)
def test_sssp_nonnative_inputs(gpubenchmark,
                               single_dataset_source_nxresults,
                               cugraph_input_type):
    test_sssp(gpubenchmark,
              single_dataset_source_nxresults,
              cugraph_input_type)


@pytest.mark.parametrize("cugraph_input_type", utils.CUGRAPH_DIR_INPUT_TYPES)
def test_sssp_edgevals(gpubenchmark, dataset_source_nxresults_weighted,
                       cugraph_input_type):
    # Extract the params generated from the fixture
    (graph_file, source, nx_paths, Gnx) = dataset_source_nxresults_weighted

    input_G_or_matrix = utils.create_obj_from_csv(graph_file,
                                                  cugraph_input_type,
                                                  edgevals=True)
    cu_paths, max_val = cugraph_call(gpubenchmark, input_G_or_matrix,
                                     source, edgevals=True)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]["weight"]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.parametrize("cugraph_input_type",
                         utils.NX_DIR_INPUT_TYPES + utils.MATRIX_INPUT_TYPES)
def test_sssp_edgevals_nonnative_inputs(
        gpubenchmark,
        single_dataset_source_nxresults_weighted,
        cugraph_input_type):
    test_sssp_edgevals(gpubenchmark,
                       single_dataset_source_nxresults_weighted,
                       cugraph_input_type)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
@pytest.mark.parametrize("source", SOURCES)
def test_sssp_data_type_conversion(graph_file, source):
    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    # cugraph call with int32 weights
    cu_M["2"] = cu_M["2"].astype(np.int32)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    # assert cugraph weights is int32
    assert G.edgelist.edgelist_df["weights"].dtype == np.int32
    df = cugraph.sssp(G, source)
    max_val = np.finfo(df["distance"].dtype).max
    verts_np = df["vertex"].to_array()
    dist_np = df["distance"].to_array()
    pred_np = df["predecessor"].to_array()
    cu_paths = dict(zip(verts_np, zip(dist_np, pred_np)))

    # networkx call with int32 weights
    M["weight"] = M["weight"].astype(np.int32)
    Gnx = nx.from_pandas_edgelist(
        M,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    # assert nx weights is int
    assert type(list(Gnx.edges(data=True))[0][2]["weight"]) is int
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]["weight"]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0


def test_scipy_api_compat():
    graph_file = utils.DATASETS[0]

    input_cugraph_graph = utils.create_obj_from_csv(graph_file, cugraph.Graph,
                                                    edgevals=True)
    input_coo_matrix = utils.create_obj_from_csv(graph_file, cp_coo_matrix,
                                                 edgevals=True)

    # Ensure scipy-only options are rejected for cugraph inputs
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph, source=0,
                              directed=False)
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph, source=0,
                              unweighted=False)
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph, source=0,
                              overwrite=False)
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph, source=0,
                              return_predecessors=False)

    # Ensure cugraph-compatible options work as expected
    # cannot set both source and indices, but must set one
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph, source=0, indices=0)
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_cugraph_graph)
    with pytest.raises(ValueError):
        cugraph.shortest_path(input_cugraph_graph, source=0,
                              method="BF")
    cugraph.shortest_path(input_cugraph_graph, indices=0)
    with pytest.raises(ValueError):
        cugraph.shortest_path(input_cugraph_graph, indices=[0, 1, 2])
    cugraph.shortest_path(input_cugraph_graph, source=0, method="auto")

    # Ensure SciPy options for matrix inputs work as expected
    # cannot set both source and indices, but must set one
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_coo_matrix, source=0, indices=0)
    with pytest.raises(TypeError):
        cugraph.shortest_path(input_coo_matrix)
    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, source=0, method="BF")
    cugraph.shortest_path(input_coo_matrix, source=0, method="auto")

    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, source=0, directed=3)
    cugraph.shortest_path(input_coo_matrix, source=0, directed=True)
    cugraph.shortest_path(input_coo_matrix, source=0, directed=False)

    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, source=0,
                              return_predecessors=3)
    (distances, preds) = cugraph.shortest_path(input_coo_matrix,
                                               source=0,
                                               return_predecessors=True)
    distances = cugraph.shortest_path(input_coo_matrix,
                                      source=0,
                                      return_predecessors=False)
    assert type(distances) != tuple

    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, source=0,
                              unweighted=False)
    cugraph.shortest_path(input_coo_matrix, source=0,
                          unweighted=True)

    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, source=0,
                              overwrite=True)
    cugraph.shortest_path(input_coo_matrix, source=0,
                          overwrite=False)

    with pytest.raises(ValueError):
        cugraph.shortest_path(input_coo_matrix, indices=[0, 1, 2])
    cugraph.shortest_path(input_coo_matrix, indices=0)

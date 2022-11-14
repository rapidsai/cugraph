# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import cupy as cp
from scipy.sparse import coo_matrix, csr_matrix
from pylibcugraph import ResourceHandle, GraphProperties, SGGraph

from pylibcugraph.testing import utils


# =============================================================================
# Test data
# =============================================================================
_test_data = {
    "graph1": {  # asymmetric
        "input": [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        "scc_comp_vertices": [
            [0],
            [1],
            [2],
            [3],
            [4],
        ],
        "wcc_comp_vertices": [
            [0, 1, 2],
            [3, 4],
        ],
    },
    "graph2": {  # symmetric
        "input": [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        "scc_comp_vertices": [
            [0, 1, 2],
            [3, 4],
        ],
        "wcc_comp_vertices": [
            [0, 1, 2],
            [3, 4],
        ],
    },
    "karate-disjoint-sequential": {
        "input": utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-disjoint-sequential.csv",
        "scc_comp_vertices": [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
            ],
            [34],
            [35],
            [36],
        ],
        "wcc_comp_vertices": [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
            ],
            [34, 35, 36],
        ],
    },
    "dolphins": {  # dolphins contains only one component
        "input": utils.RAPIDS_DATASET_ROOT_DIR_PATH / "dolphins.csv",
        "scc_comp_vertices": [
            list(range(62)),
        ],
        "wcc_comp_vertices": [
            list(range(62)),
        ],
    },
}


# =============================================================================
# Pytest fixtures
# =============================================================================
@pytest.fixture(
    scope="module",
    params=[pytest.param(value, id=key) for (key, value) in _test_data.items()],
)
def input_and_expected_output(request):
    """
    This fixture takes the above test data and converts it into everything
    needed to run a pylibcugraph CC algo for a specific input (either a
    adjacency matrix or a CSV edgelist file), and returns it along with the
    expected WCC and SCC result for each.
    """
    d = request.param.copy()
    input = d.pop("input")
    expected_output_dict = d

    if isinstance(input, Path):
        pdf = pd.read_csv(
            input,
            delimiter=" ",
            header=None,
            names=["0", "1", "weight"],
            dtype={"0": "int32", "1": "int32", "weight": "float32"},
        )
        num_verts = len(set(pdf["0"].tolist() + pdf["1"].tolist()))
        num_edges = len(pdf)
        weights = np.ones(num_edges)
        coo = coo_matrix(
            (weights, (pdf["0"], pdf["1"])),
            shape=(num_verts, num_verts),
            dtype=np.float32,
        )
        csr = coo.tocsr()
    else:
        csr = csr_matrix(input)
        num_verts = csr.get_shape()[0]
        num_edges = csr.nnz

    labels_to_populate = cp.zeros(num_verts, dtype=np.int32)

    return (
        (csr, labels_to_populate, num_verts, num_edges),
        expected_output_dict,
    )


# =============================================================================
# Helper functions
# =============================================================================
def _check_labels(vertex_ordered_labels, expected_vertex_comps):
    """
    vertex_ordered_labels is a list of labels, ordered by the position of the
    vertex ID value, as returned by pylibcugraph.CC algos. For example:
    [9, 9, 7]
    means vertex 0 is labelled 9, vertex 1 is labelled 9, and vertex 2 is
    labelled 7.

    expected_vertex_comps is a list of components, where each component is a
    list of vertex IDs the component contains. Each component corresponds to
    some label For example:
    [[0, 1], [2]]
    is two components, the first
    containing vertices 0, 1, and the other 2.  [0, 1] has the label 9 and [2]
    has the label 7.

    This asserts if the vertex_ordered_labels do not correspond to the
    expected_vertex_comps.
    """
    # Group the vertex_ordered_labels list into components based on labels by
    # creating a dictionary of labels to lists of vertices with that label.
    d = {}
    for (vertex, label) in enumerate(vertex_ordered_labels):
        d.setdefault(label, []).append(vertex)

    assert len(d.keys()) == len(
        expected_vertex_comps
    ), "number of different labels does not match expected"

    # Compare the actual components (created from the dictionary above) to
    # expected.
    actual_vertex_comps = sorted(d.values())
    assert actual_vertex_comps == sorted(expected_vertex_comps)


# =============================================================================
# Tests
# =============================================================================
def test_import():
    """
    Ensure pylibcugraph is importable.
    """
    # suppress F401 (imported but never used) in flake8
    import pylibcugraph  # noqa: F401


def test_scc(input_and_expected_output):
    """
    Tests strongly_connected_components()
    """
    import pylibcugraph

    (
        (csr, cupy_labels_to_populate, num_verts, num_edges),
        expected_output_dict,
    ) = input_and_expected_output

    cupy_offsets = cp.asarray(csr.indptr, dtype=np.int32)
    cupy_indices = cp.asarray(csr.indices, dtype=np.int32)

    pylibcugraph.strongly_connected_components(
        cupy_offsets, cupy_indices, None, num_verts, num_edges, cupy_labels_to_populate
    )

    _check_labels(
        cupy_labels_to_populate.tolist(), expected_output_dict["scc_comp_vertices"]
    )


def test_wcc(input_and_expected_output):
    """
    Tests weakly_connected_components()
    """
    import pylibcugraph

    (
        (csr, cupy_labels_to_populate, num_verts, num_edges),
        expected_output_dict,
    ) = input_and_expected_output

    # Symmetrize CSR matrix. WCC requires a symmetrized datasets
    rows, cols = csr.nonzero()
    csr[cols, rows] = csr[rows, cols]

    cupy_offsets = cp.asarray(csr.indptr, dtype=np.int32)
    cupy_indices = cp.asarray(csr.indices, dtype=np.int32)
    cupy_weights = cp.asarray(csr.data, dtype=np.float32)

    pylibcugraph.weakly_connected_components(
        None,
        None,
        cupy_offsets,
        cupy_indices,
        cupy_weights,
        cupy_labels_to_populate,
        False,
    )

    _check_labels(
        cupy_labels_to_populate.tolist(), expected_output_dict["wcc_comp_vertices"]
    )


# FIXME: scc and wcc no longer have the same API (parameters in the
# function definition): refactor this to consolidate both tests once the do
@pytest.mark.parametrize("api_name", ["strongly_connected_components"])
def test_non_CAI_input_scc(api_name):
    """
    Ensures that the *_connected_components() APIs only accepts instances of
    objects that have a __cuda_array_interface__
    """
    import pylibcugraph

    cupy_array = cp.ndarray(range(8))
    python_list = list(range(8))
    api = getattr(pylibcugraph, api_name)

    with pytest.raises(TypeError):
        api(
            src=cupy_array,
            dst=cupy_array,
            weights=cupy_array,  # should raise, weights must be None
            num_verts=2,
            num_edges=8,
            labels=cupy_array,
        )
    with pytest.raises(TypeError):
        api(
            src=cupy_array,
            dst=python_list,  # should raise, no __cuda_array_interface__
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=cupy_array,
        )
    with pytest.raises(TypeError):
        api(
            src=python_list,  # should raise, no __cuda_array_interface__
            dst=cupy_array,
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=cupy_array,
        )
    with pytest.raises(TypeError):
        api(
            src=cupy_array,
            dst=cupy_array,
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=python_list,
        )  # should raise, no __cuda_array_interface__


# FIXME: scc and wcc no longer have the same API:
# refactor this to consolidate both tests once they do
@pytest.mark.parametrize("api_name", ["strongly_connected_components"])
def test_bad_dtypes_scc(api_name):
    """
    Ensures that only supported dtypes are accepted.
    """
    import pylibcugraph

    graph = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    scipy_csr = csr_matrix(graph)
    num_verts = scipy_csr.get_shape()[0]
    num_edges = scipy_csr.nnz

    api = getattr(pylibcugraph, api_name)

    cp_offsets = cp.asarray(scipy_csr.indptr)
    cp_indices = cp.asarray(scipy_csr.indices)
    cp_labels = cp.zeros(num_verts, dtype=np.int64)  # unsupported
    with pytest.raises(TypeError):
        api(
            offsets=cp_offsets,
            indices=cp_indices,
            weights=None,
            num_verts=num_verts,
            num_edges=num_edges,
            labels=cp_labels,
        )

    cp_offsets = cp.asarray(scipy_csr.indptr, dtype=np.int64)  # unsupported
    cp_indices = cp.asarray(scipy_csr.indices)
    cp_labels = cp.zeros(num_verts, dtype=np.int32)
    with pytest.raises(TypeError):
        api(
            offsets=cp_offsets,
            indices=cp_indices,
            weights=None,
            num_verts=num_verts,
            num_edges=num_edges,
            labels=cp_labels,
        )

    cp_offsets = cp.asarray(scipy_csr.indptr)
    cp_indices = cp.asarray(scipy_csr.indices, dtype=np.float32)  # unsupported
    cp_labels = cp.zeros(num_verts, dtype=np.int32)
    with pytest.raises(TypeError):
        api(
            offsets=cp_offsets,
            indices=cp_indices,
            weights=None,
            num_verts=num_verts,
            num_edges=num_edges,
            labels=cp_labels,
        )


def test_invalid_input_wcc():
    """
    Ensures that only supported dtypes are accepted.
    """
    import pylibcugraph

    graph = [
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]
    scipy_csr = csr_matrix(graph)

    sp_offsets = scipy_csr.indptr  # unsupported
    sp_indices = scipy_csr.indices  # unsupported

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)
    with pytest.raises(TypeError):
        pylibcugraph.weakly_connected_components(
            resource_handle, None, sp_offsets, sp_indices, None, False
        )

    resource_handle = ResourceHandle()
    cp_offsets = cp.asarray(scipy_csr.indptr)  # unsupported
    cp_indices = cp.asarray(scipy_csr.indices)  # unsupported

    G = SGGraph(
        resource_handle,
        graph_props,
        cp_offsets,
        cp_indices,
        None,
        store_transposed=False,
        renumber=False,
        do_expensive_check=True,
        input_array_format="CSR",
    )

    # cannot set both a graph and csr arrays as input
    with pytest.raises(TypeError):
        pylibcugraph.weakly_connected_components(
            resource_handle, G, cp_offsets, cp_indices, None, True
        )

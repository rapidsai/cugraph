# Copyright (c) 2021, NVIDIA CORPORATION.
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

import pytest
import numpy as np

import cupy
import cugraph

from . import utils


@pytest.fixture
def package_under_test():
    """
    Create a fixture to import the package under test.  This is useful since
    bugs that prevent the package under test from being imported will not
    prevent pytest from collecting, listing, running, etc. the tests.
    """
    import pylibcugraph
    return pylibcugraph


@pytest.fixture(scope="module",
                params=[pytest.param(csv, id=csv.name)
                        for csv in utils.DATASETS])
def DiGraph_input(request):
    """
    Parameterized with a list of tuples containing the .csv file, and returns a
    DiGraph constructed from the .csv dataset

    This fixture has a module scope, so each dataset and corresponding DiGraph
    is only read and constructed once.
    """
    dataset_file = request.param
    cu_M = utils.read_csv_file(dataset_file)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    return G


###############################################################################
# Tests
def test_import():
    """
    Ensure pylibcugraph is importable.
    """
    # suppress F401 (imported but never used) in flake8
    import pylibcugraph  # noqa: F401


def test_scc(package_under_test, DiGraph_input):
    """
    Tests strongly_connected_components()
    """
    pylibcugraph = package_under_test
    G = DiGraph_input

    offsets, indices, weights = G.view_adj_list()
    cupy_off = cupy.array(offsets)
    cudf_ind = indices

    cupy_labels = cupy.array(np.zeros(G.number_of_vertices()))
    pylibcugraph.strongly_connected_components(
        cupy_off,
        cudf_ind,
        None,
        G.number_of_vertices(),
        G.number_of_edges(directed_edges=True),
        cupy_labels
    )

    # Compare to cugraph
    # FIXME: this comparison will not be helpful when cuGraph eventually calls
    # pylibcugraph. This needs to compare to known good results.
    df = cugraph.strongly_connected_components(G)
    assert (df["labels"] == cupy_labels.tolist()).all()


def test_wcc(package_under_test, DiGraph_input):
    """
    Tests weakly_connected_components()
    """
    pylibcugraph = package_under_test
    G = DiGraph_input

    cupy_src = cupy.array(G.edgelist.edgelist_df["src"])
    cudf_dst = G.edgelist.edgelist_df["dst"]
    cupy_labels = cupy.array(np.zeros(G.number_of_vertices()), dtype='int32')

    pylibcugraph.weakly_connected_components(
        cupy_src,
        cudf_dst,
        None,
        G.number_of_vertices(),
        G.number_of_edges(directed_edges=True),
        cupy_labels
    )

    # Compare to cugraph
    # FIXME: this comparison will not be helpful when cuGraph eventually calls
    # pylibcugraph. This needs to compare to known good results.
    df = cugraph.weakly_connected_components(G)
    assert (df["labels"] == cupy_labels.tolist()).all()


@pytest.mark.parametrize("api_name", ["strongly_connected_components",
                                      "weakly_connected_components"])
def test_invalid_input(package_under_test, api_name):
    """
    Ensures that the *_connected_components() APIs only accepts instances of
    the correct type.
    """
    pylibcugraph = package_under_test
    cupy_array = cupy.ndarray(range(8))
    python_list = list(range(8))
    api = getattr(pylibcugraph, api_name)

    with pytest.raises(TypeError):
        api(src=cupy_array,
            dst=cupy_array,
            weights=cupy_array,  # should raise, weights must be None
            num_verts=2,
            num_edges=8,
            labels=cupy_array)
    with pytest.raises(TypeError):
        api(src=cupy_array,
            dst=python_list,  # should raise, no __cuda_array_interface__
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=cupy_array)
    with pytest.raises(TypeError):
        api(src=python_list,  # should raise, no __cuda_array_interface__
            dst=cupy_array,
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=cupy_array)
    with pytest.raises(TypeError):
        api(src=cupy_array,
            dst=cupy_array,
            weights=None,
            num_verts=2,
            num_edges=8,
            labels=python_list)  # should raise, no __cuda_array_interface__

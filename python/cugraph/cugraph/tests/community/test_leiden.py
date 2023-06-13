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
import time

import pytest

import networkx as nx
import cugraph
import cudf
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, karate_asymmetric

from cudf.testing.testing import assert_series_equal


# =============================================================================
# Test data
# =============================================================================

_test_data = {
    "data_1": {
        "graph": {
            "src_or_offset_array": [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
            "dst_or_index_array": [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
            # fmt: off
            "weight": [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1,
                       3.1, 4.1, 7.2, 3.2],
            # fmt: on
        },
        "max_level": 10,
        "resolution": 1.0,
        "input_type": "COO",
        "expected_output": {
            "partition": [1, 0, 1, 2, 2, 2],
            "modularity_score": 0.1757322,
        },
    },
    "data_2": {
        "graph": {
            # fmt: off
            "src_or_offset_array": [0, 16, 25, 35, 41, 44, 48, 52, 56, 61, 63, 66,
                                    67, 69, 74, 76, 78, 80, 82, 84, 87, 89, 91, 93,
                                    98, 101, 104, 106, 110, 113, 117, 121, 127, 139,
                                    156],

            "dst_or_index_array": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21,
                                   31, 0, 2, 3, 7, 13, 17, 19, 21, 30, 0, 1, 3, 7, 8,
                                   9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0, 6, 10, 0,
                                   6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32,
                                   33, 2, 33, 0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33,
                                   32, 33, 5, 6, 0, 1, 32, 33, 0, 1, 33, 32, 33, 0, 1,
                                   32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31,
                                   29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1,
                                   8, 32, 33, 0, 24, 25, 28, 32, 33, 2, 8, 14, 15, 18,
                                   20, 22, 23, 29, 30, 31, 33, 8, 9, 13, 14, 15, 18, 19,
                                   20, 22, 23, 26, 27, 28, 29, 30, 31, 32],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # fmt: on
        },
        "max_level": 40,
        "resolution": 1.0,
        "input_type": "CSR",
        "expected_output": {
            # fmt: off
            "partition": [6, 6, 3, 3, 1, 5, 5, 3, 0, 3, 1, 6, 3, 3, 4, 4, 5, 6, 4, 6, 4,
                          6, 4, 4, 2, 2, 4, 4, 2, 4, 0, 2, 4, 4],
            # fmt: on
            "modularity_score": 0.3468113,
        },
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
    d = request.param.copy()

    input_graph_data = d.pop("graph")
    input_type = d.pop("input_type")
    src_or_offset_array = cudf.Series(
        input_graph_data["src_or_offset_array"], dtype="int32"
    )
    dst_or_index_array = cudf.Series(
        input_graph_data["dst_or_index_array"], dtype="int32"
    )
    weight = cudf.Series(input_graph_data["weight"], dtype="float32")

    max_level = d.pop("max_level")
    resolution = d.pop("resolution")
    output = d

    G = cugraph.Graph()

    if input_type == "COO":
        # Create graph from an edgelist
        df = cudf.DataFrame()
        df["src"] = src_or_offset_array
        df["dst"] = dst_or_index_array
        df["weight"] = cudf.Series(weight, dtype="float32")
        G.from_cudf_edgelist(
            df,
            source="src",
            destination="dst",
            edge_attr="weight",
            store_transposed=False,
        )

    elif input_type == "CSR":
        # Create graph from csr
        offsets = src_or_offset_array
        indices = dst_or_index_array
        G.from_cudf_adjlist(offsets, indices, weight)

    parts, mod = cugraph.leiden(G, max_level, resolution)

    parts = parts.sort_values("vertex").reset_index(drop=True)

    output["result_output"] = {"partition": parts["partition"], "modularity_score": mod}

    return output


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_leiden(G):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.leiden(G)
    t2 = time.time() - t1
    print("Cugraph Leiden Time : " + str(t2))

    return parts, mod


def cugraph_louvain(G):

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Louvain Time : " + str(t2))

    return parts, mod


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_leiden(graph_file):
    edgevals = True

    G = graph_file.get_graph(ignore_weights=not edgevals)
    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Leiden modularity score is smaller than Louvain's
    assert leiden_mod >= (0.75 * louvain_mod)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_leiden_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)

    G = nx.from_pandas_edgelist(
        NM, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )

    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Calculating modularity scores for comparison
    # Leiden modularity score is smaller than Louvain's
    assert leiden_mod >= (0.75 * louvain_mod)


@pytest.mark.sg
def test_leiden_directed_graph():

    edgevals = True
    G = karate_asymmetric.get_graph(
        create_using=cugraph.Graph(directed=True), ignore_weights=not edgevals
    )

    with pytest.raises(ValueError):
        parts, mod = cugraph_leiden(G)


@pytest.mark.sg
def test_leiden_golden_results(input_and_expected_output):
    expected_partition = cudf.Series(
        input_and_expected_output["expected_output"]["partition"]
    )
    expected_mod = input_and_expected_output["expected_output"]["modularity_score"]

    result_partition = input_and_expected_output["result_output"]["partition"]
    result_mod = input_and_expected_output["result_output"]["modularity_score"]

    assert abs(expected_mod - result_mod) < 0.0001

    assert_series_equal(
        expected_partition, result_partition, check_dtype=False, check_names=False
    )

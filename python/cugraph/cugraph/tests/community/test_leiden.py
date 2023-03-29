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
from cugraph.experimental.datasets import DATASETS, karate_asymmetric

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import community and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# =============================================================================
# Test data
# =============================================================================

_test_data = {
    "data_1":{
        "graph": {
            "src_or_offset_array": [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
            "dst_or_index_array": [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
            "weight": [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2]
        },
        "max_level": 10,
        "resolution": 1.0,
        "input_type": "COO",
        "expected_output": {
            "partition": [0, 1, 0, 1, 1, 1],
            "modularity_score": 0.218166
        }
    }
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
    src_or_offset_array = input_graph_data["src_or_offset_array"]
    dst_or_index_array = input_graph_data["dst_or_index_array"]
    weight = input_graph_data["weight"]
    max_level = d.pop("max_level")
    resolution = d.pop("resolution")
    output = d


    G = cugraph.Graph()

    # Done in the test
    if input_type == "COO":
        # Create graph from an edgelist
        df = cudf.DataFrame()
        df["src"] = cudf.Series(src_or_offset_array, dtype="int32")
        df["dst"] = cudf.Series(dst_or_index_array, dtype="int32")
        df["weight"] = cudf.Series(weight, dtype="float32")
        G.from_cudf_edgelist(
            df, source="src", destination="dst", edge_attr="weight", store_transposed=True)
    
    elif input_type == "CSR":
        # Create graph from csr
        offsets = src_or_offset_array
        indices = dst_or_index_array 
        G.from_cudf_adjlist(offsets, indices, weight)
    
    parts, mod = cugraph.leiden(G, max_level, resolution)

    parts = parts.sort_values("vertex").reset_index(drop=True)



    output["result_output"] = {
        "partition": parts["partition"],
        "modularity_score": mod}
    
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
@pytest.mark.parametrize("graph_file", DATASETS)
def test_leiden(graph_file):
    edgevals = True

    G = graph_file.get_graph(ignore_weights=not edgevals)
    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DATASETS)
def test_leiden_nx(graph_file):
    dataset_path = graph_file.get_path()
    NM = utils.read_csv_for_nx(dataset_path)

    G = nx.from_pandas_edgelist(
        NM, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )

    leiden_parts, leiden_mod = cugraph_leiden(G)
    louvain_parts, louvain_mod = cugraph_louvain(G)

    # Calculating modularity scores for comparison
    assert leiden_mod >= (0.99 * louvain_mod)


@pytest.mark.sg
def test_leiden_directed_graph():

    edgevals = True
    G = karate_asymmetric.get_graph(
        create_using=cugraph.Graph(directed=True), ignore_weights=not edgevals
    )

    with pytest.raises(ValueError):
        parts, mod = cugraph_leiden(G)


@pytest.mark.sg
@pytest.mark.skip("Debugging")
def test_leiden_golden_results(input_and_expected_output):
    expected_partition = cudf.Series(
        input_and_expected_output["expected_output"]["partition"])
    expected_mod = input_and_expected_output["expected_output"]["modularity_score"]

    result_partition = input_and_expected_output["result_output"]["partition"]
    result_mod = input_and_expected_output["result_output"]["modularity_score"]

    assert expected_mod == result_mod

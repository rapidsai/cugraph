# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import cupy
import cudf
import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate, dolphins


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================

DATASETS = [karate, dolphins]
IS_DIRECTED = [True, False]
IS_NORMALIZED = [True, False]
ENDPOINTS = [True, False]
SUBSET_SEEDS = [42, None]
SUBSET_SIZES = [None, 15]
VERTEX_LIST_TYPES = [list, cudf]

# =============================================================================
# Helper functions
# =============================================================================


def get_sg_graph(dataset, directed):
    dataset.unload()
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed))

    return G


def get_mg_graph(dataset, directed):
    dataset.unload()
    ddf = dataset.get_dask_edgelist()
    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="wgt",
        renumber=True,
        store_transposed=True,
    )

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("normalized", IS_NORMALIZED)
@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("subset_seed", SUBSET_SEEDS)
@pytest.mark.parametrize("subset_size", SUBSET_SIZES)
@pytest.mark.parametrize("v_list_type", VERTEX_LIST_TYPES)
def test_dask_mg_betweenness_centrality(
    dataset,
    directed,
    normalized,
    endpoint,
    subset_seed,
    subset_size,
    v_list_type,
    dask_client,
    benchmark,
):
    g = get_sg_graph(dataset, directed)
    dataset.unload()
    dg = get_mg_graph(dataset, directed)
    random_state = subset_seed

    if subset_size is None:
        k = subset_size
    elif isinstance(subset_size, int):
        # Select random vertices
        k = g.select_random_vertices(
            random_state=random_state, num_vertices=subset_size
        )
        if v_list_type is list:
            k = k.to_arrow().to_pylist()

        print("the seeds are \n", k)
        if v_list_type is int:
            # This internally sample k vertices in betweenness centrality.
            # Since the nodes that will be sampled by each implementation will
            # be random, therefore sample all vertices which will make the test
            # consistent.
            k = len(g.nodes())

    sg_cugraph_bc = cugraph.betweenness_centrality(
        g, k=k, normalized=normalized, endpoints=endpoint, random_state=random_state
    )
    sg_cugraph_bc = sg_cugraph_bc.sort_values("vertex").reset_index(drop=True)

    mg_bc_results = benchmark(
        dcg.betweenness_centrality,
        dg,
        k=k,
        normalized=normalized,
        endpoints=endpoint,
        random_state=random_state,
    )

    mg_bc_results = (
        mg_bc_results.compute().sort_values("vertex").reset_index(drop=True)
    )["betweenness_centrality"].to_cupy()

    sg_bc_results = (sg_cugraph_bc.sort_values("vertex").reset_index(drop=True))[
        "betweenness_centrality"
    ].to_cupy()

    diff = cupy.isclose(mg_bc_results, sg_bc_results)

    assert diff.all()

    # Clean-up stored dataset edge-lists
    dataset.unload()

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

import random
import pytest

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
START_LIST = [True, False]


# =============================================================================
# Helper Functions
# =============================================================================


def get_sg_graph(dataset, directed, start):
    dataset.unload()
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed))
    if start:
        # sample k nodes from the cuGraph graph
        start = G.select_random_vertices(num_vertices=random.randint(1, 10))
    else:
        start = None

    return G, start


def get_mg_graph(dataset, directed):
    dataset.unload()
    ddf = dataset.get_dask_edgelist()
    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf, source="src", destination="dst", edge_attr="wgt", renumber=True
    )

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("start", START_LIST)
def test_sg_triangles(dask_client, dataset, start, benchmark):
    # This test is only for benchmark purposes.
    sg_triangle_results = None
    G, start = get_sg_graph(dataset, False, start)

    sg_triangle_results = benchmark(cugraph.triangle_count, G, start)
    sg_triangle_results.sort_values("vertex").reset_index(drop=True)
    assert sg_triangle_results is not None
    # Clean-up stored dataset edge-lists
    dataset.unload()


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("start", START_LIST)
def test_triangles(dask_client, dataset, start, benchmark):
    G, start = get_sg_graph(dataset, False, start)
    dg = get_mg_graph(dataset, False)

    result_counts = benchmark(dcg.triangle_count, dg, start)
    result_counts = (
        result_counts.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"counts": "mg_counts"})
    )
    expected_output = (
        cugraph.triangle_count(G, start).sort_values("vertex").reset_index(drop=True)
    )

    # Update the mg triangle count with sg triangle count results
    # for easy comparison using cuDF DataFrame methods.
    result_counts["sg_counts"] = expected_output["counts"]
    counts_diffs = result_counts.query("mg_counts != sg_counts")

    assert len(counts_diffs) == 0
    # Clean-up stored dataset edge-lists
    dataset.unload()

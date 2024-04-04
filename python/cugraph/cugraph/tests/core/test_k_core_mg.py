# Copyright (c) 2024, NVIDIA CORPORATION.
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

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate, dolphins
from cudf.testing.testing import assert_frame_equal
from cugraph.structure.symmetrize import symmetrize_df


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
CORE_NUMBER = [True, False]
DEGREE_TYPE = ["bidirectional", "outgoing", "incoming"]


# =============================================================================
# Helper Functions
# =============================================================================


def get_sg_results(dataset, core_number, degree_type):
    dataset.unload()
    G = dataset.get_graph(create_using=cugraph.Graph(directed=False))

    if core_number:
        # compute the core_number
        core_number = cugraph.core_number(G, degree_type=degree_type)
    else:
        core_number = None

    sg_k_core_graph = cugraph.k_core(
        G, core_number=core_number, degree_type=degree_type
    )
    res = sg_k_core_graph.view_edge_list()
    # FIXME: The result will come asymetric. Symmetrize the results
    srcCol = sg_k_core_graph.source_columns
    dstCol = sg_k_core_graph.destination_columns
    wgtCol = sg_k_core_graph.weight_column
    res = (
        symmetrize_df(res, srcCol, dstCol, wgtCol)
        .sort_values([srcCol, dstCol])
        .reset_index(drop=True)
    )
    return res, core_number


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("core_number", CORE_NUMBER)
@pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_sg_k_core(dask_client, dataset, core_number, degree_type, benchmark):
    # This test is only for benchmark purposes.
    sg_k_core = None
    dataset.unload()
    G = dataset.get_graph(create_using=cugraph.Graph(directed=False))
    if core_number:
        # compute the core_number
        core_number = cugraph.core_number(G, degree_type=degree_type)
    else:
        core_number = None
    sg_k_core = benchmark(
        cugraph.k_core, G, core_number=core_number, degree_type=degree_type
    )
    assert sg_k_core is not None
    dataset.unload()


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("core_number", CORE_NUMBER)
@pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_dask_mg_k_core(dask_client, dataset, core_number, degree_type, benchmark):
    expected_k_core_results, core_number = get_sg_results(
        dataset, core_number, degree_type
    )

    dataset.unload()
    dg = dataset.get_dask_graph(create_using=cugraph.Graph(directed=False))
    k_core_results = benchmark(dcg.k_core, dg, core_number=core_number)
    k_core_results = (
        k_core_results.compute()
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
        .rename(columns={"weights": "weight"})
    )

    assert_frame_equal(
        expected_k_core_results, k_core_results, check_dtype=False, check_like=True
    )
    dataset.unload()


@pytest.mark.mg
def test_dask_mg_k_core_invalid_input(dask_client):
    dataset = DATASETS[0]
    dataset.unload()
    dg = dataset.get_dask_graph(create_using=cugraph.Graph(directed=True))

    with pytest.raises(ValueError):
        dcg.k_core(dg)

    dataset.unload()
    dg = dataset.get_dask_graph(create_using=cugraph.Graph(directed=False))

    degree_type = "invalid"
    with pytest.raises(ValueError):
        dcg.k_core(dg, degree_type=degree_type)

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

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate, dolphins, karate_asymmetric


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
DEGREE_TYPE = ["incoming", "outgoing", "bidirectional"]


# =============================================================================
# Helper Functions
# =============================================================================


def get_sg_results(dataset, degree_type):
    G = dataset.get_graph(create_using=cugraph.Graph(directed=False))
    res = cugraph.core_number(G, degree_type)
    res = res.sort_values("vertex").reset_index(drop=True)
    return res


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_sg_core_number(dask_client, dataset, degree_type, benchmark):
    # This test is only for benchmark purposes.
    sg_core_number_results = None
    G = dataset.get_graph(create_using=cugraph.Graph(directed=False))
    sg_core_number_results = benchmark(cugraph.core_number, G, degree_type)
    assert sg_core_number_results is not None


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("degree_type", DEGREE_TYPE)
def test_core_number(dask_client, dataset, degree_type, benchmark):
    dg = dataset.get_dask_graph(create_using=cugraph.Graph(directed=False))

    result_core_number = benchmark(dcg.core_number, dg, degree_type)
    result_core_number = (
        result_core_number.drop_duplicates()
        .compute()
        .sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"core_number": "mg_core_number"})
    )

    expected_output = get_sg_results(dataset, degree_type)

    # Update the mg core number with sg core number results
    # for easy comparison using cuDF DataFrame methods.
    result_core_number["sg_core_number"] = expected_output["core_number"]
    counts_diffs = result_core_number.query("mg_core_number != sg_core_number")

    assert len(counts_diffs) == 0


@pytest.mark.mg
def test_core_number_invalid_input():
    dg = karate_asymmetric.get_graph(create_using=cugraph.Graph(directed=True))
    invalid_degree_type = 3

    with pytest.raises(ValueError):
        dcg.core_number(dg, invalid_degree_type)

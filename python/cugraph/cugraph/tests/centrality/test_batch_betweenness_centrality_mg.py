# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
import numpy as np

from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.datasets import karate

from test_betweenness_centrality import (
    calc_betweenness_centrality,
    compare_scores,
)


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate]
DEFAULT_EPSILON = 0.0001
IS_DIRECTED = [False, True]
ENDPOINTS = [False, True]
IS_NORMALIZED = [False, True]
RESULT_DTYPES = [np.float64]
SUBSET_SIZES = [4, None]
SUBSET_SEEDS = [42]
IS_WEIGHTED = [False, True]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("subset_size", SUBSET_SIZES)
@pytest.mark.parametrize("normalized", IS_NORMALIZED)
@pytest.mark.parametrize("weight", [None])
@pytest.mark.parametrize("endpoints", ENDPOINTS)
@pytest.mark.parametrize("subset_seed", SUBSET_SEEDS)
@pytest.mark.parametrize("result_dtype", RESULT_DTYPES)
def test_mg_betweenness_centrality(
    dataset,
    directed,
    subset_size,
    normalized,
    weight,
    endpoints,
    subset_seed,
    result_dtype,
    dask_client,
):
    sorted_df = calc_betweenness_centrality(
        dataset,
        directed=directed,
        normalized=normalized,
        k=subset_size,
        weight=weight,
        endpoints=endpoints,
        seed=subset_seed,
        result_dtype=result_dtype,
        multi_gpu_batch=True,
    )
    compare_scores(
        sorted_df,
        first_key="cu_bc",
        second_key="ref_bc",
        epsilon=DEFAULT_EPSILON,
    )

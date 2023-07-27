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

import pytest
import numpy as np

from cugraph.dask.common.mg_utils import is_single_gpu

from cugraph.experimental.datasets import karate, netscience

# Get parameters from standard betwenness_centrality_test
# As tests directory is not a module, we need to add it to the path
# FIXME: Test must be reworked to import from 'cugraph.testing' instead of
# importing from other tests
from test_edge_betweenness_centrality import (
    DIRECTED_GRAPH_OPTIONS,
    NORMALIZED_OPTIONS,
    DEFAULT_EPSILON,
    SUBSET_SIZE_OPTIONS,
)

from test_edge_betweenness_centrality import (
    calc_edge_betweenness_centrality,
    compare_scores,
)

# =============================================================================
# Parameters
# =============================================================================
DATASETS = [karate, netscience]

# FIXME: The "preset_gpu_count" from 21.08 and below are not supported and have
# been removed
RESULT_DTYPE_OPTIONS = [np.float32, np.float64]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# FIXME: Fails for directed = False(bc score twice as much) and normalized = True.
@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file", DATASETS, ids=[f"dataset={d.get_path().stem}" for d in DATASETS]
)
@pytest.mark.parametrize("directed", DIRECTED_GRAPH_OPTIONS)
@pytest.mark.parametrize("subset_size", SUBSET_SIZE_OPTIONS)
@pytest.mark.parametrize("normalized", NORMALIZED_OPTIONS)
@pytest.mark.parametrize("result_dtype", RESULT_DTYPE_OPTIONS)
def test_mg_edge_betweenness_centrality(
    graph_file,
    directed,
    subset_size,
    normalized,
    result_dtype,
    dask_client,
):
    sorted_df = calc_edge_betweenness_centrality(
        graph_file,
        directed=directed,
        normalized=normalized,
        k=subset_size,
        weight=None,
        seed=42,
        result_dtype=result_dtype,
        multi_gpu_batch=True,
    )
    compare_scores(
        sorted_df,
        first_key="cu_bc",
        second_key="ref_bc",
        epsilon=DEFAULT_EPSILON,
    )

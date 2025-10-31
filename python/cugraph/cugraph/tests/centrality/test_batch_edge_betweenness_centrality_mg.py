# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import numpy as np
import networkx as nx

from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.datasets import karate, netscience

from test_edge_betweenness_centrality import (
    calc_edge_betweenness_centrality,
    compare_scores,
)


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, netscience]
IS_DIRECTED = [True, False]
IS_NORMALIZED = [True, False]
DEFAULT_EPSILON = 0.0001
SUBSET_SIZES = [4, None]
RESULT_DTYPES = [np.float32, np.float64]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Tests
# =============================================================================


# FIXME: Fails for directed = False(bc score twice as much) and normalized = True.
@pytest.mark.skipif(
    float(".".join(nx.__version__.split(".")[:2])) < 3.5,
    reason="Requires networkx >= 3.5",
)
@pytest.mark.mg
@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("subset_size", SUBSET_SIZES)
@pytest.mark.parametrize("normalized", IS_NORMALIZED)
@pytest.mark.parametrize("result_dtype", RESULT_DTYPES)
def test_mg_edge_betweenness_centrality(
    dataset,
    directed,
    subset_size,
    normalized,
    result_dtype,
    dask_client,
):
    sorted_df = calc_edge_betweenness_centrality(
        dataset,
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

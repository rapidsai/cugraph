# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest
import cugraph.dask as dcg

import cugraph
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


# =============================================================================
# Helper Functions
# =============================================================================


def get_pagerank_result(dataset, is_mg):
    """Return the cugraph.pagerank result for an MG or SG graph"""

    if is_mg:
        dg = dataset.get_dask_graph(store_transposed=True)
        return dcg.pagerank(dg).compute()
    else:
        g = dataset.get_graph(store_transposed=True)
        return cugraph.pagerank(g)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_pagerank(dask_client, directed):

    # Initialize and run pagerank on two distributed graphs
    # with same communicator

    input_data_path1 = karate.get_path()
    print(f"dataset1={input_data_path1}")
    result_pr1 = get_pagerank_result(karate, is_mg=True)

    input_data_path2 = dolphins.get_path()
    print(f"dataset2={input_data_path2}")
    result_pr2 = get_pagerank_result(dolphins, is_mg=True)

    # Calculate single GPU pagerank for verification of results
    expected_pr1 = get_pagerank_result(karate, is_mg=False)
    expected_pr2 = get_pagerank_result(dolphins, is_mg=False)

    # Compare and verify pagerank results

    err1 = 0
    err2 = 0
    tol = 1.0e-05

    compare_pr1 = expected_pr1.merge(
        result_pr1, on="vertex", suffixes=["_local", "_dask"]
    )

    assert len(expected_pr1) == len(result_pr1)

    for i in range(len(compare_pr1)):
        diff = abs(
            compare_pr1["pagerank_local"].iloc[i] - compare_pr1["pagerank_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err1 = err1 + 1
    print("Mismatches in ", input_data_path1, ": ", err1)

    assert len(expected_pr2) == len(result_pr2)

    compare_pr2 = expected_pr2.merge(
        result_pr2, on="vertex", suffixes=["_local", "_dask"]
    )

    for i in range(len(compare_pr2)):
        diff = abs(
            compare_pr2["pagerank_local"].iloc[i] - compare_pr2["pagerank_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err2 = err2 + 1
    print("Mismatches in ", input_data_path2, ": ", err2)
    assert err1 == err2 == 0

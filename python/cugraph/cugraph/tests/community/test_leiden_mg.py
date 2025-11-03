# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate_asymmetric, karate, dolphins
import cudf
from cudf.testing.testing import assert_series_equal


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
DATASETS_ASYMMETRIC = [karate_asymmetric]


# =============================================================================
# Helper Functions
# =============================================================================


def get_mg_graph(dataset, directed):
    """Returns an MG graph"""
    ddf = dataset.get_dask_edgelist()

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "wgt")

    return dg


# =============================================================================
# Tests
# =============================================================================
# FIXME: Implement more robust tests


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS_ASYMMETRIC)
def test_mg_leiden_with_edgevals_directed_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=True)
    # Directed graphs are not supported by Leiden and a ValueError should be
    # raised
    with pytest.raises(ValueError):
        parts, mod = dcg.leiden(dg)


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_mg_leiden_with_edgevals_undirected_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.leiden(dg)

    unique_parts = (
        parts["partition"]
        .compute()
        .drop_duplicates()
        .sort_values(ascending=True)
        .reset_index(drop=True)
    )

    idx_col = cudf.Series(unique_parts.index)

    # Ensure Leiden cluster's ID are numbered consecutively
    assert_series_equal(unique_parts, idx_col, check_dtype=False, check_names=False)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()

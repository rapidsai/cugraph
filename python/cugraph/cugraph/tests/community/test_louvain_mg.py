# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cugraph.dask as dcg
from cugraph.datasets import karate_asymmetric, karate, dolphins

from test_leiden_mg import get_mg_graph


# =============================================================================
# Parameters
# =============================================================================


DATASETS_ASYMMETRIC = DATASETS_ASYMMETRIC = [karate_asymmetric]
DATASETS = [karate, dolphins]


# =============================================================================
# Tests
# =============================================================================
# FIXME: Implement more robust tests


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS_ASYMMETRIC)
def test_mg_louvain_with_edgevals_directed_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=True)
    # Directed graphs are not supported by Louvain and a ValueError should be
    # raised
    with pytest.raises(ValueError):
        parts, mod = dcg.louvain(dg)


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_mg_louvain_with_edgevals_undirected_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.louvain(dg)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()

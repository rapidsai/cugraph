# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate_asymmetric, karate, dolphins


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

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()

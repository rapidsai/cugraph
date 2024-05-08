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

import gc

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import netscience


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [netscience]
# Directed graph is not currently supported
IS_DIRECTED = [False, True]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_wcc(dask_client, directed, dataset):

    input_data_path = dataset.get_path()
    print(f"dataset={input_data_path}")
    create_using = cugraph.Graph(directed=directed)

    g = dataset.get_graph(create_using=create_using)
    dg = dataset.get_dask_graph(create_using=create_using)

    if not directed:
        expected_dist = cugraph.weakly_connected_components(g)
        result_dist = dcg.weakly_connected_components(dg)

        result_dist = result_dist.compute()
        compare_dist = expected_dist.merge(
            result_dist, on="vertex", suffixes=["_local", "_dask"]
        )

        unique_local_labels = compare_dist["labels_local"].unique()

        for label in unique_local_labels.values.tolist():
            dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
            dask_labels = dask_labels_df["labels_dask"]
            assert (dask_labels.iloc[0] == dask_labels).all()
    else:
        with pytest.raises(ValueError):
            cugraph.weakly_connected_components(g)

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

import dask_cudf
import numpy as np
from cugraph.datasets import karate, dolphins, karate_disjoint
from cugraph.structure.replicate_edgelist import replicate_edgelist
from cudf.testing.testing import assert_frame_equal


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


edgeWeightCol = "weights"
edgeIdCol = "edge_id"
edgeTypeCol = "edge_type"
srcCol = "src"
dstCol = "dst"

DATASETS = [karate, dolphins, karate_disjoint]
IS_DISTRIBUTED = [True, False]
USE_WEIGHTS = [True, False]
USE_EDGE_IDS = [True, False]
USE_EDGE_TYPE_IDS = [True, False]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("distributed", IS_DISTRIBUTED)
@pytest.mark.parametrize("use_weights", USE_WEIGHTS)
@pytest.mark.parametrize("use_edge_ids", USE_EDGE_IDS)
@pytest.mark.parametrize("use_edge_type_ids", USE_EDGE_TYPE_IDS)
def test_mg_replicate_edgelist(
    dask_client, dataset, distributed, use_weights, use_edge_ids, use_edge_type_ids
):
    dataset.unload()
    df = dataset.get_edgelist()

    columns = [srcCol, dstCol]
    weight = None
    edge_id = None
    edge_type = None

    if use_weights:
        df = df.rename(columns={"wgt": edgeWeightCol})
        columns.append(edgeWeightCol)
        weight = edgeWeightCol
    if use_edge_ids:
        df = df.reset_index().rename(columns={"index": edgeIdCol})
        df[edgeIdCol] = df[edgeIdCol].astype(df[srcCol].dtype)
        columns.append(edgeIdCol)
        edge_id = edgeIdCol
    if use_edge_type_ids:
        df[edgeTypeCol] = np.random.randint(0, 10, size=len(df))
        df[edgeTypeCol] = df[edgeTypeCol].astype(df[srcCol].dtype)
        columns.append(edgeTypeCol)
        edge_type = edgeTypeCol

    if distributed:
        # Distribute the edges across all ranks
        num_workers = len(dask_client.scheduler_info()["workers"])
        df = dask_cudf.from_cudf(df, npartitions=num_workers)
    ddf = replicate_edgelist(
        df[columns], weight=weight, edge_id=edge_id, edge_type=edge_type
    )

    if distributed:
        df = df.compute()

    for i in range(ddf.npartitions):
        result_df = (
            ddf.get_partition(i)
            .compute()
            .sort_values([srcCol, dstCol])
            .reset_index(drop=True)
        )
        expected_df = df[columns].sort_values([srcCol, dstCol]).reset_index(drop=True)

        assert_frame_equal(expected_df, result_df, check_dtype=False, check_like=True)

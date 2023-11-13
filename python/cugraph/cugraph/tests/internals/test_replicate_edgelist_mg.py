# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import cugraph
from cugraph.testing import UNDIRECTED_DATASETS, karate_disjoint

from cugraph.structure.replicate_edgelist import replicate_edgelist
from cudf.testing.testing import assert_frame_equal
from pylibcugraph.testing.utils import gen_fixture_params_product


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


edgeWeightCol = "weights"
edgeIdCol = "edge_id"
edgeTypeCol = "edge_type"
srcCol = "src"
dstCol = "dst"


input_data = UNDIRECTED_DATASETS
input_data.append(karate_disjoint)
datasets = [pytest.param(d) for d in input_data]

fixture_params = gen_fixture_params_product(
    (datasets, "graph_file"),
    ([True, False], "distributed"),
    ([True, False], "use_weights"),
    ([True, False], "use_edge_ids"),
    ([True, False], "use_edge_type_ids"),
)


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    return dict(
        zip(
            (
                "graph_file",
                "use_weights",
                "use_edge_ids",
                "use_edge_type_ids",
                "distributed",
            ),
            request.param,
        )
    )


# =============================================================================
# Tests
# =============================================================================
# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.mg
def test_mg_replicate_edgelist(dask_client, input_combo):
    df = input_combo["graph_file"].get_edgelist()
    distributed = input_combo["distributed"]

    use_weights = input_combo["use_weights"]
    use_edge_ids = input_combo["use_edge_ids"]
    use_edge_type_ids = input_combo["use_edge_type_ids"]

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
        df[columns], weight=weight, edge_id=edge_id, edge_type=edge_type)

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

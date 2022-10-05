# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import cugraph.dask as dcg
import gc

# import pytest
import cugraph
import dask_cudf
import cudf

# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_pagerank(dask_client, directed):

    # Initialize and run pagerank on two distributed graphs
    # with same communicator

    input_data_path1 = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    print(f"dataset1={input_data_path1}")
    chunksize1 = dcg.get_chunksize(input_data_path1)

    input_data_path2 = (RAPIDS_DATASET_ROOT_DIR_PATH / "dolphins.csv").as_posix()
    print(f"dataset2={input_data_path2}")
    chunksize2 = dcg.get_chunksize(input_data_path2)

    ddf1 = dask_cudf.read_csv(
        input_data_path1,
        chunksize=chunksize1,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg1 = cugraph.Graph(directed=directed)
    dg1.from_dask_cudf_edgelist(ddf1, "src", "dst")

    result_pr1 = dcg.pagerank(dg1).compute()

    ddf2 = dask_cudf.read_csv(
        input_data_path2,
        chunksize=chunksize2,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg2 = cugraph.Graph(directed=directed)
    dg2.from_dask_cudf_edgelist(ddf2, "src", "dst")

    result_pr2 = dcg.pagerank(dg2).compute()

    # Calculate single GPU pagerank for verification of results
    df1 = cudf.read_csv(
        input_data_path1,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g1 = cugraph.Graph(directed=directed)
    g1.from_cudf_edgelist(df1, "src", "dst")
    expected_pr1 = cugraph.pagerank(g1)

    df2 = cudf.read_csv(
        input_data_path2,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g2 = cugraph.Graph(directed=directed)
    g2.from_cudf_edgelist(df2, "src", "dst")
    expected_pr2 = cugraph.pagerank(g2)

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
